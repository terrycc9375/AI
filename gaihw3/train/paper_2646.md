Stability and Generalization of Asynchronous SGD:
Sharper Bounds Beyond Lipschitz and Smoothness

Xiaoge Deng
Tao Sun∗
Shengwei Li
Dongsheng Li∗
Xicheng Lu
College of Computer Science and Technology
National University of Defense Technology, China
dengxg@nudt.edu.cn, suntao.saltfish@outlook.com, lucasleesw9@gmail.com
dsli@nudt.edu.cn, xclu@nudt.edu.cn

Abstract

Asynchronous stochastic gradient descent (ASGD) has evolved into an indispens-
able optimization algorithm for training modern large-scale distributed machine
learning tasks. Therefore, it is imperative to explore the generalization performance
of the ASGD algorithm. However, the existing results are either pessimistic and
vacuous or restricted by strict assumptions that fail to reveal the intrinsic impact
of asynchronous training on generalization. In this study, we establish sharper
stability and generalization bounds for ASGD under much weaker assumptions.
Firstly, this paper studies the on-average model stability of ASGD and provides
a non-vacuous upper bound on the generalization error, without relying on the
Lipschitz assumption. Furthermore, we investigate the excess generalization error
of the ASGD algorithm, revealing the effects of asynchronous delay, model initial-
ization, number of training samples and iterations on generalization performance.
Secondly, for the first time, this study explores the generalization performance of
ASGD in the non-smooth case. We replace smoothness with the much weaker
Hölder continuous assumption and achieve similar generalization results as in the
smooth case. Finally, we validate our theoretical findings by training numerous
machine learning models, including convex problems and non-convex tasks in
computer vision and natural language processing.

1
Introduction

The last decade has witnessed explosive growth in the scale of models and datasets in the machine
learning (ML) community [9, 12]. In light of this tendency, asynchronous distributed optimization
has become crucial to ensure efficient training of large-scale ML models [4]. Specifically, the
asynchronous stochastic gradient descent (ASGD) algorithm eliminates the synchronization barrier
between the distributed training workers, enabling each worker to independently perform idle-free
asynchronous gradient updates, thereby accelerating model training. Despite this asynchronous
updating introduces delays that result in model inconsistency, the convergence of ASGD is still
guaranteed under some mild assumptions [1, 26, 42, 40, 19, 44].

An intriguing observation is that ML models learned by stochastic gradient descent (SGD) [35] not
only achieve zero training error but also demonstrate good generalization performance on unknown
test datasets [52]. Generalizability is a classical topic in the statistical ML fields, and associated
analytical techniques include VC dimension [46], Rademacher complexity [20], PAC-Bayesian [27],
uniform convergence [29, 31], information-based, and compression-based bounds [3, 50]. In this
paper, we are going to study generalizability in the sense of algorithmic stability [8]. This stability-
based analytical framework allows bypassing the model dimensionality so that we can focus on

∗Corresponding authors

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
exploring the generalization properties of optimization algorithms. Hardt et al. [17] investigated the
generalization error of SGD on the basis of algorithmic uniform stability. Assuming that the loss
function is convex, L-Lipschitz and β-smooth, and running SGD for K iterations with a learning rate
ηk <2/β, they obtained an upper bound on the generalization error of O(L2 PK
k=1 ηk/n), where n
represents the total number of training samples. In a recent work [23], they proposed the on-average
model stability and established a tighter generalization bound of O(1/n) for low-noise settings,
without requiring the L-Lipschitz assumption.

Research on the generalization of asynchronous stochastic gradient descent algorithms mainly
concentrates on parsing the effect of asynchronous delay τ on algorithm stability and generalization.
Leveraging the algorithmic uniform stability tool, Regatti et al. [33] presented an upper generalization
error bound of O(L2Kβτ/nβτ) in the non-convex case, assuming L-Lipschitz, β-smooth functions,
and a decreasing learning rate. However, empirical experiments [13] show that this bound is too
loose to reflect the effect of asynchronous delay on algorithmic stability accurately. Deng et al. [13]
directed their attention towards convex quadratic functions and established an upper bound on the
generalization error as e
O((K−τ)/nτ) by utilizing the algorithmic average stability [36]. This bound
suggests that the introduced asynchronous delays can enhance algorithm stability, consequently
improving its generalization performance under appropriate learning rates. Unfortunately, the
analytical technique proposed in [13] is confined to quadratic optimizations.

In this study, we delve deeper into the generalization performance of the ASGD algorithm. In
particular, we utilize the on-average model stability tool to conduct a fine-grained analysis of the
stability and generalization for ASGD under much weaker assumptions. Our contributions are
summarized as follows.

• Without relying on the Lipschitz assumption, this study establishes the on-average model stability
of ASGD and provides an upper bound on the generalization error of O(1/τ +1/
√

K). In contrast
to existing work [13, 33], our results are non-vacuous and applicable to the general convex case.
• For the first time, we study the excess generalization error and provide an upper bound of
O(1/τ + ∥w1 −w∗∥2/n) for ASGD. Our findings demonstrate that appropriately increasing the
asynchronous delay, selecting a good initial model, and increasing the number of training samples
can improve the generalization performance.
• Under the much weaker (α, β)-Hölder continuous gradient assumption, we establish an excess
generalization error bound of O(1/
√

τ +∥w1−w∗∥
4α
1+α /√n1+α), which reveals similar properties
to the smooth case. To the best of our knowledge, this is the first study of the stability and
generalization of ASGD in the non-smooth case.
• We conduct comprehensive experiments using the ASGD algorithm, covering convex optimization
problems and non-convex computer vision and natural language processing tasks. Empirical
evidence confirms that appropriately increasing the asynchronous delay improves the algorithm
stability and reduces the generalization error, which is consistent with our theoretical findings.

2
Related Work

Asynchronous training, with origins dating back at least to [6, 45], has emerged as an essential
distributed method for training modern large-scale ML tasks. It effectively addresses the synchroniza-
tion bottleneck among multiple workers and mitigates the straggler problem inherent in distributed
systems [4]. This study focuses on the stochastic gradient descent algorithm with asynchronous
updates [1, 30]. Lian et al. [26] proved that ASGD has an asymptotic sublinear convergence rate in
non-convex smooth optimization, which is consistent with SGD. Arjevani et al. [2] provided tight
upper and lower complexity bounds for ASGD in convex quadratic optimization. These theoretical
results were subsequently extended to general quasi-convex and non-convex settings [40]. It is note-
worthy that the aforementioned theoretical analyses are based on bounded or fixed delay assumptions,
whereas recent studies [11, 28] explored the performance of ASGD under arbitrary delays.

A crucial aspect of asynchronous research revolves around the interaction between learning rates
and delays. For one thing, most existing theoretical analyses require the learning rate to be inversely
proportional to the asynchronous delay to guarantee the convergence of the ASGD algorithm [2, 26,
40]. For another, numerous studies opt for adaptive adjustments of the learning rate based on varying
asynchronous delays to improve the convergence rate of ASGD [34, 38, 49, 53]. The dependence of

2


---Page Break---
learning rate on asynchronous delay also influences the stability and generalization studies of ASGD
presented in this paper.

Algorithm stability originated from perturbation analysis [7], which measures the difference in
the algorithm’s output from changing a single input training sample. Generalization error refers
to the performance disparity of the output model between training and testing datasets. Hence,
algorithm stability is naturally connected to generalizability [8, 16, 36]. For the mainstream SGD
algorithm, extensive stability-based studies have been conducted for convex, non-convex, smooth and
non-smooth cases [5, 17, 22, 23, 32, 54]. Recently, algorithm stability analysis has been extended to
distributed training scenarios [48]. Considerable research has explored the generalization performance
of distributed decentralized SGD from the stability perspective [14, 43, 55].

However, the current generalization studies for ASGD remain inadequate. Building upon the
algorithmic uniform stability, Regatti et al. [33] presented a pessimistic generalization error bound
O(K ˆτ/nˆτ) of ASGD in the smooth non-convex case, where ˆτ represents the maximum delay.
In a recent development, Deng et al. [13] established a tighter upper generalization error bound
of O((K −ˆτ)/nˆτ) using average stability, and Sun et al. [41] investigated a high-probability
PAC-Bayesian generalization error bound O(1/√n) for ASGD. However, the theoretical analyses
presented in [13, 41] only hold in quadratic optimization problems, limiting their applications. To the
best of the authors’ knowledge, existing generalization analyses of ASGD are either pessimistic and
vacuous or constrained by strict assumptions. Therefore, the objective of this study is to establish
sharper stability and generalization bounds for ASGD under much milder assumptions.

3
Preliminaries

Notations. Lowercase and bold letters represent scalars and d-dimensional column vectors, re-
spectively. The ℓ2-norm of a vector x is denoted by ∥x∥. Calligraphic capital letters represent
mathematical sets. We write a = O(b) if there exists a constant 0 < c < +∞such that a ≤c · b,
and e
O(·) hides logarithmic factors. Moreover, we denote a ≍b if a = O(b) and b = O(a).

Let X ⊆Rd and Y ⊆R denote the input and output spaces, respectively. In this study, we focus
on the general supervised learning problem in ML. This task involves training a model on a data
set S = {z1, . . . , zn}, where each data point zi = (xi, yi) ∈Z = X ×Y is independently and
identically distributed (i.i.d.) sampled from an unknown distribution D. We evaluate the performance
of model w on training sample z with a loss function f(w; z). The training process can be formalized
as learning a model parameter w ∈Ω⊆Rd to minimize the empirical risk, denoted as

min
w∈ΩFS(w) = 1

n

n
X

i=1
f(w; zi).
(1)

SGD is the workhorse for solving the empirical risk minimization (ERM) problem (1), which
iteratively updates the model parameter by wk+1 = wk −ηk∇f(wk; zik).

ASGD is a powerful variant of SGD for distributed learning, which fully exploits the computational
power of distributed clusters to accelerate the training process. In the distributed parameter server
architecture [25], the distributed workers are responsible for computing gradients, while the model
updates occur on the parameter server side. Upon receiving the gradient from a worker, the server
immediately utilizes it to update the model without waiting for gradient information from other
workers. The ASGD procedure is described in Algorithm 1 (located in Appendix A.1). It is
noteworthy that although ASGD avoids synchronization overhead, it introduces delays in model
updating. To be specific, while worker m is computing and uploading the gradient, the model
parameter on the server side may has already been updated by another worker m′. In essence, the
model used for gradient computation on the worker is inconsistent with the model updated by the
server. This characteristic renders ASGD a delayed gradient update, expressed as

wk+1 = wk −ηk∇f(wk−τk; zik),
(2)

where wk, ηk, τk, and zik denote the model parameter, learning rate, asynchronous delay, and training
sample at the k-th iteration, respectively. It is worth noting that the index ik is chosen uniformly at
random from the set {1, . . . , n}.

3


---Page Break---
For the model w learned through ASGD by minimizing the empirical risk (1) on the training data set
S, people are more concerned with its performance on the unknown distribution D, i.e., the following
popular risk
F(w) = Ez∼D[f(w; z)].
(3)

The empirical risk (1) and the popular risk (3) of a model are not the same, and the difference between
them is referred to generalization error. More formally, denote the model learned by algorithm A on
data set S as A(S), and its generalization error is defined as
ϵgen := ES,A [F(A(S)) −FS(A(S))] .
(4)
The expectation here is taken over the randomness of the algorithm and the training data. This study
is dedicated to bounding ϵgen by algorithmic stability. Let

S′ = {z′
1, .., z′
n},
S(i) = {z1, .., zi−1, z′
i, zi+1, .., zn}.
(5)
S′ is also a data set i.i.d. sampled from the unknown distribution D, but is independent of the data
set S = {z1, . . . , zn}. S(i) is a perturbed data set formed by replacing the i-th sample in S with z′
i.
Based on these notations, Lei and Ying [23] defined the following on-average model stability.
Definition 1 (On-average model stability). A randomized algorithm A is on-average model ϵstab-
stable if

ES,S′,A

 1

n

n
X

i=1

A(S) −A(S(i))
2

≤ϵstab.

Leveraging the smoothness (Definition 2) assumption, the connection between this algorithmic
stability and the generalization error ϵgen is established in the following lemma [Theorem 2, [23]].
Lemma 1. Let γ >0. Assume that the function w7→f(w; z) is non-negative and β-smooth for any
z ∈Z. Then, if algorithm A is on-average model ϵstab-stable, the generalization error satisfies

ES,A [F(A(S)) −FS(A(S))] ≤β

γ ES,A[FS(A(S))] + β + γ

2
ϵstab.

While the smooth function assumption is common in optimization and generalization analyses
[13, 17, 33, 40], it does impose constraints on the applicability [5]. For instance, the hinge loss, which
is widely used in the ML fields, does not satisfy the smooth property. In this paper, therefore, we also
investigate the stability of ASGD under the much weaker Hölder continuous gradient assumption
(Definition 3), so as to establish broader and fine-grained generalization results. With the Hölder
continuous condition, stability and generalization can be connected similarly to Lemma 1.
Lemma 2 (Theorem 2, [23]). Let γ > 0. For any z ∈Z, the function w 7→f(w; z) is non-negative,
convex, and the gradient ∇f(w; z) is (α, β)-Hölder continuous. If algorithm A is on-average model
ϵstab-stable, then the generalization error satisfies

ES,A [F(A(S)) −FS(A(S))] ≤
c2
α,β
2γ ES,A[F
2α
1+α (A(S))] + γ

2 ϵstab.

Here, α ∈[0, 1], and cα,β is a constant dependent on α, β.

Furthermore, since the generalization performance of a model is primarily reflected in the popular
risk (3), this study also examines the excess generalization error, denoted as ϵex-gen, where w∗∈
argminw∈ΩF(w) and
ϵex-gen := ES,A [F(A(S)) −F(w∗)] .
(6)
Definition 2 (Smoothness). The function w 7→f(w; z) is β-smooth (β > 0) if for any z ∈Z and
w, v ∈Rd,
∥∇f(w; z) −∇f(v; z)∥≤β∥w −v∥.
Definition 3 (Hölder continuous). Let α ∈[0, 1], β > 0. The function w 7→∇f(w; z) is (α, β)-
Hölder continuous if for any z ∈Z and w, v ∈Rd,
∥∇f(w; z) −∇f(v; z)∥≤β∥w −v∥α.

It is noteworthy that the (α, β)-Hölder continuous gradient is equivalent to a β-smooth function when
α = 1. Whereas α = 0 implies that the function gradient is bounded, i.e., there exists a constant
L > 0 such that ∥∇f(w; z)∥≤L. Although many analyses of ASGD are grounded on the bounded
gradient condition [26, 28, 33], this assumption is somewhat unrealistic [24]. Notably, our analysis of
the algorithm stability and the generalization error does not rely on the bounded gradient assumption.

4


---Page Break---
4
Stability and Generalization Bounds

This section explores the stability and generalization of the ASGD algorithm in the context of smooth
loss functions, and the proof is given in Appendix B. Firstly, we present the assumption required for
this study.
Assumption 1. The parameter space Ω⊆Rd is a bounded convex set. Then, for any w, v ∈Ω,
there exists a constant r > 0 such that ∥w −v∥≤r.

Assumption 1 is standard in analyzing SGD and its variants, as it is easy to hold with the projection
operator [5, 17, 23, 30, 43]. More specifically, we consider the following projected ASGD updates
wk+1 = ΠΩ
 
wk −ηk∇f(wk−τk; zik)

.
(7)

Since the projection operator ΠΩis non-expansive, it has no impact on the stability and generalization
analysis of the ASGD algorithm.

Remark 1. Let wk and w(i)
k
denote the models produced by ASGD (7) after k iterations on the
datasets S and S(i) (defined in (5)), respectively. According to Assumption 1, it follows that
∥wk −w(i)
k ∥≤r. Notably, this result is intuitively understandable as the datasets S, S(i) differ only
by a single sample, and the initialization is the same (w1 = w(i)
1 ). In contrast to a recent work [55],
where the authors assumed a normal distribution with bounded mean and variance for the difference
between models wk and w(i)
k , our study does not necessitate such a strong assumption.

4.1
Algorithmic Stability of ASGD

The stability-based analysis of SGD hinges significantly on the non-expansiveness of the gradient
update operator [17, 23]. Namely, if function f is convex and smooth, then ∀w, v ∈Ω, z ∈Z
∥w −η∇f(w; z) −(v −η∇f(v; z))∥≤∥w −v∥.
However, this well-posed property is no longer applicable in the context of asynchronous gradient
updates. To address this issue, we present the following critical lemma to bound the delayed gradient
update operator.

Lemma 3. Let the loss function be convex, β-smooth, and Assumption 1 holds. Denote wk and w(i)
k
as the models produced by ASGD (7) with learning rates ηk ≤2/β for k iterations on the datasets S
and S(i), respectively. Then

wk−ηk∇f(wk−τk; zik)−
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2 ≤
wk−w(i)
k
2 + 2ηkβ2r2
τk
X

j=1
ηk−j.

By leveraging the properties established in Lemma 3, we can demonstrate an approximately non-
expansive recursive property for ∥wk+1 −w(i)
k+1∥2 and subsequently establish the on-average model
stability (Definition 1) of the ASGD algorithm as follows.
Theorem 1 (Stability). Suppose the loss function is non-negative, convex, and β-smooth. Let
Assumption 1 holds. If we run ASGD (7) with a non-increasing learning rate ηk ≤1/2β for k
iterations, then the on-average model stability satisfies (e is the natural constant)

ϵstab = 16βe(1 + k/n)

n

h
η1∥w1 −w∗∥2 +
 
4βr2 + 2F(w∗)

k
X

l=1
η2
l
i
+ 2β2r2e

k
X

l=1
ηl

τl
X

j=1
ηl−j.

In line with the findings of study [17], increasing the number of training iterations impairs the
stability of ASGD. Compared to SGD [23], we introduce an additional term O(Pk
l=1 ηl
Pτl
j=1 ηl−j)
to characterize the effect of asynchronous delay on the stability of ASGD. Also similar to the data-
dependent stability study [22], Theorem 1 indicates that model initialization affects the algorithmic
stability, i.e., selecting a better model initiation point w1 can effectively improve the stability.

4.2
Generalization Error Bounds

Together with Lemma 1 and Theorem 1, we can now present the generalization error (4) of the ASGD
algorithm under smooth conditions.

5


---Page Break---
Theorem 2 (Generalization error). Let Assumption 1 holds, and assume that the loss function is non-
negative, convex, and β-smooth. Running ASGD (7) with a non-increasing learning rate ηk ≤1/2β
for K iterations, then the generalization error is given by

ϵgen = O

ES,A[FS(wK)]+

K
X

k=1
ηk

τk
X

j=1
ηk−j+ 1+K/n

n

h
η1∥w1−w∗∥2+
 
1+F(w∗)
 K
X

k=1
η2
k
i
.

This finding suggests that both the model initialization and optimization processes have an impact
on the generalization performance. In practical applications, one can reduce the generalization error
by selecting a good initial model w1 to start the training task. Additionally, it is crucial to finish
the optimization process promptly since too many training iterations can detrimentally affect the
generalization performance.

Furthermore, Theorem 2 reveals a close relationship between the generalizability of ASGD and the
learning rate. As discussed in Section 2, asynchronous training typically utilizes delay-inverse corre-
lated learning rates to ensure algorithmic performance. In the low-noise case, namely, F(w∗) = 0,
Stich and Karimireddy [40] demonstrated that FS(wK) = O(1/
√

K) for ASGD under the condi-
tions of smooth and general quasi-convex loss functions, with a learning rate of ηk = c(τ
√

K)−1.
Employing this learning rate strategy, the following corollary can be derived.
Corollary 1. Let F(w∗)=0, K ≍n, and the conditions specified in Theorem 2 hold. If we set the
learning rate ηk = c(τ
√

K)−1 with a constant c > 0 and τ = PK
k=1 τk/K, then the generalization
error satisfies

ES,A [F(wK) −FS(wK)] = O
1

τ +
1
√

K


.

At this point, although the asynchronous training also introduces an additional generalization error
term of O(1/τ), increasing the delay can instead mitigate this detriment. Unlike previous ASGD
generalization research [14, 33], this study does not rely on the Lipschitz assumption. In contrast
to the vacuous upper bound of O(K ˆτ/nˆτ) in [33], we provide a sharper result and demonstrate
that increasing the asynchronous delay reduces the generalization error. While Deng et al. [13]
present a similar result O((K−ˆτ)/nˆτ) with respect to the maximum delay ˆτ in the convex quadratic
optimization, our bound holds in general convex settings. Furthermore, our results are associated
with the average delay τ rather than the pessimistic maximum delay ˆτ in [13, 14, 33].

4.3
Excess Generalization Error

According to definitions (4) and (6), the excess generalization error ϵex-gen can be decomposed as

ϵex-gen = ϵgen + ES,A [FS(A(S)) −FS(w∗)] ,
(8)

where the second term is known as the optimization error. The analysis of optimization error for
ASGD usually requires the following bounded gradient assumption [26, 28, 33].
Assumption 2. The gradient w 7→∇f(w; z) is bounded. That is, for any w ∈Ω, z ∈Z, there
exists a constant L > 0 such that ∥∇f(w; z)∥≤L.
Remark 2. Assumption 2, also known as the Lipschitz condition, is used in the optimization analysis
of ASGD to bound the model deviations induced by asynchronous delays, i.e., ∥wk −wk−τk∥≤
L Pτk
j=1 ηk−j.

For the excess generalization error of ASGD, we shift our focus to the average model wK :=
PK
k=1 ηkwk/ PK
k=1 ηk. It is noteworthy that since the parameter space Ωis a convex set, wK ∈Ω
and is frequently considered as the output of the ASGD algorithm. We first present the optimization
error with respect to this average model in the following lemma, followed by the excess generalization
error theorem of ASGD.
Lemma 4. Assuming that the loss function is non-negative, convex, and β-smooth. Let Assumptions 1
and 2 hold, if we run ASGD (7) with a non-increasing learning rate ηk ≤1/2β, then the optimization
error satisfies

ES,A[FS(wK)−FS(w∗)] = O
∥w1−w∗∥2

PK
k=1 ηk
+
 
1+F(w∗)
PK
k=1 η2
k
PK
k=1 ηk
+

PK
k=1 ηk
Pτk
j=1 ηk−j
PK
k=1 ηk


.

6


---Page Break---
Theorem 3 (Excess generalization error). Let Assumptions 1, 2 hold, and assume that the loss
function is non-negative, convex, and β-smooth. Running ASGD (7) with the non-increasing learning
rate ηk ≤1/2β for K iterations, then the excess generalization error is

ϵex-gen =O

 h
1 +
PK
k=1 η2
k
PK
k=1 ηk

i
F(w∗) + 1 + K/n

n

h
η1∥w1 −w∗∥2 +
 
1 + F(w∗)
 K
X

k=1
η2
k
i

+ ∥w1 −w∗∥2

PK
k=1 ηk
+
h K
X

k=1
ηk
 
ηk +

τk
X

j=1
ηk−j +

k
X

l=1
ηl

τl
X

j=1
ηl−j
i
/

K
X

k=1
ηk

!

.

Compared to the generalization error in Theorem 2, the excess generalization error is no longer
explicitly dependent on the optimization error FS(wK) and is more closely coupled to the learning
rate. Considering the low-noise case F(w∗) = 0, which is common in modern deep learning, the
following corollary can be further derived.

Corollary 2. Let F(w∗) = 0, K ≍n and the conditions in Theorem 3 hold. Set the learning rate
as ηk = c(τ
√

K)−1 with a constant c > 0 and τ = PK
k=1 τk/K. Then if τ ≤K
1
4 , the excess
generalization error satisfies

ES,A [F(wK) −F(w∗)] = O
1

τ + ∥w1 −w∗∥2

n


.

To the best of our knowledge, this is the first excess generalization error result for the ASGD algorithm.
Compared to Corollary 1, this generalization bound with appropriate delays is sharper and no longer
relies on the optimization error result in [40].

5
Generalization in Non-smooth Case

This section investigates the stability and generalization of the ASGD algorithm in the context of
non-smooth cases. The analysis follows a similar technical roadmap as in Section 4. Firstly, we
derive the stability of ASGD by leveraging the approximately non-expansive property of the delayed
gradient update operators. Then, the generalization error is given in conjunction with Lemma 2.
Subsequently, we analyze the optimization process of ASGD and present the excess generalization
error for the non-smooth settings.

However, without the smooth condition, the non-expansive property of asynchronous gradient
updates is further compromised, and the optimization process also introduces additional errors.
Under the much weaker Hölder continuous gradient assumption, We establish similar stability and
generalizability results for ASGD as in the smooth case, which has not been explored in existing
research. Please refer to Appendix C for the proof details of this section.

Lemma 5. Let Assumption 1 holds, and assume that the loss function is non-negative, convex, and
has a (α, β)-Hölder continuous gradient. Then, the delayed gradient update operator satisfies

wk−ηk∇f(wk−τk; zik)−(w(i)
k −ηk∇f(w(i)
k−τk; zik))
2 =∥wk−w(i)
k ∥2+O(ηk

τk
X

j=1
ηk−j +η

2
1−α
k
).

Compared to Lemma 3, an additional term O(η

2
1−α
k
) is introduced here to compensate for the
absence of smoothness. Fortunately, since the coefficient of ∥wk−w(i)
k ∥2 is not larger than 1, the
delayed gradient update of ASGD remains approximately non-expansive at an appropriate learning
rate. Leveraging this property, we are able to give the on-average model stability of ASGD in the
non-smooth case.

Theorem 4 (Stability). Suppose the loss function is non-negative, convex, and has a (α, β)-Hölder
continuous gradient. Let Assumption 1 holds. Then, the on-average model stability of ASGD satisfies

ϵstab = O
1 + k/n

n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i
+

k
X

l=1
ηl

τl
X

j=1
ηl−j +

k
X

l=1
η

2
1−α
l


.

7


---Page Break---
Theorem 4 shows that the algorithmic stability of ASGD not only depends on the learning rate,
but is also closely related to the optimization process. Similar to [23], we replace the gradient
bound (Lipschitz constant) in the uniform stability [17] with the loss function value, which leads
to sharper stability and generalizability results when combined with the subsequent optimization
analysis. Substituting this algorithm stability into Lemma 2 yields the generalization error of the
ASGD algorithm under the Hölder continuous condition (omitted in Appendix C.3).
Remark 3. Although Assumption 1 and the smooth (or Hölder continuous) condition implies
Lipschitz continuity, our point is to replace the upper gradient bound with function value, thereby
establishing sharper stability and generalization bounds that do not depend on the Lipschitz constant.

Subsequently, we present the optimization error of ASGD in the non-smooth case, and the excess
generalization error is followed by the decomposition (8).
Lemma 6. Assuming that the loss function is non-negative, convex, and has a (α, β)-Hölder con-
tinuous gradient. Let Assumptions 1 and 2 hold, then the optimization error of ASGD (7) with a
non-increasing learning rate satisfies

ES,A[FS(wK)−FS(w∗)] = O
∥w1 −w∗∥2 + PK
k=1 ηk
Pτk
j=1 ηα
k−j
PK
k=1 ηk

+

  PK
k=1 η2
k
 1−α

1+α
PK
k=1 ηk

h
η1∥w1 −w∗∥2 +
 
1 + F(w∗)
 K
X

k=1
η2
k +

K
X

k=1
η

3−α
1−α
k
i 2α

1+α 
.

Theorem 5 (Excess generalization error). Let Assumptions 1, 2 hold, and assume that the loss
function is non-negative, convex, and has a (α, β)-Hölder continuous gradient. Running ASGD (7)
with the learning rate ηk = c(τ
√

K)−1 for K ≍n iterations, then if F(w∗) = 0 and the average
delay satisfies τ ≤Kα′ with α′ = min{ 1

3,
α
3−2α}, the excess generalization error is

ES,A [F(wK) −F(w∗)] = O
 1
√

τ + ∥w1 −w∗∥
4α
1+α
√n1+α

.

Notably, the generalization performance decreases in the non-smooth case, but the underlying
properties remain consistent with the smooth setting (Corollary 2). That is, the generalization
performance can be improved by choosing a good initial model, increasing the number of training
samples, and appropriately adjusting the asynchronous delays. Additionally, when there is no
asynchronous delay in the training system, the first term in Theorem 5 vanishes, yielding an excess
generalization error bound of O(1/√n1+α). This outcome is consistent with the findings from the
study of the SGD algorithm in [23], but without requiring more computation K ≍n
2
1+α .

6
Experimental Validation

In this section, we extensively evaluated various machine learning tasks under the distributed pa-
rameter server architecture to investigate the practical stability and generalization performance of
ASGD. Our experiments included convex optimization problems as well as non-convex computer
vision (CV) and natural language processing (NLP) tasks. We simulated a distributed system with
M =16 workers and performed asynchronous training in a more general stochastic gradient descent
format as follows
wk+1 = wk −ηk
X

m∈Mk
gm
k−τk.
(9)

Here, Mk is a non-empty subset of {1, . . . , M} containing the workers that participated in asyn-
chronous training at the k-th iteration, and gm
k−τk represents the delayed gradient computed by worker
m on model wk−τk. Our experiments also focus on parsing the impact of asynchronous delays on
algorithmic stability and generalization. Following our theoretical findings, we set the learning rate
to 0.1/τ for different delays, where τ denotes the average delay.

For the convex optimization problem, we employed a single-layer linear network with the mean
squared error for a classification task on the RCV1 data set from the LIBSVM database [10]. This
data set contains 20, 242 training data with 47, 236 features per sample. In the field of computer
vision, we chose the popular ResNet18 model for image classification on the CIFAR10 and CIFAR100

8


---Page Break---
0.0
0.5
1.0
1.5
Iteration
1e4

0.0

0.4

0.8

1.2

1.6

Generalization error

1e
2

=4
=8
=16
=32

(a) Convex optimization on RCV1

0
1
2
3
4
Iteration
1e4

0

1

Generalization error

=4
=8
=16
=32

(b) CV task on CIFAR100

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

0

1

Generalization error

=4
=8
=16
=32

(c) NLP task on SST-2

Figure 1: The generalization errors of three categories of machine learning models trained using
ASGD with learning rate ηk = 0.1/τ. The horizontal axis denotes the number of asynchronous
training iterations, and the legend represents the average delay. A degradation in generalization
performance is observed as the number of training iterations increases, and the generalization
performance can be improved by appropriately increasing the asynchronous delay.

datasets. ResNet [18], a convolutional neural network with residual modules and shortcut connections,
has demonstrated remarkable performance across various CV tasks. CIFAR10 and CIFAR100 [21]
are widely used image datasets, both containing 60, 000 color images of 32 × 32 pixels. For
natural language processing tasks, we conducted experiments using BERT on the SST-2 task within
the GLUE platform [47]. BERT [15] is a pre-trained language model based on the Transformer
architecture, known for its impressive performance in handling various NLP tasks. The SST-2
[37] task in the GLUE evaluation benchmark comprises a total of 67, 350 training samples for
single-sentence categorization.

Due to computational resource limitations, this experiment cannot sequentially replace a single
sample to train n models and calculate the on-average model stability (Definition 1). Instead, we
construct a perturbed data set S(i) by randomly removing a sample from the data set S, and then
train on the two datasets separately to record the model difference ∥A(S) −A(S(i))∥2. Repeating
the process multiple times, we take the average value to approximate the algorithmic stability. As
for the generalization error (4), it is directly approximated by the absolute difference between the
training error and the testing error of the model.

Figure 1 and Figure 2 (located in Appendix D) illustrate the generalizability and stability of the
ASGD algorithm in training the three types of machine learning tasks. The experimental results show
that continuous training impairs the stability and generalization of ASGD, which is consistent with
the theorems presented in Sections 4 and 5. Conversely, when training with a learning rate that is
inversely correlated with the asynchronous delay, an appropriate increase in the delay improves the
algorithm stability and thus reduces the generalization error. This observation is in consistent with
the theoretical bound in Corollary 1, which utilizes the specific learning rate ηk = c/τ.

7
Concluding Remarks

This study establishes sharper and broader stability and generalization bounds for ASGD under much
weaker assumptions. We provide upper bounds for the on-average model stability and generalization
error of ASGD without relying on the Lipschitz continuous condition. Moreover, for the first time,
we study the stability and generalizability of ASGD in the non-smooth setting. Our generalization
results are non-vacuous and applicable to the general convex case. Furthermore, we validate our
theoretical findings with experiments on various machine learning tasks.

We also conducted experiments using delay-independent learning rates (Figures 3 and 4 in Appendix
D). Interestingly, these results also suggest that asynchronous training is beneficial for generalization.
This empirical finding challenges the pessimism of our generalization error result under constant
learning rates (omitted in Appendix B.4), and motivates further exploration of the generalizability of
ASGD. There are several directions for future research. The study of non-convex problems can focus
on showing that asynchronous updates are approximately non-expansive even without convexity, then
leading to non-vacuous stability and generalization results. Another avenue for research involves
investigating tighter high probability bounds that attenuate the dominant role of the learning rate on
generalization, thereby elucidating the experimental phenomena in Appendix D.

9


---Page Break---
Acknowledgments and Disclosure of Funding

This work is sponsored in part by the National Natural Science Foundation of China under Grant
No. 62025208, 62421002, and 62376278, Hunan Provincial Natural Science Foundation of China
(No. 2022JJ10065), Young Elite Scientists Sponsorship Program by CAST (No. 2022QNRC001),
Continuous Support of PDL (No. WDZC20235250101).

References

[1] Alekh Agarwal and John C Duchi. Distributed delayed stochastic optimization. In J. Shawe-
Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger, editors, Advances in Neural
Information Processing Systems, volume 24. Curran Associates, Inc., 2011.

[2] Yossi Arjevani, Ohad Shamir, and Nathan Srebro. A tight convergence analysis for stochastic
gradient descent with delayed updates. In Aryeh Kontorovich and Gergely Neu, editors,
Proceedings of the 31st International Conference on Algorithmic Learning Theory, volume 117,
pages 111–132. PMLR, 2020.

[3] Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for
deep nets via a compression approach. In Jennifer Dy and Andreas Krause, editors, International
Conference on Machine Learning, volume 80, pages 254–263. PMLR, 2018.

[4] Mahmoud Assran, Arda Aytekin, Hamid Reza Feyzmahdavian, Mikael Johansson, and
Michael G Rabbat. Advances in asynchronous parallel and distributed optimization. Pro-
ceedings of the IEEE, 108(11):2013–2031, 2020.

[5] Raef Bassily, Vitaly Feldman, Cristóbal Guzmán, and Kunal Talwar. Stability of stochastic
gradient descent on nonsmooth convex losses. In H. Larochelle, M. Ranzato, R. Hadsell, M.F.
Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33,
pages 4381–4391. Curran Associates, Inc., 2020.

[6] Gérard M Baudet. Asynchronous iterative methods for multiprocessors. Journal of the ACM
(JACM), 25(2):226–244, 1978.

[7] J Frédéric Bonnans and Alexander Shapiro. Optimization problems with perturbations: A
guided tour. SIAM review, 40(2):228–264, 1998.

[8] Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of Machine Learning
Research, 2:499–526, 2002.

[9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler,
Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott
Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle,
M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information
Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.

[10] Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. ACM
Transactions on Intelligent Systems and Technology, 2:27:1–27:27, 2011.

[11] Alon Cohen, Amit Daniely, Yoel Drori, Tomer Koren, and Mariano Schain. Asynchronous
stochastic optimization robust to arbitrary delays. In M. Ranzato, A. Beygelzimer, Y. Dauphin,
P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing
Systems, volume 34, pages 9024–9035. Curran Associates, Inc., 2021.

[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-
scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern
Recognition, pages 248–255. IEEE, 2009.

10


---Page Break---
[13] Xiaoge Deng, Li Shen, Shengwei Li, Tao Sun, Dongsheng Li, and Dacheng Tao. Towards
understanding the generalizability of delayed stochastic gradient descent.
arXiv preprint
arXiv:2308.09430, 2023.

[14] Xiaoge Deng, Tao Sun, Shengwei Li, and Dongsheng Li. Stability-based generalization analysis
of the asynchronous decentralized SGD. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 37, pages 7340–7348, 2023.

[15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of
deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran,
and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies,
pages 4171–4186. Association for Computational Linguistics, 2019.

[16] Andre Elisseeff, Theodoros Evgeniou, Massimiliano Pontil, and Leslie Pack Kaelbing. Stability
of randomized learning algorithms. Journal of Machine Learning Research, 6(3):55–79, 2005.

[17] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochas-
tic gradient descent. In Maria Florina Balcan and Kilian Q. Weinberger, editors, International
Conference on Machine Learning, volume 48, pages 1225–1234. PMLR, 2016.

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778,
2016.

[19] Anastasiia Koloskova, Sebastian U Stich, and Martin Jaggi. Sharper convergence guarantees for
asynchronous SGD for distributed and federated learning. In S. Koyejo, S. Mohamed, A. Agar-
wal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing
Systems, volume 35, pages 17202–17215. Curran Associates, Inc., 2022.

[20] Vladimir Koltchinskii and Dmitriy Panchenko. Rademacher processes and bounding the risk of
function learning. In High dimensional probability II, pages 443–457. Springer, 2000.

[21] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
pages 32–33, 2009.

[22] Ilja Kuzborskij and Christoph Lampert. Data-dependent stability of stochastic gradient descent.
In Jennifer Dy and Andreas Krause, editors, International Conference on Machine Learning,
volume 80, pages 2815–2824. PMLR, 2018.

[23] Yunwen Lei and Yiming Ying. Fine-grained analysis of stability and generalization for stochastic
gradient descent. In Hal Daumé III and Aarti Singh, editors, International Conference on
Machine Learning, volume 119, pages 5809–5819. PMLR, 2020.

[24] Yunwen Lei, Ting Hu, Guiying Li, and Ke Tang. Stochastic gradient descent for nonconvex
learning without bounded gradient assumptions. IEEE Transactions on Neural Networks and
Learning Systems, 31(10):4394–4400, 2019.

[25] Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski,
James Long, Eugene J Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the
parameter server. In 11th USENIX Symposium on Operating Systems Design and Implementation
(OSDI 14), pages 583–598. USENIX Association, 2014.

[26] Xiangru Lian, Yijun Huang, Yuncheng Li, and Ji Liu. Asynchronous parallel stochastic gradient
for nonconvex optimization. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett,
editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates,
Inc., 2015.

[27] David A McAllester. PAC-Bayesian model averaging. In Proceedings of the twelfth annual
conference on Computational learning theory, pages 164–170. Association for Computing
Machinery, 1999.

11


---Page Break---
[28] Konstantin Mishchenko, Francis Bach, Mathieu Even, and Blake E Woodworth. Asynchronous
SGD beats minibatch SGD under arbitrary delays. In S. Koyejo, S. Mohamed, A. Agarwal,
D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems,
volume 35, pages 420–433. Curran Associates, Inc., 2022.

[29] Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain
generalization in deep learning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc,
E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32.
Curran Associates, Inc., 2019.

[30] Angelia Nedi´c, Dimitri P Bertsekas, and Vivek S Borkar. Distributed asynchronous incremental
subgradient methods. Studies in Computational Mathematics, 8(C):381–407, 2001.

[31] Jeffrey Negrea, Gintare Karolina Dziugaite, and Daniel Roy. In defense of uniform conver-
gence: Generalization via derandomization with an application to interpolating predictors. In
Hal Daumé III and Aarti Singh, editors, International Conference on Machine Learning, volume
119, pages 7263–7272. PMLR, 2020.

[32] Anant Raj, Lingjiong Zhu, Mert Gurbuzbalaban, and Umut Simsekli. Algorithmic stability of
heavy-tailed SGD with general loss functions. In Andreas Krause, Emma Brunskill, Kyunghyun
Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference
on Machine Learning, volume 202, pages 28578–28597. PMLR, 2023.

[33] Jayanth Regatti, Gaurav Tendolkar, Yi Zhou, Abhishek Gupta, and Yingbin Liang. Distributed
SGD generalizes well under asynchrony. In 2019 57th Annual Allerton Conference on Commu-
nication, Control, and Computing (Allerton), pages 863–870. IEEE, 2019.

[34] Zhaolin Ren, Zhengyuan Zhou, Linhai Qiu, Ajay Deshpande, and Jayant Kalagnanam. Delay-
adaptive distributed stochastic optimization. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 34, pages 5503–5510, 2020.

[35] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of
Mathematical Statistics, 22(3):400–407, 1951.

[36] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability,
stability and uniform convergence. Journal of Machine Learning Research, 11(90):2635–2670,
2010.

[37] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew
Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a
sentiment treebank. In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and
Steven Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural
Language Processing, pages 1631–1642. Association for Computational Linguistics, 2013.

[38] Suvrit Sra, Adams Wei Yu, Mu Li, and Alex Smola. Adadelay: Delay adaptive distributed
stochastic optimization. In Arthur Gretton and Christian C. Robert, editors, Artificial Intelligence
and Statistics, volume 51, pages 957–965. PMLR, 2016.

[39] Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. In
J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, editors, Advances in neural
information processing systems, volume 23. Curran Associates, Inc., 2010.

[40] Sebastian U Stich and Sai Praneeth Karimireddy. The error-feedback framework: Better rates for
SGD with delayed gradients and compressed updates. Journal of Machine Learning Research,
21(237):1–36, 2020.

[41] Jianhui Sun, Ying Yang, Guangxu Xun, and Aidong Zhang. Scheduling hyperparameters to
improve generalization: From centralized SGD to asynchronous SGD. ACM Transactions on
Knowledge Discovery from Data, 17(2):1–37, 2023.

[42] Tao Sun, Robert Hannah, and Wotao Yin. Asynchronous coordinate descent under more realistic
assumptions. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan,
and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran
Associates, Inc., 2017.

12


---Page Break---
[43] Tao Sun, Dongsheng Li, and Bao Wang. Stability and generalization of decentralized stochastic
gradient descent. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pages 9756–9764, 2021.

[44] Tao Sun, Baihao Wu, Dongsheng Li, and Kun Yuan. Gradient normalization makes heteroge-
neous distributed learning as easy as homogenous distributed learning. arXiv preprint, 2024.

[45] John Tsitsiklis, Dimitri Bertsekas, and Michael Athans. Distributed asynchronous deterministic
and stochastic gradient optimization algorithms. IEEE Transactions on Automatic Control, 31
(9):803–812, 1986.

[46] VN Vapnik and A Ya Chervonenkis. On the uniform convergence of relative frequencies of
events to their probabilities. Theory of Probability & its Applications, 16(2):264–280, 1971.

[47] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.
GLUE: A multi-task benchmark and analysis platform for natural language understanding. In
International Conference on Learning Representations, 2019.

[48] Xinxing Wu, Junping Zhang, and Fei-Yue Wang. Stability-based generalization analysis of
distributed learning algorithms for big data. IEEE Transactions on Neural Networks and
Learning Systems, 31(3):801–812, 2019.

[49] Xuyang Wu, Sindri Magnusson, Hamid Reza Feyzmahdavian, and Mikael Johansson. Delay-
adaptive step-sizes for asynchronous learning. In Kamalika Chaudhuri, Stefanie Jegelka,
Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, International Conference on
Machine Learning, volume 162, pages 24093–24113. PMLR, 2022.

[50] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability
of learning algorithms. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems,
volume 30. Curran Associates, Inc., 2017.

[51] Yiming Ying and Ding-Xuan Zhou. Unregularized online learning algorithms with general loss
functions. Applied and Computational Harmonic Analysis, 42(2):224–244, 2017.

[52] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. In International Conference on Learning
Representations, 2017.

[53] Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-SGD for distributed
deep learning. In Subbarao Kambhampati, editor, Proceedings of the Twenty-Fifth International
Joint Conference on Artificial Intelligence, pages 2350–2356. IJCAI/AAAI Press, 2016.

[54] Yikai Zhang, Wenjia Zhang, Sammy Bald, Vamsi Pingali, Chao Chen, and Mayank Goswami.
Stability of SGD: Tightness analysis and improved bounds. In James Cussens and Kun Zhang,
editors, Uncertainty in Artificial Intelligence, volume 180, pages 2364–2373. PMLR, 2022.

[55] Tongtian Zhu, Fengxiang He, Lan Zhang, Zhengyang Niu, Mingli Song, and Dacheng Tao.
Topology-aware generalization of decentralized SGD. In Kamalika Chaudhuri, Stefanie Jegelka,
Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, International Conference on
Machine Learning, volume 162, pages 27479–27503. PMLR, 2022.

[56] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In
Proceedings of the Twentieth International Conference on International Conference on Machine
Learning, pages 928–936. AAAI Press, 2003.

13


---Page Break---
Appendix for

Stability and Generalization of Asynchronous SGD: Sharper
Bounds Beyond Lipschitz and Smoothness

A
Background Knowledge

A.1
ASGD Process

In the distributed parameter server architecture, the distributed workers are responsible for computing
gradients, while the model updates occur on the parameter server side. Upon receiving the gradient
from a worker, the server immediately utilizes it to update the model without waiting for gradient
information from other workers. The ASGD procedure is described in Algorithm 1.

Algorithm 1 Asynchronous SGD
Initialization: model parameter w
Input: learning rate η
// Worker m

1: repeat
2:
pull the current model w from the server
3:
compute gradient gm =∇f(w; z) with local data z
4:
push gm to the server
5: until terminated
// Server

6: if server received gradient from any worker m then
7:
update the model as w ←w −ηgm

8:
send w back to worker m
9: end if
Output: model w

It is noteworthy that although ASGD avoids synchronization overhead, it introduces delays in model
updating. To be specific, while worker m is computing and uploading the gradient, the model
parameter on the server side may has already been updated by another worker m′. In essence, the
model used for gradient computation on the worker (w in line 3 of Algorithm 1) is inconsistent with
the model updated by the server (w in line 7 of Algorithm 1). This characteristic renders ASGD a
delayed gradient update, expressed as

wk+1 = wk −ηk∇f(wk−τk; zik),

where wk, ηk, τk, and zik denote the model parameter, learning rate, asynchronous delay, and training
sample at the k-th iteration, respectively. It is worth noting that the index ik is chosen uniformly at
random from the set {1, . . . , n}.

A.2
Useful Inequalities

Our analysis frequently uses the following inequalities.
Lemma A.1 (Young’s inequality). If p > 1 and q > 1 are real numbers such that 1

p + 1

q = 1, then
for any a, b ∈R+,

ab ≤1

pap + 1

q bq.
(A.1)

14


---Page Break---
Lemma A.2 (Cauchy–Schwarz inequality). For any w, v ∈Rd, the following inequality holds.

⟨w, v⟩≤∥w∥· ∥v∥.
(A.2)

Lemma A.3. Let p > 0. For any a, b ∈R+, the following inequalities hold.

2ab ≤pa2 + 1

pb2,
(A.3)

(a + b)2 ≤(1 + p)a2 + (1 + 1

p)b2.
(A.4)

In addition, we rely on the self-bounding properties of the smooth and Hölder continuous gradient
functions. The proof can be found in [24, 39, 51].
Lemma A.4. If the function w 7→f(w; z) is non-negative and the gradient ∇f is (α, β)-Hölder
continuous (Definition 3) with α ∈[0, 1], β > 0. Then for any w, z, we have

∥∇f(w; z)∥≤cα,βf
α
1+α (w; z),
(A.5)

where the constant cα,β is defied as

cα,β :=

(
(1 + 1/α)
α
1+α β
1
1+α ,
if
α > 0
supz ∥∇f(0; z)∥+ β,
if
α = 0
(A.6)

Remark A.1. The case α = 1 implies that the function w 7→f(w; z) is β-smooth (Definition 2). At
this point, the constant cα,β = √2β, and the gradient satisfies

∥∇f(w; z)∥2 ≤2βf(w; z).
(A.7)

Lemma A.5. The projection operator is defined as ΠΩ(v) = argminw∈Ω∥w−v∥, and this operator
is non-expansive, i.e.,

∥ΠΩ(w) −ΠΩ(v)∥≤∥w −v∥, ∀w, v ∈Rd
and
∥ΠΩ(v) −w∥≤∥v −w∥, ∀v ∈Rd, w ∈Ω.
(A.8)

The proof of Lemma A.5 can be found in [56]. This non-expansive property not only ensures the
plausibility of Assumption 1, but also facilitates the stability and generalization analysis of the
projected ASGD algorithm (7), making it no inherently different from the standard ASGD (2).

A.3
Proof of Lemma 1 and Lemma 2 (Section 3 in the main text)

Lemma 1 and Lemma 2 were established by Lei et al. [Theorem 2, [23]]. The following proof is
derived from [Appendix B, [23]]. Recall the following definitions

S = {z1, . . . , zi−1,zi, zi+1, . . . , zn},
S′ = {z′
1, . . . , z′
i−1, z′
i, z′
i+1, . . . , z′
n},

S(i) = {z1, . . . , zi−1, z′
i, zi+1, . . . , zn},

and

F(w) = Ez∼D[f(w; z)],
FS(w) = 1

n

n
X

i=1
f(w; zi),
ϵgen := ES,A [F(A(S)) −FS(A(S))] .

(A.9)
Since for any i, the data samples zi and z′
i are both drawn i.i.d. from D, then A(S(i)) is independent
of zi and we have the following fact

ES[F(A(S))] = ES,S′[f(A(S(i)); zi)] = 1

n

n
X

i=1
ES,S′[f(A(S(i)); zi)].

Hence, the generalization error satisfies

ϵgen = ES,A [F(A(S)) −FS(A(S))] = ES,A


ES′[f(A(S(i)); zi)] −1

n

n
X

i=1
f(A(S); zi)


= 1

n

n
X

i=1
ES,S′,A
h
f(A(S(i)); zi) −f(A(S); zi)
i
.

(A.10)

15


---Page Break---
By incorporating the β-smoothness property of the function w 7→f(w; z), we have

ES,A [F(A(S))−FS(A(S))] ≤1

n

n
X

i=1
ES,S′,A

D
A(S(i))−A(S), ∇f(A(S); zi)
E
+β

2 ∥A(S(i))−A(S)∥2

.

(A.11)
Using the inequalities (A.2), (A.3) and self-bounding property (A.7), let γ > 0 then we know that
D
A(S(i)) −A(S), ∇f(A(S); zi)
E
≤∥A(S(i)) −A(S)∥∥∇f(A(S); zi)∥

≤γ

2 ∥A(S(i)) −A(S)∥2 + 1

2γ ∥∇f(A(S); zi)∥2

≤γ

2 ∥A(S(i)) −A(S)∥2 + β

γ f(A(S); zi).

Substituting back into inequality (A.11) yields

ES,A [F(A(S))−FS(A(S))] ≤β

γ ES,A

 1

n

n
X

i=1
f(A(S); zi)

+ β+γ

2
ES,S′,A

 1

n

n
X

i=1
∥A(S)−A(S(i))∥2

.

By further combining with (A.9) and Definition 1, Lemma 1 is thus derived, i.e.,

ES,A [F(A(S)) −FS(A(S))] ≤β

γ ES,A[FS(A(S))] + β + γ

2
ϵstab.
(A.12)

Without the smoothness assumption, we then need to utilize the convexity property of the function
w 7→f(w; z), i.e.,

f(A(S(i)); zi) −f(A(S); zi) ≤
D
A(S(i)) −A(S), ∇f(A(S(i)); zi)
E

≤γ

2 ∥A(S(i)) −A(S)∥2 + 1

2γ ∥∇f(A(S(i)); zi)∥2

≤γ

2 ∥A(S(i)) −A(S)∥2 +
c2
α,β
2γ f
2α
1+α (A(S(i)); zi),

where the last two inequalities use (A.2), (A.3) and the self-bounding property (A.5). Substituting it
back into inequality (A.10) leads to Lemma 2, i.e.,

ES,A [F(A(S)) −FS(A(S))]

≤
c2
α,β
2γ
1
n

n
X

i=1
ES,S′,A
h
f
2α
1+α (A(S(i)); zi)
i
+ γ

2 ES,S′,A

 1

n

n
X

i=1
∥A(S) −A(S(i))∥2


≤
c2
α,β
2γ ES,A[F
2α
1+α (A(S))] + γ

2 ϵstab.

(A.13)

Here we use the concavity of the map x 7→x
2α
1+α and the following fact

ES,S′,A
h
f
2α
1+α (A(S(i)); zi)
i
≤ES,S′,A
h
Ezi

f(A(S(i)); zi)
 2α

1+α i

= ES,S′,A
h
F
2α
1+α (A(S(i)))
i
= ES,A
h
F
2α
1+α (A(S))
i
.

B
Proof of Stability and Generalization Bounds (Section 4 in the main text)

B.1
Proof of Lemma 3 (approximately non-expansive property of delayed gradient updates)

Due to that the function w 7→f(w; z) is convex and β-smooth, the gradient ∇f is co-coercive,
namely

wk−τk−w(i)
k−τk, ∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)

≥1

β ∥∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)∥2.

16


---Page Break---
Using this co-coercivity with learning rate ηk ≤2/β, we have
wk −ηk∇f(wk−τk; zik) −
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2

= ∥wk −w(i)
k ∥2 + η2
k∥∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)∥2

−2ηk⟨wk −w(i)
k , ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

= ∥wk −w(i)
k ∥2 + η2
k∥∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)∥2

−2ηk⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

−2ηk⟨wk −wk−τk −(w(i)
k −w(i)
k−τk), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

≤∥wk −w(i)
k ∥2 −2ηk⟨wk −wk−τk −(w(i)
k −w(i)
k−τk), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩.

From the iterative scheme of ASGD (7), we know that

⟨wk −wk−τk −(w(i)
k −w(i)
k−τk), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

=

τk
X

j=1
⟨wk−j+1 −wk−j −(w(i)
k−j+1 −w(i)
k−j), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

≤

τk
X

j=1
ηk−j∥∇f(wk−j−τk−j; zik−j)−∇f(w(i)
k−j−τk−j; zik−j)∥∥∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)∥,

(B.1)

where the last inequality is due to (A.2) and (A.8). Following the β-smooth property and Assumption
1, we can derive

∥∇f(ws−τs; zis)−∇f(w(i)
s−τs; zis)∥≤β∥ws−τs −w(i)
s−τs∥≤βr, for s = k, k −j; j = 1, . . . , τk.
(B.2)

With inequalities (B.1) and (B.2), we are arrive at

wk−ηk∇f(wk−τk; zik)−
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2 ≤
wk−w(i)
k
2+2ηkβ2r2
τk
X

j=1
ηk−j.

(B.3)

B.2
Proof of Theorem 1 (algorithm stability under the smooth assumption)

Let wk and w(i)
k denote the models produced by ASGD (7) after k iterations on the datasets S and
S(i), respectively. Given that the index ik at the k-th iteration is chosen randomly from the set
{1, 2, . . . , n}, there is a probability of 1−1/n that ik ̸= i. Then, by the approximately non-expansive
property (B.3) of the ASGD iteration and (A.8), we have

∥wk+1 −w(i)
k+1∥2 ≤
wk −ηk∇f(wk−τk; zik) −
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2

≤∥wk −w(i)
k ∥2 + 2ηkβ2r2
τk
X

j=1
ηk−j.

On the other hand, there is a probability of 1/n such that the algorithm accurately selects the i-th
sample point (ik = i) that is different in the two datasets S and S(i). In this case, we can perform
the following analysis based on the inequality (A.4) with p > 0, self-bounding property (A.7) and
non-expansive projection (A.8)

∥wk+1 −w(i)
k+1∥2 ≤∥wk −ηk∇f(wk−τk; zi) −w(i)
k + ηk∇f(w(i)
k−τk; z′
i)∥2

≤(1 + p)∥wk −w(i)
k ∥2 + (1 + 1/p)η2
k∥∇f(wk−τk; zi) −∇f(w(i)
k−τk; z′
i)∥2

≤(1 + p)∥wk −w(i)
k ∥2 + 2(1 + 1/p)η2
k
h
∥∇f(wk−τk; zi)∥2 + ∥∇f(w(i)
k−τk; z′
i)∥2i

≤(1 + p)∥wk −w(i)
k ∥2 + 4β(1 + 1/p)η2
k
h
f(wk−τk; zi) + f(w(i)
k−τk; z′
i)
i
.

17


---Page Break---
Combining the two cases above and taking the expectation with respect to the randomness of the
algorithm yields

EA∥wk+1 −w(i)
k+1∥2 ≤(1 + p

n)EA∥wk −w(i)
k ∥2 + 2ηkβ2r2(n −1)

n

τk
X

j=1
ηk−j

+ 4β(1 + 1/p)η2
k
n
EA
h
f(wk−τk; zi) + f(w(i)
k−τk; z′
i)
i
.

Since zi and z′
i are i.i.d. sampled from the same distribution D, we have the following fact

ES,S′,A[f(w(i)
k−τk; z′
i)] = ES,A[f(wk−τk; zi)].

A subsequent expectation over the randomness of data produces

ES,S′,A∥wk+1 −w(i)
k+1∥2

≤(1 + p

n)ES,S′,A∥wk −w(i)
k ∥2 + 8β(1 + 1/p)η2
k
n
ES,A [f(wk−τk; zi)] + 2ηkβ2r2
τk
X

j=1
ηk−j

≤

k
X

l=1
(1 + p

n)(k−l)



8β(1 + 1/p)η2
l
n
ES,A [f(wl−τl; zi)] + 2ηlβ2r2
τl
X

j=1
ηl−j



,

where the second inequality is due to the same initialization w1 = w(i)
1 . Let p = n/k, then
(1 + p/n)(k−1) ≤e (where e is the natural constant), and the on-average model stability of ASGD
satisfies

ES,S′,A

 1

n

n
X

i=1
∥wk+1 −w(i)
k+1∥2


≤

k
X

l=1
(1 + p

n)(k−l)
8β(1 + 1/p)η2
l
n
ES,A [FS(wl−τl)] + 2ηlβ2r2
τl
X

j=1
ηl−j



≤(1 + p

n)(k−1)
8β(1 + 1/p)

n

k
X

l=1
η2
l ES,AFS(wl−τl) + 2β2r2
k
X

l=1
ηl

τl
X

j=1
ηl−j



≤8βe(1 + k/n)

n

k
X

l=1
η2
l ES,A [FS(wl−τl)] + 2β2r2e

k
X

l=1
ηl

τl
X

j=1
ηl−j.

(B.4)

For the further investigation of the algorithm stability of ASGD, it is imperative to bound error
Pk
l=1 η2
l ES,A[FS(wl−τl)] of the delayed model. With the ASGD update (7), inequality (A.8),
convexity and smooth property (A.7), we have the following derivation

∥wk+1 −w∗∥2 ≤∥wk −ηk∇f(wk−τk; zik) −w∗∥2

= ∥wk −w∗∥2 + η2
k∥∇f(wk−τk; zik)∥2 + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 + 2βη2
kf(wk−τk; zik) + 2ηk⟨w∗−wk−τk, ∇f(wk−τk; zik)⟩
+ 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 + 2βη2
kf(wk−τk; zik) + 2ηk(f(w∗; zik) −f(wk−τk; zik))
+ 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 −ηkf(wk−τk; zik) + 2ηkf(w∗; zik) + 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩,

where w∗∈argminw∈ΩF(w), and the last inequality is due to ηk ≤1/2β. Then

ηkf(wk−τk; zik) ≤∥wk −w∗∥2 −∥wk+1 −w∗∥2 + 2ηkf(w∗; zik)
+ 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩.
(B.5)

18


---Page Break---
Following the inequality (A.2), self-bounding property (A.7), and Assumption 1, we know that

2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩≤2rηk∥∇f(wk−τk; zik)∥≤2rηk
p

2βf(wk−τk; zik)

≤2r
p

2βηk ·
p

ηkf(wk−τk; zik) ≤4βr2ηk + ηk

2 f(wk−τk; zik),

where last inequality uses (A.3) with p = 1. Turning to (B.5), we have

ηkf(wk−τk; zik) ≤2∥wk −w∗∥2 −2∥wk+1 −w∗∥2 + 4ηkf(w∗; zik) + 8βr2ηk.

Multiplying both sides with the non-increasing learning rate, we get

η2
kf(wk−τk; zik) ≤2ηk∥wk −w∗∥2 −2ηk+1∥wk+1 −w∗∥2 + 4η2
kf(w∗; zik) + 8βr2η2
k.

Taking an expectation on both sides followed by a summation leads to

k
X

l=1
η2
l ES,A[FS(wl−τl)] ≤2η1∥w1 −w∗∥2 + 4

k
X

l=1
η2
l F(w∗) + 8βr2
k
X

l=1
η2
l ,
(B.6)

where we use ES[f(w∗; zik)] = ES[FS(w∗)] = F(w∗). Substituting (B.6) into (B.4), we are arrive
at

ES,S′,A

 1

n

n
X

i=1

wk+1 −w(i)
k+1
2


≤16βe(1 + k/n)

n


η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + 4βr2
k
X

l=1
η2
l


+ 2β2r2e

k
X

l=1
ηl

τl
X

j=1
ηl−j.

(B.7)

B.3
Proof of Theorem 2 (generalization error under the smooth assumption)

Together with Lemma 1 (A.12) and Theorem 1 (B.7), we have

ES,A [F(wk+1) −FS(wk+1)] ≤β

γ ES,A[FS(wk+1)] + β + γ

2
ES,S′,A

"
1
n

n
X

i=1
∥wk+1 −w(i)
k+1∥2
#

≤β

γ ES,A[FS(wk+1)] + 8βe(β + γ)(1 + k/n)

n

"

η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + 4βr2
k
X

l=1
η2
l

#

+ β2r2(β + γ)e

k
X

l=1
ηl

τl
X

j=1
ηl−j.

(B.8)

Let γ = 1, then the generalization error ϵgen of ASGD satisfies

ES,A [F(wK) −FS(wK)]

= O

ES,A[FS(wK)] +

K
X

k=1
ηk

τk
X

j=1
ηk−j + 1+K/n

n

h
η1∥w1 −w∗∥2 +
 
1 + F(w∗)
 K
X

k=1
η2
k
i
.

(B.9)

B.4
Proof of Corollary 1 (generalization error with specific learning rates)

Let K ≍n. Following Theorem 2 (B.9), if we use the delay-independent constant learning rate
ηk ≡η, the generalization error ϵgen satisfies

ES,A [F(wK) −FS(wK)] = O
 K
X

k=1
τk + ∥w1 −w∗∥2

n
+ ES,A

FS(wK) + F(w∗)

. (B.10)

19


---Page Break---
In comparison to SGD [23], asynchronous training with a fixed learning rate η introduces an additional
error O(PK
k=1 τk) due to delay. This error accumulates as iterations increase, which subsequently
deteriorates the generalization performance of ASGD. However, the experimental results in Appendix
D demonstrate that the generalization error bound in (B.10) is pessimistic.

In the low-noise case, i.e., F(w∗)=0, Stich and Karimireddy [40] proved that FS(wK)=O(1/
√

K)
under the smooth and general quasi-convex loss function conditions with the learning rate ηk =
c(τ
√

K)−1 (c>0 is a constant, τ = PK
k=1 τk/K). At this point, the generalization error of ASGD is

ES,A [F(wK) −FS(wK)] = O
 1
√

K


+ O
1 + K/n

n

hc∥w1 −w∗∥2

τ
√

K
+ c2

τ 2
i
+
c2

Kτ 2

K
X

k=1
τk



= O
1

τ +
1
√

K


.

Remark B.1. Under the assumptions of β-smoothness and (M, σ2)-bounded noise, Stich and
Karimireddy [40] proved that FS(wK) = O(1/
√

K) when the learning rate is chosen as ηk ≤
1
10β(τ+M). In the above proof, we can flexibly adjust the constant c to make the learning rate satisfy
the requirements of the study [40], enabling the safe utilization of its optimization results.
Remark B.2. The notation O hides the numerical values and specific fixed constants, such as c, e, β,
and r. This notation facilitates the reader’s comprehension by allowing for an intuitive understanding
of the effects of important variables such as asynchronous delay ¯τ, the number of iterations K, and
the amount of training data n on stability and generalization.

B.5
Proof of Lemma 4 (optimization error under the smooth assumption)

By the ASGD update (7), convexity, smooth property (A.7) and the non-expansive projection (A.8),
we can derive
∥wk+1 −w∗∥2 ≤∥wk −ηk∇f(wk−τk; zik) −w∗∥2

= ∥wk −w∗∥2 + η2
k∥∇f(wk−τk; zik)∥2 + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 + 2βη2
kf(wk−τk; zik) + 2ηk⟨w∗−wk, ∇f(wk; zik)⟩
+ 2ηk⟨w∗−wk, ∇f(wk−τk; zik) −∇f(wk; zik)⟩

≤∥wk −w∗∥2 + 2βη2
kf(wk−τk; zik) + 2ηk(f(w∗; zik) −f(wk; zik))
+ 2βηk∥wk −w∗∥· ∥wk −wk−τk∥.

(B.11)

From the iterative scheme of ASGD (7) and (A.8), we know that

∥wk −wk−τk∥≤

τk
X

j=1
∥wk−j+1 −wk−j∥≤

τk
X

j=1
ηk−j∥∇f(wk−j−τk−j; zik−j)∥.
(B.12)

Then taking an expectation on both sides of (B.11) and combing with Assumptions 1 and 2, we have
2ηkES,A[FS(wk) −FS(w∗)] ≤ES,A∥wk −w∗∥2 −ES,A∥wk+1 −w∗∥2 + 2βη2
kES,A[FS(wk−τk)]

+ 2βLrηk

τk
X

j=1
ηk−j.

Subsequently a summation of the inequality produces

2

K
X

k=1
ηkES,A[FS(wk)−FS(w∗)] ≤∥w1−w∗∥2+2β

K
X

k=1
η2
kES,A[FS(wk−τk)]+2βLr

K
X

k=1
ηk

τk
X

j=1
ηk−j.

Leveraging the optimization bound (B.6) with ηk ≤1/2β, we can derive

K
X

k=1
ηkES,A[FS(wk)−FS(w∗)] ≤1

2∥w1−w∗∥2+β

K
X

k=1
η2
kES,A[FS(wk−τk)]+βLr

K
X

k=1
ηk

τk
X

j=1
ηk−j

≤(1

2 + 2βη1)∥w1 −w∗∥2 + 4β

K
X

k=1
η2
kF(w∗) + 8β2r2
K
X

k=1
η2
k + βLr

K
X

k=1
ηk

τk
X

j=1
ηk−j.

(B.13)

20


---Page Break---
Let the average model

wK :=
PK
k=1 ηkwk
PK
k=1 ηk
∈Ω.

Following the convexity of the function FS, we know that

ES,A[FS(wK) −FS(w∗)] ≤
PK
k=1 ηkES,A[FS(wk) −FS(w∗)]

PK
k=1 ηk

≤(1

2 + 2βη1)∥w1 −w∗∥2

PK
k=1 ηk
+ 4β
 
F(w∗) + 2βr2PK
k=1 η2
k
PK
k=1 ηk
+ βLr

PK
k=1 ηk
Pτk
j=1 ηk−j
PK
k=1 ηk
.

B.6
Proof of Theorem 3 (excess generalization error under the smooth assumption)

Multiplying both sides of the generalization error (B.8) by ηk+1 followed with a summation gives

K
X

k=1
ηkES,A [F(wk)] ≤(1 + β

γ )

K
X

k=1
ηkES,A[FS(wk)] + β2r2(β + γ)e

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j

+ 8βe(β + γ)

n

K
X

k=1
ηk(1 + k

n)

"

η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + 4βr2
k
X

l=1
η2
l

#

.

(B.14)

Substituting the optimization error (B.13) into the above inequality (B.14), we have

K
X

k=1
ηkES,A [F(wk) −F(w∗)] ≤β2r2(β + γ)e

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j + β

γ

K
X

k=1
ηkF(w∗)

+ (1 + β

γ )



(1

2 + 2βη1)∥w1 −w∗∥2 + 4β

K
X

k=1
η2
kF(w∗) + 8β2r2
K
X

k=1
η2
k + βLr

K
X

k=1
ηk

τk
X

j=1
ηk−j





+ 8βe(β + γ)

n

K
X

k=1
ηk(1 + k

n)

"

η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + 4βr2
k
X

l=1
η2
l

#

.

Utilizing the convexity property of the function F, we can derive

ES,A [F(wK) −F(w∗)] ≤
PK
k=1 ηkES,A [F(wk) −F(w∗)]

PK
k=1 ηk

≤1 + β/γ
PK
k=1 ηk



(1

2 + 2βη1)∥w1 −w∗∥2 + 4β

K
X

k=1
η2
kF(w∗) + 8β2r2
K
X

k=1
η2
k + βLr

K
X

k=1
ηk

τk
X

j=1
ηk−j





+ 8βe(β + γ)(1 + K/n)

n

"

η1∥w1 −w∗∥2 + 2

K
X

k=1
η2
kF(w∗) + 4βr2
K
X

k=1
η2
k

#

+ β2r2(β + γ)e

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j/

K
X

k=1
ηk + β

γ F(w∗).

(B.15)

By setting γ = 1, we conclude that the excess generalization error ϵex-gen is

ϵex-gen = O

 h
1 +
PK
k=1 η2
k
PK
k=1 ηk

i
F(w∗) + 1 + K/n

n

h
η1∥w1 −w∗∥2 +
 
1 + F(w∗)
 K
X

k=1
η2
k
i

+ ∥w1 −w∗∥2

PK
k=1 ηk
+
h K
X

k=1
η2
k +

K
X

k=1
ηk

τk
X

j=1
ηk−j +

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j
i
/

K
X

k=1
ηk

!

.

21


---Page Break---
B.7
Proof of Corollary 2 (excess generalization error with a specific learning rate)

Let the learning rate ηk = c(τ
√

K)−1 with a constant c>0, and τ =PK
k=1 τk/K, direct calculation
gives

K
X

k=1
ηk = c
√

K/τ,

K
X

k=1
η2
k = c2/τ 2,

K
X

k=1
ηk

τk
X

j=1
ηk−j =
c2

τ 2K

K
X

k=1
τk = c2

τ ,

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j =

c
τ
√

K

3
K
X

k=1

k
X

l=1
τl ≤c3√

K
τ 2
.

From the excess generalization error (B.15), we can derive

ES,A [F(wK) −F(w∗)]

≤(1 + β

γ )

(1

2 + 2βc

τ
√

K
)∥w1 −w∗∥2 + 4βc2

τ 2 F(w∗) + 8β2r2c2

τ 2
+ βLrc2

τ


·
τ
c
√

K

+ 8βe(β + γ)(1 + K/n)

n


c
τ
√

K
∥w1 −w∗∥2 + 2c2

τ 2 F(w∗) + 4βr2c2

τ 2



+ β2r2(β + γ)ec3√

K
τ 2
·
τ
c
√

K
+ β

γ F(w∗)

≤(1 + β

γ )
 τ
√

K
∥w1 −w∗∥2

2c
+ βrc( 8βr
√

Kτ
+
L
√

K
) + 2β∥w1 −w∗∥2

K
)


+ 8βe(β + γ)(1 + K/n)

n


c
τ
√

K
∥w1 −w∗∥2 + 2c2

τ 2 F(w∗) + 4βr2c2

τ 2


+ β2r2(β + γ)ec2

τ

+
β

γ + (1 + β

γ ) 4βc

τ
√

K


F(w∗).

Let F(w∗) = 0, K ≍n and γ = 1. If the average delay satisfies τ ≤K
1
4 (quite reasonable in
asynchronous training), we have max{
τ
√

K ,
1
√

Kτ ,
1
√

K } ≤1

τ . Then the excess generalization error is

ES,A [F(wK) −F(w∗)] = O
1

τ + ∥w1 −w∗∥2

n


.

C
Proof of Generalization in Non-smooth Case (Section 5 in the main text)

C.1
Proof of Lemma 5 (approximately non-expansive property in the non-smooth case)

Under (α, β)-Hölder continuous condition, the gradients exhibit the following co-coercivity

⟨wk−τk−w(i)
k−τk, ∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)⟩≥2β−1

α α
1 + α ∥∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)∥
1+α

α .

A detailed proof of this co-coercivity can be found in [24, 39, 51], and also in Lemma D.2 of [23].
Then we have

∥∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)∥2 ≤
 1 + α

2β−1

α α
⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik)−∇f(w(i)
k−τk; zik)⟩
 2α

1+α

=
1 + α

ηkα ⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩
 2α

1+α ·

2−2α

1+α η

2α
1+α
k
β
2
1+α


≤2

ηk
⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩+ 1 −α

1 + αη

2α
1−α
k
 
2−αβ

2
1−α ,

where the last inequality we use the Young’s inequality (A.1) with p = 1+α

2α , q = 1+α

1−α. That is

η2
k∥∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)∥2 ≤2ηk⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

+ d2
α,βη

2
1−α
k
,
(C.1)

22


---Page Break---
where dα,β =
q

1−α
1+α
 
2−αβ

1
1−α > 0 is a constant dependent on α, β. Without smoothness, the
non-expansive property of delayed gradient updates would be further compromised. However, in
conjunction with inequality (C.1), we can make the following derivation.
wk −ηk∇f(wk−τk; zik) −
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2

= ∥wk −w(i)
k ∥2 + η2
k∥∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)∥2

−2ηk⟨wk −w(i)
k , ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

= ∥wk −w(i)
k ∥2 + η2
k∥∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)∥2

−2ηk⟨wk−τk −w(i)
k−τk, ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

−2ηk⟨wk −wk−τk −(w(i)
k −w(i)
k−τk), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

≤∥wk −w(i)
k ∥2 −2ηk⟨wk −wk−τk −(w(i)
k −w(i)
k−τk), ∇f(wk−τk; zik) −∇f(w(i)
k−τk; zik)⟩

+ d2
α,βη

2
1−α
k
.

Following the (α, β)-Hölder continuous condition and Assumption 1, we can derive

∥∇f(ws−τs; zis)−∇f(w(i)
s−τs; zis)∥≤β∥ws−τs −w(i)
s−τs∥α ≤βrα, for s = k, k −j; j = 1, . . . , τk.
(C.2)
By combing (B.1) and (C.2), the asynchronous gradient updates satisfy the following approximately
non-expansive property under the (α, β)-Hölder continuous condition.
wk −ηk∇f(wk−τk; zik) −
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2 ≤
wk −w(i)
k
2 + d2
α,βη

2
1−α
k

+ 2β2r2αηk

τk
X

j=1
ηk−j.

(C.3)

C.2
Proof of Theorem 4 (algorithm stability under the (α, β)-Hölder continuous gradient
assumption)

We now examine the on-average model stability of the ASGD algorithm under the (α, β)-Hölder
continuous gradient assumption. Following (C.3), if ASGD selects the same sample in both S and
S(i) at the k-th iteration (with probability 1 −1/n), we have
wk+1 −w(i)
k+1
2 ≤
wk −ηk∇f(wk−τk; zik) −
 
w(i)
k −ηk∇f(w(i)
k−τk; zik)
2

≤∥wk −w(i)
k ∥2 + 2ηkβ2r2α
τk
X

j=1
ηk−j + d2
α,βη

2
1−α
k
,

where the first inequality uses the no-expansive projection (A.8). On the other hand, with probability
1/n the selected example is different (ik = i), then

∥wk+1 −w(i)
k+1∥2 ≤∥wk −ηk∇f(wk−τk; zi) −w(i)
k + ηk∇f(w(i)
k−τk; z′
i)∥2

≤(1 + p)∥wk −w(i)
k ∥2 + (1 + 1/p)η2
k∥∇f(wk−τk; zi) −∇f(w(i)
k−τk; z′
i)∥2

≤(1 + p)∥wk −w(i)
k ∥2 + 2(1 + 1/p)η2
k
h
∥∇f(wk−τk; zi)∥2 + ∥∇f(w(i)
k−τk; z′
i)∥2i

≤(1 + p)∥wk −w(i)
k ∥2 + 2(1 + 1/p)c2
α,βη2
k
h
f
2α
1+α (wk−τk; zi) + f
2α
1+α (w(i)
k−τk; z′
i)
i
.

Here we use the inequality (A.4) and the self-bounding property of the α, β-Hölder continuous
gradient (A.5). Combining the above two cases gives

∥wk+1 −w(i)
k+1∥2 ≤(1 + p

n)∥wk −w(i)
k ∥2 + 2ηkβ2r2α
τk
X

j=1
ηk−j + d2
α,βη

2
1−α
k

+
2(1 + 1/p)c2
α,βη2
k
n

h
f
2α
1+α (wk−τk; zi) + f
2α
1+α (w(i)
k−τk; z′
i)
i
.

(C.4)

23


---Page Break---
Following the fact

ES,S′,A[f
2α
1+α (w(i)
k−τk; z′
i)] = ES,A[f
2α
1+α (wk−τk; zi)],

and the same model initialization w1 = w(i)
1 , taking the expectation followed by summation of
inequality (C.4) yields

ES,S′,A∥wk+1 −w(i)
k+1∥2 ≤(1 + p

n)ES,S′,A∥wk −w(i)
k ∥2 + 2ηkβ2r2α
τk
X

j=1
ηk−j + d2
α,βη

2
1−α
k

+
4(1 + 1/p)c2
α,βη2
k
n
ES,A
h
f
2α
1+α (wk−τk; zi)
i

≤

k
X

l=1
(1+ p

n)(k−l)



4(1+1/p)c2
α,βη2
l
n
ES,A
h
f
2α
1+α (wl−τl; zi)
i
+ 2ηlβ2r2α
τl
X

j=1
ηl−j + d2
α,βη

2
1−α
l



.

By the concavity of the function x 7→x
2α
1+α , the on-average model stability of ASGD satisfies

ES,S′,A

"
1
n

n
X

i=1
∥wk+1 −w(i)
k+1∥2
#

≤(1+ p

n)(k−1)
k
X

l=1



4(1+1/p)c2
α,βη2
l
n
ES,A


F

2α
1+α
S
(wl−τl)

+ 2ηlβ2r2α
τl
X

j=1
ηl−j + d2
α,βη

2
1−α
l



.

Let p = n/k, we have (1 + p/n)(k−1) ≤(1 + 1/k)(k−1) ≤e, then

ES,S′,A

"
1
n

n
X

i=1

wk+1 −w(i)
k+1
2
#

≤
4e(1 + k/n)c2
α,β
n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i
+ 2β2r2αe

k
X

l=1
ηl

τl
X

j=1
ηl−j + ed2
α,β

k
X

l=1
η

2
1−α
l

= O
1 + k/n

n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i
+

k
X

l=1
ηl

τl
X

j=1
ηl−j +

k
X

l=1
η

2
1−α
l


.

(C.5)

C.3
Generalization Error with (α, β)-Hölder Continuous Gradient

From the algorithm stability (C.5), it follows that we need to bound Pk
l=1 η2
l ES,A[F

2α
1+α
S
(wl−τl)]
under the α, β-Hölder continuous gradient condition. Using the non-expansive projection (A.8) and
the self-bounding property (A.5), we can derive

∥wk+1 −w∗∥2 ≤∥wk −ηk∇f(wk−τk; zik) −w∗∥2

= ∥wk −w∗∥2 + η2
k∥∇f(wk−τk; zik)∥2 + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 + c2
α,βη2
kf
2α
1+α (wk−τk; zik) + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩.

By the young’s inequality (A.1) with p = 1+α

1−α, q = 1+α

2α , we have

c2
α,βηkf
2α
1+α (wk−τk; zik) =
  2α

1 + α
 2α

1+α c2
α,βηk

·
1 + α

2α f(wk−τk; zik)
 2α

1+α

≤1 −α

1 + α

  2α

1 + α
 2α

1+α c2
α,βηk
 1+α

1−α +
2α
1 + α

1 + α

2α f(wk−τk; zik)
 2α

1+α
1+α

2α

= f(wk−τk; zik) + c′
α,βη

1+α
1−α
k
,

24


---Page Break---
where we define the constant c′
α,β = 1−α

1+α
  2α

1+α
 2α

1−α c

2+2α

1−α
α,β
> 0. With the convexity of f, we can
further derive

∥wk+1 −w∗∥2 ≤∥wk −w∗∥2 + ηkf(wk−τk; zik) + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩+ c′
α,βη

2
1−α
k
≤∥wk −w∗∥2 + ηkf(wk−τk; zik) + 2ηk(f(w∗; zik) −f(wk−τk; zik))

+ 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩+ c′
α,βη

2
1−α
k
≤∥wk −w∗∥2 −ηkf(wk−τk; zik) + 2ηkf(w∗; zik) + 2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩

+ c′
α,βη

2
1−α
k
.
(C.6)
Here, with the inequality (A.2), (A.5) and Assumption 1, we have the following derivation

2ηk⟨wk−τk −wk, ∇f(wk−τk; zik)⟩≤2ηkr · cα,βf
α
1+α (wk−τk; zik)

≤
(1 + α)ηk

2α
f(wk−τk; zik)

α
1+α · 2
1+2α

1+α η

1
1+α
k

α
1 + α


α
1+α cα,βr

≤ηk

2 f(wk−τk; zik) + c′′
α,βηk,

where the last inequality we use the young’s inequality (A.1) with p =
1+α

α , q = 1 + α, and
c′′
α,β = 21+2ααα  rcα,β

1+α
1+α is a constant. Substituting it into (C.6) yields

ηkf(wk−τk; zik) ≤2∥wk −w∗∥2 −2∥wk+1 −w∗∥2 + 4ηkf(w∗; zik) + 2c′
α,βη

2
1−α
k
+ 2c′′
α,βηk.
Multiplying both sides by the non-increasing learning rate yields

η2
kf(wk−τk; zik) ≤2ηk∥wk−w∗∥2−2ηk+1∥wk+1−w∗∥2+4η2
kf(w∗; zik)+2c′
α,βη

3−α
1−α
k
+2c′′
α,βη2
k.
Taking the expectation and summing the inequalities as above gives

k
X

l=1
η2
l ES,A[FS(wl−τl)] ≤2η1∥w1−w∗∥2 + 4

k
X

l=1
η2
l ES[FS(w∗)] + 2c′
α,β

k
X

l=1
η

3−α
1−α
l
+ 2c′′
α,β

k
X

l=1
η2
l .

According to the concavity of the function x 7→x
2α
1+α , we know that

k
X

l=1
η2
l ES,A[F

2α
1+α
S
(wl−τl)] ≤

k
X

l=1
η2
l

Pk
l=1 η2
l ES,A[FS(wl−τl)]

Pk
l=1 η2
l

 2α

1+α

≤2

k
X

l=1
η2
l
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + c′
α,β

k
X

l=1
η

3−α
1−α
l
+ c′′
α,β

k
X

l=1
η2
l
 2α

1+α .

(C.7)

Now, we are ready to analysis the generalization error of ASGD under (α, β)-Hölder continuous
gradient condition. From Lemma 2 (A.13) and stability (C.5), we have

ES,A [F(wk+1) −FS(wk+1)] ≤
c2
α,β
2γ ES,A[F
2α
1+α (wk+1)] + γ

2

n
X

i=1
ES,S′,A

 1

n∥wk+1 −w(i)
k+1)∥2


≤
c2
α,β
2γ ES,A[F
2α
1+α (wk+1)] +
2eγ(1 + k/n)c2
α,β
n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i
+ eγβ2r2α
k
X

l=1
ηl

τl
X

j=1
ηl−j

+
eγd2
α,β
2

k
X

l=1
η

2
1−α
l
.

(C.8)

Let ϵk := max

ES,A [F(wk) −FS(wk)] , 0
	
. By the concavity and sub-additivity of x 7→x
2α
1+α
we have

ES,A[F
2α
1+α (wk+1)] ≤

ES,A [F (wk+1) −FS (wk+1)] + ES,A [FS (wk+1)]
 2α

1+α

≤ϵ

2α
1+α
k+1 +

ES,A

FS (wk+1)
 2α

1+α .

25


---Page Break---
Substituting this back into (C.8) gives

ϵk+1 ≤
c2
α,β
2γ


ϵ

2α
1+α
k+1 +

ES,A

FS (wk+1)
 2α

1+α 
+
2eγ(1 + k/n)c2
α,β
n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i

+ eγβ2r2α
k
X

l=1
ηl

τl
X

j=1
ηl−j +
eγd2
α,β
2

k
X

l=1
η

2
1−α
l
.

Using the young’s inequality (A.1) with p = 1+α

1−α, q = 1+α

2α yields

c2
α,β
2γ ϵ

2α
1+α
k+1 =
 4α

1 + α

 2α

1+α c2
α,β
2γ ·
1 + α

4α ϵk+1
 2α

1+α ≤1 −α

1 + α

 4α

1 + α

 2α

1−α c2
α,β
2γ

 1+α

1−α + 1

2ϵk+1.

It then holds that

ES,A [F(wk+1) −FS(wk+1)] ≤2(1 −α)

1 + α

 4α

1 + α

 2α

1−α c2
α,β
2γ

 1+α

1−α +
c2
α,β

γ


ES,A

FS (wk+1)
 2α

1+α

+
4eγ(1 + k/n)c2
α,β
n

k
X

l=1
η2
l ES,A
h
F

2α
1+α
S
(wl−τl)
i
+ 2eγβ2r2α
k
X

l=1
ηl

τl
X

j=1
ηl−j + eγd2
α,β

k
X

l=1
η

2
1−α
l
.

Substituting (C.7), we can get the following generalization error of ASGD under the (α, β)-Hölder
continuous gradient condition

ES,A [F(wk+1) −FS(wk+1)] ≤2(1 −α)

1 + α

 4α

1 + α

 2α

1−α c2
α,β
2γ

 1+α

1−α +
c2
α,β

γ


ES,A

FS (wk+1)
 2α

1+α

+
8eγ(1 + k/n)c2
α,β
n


k
X

l=1
η2
l
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + c′
α,β

k
X

l=1
η

3−α
1−α
l
+ c′′
α,β

k
X

l=1
η2
l
 2α

1+α

+ 2eγβ2r2α
k
X

l=1
ηl

τl
X

j=1
ηl−j + eγd2
α,β

k
X

l=1
η

2
1−α
l
.

(C.9)

C.4
Proof of Lemma 6 (Optimization error under the (α, β)-Hölder continuous gradient
assumption)

Leveraging the non-expansive projection property (A.8), convexity and α, β-Hölder continuous
property (A.5), we can derive

∥wk+1 −w∗∥2 ≤∥wk −ηk∇f(wk−τk; zik) −w∗∥2

= ∥wk −w∗∥2 + η2
k∥∇f(wk−τk; zik)∥2 + 2ηk⟨w∗−wk, ∇f(wk−τk; zik)⟩

≤∥wk −w∗∥2 + c2
α,βη2
kf
2α
1+α (wk−τk; zik) + 2ηk⟨w∗−wk, ∇f(wk; zik)⟩

+ 2ηk⟨w∗−wk, ∇f(wk−τk; zik) −∇f(wk; zik)⟩

≤∥wk −w∗∥2 + c2
α,βη2
kf
2α
1+α (wk−τk; zik) + 2ηk(f(w∗; zik) −f(wk; zik))

+ 2βηk∥w∗−wk∥∥wk−τk −wk∥α.
(C.10)
From the iterative scheme of ASGD (7), (A.8) and the sub-additivity of x 7→xα, we know that

∥wk −wk−τk∥α ≤

τk
X

j=1
∥wk−j+1 −wk−j∥α ≤

τk
X

j=1
ηα
k−j∥∇f(wk−j−τk−j; zik−j)∥α.
(C.11)

Taking the expectation followed by a summation of (C.10) yields

2

K
X

k=1
ηkES,A[FS(wk) −FS(w∗)] ≤∥w1 −w∥2 + c2
α,β

K
X

k=1
η2
kES,A[F

2α
1+α
S
(wk−τk)]

+ 2βLαr

K
X

k=1
ηk

τk
X

j=1
ηα
k−j,

26


---Page Break---
where we used (C.11), Assumptions 1 and 2. Combing with (C.7), we get the following optimization
error bound of ASGD with the (α, β)-Hölder continuous gradient.

K
X

k=1
ηk[FS(wk) −FS(w∗)] ≤1

2∥w1 −w∗∥2 + βLαr

K
X

k=1
ηk

τk
X

j=1
ηα
k−j

+ c2
α,β
 K
X

k=1
η2
k
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

K
X

k=1
η2
kF(w∗) + c′
α,β

K
X

k=1
η

3−α
1−α
k
+ c′′
α,β

K
X

k=1
η2
k
 2α

1+α .

(C.12)

According to the convexity of the function FS, we know that

ES,A[FS(wK) −FS(w∗)] ≤
PK
k=1 ηkES,A[FS(wk) −FS(w∗)]

PK
k=1 ηk

= O
  PK
k=1 η2
k
 1−α

1+α
PK
k=1 ηk

h
η1∥w1 −w∗∥2 +
 
1 + F(w∗)
 K
X

k=1
η2
k +

K
X

k=1
η

3−α
1−α
k
i 2α

1+α

+
∥w1 −w∗∥2 + PK
k=1 ηk
Pτk
j=1 ηα
k−j
PK
k=1 ηk


.

C.5
Proof of Theorem 5 (Excess generalization error under the (α, β)-Hölder continuous
gradient assumption)

Multiplying both sides of the generalization error (C.9) by the learning rate ηk+1 followed by
summation yields

K
X

k=1
ηkES,A [F(wk)] ≤

K
X

k=1
ηkES,A [FS(wk)] +
c2
α,β

γ

K
X

k=1
ηk

ES,A

FS (wk)
 2α

1+α

+ 2
 4α

1+α

 2α

1−α c2
α,β
2γ

 1+α

1−α
K
X

k=1
ηk +
8eγ(1+K/n)c2
α,β
n

K
X

k=1
ηk

k
X

l=1
η2
l
 1−α

1+α

·

η1∥w1−w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + c′
α,β

k
X

l=1
η

3−α
1−α
l
+ c′′
α,β

k
X

l=1
η2
l
 2α

1+α

+ 2eγβ2r2α
K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j + eγd2
α,β

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l
.

According to the concavity of the function x 7→x
2α
1+α , we know that

K
X

k=1
ηk

ES,A

FS (wk)
 2α

1+α ≤

K
X

k=1
ηk

PK
k=1 ηkES,A[FS(wk)]

PK
k=1 ηk

 2α

1+α

=
 K
X

k=1
ηk
 1−α

1+α  K
X

k=1
ηkES,A[FS(wk)]
 2α

1+α

≤1 −α

1 + α

K
X

k=1
ηk +
2α
1 + α

K
X

k=1
ηkES,A[FS(wk)],

27


---Page Break---
where the last inequality uses the young’s inequality (A.1) with p = 1+α

1−α, q = 1+α

2α . Then

K
X

k=1
ηkES,A [F(wk)] ≤

1+ 2αc2
α,β
γ(1+α)


K
X

k=1
ηkES,A [FS(wk)]+

2
 4α

1+α

 2α

1−α c2
α,β
2γ

 1+α

1−α + (1−α)c2
α,β
γ(1+α)


K
X

k=1
ηk

+ 8eγ(1+K/n)c2
α,β
n

K
X

k=1
ηk

k
X

l=1
η2
l
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + c′
α,β

k
X

l=1
η

3−α
1−α
l
+ c′′
α,β

k
X

l=1
η2
l
 2α

1+α

+ 2eγβ2r2α
K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j + eγd2
α,β

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l
.

From the optimization error (C.12), we then have

K
X

k=1
ηkES,A [F(wk)−F(w∗)] ≤2αc2
α,β
γ(1+α)

K
X

k=1
ηkF(w∗)+

1+ 2αc2
α,β
γ(1+α)

1

2∥w1−w∗∥2+βLαr

K
X

k=1
ηk

τk
X

j=1
ηα
k−j



+

1 + 2αc2
α,β
γ(1 + α)


c2
α,β

K
X

k=1
η2
k
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

K
X

k=1
η2
kF(w∗) + c′
α,β

K
X

k=1
η

3−α
1−α
k
+ c′′
α,β

K
X

k=1
η2
k
 2α

1+α

+ 8eγ(1+K/n)c2
α,β
n

K
X

k=1
ηk

k
X

l=1
η2
l
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

k
X

l=1
η2
l F(w∗) + c′
α,β

k
X

l=1
η

3−α
1−α
l
+ c′′
α,β

k
X

l=1
η2
l
 2α

1+α

+ 2eγβ2r2α
K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j +eγd2
α,β

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l
+

2
 4α

1+α

 2α

1−α c2
α,β
2γ

 1+α

1−α + (1−α)c2
α,β
γ(1+α)


K
X

k=1
ηk.

By the convexity of F, the excess generalization error of ASGD under the (α, β)-Hölder continuous
gradient satisfies

ES,A [F(wk) −F(w∗)] ≤
PK
k=1 ηkES,A [F(wk) −F(w∗)]
PK
k=1 ηk

≤

1+ 2αc2
α,β
γ(1 + α)


c2
α,β

K
X

k=1
η2
k
 1−α

1+α 
η1∥w1−w∗∥2+2

K
X

k=1
η2
kF(w∗)+c′
α,β

K
X

k=1
η

3−α
1−α
k
+c′′
α,β

K
X

k=1
η2
k
 2α

1+α /

K
X

k=1
ηk

+ 8eγ(1 + K/n)c2
α,β
n


K
X

k=1
η2
k
 1−α

1+α 
η1∥w1 −w∗∥2 + 2

K
X

k=1
η2
kF(w∗) + c′
α,β

K
X

k=1
η

3−α
1−α
k
+ c′′
α,β

K
X

k=1
η2
k
 2α

1+α

+

2eγβ2r2α
K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j +eγd2
α,β

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l


/

K
X

k=1
ηk+2
 4α

1+α

 2α

1−α c2
α,β
2γ

 1+α

1−α + (1−α)c2
α,β
γ(1+α)

+

1 + 2αc2
α,β
γ(1 + α)

1

2∥w1 −w∗∥2 + βLαr

K
X

k=1
ηk

τk
X

j=1
ηα
k−j


/

K
X

k=1
ηk + 2αc2
α,β
γ(1 + α)F(w∗).

(C.13)

Let γ > 1, then we are arrive at

ϵex-gen = O
 K
X

k=1
η2
k
 1−α

1+α 
η1∥w1 −w∗∥2 +

K
X

k=1
η2
kF(w∗) +

K
X

k=1
η

3−α
1−α
k
+

K
X

k=1
η2
k
 2α

1+α /

K
X

k=1
ηk

+ γ(1 + K/n)

n

 K
X

k=1
η2
k
 1−α

1+α 
η1∥w1 −w∗∥2 +

K
X

k=1
η2
kF(w∗) +

K
X

k=1
η

3−α
1−α
l
+

K
X

k=1
η2
k
 2α

1+α

+ γ
 K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j +

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l


/

K
X

k=1
ηk + γ
1+α
α−1

+

∥w1 −w∗∥2 +

K
X

k=1
ηk

τk
X

j=1
ηα
k−j

/

K
X

k=1
ηk + 1

γ
 
1 + F(w∗)

.

(C.14)

28


---Page Break---
C.5.1
Excess Generalization Error with the Learning Rate ηk = c(τ
√

K)−1

Set the learning rate ηk = c(τ
√

K)−1 with c>0, and τ =PK
k=1 τk/K, direct calculation gives

K
X

k=1
ηk = c
√

K/τ,

K
X

k=1
η2
k = c2/τ 2,

K
X

k=1
ηk

k
X

l=1
ηl

τl
X

j=1
ηl−j =

c
τ
√

K

3
K
X

k=1

k
X

l=1
τl ≤c3√

K
τ 2

K
X

k=1
ηk

τk
X

j=1
ηα
k−j = c1+αK
1−α

2
τ α
,

K
X

k=1
η

3−α
1−α
k
≍K1−
3−α
2(1−α )τ −3−α

1−α ,

K
X

k=1
ηk

k
X

l=1
η

2
1−α
l
≍K2−
3−α
2(1−α) τ −3−α

1−α .

Following the excess generalization error (C.14), we know that

ϵex-gen = O
 1

τ 2
 1−α

1+α ∥w1 −w∗∥2

√

Kτ
+ 1

τ 2
 
1 + F(w∗)

+ K1−
3−α
2(1−α)

τ
3−α
1−α

 2α

1+α ·
τ
√

K

+ γ(1 + K/n)

n

 1

τ 2
 1−α

1+α ∥w1 −w∗∥2

√

Kτ
+ 1

τ 2
 
1 + F(w∗)

+ K1−
3−α
2(1−α)

τ
3−α
1−α

 2α

1+α

+ γ
√

K
τ 2 + K2−
3−α
2(1−α)

τ
3−α
1−α


·
τ
√

K
+ γ
1+α
α−1 +

∥w1 −w∗∥2 + K
1−α

2
τ α

·
τ
√

K
+ 1

γ
 
1 + F(w∗)

.

By the sub-additivity of x 7→x
2α
1+α , (α ∈[0, 1]), we can derive

ϵex-gen = O
h τ
√

K
+ γ(1 + K/n)

n

i

τ −2(1−α)

1+α
∥w1 −w∗∥
4α
1+α

K
α
1+α τ
2α
1+α
+
1

τ
4α
1+α

 
1 + F
2α
1+α (w∗)

+
K−
α
1−α

τ

2α(3−α)
(1+α)(1−α)



+ γ
1

τ + K−
α
1−α

τ
2
1−α


+ τ∥w1 −w∗∥2

√

K
+ τ 1−α

K
α

2
+ γ
1+α
α−1 + 1

γ
 
1 + F(w∗)


= O
h τ
√

K
+ γ(1 + K/n)

n

iK−
α
1+α

τ
2
1+α ∥w1 −w∗∥
4α
1+α + 1

τ 2
 
1 + F
2α
1+α (w∗)

+ K−
α
1−α

τ
2
1−α



+ γ
1

τ + K−
α
1−α

τ
2
1−α


+
τ
√

K
+ τ 1−α

K
α

2
+ γ
1+α
α−1 + 1

γ
 
1 + F(w∗)

.

Omitting the non-dominant term gives (with γ > 1)

ϵex-gen = O
h τ
√

K
+ γ(1 + K/n)

n

iK−
α
1+α

τ
2
1+α ∥w1 −w∗∥
4α
1+α + 1

τ 2
 
1 + F
2α
1+α (w∗)


+ γ

τ +
τ
√

K
+ τ 1−α

K
α

2
+ 1

γ
 
1 + F(w∗)

.

If F(w∗) = 0, K ≍n, then the excess generalization error is

ϵex-gen = O
 τ
√

K
+ γ

n

K−
α
1+α

τ
2
1+α ∥w1 −w∗∥
4α
1+α + 1

τ 2

+ γ

τ +
τ
√

K
+ τ 1−α

K
α

2
+ 1

γ


.

Let γ =
√

τ, we have that max{
τ
√

K , γ

n} =
τ
√

K , and then

ϵex-gen = O
∥w1 −w∗∥
4α
1+α

K

1+3α
2(1+α) τ
1−α
1+α
+
1
√

Kτ
+ 1
√

τ +
τ
√

K
+ τ 1−α

K
α

2


.

Furthermore, if the average delay satisfies τ ≤Kα′ with α′ = min{ 1

3,
α
3−2α}, we know that

max
n
1
√

Kτ
,
1
√

τ ,
τ
√

K
, τ 1−α

K
α

2

o
=
1
√

τ .

On the other hand, since τ ≥1 in the asynchronous training and α ∈[0, 1], we have

K

1+3α
2(1+α) τ
1−α
1+α ≥K

1+3α
2(1+α) ≥K
1+α

2
≍n
1+α

2 .
Then, we are arrive at

ϵex-gen = ES,A [F(wk) −F(w∗)] = O
 1
√

τ + ∥w1 −w∗∥
4α
1+α
√n1+α


.

The proof is complete.

29


---Page Break---
0.0
0.5
1.0
1.5
Iteration
1e4

1

2

3

4

Algorithm stability

=4
=8
=16
=32

(a) Convex optimization on RCV1

0
1
2
3
4
Iteration
1e4

140

160

180

Algorithm stability

=4
=8
=16
=32

(b) CV task on CIFAR100

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

300

301

Algorithm stability

=4
=8
=16
=32

(c) NLP task on SST-2

Figure 2: The on-average model stability in training various machine learning tasks using ASGD with
learning rate ηk = 0.1/τ. The horizontal axis denotes the number of asynchronous training iterations,
and the legend represents the average delay. A degradation in algorithm stability is observed as the
number of training iterations increases.

D
More Experiment Results

This section provides additional details on the experimental setup, as well as the stability and
generalization results trained with a delay-independent constant learning rate. Our experiments use
the more general asynchronous stochastic gradient descent format (9), i.e.,

wk+1 = wk −ηk
X

m∈Mk
gm
k−τk.

In practical applications, the gradient gm
k−τk is evaluated on a mini-batch of the training data. The
batch size of each worker in this experiment was set to 16. It should be noted that we only simulated
8 workers in the BERT experiments due to memory limitations. All of our experiments were
implemented with PyTorch on Nvidia RTX-3090 24 GB GPUs.

In Figure 3 and 4, the stability and generalization results of the ASGD algorithm, employing a delay-
independent learning rate of ηk = 0.01, are illustrated. In this scenario, increasing the delay still
improves the algorithm stability and reduces the generalization error, indicating that asynchronous
training is indeed beneficial for generalization.

0
1
2
3
4
Iteration
1e4

126

129

132

135

Algorithm stability

=4
=8
=16
=32

(a) CV task on CIFAR10

0
1
2
3
4
Iteration
1e4

128

136

144

152

Algorithm stability

=4
=8
=16
=32

(b) CV task on CIFAR100

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

300

301

Algorithm stability

=4
=8
=16
=32

(c) NLP task on SST-2

Figure 3: The on-average model stability in training various machine learning tasks using ASGD
with delay-independent constant learning rate ηk = 0.01. The horizontal axis denotes the number
of asynchronous training iterations, and the legend represents the average delay. A degradation in
algorithm stability is observed with an increase in training iterations.

Figure 5 shows the training, testing and generalization errors of three categories of machine learning
models. The generalization errors are roughly of the same order of magnitude as the training and
testing errors. In certain model tasks, particularly BERT on the SST-2 task, overfitting phenomena
are present, contributing significantly to the generalization gap. Therefore, we need to complete
the training process as soon as possible to improve the model generalization performance, which is
consistent with our theoretical analysis in Section 4.2.

30


---Page Break---
0
1
2
3
4
Iteration
1e4

0.0

1.5

3.0

4.5

Generalization error

1e
1

=4
=8
=16
=32

(a) CV task on CIFAR10

0
1
2
3
4
Iteration
1e4

0

1

Generalization error

=4
=8
=16
=32

(b) CV task on CIFAR100

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

0

1

Generalization error

=4
=8
=16
=32

(c) NLP task on SST-2

Figure 4: The generalization errors in training various machine learning tasks using ASGD with
delay-independent constant learning rate ηk = 0.01. The trend of generalizability with the number of
iterations is analogous to the algorithm stability depicted in Figure 3, and appropriately increasing
the asynchronous delay can enhance generalization performance.

0.0
0.5
1.0
1.5
Iteration
1e4

0.08

0.16

0.24

0.32

Training error

=4
=8
=16
=32

(a) Convex optimization on RCV1

0.0
0.5
1.0
1.5
Iteration
1e4

0.08

0.16

0.24

0.32

Testing error

=4
=8
=16
=32

(b) Convex optimization on RCV1

0.0
0.5
1.0
1.5
Iteration
1e4

0.0

0.4

0.8

1.2

1.6

Generalization error

1e
2

=4
=8
=16
=32

(c) Convex optimization on RCV1

0
1
2
3
4
Iteration
1e4

0

1

2

3

4

Training error

=4
=8
=16
=32

(d) CV task on CIFAR100

0
1
2
3
4
Iteration
1e4

2

3

4

Testing error

=4
=8
=16
=32

(e) CV task on CIFAR100

0
1
2
3
4
Iteration
1e4

0

1

Generalization error

=4
=8
=16
=32

(f) CV task on CIFAR100

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

0.0

0.2

0.4

0.6

Training error

=4
=8
=16
=32

(g) NLP task on SST-2

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

0.6

0.8

1.0

1.2

1.4

Testing error

=4
=8
=16
=32

(h) NLP task on SST-2

0.0
0.2
0.4
0.6
0.8
1.0
Iteration
1e5

0

1

Generalization error

=4
=8
=16
=32

(i) NLP task on SST-2

Figure 5: The training, testing and generalization errors of three categories of machine learning
models trained using ASGD with learning rate ηk = 0.1/τ. The horizontal axis denotes the number
of asynchronous training iterations, and the legend represents the average delay.

31


---Page Break---
E
Contributions and Limitations

E.1
Contributions

The main challenge of this study is to establish sharper generalization bounds for the ASGD algorithm
under much weaker assumptions.

Notably, the existing studies on the generalization of ASGD are limited. Study [33] provides
only vacuous exponential generalization bounds and relies on strict assumptions such as Lipschitz
continuous and smooth functions. Another work [13] establishes tighter generalization bounds, but
its analytical techniques only applicable to smooth quadratic convex problems.

Our contributions have been detailed in Section 1, and the following table further compares the
required assumptions and theoretical results of the related works.

Table 1: Comparison with related work.

Regatti et al. [33]
Deng et al. [13]
Ours

Lipschitz assumption?
L-Lipschitz
Not required
Not required

Smoothness assumption?
β-smooth
β-smooth
(α, β)-Hölder continuous

Convexity?
Non-convex
Quadratic convex
General convex

Generalization error
O
  K ˆτ

nˆτ

e
O
  K−ˆτ

nˆτ

O
  1

τ +
1
√

K


Excess generalization error
N/A
N/A
O
  1
√

τ + ∥w1−w∗∥
4α
1+α
√n1+α


E.2
Limitations

Assumption.
In Assumptions 1 and 2, we have listed the assumptions required for this paper and
explained their roles and plausibility. It is crucial to note that this study aims to establish sharper
stability and generalization error bounds under much weaker assumptions. If we adopt stronger
assumptions, such as the assumption in paper [55] that the difference between models wk and w(i)
k
follows a normal distribution with bounded mean and variance, we can obtain better results (in terms
of the training sample size n).

Pessimistic result.
The experiments in Appendix D, concerning delay-independent fixed learning
rates, show that the generalization error bound (B.10) is pessimistic, i.e., asynchronous training
is beneficial for generalization even if a fixed learning rate is used. A potential avenue for future
research lies in exploring tighter high probability bounds that attenuate the dominant role of the
learning rate on generalization, thereby elucidating the experimental phenomena in Appendix D.

Non-convex study.
In the non-convex setting, the delayed gradient update operator cannot maintain
the approximately non-expansive property. Consequently, directly extending the analysis of this paper
to non-convex scenarios would yield an exponential generalization error bound, similar to the findings
in study [33]. Unfortunately, this upper bound is pessimistic and vacuous. Exploring sharper stability
and generalization error bounds of ASGD in non-convex scenarios is extremely challenging. Future
research on non-convex problems could focus on demonstrating that asynchronous gradient updates
are approximately non-expansive even without the convexity property, then leading to non-vacuous
stability and generalization results.

Additionally, while our theoretical analysis is grounded in the general convex condition, our non-
convex experiments show that the theoretical results in this paper are applicable in a broader range of
non-convex machine learning tasks (particularly deep learning), which motivates us to further explore
tighter stability and generalization results for ASGD in the non-convex scenarios in the future.

32


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope, please refer to lines 53-71.
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
Justification: In Section 7 and Appendix E.2, we have discussed the limitations of this work,
namely the pessimistic result at a delay-independent fixed learning rate and not establishing
sharper stability and generalizability results in non-convex settings.
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

33


---Page Break---
Answer: [Yes]

Justification: In Assumptions 1 and 2, we have listed the assumptions required for this paper
and explained their roles and plausibility. Appendix A-C provides complete proof details of
the theorems and lemmas in the paper.

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
Justification: In Section 6 and Appendix D, we have described the experimental setup in
detail, and we have also uploaded the source code in the Supplementary Material.

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

34


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We have submitted the source code in the Supplementary Material and provided
sufficient instructions for usage in the README.md file.

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

Justification: We have described the experimental setup in detail in Section 6 and Appendix
D, and provided more details in the source code.

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

Justification: In Section 6, we have repeated the experiment several times by choosing
different random seeds to verify the theoretical findings.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

35


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
Justification: In Appendix D, we have provided sufficient information on the computer
resources needed to reproduce the experiments.

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

Justification: We have reviewed the NeurIPS Code of Ethics and the research conducted in
the paper conform with the NeurIPS Code of Ethics in every respect.

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
Justification: This paper focuses on stability and generalizability analysis of fundamental
optimization algorithm. No societal impacts are discussed or related to the research.

Guidelines:

36


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
Justification: This paper focuses on stability and generalizability analysis of fundamental
optimization algorithm and poses no such risks.

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

Justification: The models and datasets used in this paper are publicly available and we have
cited the corresponding papers.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

37


---Page Break---
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

38


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

39


---Page Break---
