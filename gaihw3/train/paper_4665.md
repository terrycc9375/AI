Few-Shot Diffusion Models Escape the Curse of
Dimensionality

Ruofeng Yang1, Bo Jiang1, Cheng Chen2, Ruinan Jin34,
Baoxiang Wang34, Shuai Li∗1

1 John Hopcroft Center for Computer Science, Shanghai Jiao Tong University
2 East China Normal University
3 The Chinese University of Hong Kong, Shenzhen
4 Vector Institute
{wanshuiyin, bjiang, shuaili8}@sjtu.edu.cn,
chchen@sei.ecnu.edu.cn, {jinruinan,bxiangwang}@cuhk.edu.cn

Abstract

While diffusion models have demonstrated impressive performance, there is a
growing need for generating samples tailored to specific user-defined concepts.
The customized requirements promote the development of few-shot diffusion
models, which use limited nta target samples to fine-tune a pre-trained diffusion
model trained on ns source samples. Despite the empirical success, no theoretical
work specifically analyzes few-shot diffusion models. Moreover, the existing
results for diffusion models without a fine-tuning phase can not explain why few-
shot models generate great samples due to the curse of dimensionality. In this
work, we analyze few-shot diffusion models under a linear structure distribution
with a latent dimension d. From the approximation perspective, we prove that
few-shot models have a eO(n−2/d
s
+ n−1/2
ta
) bound to approximate the target score
function, which is better than n−2/d
ta
results. From the optimization perspective, we
consider a latent Gaussian special case and prove that the optimization problem
has a closed-form minimizer. This means few-shot models can directly obtain an
approximated minimizer without a complex optimization process. Furthermore,
we also provide the accuracy bound eO(1/nta + 1/√ns) for the empirical solution,
which still has better dependence on nta compared to ns. The results of the real-
world experiments also show that the models obtained by only fine-tuning the
encoder and decoder specific to the target distribution can produce novel images
with the target feature, which supports our theoretical results.

1
Introduction

In recent years, diffusion models have shown an excellent ability to generate diverse, high-quality
images and show state-of-the-art performance in the large-scale, standard dataset (Rombach et al.,
2022; Ho et al., 2022; Li et al., 2024). However, users often desire to generate samples that resemble
the ones they provide, such as images related to their families, daily lives, or specific items. These
user-provided samples are typically limited in number and do not appear frequently in large-scale
datasets. Consequently, training a diffusion model from scratch using such limited, personalized
samples often results in poor performance. To cater the customized requirements of users, few-shot
diffusion models attract much attention. Few-shot diffusion models aim to fine-tune a pre-trained
diffusion model using a limited amount of data (5 ∼10 samples), and they have recently delivered
impressive results in various domains, including image generation (Ruiz et al., 2023; Han et al., 2023;
Zhu et al., 2023), video generation (Chen et al., 2023b), and the medical domain (Dutt et al., 2023).

∗Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Before the fine-tuning phase, we first need to train a diffusion model on the large source dataset
{Xs,i}ns
i=1 as the pre-trained model. A diffusion model consists of a forward process and a reverse
process (Song et al., 2020). The forward process gradually converts the data distribution into Gaussian
noise. The reverse process sequentially removes the noise in the data to generate samples, which
relies on the gradient of logarithmic forward process density (a.k.a. score function). To run the
reverse process, diffusion models use a neural network to approximate the unknown score function.

With a pre-trained diffusion model, the paradigm to obtain a few-shot diffusion model is to fine-tune
the model using a limited target dataset {Xta,i}nta
i=1. In earlier times, fully fine-tuned methods,
such as DreamBooth (Ruiz et al., 2023), provided an important boost for developing few-shot
models. However, they also show that the diffusion models suffer from the overfitting and memory
phenomenon when fine-tuning all parameters. Furthermore, a fully fine-tuned method is both memory
and time inefficient (Xiang et al., 2023). To avoid the above problems, many works freeze most
parameters and fine-tune some key parameters, such as cross-attention layers (Kumari et al., 2023;
Moon et al., 2022), some concept neurons (Liu et al., 2023) or text-embedding (Gal et al., 2022),
to approximate the ground-truth target score function. These works not only preserve the prior
information but also have a lower requirement for the target dataset size, which is more practical for
applications. Hence, we aim to explain the great performance of these models in this work.

Despite the empirical success, no existing theoretical work specifically analyzes the approximation
bound for few-shot diffusion models, and the following question remains open:

Do few-shot diffusion models with a fine-tuning phase enjoy a small approximation error with a
limited target dataset?

For the approximation error bound, some works currently analyze diffusion models without a fine-
tuning phase (Oko et al., 2023; Chen et al., 2023c; Yuan et al., 2023; Li et al., 2023b). Importantly,
when analyzing general, bounded data, these works suffer from the curse of dimensionality. More
specifically, Oko et al. (2023) analyze bounded distribution and show the n−s′/D
s
approximation
bound, where D is the data dimension of Xs. Chen et al. (2023c) analyze linear structure distribution
Xs = AsZ with subgaussian latent variable Z ∈Rd and achieve n−2/d
s
results. Since the source
dataset size is large enough, the influence of dimension is tolerable. However, for the limited target
dataset, if trivially using the above technique, the bound is n−1/D
ta
or n−2/d
ta
, which is large and can
not explain why few-shot diffusion models efficiently approximate the target score function.

In this work, for the first time, we propose the approximation bound specifically to few-shot diffusion
models with a fine-tuning phase and prove that the few-shot diffusion model can escape the curse of
dimensionality. More specifically, we show that when assuming (1) linear structure data and (2) the
source and the target data share latent distribution, the few-shot diffusion models with a fine-tuning
phase achieve eO(n−2/d
s
+ n−1/2
ta
) approximation error bound, which makes the first step to explain
why few-shot diffusion models have great performance in the application. Generally speaking, due to
the component n−1/2
ta
, the few-shot diffusion only needs a few target samples to achieve the same
bound compared to n−2/d
s
. To support our augmentation, we calculate the requirement of nta to
obtain an accurate enough approximated target score function in popular datasets. Table 1 shows
that the requirement of nta is about 5 ∼10 samples, matching the customized diffusion model
requirement. We also do experiments on the real-world datasets and show that 10 target images are
enough for few-shot models to generate novel images with the target feature (Section 6).

After directly using the property of the minimizer to obtain an approximation bound, we analyze how
to optimize the few-shot diffusion models to obtain a minimizer. Since the score-matching objective
function is highly non-convex, only a few works analyze the optimization problem of diffusion
models (Shah et al., 2023; Bruno et al., 2023; Cui et al., 2023; Li et al., 2023b). Furthermore, these
works either require (1) an exponential size neural network (Li et al., 2023b) or (2) a distribution
determined by one variable (Shah et al., 2023; Bruno et al., 2023; Cui et al., 2023) to simplify the
optimization problem. This work proves that few-shot diffusion models can simplify the optimization
problem without these requirements. When analyzing the optimization problem, we focus on a
Gaussian latent variable special case 2. Then, we prove that the expected few-shot objective function
has a closed-form minimizer, which means the empirical solution can be directly obtained without a
complex optimization process. We also prove the accuracy bound eO(1/nta + 1/√ns) of empirical

2Though it is a special case, the previous analysis can not be used since it is determined by two components.

2


---Page Break---
closed-form solution, which still has better dependence on the target dataset. In conclusion, we
accomplish the following results for few-shot diffusion models under linear structure distribution:

• For the approximation bound, we consider a subgaussian latent variable and prove eO(n−2/d
s
+
n−1/2
ta
) bound for few-shot models, which is better than n−2/d
ta
result without fine-tuning.
• For the optimization problem, we consider a latent Gaussian special case and prove that the
expected few-shot objective function has a closed-form minimizer. Furthermore, we prove
the accuracy bound eO(1/nta + 1/√ns) for the empirical closed-form solution.
• To support our theoretical results, we do real-world experiments and show that the models
obtained by only fine-tuning specific encoder and decoder can use only 10 target images to
generate novel images with the target feature.

2
Related Work

The approximation error bound. Recently, some works analyze the approximation error bound
of diffusion models without a fine-tuning phase. Oko et al. (2023) analyze s′-order bounded deriva-
tives distribution and show the approximation error bound is n−s′/D
s
. Chen et al. (2023c) analyze
distribution with linear structure and subgaussian latent variable and show that the n−2/d
s
result.
The approximation error bound of the above works suffers the curse of (latent) dimensionality. To
avoid this phenomenon, some works analyze special data distributions. Shah et al. (2023) and Cui
et al. (2023) analyze the mixture of Gaussian (MOG) with known variance and achieve a 1/ns
approximation bound. Yuan et al. (2023) analyze linear structure distribution with Gaussian latent
variable and achieve 1/√ns result. Mei & Wu (2023) analyze Ising models and prove that the term
corresponds to ns is 1/√ns. However, the remaining terms do not converge to 0 when ns goes to
+∞. For general bounded data distribution, Li et al. (2023b) provide a n−2/5
s
approximation error
bound. However, they use a 2-layer random feature network and only allow the second linear layer to
be trainable. Hence, the network size is exp (ns) compared to Poly(ns) size of all previous works.

The optimization of diffusion models. Since the score matching objective function is highly
non-convex, only a few works analyze how to optimize it to obtain a minimizer (Shah et al., 2023;
Cui et al., 2023; Bruno et al., 2023; Li et al., 2023b). These works either make assumptions about
data distribution or network size to guarantee only one optimization variable, leading to a simpler
optimization problem. For special data distributions, Bruno et al. (2023) and Cui et al. (2023) analyze
a Gaussian with fixed variance and a 2-mode mixture of Gaussian (MOG) with equal, trainable mean
and fixed variance, respectively. Shah et al. (2023) analyze a multi-mode MOG with a fixed variance
and prove a local convergence guarantee. Since they assume the distance between any two modes is
large enough and a good enough initialization, the optimization problem is similar to optimizing a
Gaussian distribution. For the large neural network size, Li et al. (2023b) analyze a general, bounded
distribution with a 2-layer NN. Note that they require exp (ns) hidden neurons and only allow the
linear layer to be trainable, which also leads the optimization problem to a convex optimization.

3
The Introduction of Few-shot Diffusion Models

With pre-trained models, the paradigm to obtain a few-shot diffusion model is to freeze most
parameters and fine-tune some key parameters corresponding to the target data distribution. Since
the analysis of few-shot diffusion models relies heavily on the pre-trained model, this section first
provides a concise overview of the fundamental concepts and notations associated with diffusion
models. Then, we introduce the paradigm of few-shot diffusion models in Section 3.2.

3.1
The Forward and Reverse Process

Let q0 be the data distribution. Given X0 ∼q0 ∈RD, non-decreasing function f(Xt, t) and g(t),
the forward process is defined by:
dXt = f(Xt, t)dt + g(t)dBt ,
where {Bt}t∈[0,T ] is a D-dimensional Brownian motion. In this work, we choose f(Xt, t) =
−1/2Xt and g(t) = 1, which corresponds to variance preserving (VP) forward process and is widely

3


---Page Break---
used in practice 3(Shah et al., 2023; Song et al., 2020). Let qt be the density function of Xt. Once a
forward process is chosen, the conditional distribution of Xt|X0 is qt(Xt|X0) = N(mtX0, σ2
t ID),
where mt = e−t/2, σ2
t = 1 −e−t. Note that when t goes to +∞, qt converges to N(0, ID), which
is helpful in choosing the initial distribution for the sampling process.

To generate samples, diffusion models first reverse the forward SDE to obtain the reverse process.
Then, we approximate the ground-truth score function ∇log qt(·) by using a neural network s(·, t)
due to the requirement of the reverse process (see Section 3.2). Finally, diffusion models choose
a discretization scheme to discretize the continuous reverse process to obtain an implementable
algorithm. Let t0 ≤t1 ≤· · · ≤tK = T be the discretization points in the forward time and
hk = tk −tk−1 be the k-th stepsize. When considering the reverse time, we define t′
k = T −tK−k.
In this work, we choose the exponential integrator (EI) discretization scheme, which has great
performance (Zhang & Chen, 2022). The EI discretization freezes the approximated score at t′
k and
runs the following process in the reverse time:

dbYt =
h
f(bYt, T −t) + g(T −t)2s(bYt′
k, T −t′
k)
i
dt + g(T −t)dBt , t ∈[t′
k, t′
k+1] ,

where bY0 ∼N(0, ID) due to the stationary distribution of the forward process.

While the discretization complexity K has been well-studied with an accurate enough score function
(Benton et al., 2023; Li et al., 2023a), there is a lack of analysis for the score-matching process.
Therefore, this work focuses on the score approximation and the optimization problem of the few-shot
score-matching objective function.

3.2
The Score Matching Objective Function

In this work, we specifically analyze few-shot diffusion models, which involve two datasets: (1) the
source dataset {Xs,i}ns
i=1; (2) the target dataset {Xta,i}nta
i=1. The approach involves first training a
pre-trained diffusion model on the source dataset and then freezing the backbone network to fine-tune
the diffusion models on the target dataset.

For data distributions, we assume that the source distribution qs and the target distribution qta are
both supported on a low-dimensional linear subspace. The low-dimensional structures have been
discovered on many popular image datasets (Pope et al., 2021; Gong et al., 2019; Tenenbaum et al.,
2000) due to the locally connected and symmetrical property, and it is crucial for diffusion models.
For image generation, current popular diffusion models, such as Stable Diffusion (Rombach et al.,
2022), transform images to a latent space and run diffusion models in the latent space. Except for the
image generation, Chen et al. (2024) recently show the latent dimension plays an important role in
diffusion models to work well in self-supervised learning, and linear subspace is enough.

We further assume that the source and target data share the same latent distribution. Note that this is a
common assumption in few-shot learning. In particular, previous theoretical works in the context of
supervised few-shot learning often assume that the source and target distributions have a common
latent representation (Du et al., 2020; Chua et al., 2021; Meunier et al., 2023).
Assumption 3.1. The source datapoints Xs and target datapoint Xta admit a low dimensional linear
structure and share the same latent distribution Xs = AsZ and Xta = AtaZ where As, Ata ∈RD×d
with orthonormal columns and Z ∼qz ∈Rd.

As mentioned in Chen et al. (2023c), when assuming linear distribution, the ground-truth score
function is decomposed into the latent score function ∇log qLD
t
(Z′) and linear encoder and decoder:

∇log qs
t (X) = As∇log qLD
t
 
A⊤
s X

−1

σ2
t

 
ID −AsA⊤
s

X ,

where qLD
t
(Z′) =
R
qt (Z′|Z) qz(Z)dZ and qt(·|Z) = N(mtZ, σ2
t Id). This form indicates that the
diffusion process happens in the latent subspace. A conceptual way to approximate the score function
is to minimize the following loss on a function class SNN:

min
s∈SNN

Z T

0
w(t)EXt∼qs
t ∥∇log qs
t (Xt) −s (Xt, t)∥2
2 dt ,

3Our analysis can be extended to f(Xt, t) = −1/2βtXt and g(t) = √βt, where {βt}t≥0 is non-decreasing
and bounded sequence.

4


---Page Break---
where w(t) is a weight function. However, the above objective function is intractable since ∇log qt(·)
is unknown. Vincent (2011) propose the following implementable loss:

Ls(s) =
Z T

0
w(t)EX0
h
EXt|X0 ∥∇log qs
t (Xt|X0) −s (Xt, t)∥2
2
i
dt .

Due to the forward process, ∇log qs
t (Xt|X0) has an analytical form and is equal to −(Xt −
mtX0)/σ2
t . Vincent (2011) also prove that this objective function only has a constant difference
compared to the above one. The empirical loss with the source datasets {Xs,i}ns
i=1 is defined by:

min
sV,θ∈SNN
bLs(sV,θ) =
1
ns(T −δ)

ns
X

i=1

Z T

δ
ℓs
t (Xs,i; sV,θ) dt ,
(1)

where

ℓs
t (Xs,i; s) = EXt|X0=Xs,i
h
∥∇log qs
t (Xt|X0) −s (Xt, t)∥2
2
i
,

and

SNN = {sV,θ(X, t) = 1

σ2
t
V fθ
 
V ⊤X, t

−1

σ2
t
X :V ∈RD×d with orthonormal columns,

fθ : Rd × [δ, T] →Rd a ReLU network } .

Note that we take w(t) = 1/(T −δ) for simplicity, where δ is the early stopping parameter to avoid
the blow-up phenomenon of score functions at the end of reverse process. Furthermore, we take the
integral over the forward time instead of discretizing the timeline since Xt is easy to generate.

The linear encoder and decoder structure and the shortcut connection in SNN is due to the form of
the ground-truth score function. The specific parameters for fθ, such as its length and width, are
identical to those used in Chen et al. (2023c). Generally, with a given network accuracy parameters ϵ,
the network size is Poly(1/ϵ). We show the parameter of neural network in Appendix A.

The diffusion models minimize the empirical loss to obtain a pre-trained approximated score function.
Let the minimizer of Equation (1) be (bVs, bθ). Chen et al. (2023c) show that (bVs, bθ) leads a n−2/d
s
approximation error bound. If trivially replacing ns with nta, we obtain a n−2/d
ta
bound for the
target dataset without the fine-tuning phase. Note that this bound suffers from the influence of the
latent dimension d, which is still large in popular datasets (Table 1). Hence, this results in a large
approximation error bound. In the next paragraph, we introduce the few-shot diffusion models with a
fine-tuning phase and show that the dependence on nta is n−1/2
ta
in the error bound (Theorem 4.3).

The Few shot Diffusion Models with a Fine-tuning Phase. Since the source and target distribution
share the same latent data distribution, we freeze ˆθ and only fine-tune the low-rank linear encoder and
decoder layer Vta. This method can significantly reduce the fine-tuning parameters and is similar to
LoRA (Hu et al., 2021), which also fine-tunes two low-rank matrices and is widely used in fine-tuning
the stable diffusion (Rombach et al., 2022).

Let ℓta
t be the loss function of the target dataset at time t, which has similar definition compared to ℓs
t.
The optimization problem for the target dataset is

min
sVta,ˆ
θ∈QNN(ˆθ)
bLta(sVta,ˆθ) =
1
nta(T −δ)

nta
X

i=1

Z T

δ
ℓta
t

Xta,i; sVta,ˆθ

dt ,

where

QNN(θ) = {sV,θ(X, t) = 1

σ2
t
V fθ
 
V ⊤X, t

−1

σ2
t
X : V ∈RD×d with orthonormal columns.} ,

Similarly, we define the minimizer of the few-shot objective function as (bVta, bθ).

Notations. We denote by ID the D-dimensional identity matrix. For X ∈RD and A ∈RD×d, we
denote by ∥X∥2 the Euclidean norm for vector and ∥A∥F the Frobenius norm for matrix. We denote
by ∥X∥2
L2(q) the expectation of X in L2 norm EX∼q[∥X∥2
2].

5


---Page Break---
Table 1: The requirement of nta in popular datasets. We use latent dimension in Pope et al. (2021).

Dataset
CIFAR-10
CIFAR-100
CelebA
MS-COCO
ImageNet
Dataset Size
6 × 104
6 × 104
2 × 105
3.3 × 105
1.2 × 106
Latent Dimension
25
22
24
37
43
The Requirement of nta
6
8
8
5
5

4
Few-shot Diffusion Models Enjoy Better Approximation Error Bound

In this section, we show that few-shot diffusion models with a fine-tuning phase escape the curse of
latent dimensionality and have a eO(n−2/d
s
+ n−1/2
ta
) approximation bound 4. This result makes the
first step to explain why few-shot models have great performance with a limited target dataset.

Before showing our results, we first introduce standard assumptions on the latent distribution and
the on-support ground-truth score function. We first assume that Z has a subgaussian tail and the
minimum eigenvalue of Z is lower bound by c0, also used in Chen et al. (2023c).
Assumption 4.1. qz > 0 is twice continuously differentiable, λmin(E

ZZ⊤
) ≥c0 and E∥Z∥2
2 ≤
CZ. Moreover, there exist positive constants B, C1, C2 such that when ∥Z∥2 ≥B, qz(Z) ≤
(2π)−d/2C1 exp
 
−C2∥Z∥2
2/2

.

Assumption 4.2. The on-support ground truth score As∇log qLD
t
(Z) and Ata∇log qLD
t
(Z) is
β-Lipschitz in Z ∈Rd for any t ∈[0, T].

Note that different from previous works directly assume ∇log qt(·) is Lipschitz (Chen et al., 2022,
2023d), the β-Lipschitz on-support score function assumption does not conflict with the blow-up
phenomenon when t goes to 0 due to the existence of (ID −AA⊤)X/σ2
t . With these assumptions,
we prove the approximation bound for few-shot models with a fine-tuning phase.

Theorem 4.3. Let α(n) = d log log n

log n
, F = (d+CZ)d2β2

δ2c0
and network parameter ϵ = n−1/2
ta
. Assume

Assumption 3.1, 4.1, 4.2 and n

d+5
4(1−α(ns))
ta
≥ns. Then, with probability 1−δ1, the following inequality
holds (hiding logarithmic factors)

1
T −δ

Z T

δ

sbVta,bθ(·, t) −∇log qta
t (·)

2

L2(qta
t ) dt ≤˜O
 (1 + β)2Dd3

δ (T −δ) √nta
+ Fn
−2−2α(ns)

d+5
s


log
 1

δ1


.

The dependence of δ is due to the blow-up property of the score function. Note that when ns is
sufficiently large, α(ns) is negligible. Then, the approximation error bound for few-shot diffusion
models is ˜O(n−2/d
s
+ n−1/2
ta
). Compared to the approximation error bound n−2/d
ta
without a few-shot
phase, it is clear that few-shot diffusion models escape the curse of the (latent) dimensionality.
Remark 4.4 (The discussion on the coefficient in Theorem 4.3). The goal of the fine-tuning phase
is to achieve the same order error bound compared with the pre-trained models, which means that
we consider the relative relationship between nta and ns. Hence, if the coefficient of nta and ns has
the same order, we can only consider 1/√nta and n−2/d
s
. To support the above augmentation, we
calculate the coefficient of ns and nta in detail. The dominated term of coefficient for nta and ns
are Dd3/δ and d3/(δ2c0), respectively. The classic choice for the early stopping parameter δ and
forward time T are 10−3 and 10, respectively (Karras et al., 2022). Then, with D = 256 × 256 × 3
as an example 5, Dd3/δ = d3 × 20 × 106 and d3/(δ2c0) = d3 × 106/c0, which has the same order.
Hence, we consider the relative relationship between 1/√nta and n−2/d
s
.

4.1
Discussion on the Approximation Bound

The relationship to empirical phenomenon. In applications, current few-shot diffusion models
only require 5 ∼10 target images to achieve great performance. Theorem 4.3 makes the first step to
explain why the few-shot diffusion models have great performance with a limited target nta. More

4Here, the approximation error means the score matching error with finite source and target datasets.
5Since smaller D is more friendly to nta, our discussion holds for all datasets in Table 1.

6


---Page Break---
specifically, with known source dataset size ns and the corresponding latent dimension d, we can

calculate the inequality n

d+5
4(1−α(ns))
ta
≥ns 6 to obtain the requirement of nta to achieve the same
accuracy compared to the pre-trained diffusion models. Combined with the latent dimension of
popular datasets (Pope et al., 2021), Table 1 shows the requirement of nta. It is clear that we only
need less than 10 target images to obtain an accurate enough few-shot diffusion model that matches
the performance in reality. The real-world experiments also support our discussion (Section 6).

Table 1 shows that the requirement of nta is heavily influenced by the latent dimension d. When
d is large (e.g. ImageNet), the approximation bound of pre-trained models is influenced by latent
dimension and has a large approximation error even with large-size source data. We only need a few
target data to achieve the same error in this setting. When d is small (e.g. CIFAR-10), pre-trained
models have a small approximation error. We need a slightly larger target data size.

The approximation error of the fully fine-tuned method. As shown in our real experiment
Section 6 and DreamBooth (Ruiz et al., 2023), when fine-tuning all parameters with a small target
dataset, models tend to overfit and lose the prior information from the pre-trained model. In our
theorem, this phenomenon means that in the fine-tuning phase, the model does not use bθ learned
by the pre-trained model and achieves a n−2/d
ta
approximation error bound, which suffers from the
curse of dimensionality. From an intuitive perspective, the probability density function (PDF) of a
distribution learned by an overfitting model is only positive at the interval around the target dataset,
which is far away from the PDF of true distribution and leads to a large error term. We also note
that it is possible to avoid this phenomenon by using a specific loss (Ruiz et al., 2023) or carefully
choosing the optimization epochs (Li et al., 2023b). We leave them as interesting future works.

Proof sketch. The first step is to prove that in QNN(bθ), there exists a solution ( ¯Vta, bθ) has the
following inequality (only focusing on ns and nta)

1
T −δ

Z T

δ

s ¯Vta,bθ(X, t) −∇log qta
t (X)

2

L2(qta
t ) dt ≤O

ϵ2 + n
−2−2α(ns)

d+5
s
log
 1

δ1


.

To prove the above inequality, we first do the following decomposition:

s ¯Vta,¯θ(·, t) −∇log qta
t (·)
2
2 +
s ¯Vta,bθ(·, t) −s ¯Vta,¯θ(·, t)

2

2 ,

where ( ¯Vta, ¯θ) ∈SNN is a constructed solution. The first term is due to the accuracy of the
constructive neural network with network accuracy parameter ϵ. For the second term, since the
latent score function is shared and few-shot diffusion models directly use bθ, it is control by the
approximation bound of the pre-trained diffusion models. Then, by using the inequality

inf
sVta,bθ∈Q(bθ)
bLta

sVta,bθ

≤bLta

s ¯Vta,bθ

,

we build the bridge between sbVta,bθ and s ¯Vta,bθ.

The second step is using the concentration to control the error between empirical bLta and expected
Lta. Roughly speaking, we have that

Lta

sbVta,bθ

−bLta

sbVta,bθ

≤
1
ntaϵ2 log

N

1/nta, QNN(bθ), ∥· ∥2

/δ1

,

where N(1/nta, QNN(bθ), ∥· ∥2) is the covering number of QNN(bθ) in L2 norm. Since only
V ∈RD×d can be optimized and bθ is fixed in QNN(bθ),

log

N

1/nta, QNN(bθ), ∥· ∥2

= eO(Dd log(1/nta)) .

Then, we balance different terms and achieve the final bound by choosing ϵ2 = 1/√nta.

6This also indicates the requirement of nta in Theorem 4.3 is easy to satisfy.

7


---Page Break---
5
The Few-shot Diffusion Model Have a Closed-form Minimizer

This section focuses on how to optimize the few-shot diffusion model. When considering the opti-
mization problem, we assume the shared latent distribution admits an isotropic Gaussian distribution
qz = N(0, λ2Id) with λ2 > 0, which indicates the score function has the following formulation:

∇log qta
t (X) = −1

λ2 AtaA⊤
taX −1

σ2
t

 
ID −AtaA⊤
ta

X .

Note that though qz = N(0, λ2Id) is a special case of Assumption 4.1, we still need to know λ2 and
Ata to generate samples, which indicates the previous optimization analysis for diffusion models
without a fine-tuning phase can not be used.

We fix a t ∈[δ, T] for the few-shot objective function since the matrix Ata is independent of time
t. More specifically, with an approximated latent distribution bqz = N(0, bΣ), where bΣ = bλ2Id, the
expected few-shot objective function at a fixed time t is

min
sVta, b
Σ∈˜
QNN(bΣ)
Lta,t(sVta,bΣ) = EXta∼qta
h
ℓta
t

Xta; sVta,bΣ
i
.

where
˜QNN(Σ) = {sV,θ(X, t) = 1

σ2
t
V fΣ
 
V ⊤X, t

−1

σ2
t
X : V ∈RD×d with rank(V ) = d.} ,

In this case, fbΣ(Z, t) = (Id −σ2
t bΣ−1
t )Z, where bΣt = m2
t bΣ + σ2
t Id. The constraint rank(V ) = d
is used to guarantee that the few-shot diffusion models learn meaningful subspace instead of 0D×d.
Note that rank(V ) = d is a weaker constraint than V ⊤V = Id since the pre-trained diffusion
model has already learned the length information. This weaker constraint means we need less prior
knowledge compared to Q(θ), which is more user-friendly. Let eVta be a minimizer of the above
expected few-shot objective function. We show that eVta has a closed form and good property.
Lemma 5.1. Assume Assumption 3.1 and qz = N(0, λ2Id). Let C = EXta∼qta 
XtaX⊤
ta

be the
expected data covariance matrix. Then, eVta has a closed form:

eVta eV ⊤
ta = m2
t bλ2 + σ2
t
bλ2
 
C + σ2
t ID
−1 C .

Lemma 5.1 indicates that few-shot diffusion models can directly obtain an approximation of the
minimizer without a complex optimization process. Furthermore, this minimizer has good properties
and exactly recovers the subspace spanned by Ata. More specifically, the expected minimizer
indicates ∥eVta eV ⊤
ta −AtaA⊤
ta∥2
F = 0 when ns and nta are infinite. However, the source datasets ns
and target datasets nta are finite, we analyze the empirical closed-form solution

¯eV ta ¯eV
⊤

ta = m2
t bλ2 + σ2
t
bλ2
(m2
t ¯C + σ2
t ID)−1 ¯C ,

where ¯C =
1
nta
Pnta
i=1 Xta,iX⊤
ta,i is the empirical covariance matrix.

Theorem 5.2. Assume Assumption 3.1 and qz = N(0, λ2Id). Let bqz be the latent distribution

generated by the pre-trained models with (bVta, bΣ) and M =
d2β2(d+λ2)

λ
p

Dd log (Ddns) (d2 ∨D).
Then, with probability 1 −δ1, we have that for any t ∈[δ, T]

¯eV ta ¯eV
⊤

ta −AtaA⊤
ta



2

F
≤eO

 
d log( 1

δ1 )

m2
tλ2 + σ2
t


M
dδ√ns
+ d

nta
(m2
tλ2 + σ2
t )2
!

.

The above result indicates that the few-shot diffusion models can still recover the true subspace with
finite ns and nta. Note that when the latent distribution is Gaussian distribution, the approximation
error bound for the source dataset is n−1/2
s
instead of n−2/d
s
(Yuan et al., 2023). Hence, ns in
Theorem 5.2 do not depend on latent dimension d.
Remark 5.3. The bound of ∥V V T −AAT ∥2
F only guarantees the subspace spanned by V and A
is close, which still holds after an orthogonal transformation on V . Hence, this bound does not
indicate ∥V −A∥2
F is small. Since all previous works (Chen et al., 2023c; Yuan et al., 2023) consider
∥V V T −AAT ∥2
F , we also use this metric to measure the subspace recovery. However, our results are
stronger due to the closed-form solution, where previous works do not consider how to obtain V V T .

8


---Page Break---
Figure 1: The experiments on CelebA64 dataset

5.1
Discussion on the Closed-form Minimizer

Better dependence on nta, δ and d. Note that Theorem 5.2 has better 1/nta dependence on the
target dataset compared to 1/√ns dependence on the source dataset. Furthermore, the coefficient of
ns term is dependent on the early stopping parameter δ and D. This is due to the δ and D dependence
of the approximation bound, which is used in generating bqz. However, the nta term only has d
dependence. Hence, even in the latent Gaussian setting, we still need a larger source dataset than the
target dataset to obtain a sufficiently accurate closed-form solution.

The relationship with principal component analysis (PCA). The expected few-shot score matching
objective can be simplified to

min
Vta∈˜
QNN(bΣ)
1/σ4
t EXt|X0=Xta,i
h
∥Vta bGtV ⊤
ta Xt −mtX0∥2
2
i
,

where bGt = Id −σ2
t bΣ−1
t . Note that when ignoring 1/σ4
t and choosing t = 0, the above minimization
problem is similar to PCA. This suggests that few-shot models implicitly optimize an objective
function akin to PCA. However, few-shot models extend beyond traditional PCA. More specifically,
when λ2 is large, classical PCA suffers from the influence of λ2. In contrast, due to (m2
tλ2 +σ2
t )/nta
term, few-shot models can select a large t to mitigate the impact of λ2 and achieve a 1/nta.

6
Experiments

To corroborate our theoretical findings, we conducted experiments utilizing real-world datasets.
These experiments show that the new model obtained by only fine-tuning appropriate encoder and
decoder layers on target datasets can produce novel images with the target feature, which shows the
effectiveness of the methods and supports our theoretical results.

Datasets and benchmark. Note that human face images tend to exhibit similarity in their latent
space, primarily due to shared facial features, while differing in specific features. Hence, we initially
pre-train a model using the CelebA64 dataset, focusing on distinct hairstyle features as the goal for
the fine-tuning phase. For the source data, we construct a large dataset (6400 images) with different
hairstyles (without the bald feature). For the target data, we choose the bald feature as the target
feature and select 10 images with this feature to constitute the target dataset, which are much smaller
than the size of target dataset (Figure 1 (a)). To show the effectiveness of our methods, we also
fine-tune all parameters of the pre-trained models as the benchmark.

Discussion. As shown in Figure 1, the results obtained by only fine-tuning the encoder and decoder
layers can generate novel face images with the bald feature. Conversely, when fine-tuning all
parameters, the models suffer from memory phenomenon and can only generate images that slightly
modify the brightness and angle of the target dataset. This phenomenon indicates that only fine-tuning
the appropriate encoder and decoder will result in a model with a generalization property.

We note that these experiments aim to verify the effectiveness of the methods instead of achieving
state-of-the-art performance since previous works carefully select specific parameters, such as specific
cross-attention layers (Kumari et al., 2023) or special neurons (Liu et al., 2023), to fine-tune pre-
trained models. However, we simply fine-tune all encoder and decoder layers simultaneously. There
are more experiments on cat faces and more discussion on why Assumption 3.1 is satisfied in our
experiments. We refer to Appendix E for more details.

9


---Page Break---
7
Conclusion

This work aims to provide a deeper understanding of few-shot diffusion models from a theoretical
perspective. Our analysis is conducted from two key perspectives: the approximation and optimization
aspects, all under linear structure distribution and shared latent space assumptions.

From the approximation error bound, we consider general subgaussian latent variable and prove
that few-shot models have a eO

n−2/d
s
+ n−1/2
ta

approximation bound, which is better than n−2/d
ta
results of diffusion models without a fine-tuning phase and escape the curse of dimensionality. This
result also makes the first step to explain why few-shot diffusion models only require 5 ∼10 images
to generate great samples. The experiments on the real-world dataset also show that the fine-tuning
phase only requires 10 images to generate novel images with the target feature.

When analyzing the optimization process, we consider a more special, shared Gaussian latent variable
and prove that the expected score matching has a closed-form minimizer, which indicates that the
few-shot diffusion models can simplify the optimization problem. Furthermore, we prove that the
empirical closed-form solution has a eO
 
1/nta + 1/
 
δ√ns

accuracy bound, which still has better
1/nta target data dependence compared to 1/
 
δ√ns

dependence on the source data.

Future work and limitation. When considering the approximation bound, we assume a distribution
with a linear structure. Though it has been supported by much empirical evidence, it is not as general
as bounded distribution. After that, we plan to consider a general, bounded distribution and show
the advantage of few-shot diffusion models. One possible way is to analyze the mixture of low-rank
Gaussian (Wang et al., 2024), which is more general than the linear subspace assumption.

We focus on a special Gaussian latent distribution when considering the optimization problem. As a
next step, we plan to consider a more general latent distribution, such as a log-concave distribution.
In this setting, we can not directly obtain the closed-form solution. However, due to the shared
information and simplified landscape, it is still possible to use some optimization algorithms, such as
gradient descent, to optimize the few-shot objective function to achieve the convergence guarantee.

Broader Impact. This paper presents work whose goal is to understand few-shot diffusion models
from the theoretical perspective. A noteworthy societal impact is that few-shot diffusion models may
be used to imitate the style of artists and generate fake images, thereby infringing on the rights of
artists (Mirsky & Lee, 2021). We recommend adding watermarks to images to determine whether the
image was generated by a generative model (Fernandez et al., 2023). The other societal impact is the
same as general generative models (Mishkin et al., 2022).

Acknowledgments and Disclosure of Funding

The author Bo Jiang is supported by National Natural Science Foundation of China (62072302).

References

Benton, J., De Bortoli, V., Doucet, A., and Deligiannidis, G. Linear convergence bounds for diffusion
models via stochastic localization. arXiv preprint arXiv:2308.03686, 2023.

Bruno, S., Zhang, Y., Lim, D.-Y., Akyildiz, Ö. D., and Sabanis, S. On diffusion-based generative
models and their error bounds: The log-concave case with full convergence estimates. arXiv
preprint arXiv:2311.13584, 2023.

Chen, H., Lee, H., and Lu, J. Improved analysis of score-based generative modeling: User-friendly
bounds under minimal smoothness assumptions. In International Conference on Machine Learning,
pp. 4735–4763. PMLR, 2023a.

Chen, H., Wang, X., Zeng, G., Zhang, Y., Zhou, Y., Han, F., and Zhu, W. Videodreamer: Customized
multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990,
2023b.

Chen, M., Huang, K., Zhao, T., and Wang, M. Score approximation, estimation and distribution
recovery of diffusion models on low-dimensional data. arXiv preprint arXiv:2302.07194, 2023c.

10


---Page Break---
Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang, A. R. Sampling is as easy as learning the score:
theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215,
2022.

Chen, S., Daras, G., and Dimakis, A. G. Restoration-degradation beyond linear diffusions: A
non-asymptotic analysis for ddim-type samplers. arXiv preprint arXiv:2303.03384, 2023d.

Chen, X., Liu, Z., Xie, S., and He, K. Deconstructing denoising diffusion models for self-supervised
learning, 2024.

Chua, K., Lei, Q., and Lee, J. D. How fine-tuning allows for effective meta-learning. Advances in
Neural Information Processing Systems, 34:8871–8884, 2021.

Cui, H., Krzakala, F., Vanden-Eijnden, E., and Zdeborová, L. Analysis of learning a flow-based
generative model from limited sample complexity. arXiv preprint arXiv:2310.03575, 2023.

Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., and Lei, Q. Few-shot learning via learning the
representation, provably. arXiv preprint arXiv:2002.09434, 2020.

Dutt, R., Ericsson, L., Sanchez, P., Tsaftaris, S. A., and Hospedales, T. Parameter-efficient fine-tuning
for medical image analysis: The missed opportunity. arXiv preprint arXiv:2305.08252, 2023.

Fernandez, P., Couairon, G., Jégou, H., Douze, M., and Furon, T. The stable signature: Rooting
watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pp. 22466–22477, 2023.

Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A. H., Chechik, G., and Cohen-Or, D. An
image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv
preprint arXiv:2208.01618, 2022.

Gong, S., Boddeti, V. N., and Jain, A. K. On the intrinsic dimensionality of image representations.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
3987–3996, 2019.

Han, L., Li, Y., Zhang, H., Milanfar, P., Metaxas, D., and Yang, F. Svdiff: Compact parameter space
for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023.

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. Video diffusion models.
arXiv preprint arXiv:2204.03458, 2022.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora:
Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial
networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 4401–4410, 2019.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based
generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.

Kumari, N., Zhang, B., Zhang, R., Shechtman, E., and Zhu, J.-Y. Multi-concept customization of
text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 1931–1941, 2023.

Li, G., Wei, Y., Chen, Y., and Chi, Y. Towards faster non-asymptotic convergence for diffusion-based
generative models. arXiv preprint arXiv:2306.09251, 2023a.

Li, H., Shi, H., Zhang, W., Wu, W., Liao, Y., Wang, L., Lee, L.-h., and Zhou, P. Dreamscene:
3d gaussian-based text-to-3d scene generation via formation pattern sampling. arXiv preprint
arXiv:2404.03575, 2024.

Li, P., Li, Z., Zhang, H., and Bian, J. On the generalization properties of diffusion models. arXiv
preprint arXiv:2311.01797, 2023b.

11


---Page Break---
Liu, Z., Feng, R., Zhu, K., Zhang, Y., Zheng, K., Liu, Y., Zhao, D., Zhou, J., and Cao, Y. Cones:
Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125,
2023.

Mei, S. and Wu, Y. Deep networks as denoising algorithms: Sample-efficient learning of diffusion
models in high-dimensional graphical models. arXiv preprint arXiv:2309.11420, 2023.

Meunier, D., Li, Z., Gretton, A., and Kpotufe, S. Nonlinear meta-learning can guarantee faster rates.
arXiv preprint arXiv:2307.10870, 2023.

Mirsky, Y. and Lee, W. The creation and detection of deepfakes: A survey. ACM computing surveys
(CSUR), 54(1):1–41, 2021.

Mishkin, P., Ahmad, L., Brundage, M., Krueger, G., and Sastry, G. Dall· e 2 preview-risks and
limitations. Noudettu, 28:2022, 2022.

Moon, T., Choi, M., Lee, G., Ha, J.-W., and Lee, J. Fine-tuning diffusion models with limited data.
In NeurIPS 2022 Workshop on Score-Based Methods, 2022.

Oko, K., Akiyama, S., and Suzuki, T. Diffusion models are minimax optimal distribution estimators.
arXiv preprint arXiv:2303.01861, 2023.

Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., and Goldstein, T. The intrinsic dimension of
images and its impact on learning. arXiv preprint arXiv:2104.08894, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis
with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 10684–10695, 2022.

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. Dreambooth: Fine tuning
text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 22500–22510, 2023.

Shah, K., Chen, S., and Klivans, A. Learning mixtures of gaussians using the ddpm objective. arXiv
preprint arXiv:2307.01178, 2023.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based
generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456,
2020.

Tenenbaum, J. B., Silva, V. d., and Langford, J. C. A global geometric framework for nonlinear
dimensionality reduction. science, 290(5500):2319–2323, 2000.

Vincent, P. A connection between score matching and denoising autoencoders. Neural computation,
23(7):1661–1674, 2011.

Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., and Qu, Q. Diffusion models learn low-dimensional
distributions via subspace clustering. arXiv preprint arXiv:2409.02426, 2024.

Xiang, C., Bao, F., Li, C., Su, H., and Zhu, J. A closer look at parameter-efficient tuning in diffusion
models. arXiv preprint arXiv:2303.18181, 2023.

Yuan, H., Huang, K., Ni, C., Chen, M., and Wang, M. Reward-directed conditional diffusion:
Provable distribution estimation and reward improvement. arXiv preprint arXiv:2307.07055, 2023.

Zhang, Q. and Chen, Y. Fast sampling of diffusion models with exponential integrator. arXiv preprint
arXiv:2204.13902, 2022.

Zhu, J., Ma, H., Chen, J., and Yuan, J. Domainstudio: Fine-tuning diffusion models for domain-driven
image generation using limited data. arXiv preprint arXiv:2306.14153, 2023.

12


---Page Break---
Appendix

A
The Neural Network Structure

In this section, we introduce the multi-layer ReLU network fθ ∈NN (L, M ′, J, K1, κ, γ, γt) in SNN.
We note that the following setting is exactly the same as the one in Chen et al. (2023c), and we
show the structure for completeness. We denote by NN (L, M ′, J, K1, κ, γ, γt) the following neural
network:

NN (L, M ′, J, K1, κ, γ, γt) = {f(Z, t) = WLσ

. . . σ

W1

Z⊤, t
⊤+ b1

. . .

+ bL :

network width bounded by M ′, sup
z,t ∥f(z, t)∥2 ≤K1,

max {∥bi∥∞, ∥Wi∥∞} ≤κ for i = 1, . . . , L,

L
X

i=1
(∥Wi∥0 + ∥bi∥0) ≤J,

∥f (Z1, t) −f (Z2, t)∥2 ≤γ ∥Z1 −Z2∥2 for any t ∈[0, T],
∥f (Z, t1) −f (Z, t2)∥2 ≤γt |t1 −t2| for any Z} ,

where σ is the ReLU activation function. Given an network accuracy ϵ > 0, the parameters is defined
by

L = O

log 1

ϵ + d

, K1 = O

2d2 log
 d

δϵ


,

M ′ = O

(1 + β)dTτdd/2+1ϵ−(d+1) logd/2
 d

δϵ


,

J = O

(1 + β)dTτdd/2+1ϵ−(d+1) logd/2
 d

δϵ

 
log 1

ϵ + d

,

κ = O

 

max

(

2(1 + β)

s

d log
 d

δϵ


, Tτ

)!

,

γ = 10d(1 + β), γt = 10τ ,

where τ = supA∈{As,Ata} supt∈[δ,T ] sup∥z∥∞≤√

d log
d
δϵ

 ∂

∂t

σtA∇log qLD
t
(z)

2.

B
The Proof of the Approximation Error Bound

Let (bVs, bθ) be the minimizer of the pre-trained objective function. The few-shot diffusion model
freezes the bottleneck network and fine-tunes Vta ∈RD×d to obtain the minimizer (bVta, bθ) of the
few-shot objective function. As the first step, we show that with the bottleneck parameterized by bθ,
there also exists a solution ( ¯Vta, bθ) ∈QNN(bθ) achieve the ϵ2 + n−2/d
s
error bound.

Lemma B.1. If ϵ ≤n
−1−α(ns)

d+5
s
, where α(n) = d log log n

log n
, then with probability 1 −δ1, there exists a

solution ( ¯Vta, ˆθ) ∈QNN(ˆθ) such that
Z T

δ
EX∼qta
t

s ¯Vta,ˆθ(X, t) −∇log qta
t (X)

2

2


dt ≤O
d

δ ϵ2 + (T −δ)(d + CZ)d2β2

δ2c0
n
−2−2α(ns)

d+5
s
log( 1

δ1
)

.

Proof. As shown in Theorem 1 of Chen et al. (2023c), there exists a solution ( ¯Vs, ¯θ) in SNN such
that

s ¯Vs,¯θ(·, t) −∇log qs
t (·)

2 ≤

√

d + 1

σ2
t
ϵ , ∀t ∈[δ, T] .

13


---Page Break---
Hence, we do the following decomposition:
s ¯Vta,ˆθ(·, t) −∇log qta
t (·)

2

2 ≲
s ¯Vta,¯θ(·, t) −∇log qta
t (·)
2
2 +
s ¯Vta,ˆθ(·, t) −s ¯Vta,¯θ(·, t)

2

2

For the encoder and decoder layer, we choose ¯Vta = Ata. The first term is bounded due to the
construction of the neural network. We first show that s ¯Vta,¯θ is ϵ-close to the true score function
∇log qta
t . Since the encoder and decoder have been chosen, we only need to focus on the latent
bottleneck. For the latent bottleneck, we need to use fθ(Z, t) to approximate ground-truth function
h(Z, t) = σ2
t ∇log qLD
t
(Z) + Z for Z ∈Rd. Chen et al. (2023c) show that for any latent variable
Z′ ∈Rd with subgaussian tail, we have that

∥h (Z′, t) −f¯θ (Z′, t)∥L2(qLD
t ) ≤(
√

d + 1)ϵ .

Then, we have that
s ¯Vta,¯θ(·, t) −∇log qta
t (·)
2
2 ≤d

σ4
t
ϵ2 .

For the second term, we know that with probability 1 −δ1:
Z T

δ
EX∼qt

s ¯Vta,¯θ(·, t) −s ¯Vta,ˆθ(·, t)

2

2



≤
Z T

δ

1
σ4
t
EZ∼qLD
t

∥fˆθ(Z) −f¯θ(Z)∥2
2

dt

≤O
T −δ

δ2
( δ

c0

 
(T −log δ) d · γ2 + dβ

+ γ2 · CZ

c0
)n
−2−2α(ns)

d+5
s


log
 1

δ1


,

where α(n) = d log log n

log n
. The first inequality follows Ata is a matrix with orthonormal columns.

Since we assume ϵ ≤n
−1−α(ns)

d+5
s
, the network has good enough ability to obtain an accurate enough
ˆθ. Hence, we can use Appendix C.4 of Chen et al. (2023c) to obtain the second inequality. Since we
directly use the true matrix ¯Vta = Ata instead of the approximate ˆVta, we do not need orthogonal
transformation and can choose U = Id in the Appendix C.4 of Chen et al. (2023c). Then, we
complete our proof.
■

To prove Theorem 4.3, we need to do the following decomposition for the population loss of the
target datasets

Lta

sbVta,ˆθ


= Lta

sbVta,ˆθ

−(1 + a) bLta

sbVta,ˆθ

+ (1 + a) bLta

sbVta,ˆθ


≤Ltrunc
ta

sbVta,ˆθ

−(1 + a) bLtrunc
ta

sbVta,ˆθ


|
{z
}
(a)

+ Lta

sbVta,ˆθ

−Ltrunc
ta

sbVta,ˆθ


|
{z
}
(b)

+(1 + a)
inf
sVta,ˆ
θ∈Q(ˆθ)
bLta

sVta,ˆθ


|
{z
}
(c)

,

where a ∈(0, 1) and Ltrunc
ta
is defined as

Ltrunc
ta

sbVta,ˆθ

= Ex∼q0
h
ℓtrunc
ta

x; sbVta,ˆθ
i
= Ex∼q0
h
ℓta

x; sbVta,ˆθ

1 {∥x∥2 ≤R} dt
i
.

In
this
section,
we
take
R
=
O
q

d log d + log K1 + log nta

δ1


to
guarantee

PXta,i∼qta  
∥Xta,i∥2 ≤R for all i = 1, . . . , nta) ≥1 −δ1, where K1 is defined in Appendix A.

Term
(a).
Similar
to
Chen
et
al.
(2023c),
we
define
a
function
class
G(ˆθ)
=
n
ℓtrunc
ta

·; sV,ˆθ

: sV,ˆθ ∈QNN(ˆθ)
o
, which is induced by Q(bθ). For the upper bound of G(ˆθ), we
directly use the augmentation of Chen et al. (2023c) to obtain that

ℓtrunc
ta

X; sV,ˆθ

≤O
K2
1 + R2

δ (T −δ)


, for any sV,ˆθ ∈QNN(ˆθ) .

14


---Page Break---
Then, by using Lemma D.1, we know that with probability 1 −δ1, term (a) is bounded by

O



(1 + 3/a)
 
(1 + β)2d2 log d

δϵ + log nta

δ


ntaδ (T −δ)
log
N

τ1, G(ˆθ), ∥· ∥∞


δ1
+ τ1



.

To bound the above term, we need to calculate the covering number of G(ˆθ), which
is related to a
ι-covering of
QNN(ˆθ).
Suppose that given
sV1,ˆθ
and
sV2,ˆθ
with

sup∥x∥2≤3R+√D log D,t∈[δ,T ]
sV1,ˆθ(x, t) −sV2,ˆθ(x, t)

2 ≤ι, we need to bound
ℓtrunc
ta

·; sV1,ˆθ

−ℓtrunc
ta

·; sV2,ˆθ

∞.

By using the same calculation compared to Term (A) of Chen et al. (2023c), we know that
ℓtrunc
ta

·; sV1,ˆθ

−ℓtrunc
ta

·; sV2,ˆθ

∞≤O

ι
T −δ (K1 + R) log T

δ + 4K1(K1 + R)

δ (T −δ)
(R/D)D−2 exp

−1

σ2
t
R2

.

The above inequality indicates that a ι-covering of QNN(ˆθ) in L2 norm leads a
ι
T −δ(K1+R) log T

δ +

4K1(K1+R)

δ(T −δ)
(R/D)D−2 exp

−1

σ2
t R2
-covering of of G(ˆθ) in L∞norm.

By taking R = O
q

d log d + log K1 + log nta

δ1


, K1 = O
 
2d2 log
  d

δϵ

, ι =
2
ntaδ(T −δ), we
know that

τ1 ≤
d2

ntaδ log(T

δ ) log( d

δϵ) log(nta

δ ) ,

which indicates with probability 1 −δ1, term (a) is bounded by

O



(1 + 3/a)
 
(1 + β)2d2 log d

δϵ + log nta

δ


ntaδ (T −δ)
log
N

1
ntaδ(T −δ), QNN(ˆθ), ∥· ∥2


δ1
+ d2

ntaδ log(T

δ ) log( d

δϵ) log(nta

δ )



.

After that, we need to determine the covering number of QNN(ˆθ) with a truncated X to bound term
(a).
Lemma B.2. The logarithmic covering number of QNN(θ) for ∥X∥2 ≤3R + √D log D, t ∈[δ, T]
is

log N (ι, QNN(θ), ∥· ∥2) = O

 

2Dd · log

 

1 + 6Kγ
√

d(3R + √D log D)

δι

!!

.

Proof. Suppose that there exists two orthonormal column matrix V1, V2 such that ∥V1 −V2∥F ≤δ2,
then we have

sup
∥X∥2≤3R+√D log D,t∈[δ,T ]
∥sV1,θ(X, t) −sV2,θ(X, t)∥2

= 1

σ2
t
sup
∥X∥2≤3R+√D log D,t∈[δ,T ]

V1fθ
 
V ⊤
1 X, t

−V1fθ
 
V ⊤
2 X, t

2 +
V1fθ
 
V ⊤
2 x, t

−V2fθ
 
V ⊤
2 X, t

2


≤1

σ2
t


γδ1
√

d(3R +
p

D log D) + δ1K


For set

V ∈RD×d : ∥V ∥2 ≤1
	
, the δ2-covering number is

1 + 2
√

d
δ2

Dd
. Then we know that

log N (ι, SNN, ∥· ∥2) = O

 

2Dd · log

 

1 + 6Kγ
√

d(3R + √D log D)

δι

!!

.

■

15


---Page Break---
Term (b).
For the term (b), the proof of Theorem 2 in Chen et al. (2023c) shows that

Term (b) ≤
1
ntaδ (T −δ) .

Term (c).
For the term (c), we know that it is bounded by the constructed solution ( ¯Vta, ˆθ):

inf
sVta,ˆo∈Q(ˆθ)
bLta

sVta,ˆθ

≤bLta

s ¯Vta,ˆθ

−(1 + a)Ltrunc 
s ¯Vta,ˆθ


|
{z
}
(C1)

+(1 + a) Ltrunc 
s ¯Vta,ˆθ


|
{z
}
(C2)

.

For the term (C.1), since s ¯Vta,ˆθ is a fixed function, we directly use the results of (Chen et al., 2023c):

Term(C1) ≤O

 
(1 + 6/a)
 
(1 + β)2d2 log d

δϵ + log n

δ


ntaδ (T −δ)
log 1

δ1

!

,

with probability 1 −δ1. For the term (C.2), we know that

Ltrunc
ta

s ¯Vta,ˆθ

≤L

s ¯Vta,ˆθ

=
1
T −δ

Z T

δ

s ¯Vta,ˆθ(·, t) −∇log qta
t (·)

2

L2(qt) dt

+ L

s ¯Vta,ˆθ

−
1
T −δ

Z T

δ

s ¯Vta,ˆθ(·, t) −∇log qta
t (·)

2

L2(qt) dt
|
{z
}
(E)

.

As we show in Section 3.2, the two terms in E are both the score matching objective function and
have a constant different E, which is independent of the trainable parameters (V, θ). We denote by
this difference E and F = (d+CZ)d2β2

δ2c0
. With probability 1 −δ1, Lemma B.1 shows that term (C.2)
is bounded by

O

d
δ(T −δ)ϵ2 + Fn
−2−2α(ns)

d+5
s
log
 1

δ1


+ E .

After bounding these three terms, we prove Theorem 4.3.

Theorem 4.3. Let α(n) = d log log n

log n
, F = (d+CZ)d2β2

δ2c0
and network parameter ϵ = n−1/2
ta
. Assume

Assumption 3.1, 4.1, 4.2 and n

d+5
4(1−α(ns))
ta
≥ns. Then, with probability 1−δ1, the following inequality
holds (hiding logarithmic factors)

1
T −δ

Z T

δ

sbVta,bθ(·, t) −∇log qta
t (·)

2

L2(qta
t ) dt ≤˜O
 (1 + β)2Dd3

δ (T −δ) √nta
+ Fn
−2−2α(ns)

d+5
s


log
 1

δ1


.

Proof. Equipped with the bound of the term (a), (b), and (c) and hiding the logarithmic term (except
for the covering number term), with probability 1 −δ1, we have that

Lta

s ˆVta,ˆθ

≤(1 + a)2E + ˜O

  
(1 + β)2d2 log d

δϵ + log nta

δ


aδ (T −δ) nta
log
N

1
ntaδ(T −δ), QNN(ˆθ), ∥· ∥2


δ1
+ d2

ntaδ

+
1
ntaδ(T −δ) +
d
δ(T −δ)ϵ2 + Fn
−2−2α(ns)

d+5
s
log
 1

δ1

 !

.

16


---Page Break---
Since Lta

s ˆVta,ˆθ

−E =
1
T −δ
R T
δ

sbVta,ˆθ(·, t) −∇log qta
t (·)

2

L2(qt), we have that the following

inequality when choosing a = ϵ2:

1
T −δ

Z T

δ

sbVta,ˆθ(·, t) −∇log qta
t (·)

2

L2(qt)

≤˜O




(1 + β)2d2

ϵ2δ (T −δ) nta
log
N

1
ntaδ(T −δ), QNN(ˆθ), ∥· ∥2


δ1
+
d
δ(T −δ)ϵ2 + d2

ntaδ + Fn
−2−2α(ns)

d+5
s
log
 1

δ1





≤˜O

 
(1 + β)2Dd3

δ (T −δ) ntaϵ2 log

 

1 + 6Kγ
√

d(3R + √D log D)ntaδ(T −δ)

δ1

!

+
d
δ(T −δ)ϵ2 + d2

ntaδ

+ Fn
−2−2α(ns)

d+5
s
log
 1

δ1

 !

,

where the second inequality follows the convering number of QNN(ˆθ) for ∥X∥≤3R + √D log D

with R = O
q

d log d + log K1 + log nta

δ1


and the network parameters is defined in Appendix A.

Finally, we choose ϵ2 = 1/√nta, then we have that

1
T −δ

Z T

δ

sbVta,ˆθ(·, t) −∇log qta
t (·)

2

L2(qt) ≤˜O
 (1 + β)2Dd3

δ (T −δ) √nta
log
 1

δ1


+ d2

ntaδ + Fn
−2−2α(ns)

d+5
s
log
 1

δ1


.

As we require in Lemma B.1, we need ϵ = 1/n1/4
ta
≤n
−1−α(ns)

d+5
s
, which indicates n

d+5
4(1−α(ns))
ta
≥
ns.
■

C
The Proof of the Optimization Problem

C.1
The Pre-trained Diffusion Model Generate Accurate Enough Latent Distribution

Since we need to use the approximated latent distribution in the few-shot phase, we show that the
pre-trained diffusion models with solution (bVs, bθ) can generate an accurate enough latent distribution.
As shown in Section 5, when considering the optimization perspective of diffusion models, we assume
the latent distribution is a Gaussian distribution qz = N(0, Σ) with Σ = diag
 
λ2
1, . . . , λ2
d

≻0.
Yuan et al. (2023) show that in the setting, the approximation error bound (Lemma D.3) for the target
dataset is

1
T −δ

Z T

δ

∇log qs
t (·) −sbVs,bθ (·, t)

2

L2(qs
t ) dt ≤O



1

δ

s

(d2 + Dd) log (Ddns) (d2 ∨D) log 1

δ1
ns



.

To generate latent distribution, we first introduce the reverse process in the latent space. The
introduction mainly follows the outline of Appendix C.2 of Chen et al. (2023c). For Xt, we can
do the following decomposition: Xt = AsZt + Xt,⊥, where Zt = A⊤
s Xt. With Z←
t
= ZT −t, the
reverse process in the latent space is

dZ←
t
=
1

2Z←
t
+ ∇log qLD
T −t (Z←
t )

dt + d
 
A⊤
s Bt


As shown in Theorem 3 of Chen et al. (2023c), the solution (bVs, bθ) of the pre-trained diffusion models
only guarantee ∥bVs bV ⊤
s −AsA⊤
s ∥2
F is small instead of ∥bVs −As∥2
F is small. Hence, Theorem 3
of Chen et al. (2023c) assume there exists an orthogonal matrix Us ∈Rd×d and do an orthogonal
transformation on bVs to obtain bVsUs, which can guarantee ∥bVsUs −As∥2
F is small. After such orthog-
onal transformation, the reverse process with an approximated score function and an approximated
reversing beginning distribution eZ←,r
0
∼N (0, Id) is

d eZ←,r
t
=
1

2
eZ←,r
t
+ sLD
Us,bθ


eZ←,r
t
, T −t

dt + d

U ⊤
s bV ⊤
s Bt

, eZ←,r
0
∼N (0, Id)
(2)

17


---Page Break---
where

eZ←,r
t
= U ⊤
s eZ←
t and sLD
Us,bθ(Z, t) = 1

σ2
t


−Z + U ⊤
s fbθ(UsZ, t)

.

Then, we discretize the above process with the exponential integrator (EI) discretization scheme
(Zhang & Chen, 2022) to obtain an implementable algorithm:

d eZ⇐,r
t
=
1

2
eZ⇐,r
t
+ sLD
Us,bθ


eZ⇐,r
kη , T −t′
k

dt + d

U ⊤
s bV ⊤
s Bt

, where t ∈[t′
k, t′
k+1] .
(3)

As shown in Appendix C.4 of Chen et al. (2023c), if the target ground truth score function has a
L2-accurate approximated score:

1
T −δ

Z T

δ

sbVs,ˆθ(·, t) −∇log qs
t (·)

2

L2(qs
t ) dt ≤ϵ2 ,

the latent score function also has an L2 norm bound ϵ2
latent-score, which is determined by ϵ:

ϵlatent-score = ϵ · O
 δ

c0

 
(T −log δ) d · γ2 + dβ

+ γ2 · CZ

c0


.

The remaining term is to determined ϵ. Since we assume Gaussian latent variable instead of sub-
Gaussian one. Hence, we do not use ϵ in Chen et al. (2023c). We use ϵ in Yuan et al. (2023) (Theorem
4.5 of Yuan et al. (2023)), which also considers Gaussian latent variable, to achieve the final results.

Finally, we have that with probability 1 −δ1:

1
T −δ

Z T

δ

∇log qLD
t
(·) −sLD
Us,bθ (·, t)

2

L2(qLD
t
) dt

≤O



d2β2  
d + λ2
max


λminδ

s

(d2 + Dd) log (Ddns) (d2 ∨D) log 1

δ1
ns



≜ϵ2
latent-score .
(4)

Let pLD
t
be the distribution of the algorithm (the above discretization process). In the following
lemma, we adopt Theorem 5 of Chen et al. (2023a) and show that the pre-trained diffusion model can
obtain an accurate enough latent distribution with the above L2-accurate latent score function.

Lemma C.1. With ϵ2
latent-score defined in Equation (4), T
=
log

λ2
max+d
ϵ2
latent-score


and K
=

Θ

d2(T +log(λ2
max))2

ϵ2
latent-score


, by using the exponentially decreasing (then the constant) stepsize hk =

c min
n
max
n
tk,
1
λ2max

o
, 1
o
, c = log(λ2
max)+T
K
, the results pLD
T
of sampling algorithm (Equation (3))
has the following guarantee with probability 1 −δ1 (hiding the logarithmic factor):

KL
 
q0∥ˆpLD
T

≤eO(ϵ2
latent-score)

= eO



d2β2  
d + λ2
max


λminδ

s

(d2 + Dd) log (Ddns) (d2 ∨D) log 1

δ1
ns





Proof. The Theorem 5 of Chen et al. (2023a) show that if ∇log qLD
0
is L-Lipschitz, diffusion models
with a L2-accurate can generate bpLD
T , which is close to q0 in KL divergence. Since qz = N(0, Σ), it
is easy to verify L = λ2
max. Then, we complete the proof.
■

C.2
The Closed-form Minimizer of Few-shot Diffusion Models

When the latent distribution is a Gaussian distribution qz = N(0, Σ) with Σ = diag
 
λ2
1, . . . , λ2
d

≻
0, the ground truth score function for the target dataset is

∇log qta
t (X) = −AtaΣ−1
t A⊤
taX −1

σ2
t

 
ID −AtaA⊤
ta

X ,

18


---Page Break---
where Σt = diag
 
. . . , m2
tλ2
k + σ2
t , . . .

. Since the matrix Ata is independent of time t, we fix a
t ∈[δ, T] and minimize the few-shot objective function at this time. With an approximated bΣ, which
is learned by the pre-trained diffusion models, fˆθ(Z, t) = (Id −σ2
t bΣ−1
t )Z, where bΣt = m2
t bΣ + σ2
t Id.
Hence, the expected objective function for the few-shot diffusion models at a fixed time t is:

min
sVta,ˆ
θ∈˜
QNN(ˆθ)
Lta,t(sVta,ˆθ) = EXta∼qta
h
ℓta
t

Xta; sVta,ˆθ
i
,

where

˜QNN(θ) = {sV,θ(X, t) = 1

σ2
t
V fθ
 
V ⊤X, t

−1

σ2
t
X : V ∈RD×d with rank(V ) = d.} ,

Note that the constraint rank(V ) = d is a weaker constraint than V ⊤V = Id since rank(V ) = d
does not involve length information.
Lemma C.2. Assume Assumption 3.1 and qz = N(0, λ2Id). Let C = EXta∼qta 
XtaX⊤
ta

be the
expected data covariance matrix. Then, eVta has a closed form:

eVta eV ⊤
ta = m2
t bλ2 + σ2
t
bλ2
 
C + σ2
t ID
−1 C .

Proof. Let bGt = Id −σ2
t bΣ−1
t , then we have that

ℓta
t

Xta,i; sVta,ˆθ

= EXt|X0=Xta,i


∥1

σ2
t
Vta bGtV ⊤
ta Xt −1

σ2
t
Xt −∇log qta
t (Xt|X0)∥2
2



= 1

σ4
t
EXt|X0=Xta,i
h
∥Vta bGtV ⊤
ta Xt −mtX0∥2
2
i

where the second equality follows ∇log qta
t (Xt|X0) = −Xt−mtX0

σ2
t
. Let C = EXta∼qta

XtaX⊤
ta


be the expected covariance matrix of the target dataset. With the fact EXt|X0[XtX⊤
t ] = m2
tX0X⊤
0 +
σ2
t ID and EXt|X0[X0X⊤
t ] = mtX0X⊤
0 , the optimization problem can be simplified to the following
form (without misunderstanding, we ignore the subscript ta):

min
V ∈RD×d,rank(V )=d L(V ) = ∥(m2
tC + σ2
t ID)
1
2 V bGtV ⊤∥2
F −2m2
ttr(V bGtV ⊤C) ,

where (m2
tC + σ2
t ID)
1
2 is meaningful since (m2
tC + σ2
t ID) is positive-definite matrix. Let eV be
the solution of the above minimization problem. We first ignore the constraint rank(V ) = d and
calculate ∂L(V )/∂V = 0 (since eV also satisfied ∂L(V )/∂V |V =eV = 0), we know that eV satisfies
the following equality:

(m2
tC + σ2
t ID)V bGtV ⊤V bGt = m2
tCV bGt ,
which indicate
((m2
tC + σ2
t ID)V bGtV ⊤−m2
tC)(V bGt) = OD×d .

The above equality means rank((m2
tC + σ2
t ID)V bGtV ⊤−m2
tC) + rank(V bGt) ≤d. Since
rank(V ) = d and rank( bGt) = d, then we have that rank(V bGt) = d and

(m2
tC + σ2
t ID)V bGtV ⊤= m2
tC .
(5)

In Section 5, we assume the latent distribution is a isotropic Gaussian distribution qz = N(0, λ2Id).

In this setting, bΣ is equal to bλ2Id and bGt =
m2
t bλ2

m2
t bλ2+σ2
t , which indicate the closed form solution of eV
is

eV eV ⊤= m2
t bλ2 + σ2
t
bλ2
(C + σ2
t ID)−1C .

The last step is to prove rank(eV eV ⊤) = d. Note that rank(C) = rank
 
EXta∼qta

XtaX⊤
ta

=
rank(AΣA⊤) = d, which indicates

min{rank(eV eV ⊤), rank(m2
tC + σ2
t ID)} ≥d .

Combined with eV ∈RD×d, we complete the proof.
■

19


---Page Break---
C.3
The Error Bound for the empirical Closed-form Solution

In this part, we prove the accuracy bound of the empirical version closed form solution ¯eV ¯eV
⊤
w.r.t.
ns and nta. The empirical solution has the following form (without misunderstanding, we ignore the
subscript ta):

¯eV ¯eV
⊤
= m2
t bλ2 + σ2
t
bλ2
(m2
t ¯C + σ2
t ID)−1 ¯C ,

where ¯C =
1
nta
Pnta
i=1 Xta,iX⊤
ta,i.

Theorem 5.2. Assume Assumption 3.1 and qz = N(0, λ2Id). Let bqz be the latent distribution

generated by the pre-trained models with (bVta, bΣ) and M =
d2β2(d+λ2)

λ
p

Dd log (Ddns) (d2 ∨D).
Then, with probability 1 −δ1, we have that for any t ∈[δ, T]

¯eV ta ¯eV
⊤

ta −AtaA⊤
ta



2

F
≤eO

 
d log( 1

δ1 )

m2
tλ2 + σ2
t


M
dδ√ns
+ d

nta
(m2
tλ2 + σ2
t )2
!

.

Proof. The empirical solution indicates that

(m2
t ¯C + σ2
t ID) ¯eV ¯eV
⊤
= m2
t bλ2 + σ2
t
bλ2
¯C .
(6)

To analyze this equality, we first show that bλ2 is accurate enough. We know that KL
 
q0∥ˆpLD
T

=

d

λ2/bλ2 −log(λ2/bλ2) −1

. Let M1 =
d2β2(d+λ2
max)
λminδ
p

(d2 + Dd) log (Ddns) (d2 ∨D). Then,
Lemma C.1 show that with probability 1 −δ1, we have

KL
 
q0∥ˆpLD
T

= d

λ2/bλ2 −log(λ2/bλ2) −1

≤eO



M1

s

log(1/δ1)

ns



.

Combined with the above inequality, we know that with probability 1 −δ1, |λ2/bλ2 −1| ≤
r

M1√

log(1/δ1)
d√ns
by using the Taylor Expansion. By using Lemma D.2, we know that with probability

1 −δ1:

m2
t bλ2 + σ2
t
bλ2
¯C ≤eO

 
m2
tλ2 + λ2σ2
t
bλ2

  

1 + 2
p

d + log (1/δ1)
√nta

!

AA⊤
!

≤eO







m2
tλ2 + σ2
t +

s

M1
p

log(1/δ1)
d√ns
σ2
t





 

1 + 2
p

d + log (1/δ1)
√nta

!

AA⊤



.

(7)

For the left hand of Equation (6), we know that

(m2
t ¯C + σ2
t ID) ¯eV ¯eV
⊤
≥

 

m2
tλ2AA⊤−m2
tλ2 2
p

d + log (1/δ1)
√nta
+ σ2
t ID

!
¯eV ¯eV
⊤
(8)

Combined with Equation (7) and Equation (8), we have that

 
m2
tλ2AA⊤+ σ2
t ID
 
¯eV ¯eV
⊤
−AA⊤


≤eO

 



s

M1
p

log(1/δ1)
d√ns
+ 2
p

d + log (1/δ1)
√nta



m2
tλ2 + σ2
t +

s

M1
p

log(1/δ1)
d√ns
σ2
t







AA⊤

+ 2m2
tλ2
p

d + log (1/δ1)
√nta
¯eV ¯eV
⊤
!

.

20


---Page Break---
Let M2(ns, nta) =

r

M1√

log(1/δ1)
d√ns
+
2√

d+log(1/δ1)
√nta

 

m2
tλ2 + σ2
t +

r

M1√

log(1/δ1)
d√ns
σ2
t

!

. Accord-

ing to symmetry, we know that

 
m2
tλ2AA⊤+ σ2
t ID
 
¯eV ¯eV
⊤
−AA⊤


2

F
≤eO

M2(ns, nta)2∥AA⊤∥2
F + m4
tλ4(d + log (1/δ1))

nta
∥¯eV ¯eV
⊤
∥2
F



≤eO
 
D2(M2(ns, nta)2
.

The last inequality follows that each element of ¯eV ¯eV
⊤
is bounded by some constant due to the form
of the empirical closed form solution and

∥AA⊤∥2
F = tr(AA⊤AA⊤) = tr(AA⊤) = tr(Id) = d .

For the right hand of the above inequality, we know that

 
m2
tλ2AA⊤+ σ2
t ID
 
¯eV ¯eV
⊤
−AA⊤


2

F

= Tr

¯eV ¯eV
⊤
−AA⊤
 
¯eV ¯eV
⊤
−AA⊤
  
m2
tλ2AA⊤+ σ2
t ID
  
m2
tλ2AA⊤+ σ2
t ID


≥
 
m2
tλ2 + σ2
t
 


¯eV ¯eV
⊤
−AA⊤


2

F
.

Then, we complete the proof
■

D
Auxiliary Lemmas

The following concentration lemma comes from Lemma 15 of Chen et al. (2023c).
Lemma D.1 (Lemma 15, (Chen et al., 2023c)). Let G be a bounded function class, i.e., there exists a
constant B such that any g ∈G : Rd 7→[0, B]. Let z1, . . . , zn ∈Rd be i.i.d. random variables. For
any δ ∈(0, 1), a ≤1, and τ > 0, we have

P

 

sup
g∈G

1
n

n
X

i=1
g (zi) −(1 + a)E[g(z)] > (1 + 3/a)B

3n
log N (τ, G, ∥· ∥∞)

δ1
+ (2 + a)τ

!

≤δ1
and

P

 

sup
g∈G
E[g(z)] −1 + a

n

n
X

i=1
g (zi) > (1 + 6/a)B

3n
log N (τ, G, ∥· ∥∞)

δ1
+ (2 + a)τ

!

≤δ1 .

In the following lemma, we show the concentration of the data covariance matrix. Note that the
proof sketch of the following lemma mainly follows Lemma 6 of (Du et al., 2020). We prove a
concentration bound that depends on n instead of a constant bound with a large enough n.
Lemma D.2 (The Modified Lemma A.6, (Du et al., 2020)). Let a1, . . . , an be i.i.d. d-dimensional
random vectors such that E [ai] = 0, E

aia⊤
i

= I, and ai is ρ2-subgaussian. Then with probability
at least 1 −δ1 we have
 

1 −2ρ2p

d + log (1/δ1)
√n

!

I ⪯1

n

n
X

i=1
aia⊤
i ⪯

 

1 + 2ρ2p

d + log (1/δ1)
√n

!

I

Proof. Let A = 1

n
Pn
i=1 aia⊤
i −I. Similar to Du et al. (2020), we use an ϵ-net argument for the unit
sphere Sd−1 =

v ∈Rd : ∥v∥= 1
	
. For any v ∈Sd−1, we know that
 
v⊤ai
2 −1 is zero-mean
and 16ρ2-subgaussian. By using the Bernstein inequality, we have for any ϵ > 0

Pr
v⊤Av
 > ϵ

≤2 exp

 

−n

2 min

(
ϵ2

(16ρ2)2 ,
ϵ
16ρ2

)!

.

21


---Page Break---
Next, we take a 1

5-net N ⊂Sd−1 of Sd−1 with size |N| ≤eO(d). By using the union bound, we
know that

Pr

max
v∈N
v⊤Av
 > ϵ

≤2|N| exp

 

−n

2 min

(
ϵ2

(16ρ2)2 ,
ϵ
16ρ2

)!

≤exp

 

O(d) −n

2 min

(
ϵ2

(16ρ2)2 ,
ϵ
16ρ2

)!

Let the right hand of the above inequality equals to δ1. We know that with probability 1 −δ1:

max
v∈N
v⊤Av
 ≤ρ2p

d + log (1/δ1)
√n
.

Note that for any u ∈Sd−1, there exists u′ ∈N such that ∥u −u′∥≤1

5. Then, we have that

|u⊤Au| ≤| (u′)⊤Au′| + 2| (u −u′)⊤Au′| + | (u −u′)⊤A (u −u′) |

≤ρ2p

d + log (1/δ1)
√n
+ 1

2∥A∥2 ,

where ∥A∥2 is the operator norm of matrix A. Taking a supreme over u ∈Sd−1, we have that

∥A∥2 ≤ρ2p

d + log (1/δ1)
√n
+ 1

2∥A∥2 .

Then, we complete the proof.
■

In Section 5, we assume the latent distribution is a Gaussian distribution instead of a subgaussian one.
Yuan et al. (2023) show that in this setting, the approximation error bound has better dependence on
ns.

Lemma D.3 (Lemma C.1, (Yuan et al., 2023)). Assume the latent distribution is a Gaussian distribu-
tion qz = N(0, Σ) with Σ = diag
 
λ2
1, . . . , λ2
d

≻0. Then, the solution (bVs, bθ) of Equation (1) has
the following approximation error bound with probability 1 −δ1:

1
T −δ

Z T

δ

∇log qs
t (·) −sbVs,bθ (·, t)

2

L2(qs
t ) dt ≤O



1

δ

s

(d2 + Dd) log (Ddns) (d2 ∨D) log 1

δ1
ns



.

E
Additional Experiments

In this section, we do experiments on real-world datasets to show that the new model obtained by
only fine-tuning appropriate encoder and decoder layers on target datasets with only 10 images
can generate novel images with the target dataset feature. On the contrary, if all parameters can be
fine-tuned, the model will suffer from memory phenomenon and only generate the ten images in the
target dataset. This phenomenon indicates that only fine-tuning the appropriate encoder and decoder
will result in a model with a generalization property.

Setting.
In this experiment, we use a U-net network with attention layers, which contains 11
downblocks, 2 middleblocks, and 15 upblocks. When only fine-tuning the encoder and decoder
layers, we fine-tune the first 4 downblock layers (encoder) and 4 upblock layers (decoder) instead of
only using linear layers as the encoder and decoder (discuss in the later discussion paragraph).

The above experiments are conduct on a GeForce RTX 4090. We train the neural network using
AdamW optimizer with learning rate 0.0001. For the pre-trained phase, we train the models for
200 epochs with batch size 20. It takes 5 hours to obtain a pre-trained diffusion models. For the
fine-tuning phase, we fine-tune the pre-trained models for 400 epochs with batch size 2. It take 3
minutes to fine-tune the pre-trained models.

22


---Page Break---
Figure 2: The experiments on cat face dataset

Dataset.
Our experiments use 2 real-world datasets: the CelebA64 dataset and the cat face dataset.

• CelebA64 (size 3 ∗64 ∗64).
(a) Source dataset: 6400 images of faces with different hairstyles (without the bald feature).
(b) Target dataset: 10 images with the bald feature in CelebA64.
• Cat face images (size 3 ∗64 ∗64).
(a) Source dataset: 4200 cat images with different colors (without black color cat).
(b) Target dataset: 10 black color cat images (The color black constitutes more than 70% of
the image’s composition.).

Discussion on results.
The experiment results of CelebA64 have been discussed in Section 6. The
experiment phenomenon is similar for the cat face images, which means the models obtained by only
fine-tuning the encoder and decoder can generate novel images with the target feature (Figure 2). We
note that when choosing the target cat face dataset if the color black constitutes more than 70% of
the image’s composition, we view this cat image as the black cat. Hence, different colors exist for
cats, such as white and grey, due to the target dataset containing a small number of these colors (such
as images 1, 3, 4, 6, 8). As a result, our fine-tuning results also contain these colors. However, our
results do not contain colors other than those in the target dataset and can produce novel samples,
which also proves the effectiveness of our fine-tuning method.

Discussion on linear encoder and decoder.
Assumption 3.1 assumes the linear subspace, which
indicates linear encoder and decoder. However, we fine-tune the first 4 downblock layers (encoder)
and 4 upblock layers (decoder) instead of only using linear layers as the encoder and decoder. We note
that this operation does not conflict with our Assumption. Recall that in Stable Diffusion (Rombach
et al., 2022), the diffusion models run in the VAE embedding space 7. Hence, we can view the first
3 downblock layers and the last 3 upblock layers as the VAE encoder and VAE decoder. Then, we
can obtain X in this paper by running the VAE encoder. The remaining 1 downblock and 1 upblock
layer can be viewed as linear encoder and decoder A. As mentioned in Section 4.2 of StyleGAN
(Karras et al., 2019), the feature of X obtained by running a good-enough VAE encoder has linear
separability, which also supports our Assumption 3.1.

7To distinguish the latent space in this paper, we use embedding space here.

23


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: This work aims to explain the reason why few-shot diffusion models can
achieve great performance with a limited dataset from the theoretical perspective. We
achieve this goal by analyzing the approximation and optimization perspective (Theorem 4.3
and Theorem 5.2). We also do real-world experiments to support our theoretical results
(Section 6).

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

Justification: We discuss future work and limitation at Section 7.

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

24


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: We have shown all assumptions, theorem and proof sketch in the main paper.
The detailed proof appears in the appendix.
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
Justification: We has shown all experiments detail in Appendix E.
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

25


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [No]
Justification: As a theoretical work, we simply train and fine-tune a diffusion models on the
datasets in Appendix E. All detail is shown in Appendix E.
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
Justification: We has shown all experiments detail including dataset and training detail in
Appendix E.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: Since this work is a theoretical work, the experiments are qualitative experi-
ments on the dataset we constructed and are used to support our theoretical results. Since we
only use 10 images to fine-tune the models to generate images with specific target features,
it is hard to calculate quantitative metrics such as FID. However, our experiment results still
show that our methods can generate novel images with the target feature compared to the
benchmark.
Guidelines:

• The answer NA means that the paper does not include experiments.

26


---Page Break---
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

Justification: We have shown the compute works and computation time in Appendix E.

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

Justification: We have checked the code of ethics and make sure that our work satisfies the
code of ethics.

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

Justification: We have discussed the broader impacts of our work at the end of main paper.

27


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

28


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

29


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

30


---Page Break---
