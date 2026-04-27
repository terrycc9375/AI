On the Comparison between Multi-modal and
Single-modal Contrastive Learning

Wei Huang∗†
RIKEN AIP
wei.huang.vr@riken.jp

Andi Han∗
RIKEN AIP
andi.han@riken.jp

Yongqiang Chen
The Chinese University of Hong Kong
yqchen@cse.cuhk.edu.hk

Yuan Cao
The University of Hong Kong
yuancao@hku.hk

Zhiqiang Xu†
MBZUAI
zhiqiang.xu@mbzuai.ac.ae

Taiji Suzuki
University of Tokyo & RIKEN AIP
taiji@mist.i.u-tokyo.ac.jp

Abstract

Multi-modal contrastive learning with language supervision has presented a
paradigm shift in modern machine learning. By pre-training on a web-scale dataset,
multi-modal contrastive learning can learn high-quality representations that exhibit
impressive robustness and transferability. Despite its empirical success, the theoret-
ical understanding is still in its infancy, especially regarding its comparison with
single-modal contrastive learning. In this work, we introduce a feature learning
theory framework that provides a theoretical foundation for understanding the
differences between multi-modal and single-modal contrastive learning. Based
on a data generation model consisting of signal and noise, our analysis is per-
formed on a ReLU network trained with the InfoMax objective function. Through
a trajectory-based optimization analysis and generalization characterization on
downstream tasks, we identify the critical factor, which is the signal-to-noise ratio
(SNR), that impacts the generalizability in downstream tasks of both multi-modal
and single-modal contrastive learning. Through the cooperation between the two
modalities, multi-modal learning can achieve better feature learning, leading to
improvements in performance in downstream tasks compared to single-modal
learning. Our analysis provides a unified framework that can characterize the
optimization and generalization of both single-modal and multi-modal contrastive
learning. Empirical experiments on both synthetic and real-world datasets further
consolidate our theoretical findings.

1
Introduction

Large-scale pre-trained models have achieved unprecedented success, including GPT series [6, 41],
LLaMa [53], among many others. CLIP [42] as a typical example, uses a multi-modal contrastive
learning framework to learn from a massive scale of image-caption data. The multi-modal contrastive
learning in CLIP has shown significant capabilities to learn high-quality representations, which are
ready to be adapted to a wide range of downstream tasks, forming the backbone of generative models

∗Equal Contribution.
†Corresponding Author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
like DALL-E2 [43], prompt learning [61] as well as general purpose multi-modal agents [62, 35].
Given the huge success of models like CLIP that have stellar zero-shot and few-shot capabilities
on a wide range of out-of-distribution (OOD) benchmarks, they have been widely recognized as
foundation models (FMs). More similar examples are given by ALIGN [28], Florence [59], BLIP
[33], Flamingo [1].

Despite the unprecedented success achieved by multi-modal contrastive learning, the fundamental
mechanism that leads to greater performance, especially compared to single-modal contrastive
learning is still under-explored. Recently, several seminal works provided theoretical explanations for
either single-modal [4, 5, 14, 48, 24, 7, 50, 49, 20] or multi-modal contrastive learning [38, 37, 44, 12].
For example, [56] studied how single-modal contrastive learning learns the feature representations for
neural networks by analyzing its feature learning process. As for multi-modal contrastive learning,
[12, 58] provided explanations for why multi-modal contrastive learning demonstrates zero-shot
transferability, and robustness to distribution shifts, than supervised learning, which offer valuable
insights. Although both lines of the existing works provide valid theoretical insights under the
respective settings, rare work has compared the optimization and generalization of the two types of
contrastive learning under a unified framework. This motivates us to establish a systematic feature
learning analysis for both single-modal and multi-modal contrastive learning.

In particular, we consider a data generation model that contains two modalities of data, which are
generated from signal and noise features. The signal feature correlates in different modalities, while
there is no correlation between noise features among modalities. We then study the optimization
of single-modal and multi-modal contrastive learning under gradient descent training. By studying
the trajectories of signal learning and noise memorization, we establish the convergence conditions
and further characterize the generalization ability in the downstream tasks. The results show that,
through the cooperation between modalities, multi-modal contrastive learning can achieve better
generalization in the downstream task. In contrast, without the help of the second modality, single-
modal contrastive learning concentrates on learning noise from the data, and thus generalizes poorly
on the downstream tasks. The main contributions of this work are summarized as follows:

• This work establishes the first systematic comparative optimization analysis for single-modal
and multi-modal contrastive learning under gradient descent training in non-convex settings. We
show that both single-modal and multi-modal can achieve near-zero training error under InfoMax
contrastive loss after polynomial number of iterations, by overcoming the non-convex difficulty.

• By a trajectory-based analysis of the signal learning and noise memorization of the ReLU network
from the data, we successfully characterize the difference in generalization between single-modal
and multi-modal contrastive learning. The distinct SNRs of different modalities lead to a divergence
in the generalization of downstream tasks for the two contrastive learning frameworks.

• Our theory suggests that the advantage of multi-modal over single-modal contrastive learning
comes from the high quality of the second modality and the cooperation between the two modalities
through contrastive learning. This divergence is ultimately reflected in the difference in feature
learning and the final gap in downstream task generalization. Experimental results on both synthetic
and real-world datasets confirm our theoretical findings and understanding.

2
Related Work

Theoretical Understanding of Single-modal Contrastive Learning. The seminal work [4] started
theoretical research on single-modal contrastive learning. They assumed that different positive
samples are independently drawn from the same latent class, making a connection to supervised
learning. [55] identified two key properties related to the contrastive loss: alignment and uniformity.
Alongside, [32] illustrated that predicting auxiliary prediction tasks helps in learning representations
effective for downstream prediction tasks, and [52] provided a theoretical analysis of contrastive
learning in the multi-view setting. Besides, [51] proposed a theoretical framework to understand
contrastive self-supervised learning from an optimization perspective. [21] proposed a loss that
performs spectral decomposition on the population augmentation graph and can be succinctly written
as a contrastive learning objective on neural net representations. [46] pointed out the importance of
inductive biases of the function class and training algorithm in understanding contrastive learning.
The most related work to us is the work by [56]. Similar to them, this work studies ReLU networks
and considers the signal-noise data model. However, we do not require the adjustable bias term in

2


---Page Break---
the activation function, which plays a critical role in [56]. Furthermore, this work adopts a unified
framework to compare with multi-modal contrastive learning, which is out of scope in [56].

Understanding of Multi-modal Contrastive Learning. As the multi-modal contrastive learning
approaches such as CLIP received great success, recent works have been proposing explanations
from empirical perspective. [36] empirically showed that high train-test similarity is insufficient to
explain CLIP’s OOD performance. [60] illustrated that CLIP behaves similarly to Bags-of-words in
language-based image retrieval, i.e., the order of words in the input sentence does not largely affect
CLIP to find the corresponding image. Besides, [16] demonstrated that training data diversity and the
ability to leverage the diversity as supervised learning is the key to the effective robustness of CLIP.
Theoretically, [13] proved that multi-modal contrastive learning can block-identify latent factors
shared between modalities by the a generative data model. [44] analyzed the training dynamics of
a simple multi-modal contrastive learning model and show that contrastive pairs are important for
the model to efficiently balance the learned representations. Furthermore, [38] showed that each
step of loss minimization by gradient descent can be seen as performing SVD on a contrastive
cross-covariance matrix. Similar to us, [25] tried to answer why multi-modal learning is better than
single model learning. However, they did not consider contrastive learning and thus cannot explain
the success of multi-modal contrastive multi-modal learning.

Data Quality Matters for Multi-modal Contrastive Learning. Aligned with our theoretical
results, there is a lot of empirical evidence showing that improving the alignment quality with
more descriptive captions improves multi-modal contrastive learning. [16] show that the training
distribution mostly determines the generalizability of CLIP. Furthermore, [47, 40, 18, 17] find filtering
poorly aligned image-caption samples used for training leads to further improvements. Besides,
[45, 39, 15] demonstrate that improving the descriptiveness of the captions could further boost
the performance of CLIP. Besides, [34] demonstrated that the caused by a combination of model
initialization and contrastive learning optimization. However, their results do not take neural network
architecture into consideration, and do not provide an analysis of test errors either.

3
Problem Setting

Notation.
We use bold-faced letters for vectors and matrices otherwise representing scalar. We
use ∥· ∥2 to denote the Euclidean norm of a vector or the spectral norm of a matrix, while denoting
∥· ∥F as the Frobenius norm of a matrix. For a neural network, we denote σ(·) as the activation
function and we adopt ReLU activation where σ(x) = max{0, x} in this work. To simplify, we
denote [n] = {1, 2, . . . , n}.

Data Model.
In this work, we consider the following data model, which consists of signal and
noise. In the first modality, example (x, y) ∼D is generated as follows:

x = [x(1)⊤, x(2)⊤]⊤= [yµ⊤, ξ⊤]⊤,
y ∼unif({−1, 1}).
(1)

where x ∈R2d is the input feature and y ∈{−1, 1} is the corresponding label generated from
Rademacher distribution. In particular, x(1) = yµ ∈Rd is the task-relevant signal vector, and
x(2) = ξ ∼N(0, σ2
ξI) ∈Rd is the task-irrelevant noise vector. Intuitively, if a network learns
primarily from signal, it can effectively generalize to unseen data and vice versa. Similar data models
have been adopted in recent theoretical works on supervised learning [2, 26, 8, 23, 31, 63, 22, 11]
and self-supervised learning [56, 50, 30].

Similarly for the second modality, a sample (ex, y) ∼eD is generated as

ex = [ex(1)⊤, ex(2)⊤]⊤= [yeµ⊤, eξ
⊤]⊤,
y ∼unif({−1, 1}),
(2)

where the input feature ex ∈R2 e
d and the label y is shared with the first modality. Besides, the signal
is a given vector eµ ∈R e
d, and noise follows eξ ∼N(0, σ2
˜ξI) ∈R ˜d. The linear data models for
multi-modal learning have also been studied in previous work [44]. To simplify the analysis, we set
d = ed, σξ = σ˜ξ. However, we highlight that extensions to deal with unmatched dimension and noise
level is possible.

3


---Page Break---
3.1
Single-modal Contrastive Learning

We use a single-layer neural network h : R2d →Rm with ReLU activation as our encoder, where m
is the number of neurons, which represents the embedding dimension. More precisely,

h(x) = [¯h1(x), . . . , ¯hm(x)]⊤∈Rm,
where ¯hr(x) = hr(x(1)) + hr(x(2)),
(3)

here we let hr(x(i)) = σ(⟨wr, x(i)⟩) for r ∈[m], i ∈[2], and σ(·) is the ReLU activation function.
We adopt a Gaussian to initialize the weights w(0)
r
∼N(0, σ2
0I), where σ0 severs as the strength.

Given a pair of positive data samples, the contrastive loss function is based on the similarity measure
defined as the inner product between the representation of two samples x, x′ ∈R2d:

Simh(x, x′) = 1

m

m
X

r=1
hr(x(1))sg(hr(x′(1))) + 1

m

m
X

r=1
hr(x(2))sg(hr(x′(2))),
(4)

where the sg(·) is the stop-gradient operation, which is inspired by recent empirical works [19, 10]
and theoretical work studying contrastive learning [56]. Here we define positive sample as

bx = [bx(1)⊤, bx(2)⊤]⊤= [yµ⊤, ξ⊤+ ϵ⊤]⊤, ϵ ∼N(0, σ2
ϵ I).
(5)

In particular, we consider the form of augmentation where the signal stays invariant while the noise
vector is corrupted with added independent noise. Similar setup has been considered in [57]. We
consider the contrastive loss presented as follows:

L = −1

n

n
X

i=1
log(
eSimh(xi,bxi)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ ),
(6)

where τ is the temperature parameter, n is the number of training samples, and M is the number of
negative pairs. In this work, to efficiently optimize the loss to near zero, we require negative sample
pairs do not share the same label, i.e., yj ̸= yi in (6). Note that this setting is aligned with supervised
contrastive learning [29, 27].

We use gradient descent to optimize the contrastive learning loss, which leads to the gradient update:

w(t+1)
r
= w(t)
r
−η∇wrL(W(t)) = w(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)h(t)
r (bx(1)
i )h′(t)(x(1)
i )yiµ

+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)h(t)
r (bx(2)
i )h′(t)
r
(x(2)
i )ξi −
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(1)
j )h′(t)(x(1)
i )yiµ

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(2)
j )h′(t)
r
(x(2)
i )ξi,
(7)

where we denote ¯h(t)
r (x) = σ(⟨w(t)
r , x⟩), η as the learning rate, and we define the loss derivatives as

ℓ′(t)
i
≜
eSimh(xi,bxi)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ , ℓ′(t)
i,j ≜
eSimh(xi,xj)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ . (8)

Intuitively, when the similarity between positive pair is high, and the similarity between negative time
is low, we can see ℓ′(t)
i
≈1 and ℓ′(t)
i,j ≈0, for i ∈[n] and j ∈[M]. Therefore, the gradient descent
in Eq. (7) is close to zero, indicating the near convergence result. Furthermore, from Eq. (7), we
observe that the evolution direction of weight is composed of signal vector µ and noise vectors ξi for
i ∈[n]. This observation plays a critical role in our following theoretical analysis.

3.2
Multi-modal Contrastive Learning

We use two neural networks h : Rd →Rm and g : R ˜d →Rm to encode two input modality x and ex
respectively. Both neural networks use ReLU activation function. More precisely,

h(x) = [¯h1(x), . . . , ¯hm(x)]⊤∈Rm,
where ¯hr(x) = hr(x(1)) + hr(x(2))

4


---Page Break---
g(ex) = [¯g1(ex), . . . , ¯gm(ex)]⊤∈Rm,
where ¯gr(x) = gr(ex(1)) + gr(ex(2))

we let hr(x(i)) = σ(⟨wr, x(i)⟩) and gr(ex(i)) = σ(⟨ewr, ex(i)⟩). Here σ(·) is the ReLU activation
function, wr ∈Rd and ewr ∈R e
d for r ∈[m] are the weights in two networks. Given the embedding,
the similarity function of the two modalities is defined as

Simh,g(x, ex)
= 1

m
Pm
r=1 hr(x(1))sg(gr(ex(1))) + 1

m
Pm
r=1 hr(x(2))sg(gr(ex(2))),

Simg,h(ex, x)
= 1

m
Pm
r=1 gr(ex(1))sg(hr(x(1))) + 1

m
Pm
r=1 gr(ex(2))sg(hr(x(2))).

The two similarity functions defined above are modality-centered with stop-gradient operation applied.
The objective function of contrastive multi-modal learning can be expressed as

L = −1

n

n
X

i=1
log(
eSimh,g(xi,exi)/τ

eSimh,g(xi,exi)/τ + PM
j̸=i eSimh,g(xi,exj)/τ )

−1

n

n
X

i=1
log(
eSimg,h(exi,xi)/τ

eSimg,h(exi,xi)/τ + PM
j̸=i eSimg,h(exi,xj)/τ ).
(9)

Same to the single-modal learning whose objective function is governed by Eq. (6), the objective
function for multi-modal contrastive learning adopt one positive pair and M negative pairs. Besides,
we require the negative pairs do not share the same label. To optimize the objective function (9) for
multi-modal learning, gradient descent is applied to train two encoders simultaneously. The gradient
descent rule for the first modal network is governed by the following expression.

w(t+1)
r
= w(t)
r
−η∇wrL(W(t)) = w(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)g(t)
r (ex(1)
i )h′(t)
r
(x(1))yiµ

+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)g(t)
r (ex(2)
i )h′(t)
r
(x(2))ξi −
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j g(t)
r (ex(1)
j )h′(t)
r
(x(1))yiµ

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j g(t)
r (ex(2)
j )h′(t)
r
(x(2))ξi.
(10)

Here with a slight abuse of notation, we use ℓ′(t)
i
, ℓ′(t)
i,j to represent the loss derivatives for both
modalities. Compared to signal-modal learning, the main difference for the multi-modal learning
is that the corresponding embedding is from another modality. The gradient update for the second
modality can be derived similarly, which we omit here for clarity.

3.3
Downstream Task Evaluation

To evaluate the out-of-distribution generalization of single-modal and multi-modal contrastive learning
for downstream task, we consider a test distribution Dtest, where a sample xtest = [y · ν⊤, ζ⊤]⊤

∼Dtest is generated as follows. The test signal ν satisfies ⟨ν, µ⟩= O(∥µ∥2
2d−1/2) and the test
noise follows ζ ∼N(0, σ2
ξI) and y follows Rademacher distribution. After the training is complete,
we introduce a linear head on top of the learned embedding h(xtest) for adapting to test distribution,
i.e., f(xtest) = ⟨w, h(xtest)⟩. Specifically, we consider the task of classification and define the
population 0-1 test error as LDtest = Pxtest∼Dtest

yf(xtest) < 0

.

4
Main Results

In this section, we introduce our key theoretical findings that elucidate the optimization and general-
ization result for both single-modal and multi-modal contrastive learning through the feature learning
analysis. We use a trajectory-based analysis for the iterations induced by gradient descent, following
a post-training analysis for the performance on the downstream test set. Below we provide the main
assumption and main theorems.

Assumption 4.1. Let SNR = ∥µ∥2/(σξ
√

d). Assume (1) d ≥eΩ(max{n2, nσ−1
0 σ−1
ξ , σ−2
0 ∥µ∥−2
2 }).
(2) η ≤O(min{m∥µ∥−2
2 , nmσ−2
ξ d−1}). (3) σ0 ≤eO((max{σξ
√

d, ∥µ∥2})−1). (4) m, n ≥eΩ(1).

5


---Page Break---
(5) σϵ ≤min{eΘ(∥µ∥2), σξ/eΩ(1)}. (6) n · SNR2 = Θ(1). (7) Cµ∥µ∥2 = ∥eµ∥2, where Cµ ≥2.66
is a constant.

(1) We adopt a high dimensional setting to ensure enough over-parameterization. (2,3) The learning
rate and the strength of initialization are chosen to make sure the that gradient descent can effectively
minimize the contrastive loss. (4) The choice of hidden size m and number of training sample n
is to provide adequate concentration. (5) The strength of augmentation is set to keep the similarity
between two positive samples. (6) The relation between number of sample and SNR is to distinguish
the feature learning process between single-modal and multi-modal contrastive learning. (7) To
differentiate single-modal and multi-modal contrastive learning, we introduce a constant Cµ, which
enables the cooperation between the two modalities in multi-modal contrastive learning.
Theorem 4.2 (Single-Modal Contrastive Learning). Under the single-modal learning setup, suppose
Assumption 4.1 holds. Then after T ∗= eΘ(η−1mnσ−2
ξ d−1 + η−1mnσ−2
ξ d−1ϵ−1), the with proba-
bility at least 1 −1/d, it holds that (1) Training error L(T ∗) ≤ϵ and (2) Test error at down-stream
task LDtest(T ∗) = Θ(1).

Theorem 4.2 states that despite the small training error achieved by single-modal contrastive learning,
the test error is large in the downstream task.
Theorem 4.3 (Multi-Modal Contrastive Learning). Under the single-modal learning setup, suppose
Assumption 4.1 holds. Then after T ∗= eΘ(η−1mnσ−2
ξ d−1 + η−1mnσ−2
ξ d−1ϵ−1), the with proba-
bility at least 1 −1/d, it holds that (1) Training error L(T ∗) ≤ϵ and (2) Test error at down-stream
task LDtest(T ∗) = o(1).

Theorem 4.3 demonstrates that trained multi-modal contrastive learning can achieve both small
training error and downstream test error. Compared to Theorem 4.2, Theorem 4.3 shows that the
generalization of multi-modal contrastive learning in downstream tasks is better than single-modal
contrastive learning. The reason behind this difference is that the two modalities can cooperate with
each other; the higher quality in one modality can boost the feature learning in the target modality,
helping to generalize to the downstream task. On the contrary, augmentation often maintains the
same SNR as the original data, so single-modal learning hardly benefits from the augmentation and
can only memorize the noise from the data, which is not applicable to downstream tasks.

5
Proof Roadmap

5.1
Proof Sketch for Single Modal Contrastive Learning

The proof is constructed by a optimization analysis followed by a generlization analysis in the
downstream task. Through the application of the gradient descent rule outlined in Eq. (7), we observe
that the gradient descent iterate w(t)
r
is a linear combination of its random initialization w(0)
r , the
signal vector µ and the noise vectors in the training data ξi for i ∈[n]. Consequently, for r ∈[m],
the decomposition of weight vector iteration can be expressed:

w(t)
r
= w(0)
r
+ γ(t)
r ∥µ∥−2
2 µ +

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ξi,
(11)

where γ(t)
r
and ρ(t)
r,i serve as coefficients and represent signal learning and noise memorization

respectively. Based on the the gradient descent update (7), the iteration of γ(t)
r
and ρ(t)
r,i are given:

Lemma 5.1 (Single-modal Contrastive Learning). The coefficients γ(t)
r ρ(t)
r,i in decomposition (11)
satisfy the following equations:

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i=1


(1 −ℓ′(t)
i
)h(t)
r (bx(1)
i ) −

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(1)
j )

h′(t)
r
(x(1)
i )yi∥µ∥2
2,
(12)

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ

(1 −ℓ′(t)
i
)hr(bx(2)
i ) −

M
X

j̸=i
ℓ′(t)
i,j hr(x(2)
j )

h′(t)
r
(x(2)
i )∥ξi∥2
2,
(13)

where the initialization γ(0)
r , ρ(0)
r,i = 0.

6


---Page Break---
Lemma 5.1 tells how the coefficients evolve under gradient descent update. In the following, we
introduce a two-stage dynamics to characterize the whole training process based on Eq 12 and Eq 13.

First Stage: Exponential growth. During the first stage, we show before γ(t)
r
or ρ(t)
r,i grow to Θ(1),
the embedding (3) is close to zero, suggesting the similarity is bounded by 1 ≤Simh(x, x′) ≤Cℓ
for some constant Cℓ> 1. The loss derivatives defined in (8) can thus be bounded within some
constant range.

Signal learning.
According to the update for signal learning in (12), we see the prop-
agation can be simplified based on the hard-negative sampling strategy, i.e., the negative
pairs do not share the same labels.
This suggests the negative term is always zero as
Pn
i=1
PM
j:yj̸=yi σ(⟨w(t)
r , yjµ⟩)σ′(⟨w(t)
r , yiµ⟩) = 0.
The resulting update of γ(t)
r
reduces to

γ(t+1)
r
= γ(t)
r +
η
nmτ
Pn
i=1(1 −ℓ′(t)
i
)σ(⟨w(t)
r , yiµ⟩)σ′(⟨w(t)
r , yiµ⟩)yi∥µ∥2
2. Examining the prop-
agation of γ(t)
r , we can divide the dynamics into two groups depending on the sign of weight
initialization ⟨w(0)
r , µ⟩. Let U(t)
+ ≜{r : ⟨w(t)
r , µ⟩> 0} and U(t)
−≜{r : ⟨w(t)
r , µ⟩< 0}. Then for
r ∈U(0)
+ , we can show γ(t)
r
≥0 increases exponentially and thus the sign of inner produce stays
invariant with U(t)
+ = U(0)
+
for all t ≥0. On the other hand, for r ∈U(0)
−, we can show γ(t)
r
≤0 and
decreases exponentially with U(t)
−= U(0)
−for all t ≥0.

Noise memorization. Compared to signal learning, the behaviour of noise memorization requires more
detailed analysis. This is mainly because the negative pairs can not be eliminated simply based on
label difference, as the noise patch ξi is generated independent of label yi. In addition, the added noise
ϵi by augmentation can also contribute to the noise dynamics. We first show when the noise level σϵ
is much smaller compared to σξ, the dynamics of noise memorization is largely remains unaffected.
By the sign of ⟨w(t)
r , ξi⟩, we partition the samples into two sets, i.e., I(t)
r,+ = {i : ⟨w(t)
r , ξi⟩> 0},

and I(t)
r,−= {i : ⟨w(t)
r , ξi⟩< 0}. We can verify for i ∈I(0)
r,−, the value of ρ(t)
r,i stays at zero based

on the update (13) with an induction argument. For samples i ∈I(0)
r,+ with positive initialization,
we analyze the noise memorization based on the joint dynamics of samples with the same label.
In particular, we define total noise memorization of positive and negative samples respectively as
B(t)
r,+ ≜P

i:yi=1(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+ and B(t)
r,−≜P

i:yi=−1(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1(t)
i∈Ir,+. The

update of ρ(t)
r,i in (13) then implies the dynamics of B(t)
r,+ and B(t)
r,−as follows

B(t+1)
r,+
≈B(t)
r,+ +
ησ2
ξd
nmτ
 
B(t)
r,+ −1

2B(t)
r,−

,
B(t+1)
r,−
≈B(t)
r,−+
ησ2
ξd
nmτ
 
B(t)
r,−−1

2B(t)
r,+

,

where the coefficient of 1/2 appears as a result of the randomness of the sign of initialization. This
result suggests, individual ρ(t)
r,i:yi=1 cannot grow too slow compared to the ρ(t)
r,i:yi=−1. Following a

similar induction argument, we are able to show ρ(t)
r,i:yi=1 has an exponential growth lower bound. On
the other hand, for samples with yi = −1 but with different neuron, we can use the same strategy to
show an exponential growth lower bound for some neurons that satisfy the initialization conditions.

Lemma 5.2. Under the Assumption 4.1, let T1 = log
 
20/(σ0σξ
√

d)

/ log
 
1 + 0.96
ησ2
ξd
nmτ

, we have
γ(t)
r
= eO(1/√n) for all r ∈[m] and 0 ≤t ≤T1 and maxr ρ(T1)
r,i
= Ω(1) for all i ∈[n].

Second stage: convergence and scale difference. At the end of first stage, the noise grows to a
constant order while signal learning remains negligible. As a result, the loss derivatives are no longer
bounded within some constant range. In the second stage, we aim to show the loss is able to converge
to an arbitrarily small value ϵ. Despite the unsupervised learning setup, we are still able to show loss
convergence thanks to the hard negative samples. Let F0(W, xi) = Sim(xi, bxi) be the similarity
to the argumentation and Fj(W, xi) = Sim(xi, xj) for j = 1, ..., M be the similarity between
the negative pairs. Then we can show there exists some W∗such that ⟨∇F0(W(t), xi), W∗⟩≥
2 log(2M/ϵ) while ⟨∇Fj(W(t), xi), W∗⟩≤log(2M/ϵ) for all j = 1, ..., M. Then we can bound
⟨∇LS(W(t)), W(t) −W∗⟩≥1

n
Pn
i=1 Li(W(t))−ϵ/2. This as a result allows to show a monotonic
decrease in the loss function as L(W(t)) ≤1

η(∥W(t) −W∗∥2
F −∥W(t+1) −W∗∥2
F ) + ϵ which
guarantees convergence by telescoping over the inequality. Upon the convergence, we can also show

7


---Page Break---
the scale difference obtained at the end of the first stage is maintained, i.e., γ(t)
r
= eO(1/√n) while
maxr ρ(T1)
r,i
= Ω(1) for all i ∈[n]. This suggests the non-linearly separability for the resulting
embeddings and thus the downstream test error is non-vanishing. The formal convergence result is
established in Lemma C.18. Combined with the generalization error demonstrated in Appendix C.3,
this completes the proof of Theorem 4.2.

5.2
Proof Sketch for Multi-Modal Learning

Similar to single-modal contrastive learning, we decompose the decomposition of weight vector
iteration for the network in the second modality and subsequently provide a two-stage analysis.

ew(t)
r
= ew(0)
r
+ eγ(t)
r ∥eµ∥−2
2 eµ +
X

i
eρ(t)
r,i∥eξi∥−2
2 eξi,
(14)

where eγ(t)
r
and eρ(t)
r,i serve as coefficients for the weight decomposition in the text modal.

Lemma 5.3 (Multi-Modal). The coefficients γ(t)
r , ρ(t)
r,i, eγ(t)
r , eρ(t)
r,i in decomposition (11) satisfy

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i=1
[(1 −ℓ′(t)
i
)h′
r(yiµ) −

M
X

j=1
ℓ′(t)
i,j h′
r(yiµ)]gr(yj eµ)yi∥µ∥2
2,
(15)

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)gr(eξi)h′
r(ξi)∥ξi∥2
2 −
η
nmτ

M
X

j=1
ℓ′(t)
i,j gr(eξj)h′
r(ξi)∥ξi∥2
2,
(16)

eγ(t+1)
r
= eγ(t)
r
+
η
nmτ

n
X

i=1
[(1 −ℓ′(t)
i
)h′
r(yiµ) −

M
X

j=1
ℓ′(t)
i,j h′
r(yiµ)]gr(yj eµ)yi∥eµ∥2
2,
(17)

eρ(t+1)
r,i
= eρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)hr(ξi)g′
r(eξi)∥eξi∥2
2 −
η
nmτ

M
X

j=1
ℓ′(t)
i,j hr(ξj)g′
r(eξi)∥eξi∥2
2,
(18)

where the initialization satisfy γ(0)
r , ρ(0)
r,i , eγ(0)
r , eρ(0)
r,i = 0

First Stage: Exponential growth The first stage of multi-modal learning shares similar characteristics
as single-modal learning.

Signal learning. For signal learning, we analyze the trajectories for both γ(t)
r
and eγ(t)
r . To this
end, we partition the neurons depending on their initialization status. Apart from U(t)
+ and U(t)
−
defined in the single-modal learning, we additionally define eU(t)
+
≜{r : ⟨ew(t)
r , eµ⟩> 0}, eU(t)
−
≜
{r : ⟨ew(t)
r , eµ⟩< 0} for the other modality. Then for r ∈U(0)
+
∩eU(0)
+ , we can show γ(t)
r
≥0,
ϕ(t)
r
≥0 and are increasing. For neurons r ∈U(t)
−∩eU(t)
−, we can show γ(t)
r
≤0, eγ(t)
r
≤0 and are
decreasing. Furthermore, the sign of inner product stays invariant, i.e., U(t)
+ ∩eU(t)
+ = U(0)
+ ∩eU(0)
+
and U(t)
−∩eU(t)
−= U(0)
−∩eU(0)
−. For neurons with only one of the modalities activated initially, i.e.,
r ∈U(0)
+ ∩eU(0)
−or r ∈U(0)
−∩eU(0)
+ , we can show there exists some time t′ ≥0 such that the neurons

are either positively or negatively activated with r ∈U(t′)
+
∩eU(t′)
+
and r ∈U(t′)
−
∩eU(t′)
−.

This shows synchronization of the signal learning patterns of two modalities. The pre-synchronization
phase reduces the speed of learning and thus we can only focus on neurons with the same sign for the
initialization in order to decide the lower and upper bound.

Noise memorization. For noise memorization, we partition the samples into two sets for both
modalities according to the initialization, namely, I(t)
r,+ = {i : ⟨w(t)
r , ξi⟩> 0}, I(t)
r,−= {i :

⟨w(t)
r , ξi⟩< 0}, and similarly for the second modality eI(t)
r,+, eI(t)
r,−. Because the noise memorization
are correlated in the two modalities, we separately analyze the samples as follows.

(1) For i ∈I(0)
r,−∩eI(0)
r,−, we can show ρ(t)
r,i = 0, I(t)
r,−= I(0)
r,−and eρ(t)
r,i = 0, eI(t)
r,−= eI(0)
r,−.

(2) For i ∈I(0)
r,−∩eI(0)
r,+, we can show ρ(t)
r,i = 0, I(t)
r,−= I(0)
r,−and eρ(t)
r,i ≤0.

(3) For i ∈I(0)
r,+ ∩eI(0)
r,−, we can show eρ(t)
r,i = 0, eI(t)
r,−= eI(0)
r,−and ρ(t)
r,i ≤0.

8


---Page Break---
0
50
100
150
200
Epoch

0

2

4

6

8

Training Loss

Single-Modal
Multi-Modal

0
50
100
150
200
Epoch

0.5

0.6

0.7

0.8

0.9

1.0
Test Accuracy

0
50
100
150
200
Epoch

0.0

0.2

0.4

0.6

0.8

1.0

Signal Learning

0
50
100
150
200
Epoch

0

1

2

3

Noise Memorization

Figure 1: Training loss, test accuracy, signal learning and noise memorization of single-modal and
multi-modal contrastive learning.

(4) For i ∈I(0)
r,+ ∩eI(0)
r,+, without loss of generality, we consider upper bounding noise memorization
for the first modality and yi = 1. To this end, we first define the individual and joint noise
memorization for the first modality as Ψ(t)
r,i ≜ρ(t)
r,i + ⟨w(0)
r , ξi⟩, and B(t)
r,+,+ ≜P

i:yi=1(ρ(t)
r,i +

⟨w(0)
r , ξi⟩)1i∈I(t)
r,+∩eI(t)
r,+, B(t)
r,+,−≜P

i:yi=1(ρ(t)
r,i +⟨w(0)
r , ξi⟩)1i∈I(t)
r,+∩eI(t)
r,−. Similar definitions
exist for the other modality. The joint dynamics of noise memorization can be first upper
bounded by the other modality as eΨ(t)
r,i ≥101Cℓ

M+1 (B(t)
r,+,+ + B(t)
r,+,−). Then we can upper bound
the individual noise memorization by

Ψ(t)
r,i ≤(1 +
1.06ησ2
ξd
nmτ
)t(Ψ(0)
r,i + eΨ(0)
r,i ).

We show the combined dynamics of γ(t)
r
and ρ(t)
r
exhibits exponential growth while the magnitude of
their difference shrinks exponentially. The results are summarized as follows

Lemma 5.4. Under the Assumption 4.1, let T1 = log
 
20/(σ0∥µ∥2)

/ log
 
1 + 0.48Cµ
η∥µ∥2
2
mτ

, we
have ρ(T1)
r,i
= eO(1/√n) for all r ∈[m], i ∈[n], and 0 ≤t ≤T1 and maxr γ(T1)
r
= Ω(1).

Second Stage: Convergence and scale difference. The second stage presents similar patterns
compared to single-modal learning. Thanks to the correlation between the two modality during
gradient descent training, the two neural network converge at the same time, minimizing the training
loss. Besides, The scale difference at the end of the first stage is carried over throughout the second
stage until convergence. Therefore, it allows to show a monotonic decrease in the loss function
as L(W(t), f
W(t)) ≤∥W(t) −W∗∥2
F + ∥f
W(t) −f
W∗∥2
F −∥W(t+1) −W∗∥2
F −∥f
W(t+1) −
f
W∗∥2
F + 2ϵ, which guarantees convergence by telescoping over the inequality. At the same time,
until convergence, we can show the scale difference obtained at the end of the first stage is maintained,
namely maxr,i ρ(t)
r,i = eO(1/√n) and maxr γ(t)
r
= Ω(1). This suggests the signal learning dominates
the noise memorization and thus the resulting embeddings are linearly separable, which guarantees a
small test error for downstream tasks. The formal convergence result is established in Lemma D.15.
Combined with the generalization error demonstrated in Appendix D.3, this completes the proof of
Theorem 4.3.

6
Experiments

Synthetic experiments
We conduct synthetic experiments to verify the theoretical results obtained
in the previous sections. We generate samples following the theoretical setups, where we set the data
dimension d = 2000, number of training samples n = 100, number of test samples ntest = 200,
and the hidden size of all encoders as m = 50. We adopt gradient descent with a learning rate
of 0.01 as the optimizer to train the model by 200 epochs. In the single-modal setting, the µ
is set to be [5, 0, ..., 0]T and the ξ ∼N(0, I) for the in-distribution data, and the augmentation
vector ϵ ∼N(0, 0.01 ∗I). For the multi-modal setting, ˜µ = [0, 15, 0, ..., 0]T . In addition, for the
OOD test data xtest = [ν⊤, ζ⊤]⊤, we set ν = [2, 0, ..., 0] and ζ ∼N(0, I). We perform logistic
regression based on the learned features h(xtest) and apply the learned classifier head to evaluate
OOD generalization error in terms of prediction accuracy.

Results. In Figure 1, we see the training loss of both single-modal and multi-modal learning converges
rapidly. At the same time, OOD test accuracy of multi-modal learning converges to nearly 1.0 while

9


---Page Break---
that of single-modal learning stagnates around 0.5. This is primarily because under the setup where
the other modality ˜µ has a higher SNR, signal learning of µ is lifted. This can be verified from the
third plot of Figure 1, where the signal learning of multi-modal framework is significantly higher
than single-modal. Further, it can be observed that single-modal contrastive learning exhibits more
severe noise memorization, which suppresses signal learning. In contrast, multi-modal contrastive
learning exhibits less severe noise memorization which would further encourage signal learning.
These phenomena again support and align with our theoretical results.

Real-world experiments
We now extend the comparison of single-modal and multi-modal learning
to realistic image data, ColoredMNIST [3, 54], which is a typical benchmark studying the generaliza-
tion capability under distribution shifts. The ColoredMNIST dataset is a variation of the standard
MNIST dataset, where each digit is assigned a specific color based on its label. The two modalities
are image, and text that describes the images. The task is a 10-class classification that recognizes
the number of the colored MNIST images. Specifically, we have 10 colors to color 10 digits, and
introduce spurious correlations via label noises following the literature:

• For the training set, 10% of labels will be clipped to a random class. For images with class ‘0’ (or
‘1’), they will be colored as red (or green) with a probability of 77.5%, and as another random
color with a probability of 22.5%. The coloring scheme introduces a spurious correlation.
• For the test set, 10% of labels will be clipped to a random class. For images with class ‘0’ (or ‘1’),
they will be colored as green (or red) with a probability of 77.5%, and as another random color
with a probability of 22.5%. The coloring scheme can be considered as reversing the training
spurious correlations. Therefore, the evaluation on test set can reflect to what extent the model
learns to use the spurious features, i.e., colors, to classify images.

We implement the multi-modal learning following the practice in [54], where we consider
an ideal language encoder that successfully encodes the caption of the images into one-
hot labels of colors and digits.
For single-modal learning, we follow the implementa-
tion of the SimCLR [9] to construct a set of augmentations to learn the representations.

Table 1: Performance comparison for single and multi-
modal contrastive learning.

Model
Train Accuracy
Test Accuracy

Single-modal
88.43%
12.68%
Multi-modal
87.77%
82.13%

Results. Under the distribution shift, we
verify that multi-modal learning archives
an out-of-distribution test accuracy of
82.13%, which outperforms that of single-
modal learning 12.68%. As a result, we
can claim that the effective SNR of invari-
ant features (the shape of the digit) will be
degraded under the impact of the injected
color. Therefore, the performance of single-modal may be suboptimal as it cannot effectively utilize
the information of the digit’s shape. On the other hand, multi-modal demonstrates a better capacity
for handling this scenario.

7
Conclusions

In this work, we have established a comprehensive comparison of the optimization differences during
the pre-training stage and the generalization gap between single-modal and multi-modal contrastive
learning for downstream tasks. With the cooperation between modalities, multi-modal contrastive
learning can achieve better feature learning and generalization on downstream tasks compared to
single-modal learning. On the other hand, data augmentation alone can hardly improve data quality
and thus cannot boost the performance of single-modal contrastive learning. Together, these results
quantitatively demonstrate the superiority of multi-modal learning over single-modal learning and
emphasize the importance of data quality in multi-modal contrastive learning.

Acknowledgments and Disclosure of Funding

We thank the anonymous reviewers for their insightful comments to improve the paper. Wei Huang is
supported by JSPS KAKENHI Grant Number 24K20848. Yuan Cao is supported by NSFC 12301657
and HK RGC-ECS 27308624. Taiji Suzuki is partially supported by JSPS KAKENHI (24K02905)
and JST CREST (JPMJCR2015).

10


---Page Break---
References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson,
Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual
language model for few-shot learning. Advances in Neural Information Processing Systems,
35:23716–23736, 2022.

[2] Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation
and self-distillation in deep learning. arXiv preprint arXiv:2012.09816, 2020.

[3] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk mini-
mization. arXiv preprint arXiv:1907.02893, 2019.

[4] Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, and Nikunj
Saunshi. A theoretical analysis of contrastive unsupervised representation learning. arXiv
preprint arXiv:1902.09229, 2019.

[5] Randall Balestriero and Yann LeCun. Contrastive and non-contrastive self-supervised learn-
ing recover global and local spectral embedding methods. Advances in Neural Information
Processing Systems, 35:26671–26685, 2022.

[6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz
Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In
Advances in Neural Information Processing Systems, 2020.

[7] Vivien Cabannes, Bobak Kiani, Randall Balestriero, Yann LeCun, and Alberto Bietti. The ssl
interplay: Augmentations, inductive bias, and generalization. In International Conference on
Machine Learning, pages 3252–3298. PMLR, 2023.

[8] Yuan Cao, Zixiang Chen, Misha Belkin, and Quanquan Gu. Benign overfitting in two-layer
convolutional neural networks. Advances in neural information processing systems, 35:25237–
25250, 2022.

[9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework
for contrastive learning of visual representations. In International conference on machine
learning, pages 1597–1607. PMLR, 2020.

[10] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758,
2021.

[11] Yongqiang Chen, Wei Huang, Kaiwen Zhou, Yatao Bian, Bo Han, and James Cheng. Under-
standing and improving feature learning for out-of-distribution generalization. In Advances in
Neural Information Processing Systems, 2023.

[12] Zixiang Chen, Yihe Deng, Yuanzhi Li, and Quanquan Gu. Understanding transferable represen-
tation learning and zero-shot transfer in clip. arXiv preprint arXiv:2310.00927, 2023.

[13] Imant Daunhawer, Alice Bizeul, Emanuele Palumbo, Alexander Marx, and Julia E Vogt.
Identifiability results for multimodal contrastive learning. arXiv preprint arXiv:2303.09166,
2023.

[14] Yann Dubois, Stefano Ermon, Tatsunori B Hashimoto, and Percy S Liang. Improving self-
supervised learning by characterizing idealized representations. Advances in Neural Information
Processing Systems, 35:11279–11296, 2022.

[15] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving CLIP
training with language rewrites. In Advances in Neural Information Processing Systems, 2023.

11


---Page Break---
[16] Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave,
and Ludwig Schmidt. Data determines distributional robustness in contrastive language image
pre-training (CLIP). In International Conference on Machine Learning, 2022.

[17] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and
Vaishaal Shankar. Data filtering networks. In NeurIPS 2023 Workshop on Distribution Shifts:
New Frontiers with Foundation Models, 2023.

[18] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao
Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim
Entezari, Giannis Daras, Sarah M Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe,
Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga
Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont,
Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt.
Datacomp: In search of the next generation of multimodal datasets. In Thirty-seventh Conference
on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

[19] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena
Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar,
et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural
information processing systems, 33:21271–21284, 2020.

[20] Jeff Z HaoChen and Tengyu Ma. A theoretical study of inductive biases in contrastive learning.
arXiv preprint arXiv:2211.14699, 2022.

[21] Jeff Z HaoChen, Colin Wei, Adrien Gaidon, and Tengyu Ma. Provable guarantees for self-
supervised deep learning with spectral contrastive loss. Advances in Neural Information
Processing Systems, 34:5000–5011, 2021.

[22] Wei Huang, Yuan Cao, Haonan Wang, Xin Cao, and Taiji Suzuki. Graph neural networks
provably benefit from structural information: A feature learning perspective. arXiv preprint
arXiv:2306.13926, 2023.

[23] Wei Huang, Ye Shi, Zhongyi Cai, and Taiji Suzuki. Understanding convergence and gener-
alization in federated learning through feature learning theory. In The Twelfth International
Conference on Learning Representations, 2023.

[24] Weiran Huang, Mingyang Yi, Xuyang Zhao, and Zihao Jiang. Towards the generalization of
contrastive self-supervised learning. arXiv preprint arXiv:2111.00743, 2021.

[25] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. What
makes multi-modal learning better than single (provably). Advances in Neural Information
Processing Systems, 34:10944–10956, 2021.

[26] Samy Jelassi and Yuanzhi Li. Towards understanding how momentum improves generalization
in deep learning. In International Conference on Machine Learning, pages 9965–10040. PMLR,
2022.

[27] Wenlong Ji, Zhun Deng, Ryumei Nakada, James Zou, and Linjun Zhang. The power of
contrast for feature learning: A theoretical analysis. Journal of Machine Learning Research,
24(330):1–78, 2023.

[28] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan
Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning
with noisy text supervision. In International conference on machine learning, pages 4904–4916.
PMLR, 2021.

[29] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron
Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural
information processing systems, 33:18661–18673, 2020.

[30] Yiwen Kou, Zixiang Chen, Yuan Cao, and Quanquan Gu. How does semi-supervised learing
with pseudo-labelers work? a case study. In International Conference on Learning Representa-
tions, 2023.

12


---Page Break---
[31] Yiwen Kou, Zixiang Chen, Yuanzhou Chen, and Quanquan Gu. Benign overfitting for two-layer
relu networks. arXiv preprint arXiv:2303.04145, 2023.

[32] Jason D Lee, Qi Lei, Nikunj Saunshi, and Jiacheng Zhuo. Predicting what you already know
helps: Provable self-supervised learning. Advances in Neural Information Processing Systems,
34:309–323, 2021.

[33] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
Blip: Bootstrapping language-
image pre-training for unified vision-language understanding and generation. In International
Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

[34] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind
the gap: Understanding the modality gap in multi-modal contrastive representation learning.
Advances in Neural Information Processing Systems, 35:17612–17625, 2022.

[35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In
NeurIPS, 2023.

[36] Prasanna Mayilvahanan, Thaddäus Wiedemer, Evgenia Rusak, Matthias Bethge, and Wieland
Brendel. Does clip’s generalization performance mainly stem from high train-test similarity?
arXiv preprint arXiv:2310.09562, 2023.

[37] Yifei Ming and Yixuan Li. Understanding retrieval-augmented task adaptation for vision-
language models. arXiv preprint arXiv:2405.01468, 2024.

[38] Ryumei Nakada, Halil Ibrahim Gulluk, Zhun Deng, Wenlong Ji, James Zou, and Linjun Zhang.
Understanding multimodal contrastive learning and incorporating unpaired data. In International
Conference on Artificial Intelligence and Statistics, pages 4348–4380. PMLR, 2023.

[39] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt.
Improving multimodal datasets with image captioning. In Thirty-seventh Conference on Neural
Information Processing Systems Datasets and Benchmarks Track, 2023.

[40] Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality
not quantity: On the interaction between dataset design and robustness of CLIP. In Advances in
Neural Information Processing Systems, 2022.

[41] OpenAI. Gpt-4 technical report, 2023.

[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning,
pages 8748–8763. PMLR, 2021.

[43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical
text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

[44] Yunwei Ren and Yuanzhi Li. On the importance of contrastive loss in multimodal learning.
arXiv preprint arXiv:2304.03717, 2023.

[45] Shibani Santurkar, Yann Dubois, Rohan Taori, Percy Liang, and Tatsunori Hashimoto. Is
a caption worth a thousand images? a study on representation learning. In International
Conference on Learning Representations, 2023.

[46] Nikunj Saunshi, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham
Kakade, and Akshay Krishnamurthy. Understanding contrastive learning requires incorporating
inductive biases. In International Conference on Machine Learning, pages 19250–19286.
PMLR, 2022.

[47] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman,
Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick
Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmar-
czyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation
image-text models. In Thirty-sixth Conference on Neural Information Processing Systems
Datasets and Benchmarks Track, 2022.

13


---Page Break---
[48] James B Simon, Maksis Knutins, Liu Ziyin, Daniel Geisz, Abraham J Fetterman, and Joshua
Albrecht. On the stepwise nature of self-supervised learning. In International Conference on
Machine Learning, pages 31852–31876. PMLR, 2023.

[49] Yuandong Tian. Understanding deep contrastive learning via coordinate-wise optimization.
Advances in Neural Information Processing Systems, 35:19511–19522, 2022.

[50] Yuandong Tian, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning
dynamics without contrastive pairs. In International Conference on Machine Learning, pages
10268–10278. PMLR, 2021.

[51] Yuandong Tian, Lantao Yu, Xinlei Chen, and Surya Ganguli. Understanding self-supervised
learning with dual deep networks. arXiv preprint arXiv:2010.00578, 2020.

[52] Christopher Tosh, Akshay Krishnamurthy, and Daniel Hsu. Contrastive learning, multi-view
redundancy, and linear models. In Algorithmic Learning Theory, pages 1179–1206. PMLR,
2021.

[53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timo-
thée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open
and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[54] Qizhou Wang, Yong Lin, Yongqiang Chen, Ludwig Schmidt, Bo Han, and Tong Zhang. Do
clip models always generalize better than imagenet models? Advances in Neural Information
Processing Systems, 2024.

[55] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through
alignment and uniformity on the hypersphere. In International Conference on Machine Learning,
pages 9929–9939. PMLR, 2020.

[56] Zixin Wen and Yuanzhi Li. Toward understanding the feature learning process of self-supervised
contrastive learning. In International Conference on Machine Learning, pages 11112–11122.
PMLR, 2021.

[57] Yihao Xue, Siddharth Joshi, Eric Gan, Pin-Yu Chen, and Baharan Mirzasoleiman. Which
features are learnt by contrastive learning? on the role of simplicity bias in class collapse and
feature suppression. In International Conference on Machine Learning, pages 38938–38970.
PMLR, 2023.

[58] Yihao Xue, Siddharth Joshi, Dang Nguyen, and Baharan Mirzasoleiman. Understanding
the robustness of multi-modal contrastive learning to distribution shift.
arXiv preprint
arXiv:2310.04971, 2023.

[59] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong
Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for
computer vision. arXiv preprint arXiv:2111.11432, 2021.

[60] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When
and why vision-language models behave like bags-of-words, and what to do about it? arXiv
e-prints, pages arXiv–2210, 2022.

[61] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for
vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022.

[62] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: En-
hancing vision-language understanding with advanced large language models. arXiv preprint
arXiv:2304.10592, 2023.

[63] Difan Zou, Yuan Cao, Yuanzhi Li, and Quanquan Gu. Understanding the generalization of
adam in learning neural networks with proper regularization. In International Conference on
Learning Representations, 2023.

14


---Page Break---
Appendix

Contents

A Limitations and broader impact
16

B
Preliminary Lemmas
16

C Single-modal Contrastive Learning: Proof of Theorem 4.2
17

C.1
First Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

C.1.1
Dynamics of Signal Learning: Upper Bound
. . . . . . . . . . . . . . . .
20

C.1.2
Dynamics of Noise Memorization: Lower Bound . . . . . . . . . . . . . .
23

C.1.3
Noise Memorization: Proof of Lemma 5.2 . . . . . . . . . . . . . . . . . .
28

C.2
Second Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

C.3
Downstream Task Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36

D Multi-Modal Contrastive Learning: Proof of Theorem 4.3
36

D.1
First Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36

D.1.1
Dynamics of Signal Learning: Lower Bound
. . . . . . . . . . . . . . . .
37

D.1.2
Dynamics of Noise Memorization: Upper Bound . . . . . . . . . . . . . .
40

D.1.3
Signal Learning: Proof of Lemma 5.4 . . . . . . . . . . . . . . . . . . . .
46

D.2
Second Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47

D.3
Downstream Task Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51

E Additional Experimental Details
51

15


---Page Break---
A
Limitations and broader impact

While our theoretical analysis is novel in terms of optimization and generalization, the data model can
be further modified to be more practical. Our theoretical analysis may be further used for empirical
and theoretical studies of contrastive learning, especially multi-modal contrastive learning. However,
we do not foresee a direct social impact from our theory.

B
Preliminary Lemmas

Before the proof, we introduce lemmas that are useful in proving our main theorem.

Lemma B.1. Let x ∼N(0, σ2). Then P(|x| ≤c) = 2erf

c
√

2σ


≤2
q

1 −exp(−2c2

σ2π).

Proof. The probability density function for x is given by

f(x) =
1
√

2πσ exp

−x2

2σ2


.

Then we know that

P(|x| ≤c) =
1
√

2πσ

Z c

−c
exp

−x2

2σ2


dx.

By the definition of erf function

erf(c) =
2
√π

Z c

0
exp(−x2)dx,

and variable substitution yields

erf

c
√

2σ


=
1
√

2πσ

Z c

0
exp

−x2

2σ2


dx.

Therefore, we first conclude P(|x| ≤c) = 2erf

c
√

2σ


.

Next, by the inequality erf(x) ≤
p

1 −exp(−4x2/π), we finally obtain

P(|x| ≤c) ≤2

s

1 −exp

−2c2

σ2π


.

Lemma B.1 introduces an anti-concentration result. In later sections, this lemma will be used to show
that with a relatively large initialization for the weight vector, some initial properties hold.

Lemma
B.2.
Under
condition
that
d
≥
400n
σ0σξ

q

log(6n/δ)
−π log(1−δ2/(4m2)),
and
ed
≥

400n
σ0σ e
ξ

q

log(6n/δ)
−π log(1−δ2/(4m2)), then with probability at least 1 −δ, we can show for all r ∈[m],

|⟨w(0)
r , µ⟩| ≥100 · SNR

r

8 log(6n/δ)

d
n,

|⟨ew(0)
r , eµ⟩| ≥100 · SNR

s

8 log(6n/δ)

ed
n.

Proof of Lemma B.2. By Lemma B.1, because ⟨w(0)
r , µ⟩∼N(0, σ2
0∥µ∥2
2), we can show

P
 
|⟨w(0)
r , µ⟩| ≤c

≤2

s

1 −exp

−
2c2

σ2
0∥µ∥2
2π


.

16


---Page Break---
Let c = 100 · SNR
q

8 log(6n/δ)

d
n = 100n∥µ∥2σ−1
ξ d−1p

8 log(6n/δ) and plug it into the RHS of
the above inequality, which becomes:

RHS = 2

v
u
u
t1 −exp

 

−160000 log(6n/δ)n2

σ2
0σ2
ξd2π

!

.

Then we can verify that when d satisfies that d ≥400n

σ0σξ

q

log(6n/δ)
−π log(1−δ2/(4m2)), it holds that RHS ≤

δ/m. This suggests for a single neuron r ∈[m], we have P(|⟨w(0)
r , µ⟩| ≤c) ≤δ/m. Applying
union bound, we can show the desired result.

Similarly, with the same procedure, we can prove the result for the other modality.

|⟨ew(0)
r , eµ⟩| ≥100 · SNR

s

8 log(6n/δ)

ed
n.

Lemma B.3 ([31]). Let S1 = {i ∈[n] : yi = 1} and S−1 = {i ∈[n] : yi = −1}. Then with
probability at least 1 −δ,

|S1|, |S−1| ∈
n

2 −
rn

2 log(4/δ), n

2 +
rn

2 log(4/δ)

.

Lemma B.3 states that when the label is randomly sampled, the number of positive samples and
negative samples is close to n

2 , adequately.
Lemma B.4 ([8]). Suppose that d ≥Ω(log(mn/δ)), m = Ω(log(1/δ)). Then with probability at
least 1 −δ, it satisfies that for all r ∈[m], i ∈[n],

|⟨w(0)
r , µ⟩| ≤
p

2 log(8m/δ)σ0∥µ∥2

|⟨w(0)
r , ξi⟩| ≤2
p

log(8mn/δ)σ0σξ
√

d

|⟨w0
r, ϵi⟩| ≤2
p

log(8mn/δ)σ0σϵ
√

d.

and for all i ∈[n]

σ0∥µ∥2/2 ≤max
r∈[m]⟨w(0)
r , µ⟩≤
p

2 log(8m/δ)σ0∥µ∥2

σ0σξ
√

d/4 ≤max
r∈[m]⟨w(0)
r , ξi⟩≤2
p

log(8mn/δ)σ0σξ
√

d.

Lemma B.5 ([8]). Suppose that δ > 0 and d = Ω(log(6n/δ))). Then with probability 1 −δ,

σ2
ξd/2 ≤∥ξi∥2
2 ≤3σ2
ξd/2,

|⟨ξi, ξi′⟩| ≤2σ2
ξ
p

d log(6n2/δ)

|⟨ξi, µ⟩| ≤∥µ∥2σξ
p

2 log(6n/δ)

σ2
ϵ d/2 ≤∥ϵi∥2
2 ≤3σ2
ϵ d/2,

|⟨ϵi, ξi′⟩| ≤2σϵσξ
p

d log(6n2/δ)

|⟨ϵi, µ⟩| ≤∥µ∥2σϵ
p

2 log(6n/δ)

for all i, i′ ∈[n].

C
Single-modal Contrastive Learning: Proof of Theorem 4.2

In this section, we provide the proof for Theorem 4.2, which states main results of single modal
learning. The training dynamics are based on the coefficient iterations presented in Lemma 5.1.
Below, we provide a proof for this lemma:

17


---Page Break---
Proof of Lemma 5.1. Recall that the weight decomposition is expressed as

w(t)
r
= w(0)
r
+ γ(t)
r ∥µ∥−2
2 µ +

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ξi.

We plug it into the gradient descent update as described by Equation 10 yields

w(t+1)
r
= w(0)
r
+ γ(t+1)
r
∥µ∥−2
2 µ +

n
X

i=1
ρ(t+1)
r,i
∥ξi∥−2
2 ξi

= w(0)
r
+ γ(t)
r ∥µ∥−2
2 µ +

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ξi +
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)h(t)
r (bx(1)
i )h′(t)(x(1)
i )yiµ

+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)h(t)
r (bx(2)
i )h′(t)
r
(x(2)
i )ξi −
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(1)
j )h′(t)(x(1)
i )yiµ

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(2)
j )h′(t)
r
(x(2)
i )ξi.

By comparing the coefficients in front of µ and ξi on both sides of the equation, we can obtain

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i=1


(1 −ℓ′(t)
i
)h(t)
r (bx(1)
i ) −

M
X

j̸=i
ℓ′(t)
i,j h(t)
r (x(1)
j )

h′(t)
r
(x(1)
i )yi∥µ∥2
2,

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ

(1 −ℓ′(t)
i
)hr(bx(2)
i ) −

M
X

j̸=i
ℓ′(t)
i,j hr(x(2)
j )

h′(t)
r
(x(2)
i )∥ξi∥2
2,

which completes the proof.

According to the behavior of the defined loss derivative (8), we split the entire training dynamics into
two phases. In the first stage, the loss derivative remains close to its initial value as the similarity is
small from initialization. Later, as the similarity grows to a constant value, the loss derivative is no
longer close to the initial value, and the dynamics transition to the second stage. In this stage, the
similarity increases logarithmically, and the empirical loss converges.

C.1
First Stage

In the first stage, the derivative of the loss is close to its initial value because the similarity is small.
Below, we provide a useful lemma for establishing such a result.

Lemma C.1. Suppose that γ(t)
r
= O(1) and ρ(t)
r,i = O(1) for all r ∈[m] and i ∈[n]. Under
Assumption 4.1, then for any δ > 0, with probability at least 1 −δ

|⟨w(t)
r
−w(0)
r , ξi⟩−ρ(t)
r,i| ≤5

r

log(6n2/δ)

d
n

|⟨w(t)
r
−w(0)
r , µ⟩−γ(t)
r | ≤SNR

r

8 log(6n/δ)

d
n

for all r ∈[m], i ∈[n].

Proof of Lemma C.1. From the signal-noise decomposition of w(t)
r , we derive

|⟨w(t)
r
−w(0)
r , ξi⟩−ρ(t)
r,i|
(a)
= |γ(t)
r ⟨µ, ξi⟩∥µ∥−2
2
+

n
X

i′=1
ρ(t)
r,i⟨ξi′, ξi⟩∥ξi′∥−2
2 |

(b)
≤∥µ∥−1
2 σξ
p

2 log(6n/δ) + 4

r

log(6n2/δ)

d
n

18


---Page Break---
(c)
≤5

r

log(6n2/δ)

d
n.

Equation (a) results from the weight decomposition (see Equation 11). In the first stage, we used the
upper bounds for |γ(t)
r | and |ρ(t)
r |, and applied Lemma B.5 in inequality (b). Finally, inequality (c)
follows from the condition nSNR2 = Θ(1).

Further,

|⟨w(t)
r
−w(0)
r , µ⟩−γ(t)
r | = |

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ⟨ξi, µ⟩| ≤2n · SNR

r

2 log(6n/δ)

d
,

where we have used Lemma B.5.

Now, we proceed to the lemma concerning the derivative of the loss as follows:

Lemma C.2. If max{γ(t)
r , ρ(t)
r,i} = O(1) and under Assumption 4.1, there exists a constant Cℓ> 1
such that
1
Cℓ(1 + M) ≤ℓ′(t)
i
≤
Cℓ
1 + M ,
1
Cℓ(M + 1) ≤ℓ′(t)
i,j ≤
Cℓ
1 + M ,

for all i ∈[n].

Proof of Lemma C.2. From the update of w(t)
r , we have

|⟨w(t)
r , ξi⟩|
(a)
≤|⟨w(0)
r , ξi⟩| + ρ(t)
r,i + 5

r

log(6n2/δ)

d
n

(b)
≤2
p

log(8mn/δ)σ0σξ
√

d + ρ(t)
r,i + 5

r

log(6n2/δ)

d
n

(c)
= O(1),

where (a) is by Lemma C.1, and (b) is by Lemma B.4. Finally, in inequality (c) we have used the
condition that σ0 ≤
1
2√

log(8mn/δ)σξ
√

d and d > n2 log(6n2/δ) according to Assumption 4.1, and

max{γ(t)
r , ρ(t)
r,i} = O(1). At the same time,

|⟨w(t)
r , µ⟩|
(a)
≤|⟨w(0)
r , µ⟩| + γ(t)
r
+ SNR

r

8 log(6n/δ)

d
n

(b)
≤
p

2 log(8m/δ)σ0∥µ∥2 + SNR

r

8 log(6n/δ)

d
n

(c)
= O(1),

where (a) is by Lemma C.1, (b) is Lemma B.4, and (c) is by σ0 ≤
1
2√

log(8m/δ)∥µ∥2 and d >

SNR2n2 log(6n2/δ) according to Assumption 4.1, and max{γ(t)
r , ρ(t)
r,i} = O(1). Besides,

|⟨w(t)
r , ϵi⟩| = |⟨w(0)
r , ϵi⟩+ γ(t)
r ∥µ∥−2
2 ⟨µ, ϵi⟩+

n
X

i=i
ρ(t)
r,i∥ξi∥−2
2 ⟨ξi, ϵi⟩|

(a)
≤|⟨w(0)
r , ϵi⟩| + ∥µ∥−1
2 σϵ
p

2 log(6n/δ) + 4

r

log(6n2/δ)

d
σϵσ−1
ξ n

(b)
≤2
p

log(8mn/δ)σ0σϵ
√

d + ∥µ∥−1
2 σϵ
p

2 log(6n/δ) + 4

r

log(6n2/δ)

d
σϵσ−1
ξ n

(c)
= O(1),

where (a) follows from Lemma B.5, (b) from Lemma B.4, and (c) from the conditions σ0 ≤
1
2√

log(8mn/δ)σϵ
√

d, σϵ ≤
∥µ∥2
√

2 log(6n/δ), d > n2 log(6n2/δ), and σϵ < σξ.

19


---Page Break---
Next, we calculate the upper bound of the similarity measure. First, we examine the negative pair.
For any i, j ∈[n], we have

Simh(xi, xj) = 1

m⟨h(x(1)
i ), sg(h(x(1)
j ))⟩+ 1

m⟨h(x(2)
i ), sg(h(x(2)
j ))⟩

= 1

m

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξj⟩) + 1

m

m
X

r=1
σ(⟨w(t)
r , yiµ⟩)σ(⟨w(t)
r , yjµ⟩)

≤max{|⟨w(t)
r , yiµ⟩⟨w(t)
r , yjµ⟩|, |⟨w(t)
r , ξi⟩⟨w(t)
r , ξj⟩|} = O(1).

Similarly, for positive pair,

Simh(xi, bxj) = 1

m

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩) + 1

m

m
X

r=1
σ(⟨w(t)
r , yiµ⟩)σ(⟨w(t)
r , yiµ⟩)

≤max{|⟨w(t)
r , yiµ⟩⟨w(t)
r , yjµ⟩|, |⟨w(t)
r , ξi⟩⟨w(t)
r , ξi + ϵi⟩|} = O(1).

According to the above result, we can say that 1 ≤eSimh(x,x′) ≤Cℓ, where Cℓis a positive constant.
Then we can provide the upper bound for ℓ′
i and ℓ′
i,j

ℓ′(t)
i
=
eSimh(xi,bxi)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ ≤
Cℓ
1 + M ,

ℓ′(t)
i
=
eSimh(xi,bxi)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ ≥
1
Cℓ(1 + M),

ℓ′(t)
i,j =
eSimh(xi,xj)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ ≥
1
Cℓ(M + 1),

ℓ′(t)
i,j =
eSimh(xi,xj)/τ

eSimh(xi,bxi)/τ + PM
j̸=i eSimh(xi,xj)/τ ≤
Cℓ
M + 1.

This completes the proof.

C.1.1
Dynamics of Signal Learning: Upper Bound

In the first stage, the growth rate of signal learning is exponential. We establish an upper bound for
the growth of signal learning.

Then we consider the growth of signal learning coefficient γ(t)
r . Depending on the initialization, we
define U(t)
+ = {r ∈[m] : ⟨w(t)
r , µ⟩> 0} and U(t)
−= {r ∈[m] : ⟨w(t)
r , µ⟩< 0}.

Lemma C.3. Under the condition d ≥400n

σ0σξ

q

log(6n/δ)
−π log(1−δ2/(4m2)) and Assumption 4.1, for all t ≥0,

we have U(t)
+
= U(0)
+ , U(t)
−
= U(0)
−
and γ(t)
r
> 0 is an increasing sequence for all r ∈U(0)
+
and
γ(t)
r
≤0 and is a decreasing sequence for all r ∈U(0)
−.

Proof of Lemma C.3. We prove the claims by induction. To better understand the dynamics, we first
derive the propagation for signal learning from the first step. For r ∈U(0)
+ , i.e., ⟨w(0)
r , µ⟩> 0, we
can see

γ(1)
r
= γ(0)
r +
η
nmτ

n
X

i=1
(1 −ℓ′(0)
i
)σ(⟨w(0)
r , yiµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(0)
i,j σ(⟨w(0)
r , yjµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

= γ(0)
r +
η
nmτ

n
X

i:yi=1
(1 −ℓ′(0)
i
)σ(⟨w(0)
r , yiµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

20


---Page Break---
−
η
nmτ

n
X

i:yi=1

M
X

j:yj=−1
ℓ′(0)
i,j σ(⟨w(0)
r , yjµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

=
η
nmτ

n
X

i:yi=1
(1 −ℓ′(0)
i
)⟨w(0)
r , µ⟩∥µ∥2
2 > 0,

where last equality is by ⟨w(0)
r , µ⟩> 0, γ(0)
r
= 0, and the fact that negative samples satisfy yj ̸= yi.
Thus, we verify that the sign of γ(1)
r
follow its initialization and γ(1)
r
> 0.

Next, we show the propagation of inner product at t = 1, for r ∈U(0)
+ , we have

⟨w(1)
r , µ⟩
(a)
≥⟨w(0)
r , µ⟩+ γ(1)
r
−SNR

r

8 log(6n/δ)

d
n

(b)
≥0.99⟨w(0)
r , µ⟩+ γ(1)
r
> 0,

where inequality (a) is by Lemma C.1, the second inequality (b) is by Lemma B.2, and the last by
γ(1)
r
> 0. Hence we verify U(1)
+
= U(0)
+ .

Now suppose at iteration t, the claims are satisfied, namely ⟨w(t)
r , µ⟩> 0 and γ(t)
r
≥γ(t−1)
r
≥0 for
r ∈U(0)
+ . Then following similar argument, for r ∈U(0)
+

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i:yi=1
(1 −ℓ′(t)
i
)⟨w(t)
r , µ⟩∥µ∥2
2 ≥γ(t)
r
≥0,

where we use the induction condition that ⟨w(t)
r , µ⟩> 0. Further by Lemma C.1 and B.2

⟨w(t+1)
r
, µ⟩≥0.99⟨w(0)
r , µ⟩+ γ(t+1)
r
> 0,

where we use γ(t+1)
r
≥0. This completes the induction for r ∈U(0)
+ .

Similarly, for those neuron r that satisfies ⟨w0
r, µ⟩< 0, i.e., r ∈U(0)
−, we have

γ(1)
r
= γ(0)
r +
η
nmτ

n
X

i=1
(1 −ℓ′(0)
i
)σ(⟨w(0)
r , yiµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(0)
i,j σ(⟨w(0)
r , yjµ⟩)σ′(⟨w(0)
r , yiµ⟩)yi∥µ∥2
2

= −
η
nmτ

n
X

i:yi=−1
(1 −ℓ′(0)
i
)⟨w(0)
r , yiµ⟩∥µ∥2
2 < 0,

where the last equality is by ⟨w(0)
r , µ⟩< 0, yi = −1, the property of ReLU activation, and the fact
that yj ̸= yi in the negative pair term. Hence we see γ(1)
r
≤γ(0)
r
= 0.

Similarly, for the inner product at t = 1,

⟨w(1)
r , µ⟩= ⟨w(0)
r , µ⟩+ γ(1)
r
+

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ⟨ξi, µ⟩

≤⟨w(0)
r , µ⟩+ γ(1)
r
+ SNR

r

8 log(6n/δ)

d
n

≤−0.99|⟨w(0)
r , µ⟩| + γ(1)
r
< 0,

where the first inequity is by Lemma C.1, the second inequality follows from Lemma B.2, and the
last inequality follows from γ(1)
r
≤0.

21


---Page Break---
Now suppose at iteration t, the claims are satisfied, namely ⟨w(t)
r , µ⟩< 0 and γ(t)
r
≤γ(t−1)
r
≤0 for
r ∈U(0)
+ . Then following similar argument, for r ∈U(0)
−

γ(t+1)
r
= γ(t)
r
−
η
nmτ

n
X

i:yi=−1
(1 −ℓ′(t)
i
)⟨w(t)
r , µ⟩∥µ∥2
2 ≤γ(t)
r
≤0,

where we use the induction condition that ⟨w(t)
r , µ⟩< 0. Further

⟨w(t+1)
r
, µ⟩= ⟨w(0)
r , µ⟩+ γ(t+1)
r
+

n
X

i=1
ρ(t)
r,i∥ξi∥−2
2 ⟨ξi, µ⟩

(a)
≤⟨w(0)
r , µ⟩+ γ(t+1)
r
+ SNR

r

8 log(6n/δ)

d
n
(b)
< 0,

where inequality (a) follows from Lemma C.1; we use γ(t+1)
r
≤0 and Lemma B.2 in deriving
inequality (b). This completes the induction for r ∈U(0)
−.

With Lemma C.3 at hand, we are ready to demonstrate the upper bound of the growth rate for signal
learning.
Lemma C.4. With the same condition as in Lemma C.2 and Lemma C.3 and n ≥2500 log(4/δ),
define A(t)
r
= γ(t)
r
+ ⟨w(0)
r , µ⟩for r ∈U(0)
+ ; and A(t)
r
= −γ(t)
r
−⟨w(0)
r , µ⟩for r ∈U(0)
−. With
probability at least 1 −δ, we have

A(t)
r
≤

1 + 0.52η∥µ∥2
2
mτ


A(0)
r .

Proof. Lemma C.3 suggests that for r ∈[m], we can upper bound for |γ(t)
r |. Without loss of
generality, we consider r ∈U(0)
+ . By Lemma C.1, we can see

⟨w(t)
r , µ⟩≤γ(t)
r
+ ⟨w(0)
r , µ⟩+ SNR

r

8 log(6n/δ)

d
n

≤1.01
 
γ(t)
r
+ ⟨w(0)
r , µ⟩

,
(19)

⟨w(t)
r , µ⟩≥γ(t)
r
+ ⟨w(0)
r , µ⟩−SNR

r

8 log(6n/δ)

d
n

≥0.99
 
γ(t)
r
+ ⟨w(0)
r , µ⟩

,
(20)

where we use Lemma B.2 and γ(t)
r
≥0.

Then, the update equation for γ(t)
r
follows

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i:yi=1
(1 −ℓ′(t)
i
)⟨w(t)
r , µ⟩∥µ∥2
2.

Then we find that

A(t+1)
r
= A(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ(t)
i )⟨w(t)
r , µ⟩∥µ∥2
2

(a)
≤A(t)
r
+
η
nmτ

X

i:yi=1


1 −
1
Cℓ(M + 1)


∥µ∥2
21.01A(t)
r

(b)
≤A(t)
r
+ η∥µ∥2
2
mτ

 
1
2 +

r

1
2n log(4/δ)

!

1.01A(t)
r

(c)
≤

1 + 0.52η∥µ∥2
2
mτ


A(t)
r ,
(21)

22


---Page Break---
where the first inequality (a) is by (19) and Lemma C.2. The second inequality (b) is by Lemma B.3
and 1 −
1
Cℓ(M+1) < 1. The last inequality (c) is by the condition n ≥2500 log(4/δ).

C.1.2
Dynamics of Noise Memorization: Lower Bound

To establish the lower bound of noise memorization in the first stage, we require that the added noise
level σϵ ≤σξ/C′ for sufficiently large, with C′ ≥eΩ(1). We prove the following result that upper
bound the scale of ⟨w(t)
r , ϵi⟩.

Lemma C.5. Under the same condition as Lemma C.1 and d = eΩ(max{σ−2
0 ∥µ∥−2
2 , nσ−1
0 σ−1
ξ }),
there exists a sufficiently large constant Cξ > 0 such that

|⟨w(0)
r , ξi⟩| ≥Cξ|⟨w(t)
r , ϵi⟩|

for all i ∈[n].

Proof of Lemma C.5. According to the decomposition of w(t)
r , we can show

|⟨w(t)
r , ϵi⟩| = |⟨w(0)
r , ϵi⟩+ γ(t)
r ∥µ∥−2
2 ⟨µ, ϵi⟩+

n
X

i′=1
ρ(t)
r,i′∥ξi′∥−2
2 ⟨ξi′, ϵi⟩|

≤|⟨w(0)
r , ϵi⟩| + γ(t)
r ∥µ∥−2
2 |⟨µ, ϵi⟩| +

n
X

i′=1
|ρ(t)
r,i|∥ξi′∥−2
2 |⟨ξi′, ϵi⟩|

(a)
≤2
p

log(8mn)/δσ0σϵ
√

d + ∥µ∥−1
2 σϵ
p

2 log(6n/δ) + 4σϵσ−1
ξ

r

log(6n2/δ)

d
n

≤1/C|⟨w(0)
r , ξi⟩|,

where the second inequality (a) follows from Lemma B.5 and the last inequality is by the following
anti-concentration result.

Because ⟨w(0)
r , ξi⟩is a Gaussian random variable with mean zero and variance σ0σξ
√

d, by Lemma
B.1, we compute

P
 
|⟨w(0)
r , ξi⟩| ≤c

≤2

v
u
u
t1 −exp

 

−
2c2

πσ2
0σ2
ξd

!

,

When d ≥
2c2

−log(1−δ2/(4n2))πσ2
0σ2
ξ , we have P
 
|⟨w(0)
r , ξi⟩| ≤c

≤δ/n and by union bound, we have

with probability at least 1−δ, it holds |⟨w(0)
r , ξi⟩| > c. Here we let c = C
 
2
p

log(8mn)/δσ0σϵ
√

d+

∥µ∥−1
2 σϵ
p

2 log(6n/δ) + 4σϵσ−1
ξ

q

log(6n2/δ)

d
n

≥C|⟨w(t)
r , ϵi⟩|. Then we can show the desired

result with the condition d = eΩ(max{σ−2
0 ∥µ∥−2
2 , nσ−1
0 σ−1
ξ }).

Define that I(t)
r,+ = {i : ⟨w(t)
r , ξi⟩> 0} and I(t)
r,−= {i : ⟨w(t)
r , ξi⟩< 0}. To show the result

regarding I(t)
r,+ and I(t)
r,−, we prepare the following anti-concentration result:

Lemma C.6. Under the condition d ≥
r

300 log(6n2/δ)
−log(1−δ2/4n2)πσ2
0σ2
ξ , then with probability at least 1 −δ,

it satisfies

|⟨w(0)
r , ξi⟩| > 150
p

log(6n2/δ)/d.

Proof of Lemma C.6. Here we want to show |⟨w(0)
r , ξi⟩| ≥150
p

log(6n2/δ)/dn ≜c with high
probability. To see this, because ⟨w(0)
r , ξi⟩is a Gaussian random variable with mean zero and

23


---Page Break---
variance σ2
0σ2
ξd, by Lemma B.1, we compute

P
 
|⟨w(0)
r , ξi⟩| ≤c

≤2

v
u
u
t1 −exp

 

−
2c2

πσ2
0σ2
ξd

!

,

Thus when d ≥
r

300 log(6n2/δ)
−log(1−δ2/4n2)πσ2
0σ2
ξ , we have P
 
|⟨w(0)
r , ξi⟩| ≤c

≤δ/n and by union bound,

we have with probability at least 1 −δ, it holds |⟨w(0)
r , ξi⟩| > 150
p

log(6n2/δ)/d.

We then show that neurons with negative inner products with the noise at initialization would stay
negative and the corresponding ρ stays zero.
Lemma C.7. Under the same condition as Lemma C.1 and Lemma C.6, for all t > 0, we have
I(t)
r,−= I(0)
r,−and ρ(t)
r,i = 0 for all i ∈I(0)
r,−.

Proof of Lemma C.7. The proof is by induction. We first consider i ∈I(0)
r,−. At t = 1,

ρ(1)
r,i = ρ(0)
r,i +
η
nmτ (1 −ℓ′(0)
i
)σ(⟨w(0)
r , ξi + ϵi⟩)σ′(⟨w(0)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(0)
i,j σ(⟨w(0)
r , ξj⟩)σ′(⟨w(0)
r , ξi⟩)∥ξi∥2
2 = 0,

where we have used the condition that ⟨w(0)
r , ξi⟩< 0 and the property of ReLU activation.

Next we consider

⟨w(1)
r , ξi⟩= ⟨w(0)
r
+ γ(1)
r µ∥µ∥−2
2
+

n
X

i=1
ρ(1)
r,i ∥ξi∥−2
2 ξi, ξi⟩

= ⟨w(0)
r , ξi⟩+ γ(1)
r ⟨µ, ξi⟩∥µ∥−2
2
+ ρ(1)
r,i +

n
X

i′̸=i
ρ(1)
r,i′∥ξi′∥−2
2 ⟨ξi′, ξi⟩

(a)
≤⟨w(0)
r , ξi⟩+ 5

r

log(6n2/δ)

d
n
(b)
< 0,

where inequality (a) is by Lemma C.1 and ρ(1)
r,i = 0, and inequality (b) is by Lemma C.6.

Suppose at iteration t, the claim is satisfied, i.e., ⟨w(t)
r , ξi⟩< 0, ρ(t)
r,i = 0, for all i ∈I(0)
r,−. Then

ρ(t+1)
r,i
= ρ(t)
r,i +
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi + ϵi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2 < 0,

Next we consider the update of inner product as

⟨w(t+1)
r
, ξi⟩= ⟨w(0)
r
+ γ(t+1)
r
µ∥µ∥−2
2
+

n
X

i=1
ρ(t+1)
r,i
∥ξi∥−2
2 ξi, ξi⟩

= ⟨w(0)
r , ξi⟩+ γ(t+1)
r
⟨µ, ξi⟩∥µ∥−2
2
+ ρ(t+1)
r,i
+

n
X

i′̸=i
ρ(t+1)
r,i′
∥ξi′∥−2
2 ⟨ξi′, ξi⟩

≤⟨w(0)
r , ξi⟩+ γ(t+1)
r
|⟨µ, ξi⟩|∥µ∥−2
2
+

n
X

i′̸=i
|ρ(t+1)
r,i′
|∥ξi′∥−2
2 |⟨ξi′, ξi⟩|

≤⟨w(0)
r , ξi⟩+ 5

r

log(6n2/δ)

d
n < 0,

where the last inequality is by a anti-concentration analysis shown in Lemma C.6. This completes the
induction for i ∈I(t)
r,−.

24


---Page Break---
Before formally stating the main lemma on the lower bound of noise memorization, we prepare
several lemmas that will be useful.
Lemma C.8. Suppose that δ > 0. Then with probability 1 −δ, for all r ∈[m], we have:


X

i:yi=1
1i∈I(0)
r,+ −n

4


≤
rn

2 log(4/δ).

Proof. The proof is by Hoeffding’s inequality, for arbitrary t > 0, we have that

P







X

i:yi=1
1i∈Ir,+ −E



X

i:yi=1
1i∈Ir,+






≤t



≤2 exp

−2t2

n


.

By the randomness of initialization and Rademacher distribution, we have:

E



X

i:yi=1
1i∈Ir,+



= n

4

Setting t =
p

n/2 log(4/δ) and taking a union bound over r ∈[m], we conclude with high
probability at least 1 −δ, it holds:


X

i:yi=1
1i∈Ir,+ −n

4


≤
rn

2 log(4/δ).

To establish the lower bound for noise memorization we define

B(t)
r,+ ≜
X

i:yi=+1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+,
B(t)
r,−≜
X

i:yi=−1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+.

Lemma C.9. Suppose δ > 0, the with probability at least 1 −δ, we have

B(0)
r,+ ≤σ0σξ
√

dn(1 +
p

log(1/δ)/2),
B(0)
r,−≤σ0σξ
√

dn(1 +
p

log(1/δ)/2).

Proof. By Bernstein’s inequality, for arbitrary t > 0, we have

P(|B(0)
r,+ −σ0σξ
√

nd| > t) ≤exp(−
t2

2n/4σ2
0σ2
ξd).

Setting t = σ0σξ
p

nd log(1/δ)/2, we further have

B(0)
r,+ ≤σ0σξ
√

dn(1 +
p

log(1/δ)/2).

Similarly, the same result holds for B(0)
r,−.

Lemma C.10. For arbitrary constant C > 0, under the condition n ≥
8C2 log(1/δ)
log(1/(1−(δ/m)2)), then with
probability at least 1 −δ, it satisfies

|⟨w(0)
r , ξi⟩| > CB(0)
r,+
n
,
|⟨w(0)
r , ξi⟩| > CB(0)
r,−
n
.

Proof of Lemma C.10. Here we want to show |⟨w(0)
r , ξi⟩| >
B(0)
r,+
n
with high probability. To see this,

because ⟨w(0)
r , ξi⟩is a random variable with positive mean and variance σ2
0σ2
ξd. Besides, by Lemma
C.9, we have

B(0)
r,+
n
≤σ0σξ
√

dn(1 +
p

log(1/δ)/2)
n
.

25


---Page Break---
By Lemma B.1, we recall

P
 
|⟨w(0)
r , ξi⟩| ≤t

≤

v
u
u
t1 −exp

 

−
8t2

πσ2
0σ2
ξd

!

,

Thus when n ≥
8C2 log(1/δ)
log(1/(1−(δ/m)2)), we have

P
 
|⟨w(0)
r , ξi⟩| ≤Cσ0σξ
√

dn(1 +
p

log(1/δ)/2)
n
) ≤δ/m

and by union bound, we have with probability at least 1 −δ, it holds |⟨w(0)
r , ξi⟩| >
B(0)
r,+
n
and

|⟨w(0)
r , ξi⟩| >
B(0)
r,−
n .

Besides, we define

Ψ(t)
r,i ≜ρ(t)
r,i + ⟨w(0)
r , ξi⟩, with yi = 1, i ∈I(t)
r,+

Φ(t)
r,i ≜ρ(t)
r,i + ⟨w(0)
r , ξi⟩, with yi = −1, i ∈I(t)
r,+
With all the results (lemmas) and definitions outlined above at hand, we are ready to state the lemmas
that provide the lower bound for noise memorization as follows.
Lemma C.11. Under the same condition as Theorem 4.2, then with probability at least 1 −δ,

Ψ(t)
r,i ≥(1 +
0.96ησ2
ξd
nmτ
)tΨ(0)
r,i ,
(22)

Ψ(t)
r,i ≥101Cℓ

M + 1B(t)
r,−,
(23)

ρ(t)
r,i:yi=11i∈I(t)
r,+ ≥0.
(24)

Proof. The proof is by induction. We can check

ρ(0)
r,i:yi=11i∈Ir,+ = 0,
Ψ(0)
r,i ≥(1 +
0.96ησ2
ξd
nmτ
)0Ψ(0)
r,i .

Besides, by Lemma C.10 and M ∈n

2 (1 ± o(n−1/2)), it holds that

Ψ(0)
r,i = ⟨w(0)
r , ξi⟩≥101
Cℓ
M + 1B(0)
r,−.

Assuming that the results hold at t, we continue the proof by calculating the propagation of B(t)
r,+ and

B(t)
r,−respectively. First, we can see by Lemma C.1 and by induction, for yi = 1 and i ∈Ir,+,

⟨w(t)
r , ξi + ϵi⟩≥⟨w(0)
r , ξi⟩+ ρ(t)
r,i −|⟨w(t)
r , ϵi⟩| −6

r

log(6n2/δ)

d
n

≥(1 −o(1))⟨w(0)
r , ξi⟩+ ρ(t)
r,i ≥0.

We can show that

B(t+1)
r,−
= B(t)
r,−+
η
nmτ

X

i:yi=−1



(1 −ℓ′(t)
i
)⟨w(t)
r , ξi + ϵi⟩−
X

j̸=i
ℓ′(t)
i,j ⟨w(t)
r , ξj⟩1i∈Ir,+



1i∈Ir,+∥ξi∥2
2

≤B(t)
r,−+
η
nmτ

X

i:yi=−1

Cℓ(M + 1) −1

Cℓ(M + 1)
(⟨w(0)
r , ξi⟩+ ρ(t)
r,i + |⟨w(t)
r , ϵi⟩| + 6

r

log(6n2/δ)

d
n)

−
X

j̸=i
1j∈Ir,+
1
Cℓ(M + 1)(⟨w(0)
r , ξj⟩+ ρ(t)
r,j −6

r

log(6n2/δ)

d
n)

(σ2
ξd + σ2
ξ
p

d log(6n2/δ))1i∈Ir,+

26


---Page Break---
≤B(t)
r,−+
η
nτm

Cℓ(M + 1) −1

Cℓ(M + 1)
1.01B(t)
r,−−
X

i:yi=−1
1i∈Ir,+
1
Cℓ(M + 1)0.99B(t)
r,−


(σ2
ξd + σ2
ξ
p

d log(6n2/δ))

−
X

i:yi=−1
1i∈Ir,+
1
Cℓ(M + 1)(B(t)
r,−−
X

j̸=i
6

r

log(6n2/δ)

d
n)

(σ2
ξd + σ2
ξ
p

d log(6n2/δ))

≤B(t)
r,−+
η
nmτ


1.01B(t)
r,−−0.992

2Cl
B(t)
r,+


1.035σ2
ξd

≤B(t)
r,−+
η
nmτ


1.05B(t)
r,−−0.5B(t)
r,+


σ2
ξd

≤B(t)
r,−+
1.05ησ2
ξd
nmτ
B(t)
r,−,

where the first inequality is by Lemma C.2, Lemma C.1 and Lemma B.5. Furthermore, the second
inequality is by Lemma C.5, i.e., |⟨w(t)
r , ϵi⟩| ≤1/Cξ|⟨w(0)
r , ξi⟩| (for Cξ > 200), |0.01⟨w(0)
r , ξi⟩| ≥
6
p

log(6n2/δ)d−1n, the definition of B(t)
r,+ and B(t)
r,−and the induction B(t)
r,+ ≥B(0)
r,+, B(t)
r,−≥B(0)
r,−.
The third inequality is by M ≥100Cℓ−1, M ∈n

2 (1 ± o(n−1/2)), d > 10000 log(6n2/δ) and
Lemma C.8. The last inequality is by B(t)
r,+ ≥0.

As a result, we conclude that

B(t)
r,−≤(1 +
1.05ησ2
ξd
nmτ
)tB(0)
r,−.

Finally, the induction step for Ψ(t)
r,i can be calculated as follows:

Ψ(t+1)
r,i
= Ψ(t)
r,i +
η
nmτ



(1 −ℓ′(t)
i
)⟨w(t)
r , ξi + ϵi⟩−
X

j̸=i
1j∈Ir,+ℓ′(t)
i,j ⟨w(t)
r , ξj⟩



1i∈Ir,+∥ξi∥2
2

≥Ψ(t)
r,i +
η
nmτ

(M + 1) −Cℓ

(M + 1)
(⟨w(0)
r , ξi⟩+ ρ(t)
r,i −|⟨w(t)
r , ϵi⟩| −6

r

log(6n2/δ)

d
n)

−
X

j̸=i
1j∈Ir,+
Cl
M + 1(⟨w(0)
r , ξj⟩+ ρ(t)
r,j + 6

r

log(6n2/δ)

d
n)

(σ2
ξd −σ2
ξ
p

d log(6n2/δ))1i∈Ir,+

≥Ψ(t)
r,i +
η
nmτ

(M + 1) −Cℓ

M + 1
0.99Ψ(t)
r,i −
Cℓ
M + 11.01B(t)
r,−


(σ2
ξd −σ2
ξ
p

d log(6n2/δ))

≥Ψ(t)
r,i +
η
nmτ


0.992Ψ(t)
r,i −1.01
Cl
M + 1B(t)
r,−


0.99σ2
ξd

≥Ψ(t)
r,i +
ησ2
ξd
nmτ

h
0.96Ψ(t)
r,i
i
,

where the first inequality is by Lemma C.2, Lemma C.1 and Lemma B.5. Furthermore, the second
inequality is by Lemma C.5, i.e., |⟨w(t)
r , ϵi⟩| ≤1/Cξ|⟨w(0)
r , ξi⟩| (for Cξ > 200), |0.01⟨w(0)
r , ξi⟩| ≥
6
p

log(6n2/δ)d−1n, the definition of B(t)
r,+ and B(t)
r,−. The third inequality is by M ≥100Cℓ−1.
The last inequality is by induction (23).

Then we check induction induction (23) through following inequalities:

B(t)
r,−≤(1 +
1.05ησ2
ξd
nmτ
)tB(0)
r,−,

Ψ(t)
r,i ≥(1 +
0.96ησ2
ξd
nmτ
)tΨ(0)
r,i .

Together it confirms that Ψ(t)
r,i ≥101Cℓ

M+1 B(t)
r,−.

27


---Page Break---
Finally, we check the sign of ρ(t)
r,i for i : yi = 1 and i ∈I(t)
r,+:

ρ(t)
r,i = Ψ(t)
r,i −⟨w(0)
r , ξ⟩≥(1 +
ησ2
ξd
2nmτ )tΨ(t)
r,i −⟨w(0)
r , ξ⟩>
ησ2
ξd
2nmτ Ψ(t)
r,i > 0.

This completes the induction proof.

C.1.3
Noise Memorization: Proof of Lemma 5.2

Before proving Lemma 5.2, we require a lower bound for the initialization. Define that U(0)
+
= {r :
⟨w(0)
r , ξi⟩> 0} for yi = 1, and U(0)
−
= {r : ⟨w(0)
r , ξi⟩< 0} for yi = −1.

Lemma C.12. Suppose that δ > 0 and m ≥eΩ(1). Then with probability at least 1 −δ, we have

1
m

X

r∈U(0)
+

Ψ(0)
r,i ≥0.2σ0σξ
√

d,
1
m

X

r∈U(0)
−

Φ(0)
r,i ≥0.2σ0σξ
√

d.

Proof of Lemma C.12. Consider yi = 1. Note that ⟨w(0)
r , ξi⟩∼N(0, σ2
0∥ξi∥2
2). We define that
the event A = {r ∈[m], ⟨w(0)
r , ξi⟩> 0}. Then we can compute that ⟨w(0)
r , ξi⟩1(A) becomes a
half-normal distribution with the expectation

E[⟨w(0)
r , ξi⟩1(A)] =

√

2σ0∥ξi∥2
√π
.

We then apply the sub-Gaussian concentration inequality that with probability at least 1 −δ


X

r∈U(0)
+

Ψ(0)
r,i −mσ0∥ξi∥2
√

2π


≤eO(m−1/2).

Then we have
1
m

X

r∈U(0)
+

Ψ(0)
r,i ≥0.4σ0∥ξi∥2 ≥σ0σξ
√

d,

where we have used Lemma B.5. Similarly, we can show that for yi = −1, the same result holds.

Proof of Lemma 5.2. From the upper bound on (21), we take the maximum over r ∈U(0)
+ , which
gives

max
r
A(t)
r
≤

1 + 0.52η∥µ∥2
2
mτ

t
max
r
A(0)
r

≤

1 + 0.52η∥µ∥2
2
mτ

t
σ0∥µ∥2
p

2 log(8m/δ).

Under the SNR condition n · SNR2 ≤1.8, we can see there exists a scale difference between
maxr,i Ψ(t)
r,i and maxr,i A(t)
r
at the end of first stage.

At the same time, for noise memorization, from the lower bound established in Lemma C.11, we
have that

1
m

X

r∈U(0)
+

Ψ(t)
r,i ≥(1 +
0.96ησ2
ξd
nmτ
)t 1

m

X

r∈U(0)
+

Ψ(0)
r,i

≥(1 +
0.96ησ2
ξd
nmτ
)t0.2σ0σξ
√

d,

where the second inequality is due to Lemma C.12.

28


---Page Break---
Let

T1 = log
 
20/(σ0σξ
√

d)

/ log
 
1 + 0.96
ησ2
ξd
nmτ

.

Then we have 1

m
P

r∈U(0)
+ Ψ(t)
r,i reach 3 within T1 iterations by (22). Similarly, we can also show that

1
m
P

r∈U(0)
−Φ(t)
r,i reach 3 within T1 iterations.

On the other hand, we compute the scale of maxr A(T1)
r
as

max
r
A(T1)
r
≤

1 + 0.52η∥µ∥2
2
mτ

T1
σ0∥µ∥2
p

2 log(8m/δ)

= exp
log(1 + 0.52 η∥µ∥2
2
mτ )

log(1 + 0.96
ησ2
ξd
nmτ )
log(12/(σ0σξ
√

d)

σ0∥µ∥2
p

2 log(8m/δ)

≤exp
 
(0.55 · n · SNR2 + O((η∥µ∥2
2
mτ
)2) log(20/(σ0σξ
√

d)

σ0∥µ∥2
p

2 log(8m/δ)

≤exp
 
(0.55 · n · SNR2 + 0.01) log(30/(σ0σξ
√

d)

σ0∥µ∥2
p

2 log(8m/δ)

≤exp
 
log(30/(σ0σξ
√

d)

σ0∥µ∥2
p

2 log(8m/δ)

= 20
p

2 log(8m/δ)SNR

= O(
p

log(m/δ)/n)

= eO(n−1/2)

where we choose η sufficiently small for the third inequality. The last inequality is by the SNR
condition. Because we can choose n ≥C log(m/δ) for sufficiently large constant C, maxr A(T1)
r
=
o(1).

C.2
Second Stage

Proposition C.13. Let T ∗be the maximum admissible iteration and let α = log(3MT ∗). Then we
can show

|γ(t)
r | ≤α,
|ρ(t)
r,i| ≤α.

Proof of Proposition C.13. We need to show ρ(t)
r,i ≤α. We prove the claim by induction. It is clear

when t = 0, ρ(t)
r,i = 0 ≤α. Suppose for all 0 ≤t ≤eT −1, we have ρ(t)
r,i ≤α. We aim to show the
claim holds for eT.

By the update of ρ(t)
r,i,

ρ(t+1)
r,i
≤ρ(t)
r,i +
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi + ϵi⟩)∥ξi∥2
2.

Here without loss of generality, we consider the r, i pairs such that i ∈I(0)
r,+, i.e., ⟨w(0)
r , ξi⟩> 0 with

yi = 1. Let tr,i be the last time t such that ρ(t)
r,i ≤0.5α. Then we have

ρ( e
T )
r,i ≤ρ(tr)
r,i +
η
nmτ (1 −ℓ′(tr,i)
i
)σ(⟨w(tr,i)
r
, ξi + ϵi⟩)∥ξi∥2
2

+
X

tr,i≤t≤e
T

η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi + ϵi⟩)∥ξi∥2
2.
(25)

The second term can be bounded as
η
nmτ (1 −ℓ′(tr,i)
i
)σ(⟨w(tr,i)
r
, ξi + ϵi⟩)∥ξi∥2
2 ≤
η
nmτ
 3

2⟨w(0)
r , ξi⟩+ ρ(tr,i)
r,i

∥ξi∥2
2

29


---Page Break---
≤
3ησ2
ξd
2nmτ
 
0.3α + 0.5α


≤0.25α,
(26)

where the first inequality is by 1 −ℓ′(tr)
i
≤1 and modified Lemma C.1 with ρ(t)
r,i = O(α). The

second inequality is by ⟨w(0)
r , ξi⟩≤0.2α and η ≤
5
24
nmτ

σ2
ξd .

For notation convenience, we let

F0(W, xi) = Simh(xi, bxi)/τ

=
1
mτ

m
X

r=1
σ(⟨wr, yiµ⟩)sg(σ(⟨wr, yiµ⟩)) +
1
mτ

m
X

r=1
σ(⟨wr, ξi⟩)sg(σ(⟨wr, ξi + ϵi⟩))

Fj(W, xi) = Simh(xi, bxj)/τ

=
1
mτ

m
X

r=1
σ(⟨wr, yiµ⟩)sg(σ(⟨wr, yjµ⟩)) +
1
mτ

m
X

r=1
σ(⟨wr, ξi⟩)sg(σ(⟨wr, ξj⟩)), for j = 1, ..., M

Next, we show the bound for F0(W(t), xi) and Fj(W(t), xi) for j = 1, ..., M as

F0(W(t), xi) ≥
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩).

Further, we have for j = 1, ..., M

Fj(W(t), xi) =
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξj⟩).

Then we can bound the difference between F0(W(t), xi) and Fj(W(t), xi) as

F0(W(t), xi) −Fj(W(t), xi) ≥
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩) −
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξj⟩)

=
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi⟩)
 
σ(⟨w(t)
r , ξi + ϵi⟩) −σ(⟨w(t)
r , ξj⟩)


≥0.5α(1

2⟨w(0)
r , ξi⟩+ ρ(t)
r,i −3

2⟨w(0)
r , ξj⟩−ρ(t)
r,j)/τ

≥0.5α(1

2⟨w(0)
r , ξi⟩+ 1

2ρ(t)
r,i −3

2⟨w(0)
r , ξj⟩)/τ

≥0.5α · 0.1α/τ
≥α,
(27)

where the second inequality is by σ(⟨w(t)
r , ξi⟩) ≥3

4⟨w(0)
r , ξi⟩+ ρ(t)
r,i ≥ρ(t)
r,i ≥0.5α for i ∈I(0)
r,+.

Further σ(⟨w(t)
r , ξi + ϵi⟩) ≥1

2⟨w(0)
r , ξi⟩+ ρ(t)
r,i, σ(⟨w(t)
r , ξj⟩) ≤3

2⟨w(0)
r , ξj⟩+ ρ(t)
r,j. The fourth

inequality is by induction and 1

2⟨w(0)
r , ξi⟩−3

2⟨w(0)
r , ξj⟩≥−0.15α. The last inequality is by the
condition that α ≥20τ.

Further, we bound the loss derivative as

1 −ℓ′(t)
i
= 1 −
1

1 + PM
j=1 exp(Fj(W(t), xi) −F0(W(t), xi))

≤1 −
1
1 + M exp(−α)

≤1 −
T ∗

1 + T ∗

30


---Page Break---
=
1
T ∗+ 1,
(28)

where the second inequality is by (27).

Finally, we bound

X

tr,i≤t≤e
T

η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi + ϵi⟩)∥ξi∥2
2 ≤
3ησ2
ξd
2nmτ · 2α ·
X

tr,i≤t≤e
T
(1 −ℓ′(t)
i
)

≤
3ησ2
ξd
nmτ · α ·
T ∗

T ∗+ 1
≤0.25α
(29)

where the first inequality is by Lemma B.5 and σ(⟨w(t)
r , ξi + ϵi⟩) ≤3

2⟨w(0)
r , ξi⟩+ ρ(t)
r,i ≤2α by
induction. The last inequality is by the condition η ≤
1
12
nmτ

σ2
ξd .

Combining (26) and (29) with (25) gives

ρ( e
T )
r,i ≤0.5α + 0.25α + 0.25α = α,

which completes the induction.

Lemma C.14. Under Assumption 4.1, for 0 ≤t ≤T ∗, we have

∥∇LS(W(t))∥2
F ≤O(max{∥µ∥2
2, σ2
ξd})LS(W(t)).

Proof of Lemma C.14. First, we can write the gradient of ∇wrLi(W) as

∇wrLi(W) =

M
X

j=0

∂Li(W)
∂Fj(W, xi)∇wrFj(W, xi),

where

∂Li(W)

∂F0
= −1 +
eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi)

∂Li(W)

∂Fj
=
eFj(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi) , for j = 1, ..., M

By the derivation of the gradient,

∥∇LS(W(t))∥2
F ≤
  1

n

n
X

i=1
∥∇Li(W)∥F
2

=
 1

n

n
X

i=1

v
u
u
t

m
X

r=1
∥∇wrLi(W(t))∥2
2
2

≤
 1

n

n
X

i=1

m
X

r=1
∥∇wrLi(W(t))∥2
2

=
 1

n

n
X

i=1

m
X

r=1
∥

M
X

j=0

∂Li(W)
∂Fj(W, xi)∇wrFj(W, xi)∥2
2

≤
 1

n

n
X

i=1

m
X

r=1

M
X

j=0


∂Li(W)
∂Fj(W, xi)

∥∇wrFj(W, xi)∥2
2
,
(30)

where the first inequality is by triangle inequality and second inequality uses P

i a2
i ≤(P

i ai)2 for
ai ≥0 and the last inequality is by triangle inequality.

31


---Page Break---
Now we upper bound ∥∇wrFj(W, xi)∥2 as

∥∇wrF0(W, xi)∥2 =
1
mτ ∥σ′(⟨wr, yiµ⟩)σ(⟨wr, yiµ⟩)yiµ + σ′(⟨wr, ξi⟩)σ(⟨wr, ξi + ϵi⟩)ξi∥2

≤
1
mτ
 
σ(⟨wr, yiµ⟩)∥µ∥2 + σ(⟨wr, ξi + ϵi⟩)∥ξi∥2


≤
1
mτ
 
σ(⟨wr, yiµ⟩) + σ(⟨wr, ξi + ϵi⟩)

O
 
max{∥µ∥2, σξ
√

d}

,

where the first inequality is by triangle inequality and the second inequality is by Jensen’s inequality.
Similarly, we can obtain for j = 1, ..., M

∥∇wrFjW, xi)∥2 ≤
1
mτ
 
σ(⟨wr, yjµ⟩) + σ(⟨wr, ξj⟩)

O
 
max{∥µ∥2, σξ
√

d}

.

For clarity let

zr,0 = σ(⟨wr, yiµ⟩) + σ(⟨wr, ξi + ϵi⟩)
zr,j = σ(⟨wr, yjµ⟩) + σ(⟨wr, ξj⟩), for j = 1, ..., M

Substituting the above results into (30) gives

∥∇LS(W(t))∥2
F ≤

1
nmτ

n
X

i=1

m
X

r=1

M
X

j=0


∂Li(W)
∂Fj(W, xi)

zr,j
2
O
 
max{∥µ∥2
2, σ2
ξd}


≤
 1

nτ

n
X

i=1

  M
X

j=0


∂Li(W)
∂Fj(W, xi)


( 1

m

m
X

r=1

M
X

j=0
zr,j)
2
O
 
max{∥µ∥2
2, σ2
ξd}

,

where the second inequality is by P
i aibi ≤(P
i ai)(P
i bi) for ai, bi ≥0.

Next we can verify that

M
X

j=0


∂Li(W)
∂Fj(W, xi)

 = 1 −
eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi) +

M
X

j=1

eFj(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi)

= 2
 
1 −
eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi)


≤−6 log(
eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi) ) = 6Li(W),
(31)

where the last inequality is by 1 −x ≤−3 log(x) for x ∈[0, 1]. Furthermore, we can show

2
 
1 −
eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi)

( 1

m

m
X

r=1

M
X

j=0
zr,j)2 = O(1),

which leads to

∥∇LS(W(t))∥2
F ≤
 1

nτ

n
X

i=1

v
u
u
t  M
X

j=0


∂Li(W(t))
∂Fj(W(t), xi)

2( 1

m

m
X

r=1

M
X

j=0
zr,j)2
2
O
 
max{∥µ∥2
2, σ2
ξd}


≤
 1

nτ

n
X

i=1

v
u
u
t

M
X

j=0


∂Li(W(t))
∂Fj(W(t), xi)


2
O
 
max{∥µ∥2
2, σ2
ξd}


≤O
 
max{∥µ∥2
2, σ2
ξd}
 1

n

n
X

i=1
Li(W(t))

≤O
 
max{∥µ∥2
2, σ2
ξd}

LS(W(t)),

where the third inequality is by (31) and Cauchy-Schwartz inequality.

32


---Page Break---
We define

w∗
r = w(0)
r
+ 2τ log(2M/ϵ)

n
X

i=1

ξi
∥ξi∥2
2
.

Recall in the first stage

w(T1)
r
= w(0)
r
+ γ(T1)
r
∥µ∥−2
2 µ +

n
X

i=1
ρ(T1)
r,i ∥ξi∥−2
2 ξi,

and we have

• maxr |γ(t)
r | = eO(n−1/2) for all 0 ≤t ≤T1.

• maxr,i ρ(T1)
r,i
≥2.

Lemma C.15. Under Assumption 4.1, we have ∥W(T1) −W∗∥F ≤eO(m1/2n1/2σ−1
ξ d−1/2).

Proof of Lemma C.15. We have

∥W(T1) −W∗∥F ≤∥W(T1) −W(0)∥F + ∥W(0) −W∗∥F

≤
X

r

γ(T1)
r
∥µ∥2
+ O(√m) max
r
∥

n
X

i=1
ρ(T1)
r,i
ξi
∥ξi∥2
2
∥2 + O(m1/2n1/2 log(1/ϵ)σ−1
ξ d−1/2)

≤eO(m1/2n1/2σ−1
ξ d−1/2).

Lemma C.16. Under Assumption 4.1, we have for all t ∈[T1, T ∗]

⟨∇F0(W(t), xi), W∗⟩≥2 log(2M/ϵ),

⟨∇Fj(W(t), xi), W∗⟩≤log(2M/ϵ).

Proof of Lemma C.16. Based on the definition of W∗and Fj(W(t), xi), we can derive for j = 0,

⟨∇F0(W(t), xi), W∗⟩

=

m
X

r=1
⟨∇wrF0(W(t), xi), w∗
r⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨w(t)
r , yiµ⟩)⟨w∗
r, yiµ⟩+
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩)⟨w∗
r, ξi⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨w(t)
r , yiµ⟩)

⟨w(0)
r , yiµ⟩+ 2τ log(2M/ϵ)

n
X

i=1
⟨ξi, yiµ⟩∥ξi∥−2
2


+
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩)

⟨w(0)
r , ξi⟩+ 2τ log(2M/ϵ) + 2τ log(2M/ϵ)
X

i′̸=i
⟨ξi, ξi′⟩∥ξi′∥−2
2


≥
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξi + ϵi⟩)2τ log(2M/ϵ)

|
{z
}
I1

−1

mτ

m
X

r=1
σ(⟨w(t)
r , yiµ⟩) eO(σ0∥µ∥2)

|
{z
}
I2

−
1
mτ

m
X

r=1
σ(⟨w(t)
r , yiµ⟩)2τ log(2M/ϵ) eO(n∥µ∥2σ−1
ξ d−1)

|
{z
}
I3

−1

mτ

m
X

r=1
σ(⟨w(t)
r , ξi + ϵi⟩) eO(σ0σξ
√

d)

|
{z
}
I4

−
1
mτ

m
X

r=1
σ(⟨w(t)
r , ξi + ϵi⟩)2τ log(2M/ϵ) eO(nd−1/2)

|
{z
}
I5

33


---Page Break---
where the inequality is by Lemma B.5.

Next, we bound I1, I2, I3, I4, I5 separately. For I1, we take maximum over r, which results in

1
m

m
X

r=1
σ(⟨w(t)
r , ξi + ϵi⟩) ≥1

m

m
X

r=1
σ(1

2⟨w(0)
r , ξi⟩+ ρ(t)
r,i) ≥2.

Thus we obtain

I1 ≥4 log(2M/ϵ).

In addition, we can show by upper bound on ρ(t)
r,i and γ(t)
r ,

⟨w(t)
r , ξi + ϵi⟩≤3

2|⟨w(0)
r , ξi⟩| + |ρ(t)
r,i| ≤eO(1),

⟨w(t)
r , µ⟩≤3

2|⟨w(0)
r , µ⟩| + |γ(t)
r | ≤eO(1).

This implies

I2 ≤eO(σ0∥µ∥2), I3 ≤log(2M/ϵ) eO(nm∥µ∥2σ−1
ξ d−1), I4 ≤eO(σ0σξ
√

d), I5 ≤eO(nmd−1/2).

Based on the conditions on σ0, d, we can show

⟨∇F0(W(t), xi), W∗⟩≥4 log(2M/ϵ) −I2 −I3 −I4 −I5 ≥2 log(2M/ϵ).

Now we prove for the claim for Fj(W(t), W∗) for j = 1, ..., M as follows.

⟨∇Fj(W(t), xi), W∗⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨w(t)
r , yjµ⟩)⟨w∗
r, yiµ⟩+
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξj⟩)⟨w∗
r, ξi⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨w(t)
r , ξj⟩)⟨w∗
r, ξi⟩

≤I3 + I4 + I5 ≤log(2M/ϵ),

where the second equality is by yi ̸= yj.

Lemma C.17. Under Assumption 4.1, we have

∥W(t) −W∗∥2
F −∥W(t+1) −W∗∥≥ηLS(W(t)) −ηϵ.

Proof of Lemma C.17. First, we verify that for j = 0,

⟨∇F0(W(t), xi), W(t)⟩=

m
X

r=1
⟨∇wrF0(W(t), xi), w(t)
r ⟩

=
1
mτ

m
X

r=1
⟨σ′(⟨wr, yiµ⟩)sg(σ(⟨wr, yiµ⟩))yiµ, w(t)
r ⟩

+
1
mτ

m
X

r=1
⟨σ′(⟨wr, ξi⟩)sg(σ(⟨wr, ξi + ϵi⟩))ξi, w(t)
r ⟩

=
1
mτ

m
X

r=1
σ(⟨wr, yiµ⟩)sg(σ(⟨wr, yiµ⟩))

+
1
mτ

m
X

r=1
σ(⟨wr, ξi⟩)sg(σ(⟨wr, ξi + ϵi⟩))

= F0(W(t), xi).
(32)

34


---Page Break---
Similarly, we can show for j = 1, ..., M, it satisfies that

⟨∇Fj(W(t), xi), W(t)⟩= Fj(W(t), xi).
(33)

By the update of W(t), we have

∥W(t) −W∗∥2
F −∥W(t+1) −W∗∥2
F
= 2η⟨∇LS(W(t)), W(t) −W∗⟩−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

M
X

j=0

∂Li(W(t))
∂Fj(W(t), xi)⟨∇Fj(W(t), xi), W(t) −W∗⟩−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

M
X

j=0

∂Li(W(t))
∂Fj(W(t), xi)


Fj(W(t), xi) −⟨∇Fj(W(t), xi), W∗⟩

−η2∥∇LS(W(t))∥2
F

≥2η

n

n
X

i=1


∂Li(W(t))
∂F0(W(t), xi)

 
F0(W(t), xi) −2 log(2M/ϵ)

+

M
X

j=1

∂Li(W(t))
∂Fj(W(t), xi)

 
Fj(W(t), xi) −log(2M/ϵ)


−η2∥∇LS(W(t))∥2
F

≥2η

n

n
X

i=1

 
Li(W(t)) + log(
e2 log(2M/ϵ)

e2 log(2M/ϵ) + Melog(2M/ϵ) )

−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

 
Li(W(t)) −log(1 + ϵ

2)

−η2∥∇LS(W(t))∥2
F

≥ηLS(W(t)) −ηϵ,

where the third equality is by (32) and (33). The first inequality is by Lemma C.16. The second
inequality is due to the convexity of negative log-Softmax function. The last inequality is by Lemma
C.14 (and the conditions on η) and log(1 + x) ≤x for x ≥0.

Lemma C.18. Under Assumption 4.1, let T = T1+⌊∥W(T1)−W∗∥2
F
ηϵ
⌋= T1+ eO(mnσ−2
ξ d−1η−1ϵ−1).

Then we have maxr |γ(t)
r | ≤eO(1/√n) for all T1 ≤t ≤T. In addition, we have

1
t −T1 + 1

t
X

s=T1
LS(W(s)) ≤∥W(T1) −W∗∥2
F
η(t −T1 + 1)
+ ϵ

for all T1 ≤t ≤T. Thus there exists an iterate W(s) for s ∈[T1, T] with training loss smaller than
2ϵ.

Proof of Lemma C.18. By Lemma C.17, for t ∈[T1, T],

∥W(s) −W∗∥2
F −∥W(s+1) −W∗∥≥ηLS(W(t)) −ηϵ

for all s ≤t. Summing over the inequality and dividing both sides by t −T1 + 1 yields

1
t −T1 + 1

t
X

s=T1
LS(W(s)) ≤∥W(T1) −W∗∥2
F + ηϵ(t −T1 + 1)
η(t −T1 + 1)
= ∥W(T1) −W∗∥2
F
η(t −T1 + 1)
+ ϵ ≤2ϵ,

where the last inequality is by the definition of T, for all T1 ≤t ≤T. In addition, we have

T
X

t=T1
LS(W(t)) ≤2∥W(T1) −W∗∥2
F
η
= eO(η−1mnd−1σ−2
ξ ).
(34)

Next we prove the claim that maxr |γ(t)
r | ≤3β, where β = | maxr γ(T1)
r
| = eO(1/√n) for all
T1 ≤t ≤T. Without loss of generality, we only consider r ∈U(0)
+ . First it is evident that at t = T1,

35


---Page Break---
we have maxr γ(t)
r
= β ≤3β. Next suppose there exists eT ∈[T1, T] such that maxr γ(t)
r
≤3β for
all t ∈[T1, eT −1]. Then we let ϕ(t) = maxr γ(t)
r
and we have by the update of γ(t)
r ,

ϕ(t+1) ≤ϕ(t) +
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
) max
r (3

2⟨w(0)
r , µ⟩+ ϕ(t))∥µ∥2
2

≤ϕ(t) +
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)(3

2

p

2 log(8m/δ)σ0∥µ∥2 + ϕ(t))∥µ∥2
2

≤ϕ(t) + η

mτ LS(W(t))(3

2

p

2 log(8m/δ)σ0∥µ∥2 + ϕ(t))∥µ∥2
2,

where the third inequality is by (31).

Now summing over t = T1, ..., eT −1, we have

ϕ( e
T ) ≤ϕ(T1) +

e
T −1
X

t=T1

η
mτ LS(W(t))(3

2

p

2 log(8m/δ)σ0∥µ∥2 + ϕ(t))∥µ∥2
2

≤ϕ(T1) + O(η∥µ∥2
2
mτ
)β

e
T −1
X

t=T1
LS(W(t))

≤ϕ(T1) + O(n∥µ∥2
2d−1σ−2
ξ )β

≤ϕ(T1) + O(nSNR2)β

≤ϕ(T1) + 2β ≤3β

where the second inequality is by induction and the third inequality is by (34). The last inequality is
by the condition of SNR.

C.3
Downstream Task Performance

Recall that after the pre-training stage on the training data at time T, the signal learning and noise
memorization satisfy

max
r
A(T )
r
= eO(1/√n),

max
r
Ψ(T )
r,i = eΩ(1) for i ∈[n].

Then, on the downstream task, the corresponding embedding can be calculated as follows:

hr(x(1)
test) = σ(⟨w(T )
r
, x(1)
test⟩) = eO(1/
√

dn),

hr(x(2)
test) = σ(⟨w(T )
r
, x(2)
test⟩) = eΩ(1/
√

d).

Then, it is straightforward to check that the embedding of a finite size of samples during the fine-tuning
stage is not linearly separable. Thus, the downstream task performance follows LDtest(T ∗) = Θ(1).

D
Multi-Modal Contrastive Learning: Proof of Theorem 4.3

D.1
First Stage

Similar to the single-modal case, in the first stage, the loss derivative is close to its initial value for
both modalities.
Lemma D.1. If max{γ(t)
r , ρ(t)
r,i, eγ(t)
r , eρ(t)
r,i} = O(1), there exists a constant Cℓ> 1 such that

1
Cℓ(1 + M) ≤ℓ′(t)
i
≤
Cℓ
1 + M
1
Cℓ(M + 1) ≤ℓ′(t)
i,j ≤
Cℓ
1 + M

for all i ∈[n].

36


---Page Break---
Proof of Lemma D.1. The proof follows from Lemma C.2.

Lemma D.2. Suppose that γ(t)
r , eγ(t)
r
= O(1) and ρ(t)
r,i, eρ(t)
r,i = O(1) for all r ∈[m] and i ∈[n].
Under Assumption 4.1, then for any δ > 0, with probability at least 1 −δ

|⟨w(t)
r
−w(0)
r , ξi⟩−ρ(t)
r,i| ≤5

r

log(6n2/δ)

d
n

|⟨w(t)
r
−w(0)
r , µ⟩−γ(t)
r | ≤SNR

r

8 log(6n/δ)

d
n,

|⟨ew(t)
r
−ew(0)
r eξi⟩−eρ(t)
r,i| ≤5

s

log(6n2/δ)

ed
n

|⟨ew(t)
r
−ew(0)
r , eµ⟩−eγ(t)
r | ≤]
SNR

s

8 log(6n/δ)

ed
n

for all r ∈[m], i ∈[n].

D.1.1
Dynamics of Signal Learning: Lower Bound

We first analyze the dynamics of signal learning for both two modalities. Similar as in the single-
modal learning, we partition the neurons, depending on the initialization, i.e., U(t)
+
= {r ∈[m] :
⟨w(t)
r , µ⟩> 0} and U(t)
−
= {r ∈[m] : ⟨w(t)
r , µ⟩< 0}, eU(t)
+
= {r ∈[m] : ⟨ew(t)
r , eµ⟩> 0} and
eU(t)
−= {r ∈[m] : ⟨ew(t)
r , eµ⟩< 0}.

Lemma D.3. Under Assumption 4.1 and the same condition as Lemma D.2, for all t > 0, we have

(1) U(t)
+ ∩eU(t)
+ = U(0)
+ ∩eU(0)
+
and γ(t)
r
≥0, eγ(t)
r
≥0 and are increasing.

(2) U(t)
−∩eU(t)
−= U(0)
−∩eU(0)
−and γ(t)
r
≤0, eγ(t)
r
≤0 and are decreasing.

(3) For r ∈U(0)
+ ∩eU(0)
−or r ∈U(0)
−∩eU(0)
+ , there exists a time t′ > 0 such that r ∈U(t′)
+
∩eU(t′)
+
or r ∈U(t′)
−
∩eU(t′)
−.

Proof of Lemma D.3. We first analyze the neurons r ∈U(t)
+ ∩eU(t)
+ . By the update of γ(t)
r

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)σ(⟨ew(t)
r , yieµ⟩)σ′(⟨w(t)
r , yiµ⟩)yi∥µ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , yj eµ⟩)σ′(⟨w(t)
r , yiµ⟩)yi∥µ∥2
2

= γ(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eµ⟩)∥µ∥2
2 ≥γ(t)
r ,
(35)

where the second equality is by yi = yj and the derivative of ReLU activation. Similarly for the other
modality,

eγ(t+1)
r
= eγ(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)σ(⟨w(t)
r , yiµ⟩)σ′(⟨ew(t)
r , yieµ⟩)yi∥eµ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , yjµ⟩)σ′(⟨ew(t)
r , yieµ⟩)yi∥eµ∥2
2

= eγ(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)σ(⟨w(t)
r , µ⟩)∥eµ∥2
2 ≥eγ(t)
r .
(36)

37


---Page Break---
Hence we see at t = 0, the claim is satisfied as γ(1)
r
≥γ(0)
r
for r ∈U(0)
+
∩eU(0)
+ . Then we prove
the claim by induction. Suppose at iteration t the claims are satisfied, i.e., γ(t)
r
≥γ(t−1)
r
≥0 for
r ∈U(0)
+
and r ∈U(t)
+ . Then we can see from (35)

γ(t+1)
r
≥γ(t)
r
≥0.

Similarly, we can show from (36) that eγ(t+1)
r
≥eγ(t)
r
≥0.

Then for the inner product, we have

⟨w(t+1)
r
, µ⟩
(a)
≥γ(t+1)
r
+ ⟨w(0)
r , µ⟩−SNR

r

8 log(6n/δ)

d
n

(b)
≥0.99⟨w(0)
r , µ⟩> 0,

where inequality (a) is by Lemma D.2, and inequality (b) is by Lemma B.2. Similarly, the same result
holds for the other modality. Together, it shows r ∈U(t+1)
+
∩eU(t+1)
+
and the induction is complete.

Similarly, we can use the same strategy to prove claim (2) for r ∈U(0)
−∩eU(0)
−.

Next, we analyze the case where r ∈U(t)
+ ∩eU(t)
−.

γ(t+1)
r
= γ(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)σ(⟨ew(t)
r , yieµ⟩)σ′(⟨w(t)
r , yiµ⟩)yi∥µ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , yj eµ⟩)σ′(⟨w(t)
r , yiµ⟩)yi∥µ∥2
2

= γ(t)
r
−
η
nmτ

n
X

i:yi=1

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , yj eµ⟩)∥µ∥2
2 < γ(t)
r ,
(37)

where the second equality follows from yi ̸= yj. Similarly,

eγ(t+1)
r
= eγ(t)
r
+
η
nmτ

n
X

i=1
(1 −ℓ′(t)
i
)σ(⟨w(t)
r , yiµ⟩)σ′(⟨ew(t)
r , yieµ⟩)yi∥eµ∥2
2

−
η
nmτ

n
X

i=1

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , yjµ⟩)σ′(⟨ew(t)
r , yieµ⟩)yi∥eµ∥2
2

= eγ(t)
r
+
η
nmτ

n
X

i:yi=−1

M
X

j=1
ℓ′(t)
i,j σ(⟨w(t)
r , yjµ⟩)∥eµ∥2
2 > eγ(t)
r .
(38)

Then it is clear that the change of γ(t)
r , eγ(t)
r
does not align with the sign of its initialization. Therefore
there exists some time t′ ≥0 such that either r ∈U(t′)
+
∩eU(t′)
+
or r ∈U(t′)
−
∩eU(t′)
−. We prove
this claim by contradiction. Suppose for all t ≥0, r ∈U(t)
+ ∩eU(t)
−, then by (37) and (38), we
see γ(t+1)
r
≤γ(t)
r
≤0 and eγ(t+1
r
≥eγ(t)
r
≥0. Because ⟨w(t)
r , µ⟩≤1.01⟨w(0)
r , µ⟩+ γ(t)
r
and
⟨ew(t)
r , eµ⟩≥0.99⟨ew(0)
r , eµ⟩+ eγ(t)
r , this raises a contradiction as either ⟨w(t)
r , µ⟩≤0 or ⟨ew(t)
r , eµ⟩≥
0.

With Lemma D.3 at hand, we are ready to demonstrate the lower bound of the growth rate for signal
learning.
Lemma D.4. With the same condition as in Lemma C.2 and Lemma C.3 and n ≥2500 log(4/δ),
define A(t)
r
= γ(t)
r
+ ⟨w(0)
r , µ⟩for r ∈U(0)
+ ; and A(t)
r
= −γ(t)
r
−⟨w(0)
r , µ⟩for r ∈U(0)
−. Similarly,
we define eA(t)
r
= eγ(t)
r
+ ⟨ew(0)
r , eµ⟩for r ∈eU(0)
+
and eA(t)
r
= −eγ(t)
r
−⟨ew(0)
r , eµ⟩for r ∈eU(0)
−. Then
with probability at least 1 −δ, we have

A(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r /Cµ) −1

38


---Page Break---
eA(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(CµA(0)
r
+ eA(0)
r ) −1.

Proof. Without loss of generality, we consider r ∈U(0)
+ ∩eU(0)
+ . Then by the update of γ(t)
r , eγ(t)
r , we
have from (35) and (36)

γ(t+1)
r
= γ(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eµ⟩)∥µ∥2
2
(39)

eγ(t+1)
r
= eγ(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)σ(⟨w(t)
r , µ⟩)∥eµ∥2
2
(40)

To achieve the lower bound, we define

A(t)
r
≜γ(t)
r
+ ⟨w(0)
r , µ⟩,
eA(t)
r
= eγ(t)
r
+ ⟨ew(0)
r , eµ⟩.

By (39), we have the lower bound of update equation,

A(t+1)
r
= A(t)
r
+
η
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eµ⟩)∥µ∥2
2

≥A(t)
r
+
η
nmτ (n

2 −O(√n))(1 −
Cℓ
1 + M ) eA(t)
r ∥µ∥2
2

≥A(t)
r
+ 0.48η∥µ∥2
2
mτ
eA(t)
r .

The inequality is by Lemma B.3, Lemma D.1 where we recall c2 = 1 −
Cℓ
M+1. Similarly, by (40) we
arrive at

eA(t+1)
r
≥eA(t)
r
+ 0.48η∥eµ∥2
2
mτ
A(t)
r
= eA(t)
r
+ 0.48ηC2
µ∥µ∥2
2
mτ
A(t)
r .

Then by combining the above two inequalities, we conclude that

CµA(t+1)
r
+ eA(t+1)
r
= CµA(t)
r
+ eA(t)
r
+ 0.48η∥µ∥2
2
mτ
(Cµ eA(t)
r
+ C2
µA(t)
r )

≥

1 + 0.48η∥µ∥2
2Cµ
mτ


(CµA(t)
r
+ eA(t)
r ).

Thus, we see the joint dynamics of γ(t)
r
and eγ(t)
r
exhibits exponential growth, i.e.,

CµA(t)
r
+ eA(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r ).
(41)

To characterize the dynamics of γ(t)
r
individually, we next track the dynamics of CµA(t)
r
−eA(t)
r
by
subtracting (40) from (39), which gives

CµA(t+1)
r
−eA(t+1)
r
= CµA(t)
r
−eA(t)
r
+ η∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
 
Cµσ(⟨v(t)
r , eµ⟩) −C2
µσ(⟨w(t)
r , µ⟩)


≥

1 −ηCµ∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
 
CµA(t)
r
−eA(t)
r

.

Then from the the propagation, we notice that the gap CµA(t)
r
−eA(t)
r
exhibits exponential decay
regardless of the sign. Thus, we have

CµA(t)
r
−eA(t)
r
≥

1 −ηCµ∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
t 
CµA(0)
r
−eA(0)
r

.
(42)

Therefore, combining (41) and (42) yields:

CµA(t)
r
+ eA(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(CµA(0)
r
+ eA(0)
r ),

39


---Page Break---
CµA(t)
r
−eA(t)
r
≥

1 −η∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
t 
CµA(0)
r
−eA(0)
r

.

Then it concludes that

A(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r /Cµ) +

1 −η∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
t 
A(0)
r
−eA(0)
r /Cµ


≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r /Cµ) −|A(0)
r
−eA(0)
r /Cµ|,

eA(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(CµA(0)
r
+ eA(0)
r ) −

1 −η∥µ∥2
2
nmτ

X

i:yi=1
(1 −ℓ′(t)
i
)
t 
CµA(0)
r
−eA(0)
r


≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(CµA(0)
r
+ eA(0)
r ) −|CµA(0)
r
−eA(0)
r |

where we use the fact that
 
1 −η∥µ∥2
2
nmτ
P

i:yi=1(1 −ℓ′(t)
i
)
t ≤1. Next, we derive an upper bound on

|A(0)
r
−eA(0)
r /Cµ|, |CµA(0)
r
−eA(0)
r | as follows.

|A(0)
r
−eA(0)
r /Cµ| ≤|A(0)
r | + | eA(0)
r |/Cµ ≤2
p

2 log(8m/δ)σ0∥µ∥2 ≤1

where we recall that Cµ∥µ∥2 = ∥eµ∥2 and the second inequality is by the condition on σ0 ≤
0.5(2 log(8m/δ))−1/2∥µ∥−1
2
= eO(∥µ∥−1
2 ). Similarly, we can show |CµA(0)
r
−eA(0)
r | ≤1 and thus
we obtain

A(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r /Cµ) −1
(43)

eA(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(CµA(0)
r
+ eA(0)
r ) −1
(44)

D.1.2
Dynamics of Noise Memorization: Upper Bound

In order to characterize the growth of ρ(t)
r,i, eρ(t)
r,i, we partition the samples according to its sign of inner
product.

I(t)
r,+ = {i ∈[n] : ⟨w(t)
r , ξi⟩> 0},
eI(t)
r,+ = {i ∈[n] : ⟨ew(0)
r , eξi⟩> 0},

I(t)
r,−= {i ∈[n] : ⟨w(t)
r , ξi⟩< 0},
eI(t)
r,−= {i ∈[n] : ⟨v(0)
r , eξi⟩< 0}.

We further define

B(t)
r,+,+ ≜
X

i:yi=1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+∩eI(t)
r,+,

B(t)
r,+,−≜
X

i:yi=1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+∩eI(t)
r,−,

B(t)
r,−,+ ≜
X

i:yi=−1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1(t)

i∈I(t)
r,+∩eI(t)
r,+,

B(t)
r,−,−≜
X

i:yi=−1
(ρ(t)
r,i + ⟨w(0)
r , ξi⟩)1i∈I(t)
r,+∩eI(t)
r,−,

On the other hand, we define

eB(t)
r,+,+ ≜
X

i:yi=1
(eρ(t)
r,i + ⟨ew(0)
r , eξi⟩)1i∈eI(t)
r,+∩I(t)
r,+,

eB(t)
r,+,−≜
X

i:yi=1
(eρ(t)
r,i + ⟨ew(0)
r , eξi⟩)1i∈eI(t)
r,+∩I(t)
r,−,

eB(t)
r,−,+ ≜
X

i:yi=−1
(eρ(t)
r,i + ⟨ew(0)
r , eξi⟩)1i∈eI(t)
r,+∩I(t)
r,+,

40


---Page Break---
eB(t)
r,−,−≜
X

i:yi=−1
(eρ(t)
r,i + ⟨ew(0)
r , eξi⟩)1i∈eI(t)
r,+∩I(t)
r,−.

Before we state the main result, we provide some useful lemmas.
Lemma D.5. Suppose δ > 0, the with probability at least 1 −δ, we have

B(0)
r,+,+ ≤1

2σ0σξ
√

dn(1 +
p

log(1/δ)/2),
B(0)
r,−,+ ≤1

2σ0σξ
√

dn(1 +
p

log(1/δ)/2).

eB(0)
r,+,+ ≤1

2σ0σeξ
√

dn(1 +
p

log(1/δ)/2),
eB(0)
r,−,+ ≤1

2σ0σeξ
√

dn(1 +
p

log(1/δ)/2).

Proof. By Bernstein’s inequality, for arbitrary t > 0, we have

P(|B(0)
r,+,+ −1

2σ0σξ
√

nd| > t) ≤exp(−
t2

2n/4σ2
0σ2
ξd).

Setting t = 1

2σ0σξ
p

nd log(1/δ)/2, we further have

B(0)
r,+ ≤1

2σ0σξ
√

dn(1 +
p

log(1/δ)/2).

Similarly, the same result holds for B(0)
r,−.

Lemma D.6. Under the condition d ≥
300√

2n3 log(6n2/δ)
σ0σξ
√π log(1/(1−(δ/m)2)), then with probability at least 1 −δ,
it satisfies

eB(0)
r,+,+ > 150

r

log(6n2/δ)

d
n2,
B(0)
r,+,+ > 150

r

log(6n2/δ)

d
n2.

Proof of Lemma D.6. Here we want to show eB(0)
r,+,+ ≥150
q

log(6n2/δ)

d
n2 with high probability. To

see this, because eB(0)
r,+,+ is a random variable with positive mean and variance σ2
0σ2
ξdn/4, by Lemma
B.1, we compute

P
  eB(0)
r,+,+ ≤t

≤

s

1 −exp
 
−
8t2

πσ2
0σ2
ξdn

,

Thus when d ≥
300√

2n3 log(6n2/δ)
σ0σξ
√π log(1/(1−(δ/m)2)), we have P
  eB(0)
r,+,+ ≤150
q

log(6n2/δ)

d
n2) ≤δ/m and by

union bound, we have with probability at least 1 −δ, it holds eB(0)
r,+,+ > 150
q

log(6n2/δ)

d
n2.

Lemma D.7. Under the condition n ≥
8 log(1/δ)
log(1/(1−(δ/m)2)), then with probability at least 1 −δ, it
satisfies

|⟨w(0)
r , ξi⟩| > B(0)
r,+,+

n
,
|⟨ew(0)
r , eξi⟩| >
eB(0)
r,+,+

n
.

Proof of Lemma D.7. Here we want to show |⟨w(0)
r , ξi⟩| >
B(0)
r,+,+

n
with high probability. To see this,

because ⟨w(0)
r , ξi⟩is a random variable with positive mean and variance σ2
0σ2
ξd. Besides, by Lemma
C.9, we have

B(0)
r,+,+

n
≤σ0σξ
√

dn(1 +
p

log(1/δ)/2)
n
.

By Lemma B.1, we compute

P
 
|⟨w(0)
r , ξi⟩| ≤t

≤

s

1 −exp
 
−
8t2

πσ2
0σ2
ξd

,

41


---Page Break---
Thus when n ≥
8 log(1/δ)
log(1/(1−(δ/m)2)), we have

P
 
|⟨w(0)
r , ξi⟩| ≤σ0σξ
√

dn(1 +
p

log(1/δ)/2)
n
) ≤δ/m

and by union bound, we have with probability at least 1 −δ, it holds |⟨w(0)
r , ξi⟩| >
B(0)
r,+,+

n
.

Similarly, we can conclude that |⟨ew(0)
r , eξi⟩| >
e
B(0)
r,+,+

n
.

Lemma D.8. Under Assumption 4.1, with probability at least 1 −δ, we have

B(t)
r,+,+ ≤(1 + 1.05η

nmτ )tB(0)
r,+,+,
B(t)
r,+,−≤B(0)
r,+,−,

B(t)
r,−,+ ≤(1 + 1.05η

nmτ )tB(0)
r,−,+,
B(t)
r,−,−≤B(0)
r,−,−,

eB(t)
r,+,+ ≤(1 + 1.05η

nmτ )t eB(0)
r,+,+,
eB(t)
r,+,−≤eB(0)
r,+,−,

eB(t)
r,−,+ ≤(1 + 1.05η

nmτ )t eB(0)
r,−,+,
eB(t)
r,−,−≤eB(0)
r,−,−.

Proof of Lemma D.8. According to the iteration equations for ρ(t)
r,i and eρ(t)
r,i, we have

B(t+1)
r,+,+ = B(t)
r,+,+ +
η
nmτ

X

i:yi=1



(1 −ℓ′(t)
i
)⟨ew(t)
r , eξi⟩−
X

j:yj=−1
ℓ′(t)
i,j ⟨ew(t)
r , eξj⟩1j∈eIr,+



1i∈Ir,+∩eIr,+∥ξi∥2
2

≤B(t)
r,+,+ +
η
nmτ

X

i:yi=1
1i∈Ir,+∩eIr,+

Cℓ(M + 1) −1

Cℓ(M + 1)
(⟨ew(0)
r , eξi⟩+ eρ(t)
r,i + 6

r

log(6n2/δ)

d
n)

−
X

j:yj=−1

1
Cℓ(M + 1)(⟨ew(0)
r , eξj⟩+ eρ(t)
r,j −6

r

log(6n2/δ)

d
n)1j∈eIr,+


(σ2
ξd + σ2
ξ
p

d log(6n2/δ))

≤B(t)
r,+,+ +
η
nτm

Cℓ(M + 1) −1

Cℓ(M + 1)
( eB(t)
r,+,+ +
X

i:yi=1
1i∈Ir,+∩eIr,+6

r

log(6n2/δ)

d
n))

−
X

i:yi=1
1i∈Ir,+∩eIr,+
1
Cℓ(M + 1)( eB(t)
r,−,+ + eB(t)
r,−,−−
X

j̸=i
6

r

log(6n2/δ)

d
n)

(σ2
ξd + σ2
ξ
p

d log(6n2/δ))

≤B(t)
r,+,+ +
η
nmτ


1.01B(t)
r,+,+ −0.992

4Cl
( eB(t)
r,−,+ + eB(t)
r,−,−)

1.035σ2
ξd

≤B(t)
r,+,+ +
η
nmτ


1.05B(t)
r,+,+ −1

4( eB(t)
r,−,+ + eB(t)
r,−,−)

σ2
ξd,

where the first inequality is by Lemma C.2, Lemma C.1 and Lemma B.5. Furthermore, the second
inequality is by Lemma C.8 and the definition of eB(t)
r,−,+ and eB(t)
r,−,−. The third inequality is by

Lemma D.6, eB(0)
r,+,+ > 150
q

log(6n2/δ)

d
n2, d > 10000 log(6n2/δ), and n ≥1280000 log(4/δ). The
last inequality is by choosing Cℓ= 1.01.

Next, we establish the following inequality

B(t+1)
r,+,−= B(t)
r,+,−−
η
nmτ

X

i:yi=1



X

j:yj=−1
ℓ′(t)
i,j ⟨ew(t)
r , eξj⟩1j∈eIr,+



1i∈Ir,+∩eIr,−∥ξi∥2
2

≤B(t)
r,+,−−
η
nmτ

X

i:yi=1
1i∈Ir,+∩eIr,+


X

j:yj=−1

1
Cℓ(M + 1)(⟨ew(0)
r , eξj⟩+ eρ(t)
r,j −6

r

log(6n2/δ)

d
n)1j∈eIr,+



(σ2
ξd −σ2
ξ
p

d log(6n2/δ))

42


---Page Break---
≤B(t)
r,+,−−
η
nτm

X

i:yi=1


1i∈Ir,+∩eIr,+
1
Cℓ(M + 1)( eB(t)
r,−,+ + eB(t)
r,−,−−
X

j̸=i
6

r

log(6n2/δ)

d
n)


(σ2
ξd −σ2
ξ
p

d log(6n2/δ))

≤B(t)
r,+,−−
η
nmτ

0.992

4Cl
( eB(t)
r,−,+ + eB(t)
r,−,−)

0.99σ2
ξd

≤B(t)
r,+,−−
η
nmτ


0.24( eB(t)
r,−,+ + eB(t)
r,−,−)

σ2
ξd,

where the first inequality is by Lemma C.2, Lemma C.1 and Lemma B.5. Furthermore, the second
inequality is by Lemma C.8 and the definition of eB(t)
r,−,+ and eB(t)
r,−,−. The third inequality is by

Lemma D.6 and eB(0)
r,+,+ > 150
q

log(6n2/δ)

d
n2, d > 10000 log(6n2/δ), and n ≥1280000 log(4/δ).

Similarly, we have,

B(t+1)
r,−,+ ≤B(t)
r,−,+ +
η
nmτ


1.05B(t)
r,−,+ −0.25( eB(t)
r,+,+ + eB(t)
r,+,−)

σ2
ξd,

B(t+1)
r,−,−≤B(t)
r,−,−−
η
nmτ


0.24( eB(t)
r,+,+ + eB(t)
r,+,−)

σ2
ξd

For the second modality, with the same derivative, we could obtain the following inequalities:

eB(t+1)
r,+,+ ≤eB(t)
r,+,+ +
η
nmτ


1.05 eB(t+1)
r,+,+ −0.25(B(t)
r,−,+ + B(t)
r,−,−)

σ2
eξd,

eB(t+1)
r,+,−≤eB(t)
r,+,+ −
η
nmτ


0.24(B(t)
r,−,+ + B(t)
r,−,−)

σ2
eξd,

eB(t+1)
r,−,+ ≤eB(t)
r,−,+ +
η
nmτ


1.05 eB(t+1)
r,−,+ −0.25(B(t)
r,+,+ + B(t)
r,+,−)

σ2
eξd,

eB(t+1)
r,−,+ ≤eB(t)
r,−,−−
η
nmτ


0.24(B(t)
r,+,+ + B(t)
r,+,−)

σ2
eξd,

which completes the proof.

We define

Ψ(t)
r,i ≜ρ(t)
r,i + ⟨w(0)
r , ξi⟩, i ∈I(t)
r,+
eΨ(t)
r,i ≜ρ(t)
r,i + ⟨ew(0)
r , eξi⟩, i ∈eI(t)
r,+

Lemma D.9. Under Assumption 4.1, with probability at least 1 −δ, for all 0 ≤t ≤T1,

(1) for i ∈I(0)
r,−∩eI(0)
r,−, we have ρ(t)
r,i = 0, I(t)
r,−= I(0)
r,−and eρ(t)
r,i = 0, eI(t)
r,−= eI(0)
r,−.

(2) for i ∈I(0)
r,−∩eI(0)
r,+, we have ρ(t)
r,i = 0, I(t)
r,−= I(0)
r,−and eρ(t)
r,i ≤0.

(3) for i ∈I(0)
r,+ ∩eI(0)
r,−, we have eρ(t)
r,i = 0, eI(t)
r,−= eI(0)
r,−and ρ(t)
r,i ≤0.

(4) for i ∈I(0)
r,+ ∩eI(0)
r,+, we have

Ψ(t)
r,i ≤(1 +
1.06ησ2
ξd
nmτ
)t(Ψ(t)
r,i + eΨ(t)
r,i),
(45)

eΨ(t)
r,i ≥101Cℓ

M + 1(B(t)
r,+,+ + B(t)
r,+,−).
(46)

Proof of Lemma D.9. We partition the dynamics of ρ(t)
r,i and eρ(t)
r,i into one of the four cases according

to its initialization (1) i ∈I(0)
r,+ ∩eI(0)
r,−(2) i ∈I(0)
r,−∩eI(0)
r,+, (3) i ∈I(0)
r,+ ∩eI(0)
r,+, (4) i ∈I(0)
r,−∩eI(0)
r,−.

43


---Page Break---
(1) In the case of i ∈I(0)
r,−∩eI(0)
r,−, it holds that

ρ(t)
r,i = 0, I(t)
r,−= I(0)
r,−;
eρ(t)
r,i = 0, eI(t)
r,−= eI(0)
r,−.

The proof is by the induction method. It is clear when t = 0, ρ(0)
r,i = 0 and eρ(0)
r,i = 0. Therefore, the
induction argument holds at the initial step.

Suppose at iteration t, we have ρ(t)
r,i = 0 and eρ(t)
r,i = 0. Then we have

⟨ew(t)
r , eξi⟩≤1

2⟨ew(0)
r , eξi⟩+ eρ(t)
r,i = 1

2⟨ew(0)
r , eξi⟩< 0,

⟨w(t)
r , ξi⟩≤1

2⟨w(0)
r , ξi⟩+ ρ(t)
r,i = 1

2⟨w(0)
r , ξi⟩< 0.

Thus by the update of eρ(t+1)
r,i
, we have

eρ(t+1)
r,i
= eρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2 = 0.

Here we have used ⟨ew(t)
r , eξi⟩< 0. Similarly,

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ

M
X

j=1
ℓ′(t)
i,j σ(⟨ew(t)
r , eξj⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2 = 0.

(2) In the case of i ∈I(0)
r,−∩eI(0)
r,+, We use induction. It is clear when t = 0, we have ρ(0)
r,i = 0.

Suppose at iteration t, it holds that ρ(t)
r,i = 0 Then by the propagation of ρ(t)
r,i, we have

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , eξj⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2 = 0.

On the other hand,

eρ(t+1)
r,i
= eρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

= eρ(t)
r,i −
η
nmτ



X

j∈Ir,+
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)



∥eξi∥2
2

≤eρ(t)
r,i,

where the first equation is by σ(⟨w(t)
r , ξi⟩) ≥0.

(3) In the case of i ∈I(0)
r,+ ∩eI(0)
r,−, we use induction to prove such claim. It is clear when t = 0,

ρ(0)
r,i = 0. Suppose at iteration t, we have eρ(t)
r,i = 0. Then we have

⟨ew(t)
r , eξi⟩≤1

2⟨ew(0)
r , eξi⟩+ eρ(t)
r,i = 1

2⟨ew(0)
r , eξi⟩< 0.

44


---Page Break---
Thus by the update of eρ(t+1)
r,i
:

eρ(t+1)
r,i
= eρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2 = 0.

On the other hand,

ρ(t+1)
r,i
= ρ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , eξj⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2,

= ρ(t)
r,i −
η
nmτ



X

j∈eIr,+
ℓ′(t)
i,j σ(⟨ew(t)
r , eξj⟩)



∥ξi∥2
2

≤ρ(t)
r,i,

where the first equation is by σ(⟨ew(t)
r , eξi⟩) ≥0.

(4) Finally, in the case of i ∈I(0)
r,+ ∩eI(0)
r,+, we have

Ψ(t+1)
r,i
≤Ψ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

≤Ψ(t)
r,i +
1.05ησ2
ξd
nmτ
eΨ(t)
r,i.

Similarly, for the other modality,

eΨ(t+1)
r,i
≤eΨ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

≤eΨ(t)
r,i +
1.05ησ2
ξd
nmτ
Ψ(t)
r,i.

Together, we can achieve that

Ψ(t)
r,i + eΨ(t)
r,i ≤(1 +
1.05ησ2
ξd
nmτ
)t(Ψ(0)
r,i + eΨ(0)
r,i ).
(47)

On the other hand size, we calculate the upper bound of Ψ(t)
r,i −eΨ(t)
r,i as follows

Ψ(t+1)
r,i
−eΨ(t+1)
r,i
= Ψ(t)
r,i −eΨ(t)
r,i+
η
nmτ (1 −ℓ′(t)
i
)σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξi⟩)∥ξi∥2
2

−
η
nmτ (1 −ℓ′(t)
i
)σ(⟨w(t)
r , ξi⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

−
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨ew(t)
r , eξi⟩)σ′(⟨w(t)
r , ξj⟩)∥eξi∥2
2

+
η
nmτ

M
X

j̸=i
ℓ′(t)
i,j σ(⟨w(t)
r , ξj⟩)σ′(⟨ew(t)
r , eξi⟩)∥eξi∥2
2

≤Ψ(t)
r,i −eΨ(t)
r,i −
0.96ησ2
ξd
nmτ
(Ψ(t)
r,i −eΨ(t)
r,i) +
1.01ησ2
ξd
nmτ
Cℓ
M + 1(B(t)
r,+,+ + B(t)
r,+,−)

≤(1 −
0.95ησ2
ξd
nmτ
)(Ψ(t)
r,i −eΨ(t)
r,i),

where the first inequality is by Lemma D.2 and Lemma D.1, the second inequality is by induction
(46). Therefore we conclude that

Ψ(t)
r,i −eΨ(t)
r,i ≤(1 −
0.95ησ2
ξd
nmτ
)t(Ψ(0)
r,i −eΨ(0)
r,i ).
(48)

45


---Page Break---
Combining (47) and (48) yields

Ψ(t)
r,i ≤(1 +
1.05ησ2
ξd
nmτ
)t(Ψ(0)
r,i + eΨ(0)
r,i ) + (1 −
0.95ησ2
ξd
nmτ
)t(Ψ(0)
r,i −eΨ(0)
r,i )

≤(1 +
1.06ησ2
ξd
nmτ
)t(Ψ(0)
r,i + eΨ(0)
r,i ).

D.1.3
Signal Learning: Proof of Lemma 5.4

Before proving Lemma 5.4, we require the following lower bound for the initialization. Recall the
definition that A(t)
r
= γ(t)
r
+ ⟨w(0)
r , µ⟩for r ∈U(0)
+ ; and A(t)
r
= −γ(t)
r
−⟨w(0)
r , µ⟩for r ∈U(0)
−.
Similarly, we have eA(t)
r
= eγ(t)
r
+ ⟨ew(0)
r , eµ⟩for r ∈eU(0)
+
and eA(t)
r
= −eγ(t)
r
−⟨ew(0)
r , eµ⟩for r ∈eU(0)
−.

Lemma D.10. Suppose δ > 0 and m ≥eΩ(1). Then with probability at least 1 −δ, we have

1
m

X

r∈U(0)
+ ∩e
U(0)
+

(A(0)
r
+ eA(0)
r /Cµ) ≥0.2σ0∥µ∥2

1
m

X

r∈U(0)
−∩e
U(0)
−

(A(0)
r
+ eA(0)
r /Cµ) ≥0.2σ0∥µ∥2

Proof of Lemma D.10. We first note that ⟨w(0)
r , µ⟩
∼
N(0, σ2
0∥µ∥2
2) and ⟨ew(0)
r , eµ⟩
∼
N(0, σ2
0∥eµ∥2
2). We define the event A = {r ∈[m] : ⟨w(0)
r , µ⟩> 0, ⟨ew(0)
r , eµ⟩> 0}. Then
we can compute

E[⟨w(0)
r , µ⟩1(A) + ⟨ew(0)
r , eµ/Cµ⟩1(A)]

= E[⟨w(0)
r , µ⟩1(A)] + E[⟨ew(0)
r , eµ/Cµ⟩1(A)]

= 1

2E[⟨w(0)
r , µ⟩1(⟨w(0)
r , µ⟩> 0)] + 1

2E[⟨ew(0)
r , eµ/Cµ⟩1(⟨ew(0)
r , eµ⟩> 0)]

= σ0∥µ∥2
√

2π

where we use the independence of neurons in two modalities. Let S := Pm
r=1⟨w(0)
r , µ⟩1(A) +
⟨ew(0)
r , eµ/Cµ⟩1(A). Then we apply the sub-Gaussian concentration inequality that with probability
at least 1 −δ/2,


X

r∈A

 
⟨w(0)
r , µ⟩+ ⟨ew(0)
r , eµ/Cµ⟩

−mσ0∥µ∥2
√

2π

 ≤eO(m−1/2).

Then suppose m = eΩ(1), we have

1
m

X

r∈A

 
⟨w(0)
r , µ⟩+ ⟨ew(0)
r , eµ/Cµ⟩

≥0.2σ0∥µ∥2.

Similarly, we can show the same for the event where ⟨w(0)
r , µ⟩< 0, ⟨ew(0)
r , eµ⟩< 0 and taking the
union bound completes the proof.

Proof of Lemma 5.4. From the upper bound on noise memorization (45), we take the maximum over
r, i, which gives

max
r,i Ψ(t)
r,i ≤

1 + 1.06
ησ2
ξd
nmτ

t
max
r,i Ψ(0)
r,i

≤

1 + 1.06
ησ2
ξd
nmτ

t
2
p

log(8mn/δ)σ0σξ
√

d.

46


---Page Break---
At the same time, for signal learning, from the lower bound on signal learning in Lemma D.4, we
have for the first modality that

A(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
(A(0)
r
+ eA(0)
r /Cµ) −1

Taking a summation over the r ∈U(0)
+ ∩eU(0)
+ , we obtain

1
m

X

r∈U(0)
+ ∩e
U(0)
+

A(t)
r
≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t 1

m

X

r∈U(0)
+ ∩e
U(0)
+

(A(0)
r
+ eA(0)
r /Cµ) −1

≥

1 + 0.48η∥µ∥2
2Cµ
mτ

t
0.2σ0∥µ∥2 −1

where the second inequality is due to Lemma D.10.

Under the SNR condition n · SNR2 ≥1.7 and Cµ > 2.66, we can see there exists a scale difference
between maxr,i Ψ(t)
r,i and 1

m
P

r∈U(0)
+ ∩e
U(0)
+ A(t)
r
at the end of first stage. Let

T1 = log
 
20/(σ0∥µ∥2)

/ log
 
1 + 0.48Cµ
η∥µ∥2
2
mτ

.

Then we have
1
m
P

r∈U(0)
+ ∩e
U(0)
+ A(t)
r
reach 3 within T1 iterations. Using similar analysis, we can

show at the same time 1

m
P

r∈U(0)
−∩e
U(0)
−A(t)
r , 1

m
P

r∈U(0)
+ ∩e
U(0)
+
eA(t)
r , 1

m
P

r∈U(0)
−∩e
U(0)
−
eA(t)
r
also reach
3.

On the other hand, we compute the scale of maxr,i Ψ(T1)
r,i
as

max
r
Ψ(T1)
r,i
≤

1 + 1.06
ησ2
ξd
nmτ

t
2
p

log(8mn/δ)σ0σξ
√

d

= exp

log(1 + 1.06
ησ2
ξd
nmτ )

log(1 + 0.48Cµ
η∥µ∥2
2
mτ )
log
 
20/(σ0∥µ∥2)

2
p

log(8mn/δ)σ0σξ
√

d

≤exp
 
(2.21/(Cµn · SNR2) + O((
ησ2
ξd
nmτ ))2) log
 
20/(σ0∥µ∥2)

2
p

log(8mn/δ)σ0σξ
√

d

≤exp
 
(2.21/(Cµn · SNR2) + 0.01) log
 
20/(σ0∥µ∥2)

2
p

log(8mn/δ)σ0σξ
√

d

≤exp
 
0.5 log
 
20/(σ0∥µ∥2)

2
p

log(8mn/δ)σ0σξ
√

d

=
p

24 log(8mn/δ)
√σ0σξ
√

d
p

∥µ∥2

= eO(n−1/2),

where we choose η sufficiently small for the second inequality. In third inequality, we have applied
the condition that nSNR2 = Θ(1) and σ0 ≤
1
∥µ∥2 . The last inequality is by the SNR condition.

Because we can choose n ≥C log(m/δ) for sufficiently large constant C, maxr,i Ψ(T1)
r,i
= o(1).

D.2
Second Stage

We first show a similar result as in Lemma C.14 for both two modalities.
Lemma D.11. Under conditions, for 0 ≤t ≤T ∗, we have

∥∇LS(W(t))∥2
F ≤O(max{∥µ∥2
2, σ2
ξd})LS(W(t)),

∥∇LS(f
W(t))∥2
F ≤O(max{∥eµ∥2
2, σ2
eξd})LS(f
W(t)).

Proof of Lemma D.11. The proof follows from that of Lemma C.14 and hence is omitted for clarity.

47


---Page Break---
For notation convenience, we let

F0(W, xi) = Simh,g(xi, exi)/τ

=
1
mτ

m
X

r=1
σ(⟨wr, yiµ⟩)sg(σ(⟨ewr, yieµ⟩)) +
1
mτ

m
X

r=1
σ(⟨wr, ξi⟩)sg(σ(⟨ewr, eξi⟩))

Fj(W, xi) = Simh,g(xi, exj)/τ

=
1
mτ

m
X

r=1
σ(⟨wr, yiµ⟩)sg(σ(⟨ewr, yj eµ⟩)) +
1
mτ

m
X

r=1
σ(⟨wr, ξi⟩)sg(σ(⟨ewr, eξj⟩)), for j = 1, ..., M

F0(f
W, exi) = Simg,h(exi, xi)/τ

=
1
mτ

m
X

r=1
sg(σ(⟨wr, yiµ⟩))σ(⟨ewr, yieµ⟩) +
1
mτ

m
X

r=1
sg(σ(⟨wr, ξi⟩))σ(⟨ewr, eξi⟩)

eFj(f
W, exi) = Simg,h(exi, xj)/τ

=
1
mτ

m
X

r=1
sg(σ(⟨wr, yjµ⟩))σ(⟨ewr, yieµ⟩) +
1
mτ

m
X

r=1
sg(σ(⟨wr, ξj⟩))σ(⟨ewr, eξi⟩), for j = 1, ..., M

It is worth mentioning that Fj(W, xi) = eFj(f
W, exi) in terms of numerical values. They differ in
terms of the derivatives.

We further denote

LS(W) = −1

n

n
X

i=1
Li(W) = −1

n

n
X

i=1
log

eSimh,g(xi,exi)/τ

eSimh,g(xi,exi)/τ + PM
j̸=i eSimh,g(xi,exj)/τ


,

where Li(W) = −log

eSimh,g(xi,exi)/τ

eSimh,g(xi,exi)/τ + PM
j̸=i eSimh.g(xi,xj)/τ


= −log

eF0(W,xi)

eF0(W,xi) + PM
j=1 eFj(W,xi)



LS(f
W) = −1

n

n
X

i=1
Li(f
W) = −1

n

n
X

i=1
log

eSimg,h(exi,xi)/τ

eSimg,h(exi,xi)/τ + PM
j̸=i eSimg,h(exi,xj)/τ


,

where Li(f
W) = −log

eSimg,h(exi,xi)/τ

eSimg,h(exi,xi)/τ + PM
j̸=i eSimg,h(exi,xj)/τ


= −log

eF0(f
W,exi)

eF0(f
W,exi) + PM
j=1 eFj(f
W,exi)



L(W, V) = LS(W) + LS(f
W)

Here L(W, f
W) is the combined loss function for two modalities.

Let θr = 1 if r ∈U(0)
+ , i.e., ⟨w(0)
r , µ⟩> 0 and θr = −1 if r ∈U(0)
−, i.e., ⟨w(0)
r , µ⟩< 0. Similarly,
we let eθr = 1 if r ∈eU(0)
+
and eθr = −1 if r ∈eU(0)
−. Then we define

w∗
r = w(0)
r
+ 2τ log(2M/ϵ) · θr ·
µ
∥µ∥2
2
,

ew∗
r = ew(0)
r
+ 2τ log(2M/ϵ) · eθr ·
eµ
∥eµ∥2
2
.

Lemma D.12. Under Assumption 4.1, we have ∥W(T1) −W∗∥F ≤eO(m1/2∥µ∥−1
2 ) and ∥f
W(T1) −
f
W∗∥F ≤eO(m1/2∥eµ∥−1
2 )

Proof of Lemma D.12. The proof follows exactly the same as the proof in single-modal case. we
include it here for completeness. Without loss of generality, we focus on the case for W(T1).

By the scale difference at T1, we have

∥W(T1) −W∗∥F ≤∥W(T1) −W(0)∥F + ∥W(0) −W∗∥F

≤
X

r

γ(T1)
r
∥µ∥2
+
X

r,i

|ρ(T1)
r,i |
∥ξi∥2
+ O(m1/2 log(1/ϵ))∥µ∥−1
2

48


---Page Break---
≤O(m∥µ∥−1
2 ) + O(nmσ0) + O(m1/2 log(1/ϵ)∥µ∥−1
2 )

≤eO(m1/2∥µ∥−1
2 )

where the first inequality is by triangle inequality and the second inequality is by decomposition of
W(T1) and W∗. The third inequality is by the bound on γ(T1)
r
and ρ(T1)
r,i
and Lemma B.5. The last
inequality is by condition on σ0.

Lemma D.13. Under Assumption 4.1, we have for all t ∈[T1, T ∗],

⟨∇F0(W(t), xi), W∗⟩≥2 log(2M/ϵ)

⟨∇Fj(W(t), xi), W∗⟩≤log(2M/ϵ), for j = 1, ..., M

⟨∇F0(f
W(t), xi), f
W∗⟩≥2 log(2M/ϵ)

⟨∇Fj(f
W(t), xj), f
W∗⟩≤log(2M/ϵ), for j = 1, ..., M

Proof of Lemma D.13. The proof follows similarly from Lemma C.16 and here we only show the
result for the first modality. Based on the definition of W∗and Fj(W(t), xi), we can derive for
j = 0,

⟨∇F0(W(t), xi), W∗⟩

=

m
X

r=1
⟨∇wrF0(W(t), xi), w∗
r⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨ew(t)
r , yieµ⟩)⟨w∗
r, yiµ⟩+
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨ew(t)
r , eξi⟩)⟨w∗
r, ξi⟩

=
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨ew(t)
r , yieµ⟩)

⟨w(0)
r , yiµ⟩+ 2τ log(2M/ϵ)θryi


+
1
mτ

m
X

r=1
σ′(⟨w(t)
r , ξi⟩)σ(⟨ew(t)
r , eξi⟩)

⟨w(0)
r , ξi⟩+ 2τθr log(2M/ϵ)⟨ξi, yiµ⟩∥µ∥−2
2


≥
1
mτ

m
X

r=1
σ′(⟨w(t)
r , yiµ⟩)σ(⟨ew(t)
r , yieµ⟩)2τ log(2M/ϵ)θryi

|
{z
}
I1

−1

mτ

m
X

r=1
σ(⟨ew(t)
r , yieµ⟩) eO(σ0∥µ∥2)

|
{z
}
I2

−
1
mτ

m
X

r=1
σ(⟨ew(t)
r , eξi⟩)2τ log(2M/ϵ) eO(σξ∥µ∥−1
2 )

|
{z
}
I3

−1

mτ

m
X

r=1
σ(⟨ew(t)
r , eξi⟩) eO(σ0σξ
√

d)

|
{z
}
I4

.

First, we can bound I2 ≤eO(σ0∥µ∥2), I3 ≤log(2M/ϵ) eO(σξ∥µ∥−1
2 ), I4 ≤eO(σ0σξ
√

d) by the
global bound on σ(⟨ew(t)
r , eξi⟩), σ(⟨ew(t)
r , yieµ⟩) = eO(1).

Further, we lower bound I1 as follows. Without loss of generality, we suppose yi = 1, then we have

I1 ≥
1
mτ

X

r∈U(0)
+ ∩e
U(0)
+

σ(⟨ew(t)
r , yieµ⟩)2τ log(2M/ϵ) ≥4 log(2M/ϵ)

where the last inequality is by Lemma 5.4 and the monotonicity of eγ(t)
r .

Then we can obtain

⟨∇F0(W(t), xi), W∗⟩≥4 log(2M/ϵ) −I2 −I3 −I −4 ≥2 log(2M/ϵ).

The proof for Fj(W(t), W∗) follows the same argument as in Lemma C.16.

Lemma D.14. Under Assumption 4.1, we have for all t ∈[T1, T ∗],

∥W(t) −W∗∥2
F + ∥f
W(t) −f
W∗∥2
F −∥W(t+1) −W∗∥2
F −∥f
W(t+1) −f
W∗∥2
F ≥ηL(W(t), f
W(t)) −2ηϵ

49


---Page Break---
Proof of Lemma D.14. First, we see that L(W(t), f
W(t)) = LS(W(t)) + LS(f
W(t)) is decompos-
able in terms of W(t) and f
W(t). This suggests that ∇WL(W(t), f
W(t)) = ∇LS(W(t)) and
∇f
WL(W(t), f
W(t)) = ∇eLS(f
W(t)). Then following similar analysis as in Lemma C.17, we can first
show

⟨Fj(W(t), xi), W(t)⟩= Fj(W(t), xi), for j = 0, ..., M,
(49)

⟨Fj(f
W(t), exi), f
W(t)⟩= Fj(f
W(t), exi), for j = 0, ..., M.
(50)

Then by the gradient descent update

∥W(t) −W∗∥2
F −∥W(t+1) −W∗∥2
F
= 2η⟨∇LS(W(t)), W(t) −W∗⟩−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

M
X

j=0

∂Li(W(t))
∂Fj(W(t), xi)⟨∇Fj(W(t), xi), W(t) −W∗⟩−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

M
X

j=0

∂Li(W(t))
∂Fj(W(t), xi)


Fj(W(t), xi) −⟨∇Fj(W(t), xi), W∗⟩

−η2∥∇LS(W(t))∥2
F

≥2η

n

n
X

i=1


∂Li(W(t))
∂F0(W(t), xi)

 
F0(W(t), xi) −2 log(2M/ϵ)

+

M
X

j=1

∂Li(W(t))
∂Fj(W(t), xi)

 
Fj(W(t), xi) −log(2M/ϵ)


−η2∥∇LS(W(t))∥2
F

≥2η

n

n
X

i=1

 
Li(W(t)) + log(
e2 log(2M/ϵ)

e2 log(2M/ϵ) + Melog(2M/ϵ) )

−η2∥∇LS(W(t))∥2
F

= 2η

n

n
X

i=1

 
Li(W(t)) −log(1 + ϵ

2)

−η2∥∇LS(W(t))∥2
F

≥ηLS(W(t)) −ηϵ

where the third equality is by (49). The first inequality is by Lemma D.13. The second inequality is
due to the convexity of negative log-Softmax function. The last inequality is by Lemma D.11 (and
the conditions on η) and log(1 + x) ≤x for x ≥0.

Similarly, we can show the same for the other modality as

∥f
W(t) −f
W∗∥2
F −∥f
W(t+1) −f
W∗∥2
F ≥ηLS(f
W(t)) −ηϵ

Combining the two results completes the proof.

Lemma D.15. Under Assumption 4.1, let T = T1 + ⌊∥W(T1)−W∗∥2
F +∥f
W(T1)−f
W∗∥2
F
ηϵ
⌋= T1 +
eO(mη−1ϵ−1∥µ∥−2
2 ). Then we have maxr,i |ρ(t)
r,i| ≤σ0σξ
√

d and maxr,i |eρ(t)
r,i| ≤σ0σξ
√

d for all
t ∈[T1, T]. In addition, we have for all T1 ≤t ≤T,

1
t −T1 + 1

t
X

s=T1

L(W(t), f
W(t)) ≤∥W(T1) −W∗∥2
F + ∥f
W(T1) −f
W∗∥2
F
η(t −T1 + 1)
+ 2ϵ.

Therefore, we can find an iterate (W(s), f
W(s)) for s ∈[T1, T] with training loss smaller than 3ϵ.

Proof of Lemma D.15. By Lemma D.14, for t ∈[T1, T], we have for any s ≤t

∥W(s) −W∗∥2
F + ∥f
W(s) −f
W∗∥2
F −∥W(s+1) −W∗∥2
F −∥f
W(s+1) −f
W∗∥2
F ≥ηL(W(s), f
W(s)) −2ηϵ.

Summing the inequality yields

t
X

s=T1

L(W(s), f
W(s)) ≤∥W(T1) −W∗∥2
F + ∥f
W(T1) −f
W∗∥2
F + 2ηϵ(t −T1 + 1)
η
.

50


---Page Break---
Dividing both sides by t −T1 + 1 and setting t = T gives

1
T −T1 + 1

t
X

s=T1

L(W(s), f
W(s)) ≤∥W(T1) −W∗∥2
F + ∥f
W(T1) −f
W∗∥2
F
η(T −T1 + 1)
+ 2ϵ ≤3ϵ.

D.3
Downstream Task Performance

Recall that after the pre-training stage on the training data at time T, the signal learning and noise
memorization satisfy

max
r
A(T )
r
= eΩ(1),

max
r
Ψ(T )
r,i = eO(1/√n) for i ∈[n].

Then, on the downstream task, the corresponding embedding can be calculated as follows:

hr(x(1)
test) = σ(⟨w(T )
r
, x(1)
test⟩) = eΩ(1/
√

d),

hr(x(2)
test) = σ(⟨w(T )
r
, x(2)
test⟩) = eO(1/
√

dn).

Then, it is straightforward to check that the embedding of a finite size of samples during the fine-tuning
stage is linearly separable. Thus, the downstream task performance follows LDtest(T ∗) = o(1).

E
Additional Experimental Details

We implement our methods using PyTorch. For the software and hardware configurations, we ensure
consistent environments for each dataset. We run all the experiments on Linux servers with NVIDIA
V100 graphics cards and CUDA 11.2, completing them within one hour.

51


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly outline the primary contributions of the
paper, including the theoretical advancements in optimization and generalization analysis of
multi-modal contrastive learning and single-modal contrative learning.

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

Justification: We have discussed the limitation of this work in Section 7.

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

52


---Page Break---
Answer: [Yes]
Justification: We have provided all assumptions in Assumption 4.1 in the main paper. The
proof sketch and complete proof are provided in Section 5 and Appendices C and D.
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
Justification: We have provided the complete configuration in Section 6. We have also
uploaded the code in the supplementary material.
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

53


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We have uploaded the code with instructions in the supplementary material.

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

Justification: We have provided all the training and test details in Section 6 and uploaded
code.

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

Justification: We plot 1-sigma error bar for results shown in Figure 1.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

54


---Page Break---
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

Justification: We have provided sufficient information about computer resources in Appendix
E.
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

Justification: The research does not involve any human subjects, personal data, or interactions
that would raise ethical concerns about consent, privacy, or respect for persons. In conclusion,
the research aligns with the ethical principles outlined in the NeurIPS Code of Ethics.
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
Justification: We have discussed the broader impacts in Section A.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

55


---Page Break---
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
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

56


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the asshts?
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
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

57


---Page Break---
