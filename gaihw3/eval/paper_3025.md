Provable and Efficient Dataset Distillation
for Kernel Ridge Regression

Yilan Chen
UCSD CSE
yic031@ucsd.edu

Wei Huang
RIEKN AIP
wei.huang.vr@riken.jp

Tsui-Wei Weng
UCSD HDSI
lweng@ucsd.edu

Abstract

Deep learning models are now trained on increasingly larger datasets, making it
crucial to reduce computational costs and improve data quality. Dataset distillation
aims to distill a large dataset into a small synthesized dataset such that models
trained on it can achieve similar performance to those trained on the original
dataset. While there have been many empirical efforts to improve dataset distillation
algorithms, a thorough theoretical analysis and provable, efficient algorithms are
still lacking. In this paper, by focusing on dataset distillation for kernel ridge
regression (KRR), we show that one data point per class is already necessary and
sufficient to recover the original model’s performance in many settings. For linear
ridge regression and KRR with surjective feature mappings, we provide necessary
and sufficient conditions for the distilled dataset to recover the original model’s
parameters. For KRR with injective feature mappings of deep neural networks, we
show that while one data point per class is not sufficient in general, k+1 data points
can be sufficient for deep linear neural networks, where k is the number of classes.
Our theoretical results enable directly constructing analytical solutions for distilled
datasets, resulting in a provable and efficient dataset distillation algorithm for KRR.
We verify our theory experimentally and show that our algorithm outperforms
previous work such as KIP while being significantly more efficient, e.g. 15840×
faster on CIFAR-100.

1
Introduction

Deep learning models are now trained on increasingly massive datasets, incurring substantial com-
putational costs and data quality challenges. For instance, Llama 3 was pre-trained on over 15
trillion tokens, while the training of GPT-4 exceeded $100 million. Reducing these burdens is crucial.
Dataset distillation [34] aims to distill a large dataset into a small synthesized dataset such that models
trained on it can achieve similar performance to those trained on the original dataset. A good small
distilled dataset is not only useful in saving computational cost and improving data quality but also
has various applications such as continual learning [39, 40, 35], privacy protection [24, 40, 18, 5, 1],
and neural architecture search [31, 39].

While there have been many empirical efforts to improve dataset distillation algorithms [24, 39, 2, 38],
a thorough theoretical analysis is still lacking. Izzo and Zou [9] show single distill data is sufficient
for a class of generalized linear models with one-dimensional output, where the data is assumed
to follow a generalized exponential density function and the negative log-likelihood is optimized
by gradient descent. For a linear ridge regression (LRR), Izzo and Zou [9] show d data points are
needed to recover the original model’s parameter for all regularization at the same time, where d is the
dimension of the data and is large even for small datasets like MNIST [13] and CIFAR [12] (d = 784
and 3072) in computer vision. For kernel regression with Gaussian kernel, they show n data points
are necessary, where n is the number of original data points and can be large for modern datasets, e.g.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: Comparison with existing theoretical analysis of dataset distillation. The number of distilled
data needed to recover original model’s performance and models analyzed. “-” means not applicable.
For linear ridge regression (LRR) and kernel ridge regression (KRR) with subjective feature mapping,
our results only need one distilled data per class (k ≤d in our setting), which is far less than the
existing work [9, 21] that require n or p points. As an example, k = 10, d = 3072, n = 50000
for CIFAR-10. The k, d, n of standard datasets are listed in Table 2. p is the dimension of feature
mapping ϕ : Rd 7→Rp.

LRR
Kernel ridge regression (KRR)

surjective ϕ
non-surjective ϕ

Izzo and Zou [9]
d
-
n (Gaussian Kernel)

Maalouf et al. [21]
-
-
p (Shift-invariant Kernels)

Our work
k, (k ≤d)
k, (k
≤
p)
(Invertible
NNs, FCNN, CNN, Random
Fourier Features)

p in general (Deep nonlinear
NNs).
k + 1 for deep linear NNs

n = 60000 and 50000 for MNIST and CIFAR (see Table 2). Maalouf et al. [21] use Random Fourier
Features (RFF) to approximate shif-invariant kernels that may have an infinite-dimensional feature
space, and construct p distilled data for such RFF model, where p is the dimension of the RFF model
and can be Ω(√n log n) in general cases. The results in [9, 21], however, have a large gap compared
with the empirical evidence that one data point per class can often achieve comparable performance
to the original model [24, 39, 2, 38].

In this paper, by focusing on dataset distillation for kernel ridge regression (KRR), we show that one
data point per class is already necessary and sufficient to recover the original model’s performance
in many settings, which is far less than n or p data points needed in prior works [9, 21]. Besides, our
analysis is more general than prior works [9, 21] and can handle more and different models, including
invertible neural networks, fully-connected neural networks (FCNN), Convolutional neural networks
(CNN), and Random Fourier Features (RFF). Table 1 compares our theoretical results with previous
analysis. We summarize our contributions as follows.

• In Sec. 4.1 and 5, for linear ridge regression (LRR) and KRR with surjective feature mappings,
we show that one distilled data point per class is necessary and sufficient to recover the original
model’s parameters and provide necessary and sufficient conditions for such distilled datasets. In
addition, we show how to find distilled data that is close to real data in Sec. 4.2.
• In Sec. 5.2, for KRR with injective feature mappings of deep neural networks (NNs), we show
that one data point per class is in general not sufficient to recover the original model’s parameters.
However, k + 1 data points can be sufficient for deep linear NNs, where k is the number of classes.
• Our theoretical results enable us to directly construct analytical solutions for the distilled datasets,
resulting in a provable and efficient dataset distillation algorithm for KRR in Algorithm 1. We
verify our theory experimentally and show that our algorithm outperforms previous SOTA dataset
distillation algorithm KIP [25] while being significantly more efficient, e.g. 15840× faster on
CIFAR-100.
• In Sec.6, we show our theoretical results can be used for several applications. First, it can be used
as necessary or sufficient conditions for KIP-type algorithms to converge to a global minimum
even if the loss function is highly non-convex. Second, our distilled dataset for KRR can provably
preserve the privacy of the original dataset while having a performance guarantee.

2
Related works

Dataset distillation. Dataset distillation aims to distill a large dataset into a small synthesized dataset
such that models trained on it can achieve similar performance to those trained on the original dataset.
Previous approaches can be mainly divided into four categories [29]: 1) Meta-model Matching: this
category formulates the problem as a bilevel optimization problem and maximize the performance of
the model trained on the distilled dataset [34]. Some recent works such as KIP [24, 25], FRePo [40],
RFAD [17], and RCIG [18] approximate the inner loop optimization of training neural networks by

2


---Page Break---
KRR with Neural Tangent Kernel [10] or neural network Gaussian process (NNGP) kernels [14]. 2)
Gradient Matching: this category minimizes the distance between the gradients of models trained
on the original dataset and distilled dataset [39, 37, 15, 11]. 3) Trajectory Matching: this category
aims to match the training trajectories of models trained on the original dataset and distilled dataset
[2, 4, 8, 7]. 4) Distribution Matching: this approach directly matches the distribution of the original
dataset and distilled dataset via a single-level optimization [38, 33, 36]. Our work is closely related
to kernel-based dataset distillation algorithms [24, 25, 40, 17, 18] in category (1). Our theoretical
analysis provides theoretical foundations and implications for these kernel-based algorithms.

Theoretical analysis of dataset distillation. In addition to the papers discussed in the introduction,
Maalouf et al. [19] propose an efficient algorithm to construct a d2 + 1 core set of the original dataset
for least mean squares problems. Maalouf et al. [20] further propose to use the SVD of the original
dataset to construct a distilled dataset of size d. Tukan et al. [32] utilize the idea of subset selection to
improve the initialization and training procedure of dataset distillation. Our paper focuses on KRR
and constructs k distilled data analytically, where k is usually much less than d (see Table 2).

3
Preliminaries

3.1
Dataset Distillation

Table 2: k (number of class), d (dimen-
sion of data), and n (number of training
data) of standard datasets.

Dataset
k
d
n

MNIST [13]
10
784
60000
CIFAR-10 [12]
10
3072
50000
CIFAR-100 [12]
100
3072
50000
ImageNet-1k [28]
1000
196608
1281167

For an original dataset {xi, yi}n
i=1, we denote X =
[x1, . . . , xn] ∈Rd×n and Y = [y1, . . . , yn] ∈Rk×n,
where d is the dimension of the data, k is the dimension
of the label or the number of the classes, and n is the
number of data points. The goal of dataset distillation is to
create a synthetic dataset XS = [xS1, . . . , xSm] ∈Rd×m

and YS = [yS1, . . . , ySm] ∈Rk×m, with the number of
distilled data points m ≪n, such that a model trained
on this synthetic dataset (XS, YS) can achieve similar
performance to those trained on the original dataset.

As the data dimension is usually larger than the label dimension in practice, e.g. MNIST has
d = 728, k = 10 and other datasets have even larger d, we consider d ≥k in this paper. For a matrix
A, we use A+ to denote its pseudoinverse and Range (A) to denote its range space.

3.2
Dataset Distillation for Kernel Ridge Regression (KRR)

Original model: Given a kernel K(x, x′) = ⟨ϕ(x), ϕ(x′)⟩, where ϕ : Rd 7→Rp is the feature
mapping from input space to a feature space of dimension p, a KRR model f(x) = Wϕ(x) trained
on original data set with a predefined regularization λ ≥0 tries to minimize following objective

min
W ∥Wϕ(X) −Y∥2
F + λ ∥W∥2
F

where W ∈Rk×p and ϕ(X) = [ϕ(x1), . . . , ϕ(xn)] ∈Rp×n. The solution can be computed
analytically as W = Yϕλ(X)+, where

ϕλ(X)+ =

(K(X, X) + λIn)−1 ϕ(X)⊤= ϕ(X)⊤ 
ϕ(X)ϕ(X)⊤+ λIp
−1 ,
if λ > 0,
ϕ(X)+,
if λ = 0.

and K(X, X) = ϕ(X)⊤ϕ(X) ∈Rn×n. ϕλ(X) can be considered as regularized features. Linear
ridge regression is a special case of kernel ridge regression (KRR) with ϕ(x) = x.

KRR is used in many dataset distillation algorithms [24, 25, 40, 17, 18]. In this paper, we mainly
consider a finite-dimensional ϕ. This matches the practical neural networks which are usually used in
dataset distillation. For shift-invariant kernels with infinite-dimensional RKHS space, e.g. Gaussian
kernel, they can be well approximated by finite-dimensional random Fourier features [27, 16].

Distilled dataset model: Similarly, a KRR trained on distilled dataset with regularization λS ≥0 is
fS(x) = WSϕ(x), where WS = YSϕλS(XS)+ ∈Rk×d. Additionally, denote XλS = ϕλS(XS)
with ϕ(x) = x, i.e.

X+
λS =
  
X⊤
S XS + λSIm
−1 X⊤
S = X⊤
S
 
XSX⊤
S + λSId
−1 ,
if λS > 0,
X+
S ,
if λS = 0.

3


---Page Break---
The goal of dataset distillation here is to find (XS, YS) such that WS = W, where W is supposed
to be given, i.e. can be computed from the original dataset (X, Y).

4
Dataset Distillation for Linear Ridge Regression (LRR)

In this section, we first analyze the dataset distillation for the linear ridge regression (LRR). In
Sec. 4.1, for a LRR model, we show that k distilled data points (one per class) are necessary and
sufficient to guarantee WS = W. We provide analytical solutions of such XS allowing us to
compute the distilled dataset analytically instead of having to learn it heuristically in existing works
[24, 25, 40, 17, 18]. Then, in Sec 4.2, we show how to find distilled data that is close to real data.
Lastly, for fixed data and only distilling labels, we show d points are needed in Sec 4.3.

4.1
Analytical Computation for Linear Ridge Regression

Theorem 4.1. When m < k, there is no XS can guarantee WS = W unless the columns of
W are in the range space of YS. When m ≥k and YS is rank k, let r = min(m, d) and take
D = Y+
S W +
 
Im −Y+
S YS

Z, where Z ∈Rm×d is any matrix of the same size as X⊤
S . Suppose
the reduced SVD of D is D = Vdiag(σ′
1, . . . , σ′
r)U⊤with σ′
1 ≥· · · ≥σ′
r ≥0, the following results
hold:

1. λS > 0: WS = W if and only if, for any D defined above, λS ≤
1
4σ′2
1
and XS =

Udiag(σ1, . . . , σr)V⊤where σi =

( 0,
if σ′
i = 0,

1±√

1−4λSσ′2
i
2σ′
i
,
otherwise.

2. λS = 0: WS = W if and only if XS = D+ for any D defined above.

Proof sketch. The proof can be found in Appendix C. The key idea is that we want to solve XS from
WS = YSX+
λS = W (Note: W is given and we can select/decide YS). When m < k, this is an
overdetermined system for X+
λS. There is no solution for X+
λS in general therefore no solution for XS.
When m ≥k and YS is rank k, the solutions of X+
λS are given by X+
λS = Y+
S W+
 
Im −Y+
S YS

Z,
where Z ∈Rm×d is any matrix of the same size as X+
λS. However, not all such X+
λS corresponds to
a XS. To solve XS, we assume we have the SVD of XS and solve it from the SVD of X+
λS.

Intuitively, original dataset (X, Y) is compressed into XS through original model’s parameter
W = Y
 
X⊤X + λIn
−1 X⊤. When m = k, i.e. one distilled data per class, D = Y+
S W is
deterministic and XS is fully determined by W and YS. In this case, when λS = 0 and W is
full rank, XS can be easily computed as XS = W+YS. As an example, Figure 1 shows the
distilled data for MNIST and CIFAR-100 when m = 10/100. When m > k, i.e. more than one
distilled data per class, there exist infinitely many distilled datasets since Z is a free variable to
choose. When m = n, one can verify that X is a distilled dataset for itself by taking YS = Y
and Z =
 
X⊤X + λIn
−1 X⊤. When m > n, we can generate more data than original dataset.
Compared with [9] that needs d data, our approach only requires k data and usually k ≤d in practice.
Our result is also more flexible to distill any m ≥k data points.

Figure 1: Distilled data of MNIST (first row) and CIFAR-100 (second row) for LRR when m = k.

Discussion. The requirement for λS can be easily satisfied by setting λS ≤
1
4σ′2
1 for a given D. If
we want to fix a predefined λS, e.g. λS = λ, we need to sample different D (by sampling different
Z) so that λS ≤
1
4σ′2
1 is satisfied. Theorem 4.1 generally suggests that a smaller λS is better for

4


---Page Break---
constructing distilled data. Practical algorithms, e.g. KIP, FRePo, and RFAD, usually use a very
small regularization, which may already satisfy the requirement.

In practice, we usually do not want the dataset to be singular. Below we show that it is easy to satisfy
as long as Z is full rank and the rows of W and Z are linearly independent.
Proposition 4.1. When m ≥k and YS, W are rank k, the XS in Theorem 4.1 is full rank for any
full-rank Z such that Range
 
W⊤
∩Range
 
Z⊤
= {0}.

4.2
Finding Realistic Distilled Data

In Theorem 4.1, any D satisfying the condition will guarantee the distilled dataset model to recover
the original model’s performance. For example, we can choose Z to be a random Gaussian matrix.
However, to make the distilled data more realistic and generalizable, we can select m real training
data ˆXS as initialization of distilled data and find the distilled data that is closest to ˆXS.

Corollary 4.1.1. Given fixed ˆXS, λS, and YS, the D that satisfies Theorem 4.1 and minimize
D −ˆX+
λS


F is

D = Y+
S W +
 
Im −Y+
S YS
 
ˆX+
λS −Y+
S W

,

where ˆX+
λS is defined analogous to X+
λS. Taking YS = W ˆXλS can further minimize the distance.

When λS = 0, YS = W ˆXS is the prediction of original model for ˆXS. Combining this with
Theorem 4.1, we summarize the computation of the distilled data in Algorithm 1 with ϕ(x) = x.
Figure 2 shows some distilled data that is close to the real data or generated with random noise.

(a) MNIST with IPC=50.

(b) CIFAR-100 with IPC=5.

Figure 2: Initialized data (first row), distilled data generated from real images using techniques in
Sec 4.2 (second row), and distilled data generated from random noise using techniques in Sec 4.1
(third row) for a LRR with m = 500 on MNIST and CIFAR-100. IPC: images per class.

4.3
Label Distillation

If we fix the XS and only distill labels, as also shown in [24], we need at least m = d to guarantee
WS = W because labels have fewer learnable parameters than data.
Theorem 4.2. For any fixed XS,

5


---Page Break---
1. when m < d, there is no YS can guarantee WS = W in general unless the rows of W are
in the row space of X+
λS. The least square solution is YS = WXλS and ∥WS −W∥=
W
 
XλSX+
λS −Id
.

2. when m ≥d, if XS is rank d, then YS = WXλS is sufficient for WS = W.

5
Dataset Distillation for Kernel Ridge Regression (KRR)

In the last section, we analyzed the dataset distillation for LRR. However, more complex models such
as KRR and neural networks (NNs) are usually used in practice for better performance. Therefore,
it is crucial to extend the analysis to these practical settings. In this section, we first extend the
results of LRR to KRR in the feature space and then construct distilled data from desired features by
considering two cases – subjective and non-surjective feature mappings.

The results in Theorem 4.1 of LRR can be directly extended to the KRR in the feature space by
replacing data XS with features ϕ(XS) in Theorem 4.1. For completeness, we state it below.

Theorem 5.1. When m < k, there is no ϕ(XS) can guarantee WS = W unless the columns of
W are in the range space of YS. When m ≥k and YS is rank k, let r = min(m, p) and take
D = Y+
S W +
 
Im −Y+
S YS

Z, where Z ∈Rm×p is any matrix of the same size as ϕ(XS)⊤.
Suppose the reduced SVD of D is D = Vdiag(σ′
1, . . . , σ′
r)U⊤with σ′
1 ≥· · · ≥σ′
r ≥0, the
following results hold:

1. λS > 0: WS = W if and only if, for any D defined above, λS ≤
1
4σ′2
1
and ϕ(XS) =

Udiag(σ1, . . . , σr)V⊤where σi =

( 0,
if σ′
i = 0,

1±√

1−4λSσ′2
i
2σ′
i
,
otherwise.

2. λS = 0: WS = W if and only if ϕ(XS) = D+ for any D defined above.

This shows that in the feature space, k features are necessary and sufficient to recover the original
model’s parameter. However, what we get is the feature of distilled data ϕ(XS) instead of distilled
data XS itself. To get XS, we need to construct the data from the features. To do this, we consider
two cases – surjective and non-surjective ϕ. For subjective ϕ, we show that we can directly construct
XS from ϕ(XS). For non-surjective ϕ such as neural networks (NNs), we show one data per class is
in general not sufficient, but k + 1 data points can be sufficient for deep linear neural networks.

5.1
Surjective Feature Mapping

When ϕ is surjective or bijective, we can always find a XS for a desired ϕ(XS). In this case, k
distilled data (one data per class) is sufficient to recover the original model’s performance, in contrast
to [21] that needs p distilled data. We summarize the computation of the distilled data in Algorithm 1.
Here we give some examples of surjective/bijective ϕ.

Example 5.1 (Invertible NN). If ϕ is invertible such as invertible NNs used in normalizing flows,
then we can directly compute x = ϕ−1(ϕ(x)).

Example 5.2 (Fully-connected NN (FCNN)). For a (L + 1)-layer FCNN f(x) = Wϕ(x) and

ϕ(x) = σ(W(L)σ(· · · W(2)σ(W(1)x))).

where σ is a surjective or bijective activation function such as LeakyReLU, and W(l) ∈Rdl×dl−1
with d = d0 ≥d1 ≥· · · ≥dL for l ∈[L]. If all W(l) are full rank, given ϕ(x), we can compute

x =

W(1)+
σ−1

· · ·

W(L)+
σ−1(ϕ(x))

,

where σ−1 is any right inverse of σ. When some W(l) are not full rank, we can still compute an
approximated solution.

Example 5.3 (Convolutional Neural Network (CNN)). CNN is known as a special type of FCNN. To
illustrate, we give an example of a convolution layer of 2 × 2 filter. Let w ∈R4 be a convolutional

6


---Page Break---
filter of size 2. Then the convolution operation with stride 1 can be represented as

ϕ(x) =





w1
w2
0
0
· · ·
0
w3
w4
0
· · ·
0
0
w1
w2
0
· · ·
0
0
w3
w4
· · ·
0
...
...
...
...
...
...
...
...
...
...
...
0
0
0
· · ·
w1
w2
0
· · ·
0
w3
w4



x

If the data is three-channel images, same operation can be done for each channel followed by a matrix
that sums over three channels. When the matrix equation has a solution, the data can be solved from
the feature.
Example 5.4 (Random Fourier Features (RFF) [27]). A shift-invariant kernel K(x, x′) = K(x −
x′) can be approximated by random Fourier features K(x, x′) ≈⟨ϕ(x), ϕ(x′)⟩. Here ϕ(x) =
q

2
p

cos(a⊤
1 x + b1), . . . , cos(a⊤
p x + bp)
⊤∈Rp where a1, . . . , ap ∈Rd are independent samples

from a distribution P (a) =
1
2π
R
e−ja⊤∆K(∆) d∆(Fourier transform of K(∆)) and b1, . . . , bp are
i.i.d. sampled from the uniform distribution on [0, 2π]. For example, Gaussian kernel K(x, x′) =

e−∥x−x′∥2
2
2σ
has P (a) = N(0, σ−2Id). Denote A = [a1, . . . , ap]⊤and b = [b1, . . . , bp]⊤, then we

have ϕ(x) =
q

2
p cos(Ax + b). Whenever p ≤d and A is rank p, given ϕ(x) we can solve x as,

x = A+

arccos
rp

2ϕ(x) −b

.

To ensure the computed
p p

2ϕ(XS) ∈[−1, 1], we can normalize it by its largest absolute value,
which is equivalently scaling D in Theorem 5.1 and does not affect the direction of WS.

[21] use random Fourier features to approximate shif-invariant kernels that may have an infinite-
dimensional feature space, and construct p distilled data for such RFF model, where p ∈Ω(√n log n)
in general cases. Their construction, however, only uses label distillation and the XS can be any
random data. Our analysis constructs XS explicitly and shows that whenever the dimension of RFF
p needs to approximate shif-invariant kernels is less than d, k distilled data suffice to recover the
performance of the original RFF model and approximate the original KRR with shif-invariant kernels.

5.2
Non-surjective Feature Mapping

When ϕ is injective or non-surjective, given a ϕ(XS), we may not find an exactly matched XS.
However, we can find an approximated distilled data ˆXS first and then adjust YS to ensure WS ≈W.
As in the label distillation for LRR case, m ≥p distilled labels can guarantee WS = W.

Below we show one data per class is in general not sufficient for non-surjective ϕ, but k + 1 can be
sufficient for deep linear NNs. Consider a deep NN, f(x) = Wϕ(x) with

ϕ(x) = σ(W(L)σ(· · · W(2)σ(W(1)x))).

where σ is an invertible activation function such as LeakyReLU, Sigmoid, and W(l) ∈Rdl×dl−1
with d = d0 < d1 = · · · = dL = p for l ∈[L]. ϕ(x) is an injective function in this definition.

Theorem 5.2. For a deep nonlinear NN defined above with fixed ϕ, assume W(2), . . . , W(L) are full
rank. Suppose λS = 0 and YS is rank k. When m = k, there is no distilled data XS that can guaran-
tee WS = W in general useless the columns of σ−1  
W(2)−1 · · ·
 
W(L)−1 σ−1  
(Y+
S W)+

are in the range space of W(1).

When m = k, only ϕ(XS) = (Y+
S W)+ that can guarantee WS = W. One data per class is not
sufficient in general as long as (Y+
S W)+ is not in the range space of ϕ. Although k data is not
sufficient, we show k + 1 data can be sufficient for deep linear neural networks, where σ(x) = x.

Theorem 5.3. For a deep linear NN defined above with fixed ϕ, assume W(2), . . . , W(L) are full rank.
Suppose λS = 0 and YS, W are rank k. Denote H =
hQL
l=1 W(l)  
W(1)+
(W+W −Ip)
i
∈

Rp×2p.

7


---Page Break---
1. When m = k, there is no distilled data XS that can guarantee WS = W in general useless the
columns of W+YS are in the range space of QL
l=1 W(l).

2. When m > k, If H is full rank and its right singular vectors VH ∈R2p×2p’s last p × p submatrix
is full rank, then there exists a XS such that WS = W.

Proof sketch. To find the distilled data theoretically, we need to guarantee 1) the feature ϕ(XS) need
to guarantee WS = W and 2) there are some distilled data XS corresponding to the feature. For the
first condition, Theorem 5.1 gives the sufficient and necessary condition of ϕ(XS). However, the
formulation involves a pseudoinverse of sum of matrices, which does not have a concise formulation
and therefore is hard to handle when solving XS. Instead, in Theorem C.2, we derive a sufficient
condition of ϕ(XS) without pseudoinverse. For the second condition, ϕ(XS) = ϕ∗for a given ϕ∗is
an overdetermined system of linear equations of XS. We find the formulation of ϕ(XS) such that the
overdetermined system has solutions. Then combining the two conditions together, we end up with
an equation that has multiple free variables. Combing the variables together and solving the equation
will give us the results. In the proof, we provide the algorithm to compute the distilled data when the
assumptions are satisfied.

6
Applications

In this section, we show that our theoretical results can be useful in some applications. We first show
the conditions in Theorem 5.1 can be necessary or sufficient conditions for KIP-type algorithms to
converge in Sec 6.1. We also show our distilled dataset for KRR can provably preserve the privacy of
the original dataset in Sec 6.2.

6.1
An Implication for KIP-type Algorithms

In KIP [25], FRePo [40], RFAD [17], and RCIG [18], a loss function as follows is optimized:

L(XS) = ∥WSϕ(X) −Y∥2
F =
YS (K(XS, XS) + λSIm)−1 K(XS, X) −Y

2

F .
(1)

Below we show our results can be sufficient conditions for the above loss function to converge to 0
even if the loss function is highly non-convex.
Theorem 6.1. Suppose ϕ(X) is full rank and W is computed with λ = 0. Then

1. when n ≤p, the ϕ(XS) that can guarantee WS = W in Theorem 5.1 is sufficient for L(XS) = 0.

2. when n ≥p, the ϕ(XS) that can guarantee WS = W in Theorem 5.1 is necessary for L(XS) =
0.

From this theorem, we see that KIP-type algorithms are enforcing WS = W to some extent. While
the KIP algorithm is computationally expensive, our solution for ϕ(XS) can be computed efficiently
and directly utilized in KIP-type algorithms for efficient optimization.

6.2
Privacy Preservation of Dataset Distillation

For our dataset distillation algorithm for KRR, we show that the original dataset cannot be recovered
from the distilled dataset, therefore provably preserving the privacy of the original dataset while
having the performance guarantees at the same time.
Proposition 6.1. Suppose n > k and Y is rank k. Given λS, ϕ, for a distilled dataset (XS, YS)
that can guarantee WS = W in Theorem 5.1, we can reconstruct W from ϕ(XS). However, given
W, there are infinitely many solutions for ϕ(X).

Since there are infinitely many solutions for ϕ(X), it is impossible to recover ϕ(X) without additional
information. As long as ϕ does not contain any information of X, then X will not be able to recover
from distilled dataset (XS, YS). Note in Sec. 4.2, we use additional information (real images as
reference points) to compute XS. Therefore XS resembles original images and we may recover
these original images from XS. However, if we generate the distilled data with random noise, the
distilled data will contain no additional information and protect the privacy of the original dataset. In

8


---Page Break---
summary, we can control whether to generate realistic data by using real data as reference points or
protect privacy by generating noisy distilled data as shown in Figure 2.

More formally, we prove that the distilled data can be differential private with respect to the original
dataset if we take Z to be random Gaussian with suitable variance.
Theorem 6.2. Under the same setting of Theorem 4.1, suppose that λ = 0, all data are
bounded ∥xi∥2 ≤B, and the smallest singular value of the original datasets is bounded from
below σmin(X) > σ0. Suppose YS is independent of X and unknown to the adversary. Let
[Y+
S ]i denote its i-th row.
Let ϵ, δ ∈(0, 1) and take the elements of Z ∼N(0, σ2) with

σ ≥maxi∈[m]
2√

ln(1.25/δ)B∥[Y+
S ]iY∥2
σ2
0ϵ∥[Im−Y+
S YS]i∥2
, then each row of XS is (ϵ, δ)-differential private with

respect to X.

7
Experiments

(I) Analytical Computation of Dataset Distillation. In Table 3, we verify our theory of dataset
distillation for LRR and KRR with subjective mapping. We compute the distilled dataset for different
models using Algorithm 1. The models are KRR with different feature mappings: 1) identity mapping
(linear model), 2) one-hidden-layer LeakyReLU neural network, and 3) Random Fourier Features
(RFF) of Gaussian kernel. The feature mappings are constructed such that the feature dimension
is equal to the data dimension, i.e. p = d. For NNs, we use random initialized ones and use the
activations as feature mappings. As increasing the depth of NNs does not improve the performance,
we only use one hidden layer. For simplicity, we set λS = 0 for all experiments. To choose the
original model’s regularization λ, we split the original training set into a training set and a validation
set, and choose the λ that performs best on the validation set. The results show that our analytically
computed distilled dataset can indeed recover the original models’ parameters and performance.
Some slight differences are caused by the numerical error in recovering the data from features and
computing the KRR solutions. As the purpose of this experiment is to verify if the distilled dataset
can recover the performance of a specific original model, we did not report the error bars.

Table 3: Verification of our theory. Test accuracy of original models and models trained on the
distilled dataset. IPC: images per class.

Dataset
IPC
Linear
FCNN
RFF

MNIST
Original model
86.41
93.89
93.82

1
86.41
93.89
93.82
10
86.41
93.89
93.82
50
86.41
93.85
93.82

CIFAR-10
Original model
39.48
47.86
42.84

1
39.48
47.87
42.84
10
39.48
47.84
42.87
50
39.48
47.81
42.73

CIFAR-100
Original model
14.37
21.42
18.71

1
14.37
21.41
18.70
10
14.37
21.52
18.69
50
14.37
21.49
18.57

(II) Comparison with KIP. In Table 4, we compare our algorithm with KIP in terms of performance
and efficiency under the setting of a subjective mapping. The test accuracy of models trained on
distilled datasets and averaged computational cost (GPU Seconds) are reported. The mean and
standard deviation of test accuracy are computed over four independent runs. As the experiment (I),
we use a randomly initialized one-hidden-layer LeakyReLU NN with p = d as the feature mapping.
For KIP, we implement their algorithm where we optimize a loss function (1) and use label distillation
at each training step. For our results, we compute the distilled dataset using Algorithm 1. Our
algorithm performs better than KIP on CIFAR-10 and CIFAR-100 while being significantly more
efficient. We did not report the result of KIP with IPC=50 on CIFAR-100 because the estimated
running time is more than 110 hours.

9


---Page Break---
This experiment mainly aims to show that our theoretical guarantee can be transferred to practice. As
the proposed Algorithm 1 is mainly for KRR with surjective mappings, we verified it and compared it
with baselines in this setting. We use a randomly initialized bijective NN in order to match previous
algorithms that use a randomly initialized NN. If a pre-trained NN is used, the accuracy can be
improved and may match the SOTA.

Table 4: Comparison between our algorithm and KIP.

Dataset
IPC
KIP [25]
Ours
Accuracy ↑
Cost ↓
(GPU Sec.)

Accuracy ↑
Cost ↓
(GPU Sec.)
Speedup
over KIP ↑

MNIST
1
93.44±0.17
159
93.72±0.14
16
9.9×
10
93.75±0.10
554
93.69±0.17
16
34.6×
50
93.72±0.11
3114
93.62±0.24
16
194.6×

CIFAR-10
1
45.83±0.29
225
47.85±0.10
21
10.7×
10
47.50±0.29
594
47.76±0.12
20
29.7×
50
47.48±0.20
3510
47.77±0.06
20
175.5×

CIFAR-100
1
20.08±0.20
616
21.58±0.15
20
30.8×
10
21.56±0.16
9323
21.59±0.15
20
466.1×
50
-
∼396000
21.58±0.13
25
∼15840.0×

(III) Privacy Protection. In this experiment, we show our algorithm can be used to protect the
privacy of the original dataset. Same as experiment (II), we use a one-hidden-layer LeakyReLU
neural network with p = d as the feature mapping and train a KRR model on the original dataset.
Then we distill the dataset using Algorithm 1 and generate the distilled data with random Gaussian
noise. As shown in Figure 3, the distilled data for MNIST are essentially random noise, which
protects the privacy of the original MNIST dataset. At the same time, the model trained on it can
recover the original model’s performance of 93.87% test accuracy.

Figure 3: Distilled images of MNIST generated from random noise for a two-layer neural network.
m = 10 (first row) and 100 (second row).

8
Conclusion and Future Works

In this paper, by focusing on dataset distillation for KRR, we show that one data point per class
is already necessary and sufficient to recover the original model’s performance in many settings.
For linear ridge regression and KRR with surjective feature mappings, we provide necessary and
sufficient conditions for the distilled dataset to recover the original model’s parameters. For KRR
with injective feature mappings of deep neural networks, we show that while one data point per class
is not sufficient in general, k + 1 data points can be sufficient for deep linear neural networks. Our
theoretical results facilitate the direct construction of analytical solutions for distilled datasets, leading
to a provable and efficient dataset distillation algorithm for KRR. Additionally, we have developed
applications for KIP-type algorithms and privacy protection.

Several future research directions are worth exploring. First, while the current analysis shows
that k data points are generally insufficient for non-surjective deep non-linear neural networks,
determining the minimum number of distilled data points required remains an open question worthy
of investigation. Second, this paper focuses on KRR with fixed feature mappings, which differs from
some empirical works that train all neural network parameters. Extending the analysis to learnable
feature mappings would bridge this gap and provide further insights.

10


---Page Break---
Acknowledgement

The authors thank the anonymous reviewers for valuable feedback on the manuscript. T.-W. Weng is
supported by National Science Foundation awards CCF-2107189, IIS-2313105, IIS-2430539, the
Hellman Fellowship, and Intel Rising Star Faculty Award.

References

[1] Nicholas Carlini, Vitaly Feldman, and Milad Nasr. No free lunch in" privacy for free: How does
dataset condensation help privacy". arXiv preprint arXiv:2209.14987, 2022.

[2] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu.
Dataset distillation by matching training trajectories. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages 4750–4759, 2022.

[3] Randall E Cline. Representations for the generalized inverse of sums of matrices. Journal of
the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis, 2(1):99–114,
1965.

[4] Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet-
1k with constant memory. In International Conference on Machine Learning, pages 6565–6590.
PMLR, 2023.

[5] Tian Dong, Bo Zhao, and Lingjuan Lyu. Privacy for free: How does dataset condensation help
privacy? In International Conference on Machine Learning, pages 5378–5396. PMLR, 2022.

[6] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Founda-
tions and Trends® in Theoretical Computer Science, 9(3–4):211–407, 2014.

[7] Yunzhen Feng, Shanmukha Ramakrishna Vedantam, and Julia Kempe. Embarrassingly simple
dataset distillation. In The Twelfth International Conference on Learning Representations, 2024.
URL https://openreview.net/forum?id=PLoWVP7Mjc.

[8] Ziyao Guo, Kai Wang, George Cazenavette, HUI LI, Kaipeng Zhang, and Yang You. Towards
lossless dataset distillation via difficulty-aligned trajectory matching. In The Twelfth Inter-
national Conference on Learning Representations, 2024. URL https://openreview.net/
forum?id=rTBL8OhdhH.

[9] Zachary Izzo and James Zou. A theoretical study of dataset distillation. In NeurIPS 2023
Workshop on Mathematics of Modern Machine Learning, 2023. URL https://openreview.
net/forum?id=dq5QGXGxoJ.

[10] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. Advances in neural information processing systems, 31,
2018.

[11] Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong,
Jung-Woo Ha, and Hyun Oh Song. Dataset condensation via efficient synthetic-data param-
eterization. In International Conference on Machine Learning, pages 11102–11118. PMLR,
2022.

[12] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.

[13] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning
applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[14] Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S Schoenholz, Jeffrey Pennington,
and Jascha Sohl-Dickstein. Deep neural networks as gaussian processes. arXiv preprint
arXiv:1711.00165, 2017.

[15] Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sungroh Yoon. Dataset
condensation with contrastive signals. In International Conference on Machine Learning, pages
12352–12364. PMLR, 2022.

11


---Page Break---
[16] Zhu Li, Jean-Francois Ton, Dino Oglic, and Dino Sejdinovic. Towards a unified analysis of
random fourier features. Journal of Machine Learning Research, 22(108):1–51, 2021.

[17] Noel Loo, Ramin Hasani, Alexander Amini, and Daniela Rus. Efficient dataset distillation
using random feature approximation. Advances in Neural Information Processing Systems, 35:
13877–13891, 2022.

[18] Noel Loo, Ramin Hasani, Mathias Lechner, and Daniela Rus. Dataset distillation with convexi-
fied implicit gradients. In International Conference on Machine Learning, pages 22649–22674.
PMLR, 2023.

[19] Alaa Maalouf, Ibrahim Jubran, and Dan Feldman. Fast and accurate least-mean-squares solvers.
Advances in Neural Information Processing Systems, 32, 2019.

[20] Alaa Maalouf, Ibrahim Jubran, and Dan Feldman. Fast and accurate least-mean-squares solvers
for high dimensional data. IEEE Transactions on Pattern Analysis and Machine Intelligence,
44(12):9977–9994, 2022.

[21] Alaa Maalouf, Murad Tukan, Noel Loo, Ramin Hasani, Mathias Lechner, and Daniela Rus. On
the size and approximation error of distilled sets. arXiv preprint arXiv:2305.14113, 2023.

[22] G. Marsaglia and G. P. H. Styan. When does rank(a+b)=rank(a)+rank(b)? Canadian Mathemat-
ical Bulletin, 15(3):451–452, 1972. doi: 10.4153/CMB-1972-082-8.

[23] Lingsheng Meng and Bing Zheng. The optimal perturbation bounds of the moore–penrose
inverse under the frobenius norm. Linear algebra and its applications, 432(4):956–963, 2010.

[24] Timothy Nguyen, Zhourong Chen, and Jaehoon Lee. Dataset meta-learning from kernel
ridge-regression. arXiv preprint arXiv:2011.00050, 2020.

[25] Timothy Nguyen, Roman Novak, Lechao Xiao, and Jaehoon Lee. Dataset distillation with
infinitely wide convolutional networks. Advances in Neural Information Processing Systems,
34:5186–5198, 2021.

[26] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
style, high-performance deep learning library. Advances in neural information processing
systems, 32, 2019.

[27] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances
in neural information processing systems, 20, 2007.

[28] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei.
ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision
(IJCV), 115(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y.

[29] Noveen Sachdeva and Julian McAuley. Data distillation: A survey. Transactions on Machine
Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=
lmXMXP74TO. Survey Certification.

[30] James R Schott. Matrix analysis for statistics. John Wiley & Sons, 2016.

[31] Felipe Petroski Such, Aditya Rawal, Joel Lehman, Kenneth Stanley, and Jeffrey Clune. Genera-
tive teaching networks: Accelerating neural architecture search by learning to generate synthetic
training data. In International Conference on Machine Learning, pages 9206–9216. PMLR,
2020.

[32] Murad Tukan, Alaa Maalouf, and Margarita Osadchy. Dataset distillation meets provable subset
selection. arXiv preprint arXiv:2307.08086, 2023.

[33] Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Shuo Yang, Shuo Wang, Guan Huang, Hakan
Bilen, Xinchao Wang, and Yang You. Cafe: Learning to condense dataset by aligning features.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 12196–12205, 2022.

12


---Page Break---
[34] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A Efros. Dataset distillation.
arXiv preprint arXiv:1811.10959, 2018.

[35] Enneng Yang, Li Shen, Zhenyi Wang, Tongliang Liu, and Guibing Guo. An efficient dataset
condensation plugin and its application to continual learning. Advances in Neural Information
Processing Systems, 36, 2023.

[36] Zeyuan Yin, Eric Xing, and Zhiqiang Shen. Squeeze, recover and relabel: Dataset condensation
at imagenet scale from a new perspective. Advances in Neural Information Processing Systems,
36, 2024.

[37] Bo Zhao and Hakan Bilen. Dataset condensation with differentiable siamese augmentation. In
International Conference on Machine Learning, pages 12674–12685. PMLR, 2021.

[38] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In Proceedings of
the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6514–6523, 2023.

[39] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching.
arXiv preprint arXiv:2006.05929, 2020.

[40] Yongchao Zhou, Ehsan Nezhadarya, and Jimmy Ba. Dataset distillation using neural feature
regression. Advances in Neural Information Processing Systems, 35:9813–9827, 2022.

13


---Page Break---
Appendices

Algorithm 1 Dataset distillation for kernel ridge regression

Input: Number of distilled data m, number of classes k, regularization λS, feature mapping ϕ,
original model’s parameter W, generate from original data or random noise
1: if Generate from original data then
2:
Sample m balanced initialized data ˆXS from original dataset. Initialize YS as corresponding
one-hot labels
3:
Compute ϕλS( ˆXS)

4:
if Rank

WϕλS( ˆXS)

= k then

5:
YS = WϕλS( ˆXS)
6:
end if
7:
Construct Z =
 
Im −Y+
S YS
 
ϕλS( ˆXS)+ −Y+
S W

.

8: else if Generate from random noise then
9:
Sample Z from random noise. Initialize YS as balanced one-hot labels
10: end if
11: Compute D = Y+
S W +
 
Im −Y+
S YS

Z
12: if λS > 0 then
13:
Compute the SVD of D, D = Vdiag(σ′
1, . . . , σ′
r)U⊤

14:
if λS >
1
4σ′2
1 then

15:
λS =
1
4σ′2
1
16:
end if

17:
Construct ϕ(XS) = Udiag(σ1, . . . , σr)V⊤where σi = 0 if σ′
i = 0 and σi =
1±√

1−4λSσ′2
i
2σ′
i
otherwise.
18: else if λS = 0 then
19:
Construct ϕ(XS) = D+

20: end if
21: Construct XS from ϕ(XS)
Output: XS, λS

A
Broader Impact

Our approach can be used to protect data privacy, which may have a positive societal impact. There
are no particular ethical concerns we are aware of.

B
Additional Experiment Details

All the experiments are implemented with PyTorch [26] and conducted on a single 24G A5000 GPU.

14


---Page Break---
C
Proofs for Linear Ridge Regularization

C.1
Analytical Computation for Linear Ridge Regression

Theorem 4.1. When m < k, there is no XS can guarantee WS = W unless the columns of
W are in the range space of YS. When m ≥k and YS is rank k, let r = min(m, d) and take
D = Y+
S W +
 
Im −Y+
S YS

Z, where Z ∈Rm×d is any matrix of the same size as X⊤
S . Suppose
the reduced SVD of D is D = Vdiag(σ′
1, . . . , σ′
r)U⊤with σ′
1 ≥· · · ≥σ′
r ≥0, the following results
hold:

1. λS > 0: WS = W if and only if, for any D defined above, λS ≤
1
4σ′2
1
and XS =

Udiag(σ1, . . . , σr)V⊤where σi =

( 0,
if σ′
i = 0,

1±√

1−4λSσ′2
i
2σ′
i
,
otherwise.

2. λS = 0: WS = W if and only if XS = D+ for any D defined above.

Proof. Recall WS = YSX+
λS where

X+
λS =
  
X⊤
S XS + λSIm
−1 X⊤
S = X⊤
S
 
XSX⊤
S + λSId
−1 ,
if λS > 0,
X+
S ,
if λS = 0.

Let WS = W,
YSX+
λS = W.

When m < k, this is an overdetermined system for X+
λS. There is no solution for X+
λS in general
therefore no solution for XS unless all the columns of W are in the range space of YS. When there
is a solution, we can solve it as following m ≥k cases.

In the following, we consider the m ≥k case. Since m ≥k and YS is rank k, the solutions of X+
λS
are given by
X+
λS = Y+
S W +
 
Im −Y+
S YS

Z
(2)

where Z ∈Rm×d is any matrix of the same size as X+
λS. When k = m, the solution is unique
X+
λS = Y−1
S W. However, there are solutions for X+
λS does not mean there are solutions for XS.
Next, we need to solve XS from X+
λS.

1.
When m ≤d.
Suppose the reduced SVD of XS is XS
= UΣV⊤, where Σ =
diag(σ1, . . . , σm) ∈Rm×m, V ∈Rm×m is a unitary matrix and U ∈Rd×m is the first m columns
of a unitary matrix. Then when λS > 0,

X+
λS =
 
X⊤
S XS + λSIm
−1 X⊤
S

=
 
VΣU⊤UΣV⊤+ λSIm
−1 VΣU⊤

=
 
VΣ2V⊤+ λSIm
−1 VΣU⊤

=
 
V
 
Σ2 + λSIm

V⊤−1 VΣU⊤

= V
 
Σ2 + λSIm
−1 V⊤VΣU⊤

= V
 
Σ2 + λSIm
−1 ΣU⊤

= Vdiag(
σ1
σ2
1 + λS
, . . . ,
σm
σ2m + λS
)U⊤.
(3)

Combining (2) and (3), we must have

Vdiag(
σ1
σ2
1 + λS
, . . . ,
σm
σ2m + λS
)U⊤= Y+
S W +
 
Im −Y+
S YS

Z.

Denote D = Y+
S W +
 
Im −Y+
S YS

Z. Given D, we can compute its reduced SVD D =
V′diag(σ′
1, . . . , σ′
m)U′⊤with σ′
1 ≥· · · ≥σ′
m. Note that SVD of a matrix is unique. Since

15


---Page Break---
D = V′diag(σ′
1, . . . , σ′
m)U′⊤= Vdiag(
σ1
σ2
1+λS , . . . ,
σm
σ2m+λS )U⊤, we must have V = V′, U = U′,
and
σ′
i =
σi
σ2
i + λS
That is
σ′
iσ2
i −σi + λSσ′
i = 0
When σ′
i = 0, we have σi = 0. When σ′
i ̸= 0 and λS ≤
1
4σ′2
1 ≤
1
4σ′2
i , it has solutions given by

σi = 1 ±
p

1 −4λSσ′2
i
2σ′
i
Take the above computed U, V, and Σ, we can construct XS. Above shows such XS is a necessary
condition for WS = W. To show the sufficiency, take such XS into WS.

WS = YSVdiag(
σ1
σ2
1 + λS
, . . . ,
σm
σ2m + λS
)U⊤

= YSVdiag(σ′
1, . . . , σ′
m)U⊤

= YSD

= YS
 
Y+
S W +
 
Im −Y+
S YS

Z


= W

which shows it is a sufficient condition.

When λS = 0, we have WS = YSX+
S . Let WS = YSX+
S = W. The solution for X+
S is

X+
S = Y+
S W +
 
Im −Y+
S YS

Z

where Z ∈Rm×d is any matrix of the same size as X⊤
S . Therefore

XS =
 
Y+
S W +
 
Im −Y+
S YS

Z
+ .

Similarly, this is a necessary condition for WS = W. To show the sufficiency, take such XS into
WS.

WS = YS
 
Y+
S W +
 
Im −Y+
S YS

Z


= W

which shows it is a sufficient condition.

2. When m > d. Suppose the reduced SVD of XS is XS = UΣV⊤, where Σ = diag(σ1, . . . , σd) ∈
Rd×d, U ∈Rd×d is a unitary matrix and V ∈Rm×d is the first d columns of a unitary matrix. Then
when λS > 0,

X+
λS = X⊤
S
 
XSX⊤
S + λSId
−1

= VΣU⊤ 
UΣV⊤VΣU⊤+ λSId
−1

= VΣU⊤ 
UΣ2U⊤+ λSId
−1

= VΣU⊤ 
U
 
Σ2 + λSId

U⊤−1

= VΣU⊤U
 
Σ2 + λSId
−1 U⊤

= VΣ
 
Σ2 + λSId
−1 U⊤

= Vdiag(
σ1
σ2
1 + λS
, . . . ,
σd
σ2
d + λS
)U⊤
(4)

Then we proceed similarly to the m ≤d case. Last, we can unify two cases by taking r = min(m, d).

Proposition 4.1. When m ≥k and YS, W are rank k, the XS in Theorem 4.1 is full rank for any
full-rank Z such that Range
 
W⊤
∩Range
 
Z⊤
= {0}.

16


---Page Break---
Proof. XS is computed from D = Y+
S W +
 
Im −Y+
S YS

Z. It is easy to check that XS is full
rank if and only if D is full rank.

When YS and W are rank k, Rank(Y+
S W) = k. Since Rank(Im −Y+
S YS) = m −k and Z is full
rank, by Sylvester’s rank inequality,

Rank(
 
Im −Y+
S YS

Z) ≥Rank(Im −Y+
S YS) + Rank(Z) −m

= m −k + min(m, d) −m
= min(m, d) −k

For D = Y+
S W +
 
Im −Y+
S YS

Z, since the columns of Y+
S W ∈Range
 
Y⊤
S

and the
columns of
 
Im −Y+
S YS

Z ∈Null (YS).
By the fundamental theorem of linear algebra,
Range
 
Y⊤
S

and Null (YS) are orthogonal subspaces of Rm.
Therefore Range
 
Y+
S W

∩

Range
  
Im −Y+
S YS

Z

= {0}. This can also be seen from
 
Y+
S W
⊤ 
Im −Y+
S YS

Z = 0,
which shows their columns are orthogonal to each other. If we have Range
 
W⊤
∩Range
 
Z⊤
=

{0}, then Range
 
Y+
S W
⊤
∩Range
 
Z⊤ 
Im −Y+
S YS

= {0}. By [22],

Rank(D) = Rank(Y+
S W) + Rank(
 
Im −Y+
S YS

Z) ≥k + min(m, d) −k = min(m, d).

Therefore D is full rank and XS is full rank.

C.2
Characterization of Distilled Data without Pseudoinverse

In the last section, we give analytical solutions for XS that can guarantee WS = W. However, the
expression of XS involves some pseudoinverse calculation and the explicit expression of XS remains
unclear because there is no concise formulation for the pseudoinverse of sum of matrices. In this
section, we give some direct characterization for XS.

Again, supposed its reduced SVD is XS = Udiag(σ1, . . . , σr)V⊤, where r = min(m, d). When
λS > 0, from Eq. (3) and Eq. (4), X+
λS = D = Vdiag(
σ1
σ2
1+λS , . . . ,
σr
σ2r+λS )U⊤and XλS =

Udiag(
σ1
σ2
1+λS , . . . ,
σr
σ2r+λS )+V⊤, where
h
diag(
σ1
σ2
1+λS , . . . ,
σr
σ2r+λS )+i

i,i = σi + λS

σi if σi > 0 else

0. When λS = 0, X+
λS = X+
S and XλS = XS. Given XλS and λS, we can easily compute XS by
SVD. Below we give conditions for WS = W through XλS.

Suppose the eigenvalues of XλS are σ′
i. Then by the definition of XλS, σi + λS

σi = σ′
i if σi > 0. That
is
σ2
i −σ′
iσi + λS = 0

Only when σ′2
i ≥4λS, there are solution(s) σi =
σ′
i±√

σ′2
i −4λS
2
. Therefore, to make sure there is a
XS corresponds to XλS, the nonzero singular values of XλS need to be larger than or equal to 2√λS.
When λS = 0, there is no requirement.

Theorem C.1. Suppose k ≤d and W is rank k. Take

XλS = W+YS +
 
Id −W+W

Z′,

where Z′ ∈Rd×m is any matrix of the same size as XλS such that XλS is full rank.

1. When m ≤d, it is a necessary condition for WS = W.

2. When m ≥d, it is a sufficient condition for WS = W.

Proof. Case 1. When m ≤d, recall that WS = YSX+
λS. Set the parameter to be the same
WS = W and try to solve XλS
WS = YSX+
λS = W

Multiply XλS on both sides. Since XλS is full rank and X+
λSXλS = Im, we have

WXλS = YS

17


---Page Break---
Since k ≤d and W is rank k, XS has infinite many solutions. The general solutions are

XλS = W+YS +
 
Id −W+W

Z′
(5)

where Z′ ∈Rd×m is any matrix of the same size as XS. Therefore, this is a necessary condition for
WS = W.

Case 2. When m ≥d, to show the sufficiency, for any XλS = W+YS + (Id −W+W) Z′,

WXλS = W

W+YS +
 
Id −W+W

Z′
= YS.

Multiply X+
λS on both sides. Since XλSX+
λS = Id,

W = YSX+
λS = WS.

From these two cases, we can also conclude that when m = d, such XλS
= W+YS +
(Id −W+W) Z′ is a sufficient and necessary condition for WS = W.

Below we give a sufficient condition of XλS when m ≥k. It will be used in the proof of Theorem 5.3.

Theorem C.2. When m ≥k, a sufficient condition for WS = W is YS is rank k and

XλS = W+YS +
 
Id −W+W

Z′  
Im −Y+
S YS

,

where Z′ ∈Rd×m is any matrix of the same size as XλS.

Proof. When
m
≥
k,
for
the
sufficient
condition
XλS
=
W+YS
+
(Id −W+W) Z′  
Im −Y+
S YS

,
denote
A
=
W+YS
and
B
=
(Id −W+W) Z′  
Im −Y+
S YS

. Since A⊤B = 0 and AB⊤= 0, by [3], (A + B)+ = A+ +B+.
Therefore

X+
λS =
 
W+YS
+ +
 
Id −W+W

Z′  
Im −Y+
S YS
+

= Y+
S W +
 
Id −W+W

Z′  
Im −Y+
S YS
+

where the last equality is because W and YS are full rank and therefore (W+YS)+ = Y+
S W.
From this, we have

WS = YSX+
λS

= YS

Y+
S W +
 
Id −W+W

Z′  
Im −Y+
S YS
+

= W + YS
 
Id −W+W

Z′  
Im −Y+
S YS
+

For
any
matrix
A
and
B,
if
AB
=
0
then
B+A+
=
0
[30].
Since
(Id −W+W) Z′  
Im −Y+
S YS

Y+
S = 0, YS

(Id −W+W) Z′  
Im −Y+
S YS
+ = 0. There-
fore we conclude WS = W. Note in this case, we do not require XλS to be full rank.

C.3
Finding Realistic Distilled Data

Corollary 4.1.1. Given fixed ˆXS, λS, and YS, the D that satisfies Theorem 4.1 and minimize
D −ˆX+
λS


F is

D = Y+
S W +
 
Im −Y+
S YS
 
ˆX+
λS −Y+
S W

,

where ˆX+
λS is defined analogous to X+
λS. Taking YS = W ˆXλS can further minimize the distance.

18


---Page Break---
Proof. Given fixed ˆXS, λS, and YS, the linear ridge regression trained on ˆXS is

WS = YS ˆX+
λS ∈Rm×d

By Theorem 4.1, to ensure WS = W, we need ˆX+
λS to be equal to some D = Y+
S W +
 
Im −Y+
S YS

Z, where Z is a free variable to be determined. Therefore let

Y+
S W +
 
Im −Y+
S YS

Z = ˆX+
λS
That is
 
Im −Y+
S YS

Z = ˆX+
λS −Y+
S W

Since
 
Im −Y+
S YS

is idempotent and therefore singular, Z does not have a solution in general
(because the system of equations can be inconsistent). The least-squares solution is

Z =
 
Im −Y+
S YS
+ 
ˆX+
λS −Y+
S W

=
 
Im −Y+
S YS
 
ˆX+
λS −Y+
S W


where one can verify that
 
Im −Y+
S YS
+ = Im −Y+
S YS by SVD. This least-squares solution

minimize

 
Im −Y+
S YS

Z −ˆX+
λS + Y+
S W

F =
D −ˆX+
λS


F . Take such Z into D, we have

D = Y+
S W +
 
Im −Y+
S YS
  
Im −Y+
S YS
 
ˆX+
λS −Y+
S W


= Y+
S W +
 
Im −Y+
S YS
 
ˆX+
λS −Y+
S W


Then we have the difference between D and ˆX+
λS is

ˆX+
λS −D = ˆX+
λS −Y+
S W −
 
Im −Y+
S YS
 
ˆX+
λS −Y+
S W


= Y+
S YS

ˆX+
λS −Y+
S W


= Y+
S

YS ˆX+
λS −W


To further minimize the difference, we can let YS ˆX+
λS = W. The least square solution is YS =
W ˆXλS.

C.4
Label Distillation

Theorem 4.2. For any fixed XS,

1. when m < d, there is no YS can guarantee WS = W in general unless the rows of W are
in the row space of X+
λS. The least square solution is YS = WXλS and ∥WS −W∥=
W
 
XλSX+
λS −Id
.

2. when m ≥d, if XS is rank d, then YS = WXλS is sufficient for WS = W.

Proof. When m < d, let

WS = YSX+
λS = W

and solve YS. YS does not have a solution in general unless the equations are consistent, i.e. the
rows of W are in the row space of X+
λS. The least-squares solution is

YS = WXλS
Therefore we have

WS = YSXλS = WXλSX+
λS
Then we can bound the difference between WS and W,

∥WS −W∥=
W
 
XλSX+
λS −Id
 ≤∥W∥
 
XλSX+
λS −Id
 .

19


---Page Break---
When m ≥d, let

WS = YSX+
λS = W

and solve YS. Since XS is rank d, then X+
λS = X⊤
S
 
XSX⊤
S + λSId
−1 is rank d and YS has
solutions. Take the minimum norm one,

YS = WXλS

To show the sufficiency, take YS into WS.

WS = WX+
λSXλS = W

20


---Page Break---
D
Proofs for Kernel Ridge Regression

Theorem 5.1. When m < k, there is no ϕ(XS) can guarantee WS = W unless the columns of
W are in the range space of YS. When m ≥k and YS is rank k, let r = min(m, p) and take
D = Y+
S W +
 
Im −Y+
S YS

Z, where Z ∈Rm×p is any matrix of the same size as ϕ(XS)⊤.
Suppose the reduced SVD of D is D = Vdiag(σ′
1, . . . , σ′
r)U⊤with σ′
1 ≥· · · ≥σ′
r ≥0, the
following results hold:

1. λS > 0: WS = W if and only if, for any D defined above, λS ≤
1
4σ′2
1
and ϕ(XS) =

Udiag(σ1, . . . , σr)V⊤where σi =

( 0,
if σ′
i = 0,

1±√

1−4λSσ′2
i
2σ′
i
,
otherwise.

2. λS = 0: WS = W if and only if ϕ(XS) = D+ for any D defined above.

Proof. The proof is same as Theorem 4.1 but just replace XS with ϕ(XS).

D.1
Deep Nonlinear Neural Networks

Theorem 5.2. For a deep nonlinear NN defined above with fixed ϕ, assume W(2), . . . , W(L) are full
rank. Suppose λS = 0 and YS is rank k. When m = k, there is no distilled data XS that can guaran-
tee WS = W in general useless the columns of σ−1  
W(2)−1 · · ·
 
W(L)−1 σ−1  
(Y+
S W)+

are in the range space of W(1).

Proof. To get a distilled data XS that can guarantee WS = W, the sufficient and necessary condition
is that

1. ϕ(XS) need guarantee WS = W.

2. There is some XS corresponds to such ϕ(XS). Equivalently XS is recoverable from ϕ(XS).

1. For the first condition, when λS = 0, we have shown in Theorem 5.1, ϕ(XS) has to be

ϕ(XS) =
 
Y+
S W +
 
Im −Y+
S YS

Z
+
(6)

When m = k and YS is rank k, this reduce to ϕ(XS) = (Y+
S W)+.

2. For the second condition, given ϕ(XS) = σ
 
W(L) · · · σ
 
W(1)XS

, solving XS is same as
solving

W(1)XS = σ−1

W(2)−1
· · ·

W(L)−1
σ−1 (ϕ(XS))


When m = k and combined with the first condition, it becomes

W(1)XS = σ−1

W(2)−1
· · ·

W(L)−1
σ−1  
(Y+
S W)+

Since this is an over-determined system of linear equations and RHS is fixed, It does not have a
solution in general unless The RHS is in the range space of W(1).

D.2
Deep Linear Neural Networks

Theorem 5.3. For a deep linear NN defined above with fixed ϕ, assume W(2), . . . , W(L) are full rank.
Suppose λS = 0 and YS, W are rank k. Denote H =
hQL
l=1 W(l)  
W(1)+
(W+W −Ip)
i
∈

Rp×2p.

1. When m = k, there is no distilled data XS that can guarantee WS = W in general useless the
columns of W+YS are in the range space of QL
l=1 W(l).

21


---Page Break---
2. When m > k, If H is full rank and its right singular vectors VH ∈R2p×2p’s last p × p submatrix
is full rank, then there exists a XS such that WS = W.

Proof. To get a distilled data XS that can guarantee WS = W, the sufficient and necessary condition
is that

1. ϕ(XS) need guarantee WS = W.

2. There is some XS corresponds to such ϕ(XS). Equivalently XS is solvable from ϕ(XS).

1. For the first condition, when λS = 0, we have shown in Theorem 5.1, ϕ(XS) has to be

ϕ(XS) =
 
Y+
S W +
 
Im −Y+
S YS

Z
+
(7)

When m = k and YS, W are rank k, this reduce to ϕ(XS) = W+YS. When m ≥k, from
Theorem C.2, we know ϕ(XS) = W+YS + (Ip −W+W) Z′  
Im −Y+
S YS

for any Z′ ∈Rp×m
is also a sufficient condition for WS = W.

2. For the second condition, given ϕ(XS) = QL
l=1 W(l)XS, solving XS is same as solving

W(1)XS =

 L
Y

l=2
W(l)
!−1

ϕ(XS)

This is an over-determined system of linear equations.
A necessary and sufficient condition
for any solution(s) to exist is that RHS is in the range space of W(1) or equivalently XS =
 
W(1)+ QL
l=2 W(l)−1
ϕ(XS) is a solution. Take this solution into equation,

W(1) 
W(1)+
 L
Y

l=2
W(l)
!−1

ϕ(XS) =

 L
Y

l=2
W(l)
!−1

ϕ(XS)


Ip −W(1) 
W(1)+  L
Y

l=2
W(l)
!−1

ϕ(XS) = 0

Solve the equation for
QL
l=2 W(l)−1
ϕ(XS), we have

 L
Y

l=2
W(l)
!−1

ϕ(XS) =

 

Ip −

Ip −W(1) 
W(1)++ 
Ip −W(1) 
W(1)+!

Z1

=

Ip −

Ip −W(1) 
W(1)+ 
Ip −W(1) 
W(1)+
Z1

=

Ip −

Ip −W(1) 
W(1)+
Z1

= W(1) 
W(1)+
Z1

for any Z1 ∈Rp×m. Therefore to guarantee XS is solvable from ϕ(XS), ϕ(XS) have to be in the
form of

ϕ(XS) =

L
Y

l=1
W(l) 
W(1)+
Z1
(8)

In this case, XS =
 
W(1)+ Z1. For any Z1, W(1)  
W(1)+ is a projector that projects Z1 to the
range space of W(1).

Combing two conditions (7) and (8), we need to solve

L
Y

l=1
W(l) 
W(1)+
Z1 =
 
Y+
S W +
 
Im −Y+
S YS

Z
+

for Z and Z1.

22


---Page Break---
1. When m = k, it becomes
L
Y

l=1
W(l)XS = W+YS

Since QL
l=1 W(l) ∈Rp×d and p > d, the equation has a solution only when W+YS is in the
range space of QL
l=1 W(l).

2. When
m
>
k,
from
Theorem
C.2,
we
know
ϕ(XS)
=
W+YS
+
(Ip −W+W) Z′  
Im −Y+
S YS

for any Z′ ∈Rp×m is a sufficient condition for WS = W.
Therefore we can instead solve

L
Y

l=1
W(l) 
W(1)+
Z1 = W+YS +
 
Ip −W+W

Z′  
Im −Y+
S YS


Combine the variables Z′ and Z1, we have
hQL
l=1 W(l)  
W(1)+
(W+W −Ip)
i 
Z1
Z′  
Im −Y+
S YS


= W+YS

Denote H =
hQL
l=1 W(l)  
W(1)+
(W+W −Ip)
i
∈Rp×2p. If H is full rank (rank p), then
the solutions are

Z1
Z′  
Im −Y+
S YS


= H+W+YS +
 
I2p −H+H

Z2

where Z2 ∈R2p×m is any matrix. For any RHS, we can find a solution for Z1. Next, we
try to find a solution for Z′ and Z2 such that the equation is consistent. Suppose the full
SVD of H = UΣV⊤, where U ∈Rp×p, Σ ∈Rp×2p, V ∈R2p×2p. Then I2p −H+H =
Vdiag(0, . . . , 0
| {z }
p
, 1, . . . , 1
| {z }
p
)V⊤= [0
. . .
0
Vp+1
. . .
Vp] V⊤.
Then the equation be-

comes

[0
. . .
0
Vp+1
. . .
V2p] V⊤Z2 =

Z1
Z′  
Im −Y+
S YS


−H+W+YS

Denote B = V⊤Z2 =





b⊤
1...
b⊤
2p



∈R2p×m. Since Z2 is solvable from any B, we can instead solve

B. Then the equation is

[Vp+1
. . .
V2p]





b⊤
p+1
...
b⊤
2p



=

Z1
Z′  
Im −Y+
S YS


−H+W+YS

Denote H+W+YS =

C1
C2


where C1, C2 ∈Rp×m are the first and last p rows. Then the

equation can be partitioned into two parts

[Vp+1
. . .
V2p]1:p





b⊤
p+1
...
b⊤
2p



= Z1 −C1
(9)

[Vp+1
. . .
V2p]p+1:2p





b⊤
p+1
...
b⊤
2p



= Z′  
Im −Y+
S YS

−C2
(10)

where [Vp+1
. . .
V2p]1:p denotes its first p rows and [Vp+1
. . .
V2p]p+1:2p denotes its
last p rows. For the first equation, there is always a solution for Z1. So we will mainly

23


---Page Break---
care about if there is a solution for the second equation.
For the second equation, when

[Vp+1
. . .
V2p]p+1:2p is full rank, then there is always a solution for





b⊤
p+1
...
b⊤
2p



and therefore

solutions for B and Z2. The equations do not depend on b1, . . . , bp so they can be anything.

In conclusion, when [Vp+1
. . .
V2p]p+1:2p is full rank, there is a solution for Z1 and XS. To
construct XS, take any Z′ and ϕ(XS) = W+YS + (Ip −W+W) Z′  
Im −Y+
S YS

. Then

construct H =
hQL
l=1 W(l)  
W(1)+
(W+W −Ip)
i
∈Rp×2p and its SVD H = UΣV⊤. If

[Vp+1
. . .
V2p]p+1:2p is full rank, solve (10),




b⊤
p+1
...
b⊤
2p



= [Vp+1
. . .
V2p]−1
p+1:2p

Z′  
Im −Y+
S YS

−C2


Then we get Z1 from (9)

Z1 = [Vp+1
. . .
V2p]1:p





b⊤
p+1
...
b⊤
2p



+ C1

Then we can construct XS =
 
W(1)+ Z1.

D.3
Additional Trainable Layer

Here we consider whether adding an additional trainable layer to the distilled dataset model will
help dataset distillation. Suppose original model is f(x) = Wϕ(x) and distilled dataset model is
fS(x) = WSAϕ(x) where A ∈Rp×p is the additional trainable layer. In this case, the feature of
distilled dataset model is Aϕ(x) instead of ϕ(x) and the analytical solution for WS becomes WS =

YS

(Aϕ(XS))⊤Aϕ(XS) + λSIm
−1
(Aϕ(XS))⊤. Here the objective becomes WSA = W.

Theorem D.1. When k ≤m < p and λS > 0, suppose YS is rank k, there exists a distilled dataset
(XS, YS) can guarantee WSA = W if below equation has a solution for some Z ∈Rm×p such
that ϕ(XS) is full rank and some c > 0:
c
c + 1ϕ(XS)+ = Y+
S W +
 
Im −Y+
S YS

Z.

Proof. What we want now is WSA = W. That is

YS

(Aϕ(XS))⊤Aϕ(XS) + λSIm
−1
(Aϕ(XS))⊤A = W

For a given YS, since k ≤m, the solution of LHS is
 
ϕ(XS)⊤A⊤Aϕ(XS) + λSIm
−1 ϕ(XS)⊤A⊤A = Y+
S W +
 
Im −Y+
S YS

Z

for any Z ∈Rm×p. Denote RHS as D = Y+
S W +
 
Im −Y+
S YS

Z and multiply the inverse on
both sides,
ϕ(XS)⊤A⊤A =
 
ϕ(XS)⊤A⊤Aϕ(XS) + λSIm

D

Arrange the terms,
ϕ(XS)⊤A⊤A (Ip −ϕ(XS)D) = λSD

If there exists a D such that Ip −ϕ(XS)D is rank p,

ϕ(XS)⊤A⊤A = λSD (Ip −ϕ(XS)D)−1

24


---Page Break---
Here ϕ(XS)⊤∈Rm×p. Since m < p, there are solutions for A⊤A. Take the minimum norm one:

A⊤A = λS
 
ϕ(XS)⊤+ D (Ip −ϕ(XS)D)−1

Since A⊤A is symmetric and positive semidefinite, the RHS also needs to be positive semidefinite. A
sufficient condition is that D (Ip −ϕ(XS)D)−1 = cϕ(XS)+ for some constant c > 0. Solve it we
get D =
c
c+1ϕ(XS)+. In this case, Ip −ϕ(XS)D = Ip −
c
c+1ϕ(XS)ϕ(XS)+ is indeed full rank.

By the definition of D and D =
c
c+1ϕ(XS)+, the problem boils down to

c
c + 1ϕ(XS)+ = Y+
S W +
 
Im −Y+
S YS

Z

Compared with Eq. (7), adding one additional trainable layer only relaxes the original equation with
constant scaling and does not help too much.

25


---Page Break---
E
Applications

E.1
An Implication for KIP-type Algorithms

Theorem 6.1. Suppose ϕ(X) is full rank and W is computed with λ = 0. Then

1. when n ≤p, the ϕ(XS) that can guarantee WS = W in Theorem 5.1 is sufficient for L(XS) = 0.

2. when n ≥p, the ϕ(XS) that can guarantee WS = W in Theorem 5.1 is necessary for L(XS) =
0.

Proof. The ϕ(XS) in Theorem 5.1 guarantees WS = W. Since λ = 0 and ϕ(X) is full rank,
W = Yϕ(X)+. Therefore we have
WS = Yϕ(X)+

When n ≤p, multiply ϕ(X) on both sides,

WSϕ(X) = Y

which implies L(XS) = ∥WSϕ(X) −Y∥2 = 0. This shows that ϕ(XS) in Theorem 5.1 is sufficient
for L(XS) = 0.

When n ≥p, the L(XS) = 0 means
WSϕ(X) = Y

Multiply ϕ(X)+ on both sides, we have

WS = Yϕ(X)+ = W.

This implies that WS = W is a necessary condition for L(XS) = 0. Since the ϕ(XS) in Theorem 5.1
is sufficient and necessary for WS = W. Therefore ϕ(XS) in Theorem 5.1 is necessary for
L(XS) = 0.

E.2
Privacy Preservation of Dataset Distillation

Proposition 6.1. Suppose n > k and Y is rank k. Given λS, ϕ, for a distilled dataset (XS, YS)
that can guarantee WS = W in Theorem 5.1, we can reconstruct W from ϕ(XS). However, given
W, there are infinitely many solutions for ϕ(X).

Proof. Since W = WS, we can reconstruct W by simply compute WS.

When n > k, since W = Yϕλ(X)+ and Y is rank k, given W, there are infinitely many solutions
for ϕλ(X)+,
ϕλ(X)+ = Y+W +
 
In −Y+Y

Z

for any Z ∈Rn×p. Using a similar approach as the proof of Theorem 4.1, we can solve ϕ(X) by
SVD and there are infinitely many solutions for ϕ(X). Therefore it is impossible to recover ϕ(X)
without additional information.

Theorem 6.2. Under the same setting of Theorem 4.1, suppose that λ = 0, all data are
bounded ∥xi∥2 ≤B, and the smallest singular value of the original datasets is bounded from
below σmin(X) > σ0. Suppose YS is independent of X and unknown to the adversary. Let
[Y+
S ]i denote its i-th row.
Let ϵ, δ ∈(0, 1) and take the elements of Z ∼N(0, σ2) with

σ ≥maxi∈[m]
2√

ln(1.25/δ)B∥[Y+
S ]iY∥2
σ2
0ϵ∥[Im−Y+
S YS]i∥2
, then each row of XS is (ϵ, δ)-differential private with

respect to X.

Proof. We prove that D = Y+
S W +
 
Im −Y+
S YS

Z is (ϵ, δ)-differential private with respect to
X using the Gaussian mechanism [6]. Then it follows that XS is (ϵ, δ)-differential private since the
computation from D to XS is deterministic and independent with X.

26


---Page Break---
We first show the sensitivity of Y+
S W is bounded and then use
 
Im −Y+
S YS

Z as random Gaussian
to apply the Gaussian mechanism. Without loss of generality, suppose we have two datasets X =
[x1, x2, . . . , xn] , X′ = [x′
1, x2, . . . , xn] ∈Rd×n that differ only in the first point and their resulting
parameters are W and W′. Since λ = 0, W = YX+. For each row of Y+
S W, we have
[Y+
S ]iW −[Y+
S ]iW′
2 =
[Y+
S ]iY
 
X+ −X′+
2
≤
[Y+
S ]iY

2
X+ −X′+
2
≤
[Y+
S ]iY

2
X+ −X′+
F
≤
[Y+
S ]iY

2
X+
2
X′+
2 ∥x1 −x′
1∥F

≤
[Y+
S ]iY

2
2B

σ2
0
where the third inequality is due to Theorem 2.2 in Meng and Zheng [23]. Therefore the sensitivity
of [Y+
S ]iW is bounded. Suppose the elements of Z ∼N(0, σ2), the elements of
 
Im −Y+
S YS

Z
are also Gaussian. The i-th row of
 
Im −Y+
S YS

Z is

Im −Y+
S YS


i Z

whose elements are independent Gaussian with the variance

Im −Y+
S YS


i
2

2 σ2. Therefore, for
each row [Y+
S ]iW +

Im −Y+
S YS


i Z, we can apply the Gaussian mechanism. Let ϵ ∈(0, 1), by
Theorem 3.22 in [6], as long as


Im −Y+
S YS


i

2 σ ≥
p

ln(1.25/δ)
[Y+
S ]iY

2
2B
σ2
0ϵ

⇔σ ≥2
p

ln(1.25/δ)B
[Y+
S ]iY

2
σ2
0ϵ

Im −Y+
S YS


i

2

[Y+
S ]iW +

Im −Y+
S YS


i Z is (ϵ, δ)-differential private. Take σ to be the maximum one so that
all the rows are (ϵ, δ)-differential private.

σ ≥max
i∈[m]
2
p

ln(1.25/δ)B
[Y+
S ]iY

2
σ2
0ϵ

Im −Y+
S YS


i

2
.

27


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope?

Answer: [Yes]

Justification: The claims made match the theoretical and experimental results

Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the
paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions
made in the paper and important assumptions and limitations. A No or NA answer to this
question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the
results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not
attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See for example the last paragraph in Section 8.

Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the
paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of
these assumptions (e.g., independence assumptions, noiseless settings, model well-specification,
asymptotic approximations only holding locally). The authors should reflect on how these
assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested
on a few datasets or with a few runs. In general, empirical results often depend on implicit
assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For
example, a facial recognition algorithm may perform poorly when image resolution is low or
images are taken in low lighting. Or a speech-to-text system might not be used reliably to
provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how
they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers
as grounds for rejection, a worse outcome might be that reviewers discover limitations that
aren’t acknowledged in the paper. The authors should use their best judgment and recognize
that individual actions in favor of transparency play an important role in developing norms that
preserve the integrity of the community. Reviewers will be specifically instructed to not penalize
honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?

Answer: [Yes]

Justification: The assumptions are stated in the Theorems and all the proofs are in the Appendix.

Guidelines:

28


---Page Break---
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear
in the supplemental material, the authors are encouraged to provide a short proof sketch to
provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by
formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experi-
mental results of the paper to the extent that it affects the main claims and/or conclusions of the
paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The experiment details are provided in Section 7.

Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the
reviewers: Making the paper reproducible is important, regardless of whether the code and data
are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make
their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For
example, if the contribution is a novel architecture, describing the architecture fully might
suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary
to either make it possible for others to replicate the model with the same dataset, or provide
access to the model. In general. releasing code and data is often one good way to accomplish
this, but reproducibility can also be provided via detailed instructions for how to replicate the
results, access to a hosted model (e.g., in the case of a large language model), releasing of a
model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions
to provide some reasonable avenue for reproducibility, which may depend on the nature of the
contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to
reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe the
architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should either
be a way to access this model for reproducing the results or a way to reproduce the model
(e.g., with an open-source dataset or instructions for how to construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are
welcome to describe the particular way they provide for reproducibility. In the case of
closed-source models, it may be that access to the model is limited in some way (e.g.,
to registered users), but it should be possible for other researchers to have some path to
reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to
faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: The experiment details are provided in the paper. We will release the code once the
paper is published.

Guidelines:
• The answer NA means that paper does not include experiments requiring code.

29


---Page Break---
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible,
so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless
this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce
the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access
the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed
method and baselines. If only a subset of experiments are reproducible, they should state which
ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if
applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is
recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters,
how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The experiment details are provided in Section 7.

Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is
necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: See for example Table 4.

Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the main claims
of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example,
train/test split, initialization, random drawing of some parameter, or overall run with given
experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a
library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the
mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably
report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of
errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures
symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were
calculated and reference the corresponding figures or tables in the text.

30


---Page Break---
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the experi-
ments?

Answer: [Yes]

Justification: See Section 7 and Appendix B.

Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud
provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental
runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the
experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it
into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: There are no particular ethical concerns we are aware of.

Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation
from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due
to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?

Answer: [Yes]

Justification: Our approach can be used to protect data privacy, which may have a positive societal
impact.

Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or
why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deploy-
ment of technologies that could make decisions that unfairly impact specific groups), privacy
considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to par-
ticular applications, let alone deployments. However, if there is a direct path to any negative
applications, the authors should point it out. For example, it is legitimate to point out that
an improvement in the quality of generative models could be used to generate deepfakes for
disinformation. On the other hand, it is not needed to point out that a generic algorithm for
optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used
as intended and functioning correctly, harms that could arise when the technology is being used
as intended but gives incorrect results, and harms following from (intentional or unintentional)
misuse of the technology.

31


---Page Break---
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies
(e.g., gated release of models, providing defenses in addition to attacks, mechanisms for
monitoring misuse, mechanisms to monitor how a system learns from feedback over time,
improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of
data or models that have a high risk for misuse (e.g., pretrained language models, image generators,
or scraped datasets)?

Answer: [NA]

Justification: This paper will not release models and datasets that have a high risk for misuse.

Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary
safeguards to allow for controlled use of the model, for example by requiring that users adhere
to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should
describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require
this, but we encourage authors to take this into account and make a best faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the
paper, properly credited and are the license and terms of use explicitly mentioned and properly
respected?

Answer: [Yes]

Justification: We cited the datasets and packages used.

Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of
that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should
be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for
some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived
asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset’s
creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their sub-
missions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset is
used.

32


---Page Break---
• At submission time, remember to anonymize your assets (if applicable). You can either create
an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing or research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human
subjects.
• Including this information in the supplemental material is fine, but if the main contribution of
the paper involves human subjects, then as much detail as possible should be included in the
main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other
labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Sub-
jects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals
(or an equivalent approval/review based on the requirements of your country or institution) were
obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human
subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be
required for any human subjects research. If you obtained IRB approval, you should clearly
state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and
locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for
their institution.
• For initial submissions, do not include any information that would break anonymity (if applica-
ble), such as the institution conducting the review.

33


---Page Break---
