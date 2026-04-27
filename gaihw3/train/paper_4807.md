On Statistical Rates and Provably Efficient Criteria
of Latent Diffusion Transformers (DiTs)

Jerry Yao-Chieh Hu∗†‡
Weimin Wu∗†‡
Zhuoru Li♭

Sophia Pi‡
Zhao Song§
Han Liu†‡♮

†Center for Foundation Models and Generative AI, ‡Department of Computer Science, ♮Department of Statistics
and Data Science, Northwestern University, Evanston, IL 60208, USA
♭School of Mathematical Science, Fudan University, Yangpu, Shanghai 200433, China
§Simons Institute for the Theory of Computing, UC Berkeley, Berkeley, CA 94720, USA

{jhu,wwm,sophiapi2026.1}@u.northwestern.edu; 21300180107@m.fudan.edu.cn;

magic.linuxkde@gmail.com; hanliu@northwestern.edu

Abstract

We investigate the statistical and computational limits of latent Diffusion
Transformers (DiTs) under the low-dimensional linear latent space assumption.
Statistically, we study the universal approximation and sample complexity of the
DiTs score function, as well as the distribution recovery property of the initial data.
Specifically, under mild data assumptions, we derive an approximation error bound
for the score network of latent DiTs, which is sub-linear in the latent space dimen-
sion. Additionally, we derive the corresponding sample complexity bound and show
that the data distribution generated from the estimated score function converges
toward a proximate area of the original one. Computationally, we characterize
the hardness of both forward inference and backward computation of latent DiTs,
assuming the Strong Exponential Time Hypothesis (SETH). For forward inference,
we identify efficient criteria for all possible latent DiTs inference algorithms and
showcase our theory by pushing the efficiency toward almost-linear time inference.
For backward computation, we leverage the low-rank structure within the gradient
computation of DiTs training for possible algorithmic speedup. Specifically, we
show that such speedup achieves almost-linear time latent DiTs training by casting
the DiTs gradient as a series of chained low-rank approximations with bounded
error. Under the low-dimensional assumption, we show that the statistical rates and
the computational efficiency are all dominated by the dimension of the subspace,
suggesting that latent DiTs have the potential to bypass the challenges associated
with the high dimensionality of initial data.

1
Introduction

We investigate the statistical and computational limits of latent diffusion transformers (DiTs), assum-
ing the data is supported on an unknown low-dimensional linear subspace. This analysis is not only
practical but also timely. On one hand, DiTs have demonstrated revolutionary success in generative
AI and digital creation by using Transformers as score networks [Esser et al., 2024, Ma et al., 2024,
Chen et al., 2024a, Mo et al., 2023, Peebles and Xie, 2023]. On the other hand, they require significant
computational resources [Liu et al., 2024], making them challenging to train outside of specialized
industrial labs. Therefore, it is natural to ask whether it is possible to make them lighter and faster
without sacrificing performance. Answering these questions requires a fundamental understanding of

∗Equal contribution. Future updates are on arXiv.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the DiT architecture. This work provides a timely theoretical analysis of the fundamental limits of
DiT architecture, aided by the analytical feasibility provided by the low-dimensional data assumption.

Empirically, Latent Diffusion is a go-to design for effectiveness and computational efficiency [Rom-
bach et al., 2022, Liu et al., 2021, Pope et al., 2021, Su and Wu, 2018]. Theoretically, it is capable of
hosting the assumption of low-dimensional data structure (see Assumption 2.1 for formal definition)
for detailed analytical characterization [Chen et al., 2023, Bortoli, 2022]. In essence, diffusion models
with low-dimensional data structures manifest a natural lower-dimensional diffusion process through
an encoder/decoder within a robust and informative latent representation feature space [Rombach
et al., 2022, Pope et al., 2021]. Such lower-dimensional diffusion improves computational efficiency
by reducing data complexity without sacrificing essential information [Liu et al., 2021]. With this
assumption, Chen et al. [2023] decompose the score function of U-Net based diffusion models into
on-support and orthogonal components. This decomposition allows for the characterization of the
distinct behaviors of the two components: the on-support component facilitates latent distribution
learning, while the orthogonal component facilitates subspace recovery.

In our work, we utilize low-dimensional data structure assumption to explore statistical and com-
putational limits of latent DiTs. Our analysis includes the characterizations of statistical rates and
provably efficient criteria. Statistically, we pose two questions and provide a theory to characterize
the statistical rates of latent DiT under the assumption of a low-dimensional data:

Question 1. What is the approximation limit of using transformers to approximate the DiT score
function, particularly in the low-dimensional data subspace?

Question 2. How accurate is the estimation limit for such a score estimator in practical training
scenarios? With the score estimator, how well can diffusion transformers recover the data distribution?

Computationally, the primary challenge of DiT lies in the transformer blocks’ quadratic complexity.
This computational burden applies to both inference and training, even with latent diffusion. Thus, it
is essential to design algorithms and methods to circumvent this Ω(L2) where L is the latent DiT
sequence length. However, there are no formal results to support and characterize such algorithms.
To address this gap, we pose the following questions and provide a fundamental theory to fully
characterize the complexity of latent DiT under the low-dimensional linear subspace data assumption:

Question 3. Is it possible to improve the Ω(L2) time complexity with a bounded approximation error
for both forward and backward passes? What is the computational limit for such an improvement?

Contributions.
We study the fundamental limits of latent DiT. Our contributions are threefold:

• Score Approximation. We address Question 1 by characterizing the approximation limit of
matching the DiT score function with a transformer-based score estimator. Specifically, under mild
data assumptions, we derive an approximation error bound for the score network, sub-linear in
the latent space dimension (Theorem 3.1). These results not only explain the expressiveness of
latent DiT (under mild assumptions) but also provide guidance for the structural configuration of
the score network for practical implementations (Theorem 3.1).
• Score and Distribution Estimation. We address Question 2 by exploring the limitations of score
and distribution estimations of latent DiTs in practical training scenarios. Specifically, we provide a
sample complexity bound for score estimation (Theorem 3.2), using norm-based covering number
bound of transformer architecture. Additionally, we show that the learned score estimator is able
to recover the initial data distribution (Corollary 3.2.1).
• Provably Efficient Criteria and Existence of Almost Linear Time Algorithms. We address
Question 3 by providing provably efficient criteria for latent DiTs in both forward inference and
backward computation/training. For forward inference, we characterize all possible efficient
DiT algorithms using a norm-based efficiency threshold for both conditional and unconditional
generation (Proposition 4.1). Efficient algorithms, including almost-linear time algorithms (Propo-
sition 4.2), are possible only below this threshold. For backward computation, we prove the
existence of almost-linear time DiT training algorithms (Theorem 4.1) by utilizing the inherent
low-rank structure in DiT gradients through a chained low-rank approximation.

Interestingly, both our statistical and computational results are dominated by the subspace dimen-
sion under the low-dimensional assumption, suggesting that latent DiT can potentially bypass the
challenges associated with the high dimensionality of initial data.

2


---Page Break---
Organization.
Section 2 includes background on score decomposition and Transformer-based
score networks. Section 3 includes DiTs’ statistical rates. Section 4 includes DiTs’ provably efficient
criteria. Section 5 includes concluding remarks. We defer discussions of related works to Appendix C.

Notations.
We use lower case letters to denote vectors, e.g., z ∈RD.
∥z∥2 and ∥z∥∞
denote its Euclidean norm and Infinite norm respectively.
We use upper case letters to de-
note matrix, e.g., Z ∈Rd×L.
∥Z∥2, ∥Z∥op, and ∥Z∥F denote the 2-norm, operator norm
and Frobenius norm respectively.
∥Z∥p,q denotes the p, q-norm where the p-norm is over
columns and q-norm is over rows. Given a function f, let ∥f(x)∥L2 := (
R
∥f(x)∥2
2dx)1/2, and
∥f(·)∥Lip = supx̸=y(∥f(x) −f(y)∥2/∥x −y∥2). With a distribution P, we denote ∥f∥L2(P ) =

(
R

P ∥f(x)∥2
2dx)1/2 as the L2(P) norm. Let f♯P be a pushforward measure, i.e., for any measurable
Ω, (f♯P)(Ω) = P(f −1(Ω)). We use ψ for (conditional) Gaussian density functions.

2
Background

This section reviews the ideas we built on, including an overview of diffusion models (Section 2.1),
the score decomposition under the linear latent space assumption (Section 2.2), and the transformer
backbone in DiT (Section 2.3).

2.1
Score-Matching Denoising Diffusion Models
We briefly review forward process, backward process and score matching in diffusion models.

Forward and Backward Process.
In the forward process, Diffusion models gradually add noise
to the original data x0 ∈RD, and x0 ∼P0. Let xt denote the noisy data at time stamp t, with
marginal distribution and destiny as Pt and pt. The conditional distribution P(xt|x0) follows
N(β(t)x0, σ(t)ID), where β(t) = exp(−
R t
0 w(s)ds/2), σ(t) = 1 −β2(t), and w(t) > 0 is a
nondecreasing weighting function. In practice, the forward process terminates at a large enough T
such that PT is close to N(0, ID). In the backward process, we obtain yt by reversing the forward
process. The generation of yt depends on the score function ∇log pt(·). However, this is unknown in
practice, we use a score estimator sW (·, t) to replace ∇log pt(·), where sW (·, t) is usually a neural
network with parameters W. See Appendix D.1 for the details.

Score Matching.
To estimate the score function, we use the following loss

min
W

Z T

T0
γ(t)Ext∼Pt
h
∥sW (xt, t) −∇log pt(xt)∥2
2
i
dt,

where γ(t) is the weight function, and T0 is a small value to stabilize training and prevent score
function from blowing up [Vahdat et al., 2021]. However, it is hard to compute ∇log pt(·) with
available data samples. Therefore, we minimize the equivalent denoising score matching objective

min
W

Z T

T0
γ(t)Ex0∼P0
h
Ext|x0
h
∥sW (xt, t) −∇xt log ψt(xt | x0)∥2
2
ii
dt,
(2.1)

where ψt(xt|x0) is the transition kernel, then ∇xt log ψt(xt|x0) = (β(t)x0 −xt) /σ(t).

To train the parameters W in the score estimator sW (·, t), we use the empirical version of (2.1). We
select n i.i.d. data samples {x0,i}n
i=1 ∼P0, and sample time ti (1 ≤i ≤n) uniformly from interval
[T0, T]. Given x0,i, we sample xti from N(β(ti)x0,i, σ(ti)ID). The empirical loss is

bL(W) = 1

n

n
X

i=1
∥sW (xti, ti) −x0,i∥2
2.
(2.2)

For convenience of notation, we denote population loss L(W) = EP0[ bL(W)].

2.2
Score Decomposition in Linear Latent Space
In this part, we review the score decomposition in [Chen et al., 2023]. We consider that the D-
dimensional input data x supported on a d0-dimensional subspace, where d0 ≤D.

Assumption 2.1 (Low-Dimensional Linear Latent Space). Let x denote the initial data at t = 0. x
has a latent representation via x = Bh, where B ∈RD×d0 is an unknown matrix with orthonormal
columns. The latent variable h ∈Rd0 follows the distribution Ph with a density function ph.

3


---Page Break---
Remark 2.1. By “linear latent space,” we mean that each entry of a given latent vector is a linear
combination of the corresponding input, i.e., x = Bh. This is also known as the “low-dimensional
data” assumption in literature [Chen et al., 2023].

Let x and h denote the perturbed data and its associated latent variable at t > 0, respectively. Based
on the low-dimensional data structure assumption, we have the following score decomposition theory:
on-support score s+(B⊤x, t) and orthogonal score s−(x, t).

Lemma 2.1 (Score Decomposition, Lemma 1 of [Chen et al., 2023]). Let data x = Bh follow
Assumption 2.1. The decomposition of score function ∇log pt(x) is

∇log pt(x) = B∇log ph
t (h)
|
{z
}

s+(h,t)

−
 
ID −BB⊤

x/σ(t)
|
{z
}
s−(x,t)

,
h = B⊤x,
(2.3)

where ph
t (h) :=
R
ψt(h|h)ph(h)dh, ψt(·|h) is the Gaussian density function of N(β(t)h, σ(t)Id0),
β(t) = e−t/2 and σ(t) = 1 −e−t. We restate the proof in Appendix D.2 for completeness.

Additionally, our theoretical analysis is based on two following assumptions as in [Chen et al., 2023].

Assumption 2.2 (Tail Behavior of Ph). The density function ph > 0 is twice continuously differen-
tiable. Moreover, there exist positive constants A0, A1, A2 such that when ∥h∥2 ≥A0, the density
function ph(h) ≤(2π)−d0/2A1exp(−A2∥h∥2
2/2).

Assumption 2.3 (Ls+-Lipschitz of s+(h, t)). The on-support score function s+(h, t) is Ls+-
Lipschitz in h ∈Rd0 for any t ∈[0, T].

2.3
Score Network and Transformers
In this part, we introduce the score network architecture and Transformers. Transformers are the
backbone of the score network in DiT. By Assumption 2.1, h = B⊤x ∈Rd0 with d0 < D.

(Latent) Score Network.
Following [Chen et al., 2023], we rearrange (2.3) into

∇log pt(x) = B(σ(t)∇log ph
t (B⊤x) + B⊤x
|
{z
}
:=q(B⊤x,t): Rd0×[T0,T ] →Rd0

)/σ(t) −x/σ(t).
(2.4)

We use WB ∈RD×d0 to approximate B ∈RD×d0, and a neural network f(W ⊤
B x, t) to approximate
q(B⊤x, t). We adopt the following score network class for diffusion in latent space (i.e., in h ∈Rd0)

S =

sW (x, t) = WBf(W T
B x, t)/σ(t) −x/σ(t), W = {WB, f}
	
,
(2.5)

where the columns in WB are orthogonal, f : Rd0 × [T0, T] →Rd0 is a neural network. In this work,
we focus on the diffusion transformers (DiTs), i.e., using Transformer for f [Peebles and Xie, 2023].

Transformers.
A Transformer block consists of a self-attention layer and a feed-forward layer,
with both layers having skip connection. We use τ r,m,l : Rd×L →Rd×L to denote a Transformer
block. Here r and m are the number of heads and head size in self-attention layer, and l is the hidden
dimension in feed-forward layer. Let X ∈Rd×L be the model input, then we have the model output

Attn(X) = X +
Xr

i=1 W i
OW i
V X · Softmax
 
W i
KX
T W i
QX

,
(2.6)

FF ◦Attn(X) = Attn(X) + W2 · ReLU(W1 · Attn(X) + b11T
L) + b21T
L,
(2.7)

where W i
K, W i
Q, W i
V ∈Rm×d, W i
O ∈Rd×m, W1 ∈Rl×d, W2 ∈Rd×l, b1 ∈Rl, b2 ∈Rd.

In our work, we use Transformer networks with positional encoding E ∈Rd×L. We define the
Transformer networks as the composition of Transformer blocks

T r,m,l
P
= {fT : Rd×L →Rd×L | fT is a composition of blocks τ r,m,l’s}.
For example, the following is a Transformer network consisting K blocks and positional encoding

fT (X) = FF(K) ◦Attn(K) ◦· · · FF(1) ◦Attn(1)(X + E).
(2.8)

3
Statistical Rates of Latent DiTs with Subspace Data Assumption
In this section, we analyze the statistical rates of latent DiTs. Section 3.1 introduces the class of
latent DiT score networks. In Section 3.2, we prove the approximation limit of matching the DiT

4


---Page Break---
W ⊤
B

Latent Encoder

WB

Latent Decoder

R(·)

Reshape Layer

fT ∈T r,m,l

Transformer Network

R−1(·)

Reversed
Reshape Layer

⊕
x ∈RD
x ∈Rd0
Rd×L
Rd0
RD

1/σ(t)

Rd×L

−1/σ2
t

f = R−1 ◦fT ◦R
:
Rd0 →Rd×L

sW (x, t)

Figure 1: Overview of DiT Score Network Architecture sW (·, t). W T
B denotes the linear layer from the
input data space to the linear latent space. f(·) = R−1 ◦fT ◦R(·) denotes the transformer network fT (·) with
reshaping layer R(·), where fT (·) ∈T r,m,l
p
. WB denotes the linear layer from the linear latent space to the
input data space. σ(t) denote the variance of the conditional distribution P(xt | x0).

score function with the score network class, and characterize the structural configuration of the score
network when a specified approximation error is required. Following this, in Section 3.3, utilizing the
characterized structural configuration, we prove the score and distribution estimation for latent DiTs.

3.1
DiT Score Network Class
Here, we provide the details about DiT score network class used in our analysis. In (2.5), f is
a network with Transformer as the backbone, and (h, t) ∈Rd0 × [T0, T] denotes the input data.
Following [Peebles and Xie, 2023], DiT uses time point t to calculate the scale and shift value in the
Transformer backbone, and it transforms an input picture into a sequential version. To achieve the
transformation, we introduce a reshape layer.

Definition 3.1 (DiT Reshape Layer R(·)). Let R(·) : Rd0 →Rd×L be a reshape layer that transforms
the d0-dimensional input into a d × L matrix. Specifically, for any d0 = i × i image input, R(·)
converts it into a sequence representation with feature dimension d := p2 (where p ≥2) and
sequence length L := (i/p)2. Besides, we define the corresponding reverse reshape (flatten) layer
R−1(·) : Rd×L →Rd0 as the inverse of R(·). By d0 = dL, R, R−1 are associative w.r.t. their input.

To simplify the self-attention block in (2.6), let W i
OV = W i
OW i
V and W i
KQ = (W i
K)TW i
Q.

Definition 3.2 (Transformer Network Class T r,m,l
p
). We define the Transformer network class as

T r,m,l
p
(K, CT , C2,∞
OV , COV , C2,∞
KQ , CKQ, C2,∞
F
, CF , CE, LT ), satisfying the constraints

• Model architecture with K blocks: fT (X) = FF(K) ◦Attn(K) ◦· · · FF(1) ◦Attn(1)(X);
• Model output bound: supX ∥fT (X)∥2 ≤CT ;
• Parameter bound in Attn(i):
(W i
OV )⊤
2,∞≤C2,∞
OV ,
(W i
OV )⊤
2 ≤COV ,
W i
KQ

2,∞≤

C2,∞
KQ ,
W i
KQ

2 ≤CKQ,
E⊤
2,∞≤CE, ∀i ∈[K];

• Parameter bound in FF(i):
W i
j

2,∞≤C2,∞
F
,
W i
j

2 ≤CF , ∀j ∈[2], i ∈[K];

• Lipschitz of fT : ∥fT (X1) −fT (X2)∥F ≤LT ∥X1 −X2∥F , ∀X1, X2 ∈Rd×L.

Definition 3.3 (DiT Score Network Class ST r,m,l
p
(Figure 1)). We denote ST r,m,l
p
as the DiT score

network class in (2.5), replacing f with R−1 ◦fT ◦R, and fT is from the Transformer class T r,m,l
p
.

3.2
Score Approximation of DiT
Here, we explore the approximation limit of latent DiT score network class ST r,m,l
p
under linear
latent space assumption. Recall that Pt is the distribution of xt, σ(t) is the variance of P(xt|x0),
d0 is the dimension of latent space, L is the sequence length of transformer input, T is the stopping
time in forward process, T0 is the early stopping time in backward process, and Ls+ is the Lipschitz
coefficient of on-support score function. Then we have the following Theorem 3.1.

Theorem 3.1 (Score Approximation of DiT). For any approximation error ϵ > 0 and any data
distribution P0 under Assumptions 2.1 to 2.3, there exists a DiT score network s b
W from ST 2,1,4
p
(defined in Definition 3.2), where c
W = {c
WB, bfT }, such that for any t ∈[T0, T], we have:
s b
W (·, t) −∇log pt(·)

L2(Pt) ≤ϵ ·
p

d0/σ(t),

5


---Page Break---
where σ(t) = 1 −e−t, and the upper bound of hyperparameters in ST 2,1,4
p
are

K = O(ϵ−2L), CT = O

d0Ls+
p

d0 log(d0/T0) + log(1/ϵ)

,

C2,∞
OV = (1/ϵ)O(1), COV = (1/ϵ)O(1), C2,∞
KQ = (1/ϵ)O(1), CKQ = (1/ϵ)O(1),

CE = O(L3/2), C2,∞
F
= (1/ϵ)O(1), CF = (1/ϵ)O(1), LT = O
 
d0Ls+

.

Proof Sketch. Our proof is built on the key observation that there is a tail behavior of the
low-dimensional latent variable distribution Ph (Assumption 2.2).
Recall that ∇log pt(x) =
Bq(h, t)/σ(t) −x/σ(t), where h = B⊤x (defined in (2.4)). By taking c
WB = B, our aim reduces
to construct a transformer network to approximate q(h, t). To achieve this, we firstly approximate
q(h, t) with a compact-supported continuous function, based on the tail behavior of Ph. Then we
construct a transformer to approximate the compact-supported continuous function using the universal
approximation capacity of transformer [Yun et al., 2020]. See Appendix F.1 for a detailed proof.

Intuitively, Theorem 3.1 indicates the capability of the transformer-based score network to approx-
imate the score function with precise guarantees. Furthermore, Theorem 3.1 provides empirical
guidance for the design choices of the score network when a specified approximation error is required.
Remark 3.1 (Comparing with Existing Works). Theoretical analysis of DiTs is limited. Previous
works that do not specify the model architecture assume that the score estimator is well-approximated
[Benton et al., 2024, Wibisono et al., 2024]. To the best of our knowledge, this work is the first to
present an approximation theory for DiTs, offering the estimation theory in Theorem 3.2 and Corol-
lary 3.2.1 based on the estimated score network, rather than a perfectly trained one.

Remark 3.2 (Latent Dimension Dependency). Theorem 3.1 suggests that the approximation capacity
and Transformer network size primarily depend on the latent variable dimension d0 = d × L. This
indicates that DiTs can potentially bypass the challenges associated with the high dimensionality of
initial data by transforming input data into a low-dimensional latent variable.

3.3
Score Estimation and Distribution Estimation
Besides score approximation capability, Theorem 3.1 also characterizes the structural configuration
of the score network for any specific precision, e.g., K, CE, CF , etc. This characterization enables
further analysis of the performance of score network in practical scenarios. In Theorem 3.2, we
provide a sample complexity bound for score estimation. In Corollary 3.2.1, show that the learned
score estimator is able to recover the initial data distribution.

Score Estimation.
To derive a sample complexity for score estimation using ST 2,1,4
p
, we rewrite

the score matching objective in (2.2) as c
W ∈argminsW ∈ST 2,1,4
p
bL(sW ), c
W = {c
WB, bfT }.

Theorem 3.2 shows that as sample size n →∞, sW (·, t) convergences to ∇log pt(·).

Theorem 3.2 (Score Estimation of DiT). Under Assumptions 2.1 to 2.3, we choose ST 2,1,4
p
as in
Theorem 3.1 using ϵ ∈(0, 1) and L > 1, With probability 1 −1/poly(n), we have

1
T −T0

Z T

T0

s b
W (·, t) −∇log pt(·)

L2(Pt)dt = e
O

1
n1/3T0T · 2(1/ϵ)2L +
1
n1/3T0T +
1
T0T ϵ2

,

(3.1)

where e
O hides the factors related to D, d0, d, Ls+, and log n.

Proof. See Appendix F.2 for a detailed proof.

Intuitively, Theorem 3.2 shows a sample complexity bound for score estimation in practice.
Remark 3.3 (Comparing with Existing Works). [Zhu et al., 2023] provides a sample complexity
for simple ReLU-based diffusion models under the assumption of an accurate score estimator. To
the best of our knowledge, we are the first to provide a sample complexity for DiTs, based on the
learned score network in Theorem 3.1 and the quantization (piece-wise approximation) approach for

6


---Page Break---
transformer universality [Yun et al., 2020]. Furthermore, our first term shows a convergence rate of
1/T, outperforming [Chen et al., 2023], in which the first term is independent of T.

Remark 3.4 (Double Exponential Factor and Inconsistent Convergence). Theorem 3.2 reports an
explicit result on sample complexity bounds for score estimation of latent DiTs: a double exponential
factor 2(1/ϵ)2L in the first term. We remark that this arises from the required depth K is O(ϵ−2L),
and the norm of required weight parameters is (1/ϵ)O(1) as shown in Theorem 3.1, assuming the
universality of transformers requires dense layers [Yun et al., 2020]. This double exponential factor
causes inconsistent convergence with respect to sample size n, as its large value prevents setting ϵ as
a function of n to balance the first and second terms in (3.1). This motivates us to rethink transformer
universality and explore new proof techniques for DiTs, which we leave for future work.

Definition 3.4. For later convenience, we define ξ(n, ϵ, L) :=
1
n1/3 · 2(1/ϵ)2L +
1
n1/3 + ϵ2.

Distribution Estimation.
In practice, DiTs generate data using the discretized version with step size
µ, see Appendix D.1 for details. Let bPT0 be the distribution generated by s b
W using the discretized
backward process in Theorem 3.2. Let P h
T0 and ph
T0 be the distribution and density function of
on-support latent variable h at T0. We have the following results for distribution estimation.

Corollary 3.2.1 (Distribution Estimation of DiT, Modified From Theorem 3 of [Chen et al., 2023]).
Let T = O(log n), T0 = O(min{c0, 1/Ls+}), where c0 is the minimum eigenvalue of EPh[hh⊤].
With the estimated DiT score network s b
W in Theorem 3.2, we have the following with probability
1 −1/poly(n).

(i) The accuracy to recover the subspace B is
WBW ⊤
B −BB⊤2
F = e
O (ξ(n, ϵ, L)/c0).
(ii) With the conditions KL(Ph||N(0, Id0)) < ∞, there exists an orthogonal matrix U ∈Rd×d such
that we have the following upper bound for the total variation distance

TV(P h
T0, (WBU)⊤
♯bPT0) = e
O(
p

ξ(n, ϵ, L) · log n),
(3.2)

where e
O hides the factor about D, d0, d, Ls+, log n, and T −T0. and (WBU)⊤
♯bPT0 denotes the
pushforward distribution.
(iii) For the generated data distribution bPT0, the orthogonal pushforward (I −WBW ⊤
B )♯bPT0 is
N(0, Σ), where Σ ⪯aT0I for a constant a > 0.

Proof. See Appendix F.3 for a detailed proof.

Intuitively, Corollary 3.2.1 shows the estimation results in 3 parts: (i) the accuracy of recovering the
subspace B; (ii) the estimation error between bPT0 and P h
T0; and (iii) the vanishing behavior of bPT0 in
the orthogonal space. These indicate that the learned score estimator is capable of recovering the
initial data distribution. Notably, Corollary 3.2.1 is agnostic to the specifics of ξ(n, ϵ, L).

Remark 3.5 (Comparing with Existing Works). Oko et al. [2023] analyze the distribution estimation
under the assumption that the initial density is supported on [−1, 1]D and smooth in the boundary.
Our Assumption 2.2 demonstrates greater practical relevance. This suggests that our method of
distribution estimation aligns more closely with empirical realities.

Remark 3.6 (Subspace Recovery Accuracy). (i) of Corollary 3.2.1 confirms that the subspace is
learned by DiTs. The error is proportional to the sample complexity for score estimation and depends
on the minimum eigenvalue of the covariance of Ph.

4
Provably Efficient Criteria

Here, we analyze the computational limits of latent DiTs under low-dimensional linear subspace data
assumption (i.e., Assumption 2.1). The hardness of DiT models ties to both forward and backward
passes of the score network in Definition 3.3. We characterize them separately.

4.1
Computational Limits of Backward Computation
Following Section 2, suppose we have n i.i.d. data samples {x0,i}n
i=1 ∼Pd, and time ti0 (1 ≤
i ≤n) uniformly sampled from [T0, T]. For each data x0,i ∈RD, we sample xti0 ∈RD from

7


---Page Break---
N(β(ti0)x0,i, σ(ti0)ID). Let (WAR−1(·))† be the inverse transformation of WAR−1(·), and denote
Y0,i := (WAR−1)†(x0,i) ∈Rd×L. We rewrite the empirical denoising score-matching loss (2.2) as

1
n

n
X

i=1

WAR−1(fT (R(W ⊤
A xti0
| {z }
d0×1

))) −x0,i

2

F = 1

n

n
X

i=1

 WA
|{z}
D×d0

d×L

R−1  z
}|
{
fT (R(W ⊤
A xti0)
|
{z
}
d0×1

) −Y0,i
|{z}
d×L


2

F .

(4.1)
For efficiency, it suffices to focus on just transformer attention heads of the DiT score network due to
their dominating quadratic time complexity in both passes. Thus, we consider only a single layer
attention for fT , to simplify our analysis. Further, we consider the following simplifications:

(S0) To prove the hardness of (4.1) for both full gradient descent and stochastic mini-batch gradient
descent methods, it suffices to consider training on a single data point.
(S1) For the convenience of our analysis, we consider the following expression for attention mech-
anism.
Let X, Y
∈Rd×L.
Let WK, WQ, WV
∈Rs×d be attention weights such that
Q = WQX ∈Rd×L, K = WKX ∈Rs×L and V = WV X ∈Rs×L. We write attention
mechanism of hidden size s and sequence length L as

Att(X) = (WOWV X)
|
{z
}
V multiplication

D−1 exp
 
XTW T
KWQX


|
{z
}
K-Q multiplication

∈Rd×L,
(4.2)

with D := diag(exp
 
XWQW T
KXT
1L). Here, exp(·) is entry-wise exponential function, i.e.,
exp(A)i,j = exp(Ai,j) for any matrix A , diag(·) converts a vector into a diagonal matrix with
the vector’s entries on the diagonal, and 1L is the length-L all ones vector.
(S2) Since V multiplication is linear in weight while K-Q multiplication is exponential in weights,
we only need to focus on the gradient update of K-Q multiplication. Therefore, for efficiency
analysis of gradient, it is equivalent to analyzing a reduced problem with fixed WOWV X =
const..
(S3) To focus on the DiT, we consider the low-dimensional linear encoder WA to be pretrained and
to not participate in gradient computation. This aligns with common practice [Rombach et al.,
2022] and is justified by the trivial computation cost due to the linearity of WA2.
(S4) To further simplify, we introduce A1, A2, A3 ∈Rs×L and W ∈Rd×d via
WAR−1 
fT (R(W ⊤
A xti0)
|
{z
}
:=X∈Rd×L

) −
Y0,i
|{z}
:=Y ∈Rd×L


2

F

 
By (S0), (S1) and (S2)

=
WAR−1 
WOWV
| {z }
:=WOV ∈Rd×d
X
|{z}
:=A3∈Rd×L
D−1 exp
 
XT
|{z}
:=A⊤
1 ∈RL×d
W T
KWQ
| {z }
:=W ∈Rd×d
X
|{z}
:=A2∈Rd×L


−Y

2

F .

(4.3)
Notably, A1, A2, A3, X, Y are constants w.r.t. training above loss with gradient updates.

Therefore, we simplify the objective of training DiT into

Definition 4.1 (Training Generic DiT Loss). Given A1, A2, A3, Y ∈Rd×L and WOV , W ∈Rd×d
following (S4), Training a DiT with ℓ2 loss on a single data point X, Y ∈Rd×L is formulated as

min
W
L0(W) = min
W
1
2

WAR−1 
WOV A3D−1 exp
 
A⊤
1 WA2

−Y

2

F .
(4.4)

Here D := diag(exp
 
A⊤
1 WA2

1n) ∈RL×L.

Remark 4.1 (Conditional and Unconditional Generation). L0 is generic. If A1 ̸= A2 ∈Rd×L,
Definition 4.1 reduces to cross-attention in DiT score net (for conditional generation). If A1 = A2 ∈
Rd×L, Definition 4.1 reduces to self-attention in DiT score net (for unconditional vanilla generation).

We introduce the next problem to characterize all possible gradient computations of optimizing (4.4).

2The gradient computation is linear in WA, and hence the computation w.r.t. WA is cheap and upper-bounded
by L · poly(d) time in a straightforward way.

8


---Page Break---
Problem
1
(Approximate
DiT
Gradient
Computation
(ADITGC(L, d, Γ, ϵ))).
Given
A1, A2, A3, Y
∈Rd×L.
Let ϵ > 0.
Assume all numerical values are in O(log(L))-bits
encoding. Let loss function L0 follow Definition 4.1. The problem of approximating gradient
computation of optimizing empirical DiT loss (4.4) is to find an approximated gradient matrix
eG(W ) ∈Rd×d such that
 eG

(W ) −∂L

∂W


max ≤1/poly(L). Here, ∥A∥max := maxi,j |Aij| for any
matrix A.

In this work, we aim to investigate the computational limits of all possible efficient algorithms of
ADITGC with ϵ = 1/poly(L). Yet, the explicit gradient of DiT denoising score matching loss (4.4)
is too complicated to characterize ADITGC. To combat this, we make the following observations.

(O1) Let g1(·) := WAR−1(·) : Rd×L →Rd0, g2(·) := Att(·) : Rd×L →Rd×L, and g3(·) :=
R(W ⊤
A ·) : RD →Rd×L such that g3(x) = X for x ∈RD (with D > d0 = dL).
(O2) Vectorization of fT . For the ease of presentation, we use notation flexibly that fT to denote
both a matrix in Rd×L and a vector in RdL in the following analysis. This practice does not
affect correctness. The context in which fT is used should clarify whether it refers to a matrix
or a vector. Explicit vectorization follows Definition D.1.
(O3) Linearity of g1. By linearity of WAR−1(·), we treat g1 as a matrix in Rd0×dL acting on vector
fT (·) ∈RdL.

Therefore, we have L0 = ∥g1 · [g2(g3) −Y ]∥2
2, such that its gradient involves dL0

dW = g1
dg2
dW . From
above, we only need to focus on proving the computation time and error control of term dg2

dW
for gradient w.r.t W. Luckily, with tools from fine-grained complexity theory [Alman and Song,
2023, 2024a,b,c] and tensor trick (see Appendix D.3), we prove the existence of almost-linear time
algorithms for Problem 1 in the next theorem. Let vec(W) := W for any matrix W following
Definition D.1.

Theorem 4.1 (Existence of Almost-Linear Time Algorithms for ADITGC). Suppose all numerical
values are in O(log L)-bits encoding. Let max(∥WOV A3∥max, ∥WKA1∥max, ∥WQA2∥max) ≤Γ.
There exists a L1+o(1) time algorithm to solve ADITGC(Lp, L, d = O(log L), Γ = o(√log L)) (i.e.,
Problem 1) with loss L0 from Definition 4.1 up to 1/poly(L) accuracy. In particular, this algorithm

outputs gradient matrices eG(W ) ∈Rd×d such that
 eG

(W ) −∂L

∂W


max ≤1/poly(L).

Proof Sketch. Our proof is built on the key observation that there exist low-rank structures within the
DiT training gradients. Using the tensor trick [Diao et al., 2019, 2018] and computational hardness
results of attention [Hu et al., 2024b, Alman and Song, 2023], we approximate DiT training gradients
with a series of low-rank approximations and carefully match the multiplication dimensions so
that the computation of dg2

dW forms a chained low-rank approximation. We complete the proof by
demonstrating that this approximation is bounded by a 1/poly(L) error and requires only almost-
linear time. See Appendix G.2 for a detailed proof.

Remark 4.2. We remark that Theorem 4.1 is dominated by the relation between L and d, hence by
the subspace dimension3 d0 = dL. A smaller d0 makes Theorem 4.1 more likely to hold.

4.2
Computational Limits of Forward Inference
Since the inference of score-matching diffusion models is a forward pass of the trained score estimator
sW , the computational hardness of DiT ties to the transformer-based score network,

sW (A1, A2, A3) = WAR−1 
WOV A3
|
{z
}
d×L
D−1
|{z}
L×L
exp
 
A⊤
1 W ⊤
K
| {z }
L×s

WQA2
| {z }
s×L


,
(4.5)

following notation in Definition 4.1. For inference, we study the following approximation problem.
Notably, by Remark 4.1, (4.5) subsumes both conditional and unconditional DiT inferences.

Problem 2 (Approximate DiT Inference ADITI(d, L, Γ, δF )). Let δF > 0 and B > 0. Given
A1, A2, A3
∈
Rd×L, and WOV , WK, WQ
∈
Rd×d with guarantees that ∥WOV A3∥∞
≤
B, ∥WKA1∥∞
≤B and ∥WQA2∥∞
≤B, we aim to study an approximation problem

3See Assumption 2.1.

9


---Page Break---
ADITI(d, L, B, δF ), that approximates sW (A1, A2, A3) with a vector ez ∈Rd0 (with d0 = d·L) such
that
ez −WAR−1  
WOV A3D−1 exp
 
A⊤
1 W ⊤
KWQA2

max ≤δF . Here, ∥A∥max := maxi,j |Aij|
for any matrix A.

By (O2) and (O3), we make an observation that Problem 2 is just a special case of [Alman and Song,
2023]. Hence, we characterize the all possible efficient algorithms for ADITI with next proposition.

Proposition 4.1 (Norm-Based Efficiency Phase Transition). Let ∥WQA2∥∞≤B, ∥WKA1∥∞≤B
and ∥WOV A3∥∞≤B with B = O(√log L). Assuming SETH (Hypothesis 1), for every q > 0,
there are constants C, Ca, Cb > 0 such that: there is no O(n2−q)-time (sub-quadratic) algorithm for
the problem ADITI(L, d = C log L, B = Cb
√log L, δF = L−Ca).

Remark 4.3. Proposition 4.1 suggests an efficiency threshold for the upper bound of ∥WKA1∥∞,
∥WQA2∥∞, ∥WOV A3∥∞. Only below this threshold are efficient algorithms for Problem 2 possible.

Moreover, there exist almost-linear DiT inference algorithms following [Alman and Song, 2023].

Proposition 4.2 (Almost-Linear Time DiT Inference). Assuming SETH, the DiT inference problem
ADITI(L, d = O(log L), B = o(√log L), δF = 1/poly(L)) can be solved in L1+o(1) time.

Remark 4.4. Proposition 4.2 is a special case of Proposition 4.1 under the efficiency threshold.
Remark 4.5. Propositions 4.1 and 4.2 are dominated by the relation between L and d, hence by the
subspace dimension d0 = dL. A smaller d0 makes Propositions 4.1 and 4.2 more likely to hold.

5
Discussion and Concluding Remarks

We explore the fundamental limits of latent DiTs with 3 key contributions. First, we prove that
transformers are universal approximators for the score functions in DiTs (Theorem 3.1), with
approximation capacity and model size dependent only on the latent dimension, suggesting DiTs
can handle high-dimensional data challenges. Second, we show that Transformer-based score
estimators converge to the true score function (Theorem 3.2), ensuring the generated data distribution
closely approximates the original (Corollary 3.2.1). Third, we provide provably efficient criteria
(Proposition 4.1) and prove the existence of almost-linear time algorithms for forward inference
(Proposition 4.2) and backward computation (Theorem 4.1). Our computational results hold for both
unconditional and conditional generation of DiTs (Remark 4.1). These results highlight the potential
of latent DiTs to achieve both computational efficiency and robust performance in practical scenarios.

Practical Guidance from Computational Results.
Section 4 analyzes the computational feasibility
and identifies all possible efficient DiT algorithms/methods for both forward inference and backward
training. These results provide practical guidance for designing efficient methods:

• The latent dimension should be sufficiently small: d = O(log L) (Theorem 4.1, Propositions 4.1
and 4.2).
• Normalization of K, Q, and V in DiT attention heads enhances performance and efficiency:
– For efficient inference: max {∥WKA1∥, ∥WQA2∥, ∥WOV A3∥} ≤B with B = o(√log L)
(Proposition 4.2) and A1, A2, A3 being the input data associated with K, Q, V .
– For efficient training: max {∥WKA1∥, ∥WQA2∥, ∥WOV A3∥} ≤Γ with Γ = o(√log L) (The-
orem 4.1).

We remark that these conditions are necessary but not sufficient; sufficient conditions depend on the
specific design of the methods used. This is due to the best- or worst-case nature of hardness results.

Limitations and Future Direction.
As discussed in Remark 3.4, the double exponential factor in
our explicit sample complexity bound (Theorem 3.2) suggests a possible gap in our understanding
of transformer universality and its interplay with DiT architecture. This motivates us to rethink
transformer universality and explore new proof techniques for DiTs, which we leave for future work.
Besides, due to its formal nature, this work does not provide immediate practical implementations.
However, we expect that our findings provide valuable insights for future diffusion generative models.

Post-Acceptance Note [October, 29, 2024].
During preparation of the camera-ready version, we
learn of a follow-up work [Anonymous, 2024b] that alleviates the double exponential factor and
achieves minimax optimal statistical rates for DiTs under Hölder smoothness data assumptions.

10


---Page Break---
Broader Impact

This theoretical work aims to shed light on the foundations of diffusion generative models and is not
anticipated to have negative social impacts.

Acknowledgments

JH would like to thank to Minshuo Chen, Sophia Pi, Yi-Chen Lee, Yu-Chao Huang, Yibo Wen,
Damien Jian, Jialong Li, Zijia Li, Tim Tsz-Kit Lau, Chenwei Xu, Dino Feng and Andrew Chen for
enlightening discussions on related topics, and the Red Maple Family for support. The authors would
like to thank the anonymous reviewers and program chairs for constructive comments.

JH is partially supported by the Walter P. Murphy Fellowship. HL is partially supported by NIH
R01LM1372201, AbbVie and Dolby. The content is solely the responsibility of the authors and does
not necessarily represent the official views of the funding agencies.

References

Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya, Dmitry Krotov, Carlo
Lucibello, and Luca Ambrogioni. Losing dimensions: Geometric memorization in generative
diffusion. arXiv preprint arXiv:2410.08727, 2024.

Josh Alman and Zhao Song. Fast attention requires bounded entries. Advances in Neural Information
Processing Systems (NeurIPS), 36, 2023.

Josh Alman and Zhao Song. How to capture higher-order correlations? generalizing matrix soft-
max attention to kronecker computation. In The Twelfth International Conference on Learning
Representations (ICLR), 2024a.

Josh Alman and Zhao Song. The fine-grained complexity of gradient computation for training large
language models. In NeurIPS. arXiv preprint arXiv:2402.04497, 2024b.

Josh Alman and Zhao Song. Fast rope attention: Combining the polynomial method and fast fourier
transform. In manuscript, 2024c.

Luca Ambrogioni. In search of dispersed memories: Generative diffusion models are associative
memory networks. arXiv preprint arXiv:2309.17290, 2023.

Anonymous. Fundamental limits of prompt tuning transformers: Universality, capacity and efficiency.
In Submitted to The Thirteenth International Conference on Learning Representations, 2024a.
URL https://openreview.net/forum?id=jDpdQPMosW. under review.

Anonymous. On statistical rates of conditional diffusion transformer: Approximation and estimation.
In Submitted to The Thirteenth International Conference on Learning Representations, 2024b.
URL https://openreview.net/forum?id=c54apoozCS. under review.

Fan Bao, Chongxuan Li, Yue Cao, and Jun Zhu. All are worth words: a vit backbone for score-based
diffusion models. In NeurIPS 2022 Workshop on Score-Based Methods, 2022.

Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-linear con-
vergence bounds for diffusion models via stochastic localization. In The Twelfth International
Conference on Learning Representations (ICLR), 2024.

Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis.
Transactions on Machine Learning Research, 2022. ISSN 2835-8856.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok,
Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-$\alpha$: Fast training of diffusion transformer
for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning
Representations (ICLR), 2024a.

11


---Page Break---
Minshuo Chen, Xingguo Li, and Tuo Zhao. On generalization bounds of a family of recurrent neural
networks. In Proceedings of the Twenty Third International Conference on Artificial Intelligence
and Statistics (AISTATS), volume 108, pages 1233–1243, 2020a.

Minshuo Chen, Wenjing Liao, Hongyuan Zha, and Tuo Zhao. Distribution approximation and statisti-
cal estimation guarantees of generative adversarial networks. arXiv preprint arXiv:2002.03938,
2020b.

Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and
distribution recovery of diffusion models on low-dimensional data. In International Conference on
Machine Learning (ICML), pages 4672–4712. PMLR, 2023.

Minshuo Chen, Song Mei, Jianqing Fan, and Mengdi Wang. An overview of diffusion models: Ap-
plications, guided generation, statistical rates and optimization. arXiv preprint arXiv:2404.07771,
2024b.

Marek Cygan, Holger Dell, Daniel Lokshtanov, Dániel Marx, Jesper Nederlof, Yoshio Okamoto,
Ramamohan Paturi, Saket Saurabh, and Magnus Wahlström. On problems as hard as cnf-sat. ACM
Transactions on Algorithms (TALG), 12(3):1–24, 2016.

Huaian Diao, Zhao Song, Wen Sun, and David Woodruff. Sketching for kronecker product regression
and p-splines. In International Conference on Artificial Intelligence and Statistics (AISTATS),
pages 1299–1308. PMLR, 2018.

Huaian Diao, Rajesh Jayaram, Zhao Song, Wen Sun, and David Woodruff. Optimal sketching
for kronecker product regression and low rank approximation. Advances in neural information
processing systems (NeurIPS), 32, 2019.

Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable
creation in self-attention mechanisms. In International Conference on Machine Learning (ICML),
pages 5793–5831. PMLR, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for
high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Luciano Floridi and Massimo Chiriatti. Gpt-3: Its nature, scope, limits, and consequences. Minds
and Machines, 30:681–694, 2020.

Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a
strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 23164–23173, 2023a.

Yeqi Gao, Zhao Song, Weixin Wang, and Junze Yin. A fast optimization view: Reformulating single
layer attention in llm based on tensor and svm trick, and solving it in matrix multiplication time.
arXiv preprint arXiv:2309.07418, 2023b.

Yeqi Gao, Zhao Song, and Shenghao Xie. In-context learning for attention scheme: from single soft-
max regression to multiple softmax regression via a tensor trick. arXiv preprint arXiv:2307.02419,
2023c.

Jiuxiang Gu, Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Tensor attention training:
Provably efficient learning of higher-order transformers. arXiv preprint arXiv:2405.16411, 2024.

Jiaqi Guan, Xiangxin Zhou, Yuwei Yang, Yu Bao, Jian Peng, Jianzhu Ma, Qiang Liu, Liang Wang,
and Quanquan Gu. Decompdiff: diffusion models with decomposed priors for structure-based
drug design. arXiv preprint arXiv:2403.07902, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in
neural information processing systems, 33:6840–6851, 2020.

Benjamin Hoover, Hendrik Strobelt, Dmitry Krotov, Judy Hoffman, Zsolt Kira, and Duen Horng
Chau. Memory in plain sight: A survey of the uncanny resemblances between diffusion models
and associative memories. arXiv preprint arXiv:2309.16750, 2023.

12


---Page Break---
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen.
Lora: Low-rank adaptation of large language models.
arXiv preprint
arXiv:2106.09685, 2021.

Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, and Han Liu. On sparse
modern hopfield model. In Thirty-seventh Conference on Neural Information Processing Systems
(NeurIPS), 2023.

Jerry Yao-Chieh Hu, Pei-Hsuan Chang, Haozheng Luo, Hong-Yu Chen, Weijian Li, Wei-Po Wang,
and Han Liu. Outlier-efficient hopfield layers for large transformer-based models. In Forty-first
International Conference on Machine Learning (ICML), 2024a.

Jerry Yao-Chieh Hu, Thomas Lin, Zhao Song, and Han Liu. On computational limits of modern
hopfield models: A fine-grained complexity analysis. In Forty-first International Conference on
Machine Learning (ICML), 2024b.

Jerry Yao-Chieh Hu, Maojiang Su, En-Jui Kuo, Zhao Song, and Han Liu. Computational limits of
low-rank adaptation (lora) for transformer-based models. arXiv preprint arXiv:2406.03136, 2024c.

Jerry Yao-Chieh Hu, Dennis Wu, and Han Liu. Provably optimal memory capacity for modern
hopfield models: Transformer-compatible dense associative memories as spherical codes. In
Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS), volume 38, 2024d.

Russell Impagliazzo and Ramamohan Paturi. On the complexity of k-sat. Journal of Computer and
System Sciences, 62(2):367–375, 2001.

Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V Davuluri. Dnabert: pre-trained bidirectional
encoder representations from transformers model for dna-language in genome. Bioinformatics, 37
(15):2112–2120, 2021.

Haotian Jiang and Qianxiao Li. Approximation theory of transformer networks for sequence modeling.
arXiv preprint arXiv:2305.18475, 2023.

Tokio Kajitsuka and Issei Sato. Are transformers with one layer self-attention using low-rank weight
matrices universal approximators? arXiv preprint arXiv:2307.14023, 2023.

Junghwan Kim, Michelle Kim, and Barzan Mozafari. Provable memorization capacity of transformers.
In The Eleventh International Conference on Learning Representations (ICLR), 2022.

Yingyu Liang, Heshan Liu, Zhenmei Shi, Zhao Song, Zhuoyan Xu, and Junze Yin. Conv-basis: A
new paradigm for efficient attention inference and gradient computation in transformers. arXiv
preprint arXiv:2405.05219, 2024a.

Yingyu Liang, Jiangxuan Long, Zhenmei Shi, Zhao Song, and Yufa Zhou. Beyond linear approx-
imations: A novel pruning approach for attention matrix. arXiv preprint arXiv:2410.11261,
2024b.

Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, and Yufa Zhou. Multi-layer transformers
gradient can be approximated in almost linear time. arXiv preprint arXiv:2408.13233, 2024c.

Yingyu Liang, Zhenmei Shi, Zhao Song, and Yufa Zhou. Tensor attention training: Provably efficient
learning of higher-order transformers. arXiv preprint arXiv:2405.16411, 2024d.

Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang,
Hanchi Sun, Jianfeng Gao, Lifang He, and Lichao Sun. Sora: A review on background, technology,
limitations, and opportunities of large vision models, 2024.

Zhonghua Liu, Yue Lu, Zhihui Lai, Weihua Ou, and Kaibing Zhang. Robust sparse low-rank
embedding for image dimension reduction. Applied Soft Computing, 113:107907, 2021.

Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao,
Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video
generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 10209–10218. IEEE, 2023.

13


---Page Break---
Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and
Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant
transformers. arXiv preprint arXiv:2401.08740, 2024.

Sadegh Mahdavi, Renjie Liao, and Christos Thrampoulidis. Memorization capacity of multi-head
attention in transformers. arXiv preprint arXiv:2306.02010, 2023.

Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d:
Exploring plain diffusion transformers for 3d shape generation. Advances in Neural Information
Processing Systems (NeurIPS), 36, 2023.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew,
Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with
text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution
estimators. In International Conference on Machine Learning (ICML), pages 26517–26582. PMLR,
2023.

OpenAI. Sora: A video generative model based on transformer diffusion. OpenAI Research, 2024.
Accessed: 08/16/2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, and Tom Goldstein. The intrinsic
dimension of images and its impact on learning. arXiv preprint arXiv:2104.08894, 2021.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-
conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Hubert Ramsauer, Bernhard Schafl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler,
Lukas Gruber, Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, et al. Hopfield networks
is all you need. arXiv preprint arXiv:2008.02217, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF confer-
ence on computer vision and pattern recognition (CVPR), pages 10684–10695, 2022.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
Advances in neural information processing systems (NeurIPS), 32, 2019.

Bing Su and Ying Wu.
Learning low-dimensional temporal representations.
In International
Conference on Machine Learning (ICML), pages 4761–4770. PMLR, 2018.

Wenpin Tang and Hanyang Zhao. Score-based diffusion models via stochastic differential equations–a
technical tutorial. arXiv preprint arXiv:2402.07487, 2024.

Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In
Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 11287–11302,
2021.

Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion
language models are versatile protein learners. arXiv preprint arXiv:2402.18567, 2024a.

Yan Wang, Lihao Wang, Yuning Shen, Yiqun Wang, Huizhuo Yuan, Yue Wu, and Quanquan
Gu. Protein conformation generation via force-guided se (3) diffusion models. arXiv preprint
arXiv:2403.14088, 2024b.

Yihan Wang, Jatin Chauhan, Wei Wang, and Cho-Jui Hsieh. Universality and limitations of prompt
tuning. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023.

14


---Page Break---
Andre Wibisono, Yihong Wu, and Kaylee Yingxi Yang. Optimal score estimation via empirical bayes
smoothing. arXiv preprint arXiv:2402.07747, 2024.

Virginia Vassilevska Williams. On some fine-grained questions in algorithms and complexity. In
Proceedings of the international congress of mathematicians: Rio de janeiro 2018, pages 3447–
3487. World Scientific, 2018.

Dennis Wu, Jerry Yao-Chieh Hu, Teng-Yun Hsiao, and Han Liu. Uniform memory retrieval with
larger capacity for modern hopfield models. In Forty-first International Conference on Machine
Learning (ICML), 2024a.

Dennis Wu, Jerry Yao-Chieh Hu, Weijian Li, Bo-Yu Chen, and Han Liu. STanhop: Sparse tan-
dem hopfield model for memory-enhanced time series prediction. In The Twelfth International
Conference on Learning Representations (ICLR), 2024b.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. Are trans-
formers universal approximators of sequence-to-sequence functions? In International Conference
on Learning Representations (ICLR), 2020.

Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models
with masked transformers. arXiv preprint arXiv:2306.09305, 2023.

Xiangxin Zhou, Xiwei Cheng, Yuwei Yang, Yu Bao, Liang Wang, and Quanquan Gu. Decompopt:
Controllable and decomposed diffusion models for structure-based molecular optimization. arXiv
preprint arXiv:2403.13829, 2024a.

Xiangxin Zhou, Dongyu Xue, Ruizhe Chen, Zaixiang Zheng, Liang Wang, and Quanquan Gu.
Antigen-specific antibody design via direct energy-based preference optimization. arXiv preprint
arXiv:2403.16576, 2024b.

Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana Davuluri, and Han Liu. Dnabert-2: Effi-
cient foundation model and benchmark for multi-species genome. arXiv preprint arXiv:2306.15006,
2023.

Zhihan Zhou, Weimin Wu, Harrison Ho, Jiayi Wang, Lizhen Shi, Ramana V Davuluri, Zhong Wang,
and Han Liu. Dnabert-s: Learning species-aware dna embedding with genome foundation models.
ArXiv, 2024c.

Zhenyu Zhu, Francesco Locatello, and Volkan Cevher. Sample complexity bounds for score-matching:
Causal discovery and generative modeling. Advances in Neural Information Processing Systems
(NeurIPS), 36, 2023.

15


---Page Break---
Appendix

A More Discussions on Low-Dimensional Linear Latent Space
17

B
Notation Table
18

C Related Works
19

D Supplementary Theoretical Background
21
D.1
Diffusion Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
D.2
Proof of Lemma 2.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
D.3
Preliminaries: Strong Exponential Time Hypothesis (SETH) and Tensor Trick . . .
23

E
More Background and Auxiliary Lemmas: Universal Approximation of Transformers
via Piecewise Approximation
25
E.1
Piecewise-Constant Function Approximates Compact-Supported Continuous Function 25
E.2
Modified Transformer Approximates Piecewise-constant Function . . . . . . . . .
26
E.2.1
Quantization by Modified Feed-forward Layers . . . . . . . . . . . . . . .
27
E.2.2
Contextual Mapping by Modified Self-attention Layers . . . . . . . . . . .
28
E.2.3
Map to the Desired Output by Modified Feed-forward Layers
. . . . . . .
29
E.3
Standard Transformers Approximate Modified Transformers . . . . . . . . . . . .
29
E.4
All Together: Standard Transformers Approximate Compact-supported Continuous
Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29
E.5
Supplementary Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30
E.5.1
Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30
E.5.2
Proof of Lemma E.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32
E.5.3
Proof of Lemma E.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32
E.5.4
Proof of Lemma E.5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
37
E.5.5
Proof of Lemma E.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
39

F
Proofs of Section 3
40
F.1
Proof of Theorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
40
F.1.1
Auxiliary Lemmas for Theorem 3.1. . . . . . . . . . . . . . . . . . . . . .
40
F.1.2
Main Proof of Theorem 3.1
. . . . . . . . . . . . . . . . . . . . . . . . .
41
F.2
Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46
F.2.1
Auxiliary Lemmas for Theorem 3.2 . . . . . . . . . . . . . . . . . . . . .
46
F.2.2
Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47
F.3
Proof of Corollary 3.2.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52
F.3.1
Auxiliary Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52
F.3.2
Main Proof of Corollary 3.2.1 . . . . . . . . . . . . . . . . . . . . . . . .
54

G Proofs of Section 4
55
G.1 Auxiliary Theoretical Results for Theorem 4.1 . . . . . . . . . . . . . . . . . . . .
55
G.1.1
Low-Rank Decomposition of DiT Gradients . . . . . . . . . . . . . . . . .
55
G.1.2
Low-Rank Approximations of Building Blocks Part I: f(·), q(·), and c(·)
.
58
G.1.3
Low-Rank Approximations of Building Blocks Part II: p(·) . . . . . . . . .
59
G.2
Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
61

16


---Page Break---
A
More Discussions on Low-Dimensional Linear Latent Space

Our analysis is based on the low-dimensional linear latent space assumption (Assumption 2.1). Here
we further discuss this in light of our theoretical results

Our results are more general and extend beyond Assumption 2.1. In addition to the case where
d0 < D, our theoretical results apply to two other settings: d0 = D and d0 > D. Especially, for
d0 = D, our results still hold by setting B as the identity matrix ID. Namely, our results hold after
removing the linear subspace assumption.

• Statistically, for score approximation, score estimation, and distribution estimation, the upper
bounds depend on the dimension of the latent variable d0, other than d. A smaller d0 allows
for a reduced model size to achieve a specified approximation error compared to a larger one
(Theorem 3.1). Additionally, with a smaller d0, both score and distribution estimation errors are
reduced relative to scenarios with larger ones (Theorem 3.2 and Corollary 3.2.1).
• Computationally, smaller d0 benefits the provably efficient criteria (Proposition 4.1, almost-linear
time algorithms for forward inference (Proposition 4.2) and backward computation (Theorem 4.1).

17


---Page Break---
B
Notation Table

We summarize our notations in the following table for easy reference.

Table 1: Mathematical Notations and Symbols
Symbol
Description

∥z∥2
Euclidean norm, where z is a vector
∥z∥∞
Infinite norm, where z is a vector
∥Z∥2
2-norm, where Z is a matrix
∥Z∥op
Operator norm, where Z is a matrix
∥Z∥F
Frobenius norm, where Z is a matrix
∥Z∥p,q
p, q-norm, where Z is a matrix
∥f(x)∥L2
L2-norm, where f is a function
∥f(x)∥L2(P )
L2(P)-norm, where f is a function and P is a distribution
∥f(·)∥Lip
Lipschitz-norm, where f is a function
f♯P
Pushforward measure, where f is a function and P is a distribution

n
Sample size
x
Data point in original data space, x ∈RD

h
Latent variable in low-dimensional subspace, h ∈Rd0
ph
The destiny function of h
B
The matrix with orthonormal columns to transform h to x, where B ∈RD×d0
x
Perturbed data variable at t > 0
h
h = B⊤x

T
Stopping time in the forward process of Diffusion model
T0
Stopping time in the backward process of Diffusion model
µ
Discretized step size in backward process
pt(·)
The density function of x for at time t
ph
t (·)
The density function of h at time t
ψ
(Conditional) Gaussian density function

d
Input dimension of each token in the transformer network of DiT
L
Token length in the transformer network of DiT
X
Sequence input of transformer network in DiT, where X ∈Rd×L

E
Position encoding, where E ∈Rd×L

R(·)
Reshape layer in DiT, R(·) : Rd0 →Rd×L

WB
The orthonormal matrix to approximate B, where WB ∈RD×d0

18


---Page Break---
C
Related Works

Organization.
In the following, we first discuss recent developments in DiTs. Then, we discuss the
main technique of our statistical results: the universality (universal approximation) of transformer.
Next, we discuss recent theoretical developments in diffusion generative models. Lastly, we discuss
other aspects of transformer in foundation models beyond diffusion models.

Diffusion Transformers.
Diffusion [Ho et al., 2020] and score-based generative models [Song
and Ermon, 2019] have been particularly successful as generative models of images, video and
biomedical data [Nichol et al., 2021, Ramesh et al., 2022, Liu et al., 2024, Zhou et al., 2024a,b,
Wang et al., 2024a,b]. Recently, transformer-based diffusion models have garnered significant
attention in research. The U-ViT model [Bao et al., 2022] incorporates transformer blocks into a
U-net architecture, treating all inputs as tokens. In contrast, DiT [Peebles and Xie, 2023] utilizes a
straightforward, non-hierarchical transformer structure. Empirically, diffusion transformers (DiTs)
[Peebles and Xie, 2023] have emerged as a significant advancement (e.g., SoRA [OpenAI, 2024,
Liu et al., 2024] from OpenAI), effectively combining the strengths of transformer architectures and
diffusion-based approaches. Models like MDT [Gao et al., 2023a] and MaskDiT [Zheng et al., 2023]
improve the training efficiency of DiT by applying a masking strategy.

Universality and Memory Capacity of Transformers.
The universality of transformers refers to
their ability to serve as universal approximators. This means that transformers theoretically models
any sequence-to-sequence function to a desired degree of accuracy. Yun et al. [2020] establish that
transformers can universally approximate sequence-to-sequence functions by stacking numerous
layers of feed-forward functions and self-attention functions. In a different approach, Jiang and Li
[2023] affirm the universality of transformers by utilizing the Kolmogorov-Albert representation
Theorem. Most recently, Kajitsuka and Sato [2023] show that transformers with one self-attention
layer is a universal approximator.

The memory capacity of a transformer is a practical measure to test the theoretical results of the
transformer’s universality, by ensuring the model can handle necessary context and dependencies. By
memory capacity, we refer to the minimal set of parameters such that the model (i.e., transformer)
approximates all input-output pairs in the training dataset with a bounded error. Several works address
the memory capacity of transformers. Kim et al. [2022] show that transformers with eO(d+L+
√

NL)
parameters are sufficient to memorize N length-L and dimension-d sequence-to-sequence data points
by constructing a contextual mapping with O(L) attention layers. Mahdavi et al. [2023] show that a
multi-head-attention with h heads is able to memorize O(hL) examples under a linear independence
data assumption. Kajitsuka and Sato [2023] show that a single layer transformer with O(NLd + d2)
parameters is able to memorize N length-L and dimension-d sequence-to-sequence data points by
utilizing the connection between the softmax function and Boltzmann operator. Anonymous [2024a],
Wang et al. [2023] extend the results of [Kajitsuka and Sato, 2023, Yun et al., 2020] to prompt
tuning and discuss the memorization of the data sequences. Another line of research establishes a
different kind of memory capacity for transformers by connecting transformer attention with dense
associative memory models (modern Hopfield models) [Hu et al., 2024a,b,d, 2023, Wu et al., 2024a,b,
Ramsauer et al., 2020]. Notably, they define memory capacity as the smallest number of (length-L and
dimension-d) data points the model (transformer attention) is able to store and derive exponential-in-d
high-probability capacity lower bounds. In particular, Hu et al. [2024d] report a tight exponential
scaling of capacity with feature dimension from the perspective of spherical codes.

Our work is motivated by and builds on [Yun et al., 2020] to bridge the transformer’s function
approximation ability with data distribution estimation. While we do not address the memorization of
DiTs (or diffusion models in general), recent studies on dense associative models suggest viewing pre-
trained diffusion generative models as associative memory models [Achilli et al., 2024, Ambrogioni,
2023, Hoover et al., 2023]. We plan to explore this aspect in future work.

Theories of Diffusion Models.
In addition to empirical success, there has been several theoretical
analysis about diffusion models [Chen et al., 2024b, Tang and Zhao, 2024]. Chen et al. [2023] studies
score approximation, estimation, and distribution recovery of U-Net based diffusion models. Benton
et al. [2024] provide convergence bounds linear in data dimensions, assuming accurate score function
approximation. Zhu et al. [2023], Wibisono et al. [2024] provide statistical sample complexity
bounds for score-matching under the similar assumptions. Oko et al. [2023] analyze the distribution

19


---Page Break---
estimation under the assumption that the initial density is supported on [−1, 1]D and smooth in the
boundary.

Among these works, our work is built on and closest to [Chen et al., 2023], as both assume the data
has a low-dimensional structure. However, our work differs in three key aspects. First, beyond the
simple ReLU networks considered in [Chen et al., 2023], we provide the first score approximation
analysis for DiTs with a transformer-based score estimator. Second, our work is the first to provide
the statistical rates of DiTs (score and distribution estimation) based on transformer universality
[Yun et al., 2020] and norm-based converging number bound [Edelman et al., 2022], supporting the
practical success of DiTs [Esser et al., 2024, Ma et al., 2024]. Lastly, our work provides the first
comprehensive analysis of the computational limits and all possible efficient DiT algorithms/methods
for both forward inference and backward training. This offers timely insights into the empirical
computational inefficiency of DiTs [Liu et al., 2024] and guidance for future DiT architectures.

Transformers in Foundation Models: Transformer-Based Pretrained Models.
Transformer-
based pretrained models utilize attention mechanisms to process sequential data, enabling the learning
of contextual relationships for tasks like natural language understanding and generation. These models
encompass three types: encoder-based, decoder-based, and diffusion transformers. Encoder-based
transformers, such as DNABERT [Zhou et al., 2024c, 2023, Ji et al., 2021], employ bidirectional
attention to extract feature representations DNABERT shows great potential to capture complex
patterns of genome sequences and improve tasks such as gene prediction. Decoder-based transformers
generate output sequences from encoded information using unidirectional attention, such as ChatGPT
[Radford et al., 2019, Floridi and Chiriatti, 2020, Brown et al., 2020] for natural language. The
diffusion transformers generate a sequence toward a target distribution, such as SoRA [Liu et al.,
2024] and Videofusion [Luo et al., 2023] for video generation and DecompDiff [Guan et al., 2024]
for drug design. In our paper, we present an early exploration of the statistical and computational
limits of diffusion transformer models.

20


---Page Break---
D
Supplementary Theoretical Background

In this section, we provide some further background. We show the details about the forward and
backward process in Diffusion Models in Appendix D.1. Besides, we give the details of the proof
about the score decomposition in Appendix D.2.

D.1
Diffusion Models

Forward Process.
Diffusion models gradually add noise to the original data in the forward process.
We describe the forward process as the following SDE

dxt = −1

2w(t)xtdt +
p

w(t)dWt, xt ∈RD,
(D.1)

where x0 ∼P0, (Wt)t≥0 is a standard Brownian motion, and w(t) > 0 is a nondecreasing weighting
function. Let Pt and pt denote the marginal distribution and destiny of xt. The conditional distribution
P(xt|x0) follows N(β(t)x0, σ(t)ID), where β(t) = exp

−
R t
0 w(s)ds/2

and σ(t) = 1 −β2(t).

In practice, (D.1) terminates at a large enough T such that PT is close to N(0, ID).

Backward Process.
We obtain the backward process yt := xT −t by reversing (D.1). The backward
process satisfies

dyt =
1

2w(T −t)yt + w(T −t)∇log pT −t(yt)

dt +
p

w(T −t)dW t,
(D.2)

where the score function ∇log pt(·) is the gradient of log probability density function of xt, and
W t is a reversed Brownian motion. However, ∇log pt(·) and PT are both unknown in (D.2). To
resolve this, we use a score estimator sW (·, t) to replace ∇log pt(·), where sW (·, t) is usually a
neural network with parameters W. Secondly, we replace PT by the standard Gaussian distribution.
Consequently, we obtain the following SDE

deyt =
1

2w(T −t)eyt + w(T −t)sW (eyt, T −t)

dt +
p

w(T −t)dW t, ey0 ∼N(0, ID). (D.3)

In practice, we use discrete schemes of (D.3) to generate data, following [Song and Ermon, 2019].
We use µ > 0 to denote the discretization step size. For t ∈[kµ, (k + 1)µ], we have

deyµ
t =
1

2w(T −t)eyµ
kµ + w(T −t)sW (eyµ
kµ, T −kµ)

dt +
p

w(T −t)dW t.
(D.4)

D.2
Proof of Lemma 2.1

Here we restate the proof of [Chen et al., 2023, Lemma 1] for completeness.

Proof. Recall x = Bh by Assumption 2.1 with x ∈RD, B ∈RD×d0 and h ∈Rd0.

By the forward process (D.1), we have

pt(x) =
Z
ψt(x | Bh)ph(h)dh,
(D.5)

where

ψt(x | Bh) = [2πh(t)]−D/2 exp

 

−∥β(t)Bh −x∥2
2
2σ(t)

!

,
(D.6)

is the Gaussian transition kernel.

21


---Page Break---
Then we write the score function as

∇log pt(x) = ∇pt(x)

pt(x)
(D.7)

= ∇
R
ψt(x | Bh)ph(h)dh
R
ψt(x | Bh)ph(h)dh

 
By pluging in pt(x)

=

R
∇ψt(x | Bh)ph(h)dh
R
ψt(x | Bh)ph(h)dh ,
 
By interchanging R
with ∇

where the last equality holds since ψt(x | Bh) is continuously differentiable in x.

Plugging (D.6) into (D.7), we have

∇log pt(x)

=
[2πh(t)]−D/2
R
ψt(x | Bh)ph(h)dh

Z
1
σ(t) (β(t)Bh −x) exp

 

−∥β(t)Bh −x∥2
2
2σ(t)

!

ph(h)dh.

We then decompose above score function by projecting of x into Span(B), i.e., replacing −x with
−BB⊤x −(ID −BB⊤)x:

∇log pt(x)

=
[2πh(t)]−D/2
R
ψt(x | Bh)ph(h)dh

·
Z
1
σ(t)

"
 
β(t)Bh −BB⊤x

−
 
ID −BB⊤

x

#

exp

 

−∥β(t)Bh −x∥2
2
2σ(t)

!

ph(h)dh.

Absorbing the factor of [2πh(t)]−D/2 into the Gaussian kernel ψt(x | Bh), we have

∇log pt(x)

=
[2πh(t)]−D/2
R
ψt(x | Bh)ph(h)dh

Z
1
σ(t)
 
β(t)Bh −BB⊤x

exp

 

−∥β(t)Bh −x∥2
2
2σ(t)

!

ph(h)dh

−
1
R
ψt(x|Bh)ph(h)dh

 1

σ(t)
 
ID −BB⊤

x
 Z
ψt(x | Bh)ph(h)dh

=
1
R
ψt(x | Bh)ph(h)dh

Z
1
σ(t)
 
β(t)Bh −BB⊤x

ψt(x | Bh)ph(h)dh
|
{z
}
:=s+

−1

σ(t)
 
ID −BB⊤

x
|
{z
}
:=s−

.

To further simplify s+, we decompose ψt(x | Bh) as

ψt(x | Bh)

= [2πh(t)]−D/2 exp

−
1
2σ(t)∥β(t)Bh −x∥2
2



= [2πh(t)]−D/2 exp

−
1
2σ(t)

β(t)Bh −BB⊤x −
 
ID −BB⊤

x
2

2



= [2πh(t)]−D/2 exp

 

−
1
2σ(t)

β(t)Bh −BB⊤x
2

2 +
 
ID −BB⊤

x
2

2

−2(B(β(t)h −B⊤x))⊤(ID −BB⊤)x
!

22


---Page Break---
= [2πh(t)]−D/2 exp

−
1
2σ(t)

β(t)Bh −BB⊤x
2

2 +
 
ID −BB⊤

x
2

2



 
B(β(t)h −B⊤x) is in Span(B) while (ID −BB⊤)x is orthogonal to Span(B)

= [2πh(t)]−d0/2 exp

 

−

β(t)h −B⊤x
2
2
2σ(t)

!

|
{z
}
:=ψt(B⊤x|h)

· [2πh(t)]−(D−d0)/2 exp

 

−

 
ID −BB⊤

x
2
2
2σ(t)

!

|
{z
}
:=ψt((ID−BB⊤)x)

,

 
since B has orthonormal columns

where both ψt
 
B⊤x | h

and ψt
 
(ID −BB⊤)x

are Gaussian.

Plugging ψt(x | Bh) = ψt
 
B⊤x | h

ψt
 
(ID −BB⊤)x

into s+, we obtain

s+(x, t) = C
Z
1
σ(t)
 
β(t)Bh −BB⊤x

ψt(B⊤x | h)ψt((ID −BB⊤)x)ph(h)dh

= Cψt((ID −BB⊤)x)
Z
1
σ(t)
 
β(t)Bh −BB⊤x

ψt(B⊤x | h)ph(h)dh

=
1
R
ψt(B⊤x | h)ph(h)dh

Z
1
σ(t)
 
β(t)Bh −BB⊤x

ψt(B⊤x | h)ph(h)dh,

where C := [ψt((ID −BB⊤)x)
R
ψt(B⊤x | h)ph(h)dh]−1.

Notably, s+ depends only on the projected data B⊤x. Therefore, we are able to replace s+(x, t) with
s+(B⊤x, t). The benefit is that the dimension d0 of the first input in s+(B⊤x, t) is much smaller.

Lastly, by denoting h = B⊤x such that ∇hψt(h | h) = (β(t)h −h)ψt(h | h)/σ(t), we arrive at

s+(B⊤x, t) = B
Z
∇hψt(h | h)ph(h)
R
ψt(h | h)ph(h)dhdh

= B∇log ph
t (B⊤x).
 
ph
t (h) := R
ψt(h|h)ph(h)dh

This completes the proof.

D.3
Preliminaries: Strong Exponential Time Hypothesis (SETH) and Tensor Trick

Here we present the ideas we built upon for Section 4.

Strong Exponential Time Hypothesis (SETH).
Impagliazzo and Paturi [2001] introduce the
Strong Exponential Time Hypothesis (SETH) as a stronger form of the P ̸= NP conjecture. It suggests
that our current best SAT algorithms are optimal and is a popular conjecture for proving fine-grained
lower bounds for a wide variety of algorithmic problems [Cygan et al., 2016, Williams, 2018].

Hypothesis 1 (SETH). For every ϵ > 0, there is a positive integer k ≥3 such that k-SAT on formulas
with n variables cannot be solved in O(2(1−ϵ)n) time, even by a randomized algorithm.

Tensor Trick for Computing Gradients.
The tensor trick [Diao et al., 2019, 2018] is an instrument
to compute complicated gradients in a clean and tractable fashion. We start with some definitions.

Definition D.1 (Vectorization). For any matrix X ∈RL×d, we define X := vec (X) ∈RLd such
that Xi,j = X(i−1)d+j for all i ∈[L] and j ∈[d].

Definition D.2 (Matrixization). For any vector X ∈RLd, we define mat(X) = X such that
Xi,j = mat(X) := X(i−1)d+j for all i ∈[L] and j ∈[d], namely mat(·) = vec−1(·).

Definition D.3 (Kronecker Product). Let A ∈RLa×da and B ∈RLb×db. We define the Kronecker
product of A and B as A ⊗B ∈RLaLb×dadb such that (A ⊗B)(ia−1)Lb+ib,(ja−1)db+jb, is equal to
Aia,jaBib,jb with ia ∈[La], ja ∈[da], ib ∈[Lb], jb ∈[db].

23


---Page Break---
Definition D.4 (Sub-Block of a Tensor). For any A ∈RLa×da and B ∈RLb×db, let A := A ⊗B ∈
RLaLb×dadb. For any j ∈[La], we define Aj ∈RLb×dadb be the j-th Lb × dadb sub-block of A.

Lemma D.1 (Tensor Trick [Diao et al., 2019, 2018]). For any A ∈RLa×da, B ∈RLb×db and
X ∈Rda×db, it holds vec
 
A⊤XB

= (A⊤⊗B⊤)X ∈RLaLb.

To showcase the tensor trick, let’s consider a (single data point) attention following [Gao et al.,
2023b,c]. Setting D := diag
 
exp
 
XTW T
KWQX

1L

and W := WKW T
Q ∈Rd×d, we have

L0 :=
 WV
|{z}
d×d
X
|{z}
∈Rd×L
D−1
|{z}
∈RL×L
exp

XTWX
	

|
{z
}
∈RL×L
−
Y
|{z}
∈Rd×L

2
2.
(D.8)

Proposition D.1 (Definition 4.7 of [Gao et al., 2023b]). By Definition D.3 and Definition D.4,
we identify Dj,j
:=
D
exp

Aj W


, 1L
E
∈R for all j
∈[L], with A := X ⊗X
∈

RL2×d2 and W
∈
Rd2.
Therefore, for each j
∈
[L] and i
∈
[d], it holds L0
=
PL
j=1
Pd
i=1
1
2
D
D−1
j,j exp

Aj W


, XWV [·, i]
E
−Yj,i

2
.

The elegance of Proposition D.1 emerges when we vectorize the weights into vectors W, W V , making
the gradient computations (e.g., dL0/W and dL0/W V ) more tractable by avoiding complex matrix or
tensor derivatives. This approach systematically simplifies the handling of chain-rule terms in the
gradient computation of losses like L0.

Fine-Grained Complexity for Transformer.
Many recent works also utilize similar techniques
from fine-grained complexity to analyze transformer architectures. Alman and Song [2023, 2024b],
Liang et al. [2024d], Alman and Song [2024a] explore the computational feasibility of inference
and training for standard softmax and tensor attention. Liang et al. [2024c] extend the single-
layer training results from [Alman and Song, 2024b] to deep transformer models. [Liang et al.,
2024a] extend [Alman and Song, 2024b] to provide a fast attention gradient approximation based on
Fourier transform. [Liang et al., 2024b] extend [Alman and Song, 2024b] to sparse attention matrix.
Anonymous [2024a] study the computational limits of inference and training in prompt-tuning for
pretrained transformers. Hu et al. [2024c] study the computational limits of LoRA [Hu et al., 2021]
in transformers, identifying norm-bound conditions for efficient LoRA training and proving the
existence of nearly linear-time LoRA algorithms.

Our work is closest to [Alman and Song, 2024b, 2023]. Our forward inference computational results
build on [Alman and Song, 2023]. Our backward training computational results are related to [Alman
and Song, 2024b] but include additional analysis on reshaping and latent embedding.

24


---Page Break---
E
More Background and Auxiliary Lemmas: Universal Approximation of
Transformers via Piecewise Approximation

Here, we review the universal approximation of transformers following [Yun et al., 2020].

Our goal is to reproduce the results of [Yun et al., 2020] and use or modify them as auxiliary lemmas
for proofs of Section 3 (i.e., Appendix F.)

We start with their central result and prove it in the rest of the section.

Lemma E.1 (Universal Approximation of Transformers, Theorem 3 of [Yun et al., 2020]). Let ϵ > 0.
For any given compact-supported continuous function f : Rd×L →Rd×L, there exists a transformer
network fT ∈T 2,1,4
p
, such that

Z
∥fT (X) −f(X)∥2
F dX
1/2
≤ϵ.

Proof Overview.
We use the following proof strategy:

• Step 1. We show that the piecewise-constant function is able to approximate compact-supported
continuous function in Appendix E.1.

• Step 2. We define modified self-attention and feed-forward layers to construct the modified
transformer. We show that the modified transformer is able to approximate piecewise-constant
function in Appendix E.2.

• Step 3. We show that the modified transformer is able to approximate the normal transformer in
Appendix E.3.

We provide details of Step 1. in Appendix E.1, Step 2. in Appendix E.2, and Step 3. in Appendix E.3.
Then we summarize our results in Appendix E.4.

E.1
Piecewise-Constant Function Approximates Compact-Supported Continuous Function

In this subsection, we show that the piecewise-constant function is able to approximate compact-
supported continuous function.

We start with the definition of the compact-supported continuous functions of interest.

Assumption E.1. Without loss of generality, we assume that the target function in discussion is
supported on [0, 1]d×L. We denote the set of [0, 1]d×L-supported continuous functions as F.

We introduce the notion of grid and cube for the compact support [0, 1]d×L.

Definition E.1 (Grid and Cube with Width δ). Given a grid width δ, let Gδ := {0, δ, . . . , 1 −δ}d×L

denote the set of grids within [0, 1]d×L. For a grid point G = (Gj∈[d],k∈[L]) ∈Gδ, we denote its
associated cube as

SG := ⊗d
j=1 ⊗L
k=1 [Gj,k, Gj,k + δ) ⊂[0, 1]d×L.

Each cube SG represents a hyper rectangular in the multi-dimensional space [0, 1]d×L, constructed to
discretize the space into smaller subspaces.

We introduce the notion of piecewise-constant fucntion class w.r.t. the [0, 1]d×L-supported continuous
function class F.

Definition E.2 (Piecewise-Constant Function Class). Let fδ denote the piesewise constant function of
grid width δ, and 1{·} denote the indicator function. For each G ∈Gδ, and any matrix AG ∈Rd×L,
we define the piecewise-constant function class as

F(δ) :=
n
fδ : X →
X

G∈Gδ AG · 1{X ∈SG}, AG ∈Rd×Lo
.
(E.1)

25


---Page Break---
We recall that for a given sequence-to-sequence function f,

∥f∥L2 :=
 Z
∥f(X)∥2
F dX
1/2
.

We approximate the compact-supported function with a piecewise-constant function in the next
lemma.

Lemma E.2. (Lemma 8 of [Yun et al., 2020]) For any given f ∈F and ϵ/3 > 0, we can find a
δ⋆> 0, such that there exists a fδ⋆∈F(δ⋆) satisfying ∥f −fδ⋆∥L2 ≤ϵ/3.

Proof. See Appendix E.5.2 for a detailed proof.

E.2
Modified Transformer Approximates Piecewise-constant Function

In this subsection, we define modified self-attention and feed-forward layers to construct the modified
transformers. We use the modified transformers to approximate the piecewise-constant function.

Definition E.3 (Modified Transformer Networks). The modified transformer network T r,m,l
p
includes
two modifications to the standard transformer network T r,m,l
p
:

• Modified attention layer: Replace Softmax operator with Hardmax operator σH(·).
• Modified feed-forward layer: Replace ReLU(·) with an activation function ζ ∈Ψ. Here, Ψ
denotes the set of all piecewise linear functions with at most three pieces and at least one constant.

We approximate F(δ) with this modified transformer networks T r,m,l
p
.

Lemma E.3 (Modified from Proposition 4 of [Yun et al., 2020]). For each fδ ∈F(δ), there exists a
fT ,c ∈T 2,1,1
p
such that ∥fδ −fT ,c∥L2 = O(δd/2).

Proof Sketch. Given δ, and for any grid G ∈Gδ, we have a grid set Gδ and the cube SG.

Our proof follows two steps:

• Quantization. For all X ∈Rd×L, we quantize it to a finite set:

– If X ∈SG ⊂[0, 1]d×L, we quantize it to the element G ∈Gδ.
– If X /∈[0, 1]d×L, we quantize it to an element out of Gδ.

• Mapping. For any G ∈Gδ, we map it to the desired output AG.

For Quantization, we achieve this by a series of modified feed-forward layers. We show this in
Appendix E.2.1.

For Mapping, we follow two steps:

• For any G ̸= G′ ∈Gδ, we use a “contextual mapping” qc(·) (defined as Definition E.4). The
mapping maps all the elements in qc(G) and qc(G′) to different values. Then, we use a series of
modified self-attention layers to achieve “contextual mapping”. We show this in Appendix E.2.2.

Definition E.4 (Contextual Mapping). Consider a finite set Gδ ∈Rd×L. A map qc : Gδ →R1×L
defines a contextual mapping if the map satisfies the following:

– For any G ∈Gδ, the entries in qc(G) are all distinct.
– For any G ̸= G′ ∈Gδ, all entries of qc(G) and qc(G′) are distinct.

• For any G ∈Gδ, we use a series of modified feed-forward layers to map qc(G) to AG. We show
this in Appendix E.2.3.

26


---Page Break---
Remark E.1. Our proof differs from [Yun et al., 2020] in one aspect: Although [Yun et al., 2020,
Proposition 4] outlines a proof for transformer networks without positional encoding and sketches
the proof for networks with it, we provide a detailed proof for the latter to support our proof.

E.2.1
Quantization by Modified Feed-forward Layers

We use a series of modified feed-forward layers in T r,m,l
p
to quantize an input X ∈Rd×L to an
element G of the following grid:

{−J, 0, δ, . . . , 1 −δ}d×L,

where J > L > 0 is a large number to be determined later. We achieve this via two steps.

• Step 1: Map the element out of [0, 1) to −J.
We use ei to represent the standard unit vector where the i-th element is 1. For the i-th row of X,
we define the following feed-forward layer to achieve our aim.

Definition E.5 (Feed-forward Layer 1). The vector ei acts as the weight parameters, and ζ1(·) acts
as the activation function in the feed-forward layer

X →X + eiζ1(e⊤
i X), ζ1(t) =
−t −J,
for t < 0 or t ≥1,
0,
otherwise.
(E.2)

We take i = 1 as an example to give the specific calculation. Let X = (xi,j)d×L, then we have

FF(X) = X +







1
0
...
0





(ζ1(x1,1)
ζ1(x1,2)
· · ·
ζ1(x1,L))

= X +







ζ1(x1,1)
ζ1(x1,2)
· · ·
ζ1(x1,L)
0
0
· · ·
0
...
...
...
...
0
0
· · ·
0





.

In the first row of X, the above layer transforms the element that is out of [0, 1) to −J.
We stack the above layers together for i = 1, 2, . . . , d. If the element of X is out of [0, 1), the
series of layers maps it to J.
• Step 2: Map the element in [0, 1) to {0, δ, 2δ, . . . , 1 −δ}.
For the i-th row of X, we take k = 0, 1, . . . , 1/δ −1 respectively. We define the following layer.

Definition E.6 (Feed-forward Layer 2). The vector ei acts as the weight parameters and ζ2(·) acts
as the activation function in the feed-forward layer

X →X + eiζ2(e⊤
i X −kδ1⊤
n ), ζ2(t) =
0,
t < 0 or t ≥δ,
−t,
0 ≤t < δ.
(E.3)

We take i = 1 and k = 1 as an example. We give the following specific calculation

FF(X) = X +







1
0
...
0





(ζ2(x1,1 −δ)
ζ2(x1,2 −δ)
· · ·
ζ2(x1,L −δ))

= X +







ζ2(x1,1 −δ)
ζ2(x1,2 −δ)
· · ·
ζ2(x1,L −δ)
0
0
· · ·
0
...
...
...
...
0
0
· · ·
0





.

27


---Page Break---
In the first row of X, the above layer transforms the element in [δ, 2δ] to δ.
We stack the above layers together for i = 1, 2, . . . , d and k = 0, 1, . . . , 1/δ −1. If the element of
X is in [kδ, (k + 1)δ], the series layers maps it to kδ.

Combining the above two parts, we achieve our goal with d/δ + d feed-forward layers. We denote
the d/δ + d series layers as fT ,c1.

E.2.2
Contextual Mapping by Modified Self-attention Layers

In our attention layers, we use the following positional encoding E ∈Rd×L

E =







0
1
2
· · ·
L −1
0
1
2
· · ·
L −1
...
...
...
...
0
1
2
· · ·
L −1





.
(E.4)

According to Appendix E.2.1, the output of fT ,c1 is in the grid {−J, 0, δ, . . . , 1 −δ}d×L. For any X
in this grid, the first column of X + E is in

{−J, 0, δ, . . . , 1 −δ}d,

and the second column is in

{−J + 1, 1, 1 + δ, . . . , 2 −δ}d.

The results are similar in the other columns.

For i = 0, 1, . . . , L −1, we use the following notation:

[i : δ : i + 1 −δ]J := {i −J, i, i + δ, . . . , i + 1 −δ}.

Then, we define the grid G+
δ as the following.

Definition E.7 (Grid G+
δ ). We add E to all the grid points in Gδ to generate the modified grid G+
δ ,
defined as follows:

G+
δ := [0 : δ : 1 −δ]d
J × [1 : δ : 2 −δ]d
J × · · · × [L −1 : δ : L −δ]d
J.

Next, we show that the modified attention layer computes contextual mapping (Definition E.4) for
G+
δ . For i = 1, 2, . . . , L −1, we use the following notation:

[i : δ : i + 1 −δ] := {i, i + δ, i + 2δ, . . . , i + 1 −δ}.

Lemma E.4 (Modified from Lemma 6 of [Yun et al., 2020]). We consider the following subset of
G+
δ :

eGδ := [0 : δ : 1 −δ]d × [1 : δ : 2 −δ]d × · · · × [L −1 : δ : L −δ]d
|
{z
}
L

.

Assume that L ≥2 and δ−1 ≥2. Then, there exist a function fT ,c2 : Rd×L →Rd×L composed of
δ−d + 1 modified attention layers (Definition E.3), a vector u ∈Rd, and two constants tl, tr ∈R
(0 < tl < tr), such that qc(G) := u⊤fT ,c2(G), G ∈G+
δ satisfies the following properties:

1. For any G ∈eGδ, all the entries of qc(G) are distinct.

2. For any different G, G′ ∈eGδ, all the entries of qc(G), qc(G′) are distinct.

3. For any G ∈eGδ, all the entries of qc(G) are in [tl, tr].

4. For any G ∈G+
δ \ eGδ, all the entries of qc(G) are outside [tl, tr].

28


---Page Break---
Proof. See Appendix E.5.3 for a detailed proof.

Remark E.2. Our proof differs from [Yun et al., 2020] in one aspect: The original [Yun et al., 2020,
Lemma 6] does not include positional encoding (E.4). Although Yun et al. [2020] sketches the proof
for networks with (E.4) in the attention layer input, we detail the proof.

E.2.3
Map to the Desired Output by Modified Feed-forward Layers

Next, we show that a series of feed-forward layers map the output of modified attention layers fT ,c2
to the desired output of function fδ⋆.

Lemma E.5 (Lemma 7 of [Yun et al., 2020]). There exists a function fT ,c3 : Rd×L →Rd×L

composed of O(L(1/δ)dL/L!) modified feed-forward layers, such that

fT ,c3 ◦fT ,c2(G) =

(
AG
if G ∈eGδ,
0d×L
if G ∈G+
δ \ eGδ.

Proof. See Appendix E.5.4 for a detailed proof.

In conclusion, we have the following lemma for the required number of layers in the modified
transformer.

Lemma E.6 (Total Number of Layers). From the proof of Lemma E.3, if we want to achieve a
approximation error O(δd/2) by the modified transformer, we need O(δ−1) modified feed-forward
layers in fT ,c1, O(δ−d) modified self-attention layers in fT ,c2, and O(δ−dL) modified feed-forward
layers in fT ,c3.

Proof. By the proof of Lemma E.3, we complete the proof.

E.3
Standard Transformers Approximate Modified Transformers

In this subsection, we show that standard neural network layers are able to approximate the modified
self-attention layers and the modified feed-forward layers (Definition E.3). We have the following
Lemma E.7.

Lemma E.7 (Lemma 9 of [Yun et al., 2020]). For each fT ,c ∈T 2,1,1
p
and any ϵ > 0, there exists
fT ∈T 2,1,4
p
such that ∥fT −fT ,c∥L2 ≤ϵ/3.

Proof. See Appendix E.5.5 for a detailed proof.

E.4
All Together: Standard Transformers Approximate Compact-supported Continuous
Functions

We summarize the results of Lemmas E.2, E.3 and E.7. Then we prove Lemma E.1.

Furthermore, to achieve the ϵ approximation error in Lemma E.1, we take δ = O(ϵ2/d) in Lemma E.3.

29


---Page Break---
E.5
Supplementary Proofs

We first present two preliminary concepts: selective shift operation and bijective column ID mapping
in Appendix E.5.1.

Then we show

• Proof of Lemma E.2 in Appendix E.5.2

• Proof of Lemma E.4 in Appendix E.5.3

• Proof of Lemma E.5 in Appendix E.5.4

• Proof of Lemma E.7 in Appendix E.5.5

E.5.1
Preliminaries

Here, we give the definition of two preliminary concepts: selective shift operation and bijective
column ID mapping.

Selective Shift Operation.
This operation refers to shifting certain entries of the input selectively.

To achieve this, we consider the following function ξ(·; ·) : Rd×L →Rd×L

ξ(X; bQ) = e1u⊤XσH

(u⊤X)⊤(u⊤X −bQ1⊤
n )

,
(E.5)

where X ∈Rd×L, e1 = (1, 0, 0, · · · , 0)⊤∈Rd, and bQ ∈R. u ∈Rd is a vector to be determined.

To see the output, we consider the j-th column of u⊤XσH

(u⊤X)⊤(u⊤X −bQ1⊤
n )

:

• If u⊤X:,j > bQ, it calculates argmax of u⊤X;

• If u⊤X:,j < bQ, it calculates argmin of u⊤X.

All rows of ξ(X; bQ) except the first row are zero. We consider the j-th entry of the first row in
ξ(X; bQ), which is denoted as ξ(X; bQ)1,j. Then for all j ∈[L], we have

ξ(X; bQ)1,j = u⊤XσH

(u⊤X)⊤(u⊤X:,j −bQ)

=
maxk u⊤X:,k
if u⊤X:,j > bQ,
mink u⊤X:,k
if u⊤X:,j < bQ.

From this observation, we define a function parametrized by bQ and b′
Q (with bQ < b′
Q)

ξ(X; bQ, b′
Q) := ξ(X; bQ) −ξ(X; b′
Q).
(E.6)

Then we have

ξ(X; bQ, b′
Q)1,j =
maxk u⊤X:,k −mink u⊤X:,k,
if bQ < u⊤X:,j < b′
Q,
0,
others.

We define an attention layer of the form X →X + ξ(X; bQ, b′
Q). For any column X:,j, if bQ <
u⊤X:,j < b′
Q, its first coordinate X1,j is shifted up by maxk u⊤X:,k −mink u⊤X:,k, while all the
other coordinates stay untouched. We call this the selective shift operation because we can choose bQ
and b′
Q to shift certain entries of the input selectively.

Bijective Column ID Mapping.
We consider the input G ∈G+
δ (Definition E.7). We use

J = L + 3Lδ−dL, and u = (1, δ−1, δ−2, . . . , δ−d+1).
(E.7)

For any j ∈[L], we have the following two conclusions:

30


---Page Break---
• If Gi,j ≥0 for all i ∈[d], i.e., G:,j ∈[j −1 : δ : j −δ]d, then we have

u⊤G:,j ∈

δj : δ : δj + δ−d+1 −δ

, where δj = (j −1) ·
δ −δ−d+1

δ −1


.
(E.8)

The mapping G:,j →u⊤G:,j maps the elements in [j −1 : δ : j −δ]d to

δj : δ : δj + δ−d+1 −δ

.
This is a bijection.
• If there exists i ∈[d] such that Gi,j = −J + j, then

u⊤G:,j ≤−3Lδ−dL + (j −1) ·
δ−d+1 −δ

1 −δ


+ δ−d+1 < 0.
(E.9)

We say that u⊤G:,j gives the “column ID” for each possible value of G:,j ∈[j −1 : δ : j −δ]d.
Remark E.3 (Illustration of Bijection Properity). For the bijection property, we give the following
illustration. Let G:j = (g1j, g2j, · · · , gdj)⊤and G:j = (g1j, g2j, · · · , gdj)⊤. If u⊤G:j = u⊤G:j
and G:j ̸= G:j, we deduce

(g1j −g1j) + δ−1(g2j −g2j) + · · · + δ−d+1(gdj −gdj) = 0.
(E.10)

Because G:j ̸= G:j, then there exists a k (k < d), such that gkj ̸= gkj and gij = gij(i > k). We
have
δ−k+1(gkj −gkj)
 ≥δ−k+2.

However,
(g1j −g1j) + · · · + δ−k+2(gk−1,j −gk−1,j)


≤|g1j −g1j| + · · · +
δ−k+2(gk−1,j −gk−1,j)


≤(1 −δ) + · · · + δ−k+2(1 −δ)

< δ−k+2.

This contradicts with (E.10). Thus we prove the property of bijection.

31


---Page Break---
E.5.2
Proof of Lemma E.2

Proof of Lemma E.2. We restate the proof from [Yun et al., 2020] for completeness.

By the nature of the compact-supported continuous function, f is uniformly continuous.

Because ∥·∥∞is equivalent to ∥·∥F when the number of entries are finite, we have the following by
the definition of uniform continuity.

For any ϵ/3 > 0, there exists a δ⋆> 0, such that for any X, Y ∈Rd×L, and ∥X −Y ∥∞< δ⋆, we
have ∥f(X) −f(Y )∥F < ϵ/3.

Then we perform the following steps following Definitions E.1 and E.2:

• We create a grid Gδ⋆by choosing grid width δ⋆. We also create cube SG with respect to G ∈Gδ⋆.

• For any grid point G ∈Gδ⋆, we define CG ∈SG as the center point of the cube SG.

• We define a piecewise-constant function fδ⋆(X) = P

L∈Gδ⋆f(CG)1{X ∈SG}.

For any X ∈SG, we have ∥X −CG∥∞< δ⋆. According to the uniform continuity, we drive

∥f(X) −fδ⋆(X)∥F = ∥f(X) −f(CG)∥F < ϵ/3.

This implies that ∥f −fδ⋆∥L2 < ϵ/3 and completes the proof.

E.5.3
Proof of Lemma E.4

We give the proof of Lemma E.4 by constructing the network to satisfy the requirements.

Proof of Lemma E.4. Recall the selective shift operation in Appendix E.5.1. The overall idea of the
construction includes two steps:

• Step 1: For each j ∈[L], we stack δ−d attention layers. For g ∈[δj : δ : δj + δ−d+1 −δ] (E.8) in
the increasing order, we use the attention layer as

δ−dξ(·; g −δ/2, g + δ/2).
(E.11)

The total number of layers is Lδ−d. These layers cast G ∈eGδ to L different entries required by
Property 1 of Lemma E.4.

• Step 2: We add an extra single-head attention layer with the following attention part

Lδ−(L+1)d−1ξ(·; 0).
(E.12)

This layer achieves a global shifting and casts different G ∈eGδ to unique elements required by the
Property 2 of Lemma E.4.

The two operations map eGδ and G+
δ \ eGδ to different sets, as required by Property 3 and Property 4 of
Lemma E.4. The bounds tl and tr are calculated then.

Then, we give a detailed proof by showing the impact of the two steps and verifying the four properties
of Lemma E.4. We achieve this by making a category division of G+
δ :

• Category 1: G ∈eGδ, all entries in the point G are between 0 and L −δ.

• Category 2: G ∈G+
δ \ eGδ, the point G has at least one entry that equals to −J.

Let u = (1, δ−1, δ−2, . . . , δ−d+1). Recall that δj = (j −1)(δ −δ−d+1)/(δ −1) for any j ∈[L] in
(E.8).

32


---Page Break---
Category 1.
We denote gj := u⊤G:,j, then we have g1 < g2 < · · · < gL. The first δ−d layers
sweep the set [δj : δ : δj + δ−d+1 −δ], j ∈[L] and apply selective shift operation on each element
in the set. This means that selective shift operation will be applied to g1 first, then g2, followed by g3,
and so on.

• The First Shift Operation. In the first selective shift operation with g going through [δ1 : δ :
δ1 + δ−d+1 −δ], the (1, 1)-th entry of G (i.e., G1,1) is shifted by the operation, while the other
entries are left untouched. The updated value eG1,1 is

eG1,1 = G1,1 + δ−d

max
k
 
u⊤G:,k

−min
k
 
u⊤G:,k

= G1,1 + δ−d(gL −g1).

Therefore, the output of the layer after the operation is
  eG:,1
G:,2
· · ·
G:,L

.

Let eg1 := uT eG:,1. We have

eg1 = eG1,1 +

d
X

i=2
δ−i+1Gi,1

= G1,1 + δ−d(gL −g1) +

d
X

i=2
δ−i+1Gi,1

= g1 + δ−d(gL −g1).

Then we deduce gL < eg1, because

eg1 = g1 + δ−d(gL −g1)

≥0 + δ−d

(L −1) · δ −δ−d+1

δ −1
−δ−d+1 + δ

 
By (E.8)

= δ−d

(L −1)
δ
1 −δ + δ + (L −1)δ−d+1

1 −δ −δ−d+1


≥δ−d ·

(L −1)
δ
1 −δ + δ


= (L −1)δ−d+1

1 −δ + δ−d+1

> gL.
 
By δ < 1 and (E.8)

Thus, after updating, we have

max u⊤  eG:,1
G:,2
· · ·
G:,L

= max{eg1, g2, . . . , gL} = eg1,

and the new minimum is g2.

• The Second Shift Operation. In the second selective shift operation with g going through
[δ2 : δ : δ2 + δ−d+1 −δ], the (1, 2)-th entry of G (i.e., G1,2) is shifted by the operation, while the
other entries are left untouched. The updated value eG1,2 is

eG1,2 = G1,2 + δ−d(eg1 −g2)

= G1,2 + δ−d(g1 −g2) + δ−2d(gL −g1).

Therefore, the output of the layer after the operation is
  eG:,1
eG:,2
· · ·
G:,L

.

33


---Page Break---
We have

eg2 := u⊤eG:,2
= g2 + δ−d(g1 −g2) + δ−2d(gL −g1).

Then we deduce eg1 < eg2, because

g1 + δ−d(gL −g1) < g2 + δ−d(g1 −g2) + δ−2d(gL −g1)

⇐⇒(δ−d −1)(g2 −g1) < δ−d(δ−d −1)(gL −g1).
 
By δ−d > 1 and gL > g2


Thus, after updating, we have

max u⊤  eG:,1
eG:,2
· · ·
G:,L

= max{eg1, eg2, . . . , gL} = eg2,

and the new minimum is g3.

• Repeating the Process. By repeating this process, we show that the j-th shift operation shifts
G1,j by δ−d(egj−1 −gj). Then we have

egj := u⊤eG:,j

= gj +

j−1
X

k=1
δ−kd(gj−k −gj−k+1) + δ−jd(gL −g1).

We deduce egj−1 < egj holds for all 2 ≤j ≤L, because

egj−1 < egj

⇐⇒
gj−1 +

j−1
X

k=2
δ−kd+d(gj−k −gj−k+1) + δ−(j−1)d(gL −g1)

< gj +

j−1
X

k=1
δ−kd(gj−k −gj−k+1) + δ−jd(gL −g1)

⇐⇒

j−1
X

k=1
δ−kd+d(δ−d −1)(gj−k+1 −gj−k) < δ−(j−1)d(δ−d −1)(gL −g1),

where the last inequality holds because

j−1
X

k=1
δ−kd+d(gj−k+1 −gj−k)

< δ−(j−1)d
j−1
X

k=1
(gj−k+1 −gj−k)

< δ−(j−1)d(gL −g1).

Therefore,
after the j-th selective shift operation,
egj
is the new maximum among
{eg1, . . . , egj, gj+1, . . . , gL} and gj+1 is the new minimum.

• After L Shift Operations. After the whole L shift operations, the input G is mapped to a new
point eG, where u⊤eG = (eg1
eg2
. . .
egL) and eg1 < eg2 < · · · < egL. For the lower and upper
bound of egL, we have the following lemma.

Lemma E.8 (Lemma 10 of [Yun et al., 2020]). egL = u⊤eG:,L satisfies the following bounds:

δ−(L−1)d+1(δ−d −1) ≤egL ≤Lδ−(L+1)d.

34


---Page Break---
Also, the mapping from (g1
g2
· · ·
gL) to egL is one-to-one mapping.

• Global Shifting by the Last Layer. We note that after the above L shift operations, there is
another attention layer with attention part Lδ−(L+1)d−1ξ(·; 0). Since 0 < eg1 < · · · < egL, it adds
the following to each entry in the first row of eG:

Lδ−(L+1)d−1 max
k
u⊤eG:,k = Lδ−(L+1)d−1egL.

The output of this layer is defined to be the function fT ,c2(G).

In summary, for any G ∈eGδ, i ∈[d], and j ∈[L], we have

fT ,c2(G)i,j =
G1,j + δ+
j
if i = 1,
Gi,j
if 2 ≤i ≤d,

where δ+
j = Pj−1
k=1 δ−kd(gj−k −gj−k+1) + δ−jd(gL −g1) + Lδ−(L+1)d−1egL.

For any G ∈eGδ and j ∈[L],

u⊤fT ,c2(G):,j = egj + Lδ−(L+1)d−1egL.

Next, we check the Property 1, Property 2 and Property 3 of Lemma E.4.

• Checking Property 1 of Lemma E.4. Given any G ∈eGδ, we already prove that

eg1 < eg2 < · · · < egL,

All of them are distinct.

• Checking Property 2 of Lemma E.4. Note that the upper bound on egL from Lemma E.8 also
holds for other egj (j ∈[L −1]). For all j ∈[L], we have

Lδ−(L+1)d−1egL ≤u⊤fT ,c2(G):,j < Lδ−(L+1)d−1egL + Lδ−(L+1)d.

By Lemma E.8, two different G, G′ ∈eGδ are mapped to different egL and eg′
L, and they differ at
least by δ. This means that the following two intervals are guaranteed to be disjoint:

[Lδ−(L+1)d−1egL, Lδ−(L+1)d−1egL + Lδ−(L+1)d),

[Lδ−(L+1)d−1eg′
L, Lδ−(L+1)d−1eg′
L + Lδ−(L+1)d).

Thus, the entries of u⊤fT ,c2(G) and u⊤fT ,c2(G′) are all distinct.

Now, we finish showing that the mapping fT ,c2(·) uses (1/δ)d + 1 attention layers to implement a
contextual mapping on eGδ.

• Checking Property 3 of Lemma E.4.
Given Lemma E.8 and u⊤fT ,c2(G):,j
∈
[Lδ−(L+1)d−1egL, Lδ−(L+1)d−1egL + Lδ−(L+1)d), for any G ∈eGδ, we have

u⊤fT ,c2(G):,j ≥Lδ−2(L+1)d(δ−d −1),

u⊤fT ,c2(G):,j < L2δ−2(L+1)d−1 + Lδ−(L+1)d.

This proves that all u⊤fT ,c2(L):,j are between tl and tr, where

tl = Lδ−2(L+1)d(δ−d −1),

35


---Page Break---
tr = L2δ−2(L+1)d−1 + Lδ−(L+1)d.

Category 2.
Now we check the Property 4 of Lemma E.4. For the input points G ∈G+
δ \ eGδ,
note that the point G has at least one entry that equals to −J + k, k ∈[L −1]. Let gj := u⊤G:,j.
Recall that whenever a column G:,j has an entry that equals to −J + k, k ∈[L −1], we have gj < 0.
Without loss of generality, assume that g1 < 0.

Because the selective shift operation is applied to each element of [0 : δ : δL + δ−d+1 −δ] and is not
applied to negative values, thus we have mink u⊤G:,k = g1 < 0. g1 never gets shifted upwards and
remains the minimum for the whole time.

• All gj’s are Negative. When all gj’s are negative, selective shift operation never shifts the input
G. Thus eG = G. Recall that u⊤eG:,j < 0 for all j ∈[L]. The last layer with attention part
Lδ−(L+1)d−1ξ(·; 0) adds Lδ−(L+1)d−1 mink u⊤eG:,k < 0 to each entry in the first row of eG. This
makes eG remain negative. Therefore, fT ,c2(G) satisfies u⊤fT ,c2(G):,j < 0 < tl for all j ∈[L].

• Not All gj’s are Negative. Now consider the case where at least one gj is positive. Suppose that
there are k positive elements and they satisfy gi1 < gi2 < · · · < gik. Thus selective shift operation
does not affect gi, where i ∈[L] \ {i1, . . . , ik}. It shifts gi1 by

δ−d(max
k
u⊤G:,k −min
k u⊤G:,k)

≥δ−d(2Lδ−dL −(L −1)δ−d+1 −δ

1 −δ
−δ−d+1 + (ik −1)δ−d+1 −δ

1 −δ
)
 
By (E.9)

= δ−d(3Lδ−dL −δ−d+1 −(L −ik)δ−d+1 −δ

1 −δ
)

≥δ−d · 2Lδ−dL
 
By δ−1 ≥2

= 2Lδ−(L+1)d.

The next shift operations shift gi2, . . . , gik by an even larger amount. Therefore, at the end
of the first L(1/δ)d layers, we have Lδ−(L+1)d ≤egi1 ≤· · · ≤egik, and egj < 0 for all j ∈
[L] \ {i1, . . . , ik}.

Then, we shift G by the last layer. The last layer with attention part Lδ−(L+1)d−1ξ(·; 0) acts
differently for negative and positive egj’s. (i). For negative egj’s, it adds the following to egj, j ∈
[L] \ {i1, . . . , ik}:

Lδ−(L+1)d−1 min
k u⊤eG:,k = Lδ−(L+1)d−1g1 < 0.

This term pushes them further to the negative side. (ii). For positive egi’s, it adds

Lδ−(L+1)d−1 max
k
u⊤eGk = Lδ−(L+1)d−1egik ≥2L2δ−2(L+1)d−1.

Thus they are all greater than or equal to 2L2δ−2(L+1)d+1. Note that

2L2δ−2(L+1)d−1 > tr, where tr = L2δ−2(L+1)d−1 + Lδ−(L+1)d.

Then the final output fT ,c2(G) satisfies u⊤fT ,c2(G):,j /∈[tl, tr], for all j ∈[L]. This completes
the verification of Property 4 of Lemma E.4.

In conclusion, we need O(Lδ−d) layers of modified self-attention layer to obtain our approximation.
This completes the proof.

36


---Page Break---
E.5.4
Proof of Lemma E.5

Proof of Lemma E.5. We restate the proof from [Yun et al., 2020] for completeness.

Note that |G+
δ | = (1/δ + 1)dL < ∞, so the output of fT ,c2(G+
δ ) has finite number of distinct real
values. Let M be the upper bound of all these possible values. By the construction of fT ,c2, M > 0.

Construct the Layers: fT ,c3(fT ,c2(G)) = 0d×L if G ∈G+
δ \ eGδ.
According to Lemma E.4, for
all j ∈[L], we have u⊤fT ,c2(G):,j ∈[tl, tr] if G ∈eGδ, and u⊤fT ,c2(G):,j /∈[tl, tr] if G ∈G+
δ \ eGδ.
Due to this property, we add the following feed-forward layer.

Definition E.8 (Feed-forward Layer 3). The vectors u and 1L act as the weight parameters, and ζ3(·)
acts as the activation function in the feed-forward layer.

X →X −(M + 1)1Lζ3(u⊤X), ζ3(t) =
0
if t ∈[tl, tr]
1
if t /∈[tl, tr].
(E.13)

• Case for G ∈G+
δ \ eGδ. We have ζ3(u⊤fT ,c2(G)) = 1⊤
L. Thus, all the entries of the input are
shifted by −M −1 and become strictly negative.

• Case for G ∈eGδ. We have ζ3(u⊤fT ,c2(G)) = 0⊤
L, so the output stays the same as the fT ,c2(G).

With the input fT ,c2(G), if G ∈eGδ, then ζ3(u⊤fT ,c2(G)) = 0⊤
L. Thus, the output stays the same as
the input. If G ∈G+
δ \ eGδ, then ζ3(u⊤fT ,c2(G)) = 1⊤
L. Thus, all the entries of the input are shifted
by −M −1 and become strictly negative.

Next, we map those negative entries to zero. For i = 1, 2, · · · , d, we add the following layer:

Definition E.9 (Feed-forward Layer 4). The vectors u and ei act as the weight parameters and ζ4(·)
acts as the activation function in the feed-forward layer.

X →X + eiζ4((ei)⊤X), ζ4(t) =
−t
if t < 0
0
if t ≥0.
(E.14)

After these d layers, the output for G ∈G+
δ \ eGδ is a zero matrix, while the output for G ∈eGδ remains
fT ,c2(G).

Construct the Layers: fT ,c3(fT ,c2(G)) = AG if G ∈eGδ.
Each different G is mapped to L unique
numbers u⊤fT ,c2(G), which are at least δ apart from each other. We map each unique number to the
corresponding output column as follows. We choose one G ∈eGδ. For each u⊤fT ,c2(G):,j, j ∈[L],
we add the following feed-forward layer.

Definition E.10 (Feed-forward Layer 5). The vectors u and ei act as the weight parameters, and
ζ4(·) acts as the activation function in the feed-forward layer.

X →X +
 
(AG):,j −fT ,c2(G):,j

ζ5(u⊤X −u⊤fT ,c2(G):,j1⊤
L),
(E.15)

ζ5(t) =
1
−δ/2 ≤t < δ/2,
0
others.
(E.16)

• Case for G ∈G+
δ \ eGδ. Recall that the input X of this layer is fT ,c2(G). If X is a zero matrix,
which is the case for G ∈G+
δ \ eGδ, we have u⊤X = 0⊤
L. Then u⊤X −u⊤fT ,c2(G):,j1⊤
L < −tl1L.
Since tl > δ/2, the output remains the same as X.

37


---Page Break---
• Case for G ∈eGδ.
Let the input X be fT ,c2(G), where G ∈eGδ is not equal to G. According
to the Property 2 of Lemma E.4 and given a j ∈[L], u⊤fT ,c2(G):,k, (k ∈[L]) differs from
u⊤fT ,c2(G):,j by at least δ. Then we have

ζ5(u⊤fT ,c2(G) −u⊤fT ,c2(G):,j1⊤
L) = 0⊤
L.

Thus the input is left untouched.

If G = G, then

ζ5(u⊤fT ,c2(G) −u⊤fT ,c2(G):,j1⊤
L) = (ej)⊤.

Thus we shift the j-th column of fT ,c2(G) to

fT ,c2(G):,j + ((AG):,j −fT ,c2(G):,j) = fT ,c2(G):,j + ((AG):,j −fT ,c2(G):,j) = (AG):,j.

In other word, this layer maps the column fT ,c2(G):,j to (AG):,j, without affecting any other
columns.

For each G ∈eGδ, we defer that we need one layer for each unique value of u⊤fT ,c2(G):,j. Note that
there are O(δ−dL) such numbers, so we use O(δ−dL) layers to finish our construction.

This completes the proof.

38


---Page Break---
E.5.5
Proof of Lemma E.7

Proof of Lemma E.7. We restate the proof from [Yun et al., 2020] for completeness.

The proof follows two steps: (i) Approximate the modified self-attention layers. (ii) Approximate the
modified feed-forward layers.

• Step 1: Approximate the Modified Self-Attention Layers.

We achieve this by approximating the Softmax operator σS with the Hardmax operator σH.
Given a matrix X ∈Rd×L, we have

σS(λX) →σH(X),
as
λ →∞.

The operator is the only difference between the normal and the modified self-attention layers. We
approximate the modified self-attention layer in T r,m,l
p
by the normal self-attention layer with the
same number of heads r and head size m.

• Step2: Approximate the Modified Feed-Forward Layers.

We achieve this by approximating the activation function in Ψ with four ReLU functions. From
Definition E.3, we recall that Ψ denotes three-piecewise functions with at least a constant piece.
We consider the following ζ ∈Ψ:

ζ(x) =






b1
if x < c1,
a2x + b2
if c1 ≤x < c2,
a3x + b3
if c2 ≤x,

where a2, a3, b1, b2, b3, c1, c2 ∈R, and c1 < c2.

We approximate ζ(x) by eζ(x) composed of four ReLU functions:

eζ(x) =b1 + a2c1 + b2 −b1

ϵ
ReLU(x −c1 + ϵ) +

a2 −a2c1 + b2 −b1

ϵ


ReLU(x −c1)

+
a3c2 + b3 −a2(c2 −ϵ) −b2

ϵ
−a2


ReLU(x −c2 + ϵ)

+

a3 −a3c2 + b3 −a2(c2 −ϵ) −b2

ϵ


ReLU(x −c2)

=
















b1
if x < c1 −ϵ,
(a2c1 + b2 −b1)(x −c1)/ϵ + a2c1 + b2
if c1 −ϵ ≤x < c1,
a2x + b2
if c1 ≤x < c2 −ϵ,
(a3c2 + b3 −a2(c2 −ϵ) −b2)(x −c2)/ϵ + a3c2 + b3
if c2 −ϵ ≤x < c2,
a3x + b3
if c2 ≤x.

As ϵ →0, we approximate ζ(x) by eζ(x). The activation function is the only difference between
the normal and modified feed-forward layers. We approximate the modified feed-forward layer in
T r,m,l
p
by the normal one.

Thus, for any fT ,c ∈T 2,1,1
p
, there exists a function fT ∈T 2,1,4
p
to approximate fT ,c.

This completes the proof.

39


---Page Break---
F
Proofs of Section 3

Our proofs are motivated by the approximation and estimation theory of U-Net-based diffusion models
in [Chen et al., 2023]. We use transformer networks’ universal approximation theory in Appendix E
and the covering number to proceed with our proof. Specifically, we derive the approximation error
bound in Appendix F.1 and the corresponding sample complexity bound in Appendix F.2. Then
we show that the data distribution generated from the estimated score function converges toward a
proximate area of the original one in Appendix F.3.

F.1
Proof of Theorem 3.1

Here we present some auxiliary theoretical results in Appendix F.1.1 to prepare for our main proof of
Theorem 3.1. Then we derive the approximation error bound of DiTs (i.e., the proof of Theorem 3.1)
in Appendix F.1.2.

F.1.1
Auxiliary Lemmas for Theorem 3.1.

We restate some auxiliary lemmas and their proofs from [Chen et al., 2023] for later convenience.

Lemma F.1 (Lemma 16 of [Chen et al., 2023]). Consider a probability density function ph(h) =

exp

−C∥h∥2
2/2

for h ∈Rd0 and constant C > 0. Let rh > 0 be a fixed radius. Then it holds

Z

∥h∥2>rh
ph(h)dh ≤
2d0πd0/2

CΓ(d0/2 + 1)rd0−2
h
exp
 
−Cr2
h/2

,

Z

∥h∥2>rh
∥h∥2
2ph(h)dh ≤
2d0πd0/2

CΓ(d0/2 + 1)rd0
h exp
 
−Cr2
h/2

.

Lemma F.2 (Lemma 2 of [Chen et al., 2023]). Suppose Assumption 2.2 holds and g is defined as:

q(h, t) =
Z
hψt(h|h)ph(h)
R
ψt(h|h)ph(h)dhdh,
h = B⊤x.

Given ϵ > 0, with rh = c
p

d0 log(d0/T0) + log(1/ϵ)

for an absolute constant c, it holds

q(h, t)1{
h

2 ≥rh}

L2(Pt) ≤ϵ, for t ∈[T0, T].

Lemma F.3 (Theorem 1 of [Chen et al., 2023]). We denote

τ(rh) =
sup
t∈[T0,T ]
sup
h∈[0,rh]d


∂
∂tq(h, t)

2
.

With q(h, t) =
R
hψt(h|h)ph(h)/(
R
ψt(h|h)ph(h)dh)dh and ph satisfies Assumption 2.2, we have
a coarse upper bound for τ(rh):

τ(rh) = O
1 + β2(t)

β(t)


Ls+ +
1
σ(t)

 p

d0rh


= O

eT/2Ls+rh
p

d0

.

Lemma F.4 (Lemma 10 of [Chen et al., 2020b]). For any given ϵ > 0, and L-Lipschitz function g
defined on [0, 1]d0, there exists a continuous function f constructed by trapezoid function, such that
g −f

∞≤ϵ.

40


---Page Break---
Moreover, the Lipschitz continuity of f is bounded:
f(x) −f(y)
 ≤10d0L∥x −y∥2
for any
x, y ∈[0, 1]d0.

F.1.2
Main Proof of Theorem 3.1

Proof of Theorem 3.1. With ∇log ph
t
 

h

= B⊤s+(h, t), we have the following in (2.4)

q(h, t) = σ(t)∇log ph
t
 

h

+ B⊤x = σ(t)B⊤(s+(h, t) + x).
(F.1)

We proceed as follows:

• Step 1. Approximate q(h, t) with a compact-supported continuous function f(h, t).

• Step 2. Approximate f(h, t) with a transformer network.

Step 1. Approximate q(h, t) with a Compact-supported Continuous Function f(h, t).
We
partition Rd0 into a compact subset H1 := {h|
h

2 ≤rh} and its complement H2, where rh is
to be determined later. We approximate q(h, t) on the two subsets respectively and then prove f’s
continuity. Such a step achieves an estimation error of √d0ϵ between q(h, t) and f(h, t). We show
the main proof here.

• Approximation on H2 × [T0, T]. For any ϵ > 0, we take rh = c(
p

d0 log(d0/T0) −log ϵ). From
Lemma F.2, we have
q(h, t)1{
h

2 ≥rh}

L2(Pt) ≤ϵ
for
t ∈[T0, T].

So we set f(h, t) = 0 on H2 × [T0, T].

• Approximation on H1 ×[T0, T]. On H1 ×[T0, T], we approximate q(h, t) by approximating each
coordinate qk(h, t) respectively, where q(h, t) = [q1(h, t), q2(h, t), · · · , qd0(h, t)]. We rescale the
input by y′ = (h+rh1)/2rh and t′ = t/T. Then the transformed input space is [0, 1]d0×[T0/T, 1].
We implement such a transformation by a single feed-forward layer.

By Assumption 2.3, on-support score s+(h, t) is Ls+-Lipschitz in h. This implies q(h, t) is
(1 + Ls+)-Lipschitz in h. When taking the transformed inputs, g(y′, t′) = q(2rhy′ −rh1, Tt′)
becomes 2rh(1 + Ls+)-Lipschitz in y′. Similarly, each coordinate gk(y′, t) is also 2rh(1 + Ls+)-
Lipschitz in y′. Here we take Lh = 1 + Ls+.

Besides, g(y′, t′) is Tτ(rh)-Lipsichitz with respect to t, where

τ(rh) =
sup
t∈[T0,T ]
sup
h∈[0,rh]d


∂
∂tq(h, t)

2
.

We have a coarse upper bound for τ(rh) in Lemma F.3. We restate it here for convenience

τ(rh) = O
1 + β2(t)

β(t)


Ls+ +
1
σ(t)

 p

d0rh


= O

eT/2Ls+rh
p

d0

.

In conclusion, each gk(y′, t) is Lipsichitz continuous. So we can apply Lemma F.4 to determine
f k(y′, t) for approximating each coordinate. We concatenate f i’s together and construct f =
[f 1, . . . , f d0]⊤. According to the construction in Lemma F.4 and for any given ϵ, we achieve

sup
y′,t′∈[0,1]d×[T0/T,1]

f(y′, t′) −g(y′, t′)

∞≤ϵ,

41


---Page Break---
Considering the input rescaling (i.e., h →y′ and t →t′), we obtain:

– The constructed function is Lipschitz continuous in h. For any h1, h2 ∈H1 and t ∈[T0, T], it
holds
f(h1, t) −f(h2, t)

∞≤10d0Lh
h1 −h2

2.
(F.2)

– The function is also Lipschitz in t. For any t1, t2 ∈[T0, T] and
h

2 ≤rh, it holds

f(h, t1) −f(h, t2)

∞≤10τ(rh)∥t1 −t2∥2.

Due to the fact that the construction of f(h, t) is based on trapezoid function, we have f(h, t) = 0
for
h

2 = rh and any t ∈[T0, T]. Thus, the two parts of f(h, t) can be joined together. To be
more specific, the above Lipschitz continuity in h extends to the whole Rd0.

• Approximation Error Analysis under L2 Norm. The L2 approximation error of f can be
decomposed into two terms:
q(h, t) −f(h, t)

L2(P h
t )

=
(q(h, t) −f(h, t))1{
h

2 < rh}

L2(P h
t ) +
q(h, t)1{
h

2 > rh}

L2(P h
t ).

The second term in the RHS above has already been bounded with the selection of rh:
g(h, t)1{
h

2 > rh}

L2(P h
t ) ≤ϵ.

The first term is bounded by:
(q(h, t) −f(h, t))1{
h

2 < rh}

L2(P h
t )

≤
p

d0
sup
y′,t′∈[0,1]d×[T0/T,1]

f(y′, t′) −g(y′, t′)

∞

≤
p

d0ϵ.

Then we obtain
q(h, t) −f(h, t)

L2(P h
t ) ≤(
p

d0 + 1)ϵ.

If we substitute ϵ with ϵ/2, we obtain that the approximation error of f(h, t) is √d0ϵ.

Step 2. Approximate f(h, t) by a Transformer. This step is based on the universal approximation
of transformers for the compact-supported continuous function in Lemma E.1. DiT uses time point
t to calculate the scale and shift value in the transformer backbone [Peebles and Xie, 2023]. It
also transforms an input picture into a sequential version. We ignore time point t in the notation of
the transformer network in DiT. Recall the reshape layer R(·) in Definition 3.1, we consider using
f(·) := R−1 ◦fT ◦R(·) to approximate f t(·) := f(·, t), where fT ∈T 2,1,4
p
.

• Overall Approximation Error.
With Lemma E.1, we approximate f t(·) with bf(·) :=
R−1 ◦bfT ◦R(·). We denote

H = R(h).

We have

f t(h) −bf(h)

L2(P h
t ) =

 Z

P h
t

f t(h) −bf(h)

2

2dh

!1/2

42


---Page Break---
=

 Z

P h
t

R ◦f t ◦R−1(H) −R ◦bf ◦R−1(H)

2

F dh

!1/2

=

 Z

P h
t

R ◦f t ◦R−1(H) −bfT (H)

2

F dh

!1/2

≤ϵ.
(F.3)

Along with Step 1, we obtain
q(h, t) −bf(h)

L2(P h
t ) ≤
q(h, t) −f(h, t)

L2(P h
t ) +
f(h, t) −bf(h)

L2(P h
t ) ≤(1 +
p

d0)ϵ.

The constructed approximator to ∇log pt(x) is s b
W = (B bf(B⊤x, t) −x)/σ(t), and the approxi-
mation error is
∇log pt(·) −s b
W (·, t)

L2(Pt) ≤1 + √d0

σ(t)
ϵ
for any
t ∈[T0, T].

• Settling-down of Hyperparameters.
We settle down the hyperparameters to configure our
network here. We refer to Appendix E.2 for some of the following calculations.

1. Model Architecture Depth K.
From Lemma E.6, we have K = O((1/δ)dL). To achieve ϵ-error approximation, we set
δ = O
 
ϵ2/d
according to Lemma E.3. Thus we obtain

K = O
 
ϵ−2L
.
(F.4)

2. Lipchitz Upperbound for Transformer: LT .
We denote f t,R(·) = R ◦f t ◦R−1(·). We get the Lipshitz upper bound for bfT ∈T 2.1.4
p
in the
following way

 bfT (H1) −bfT (H2)

F ≤
 bfT (H1) −f t,R (H1)

F +
f t,R (H1) −f t,R (H2)

F

+
f t,R (H2) −bfT (H2)

F
≤2ϵ +
f t,R (H1) −f t,R (H2)

F

 
By (F.3)

≤2ϵ + 10d0Ls+∥H1 −H2∥F .
 
By (F.2)

Then we get

LT = O
 
d0Ls+

.
(F.5)

3. Model Output Bound for ST 2,1,4
p
.

For the output of the constructed transformer bfT (·), according to Lemma E.5, we have bfT (O) =
O, where O = 0d×L. Thus, with the Lipschitz upperbound O(d0Ls+), we have ∥bfT (H)∥F =
O(d0Ls+rh), where ∥H∥F ≤rh. With rh = c(
p

d0 log(d0/T0) + log(1/ϵ)), we obtain

CT = O

d0Ls+ ·
p

d0 log(d0/T0) + log(1/ϵ)

.
(F.6)

4. Model Parameters Bound: C2,∞
OV , COV , C2,∞
KQ , CKQ, CE.
By definition, we have:
(W i
OV )⊤
2,∞≤C2,∞
OV ,
(W i
OV )⊤
2 ≤COV ,
W i
KQ

2,∞≤C2,∞
KQ ,
W i
KQ

2 ≤CKQ,

43


---Page Break---
where i = 1, 2. For simplicity, we omit i hereafter, which does not affect our discussion.

Recall that ∥Z∥2,∞denotes the 2, ∞-norm, where the 2-norm is over columns and ∞-norm is
over rows. By the construction of modified attention layers (E.11) and (E.12) in Appendix E.5.3,
we consider WOV to have the largest norm, i.e.,

WOV = Lδ−(L+1)d−1 ·







1
δ−1
· · ·
δ−d+1
0
0
· · ·
0
...
...
· · ·
...
0
0
· · ·
0





.

We give the following upper bounds
W ⊤
OV

2,∞= Ldδ−(L+2)d = O
 
δ−Ld
,
(F.7)

W ⊤
OV

2 =
sup
∥x∥2=1

W ⊤
OV x

2 = Lδ−(L+1)d−1 ·

v
u
u
t

d−1
X

i=0
δ−2i = O
 
δ−Ld
.
(F.8)

By (E.11) and (E.12) in Appendix E.5.3, and the self-attention layers in Appendix E.5.5, we
consider WKQ to have the largest norm, i.e.,

WKQ :=







1
δ−1
...
δ−d+1






 
1, δ−1, · · · , δ−d+1
=








1
δ−1
· · ·
δ−d+1

δ−1
δ−2
· · ·
δ−d
...
...
· · ·
...
δ−d+1
δ−d
· · ·
δ−2d+2






.

Then we have

∥WKQ∥2,∞=

v
u
u
t

d−1
X

i=0
δ−2i−2d+2 = O(δ−2d),
(F.9)

∥WKQ∥2 =
sup
∥x∥2=1
∥WKQx∥2 = δ−2d+2 = O(δ−2d).
(F.10)

We substitute δ with O
 
ϵ2/d
(according to Appendix E.4) and get:

C2,∞
OV = (1/ϵ)O(1),

COV = (1/ϵ)O(1),

C2,∞
KQ = (1/ϵ)O(1),

CKQ = (1/ϵ)O(1).

From the construction of positional encoder (E.4) in Appendix E.2, we have

E =










0
1
· · ·
L −1
0
1
· · ·
L −1
...
...
...
...
...
...
...
...
0
1
· · ·
L −1









.

We deduce
E⊤
2,∞=
√

L(L −1) = O(L3/2).

44


---Page Break---
Thus we have

CE = O(L3/2).
(F.11)

5. Parameters Bound in Feed Forward Layers: C2,∞
F
, CF .
Recall the construction of modified feed-forward layers in the proof of Lemma E.4, which
includes Definitions E.5, E.6 and E.8 to E.10. With the approximation by normal feed-forward
layers in Appendix E.5.5, we consider the weight parameters with the largest norm in the
feed-forward layers, i.e.,

W1 :=






1
1
1
1





 
1, δ−1, · · · , δ−d+1
=







1
δ−1
· · ·
δ−d+1

1
δ−1
· · ·
δ−d+1

1
δ−1
· · ·
δ−d+1

1
δ−1
· · ·
δ−d+1





∈R4×d.

Then we have

C2,∞
F
= O





v
u
u
t

d−1
X

i=0
δ−2i



= O
 
δ−d
(F.12)

= (1/ϵ)O(1).
 
By setting δ = O(ϵ2/d) according to Appendix E.4

and

CF =
sup
∥x∥2=1
∥W1x∥2 = O
 
δ−d
(F.13)

= (1/ϵ)O(1).
 
By setting δ = O(ϵ2/d) according to Appendix E.4

This completes the proof.

45


---Page Break---
F.2
Proof of Theorem 3.2

Here we present the auxiliary theoretical results about the covering number of transformer networks
in Appendix F.2.1. The results are based on [Edelman et al., 2022, Theorem A.17]. Then we derive
the sample complexity bound of DiTs (i.e., the proof of Theorem 3.2) in Appendix F.2.

F.2.1
Auxiliary Lemmas for Theorem 3.2

Lemma F.5 (Lemma 15 of [Chen et al., 2023]). Let G be a bounded function class. Then there exists a
constant b such that the output of any g ∈G : Rd0 7→[0, b] is bounded by b. Let z1, z2, · · · , zn ∈Rd0
be i.i.d. random variables. For any δ ∈(0, 1), a ≤1, and c > 0, we have

P

 

sup
g∈G

1
n

n
X

i=1
g(zi) −(1 + a)E [g(z)] > (1 + 3/a)B

3n
log N(c, G, ∥·∥∞)

δ
+ (2 + a)c

!

≤δ,

P

 

sup
g∈G
E [g(z)] −1 + a

n

n
X

i=1
g(zi) > (1 + 6/a)B

3n
log N(c, G, ∥·∥∞)

δ
+ (2 + a)c

!

≤δ.

Now, we give the definition of the covering number as follows.

Definition F.1 (Covering Number). Given a function class F and a data distribution P. Sample
n data points {Xi}n
i=1 from P. For any ϵ > 0, the covering number N(ϵ, F, {Xi}n
i=1, ∥·∥) is the
smallest size of a collection (a cover) C ∈F, such that for any f ∈F, there exists a bf ∈C satisfying

max
i

f(Xi) −bf(Xi)
 ≤ϵ.

Furthermore, we define the covering number with respect to the data distribution as

N(ϵ, F, ∥·∥) =
sup
{Xi}n
i=1∼P
N(ϵ, F, {Xi}n
i=1, ∥·∥).

Then we give the covering number of the transformer networks.

Lemma
F.6
(Modified
from
Theorem
A.17
of
[Edelman
et
al.,
2022]).
Let
T r,m,l
p
(K, CT , C2,∞
OV , COV , C2,∞
KQ , CKQ, C2,∞
F
, CF , CE, LT ) represent the class of functions
of K-layer transformer blocks satisfying the norm bound for matrix and Lipsichitz property for
feed-forward layers. Then for all data point ∥X∥2,∞≤CX, we have

log N(ϵc, T r,m,l
p
(K, CT , C2,∞
OV , COV , C2,∞
KQ , CKQ, C2,∞
F
, CF , CE, LT ), ∥·∥2)

≤log(nL)

ϵ2c
·

 K
X

i=1
α
2
3

d
2
3

C2,∞
F
 4

3 + d
2
3

2(CF )2COV C2,∞
KQ
 2

3 + τm
2
3

(CF )2C2,∞
OV
 2

3 !3

,

where α := Q
j<i(CF )2COV (1 + 4CKQ)(CX + CE).

Remark F.1. We modify [Edelman et al., 2022, Theorem A.17] in seven aspects:

1. We do not consider the last linear layer in the model, which converts each column vector of the
transformer output to a scalar. Therefore, we ignore the item related to the last linear layer in
[Edelman et al., 2022, Theorem A.17].

2. We do not consider the normalization layer in our model.
Because the normalization
layer Q

norm(·) in the original proof only ensures that ∥Q

norm(X1) −Q

norm(X2)∥2,∞≤
∥X1 −X2∥2,∞, ignoring this layer does not change the result.

3. Our activation function is ReLU. Thus, we replace the Lipschitz upperbound of activate function
by 1.

46


---Page Break---
4. We consider the positional encoding (E.4). Then we need to replace the upperbound CX for
the inputs with the upperbound CX + CE. Besides, for multi-layer transformer, the original
conclusion in [Edelman et al., 2022, Theorem A.17] uses 1 as the upperbound for the 2, ∞-norm
of inputs. We incorporate the upperbound for the inputs into the result stated in Lemma F.6.

5. We use (2.7) as the feed-forward layer, including two linear layers and a residual layer. Thus, we
replace the original upperbound for the norm of weight matrix with the upperbound for the norm
of Id + W2W1 in Lemma F.6. In the following, we use O to estimate the log-covering number,
thus we ignore the item for Id here for converience. This is the same for the self-attention layer.

6. We use multi-head attention, and incorporate the number of heads τ into our result, which is
similar to [Edelman et al., 2022, Theorem A.12].

7. In our work, we use the transformer T 2,1,4
p
, i.e., τ = 2, m = 1.

F.2.2
Proof of Theorem 3.2

Proof of Theorem 3.2. Our proof is built on [Chen et al., 2023, Appendix B.2]. For one data sample,
we define the empirical score matching loss objective (2.1) as follows

ℓ(x; s b
W ) =
1
T −T0

Z T

T0
Ext|x0=x[
∇xt log ψt(xt|x0) −s b
W (xt, t)

2

2]dt.

Then we define L(s b
W ) = Ex∼P0
h
ℓ(x; s b
W )
i
.

Following [Chen et al., 2023, Appendix B.2], for any a ∈(0, 1), we have

L(s b
W )

≤Ltrunc(s b
W ) −(1 + a) bLtrunc(s b
W )
|
{z
}
(I)

+ L(s b
W ) −Ltrunc(s b
W )
|
{z
}
(II)

+(1 + a)
inf
sW ∈SNN
bL(sW )
|
{z
}
(III)

,

where

Ltrunc(s b
W ) := Ex∼P0
h
ℓtrunc(x; s b
W )
i
= Ex∼P0
h
ℓ(x; s b
W )1{∥x∥2 ≤rx}
i
, rx > B.

We denote

η := 4CT (CT + rx)(rx/D)D−2 exp
 
−r2
x/σ(t)

/(T0(T −T0)),

rx := O
q

d0 log d0 + log CT + log
 
n/δ

.

Then we have

η ≤
1
nT0(T −T0).
(F.14)

For any δ > 0, according to Lemma F.5, the following holds for term (I) with probability 1 −δ,

(I) = O



(1 + 6/a)(C2
T + r2
x)
nT0(T −T0)
log
N

(T −T0)(ι−η)
(CT +rx) log(T/T0), ST 2,1,4
p
, ∥·∥2


δ
+ (2 + a)ι



,

where c ≤0 is a constant, and ι > 0 will be determined later.

47


---Page Break---
We set

ι :=
2
nbT0(T −T0),

where 0 < b ≤1 is a constant to be determined later.

Remark F.2 (Selection Criteria of τ). We have two criteria:

• Recall that the covering number used in our setting is N

(T −T0)(ι−η)
(CT +rx) log(T/T0), ST 2,1,4
p
, ∥·∥2

. Thus,
we must ensure ι ≥η. According to (F.14), we consider ι satisfying the condition ι ≥(nT0(T −
T0))−1. Therefore, we consider 0 < b ≤1.

• For the exponent of (T −T0), although selecting a value smaller than 1 is possible, we find that
the convergence rate with respect to T is dominated by the 1/T term appearing later in the second
term of (F.18). Therefore, we continue to consider the exponent to be 1.

Then we have

(I) = O



(1 + 6/a)  
C2
T + r2
x


nT0(T −T0)
log
N

(nb(CT + rx)T0 log(T/T0))−1, ST 2,1,4
p
, ∥·∥2


δ
+
4 + 2a
nbT0(T −T0)



,

with probability 1 −δ.

Following the proof structure of term (II) in [Chen et al., 2023, Appendix B.2], we have

(II) = O
 1

T0
C2
T r2
x exp

−A2r2
x/2
	
.

For any ϵ > 0, let sW be the transformer network approximator to the score function in Theorem 3.1.
For the term (III), we have

(III) ≤bL(sW ) −(1 + a)Ltrunc(sW )
|
{z
}
(III)1

+(1 + a) Ltrunc(sW )
|
{z
}
(III)2

.

For any δ > 0, according to Lemma F.5 and given that sW is a fixed function, the following holds for
term (III)1 with probability 1 −δ,

(III)1 = O

 
(1 + 3/a)
 
C2
T + r2
x


nT0(T −T0)
log 1

δ

!

.

Following the proof structure of term (III)2 in [Chen et al., 2023, Appendix B.2], we have

(III)2 = O

dϵ2

T0(T −T0)


+ C3,

where C3 is a constant.

Putting (I), (II), and (III) together and setting a = ϵ2, then we have

1
T −T0

Z T

T0

sb
W (·, t) −∇log pt(·)
2

L2(Pt)dt

= O




 
C2
T + r2
x


ϵ2nT0(T −T0) log
N

(nb(CT + rx)T0 log(T/T0))−1, ST 2,1,4
p
, ∥·∥2


δ
+ n−b + d0ϵ2

T0(T −T0)



, (F.15)

with probability 1 −3δ.

48


---Page Break---
Covering Number of ST 2,1,4
p
.
The next step is to calculate the covering number of ST 2,1,4
p
. ST 2,1,4
p
consists of two components: (i) Matrix WB with orthonormal columns; (ii) Network function fT .

Suppose
we
have
WB1, WB2
and
f1, f2,
such
that
∥WB1 −WB2∥F
≤
δ1
and
sup∥x∥2≤3rx+√

D log D,t∈[T0,T ] ∥f1(x, t) −f2(x, t)∥2 ≤δ2, where f1 = R−1 ◦fT 1 ◦R, f2 =

R−1 ◦fT 2 ◦R. Then we have

sup

∥x∥2≤3rx+√

D log D,t∈[T0,T ]
∥sWB1,fT 1(x, t) −sWB2,fT 2(x, t)∥2

=
1
σ(t)
sup

∥x∥2≤3rx+√

D log D,t∈[T0,T ]

WB1f1(W ⊤
B1x, t) −WB2f2(W ⊤
B2x, t)

2

≤
1
σ(t)
sup

∥x∥2≤3rx+√

D log D,t∈[T0,T ]

 
WB1f1(W ⊤
B1x, t) −WB1f1(W ⊤
B2x, t)

2

+
WB1f1(W ⊤
B2x, t) −WB1f2(W ⊤
B2x, t)

2 +
WB1f2(W ⊤
B2x, t) −WB2f2(W ⊤
B2x, t)

2

!

≤
1
σ(t)


LT δ1
p

d0(3rx +
p

D log D) + δ2 + δ1K

,
(F.16)

where LT upper bounds the Lipschitz constant of fT .

For the set {WB ∈RD×d0 : ∥WB∥2 ≤1}, its δ1-covering number is
 
1 + 2√d0/δ1
Dd0 [Chen
et al., 2020a, Lemma 8]. The δ2-covering number of f needs further discussion as there is a reshaping
process in our network. The input is reshaped from h ∈Rd0 to H ∈Rd×L, and
h

2 ≤rx ⇐⇒∥H∥F ≤rx.

Thus we have

sup
∥h∥2≤3rx+√

D log D,t∈[T0,T ]

f1(h, t) −f2(h, t)

2 ≤δ2

⇐⇒
sup

∥H∥F ≤3rx+√

D log D,t∈[T0,T ]
∥fT 1(H) −fT 2(H)∥2 ≤δ2.

Then we follow the covering number of sequence-to-sequence transformer T 2,1,4
p
in Lemma F.6. We
get the following δ2-covering number

log(nL)

δ2
2
·

 K
X

i=1
α

2
3
i


d
2
3

C2,∞
F
 4

3 + d
2
3

2(CF )2COV C2,∞
KQ
 2

3 + τm
2
3

(CF )2C2,∞
OV
 2

3 !3

,

where

αi :=
Y

j<i
(CF )2COV (1 + 4CKQ)(CX + CE).

According to the (F.4), (F.5), (F.7), (F.8), (F.9), (F.10), (F.12), (F.13), (F.11) and (F.6) in Ap-
pendix F.1.2, we derive the following with δ = O(ϵ2/d) (Appendix E.4) and d = 4 (Theorem 3.1):

K = O
 
ϵ−2L
, LT = O
 
d0Ls+

, C2,∞
OV = O(dϵ−4L), COV = O(ϵ−4L),

C2,∞
KQ = O(ϵ−4), CKQ = O(ϵ−4), C2,∞
F
= O(ϵ−4), CF = O(ϵ−2), CE = O(L3/2),
(F.17)

CT = O

d0Ls+ ·
p

d0 log(d0/T0) + log(1/ϵ)

, rx = O
q

d0 log d0 + log CT + log
 
n/δ

.

Each element of the input data is within [0, 1], as shown in Appendix E.

49


---Page Break---
For any δ3 > 0, we get the log-covering number of T 2,1,4
p
,

log N
 
δ3, T 2,1,4
p
, ∥·∥2

= O
ϵ−8K · LKd2 log(nL)

δ3



= O(1) ·
28K log(L/ϵ)d2 log(nL)

δ3


.

According to (F.15), we adopt the following value for δ3 in our setting

δ3 =
1
nb(CT + rx)T0 log(T/T0).

According to [Chen et al., 2023, Appendix B.2], the log-covering number of ST 2,1,4
p
is

log N

δ3, ST 2,1,4
p
, ∥·∥2


= O

2Dd0 · log

1 + 6CT LT
√d0(3rx + √D log D)

T0δ3


+ 28K log(L/ϵ)d2 log(nL)

T 2
0 δ2
3



 
By (F.16)

= O

n2b28(1/ϵ)L log(L/ϵ)Dd2d6
0L2
s+ · log(nL)

 
By (F.17)

= O

n2b2(1/ϵ)2LDd2d6
0L2
s+ · log(nL)

 
By (1/ϵ)L ≥8 log(L/ϵ)

= e
O

n2b2(1/ϵ)2LDd2d6
0L2
s+

 
By ignoring the log factors

= e
O

n2b2(1/ϵ)2LDd2d6
0L2
s+

.

Substituting the log-covering number into (F.15), we have

1
T −T0

Z T

T0

s b
W (·, t) −∇log pt(·)

2

L2(Pt)dt

= O

C2
T + r2
x
ϵ2nT0(T −T0)(log N(δ3, ST 2,1,4
p
, ∥·∥2) + log
 
1/δ

) +
1
nbT0(T −T0) +
d0
T0(T −T0)ϵ2

= O
 C2
T + r2
x
ϵ2nT0T (log N(δ3, ST 2,1,4
p
, ∥·∥2) + log
 
1/δ

)
|
{z
}
1st term

+
1
nbT0T +
d0
T0T ϵ2

| {z }
2nd term


.
(F.18)

Recall the following parameters:

• C2
T = O(d2
0L2
s+d0 log(d0/T0) + log(1/ϵ)),

• r2
x = O(d0 log d0 + log CT + log
 
n/δ

),

• δ: probability error,

• ϵ: approximation error,

• n: sample size,

• T0 < T/2,

• D, d, d0 > 1: feature dimension,

• L > 1: sequence length,

50


---Page Break---
• d0 = L · d,

• Ls+: Lipschitz coefficient.

Ignoring the log factors and poly(D, d, d0, LS+), the first term in (F.18) becomes

1
n1−2b ·
1
T0T · 2(1/ϵ)2L.

The second term is simplified to

1
T0T ϵ2.

Thus, the final bound is

eO

 
1
n1−2b ·
1
T0T · 2(1/ϵ)2L +
1
nbT0T +
1
T0T ϵ2
!

.

To balance the first and second terms with respect to n, we select b = 1/3. Therefore, we give the
final bound as

eO

 
1
n1/3 ·
1
T0T · 2(1/ϵ)2L +
1
n1/3T0T +
1
T0T ϵ2
!

.

This completes the proof.

51


---Page Break---
F.3
Proof of Corollary 3.2.1

Our proof is built on [Chen et al., 2023, Appendix C]. The main difference between our work and
[Chen et al., 2023] is our score estimation error in Theorem 3.2. Consequently, only the subspace
error and the total variation distance differ from [Chen et al., 2023, Theorem 3].

First, we introduce the ground truth backward SDE and the learned backward SDE of the latent
variable. Recall from (D.2), yt denotes the backward process. We denote the backward latent variable
by h←
t
= B⊤yt. Since we write the time index explicitly, we drop the y, h notation for t > 0.

Following [Chen et al., 2023, Appendix C.2], we have the following ground truth backward process

dh←
t
=
1

2h←
t + ∇log ph
T −t(h←
t )

dt + d
 
B⊤W t

,

where W t denotes the reversed Wiener process (standard Brownian motion) at time t (see Section 2
for more details).

We define P h
T0 as the ground truth marginal distribution of h←
T0.

For the learned process eyt, we consider eh←
t
= W ⊤
B eyt. For any orthogonal matrix U ∈Rd0×d0, we
define the U transformed version of eh←
t as eh←,U
t
= U ⊤eh←
t . Then the backward SDE for eh←,U
t
is

deh←,U
t
=
h
eh←,U
t
+ esh
U,f(eh←,U
t
, T −t)
i
dt + d
 
U ⊤W ⊤
B W t

,

where

esh
U,f

eh←,U
t
, t

:=
1
σ(t)[−eh←,U
t
+ U ⊤f(Ueh←,U
t
, t)].

We define bP h
T0 as the estimated marginal distribution of eh←,U
T0
from above continuous SDE.

The discretized backward SDE of eh←,U
T0
is

deh⇐,U
t
=
h
eh⇐,U
kµ
+ esh
U,f(eh⇐,U
kµ , T −kµ)
i
dt + d
 
U ⊤W ⊤
B W t

, t ∈[kµ, (k + 1)µ).

We define bP h,dis
T0
as the estimated marginal distribution of eh⇐,U
T0
from above discrete SDE.

Next, we present the auxiliary theoretical results in Appendix F.3.1 to prepare our main proof of
Corollary 3.2.1. Then we give a detailed proof of Corollary 3.2.1 in Appendix F.3.2.

F.3.1
Auxiliary Lemmas

Here we include a few auxiliary lemmas from [Chen et al., 2023] without proofs. Recall the definition
of Lipschitz norm: for a given function f, ∥f(·)∥Lip = supx̸=y(∥f(x) −f(y)∥2/∥x −y∥2).

Lemma F.7 (Lemma 3 of [Chen et al., 2023]). Assume that the following holds

Eh∼Ph∥∇log ph(h)∥2
2 ≤Csh,
λminEh∼Ph[hh⊤] ≥c0,
Eh∼Ph∥h∥2
2 ≤Ch,

where λmin denotes the smallest eigenvalue. We denote

E[ϕ(h, t)] =
Z T

T0

1
σ2(t)Ex∼Pt[ϕ(B⊤x, t)]dt.
(F.19)

Let T0 ≤min{2 log(d0/Csh), 1, 2 log(c0), c0} and T ≥max{2 log(Ch/d0), 1}. Suppose we have

E
WBf(W ⊤
B x, t) −Bq(B⊤x, t)
2

2 ≤ϵ.
(F.20)

52


---Page Break---
Then we have
WBW ⊤
B −BB⊤2

F = O(ϵT0/c0),

and there exists an orthogonal matrix U ∈Rd0×d0, such that:

E
U ⊤f(Uh, t) −q(h, t)
2

2

= ϵ · O

 

1 + T0

c0

h
(T −log T0)d0 · max
t
∥f(·, t)∥2
Lip + Csh
i
+
maxt ∥f(·, t)∥2
Lip · Ch
c0

!

.

Lemma F.8 (Lemma 4 of [Chen et al., 2023]). Assume that Ph is sub-Gaussian and that f(h, t) and
∇log ph
t (h) are Lipschitz continuous with respect to h and t. For any orthogonal matrix U ∈Rd0×d0,
we define

esh
U,f
 

h, t
 :=
1
σ(t)[−h + U ⊤f(Uh, t)].

Assume that we have the latent score matching error-bound

Z T

T0
Eh∼P h
t
esh
U,f
 

h, t

−∇log ph
t
 

h
2

2 dt ≤ϵlatent (T −T0),

where ϵlatent > 0. Then we have the following latent distribution estimation error for the continuous
backward SDE:

TV

P h
T0, bP h
T0

≲
p

ϵlatent (T −T0) +
p

KL (Ph∥N (0, Id0)) · exp(−T),

where bP h
T0 is the marginal distribution of the generated hT0 using the continuous backward SDE.
Furthermore, let bP h,dis
T0
denote the marginal distribution of the generated hT0 using the discretized
backward SDE. Then we have the following latent distribution estimation error for the discretized
backward SDE

TV

P h
T0, bP h,dis
T0


≲
p

ϵlatent(T −T0) +
p

KL (Ph∥N (0, Id0)) · exp(−T) +
p

ϵdis(T −T0),

where

ϵdis =

 maxh
f(h, ·)

Lip
σ (T0)
+
maxh,t
f(h, t)

2
T 2
0

!2

η2

+

 
maxt ∥f(·, t)∥Lip

σ (T0)

!2

η2 max
n
E ∥h0∥2 , d0
o
+ ηd0,

and η is the step size in the backward process.

Lemma F.9 (Lemma 6 of [Chen et al., 2023]). Consider the following discretized SDE with step
size µ satisfying T −T0 = KT µ for some KT ∈N+,

dyt =
1

2 −
1
σ(T −kµ)


ykµdt + dBt,
for
t ∈[kµ, (k + 1)µ),

where y0 ∼N(0, I). Then, for T > 1 and T0 + µ ≤1, we have yT −T0 ∼N
 
0, σ2I

with
σ2 ≤e (T0 + µ).

53


---Page Break---
Lemma F.10 (Lemma 10 in [Chen et al., 2023]). Assume that ∇log ph(h) is Lh-Lipschitz. Then we
have Eh∼Ph ∥∇log ph(h)∥2
2 ≤d0Lh.

F.3.2
Main Proof of Corollary 3.2.1

Proof. Recall the estimation error in Theorem 3.2 is ξ(n, ϵ, L)/(TT0), where

ξ(n, ϵ, L) :=
1
n1/3 · 2(1/ϵ)2L +
1
n1/3 + ϵ2.

• Proof of (i). By the definition of (F.19) and the estimation error in Theorem 3.2, the error bound in
(F.20) equals to ξ(n, ϵ, L)(T −T0)/(TT0) in Lemma F.7. By Lemma F.10, we set Csh = d0Lh.
Then, we have

WBW ⊤
B −BB⊤2

F = O

 
ξ(n, ϵ, L)

c0

!

.

By substituting the value of ξ(n, ϵ, L) and T = O(log n) into the bound above, we deduce

WBW ⊤
B −BB⊤2

F = O

1
c0n1/3 2(1/ϵ)2L +
1
c0n1/3 + ϵ2

c0


.

• Proof of (ii). Recall that maxt ∥f(·, t)∥Lip ≤LT . Furthermore, according to Lemma F.7 and
Lemma F.10, we have

E
U ⊤f(Uh, t) −q(h, t)
2

2 = O(ϵlatent(T −T0)),

where

ϵlatent = ξ(n, ϵ, L)

TT0
· O
T0

c0


(T −log T0)d0 · L2
T + d0Lh

+ L2
T · Ch

c0


.

Following the proof structure in [Chen et al., 2023, Appendix C.4], we get

E
U ⊤f(Uh, t) −q(h, t)
2

2 =
Z T

T0
Eh∼P h
t


U ⊤f(Uh, t) −h

σ(t)
−∇log ph
t (h)


2

2
dt

≤ϵlatent(T −T0).

Following the proof structure in [Chen et al., 2023, Appendix C.4] and setting T = O(log n), we
obtain

TV(P h
T0, bP h,dis
T0
) = e
O
p

ϵlatent (T −T0)


= e
O

 s 1

n1/3 2(1/ϵ)2L +
1
n1/3 + ϵ2

· log n

!

,

where e
O hides the factor about D, d0, d, Ls+, log n, and T −T0

By definition, bP h,dis
T0
= (UWB)⊤
♯bPT0, where bPT0 is the distribution generated by s b
W using the
discretized backward process. This completes the proof of the total variation distance.

• Proof of (iii). We apply Lemma F.9 due to our score decomposition. With the marginal distribution
at time T −T0 and observing µ ≪T0, we obtain the last property.

This completes the proof.

54


---Page Break---
G
Proofs of Section 4

Our proofs are motivated by the observation of low-rank gradient decomposition in transformer-like
models [Alman and Song, 2024b, Gu et al., 2024]. With our simplifications and observations made
in Section 4, we utilize the fine-grained complexity results of transformer and attention [Hu et al.,
2024b, Alman and Song, 2024a,b] and tensor trick (Lemma D.1 and [Diao et al., 2019, 2018]) to
proceed our proofs. Specifically, we approximate DiT training gradients with a series of low-rank
approximations in Appendices G.1.1 to G.1.3, and carefully match the multiplication dimensions so
that the computation of dg2

dW forms a chained low-rank approximation in Appendix G.2.

G.1
Auxiliary Theoretical Results for Theorem 4.1

Here we present some auxiliary theoretical results to prepare our main proof of the Existence of
almost-linear Time Algorithms for ADITGC Theorem 4.1.

G.1.1
Low-Rank Decomposition of DiT Gradients

We start by some definitions. Recall that W ∈Rd×d and W ∈Rd2 denotes the vectorization of
W ∈Rd×d following Definition D.1.

Definition G.1. Let A1, A2 ∈Rd×L be two matrices. Suppose A = A⊤
1 ⊗A⊤
2 ∈RL2×d2. Define
Aj0 ∈RL×d2 as an L × d2 sub-block of A. There are L such sub-blocks in total. For each j0 ∈[L],
define the function u(W)j0 : Rd2 →RL by u(W)j0 := exp(Aj0 W) ∈RL.

Definition G.2. Let A1, A2 ∈Rd×L be two matrices. Suppose A = A⊤
1 ⊗A⊤
2 ∈RL2×d2. Define
Aj0 ∈RL×d2 as an L × d2 sub-block of A. There are L such sub-blocks in total. For every index
j0 ∈[L], consider the function α(W)j0 : Rd2 →R defined by α(W)j0 := ⟨exp(Aj0 W)
|
{z
}
L×1

, 1L
|{z}
L×1
⟩.

Definition G.3. Suppose that α(W)j0 ∈R and u(W)j0 ∈RL are defined as in Definitions G.1
and G.2, respectively. For a fixed j0 ∈[L], consider the function f(W)j0 : Rd2 →RL defined by

f(W)j0 := α(W)−1
j0
|
{z
}
scalar

u(W)j0
| {z }
L×1

.

Define f(W) ∈RL×L as the matrix where the j0-th row is (f(W)j0)⊤.

Definition G.4. For every i0 ∈[d], define the function h(W OV )i0 : Rd2 →RL by

h(W OV )i0 := A⊤
3
|{z}
L×d

(W ⊤
OV )∗,i0
|
{z
}
d×1

.

Here, WOV ∈Rd×d denotes the matrix representation of W OV ∈Rd2, and (WOV )⊤
∗,i0 represents the
i0-th column of W ⊤
OV . Define h(W OV ) ∈RL×d as the matrix where the i0-th column is h(W OV )i0.

Definition G.5. For each j0 ∈[L], we denote f(W)j0 ∈RL as the normalized vector defined
by Definition G.3. For each i0 ∈[d], h(W OV )i0 is defined as per Definition G.4. For every pair
(j0, i0) ∈[L] × [d], define the function c(W)j0,i0 : Rd2 × Rd2 →R by

c(W)j0,i0 := ⟨f(W)j0, h(W OV )i0⟩−Y ⊤
j0,i0,

55


---Page Break---
where (WOV )j0,i0 is the element at the (j0, i0) position of the matrix WOV ∈RL×d. c(·) has matrix
form

c(W)
| {z }
L×d

= f(W)
| {z }
L×L

h(W OV )
|
{z
}
L×d

−Y ⊤
|{z}
L×d
.

With the tensor trick (Appendix D.3), we compute the gradient dg2

dW of the DiT loss as follows:

dg2
dW =
d
dW



1

2

L
X

j0=1

d
X

i0=1
c2
j0,i0(W)



.
(G.1)

(G.1) presents a neat decomposition of dg2

dW . Each term is easy enough to handle. Thus, we arrive at
the following lemma. Let Z[i, ·] and Z[·, j] be the i-th row and j-th column of matrix Z.

Lemma G.1 (Low-Rank Decomposition of DiT Gradient). Let matrix A1, A2, A3, W, WOV , Y and
loss function L follow Definition 4.1, and A := A⊤
1 ⊗A⊤
2 . It holds

dg2
dW =

L
X

j0=1

d
X

i0=1
c(W)j0,i0 A⊤
j0

(II)
z
}|
{
diag (f(W)j) −

(III)
z
}|
{
f(W)j0f(W)⊤
j0


|
{z
}
(I)

h(W OV )i0.
(G.2)

Proof. Let Z[i, ·] and Z[·, j] be the i-th row and j-th column of matrix Z.

With DiT loss Definition 4.1, we have

dg2
dW = 1

2

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0(W)

=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 · dc(W)j0,i0

dW i0

=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 · d ⟨f(W)j0, h(W OV )i0⟩

dW i0

 
By Definition G.5

=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 ·
df(W)j0

dW i
, h(W OV )i0



=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 ·
dα−1(W)j0u(W)j0

dW i
, h(W OV )i0


 
By Definition G.3

=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 ·

*

α(W)−1
j0 · du(W)j0

dW i0
+
dα(W)−1
j0
dW i0
· u(W)j0, h(W OV )i0

+

=

L
X

j0=1

d
X

i=1

d
dW c2
j0,i0c(W)j0,i0 ·

α(W)−1
j0 · du(W)j0

dW i0
−α(W)−2
j0
dα(W)j0

dW i0
· u(W)j0, h(W OV )i0


.

 
By chain rule

For each j0 ∈[L], we have

d (Aj0 W)

dW i0
= Aj0 · dW

dW i0
= (Aj0) [·, i].

56


---Page Break---
Therefore, for each j0 ∈[L], we have

du(W)j0

dW i0
= d exp (Aj0 W)

dW i0

 
By Definition G.1

= exp (Aj0 W) ⊙d Aj0 W

dW i0

 
By entry-wise product rule

= Aj0[·, i] ⊙u(W)j0.
 
By Definition G.1 again

Similarly,

dα(W)j0

dW i0
= d ⟨u(W)j0, 1L⟩

dW i0

 
By Definition G.2

= ⟨Aj0[·, i] ⊙u(W)j0, 1L⟩
 
By entry-wise product rule

= ⟨Aj0[·, i], u(W)j0⟩.
 
By Definition G.1 again

Putting all together, we have

dg2(W)j0,i0

dW i0
= [⟨h(W OV )i0, Aj0[·, i] ⊙f(W)j0⟩−⟨h(W OV )i0, f(W)j0⟩· ⟨Aj0[·, i], f(W)j0⟩] · c(W)j0,i0,

where

⟨h(W OV )i0, Aj0[·, i] ⊙f(W)j0⟩−⟨h(W OV )i0, f(W)j0⟩· ⟨Aj0[·, i], f(W)j0⟩

= A⊤
j0
 
diag (f(W)j0) −f(W)j0f(W)⊤
j0

h(W OV )i0.

This completes the proof.

Observe (G.2) carefully. We see that (I) is diagonal and (II) is low-rank. This provides a hint
for algorithmic speedup through low-rank approximation: If we approximate the other parts with
low-rank approximation and carefully match the multiplication dimensions, we might formulate the
computation of dg2

dW as a chained low-rank approximation.

Surprisingly, such an approach makes computing (G.2) as fast as in almost-linear time. To proceed,
we further decompose (G.2) according to the chain-rule in the next lemma, and then conduct the
approximation term-by-term.

To facilitate our proof, it’s convenient to introduce the following notations.

Definition G.6 (q(·)). Define c(W) ∈RL×d as specified in Definition G.5 and h(W OV ) ∈RL×d as
described in Definition G.4. Define q(W) ∈RL×L by

q(W) := c(W)
| {z }
L×d

h(W OV )⊤
|
{z
}
d×L

.

In addition, q(W)⊤
j0 denotes the j0-th row of q(W), transposed, making it an L × 1 vector.

Definition G.7 (p(·),p1(·), p2(·)). For each index j0 ∈[L], we define p(W)j0 ∈Rn as follows:

p(W)j0 :=
 
diag(f(W)j0) −f(W)j0f(W)⊤
j0

q(W)j0.

57


---Page Break---
We define p(W) ∈RL×L such that p(W)⊤
j0 forms the j0-th row of p(W). In addition, for every
index j0 ∈[L], we define p1(W)j0, p2(W)j0 ∈RL as

p1(W)j0 := diag

f (W)j0


q(W)j0,
p2(W)j0 := f (W)j0 f (W)⊤
j0 q(W)j0,

such that p(W) = p1(W) −p2(W).

p(·) allows us to express dg2

dW in a neat form:

Lemma G.2. Define the functions f(W) ∈RL×L, c(W) ∈Rd×L, h(W OV ) ∈Rd×L, q(W) ∈
RL×L, and p(W) ∈RL×L as specified in Definitions G.3 to G.7, respectively. Let A1, A2 ∈Rd×L

be two given matrices, and define A = A⊤
1 ⊗A⊤
2 . Define g2 according to (O1), and let g2(W)j0,i0
be as described in (G.1). It holds

dg2
dW = vec
 
A1p(W)A⊤
2

.
(G.3)

Proof. By definitions, (G.1) gives

d(g2)j0,i0

dW i0
(G.4)

= cj0,i0 · (⟨f(W)j0 ⊙Aj0,i0, h(W OV )i0⟩
|
{z
}
=A⊤
j0,i diag(f(W )j0)h(W OV )i0

−⟨f(W)j0, h(W OV )i0⟩· ⟨f(W)j0, Aj0,i0⟩)
|
{z
}
=A⊤
j0,i f(W )j0f(W )⊤
j0h(W OV )i0

.

 
By ⟨a ⊙b, c⟩= a⊤diag(b)c for a, b, c ∈RL

Therefore, (G.4) becomes

d(g2)j0,i0

dW i0
= cj0,i0 · (A⊤
j0,i diag(f(W)j0)h(W OV )i0 −A⊤
j0,i f(W)j0f(W)⊤
j0h(W OV )i0)

= cj0,i0 · A⊤
j0,i(diag(f(W)j0) −f(W)j0f(W)⊤
j0)h(W OV )i0.
(G.5)

Then, by definitions of q(·), p(·), we complete the proof.

G.1.2
Low-Rank Approximations of Building Blocks Part I: f(·), q(·), and c(·)

The definitions of p, p1, p2, and Lemma G.2 show that the DiT training gradient dg2

dW involves
entry-wise products of f, q, and c. Therefore, if we approximate these with inner-dimension-matched
low-rank approximations, computing dg2

dW itself becomes a low-rank approximation. In the following
sections, we present low-rank approximations for f, q, and c.

Lemma G.3 (Approximate f(·), Modified from [Alman and Song, 2023]). Let Γ = o(√log L)
and k1 = Lo(1).
Let A1, A2, ∈Rd×L, W ∈Rd×d and f(W) = D−1 exp
 
A⊤
1 XA2

with
D = diag
 
exp
 
A⊤
1 WA2

1L

follows Definitions G.1 to G.3 and G.5. If max
 A⊤
1 W

max ≤
Γ,∥A2∥max

≤Γ, then there exist two matrices U1, V1 ∈RL×k1 such that
U1V ⊤
1 −f(W)

max ≤
ϵ/poly(L). In addition, it takes L1+o(1) time to construct U1 and V1.

Proof. By [Alman and Song, 2023, Theorem 3], we complete the proof.

Lemma G.4 (Approximate c(·)). Assume all numerical values are in O(log L) bits. Let d = O(log L)
and c(W) ∈RL×d follows Definition G.5. There exist two matrices U1, V1 ∈RL×k1 such that
U1V ⊤
1 h(WOV ) −Y ⊤−c(W)

max ≤ϵ/poly(L).

58


---Page Break---
Proof of Lemma G.4.

U1V ⊤
1 h(WOV ) −Y ⊤−c(W)

max =
U1V ⊤
1 h(WOV ) −Y ⊤−(f(W)h(WOV ) −Y ⊤)

max
 
By Definition G.5

=

U1V ⊤
1 −f(W)

h(WOV )

max
≤ϵ/poly(L).
 
By [Alman and Song, 2023, Theorem 3]

Lemma G.5 (Approximate q(·)). Let k2 = Lo(1), c(·) ∈RL×d follow Definition G.5 and let
q(W) := c(W)h(W OV )T ∈RL×L (follow Definition G.6). There exist two matrices U2, V2 ∈
RL×k2 such that
U2V ⊤
2 −q(W)

max ≤ϵ/poly(L). In addition, it takes L1+o(1) time to construct
U2, V2.

Proof of Lemma G.5. Our proof is built on [Alman and Song, 2023, Lemma D.3].

Let eq(·) denote an approximation to q(·).

By Lemma G.4, U1V ⊤
1 h(WOV ) −Y approximates c(W) up to accuracy ϵ = 1/poly(L).

Thus, by setting eq(W) = h(WOV )
 
U1V ⊤
1 h(WOV ) −Y
⊤, we find a low-rank form for eq(·):

eq(W) = h(WOV ) (h(WOV ))⊤V1U ⊤
1 −h(WOV )Y ⊤,

such that

∥eq(W) −q(W)∥max =
h(WOV )
 
U1V ⊤
1 h(WOV ) −Y
⊤−h(WOV )Y ⊤
max
≤d ∥h(WOV )∥max
U1V ⊤
1 h(WOV ) −Y −c(W)

max
≤ϵ/poly(L).

By k1, d = Lo(1), compute (h(WOV ))⊤
|
{z
}
d×L

V1
|{z}
L×k1

U ⊤
1
|{z}
k1×L

takes only L1+o(1) time. This completes the

proof.

G.1.3
Low-Rank Approximations of Building Blocks Part II: p(·)

Now, we use the low-rank approximations of f, q, c to construct low-rank approximations for
p1(·), p2(·), p(·).

Lemma G.6 (Approximate p1(·)). Let k1, k2 = Lo(1). Suppose U1, V1 ∈RL×k1 approximates
f(W) ∈RL×L such that
U1V ⊤
1 −f(W)

max ≤ϵ/poly(L), and U2, V2 ∈RL×k2 approximates
the q(W) ∈RL×L such that
U2V ⊤
2 −q(W)

max ≤ϵ/poly(L). Then there exist two matrices
U3, V3 ∈RL×k3 such that
U3V ⊤
3 −p1(W)

max ≤ϵ/poly(L). In addition, it takes L1+o(1) time
to construct U3, V3.

Proof of Lemma G.6. By tensor trick, we construct U3, V3 as tensor products of U1, V1 and U2, V2,
respectively, while preserving their low-rank structures. Then, we show the low-rank approximation
of p1(·) with bounded error by Lemma G.3 and Lemma G.5.

Let ⊘be column-wise Kronecker product such that A ⊘B := [A[·, 1] ⊗B[·, 1] | . . . | A[·, k1] ⊗
B[·, k1]] ∈RL×k1k2 for A ∈RL×k1, B ∈RL×k2.

Let ef(W) := U1V T
1 and eq(W) := U2V T
2 denote matrix-multiplication approximations to f(W) and
q(W), respectively.

59


---Page Break---
For the case of presentation, let U3 =

L×k1
z}|{
U1 ⊘

L×k2
z}|{
U2 and V3 =

L×k1
z}|{
V1 ⊘

L×k2
z}|{
V2 . It holds
U3V ⊤
3 −p1(W)

max
=
U3V ⊤
3 −f(W) ⊙q(W)

max

 
By p1(W) = f(W) ⊙q(W)

=
(U1 ⊘U2) (V1 ⊘V2)⊤−f(W) ⊙q(W)

max
=
 
U1V ⊤
1

⊙
 
U2V ⊤
2

−f(W) ⊙q(W)

max
= ∥ef(W) ⊙eq(W) −f(W) ⊙q(W)∥max

≤∥ef(W) ⊙eq(W) −ef(W) ⊙q(W)∥max
|
{z
}
≤ϵ/poly(L)

+ ∥ef(W) ⊙q(W) −f(W) ⊙q(W)∥max
|
{z
}
≤ϵ/poly(L)

≤ϵ/poly(L).
 
By Lemma G.3 and Lemma G.5

Computationally, by k1, k2 = Lo(1), computing U3 and V3 takes L1+o(1) time. This completes the
proof.

Lemma G.7 (Approximate p2(·)). Let k1, k2, k4 = Lo(1). Let p2(W) ∈RL×L follow Definition G.7
such that its j0-th column is p2(W)j0 = f(W)j0f(W)⊤
j0q(W)j0 for each j0 ∈[L]. Suppose
U1, V1 ∈RL×k1 approximates the f(X) such that
U1V ⊤
1 −f(W)

max ≤ϵ/poly(L), and U2, V2 ∈
RL×k2 approximates the q(W) ∈RL×L such that
U2V ⊤
2 −q(W)

max ≤ϵ/poly(L). Then there
exist matrices U4, V4 ∈RL×k4 such that
U4V ⊤
4 −p2()

max ≤ϵ/poly(L). In addition, it takes
L1+o(1) time to construct U4, V4.

Proof of Lemma G.7. From Definition G.7,

p2(W)j0 :=

(II)
z
}|
{
f (W)j0 f (W)⊤
j0 q(W)j0
|
{z
}
(I)

.

For (I), we show its low-rank approximation by observing the low-rank-preserving property of the
multiplication between f(·) and q(·) (from Lemma G.3 and Lemma G.5). For (II), we show its
low-rank approximation by the low-rank structure of f(·) and (I).

Part (I).
We define a function r(W) : Rd2 →RL such that the j0-th component r(W)j0 :=
(f(W)j0)⊤q(W)j0 for all j0 ∈[L]. Let er(W) denote the approximation of r(W) via decomposing
into f(·) and q(·):

er(W)j0 :=
D
ef(W)j0, eq(W)j0
E
=
 
U1V ⊤
1

[j0, ·] ·
 
U2V ⊤
2

[j0, ·]
⊤

= U1[j0, ·] V ⊤
1
|{z}
k1×L

V2
|{z}
L×k2

(U2[j0, ·])⊤,
(G.6)

for all j0 ∈[L]. This allows us to write p2(W) = f(W) diag(r(W)) with diag(er(W)) denoting a
diagonal matrix with diagonal entries being components of er(W).

Part (II).
With r(·), we approximate p2(·) with ep2(W) = ef(W) diag(er(W)) as follows.

Since ef(W) has low rank representation, and diag(er(W)) is a diagonal matrix, ep2(·) has low-rank
representation by definition. Thus, we set ep2(W) = U4V T
4 with U4 = U1 and V4 = diag(er(W))V1.
Then, we bound the approximation error
U4V ⊤
4 −p2(W)

max

60


---Page Break---
= ∥ep2(W) −p2(W)∥max

= max
j0∈[L]

 ef(W)j0er(W)j0 −f(W)j0r(W)j0

max

≤max
j0∈[L]

h ef(W)j0er(W)j0 −f(W)j0r(W)j0

max +
 ef(W)j0er(W)j0 −f(W)j0r(W)j0

max

i

 
By triangle inequality

≤ϵ/poly(L).

Computationally, computing V ⊤
1 V2 takes L1+o(1) time by k1, k2 = Lo(1). Once we have V ⊤
1 V2
precomputed, (G.6) only takes O(k1k2) time for each j0 ∈[L]. Thus, the total time is O (Lk1k2) =
L1+o(1). Since U1 and V1 takes L1+o(1) time to construct and V4 = diag(er(W))
|
{z
}
L×L

V1
|{z}
L×k1

also takes

L1+o(1) time, U4 and V4 takes L1+o(1) time to construct. This completes the proof.

G.2
Proof of Theorem 4.1

Proof of Theorem 4.1. By the definitions of matrices p(·), p1(·) and p2(·) (Definition G.7), we have

p(W) = p1(W) −p2(W).

By Lemma G.2, we have

dg2
dW = vec
 
A1p(W)A⊤
2

.
(G.7)

To show the existence of L1+o(1) algorithms for DiT backward computation Problem 1, we prove
fast low-rank approximations for A1p1(W)A⊤
2 and A1p2(W)A⊤
2 as follows.

Let ep1(W), ep2(W) denote the approximations to p1(W), p2(W), respectively.

By Lemma G.6, it takes L1+o(1) time to construct U3, V3 ∈RL×k3 such that

A1ep1(W)A⊤
2 = A1U3V ⊤
3 A⊤
2 .

Then, computing A1
|{z}
d×L
U3
|{z}
L×k3

V ⊤
3
|{z}
k3×L

A⊤
2
|{z}
L×d

takes L1+o(1) due to the fact that d, k1k3 = Lo(1).

Therefore, total running time for A1p1(W)A⊤
2 is L · Lo(1) = L1+o(1).

For the same reason (by Lemma G.7), total running time for A1p2(W)A⊤
2 is L · Lo(1) = L1+o(1).

Lastly, we have

∂g2
∂W −eG(W )

max
=
vec
 
A1ep(W)A⊤
2

−vec
 
A1ep(W)A⊤
2

max

 
By Lemma G.2

=
 
A1ep(W)A⊤
2

−
 
A1ep(W)A⊤
2

max

 
By definition, ∥A∥max := maxi,j |Aij| for any matrix A

≤
 
A1 [p1(W) −ep1(W)] A⊤
2

max +
 
A1 [p2(W) −ep2(W)] A⊤
2

max
 
By Definition G.7 and triangle inequality

≤∥A1∥∞∥A2∥∞(∥(p1(W) −ep1(W))∥max + ∥(p2(W) −ep2(W))∥max)
 
By the sub-multiplicative property of ∥·∥∞


≤ϵ/poly(L).
 
By Lemma G.6 and Lemma G.7

Set ϵ = 1/poly(L). We complete the proof.

61


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope?
Answer: [Yes]
Justification: Our contributions and scope in Section 3 and Section 4 are reflected by the claims in
abstract and introduction.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the
paper.
• The abstract and/or introduction should clearly state the claims made, including the contribu-
tions made in the paper and important assumptions and limitations. A No or NA answer to this
question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the
results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not
attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We discuss the limitations in Section 5.
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
that individual actions in favor of transparency play an important role in developing norms
that preserve the integrity of the community. Reviewers will be specifically instructed to not
penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?
Answer: [Yes]
Justification: Yes. We include our proofs in the appendix and have made every effort to ensure the
correctness of our theoretical results.

62


---Page Break---
Guidelines:

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
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by
the reviewers: Making the paper reproducible is important, regardless of whether the code and
data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to
make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For
example, if the contribution is a novel architecture, describing the architecture fully might
suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary
to either make it possible for others to replicate the model with the same dataset, or provide
access to the model. In general. releasing code and data is often one good way to accomplish
this, but reproducibility can also be provided via detailed instructions for how to replicate the
results, access to a hosted model (e.g., in the case of a large language model), releasing of a
model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to
provide some reasonable avenue for reproducibility, which may depend on the nature of the
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
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.

63


---Page Break---
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible,
so “No” is an acceptable answer. Papers cannot be rejected simply for not including code,
unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce
the results. See the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
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

Answer: [NA]

Justification: This is a formal theory work without experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is
necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [NA]

Justification: This is a formal theory work without experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the main
claims of the paper.
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
• If error bars are reported in tables or plots, The authors should explain in the text how they
were calculated and reference the corresponding figures or tables in the text.

64


---Page Break---
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the experi-
ments?

Answer: [NA]

Justification: This is a formal theory work without experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud
provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experi-
mental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the
experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it
into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: Yes. We follow the code of ethics in this work.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation
from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration
due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?

Answer: [Yes]

Justification: This theoretical work aims to shed light on the foundations of diffusion generative
models and is not anticipated to have negative social impacts.

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

65


---Page Break---
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms
for monitoring misuse, mechanisms to monitor how a system learns from feedback over time,
improving the efficiency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release
of data or models that have a high risk for misuse (e.g., pretrained language models, image
generators, or scraped datasets)?
Answer: [NA]
Justification: This is a formal theory work without experiments.
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
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of
that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the
derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset’s
creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their sub-
missions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset is
used.

66


---Page Break---
• At submission time, remember to anonymize your assets (if applicable). You can either create
an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human
subjects.
• Including this information in the supplemental material is fine, but if the main contribution of
the paper involves human subjects, then as much detail as possible should be included in the
main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or
other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals
(or an equivalent approval/review based on the requirements of your country or institution) were
obtained?
Answer: [NA]
Justification: This is a formal theory work without experiments.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human
subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be
required for any human subjects research. If you obtained IRB approval, you should clearly
state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and
locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines
for their institution.
• For initial submissions, do not include any information that would break anonymity (if applica-
ble), such as the institution conducting the review.

67


---Page Break---
