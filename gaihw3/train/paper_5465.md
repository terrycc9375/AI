Efficiently Learning Significant Fourier Feature Pairs
for Statistical Independence Testing

Yixin Ren1, Yewei Xia1,4, Hao Zhang3,∗, Jihong Guan2, Shuigeng Zhou1,∗

1Shanghai Key Lab of Intelligent Information Processing, and School of
Computer Science, Fudan University, Shanghai, China
2Department of Computer Science and Technology, Tongji University, Shanghai, China
3SIAT, Chinese Academy of Sciences, Shenzhen, China
4Machine Learning Department, MBZUAI, Abu Dhabi, UAE
{yxren21, ywxia23}@m.fudan.edu.cn, h.zhang10@siat.ac.cn
jhguan@tongji.edu.cn, sgzhou@fudan.edu.cn

Abstract

We propose a novel method to efficiently learn significant Fourier feature pairs for
maximizing the power of Hilbert-Schmidt Independence Criterion (HSIC) based
independence tests. We first reinterpret HSIC in the frequency domain, which
reveals its limited discriminative power due to the inability to adapt to specific
frequency-domain features under the current inflexible configuration. To remedy
this shortcoming, we introduce a module of learnable Fourier features, thereby
developing a new criterion. We then derive a finite sample estimate of the test
power by modeling the behavior of the criterion, thus formulating an optimization
objective for significant Fourier feature pairs learning. We show that this optimiza-
tion objective can be computed in linear time (with respect to the sample size n),
which ensures fast independence tests. We also prove the convergence property
of the optimization objective and establish the consistency of the independence
tests. Extensive empirical evaluation on both synthetic and real datasets validates
our method’s superiority in effectiveness and efficiency, particularly in handling
high-dimensional data and dealing with large-scale scenarios.

1
Introduction

Testing for independence is a crucial and challenging task in machine learning and statistics, with
wide-range applications in causal inference [16, 31], feature selection [6] and deep learning [23, 42].
Its primary objective is to determine whether two random variables, X and Y are independent, based
on the observations of the underlying joint distribution PXY . While traditional independence tests,
such as Pearson’s correlation coefficient [9] and Kendall’s τ, can only detect monotonic relationships
between low-dimensional variables, more modern tests [26, 43, 7, 25, 27, 35, 19, 20] aim to deal
with complex non-linear interactions in much more challenging higher-dimensional space [45, 29].

One class of nonlinear dependence measures [3, 15] aims to capture distributional characteristics using
kernel embeddings [13], primarily derived from the cross-covariance operators in the reproducing
kernel Hilbert space (RKHS). Among them, Hilbert-Schmidt Independence Criterion (HSIC) [14]
is the most popular one. It utilizes the squared Hilbert-Schmidt norm to detect dependence and
exhibits outstanding performance across various data contexts by choosing suitable kernels. On the
other hand, some other fundamental nonlinear dependence measures employ characteristic functions

∗Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
to detect the smoothed discrepancy between the joint distribution and the product of marginals.
By employing appropriate characteristic functions, the statistic [39, 40] computes the covariance
between distances of variable pairs. It has been demonstrated that these distance-based methods
are equivalent to HSIC with specific kernels [33]. However, all these measures suffer from the
drawback of requiring quadratic time (w.r.t. the sample size n) to compute the feature covariance and
necessitating fixed kernel or distance functions, rendering them impractical on large-scale datasets
due to the unaffordable time cost and lacking flexibility in handling complex scenarios.

To address these challenges, a multitude of works grounded on these measures have emerged. Upon
HSIC, [44] proposes some linear-time tests including a block-averaged statistic, a statistic with
Nyström approximation, and one with finite-dimensional feature mappings using random Fourier
features (RFF) [28]. For convenience, these tests are referred to as BHSIC, NyHSIC, and FHSIC,
respectively. FHSIC and NyHSIC are observed to have a considerable advantage over BHSIC.
However, a remaining drawback of these methods is that the features are not learnable. Therefore,
these methods lack enough adaptability to complex settings, thus leading to performance degradation.

In addition to time efficiency, another research direction [1, 30] aims to make independence tests
adaptive to better capturing distributional distinctions. These methods either select/combine appro-
priate kernels from a predefined set or learn parameterized kernels. Nonetheless, their criteria still
inherit the quadratic time complexity of HSIC, thus cannot be readily applied to large-scale data.

Furthermore, some approaches [17, 32] try to address both challenges simultaneously. For instance,
HSICAgg [32] suggests combining several kernels from a predefined set (e.g. kernels with different
preset bandwidths) and aggregating the test results for improving performance. Additionally, an
incomplete U-statistic of HSIC is proposed to ensure computational efficiency. Nevertheless, selecting
from a predefined set of kernels imposes limitations on flexibility, and in cases where scaling
optimization is required on each dimension, the number of kernel pairs escalates exponentially. Also,
NFSIC [18] proposes to combine a time-efficient technique called analytic kernel embeddings [8, 17]
and learn the important local distributional features. However, its learning objective is merely a lower
bound of test power and demands a substantial number of samples to ensure accuracy.

In this paper, we propose a novel test method that flexibly learns distributional features while
maintaining high efficiency. We first reinterpret HSIC from a frequency-domain perspective, then
we point out its potential shortcomings with an elaborate example and indicate corresponding
improvement directions. Finally an central optimization objective is derived by directly modeling test
power, which can be computed in linear time while maximizing the test performance. Comparing
with [30] that also addresses the kernel learning problem in independence testing with a time/space
complexity of O(n2), our criteria for learning are designed to have a complexity of O(n) for both
space and time. Consequently, the whole test framework can efficiently handle large-scale data.

Contributions. In summary, the contributions of the work are as follows: 1) We propose a novel
approach that efficiently learns significant Fourier feature pairs for maximizing the power of HSIC-
based independence tests. 2) We design an optimization objective that can be computed in linear time,
which is derived by directly modeling test power. 3) We theoretically establish the non-asymptotic
convergence property of the optimization objective and demonstrate the consistency of our method.
4) We conduct extensive experiments on both synthetic and real data, showcasing its superiority
in effectiveness and efficiency in handling high-dimensional data (e.g. image data) and addressing
large-scale scenarios.

Outline. The rest of the paper is organized as follows: Sec. 2 reviews HSIC-based statistical
independence tests. Sec. 3 reinterprets HSIC from a frequency-domain perspective, and explain its
potential shortcomings with an elaborate example and indicate corresponding improvement directions.
Sec. 4 designs an optimization objective by directly modeling test power, which can be computed
in linear time. Sec. 5 presents the theoretical analysis and Sec. 6 evaluates the performance of the
proposed method on synthetic and real dataset. We conclude the paper in Sec. 7.

2
Preliminaries and Notations

We begin by introducing notions and reviewing the hypothesis testing framework for independence
tests. Let X × Y be separable metric space, typically Rdx × Rdy. PXY denotes a Borel probability
measure defined on X × Y, while PX and PY denote the respective marginal distributions. Given n

2


---Page Break---
independent and identically distributed (i.i.d) samples Z := (X, Y ) = {(xi, yi)}n
i=1 with distribution
PXY , we aim to test whether X, Y are independent (i.e., X ⊥⊥Y ). This corresponds to a hypothesis
testing problem formulated as H0 : PXY = PXPY versus H1 : PXY ̸= PXPY .

The testing procedure is as follows: First, define the statistic ρ and calculate its estimated value
using the samples. Then, choose a significance level α (typically set to 0.05), which represents the
probability that the sampling of ρ under H0 is at least as extreme as the observed value. Finally, the
null hypothesis H0 is rejected if the p-value is not greater than α.

Two types of errors may occur in this procedure. Type I error occurs when H0 is falsely rejected,
while Type II error happens when H0 is incorrect but not rejected. A good test [43] needs to control
Type I error within α while maximizing the testing power (1−Type II error rate).

For independence tests, a commonly used statistic is HSIC, defined as follows:
Definition 1. [14]. Let F be an RKHS with kernel k : X × X 7→R and G be a second RKHS on Y
with kernel l : Y × Y 7→R, the HSIC between X and Y , denoted as HSIC(X, Y ) is defined as

E

k(X, X′)l(Y, Y ′)

+E

k(X, X′)

E

l(Y, Y ′)

−2EX′Y ′
EXk(X, X′)EY l(Y, Y ′)

,
(1)

where (X′, Y ′) is a independent copy of (X, Y ). An estimator of HSIC(X, Y ) is given by

HSICb(Z) := 1

n2
X

i,j
kijlij + 1

n4
X

i,j,q,r
kijlqr −2 1

n3
X

i,j,q
kijliq = 1

n2 Tr(KHLH),
(2)

where kij := k(xi, xj), lij := l(yi, yj) are the entries of the n × n kernel matrices K, L respectively,
H = I −1

n11T is the centering matrix and 1 is a vector of ones.

3
Revisiting HSIC from Frequency Domain Perspective

We denote F as the Fourier transform, and F−1 as its inverse. When the kernels k, l are translation-
invariant, i.e., there exist functions ψ, ψk, ψl such that for all (x, x′) ∈X × X and (y, y′) ∈Y × Y,

ψ(x −x′, y −y′) = ψk(x −x′)ψl(y −y′) = k(x, x′)l(y, y′).
(3)

Then, according to the results of [36, Corollary 4], the HSIC with function ψ can be formulated as

HSIC(X, Y ) =
Z

Rdx×Rdy

ϕPXY (ω) −ϕPXPY (ω)
2(F−1ψ)(ω)dω,
(4)

where ω = (ωx, ωy) ∈Rdx × Rdy, ωx, ωy are the frequencies of X and Y respectively, and

ϕPXY (ω) :=
Z
e−i(ωT
x x+ωT
y y)dPXY , ϕPXPY (ω) :=
Z
e−iωT
x xdPX

 Z
e−iωT
y ydPY


(5)

are the characteristic functions of PXY and PXPY , respectively. Intuitively, Eq. (4) means that HSIC
can be understood as the difference between the joint distribution and the product of the marginal
distributions in the frequency domain, with different weights (F−1ψ)(ω) being attached to different
frequencies, which are determined by the kernel function. When F−1ψ is almost everywhere non-
zero, it can be shown that the kernel is characteristic [36, 10]. The characteristic condition ensures that
the criterion is discriminative for discrepancies at almost all frequencies. However, with inappropriate
choices of F−1ψ, the differences may not be significant enough. We explain this with an example:

Example. Consider the Sinusoid model that X × Y := [−π, π]2 and (X, Y ) ∼pxy(x, y) ∝
1 + sin(ω0x) sin(ω0y), where pxy is the probability density function and ω0 is a positive integer.
Combining Eq. (5), we can calculate that ϕPXPY (ω) = δ(ωx)δ(ωy) and ϕPXY (ω) = δ(ωx)δ(ωy) +
[δ(ωx + ω0) + δ(ωx −ω0)][δ(ωy + ω0) + δ(ωy −ω0)], where δ is the Dirac delta function, thus the
difference between them only relies on the frequency ω0. When the Gaussian kernels with width
√

2λx
and
√

2λy are used, i.e., k(x, x′) = exp(−∥x −x′∥2
2/(4λ2
x)), l(y, y′) = exp(−∥y −y′∥2
2/(4λ2
y)),
then the inverse Fourier transform of ψ is (F−1ψ)(ωx, ωy) = π−1λxλy exp(−(λ2
xω2
x + λ2
yω2
y)).
Hence HSIC(X, Y ) = 4π−1λxλy exp(−(λ2
x + λ2
y)ω2
0) whose maximum is taken at λ∗
x = λ∗
y =
1/(
√

2ω0), indicating that the widths need to be adjusted to focus on some specific frequencies.
If the common setting [14, 44] is adapted, which uses mid-widths (i.e., the median distance does

3


---Page Break---
not change with ω0 since the marginal distributions do not change with ω0), then the criterion will
exponentially decline to 0 as ω0 increases. In contrast, the criterion using the adaptive optimization
width (1/ω0, 1/ω0) decreases at a rate of O(ω−2
0 ), which is a considerable improvement.

This example illustrates the loss of the discriminatory power of the criterion when an inappropriate
F−1ψ is chosen. The discriminatory power of the criterion heavily impacts the sample size required
for the test to obtain significant results in practice, and existing inflexible configurations may lead to
inadequate test power in the presence of reasonably large sample sizes. Consequently, it is important
to design learnable F−1ψ. To this end, we subsequently design a learnable objective and let it
be optimized in a data-driven manner. Before this, we provide an approach to make the criterion
be computed efficiently. This can be achieved by sampling in the frequency domain. Formally,
a finite-dimensional approximation in the frequency domain of the integral in Eq. (4) is given as
follows:

HSICω(X, Y ) :=
1
DxDy

Dx
X

i=1

Dy
X

j=1

ϕPXY (ωx;i, ωy;j) −ϕPXPY (ωx;i, ωy;j)
2,
(6)

where {ωx;i}Dx
i=1, {ωy;j}Dy
j=1 are sampled independently with the measure F−1ψk, F−1ψl, respec-
tively. Note that F−1ψ is a product measure, i.e., F−1ψ = (F−1ψk) ⊗(F−1ψl). This type of
approximation is also called random Fourier features (RFF) [28] that had been applied to various
kernel algorithms. We will incorporate this technique to efficiently perform computation later.

4
Learning Significant Fourier Feature Pairs

4.1
HSIC with Learnable Fourier Feature Pairs

To design F−1ψ, we need to make sure that supp(F−1ψ) = Rdx × Rdy to meet the characteristic
condition and that its integral over the full space is 1 to ensure it is a probability measure. Also, for
practical utility, F−1ψ should embody a familiar probability density function, facilitating sampling
procedures. Fortunately, a versatile array of options emerges through the judicious selection of
kernels 2 with adjustable parameters. Take kernel k as an example, some commonly used kernels
are listed in Tab. 1, and their inverse Fourier transforms are listed simultaneously. Additionally, to

Table 1: Some popular kernels (parameterized by σ, Σ) with corresponding density functions.

Kernel
ψk(∆)
F −1ψk(ω)
Tθk(x)
pk(ω)

Gaussian
e−
∥∆∥2
2
2σ2
(2π)−dx/2σe−σ2∥ω∥2
2/2
x/σ
(2π)−dx/2e−∥ω∥2
2/2

Laplace
e−∥∆∥1

σ
q

2
π
Q

d
σ
σ2+ω2
d
x/σ
q

2
π
Q

d
1
1+ω2
d
Mahalanobis
e−1

2 ∆T Σ−1∆
(2π)−dx/2|Σ|−1/2e−ωT Σ−1ω/2
Σ1/2x
(2π)−dx/2e−∥ω∥2
2/2

be able to apply gradient-based optimization techniques, we invoke a method that disentangle the
sampled objects and the learnable parameters. Specifically, we leverage a variable transform Tθk (a
bijection function parameterized with θk) to convert the probability measure F−1ψk into a simple
distribution (e.g. a standard Gaussian distribution) pk(ω). Simultaneously, we relocate the learnable
component onto X. Consequently, we can focus on learning parameterized transformations Tθk and
simplifying the computation by enabling sampling directly from pk(ω).

Remark. The above scheme provides a broader form for designing. The mapping Tθk can be viewed
as a feature extractor, which makes it possible to flexibly combine models (e.g., neural network) thus
incorporating deep kernel [24] into the framework. Also, it should be noted that the single kernel
example can also be extended to multi-kernel setting [11] by executing the procedure for each kernel.

Next, we obtain the learnable independence criterion and utilize the sampling technique as in Eq. (6)
to compute efficiently. Note that for simplicity, we take the same value for both Dx and Dy in Eq. (6)
by default. By the definition, the kernel function can be expressed as

ψk (Tθkx −Tθkx′) = F[F−1ψk(ω)] =
Z
e−iωT (Tθk x−Tθk x′)pk(ω)dω.
(7)

2The bounded, continuous, translation-invariant kernel satisfies the characteristic condition [12].

4


---Page Break---
By applying the frequency sampling technique, we obtain the approximation as

ψ(ω)
k
(Tθkx −Tθkx′) := 2

D

D/2
X

j=1
e−iωT
k;j(Tθk x−Tθk x′) = 2

D

D/2
X

j=1
cos
 
ωT
k;j(Tθkx −Tθkx′)

,
(8)

where {ωk;j}D/2
j=1 are sampled independently with distribution pk(ω) and the last equation is because
the kernel function is real. To get a more computationally tractable form, we define

Λk(x) :=

r

2
D

h
cos
 
ωT
1 Tθkx

, sin
 
ωT
1 Tθkx

, ..., cos
 
ωT
D/2Tθkx

, sin
 
ωT
D/2Tθkx
i
,
(9)

called learnable RFF of k then Eq. (8) becomes ψ(ω)
k
(Tθkx −Tθkx′) = Λk(x)Λk(x′)T . The ex-
pression with a similar form is also given in [44], with the difference that we have added learnable
parts. For Y , we define the corresponding symbols by substituting k for l and x for y. Also, for
convenience, we default to keeping Y and X the same number of samples D from here on. Then the
HSIC with learnable RFF pairs can be obtained by replacing k, l in Eq. (1) to ψ(ω)
k
, ψ(ω)
l
. Also, the
corresponding estimator with sample Z can be obtained by replacing K, L in Eq. (2) to the matrices
ΛXΛT
X, ΛY ΛT
Y , where ΛX := [Λk(x1); ...; Λk(xn)]n×D and so as define for ΛY . As a result,

HSICω(Z) := 1

n2 Tr(ΛXΛT
XHΛY ΛT
Y H) = 1

n2 Tr(ΛT
XHΛY ΛT
Y HΛX) = 1

n2 ∥ΛT
XcΛY c∥2
F ,
(10)
where ΛXc := HΛX, ΛY c := HΛY . The time complexity is analyzed as follows. Since the
computation of the mapping Tθkx depends on the specific design, here we default to analyzing the
kernel case shown in Tab. 1. In this case, computing ΛX, ΛY requires O
 
nD(dx + dy)

time. Then
calculate ΛXc, ΛY c cost O(nD). After that, calculate HSICω(Z) cost O(nD2). Hence, the overall
time complexity is O
 
nD(dx + dy + D)

, i.e. the running time is linear with n.

4.2
Linear-time Optimization Objective

Next, we model the behavior of HSICω(Z) to obtain an optimization objective for maximizing
the power of the test. By utilizing the property that HSICω(Z) is a V-statistic, we can extend the
results [14, Theorem 1, 2] for HSICω(Z), as shown in the following proposition with the proof
given in the Appendix. To simplify, we denote (xi, yi) as zi to represent the i-th sample and denote
ψ(ω)
k
(Tθkxt −Tθkxu) as k(ω)
tu and ψ(ω)
l
(Tθlyt −Tθlyu) as l(ω)
tu .

Proposition 1 (Asymptotics). Let h(ω)
ijqr := 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vw −2k(ω)
uv l(ω)
tv , where the
sum represents all ordered quadruples (t, u, v, w) drawn without replacement from (i, j, q, r). Then,
Under the null hypothesis H0, HSICω(Z) coverages in distribution to

nHSICω(Z)
d−→

∞
X

l=1
λlχ2
1l, λlgl(zj) =
Z

zi,zq,zr
h(ω)
ijqrgl(zi)dFzi,zq,zr,
(11)

where χ2
11, χ2
12, ... are independent χ2
1 variates and λl is the solution to the eigenvalue problem as in
the right of Eq. (11). Also, under the alternative H1, HSICω(Z) converges in distribution as

n
1
2

HSICω(Z) −EZHSICω(Z)
 d−→N(0, σ2
ω), σ2
ω := 16
h
Ei(Ej,q,rh(ω)
ijqr)2 −
 
EZh(ω)
ijqr
2i
(12)

with the simplified notation Ej,q,r := Ezj,zq,zr and EZ := Ezi,zj,zq,zr.

According to Proposition 1, the power of the test with HSICω can be formulated by

PH1 (nHSICω(Z) > rω) →Φ
nEZHSICω(Z) −rω
√nσω


,
(13)

where Φ is the standard normal CDF and rω is the threshold, i.e. (1 −α)-quantile of distribution
given in Eq. (11) that exactly controls Type I error rate to the nominal level α. Hence, to maximize
the power of the test, a natural criterion is [nEZHSICω(Z) −rω]/(√nσω). Next, we provide its
estimation which can be computed in linear time.

5


---Page Break---
We first consider obtaining the estimator of the numerator part. For the term EZHSICω(Z), we can
estimate it with HSICω(Z) as in Eq. (10). The estimation of the threshold rω poses a challenge,
primarily stemming from the lack of an explicit expression for the distribution of the infinite sum
of chi-square variables. One avenue to address this challenge involves employing the permutation
method [2, 38] to simulate the distribution under H0. However, this method necessitates a significant
number of shuffles to accurately approximate the distribution. Furthermore, even with the imple-
mentation of parallel schemes, it incurs memory costs proportional to the number of permutations,
rendering it impractical for resource-constrained scenarios. Here, we adopt a lightweight approach
in practice, leveraging the gamma approximation as proposed by [14]. A gamma distribution is
uniquely determined by its first and second-order moments. For these two moments, we present their
corresponding linear-time estimators in Theorem 1. As a result, we can obtain the (1 −α)-quantile of
the gamma distribution, denoted as c
α, with estimated parameters γ := E2
0/V0, β := V0/E0 in linear
time. Formally, with the term E0 and V0 defined in Theorem 1, c
α is calculated by

H0 : nHSICω(Z) ∼xγ−1e−x/β

βγΓ(γ)
, γ = E2
0
V0
, β = V0

E0
,
Z
d
cα

β

0

xγ−1e−x

Γ(γ)
dx = 1 −α,
(14)

where Γ(·) is the gamma function. By combining the way to estimate the gradients of c
α [30], we
enable it for gradient-based optimization with automatic differentiation framework. As a result, we
obtain a linear-time differentiable estimator of the numerator part.
Theorem 1 (Linear-Time Estimators). Under H0, the estimators of mean and variance with bias of
O(n−1) to EZ[nHSICω(Z)] and VarZ[nHSICω(Z)], denote as E0 and V0, respectively, are given by

E0 := [1T Λ·2
Xc1][1T Λ·2
Y c1]
(n −1)2
, V0 :=
2n(n −4)(n −5)
(n −1)(n −2)(n −3)
[1T (ΛT
XcΛXc)·21][1T (ΛT
Y cΛY c)·21]
n4
,

(15)
where ()·2 is the entry-wise matrix power. Both E0 and V0 can be calculated in O(nD2) time.

For the remain term σω, we estimate it with bσω that bσ2
ω := 16
 1

n
P

i( 1

n3
P

j,q,r h(ω)
ijqr)2−HSIC2
ω(Z)

.

To calculate P

j,q,r h(ω)
ijqr, the straightforward way is to compute each item h(ω)
ijqr, which requires
total O(n4) of computation. Here we provide a way to enable it to be calculated in linear time by
obtaining a matrix expression. The main result is given by
X

j,q,r
h(ω)
ijqr = 1

2

n1T A1 + n2(A1)i + (1T C)Bi + (1T B)Ci −nEi −nFi −nDi −1T D

,

(16)
where the definition of variables A to F with the calculation cost are given in the Fig. 1 and the
derivation of Eq. (16) is given in the Appendix. By checking the complexity of the remaining matrix
operations in Eq. (16), all the elements with index i can be calculated in O(nD2). Combining
the results obtained before that HSICω(Z) can also be calculated in O(nD2), thus calculating the
term bσω cost O(nD2) time. As a result, we obtain the overall linear-time optimization objective
J := [HSICω(Z) −c
α/n]/bσω, which is a clear contrast to the existing quadratic-time schemes [30].

Time Complexity: 

Figure 1: The diagram shows the definition of the quantities in Eq. (16), with styles representing the
time complexity of the computational process in the current box. ⊙: the element-wise product.

4.3
The Overall Learning Framework

After obtaining the differentiable optimization objective J, we can perform the training process
end-to-end. In this process, the overfitting issues may happen especially with insufficient samples,
which could influence both Type I and II errors. If we use the same sample for testing, the Type I
errors may be uncontrollable [22] when the overfitting issues happen. To address this, we adopt the
split scheme as in [24, 18] to allow our tests to maintain validity (controllable Type I errors). The
split ratio is set to 0.5 to facilitate the balance between the two. Apart from controlling Type I errors,
we still want to mitigate overfitting issues as much as possible in order to generalize the optimized

6


---Page Break---
Fourier feature pairs on test data thus improving the power of our tests. To this end, we select smooth
function classes to control the model complexity (e.g., as measured by the VC dimension), specifically
in this paper, we consider the two classes in Tab. 1 and implement them for experiments later. One is
the choice of Gaussian classes that optimize the global scale, and the other we consider Mahalanobis
classes and set Σ to be diag(σ1, ..., σd) for optimization, which corresponds to optimizing the scale
in each dimension (which allows to capture high-frequency signals such as the edge in the image).
These smooth choices also bring the advantage of interpretability [30] and it is experimentally proven
that this simple choice is already able to handle most of the cases in different settings.

More discussion about split strategy. Currently, there are two major classes of approaches for
adaptive independence tests. One involves selecting kernels from a finite/countable set (discrete
scenario) and the other involves performing kernel parameter searches in a continuous space (con-
tinuous scenario). For the former case, some methods [22, 32] control Type I errors by applying
techniques from the selective inference literature without data splitting. However, these methods
cannot be directly applied to a continuous scenario due to the uncountable set of kernels involved.
To the best of our knowledge, both our scheme and existing methods [24, 18] rely on data splitting
for the continuous case. Designing methods to control Type I errors in the continuous case without
sample splitting remains a challenging and significant problem for future research.

Algorithm. Our algorithm is outlined in Alg. 1. As a pre-processing step, we split the data into the
training data Ztr and the testing data Zte (Line 1). The test contains two phases: 1) We learn the
Fourier feature pairs with Adam [21] optimizer using full batches on Ztr (Lines 2-7). 2) With the
learned Fourier feature pairs, we calculate the test statistic and threshold (Lines 8-10) to determine
the independence (Lines 11) on Zte. The overall time complexity is O
 
TnD(dx + dy + D)

and
the space cost is O
 
n(dx + dy + D)

for storing the data as well as the Fourier feature pairs.

Algorithm 1 The learning and testing framework
Input: samples Z of X, Y , significance level α, the number of Fourier feature D.
Output: X ⊥⊥Y or X ̸⊥⊥Y .

1: Split the data as Z = Ztr ∪Zte. Sampling {ωj}D/2
j=1 = {(ωk;j, ωl;j)}D/2
j=1 with p(ω).
2: ◁Learning significant Fourier feature pairs on Ztr.
3: Initialize parameters θk, θl, set learning rate ϵ, and set iteration steps T.
4: for t = 1, 2, ..., T do
5:
Obtain learnable Fourier feature pairs ΛX, ΛY with parameters θk, θl and {ωj}D/2
j=1.
6:
Calculate criterion J with ΛX, ΛY then optimize J with (θk, θl) ←(θk, θl) + ϵ∇(θk,θl)J.
7: end for
8: After training, obtain optimized parameters θ∗
k, θ∗
l .
9: ◁Testing with learned Fourier feature pairs on Zte.
10: Calculate the statistic nteHSICω(Zte), threshold c
α(Zte) with parameters θ∗
k, θ∗
l and {ωj}D/2
j=1.
11: Return X ̸⊥⊥Y if c
α(Zte) ≤nteHSICω(Zte) holds, otherwise X ⊥⊥Y .

5
Theoretical Results

We first give the uniform bound results over a ball in parameter space which guarantees the conver-
gence of our optimizing objective thus ensuring its effectiveness in modeling test power.

Theorem 2 (Uniform Bound). Let θk, θl parameterize Tθk, Tθl in Banach spaces of dimension dk, dl.
And Tθk, Tθl are Lipschitz to the parameters θk, θl with the non-negative constant Lk, Ll, respectively.
Let Θc be a set of (θk, θl) for which σω ≥c > 0 with a positive constant c and ∥θk∥≤Rθk, ∥θl∥≤
Rθl. Let r denote the threshold, i.e., (1 −α)-quantile for the distribution in Eq. (11) and r(n)

be the threshold with sample size n. Let {(ωk;j, ωl;j)}D/2
j=1 be the samplings of frequency with the
sampling number D. Also, we define Rωk := supj ∥ωk;j∥, Rωl := supj ∥ωl;j∥, ds := max{dk, dl}
and ξω := HSICω(Z). Then with probability at least 1 −δ, we have

sup
(θk,θl)∈Θc


ξω −r(n)
ω /n
bσω
−EZξω −rω/n

σω

∼O

 r

1
n log 1

δ + ds
log n

n
+ RωkLk + RωlLl
√n

!

.

7


---Page Break---
Next, we show the consistency of the tests, i.e. the power of the test tends to 1 as the sample size
increases. Let the U-statistic of HSICω(Zte) be HSIC(u)
ω (Zte), then we have the following results.
Theorem 3 (Consistency). Let θ∗
k, θ∗
l be the parameters after learning, Zte be the test samples of
size m, when EZHSIC(u)
ω (Zte) > 0, then the probability of the Type II error

P
 
Type II error

= PH1
 
mHSICω(Zte) ≤r(m)
ω
|θ∗
k, θ∗
l

∼O(m−1/2).
(17)

Let the mapping functions with learned parameters θ∗
k, θ∗
l be Tθ∗
k, Tθ∗
l , and the corresponding range
space be compact subsets of RdTx, RdTy , respectively. Also, the diameters of two range spaces are
denoted by diam(Tθ∗
k), diam(Tθ∗
l ), respectively. Let {(ωk;j, ωl;j)}D/2
j=1 be the frequency samplings
with their second moment denoted by σ2
ωk := Epk(ω)[ωT
k;jωk;j], σ2
ωl := Epl(ω)[ωT
l;jωl;j]. Additionally,
we denote ξu := HSIC(X, Y ), then under H1, we have EZHSIC(u)
ω (Zte) > 0 with any constant

probability when D = Ω
 dTx+dTy

ξ2u
log
σωk diam(Tθ∗
k )+σωldiam(Tθ∗
l )

ξu


.

This result can be understood in two parts. The first one is about consistency, i.e., the Type II
error rate tends to 0 at the rate of m−1/2 when condition EZHSIC(u)
ω (Zte) > 0 holds. The second
part provides the condition when EZHSIC(u)
ω (Zte) > 0 holds, which requires sufficiently many
frequency samplings. The theorem shows that the large value of the criterion HSIC(X, Y ) helps
to reduce the required D. According to the results discussed in Sec. 3, there is an improvement in
the criterion by finding the more significant features and thus helps to reduce the required D. To
summarize, the significant features further help to guarantee the consistency of the test under the
efficient requirements (smaller D). All proofs as well as additional results are given in the Appendix.

6
Performance Evaluation

We compare the following tests: distance-based statistic dCor [39], the original HSIC QHSIC [14],
the copula-based method RDC [26], the three variants of HSIC NyHSIC [44], FHSIC [44], BH-
SIC [44] and HSICAgg [32], NFSIC [18] as introduced in Sec. 1. Among them, dCor and QHSIC are
O(n2) tests. RDC is calculated in O(n log n) time and the rest are O(n) tests. A detailed description
of the comparing methods is given in the Appendix. For our methods, We provide two variants as
mentioned in Sec. 4.3. We name the Gaussian class case as LFHSIC-G, and name the Mahalanobis
class case (and set Σ as a diagonal matrix) LFHSIC-M. Additionally, for the comparative meth-
ods [30] that are relevant to us, due to their high time overhead and therefore inability to handle some
settings of evaluation, we separately provide a comparison with our method under certain feasible
experimental settings, the results are given in the Appendix.

Experimental setup. The significance level α is set to 0.05. We use Gaussian kernels for both X and
Y in all kernel-based methods. And QHSIC, RDC, NyHSIC, FHSIC, BHSIC are all with the kernel
width being set to the Euclidean distance median of the samples. The number of random features D
for FHSIC, LFHSIC-G/M, the number of induced variables for NyHSIC, the block size for BHSIC
as well as the number of sub-diagonals R for HSICAgg are all kept consistent as recommended
in [44, 32] for fair evaluation. Parameter settings for the rest of the methods follow the defaults in the
code. More details of the setups are given in the Appendix.

Evaluation protocol. We evaluate on four synthetic datasets [18, 30] and two real datasets [44, 30].
Synthetic datasets consist of Sine Dependency (SD), Sinusoid (Sin), Gaussian Sign (GSign), and
independent subspace analysis (ISA) dataset [14]. On real data, we introduce high-dimensional image
data and another music dataset to evaluate the capability of all methods in different data scenarios.
Unless otherwise specified, we perform 100 repeated randomized experiments and report the average
result of test power as default. More details of the generating process of each dataset and the details
of the evaluation (including running time) are provided in the Appendix.

6.1
Results on Synthetic Datasets

Settings of SD, Sin, and GSign Dataset. The Sin data corresponds to the example in Sec. 3 that
requires the method to focus on differences in specific frequencies. In SD, Y is dependent solely on
the first two dimensions of X. In contrast, in GSign, Y is independent of any proper subset of X

8


---Page Break---
SD
Sin
GSign
0.00

0.05

0.10

0.15

0.20

0.25

Type I error rate( ↓)

Average Results

2000
2500
3000
3500
4000
0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Sine Dependency (SD)

1000
1500
2000
2500
3000
0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Sinusoid (Sin)

2000
3000
4000
5000
6000
0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Gaussian Sign (GSign)

SD
Sin
GSign
Datasets

0.00

0.05

0.10

0.15

0.20

0.25

Type I error rate( ↓)

2000
2500
3000
3500
4000
Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

1000
1500
2000
2500
3000
Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

dCor
QHSIC

RDC
NyHSIC

FHSIC
BHSIC

HSICAgg
NFSIC

LFHSIC-G
LFHSIC-M

2000
3000
4000
5000
6000
Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Figure 2: Top: (D = 100). Below: (D = 500). Left: The average Type I error rate on SD, Sin, and
GSign datasets. The other three plots: The results of average test power on these three datasets.

0.00
0.25
0.50
0.75
1.00
Angle(×π/4)

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Samp:10000, Dim:16, D:300

0.00
0.25
0.50
0.75
1.00
Angle(×π/4)

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Samp:60000, Dim:32, D:300

0.00
0.25
0.50
0.75
1.00
Angle(×π/4)

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Samp:10000, Dim:16, D:500

RDC
NyHSIC
BHSIC
FHSIC
HSICAgg
NFSIC
LFHSIC-G
LFHSIC-M

0.00
0.25
0.50
0.75
1.00
Angle(×π/4)

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Samp:60000, Dim:32, D:500

Figure 3: The average test power v.s. the rotation angle of each method on the ISA dataset.

but dependent on X as a whole. Therefore, it requires the method to learn important local/global
features based on the characteristics of the data to improve the test power. For SD and GSign, we
set the dimension of X as 4 and 5, respectively, and the dimension of Y is 1 for both. For Sin, we
set the frequency parameter ω = 5. For calculating the Type I error rate, we evaluate using samples
(n = 2000) obtained by permutation for all three datasets.

Performance. The results for D = 100 and D = 500 are shown in Fig. 2. Except for NFSIC and
BHSIC, all the other methods succeed in controlling the Type I error rate ≤0.05. LFHSIC-M/G,
NFHSIC, and HSICAgg perform much better than other methods due to their ability to obtain more
appropriate kernels/features for testing. LFHSIC-G/M performs on both settings of D and has a
more significant advantage over the others when D is small, implying the optimization objective can
still be successfully optimized and the criterion is still powerful under high-speed requirements. In
addition, as the sample size increases the test power of LFHSIC-G/M is gradually converging to 1 in
both settings, which corroborates the results of Theorem 3.

Settings of ISA Dataset (Large Scale). We set dimension (of both X, Y ) and sample size as
d = 16, n = 10000 and d = 32, n = 60000, then evaluate the average test power with angle
parameter θ ∈[0, π/4]. Note that a larger angle signifies stronger dependency. The quadratic-time
methods are not involved in the evaluation due to their inability to handle large-scale settings. For
HSICAgg under the challenging setting n = 60000, d = 32, the memory space required for parallel
implementation leads to memory overflow and hence the results are not given.

Performance. The results for D = 300 and D = 500 are shown in Fig. 3. The results obtained
at θ = 0 reflect the Type I error rate. All methods successfully control the Type I error rate ≤
0.05. LFHSIC-M stably outperforms other methods significantly as the angle increases. Method
(LFHSIC-G) that simply optimizes the global bandwidth performs worse as d increases, corroborating
the need for more flexible kernel designs for more challenging tasks. Furthermore, comparing the

9


---Page Break---
Type I error
Type II error
0.0

0.2

0.4

0.6

0.8

1.0

Average error rate ( ↓)

0.05

Results on 3Dshapes

dCor
QHSIC
RDC
NyHSIC
FHSIC

BHSIC
HSICAgg
NFSIC
LFHSIC-G
LFHSIC-M

dCor

QHSIC

RDC

NyHSIC

FHSIC

BHSIC

HSICAgg

NFSIC

LFHSIC-G

LFHSIC-M

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Sample size: 500

dCor

QHSIC

RDC

NyHSIC

FHSIC

BHSIC

HSICAgg

NFSIC

LFHSIC-G

LFHSIC-M

0.0

0.2

0.4

0.6

0.8

1.0

Average test power( ↑)

Sample size: 1000

Figure 4: The results on two real data. Left: 3DShapes. Right two: MSD Dataset.

results under different settings of D, our method LFHSIC-M performs consistently well and exhibits
progressively better performance as D increases.

6.2
Results on Real Data

Settings of Two Real Data. The first real dataset used is a high-dimensional image dataset 3Dshapes
as in [30]. In our experiments, we vectorize image X to a vector with dimension 64×64×3 = 12, 288.
The sample size is set as 128. We add standard Gaussian noise N(0, 1) to the angle Y to make the
setting more challenging. The Type I error rate is evaluated by the samples obtained by permutation.
Besides, we consider the Million Song Data (MSD) as the second real dataset. The first dimension
represents the year of release of each song and is referred to as variable Y . The remaining 90-
dimensional features (e.g., mean timbre and timbre covariance) constitute variable X. We follow
the recommended setting [44], i.e., disturbing each entry of the X with an independent Gaussian
noise N(0, 1000). For this dataset MSD, in order to fully utilize the data, we randomly select
n ∈{500, 1000} samples as the training set and other n samples from the remaining data 100 times
for the evaluation and obtain the average result. The above training and testing processes are repeated
10 times to evaluate the robustness of the optimization scheme.

Performance. The results of two real data with D = 10 are presented in Fig. 4. For the results on
3Dshapes (shown in the left of Fig. 4), all methods except BHSIC and NFSIC control the Type I
error well. The linear-time test has relatively lower power compared to the quadratic-time test except
for LFHSIC-M, proving that its more significant features obtained in high-dimensional scenarios
enable it to achieve outstanding performance even in scenarios with high approximation requirements
(D = 10). Similar conclusions can drawn from the MSD dataset (shown in the right of Fig. 4).
Additionally, the results for NFSIC and LFHSIC-G/M with different sample sizes indicate increased
robustness of the optimization as the sample size increases (reflected in the reduction of variance),
and the more flexible design also contributes to this (comparing LFHSIC-M and LFHSIC-G), thus
can be more effectively applied to real-world scenarios.

7
Conclusion

In this paper, we propose a novel method to efficiently learn significant Fourier feature pairs for
maximizing the power of HSIC-based independence tests. By integrating a learnable Fourier feature
module, we improve the flexibility of existing configurations and design a new criterion. The proposed
linear-time optimization objective accurately models the power of the test and can be trained end-to-
end in a data-driven manner, ensuring both effectiveness and efficiency. Both theoretical results and
experimental results show the effectiveness of our proposed method. Future work includes further
improving the sampling method in the frequency domain.

Acknowledgments and Disclosure of Funding

This work was supported by National Natural Science Foundation (NSFC) (62372116 and
62472415), and National Key Research and Development Program of China (2021YFC3340302 and
2021YFC3300304).

10


---Page Break---
References

[1] Albert, M., Laurent, B., Marrel, A., and Meynaoui, A. (2022). Adaptive test of independence based on hsic
measures. The Annals of Statistics, 50(2):858–879.

[2] Arcones, M. A. and Gine, E. (1992). On the bootstrap of u and v statistics. The Annals of Statistics, pages
655–674.

[3] Bach, F. R. and Jordan, M. I. (2002). Kernel independent component analysis. Journal of machine learning
research, 3(Jul):1–48.

[4] Bertin-Mahieux, T. (2011).
YearPredictionMSD.
UCI Machine Learning Repository.
DOI:
https://doi.org/10.24432/C50K61.

[5] Burgess, C. and Kim, H. (2018). 3d shapes dataset. https://github.com/deepmind/3dshapes-dataset/.

[6] Camps-Valls, G., Mooij, J., and Scholkopf, B. (2010). Remote sensing feature selection by kernel dependence
measures. IEEE Geoscience and Remote Sensing Letters, 7(3):587–591.

[7] Chatterjee, S. (2021). A new coefficient of correlation. Journal of the American Statistical Association,
116(536):2009–2022.

[8] Chwialkowski, K. P., Ramdas, A., Sejdinovic, D., and Gretton, A. (2015). Fast two-sample testing with
analytic representations of probability measures. Advances in Neural Information Processing Systems, 28.

[9] Cohen, I., Huang, Y., Chen, J., Benesty, J., Benesty, J., Chen, J., Huang, Y., and Cohen, I. (2009). Pearson
correlation coefficient. Noise reduction in speech processing, pages 1–4.

[10] Fukumizu, K., Gretton, A., Schölkopf, B., and Sriperumbudur, B. K. (2008). Characteristic kernels on
groups and semigroups. Advances in neural information processing systems, 21.

[11] Gönen, M. and Alpaydın, E. (2011). Multiple kernel learning algorithms. The Journal of Machine Learning
Research, 12:2211–2268.

[12] Gretton, A. (2015). A simpler condition for consistency of a kernel independence test. arXiv preprint
arXiv:1501.06103.

[13] Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., and Smola, A. (2006). A kernel method for the
two-sample-problem. Advances in neural information processing systems, 19.

[14] Gretton, A., Fukumizu, K., Teo, C., Song, L., Schölkopf, B., and Smola, A. (2007). A kernel statistical test
of independence. Advances in neural information processing systems, 20.

[15] Gretton, A., Smola, A., Bousquet, O., Herbrich, R., Belitski, A., Augath, M., Murayama, Y., Pauls, J.,
Schölkopf, B., and Logothetis, N. (2005). Kernel constrained covariance for dependence measurement. In
International Workshop on Artificial Intelligence and Statistics, pages 112–119. PMLR.

[16] Hoyer, P., Janzing, D., Mooij, J. M., Peters, J., and Schölkopf, B. (2008). Nonlinear causal discovery with
additive noise models. Advances in neural information processing systems, 21.

[17] Jitkrittum, W., Szabó, Z., Chwialkowski, K. P., and Gretton, A. (2016). Interpretable distribution features
with maximum testing power. Advances in Neural Information Processing Systems, 29.

[18] Jitkrittum, W., Szabó, Z., and Gretton, A. (2017). An adaptive test of independence with analytic kernel
embeddings. In International Conference on Machine Learning, pages 1742–1751. PMLR.

[19] Kalinke, F. and Szabo, Z. (2024). The minimax rate of hsic estimation for translation-invariant kernels.
arXiv preprint arXiv:2403.07735.

[20] Kim, I. and Schrab, A. (2023). Differentially private permutation tests: Applications to kernel methods.
arXiv preprint arXiv:2310.19043.

[21] Kingma, D. P. and Ba, J. (2014).
Adam: A method for stochastic optimization.
arXiv preprint
arXiv:1412.6980.

[22] Kübler, J., Jitkrittum, W., Schölkopf, B., and Muandet, K. (2020). Learning kernel tests without data
splitting. Advances in Neural Information Processing Systems, 33:6245–6255.

[23] Li, Y., Pogodin, R., Sutherland, D. J., and Gretton, A. (2021). Self-supervised learning with kernel
dependence maximization. Advances in Neural Information Processing Systems, 34:15543–15556.

11


---Page Break---
[24] Liu, F., Xu, W., Lu, J., Zhang, G., Gretton, A., and Sutherland, D. J. (2020). Learning deep kernels for
non-parametric two-sample tests. In International conference on machine learning, pages 6316–6326. PMLR.

[25] Liu, L., Pal, S., and Harchaoui, Z. (2022). Entropy regularized optimal transport independence criterion.
In International Conference on Artificial Intelligence and Statistics, pages 11247–11279. PMLR.

[26] Lopez-Paz, D., Hennig, P., and Schölkopf, B. (2013). The randomized dependence coefficient. Advances
in neural information processing systems, 26.

[27] Podkopaev, A. and Ramdas, A. (2024). Sequential predictive two-sample and independence testing.
Advances in Neural Information Processing Systems, 36.

[28] Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. Advances in neural
information processing systems, 20.

[29] Ramdas, A., Reddi, S. J., Póczos, B., Singh, A., and Wasserman, L. (2015). On the decreasing power of
kernel and distance based nonparametric hypothesis tests in high dimensions. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 29.

[30] Ren, Y., Xia, Y., Zhang, H., Guan, J., and Zhou, S. (2024). Learning adaptive kernels for statistical
independence tests. In International Conference on Artificial Intelligence and Statistics, pages 2494–2502.
PMLR.

[31] Ren, Y., Zhang, H., Xia, Y., Guan, J., and Zhou, S. (2023). Multi-level wavelet mapping correlation for
statistical dependence measurement: methodology and performance. In Proceedings of the AAAI Conference
on Artificial Intelligence, volume 37(5), pages 6499–6506.

[32] Schrab, A., Kim, I., Guedj, B., and Gretton, A. (2022). Efficient aggregated kernel tests using incomplete
u-statistics. Advances in Neural Information Processing Systems, 35:18793–18807.

[33] Sejdinovic, D., Sriperumbudur, B., Gretton, A., and Fukumizu, K. (2013). Equivalence of distance-based
and rkhs-based statistics in hypothesis testing. The annals of statistics, pages 2263–2291.

[34] Serfling, R. J. (2009). Approximation theorems of mathematical statistics. John Wiley & Sons.

[35] Shekhar, S., Kim, I., and Ramdas, A. (2023). A permutation-free kernel independence test. Journal of
Machine Learning Research, 24(369):1–68.

[36] Sriperumbudur, B. K., Gretton, A., Fukumizu, K., Schölkopf, B., and Lanckriet, G. R. (2010). Hilbert space
embeddings and metrics on probability measures. The Journal of Machine Learning Research, 11:1517–1561.

[37] Sutherland, D. J. and Schneider, J. (2015). On the error of random fourier features. arXiv preprint
arXiv:1506.02785.

[38] Sutherland, D. J., Tung, H.-Y., Strathmann, H., De, S., Ramdas, A., Smola, A., and Gretton, A.
(2016). Generative models and model criticism via optimized maximum mean discrepancy. arXiv preprint
arXiv:1611.04488.

[39] Székely, G., Rizzo, M., and Bakirov, N. (2007). Measuring and testing dependence by correlation of
distances. Annals of Statistics, 35(6):2769–2794.

[40] Székely, G. J. and Rizzo, M. L. (2013). The distance correlation t-test of independence in high dimension.
Journal of Multivariate Analysis, 117:193–213.

[41] Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science,
volume 47. Cambridge university press.

[42] Wang, Z., Zhan, Z., Gong, Y., Shao, Y., Ioannidis, S., Wang, Y., and Dy, J. (2023). Dualhsic: Hsic-
bottleneck and alignment for continual learning. In International Conference on Machine Learning, pages
36578–36592. PMLR.

[43] Zhang, K., Peters, J., Janzing, D., and Schölkopf, B. (2012). Kernel-based conditional independence test
and application in causal discovery. arXiv preprint arXiv:1202.3775.

[44] Zhang, Q., Filippi, S., Gretton, A., and Sejdinovic, D. (2018). Large-scale kernel methods for independence
testing. Statistics and Computing, 28:113–130.

[45] Zhang, T., Zhang, Y., and Zhou, T. (2024). Statistical insights into hsic in high dimensions. Advances in
Neural Information Processing Systems, 36.

12


---Page Break---
Appendix Organization

• Section A: List of Symbols and Notations.
• Section B: Assumptions.
• Section C: Some Auxiliary Lemma.
• Section D: Proof of Proposition 1.
• Section E: Proof of Theorem 1.
• Section F: Calculation of Eq. (16).
• Section G: Proof of Theorem 2.
• Section H: Proof of Theorem 3.
• Section I: Smoothness of Optimization Objective.
• Section J: Details of Experiment Setup.
• Section K: Additional Experiment Results.
• Section L: Limitations and Broader Impacts.

A
List of Symbols and Notations

O
big O notion
o
small O notion
i.i.d.
independent and identically distributed
R
the set of real numbers
B(R)
Borel σ-algebra on R
PX
marginal distribution of X
PXY
joint distribution of X, Y
FX
distribution function of X
E[X]
expectation of X
Var(X)
variance of X
X ⊥⊥Y
random variables X, Y are independent
X ̸⊥⊥Y
random variables X, Y are not independent
in
r
the set of all r-tuples drawn without replacement from the set {1, ..., n}
 n
k

number of k-combinations of n elements
(n)k
number of permutations, define as
n!
(n−k)!
Tr(·)
the trace of a square matrix
1
an vector of all ones
H
centering matrix define as H = I −1

n11T
⊙
element-wise product
()·2
element-wise power
d−→
convergence in distribution
⊗
the product symbol of measure
×
the product symbol of topological space
N(Θ, r)
covering number with radii r for space Θ
F, F−1
the Fourier transform, Fourier inverse transform

B
Assumptions

The following are the assumptions required. We denote the parameter spaces of θk, θl as Θk, Θl.

(a) The mapping functions Tθk, Tθl are Lipschitz to the parameters θk, θl, i.e. for all x ∈X, y ∈Y
and for all θk, θ′
k ∈Θk, θl, θ′
l ∈Θl,
Tθk(x) −Tθ′
k(x)
≤Lk · ∥θk −θ′
k∥,
Tθl(y) −Tθ′
l(y)
≤Ll · ∥θl −θ′
l∥
(18)
with the nonnegative Lipschitz constant Lk, Ll.
(b) The range of the mapping functions Tθk, Tθl are bounded.

(c) The parameters θ0, θ1 lie in Banach spaces of dimension dk, dl respectively. Also, the parameters
θk, θl are bounded by Rθk, Rθl respectively, i.e., ∥θk∥≤Rθk, ∥θl∥≤Rθl.

13


---Page Break---
C
Some Auxiliary Lemma

C.1
A Useful Expression

In this part, we give a useful expression for h(ω)
ijqr for the subsequent proof. By the definition, we have

h(ω)
ijqr = 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vw −2k(ω)
tu l(ω)
tv . We simplify it by setting t = i, u = i, v = i

and w = i in turn. Then we can show that h(ω)
ijqr is equal to

1
4!

(j,q,r)
X

(u,v,w)
(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −2k(ω)
iu l(ω)
iv ) + 1

4!

(j,q,r)
X

(t,v,w)
(k(ω)
ti l(ω)
ti
+ k(ω)
ti l(ω)
vw −2k(ω)
ti l(ω)
tv )

+ 1

4!

(j,q,r)
X

(t,u,w)
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −2k(ω)
tu l(ω)
ti ) + 1

4!

(j,q,r)
X

(t,u,v)
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vi −2k(ω)
tu l(ω)
tv ).

(19)

By the definition, k(ω)
tu := ψ(ω)
k
(Tθkxt −Tθkxu) = ΛX(xt)ΛX(xu)T is symmetric, i.e. k(ω)
tu = k(ω)
ut .
And so as l(ω)
tu . Hence we can merge the identical items (marked with the same color). As a result,

h(ω)
ijqr = 1

4!

(j,q,r)
X

(u,v,w)
(2k(ω)
iu l(ω)
iu + 2k(ω)
iu l(ω)
vw −2k(ω)
iu l(ω)
iv ) −1

4!

(j,q,r)
X

(t,v,w)
(2k(ω)
ti l(ω)
tv )

+ 1

4!

(j,q,r)
X

(t,u,w)
(2k(ω)
tu l(ω)
tu + 2k(ω)
tu l(ω)
iw −2k(ω)
tu l(ω)
ti ) −1

4!

(j,q,r)
X

(t,u,v)
(2k(ω)
tu l(ω)
tv ).

(20)

We will use Eq. (20) many times in subsequent proofs.

C.2
Properties of Learnable Random Fourier Feature

Under the assumption (a), RFFs ψ(ω)
k
, ψ(ω)
l
are Lipschitz to the parameters θk, θl. Formally,
Lemma 1. (Lipschitz Property of Fourier Feature). Let Tθk, Tθl be the mapping functions of X, Y
that are Lipschitz to the parameters θk, θl with the non-negative constant Lk, Ll, respectively. Let
{(ωk;j, ωl;j)}D/2
j=1 be the samplings of frequency with the sampling number D. Also, we define

Rωk := supj ∥ωk;j∥, Rωl := supj ∥ωl;j∥, then for the RFFs ψ(ω)
k
, ψ(ω)
l
with mapping functions

Tθk, Tθl and frequency samplings {(ωk;j, ωl;j)}D/2
j=1, for all (x, x′) ∈X × X, (y, y′) ∈Y × Y and
for all θk, θ′
k ∈Θk, θl, θ′
l ∈Θl, we have

∥ψ(ω)
k
(∆x,x′) −ψ(ω)
k
(∆′
x,x′)∥≤2RωkLk · ∥θk −θ′
k∥,

∥ψ(ω)
l
(∆y,y′) −ψ(ω)
l
(∆′
y,y′)∥≤2RωlLl · ∥θl −θ′
l∥,
(21)

where ∆x,x′ := Tθkx −Tθkx′, ∆′
x,x′ := Tθ′
kx −Tθ′
kx′ and ∆y,y′, ∆′
y,y′ are defined by analogy.

Proof. We prove the result for ψ(ω)
k
only since the proof for ψ(ω)
l
can be obtained in the same way.
We start by recall the definition ψ(ω)
k
(∆x,x′) := 2

D
PD/2
j=1 cos(ωT
k;j∆x,x′). Then

∥ψ(ω)
k
(∆x,x′) −ψ(ω)
k
(∆′
x,x′)∥=
 2

D

D/2
X

j=1
cos(ωT
k;j∆x,x′) −2

D

D/2
X

j=1
cos(ωT
k;j∆′
x,x′)


≤2

D

D/2
X

j=1
∥cos(ωT
k;j∆x,x′) −cos(ωT
k;j∆′
x,x′)∥

(22)

Since the cosine function is bounded by 1, by the mean value theorem, for fixed j, we have

∥cos(ωT
k;j∆x,x′) −cos(ωT
k;j∆′
x,x′)∥≤|ωT
k;j∆x,x′ −ωT
k;j∆′
x,x′|.
(23)

14


---Page Break---
Then according to the Cauchy–Schwarz inequality,

|ωT
k;j∆x,x′ −ωT
k;j∆′
x,x′| ≤∥ωk;j∥· ∥∆x,x′ −∆′
x,x′∥.
(24)

By the definition of ∆x,x′ and the Lipschitz property of the mapping functions Tθk, Tθl, we have

∥∆x,x′ −∆′
x,x′∥= ∥(Tθkx −Tθkx′) −(Tθ′
kx −Tθ′
kx′)∥

≤∥Tθkx −Tθ′
kx∥+ ∥Tθkx′ −Tθ′
kx′∥≤2Lk · ∥θk −θ′
k∥.
(25)

Combining the above results, we complete the proof.

Under the assumption (b), we can obtain the uniform convergence property as follows.
Lemma 2. (Uniform Convergence of Fourier Features). Let the mapping function of X, Y with
parameters θk, θl be Tθk, Tθl, and the corresponding range space be a compact subset of RdTx, RdTy ,
respectively. Also, the diameter of two range spaces is denoted by diam(Tθk), diam(Tθl), respectively.
Let {(ωk;j, ωl;j)}D/2
j=1 be the samplings of frequency with the sampling number D, then for the RFFs

with mapping functions Tθk, Tθl and frequency samplings {(ωk;j, ωl;j)}D/2
j=1, we have

P

sup
x,x′∈X
|Λk(x)T Λk(x′) −k(x, x′)| ≥ϵ

≤28
σωkdiam(Tθk)

ϵ

2
exp

−
Dϵ2

4(dTx + 2)


,

P

sup
y,y′∈Y
|Λl(y)T Λl(y′) −l(y, y′)| ≥ϵ

≤28
σωldiam(Tθl)

ϵ

2
exp

−
Dϵ2

4(dTy + 2)


,

(26)

where the second moment of frequency samplings σ2
ωk := Epk(ω)[ωT
k;jωk;j], σ2
ωl := Epl(ω)[ωT
l;jωl;j].

Proof. Based on the derivation of RFFs in Sec. 4.1, We can view it as if the frequency sampling
process is performed after the range space is obtained. Since the convergence bounds of the sampling
process can be obtained directly through the results of [28, Claim 1], by replacing the input space
in [28, Claim 1] to the range space here, then this part of the proof can be completed.

Remark. Combining the technique in [37], the constants in bounds can be further improved.

C.3
Approximation Error Bound

Let HSIC(u)
ω (Z), also denoted as ξ(u)
ω , be the U-statistic that corresponding to HSICω(Z), i.e.,
HSIC(u)
ω (Z) :=
1
(n)4
P

(i,j,q,r)∈in
4 h(ω)
ijqr. The population value of HSIC(u)
ω (Z) is given by EZξ(u)
ω
which can be viewed as the result obtained after a frequency sampling approximation on HSIC(X, Y )
is performed. The bound of approximation error is given by the following Lemma.
Lemma 3. (Approximation Error Bound). For simplify, we denote Λk(x)T Λk(x′), Λl(y)T Λl(y′) as
k(ω)(x, x′), l(ω)(y, y′), respectively. Then we have

|EZξ(u)
ω
−HSIC(X, Y )| ≤4 ·
sup
x,x′∈X,y,y′∈Y
|k(ω)(x, x′)l(ω)(y, y′) −k(x, x′)l(y, y′)|.
(27)

Proof. We first represent EZξ(u)
ω
in the form corresponding to Eq. (1), i.e.,

EZξ(u)
ω
= EXX′Y Y ′
k(ω)(X, X′)l(ω)(Y, Y ′)

+EXX′
k(ω)(X, X′)

EY Y ′
l(ω)(Y, Y ′)


−2EX′Y ′
EXk(ω)(X, X′)EY l(ω)(Y, Y ′)

.
(28)

Taking one of the items as an example and comparing it to the corresponding item in Eq. (1),
EX′Y ′
EXk(ω)(X, X′)EY l(ω)(Y, Y ′)

−EX′Y ′
EXk(X, X′)EY l(Y, Y ′)


≤
Z

X′,Y ′

Z

X

Z

Y
|k(ω)(X, X′)l(ω)(Y, Y ′) −k(X, X′)l(Y, Y ′)|dPXdPY dPX′Y ′

≤
sup
x,x′∈X,y,y′∈Y
|k(ω)(x, x′)l(ω)(y, y′) −k(x, x′)l(y, y′)|.

(29)

The results can be obtained for the other terms in a similar way, which completes the proof.

15


---Page Break---
D
Proof of Proposition 1

In this section, we give a proof of the Proposition 1. We first restate the Proposition 1 here.

Proposition 1 (Asymptotics). Let h(ω)
ijqr := 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vw −2k(ω)
uv l(ω)
tv , where the
sum represents all ordered quadruples (t, u, v, w) drawn without replacement from (i, j, q, r). Then,
Under the null hypothesis H0, HSICω(Z) coverages in distribution to

nHSICω(Z)
d−→

∞
X

l=1
λlχ2
1l, λlgl(zj) =
Z

zi,zq,zr
h(ω)
ijqrgl(zi)dFzi,zq,zr,
(30)

where χ2
11, χ2
12, ... are independent χ2
1 variates and λl is the solution to the eigenvalue problem as in
the right of Eq. (30). Also, under the alternative H1, HSICω(Z) converges in distribution as

n
1
2

HSICω(Z) −EZHSICω(Z)
 d−→N(0, σ2
ω), σ2
ω = 16
h
Ei(Ej,q,rh(ω)
ijqr)2 −
 
EZh(ω)
ijqr
2i
(31)

with the simplified notation Ej,q,r := Ezj,zq,zr and EZ := Ezi,zj,zq,zr.

Proof. The proof is mainly based on [34, Chapter 5]. A proof for a similar result has been given
in [14, Theorem 1,2]. The difference is that we consider the asymptotic distributions of HSICω(Z)
that used learnable RFF thus is a function of frequency samplings while they consider HSICb(Z).
Thus some steps need to be modified.

Step 1: we show that HSICω(Z) is a V-statistic, this can be done since it can be expressed as
HSICω(Z) =
1
n4
P

i,j,q,r h(ω)
ijqr. To facilitate the conclusions in [34] for the U-statistic, we define the

U-statistic HSIC(u)
ω (Z) :=
1
(n)4
P

(i,j,q,r)∈in
4 h(ω)
ijqr that corresponds to HSICω(Z).

Step 2: Then we prove the result under H0. To begin with, we first show that Ej,q,rh(ω)
ijqr = 0 with

(i, j, q, r) ∈in
4 under H0. According to Eq. (20), we can calculate Ej,q,rh(ω)
ijqr as

Ej,q,rh(ω)
ijqr = 2

4!

(j,q,r)
X

(u,v,w)
Eu,v,w(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −k(ω)
iu l(ω)
iv ) −2

4!

(j,q,r)
X

(t,v,w)
Et,v,w(k(ω)
ti l(ω)
tv )

+ 2

4!

(j,q,r)
X

(t,u,w)
Et,u,w(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −k(ω)
tu l(ω)
ti ) −2

4!

(j,q,r)
X

(t,u,v)
Et,u,v(k(ω)
tu l(ω)
tv )

=1

2Ei̸=u̸=v̸=w
u,v,w
(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −k(ω)
iu l(ω)
iv ) −1

2Ei̸=t̸=v̸=w
t,v,w
(k(ω)
ti l(ω)
tv )

+ 1

2Ei̸=t̸=u̸=w
t,u,w
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −k(ω)
tu l(ω)
ti ) −1

2Ei̸=t̸=u̸=v
t,u,v
(k(ω)
tu l(ω)
tv ),

(32)

where we define simplified notions Ei̸=u̸=v̸=w
u,v,w
whose superscript indicates the restriction. For

readability, we will require additional notations: Exk(ω)
i
:= Ei̸=u
u
k(ω)
iu and Exk(ω) := Et̸=u
t,u k(ω)
tu
(the notations for Y is defined by analogy). Under H0, X and Y are independence. Hence

2Ej,q,rh(ω)
ijqr =Exk(ω)
i
Eyl(ω)
i
+ Exk(ω)
i
Eyl(ω) −Exk(ω)
i
Eyl(ω)
i
−Exk(ω)
i
Eyl(ω)

+ Exk(ω)Eyl(ω) + Exk(ω)Eyl(ω)
i
−Exk(ω)Eyl(ω)
i
−Exk(ω)Eyl(ω) = 0.
(33)

Then combining the results [34, Section 5.5.2], we can prove Eq. (30).

Step 3: Next we prove the asymptotic distribution under H1. We only need to show that |HSICω(Z)−
HSIC(u)
ω (Z)| ∼O(1/n). By the definition of k(ω)
tu , l(ω)
tu , we can check that |k(ω)
tu | ≤1, |l(ω)
tu | ≤1 for
all t, u, thus |h(ω)
ijqr| ≤4 for all i, j, q, r. Hence we have

|HSICω(Z) −HSIC(u)
ω (Z)| ≤n4 −(n)4

n4
· 4 +
 1

(n)4
−1

n4


·(n)4 · 4 ∼O(1/n).
(34)

Combining the results [34, Section 5.5.1], we can prove Eq. (31).

16


---Page Break---
E
Proof of Theorem 1

In this section, we give a proof of the Theorem 1. We first restate the Theorem 1 here.

Theorem 1 (Linear-Time Estimators). Under H0, the estimators of mean and variance with bias of
O(n−1) to EZ[nHSICω(Z)] and VarZ[nHSICω(Z)], denote as E0 and V0, respectively, are given by

E0 := [1T Λ·2
Xc1][1T Λ·2
Y c1]
(n −1)2
, V0 :=
2n(n −4)(n −5)
(n −1)(n −2)(n −3)
[1T (ΛT
XcΛXc)·21][1T (ΛT
Y cΛY c)·21]
n4
,

(35)
where ()·2 is the entrywise matrix power. Both E0 and V0 can be calculated in O(nD2) time.

Proof. We first prove the part of the mean. Recall the definition of HSICω(Z), we have HSICω(Z) =
1
n4
P

i,j,q,r h(ω)
ijqr. Hence EZ[HSICω(Z)] =
1
n4
P

i,j,q,r EZh(ω)
ijqr. When (i, j, q, r) ∈in
4, then we

can show that under H0, Ei,j,q,rh(ω)
ijqr = 0 by performing the same analysis as in Eqs. (32) and
(33). Then we consider the case where exactly two elements of i, j, q, r are the same, for a total of
6n(n −1)(n −2) terms. By the symmetry of h(ω)
ijqr, the expectation of these terms all take the same

value, and here we take h(ω)
iiqr as an example. According to Eq. (20), we can represent h(ω)
iiqr as

h(ω)
iiqr = 2

4!

(i,q,r)
X

(u,v,w)
(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −k(ω)
iu l(ω)
iv ) −2

4!

(i,q,r)
X

(t,v,w)
(k(ω)
ti l(ω)
tv )

+ 2

4!

(i,q,r)
X

(t,u,w)
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −k(ω)
tu l(ω)
ti ) −2

4!

(i,q,r)
X

(t,u,v)
(k(ω)
tu l(ω)
tv ).

(36)

Under H0, X and Y are independence. Take the expectation on both sides, we have

12Ei,q,rh(ω)
iiqr = (2 + 4Exk(ω)Eyl(ω)) + (2Eyl(ω) + 4Exk(ω)Eyl(ω))

−(2Exk(ω) + 2Eyl(ω) + 2Exk(ω)Eyl(ω)) −(2Eyl(ω) + 4Exk(ω)Eyl(ω))

+ (6Exk(ω)Eyl(ω)) + (2Exk(ω) + 4Exk(ω)Eyl(ω))

−(2Exk(ω) + 4Exk(ω)Eyl(ω)) −(6Exk(ω)Eyl(ω))

= 2(1 −Exk(ω) −Eyl(ω) + Exk(ω)Eyl(ω)),

(37)

where we define additional notation Exk(ω) := Et̸=u
t,u k(ω)
tu (the notation for Y is defined by analogy)
and use kω
tt = lω
tt = 1. Hence in this case, the sum of the contributions of all terms to EZ[nHSICω(Z)]
is (1 −Exk(ω) −Eyl(ω) + Exk(ω)Eyl(ω)) + O(1/n). For the remaining terms, i.e., the case where
at least three of i, j, q, r are equal, combined with the boundedness of h(ω)
ijqr, we can conclude that the
sum of their contributions is O(1/n). As a result, we have shown that

EZ[nHSICω(Z)] = (1 −Exk(ω))(1 −Eyl(ω)) + O(n−1).
(38)

The unbiased estimators of Exk(ω), Eyl(ω) are given by 1T (ΛXΛT
X −In)1, 1T (ΛY ΛT
Y −In)1,
respectively. Hence under H0, we obtain the estimator of mean with bias of O(n−1) as

1 −1T (ΛXΛT
X −In)1
n(n −1)

 
1 −1T (ΛY ΛT
Y −In)1
n(n −1)


= [1T Λ·2
Xc1][1T Λ·2
Y c1]
(n −1)2
.
(39)

Next, we prove the part of the variance. We start by calculating VarZ[nHSIC(u)
ω (Z)], where the U-
statistic HSIC(u)
ω (Z) :=
1
(n)4
P

(i,j,q,r)∈in
4 h(ω)
ijqr. According to the results [34, Section 5.2.1, Lemma
A], we have

Var[HSIC(u)
ω (Z)] =
n
4

−1
4
X

c=1

4
c

n −4
4 −c


ζc = 4
 n−4
3

 n
4

ζ1 + 6
 n−4
2

 n
4

ζ2 + O(n−3),
(40)

17


---Page Break---
where ζ1 := Ei
 
Ej,q,rh(ω)
ijqr
2 and ζ2 := Ei,j
 
Eq,rh(ω)
ijqr
2. Under H0, when (i, j, q, r) ∈in
4, we

can show that Ej,q,rh(ω)
ijqr = 0 by performing the same analysis as in Eqs. (32) and (33), thus ζ1 = 0.

For calculating ζ2, we mainly focus on the term Eq,rh(ω)
ijqr. We use Eq. (20) again,

12h(ω)
ijqr =

(j,q,r)
X

(u,v,w)
(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −k(ω)
iu l(ω)
iv ) −

(j,q,r)
X

(t,v,w)
(k(ω)
ti l(ω)
tv )

+

(j,q,r)
X

(t,u,w)
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −k(ω)
tu l(ω)
ti ) −

(j,q,r)
X

(t,u,v)
(k(ω)
tu l(ω)
tv ).

(41)

Under H0, X and Y are independence. Take the expectation Eq,r on both sides, we have

12Eq,rh(ω)
ijqr = (2k(ω)
ij l(ω)
ij
+ 4Exk(ω)
i
Eyl(ω)
i
) + (2k(ω)
ij Eyl(ω) + 4Exk(ω)
i
Eyl(ω)
j
)

−(2k(ω)
ij Eyl(ω)
i
+ 2Exk(ω)
i
l(ω)
ij
+ 2Exk(ω)
i
Eyl(ω)
i
)

−(2k(ω)
ij Eyl(ω)
j
+ 2Exk(ω)
i
Eyl(ω)
j
+ 2Exk(ω)
i
Eyl(ω))

+ (4Exk(ω)
j
Eyl(ω)
j
+ 2Exk(ω)Eyl(ω)) + (4Exk(ω)
j
Eyl(ω)
i
+ 2Exk(ω)l(ω)
ij )

−(2Exk(ω)
j
Eyl(ω)
ij
+ 2Exk(ω)
j
Eyl(ω)
i
+ 2Exk(ω)Eyl(ω)
i
)

−(2Exk(ω)
j
Eyl(ω)
j
+ 2Exk(ω)
j
Eyl(ω) + 2Exk(ω)Eyl(ω)
j
)

= 2(k(ω)
ij
−Exk(ω)
i
−Exk(ω)
j
+ Exk(ω))(l(ω)
ij
−Eyl(ω)
i
−Eyl(ω)
j
+ Eyl(ω)),

(42)

where we define additional notation Exk(ω) := Et̸=u
t,u k(ω)
tu , Exk(ω)
j
:= Ej̸=u
u
k(ω)
ju (the notations for

Y are defined by analogy). For simplify, we denote k(ω)
c;ij := k(ω)
ij
−Exk(ω)
i
−Exk(ω)
j
+ Exk(ω) and

l(ω)
c;ij := l(ω)
ij
−Eyl(ω)
i
−Eyl(ω)
j
+ Eyl(ω). Hence combining Eq. (40), we have

Var[nHSIC(u)
ω (Z)] =
2n(n −4)(n −5)
(n −1)(n −2)(n −3)Ei,j(k(ω)
c;ij)2 · Ei,j(l(ω)
c;ij)2 + O(n−1).
(43)

Since under H0, the bias lead by the difference terms between HSIC(u)
ω (Z) and HSICω(Z) vanish
faster than Eq. (43), hence the variance of HSICω(Z) is identical. In the following part, we consider
the empirical estimate of the leading term in Eq. (43). We estimate k(ω)
c;ij with (ΛXcΛT
Xc)ij, then the

estimation of Ei,j(k(ω)
c;ij)2 is given by

1
n2
X

i,j


(Λk(xi) −Λk)(Λk(xj) −Λk)T 2= 1

n2
X

i,j
(ΛXcΛT
Xc)2
ij = 1T (ΛXcΛT
Xc)·21
n2
,
(44)

where we define notions Λk := 1

n
Pn
u=1 Λk(xu). Since computing the value of 1T (ΛXcΛT
Xc)·21
requires O(n2) time complexity, we transform it into a more computationally tractable form. We
perform the following calculating as

1T (ΛXcΛT
Xc)·21 = Tr(ΛXcΛT
XcΛXcΛT
Xc) = Tr(ΛT
XcΛXcΛT
XcΛXc) = 1T (ΛT
XcΛXc)·21.
(45)
Recall the definition of ΛXc := [HΛX]n×D that can be calculated in O(nD) time, thus the term
[ΛT
XcΛXc]D×D can be calculated in O(nD + nD2) time. As a result, we obtain the estimator
[1T (ΛT
XcΛXc)·21][1T (ΛT
Y cΛY c)·21] for the term Ei,j(k(ω)
c;ij)2 · Ei,j(l(ω)
c;ij)2 that can be calculated
in O(nD2) time. The only thing left to do is to determine the bias of the estimator. For readable,
we define bk(ω)
ij
:= k(ω)
ij , bk(ω)
i
:= 1

n
P

u k(ω)
iu and bk(ω) :=
1
n2
P

u,v k(ω)
uv , then by removing the terms
with i = j, a estimate with difference O(n−1) to Eq. (44) is given by

1
n(n −1)

X

i̸=j

h
bk(ω)
ij
−bk(ω)
i
−bk(ω)
j
+ bk(ω)i2
.
(46)

18


---Page Break---
By comparing the difference between the expectation of Eq. (46) and Ei,j(k(ω)
c;ij)2, we can show that
this error is bound by O(1/n). We illustrate this by taking one of the cross terms as an example and
the other terms by analogy, as shown in the following,

E
h
1
n(n −1)

X

i̸=j
bk(ω)
i
bk(ω)i
=
1
n3(n −1)E
hX

i

X

q,r

X

u
k(ω)
iu k(ω)
qr
i

=
1
(n)4
E
h
X

(i,q,r,u)∈in
4
k(ω)
iu k(ω)
qr
i
+O(n−1) = Exk(ω)
i
Exk(ω) + O(n−1).
(47)

Similarly, we can obtain the results for Ei,j(l(ω)
c;ij)2. As a result, we have shown that V0 is a estimator
of VarZ[nHSICω(Z)] with bias O(n−1) thus complete the whole proof.

F
Calculation of Eq. (16)

Here, we give the computational details of Eq. (16). We mark colors to indicate correspondences.

According to Eq. (20), we can calculate P

j,q,r h(ω)
ijqr as

X

j,q,r
h(ω)
ijqr = 1

2

X

u,v,w
(k(ω)
iu l(ω)
iu + k(ω)
iu l(ω)
vw −k(ω)
iu l(ω)
iv ) −1

2

X

t,v,w
(k(ω)
ti l(ω)
tv )

+ 1

2

X

t,u,w
(k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
iw −k(ω)
tu l(ω)
ti ) −1

2

X

t,u,v
(k(ω)
tu l(ω)
tv ).
(48)

We can further represent Eq. (48) in matrices form as
X

j,q,r
h(ω)
ijqr = 1

2

h

n2(ΛXΛT
XΛY ΛT
Y )i,i + (ΛXΛT
X1)i(1T ΛY ΛT
Y 1) −n[(ΛXΛT
X1) ⊙(ΛY ΛT
Y 1)]i

+ nTr(ΛXΛT
XΛY ΛT
Y ) + (ΛY ΛT
Y 1)i(1T ΛXΛT
X1) −n(ΛY ΛT
Y ΛXΛT
X1)i

−n(ΛXΛT
XΛY ΛT
Y 1)i −(1T ΛXΛT
XΛY ΛT
Y 1)

i
.

(49)

Next, by variable substitution, we obtain the result as
X

j,q,r
h(ω)
ijqr = 1

2


n1T A1 + n2(A1)i + (1T C)Bi + (1T B)Ci −nEi −nFi −nDi −1T D


.

(50)
where the definition of variables A to F with the calculation cost are given in the Fig. 1. For
convenience, we re-show the diagram here for reference.

Time Complexity: 

Figure 5: The diagram shows the definition of the quantities, with styles representing the time
complexity of the computational process in the current box. ⊙: the element-wise product.

The computational complexity of each step is illustrated in Fig. 5. We explain some steps here. As
a start, recall that the size of ΛX, ΛY are both n × D. Therefore a time complexity O(nD2) is
required to compute ΛT
XΛY by matrix multiplication operation. Further multiplying the obtained
[ΛT
XΛY ]D×D with ΛT
Y requires O(nD2) time complexity. Next, since both ΛX and (ΛT
XΛY ΛT
Y )T
are of size n × D, the elemental product operation requires a time complexity of O(nD). In a similar
way, we can check the time complexity for each remained step. After getting the variables A to
F, since 1T A1, A1 all can be calculated in O(nD) and 1T B, 1T C, 1T D all can be calculated in
O(n), we conclude that the results with index i in Eq. (50) can be obtained in O(nD2) time.

19


---Page Break---
G
Proof of Theorem 2

In this section, we give a proof of the Theorem 2. We first restate the Theorem 2 here.
Theorem 2 (Uniform Bound). Let θk, θl parameterize Tθk, Tθl in Banach spaces of dimension dk, dl.
And Tθk, Tθl are Lipschitz to the parameters θk, θl with the non-negative constant Lk, Ll, respectively.
Let Θc be a set of (θk, θl) for which σω ≥c > 0 with a positive constant c and ∥θk∥≤Rθk, ∥θl∥≤
Rθl. Let r denote the threshold, i.e., (1 −α)-quantile for the distribution in Eq. (11) and r(n)

be the threshold with sample size n. Let {(ωk;j, ωl;j)}D/2
j=1 be the samplings of frequency with the
sampling number D. Also, we define Rωk := supj ∥ωk;j∥, Rωl := supj ∥ωl;j∥, ds := max{dk, dl}
and ξω := HSICω(Z). Then with probability at least 1 −δ, we have

sup
(θk,θl)∈Θc


ξω −r(n)
ω /n
bσω
−EZξω −rω/n

σω

∼O

 r

1
n log 1

δ + ds
log n

n
+ RωkLk + RωlLl
√n

!

.

Proof. We take a similar roadmap of proof as [30] and extend it to our optimization objective. The
roadmap of the proof is as follows: we first obtain the convergence results (with sample size n) for
each estimator with fixed parameters θk, θl, and then extend the results to the entire parameter space
via ϵ-net arguments. We begin the proof of the first part, which is based on bounded differences
inequality (McDiarmid’s inequality) [41, Theorem 2.9.1].

Bound of |ξω −EZξω|. Recall the definition of ξω := HSICω(Z) =
1
n4
P

i,j,q,r h(ω)
ijqr. By the

definition of k(ω)
tu , l(ω)
tu , we have |k(ω)
tu | ≤1, |l(ω)
tu | ≤1 for all t, u, thus |h(ω)
ijqr| ≤4 for all i, j, q, r.

Now we begin by showing the bounded differences property of h(ω)
ijqr. Concretely, we replace the first
sample z1 = (x1, y1) with z′
1 = (x′
1, y′
1) and keep the remaining samples the same. The obtained
samples are named as Z′. Then the difference terms between h(ω)
ijqr and the new substitution ˘h(ω)
ijqr can
only happen in the case that at least one of i, j, q, r is equal to 1. For the case that only one subscript
is 1 (here take i = 1 for example), combining Eq. (20), we have

X

j,q,r
h(ω)
1jqr −
X

j,q,r
˘h(ω)
1jqr
≤2

4!(n −1)(n −2)(n −3) · 6 · 16 = 8(n −1)(n −2)(n −3).
(51)

The whole contributes of remaining terms that at least two i, j, q, r are less than O(n−2), thus
 1

n4
X

i,j,q,r
h(ω)
ijqr −1

n4
X

i,j,q,r
˘h(ω)
ijqr
≤4 ·
 1

n4
X

j,q,r
h(ω)
1jqr −1

n4
X

j,q,r
˘h(ω)
1jqr
+O(n−2) = O(n−1). (52)

Hence HSICω(Z) satisfy the bounded differences property with O(n−1). Using McDiarmid’s
inequality, for fixed θ0, θ1, with probability at least 1 −δ, there exist a universal constant C1 such that

ξω −EZξω
≤C1

r

1
n log 2

δ .
(53)

Bound of |r(n)
ω
−rω|. As r(n)
ω
is the (1 −α) of the distribution of nξω with sample size n under H0,
according to the Eq. (53), when n is large enough, there exist a universal constant C2 such that

|r(n)
ω |/n ≤C1

r

1
n log 2

α + |EZξω| ≤C2

r

1
n log 1

α,
(54)

where the last inequation is because EZξω ∼O(n−1) under H0 (see Theorem 1 for a detailed
explanation). Hence |r(n)
ω | ∼O
 p

n log(1/α)

. Also, by definition rω is a constant related to α.

Bound of |bσ2
ω −σ2
ω|. In this part, We first obtain the bound of |bσ2
ω −EZbσ2
ω|, then obtain the bound
of |EZbσ2
ω −σ2
ω|. As before we start by showing the bounded variance property of bσ2
ω. We replace
z1 = (x1, y1) with z′
1 = (x′
1, y′
1) and keep the remaining samples the same. The obtained samples
are named as Z′. For readable, we denote bσ2
ω with sample Z, Z′ as bσ2
ω(Z), bσ2
ω(Z′) respectively.

Recall the definition bσ2
ω := 16
 1

n
P

i( 1

n3
P

j,q,r h(ω)
ijqr)2 −HSIC2
ω(Z)

. Since |h(ω)
ijqr| ≤4, we have
 1

n

X

i

 1

n3
X

j,q,r
h(ω)
ijqr
2
−1

n

X

i

 1

n3
X

j,q,r
˘h(ω)
ijqr
2≤8

n4
X

i

X

j,q,r

h(ω)
ijqr −˘h(ω)
ijqr
,
(55)

20


---Page Break---

 1

n4
X

i,j,q,r
h(ω)
ijqr
2
−
 1

n4
X

i,j,q,r
˘h(ω)
ijqr
2≤8

n4
X

i,j,q,r

h(ω)
ijqr −˘h(ω)
ijqr
.
(56)

Again, the difference terms between h(ω)
ijqr and the new substitution ˘h(ω)
ijqr can only happen in the case
that at least one of i, j, q, r is equal to 1. Hence, Eqs. (55) and (56) are both O(n−1). As a result, bσ2
ω
satisfy the bounded differences property with bound O(n−1). Using McDiarmid’s inequality, with

probability at least 1 −δ, there exist a universal constant C3 such that
bσ2
ω −EZbσ2
ω
≤C3
q

1
n log 2

δ .

Next we obtain the bound of |EZbσ2
ω −σ2
ω|. We rewrite EZbσ2
ω as

EZbσ2
ω = 16
 1

n7
X

ijqrj′q′r′
E[h(ω)
ijqrh(ω)
ij′q′r′] −1

n8
X

ijqri′j′q′r′
E[h(ω)
ijqrh(ω)
i′j′q′r′]

.
(57)

By adding further restrictions that i, i′, j, q, r, j′, q′, r′ are all different, we can obtain the correspond-
ing expression for σ2
ω. Hence the difference between them can only happen when at least one subscript
in i′, j, q, r, j′, q′, r′ is equal to i. Combining |h(ω)
ijqr| ≤4, we have |EZbσ2
ω −σ2
ω| ∼O(n−1).

ϵ-net arguments. Next, we prove the second part with ϵ-net arguments. Take the parameter
space Θk of θk as an example. We choose a cover with N(Θk, rk) points {pi}N(Θk,rk)
i=1
such that
for any point p ∈Θk, we have mini ∥p −pi∥≤rk. According to [41, Proposition 4.2.12], by
comparing the volumes, we have N(Θk, rk) ≤(4RΘk/rk)dk. As for the parameter space Θl of
θl, we can also obtain a cover with N(Θl, rl) points that N(Θk, rk) ≤(4RΘl/rl)dl. Here, we
set rk = 4RΘk/√n, rl = 4RΘl/√n, thus N(Θk, rk) ≤(√n)dk, N(Θl, rl) ≤(√n)dl. Then
combining the Lipschitz property as shown in Lemma 4, we have with probability at least 1 −δ,

sup
(θk,θl)∈Θc

ξω −EZξω
 ≤C1

r

1
n log 2N(Θk, rk)N(Θl, rl)

δ
+ 8RωkLk · rk + 8RωlLlrl

≤C1

r

1
n log 2

δ + (dk + dl)log n

2n
+ 32RωkLkRΘk
√n
+ 32RωlLlRΘl
√n
.

Hence when n is large enough, there exists a positive constant C3, with probability at least 1 −δ,

sup
(θk,θl)∈Θc

ξω −EZξω
≤C3

"r

1
n log 1

δ + (dk + dl)log n

n
+ RωkLkRΘk
√n
+ RωlLlRΘl
√n

#

. (58)

Similar, when n is large enough, there exists a positive constant C4, with probability at least 1 −δ,

sup
(θk,θl)∈Θc

bσ2
ω −σ2
ω
≤C4

"r

1
n log 1

δ + (dk + dl)log n

n
+ RωkLkRΘk
√n
+ RωlLlRΘl
√n

#

.
(59)

Overall Bound. Now we combine the previously obtained results. We have

ξω −r(n)
ω /n
bσω
−EZξω −rω/n

σω

 ≤

ξω −EZξω

bσω

 +


r(n)
ω
−rω
nbσω

 + |EZξω −rω/n| ·

1
bσω
−1

σω

 .

Since on Θc, σω ≥c, according to Eq. (59), we can make bσω ≥c/2 happen by assigning probability
budget δ/2 when n is large enough. Also, combining Eq. (54), |EZξω| ≤1 and rω is a constant
related to α, then with n large enough, there exist positive constants C5, C6,

ξω −r(n)
ω /n
bσω
−EZξω −rω/n

σω

 ≤2

c |ξω −EZξω| + C5

c

r

1
n log 1

α + C6

c3
bσ2
ω −σ2
ω
 .
(60)

Note that we need to pay for probability budget δ/2 for the above conclusion to hold. Then by
taking the supremum on both sides in Eq. (60) and assigning the remained probability budget δ/2 to
Eqs. (58) and (59), we can show that when n is large enough, with probability at least 1 −δ,

sup
(θk,θl)∈Θc


ξω −r(n)
ω /n
bσω
−EZξω −rω/n

σω



∼O

 
1
c3

r

1
n log 1

δ + (dk + dl)log n

n
+ RωkLkRΘk + RωlLlRΘl
√n

!
(61)

thus complete the proof.

21


---Page Break---
H
Proof of Theorem 3

In this section, we give a proof of the Theorem 3. We first restate the Theorem 3 here.
Theorem 3 (Consistency). Let θ∗
k, θ∗
l be the parameters after learning, Zte be the testing samples of
size m, when EZHSIC(u)
ω (Zte) > 0, then the probability of Type II error

P
 
Type II error

= PH1
 
mHSICω(Zte) ≤r(m)
ω
|θ∗
k, θ∗
l

∼O(m−1/2).
(62)

Let the mapping functions with learned parameters θ∗
k, θ∗
l be Tθ∗
k, Tθ∗
l , and the corresponding range
space be a compact subset of RdTx, RdTy , respectively. Also, the diameter of two range spaces is
denoted by diam(Tθ∗
k), diam(Tθ∗
l ), respectively. Let {(ωk;j, ωl;j)}D/2
j=1 be the frequency samplings
with their second moment denoted by σ2
ωk := Epk(ω)[ωT
k;jωk;j], σ2
ωl := Epl(ω)[ωT
l;jωl;j]. Additionally,
we denote ξu := HSIC(X, Y ), then under H1, we have EZHSIC(u)
ω (Zte) > 0 with any constant

probability when D = Ω
 dTx+dTy

ξ2u
log
σωk diam(Tθ∗
k )+σωldiam(Tθ∗
l )

ξu


.

Proof. The proof consists of two parts, we first give the rate of convergence of Type II error under
condition EZHSIC(u)
ω (Zte) > 0, and next for condition EZHSIC(u)
ω (Zte) > 0 we give a lower
bound on the number of frequency samplings required for it to hold. To simplify, we denote the
U-statistic HSIC(u)
ω (Zte) as ξ(u)
ω
in this proof. We begin to prove the first part. With the learned
parameters θ∗
k, θ∗
l , the probability of the Type II error is given by

P

Type II error

= PH1

mξω ≤r(m)
ω
|θ∗
k, θ∗
l

.
(63)

Combing the result of the difference between ξω and ξ(u)
ω
as shown in Eq. (34), we have

PH1

mξω ≤r(m)
ω
|θ∗
k, θ∗
l

≤PH1

mξ(u)
ω
≤r(m)
ω
+ C0
θ∗
k, θ∗
l

,
(64)

where C0 is a positive constant. To apply the rate of convergence of the Central Limit Theorem, we
rewrite the right equation in Eq. (64) as

PH1

 √m(ξ(u)
ω
−EZξ(u)
ω )

4σ1/2
ω
≤r(m)
ω
/√m −√mEZξ(u)
ω
+ C0/√m

4σ1/2
ω

θ∗
k, θ∗
l

!

,
(65)

where the standard deviation (defined in Proposition 1) σω > 0 under H1. Then according to the
results in [34, Section 5.5.1 Theorem B], there exist nonnegative constant C1 such that

P

Type II error

≤Φ

 
r(m)
ω
/√m −√mEZξ(u)
ω
+ C0/√m

4σ1/2
ω

!

+C1νh

σ3/2
ω

1
√m
(66)

where νh := Ei̸=j̸=q̸=r
Z
|h(ω)
ijqr|3 < ∞. When m is large enough, we further have

P

Type II error

≤Φ

C2 −C3
√mEZξ(u)
ω
+ C4/√m

+C5
1
√m.
(67)

where C2, C3, C4 are positive constants and using r(m) ∼O(m1/2) we prove in Eq. (54). Hence
when EZξ(u)
ω
> 0, the leading term √mEZξ(u)
ω
decrease as m increase. Further, to obtain the
decrease rate when m is close to infinity, we consider the asymptotic expansion (when x is close to
negative infinity) for the function Φ(x) as given by

Φ(x) = −e−x2

2x√π

 

1 +

∞
X

n=1
(−1)n 1 · 3 · 5 · · · (2n −1)

(2x2)n

!

,
(68)

thus Φ

C2 −C3
√mEZξ(u)
ω
+ C4/√m

∼O(m−1/2). As a result, the decreasing rate is at least

O(m−1/2). We have so far completed the first part of the proof, and we next begin the second part of
the proof, i.e. obtain the number of frequency samplings required for the condition EZξ(u)
ω
> 0 to

22


---Page Break---
hold. For simplify, we denote Λk(x)T Λk(x′), Λl(y)T Λl(y′) as k(ω)(x, x′), l(ω)(y, y′), respectively.
Then according to Lemma 2, we have

P

sup
x,x′∈X
|k(ω)(x, x′) −k(x, x′)| ≥ϵ

≤28
σωkdiam(Tθ∗
k)
ϵ

2
exp

−
Dϵ2

4(dTx + 2)


,

P

sup
y,y′∈Y
|l(ω)(y, y′) −l(y, y′)| ≥ϵ

≤28
σωldiam(Tθ∗
l )
ϵ

2
exp

−
Dϵ2

4(dTy + 2)


.

(69)

Also, we denote the bounds in Eq. (69) as δx(ϵ, D), δy(ϵ, D), respectively. Next we get the bound
between EZξ(u)
ω
and HSIC(X, Y ). According to Lemma 3, the bound is given by

|EZξ(u)
ω
−HSIC(X, Y )| ≤4 ·
sup
x,x′∈X,y,y′∈Y
|k(ω)(x, x′)l(ω)(y, y′) −k(x, x′)l(y, y′)|.
(70)

Since by the definition, for all (x, x′) ∈X × X, (y, y′) ∈Y × Y, we have |k(ω)(x, x′)| ≤1,
|l(ω)(y, y′)| ≤1, |k(x, x′)| ≤1, |l(y, y′)| ≤1. Hence we have

|k(ω)(x, x′)l(ω)(y, y′) −k(x, x′)l(y, y′)| ≤|k(ω)(x, x′) −k(x, x′)| + |l(ω)(y, y′) −l(y, y′)|. (71)

Combining the results of Eqs. (70) and (71), we obtain

|EZξ(u)
ω −HSIC(X, Y )| ≤4· sup
x,x′∈X
|k(ω)(x, x′)−k(x, x′)|+4· sup
y,y′∈Y
|l(ω)(y, y′)−l(y, y′)|. (72)

Combining the results as shown in Eq. (69) and allocating the probability budget ϵ, we have

P

sup
x,x′∈X,y,y′∈Y
|EZξ(u)
ω
−HSIC(X, Y )| ≥ϵ

≤δx(ϵ/8, D) + δy(ϵ/8, D).
(73)

By setting ϵ = ξu/2, and since ξu > 0 under H1, we conclude that EZξ(u)
ω
> 0 holds with any

constant probability when D = Ω
 dTx+dTy

ξ2u
log
σωk diam(Tθ∗
k )+σωldiam(Tθ∗
l )

ξu


.

In the proof of Theorem 3, we obtain the convergence bound of the statistic EZξ(u)
ω . Actually, the
convergence result for its estimation can also be obtained, as shown in the following Corollary.
Corollary 1 (Approximation Error Bound of HSICω(Zte)). Maintaining the same conditions and
notions as in Theorem 3, we have the uniform convergence bound of HSICω(Zte) as

P

sup
Zte∈X×Y
|HSICω(Zte) −HSICb(Zte)| ≥ϵ

≤δx(ϵ/8, D) + δy(ϵ/8, D).
(74)

Proof. By the definition, we have for all i, j, q, r, |k(ω)
ij | ≤1, |l(ω)
ij | ≤1, |kij| ≤1, |lij| ≤1, thus

|k(ω)
ij l(ω)
qr −kijlqr| ≤|k(ω)
ij
−kij||l(ω)
qr | + |kij||l(ω)
qr −lqr| ≤|k(ω)
ij
−kij| + |l(ω)
qr −lqr|.
(75)

Then according to the results as shown in Eq. (69), we have for all i, j, q, r

P

"

sup
xi,xj∈X,yq,yr∈Y
|k(ω)
ij l(ω)
qr −kijlqr| ≥ϵ

#

≤δx(ϵ/2, D) + δy(ϵ/2, D).
(76)

Recall the definition of h(ω)
ijqr = 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vw −2k(ω)
uv l(ω)
tv and we further define

the corresponding hijqr = 1

4!
P(i,j,q,r)
(t,u,v,w) ktultu + ktulvw −2kuvltv, then for all i, j, q, r,

P

"

sup
xi,xj∈X,yq,yr∈Y
|h(ω)
ijqr −hijqr| ≥ϵ

#

≤δx(ϵ/8, D) + δy(ϵ/8, D).
(77)

After that, using the expressions that we obtained before, i.e., HSICω(Zte) :=
1
n4
P

i,j,q,r h(ω)
ijqr and
HSICb(Zte) :=
1
n4
P

i,j,q,r hijqr, we obtain the final bound that

P

sup
Zte∈X×Y
|HSICω(Zte) −HSICb(Zte)| ≥ϵ

≤δx(ϵ/8, D) + δy(ϵ/8, D).
(78)

and thus complete the proof.

23


---Page Break---
I
Smoothness of Optimization Objective

We first prove the Lipschitz property for some functions. For ease of reference, we re-list here the def-
initions of the terms that related to the optimization objective: ξω := HSICω(Z) =
1
n4
P
i,j,q,r h(ω)
ijqr,

bσ2
ω := 16
h
1
n
P

i( 1

n3
P

j,q,r h(ω)
ijqr)2 −HSIC2
ω(Z)
i
and σ2
ω := 16
h
Ei(Ej,q,rh(ω)
ijqr)2 −
 
EZh(ω)
ijqr
2i
.
The Lipschitz property of these terms are shown as follows.
Lemma 4 (Lipschitz Property of of ξω, EZξω, bσ2
ω, σ2
ω). Maintaining the same conditions and notions
as in Lemma 1, we have the following Lipschitz property

|ξω(θk, θl) −ξω(θ′
k, θ′
l)| ≤8RωkLk · ∥θk −θ′
k∥+ 8RωlLl · ∥θl −θ′
l∥,

|EZ[ξω(θk, θl)] −EZ[ξω(θ′
k, θ′
l)]| ≤8RωkLk · ∥θk −θ′
k∥+ 8RωlLl · ∥θl −θ′
l∥,

|bσ2
ω(θk, θl) −bσ2
ω(θ′
k, θ′
l)| ≤1024RωkLk · ∥θk −θ′
k∥+ 1024RωlLl · ∥θl −θ′
l∥,

|σ2
ω(θk, θl) −σ2
ω(θ′
k, θ′
l)| ≤1024RωkLk · ∥θk −θ′
k∥+ 1024RωlLl · ∥θl −θ′
l∥,

(79)

where we use the symbol ξω(θk, θl) to denote ξω with the parameter θk, θl and the others by analogy.

Proof. We start by obtaining the result of h(ω)
ijqr for all i, j, q, r. Since for all i, j, q, r,
k(ω)
ij (θk)l(ω)
qr (θl) −k(ω)
ij (θ′
k)l(ω)
qr (θ′
l)
≤
k(ω)
ij (θk) −k(ω)
ij (θ′
k)
+
l(ω)
qr (θl) −l(ω)
qr (θ′
l)
,
(80)

where the property |k(ω)
ij | ≤1 and |l(ω)
qr | ≤1 are used. According the Lemma 1, we have
k(ω)
ij (θk) −k(ω)
ij (θ′
k)
≤2RωkLk · ∥θk −θ′
k∥,
l(ω)
qr (θl) −l(ω)
qr (θ′
l)
≤2RωlLl · ∥θl −θ′
l∥.
(81)

Combing the definition that h(ω)
ijqr := 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu l(ω)
tu + k(ω)
tu l(ω)
vw −2k(ω)
uv l(ω)
tv , then
h(ω)
ijqr(θk, θl) −h(ω)
ijqr(θ′
k, θ′
l)
≤8RωkLk · ∥θk −θ′
k∥+ 8RωlLl · ∥θl −θ′
l∥.
(82)

Also, combining the definition of ξω := HSICω(Z) =
1
n4
P
i,j,q,r h(ω)
ijqr, we obtain

|ξω(θk, θl) −ξω(θ′
k, θ′
l)| ≤8RωkLk · ∥θk −θ′
k∥+ 8RωlLl · ∥θl −θ′
l∥.
(83)

By using |EZ[ξω(θk, θl)] −EZ[ξω(θ′
k, θ′
l)]| ≤EZ[|ξω(θk, θl) −ξω(θ′
k, θ′
l)|], we have

|EZ[ξω(θk, θl)] −EZ[ξω(θ′
k, θ′
l)]| ≤8RωkLk · ∥θk −θ′
k∥+ 8RωlLl · ∥θl −θ′
l∥.
(84)

For the results of bσ2
ω, σ2
ω, we first proof the following results. For all i, j, q, r, i′, j′, q′, r′, we have
h(ω)
ijqr(θk, θl)h(ω)
i′j′q′r′(θk, θl) −h(ω)
ijqr(θ′
k, θ′
l)h(ω)
i′j′q′r′(θ′
k, θ′
l)


≤4 ·
h(ω)
ijqr(θk, θl) −h(ω)
ijqr(θ′
k, θ′
l)| + 4 · |h(ω)
i′j′q′r′(θk, θl) −h(ω)
i′j′q′r′(θ′
k, θ′
l)


≤64RωkLk · ∥θk −θ′
k∥+ 64RωlLl · ∥θl −θ′
l∥,

(85)

where the first inequality holds due to property |h(ω)
ijqr| ≤4. Then we use the expression

bσ2
ω = 16
h 1

n7
X

i,j,q,r,j′,q′,r′
h(ω)
ijqrh(ω)
ij′q′r′ −1

n8
X

i,j,q,r,i′,j′,q′,r′
h(ω)
ijqrh(ω)
i′j′q′r′
i
(86)

and combine the results in Eq. (85). As a result, we obtain that

|bσ2
ω(θk, θl) −bσ2
ω(θ′
k, θ′
l)| ≤1024RωkLk · ∥θk −θ′
k∥+ 1024RωlLl · ∥θl −θ′
l∥.
(87)

In a similar way, we can obtain the corresponding expression of σ2
ω as

σ2
ω = 16
h
Ei,j,q,r,j′,q′,r′h(ω)
ijqrh(ω)
ij′q′r′ −Ei,j,q,r,i′,j′,q′,r′h(ω)
ijqrh(ω)
i′j′q′r′
i
.
(88)

Then we obtain a similar result as before, i.e.,

|σ2
ω(θk, θl) −σ2
ω(θ′
k, θ′
l)| ≤1024RωkLk · ∥θk −θ′
k∥+ 1024RωlLl · ∥θl −θ′
l∥
(89)

which completes the proof.

24


---Page Break---
Also for the term associated with the estimated threshold (recall that it is computed from the first two
moments), we obtain the following properties of E0, V0 as defined in Theorem 1.
Lemma 5 (Lipschitz Property of of E0, V0). Maintaining the same conditions and notions as in
Lemma 4, we have the following Lipschitz property
|E0(θk, θl) −E0(θ′
k, θ′
l)| ≤2C0RωkLk · ∥θk −θ′
k∥+ 2C0RωlLl · ∥θl −θ′
l∥,

|V0(θk, θl) −V0(θ′
k, θ′
l)| ≤128C1RωkLk · ∥θk −θ′
k∥+ 128C1RωlLl · ∥θl −θ′
l∥,
(90)

where the constant C0(n) :=
n2
(n−1)2 and C1(n) :=
n(n−4)(n−5)
(n−1)(n−2)(n−3).

Proof. The expression in Theorem 1, while easy to compute, is not suitable for this part of our proof.
We begin by obtaining equivalent expressions for E0, V0. According to Eq. (39), we have

E0 =

1 −1T (ΛXΛT
X −In)1
n(n −1)

 
1 −1T (ΛY ΛT
Y −In)1
n(n −1)



= C0


1 −1T (ΛXΛT
X)1
n2

 
1 −1T (ΛY ΛT
Y )1
n2


.
(91)

And for V0, we use Eq. (45) and obtain

V0 = 2C1

n4

Tr(ΛXΛT
XHΛXΛT
XH)

Tr(ΛY ΛT
Y HΛY ΛT
Y H)

.
(92)

Also, we define h(k)
ijqr := 1

4!
P(i,j,q,r)
(t,u,v,w) k(ω)
tu k(ω)
tu + k(ω)
tu k(ω)
vw −2k(ω)
uv k(ω)
tv that corresponding to h(ω)
ijqr
and also define h(l)
ijqr := 1

4!
P(i,j,q,r)
(t,u,v,w) l(ω)
tu l(ω)
tu + l(ω)
tu l(ω)
vw −2l(ω)
uv l(ω)
tv . Then we can further rewrite

the term for X as
1
n2 Tr(ΛXΛT
XHΛXΛT
XH) =
1
n4
P

i,j,q,r h(k)
ijqr and for Y by analogy. Then the

properties of h(ω)
ijqr can also be obtained for h(k)
ijqr and h(l)
ijqr, e.g., |h(k)
ijqr| ≤4, |h(l)
ijqr| ≤4 for all
i, j, q, r. Next we start to prove the Lipschitz property of E0, V0. For E0, according to Eq. (81) and
combining the results 0 ≤1T (ΛXΛT
X)1 ≤n2 and that for Y , we can show that

|E0(θk, θl) −E0(θ′
k, θ′
l)| ≤C0

n2
X

i,j

k(ω)
ij (θk) −k(ω)
ij (θ′
k)
+C0

n2
X

q,r

l(ω)
qr (θl) −l(ω)
qr (θ′
l)


≤2C0RωkLk · ∥θk −θ′
k∥+ 2C0RωlLl · ∥θl −θ′
l∥,

(93)

where 1T (ΛXΛT
X)1 = P

i,j k(ω)
ij
by definition and so as for Y . And for V0, we can prove that

|V0(θk, θl) −V0(θ′
k, θ′
l)|

≤2C1

n8
X

i,j,q,r,i′,j′,q′,r′

h(k)
ijqr(θk, θl)h(l)
i′j′q′r′(θk, θl) −h(k)
ijqr(θ′
k, θ′
l)h(l)
i′j′q′r′(θ′
k, θ′
l)


≤128C1RωkLk · ∥θk −θ′
k∥+ 128C1RωlLl · ∥θl −θ′
l∥

(94)

where the last inequation is obtained similar to Eq. (85), thus completes the proof.

The following results extend the results in [30] to the more general case (we only restrict the mapping
functions to satisfy the Lipschitz property, and thus include the Gaussian kernel case of their proof).
Theorem 4 (Smoothness of Optimization Objective). Let the sample of size n be Z, and with a
small positive constant c, let the set of the parameters be Θc := {(θk, θl)|bσω ≥c, V0 ≥c, E0 ≥c} ,
then there exist a nonnegative constant L such that ∥∇(θk,θl)J∥≤L on Θc, where the optimization
objective J := [HSICω(Z) −c
α/n]/bσω is that we used in practice.

Proof. According to Lemma 4, we have shown that HSICω(Z), bσω both fits the Lipschitz condition.
Also, according to Lemma 5, we have shown that E0, V0 are also Lipschitz with respect to θk, θl.
Since the threshold c
α is completely determined by these two moments, combining the smoothness
property of the mapping from two moments to thresholds as analyzed in [30, Theorem 2], we obtain
that c
α is also Lipschitz with respect to θk, θl on Θc. As a result, we complete the entire proof based
on the Lipschitz property of composite mappings.

Remark. E0 and V0 are positive is almost satisfied in practice since according to the definition only
[1T Λ·2
Xc1][1T Λ·2
Y c1] and [1T (ΛT
XcΛXc)·21][1T (ΛT
Y cΛY c)·21] need to be greater than 0.

25


---Page Break---
J
Details of Experiment Setup

In this section, we give an introduction to the comparison methods in our experiments and provide
the implementation details of each method.

J.1
Details of Comparison Methods

The methods of comparison used in the experiment are described below.

• dCor [39]: An independence test that is based on the distance covariance.
• QHSIC [14]: The original quadratic-time HSIC independence test.
• RDC [26]: The randomized dependence coefficient that measures the independence using the
canonical correlation between a finite set of random features of the copula.
• NyHSIC [44]: A variant of HSIC that uses the Nyström method to approximate kernels.
• FHSIC [44]: A variant of HSIC that uses the random Fourier feature to approximate kernels.
• BHSIC [44]: A variant of HSIC with the block-based statistic.
• HSICAgg [32]: An aggregated kernel test with the incomplete statistic of HSIC.
• NFSIC [18]: A test uses the normalized version of the finite set independence criterion and
chooses features on a hold-out validation set to optimize a lower bound on the test power.

Below are the GitHub URLs for each comparison method.

• dCor: https://pypi.org/project/dcor.
• QHSIC: https://github.com/amber0309/HSIC/blob/master/HSIC.py.
• RDC: https://github.com/lopezpaz/randomized_dependence_coefficient.
• NyHSIC: https://github.com/oxcsml/kerpy/blob/master/independence_testing.
• FHSIC: https://github.com/oxcsml/kerpy/blob/master/independence_testing.
• BHSIC: https://github.com/oxcsml/kerpy/blob/master/independence_testing.
• HSICAgg: https://github.com/antoninschrab/mmdagg/tree/master/mmdagg.
• NFSIC: https://github.com/wittawatj/fsic-test/blob/master/fsic.

Time Complexity. Among them, dCor and QHSIC are the tests of quadratic complexity with sample
size n, i.e., O(n2). RDC is calculated in O(n log n) and the rest are linear-time tests, i.e., O(n).

Threshold. For dCor, QHSIC, RDC, NyHSIC, FHSIC, and BHSIC, we permute the samples 100
times to simulate the null distribution and compute the threshold. The thresholds for the remaining
methods are obtained by asymptotic null distribution, i.e., we set the test threshold to the (1 −α)-
quantile of χ2(J) for NFSIC and obtain the test threshold of LFHSIC-G/M by gamma approximation.

Details of Setup. The number of random features for FHSIC, LFHSIC-G/M, the number of induced
variables for NyHSIC, the block size for BHSIC and the number of sub-diagonals R for HSICAgg
are all kept consistent as recommended in [44] for fair evaluation. Specifically, we set the number of
random mappings in RDC to 20 to ensure compatibility with large-scale datasets. The test location
parameter J of NFHSIC is set as default as 10, since it differs from other approximation methods that
blindly increasing J may lead to a loss of power as shown in [18] and can significantly escalate time
costs due to its cubic time complexity O(J3). In the optimization step, for stabilizing the training,
in the implementation of NFSIC we determine the initial bandwidth by searching the best from 25
bandwidth combinations (including the median bandwidth combination). For LFHSIC-G/M, to be
fair, we perform the same grid search on SD, Sin, and GSign datasets. In other experiments, we
still use the median bandwidth as initialization for LFHSIC-G/M. Also, the maximum number of
iterations for the optimization is set to 100 for NFSIC and LFHSIC-G/M. The default learning rate of
the optimization of LFHSIC-G/M is set as 0.05 in all the experiments. As for HSICAgg, the default
implementation of the predefined 25 pairs of bandwidths in its code is used. For synthetic data, we
set the split ratio to 0.5 for NFSIC and LFHSIC-G/M, i.e., we randomly sample half of the data for
training and use the remaining for independence testing, while the other methods use all data for
testing. For real MSD data, we divide a small portion of the data for training and then extract 100
random subsets of the remaining data (disjoint from the training set) for evaluation.

26


---Page Break---
J.2
Details of Datasets

The details of the four synthetic datasets and two real datasets are described below.

• Sine Dependency (SD): In this model, X follows a d-dimension multivariate normal distribution
Nd(0, Id), and Y is defined as 20 sin
 
4π(X2
1 + X2
2)

+Z, where Xi is the i-th dimension of X,
and Z ∼N(0, 1) represents independent noise. Notably, when d > 2, Y exhibits a nonlinear
relationship solely with the first two dimensions of X.
• Sinusoid (Sin): This model introduces a localized alteration in the probability density function
pxy over X × Y := [−π, π]2, specified as (X, Y ) ∼pxy(x, y) ∝1 + sin(ωx) sin(ωy), where ω
denotes the frequency. Increasing the frequency enhances the similarity between the sampled
data and that drawn from Uniform([−π, π]2), thereby augmenting the challenge of detecting
dependency with limited sample sizes. An example visualization is shown on the left of Fig. 6.
• Gaussian Sign (GSign): In this model, X follows a d-dimension multivariate normal distribution
Nd(0, Id), and Y is expressed as |Z| Qd
i=1 sgn(Xi), where sgn(·) represents the sign function,
Xi denotes the i-th dimension of X, and Z ∼N(0, 1) is independent of X. The challenge lies
in Y being independent of any proper subset of X but dependent on X as a whole, underscoring
the importance of considering all dimensions of X simultaneously in independence testing.
• ISA Dataset. We construct the data through the following steps: First, we generate n i.i.d samples
of two univariate random variables with a mixture of Gaussian model, i.e.
1
2N(−1, 0.01) +
1
2N(1, 0.01). Second, we mix these random variables using a rotation matrix parameterized by
an angle θ, which varies from 0 to π/4. A zero angle implies independence between the data,
while a larger angle signifies stronger dependency. Third, we append noise with a distribution
of Nd−1(0, Id−1) to each of the mixtures. Finally, we multiply an independent random d-
dimensional orthogonal matrix to obtain vectors dependent across all observed dimensions. The
resulting random variables X and Y are dependent but uncorrelated. When d is greater than 1,
the problem is associated with the independent subspace analysis (ISA) problem [14]. For the
case d = 1, θ = π/10, an example visualization is shown in the middle of Fig. 6.
• 3DShapes Dataset. This dataset [5] comprises images depicting 3D scenes, complete with
additional features like shadows and backgrounds. It encompasses six fundamental latent factors:
floor hue, wall hue, object hue, object scale, object shape, and orientation, all adjustable to
generate corresponding images. Orientation is treated as a dependency factor for independence
testing, where we test the dependency between the image X and its orientation Y . To heighten
the challenge, we maintain the object shape as a ball, minimizing the apparent orientation feature
compared to other shapes like squares, while randomizing the remaining factors. An example
visualization is given on the right side of Fig. 6.
• Million Song Dataset. The dataset, a subset of the Million Song Data3 [4], comprises 515, 345
songs with 91-dimensional features. The first dimension represents the release year of each
song, designated as variable Y , while the remaining features (e.g., timbre average and timbre
covariance) form variable X. Our objective is to identify the dependency between X and Y .

−2.5
0.0
2.5
X

−2

0

2

Y

n = 5000, ω = 2

0.012

0.012

0.012

0.012

0.012

0.012

0.012

0.012

0.018

0.018

0 018

0.018

0.018

0.018

0.018

0.024

0.024

0.024

0.030

0.030

0.030

0.030

0.030

0.030

0.030

0.030

0.036

0.036

0.036

0.036

0.036

0.036

0.036

0.042

−2
0
2
X

−2

−1

0

1

2

Y

n = 1000, θ = π/10

0.04

0.04
0.04

0.04

0.08

0.08

0.08

0.08

0.12

0.12

0.12

0.12

0.16

0.16

0.16

0.16

0.20

0.20

0.20

0.20

0.24

0.24

Y(Orientation)

X(Image)

Floor hue

Wall hue

Object hue

Object shape

(ball)

Object scale

Figure 6: Examples of visualization of samples from different datasets. The two plots on the left
correspond to the samples and their contour under Sin dataset (n = 5000, w = 2), and ISA dataset
(n = 1000, d = 1, θ = π/10), respectively. Right: a visualization of the causal diagram of the data
generation process and some generated examples.

3Million Song Data subset: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd

27


---Page Break---
K
Additional Experiment Results

In this section, we provide additional experimental results, mainly including the visualization results
on the Sin synthetic dataset, the results with more comparing methods (as explained in the main
paper, due to the high time overhead therefore do not participate in the evaluation of the main paper)
as well as the running time of each method.

K.1
The visualization results on the Sin model

We provide the visualization results on the Sin synthetic dataset to illustrate the performance of
our optimization objective for the example mentioned in the main paper. The results are shown in
Fig. 7, where the setup follows our experiments (n = 2000, ω = 5, D = 100). For visualization,
the negative of our optimization objective J is shown. As can be seen, our optimization objective
guides to letting the bandwidth adapt to improve the test power, and here we can see that regions
with bandwidths around 0.2 (corresponding to the theoretical optimal solution in our main paper)
indicate better power, thus corroborating the validity of our optimization objective. Also, notice that
the landscape is smooth over a wide range as demonstrated in Theorem 4, and thus contributes to
non-convex optimization.

−2
0
2
X

−3

−2

−1

0

1

2

3

Y

n = 2000, ω = 5

Width of X

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

Width of Y

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

Optimization objective: −J

−0.20

−0.15

−0.10

−0.05

0.00

0.05

0.10

Mid-widths

−0.210

−0.175

−0.140

−0.105

−0.070

−0.035

0.000

0.035

0.070

0.105

Figure 7: The visualization results on the Sin model. Left: the samples with n = 2000, ω = 5. The
visualization of the landscape of the negative of our optimization objective (D = 100).

K.2
Additional experimental comparisons

As mentioned in the main paper, the method4 [30] is not involved in the comparison because it takes
too much time to run in some settings. Here, we compare it with ours under some feasible settings to
illustrate the improvement of our method on power-runtime trade-offs. The methods using Gaussian
kernel with global width and Gaussian kernel with widths of each dimension (corresponding with
ours) are employed, referred to QHSIC-O and QHSIC-W, respectively. For fairness, the same grid
search procedure is employed as the initialization of the optimization. We perform the evaluation
on the SD data and plot the results of the test power over time as shown in Fig. 8. Also, for our
methods, we provide the results under the setting D = 100 and D = 500 as in the main paper. The
experiments are conducted with the same equipment, specifically a 6-core CPU with a 3080 GPU.

Results. Our test consistently results in a better power-runtime tradeoff at different D settings. At
D = 100, one test can be completed in less than a second when the sample size n = 6000. As D
increases (D = 500), the number of samples required to achieve the same power decreases, but the
increase in D leads to an overhead in runtime, and overall our test is still completed in a few seconds.
In contrast, even though QHSIC-O/W requires fewer samples to reach the same power than our tests,
the runtime rises rapidly as the sample size increases. When n = 1000, it already takes more than
10s to perform a test. When n = 3000, it needs nearly a minute to perform a test, which may greatly
limit the practical application.

4The code is downloaded from https://github.com/renyixin666/HSIC-LK

28


---Page Break---
0.7
1.0
2.0
3.0
5.0
10.0
20.0
40.0
60.0
Average running time (s)

0.2

0.4

0.6

0.8

1.0

Test power ( ↑)

n=1000

n=3000

n=1000

n=3000

n=4000

n=4000
n=6000

n=6000

n=2000

n=2000

n=4000

n=4000

QHSIC-O
QHSIC-W
LFHSIC-G D=100
LFHSIC-M D=100
LFHSIC-G D=500
LFHSIC-M D=500

Figure 8: Time-power trade-off curves on the SD dataset.

K.3
Running Time

In this part, we evaluate the running time of each method on ISA datasets d = 10. We set D = 100
and plot the results of the running time versus sample size n as shown in Fig. 9. Shown on the left are
the results of tests with O(n) and O(n log n) time complexity. Since the quadratic time complexity
test cannot handle large-scale inputs of 100,000 sample size (excessive runtime and large memory
overhead to store the kernel matrix), we evaluate them separately on the right. The experiments are
all conducted on the same equipment, specifically a 14-core CPU with a 4090 GPU.

0
20000
40000
60000
80000 100000
Sample size

0

10

20

30

40

50

60

Running time (s)

RDC
NyHSIC
BHSIC
FHSIC
HSICAgg
NFSIC
LFHSIC-G
LFHSIC-M

0
1000
2000
3000
4000
5000
Sample size

0

50

100

150

200

Running time (s)

dcor
QHSIC
QHSIC-O
QHSIC-W
LFHSIC-G
LFHSIC-M

Figure 9: The running time curves with sample size n on the ISA dataset (d = 10).

Results. The experimental results on the left show that our method is faster than other methods with
optimizable options (HSICAgg and NFSIC), and can complete a test within 10 seconds even with
100, 000 samples. Even though HSICAgg uses parallelism to optimize the computational efficiency
of the scheme, the actual implementation of the parallelism is time-consuming, and hence leads to
a high time overhead for a single practical test. For the results on the right, it can be seen that the
quadratic complexity methods face a dramatic increase in time overhead as the sample size rises, and
this is especially severe for the methods (QHSIC-O/W) that need to be optimized since the optimizing
objective needs to perform multiple squared complexity operations. In contrast, our linear-time
learning objective allows us to handle huge data samples very efficiently.

L
Limitations and Broader Impacts

Limitations. According to the experimental results in the main paper as well as in the Appendix, no
one method is better than the others in all settings, so it is important to choose several appropriate
tests for real scenarios and summarize their results in order to obtain a more reliable conclusion.

Broader Impacts. This work proposes a novel framework for independence testing. The proposed
linear-time optimization objective can be trained end-to-end in a data-driven manner, ensuring both
effectiveness and efficiency in high-dimensional and large-scale scenarios. This could be beneficial
for developing more reliable downstream algorithms in a variety of areas, including causal discovery,
feature selection, and deep learning.

29


---Page Break---
NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and precede the (optional) supplemental material. The checklist does NOT
count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: I’ve clearly stated the contribution.
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
Justification: See the Sec. L in the Appendix.

30


---Page Break---
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
Justification: See Sec. 5 in the main paper. Also, the summarized assumptions and the
proofs are provided in Appendix.
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
Justification: We provide the details to reproduce the main experimental results, please See
Sec. J in Appendix, and we also provide the experimental data/code in the supplemental
material.
Guidelines:

31


---Page Break---
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
Answer: [Yes]
Justification: We provide the experimental data/code in the supplemental material.
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

32


---Page Break---
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: we provide the details. See Sec. J in Appendix.

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

Justification: The experimental results are accompanied by statistical significance tests. See
Sec. 6.

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

Justification: We provide sufficient information on the computer resources. See Sec. K in
Appendix.

Guidelines:

• The answer NA means that the paper does not include experiments.

33


---Page Break---
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
Justification: I read it.
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
Justification: See the Sec. L in the Appendix.
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
Justification: [TODO]

34


---Page Break---
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

Justification: We cite the methods used and list the URLs of the comparison methods.

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

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: [TODO]

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

35


---Page Break---
Answer: [NA]
Justification: [TODO]
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
Justification: [TODO]
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

36


---Page Break---
