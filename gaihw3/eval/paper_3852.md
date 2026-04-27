Stability and Generalizability in SDE Diffusion Models
with Measure-Preserving Dynamics

Weitong Zhang1
Chengqi Zang2
Liu Li1
Sarah Cechnicka1

Cheng Ouyang1,3
Bernhard Kainz1,4

1 Imperial College London, UK, 2 University of Tokyo, JP, 3 University of Oxford, UK
4Friedrich-Alexander University Erlangen-Nürnberg, GER
weitong.zhang20@imperial.ac.uk

Abstract

Inverse problems describe the process of estimating the causal factors from a
set of measurements or data. Mapping of often incomplete or degraded data to
parameters is ill-posed, thus data-driven iterative solutions are required, for ex-
ample when reconstructing clean images from poor signals. Diffusion models
have shown promise as potent generative tools for solving inverse problems due to
their superior reconstruction quality and their compatibility with iterative solvers.
However, most existing approaches are limited to linear inverse problems repre-
sented as Stochastic Differential Equations (SDEs). This simplification falls short
of addressing the challenging nature of real-world problems, leading to amplified
cumulative errors and biases. We provide an explanation for this gap through
the lens of measure-preserving dynamics of Random Dynamical Systems (RDS)
with which we analyse Temporal Distribution Discrepancy and thus introduce
a theoretical framework based on RDS for SDE diffusion models. We uncover
several strategies that inherently enhance the stability and generalizability of dif-
fusion models for inverse problems and introduce a novel score-based diffusion
framework, the Dynamics-aware SDE Diffusion Generative Model (D3GM). The
Measure-preserving property can return the degraded measurement to the original
state despite complex degradation with the RDS concept of stability. Our extensive
experimental results corroborate the effectiveness of D3GM across multiple bench-
marks including a prominent application for inverse problems, magnetic resonance
imaging.

1
Introduction

Diffusion probabilistic models [53, 54, 52] have demonstrated impressive performance across various
image generation tasks, primarily by modeling a diffusion process and then learning an associated
reverse process. Among the many commonly used approaches [63], diffusion models that incorporate
the concept of score functions [28, 55] can capture the intrinsic random fluctuations of the forward
diffusion process, positioning them as a good choice for in-depth analysis. Score-based generative
models (SGMs) entail gradually diffusing images towards a noise distribution, and then generating
samples by chaining the score functions at decreasing noise levels with score-based sampling
approaches. One such example of an SGM with a score-based sampling technique, known as score
matching [54], has gained popularity for density estimation [16]. It employs methods such as

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Langevin dynamics [21, 43, 30] and SDEs [29, 37, 31, 71] to simulate the underlying probability
distribution of training samples.

However, vanilla unconditional SGMs can be extended to inverse problems by leveraging an implicit
prior distribution, based on the available counterpart measurement, subjected to corruption and/or
noise. To this end transitionary SGMs enable an iterative recovery of the data from this noisy
counterpart, instead of relying on Gaussian white noise as a starting point [66, 37, 39, 12, 20, 34].

Intuitively, leveraging priors and a generative capacity into transitionary SGMs offers the possibility
of exploring high quality reconstruction and restoration and gaining better performance. However,
current transitionary SGMs that incorporate priors have largely overlooked the unreliable quality of
the prior and its measurement. Empirically, transitionary SGMs cannot always be trusted in terms of
stability and efficiency, especially in a regime of non-uniformly distributed noise or corrupted signal
quality [9, 66, 37]. Hence, the exploitation of transitional learning within SGMs does not come
without costs as their advantages vanish in limited data quality settings. Theoretical understanding is
notably lacking in this field with the following fundamental open problem: Can we realize reliable
transitionary diffusion processes in practice for inverse problems with a theoretical guarantee?

While recent works have started to lay down a theoretical foundation for these models, a detailed
understanding is still lacking. Current best practice advocates for smaller initialisation values (e.g.,
noise schedule [66, 20], instead of large values [15, 61] to ensure that the forward dynamics brings
the diffusion sufficiently close to a known prior and simple noise distribution. However, a proper
choice of the values conditioned on the prior within a theoretical framework should be preferred
for a better approximation of the score-matching objective and higher computational efficiency. To
fully facilitate the power of reversion and generation of transitionary SGMs and to mitigate the
influence of low-quality measurements for solving inverse problems, this paper provides a measure-
preserving dynamics of random dynamical system (RDS) perspective as a promising way to obtain
reliable reversion and generation. Notably, our ‘measure’ is not only the observations (e.g., degraded
images), but also represents the invariant probability measure (distribution) of the RDS. This allows
to consider the concept of an RDS stability and to frame challenging degradation learning within a
measure-preserving dynamical system. Thus, we can start from a transitionary SGM interpretation of
diffusion models and connect RDS to the SDE in transitionary SGMs. The pitfalls (e.g., Instability)
are discussed in Sect. 3 and further implications can be found in the Appendix. Transitionary SGMs
have not been fully explored before, and we provide a theoretical interpretation of a stationary process
as a possible solution.

Our D3GM framework is abstracted from transitionary SGMs. The key to our framework is a station-
ary process following measure-preserving dynamics to ensure the stability and generalizability of the
diffusion, as well as reducing the influence of accumulated error, distribution bias and degradation
discrepancy. Our contributions can be summarised as follows:

1. Temporal Distribution Discrepancy: We conduct a rigorous theoretical examination of the
instability issue of transitionary SGMs, measured as Temporal Distribution Discrepancy (i.e., lower
bound of modeling error). This analysis sheds light on critical aspects related to stability and
generalizability1, effectively addressing an unexplored fundamental gap in the understanding of
solving challenging inverse problems with SDEs.

2. D3GM Framework: We propose a solution, D3GM, and an explanation from measure-preserving
dynamics of Random Dynamical Systems (RDS). ‘Measure’ includes both measurements (degraded
image) and invariant measures (distribution) of RDS, which allows complex degradation learning and
enhances restoration and reconstruction accuracy.

3. Thorough Evaluation: Our contributions are substantiated by extensive validation. We demon-
strate the practical benefits of our D3GM framework across various benchmarks, including challenging
tasks such the reconstruction of Magnetic Resonance Imaging (MRI) data.

We address the instability of diffusion models for inverse problems under domain shift and concept
drift (unknown and heterogeneous degradation). This leads to what we believe is a completely
novel view on the theoretical foundation of how the degradation process is modelled. The result
is an approach that is more in line with the original intention of the theory of diffusion. We

1Generalizability refers to the extent to which out-of-distribution and domain shift impacts the fidelity of the
restoration process. Stability in SDE diffusion models is demonstrated by its resilience to degradation beyond its
domain and its consistent ability to restore high-quality images.

2


---Page Break---
chose inverse problems as a relevant application area to demonstrate our ideas but also included a
variety of challenging problem settings to explore the generalizability of D3GM. To the best of
our knowledge, no other method can handle a diverse range of challenging tasks like real-world
dehazing, compressed MRI reconstruction, blind MRI super-resolution, etc. with a unified underlying
theoretical framework. In Tab. 1 we illustrate the key differences of D3GM compared with SGMs
and transitionary SGMs.

2
Preliminaries

SGMs. We will follow the typical construction of the diffusion process x(t), t ∈[0, T] with
x(t) ∈Rd. Concretely, we want x(0) ∼p0(x), where p0 = pdata , and x(T) ∼pT , where pT is a
tractable distribution that can be sampled. In this work, we consider the score-based diffusion form
of the SDE [55]. Consider the following Itô diffusion process defined by an SDE:
dx = f(x, t)dt + g(t)dW ,
(1)
where f : Rd 7→Rd is the drift coefficient of x(t), g : R 7→R is the diffusion coefficient coupled
with the standard d-dimensional Wiener process w ∈Rd. By carefully choosing ¯f, g, one can
achieve a spherical Gaussian distribution as t →T. For the forward SDE in Eq. 1, there exists a
corresponding reverse-time SDE [3, 55]:
dx = [f(x, t) −g(t)2 ∇x log pt(x)
|
{z
}
score function

]dt + g(t)dW ,
(2)

where dt is the infinitesimal negative time step, and w is the Brownian motion running backwards.
The score function ∇x log pt(x) is in general intractable and thus SDE-based diffusion models
approximate it by training a time-dependent neural network under a score function [57, 28].

Transitionary SGMs. [17, 37, 66, 34, 20, 12] leverage a transitionary iterative denoising paradigm
for the inverse problems. In inverse problems, such as super-resolution, we have an (nonlinear,
partial, and noisy) observation y of the underlying high-quality signal x. The mapping x 7→y is
many-to-one, posing an ill-posed problem. In this case, a strong prior on x is needed for finding a
realistic solution. Formally, the general form of the forward (measurement) model is:
y = A (x) + n,
y, n ∈Rn, x ∈Rd,
(3)
where A(·): Rd 7→Rn2, oftentimes n ≪d is the forward measurement operator and n is the
measurement noise, assuming n ∼N
 
0, σ2I

. While sharing a similar aim of bridging y and x in
transitionary SGMs, different mathematical frameworks have been used: [17] employs Inversion by
Direct Iteration; [37, 34, 20] model it as a Mean-reverting SDE.

Transitionary SGM has become an increasingly important line of SDE research due to the applicability
on images with theoretical guarantees. However, they often perfrom poorly in real-world scenarios. To
provide a theoretical investigation of this gap, we interpret Transitionary SGM as Ornstein-Uhlenbeck
(OU) process. This perspective allows us to understand the random fluctuations in image degradation
as stochastic processes, providing a foundation to integrate random dynamical systems (RDS) with
the diffusion process as a natural extension of the SDE framework involving the OU process.

The Measure-preserving property is introduced from the perspective of RDS: The distribution can
still return to the original state despite severe degradation. Our approach constructs a bridge from
measure-preserving dynamical system to transitionary SGM through measure-preserving dynamics,
and highlights the Temporal Distribution Discrepancy in Sect. 3. Subsequently, we address this issue
of instability: by incorporating a measure-preserving strategy into the solution of inverse problems,
which is detailed in Sect. 4. This covers counterpart modeling, bridging a transition from uncertain
diffusion modeling to deterministic solutions, yielding significant improvements in both performance
and efficiency as demonstrated in Sect. 5. More details can be found in Sect. 6 and Appendix.

3
Instability Analysis: Transitionary SGMs with Corrupted SDE Diffusion

Ornstein-Uhlenbeck (OU) process. An OU process is a common case in transitionary SGMs, where
xt is defined using an SDE: dxt = −θtxtdt + σtdWt. Wt is standard Brownian motion. A drift term
µ can be introduced:

2MRI signals are defined on Cn and Cd. We demonstrate in Sect. 5 that our approach is applicable to MRI.

3


---Page Break---
Table 1: Differences between state-of-the-art SDE diffusion-based approaches.

Model
p(X0)
p(X1)
Theorem Foundation
Properties
TDD (Prop.2)
Attractor
Operator Dir.
Inverse problem solver
SGM [53]
pA
N(0, I)
SDE diffusion
No prior
Unsolvable
No subset attracts
one-sided
IR-SDE [37]
pA
pB(·|X0, µ)
Mean-reverting SDE
Instability
Limited
Gaussian
one-sided
Lin/Non-Lin
I2SB [34]
δa
pB(·|X0)
Schrodinger Bridge
Strict Prior
Limited
No subset attracts
one-sided
Lin/ Non-Lin

D3GM (ours)
pA
pB(·|X0, µ, τ)
Measure-Preserving RDS
Stability
Robust
N(µ, τ 2σ2I)
two-sided
Lin/ Non-Lin/Blind

dxt = θt(µ −xt)dt + σtdWt,
(4)

where µ denotes state mean, reflecting the expected state of the measurement (e.g., corrupted
image [37], noisy speech [59]) over time. θt, σt are time-dependent parameters. The drift term
corrects deviations from the constant µ, effectively pulling the process towards µ (t →∞) with
Stability (in Appx. G) as opposed to pure noise in Eq. 1.

Measure-preserving Dynamics in SDE Diffusion. The solution of the above SDE can be represented
by a continuous-time random dynamical system φ defined on a complete separable metric space
(X, d). (See precise definition of RDS in Appx. C). More generally, we can extend the RDS to a
two-sided solution operator with a flow map. The base flow driven by Brownian motion can be
written as W (t, ϑs(ω)) = W(t + s, ω) −W(s, ω).

Proposition 1 After extending the solution of the OU process to RDS, the measure-preserving RDS
φ should meet the property φ(t, s; ω)x = φ (t −s, 0; ϑsω) x. However, OU processes with time-
varying coefficients usually do not satisfy this property. In this situation, the system breaks the
forward-reverse processes, making it difficult to maintain stability.

Intuition 1. A two-sided measure-preserving random dynamical (MP-RDS) system for-
mulation enables us to use the Poincare recurrence theorem [45], (see precise statement in
Appx.D), intuitively, with a two-sided MP-RDS φt, the Poincare recurrence theorem ensures
that the system φt starts from terminal condition xT , run backward in time, will hit a region
(x0 −ϵ, x0 + ϵ) for small ϵ in finite time, where x0 is the high-quality image.

Example 1. Following Intuition 1. and Prop. 1, suppose that the OU’s θt follows a cosine
schedule, such that θt = cos(t) for 0 ≤t ≤T, then for some 0 ≤s ≤t ≤T, φ(t, s; ω)x ̸=
φ (t −s, 0; ϑsω) x because the change of θt w.r.t. time is not uniform. The OU-process
instability exists due to Temporal Distribution Discrepancy (Prop. 2).

At a high level, Proposition 1 can be extended to show that there exists a compact attracting set at any
−∞< t < ∞, and this convention has allowed us to characterize the attractor K(ω) = N(µ, σ2/2θ).
The closed-form distribution for y can be complex and may not be tractable depending on the
particular scenarios of the actual image degradation process µ. The modification of σ and θ is used
to regularize the perturbation and attempt to close the distribution. However, these injections might
bypass the stationary process. More details can be found in Appx. D.

Instability-Temporal Distribution Discrepancy. Given the process OU (xt, µ; t, θ) with Eq. 4,
where xT ̸= x∞for finite T, indicates that the perturbed state cannot move towards the degraded LQ
image and fails in matching the theoretical distribution. This inherent discrepancy further causes bias
in the estimation of µt, which gradually accumulates into error in the reverse process.

Temporal Distribution Discrepancy is illustrated by Proposition 2 (proof can be found in Appx. E):

Proposition 2 Given Eq. 3 and Eq. 4, and assume that the score function is bounded by C in L2
norm, then the discrepancy between the reference and the retrieved data is, with probability (1 −δ)
at least:

∥x0 −OU(x0, µ; T, θ)∥2
2 ≥|
 
(x0 −µ)2 −σ2
T /2θT

e−2¯θT + σ2
T /2θT

−σ2
max

Cσ2
max + d + 2
p

−d · log δ −2 log δ

|,
(5)

4


---Page Break---
Intuition 2. Intuitively, Proposition 2 provides a theoretical measurement on how the differ-
ence between finite iteration distribution and the asymptotic distribution of the OU-process in
L2 could further enlarge the discrepancy between the retrieved image and the actual HQ image.
Discrepancies are typically explained by complex degradation vs. monotonic modeling.

For a noisy inverse problem, the retrieved data with any finite T depends on σt, µt, λ, T, ¯K (Lipschitz
constant). This proposition also correlates to [39], where the lower bound of the distance between
the high quality image and retrieved image in L2 norm in our model is further enlarged by this
discrepancy, which correlates to the term
 
(x0 −µ)2 −(σ2
T /2θT )

e−2¯θT + σ2
T /2θT .

Another way to further minimize this bound is through the term e−2¯θT with [0, T] normalized to [0, 1].
What we refer to as θ-schedule corresponds to the exact functional form of θt, several schedules
can be set here, e.g., constant, linear, cosine, and log. At a high level, the discrepancy between the
reference and the retrieved data stems from the divergence between the forwarded final state and the
low quality image. Eq. 5 can be factored into three constituent parts: the data residual, the stationary
disturbance, and the random noise. While conditional diffusion generation entails a trade-off between
variability and faithfulness [66], the persistent discrepancy within the residual has a significant impact
on the generalizability of solving the transitionary tasks. This also establishes a connection with
SDEdit [39] and CCDF [12]. When fitting inverse problems involving paired data into diffusion
models, while accounting for deviations and degradation, inserting them into Eq. 4 directly may not
be the most effective strategy.

During sampling and inference with the degraded input y, the discrepancy identified in Prop. 2
intensifies. The complex degradations in y exacerbate the divergence from the expected µ distribution,
significantly impacting the accuracy of the restored data ˆx0. More details are in Appx. F.

4
Towards Stability: Measure-preserving Dynamics in SDE Diffusion

In Sect. 3, we extrapolate and theorize the Temporal Distribution Discrepancy on µ and x in the
diffusion model for the inverse problem. Our key idea is to combine the stationary process to
alleviate the Temporal Distribution Discrepancy problem following the measure-preserving dynamics
from RDS. Recall that our ‘measure’ is not only the measurements (i.e., degraded image), but also
represents the invariant measure (distribution) of the RDS.

We begin by describing the forward and reverse processes of the D3GM, which serves as a stable
bridge between the quality data and the counterpart measurement. We adapt score-based training
methods to estimate this SDE. Following this, we describe the essential constructions for preserving
the stationary process in the diffusion model and solving for Temporal Distribution Discrepancy on
an orthogonal basis compared to current transitionary diffusion models.

Measure-preserving Dynamics with the Stationary Process.
Following Prop. 1, in a
SDE Diffusion from 0 to T, the corresponding ‘attractors’ (states) can be viewed as
N

µ + (x0 −µ)e−¯θt, σ2
t (1 −e−2¯θt)/2θt

. We can guide SDE Diffusion towards a stable and
robust solution based on the properties of Measure-preserving in RDS. It can be extended to impose
that for every t, σ2
t
2θt = λ2, where λ is the variance of the designated stationary measure forward
process. This convention allows us to reduce the regularization on two variables σt, θt to just one
variable to satisfy the property of the measure-preserving dynamics in the asymptotic sense, i.e.,
limt→∞φ(t, s; ω)x = limt→∞φ (t −s, 0; ϑsω) x. This convention allows us to characterize the
attractor of the system as K(ω) = N(µ, λ2).

The definition and constraint of the attractor are significant; without imposing this constraint, the
measure-preserving property cannot be maintained, and the system would degrade into a Coefficient
Decoupled SDE (Coe. Dec. SDE), we also analyse this in Fig. 2 and Tab. 11 in Appx. H.

5


---Page Break---
High Quality Image
Output
Perturbed Image
Forward Process 
Reverse Process

Stationary Process
Measure-preserving

Dynamics

Distribution Retrieval

Low Quality Image

D𝟑𝑮𝑴
GT
LQ

Rain Removal
Real Dense Haze Removal 
MRI Reconstruction
MRI Super-resolution

16x
8x

Figure 1: Dynamics-aware SDE Diffusion Generative Model (D3GM). When extending transitionary
SDEs to random dynamical systems (RDS), their measure-preserving property should be kept to
maintain stability. This corresponds to driving the SDE towards the drift term µ (LQ). There is a
Temporal Distribution Discrepancy which results from the gap between the forward estimation xT
and the low quality image in the SDE. With the distribution aligned between xT and µ, the SDE
can be made more robust to inverse problems. Reconstruction results for low quality (LQ) images
after application of our D3GM method, on different tasks, compared to the ground truth (GT) on two
domains - The frequency domain: MRI Reconstruction (undersampling factor 8x, 16x, frequency
masks are colored red); MRI Super-resolution (up-scaling factor of X4, cross-domain evaluation).
The image domain: Real Dense Haze Removal; Rain Removal (light, heavy).

Example 2. When σ

θ →∞, the attractor becomes excessively large, reducing the significance
of µ. SDEs exhibiting this behavior are defined as Coefficient Decoupled SDEs. In practice, µ
demonstrates non-infinite properties as an input image, while the corresponding sigma and
theta are indeed unconstrained. In such decoupled forms, the coefficients of the attractor size
increases σ

µ, diminishing the significance of µ.

Based on Prop. 2, since the temporal distribution discrepancy always exists as long as the running
time T is finite, and we want the final state xT to be as close as possible to the distribution of
x∞. Therefore, we introduce τ such that given T, the distribution of xT follows N(µ(1 −e¯θT ) +
x0e¯θT , τ 2λ2(1 −e2¯θT )), and x∞follows N(µ, τ 2λ2), with τ > 1, we increase the possibility of a
sample ˜xT from xT to become closer to the distribution of x∞, and thus serves as a plausible initial
state for the reverse process. We can control how much to close the distributions, either by increasing
the stiffness τ at the cost of potentially destabilizing the reverse process, or by decreasing τ to further
smooth the density functions of both distributions at the cost of more reverse iterations.

By connecting the inverse problem with the analysis above, we clarify the discrepancy in the stationary
modeling process from measure-preserving dynamics and thereby improve the generalization of
diffusion processes and the accuracy of the reverse process. This is particularly important for
accommodating the diversity of degradation states and to ensure accurate sampling.

Forward Process. We describe the forward process as: dxt = θt(µ−xt)dt+τσtdWt, parameterized
by τ to calibrate the SDE modeling, µ is the state mean. The parameters θt and σt, both being time-
dependent and strictly positive, correspond to the rate of mean reversion and the stochastic volatility,
respectively. The selection of θt and σt offers flexibility in Tab. 2, c.p., Sect. 3.

Considering the trade-off between complexity and effectiveness, Cos has been chosen for both θt
and σt. This aims at capturing complex temporal dynamics in a computationally tractable manner,
thereby optimizing the balance between the performance and calculation convenience.

6


---Page Break---
Table 2: µt(xt, t), vt(xt, t) solutions with various θt, σt.

µt(xt, t)
vt(xt, t)

Lin
eθ x2

2 x0 +

1 −eθ x2

2

y
τ 2λ2 
1 −eθx2
y

Log
e
θ
k log(1+ekt)x0 +

1 −e
θ
k log(1+ekt)
y
τ 2λ2 
1 −e
2θ

k log(1+ekt)
y

Cos
e
−θ

t−sin(θt)

θ


x0 +

1 −e
−θ

t−sin(θt)

θ


y
τ 2λ2

1 −e
−2θ

t−sin(θt)

θ


y

Quad
eθ x3

3 x0 + (1 −eθ x3

3 )y
τ 2λ2(1 −e2θ x3

3 )y

In the forward process, the mean µt approaches the low-quality image with E (xt) = µ, while
the variance tends toward the stationary variance var (xt) = τ 2σ2

2θ . Essentially, the forward SDE
transitions the high-quality image to a low-quality counterpart infused with Gaussian noise. The
discretized SDE for the forward process is xti = xti−1 + θti−1(µ −xti−1)∆t + τσti−1∆Wi. We
employ a transition strategy utilizing a varied stationary variance. Additionally, we execute an
unconditional update, which operates without the need for matching in the reverse process. These not
only allow image corruption but also provides effective adaptability for improvements.

Reverse Process. The reverse process aims at reconstructing the original image by gradually denois-
ing a low quality image. It utilizes the score of the marginal distribution, denoted as ∇x log ˆpt(x),
and is governed by:

dxt =

θt(µ −xt) −τ 2σ2
t ∇x log ˆpt(x)

dt + τσtdc
Wt.
(6)

The reverse-time D3GM process of Eq. 6 can be found in Appx. D. This closely mirrors the forward
process and incorporates an additional drift term proportional to the score of the marginal distribution.
The ground truth score for this process, necessary for training our generative model, is:

∇x log ˆpt(x | x0) = −xt −µt(x)

vt
,
(7)

where µt(x) represents the random attractor of the process at time t, and vt is the variance. Our
training objective is defined as the minimization of the expected discrepancy between the predicted
and true scores over the data distribution:

θ∗= arg min
θ
Et,(x0,y),z,xt
h
w ∥Sθ(xt, y, t) −∇xt log p0t(xt | x0, y)∥2
2
i
,
(8)

where w = −1/τ 2 is a time-dependent weighting function, and Sθ denotes the score network
parameterized by θ which approximates the score of the marginal distribution. The optimization is
conducted over the network parameters θ, under the expectation with respect to the time variable t,
the initial image x0, noisy image xt, and data y.

5
Experiments

Experimental Settings: We evaluate D3GM on various challenging restoration and reconstruction
problems. We initially analyze our method by examining its performance with closely related
diffusion formulation variants. Subsequently, we benchmark D3GM against the state-of-the-art
techniques in these domains. For comprehensive evaluation across all experiments, we report the
PSNR [25] and SSIM [58] for pixel- and structural-level alignment, LPIPS [72] for measuring
perceptual variance. An in-depth description of our implementation is provided in Appx. G.

5.1
Stability: Illustrations of the Measure-preserving Dynamics within Diffusion Models

SGM, and Transitionary SGMs vs. D3GM:

We perform qualitative and quantitative analyses using variants of closely related formulations for
Prop. 1 and 2 and evaluate across (A) SGMs and (B) transitionary SGMs. (A) uses a common
score-based SDE, (B) uses a Coefficient Decoupled SDE (e.g., variance exploding SDE with the drift
term µ) according to Prop. 1 and OU SDE, alongside our D3GM.

7


---Page Break---
Table
3:
Quantitative
results
for
Rain100H
and
Rain100L.(best in bold and second best underlined)

Rain100H
Rain100L

Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓

JORDER [64]
26.25
0.835
0.197
36.61
0.974
0.028
IRCNN [37]
29.12
0.882
0.153
33.17
0.958
0.068
PreNet [49]
29.46
0.899
0.128
37.48
0.979
0.020
MPRNet [68]
30.41
0.891
0.158
36.40
0.965
0.077
MAXIM [56]
30.81
0.903
0.133
38.06
0.977
0.048

VPB (CD) [73]
30.89
0.885
0.051
38.12
0.968
0.023
IR-SDE (OU) [37]
31.65
0.904
0.047
38.30
0.981
0.014
D3GM
32.41
0.912
0.040
38.40
0.982
0.013

Table 4: Quantitative results for O-
HAZE and Dense-Haze.

O-HAZE
Dense-Haze

Methods
PSNR↑
SSIM↑
PSNR↑
SSIM↑

DCP [23]
16.78
0.653
12.72
0.442
DehazeNet [6]
17.57
0.770
13.84
0.430
GFN [50]
18.16
0.671
-
-
GDN [36]
18.92
0.672
14.96
0.536
MSBDN [18]
24.36
0.749
15.13
0.555
FFA-Net [46]
22.12
0.770
15.70
0.549
AECR-Net [60]
-
-
15.80
0.466
SGID-PFF [4]
20.96
0.741
12.49
0.517
Restormer [67]
23.58
0.768
15.78
0.548
Dehamer [22]
25.11
0.777
16.62
0.560
MB-TF [47]
25.31
0.782
16.44
0.566
D3GM
26.23
0.786
15.85
0.551

MSBDN
LQ 
FFA-Net
SGID-PFF
Dehamer
MB-TF
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)
GT

Dense-Haze
O-HAZE

(a) O-HAZE and Dense-Haze.

𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)
GT
IR-SDE
MPRNet
PReNet
JORDER
LQ

(b) Rain100H and Rain100L.

Figure 3: Qualitative results for (a) deraining and (b) dehazing.

OU SDE
Coef. Dec. SDE
SGM
𝑫𝟑𝑮𝑴

t = 0.00
t = 0.25
t = 0.50
t = 0.75
t = 1.00

Figure 2: Sampling trajectories of SGM, transi-
tionary SGMs: Coef. Dec., OU SDE, and D3GM.

Following Tab. 1, VPB [73] can be regarded as
a Coefficient Decoupled SDE, and IR-SDE [37]
as an OU SDE. Our results in Fig. 2 illustrate
that D3GM converges stably towards the ex-
pected distribution, unlike other methods which
exhibit instability or deviation. This highlights
the reliance of other techniques, e.g., score-
based SDEs, on retrospective measurement con-
sistency corrections.

Simulated Deraining: We evaluated D3GM to-
gether with the state-of-the-art deraining strate-
gies: (1) OU SDE method IR-SDE [37], Co-
efficient Decoupled (CD) VPB [73] and other
CNNs [64, 49, 68, 56].
We use two of
the most renowned synthetic raining datasets:
Rain100H [65] and Rain100L [65]. Rain100H
contains 1800 pairs of images with and without
rain, along with 100 test pairs. As for Rain100L,
it consists of 200 pairs for training and 100 pairs for testing. We present results based on the PSNR,
SSIM, and LPIPS metrics.

Quantitative results from the two raining datasets are presented in Tab. 3. Based on both distortion
and perceptual metrics, D3GM is capable of generating the most realistic and high fidelity results as
shown in Fig. 3b.

5.2
Generalizability: D3GM for real-world data

Case Study 1: Dehazing. We utilize the real-world datasets O-HAZE [2], Dense-Haze [1], which
contain 45 and 55 paired images, respectively. We use the last 5 images of each dataset as the testing
set and the rest as the training set following the common split of other methods. Results are shown
in Tab. 4 and Fig. 3a. Our work improves results on O-HAZE both quantitatively and qualitatively.
Smaller improvements are observed on Dense-Haze. This can be attributed to the severe signal
corruption of the Dense-Haze data. A combination with tailored task-specific, Transformer-based

8


---Page Break---
methods [47, 22] might lead to further performance gains for such data. Such extensions are beyond
the focus of this paper. Qualitatively D3GM achieves excellent visual results (Fig. 1 and Fig. 3a).

Case Study 2: MRI Reconstruction. MRI data is represented in the complex-valued frequency
domain, which is distinctly different from the natural image domain. We utilized the fastMRI
dataset [69], containing single-channel, complex-valued MRI samples. Implementation details can be
found in Appx. G. For a robust comparison, we benchmarked against a diverse set of deep learning-
based state-of-the-art reconstruction methods. Although our method does not have a task-specific
design, we still get comparable performance (more details and results are provided in Appx. H).
Fig. 1 illustrates our reconstruction results from masked k-space data for 8x and 16x acceleration, i.e.,
under-sampling for faster data acquisition.

Table 5: Quantitative results for fastMRI dataset with
acceleration rates x8 and x16.

Undersampling factor 8x
Undersampling factor 16x

Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓

ZeroFilling
22.74
0.678
0.504
20.04
0.624
0.580
D5C5 [51]
25.99
0.719
0.291
23.35
0.667
0.412
DAGAN [62]
25.19
0.709
0.262
23.87
0.673
0.317
SwinMR [27]
26.98
0.730
0.254
24.85
0.673
0.327
DiffuseRecon [44]
27.40
0.738
0.286
25.75
0.688
0.362
CDiffMR [26]
27.26
0.744
0.236
25.77
0.707
0.293
D3GM
27.92
0.740
0.175
25.26
0.701
0.153

Table 6: Quantitative results for IXI MRI
SR on unseen datasets.

X4

HH
Guys
IOP

Methods
PSNR↑
SSIM↑
PSNR↑
SSIM↑

IXI T2w

EDSR [33]
23.03
0.700
25.10
0.727
SFM [19]
23.28
0.711
25.18
0.731
PDM [70]
22.89
0.709
27.93
0.851
ACT [48]
22.80
0.707
26.38
0.826
CST [14]
23.70
0.714
28.55
0.837
D3GM
25.13
0.799
28.60
0.863

Case Study 3: MRI Super-resolution (SR). IXI3 dataset is the largest benchmark considered in our
MRI SR evaluation. Clinical MRI T2-weighted (T2w) scans are collected from three hospitals with
different imaging protocols: HH, Guys, and IOP. For investigating our cross-domain generalization
and robustness, a challenging task for both MRI SR and natural image restoration, we trained on HH
data with k-space truncation, and tested on Guys and IOP with kernel degradation with an up-scaling
factor of X4. More details can be found in Appx. G.

The methods are tested under unseen data conditions, including different acquisition parameters, MRI
scanners (different vendors and field-strengths) and unseen degradations. With D3GM we are able to
demonstrate varying degrees of improvement, as well as generalizability to the discrepancy within
the training domain and across the test domain as shown in Tab. 6. Qualitative results are shown in
Fig. 11 and further results across domains in Appx. H.

6
Discussion and limitations

Other works, like VPB (CD) [73], I2SB [34] are based on diffusion bridges assuming that clean and
degraded images are already close. Thus, the tractability of the reverse process heavily relies on
the validity of the assumed Dirac delta distribution. IR-SDE [37] employs the mean-reverting SDE
theorem based on running the reverse SDE with instability. Since unstable errors accumulate in each
step, this model will eventually become unable to learn the transformation, e.g., degradation. DPS [9]
and CDDB [10] assume that the degradation process is known, or linear operations are directly used
to simulate the degradation process, which limits the generalizability of the method. In contrast,
D3GM is built on the theorem of Measure-preserving RDS, which bridges clean and degraded image
distributions while taking both degradation and measurements into account. Moreover, D3GM can be
extended to a two-sided solution operator (tractability) with a flow map according to Prop. 1.

Limitations. Even though our results are better than others when the degradation process is very
severly corrupted (e.g., real dehazing), the overall quality of the restored image is still limited, which
is consistent with Prop. 2. This might be alleviated via guiding the sampling process with priors
and enhanced µ, such as posterior sampling or degradation maps on the data manifold, but such
approaches are still limited as shown in Tab. 7 and Appx I.

Computational Complexity vs. Performance. Tab. 7 highlights that prior work is often tailor-made
for a specific subset of tasks and thus also generalisation-limited for challenging environments in
practice. D3GM’s focus on a generic robust solutions from an RDS perspective can mitigate this,
while maintaining en-par performance with task-specific approaches in Tab. 8.

3http://brain-development.org/ixi-dataset/

9


---Page Break---
Table 7: Comparison to tailor-made approaches.

Real Dehazing
Diffu. for Inverse Problems.
Transitionary SGMs.

Methods
DPS [9]
CDDB [10]
I2SB [34]
IR-SDE [37]
D3GM

PSNR↑
18.63
21.55
21.51
24.52
26.23
SSIM↑
0.448
0.591
0.583
0.691
0.786

Table 8: D3GM vs. recent deraining works.

Deraining
rain200H
Model Complexity

Methods
PSNR↑
SSIM↑
Param.
FLOPs

DRSformer [7]
32.17
0.933
33.7M
242.9G
D3GM
32.21
0.925
36.5M
104.7G

7
Conclusion

The proposed D3GM framework enhances the stability and generalizability of SDE-based diffusion
methods for challenging inverse problems. Our approach, grounded in measure-preserving dynamics
of random dynamical systems, ensures broad applicability and relevance. We demonstrate D3GM’s
effectiveness across various benchmarks, including challenging tasks like MRI reconstruction.

Acknowledgements: This work was supported by the JADS programme and UK Research and
Innovation [UKRI Centre for Doctoral Training in AI for Healthcare grant number EP/S023283/1].
HPC resources were provided by the Erlangen National High Performance Computing Center
(NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR
project b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities.
NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.
Support was also received by the ERC - projects MIA-NORMAL 101083647 as well as DFG
513220538, 512819079.

References

[1] C. O. Ancuti, C. Ancuti, M. Sbert, and R. Timofte. Dense-haze: A benchmark for image dehazing with
dense-haze and haze-free images. In ICIP, pages 1014–1018. IEEE, 2019.

[2] C. O. Ancuti, C. Ancuti, R. Timofte, and C. De Vleeschouwer. O-haze: a dehazing benchmark with real
hazy and haze-free outdoor images. In CVPR workshops, pages 754–762, 2018.

[3] B. D. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications,
12(3):313–326, 1982.

[4] H. Bai, J. Pan, X. Xiang, and J. Tang. Self-guided image dehazing using progressive feature fusion. TIP,
31:1217–1229, 2022.

[5] S. Bell-Kligler, A. Shocher, and M. Irani. Blind super-resolution kernel estimation using an internal-gan.
NeurIPS, 32, 2019.

[6] B. Cai, X. Xu, K. Jia, C. Qing, and D. Tao. Dehazenet: An end-to-end system for single image haze
removal. IEEE TIP, 25(11):5187–5198, 2016.

[7] X. Chen, H. Li, M. Li, and J. Pan. Learning a sparse transformer network for effective image deraining. In
CVPR, 2023.

[8] X. Chu, L. Chen, and W. Yu. Nafssr: Stereo image super-resolution using nafnet. In CVPR, pages
1239–1248, 2022.

[9] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye. Diffusion posterior sampling for general
noisy inverse problems. ICLR, 2023.

[10] H. Chung, J. Kim, and J. C. Ye. Direct diffusion bridge using data consistency for inverse problems.
NeurIPS, 2023.

[11] H. Chung, D. Ryu, M. T. McCann, M. L. Klasky, and J. C. Ye. Solving 3d inverse problems using
pre-trained 2d diffusion models. In CVPR, pages 22542–22551, 2023.

[12] H. Chung, B. Sim, and J. C. Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for
inverse problems through stochastic contraction. In CVPR, pages 12413–12422, 2022.

[13] H. Crauel, A. Debussche, and F. Flandoli. Random attractors. Journal of Dynamics and Differential
Equations, 9:307–341, 1997.

[14] M. Z. Darestani, J. Liu, and R. Heckel. Test-time training can close the natural distribution shift performance
gap in deep learning based compressed sensing. In ICML, pages 4754–4776. PMLR, 2022.

[15] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion schrodinger bridge with applications to
score-based generative modeling. NeurIPS, 34:17695–17709, 2021.

10


---Page Break---
[16] K. Dehnad. Density estimation for statistics and data analysis, 1987.

[17] M. Delbracio and P. Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image
restoration. TMLR, 2023.

[18] H. Dong, J. Pan, L. Xiang, Z. Hu, X. Zhang, F. Wang, and M.-H. Yang. Multi-scale boosted dehazing
network with dense feature fusion. In CVPR, pages 2157–2167, 2020.

[19] M. El Helou, R. Zhou, and S. Süsstrunk. Stochastic frequency masking to improve super-resolution and
denoising networks. In ECCV, pages 749–766. Springer, 2020.

[20] G. Franzese, S. Rossi, L. Yang, A. Finamore, D. Rossi, M. Filippone, and P. Michiardi. How much is
enough? a study on diffusion times in score-based generative models. Entropy, 25(4):633, 2023.

[21] U. Grenander and M. I. Miller. Representations of knowledge in complex systems. Journal of the Royal
Statistical Society: Series B (Methodological), 56(4):549–581, 1994.

[22] C.-L. Guo, Q. Yan, S. Anwar, R. Cong, W. Ren, and C. Li. Image dehazing transformer with transmission-
aware 3d position embedding. In CVPR, pages 5812–5820, 2022.

[23] K. He, J. Sun, and X. Tang. Single image haze removal using dark channel prior. IEEE TPAMI, 33(12):2341–
2353, 2010.

[24] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020.

[25] A. Hore and D. Ziou. Image quality metrics: Psnr vs. ssim. In ICPR, pages 2366–2369. IEEE, 2010.

[26] J. Huang, A. I. Aviles-Rivero, C.-B. Schönlieb, and G. Yang. Cdiffmr: Can we replace the gaussian noise
with k-space undersampling for fast mri? In MICCAI, pages 3–12. Springer, 2023.

[27] J. Huang, Y. Fang, Y. Wu, H. Wu, Z. Gao, Y. Li, J. Del Ser, J. Xia, and G. Yang. Swin transformer for fast
mri. Neurocomputing, 493:281–304, 2022.

[28] A. Hyvärinen and P. Dayan. Estimation of non-normalized statistical models by score matching. Journal
of Machine Learning Research, 6(4), 2005.

[29] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas. Gotta go fast when
generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021.

[30] A. Jolicoeur-Martineau, R. Piché-Taillefer, R. T. d. Combes, and I. Mitliagkas. Adversarial score matching
and improved sampling for image generation. arXiv preprint arXiv:2009.05475, 2020.

[31] T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative
models. arXiv preprint arXiv:2206.00364, 2022.

[32] B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. Annals of
statistics, pages 1302–1338, 2000.

[33] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee. Enhanced deep residual networks for single image
super-resolution. In CVPR workshops, pages 136–144, 2017.

[34] G.-H. Liu, A. Vahdat, D.-A. Huang, E. A. Theodorou, W. Nie, and A. Anandkumar. I2sb: Image-to-image
schrbackslash odinger bridge. ICML, 2023.

[35] H.-T. D. Liu, F. Williams, A. Jacobson, S. Fidler, and O. Litany. Learning smooth neural functions via
lipschitz regularization. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–13, 2022.

[36] X. Liu, Y. Ma, Z. Shi, and J. Chen. Griddehazenet: Attention-based multi-scale network for image dehazing.
In ICCV, pages 7314–7323, 2019.

[37] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjölund, and T. B. Schön. Image restoration with mean-reverting
stochastic differential equations. ICML, 2023.

[38] K. Mei, A. Jiang, J. Li, J. Ye, and M. Wang. An effective single-image super-resolution model using
squeeze-and-excitation networks. In ICONIP, pages 542–553. Springer, 2018.

[39] C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon. Sdedit: Guided image synthesis and
editing with stochastic differential equations. ICLR, 2022.

[40] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial
networks. ICLR, 2018.

11


---Page Break---
[41] J. C. Nguyen, A. A. De Smet, B. K. Graf, and H. G. Rosas. Mr imaging–based diagnosis and classification
of meniscal tears. Radiographics, 34(4):981–999, 2014.

[42] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In ICML, pages
8162–8171. PMLR, 2021.

[43] G. Parisi. Correlation functions and computer simulations. Nuclear Physics B, 180(3):378–384, 1981.

[44] C. Peng, P. Guo, S. K. Zhou, V. M. Patel, and R. Chellappa. Towards performant and reliable undersampled
mr reconstruction via diffusion model sampling. In MICCAI, pages 623–633. Springer, 2022.

[45] H. Poincaré. Sur le problème des trois corps et les équations de la dynamique. Acta mathematica,
13(1):A3–A270, 1890.

[46] X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia. Ffa-net: Feature fusion attention network for single image
dehazing. In AAAI, 2020.

[47] Y. Qiu, K. Zhang, C. Wang, W. Luo, H. Li, and Z. Jin. Mb-taylorformer: Multi-branch efficient transformer
expanded by taylor formula for image dehazing. In CVPR, pages 12802–12813, 2023.

[48] M. S. Rad, T. Yu, B. Bozorgtabar, and J.-P. Thiran. Test-time adaptation for super-resolution: You only
need to overfit on a few more images. In ICCV, pages 1845–1854, 2021.

[49] D. Ren, W. Zuo, Q. Hu, P. Zhu, and D. Meng. Progressive image deraining networks: A better and simpler
baseline. In CVPR, pages 3937–3946, 2019.

[50] W. Ren, L. Ma, J. Zhang, J. Pan, X. Cao, W. Liu, and M.-H. Yang. Gated fusion network for single image
dehazing. In CVPR, pages 3253–3261, 2018.

[51] J. Schlemper, J. Caballero, J. V. Hajnal, A. Price, and D. Rueckert. A deep cascade of convolutional neural
networks for mr image reconstruction. In IPMI, pages 647–658. Springer, 2017.

[52] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using
nonequilibrium thermodynamics. In ICML, pages 2256–2265. PMLR, 2015.

[53] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In ICLR.

[54] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 32,
2019.

[55] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative
modeling through stochastic differential equations. In ICLR, 2020.

[56] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li. Maxim: Multi-axis mlp for image
processing. In CVPR, pages 5769–5780, 2022.

[57] P. Vincent. A connection between score matching and denoising autoencoders. Neural computation,
23(7):1661–1674, 2011.

[58] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility
to structural similarity. IEEE TIP, 13(4):600–612, 2004.

[59] S. Welker, J. Richter, and T. Gerkmann. Speech enhancement with score-based generative models in the
complex stft domain. arXiv preprint arXiv:2203.17004, 2022.

[60] H. Wu, Y. Qu, S. Lin, J. Zhou, R. Qiao, Z. Zhang, Y. Xie, and L. Ma. Contrastive learning for compact
single image dehazing. In CVPR, pages 10551–10560, 2021.

[61] Z. Xiao, K. Kreis, and A. Vahdat. Tackling the generative learning trilemma with denoising diffusion gans.
In ICLR, 2021.

[62] G. Yang, S. Yu, H. Dong, G. Slabaugh, P. L. Dragotti, X. Ye, F. Liu, S. Arridge, J. Keegan, Y. Guo, et al.
Dagan: Deep de-aliasing generative adversarial networks for fast compressed sensing mri reconstruction.
TMI, 37(6):1310–1321, 2017.

[63] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, Y. Shao, W. Zhang, B. Cui, and M.-H. Yang.
Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 2022.

[64] W. Yang, R. T. Tan, J. Feng, Z. Guo, S. Yan, and J. Liu. Joint rain detection and removal from a single
image with contextualized deep networks. IEEE TPAMI, 42(6):1377–1393, 2019.

12


---Page Break---
[65] W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan. Deep joint rain detection and removal from a
single image. In CVPR, pages 1357–1366, 2017.

[66] Z. Yue, J. Wang, and C. C. Loy. Resshift: Efficient diffusion model for image super-resolution by residual
shifting. NeurIPS, 2023.

[67] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang. Restormer: Efficient transformer
for high-resolution image restoration. In CVPR, pages 5728–5739, 2022.

[68] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao. Multi-stage progressive
image restoration. In CVPR, pages 14821–14831, 2021.

[69] J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. John-
son, M. Bruno, et al. fastmri: An open dataset and benchmarks for accelerated mri. arXiv preprint
arXiv:1811.08839, 2018.

[70] K. Zhang, J. Liang, L. Van Gool, and R. Timofte. Designing a practical degradation model for deep blind
image super-resolution. In CVPR, pages 4791–4800, 2021.

[71] Q. Zhang and Y. Chen. Fast sampling of diffusion models with exponential integrator. ICLR, 2022.

[72] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, pages 586–595, 2018.

[73] L. Zhou, A. Lou, S. Khanna, and S. Ermon.
Denoising diffusion bridge models.
arXiv preprint
arXiv:2309.16948, 2023.

13


---Page Break---
Appendix

A
Broader Impacts

Our proposed method offers significant advancements in the restoration of degraded images with
minimal risk of hallucinations due to stability guarantees and with applications including MRI
reconstruction and super-resolution. In the clinical domain, the adoption of our method for MRI
reconstruction must adhere to stringent regulatory and approval processes. The results generated by
our model should serve as an auxiliary tool to assist healthcare professionals in their diagnostic and
treatment decisions, rather than as a standalone diagnostic tool.

B
Intuition of Measure-Preserving Dynamics in SDE:

Consider the analogy of a stretched rubber band, which naturally seeks to return to its original but
does so with a lot of oscillations when released. This elastic behavior parallels the dynamics of the
OU process, where deviations from a mean state are counteracted by a restorative force, guiding the
system back towards equilibrium (i.e., final state), with random perturbations.

Our process models the noise as a stochastic component that fluctuates around a stationary process
and improve the OU process with RDS. Measure-preserving dynamics ensure that while the image
undergoes transformations during the denoising process, the overall statistical properties remain
consistent (i.e., invariant image features), which cannot be satisfied by vanilla OU processes or
previous approaches (Tab. 1).

C
Mathematical Preliminaries

Consider a probability space (Ω, F, P) accompanied by a standard Brownian motion Wt. A stochastic
process xt over the interval 0 ≤t ≤T can be formulated by the following stochastic differential
equation (SDE):

dxt = b(t, xt)dt + σ(t, xt)dWt
(9)

Definition 1 Filtration A collection of sigma-fields,

F := {Ft, 0 ≤t ≤T},

is termed a filtration if:

1. Ft ⊂F is a sub-σ-field for every t ∈[0, T];

2. If 0 ≤t1 < t2 ≤T, then Ft1 ⊂Ft2.

Here, Ft represents the information set at time t.

Definition 2 Strong Solution A process x, which is F-progressively measurable, is considered a
strong solution to the SDE given by Eq. 1 if:
Z T

0
(|b(t, xt)|2 + |σ(t, xt)|2)dt < ∞

almost surely. This is captured by:

xt = x0 +
Z t

0
b(s, xs)ds +
Z t

0
σ(s, xs)dWs, ∀t ∈[0, T]
(10)

Definition 3 Lipschitz Continuity For an N-dimensional stochastic process xt over t ∈[0, ∞),
adapted to the filtration F, function w(xt) exhibits Lipschitz continuity in x if:

1. w(xt) is F-measurable with the requisite dimensions.

14


---Page Break---
2. There exists a non-negative constant K such that, for every xt and xs,

|w(xt) −w(xs)| ≤K|xt −xs|.

In numerous diffusion models, Lipschitz continuity is inherent, ensuring the existence and uniqueness
of the solution to the stochastic process. However, for clarity in network design, the emphasis on
Lipschitz continuity ensures the foundation of neural networks remains consistent.

Definition 4 Random Dynamical System A random dynamical system(RDS) consists of a base flow,
the "noise", and a cocycle dynamical system on the "physical" phase space, we first discuss one
fundamental element of our RDS, the base flow.
Let (Ω, F, P) be a probability space, the noise space. Define the base flow ϑ : R × Ω→Ωas
follows: for each "time" s ∈R, let ϑs : Ω→Ωbe a measure-preserving measurable function:
P(E) = P
 
ϑ−1
s (E)

for all E ∈F and s ∈R
Suppose also that
1. ϑ0 = idΩ: Ω→Ω, the identity function on Ω;
2. for all s, t ∈R, ϑs ◦ϑt = ϑs+t.
That is, ϑs, s ∈R, forms a group of measure-preserving transformation of the noise (Ω, F, P)
Now we are ready to define the random dynamical system(RDS).
let (X, d) be a complete separable metric space, the phase space. Let φ : R × Ω× X →X be a
(B(R)⊗F ⊗B(X), B(X)) measurable function such that 1. for all ω ∈Ω, φ(0, ω) = idX : X →X,
the identity function on X; 2. for (almost) all ω ∈Ω, (t, x) 7→φ(t, ω, x) is continuous; 3. φ satisfies
the (crude) cocycle property: for almost all ω ∈Ω,

φ (t, ϑs(ω)) ◦φ(s, ω) = φ(t + s, ω)

In the case of random dynamical systems driven by a Wiener process W : R × Ω→X, the base flow
ϑs : Ω→Ωwould be given by

W (t, ϑs(ω)) = W(t + s, ω) −W(s, ω).

Theorem 1 Existence and Uniqueness If the initial condition x0 ∈L2 is a random variable that’s
independent of W and both µ(0, x0) and σ(0, x0) ∈H2, then, provided there exists a constant
K > 0 that satisfies:

|b(t, x) −b(t, y)| + |σ(t, x) −σ(t, y)|
≤K|x −y|,
∀t ∈[0, T], x, y ∈Rn
(11)

(an attribute also recognized as Lipschitz continuity), a unique strong solution to Eq. 1 exists in H2
for every T > 0. Additionally:

E

sup
t≤T
|xt|2

≤C(1 + E|x0|2)eCT
(12)

holds true, where the constant C depends on both T and K.

D
Preliminaries and Proof for Proposition 1

D.1
Proof for reverse-time D3GM process:

There is a one-to-one and onto correspondence between the stochastic differential equation and the
Kolmogorov equation for p (xt, t | xs, s) , t ⩾s, which describes the evolution of the underlying
probability distribution. Consequently, there should be a one-to-one and onto correspondence between
a reverse-time equation for ˜xt and a Kolmogorov equation for p(xt, t|xs, s), s ⩾t

dxt = θt(µ −xt)dt + τσdWt

We have the corresponding Kolmogorov backward equation given by

−∂p (xs, s | xt, t)

∂t
=θt(µ −xt) · ∂p (xs, s | xt, t)

∂xt

+ 1

2τ 2σ2
t · ∂2p (xs, s | xt, t)

∂x2
t
,
(13)

15


---Page Break---
The unconditioned Kolmogorov forward equation is given by

−∂p (xs, s | xt, t)

∂t
=∂(θt(µ −xt) · p (xs, s | xt, t))

∂xt

−1

2 · ∂2  
τ 2σ2
t · p (xs, s | xt, t)


∂x2
t
,
(14)

See [3] for more details on Kolmogorov equations.
Bayes rule gives
p (xt, t, xs, s) = p (xs, s | xt, t) p (xt, t)

We plug this result into 13, which gives us the Kolmogorov equation

−∂

∂tp (xt, t, xs, s) = ∂

∂xt

 ¯f (xt, t) p (xt, t, xs, s)


+ 1

2
∂2 
p (xt, t, xs, s) · τ 2σ2
t


∂x2
t

(15)

and the expression for ¯f is given by

¯f (xt, t) = θt(µ −xt) −
1
p (xt, t)
∂
∂xt


p (xt, t) τ 2σ2
t


= θt(µ −xt) −τ 2σ2
t log ∂

∂xt
[p(xt, t)]
(16)

Therefore, we have that the reverse process corresponds to the Kolmogorov equation 16 is given by

dxt = θt(µ −xt) −τ 2σ2
t log ∇xpt(xt) + τ 2σ2
t d ¯Wt

Definition 5 Different from deterministic dynamical systems, random dynamical systems usually
consider a pullback attractor rather than a forward attractor due to the non-autonomousness
introduced by the random noise. The pullback attractor (or random global attractor) A(ω) for the
RDS φ we defined in 1 is a P-almost surely unique random set such that:
1. A(ω) is a random compact set: A(ω) ⊆X is almost surely compact and ω 7→d(x, A(ω)) is a
(F, B(X))-measurable function for every x ∈X
2. A(ω) is invariant: for all φ(t, ω)(A(ω)) = A (ϑtω) almost surely;
3.
A(ω)
is
attractive:
for
any
deterministic
bounded
set
B
⊆
X,
limt→+∞d (φ (t, ϑ−tω) (B), A(ω)) = 0 almost surely.
B(X) denotes the Borel σ-algebra generated by the space X where the RDS is defined.

Definition 6 Poincare Recurrence Theorem
Let
(X, Σ, µ)

be a finite measure space and let
f : X →X

be a measure-preserving transformation. then we have that for any E ∈Σ, the set of those points x
of E for which there exists N ∈N such that f n(x) /∈E for all n > N has zero measure.
In other words, almost every point of E returns to E. In fact, almost every point returns infinitely
often; i.e. µ ({x ∈E : there exists N such that f n(x) /∈E for all n > N}) = 0.

D.2
Analysis of Proposition 1

Proposition 1 After extending the solution of the OU process to RDS, the measure-preserving
flow map of the solution should meet the property φ(t, s; ω)x = φ (t −s, 0; ϑsω) x. However, OU
processes with time-varying coefficients are usually not satisfied for this property(can be referred to
as time-homogeneity) and thus the stability of the system breaks.
The forward process in SDE notation

dxt = θt(µ −xt)dt + σtdWt

16


---Page Break---
and the solution to the above SDE is given by

xt = µ + (xs −µ)e−¯θs:t +
Z t

s
σze−¯θz:tdWz

where ¯θs:t =
R t
s θzdz.

This solution above can be represented by a continuous random dynamical system (RDS) φ defined
on a complete separable metric space (X, d), where the noise is chosen from a probability space
(Ω, F, P). More details can be found in [13].
More generally, we can extend the RDS to two-sided, infinite time, define a flow map or (solution
operator) φ : R×Ω×Rd →Rd by φ (t, s, |ω, x0) := x (t, s|ω, x0) with ω ∈Ω, −∞< s ⩽t < ∞.
The base flow driven by Brownian motion can be explicitly written as W (t, ϑs(ω)) = W(t+s, ω)−
W(s, ω).
Now suppose that:
1. The flow map ϑt, t ∈R is a measure-preserving transformations of (Ω, F, P), with the property
that for all s < t and x ∈X,

φ(t, s; ω)x = φ (t −s, 0; ϑsω) x,
P-a.s.
(17)

2. (i) φ(t, r; ω)φ(r, s; ω)x = φ(t, s; ω)x for all s ⩽r ⩽t and x ∈X;
(ii) φ(t, s; ω) is continuous in X, for all s ⩽t.
(iii) for all s < t and x ∈X, the mapping

ω 7→φ(t, s; ω)x

is measurable from (Ω, F) to (X, B(X)); and
(iv) for all t, x ∈X, and P-a.e. ω, the mapping s 7→φ(t, s; ω)x is right continuous at any point.
Where B(X) denotes the σ-algebra generated by X.
Under assumptions (i), (ii), (iii), (iv) and suppose that for P-a.e. ω there exists a compact attracting
set K(ω) at time 0, i.e., such that for all bounded sets B ⊂X,

d(φ(0, s; ω)B, K(ω)) →0
as
s →−∞

We can see that the attractor of this system is defined in the pullback sense, such that time is rewind
backward before iterating forward.
Moreover, the reverse process with any starting time t to s is defined as the RDS going backward in
time
φ(s, t|ϑtω, xt)

start from the time t realization and run backwards to s.

the above proposition can be extended to show that there exists a compact attracting set at any
−∞< t < ∞. and this convention has allowed us to characterize the attractor K(ω) = N(µ, λ2),
when the time becomes finite, for example from 0 to T, the random attractors can be abstractly
viewed as N

µ + (xs −µ)e−¯θ0:t, λ(1 −e−¯θ0:t)

.

Moreover, an important assumption in the 17 is usually not satisfied for OU processes with time-
varying coefficients, therefore, we impose that for every t, σ2
t
2θt = λ2, where λ is a constant, and
will be the asymptotic variance of the forward process. This convention has allowed us to reduce
the regularization on two variables σt, θt to just one variable to satisfy 17; and this convention
has allowed as to characterize the attractor K(ω) = N(µ, λ2), when the time becomes finite, for
example from 0 to T, the random attractors can be abstractly viewed as the Gaussian measure
N

µ + (xs −µ)e−¯θ0:t, λ(1 −e−¯θ0:t)

.

E
Proof for Proposition 2

Proposition 2 Given Eq. 3 and Eq. 4, and assume that the score function is bounded by C in L2 norm,
then the discrepancy between the reference and the retrieved data is, with probability at least (1 −δ):

17


---Page Break---
∥x0 −OU(x0, µ; T, θ)∥2
2
≥|
 
(x0 −µ)2 −σ2
T /2θT

e−2¯θT + σ2
T /2θT

−σ2
max

Cσ2
max + d + 2
p

−d · log δ −2 log δ

|,

(18)

where x0 ˆx0 are the quality reference and sampling data. For a noisy inverse problem scenario, the
retrieved data with any finite T always indicates difference depends on σt, µt, λ, T, ¯K, where ¯K is
the Lipschitz constant for the reverse process.

we have that the absolute value between the theoretical expectation and actual expectation after T
period is given by
∥µ −E(ˆxT )∥= ∥(x0 −µ)e−¯θT ∥> 0

Similarly, the difference between theoretical variance and T-period variance also has a strictly positive
difference. where ¯θt =
R t
0 θsds.
Therefore, with finite T, the final state of the forward process can only reach a ˆxT rather than the
theoretical stationary distribution, which we denote by x∞, we denote the retrieved image after
T-periods from theoretical stationary distribution by ˆx0, x0 by the ground truth HQ image, ˆxT the
true distribution after T iteration,

∥xQ −fOUn(xQ, µ; t0, )∥= ∥ˆxT −x∞−(ˆx0) −x∞∥2
2
≥∥∥ˆxT −x∞∥2
2 −∥ˆx0 −x∞∥2
2∥2
2
(19)

Inside the norm, the first term is bounded below, since both ˆxT and x∞both follow a normal
distribution and are independent of each other, the difference between those two random variables,
we denote by zT , that follows a normal distribution

N

µ + (x0 −µ)e−¯θt −µ, λ2(1 −e−2¯θt) + λ2

=N

(x0 −µ)e−¯θt, λ2(2 −e−2¯θt)

(20)

Therefore, we could rewrite the equality as:

∥xQ −fOU(xQ, µ; t0)∥

≥

∥zT ∥2
2 −


Z 0

T
−dσ2
t
dt ∇x log pt(x) dt + dwt



2

2



2

2

=


(x0 −µ)2e−2¯θt + λ2(2 −e−2¯θt) −Cσ4
max −



Z 0

T

r

dσ2
t
dt dwt



2

2



2

2

(21)

Since we require λ =
σ2
t
2θt , we can find a σmax such that σt < σmax for all t. The last term only
concerns the random noise, according to [32], we have that the last term is equivalent to the squared
L2 norm of a random variable from a Wiener process at time t = 0, with marginal distribution being
ϵ ∼N
 
0, σ2
T I

. The squared L2 norm of ϵ divided by σ2
T is a χ2-distribution with d-degrees of
freedom, we have the following one-sided tail bound, according to [32]

Pr

∥ϵ∥2
2/σ2 (t0) ≥d + 2
p

d · −log δ −2 log δ

≤exp(log δ) = δ

Therefore, with probability 1 −δ, the HQ image and retrieved image has a lower bound of,(d is the
number of dimension for x0),

|
 
(x0 −µ)2 −λ2
e−2¯θT + 2λ2

−σ2
max

Cσ2
max + d + 2
p

−d · log δ −2 log δ

|

A SDE modeling that has suffered complex degradation is considered ’corrupted’:

18


---Page Break---
Example 1. Based on the Prop. The conditional SDE diffusion is composed of three general
stages, forward, backward, and sampling. Both time-reversal processes have been declared
unstable under the degradation process µ according to the Prop. 2.

The following advantages would be offered by our measure-preserving dynamical system when
formulated as SDEs:

Intuition 1. A two-sided measure-preserving random dynamical(MP-RDS) system formulation
enables us to use the Poincare recurrence theorem, intuitively, with a two-sided MP-RDS φt,
the Poincare recurrence theorem ensures that the system φt starts from terminal condition xT ,
run backwards in time, will hit a region (x0 −ϵ, x0 + ϵ) for small ϵ in finite time, where x0 is
the high-quality image.

F
Temporal Distribution Discrepancy during Sampling

Theorem 2 Suppose that both the drift bt(x) and diffusion σt(x) term of a stochastic process xt is
Lipschitz continuous with some constant K, moreover, x ∈L2 (F, R) is a solution to the SDE

dxt = b(t, x)dt + σ(t, x)dWt

and initial condition b0, σ0 then we have that

E
h
|xT −x0|2i
≤CI2
0,
(22)

such that

I2
0 := E





 Z T

0
|b0| dt

!2

+
Z T

0
|σ0|2 dt





and C depends only on T, K, which is the running time and Lipschitz constant

Firstly, we have the following relationship:

xT ≤|x0| +
Z T

0
|b(t, x)| dt + sup
0≤t≤T



Z t

0
σ(s, x) dWs

 ,

|xT −x0| ≤
Z T

0
|b(t, x)| dt + sup
0≤t≤T



Z t

0
σ(s, x) dWs

 .

(23)

Squaring both sides, taking expectations, and applying the Burkholder-Davis-Gundy inequality, we
get:

E
h
|xT −x0|2i

≤CE





 Z T

0
|b(t, xt)| dt

!2

+ sup
0≤t≤T

Z t

0
σ(s, xs) dBs

2



≤CE





 Z T

0
[|b0| + |xt|] dt

!2

+
Z T

0
|σ(t, xt)|2 dt





≤CE





 Z T

0
|b0| dt

!2

+
Z T

0

h
|σ0|2 + |xt|2i
dt



.

(24)

Remark: It should be noted that the constant C, which depends on T and K, varies from line to line.

Next, we show that for any ε > 0, there exists a constant Cε > 0 such that:

19


---Page Break---
sup
0≤t≤T
E
h
|xt|2i
≤εE
h
|x∗
T |2i
+ CεI2
0.
(25)

Applying Itô’s formula, we get:

d |xt|2 =
h
2xtb(t, xt) + |σ(t, xt)|2i
dt + 2xtσ(t, xt) dBt.
(26)

Considering the martingale property of the third term, integrating, and taking expectation on both
sides, we have:

E
h
|xt|2i
= E

|x0|2 +
Z t

0

h
2xsb(s, xs) + |σ(s, xs)|2i
ds


≤E

|x0|2 +
Z t

0

h
C |xs|2 + 2 |xs| |b0| + C |σ0|2i
ds


≤C
Z t

0
E
h
|xs|2i
ds + 2E

"

xT

Z T

0
|b0| ds

#

+ CI2
0.

(27)

Using Gronwall’s inequality, we can prove (25) with the result above. Substituting (25) into (24)
completes the proof, showing that the distance between xT and x0 in the L2-sense is bounded by
C(K, T).

With theorem 1 in hand, since the OU process is Lipschitz continuous, then the reverse process for
the OU process is also Lipschitz continuous.
Now, suppose that the Lipschitz constant for the reverse process is given by ¯K. then we have that
in L2-norm, the distance between any final state xT ∈N(µ, λ) and the initial state(HQ image) is
bounded by a constant that only depends on time T and Lipschitz constant ¯K, and the initial condition
for the drift and diffusion term, where we denote as C( ¯K, T, µ, λ), which will be written as C( ¯K, T)
for short.
Now, since the time T is finite, where theoretically only when T →∞would xT converge to the
theoretical stationary distribution, thus, if we denote the sample from T time steps by ˆxT , and xT by
the sample from the theoretical stationary distribution, then we have E[xT −ˆxT )] > ϵ(T, K, µ, λ),
note that the K here is the Lipschitz constant for the forward process, and this distance strictly
decrease in T.
Suppose that in inference, when the ground truth x0 is unknown, the distance between ground truth
x0 and sample from the theoretical distribution xT is bounded from below by

||x0 −x∞|| > C( ¯K, T)I2
0 + ϵ
(28)

Then, we can see that the gap between xT and ˆxT increases such bound, which is

||x0 −ˆxT || ≥||x0 −x∞−(ˆxT −x∞)||

> C( ¯K, T)I2
0 + ϵ(T, K, µ, λ)

where T is the time step for the forward process, and ϵ is some strictly positive constant. Now
suppose that ˆxt is the solution to the reverse process, which runs for T periods in total, then we have
that for t ∈[0, T], denote x0, ˆx0 as the original HQ image and the final state of the reverse process,
respectively.

xQuality −OU
 
xQuality, µ; t0, θ
2

2 =

∥x0 −ˆx0∥= ∥x0 −ˆxT −(ˆx0 −ˆxT )∥≥
∥∥x0 −ˆxT ∥−∥(ˆx0 −xT )∥∥>

ϵ + C( ¯K, T)I2
0 −C( ¯K, T)I2
0 = ϵ

(29)

Therefore, the bias created in inference depends on the LQ image µ, stationary variance, Lipschitz
constant, and the time steps T.

20


---Page Break---
G
Stable in Probability

Definition 7 Stable in Probability
Given a probability space (Ω, F, P) and a standard Brownian motion Wt, a general form of SDE
for a stochastic process xt, 0 ≤t ≤T is given by

dxt = b(t, x)dt + σ(t, x)dWt
(30)

Such that the Lipschitz condition is satisfied, for both b(t, x), σ(t, x), A solution x(t, ω) ≡0 is said
to be stable in probability for t ≥0 if for any s ≥0 and ε > 0

lim
x0→0 P

sup
t>s |xs,x(t)| > ε

= 0.
(31)

It says that the sample path of the process issuing from a point x at time s will always remain within
any prescribed neighbourhood of the origin with probability tending to one as x →0. In practice,
this property ensures that the perturbation from the initial state caused by a stable process is bounded
for all t with probability one.
For example, the OU process 4 admits a unique unconditional stationary solution provided by theorem
19, however, in this example, without specifying the value that determines the stationary variance, i.e.,
σ, θ. If for large σ and a sample ˆx from the stationary distribution of the OU process, we have that

P(ˆx > µ ± σ

2θ) > 0
(32)

This means that a sample from the stationary distribution of the forward process could deviate largely
from µ, thus making the result no different from the traditional VE(variance exploding) diffusion
models that are defined as
dxt = σtdWt
because the variance could be set arbitrarily large if no restriction is specified. Therefore, how should
such a problem be approached most easily? The Lyapunov theorem for stability has provided an easy
way, without explicitly solving the SDEs, to ensure stability just from the coefficients.
Then we provide the main theorem that ensures the stability of SDE, first, we give the definition of
positive definite in the Lyapunov sense

Definition 8 Let K denote the family of all continuous nondecreasing functions µ : R+ →R+such
that µ(0) = 0 and µ(r) > 0 if r > 0. For h > 0, let Sh = {x ∈Rn : |x| < h}. A continuous
function V (x, t) defined on Sh × [t0, ∞) is said to be positive-definite (in the sense of Lyapunov) if
V (0, t) ≡0 and, for some µ ∈K,
V (x, t) ≥µ(|x|)

for all (x, t) ∈Sh × [t0, ∞)

Then we will use the convention of the Lyapunov quadratic function, such that

Definition 9 Lyapunov quadratic function V is given

V (xt) = xT
t Qxt,

where Q is a symmetric positive-definite matrix.

Theorem 3 The function LV

LV (xt) =xT
t Qb (t, xt) + b (t, xt)T Qxt+

σ (t, xt)T Qσ (t, xt) ,
(33)

is negative-definite in some neighbourhood of xt = 0 for t ≥t0, with respect to system 7. Then the
trivial solution of equation 7 is stochastically asymptotically stable.

21


---Page Break---
Since this theorem is important and the proof will be intuitive in explaining why such a condition
could ensure stability, the proof will be put here.
Proof:
First, we compute dV (x), which is the instantaneous growth of the Lyapunov quadratic function,
gives
dV (xt) =V (xt + dxt) −V (xt)

=
 
xT
t + dxT
t

Q (xt + dxt) −xT
t Qxt
=xT
t Qb (t, xt) dt + xT
t Qσ (t, xt) dBt+

b (t, xt)T dtQxt + σ (t, xt)T dBtQxt+

σ (t, xt)T Qσ (t, xt) dt
Then, take expectation, we can get

E {dV (xt)} = xT
t Qb (t, xt) dt + b (t, xt)T Qxtdt+

σ (t, xt)T Qσ (t, xt) dt
= LV (xt) dt

Then, if we assume that LV (x)
−LV (xt) ≥kV (xt)
such that k is a constant, then

d
dtE {V (xt)} ≤−kE {V (xt)} ,

E {V (xt)} ≤exp(−kt).
(34)

As can be seen from the proof, the operator LV as the function of the SDE xt is the expectation of the
dV (xt), and the negative semi-definiteness can be regarded as requiring dV (xt) to be a contraction.
This can be understood from 34 such that
d
dtE(V (xt))/E(V (xt)) < −k

for k > 0

H
Implementation details

Model Implementation: Our exploration into mitigating Temporal Distribution Discrepancy in
diffusion models employs two neural network architectures, each catering to different dataset com-
plexities.

We utilize the adopted UNet, a staple in DDPM [24] and DDIM[53] frameworks, chosen for its
widespread use and strong benchmarking capabilities. By solving the discrepancy through this
established structure, we achieve state-of-the-art results on synthetic data, showcasing the potential
of improving transanary SDE diffusion models in terms of Temporal Distribution Discrepancy and
stationary process. Additionally, the use of this prevalent architecture allows for comprehensive
analysis and discussion.

Real-world data with its inherent complexity, such as combined degradations, large resolutions, and
extensive interdependencies, requires an architecture beyond the conventional UNet. Our improved
model incorporates Squeeze-and-Excitation [38] and NAF [8], explicitly designed to capture intricate
feature interrelations. While these models do not seek to innovate the architectural paradigm, it
provides a solid baseline that provides better feature extraction ability than UNet performance in
demanding scenarios.

Additional details: The U-Net we adopted is similar to DDPM as described in [37, 11, 9], where
the improved model incorporates Squeeze-and-Excitation [38] to replace the Attention module
within NAF [8]. EDSR [33] is employed as the base model for TTA-based comparison methods
in MRI super-resolution. In different downstream tasks, we follow the common setting of the
latest compared methods: Deraining [37], Real Dehazing [47], MRI Reconstruction [26], and MRI
super-resolution [14] and [33]. Below are more specific details about MRI reconstruction and MRI

22


---Page Break---
Table 9: Datasets parameters and split setting. IOP details of the scan parameters are not available.

Datasets
Data setting
Collection
Sequence parameters

Source domain
Subjects
Slices
Hospitals
Scanner
Repetition time
Echo train length
Matrix size
Receiver coil

HH (IXI Brain)
184
60
Hammersmith
Philips 3T
5725
16
192 x 187
Single

Target domain

Guys (IXI Brain)
30
60
Guy’s Hospital
Philips 1.5T
8178
16
Unnormalized
Single
IOP (IXI Brain)
30
60
Institute of Psychiatry
GE 1.5T
Unknown
Single

Method
PSNR↑
SSIM↑
LPIPS↓
WD UNet
27.31
0.8322
0.227
SN UNet
28.63
0.8687
0.144
UNet
31.03
0.9001
0.058
Table 10: Evaluation of the Lipschitz continuity.

superresolution. For most of the experiments, the training patch-size is set to 128x128 with a batch
size of 16. We utilize the Adam optimizer with β1 = 0.9, β2 = 0.999, a learning rate of 10−4 with a
decay strategy. Our models are trained on three RTX 6000 GPUs for about four days, each with 40GB
of memory. Random seed is 42. All mathematical variants follow the cosine schedule as per [42].
The variance λ is set to 10 for the OU and stationary processes. In the coef. decoupled SDE, σ is
kept decoupled and does not vary with θ. We observed that the adaptability of τ to tasks is contingent
upon the task’s corruption for model stability and generalizability.

τ is a hyperparameter as one of the ways to introduce Measure-preserving Dynamics to shape more
stable SDE diffusion. It is informed by the deviation between the degraded image and the expected
high-quality image. For tasks with moderate intra-domain deviations, τ is set to 2, while for tasks
with substantial cross-domain and degradation discrepancies, a larger τ is used. This metric is based
on RDS settings of different degradations, rather than learned. A MLP can be used to learn τ as
adaptive metric, although it is not comparable with obvious improvements.

SGM, and Transitionary SGMs vs. D3GM: We perform qualitative 2 and quantitative 11 analyses
using variants of closely related formulations for Prop. 1 and 2 and evaluate across (A) SGMs and (B)
transitionary SGMs. (A) uses a common score-based SDE, (B) uses a Coefficient Decoupled SDE
(e.g., variance exploding SDE with the drift term µ) according to Prop. 1 and OU SDE, alongside our
D3GM.

MRI Reconstruction:
We applied 584 proton-density weighted knee MRI scans without fat
suppression. These were subsequently partitioned into a training set (420 scans), a validation
set (64 scans), and a testing set (100 scans). For each scan, we extracted 20 coronal 2D single-
channel complex-valued slices, predominantly from the central region with the uniform size of 320
x 320. Our test set differs from the 200 reported in [26], possibly as a result of the change in the
official versions of FastMRI. We subjected all experiments to Cartesian under-sampling masks with
undersampling factor 8x and 16x. In the undersampled MRI scans, the acquisition process entails
sampling a fractional subset of the Fourier-space (k-space), typically governed by a mask along the
dimension of undersampling rate. Undersampling inherently induces aliasing artifacts in the resultant
images. Considering the domain deviation, we did not supplement the extra test set into the final 100
samples. Otherwise, we stayed consistent with the details of the paper. For a robust comparison, we
benchmarked against a diverse set of deep learning-based state-of-the-art reconstruction methods,
including CNN-based approaches such as D5C5 [51] and DAGAN [62], the Transformer-based
SwinMR [27], diffusion model-inspired DiffuseRecon [44], and the CDiffMR [26]. Quantitative

Table 11: Exemplary deraining results of (A) SGM, (B) transitionary SGMs: Coefficent Decoupled
SDE, OU SDE, and D3GM.

Method
PSNR↑
SSIM↑
LPIPS↓

A
SGM
27.27
0.840
0.144

Coef. Dec. SDE
26.18
0.826
0.205
B
OU SDE
30.58
0.900
0.051
D3GM (ours)
32.41
0.912
0.040

23


---Page Break---
results in Tab. 5 exhibit a differing trend from the image domain. The task-specific diffusion models
achieve better results and are more capable of capturing the complex degradation that occurs in the
frequency domain.

Generally, MRI uses a mask in the phase-encoding direction (the shortest anatomical direction [41])
to model the complex degradation caused by undersampling. In the knee data, based upon the
uncertainty of clinical diagnosis (longitudinal artifacts can significantly confuse the diagnosis of
meniscal injury), we also masked the frequency direction, which will cause resolution reduction,
deterioration of image features, and longitudinal artifacts, more qualitative results can be found in the
next section.

MRI Super-resolution: IXI4 contains clinical T1- and T2-weighted scans from three hospitals with
different imaging protocols: HH, Guys, and IOP. We selected 184 HH T2 subjects as the source-
domain data [train/val/test ratio: 7:1:2], and 30 subjects each from Guys and IOP as two target-domain
datasets without degradation, acquisition parameters, datasets, and patient-wise crossovers. The
central 60 slices were selected.

We consider two benchmark ideas: (1) Blind SR degradation methods in frequency SFM [19] and
spatial PDM [70] domain. (2) Source-free SR adaptation ACT [14] using external priors and test-
time adaptation proposed initially for MRI CST [14] with data consistency in the source-domain.
We are concerned that existing work on robustness and generalization focuses on medical image
segmentation and natural images, and rarely on super-resolution and reconstruction of medical images.
Employing the available evaluation criteria in this inevitable problem in practice makes it difficult to
reflect the performance of D3GM. Therefore, domain-aware datasets isolation standard for public
datasets is designed for the SRR adaptation to cross-domain data rather than same-domain in a
source-free manner. The publicly available data were split into several subsets based on hospitals,
scanner, acquisition parameters, modality, and anatomy as illustrated in Table 9. Explicit reference
standards implicitly correspond to various degradation patterns, thus enabling the isolation of natural
degradation patterns in source training domain and target-testing domain. In addition to this, different
artificial degradation patterns for the subsets in the training and testing domains are employed:
K-space truncation downsampling was applied to obtain LR data in the source domain and a kernel
degradation [5] was applied in target domain.

We reproduce two types of benchmark ideas on the top of EDSR [33] backbone to achieve the multi-
purpose goal: (1) Repurposed blind SR (BSR) for cross-domain data: We utilized BSR degradation
methods in frequency SFM [19] and spatial PDM [70] domain. (2) Test-time adaptation (TTA): We
compare to a source-free TTA method ACT [14] using external priors and a second TTA method
proposed initially for MR reconstruction CST [14] with cycle consistency at source-domain. The
settings and adaptation strategies of the comparison methods were used directly.

Lipschitz Continuity for Stationary Process:

We hypothesize that ensuring Lipschitz continuity of the neural network is pivotal for the convergence
and stability of the diffusion process, particularly in the context of achieving a stationary process.
Theoretically, Lipschitz continuity offers two key benefits: (1) it mitigates the impact of small
perturbations in input data or model parameters, thus safeguarding against excessive output variability
which could cause numerical instabilities or result in an ill-posed problem, and (2) it guarantees the
existence of a unique solution to the diffusion process, underpinning the reliability and convergence
of the numerical methods employed to solve these equations.

To instantiate these theoretical benefits within our architecture, we integrated spectral normalization
(SN) [40] and weight decay (WD) [35] into a U-Net structured score network. In the case of SN, it is
achieved by rescaling each layer’s weight matrix W (l) by its spectral norm, σmax(W (l)), to obtain
a normalized weight ˜W (l) =
W (l)

σmax(W (l)). By doing so, we intend to control the overall Lipschitz
constant of the network for robust score matching within our diffusion model framework. A quantitive
result can be seen in Tab. 10.

4http://brain-development.org/ixi-dataset/

24


---Page Break---
𝒙𝟎
𝒙"

𝒚

𝒚’
𝒙$𝟎

𝓐

𝓐

𝒕−𝟏  →𝒕

𝒕   → 𝒕−𝟏

Basin of attraction

Stability

Figure 4: Reverse Initialization with Basin of attraction.

I
Additional Insights

Basin of Attraction in Reverse Sampling: Within our diffusion framework, we establish a forward
process with variety to accommodate a wide range of potential corruptions, ensuring the desired
final distribution close to the expected distribution. We consider that our diffusion models inherently
encompass the forward operator A within their structure. The transition from high to low-quality
images is implicitly encoded in the diffusion pathway, hence the additional forward operator guidance
might not contribute supplementary information, which we initially hypothesized would enhance the
reverse process.

Thus, the keypoint transfers from the A to the initial y′ (Detailed analysis based on Prop. 2 is provided
in Appx. F). Also we found this problem fall into the Basin of Attraction (BA) in dynamical system
as shown in Fig. 4. BA can be interpreted as the quality of the attractor of the degraded image in the
reverse process here. A BA-guided initialization might be a more effective approach for posterior
sampling in transitional SDE diffusion models. By initializing the reverse process closer to the
expected solution, we may bypass the initial hurdle of distribution mismatch.

J
More Qualitative Results

25


---Page Break---
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

Figure 5: Deraining results with light rain images of our method.

26


---Page Break---
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

Figure 6: Deraining results with heavy rain images of our method.

27


---Page Break---
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

Figure 7: Dehazing results with real hazy images of our method.

28


---Page Break---
8 X
16 X
8 X
16 X

LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

𝐹𝑟𝑒𝑞𝑢𝑒𝑛𝑐𝑦−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛
𝑃ℎ𝑎𝑠𝑒−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛

Figure 8: MRI reconstruction results with undersampling rate x8 and x16, on Frequency-encoding
and Phase-encoding directions.

29


---Page Break---
HH 4X

GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

Figure 9: MRI super-resolution results with in-domain images of our method.

30


---Page Break---
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

Guys 4X
IOP 4X

Figure 10: MRI super-resolution results with cross-domain (different imaging devices and degradation
methods) images of our method.

31


---Page Break---
8 X
16 X

LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

𝐹𝑟𝑒𝑞𝑢𝑒𝑛𝑐𝑦−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛
𝑃ℎ𝑎𝑠𝑒−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛

LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)
GT
LQ
𝑫𝟑𝑮𝑴 (𝐨𝐮𝐫𝐬)

𝐹𝑟𝑒𝑞𝑢𝑒𝑛𝑐𝑦−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛
𝑃ℎ𝑎𝑠𝑒−𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛

Figure 11: MRI reconstruction results with undersampling rate x8 and x16 of our method, on Phase-
encoding and Frequency-encoding directions.

32


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The main claims made in the abstract and introduction accurately reflect
the paper’s contributions and scope. The abstract clearly outlines the claims of existing
approaches, introduces the novel score-based diffusion framework (D3GM), and highlights
the effectiveness of the proposed method through extensive experimental results.
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
Justification: The paper discusses the limitations of the method under the discussion, and
provide more details in the appendix. The main claims made in the paper accurately reflect
the paper’s scope. Thus, we also provide more extra findings beyond the limitation.
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

33


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: The paper provides a full set of assumptions and complete proofs for the
theoretical results introduced, including the measure-preserving dynamics of Random
Dynamical Systems (RDS) and the Temporal Distribution Discrepancy analysis.

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

Justification: The paper fully discloses all the information needed to reproduce the main ex-
perimental results, including details about the benchmarks used, such as magnetic resonance
imaging, and the effectiveness of the D3GM framework across multiple benchmarks.

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

34


---Page Break---
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [No]

Justification: We have provide all the details and instructions along with the paper and the
appendix for the reproducible results. The datasets involved in the paper are public.

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

Justification: The paper specifies all necessary training and test details, such as data splits,
hyperparameters, and optimizer types, ensuring that the results can be fully understood and
appreciated. Full details are provided in the supplemental material to complement the core
paper.

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

35


---Page Break---
Justification: The paper reports evaluation metrics and other appropriate statistical signifi-
cance information for the experiments, clearly explaining the factors of variability and the
methods used for calculating these measures. On different downstream experiments, we
adapted appropriate evaluation metrics, following state-of-the-art work.
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
figures symmetric error bars that would yield results that are out of range (e.g., negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]

Justification: The paper provides detailed information on the compute resources required for
the experiments.
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
Justification: The research conducted in the paper conforms to the NeurIPS Code of Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consider-
ation due to laws or regulations in their jurisdiction).

36


---Page Break---
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: The paper discusses both potential positive and negative societal impacts of
the work, such as the improvement of diffusion models for medical imaging and the risk of
misuse in generating deceptive content.
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
Answer: [Yes]
Justification: Necessary safeguards are described for the responsible release of models
medical and image generators. These safeguards help ensure controlled and ethical use.
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

37


---Page Break---
Justification: The paper properly credits the contributors of existing assets used in the
research.
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
Answer: [Yes]
Justification: The new assets introduced in the paper, including the D3GM framework, are
well-documented, with comprehensive documentation provided alongside the assets. This
facilitates their use and further development by the research community.
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
Justification: NA. The paper does not involve crowdsourcing or research with human
subjects, so this section is not applicable.
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

38


---Page Break---
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: NA. The paper does not involve crowdsourcing or research with human
subjects, so this section is not applicable.
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

39


---Page Break---
