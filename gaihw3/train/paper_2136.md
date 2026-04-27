SubgDiff: A Subgraph Diffusion Model to Improve
Molecular Representation Learning

Jiying Zhang, Zijing Liu∗, Yu Wang, Bin Feng, Yu Li∗
International Digital Economy Academy (IDEA)
{zhangjiying,liuzijing,fengbin,liyu}@idea.edu.cn

https://github.com/IDEA-XL/SubgDiff

Abstract

Molecular representation learning has shown great success in advancing AI-based
drug discovery. A key insight of many recent works is that the 3D geometric
structure of molecules provides essential information about their physicochemical
properties. Recently, denoising diffusion probabilistic models have achieved im-
pressive performance in molecular 3D conformation generation. However, most
existing molecular diffusion models treat each atom as an independent entity,
overlooking the dependency among atoms within the substructures. This paper
introduces a novel approach that enhances molecular representation learning by
incorporating substructural information in the diffusion model framework. We
propose a novel diffusion model termed SubgDiff for involving the molecular
subgraph information in diffusion. Specifically, SubgDiff adopts three vital tech-
niques: i) subgraph prediction, ii) expectation state, and iii) k-step same subgraph
diffusion, to enhance the perception of molecular substructure in the denoising
network. Experiments on extensive downstream tasks, especially the molecular
force predictions, demonstrate the superior performance of our approach.

1
Introduction

Figure 1: Equilibrium probability of the six dif-
ferent 3D structures (c1–c6) of the same molecule
ibuprofen (C13H18O2) in four different conditions.
(Adapted with permission from [26]. Copyright 2018 American
Chemical Society.)

Molecular representation learning (MRL) has at-
tracted tremendous attention due to its significant role
in learning from limited labeled data for applications
like AI-based drug discovery [36, 3, 4, 24] and ma-
terial science [32]. From the perspective of physical
chemistry, the 3D molecular conformation can largely
determine the properties of molecules and the activ-
ities of drugs [6]. Thus, numerous geometric neural
network architectures and self-supervised learning
strategies have been proposed to explore 3D molecu-
lar structures to improve performance on downstream
molecular property prediction tasks [34, 55, 22, 57].

Meanwhile, diffusion probabilistic models (DPMs)
have shown remarkable power to generate realistic
samples, especially in synthesizing high-quality im-
ages and videos [39, 9]. By modeling the generation
as a reverse diffusion process, DPMs transform a
random noise into a sample in the target distribution.
Recently, diffusion models/flow models have also

∗Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
𝝐𝑖: Gaussian noise;  
𝐑𝒊: Atomic coordinates of 3D molecule

𝝐1

𝐬2

𝝐3
𝝐𝑇, 𝒔𝑇

𝐬1

𝝐2

𝐬3

𝐑0
𝐑1
𝐑1
𝐑2
𝐑2
𝐑𝑇

SUBGDIFF

𝐑1

𝐑0
𝐑2

𝝐1
𝝐3
𝝐𝑇
𝝐2

𝐑𝑇

DDPM

: Subgraph selected by mask vector 𝐬𝑖

Figure 2: Comparison of forward process between DDPM [9] and subgraph diffusion. For each step, DDPM
adds noise into all atomic coordinates, while subgraph diffusion selects a subset of the atoms to diffuse.

demonstrated strong capabilities of molecular 3D conformation generation [51, 15, 58, 42]. The
training process of a DPM for conformation generation can be viewed as the reconstruction of the
original conformation from a noisy version, where the noise is modulated by different time steps.
Consequently, the denoising objective in the diffusion model can naturally be used as a self-supervised
representation learning technique [31]. Inspired by this intuition, several works have used this tech-
nique for molecule pretraining [23, 55]. Despite considerable progress, the potential of DPMs for
molecular representation learning has not been fully explored. We therefore raise the question: Can
we effectively enhance MRL with the denoising network (noise predictor) of DPM? If yes, how to
achieve it?

To answer this question, we first analyze the gap between the current DPMs and the characteristics of
molecular structures. Most diffusion models on molecules propose to independently inject continuous
Gaussian noise into the every node feature [10] or atomic coordinates of 3D molecular geometry [51,
55]. However, this approach treats each atom as an individual particle, overlooking the substructure
within molecules, which is pivotal in molecular representation learning [54, 45, 27]. As shown
in Figure 1, the substructures are mostly invariant in different 3D molecular conformations, which
contains crucial information about the properties, such as the equilibrium distribution, crystallization
and solubility [26]. As a result, uniformly adding same-scale Gaussian noise to all atoms makes
it difficult for the denoising network to learn the joint distribution of global structure and local
substructure, hindering the downstream prediction performance on molecular properties closely
related to 3D conformation (e.g. most physicochemical properties). So here we try to tackle the
previous question by designing a DPM involving the knowledge of substructures.

Toward this goal, we propose a novel diffusion model termed SubgDiff, adding distinct Gaussian noise
to different substructures of 3D molecular conformation. Specifically, instead of adding the same
Gaussian noise to every atomic coordinate, SubgDiff only adds noise to a randomly selected subgraph
at each time step in the diffusion process (Figure 2). In the training phase of SubgDiff, a subgraph
prediction task is integrated into the training objective, which explicitly directs the denoising network
to capture substructure information from the molecules. Additionally, we propose two techniques:
expectation state diffusion and k-step same-subgraph diffusion, to train the model effectively and
enhance its sampling capability.

With the ability to capture the substructure information from the noisy 3D molecule, the denoising
networks tend to gain more representation power. The experiments on various 2D and 3D molecular
property prediction tasks demonstrate the superior performance of our approach. To summarize, our
contributions are as follows: (1) we incorporate the substructure information into diffusion models
to improve molecular representation learning; (2) we propose a new diffusion model SubgDiff that
adopts subgraph prediction, expectation state and k-step same-subgraph diffusion to improve its
sampling and training; (3) the proposed representation learning method achieves superior performance
on various downstream tasks, especially molecular forces prediction.

2


---Page Break---
2
Related work

Diffusion models on graphs. The diffusion models on graphs can be mainly divided into two
categories: continuous diffusion and discrete diffusion. Continuous diffusion applies a Gaussian
noise process on each node or edge [13, 29], including GeoDiff [51], EDM [10], SubDiff [53].
Meanwhile, discrete diffusion constructs the Markov chain on discrete space, including Digress [8]
and GraphARM [17]. However, it remains open to exploring fusing the discrete characteristic into
the continuous Gaussian on graph learning, although a closely related work has been proposed for
images and cannot be used for generation [31]. Our work, SubgDiff, is the first diffusion model
fusing subgraph, combining discrete characteristics and the continuous Gaussian.

Conformation generation. Various deep generative models have been proposed for conformation
generation, including CVGAE [25], GraphDG [38], CGCF [49], ConfVAE [50], ConfGF [37] and
GeoMol [7]. Recently, diffusion-based methods have shown competitive performance. Torsional
Diffusion [15] raises a diffusion process on the hypertorus defined by torsion angles. However, it is
not suitable as a representation learning technique due to the lack of local information (length and
angle of bonds). GeoDiff [51] generates molecular conformation with a diffusion model on atomic
coordinates. However, it views the atoms as separate particles, without considering the dependence
between atoms from the substructure.

SSL for molecular property prediction. There exist several works leveraging the 3D molecular
conformation to boost the representation learning, including GraphMVP [21], GeoSSL [23], the
denoising pretraining approach raised by Zaidi et al. [55] and MoleculeSDE [22], etc. However,
those studies have not considered the molecular substructure in the pertaining. In this paper, we
concentrate on how to boost the perception of molecular substructure in the denoising networks
through the diffusion model.

The discussion with more related works (e.g. MDM [31], MDSM [18] and SSSD [1]) can be found
in Appendix B.1.

3
Preliminaries

Notations. We use I to denote the identity matrix with dimensionality implied by context, ⊙to
represent the element product, and diag(s) to denote the diagonal matrix with diagonal elements of the
vector s. If not specified, both ϵ and z represent noise sampled from the standard Gaussian distribution
N(0, I). The topological molecular graph can be denoted as G(V, E, X) where V is the set of nodes,
E is the set of edges, X is the node feature matrix, and its corresponding 3D Conformational
Molecular Graph is represented as G3D(G, R), where R = [R1, · · · , R|V|] ∈R|V|×3 is the set of
3D coordinates of atoms.

DDPM. Denoising diffusion probabilistic models (DDPM) [9] is a typical diffusion model [39] which
consists of a diffusion (aka forward) and a reverse process. In the setting of molecular conformation
generation, the diffusion model adds noise on the 3D molecular coordinates R [51].

Forward and reverse process. Given the fixed variance schedule β1, β2, · · · , βT , the posterior
distribution q(R1:T |R0) that is fixed to a Markov chain can be written as

q(R1:T |R0) =

T
Y

t=1
q(Rt|Rt−1);
q(Rt|Rt−1) = N(Rt;
p

1 −βtRt−1, βtI).
(1)

To simplify notation, we consider the diffusion on single atom coordinate Rv and omit the subscript
v to get the general notion R throughout the paper. Let αt = 1 −βt, ¯αt = Qt
i=1(1 −βi), and then
the sampling of Rt at any time step t has the closed form: q(Rt|R0) = N(Rt; √¯αtR0, (1 −¯αt)I).

The reverse process of DDPM is defined as a Markov chain starting from a Gaussian distribution
p(RT ) = N(RT ; 0, I):

pθ(R0:T ) = p(RT )

T
Y

t=1
pθ(Rt−1|Rt);
pθ(Rt−1|Rt) = N(Rt−1; µθ(Rt, t), σt),
(2)

where σt = 1−¯αt−1

1−¯αt βt denote time-dependent constant. In DDPM, µθ(Rt, t) is parameterized as
µθ(Rt, t) =
1
¯αt (Rt −
βt
√1−¯αt ϵθ(Rt, t)) and ϵθ, i.e., the denoising network, is parameterized by a
neural network where the inputs are Rt and time step t.

3


---Page Break---
Training and sampling. The training objective of DDPM is:

Lsimple(θ) = Et,R0,ϵ[∥ϵ −ϵθ(√¯αtR0 +
√

1 −¯αtϵ, t)∥2].
(3)

After training, samples are generated through the reverse process pθ(R0:T ). Specifically, RT is first
sampled from N(0, I), and Rt in each step is predicted as follows,

Rt−1 =
1
√αt (Rt −1 −αt
√1 −¯αt
ϵθ(Rt, t)) + σtz,
z ∼N(0, I).
(4)

4
SubgDiff

Directly using DDPM on atomic coordinates of 3D molecules means each atom is viewed as an
independent single data point. However, the substructures play an important role in molecular
generation [14] and representation learning [56]. Ignoring the inherent interactions among atoms
within substructures may hinder the denoising network from learning features for molecular prop-
erties related to 3D conformation. In this paper, we propose to involve a mask operation in each
diffusion step, leading to a new diffusion SubgDiff for molecular representation learning. Each mask
corresponds to a subgraph in the molecular graph, aligning with the substructure in the 3D molecule.
Furthermore, we incorporate a subgraph predictor and reset the state of the Markov Chain to the
expectation of atomic coordinates, thereby enhancing the effectiveness of SubgDiff in sampling.
Additionally, we also propose k-step same-subgraph diffusion for training to effectively capture the
substructure information.

4.1
Involving subgraph into diffusion process

𝑅𝑡−1
𝑅𝑡

𝑠𝑡= 0

𝑠𝑡= 1

Figure 3: The Markov Chain of
SubgDiff is a lazy Markov Chain.

In the forward process of DDPM, we have Rt
v = √1 −βtRt−1
v
+
√βtϵt−1, ∀v ∈V, in which the Gaussian noise ϵt−1 is injected to
every atom. Moreover, the training objective in Equation 3 shows
that the denoising networks would always predict a Gaussian noise
for all atoms. Neither the forward nor reverse process of DDPM takes
into account the substructure of the molecule. Instead, in SubgDiff,
a mask vector st = [st1, · · · , st|V|]⊤∈{0, 1}|V| is introduced to
determine which atoms will be added noise at step t. The mask
vector st is sampled from a discrete distribution pst(S | G) to
select a subset of the atoms. In molecular graphs, the discrete
mask distribution pst(S | G) is equivalent to the subgraph distribution, defined over a predefined
sample space χ = {Gi
sub}N
i=1, where each sample is a connected subgraph extracted from G.
Further, the distribution pst(S | G) should keep the selected connected subgraph to cohere with
the molecular substructures. Here, we adopt a Torsional-based decomposition method [15] (Details
in Appendix A.2). With the mask vector as latent variables s1:t, the state transition of the forward
process can be formulated as (Figure 3):

Rt
v =
√1 −βtRt−1
v
+ √βtϵt−1
if stv = 1
Rt
v
if stv = 0,
(5)

which can be rewritten as Rt
v =
p

1 −stvβtRt−1
v
+
p

stvβtϵt−1.

The posterior distribution q(R1:T |R0, s1:T ) can be expressed as matrix form:

q(R1:T |R0, s1:T ) =

T
Y

t=1
q(Rt|Rt−1, st); q(Rt|Rt−1, st) = N(Rt;
p

1 −βtdiag(st)Rt−1, βtdiag(st)I). (6)

To simplify the notation, we consider the diffusion on a single node v and omit the subscript v in Rt
v
and stv to get the notion Rt and st. By defining γt := 1 −stβt, ¯γt := Qt
i=1(1 −stβt), the closed
form of sampling Rt given R0 is
q(Rt|R0, s1:t) = N(Rt; √¯γtR0, (1 −¯γt)I).
(7)

4.2
Reverse process learning

The reverse process is decomposed as follows:

4


---Page Break---
pθ,ϑ(R0:T , s1:T ) = p(RT )

T
Y

t=1
pθ(Rt−1|Rt, st)pϑ(st|Rt),
(8)

where pθ(Rt−1|Rt, st) and pθ(st|Rt) are both learnable models. In the context of molecular learning,
the model can be regarded as first predicting which subgraph st should be denoised and then using
the noise prediction network pθ(Rt−1|Rt, st) to denoise the node position in the subgraph.

However, it is tricky to generate a 3D structure by adopting the typical training and sampling method
used in Ho et al. [9]. Specifically, following Ho et al. [9], the reverse process can be optimized by
maximizing the variational lower bound (VLB) of log p(R0) as follows,

log p(R0) ≥

T
X

t=1
Eq(Rt,st|R0)


log pϑ(st|Rt)

q(st)



|
{z
}
subgraph prediction term

+ Eq(R1,s1|R0)

log pθ(R0|R1)


|
{z
}
reconstruction term

−

Eq(s1:t)DKL(q(RT |R0, s1:T ) ∥p(RT ))
|
{z
}
prior matching term

−

T
X

t=2
Eq(Rt,s1:t|R0)

DKL(q(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))


|
{z
}
denoising matching term

.

(9)

Details of the derivation are provided in the Appendix D.2. The subgraph predictor pϑ(st|Rt) in
the first term can be parameterized by a node classifier sϑ. For the denoising matching term that is
closely related to sampling, by Bayes rule, the posterior q(Rt−1|Rt, R0, s1:t) can be written as:

q(Rt−1|Rt, R0, s1:t) ∝N(Rt−1; µq(Rt, R0, s1:t, ϵ0), σ2
q(t)),
(10)

µq(Rt, R0, s1:t, ϵ0) =
1
√1 −βtst
(Rt −
βtst
p

(1 −βtst)(1 −¯γt−1) + βtst

ϵ0),

where σq(t) is the standard deviation and s1:t−1 are contained in ¯γt−1. Following DDPM and
parameterizing pθ(Rt−1|Rt, st) as N(Rt−1; µq(Rt, R0, s1:t−1, sϑ(G, Rt, t), ϵθ(G, Rt, t)), σq(t)I),
the training objective is

Lsimple(θ, ϑ) = Et,R0,st,ϵ[∥diag(st)(ϵ −ϵθ(G, Rt, t))∥2 + λBCE(st, sϑ(G, Rt, t))],
(11)

where BCE(st, sϑ) is the binary cross entropy loss, λ is the weight used for the trade-off, and sϑ
is the subgraph predictor implemented as a node classifier with G3D(G, Rt) as input and shares a
molecule encoder with ϵθ. The BCE loss employed here uses the subgraph selected at time-step t as
the target, thereby explicitly compelling the denoising network to capture substructure information
from molecules. Eventually, the sϑ can be used to infer the mask vector ˆst = sϑ(G, Rt, t) during
sampling. Thus, the sampling process is:

Rt−1 = µq(Rt, R0, s1:t−1, sϑ(G, Rt, t), ϵθ(G, Rt, t)) + σq(t)z.
(12)

However, using Equation 11 and Equation 12 directly for training and sampling faces two issues.
First, the inability to access s1:t−1 in µq during the sampling process hinders the step-wise denoising
procedure, posing a challenge to the utilization of conventional sampling methods in this context.
Inferring s1:t−1 solely from Rt using another model pθ(s1:t−1|Rt, st) is also difficult due to the
intricate modulation of noise introduced in Rt through multi-step Gaussian noising. Second, training
the subgraph predictor with Equation 11 is challenging. To be specific, the subgraph predictor should
be capable of perceiving the sensible noise change between time steps t −1 and t. However, the
noise scale βt is relatively small when t is small, especially if the diffusion step is large (e.g. 1000).
As a result, it is difficult to precisely predict the subgraph.

Next, to effectively tackle the above issues, we design two techniques: expectation state diffusion and
k-step same subgraph diffusion.

4.3
Expectation state diffusion.

We first devise a new way to calculate the denoising term. To eliminate the effect of mask series s1:t
and improve the training of subgraph prediction loss, we use a new lower bound of the denoising
term as follows:

Eq(Rt,s1:t|R0)

DKL(q(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))


≤Eq(Rt,s1:t|R0)

DKL(ˆq(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))
,

5


---Page Break---
where ˆq(Rt−1|Rt, R0, s1:t) is defined as
q(Rt|Es1:t−1Rt−1,R0,st)q(Es1:t−1Rt−1|R0)

q(Rt|R0,s1:t)
. It is an approxi-
mated posterior that only relies on the expectation of Rt and st. This lower bound defines a new
forward process, in which, state 0 to state t−1 use the Es1:t−1Rt−1 and state t remains as Equation 6.
Assume each node v ∈V, stv ∼Bern(p) (i.i.d. w.r.t. t). Formally, we have

q(Rt|Rt−1, st) = N(Rt;
p

1 −stβtERt−1, (stβt)I),
(13)

q(ERt−1|R0, s1:t−1) = N(ERt−1;

t−1
Y

i=1

√αiR0, p2
t−1
X

i=1

t−1
Y

j=i+1
αjβiI),
(14)

where αi := (p√1 −βi + 1 −p)2 and ¯αt := Qt
i=1 αi are general form of αj and ¯αj in DDPM
(p = 1), respectively. Intuitively, this process is equivalent to using mean state ERt−1 to replace
Rt−1 during the forward process. This estimation is reasonable since the expectation Es1:t−1Rt−1 is
like a cluster center of Rt−1, which can represent Rt−1 properly. Thus, the approximated posterior
becomes

ˆq(Rt−1|Rt, R0, s1:t) ∝N(Rt−1; µˆq(Rt, R0, s1:t, ϵ0), σ2
ˆq(t)),
(15)

where

µˆq(Rt, R0, s1:t, ϵ0) :=
1
√1 −βtst
(Rt −
stβt
q

(stβt + (1 −stβt)p2 Pt−1
i=1
¯αt−1

¯αi βi

ϵ0)

σ2
ˆq(t) := stβtp2
t−1
X

i=1

¯αt−1

¯αi
βi/(stβt + p2(1 −stβt)

t−1
X

i=1

¯αt−1

¯αi
βi).

We parameterize pθ(Rt−1|Rt, st) as N(Rt−1; µˆq(Rt, R0, s1:t−1, sϑ(G, Rt, t), ϵθ(Rt, t)), σˆq(t)I),
and adopt the same training objective as Equation 11. By employing the sampling method Rt−1 =
µˆq(Rt, R0, s1:t−1, sϑ(G, Rt, t), ϵθ(Rt, t)) + σˆq(t)z, we observe that the proposed expectation state
enables a step-by-step execution of the sampling process. Moreover, using expectation is beneficial
to reduce the complexity of Rt for predicting the mask st during training. This will improve the
denoising network to perceive the substructure when we use the diffusion model for self-supervised
learning.

4.4
k-step same-subgraph diffusion.

To reduce the complexity of the mask series (s1, s2, · · · , sT ) and accumulate more noise on the same
subgraph for facilitating the convergence of the subgraph prediction loss, we generalize the one-step
subgraph diffusion to k-step same subgraph diffusion (Figure 8 in Appendix), in which the selected
subgraph will be continuously diffused k steps. After that, the difference between the selected and
unselected parts will be distinct enough to help the subgraph predictor perceive it. The forward
process of k-step same subgraph diffusion can be written as (t > k, k ∈N):

q(Rt|Rt−k, st−k+1:t) = N



Rt,

v
u
u
t

tY

i=t−k+1
(1 −st−k+1βi)Rt−k, σk
t



,
(16)

where σk
t = (1 −Qt
i=t−k(1 −st−k+1βi)I.

4.5
Training and sampling of SubgDiff

By combining the expectation state and k-step same-subgraph diffusion, SubgDiff first divides the
entire diffusion step T into T/k diffusion intervals. In each interval [ki, k(i + 1)], the mask vectors
{sj}k(i+1)
j=ki+2 are equal to ski+1. SubgDiff then adopts the expectation state at the split time step
{ik|i = 1, 2, · · · } to eliminate the effect of {sik+1|i = 1, 2, . . .}, that is, gets the expectation of
ERik at step ik w.r.t. sik+1. Overall, the diffusion process of SubgDiff is a two-phase diffusion
process. In the first phase, the state 1 to state k⌊(t −1)/k⌋use the expectation state diffusion, while
in the second phase, state k(⌊(t −1)/k⌋) + 1 to state t use the k-step same subgraph diffusion
(see Figure 4). With m := ⌊(t −1)/k⌋, the two phases can be formulated as follows,

Phase I: Step 0 →km: Es1:kmRkm = √¯αmR0 + p
qPm
l=1
¯αm

¯αl (1 −Qkl
i=(l−1)k+1(1 −βi))ϵ0,

where αj = (p
qQkj
i=(j−1)k+1(1 −βi) + 1 −p)2 is a general forms of αj in Equation 14 (in which

6


---Page Break---
𝝐𝑘−1 𝑚+1:𝑘𝑚

𝐑0

(𝑡−𝑘𝑚)-step same-subgraph diffusion

𝝐1:𝑘

𝔼𝐬1:𝑘𝐑𝑘

𝐬𝑖= 𝐬1 𝑖=1

𝑘
𝐬𝑖= 𝐬𝑘𝑚−𝑚+1 𝑖=𝑘𝑚−𝑚+1

𝑘𝑚

𝔼𝐬1:𝑘𝑚𝐑𝑘m

𝝐𝑘+1:2𝑘

𝐬𝑖= 𝐬𝑘+1 𝑖=𝑘+1

2𝑘

𝐑𝑡

𝝐𝑘𝑚+1:𝑡

𝐬𝑖= 𝐬𝑘𝑚+1 𝑖=𝑘𝑚+1

𝑡

Expectation state diffusion

𝐬𝑘𝑚+1

𝔼𝐬1:𝑘𝑚𝐑𝑘m

𝐬𝑖: mask random variable; 𝝐𝑖: Gaussian noise; 𝔼𝒔𝐑: the expectation of 𝐑w.r.t 𝐬
: subgraph selected by mask sample 𝐬𝑘𝑚+1

Figure 4: The forward process of SubgDiff. The state 0 to km uses the expectation state and the mask variables
are the same in the interval [ki, ki + k], i = 0, 1, ..., m −1. The state km + 1 to t applies the same subgraph
diffusion.

𝝐1:𝑘

ො𝒔2𝑘

𝝐2𝑘+1:3𝑘

𝝐𝑇, 𝒔𝑇

𝒔1 = ො𝒔𝑘

𝝐𝑘+1:2𝑘

ො𝒔2𝑘+1 = ො𝒔3𝑘

𝐑0
𝐑𝑘
𝐑𝑘
𝐑2𝑘
𝐑2𝑘
𝐑𝑇

Reverse Process
ො𝒔𝑘

ො𝒔𝑘+1 = ො𝒔2𝑘

ො𝒔𝑘𝑚+1 = ො𝒔𝑇

Figure 5: The reverse process of SubgDiff. The selected subgraph s is the same in the interval [ki, min(ki +
k, T)], i = 0, ..., m

case k = 1) and ¯αi = Qt
j=1 αj. In the rest of the paper, αj denotes the general version without
a special statement. Actually, Es1:kmRkm only calculate the expectation w.r.t. random variables
{sik+1|i = 1, 2, · · · }.

Phase II: Step km + 1 →t: The phase is a (t −km)-step same mask diffusion.
Rt =
qQt
i=km+1(1 −βiskm+1)Es1:kmRkm +
q

1 −Qt
i=km+1(1 −βiskm+1)ϵkm.

Let γi = 1 −βiskm+1, ¯γt = Qt
i=1 γi, and ¯βt = Qt
i=1(1 −βi). We can drive the single-step state
transition: q(Rt|Rt−1) = N(Rt; √γtRt−1, (1 −γt)I) and

q(Rt−1|R0) = N(Rt−1;
r ¯γt−1¯αm

¯γkm
R0, δI);
δ := ¯γt−1

¯γkm
p2
m
X

l=1

¯αm

¯αl
(1 −
¯βkl
¯β(l−1)k
) + 1 −¯γt−1

¯γkm .
(17)

Then we reuse the training objective in Equation 11 as the objective of SubgDiff:

Lsimple(θ, ϑ) = Et,R0,st,ϵ[∥diag(st)(ϵ −ϵθ(G, Rt, t))∥2 + λBCE(st, sϑ(G, Rt, t))],
(18)

where Rt is calculated by Equation 17.

Sampling.
Although the forward process uses the expectation state w.r.t. s, we can only update the
mask ˆst at t = ik, i = 1, 2, · · · because sampling only needs to get a subgraph from the distribution in
the k-step interval. Eventually, adopting the δ defined in Equation 17, the sampling process (Figure 5)
is shown below,

Rt−1 =
1
√γt
(Rt −
ˆskm+1βt
p

γtδ + ˆskm+1βt
ϵθ(Rt, t))+

p

ˆskm+1βt
q

¯γt−1

¯γkm p2 Pm
l=1
¯αm

¯αl (1 −
¯βkl
¯β(l−1)k ) + 1 −¯γt−1

¯γkm
p

γtδ + ˆskm+1βt
z,
(19)

where z ∼N(0, I), m = ⌊(t −1)/k⌋and ˆskm+1 = sϑ(G, Rkm+k, km+k). The subgraph selected
by ˆskm+1 will be generated in from the steps km + k to km. The mask predictor can be viewed as a
discriminator of important subgraphs, indicating the optimal subgraph should be recovered in the
next k steps. After one subgraph (substructure) is generated properly, the model can gently fine-tune
the other parts of the molecule (c.f. the video in supplementary material). This subgraph diffusion
would intuitively increase the robustness and generalization of the generation process, which is also
verified by the experiments in Section 5.2. The training and sampling algorithms of SubgDiff are
summarized in Algorithm 1 and Algorithm 2.

7


---Page Break---
Algorithm 1: Training SubgDiff
Input: A molecular graph G3D, k for same mask diffusion, m := ⌊(t −1)/k⌋
Sample t ∼U(1, ..., T) , ϵ ∼N(0, I)
Sample skm+1 ∼pskm+1(S | G)
▷Sample a subgraph
Rt ←q(Rt|R0)
▷Equation 17
L1 = BCE(skm+1, sϑ(G, Rt, t))
▷Subgraph prediction loss
L2 = ∥diag(skm+1)(ϵ −ϵθ(G, Rt, t))∥2
▷Denoising loss
optimizer. step(Et,R0,st,ϵ[λL1 + L2])
▷Optimize parameters θ, ϑ

Algorithm 2: Sampling from SubgDiff
k is the same as training, for k-step same-subgraph diffusion;
Sample RT ∼N(0, I)
▷Random noise initialization
for t = T to 1 do

z ∼N(0, I) if t > 1, else z = 0
▷Random noise
If t%k == 0 or t == T: ˆs ←sϑ(G, Rt, t)
▷Subgraph prediction
ˆϵ ←ϵθ(G, Rt, t)
▷Posterior
Rt−1 ←Equation 19
▷sampling
end
return R0

5
Experiments

We conduct experiments to address the following two questions: 1) Can substructures improve the
representation ability of the denoising network when using diffusion as self-supervised learning? 2)
How does the proposed subgraph diffusion affect the generative ability of the diffusion models? For
the first question, we employ SubgDiff as a denoising pretraining task and evaluate the performances
of the denoising network on various downstream tasks. For the second one, we compare SubgDiff
with the vanilla diffusion model GeoDiff [51] on the task of molecular conformation generation.

5.1
SubgDiff improves molecular representation learning

To verify the introduced substructure in the diffusion can enhance the denoising network for represen-
tation learning, we pretrain with SubgDiff objective and finetune on various downstream tasks.

Dataset and settings. For pretraining, we follow [22] and use PCQM4Mv2 dataset [11]. It’s a sub-
dataset of PubChemQC [28] with 3.4 million molecules with 3D geometric conformations. We use
various molecular property prediction datasets as downstream tasks. For tasks with 3D conformations,
we consider the dataset MD17 and follow the literature [34, 35, 23] of using 1K for training and 1K
for validation, while the test set (from 48K to 991K) is much larger. For downstream tasks with only
2D molecule graphs, we use eight molecular property prediction tasks from MoleculeNet [47].

Pretraining framework. To explore the potential of the proposed method for representation learning,
we consider MoleculeSDE [22], a SOTA pretraining framework, to be the training backbone, where
SubgDiff is used for the 2D →3D model and the mask operation is extended to the node feature
and graph adjacency for the 3D →2D model. The details can be found in Appendix A.4.2.

Baselines. For 3D tasks, we incorporate two self-supervised methods [Type Prediction, Angle Pre-
diction], and three contrastive methods [InfoNCE [30] and EBM-NCE [21] and 3D InfoGraph [23]].
Two denoising baselines are also included [GeoSSL [23], Denoising [55] and MoleculeSDE]. For 2D
tasks, the baselines are AttrMask, ContexPred [12], InfoGraph [44], MolCLR [46], 3D InfoMax [43],
GraphMVP [21] and MoleculeSDE.

Results. As shown in Table 1 and Table 2, SubgDiff outperforms MoleculeSDE in both 2D and
3D downstream tasks. Particularly, the significant improvement on the MD17 force prediction task,
which is closely related to 3D molecular conformation, demonstrates that the introduced subgraph
diffusion helps the perception of molecular 3D structure in the denoising network during pretraining.
Further, SubgDiff achieves SOTA performance compared to all the baselines. This also indicates
that the proposed SubgDiff objective is promising for molecular representation learning due to the

8


---Page Break---
Table 1: Results (mean absolute error) on MD17 force prediction. The best and second best results are marked
in bold and underlined.

Pretraining
Aspirin ↓
Benzene ↓
Ethanol ↓
Malonaldehyde ↓
Naphthalene ↓
Salicylic ↓
Toluene ↓
Uracil ↓

– (random init)
1.203
0.380
0.386
0.794
0.587
0.826
0.568
0.773
Type Prediction
1.383
0.402
0.450
0.879
0.622
1.028
0.662
0.840
Angle Prediction
1.542
0.447
0.669
1.022
0.680
1.032
0.623
0.768
3D InfoGraph
1.610
0.415
0.560
0.900
0.788
1.278
0.768
1.110
InfoNCE
1.132
0.395
0.466
0.888
0.542
0.831
0.554
0.664
EBM-NCE
1.251
0.373
0.457
0.829
0.512
0.990
0.560
0.742
Denoising
1.364
0.391
0.432
0.830
0.599
0.817
0.628
0.607
GeoSSL
1.107
0.360
0.357
0.737
0.568
0.902
0.484
0.502
MoleculeSDE (VE)
1.112
0.304
0.282
0.520
0.455
0.725
0.515
0.447
MoleculeSDE (VP)
1.244
0.315
0.338
0.488
0.432
0.712
0.478
0.468

Ours
0.880
0.252
0.258
0.459
0.325
0.572
0.362
0.420

Table 2: Results for MoleculeNet (with 2D topology only). We report the mean (and standard deviation)
ROC-AUC of three random seeds with scaffold splitting for each task. The backbone is GIN. The best and
second best results are marked bold and underlined, respectively.

Pre-training
BBBP ↑
Tox21 ↑
ToxCast ↑
Sider ↑
ClinTox ↑
MUV ↑
HIV ↑
Bace ↑
Avg ↑

– (random init)
68.1±0.59
75.3±0.22
62.1±0.19
57.0±1.33
83.7±2.93
74.6±2.35
75.2±0.70
76.7±2.51
71.60
AttrMask
65.0±2.36
74.8±0.25
62.9±0.11
61.2±0.12
87.7±1.19
73.4±2.02
76.8±0.53
79.7±0.33
72.68
ContextPred
65.7±0.62
74.2±0.06
62.5±0.31
62.2±0.59
77.2±0.88
75.3±1.57
77.1±0.86
76.0±2.08
71.28
InfoGraph
67.5±0.11
73.2±0.43
63.7±0.50
59.9±0.30
76.5±1.07
74.1±0.74
75.1±0.99
77.8±0.88
70.96
MolCLR
66.6±1.89
73.0±0.16
62.9±0.38
57.5±1.77
86.1±0.95
72.5±2.38
76.2±1.51
71.5±3.17
70.79
3D InfoMax
68.3±1.12
76.1±0.18
64.8±0.25
60.6±0.78
79.9±3.49
74.4±2.45
75.9±0.59
79.7±1.54
72.47
GraphMVP
69.4±0.21
76.2±0.38
64.5±0.20
60.5±0.25
86.5±1.70
76.2±2.28
76.2±0.81
79.8±0.74
73.66
MoleculeSDE(VE)
68.3±0.25
76.9±0.23
64.7±0.06
60.2±0.29
80.8±2.53
76.8±1.71
77.0±1.68
79.9±1.76
73.15
MoleculeSDE(VP)
70.1±1.35
77.0±0.12
64.0±0.07
60.8±1.04
82.6±3.64
76.6±3.25
77.3±1.31
81.4±0.66
73.73

Ours
70.2±2.23
77.2±0.39
65.0±0.48
62.2±0.97
88.2±1.57
77.3±1.17
77.6±0.51
82.1±0.96
74.85

involvement of the knowledge of substructures during training. More results on the QM9 dataset [34]
on the quantum mechanics property prediction can be found in Appendix A.4.3.

5.2
SubgDiff benefits conformation generation

We have proposed a new diffusion model to enhance molecular representing learning, where the base
diffusion model (GeoDiff) is initially designed for conformation generation. To evaluate the effects
of SubgDiff on the generative ability of diffusion models, we assess its generation performance and
generalization ability. Following prior works [51], we utilize the GEOM-QM9 [33] and GEOM-
Drugs [2] datasets. The detailed description of the dataset splitting, metrics and model architecture
can be found in Appendix A.5.

Conformation generation. The comparison with GeoDiff on the GEOM-QM9 dataset is reported in
Table 3. From the results, it is easy to see that SubgDiff significantly outperforms the GeoDiff baseline
on both metrics (COV-R and MAT-R) across different sampling steps. It indicates that by training
with the substructure information, SubgDiff has a positive effect on the conformation generation task.
Moreover, SubgDiff with 500 steps achieves much better performance than GeoDiff with 5000 steps
on 5 out of 8 metrics, which implies our method can accelerate the sampling efficiency (10x).

Table 3: Results for conformation generation on GEOM-QM9 dataset with different diffusion timesteps.
DDPM [9] is the sampling method used in GeoDiff. Our proposed sampling method (Algorithm 2) can be
viewed as a DDPM variant. ■/ ■denotes SubgDiff outperforms/underperforms GeoDiff.

COV-R (%) ↑
MAT-R (Å) ↓
COV-P (%) ↑
MAT-P (Å) ↓
Models
Timesteps
Sampling method
Mean
Median
Mean
Median
Mean
Median
Mean
Median

GeoDiff
5000
DDPM
80.36
83.82
0.2820
0.2799
53.66
50.85
0.6673
0.4214
SubgDiff
5000
DDPM (ours)
90.91
95.59
0.2460
0.2351
50.16
48.01
0.6114
0.4791

GeoDiff
500
DDPM
80.20
83.59
0.3617
0.3412
45.49
45.45
1.1518
0.5087
SubgDiff
500
DDPM (ours)
89.78
94.17
0.2417
0.2449
50.03
48.31
0.5571
0.4921

GeoDiff
200
DDPM
69.90
72.04
0.4222
0.4272
36.71
33.51
0.8532
0.5554
SubgDiff
200
DDPM (ours)
85.53
88.99
0.2994
0.3033
47.76
45.89
0.6971
0.5118

9


---Page Break---
Table 4: Results on the GEOM-QM9 dataset for domain generalization.
Except for GeoDiff and SubgDiff, the other methods are trained with
in-domain data.

COV-R (%) ↑
MAT-R (Å) ↓
Models
Train data
Mean
Median
Mean
Median

CVGAE [25]
QM9
0.09
0.00
1.6713
1.6088
GraphDG [38]
QM9
73.33
84.21
0.4245
0.3973
CGCF [49]
QM9
78.05
82.48
0.4219
0.3900
ConfVAE [50]
QM9
77.84
88.20
0.4154
0.3739
GeoMol [7]
QM9
71.26
72.00
0.3731
0.3731
GeoDiff
Drugs
74.94
79.15
0.3492
0.3392

SubgDiff
Drugs
83.50
88.70
0.3116
0.3075

Domain generalization.
To
further illustrate the benefits of
SubgDiff, we design two cross-
domain tasks:
(1) Training on
QM9 (small molecular with up
to 9 heavy atoms) and testing on
Drugs (medium-sized organic com-
pounds); (2) Training on Drugs and
testing on QM9. The results (Ta-
ble 4 and Appendix Table 13) show
that SubgDiff consistently outper-
forms GeoDiff and other models
trained on the in-domain dataset,
demonstrating the introduced sub-
structure effectively enhances the robustness and generalization of the diffusion model.

6
Conclusion

We present a novel diffusion model SubgDiff, which involves the subgraph constraint in the diffusion
model by introducing a mask vector to the forward process. Benefiting from the expectation state and
k-step same-subgraph diffusion, SubgDiff effectively boosts the perception of molecular substructure
in the denoising network, thereby achieving state-of-the-art performance at various downstream
property prediction tasks. There are several exciting avenues for future work. The mask distribution
can be made flexible such that more chemical prior knowledge may be incorporated into efficient
subgraph sampling. Besides, the proposed SubgDiff can be generalized to proteins such that the
denoising network can learn meaningful secondary structures.

Acknowledgement

This project was supported by Shenzhen Hetao Shenzhen-Hong Kong Science and Technology
Innovation Cooperation Zone, under Grant No. HTHZQSWS-KCCYB-2023052.

References

[1] Juan Lopez Alcaraz and Nils Strodthoff. Diffusion-based time series imputation and forecasting
with structured state space models. Transactions on Machine Learning Research, 2022.

[2] Simon Axelrod and Rafael Gomez-Bombarelli. GEOM, energy-annotated molecular conforma-
tions for property prediction and molecular generation. Scientific Data, 9(1):185, 2022.

[3] He Cao, Zijing Liu, Xingyu Lu, Yuan Yao, and Yu Li. Instructmol: Multi-modal integration
for building a versatile and reliable molecular assistant in drug discovery.
arXiv preprint
arXiv:2311.16208, 2023.

[4] He Cao, Yanjun Shao, Zhiyuan Liu, Zijing Liu, Xiangru Tang, Yuan Yao, and Yu Li. PRESTO:
Progressive pretraining enhances synthetic chemistry outcomes. arXiv preprint arXiv:2406.13193,
2024.

[5] Stefan Chmiela, Alexandre Tkatchenko, Huziel E Sauceda, Igor Poltavsky, Kristof T Schütt, and
Klaus-Robert Müller. Machine learning of accurate energy-conserving molecular force fields.
Science advances, 3(5):e1603015, 2017.

[6] Aurora J Cruz-Cabeza and Joel Bernstein. Conformational polymorphism. Chemical reviews,
114(4):2170–2191, 2014.

[7] Octavian Ganea, Lagnajit Pattanaik, Connor Coley, Regina Barzilay, Klavs Jensen, William
Green, and Tommi Jaakkola. Geomol: Torsional geometric generation of molecular 3d conformer
ensembles. Advances in Neural Information Processing Systems, 34:13757–13769, 2021.

10


---Page Break---
[8] Kilian Konstantin Haefeli, Karolis Martinkus, Nathanaël Perraudin, and Roger Wattenhofer.
Diffusion models for graphs benefit from discrete state spaces. In The First Learning on Graphs
Conference, 2022.

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
in neural information processing systems, 33:6840–6851, 2020.

[10] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant
diffusion for molecule generation in 3d. In International conference on machine learning, pp.
8867–8887, 2022.

[11] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele
Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs.
arXiv preprint arXiv:2005.00687, 2020.

[12] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure
Leskovec. Strategies for pre-training graph neural networks. In International Conference on
Learning Representations, 2020.

[13] John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for
graph-based protein design. Advances in neural information processing systems, 32, 2019.

[14] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Hierarchical generation of molecular
graphs using structural motifs. In International conference on machine learning, pp. 4839–4848,
2020.

[15] Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi S Jaakkola. Torsional
diffusion for molecular conformer generation. In Advances in Neural Information Processing
Systems, 2022.

[16] Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Acta Crystallo-
graphica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography, 32
(5):922–923, 1976.

[17] Lingkai Kong, Jiaming Cui, Haotian Sun, Yuchen Zhuang, B. Aditya Prakash, and Chao Zhang.
Autoregressive diffusion model for graph generation. In Andreas Krause, Emma Brunskill,
Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International
Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research,
pp. 17391–17408, 23–29 Jul 2023.

[18] Jiachen Lei, Peng Cheng, Zhongjie Ba, and Kui Ren. Masked diffusion models are fast learners.
arXiv preprint arXiv:2306.11363, 2023.

[19] Haitao Lin, Yufei Huang, Odin Zhang, Yunfan Liu, Lirong Wu, Siyuan Li, Zhiyuan Chen,
and Stan Z. Li. Functional-group-based diffusion for pocket-specific molecule generation and
elaboration. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[20] Shengchao Liu, Mehmet F Demirel, and Yingyu Liang. N-gram graph: Simple unsupervised
representation for graphs, with applications to molecules. Advances in neural information
processing systems, 32, 2019.

[21] Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, and Jian Tang.
Pre-training molecular graph representation with 3D geometry. In International Conference on
Learning Representations, 2022.

[22] Shengchao Liu, Weitao Du, Zhi-Ming Ma, Hongyu Guo, and Jian Tang. A group symmetric
stochastic differential equation model for molecule multi-modal pretraining. In International
Conference on Machine Learning, pp. 21497–21526, 2023.

[23] Shengchao Liu, Hongyu Guo, and Jian Tang. Molecular geometry pretraining with SE(3)-
invariant denoising distance matching. In The Eleventh International Conference on Learning
Representations, 2023.

11


---Page Break---
[24] Xingyu Lu, He Cao, Zijing Liu, Shengyuan Bai, Leqing Chen, Yuan Yao, Hai-Tao Zheng, and
Yu Li. MoleculeQA: A dataset to evaluate factual accuracy in molecular comprehension. arXiv
preprint arXiv:2403.08192, 2024.

[25] Elman Mansimov, Omar Mahmood, Seokho Kang, and Kyunghyun Cho. Molecular geometry
prediction using a deep generative graph neural network. Scientific reports, 9(1):20381, 2019.

[26] Veselina Marinova, Geoffrey PF Wood, Ivan Marziano, and Matteo Salvalaglio. Dynamics and
thermodynamics of ibuprofen conformational isomerism at the crystal/solution interface. Journal
of chemical theory and computation, 14(12):6484–6494, 2018.

[27] Siqi Miao, Mia Liu, and Pan Li. Interpretable and generalizable graph learning via stochastic
attention mechanism. In International Conference on Machine Learning, pp. 15524–15543,
2022.

[28] Maho Nakata and Tomomi Shimazaki.
Pubchemqc project: a large-scale first-principles
electronic structure database for data-driven chemistry. Journal of chemical information and
modeling, 57(6):1300–1308, 2017.

[29] Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon.
Permutation invariant graph generation via score-based generative modeling. In International
Conference on Artificial Intelligence and Statistics, pp. 4474–4484, 2020.

[30] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive
predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[31] Zixuan Pan, Jianxu Chen, and Yiyu Shi. Masked diffusion as self-supervised representation
learner. arXiv preprint arXiv:2308.05695, 2023.

[32] Robert Pollice, Gabriel dos Passos Gomes, Matteo Aldeghi, Riley J Hickman, Mario Krenn,
Cyrille Lavigne, Michael Lindner-D’Addario, AkshatKumar Nigam, Cher Tian Ser, Zhenpeng
Yao, et al. Data-driven strategies for accelerated materials design. Accounts of Chemical Research,
54(4):849–860, 2021.

[33] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld.
Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7,
2014.

[34] Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre
Tkatchenko, and Klaus-Robert Müller. Schnet: A continuous-filter convolutional neural network
for modeling quantum interactions. Advances in neural information processing systems, 30,
2017.

[35] Kristof Schütt, Oliver Unke, and Michael Gastegger. Equivariant message passing for the
prediction of tensorial properties and molecular spectra. In International Conference on Machine
Learning, pp. 9377–9388, 2021.

[36] Jie Shen and Christos A Nicolaou. Molecular property prediction: recent trends in the era of
artificial intelligence. Drug Discovery Today: Technologies, 32:29–36, 2019.

[37] Chence Shi, Shitong Luo, Minkai Xu, and Jian Tang. Learning gradient fields for molecular
conformation generation. In International conference on machine learning, pp. 9558–9568,
2021.

[38] Gregor Simm and Jose Miguel Hernandez-Lobato. A generative model for molecular distance
geometry. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International
Conference on Machine Learning, volume 119, pp. 8949–8958, 2020.

[39] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsuper-
vised learning using nonequilibrium thermodynamics. In International conference on machine
learning, pp. 2256–2265, 2015.

[40] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data
distribution. In Advances in Neural Information Processing Systems, pp. 11918–11930, 2019.

12


---Page Break---
[41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon,
and Ben Poole. Score-based generative modeling through stochastic differential equations. In
International Conference on Learning Representations, 2020.

[42] Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou,
and Wei-Ying Ma. Equivariant flow matching with hybrid probability transport for 3d molecule
generation. Advances in Neural Information Processing Systems, 36, 2023.

[43] Hannes Stärk, Dominique Beaini, Gabriele Corso, Prudencio Tossou, Christian Dallago, Stephan
Günnemann, and Pietro Liò. 3D infomax improves GNNs for molecular property prediction. In
International Conference on Machine Learning, pp. 20479–20502, 2022.

[44] Fan-Yun Sun, Jordan Hoffmann, Vikas Verma, and Jian Tang. Infograph: Unsupervised and
semi-supervised graph-level representation learning via mutual information maximization. In
International Conference on Learning Representations (ICLR), 2020.

[45] Yuyang Wang, Rishikesh Magar, Chen Liang, and Amir Barati Farimani. Improving molecular
contrastive learning via faulty negative mitigation and decomposed fragment contrast. Journal of
Chemical Information and Modeling, 62(11):2713–2725, 2022.

[46] Yuyang Wang, Jianren Wang, Zhonglin Cao, and Amir Barati Farimani. Molecular contrastive
learning of representations via graph neural networks. Nature Machine Intelligence, 4(3):279–
287, 2022.

[47] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S
Pappu, Karl Leswing, and Vijay Pande. MoleculeNet: a benchmark for molecular machine
learning. Chemical science, 9(2):513–530, 2017.

[48] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? In International Conference on Learning Representations, 2018.

[49] Minkai Xu, Shitong Luo, Yoshua Bengio, Jian Peng, and Jian Tang. Learning neural generative
dynamics for molecular conformation generation. In International Conference on Learning
Representations, 2021.

[50] Minkai Xu, Wujie Wang, Shitong Luo, Chence Shi, Yoshua Bengio, Rafael Gomez-Bombarelli,
and Jian Tang. An end-to-end framework for molecular conformation generation via bilevel
programming. In International Conference on Machine Learning, pp. 11537–11547, 2021.

[51] Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. GeoDiff: A
geometric diffusion model for molecular conformation generation. In International Conference
on Learning Representations, 2022.

[52] Zhao Xu, Youzhi Luo, Xuan Zhang, Xinyi Xu, Yaochen Xie, Meng Liu, Kaleb Dickerson,
Cheng Deng, Maho Nakata, and Shuiwang Ji. Molecule3d: A benchmark for predicting 3d
geometries from molecular graphs. arXiv preprint arXiv:2110.01717, 2021.

[53] Kuo Yang, Zhengyang Zhou, Limin Li, Pengkun Wang, Xu Wang, and Yang Wang. Subdiff:
Subgraph latent diffusion model, 2024.

[54] Zhaoning Yu and Hongyang Gao. Molecular representation learning via heterogeneous motif
graph neural networks. In International Conference on Machine Learning, pp. 25581–25594,
2022.

[55] Sheheryar Zaidi, Michael Schaarschmidt, James Martens, Hyunjik Kim, Yee Whye Teh, Alvaro
Sanchez-Gonzalez, Peter Battaglia, Razvan Pascanu, and Jonathan Godwin. Pre-training via
denoising for molecular property prediction. In The Eleventh International Conference on
Learning Representations, 2023.

[56] Xuan Zang, Xianbing Zhao, and Buzhou Tang. Hierarchical molecular graph self-supervised
learning for property prediction. Communications Chemistry, 6(1):34, 2023.

13


---Page Break---
[57] Jiying Zhang, Xi Xiao, Long-Kai Huang, Yu Rong, and Yatao Bian. Fine-tuning graph neural
networks via graph topology induced optimal transport. In Lud De Raedt (ed.), Proceedings of
the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 3730–
3736. International Joint Conferences on Artificial Intelligence Organization, 7 2022. doi:
10.24963/ijcai.2022/518. Main Track.

[58] Jiying Zhang, Zijing Liu, Shengyuan Bai, He Cao, Yu Li, and Lei Zhang. Efficient antibody
structure refinement using energy-guided se (3) flow matching. arXiv preprint arXiv:2410.16673,
2024.

[59] Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, and Jian Tang. DiffPack:
A torsional diffusion model for autoregressive protein side-chain packing. In Thirty-seventh
Conference on Neural Information Processing Systems, 2023.

14


---Page Break---
Appendix

Contents

A Experiment details and more results
15

A.1 Visualization of representations . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15

A.2
Mask distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

A.3
Hyperparameters
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

A.4
Settings and more results on molecular representation learning task . . . . . . . . .
17

A.5
Settings and more results on conformation generation task
. . . . . . . . . . . . .
19

A.6
Sentivity analysis of k in k-step same-subgraph diffusion . . . . . . . . . . . . . .
21

B
More discussions
22

B.1
More related works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

B.2
Limitations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

C An important lemma for diffusion model
23

D Derivations of training objectives
26

D.1
SubgDiff (1-same step and without expectation state) . . . . . . . . . . . . . . . .
26

D.2
ELBO of SubgDiff
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

D.3
Single-step subgraph diffusion . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

E
Expectation state distribution
33

F
The derivation of SubgDiff
34

A
Experiment details and more results

A.1
Visualization of representations

We conduct an alignment analysis to validate that our method can capture subgraphs during pretraining.
Specifically, we employ t-distributed stochastic neighbor embedding (t-SNE) to visually represent
molecules with various scaffolds. The purpose is to investigate whether molecules sharing the same
scaffold exhibit similar representations, which are extracted by the pretrained molecular encoder.
A scaffold, which is the core structure of a molecule, holds significant importance in the field of
chemistry as it serves as a foundation for systematic exploration of molecular cores and building
blocks. It is usually represented by a substructure of a molecule and can be regarded as the subgraph
in our SubgDiff.

In our analysis, we select the nine most prevalent scaffolds from each dataset (BBBP, Sider, ClinTox,
and Bace) and assign each molecule to a cluster according to its scaffold. To quantify the molecule
embedding, we compute the Silhouette index of the embeddings for each dataset.

As shown in Table 5, SubgDiff enables the generation of more distinctive representations of molecules
with different scaffolds. This implies that SubgDiff enables the denoising network (molecular encoder)
to better capture the subgraph (scaffold) information. We also provide the t-SNE visualizations in
Figure 6.

15


---Page Break---
Silhouette: 0.2632

BBBP

Silhouette: 0.1739

Sider

Silhouette: 0.1780

Clintox

Silhouette: 0.2571

BACE

Figure 6: T-distributed stochastic neighbor embedding (t-SNE) visualization of the learned molecules representa-
tions, colored by the scaffolds of the molecules.

Table 5: Silhouette index (higher is better) of the molecule embeddings on Moleculenet dataset (with 2D
topology only)

BBBP ↑
ToxCast ↑
Sider ↑
ClinTox ↑
Bace ↑

MoleculeSDE
0.2344
0.0611
0.1664
0.1394
0.1860
SubgDiff (ours)
0.2632
0.0650
0.1739
0.1780
0.2571

A.2
Mask distribution

In this paper, we pre-define the mask distribution to be a discrete distribution, with sample space
χ = {Gi
sub}N
i=1, and pt(S = Gi
sub) = 1/N, t > 1, where Gi
sub is the subgraph split by the Torsional-
based decomposition methods [15]. The decomposition approach will cut off one torsional edge in a
3D molecule to make the molecule into two components, each of which contains at least two atoms.
The two components are represented as two complementary mask vectors (i.e. s′ + s = 1). Thus n
torsional edges in Gi
3D will generate 2n subgraphs. Finally, for each atom v, the stv ∼Bern(0.5),
i.e. p = 0.5 in SubgDiff.

A.3
Hyperparameters

All models are trained with SGD using the ADAM optimizer. Pre-training takes around 24 hours
with a single Nvidia A6000 GPU of 48GB RAM. The hyperparameters can be seen in Table 6 and
Table 7.

Table 6: Additional hyperparameters of our SubgDiff.
Task
β1
βT
β scheduler
T
k (k-same mask)
τ
Batch Size
Train Iter.

QM9
1e-7
2e-3
sigmoid
5000
250
10Å
64
2M
Drugs
1e-7
2e-3
sigmoid
5000
250
10Å
32
6M

16


---Page Break---
Table 7: Additional hyperparameters of our SubgDiff with different timesteps.
Task
β1
βT
β scheduler
T
k (k-same mask)
τ
Batch Size
Train Iter.

500-step QM9
1e-7
2e-2
sigmoid
500
25
10Å
64
2M
200-step QM9
1e-7
5e-2
sigmoid
200
10
10Å
64
2M
500-step Drugs
1e-7
2e-2
sigmoid
500
25
10Å
32
4M
1000-step Drugs
1e-7
9e-3
sigmoid
500
50
10Å
32
4M

A.4
Settings and more results on molecular representation learning task

A.4.1
Model architecture

We use the pretraining framework MoleculeSDE proposed by [22] and extend our SubgDiff to multi-
modality pertaining. The two key components of MoleculeSDE are two SDEs(stochastic differential
equations Song et al. [41]): an SDE from 2D topology to 3D conformation (2D →3D) and an SDE
from 3D conformation to 2D topology (3D →2D). In practice, these two SDEs can be replaced by
discrete diffusion models. In this paper, we use the proposed SubgDiff to replace the SDEs.

2D topological molecular graph. A topological molecular graph is denoted as g2D = G(V, E, X),
where X is the atom attribute matrix and X is the bond attribute matrix. The 2D graph representation
with graph neural network (GNN) is:

x ≜H2D = GIN(g2D) = GIN(G, X),
(20)
where GIN is the a powerful 2D graph neural network [48] and H2D = [h0
2D, h1
2D, . . .], where hi
2D is
the i-th node representation.

3D conformational molecular graph. The molecular conformation is denoted as g3D := G3D(G, R).
The conformational representations are obtained by a 3D GNN SchNet [34]:

y ≜H3D = SchNet(g3D) = SchNet(G, R),
(21)
where H3D = [h0
3D, h1
3D, . . .], and hi
3D is the i-th node representation.

An SE(3)-Equivariant Conformation Generation
The first objective is the conditional generation
from topology to conformation, p(y|x), implemented as SubgDiff. The denoising network we
adopt is the SE(3)-equivariance network (S2D→3D
θ
) used in MoleculeSDE. The details of the network
architecture refer to [22].

Therefore, the training objective from 2D topology graph to 3D confirmation is:

L2D→3D = Ex,R,t,stERt|R
hdiag(st)(ϵ −S2D→3D
θ
(x, Rt, t))

2

2 + BCE(st, s2D→3D
ϑ
(x, Rt, t))
i
,
(22)

where s2D→3D
ϑ
(x, Rt, t) gets the invariant feature from Sθ and introduces a mask head (MLP) to read
out the mask prediction.

An SE(3)-Invariant Topology Generation. The second objective is to reconstruct the 2D topology
from 3D conformation, i.e., p(x|y). We also use the SE(3)-invariant score network S3D→2D
θ
proposed
by MoleculeSDE. The details of the network architecture refer to [22]. For modeling S3D→2D
θ
,
it needs to satisfy the SE(3)-invariance symmetry property. The inputs are 3D conformational
representation y, the noised 2D information xt at time t, and time t. The output of S3D→2D
θ
is the
Gaussian noise, as (ϵX, ϵE). The diffused 2D information contains two parts: xt = (Xt, Et). For
node feature X, the training objective is

LX
3D→2D = EX,yEt,stEXt|X
(23)
hdiag(st)(ϵ −S3D→2D
θ
(y, Xt, t))

2

2 + BCE(st, s3D→2D
ϑ
(y, Xt, t))
i

.
(24)

For edge feature E, we define a mask matrix S from mask vector s: Sij = 1 if si = 1 or sj = 1,
otherwise, Sij = 0. Eventually, the ojective can be written as:

LE
3D→2D = EE,yEt,stEEt|E
(25)
hSt ⊙(ϵ −S3D→2D
θ
(y, Et, t))

2

2 + BCE(st, s3D→2D
ϑ
(y, Et, t))
i

,
(26)

17


---Page Break---
Then the score network S3D→2D
θ
is also decomposed into two parts for the atoms and bonds: SXt
θ (xt)
and SEt
θ (xt). Similarly, the mask predictor s3D→2D
ϑ
is also decomposed into two parts for the atoms
and bonds: sXt
ϑ (xt) and sEt
ϑ (xt).

Similar to the topology to conformation generation procedure, the s3D→2D
ϑ
(x, Rt, t) gets the invariant
feature from S3D→2D
θ
and introduces a mask head (MLP) to read out the mask prediction.

Learning. Following MoleculeSDE, we incorporate a contrastive loss called EBM-NCE [21]. EBM-
NCE provides an alternative approach to estimate the mutual information I(X; Y ) and is anticipated
to complement the generative self-supervised learning (SSL) method. As a result, the ultimate
objective is:

Loverall = α1LContrastive + α2L2D→3D + α3(LX
3D→2D + LE
3D→2D),
(27)

where α1, α2, α3 are three coefficient hyperparameters.

A.4.2
Dataset and settings

Dataset. For pretraining, following MoleculeSDE, we use PCQM4Mv2 [11]. It’s a sub-dataset
of PubChemQC [28] with 3.4 million molecules with both the topological graph and geometric
conformations. For finetuning, in addition to QM9 [33], we also include MD17. To be specific,
MD17 comprises eight molecular dynamics simulations focused on small organic molecules. These
datasets were initially presented by Chmiela et al. [5] for the development of energy-conserving force
fields using GDML. Each dataset features the trajectory of an individual molecule, encompassing a
broad spectrum of conformations. The objective is to predict energies and forces for each trajectory
by employing a single model.

Baselines for 3D property prediction
We begin by incorporating three coordinate-MI-unaware
SSL methods: (1) Type Prediction, which aims to predict the atom type of masked atoms; (2) Angle
Prediction, which focuses on predicting the angle among triplet atoms, specifically the bond angle
prediction; (3) 3D InfoGraph, which adopts the contrastive learning paradigm by considering the
node-graph pair from the same molecule geometry as positive and negative otherwise. Next, in
accordance with the work of [23], we include two contrastive baselines: (4) GeoSSL-InfoNCE
[30] and (5) GeoSSL-EBM-NCE [21]. We also incorporate a generative SSL baseline named (6)
GeoSSL-RR (RR for Representation Reconstruction). The above baselines are pre-trained on a subset
of 1M molecules with 3D geometries from Molecule3D [52] and we reuse the results reported by
[23] with SchNet as backbone.

Baselines for 2D topology pretraining. We pick up the most promising ones as follows. Attr-
Mask [12, 20], ContexPred [12], InfoGraph [44], and MolCLR [46].

Baselines for 2D and 3D multi-modality pretraining. We include MoleculeSDE[22](Variance
Exploding (VE) and Variance Preserving (VP)) as a crucial baseline to verify the effectiveness of our
methods due to the same pertaining framework. We reproduce the results from the released Code.

A.4.3
3D molecular property prediction Results on QM9.

By adopting the pertaining setting in Appendix A.4.2, we also take the QM9 dataset for finetuning
and follow the literature [34, 35, 22], using 110K for training, 10K for validation and 11k for testing.
In addition, the QM9 dataset encompasses 12 tasks that pertain to quantum properties, which are
commonly used for evaluating representation learning tasks [34, 23]. The experimental results can be
seen in Table 8. The results also suggest the superior performance of our method.

A.4.4
Compared with GeoDiff.

We directly reuse the pre-trained model of the molecular conformation generation in Section 5.2 for
fine-tuning, to compare our method with GeoDiff from naive denoising pretraining perspective [55].
The results are shown in Table 9.

18


---Page Break---
Table 8: Results on 12 quantum mechanics prediction tasks from QM9. We take 110K for training, 10K for
validation, and 11K for testing. The evaluation is mean absolute error (MAE), and the best and the second best
results are marked in bold and underlined, respectively. The backbone is SchNet.
Pretraining
Alpha ↓
Gap ↓
HOMO↓
LUMO ↓
Mu ↓
Cv ↓
G298 ↓
H298 ↓
R2 ↓
U298 ↓
U0 ↓
Zpve ↓

Random init
0.070
50.59
32.53
26.33
0.029
0.032
14.68
14.85
0.122
14.70
14.44
1.698
Supervised
0.070
51.34
32.62
27.61
0.030
0.032
14.08
14.09
0.141
14.13
13.25
1.727
Type Prediction
0.084
56.07
34.55
30.65
0.040
0.034
18.79
19.39
0.201
19.29
18.86
2.001
Angle Prediction
0.084
57.01
37.51
30.92
0.037
0.034
15.81
15.89
0.149
16.41
15.76
1.850
3D InfoGraph
0.076
53.33
33.92
28.55
0.030
0.032
15.97
16.28
0.117
16.17
15.96
1.666
GeossL-RR
0.073
52.57
34.44
28.41
0.033
0.038
15.74
16.11
0.194
15.58
14.76
1.804
GeossL-InfoNCE
0.075
53.00
34.29
27.03
0.029
0.033
15.67
15.53
0.125
15.79
14.94
1.675
GeossL-EBM-NCE
0.073
52.86
33.74
28.07
0.031
0.032
14.02
13.65
0.121
13.70
13.45
1.677
MoleculeSDE
0.062
47.74
28.02
24.60
0.028
0.029
13.25
12.70
0.120
12.68
12.93
1.643

Ours
0.054
44.88
25.45
23.75
0.027
0.028
12.03
11.46
0.110
11.32
11.25
1.568

Table 9: Results on 12 quantum mechanics prediction tasks from QM9. We take 110K for training, 10K for
validation, and 11K for testing. The evaluation is mean absolute error (MAE), and the best and the second best
results are marked in bold and underlined, respectively. The backbone is SchNet.

Pretraining
Alpha ↓
Gap ↓
HOMO↓
LUMO ↓
Mu ↓
Cv ↓
G298 ↓
H298 ↓
R2 ↓
U298 ↓
U0 ↓
Zpve ↓

GeoDiff
0.078
51.84
30.88
28.29
0.028
0.035
15.35
11.37
0.132
15.76
15.24
1.869

SubgDiff (ours)
0.076▲
50.80▲
31.15▼
26.62▲
0.025▲
0.032▲
14.92▲
12.86▲
0.129▲
14.74▲
14.53▲
1.710▲

A.5
Settings and more results on conformation generation task

A.5.1
Dataset and network.

Following prior works [51], we utilize the GEOM-QM9 [33] and GEOM-Drugs [2] datasets. The
former dataset comprises small molecules of up to 9 heavy atoms, while the latter contains larger drug-
like compounds. We reuse the data split provided by Xu et al. [51]. For both datasets, the training
dataset comprises 40, 000 molecules, each with 5 conformations, resulting in 200, 000 conformations
in total. The test split includes 200 distinctive molecules, with 14, 324 conformations for Drugs and
22, 408 conformations for QM9.

We adopt the graph field network (GFN) from [51] as the GNN encoder for extracting the 3D
molecular information. In the l-th layer, the GFN receives node embeddings hl ∈Rn×b (where b
represents the feature dimension) and corresponding coordinate embeddings xl ∈Rn×3 as input. It
then produces the output hl+1 and xl+1 according to the following process:

ml
ij = Φl
m
 
hl
i, hl
j, ∥xl
i −xl
j∥2, eij; θm

(28)

hl+1
i
= Φl
h

hl
i,
X

j∈N(i)
ml
ij; θh

(29)

xl+1
i
=
X

j∈N(i)

1
dij
(Ri −Rj) Φl
x
 
ml
ij; θx

(30)

where Φ are implemented as feed-forward networks and dij denotes interatomic distances. The initial
embedding h0 is composed of atom embedding and time step embedding while x0 represents atomic
coordinates. N(i) is the neighborhood of ith node, consisting of connected atoms and other ones
within a radius threshold τ, helping the model capture long-range interactions explicitly and support
disconnected molecular graphs.

Eventually, the Gaussian noise and mask can be predicted as follows (C.f. Figure 7):

ˆϵi = xL
i
(31)

ˆsi = MLP(hL
i )
(32)

where ˆϵi is equivalent and ˆsi is invariant.

Settings. For GeoDiff [51] with 5000 steps, we use the checkpoints released in public GitHub to
reproduce the results. For 200 and 500 steps, we retrain it and do the DDPM sampling.

A.5.2
Evaluation metrics for conformation generation.

To compare the generated and ground truth conformer ensembles, we employ the same evaluation
metrics as in a prior study [7]: Average Minimum RMSD (AMR) and Coverage. These metrics

19


---Page Break---
3D Molecules

Adding

noise

GNN Encoder
Ƹ𝑠𝑡: Mask prediction;  Ƹ𝜖𝑡: Noise prediction 
𝐑0

Ƹ𝑠𝑡

Ƹ𝜖𝑡
Noise head

Mask head

𝐑𝑡

ℒ1

ℒ2

ℒ1 = BCE(𝐬𝑡, Ƹ𝑠𝑡);
ℒ2 =
diag 𝐬𝑡
𝜖−Ƹ𝜖𝑡
Ƹ𝑠𝑡= 𝑠ϑ 𝒢, 𝐑𝑡, 𝑡;
Ƹ𝜖𝑡= 𝜖𝜃𝒢, 𝐑𝑡, 𝑡;

Figure 7: The model architecture for denoising SubgDiff.

enable us to assess the quality of the generated conformers from two perspectives: Recall (R) and
Precision (P). Recall measures the extent to which the generated ensemble covers the ground-truth
ensemble, while Precision evaluates the accuracy of the generated conformers.

The four metrics built upon root-mean-square deviation (RMSD), which is defined as the normalized
Frobenius norm of two atomic coordinates matrices, after alignment by Kabsch algorithm [16].
Formally, let Sg and Sr denote the sets of generated and reference conformers respectively, then the
Coverage and Matching metrics [49] can be defined as:

COV-R(Sg, Sr) =
1
|Sr|


n
C ∈Sr| RMSD(C, ˆC) ≤δ, ˆC ∈Sg
o,
(33)

MAT-R(Sg, Sr) =
1
|Sr|

X

C∈Sr
min
ˆC∈Sg
RMSD(C, ˆC),
(34)

where δ is a threshold. The other two metrics COV-P and MAT-P can be defined similarly but with
the generated sets Sg and reference sets Sr exchanged.

COV-P(Sr, Sg) =
1
|Sg|


n
ˆC ∈Sg| RMSD(C, ˆC) ≤δ, C ∈Sr
o,
(35)

MAT-P(Sr, Sg) =
1
|Sg|

X

ˆC∈Sg
min
C∈Sr RMSD(C, ˆC),
(36)

In practice, Sg is set as twice of the size of Sr for each molecule.

A.5.3
Comparison with GeoDiff using Langevin Dynamics sampling method.

In order to verify that our proposed diffusion process can bring benefits to other sampling methods,
we conduct the experiments to compare our proposed diffusion model with GeoDiff by adopting a
typical sampling method Langevin dynamics (LD sampling)[40] :

Rt−1 = Rt + αtϵθ(G, Rt, t) +
√

2αtzt−1
(37)

where zt ∼N(0, I) and hσ2
t . h is the hyper-parameter referring to step size and σt is the noise
schedule in the forward process. We use various time-step to evaluate the generalization and
robustness of the proposed method, and the results shown in Table 10 indicate that our method
significantly outperforms GeoDiff, especially when the time-step is relatively small (200,500), which
implies that our training method can effectively improve the efficiency of denoising. Similar results
are also observed on the GEOM-drugs dataset (Table 11).

Table 10: Results on GEOM-QM9 dataset with different time steps. Langevin dynamics [40] is a typical
sampling method used in DPM. ▲denotes SubgDiff outperforms GeoDiff. The threshold δ = 0.5Å.

COV-R (%) ↑
MAT-R (Å) ↓
COV-P (%) ↑
MAT-P (Å) ↓
Steps
Sampling method
Models
Mean
Median
Mean
Median
Mean
Median
Mean
Median

500
Langevin dynamics
GeoDiff
87.80
93.66
0.3179
0.3216
46.25
45.02
0.6173
0.5112
500
Langevin dynamics
SubgDiff
91.40▲
95.39▲
0.2543▲
0.2601▲
51.71▲
48.50▲
0.5035▲
0.4734▲

200
Langevin dynamics
GeoDiff
86.60
93.09
0.3532
0.3574
42.98
42.60
0.5563
0.5367
200
Langevin dynamics
SubgDiff
90.36▲
95.93▲
0.3064▲
0.3098▲
48.56▲
46.46▲
0.5540▲
0.5082▲
.

20


---Page Break---
Table 11: Results on the GEOM-Drugs dataset under different diffusion timesteps. DDPM [9] is the sam-
pling method used in GeoDiff and Langevin dynamics [40] is a typical sampling method used in DPM. Our
proposed sampling method (Algorithm 2) can be viewed as a DDPM variant. ▲/▼denotes SubgDiff outper-
forms/underperforms GeoDiff. The threshold δ = 1.25Å.

COV-R (%) ↑
MAT-R (Å) ↓
Models
Timesteps
Sampling method
Mean
Median
Mean
Median

GeoDiff
500
DDPM
50.25
48.18
1.3101
1.2967
SubgDiff
500
DDPM (ours)
76.16▲
86.43▲
1.0463▲
1.0264▲

GeoDiff
500
LD
64.12
75.56
1.1444
1.1246
SubgDiff
500
LD (ours)
74.30▲
77.87▲
1.0003▲
0.9905▲

Table 12: Results on GEOM-QM9 dataset. The threshold δ = 0.5Å.

COV-R (%) ↑
MAT-R (Å) ↓
COV-P (%) ↑
MAT-P (Å) ↓
Models
Mean
Median
Mean
Median
Mean
Median
Mean
Median

CVGAE
0.09
0.00
1.6713
1.6088
-
-
-
-
GraphDG
73.33
84.21
0.4245
0.3973
43.90
35.33
0.5809
0.5823
CGCF
78.05
82.48
0.4219
0.3900
36.49
33.57
0.6615
0.6427
ConfVAE
77.84
88.20
0.4154
0.3739
38.02
34.67
0.6215
0.6091
GeoMol
71.26
72.00
0.3731
0.3731
-
-
-
-
ConfGF
88.49
94.31
0.2673
0.2685
46.43
43.41
0.5224
0.5124
GeoDiff
80.36
83.82
0.2820
0.2799
53.66
50.85
0.6673
0.4214

SubgDiff
90.91
95.59
0.2460
0.2351
50.16
48.01
0.6114
0.4791

A.5.4
Comparison with SOTAs.

i) Baselines: We compare SubgDiff with 7 state-of-the-art baselines: CVGAE [25], GraphDG [38],
CGCF [49], ConfVAE [50], ConfGF [37] and GeoDiff [51]. For the above baselines, we reuse the
experimental results reported by [51]. For GeoDiff [51], we use the checkpoints released in public
GitHub to reproduce the results. ii) Results: The results on the GEOM-QM9 dataset are reported in
Table 12. From the results, we get the following observation: SubgDiff significantly outperforms the
baselines on COV-R, indicating the SubgDiff tends to explore more possible conformations. This
implicitly demonstrates the subgraph will help fine-tune the generated conformation to be a potential
conformation.

A.5.5
Domain generelizaion

The results of Training on QM9 (small molecular with up to 9 heavy atoms) and testing on Drugs
(medium-sized organic compounds) can be found in table 13.

Table 13: Results on the GEOM-Drugs dataset. The threshold δ = 1.25Å

Train
COV-R (%) ↑
MAT-R (Å) ↓
Models
data
Mean
Median
Mean
Median

CVGAE
Drugs
0.00
0.00
3.0702
2.9937
GraphDG
Drugs
8.27
0.00
1.9722
1.9845
GeoDiff
QM9
7.99
0.00
2.7704
2.3297

SubgDiff
QM9
24.01
9.93
1.6128
1.5819

A.6
Sentivity analysis of k in k-step same-subgraph diffusion

The results from Table 14 indicate that k = 25 is more likely to give the best performance when the
diffusion step N is 500. From our experience, k and the number of the diffusion step N can maintain
a certain ratio, e.g. N : k = 20 in our experiments, and the model can achieve better performance.

21


---Page Break---
Table 14: The sensitivity analysis for different k in k-step same subgraph diffusion on conformation generation

k
1ld sampling∗
10
25
50

COV-R(Mean) ↑
89.70
88.06
89.78
89.02
COV-R(Median) ↑
93.96
93.26
94.17
93.21
COV-P(Mean) (%) ↑
49.90
47.38
50.03
48.63
COV-P(Median) (%) ↑
47.00
47.06
48.31
46.77
MAT-R(Mean) (A) ↓
0.5235
0.2623
0.2417
0.2706
MAT-R(Median) (A) ↓
0.2710
0.2597
0.2449
0.2709
MAT-P(Mean) (A) ↓
4.7816
4.2922
0.5571
0.7512
MAT-P(Median) (A) ↓
0.5378
0.5615
0.4921
0.4995

∗The subgraph predictor cannot predict the correct subgraphs so we use Langevin Dynamics sampling rather than DDPM when k = 1

B
More discussions

B.1
More related works

Masks on diffusion models.
Previous works also share a similar idea of subgraph (mask) diffusion,
such as MDM [31], MDSM [18] and SSSD [1]. However, the difference between our SubgDiff
and theirs mainly lies in the following two aspects: i) Usage: the mask matrix/vector in SSSD and
MDSM is fixed in all training steps, which means some segments of the data (time series or images)
will never be diffused. But our method samples the st ∼pst(S) at each time step, hence a suitable
discrete distribution p(S) can ensure that almost all nodes can be added noise. ii) Purpose: MDSM
and MDM concentrate on self-supervised pre-training, while SubgDiff serves as a potent generative
model and self-supervised pre-training algorithm. Notably, when st = s0, ∀t, SubgDiff can recover
to MDSM.

Graph generation models.
D3FG [19]: D3FG adopts three different diffusion models (D3PM,
DDPM, and SO(3) Diffusion) to generate three different parts of molecules(linkerr types, center
atom position, and functional group orientations), respectively. In general, these three parts can also
be viewed as three subgraphs(subset). DiffPACK[59] is an Autoregressive generative method that
predicts the torsional angle χi(i = 1, 2, .., 4) of protein side-chains with the condition χ1,...,i−1,
where χi is a predefined subset of atoms. It uses a torsional-based diffusion model to approximate
the distribution p(χi|χ1,...,i−1), in which every subset χi needs a separate score network to estimate.
Essentially, both D3FG and DiffPACK can be viewed as selecting a subset first and then only adding
noise on the fixed subset during the entire diffusion process. In contrast, our method proposes to
randomly sample a subset from mask distribution p(S) in each time-step during the forward process.
[17] proposes an autoregressive diffusion model named GraphARM, which absorbs one node in each
time step by masking it along with its connecting edges during the forward process. Differently from
GraphARM, our SubgDiff selects a subgraph in each time step to inject the Gaussian noise, which is
equivalent to masking several nodes during the forward process. In addition, the number of steps in
GraphARM must be the same as the number of nodes due to the usage of the absorbing state, while
our method can set any time-step during diffusion theoretically since we use the real-value Gaussian
noise. Concurrently, SubDiff [53] is proposed to use subgraphs as minimum units to train a latent
diffusion model, while our method directly involves the subgraph during the forward process, which
is a new type of diffusion model.

B.2
Limitations

One limitation of our method is that it does not currently align with the Boltzmann distribution
over the conformations and it cannot accurately estimate the likelihood of a sampled conformation.
Currently, we generate the subgraph by dividing the molecule with the rotatable bonds. More
chemical prior knowledge can be used to guide the mask of the subgraph. Additionally, there is a
requirement for the number of steps in sampling to be the same as that in training, which can be
improved with the recent progress of diffusion models such as DDIM.

22


---Page Break---
C
An important lemma for diffusion model

According to [39, 9], the diffusion model is trained by optimizing the variational bound
on the negative log-likelihood −log pθ(R0),
in which the tricky terms are Lt−1
=
DKL(q(Rt−1|Rt, R0)||pθ(Rt−1|Rt))), T ≥t > 1. Here we provide a lemma that tells us the
posterior distribution q(Rt−1|Rt, R0) used in the training and sampling algorithms of the diffusion
model can be determined by q(Rt|Rt−1, R0), q(Rt−1|R0). Formally, we have

Lemma C.1 Assume the forward and reverse processes of the diffusion model are both Markov
chains.
Given the forward Gaussian distribution q(Rt|Rt−1, R0)
=
N(Rt; µ1Rt−1, σ2
1I),
q(Rt−1|R0) = N(Rt−1; µ2R0, σ2
2I) and ϵ0 ∼N(0, I), the distribution q(Rt−1|Rt, R0) is

q(Rt−1|Rt, R0) ∝N(Rt−1; 1

µ1
(Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵ0),
σ2
1σ2
2
µ2
1σ2
2 + σ2
1
I).

Parameterizing pθ(Rt−1|Rt) in the reverse process as

N(Rt−1; 1

µ1
(Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)),
σ2
1σ2
2
µ2
1σ2
2 + σ2
1
I),

the training objective of the DPM can be written as

L(θ) = Et,R0,ϵ
h
σ2
1
2µ2
1σ2
2
∥ϵ −ϵθ(µ1µ2R0 +
q

µ2
1σ2
2 + σ2
1ϵ, t)∥2i

,

and the sampling process is

Rt−1 = 1

µ1

 

Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)

!

+
σ1σ2
p

µ2
1σ2
2 + σ2
1
z,
(38)

where z ∼N(0, I).

Once we get the variables (µ1, σ1, µ2, σ2), we can directly obtain the training objective and sampling
process via lemma C.1, which will help the design of new diffusion models.

Proof:
Given the forward Gaussian distribution q(Rt|Rt−1, R0) = N(Rt; µ1Rt−1, σ2
1I) and
q(Rt−1|R0) = N(Rt−1; µ2R0, σ2
2I), we have

q(Rt|R0) = q(Rt|Rt−1, R0)q(Rt−1|R0) = N(Rt; µ1µ2R0, (σ2
1 + µ2
1σ2
2)I)
(39)

From the DDPM, we know training a diffusion model should optimize the ELBO of the data

log p(R) ≥Eq(R1:T |R0)


log
p(R0:T )
q(R1:T |R0)



(40)

= Eq(R1|R0)

log pθ(R0|R1)


|
{z
}
reconstruction term

−DKL(q(RT |R0) ∥p(RT ))
|
{z
}
prior matching term

−

T
X

t=2
Eq(Rt|R0)

DKL(q(Rt−1|Rt, R0) ∥pθ(Rt−1|Rt))


|
{z
}
denoising matching term

(41)

23


---Page Break---
To compute the KL divergence DKL(q(Rt−1|Rt, R0)
∥
pθ(Rt−1|Rt)), we first rewrite
q(Rt−1|Rt, R0) by Bayes rule

q(Rt−1|Rt, R0) = q(Rt|Rt−1, R0)q(Rt−1|R0)

q(Rt|R0)
(42)

= N(Rt; µ1Rt−1, σ2
1I)N(Rt−1; µ2R0, σ2
2I)
N(Rt; µ1µ2R0, (σ2
1 + µ2
1σ2
2)I)
(43)

∝exp

−
(Rt −µ1Rt−1)2

2σ2
1
+ (Rt−1 −µ2R0)2

2σ2
2
−(Rt −µ1µ2R0)2

2(σ2
1 + µ2
1σ2
2)



(44)

= exp

−1

2

(Rt −µ1Rt−1)2

σ2
1
+ (Rt−1 −µ2R0)2

σ2
2
−(Rt −µ1µ2R0)2

σ2
1 + µ2
1σ2
2



(45)

= exp

−1

2

(−2µ1RtRt−1 + µ2
1(Rt−1)2)
σ2
1
+ ((Rt−1)2 −2µ2Rt−1R0)

σ2
2
+ C(Rt, R0)


(46)

∝exp

−1

2


−2µ1RtRt−1

σ2
1
+ µ2
1(Rt−1)2

σ2
1
+ (Rt−1)2

σ2
2
−2µ2Rt−1R0

σ2
2



(47)

= exp

−1

2


(µ2
1
σ2
1
+ 1

σ2
2
)(Rt−1)2 −2
µ1Rt

σ2
1
+ µ2R0

σ2
2


Rt−1


(48)

= exp

−1

2

σ2
1 + µ2
1σ2
2
σ2
1σ2
2
(Rt−1)2 −2
µ1Rt

σ2
1
+ µ2R0

σ2
2


Rt−1


(49)

= exp




−1

2

σ2
1 + µ2
1σ2
2
σ2
1σ2
2

 

(Rt−1)2 −2


µ1Rt

σ2
1
+ µ2R0

σ2
2



σ2
1+µ2
1σ2
2
σ2
1σ2
2

Rt−1









(50)

= exp




−1

2

σ2
1 + µ2
1σ2
2
σ2
1σ2
2

 

(Rt−1)2 −2


µ1Rt

σ2
1
+ µ2R0

σ2
2


σ2
1σ2
2
σ2
1 + µ2
1σ2
2
Rt−1









(51)

= exp




−1

2




1

σ2
1σ2
2
σ2
1+µ2
1σ2
2





(Rt−1)2 −2µ1σ2
2Rt + µ2σ2
1R0

σ2
1 + µ2
1σ2
2
Rt−1




(52)

∝N(Rt−1; µ1σ2
2Rt + µ2σ2
1R0

σ2
1 + µ2
1σ2
2
|
{z
}
µq(Rt,R0)

,
σ2
1σ2
2
σ2
1 + µ2
1σ2
2
I
|
{z
}
Σq(t)

)
(53)

We can rewrite our variance equation as Σq(t) = σ2
q(t)I, where:

σ2
q(t) =
σ2
1σ2
2
σ2
1 + µ2
1σ2
2
(54)

From (39), we have the relationship between Rt and R0:

R0 = Rt −
p

σ2
1 + µ2
1σ2
2ϵ
µ1µ2
(55)

Substituting this into µq(Rt, R0), we can get

µq(Rt, R0) = µ1σ2
2Rt + µ2σ2
1R0

σ2
1 + µ2
1σ2
2
(56)

=
µ1σ2
2Rt + µ2σ2
1
Rt−√

σ2
1+µ2
1σ2
2ϵ
µ1µ2
σ2
1 + µ2
1σ2
2
(57)

=
µ1σ2
2Rt + σ2
1R2

µ1
−
σ2
1
√

σ2
1+µ2
1σ2
2ϵ
µ1
σ2
1 + µ2
1σ2
2
(58)

= 1

µ1
Rt −
σ2
1
µ1
p

σ2
1 + µ2
1σ2
2
ϵ
(59)

24


---Page Break---
Thus,

q(Rt−1|Rt, R0) ∝N(Rt−1; 1

µ1
(Rt −
σ2
1
p

σ2
1 + µ2
1σ2
2
ϵ)
|
{z
}
µq(Rt,t)

,
σ2
1σ2
2
σ2
1 + µ2
1σ2
2
I
|
{z
}
Σq(t)

)
(60)

Parameterizing
pθ(Rt−1|Rt)
in
the
reverse
process
as
N(Rt−1; 1

µ1 (Rt
−

σ2
1
√

µ2
1σ2
2+σ2
1 ϵθ(Rt, t)),
σ2
1σ2
2
µ2
1σ2
2+σ2
1 I) , and the corresponding optimization problem becomes:

arg min
θ
DKL(q(Rt−1|Rt, R0) ∥pθ(Rt−1|Rt))

= arg min
θ
DKL(N
 
Rt−1; µq, Σq (t)

∥N
 
Rt−1; µθ, Σq (t)

)
(61)

= arg min
θ

1
2σ2q(t)






σ2
1
µ1
p

σ2
1 + µ2
1σ2
2
ϵ0 −
σ2
1
µ1
p

σ2
1 + µ2
1σ2
2
ϵθ(Rt, t)



2

2




(62)

= arg min
θ

1
2σ2q(t)






σ2
1
µ1
p

σ2
1 + µ2
1σ2
2
(ϵ0 −ˆϵθ(Rt, t))



2

2




(63)

= arg min
θ

1
2σ2q(t)

 
σ2
1
µ1
p

σ2
1 + µ2
1σ2
2

!2 hϵ0 −ˆϵθ(Rt, t)
2
2

i
(64)

= arg min
θ

σ2
1
2σ2
2µ2
1

hϵ0 −ˆϵθ(Rt, t)
2
2

i
(65)

Therefore, the training objective of the DPM can be written as

L(θ) = Et,R0,ϵ[
σ2
1
2µ2
1σ2
2
∥ϵ −ϵθ(µ1µ2R0 +
q

µ2
1σ2
2 + σ2
1ϵ, t)∥2],
(66)

During the reverse process, we sample Rt−1 ∼pθ(Rt−1|Rt). Formally, the sampling (reverse)
process is

Rt−1 = 1

µ1

 

Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)

!

+
σ1σ2
p

µ2
1σ2
2 + σ2
1
z,
z ∼N(0, I)
(67)

25


---Page Break---
This lemma can be easily extended to the conditional version:

Lemma C.2 Assume the forward and reverse processes of the diffusion model are both Markov
chains. {Rt}T
t=0 are the states and y1, y2 are the given conditions. Given the forward Gaussian
distribution
q(Rt|Rt−1, R0, y1) = N(Rt; µ1Rt−1, σ2
1I);

q(Rt−1|R0, y2) = N(Rt−1; µ2R0, σ2
2I)
and ϵ0 ∼N(0, I), we have the distribution q(Rt|R0, y1, y2) = N(Rt; µ1µ2R0, (σ2
1 + µ2
1σ2
2)I).
Thus, the posterior distribution q(Rt−1|Rt, R0) is

q(Rt−1|Rt, R0) ∝N(Rt−1; 1

µ1
(Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵ0),
σ2
1σ2
2
µ2
1σ2
2 + σ2
1
I).

Parameterizing pθ(Rt−1|Rt, y1, y2) in the reverse process as

N(Rt−1; 1

µ1
(Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)),
σ2
1σ2
2
µ2
1σ2
2 + σ2
1
I),

the training objective of the DPM can be written as

L(θ) = Et,R0,ϵ
h
σ2
1
2µ2
1σ2
2
∥ϵ −ϵθ(µ1µ2R0 +
q

µ2
1σ2
2 + σ2
1ϵ, t)∥2i

,

and the sampling process is

Rt−1 = 1

µ1

 

Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)

!

+
σ1σ2
p

µ2
1σ2
2 + σ2
1
z,
(68)

where z ∼N(0, I).

D
Derivations of training objectives

D.1
SubgDiff (1-same step and without expectation state)

Here, we utilize the binary characteristic of the mask vector to derive the ELBO for SubgDiff, and
we also provide a general proof in Section D.2:

log p(R0) ≥Eq(R1:T ,s1:T |R0)


log
p(R0:T , s1:T )
q(R1:T |R0, s1:T )q(s1:T )



(69)

= Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1, st|Rt)
QT
t=1 q(Rt|Rt−1, st)q(st)

#

(70)

= Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt)pθ(st|Rt)
QT
t=1 q(Rt|Rt−1, st)q(st)

#

(71)

= Eq(R1:T ,s1:T |R0)

"

log
QT
t=1 pθ(st|Rt)
QT
t=1 q(st)
+ log p(RT ) QT
t=1 pθ(Rt−1|Rt)
QT
t=1 q(Rt|Rt−1, st)

#

(72)

= Eq(R1:T ,s1:T |R0)

" T
X

t=1
log pθ(st|Rt)

q(st)

#

|
{z
}
mask prediction term

+Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt)
QT
t=1 q(Rt|Rt−1, st)

#

(73)
(74)
The first term is mask prediction while the second term is similar to the ELBO of the classical
diffusion model. The only difference is the st in q(Rt|Rt−1, st). According to Bayes rule, we can
rewrite each transition as:

q(Rt|Rt−1, R0, st) =

(
q(Rt−1|Rt,R0)q(Rt|R0)

q(Rt−1|R0)
,
if st = 1
δRt−1(Rt).
if st = 0
(75)

26


---Page Break---
where δa(x) := δ(x −a) is Dirac delta function, that is, δa(x) = 0 if x ̸= a and
R ∞
−∞δa(x)dx = 1.
Without loss of generality, assume that s1 and sT both equal 1. Armed with this new equation, we
drive the second term:

Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt)
QT
t=1 q(Rt|Rt−1, st)

#

(76)

= Eq(R1:T ,s1:T |R0)

"

log p(RT )pθ(R0|R1) QT
t=2 pθ(Rt−1|Rt)

q(R1|R0) QT
t=2 q(Rt|Rt−1, st)

#

(77)

= Eq(R1:T ,s1:T |R0)

"

log p(RT )pθ(R0|R1) QT
t=2 pθ(Rt−1|Rt)

q(R1|R0) QT
t=2 q(Rt|Rt−1, R0, st)

#

(78)

= Eq(R1:T ,s1:T |R0)

"

log pθ(RT )pθ(R0|R1)

q(R1|R0)
+ log

T
Y

t=2

pθ(Rt−1|Rt)
q(Rt|Rt−1, R0, st)

#

(79)

= Eq(R1:T ,s1:T |R0)



log p(RT )pθ(R0|R1)

q(R1|R0)
+ log
Y

t∈{t|st=1}

pθ(Rt−1|Rt)

q(Rt−1|Rt,R0)q(Rt|R0)

q(Rt−1|R0,s1)
+ log
Y

t∈{t|st=0}

pθ(Rt−1|Rt)

δRt−1(Rt)





(80)

= Eq(R1:T |R0)



log p(RT )pθ(R0|R1)

q(R1|R0)
+ log
Y

t∈{t|st=0}

pθ(Rt−1|Rt)

δRt−1(Rt)
+ log
Y

t∈{t|st=1}

pθ(Rt−1|Rt)

q(Rt−1|Rt,R0)
q(Rt|R0)
(((((
q(Rt−1|R0)





(81)

= Eq(R1:T |R0)



log
Y

t∈{t|st=0}

pθ(Rt−1|Rt)

δRt−1(Rt)
+ log p(RT )pθ(R0|R1)



q(R1|R0)
+ log 

q(R1|R0)
q(RT |R0) + log
Y

t∈{t|st=1}

pθ(Rt−1|Rt)
q(Rt−1|Rt, R0)





(82)

= Eq(R1:T |R0)




X

t∈{t|st=0}
log pθ(Rt−1|Rt)

δRt−1(Rt)
+ log p(RT )pθ(R0|R1)

q(RT |R0)
+
X

t∈{t|st=1}
log
pθ(Rt−1|Rt)
q(Rt−1|Rt, R0)





(83)

=

X

t∈{t|st=0}
Eq(R1:T |R0)


log pθ(Rt−1|Rt)

δRt−1(Rt)


+ Eq(R1:T |R0)

log pθ(R0|R1)

(84)

+Eq(R1:T |R0)


log
p(RT )
q(RT |R0)


+
X

t∈{t|st=1}
Eq(R1:T |R0)


log
pθ(Rt−1|Rt)
q(Rt−1|Rt, R0)



(85)

=

X

t∈{t|st=0}
Eq(R1:T |R0)


log pθ(Rt−1|Rt)

δRt−1(Rt)


+ Eq(R1|R0)

log pθ(R0|R1)

(86)

+Eq(RT |R0)


log
p(RT )
q(RT |R0)


+
X

t∈{t|st=1}
Eq(Rt,Rt−1|R0)


log
pθ(Rt−1|Rt)
q(Rt−1|Rt, R0)



(87)

=

X

t∈{t|st=0}
Eq(R1:T |R0)


log pθ(Rt−1|Rt)

δRt−1(Rt)



|
{z
}
decay term

+ Eq(R1|R0)

log pθ(R0|R1)


|
{z
}
reconstruction term

(88)

−DKL(q(RT |R0) ∥p(RT ))
|
{z
}
prior matching term

−
X

t∈{t|st=1}
Eq(Rt|R0)

DKL(q(Rt−1|Rt, R0) ∥pθ(Rt−1|Rt))


|
{z
}
denoising matching term

(89)

27


---Page Break---
Here, the decay term represents the terms with st = 0, which are unnecessary to minimize when we
set pθ(Rt−1|Rt) := δRt−1(Rt). Eventually, the ELOB can be rewritten as follows:

log p(R0) ≥

T
X

t=1
Eq(R1:T |R0)


log pϑ(st|Rt)

q(st)



|
{z
}
mask prediction term

+ Eq(R1|R0)

log pθ(R0|R1)


|
{z
}
reconstruction term

−DKL(q(RT |R0) ∥p(RT ))
|
{z
}
prior matching term

−
X

t∈{t|st=1}
Eq(Rt|R0)

DKL(q(Rt−1|Rt, R0) ∥pθ(Rt−1|Rt))


|
{z
}
denoising matching term

(90)

The mask prediction term can be implemented by a node classifier and the denoising matching term
can be calculated via Lemma C.1. In detail,

q(Rt|Rt−1, R0) = N(Rt−1,
p

1 −βtstRt−1, (βtst)I),
(91)

q(Rt−1|R0) = N(Rt−1, √¯γt−1R0, (1 −¯γt−1)I).
(92)

Thus, the training objective of SubgDiff is:

L(θ, ϑ) = Et,R0,ϵ


stβt
2(1 −stβt)(1 −¯γt−1)∥ϵ −ϵθ(√¯γtR0 +
p

(1 −¯γt)ϵ, t, G)∥2 + λBCE(st, sϑ(G, Rt, t))


(93)

To recover the existing work, we omit the mask prediction term (i.e. Let pθ(st|Rt) := q(st)) of
SubgDiff in the main text.

D.2
ELBO of SubgDiff

Here, we can derive the ELBO for SubgDiff:

log p(R0) = log
Z Z
p(R0:T , s1:T )dR1:Tds1:T
(94)

= log
Z Z p(R0:T , s1:T )q(R1:T , s1:T |R0)

q(R1:T , s1:T |R0)
dR1:Tds1:T
(95)

= log
Z Z p(R0:T , s1:T )q(R1:T |R0, s1:T )q(s1:T )

q(R1:T , s1:T |R0)


dR1:T ds1:T
(96)

= log Eq(s1:T )Eq(R1:T |R0,s1:T )

 p(R0:T , s1:T ))

q(R1:T , s1:T |R0)



(97)

≥Eq(R1:T |R0,s1:T )


log Eq(s1:T )
p(R0:T , s1:T )
q(R1:T |R0, s1:T )q(s1:T )



(98)

≥Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1, st|Rt)
QT
t=1 q(Rt|Rt−1, st)q(st)

#

(99)

= Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt)pθ(st|Rt)
QT
t=1 q(Rt|Rt−1, st)q(st)

#

(100)

= Eq(R1:T ,s1:T |R0)

"

log
QT
t=1 pθ(st|Rt)
QT
t=1 q(st)
+ log p(RT ) QT
t=1 pθ(Rt−1|Rt, st)
QT
t=1 q(Rt|Rt−1, st)

#

(101)

= Eq(R1:T ,s1:T |R0)

" T
X

t=1
log pθ(st|Rt)

q(st)

#

|
{z
}
mask prediction term

+Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt, st)
QT
t=1 q(Rt|Rt−1, st)

#

(102)
(103)

According to Bayes rule, we can rewrite each transition as:

q(Rt|Rt−1, R0, s1:t) = q(Rt−1|Rt, R0, s1:t)q(Rt|R0, s1:t)

q(Rt−1|R0, s1:t−1)
,
(104)

28


---Page Break---
Armed with this new equation, we drive the second term:

Eq(R1:T ,s1:T |R0)

"

log p(RT ) QT
t=1 pθ(Rt−1|Rt, st)
QT
t=1 q(Rt|Rt−1, st)

#

(105)

= Eq(R1:T ,s1:T |R0)

"

log p(RT )pθ(R0|R1, s1) QT
t=2 pθ(Rt−1|Rt, st)

q(R1|R0, s1) QT
t=2 q(Rt|Rt−1, st)

#

(106)

= Eq(R1:T ,s1:T |R0)

"

log p(RT )pθ(R0|R1, s1) QT
t=2 pθ(Rt−1|Rt, st)

q(R1|R0, s1) QT
t=2 q(Rt|Rt−1, R0, s1:t)

#

(107)

= Eq(R1:T ,s1:T |R0)

"

log p(RT )pθ(R0|R1, s1)

q(R1|R0, s1)
+ log

T
Y

t=2

pθ(Rt−1|Rt, st)
q(Rt|Rt−1, R0, s1:t)

#

(108)

= Eq(R1:T ,s1:T |R0)



log p(RT )pθ(R0|R1, s1)

q(R1|R0, s1)
+ log

T
Y

t=2

pθ(Rt−1|Rt, st)

q(Rt−1|Rt,R0,s1:t)q(Rt|R0,s1:t)

q(Rt−1|R0,s1:t−1)




(109)

= Eq(R1:T ,s1:t|R0)



log p(RT )pθ(R0|R1, s1)

q(R1|R0, s1)
+ log

T
Y

t=2

pθ(Rt−1|Rt, st)

q(Rt−1|Rt,R0,s1:t)(((((
q(Rt|R0,s1:t)
(((((((
q(Rt−1|R0,s1:t−1)




(110)

= Eq(R1:T ,s1:t|R0)

"

log p(RT )pθ(R0|R1, s1)

((((((
q(R1|R0, s1)
+ log ((((((
q(R1|R0, s1)
q(RT |R0, s1:T ) + log

T
Y

t=2

pθ(Rt−1|Rt, st)
q(Rt−1|Rt, R0, s1:t)

#

(111)

= Eq(R1:T ,s1:t|R0)

"

log p(RT )pθ(R0|R1, s1)

q(RT |R0, s1:T )
+

T
X

t=2
log
pθ(Rt−1|Rt, st)
q(Rt−1|Rt, R0, s1:t)

#

(112)

= Eq(R1:T ,s1:t|R0)

log pθ(R0|R1, s1)

(113)

+Eq(R1:T ,s1:t|R0)


log
p(RT )
q(RT |R0, s1:T )


+

T
X

t=2
Eq(R1:T ,s1:t|R0)


log
pθ(Rt−1|Rt, st)
q(Rt−1|Rt, R0, s1:t)



(114)

= Eq(R1,s1|R0)

log pθ(R0|R1, s1)

(115)

+Eq(RT |R0,s1:T )q(s1:T )


log
p(RT )
q(RT |R0, s1:T )


+

T
X

t=2
Eq(Rt,Rt−1,s1:t|R0)


log
pθ(Rt−1|Rt, st)
q(Rt−1|Rt, R0, s1:t)



(116)

= Eq(R1,s1|R0)

log pθ(R0|R1, s1)


|
{z
}
reconstruction term

(117)

−Eq(s1:t)DKL(q(RT |R0, s1:T ) ∥p(RT ))
|
{z
}
prior matching term

−

T
X

t=2
Eq(Rt,s1:t|R0)

DKL(q(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))


|
{z
}
denoising matching term

(118)

Eventually, the ELOB can be rewritten as follows:

log p(R0) ≥

T
X

t=1
Eq(Rt,st|R0)


log pϑ(st|Rt)

q(st)



|
{z
}
mask prediction term

+ Eq(R1,s1|R0)

log pθ(R0|R1, s1)


|
{z
}
reconstruction term

(119)

−Eq(s1:t)DKL(q(RT |R0, s1:T ) ∥p(RT ))
|
{z
}
prior matching term

−

T
X

t=2
Eq(Rt,s1:t|R0)

DKL(q(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))


|
{z
}
denoising matching term

(120)

The mask prediction term can be implemented by a node classifier sϑ. For the denoising matching
term, by Bayes rule, the q(Rt−1|Rt, R0, s1:t) can be written as:

q(Rt−1|Rt, R0, s1:t) = q(Rt|Rt−1, R0, s1:t)q(Rt−1|R0, s1:t−1)

q(Rt|R0, s1:t)
,
(121)

29


---Page Break---
For the naive SubgDiff, we have

q(Rt|Rt−1, R0, s1:t) := N(Rt−1,
p

1 −βtstRt−1, (βtst)I),
(122)

q(Rt−1|R0, s1:t−1) := N(Rt−1, √¯γt−1R0, (1 −¯γt−1)I).
(123)

Then the denoising matching term can also be calculated via Lemma C.1 (let q(Rt|Rt−1, R0) :=
q(Rt|Rt−1, R0, s1:t) , q(Rt−1|R0) := q(Rt−1|R0, s1:t−1) and pθ(Rt−1|Rt) = pθ(Rt−1)). Thus,
the training objective of SubgDiff is:

L(θ, ϑ) = Et,R0,ϵ


stβt
2(1 −stβt)(1 −¯γt−1)∥ϵ −ϵθ(√¯γtR0 +
p

(1 −¯γt)ϵ, t, G)∥2 + λBCE(st, sϑ(G, Rt, t))


(124)

D.2.1
Expectation of s1:T

The denoising matching term in (120) can be calculated by only sampling (Rt, st) instead of
(Rt, s1:t). Specifically, we substitute (121) into the denoising matching term:

Eq(Rt,Rt−1,s1:t|R0)


log
pθ(Rt−1|Rt, st)
q(Rt−1|Rt, R0, s1:t)


(125)

= Eq(Rt,Rt−1,s1:t|R0)



log
pθ(Rt−1|Rt, st)

q(Rt|Rt−1,R0,s1:t)q(Rt−1|R0,s1:t−1)

q(Rt|R0,s1:t)




(126)

= Eq(Rt,Rt−1,s1:t|R0)



log pθ(Rt−1|Rt, st)

q(Rt|Rt−1,R0,st)

q(Rt|R0,s1:t)
−log q(Rt−1|R0, s1:t−1)




(127)

≥Eq(Rt,Rt−1,|R0,s1:t)

Eq(s1:t) log pθ(Rt−1|Rt, st)q(Rt|R0, s1:t)

(128)

−Eq(st)



log Eq(s1:t−1)q(Rt−1|R0, s1:t−1)
|
{z
}
:=q(EsRt−1|R0)

+ log Eq(s1:t−1)q(Rt|Rt−1, R0, s1:t)
|
{z
}
:=q(Rt|EsRt−1,R0,st)




(129)

= Eq(Rt,Rt−1,|R0,s1:t)

"

Eq(s1:t) log pθ(Rt−1|Rt, st)

1
q(Rt|R0,s1:t)
−Eq(st) log q(EsRt−1|R0) −Eq(st) log q(Rt|EsRt−1, R0, st)

#

(130)

= Eq(Rt,Rt−1,|R0,s1:t)



Eq(s1:t) log
pθ(Rt−1|Rt, st)

q(Rt|EsRt−1,R0,st)q(EsRt−1|R0)

q(Rt|R0,s1:t)




(131)

= Eq(Rt,Rt−1,s1:t|R0)



log
pθ(Rt−1|Rt, st)

q(Rt|EsRt−1,R0,st)q(EsRt−1|R0)

q(Rt|R0,s1:t)





|
{z
}
denoising matching term

(132)

= Eq(Rt,Rt−1,s1:t|R0)


log
pθ(Rt−1|Rt, st)
ˆq(Rt−1|Rt, R0, s1:t)



(133)

= Eq(Rt,s1:t|R0)

DKL(ˆq(Rt−1|Rt, R0, s1:t) ∥pθ(Rt−1|Rt, st))


|
{z
}
denoising matching term

(134)

Thus, we should focus on calculating the distribution

ˆq(Rt−1|Rt, R0, s1:t) := q(Rt|EsRt−1, R0, st)q(EsRt−1|R0)

q(Rt|R0, s1:t)
(135)

By lemma C.1, if we can gain the expression of q(Rt|EsRt−1, R0, st) and q(EsRt−1|R0), we can
get the training objective and sampling process.

D.3
Single-step subgraph diffusion

D.3.1
Training

I: Step 0 to Step t −1 (R0 →Rt−1):
The state space of the mask diffusion should be the mean of
the random state.

30


---Page Break---
EsRt ∼N(EsRt;
p

1 −βtEsRt−1, βtI)
(136)

q(Rt|R0, s1:t) = N(Rt, √¯γtR0, (1 −¯γt)I).
(137)

Form (137), we have:

Rt =
p

1 −stβtRt−1 +
p

stβtϵt−1
(138)

ERt = (p
p

1 −βt + 1 −p)ERt−1 + p
p

βtϵt−1
(139)

= (p
p

1 −βt + 1 −p)(p
p

1 −βt−1 + 1 −p)ERt−2 + (p
p

1 −βt + 1 −p)p
p

βt−1ϵt−2 + p
p

βtϵt−1
(140)

= (p
p

1 −βt + 1 −p)(p
p

1 −βt−1 + 1 −p)ERt−2 +
q

[(p
p

1 −βt + 1 −p)p
p

βt−1]2 + [p
p

βt]2ϵt−2
(141)
= ....
(142)

=

tY

i=1
(p
p

1 −βi + 1 −p)R0 +

v
u
u
t[

tY

j=2
(p
p

1 −βj + 1 −p)p
p

β1]2 + [

tY

j=3
(p
p

1 −βj + 1 −p)p
p

β2]2 + ...+ϵ0

(143)

=

tY

i=1
(p
p

1 −βi + 1 −p)R0 +

v
u
u
t

t
X

i=1
[

tY

j=i+1
(p
p

1 −βj + 1 −p)p
p

βi]2
(144)

=

tY

i=1

√αiR0 +

v
u
u
t

t
X

i=1
[

tY

j=i+1

√αip
p

βi]2ϵ0
(145)

=

tY

i=1

√αiR0 + p

v
u
u
t

t
X

i=1

tY

j=i+1
αjβiϵ0
(146)

= √¯αtR0 + p

v
u
u
t

t
X

i=1

¯αt
¯αi
βiϵ0
(147)

(148)

where αi := (p√1 −βi + 1 −p)2 and ¯αt = Qt
i=1 αi.

q(ERt|R0) = N(Rt; √¯αtR0, p2
t
X

i=1

¯αt
¯αi
βiI)
(149)

II: Step t −1 to Step t (Rt−1 →Rt):
We build the step t −1 →t is a discrete transition from
q(Rt−1|R0), with

q(EsRt−1|R0) = N(Rt−1;

t−1
Y

i=1

√αiR0, p2
t−1
X

i=1

t−1
Y

j=i+1
αjβiI)
(150)

q(Rt|EsRt−1, st) = N(Rt;
p

1 −stβtERt−1, stβtI)
(151)

31


---Page Break---
Rt =
p

1 −stβtERt−1 +
p

stβtϵt−1
(152)

=
p

1 −stβt



√¯αt−1R0 + p

v
u
u
t

t−1
X

i=1

¯αt−1

¯αi
βiϵ0



+
p

stβtϵt−1
(153)

=
p

1 −stβt
√¯αt−1R0 + p
p

1 −stβt

v
u
u
t

t−1
X

i=1

¯αt−1

¯αi
βiϵ0 +
p

stβtϵt−1
(154)

=
p

1 −stβt
√¯αt−1R0 +

v
u
u
tp2(1 −stβt)

t−1
X

i=1

¯αt−1

¯αi
βi + stβtϵ0
(155)

Step 0 to Step t (R0 →Rt):

q(Rt|R0) =
Z
q(Rt|ERt−1)q(ERt−1|R0)dERt−1
(156)

= N(Rt;
p

1 −stβt
√¯αiR0, (p2(1 −stβt)

t−1
X

i=1

¯αt−1

¯αi
βi + stβt)I)
(157)

Thus, from subsection D.2.1, the training objective of 1-step SubgDiff is:

Lsimple(θ, ϑ) = Et,R0,st,ϵ[st∥ϵ −ϵθ(Rt, t)∥2 −BCE(st, sϑ(Rt, t))]
(158)

where BCE(st, sϑ) = st log sϑ(Rt, t) + (1 −st) log (1 −sϑ(Rt, t)) is Binary Cross Entropy loss.
However, training the SubgDiff is not trivial. The challenges come from two aspects: 1) the mask
predictor should be capable of perceiving the sensible noise change between (t −1)-th and t-th step.
However, the noise scale βt is relatively small when t is small, especially if the diffusion step is larger
than a thousand, thereby mask predictor cannot precisely predict. 2) The accumulated noise for each
node at (t −1)-th step would be mainly affected by the mask sampling from 1 to t −1 step, which
heavily increases the difficulty of predicting the noise added between (t −1)-step to t-step.

D.3.2
Sampling

Finally, the sampling can be written as:

Rt−1 =


(1 −stβt)p2 Pt−1
i=1
¯αt−1

¯αi βi + stβt

Rt −

stβt
q

p2(1 −stβt) Pt−1
i=1
¯αt−1

¯αi βi + stβt

ϵθ(Rt, t)
√1 −stβt(stβt + (1 −stβt)p2 Pt−1
i=1
¯αt−1

¯αi βi)
+ σtz

(159)

=
1
√1 −stβt
Rt −


stβt
q

p2(1 −stβt) Pt−1
i=1
¯αt−1

¯αi βi + stβt


√1 −stβt(stβt + (1 −stβt)p2 Pt−1
i=1
¯αt−1

¯αi βi)
ϵθ(Rt, t) + σtz
(160)

=
1
√1 −stβt
Rt −
stβt
√1 −stβt
q

stβt + (1 −stβt)p2 Pt−1
i=1
¯αt−1

¯αi βi
ϵθ(Rt, t) + σtz
(161)

(162)

where st = sϑ(Rt, t) and

σt =
sϑ(Rt, t)βtp2 Pt−1
i=1
¯αt−1

¯αi βi

sϑ(Rt, t)βt + p2(1 −sϑ(Rt, t)βt) Pt−1
i=1
¯αt−1

¯αi βi
(163)

32


---Page Break---
𝐑𝑘𝑚
𝐑𝑘𝑚+1
𝐑𝑚+1 𝑘

𝝐𝑘𝑚+1
𝒔𝑘𝑚+1

𝝐𝑘𝑚+3
𝒔𝑘𝑚+1
𝝐𝑘𝑚+2
𝒔𝑘𝑚+1

𝝐𝑚+1 𝑘

𝒔𝑘𝑚+1

𝐑𝑘𝑚+2

Figure 8: An example of k-step same subgraph diffusion, where the mask vectors are same as skm+1 from step
km to (m + 1)k, m ∈N+ .

E
Expectation state distribution

The state space of the mask diffusion should be the mean of the random state.

EstRt ∼N(ERt;
p

1 −βtEst−1Rt−1, βtI)
(164)

Form Equation 137, we have:

Rt =
p

1 −stβtRt−1 +
p

stβtϵt−1
(165)

ERt = (p
p

1 −βt + 1 −p)ERt−1 + p
p

βtϵt−1
(166)

= (p
p

1 −βt + 1 −p)(p
p

1 −βt−1 + 1 −p)ERt−2
(167)

+ (p
p

1 −βt + 1 −p)p
p

βt−1ϵt−2 + p
p

βtϵt−1
(168)

= (p
p

1 −βt + 1 −p)(p
p

1 −βt−1 + 1 −p)ERt−2
(169)

+
q

[(p
p

1 −βt + 1 −p)p
p

βt−1]2 + [p
p

βt]2ϵt−2
(170)

= ....
(171)

=

tY

i=1
(p
p

1 −βi + 1 −p)R0
(172)

+

v
u
u
t[

tY

j=2
(p
p

1 −βj + 1 −p)p
p

β1]2 + [

tY

j=3
(p
p

1 −βj + 1 −p)p
p

β2]2 + ...+ϵ0 (173)

=

tY

i=1
(p
p

1 −βi + 1 −p)R0 +

v
u
u
t

t
X

i=1
[

tY

j=i+1
(p
p

1 −βj + 1 −p)p
p

βi]2
(174)

=

tY

i=1

√αiR0 +

v
u
u
t

t
X

i=1
[

tY

j=i+1

√αip
p

βi]2ϵ0
(175)

=

tY

i=1

√αiR0 + p

v
u
u
t

t
X

i=1

tY

j=i+1
αjβiϵ0
(176)

= √¯αiR0 + p

v
u
u
t

t
X

i=1

¯αt
¯αi
βiϵ0
(177)

(178)

where αi := (p√1 −βi + 1 −p)2 and ¯αt = Qt
i=1 αi.

Finally, the Expectation state distribution is:

q(ERt|R0) = N(ERt;

tY

i=1

√αiR0, p2
t
X

i=1

tY

j=i+1
αjβiI)
(179)

33


---Page Break---
F
The derivation of SubgDiff

When t is an integer multiple of k,

ERt =

t/k
Y

j=1
(p

v
u
u
t

kj
Y

i=(j−1)k+1
(1 −βi) + 1 −p)R0
(180)

+

v
u
u
u
t

t/k
X

l=1





t/k
Y

j=l+1
(p

v
u
u
t

kj
Y

i=(j−1)k+1
(1 −βi) + 1 −p)p

v
u
u
t1 −

kl
Y

i=(l−1)k+1
(1 −βi)





2

ϵ0
(181)

=

t/k
Y

j=1

√αjR0 + p

v
u
u
u
t

t/k
X

l=1

t/k
Y

j=l+1
αj(1 −

kl
Y

i=(l−1)k+1
(1 −βi))ϵ0
(182)

= p¯αt/kR0 + p

v
u
u
u
t

t/k
X

l=1

¯αt/k

¯αl
(1 −

kl
Y

i=(l−1)k+1
(1 −βi))ϵ0
(183)

where αj = (p
qQkj
i=(j−1)k+1(1 −βi) + 1 −p)2.

When t ∈N, let m := ⌊(t −1)/k⌋, and we have

Rt =

v
u
u
t

tY

i=km+1
(1 −βiskm+1)ERm×k +

v
u
u
t1 −

tY

i=km+1
(1 −βiskm+1)ϵm×k
(184)

=

v
u
u
t

tY

i=km+1
(1 −βiskm+1)



√¯αmR0 + p

v
u
u
t

m
X

l=1

¯αm

¯αl
(1 −

kl
Y

i=(l−1)k+1
(1 −βi))ϵ0




(185)

+

v
u
u
t1 −

tY

i=km+1
(1 −βiskm+1)ϵm
(186)

=

v
u
u
t

tY

i=km+1
γi
√¯αmR0
(187)

+

v
u
u
t

 
tY

i=km+1
γi

!

p2
m
X

l=1

¯αm

¯αl
(1 −

kl
Y

i=(l−1)k+1
(1 −βi)) +

 

1 −

tY

i=km+1
γi

!

ϵ0
(188)

where γi = 1 −βiskm+1.

q(Rt|R0) = N(Rkm;

v
u
u
t

tY

i=km+1
γi
√¯αmR0,
(189)





 
tY

i=km+1
γi

!

p2
m
X

l=1

¯αm

¯αl
(1 −

kl
Y

i=(l−1)k+1
(1 −βi)) + 1 −

tY

i=km+1
γi



I)
(190)

Let ¯γi = Qi
t=1 γt, and ¯βt = Qt
i=1(1 −βi)

q(Rt|R0) = N(Rkm;
r ¯γt

¯γkm

√¯αmR0,

 
¯γt
¯γkm
p2
m
X

l=1

¯αm

¯αl
(1 −
¯βkl
¯β(l−1)k
) + 1 −
¯γt
¯γkm

!

I)
(191)

34


---Page Break---
F.0.1
Sampling

µ1 =
p

1 −skm+1βt,
(192)

σ2
1 = skm+1βt
(193)

µ2 =
r ¯γt−1

¯γkm

√¯αm
(194)

σ2
2 = ¯γt−1

¯γkm
p2
m
X

l=1

¯αm

¯αl
(1 −

kl
Y

i=(l−1)k+1
(1 −βi)) + 1 −¯γt−1

¯γkm
(195)

According to the Lemma C.1, we have

Rt−1 = 1

µ1

 

Rt −
σ2
1
p

µ2
1σ2
2 + σ2
1
ϵθ(Rt, t)

!

+
σ1σ2
p

µ2
1σ2
2 + σ2
1
z
(196)

=
1
p

1 −skm+1βt
(Rt−
(197)

skm+1βt
q

(1 −skm+1βt)( ¯γt−1

¯γkm p2 Pm
l=1
¯αm

¯αl (1 −Qkl
i=(l−1)k+1(1 −βi)) + 1 −¯γt−1

¯γkm ) + skm+1βt
ϵθ(Rt, t))

(198)

+

p

skm+1βt
q

¯γt−1

¯γkm p2 Pm
l=1
¯αm

¯αl (1 −Qkl
i=(l−1)k+1(1 −βi)) + 1 −¯γt−1

¯γkm
q

(1 −skm+1βt)( ¯γt−1

¯γkm p2 Pm
l=1
¯αm

¯αl (1 −Qkl
i=(l−1)k+1(1 −βi)) + 1 −¯γt−1

¯γkm ) + skm+1βt
z

(199)

The schematic can see Figure 5.

35


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We have summarized our contributions at the end of the introduction.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: See Appendix Section B.2.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: See Appendix Sections C, D.1, D.2, E, F.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We provide experimental details in Appendix Section A.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: The source code is available at this repo.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: Appendix Section A. Specifically, the hyperparameters are in Table 6 and
Table 7 in the Appendix.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: When possible, we report the mean and standard deviation of three random
seeds.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: See Appendix Section A.3.
9. Code Of Ethics

36


---Page Break---
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: This research conducted in the paper conforms with the NeurIPS Code of
Ethics.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [No]
Justification: This paper presents work whose goal is to advance the field of Machine
Learning. There are some potential societal consequences of our work, none of which we
feel must be specifically highlighted here.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This paper poses no such risks.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: The materials in this paper are used with permission and properly cited.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The weights of trained models are provided in this repo.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: This paper does not involve crowdsourcing or research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: This paper does not involve crowdsourcing nor research with human subjects.

37


---Page Break---
