PureGen: Universal Data Purification for Train-Time
Poison Defense via Generative Model Dynamics

Sunay Bhat
∗
sunaybhat1@ucla.edu
Jeffrey Jiang
jeffrey.jiang@ucla.edu

Omead Pooladzandi
opooladz@caltech.edu
Alexander Branch
alexrbranch@ucla.edu
Gregory Pottie
pottie@ee.ucla.edu

Abstract
Train-time data poisoning attacks threaten machine learning models by introducing
adversarial examples during training, leading to misclassification. Current defense
methods often reduce generalization performance, are attack-specific, and impose
significant training overhead. To address this, we introduce a set of universal
data purification methods using a stochastic transform, Ψ(x), realized via itera-
tive Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion
Probabilistic Models (DDPMs), or both. These approaches purify poisoned data
with minimal impact on classifier generalization. Our specially trained EBMs and
DDPMs provide state-of-the-art defense against various attacks (including Narcis-
sus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and
CINIC-10, without needing attack or classifier-specific information. We discuss
performance trade-offs and show that our methods remain highly effective even
with poisoned or distributionally shifted generative model training data.

1
Introduction

Large datasets enable modern deep learning models but are vulnerable to data poisoning, where
adversaries inject imperceptible poisoned images to manipulate model behavior at test time. Poisons
can be created with or without knowledge of the model’s architecture or training settings. As deep
learning models grow in capability and usage, securing them against such attacks while preserving
accuracy is critical.

Numerous methods of poisoning deep learning systems to create backdoors have been proposed in
recent years. These disruptive techniques typically fall into two distinct categories: explicit backdoor,
triggered data poisoning, or triggerless poisoning attacks. Triggered attacks conceal an imperceptible
trigger pattern in the samples of the training data leading to the misclassification of test-time samples
containing the hidden trigger [1, 2, 3, 4]. In contrast, triggerless poisoning attacks involve introducing
slight, bounded perturbations to individual images that align them with target images of another class
within the feature or gradient space resulting in the misclassification of specific instances without
necessitating further modification during inference [5, 6, 7, 8, 9]. Alternatively, data availability
attacks pose a training challenge by preventing model learning at train time, but do not introduce
any latent backdoors that can be exploited at inference time [10, 11, 12]. In all these scenarios,
poisoned examples often appear benign and correctly labeled making them challenging for observers
or algorithms to detect.

∗Affiliation for all authors: University of California, Los Angeles
Equal Contributions
GitHub: https://github.com/SunayBhat1/PureGen_PoisonDefense.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
+

Adversarial
Poisons

Clean Images

...

Poisoned
Dataset

PUREGEN
ΨT (x)

Proposed PURE-
GEN Prepro-
cessing Step

...

Purified
Dataset

Standard
Classifier
Train
F(x)

Unchanged CNN
Model Training

Training Phase

...

Poisoned or
Clean Images

Standard
Classifier
Inference
F(x)

Unchanged CNN
Model Inference

...

...

‘Dog’

Class 4

Class 3

Class 6

Defended Model
Predictions

Inference Phase

0.6
0.4
0.2
0.0
0.2
0.4
0.6

0

50

100

150

200

250

300

350

Poisoned
Clean
PureGEN-EBM
PureGEN-DDPM
Uniform Noise

1.25
1.30
1.35
1.40
Energy 
(x)

PUREGEN

Clean
Poisoned

Initial
PureGEN-EBM
PureGEN-DDPM
PureGEN-DDPM
Figure 1: Top The full PUREGEN pipeline is shown where we apply our method as a preprocessing
step with no further downstream changes to the classifier training or inference. Poisoned images are
moderately exaggerated to show visually. Bottom Left Energy distributions of clean, poisoned, and
PUREGEN purified images. Our methods push poisoned images via purification into the natural,clean
image energy manifold. Bottom Right The removal of poison artifacts and the similarity of clean
and poisoned images after purification using PUREGEN EBM and DDPM dynamics. The purified
dataset results in SoTA defense and high classifier natural accuracy.

Current defense strategies against data poisoning exhibit significant limitations. While some methods
rely on anomaly detection through techniques such as nearest neighbor analysis, training loss
minimization, singular-value decomposition, feature activation or gradient clustering [13, 14, 15,
16, 17, 18, 19], others resort to robust training strategies including data augmentation, randomized
smoothing, ensembling, adversarial training and maximal noise augmentation [20, 21, 22, 23, 24, 25,
26]. However, these approaches either undermine the model’s generalization performance [27, 18],
offer protection only against specific attack types [27, 17, 15], or prove computationally prohibitive
for standard deep learning workflows [22, 16, 28, 18, 27, 17, 26]. There remains a critical need for
more effective and practical defense mechanisms in the realm of deep learning security.

Generative models have been used for robust/adversarial training, but not for train-time backdoor
attacks, to the best of our knowledge. Recent works have demonstrated the effectiveness of both
EBM dynamics and Diffusion models to purify datasets against inference or availability attacks
[29, 30, 31], but train-time backdoor attacks present additional challenges in both evaluation and
application, requiring training using Public Out-of-Distribution (POOD) datasets and methods to
avoid cumbersome computation or setup for classifier training.

We propose PUREGEN, a set of powerful stochastic preprocessing defense techniques, ΨT (x),
against train-time poisoning attacks. PUREGEN-EBM uses EBM-guided Markov Chain Monte
Carlo (MCMC) sampling to purify poisoned images, while PUREGEN-DDPM uses a limited for-
ward/reverse diffusion process, specifically for purification. Training DDPM models on a subset of
the noise schedule improves purification by dedicating more model capacity to ‘restoration’ rather
than generation. We further find that the energy of poisoned images is significantly higher than the
baseline images, for a trained EBM, and PUREGEN techniques move poisoned samples to a lower-
energy, natural data manifold with minimal accuracy loss. The PUREGEN pipeline, sample energy
distributions, and purification on a sample image can be seen in Figure 1.PUREGEN significantly
outperforms current defenses in all tested scenarios. Our key contributions in this work are as follows.

• A set of state-of-the-art (SoTA) stochastic preprocessing defenses Ψ(x) against adversarial
poisons using MCMC dynamics of EBMs and DDPMs trained specifically for purification

2


---Page Break---
named PUREGEN-EBM and PUREGEN-DDPM with analysis providing further intuition
on effectiveness
• Experimental results showing the broad application of Ψ(x) with minimal tuning and no
prior knowledge needed of the poison type and classification model
• Results showing SoTA performance can be maintained even when PUREGEN models’
training data includes poisons or is from a significantly different distribution than the
classifier/attacked train data distribution
• Results showing even further performance gains from combinations of PUREGEN-EBM
and PUREGEN-DDPM and robustness to defense-aware poisons

2
Related Work

2.1
Targeted Data Poisoning Attack

Poisoning of a dataset occurs when an attacker injects small adversarial perturbations δ (where
∥δ∥∞≤ξ and typically ξ = 8 or 16/255) into a small fraction, α, of training images, making
poisoning incredibly difficult to detect. These train-time attacks introduce local sharp regions with a
considerably higher training loss [26]. A successful attack occurs when, after SGD optimizes the
cross-entropy training objective on these poisoned datasets, invisible backdoor vulnerabilities are
baked into a classifier, without a noticeable change in overall test accuracy. This is in contrast to
inference-time or other adversarial scenarios where an attacker might be defense or model-aware.
The goal in train-time attacks is "stealth" via minimal impact to the dataset and training and testing
curves while creating backdoors to exploit at deployment.

In the realm of deep network poison security, we encounter two primary categories of attacks:
triggered and triggerless attacks. Triggered attacks, often referred to as backdoor attacks, involve
contaminating a limited number of training data samples with a specific trigger (often a patch) ρ
(similarly constrained ∥ρ∥∞≤ξ) that corresponds to a target label, yadv. After training, a successful
backdoor attack misclassifies when the perturbation ρ is added:

F(x) =
y
x ∈{x : (x, y) ∈Dtest}
yadv
x ∈{x + ρ : (x, y) ∈Dtest, y ̸= yadv}
(1)

Early backdoor attacks were characterized by their use of non-clean labels [32, 1, 33, 3], but more
recent iterations of backdoor attacks have evolved to produce poisoned examples that lack a visible
trigger [2, 34, 4].

On the other hand, triggerless poisoning attacks involve the addition of subtle adversarial perturbations
to base images ∥ϵ∥∞≤ξ, aiming to align their feature representations or gradients with those of
target images of another class, causing target misclassification [5, 6, 7, 8, 9]. These poisoned images
are virtually undetectable by external observers. Remarkably, they do not necessitate any alterations
to the target images or labels during the inference stage. For a poison targeting a group of target
images Π = {(xπ, yπ)} to be misclassified as yadv, an ideal triggerless attack would produce a
resultant function:

F(x) =
y
x ∈{x : (x, y) ∈Dtest \ Π}
yadv
x ∈{x : (x, y) ∈Π}
(2)

Background for data availability attacks can be found in [35]. We include results for one leading
data availability attack Neural Tangent Gradient Attack (NTGA) [12], but we do not focus on such
attacks since they are realized in model results during training. They do not pose a latent security risk
in deployed models, and arguably have ethical applications within data privacy and content creator
protections as discussed in App. 6.

The current leading poisoning attacks that we assess our defense against are listed below. More
details about their generation can be found in App. A.1.

• Bullseye Polytope (BP): BP crafts poisoned samples that position the target near the center
of their convex hull in a feature space [9].
• Gradient Matching (GM): GM generates poisoned data by approximating a bi-level objec-
tive by aligning the gradients of clean-label poisoned data with those of the adversarially-
labeled target [8]. This attack has shown effectiveness against data augmentation and
differential privacy.

3


---Page Break---
• Narcissus (NS): NS is a clean-label backdoor attack that operates with minimal knowledge
of the training set, instead using a larger natural dataset, evading state-of-the-art defenses by
synthesizing persistent trigger features for a given target class. [4].
• Neural Tangent Generalization Attacks (NTGA): NTGA is a clean-label, black-box data
availability attack that can collapse model test accuracy [12].

2.2
Train-Time Poison Defense Strategies

Poison defense categories broadly take two primary approaches: filtering and robust training tech-
niques. Filtering methods identify outliers in the feature space through methods such as thresholding
[14], nearest neighbor analysis [17], activation space inspection [16], or by examining the covariance
matrix of features [15]. These defenses often assume that only a small subset of the data is poisoned,
making them vulnerable to attacks involving a higher concentration of poisoned points. Further-
more, these methods substantially increase training time, as they require training with poisoned data,
followed by computationally expensive filtering and model retraining [16, 17, 14, 15].

On the other hand, robust training methods involve techniques like randomized smoothing [20],
extensive data augmentation [36], model ensembling [21], gradient magnitude and direction con-
straints [37], poison detection through gradient ascent [24], and adversarial training [27, 28, 25].
Additionally, differentially private (DP) training methods have been explored as a defense against
data poisoning [22, 38]. Robust training techniques often require a trade-off between generalization
and poison success rate [22, 37, 24, 28, 25, 26] and can be computationally intensive [27, 28]. Some
methods use optimized noise constructed via Generative Adversarial Networks (GANs) or Stochastic
Gradient Descent methods to make noise that defends against attacks [39, 26].

Recently Yang et al. [2022] proposed EPIC, a coreset selection method that rejects poisoned images
that are isolated in the gradient space while training, and Liu et al. [2023] proposed FRIENDS, a
per-image preprocessing transformation that solves a min-max problem to stochastically add ℓ∞norm
ζ-bound ‘friendly noise’ (typically 16/255) to combat adversarial perturbations (of 8/255) [18, 26].

These two methods are the previous SoTA and will serve as a benchmark for our PUREGEN methods
in the experimental results. Finally, simple compression JPEG has been shown to defend against a
variety of other adversarial attacks, and we apply it as a baseline defense in train-time poison attacks
here as well, finding that it often outperforms previous SoTA methods [40].

3
PUREGEN: Purifying Generative Dynamics against Poisoning Attacks

3.1
Energy-Based Models and PUREGEN-EBM

An Energy-Based Model (EBM) is formulated as a Gibbs-Boltzmann density, as introduced in [41].
This model can be mathematically represented as:

pθ(x) =
1
Z(θ) exp(−Gθ(x))q(x),
(3)

where x ∈X ⊂RD denotes an image signal, and q(x) is a reference measure, often a uniform or
standard normal distribution. Here, Gθ signifies the energy potential, parameterized by a ConvNet
with parameters θ.

The EBM Gθ(x) can be interpreted as an unnormalized probability of how natural the image is to
the dataset. Thus, we can use Gθ(x) to filter images based on their likelihood of being poisoned.
Furthermore, the EBM can be used as a generator. Given a starting clear or purified image xτ, we use
Markov Chain Monte Carlo (MCMC) Langevin dynamics to iteratively generate more natural images
via Equation 4.

xτ+∆τ = xτ −∆τ∇xτ Gθ(xτ) +
√

2∆τετ,
(4)

where εk ∼N(0; ID), τ indexes the time step of the Langevin dynamics, and ∆τ is the discretization
of time [41]. ∇xGθ(x) = ∂Gθ(x)/∂x can be obtained by back-propagation. Intuitively, the EBM
informs a noisy stochastic gradient descent toward natural images. More details on the convergent
contrastive learning mechanism of the EBM and mid-run generative dynamics that makes purification
possible can be found in App. A.2.1. Ultimately, the training modifications of using realistic images

4


---Page Break---
to initialize the MCMC runs of negative samples produces mid-run, meta-stable EBM dynamics
which can be leveraged for better purification. Further intuition is in Section 3.4.

3.2
Diffusion Models and PUREGEN-DDPM

Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models proposed by
[Ho et al., 2020] where the key idea is to define a forward diffusion process that adds noise until the
image reaches a noise prior and then learn a reverse process that removes noise to generate samples
as discussed further in App. A.3 [42]. For purification, we are interested in the stochastic “restoration”
of the reverse process, where the forward process can degrade the image enough to remove adversarial
perturbations. We find that only training the DDPM with a subset of the standard βt schedule, where
the original image never reaches the prior, sacrifices generative capabilities for slightly improved
poison defense while reducing training costs. Thus we introduce PUREGEN-DDPM which makes
the simple adjustment of only training DDPMs for an initial portion of the standard forward process,
improving purification capabilities. For our experiments, we find models trained up to 250 steps
outperformed models in terms of poison purification than those trained on higher steps, up to the
standard 1000 steps. We show visualizations and empirical evidence of this in Figure 2 below. In
App. E.2.2 we show that pre-trained, standard DDPMs can offer comparable defense performance,
but with added training cost.

t: 0
t: 150
t: 250
t: 500
t: 750

Standard DDPM q(xt|xt
1)

PureGen-DDPM q(xt|xt
1)

t: 1000

PureGen-DDPM
 250 Timesteps

PureGen-DDPM
 500 Timesteps

Standard
 1000 Timesteps

Narcissus ϵ = 16 1%

Purify Steps
75
100
125
150

Forward Train Steps
Avg Natural Accuracy (%) ↑

150
90.96 ± 0.15
90.21 ± 0.20
89.18 ± 0.11
88.46 ± 0.22
250
91.04 ± 0.17
90.55 ± 0.19
89.75 ± 0.17
89.60 ± 0.17
500
90.48 ± 0.21
89.77 ± 0.20
88.99 ± 0.19
88.19 ± 0.15
750
90.25 ± 0.12
89.06 ± 0.18
88.14 ± 0.10
87.19 ± 0.21
1000
90.11 ± 0.16
89.00 ± 0.25
87.98 ± 0.18
86.83 ± 0.10

Forward Train Steps
Avg Poison Success (%) ↓

150
8.03 ± 6.36
6.36 ± 5.84
5.51 ± 4.07
5.43 ± 4.51
250
7.14 ± 6.94
5.58 ± 5.25
4.36 ± 3.63
4.15 ± 3.24
500
8.88 ± 7.31
6.34 ± 5.10
5.45 ± 4.22
4.93 ± 4.36
750
9.27 ± 6.26
7.01 ± 5.19
5.96 ± 4.64
5.36 ± 3.42
1000
9.12 ± 6.61
7.01 ± 4.82
6.43 ± 5.12
5.12 ± 3.18

Figure 2: Top We compare PUREGEN-DDPM forward steps with the standard DDPM where 250
steps degrades images for purification but does not reach a noise prior. Note that all model are
trained with the same linear β schedule. Bottom Left Generated images from models with 250, 750,
and 1000 (Standard) train forward steps where it is clear 250 steps does not generate realistic images
Bottom Right Significantly improved poison defense performance of PUREGEN-DDPM with 250
train steps indicating a trade-off between data purification and generative capabilities.

3.3
Classification with Stochastic Transformation

Let ΨT : RD →RD be a stochastic pre-processing transformation. In this work, ΨT (x), is the
random variable of a fixed image x, and we define T = (TEBM, TDDPM, TReps) ∈R3, hyperparameters
specifying the number of EBM MCMC steps, the number of diffusion steps, and the number of times
these steps are repeated, respectively. Then, TPUREGEN-EBM = (TEBM, 0, 1) and TPUREGEN-DDPM =
(0, TDDPM, 1).

5


---Page Break---
We compose a stochastic transformation ΨT,k(x) with a randomly initialized deterministic classifier
fϕ(x) ∈RJ (for us, a naturally trained classifier) to define a new deterministic classifier Fϕ(x) ∈RJ
as
Fϕ(x) = EΨT,k(x)[fϕ0(ΨT,k(x))]
(5)

which is trained with cross-entropy loss via SGD to realize Fϕ(x). As this is computationally
infeasible we take fϕ(ΨT,k(x)) as the point estimate of Fϕ(x), which is valid because ΨT,k(x) has
low variance.

3.4
Erasing Poison Signals via Mid-Run MCMC

The stochastic transform ΨT (x) is an iterative process. PUREGEN-EBM is akin to a noisy gradient
descent over the unconditional energy landscape of a learned data distribution. This is more implicit
in the PUREGEN-DDPM dynamics. As T increases, poisoned images move from their initial higher
energy towards more realistic lower-energy samples that lack poison perturbations. As shown in
Figure 1, the energy distributions of poisoned images are much higher, pushing the poisons away
from the likely manifold of natural images. By using Langevin dynamics of EBMs and DDPMs, we
transport poisoned images back toward the center of the energy basin.

1500.82
4001.96
8004.75
12008.01
15009.45
200011.49
PureGEN-EBM Steps (T[0])PurifyTime(s)

0

1

2

3

4

5

6

7

L2 Distance

x
T(x) 2
x
T(x +
) 2
(x +
)
T(x +
) 2
x
T(x +
) 2 
 (x +
)
T(x +
) 2

00.00
250.78
501.48
752.22
1002.95
1253.67
1504.47
1755.26
2005.96
PureGEN-DDPM Steps (T[1])PurifyTime(s)

0

1

2

3

4

5

6

7

L2 Distance

x
T(x) 2
x
T(x +
) 2
(x +
)
T(x +
) 2
x
T(x +
) 2 
 (x +
)
T(x +
) 2

Figure 3: Plot of ℓ2 distances for PUREGEN-EBM (Left) and PUREGEN-DDPM (Right) between
clean images and clean purified (blue), clean images and poisoned purified (green), and poisoned
images and poisoned purified images (orange) at points on the Langevin dynamics trajectory. Purify-
ing poisoned images for less than 250 steps moves a poisoned image closer to its clean image with a
minimum around 150, preserving the natural image while removing the adversarial features.

In from-scratch ϵ = 8 poison scenarios, 150 EBM Langevin steps or 75 DDPM steps fully purifies
the majority of the dataset with minimal feature loss to the original image. In Figure 3, we explore
the Langevin trajectory’s impacts on ℓ2 distance of both purified clean and poisoned images from
the initial clean image (∥x −ΨT (x)∥2 and ∥x −ΨT (x + δ)∥2), and the purified poisoned image’s
trajectory away from its poisoned starting point (∥(x + δ) −ΨT (x + δ)∥2). Both poisoned and clean
distance trajectories converge to similar distances away from the original clean image (limT →∞∥x −
ΨT (x)∥2 = limT →∞∥x −ΨT (x + δ)∥2), and the intersection where ∥(x + δ) −ΨT (x + δ)∥2 >
∥x −ΨT (x + δ)∥2 (indicated by the dotted red line), occurs at ∼150 EBM and 75 DDPM steps,
indicating when purification has moved the poisoned image closer to the original clean image than
the poisoned version of the image.

These dynamics provide a concrete intuition for choosing step counts that best balance poison
defense with natural accuracy (given a poison ϵ), hence why we use 150-1000 EBM steps of 75-125
(specifically 150 EBM, 75 DDPM steps in from-scratch scenarios) shown in App. D.2. Further,
PUREGEN-EBM dynamics stay closer to the original images, while PUREGEN-DDPM moves
further away as we increase the steps as the EBM has explicitly learned a probable data distribution,
while the DDPM restoration is highly dependent on the conditional information in the degraded
image. More experiments comparing the two are shown in App. G.2. These dynamics align with
empirical results showing that EBMs better maintain natural accuracy and poison defense with
smaller perturbations and across larger distributional shifts, but DDPM dynamics are better suited
for larger poison perturbations. Finally, we note the purify times in the x-axes of Fig. 3, where
PUREGEN-EBM is much faster for the same step counts to highlight the computational differences
for the two methods, which we further explore Section 4.5.

Ultimately, one can think of PureGen as sampling from a “close” region in the pixel space around the
original image where proximity is determined by a stochastic process that is initialized at the image

6


---Page Break---
and remains “close” due to an explicit (EBM) or implicit (DDPM) energy gradient - all but assuring
the original poison is mitigated in the process.

4
Experiments

4.1
Experimental Details

Table 1: Poison success and natural accuracy in all ResNet poison training scenarios. We report the
mean and the standard deviations (as subscripts) of 100 GM experiments, 50 BP experiments, and
NS triggers over 10 classes.

From Scratch

CIFAR-10 (ResNet-18)
CINIC-10 (ResNet-18)
Tiny ImageNet (ResNet-34)

Gradient Matching-1%
Narcissus-1%
Narcissus-1%
Gradient Matching-0.25%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
44.00
94.840.2
43.9533.6
94.890.2
93.59
62.060.21
86.320.10
90.79
26.00
65.200.5
EPIC
10.00
85.141.2
27.3134.0
82.201.1
84.71
49.500.27
81.910.08
91.35
18.00
60.550.7
FRIENDS
0.00
91.150.4
8.3222.3
91.010.4
83.03
11.170.25
77.530.60
82.21
2.00
42.747.5
JPEG
0.00
90.000.19
1.781.17
92.940.15
4.13
18.8927.46
81.060.18
92.12
10.00
60.010.47
PUREGEN-DDPM
0.00
90.930.20
1.640.82
90.990.22
2.83
4.762.37
79.350.08
7.74
0.00
50.500.80
PUREGEN-EBM
1.00
92.980.2
1.390.8
92.920.2
2.50
7.730.08
82.370.14
29.48
2.00
63.270.4
Transfer Learning (CIFAR-10, ResNet-18)

Fine-Tune
Linear - Bullseye Polytope

Bullseye Polytope-10%
Narcissus-10%
BlackBox-10%
WhiteBox-1% (CIFAR-100)

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
46.00
89.840.9
33.4133.9
90.142.4
98.27
93.75
83.592.4
98.00
70.090.2
EPIC
42.00
81.955.6
20.9327.1
88.582.0
91.72
66.67
84.343.8
63.00
60.861.5
FRIENDS
8.00
87.821.2
3.045.1
89.810.5
17.32
33.33
85.182.3
19.00
60.900.6
JPEG
0.00
90.400.44
2.953.71
87.630.49
12.55
0.00
92.440.47
8.00
50.420.73
PUREGEN-DDPM
0.00
91.530.15
1.881.12
90.690.26
3.42
0.00
93.810.08
9.0
54.530.64
PUREGEN-EBM
0.00
87.521.2
2.021.0
89.780.6
3.85
0.00
92.380.3
6.00
64.980.3

We evaluate PUREGEN-EBM and PUREGEN-DDPM against state-of-the-art defenses EPIC and
FRIENDS, and baseline defense JPEG, on leading poisons Narcissus (NS), Gradient Matching (GM),
and Bullseye Polytope (BP). Using ResNet18 and CIFAR-10, we measure poison success, natural
accuracy, and max poison success across classes for triggered NS attacks. All poisons and poison
scenario settings come from previous baseline attack and defense works, and additional details on
poison sources, poison crafting, definitions of poison success, and training hyperparameters can be
found in App. D. Poisons were chosen for their availability or ease of generation from the poison-
crafting research community, which is why there are no GM results on CINIC-10 and no Narcissus
results on Tiny-ImageNet. And we note that certain poison successes (GM and BP) are for moving a
single image to a target class per 50-100 classifier scenarios and, hence, lack a standard deviation.
Athough, we show the results are low variance using different seeds on a subset of scenarios in App.
E.2.3.

Our EBMs and DDPMs are trained on the ImageNet (70k) portion of the CINIC-10 dataset, CIFAR-10,
and CalTech-256 for poisons scenarios using CIFAR-10, CINIC-10, and Tiny-ImageNet respectively,
to ensure no overlap of PUREGEN train and attacked classifier train datasets [43, 44, 45].

4.2
Benchmark Results

Table 1 shows our primary results using ResNet18 (34 for Tiny-IN) in which PUREGEN achieves
state-of-the-art (SoTA) poison defense and natural accuracy in all poison scenarios. Both PUREGEN
methods show large improvements over baselines in triggered NS attacks (PUREGEN matches or
exceeds previous SoTA with a 1-6% poison defense reduction and 0.5-1.5% less degradation in
natural accuracy), while maintaining perfect or near-perfect defense with improved natural accuracy
in triggerless BP and GM scenarios. Note that PUREGEN-EBM does a better job maintaining natural
accuracy in the 100 class scenarios (BP-WhiteBox and Tiny-IN), while PUREGEN-DDPM tends to
get much better poison defense when the PUREGEN-EBM is not already low.

Table 2 shows selected results for additional models MobileNetV2, DenseNet121, and Hyperlight
Benchmark (HLB), which is a unique case study with a residual-less network architecture, unique
initialization scheme, and super-convergence training method that recently held the world record of
achieving 94% test accuracy on CIFAR-10 with just 10 epochs [46]. Due to the fact that PUREGEN
is a preprocessing step, it can be readily applied to novel training paradigms with no modifications

7


---Page Break---
Table 2: Results for additional models (MobileNetV2, DenseNet121, and HLB) and the NTGA
data-availability attack. PUREGEN remains state-of-the-art for all train-time latent attacks, while
NTGA defense shows near SoTA performance. *All NTGA baselines pulled from [30].

Scenario
From Scratch NS(ϵ = 8)-1%
Linear Transfer BlackBox BP-10%

Model
MobileNetV2
DenseNet121
Hyperlight Bench
MobileNetV2
DenseNet121

Defense
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
32.700.25
93.920.13
46.5232.2
95.330.1
76.3916.35
93.950.10
81.25
73.270.97
73.47
82.131.62
EPIC
22.350.24
78.169.93
32.6029.4
85.122.4
10.5818.35
24.886.04
56.25
54.475.57
66.67
70.2010.15
FRIENDS
2.000.01
88.820.57
8.6021.2
91.550.3
11.3518.45
87.031.52
41.67
68.861.50
60.42
80.221.90
JPEG
2.301.20
86.600.13
1.901.54
92.030.22
1.730.97
90.920.22
2.08
73.140.71
0.00
78.671.60
PUREGEN-DDPM
2.131.02
86.910.23
1.710.94
90.940.23
1.750.90
89.340.25
0.00
83.150.02
0.00
89.020.15
PUREGEN-EBM
1.640.01
91.750.13
1.420.7
93.480.1
1.891.06
91.940.14
0.00
78.571.37
0.00
89.290.94

NTGA Data Availability Attack

Defense
None
FAutoAug.*
Median Blur*
TVM*
Grayscale*
AVATAR *
JPEG
PUREGEN-DDPM
PUREGEN-EBM

Avg Natural Accuracy (%) ↑
11.490.69
27.562.45
28.431.41
41.411.37
81.270.27
86.220.38
79.220.25
83.480.43
85.220.38

unlike previous baselines EPIC and FRIENDS. In all results, PUREGEN is again SoTA, except for
NTGA data-availability attack, where PUREGEN is just below SoTA method AVATAR (which is also
a diffusion based approach). But we again emphasize data-availability attacks are not the focus of
PUREGEN which secures against latent attacks.

The complete results for all models and all versions of each baseline can be found in App. E.

4.3
PUREGEN Robustness to Train Data Shifts, Poisoning, and Defense-Aware Poisons

CIFAR-10*

FID: 8

CINIC-10_IN

FID: 17

Oxford-IIIT-Pet

FID: 43

FGVC-Aircraft

FID: 100

Office-Home

FID: 100

Food-101

FID: 107

Dtd-Textures

FID: 137

LFW People

FID: 145

Flowers-102

FID: 152

CIFAR-10*

CINIC-10_IN

Oxford-IIIT-Pet

FGVC-Aircraft

Office-Home

Food-101

Dtd-Textures

LFW People

Flowers-102

75.0%

77.5%

80.0%

82.5%

85.0%

87.5%

90.0%

92.5%

Defended Natural Accuracy

NTGA-PureDDPM
NTGA-PureEBM
NS-PureDDPM
NS-PureEBM
NS Baseline (91.01%)

CIFAR-10*

CINIC-10_IN

Oxford-IIIT-Pet

FGVC-Aircraft

Office-Home

Food-101

Dtd-Textures

LFW People

Flowers-102

3.0%

4.0%

5.0%

6.0%

7.0%

8.0%

9.0%

Poison Success

PureDDPM-NS Avg
PureEBM-NS Avg
PureDDPM-NS Max
PureEBM-NS Max
NS Baseline (8.32%) - off plot

Figure 4: PUREGEN-EBM vs. PUREGEN-DDPM with increasingly Out-of-Distribution training
data (for generative model training) and purifying target/attacked distribution CIFAR-10. PUREGEN-
EBM is much more robust to distributional shift for natural accuracy while both PUREGEN-EBM
and PUREGEN-DDPM maintain SoTA poison defense across all train distributions *CIFAR-10 is a
“cheating” baseline as clean versions of poisoned images are present in training data.

An important consideration for PUREGEN is the distributional shift between the data used to train
the generative models and the target dataset to be purified. Figure 4 explores this by training
PUREGEN-EBM and PUREGEN-DDPM on increasingly out-of-distribution (OOD) datasets while
purifying the CIFAR-10 dataset (NS attack). We quantify the distributional shift using the Fréchet
Inception Distance (FID) [47] between the original CIFAR-10 training images and the OOD datasets.
Notably, both methods maintain SoTA or near SoTA poison defense across all training distributions,
highlighting their effectiveness even under distributional shift. The results show that PUREGEN-
EBM is more robust to distributional shift in terms of maintaining natural accuracy, with only a
slight drop in performance even when trained on highly OOD datasets like Flowers-102 and LFW
people. In contrast, PUREGEN-DDPM experiences a more significant drop in natural accuracy as
the distributional shift increases. Note that the CIFAR-10 is a “cheating” baseline, as clean versions
of the poisoned images are present in the generative model training data, but it provides an upper
bound on the performance that can be achieved when the generative models are trained on perfectly
in-distribution data.

8


---Page Break---
Table 3: Both PUREGEN-EBM and PUREGEN-DDPM are robust to NS attack even when fully
poisoned (all classes at once) during model training except for NS Eps-16 for PUREGEN-EBM

Classifier NS Attack Eps
8
16

PureGen w/NS Training Poison
Nat Acc (%) ↑
Poison Success (%) ↓
Max Poison (%) ↓
Nat Acc (%) ↑
Poison Success (%) ↓
Max Poison (%) ↓

PUREGEN-DDPM
91.510.13
2.623.75
12.70
90.310.18
4.613.99
12.86
0
PUREGEN-EBM
91.370.14
1.600.82
2.82
88.210.15
8.736.29
23.05

PUREGEN-DDPM
88.990.16
1.650.79
2.87
85.240.10
4.792.83
10.53
8
PUREGEN-EBM
91.110.18
1.550.89
2.87
87.600.18
5.353.30
12.05

PUREGEN-DDPM
88.020.21
1.570.79
2.79
83.740.21
2.901.54
6.11
16
PUREGEN-EBM
90.760.14
1.280.86
3.43
85.580.40
17.7314.62
44.15

Another important consideration is the robustness of PUREGEN when the generative models them-
selves are trained on poisoned data. Table 3 shows the performance of PUREGEN-EBM and
PUREGEN-DDPM when their training data is fully poisoned with the Narcissus (NS) attack, meaning
that all classes are poisoned simultaneously. The results demonstrate that both PUREGEN-EBM and
PUREGEN-DDPM are highly robust to poisoning during model training, maintaining SoTA poison
defense and natural accuracy with only exception being PUREGEN-EBM’s performance on the more
challenging NS ϵ = 16 attack when poisoned with the same perturbations. While it is unlikely an
attacker would have access to both the the generative model and classifier train datasets, these findings
highlight the inherent robustness of PUREGEN, as the generative models can effectively learn the
underlying clean data distribution even in the presence of poisoned samples during training. This is a
key advantage of PUREGEN compared to other defenses, especially when there is no secure dataset.

Finally, in App. E.2.1, we show results where we integrate an EBM into the Narcissus crafting
pipeline, done by taking a gradient through the EBM MCMC dynamics in three ways based on the
specifics of crafting. In all cases, PUREGEN shows almost no defense degradation, and we actually
show we can generate more effective poisons over baseline this way. These results further validate
the effectiveness of PUREGEN even with defense-aware crafting, which is not a typical assumption in
train-time attacks.

4.4
PUREGEN Extensions on Higher Power Attacks

To leverage the strengths of both PUREGEN-EBM and PUREGEN-DDPM, we propose PUREGEN
combinations that utilize either both EMB and DPPM back-to-back (PUREGEN-NAIVE), EBM and
DDPM with multiple repetitions of smaller steps (PUREGEN-REPS), and finally EBM as a filter
to then use EBM/DDPM on only the k highest energy samples as described in 3.3. For additional
description, see C, and note that these extensions required extensive hyperparameter search with
performance sweeps shown in App F, as there was little intuition for the amount of reps (TReps) or
the filtering threshold (k) needed. Thus, we do not include these methods in our core results, but we
do show the added performance gains on higher power poisons in Table 4, both in terms of increased
perturbation size ϵ = 16 and increased poison % (and both together).

Table 4: PUREGEN-NAIVE, PUREGEN-REPS, and PUREGEN-FILT results showing further perfor-
mance gains on increased poison power scenarios

Narcissus ϵ = 8 10%
Narcissus ϵ = 16 1%
Narcissus ϵ = 16 10%

Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

None
96.276.62
84.570.60
99.97
83.6312.09
93.670.11
97.36
99.350.81
84.580.63
99.97
Best Baseline
16.959.72
84.661.51
33.96
11.8512.60
87.720.19
36.90
71.2822.90
82.830.43
99.25

PUREGEN-DDPM
6.385.16
85.860.46
16.29
5.213.35
86.160.19
13.32
69.3816.73
83.581.02
89.35
PUREGEN-EBM
52.4823.29
86.141.82
99.86
7.354.46
85.610.25
16.94
77.507.01
78.790.93
90.84
PUREGEN-NAIVE
10.438.58
88.200.54
27.42
5.202.61
85.950.23
9.80
63.0115.24
83.140.90
87.17
PUREGEN-REPS
3.752.28
85.560.22
7.74
4.952.48
85.790.18
10.75
53.7917.14
83.921.02
81.09
PUREGEN-FILT
6.476.98
86.082.00
18.81
5.744.05
90.520.18
16.08
69.1312.94
85.471.45
87.66

4.5
PUREGEN Timing and Limitations

Table 5 presents the training times for using PUREGEN-EBM and PUREGEN-DDPM (923 and 4181
seconds respectively to purify) on CIFAR-10 using a TPU V3. Although these times may seem
significant, PUREGEN is a universal defense applied once per dataset, making its cost negligible
when reused across multiple tasks and poison scenarios. To highlight this, we also present the
purification times amortized over the 10 and 100 NS and GM poison scenarios, demonstrating that

9


---Page Break---
the cost becomes negligible when the purified dataset is used multiple times relative to baselines like
FRIENDS which require retraining for each specific task and poison scenario (while still utilizing
the full dataset unlike EPIC). PUREGEN-EBM generally has lower purification times compared to
PUREGEN-DDPM, making it more suitable for subtle and rare perturbations. Conversely, PUREGEN-
DDPM can handle more severe perturbations but at a higher computational cost and potential
reduction in natural accuracy.

Table 5: PUREGEN and baseline Timing Analysis on TPU V3

Train Time (seconds)

Single Classifier
(Median)
Gradient Matching
100 Classifiers
Narcissus
10 Classifiers

None, JPEG
3690
367348
528829
EPIC
3650
3624153
5212295
FRIENDS
11502
11578627
12868s573
PUREGEN-EBM T =[150,0,1]
4613
369948
538032
PUREGEN-DDPM T =[0,75,1]
7871
373148
570637

Training the generative models for PUREGEN involves substantial computational cost and data
requirements. However, as shown in Table 3 and Figure 4, these models remain effective even
when trained on poisoned or out-of-distribution data. This universal applicability justifies the initial
training cost, as the models can defend against diverse poisoning scenarios. So while JPEG is a fairly
effective baseline, the added benefits of PUREGEN start to outweigh the compute as the use cases of
the dataset increase. While PUREGEN combinations (PUREGEN-REPS and PUREGEN-FILT) show
enhanced performance on higher power attacks (Table 4), further research is needed to fully exploit
the strengths of both PUREGEN-EBM and PUREGEN-DDPM.

5
Conclusion

Poisoning has the potential to become one of the greatest attack vectors to AI models, decreasing
model security and eroding public trust. In this work, we introduce PUREGEN, a suite of universal
data purification methods that leverage the stochastic dynamics of Energy-Based Models (EBMs)
and Denoising Diffusion Probabilistic Models (DDPMs) to defend against train-time data poisoning
attacks. PUREGEN-EBM and PUREGEN-DDPM effectively purify poisoned datasets by iteratively
transforming poisoned samples into the natural data manifold, thus mitigating adversarial perturba-
tions. Our extensive experiments demonstrate that these methods achieve state-of-the-art performance
against a range of leading poisoning attacks and can maintain SoTA performance in the face of
poisoned or distributionally shifted generative model training data. These versatile and efficient
methods set a new standard in protecting machine learning models against evolving data poisoning
threats, potentially inspiring greater trust in AI applications.

6
Potential Social Impacts

Poisoning represents one of the greatest emerging threats to AI systems, particularly as founda-
tion models increasingly rely on large, diverse datasets without rigorous quality control against
imperceptible perturbations. This vulnerability is especially concerning in high-stakes domains like
healthcare, security, and autonomous vehicles, where model integrity is crucial and erroneous outputs
could have catastrophic consequences. Our research provides a universal defense method that can
be implemented with minimal impact to existing training infrastructure, enabling practitioners to
preemptively secure their datasets against state-of-the-art poisoning attacks.

While we acknowledge that the poison defense space can promote an ‘arms race’ of increasingly
sophisticated attacks and defenses, our approach’s universality poses a fundamentally harder challenge
for attackers, even when using defense-aware crafting E.2.1. We specifically focus on defending
against latent backdoor vulnerabilities rather than data availability attacks, as the latter can serve
legitimate purposes in protecting content creators’ rights. By providing robust defense against
malicious poisoning while preserving natural model performance, our method helps build trust in AI
systems for increasingly consequential real-world applications.

10


---Page Break---
7
Acknowledgments

This work is supported with Cloud TPUs from Google’s Tensorflow Research Cloud (TFRC). We
would like to acknowledge Jonathan Mitchell, Mitch Hill, Yuan Du and Kathrine Abreu for support
and discussion on base EBM and Diffusion code, and Yunzheng Zhu for his help in crafting poisons.

References

[1] T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine
learning model supply chain,” arXiv preprint arXiv:1708.06733, 2017.
[2] A. Turner, D. Tsipras, and A. Madry, “Clean-label backdoor attacks,” 2018.
[3] H. Souri, M. Goldblum, L. Fowl, R. Chellappa, and T. Goldstein, “Sleeper agent: Scal-
able hidden trigger backdoors for neural networks trained from scratch,” arXiv preprint
arXiv:2106.08970, 2021.
[4] Y. Zeng, M. Pan, H. A. Just, L. Lyu, M. Qiu, and R. Jia, “Narcissus: A practical clean-label
backdoor attack with limited information,” arXiv preprint arXiv:2204.05255, 2022.
[5] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer, T. Dumitras, and T. Goldstein, “Poison
frogs! targeted clean-label poisoning attacks on neural networks,” 2018.
[6] C. Zhu, W. R. Huang, H. Li, G. Taylor, C. Studer, and T. Goldstein, “Transferable clean-label
poisoning attacks on deep neural nets,” in International Conference on Machine Learning, 2019,
pp. 7614–7623.
[7] W. R. Huang, J. Geiping, L. Fowl, G. Taylor, and T. Goldstein, “Metapoison: Practical general-
purpose clean-label data poisoning,” Advances in Neural Information Processing Systems,
vol. 33, 2020.
[8] J. Geiping,
L. H. Fowl,
W. R. Huang,
W. Czaja,
G. Taylor,
M. Moeller,
and
T. Goldstein, “Witches’ brew:
Industrial scale data poisoning via gradient matching,”
in International Conference on Learning Representations, 2021. [Online]. Available:
https://openreview.net/forum?id=01olnfLIbD
[9] H. Aghakhani, D. Meng, Y.-X. Wang, C. Kruegel, and G. Vigna, “Bullseye polytope: A scalable
clean-label poisoning attack with improved transferability,” in 2021 IEEE European Symposium
on Security and Privacy (EuroS&P).
IEEE, 2021, pp. 159–178.
[10] J. Feng, Q.-Z. Cai, and Z.-H. Zhou, “Learning to confuse: Generating training time adversarial
data with auto-encoder,” 2019.
[11] H. Huang, X. Ma, S. M. Erfani, J. Bailey, and Y. Wang, “Unlearnable examples: Making
personal data unexploitable,” 2021.
[12] C.-H. Yuan and S.-H. Wu, “Neural tangent generalization attacks,” in Proceedings of the
38th International Conference on Machine Learning, ser. Proceedings of Machine Learning
Research, M. Meila and T. Zhang, Eds., vol. 139.
PMLR, 18–24 Jul 2021, pp. 12 230–12 240.
[Online]. Available: https://proceedings.mlr.press/v139/yuan21b.html
[13] G. F. Cretu, A. Stavrou, M. E. Locasto, S. J. Stolfo, and A. D. Keromytis, “Casting out demons:
Sanitizing training data for anomaly sensors,” in 2008 IEEE Symposium on Security and Privacy
(sp 2008).
IEEE, 2008, pp. 81–95.
[14] J. Steinhardt, P. W. Koh, and P. Liang, “Certified defenses for data poisoning attacks,” 2017.
[15] B. Tran, J. Li, and A. Madry, “Spectral signatures in backdoor attacks,” in Advances in Neural
Information Processing Systems, 2018, pp. 8000–8010.
[16] B. Chen, W. Carvalho, N. Baracaldo, H. Ludwig, B. Edwards, T. Lee, I. Molloy, and B. Srivas-
tava, “Detecting backdoor attacks on deep neural networks by activation clustering,” in SafeAI@
AAAI, 2019.
[17] N. Peri, N. Gupta, W. R. Huang, L. Fowl, C. Zhu, S. Feizi, T. Goldstein, and J. P. Dickerson,
“Deep k-nn defense against clean-label data poisoning attacks,” in European Conference on
Computer Vision.
Springer, 2020, pp. 55–70.
[18] Y. Yang, T. Y. Liu, and B. Mirzasoleiman, “Not all poisons are created equal: Robust training
against data poisoning,” 2022.

11


---Page Break---
[19] O. Pooladzandi, D. Davini, and B. Mirzasoleiman, “Adaptive second order coresets for data-
efficient machine learning,” 2022.

[20] M. Weber, X. Xu, B. Karlaš, C. Zhang, and B. Li, “Rab: Provable robustness against backdoor
attacks,” arXiv preprint arXiv:2003.08904, 2020.

[21] A. Levine and S. Feizi, “Deep partition aggregation: Provable defenses against general poisoning
attacks,” in International Conference on Learning Representations, 2020.

[22] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep
learning with differential privacy,” in Proceedings of the 2016 ACM SIGSAC conference on
computer and communications security, 2016, pp. 308–318.

[23] Y. Ma, X. Z. Zhu, and J. Hsu, “Data poisoning against differentially-private learners: Attacks
and defenses,” in International Joint Conference on Artificial Intelligence, 2019.

[24] Y. Li, X. Lyu, N. Koren, L. Lyu, B. Li, and X. Ma, “Anti-backdoor learning: Training clean
models on poisoned data,” Advances in Neural Information Processing Systems, vol. 34, 2021.

[25] L. Tao, L. Feng, J. Yi, S.-J. Huang, and S. Chen, “Better safe than sorry: Preventing delusive
adversaries with adversarial training,” Advances in Neural Information Processing Systems,
vol. 34, 2021.

[26] T. Y. Liu, Y. Yang, and B. Mirzasoleiman, “Friendly noise against adversarial noise: A powerful
defense against data poisoning attacks,” 2023.

[27] J. Geiping, L. Fowl, G. Somepalli, M. Goldblum, M. Moeller, and T. Goldstein, “What doesn’t
kill you makes you robust (er): Adversarial training against poisons and backdoors,” arXiv
preprint arXiv:2102.13624, 2021.

[28] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models
resistant to adversarial attacks,” in International Conference on Learning Representations, 2018.

[29] M. Hill, J. C. Mitchell, and S.-C. Zhu, “Stochastic security: Adversarial defense using long-run
dynamics of energy-based models,” in International Conference on Learning Representations,
2021. [Online]. Available: https://openreview.net/forum?id=gwFTuzxJW0

[30] H. M. Dolatabadi, S. Erfani, and C. Leckie, “The devil’s advocate: Shattering the illusion of
unexploitable data using diffusion models,” 2024.

[31] W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat, and A. Anandkumar, “Diffusion models for
adversarial purification,” arXiv preprint arXiv:2205.07460, 2022.

[32] X. Chen, C. Liu, B. Li, K. Lu, and D. Song, “Targeted backdoor attacks on deep learning
systems using data poisoning,” arXiv preprint arXiv:1712.05526, 2017.

[33] Y. Liu, S. Ma, Y. Aafer, W.-C. Lee, J. Zhai, W. Wang, and X. Zhang, “Trojaning attack on neural
networks,” 2017.

[34] A. Saha, A. Subramanya, and H. Pirsiavash, “Hidden trigger backdoor attacks,” 2019.

[35] D. Yu,
H. Zhang,
W. Chen,
J. Yin,
and T.-Y. Liu,
“Availability attacks create
shortcuts,”
in Proceedings of the 28th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, ser. KDD ’22.
ACM, Aug. 2022. [Online]. Available:
http://dx.doi.org/10.1145/3534678.3539241

[36] E. Borgnia, V. Cherepanova, L. Fowl, A. Ghiasi, J. Geiping, M. Goldblum, T. Goldstein, and
A. Gupta, “Strong data augmentation sanitizes poisoning and backdoor attacks without an
accuracy tradeoff,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP).
IEEE, 2021, pp. 3855–3859.

[37] S. Hong, V. Chandrasekaran, Y. Kaya, T. Dumitra¸s, and N. Papernot, “On the effectiveness
of mitigating data poisoning attacks with gradient shaping,” arXiv preprint arXiv:2002.11497,
2020.

[38] B. Jayaraman and D. Evans, “Evaluating differentially private machine learning in practice,” in
28th {USENIX} Security Symposium ({USENIX} Security 19), 2019, pp. 1895–1912.

[39] D. Madaan, J. Shin, and S. J. Hwang, “Learning to generate noise for multi-attack robustness,”
2021.

12


---Page Break---
[40] N. Das, M. Shanbhogue, S.-T. Chen, F. Hohman, S. Li, L. Chen, M. E. Kounavis, and D. H.
Chau, “Shield: Fast, practical defense and vaccination for deep learning using jpeg compression,”
2018.
[41] J. Xie, Y. Lu, S.-C. Zhu, and Y. Wu, “A theory of generative convnet,” in Proceedings of the
33rd International Conference on Machine Learning, 2016, pp. 2635–2644.
[42] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” 2020.
[43] A. Krizhevsky, V. Nair, and G. Hinton, “Cifar-10 (canadian institute for advanced research).”
[Online]. Available: http://www.cs.toronto.edu/~kriz/cifar.html
[44] L. N. Darlow, E. J. Crowley, A. Antoniou, and A. J. Storkey, “Cinic-10 is not imagenet or
cifar-10,” 2018.
[45] G. Griffin, A. Holub, and P. Perona, “Caltech 256,” Apr 2022.
[46] T. Balsam, “hlb-cifar10,” 2023, released on 2023-02-12. [Online]. Available:
https:
//github.com/tysam-code/hlb-CIFAR10
[47] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two
time-scale update rule converge to a local nash equilibrium,” 2018.
[48] E. Nijkamp, M. Hill, T. Han, S.-C. Zhu, and Y. N. Wu, “On the anatomy of MCMC-based
maximum likelihood learning of energy-based models,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 34, 2020.
[49] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral normalization for generative
adversarial networks,” arXiv preprint arXiv:1802.05957, 2018.
[50] A. Schwarzschild, M. Goldblum, A. Gupta, J. P. Dickerson, and T. Goldstein, “Just how toxic is
data poisoning? a unified benchmark for backdoor and data poisoning attacks,” 2021.
[51] J. Whitaker, “Hugging face ddpm butterflies model,” https://huggingface.co/johnowhitaker/
ddpm-butterflies-32px, accessed: 2024-08-05.
[52] N.
Onzo,
“Hugging
face
ddpm
anime
model,”
https://huggingface.co/onragi/
anime-ddpm-32-res2-v3, accessed: 2024-08-05.
[53] N. Kokhlikyan, V. Miglani, M. Martin, E. Wang, B. Alsallakh, J. Reynolds, A. Melnikov,
N. Kliushkina, C. Araya, S. Yan, and O. Reblitz-Richardson, “Captum: A unified and generic
model interpretability library for pytorch,” 2020.

13


---Page Break---
A
Further Background

A.1
Poisons

The goal of adding train-time poisons is to change the prediction of a set of target examples Π =
{(xπ, yπ)} ⊂Dtest or triggered examples {(x + ρ, y) : (x, y) ∈Dtest} to an adversarial label yadv.

Targeted clean-label data poisoning attacks can be formulated as the following bi-level optimization
problem:

argmin
δi∈Cδ,ρ∈Cρ
Pn
i=0 1δi̸=0≤αn

X

(xπ,yπ)∈Π
L
 
F(xπ + ρ; ϕ(δ)), yadv

s.t.
ϕ(δ)=argmin
ϕ

X

(x,y)∈D
L (F(x+δi; ϕ), y)
(6)

For a triggerless poison, we solve for the ideal perturbations δi to minimize the adversarial loss on the
target images, where Cδ = C, Cρ = {0 ∈RD}, and D = Dtrain. To address the above optimization
problem, powerful poisoning attacks such as Meta Poison (MP) [7], Gradient Matching (GM) [8],
and Bullseye Polytope (BP) [9] craft the poisons to mimic the gradient of the adversarially labeled
target, i.e.,

∇L
 
Fϕ (xπ) , yadv
∝
X

i:δi̸=0
∇L (Fϕ(xi + δi), yi)
(7)

Minimizing the training loss on RHS of Equation 7 also minimizes the adversarial loss objective of
Equation 6.

For the triggered poison, Narcissus (NS), we find the most representative patch ρ for class π given
C, defining Equation 6 with Cδ = {0 ∈RD}, Cρ = C, Π = Dπ
train, yadv = yπ, and D = DP OOD ∪
Dπ
train. In particular, this patch uses a public out-of-distribution dataset DP OOD and only the targeted
class Dπ
train. As finding this patch comes from another natural dataset and does not depend on other
train classes, NS has been more flexible to model architecture, dataset, and training regime [4].

Background for data availability attacks can be found in [35]. The goal for data availability attacks
(sometimes referred to as “unlearnable” attacks is to collapse the test accuracy, and hence the model’s
ability to generalize, or learn useful representations, from the train dataset. As we discuss in the main
paper, such attacks are immediately obvious when training a model, or rather create a region of poor
performance in the model. These attacks do not create latent vulnerabilities that can then be exploited
by an adversary. Thus we do not focus on, or investigate our methods with any detail for availability
attacks. Further, we discuss in the Societal Impacts section how such attacks have many ethical uses
for privacy and data protection 6.

A.2
Further EBM Discussions

Recalling the Gibbs-Boltzmann density from Equation 3,

pθ(x) =
1
Z(θ) exp(−Gθ(x))q(x),
(8)

where x ∈X ⊂RD denotes an image signal, and q(x) is a reference measure, often a uniform or
standard normal distribution. Here, Gθ signifies the energy potential, parameterized by a ConvNet
with parameters θ.

The normalizing constant, or the partition function, Z(θ)
=
R
exp{−Gθ(x)}q(x)dx
=
Eq[exp(−Gθ(x))], while essential, is generally analytically intractable. In practice, Z(θ) is not
computed explicitly, as Gθ(x) sufficiently informs the Markov Chain Monte Carlo (MCMC) sampling
process.

As which α of the images are poisoned is unknown, we treat them all the same for a universal defense.
Considering i.i.d. samples xi ∼P for i = 1, . . . , n, with n sufficiently large, the sample average over
xi converges to the expectation under P and one can learn a parameter θ∗such that pθ∗(x) ≈P(x).
For notational simplicity, we equate the sample average with the expectation.

14


---Page Break---
The objective is to minimize the expected negative log-likelihood, formulated as:

L(θ) = 1

n

n
X

i=1
log pθ(xi) .= EP[log pθ(x)].
(9)

The derivative of this log-likelihood, crucial for parameter updates, is given by:

∇θL(θ) = EP [∇θGθ(x)] −Epθ [∇θGθ(x)]

.= 1

n

n
X

i=1
∇θGθ(x+
i ) −1

k

k
X

i=1
∇θGθ(x−
i ),
(10)

where solving for the critical points results in the average gradient of a batch of real images (x+
i )
should be equal to the average gradient of a synthesized batch of examples from the real images
x−
i ∼pθ(x). The parameters are then updated as θt+1 = θt + ηt∇L(θt), where ηt is the learning
rate.

In this work, to obtain the synthesized samples x−
i from the current distribution pθ(x) we use the
iterative application of the Langevin update as the Monte Carlo Markov Chain (MCMC) method,
first introduced in Equation 4:

xτ+∆τ = xτ −∆τ∇xτ Gθ(xτ) +
√

2∆τϵτ,
(11)

where ϵτ ∼N(0, ID), τ indexes the time step of the Langevin dynamics, and ∆τ is the discretization
of time [41]. ∇xGθ(x) = ∂Gθ(x)/∂x can be obtained by back-propagation. If the gradient term
dominates the diffusion noise term, the Langevin dynamics behave like gradient descent. We
implement EBM training following [48], see App. A.2.1 for details.

Algorithm 1 Data Preprocessing with PUREGEN-EBM: ΨT (x)

Require: Trained ConvNet potential Gθ(x), training images x ∈X, Langevin steps T, Time
discretization ∆τ
for τ in 1 . . . T do

Langevin Step: draw ϵτ ∼N(0, ID)

xτ+1 = xτ −∆τ∇xτ Gθ(xτ) +
√

2∆τϵτ

end for
Return: Purified set ˜X from final Langevin updates

A.2.1
EBM Training

Algorithm 2 is pseudo-code for the training procedure of a data-initialized convergent EBM. We use
the generator architecture of the SNGAN [49] for our EBM as our network architecture. Further
intuiton can be found in App. B.1.

A.3
DDPM Background

The forward process successively adds noise over a sequence of time steps, eventually resulting in
values that follow a prior distribution, typically a standard Gaussian as in:

q(xt|xt−1) = N(xt;
p

1 −βtxt−1, βtI)
(12)

where x0 ∼q(x0) be a clean image sampled from the data distribution. The forward process is
defined by a fixed Markov chain with Gaussian transitions for a sequence of timesteps t = 1, . . . , T:

The reverse process is defined as the conditional distribution of the previous variable at a timestep,
given the current one:

pθ(xt−1|xt) = N(xt−1; µθ(xt, t), Σθ(xt, t))
(13)

15


---Page Break---
Algorithm 2 ML with SGD for Convergent Learning of EBM (3)

Require: ConvNet potential Gθ(x), number of training steps J = 150000, initial weight θ1, training
images {x+
i }Ndata
i=1 , data perturbation τdata = 0.02, step size τ = 0.01, Langevin steps T = 100,
SGD learning rate γSGD = 0.00005.
Ensure: Weights θJ+1 for energy Gθ(x).

Set optimizer g ←SGD(γSGD). Initialize persistent image bank as Ndata uniform noise images.
for j=1:(J+1) do

1. Draw batch images {x+
(i)}m
i=1 from training set, where (i) indicates a randomly selected index
for sample i, and get samples X+
i = x(i) + τdataϵi, where i.i.d. ϵi ∼N(0, ID).

2. Draw initial negative samples {Y (0)
i
}m
i=1 from persistent image bank. Update {Y (0)
i
}m
i=1 with
the Langevin equation

Y (k)
i
= Y (k−1)
i
−∆τ∇Yτ fθj(Y τ−1
i
) +
√

2∆τϵi,k,

where ϵi,k ∼N(0, ID) i.i.d., for K steps to obtain samples {X−
i }m
i=1 = {Y (K)
i
}m
i=1. Update
persistent image bank with images {Y (K)
i
}m
i=1.
3. Update the weights by θj+1 = θj −g(∆θj), where g is the optimizer and

∆θj = ∂

∂θ

 
1
n

n
X

i=1
fθj(X+
i ) −1

m

m
X

i=1
fθj(X−
i )

!

is the ML gradient approximation.
end for

where βt ∈(0, 1) is a variance schedule. After T steps, xT is nearly an isotropic Gaussian distribution.
This reverse process is parameterized by a neural network and trained to de-noise a variable from
the prior to match the real data distribution. Generating from a standard DDPM involves drawing
samples from the prior, and then running the learned de-noising process to gradually remove noise
and yield a final sample.

B
PUREGEN Further Intuition

B.1
Why EBM Langevin Dynamics Purify

The theoretical basis for eliminating adversarial signals using MCMC sampling is rooted in the
established steady-state convergence characteristic of Markov chains. The Langevin update, as
specified in Equation (4), converges to the distribution pθ(x) learned from unlabeled data after an
infinite number of Langevin steps. The memoryless nature of a steady-state sampler guarantees
that after enough steps, all adversarial signals will be removed from an input sample image. Full
mixing between the modes of an EBM will undermine the original natural image class features,
making classification impossible [29]. Nijkamp et al. [2020] reveals that without proper tuning,
EBM learning heavily gravitates towards non-convergent ML where short-run MCMC samples have a
realistic appearance and long-run MCMC samples have unrealistic ones. In this work, we use image
initialized convergent learning. pθ(x) is described further by Algorithm 1 [48].

The metastable nature of EBM models exhibits characteristics that permit the removal of adversarial
signals while maintaining the natural image’s class and appearance [29]. Metastability guarantees that
over a short number of steps, the EBM will sample in a local mode, before mixing between modes.
Thus, it will sample from the initial class and not bring class features from other classes in its learned
distribution. Consider, for instance, an image of a horse that has been subjected to an adversarial
ℓ∞perturbation, intended to deceive a classifier into misidentifying it as a dog. The perturbation,
constrained by the ℓ∞-norm ball, is insufficient to shift the EBM’s recognition of the image away
from the horse category. Consequently, during the brief sampling process, the EBM actively replaces
the adversarially induced ‘dog’ features with characteristics more typical of horses, as per its learned
distribution resulting in an output image resembling a horse more closely than a dog. It is important

16


---Page Break---
to note, however, that while the output image aligns more closely with the general characteristics of a
horse, it does not precisely replicate the specific horse from the original, unperturbed image.

We use mid-run chains for our EBM defense to remove adversarial signals while maintaining image
features needed for accurate classification. The steady-state convergence property ensures adversarial
signals will eventually vanish, while metastable behaviors preserve features of the initial state. We
can see PUREGEN-EBM sample purification examples in Fig. 5 below and how clean and poisoned
sampled converge and poison perturbations are removed in the mid-run region (100-2000 steps for
us).

Figure 5: PUREGEN-EBM purification with various MCMC steps

Our experiments show that the mid-run trajectories (100-1000 MCMC steps) we use to preprocess the
dataset X capitalize on these metastable properties by effectively purifying poisons while retaining
high natural accuracy on Fϕ(x) with no training modification needed. Intuitively, one can think of
the MCMC process as a directed random walk toward a low-energy, more probable natural image
version of the original image. The mid-range dynamics stay close to the original image, but in a lower
energy manifold which removes the majority of poisoned perturbations.

B.2
PUREGEN Additional L2 Dynamics Images

Additional L2 Dynamics specifically for Narcissus ϵ = 16 are shown in Figures 6 and 7.

1500.86
4002.11
8004.94
12007.97
15009.67
200012.03
PureGEN-EBM Steps (T[0])PurifyTime(s)

0

1

2

3

4

5

6

7

L2 Distance

x
T(x) 2
x
T(x +
) 2
(x +
)
T(x +
) 2
x
T(x +
) 2 
 (x +
)
T(x +
) 2

00.00
25407.98
50441.80
75455.56 100475.76 125462.75 150474.89 175484.96 200482.75
PureGEN-DDPM Steps (T[1])PurifyTime(s)

0

1

2

3

4

5

6

7

L2 Distance

x
T(x) 2
x
T(x +
) 2
(x +
)
T(x +
) 2
x
T(x +
) 2 
 (x +
)
T(x +
) 2

Figure 6: L2 Dynamics for Narcissus ϵ = 16

17


---Page Break---
Figure 7: Energy distributions with Narcissus ϵ = 16 poison.

C
PUREGEN Extensions on Higher Power Attacks

We continue with the notation introduced in 3.3, where ΨT (x) is the random variable of a fixed
image x, and we define T = (TEBM, TDDPM, TReps) ∈R3, where TEBM represents the number of EBM
MCMC steps, TDDPM represents the number of diffusion steps, and TREPS represents the number of
times these steps are repeated.

To incorporate EBM filtering, we order D by Gθ(x) and partition the ordering based on k into
D(k)
max ∪D(1−k)
min , where D(k)
max contains k|D| datapoints with the maximal energy (where k = 1
results in purifying everything and k = 0 is traditional training). Then, with some abuse of notation,

ΨT,k(D) = ΨT (D(k)
max) ∪D(1−k)
min
(14)

To leverage the strengths of both PUREGEN-EBM and PUREGEN-DDPM, we propose PUREGEN
combinations:

1. PUREGEN-NAIVE (ΨT |TEBM>0,TDDP M>0,TReps=1): Apply a fixed number of PUREGEN-
EBM steps followed by PUREGEN-DDPM steps. While this approach does improve the
purification results compared to using either method alone, it does not fully exploit the
synergy between the two techniques.

2. PUREGEN-REPS (ΨT |TEBM>0,TDDP M>0,TReps>1): To better leverage the strengths of both
methods, we propose a repetitive combination, where we alternate between a smaller number
of PUREGEN-EBM and PUREGEN-DDPM steps for multiple iterations.

3. PUREGEN-FILT (ΨT,k|TEBM≥0,TDDP M≥0,0<k<1):
In this combination, we first use
PUREGEN-EBM to identify a percentage of the highest energy points in the dataset, which
are more likely to be samples with poisoned perturbations as shown in Fig. 1. We then
selectively apply PUREGEN-EBM or PUREGEN-DDPM purification to these high-energy
points.

We note that methods 2 and 3 require extensive hyperparameter search with performance sweeps
using the HLB model in App F, as there was little intuition for the amount of reps (TReps) or the
filtering threshold (k) needed. Thus, we do not include these methods in our core results, but instead
show the added performance gains on higher power poisons in Table 4, both in terms of increased
perturbation size ϵ = 16 and increased poison % (and both together). We note that 10% would mean
the adversary has poisoned the entire class in CIFAR-10 with an NS trigger, and ϵ = 16 is starting to
approach visible perturbations, but both are still highly challenging scenarios worth considering for
purification.

18


---Page Break---
D
Poison Sourcing and Experiment Implementation Details

Triggerless attacks GM and BP poison success refers to the number of single-image targets success-
fully flipped to a target class (with 50 or 100 target image scenarios) while the natural accuracy is
averaged across all target image training runs. Triggered attack Narcissus poison success is measured
as the number of non-class samples from the test dataset shifted to the trigger class when the trigger is
applied, averaged across all 10 classes, while the natural accuracy is averaged across the 10 classes on
the un-triggered test data. We include the worst-defended class poison success. The Poison Success
Rate for a single experiment can be defined for triggerless PSRnotr and triggered PSRtr poisons
as:

PSRnotr(F, i) = 1F (xπ
i )=yadv
i
(15)

PSRtr(F, yπ) =

P
(x,y)∈Dtest\Dπ
test 1F (x+ρπ)=yπ

|Dtest \ Dπ
test|
(16)

Note that all results except for Poison Success Rate for GM and BP attacks have a standard deviation,
since those attacks are based on a single image class flip for a single classifiers training run. We do
provide a subset with results for a single poison paradigm of BP and GM in App. E.2.3 where we
used 3 different seeds for the training 3 classifiers for each of the 50 and 100 runs respectively to
get a standard deviation, showing these results are low variance relative to the difference in results
between baselines and our method. The compute required to collect such results across all scenarios
would be extensive. .

D.1
Poison Sourcing

D.1.1
Bullseye Polytope

The Bullseye Polytope (BP) poisons are sourced from two distinct sets of authors. From the original
authors of BP [9], we obtain poisons crafted specifically for a black-box scenario targeting ResNet18
and DenseNet121 architectures, and grey-box scenario for MobileNet (used in poison crafting). These
poisons vary in the percentage of data poisoned, spanning 1%, 2%, 5% and 10% for the linear-transfer
mode and a single 1% fine-tune mode for all models over a 500 image transfer dataset. Each of these
scenarios has 50 datasets that specify a single target sample in the test-data. We also use a benchmark
paper that provides a pre-trained white-box scenario on CIFAR-100 [50]. This dataset includes 100
target samples with strong poison success, but the undefended natural accuracy baseline is much
lower.

D.1.2
Gradient Matching

For GM, we use 100 publicly available datasets provided by [8]. Each dataset specifies a single target
image corresponding to 500 poisoned images in a target class. The goal of GM is for the poisons to
move the target image into the target class, without changing too much of the remaining test dataset
using gradient alignment. Therefore, each individual dataset training gives us a single datapoint of
whether the target was correctly moved into the poisoned target class and the attack success rate is
across all 100 datasets provided.

D.1.3
Narcissus

For Narcissus triggered attack, we use the same generating process as described in the Narcissus
paper, we apply the poison with a slight change to more closely match with the baseline provided
by [50]. We learn a patch with ϵ = 8/255 on the entire 32-by-32 size of the image, per class, using
the Narcissus generation method. We keep the number of poisoned samples comparable to GM for
from-scratch experiment, where we apply the patch to 500 images (1% of the dataset) and test on
the patched dataset without the multiplier. In the fine-tune scenarios, we vary the poison% over
1%, 2.5%, and 10%, by modifying either the number of poisoned images or the transfer dataset size
(specifically 20/2000, 50/2000, 50/500 poison/train samples).

19


---Page Break---
D.1.4
Neural Tangent Availability Attacks

For Neural Tangent Availability Attack, the full NTGA dataset (all samples poisoned) is sourced from
the authors of the original NTGA attack paper [12]. Baseline defenses are pull from AVATAR [30].

D.2
Training Parameters

We follow the training hyperparameters given by [18, 4, 9, 50] for GM, NS, BP Black/Gray-Box,
and BP White-Box respectively as closely as we can, with moderate modifications to align poison
scenarios. HyperlightBench training followed the original creators settings and we only substituted in
a poisoned dataloader [46].

Parameter
From Scratch
Transfer Linear
Transfer Fine-Tune

PUREGEN-EBM Steps (TEBM)
150
500
1000
PUREGEN-DDPM Steps (TDDP M)
75
125
125
PUREGEN-REPS (TReps)
7
-
-
PUREGEN-FILT (k)
0.5
-
-

Device Type
TPU-V3
TPU-V3
TPU-V3
Weight Decay
5e-4
5e-4
5e-4
Batch Size
128
64
128
Augmentations
RandomCrop(32, padding=4)
None
None
Epochs
200 or 80
40
60
Optimizer
SGD(momentum=0.9)
SGD
Adam
Learning Rate
0.1
0.1
0.0001
Learning Rate Schedule
(Multi-Step Decay)

100, 150 - 200 epochs
30, 50, 70 - 80 epochs
15, 25, 35
15, 30, 45

Reinitialize Linear Layer
NA
True
True

D.3
Core Results Compute

Training compute for core result only which is in Table 1 on a TPU V3.

Table 6: Compute Hours TPU V3

From Scratch
Transfer
Total
Narcissus
Gradient Matching
NTGA
Fine Tune BP BlackBox
Fine Tune Narc
Linear BP BlackBox
Linear BP WhiteBox

Train Time
(Hours)
1959.91
4155.74
283.75
73.82
45.89
548.15
70.76
7138.03

E
Additional Results

E.1
Extended Core Results

E.1.1
Full Results Primary Experiments

Results on all primary poison scenarios with ResNet18 classifier including all EPIC versions (vari-
ous subset sizes and selection frequency), FRIENDS versions (bernouilli or gaussian added noise
trasnform), and all natural JPEG versions (compression ratios). Green highlight indicates a base-
line defense that was selected for the main paper results table chosen by the best poison defense
performance that did not result in significant natural accuracy degradaion. For both TinyImageNet
and CINIC-10 from-scratch results, best performing baseline settings were used from respective
poison scenarios in CIFAR-10 for compute reasons (so there are no additonal results and hence they
are removed from this table).

20


---Page Break---
From Scratch

CIFAR-10 (ResNet-18)

Gradient Matching-1%
Narcissus-1%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

None
44.00
94.840.2
43.9533.6
94.890.2
93.59

EPIC-0.1
34.00
91.270.4
30.1832.2
91.170.2
81.50
EPIC-0.2
21.00
88.040.7
32.5033.5
86.890.5
84.39
EPIC-0.3
10.00
85.141.2
27.3134.0
82.201.1
84.71

FRIENDS-B
1.00
91.160.4
8.3222.3
91.010.4
71.76
FRIENDS-G
0.00
91.150.4
9.4925.9
91.060.2
83.03

JPEG-25
0.00
90.000.19
1.670.88
90.150.26
3.38
JPEG-50
0.00
91.700.18
1.700.98
91.830.20
3.83
JPEG-75
2.00
92.730.20
1.781.17
92.940.15
4.13
JPEG-85
5.00
93.430.16
5.7613.24
93.430.20
43.36

PUREGEN-DDPM
0.00
90.930.20
1.640.82
90.990.22
2.83
PUREGEN-EBM
1.00
92.980.2
1.390.8
92.920.2
2.50

Transfer Learning (CIFAR-10, ResNet-18)

Fine-Tune
Linear - Bullseye Polytope

Bullseye Polytope-10%
Narcissus-10%
BlackBox-10%
WhiteBox-1%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
46.00
89.840.9
33.4133.9
90.142.4
98.27
93.75
83.592.4
98.00
70.090.2
EPIC-0.1
50.00
89.001.8
32.4033.7
90.022.2
98.95
91.67
83.482.9
98.00
69.350.3
EPIC-0.2
42.00
81.955.6
20.9327.1
88.582.0
91.72
66.67
84.343.8
91.00
64.790.7
EPIC-0.3
44.00
86.756.3
28.0134.9
84.366.3
99.91
66.67
83.233.8
63.00
60.861.5
FRIENDS-B
8.00
87.801.1
3.345.7
89.620.5
19.48
35.42
84.972.2
19.00
60.850.6
FRIENDS-G
8.00
87.821.2
3.045.1
89.810.5
17.32
33.33
85.182.3
19.00
60.900.6
JPEG-25
0.00
88.930.66
2.953.71
87.630.49
12.55
0.00
92.440.47
8.0
50.420.73
JPEG-50
0.00
90.400.44
3.514.64
88.410.58
15.76
0.00
86.032.23
16.0
53.490.54
JPEG-75
4.00
90.110.78
18.2825.83
89.120.51
86.39
16.67
84.231.76
36.0
56.020.50
JPEG-85
5.00
93.430.16
25.1931.44
88.631.59
94.41
54.17
83.611.36
51.0
58.080.54
PUREGEN-DDPM
0.00
91.530.15
1.881.12
90.690.26
3.42
0.00
93.810.08
9.0
54.530.64
PUREGEN-EBM
0.00
87.521.2
2.021.0
89.780.6
3.85
0.00
92.380.3
6.00
64.980.3

E.1.2
From Scratch 80 Epochs Experiments

Baseline FRIENDS [26] includes an 80-epoch from-scratch scenario to show poison defense on a
faster training schedule. None of these results are included in the main paper, but we show again
SoTA or near SoTA for PUREGEN against all baselines (and JPEG is again introduced as a baseline).

Table 7: From-Scratch 80-Epochs Results (ResNet-18, CIFAR-10)

From Scratch (80 - Epochs)

Gradient Matching-1%
Narcissus-1%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

None
47.00
93.790.2
32.5130.3
93.760.2
79.43
EPIC-0.1
27.00
90.870.4
24.1530.1
90.920.4
79.42
EPIC-0.2
28.00
91.020.4
23.7529.2
89.720.3
74.28
EPIC-0.3
44.00
92.460.3
21.5328.8
88.051.1
80.75

FRIENDS-B
2.00
90.070.4
1.420.8
90.060.3
2.77
FRIENDS-G
1.00
90.090.4
1.370.9
90.010.2
3.18

JPEG-25
1.00
88.730.24
1.660.92
90.010.20
3.18
JPEG-50
2.00
90.550.23
1.761.07
90.560.28
3.67
JPEG-75
0.00
91.690.23
1.681.14
91.670.23
3.79
JPEG-85
4.00
92.310.23
1.871.41
92.420.16
5.13

PUREGEN-DDPM
1.00
89.820.26
1.540.82
90.000.13
2.52
PUREGEN-EBM
1.00
92.020.2
1.520.8
92.020.3
2.81

21


---Page Break---
E.1.3
Full Results for MobileNetV2 and DenseNet121

Table 8: MobileNetV2 Full Results

From Scratch
Transfer Learning

Gradient Matching-1%
Narcissus-1%
Fine-Tune Narcissus-10%
Linear BP BlackBox-10%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
20.00
93.860.2
32.7024.5
93.920.1
73.97
23.5923.2
88.301.2
66.54
81.25
73.271.0
EPIC-0.1
37.50
91.280.2
40.0927.1
91.150.2
79.74
23.2522.8
88.351.0
65.97
81.25
69.782.0
EPIC-0.2
19.00
91.240.2
38.5527.5
87.650.5
74.72
19.9519.2
87.671.3
50.05
56.25
54.475.6
EPIC-0.3
9.78
87.801.6
22.3523.9
78.169.9
69.52
21.7028.1
78.176.0
74.96
58.33
58.749.0
FRIENDS-B
6.00
84.302.7
2.001.3
88.820.6
4.88
2.211.5
83.050.7
5.63
41.67
68.861.5
FRIENDS-G
5.00
88.840.4
2.051.7
88.930.3
6.33
2.201.4
83.040.7
5.42
47.92
68.941.5
JPEG-25
1.00
85.180.31
2.431.16
85.000.24
4.40
6.289.05
80.001.04
29.25
2.08
73.140.71
JPEG-50
1.00
86.820.34
2.301.20
86.600.13
3.99
6.7610.23
83.700.94
33.34
12.50
76.121.75
JPEG-75
1.00
88.080.34
2.461.42
87.880.23
4.87
13.8418.74
84.671.41
52.96
68.75
73.111.53
JPEG-85
2.00
88.830.31
10.0316.93
88.610.34
52.46
14.9318.42
85.461.31
56.82
77.08
71.991.07
PUREGEN-EBM
1.00
90.930.2
1.640.8
91.750.1
2.91
3.665.4
84.180.5
18.85
0.00
78.571.4
PUREGEN-DDPM
1.00
86.790.26
2.131.02
86.910.23
3.74
3.414.82
86.920.39
16.79
0.00
83.140.20

Table 9: DenseNet121 Full Results

From Scratch
Transfer Learning

Gradient Matching-1%
Narcissus-1%
Fine-Tune Narcissus-10%
Linear BP BlackBox-10%

Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Poison
Success (%) ↓
Avg Nat
Acc (%) ↑

None
14.00
95.300.1
46.5232.2
95.330.1
91.96
56.5238.6
87.032.8
99.56
73.47
82.131.6
EPIC-0.1
14.00
93.00.3
43.3832.0
93.070.2
88.97
53.9739.0
87.042.8
99.44
62.50
78.882.1
EPIC-0.2
7.00
90.670.5
41.9733.2
90.230.6
86.85
43.6636.5
85.972.6
97.17
41.67
70.135.2
EPIC-0.3
4.00
88.31.0
32.6029.4
85.122.4
71.50
43.2443.0
72.7610.8
100.00
66.67
70.2010.1
FRIENDS-B
1.00
91.330.4
8.6021.2
91.550.3
68.57
5.349.9
88.620.8
33.42
60.42
80.221.9
FRIENDS-G
1.00
91.330.4
10.1325.2
91.320.4
81.47
5.5510.4
88.750.6
34.91
56.25
80.121.8
JPEG-25
0.00
90.090.17
1.681.10
90.150.26
3.62
2.462.92
83.820.81
9.79
0.00
78.671.60
JPEG-50
0.00
91.940.20
1.901.54
92.030.22
5.41
3.072.69
85.921.07
8.70
12.50
84.241.39
JPEG-75
0.00
93.080.19
2.733.12
93.160.07
8.88
32.2135.67
87.191.28
98.64
27.08
81.311.34
JPEG-85
7.00
93.850.19
13.3628.10
93.770.27
90.77
38.7037.98
86.252.29
97.19
70.83
80.571.40
PUREGEN-EBM
0.00
92.850.2
1.420.7
93.480.1
2.60
2.481.9
88.750.5
7.41
0.00
89.290.9
PUREGEN-DDPM
3.00
91.090.24
1.710.94
90.940.23
2.97
2.792.51
88.210.62
8.95
0.00
89.020.15

E.1.4
PUREGEN Combos on Increased Poison Power

Table 10: PUREGEN Combos with Narcissus Increased Poison % and ϵ

Narcissus ϵ = 8 1%
Narcissus ϵ = 8 10%
Narcissus ϵ = 16 1%
Narcissus ϵ = 16 10%

Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

None
40.3338.63
93.610.12
90.26
96.276.62
84.570.60
99.97
83.6312.09
93.670.11
97.36
99.350.81
84.580.63
99.97

EPIC-0.1
36.3934.52
92.020.63
87.34
98.632.06
83.120.69
99.98
74.8113.75
92.220.33
94.26
98.813.04
82.890.59
99.96
EPIC-0.2
29.7531.82
88.370.37
82.41
96.653.73
79.750.94
99.78
73.1114.02
88.240.59
91.51
98.541.43
79.730.93
99.57
EPIC 03
28.5333.15
83.001.87
93.68
93.5412.99
76.311.54
99.99
56.1617.91
82.941.53
81.59
97.403.51
75.951.59
99.97

FRIENDS-G
10.0126.92
91.180.30
86.53
32.4043.26
84.951.70
99.89
18.9020.93
91.150.34
71.46
73.5620.71
82.820.42
97.31
FRIENDS-B
7.8920.37
91.250.26
65.78
30.8741.67
85.171.70
100.00
18.0616.88
91.160.29
54.17
71.2822.90
82.830.43
99.25

JPEG-25
1.820.98
87.760.15
3.46
16.959.72
84.661.51
33.96
11.8512.60
87.720.19
36.90
79.288.05
79.870.79
91.99
JPEG-50
1.751.07
89.350.27
3.71
31.1414.99
84.151.63
50.97
21.6717.81
89.320.19
57.58
90.486.66
81.140.86
100.00
JPEG-75
1.660.92
90.700.14
3.30
56.9927.60
84.071.51
99.53
34.0621.16
90.770.13
65.14
92.006.90
82.200.73
99.86
JPEG-85
4.679.82
91.740.12
32.52
69.7525.47
83.721.39
100.00
42.2126.43
91.660.24
82.82
93.226.90
82.900.53
99.83

PUREGEN-DDPM
1.720.92
89.610.14
3.20
6.385.16
85.860.46
16.29
5.213.35
86.160.19
13.32
69.3816.73
83.581.02
89.35
PUREGEN-EBM
1.590.86
91.410.16
3.01
52.4823.29
86.141.82
99.86
7.354.46
85.610.25
16.94
77.507.01
78.790.93
90.84
PUREGEN-NAIVE T =[150,75,1]
1.710.89
88.840.16
3.17
10.438.58
88.200.54
27.42
5.202.61
85.950.23
9.80
63.0115.24
83.140.90
87.17
PUREGEN-REPS T =[10,50,5]
1.730.84
87.250.18
3.25
3.752.28
85.560.22
7.74
4.952.48
85.790.18
10.75
53.7917.14
83.921.02
81.09
PUREGEN-FILT T =[0,125,1],k=0.5
1.730.89
90.610.16
2.88
6.476.98
86.082.00
18.81
5.744.05
90.520.18
16.08
69.1312.94
85.471.45
87.66

E.2
Additional Experiment Results

E.2.1
Defense/EBM-Aware Narcissus Experiments

To further demonstrate the robustness of PUREGEN, we conduct an additional experiment simulating
a scenario where an adversary is aware that PUREGEN is in use. Specifically, we incorporate the
Energy-Based Model (EBM) within the Narcissus poison generation framework. Narcissus generates
poisons by training a surrogate model, then refining the poison through SGD on frozen surrogate
outputs (further details in [4]). For simplicity, we assume that the adversary possesses knowledge of
the EBM defense mechanism but is unaware of the specific image instances ultimately used during
training.

In this experiment, we include our pre-trained EBM as a preprocessing layer for the ResNet-based
surrogate model, using 50 EBM steps—–a choice that balances computational efficiency with reliable
performance. When training the EBM-aware surrogate, we freeze the EBM to allow the classifier to
train on a diverse set of stochastic EBM outputs. After a brief warmup phase, the entire surrogate

22


---Page Break---
model is frozen, and we propagate gradients to optimize poison effectiveness. We test variations of
when the surrogate and poison generation do or do not have EBM information, applied to 3 of the 10
CIFAR-10 classes. The results are recorded in Table 11.

Our results show that PUREGEN consistently defends against all new EBM-aware Narcissus poisons,
demonstrating the robustness of our approach. Interestingly, we observe that when both the surrogate
and poison-generating models include the EBM, the Narcissus method struggles to produce effective
poisons, even in an undefended model context. Conversely, with a traditionally trained surrogate
model but EBM augmentation in the poison generation process, the new poisons slightly improve
efficacy on an undefended model. Overall, these findings reinforce the reliability of PUREGEN as a
robust defense mechanism against adaptive poisoning attacks.

Table 11: Success Rates of EBM-Aware Narcissus Poisons Against PUREGEN Defense on classes
0,1,2. This table shows that even with the knowledge of the EBM, Narcissus is unable to generate
effective poisons against PUREGEN. Furthermore, when the EBM is taken in the entire Narcissus
process, Narcissus struggles to find a poison that works for undefended models.

Narcissus Baseline
Narcissus EBM Generation Model

Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

No Defense
31.0249.32
93.640.22
87.94
46.8237.92
93.360.14
90.52
PUREGEN-DDPM
1.490.83
91.120.10
2.36
1.060.53
91.100.09
1.64
PUREGEN-EBM
1.570.82
91.480.23
2.37
1.290.59
91.430.15
1.83

Narcissus EBM Surrogate Model
Narcissus EBM Surrogate and Generation Model

Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓
Avg Poison
Success (%) ↓
Avg Nat
Acc (%) ↑
Max Poison
Success (%) ↓

No Defense
20.0832.55
93.570.17
57.67
3.042.46
93.650.12
5.74
PUREGEN-DDPM
1.461.06
91.040.17
2.64
1.421.03
91.120.11
2.60
PUREGEN-EBM
1.561.03
91.360.23
2.73
1.590.82
91.290.15
2.50

E.2.2
Pre-Trained Public DDPM Comparison

We include results using two pre-trained diffusion models from HuggingFace [51, 52]. The results
show that these models can achieve defense performance similar to that of some of our POOD
in-house trained models. The table below includes 4 baseline PUREGEN models and the two Hugging
Face models trained on butterflies and anime datasets, showing both are comparable for poison
defense and natural accuracy to some POOD datasets in performance. These results how that, for
PUREGEN-DDPM pre-trained models could be adequate, but come with the risks of using a
model with unknown data security. We reiterate our primary contribution for PUREGEN-DDPM was
in reducing training and improving performance for a given architecture and dataset if one needs to
train a diffusion model and if purification is the known use-case.

Table 12: Two pre-trained diffusion models from HuggingFace, showing similar results to our POOD
DDPM results on Narcissus From-Scratch attack [51, 52].

DDPM Model
Poison Success (%)
Nat Acc (%)
Max Poison (%)

PUREGEN-EBM CINIC-10_IN
1.39 ± 0.80
92.92 ± 0.20
2.50
PUREGEN-DDPM CINIC-10_IN
1.64 ± 0.82
90.99 ± 0.22
3.83
PUREGEN-DDPM Food-101
1.71 ± 0.74
88.35 ± 0.21
2.72
PUREGEN-DDPM Office-Home
1.80 ± 0.83
87.32 ± 0.22
3.16

HuggingFace Butterflies [51]
1.65 ± 0.83
87.79 ± 0.18
3.01
HuggingFace Anime [52]
1.47 ± 0.75
90.91 ± 0.13
2.95

23


---Page Break---
E.2.3
GM and BP Poison Success Standard Deviation

Table 13: Core results poison success for one GM and one BP scenario where we compute poison
success across 3 different seeds to show the relatively low variance of these results where our method
is still SoTA.

Poison Success (%)

Poison Scenario
Baseline
EPIC
FRIENDS
JPEG
PUREGEN-EBM
PUREGEN-DDPM

From-Scratch, GM-1%, CIFAR-10 (ResNet-18)
44.442.17
10.584.17
0.330.47
0.670.47
1.000.00
0.000.00
Fine-Tune, BP-10%, CIFAR-10 (ResNet-18)
48.636.18
44.675.73
7.194.8
0.670.94
0.00.0
0.670.94

24


---Page Break---
F
PUREGEN Extensions T Sweeps

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

91.17%
91.06%
90.73%
90.37%
90.35%

90.79%
90.36%
90.21%
89.79%
89.60%

90.37%
90.08%
89.66%
89.39%
88.86%

90.05%
89.66%
89.14%
88.81%
88.54%

89.77%
89.34%
88.76%
88.31%
87.97%

89.37%
89.15%
88.48%
87.88%
87.34%

89.04%
88.91%
88.01%
87.48%
86.88%

88.65%
88.42%
87.56%
87.19%
86.56%

DDPM-25 Nat Acc

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

33.34%
28.98%
26.54%
21.82%
22.84%

24.20%
23.60%
17.35%
16.28%
14.20%

21.31%
20.85%
15.46%
12.40%
10.73%

15.57%
14.19%
12.39%
10.07%
10.08%

14.85%
12.89%
11.15%
10.10%
9.33%

11.09%
10.37%
9.65%
8.82%
8.73%

11.61%
10.40%
8.87%
8.19%
7.93%

10.50%
8.88%
8.31%
8.87%
7.95%

DDPM-25 Poison Success

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

71.75%
71.58%
66.33%
68.11%
65.11%

62.68%
64.48%
41.80%
42.03%
27.55%

56.68%
57.31%
31.39%
23.00%
22.09%

29.00%
39.49%
24.27%
19.12%
19.74%

29.24%
26.22%
22.12%
23.17%
20.13%

21.79%
22.12%
18.11%
19.01%
18.44%

25.90%
18.93%
17.57%
17.18%
16.32%

24.85%
16.90%
19.35%
20.06%
16.90%

DDPM-25 Max Poison Success

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

90.57%
90.42%
90.01%
90.05%
89.75%

89.98%
89.73%
89.38%
89.21%
88.79%

89.26%
89.12%
88.60%
88.47%
88.02%

89.12%
88.65%
88.34%
87.92%
87.62%

88.43%
88.17%
87.81%
87.34%
87.08%

88.09%
87.72%
87.21%
86.92%
86.64%

87.61%
87.44%
86.96%
86.48%
85.99%

87.38%
86.90%
86.43%
86.10%
85.47%

DDPM-35 Nat Acc

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

22.23%
19.26%
15.87%
18.87%
14.34%

12.38%
12.76%
11.61%
11.49%
11.12%

11.32%
10.93%
9.83%
8.97%
9.00%

9.06%
8.97%
8.95%
7.68%
8.00%

8.94%
8.48%
7.71%
7.60%
7.42%

7.39%
7.36%
7.41%
6.89%
6.93%

7.09%
7.08%
6.70%
6.84%
6.83%

7.01%
6.50%
6.36%
5.70%
6.04%

DDPM-35 Poison Success

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

60.51%
50.78%
40.07%
55.87%
37.82%

24.77%
24.78%
24.11%
23.28%
22.72%

22.29%
21.66%
21.05%
21.54%
18.60%

25.09%
17.17%
21.45%
17.52%
16.41%

22.32%
16.09%
16.60%
15.50%
14.34%

15.95%
16.93%
15.77%
17.02%
16.88%

15.78%
16.56%
13.44%
16.20%
13.21%

20.24%
14.60%
13.07%
12.15%
13.44%

DDPM-35 Max Poison Success

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

89.36%
89.27%
89.12%
89.11%
88.81%

88.63%
88.38%
88.23%
87.85%
87.74%

87.80%
87.48%
87.38%
86.99%
86.69%

87.02%
86.86%
86.68%
86.37%
86.44%

86.24%
86.23%
85.76%
85.64%
85.40%

85.98%
85.74%
85.29%
85.00%
84.67%

85.43%
85.08%
84.90%
84.42%
84.28%

84.90%
84.79%
84.22%
84.03%
83.73%

DDPM-50 Nat Acc

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

10.04%
9.65%
8.56%
8.97%
9.18%

8.05%
7.56%
7.66%
7.85%
7.45%

6.81%
6.85%
6.28%
6.09%
6.30%

5.93%
6.07%
5.91%
5.80%
5.63%

6.04%
6.05%
5.38%
5.36%
5.15%

5.20%
4.93%
5.09%
5.21%
5.49%

5.46%
4.84%
5.12%
5.11%
4.76%

5.05%
5.02%
5.09%
4.86%
4.67%

DDPM-50 Poison Success

10
25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

23.23%
23.65%
19.29%
18.91%
24.78%

19.39%
17.15%
21.01%
25.00%
15.74%

16.07%
15.66%
13.97%
14.23%
13.36%

13.35%
13.79%
15.09%
10.80%
13.93%

13.29%
14.12%
12.31%
11.20%
8.99%

11.74%
8.19%
10.90%
11.00%
10.56%

11.44%
10.32%
12.50%
9.51%
8.04%

12.21%
12.29%
9.52%
9.25%
8.61%

DDPM-50 Max Poison Success

25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

87.16%
86.98%
86.92%
86.89%

85.95%
85.48%
85.72%
85.50%

84.56%
84.42%
84.34%
84.47%

83.82%
83.63%
83.57%
83.43%

83.10%
82.73%
82.60%
82.51%

82.12%
82.08%
81.75%
81.64%

81.53%
81.43%
81.36%
80.81%

81.07%
80.77%
80.64%
80.29%

DDPM-75 Nat Acc

25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

5.92%
5.95%
6.24%
5.79%

5.34%
5.44%
5.20%
5.34%

4.95%
4.70%
4.54%
4.50%

4.19%
4.50%
4.50%
4.34%

4.45%
4.42%
4.38%
4.55%

4.20%
4.10%
4.37%
4.05%

4.32%
4.32%
4.11%
4.41%

3.98%
3.98%
4.09%
3.83%

DDPM-75 Poison Success

25
50
75
100
MCMC Steps

2
3
4
5
6
7
8
9
Purify Reps

15.48%
13.53%
15.78%
13.71%

12.71%
12.72%
9.85%
12.22%

10.61%
8.06%
7.89%
8.15%

7.68%
8.11%
9.53%
8.64%

9.04%
8.04%
7.44%
8.35%

8.15%
7.74%
7.52%
8.25%

8.26%
8.90%
7.80%
7.99%

7.83%
7.03%
8.27%
7.13%

DDPM-75 Max Poison Success

0.87

0.88

0.89

0.90

0.91

0.10

0.15

0.20

0.25

0.30

0.2

0.3

0.4

0.5

0.6

0.7

0.86

0.87

0.88

0.89

0.90

0.06

0.08

0.10

0.12

0.14

0.16

0.18

0.20

0.22

0.2

0.3

0.4

0.5

0.6

0.84

0.85

0.86

0.87

0.88

0.89

0.05

0.06

0.07

0.08

0.09

0.10

0.10

0.12

0.14

0.16

0.18

0.20

0.22

0.24

0.81

0.82

0.83

0.84

0.85

0.86

0.87

0.040

0.045

0.050

0.055

0.060

0.08

0.09

0.10

0.11

0.12

0.13

0.14

0.15

Figure 8: PUREGEN-REPS Sweeps with HLB Model on Narcissus

25


---Page Break---
150
250
400
600
800
1000
1500
2000
MCMC Steps

25
50
75
100
125
150
Diffusion Steps

90.85% 90.49% 89.74% 89.15% 88.53% 87.97%

89.94% 89.60% 89.08% 88.44% 88.28% 87.60%

88.69% 88.52% 88.07% 87.67%
86.02% 85.29%

87.30% 87.03% 86.68% 86.57%
85.06% 84.45%

85.31% 84.81%
83.96% 83.27%

83.63% 83.58%
82.65% 82.19%

Full DDPM-Nat Acc

150
250
400
600
800
1000
1500
2000
MCMC Steps

25
50
75
100
125
150
Diffusion Steps

27.44% 21.09% 19.42% 12.38% 10.71% 10.11%

15.84% 14.16% 10.59% 10.25% 10.04% 10.02%

8.39%
8.26%
8.81%
8.00%
7.17%
7.46%

6.95%
6.57%
6.61%
6.36%
5.87%
6.04%

5.46%
5.45%
5.56%
5.46%

5.41%
5.06%
5.37%
4.93%

Full DDPM-Poison Success

150
250
400
600
800
1000
1500
2000
MCMC Steps

25
50
75
100
125
150
Diffusion Steps

68.87% 66.07% 61.97% 27.84% 21.56% 18.87%

51.13% 32.69% 22.01% 24.36% 20.32% 20.75%

19.64% 17.99% 21.71% 14.33%
16.12% 15.41%

14.22% 14.44% 16.00% 11.99%
13.92% 13.55%

11.50%
9.71%
10.39%
9.66%

11.72% 10.58%
10.26%
8.94%

Full DDPM-Max Poison Success

0.83

0.84

0.85

0.86

0.87

0.88

0.89

0.90

0.05

0.10

0.15

0.20

0.25

0.1

0.2

0.3

0.4

0.5

0.6

Figure 9: PUREGEN-NAIVE Sweeps with HLB Model on Narcissus

150.0
500.0
750.0
1000.0
1250.0
1500.0
2000.0
MCMC Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

92.91% 92.64% 92.65% 92.58% 92.47% 92.54% 92.42%

92.62% 92.25% 92.24% 91.99% 91.99% 91.91% 91.76%

92.41% 91.81% 91.67% 91.53% 91.47% 91.35% 90.89%

92.31% 91.51% 91.19% 90.99% 90.87% 90.56% 90.40%

92.09% 91.09% 90.59% 90.44% 90.16% 89.88% 89.54%

91.86% 90.72% 90.22% 89.86% 89.58% 89.26% 88.84%

91.64% 90.41% 89.62% 89.35% 89.01% 88.64% 88.01%

91.53% 89.96% 89.36% 88.91% 88.31% 87.94% 87.20%

91.31% 89.71% 88.95% 88.52% 87.77% 87.36% 86.28%

Nat Acc

150.0
500.0
750.0
1000.0
1250.0
1500.0
2000.0
MCMC Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

48.98% 31.47% 30.05% 24.43% 23.67% 21.69% 21.13%

45.01% 28.11% 22.93% 17.72% 17.09% 15.71% 16.97%

38.90% 23.51% 18.59% 13.74% 13.89% 12.48% 11.40%

33.62% 19.07% 15.56% 11.79% 10.43% 10.53%
9.53%

29.46% 18.71% 13.99% 10.58%
9.04%
8.99%
8.85%

26.56% 16.81% 10.92%
9.26%
9.75%
8.47%
7.46%

26.18% 13.54%
9.83%
9.60%
8.80%
8.16%
7.16%

23.74% 13.18% 11.33%
8.38%
8.46%
7.72%
7.17%

21.94% 14.38% 10.93%
8.43%
7.94%
7.94%
7.86%

Poison Success

150.0
500.0
750.0
1000.0
1250.0
1500.0
2000.0
MCMC Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

90.03% 58.20% 54.36% 45.86% 46.02% 34.90% 38.27%

90.33% 64.55% 43.49% 39.43% 34.94% 32.16% 32.87%

65.91% 57.95% 51.05% 32.32% 28.68% 25.04% 21.91%

60.63% 53.13% 43.50% 28.83% 18.86% 19.69% 18.70%

61.03% 57.47% 43.32% 26.88% 15.45% 15.42% 16.71%

58.12% 52.57% 22.70% 19.87% 20.91% 15.73% 12.06%

59.32% 40.11% 21.12% 16.89% 15.49% 13.91% 12.99%

63.19% 42.15% 40.42% 15.88% 16.62% 13.94% 14.74%

59.73% 51.37% 37.23% 15.17% 17.19% 17.39% 17.29%

Max Poison Success

75.0
100.0
125.0
Diffusion Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

92.74%
92.64%
92.52%

92.29%
92.14%
92.02%

92.06%
91.67%
91.43%

91.73%
91.20%
90.71%

91.22%
90.53%
89.96%

90.96%
90.02%
89.19%

90.42%
89.44%
88.42%

90.01%
88.88%
87.41%

89.61%
88.15%
86.69%

Nat Acc

75.0
100.0
125.0
Diffusion Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

20.91%
13.28%
8.74%

14.76%
8.98%
7.46%

12.81%
7.26%
5.80%

11.12%
6.51%
5.12%

9.44%
6.30%
5.37%

8.92%
6.15%
4.89%

8.45%
6.25%
4.70%

8.63%
6.37%
4.68%

7.83%
5.90%
5.10%

Poison Success

75.0
100.0
125.0
Diffusion Steps

0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Filter

40.54%
28.12%
17.23%

32.50%
20.57%
16.33%

23.73%
16.19%
10.93%

20.84%
15.13%
11.97%

20.03%
12.64%
10.92%

16.81%
13.07%
10.20%

17.37%
13.43%
8.92%

19.23%
15.76%
8.87%

14.28%
12.63%
9.59%

PureGen-FiltEBM

PureGen-FiltDDPM

Max Poison Success

0.87

0.88

0.89

0.90

0.91

0.92

0.10

0.15

0.20

0.25

0.30

0.35

0.40

0.45

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

0.87

0.88

0.89

0.90

0.91

0.92

0.06

0.08

0.10

0.12

0.14

0.16

0.18

0.20

0.10

0.15

0.20

0.25

0.30

0.35

0.40

Figure 10: PUREGEN-FILT Sweeps with HLB Model on Narcissus

G
Interpreting PUREGEN Results

G.1
Model Interpretability

Using the Captum interpretability library, in Figure 11, we compare a clean model with clean data to
various defense techniques on a sample image poisoned with the NS Class 5 trigger ρ [53]. Only the
clean model and the model that uses PUREGEN-EBM correctly classify the sample as a horse, and
the regions most important to prediction, via occlusion analysis, most resemble the shape of a horse
in the clean and PUREGEN-EBM images. Integrated Gradient plots show how PUREGEN-EBM
actually enhances interpretability of relevant features in the gradient space for prediction compared
to even the clean NN.

26


---Page Break---
Clean

Image

Clean
prediction: horse

Poisoned

Image

None
prediction: dog

EPIc
prediction: dog

FrieNDs
prediction: dog

PureEBM
prediction: horse

Occlusions
Integrated

Gradients

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

Clean

Image

Clean
prediction: frog

Poisoned

Image

No Defense
prediction: dog

Epic
prediction: dog

Friendly
prediction: dog

EBM
prediction: frog

Occlusions
Integrated

Gradients

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

0.0
0.5
1.0

Figure 11: Defense Interpretability: Model using PUREGEN-EBM focuses on the outline of the
horse in the occlusions analysis and to a higher degree on the primary features in the gradient space
than even the clean model on clean data.

27


---Page Break---
G.2
Differences between PUREGEN-EBM and PUREGEN-DDPM

In this section we visualize a Narcissus ϵ = 64 trigger patch to better see the PUREGEN-EBM and
PUREGEN-DDPM samples on a visible perturbation. In Figure 12 we see again how the EBM
struggles to purify larger perturbations but better preserves the image content, while DDPM can
degrade such perturbations better at the cost of degrading the image content as well.

Figure 12: Narcissus ϵ = 64 trigger patch purification samples Top Left: Original Poisoned. Top
Right: PUREGEN-EBM 500 Steps. Top Left: PUREGEN-DDPM 75 Steps.

28


---Page Break---
1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer:[Yes]

Justification: The claims in the abstract and introduction are supported by the detailed
methodologies and comprehensive experimental results presented in Sections 3 and 4,
respectively. These sections demonstrate the effectiveness and universality of PUREGEN
methods in defending against a variety of train-time poisoning attacks.

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

Answer:[Yes]

Justification: The limitations are discussed in Section 4.5, where we address the computa-
tional cost, data requirements for training the generative models, and the trade-offs between
purification times and performance.

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

29


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The theoretical results, assumptions, and proofs are detailed in Sections 3.1
and 3.2, with additional mathematical formulations provided in the appendices.
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
Justification: The paper includes a commitment to release code and models upon publication
(along with an anonymous codebase attached for submission). Detailed instructions for
reproducing the experiments are provided in the Appendix.
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

30


---Page Break---
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: All relevant training and testing details, including data splits, hyperparameters,
and optimizer settings, are thoroughly described in Section 4.1 and Appendix D. The
paper includes a commitment to release code and models upon publication (along with an
anonymous codebase attached for submission).
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
Justification: Training and test details are, to the best extent of the authors, following
previous benchmarks provided by previous poisons authors [50, 18, 4, 9]. Otherwise,
hyperparameter explanations specific to PUREGEN are given in Section 3.4. The details of
all hyperparameters used are listed in App. D.2.
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
Justification: The results include error bars in all experiential results in Section 4.
Guidelines:

31


---Page Break---
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
Justification: Information on the compute resources used, including the type of hardware
(TPU V3), memory, and execution time, is provided in Section 4.5 and Table 5.
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

Justification: The research adheres to the NeurIPS Code of Ethics, ensuring responsible use
of AI technologies and addressing potential ethical concerns related to data poisoning and
model security, as discussed in the “Potential Social Impacts” Section 6.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

32


---Page Break---
Answer: [Yes]
Justification: The social impacts section in App. 6 provides a balanced discussion of both
positive and negative societal impacts of the research, including ethical considerations and
the potential for misuse.
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
Justification: The research does not involve the release of high-risk models or datasets
that require specific safeguards, but broader risks of the research in general is discussed in
Section 6.
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

Justification: All datasets and models used are properly cited in Section 4 or in Appendix D.
Guidelines:

33


---Page Break---
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
Justification: Code and models will be released upon acceptance. Anonymous code is
attached with submission. No additional assets.
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
Justification: No Human Subjects
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

34


---Page Break---
Justification: No Human Subjects
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
