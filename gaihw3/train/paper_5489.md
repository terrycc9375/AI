BitsFusion: 1.99 bits Weight Quantization of Diffusion
Model

Yang Sui1,2,†
Yanyu Li1
Anil Kag1
Yerlan Idelbayev1
Junli Cao1
Ju Hu1

Dhritiman Sagar1
Bo Yuan2
Sergey Tulyakov1
Jian Ren1,∗

1Snap Inc.
2Rutgers University
Project Page: https://snap-research.github.io/BitsFusion

Figure 1: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images generated
from BitsFusion, where the weights of UNet are quantized into 1.99 bits, achieving 7.9× smaller
storage than the one from Stable Diffusion v1.5. All the images are synthesized under the setting
of using PNDM sampler [49] with 50 sampling steps and random seed as 1024. Prompts and more
generations are provided in App. M.

Abstract

Diffusion-based image generation models have achieved great success in recent
years by showing the capability of synthesizing high-quality content. However,
these models contain a huge number of parameters, resulting in a significantly
large model size. Saving and transferring them is a major bottleneck for various
applications, especially those running on resource-constrained devices. In this
work, we develop a novel weight quantization method that quantizes the UNet
from Stable Diffusion v1.5 to 1.99 bits, achieving a model with 7.9× smaller size
while exhibiting even better generation quality than the original one. Our approach
includes several novel techniques, such as assigning optimal bits to each layer,
initializing the quantized model for better performance, and improving the training
strategy to dramatically reduce quantization error. Furthermore, we extensively
evaluate our quantized model across various benchmark datasets and through
human evaluation to demonstrate its superior generation quality.

1
Introduction

Recent efforts in developing diffusion-based image generation models [77, 31, 79, 21, 80] have
demonstrated remarkable results in synthesizing high-fidelity and photo-realistic images, leading
to various applications such as content creation and editing [68, 67, 61, 71, 90, 88, 50, 40], video
generation [20, 75, 3, 1, 16, 57, 15], and 3D asset synthesis [87, 44, 64, 74, 65], among others.

†Work done during an internship at Snap Inc.
*Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
However, Diffusion Models (DMs) come with the drawback of a large number of parameters, e.g.,
millions or even billions, causing significant burdens for transferring and storing due to the bulky
model size, especially on resource-constrained hardware such as mobile and wearable devices.

Existing studies have explored reducing the model size of large-scale text-to-image diffusion models
by designing efficient architectures and network pruning [41, 92, 32]. These approaches usually
require significant amounts of training due to the changes made to the pre-trained networks. Another
promising direction for model storage reduction is quantization [12, 30], where floating-point weights
are converted to low-bit fixed-point representations, thereby saving computation memory and storage.

There have been emerging efforts on compressing the DMs through quantization [73, 38, 17, 39].
However, these approaches still face several major challenges, especially when quantizing large-scale
text-to-image diffusion models like Stable Diffusion v1.5 (SD-v1.5) [70]. First, many of these
methods are developed on relatively small-scale DMs trained on constrained datasets. For example,
models trained on CIFAR-10 require modest storage of around 100 MB [21, 39]. In contrast, SD-v1.5
necessitates 3.44 GB of storage in a full-precision format. Adapting these methods to SD-v1.5
remains to be a challenging problem. Second, current arts mainly focus on quantizing weights to 4
bits. How to quantize the model to extremely low bit is not well studied. Third, there is a lack of fair
and extensive evaluation of how quantization methods perform on large-scale DMs, i.e., SD-v1.5.

To tackle the above challenges, this work proposes BitsFusion, a quantization-aware training frame-
work that employs a series of novel techniques to compress the weights of pre-trained large-scale
DMs into extremely low bits (i.e., 1.99 bits), achieving even better performance (i.e., higher image
quality and better text-image alignment). Consequently, we compress the 1.72 GB UNet (FP16)1 of
SD-v1.5 into a 219 MB model, achieving a 7.9× compression ratio. Specifically, our contributions
can be summarized into the following four dimensions:

• Mixed-Precision Quantization for DMs. We propose an effective approach for quantizing DMs
in a mixed-precision manner. First, we thoroughly analyze the appropriate metrics to understand
the quantization error in the quantized DMs (Sec. 3.2). Second, based on the analysis, we quantize
different layers into different bits according to their quantization error (Sec. 3.3).
• Initialization for Quantized DMs. We introduce several techniques to initialize the quantized
model to improve performance, including time embedding pre-computing and caching, adding
balance integer, and alternating optimization for scaling factor initialization (Sec. 4.1).
• Improved Training Pipeline for Quantized DMs. We improve the training pipeline for the
quantized model with the proposed two-stage training approach (Sec. 4.2). In the first stage, we
use the full-precision model as a teacher to train the quantized model through distillation. Our
distillation loss forces the quantized model to learn both the predicted noise and the intermediate
features from the teacher network. Furthermore, we adjust the distribution of time step sampling
during training, such that the time steps causing larger quantization errors are sampled more
frequently. In the second stage, we fine-tune the model using vanilla noise prediction [21].
• Extensive Quantitative Evaluation. For the first time in the literature, we conduct extensive
quantitative analysis to compare the performance of the quantized model against the original
SD-v1.5. We include results on various benchmark datasets, i.e., TIFA [25], GenEval [13],
CLIP score [66] and FID [19] on MS-COCO 2014 validation set [46]. Additionally, we perform
human evaluation on PartiPrompts [86]. Our 1.99-bit weights quantized model consistently
outperforms the full-precision model across various evaluations, demonstrating the effectiveness
of our approach.

2
Related Works

To enhance model efficiency in terms of storage and computational costs, quantization [11, 59, 58,
43, 36, 60, 48, 84, 45, 53] is adopted for diffusion models [73, 38, 18, 76, 81, 83, 51, 85, 4, 82, 7, 93,
27, 17, 39, 91] with primarily two types: post-training quantization (PTQ) and quantization-aware
training (QAT). PTQ does not require a full training loop; instead, it utilizes a limited calibration
dataset to adjust the quantization parameters. For example, PTQ4DM [73] calibrates the quantization
parameters to minimize the quantization error of DMs. Q-Diffusion [38] minimizes the quantization
error via the block-wise reconstruction [42]. PTQD [18] integrates quantization noise into the

1For SD-v1.5, we measure the generation quality using the FP32 format. However, since SD-v1.5 FP16 has
similar performance to SD-v1.5 FP32, we use SD-v1.5 FP16 to calculate our compression ratio.

2


---Page Break---
stochastic noise inherent in the sampling steps of DMs. TDQ [76] optimizes scaling factors for
activations across different time steps, applicable to both PTQ and QAT strategies. TFMQ [27]
focuses on reconstructing time embedding and projection layers to prevent over-fitting. However, PTQ
often results in performance degradation compared to QAT, particularly when aiming for extremely
low-bit DMs. In contrast, QAT involves training the full weights to minimize the quantization error,
thereby achieving higher performance compared to PTQ. For instance, EfficientDM [17], inspired
by LoRA [24], introduces a quantization-aware low-rank adapter to update the LoRA weights,
avoiding training entire weights. Q-DM [39] employs normalization and smoothing operation
on attention features through proposed Q-attention blocks, enhancing quantization performance.
Nevertheless, existing works primarily study 4 bits and above quantization on small-scale DMs
trained on constrained datasets. In this paper, we focus on quantizing large-scale Stable Diffusion to
extremely low bits and extensively evaluating the performance across different benchmark datasets.

3
Mixed Precision Quantization for Diffusion Models

In this section, we first go through the formulations of weight quantization and generative diffusion
models. We then determine the mixed-precision strategy, assigning optimized bit widths to different
layers to reduce the overall quantization error. Specifically, we first analyze the quantization error of
each layer in the diffusion model and conclude sensitivity properties. Then, based on the analysis, we
assign appropriate bits to each layer by jointly considering parameter efficiency (i.e., size savings).

3.1
Preliminaries

Quantization is a popular and commonly used technique to reduce model size. While many
quantization forms exist, we focus on uniform quantization, where full-precision values are mapped
into discrete integer values as follows:

θint = Clip(⌊θfp

s ⌉+ Iz, 0, 2b −1),
(1)

where θfp denotes the floating-point weights, θint is the quantized integer weights, s is the scaling
factor, Iz is the zero point, and b is the quantization bit-width. ⌊·⌉denotes the nearest rounding
operation and Clip(·) denotes the clipping operation that constrains θint within the target range.
Following the common settings [38, 17], we apply the channel-wise quantization and set 8 bits for
the first and last convolutional layer of the UNet.

Stable Diffusion. Denoising diffusion probabilistic models [77, 21] learn to predict real data
distribution x ∼pdata by reversing the ODE flow. Specifically, given a noisy data sample zt =
αtx + σtϵ (αt and σt are SNR schedules and ϵ is the added ground-truth noise), and a quantized
denoising model ˆϵθint,s parameterized by θint and s, the learning objective can be formulated as
follows,
Lθint,s = Et,x [∥ϵ −ˆϵθint,s(t, zt, c)∥] ,
(2)
where t is the sampled time step and c is the input condition (e.g., text embedding). Note that during
the training of quantized model, we optimize θfp and s by backpropagating Lθint,s via Straight-
Through Estimator (STE) [2] and quantize the weights to the integers for deployment. Here, for the
notation simplicity, we directly use θint to represent the optimized weights in the quantized models.

The latent diffusion model [70] such as Stable Diffusion conducts the denoising process in the latent
space encoded by variational autoencoder (VAE) [34, 69], where the diffusion model is the UNet [9].
This work mainly studies the quantization for the UNet model, given it is the major bottleneck for the
storage and runtime of the Stable Diffusion [41]. During the inference time, classifier-free guidance
(CFG) [22] is usually applied to improve the generation,

˜ϵθint,s(t, zt, c) = wˆϵθint,s(t, zt, c) −(w −1)ˆϵθint,s(t, zt, ∅),
(3)

where w ≥1 and ˆϵθint,s(t, zt, ∅) denotes the generation conditioned on the null text prompt ∅.

3.2
Per-Layer Quantization Error Analysis

Obtaining Quantized Models. We first perform a per-layer sensitivity analysis for the diffusion
model. Specifically, given a pre-trained full-precision diffusion model, we quantize each layer to 1, 2,
and 3 bits while freezing others at full-precision, and performing quantization-aware training (QAT)

3


---Page Break---
SD-v1.5
CA toq
CA tok
CA tov
CA tok
CA tok
RB conv
RB conv
shortcut
CA tok
RB conv
shortcut

(a) Left most column shows the images synthesized by SD-v1.5 FP32 and other columns show images generated
by the quantized models, where only one layer is quantized (e.g., CA toq denotes the cross-attention layer for
Query projection is quantized and RB conv shortcut denotes the Convolution Shotcut layer in Residual Block is
quantized. The quantized layers follow the same order of highlighted layers in (b) and (c), from left to right.
Quantizing the layers impact both the image quality (as in RB conv shortcut) and text-image alignment (e.g., the
teddy bear disappears after quantizing some CA tok layers).

CA tok
CA toq CA tov

CA tok
RB conv
RB conv shortcut

CA tok
RB conv shortcut

CA tok

(b) MSE value by quantizing layers in SD-v1.5.

CA tok

CA toq

CA tov

CA tok

RB conv
RB conv shortcut

CA tok

RB conv shortcut

CA tok

(c) CLIP score drop by quantizing layers in SD-v1.5.

Figure 2: 1-bit quantization error analysis for all the layers from the UNet of SD-v1.5.

respectively. For instance, for the SD-v1.5 UNet with 256 layers (excluding time embedding, the first
and last layers), we get a total of 768 quantized candidates. We perform QAT over each candidate on
a pre-defined training sub dataset, and validate the incurred quantization error of each candidate by
comparing it against the full-precision model (more details in App. B).

Measuring Quantization Errors. To find the appropriate way to interpret the quantization error,
we analyze four metrics: Mean-Squared-Error (MSE) that quantifies the pixel-level discrepancies
between images (generations from floating and the quantized model in our case), LPIPS [89] that
assesses human-like perceptual similarity judgments, PSNR [23] that measures image quality by
comparing the maximum possible power of a signal with the power of a corrupted noise, and CLIP
score [66] that evaluates the correlation between an image and its language description. After
collecting the scores (examples in Fig. 2b and Fig. 2c, full metrics are listed in App. F), we further
measure the consistency of them by calculating the Pearson correlation [8] for different metrics under
the same bit widths (in Tab. 1), and different bit widths under the same metric (in Tab. 2). With these
empirical results, we draw the following two main observations.

Observation 1: MSE, PSNR, and LPIPS show strong correlation and they correlate well with the
visual perception of image quality.

Tab. 1 shows that MSE is highly correlated with PSNR and LPIPS under the same bit width.
Additionally, we observe a similar trend of per-layer quantization error under different bit widths,
as in Tab. 2. As for visual qualities in Fig. 2a and 2b, we can see that higher MSE errors lead to
severe image quality degradation, e.g., the highlighted RB conv shortcut. Therefore, the MSE metric
effectively reflects quality degradations incurred by quantization, and it is unnecessary to incorporate
PSNR and LPIPS further.

Observation 2: After low-bit quantization, changes in CLIP score are not consistently correlated
with MSE across different layers. Although some layers show smaller MSE, they may experience
larger semantic degradation, reflected in larger CLIP score changes.

We notice that, after quantization, the CLIP score changes for all layers only have a weak correlation
with MSE, illustrated in Tab. 1. Some layers display smaller MSE but larger changes in CLIP score.

4


---Page Break---
Table 1: Pearson correlation (absolute value) of
quantization error between different metrics (e.g.,
MSE vs. PSNR denotes the correlation between
two metrics) when quantizing individual layers to
1, 2, and 3 bits. CS denotes CLIP Score.

MSE vs. PSNR MSE vs. LPIPS MSE vs. CS

1 bit
0.870
0.984
0.733
2 bit
0.882
0.989
0.473
3 bit
0.869
0.991
0.535

Table 2: Pearson correlation (absolute value)
of quantization error between different bit pairs
(e.g., 1 vs. 2 denotes the correlation between the
two bit widths) for a single metric when quantiz-
ing individual layers to 1, 2, and 3 bits.

MSE
PSNR
LPIPS
CLIP Score

1 vs. 2 bit
0.929
0.954
0.943
0.504
1 vs. 3 bit
0.766
0.843
0.802
0.344
2 vs. 3 bit
0.887
0.923
0.895
0.428

For example, in Fig. 2b, the MSE of CA tok layer (5th highlighted layer (green) from left to right) is
less than that of RB conv layer (6th highlighted layer (orange) from left to right), yet the changes
in CLIP score are the opposite. As observed in the first row of Fig. 2a, compared to RB conv layer,
quantizing this CA tok layer changes the image content from "a teddy bear" to "a person", which
diverges from the text prompt A teddy bear on a skateboard in Times Square, doing tricks on a
cardboard box ramp. This occurs because MSE measures only the difference between two images,
which does not capture the semantic degradation. In contrast, the CLIP score reflects the quantization
error in terms of semantic information between the text and image. Thus, we employ the CLIP score
as a complementary metric to represent the quantization error.

3.3
Deciding the Optimal Precision

With the above observations, we then develop the strategy for bit-width assignments. We select
MSE and CLIP as our quantitative metrics, along with the number of parameters of each layer as the
indicator of size savings.

Assigning bits based on MSE. Intuitively, layers with more parameters and lower quantization
error are better candidates for extremely low-bit quantization, as the overall bit widths of the model
can be significantly reduced. According to this, we propose a layer size-aware sensitivity score S.
For the ith layer, its sensitivity score for the b-bits (b ∈{1, 2, 3}) is defined as Si,b = Mi,bN −η
i
,
where M denotes the MSE error, N is the total number of parameters of the layer, and η ∈[0, 1]
denotes the parameter size factor. To determine the bit width (i.e., b∗) for each layer, we define a
sensitivity threshold as So, and the ith layer is assigned to b∗
i -bits, where b∗
i = min{b|Si,b < So}.
The remaining layers are 4 bits.

Assigning bits based on CLIP score. For the layers with a high CLIP score dropping after quantiza-
tion, instead of assigning bits based on sensitivity score as discussed above, we directly assign higher
bits to those layers. Therefore, the quantized model can produce content that aligns with the semantic
information of the prompt. We provide the detailed mixed-precision algorithm in Alg. 1 of App. B.

4
Training Extreme Low-bit Diffusion Model

With the bits of each layer decided, we then train the quantized model with a series of techniques to
improve performance. The overview of our approach is illustrated in Fig. 3.

4.1
Initializing the Low-bit Diffusion Model

Time Embedding Pre-computing and Caching. During the inference time of a diffusion model,
a time step t is transformed into an embedding through projection layers to be incorporated into
the diffusion model. As mentioned by existing works [27], the quantization of the projection layers
can lead to large quantization errors. However, the embedding from each time step t is always the
same, suggesting that we can actually pre-compute the embedding offline and load cached values
during inference, instead of computing the embedding every time. Furthermore, the storage size of
the time embedding is 25.6× smaller than the projection layers. Therefore, we pre-compute the time
embedding and save the model without the project layers. More details are provided in App. C.

Adding Balance Integer. In general, weight distributions in deep neural networks are observed
as symmetric around zero [94]. To validate the assumption on SD-v1.5, we analyze its weight

5


---Page Break---
E

Predicted 

Noise

Stable Diffusion v1.5

Quantized UNet

Stage-II
Training

Stage-I
Training
𝒕~ Beta(𝜶, 𝜷)

Prompt: A cat

Removed Time 
Projection Layers

Pre-computed and Cached 

Time Features

Embedding

Tokenizer

Tokenizer /
Embedding

Mixed 
Precision 

Recipe

Stable Diffusion v1.5

Quantized UNet

Initialization

Analysis

Predicted 

Noise

Inference Stage 
Training Stage 

D

UNet

1.99 bits UNet

Freeze
Trainable

Lnoise

θint,s
Lfeat

θint,s

Lθint,s

ˆϵθfp

ˆϵθint,s

Prompt: A dog

Figure 3: Overview of the training and inference pipeline for the proposed BitsFusion. Left: We
analyze the quantization error for each layer in SD-v1.5 (Sec. 3.2) and derive the mixed-precision
recipe (Sec. 3.3) to assign different bit widths to different layers. We then initialize the quantized
UNet by adding a balance integer, pre-computing and caching the time embedding, and alternately
optimizing the scaling factor (Sec. 4.1). Middle: During the Stage-I training, we freeze the teacher
model (i.e., SD-v1.5) and optimize the quantized UNet through CFG-aware quantization distillation
and feature distillation losses, along with sampling time steps by considering quantization errors (Sec.
4.2). During the Stage-II training, we fine-tune the previous model with the noise prediction. Right:
For the inference stage, using the pre-cached time features, our model processes text prompts and
generates high-quality images.

distribution for the layers under full precision by calculating the skewness of weights. Notably, the
skewness of more than 97% of the layers ranges between [−0.5, 0.5], indicating that the weight
distributions are symmetric in almost all layers. Further details are provided in App. D.

However, existing works on diffusion model quantization overlook the symmetric property [38, 73,
39], as they perform relatively higher bits quantization, e.g., 4 or 8 bits. This will hurt the model
performance at extremely low bit levels. For example, in 1-bit quantization, the possible most
symmetric integer outcomes can only be {0, 1} or {−1, 0}. Similarly, for 2-bit quantization, the most
balanced mapping integers can be either {−2, −1, 0, 1} or {−1, 0, 1, 2}, significantly disrupting the
symmetric property. The absence of a single value among 2 or 4 numbers under low-bit quantization
can have a significant impact. To tackle this, we leverage the bit balance strategy [37, 56] to initialize
the model. Specifically, we introduce an additional value to balance the original quantization values.
Namely, in a 1-bit model, we adjust the candidate integer set from {0, 1} to {−1, 0, 1}, achieving a
more balanced distribution. By doing so, we treat the balanced n-bits weights as log(2n + 1)-bits.

Scaling Factor Initialization via Alternating Optimization. Initializing scaling factors is an
important step in quantization. Existing QAT works typically employ the Min-Max initialization
strategy [17, 52] to ensure the outliers are adequately represented and preserved. However, such a
method faces challenges in extremely low-bit quantization settings like 1-bit, since the distribution
of the full-precision weights is overlooked, leading to a large quantization error and the increased
difficulty to converge. Therefore, we aim to minimize the ℓ2 error between the quantized weights and
full-precision weights with the optimization objective as:

min
s
∥s · (θint −Iz) −θfp∥2.
(4)

Nevertheless, considering the rounding operation, calculating an exact closed-form solution is not
straightforward [29]. Inspired by the Lloyd-Max algorithm [28, 54], we use an optimization method
on scaling factor s to minimize the initialization error of our quantized diffusion model as follows:

θj
int = Qint(θfp, sj−1); sj =
θj
fp(θj
int −Iz)⊺

(θj
int −Iz)(θj
int −Iz)⊺,
(5)

where Qint(·) denotes the integer mapping quantization operation that converts the full-precision
weights to integer as Eq. (1), and j represents the iterative step. The optimization is done for 10 steps.

6


---Page Break---
4.2
Two-Stage Training Pipeline

With the mixed-precision model initialized, we introduce the two-stage training pipeline. In Stage-I,
we train the quantized model using the full-precision model as the teacher through distillation loss. In
Stage-II, we fine-tune the model from the previous stage using noise prediction [21, 80].

CFG-aware Quantization Distillation. Similar to existing works [11], we fine-tune the quantized
diffusion model to improve the performance. Here both the weights and scaling factors are opti-
mized. Additionally, we notice that training the quantized model in a distillation fashion using the
full-precision model yields better performance than training directly with vanilla noise prediction.
Furthermore, during distillation, it is crucial for the quantized model to be aware of CFG, i.e., text
dropping is applied during distillation. Specifically, our training objective is as follows:

Lnoise
θint,s = Et,x

∥ˆϵθfp(t, zt, c) −ˆϵθint,s(t, zt, c)∥

, c = ∅if P ∼U[0, 1] < p else c,
(6)

where P controls the text dropping probability during training and p is set as 0.1.

Feature Distillation. To further improve the generation quality of the quantized model, we distill the
full-precision model at a more fine-grained level through feature distillation [32] as follows:

Lfeat
θint,s = Et,x

∥Fθfp(t, zt, c) −Fθint,s(t, zt, c)∥

,
(7)

where Fθ(·) denotes the operation for getting features from the Down and Up blocks in UNet. We
then have the overall distillation loss Ldist in Stage-I as follows:

Ldist = Lnoise
θint,s + λLfeat
θint,s,
(8)

where λ is empirically set as 0.01 to balance the magnitude of the two loss functions.

0
200
400
600
800
1000
Time Steps

0

10

20

30

40

50

60

Sampling Frequency

0.000

0.004

0.008

0.012

Quantization Error

=1.5, =1
=3, =1
Quantization Error

Figure 4: More time steps are sampled
towards where larger quantization error
occurs.

Quantization Error-aware Time Step Sampling. The
training of diffusion models requires sampling differ-
ent time steps in each optimization iteration.
We ex-
plore how to adjust the strategy for time step sampling
such that the quantization error in each time step can
be effectively reduced during training. We first train a
1.99-bit quantized model with Eq. (8). Then, we cal-
culate the difference of the predicted latent features be-
tween the quantized model and the full-precision model
as Et,x[ 1−¯αt

¯αt ∥ˆϵθfp(t, zt, c) −ˆϵθint,s(t, zt, c)∥2], where
t ∈[0, 1, · · · , 999] and ¯αt is the noise scheduler (detailed
derivation in App. E). The evaluation is conducted on a
dataset with 128 image-text pairs. Fig. 4 shows the quan-
tization error does not distribute equally across all time
steps. Notably, the quantization error keeps increasing as
the time steps approach t = 999.

To mitigate the quantization error prevalent near the time steps t = 999, we propose a sampling
strategy by utilizing a distribution specifically tailored to sample more time steps exhibiting the
largest quantization errors, thereby enhancing performance. To achieve this goal, we leverage the
Beta distribution. Specifically, time steps are sampled according to t ∼Beta(α, β), as shown in Fig.
4. We empirically set α = 3.0 and β = 1.0 for the best performance. Combining the strategy of time
steps sampling with Eq. (8), we conduct the Stage-I training.

Fine-tuning with Noise Prediction. After getting the model trained with the distillation loss in
Stage-I, we then fine-tune it with noise prediction, as in Eq. (2), in Stage-II. We apply a text dropping
with probability as 10% and modify the distribution of time step sampling based on the quantization
error, as introduced above. The reason we leverage two-stage fine-tuning, instead of combining
Stage-I and Stage-II, is that we observe more stabilized training results.

5
Experiments

Implementation Details. We develop our code using diffusers library2 and train the models with
AdamW optimizer [33] and a constant learning rate as 1e−05 on an internal dataset. For Stage-I,

2https://github.com/huggingface/diffusers

7


---Page Break---
2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0.300

0.305

0.310

0.315

CLIP Scores

SD-v1.5 (32 bits)
Ours-I (1.99 bits)
Ours-II (1.99 bits)

(a) CLIP score on MS-COCO.

2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0.74

0.76

0.78

0.80

0.82

TIFA Scores

SD-v1.5 (32 bits)
Ours-I (1.99 bits)
Ours-II (1.99 bits)

(b) Evaluation on TIFA.

2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0.32

0.36

0.40

0.44

0.48

GenEval Scores

SD-v1.5 (32 bits)
Ours-I (1.99 bits)
Ours-II (1.99 bits)

(c) Evaluation on GenEval.

Figure 5: Comparison between our 1.99-bits model vs. SD-v1.5 on various evaluation metrics with
CFG scales ranging from 2.5 to 9.5. Ours-I denotes the model with Stage-I training and Ours-II
denotes the model with Stage-II training.

Table 3:
Comparison with
existing quantization meth-
ods, including LSQ [11], Q-
Diffusion [38], EfficientDM
[17], and Apple-MBP [62].
The CLIP score is measured
on 1K PartiPrompts.

Method
Bit-width CLIP score

SD-v1.5
32
0.3175
LSQ
2
0.2849
Q-Diffusion
4
0.3137
EfficientDM
2
0.2918
Apple-MBP
2
0.3023

Ours
1.99
0.3212

Table 4: Analysis of our proposed methods measured under vari-
ous CFG scales, i.e., 3.5, 5.5, 7.5, and 9.5. We use LSQ [11] as
the basic QAT method, which involves the training of weights
and scaling factors of a uniformly 2-bit quantized UNet. Then,
we gradually introduce each proposed technique to evaluate their
effectiveness. CLIP scores are measured on 1K PartiPrompts.

Method
Bit-width
3.5
5.5
7.5
9.5
Average
∆

SD-v1.5
32
0.3110 0.3159 0.3175 0.3180
0.3156
-
QAT-Base
2
0.2679 0.2793 0.2849 0.2868
0.2797
-
+Balance
2.32
0.2990 0.3059 0.3080 0.3086
0.3054 +0.0257
+Alternating Opt.
2.32
0.3061 0.3108 0.3117 0.3115
0.3100 +0.0046
+Mixed/Caching
1.99
0.3055 0.3129 0.3142 0.3145
0.3118 +0.0018
+Feat Dist.
1.99
0.3086 0.3147 0.3167 0.3169
0.3142 +0.0024
+Time Sampling
1.99
0.3098 0.3159 0.3181 0.3184
0.3156 +0.0014
+Fine-tuning
1.99
0.3163 0.3192 0.3212 0.3205
0.3183 +0.0027

we use 8 NVIDIA A100 GPUs with a total batch size of 256 to train the quantized model for 20K
iterations. For Stage-II, we use 32 NVIDIA A100 GPUs with a total batch size of 1024 to train the
quantized model for 50K iterations. During inference, we adopt the PNDM scheduler [49] with 50
sampling steps to generate images for comparison. Other sampling approaches (e.g., DDIM [78] and
DPMSolver [55]) lead to the same conclusion (App. K).

Evaluation Metrics. We conduct evaluation on CLIP Score and FID on MS-COCO [47], TIFA [26],
GenEval [14], and human evaluation on PartiPrompts [86]. We adopt ViT-B/32 model [10] in CLIP
score and the Mask2Former(Swin-S-8×2) [5] in GenEval. App. I provides details for the metrics.

5.1
Main Results

Comparison with SD-v1.5. Our quantized 1.99-bits UNet consistently outperforms the full-precision
model across all metrics.

• 30K MS-COCO 2014 Validation Set. For the CLIP score, as demonstrated in Fig. 5a, attributed
to the proposed mixed-precision recipe with the introduced initialization techniques and advanced
training schemes in Stage-I, our 1.99-bits UNet, with a storage size of 219MB, achieves perfor-
mance comparable to the original SD-v1.5. Following Stage-II training, our model surpasses the
performance of the original SD-v1.5. With CFG scales ranging from 2.5 to 9.5, our model yields
0.002 ∼0.003 higher CLIP scores.
• TIFA. As shown in Fig. 5b, our 1.99-bits model with Stage-I training performs comparably to the
SD-v1.5. With the Stage-II training, our model achieves better metrics over the SD-v1.5.
• GenEval. We show the comparison results for GenEval in Fig. 5c (detailed comparisons of
GenEval score are presented in Appn. L). Our model outperforms SD-v1.5 for all CFG scales.
• Human Evaluation. With the question: Given a prompt, which image has better aesthetics and
image-text alignment? More users prefer the images generated by our quantized model over SD-
v1.5, with the ratio as 54.4%. The results are shown in Fig. 6. We provide a detailed comparison
in App. J.

Comparison with Other Quantization Approaches. Additionally, we conduct the experiments by
comparing our approach with other works including LSQ [11], Q-Diffusion [38], EfficientDM [17],

8


---Page Break---
Table 5:
Analysis of η in the mixed-
precision strategy.

η
0
0.1
0.2
0.3
0.4
0.5

CLIP score 0.3155 0.3173 0.3162 0.3181 0.3171 0.3168

Table 6: Anlysis of λ in dis-
tillation loss.

λ
1
0.1
0.01

CLIP score 0.3164 0.3159 0.3181

Table 7: Analysis of α in
time step-aware sampling.

α
1.5
2.0
3.0

CLIP score 0.3169 0.3173 0.3181

and Apple-MBP [62], as shown in Tab. 3. Our model achieves a higher CLIP score compared with
all other works and better performance than SD-v1.5.

5.2
Ablation Analysis

Here we perform extensive analysis for our proposed method. We mainly evaluate different experi-
mental settings using the CLIP score measured on 1K PartiPrompts [86].

Analysis of the Proposed Techniques. We adopt the LSQ [11] as the basic QAT method to update
the weights and scaling factors of a uniform 2-bit UNet with Min-Max initialization. Results are
presented in Tab. 4 with the following details:

• +Balance. By adding a balance integer, a 2-bit model that typically represents 4 integer values can
now represent 5 integers, becoming a 2.32-bit model by log(4 + 1) bits. The average CLIP score
has significantly increased from 0.2797 to 0.3054.
• +Alternating Opt. By further utilizing the scaling factor initialization via alternating optimization,
the average CLIP score of the 2.32-bit model increases to 0.3100.
• +Mixed/Caching. By leveraging time embedding pre-computing and caching, we minimize the
storage requirements for time embedding and projection layers by only retaining the calculated
features. This significantly reduces the averaged bits. Combined with our mixed-precision strategy,
this approach reduces the average bits from 2.32 to 1.99 bits and can even improve the performance,
i.e., CLIP score improved from 0.3100 to 0.3118.
• +Feat Dist. By incorporating the feature distillation loss, i.e., Eq. (7), the model can learn more
fine-grained information from the teacher model, improving CLIP score from 0.3118 to 0.3142.
• +Time Sampling. By employing a quantization error-aware sampling strategy at various time
steps, the model focuses more on the time step near t = 999. With this sampling strategy, our
1.99-bits model performs very closely to, or even outperforms, the original SD-v1.5.
• +Fine-tuning. By continuing with Stage-II training that incorporates noise prediction, our 1.99-
bits model consistently outperforms the SD-v1.5 across various guidance scales, improving the
CLIP score to 0.3183.

Effect of η in Mixed-Precision Strategy. Tab. 5 illustrates the impact of the parameter size factor
η (as discussed in Sec. 3.3) in determining the optimal mixed precision strategy. We generate six
different mixed precision recipes with different η with 20K training iterations for comparisons.
Initially, we explore the mixed precision strategy determined with and without the parameter size
factor. Setting η = 0 results in N −η = 1, indicating that the mixed precision is determined without
considering the impact of parameter size. The results show that neglecting the parameter size
significantly degrades performance. Further, we empirically choose η = 0.3 in our experiments after
comparing different values of η.

Effect of λ of Distillation Loss. Tab. 6 illustrates the impact of the balance factor λ for loss functions
in Eq. (8). We empirically choose λ = 0.01 in our experiments after comparing the performance.

Effect of α in Time Step-aware Sampling Strategy. Tab. 7 illustrates the impact of the α for
different Beta sampling distribution. As analyzed in Sec. 4.2, the quantization error increases near
t = 999. To increase sampling probability near this time step, Beta distribution requires α > 1 with
β = 1. A larger α enhances the sampling probability near t = 999. Compared to α = 1.5 and
α = 2.0, α = 3.0 concentrates more on later time steps and achieves the best performance. We
choose α = 3.0 in our experiments.

Analysis for Different Schedulers. One advantage of our training-based quantization approach is
that our quantized model consistently outperforms SD-v1.5 for various sampling approaches. We
conduct extensive evaluations on TIFA to show we achieve better performance than SD-v1.5 for using
both DDIM [78] and DPMSolver [55] to perform the sampling. More details are shown in App. K.

FID Results. As stated in SDXL [63] and PickScore [35], FID may not honestly reflect the actual
performance of the model in practice. FID measures the average distance between generated images

9


---Page Break---
and reference real images, which is largely influenced by the training datasets. Also, FID does not
capture the human preference which is the crucial metric for evaluating text-to-image synthesis. We
present FID results evaluated on the 30K MS-COCO 2014 validation set in Fig. 7. Our Stage-I
model has a similar FID as SD-v1.5. However, as training progresses, although our Stage-II model is
preferred by users, its FID score is higher than both Stage-I and SD-v1.5.

Ours (1.99 bits)
SD-v1.5 (32 bits)

20

40

60

Win Rate (%)

54.41

45.59

Figure 6: Overall human evaluation com-
parisons between SD-v1.5 and BitsFusion.
Notably, BitsFusion, is favored 54.41% of
the time over SD-v1.5.

2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0

5

10

15

20

25

30

FID (30K)

SD-v1.5 (32 bits)
Ours-I (1.99 bits)
Ours-II (1.99 bits)

Figure 7: FID results evaluated on 30K MS-COCO
2014 validation set.

6
Conclusion

To enhance the storage efficiency of the large-scale diffusion models, we introduce an advanced
weight quantization framework, BitsFusion, which effectively compresses the weights of UNet from
SD-v1.5 to 1.99 bits, achieving a 7.9× smaller model size. BitsFusion even outperforms SD-v1.5 in
terms of generation quality. Specifically, we first conduct a comprehensive analysis to understand
the impact of each layer during quantization and establish a mixed-precision strategy. Second, we
propose a series of effective techniques to initialize the quantized model. Third, during the training
stage, we enforce the quantized model to learn the full-precision SD-v1.5 by using distillation losses
with the adjusted distribution of time step sampling. Finally, we fine-tune the previous quantized
model through vanilla noise prediction. Our extensive evaluations on TIFA, GenEval, CLIP score, and
human evaluation consistently demonstrate the advantage of BitsFusion over full-precision SD-v1.5.

References

[1] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat,
Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for
video generation. arXiv preprint arXiv:2401.12945, 2024. 1

[2] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients
through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
3

[3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja
Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent
diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 22563–22575, 2023. 1

[4] Hanwen Chang, Haihao Shen, Yiyang Cai, Xinyu Ye, Zhenzhong Xu, Wenhua Cheng, Kaokao
Lv, Weiwei Zhang, Yintong Lu, and Heng Guo. Effective quantization for diffusion models on
cpus. arXiv preprint arXiv:2311.16133, 2023. 2

[5] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar.
Masked-attention mask transformer for universal image segmentation. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 8,
30

10


---Page Break---
[6] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social
biases of text-to-image generation models. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 3043–3054, 2023. 30

[7] Huanpeng Chu, Wei Wu, Chengjie Zang, and Kun Yuan. Qncd: Quantization noise correction
for diffusion models. arXiv preprint arXiv:2403.19140, 2024. 2

[8] Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Benesty, Jacob Benesty, Jingdong Chen,
Yiteng Huang, and Israel Cohen. Pearson correlation coefficient. Noise reduction in speech
processing, pages 1–4, 2009. 4

[9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis.
Advances in neural information processing systems, 34:8780–8794, 2021. 3

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at scale. In International
Conference on Learning Representations, 2020. 8

[11] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dhar-
mendra S Modha. Learned step size quantization. In International Conference on Learning
Representations, 2019. 2, 7, 8, 9

[12] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer.
A survey of quantization methods for efficient neural network inference. In Low-Power Com-
puter Vision, pages 291–326. Chapman and Hall/CRC, 2022. 2

[13] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused
framework for evaluating text-to-image alignment. Advances in Neural Information Processing
Systems, 36, 2024. 2

[14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused
framework for evaluating text-to-image alignment. Advances in Neural Information Processing
Systems, 36, 2024. 8, 30

[15] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh
Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-
to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.
1

[16] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang,
and José Lezama. Photorealistic video generation with diffusion models. arXiv preprint
arXiv:2312.06662, 2023. 1

[17] Yefei He, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang.
Efficientdm: Efficient
quantization-aware fine-tuning of low-bit diffusion models.
In The Twelfth International
Conference on Learning Representations, 2023. 2, 3, 6, 8

[18] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate
post-training quantization for diffusion models. Advances in Neural Information Processing
Systems, 36, 2024. 2

[19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in
neural information processing systems, 30, 2017. 2

[20] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko,
Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High
definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2, 3, 7, 20

11


---Page Break---
[22] Jonathan Ho and Tim Salimans.
Classifier-free diffusion guidance.
arXiv preprint
arXiv:2207.12598, 2022. 3

[23] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international
conference on pattern recognition, pages 2366–2369. IEEE, 2010. 4

[24] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu
Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference
on Learning Representations, 2021. 3

[25] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and
Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with
question answering. arXiv preprint arXiv:2303.11897, 2023. 2

[26] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and
Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with
question answering. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 20406–20417, 2023. 8, 30

[27] Yushi Huang, Ruihao Gong, Jing Liu, Tianlong Chen, and Xianglong Liu. Tfmq-dm: Temporal
feature maintenance quantization for diffusion models. arXiv preprint arXiv:2311.16503, 2023.
2, 3, 5, 18

[28] Kyuyeon Hwang and Wonyong Sung. Fixed-point feedforward deep neural network design
using weights+ 1, 0, and- 1. In 2014 IEEE Workshop on Signal Processing Systems (SiPS),
pages 1–6. IEEE, 2014. 6

[29] Yerlan Idelbayev, Pavlo Molchanov, Maying Shen, Hongxu Yin, Miguel A Carreira-Perpinán,
and Jose M Alvarez. Optimal quantization using scaled codebook. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12095–12104, 2021.
6

[30] Qing Jin, Jian Ren, Richard Zhuang, Sumant Hanumante, Zhengang Li, Zhiyu Chen, Yanzhi
Wang, Kaiyuan Yang, and Sergey Tulyakov. F8net: Fixed-point 8-bit only multiplication for
network quantization. arXiv preprint arXiv:2202.05239, 2022. 2

[31] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of
diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022. 1

[32] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. On architectural
compression of text-to-image diffusion models. arXiv preprint arXiv:2305.15798, 2023. 2, 7

[33] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014. 7

[34] Diederik P Kingma and Max Welling.
Auto-encoding variational bayes.
arXiv preprint
arXiv:1312.6114, 2013. 3

[35] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy.
Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in
Neural Information Processing Systems, 36, 2024. 9

[36] Junghyup Lee, Dohyung Kim, and Bumsub Ham. Network quantization with element-wise
gradient scaling. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 6448–6457, 2021. 2

[37] Fengfu Li, Bin Liu, Xiaoxing Wang, Bo Zhang, and Junchi Yan. Ternary weight networks.
arXiv preprint arXiv:1605.04711, 2016. 6

[38] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang,
and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 17535–17545, 2023. 2, 3, 6, 8

12


---Page Break---
[39] Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and Baochang Zhang. Q-dm: An efficient
low-bit quantized diffusion model. Advances in Neural Information Processing Systems, 36,
2024. 2, 3, 6

[40] Yanyu Li, Xian Liu, Anil Kag, Ju Hu, Yerlan Idelbayev, Dhritiman Sagar, Yanzhi Wang, Sergey
Tulyakov, and Jian Ren. Textcraftor: Your text encoder can be image quality controller. arXiv
preprint arXiv:2403.18978, 2024. 1

[41] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey
Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within
two seconds. Advances in Neural Information Processing Systems, 36, 2024. 2, 3

[42] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang,
and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. In
International Conference on Learning Representations, 2020. 2

[43] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang,
and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction.
arXiv preprint arXiv:2102.05426, 2021. 2

[44] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten
Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d
content creation. arXiv preprint arXiv:2211.10440, 2022. 1

[45] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan
Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization
for on-device llm compression and acceleration. Proceedings of Machine Learning and Systems,
6:87–100, 2024. 2

[46] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer
Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014,
Proceedings, Part V 13, pages 740–755. Springer, 2014. 2

[47] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer
Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014,
Proceedings, Part V 13, pages 740–755. Springer, 2014. 8, 30

[48] Jing Liu, Bohan Zhuang, Peng Chen, Chunhua Shen, Jianfei Cai, and Mingkui Tan. Single-path
bit sharing for automatic loss-aware model compression. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 45(10):12459–12473, 2023. 2

[49] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models
on manifolds. arXiv preprint arXiv:2202.09778, 2022. 1, 8

[50] Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu,
Ziwei Liu, and Sergey Tulyakov. Hyperhuman: Hyper-realistic human generation with latent
structural diffusion. arXiv preprint arXiv:2310.08579, 2023. 1

[51] Xuewen Liu, Zhikai Li, Junrui Xiao, and Qingyi Gu. Enhanced distribution alignment for
post-training quantization of diffusion models. arXiv preprint arXiv:2401.04585, 2024. 2

[52] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad,
Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quan-
tization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023.
6

[53] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman
Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant–llm
quantization with learned rotations. arXiv preprint arXiv:2405.16406, 2024. 2

[54] Stuart Lloyd. Least squares quantization in pcm. IEEE transactions on information theory,
28(2):129–137, 1982. 6

13


---Page Break---
[55] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver:
A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in
Neural Information Processing Systems, 35:5775–5787, 2022. 8, 9, 31, 32

[56] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong,
Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are
in 1.58 bits. arXiv preprint arXiv:2402.17764, 2024. 6

[57] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen,
Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled
spatiotemporal transformers for text-to-video synthesis. arXiv preprint arXiv:2402.14797, 2024.
1

[58] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort.
Up or down? adaptive rounding for post-training quantization. In International Conference on
Machine Learning, pages 7197–7206. PMLR, 2020. 2

[59] Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. Data-free quantization
through weight equalization and bias correction. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 1325–1334, 2019. 2

[60] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart Van Baalen,
and Tijmen Blankevoort. A white paper on neural network quantization. arXiv preprint
arXiv:2106.08295, 2021. 2

[61] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew,
Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing
with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1

[62] Atila Orhon, Michael Siracusa, and Aseem Wadhwa. Stable diffusion with core ml on apple
silicon, 2022. 8, 9

[63] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image
synthesis. In The Twelfth International Conference on Learning Representations, 2023. 9

[64] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using
2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1

[65] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-
Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image
to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint
arXiv:2306.17843, 2023. 1

[66] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning,
pages 8748–8763. PMLR, 2021. 2, 4, 18

[67] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical
text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1

[68] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark
Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on
Machine Learning, pages 8821–8831. PMLR, 2021. 1

[69] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation
and approximate inference in deep generative models. In International conference on machine
learning, pages 1278–1286. PMLR, 2014. 3

[70] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3

14


---Page Break---
[71] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David
Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH
2022 Conference Proceedings, pages 1–10, 2022. 1

[72] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton,
Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al.
Photorealistic text-to-image diffusion models with deep language understanding. Advances in
neural information processing systems, 35:36479–36494, 2022. 30

[73] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization
on diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 1972–1981, 2023. 2, 6

[74] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang.
Mvdream:
Multi-view diffusion for 3d generation. In The Twelfth International Conference on Learning
Representations, 2023. 1

[75] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu,
Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without
text-video data. arXiv preprint arXiv:2209.14792, 2022. 1

[76] Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, and Eunhyeok Park. Temporal
dynamic quantization for diffusion models. Advances in Neural Information Processing Systems,
36, 2024. 2, 3

[77] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsuper-
vised learning using nonequilibrium thermodynamics. In International Conference on Machine
Learning, pages 2256–2265. PMLR, 2015. 1, 3

[78] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv
preprint arXiv:2010.02502, 2020. 8, 9, 31, 32

[79] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data
distribution. Advances in neural information processing systems, 32, 2019. 1

[80] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and
Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv
preprint arXiv:2011.13456, 2020. 1, 7

[81] Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, and Wenwu Zhu.
Post-training quantization with progressive calibration and activation relaxing for text-to-image
diffusion models. arXiv preprint arXiv:2311.06322, 2023. 2

[82] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards
accurate data-free quantization for diffusion models. arXiv preprint arXiv:2305.18723, 2023. 2

[83] Haoxuan Wang, Yuzhang Shang, Zhihang Yuan, Junyi Wu, and Yan Yan. Quest: Low-bit
diffusion model quantization via efficient selective finetuning. arXiv preprint arXiv:2402.03666,
2024. 2

[84] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han.
Smoothquant: Accurate and efficient post-training quantization for large language models.
In International Conference on Machine Learning, pages 38087–38099. PMLR, 2023. 2

[85] Yuewei Yang, Xiaoliang Dai, Jialiang Wang, Peizhao Zhang, and Hongbo Zhang. Efficient
quantization strategies for latent diffusion models. arXiv preprint arXiv:2312.05431, 2023. 2

[86] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay
Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive
models for content-rich text-to-image generation. Transactions on Machine Learning Research,
2022. 2, 8, 9, 30

[87] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and
Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. arXiv preprint
arXiv:2210.06978, 2022. 1

15


---Page Break---
[88] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image
diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 3836–3847, 2023. 1

[89] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreason-
able effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4

[90] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single
image editing with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6027–6037, 2023. 1

[91] Tianchen Zhao, Xuefei Ning, Tongcheng Fang, Enshu Liu, Guyue Huang, Zinan Lin, Shengen
Yan, Guohao Dai, and Yu Wang. Mixdq: Memory-efficient few-step text-to-image diffusion
models with metric-decoupled mixed precision quantization. arXiv preprint arXiv:2405.17873,
2024. 2

[92] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-
image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023. 2

[93] Xingyu Zheng, Haotong Qin, Xudong Ma, Mingyuan Zhang, Haojie Hao, Jiakai Wang, Zixiang
Zhao, Jinyang Guo, and Xianglong Liu. Binarydm: Towards accurate binarization of diffusion
model. arXiv preprint arXiv:2404.05662, 2024. 2

[94] Chenzhuo Zhu, Song Han, Huizi Mao, and William J Dally. Trained ternary quantization. In
International Conference on Learning Representations, 2016. 5

16


---Page Break---
Appendix

Table of Contents

A Limitations
18

B
More details for Mixed-Precision Algorithm
18

C More Details for Time Embedding Pre-computing and Caching
18

D Analysis of Symmetric Weight Distribution
20

E
More Details for Quantization Error Across Different Time Steps
20

F
Detailed Metrics for Quantization Error by Quantizing Different Layers
21

G More Visualization for Quantization Error by Quantizing Different Layers
25

H 1.99 Bits Mixed Precision Recipe
26

I
Details for Evaluation Metrics
30

J
Human Evaluation
30
J.1
Analysis on Categories
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30
J.2
Analysis on Challenges . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

K Evaluation on Different Schedulers
31

L
Detailed GenEval Results
32

M More Comparisons
33
M.1 Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33
M.2 Additional Image Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . .
33

17


---Page Break---
A
Limitations

In this work, we study the storage size reduction of the UNet in Stable Diffusion v1.5 through
weight quantization. The compression of VAE and CLIP text encoder [66] is also an interesting
direction, which is not explored in this work. Additionally, our weight quantization techniques could
be extended to the activations quantization, as a future exploration.

B
More details for Mixed-Precision Algorithm

In Sec. 3, we analyze the per-layer quantization error and develop the mixed-precision strategy. Here,
we provide the detailed algorithm as outlined in Alg. 1. The inputs include: a pre-defined candidate
set of bit-width b ∈{1, 2, 3}, the full-precision SD-v1.5 D, the total number of layers L (except for
the time embedding, time projection, the first and last convolutional layers), the training dataset X,
the number of training iterations T, the number of evaluation images for calculating metrics K, the
bit threshold So, the parameter size factor η, and the number of parameters of the ith layer Ni.

In the first stage, we aim to obtain quantized models by quantizing each individual layer. Given the
full-precision SD-v1.5 UNet D, we consecutively perform the quantization on every single layer
to 1, 2, or 3 bits individually, while maintaining the remaining layers at FP32 format. Notice, to
align with our experiments, we add the balance integer and initialize the scaling factor with our
alternating optimization. For each quantized model, the weights and scaling factors are fine-tuned
using quantization-aware training to minimize the quantization error by learning the predicted noise
of the SD-v1.5. We obtain quantized models Di,b, i = 1, 2, · · · , L, b = 1, 2, 3.

In the second stage, we measure the quantization error of each layer by calculating various metrics
from comparing images generated by the quantized model Di,b with those from the unquantized
SD-v1.5 D. Specifically, we generate K = 100 baseline images Id from the full-precision SD-v1.5
model with PartiPrompts. Then, for each quantized model Di,b, we use identical prompts and seed to
generate corresponding images Ii,b. We calculate the quantization error by measuring the metrics
including MSE, CLIP score, PSNR, and LPIPS using these images and prompts.

In the third stage, we collect the mixed-precision recipe. We first compute a sensitivity score for each
layer, factoring in both the MSE and the parameter size adjusted by η. For the ith layer, its sensitivity
score for the b-bits (b ∈{1, 2, 3}) is defined as Si,b = Mi,bN −η
i
, where M denotes the MSE error,
N is the total number of parameters of the layer, and η ∈[0, 1] denotes the parameter size factor.
To determine the bit width (i.e., b∗) for each layer, we define a sensitivity threshold as So, and the
ith layer is assigned to b∗
i -bits, where b∗
i = min{b|Si,b < So}. The remaining layers are set as 4
bits if they fail to meet the threshold. After determining the initial bits based on the MSE error, we
refine this recipe by considering the degradation in the CLIP score associated with each bit-width.
We simply consider the CLIP score change at 3 bits. We assign layers with the highest 10%, 5%, 2%
CLIP score drop with 1, 2, 3 more bits, respectively.

The final output is a mixed-precision recipe {b∗
i }, i = 1, 2, · · · , L, specifying the bit-width for each
layer. Then, we set the first and last convolutional layers as 8 bits and pre-computing and caching the
time embedding and projection layers.

C
More Details for Time Embedding Pre-computing and Caching

In Sec. 4.1, we introduce "Time Embedding Pre-computing and Caching". Here, we provide
more details for the algorithm. In the Stable Diffusion model, the time step t ∈[0, 1, · · · , 999] is
transformed into a time embedding embt through the equation embt = e(t), where e(t) denotes the
time embedding layer and embt ∈Rdte. In SD-v1.5, dte = 1280. Then, for each ResBlock, denoted
as Ri for i = 1, 2, · · · , Nr, where Nr is total number of ResBlocks with time projection layers, the
embt is encoded by time projection layers ri(·) by Fi,t = ri(embt). Notice that ri(·) and e(·) are both
linear layers. Finally, Fi,t is applied to the intermediate activations of each Ri via addition operation,
effectively incorporating temporal information into the Stable Diffusion model.

As observed before [27], time embedding and projection layers exhibit considerable sensitivity to
quantization during PTQ on DM. To address this problem, existing work specifically pays attention
to reconstructing layers related to time embedding [27]. In this study, we propose a more effective

18


---Page Break---
Algorithm 1 Mixed-Precision Algorithm
Input: Candidate bits set b ∈{1, 2, 3}, SD-v1.5 model D, number of total layers L (except for the time
embedding, time projection, the first and last convolutional layers), dataset X, training iterations T, number of
evaluation images K, threshold So, parameter size factor η, number of parameters of the ith layer Ni.
Output: Mixed precision recipe {b∗
i }, i = 1, 2, · · · , L.
1: 1: Obtaining the quantized models.
2: for b = 1 to 3 do
3:
for i = 1 to L do
4:
Quantize the i-th layer to b bits via Eq. (1) and proposed initialization methods in Sec. 4.1 to get model
Di,b;
5:
for t = 1 to T do
6:
Updating weights and scaling factors by minimizing the quantization error using quantization-aware
training on Di,b with data X;
7:
end for
8:
end for
9: end for
10: 2: Calculating quantization error metrics.
11: Generating K images Id via SD-v1.5;
12: for b = 1 to 3 do
13:
for i = 1 to L do
14:
Generating K images Ii,b via quantized model Di,b;
15:
Calculating MSE, Mi,b via images Ii,b and Id;
16:
Calculating PSNR, Pi,b via images Ii,b and Id;
17:
Calculating LPIPS, Li,b via images Ii,b and Id;
18:
Calculating CLIP score drop, Ci,b via images Ii,b and prompts;
19:
end for
20: end for
21: 2: Deciding the optimal precision.
22: Calculating sensitivity score Si,b = Mi,bN −η
i
;
23: for i = 1 to L do
24:
b∗
i ←4;
25:
for b = 3 to 1 do
26:
if Si,b < So then
27:
Assign the i-th layer with b bits with b∗
i ←b;
28:
end if
29:
end for
30: end for
31: Calculating CLIP score drop, Ci,3 and its pth percentile Cp;
32: for i = 1 to L do
33:
if Ci,3 > C90 then
34:
b∗
i ←b∗
i + 1;
35:
end if
36:
if Ci,3 > C95 then
37:
b∗
i ←b∗
i + 1;
38:
end if
39:
if Ci,3 > C98 then
40:
b∗
i ←b∗
i + 1;
41:
end if
42: end for

method. We observe that 1) during the inference stage, for each time step t, the embt and consequently
Fi,t remain constant. 2) In the Stable Diffusion model, the shape of Fi,t are considerably smaller
compared to time embedding and projection layers. Specifically, in SD-v1.5, Fi,t is with the dimension
in {320, 640, 1280} which is largely smaller than time projection layers Wr ∈RD×1280, where
D ∈{320, 640, 1280}. Therefore, we introduce an efficient and lossless method named Time
Embedding Pre-computing and Caching. Specifically, for total Tinf inference time steps, we opt to
store only Tinf time features, rather than retaining the original time embedding layers e(·) and the
time projection layers in the i-th ResBlock ri(·).

The inference time steps are set as 50 or less in most Stable Diffusion models. This method signif-
icantly reduces more than 1280/50 = 25.6× storage requirements and entire computational costs
in terms of time-related layers. Given that the storage size of the pre-computed Fi,t is substantially

19


---Page Break---
smaller than that of the original linear layers, this approach effectively diminishes the average bit of
our quantized model without any performance degradation.

D
Analysis of Symmetric Weight Distribution

In Sec. 4.1, we introduce "Adding Balance Integer" by assuming the weight distribution in Stable
Diffusion is symmetric. Here, we provide more analysis for the assumption. To verify the weight
distribution is symmetric around zero in SD-v1.5, we measure the skewness of the weight distribution
of each layer. Lower skewness indicates a more symmetric weight distribution. As illustrated in Fig.
8, 97% of layers exhibiting skewness between [-0.5, 0.5], this suggests that most layers in SD-v1.5
have symmetric weight distributions.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.5

0.0

0.5

Skewness

Figure 8: Skewness of weight distribution of each layer in SD-v1.5. Lower skewness represents the
weight distribution is more symmetric. 97% layers are with skewness between [-0.5, 0.5], indicating
that most layers have symmetric weight distribution in SD-v1.5.

E
More Details for Quantization Error Across Different Time Steps

In Sec. 4.2, we introduce the "Quantization Error-aware Time Step Sampling" method. Here, we
provide more details for measuring the quantization error from the predicted latent instead of the
predicted noise. During the inference stage, the actual denoising step requires the scaling operation
on the predicted noise in diffusion models. Therefore, directly calculating the quantization error
via noise prediction is not accurate. Instead, we calculate the quantization error in the latent feature
space. We derive the relationship of quantization error calculated from the predicted latent and noise
as follows:

E = Et,x

∥ˆzθfp(t, zt, c) −ˆzθint,s(t, zt, c)∥2
,

= Et,x

"

 1
√¯αt
zt −
√1 −¯αt
√¯αt
ˆϵθfp(t, zt, c)

−
 1
√¯αt
zt −
√1 −¯αt
√¯αt
ˆϵθint,s(t, zt, c)


2#

,

= Et,x

1 −¯αt

¯αt

ˆϵθfp(t, zt, c) −ˆϵθint,s(t, zt, c)
2

,

(9)
where ¯αt is the noise scheduler in [21].

20


---Page Break---
F
Detailed Metrics for Quantization Error by Quantizing Different Layers

In Sec. 3.2, we calculate the various metrics for representing the quantization error when quantizing
different layers. Here, we provide detailed metrics when quantizing each layer of SD-v1.5 to 1, 2,
and 3 bits.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

1000

2000

3000

4000

5000

6000

Quantization Error (MSE)

(a) MSE value caused by the 1-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

500

1000

1500

2000

2500

3000

Quantization Error (MSE)

(b) MSE value caused by the 2-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

500

1000

1500

2000

Quantization Error (MSE)

(c) MSE value caused by the 3-bit quantized layers in SD-v1.5.

Figure 9: MSE value caused by the quantized layers in SD-v1.5..

21


---Page Break---
0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.00

0.01

0.02

0.03

0.04

Quantization Error (CLIP score)

(a) CLIP score degradation caused by the 1-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.00

0.01

0.02

Quantization Error (CLIP score)

(b) CLIP score degradation caused by the 2-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.00

0.01

Quantization Error (CLIP score)

(c) CLIP score degradation caused by the 3-bit quantized layers in SD-v1.5.

Figure 10: CLIP score degradation caused by quantized layers in SD-v1.5.

22


---Page Break---
0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Quantization Error (LPIPS)

(a) LPIPS value of the 1-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.0

0.1

0.2

0.3

0.4

0.5

Quantization Error (LPIPS)

(b) LPIPS value of the 2-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

Quantization Error (LPIPS)

(c) LPIPS value of the 3-bit quantized layers in SD-v1.5.

Figure 11: LPIPS value of quantized layers in SD-v1.5.

23


---Page Break---
0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

5

10

15

20

25

30

Quantization Error (PSNR)

(a) PSNR value of the 1-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

5

10

15

20

25

30

35

Quantization Error (PSNR)

(b) PSNR value of the 2-bit quantized layers in SD-v1.5.

0
10
20
30
40
50
60
70
80
90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250

Layers

0

5

10

15

20

25

30

35

40

Quantization Error (PSNR)

(c) PSNR value of the 3-bit quantized layers in SD-v1.5.

Figure 12: PSNR value of quantized layers in SD-v1.5.

24


---Page Break---
G
More Visualization for Quantization Error by Quantizing Different Layers

In Sec. 3.2, we show the images for representing the quantization error when quantizing different lay-
ers. Here, we provide more visualization for demonstrating the different quantization errors caused by
quantizing different layers to 1 bit. The quantized layers from left to right correspond to the annotated
layers at the bottom: SD-v1.5 w/o quantization, Down.0.0.attn2.toq, Down.0.0.attn2.tok,
Down.0.0.attn2.tov, Down.1.0.attn2.tok, Down.1.1.attn2.tok, Down.2.res.0.conv1,
Up.2.res.2.convshortcut, Up.3.2.attn2.tok, Up.3.res.2.convshortcut.

Figure 13: Quantization errors demonstrated in generated images (via PartiPrompts) after performing
1-bit quantization on different individual layers.

25


---Page Break---
H
1.99 Bits Mixed Precision Recipe

We provide our 1.99 bits recipe in our experiments. During the training and inference stage, we add a
balancing integer to the n-bits values, resulting in log(2n + 1) bits. We calculate the average bits by
P

i log(2b∗
i +1)×Ni+16∗Ntf

Nw
, where b∗
i is the calculated bit-width in the ith layer, Ni is the number of
weights of the ith layer, Ntf is the number of parameters for pre-cached time features, and Nw is the
total number of weights in linear and convolutional layers. We calculate the model size by integrating
all other parameters as 32 bits. The index and name of each layer are listed:

1 down_blocks.0.attentions.0.proj_in: 6
2 down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q: 5
3 down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k: 5
4 down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v: 4
5 down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0: 6
6 down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q: 5
7 down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k: 7
8 down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v: 3
9 down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0: 3
10 down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj: 3
11 down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2: 3
12 down_blocks.0.attentions.0.proj_out: 5
13 down_blocks.0.attentions.1.proj_in: 4
14 down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q: 3
15 down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k: 4
16 down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v: 6
17 down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0: 5
18 down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q: 5
19 down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k: 7
20 down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v: 2
21 down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0: 3
22 down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj: 3
23 down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2: 3
24 down_blocks.0.attentions.1.proj_out: 6
25 down_blocks.0.resnets.0.conv1: 3
26 down_blocks.0.resnets.0.conv2: 3
27 down_blocks.0.resnets.1.conv1: 3
28 down_blocks.0.resnets.1.conv2: 4
29 down_blocks.0.downsamplers.0.conv: 4
30 down_blocks.1.attentions.0.proj_in: 4
31 down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q: 3
32 down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k: 3
33 down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v: 4
34 down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0: 4
35 down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q: 3
36 down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k: 5
37 down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v: 4
38 down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0: 3
39 down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj: 2
40 down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2: 2
41 down_blocks.1.attentions.0.proj_out: 4
42 down_blocks.1.attentions.1.proj_in: 4
43 down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q: 2
44 down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k: 2
45 down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v: 4
46 down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0: 4
47 down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q: 3
48 down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k: 6
49 down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v: 4
50 down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0: 3
51 down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj: 2
52 down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2: 2
53 down_blocks.1.attentions.1.proj_out: 4
54 down_blocks.1.resnets.0.conv1: 3

26


---Page Break---
55 down_blocks.1.resnets.0.conv2: 3
56 down_blocks.1.resnets.0.conv_shortcut: 7
57 down_blocks.1.resnets.1.conv1: 3
58 down_blocks.1.resnets.1.conv2: 2
59 down_blocks.1.downsamplers.0.conv: 4
60 down_blocks.2.attentions.0.proj_in: 3
61 down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q: 3
62 down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k: 2
63 down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v: 3
64 down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0: 3
65 down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q: 3
66 down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k: 4
67 down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v: 4
68 down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0: 3
69 down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj: 2
70 down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2: 1
71 down_blocks.2.attentions.0.proj_out: 3
72 down_blocks.2.attentions.1.proj_in: 4
73 down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q: 4
74 down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k: 2
75 down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v: 3
76 down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0: 3
77 down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q: 3
78 down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k: 4
79 down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v: 4
80 down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0: 3
81 down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj: 2
82 down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2: 2
83 down_blocks.2.attentions.1.proj_out: 4
84 down_blocks.2.resnets.0.conv1: 3
85 down_blocks.2.resnets.0.conv2: 2
86 down_blocks.2.resnets.0.conv_shortcut: 4
87 down_blocks.2.resnets.1.conv1: 2
88 down_blocks.2.resnets.1.conv2: 1
89 down_blocks.2.downsamplers.0.conv: 1
90 down_blocks.3.resnets.0.conv1: 1
91 down_blocks.3.resnets.0.conv2: 1
92 down_blocks.3.resnets.1.conv1: 1
93 down_blocks.3.resnets.1.conv2: 1
94 up_blocks.0.resnets.0.conv1: 1
95 up_blocks.0.resnets.0.conv2: 2
96 up_blocks.0.resnets.0.conv_shortcut: 1
97 up_blocks.0.resnets.1.conv1: 1
98 up_blocks.0.resnets.1.conv2: 1
99 up_blocks.0.resnets.1.conv_shortcut: 1
100 up_blocks.0.resnets.2.conv1: 2
101 up_blocks.0.resnets.2.conv2: 1
102 up_blocks.0.resnets.2.conv_shortcut: 1
103 up_blocks.0.upsamplers.0.conv: 1
104 up_blocks.1.attentions.0.proj_in: 3
105 up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q: 2
106 up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k: 1
107 up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v: 2
108 up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0: 3
109 up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q: 3
110 up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k: 4
111 up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v: 4
112 up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0: 2
113 up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj: 2
114 up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2: 2
115 up_blocks.1.attentions.0.proj_out: 3
116 up_blocks.1.attentions.1.proj_in: 3
117 up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q: 2
118 up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k: 2
119 up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v: 2

27


---Page Break---
120 up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0: 2
121 up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q: 2
122 up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k: 4
123 up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v: 3
124 up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0: 1
125 up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj: 1
126 up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2: 1
127 up_blocks.1.attentions.1.proj_out: 3
128 up_blocks.1.attentions.2.proj_in: 3
129 up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q: 1
130 up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k: 1
131 up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v: 2
132 up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0: 2
133 up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q: 1
134 up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k: 3
135 up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v: 2
136 up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0: 1
137 up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj: 1
138 up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2: 1
139 up_blocks.1.attentions.2.proj_out: 2
140 up_blocks.1.resnets.0.conv1: 1
141 up_blocks.1.resnets.0.conv2: 1
142 up_blocks.1.resnets.0.conv_shortcut: 3
143 up_blocks.1.resnets.1.conv1: 1
144 up_blocks.1.resnets.1.conv2: 1
145 up_blocks.1.resnets.1.conv_shortcut: 3
146 up_blocks.1.resnets.2.conv1: 1
147 up_blocks.1.resnets.2.conv2: 1
148 up_blocks.1.resnets.2.conv_shortcut: 3
149 up_blocks.1.upsamplers.0.conv: 2
150 up_blocks.2.attentions.0.proj_in: 4
151 up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q: 2
152 up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k: 2
153 up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v: 3
154 up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0: 3
155 up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q: 1
156 up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k: 2
157 up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v: 1
158 up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0: 1
159 up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj: 1
160 up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2: 1
161 up_blocks.2.attentions.0.proj_out: 3
162 up_blocks.2.attentions.1.proj_in: 4
163 up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q: 2
164 up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k: 3
165 up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v: 3
166 up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0: 3
167 up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q: 1
168 up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k: 3
169 up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v: 1
170 up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0: 1
171 up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj: 1
172 up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2: 1
173 up_blocks.2.attentions.1.proj_out: 3
174 up_blocks.2.attentions.2.proj_in: 4
175 up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q: 2
176 up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k: 2
177 up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v: 2
178 up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0: 3
179 up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q: 2
180 up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k: 3
181 up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v: 1
182 up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0: 1
183 up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj: 1
184 up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2: 1

28


---Page Break---
185 up_blocks.2.attentions.2.proj_out: 3
186 up_blocks.2.resnets.0.conv1: 1
187 up_blocks.2.resnets.0.conv2: 2
188 up_blocks.2.resnets.0.conv_shortcut: 4
189 up_blocks.2.resnets.1.conv1: 1
190 up_blocks.2.resnets.1.conv2: 2
191 up_blocks.2.resnets.1.conv_shortcut: 4
192 up_blocks.2.resnets.2.conv1: 1
193 up_blocks.2.resnets.2.conv2: 1
194 up_blocks.2.resnets.2.conv_shortcut: 4
195 up_blocks.2.upsamplers.0.conv: 3
196 up_blocks.3.attentions.0.proj_in: 4
197 up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q: 2
198 up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k: 2
199 up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v: 6
200 up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0: 3
201 up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q: 2
202 up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k: 3
203 up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v: 1
204 up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0: 1
205 up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj: 1
206 up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2: 1
207 up_blocks.3.attentions.0.proj_out: 4
208 up_blocks.3.attentions.1.proj_in: 4
209 up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q: 2
210 up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k: 3
211 up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v: 5
212 up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0: 3
213 up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q: 2
214 up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k: 3
215 up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v: 1
216 up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0: 1
217 up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj: 2
218 up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2: 1
219 up_blocks.3.attentions.1.proj_out: 4
220 up_blocks.3.attentions.2.proj_in: 6
221 up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q: 2
222 up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k: 3
223 up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v: 4
224 up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0: 3
225 up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q: 4
226 up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k: 5
227 up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v: 1
228 up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0: 1
229 up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj: 3
230 up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2: 2
231 up_blocks.3.attentions.2.proj_out: 4
232 up_blocks.3.resnets.0.conv1: 1
233 up_blocks.3.resnets.0.conv2: 2
234 up_blocks.3.resnets.0.conv_shortcut: 4
235 up_blocks.3.resnets.1.conv1: 2
236 up_blocks.3.resnets.1.conv2: 2
237 up_blocks.3.resnets.1.conv_shortcut: 4
238 up_blocks.3.resnets.2.conv1: 2
239 up_blocks.3.resnets.2.conv2: 2
240 up_blocks.3.resnets.2.conv_shortcut: 4
241 mid_block.attentions.0.proj_in: 2
242 mid_block.attentions.0.transformer_blocks.0.attn1.to_q: 3
243 mid_block.attentions.0.transformer_blocks.0.attn1.to_k: 1
244 mid_block.attentions.0.transformer_blocks.0.attn1.to_v: 2
245 mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0: 2
246 mid_block.attentions.0.transformer_blocks.0.attn2.to_q: 1
247 mid_block.attentions.0.transformer_blocks.0.attn2.to_k: 4
248 mid_block.attentions.0.transformer_blocks.0.attn2.to_v: 4
249 mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0: 3

29


---Page Break---
250 mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj: 2
251 mid_block.attentions.0.transformer_blocks.0.ff.net.2: 1
252 mid_block.attentions.0.proj_out: 3
253 mid_block.resnets.0.conv1: 1
254 mid_block.resnets.0.conv2: 1
255 mid_block.resnets.1.conv1: 1
256 mid_block.resnets.1.conv2: 1
conv_in: 8
conv_out: 8

I
Details for Evaluation Metrics

In Sec. 5, we measure the performance on various metrics such as TIFA, GenEval, CLIP score and
FID. Here, we provide more details for these metrics.

TIFA Score. TIFA v1.0 [26] aims to measure the faithfulness of generated images. It includes various
4K text prompts sampled from the MS-COCO captions [47], DrawBench [72], PartiPrompts [86],
and PaintSkill [6], associated with a pre-generated set of question-answer pairs, resulting in 25K
questions covering 4.5K diverse elements. Image faithfulness is measured by determining if the VQA
model can accurately answer the questions from the generated images.

GenEval Score. GenEval [14] measures the consistency between the generated images and the
description, including 6 different tasks: single object, two object, counting, colors, position, color
attribution. All text prompts are generated from task-specific templates filled in with: randomly
sampled object names from MS-COCO [47], colors from Berlin-Kay basic color theory, numbers
with 2, 3, 4, and relative positions from "above", "below", "to the left of", or "to the right of". We
adopt the pre-trained object detection model Mask2Former (Swin-S-8×2) [5] for evaluation.

CLIP score and FID. CLIP score measures measure the similarity between text prompts and
corresponding generated images. FID is used to evaluate the quality of generated images by measuring
the distance between the distributions of features extracted from generated images and target images.
In the main experiments, evaluation are measured based on MS-COCO 2014 validation set with 30K
image-caption pairs [47]. We adopt ViT-B/32 model to evaluate the CLIP score in our experiments.

J
Human Evaluation

In Sec. 5, we provide the human evaluation results. Here, we provide more detailed human evaluation
with category and challenge comparisons on PartiPrompts (P2), comparing Stable Diffusion v1.5 and
BitsFusion, with the question: Given a prompt, which image has better aesthetics and image-text
alignment? Our model is selected 888 times out of 1632 comparisons, indicating a general preference
over SD-v1.5, which is chosen 744 times, demonstrating more appealing and accurate generated
images.

J.1
Analysis on Categories

Illustrations, People, and Arts. Our model significantly outperforms SD-v1.5 in generating illustra-
tions (77 wins out of 124), images of people (101 out of 174), and arts (45 out of 65).

Outdoor and Indoor Scenes. Our model also shows strength in generating both outdoor (73 out of
131) and indoor scenes (23 out of 40), suggesting better environmental rendering capabilities.

J.2
Analysis on Challenges

Complex and Fine-grained Detail: Our model excels in generating images with complex details (73
out of 113) and fine-grained details (173 out of 312), suggesting advanced capabilities in maintaining
detail at varying complexity levels.

Imagination and Style & Format: Our model also shows a strong performance in tasks requiring
imaginative (92 out of 149) and stylistic diversity (118 out of 204), highlighting its flexibility and
creative handling of artistic elements.

30


---Page Break---
Produce & Plants

Animals

Indoor Scenes

Illustrations

People

Abstract

Outdoor Scenes

Artifacts

World Knowledge

Food & Beverage

Vehicles

Arts

52

51.9

57.5

62.1

58

51

55.7

54.7

44.9

58.1

54.8

69.2

48

48.1

42.5

37.9

42

49

44.3

45.3

55.1

41.9

45.2

30.8

Ours
SD-v1.5

Figure 14: Human evaluation across particular categories.

Imagination

Simple Detail

Complex

Fine-grained Detail

Writing & Symbols

Basic

Properties & Positioning

Quantity

Style & Format

Linguistic Structures

Perspective

61.7

46.1

64.6

55.4

57.1

49.1

57.1

56.7

57.8

55.7

48.6

38.3

53.9

35.4

44.6

42.9

50.9

42.9

43.3

42.2

44.3

51.4

Ours
SD-v1.5

Figure 15: Human evaluation across particular challenges.

The strong performance in imaginative and artistic categories presents an opportunity to target
applications in creative industries, such as digital art and entertainment, where these capabilities can
be particularly valuable.

K
Evaluation on Different Schedulers

In the main experiments in Sec. 5, we leverage the PNDM scheduler to generate images. Here,
we measured the performance of different schedulers, such as DDIM [78] and DPMSolver [55],
to demonstrate the generality and effectiveness of BitsFusion. We set 50 inference steps and fix

31


---Page Break---
the random seed as 1024. As shown in Fig. 16, BitsFusionconsistently outperforms SD-v1.5 with
different schedulers.

2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0.74

0.76

0.78

0.80

0.82

TIFA Scores

SD-v1.5 (32 bits)
Ours-II (1.99 bits)

2.5
3.5
4.5
5.5
6.5
7.5
8.5
9.5
CFG Scales

0.74

0.76

0.78

0.80

0.82

TIFA Scores

SD-v1.5 (32 bits)
Ours-II (1.99 bits)

(a) DDIM
(b) DPMSolver

Figure 16: TIFA scores comparisons between SD-v1.5 and BitsFusion, with different schedulers. Left:
TIFA scores measured with DDIM [78] scheduler. Right: TIFA score measured with DPMSolver [55]
scheduler.

L
Detailed GenEval Results

In Sec. 5, we provide the overall GenEval results. Here, we provide detailed GenEval results for
further comparisons as illustrated in Tab. 8.

Table 8: Detailed GenEval with different CFG scales.

Method
Overall
Single
Two
Counting
Colors
Position
Color Attribution

Guidance Scale = 2.5

SD-v1.5
0.3589
0.9350
0.2626
0.2775
0.6043
0.0340
0.0400
Ours-I
0.3353
0.9075
0.2444
0.2550
0.5426
0.0280
0.0340
Ours-II
0.4024
0.8975
0.3859
0.2750
0.6979
0.0560
0.1020

Guidance Scale = 3.5

SD-v1.5
0.3879
0.9400
0.3010
0.3275
0.6787
0.0300
0.0500
Ours-I
0.3650
0.9500
0.2808
0.2575
0.6277
0.0280
0.0460
Ours-II
0.4370
0.9350
0.4727
0.3125
0.7340
0.0600
0.1080

Guidance Scale = 4.5

SD-v1.5
0.4056
0.9700
0.3010
0.3200
0.7426
0.0340
0.0660
Ours-I
0.3851
0.9500
0.3091
0.3100
0.6574
0.0340
0.0500
Ours-II
0.4516
0.9575
0.4788
0.3450
0.7723
0.0520
0.1040

Guidance Scale = 5.5

SD-v1.5
0.4094
0.9750
0.3111
0.3325
0.7319
0.0400
0.0660
Ours-I
0.4039
0.9675
0.3232
0.3425
0.7000
0.0300
0.0600
Ours-II
0.4567
0.9600
0.4909
0.3175
0.7979
0.0540
0.1200

Guidance Scale = 6.5

SD-v1.5
0.4224
0.9800
0.3293
0.3725
0.7447
0.0400
0.0680
Ours-I
0.4161
0.9675
0.3414
0.3350
0.7425
0.0360
0.0740
Ours-II
0.4612
0.9750
0.4990
0.3275
0.7957
0.0540
0.1160

Guidance Scale = 7.5

SD-v1.5
0.4262
0.9775
0.3313
0.3850
0.7596
0.0440
0.0600
Ours-I
0.4226
0.9775
0.3495
0.3600
0.7447
0.0360
0.0680
Ours-II
0.4682
0.9800
0.5091
0.3300
0.8085
0.0680
0.1140

Guidance Scale = 8.5

SD-v1.5
0.4271
0.9825
0.3273
0.3925
0.7745
0.0320
0.0540
Ours-I
0.4269
0.9800
0.3616
0.3475
0.7702
0.0400
0.0620
Ours-II
0.4747
0.9700
0.5111
0.3675
0.8213
0.0620
0.1160

Guidance Scale = 9.5

SD-v1.5
0.4260
0.9825
0.3556
0.3825
0.7553
0.0280
0.0520
Ours-I
0.4190
0.9825
0.3495
0.3450
0.7447
0.0300
0.0620
Ours-II
0.4736
0.9700
0.5192
0.3625
0.8277
0.0560
0.1060

32


---Page Break---
M
More Comparisons

We provide the prompts for the images featured in the Fig. 1. Additionally, we provide more generated
images for the comparison.

M.1
Prompts

Prompts of Fig. 1 from left to right are:

1. a portrait of an anthropomorphic cyberpunk raccoon smoking a cigar, cyberpunk!,
fantasy, elegant, digital painting, artstation, concept art, matte, sharp focus,
illustration, art by josan Gonzalez

2. Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach
whirlpool engine, volumetric lighting, spectacular, ambient lights,
light pollution, cinematic atmosphere, art nouveau style,
illustration art artwork by SenseiJaye, intricate detail.

3. tropical island, 8 k, high resolution, detailed charcoal drawing,
beautiful hd, art nouveau, concept art, colourful, in the style of vadym meller

4. anthropomorphic art of a fox wearing a white suit, white cowboy hat,
and sunglasses, smoking a cigar, texas inspired clothing by artgerm,
victo ngai, ryohei hase, artstation. highly detailed digital painting,
smooth, global illumination, fantasy art by greg rutkowsky, karl spitzweg

5. a painting of a lantina elder woman by Leonardo da Vinci . details, smooth,
sharp focus, illustration, realistic, cinematic, artstation, award winning, rgb ,
unreal engine, octane render, cinematic light, macro, depth of field, blur,
red light and clouds from the back, highly detailed epic cinematic concept art CG
render made in Maya, Blender and Photoshop, octane render, excellent composition,
dynamic dramatic cinematic lighting, aesthetic, very inspirational, arthouse.

6. panda mad scientist mixing sparkling chemicals, high-contrast painting

7. An astronaut riding a horse on the moon, oil painting by Van Gogh.

8. A red dragon dressed in a tuxedo and playing chess. The chess pieces are
fashioned after robots.

M.2
Additional Image Comparisons

We provide more images for further comparisons. For each set of two rows, the top row displays
images generated using the full-precision Stable Diffusion v1.5, while the bottom row features images
generated from BitsFusion, where the weights of UNet are quantized into 1.99 bits and the model
size is 7.9× smaller than the one from SD-v1.5. All the images are synthesized under the setting of
using PNDM sampler with 50 sampling steps and random seed as 1024.

33


---Page Break---
a
b
c
d
e
f

Figure 17: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: A person standing on the desert, desert
waves, gossip illustration, half red, half blue, abstract image of sand, clear style, trendy illustration,
outdoor, top view, clear style, precision art, ultra high definition image; b: A detailed oil painting of
an old sea captain, steering his ship through a storm. Saltwater is splashing against his weathered
face, determination in his eyes. Twirling malevolent clouds are seen above and stern waves threaten
to submerge the ship while seagulls dive and twirl through the chaotic landscape. Thunder and lights
embark in the distance, illuminating the scene with an eerie green glow.; c: A solitary figure shrouded
in mists peers up from the cobble stone street at the imposing and dark gothic buildings surrounding
it. an old-fashioned lamp shines nearby. oil painting.; d: A deep forest clearing with a mirrored
pond reflecting a galaxy-filled night sky; e: a handsome 24 years old boy in the middle with sky color
background wearing eye glasses, it’s super detailed with anime style, it’s a portrait with delicated
eyes and nice looking face; f: A dog that has been meditating all the time.

34


---Page Break---
a
b
c
d
e
f

Figure 18: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: A small cactus with a happy face in the
Sahara desert.; b: A middle-aged woman of Asian descent, her dark hair streaked with silver, appears
fractured and splintered, intricately embedded within a sea of broken porcelain. The porcelain
glistens with splatter paint patterns in a harmonious blend of glossy and matte blues, greens, oranges,
and reds, capturing her dance in a surreal juxtaposition of movement and stillness. Her skin tone, a
light hue like the porcelain, adds an almost mystical quality to her form.; c: A high contrast portrait
photo of a fluffy hamster wearing an orange beanie and sunglasses holding a sign that says "Let’s
PAINT!”; d: An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought
pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen
as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down
shirt , he wears a brown beret and glasses and has a very professorial appearance, and the end he
offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very
cinematic with the golden light and the Parisian streets and city in the background, depth of field,
cinematic 35mm film.; e: poster of a mechanical cat, techical Schematics viewed from front and side
view on light white blueprint paper, illustartion drafting style, illustation, typography, conceptual art,
dark fantasy steampunk, cinematic, dark fantasy; f: I want to supplement vitamin c, please help me
paint related food.

35


---Page Break---
a
b
c
d
e
f

Figure 19: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: new cyborg with cybertronic gadgets and
vr helmet, hard surface, beautiful colours, sharp textures, shiny shapes, acid screen, biotechnology,
tim hildebrandt, bruce pennington, donato giancola, larry elmore, masterpiece, trending on artstation,
featured on pixiv, cinematic composition, dramatic pose, beautiful lighting, sharp, details, hyper
- detailed, hd, hdr, 4 k, 8 k; b: portrait of teenage aphrodite, light freckles, curly copper colored
hair, smiling kindly, wearing an embroidered white linen dress with lace neckline, intricate, elegant,
mother of pearl jewelry, glowing lights, highly detailed, digital painting, artstation, concept art,
smooth, sharp focus, illustration, art by wlop, mucha, artgerm, and greg Rutkowski; c: portrait of
a dystopian cute dog wearing an outfit inspired by the handmaid ï¿½ s tale ( 2 0 1 7 ), intricate,
headshot, highly detailed, digital painting, artstation, concept art, sharp focus, cinematic lighting,
digital painting, art by artgerm and greg rutkowski, alphonse mucha, cgsociety; d: Portrait of a
man by Greg Rutkowski, symmetrical face, a marine with a helmet, using a VR Headset, Kubric
Stare, crooked smile, he’s wearing a tacitcal gear, highly detailed portrait, scifi, digital painting,
artstation, book cover, cyberpunk, concept art, smooth, sharp foccus ilustration, Artstation HQ; e:
Film still of female Saul Goodman wearing a catmaid outfit, from Red Dead Redemption 2 (2018
video game), trending on artstation, artstationHD, artstationHQ; f: oil paining of robotic humanoid,
intricate mechanisms, highly detailed, professional digital painting, Unreal Engine 5, Photorealism,
HD quality, 8k resolution, cinema 4d, 3D, cinematic, professional photography, art by artgerm and
greg rutkowski and alphonse mucha and loish and WLOP

36


---Page Break---
a
b
c
d
e
f

Figure 20: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: anthropomorphic tetracontagon
head in opal edgy darknimite mudskipper, intricate, elegant, highly detailed animal monster, digital
painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm, bob eggleton,
michael whelan, stephen hickman, richard corben, wayne barlowe, trending on artstation and
greg rutkowski and alphonse mucha, 8 k; b: background shows moon, many light effects, particle,
lights, gems, symmetrical!!! centered portrait dark witch, large cloak, fantasy forest landscape,
dragon scales, fantasy magic, undercut hairstyle, short purple black fade hair, dark light night,
intricate, elegant, sharp focus, digital painting, concept art, matte, art by wlop and artgerm and
greg rutkowski and alphonse mucha, masterpiece; c: cat seahorse fursona, autistic bisexual graphic
designer and musician, long haired attractive androgynous fluffy humanoid character design, sharp
focus, weirdcore voidpunk digital art by artgerm, akihiko yoshida, louis wain, simon stalenhag,
wlop, noah bradley, furaffinity, artstation hd, trending on deviantart; d: concept art of ruins of a
victorian city burning down by j. c. leyendecker, wlop, ruins, dramatic, octane render, epic painting,
extremely detailed, 8 k; e: hyperrealistic Gerald Gallego as a killer clown from outer space, trending
on artstation, portrait, sharp focus, illustration, art by artgerm and greg rutkowski and magali
Villeneuve; f: low angle photo of a squirrel dj wearing on - ear headphones and colored sunglasses,
stadning at a dj table playing techno music at a dance club, hyperrealistic, highly detailed, intricate,
smoke, colored lights, concept art, digital art, oil painting, character design by charlie bowater, ross
tran, artgerm, makoto shinkai, wlop

a
b
c
d
e
f

Figure 21: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: a photograph of an ostrich wearing a
fedora and singing soulfully into a microphone; b: a pirate ship landing on the moon; c: a pumpkin
with a candle in it; d: a rabbit wearing a black tophat and monocle; e: a red sports car on the road;
f: a robot cooking in the kitchen.

37


---Page Break---
a
b
c
d
e
f

Figure 22: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: a baby daikon radish in a tutu; b: a
baby penguin wearing a blue hat, red gloves, green shirt, and yellow pants; c: a woman with long
black hair and dark skin; d: an emoji of a baby penguin wearing a blue hat, red gloves, green shirt,
and yellow pants; e: a blue sports car on the road; f: a butterfly.

a
b
c
d
e
f

Figure 23: Top: Images generated from full-precision Stable Diffusion v1.5. Bottom: Images
generated from BitsFusion. Prompts from left to right are: a: Helmet of a forgotten Deity, clowing
corals, extremly detailed digital painting, in the style of Fenghua Zhong and Ruan Jia and jeremy
lipking and Peter Mohrbacher, mystical colors, rim light, beautiful lighting, 8k, stunning scene,
raytracing, octane, trending on artstation; b: Jeff Bezos as a female amazon warrior, closeup, D&D,
fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus,
illustration, hearthstone, art by Artgerm and Greg Rutkowski and Alphonse Mucha; c: Portrait of a
draconic humanoid, HD, illustration, epic, D&D, fantasy, intricate, elegant, highly detailed, digital
painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski
and alphonse mucha, monster hunter illustrations art book; d: [St.Georges slaying a car adorned with
checkered flag. Soviet Propaganda!!! poster!!!, elegant, highly detailed, digital painting, artstation,
concept art, matte, sharp focus, illustration, octane render, unreal engine, photography]; e: a fire
- breathing dragon at a medieval hobbit home, ornate, beautiful, atmosphere, vibe, mist, smoke,
chimney, rain, wet, pristine, puddles, waterfall, clear stream, bridge, forest, flowers, concept art
illustration, color page, 4 k, tone mapping, doll, akihiko yoshida, james jean, andrei riabovitchev,
marc simonetti, yoshitaka amano, digital illustration, greg rutowski, volumetric lighting, sunbeams,
particles; f: portrait of a well-dressed raccoon, oil painting in the style of Rembrandt

38


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]
Justification: We propose BitsFusion that quantizes the UNet from Stable Diffusion v1.5
to 1.99 bits, achieving a model with 7.9× smaller size while even better generation quality
than the original one.

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

Justification: We list the limitations in Section "Limitations" in Appendix.

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

39


---Page Break---
Answer: [NA]
Justification: We do not include theory assumptions and proofs.
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
Justification: We provide the details of the quantization strategy, quantization error met-
rics, mixed precision recipe and implementation setting in the Method, Experiments, and
Appendix.
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

40


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We plan to release our code and trained models to facilitate the research efforts
towards extreme low-bits quantization.
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
Justification: We provide detailed experimental settings in the Experiment section.
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
Justification: Similar to other Stable Diffusion and quantization works, we report the results
on large-scale datasets without the error bar.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

41


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
Justification: We provide the experimental resources in the Experiment section.
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
Justification: We conform with the NeurIPS Code of Ethics.
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
Justification: We list the broader impacts in Section "Broader Impacts" in Appendix.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

42


---Page Break---
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

Justification: This paper does not have such risks.

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

Justification: We claim and cite the asset in the paper.

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

43


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not release new assets.
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
Justification: Our research does not involve crowdsourcing nor research with human subjects.
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
Justification: Our research does not involve crowdsourcing nor research with human subjects.
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

44


---Page Break---
