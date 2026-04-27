ReF-LDM: A Latent Diffusion Model for
Reference-based Face Image Restoration

Chi-Wei Hsiao1
Yu-Lun Liu2
Cheng-Kun Yang1
Sheng-Po Kuo1

Yucheun Kevin Jou1
Chia-Ping Chen1

1MediaTek
2National Yang Ming Chiao Tung University

Abstract

While recent works on blind face image restoration have successfully produced
impressive high-quality (HQ) images with abundant details from low-quality (LQ)
input images, the generated content may not accurately reflect the real appearance
of a person. To address this problem, incorporating well-shot personal images as
additional reference inputs could be a promising strategy. Inspired by the recent
success of the Latent Diffusion Model (LDM), we propose ReF-LDM—an adapta-
tion of LDM designed to generate HQ face images conditioned on one LQ image
and multiple HQ reference images. Our model integrates an effective and efficient
mechanism, CacheKV, to leverage the reference images during the generation
process. Additionally, we design a timestep-scaled identity loss, enabling our
LDM-based model to focus on learning the discriminating features of human faces.
Lastly, we construct FFHQ-Ref, a dataset consisting of 20,405 high-quality (HQ)
face images with corresponding reference images, which can serve as both training
and evaluation data for reference-based face restoration models.

1
Introduction

Recent works [5, 26, 32] have achieved impressive results in generating a realistic high-quality (HQ)
face image from an input low-quality (LQ) image. However, the important features of a person’s face
may be corrupted in the LQ image, and thus the reconstructed image may look like a different person.
To tackle this problem, besides the LQ image, additional HQ images of this person can be used as
reference input. Moreover, allowing multiple reference images may lead to better quality because
they offer more comprehensive appearance of this person in different conditions, e.g., different poses,
expressions, or lighting.

A previous work [16] has explored using multiple reference images for face restoration. Their
method, however, depends on a face landmark model to detect facial components (i.e., eyes, nose, and
mouse), which may become unreliable when the input LQ image is severely degraded. Besides, latent
diffusion model (LDM) [22] has also been used in different image generating tasks with different
input conditions, such as low-resolution images, semantic maps, or sketch images [22, 29].

Inspired by the recent success of LDM, we propose ReF-LDM for reference-based face image
restoration. Unlike previous conditional LDM methods where their input conditions are usually
spatially aligned with the target image, the reference images are not aligned with the target HQ image
in our case. Therefore, we design a CacheKV mechanism, which effectively and efficiently integrates
the reference images, albeit with different poses and expressions. Furthermore, we introduce a
timestep-scaled identity loss to drive the reconstructed image to look like the same person of the
LQ and reference images. Lastly, we also construct a new large-scale dataset of face images with
corresponding reference images, which can serve as both training and evaluation datasets for future
reference-based face restoration research.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
(a) Input LQ image
(b) LDM
(c) ReF-LDM

(d) Input reference images

Figure 1: Reference-based face image restoration. Given an input low-quality face image (a), a
Latent Diffusion Model (LDM) can reconstruct a high-quality image (b); however, it may not be
faithful to the individual’s facial identity. To address this problem, we propose ReF-LDM, which
restores a high-quality image with faithful details (c) by utilizing additional reference images (d).

With the above components, our ReF-LDM outperforms recent state-of-the-art methods with a
significant improvement in face identity similarity. Extensive ablation studies for the proposed
CacheKV mechanism and timestep-scaled identity loss are also conducted and reported. The main
contributions of this work can be summarized as:

• We propose ReF-LDM, which features an effective and efficient CacheKV mechanism, for
restoring an LQ face image using multiple reference images.
• We introduce a timestep-scaled identity loss, which considers the characteristics of diffusion
models and helps ReF-LDM better learn the discriminating features of human identities.
• We construct FFHQ-Ref, a dataset comprising 20,406 high-quality face images and their corre-
sponding reference images, to facilitate the advance of reference-based face image restoration.

2
Related work

Face restoration without personal reference images
Numerous studies have been proposed
for blind face image restoration [28, 2, 5, 32, 20, 13, 26]. Recent works such as VQFR [5] and
CodeFormer [32] have achieved promising results by exploiting VQGAN, while DAEFR [26] further
employs a dual-branch encoder to mitigate the domain gap between LQ and HQ images. Inspired by
the success of diffusion models, several works [23, 31, 17, 27, 25] have adopted diffusion models for
face image restoration. However, as these methods do not leverage reference images, the restored
images may differ from the authentic facial appearance of a person, especially when an input image
is severely degraded.

Face restoration with personal reference images
Several methods [14, 15, 16, 19] have attempted
to utilize additional reference images to enhance personal fidelity in face restoration. GFRNet [14]
warps a single reference image to match the face pose of the LQ image, while ASFNet [15] selects
the reference image with the closest matching facial landmarks to serve as the network input. Closer
to the setting of this work, DMDNet [16] also utilizes multiple reference images. It detects facial

2


---Page Break---
landmarks on the LQ image and the reference images to extract features of facial components, and
then integrates these features into the model by querying the corresponding components. However,
their method relies on landmark detection, which may not be robust on severely degraded LQ images.
In contrast, our ReF-LDM implicitly learns the correspondences between the features of the LQ image
and the reference images, without the need for landmark detection. From a different perspective,
MyStyle [19] adopts a per-person optimization setting, leveraging hundreds of images of an individual
to define a personalized subspace within the latent space of a StyleGAN [12]. In comparison, our
approach offers greater flexibility, capable of utilizing one to several reference images without the
need for personalized model optimization for each individual.

Latent diffusion models with image conditions
Previous work demonstrates that LDM can
generate an image from a low-resolution image by simple channel-axis concatenation [22]. However,
reference images in our task are not spatially aligned with the target HQ image, thus requiring a
more sophisticated integration mechanism. MasaCtrl [1] achieves text-to-image synthesis with a
single reference image by replacing the original keys and values tokens with those from the reference
image. However, their solution requires passing the reference image through the denoising network
for multiple timesteps, which increases computation and limits its feasibility for extending to multiple
reference images. In contrast, we propose an efficient CacheKV mechanism that leverages multiple
reference images by eliminating the redundant network passes.

3
The proposed ReF-LDM model

In this section, we present the proposed ReF-LDM model. We introduce the network architecture in
Sec. 3.1, where a CacheKV mechanism is designed to leverage reference images. We illustrate how
to train our model with the timestep-scaled identity loss in Sec. 3.2.

Figure 2: The proposed ReF-LDM pipeline. Our model accepts a low-quality image and multiple
high-quality reference images as input and generates a high-quality image. The blue top panel alone
represents a typical LDM [22] denoising process. For an LQ image xLQ, we concatenate its latent zLQ
with zt along the channel axis to serve as the input for the denoising U-net. For the reference images
{xref}, we design a CacheKV mechanism, depicted in the red panel, to extract and cache their key
and value tokens using the same denoising U-net for just one time. These cached KV tokens can then
be utlized repeatedly in each of the T timesteps of the main denoising process. During training, we
adopt the classic LDM loss (LLDM) and introduce a timestep-scaled identity loss (Ltime ID).

3


---Page Break---
3.1
Model architecture of ReF-LDM

The proposed ReF-LDM accepts an input LQ image and multiple reference images to generate a
target HQ image. Its model architecture is based on the latent diffusion model [22], with additional
designs to incorporate the input LQ image and the reference images.

3.1.1
Preliminaries on Latent Diffusion Model

To generate an image, an image diffusion model [8] starts from a noisy image xT ∈RH×W ×3,
initialized with a Gaussian distribution, and gradually denoises it to a clean image x0 with a denoising
network over T timesteps. A latent diffusion model [22] operates similarly, but the diffusion process
takes place in a more compact latent space of a pre-trained and frozen autoencoder (encoder E and
decoder D). That is, it begins with a random latent zT ∈RHz×Wz×Cz and progressively denoises it
to a clean latent z0. A clean image is then generated by passing the clean latent through the decoder
during the inference phase, i.e., x0 = D(z0); conversely, a ground truth clean latent is obtained by
encoding a clean image with the encoder during the training phase, i.e., z∗
0 = E(x∗
0). A typical choice
for the denoising network is a U-net with self-attention layers at multiple scales.

3.1.2
CacheKV: a mechanism for incorporating reference images

As illustrated in Fig. 2, our ReF-LDM leverages an input LQ image xLQ and multiple reference images
{xref} to generate a target HQ image xHQ. For an LQ image, we simply concatenate its latent encoded
by the frozen encoder, zLQ = E(xLQ), with the diffusion denoising latent zt along the channel axis to
serve as the input of the denoising U-net. For reference images, we design a CacheKV mechanism.
Essentially, we extract and cache the features of reference images using the same denoising U-net
just once; these cached features can then be used repeatedly at each of the T timesteps in the main
denoising process. Specifically, we pass the encoded latent of each reference image, zref = E(xref),
through the U-net to extract their keys and values (KVs) at each self-attention layer and store them
in a CacheKV. Subsequently, during the main diffusion process, within each self-attention layer of
the U-net, we concatenate the reference KVs (from the corresponding self-attention layer) with the
main KVs along the token axis. This mechanism enables the U-net to incorporate the additional KVs
from the reference images into the main denoising process. When extracting KVs from the reference
images, we use a timestep embedding of t = 0 and pad zref with a zero tensor to accommodate the
additional channels introduced for the LQ image.

To summarize, for inference, we first run the U-net once to extract CacheKV from the reference
images; subsequently, we proceed through the main denoising process for T timesteps, during which
the U-net integrates zLQ and reference CacheKV. For training, in each iteration, we first run the U-net
to extract CacheKV, and then we run the U-net again to estimate the target latent from a sampled
noisy latent zt, incorporating the conditions zLQ and reference CacheKV.

3.1.3
Comparing CacheKV with other designs

There are other intuitive designs for integrating the reference latents {zref} into the diffusion denoising
process. However, they are either ineffective or computationally inefficient compared to the proposed
CacheKV. The quantitative evaluation and computational analysis is reported in Sec. 5.2.1. We depict
these designs in Fig. 3 and provide an intuitive explanation as follows:

• Channel-concatenation: Concatenating the condition with zt along the channel axis works
well for LQ images (and for other 2D conditions such as semantic maps [22]); however, it is not
effective for reference images. A critical difference between these conditions is that—while the
LQ image is spatially aligned with the target HQ image, the reference images are not. Therefore,
it is challenging for the model to leverage reference images using simple channel-concatenation.
• Cross-attention: Cross-attention layers have been proven useful for text conditions in text-
to-image models [22]. In our ablation experiment, we insert a cross-attention layer after each
self-attention layer and use the reference latents {zref} to produce keys and values. While
cross-attention appears to have the potential to address the spatial misalignment problem, it still
fails to effectively utilize the reference images. The difference between our CacheKV and the
cross-attention setting is that CacheKV provides the reference images in a more aligned feature
space for the main denoising process to leverage. Specifically, the CacheKV is extracted using
the same U-net and the corresponding self-attention layer as in the main denoising process. In

4


---Page Break---
contrast, the cross-attention setting processes the reference images only with the frozen encoder,
resulting in features that are less aligned with those in the U-net of the denoising process.
• Spatial-concatenation: Concatenating {zref} with zt along the spatial dimension to serve
as the input for U-net also effectively leverages the reference images. Conceptually, spatial-
concatenation treats reference images in a very similar way to our CacheKV. In both mechanisms,
{zref} are processed through the denoising U-net, allowing the reference KVs to be accessed
by the queries (Qs) of the main diffusion latent zt. However, spatial-concatenation requires
significantly more computational resources compared to our CacheKV. It passes {zref} with zt
to the U-net at each of the T denoising timesteps, whereas CacheKV only passes {zref} through
the U-net once. Moreover, spatial-concatenation also requires significantly more GPU memory,
as the spatial size of the input for the U-net increases with the number of reference images.
As for a self-attention layer in the U-net, both mechanisms increase memory usage; CacheKV
introduces additional reference KVs, while spatial-concatenation introduces reference QKVs.

(a) Channel-concatenation
(b) Cross-attention

(c) Spatial-concatenation
(d) CacheKV (ours)

Figure 3: Different mechanisms for incorporating reference images into the main denoising process.

3.2
Timestep-scaled identity loss

3.2.1
Timestep-scaled identity loss

As this work aims for face image restoration, we employ the identity loss to enhance face similarity,
which is adopted in many face-related tasks [9, 21, 28]. The identity loss minimizes the distance
within the embedding space of a face recognition model, thereby capturing the discriminating features
of human faces more effectively than the plain RGB pixel space. In our experiments, we use the
ArcFace model [3] with cosine distance between the 1D embedding vectors as the identity loss.

However, naively adding identity loss to the training of ReF-LDM significantly worsens the image
quality. One possible explanation might be that, the one-step model prediction x0|t = D(z0|zt) at
a very noisy timestep (e.g., t = T) is very different from a natural face image and thus out of the
distribution that the ArcFace model is trained on; therefore, the identity loss provides ineffective
supervision for diffusion models at large timesteps.

5


---Page Break---
Based on this assumption, we propose a timestep-scaled identity loss, where a timestep-dependent
scaling factor is introduced to scale down the identity loss when a larger timestep is sampled in a
training step. Specifically, the timestep-scaled identity loss is defined as:

Ltime ID = √¯αt · LID = √¯αt ·

1 −
R(x) · R(x∗)
∥R(x)∥∥R(x∗)∥


,
(1)

where R is a face recognition model, and √¯αt follows the definition in a typical diffusion process [8,
22] in which a noisy latent zt is sampled given a clean latent z∗
0 as:

q(zt|z∗
0) = N(√¯αtz∗
0, (1 −¯αt)I)
(2)

3.2.2
Training ReF-LDM with timestep-scaled identity loss

We train our ReF-LDM with the classic LDM loss and the proposed timestep-scaled identity loss:

Ltotal = LLDM + λtime ID Ltime ID
(3)

Recall that the denoising U-net estimates the target latent in the latent space of the frozen autoencoder,
and a typical LLDM is computed as the L1 distance between the estimated latent and the target latent.
To compute the identity loss with the face recognition model, which accepts an image as input, we
decode the estimated latent into the image space using the frozen decoder, i.e, x0 = D(z0). The
experiments in Sec. 5.2.2 show that timestep-scaled identity loss can improve face similarity without
degrading image quality, unlike the naive usage of identity loss.

4
FFHQ-Ref dataset

Recent works for non-reference-based face restoration commonly train their models with FFHQ
dataset [12], which comprises 70,000 high-quality face images of wide appearance variety with
appropriate licenses crawled from Flickr. These images are not provided with reference labels
originally; however, we find that a good portion of the images are of the same identities. Thus,
we construct a reference-based dataset—FFHQ-Ref—based on the FFHQ dataset, with careful
consideration described as follows.

4.1
Finding reference images of the same identity

To determine whether two images belong to the same identity, we utilize the face recognition
model ArcFace [3]. Specifically, we first extract the 1D ArcFace embeddings for all images. Then,
for each image, we compute the cosine distances between its embedding and the embeddings of
all other images. A distance less than a threshold r = 0.4 indicates that the images are valid
references belonging to the same person. Following this procedure, we identify 20,405 images with
corresponding reference images.

4.2
Splitting data according to identity

To enable the FFHQ-Ref dataset to serve as both training and evaluation datasets for reference-based
face restoration models, we divide the images into train, validation, and test splits. However, random
data splitting may result in the train and test splits containing images of the same individual, which is
not ideal for a fair evaluation. To ensure that all images of a single identity are assigned to only one
data split, we group the images based on their identities. Specifically, we consider identity grouping
as a graph problem, where each image acts as a vertex and any pair of images with a distance less
than r are connected by edges. We then apply the connected component algorithm from graph theory,
where each connected component represents a group of images belonging to the same person. Finally,
we identified 6,523 identities and divided them into three splits: a train split with 18,816 images of
6,073 identities, a validation split with 732 images of 300 identities , and a test split with 857 images
of 150 identities. We report more statistics in Appendix C.

4.3
Constructing evaluation dataset with practical considerations

Practical Considerations
For a fair and meaningful evaluation, the input reference images should
not be excessively similar to the target image; hence, we set a minimum cosine distance threshold

6


---Page Break---
of 0.1 for the test set. Additionally, we manually check the images in the test split to verify that all
reference images indeed correspond to the same identity. Furthermore, in the context of reference-
based face restoration applications, it is preferable to select input reference images that capture a more
comprehensive representation of a person’s appearance, such as varying face poses or expressions.
Although a target image in the test split of our FFHQ-Ref may have two to nine reference images,
different reference-based methods may have their own constraints on the maximum number of input
reference images. To emulate a more representative set of reference images, we sort all available
reference images of a target image using farthest point sampling on the ArcFace distance.

Degradation synthesis for input LQ images
For synthesizing input LQ images from ground truth
HQ images, we follow the degradation model used in previous works [28, 5, 32]:

xLQ = {[(xHQ ∗kσ) ↓r +nδ]JPEGq} ↑r,
(4)

where an HQ image is blurred with a Gaussian kernel kσ, downsampled by r scale, added with a
Gaussian noise nδ, compressed with JPEG quality level q, and upscaled to the original size.

We construct two evaluation datasets with different degradation levels:

• FFHQ-Ref-Moderate: σ, r, δ, and q are sampled from [0, 8], [1, 8], [0, 15], and [60, 100].
• FFHQ-Ref-Severe: σ, r, δ, and q are sampled from [8, 16], [8, 32], [0, 20], and [30, 100].

4.4
Comparison between FFHQ-Ref and existing datasets

Table 1 summarizes the differences between our proposed FFHQ-Ref and existing datasets. While the
CelebRef-HQ dataset [16] has been constructed to train and evaluate reference-based face restoration
models, our FFHQ-Ref dataset contains twice as many images and six times the number of identities
compared to CelebRef-HQ. Moreover, built upon FFHQ [12], FFHQ-Ref provides superior image
quality over CelebRef-HQ, as indicated by the lower NIQE score (3.68 vs. 3.97). Some ground-
truth images in CelebRef-HQ are affected by watermarks and mirror padding artifacts, as shown in
Appendix B.

Table 1: Comparison between the proposed FFHQ-Ref and existing datasets.
Dataset
With reference
Licensed
Quality
Images
Identities

FFHQ [12]
✓
✓
70,000
-

CelebRef-HQ [16]
✓
10,555
1,005
FFHQ-Ref
✓
✓
✓
20,405
6,523

5
Experiments

In this section, we describe the experimental setup in Sec. 5.1, discuss ablation studies in Sec. 5.2,
and provide the comparison between our ReF-LDM and the state-of-the-art methods in Sec. 5.3

5.1
Experimental setup

5.1.1
Implementation details

To exploit more ground truth images without available reference images, we use 68,411 images
in the FFHQ dataset to train a VQGAN [4] as the frozen autoencoder and an LDM with only LQ
condition. We then finetune our ReF-LDM from the LQ-conditioned LDM with the 18,816 images in
our FFHQ-Ref dataset. All models are trained excluding the test split images to ensure fair evaluation
on our FFHQ-Ref benchmark. In our experiments, we adopt a 512x512 image resolution, fix the
number of reference images to five, and set loss scale λtime ID to 0.1. During training, we synthesize
input LQ images with σ, r, δ, and q sampled from [0, 16], [1, 32], [0, 20], and [30, 100], respectively.
For inference, we use 100 DDIM [24] steps and a classifier-free-guidance [7] with a scale of 1.5
towards reference images. We provides more implementation details in the Appendix G.

7


---Page Break---
5.1.2
Evaluation datasets and metrics

For evaluation datasets, we use the test split of our FFHQ-Ref with two different degradation
levels: severe and moderate. In addition, previous non-reference-based methods commonly use
CelebA-Test [28] for evaluation, which comprises 3,000 LQ and HQ image pairs sampled from the
CelebA-HQ dataset [11]. Therefore, we follow the same procedures described in Sec. 4 to construct
a subset of 2,533 images with available reference images, termed CelebA-Test-Ref.

For evaluation metrics, we adopt the identity similarity (IDS) [5, 32], which is the cosine similarity
calculated using the face recognition model ArcFace [3]. We also use the widely used perceptual
metrics LPIPS [30]. As face pixels are more of concern in the task of face restoration, we also
measure the face-region LPIPS (fLPIPS), which is the LPIPS calculated using only the pixels in face
regions. For assessing no-reference image quality, we adopt NIQE [18]. Furthermore, we measure
the FID [6], using 70,000 images from the FFHQ dataset as the target distribution.

5.2
Ablation studies

We provide the ablation studies of the proposed CacheKV, timestep-scaled identity loss, and the
number of input reference images. In each ablation experiment, we fine-tune the model for 50,000
steps from the same LDM pre-trained without reference images. We compare the difference settings
with the FFHQ-Ref-Severe dataset.

5.2.1
CacheKV and other mechanisms

The CacheKV is proposed for integrating the input reference images into the diffusion denoising
process. We compare it with other mechanisms illustrated in Sec. 3.1.3. According to Table 2,
channel-concatenation and cross-attention fail to leverage reference images to improve the identity
similarity (IDS). In contrast, both spatial-concatenation and our CacheKV significantly enhance IDS.
Moreover, our CacheKV is more computationally efficient than spatial-concatenation, requiring only
20% of the inference time and 39% of the GPU memory.

Table 2: Comparison between CacheKV and other mechanisms for input reference images (run with
five reference images on a single GTX 1080).

Mechanism
IDS↑
NIQE↓
LPIPS↓
Inference time↓
Memory↓

Channel-concatenation
0.23
4.49
0.46
4.17
1.77
Cross-attention
0.23
4.56
0.46
14.54
2.80
Spatial-concatenation
0.69
4.84
0.43
58.36
7.44
CacheKV
0.65
4.38
0.43
12.15
2.87

Table 3: Ablation results for the timestep-
scaled identity loss.

Loss
IDS↑
NIQE↓

LLDM
0.52
4.56
LLDM + LID
0.69
6.56
LLDM + Ltime ID
0.65
4.38

Table 4: Design choices for ID loss scaling.

Scale for ID loss
IDS↑
NIQE↓
√¯αt
0.65
4.38
1t<100
0.52
4.55
1t<500
0.61
4.44

(e) Input references

(f) Input LQ
(g) LLDM
(h) +LID
(i) +Ltime ID

Figure 4: Visual ablation results for the timestep-
scaled identity loss.

8


---Page Break---
5.2.2
Timestep-scaled identity loss

To validate the benefits of the proposed timestep-scaled identity loss, we train ReF-LDM with three
different loss settings: without identity loss (LLDM), with naive identity loss (LLDM + LID), and
with the proposed timestep-scaled identity loss (LLDM + Ltime ID). As show in Table 3 and Fig. 4,
while the naive identity loss can improve identity similarity (IDS), our timestep-scaled identity loss
can do so without sacrificing the image quality (NIQE).

As explained in Sec. 3.2, we employ √¯αt to scale down the identity loss for a larger and noisier
timestep t. In Table 4, we compare this design choice with other alternative scaling factors, 1t<100
and 1t<500, which apply the identity loss only when the sampled timestep t is smaller than 100 or
500, respectively. The results suggest that √¯αt is more effective than the alternatives.

5.2.3
Multiple input reference images

There are two to nine reference images for a target image in the test split of our FFHQ-Ref. While
we fix the number of reference images to five when training ReF-LDM, the proposed CacheKV
mechanism has the flexibility to take varying number of reference images during inference. To
validate the effectiveness of utilizing multiple reference images, we evaluate ReF-LDM with a
maximum of 1, 3, 5, and 8 reference images, respectively. As shown in Table 5, using more reference
images significantly improves the identity similarity (from 0.52 to 0.66). However, increasing the
number of reference images also increases the computation time, as shown in Table 6. Since using
eight reference images encounters an out-of-memory issue on a single GTX 1080, we use at most
five reference images in our experiments for simplicity.

Table 5: Image quality with different numbrs of
reference images.

Max num refs
IDS↑
LPIPS↓

1
0.52
0.45
3
0.62
0.44
5
0.65
0.43
8
0.66
0.43

Table 6: Inference time with different numbers
of reference images.

Num refs
Time@1080↓
Time@3090↓

1
4.86
3.00
3
7.09
3.54
5
12.03
4.51
8
out-of-memory
6.26

5.3
Comparison with state-of-the-art methods

Table 7: Comparison of ReF-LDM with state-of-the-art methods across three benchmarks. Note the
highlighting 1st, 2nd, and a gray cell indicating evaluation data leakage for prior methods .

Method
FFHQ-Ref-Severe
FFHQ-Ref-Moderate
CelebA-Test-Ref

IDS↑
fLPIPS↓LPIPS↓FID↓
IDS↑
fLPIPS↓LPIPS↓FID↓
IDS↑
fLPIPS↓LPIPS↓

CodeFormer [32]
0.323
0.108
0.398 51.51
0.760
0.084
0.301 38.78
0.660
0.092
0.340
VQFR [5]
0.308
0.112
0.415 52.96
0.659
0.089
0.324 36.77
0.558
0.096
0.352
DAEFR [26]
0.294
0.118
0.435 49.08
0.614
0.093
0.333 33.86
0.491
0.101
0.367
LDM
0.231
0.125
0.453 34.40
0.753
0.095
0.344 32.16
0.663
0.093
0.368

DMDNet [16]†
0.185
0.162
0.511 72.66
0.810
0.096
0.348 36.60
0.752
0.097
0.362
ReF-LDM
0.676
0.110
0.429 37.60
0.840
0.088
0.332 33.05
0.779
0.093
0.368

†As DMDNet encounters landmark detection failures and fails to yield results for 214/857, 29/857, and 488/2,533 images on the three
benchmarks respectively, we compute the metrics for DMDNet using the remaining images.

5.3.1
Quantitative comparison

We compare our ReF-LDM with state-of-the-art methods on FFHQ-Ref-Severe, FFHQ-Ref-Moderate,
and CelebA-Test-Ref. Table 7 reports the performance of competing methods in terms of IDS, fLPIPS,
LPIPS, and FID (targeting the FFHQ image distribution). Without the information in reference images,
the existing non-reference-based restoration methods (CodeFormer [32], VQFR [5], and DAEFR [26])
fail to preserve the facial identity, leading to significantly lower IDS. The reference-based method,

9


---Page Break---
DMDNet [16], fails to restore the severely degraded images because it depends on unreliable facial
landmark detection, reflected by higher fLPIPS. In contrast, our ReF-LDM consistently outperforms
DMDNet on identity similarity and other metrics, owing to the proposed CacheKV mechanism and
timestep-scaled identity loss, which effectively leverage the input reference images without the need
for landmark detection. We also note that our method exhibits slightly inferior results in LPIPS metric.
This is due to the difference in the background pixels, we provide further details in the Appendix D.
It is also worth mentioning that the competing methods benefit from data leakage on the FFHQ-Ref
benchmarks, as their models are trained with the entire FFHQ dataset or with a different train split
than the identity-based one in the proposed FFHQ-Ref.

5.3.2
Qualitative comparison

In Fig. 5, we present a qualitative comparison between our ReF-LDM, the pre-trained LDM with-
out reference images, CodeFomer (a SOTA non-reference-based method), and DMDNet (a SOTA
reference-based method). Given the severely degraded image, DMDNet generates distorted face
images based on incorrectly detected landmarks. While CodeFormer yields realistic face images, it
does not preserve the facial identity well. In contrast, our ReF-LDM produces results that are both
realistic and faithful to the individual’s facial identity.

Input LQ
GT
CodeFormer
DMDNet
LDM
ReF-LDM

Figure 5: Qualitative comparison. From left to right: input LQ, ground truth, other methods, and our
ReF-LDM. From top to bottom: FFHQ-Ref-Severe, FFHQ-Ref-Moderate, and CelebA-Test-Ref.

6
Limitations

When the face region is occluded by other objects, our model may generate artifacts. For certain
face poses (e.g, side face), the reconstructed eyes may appear unnatural. These problems are also
commonly observed in other methods and might be caused due to the lack of such training images.
However, there are some examples showing that these problems can be alleviated if our model is
provided reference images with similar face poses to the target image. Visual examples of these
limitations are provided in Appendix F.

7
Conclusion

In summary, we propose ReF-LDM, which incorporates the CacheKV mechanism and the timestep-
scaled identity loss, to effectively utilize multiple reference images for face restoration. Additionally,
we construct the FFHQ-Ref dataset, which surpasses the existing dataset in both quantity and quality,
to facilitate the research in reference-based face restoration. Evaluation results demonstrate that
ReF-LDM achieves superior performance in face identity similarity over state-of-the-art methods.

10


---Page Break---
Acknowledgments
The authors wish to express their gratitude to Professor Wei-Chen Chiu for his
valuable suggestion to exclude the reference images that are too similar to the target
images when constructing the proposed FFHQ-Ref dataset.

References

[1] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng. Masactrl: Tuning-free
mutual self-attention control for consistent image synthesis and editing. In
Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV), pages 22560–22570, October 2023.
[2] C. Chen, X. Li, L. Yang, X. Lin, L. Zhang, and K.-Y. K. Wong. Progressive
semantic-aware style transformation for blind face restoration. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition, pages
11896–11905, 2021.
[3] J. Deng, J. Guo, X. Niannan, and S. Zafeiriou. Arcface: Additive angular
margin loss for deep face recognition. In CVPR, 2019.
[4] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution
image synthesis. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 12873–12883, 2021.
[5] Y. Gu, X. Wang, L. Xie, C. Dong, G. Li, Y. Shan, and M.-M. Cheng. Vqfr:
Blind face restoration with vector-quantized dictionary and parallel decoder. In
European Conference on Computer Vision, pages 126–143. Springer, 2022.
[6] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium.
Advances in neural information processing systems, 30, 2017.
[7] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint
arXiv:2207.12598, 2022.
[8] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Ad-
vances in neural information processing systems, 33:6840–6851, 2020.
[9] R. Huang, S. Zhang, T. Li, and R. He. Beyond face rotation: Global and local
perception gan for photorealistic and identity preserving frontal view synthesis.
In Proceedings of the IEEE international conference on computer vision, pages
2439–2448, 2017.
[10] K. Karkkainen and J. Joo. Fairface: Face attribute dataset for balanced race,
gender, and age for bias measurement and mitigation. In Proceedings of
the IEEE/CVF winter conference on applications of computer vision, pages
1548–1558, 2021.
[11] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of gans for
improved quality, stability, and variation. arXiv preprint arXiv:1710.10196,
2017.
[12] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for
generative adversarial networks. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 4401–4410, 2019.

11


---Page Break---
[13] Y.-F. Lau, T. Zhang, Z. Rao, and Q. Chen. Ented: Enhanced neural texture
extraction and distribution for reference-based blind face restoration. In Pro-
ceedings of the IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 5162–5171, 2024.
[14] X. Li, M. Liu, Y. Ye, W. Zuo, L. Lin, and R. Yang. Learning warped guidance
for blind face restoration. In Proceedings of the European conference on
computer vision (ECCV), pages 272–289, 2018.
[15] X. Li, W. Li, D. Ren, H. Zhang, M. Wang, and W. Zuo. Enhanced blind face
restoration with multi-exemplar images and adaptive spatial feature fusion. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2706–2715, 2020.
[16] X. Li, S. Zhang, S. Zhou, L. Zhang, and W. Zuo. Learning dual memory
dictionaries for blind face restoration. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 45(5):5904–5917, 2022.
[17] X. Lin, J. He, Z. Chen, Z. Lyu, B. Fei, B. Dai, W. Ouyang, Y. Qiao, and
C. Dong. Diffbir: Towards blind image restoration with generative diffusion
prior. arXiv preprint arXiv:2308.15070, 2023.
[18] A. Mittal, R. Soundararajan, and A. C. Bovik. Making a “completely blind”
image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.
[19] Y. Nitzan, K. Aberman, Q. He, O. Liba, M. Yarom, Y. Gandelsman, I. Mosseri,
Y. Pritch, and D. Cohen-Or. Mystyle: A personalized generative prior. ACM
Transactions on Graphics (TOG), 41(6):1–10, 2022.
[20] S. Pouyanfar, S. Sengupta, M. Mohammadi, E. Abraham, B. Bloomquist,
L. Dauterman, A. Parikh, S. Lim, and E. Sommerlade. Frr-net: A real-time
blind face restoration and relighting network. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 1240–1250,
2023.
[21] E. Richardson, Y. Alaluf, O. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro, and
D. Cohen-Or.
Encoding in style: a stylegan encoder for image-to-image
translation. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 2287–2296, 2021.
[22] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer.
High-
resolution image synthesis with latent diffusion models. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition, pages
10684–10695, 2022.
[23] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi. Image
super-resolution via iterative refinement. IEEE transactions on pattern analysis
and machine intelligence, 45(4):4713–4726, 2022.
[24] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv
preprint arXiv:2010.02502, 2020.
[25] M. Suin, N. G. Nair, C. P. Lau, V. M. Patel, and R. Chellappa.
Diffuse
and restore: A region-adaptive diffusion model for identity-preserving blind
face restoration.
In Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, pages 6343–6352, 2024.

12


---Page Break---
[26] Y.-J. Tsai, Y.-L. Liu, L. Qi, K. C. Chan, and M.-H. Yang. Dual associated
encoder for face restoration. In The Twelfth International Conference on
Learning Representations, 2024.

[27] J. Wang, Z. Yue, S. Zhou, K. C. Chan, and C. C. Loy. Exploiting diffusion
prior for real-world image super-resolution. 2024.

[28] X. Wang, Y. Li, H. Zhang, and Y. Shan. Towards real-world blind face restora-
tion with generative facial prior. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 9168–9178, 2021.
[29] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-
to-image diffusion models. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 3836–3847, 2023.

[30] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable
effectiveness of deep features as a perceptual metric. In CVPR, 2018.

[31] Y. Zhao, T. Hou, Y.-C. Su, X. Jia, Y. Li, and M. Grundmann. Towards authentic
face restoration with iterative diffusion models and beyond. In Proceedings of
the IEEE/CVF International Conference on Computer Vision, pages 7312–7322,
2023.

[32] S. Zhou, K. Chan, C. Li, and C. C. Loy. Towards robust blind face restoration
with codebook lookup transformer. Advances in Neural Information Processing
Systems, 35:30599–30611, 2022.

13


---Page Break---
A
Broader Impacts

The ReF-LDM has the capability to leverage personal appearances from reference
images. This introduces a potential risk of misuse, where it could be employed for
malicious face editing by using a low-quality image in conjunction with reference
images from a different individual.

Reference
Input LQ
ReF-LDM
Reference
Input LQ
ReF-LDM

Figure 6: Examples of ReF-LDM using reference images from two different individuals.

B
Image quality issues in the previous dataset CelebRef-HQ

As described in Sec. 4.4, the previous dataset for the reference-based face restoration
task, CelebRef-HQ [16], exhibits issues with image quality. We provide examples
where the ground truth images in this dataset are corrupted by watermarks and mirror
padding in Fig. 7.

Figure 7: Example images with mirror padding and watermark artifacts in the CelebRef-HQ dataset.

C
Statistics of FFHQ-Ref dataset

We analyze the statistics of the proposed FFHQ-Ref dataset, introduced in Sec. 4.

In Fig. 8, we plot the distribution of the number of available reference images.

Furthermore, we assess the race, age, and gender distributions of the dataset using
labels predicted by FairFace [10]. As depicted in Fig. 9, the race distribution within
FFHQ-Ref is imbalanced, with a predominance of the ’white’ category. To mitigate
this, we intentionally sampled a greater number of images from other races to
construct a more balanced test set. Additionally, as illustrated in Fig. 10, FFHQ-Ref
encompasses a broad age range, from infants (0-2 years) to the elderly (70+ years).
However, the distribution is not uniform across ages and genders. For example, there
is a notably higher proportion of young females (29.2% of ’20-29 female’).

14


---Page Break---
(a) Training set
(b) Validation set
(c) Test set

Figure 8: Distribution of the number of available reference images per image in the FFHQ-Ref dataset
of train, validation, and test splits.

(a) FFHQ-Ref
(b) FFHQ-Ref-Test

Figure 9: Race distribution within FFHQ-Ref dataset.

(a) FFHQ-Ref
(b) FFHQ-Ref-Test

Figure 10: Age and gender distribution within FFHQ-Ref dataset.

D
Examples of differences in background regions

In Fig. 11, we provide some examples where our ReF-LDM are more different to the
ground truth in background pixels compared to prior methods. In the first example,
the ReF-LDM attempts to restore another face in the background. In the second

15


---Page Break---
example, our ReF-LDM restored the mirror padding in the CelebA-Test dataset as
hairs.

Input LQ
GT
CodeFormer
DMDNet
LDM
ReF-LDM

Figure 11: Examples where ReF-LDM generates background-region details that differ more from the
ground truth.

E
Examples of illumination change

Fig. 12 shows an example where ReF-LDM exhibits warmer illumination compared
to LDM. We conjecture that this may be due to the impact of the strong warm
lighting in the input reference images. To address this issue, one could employ
post-processing tricks, such as adjusting the means of the R, G, B channels to match
those of the input LQ image. Another potential solution might be training ReF-LDM
with data augmentation on the illuminations of input reference images, to encourage
the model to disregard the illuminations of input references and maintain consistency
with that of the input LQ image.

(a) Input LQ
(b) GT
(c) LDM
(d) RefLDM

(e) Input references

Figure 12: An example of (d) ReF-LDM demonstrating an illumination change, likely influenced by
the strong warm lighting of the (e) input reference images.

16


---Page Break---
F
Examples of failure cases

Here we provide visual examples for the limitation described in Sec. 6. As shown
Fig. 13, when the face region is occluded, our ReF-LDM and the prior models tend
to generate from unnatural artifacts. For side face images, the ReF-LDM may not
work well when the input reference images do not contain faces of similar pose,
as shown in Fig. 14. However, Fig. 15 suggests that our ReF-LDM can effectively
exploit the reference images of similar face poses to improve the results.

Input LQ
GT
CodeFormer
DMDNet
LDM
ReF-LDM

Figure 13: Some failure cases when the input LQ images are occluded.

Input LQ
GT
CodeFormer
DMDNet
LDM
ReF-LDM

Figure 14: Some failure cases of side faces when side-face images are absent in the references.

Input LQ
GT
CodeFormer
DMDNet
LDM
ReF-LDM

Figure 15: Some successful cases of side faces when side-face images are included in the references.

17


---Page Break---
G
More implementation details

G.1
Classifier-free guidance towards reference images

Classifier-free guidance [7] is a technique widely used in diffusion models for
guiding the generated results towards a condition c with a controllable scale factor s
at inference time:

˜ϵθ(zt, c) = ϵθ(zt, ∅) + s · (ϵθ(zt, c) −ϵθ(zt, ∅))
(5)

In our experiments, we use classifier-free guidance towards reference images with
s = 1.5.

˜ϵθ(zt, zLQ, {zref}) = ϵθ(zt, zLQ, ∅) + s · (ϵθ(zt, zLQ, {zref}) −ϵθ(zt, zLQ, ∅))
(6)

During the training phase, we randomly drop the conditions by setting them to zero
tensors with a probability of 0.1.

G.2
Data augmentation for input reference images

During the training phase, we use a fixed number of five input reference images.
When a target images with less than five reference images are sampled, we repeat
the reference images to obtain five reference images. In addition, we apply image
augmentation to the input reference images with the following operations: color
jitter (brightness ± 0.2, contrast ± 0.2, saturation ± 0.2, hue ± 0.02), affine transform
(rotation ± 2, translation ± 0.05, scale ± 0.05), perspective transform (scale ± 0.2,
probability 0.5), and horizontal flip (probability 0.5). Lastly, we randomly shuffle the
order of available reference images for a target image, so that a different combination
of reference images can be sampled at each training iteration. In Fig. 16, we provide
an example where a set of two reference images is augmented to a set of five
reference images.

Figure 16: Data augmentation for reference images.

G.3
Training details

We trained the VQGAN for 200,000 iterations with batch size 32 on four A6000
GPUs for 7 days. We trained the LDM with only LQ condition for 500,000 iterations
with batch size 40 on four A6000 GPUs for 7 days. We finetuned the ReF-LDM for
150,000 iterations with batch size 8 on four 3090 GPUs for 6 days. For training losses,
the LDM is trained using only the typical LDM loss LLDM, while the ReF-LDM is
trained with both LLDM and the proposed Ltime ID.

18


---Page Break---
G.4
Hyperparameters of networks

For the frozen autoencoder, we use a VQGAN as in the LDM [22] with the following
settings:

• input image: 512x512x3
• latent representation: 64x64x8

• code booksize: 8192

• network hyperparameters: base channel as 128, multiplier for each scale as
[1, 1, 2, 4] with 2 residual blocks.

For the denoising U-net, we use the following settings:

• input latent: 64x64x16

• output latent: 64x64x8
• attention layer at resolutions: 32x32, 16x16, and 8x8

• network hyperparameters: base channel as 160, multiplier for each scale as
[1, 2, 2, 4] with 2 residual blocks.

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accu-
rately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The method is described in Sec. 3 and the ablation studies and
evaluation results are provided in Sec. 5.

Guidelines:

• The answer NA means that the abstract and introduction do not include
the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made,
including the contributions made in the paper and important assumptions
and limitations. A No or NA answer to this question will not be perceived
well by the reviewers.
• The claims made should match theoretical and experimental results,
and reflect how much the results can be expected to generalize to other
settings.
• It is fine to include aspirational goals as motivation as long as it is clear
that these goals are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by
the authors?

Answer: [Yes]

Justification: We discuss the limitation in Sec. 6 and provide visual examples
in Appendix F.

Guidelines:

• The answer NA means that the paper has no limitation while the answer
No means that the paper has limitations, but those are not discussed in
the paper.
• The authors are encouraged to create a separate "Limitations" section in
their paper.
• The paper should point out any strong assumptions and how robust
the results are to violations of these assumptions (e.g., independence
assumptions, noiseless settings, model well-specification, asymptotic
approximations only holding locally). The authors should reflect on how
these assumptions might be violated in practice and what the implications
would be.
• The authors should reflect on the scope of the claims made, e.g., if the
approach was only tested on a few datasets or with a few runs. In general,
empirical results often depend on implicit assumptions, which should be
articulated.

20


---Page Break---
• The authors should reflect on the factors that influence the performance of
the approach. For example, a facial recognition algorithm may perform
poorly when image resolution is low or images are taken in low lighting.
Or a speech-to-text system might not be used reliably to provide closed
captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed
algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their
approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations
might be used by reviewers as grounds for rejection, a worse outcome
might be that reviewers discover limitations that aren’t acknowledged
in the paper. The authors should use their best judgment and recognize
that individual actions in favor of transparency play an important role in
developing norms that preserve the integrity of the community. Review-
ers will be specifically instructed to not penalize honesty concerning
limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of
assumptions and a complete (and correct) proof?
Answer: [NA]
Justification: This work does not involve theoretical results.
Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered
and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement
of any theorems.
• The proofs can either appear in the main paper or the supplemental
material, but if they appear in the supplemental material, the authors are
encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be
complemented by formal proofs provided in appendix or supplemental
material.
• Theorems and Lemmas that the proof relies upon should be properly
referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to repro-
duce the main experimental results of the paper to the extent that it affects
the main claims and/or conclusions of the paper (regardless of whether the
code and data are provided or not)?
Answer: [Yes]
Justification: We provide implementation details in Sec. ?? and Appendix G.

21


---Page Break---
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not
be perceived well by the reviewers: Making the paper reproducible is
important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe
the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in
various ways. For example, if the contribution is a novel architecture,
describing the architecture fully might suffice, or if the contribution is a
specific model and empirical evaluation, it may be necessary to either
make it possible for others to replicate the model with the same dataset,
or provide access to the model. In general. releasing code and data
is often one good way to accomplish this, but reproducibility can also
be provided via detailed instructions for how to replicate the results,
access to a hosted model (e.g., in the case of a large language model),
releasing of a model checkpoint, or other means that are appropriate to
the research performed.
• While NeurIPS does not require releasing code, the conference does
require all submissions to provide some reasonable avenue for repro-
ducibility, which may depend on the nature of the contribution. For
example
(a) If the contribution is primarily a new algorithm, the paper should
make it clear how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper
should describe the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then
there should either be a way to access this model for reproducing the
results or a way to reproduce the model (e.g., with an open-source
dataset or instructions for how to construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in
which case authors are welcome to describe the particular way they
provide for reproducibility. In the case of closed-source models, it
may be that access to the model is limited in some way (e.g., to
registered users), but it should be possible for other researchers to
have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with
sufficient instructions to faithfully reproduce the main experimental results,
as described in supplemental material?

Answer: [No]

Justification: We plan to release our dataset and model after the paper’s
acceptance. They are not included in the current submission because we are
waiting formal permission from the associated company.

22


---Page Break---
Guidelines:

• The answer NA means that paper does not include experiments requiring
code.
• Please see the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more de-
tails.
• While we encourage the release of code and data, we understand that this
might not be possible, so “No” is an acceptable answer. Papers cannot
be rejected simply for not including code, unless this is central to the
contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environ-
ment needed to run to reproduce the results. See the NeurIPS code
and data submission guidelines (https://nips.cc/public/guides/
CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation,
including how to access the raw data, preprocessed data, intermediate
data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results
for the new proposed method and baselines. If only a subset of experi-
ments are reproducible, they should state which ones are omitted from
the script and why.
• At submission time, to preserve anonymity, the authors should release
anonymized versions (if applicable).
• Providing as much information as possible in supplemental material
(appended to the paper) is recommended, but including URLs to data
and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data
splits, hyperparameters, how they were chosen, type of optimizer, etc.)
necessary to understand the results?
Answer: [Yes]
Justification: We provide details in Sec. ?? and Appendix G.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper
to a level of detail that is necessary to appreciate the results and make
sense of them.
• The full details can be provided either with the code, in appendix, or as
supplemental material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined
or other appropriate information about the statistical significance of the
experiments?

23


---Page Break---
Answer: [No]

Justification: We do not report error bars, as the training our LDM-based
model for multiple times costs too much time and computational resources.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error
bars, confidence intervals, or statistical significance tests, at least for the
experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be
clearly stated (for example, train/test split, initialization, random drawing
of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed
form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed er-
rors).
• It should be clear whether the error bar is the standard deviation or the
standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors
should preferably report a 2-sigma error bar than state that they have a
96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show
in tables or figures symmetric error bars that would yield results that are
out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain
in the text how they were calculated and reference the corresponding
figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information
on the computer resources (type of compute workers, memory, time of
execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the computer resources for inference in Table. 2
and for training in Appendix G.3.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, in-
ternal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of
the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more
compute than the experiments reported in the paper (e.g., preliminary or
failed experiments that didn’t make it into the paper).

24


---Page Break---
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every
respect, with the NeurIPS Code of Ethics https://neurips.cc/public/
EthicsGuidelines?

Answer: [Yes]

Justification: Our FFHQ-Ref dataset in this work is based on an exisiting
dataset FFHQ where only images with appropriate licences are collected.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS
Code of Ethics.
• If the authors answer No, they should explain the special circumstances
that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a
special consideration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and
negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the broader impacts in Appedix A.

Guidelines:

• The answer NA means that there is no societal impact of the work
performed.
• If the authors answer NA or No, they should explain why their work has
no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or
unintended uses (e.g., disinformation, generating fake profiles, surveil-
lance), fairness considerations (e.g., deployment of technologies that
could make decisions that unfairly impact specific groups), privacy con-
siderations, and security considerations.
• The conference expects that many papers will be foundational research
and not tied to particular applications, let alone deployments. However,
if there is a direct path to any negative applications, the authors should
point it out. For example, it is legitimate to point out that an improvement
in the quality of generative models could be used to generate deepfakes
for disinformation. On the other hand, it is not needed to point out that a
generic algorithm for optimizing neural networks could enable people to
train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the
technology is being used as intended and functioning correctly, harms
that could arise when the technology is being used as intended but gives
incorrect results, and harms following from (intentional or unintentional)
misuse of the technology.

25


---Page Break---
• If there are negative societal impacts, the authors could also discuss
possible mitigation strategies (e.g., gated release of models, providing
defenses in addition to attacks, mechanisms for monitoring misuse,
mechanisms to monitor how a system learns from feedback over time,
improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for
responsible release of data or models that have a high risk for misuse (e.g.,
pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We haven’t release the dataset or models.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be
released with necessary safeguards to allow for controlled use of the
model, for example by requiring that users adhere to usage guidelines or
restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks.
The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and
many papers do not require this, but we encourage authors to take this
into account and make a best faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data,
models), used in the paper, properly credited and are the license and terms of
use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly states that our FFHQ-Ref is constructed based on
the FFHQ dataset in Sec 4.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package
or dataset.
• The authors should state which version of the asset is used and, if possi-
ble, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each
asset.
• For scraped data from a particular source (e.g., website), the copyright
and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and
terms of use in the package should be provided.
For popular
datasets, paperswithcode.com/datasets has curated licenses for

26


---Page Break---
some datasets. Their licensing guide can help determine the license
of a dataset.
• For existing datasets that are re-packaged, both the original license and
the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to
reach out to the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is
the documentation provided alongside the assets?
Answer: [NA]
Justification: We haven’t release the dataset.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as
part of their submissions via structured templates. This includes details
about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from
people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable).
You can either create an anonymized URL or include an anonymized zip
file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects,
does the paper include the full text of instructions given to participants and
screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA]
Justification: This work does not contain crowdsourcing experiments.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing
nor research with human subjects.
• Including this information in the supplemental material is fine, but if the
main contribution of the paper involves human subjects, then as much
detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data
collection, curation, or other labor should be paid at least the minimum
wage in the country of the data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research
with Human Subjects
Question: Does the paper describe potential risks incurred by study par-
ticipants, whether such risks were disclosed to the subjects, and whether
Institutional Review Board (IRB) approvals (or an equivalent approval/review
based on the requirements of your country or institution) were obtained?

27


---Page Break---
Answer: [NA]
Justification: This work does not contain crowdsourcing experiments.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing
nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval
(or equivalent) may be required for any human subjects research. If you
obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between
institutions and locations, and we expect authors to adhere to the NeurIPS
Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break
anonymity (if applicable), such as the institution conducting the review.

28


---Page Break---
