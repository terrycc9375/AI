SeeClear: Semantic Distillation Enhances Pixel
Condensation for Video Super-Resolution

Qi Tang1,2,
Yao Zhao1,2,
Meiqin Liu1,2∗,
Chao Yao3∗

1 Institute of Information Science, Beijing Jiaotong University
2 Visual Intelligence + X International Cooperation Joint Laboratory of MOE,
Beijing Jiaotong University
3 School of Computer and Communication Engineering,
University of Science and Technology Beijing
{qitang, yzhao, mqliu}@bjtu.edu.cn, yaochao@ustb.edu.cn

Abstract

Diffusion-based Video Super-Resolution (VSR) is renowned for generating percep-
tually realistic videos, yet it grapples with maintaining detail consistency across
frames due to stochastic fluctuations. The traditional approach of pixel-level align-
ment is ineffective for diffusion-processed frames because of iterative disruptions.
To overcome this, we introduce SeeClear–a novel VSR framework leveraging
conditional video generation, orchestrated by instance-centric and channel-wise
semantic controls. This framework integrates a Semantic Distiller and a Pixel Con-
denser, which synergize to extract and upscale semantic details from low-resolution
frames. The Instance-Centric Alignment Module (InCAM) utilizes video-clip-wise
tokens to dynamically relate pixels within and across frames, enhancing coherency.
Additionally, the Channel-wise Texture Aggregation Memory (CaTeGory) infuses
extrinsic knowledge, capitalizing on long-standing semantic textures. Our method
also innovates the blurring diffusion process with the ResShift mechanism, finely
balancing between sharpness and diffusion effects. Comprehensive experiments
confirm our framework’s advantage over state-of-the-art diffusion-based VSR tech-
niques. The code is available: https://github.com/Tang1705/SeeClear-NeurIPS24.

1
Introduction

Video super-resolution (VSR) is a challenging low-level vision task that involves improving the
resolution and visual quality of the given low-resolution (LR) observations and maintaining the
temporal coherence of high-resolution (HR) components. Various deep-learning-based VSR ap-
proaches [1, 3, 30, 17, 13, 20, 21] explore effective inter-frame alignment to reconstruct satisfactory
sequences. Despite establishing one new pole after another in the quantitative results, they struggle to
generate photo-realistic textures.

With the explosion of diffusion model (DM) in visual generation [10, 31, 26], super-resolution (SR)
from the generative perspective also garners the broad attention [28, 29, 45, 5]. DM breaks the
generation process into sequential sub-processes and iteratively samples semantic-specific images
from Gaussian noise, equipped with a paired forward diffusion process and reverse denoising process.
The former progressively injects varied intensity noise into the image along a Markov chain to
simulate diverse image distributions. The latter leverages a denoising network to generate an image
based on the given noise and conditions. Early efforts directly apply the generation paradigm to
super-resolution, overlooking its characteristic while generating pleasing content, thus trapping in
huge sampling overhead.

∗Corresponding Authors

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Different from generation from scratch, SR resembles partial generation. The structural information
that dominates the early stages of diffusion is contained in the LR priors, while SR tends to focus on
generating high-frequency details [33, 15]. Besides, the loss of high-frequency information in LR
videos stems from the limited sensing range of imaging equipment. As a result, solely disrupting
frames with additive noise is inadequate to depict the degradation of HR videos [9]. Moreover,
prevalent VSR methods employ delicate inter-frame alignment (e.g., optical flow or deformable
convolution) to fuse the sub-pixel information across adjacent frames. However, the disturbed pixels
pose a severe challenge to these methods, rendering the accuracy to deteriorate in the pixel space.

𝐼!"#

$
/ 𝐼!"#

%&

𝐼!"'

$
/ 𝐼!"'

%&

𝐼!

$ / 𝐼!

%&

𝐼!('

$
 / 𝐼!('

%&

𝐼!(#

$
 / 𝐼!(#

%&

𝐼!"#

)&

𝐼!"'

)&

𝐼!

)&

𝐼!('

)&

𝐼!(#

)&

Semantic 

Distiller

Pixel Condenser

𝓢

𝑂*, 𝐶+, 𝑇+

𝐼{!"#,⋯,!(#}

)&

𝓟

Frozen
Learnable

𝑃!

𝑂!
𝐶"
𝑇"

Instance-Centric 
Alignment Module

Channel-wise Texture 
Aggregation Memory

𝑂#

𝐹&!,% %&'

(

Associate

Assemble

Figure 1: The sketch of SeeClear. It consists of a Seman-
tic Distiller and a Pixel Condenser, which are responsible
for distilling instance-centric semantics from LR frames
and generating HR frames. The instance-centric and
assembled channel-wise semantics act as thermometer
to control the condition for generation.

To alleviate the above issue, we introduce
SeeClear, an innovative diffusion model em-
powering distilled semantics to enhance the
pixel condensation for video super-resolution.
During the forward diffusion process, the low-
pass filter is applied within the patch, gradually
diminishing the high-frequency component, all
while the residual is progressively shifted in the
frequency domain to transform the HR frames to
corresponding LR versions step by step. To re-
duce the computational overhead, the intermedi-
ate states are decomposed into various frequency
sub-bands via 2D discrete wavelet transform and
subsequently processed by the attention-based
U-Net. Furthermore, we devise a dual semantic-
controlled conditional generation schema to en-
hance the temporal coherence of VSR. Specifi-
cally, a segmentation framework for open vocab-
ulary is employed to distill instance-centric se-
mantics from LR frames. They serve as prompts,
enabling the Instance-Centric Alignment Module (InCAM) to highlight and associate semantically
related pixels within the local temporal scope. Besides, abundant semantic cues in channel dimensions
are also explored to form an extensional memory dubbed Channel-wise Texture Aggregation Memory
(CaTeGory). It aids in global temporal coherence and boosts performance. Experimental results
demonstrate that our method consistently outperforms existing state-of-the-art methods.

In summary, the main contributions of this work are as follows:

• We present SeeClear, a diffusion-based framework for video super-resolution that distills semantic
priors from low-resolution frames for spatial modulation and temporal association, controlling the
condition of pixel generation.

• We reformulate the diffusion process by integrating residual shifting with patch-level blurring, and
introduce an attention-based architecture to explore valuable information among wavelet spectra
during the sampling process, incorporating feature modulation of intra-frame semantics.

• We devise a dual semantic distillation schema that extracts instance-centric semantics of each
frame and further assembles them into texture memory based on the semantic category of channel
dimension, ensuring both short-term and long-term temporal coherence.

2
Related Work

2.1
Video Super-Resolution

Prominent video super-resolution techniques concentrate on leveraging sub-pixel information across
frames to enhance performance. EDVR [37] employs cascading deformable convolution layers
(DCN) for inter-frame alignment in a coarse-to-fine manner, tackling large amplitude video motion.
BasicVSR [1] comprehensively explores each module’s role in VSR and delivers a simple yet effective
framework by reusing previous designs with slight modifications. Given the similarity between DCN
and optical flow, BasicVSR++ [3] devises flow-guided deformable alignment, exploiting the offset
diversity of DCN without instability during the training. VRT [16] combines mutual attention with
self-attention, which is respectively in charge of inter-frame alignment and information preservation.

2


---Page Break---
RVRT [17] extends this by incorporating optical flow with deformable attention, aligning and fusing
features directly at non-integer locations clip-to-clip. PSRT [30] reassesses prevalent alignment meth-
ods in transformer-based VSR and implements patch alignment to counteract inaccuracies in motion
estimation and compensation. DFVSR [8] represents video with the proposed directional frequency
representation, amalgamating object motion into multiple directional frequencies, augmented with a
frequency-based implicit alignment, thus enhancing alignment.

2.2
Diffusion-Based Super-Resolution

Building on the success of diffusion models in the realm of image generation [10, 31, 26, 25, 48],
diffusion-based super-resolution (SR) is advancing. SR3 [28], a pioneering approach, iteratively
samples an HR image from Gaussian noise conditioned on the LR image. In contrast, StableSR [36]
applies diffusion-based SR in a low-dimensional latent space using the pre-trained auto-encoder to
reduce computation and generate improved results through the generative priors contained in weights
of Latent Diffusion. ResDiff [29] combines a lightweight CNN with DM to restore low-frequency
and predict high-frequency components, and ResShift [45] redefines the initial step as a blend of the
low-resolution image and random noise to boost efficiency. Applying a different approach, DiWa [23]
migrates the diffusion process into the wavelet spectrum to effectively hallucinate high-frequency
information. Upscale-A-Video [49], for video super-resolution, introduces temporal layers into the
U-Net and VAE-Decoder and deploys a flow-guided recurrent latent propagation module to ensure
temporal coherence and overall video stability when applying image-wise diffusion model.

2.3
Semantic-Assisted Restoration

Traditionally seen as a preparatory step for subsequent tasks [50, 12], restoration is now reformulated
with the assistance of semantics. SFT [38] utilizes semantic segmentation probability maps for
spatial modulation of intermediate features in the SR network, yielding more realistic textures.
SKF [40] supports low-light image enhancement model to learn diverse priors encapsulated in a
semantic segmentation model by semantic-aware embedding module paired with semantic-guided
losses. SeD [14] integrates semantics into the discriminator of GAN-based SR for fine-grained
texture generation rather than solely learning coarse-grained distribution. CoSeR [32] bridges image
appearance and language understanding to empower SR with global cognition buried in LR image,
regarding priors of text-to-image (T2I) diffusion model and a high-resolution reference image as
powerful conditions. SeeSR [39] analyzes several types of semantic prompts and opts tag-style
semantics to harness the generative potential of the T2I model for real SR. Semantic Lens [34]
forgoes pixel-level inter-frame alignment and distills diverse semantics for temporal association in
the instance-centric semantic space, attaining better performance.

3
Methodology

Given a low-resolution (LR) video sequence of N frames ILR
i
∈RN×C×H×W , where i is the frame
index, H × W represents spatial dimensions, and C stands for the channel of frame, SeeClear aims to
exploit rich semantic priors to generate the high-resolution (HR) video IHR
i
∈RN×C×sH×sW , with
s as the upscaling factor. In the iterative paradigm of the diffusion model, HR frames are corrupted
according to handcrafted transition distribution at each diffusion step (t = 1, 2, · · · , T). And a
U-shaped network is employed to estimate the posterior distribution using LR frames as condition
during reverse generation. As illustrated in Figure 1, it consists of a Semantic Distiller and a Pixel
Condenser, respectively responsible for semantic extraction and texture generation.

The LR video is initially split into non-overlapping clips composed of m frames for parallel processing.
Semantic Distiller, a pre-trained network for open-vocabulary segmentation, distills semantics related
to both instances and background clip by clip, denoted as instance-centric semantics. Pixel Condenser
is an attention-based encoder-decoder architecture, in which the encoder extracts multi-scale features
under the control of LR frames, and the decoder generates HR frames from coarse to fine. They are
also bridged via skip connections to transmit high-frequency information at the same resolution. To
maximize the network’s generative capacity, instance-centric semantics are utilized as conditions for
individual frame generation in the decoder. They also serve as the cues of inter-frame alignment for
temporal coherence within the video clip and further cluster into a semantic-texture memory along
channel dimension for consistency across clips.

3


---Page Break---
3.1
Blurring ResShift

During the video capturing, frequencies exceeding the imaging range of the device are truncated,
leading to the loss of high-frequency information in LR videos. Therefore, an intuition is to construct
a Markov chain between HR frames and LR frames in the frequency domain. Inspired by blurring
diffusion [11], the forward diffusion process of SeeClear initializes with the approximate distribution
of HR frames. It then iterates and terminates with the approximate distribution of LR frames using a
Gaussian kernel convolution in frequency space facilitated by the Discrete Cosine Transformation
(DCT). Considering the correlation of neighboring information, blurring is conducted within a local
patch instead of the whole image. The above process is formulated as:

q (ut | u0) = N (ut | Dtu0, ηtE) ,
t ∈{1, · · · , T} ,
(1)

u0 = V TIHR
i
,
(2)

where u0 and ut denote HR frames and intermediate states in the frequency space for brevity.
V T denotes the projection matrix of DCT. Dt = eΛt is diagonal blurring matrix with Λx×p+y =
−π2( x2

p2 + y2

p2 ) for coordinate (x, y) within patch of size p × p, and ηt is the variance of noise. E is
the identity matrix.

In the realm of generation, the vanilla destruction process progressively transforms the image into
pure Gaussian noise, leading to numerous sampling steps and tending to be suboptimal for VSR.
An alternative way is to employ a transition kernel that shifts residuals between HR and LR frames,
accompanied by patch-level blurring. The forward diffusion process is formulated as:

q (ut | u0, ul) = N
 
ut | Dtu0 + ηtet, κ2ηtE

,
t ∈{1, · · · , T} ,
(3)

et = ul −Dtu0,
(4)

where ul denotes LR frames transformed into the frequency space. et indicates the residuals between
LR and blurred HR frames at time step t. ηt represents the shifting sequence and κ is a hyper-
parameter determining the intensity of noise. Upon this, SeeClear can yield HR frames by estimating
the posterior distribution p(u0|ul) in the reverse sampling progress, formulated as:

p (u0 | ul) =
Z
p (uT | ul)

T
Y

t=1
pθ (ut−1 | ut, ul) du1:T ,
(5)

p (uT | ul) ≈N
 
uT | ul, κ2E

,
(6)

where pθ (ut−1 | ut, ul) represents the inverse transition kernel restoring ut to ut−1. θ denotes
learnable parameters of attention-based U-Net.

To alleviate the computational overhead, preceding methods introduce an autoencoder to transform
pixel-level images in the perceptually equivalent space, concentrating on the semantic composition and
bypassing the impedance of high-frequency details. However, the loss of high-frequency information
during encoding is hard to recover in the decoding and will deteriorate the visual quality. Therefore,
we forgo the autoencoding method in SeeClear and incorporate discrete wavelet transform (DWT)
in the diffusion process. Specifically, the HR and LR frames are recursively decomposed into four
sub-bands:

IHR
ll
, IHR
lh , IHR
hl , IHR
hh = DWT2D
 
IHR
i

,
(7)

where IHR
ll
denotes the low-frequency approximation, IHR
lh , IHR
hl
and IHR
hh correspond to horizontal,
vertical and diagonal high-frequency details. DWT2D (·) represents the 2D Discrete Wavelet Trans-
form (DWT). After k decompositions, each of them possesses a size of H

2k × W

2k . These coefficients
are contacted along the channel dimension and serve in the diffusion process. The rationale behind
employing DWT as a substitute is two-fold. Firstly, it enables the U-Net to perform on a small
spatial size without information loss. Secondly, it benefits from the U-Net scaling [44] hindered
by additional parameters of the autoencoding network. To make full use of DWT, window-based
self-attention followed by channel-wise self-attention is stacked as the basic unit of U-Net, in charge
of the correlation of intra-sub-bands and inter-sub-bands, respectively.

4


---Page Break---
𝓖

𝑂!

𝛾

𝛽

..

.

𝐄𝐧𝐜
𝐃𝐞𝐜

Multi-Head Cross-Attention

𝑂"

.

Multi-Frame Self-Attention

.

𝑓!

𝐹!

𝐹,!

𝑃!

𝑚−1

𝐹1!

Figure 2: The illustration of Instance-Centric Alignment Module (InCAM). It utilizes the segmenta-
tion features to bridge the pixel-level information and instance-centric semantic tokens. And then,
the semantic-aware features can be aligned in the semantic space based on their semantic relevance.

3.2
Instance-Centric Alignment within Video Clips

Due to the destruction of the diffusion process, pixel-level inter-frame alignment, such as optical
flow, is no longer applicable. With the premise of semantic embedding, we devise the Instance-
Centric Alignment Module (InCAM) within video clips, as illustrated in Figure 2. It establishes
temporal association in the semantic space instead of intensity similarity among frames, avoiding the
interference of noise and blurriness. Specifically, Semantic Distiller predicts a set of image features
Fimg and text embedding Ftxt from LR frames and predefined vocabulary V. After that, the k image
features with the highest similarity to the text embedding are retained, including token-level semantic
priors and pixel-level segmentation features from LR frames. The above procedure is formulated as:

Fimg, Ftxt = S
 
ILR
i
, V

,
(8)

Oi, Pi = topk (Sim (Fimg, Ftxt)) ,
(9)

where S (·) represents Semantic Distiller. topk (·) and Sim (·) denote the operations of selecting
the k largest items and calculating the similarity respectively. Oi and Pi are semantic tokens and
segmentation features, in which the former represents high-level semantics and can locate related
pixels in the segmentation features. The segmentation features contain both semantics and low-
level structural information, which is suitable for bridging the semantics and features of the Pixel
Condenser. It is utilized to generate spatial modulation pairs prepared for semantic embedding,
formulated as:

(γ, β) = G (Pi) ,
(10)

Fi = (fi ⊙γ + β) + fi,
(11)

where γ and β represent scale and bias for modulation. fi and Fi correspond to original and modulated
features. G denotes two convolutional layers followed by a ReLU activation. “⊙” represents the
Hadamard product. After that, InCAM embeds semantics into modulated features based on multi-head
cross-attention, yielding semantic-embedded features ˆFi:

Qi = FiW Q, Ki = OiW K, Vi = OiW V ,
(12)

ˆFi = SoftMax

QiKT
i /
√

d

Vi,
(13)

where Qi, Ki and Vi denote matrices derived from modulated features and semantic tokens.
W Q, W K and W V represent the linear projections, and d is the dimension of projected matri-
ces. SoftMax (·) denotes the SoftMax operation. To benefit from the adjacent supporting frames,
it is necessary to establish semantic associations among frames. The frame-wise semantic tokens
are further fed into the instance encoder-decoder for communicating semantics along the temporal

5


---Page Break---
dimension, generating clip-wise semantic tokens. It gathers all the information of the same semantic
object within a clip and serves as the guide for inter-frame alignment. Specifically, InCAM combines
semantic guidance and enhanced features to activate the related pixels across frames and utilizes
multi-frame self-attention for parallel alignment and fusion. The above procedure is formulated as:

Oc = Dec

Enc (Oi) , ˆO

,
(14)

¯Fi = MFSA

Oc · ˆFi

,
(15)

where Oc and ˆO respectively denote clip-wise semantic and randomly initialized tokens. Enc (·) and
Dec (·) represent instance encoder and decoder. MFSA (·) denotes the multi-frame self-attention [30],
the extended version of self-attention in video. ¯Fi is aligned feature. The product of semantics and
enhanced features is akin to the class activation mapping, which highlights the most similar pixels in
the instance-centric semantic space among frames.

3.3
Channel-wise Aggregation across Video Clips

𝑂!
𝑂!"

Channel-wise

Semantic Assembling

𝐶#

𝑇#

Semantic Similarity Map 𝐴#

𝐹&$,& &'(

)

…

…

…

…

…

…

…

…

…

…

Multi-Head Cross Attention

Reform

.

𝐹'$

Figure 3: The illustration of Channel-wise
Texture Aggregation Memory (CaTeGory). It
assembles the textures based on the semantic
class along the channel dimension.

Due to the limited size of the temporal window, the per-
formance of clip-wise mutual information enhancement
in long video sequences is unsatisfactory, which could
lead to inconsistent content. To stabilize video content
and enhance visual texture, the Channel-wise Texture
Aggregation Memory (CaTeGory) is constructed to cluster
abundant textures according to channel-wise semantics,
as graphically depicted in Figure 3. It is comprised of
the channel-wise semantic and the corresponding texture.
Specifically, channels of instance-centric semantics also
contain distinguishing traits, which are assembled and
divided into different groups to form the channel-wise
semantic. Concurrently, hierarchical features from the
decoder of the Pixel Condenser are clustered into the cor-
responding semantic group to portray the textures. The
connection between them is established in a manner sim-
ilar to the position embedding in the attention mechanism.
The above process can be formulated as:

(Cj, Tj) = M

¯Cj, ¯Tj,
 ¯Fi,k
	4
k=1


,
(16)

where ¯Fi,k is the features benefited from adjacent frames of k-th layer. ¯Cj and ¯Tj respectively denote
channel-wise semantics and textures of j-th group, which are zero-initialized as network parameters.
They are iteratively updated towards the final version (i.e., Cj and Tj) by injecting external knowledge
from the whole dataset and previous clips. And M (·) represents the construction of CaTeGory. It
concatenates the multi-scale features and incorporates them into channel-wise semantics and textures:
ˆTj = ¯Cj × ¯Tj,
(17)

Tj = SA

CA

ˆTj,
 ¯Fi,k
	4
k=1


,
(18)

where ˆTj is the textures embedded channel-wise semantics. SA (·) and CA (·) indicate multi-head
self-attention and cross-attention. The layer normalization and feed-forward network are omitted for
brevity. It bridges the channel-wise semantics and textures via matrix multiplication and further fuses
high-value information from the pyramid feature, delivering augmented semantic-texture pairs. The
hierarchical features not only provide rich structural information but also carry relatively abstract
amid features in order to benefit different decoder layers more effectively. At each layer, the prior
knowledge stored in CaTeGory is firstly queried by the clip-wise semantics along channel dimension
and aggregated for feature enhancement of the current clip, which is formulated as:

Aj = SoftMax
 
OT
c Cj

,
(19)

˜Fi = CA
  ¯Fi, AjTj

+ ¯Fi,
(20)

6


---Page Break---
where Aj depicts the similarity between the clip-wise semantics and items of CaTeGory along
channel dimensions, and ˜Fi is the refined features as input of the next layer. As mentioned before,
semantic-texture pairs are optimized as parts of the network during the training stage, absorbing ample
priors from the whole dataset. In the sampling process, the update mechanism is reused to integrate
the long-term information of video into memory to improve the super-resolution of subsequent clips.

4
Experiments

4.1
Experimental Setup

Datasets To assess the effectiveness of the proposed SeeClear, we employ two commonly used
datasets for training: REDS [24] and Vimeo-90K [41]. The REDS dataset, characterized by its
realistic and dynamic scenes, consists of three subsets used for training and testing. In accordance
with the conventions established in previous works [1, 3], we select four clips2 from the training
dataset to serve as a validation dataset, referred to as REDS4. The Vid4 [19] dataset is used as
the corresponding test dataset for Vimeo-90K. The LR sequences are degraded through bicubic
downsampling (BI), with a downsampling factor of 4×.

Implementation Details The pre-trained OpenSeeD [46] is opted as the semantic distiller with frozen
weights, while all learnable parameters are contained in the pixel condenser. And the schedulers of
blur and noise in the diffusion process follow the settings of IHDM [25] and ResShift [45]. During
the training, the pixel condenser is first trained to generate an HR clip with 5 frames under the control
of instance-centric semantics. And then, Channel-wise Texture Memory is independently trained
to inject valuable textures from the whole dataset and be capable of fusing long-term information.
Finally, the whole network is jointly fine-tuned. All training stages utilize the Adam optimizer with
β1 = 0.5 and β2 = 0.999, where the learning rate decays with the cosine annealing scheme. The
Charbonnier loss [4] is applied on the whole frames between the ground truth and the reconstructed
frame, formulated as L =
p

||IHR
i
−ISR
i
||2 + ϵ2. The SeeClear framework is implemented with
PyTorch-2.0 and trained across 4 NVIDIA 4090 GPUs, each accommodating 4 video clips.

Evaluation Metrics Comparative analysis is conducted among different VSR methods, with the
evaluation being anchored on both pixel-based and perception-oriented metrics. Peak Signal-to-
Noise Ratio (PSNR) and Structural Similarity Index (SSIM) are utilized to evaluate the quantitative
performance as pixel-based metrics. All of them are calculated based on the Y-channel, with the
exception of the REDS4, for which the RGB-channel is used. On the perceptual side, Learned
Perceptual Image Patch Similarity (LPIPS) [47] is elected for assessment from the perspective of
human visual preference. It leverages a VGG model to extract features from the generated HR video
and the ground truth, subsequently measuring the extent of similarity between these features.

4.2
Comparisons with State-of-the-Art Methods

We compare SeeClear with several state-of-the-art methods, including regression-based and diffusion-
based ones. As shown in Table 3, SeeClear achieves superior perceptual quality compared to
regression-based methods despite slightly underperforming in pixel-based metrics. We also provide
an extended version, which leverages the generative capability of SeeClear to enhance the features
of the regression-based model, akin to references such as [6, 2]. An observable increase in fidelity
is accompanied by a notable further improvement in the perceptual metrics of the reconstructed
results. Similar performance trends can be noted on Vid4 as those on REDS4. In particular, SeeClear
achieves an LPIPS score of 0.1548, marking a relative improvement of 10.8% compared to the top
competitor, SATeCo [6]. When pitted against a variant of SATeCo, which is not modulated by LR
videos, SeeClear demonstrates a higher PSNR value with a comparable LPIPS score. It suggests that
SeeClear benefits from the control of dual semantics, striking a balance between superior fidelity and
the generation of realistic textures.

As visualized in Figure 4, SeeClear showcases its ability to restore textures with high fidelity more
effectively compared with other methods. Despite large blurriness, SeeClear still demonstrates
robust restoration capabilities for video super-resolution, reinforcing the efficacy of utilizing instance-
specific and channel-wise semantic priors for video generation control. To further substantiate

2Clip 000, 011, 015, 020

7


---Page Break---
Table 1: Performance comparisons in terms of pixel-based (PSNR and SSIM) and perception-oriented
(LPIPS) evaluation metrics on the REDS4 [24] and Vid4 [19] datasets. The extended version of
SeeClear is marked with ⋆. Red indicates the best, and blue indicates the runner-up performance
(best view in color) in each group of experiments.

Methods
Frames
REDS4 [24]
Vid4 [19]
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Bicubic
-
26.14
0.7292
0.3519
23.78
0.6347
0.3947
TOFlow [41]
7
29.98
0.7990
0.3104
25.89
0.7651
0.3386
EDVR-M [37]
5
30.53
0.8699
0.2312
27.10
0.8186
0.2898
BasicVSR [1]
15
31.42
0.8909
0.2023
27.24
0.8251
0.2811
VRT [16]
6
31.60
0.8888
0.2077
27.93
0.8425
0.2723
IconVSR [1]
15
31.67
0.8948
0.1939
27.39
0.8279
0.2739
StableSR [36]
1
24.79
0.6897
0.2412
22.18
0.5904
0.3670
ResShift [45]
1
27.76
0.8013
0.2346
24.75
0.7040
0.3166
SATeCo [6]
6
31.62
0.8932
0.1735
27.44
0.8420
0.2291
SeeClear (Ours)
5
28.92
0.8279
0.1843
25.63
0.7605
0.2573
SeeClear⋆(Ours)
5
31.32
0.8856
0.1548
27.80
0.8404
0.2054

Clip 011, REDS4

Bicubic

VRT [16]

EDVR [37]

SeeClear⋆(Ours)

BasicVSR [1]

HR

Clip Calendar, Vid4

Bicubic

VRT [16]

EDVR [37]

SeeClear⋆(Ours)

BasicVSR [1]

HR

Figure 4: Qualitative results on the REDS4 and Vid4 datasets. SeeClear generates clearer content and
sharper textures.

the temporal coherence acquired by SeeClear, we also visualize two consecutive frames from the
generated HR videos, constructed using different diffusion-based VSR methodologies, as depicted
in Figure 5. ResShift synthesizes varied visual contents across two frames, such as the fluctuating
figures on the license plate. Contrarily, HR frames generated via SeeClear maintain a higher temporal
consistency and deliver pleasing textures.

4.3
Ablation Study

To assess the contribution of each component within the proposed SeeClear, we begin with a baseline
model and gradually integrate these modules. Specifically, all semantic-related operations are
bypassed, retaining solely spatial and channel self-attention and residual blocks, degenerating into a
diffusion-based image SR model without any condition. Subsequently, we incrementally introduce
the crafted semantic-conditional module into the baseline and formulate several variants. Their results
are listed in Table 2 and partially visualized in Figure 6.

8


---Page Break---
Table 2: Performance comparisons on REDS4 among variants with different semantic-condition
control by integrating InCAM and CaTeGory.

Baseline
DWT
Semantic
MFSA
InCAM
CaTeGory
PSNR ↑
SSIM ↑
LPIPS ↓
1
✓
✓
28.05
0.7993
0.2120
2
✓
✓
✓
28.08
0.7998
0.2088
3
✓
✓
✓
✓
27.99
0.7961
0.2053
4
✓
✓
✓
✓
✓
28.46
0.8098
0.1917
5
✓
✓
✓
✓
28.21
0.7986
0.2149
6
✓
✓
✓
✓
✓
28.74
0.8267
0.1938
7
✓
✓
✓
✓
✓
✓
28.92
0.8279
0.1843

Frame 76, Clip 010, REDS4
Frame 77, Clip 010, REDS4
(a)
(b)
(c)
(d)

Figure 5: Qualitative comparison of regions between consecutive frames. (a) and (b) are patches
produced by ResShift [45], derived from Frames 76 and 77 respectively. (c) and (d) display the
corresponding regions as generated through SeeClear.

Clip 020, REDS4

Model #1

Model #6

Model #4

HR

Figure 6: Visual comparisons of ablation for investigat-
ing the contribution of key modules.

First, the intra-frame semantic condition brings
about 1.5% improvements in LPIPS. Albeit the
multi-frame self-attention further improves the
perceptual quality, it also impairs the fidelity of
the restored video. Under the control of InCAM,
SeeClear can correlate semantically consistent
pixels in adjacent frames by combining intra-
frame and inter-frame semantic priors, elevating
the PSNR from 27.99 dB to 28.46 dB, and bring-
ing about 6.6% improvements in LPIPS. Further-
more, upon integrating the semantic priors from
CaTeGory, the fully-fledged SeeClear notably
enhances both the pixel-based and perception-
oriented metrics simultaneously. It indicates that the cooperative control of semantics is more
beneficial for generating videos of higher fidelity and better perceptual quality. As illustrated in
Figure 6, the baseline struggles to restore tiny and fine patterns without semantic condition, and it
gradually gains improvement accompanied by the strengthening of semantic control.

5
Conclusion

In this work, we present a novel diffusion-based video super-resolution framework named SeeClear.
It formulates the diffusion process by incorporating residual shifting mechanism and patch-level
blurring, constructing a Markov chain initiated with high-resolution frames and terminated at low-
resolution frames. It employs a semantic distiller and a pixel condenser for super-resolution during the
inverse sampling process. The instance-centric semantics distilled by the semantic distiller prompts
spatial modulation and temporal association in the devised Instance-Centric Alignment Module.
They are further assembled into Channel-wise Texture Aggregation Memory, providing abundant
conditions for temporal coherence and realistic content.

Acknowledgment. This work is supported in part by the National Natural Science Foundation of
China under Grant 62120106009 and Grant 62372036; and in part by the National Key Research and
Development Program of China under Grant 2022ZD0118001 and Grant 2021ZD0112100.

9


---Page Break---
References

[1] Kelvin C. K. Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. BasicVSR: The search for
essential components in video super-resolution and beyond. In CVPR, 2021.

[2] Kelvin C. K. Chan, Xiangyu Xu, Xintao Wang, Jinwei Gu, and Chen Change Loy. GLEAN: Generative
latent bank for image super-resolution and beyond. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 45(3):3154–3168, 2022.

[3] Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. BasicVSR++: Improving video
super-resolution with enhanced propagation and alignment. In CVPR, 2022.

[4] Pierre Charbonnier, Laure Blanc-Feraud, Gilles Aubert, and Michel Barlaud. Two deterministic half-
quadratic regularization algorithms for computed imaging. In ICIP, 1994.

[5] Chaofeng Chen, Shangchen Zhou, Liang Liao, Haoning Wu, Wenxiu Sun, Qiong Yan, and Weisi Lin.
Iterative token evaluation and refinement for real-world super-resolution. In AAAI, 2024.

[6] Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, and Tao Mei. Learning
spatial adaptation and temporal coherence in diffusion models for video super-resolution. In CVPR, 2024.

[7] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure
and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581,
2020.

[8] Shuting Dong, Feng Lu, Zhe Wu, and Chun Yuan. DFVSR: Directional frequency video super-resolution
via asymmetric and enhancement alignment network. In IJCAI, 2023.

[9] Jinjin Gu and Chao Dong. Interpreting super-resolution networks with local attribution maps. In CVPR,
2021.

[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

[11] Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. In ICLR, 2022.

[12] Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, and Humphrey Shi. Learning mask-aware clip
representations for zero-shot segmentation. In NeurIPS, 2023.

[13] Shuo Jin, Meiqin Liu, Chao Yao, Chunyu Lin, and Yao Zhao. Kernel dimension matters: To activate
available kernels for real-time video super-resolution. In ACM MM, 2023.

[14] Bingchen Li, Xin Li, Hanxin Zhu, Yeying Jin, Ruoyu Feng, Zhizheng Zhang, and Zhibo Chen. SeD:
Semantic-aware discriminator for image super-resolution. In CVPR, 2024.

[15] Feng Li, Yixuan Wu, Huihui Bai, Weisi Lin, Runmin Cong, and Yao Zhao. Learning detail-structure
alternative optimization for blind super-resolution. IEEE Transactions on Multimedia, 25:2825–2838,
2022.

[16] Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, and Luc
Van Gool. VRT: A video restoration transformer. IEEE Transactions on Image Processing, 33:2171–2182,
2024.

[17] Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai
Zhang, Radu Timofte, and Luc Van Gool. Recurrent video restoration transformer with guided deformable
attention. In NeurIPS, 2022.

[18] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and
Chao Dong. DiffBIR: Towards blind image restoration with generative diffusion prior. In ECCV, 2024.

[19] Ce Liu and Deqing Sun. A bayesian approach to adaptive video super resolution. In CVPR, 2011.

[20] Meiqin Liu, Shuo Jin, Chao Yao, Chunyu Lin, and Yao Zhao. Temporal consistency learning of inter-
frames for video super-resolution. IEEE Transactions on Circuits and Systems for Video Technology,
33(4):1507–1520, 2023.

[21] Meiqin Liu, Chenming Xu, Chao Yao, Chunyu Lin, and Yao Zhao. JNMR: Joint non-linear motion
regression for video frame interpolation. IEEE Transactions on Image Processing, 32:5283–5295, 2023.

[22] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer.
IEEE Signal Processing Letters, 20(3):209–212, 2012.

10


---Page Break---
[23] Brian Moser, Stanislav Frolov, Federico Raue, Sebastian Palacio, and Andreas Dengel. Waving goodbye to
low-res: A diffusion-wavelet approach for image super-resolution. In IJCNN, 2024.

[24] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung
Mu Lee. Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In CVPRW,
2019.

[25] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation.
In ICLR, 2022.

[26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution
image synthesis with latent diffusion models. In CVPR, 2022.

[27] Claudio Rota, Marco Buzzelli, and Joost van de Weijer. Enhancing perceptual quality in video super-
resolution through temporally-consistent detail synthesis using diffusion models. In ECCV, 2024.

[28] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi.
Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 45(4):4713–4726, 2022.

[29] Shuyao Shang, Zhengyang Shan, Guangxing Liu, LunQian Wang, XingHua Wang, Zekai Zhang, and
Jinglin Zhang. ResDiff: Combining cnn and diffusion model for image super-resolution. In AAAI, 2024.

[30] Shuwei Shi, Jinjin Gu, Liangbin Xie, Xintao Wang, Yujiu Yang, and Chao Dong. Rethinking alignment in
video super-resolution transformers. In NeurIPS, 2022.

[31] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based
diffusion models. In NeurIPS, 2021.

[32] Haoze Sun, Wenbo Li, Jianzhuang Liu, Haoyu Chen, Renjing Pei, Xueyi Zou, Youliang Yan, and Yujiu
Yang. CoSeR: Bridging image and language for cognitive super-resolution. In CVPR, 2024.

[33] Lingchen Sun, Rongyuan Wu, Zhengqiang Zhang, Hongwei Yong, and Lei Zhang. Improving the stability
of diffusion models for content consistent super-resolution. arXiv preprint arXiv:2401.00877, 2023.

[34] Qi Tang, Yao Zhao, Meiqin Liu, Jian Jin, and Chao Yao. Semantic Lens: Instance-centric semantic
alignment for video super-resolution. In AAAI, 2024.

[35] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of
images. In AAAI, 2023.

[36] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting
diffusion prior for real-world image super-resolution. International Journal of Computer Vision, pages
1–21, 2024.

[37] Xintao Wang, Kelvin C. K. Chan, Ke Yu, Chao Dong, and Chen Change Loy. EDVR: Video restoration
with enhanced deformable convolutional networks. In CVPRW, 2019.

[38] Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Recovering realistic texture in image super-
resolution by deep spatial feature transform. In CVPR, 2018.

[39] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. SeeSR: Towards
semantics-aware real-world image super-resolution. In CVPR, 2024.

[40] Yuhui Wu, Chen Pan, Guoqing Wang, Yang Yang, Jiwei Wei, Chongyi Li, and Heng Tao Shen. Learning
semantic-aware knowledge guidance for low-light image enhancement. In CVPR, 2023.

[41] Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with
task-oriented flow. International Journal of Computer Vision, 127(8):1106–1125, 2019.

[42] Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for
realistic image super-resolution and personalized stylization. In ECCV, 2024.

[43] Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motion-guided latent diffusion for temporally consistent
real-world video super-resolution. In ECCV, 2024.

[44] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and
Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the
wild. In CVPR, 2024.

11


---Page Break---
[45] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. ResShift: Efficient diffusion model for image
super-resolution by residual shifting. In NeurIPS, 2023.

[46] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A simple
framework for open-vocabulary segmentation and detection. In CVPR, 2023.

[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable
effectiveness of deep features as a perceptual metric. In CVPR, 2018.

[48] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. ControlVideo:
Training-free controllable text-to-video generation. In ICLR, 2024.

[49] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-A-Video:
Temporal-consistent diffusion model for real-world video super-resolution. In CVPR, 2024.

[50] Hongguang Zhu, Yunchao Wei, Xiaodan Liang, Chunjie Zhang, and Yao Zhao. CTP: Towards vision-
language continual pretraining via compatible momentum contrast and topology preservation. In ICCV,
2023.

12


---Page Break---
A
Mathematical Details

Patch-level Blurring Diffusion. Blurring Diffusion [11] designs the destruction process based on
heat dissipation [25] that retains low-frequency components of images over high frequencies. It
describes the thermodynamic process in which the temperature u(x, y, t) changes at position (x, y)
in a 2D plane with respect to the time t via a partial differential equation:
∂
∂tu (x, y, t) = ∆u (x, y, t) ,
t ∈{1, · · · , T} ,
(21)

where ∆= ∇2 denotes the Laplace operator. Based on Neumann boundary conditions (∂u/∂x =
∂u/∂y = 0) with zero-derivatives at the boundaries of the image, the solution is given by a diagonal
matrix in the frequency domain of the discrete cosine transform:

ut = eΛtu0,
t ∈{1, · · · , T} ,
(22)

where u0 is the initial state and Λ is a diagonal matrix with negative squared frequencies on the
diagonal. It is equivalent to a convolution with a Gaussian kernel with variance σ2
blur = 2t in the
realm of image processing. Blurring Diffusion further mixes Gaussian noise with variance σ2
t into
the blurring process to incorporate stochasticity into deterministic dissipation.

Patch-level blurring diffusion conducts the blurring process within a patch with the size of p × p to
make all pixel intensity the same instead of the whole image, which means Λx×p+y = −π2( x2

p2 + y2

p2 ).
More generally, it can be extended with any invertible transformation, corresponding to the marginal
distribution in the pixel space:

q
 
It
i | IHR
i

= N
 
It
i | R diag (αt) R−1IHR
i
, R diag (βt) R−1
,
t ∈{1, · · · , T} ,
(23)

where R and R−1 represent invertible transformation and corresponding inverse transformation.
diag (·) denotes the operation that projects a vector to a diagonal matrix. αt and βt are specific
schedules for mean and noise.

Blurring ResShift. ResShift [45] introduces a monotonically increasing sequence {ηt}T
t=1 to
gradually shift residual between low-resolution image and high-resolution one, whose marginal
distribution is defined as:

q
 
It
i | I0
i , ILR
i

= N
 
It
i | I0
i + ηte0, κ2ηtE

,
t ∈{1, · · · , T} ,
(24)

where e0 is the residuals between low-resolution and high-resolution frames. ηt controls the speed
of residual shifting and satisfies η1 →0 and ηT →1. After incorporation of patch-level blurring
diffusion and ResShift, ut can be sampled via

ut = eΛtu0 + ηtet + κ√ηtϵt ⇔ut = (1 −ηt) eΛtu0 + ηtul + κ√ηtϵt,
(25)

et = ul −Dtu0 ⇔et = ul −eΛtu0,
(26)

where ϵt ∼N (u | 0, E). Thus, the relation between ut and ut−1 can be obtained:

ˆu0 =
ut−1 −ηt−1ul
(1 −ηt−1)eΛ(t−1) ,
(27)

ut =
1 −ηt
1 −ηt−1
eΛ (ut−1 −ηt−1ul) + ηtul + κ√αtϵt,
(28)

where αt = ηt −ηt−1. ˆu0 is approximate HR frame. By recursively applying the sampling procedure
and reparameterization trick, we can rewrite the marginal distribution of Blurring ResShift as follows:

q (ut | u0, ul) = N
 
ut | Dtu0 + ηtet, κ2ηtE

,
t ∈{1, · · · , T} ,
(29)

According to Bayes’s theorem, there is

q (ut−1 | ut, u0, ul) ∝q (ut | ut−1, ul) q (ut−1 | u0, ul) ,
(30)

where

q (ut | ut−1, ul) = N

ut |
1 −ηt
1 −ηt−1
eΛ (ut−1 −ηt−1ul) + ηtul, κ2αtE

,
(31)

q (ut−1 | u0, ul) = N

ut−1 | (1 −ηt−1) eΛ(t−1)u0 + ηt−1ul, κ2ηt−1E

,
(32)

13


---Page Break---
Comparative Power Spectral Density (PSD) analysis

HR

LR

IHDM

Patch-Level Blurring

ResShift

SeeClear

Figure 4: Visual comparison of intermediate state at time step t via different diffusion processes.

And then, it only needs to focus on the quadratic term in the exponent of q (ut−1 | ut, u0, ul):

−


ut −
1−ηt
1−ηt−1 eΛ (ut−1 −ηt−1ul) −ηtul
 
ut −
1−ηt
1−ηt−1 eΛ (ut−1 −ηt−1ul) −ηtul
T

2κ2αt

−

 
ut−1 −(1 −ηt−1) eΛ(t−1)u0 −ηt−1ul
  
ut−1 −(1 −ηt−1) eΛ(t−1)u0 −ηt−1ul
T

2κ2ηt−1

= −1

2





(1−ηt)2

(1−ηt−1)2 e2Λ

κ2αt
+
1
κ2ηt−1



ut−1uT
t−1 +



1 −ηt

1 −ηt−1
eΛ ut +

ηt−1
1−ηt
1−ηt−1 eΛ −ηt

ul

κ2αt

+(1 −ηt−1) eΛ(t−1)u0 −ηt−1ul

κ2ηt−1


uT
t−1 + const

= −(ut−1 −µ) (ut−1 −µ)T

2σ2
+ const,

(33)

where

µ = ληt−1ut + αt (1 −ηt−1) eΛ(t−1)u0 +
 
λ2η2
t−1 −ληt−1ηt −αtηt−1

ul
λ2ηt−1 + αt
(34)

σ2 =
κ2αtηt−1
λ2ηt−1 + αt
,
(35)

λ =
1 −ηt
1 −ηt−1
eΛ,
(36)

where ‘const’ denotes the item that is independent of ut−1.

B
Rational Explanation

We analyze the final states of different diffusion processes via the power spectral density, which
reflects the distribution of frequency content in an image, as illustrated in Figure 4. It can be observed
that IHDM performs blurring globally and has a significant difference in frequency distribution
compared to the LR image, while the patch-level blurring is closer to the frequency distribution of
the LR. On this basis, SeeClear further introduces residual and noise. Compared to ResShift without
blurring, the diffusion process adopted by SeeClear makes the image more consistent with the LR in
the low-frequency components and introduces more randomness in the high-frequency components,
compelling the model to focus on the generation of high-frequency components.

14


---Page Break---
C
Network Structure

𝐼!"#

$
/ 𝐼!"#

%&

𝐼!"'

$
/ 𝐼!"'

%&

𝐼!

$ / 𝐼!

%&

𝐼!('

$
 / 𝐼!('

%&

𝐼!(#

$
 / 𝐼!(#

%&

𝐼!"#

)&

𝐼!"'

)&

𝐼!

)&

𝐼!('

)&

𝐼!(#

)&

Semantic 

Distiller

Pixel Condenser

𝓢

𝑂*, 𝐶+, 𝑇+

𝐼{!"#,⋯,!(#}

)&

𝓟

Frozen
Learnable

𝑃!

𝑂!
𝐶"
𝑇"

Instance-Centric 
Alignment Module

Channel-wise Texture 
Aggregation Memory

𝑂#

𝐹&!,% %&'

(

Associate

Assemble

Semantic Distiller

Pixel Condenser

Figure 8: The illustration of SeeClear. It comprises the diffusion process incorporating patch-level
blurring and residual shift mechanism and a reverse process. During the reverse process, Semantic
Distiller for semantic embedding extraction and U-shaped Pixel Condenser are employed for iterative
denoising. The devised InCAM and CaTeGory are inserted into the U-Net to utilize the diverse
semantics for inter-frame alignment in the diffusion-based VSR framework.

SeeClear consists of a forward diffusion process and a reverse process for VSR. In the diffusion
process, patch-level blurring and residual shift mechanism are integrated to degrade HR frames
based on the handcrafted time schedule. During the reverse process, a transformer-based network
for open vocabulary segmentation and a U-Net are employed for iterative denoising. The former is
responsible for extracting semantic embeddings related to instances from LR videos, similar to the
process of distillation in physics, and is therefore named the semantic distiller. The latter is utilized
to filter out interfering noise and retain valuable information from low-quality frames, similar to the
condensation process. All of them are tailored for image processing, and SeeClear takes diverse
semantic embeddings as conditions to enable the network to be aware of the generated content and
determine the aligned pixels from adjacent frames for the temporal consistency of the whole video.

As depicted in Figure 8, the attention-based Pixel Condenser primarily consists of three parts,
i.e., encoder, decoder, and middle block. As the input of the encoder, low-resolution frames are
concatenated with the intermediate states and processed through a convolution layer. The encoder
incorporates a window-based self-attention and channel-wise self-attention, alternating between two
residual blocks. These attention operations mine valuable information within and across the wavelet
sub-bands, and the residual blocks infuse the features with the intensity of degradation. Following
this, a wavelet-based downsample layer is deployed for feature downsampling.

Specifically, the features are decomposed into various sub-bands via a 2D discrete wavelet transform,
reducing the spatial dimensions while keeping the original data intact. Low-frequency features are
further fed into the subsequent layer of the encoder, while others are transmitted to the corresponding
wavelet-based upsample layer in the decoder via a skip connection. Additionally, the wavelet-based
downsample layers are utilized parallel along the encoder, refilling the downsampled features with
information derived from the low-resolution frames.

The devised Instance-Centric Alignment Module (InCAM) and Channel-wise Texture Aggregation
Memory (CaTeGory) are inserted into both the middle and decoder. Firstly, the InCAM spatially
modulates the features based on instance-centric semantics and aligns adjacent frames within the
clip in semantic space. Spatial self-attention employs these aligned features, substituting the original
features as input. Subsequently, features enhanced through channel-wise self-attention are embedded
as queries to seek assistance from the CaTeGory. Furthermore, the wavelet-based upsample layer
accepts features from the decoder and high-frequency information transmitted from the encoder. It
reinfuses the lost information from the encoder while scaling the feature size. The network concludes
with a convolution layer refining the features, which are added to the interpolated frames to generate
the final output.

15


---Page Break---
Table 3: Performance comparisons in terms of Full-Reference IQA (DISTS [7]) and No-Reference
IQA (NIQE [22] and CLIP-IQA [35]) evaluation metrics on the REDS4 [24] and Vid4 [19] datasets.
The extended version of SeeClear is marked with ⋆. Red indicates the best, and blue indicates the
runner-up performance (best view in color) in each group of experiments.

Methods
Frames
REDS4 [24]
Vid4 [19]
DISTS ↓NIQE ↓CLIP-IQA ↑DISTS ↓NIQE ↓CLIP-IQA ↑
Bicubic
-
0.1876
7.257
0.6045
0.2201
7.536
0.6817
TOFlow [41]
7
0.1468
6.260
0.6176
0.1776
7.229
0.7356
EDVR-M [37]
5
0.0943
4.544
0.6382
0.1468
5.528
0.7380
BasicVSR [1]
15
0.0808
4.197
0.6353
0.1442
5.340
0.7410
VRT [16]
6
0.0823
4.252
0.6379
0.1372
5.242
0.7434
IconVSR [1]
15
0.0762
4.117
0.6162
0.1406
5.392
0.7411
StableSR [36]
1
0.0755
4.116
0.6579
0.1385
5.237
0.7644
ResShift [45]
1
0.1432
6.391
0.6711
0.1716
6.868
0.7157
SATeCo [6]
6
0.0607
4.104
0.6622
0.1015
5.212
0.7451
SeeClear (Ours)
5
0.0762
4.381
0.6870
0.0947
5.305
0.7106
SeeClear⋆(Ours)
5
0.0641
3.757
0.6848
0.0919
4.896
0.7303

Frames
Bicubic
EDVR [37]
BasicVSR [1]
VRT [16]
SeeClear
HR

Figure 9: Visual examples of video super-resolution results by different approaches on the REDS4
and Vid4 datasets. The region in the red box is presented in the zoom-in view for comparison.

D
Additional Experiments

We provide additional evaluation metrics and visual comparisons contrasting the existing VSR
methods with our proposed SeeClear. As demonstrated in Figure 9, our proposed method successfully
generates pleasing images without priors pre-trained on large-scale image datasets, showcasing
sharp edges and clear details, evident in vertical bar patterns of windows and characters on license
plates. Conversely, prevalent methods seemingly struggle, causing texture distortion or detail loss in
analogous scenes.

We also execute several experiments focused on the degradation scheme of the diffusion process,
verifying the performance of the models on the first frame, as indicated in Table 4. Compared to the
straightforward diffusion process of simply adding Gaussian noise into HR frames, the combination
of blurring diffusion proves beneficial in generating the high-frequency details discarded in LR

16


---Page Break---
Table 4: Performance comparisons on the first frame of REDS4 among variants with different
deterioration during the diffusion process and U-Net.

Noise
σ2
B
Model
PSNR ↑
SSIM ↑
LPIPS ↓
1
ResShift
0
WaveDiff
26.78
0.7960
0.2096
2
ResShift
2
WaveDiff
26.45
0.7927
0.2008
3
ResShift
3
WaveDiff
27.74
0.8047
0.2067
4
ResShift
0
SwinUNet
27.76
0.8013
0.2346
5
ResShift
2
SeeClear
28.04
0.8134
0.1971

frames. Specifically, the variation in the intensity of blur (Line 1-3) affects the fidelity and perceptual
quality. Among them, there is no blurring when σ2
B = 0, and the greater the value of σ2
B, the greater
the blurring intensity. It can be observed there is a 0.96 dB improvement in PSNR and the value of
LPIPS ranges from 0.2096 to 0.2067 with the increasement of σ2
B.

On another note, implementing attention mechanisms in the wavelet spectrum proves more successful
in uncovering valuable insights than in the pixel domain. Additionally, SeeClear introduces the
alternation of spatial self-attention and channel self-attention, refining the modeling of intra and
inter-frequency sub-band correlations and remarkably enhancing the quality of the generated high-
resolution frames.

E
Complexity Analysis

Table 5: Complexity comparisons between different
diffusion-based super-resolution methods.

Methods
Params
Time
Infer
Steps Time (s)

SISR

LDM [26]
169.0 M
200
5.21
DiffBIR [18]
1716.7M
50
5.85
ResShift [45]
173.9M
15
1.12
StableSR [36]
1409.1M 200
18.70
CoSeR [32]
2655.52M 200
-
SeeSR [39]
2283.7M
50
7.24
PASD [42]
1900.4M
20
6.07

VSR

StableVSR [27]
712M
50
28.57
MGLD-VSR [43]
1.5B
50
1.113
SeeClear (Ours)
229.23M
15
1.142

Table 5 compares the efficiency between our
proposed method and diffusion-based meth-
ods.
It presents the number of parame-
ters of different models and their inference
time for super-resolving 512 × 512 frames
from 128 × 128 inputs.
Combining these
comparative results, we draw the follow-
ing conclusions: i) Compared to semantic-
assisted single-image super-resolution (e.g.,
CoSeR [32] and SeeSR [39]), our proposed
method possesses fewer parameters and higher
inference efficiency. ii) In contrast to existing
diffusion-based methodologies for VSR [27,
43], SeeClear is much smaller and runs faster,
benefiting from the reasonable module de-
signs and diffusion process combing patch-
level blurring and residual shift mechanism.

F
Generation Process

A video clip consisting of five frames is parallelly sampled during the inference process. These LR
frames are first fed into the semantic distiller to extract semantic tokens and then corrupted by random
noise as the input of the pixel condenser. The pixel condenser iteratively generates the corresponding
HR counterparts from noisy LR frames under the condition of LR frames and semantic priors. The
pseudo-code of the inference is depicted in the Algorithm 1.

G
Limitations and Societal Impacts

Limited by the size and diversity of the dataset, SeeClear, being solely trained on video super-
resolution datasets, does not fully leverage the generative capabilities of diffusion models as efficiently
as those that benefit from pre-training on large-scale image datasets such as ImageNet. While SeeClear
is capable of generating sharp textures and maintaining consistent details, it falls short in restoring
tiny objects or intricate structures with complete satisfaction.

17


---Page Break---
Algorithm 1: Generation Process of SleeCear

Input: LR frames ILR
n
∈RN×C×H×W ; time steps T; and predefined schedule {α, η}
Output: SR frames ISR
n
∈RN×C×sH×sW

1 ¯ILR
n
= VRT(ILR
n )

2 uT
n = DWT2D(¯ILR
n ) + ϵ, ϵ ∼N(0, I)

3 —————————— Parallel Generation within Local Video Clip ——————————

4 for m = 1 to N

M do
// M refers to the number of frames in a video clip

5
for t = T −1 to 0 do

6
if t=T-1 then

7
On, Pn = S(ILR
n , V)

8
ut
n = uT
n

9
else

10
ut
n = DWT2D(¯It+1
n
)

11
skipH = [ ]
// Array of high-frequency spectrums for skip connection

12
for i = 1 to 4 do
// i refers to ith layer of encoder E

13
if i=1 then

14
f i−1
n
= [It
n, ut
n]

15
else

16
f i−1
n
= [WD(ILR
n ), ¯f i−1
n
]
// WD denotes Wavelet-based DownSample

17
ˆf i
n = Ei(f i−1
n
)

18
if i ̸= 4 then

19
Li
n, Hi
n = WD( ˆf i
n)

20
¯f i
n = Li
n
21
skipH = skipH + Hi
n

22
else

23
¯f i
n = ˆf i
n

24
Om = Dec(Enc(On), ˆO)

25
for j = 1 to 3 do
// j refers to jth layer of middle blocks B

26
γj
n, βj
n = G(Pn)

27
f j
n = ( ¯f j−1
n
J γj
n + βj
n) + ¯f j−1
n

28
ˆf j
n = CA(f j
n, Om, Om)

29
¯f j
n = Bj(MFSA(Oc · ˆf j
n))

30
feats = [ ]
// Array of multi-scale features for CaTeGory

31
for k = 1 to 4 do
// k refers to kth layer of decoder D

32
if k̸=4 then

33
f k
n = WU([ ¯f k−1
n
, skipH[−k]]) // WU denotes Wavelet-based UpSample

34
γk
n, βk
n = G(Pn)

35
ˆf k
n = (f k−1
n
J γk
n + βk
n) + ¯f k−1
n

36
ef k
n = CA( ˆf k
n, Oc, Oc)

37
¨f k
n = Dk(MFSA(Oc · ef k
n))

38
¯f k
n = CA( ¨f k
n, Cm, Tm)

39
feats = feats + ¯f k
n

40
It
n = IDWT2D( ¯fn) + ¯ILR
n
41
¯It
n = αtµ(It
n, ¯It+1
n
) + ηtϵ

42
———————— Update CaTeGory for Global Video Consistency ————————

43
if t=0 then

44
Cm+1, Tm+1 = M(Cm, Tm, feats)

45
ISR
n
= It
n

18


---Page Break---
In addition, compared to synthetic datasets, videos captured in real-world scenarios display more
complex and unknown degradation processes. Although real-world VSR is garnering significant
attention, it remains an unexplored treasure and a steep summit to conquer for diffusion-based VSR,
including SeeClear.

Significantly, substituting an Autoencoder with a traditional wavelet transform lessens the computa-
tional burden while ensuring the preservation of the original input information. Nevertheless, in the
process of inverse wavelet transform, the spatio-temporal information within videos is not delved
into further, leading to subpar outcomes. Meanwhile, some methods develop trainable wavelet-like
transformations based on the lifting scheme, allowing for end-to-end training of the whole network.
Such a schema presents a promising direction for future research by potentially boosting model
performance.

As for societal impacts, similar to other restoration methods, SeeClear may bring privacy concerns
after restoring blurry videos and lead to misjudgments if used for medical diagnosis and intelligent
security systems, etc.

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: See Section 3 and 4 of the main paper and Section A, C and D of the
supplementary material.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made
in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or NA
answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much
the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: See Section G of the supplemental material.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings, model
well-specification, asymptotic approximations only holding locally). The authors should
reflect on how these assumptions might be violated in practice and what the implications
would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only
tested on a few datasets or with a few runs. In general, empirical results often depend on
implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is
low or images are taken in low lighting. Or a speech-to-text system might not be used
reliably to provide closed captions for online lectures because it fails to handle technical
jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and
how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important
role in developing norms that preserve the integrity of the community. Reviewers will be
specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]

20


---Page Break---
Justification: See Section A of the supplementary material.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they
appear in the supplemental material, the authors are encouraged to provide a short proof
sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: See Section 4 of the main paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well
by the reviewers: Making the paper reproducible is important, regardless of whether the
code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may be
necessary to either make it possible for others to replicate the model with the same dataset,
or provide access to the model. In general. releasing code and data is often one good
way to accomplish this, but reproducibility can also be provided via detailed instructions
for how to replicate the results, access to a hosted model (e.g., in the case of a large
language model), releasing of a model checkpoint, or other means that are appropriate to
the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to
reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct the
dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors
are welcome to describe the particular way they provide for reproducibility. In the
case of closed-source models, it may be that access to the model is limited in some
way (e.g., to registered users), but it should be possible for other researchers to have
some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

21


---Page Break---
Answer: [Yes]

Justification: See Section 4 of the main paper. We plan to release the code after the paper is
accepted.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to
access the raw data, preprocessed data, intermediate data, and generated data, etc.
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

Justification: See Section 4 of the main paper.

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

Justification: See Section D of the supplementary material.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the main
claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall run
with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call
to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).

22


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error of
the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably
report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality
of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures
symmetric error bars that would yield results that are out of range (e.g. negative error
rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: See Section 4 of the main paper.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or
cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than
the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t
make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We confirm that the research conform with the NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special considera-
tion due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

Justification: See Section G of the supplementary material.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g.,
deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

23


---Page Break---
• The conference expects that many papers will be foundational research and not tied to
particular applications, let alone deployments. However, if there is a direct path to any
negative applications, the authors should point it out. For example, it is legitimate to point
out that an improvement in the quality of generative models could be used to generate
deepfakes for disinformation. On the other hand, it is not needed to point out that a
generic algorithm for optimizing neural networks could enable people to train models that
generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being
used as intended and functioning correctly, harms that could arise when the technology is
being used as intended but gives incorrect results, and harms following from (intentional
or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks, mecha-
nisms for monitoring misuse, mechanisms to monitor how a system learns from feedback
over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not
require this, but we encourage authors to take this into account and make a best faith
effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]

Justification: We cite the original paper that produced the code package and dataset we used.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a
dataset.
• For existing datasets that are re-packaged, both the original license and the license of the
derived asset (if it has changed) should be provided.

24


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to the
asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We plan to gradually open-source after the paper is accepted.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their sub-
missions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset
is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribution
of the paper involves human subjects, then as much detail as possible should be included
in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or
other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
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

25


---Page Break---
