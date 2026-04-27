SC3D: Self-conditioned Generative Gaussian Model
with 3D-aware Feedback

Anonymous Author(s)
Affiliation
Address
email

Abstract

Existing single image-to-3D creation methods typically involve a two-stage pro-
1

cess, first generating multi-view images, and then using these images for 3D
2

reconstruction. However, training these two stages separately leads to significant
3

data bias in the inference phase, thus affecting the quality of reconstructed results.
4

We introduce a unified 3D generation framework, named SC3D, which integrates
5

diffusion-based multi-view image generation and 3D reconstruction through a
6

self-conditioning mechanism. In our framework, these two modules are established
7

as a cyclic relationship so that they adapt to the distribution of each other. During
8

the denoising process of multi-view generation, we feed rendered color images and
9

maps by SC3D itself to the multi-view generation module. This self-conditioned
10

method with 3D aware feedback unites the entire process and improves geometric
11

consistency. Experiments show that our approach enhances sampling quality, and
12

improves the efficiency and output quality of the generation process.
13

1
Introduction
14

3D content creation from a single image have improved rapidly in recent years with the adoption of
15

large 3D datasets [1, 2, 3] and diffusion models [4, 5, 6]. A body of research [7, 8, 9, 10, 11, 12, 13, 14]
16

has focused on multi-view diffusion models, fine-tuning pretrained image or video diffusion models
17

on 3D datasets to enable consistent multi-view synthesis. These methods demonstrate generalizability
18

and produce promising results. Another group of works [15, 16, 17, 18, 19] propose generalizable
19

reconstruction models, generating 3D representation from one or few views in a feed-forward process.
20

Theses reconstruction models built upon convolutional network or transformer backbone, have led to
21

efficient image-to-3D creation.
22

Since single-view reconstruction models [15] trained on 3D datasets [1, 20] lack generalizability
23

and often produce blurring at unseen viewpoints, several works [21, 16, 18, 19] extend models to
24

sparse-view input, boosting the reconstruction quality. As shown in Fig. 1, these methods split 3D
25

generation into two stages: multi-view synthesis and 3D reconstruction. By combining generalizable
26

multi-view diffusion models and robust sparse-view reconstruction models, such pipelines achieve
27

high-quality image to 3D generation. However, combining the two independently designed models
28

introduces a significant “data bias” to the reconstruction model. The data bias is mainly reflected in
29

two aspects: (1) Multi-view bias. Multi-view diffusion models learn consistency at the image level,
30

struggle to ensure geometric consistency. When it comes to reconstruction, multi-view images that
31

lack geometric consistency affect the subsequent stage. (2) Limited data for reconstruction model.
32

Unlike multi-view diffusion models, reconstruction models which are trained from scratch on limited
33

3D dataset, lacks the generalization ability.
34

Recent works like IM-3D [22] and VideoMV [23] have attempted to aggregate the rendered views of
35

the reconstructed 3D model into previous-step multi-view synthesis, thus improving the capability
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Input Image
Multi-view
Diffusion Model

Reconstruction 

Model
3D Representation

T steps

Input Image
Multi-view
Diffusion Model

Reconstruction 

Model
3D Representation

SC3D
3D-aware Feedback

T steps

𝑥"!

(a) Two-stage pipeline

(b) SC3D framework 

Figure 1: Concept comparison between SC3D and previous two-stage methods. Instead of directly
combining multi-view diffusion model and reconstruction model, our self-conditioned framework
involves joint training of these two models and establish them as a cyclic association. During the
denoising process, rendered 3D-aware maps are fed to the multi-view generation module.

and consistency of the generated multi-view images. These methods integrate the aforementioned two
37

stages at the inference phase. But the models at both stages still lack joint training, which prevents
38

the reconstruction model from enhancing its robustness to the generated poor multiviews. Moreover,
39

these test-time aggregating methods cannot directly utilize geometric information such as depth maps,
40

normal maps, or position maps that can also be obtained from the reconstructed 3D. Notably, these
41

explicit 3D aware maps can better guide the multi-view generation.
42

To address these challenges, we propose a unified single image-to-3D creation framework, named
43

SC3D, which integrates multi-view generation and 3D reconstruction through a self-conditioning
44

mechanism. Our framework involves jointly training the multi-view diffusion model and the recon-
45

struction model. In SC3D, these two modules are established as a cyclic relationship so that they
46

adapt to the characteristics of each other, enabling robust generation at inference. Specifically, during
47

the denoising process, we feed rendered 3D-aware maps from the reconstructed 3D to the multi-view
48

generation module. By leveraging the color maps and spatial canonical coordinates maps from the
49

reconstruction 3D representation as condition, our multi-view diffusion model synthesizes multi-view
50

images that better conform to the actual 3D structure. This self-conditioned framework with 3D
51

aware feedback unites the 3D generation process and enhances the robustness for unseen complex
52

scenes. Experiments on the GSO dataset [24] validate that our SC3D reduces data bias between
53

training and inference, and enhances the overall efficiency and output quality.
54

Our key contributions are as follows:
55

• We introduce SC3D, which unifies multi-view generation and 3D reconstruction in a single
56

framework and involves jointly training these two modules, enabling adaption to each other.
57

• SC3D employs a self-conditioning mechanism with 3D-aware feedback, using rendered 3D-aware
58

maps to guide the multi-view generation, ensuring better geometric consistency and robustness.
59

• Experiments show that SC3D significantly reduces data bias, improves the quality of 3D recon-
60

struction, and enhances overall efficiency in creating 3D content from a single image.
61

2
Related Work
62

Image/Video Diffusion for Multi-view Generation
Diffusion models [25, 26, 27, 28, 29, 30, 31,
63

32, 33, 34] have demonstrated their powerful generative capabilities in image and video generation
64

fields. Current research [7, 8, 9, 10, 11, 12, 13, 14, 35] fine-tunes pretrained image/video diffusion
65

models on 3D datasets like Objaverse [1] and MVImageNet [20]. Zero123 [7] introduces relative
66

view condition to image diffusion models, enabling novel view synthesis from a single image
67

and preserving generalizability. Based on it, methods like SyncDreamer [9], ConsistNet [36] and
68

EpiDiff [11] design attention modules to generate consistent multi-view images. These methods fine-
69

2


---Page Break---
tuned from image diffusion models produce generally promising results. By considering multi-view
70

images as consecutive frames of a video (e.g., orbiting camera views), it naturally leads to the idea of
71

applying video generation models to 3D generation [13]. However, since the diffusion model is not
72

explicitly modeled in 3D space, the generated multi-view images often struggle to achieve consistent
73

and robust details.
74

Image to 3D Reconstruction
Recently, the task of reconstructing 3D objects has evolved from
75

traditional multi-view reconstruction methods [37, 38, 39, 40] to feed-forward reconstruction mod-
76

els [15, 41, 42, 16, 17, 18, 19]. Ultilizing one or few shot as input, these highly generalizable
77

reconstruction models synthesize 3D representation, enabling the rapid generation of 3D objects.
78

LRM [15] proposes a transformer-based model to effectively map image tokens to 3D triplanes.
79

Instant3D [21] further extends LRM to sparse-view input, significantly boosting the reconstruction
80

quality. LGM [16] and GRM [17] replace the triplane representation with 3D Gaussians [40] to enjoy
81

its superior rendering efficiency. CRM [18] and InstantMesh [19] optimize on the mesh representation
82

for high-quality geometry and texture modeling. These reconstrucion models built upon convolutional
83

network architecture or transformer backbone, have led to efficient image-to-3D creation.
84

Pipelines of 3D Generation
Early works propose to distill knowledge of image prior to create 3D
85

models via Score Distillation Sampling (SDS) [43, 44, 45], limited by the low speed of per-scene
86

optimization. Several works [9, 11, 14, 22] fine-tune image diffusion models to generate multi-view
87

images, which are then utilized for 3D shape and appearance recovery with traditional reconstruction
88

methods [46, 40]. More recently, several works [21, 16, 18, 19, 23] involve both multi-view diffusion
89

models and feed-forward reconstruction models in the generation process. Such pipelines attempt
90

to combine the processes into a cohesive two-stage approach, thus achieving highly generalizable
91

and high-quality single-image to 3D generation. However, due to the lack of explicit 3D modeling,
92

the results generated by the multi-view diffusion model cannot guarantee strong consistency, which
93

will lead to data deviation for the reconstructed model between the testing phase and the training
94

phase. Compared to them, we propose a unified pipeline, integrating the two stages through a
95

self-conditioning mechanism at the training stage, with 3D aware feedback for high consistency.
96

3
Method
97

Given a single image, SC3D aims to generate multiview-consistent images with a reconstructed 3D
98

Gaussion model. To reduce the data bias and improve robustness of the generation, we propose SC3D,
99

a unified 3D generation framework which integrates multi-view synthesis and 3D reconstruction
100

through a self-conditioning mechanism. As illustrated in Fig. 2, the proposed framework involves a
101

video diffusion model (SVD [32]) as multi-view generator (refer to Section 3.1) and a feed-forward
102

reconstruction model to recover a 3D Gaussian Splatting (refer to Section 3.2. Moreover, we introduce
103

a self-conditioning mechanism, feeding the 3D-aware information obtained from the reconstruction
104

module back to the multi-view generation process (refer to Section 3.3). The 3D-aware denoising
105

sampling strategy iteratively refines the multi-view images and the 3d model, thus enhancing the final
106

production.
107

3.1
Video Diffusion Model as Multiview Generator
108

Recent video diffusion models such as those in [13, 34] have demonstrated a remarkable capability
109

to generate 3D-aware videos by scaling up both the model and dataset. Our research employs
110

the well-known Stable Video Diffusion (SVD) Model, which generates videos from image input.
111

Formally, given an image I ∈R3×h×w, the model is designed to generate a video V ∈Rf×3×h×w.
112

Further details about SVD can be found in Appendix A.1.
113

We enhance the video diffusion model with camera control c to generate images from different
114

viewpoints. Traditional methods encode camera positions at the frame level, which results in all
115

pixels within one view sharing the same positional encoding [47, 13]. Building on the innovations
116

of previous work [11, 35], we integrate the camera condition c into the denoising network by
117

parameterizing the rays r = (o, o × d). Specifically, we use two-layered MLP to inject Plücker
118

ray embeddings for each latent pixel, enabling precise positional encoding at the pixel level. This
119

approach allows for more detailed and accurate 3D rendering, as pixel-specific embedding enhances
120

the model’s ability to handle complex variations in depth and perspective across the video frames.
121

3


---Page Break---
𝒙!

"

U-Net

𝒙!#$

"

…

𝒙"%

"

Multi-view Generation

Decoder

Reconstruction 

Model

Reconstruction

Feedback

+
color 
images

coordinates

maps

+

𝒙%

"

Decoder

Images

Color
Encoder

CCM
Encoder

Input Image

Ray Embeddings

Figure 2: Overview of SC3D. We adopt a video diffusion model as the multi-view generator by
incorporating the input image and relative camera poses. In the denoising sampling loop, we decode
the predicted exf
0 to noise-corrupted images, which are then used to recover 3D representation by
a feed-forward reconstruction model. Then the rendered color images and coordinates maps are
encoded and fed into the next denoising step. At inference, the 3D-aware denoising sampling strategy
iteratively refines the images by incorporating feedback from the reconstructed 3D into the denoising
loop, enhancing multi-view consistency and image quality.

In our framework, unlike existing two-stage methods, our multi-view diffusion model does not
122

complete multiple denoising steps independently. In contrast, in the denoising sampling loop, we
123

obtain the straightly predicted exf
0 at the current timestep, which will be used for subsequent 3D
124

reconstruction. Then we use rendered 3d-aware view maps as conditions to guide the next denoising
125

step. Therefore, at each sampling step, we do the reparameterization of the output from the denoising
126

network Fθ to convert it into exf
0. Taking a single view as an example, we processes the denoised
127

image cin(σ)x and the associated noise level cnoise(σ), which σ indicates the standard deviation of
128

the noise. The reparameterization is formulated as follows:
129

˜x0 = cskip(σ)x + cout(σ)Fθ(cin(σ)x; cnoise(σ)).
(1)

The above operation process adjusts the output of Fθ to exf
0, which will be decoded into images and
130

passed to the subsequent 3D reconstruction module.
131

3.2
Feed-Forward Reconstruction Model
132

In the SC3D framework, the feed-forward reconstruction model is designed to recover 3D models
133

from pre-generated multi-view images, which can be images decoded from straightly predicted exf
0,
134

or completely denoised images. We utilize Large Multi-View Gaussian Model (LGM) [16] G as our
135

reconstruction module due to its real-time rendering capabilities that benefit from 3D representation of
136

Gaussian Splatting. This method integrates seamlessly with our jointly training framework, allowing
137

for quick adaptation and efficient processing.
138

We pass four specific views from the reparameterized output exf
0 to the Large Gaussian Model (LGM)
139

for 3D Gaussian Splatting reconstruction. To enhance the performance of LGM, particularly its
140

sensitivity to different noise levels cnoise(σ) and image details, we introduce a zero-initialized time
141

embedding layer within the original U-Net structure of the LGM. This innovative modification
142

enables the LGM to dynamically adapt to the diverse outputs that arise at different stages of the
143

4


---Page Break---
denoising process, thereby substantially improving its capacity to accurately reconstruct 3D content
144

from images that have undergone partial denoising.
145

The loss function employed for the fine-tuning of the LGM is articulated as follows:
146

LG = Lrgb(x0, G(˜x0, cnoise(σ))) + λLLPIPS(x0, G(˜x0, cnoise(σ))).
(2)

where we have utilized the mean square error loss Lrgb for the color channel and a VGG-based
147

perceptual loss LLPIPS[43] for the LPIPS term. In practical applications, the weighting factor λ is
148

conventionally set to 1.
149

Additionally, to maintain the model’s reconstruction capability for normal images, we also input the
150

model without adding noise and calculate the corresponding loss. In this case, we set cnoise(σ) to 0.
151

3.3
3D-Aware Feedback Mechanism
152

As shown in Fig. 2, we adopt a 3D-aware feedback mechanism that involves the rendered color
153

images and geometric maps produced by our reconstruction module in a denoising loop to further
154

improve the multi-view consistency of the resulting images and facilitate cyclic adaptation of the
155

two stages. Instead of integrating multi-view generation and 3D reconstruction at the inference stage
156

using re-sampling strategy [22, 23], we propose to train these two modules jointly to support more
157

informative feedback. Specifically, in addition to the rendered color images, our flexible framework
158

is able to derive additional geometric features to guide the generation process, which brings guidance
159

of more explicit 3D information to multi-view generation.
160

In practice, we obtain color images and canonical coordinates maps [48] from the reconstructed 3D
161

model, and utilize them as condition to guide the next denoising step of multi-view generation. We
162

use position maps instead of depth maps or normal maps as the representative of geometric maps
163

because canonical coordinate maps record the vertex coordinate values after normalization of the
164

overall 3D model, rather than the normalization of the relative self-view (such as depth maps). This
165

operation enables the rendered maps to be characterized as cross-view alignment, providing the strong
166

guidance of more explicit cross-view geometry relationship. The details of canonical coordinates
167

map can be found in Appendix A.2.
168

We adopt a 3D-aware self-conditioning [49] training and inference strategy that leverages reconstruc-
169

tion stage results to enhance multi-view consistency and the quality of generated images. During
170

training, the original denoising network Fθ(x; σ) is augmented with a 3D-aware feedback denoising
171

network Fθ(G(˜x0); σ), where G(˜x0) is the output of the LGM reconstruction.
172

To encode color images and coordinates maps into the denoising network of multi-view generation
173

module, we design two simple and lightweight encoders for color images and coordinates maps using
174

a series of convolutional neural networks, like T2I-Adapter [50]. The encoders are composed of four
175

feature extraction blocks and three downsample blocks to change the feature resolution, so that the
176

dimension of the encoded features is the same as the intermediate feature in the encoder of U-Net
177

denoiser. The extracted features from the two conditional modalities are then added to the U-Net
178

encoder at each scale.
179

Training Strategy
As illustrated in Algorithm 1, to train a 3D-aware multi-view generation network,
180

we use the rendered maps by the 3D reconstruction module as the self-conditioning input. In practice,
181

we randomly use this self-conditioning mechanism with a probability of 0.5. When not using the 3D
182

reconstruction result, we set G(˜x0) = 0 as the input. This probabilistic approach ensures balanced
183

learning, allowing the model to effectively incorporate 3D information without over-reliance on it.
184

5


---Page Break---
Algorithm 1 Training SC3D with the self-conditioned strategy.
def train_loss(x, cond_image):

"""Returns the loss on a training example x."""
# Sample sigma from a log-normal distribution
sigma = log_normal(P_mean, P_std)

# Reparameterize sigma to obtain conditioning parameters
c_in, c_out, c_skip, c_noise, lambda_param = reparameterizing(sigma)

# Add noise to input data
noise_x = x + sigma * normal(mean=0, std=1)
input_x = c_in * noise_x

# Initial prediction without self-conditioning
self_cond = None
F_pred = net(input_x, c_noise, cond_image, self_cond)
pred_x = c_out * F_pred + c_skip * noise_x

# Update self_cond using the reconstruction model
self_cond = recon_model(pred_x, c_noise)

# Use rendered maps as condition and denoise
if self_cond and np.random.uniform(0, 1) > 0.5:
F_pred = net(input_x, t, cond_image, self_cond.detach())
pred_x = c_out * F_pred + c_skip * noise_x

# Compute loss
loss = lambda_param * (pred_x - target) ** 2
recon_loss = recon_loss_fn(self_cond, x)

return loss.mean() + recon_loss

Inference/sampling strategy
At the inference stage, as shown in Algorithm 2, the 3D feedback
185

G(˜x0) is initially set to 0. At each timestep, this feedback is updated with the previous reconstruction
186

result G(˜x0). This iterative process refines the 3D representation, ensuring each frame benefits from
187

prior reconstructions, leading to higher quality and more consistent 3D-aware images.
188

Algorithm 2 Sampling algorithm of SC3D.
def generate(sigmas, cond_image):
self_cond = None
x_T = normal(mean=0, std=1)
# Initialize latent variable with Gaussian noise
for sigma in sigmas:

# Reparameterize sigma to obtain conditioning parameters
c_in, c_out, c_skip, c_noise, lambda_param = reparameterizing(sigma)

# Add noise to the latent variable
noise_x = x_T + sigma * normal(mean=0, std=1)
input_x = c_in * noise_x

# Generate prediction
F_pred = net(input_x, t, cond_image, self_cond)
pred_x = c_out * F_pred + c_skip * noise_x

# Update self_cond using the reconstruction model
self_cond = recon_model(pred_x, c_noise)

return pred_x

6


---Page Break---
Figure 3: Qualitative comparison with ImageDream-LGM and Our LGM.

Figure 4: Qualitative comparison with no-feedback and 3d-aware feedback.

4
Experiments
189

We focus on 3D asset content synthesis, training our model on the G-Objaverse [1, 51] dataset and
190

the LVIS subset of Objaverse, which consists of 300K high-quality 3D objects and is widely used in
191

3D generation. We evaluate SC3D on the Google Scanned Object (GSO) dataset [24], which consists
192

of approximately 1,000 scanned models, and we randomly select 100 samples for comparison. We
193

adopt TripoSR[42], SyncDreamer[9], SV3D[13], ImageDream [8] combined with LGM [16] as the
194

baseline approach [16] and VideoMV[23] as baseline methods. For each baseline, we report PSNR,
195

SSIM, and LPIPS metrics.
196

4.1
Comparison results
197

For LGM, we utilize the official LGM single-image generation pipeline, which employs ImageDream
198

[52] to transition from a single image input to multiple images. However, the conical coordinate
199

system employed by ImageDream complicates the direct evaluation of the output. To address this,
200

we use the official code to test on the GSO dataset, followed by manual calibration to assess the
201

generated quality, as illustrated in Fig. 3. The misalignment between the two stages of ImageDream
202

and LGM often results in generated models with blurred linear edges and geometric ambiguities.
203

Nonetheless, our LGM, enhanced by a feedback mechanism, demonstrates significantly improved
204

geometric and texture quality, producing results that closely approximate reality.
205

As illustrate in 6, We find that although it can generate very continuous frames, the generated
206

content tends to deviate from the given input image. This results in sub-optimal performance in
207

7


---Page Break---
Method
Resolution
PSNR↑
SSIM↑
LPIPS↓

TripoSR
256 × 256
18.481
0.8506
0.1357
SyncDreamer
256 × 256
20.056
0.8163
0.1596
SV3D
576 × 576
21.042
0.8497
0.1296
VideoMV(SD)
256 × 256
17.459
0.806
0.1446
VideoMV(GS)
256 × 256
17.577
0.807
0.1454

SC3D (SVD)
512 × 512
21.625
0.9045
0.1011
SC3D (GS)
512 × 512
21.761
0.9094
0.0991
Table 1: Comparison of performance metrics across different models and configurations.

Input image
Rendered multi-views from Generated 3DGS

Figure 5: Out of distribution testing results.

the reconstruction metric. Additionally, VideoMV training the LGM separately with noisy images
208

deteriorates, resulting in a visually noticeable reduction in its ability to generate texture details.
209

4.2
Ablation study
210

To validate the effectiveness of the proposed SC-3D framework, we conducted a series of ablation
211

studies comparing PSNR, SSIM, and LPIPS metrics for different configurations (Table 2). We start
212

with the base video diffusion model we trained, We then introduced 3D coordinates map feedback
213

and RGB texture feedback from the reconstruction model to the diffusion model, which improved
214

geometric consistency and texture detail across views. Combining both feedback mechanisms in the
215

SVD + 3D-aware Feedback configuration resulted in the best performance, demonstrating significant
216

improvements in the final 3D reconstruction quality by enhancing both geometric consistency and
217

texture detail preservation.
218

Method
Variant
PSNR ↑
SSIM ↑
LPIPS ↓

SVD
SVD
20.038
0.8745
0.1253
GS
20.549
0.8651
0.1183
SVD + Coordinates Map Feedback
SVD
21.021
0.8973
0.1110
GS
21.325
0.8937
0.1092
SVD + 3D-aware Feedback
SVD
21.752
0.9122
0.0993
GS
21.761
0.9094
0.0991
Table 2: Performance metrics of different feedback mechanisms.

8


---Page Break---
Figure 6: The Generation Example of VideoMV

We also demonstrate the impact of incorporating feedback mechanisms on the two models, as shown
219

in Table 3. It can be observed that when no feedback mechanism is used, there is a significant
220

discrepancy between the two models’ modalities, which leads to a degradation in their combined
221

performance.
222

Method
Delta PSNR
Delta SSIM
Delta LPIPS

SVD
0.511
0.0094
0.0070
SVD + Coordinates Map Feedback
0.304
0.0036
0.0018
SVD + 3D-aware Feedback
0.009
0.0028
0.0002
Table 3: The absolute differences in performance metrics between GS and SVD generation results..

4.3
Limitations
223

Current models utilize Gaussian splatting as a 3D representation, mapping and rendering coordinates
224

to textures for feedback. Although algorithms for converting Gaussian Splatting to meshe are under
225

development, achieving high quality in converting Gaussian models to general meshes remains
226

challenging. Directly employing a NeRF-based feed-forward model during the training process
227

significantly reduces training speed due to the computational demands of volumetric rendering. Our
228

model currently lacks the ability to generalize to the scene level, a limitation we intend to address in
229

future research.
230

5
Conclusion
231

In this paper, we introduce SC3D, a unified framework for 3D generation from a single image that
232

integrates multi-view image generation and 3D reconstruction through a self-conditioning mechanism.
233

By establishing a cyclic relationship between these two stages, our approach effectively mitigates the
234

data bias encountered in traditional methods. The self-conditioned method with 3D-aware feedback
235

enhances geometric consistency throughout the generation process.
236

Our experiments demonstrate that SC3D not only improves the quality and efficiency of the generation
237

process but also achieves superior geometric consistency and detail in the reconstructed 3D models.
238

By jointly training the multi-view diffusion model and the reconstruction model, SC3D adapts to the
239

inherent biases of each stage, resulting in more robust and accurate outputs.
240

9


---Page Break---
References
241

[1] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt,
242

Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In
243

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–
244

13153, 2023.
245

[2] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan,
246

Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects.
247

Advances in Neural Information Processing Systems, 36, 2024.
248

[3] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang,
249

Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction
250

and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
251

pages 803–814, 2023.
252

[4] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning
253

using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd
254

International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research,
255

pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.
256

[5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural
257

information processing systems, 33:6840–6851, 2020.
258

[6] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
259

Poole.
Score-based generative modeling through stochastic differential equations.
arXiv preprint
260

arXiv:2011.13456, 2020.
261

[7] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-
262

1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on
263

Computer Vision, pages 9298–9309, 2023.
264

[8] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view
265

diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
266

[9] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang.
267

Syncdreamer: Generating multiview-consistent images from a single-view image.
arXiv preprint
268

arXiv:2309.03453, 2023.
269

[10] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta Mahajan, and Kwang Moo Yi. Vivid-1-to-3:
270

Novel view synthesis with video diffusion models. arXiv preprint arXiv:2312.01305, 2023.
271

[11] Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang, Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding
272

Liang, Yu Qiao, Bo Dai, et al. Epidiff: Enhancing multi-view synthesis via localized epipolar-constrained
273

diffusion. arXiv preprint arXiv:2312.06725, 2023.
274

[12] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra,
275

Yasutaka Furukawa, and Rakesh Ranjan. Mvdiffusion++: A dense high-resolution multi-view diffusion
276

model for single or sparse-view 3d object reconstruction. arXiv preprint arXiv:2402.12712, 2024.
277

[13] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian
278

Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a
279

single image using latent video diffusion, 2024.
280

[14] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai
281

Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain
282

diffusion. arXiv preprint arXiv:2310.15008, 2023.
283

[15] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli,
284

Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint
285

arXiv:2311.04400, 2023.
286

[16] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large
287

multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054,
288

2024.
289

[17] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon
290

Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv
291

preprint arXiv:2403.14621, 2024.
292

10


---Page Break---
[18] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang
293

Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv
294

preprint arXiv:2403.05034, 2024.
295

[19] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient
296

3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint
297

arXiv:2404.07191, 2024.
298

[20] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming
299

Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In
300

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161,
301

2023.
302

[21] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli,
303

Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large
304

reconstruction model. arXiv preprint arXiv:2311.06214, 2023.
305

[22] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and
306

Filippos Kokkinos. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation.
307

arXiv preprint arXiv:2402.08682, 2024.
308

[23] Qi Zuo, Xiaodong Gu, Lingteng Qiu, Yuan Dong, Zhengyi Zhao, Weihao Yuan, Rui Peng, Siyu Zhu, Zilong
309

Dong, Liefeng Bo, et al. Videomv: Consistent multi-view generation based on large video generative
310

model. arXiv preprint arXiv:2403.12010, 2024.
311

[24] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann,
312

Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d
313

scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages
314

2553–2560. IEEE, 2022.
315

[25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution
316

image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer
317

vision and pattern recognition, pages 10684–10695, 2022.
318

[26] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
319

Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-
320

image diffusion models with deep language understanding. Advances in neural information processing
321

systems, 35:36479–36494, 2022.
322

[27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna,
323

and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv
324

preprint arXiv:2307.01952, 2023.
325

[28] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rom-
326

bach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint
327

arXiv:2403.12015, 2024.
328

[29] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet.
329

Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
330

[30] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P
331

Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video
332

generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
333

[31] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang,
334

Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv
335

preprint arXiv:2209.14792, 2022.
336

[32] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz,
337

Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video
338

diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
339

[33] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao.
340

Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
341

[34] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor,
342

Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as
343

world simulators. 2024.
344

11


---Page Break---
[35] Chuanxia Zheng and Andrea Vedaldi. Free3d: Consistent novel view synthesis without 3d representation.
345

arXiv preprint arXiv:2312.04551, 2023.
346

[36] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. Consistnet: Enforcing 3d consistency
347

for multi-view images diffusion. arXiv preprint arXiv:2310.10343, 2023.
348

[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren
349

Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM,
350

65(1):99–106, 2021.
351

[38] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P
352

Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings
353

of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864, 2021.
354

[39] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives
355

with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022.
356

[40] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for
357

real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4):1–14, 2023.
358

[41] Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang. Leap: Liberate sparse-view 3d modeling from
359

camera poses. arXiv preprint arXiv:2310.01410, 2023.
360

[42] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang.
361

Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers.
362

arXiv preprint arXiv:2312.09147, 2023.
363

[43] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion.
364

arXiv preprint arXiv:2209.14988, 2022.
365

[44] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis,
366

Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In
367

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309,
368

2023.
369

[45] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen,
370

Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d
371

content generation. https://github.com/threestudio-project/threestudio, 2023.
372

[46] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang.
Neus:
373

Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint
374

arXiv:2106.10689, 2021.
375

[47] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-
376

1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on
377

Computer Vision, pages 9298–9309, 2023.
378

[48] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion
379

for consistent text-to-3d. arXiv preprint arXiv:2310.02596, 2023.
380

[49] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion
381

models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.
382

[50] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-
383

adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In
384

Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296–4304, 2024.
385

[51] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong
386

Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for
387

detail richness in text-to-3d. arXiv preprint arXiv:2311.16918, 2023.
388

[52] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv
389

preprint arXiv:2312.02201, 2023.
390

[53] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based
391

generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.
392

[54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion
393

models.
394

12


---Page Break---
[55] Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel
395

Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel
396

convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern
397

recognition, pages 1874–1883, 2016.
398

[56] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
399

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
400

[57] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations
401

enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM
402

SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.
403

A
Technical Details
404

A.1
Video model finetuning
405

Based on the approach outlined in [32], the generation process employs the EDM framework[53].
406

Let pdata(x0) represent the video data distribution, and p(x; σ) be the distribution obtained by adding
407

Gaussian noise with variance σ2 to the data. For sufficiently large σmax, p(x; σ2
max) approximates
408

a normal distribution N(0, σ2
max). Diffusion models (DMs) leverage this property and begin with
409

high variance Gaussian noise, xM ∼N(0, σ2
max), and then iteratively denoise the data until reaching
410

σ0 = 0.
411

In practice, this iterative refinement process can be implemented through the numerical simulation of
412

the Probability Flow ordinary differential equation (ODE):
413

dx = −˙σ(t)σ(t)∇x log p(x; σ(t)) dt
(3)

where ∇x log p((x; σ) is called as score function.
414

DM training is to learn a model sθ(x; σ) to approximate the score function ∇x log p((x; σ). The
415

model can be parameterized as:
416

∇x log p((x; σ) ≈sθ((x; σ) = Dθ(x; σ) −x

σ2
,
(4)

where Dθ is a learnable denoiser that aims to predict ground truth x0.
417

The denoiser Dθ is trained via denoising score matching (DSM):
418

Ex0∼pdata(x0),(σ,n)∼p(σ,n)

λσ∥Dθ(x0 + n; σ) −x0∥2
2

,
(5)

where p(σ, n) = p(σ)N(n; 0, σ2), p(σ) is a distribution over noise levels σ, λσ is a weighting
419

function. The learnable denoiser Dθ is parameterized as:
420

Dθ(x; σ) = cskip(σ)x + cout(σ)Fθ(cin(σ)x; cnoise(σ)),
(6)

where Fθ is the network to be trained.
421

We sample log σ ∼N(Pmean, P 2
std), with Pmean = 1.0 and Pstd = 1.6. Then we obtain all the
422

parameters as follows:
423

cin =
1
√

σ2 + 1
(7)

424

cout =
−σ
√

σ2 + 1
(8)

425

cskip(σ) =
1
σ2 + 1
(9)

426

cnoise(σ) = 0.25 log σ
(10)
427

λ(σ) = 1 + σ2

σ2
(11)

We fine-tune the network backbone Fθ on multi-view images of size 512 × 512. During training, for
428

each instance in the dataset, we uniformly sample 8 views and choose the first view as the input view.
429

view images of size 512 × 512.
430

13


---Page Break---
Figure 7: The projection process of coordinates map.

A.2
Coordinates Map
431

In conditional control models such as ControlNet[54], T2IAdapter, when depth maps are used as
432

input, their range needs to be normalized to the [0, 1] interval, typically using the formula: (p −
433

pmean)/(pmax −pmin). However, this normalization process may introduce scale ambiguity, which
434

can affect the multi-view generation performance. To avoid the issues caused by normalization, we use
435

coordinate maps. Coordinate maps transform the depth value d to a common world coordinate system
436

using the camera’s intrinsic and extrinsic parameters, represented as (X, Y, Z). The transformation
437

formula is:
438

 X
Y
Z

!

= K−1 ·

 u
v
1

!

· d

where (u, v) are the pixel coordinates, d is the corresponding depth value, and K is the camera
439

intrinsic matrix.
440

A.3
3D Feedback
441

Figure 8: Architecture of the residual block
used in feedback stage.

Input
inp ∈R3×512×512

PixelUnshuffle [55]
192 × 64 × 64
ResBlock ×3
320 × 64 × 64
ResBlock ×3
640 × 32 × 32
ResBlock ×3
1280 × 16 × 16
ResBlock ×3
1280 × 8 × 8
Table 4: The detailed structure of all layers in
the feedback injection network.

442

With reference to Section 3.3 in the main paper, Fig. 8 and Table 4 provide a detailed illustration of
443

the feedback injection netwrok. We use two networks to inject the coordinates map and RGB texture
444

map feedback into the score function. Each network consists of four feature extraction blocks and
445

three downsample blocks to adjust the feature resolution. The reconstruction coordinates map and
446

14


---Page Break---
RGB texture map initially have a resolution of 512 × 512. We employ the pixel unshuffle operation
447

to downsample these maps to 64 × 64.
448

At each scale, three residual blocks[56] are used to extract the multi-scale feedback features,
449

denoted as FP
= {F 1
p , F 2
p , F 3
p , F 4
p } and FT
= {F 1
t , F 2
t , F 3
t , F 4
t } for the coordinates map
450

and RGB texture map, respectively. These feedback features match the intermediate features
451

Fenc = {F 1
enc, F 2
enc, F 3
enc, F 4
enc} in the encoder of the UNet denoiser. The feedback features FP
452

and FT are added to the intermediate features Fenc at each scale as described by the following
453

equations:
454

Fp = F0(P)
(12)
455

Ft = F1(T)
(13)
456

Fi
enc = Fi
enc + Fi
p + Fi
t,
i ∈{1, 2, 3, 4}
(14)

where P represents the coordinates map feedback input, and T represents the RGB texture feedback
457

input. F0 and F1 denote the functions of the feedback inject network applied to the coordinates map
458

and RGB texture map, respectively.
459

B
Training Details and Experimental Settings
460

Implementation As illustrate in Table 5, all models are trained for 30,000 iterations using 8 A100
461

GPUs with a total batch size of 32. We clip the gradient with a maximum norm of 1.0. We use
462

the AdamW optimizer with a learning rate of 1 × 10−5 and employ FP16 mixed precision with
463

DeepSeed[57] with Zero-2 for efficient training. We adjust the cameras in each batch so that the
464

initial input view consistently represents the reference frame, using an identity rotation matrix and a
465

fixed translation for alignment.
466

The inference settings are shown in Table 6.
467

Hyperparameter
SVD (1.8 B)
LGM (424M)

Training
Optimizer
AdamW
AdamW
Learning rate
1e-5
1e-5
Batch size per GPU
4
4
# training steps
40k
40k
# GPUs
8
8
Training time (days)
4
4
Input Resolution
8 × 512 × 512 × 3
4 × 256 × 256 × 3
Output Resolution
8 × 512 × 512 × 3
−× 512 × 512 × 3

Diffusion setup
Pmean
1.0
-
Pstd
1.6
-
Table 5: Hyperparameters for the training stage.

Hyperparameter
SC3D
VideoMV
SV3D
SyncDreamer

Sampling parameters
Sampler
Euler
DDIM
Euler
DDIM
steps
25
50
50
50
cfg gudiance
1.0 ∼3.0
6.0
6.0
2.0
Table 6: Hyperparameters for the inference stage.

C
Additional Visualization Results
468

15


---Page Break---
Figure 9: Visualization results generated by our SC3D. For each sample (3 rows), the 1st row is
ground truth, 2nd row is the generated multi-view images, while 3rd row is the rendered views from
reconstructed 3DGS. For each row, the first image is the input image.

16


---Page Break---
NeurIPS Paper Checklist
469

1. Claims
470

Question: Do the main claims made in the abstract and introduction accurately reflect the
471

paper’s contributions and scope?
472

Answer: [Yes]
473

Justification: The abstract and introduction clearly outline the primary contributions of the
474

paper. The claims made are directly supported by the experiments presented in the paper,
475

ensuring an accurate representation of the work’s contributions and limitations.
476

Guidelines:
477

• The answer NA means that the abstract and introduction do not include the claims
478

made in the paper.
479

• The abstract and/or introduction should clearly state the claims made, including the
480

contributions made in the paper and important assumptions and limitations. A No or
481

NA answer to this question will not be perceived well by the reviewers.
482

• The claims made should match theoretical and experimental results, and reflect how
483

much the results can be expected to generalize to other settings.
484

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
485

are not attained by the paper.
486

2. Limitations
487

Question: Does the paper discuss the limitations of the work performed by the authors?
488

Answer: [Yes]
489

Justification: See in Section 4.3.
490

Guidelines:
491

• The answer NA means that the paper has no limitation while the answer No means that
492

the paper has limitations, but those are not discussed in the paper.
493

• The authors are encouraged to create a separate "Limitations" section in their paper.
494

• The paper should point out any strong assumptions and how robust the results are to
495

violations of these assumptions (e.g., independence assumptions, noiseless settings,
496

model well-specification, asymptotic approximations only holding locally). The authors
497

should reflect on how these assumptions might be violated in practice and what the
498

implications would be.
499

• The authors should reflect on the scope of the claims made, e.g., if the approach was
500

only tested on a few datasets or with a few runs. In general, empirical results often
501

depend on implicit assumptions, which should be articulated.
502

• The authors should reflect on the factors that influence the performance of the approach.
503

For example, a facial recognition algorithm may perform poorly when image resolution
504

is low or images are taken in low lighting. Or a speech-to-text system might not be
505

used reliably to provide closed captions for online lectures because it fails to handle
506

technical jargon.
507

• The authors should discuss the computational efficiency of the proposed algorithms
508

and how they scale with dataset size.
509

• If applicable, the authors should discuss possible limitations of their approach to
510

address problems of privacy and fairness.
511

• While the authors might fear that complete honesty about limitations might be used by
512

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
513

limitations that aren’t acknowledged in the paper. The authors should use their best
514

judgment and recognize that individual actions in favor of transparency play an impor-
515

tant role in developing norms that preserve the integrity of the community. Reviewers
516

will be specifically instructed to not penalize honesty concerning limitations.
517

3. Theory Assumptions and Proofs
518

Question: For each theoretical result, does the paper provide the full set of assumptions and
519

a complete (and correct) proof?
520

17


---Page Break---
Answer: [NA] .
521

Justification: The paper does not include theoretical results.
522

Guidelines:
523

• The answer NA means that the paper does not include theoretical results.
524

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
525

referenced.
526

• All assumptions should be clearly stated or referenced in the statement of any theorems.
527

• The proofs can either appear in the main paper or the supplemental material, but if
528

they appear in the supplemental material, the authors are encouraged to provide a short
529

proof sketch to provide intuition.
530

• Inversely, any informal proof provided in the core of the paper should be complemented
531

by formal proofs provided in appendix or supplemental material.
532

• Theorems and Lemmas that the proof relies upon should be properly referenced.
533

4. Experimental Result Reproducibility
534

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
535

perimental results of the paper to the extent that it affects the main claims and/or conclusions
536

of the paper (regardless of whether the code and data are provided or not)?
537

Answer: [Yes]
538

Justification: We provide the GSO generation result and code in the supplemental materials.
539

Guidelines:
540

• The answer NA means that the paper does not include experiments.
541

• If the paper includes experiments, a No answer to this question will not be perceived
542

well by the reviewers: Making the paper reproducible is important, regardless of
543

whether the code and data are provided or not.
544

• If the contribution is a dataset and/or model, the authors should describe the steps taken
545

to make their results reproducible or verifiable.
546

• Depending on the contribution, reproducibility can be accomplished in various ways.
547

For example, if the contribution is a novel architecture, describing the architecture fully
548

might suffice, or if the contribution is a specific model and empirical evaluation, it may
549

be necessary to either make it possible for others to replicate the model with the same
550

dataset, or provide access to the model. In general. releasing code and data is often
551

one good way to accomplish this, but reproducibility can also be provided via detailed
552

instructions for how to replicate the results, access to a hosted model (e.g., in the case
553

of a large language model), releasing of a model checkpoint, or other means that are
554

appropriate to the research performed.
555

• While NeurIPS does not require releasing code, the conference does require all submis-
556

sions to provide some reasonable avenue for reproducibility, which may depend on the
557

nature of the contribution. For example
558

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
559

to reproduce that algorithm.
560

(b) If the contribution is primarily a new model architecture, the paper should describe
561

the architecture clearly and fully.
562

(c) If the contribution is a new model (e.g., a large language model), then there should
563

either be a way to access this model for reproducing the results or a way to reproduce
564

the model (e.g., with an open-source dataset or instructions for how to construct
565

the dataset).
566

(d) We recognize that reproducibility may be tricky in some cases, in which case
567

authors are welcome to describe the particular way they provide for reproducibility.
568

In the case of closed-source models, it may be that access to the model is limited in
569

some way (e.g., to registered users), but it should be possible for other researchers
570

to have some path to reproducing or verifying the results.
571

5. Open access to data and code
572

18


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
573

tions to faithfully reproduce the main experimental results, as described in supplemental
574

material?
575

Answer: [Yes]
576

Justification: We provide the code in the supplemental materials.
577

Guidelines:
578

• The answer NA means that paper does not include experiments requiring code.
579

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
580

public/guides/CodeSubmissionPolicy) for more details.
581

• While we encourage the release of code and data, we understand that this might not be
582

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
583

including code, unless this is central to the contribution (e.g., for a new open-source
584

benchmark).
585

• The instructions should contain the exact command and environment needed to run to
586

reproduce the results. See the NeurIPS code and data submission guidelines (https:
587

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
588

• The authors should provide instructions on data access and preparation, including how
589

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
590

• The authors should provide scripts to reproduce all experimental results for the new
591

proposed method and baselines. If only a subset of experiments are reproducible, they
592

should state which ones are omitted from the script and why.
593

• At submission time, to preserve anonymity, the authors should release anonymized
594

versions (if applicable).
595

• Providing as much information as possible in supplemental material (appended to the
596

paper) is recommended, but including URLs to data and code is permitted.
597

6. Experimental Setting/Details
598

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
599

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
600

results?
601

Answer: [Yes]
602

Justification: See in Appendix B.
603

Guidelines:
604

• The answer NA means that the paper does not include experiments.
605

• The experimental setting should be presented in the core of the paper to a level of detail
606

that is necessary to appreciate the results and make sense of them.
607

• The full details can be provided either with the code, in appendix, or as supplemental
608

material.
609

7. Experiment Statistical Significance
610

Question: Does the paper report error bars suitably and correctly defined or other appropriate
611

information about the statistical significance of the experiments?
612

Answer: [No]
613

Justification: The paper does not provide error bars or any statistical significance measures
614

for the experimental results.
615

Guidelines:
616

• The answer NA means that the paper does not include experiments.
617

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
618

dence intervals, or statistical significance tests, at least for the experiments that support
619

the main claims of the paper.
620

• The factors of variability that the error bars are capturing should be clearly stated (for
621

example, train/test split, initialization, random drawing of some parameter, or overall
622

run with given experimental conditions).
623

19


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
624

call to a library function, bootstrap, etc.)
625

• The assumptions made should be given (e.g., Normally distributed errors).
626

• It should be clear whether the error bar is the standard deviation or the standard error
627

of the mean.
628

• It is OK to report 1-sigma error bars, but one should state it. The authors should
629

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
630

of Normality of errors is not verified.
631

• For asymmetric distributions, the authors should be careful not to show in tables or
632

figures symmetric error bars that would yield results that are out of range (e.g. negative
633

error rates).
634

• If error bars are reported in tables or plots, The authors should explain in the text how
635

they were calculated and reference the corresponding figures or tables in the text.
636

8. Experiments Compute Resources
637

Question: For each experiment, does the paper provide sufficient information on the com-
638

puter resources (type of compute workers, memory, time of execution) needed to reproduce
639

the experiments?
640

Answer: [Yes]
641

Justification: See Appendix B.
642

Guidelines:
643

• The answer NA means that the paper does not include experiments.
644

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
645

or cloud provider, including relevant memory and storage.
646

• The paper should provide the amount of compute required for each of the individual
647

experimental runs as well as estimate the total compute.
648

• The paper should disclose whether the full research project required more compute
649

than the experiments reported in the paper (e.g., preliminary or failed experiments that
650

didn’t make it into the paper).
651

9. Code Of Ethics
652

Question: Does the research conducted in the paper conform, in every respect, with the
653

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
654

Answer: [Yes]
655

Justification: We have reviewed the NeurIPS Code of Ethics.
656

Guidelines:
657

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
658

• If the authors answer No, they should explain the special circumstances that require a
659

deviation from the Code of Ethics.
660

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
661

eration due to laws or regulations in their jurisdiction).
662

10. Broader Impacts
663

Question: Does the paper discuss both potential positive societal impacts and negative
664

societal impacts of the work performed?
665

Answer: [Yes]
666

Justification: In the Section 1, we discuss how 3D generation can accelerate various in-
667

dustries by enhancing design processes, improving simulations, and reducing production
668

costs.
669

Guidelines:
670

• The answer NA means that there is no societal impact of the work performed.
671

• If the authors answer NA or No, they should explain why their work has no societal
672

impact or why the paper does not address societal impact.
673

20


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
674

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
675

(e.g., deployment of technologies that could make decisions that unfairly impact specific
676

groups), privacy considerations, and security considerations.
677

• The conference expects that many papers will be foundational research and not tied
678

to particular applications, let alone deployments. However, if there is a direct path to
679

any negative applications, the authors should point it out. For example, it is legitimate
680

to point out that an improvement in the quality of generative models could be used to
681

generate deepfakes for disinformation. On the other hand, it is not needed to point out
682

that a generic algorithm for optimizing neural networks could enable people to train
683

models that generate Deepfakes faster.
684

• The authors should consider possible harms that could arise when the technology is
685

being used as intended and functioning correctly, harms that could arise when the
686

technology is being used as intended but gives incorrect results, and harms following
687

from (intentional or unintentional) misuse of the technology.
688

• If there are negative societal impacts, the authors could also discuss possible mitigation
689

strategies (e.g., gated release of models, providing defenses in addition to attacks,
690

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
691

feedback over time, improving the efficiency and accessibility of ML).
692

11. Safeguards
693

Question: Does the paper describe safeguards that have been put in place for responsible
694

release of data or models that have a high risk for misuse (e.g., pretrained language models,
695

image generators, or scraped datasets)?
696

Answer: [NA]
697

Justification: The paper does not involve the release of data or models that have a high risk
698

for misuse.
699

Guidelines: The paper focuses on foundational research and does not have direct societal
700

implications. It does not address societal impacts.
701

• The answer NA means that the paper poses no such risks.
702

• Released models that have a high risk for misuse or dual-use should be released with
703

necessary safeguards to allow for controlled use of the model, for example by requiring
704

that users adhere to usage guidelines or restrictions to access the model or implementing
705

safety filters.
706

• Datasets that have been scraped from the Internet could pose safety risks. The authors
707

should describe how they avoided releasing unsafe images.
708

• We recognize that providing effective safeguards is challenging, and many papers do
709

not require this, but we encourage authors to take this into account and make a best
710

faith effort.
711

12. Licenses for existing assets
712

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
713

the paper, properly credited and are the license and terms of use explicitly mentioned and
714

properly respected?
715

Answer: [Yes]
716

Justification: The SVD model (https://huggingface.co/stabilityai/stable-video-diffusion-
717

img2vid) is intended for research purposes only. The following assets are used in the paper,
718

and their licenses are properly acknowledged:
719

• Gobjaverse: https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse
720

• LGM: https://github.com/3DTopia/LGM.git
721

• Syncdreamer: https://github.com/liuyuan-pal/SyncDreamer.git
722

• Objaverse: https://huggingface.co/datasets/allenai/objaverse
723

The use of the Objaverse dataset as a whole is licensed under the ODC-By v1.0 license.
724

Individual objects in Objaverse are licensed under various Creative Commons licenses,
725

including:
726

21


---Page Break---
• CC-BY 4.0 - 721K objects
727

• CC-BY-NC 4.0 - 25K objects
728

• CC-BY-NC-SA 4.0 - 52K objects
729

• CC-BY-SA 4.0 - 16K objects
730

• CC0 1.0 - 3.5K objects
731

Guidelines:
732

• The answer NA means that the paper does not use existing assets.
733

• The authors should cite the original paper that produced the code package or dataset.
734

• The authors should state which version of the asset is used and, if possible, include a
735

URL.
736

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
737

• For scraped data from a particular source (e.g., website), the copyright and terms of
738

service of that source should be provided.
739

• If assets are released, the license, copyright information, and terms of use in the
740

package should be provided. For popular datasets, paperswithcode.com/datasets
741

has curated licenses for some datasets. Their licensing guide can help determine the
742

license of a dataset.
743

• For existing datasets that are re-packaged, both the original license and the license of
744

the derived asset (if it has changed) should be provided.
745

• If this information is not available online, the authors are encouraged to reach out to
746

the asset’s creators.
747

13. New Assets
748

Question: Are new assets introduced in the paper well documented and is the documentation
749

provided alongside the assets?
750

Answer: [Yes]
751

Justification: We provide the code and generation results in supplemental materials.
752

Guidelines:
753

• The answer NA means that the paper does not release new assets.
754

• Researchers should communicate the details of the dataset/code/model as part of their
755

submissions via structured templates. This includes details about training, license,
756

limitations, etc.
757

• The paper should discuss whether and how consent was obtained from people whose
758

asset is used.
759

• At submission time, remember to anonymize your assets (if applicable). You can either
760

create an anonymized URL or include an anonymized zip file.
761

14. Crowdsourcing and Research with Human Subjects
762

Question: For crowdsourcing experiments and research with human subjects, does the paper
763

include the full text of instructions given to participants and screenshots, if applicable, as
764

well as details about compensation (if any)?
765

Answer: [NA]
766

Justification: The paper does not involve crowdsourcing nor research with human subjects.
767

Guidelines:
768

• The answer NA means that the paper does not involve crowdsourcing nor research with
769

human subjects.
770

• Including this information in the supplemental material is fine, but if the main contribu-
771

tion of the paper involves human subjects, then as much detail as possible should be
772

included in the main paper.
773

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
774

or other labor should be paid at least the minimum wage in the country of the data
775

collector.
776

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
777

Subjects
778

22


---Page Break---
Question: Does the paper describe potential risks incurred by study participants, whether
779

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
780

approvals (or an equivalent approval/review based on the requirements of your country or
781

institution) were obtained?
782

Answer: [NA]
783

Justification: The paper does not involve crowdsourcing nor research with human subjects.
784

Guidelines:
785

• The answer NA means that the paper does not involve crowdsourcing nor research with
786

human subjects.
787

• Depending on the country in which research is conducted, IRB approval (or equivalent)
788

may be required for any human subjects research. If you obtained IRB approval, you
789

should clearly state this in the paper.
790

• We recognize that the procedures for this may vary significantly between institutions
791

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
792

guidelines for their institution.
793

• For initial submissions, do not include any information that would break anonymity (if
794

applicable), such as the institution conducting the review.
795

23


---Page Break---
