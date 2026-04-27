SyntheOcc: Synthesize Geometric-Controlled
Street View Images through 3D Semantic MPIs

Anonymous Author(s)
Affiliation
Address
email

Abstract

The advancement of autonomous driving is increasingly reliant on high-quality
1

annotated datasets, especially in the task of 3D occupancy prediction, where the
2

occupancy labels require dense 3D annotation with significant human effort. In
3

this paper, we propose SytheOcc, which denotes a diffusion model that Synthesize
4

photorealistic and geometric-controlled images by conditioning Occupancy labels
5

in driving scenarios. This yields an unlimited amount of diverse, annotated, and
6

controllable datasets for applications like training perception models and simu-
7

lation. SyntheOcc addresses the critical challenge of how to efficiently encode
8

3D geometric information as conditional input to a 2D diffusion model. Our ap-
9

proach innovatively incorporates 3D semantic multi-plane images (MPIs) to pro-
10

vide comprehensive and spatially aligned 3D scene descriptions for conditioning.
11

As a result, SyntheOcc can generate photorealistic multi-view images and videos
12

that faithfully align with the given geometric labels (semantics in 3D voxel space).
13

Extensive qualitative and quantitative evaluations of SyntheOcc on the nuScenes
14

dataset prove its effectiveness in generating controllable occupancy datasets that
15

serve as an effective data augmentation to perception models.
16

1
Introduction
17

With the rapid development of generative models, they have shown realistic image synthesis and
18

diverse controllability. This progress has opened up new avenues for dataset generation in autonomous
19

driving [5, 12, 23, 30]. The task of dataset generation is usually modeled as controllable image
20

generation, where the ground truth (e.g. 3D Box) is employed to control the generation of new datasets
21

in downstream tasks (e.g. 3D detection). This approach helps to mitigate the data collection and
22

annotation effort as it can generate labeled data for free. However, a novel task of vital importance,
23

occupancy prediction [24, 27], poses new challenges for dataset generation compared with 3D
24

detection. It requires finer and more nuanced geometry controllability, which refers to use the
25

occupancy state and semantics of voxels in the whole 3D space to control the image generation.
26

We argue that solving this problem not only allows us to synthesize occupancy datasets, but also
27

empowers valuable applications such as editing geometry to generate rare data for corner case
28

evaluation, as shown in Fig. 1. In the following, we first illustrate why prior work struggles to achieve
29

the above objective, and then demonstrate how we address these challenges.
30

In the area of diffusion models, several representative works have displayed high-quality image
31

synthesis; however, they are constrained by limited 3D controllability: they are incapable of editing 3D
32

voxels for precise control. For example, BEVGen [23] generates street view images by conditioning
33

BEV layouts using diffusion models. MagicDrive [5] extend BEVGen and additionally converts the
34

3D box parameters into text embedding through Fourier mapping that is similar to NeRF [19], and
35

uses cross-attention to learn conditional generation. Although these methods achieve satisfactory
36

results in image generation, their 3D controllability is inherently limited. These approaches are
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Add traffic cone

Testing
End-to-end

Planner

VAD/
UniAD

Corner

Case
Evaluation

Predicted waypoints

Predicted waypoints

Original occupancy

Edited occupancy
Generated image

Generated image

Geometric
Controlled
Generation

SyntheOcc

User Editing 

To Create
Corner Case

Figure 1: A showcase of application of SytheOcc. We enable geometric-controlled generation that
conveys the user editing in 3D voxel space to generate realistic street view images. In this case, we
create a rare scene that traffic cones block the way. This advancement facilitates the evaluation of
autonomous systems, such as the end-to-end planner VAD [9], in simulated corner case scenes.

restricted to manipulating the scene in types of 3D boxes and BEV layouts, and hardly adapt to finer
38

geometry control such as editing the shape of objects and scenes. Meanwhile, they usually convert
39

conditional input into 1D embedding that aligns with prompt embedding, which is less effective in
40

3D-aware generation due to lack of spatial alignment with the generated images. This limitation
41

hinders their utility in downstream applications, such as occupancy prediction and editing scene
42

geometry to create long-tailed scenes, where granular volumetric control is paramount in both tasks.
43

ControlNet [41] and GLIGEN [14] is another type of prominent method in the field of controllable
44

image generation. These approaches exhibit several desirable attributes in terms of controllability.
45

They leverage conditional images such as semantic masks for control, thereby offering a unified
46

framework to manipulate both foreground and background. However, despite its precise spatial
47

control, ControlNet does not align with our specific requirements. Their conditions of pixel-level
48

images differ fundamentally from what we require in 3D contexts. Our experimental results also find
49

that ControlNet struggles to handle overlapping objects with varying depths (see Fig. 6 (a)), as it only
50

utilizes an ambiguous 2D semantic map as conditional input. As a result, it is non-trivial to extend
51

the ControlNet framework and convey their desirable attributes for 3D conditioning.
52

To address the above challenges, we propose an innovative representation, 3D semantic multi-plane
53

images (MPIs), which contribute to image generation with finer geometric control. In detail, we
54

employ multi-plane images [43] to represent the occupancy, where each plane represents a slice of
55

semantic label at a specific depth. Our 3D semantic MPIs not only preserve accurate and authentic 3D
56

information, but also keep pixel-wise alignment with the generated images. We additionally introduce
57

the MPI encoder to encode features, and the reweighing methods to ease the training with long-tailed
58

cases. As a collection, our framework enables 3D geometry and semantic control for image generation
59

and further facilitates corner case evaluation as depicted in Fig. 1. Finally, experimental results
60

demonstrate that our synthetic data achieve better recognizability, and are effective in improving the
61

perception model on occupancy prediction. In summary, our contributions include:
62

• We present SytheOcc, a novel image generation framework to attain finer and precise 3D
63

geometric control, thereby unlocking a spectrum of applications such as 3D editing, dataset
64

generation, and long-tailed scene generation.
65

• Incorporating the proposed 3D semantic MPI, MPI encoder, and reweighing strategy, we
66

deliver a substantial advancement in image quality and recognizability over prior works.
67

• Our extensive experimental results demonstrate that our synthetic data yields an effective
68

data augmentation in the realm of 3D occupancy prediction.
69

2
Related Work
70

2.1
3D Occupancy Prediction
71

The task of 3D occupancy prediction aims to predict the occupancy status of each voxel in 3D space,
72

as well as its semantic label if occupied. Compared with previous perception methods like 3D object
73

2


---Page Break---
detection, occupancy prediction offers a more detailed and nuanced understanding of the environment,
74

as it provides finer geometric details, is capable of handling general, out-of-vocabulary objects, and
75

finally, enriches the planning stack with comprehensive 3D information. Early methods exploited
76

LiDAR as inputs to complete the 3D occupancy of the entire 3D scene [18, 33]. Recent methods
77

began to explore the more challenging vision-based 3D occupancy prediction [24,25,27,29]. By
78

predicting the geometric and semantic properties of both dynamic and static elements, 3D occupancy
79

prediction offers a more comprehensive understanding of the surrounding environment.
80

2.2
Diffusion-based Image Generation
81

Recent advancements in diffusion models (DMs) have achieved remarkable progress in image
82

generation. In particular, Stable Diffusion (SD) [21] employs DMs within the latent space of
83

autoencoders, striking a balance between computational efficiency and high image quality. Beyond
84

text control, there is also the introduction of additional control signals. A noteworthy work is
85

ControlNet [41], which incorporates a trainable copy of the SD encoder to extract the feature of
86

conditional images and adds it to the UNet feature. It significantly enhances the controllability and
87

unlocking pathways for advanced applications. We refer readers to recent survey [35] for more details.
88

2.3
Image Generation in Autonomous Driving
89

As training neural networks relies heavily on labeled data, numerous studies are delving into dataset
90

generation to boost training. Lift3D [12] designs generative NeRF to synthesize labeled datasets
91

for 3D detection for the first time. Several other works employ BEV layouts to synthesize image
92

data, proving beneficial for perception models. For example, BEVGen [23] conditions BEV layouts
93

to generate multi-view street images, while BEVControl [34] separately generates foregrounds and
94

backgrounds from BEV layouts. MagicDrive [5] generates images with 3D geometry controls by
95

independently encoding objects and maps through a text encoder or map encoder. Compared with
96

MagicDrive, our geometry control is characterized by a more detailed and lossless representation of
97

3D scenes for control, which poses significant challenges than projected layout or box embedding.
98

Recently, DriveDreamer [26], DrivingDiffusion [13], Drive-WM [28] and Panacea [30] use a Con-
99

trolNet framework, which involves projecting bounding boxes and road maps onto 2D FoV images as
100

a conditioning input. This approach has proven to be effective for geometric control. However, it is
101

limited in that it only achieves alignment at the 2D-pixel level. Consequently, this method falls short
102

in capturing the depth hierarchy and fails to account for the occlusion relationships present in the 3D
103

real world. Besides, adding a depth channel like Panacea [30] may address the limitations of depth
104

order, but it discards the occluded part and only contains partial observation. UrbanGiraffe [37] train
105

a generative NeRF to perform image generation. WoVoGen [17] creates a 4D world volume feature
106

using occupancy to guide the generation, but seems to rely on object mask guidance.
107

As described above, most of the prior work is restricted by only modeling a projected primitive of 3D
108

boxes and road maps as conditions. They suffer from ill-posed un-projection ambiguity. In contrast,
109

we model 3D occupancy labels as conditions, as they provide finer geometric details and semantic
110

information. However, designing an input representation of 3D occupancy labels into a 2D diffusion
111

model is challenging. In this paper, we propose a novel representation: 3D semantic Multi-Plane
112

Images (MPIs) as conditional inputs, which not only provide spatial alignment that improves visual
113

consistency, but also encode comprehensive 3D geometric information including occluded parts.
114

3
Method
115

Overview
The overview of our method is depicted in Fig. 2. Built upon the SD pipeline, we
116

aim to perform geometry-controlled image generation by conditioning on 3D geometry labels with
117

semantics (occupancy labels). One requirement is that the images should faithfully align with the
118

given label. This task is more challenging than conditioned on 3D box due to the sparse and irregular
119

nature of occupancy. We first discuss how to efficiently represent occupancy in Sec. 3.2, followed
120

by our designed MPI encoder to enhance generation quality in Sec. 3.3, and reweighing strategy to
121

handle the long-tailed depth and category in Sec. 3.5.
122

3.1
Representation of Condition: Local Control Aligns Better than Global Control
123

One of the key challenges is how to represent our conditional occupancy input. A straightforward
124

method [3,5] is to convert the 3D occupancy voxel to 1D global embedding that is similar to text
125

embedding, and then use cross-attention to learn controllable generation. However, these global
126

methods can be less effective when dealing with dense or irregular data due to the following reasons:
127

3


---Page Break---
3D Semantic 
Multiplane Images

Prompt: show a realistic
street view image

N×D×H×W
Occupancy Label

…

Latent Noise

N×4×H×W
…

VAE 
Decoder

Conv1×1

Conv1×1

ReLU

UNet Encoder

MPI Encoder

Intra
View
Diffusion

UNet

Cross
View/
Frame

Training 
Perception 
Model

Corner 
Case 
Evaluation

Long-tailed
Scene 
Generation
Optional 
User Edit:

Add
Delete

….

Applications

Multi-view images

……

Simulation

Figure 2: The overall architecture of SytheOcc. We achieve 3D geometric control in image generation
by utilizing our proposed 3D semantic multiplane images to encode scene occupancy. In our
framework, we can edit the occupied state and semantics of every voxel in 3D space to control the
image generation, thereby opening up a wide spectrum of applications as shown in the top right.

(i) They perform controllable generation through hard encoding the spatial relationship between 1D
128

global embedding and 2D UNet features. (ii) Ignore the underlying geometry alignment between the
129

conditional input and the generated image. In contrast, local methods like ControlNet, directly add
130

spatial features to the UNet features, providing 2D local control with pixel-level spatial alignment.
131

They are better than the global method (see Tab. 1), but suffer from 3D ambiguity (see Fig. 6 (a)).
132

Consequently, this comparison motivates us to seek a more compact and efficient manner to encode
133

and condition our 3D occupancy labels.
134

3.2
Represent Occupancy as 3D Semantic Multiplane Images
135

It is non-trivial to design a 3D representation for conditioning. To efficiently store both the semantic
136

and geometric information of the irregular occupancy input, we propose to use multiplane images
137

(MPIs) [43] as representation. An MPI is composed of a series of fronto-parallel RGBA layers within
138

the frustum of the source camera with a specific viewpoint. These planes are arranged at varying
139

depths, from dmin to dmax, starting from the nearest to the farthest. Each layer of these images
140

contains both an RGB image and an alpha map, which collectively capture the visual and geometric
141

details of the scene at the respective depth. In our work, instead of storing RGB value and alpha map
142

in the original MPI, we store our 3D semantic labels. Each layer of MPI represents the semantic
143

index at the corresponding depth. We display the colored MPI in the top row of Fig. 2 for visual
144

clarity, but we actually use the integer index for learning. We obtain our 3D semantic MPI by:
145

Pl = (u × dl, v × dl, dl)T , dl = dmin + (dmax −dmin) × l/D,
(1)

MPIn,l = Interpolate(Occupancy, Tn · K−1
n
· Pl),
(2)
MPI = Concatenate(MPIi,j), i ∈(0, N), j ∈(0, D),
(3)

where (u, v) is a pixel coordinate in image space, dl is depth value of the lth layer, n denotes the nth
146

camera view. This equation implies we first back project points P in camera frustum space (u, v, d)
147

to Euclid space (x, y, z) by multiplying inverse intrinsic K−1. Then we use transformation matrix T
148

to map points from camera coordinates to occupancy coordinates. We then use the point coordinates
149

to interpolate the nearest semantic index from the dense occupancy voxel to form a slice of MPI.
150

Finally, we concatenate all slices to form MPI ∈RN×D×H×W , where D is the number of layers that
151

is set at 256, N is the number of camera views in the case of batch size = 1.
152

By representing occupancy as 3D semantic MPI, every pixel in MPI contains geometry and semantic
153

information with implicit depth, seamlessly integrating occluded elements, and ensuring a precise
154

spatial alignment with the generated images.
155

3.3
3D Semantic MPI Encoder
156

To enable local control with spatially aligned conditions, we develop a simple but effective MPI
157

encoder that aligns the 3D multi-plane feature to the latent space of the diffusion model. The
158

purpose of the MPI encoder is to obtain features from multi-plane images to perform 3D-aware
159

4


---Page Break---
(c) Editing: Foreground removal
(b) Editing: Object manipulation (copy object)
(a) Raw generation

Figure 3: Visualizations of geometric controlled generation. Top row: Fusion of 3D semantic MPI.
Bottom row: our generation concatenated from neighboring views.

image synthesis. Unlike the original ControlNet which downsampling conditional input through 3×3
160

convolutions with padding, we design a 1×1 convolutional encoder without downsampling to encode
161

features. In detail, the 3D multiplane features which have the sample resolution with latent features,
162

are transformed by a 1×1 convolution layer and ReLU activation [1] in the MPI encoder.
163

After obtaining the multi-scale feature after the MPI encoder, we add the feature to the decoder of
164

diffusion UNet to provide spatial features. Experimental results in Tab. 3 will show that our 1×1 conv
165

in MPI encoder is more effective than 3×3 conv, as the 1×1 conv with receptive field = 1 provides a
166

spatial align feature to the latent feature in the diffusion UNet. In contrast, 3×3 conv is conducted
167

in a camera frustum space rather than Euclid space, making an imprecise correspondence between
168

3D multiplane features and 2D image features. Moreover, using 3×3 conv to process 3D semantic
169

MPI will introduce a large computational burden as the channel number increases from 3 channels of
170

RGB to 256 planes. We display our 3D geometry and semantic control property in Fig. 3.
171

In summary, we chose MPIs as the representation because they (i) Incorporate lossless 3D information,
172

including scene geometry rather than 2.5D depth. (ii) Provide spatially aligned conditional features
173

that naturally extend the ControlNet framework from image level to 3D level. (iii) Capable of
174

representing geometry and semantics including occluded elements.
175

3.4
Cross-View and Cross-Frame Attention
176

The sensor arrangement in a self-driving car usually requires a full surround view of cameras to capture
177

the entire 360-degree environment. To effectively simulate the multi-view and subsequent multi-frame
178

generation, zero-initialized [41] cross-view and cross-frame attention are integrated into the diffusion
179

model to maintain consistency between views and frames. Following prior work [5, 28, 30, 31],
180

each cross-view attention allows the target view to access information from its neighboring left and
181

right views, thus training cross-view attention using multi-view consistent images will enforce it to
182

generate the same instance in the overlapping region of multi-view cameras.
183

Attention(Q, K, V ) = softmax( QKT

√

d ) · V ,
(4)

hout = hin + P

i∈{l,r}Attention(Qin, Ki, Vi),
(5)

where l, and r is the camera view of left and right. Qin and hin denotes the query and the hidden
184

state of input view. Similarly, we add cross-frame attention that attend previous frame and future
185

frame to enable video generation. In this case, we use the same formulation while i ∈{f, h}, where
186

f and h is the camera view of future and history frames.
187

3.5
Importance Reweighing
188

0
200
400
600
800
1000

1.00

1.25

1.50

1.75

2.00

2.25

2.50

2.75

3.00
m = 3, n = 700
m = 3, n = 1000
m = 2, n = 1000

Figure 4: Visualizations of the reweighing
function in Eq. 6.

To deal with the extreme imbalance problem between
189

foreground, background, and object categories, and
190

also to ease the training, we propose three types of
191

reweighting methods to improve the generation quality
192

of foreground objects.
193

Progressive Foreground Enhancement
To miti-
194

gate the complexity of the learning task, we propose a
195

progressive reweighting method that incrementally en-
196

hances the loss associated with the foreground regions
197

(based on semantic class) as the training progresses.
198

The detailed formulation is:
199

w(x, m, n) = (m −1)

2
·(1+cos( x
n ·π+π))+1, (6)

5


---Page Break---
(a) Top: Fusion of 3D semantic MPI. Bottom: GT

(b) Generation1: Ordinary scenes. Red rectangle denotes geometry alignment of trees

(c) Generation2, Weather variation: Snow (top) and Sandstorm (bottom) 

(d) Generation3, Style control: Minecraft style (top) and Diablo style (bottom) 

CAM_FRONT_LEFT
CAM_FRONT
CAM_FRONT_RIGHT
CAM_BACK_RIGHT
CAM_BACK
CAM_BACK_LEFT

Figure 5: Visualizations of generated multi-view images. The generation conditions (occupancy
labels) are from nuScenes validation set. We highlight that (i) Geometry alignment of trees in red
rectangle in (b). (ii) Use text prompt to control high-level appearance in (c,d).

where x is the current training step, m is the maximum value of weights that set at 2, and n is the
200

total training steps. This approach is engineered to facilitate a learning trajectory that progresses
201

from simplicity to complexity, thereby aiding in the convergence of the model. This curve can be
202

interpreted as a cosine annealing but inverted to amplify the importance of the foreground region.
203

Depth-aware Foreground Reweighing
In the meantime, we acknowledge the learning difficulty
204

in different depth places in 3D scenes. Following GeoDiffusion [3], we perform depth reweighing to
205

foreground objects by adaptively assigning higher weights to farther foreground areas. This enables
206

the model to focus more thoroughly on hard examples with depth-aware importance reweighting.
207

Instead of using their exponential function to increase weights, we use our designed cosine function
208

Eq. 6 for stability. Here x is the input depth value, and n is the maximum depth that set at 50.
209

CBGS Sampling
To deal with the class imbalance problem in driving scenarios, where cer-
210

tain object categories appear infrequently, we employ the Class-Balanced Grouping and Sampling
211

(CBGS) [44] to better handle the long-tailed classes. CBGS addresses the challenge of class imbal-
212

ance by grouping and re-sampling training data to ensure each group has a balanced distribution of
213

sample frequency across different object categories. This method reduces the bias towards more
214

frequent classes and enables better generalization to rare scenarios.
215

3.6
Model Training
216

To ease the training of the MPI encoder and added attention module, we use a two stage training
217

pipeline. We first train MPI encoder and cross-view attention in a multi-view image generation setting.
218

Then we train cross-frame attention and freeze other components in a video generation setting.
219

6


---Page Break---
Method
Train
Val mIoU

■barrier

■bicycle

■bus

■car

■cons. veh.

■moto.

■pedes.

■traf. cone

■trailer

■truck

■drive. suf.

■other flat

■sidewalk

■terrain

■manmade

■vegetation

Oracle (FB-Occ [15])
Real
Real 39.3 45.4 28.2 44.1 49.4 25.9 28.8 28.0 27.7 32.4 37.3 80.4 42.2 49.9 55.2 42.0 37.7
SytheOcc-Aug
Real+Gen Real 40.3 45.4 27.2 46.6 49.5 26.4 27.8 28.4 29.4 34.0 37.2 81.3 46.0 52.4 56.5 43.3 38.9

MagicDrive
Real
Gen 13.4
0.7 0.0 11.8 32.4 0.0 6.6 2.8 0.3 2.6 19.6 60.1 12.1 26.2 23.4 15.5 12.8
ControlNet
Real
Gen 17.3 17.7 0.2 13.6 21.0 0.6 0.8 8.6 10.4 6.9 11.9 67.4 18.8 36.4 36.9 20.8 22.4
ControlNet+depth
Real
Gen 17.5 19.3 0.3 14.0 23.7 1.0 0.6 9.2 9.2 5.7 12.1 68.8 19.2 36.0 35.3 19.8 22.8
SytheOcc-Gen
Real
Gen 25.5 32.6 13.8 27.7 33.4 7.5 6.5 15.7 16.5 16.5 25.6 74.3 24.5 39.4 40.5 28.6 28.8

Table 1: Downstream evaluation on the nuScenes-Occupancy validation set. Based on the used train
and val data, two types of settings are reported. The first is to use generated training set to augment the
real training set, and evaluate on the real validation set, denoted as Aug. The second is to use pretrained
models trained on the real training datasets to test on the generated validation set, denoted as Gen.

Objective Function
Our final objective function can be formulated as a standard denoising
220

objective with reweighing:
221

L = EE(x),ϵ,t∥ϵ −ϵθ(zt, t, τθ(y))∥2 ⊙w,
(7)
where w is the multiplication of progressive reweighing and depth-aware reweighing.
222

4
Experiments
223

4.1
Dataset and Setups
224

We conduct our experiments on the nuScenes dataset [2], which is collected using 6 surrounded-view
225

cameras that cover the full 360° field of view around the ego-vehicle. It contains 700 scenes for
226

training and 150 scenes for validation. We resize the original image from 1600 × 900 to 800 × 448 for
227

training. In our work, we use the occupancy label with a resolution of 0.2m from OpenOccupancy [27]
228

as condition input, while the benchmark of occupancy prediction uses a resolution of 0.4m from
229

Occ3D [24] dataset for its popularity.
230

Networks
We use Stable Diffusion [21] v2.1 checkpoint as initialization and only train occupancy
231

encoder, cross-view attention. We additionally add cross-frame attention if in video experiments. We
232

adopt FB-Occ [15] as the target model for occupancy prediction for its SOTA performance in this task.
233

The pretrained checkpoint of the network is obtained from their official repository. Since FB-Occ
234

predicts occupancy using only single frame images, we thus train SyntheOcc without cross-frame
235

attention in related experiments. For video generation, we provide experimental results in appendix.
236

Metrics
We use Frechet Inception Distance (FID) [6] to measure the perceptual quality of generated
237

images, and use mIoU to measure the precision of occupancy prediction.
238

Hyperparameters
We set D = 256, dmin = 0 and dmax = 50. The depth resolution of MPI is
239

thus higher than occupancy voxel. We train our model in 6 epochs with batch size = 8. The learning
240

rate is set at 2e−5. The training phase takes around 1 day using 8 NVIDIA A100 80G GPUs. We use
241

UniPC scheduler [42] with the classifier-free guidance (CFG) [7] that is set as 7.0. During inference,
242

we use 20 denoising steps for dataset generation.
243

Baselines
We compare our method with prior methods in Tab. 1. ControlNet denotes we train
244

a ControlNet using an RGB semantic mask as the condition. ControlNet+depth denotes we add a
245

depth channel after the semantic mask to provide 2.5D depth information. The depth map rendered
246

by occupancy is normalized to [0-255] to accommodate the RGB value. The ControlNet+depth can
247

be regarded as a degradation of SytheOcc which is reduced to a single plane. Then we evaluate
248

MagicDrive since it is the only open-sourced method in this area. MagicDrive separately encodes
249

foreground and background using prompt and BEV layout. Furthermore, we evaluate the image
250

quality (FID [6]) of our method in Tab. 2. Compared with prior methods, we use a unified 3D
251

representation that seamlessly handles foreground and background, surpassing them by a large margin.
252

4.2
Qualitative Results
253

High-level Control using Prompt
In Fig. 5 (c,d) and Fig. 6 (c), we demonstrate the capability
254

to employ user-defined prompts to generate images with specific weather conditions and high-level
255

7


---Page Break---
(a0) Fusion of MPI
(a1) Occupancy in 3D space
(a4) Ground truth
(a3) Ours generation
(a2) ControlNet generation

(b) A scene with human

(d) A scene with hinged-articulated trucks

(c) A scene of seasonal changes use prompt control

(e) A scene with excavator use geometry control

Figure 6: Top row: Comparison with ControlNet. We achieve a precise alignment between condi-
tional labels and synthesized images, while ControlNet generates objects with incorrect pose due
to ambiguous 2D condition. Mid and Bottom row: Visualizations of geometry-controlled image
generation. We can faithfully generate objects with the desired topology in a specific 3D position.

style. Although the nuScenes dataset doesn’t contain rare weather images like snow and sandstorms,
256

our method successfully conveys prior knowledge pretrained from stable diffusion to our scenes.
257

Compared with visualization results in prior work like Fig. 8 of MagicDrive, our method shows better
258

alignment with the text prompt, demonstrating the cross-domain generalization ability of our method.
259

3D Geometric Control
Our flexible framework enables us to create novel scenes by manipulating
260

voxels as displayed in Fig. 1 and Fig. 3. Basically, we can edit the occupied state and semantics of
261

every voxel in our scenes for generation. We highlight that we can create a hinged-articulated truck
262

and an excavator as shown in Fig. 6 (d,e). The generated excavator image exhibits a remarkable
263

alignment with the input occupancy that is delineated by a black outline.
264

Long-tailed Scene Generation
The flexibility of 3D semantic MPI has conferred significant
265

advantages upon our approach. In the following, we create long-tail scenes that rarely occur in
266

our real world for evaluation. In Fig. 1, we show that we manually add parallel traffic cones in
267

front of the ego vehicle. This scene has never happened in the training dataset, but our geometric
268

controllability provides us the capability to create such data. We then use the created scene to test
269

autonomous driving systems such as end-to-end planner VAD [9] to validate its effectiveness. In
270

this case, VAD successfully predicts correct waypoints with the high-level command ‘turn left’.
271

Moreover, in appendix Sec. B, we generate long-tailed scenes with extreme weather such as snow
272

and sandstorms, and evaluate perception model on it to examine its generalizability of rare weather.
273

Comparison with Baselines
In Fig. 6 (a), we visualize a comparison with ControlNet. We find
274

that ControlNet struggles to distinguish the overlapping instances in 2D-pixel space. This leads to the
275

two parked cars being merged into a single car with incorrect pose. In contrast, our 3D semantic MPIs
276

contain more than 2D semantic mask, but also account for complete scene geometry with occluded
277

Method
Condition Type
FID

BEVGen [23]
BEV map
25.54
BEVControl [34]
BEV map
24.85
DriveDreamer [26]
Box + FoV map
52.60
MagicDrive [5]
Box + BEV map
16.20
Panacea [30]
Box + FoV map
16.96
Ours
3D Semantic MPI
14.75

Table 2: Comparison of FID with previous methods
on the nuScenes dataset.

MPI Encoder
Reweighing Method
Metric

design
Progressive
Depth
CBGS
mIoU

3×3
-
-
-
21.96
1×1
-
-
-
23.05
1×1
✓
-
-
23.63
1×1
✓
✓
-
24.40
1×1
✓
✓
✓
25.50

Table 3: Ablation of different designs of the MPI encoder and
reweighing methods.

8


---Page Break---
parts. Together with our proposed MPI encoder and reweighing strategy, our framework yields a
278

realistic image generation with high-quality label alignment. More comparison is provided in Sec. D.
279

4.3
Quantitative Results
280

Recognizability, Realism and Controllability Evaluation
To evaluate whether our generated
281

images aligned with given annotations, we provide Gen experiment in Tab. 1. Using the annotation of
282

val set, we synthesize a copy of val set’s images, then use perception model trained on real training set
283

to perform evaluation. The performance will be more effective as it is close to the oracle performance.
284

We find that local method (ControlNet) perform better than global method (MagicDrive). Furthermore,
285

SytheOcc generalizes the locality for 3D conditioning to yield better performance.
286

Data Augmentation for 3D Occupancy Prediction
Notably, we conduct experiments using our
287

synthesized dataset to enhance the real training set in Tab. 1. We first use the occupancy labels from
288

training set to create a synthetic training set. Then we modify the loading pipeline in perception model
289

to randomly sample images from real dataset or synthetic dataset and train network from scratch.
290

Therefore, our approach preserves the inherent training dynamics of the neural network by solely
291

modifying the training images, without any alteration to the number of training iterations or epochs.
292

As MagicDrive-Aug exhibits numerical overflow when training FB-Occ, which may attributed to
293

unsatisfactory recognizability, we have to omit it and only provide MagicDrive-Gen experiments.
294

As shown in Tab. 1, where SytheOcc-Aug denotes the augmentation experiments using our generated
295

dataset, shows a satisfactory improvement over the prior state of the art. We emphasize that surpassing
296

the performance of the original dataset is not the primary objective of our work; rather, it is an
297

ancillary benefit that emerges from our framework for geometry-controlled generation.
298

Ablations
In Tab. 3, we present ablation studies across several design spaces of our model, analo-
299

gous to the Gen experiment in Tab. 1. We find that our designed MPI encoder of 1×1 conv have sig-
300

nificant improvement when compared to the conventional 3×3 conv approach. Besides, our proposed
301

three types of reweighing methods demonstrate a consistent improvement over the baseline. As a
302

result, the improved image quality and label alignment enable higher precision in downstream tasks.
303

5
Limitation and Broader Impacts
304

Layout Genereation
Our method is restricted in a conditional generation framework that should
305

have a conditional input at first. Our condition signal is from the original dataset annotation. Thus
306

most of the augmented data is generated using the same occupancy layout, or with minimal human
307

editing. Future research can incorporate the recent research [10,17,32,40] that generates occupancy
308

descriptions of the scenes to synthesize images with novel occupancy layouts.
309

Closed-loop Simulation
Given the underlying diverse and controllable image generation of our
310

method, it would be advantageous and valuable to extend our work to a broader domain such as closed-
311

loop simulation [16,38], to enable high-fidelity autonomous systems testing. This line of work can
312

be conducted by utilizing motion conditions to generate future frames as in world model [17,28,36],
313

or by explicitly modeling scene graph as in the case of UniSim [20,38] and NeuroNCAP [16].
314

Long-tailed Scene Generation
In this paper, we only investigate a limited number of long-tailed
315

scene generation and corner case evaluations such as rare layout in Fig. 1 and extreme weather in
316

Sec. B. Future work can extend our framework to (i) Synthesize more samples for tail classes to boost
317

performance. (ii) Generate or replicate large-scale databases of corner cases [11] for robust perception.
318

6
Conclusion
319

In this paper, we propose SytheOcc, an innovative image generation framework that is empowered
320

with geometry-controlled capabilities using occupancy. We introduce a novel 3D representation,
321

3D semantic MPIs, to address the critical challenge of how to efficiently encode occupancy. This
322

representation not only preserves the authentic and complete 3D geometry details with semantics, but
323

also provides a spatial-align feature representation for 2D diffusion models. With this property, our
324

method enjoys photorealistic appearances and fine-grained 3D controllability, serves as a generative
325

data engine to enable a broad range of applications. Extensive experiments demonstrate that our
326

synthetic data facilitate the training for perception models on occupancy prediction, and provide
327

valuable corner case evaluation in a simulated world.
328

9


---Page Break---
References
329

[1] Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375,
330

2018. 5
331

[2] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan,
332

Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving.
333

In CVPR, 2020. 7
334

[3] Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, and Dit-Yan Yeung. Integrating geometric
335

control into text-to-image diffusion models for high-quality detection data generation via text prompt.
336

arXiv preprint arXiv:2306.04607, 2023. 3, 6
337

[4] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-
338

free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 12
339

[5] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive:
340

Street view generation with diverse 3d geometry control. In ICLR, 2024. 1, 3, 5, 8, 14
341

[6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
342

trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 7
343

[7] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint:2207.12598, 2022. 7
344

[8] Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting
345

textured 3d meshes from 2d text-to-image models. In ICCV, 2023. 12
346

[9] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu,
347

Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving.
348

In ICCV, 2023. 2, 8, 12, 13
349

[10] Jumin Lee, Sebin Lee, Changho Jo, Woobin Im, Juhyeong Seon, and Sung-Eui Yoon. Semcity: Semantic
350

scene generation with triplane diffusion. arXiv preprint arXiv:2403.07773, 2024. 9
351

[11] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang,
352

Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in
353

autonomous driving. In ECCV, 2022. 9
354

[12] Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, and Ying-Cong Chen. Lift3d: Synthesize 3d training
355

data by lifting 2d gan to 3d generative radiance field. In CVPR, 2023. 1, 3
356

[13] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scene
357

video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023. 3
358

[14] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and
359

Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 2
360

[15] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez.
361

Fb-occ: 3d occupancy prediction based on forward-backward view transformation.
arXiv preprint
362

arXiv:2307.01492, 2023. 7
363

[16] William Ljungbergh, Adam Tonderski, Joakim Johnander, Holger Caesar, Kalle Åström, Michael Felsberg,
364

and Christoffer Petersson. Neuroncap: Photorealistic closed-loop safety testing for autonomous driving.
365

arXiv preprint arXiv:2404.07762, 2024. 9
366

[17] Jiachen Lu, Ze Huang, Jiahui Zhang, Zeyu Yang, and Li Zhang. Wovogen: World volume-aware diffusion
367

for controllable multi-camera driving scene generation. arXiv preprint arXiv:2312.02934, 2023. 3, 9
368

[18] Jianbiao Mei, Yu Yang, Mengmeng Wang, Tianxin Huang, Xuemeng Yang, and Yong Liu. Ssc-rs: Elevate
369

lidar semantic scene completion with representation separation and bev fusion. In IROS, 2023. 3
370

[19] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren
371

Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1
372

[20] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic
373

scenes. In CVPR, 2021. 9
374

[21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution
375

image synthesis with latent diffusion models. In CVPR, 2022. 3, 7
376

[22] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao.
377

Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint
378

arXiv:2305.11337, 2023. 12
379

[23] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird’s-eye view
380

layout. IEEE RAL, 2024. 1, 3, 8
381

10


---Page Break---
[24] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang
382

Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. NeurIPS, 2024.
383

1, 3, 7
384

[25] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping
385

Luo, Dahua Lin, et al. Scene as occupancy. In ICCV, 2023. 3
386

[26] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards real-world-
387

driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 3, 8
388

[27] Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen
389

Lu, and Xingang Wang. Openoccupancy: A large scale benchmark for surrounding semantic occupancy
390

perception. In ICCV, 2023. 1, 3, 7
391

[28] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future:
392

Multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint
393

arXiv:2311.17918, 2023. 3, 5, 9
394

[29] Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera
395

3d occupancy prediction for autonomous driving. In ICCV, 2023. 3
396

[30] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang,
397

Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous
398

driving. arXiv preprint arXiv:2311.16813, 2023. 1, 3, 5, 8
399

[31] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying
400

Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for
401

text-to-video generation. In ICCV, 2023. 5, 14
402

[32] Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu,
403

Hiroyuki Sato, Hongdong Li, et al. Blockfusion: Expandable 3d scene generation using latent tri-plane
404

extrapolation. arXiv preprint arXiv:2401.17053, 2024. 9
405

[33] Xu Yan, Jiantao Gao, Jie Li, Ruimao Zhang, Zhen Li, Rui Huang, and Shuguang Cui. Sparse single sweep
406

lidar point cloud segmentation via learning contextual shape priors from scene completion. In AAAI, 2021.
407

3
408

[34] Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, and Kaicheng Yu. Bevcontrol: Accurately
409

controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv preprint
410

arXiv:2308.01661, 2023. 3, 8
411

[35] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui,
412

and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM
413

Computing Surveys, 2023. 3
414

[36] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel.
415

Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 9
416

[37] Yuanbo Yang, Yifei Yang, Hanlei Guo, Rong Xiong, Yue Wang, and Yiyi Liao. Urbangiraffe: Representing
417

urban scenes as compositional generative neural feature fields. In ICCV, 2023. 3
418

[38] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel
419

Urtasun. Unisim: A neural closed-loop sensor simulator. In CVPR, 2023. 9
420

[39] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester
421

Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere.
422

arXiv preprint arXiv:2312.03884, 2023. 12
423

[40] Junge Zhang, Qihang Zhang, Li Zhang, Ramana Rao Kompella, Gaowen Liu, and Bolei Zhou. Urban
424

scene diffusion through semantic occupancy map. arXiv preprint arXiv:2403.11697, 2024. 9
425

[41] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion
426

models. In ICCV, 2023. 2, 3, 5
427

[42] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector
428

framework for fast sampling of diffusion models. NeurIPS, 2023. 7
429

[43] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification:
430

Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2, 4
431

[44] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu. Class-balanced grouping and
432

sampling for point cloud 3d object detection. arXiv preprint arXiv:1908.09492, 2019. 6
433

11


---Page Break---
Appendix
434

In the appendix, we provide the following content:
435

Sec. A: Statement of Geometric Control.
Sec. E: Results of Video Generation.
Sec. B: Long-Tailed Scene Evaluation.
Sec. F: Generalize to New Cameras.
Sec. C: Ablation of plane number in MPIs.
Sec. G: Impact of Amount of Augment Data.
Sec. D: Additional Qualitative Comparison.
Sec. H: Visualization of Failure Cases.

A
Statement of Geometric Control
436

In our paper, we refer the geometric controllable generation as using a voxel grid in 3D space to
437

control the image generation. Although the voxel is a quantized representation of the 3D world,
438

when the resolution goes larger, it can already faithfully represent the geometry detail of scenes.
439

Currently, we are limited by the precision of ground truth labels. The 0.2m occupancy grid is a tensor
440

of 500×500×40 that cover a space in x-axis spanning [−50m, 50m], y-axis spanning [−50m, 50m],
441

z-axis spanning [−5m, 3m]. In the future, we plan to explore a higher resolution of geometric control
442

to refine our generation.
443

Except for occupancy, several other 3D representations can be expressed by 3D semantic MPI,
444

such as mesh, dense point clouds, and even 3D boxes or HD maps. The underlying mechanism is
445

to cast several slices of multi-plane images at different depths to retrieve geometric information.
446

Thus, our 3D semantic MPI can be regarded as a general 3D conditioning representation to benefit
447

a wide spectrum of practical systems. These encompass but are not limited to 3D generation such
448

as text2room [8], RoomDreamer [22], WonderJourney [39], and LucidDreamer [4], each of which
449

stands to benefit from the rich geometric context provided by our approach.
450

B
Long-Tailed Scene Evaluation
451

In this section, we explore to use SytheOcc to create long-tailed scenes for downstream evaluation.
452

This also stands for evaluating our model using several corner cases. Similar to the SytheOcc-Gen
453

experiment in Tab. 1, we generate a synthetic validation set but use prompts control to manipulate
454

weather patterns or the intensity of illumination.
455

As depicted in Fig. 7. We create a variety of weather conditions including sandstorms, snow, foggy,
456

rainy, day night, and day time. The motivation behind the creation of these scenes lies in their extreme
457

rarity compared to the ordinary scenes we have captured. The generation of such data is of significant
458

value, as it aids in addressing the long-tailed distribution of scenes, thereby enriching the diversity of
459

our dataset. More visualization is provided in Fig. 13 to Fig. 14.
460

In Tab. 4, we observe that all kinds of extreme weather lead to a degradation in performance. This
461

observation underscores the limitations of the perception model in terms of its generalizability to
462

infrequent weather scenarios. Among them, we find that foggy, rainy, and day night exert the most
463

severe impact, as they contribute to a large reduction in visibility as shown in Fig. 7. To improve the
464

generalizability to handle various weather conditions, future work can leverage our generated data to
465

cover the long-tailed scenes, or use adversarial search to find severe scenes based on our framework.
466

Scenes
Sandstorm
Snow
Foggy
Rainy
Day night
Day time (raw data)

FB-Occ mIOU
22.88
18.25
10.29
9.71
9.95
25.50

Table 4: Experiments of downstream evaluation on long-tailed scenes with extreme weather.

Furthermore, we perform long-tailed scene evaluation in Fig. 8. We display the failure of the
467

downstream model VAD [9] in our synthetic long-tailed scene. In this case, we simulate a foggy
468

environment that the dense fog obscures the majority of the ego view. Our experiment reveals that
469

due to the lack of training images of foggy scenes, VAD erroneously predicts waypoints that would
470

result in a collision with the bus. This experiment elucidates the boundaries and failure cases of the
471

VAD model [9]. It exposes the limitations of the system under certain conditions, thereby providing
472

insights into scenarios where the model’s performance may be compromised.
473

12


---Page Break---
Figure 7: From top to bottom, we display images of fusion of 3D semantic MPI, synthesized images
of sandstorm, snow, foggy, rainy, day night, day time, and ground truth.

Testing
End-to-end

Planner

VAD

Corner

Case
Evaluation

Predicted waypoints (Static)

Predicted waypoints
Generated image

Raw image

Prompt-level

Control: 
Add fog

SyntheOcc

User Editing 

To Create
Corner Case

Scene geometry

Figure 8: Use SytheOcc to create long-tailed scenes for testing. Top: In the ordinary scene of a
bus placed in front of the ego vehicle, the end-to-end planner VAD [9] predicts future waypoints
without movement, thus not plotted in the image. Bottom: By harnessing the prompt-level control in
our framework, we simulate a scene with the same layout but filled with fog. VAD predicts wrong
waypoints that will collide with the bus.

C
Ablation of plane number of MPIs
474

In our proposed 3D semantic MPIs, the number of planes is a hyperparameter that affects the precision
475

of 3D representation. The plane number can be regarded as the 3D resolution in depth axis. The
476

larger the plane number, the MPI will contain more details. We find that an increase in the number of
477

planes is associated with improved accuracy in downstream tasks. This finding denotes that more
478

condition information leads to better downstream task performance.
479

13


---Page Break---
Fusion of MPIs
MagicDrive
ControlNet
ControlNet+depth
SyntheOcc
GT

Figure 9: Comparison with baselines.

Number of Planes
96
128
256

FB-Occ mIOU
23.36
24.28
25.50

Table 5: Ablation of the number of multi-plane images.

D
Qualitative Comparison with Baselines and SOTA
480

In Fig. 9, we conduct a qualitative comparison of our method against MagicDrive, ControlNet, and
481

ControlNet+depth. We find that all the methods display a satisfactory image quality, as they build upon
482

the foundation of the stable diffusion model. The generation of MagicDrive fails to synthesize barriers
483

as shown in the bottom row. ControlNet struggles to generate objects with the correct pose solely
484

from only 2D conditions as shown in the second row. ControlNet+depth, a degradation of our method,
485

an enhancement over ControlNet in terms of alignment, nevertheless suffers from a loss of finer detail
486

in scenes with heavy occlusion, as shown in the human of the third row. Our method, in contrast, aims
487

to address these challenges and provide a more nuanced and accurate generation of complex scenes.
488

E
Extend to Video Generation
489

As described in the main paper Sec. 3.4, we further extend the cross-view attention to cross-frame
490

attention to perform video generation. Our generation results are Fig. 11, Fig. 12 and Fig. 16.
491

Our implementation is adopted from MagicDrive [5] which is similar to Tune-a-video [31]. The
492

formulation of cross-frame attention is:
493

Attention(Q, K, V ) = softmax( QKT

√

d ) · V ,
(8)

hout = hin + P

i∈{f,h}Attention(Qin, Ki, Vi),
(9)

where f, and h are the camera view of future and history frames. Qin and hin denotes the query and
494

the hidden state of input view. We train our model in a two-stage pipeline. We first train the MPI
495

encoder and cross-view attention in a multi-view image generation setting. Then we train cross-frame
496

attention and freeze other components in a video generation setting.
497

In practice, we use the keyframe annotation of the nuScenes dataset to train our video model. We start
498

with our pretrained MPI encoder and cross-view attention and only train our cross-frame attention
499

while keeping others frozen. We employ a sequence of 7 frames as a batch, resulting in a batch size
500

of 42 images for the training process.
501

Given that our primary contribution does not lie in video generation, this experiment serves as a
502

proof of concept, demonstrating the potential of our framework. Future research may extend our
503

methodology to facilitate the generation of longer video sequences, thereby expanding the scope and
504

applicability of our framework.
505

14


---Page Break---
(a) Generation with original intrinsic

(b) Generation with intrinsic modification (focal length * 0.8)

(c) Generation with intrinsic modification (focal length * 1.2)

Figure 10: We demonstrate the generalizability of SytheOcc to new camera intrinsic. We multiply
factors to the focal length while keeping the resolution the same. In (b,c), focal length ×0.8 denotes
a camera with a larger field of view similar to zoom out, focal length ×1.2 denotes a camera with a
smaller field of view similar to zoom in.

F
Generalize to New Cameras
506

In this section, we investigate the adaptability of our method to a new set of cameras with different
507

intrinsic. Given that our training set has a fixed camera intrinsic and extrinsic, generalizing to novel
508

cameras indicates that our approach possesses robust generalization capabilities. As shown in Fig. 10,
509

benefiting from our local type of condition, SytheOcc generates images that faithfully align with
510

the new intrinsic, proving that SytheOcc do not over-fit certain parameters. Regarding extrinsic
511

parameters, we can cast our MPI at the desirable locations to retrieve geometric information, thus
512

inherently ensuring generalizability without doubt.
513

G
The Influence of the Amount of Augmented Data
514

As SytheOcc is capable of generating an infinite number of synthetic data, we investigate the influence
515

of the amount of augmented data on downstream tasks in Tab. 6. We find that when our augmented
516

data is expanded from one-fold to two-fold of the training dataset, the performance of perception
517

model slightly decreases. This may indicate the generated data has an optimal ratio for downstream
518

tasks. Due to limited computational resources, we only experiment with a limited amount of ratio.
519

Future work can conduct more thorough experiments to find a universal theorem.
520

Amount of Augmented Data
0 (no augmentation)
1
2

FB-Occ mIOU
39.3
40.3
40.1

Table 6: Ablation of the amount of augmented data.

15


---Page Break---
Figure 11: Video generation results. In the temporal progression, the distant buildings maintain a
high degree of consistency, and objects retain their identical shapes and textures across different
views and frames.

Figure 12: Video generation results of large dynamics scenes. The white car comes across different
views and frames depicting consistent shapes with only a slight appearance change.

16


---Page Break---
Figure 13: From top to bottom, we display images of fusion of 3D semantic MPI, synthesized images
of sandstorm, snow, foggy, rainy, day night, day time, and ground truth.

Figure 14: Weather variation. Same structure with Fig. 13.

17


---Page Break---
Figure 15: Out of distribution generation. We use prompts to control the high-level appearance of
images with specific styles. From top to bottom, we display (1) fusion of 3D semantic MPI. (2) Sunny
day. (3) Science fiction style. (4) 8-bit pixel art style. (5) Snowfall. (6) Minecraft style. (7) Pokémon
style. (8) Diablo style. (9) Ghibli style. (10) Metropolis style. (11) Gotham style. (12) Ground truth.

18


---Page Break---
H
Failure Cases
521

We display several failure cases of our method. In Fig. 16, we show a crowd scenes. In this scenario,
522

the excessive number of pedestrians presents a challenge to the cross-view attention and cross-frame
523

attention modules. We find our method incapable of discerning individual entities with clarity. Future
524

research can improve the model capacity or enrich high-quality data to mitigate this problem.
525

Figure 16: Failure case of video generation results. Our cross-frame attention module is challenging
to distinguish a crowd of people across different views and frames.

19


---Page Break---
NeurIPS Paper Checklist
526

The checklist is designed to encourage best practices for responsible machine learning research,
527

addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
528

the checklist: The papers not including the checklist will be desk rejected. The checklist should
529

follow the references and follow the (optional) supplemental material. The checklist does NOT count
530

towards the page limit.
531

Please read the checklist guidelines carefully for information on how to answer these questions. For
532

each question in the checklist:
533

• You should answer [Yes] , [No] , or [NA] .
534

• [NA] means either that the question is Not Applicable for that particular paper or the
535

relevant information is Not Available.
536

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).
537

The checklist answers are an integral part of your paper submission. They are visible to the
538

reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
539

(after eventual revisions) with the final version of your paper, and its final version will be published
540

with the paper.
541

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
542

While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
543

proper justification is given (e.g., "error bars are not reported because it would be too computationally
544

expensive" or "we were unable to find the license for the dataset we used"). In general, answering
545

"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
546

acknowledge that the true answer is often more nuanced, so please just use your best judgment and
547

write a justification to elaborate. All supporting evidence can appear either in the main paper or the
548

supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
549

please point to the section(s) where related material for the question can be found.
550

IMPORTANT, please:
551

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
552

• Keep the checklist subsection headings, questions/answers and guidelines below.
553

• Do not modify the questions and only use the provided macros for your answers.
554

1. Claims
555

Question: Do the main claims made in the abstract and introduction accurately reflect the
556

paper’s contributions and scope?
557

Answer: [Yes]
558

Justification: Please find this part in Sec. 3.
559

Guidelines:
560

• The answer NA means that the abstract and introduction do not include the claims
561

made in the paper.
562

• The abstract and/or introduction should clearly state the claims made, including the
563

contributions made in the paper and important assumptions and limitations. A No or
564

NA answer to this question will not be perceived well by the reviewers.
565

• The claims made should match theoretical and experimental results, and reflect how
566

much the results can be expected to generalize to other settings.
567

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
568

are not attained by the paper.
569

2. Limitations
570

Question: Does the paper discuss the limitations of the work performed by the authors?
571

Answer: [Yes]
572

Justification: Please find this part in Sec. 5.
573

20


---Page Break---
Guidelines:
574

• The answer NA means that the paper has no limitation while the answer No means that
575

the paper has limitations, but those are not discussed in the paper.
576

• The authors are encouraged to create a separate "Limitations" section in their paper.
577

• The paper should point out any strong assumptions and how robust the results are to
578

violations of these assumptions (e.g., independence assumptions, noiseless settings,
579

model well-specification, asymptotic approximations only holding locally). The authors
580

should reflect on how these assumptions might be violated in practice and what the
581

implications would be.
582

• The authors should reflect on the scope of the claims made, e.g., if the approach was
583

only tested on a few datasets or with a few runs. In general, empirical results often
584

depend on implicit assumptions, which should be articulated.
585

• The authors should reflect on the factors that influence the performance of the approach.
586

For example, a facial recognition algorithm may perform poorly when image resolution
587

is low or images are taken in low lighting. Or a speech-to-text system might not be
588

used reliably to provide closed captions for online lectures because it fails to handle
589

technical jargon.
590

• The authors should discuss the computational efficiency of the proposed algorithms
591

and how they scale with dataset size.
592

• If applicable, the authors should discuss possible limitations of their approach to
593

address problems of privacy and fairness.
594

• While the authors might fear that complete honesty about limitations might be used by
595

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
596

limitations that aren’t acknowledged in the paper. The authors should use their best
597

judgment and recognize that individual actions in favor of transparency play an impor-
598

tant role in developing norms that preserve the integrity of the community. Reviewers
599

will be specifically instructed to not penalize honesty concerning limitations.
600

3. Theory Assumptions and Proofs
601

Question: For each theoretical result, does the paper provide the full set of assumptions and
602

a complete (and correct) proof?
603

Answer: [NA]
604

Justification: The paper does not include theoretical results.
605

Guidelines: Do not have theoretical results.
606

• The answer NA means that the paper does not include theoretical results.
607

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
608

referenced.
609

• All assumptions should be clearly stated or referenced in the statement of any theorems.
610

• The proofs can either appear in the main paper or the supplemental material, but if
611

they appear in the supplemental material, the authors are encouraged to provide a short
612

proof sketch to provide intuition.
613

• Inversely, any informal proof provided in the core of the paper should be complemented
614

by formal proofs provided in appendix or supplemental material.
615

• Theorems and Lemmas that the proof relies upon should be properly referenced.
616

4. Experimental Result Reproducibility
617

Question: Does the paper fully disclose all the information needed to reproduce the main
618

experimental results of the paper to the extent that it affects the main claims and/or conclu-
619

sions of the paper (regardless of whether the code and data are provided or not)?
620

Answer: [Yes]
621

Justification: Please find this part in Sec. 4.
622

Guidelines:
623

• The answer NA means that the paper does not include experiments.
624

21


---Page Break---
• If the paper includes experiments, a No answer to this question will not be perceived
625

well by the reviewers: Making the paper reproducible is important, regardless of
626

whether the code and data are provided or not.
627

• If the contribution is a dataset and/or model, the authors should describe the steps taken
628

to make their results reproducible or verifiable.
629

• Depending on the contribution, reproducibility can be accomplished in various ways.
630

For example, if the contribution is a novel architecture, describing the architecture fully
631

might suffice, or if the contribution is a specific model and empirical evaluation, it may
632

be necessary to either make it possible for others to replicate the model with the same
633

dataset, or provide access to the model. In general. releasing code and data is often
634

one good way to accomplish this, but reproducibility can also be provided via detailed
635

instructions for how to replicate the results, access to a hosted model (e.g., in the case
636

of a large language model), releasing of a model checkpoint, or other means that are
637

appropriate to the research performed.
638

• While NeurIPS does not require releasing code, the conference does require all submis-
639

sions to provide some reasonable avenue for reproducibility, which may depend on the
640

nature of the contribution. For example
641

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
642

to reproduce that algorithm.
643

(b) If the contribution is primarily a new model architecture, the paper should describe
644

the architecture clearly and fully.
645

(c) If the contribution is a new model (e.g., a large language model), then there should
646

either be a way to access this model for reproducing the results or a way to reproduce
647

the model (e.g., with an open-source dataset or instructions for how to construct
648

the dataset).
649

(d) We recognize that reproducibility may be tricky in some cases, in which case
650

authors are welcome to describe the particular way they provide for reproducibility.
651

In the case of closed-source models, it may be that access to the model is limited in
652

some way (e.g., to registered users), but it should be possible for other researchers
653

to have some path to reproducing or verifying the results.
654

5. Open access to data and code
655

Question: Does the paper provide open access to the data and code, with sufficient instruc-
656

tions to faithfully reproduce the main experimental results, as described in supplemental
657

material?
658

Answer: [Yes]
659

Justification: Please find this part in Sec. 4.
660

Guidelines:
661

• The answer NA means that paper does not include experiments requiring code.
662

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
663

public/guides/CodeSubmissionPolicy) for more details.
664

• While we encourage the release of code and data, we understand that this might not be
665

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
666

including code, unless this is central to the contribution (e.g., for a new open-source
667

benchmark).
668

• The instructions should contain the exact command and environment needed to run to
669

reproduce the results. See the NeurIPS code and data submission guidelines (https:
670

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
671

• The authors should provide instructions on data access and preparation, including how
672

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
673

• The authors should provide scripts to reproduce all experimental results for the new
674

proposed method and baselines. If only a subset of experiments are reproducible, they
675

should state which ones are omitted from the script and why.
676

• At submission time, to preserve anonymity, the authors should release anonymized
677

versions (if applicable).
678

22


---Page Break---
• Providing as much information as possible in supplemental material (appended to the
679

paper) is recommended, but including URLs to data and code is permitted.
680

6. Experimental Setting/Details
681

Question: Does the paper specify all the training and test details (e.g., data splits, hyperpa-
682

rameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
683

Answer: [Yes]
684

Justification: Please find this part in Sec. 4.
685

Guidelines:
686

• The answer NA means that the paper does not include experiments.
687

• The experimental setting should be presented in the core of the paper to a level of detail
688

that is necessary to appreciate the results and make sense of them.
689

• The full details can be provided either with the code, in appendix, or as supplemental
690

material.
691

7. Experiment Statistical Significance
692

Question: Does the paper report error bars suitably and correctly defined or other appropriate
693

information about the statistical significance of the experiments?
694

Answer: [Yes]
695

Justification: Please find this part in Sec. 4.
696

Guidelines:
697

• The answer NA means that the paper does not include experiments.
698

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
699

dence intervals, or statistical significance tests, at least for the experiments that support
700

the main claims of the paper.
701

• The factors of variability that the error bars are capturing should be clearly stated (for
702

example, train/test split, initialization, random drawing of some parameter, or overall
703

run with given experimental conditions).
704

• The method for calculating the error bars should be explained (closed form formula,
705

call to a library function, bootstrap, etc.)
706

• The assumptions made should be given (e.g., Normally distributed errors).
707

• It should be clear whether the error bar is the standard deviation or the standard error
708

of the mean.
709

• It is OK to report 1-sigma error bars, but one should state it. The authors should
710

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
711

of Normality of errors is not verified.
712

• For asymmetric distributions, the authors should be careful not to show in tables or
713

figures symmetric error bars that would yield results that are out of range (e.g. negative
714

error rates).
715

• If error bars are reported in tables or plots, The authors should explain in the text how
716

they were calculated and reference the corresponding figures or tables in the text.
717

8. Experiments Compute Resources
718

Question: For each experiment, does the paper provide sufficient information on the com-
719

puter resources (type of compute workers, memory, time of execution) needed to reproduce
720

the experiments?
721

Answer: [Yes]
722

Justification: Please find this part in Sec. 4.
723

Guidelines:
724

• The answer NA means that the paper does not include experiments.
725

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
726

or cloud provider, including relevant memory and storage.
727

• The paper should provide the amount of compute required for each of the individual
728

experimental runs as well as estimate the total compute.
729

23


---Page Break---
• The paper should disclose whether the full research project required more compute
730

than the experiments reported in the paper (e.g., preliminary or failed experiments that
731

didn’t make it into the paper).
732

9. Code Of Ethics
733

Question: Does the research conducted in the paper conform, in every respect, with the
734

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
735

Answer: [Yes]
736

Justification: It should be fine.
737

Guidelines:
738

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
739

• If the authors answer No, they should explain the special circumstances that require a
740

deviation from the Code of Ethics.
741

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
742

eration due to laws or regulations in their jurisdiction).
743

10. Broader Impacts
744

Question: Does the paper discuss both potential positive societal impacts and negative
745

societal impacts of the work performed?
746

Answer: [Yes]
747

Justification: Please find this part in Sec. 5.
748

Guidelines:
749

• The answer NA means that there is no societal impact of the work performed.
750

• If the authors answer NA or No, they should explain why their work has no societal
751

impact or why the paper does not address societal impact.
752

• Examples of negative societal impacts include potential malicious or unintended uses
753

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
754

(e.g., deployment of technologies that could make decisions that unfairly impact specific
755

groups), privacy considerations, and security considerations.
756

• The conference expects that many papers will be foundational research and not tied
757

to particular applications, let alone deployments. However, if there is a direct path to
758

any negative applications, the authors should point it out. For example, it is legitimate
759

to point out that an improvement in the quality of generative models could be used to
760

generate deepfakes for disinformation. On the other hand, it is not needed to point out
761

that a generic algorithm for optimizing neural networks could enable people to train
762

models that generate Deepfakes faster.
763

• The authors should consider possible harms that could arise when the technology is
764

being used as intended and functioning correctly, harms that could arise when the
765

technology is being used as intended but gives incorrect results, and harms following
766

from (intentional or unintentional) misuse of the technology.
767

• If there are negative societal impacts, the authors could also discuss possible mitigation
768

strategies (e.g., gated release of models, providing defenses in addition to attacks,
769

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
770

feedback over time, improving the efficiency and accessibility of ML).
771

11. Safeguards
772

Question: Does the paper describe safeguards that have been put in place for responsible
773

release of data or models that have a high risk for misuse (e.g., pretrained language models,
774

image generators, or scraped datasets)?
775

Answer: [NA]
776

Justification: Our paper poses no such risks.
777

Guidelines:
778

• The answer NA means that the paper poses no such risks.
779

24


---Page Break---
• Released models that have a high risk for misuse or dual-use should be released with
780

necessary safeguards to allow for controlled use of the model, for example by requiring
781

that users adhere to usage guidelines or restrictions to access the model or implementing
782

safety filters.
783

• Datasets that have been scraped from the Internet could pose safety risks. The authors
784

should describe how they avoided releasing unsafe images.
785

• We recognize that providing effective safeguards is challenging, and many papers do
786

not require this, but we encourage authors to take this into account and make a best
787

faith effort.
788

12. Licenses for existing assets
789

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
790

the paper, properly credited and are the license and terms of use explicitly mentioned and
791

properly respected?
792

Answer: [Yes]
793

Justification: Please find this part in Sec. 4.
794

Guidelines:
795

• The answer NA means that the paper does not use existing assets.
796

• The authors should cite the original paper that produced the code package or dataset.
797

• The authors should state which version of the asset is used and, if possible, include a
798

URL.
799

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
800

• For scraped data from a particular source (e.g., website), the copyright and terms of
801

service of that source should be provided.
802

• If assets are released, the license, copyright information, and terms of use in the package
803

should be provided. For popular datasets, paperswithcode.com/datasets has
804

curated licenses for some datasets. Their licensing guide can help determine the license
805

of a dataset.
806

• For existing datasets that are re-packaged, both the original license and the license of
807

the derived asset (if it has changed) should be provided.
808

• If this information is not available online, the authors are encouraged to reach out to
809

the asset’s creators.
810

13. New Assets
811

Question: Are new assets introduced in the paper well documented and is the documentation
812

provided alongside the assets?
813

Answer: [NA]
814

Justification: Our paper does not release new assets.
815

Guidelines:
816

• The answer NA means that the paper does not release new assets.
817

• Researchers should communicate the details of the dataset/code/model as part of their
818

submissions via structured templates. This includes details about training, license,
819

limitations, etc.
820

• The paper should discuss whether and how consent was obtained from people whose
821

asset is used.
822

• At submission time, remember to anonymize your assets (if applicable). You can either
823

create an anonymized URL or include an anonymized zip file.
824

14. Crowdsourcing and Research with Human Subjects
825

Question: For crowdsourcing experiments and research with human subjects, does the paper
826

include the full text of instructions given to participants and screenshots, if applicable, as
827

well as details about compensation (if any)?
828

Answer: [NA]
829

Justification: Our paper does not involve crowdsourcing nor research with human subjects.
830

25


---Page Break---
Guidelines:
831

• The answer NA means that the paper does not involve crowdsourcing nor research with
832

human subjects.
833

• Including this information in the supplemental material is fine, but if the main contribu-
834

tion of the paper involves human subjects, then as much detail as possible should be
835

included in the main paper.
836

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
837

or other labor should be paid at least the minimum wage in the country of the data
838

collector.
839

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
840

Subjects
841

Question: Does the paper describe potential risks incurred by study participants, whether
842

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
843

approvals (or an equivalent approval/review based on the requirements of your country or
844

institution) were obtained?
845

Answer: [NA]
846

Justification: Our paper does not involve crowdsourcing nor research with human subjects.
847

Guidelines:
848

• The answer NA means that the paper does not involve crowdsourcing nor research with
849

human subjects.
850

• Depending on the country in which research is conducted, IRB approval (or equivalent)
851

may be required for any human subjects research. If you obtained IRB approval, you
852

should clearly state this in the paper.
853

• We recognize that the procedures for this may vary significantly between institutions
854

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
855

guidelines for their institution.
856

• For initial submissions, do not include any information that would break anonymity (if
857

applicable), such as the institution conducting the review.
858

26


---Page Break---
