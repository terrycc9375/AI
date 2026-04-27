Deep Implicit Optimization for Robust and Flexible
Image Registration

Anonymous Author(s)
Affiliation
Address
email

Abstract

Deep Learning in Image Registration (DLIR) methods have been tremendously
1

successful in image registration due to their speed and ability to incorporate weak
2

label supervision at training time. However, DLIR methods forego many of the
3

benefits of classical optimization-based methods. The functional nature of deep
4

networks do not guarantee that the predicted transformation is a local minima
5

of the registration objective, the representation of the transformation (displace-
6

ment/velocity field/affine) is fixed, and the networks are not robust to domain shift.
7

Our method aims to bridge this gap between classical and learning methods by
8

incorporating optimization as a layer in a deep network. A deep network is trained
9

to predict multi-scale dense feature images that are registered using a black box
10

iterative optimization solver. This optimal warp is then used to minimize image and
11

label alignment errors. By implicitly differentiating end-to-end through an iterative
12

optimization solver, our learned features are registration and label-aware, and the
13

warp functions are guaranteed to be local minima of the registration objective
14

in the feature space. Our framework shows excellent performance on in-domain
15

datasets, and is agnostic to domain shift such as anisotropy and varying inten-
16

sity profiles. For the first time, our method allows switching between arbitrary
17

transformation representations (free-form to diffeomorphic) at test time with zero
18

retraining. End-to-end feature learning also facilitates interpretability of features,
19

and out-of-the-box promptability using additional label-fidelity terms at inference.
20

1
Introduction
21

Deformable Image Registration (DIR) refers to the local, non-linear alignment of images by estimating
22

a dense displacement field. Many workflows in medical image analysis require images to be in a
23

common coordinate system for comparison, analysis, and visualization, including comparing inter-
24

subject data in neuroimaging [53, 104, 97, 38, 89, 94], biomechanics and dynamics of anatomical
25

structures including myocardial motions, airflow and pulmonary function in lung imaging, organ
26

motion tracking in radiation therapy [78, 77, 11, 70, 29, 105, 50, 18, 71, 84], and life sciences
27

research [112, 104, 99, 80, 98, 72, 17].
28

Classical DIR methods are based on solving a variational optimization problem, where a similarity
29

metric is optimized to find the best transformation that aligns the images. However, these methods are
30

typically slow, and cannot leverage learning to incorporate a training set containing weak supervision
31

such as anatomical landmarks or expert annotations. The quality of the registration is therefore
32

limited by the fidelity of the intensity image. Deep Learning for Image Registration (DLIR) is an
33

interesting paradigm to overcome these challenges. DLIR methods take a pair of images as input
34

to a neural network and output a warp field that aligns the images, and their associated anatomical
35

landmarks. The neural network parameters are trained to minimize the alignment loss over image
36

pairs and landmarks in a training set. A benefit of this method is the ability to incorporate weak
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
supervision like anatomical landmarks or expert annotations during training, which performs better
38

landmark alignment without access to landmarks at inference time.
39

Motivation. However, DLIR methods face several limitations. First, the prediction paradigm of
40

deep learning implies the feature learning and amortized optimization steps are fused; transformations
41

predicted at test-time may not even be a local minima of the alignment loss between the fixed and
42

moving image. The end-to-end prediction also implies that the representation of the transforma-
43

tion is fixed (as a design choice of the network), and the model cannot switch between different
44

representations like free-form, stationary velocity, geodesic, LDDMM, B-Splines, or affine at test
45

time without additional finetuning, in sharp contrast to the flexibility of classical methods. Typical
46

registration workflows require a practitioner to try different parameterizations of the transformation
47

(free-form, stationary velocity, geodesic, LDDMM, B-Splines, affine) to determine the representation
48

most suitable for their application and additional retraining becomes expensive. Moreover, design
49

decisions like sparse keypoint learning for affine registration [103, 16, 69, 40] do not facilitate dense
50

deformable registration. Furthermore, DLIR methods do not allow interactive registration using
51

additional landmarks or label maps at test time, which is crucial for clinical applications. Hyper-
52

parameter tuning for regularization is also expensive for DLIR methods. Although recent methods
53

propose conditional registration [44, 67] to amortize over the hyperparameter search during training,
54

the family of regularization is fixed in such cases, and space of hyperparameters becomes exponential
55

in the number of hyperparameter families considered. Lastly, current DLIR methods are not robust
56

to minor domain shifts like varying anisotropy and voxel resolutions, different image acquisition
57

and preprocessing protocols [62, 53, 70, 43]. Robustness to domain shift is imperative to biomedical
58

and clinical imaging where volumes are acquired with different scanners, protocols, and resolutions,
59

where the applicability of DLIR methods is limited to the training domain.
60

Contributions. We introduce DIO, a generic differentiable implicit optimization layer to a
61

learnable feature network for image registration. By decoupling feature learning and optimization,
62

our framework incorporates weak supervision like anatomical landmarks into the learned
63

features during training, which improves the fidelity of the feature images for registration. Feature
64

learning also leads to dense feature images, which smoothens the optimization landscape compared
65

to intensity-based registration due to homogenity present in most medical imaging modalities. Since
66

optimization frameworks are agnostic to spatial resolutions and feature distortions, DIO is extremely
67

robust to domain shifts like varying anisotropy, difference in sizes of fixed and moving images, and
68

different image acquisition and preprocessing protocols, even when compared to models trained
69

on contrast-agnostic synthetic data [43]. Moreover, our framework allows zero-cost plug-and-
70

play of arbitrary transformation representations (free-form, geodesics, B-Spline, affine, etc.) and
71

regularization at test time without additional training and loss of accuracy. This also paves the way for
72

practitioners to perform quick and interactive registration, and use additional arbitrary ‘prompts’
73

such as new landmarks or label maps out-of-the-box at test time, as part of the optimization layer.
74

2
Related Work
75

Deep Learning for Image Registration
DIR refers to the alignment of a fixed image If with a
76

moving image Im using a transformation φ ∈T where T is a family of transformations. Classical
77

methods formulate a variational optimization problem to find the optimal φ that aligns the images [15,
78

4, 7, 5, 6, 2, 15, 25, 24, 23, 27, 39, 63, 102, 101, 100, 46, 60, 61, 76, 33, 32, 12]. In contrast,
79

earliest DLIR methods used supervised learning [19, 55, 82, 88] to predict the transformation φ.
80

Voxelmorph [13] was the first unsupervised method utilizing a UNet [83] for unsupervised registration
81

on brain MRI data. Recent works considered different architectural designs [21, 56, 48, 66],
82

cascade-based architectures and loss functions [116, 115, 49, 26, 68, 114, 79, 20], and symmetric
83

or inverse consistency-based formulations [65, 51, 52, 92, 116]. [67, 44] inject the hyperparameter
84

as input and perform amortized optimization over different values of the hyperparameter. Domain
85

randomization and finetuning [43, 96, 73, 30] are also proposed to improve robustness of registration
86

to domain shift, that is a core necessity in medical imaging since different institutions follow
87

varying acquisition and preprocessing pipelines. Foundational models are also proposed to improve
88

registration accuracy [57, 93]. Another line of work propose to use the implicit priors of deep
89

learning [95] within an optimization framework [110, 106, 49, 45]. We refer the reader to [36, 41, 28]
90

for other detailed reviews.
91

Iterative methods for DLIR
Owing to the success of iterative optimization methods, few DLIR
92

methods propose emulating the iterative optimization within a network. [115, 116] use a cascade of
93

2


---Page Break---
networks to iteratively predict a warp field, and use the warped moving image as the input to the next
94

layer in the cascade. TransMorph-TVF [20] uses a recurrent network to predict a time-dependent
95

velocity field. [114] use a shared weights encoder to output feature images at multiple scales, and a
96

deformation field estimator utilizing a correlation layer. RAFT [91] similarly builds a 4D correlation
97

volume from two 2D feature maps, and updates the optical flow field using a recurrent unit that
98

performs lookup on the correlation volume. However, such recursive formulations have a large
99

memory footprint due to explicit backpropagation through the entire cascade [8], and are not adaptive
100

or optimal with respect to the inputs. In contrast, DIO uses optimization as a layer – guaranteeing
101

convergence to a local minima, and implicit differentiation avoids storing the entire computation
102

graph making the framework both memory and time efficient.
103

Feature Learning for Image Registration
[103, 16, 69, 40] learn keypoints from images which
104

is then used to compute the optimal affine transform using a closed form solution. However, these
105

methods are restricted to transformations that can be represented by differentiable closed-form
106

analytical solutions, making backpropagation trivial. These sparse keypoints cannot be reused for
107

dense deformable registration either. On the other hand, dense deformable registration (diffeomorphic
108

or otherwise) is almost universally solved using iterative optimization methods. This motivates the
109

need to perform implicit differentiation through an iterative optimization solver to perform feature
110

learning for registration. Other approaches learn image features to perform registration [108, 59, 107,
111

81], but do not perform feature learning and registration end-to-end, i.e., the features obtained are not
112

task-aware and may not be optimal for registration, especially for anatomical landmarks. Learned
113

features are either fed into a functional form to compute the transformation end-to-end, or are learned
114

using unsupervised learning in a stagewise manner. In contrast, by implicitly differentiating through
115

a black-box iterative solver, and minimizing the image and label alignment losses end-to-end, DIO
116

learns features that are registration-aware, label-aware, and dense. The optimization routine also
117

guarantees that the transformation is a local minima of the alignment of high-fidelity feature images.
118

Deep Equilibrium models
Deep Equilibrium (DEQ) models [9, 34] have emerged as an interesting
119

alterative to recurrent architectures. DEQ layers solve a fixed-point equation of a layer to find its
120

equilibrium state without unrolling the entire computation graph. This leads to high expressiveness
121

without the need for memory-intensive backpropagation through time [10, 8, 31, 75, 37, 111].
122

PIRATE [45] uses DEQ to finetune the PnP denoiser network for registration, but unlike our work,
123

the data-fidelity term comes from the intensity images. However, these methods use DEQ to emulate
124

an infinite-layer network, which typically consists of learnable parameters within the recurrent layer.
125

Conceptually, our work does not aim to simply emulate such an infinite cascade, but rather use
126

DEQ to decouple feature learning and optimization in an end-to-end registration framework.
127

This inherits all the robustness and agnosticity of optimization-based methods, while retaining the
128

fidelity of learned features. DEQ allows us to avoid the layer-stacking paradigm for cascades, and use
129

optimization as a black box layer without storing the entire computation graph, leading to constant
130

memory footprint and faster convergence. This allows learnable features to be registration-aware
131

since gradients are backpropagated to the feature images through the optimization itself.
132

3
Methods
133

The registration problem is formulated as a variational optimization problem:
134

φ∗= arg min
φ L(If, Im ◦φ) + R(φ) = arg min
φ C(φ, If, Im)
(1)

where If and Im are fixed and moving images respectively, L is a loss function that measures
135

the dissimilarity between the fixed image and the transformed moving image, and R is a suitable
136

regularizer that enforces desirable properties of the transformation φ. We call this the image matching
137

objective. If the images If and Im are supplemented with anatomical label maps Lf and Lm, we call
138

this the label matching objective. Classical methods perform image matching on the intensity images,
139

but the label matching performance is bottlenecked by the fidelity of image gradients with respect to
140

the label matching objective, and dynamics of the optimization algorithm. Deep learning methods
141

mitigate this by injecting label matching objectives (for example, Dice score) into the objective
142

Eq. (1) and using a deep network with parameters θ to predict φ for every image pair as input. In
143

essence, learning-based problems solve the following objective:
144

θ∗= arg min
θ

X

f,m
L(If, Im ◦φθ) + D(Sf, Sm ◦φθ) + R(φθ) = arg min
θ

X

f,m
T(φθ, If, Im, Sf, Sm) (2)

3


---Page Break---
(a) Multi-scale feature extraction
(b) Optimization solver

(c) Resampling

Loss

(d) Image and label loss

Figure 1: Overview of our framework. (a) A neural network extracts multi-scale features from the
input images. (b)These features are used to optimize warp fields using a multi-scale differentiable
optimization solver. (c) The optimized transform is used to warp the moving image and labels. (d)
The warped image/label are compared with the fixed image/label using a similarity metric.

where φθ(If, Im) is abbreviated to φθ. This leads to learned transformations φθ that perform both
145

good image and label matching. However, the feature learning and optimization are coupled, and the
146

learned features are optimized only for a specific training domain. This limitation primarily marks
147

the difference between DIO and existing DLIR methods.
148

Fig. 1 shows the overview of our method. Our goal is to learn feature images such that regis-
149

tration in this feature space corresponds to both image and label matching performance, by
150

disentangling feature learning and optimization. We do this by using a feature network to extract
151

dense features from the intensity image, that are used to solve Eq. (1) using a black-box optimization
152

solver, and obtain an optimal transform φ∗. Once φ∗is obtained, this is plugged into Eq. (2) to obtain
153

gradients with respect to φ∗. Since φ∗is a function of the feature images, we implicitly differentiate
154

through the optimization to backpropagate gradients to the feature images and to the deep network.
155

We discuss the details of our method in the following sections.
156

3.1
Feature Extractor Network
157

The first component of our framework is a feature network that extracts dense features from the
158

intensity images. This network is parameterized by θ, and takes an image I ∈RH×W ×D×Cin as
159

input and outputs a feature map F ∈RH×W ×D×C, where C is the number of feature channels, i.e.
160

F = gθ(I). Unlike existing DLIR methods where moving and fixed images are concatenated and
161

passed to the network, our feature network processes the images independently. This allows the fixed
162

and moving images to be of different voxel sizes. The feature network can also output multi-feature
163

feature maps F = gθ(I) = [F 0, F 1, . . . , F N], where F k ∈RH/2k×W/2k×D/2k×Ck, which can be
164

used by multi-scale optimization solvers. The feature network is agnostic to architecture choice, and
165

we ablate on different architectures in the experiments.
166

3.2
Implicit Differentiation through Optimization
167

Given the feature maps Ff and Fm extracted from the fixed and moving images, an optimization
168

solver optimizes Eq. (1) to obtain the transformation φ∗. This can be written by modifying Eq. (1) to
169

use the feature maps F; i.e. φ∗= arg minφ C(Ff, Fm ◦φ). A local minima of this equation satisfies:
170

ϱ(φ∗, Ff, Fm) = ∂C

∂φ


φ∗
= 0
(3)

This φ∗is used to compute the loss Eq. (2) to minimize image and label matching objective. To
171

propagate derivatives from φ∗to the feature images Ff, Fm, we invoke the Implicit Function Theo-
172

rem [54]:
173

4


---Page Break---
Theorem 1 For a function ϱ : Rn × Rm1+m2
→Rn that is continuously differentiable, if
174

ϱ(φ∗, Ff, Fm) = 0 and
 ∂ϱ

∂φ

|φ∗̸= 0, then there exist open sets U, Vf, Vm containing φ∗, Ff, Fm,
175

and a function φ∗(Ff, Fm) defined on these open sets such that ϱ(φ∗(Ff, Fm), Ff, Fm) = 0.
176

Given the Implicit Function Theorem, we write ϱ(φ∗(Ff, Fm), Ff, Fm) = 0 and differentiate with
177

respect to Ff to obtain:
178

dϱ
dFf
= ∂ϱ

∂φ
∂φ
∂Ff
+ ∂ϱ

∂Ff
= 0 =⇒
∂φ
∂Ff
= −
 ∂ϱ

∂φ

−1 ∂ϱ

∂Ff
(4)

The gradients of φ come from Eq. (2) (i.e. ∂T

∂φ), and the gradients of Ff w.r.t. Eq. (2) are obtained as
179

∂T
∂Ff = −∂T

∂φ

∂ϱ
∂φ
−1
∂ϱ
∂Ff . The gradients of Fm are obtained similarly.
180

This design ensures that optimal registration in the feature space corresponds to optimal registration
181

both in the image and label spaces. Furthermore, the optimization layer ensures that the φ∗is a local
182

minima of this high-fidelity feature matching objective, i.e., the features obtained by the network.
183

Jacobian-Free Backprop
In practice, the Jacobian ∂ϱ

∂φ is expensive to compute, given the high
184

dimensionality of φ and ϱ. Following [31], we substitute the Jacobian to identity, and compute
185

ˆ
∂T
∂Ff ≈−∂T

∂φ
∂ϱ
∂Ff . This leads to much less memory and stable training dynamics compared to other
186

estimates of Jacobian like phantom gradients, damped unrolling, or Neumann series [35, 34].
187

3.3
Multi-scale optimization
188

Intensity image
Feature level 4
Feature level 2
Feature level 0

t1

40

20

0

20

40

t2

40

20

0

20

40

0.0

0.2

0.4

0.6

0.8

1.0

Dice loss

t1

40 20 0
20
40

t2

40

20

0

20

40

0.00
0.25
0.50
0.75
1.00
1.25

Feature level 4

t1

40 20 0
20
40

t2

40

20

0

20

40

0.00
0.25
0.50
0.75
1.00

1.25

Feature level 2

t1

40 20 0
20
40

t2

40

20

0

20

40

0.0

0.5

1.0

1.5

2.0

Feature level 0

Figure 2: Dense feature learning leads to flatter loss landscapes.
Top row shows the intensity image with the corresponding multi-
scale features predicted by the deep network, where the Lth level
denotes a feature of size H/2k×W/2k×Ck. Bottom row shows the
loss landscape as a function of the relative translation between the
squares in the fixed and moving image. Note the flat maxima which
occurs when there is no overlap between the fixed and moving
image, making optimization impossible if there is no overlap of the
squares. On the contrary, the loss landscape for learned features is
smooth, even at the finest scale, leading to much faster convergence
even when there is no overlap between the intensity images.

Optimization based methods typically
189

use a multi-scale approach to improve
190

convergence and avoid local minima
191

with the image matching objective [7,
192

5, 3, 15]. However, the downsampling
193

of intensity images leads to indiscrim-
194

inate blurring and loss of details at the
195

coarser scales. We adopt a multi-scale
196

approach by using pyramidal features
197

from the network, which are naturally
198

built into many convolutional archi-
199

tectures. We perform optimization at
200

the coarsest scale, and use the result
201

as initialization for the next finer scale
202

(Algorithm 2). This is similar to opti-
203

mization methods, but our multi-scale
204

features obtained from different layers
205

in the network correspond to different
206

semantic content, in contrast to clas-
207

sical methods where the multi-scale
208

features are simply downsampled ver-
209

sions of the original images.
This
210

allows the multi-scale registration to
211

align different anatomical regions at
212

different scales, which may be hard to
213

align at other finer or coarser scales.
214

4
Experiments
215

4.1
DIO learns dense features from sparse images
216

A key strength of DIO is the ability to learn interpretable dense features from sparse intensity images
217

for accurate and robust image matching. This is especially relevant for medical image registration,
218

which typically contain a lot of homogenity in the intensity images, making registration difficult.
219

We design a toy task to isolate and demonstrate this behavior. The fixed and moving images are
220

generated by placing a square of size 32×32 pixels on an image of 128×128 pixels. The squares in
221

5


---Page Break---
the fixed and moving images overlap with a 50% chance. The task is to find an affine transformation
222

to align the two images. However, classical optimization methods will fail this task 50% of the time,
223

because when the squares do not overlap, there is no gradient of the loss function, illustrated by the
224

flat loss landscape in Fig. 2. However, deep networks discover features that significantly flatten this
225

loss landscape in the feature matching space. To show this, we train a network to output multi-scale
226

feature maps that is used to optimize Eq. (1) to recover an affine transform. We choose a 2D UNet
227

architecture, and the multi-scale feature maps are recovered from different layers of the decoder path
228

of the UNet. Since the features are trained to maximize label matching, the loss landscape is much
229

flatter, and the network is able to recover the affine transform with > 99% overlap (Appendix A.4).
230

End-to-end learning enables learning of features that are most conducive to registration, unlike
231

existing work [108, 59, 107, 81] that may not contain discriminative registration-aware features
232

about anatomical labels due to lack of task-awareness.
233

4.2
Results on brain MRI registration
234

Setup: We evaluated our method on inter-subject registration on the OASIS dataset [62]. The
235

OASIS dataset contains 414 T1-weighted MRI scans of the brain with label maps containing 35
236

subcortical structures extracted from automatic segmentation with FreeSurfer and SAMSEG. We use
237

the preprocessed version from the Learn2Reg challenge [42] where all the volumes are skull-stripped,
238

intensity-corrected and center-cropped to 160×192×224. We use the same training and validation
239

sets as provided in the Learn2Reg challenge to enable fair comparison with other methods.
240

Table 1: Performance on OASIS validation set.
DIO is highly competitive with state-of-the-art DLIR
methods in the in-distribution setting. Our feature
learning incorporates label-aware features, which is
evident from the superior performance compared to
four SOTA optimization-based classical methods.

Validation
Method
Dice
HD95
ANTs [5]
0.786 ± 0.033
2.209 ± 0.534
NiftyReg [64]
0.775 ± 0.029
2.382 ± 0.723
LogDemons [100]
0.804 ± 0.022
2.068 ± 0.448
FireANTs [46]
0.791 ± 0.028
2.793 ± 0.602
Progressive C2F [58]
0.827 ± 0.013
1.722 ± 0.318
Little learning[87]
0.846 ± 0.016
1.500 ± 0.304
CLapIRN [67]
0.861 ± 0.015
1.514 ± 0.337
Voxelmorph-huge [14]
0.847 ± 0.014
1.546 ± 0.306
TransMorph [22]
0.858 ± 0.014
1.494 ± 0.288
TransMorph-Large [22]
0.862 ± 0.014
1.431 ± 0.282
Ours (UNet-E)
0.845 ± 0.018
1.790 ± 0.433
Ours (LKU-E)
0.849 ± 0.018
1.733 ± 0.401
Ours (UNet)
0.853 ± 0.018
1.675 ± 0.379
Ours (LKU)
0.862 ± 0.017
1.584 ± 0.351

Architectures: We consider four architec-
241

tures for the task, representing different in-
242

ductive biases in the network.
We use a
243

3D UNet architecture (denoted as UNet in
244

experiments), and a large-kernel UNet (de-
245

noted as LKU) [48]. To extract multi-scale
246

features from the networks, we attach sin-
247

gle convolutional layers to the feature of the
248

desired scales from the decoder path. For
249

each of these architectures, we also consider
250

“Encoder-Only” versions by discarding the de-
251

coder path, and creating independent encoders
252

for each scale Fig. 9, denoted as UNet-E and
253

LKU-E. We choose Encoder-Only versions to
254

ablate the performance using shared features
255

from the decoder path versus independent fea-
256

ture extraction at each scale.
257

Results: We compare our method with ex-
258

isting methods on the Learn2Reg OASIS chal-
259

lenge (Table 1). We compare with state-of-
260

the-art classical methods [5, 46, 64, 100], and
261

deep networks [58, 87, 67, 14, 22, 48]. DIO
262

is highly competitive with existing methods,
263

especially with TransMorph which uses up to two orders of magnitude more trainable parameters
264

than DIO to achieve a similar performance. We note that the Large Kernel UNet architecture performs
265

better than the standard UNet architecture, which is consistent with the findings in [48], even for
266

dense feature extraction. This is due to the larger receptive field of LKUNet, which is able to capture
267

more context in the image. Moreover, the Encoder-Only versions of the network perform slightly
268

worse than the full networks, showing that sharing features across scales is beneficial for the task.
269

4.3
Optimization-in-the-loop introduces robustness to domain shift
270

A key requirement of registration algorithms is to generalize over a spectrum of acquisition and
271

preprocessing protocols, since medical images are rarely acquired with the same configuration.
272

Existing DLIR methods are extremely sensitive to domain shift, and catastrophically fail on other
273

brain datasets. On the contrary, DIO inherits the domain agnosticism of the optimization solver, and
274

is robust under feature distortions introduced by domain shift.
275

We evaluate the robustness of the trained models on three brain datasets: LPBA40, IBSR18, and
276

CUMC12 datasets [85, 1, 53]. Contrary to the OASIS dataset, these datasets were obtained on
277

6


---Page Break---
different scanners, aligned to different atlases (MNI305, Talairach) with varying algorithms used
278

for skull-stripping, bias correction (BrainSuite, autoseg), and different manual labelling protocols
279

of different anatomical regions (as opposed to automatically generated Freesurfer labels in OASIS).
280

Unlike the OASIS dataset, these datasets have different volume sizes, and IBSR18 and CUMC12
281

datasets are not 1mm isotropic. More details about the datasets are provided in Appendix A.6.
282

Isotropic,Crop

0.2

0.4

0.6

0.8

Dice

Isotropic, No Crop

0.30

0.45

0.60

0.75

Anisotropic,Crop

0.2

0.4

0.6

0.8

Anisotropic, No Crop

0.2

0.4

0.6

0.8

IBSR18

Isotropic,Crop

0.15

0.30

0.45

0.60

Dice

Isotropic, No Crop

0.2

0.3

0.4

0.5

0.6

Anisotropic,Crop

0.15

0.30

0.45

0.60

Anisotropic, No Crop

0.15

0.30

0.45

0.60

CUMC12

Isotropic,Crop
0.48

0.56

0.64

0.72

Dice

Isotropic, No Crop

0.48

0.56

0.64

0.72

LPBA40

Method

Ours
TransMorph Regular
TransMorph Large (w/ Dice sup.)
Conditional LapIRN
SynthMorph
SymNet
TransMorph Regular (w/ Dice sup.)
SymNet (w/ Dice sup.)
LapIRN (w/ Dice sup.)
LKU-Net
LKU-Net (w/ Dice sup.)
LapIRN
VoxelMorph

Figure 3: Boxplots of Dice scores for three out-of-distribution datasets. DIO performs significantly
better across three datasets without additional finetuning. Contrary to other baselines that output warp fields
considering 1mm isotropic data, leading to a performance drop with anisotropic volumes, DIO performs better
with anisotropic data due to the optimization’s resolution-agnostic nature.

Results. We evaluate across a variety of configurations – (i) preserving the anisotropy of the
283

volumes or resampling to 1mm isotropic (denoted as anisotropic or isotropic), and (ii) center-cropping
284

the volumes to match the size of the OASIS dataset (denoted as Crop and No Crop). The results for all
285

three datasets are shown in Fig. 3 sorted by mean Dice score; quantitative comparison is also shown
286

in Appendix Table 4. Note that TransMorph, VoxelMorph, and SynthMorph do not work for sizes that
287

are different than the OASIS dataset, therefore they only work in the Crop setting. The IBSR18 dataset
288

also has volumes with different spatial sampling, and resampling to 1mm isotropic leads to different
289

voxel sizes. These volumes cannot be concatenated along the channel dimension, consequently every
290

DLIR method cannot run under this configuration (Fig. 3(a)). Since our method takes as input only a
291

single volume, and the convolutional architecture preserves the volume size, the fixed and moving
292

images can have different voxel sizes, i.e. feature extraction is not contingent on the voxel sizes of
293

the moving and fixed images being equal. The optimization solver can also handle different voxel
294

sizes for the fixed and moving volumes – which is useful in applications like multimodal registration
295

(in-vivo to ex-vivo, histology to 3D, MRI to microscopy). This unprecedented flexibility brings forth
296

7


---Page Break---
a new operational paradigm in deep learning for registration that was unavailable before, widening
297

the scope of applications for registration with deep features.
298

We compare our method with a variety of DLIR baselines, trained with and without label super-
299

vision (the former denoted as ‘w/ Dice sup.’ in Fig. 3). Our method performs substantially better
300

than all the baselines with a significantly narrower interquartile range on the IBSR18 and CUMC12
301

datasets. The differences are significant – on IBSR18 and CUMC12, our median performance is
302

higher than the third quartile of almost all baselines. The sturdy performance against domain shift
303

provides a strong motivation for using optimization-in-the-loop for learnable registration.
304

4.4
Robust feature learning enables zero-shot performance by switching optimizers at
305

test-time
306

Another major advantage of our framework is that we can switch the optimizer at test time without
307

any retraining. This is useful when the registration constraints change over time (i.e. initially
308

diffeomorphic transforms were required but now non-diffeomorphic transforms are acceptable), or
309

when the registration is used in a pipeline where different parameterizations (freeform, diffeomorphic,
310

geodesic, B-spline) may be compared. Since our framework decouples the feature learning from the
311

optimization, we can switch the optimizer arbitrarily at test time, at no additional cost. A crucial
312

requirement is that learned features should not be too sensitive to the training optimizer.
313

Optimizer
SGD
FireANTs (diffeomorphic)
Architecture
DSC
HD95
%(∥J∥< 0)
DSC
HD95
%(∥J∥< 0)
UNet Encoder
0.845 ± 0.018
1.790 ± 0.433
0.7866 ± 0.1371
0.834 ± 0.018
1.847 ± 0.410
0.0000 ± 0.0000
LKU Encoder
0.849 ± 0.018
1.733 ± 0.401
0.8079 ± 0.1308
0.838 ± 0.018
1.806 ± 0.373
0.0000 ± 0.0000
UNet
0.853 ± 0.018
1.675 ± 0.379
1.0718 ± 0.1662
0.842 ± 0.018
1.748 ± 0.397
0.0000 ± 0.0000
LKU
0.862 ± 0.017
1.584 ± 0.351
0.8646 ± 0.1429
0.849 ± 0.017
1.740 ± 0.345
0.0000 ± 0.0000
Table 2: Zero shot performance by switching optimizers at test-time. Our method is trained on the OASIS
dataset with the SGD optimizer to obtain the warp field. At inference time, we use an SGD optimizer for no
constraint on the warp field, and the FireANTs optimizer to ensure diffeomorphic warps. Across all architectures,
the Dice Score remains robust, with only a slight dip attributed to the constraints introduced by diffeomorphic
mappings. The SGD optimization introduces ∼1% singularities, while FireANTs shows no singularities.

Figure 4: Examples of multi-scale features learned
by the feature extractor. Scale-space features (bottom
row) obtained by downsampling the image downsam-
ple all image features indiscriminately. Our features
(top row) preserve necessary anatomical information
at all scales, and introduce inhomogenity in the fea-
ture space for better optimization (watershed effect
and enhanced contrast near gyri and a halo around
the outer surface to delineaate background from gray
matter).

To demonstrate this functionality, we use the val-
314

idation set of the OASIS dataset and the four net-
315

works trained in Section 4.2. The networks were
316

initially trained on the SGD optimizer without any
317

additional constraints on the warp field. At test
318

time, we switch the optimizer to the FireANTs
319

optimizer [46], that uses a Riemannian Adam op-
320

timizer for multi-scale diffeomorphisms. Results
321

in Table 2 compare the Dice score, 95th percentile
322

of the Haussdorf distance (denoted as HD95) and
323

percentage of volume with negative Jacobians (de-
324

noted as %(∥J∥< 0)) for the two optimizers. The
325

SGD optimizer introduces anywhere from 0.79%
326

to 1.1% of singularities in the registration, while
327

the FireANTs optimizer does not introduce any sin-
328

gularities. A slight drop in performance can be at-
329

tributed to the additional constraints imposed by dif-
330

feomorphic transforms. However, the high-fidelity
331

features lead to a much better label overlap than
332

FireANTs run with image features (Table 1). Our
333

framework introduces an unprecedented amount
334

of flexibility at test time that is an indispensible
335

feature in deep learning for registration, and can
336

be useful in a variety of applications where the reg-
337

istration requirements change over time, without
338

expensive retraining.
339

8


---Page Break---
4.5
Interpretability of features
340

Decoupling of feature learning and optimization allows us to examine the feature images obtained at
341

each scale to understand what feature help in the registration task. Classical methods use scale-space
342

images (smoothened and downsampled versions of the original image) to avoid local minima, but
343

lose discriminative image features at lower resolutions. Moreover, intensity images may not provide
344

sufficient details to perform label-aware registration. Since our method learns dense features to
345

minimize label matching losses, we can observe which features are necessary to enable label-aware
346

registration. Fig. 4 highlights differences between scale-space images and features learned by our
347

network. At all scales, the features introduces heterogeneity using a watershed effect and enhanced
348

contrast to improve label matching performance.
349

4.6
Inference time
350

DLIR methods have been very popular due to their fast inference time by performing amortized
351

optimization [14]. Classical methods generally focus on robustness and reproducubility, and do have
352

GPU implementations for fast inference. However, modern optimization toolkits [60, 46] utilize
353

massively parallel GPU computing to register images in seconds, and scale very well to ultrahigh
354

resolution imaging. A concern with optimization-in-the-loop methods is the inference time. Table
355

Table 3 shows the inference time for our method for all four architectures. These inference times are
356

fast for a lot of applications, and the plug-and-play nature of our framework makes DIO amenable to
357

rapid experimentation and hyperparameter tuning.
358

5
Conclusion and Limitations
359

Architecture
Neural net
Optimization
UNet
0.444
1.693
UNet-E
0.433
1.555
LKU
0.795
1.463
LKU-E
2.281
1.457

Table 3: Inference time for various architec-
tures. A multi-scale optimization takes only ∼1.5
seconds to run all iterations (no early stopping)
making it suitable for most applications. This is
compared to the time for neural network’s feature
extraction which is architecture dependent.

Conclusion
DLIR methods provide several bene-
360

fits such as amortized optimization, integration of
361

weak supervision, and the ability to learn from large
362

(labeled) datasets. However, coupling of the feature
363

learning and optimization steps in DLIR methods lim-
364

its the flexibility and robustness of the deep networks.
365

In this paper, we we introduce a novel paradigm
366

that incorporates optimization-as-a-layer for learning-
367

based frameworks. This paradigm retains all the flexi-
368

bility and robustness of classical multi-scale methods
369

while leverging large scale weak supervision such as
370

anatomical landmarks into high-fidelity, registration-
371

aware feature learning. Our paradigm allows “promptable” registration out-of-the-box as part of
372

the plug-and-play optimization, where additional supervision such as labelmaps or landmarks can
373

be added to the optimization loss at test time. Our fast implementation allows for implementation
374

of optimization-as-a-layer in deep learning, which was previously thought to be infeasible, due
375

to existing optimization frameworks being prohibitively slow. Densification of features from our
376

method also leads to better optimization landscapes, and our method is robust to unseen anisotropy
377

and domain shift. To our knowledge, our method is the first to switch between transformation
378

representations (free-form to diffeomorphic) at test time without any retraining. This comes with fast
379

inference runtimes, and interpretability of the features used for optimization. Potential future work
380

can explore multimodal registration, online hyperparameter tuning and few-shot learning.
381

Limitations
The first limitation is unlike existing DLIR methods that concatenate the fixed and
382

moving images to feed into the network, DIO processes the images independently. The features
383

extracted from an image are therefore trained to marginalize the label matching performance over all
384

possible moving images, and cannot adapt to the moving image. This leads to slightly asymptotically
385

lower in-domain performance than methods like [48]. The second limitation is the implicit bias of
386

the optimization algorithm. Implicit bias in SGD restricts the space of solutions for optimization
387

problems that are overparameterized, such as deep networks [113, 90, 47, 74, 109]. In deformable
388

registration, the implicit bias of SGD restricts the direction of the gradient of the particle at φ(x),
389

which is always parallel to ∇Fm(φ(x)), independent of the fixed image and dissimilarity function.
390

This limits the degrees of freedom of the optimization by N-fold for N-D images. This is unlike DLIR
391

methods where the warp is not constrained to move along ∇Fm(φ(x)). This behavior is explored in
392

more detail in Appendix A.1. Future work aims to mitigate this implicit bias for better performance.
393

9


---Page Break---
References
394

[1] Internet brain segmentation repository (IBSR).
http://www.cma.mgh.harvard.edu/
395

ibsr/.
396

[2] V. Arsigny, O. Commowick, X. Pennec, and N. Ayache. A Log-Euclidean Framework for
397

Statistics on Diffeomorphisms. In R. Larsen, M. Nielsen, and J. Sporring, editors, Medical
398

Image Computing and Computer-Assisted Intervention – MICCAI 2006, Lecture Notes in
399

Computer Science, pages 924–931, Berlin, Heidelberg, 2006. Springer.
400

[3] J. Ashburner. A fast diffeomorphic image registration algorithm. Neuroimage, 38(1):95–113,
401

2007.
402

[4] B. Avants and J. C. Gee. Geodesic estimation for large deformation anatomical shape averaging
403

and interpolation. NeuroImage, 23:S139–S150, Jan. 2004.
404

[5] B. B. Avants, C. L. Epstein, M. Grossman, and J. C. Gee. Symmetric diffeomorphic image
405

registration with cross-correlation: evaluating automated labeling of elderly and neurodegener-
406

ative brain. Medical Image Analysis, 12(1):26–41, Feb. 2008.
407

[6] B. B. Avants, C. L. Epstein, M. Grossman, and J. C. Gee. Symmetric diffeomorphic image
408

registration with cross-correlation: Evaluating automated labeling of elderly and neurodegen-
409

erative brain. Medical Image Analysis, 12(1):26–41, Feb. 2008.
410

[7] B. B. Avants, P. T. Schoenemann, and J. C. Gee. Lagrangian frame diffeomorphic image
411

registration: Morphometric comparison of human and chimpanzee cortex. Medical Image
412

Analysis, 10(3):397–412, June 2006.
413

[8] S. Bai, Z. Geng, Y. Savani, and J. Z. Kolter. Deep Equilibrium Optical Flow Estimation. In
414

2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
415

610–620, New Orleans, LA, USA, June 2022. IEEE.
416

[9] S. Bai, J. Z. Kolter, and V. Koltun. Deep equilibrium models. Advances in neural information
417

processing systems, 32, 2019.
418

[10] S. Bai, V. Koltun, and J. Z. Kolter. Multiscale deep equilibrium models. Advances in neural
419

information processing systems, 33:5238–5250, 2020.
420

[11] W. Bai, H. Suzuki, J. Huang, C. Francis, S. Wang, G. Tarroni, F. Guitton, N. Aung, K. Fung,
421

S. E. Petersen, et al. A population-based phenome-wide association study of cardiac and aortic
422

structure and function. Nature medicine, 26(10):1654–1662, 2020.
423

[12] R. Bajcsy, R. Lieberson, and M. Reivich. A computerized system for the elastic matching
424

of deformed radiographic images to idealized atlas images. Journal of computer assisted
425

tomography, 7(4):618–625, 1983.
426

[13] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca. VoxelMorph: A
427

Learning Framework for Deformable Medical Image Registration. IEEE Transactions on
428

Medical Imaging, 38(8):1788–1800, Aug. 2019. arXiv:1809.05231 [cs].
429

[14] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca. Voxelmorph: a learning
430

framework for deformable medical image registration. IEEE transactions on medical imaging,
431

38(8):1788–1800, 2019.
432

[15] M. F. Beg, M. I. Miller, A. Trouvé, and L. Younes. Computing large deformation metric
433

mappings via geodesic flows of diffeomorphisms. International journal of computer vision,
434

61:139–157, 2005.
435

[16] B. Billot, D. Moyer, N. Dey, M. Hoffmann, E. A. Turk, B. Gagoski, E. Grant, and P. Golland.
436

Se (3)-equivariant and noise-invariant 3d motion tracking in medical images. arXiv preprint
437

arXiv:2312.13534, 2023.
438

[17] B. E. Brezovec, A. B. Berger, Y. A. Hao, F. Chen, S. Druckmann, and T. R. Clandinin.
439

Mapping the neural dynamics of locomotion across the drosophila brain. Current Biology,
440

34(4):710–726, 2024.
441

[18] K. K. Brock, S. Mutic, T. R. McNutt, H. Li, and M. L. Kessler. Use of image registration
442

and fusion algorithms and techniques in radiotherapy: Report of the aapm radiation therapy
443

committee task group no. 132. Medical physics, 44(7):e43–e76, 2017.
444

10


---Page Break---
[19] X. Cao, J. Yang, J. Zhang, D. Nie, M. Kim, Q. Wang, and D. Shen. Deformable image
445

registration based on similarity-steered cnn regression. In Medical Image Computing and
446

Computer Assisted Intervention- MICCAI 2017: 20th International Conference, Quebec City,
447

QC, Canada, September 11-13, 2017, Proceedings, Part I 20, pages 300–308. Springer, 2017.
448

[20] J. Chen, E. C. Frey, and Y. Du. Unsupervised learning of diffeomorphic image registration
449

via transmorph. In International Workshop on Biomedical Image Registration, pages 96–102.
450

Springer, 2022.
451

[21] J. Chen, E. C. Frey, Y. He, W. P. Segars, Y. Li, and Y. Du. TransMorph: Transformer for
452

unsupervised medical image registration. Medical Image Analysis, 82:102615, Nov. 2022.
453

[22] J. Chen, E. C. Frey, Y. He, W. P. Segars, Y. Li, and Y. Du. TransMorph: Transformer for
454

unsupervised medical image registration. Medical Image Analysis, 82:102615, Nov. 2022.
455

arXiv:2111.10480 [cs, eess].
456

[23] G. E. Christensen and H. J. Johnson. Consistent image registration. IEEE transactions on
457

medical imaging, 20(7):568–582, 2001.
458

[24] G. E. Christensen, S. C. Joshi, and M. I. Miller. Volumetric transformation of brain anatomy.
459

IEEE transactions on medical imaging, 16(6):864–877, 1997.
460

[25] G. E. Christensen, R. D. Rabbitt, and M. I. Miller. Deformable templates using large deforma-
461

tion kinematics. IEEE transactions on image processing, 5(10):1435–1447, 1996.
462

[26] B. D. De Vos, F. F. Berendsen, M. A. Viergever, H. Sokooti, M. Staring, and I. Išgum. A deep
463

learning framework for unsupervised affine and deformable image registration. Medical image
464

analysis, 52:128–143, 2019.
465

[27] F. Dru, P. Fillard, and T. Vercauteren. An ITK Implementation of the Symmetric Log-Domain
466

Diffeomorphic Demons Algorithm. The Insight Journal, Sept. 2010.
467

[28] Y. Fu, Y. Lei, T. Wang, W. J. Curran, T. Liu, and X. Yang. Deep learning in medical image
468

registration: a review. Physics in Medicine & Biology, 65(20):20TR01, Oct. 2020.
469

[29] Y. Fu, Y. Lei, T. Wang, K. Higgins, J. D. Bradley, W. J. Curran, T. Liu, and X. Yang. LungReg-
470

Net: an unsupervised deformable image registration method for 4D-CT lung. Medical physics,
471

47(4):1763–1774, Apr. 2020.
472

[30] Y. Fu, Y. Lei, J. Zhou, T. Wang, S. Y. David, J. J. Beitler, W. J. Curran, T. Liu, and X. Yang.
473

Synthetic ct-aided mri-ct image registration for head and neck radiotherapy. In Medical
474

Imaging 2020: Biomedical Applications in Molecular, Structural, and Functional Imaging,
475

volume 11317, pages 572–578. SPIE, 2020.
476

[31] S. W. Fung, H. Heaton, Q. Li, D. McKenzie, S. Osher, and W. Yin. JFB: Jacobian-Free
477

Backpropagation for Implicit Networks, Dec. 2021. arXiv:2103.12803 [cs].
478

[32] J. C. Gee and R. K. Bajcsy. Elastic matching: Continuum mechanical and probabilistic analysis.
479

Brain warping, 2:183–197, 1998.
480

[33] J. C. Gee, M. Reivich, and R. Bajcsy. Elastically deforming a three-dimensional atlas to match
481

anatomical brain images. 1993.
482

[34] Z. Geng and J. Z. Kolter. TorchDEQ: A Library for Deep Equilibrium Models, Oct. 2023.
483

arXiv:2310.18605 [cs].
484

[35] Z. Geng, X.-Y. Zhang, S. Bai, Y. Wang, and Z. Lin. On training implicit models. Advances in
485

Neural Information Processing Systems, 34:24247–24260, 2021.
486

[36] A. Gholipour, N. Kehtarnavaz, R. Briggs, M. Devous, and K. Gopinath. Brain functional
487

localization: a survey of image registration techniques. IEEE transactions on medical imaging,
488

26(4):427–451, 2007.
489

[37] D. Gilton, G. Ongie, and R. Willett. Deep equilibrium architectures for inverse problems in
490

imaging. IEEE Transactions on Computational Imaging, 7:1123–1133, 2021.
491

[38] M. Goubran, C. Crukley, S. De Ribaupierre, T. M. Peters, and A. R. Khan. Image registration
492

of ex-vivo mri to sparsely sectioned histology of hippocampal and neocortical temporal lobe
493

specimens. Neuroimage, 83:770–781, 2013.
494

[39] U. Grenander and M. I. Miller. Computational anatomy: An emerging discipline. Quarterly of
495

applied mathematics, 56(4):617–694, 1998.
496

11


---Page Break---
[40] G. Haskins, J. Kruecker, U. Kruger, S. Xu, P. A. Pinto, B. J. Wood, and P. Yan. Learning deep
497

similarity metric for 3d mr–trus image registration. International journal of computer assisted
498

radiology and surgery, 14:417–425, 2019.
499

[41] G. Haskins, U. Kruger, and P. Yan. Deep learning in medical image registration: a survey.
500

Machine Vision and Applications, 31(1):8, Jan. 2020.
501

[42] A. Hering, L. Hansen, T. C. Mok, A. C. Chung, H. Siebert, S. Häger, A. Lange, S. Kuckertz,
502

S. Heldmann, W. Shao, et al. Learn2reg: comprehensive multi-task medical image registration
503

challenge, dataset and evaluation in the era of deep learning. IEEE Transactions on Medical
504

Imaging, 42(3):697–712, 2022.
505

[43] M. Hoffmann, B. Billot, D. N. Greve, J. E. Iglesias, B. Fischl, and A. V. Dalca. Synthmorph:
506

learning contrast-invariant registration without acquired images. IEEE transactions on medical
507

imaging, 41(3):543–558, 2021.
508

[44] A. Hoopes, M. Hoffmann, B. Fischl, J. Guttag, and A. V. Dalca. Hypermorph: Amortized
509

hyperparameter learning for image registration. In Information Processing in Medical Imaging:
510

27th International Conference, IPMI 2021, Virtual Event, June 28–June 30, 2021, Proceedings
511

27, pages 3–17. Springer, 2021.
512

[45] J. Hu, W. Gan, Z. Sun, H. An, and U. S. Kamilov. A Plug-and-Play Image Registration
513

Network, Mar. 2024. arXiv:2310.04297 [eess].
514

[46] R. Jena, P. Chaudhari, and J. C. Gee. Fireants: Adaptive riemannian optimization for multi-
515

scale diffeomorphic registration. arXiv preprint arXiv:2404.01249, 2024.
516

[47] Z. Ji and M. Telgarsky. Gradient descent aligns the layers of deep linear networks. arXiv
517

preprint arXiv:1810.02032, 2018.
518

[48] X. Jia, J. Bartlett, T. Zhang, W. Lu, Z. Qiu, and J. Duan. U-net vs transformer: Is u-net
519

outdated in medical image registration? arXiv preprint arXiv:2208.04939, 2022.
520

[49] A. Joshi and Y. Hong. Diffeomorphic Image Registration using Lipschitz Continuous Residual
521

Networks. page 13.
522

[50] M. L. Kessler. Image registration and data fusion in radiation therapy. The British journal of
523

radiology, 79(special_issue_1):S99–S108, 2006.
524

[51] B. Kim, D. H. Kim, S. H. Park, J. Kim, J.-G. Lee, and J. C. Ye. Cyclemorph: cycle consistent
525

unsupervised deformable image registration. Medical image analysis, 71:102036, 2021.
526

[52] B. Kim, J. Kim, J.-G. Lee, D. H. Kim, S. H. Park, and J. C. Ye. Unsupervised deformable image
527

registration using cycle-consistent cnn. In Medical Image Computing and Computer Assisted
528

Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17,
529

2019, Proceedings, Part VI 22, pages 166–174. Springer, 2019.
530

[53] A. Klein, J. Andersson, B. A. Ardekani, J. Ashburner, B. Avants, M.-C. Chiang, G. E. Chris-
531

tensen, D. L. Collins, J. Gee, P. Hellier, J. H. Song, M. Jenkinson, C. Lepage, D. Rueckert,
532

P. Thompson, T. Vercauteren, R. P. Woods, J. J. Mann, and R. V. Parsey. Evaluation of 14
533

nonlinear deformation algorithms applied to human brain MRI registration. NeuroImage,
534

46(3):786–802, July 2009.
535

[54] S. G. Krantz and H. R. Parks. The implicit function theorem: history, theory, and applications.
536

Springer Science & Business Media, 2002.
537

[55] J. Krebs, T. Mansi, H. Delingette, L. Zhang, F. C. Ghesu, S. Miao, A. K. Maier, N. Ayache,
538

R. Liao, and A. Kamen. Robust non-rigid registration through agent-based action learning.
539

In Medical Image Computing and Computer Assisted Intervention- MICCAI 2017: 20th
540

International Conference, Quebec City, QC, Canada, September 11-13, 2017, Proceedings,
541

Part I 20, pages 344–352. Springer, 2017.
542

[56] L. Lebrat, R. Santa Cruz, F. de Gournay, D. Fu, P. Bourgeat, J. Fripp, C. Fookes, and O. Salvado.
543

CorticalFlow: A Diffeomorphic Mesh Transformer Network for Cortical Surface Reconstruc-
544

tion. In Advances in Neural Information Processing Systems, volume 34, pages 29491–29505.
545

Curran Associates, Inc., 2021.
546

[57] F. Liu, K. Yan, A. P. Harrison, D. Guo, L. Lu, A. L. Yuille, L. Huang, G. Xie, J. Xiao, X. Ye,
547

and D. Jin. SAME: Deformable Image Registration Based on Self-supervised Anatomical
548

Embeddings. In M. de Bruijne, P. C. Cattin, S. Cotin, N. Padoy, S. Speidel, Y. Zheng, and
549

12


---Page Break---
C. Essert, editors, Medical Image Computing and Computer Assisted Intervention – MICCAI
550

2021, Lecture Notes in Computer Science, pages 87–97, Cham, 2021. Springer International
551

Publishing.
552

[58] J. Lv, Z. Wang, H. Shi, H. Zhang, S. Wang, Y. Wang, and Q. Li. Joint progressive and
553

coarse-to-fine registration of brain mri via deformation field integration and non-rigid feature
554

fusion. IEEE Transactions on Medical Imaging, 41(10):2788–2802, 2022.
555

[59] J. Ma, X. Jiang, A. Fan, J. Jiang, and J. Yan. Image matching from handcrafted to deep
556

features: A survey. International Journal of Computer Vision, 129(1):23–79, 2021.
557

[60] A. Mang, A. Gholami, C. Davatzikos, and G. Biros. CLAIRE: A distributed-memory solver for
558

constrained large deformation diffeomorphic image registration. SIAM Journal on Scientific
559

Computing, 41(5):C548–C584, Jan. 2019. arXiv:1808.04487 [cs, math].
560

[61] A. Mang and L. Ruthotto. A lagrangian gauss–newton–krylov solver for mass-and intensity-
561

preserving diffeomorphic image registration.
SIAM Journal on Scientific Computing,
562

39(5):B860–B885, 2017.
563

[62] D. S. Marcus, T. H. Wang, J. Parker, J. G. Csernansky, J. C. Morris, and R. L. Buckner.
564

Open access series of imaging studies (oasis): cross-sectional mri data in young, middle aged,
565

nondemented, and demented older adults. Journal of cognitive neuroscience, 19(9):1498–1507,
566

2007.
567

[63] M. I. Miller, A. Trouvé, and L. Younes. On the Metrics and Euler-Lagrange Equations of
568

Computational Anatomy. Annual Review of Biomedical Engineering, 4(1):375–405, 2002.
569

_eprint: https://doi.org/10.1146/annurev.bioeng.4.092101.125733.
570

[64] M. Modat, G. R. Ridgway, Z. A. Taylor, M. Lehmann, J. Barnes, D. J. Hawkes, N. C. Fox, and
571

S. Ourselin. Fast free-form deformation using graphics processing units. Computer methods
572

and programs in biomedicine, 98(3):278–284, 2010.
573

[65] T. C. Mok and A. Chung. Fast symmetric diffeomorphic image registration with convolutional
574

neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern
575

recognition, pages 4644–4653, 2020.
576

[66] T. C. Mok and A. Chung.
Affine medical image registration with coarse-to-fine vision
577

transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
578

Recognition, pages 20835–20844, 2022.
579

[67] T. C. Mok and A. C. Chung. Conditional deformable image registration with convolutional
580

neural network. pages 35–45, 2021.
581

[68] T. C. W. Mok and A. C. S. Chung. Large Deformation Diffeomorphic Image Registration with
582

Laplacian Pyramid Networks, June 2020. arXiv:2006.16148 [cs, eess].
583

[69] D. Moyer, E. Abaci Turk, P. E. Grant, W. M. Wells, and P. Golland. Equivariant filters
584

for efficient tracking in 3d imaging. In Medical Image Computing and Computer Assisted
585

Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September
586

27–October 1, 2021, Proceedings, Part IV 24, pages 193–202. Springer, 2021.
587

[70] K. Murphy, B. Van Ginneken, J. M. Reinhardt, S. Kabus, K. Ding, X. Deng, K. Cao, K. Du,
588

G. E. Christensen, V. Garcia, et al. Evaluation of registration methods on thoracic ct: the
589

empire10 challenge. IEEE transactions on medical imaging, 30(11):1901–1920, 2011.
590

[71] S. Oh and S. Kim. Deformable image registration in radiation therapy. Radiation oncology
591

journal, 35(2):101, 2017.
592

[72] H. Peng, P. Chung, F. Long, L. Qu, A. Jenett, A. M. Seeds, E. W. Myers, and J. H. Simpson.
593

Brainaligner: 3d registration atlases of drosophila brains. Nature methods, 8(6):493–498,
594

2011.
595

[73] J. Pérez de Frutos, A. Pedersen, E. Pelanis, D. Bouget, S. Survarachakan, T. Langø, O.-J. Elle,
596

and F. Lindseth. Learning deep abdominal ct registration through adaptive loss weighting and
597

synthetic data generation. Plos one, 18(2):e0282110, 2023.
598

[74] S. Pesme, L. Pillaud-Vivien, and N. Flammarion. Implicit bias of sgd for diagonal linear
599

networks: a provable benefit of stochasticity. Advances in Neural Information Processing
600

Systems, 34:29218–29230, 2021.
601

13


---Page Break---
[75] A. Pokle, Z. Geng, and J. Z. Kolter. Deep equilibrium approaches to diffusion models.
602

Advances in Neural Information Processing Systems, 35:37975–37990, 2022.
603

[76] Y. Qiao, B. P. Lelieveldt, and M. Staring. An efficient preconditioner for stochastic gra-
604

dient descent optimization of image registration. IEEE transactions on medical imaging,
605

38(10):2314–2325, 2019.
606

[77] C. Qin, S. Wang, C. Chen, W. Bai, and D. Rueckert. Generative Myocardial Motion Tracking
607

via Latent Space Exploration with Biomechanics-informed Prior, June 2022. arXiv:2206.03830
608

[cs, eess].
609

[78] C. Qin, S. Wang, C. Chen, H. Qiu, W. Bai, and D. Rueckert. Biomechanics-informed Neural
610

Networks for Myocardial Motion Tracking in MRI, July 2020. arXiv:2006.04725 [cs, eess].
611

[79] H. Qiu, C. Qin, A. Schuh, K. Hammernik, and D. Rueckert. Learning diffeomorphic and
612

modality-invariant registration using b-splines. 2021.
613

[80] L. Qu, F. Long, and H. Peng. 3-d registration of biological images and models: registration
614

of microscopic images and its uses in segmentation and annotation. IEEE Signal Processing
615

Magazine, 32(1):70–77, 2014.
616

[81] D. Quan, H. Wei, S. Wang, R. Lei, B. Duan, Y. Li, B. Hou, and L. Jiao. Self-distillation feature
617

learning network for optical and sar image registration. IEEE Transactions on Geoscience and
618

Remote Sensing, 60:1–18, 2022.
619

[82] M.-M. Rohé, M. Datar, T. Heimann, M. Sermesant, and X. Pennec. Svf-net: learning de-
620

formable image registration using shape matching. In Medical Image Computing and Com-
621

puter Assisted Intervention- MICCAI 2017: 20th International Conference, Quebec City, QC,
622

Canada, September 11-13, 2017, Proceedings, Part I 20, pages 266–274. Springer, 2017.
623

[83] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image
624

segmentation. In Medical image computing and computer-assisted intervention–MICCAI
625

2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part
626

III 18, pages 234–241. Springer, 2015.
627

[84] J. G. Rosenman, E. P. Miller, and T. J. Cullip. Image registration: an essential part of radiation
628

therapy treatment planning. International Journal of Radiation Oncology* Biology* Physics,
629

40(1):197–205, 1998.
630

[85] D. W. Shattuck, M. Mirza, V. Adisetiyo, C. Hojatkashani, G. Salamon, K. L. Narr, R. A.
631

Poldrack, R. M. Bilder, and A. W. Toga. Construction of a 3d probabilistic atlas of human
632

cortical structures. Neuroimage, 39(3):1064–1080, 2008.
633

[86] A. Siarohin. cuda-gridsample-grad2. GitHub Repository, 2023.
634

[87] H. Siebert, L. Hansen, and M. P. Heinrich. Fast 3d registration with accurate optimisation and
635

little learning for learn2reg 2021. In International Conference on Medical Image Computing
636

and Computer-Assisted Intervention, pages 174–179. Springer, 2021.
637

[88] H. Sokooti, B. De Vos, F. Berendsen, B. P. Lelieveldt, I. Išgum, and M. Staring. Nonrigid image
638

registration using multi-scale 3d convolutional neural networks. In Medical Image Computing
639

and Computer Assisted Intervention- MICCAI 2017: 20th International Conference, Quebec
640

City, QC, Canada, September 11-13, 2017, Proceedings, Part I 20, pages 232–239. Springer,
641

2017.
642

[89] J. H. Song, G. E. Christensen, J. A. Hawley, Y. Wei, and J. G. Kuhl. Evaluating image
643

registration using nirep. In Biomedical Image Registration: 4th International Workshop, WBIR
644

2010, Lübeck, Germany, July 11-13, 2010. Proceedings 4, pages 140–150. Springer, 2010.
645

[90] D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, and N. Srebro. The implicit bias of gradient
646

descent on separable data. Journal of Machine Learning Research, 19(70):1–57, 2018.
647

[91] Z. Teed and J. Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow, Aug.
648

2020. arXiv:2003.12039 [cs].
649

[92] L. Tian, H. Greer, F.-X. Vialard, R. Kwitt, R. S. J. Estépar, R. J. Rushmore, N. Makris,
650

S. Bouix, and M. Niethammer. Gradicon: Approximate diffeomorphisms via gradient inverse
651

consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
652

Recognition, pages 18084–18094, 2023.
653

14


---Page Break---
[93] L. Tian, Z. Li, F. Liu, X. Bai, J. Ge, L. Lu, M. Niethammer, X. Ye, K. Yan, and D. Jin. SAME++:
654

A Self-supervised Anatomical eMbeddings Enhanced medical image registration framework
655

using stable sampling and regularized transformation, Nov. 2023. arXiv:2311.14986 [cs].
656

[94] A. W. Toga and P. M. Thompson. The role of image registration in brain mapping. Image and
657

vision computing, 19(1-2):3–24, 2001.
658

[95] D. Ulyanov, A. Vedaldi, and V. Lempitsky. Deep Image Prior. International Journal of
659

Computer Vision, 128(7):1867–1888, July 2020. arXiv:1711.10925 [cs, stat].
660

[96] H. Uzunova, M. Wilms, H. Handels, and J. Ehrhardt. Training cnns for image registration
661

from few samples with model-based data augmentation. In Medical Image Computing and
662

Computer Assisted Intervention- MICCAI 2017: 20th International Conference, Quebec City,
663

QC, Canada, September 11-13, 2017, Proceedings, Part I 20, pages 223–231. Springer, 2017.
664

[97] D. C. Van Essen, H. A. Drury, S. Joshi, and M. I. Miller. Functional and structural mapping of
665

human cerebral cortex: solutions are in the surfaces. Proceedings of the National Academy of
666

Sciences, 95(3):788–795, 1998.
667

[98] E. Varol, A. Nejatbakhsh, R. Sun, G. Mena, E. Yemini, O. Hobert, and L. Paninski. Statistical
668

atlas of c. elegans neurons. In Medical Image Computing and Computer Assisted Intervention–
669

MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings,
670

Part V 23, pages 119–129. Springer, 2020.
671

[99] V. Venkatachalam, N. Ji, X. Wang, C. Clark, J. K. Mitchell, M. Klein, C. J. Tabone, J. Flor-
672

man, H. Ji, J. Greenwood, et al. Pan-neuronal imaging in roaming caenorhabditis elegans.
673

Proceedings of the National Academy of Sciences, 113(8):E1082–E1088, 2016.
674

[100] T. Vercauteren, X. Pennec, A. Perchant, and N. Ayache. Symmetric Log-Domain Diffeomor-
675

phic Registration: A Demons-Based Approach. In D. Metaxas, L. Axel, G. Fichtinger, and
676

G. Székely, editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI
677

2008, Lecture Notes in Computer Science, pages 754–761, Berlin, Heidelberg, 2008. Springer.
678

[101] T. Vercauteren, X. Pennec, A. Perchant, and N. Ayache. Diffeomorphic demons: Efficient
679

non-parametric image registration. NeuroImage, 45(1):S61–S72, Mar. 2009.
680

[102] T. Vercauteren, X. Pennec, A. Perchant, N. Ayache, et al. Diffeomorphic demons using itk’s
681

finite difference solver hierarchy. The Insight Journal, 1, 2007.
682

[103] A. Q. Wang, M. Y. Evan, A. V. Dalca, and M. R. Sabuncu. A robust and interpretable deep
683

learning framework for multi-modal registration via keypoints. Medical Image Analysis,
684

90:102962, 2023.
685

[104] Q. Wang, S.-L. Ding, Y. Li, J. Royall, D. Feng, P. Lesnar, N. Graddis, M. Naeemi, B. Facer,
686

A. Ho, T. Dolbeare, B. Blanchard, N. Dee, W. Wakeman, K. E. Hirokawa, A. Szafer, S. M.
687

Sunkin, S. W. Oh, A. Bernard, J. W. Phillips, M. Hawrylycz, C. Koch, H. Zeng, J. A. Harris,
688

and L. Ng. The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas.
689

Cell, 181(4):936–953.e20, May 2020.
690

[105] Y. Wang, X. Wei, F. Liu, J. Chen, Y. Zhou, W. Shen, E. K. Fishman, and A. L. Yuille. Deep
691

Distance Transform for Tubular Structure Segmentation in CT Scans. In 2020 IEEE/CVF
692

Conference on Computer Vision and Pattern Recognition (CVPR), pages 3832–3841, Seattle,
693

WA, USA, June 2020. IEEE.
694

[106] J. M. Wolterink, J. C. Zwienenberg, and C. Brune. Implicit Neural Representations for
695

Deformable Image Registration. page 11.
696

[107] G. Wu, M. Kim, Q. Wang, Y. Gao, S. Liao, and D. Shen. Unsupervised deep feature learning
697

for deformable registration of mr brain images. In Medical Image Computing and Computer-
698

Assisted Intervention–MICCAI 2013: 16th International Conference, Nagoya, Japan, Septem-
699

ber 22-26, 2013, Proceedings, Part II 16, pages 649–656. Springer, 2013.
700

[108] G. Wu, M. Kim, Q. Wang, B. C. Munsell, and D. Shen. Scalable high-performance image reg-
701

istration framework by unsupervised deep feature representations learning. IEEE transactions
702

on biomedical engineering, 63(7):1505–1516, 2015.
703

[109] J. Wu, D. Zou, V. Braverman, and Q. Gu. Direction matters: On the implicit bias of stochastic
704

gradient descent with moderate learning rate. arXiv preprint arXiv:2011.02538, 2020.
705

15


---Page Break---
[110] Y. Wu, T. Z. Jiahao, J. Wang, P. A. Yushkevich, M. A. Hsieh, and J. C. Gee. NODEO: A
706

Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image
707

Registration. arXiv:2108.03443 [cs], Feb. 2022. arXiv: 2108.03443.
708

[111] Z. Yang, T. Pang, and Y. Liu. A closer look at the adversarial robustness of deep equilibrium
709

models. Advances in Neural Information Processing Systems, 35:10448–10461, 2022.
710

[112] I. Yoo, D. G. Hildebrand, W. F. Tobin, W.-C. A. Lee, and W.-K. Jeong. ssemnet: Serial-section
711

electron microscopy image registration using a spatial transformer network with learned
712

features. pages 249–257, 2017.
713

[113] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning (still)
714

requires rethinking generalization. Communications of the ACM, 64(3):107–115, 2021.
715

[114] L. Zhang, L. Zhou, R. Li, X. Wang, B. Han, and H. Liao. Cascaded feature warping network
716

for unsupervised medical image registration. In 2021 IEEE 18th International Symposium on
717

Biomedical Imaging (ISBI), pages 913–916. IEEE, 2021.
718

[115] S. Zhao, Y. Dong, E. I.-C. Chang, and Y. Xu. Recursive cascaded networks for unsupervised
719

medical image registration. In Proceedings of the IEEE/CVF International Conference on
720

Computer Vision (ICCV), October 2019.
721

[116] S. Zhao, T. Lau, J. Luo, I. Eric, C. Chang, and Y. Xu. Unsupervised 3d end-to-end medical
722

image registration with volume tweening network. IEEE journal of biomedical and health
723

informatics, 24(5):1394–1404, 2019.
724

16


---Page Break---
A
Appendix
725

A.1
Implicit bias of optimization for registration
726

Model based systems, such as deep networks are not immune to inductive biases due to architecture,
727

loss functions, and optimization algorithms used to train them. Functional forms of the deep
728

network induce constraints on the solution space, but optimization algorithms are not excluded
729

from such biases either. The implicit bias for Gradient Descent is a well-studied phenomena for
730

overparameterized linear and shallow networks. Gradient Descent for linear systems leads to an
731

optimum that is in the span of the input data starting from the initialization [113, 90, 47, 74, 109].
732

This bias is also dependent on the chosen representation, since that defines the functional relationship
733

of the gradients with the parameters and inputs. This limits the reachable set of solutions by the
734

optimization algorithm when multiple local minima exist.
735

In the case of image registration, the optimization limits the space of solutions (warps) that can be
736

obtained by the SGD algorithm. To show this, we consider the transformation φ as a set of particles
737

in a Langrangian frame that are displaced by the optimization algorithm to align the moving image to
738

the fixed image. Consider a regular grid of particles, whose locations specify the warp field. Let the
739

location of i-th particle at iteration t be φ(t)(xi). For a fixed feature image Ff, moving image Fm and
740

current iterate φ(t), the gradient of the registration loss with respect to particle i at iteration t is given
741

by
742

∂C(Ff, Fm ◦φ(t))

∂φ(t)(xi)
= C′
i(Ff, Fm ◦φ(t))∇Fm(φ(t)(xi))
(5)

where

C′
i(Ff, Fm ◦φ(t)) = ∂C(Ff, Fm ◦φ(t))

∂M(φ(t)(xi))
is the (scalar) derivative of scalar loss C with respect to the intensity of i-th particle computed at
743

the current iterate, and ∇Fm(φ(t)(xi)) is the spatial gradient of the moving image at the location of
744

the particle. Note that the direction of the gradient of particle i is independent of the fixed image,
745

loss function, and location of other particles – it only depends on the spatial gradient of the moving
746

image at the location of the particle. This restricts the movement of a particle located at any given
747

location along a 1D line whose direction is the spatial gradient of the moving image at that location.
748

Since Ff and Fm are computed independently of each other (and therefore no information of Ff and
749

Fm is contained in each other), the space of solutions of φ is restricted by this implicit bias. This
750

is restrictive because the similarity function and fixed image do not influence the direction of the
751

gradient, and the optimization algorithm is biased towards solutions that are in the direction of the
752

gradient of the moving image.
753

0
50
100
150
200
Iterations

0.2

0.4

0.6

0.8

1.0

Loss

Optimization at scale = 4x

0
20
40
60
80
100
Iterations

0.0

0.2

0.4

0.6

0.8

1.0

Loss

Optimization at scale = 2x

0
10
20
30
40
50
Iterations

0.2

0.4

0.6

0.8

1.0

Loss

Optimization at scale = 1x

0.996

0.998

1.000

1.002

1.004

Cosine similarity

0.996

0.998

1.000

1.002

1.004

Cosine similarity

0.996

0.998

1.000

1.002

1.004

Cosine similarity

Figure 5: Implicit bias in SGD for image registration. The plot shows the loss curves for a
multi-scale optimization of two feature images. Each plot also shows the absolute cosine similarity
of per-pixel gradients obtained by C and Csurrogate at each iteration. Note that over the course of
optimization, the cosine similarity is always 1 – demonstrating the implicit bias of the optimization
for registration.

We show this bias empirically – we perform multi-scale optimization algorithm using feature maps
754

obtained from the network. We keep track of two gradients, one obtained by the loss function, and
755

another obtained by the gradient of a surrogate loss Csurrogate(Fm, φ(t)) = P
i Fm(φ(t)(xi)). Note
756

that Csurrogate does not depend on the fixed image or the loss function. The gradient of Csurrogate with
757

17


---Page Break---
respect to the i-th particle is given by ∇Fm(φ(t)(xi)). At each iteration, we compute the magnitude
758

of cosine similarly between the gradients of C and Csurrogate. Fig. 5 shows that the loss converges, and
759

the per-pixel gradients can be predicted by Csurrogate alone, as depicted by the magnitude and standard
760

deviation of cosine similarity between C and Csurrogate. This limits the movement of each particle
761

along a 1D line in an N-D space, and limits the degrees of freedom of the optimization by N-fold
762

for N-D images. Future work will aim at alleviating this implicit bias to allow for more flexible
763

solutions.
764

A.2
Algorithm details
765

DIO is a learnable framework that leverages implicit differentiation of an arbitrary black-box optimiza-
766

tion solver to learn features such that registration in this feature space corresponds to good registration
767

of the images and additional label maps. This additional indirection leads to learnable features that
768

are registration-aware, interpretable, and the framework inherits the optimization solver’s versatility
769

to variability in the data like difference in contrast, anisotropy, and difference in sizes of the fixed and
770

moving images. We contrast our approach with a typical classical optimization-based registration
771

algorithm in Fig. 6. A classical multi-scale optimization routine indiscriminately downsamples the
772

intensity images, and does not retain discriminative information that is useful for registration. Since
773

our method is trained to maximize label alignment from all scales, multi-scale features obtained from
774

our method are more discriminative and registration-aware. We also compare DIO with a typical
775

DLIR method in Fig. 7. Note that the fixed end-to-end architecture and functional form of a deep
776

network subsumes the representation choice into the architecture as well, limiting its ability to switch
777

to arbitrary transformation representations at inference time without additional retraining. Our frame-
778

work therefore combines the benefits of both classical (robustness to out-of-distribution datasets,
779

and zero-shot transfer to other optimization routines) and learning-based methods (high-fidelity,
780

label-aware, and registration-aware).
781

A.3
Implementation Details
782

For all experiments, we use downsampling scales of 1, 2, 4 for the multi-scale optimization. All our
783

methods are implemented in PyTorch, and use the Adam optimizer for learning the parameters of the
784

feature network. Note that in Eq. (3), ϱ is the partial derivative of the loss function C with respect
785

to the transformation φ, which contains a ∇(Fm ◦φ) term, which is the backward transform of the
786

grid_sample operator in PyTorch. Since this operation is not implemented using PyTorch primitives,
787

a backward pass for the gradient operation does not exist in PyTorch. We use the gridsample_grad2
788

library [86] to compute the gradients of the backward pass of the grid_sample operator, used in
789

Eq. (3). All experiments are performed on a single NVIDIA A6000 GPU.
790

A.4
Toy example
791

Fig. 8 shows the loss curves for the toy dataset described in Section 4.1. An image-based optimization
792

algorithm would correspond to the green curve being a flat line at 1 due to the flat landscape of the
793

intensity-based loss function.
794

A.5
Quantitative Results
795

Table 4 shows the quantitative results of our method for out-of-distribution performance on the
796

IBSR18, CUMC12, and LPBA40 datasets. In 9 out of 10 cases, DIO demonstrates the best accuracy
797

with fairly lower standard deviations, highlighting the robustness of the model. DIO therefore serves
798

as a strong candidate for out-of-distribution performance, and can be used in a variety of settings
799

where the training and test distributions differ.
800

A.6
Datasets
801

We consider four brain MRI datasets in this paper: OASIS dataset for in-distribution performance,
802

and LPBA40, IBSR18, and CUMC12 datasets for out-of-distribution performance [85, 1, 53, 62].
803

More details about the datasets are provided below.
804

• OASIS. The Open Access Series of Imaging Studies (OASIS) dataset contains 414 T1-weighted
805

brain images in Young, Middle Aged, Nondemented, and Demented Older adults. The images are
806

skull-stripped and bias-corrected, followed by a resampling and afine alignment to the FreeSurfer’s
807

Talairach atlas. Label segmentations of 35 subcortical structures were obtained using automatic
808

segmentation using Freesurfer software.
809

18


---Page Break---
Algorithm 1 Classical registration pipeline

1: Input: Fixed image If, Moving image Im
2: Scales [s1, s2, . . . , sn], Iterations [T1, T2, . . . Tn], n levels.
3: Initialize φ = Ids1.
▷Initialize warp to identity at first scale
4: Initialize l = 1.
▷Initialize current scale
5: while l ≤n do
6:
Initialize i = 0
7:
Initialize Il
f, Ilm = downsample(If, sl), downsample(Im, sl)
8:
while i < Tl do
9:
Li = C(Il
f, Ilm ◦φi)
10:
Compute ∇φL
11:
Update φ(i+1) = Optimize(φi, ∇φLi)
▷Optimization algorithm
12:
i = i + 1
13:
end while
14:
if l < n then
15:
φ = Upsample(φ, s(l+1))
▷Upsample warp to next level
16:
end if
17:
l = l + 1
18: end while

Algorithm 2 Differentiable Implicit Optimization for Registration (Our algorithm)

1: Input: Fixed features Ff = [F 1
f , F 2
f . . . F n
f ], Moving features Ff = [F 1
f , F 2
f . . . F n
f ]
2: Scales [s1, s2, . . . , sn], Iterations [T1, T2, . . . Tn], n levels.
3: Initialize φ = Ids1.
▷Initialize warp to identity at first scale
4: Initialize l = 1.
▷Initialize current scale
5: Outputs = [].
▷Save intermediate outputs for backpropagation
6: while l ≤n do
7:
Initialize i = 0
8:
Initialize Il
f, Ilm = F l
f, F lm
9:
while i < Tl do
10:
Li = C(Il
f, Ilm ◦φi)
11:
Compute ∇φL
12:
Update φ(i+1) = Optimize(φi, ∇φLi)
▷Optimization algorithm
13:
i = i + 1
14:
end while
15:
Outputs.append

φ(Tl)

▷Save final warp at this level for backpropagation

16:
if l < n then
17:
φ = Upsample(φ, s(l+1))
▷Upsample warp for next level
18:
end if
19:
l = l + 1
20: end while

Figure 6: Comparison of a typical classical registration algorithm and DIO: Algorithm 1 shows
a typical classical registration algorithm that uses a multi-scale optimization routine to register the
fixed and moving images. At each level l, the fixed and moving images are downsampled by a factor
of sl, therefore trading off between discriminative information and vulnerability to local minima.
Algorithm 2 shows our algorithm (red text highlights differences compared to Algorithm 1) that uses
a separate scale-space feature at each level. Unlike classical methods, the scale-space feature can
capture different discriminative features at each level to maximize label alignment and the multi-scale
nature helps avoid local minima.

• LPBA40. 40 brain images and their labels are used to construct the LONI Probabilistic Brain Atlas
810

(LPBA40) dataset at the Laboratory of Neuroimaging (LONI) at UCLA [85]. All volumes are
811

preprocessed according to LONI protocols to produce skull-stripped volumes. These volumes are
812

aligned to the MNI305 atlas – this is relevant since existing DLIR methods may be biased towards
813

19


---Page Break---
Fixed and moving images
Deep network
Displacement field
Output
Parameterization

Fixed and moving images
Deep network
Feature images
Optimization layer (switchable)
Displacement 

field

(a)

(b)

🔥

Figure 7: Comparison of typical DLIR method and our method. (a) shows the pipeline of a typical
deep network. The neural network architecture takes the channelwise concatenation of the fixed and
moving images as input, and outputs a warp field, which has a fixed transformation representation
(SVF, free-form, B-splines, affine, etc. denoted as the blue locked layer). This representation is
fixed throughout training and cannot be switched at test-time, without additional finetuning of the
network. (b) shows our framework wherein the fixed and moving images are input separately into a
feature extraction network that outputs multi-scale features. These features are then passed onto an
iterative black-box solver than can be implicitly differentiated to backpropagate the gradients from
the optimized warp field back to the feature network. This allows for a more flexible transformation
representation, and the optimization solver can be switched at test-time with zero finetuning.

images that are aligned to the Talairach and Tournoux (1988) atlas which is used to align the images
814

in the OASIS dataset. This is followed by a custom manual labelling protocol of 56 structures from
815

each of the volumes. Bias correction is perfrmed using the BrainSuite’s Bias Field Corrector.
816

• IBSR18. the Internet Brain Segmentation Repository contains 18 different brain images acquired
817

at different laboratories as IBSRv2.0. The dataset consists of T1-weighted brains aligned to the
818

Talairach and Tournoux (1988) atlas, and manually segmented into 84 labelled regions. Bias
819

correction of the images are performed using the ‘autoseg’ bias field correction algorithm.
820

• CUMC12. The Columbia University Medical Center dataset contains 12 T1-weighted brain images
821

with manual segmentation of 128 regions. The images were scanned on a 1.5T GE scanner, and the
822

images were resliced coronally to a slice thickness of 3mm, rotated into cardinal orientation, and
823

segmented by a technician trained according to the Cardviews labelling scheme.
824

20


---Page Break---
0
5000
10000
15000
20000
25000
30000
Iteration

10
3

10
2

10
1

Dice Loss

Validation score on toy square alignment task

All Pairs
Overlapping pairs
Nonoverlapping pairs

Figure 8: Loss curves for toy dataset. Plot shows three curves - the Dice score for (a) all validation
image pairs, (b) image pairs that have non-zero overlap in the image space (therefore a gradient-based
affine solver will recover a transform from intensity images), and (c) image pairs that have zero
overlap in the image space (therefore any gradient-based solver using intensity images will fail).
Our feature network recovers dense multi-scale features (see Fig. 2) which allows all subsets to be
registered with >0.99 Dice score.

21


---Page Break---
Method
Dice
Isotropic
Anisotropic
supervision
Crop
No Crop
Crop
No Crop
Conditional LapIRN
✗
0.7367 ± 0.0237
✗
0.7269 ± 0.0328
0.7317 ± 0.0303
LapIRN
✗
0.5257 ± 0.1316
✗
0.5435 ± 0.1266
0.5001 ± 0.1271
LapIRN
✓
0.6259 ± 0.1238
✗
0.6209 ± 0.1163
0.5759 ± 0.1207
LKU-Net
✗
0.6309 ± 0.0839
✗
0.6276 ± 0.0838
0.6072 ± 0.0787
LKU-Net
✓
0.6267 ± 0.0776
✗
0.6231 ± 0.0730
0.5992 ± 0.0757
SymNet
✗
0.7213 ± 0.0273
✗
0.7116 ± 0.0398
0.7117 ± 0.0398
SymNet
✓
0.6731 ± 0.0688
✗
0.6672 ± 0.0731
0.6674 ± 0.0728
TransMorph Large
✓
0.7383 ± 0.0353
✗
0.7312 ± 0.0405
✗
TransMorph Regular
✗
0.7221 ± 0.0400
✗
0.7289 ± 0.0417
✗
TransMorph Regular
✓
0.7293 ± 0.0370
✗
0.7113 ± 0.0520
✗
VoxelMorph
✗
0.5118 ± 0.1774
✗
0.5233 ± 0.1693
✗
SynthMorph
✓
0.7423 ± 0.0225
✗
0.7476 ± 0.0238
✗
Ours (LKU)
✓
0.7698 ± 0.0193
0.7587 ± 0.0208
0.7728 ± 0.0219
0.7572 ± 0.0369
Conditional LapIRN
✗
0.4793 ± 0.0373
0.4804 ± 0.0368
0.4880 ± 0.0416
0.4827 ± 0.0408
LapIRN
✗
0.3719 ± 0.0897
0.3491 ± 0.0895
0.3524 ± 0.1001
0.3556 ± 0.0989
LapIRN
✓
0.4121 ± 0.0907
0.3838 ± 0.0929
0.3911 ± 0.1060
0.3896 ± 0.1063
LKU-Net
✗
0.4054 ± 0.0641
0.3922 ± 0.0679
0.4086 ± 0.0732
0.3999 ± 0.0697
LKU-Net
✓
0.3904 ± 0.0547
0.3827 ± 0.0574
0.3967 ± 0.0745
0.3960 ± 0.0678
SymNet
✗
0.4761 ± 0.0524
0.4761 ± 0.0524
0.4822 ± 0.0565
0.4820 ± 0.0565
SymNet
✓
0.4457 ± 0.0675
0.4457 ± 0.0675
0.4518 ± 0.0787
0.4521 ± 0.0786
TransMorph Large
✓
0.4827 ± 0.0531
✗
0.4858 ± 0.0587
✗
TransMorph Regular
✗
0.4929 ± 0.0502
✗
0.4967 ± 0.0540
✗
TransMorph Regular
✓
0.4737 ± 0.0549
✗
0.4741 ± 0.0628
✗
VoxelMorph
✗
0.3519 ± 0.1271
✗
0.3469 ± 0.1308
✗
SynthMorph
✓
0.4761 ± 0.0397
✗
0.4797 ± 0.0426
✗
Ours (LKU)
✓
0.5137 ± 0.0410
0.5126 ± 0.0412
0.5237 ± 0.0433
0.5162 ± 0.0448
Conditional LapIRN
✗
0.7113 ± 0.0178
0.7109 ± 0.0178
-
-
LapIRN
✗
0.6026 ± 0.0317
0.5878 ± 0.0325
-
-
LapIRN
✓
0.6395 ± 0.0269
0.6211 ± 0.0294
-
-
LKU-Net
✗
0.6746 ± 0.0230
0.6708 ± 0.0249
-
-
LKU-Net
✓
0.6266 ± 0.0299
0.6220 ± 0.0296
-
-
SymNet
✗
0.6797 ± 0.0239
0.6797 ± 0.0238
-
-
SymNet
✓
0.6700 ± 0.0248
0.6698 ± 0.0248
-
-
TransMorph Large
✓
0.6918 ± 0.0219
✗
-
-
TransMorph Regular
✗
0.6919 ± 0.0191
✗
-
-
TransMorph Regular
✓
0.6855 ± 0.0225
✗
-
-
VoxelMorph
✗
0.6776 ± 0.0365
✗
-
-
SynthMorph
✓
0.7189 ± 0.0172
✗
-
-
Ours (LKU)
✓
0.7139 ± 0.0181
0.7131 ± 0.0181
-
-
Table 4: Quantitative evaluation on out-of-distribution performance on IBSR18, CUMC12,
and LPBA40 datasets. We compare DIO with other state-of-the-art DLIR methods. The ‘Dice
supervision’ column shows if the method is trained with label matching on the OASIS dataset. We
evaluate the performance of the methods with and without isotropic and anisotropic data resampling.
The results are reported as mean ± standard deviation.
= First,
= Second,
= Third best
result.

22


---Page Break---
(a)

(b)

= Input image

= Encoder
    feature

= Decoder
    feature

= Output 
    feature
    = skip

    connection

= single conv
    layer

Figure 9: Architecture details. (a) illustrates the UNet and Large Kernel U-Net (LKUNet) archi-
tecture designs, which consists of encoder blocks (red) and decoder blocks (purple) linked using
skip connections. Multi-scale features are extracted from the intermediate decoder layers using a
single convolutional layer. This design leads to shared features across multiple scales. UNet and
LKUNet differ in the kernel parameters within each encoder and decoder blocks. (b) illustrates the
‘Encoder-Only’ versions of the same networks. The decoder path is entirely discarded, and each
feature image is extracted using a separate encoder. This design enables independent learning of each
multi-scale feature.

23


---Page Break---
NeurIPS Paper Checklist
825

1. Claims
826

Question: Do the main claims made in the abstract and introduction accurately reflect the
827

paper’s contributions and scope?
828

Answer: [Yes]
829

Justification: Yes. Experiments are shown on community-standard, out-of-distribution
830

datasets for demonstrating robustness. Zero-shot performance by switching optimizers at
831

test time is shown.
832

Guidelines:
833

• The answer NA means that the abstract and introduction do not include the claims
834

made in the paper.
835

• The abstract and/or introduction should clearly state the claims made, including the
836

contributions made in the paper and important assumptions and limitations. A No or
837

NA answer to this question will not be perceived well by the reviewers.
838

• The claims made should match theoretical and experimental results, and reflect how
839

much the results can be expected to generalize to other settings.
840

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
841

are not attained by the paper.
842

2. Limitations
843

Question: Does the paper discuss the limitations of the work performed by the authors?
844

Answer: [Yes]
845

Justification: An implicit bias of the representation and optimization algorithm is discussed
846

in the Discussion and Appendix.
847

Guidelines:
848

• The answer NA means that the paper has no limitation while the answer No means that
849

the paper has limitations, but those are not discussed in the paper.
850

• The authors are encouraged to create a separate "Limitations" section in their paper.
851

• The paper should point out any strong assumptions and how robust the results are to
852

violations of these assumptions (e.g., independence assumptions, noiseless settings,
853

model well-specification, asymptotic approximations only holding locally). The authors
854

should reflect on how these assumptions might be violated in practice and what the
855

implications would be.
856

• The authors should reflect on the scope of the claims made, e.g., if the approach was
857

only tested on a few datasets or with a few runs. In general, empirical results often
858

depend on implicit assumptions, which should be articulated.
859

• The authors should reflect on the factors that influence the performance of the approach.
860

For example, a facial recognition algorithm may perform poorly when image resolution
861

is low or images are taken in low lighting. Or a speech-to-text system might not be
862

used reliably to provide closed captions for online lectures because it fails to handle
863

technical jargon.
864

• The authors should discuss the computational efficiency of the proposed algorithms
865

and how they scale with dataset size.
866

• If applicable, the authors should discuss possible limitations of their approach to
867

address problems of privacy and fairness.
868

• While the authors might fear that complete honesty about limitations might be used by
869

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
870

limitations that aren’t acknowledged in the paper. The authors should use their best
871

judgment and recognize that individual actions in favor of transparency play an impor-
872

tant role in developing norms that preserve the integrity of the community. Reviewers
873

will be specifically instructed to not penalize honesty concerning limitations.
874

3. Theory Assumptions and Proofs
875

Question: For each theoretical result, does the paper provide the full set of assumptions and
876

a complete (and correct) proof?
877

24


---Page Break---
Answer: [Yes]
878

Justification: Only Implicit Function Theorem is used with all its assumptions.
879

Guidelines:
880

• The answer NA means that the paper does not include theoretical results.
881

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
882

referenced.
883

• All assumptions should be clearly stated or referenced in the statement of any theorems.
884

• The proofs can either appear in the main paper or the supplemental material, but if
885

they appear in the supplemental material, the authors are encouraged to provide a short
886

proof sketch to provide intuition.
887

• Inversely, any informal proof provided in the core of the paper should be complemented
888

by formal proofs provided in appendix or supplemental material.
889

• Theorems and Lemmas that the proof relies upon should be properly referenced.
890

4. Experimental Result Reproducibility
891

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
892

perimental results of the paper to the extent that it affects the main claims and/or conclusions
893

of the paper (regardless of whether the code and data are provided or not)?
894

Answer: [Yes]
895

Justification: Code contains scripts to reproduce all experiments of the paper. Appendix con-
896

tains algorithm details. Code will be published to Github upon acceptance, with additional
897

documentation, tutorials and instructions. Data is publicly available.
898

Guidelines:
899

• The answer NA means that the paper does not include experiments.
900

• If the paper includes experiments, a No answer to this question will not be perceived
901

well by the reviewers: Making the paper reproducible is important, regardless of
902

whether the code and data are provided or not.
903

• If the contribution is a dataset and/or model, the authors should describe the steps taken
904

to make their results reproducible or verifiable.
905

• Depending on the contribution, reproducibility can be accomplished in various ways.
906

For example, if the contribution is a novel architecture, describing the architecture fully
907

might suffice, or if the contribution is a specific model and empirical evaluation, it may
908

be necessary to either make it possible for others to replicate the model with the same
909

dataset, or provide access to the model. In general. releasing code and data is often
910

one good way to accomplish this, but reproducibility can also be provided via detailed
911

instructions for how to replicate the results, access to a hosted model (e.g., in the case
912

of a large language model), releasing of a model checkpoint, or other means that are
913

appropriate to the research performed.
914

• While NeurIPS does not require releasing code, the conference does require all submis-
915

sions to provide some reasonable avenue for reproducibility, which may depend on the
916

nature of the contribution. For example
917

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
918

to reproduce that algorithm.
919

(b) If the contribution is primarily a new model architecture, the paper should describe
920

the architecture clearly and fully.
921

(c) If the contribution is a new model (e.g., a large language model), then there should
922

either be a way to access this model for reproducing the results or a way to reproduce
923

the model (e.g., with an open-source dataset or instructions for how to construct
924

the dataset).
925

(d) We recognize that reproducibility may be tricky in some cases, in which case
926

authors are welcome to describe the particular way they provide for reproducibility.
927

In the case of closed-source models, it may be that access to the model is limited in
928

some way (e.g., to registered users), but it should be possible for other researchers
929

to have some path to reproducing or verifying the results.
930

5. Open access to data and code
931

25


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
932

tions to faithfully reproduce the main experimental results, as described in supplemental
933

material?
934

Answer: [Yes]
935

Justification: Code is provided in the supplemental material. Data is publicly available and
936

instructions are provided in the supplemental material.
937

Guidelines:
938

• The answer NA means that paper does not include experiments requiring code.
939

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
940

public/guides/CodeSubmissionPolicy) for more details.
941

• While we encourage the release of code and data, we understand that this might not be
942

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
943

including code, unless this is central to the contribution (e.g., for a new open-source
944

benchmark).
945

• The instructions should contain the exact command and environment needed to run to
946

reproduce the results. See the NeurIPS code and data submission guidelines (https:
947

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
948

• The authors should provide instructions on data access and preparation, including how
949

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
950

• The authors should provide scripts to reproduce all experimental results for the new
951

proposed method and baselines. If only a subset of experiments are reproducible, they
952

should state which ones are omitted from the script and why.
953

• At submission time, to preserve anonymity, the authors should release anonymized
954

versions (if applicable).
955

• Providing as much information as possible in supplemental material (appended to the
956

paper) is recommended, but including URLs to data and code is permitted.
957

6. Experimental Setting/Details
958

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
959

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
960

results?
961

Answer: [Yes]
962

Justification: Implementation details are provided in Appendix and supplemental material.
963

Guidelines:
964

• The answer NA means that the paper does not include experiments.
965

• The experimental setting should be presented in the core of the paper to a level of detail
966

that is necessary to appreciate the results and make sense of them.
967

• The full details can be provided either with the code, in appendix, or as supplemental
968

material.
969

7. Experiment Statistical Significance
970

Question: Does the paper report error bars suitably and correctly defined or other appropriate
971

information about the statistical significance of the experiments?
972

Answer: [Yes]
973

Justification: All results are reported either with an error bar of one standard deviation, or
974

boxplots with interquartile ranges and outliers are reported.
975

Guidelines:
976

• The answer NA means that the paper does not include experiments.
977

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
978

dence intervals, or statistical significance tests, at least for the experiments that support
979

the main claims of the paper.
980

• The factors of variability that the error bars are capturing should be clearly stated (for
981

example, train/test split, initialization, random drawing of some parameter, or overall
982

run with given experimental conditions).
983

26


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
984

call to a library function, bootstrap, etc.)
985

• The assumptions made should be given (e.g., Normally distributed errors).
986

• It should be clear whether the error bar is the standard deviation or the standard error
987

of the mean.
988

• It is OK to report 1-sigma error bars, but one should state it. The authors should
989

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
990

of Normality of errors is not verified.
991

• For asymmetric distributions, the authors should be careful not to show in tables or
992

figures symmetric error bars that would yield results that are out of range (e.g. negative
993

error rates).
994

• If error bars are reported in tables or plots, The authors should explain in the text how
995

they were calculated and reference the corresponding figures or tables in the text.
996

8. Experiments Compute Resources
997

Question: For each experiment, does the paper provide sufficient information on the com-
998

puter resources (type of compute workers, memory, time of execution) needed to reproduce
999

the experiments?
1000

Answer: [Yes]
1001

Justification: Compute resources are provided in the Appendix.
1002

Guidelines:
1003

• The answer NA means that the paper does not include experiments.
1004

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
1005

or cloud provider, including relevant memory and storage.
1006

• The paper should provide the amount of compute required for each of the individual
1007

experimental runs as well as estimate the total compute.
1008

• The paper should disclose whether the full research project required more compute
1009

than the experiments reported in the paper (e.g., preliminary or failed experiments that
1010

didn’t make it into the paper).
1011

9. Code Of Ethics
1012

Question: Does the research conducted in the paper conform, in every respect, with the
1013

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
1014

Answer: [Yes]
1015

Justification: No research is performed involving new human subjects, animals, or environ-
1016

mental impact. Existing datasets comply with Code of Ethics. The proposed research is
1017

theoretical and computational. The proposed research has no immediate negative societal
1018

impact.
1019

Guidelines:
1020

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
1021

• If the authors answer No, they should explain the special circumstances that require a
1022

deviation from the Code of Ethics.
1023

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
1024

eration due to laws or regulations in their jurisdiction).
1025

10. Broader Impacts
1026

Question: Does the paper discuss both potential positive societal impacts and negative
1027

societal impacts of the work performed?
1028

Answer: [No]
1029

Justification: Medical image registration has no immediate negative societal impact necessi-
1030

tating a dedicated discussion.
1031

Guidelines:
1032

• The answer NA means that there is no societal impact of the work performed.
1033

27


---Page Break---
• If the authors answer NA or No, they should explain why their work has no societal
1034

impact or why the paper does not address societal impact.
1035

• Examples of negative societal impacts include potential malicious or unintended uses
1036

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
1037

(e.g., deployment of technologies that could make decisions that unfairly impact specific
1038

groups), privacy considerations, and security considerations.
1039

• The conference expects that many papers will be foundational research and not tied
1040

to particular applications, let alone deployments. However, if there is a direct path to
1041

any negative applications, the authors should point it out. For example, it is legitimate
1042

to point out that an improvement in the quality of generative models could be used to
1043

generate deepfakes for disinformation. On the other hand, it is not needed to point out
1044

that a generic algorithm for optimizing neural networks could enable people to train
1045

models that generate Deepfakes faster.
1046

• The authors should consider possible harms that could arise when the technology is
1047

being used as intended and functioning correctly, harms that could arise when the
1048

technology is being used as intended but gives incorrect results, and harms following
1049

from (intentional or unintentional) misuse of the technology.
1050

• If there are negative societal impacts, the authors could also discuss possible mitigation
1051

strategies (e.g., gated release of models, providing defenses in addition to attacks,
1052

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
1053

feedback over time, improving the efficiency and accessibility of ML).
1054

11. Safeguards
1055

Question: Does the paper describe safeguards that have been put in place for responsible
1056

release of data or models that have a high risk for misuse (e.g., pretrained language models,
1057

image generators, or scraped datasets)?
1058

Answer: [NA]
1059

Justification: [NA]
1060

Guidelines:
1061

• The answer NA means that the paper poses no such risks.
1062

• Released models that have a high risk for misuse or dual-use should be released with
1063

necessary safeguards to allow for controlled use of the model, for example by requiring
1064

that users adhere to usage guidelines or restrictions to access the model or implementing
1065

safety filters.
1066

• Datasets that have been scraped from the Internet could pose safety risks. The authors
1067

should describe how they avoided releasing unsafe images.
1068

• We recognize that providing effective safeguards is challenging, and many papers do
1069

not require this, but we encourage authors to take this into account and make a best
1070

faith effort.
1071

12. Licenses for existing assets
1072

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
1073

the paper, properly credited and are the license and terms of use explicitly mentioned and
1074

properly respected?
1075

Answer: [Yes]
1076

Justification: Appropriate citations are provided for existing code and data.
1077

Guidelines:
1078

• The answer NA means that the paper does not use existing assets.
1079

• The authors should cite the original paper that produced the code package or dataset.
1080

• The authors should state which version of the asset is used and, if possible, include a
1081

URL.
1082

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
1083

• For scraped data from a particular source (e.g., website), the copyright and terms of
1084

service of that source should be provided.
1085

28


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the
1086

package should be provided. For popular datasets, paperswithcode.com/datasets
1087

has curated licenses for some datasets. Their licensing guide can help determine the
1088

license of a dataset.
1089

• For existing datasets that are re-packaged, both the original license and the license of
1090

the derived asset (if it has changed) should be provided.
1091

• If this information is not available online, the authors are encouraged to reach out to
1092

the asset’s creators.
1093

13. New Assets
1094

Question: Are new assets introduced in the paper well documented and is the documentation
1095

provided alongside the assets?
1096

Answer: [Yes]
1097

Justification: Code is reasonably commented for a new reader to understand the implemen-
1098

tation.
1099

Guidelines:
1100

• The answer NA means that the paper does not release new assets.
1101

• Researchers should communicate the details of the dataset/code/model as part of their
1102

submissions via structured templates. This includes details about training, license,
1103

limitations, etc.
1104

• The paper should discuss whether and how consent was obtained from people whose
1105

asset is used.
1106

• At submission time, remember to anonymize your assets (if applicable). You can either
1107

create an anonymized URL or include an anonymized zip file.
1108

14. Crowdsourcing and Research with Human Subjects
1109

Question: For crowdsourcing experiments and research with human subjects, does the paper
1110

include the full text of instructions given to participants and screenshots, if applicable, as
1111

well as details about compensation (if any)?
1112

Answer: [NA]
1113

Justification: [NA]
1114

Guidelines:
1115

• The answer NA means that the paper does not involve crowdsourcing nor research with
1116

human subjects.
1117

• Including this information in the supplemental material is fine, but if the main contribu-
1118

tion of the paper involves human subjects, then as much detail as possible should be
1119

included in the main paper.
1120

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1121

or other labor should be paid at least the minimum wage in the country of the data
1122

collector.
1123

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1124

Subjects
1125

Question: Does the paper describe potential risks incurred by study participants, whether
1126

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1127

approvals (or an equivalent approval/review based on the requirements of your country or
1128

institution) were obtained?
1129

Answer: [NA]
1130

Justification: [NA]
1131

Guidelines:
1132

• The answer NA means that the paper does not involve crowdsourcing nor research with
1133

human subjects.
1134

• Depending on the country in which research is conducted, IRB approval (or equivalent)
1135

may be required for any human subjects research. If you obtained IRB approval, you
1136

should clearly state this in the paper.
1137

29


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
1138

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1139

guidelines for their institution.
1140

• For initial submissions, do not include any information that would break anonymity (if
1141

applicable), such as the institution conducting the review.
1142

30


---Page Break---
