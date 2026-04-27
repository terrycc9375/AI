NeCGS: Neural Compression for 3D Geometry Sets

Anonymous Author(s)
Affiliation
Address
email

Abstract

This paper explores the problem of effectively compressing 3D geometry sets
1

containing diverse categories. We make the first attempt to tackle this fundamental
2

and challenging problem and propose NeCGS, a neural compression paradigm,
3

which can compress hundreds of detailed and diverse 3D mesh models (∼684 MB)
4

by about 900 times (0.76 MB) with high accuracy and preservation of detailed
5

geometric details. Specifically, we first represent each irregular mesh model/shape
6

in a regular representation that implicitly describes the geometry structure of the
7

model using a 4D regular volume, called TSDF-Def volume. Such a regular rep-
8

resentation can not only capture local surfaces more effectively but also facilitate
9

the subsequent process. Then we construct a quantization-aware auto-decoder
10

network architecture to regress these 4D volumes, which can summarize the sim-
11

ilarity of local geometric structures within a model and across different models
12

for redundancy elimination, resulting in more compact representations, including
13

an embedded feature of a smaller size associated with each model and a network
14

parameter set shared by all models. We finally encode the resulting features and
15

network parameters into bitstreams through entropy coding. After decompressing
16

the features and network parameters, we can reconstruct the TSDF-Def volumes,
17

where the 3D surfaces can be extracted through the deformable marching cubes.
18

Extensive experiments and ablation studies demonstrate the significant advantages
19

of our NeCGS over state-of-the-art methods both quantitatively and qualitatively.
20

We have included the source code in the Supplemental Material.
21

1
Introduction
22

3D mesh models/shapes are widely used in various fields, such as computer graphics, virtual reality,
23

robotics, and autonomous driving. As geometric data becomes increasingly complex and voluminous,
24

effective compression techniques have become critical for efficient storage and transmission. More-
25

over, current geometry compression methods primarily focus on individual 3D models or sequences
26

of 3D models that are temporally correlated, but struggle to handle more general data sets, such as
27

compressing large numbers of unrelated 3D shapes.
28

Unlike images and videos represented as regular 2D or 3D volumes, mesh models are commonly
29

represented as triangle meshes, which are irregular and challenging to compress. Thus, a natural
30

idea is to structure the mesh models and then leverage image or video compression techniques to
31

compress them.Converting mesh models into voxelized point clouds is a common practice, and the
32

mesh models can be recovered from the point clouds via surface reconstruction methods [22, 24].
33

Based on this, in recent years, MPEG has developed two types of 3D point cloud compression (PCC)
34

standards [46, 28]: geometry-based PCC (GPCC) for static models and video-based PCC (VPCC) for
35

sequential models. And with advancements in deep learning, numerous learning-based PCC methods
36

[41, 14, 55, 19, 54] have emerged, enhancing compression efficiency. However, the voxelized point
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
clouds require a high resolution (typically 210 or more) to accurately represent geometry data, which
38

is redundancy, limiting the compression efficiency.
39

Another regular representation involves utilizing implicit fields of mesh models, such as signed
40

distance fields (SDF) and truncated signed distance fields (TSDF). This is achieved by calculating
41

the value of the implicit field at each uniformly distributed grid point, resulting in a regular volume.
42

And the mesh models can be recovered from the implicit fields through Matching Cubes [32] or its
43

variants [15, 45]. Compared with point clouds, the implicit volume could represent the mesh models
44

in a relatively small resolution. Recently proposed methods, such as DeepSDF [36], utilize multilayer
45

perceptrons (MLPs) to regress the SDFs of any given query points. While this representation achieves
46

high accuracy for single or similar models (e.g., chairs, tables), the limited receptive field of MLPs
47

makes it challenging to represent large numbers of models in different categories, which is a more
48

common scenario in practice.
49

We propose NeCGS, a novel framework for compressing large sets of geometric models. Our NeCGS
50

framework consists of two stages: regular geometry representation and compact neural compression.
51

In the first stage, each model is converted into a regular 4D volumetric format, called the TSDF-Def
52

volume, which can be considered a 3D ‘image’. In the second stage, we use an auto-decoder to
53

regress these 4D volumes. The embedded features and decoder parameters represent these models,
54

and compressing these components allows us to compress the entire geometry set. We conducted
55

extensive experiments on various datasets, demonstrating that our NeCGS framework achieves higher
56

compression efficiency compared to existing geometry compression methods when handling large
57

numbers of models. Our NeCGS can achieve a compression ratio of nearly 900 on some datasets,
58

compressing hundreds or even thousands of different models into 1∼2 MB while preserving detailed
59

structures.
60

Figure 1: Our NeCGeS can compress geometry data with hundreds or even thousands of shapes into 1~2 MB
while preserving details. Left: Original Geometry Data. Right: Decompressed Geometry Data. ü Zoom in for
details.

2


---Page Break---
2
Related Work
61

2.1
Geometry Representation
62

In general, the representation of geometry data is divided into two main categories, explicit represen-
63

tation and implicit representation, and they could be transformed into another.
64

Explicit Representation. Among the explicit representations, voxelization [7] is the most intuitive.
65

In this method, geometry models are represented by regularly distributed grids, effectively converting
66

them into 3D ‘images’. While this approach simplifies the processing of geometry models using
67

image processing techniques, it requires a high resolution to accurately represent the models, which
68

demands substantial memory and limits its application. Another widely used geometry representation
69

method is the point cloud, which consists of discrete points sampled from the surfaces of models.
70

This method has become a predominant approach for surface representation [2, 39, 40]. However, the
71

discrete nature of the points imposes constraints on its use in downstream tasks such as rendering and
72

editing. Triangle meshes offer a more precise and efficient geometry representation. By approximating
73

surfaces with numerous triangles, they achieve higher accuracy and efficiency for certain downstream
74

tasks.
75

Implicit Representation. Implicit representations use the isosurface of a function or field to represent
76

surfaces. The most widely used implicit representations include Binary Occupancy Field (BOF)
77

[22, 35], Signed Distance Field (SDF) [36, 29], and Truncated Signed Distance Field (TSDF) [11],
78

from which the model’s surface can be easily extracted. However, these methods are limited to
79

representing watertight models. The Unsigned Distance Field (UDF) [8], which is the absolute value
80

of the SDF, can represent more general models, not just watertight ones. Despite this advantage,
81

extracting surfaces from UDF is challenging, which limits its application.
82

Conversion between Geometry Representations. Geometry representations can be converted
83

between explicit and implicit forms. Various methods [21, 22, 24, 6, 35, 29, 45] are available for
84

calculating the implicit field from given models. Conversely, when converting from implicit to
85

explicit forms, Marching Cubes [32] and its derivatives [48, 49, 15, 45] can reconstruct continuous
86

surfaces from various implicit fields.
87

2.2
3D Geometry Data Compression
88

Single 3D Geometric Model Compression. In recent decades, compression techniques for images
89

and videos have rapidly advanced [51, 34, 59, 5, 4]. However, the irregular nature of geometry
90

data makes it more challenging to compress compared to images and video, which are represented
91

as volumetric data. A natural approach is to convert geometry data into voxelized point clouds,
92

treating them as 3D ‘images’, and then applying image and video compression techniques to them.
93

Following this intuition, MPEG developed the GPCC standards [13, 28, 47], where triangle meshes or
94

triangle soup approximates the surfaces of 3D models, enabling the compression of models with more
95

complex structures. Subsequently, several improved methods [37, 60, 53, 62] and learning-based
96

methods [18, 43, 10, 9, 3, 42, 54] have been proposed to further enhance compression performance.
97

However, these methods rely on voxelized point clouds to represent geometry models, which is
98

inefficient and memory-intensive, limiting their compression efficiency. In contrast to the previously
99

mentioned methods, Draco [12] uses a kd-tree-based coding method to compress vertices and employs
100

the EdgeBreaker algorithm to encode the topological relationships of the geometry data. Draco
101

utilizes uniform quantization to control the compression ratio, but its performance decreases at higher
102

compression ratios.
103

Multiple Model Compression. Compared to compressing single 3D geometric models, compressing
104

multiple objects is significantly more challenging. SLRMA [17] addresses this by using a low-rank
105

matrix to approximate vertex matrices, thus compressing sequential models. Mekuria et al. [33]
106

proposed the first codec for compressing sequential point clouds, where each frame is coded using
107

Octree subdivision through an 8-bit occupancy code. Building on this concept, MPEG developed the
108

VPCC standards [13, 28, 47], which utilize 3D-to-2D projection and encode time-varying projected
109

planes, depth maps, and other data using video codecs. Several improved methods [57, 26, 1, 44]
110

have been proposed to enhance the compression of sequential models. Recently, shape priors like
111

SMPL [31] and SMAL [63] have been introduced, allowing the pose and shape of a template frame
112

to be altered using only a few parameters. Pose-driven geometry compression methods [16, 58, 56]
113

3


---Page Break---
Embedded Features

……

Geometry Set

TSDF-Def Volumes
Predicted TSDF-Def Volumes

Regression
Loss
DMC

Compact Neural Representation
Regular Geometry Representation

Bitstream

……

Decompressed Geometry Set

DMC
Optimization

Decoder

……

……

……

……

……

……

Entropy Codec

Figure 2: The pipeline of NeCGS. It first represents original meshes regularly into TSDF-Def volumes, and an
auto-decoder network is utilized to regress these volume. Then the embedded features and decoder parameters
are compressed into bitstreams through entropy coding. When decompressing the models, the decompressed
embedded features are fed into the decoder with the decompressed parameters from the bitstreams, reconstructing
the TSDF-Def volumes, and the models can be extracted from them.

leverage this approach to achieve high compression efficiency. However, these methods are limited to
114

sequences of corresponding geometry data and cannot handle sets of unrelated geometry data, which
115

is more common in practice.
116

3
Proposed Method
117

Overview. Given a set of N 3D mesh models containing diverse categories, denoted as S = {Si}N
i=1,
118

we aim to compress them into a bitstream while maintaining the quality of the decompressed models
119

as much as possible. To this end, we propose a neural compression paradigm called NeCGS. As
120

shown in Fig. 2, NeCGS consists of two main modules, i.e., Regular Geometry Representation (RGR)
121

and Compact Neural Representation (CNR). Specifically, RGR first represents each irregular mesh
122

model within S into a regular 4D volume, namely TSDF-Def volume that mplicitly describes the
123

geometry structure of the model, via a rendering-based optimization, thus leading to a set of 4D
124

volumes V := {Vi}N
i=1 with Vi corresponding to Si. Then CNR further obtains a more compact
125

neural representation of V, where a quantization-aware auto-decoder-based network is constructed
126

to regress these volumes, producing an embedded feature for each volume. Finally, the embedded
127

features along with the network parameters are encoded into a bitstream through a typical entropy
128

coding method to achieve compression. We also want to note that NeCGS can also be applied to
129

compress 3D geometry sets represented in 3D point clouds, where one can either reconstruct from the
130

given point clouds 3D surfaces through a typical surface reconstruction method or adopt a pre-trained
131

network for SDF estimation from point clouds, e.g., SPSR [22] or IMLS [24], to bridge the gap
132

between 3D mesh and point cloud models. In what follows, we will detail NeCGS.
133

3.1
Regular Geometry Representation
134

Figure 3: 2D visual illustration of DMC.
The blue points refer to the deformable grid
points, the green points refer to the vertices of
the extracted surfaces, and the orange lines
refer to the faces of the extracted surfaces.
Left: The original grid points. Right: The
surface extraction.

Unlike 2D images and videos, where pixels are uniformly
135

distributed on 2D regular girds, the irregular characteristic
136

of 3D mesh models makes it challenging to compress them
137

efficiently and effectively. We propose to convert each
138

3D mesh model to a 4D regular volume called TSDF-
139

Def volume, which implicitly represents the geometry
140

structure of the model. Such a regular representation can
141

describe the model precisely, and its regular nature proves
142

beneficial for compression in the subsequent stage.
143

TSDF-Def Volume. Although 3D regular SDF or TSDF
144

volumes are widely used for representing 3D geometry
145

models, they may introduce distortions when the volume
146

4


---Page Break---
resolution is relatively limited. Inspired by recent shape extracting methods [48, 49], we propose
147

TSDF-Def, which extends the regular TSDF volume by introducing an additional deformation for
148

each grid point to adjust the detailed structure during the extraction of models, as shown in Fig.
149

3. Accordingly, we develop the differentiable Deformable Marching Cubes (DMC), the variant of
150

the Marching Cubes method [32], for surface extraction from a TSDF-Def volume. Consequently,
151

each shape S is represented as a 4D TSDF-Def volume, denoted as V ∈RK×K×K×4, where K
152

is the volume resolution. More specifically, the value of the grid point located at (u, v, w) is
153

V(u, v, w) := [TSDF(u, v, w), ∆u, ∆v, ∆w], where (∆u, ∆v, ∆w) are the deformation for the grid
154

point and 1 ≤u, v, w ≤K. TSDF-Def enhances representation accuracy, particularly when the grid
155

resolution is relatively low.
156

Optimization of TSDF-Def Volumes. To obtain the optimal TSDF-Def volume V for a given model
157

S, after initializing the deformations of each grid to zero and computing the TSDF value for each
158

grid we optimize the following problem:
159

min
V ERec(DMC(V), S),
(1)

where DMC(·) refers to the differentiable DMC process for extracting surfaces from TSDF-Def
160

volumes, and the EReg(·, ·) measures the differences between the rendered depth and silhouette
161

images of two mesh models through the differentiable rasterization [25]. Algorithm 1 summarizes
162

the whole optimization process. More details can be found in Sec. A.2 of the subsequent Appendix.
163

Algorithm 1: Optimization of TSDF-Def Volumes
Input: 3D mesh model S; the maximum number of iterations maxIter.
Output: The optimal TSDF-Def volume V ∈RK×K×K×4.

1 Place uniformly distributed grids in the cube of S, denoted as G ∈RK×K×K×3;

2 Initialize V[..., 0] as the ground truth TSDF of S at the location of G, the deformation
V[..., 1 :]=0, and the current iteration Iter = 0;

3 while Iter < maxIter do

4
Recover shape from V according to DMC, DMC(V);

5
Calculate the reconstruction error, ERec(DMC(V), S);

6
Optimize V using ADAM optimizer based on the reconstruction error;

7
Iter:=Iter+1;

8 end

9 return V;

3.2
Compact Neural Representation
164

Observing the similarity of local geometric structures within a typical 3D model and across different
165

models, i.e., redundancy, we further propose a quantization-aware neural representation process
166

to summarize the similarity within V, leading to more compact representations with redundancy
167

removed.
168

Network Architecture. We construct an auto-decoder network architecture to regress these 4D
169

TSDF-Def volumes. Specifically, it is composed of a head layer, which increases the channel of its
170

input, and L cascaded upsampling modules, which progressively upscale the feature volume. We
171

also utilize the PixelShuffle technique [50] between the convolution and activation layers to achieve
172

upscaling. We refer reviewers to Sec. B of Appendix for more details. For TSDF-Def volume Vi,
173

the corresponding input to the auto-decoder is the embedded feature, denoted as Fi ∈RK′×K′×K′×C,
174

where K′ is the resolution satisfying K′ ≪K and C is the number of channels. Moreover, we
175

integrate differentiable quantization to the embedded features and network parameters in the process,
176

which can efficiently reduce the quantization error. In all, the compact neural representation process
177

can be written as
178

bVi = DQ(Θ)(Q(Fi)).
(2)

where Q(·) stands for the differentiable quantization operator, and bVi is the regressed TSDF-Def.
179

Loss Function. We employ a joint loss function comprising Mean Absolute Error (MAE) and
180

Structural Similarity Index (SSIM) to simultaneously optimize the embedded features {Fi} and
181

5


---Page Break---
the network parameters Θ. In computing the MAE between the predicted and ground truth TSDF-
182

Def volumes, we concentrate more on the grids close to the surface. These surface grids crucially
183

determine the surfaces through their TSDFs and deformations; hence we assign them higher weights
184

during optimization than the grids farther away from the surface. The overall loss function for the
185

i-th model is written as
186

L( bVi, Vi) = ∥bVi −Vi∥1 + λ1∥Mi ⊙( bVi −Vi)∥1 + λ2(1 −SSIM( bVi, Vi)),
(3)

where Mi = 1(|Vi[..., 0])| < τ) is the mask, indicating whether a grid is near the surface, i.e., its
187

TSDF is less than the threshold τ, while λ1 and λ2 are the weights to balance each term of the loss
188

function.
189

Entropy Coding. After obtaining the quantized features {eFi = Q(Fi)} and quantized network
190

parameters eΘ = Q(Θ), we adopt the Huffman Codec [20] to further compress them into a bit-
191

stream. More advanced entropy coding methods can be employed to further improve compression
192

performance.
193

3.3
Decompression
194

To obtain the 3D mesh models from the bitstream, we first decompress the bitstream to derive the
195

embedded features, {eFi} and the decoder parameter, eΘ. Then, for each eFi, we feed it to the decoder
196

D e
Θ(·) to generate its corresponding TSDF-Def volume
197

bVi = D e
Θ(eFi).
(4)

Finally, we utilize DMC to recover each shape from bVi, bSi = DMC( bVi), forming the set of decom-
198

pressed geometry data, bS = {bSi}N
i=1.
199

4
Experiment
200

4.1
Experimental Setting
201

Implementation details. In the process of optimizing TSDF-Def volumes, we employed the ADAM
202

optimizer [23] for 500 iterations per shape, using a learning rate of 0.01. The resolution of TSDF-Def
203

volumes was K = 128. The resolution and the number of channels of the embedded features were
204

K′ = 4 and C = 16, respectively. And the decoder is composed of L = 5 upsampling modules with
205

an up-scaling factor of 2. During the optimization, we set λ1 = 5 and λ2 = 10, and the embedded
206

features and decoder parameters were optimized by the ADAM optimizer for 400 epochs, with a
207

learning rate of 1e-3. We achieved different compression efficiencies by adjusting decoder sizes. We
208

conducted all experiments on an NVIDIA RTX 3090 GPU with Intel(R) Xeon(R) CPU.
209

210

Table 1: Details of the selected datasets1.

Dataset
Original Size (MB)
# Models
AMA
378.41
500
DT4D
683.80
500
Thingi10K
335.92
1000
Mixed
496.16
600

Datasets. We tested our NeCGS on various types
211

of datasets, including humans, animals, and CAD
212

models. For human models, we randomly selected
213

500 shapes from the AMA dataset [52]. For animal
214

models, we randomly selected 500 shapes from
215

the DT4D dataset [27]. For the CAD models, we
216

randomly selected 1000 shapes from the Thingi10K
217

dataset [61]. Besides, we randomly selected 200
218

models from each dataset, forming a more challenging dataset, denoted as Mixed. The details about
219

the selected datasets are shown in Table 1. In all experiments, we scaled all models in a cube with a
220

range of [−1, 1]3 to ensure they are in the same scale.
221

Methods under Comparison. In terms of traditional geometry codecs, we chose the three most
222

impactful geometry coding standards with released codes, G-PCC2 and V-PCC3 from MPEG (see
223

1The original geometry data is kept as triangle meshes, so the storage size is much less than the voxelized
point clouds.
2https://github.com/MPEGGroup/mpeg-pcc-tmc13
3https://github.com/MPEGGroup/mpeg-pcc-tmc2

6


---Page Break---
more details about them in [13, 28, 47]), and Draco 4 from Google as the baseline methods. Addi-
224

tionally, we compared our approach with state-of-the-art deep learning-based compression methods,
225

specifically PCGCv2 [54]. Furthermore, we adapted DeepSDF [36] with quantization to serve as
226

another baseline method, denoted as QuantDeepSDF. It is worth noting that while some of the chosen
227

baseline methods were originally designed for point cloud compression, we utilized voxel sampling
228

and SPSR [22] to convert them between the forms of point cloud and surface. More details can be
229

found in Sec. C.2 appendix.
230

200
400
600
800
Compression Ratio

5

10

15

20

25

CD (10
3)

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

200
400
600
800
Compression Ratio

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

F-0.005

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

(a) AMA

200
400
600
800
1000 1200 1400
Compression Ratio

5

10

15

20

25

CD (10
3)

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

200
400
600
800
1000 1200 1400
Compression Ratio

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

F-0.005

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

(b) DT4D

100
200
300
400
Compression Ratio

5

10

15

20

25

30

35

40

CD (10
3)

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

100
200
300
400
Compression Ratio

0.2

0.3

0.4

0.5

0.6

F-0.005

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

(c) Thingi10K

0
200
400
600
800
Compression Ratio

5.0

7.5

10.0

12.5

15.0

17.5

20.0

22.5

CD (10
3)

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

0
200
400
600
800
Compression Ratio

0.1

0.2

0.3

0.4

0.5

0.6

0.7

F-0.005

GPCC
VPCC
PCGCv2
Draco
QuantDeepSDF
Ours

(d) Mixed

Figure 4: Quantitative comparisons of different methods on four 3D geometry sets.

Evaluation Metrics. Following previous reconstruction methods [35, 38], we utilize Chamfer
231

Distance (CD), Normal Consistency (NC), F-Score with the thresholds of 0.005 and 0.01 (F1-0.005
232

and F1-0.01) as the evaluation metrics. Furthermore, to comprehensively compare the compression
233

efficiency of different methods, we use Rate-Distortion (RD) curves. These curves illustrate the
234

distortions at various compression ratios, with CD and F1-0.005 specifically describing the distortion
235

of the decompressed models. Our goal is to minimize distortion, indicated by a low CD and a high
236

F1-Score, while maximizing the compression ratio. Therefore, for the RD curve representing CD,
237

optimal compression performance is achieved when the curve is closest to the lower right corner.
238

Similarly, for the RD curve representing the F1-Score, the ideal compression performance is when
239

the curve is nearest to the upper right corner. Their detailed definition can be found in Sec. C.1 of
240

appendix.
241

4.2
Results
242

The RD curves of different compression methods under different datasets are shown in Fig. 4. As
243

the compression ratio increases, the distortion also becomes larger. It is obvious that our NeCGS
244

can achieve much better compression performance than the baseline methods when the compression
245

ratio is high, even in the challenging Mixed dataset. In particular, our NeCGS achieves a minimum
246

compression ratio of 300, and on the DT4D dataset, the compression ratio even reaches nearly 900,
247

with minimal distortion. Due to the larger model differences within the Thingi10K and Mixed datasets
248

compared to the other two datasets, the compression performance on these two datasets is inferior.
249

(a) Ori.
(b) 455.25
(c) 651.85
(d) 899.73
Figure 6: Decompressed models under different com-
pression ratios.

The visual results of different compression meth-
250

ods are shown in Fig. 5. Compared to other
251

methods, models compressed using our ap-
252

proach occupy a larger compression ratio and
253

retain more details after decompression. Fig. 6
254

4https://github.com/google/draco

7


---Page Break---
(a) GPCC
(b) VPCC
(c) PCGCv2
(d) Draco (e) QuantDeepSDF
(f) Ours
(g) Ori.

AMA
DT4D
Thingi10K
Mixed

312.83
41.95
256.01
123.20
272.91
307.79

148.50
49.81
103.90
96.52
165.47
166.79

457.39
244.38
402.70
153.45
409.21
455.25

299.13
165.75
267.42
99.94
224.17
362.80

Figure 5: Visual comparisons of different compression methods. All numbers in corners represent the
compression ratio. ü Zoom in for details.

illustrates the decompressed models under different compression ratio. Even when the compression
255

ratio reaches nearly 900, our method can still retain the details of the models.
256

4.3
Ablation Study
257

In order to illustrate the efficiency of each design of our NeCGS, we conducted extensive ablation
258

study about them on the Mixed dataset.
259

Figure 7: Models recovered from different regular geometry repre-
sentations under various volume resolutions. From Left to Right:
Original, TSDF with K = 64, TSDF with K = 128, TSDF-Def
with K = 64, and TSDF-Def with K = 128.

Necessity of the Deformation of
260

Grids. We utilize TSDF-Def volumes
261

to as the regular geometry representa-
262

tion, instead of TSDF volumes like
263

previous methods. Compared with
264

models recovered from TSDF vol-
265

umes through MC, the models recov-
266

ered from TSDF-Def volumes through
267

DMC preserve more details of the thin
268

structures, especially when the volume resolutions are relatively small, as shown in Fig. 7. We also
269

conducted a numerical comparison of the decompressed models on the AMA dataset under these two
270

settings, and the results are shown in Table. 2, demonstrating its advantages.
271

Table 2: Quantitative comparisons of different RGRs.

RGR
Size (MB)
Com. Ratio
CD (×10−3) ↓
NC ↑
F1-0.005 ↑
F1-0.01 ↑
TSDF
1.631
304.20
5.015
0.944
0.662
0.936
TSDF-Def
1.612
307.79
4.913
0.947
0.674
0.943

Neural Representation Structure. To illustrate the superiority of auto-decoder framework, we
272

utilize an auto-encoder to regress the TSDF-Def volume. Technically, we used a ConvNeXt block
273

[30] as the encoder by replacing 2D convolutions with 3D convolutions. Under the auto-encoder
274

framework, we optimize the parameters of the encoder to change the embedded features. The RD
275

8


---Page Break---
300
400
500
600
Compression Ratio

5

6

7

8

9

CD (10
3)

Auto-encoder
Auto-decoder

300
400
500
600
Compression Ratio

0.35

0.40

0.45

0.50

0.55

0.60

0.65

F-0.005

Auto-encoder
Auto-decoder

(a)

300
350
400
450
500
550
Compression Ratio

5

10

15

20

25

30

35

40

CD (10
3)

w/o SSIM
w/ SSIM

300
350
400
450
500
550
Compression Ratio

0.475

0.500

0.525

0.550

0.575

0.600

0.625

0.650

0.675

F-0.005

w/o SSIM
w/ SSIM

(b)
Figure 8: (a) RD curves of different neural representation structures. (b) RD curves of different regression
losses.

curves about these two structures are shown in Fig. 8(a), demonstrating rationality of our decoder
276

structure.
277

(a) Original
(b) w/o SSIM
(c) w/ SSIM

Figure 9: Visual comparison of regression loss w/
and w/o SSIM item.

SSIM Loss. Compared to MAE, which focuses on
278

one-to-one errors between predicted and ground truth
279

volumes, the SSIM item in Eq. 3 emphasizes more
280

on the local similarity between volumes, increasing
281

the regression accuracy. To verify this, we removed
282

the SSIM item and kept others unchanged. Their RD
283

curves are shown in Fig. 8(b), and it is obvious that
284

the SSIM item in the regression loss increases the
285

compression performance. The visual comparison is
286

shown in Fig. 9, and without SSIM, there are floating
287

parts around the decompressed models.
288

(a) Ori.
(b) 64
(c) 128
(d) 256
Figure 10: Visual comparison under differ-
ent resolutions of TSDF-Def volume.

Resolution of TSDF-Def Volumes. We tested the com-
289

pression performance at different resolutions of TSDF-
290

Def volumes by adjusting the decoder layers accordingly.
291

Specifically, we removed the last layer for a resolution
292

of 64 and added an extra layer for a resolution of 256.
293

The quantitative and numerical comparisons are shown in
294

Table 3 and Fig. 10, respectively. Obviously, increasing
295

the volume resolution can enhance the compression effec-
296

tiveness, resulting in more detailed structures preserved
297

after decompression. However, the optimization and in-
298

ference time also increase accordingly due to more layers
299

involved.
300

Table 3: Quantitative comparisons of different resolutions of TSDF-Def volumes.

Res.
Size (MB)
Com. Ratio
CD (×10−3) ↓
NC ↑
F1-0.005 ↑
F1-0.01 ↑
Opt Time (h)
Infer. Time (ms)
64
1.408
268.75
4.271
0.927
0.721
0.966
2.16
38.97
128
1.493
253.45
3.436
0.952
0.842
0.991
16.32
98.95
256
1.627
232.58
3.234
0.962
0.870
0.995
94.50
421.94

5
Conclusion and Discussion
301

We have presented NeCGS, a highly effective neural compression scheme for 3D geometry sets.
302

NeCGS has achieved remarkable compression performance on various datasets with diverse and
303

detailed shapes, outperforming state-of-the-art compression methods to a large extent. These advan-
304

tages are attributed to our regular geometry representation and the compression accomplished by a
305

convolution-based auto-decoder. We believe our NeCGS framework will inspire further advancements
306

in the field of geometry compression.
307

However, our method still suffers from the following two limitations. One is that it requires more
308

than 15 hours to regress the TSDF-Def volumes, and the other one is that the usage of 3D convolution
309

layers limits the inference speed. Our future work will focus on addressing these challenges by
310

accelerating the optimization process and incorporating more efficient network modules.
311

9


---Page Break---
References
312

[1] A. Ahmmed, M. Paul, M. Murshed, and D. Taubman. Dynamic point cloud geometry compression using
313

cuboid based commonality modeling framework. In 2021 IEEE International Conference on Image
314

Processing (ICIP), pages 2159–2163. IEEE, 2021. 3
315

[2] P. J. Besl and N. D. McKay. Method for registration of 3-d shapes. In Sensor Fusion IV: Control Paradigms
316

and Data Structures, volume 1611, pages 586–606. Spie, 1992. 3
317

[3] S. Biswas, J. Liu, K. Wong, S. Wang, and R. Urtasun. Muscle: Multi sweep compression of lidar using
318

deep entropy models. Advances in Neural Information Processing Systems, 33:22170–22181, 2020. 3
319

[4] H. Chen, M. Gwilliam, S.-N. Lim, and A. Shrivastava. Hnerv: A hybrid neural representation for
320

videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
321

10270–10279, 2023. 3
322

[5] H. Chen, B. He, H. Wang, Y. Ren, S. N. Lim, and A. Shrivastava. Nerv: Neural representations for videos.
323

Advances in Neural Information Processing Systems, 34:21557–21568, 2021. 3
324

[6] Z.-Q. Cheng, Y.-Z. Wang, B. Li, K. Xu, G. Dang, and S.-Y. Jin. A survey of methods for moving least
325

squares surfaces. In Proceedings of the Fifth Eurographics/IEEE VGTC conference on Point-Based
326

Graphics, pages 9–23, 2008. 3
327

[7] J. Chibane, T. Alldieck, and G. Pons-Moll. Implicit functions in feature space for 3d shape reconstruction
328

and completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
329

pages 6970–6981, June 2020. 3
330

[8] J. Chibane, G. Pons-Moll, et al. Neural unsigned distance fields for implicit function learning. Advances in
331

Neural Information Processing Systems, 33:21638–21652, 2020. 3
332

[9] T. Fan, L. Gao, Y. Xu, D. Wang, and Z. Li. Multiscale latent-guided entropy model for lidar point cloud
333

compression. IEEE Transactions on Circuits and Systems for Video Technology, 33(12):7857–7869, 2023.
334

3
335

[10] C. Fu, G. Li, R. Song, W. Gao, and S. Liu. Octattention: Octree-based large-scale contexts model for point
336

cloud compression. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages
337

625–633, 2022. 3
338

[11] P. Gao, Z. Jiang, H. You, P. Lu, S. C. Hoi, X. Wang, and H. Li. Dynamic fusion with intra-and inter-modality
339

attention flow for visual question answering. In Proceedings of the IEEE/CVF conference on computer
340

vision and pattern recognition, pages 6639–6648, 2019. 3
341

[12] Google. Point cloud compression reference software. Website. https://github. com/google/draco. 3
342

[13] D. Graziosi, O. Nakagami, S. Kuma, A. Zaghetto, T. Suzuki, and A. Tabatabai. An overview of ongoing
343

point cloud compression standardization activities: Video-based (v-pcc) and geometry-based (g-pcc).
344

APSIPA Transactions on Signal and Information Processing, 9:e13, 2020. 3, 7
345

[14] A. F. Guarda, N. M. Rodrigues, and F. Pereira. Point cloud coding: Adopting a deep learning-based
346

approach. In 2019 Picture Coding Symposium (PCS), pages 1–5. IEEE, 2019. 1
347

[15] B. Guillard, F. Stella, and P. Fua. Meshudf: Fast and differentiable meshing of unsigned distance field
348

networks. In European Conference on Computer Vision, pages 576–592, 2022. 2, 3
349

[16] J. Hou, L.-P. Chau, N. Magnenat-Thalmann, and Y. He. Compressing 3-d human motions via keyframe-
350

based geometry videos. IEEE Transactions on Circuits and Systems for Video Technology, 25(1):51–62,
351

2014. 3
352

[17] J. Hou, L.-P. Chau, N. Magnenat-Thalmann, and Y. He. Sparse low-rank matrix approximation for data
353

compression. IEEE Transactions on Circuits and Systems for Video Technology, 27(5):1043–1054, 2015. 3
354

[18] L. Huang, S. Wang, K. Wong, J. Liu, and R. Urtasun. Octsqueeze: Octree-structured entropy model for
355

lidar compression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
356

pages 1313–1323, 2020. 3
357

[19] T. Huang and Y. Liu. 3d point cloud geometry compression on deep learning. In Proceedings of the 27th
358

ACM international conference on multimedia, pages 890–898, 2019. 1
359

[20] D. A. Huffman. A method for the construction of minimum-redundancy codes. Proceedings of the IRE,
360

40(9):1098–1101, 1952. 6
361

10


---Page Break---
[21] M. Kazhdan, M. Bolitho, and H. Hoppe. Poisson surface reconstruction. In Proceedings of the fourth
362

Eurographics symposium on Geometry processing, pages 61–70, 2006. 3
363

[22] M. Kazhdan and H. Hoppe. Screened poisson surface reconstruction. ACM Transactions on Graphics
364

(ToG), 32(3):1–13, 2013. 1, 3, 4, 7
365

[23] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,
366

2014. 6
367

[24] R. Kolluri. Provably good moving least squares. ACM Transactions on Algorithms, 4(2):1–25, 2008. 1, 3,
368

4
369

[25] S. Laine, J. Hellsten, T. Karras, Y. Seol, J. Lehtinen, and T. Aila. Modular primitives for high-performance
370

differentiable rendering. ACM Transactions on Graphics (ToG), 39(6):1–14, 2020. 5
371

[26] L. Li, Z. Li, V. Zakharchenko, J. Chen, and H. Li. Advanced 3d motion prediction for video-based dynamic
372

point cloud compression. IEEE Transactions on Image Processing, 29:289–302, 2019. 3
373

[27] Y. Li, H. Takehara, T. Taketomi, B. Zheng, and M. Nießner. 4dcomplete: Non-rigid motion estimation
374

beyond the observable surface. In Proceedings of the IEEE/CVF International Conference on Computer
375

Vision, pages 12706–12716, 2021. 6
376

[28] H. Liu, H. Yuan, Q. Liu, J. Hou, and J. Liu. A comprehensive study and comparison of core technologies
377

for mpeg 3-d point cloud compression. IEEE Transactions on Broadcasting, 66(3):701–717, 2019. 1, 3, 7
378

[29] S.-L. Liu, H.-X. Guo, H. Pan, P.-S. Wang, X. Tong, and Y. Liu. Deep implicit moving least-squares
379

functions for 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and
380

Pattern Recognition, pages 1788–1797, June 2021. 3
381

[30] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In Proceedings
382

of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022. 8
383

[31] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black. Smpl: A skinned multi-person linear
384

model. ACM Trans. Graph., 34(6), oct 2015. 3
385

[32] W. E. Lorensen and H. E. Cline. Marching cubes: A high resolution 3d surface construction algorithm.
386

ACM siggraph computer graphics, 21(4):163–169, 1987. 2, 3, 5
387

[33] R. Mekuria, K. Blom, and P. Cesar. Design, implementation, and evaluation of a point cloud codec for
388

tele-immersive video. IEEE Transactions on Circuits and Systems for Video Technology, 27(4):828–842,
389

2016. 3
390

[34] F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, and L. V. Gool. Practical full resolution learned
391

lossless image compression. In Proceedings of the IEEE/CVF conference on computer vision and pattern
392

recognition, pages 10629–10638, 2019. 3
393

[35] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks: Learning 3d
394

reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and
395

Pattern Recognition, pages 4460–4470, June 2019. 3, 7
396

[36] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous signed
397

distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer
398

Vision and Pattern Recognition, pages 165–174, June 2019. 2, 3, 7
399

[37] E. Peixoto. Intra-frame compression of point cloud geometry using dyadic decomposition. IEEE Signal
400

Processing Letters, 27:246–250, 2020. 3
401

[38] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger. Convolutional occupancy networks. In
402

European Conference on Computer Vision, pages 523–540. Springer, 2020. 7
403

[39] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and
404

segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages
405

652–660, 2017. 3
406

[40] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a
407

metric space. Advances in neural information processing systems, 30:1–xxx, 2017. 3
408

[41] M. Quach, G. Valenzise, and F. Dufaux. Learning convolutional transforms for lossy point cloud geometry
409

compression. In 2019 IEEE international conference on image processing (ICIP), pages 4320–4324. IEEE,
410

2019. 1
411

11


---Page Break---
[42] M. Quach, G. Valenzise, and F. Dufaux. Learning convolutional transforms for lossy point cloud geometry
412

compression. In 2019 IEEE international conference on image processing (ICIP), pages 4320–4324. IEEE,
413

2019. 3
414

[43] Z. Que, G. Lu, and D. Xu. Voxelcontext-net: An octree based framework for point cloud compression. In
415

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6042–6051,
416

2021. 3
417

[44] E. Ramalho, E. Peixoto, and E. Medeiros. Silhouette 4d with context selection: Lossless geometry
418

compression of dynamic point clouds. IEEE Signal Processing Letters, 28:1660–1664, 2021. 3
419

[45] S. Ren, J. Hou, X. Chen, Y. He, and W. Wang. Geoudf: Surface reconstruction from 3d point clouds via
420

geometry-guided distance representation. In Proceedings of the IEEE/CVF Internation Conference on
421

Computer Vision, pages 14214–14224, 2023. 2, 3
422

[46] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A. Chou, R. A. Cohen, M. Krivoku´ca,
423

S. Lasserre, Z. Li, et al. Emerging mpeg standards for point cloud compression. IEEE Journal on Emerging
424

and Selected Topics in Circuits and Systems, 9(1):133–148, 2018. 1
425

[47] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A. Chou, R. A. Cohen, M. Krivoku´ca,
426

S. Lasserre, Z. Li, et al. Emerging mpeg standards for point cloud compression. IEEE Journal on Emerging
427

and Selected Topics in Circuits and Systems, 9(1):133–148, 2018. 3, 7
428

[48] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler. Deep marching tetrahedra: a hybrid representation for
429

high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087–6101,
430

2021. 3, 5
431

[49] T. Shen, J. Munkberg, J. Hasselgren, K. Yin, Z. Wang, W. Chen, Z. Gojcic, S. Fidler, N. Sharp, and J. Gao.
432

Flexible isosurface extraction for gradient-based mesh optimization. ACM Transactions on Graphics
433

(TOG), 42(4):1–16, 2023. 3, 5
434

[50] W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang. Real-time
435

single image and video super-resolution using an efficient sub-pixel convolutional neural network. In
436

Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1874–1883, 2016.
437

5
438

[51] Y. Strümpler, J. Postels, R. Yang, L. V. Gool, and F. Tombari. Implicit neural representations for image
439

compression. In European Conference on Computer Vision, pages 74–91. Springer, 2022. 3
440

[52] D. Vlasic, I. Baran, W. Matusik, and J. Popovi´c. Articulated mesh animation from multi-view silhouettes.
441

ACM Transactions on Graphics, 27(3):1–9, 2008. 6
442

[53] C. Wang, W. Zhu, Y. Xu, Y. Xu, and L. Yang. Point-voting based point cloud geometry compression. In
443

2021 IEEE 23rd International Workshop on Multimedia Signal Processing (MMSP), pages 1–5. IEEE,
444

2021. 3
445

[54] J. Wang, D. Ding, Z. Li, and Z. Ma. Multiscale point cloud geometry compression. In 2021 Data
446

Compression Conference (DCC), pages 73–82. IEEE, 2021. 1, 3, 7
447

[55] J. Wang, H. Zhu, H. Liu, and Z. Ma. Lossy point cloud geometry compression via end-to-end learning.
448

IEEE Transactions on Circuits and Systems for Video Technology, 31(12):4909–4923, 2021. 1
449

[56] X. Wu, P. Zhang, M. Wang, P. Chen, S. Wang, and S. Kwong. Geometric prior based deep human point
450

cloud geometry compression. IEEE Transactions on Circuits and Systems for Video Technology, 2024. 3
451

[57] J. Xiong, H. Gao, M. Wang, H. Li, K. N. Ngan, and W. Lin. Efficient geometry surface coding in v-pcc.
452

IEEE Transactions on Multimedia, 25:3329–3342, 2022. 3
453

[58] R. Yan, Q. Yin, X. Zhang, Q. Zhang, G. Zhang, and S. Ma. Pose-driven compression for dynamic 3d
454

human via human prior models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 3
455

[59] Y. Yang, R. Bamler, and S. Mandt. Improving inference for neural image compression. Advances in Neural
456

Information Processing Systems, 33:573–584, 2020. 3
457

[60] X. Zhang, W. Gao, and S. Liu. Implicit geometry partition for point cloud compression. In 2020 Data
458

Compression Conference (DCC), pages 73–82. IEEE, 2020. 3
459

[61] Q. Zhou and A. Jacobson.
Thingi10k: A dataset of 10,000 3d-printing models.
arXiv preprint
460

arXiv:1605.04797, 2016. 6
461

12


---Page Break---
[62] W. Zhu, Y. Xu, D. Ding, Z. Ma, and M. Nilsson. Lossy point cloud geometry compression via region-wise
462

processing. IEEE Transactions on Circuits and Systems for Video Technology, 31(12):4575–4589, 2021. 3
463

[63] S. Zuffi, A. Kanazawa, D. Jacobs, and M. J. Black. 3D menagerie: Modeling the 3D shape and pose of
464

animals. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), July 2017. 3
465

13


---Page Break---
Appendix
466

A
Regular Geometry Representation
467

A.1
Tensor Quantization
468

Denoted x is a tensor, we quantize it in a fixed interval, [a, b], at (2N + 1) levels5 by
469

Q(x) = Round
Clamp(x, a, b) −a

s


× s + a,
(5)

where s = (b −a)/2N. In our experiment, we set a = −1 and b = 1.
470

A.2
Optimization of TSDF-deformation Volumes
471

We set a series of camera pose, T = {Ti}E
i=1, around the meshes. Let ID
1 (Ti) and ID
2 (Ti) represent
472

the depth images obtained from the reconstructed mesh DMC(V) and the given mesh S at the pose Ti
473

respectively. Similarly, let IM
1 (Ti) and IM
2 (Ti) denote their respective silhouette images at pose Ti.
474

The reconstruction error produced by silhouette and depth images at all pose are
475

EM(DMC(V), S) =
X

Ti∈T
∥I
M
1 (Ti) −I
M
2 (Ti)∥1
(6)

and
476

ED(DMC(V), S) =
X

Ti∈T
∥(I
D
1 (Ti) −I
D
2 (Ti)) ∗I
M
2 (Ti)∥1.
(7)

Then the reconstruction error is defined as
477

ERec(DMC(V), S) = EM(DMC(V), S) + λrecED(DMC(V), S),
(8)

where E = 4 and λrec = 10 in our experiment.
478

B
Auto-decoder-based Neural Compression
479

B.1
Upsampling Module
480

In each upsampling module, we utilize a PixelShuffle layer between the convolution and activa-
481

tion layers to upscale the input, as shown in Fig. 11. The input feature volume has dimensions
482

(Nin, Nin, Nin, Cin), with an upsampling scale of s and an output channel count of Cout.
483

C
Experiment
484

C.1
Evaluation Metric
485

Let SRec and SGT denote the reconstructed and ground-truth 3D shapes, respectively. We then
486

randomly sample Neval = 105 points on them, obtaining two point clouds, PRec and PGT. For each
487

point of PRec and PGT, the normal of the triangle face where it is sampled is considered to be its
488

normal vector, and the normal sets of PRec and PGT are denoted as NRec and NGT, respectively.
489

Let NN_Point(x, P) be the operator that returns the nearest point of x in the point cloud P. The CD
490

between them is defined as
491

CD(SRec, SGT) =
1
2Neval

X

x∈PRec
∥x −NN_Point(x, PGT)∥2

+
1
2Neval

X

x∈PGT
∥x −NN_Point(x, PRec)∥2.
(9)

5We partition the interval [a, b] into (2N + 1) levels, rather than 2N levels, to ensure the inclusion of the
value 0.

14


---Page Break---
Input

Output

Convolution

PixelShuffle

Activation

Upsampling Module

Figure 11: Upsampling Module.

Let NN_Normal(x, P) be the operator that returns the normal vector of the point x’s nearest point in
492

the point cloud P. The NC is defined as
493

NC(SRec, SGT) =
1
2Neval

X

x∈PRec
|NRec(x) · NN_Normal(x, PGT)|

+
1
2Neval

X

x∈PGT
|NGT(x) · NN_Normal(x, PRec)|.
(10)

F-Score is defined as the harmonic mean between the precision and the recall of points that lie within
494

a certain distance threshold ϵ between SRec and SGT,
495

F −Score(SRec, SGT, ϵ) = 2 · Recall · Precision

Recall + Precision ,
(11)

where
496

Recall(SRec, SGT, ϵ) =



x1 ∈PRec, s.t.
min
x2∈PGT ∥x1 −x2∥2 < ϵ
 ,

Precision(SRec, SGT, ϵ) =



x2 ∈PGT, s.t.
min
x1∈PRec ∥x1 −x2∥2 < ϵ
 .
(12)

Decoder

Figure 12: Pipeline of QuantDeepSDF.

C.2
QuantDeepSDF
497

Compared to DeepSDF, our QuantDeepSDF incorporates the following two modifications:
498

• The decoder parameters are quantized to enhance compression efficiency.
499

15


---Page Break---
• To maintain consistency with our NeCGS, the points sampled during training are drawn
500

from TSDF-Def volumes.
501

The pipeline of QuantDeepSDF is shown in Fig. 12. Specifically, the decoder is an MLP, where the
502

input is the concatenated vector of coordinate x ∈R3 and the i-th embedded feature vector Fi ∈RC,
503

and the output is the corresponding TSDF-Def value. In our experiment, the decoder consists of 8
504

layers, and the compression ratio is controled by changing the width of each layer.
505

C.3
Auto-Encoder in Ablation Study
506

Different from the auto-encoder used in our framework, where the embed features are directly
507

optimized, auto-encoder utilizes an encoder to produce the embedded features, where the inputs are
508

the TSDF-Def volumes. And the decoder is kept the same as our framework. During the optimization,
509

the parameters of encoder and decoder are optimized. Once optimized, the embedded features
510

produced by the encoder and decoder parameters are compressed into bitstreams.
511

C.4
More Visual Results
512

Fig. 13 depicts the visual results of the decompresed models from the AMA dataset, DT4D dataset,
513

and Thingi10K dataset under various compression ratios, respectively. With the compression ratio
514

increasing, the decompressed models still preserve the detailed structures, without large distortion.
515

Ori.
253.45
362.80
500.54

Ori.
455.26
651.85
899.73

Ori.
166.79
219.84
273.32

Figure 13: Visual results of the decompressed models under different compression ratios. From Top to Bottom:
AMA, DT4D, and Thingi10K. ü Zoom in for details.

16


---Page Break---
NeurIPS Paper Checklist
516

1. Claims
517

Question: Do the main claims made in the abstract and introduction accurately reflect the
518

paper’s contributions and scope?
519

Answer: [Yes]
520

Justification: Abstract.
521

Guidelines:
522

• The answer NA means that the abstract and introduction do not include the claims
523

made in the paper.
524

• The abstract and/or introduction should clearly state the claims made, including the
525

contributions made in the paper and important assumptions and limitations. A No or
526

NA answer to this question will not be perceived well by the reviewers.
527

• The claims made should match theoretical and experimental results, and reflect how
528

much the results can be expected to generalize to other settings.
529

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
530

are not attained by the paper.
531

2. Limitations
532

Question: Does the paper discuss the limitations of the work performed by the authors?
533

Answer: [Yes]
534

Justification: Sec. 5.
535

Guidelines:
536

• The answer NA means that the paper has no limitation while the answer No means that
537

the paper has limitations, but those are not discussed in the paper.
538

• The authors are encouraged to create a separate "Limitations" section in their paper.
539

• The paper should point out any strong assumptions and how robust the results are to
540

violations of these assumptions (e.g., independence assumptions, noiseless settings,
541

model well-specification, asymptotic approximations only holding locally). The authors
542

should reflect on how these assumptions might be violated in practice and what the
543

implications would be.
544

• The authors should reflect on the scope of the claims made, e.g., if the approach was
545

only tested on a few datasets or with a few runs. In general, empirical results often
546

depend on implicit assumptions, which should be articulated.
547

• The authors should reflect on the factors that influence the performance of the approach.
548

For example, a facial recognition algorithm may perform poorly when image resolution
549

is low or images are taken in low lighting. Or a speech-to-text system might not be
550

used reliably to provide closed captions for online lectures because it fails to handle
551

technical jargon.
552

• The authors should discuss the computational efficiency of the proposed algorithms
553

and how they scale with dataset size.
554

• If applicable, the authors should discuss possible limitations of their approach to
555

address problems of privacy and fairness.
556

• While the authors might fear that complete honesty about limitations might be used by
557

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
558

limitations that aren’t acknowledged in the paper. The authors should use their best
559

judgment and recognize that individual actions in favor of transparency play an impor-
560

tant role in developing norms that preserve the integrity of the community. Reviewers
561

will be specifically instructed to not penalize honesty concerning limitations.
562

3. Theory Assumptions and Proofs
563

Question: For each theoretical result, does the paper provide the full set of assumptions and
564

a complete (and correct) proof?
565

Answer: [NA]
566

17


---Page Break---
Justification: [NA]
567

Guidelines:
568

• The answer NA means that the paper does not include theoretical results.
569

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
570

referenced.
571

• All assumptions should be clearly stated or referenced in the statement of any theorems.
572

• The proofs can either appear in the main paper or the supplemental material, but if
573

they appear in the supplemental material, the authors are encouraged to provide a short
574

proof sketch to provide intuition.
575

• Inversely, any informal proof provided in the core of the paper should be complemented
576

by formal proofs provided in appendix or supplemental material.
577

• Theorems and Lemmas that the proof relies upon should be properly referenced.
578

4. Experimental Result Reproducibility
579

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
580

perimental results of the paper to the extent that it affects the main claims and/or conclusions
581

of the paper (regardless of whether the code and data are provided or not)?
582

Answer: [Yes]
583

Justification: Sec. 4.1.
584

Guidelines:
585

• The answer NA means that the paper does not include experiments.
586

• If the paper includes experiments, a No answer to this question will not be perceived
587

well by the reviewers: Making the paper reproducible is important, regardless of
588

whether the code and data are provided or not.
589

• If the contribution is a dataset and/or model, the authors should describe the steps taken
590

to make their results reproducible or verifiable.
591

• Depending on the contribution, reproducibility can be accomplished in various ways.
592

For example, if the contribution is a novel architecture, describing the architecture fully
593

might suffice, or if the contribution is a specific model and empirical evaluation, it may
594

be necessary to either make it possible for others to replicate the model with the same
595

dataset, or provide access to the model. In general. releasing code and data is often
596

one good way to accomplish this, but reproducibility can also be provided via detailed
597

instructions for how to replicate the results, access to a hosted model (e.g., in the case
598

of a large language model), releasing of a model checkpoint, or other means that are
599

appropriate to the research performed.
600

• While NeurIPS does not require releasing code, the conference does require all submis-
601

sions to provide some reasonable avenue for reproducibility, which may depend on the
602

nature of the contribution. For example
603

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
604

to reproduce that algorithm.
605

(b) If the contribution is primarily a new model architecture, the paper should describe
606

the architecture clearly and fully.
607

(c) If the contribution is a new model (e.g., a large language model), then there should
608

either be a way to access this model for reproducing the results or a way to reproduce
609

the model (e.g., with an open-source dataset or instructions for how to construct
610

the dataset).
611

(d) We recognize that reproducibility may be tricky in some cases, in which case
612

authors are welcome to describe the particular way they provide for reproducibility.
613

In the case of closed-source models, it may be that access to the model is limited in
614

some way (e.g., to registered users), but it should be possible for other researchers
615

to have some path to reproducing or verifying the results.
616

5. Open access to data and code
617

Question: Does the paper provide open access to the data and code, with sufficient instruc-
618

tions to faithfully reproduce the main experimental results, as described in supplemental
619

material?
620

18


---Page Break---
Answer: [Yes]
621

Justification: We include the code in the supplemental material.
622

Guidelines:
623

• The answer NA means that paper does not include experiments requiring code.
624

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
625

public/guides/CodeSubmissionPolicy) for more details.
626

• While we encourage the release of code and data, we understand that this might not be
627

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
628

including code, unless this is central to the contribution (e.g., for a new open-source
629

benchmark).
630

• The instructions should contain the exact command and environment needed to run to
631

reproduce the results. See the NeurIPS code and data submission guidelines (https:
632

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
633

• The authors should provide instructions on data access and preparation, including how
634

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
635

• The authors should provide scripts to reproduce all experimental results for the new
636

proposed method and baselines. If only a subset of experiments are reproducible, they
637

should state which ones are omitted from the script and why.
638

• At submission time, to preserve anonymity, the authors should release anonymized
639

versions (if applicable).
640

• Providing as much information as possible in supplemental material (appended to the
641

paper) is recommended, but including URLs to data and code is permitted.
642

6. Experimental Setting/Details
643

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
644

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
645

results?
646

Answer: [Yes]
647

Justification: Sec. 4.1
648

Guidelines:
649

• The answer NA means that the paper does not include experiments.
650

• The experimental setting should be presented in the core of the paper to a level of detail
651

that is necessary to appreciate the results and make sense of them.
652

• The full details can be provided either with the code, in appendix, or as supplemental
653

material.
654

7. Experiment Statistical Significance
655

Question: Does the paper report error bars suitably and correctly defined or other appropriate
656

information about the statistical significance of the experiments?
657

Answer: [Yes]
658

Justification: Sec. 4.2 and 4.3.
659

Guidelines:
660

• The answer NA means that the paper does not include experiments.
661

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
662

dence intervals, or statistical significance tests, at least for the experiments that support
663

the main claims of the paper.
664

• The factors of variability that the error bars are capturing should be clearly stated (for
665

example, train/test split, initialization, random drawing of some parameter, or overall
666

run with given experimental conditions).
667

• The method for calculating the error bars should be explained (closed form formula,
668

call to a library function, bootstrap, etc.)
669

• The assumptions made should be given (e.g., Normally distributed errors).
670

• It should be clear whether the error bar is the standard deviation or the standard error
671

of the mean.
672

19


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
673

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
674

of Normality of errors is not verified.
675

• For asymmetric distributions, the authors should be careful not to show in tables or
676

figures symmetric error bars that would yield results that are out of range (e.g. negative
677

error rates).
678

• If error bars are reported in tables or plots, The authors should explain in the text how
679

they were calculated and reference the corresponding figures or tables in the text.
680

8. Experiments Compute Resources
681

Question: For each experiment, does the paper provide sufficient information on the com-
682

puter resources (type of compute workers, memory, time of execution) needed to reproduce
683

the experiments?
684

Answer: [Yes]
685

Justification: Sec. 4.1 and 4.3.
686

Guidelines:
687

• The answer NA means that the paper does not include experiments.
688

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
689

or cloud provider, including relevant memory and storage.
690

• The paper should provide the amount of compute required for each of the individual
691

experimental runs as well as estimate the total compute.
692

• The paper should disclose whether the full research project required more compute
693

than the experiments reported in the paper (e.g., preliminary or failed experiments that
694

didn’t make it into the paper).
695

9. Code Of Ethics
696

Question: Does the research conducted in the paper conform, in every respect, with the
697

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
698

Answer: [Yes]
699

Justification: [NA]
700

Guidelines:
701

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
702

• If the authors answer No, they should explain the special circumstances that require a
703

deviation from the Code of Ethics.
704

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
705

eration due to laws or regulations in their jurisdiction).
706

10. Broader Impacts
707

Question: Does the paper discuss both potential positive societal impacts and negative
708

societal impacts of the work performed?
709

Answer: [Yes]
710

Justification: [NA]
711

Guidelines:
712

• The answer NA means that there is no societal impact of the work performed.
713

• If the authors answer NA or No, they should explain why their work has no societal
714

impact or why the paper does not address societal impact.
715

• Examples of negative societal impacts include potential malicious or unintended uses
716

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
717

(e.g., deployment of technologies that could make decisions that unfairly impact specific
718

groups), privacy considerations, and security considerations.
719

• The conference expects that many papers will be foundational research and not tied
720

to particular applications, let alone deployments. However, if there is a direct path to
721

any negative applications, the authors should point it out. For example, it is legitimate
722

to point out that an improvement in the quality of generative models could be used to
723

20


---Page Break---
generate deepfakes for disinformation. On the other hand, it is not needed to point out
724

that a generic algorithm for optimizing neural networks could enable people to train
725

models that generate Deepfakes faster.
726

• The authors should consider possible harms that could arise when the technology is
727

being used as intended and functioning correctly, harms that could arise when the
728

technology is being used as intended but gives incorrect results, and harms following
729

from (intentional or unintentional) misuse of the technology.
730

• If there are negative societal impacts, the authors could also discuss possible mitigation
731

strategies (e.g., gated release of models, providing defenses in addition to attacks,
732

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
733

feedback over time, improving the efficiency and accessibility of ML).
734

11. Safeguards
735

Question: Does the paper describe safeguards that have been put in place for responsible
736

release of data or models that have a high risk for misuse (e.g., pretrained language models,
737

image generators, or scraped datasets)?
738

Answer: [NA]
739

Justification: [NA]
740

Guidelines:
741

• The answer NA means that the paper poses no such risks.
742

• Released models that have a high risk for misuse or dual-use should be released with
743

necessary safeguards to allow for controlled use of the model, for example by requiring
744

that users adhere to usage guidelines or restrictions to access the model or implementing
745

safety filters.
746

• Datasets that have been scraped from the Internet could pose safety risks. The authors
747

should describe how they avoided releasing unsafe images.
748

• We recognize that providing effective safeguards is challenging, and many papers do
749

not require this, but we encourage authors to take this into account and make a best
750

faith effort.
751

12. Licenses for existing assets
752

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
753

the paper, properly credited and are the license and terms of use explicitly mentioned and
754

properly respected?
755

Answer: [Yes]
756

Justification: [NA]
757

Guidelines:
758

• The answer NA means that the paper does not use existing assets.
759

• The authors should cite the original paper that produced the code package or dataset.
760

• The authors should state which version of the asset is used and, if possible, include a
761

URL.
762

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
763

• For scraped data from a particular source (e.g., website), the copyright and terms of
764

service of that source should be provided.
765

• If assets are released, the license, copyright information, and terms of use in the
766

package should be provided. For popular datasets, paperswithcode.com/datasets
767

has curated licenses for some datasets. Their licensing guide can help determine the
768

license of a dataset.
769

• For existing datasets that are re-packaged, both the original license and the license of
770

the derived asset (if it has changed) should be provided.
771

• If this information is not available online, the authors are encouraged to reach out to
772

the asset’s creators.
773

13. New Assets
774

Question: Are new assets introduced in the paper well documented and is the documentation
775

provided alongside the assets?
776

21


---Page Break---
Answer: [NA]
777

Justification: [NA]
778

Guidelines:
779

• The answer NA means that the paper does not release new assets.
780

• Researchers should communicate the details of the dataset/code/model as part of
781

their submissions via regular templates. This includes details about training, license,
782

limitations, etc.
783

• The paper should discuss whether and how consent was obtained from people whose
784

asset is used.
785

• At submission time, remember to anonymize your assets (if applicable). You can either
786

create an anonymized URL or include an anonymized zip file.
787

14. Crowdsourcing and Research with Human Subjects
788

Question: For crowdsourcing experiments and research with human subjects, does the paper
789

include the full text of instructions given to participants and screenshots, if applicable, as
790

well as details about compensation (if any)?
791

Answer: [NA]
792

Justification: [NA]
793

Guidelines:
794

• The answer NA means that the paper does not involve crowdsourcing nor research with
795

human subjects.
796

• Including this information in the supplemental material is fine, but if the main contribu-
797

tion of the paper involves human subjects, then as much detail as possible should be
798

included in the main paper.
799

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
800

or other labor should be paid at least the minimum wage in the country of the data
801

collector.
802

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
803

Subjects
804

Question: Does the paper describe potential risks incurred by study participants, whether
805

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
806

approvals (or an equivalent approval/review based on the requirements of your country or
807

institution) were obtained?
808

Answer: [NA]
809

Justification: [NA]
810

Guidelines:
811

• The answer NA means that the paper does not involve crowdsourcing nor research with
812

human subjects.
813

• Depending on the country in which research is conducted, IRB approval (or equivalent)
814

may be required for any human subjects research. If you obtained IRB approval, you
815

should clearly state this in the paper.
816

• We recognize that the procedures for this may vary significantly between institutions
817

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
818

guidelines for their institution.
819

• For initial submissions, do not include any information that would break anonymity (if
820

applicable), such as the institution conducting the review.
821

22


---Page Break---
