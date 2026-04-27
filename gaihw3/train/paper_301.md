Night-to-Day Translation via Illumination Degradation
Disentanglement

Anonymous Author(s)
Affiliation
Address
email

Abstract

Night-to-Day translation (Night2Day) aims to achieve day-like vision for nighttime
1

scenes. However, processing night images with complex degradations remains a
2

significant challenge under unpaired conditions. Previous methods that uniformly
3

mitigate these degradations have proven inadequate in simultaneously restoring
4

daytime domain information and preserving underlying semantics. In this paper,
5

we propose N2D3 (Night-to-Day via Degradation Disentanglement) to identify
6

different degradation patterns in nighttime images. Specifically, our method com-
7

prises a degradation disentanglement module and a degradation-aware contrastive
8

learning module. Firstly, we extract physical priors from a photometric model
9

based on Kubelka-Munk theory. Then, guided by these physical priors, we design a
10

disentanglement module to discriminate among different illumination degradation
11

regions. Finally, we introduce the degradation-aware contrastive learning strategy
12

to preserve semantic consistency across distinct degradation regions. Our method
13

is evaluated on two public datasets, demonstrating a significant improvement of
14

5.4 FID on BDD100K and 10.3 FID on Alderley.
15

1
Introduction
16

Nighttime images often suffer from severe information loss, posing significant challenges to both
17

human visual recognition and computer vision tasks including detection, segmentation, etc. [14].
18

In contrast, daylight images exhibit rich content and intricate details. Achieving day-like nighttime
19

vision remains a primary objective in nighttime perception, sparking numerous pioneering works [30].
20

Night-to-Day image translation (Night2Day) offers a comprehensive solution to achieve day-like
21

vision at night. The primary goal is to transform images from nighttime to daytime while maintaining
22

their underlying semantic structure. However, achieving this goal is challenging. It requires to process
23

complex degraded images using unpaired data, which raises additional difficulties compared to other
24

image translation tasks.
25

Recently, explorations have been made in Night2Day. Early approaches, such as ToDayGAN,
26

demonstrated the effectiveness of cycle-consistent learning in maintaining semantic structure [1].
27

Subsequent methods incorporated auxiliary structure regularization techniques, including perceptual
28

loss and uncertainty regularization, to better preserve the original structure [33, 18]. Furthermore,
29

some methods utilized daytime images with nearby GPS locations to aid in coarse structure regular-
30

ization [26]. However, these methods often neglect the complex degradations at nighttime, applying
31

structure regularization uniformly and resulting in severe artifacts. To address this issue, more recent
32

approaches adopt auxiliary human annotations to maintain semantic consistency, such as segmenta-
33

tion maps and bounding boxes [16, 22]. Despite their potential, these methods are labor-intensive
34

and challenging, especially since many nighttime scenes are beyond human cognition.
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Physical 

Prior

w Disentangled Regularization 

w/o Disentangled Regularization 

(c) Visual Comparison
(b) Disentangled Regularization 
(a) Disentanglement Process

Darkness

Well-lit

Generated Patches

High-light

Light Effects

Figure 1: Illustration of our motivation. (a) The disentanglement process leverages physical priors. (b)
The image patches are restored individually for each degradation type. (c) The proposed Disentangled
Regularization improves the overall performance.

The critical limitation of the aforementioned methods is the disregard for complex degraded regions.
36

Specifically, different regions in nighttime images possess varying characteristics, such as extreme
37

darkness, well-lit regions, light effects, etc. Treating all these degraded regions equally could adversely
38

impact the results. As illustrated in Figure 1, our key insight emphasizes that nighttime images suffer
39

from various degradations, necessitating customizing restoration for different degradation types.
40

Intuitively, we manage to disentangle nighttime images into patches according to the recognized
41

degradation type and learn individual restoration patterns for them to enhance the overall performance.
42

Motivated by this point, we propose N2D3 (Night to Day via Degradation Disentanglement), which
43

utilizes Generative Adversarial Networks (GANs) to bridge the domain gap between nighttime and
44

daytime in a degradation-aware manner, as illustrated in Figure 2. There are two modules in N2D3,
45

including physical-informed degradation disentanglement and degradation-aware contrastive learning,
46

which are employed to preserve the semantic structure of nighttime images. In the disentanglement
47

of nighttime degradation, a photometric model tailored to nighttime scenes is conducted to extract
48

physical priors. Subsequently, the illuminance and physical priors are integrated to disentangle
49

regions into darkness, well-lit, high-light, and light effects. Building on this, degradation-aware
50

contrastive learning is designed to constrain the similarity of the source and generated images in
51

different regions. It comprises disentanglement-guided sampling and reweighting strategies. The
52

sampling strategy mines valuable anchors and hard negative examples, while the reweighting process
53

assigns their weights. They enhance vanilla contrastive learning by prioritizing valuable patches with
54

appropriate attention. Ultimately, our method yields highly faithful results that are visually pleasing
55

and beneficial for downstream vision tasks including keypoint matching and semantic segmentation.
56

Our contributions are summarized as follows:
57

(1) We propose the N2D3 translation method based on the illumination degradation disentanglement
58

module, which enables degradation-aware restoration of nighttime images.
59

(2) We present a novel degradation-aware contrastive learning module to preserve the semantic
60

structure of generated results. The core design incorporates disentanglement-guided sampling and
61

reweighting strategies, which greatly enhance the performance of vanilla contrastive learning.
62

(3) Experimental results on two public datasets underscore the significance of considering distinct
63

degradation types in nighttime scenes. Our method achieves state-of-the-art performance in visual
64

effects and downstream tasks.
65

2
Related Work
66

Unpaired Image-to-Image Translation. Unpaired image-to-image translation addresses the chal-
67

lenge of lacking paired data, providing an effective self-supervised learning strategy. To overcome the
68

efficiency limitations of traditional cycle-consistency learning, Park et al., first introduces contrastive
69

learning to this domain, achieving efficient one-sided learning[20]. Following this work, several stud-
70

ies have improved the contrastive learning by generating hard negative examples [24], re-weighting
71

positive-negative pairs [31], and selecting key samples [9]. Furthermore, other constraints, such as
72

density [27] and path length [28], have been explored in unpaired image translation. However, all
73

these works neglect physical priors in the nighttime, leading to suboptimal results in Night2Day.
74

2


---Page Break---
Physical 

Prior 
Darkness
Light Effect
Well-lit

𝑥1𝑦1 𝑥1𝑦2 𝑥1𝑦3 𝑥1𝑦4 𝑥1𝑦5 𝑥1𝑦6

𝑥2𝑦1 𝑥2𝑦2 𝑥2𝑦3 𝑥2𝑦4 𝑥2𝑦5 𝑥2𝑦6

𝑥3𝑦1 𝑥3𝑦2 𝑥3𝑦3 𝑥3𝑦4 𝑥3𝑦5 𝑥3𝑦6

𝑥4𝑦1 𝑥4𝑦2 𝑥4𝑦3 𝑥4𝑦4 𝑥4𝑦5 𝑥4𝑦6

𝑥5𝑦1 𝑥5𝑦2 𝑥5𝑦3 𝑥5𝑦4 𝑥5𝑦5 𝑥5𝑦6

𝑥6𝑦1 𝑥6𝑦2 𝑥6𝑦3 𝑥6𝑦4 𝑥6𝑦5 𝑥6𝑦6

𝑥1𝑦1
𝑤12
0
0
0
0

𝑤21
𝑥2𝑦2
0
0
0
0

0
0
𝑥3𝑦3
𝑤34
0
0

0
0
𝑤43
𝑥4𝑦4
0
0

0
0
0
0
𝑥5𝑦5
𝑤56

0
0
0
0
𝑤65
𝑥6𝑦6

min
wij,i,j∈[1,N][෍

i

N

෍

j

N

wij ⋅exiyi/β ]

s. t.෍

i=1

N

wij = 1, ෍

j=1

N

wij = 1

(b)  Training Phase

𝑦1

𝑦2

𝑦3

𝑦4

𝑦5

𝑦6

𝑥1
𝑥2
𝑥3
𝑥4
𝑥5
𝑥6

…

…

…

…

…

…

…

…

(a) Inference Phase

GAN 

Loss

Deg. Aware 
Reweighting

Optimal Transport

Block Diagonal SM
Similarity Matrix(SM)

Deg. Aware 

Sampling

Disentangled Map
Disentangled Categories

Generator

Shared 
Encoder
Deg. Aware 
Contrastive 

Loss

Figure 2:
The overall architecture of the proposed N2D3 method. The training phase contains
the physical prior informed degradation disentanglement module and degradation-aware contrastive
learning module. They are utilized to optimize the ResNet-based generator which is the main part in
the inference phase.

Nighttime Domain Translation. Domain translation techniques have been applied to address adverse
75

nighttime conditions. An early contribution is made by Anoosheh et al., which demonstrates the
76

effectiveness of cycle-consistent learning in Night2Day[1]. Following this, many works incorporate
77

different modules into cycle-consistent learning to enhance structural modeling capabilities. Zheng et
78

al. incorporate a fork-shaped encoder to enhance visual perceptual quality[33]. AUGAN employs
79

uncertainty estimation to mine useful features in nighttime images[18]. Fan et al. explore inter-
80

frequency relation knowledge to streamline the Night2Day process[5]. Xia et al. utilize nearby GPS
81

locations to form paired night and daytime images, providing weak supervision[26]. Some other
82

studies incorporate human annotations to impose structural constraints, overlooking the practical
83

difficulty of acquiring such annotations at nighttime with multiple degradations [11][16] [22]. To
84

address the concerns of the aforementioned methods, the proposed N2D3 explores patch-wise
85

contrastive learning with physical guidance, so as to achieve degradation-aware Night2Day. N2D3 is
86

free of human annotations and offers comprehensive structural modeling to provide faithful translation
87

results.
88

3
Methods
89

Given nighttime image IN ∈N and daytime image ID ∈D, the goal of Night2Day is to translate
90

images from nighttime to daytime while preserving content semantic consistency. This involves the
91

construction of a mapping function F with parameters θ, which can be formulated as Fθ : IN →ID.
92

Our method N2D3 is illustrated in Figure 2. To train a generator for Night2Day, we employ GANs as
93

the overall learning framework to bridge the domain gap between nighttime and daytime. Our core
94

design, consisting of the degradation disentanglement module and the degradation-aware contrastive
95

learning module, aims to preserve the structure from the source images and suppress artifacts.
96

In this section, we first introduce physical priors in the nighttime environment, and then describe
97

the degradation disentanglement module and the degradation-aware contrastive learning module,
98

respectively.
99

3.1
Physical Priors for Nighttime Environment
100

The illumination degradations at night are primarily categorized as darkness, well-lit regions, high-
101

light regions, and light effects. As shown in Figure 3, well-lit represents the diffused reflectance under
102

normal light, while the light effects denote phenomena such as flare, glow, and specular reflections.
103

Intuitively, these regions can be disentangled through the analysis of illumination distribution. Among
104

these degradation types, darkness and high-light are directly correlated with illuminance and can be
105

effectively disentangled through illumination estimation.
106

As a common practice, we estimate the illuminance map L by utilizing the maximum RGB channel
107

of image IN as L = maxc∈R,G,B Ic
N . Then k-nearest neighbors [4] is employed to acquire three
108

clusters representing darkness, well-lit, and high-light regions. These clusters are aggregated as
109

masks Md, Mn, Mh. However, the challenge arises with light effects that are mainly related to
110

3


---Page Break---
Figure 3: The first row displays nighttime images, while the second row shows the corresponding
degradation disentanglement results. The color progression from blue, light blue, green to yellow
corresponds to the following regions: darkness, well-lit, light effects, and high-light, respectively.

the illumination. Light effects regions tend to intertwine with well-lit regions when using only the
111

illumination map, as they often share similar illumination densities. To disentangle light effects from
112

well-lit regions, we need to introduce additional physical priors.
113

To extract the physical priors for disentangling light effects, we develop a photometric model derived
114

from Kubelka-Munk theory [17]. This model characterizes the spectrum of light E reflected from an
115

object as follows:
116

E(λ, x) = e(λ, x)(1 −ρf(x))2R∞(λ, x) + e(λ, x)ρf(x),
(1)
here x represents the horizontal component for analysis, while the analysis of the vertical component
117

y is the same as the horizontal component. λ corresponds to the wavelength of light. e(λ, x) signifies
118

the spectrum, representing the illumination density and color. ρf stands for the Fresnel reflectance
119

coefficient. R∞is the material reflectivity function, formulated as follows at a specific location
120

x = x0:
121

R(λ) = a(λ) −
p

a(λ)2 −1, a(λ) = 1 + k(λ)

s(λ) ,
(2)

where k(λ) and s(λ) denote the absorption and scattering coefficients, respectively. This formulation
122

implies that for any local pixels, the material reflectivity is determined if the material is given.
123

Assuming C is the material distribution function, which describes the material type varying across
124

locations, the material reflectivity R∞can be formulated as:
125

R∞(λ, x) = R(λ)C(x).
(3)
Since the mixture of light effects and well-lit regions has been obtained previously, the core of
126

disentangling light effects from well-lit regions lies in separating the illumination e(λ, x) and re-
127

flectance components R(λ)C(x). Note that the Fresnel reflectance coefficient ρf(x) approaches 0 in
128

reflectance-dominating well-lit regions, while ρf(x) approaches 1 in illumination-dominating light
129

effects regions. According to Equation (1), the photometric model for the mixture of light effects and
130

well-lit regions is formulated as:
131

E(λ, x) =
e(λ, x),
if x /∈Ω
e(λ, x)R(λ)C(x),
if x ∈Ω,
(4)

where Ωdenotes the reflectance-dominating well-lit regions.
132

Subsequently, we observe that the following color invariant response to the regions with high color
133

saturation, which is suitable to extract the illumination:
134

Nλmxn =
∂m+n−1

∂λm−1∂xn {
1
E(λ, x)
∂E(λ, x)

∂λ
},
(5)

This invariant has the following characteristics:
135

Nλmxn =
∂m+n−2

∂λm−1∂xn−1
∂
∂x


1
E(λ, x)
∂E(λ, x)

∂λ



=
∂m+n−2

∂λm−1∂xn−1
∂
∂x


1
e(λ, x)
∂e(λ, x)

∂λ
+
1
R(λ)C(x)
∂R(λ)C(x)

∂λ



=
∂m+n−1

∂λm−1∂xn


1
e(λ, x)
∂e(λ, x)

∂λ


.

(6)

4


---Page Break---
Equation (5) to Equation (6) demonstrate that the invariant Nλmxn captures the features only related
136

to illumination e(λ, x). Consequently, we assert that Nλmxn functions as a light effects detector
137

because light effects are mainly related to the illumination. It allows us to design the illumination
138

disentanglement module based on this physical prior.
139

3.2
Degradation Disentanglement Module
140

In this subsection, we will elucidate how to incorporate the invariant for extracting light effects into
141

the disentanglement in computation. As common practice, the following second and third-order
142

components, both horizontally and vertically, are taken into account in the practical calculation of the
143

final invariant, which is denoted as N:
144

N =
q

N 2
λx + N 2
λλx + N 2
λy + N 2
λλy.
(7)

here Nλx and Nλλx can be computed through E(λ, x) by simplifying Equation (5). The calculation
145

of Nλy and Nλλy are the same. Specifically,
146

Nλx = EλxE −EλEx

E2
, Nλλx
= EλλxE2 −EλλExE −2EλxEλE + 2E2
λEx
E3
,
(8)

where Ex and Eλ denote the partial derivatives of x and λ.
147

To compute each component in the invariant N, we develop a computation scheme starting with the
148

estimation of E and its partial derivatives Eλ and Eλλ using the Gaussian color model:
149

" E(x, y)
Eλ(x, y)
Eλλ(x, y)

#

=

"
0.06,
0.63,
0.27
0.3,
0.04,
−0.35
0.34,
−0.6,
0.17

# "R(x, y)
G(x, y)
B(x, y)

#

,
(9)

where x, y are pixel locations of the image. Then, the spatial derivatives Ex and Ey are calculated by
150

convolving E with Gaussian derivative kernel g and standard deviation σ:
151

Ex(x, y, σ) =
X

t∈Z
E(t, y)∂g(x −t, σ)

∂x
,
(10)

where t denotes the index of the horizontal component x and Z represents set of integers. The spatial
152

derivatives for Eλx and Eλλx are obtained by applying Equation (10) to Eλ and Eλλ. Then invariant
153

N can be obtained following Equation (8) and Equation (7).
154

To extract the light effects, ReLU and normalization functions are first applied to filter out minor
155

disturbances. Then, by filtering invariant N with the well-lit mask Mn, we obtain the light effects
156

from the well-lit regions. The operations above can be formulated as:
157

Mle = ReLU(N −µ(N)

σ(N)
) ⊙Mn,
(11)

while the well-lit mask are refined: Mn ←Mn −Mle.
158

With the initial disentanglement in Section 3.1, we obtain the final disentanglement: Md, Mn, Mh
159

and Mle. All the masks are stacked to obtain the disentanglement map. Through the employment of
160

the aforementioned techniques and processes, we successfully achieve the disentanglement of various
161

degradation regions.
162

3.3
Degradation-Aware Contrastive Learning
163

For unpaired image translation, contrastive learning has validated its effectiveness for the preservation
164

of content. It targets to maximize the mutual information between patches in the same spatial location
165

from the generated image and the source image as below:
166

ℓ(v, v+, v−) = −log
exp(v · v+/τ)

exp(v · v+/τ) + PQ
n=1 exp(v · v−
n /τ)
,
(12)

v is the anchor that denotes the patch from the generated image. The positive example v+ corresponds
167

to the source image patch with the same location as the anchor v. The negative examples v−represent
168

5


---Page Break---
patches with locations distinct from that of the anchor v. Q denotes the total number of negative
169

examples. In our work, the key insight of degradation-aware contrastive learning lies in two folds: (1)
170

How to sample the anchor, positive, and negative examples. (2) How to manage the focus on different
171

negative examples.
172

Degradation-Aware Sampling. In this paper, N2D3 selects the anchor, positive, and negative patches
173

under the guidance of the disentanglement results. Initially, based on the disentanglement mask
174

obtained in the Section 3.2, we compute the patch count for different degradation types, denoting as
175

Ks, s ∈[1, 4]. Then, within each degradation region, the anchors v are randomly selected from the
176

patches of generated daytime images IN →D. The positive examples v+ are sampled from the same
177

locations with the anchors in the source nighttime images IN , and the negative examples v−are
178

randomly selected from other locations of IN . For each anchor, there is one corresponding positive
179

example and Ks negative examples. Subsequently, the sample set with the same degradation type
180

will be assigned weights and the contrastive loss will be computed in the following steps.
181

Degradation-Aware Reweighting. Despite the careful selection of anchor, positive, and negative
182

examples, the importance of anchor-negative pairs still differs within the same degradation. A known
183

principle of designing contrastive learning is that the hard anchor-negative pairs (i.e., the pairs with
184

high similarity) should assign higher attention. Thus, weighted contrastive learning can be formulated
185

as:
186

ℓ(v, v+, v−, wn) = −log
exp(v · v+/τ)

exp(v · v+/τ) + PQ
n=1 wn exp(v · v−
n /τ)
,
(13)

wn denotes the weight of the n-th anchor-negative pairs.
187

The contrastive objective is depicted in the Similarity Matrix in Figure 2. The patches in different
188

regions are obviously easy examples. We suppress their weights to 0, which transforms the similarity
189

matrix into a blocked diagonal matrix with diag(A1, . . . , A4). Within each degradation matrix
190

As, s ∈[1, 4], a soft reweighting strategy is implemented. Specifically, for each anchor-negative
191

pair, we apply optimal transport to yield an optimal transport plan, serving as a reweighting matrix
192

associated with the disentangled results. It can adaptively optimize and avoid manual design. The
193

reweight matrix for each degradation type is formulated as:
194

min
wij,i,j∈[1,Ks][

Ks
X

i=1

Ks
X

j=1,i̸=j
wij · exp (vi · v−
j /τ)],

Ks
X

i=1
wij = 1,

Ks
X

j=1
wij = 1, i, j ∈[1, Ks],

(14)

The aforementioned operations transform the contrastive objective to the Block Diagonal Similarity
195

Matrix depicted in Figure 2. As a common practice, our degradation-aware contrastive loss is applied
196

to the S layers of the CNN feature extractor, formulated as:
197

LDegNCE(F) =

S
X

l=1
ℓ(v, v+, v−, wn).
(15)

3.4
Other Regularizations
198

As a common practice, GANs are employed to bridge the domain gap between daytime and nighttime.
199

The adversarial loss is formulated as:
200

Ladv(F) = ||D(IN →D) −1||2
2,

Ladv(D) = ||D(ID) −1||2
2 + ||D(IN →D)||2
2,
(16)

where D denotes the discriminator network. The final loss function is formatted as :
201

L(F) = Ladv(F) + LDegNCE(F),
L(D) = Ladv(D).
(17)

6


---Page Break---
4
Experiments
202

4.1
Experimental Settings
203

Datasets. Experiments are conducted on the two public datasets BDD100K [29] and Alderley [19].
204

Alderley dataset consists of images captured along the same route twice: once on a sunny day and
205

another time during a stormy rainy night. The nighttime images in this dataset are often blurry due to
206

the rainy conditions, which makes Night2Day challenging. BDD100K dataset is a large-scale high-
207

resolution autonomous driving dataset. It comprises 100,000 video clips under various conditions.
208

For each video, a keyframe is selected and meticulously annotated with details. We reorganized this
209

dataset based on its annotations, resulting in 27,971 night images for training and 3,929 night images
210

for evaluation.
211

Evaluation Metric. Following common practice, we utilize the Fréchet Inception Distance (FID)
212

scores [7] to assess whether the generated images align with the target distribution. This assessment
213

helps determine if a model effectively transforms images from the night domain to the day domain.
214

Additionally, we seek to understand the extent to which the generated daytime images maintain
215

structural consistency compared to the original inputs. To measure this, we employ SIFT scores,
216

mIoU scores and LPIPS distance [32].
217

DownStream Vision Task. Two downstream tasks are conducted. In the Alderley dataset, GPS
218

annotations indicate the locations of two images, one in the nighttime and the other in the daytime,
219

as the same. We calculate the number of SIFT-detected key points between the generated daytime
220

images and their corresponding daytime images to measure if the two images represent the same
221

location. The BDD100K dataset includes 329 night images with semantic annotations. We employ
222

Deeplabv3 pretrained on the Cityscapes dataset as the semantic segmentation model [2], then perform
223

inference on our generated daytime images without any additional training and compute the mIoU
224

(mean Intersection over Union).
225

Table 1: The quantitative results on Alderley and BDD100k. ↓means lower result is better. ↑means
higher is better.

Dataset
Alderley
BDD100k
Methods
FID↓
LPIPS↓
SIFT↑
FID↓
LPIPS↓
mIoU↑
Original
Conf./Jour.
210
-
3.12
101
-
15.63
CycleGAN[34]
ICCV 2017
167
0.706
3.36
51.7
0.477
13.42
StarGAN[3]
CVPR 2018
117
-
3.28
68.3
-
-
ToDayGAN[1]
ICRA 2019
104
0.770
4.14
43.8
0.577
16.77
UGATIT[15]
ICLR 2020
170
-
2.51
72.2
-
-
CUT[20]
ECCV 2020
64.7
0.707
6.78
55.5
0.583
9.30
ForkGAN[33]
ECCV 2020
61.2
0.759
12.1
37.6
0.581
11.81
AUGAN[18]
BMVC 2021
65.2
-
-
38.6
-
-
MoNCE[31]
CVPR 2022
72.7
0.737
6.35
40.2
0.502
17.21
Decent[27]
NIPS 2022
76.5
0.768
6.31
40.3
0.582
10.49
Santa[28]
CVPR 2023
67.1
0.757
6.93
36.9
0.559
11.03
N2D-LPNet[5]
CVPR 2023
-
-
-
69.1
-
-
EnlightenGAN [13]
TIP 2021
209.8
-
2.00
103.5
-
16.10
Zero-DCE [6]
TPAMI 2022
246.4
-
4.34
90.5
-
15.90
DeLight [21]
ECCV 2022
222.9
-
3.07
113.8
-
14.48
LLformer [23]
AAAI 2023
275.6
-
7.62
123.1
-
15.28
WCDM [12]
ToG 2023
239.6
-
7.10
124.3
-
16.32
GSAD [8]
NIPS 2023
214.7
-
6.29
116.0
-
15.76
N2D3(Ours)
-
50.9
0.650
16.62
31.5
0.466
21.58

4.2
Results on Alderley
226

We first apply Night2Day on the Alderley dataset, a challenging collection of nighttime images
227

captured on rainy nights. In Figure 4, we present a visual comparison of the results. CycleGAN [34]
228

and CUT [20] manage to preserve the general structural information of the entire image but often
229

lose many fine details. ToDayGAN [1], ForkGAN [33], Decent [27], and Santa [28] tend to miss
230

important elements such as cars in their results.
231

In Table 1, thirteen translation methods and three enhancement methods are compared, considering
232

both visual effects and keypoint matching metrics. Our method showcases an improvement of 10.3
233

7


---Page Break---
Real Night
CycleGAN
ForkGAN
ToDayGAN

N2D3 (ours)
Decent
Santa
CUT

Figure 4: The qualitative comparison results on the Alderley dataset.

Real Night
CycleGAN
ForkGAN
ToDayGAN

N2D3 (ours)
Decent
Santa
CUT

Figure 5: The qualitative comparison results on the BDD100K dataset.

in FID scores and 4.52 in SIFT scores compared to the previous state-of-the-art. This suggests that
234

N2D3 successfully achieves photorealistic daytime image generation, underscoring its potential for
235

robotic localization applications. The qualitative comparison results are demonstrated in Figure 4. In
236

conclusion, N2D3 achieves top scores in both FID and LPIPS metrics, demonstrating its superiority
237

in the Night2Day task. N2D3 excels in generating photorealistic daytime images while effectively
238

preserving structures, even in challenging scenarios such as rainy nights in the Alderley.
239

4.3
Results on BDD100K
240

We conducted experiments on a larger-scale dataset, BDD100K, focusing on more general night
241

scenes. The qualitative results can be found in Figure 5. CycleGAN, ToDayGAN, and CUT succeed
242

in preserving the structure in well-lit regions. ForkGAN, Santa, and Decent demonstrate poor
243

performance in such challenging scenes. Regretfully, none of them excel in handling light effects and
244

exhibit weak performance in maintaining global structures. With a customized design specifically
245

addressing light effects, our method successfully preserves the structure in all regions.
246

The quantitative results are presented in Table 1. As the scale of the dataset increases, all the
247

compared methods show an improvement in their performance. Notably, N2D3 demonstrates the best
248

performance with a significant improvement of 5.4 in FID scores, showcasing its ability to handle a
249

broader range of nighttime scenes and establishing itself as the most advanced method in this domain.
250

We also investigate the potential of Night2Day in enhancing downstream vision tasks in nighttime
251

environments using the BDD100K dataset. The quantitative results are summarized in Table 1.
252

The enhancement methods demonstrate a slight improvement in segmentation results, while some
253

image-to-image translation methods have a negative impact on performance. N2D3 exhibits the best
254

performance in enhancing nighttime semantic segmentation with a remarkable improvement of
255

5.95 in mIoU compared to inferring the segmentation model directly on nighttime images.
256

In conclusion, N2D3 achieves top scores in both FID and LPIPS metrics, establishing itself as the
257

most advanced method for the Night2Day task. It excels in generating photorealistic daytime images
258

while preserving local and global structures. Moreover, the substantial improvement in nighttime
259

semantic segmentation highlights its benefits for downstream tasks and its potential for wide-ranging
260

applications.
261

8


---Page Break---
50.8

45.4

31.5

35.4
36.7

65.0

59.4

50.9

58.2

53.9

20

30

40

50

60

70

64
128
256
512
1024

FID

num samping

BDD100K
Alderley

0.582
0.579

0.466

0.508
0.491

0.681
0.673
0.650

0.685

0.739

0.4

0.5

0.6

0.7

0.8

64
128
256
512
1024

LPIPS

num samping

BDD100K
Alderley

(a) Ablation in FID
(b) Ablation in LPIPS

Figure 6: The quantitative results of ablation on the number of patches of the degradation-aware
sampling.
Table 2: The quantitative results of ablation on the main component of degradation-aware con-
trastive learning. (a) denotes the degradation-aware sampling, and (b) denotes the degradation-aware
reweighting. L and N denotes the invariant types.

Main Component
BDD100K
Alderley
(a)
(b)
FID
LPIPS
FID
LPIPS
SIFT
%
%
55.5
0.583
64.7
0.707
6.78
!
%
36.9
0.495
56.6
0.698
16.52
!
!
31.5
0.466
50.9
0.650
16.62

Invariant Type
BDD100K
Alderley
L
N
FID
LPIPS
FID
LPIPS
SIFT
%
%
55.5
0.583
64.7
0.707
6.78
!
%
49.1
0.592
62.9
0.726
9.83
!
!
31.5
0.466
50.9
0.650
16.62

4.4
Ablation Study
262

Ablation on the main component of degradation-aware contrastive learning. The core design of
263

the degradation-aware contrastive learning module relies on two main components: (a) degradation-
264

aware sampling, and (b) degradation-aware reweighting. As shown in Table 2, when degradation-
265

aware sampling is exclusively activated, there is a noticeable decrease in FID on both datasets
266

compared to the baseline (no components activated). Notably, the combination of degradation-aware
267

sampling and reweighting achieves the lowest FID on both BDD100K and Alderley, indicating the
268

effectiveness of degradation-aware sampling in conjunction with degradation-aware reweighting.
269

Ablation on the number of patches in the degradation-aware sampling. To explore the impact
270

of the number of sampling patches in our method, we conduct an ablation study on the number of
271

sampling patches with settings of 64, 128, 256, 512, and 1024 for degradation-aware sampling. The
272

FID and LPIPS scores are evaluated, as shown in Figure 6. The optimal performance is achieved with
273

256 patches, and increasing the number of sampling patches beyond this point leads to a degradation
274

in performance.
275

Ablation on the type of the invariant in disentanglement. To explore different invariants for
276

obtaining degradation-disentangled prototypes, we conduct an ablation study on the type of invariant.
277

As shown in Table 2, when L is enabled, the FID decreases from 55.5 to 49.1 on BDD100K and
278

from 64.7 to 62.9 on Alderley. This suggests that incorporating illuminance maps helps in reducing
279

the perceptual gap between generated and source nighttime images. When N is activated, there
280

is a consistent improvement in FID on both datasets, indicating that considering physical priors
281

invariant contributes to more realistic image generation. The combination of both illuminance map
282

and physical prior invariant results in the lowest FID on both datasets, showcasing the complementary
283

nature of these degradation types in improving contrastive learning.
284

5
Conclusion
285

This paper introduces a novel solution for the Night2Day image translation task, focusing on trans-
286

lating nighttime images to their corresponding daytime counterparts while preserving semantic
287

consistency. To achieve this objective, the proposed method begins by disentangling the degradation
288

presented in nighttime images, which is the key insight of our method. To achieve this, we contribute
289

a degradation disentanglement module and a degradation-aware contrastive learning module. Our
290

method outperforms the existing state-of-the-art, which shows the effectiveness of N2D3 and the
291

superiority of the insight to disentangle the degradation.
292

9


---Page Break---
References
293

[1] Asha Anoosheh, Torsten Sattler, Radu Timofte, Marc Pollefeys, and Luc Van Gool. Night-to-day
294

image translation for retrieval-based localization. In 2019 International Conference on Robotics
295

and Automation (ICRA), pages 5958–5964. IEEE, 2019.
296

[2] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous
297

convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.
298

[3] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo.
299

Stargan: Unified generative adversarial networks for multi-domain image-to-image translation.
300

In Proc. IEEE Conference on Computer Vision and Pattern Recognition, pages 8789–8797,
301

2018.
302

[4] T. Cover and P. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information
303

Theory, 13(1):21–27, 1967.
304

[5] Zhentao Fan, Xianhao Wu, Xiang Chen, and Yufeng Li. Learning to see in nighttime driving
305

scenes with inter-frequency priors. In Proceedings of the IEEE/CVF Conference on Computer
306

Vision and Pattern Recognition, pages 4217–4224, 2023.
307

[6] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and
308

Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proc.
309

IEEE Conference on Computer Vision and Pattern Recognition, pages 1780–1789, 2020.
310

[7] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
311

Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in
312

Neural Information Processing Systems, 30, 2017.
313

[8] Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, and Hui Yuan. Global structure-
314

aware diffusion process for low-light image enhancement. Advances in Neural Information
315

Processing Systems, 36, 2024.
316

[9] Xueqi Hu, Xinyue Zhou, Qiusheng Huang, Zhengyi Shi, Li Sun, and Qingli Li. Qs-attn: Query-
317

selected attention for contrastive learning in i2i translation. In Proceedings of the IEEE/CVF
318

Conference on Computer Vision and Pattern Recognition, pages 18291–18300, 2022.
319

[10] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei Efros. Image-to-image translation with
320

conditional adversarial networks. In Proc. IEEE Conference on Computer Vision and Pattern
321

Recognition, pages 1125–1134, 2017.
322

[11] Somi Jeong, Youngjung Kim, Eungbean Lee, and Kwanghoon Sohn. Memory-guided unsuper-
323

vised image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer
324

Vision and Pattern Recognition, pages 6558–6567, 2021.
325

[12] Hai Jiang, Ao Luo, Haoqiang Fan, Songchen Han, and Shuaicheng Liu. Low-light image
326

enhancement with wavelet-based diffusion models. ACM Transactions on Graphics (TOG),
327

42(6):1–14, 2023.
328

[13] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan
329

Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision.
330

IEEE Transactions on Image Processing, 30:2340–2349, 2021.
331

[14] Mikhail Kennerley, Jian-Gang Wang, Bharadwaj Veeravalli, and Robby T Tan. 2pcnet: Two-
332

phase consistency training for day-to-night unsupervised domain adaptive object detection. In
333

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
334

11484–11493, 2023.
335

[15] Junho Kim, Minjae Kim, Hyeonwoo Kang, and Kwanghee Lee. U-gat-it: Unsupervised
336

generative attentional networks with adaptive layer-instance normalization for image-to-image
337

translation. arXiv preprint arXiv:1907.10830, 2019.
338

10


---Page Break---
[16] Soohyun Kim, Jongbeom Baek, Jihye Park, Gyeongnyeon Kim, and Seungryong Kim. In-
339

staformer: Instance-aware image-to-image translation with transformer. In Proceedings of
340

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18321–18331,
341

2022.
342

[17] Paul Kubelka. Ein beitrag zur optik der farbanstriche (contribution to the optic of paint).
343

Zeitschrift fur technische Physik, 12:593–601, 1931.
344

[18] Jeong-gi Kwak, Youngsaeng Jin, Yuanming Li, Dongsik Yoon, Donghyeon Kim, and Hanseok
345

Ko. Adverse weather image translation with asymmetric and uncertainty-aware gan. arXiv
346

preprint arXiv:2112.04283, 2021.
347

[19] Michael J. Milford and Gordon. F. Wyeth. Seqslam: Visual route-based navigation for sunny
348

summer days and stormy winter nights. In 2012 IEEE International Conference on Robotics
349

and Automation, pages 1643–1649, 2012.
350

[20] Taesung Park, Alexei Efros, Richard Zhang, and Jun-Yan Zhu. Contrastive learning for unpaired
351

image-to-image translation. In European Conference on Computer Vision, pages 319–345,
352

2020.
353

[21] Aashish Sharma and Robby T Tan. Nighttime visibility enhancement by increasing the dynamic
354

range and suppression of light effects. In Proceedings of the IEEE/CVF Conference on Computer
355

Vision and Pattern Recognition, pages 11977–11986, 2021.
356

[22] Seokbeom Song, Suhyeon Lee, Hongje Seong, Kyoungwon Min, and Euntai Kim. Shunit: Style
357

harmonization for unpaired image-to-image translation. Proceedings of the AAAI Conference
358

on Artificial Intelligence, 37(2):2292–2302, Jun. 2023.
359

[23] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Tong Lu. Ultra-
360

high-definition low-light image enhancement: A benchmark and transformer-based method. In
361

Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2654–2662,
362

2023.
363

[24] Weilun Wang, Wengang Zhou, Jianmin Bao, Dong Chen, and Houqiang Li. Instance-wise hard
364

negative example generation for contrastive learning in unpaired image-to-image translation.
365

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14020–
366

14029, 2021.
367

[25] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment:
368

from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–
369

612, 2004.
370

[26] Youya Xia, Josephine Monica, Wei-Lun Chao, Bharath Hariharan, Kilian Q Weinberger, and
371

Mark Campbell. Image-to-image translation for autonomous driving from coarsely-aligned
372

image pairs. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages
373

7756–7762. IEEE, 2023.
374

[27] Shaoan Xie, Qirong Ho, and Kun Zhang. Unsupervised image-to-image translation with density
375

changing regularization. In Advances in Neural Information Processing Systems, 2022.
376

[28] Shaoan Xie, Yanwu Xu, Mingming Gong, and Kun Zhang. Unpaired image-to-image translation
377

with shortest path regularization. In Proceedings of the IEEE/CVF Conference on Computer
378

Vision and Pattern Recognition (CVPR), pages 10177–10187, June 2023.
379

[29] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht
380

Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous mul-
381

titask learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern
382

recognition, pages 2636–2645, 2020.
383

[30] Zhenjie Yu, Shuang Li, Yirui Shen, Chi Harold Liu, and Shuigen Wang. On the difficulty of
384

unpaired infrared-to-visible video translation: Fine-grained content-rich patches transfer. In
385

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
386

pages 1631–1640, June 2023.
387

11


---Page Break---
[31] Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Rongliang Wu, and Shijian Lu. Modulated contrast
388

for versatile image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision
389

and Pattern Recognition (CVPR), pages 18280–18290, June 2022.
390

[32] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unrea-
391

sonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE
392

conference on computer vision and pattern recognition, pages 586–595, 2018.
393

[33] Ziqiang Zheng, Yang Wu, Xinran Han, and Jianbo Shi. Forkgan: Seeing into the rainy night. In
394

European conference on computer vision, pages 155–170. Springer, 2020.
395

[34] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei Efros. Unpaired image-to-image transla-
396

tion using cycle-consistent adversarial networks. In Proc. IEEE International Conference on
397

Computer Vision, pages 2223–2232, 2017.
398

12


---Page Break---
A
Overview
399

This supplementary material is organized as follows. Appendix B provides additional details about
400

the proof that the invariant Nλmxn is exclusively related to the illumination. Appendix C outlines the
401

limitations and failure case of N2D3. Appendix D illustrates the implementation details, including
402

N2D3 and other methods used in the experiments. Appendix E presents additional visualization
403

results.
404

B
More Proof Details
405

We provide a detailed proof process to demonstrate how the invariant Nλmxn is exclusively related
406

to the illumination and can function as the light effect detector. First, consider the following
407

equations, corresponding to Equation (5) in the main paper:
408

Nλmxn =
∂m+n−2

∂λm−1∂xn−1
∂
∂x{
1
E(λ, x)
∂E(λ, x)

∂λ
}

=
∂m+n−2

∂λm−1∂xn−1
∂
∂x{
1
e(λ, x)
∂e(λ, x)

∂λ
+
1
R(λ)C(x)
∂R(λ)C(x)

∂λ
},
(18)

by applying the additivity of linear differential operators, the first term represents the invariants only
409

related to the illumination. The second term can be simplified by applying the chain rule as follows:
410

∂
∂x{
1
R(λ)C(x)
∂R(λ)C(x)

∂λ
}

=
1
R(λ)2C(x)2 (∂2{R(λ)C(x)}

∂λ∂x
· R(λ)C(x) −∂{R(λ)C(x)}
∂λ
· ∂{R(λ)C(x)}
∂x
)

=
1
R(λ)2C(x)2 (∂R(λ)

∂λ
∂C(x)

∂x
· R(λ)C(x) −∂R(λ)
∂λ
C(x) · R(λ)∂C(x)

∂x
) = 0.

(19)

Finally, we conclude that the invariant Nλmxn is exclusively related to the illumination and can be
411

formulated as follows:
412

Nλmxn =
∂m+n−2

∂λm−1∂xn−1
∂
∂x{
1
E(λ, x)
∂E(λ, x)

∂λ
}

=
∂m+n−1

∂λm−1∂xn {
1
e(λ, x)
∂e(λ, x)

∂λ
}.
(20)

Figure 7: Failure Cases of N2D3: Our method struggles to handle various other types of degradation.

C
Limitations and Failure Case
413

Despite the superior performance of N2D3 in Night2Day, it still exhibits certain limitations. On the
414

one hand, this work focuses solely on addressing light degradation, while nighttime environments
415

encompass various other types of degradation, including blur caused by rain, motion, and other
416

13


---Page Break---
Figure 8: More disentanglement results. The first and third rows display nighttime images, while
the second and fourth rows show the corresponding degradation disentanglement results. The color
progression from blue, light blue, green to yellow corresponds to the following regions: darkness,
well-lit, light effects, and high-light.

w/o N
w/o reweighting
N2D3 (ours)
Real Night

Figure 9: Qualitative comparison abalation results.

factors. Our method currently struggles to handle these situations effectively. On the other hand, the
417

limitations of visible imaging in night vision arise from the scarcity of photos captured in low-light
418

conditions, as illustrated by the failure cases presented inFigure 7. Future advancements in night
419

vision will likely incorporate additional modalities, such as infrared images, radar, and other sensor
420

data, to overcome these challenges and improve performance.
421

D
Implementation Details
422

Training Details. We adopt the resnet 9blocks, a ResNetbased model with nine residual blocks, as
423

the backbone for generator G. Additionally, we utilize the patch-wise discriminator D following
424

PatchGAN[10]. To conduct degradation-aware contrastive learning on multiple layers, we extract
425

features from 5 layers of the generator G encoder, as done in [20]. These layers include RGB pixels,
426

the first and second downsampling convolution, and the first and fifth residual block. For the features
427

of each layer, we apply a 2-layer MLP to acquire final 256-dimensional features. These features are
428

then utilized in our degradation-aware contrastive learning.
429

All the comparison methods are reproduced using their released source code with default settings.
430

Training procedures are consistent across all methods. All models are trained using the Adaptive
431

Moment Estimation optimizer with an initial learning rate of 10−4, a momentum of 0.9, and weight
432

decay of 10−4. For the BDD100K dataset, training consists of 10 epochs with the initial learning
433

rate, followed by another 10 epochs with a decreased learning rate using the polynomial annealing
434

procedure with a power of 0.9. On the Alderley dataset, given the limited training data compared
435

to BDD100K, we extend the training to 20 epochs with the initial learning rate and an additional
436

14


---Page Break---
CUT
N2D3 (ours)
Decent
Santa

Real Night
CycleGAN
ForkGAN
ToDayGAN

Figure 10: More qualitative comparison results on the Alderley dataset.

15


---Page Break---
20 epochs with the decayed learning rate. All the experiments are run on a single A100 GPU with
437

80GB of memory. Training our method with a smaller patch size and batch size on a device with less
438

memory is feasible.
439

Evaluation Details. In the evaluation, we compute the Fréchet Inception Distance (FID) [7],
440

Structural Similarity Index (SSIM) [25], and Learned Perceptual Image Patch Similarity (LPIPS)
441

[32] scores on 256 × 512 images. Partial FID scores are provided by ForkGAN [33], and all SSIM
442

and LPIPS scores are reproduced by us.
443

Semantic segmentation evaluation are conducted as follows. First, we use Deeplabv3 pretrained
444

on the Cityscapes dataset as the semantic segmentation model [2]. The model is provided by
445

https://github.com/open-mmlab/mmsegmentation with an R-18-D8 backbone and trained at
446

a resolution of 512 × 1024. Second, we perform 512 × 1024 Night2Day translation to obtain the
447

generation results. Finally, we infer the semantic segmentation on the generated daytime images.
448

E
More Visualization Results
449

More Ablation Visualization Results. We provide ablation visualization results on both Alderley
450

and BDD100K in Figure 9. The complete method is presented along with ablation studies on the
451

invariant N and without degradation-aware reweighting. All the modules contribute to improving the
452

ability to maintain semantic consistency.
453

More Disentanglement Results. We provide additional disentanglement results in Figure 8. Our
454

disentanglement methods offer a comprehensive representation of different illumination degradation
455

types in various nighttime scenes.
456

More Qualitative Comparison. We present more qualitative comparisons in Figure 10 and Figure 11
457

alongside other methods.Our method demonstrates visually pleasing results under various nighttime
458

conditions.
459

16


---Page Break---
Real Night
CycleGAN
ToDayGAN
ForkGAN

CUT
N2D3 (ours)
Decent
Santa

Figure 11: More qualitative comparison results on the BDD100K dataset.

17


---Page Break---
NeurIPS Paper Checklist
460

1. Claims
461

Question: Do the main claims made in the abstract and introduction accurately reflect the
462

paper’s contributions and scope?
463

Answer: [Yes]
464

Justification: We claim our main contribution as N2D3, which achieves SOTA performance
465

by bridging the domain gap between nighttime and daytime in a degradation-aware manner.
466

Guidelines:
467

• The answer NA means that the abstract and introduction do not include the claims
468

made in the paper.
469

• The abstract and/or introduction should clearly state the claims made, including the
470

contributions made in the paper and important assumptions and limitations. A No or
471

NA answer to this question will not be perceived well by the reviewers.
472

• The claims made should match theoretical and experimental results, and reflect how
473

much the results can be expected to generalize to other settings.
474

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
475

are not attained by the paper.
476

2. Limitations
477

Question: Does the paper discuss the limitations of the work performed by the authors?
478

Answer: [Yes]
479

Justification: We discuss our limitation in degradations beyond light and low-light image
480

scarcity in the appendix.
481

Guidelines:
482

• The answer NA means that the paper has no limitation while the answer No means that
483

the paper has limitations, but those are not discussed in the paper.
484

• The authors are encouraged to create a separate "Limitations" section in their paper.
485

• The paper should point out any strong assumptions and how robust the results are to
486

violations of these assumptions (e.g., independence assumptions, noiseless settings,
487

model well-specification, asymptotic approximations only holding locally). The authors
488

should reflect on how these assumptions might be violated in practice and what the
489

implications would be.
490

• The authors should reflect on the scope of the claims made, e.g., if the approach was
491

only tested on a few datasets or with a few runs. In general, empirical results often
492

depend on implicit assumptions, which should be articulated.
493

• The authors should reflect on the factors that influence the performance of the approach.
494

For example, a facial recognition algorithm may perform poorly when image resolution
495

is low or images are taken in low lighting. Or a speech-to-text system might not be
496

used reliably to provide closed captions for online lectures because it fails to handle
497

technical jargon.
498

• The authors should discuss the computational efficiency of the proposed algorithms
499

and how they scale with dataset size.
500

• If applicable, the authors should discuss possible limitations of their approach to
501

address problems of privacy and fairness.
502

• While the authors might fear that complete honesty about limitations might be used by
503

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
504

limitations that aren’t acknowledged in the paper. The authors should use their best
505

judgment and recognize that individual actions in favor of transparency play an impor-
506

tant role in developing norms that preserve the integrity of the community. Reviewers
507

will be specifically instructed to not penalize honesty concerning limitations.
508

3. Theory Assumptions and Proofs
509

Question: For each theoretical result, does the paper provide the full set of assumptions and
510

a complete (and correct) proof?
511

18


---Page Break---
Answer: [Yes]
512

Justification: We provide the full set of assumptions and complete proofs in both Section 3.1
513

and Appendix B .
514

Guidelines:
515

• The answer NA means that the paper does not include theoretical results.
516

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
517

referenced.
518

• All assumptions should be clearly stated or referenced in the statement of any theorems.
519

• The proofs can either appear in the main paper or the supplemental material, but if
520

they appear in the supplemental material, the authors are encouraged to provide a short
521

proof sketch to provide intuition.
522

• Inversely, any informal proof provided in the core of the paper should be complemented
523

by formal proofs provided in appendix or supplemental material.
524

• Theorems and Lemmas that the proof relies upon should be properly referenced.
525

4. Experimental Result Reproducibility
526

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
527

perimental results of the paper to the extent that it affects the main claims and/or conclusions
528

of the paper (regardless of whether the code and data are provided or not)?
529

Answer: [Yes]
530

Justification: All the information needed to reproduce the main experimental results is
531

included in the Section 3 and Appendix D.
532

Guidelines:
533

• The answer NA means that the paper does not include experiments.
534

• If the paper includes experiments, a No answer to this question will not be perceived
535

well by the reviewers: Making the paper reproducible is important, regardless of
536

whether the code and data are provided or not.
537

• If the contribution is a dataset and/or model, the authors should describe the steps taken
538

to make their results reproducible or verifiable.
539

• Depending on the contribution, reproducibility can be accomplished in various ways.
540

For example, if the contribution is a novel architecture, describing the architecture fully
541

might suffice, or if the contribution is a specific model and empirical evaluation, it may
542

be necessary to either make it possible for others to replicate the model with the same
543

dataset, or provide access to the model. In general. releasing code and data is often
544

one good way to accomplish this, but reproducibility can also be provided via detailed
545

instructions for how to replicate the results, access to a hosted model (e.g., in the case
546

of a large language model), releasing of a model checkpoint, or other means that are
547

appropriate to the research performed.
548

• While NeurIPS does not require releasing code, the conference does require all submis-
549

sions to provide some reasonable avenue for reproducibility, which may depend on the
550

nature of the contribution. For example
551

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
552

to reproduce that algorithm.
553

(b) If the contribution is primarily a new model architecture, the paper should describe
554

the architecture clearly and fully.
555

(c) If the contribution is a new model (e.g., a large language model), then there should
556

either be a way to access this model for reproducing the results or a way to reproduce
557

the model (e.g., with an open-source dataset or instructions for how to construct
558

the dataset).
559

(d) We recognize that reproducibility may be tricky in some cases, in which case
560

authors are welcome to describe the particular way they provide for reproducibility.
561

In the case of closed-source models, it may be that access to the model is limited in
562

some way (e.g., to registered users), but it should be possible for other researchers
563

to have some path to reproducing or verifying the results.
564

5. Open access to data and code
565

19


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
566

tions to faithfully reproduce the main experimental results, as described in supplemental
567

material?
568

Answer: [No]
569

Justification: Code will be released latter.
570

Guidelines:
571

• The answer NA means that paper does not include experiments requiring code.
572

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
573

public/guides/CodeSubmissionPolicy) for more details.
574

• While we encourage the release of code and data, we understand that this might not be
575

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
576

including code, unless this is central to the contribution (e.g., for a new open-source
577

benchmark).
578

• The instructions should contain the exact command and environment needed to run to
579

reproduce the results. See the NeurIPS code and data submission guidelines (https:
580

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
581

• The authors should provide instructions on data access and preparation, including how
582

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
583

• The authors should provide scripts to reproduce all experimental results for the new
584

proposed method and baselines. If only a subset of experiments are reproducible, they
585

should state which ones are omitted from the script and why.
586

• At submission time, to preserve anonymity, the authors should release anonymized
587

versions (if applicable).
588

• Providing as much information as possible in supplemental material (appended to the
589

paper) is recommended, but including URLs to data and code is permitted.
590

6. Experimental Setting/Details
591

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
592

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
593

results?
594

Answer: [Yes]
595

Justification: The training details and dataset information are provided in Section 4.
596

Guidelines:
597

• The answer NA means that the paper does not include experiments.
598

• The experimental setting should be presented in the core of the paper to a level of detail
599

that is necessary to appreciate the results and make sense of them.
600

• The full details can be provided either with the code, in appendix, or as supplemental
601

material.
602

7. Experiment Statistical Significance
603

Question: Does the paper report error bars suitably and correctly defined or other appropriate
604

information about the statistical significance of the experiments?
605

Answer: [No]
606

Justification: Error bars are not reported because it would be too computationally expensive.
607

We report our results using a fixed random seed.
608

Guidelines:
609

• The answer NA means that the paper does not include experiments.
610

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
611

dence intervals, or statistical significance tests, at least for the experiments that support
612

the main claims of the paper.
613

• The factors of variability that the error bars are capturing should be clearly stated (for
614

example, train/test split, initialization, random drawing of some parameter, or overall
615

run with given experimental conditions).
616

20


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
617

call to a library function, bootstrap, etc.)
618

• The assumptions made should be given (e.g., Normally distributed errors).
619

• It should be clear whether the error bar is the standard deviation or the standard error
620

of the mean.
621

• It is OK to report 1-sigma error bars, but one should state it. The authors should
622

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
623

of Normality of errors is not verified.
624

• For asymmetric distributions, the authors should be careful not to show in tables or
625

figures symmetric error bars that would yield results that are out of range (e.g. negative
626

error rates).
627

• If error bars are reported in tables or plots, The authors should explain in the text how
628

they were calculated and reference the corresponding figures or tables in the text.
629

8. Experiments Compute Resources
630

Question: For each experiment, does the paper provide sufficient information on the com-
631

puter resources (type of compute workers, memory, time of execution) needed to reproduce
632

the experiments?
633

Answer: [Yes]
634

Justification: We report the compute resources in Appendix D.
635

Guidelines:
636

• The answer NA means that the paper does not include experiments.
637

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
638

or cloud provider, including relevant memory and storage.
639

• The paper should provide the amount of compute required for each of the individual
640

experimental runs as well as estimate the total compute.
641

• The paper should disclose whether the full research project required more compute
642

than the experiments reported in the paper (e.g., preliminary or failed experiments that
643

didn’t make it into the paper).
644

9. Code Of Ethics
645

Question: Does the research conducted in the paper conform, in every respect, with the
646

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
647

Answer: [Yes]
648

Justification: The research conducted in this paper conforms, in every respect, with the
649

NeurIPS Code of Ethics.
650

Guidelines:
651

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
652

• If the authors answer No, they should explain the special circumstances that require a
653

deviation from the Code of Ethics.
654

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
655

eration due to laws or regulations in their jurisdiction).
656

10. Broader Impacts
657

Question: Does the paper discuss both potential positive societal impacts and negative
658

societal impacts of the work performed?
659

Answer: [Yes]
660

Justification: The societal impacts are discussed in the manuscript and appendix.
661

Guidelines:
662

• The answer NA means that there is no societal impact of the work performed.
663

• If the authors answer NA or No, they should explain why their work has no societal
664

impact or why the paper does not address societal impact.
665

21


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
666

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
667

(e.g., deployment of technologies that could make decisions that unfairly impact specific
668

groups), privacy considerations, and security considerations.
669

• The conference expects that many papers will be foundational research and not tied
670

to particular applications, let alone deployments. However, if there is a direct path to
671

any negative applications, the authors should point it out. For example, it is legitimate
672

to point out that an improvement in the quality of generative models could be used to
673

generate deepfakes for disinformation. On the other hand, it is not needed to point out
674

that a generic algorithm for optimizing neural networks could enable people to train
675

models that generate Deepfakes faster.
676

• The authors should consider possible harms that could arise when the technology is
677

being used as intended and functioning correctly, harms that could arise when the
678

technology is being used as intended but gives incorrect results, and harms following
679

from (intentional or unintentional) misuse of the technology.
680

• If there are negative societal impacts, the authors could also discuss possible mitigation
681

strategies (e.g., gated release of models, providing defenses in addition to attacks,
682

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
683

feedback over time, improving the efficiency and accessibility of ML).
684

11. Safeguards
685

Question: Does the paper describe safeguards that have been put in place for responsible
686

release of data or models that have a high risk for misuse (e.g., pretrained language models,
687

image generators, or scraped datasets)?
688

Answer: [NA]
689

Justification: Our model does not have such risks, and all the datasets used in the experiments
690

are open-source benchmarks in this field.
691

Guidelines:
692

• The answer NA means that the paper poses no such risks.
693

• Released models that have a high risk for misuse or dual-use should be released with
694

necessary safeguards to allow for controlled use of the model, for example by requiring
695

that users adhere to usage guidelines or restrictions to access the model or implementing
696

safety filters.
697

• Datasets that have been scraped from the Internet could pose safety risks. The authors
698

should describe how they avoided releasing unsafe images.
699

• We recognize that providing effective safeguards is challenging, and many papers do
700

not require this, but we encourage authors to take this into account and make a best
701

faith effort.
702

12. Licenses for existing assets
703

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
704

the paper, properly credited and are the license and terms of use explicitly mentioned and
705

properly respected?
706

Answer: [Yes]
707

Justification: The code and data are properly credited, and the license and terms of use are
708

explicitly mentioned and properly documented.
709

Guidelines:
710

• The answer NA means that the paper does not use existing assets.
711

• The authors should cite the original paper that produced the code package or dataset.
712

• The authors should state which version of the asset is used and, if possible, include a
713

URL.
714

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
715

• For scraped data from a particular source (e.g., website), the copyright and terms of
716

service of that source should be provided.
717

22


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the
718

package should be provided. For popular datasets, paperswithcode.com/datasets
719

has curated licenses for some datasets. Their licensing guide can help determine the
720

license of a dataset.
721

• For existing datasets that are re-packaged, both the original license and the license of
722

the derived asset (if it has changed) should be provided.
723

• If this information is not available online, the authors are encouraged to reach out to
724

the asset’s creators.
725

13. New Assets
726

Question: Are new assets introduced in the paper well documented and is the documentation
727

provided alongside the assets?
728

Answer: [Yes]
729

Justification: The code introduced in the paper is well-documented, and the documentation
730

is provided alongside it.
731

Guidelines:
732

• The answer NA means that the paper does not release new assets.
733

• Researchers should communicate the details of the dataset/code/model as part of their
734

submissions via structured templates. This includes details about training, license,
735

limitations, etc.
736

• The paper should discuss whether and how consent was obtained from people whose
737

asset is used.
738

• At submission time, remember to anonymize your assets (if applicable). You can either
739

create an anonymized URL or include an anonymized zip file.
740

14. Crowdsourcing and Research with Human Subjects
741

Question: For crowdsourcing experiments and research with human subjects, does the paper
742

include the full text of instructions given to participants and screenshots, if applicable, as
743

well as details about compensation (if any)?
744

Answer: [NA]
745

Justification: The paper does not involve crowdsourcing nor research with human subjects.
746

Guidelines:
747

• The answer NA means that the paper does not involve crowdsourcing nor research with
748

human subjects.
749

• Including this information in the supplemental material is fine, but if the main contribu-
750

tion of the paper involves human subjects, then as much detail as possible should be
751

included in the main paper.
752

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
753

or other labor should be paid at least the minimum wage in the country of the data
754

collector.
755

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
756

Subjects
757

Question: Does the paper describe potential risks incurred by study participants, whether
758

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
759

approvals (or an equivalent approval/review based on the requirements of your country or
760

institution) were obtained?
761

Answer: [NA]
762

Justification: The paper does not involve crowdsourcing nor research with human subjects.
763

Guidelines:
764

• The answer NA means that the paper does not involve crowdsourcing nor research with
765

human subjects.
766

• Depending on the country in which research is conducted, IRB approval (or equivalent)
767

may be required for any human subjects research. If you obtained IRB approval, you
768

should clearly state this in the paper.
769

23


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
770

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
771

guidelines for their institution.
772

• For initial submissions, do not include any information that would break anonymity (if
773

applicable), such as the institution conducting the review.
774

24


---Page Break---
