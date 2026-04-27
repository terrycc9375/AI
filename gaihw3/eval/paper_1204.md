AdaFace – A Versatile Face Encoder for Zero-Shot
Diffusion Model Personalization

Anonymous Author(s)
Affiliation
Address
email

Reference “Jedi fighting pose with a lightsaber”
Reference “Dancing pose among folks in a park”

“Playing guitar on a boat, ocean waves”
“On a serene beach at sunset”

“Cybergirl, futuristic silver armor suit”
“In superman costume flying pose”

Figure 1: Although AdaFace is solely trained on static images, the subject embeddings it generates
can directly condition AnimateDiff to produce personalized videos across diverse scenes without
requiring any modifications.

Abstract

Since the advent of diffusion models, personalizing these models – conditioning
1

them to render novel subjects – has been widely studied. Recently, several methods
2

propose training a dedicated image encoder on a large variety of subject images.
3

This encoder maps the images to identity embeddings (ID embeddings). During
4

inference, these ID embeddings, combined with conventional prompts, condition a
5

diffusion model to generate new images of the subject. However, such methods
6

often face challenges in achieving a good balance between authenticity and compo-
7

sitionality – accurately capturing the subject’s likeness while effectively integrating
8

them into varied and complex scenes. A primary source for this issue is that the ID
9

embeddings reside in the image token space (“image prompts"), which is not fully
10

composable with the text prompt encoded by the CLIP text encoder. In this work,
11

we present AdaFace, an image encoder that maps human faces into the text prompt
12

space. After being trained only on 400K face images with 2 GPUs, it achieves high
13

authenticity of the generated subjects and high compositionality with various text
14

prompts. In addition, as the ID embeddings are integrated in a normal text prompt,
15

it is highly compatible with existing pipelines and can be used without modification
16

to generate authentic videos. We showcase the generated images and videos of
17

celebrities under various compositional prompts. The source code is released on an
18

anonymous repository https://github.com/adaface-neurips/adaface.
19

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Face2Vec

Initial Face Embedding

Face2Image 

Encoder

Custom U-Net

Input
Output

𝑣1
𝑣2
𝑣3
𝑣𝑛
⋯

Non-editable ID Embeddings in 

the image space

Figure 2: A typical zero-shot face encoder pipeline for diffusion models. First, a Face2Vec module
(e.g., ArcFace [Deng et al., 2019]) extracts a single vector that captures the facial features. Then a
trainable Face2Image encoder (e.g., Arc2Face [Papantoniou et al., 2024]) maps it to n facial tokens
v1, · · · , vn within the image embedding spaces. The facial tokens condition the U-Net (either original
or fine-tuned) to generate authentic-looking subject images. However, since the facial tokens is not
blended with other text prompts (sometimes they are simply concatenated), the whole pipeline has
weaker compositionality than using text prompts alone. Moreover, such models are often incompatible
with existing diffusion pipelines, such as AnimateDiff Guo et al. [2024a].

1
Introduction
20

Recent years have witnessed the blossom of diffusion models, which have been widely used in image
21

generation, image editing, and video generation [Ho et al., 2020, Nichol et al., 2022, Saharia et al.,
22

2022, Rombach et al., 2022, Podell et al., 2024, Chen et al., 2024a, Kawar et al., 2023, Peebles and
23

Xie, 2023, Guo et al., 2024a]. A particularly interesting application of these models is personalization,
24

where they are conditioned to generate images of specific subjects. Previously, this was primarily
25

achieved through test-time fine-tuning [Ruiz et al., 2022, Gal et al., 2022a, Kumari et al., 2022,
26

Tewel et al., 2023], which introduced additional computational demands and complexity to the
27

image generation process. Recent advancements have seen the development of zero-shot, tuning-free
28

methods [Wei et al., 2023, Ye et al., 2023, Shi et al., 2023, Wang et al., 2024, Papantoniou et al.,
29

2024, Guo et al., 2024b, Huang et al., 2024, Han et al., 2024, Chen et al., 2024b, He et al., 2024].
30

These methods train a dedicated image encoder to convert subject images to identity embeddings
31

(ID embeddings) using a large dataset. During inference, these ID embeddings are combined with
32

standard text prompts to generate new images of the subject (Figure 2). Despite these innovations,
33

these approaches often struggle to strike a good balance between authenticity and compositionality.
34

Authenticity ensures the model captures the true likeness of the subject, whereas compositionality
35

concerns the model’s ability to seamlessly integrate the subject into diverse and intricate scenes.
36

The challenge primarily stems from how ID embeddings are utilized: in many zero-shot methods,
37

the embeddings exist in the image token space (“image prompts") and do not fully mesh with text
38

prompts. In cases like [Huang et al., 2024], while the ID embeddings are within the text space, there
39

lacks targeted training to enhance their integration with other text prompts, resulting in compromised
40

compositionality.
41

Given the limitations of existing methods, we propose AdaFace, a versatile face encoder that maps
42

human faces into the text prompt space. First, the ID embeddings generated by AdaFace seamlessly
43

integrate with text prompts via the CLIP text encoder, allowing for more coherent and expressive
44

conditioning. Second, we employ targeted training strategies to enhance the compositionality of the ID
45

embeddings, ensuring they are able to be used to generate diverse and complex scenes. Furthermore,
46

AdaFace is highly compatible with existing diffusion pipelines, requiring no modifications to generate
47

authentic videos, as demonstrated in Figure 1. Notably, due to efficient model design and distillation
48

techniques, AdaFace is trained on merely 406,567 face images with 2 RTX A6000 GPUs, all within a
49

constrained compute budget.
50

We demonstrate the effectiveness of AdaFace by showcasing the generated images and videos of
51

celebrities under various compositional prompts. We also perform quantitative evaluations to validate
52

that AdaFace achieves a good balance between authenticity and compositionality, measured by
53

ArcFace similarity and CLIP-Text similarity, respectively.
54

2


---Page Break---
Initial Face Embedding

Face2Image 

Encoder

Original U-Net

Input

AdaFace Prompt 

Inverter

Original CLIP 
Prompt Encoder

Editable ID Embeddings 𝑤1~𝑤𝑛 in the text 
space, seamlessly compatible with a text prompt

𝑤1
𝑤2
𝑤3
𝑤𝑛
⋯
a photo of
in a chef outfit

Output

Non-editable ID Embeddings in 

the image space

To image space

To text space

𝑣1
𝑣2
𝑣3
𝑣𝑛
⋯

Face2Vec

Figure 3: The core of AdaFace is the Prompt Inverter, which inverts the image-space ID embeddings
from another model to the text prompt space, represented as w1, · · · , wn. These embeddings are
integrated into a standard text prompt and encoded by a CLIP prompt encoder. CLIP coherently
composes the semantics of the ID embeddings and the text prompt, providing good compositionality.

2
Method
55

Motivated by the advantages of text space face prompts, we propose techniques to distill one or more
56

image-space face encoders into the text space, and further enhance its compositionality. The overall
57

architecture of AdaFace is shown in Figure 3. The core module of AdaFace is the AdaFace Prompt
58

Inverter, which inverts the image-space ID embeddings to the text space, enabling the integration
59

of the ID embeddings into a standard text prompt. The ID embeddings are then encoded by a CLIP
60

prompt encoder, which coherently composes the semantics of the ID embeddings and the text prompt.
61

The text-level composition also facilitates Composition Distillation (Figure 5), which significantly
62

improves the compositionality of the ID embeddings without additional training data. A side-effect
63

of composition distillation is that, when there is spatial misalignment between the subject-single
64

and subject-composition images, the subject features will be gradually contaminated by background
65

features, reducing their authenticity. Accordingly, we propose a Elastic Face Preserving Loss (Figure
66

6), to prevent the subject features from degeneration.
67

2.1
AdaFace Architecture
68

The core module of AdaFace is the AdaFace Prompt Inverter, which converts the image-space ID
69

embeddings from a Face2Image model to the text space.
70

The architecture and initialization of the prompt inverter significantly impacts the training efficiency.
71

Compared to other deep learning tasks, the diffusion training is highly stochastic and the gradients
72

have a much lower signal-to-noise ratio. It is highly challenging to train a sizable diffusion component
73

from scratch without high compute budgets and large batch sizes. To achieve efficient learning, we
74

adopt the same architecture as the CLIP text encoder for the AdaFace Prompt Inverter, and initialize
75

it with the pre-trained weights. This ensures that the output embeddings are not very distant from the
76

text space from the beginning of training, and the model learns more signals from the gradients.
77

One may raise the question that since the output of a pre-trained CLIP encoder is in the image space,
78

why it is able to adapt quickly to generate text-space embeddings? We speculate that in CLIP, the
79

semantics of low-level layers and high-level layers are not in totally incompatible spaces, but rather,
80

3


---Page Break---
Input
Custom U-Net

AdaFace Prompt 

Inverter

Face2Image 

Encoder

Original CLIP 
Prompt Encoder

Face 
Distillation 

Loss

Original UNet

𝑤1
𝑤𝑛
⋯
Subject-ID only prompt

Face2Vec

Figure 4: Face distillation on face images. The output of the AdaFace stream is compared with the
Face2Image stream. During this process, only the AdaFace Prompt Inverter is optimized.

the high-level semantics enrich the low-level ones. Our hypothesis is corroborated by [Toker et al.,
81

2024], as well as the community practice of ad-hoc fusing the output embeddings of multiple CLIP
82

text encoder layers1. The semantics of layer features gradually transition from the text space to
83

the image space. As a result, during fine-tuning, the skip connections within CLIP will allow the
84

low-level semantics to take shortcut towards the output embeddings, and the high-level layers will
85

gradually learn to enrich the low-level semantics in the text space instead.
86

The training of the prompt inverter is divided into two stages. In the first face distillation stage, a
87

Face2Image model guides the prompt inverter to generate authentic faces in the text prompt space. In
88

the second composition distillation stage, the prompt inverter observes how the original model output
89

responds to compositional prompts, and learns to generate similar responses, so as to allow the text
90

prompts to control the composition of the generated images.
91

2.2
Face Distillation
92

The face distillation stage is illustrated in Figure 4, where the objective is to minimize the difference
93

between the generated images by the original Face2Image model and by the AdaFace Prompt Inverter
94

on the same initial noise. The training objective, namely the face distillation loss, is formulated as a
95

reconstruction loss between the two generated images:
96

Lface = Ef∼F,z∼N(0,I),t∈[1,T ]
h
∥GAdaFace(f, z, t|θ) −GFace2Image(f, z, t|θ′)∥2
2
i
,
(1)

where GFace2Image and GAdaFace are the Face2Image and the AdaFace Prompt Inverter conditioned
97

U-Nets, respectively, f is a random face drawn from the face space F, z is the initial noise, and θ and
98

θ′ are the parameters of the AdaFace Prompt Inverter and the Face2Image model, respectively. For
99

some models such as Ada2Face, θ′ ̸= θ.
100

In order to sweep the input space {f, z, t} as completely as possible, we adopt a few techniques:
101

Random Gaussian Face Embeddings.
Empirically, we observe that almost all random face
102

embeddings result in legitimate face images when processed by the Face2Image model. Therefore,
103

we expand the candidate face space F by including random face embeddings drawn from a Gaussian
104

distribution, alongside the face embeddings extracted from real face images: F = Freal ∪Frand.
105

Multi-Timestep Distillation.
We use multiple denoising steps on the same initial noise, and
106

compute the reconstruction loss on all the steps, so that the prompt inverter learns to imitate the
107

Face2Image model’s behavior on intermediate noise levels:
108

Lface = Ef∼F,z1∼N(0,I),t1>···>tk∈[1,T ]

k
X

i=1

h
∥GAdaFace(f, zi, ti|θ) −GFace2Image(f, zi, ti|θ′)∥2
2
i
,
(2)

1https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5674

4


---Page Break---
Input

AdaFace Prompt 

Inverter

Composition 

Distillation 

Loss

𝑤1
𝑤𝑛
⋯

Face2Image 
components

a photo of

𝑤1
𝑤𝑛
⋯
a photo of
in a chef outfit

𝑃(subj, compos)

𝑃(subj)

a photo of

a photo of
in a chef outfit

𝑃(class, compos)

𝑃(class)
boy

boy

CLIP + UNet

𝛥(subj)

𝛥(class)

Feature maps 
from 16 cross-
attention layers

Figure 5: Composition distillation on four types of prompts: subject-single, subject-composition,
class-single and class-composition. The four generated images form two contrastive pairs, and their
feature deltas are encouraged to be aligned through a composition distillation loss.

where t1, · · · , tk are a randomly sampled sequence of timesteps, and when i > 1, zi is the partially
109

denoised image by GFace2Image in the previous step.
110

Dynamic Model Expansion.
When the training loss plateaus, it suggests that the model has reached
111

the limits of its capacity to capture nuanced facial features. In this situation, we expand the model
112

capacity by incorporating additional query and value projections within the attention layers of the
113

prompt inverter. As a result, each token is represented by multiple, subtly distinct query and value
114

tokens. This enables the model to better grasp the subtle facial features of the subject, thanks to the
115

increased diversity and richness of the queries and values. Note that the number of keys and output
116

tokens remain unchanged, ensuring that the computational load does not increase drastically.
117

Specifically, when a query projection Q is expanded by N times, we make N identical copies of Q
118

and add Gaussian noises to N −1 of them. The same operation is applied to the value projection V .
119

This is to ensure that the expanded Q′ and V ′ do not deviate too much from the original Q and V ,
120

and the model augments the original features with slightly varied replicas.
121

The attention expansion proves to be particularly beneficial at the lower layers of the prompt inverter.
122

Intuitively, once some information in the features from the upstream Face2Image encoder is lost in
123

the lower layers, it is hard to recover in the higher layers. The mechanism of expanding queries and
124

values creates multiple, slightly varied replicas of the same information, thereby allowing the model
125

to select the most informative copy for preservation and further processing in subsequent layers.
126

This approach is conceptually akin to the role of the excitation operator in a squeeze-and-excitation
127

network [Hu et al., 2018], which also emphasizes selectively retaining the most significant features.
128

2.3
Composition Distillation
129

A prevalent issue with existing face encoders is that the subject token tends to dominate the generated
130

images, resulting in degeneration of compositionality. To mitigate this issue, we employ composi-
131

tion distillation (Figure 5) to regularize the subject embeddings, ensuring that their semantics are
132

effectively integrated with other tokens, enhancing the overall expression. During this process, the
133

model observes how the original diffusion model adjusts output features to incorporate additional
134

compositional prompts into the output image. The model then imitates these adjustments when
135

encountering similar compositional prompts.
136

5


---Page Break---
For this purpose, four types of prompts are employed to form two contrastive pairs: 1) a “subject-
137

single” prompt that only contains the subject, such as “A photo of a [Zendaya]”, 2) a “subject-
138

composition” prompt such as “A photo of a [Zendaya] in the forest”, 3) a “class-single” prompt that
139

only contains a general class, such as “A photo of a woman”, and 4) a “class-composition” prompt
140

such as “A photo of a woman in the forest”. Ideally, the semantic differences between “A photo of x”
141

and “A photo of x in the forest” should only be relevant to “in the forest”, and is independent of x.
142

We represent the semantic differences between two pairs of prompts as their “feature deltas”. The train-
143

ing objective is to encourage the feature deltas between the subject-single and subject-composition
144

images to be aligned with the feature deltas between the class-single and class-composition images.
145

In other words, the following equation is expected to hold approximately:
146

∆(subject, compos) .= feat(subject, compos) −feat(subject)

≈∆(class, compos)
.= feat(class, compos)
−feat(class),
(3)

where subject, class, (subject, compos) and (class, compos) denote the four types of prompts, re-
147

spectively. (subject, compos) and (class, compos) are randomly drawn from a pool of common
148

compositional prompts consisting of various backgrounds, additional objects, dresses, image styles
149

and lighting conditions. feat(x) refers to relevant features, including 1) the output features from all
150

the cross-attention layers, 2) the attention maps in all the cross-attention layers, and 3) the encoded
151

prompt embeddings by CLIP text encoder. feat(x) −feat(y) is the orthogonal subtraction between
152

two feature maps, defined below.
153

We define a compositional delta loss that aligns the feature deltas ∆i(subject, compos) and
154

∆i(class, compos) on the three types of features listed above:
155

L∆=
X

i
{1 −Ecompos∼U(C) cos(∆i(subject, compos), ∆i(class, compos))},
(4)

in which i indexes the feature type (cross-attention output features, attention maps or CLIP prompt
156

embeddings), and U(C) is a uniform distribution on a set of compositional prompts C.
157

Orthogonal Subtraction.
We wish to remove subject-specific features through the feature sub-
158

traction “feat(subject, compos) −feat(subject)”. However, it is commonly observed that the subject-
159

specific features may have different magnitudes (often smaller under compositional prompts). To
160

mitigate this issue, we propose to use orthogonal subtraction, which is invariant to the scale of
161

the subject-specific features. A relevant idea [Wang et al., 2023] is explored for language model
162

fine-tuning. Specifically, the feature deltas are calculated using the following equation:
163

∆feat(s, c) = feat(s, c) −projfeat(s)(feat(s, c)),
(5)

where projfeat(s)(feat(s, c)) is the projection of feat(s, c) onto feat(s), computed as:
164

projfeat(s)(feat(s, c)) = ⟨feat(s, c), feat(s)⟩feat(s),
(6)

with ⟨feat(s, c), feat(s)⟩being the inner product between the two features. The operation effectively
165

projects feat(s, c) onto the orthogonal complement of feat(s) and then subtracts this projection from
166

feat(s, c). As a result, ∆feat(s, c), the feature delta, is orthogonal to feat(s). This methodology
167

ensures that the deltas remove as much of the subject-specific features as possible, thereby minimizing
168

the influence of the scales of the subject-specific features contained within feat(s, c).
169

Differences with Previous Methods.
While previous methods have explored analogous concepts,
170

such as StyleGAN-NADA [Gal et al., 2022b], which applies similar regularizations in the CLIP
171

prompt embedding space, and PuLID [Guo et al., 2024b], which introduces similar contrastive
172

regularizations on cross-attention queries, our approach is more comprehensive and effective. Our
173

compositional delta loss encompasses a broader range of relevant features, including the attention
174

maps and output features from cross-attention layers, and the CLIP prompt embeddings. Moreover,
175

we introduce an orthogonal subtraction technique for computing the feature deltas. This technique
176

isolates and extracts composition-specific features, making the distillation more effective.
177

2.4
Elastic Face Preserving Loss
178

The composition distillation is done on instances with different prompts starting from the same initial
179

noise. This is to encourage the diffusion model to generate images that are compositionally similar
180

6


---Page Break---
Figure 6: To prevent subject features from degeneration due to spatial misalignment during composi-
tion distillation, we propose a Elastic Face Preserving Loss. The second row shows the cross-attention
maps at selected four points on the subject-single image. The highlighted pixels associate the corre-
sponding facial areas across the two images. The features of matching pixels are required to be close
to each other to achieve subject feature preservation.

[Zhang et al., 2024], to achieve more accurate alignment between the image pairs. Despite this effort,
181

spatial misalignment often persists between the images differently prompted. This misalignment
182

can result in delta loss providing erroneous signals from non-facial to facial areas, slowly reducing
183

the authenticity of the generated subjects. For instance, on a noisy input face image, the output
184

image from the subject-single instance is expected to largely retain the same facial contours as the
185

input. However, the output from the subject-composition instance often deviate from the original
186

face contours, due to the introduction of additional compositional elements. An illustrative example
187

provided in the first row of Figure 6 shows how a chef hat in one image spatially aligns with the hair
188

in another, leading to potential contamination in the subject’s hair representations.
189

To tackle this challenge, we view the subject-composition image as a “warped” version of the subject-
190

single image, and turn to techniques from the Optical Flow literature[Teed and Deng, 2020, Sui et al.,
191

2022] to estimate a matching field. The matching field is used to spatially align the subject features
192

across different images, ensuring them to be consistently maintained after “warping”.
193

Specifically, the model takes as input a noisy face image from the training data. The face image is
194

accompanied by a segmentation mask, isolating the face area for matching. We compute the cross
195

attention matrix2 between the queries of a subject-single instance and a subject-composition instance:
196

CA(subj, compos) = softmax(QsubjQT
compos),
(7)

By looking up the cross-attention map CA(subj, compos), we can find the pixels best matching a
197

subject-single image pixel in a subject-composition image. The second row in Figure 6 shows the
198

attention maps of four points on the face in the left image. We “soft-warp” the subject-composition
199

features to align with the subject-single features through matrix multiplication, and require the warped
200

features to be close to the facial features in the subject-single image:
201

Lface-preserving = 1 −cos

CA(subj, compos) ⊙feat(compos), feat(subj)


mask.
(8)

Here for clarity, feat(subject, compos) is abbreviated as feat(compos). The cosine similarity cos(·, ·)
202

is computed on the masked area. The face-preserving loss is computed on each U-Net cross-attention
203

layer. It encourages the subject features in the subject-composition instance to be consistent with
204

those in the subject-single instance, preventing them from being contaminated in the composition
205

distillation process.
206

2The inner product is not scaled to make the matching scores more polarized.

7


---Page Break---
A woman with a 

mountain in the 

background

A man in a police 

outfit

A man wearing a 

black top hat

A woman 
wearing a santa 

hat 

A woman in a 

chef outfit

InstantID        ConsistentID        PuLID            AdaFace
Input

Figure 7: Qualitative comparison of AdaFace with state-of-the-art face encoders. AdaFace generates
images that maintain the highest authenticity of the subjects, while still follow the target prompts.

3
Experiments
207

3.1
Dataset and Training Details
208

We trained AdaFace on a combination of two face datasets: Flickr-Faces-HQ (FFHQ) [Karras et al.,
209

2019], which comprises 70,000 images, and VGGFace2-HQ [Cao et al., 2018], which comprises
210

336,567 images after filtering. Face masks were generated using the BiSeNet face segmentation
211

model [Yu et al., 2018]. The distilled Face2Image model is Ada2Face [Papantoniou et al., 2024], as it
212

is able to generate authentic and diverse face images. The training employed the Prodigy optimizer
213

[Mishchenko and Defazio, 2024] with d_coef=2 (akin to the learning rate in other optimizers) during
214

face distillation, and d_coef=0.5 during composition distillation. Batch sizes were set to 4 and 3 for
215

the two stages, respectively, with a gradient accumulation of 2. The model was trained with 240,000
216

iterations in the face distillation stage and 120,000 iterations in the composition distillation stage.
217

During face distillation, the loss reached a plateau twice, resulting in two dynamic expansions of the
218

model capacity. Eventually, the attention layers in the trained prompt inverter were expanded with
219

multipliers of (8x, 8x, 8x, 4x, 4x, ..., 4x) relative to the original CLIP text encoder. This resulted in
220

a total of 2M parameters, in contrast to the 1.2M parameters of the original model.
221

In addition, we collected the images of 23 celebrities, each with 9 10 images, as the evaluated subjects.
222

These celebrities include actors, singers and internet celebrities on Instagram. This dataset will be
223

released along with the code.
224

3.2
Qualitative Comparisons
225

We compared AdaFace with a few state-of-the-art face encoders, including InstantID [Wang et al.,
226

2024], ConsistentID [Huang et al., 2024] and PuLID [Guo et al., 2024b]. The input were images
227

from our celebrity-23 dataset.
228

The results presented in Figure 7 demonstrate that AdaFace produces images that not only exhibit
229

high authenticity of the subjects but also show good consistency with the text prompts. In comparison,
230

other models often fall short in generating images that are either less authentic or less compositional.
231

For instance, InstantID tends to produce overly stylized images with significant variability in au-
232

thenticity across different subjects. PuLID, while generating aesthetically pleasing images, achieves
233

slightly lower authenticity levels compared to AdaFace. Despite also utilizing a text-space approach,
234

8


---Page Break---
Jensen Huang dancing pose among folks in a 

park, waving hands
Yann Lecun in a white apron and chef hat, 

garnishing a gourmet dish

Figure 8: Comparison of AdaFace with ID-Animator on personalized video generation. AdaFace
generates videos with higher authenticity and compositionality.

ConsistentID has the least compositional output among the models evaluated, largely due to the
235

absence of compositional training in its ID embeddings.
236

In addition, we plugged AdaFace into AnimateDiff, and generated personalized videos of celebrities
237

under various compositional prompts. The results are shown in Figure 1. Figure 8 compares with a
238

recent method ID-Animator [He et al., 2024]. AdaFace generated videos with high authenticity and
239

compositionality, while ID-Animator usually produces videos with less authentic subjects.
240

3.3
Quantitative Evaluations
241

To assess the performance of AdaFace quantitatively, we evaluated a few baseline methods and
242

AdaFace, on the “celebrity-23" images and DreamBench compositional prompts, comparing AdaFace
243

with two baseline methods PuLID and InstantID. First, we measured the face similarity using the
244

cosine similarity between the ArcFace embedding of the generated images and reference images. In
245

addition, the CLIP-Text (CLIP-T) metric determines the consistency of the generated images with the
246

prompts. The DINO and CLIP-I metrics are less indicative and are only for reference. The results,
247

detailed in Table 1, show that AdaFace achieved comparable face similarity and prompt consistency
248

scores to PuLID, and slightly outperformed InstantID. Note that the results of AdaFace is achieved
249

on the original Stable Diffusion 1.5 model weight, which usually leads to much lower composition
250

scores than other fine-tuned SD 1.5 model weights, such as RealisticVision.
251

ArcFace (subj) CLIP-T (comp) DINO CLIP-I
DB
0.349
0.324
0.470 0.656
TI
0.326
0.250
0.508 0.675
PuLID
0.468
0.280
0.512 0.630
InstantID
0.455
0.257
0.472 0.595
Ada
0.476
0.270
0.544 0.670
-Comp
0.505
0.235
0.598 0.685
Table 1: Quantitative evaluation on the “celebrity-23" images and DreamBench compositional
prompts. -Comp is the model trained only with the face distillation stage.

As an ablation study, we list the performance of the AdaFace model without composition distillation.
252

It can be seen that the face authenticity is slightly reduced after composition distillation, however, the
253

generated images become much more consistent with the prompts.
254

4
Conclusions and Discussions
255

In this work, we present AdaFace, a versatile face encoder that maps human faces into the text
256

prompt space. AdaFace is trained with a low compute budget and achieves high authenticity and
257

compositionality in zero-shot generation of subject images. We demonstrate the effectiveness of
258

AdaFace by showcasing the generated images and videos of celebrities under various compositional
259

prompts. Additionally, our quantitative evaluations further underscore its performance.
260

A notable limitation of AdaFace is that the authenticity of the output face embeddings are constrained
261

by the Face2Image model it distills from. However, this limitation can be addressed by distilling on
262

more powerful Face2Image models and expanding the model capacity. For future work, we would
263

extend the AdaFace method to object images. For instance, applying AdaFace distillation techniques
264

to IP-Adapter [Ye et al., 2023] could enable the generation of both human and object images.
265

9


---Page Break---
References
266

References
267

Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman. VGGFace2: a dataset for recognising
268

faces across pose and age. In 2018 13th IEEE international conference on automatic face &amp;
269

gesture recognition (FG 2018), pages 67–74, Los Alamitos, CA, USA, May 2018. IEEE Computer
270

Society. doi: 10.1109/FG.2018.00020. URL https://doi.ieeecomputersociety.org/10.
271

1109/FG.2018.00020.
272

J. Chen, J. YU, C. GE, L. Yao, E. Xie, Z. Wang, J. Kwok, P. Luo, H. Lu, and Z. Li. PixArt-$\alpha$:
273

Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The twelfth
274

international conference on learning representations, 2024a. URL https://openreview.net/
275

forum?id=eAKmQPe3m1.
276

W. Chen, J. Zhang, J. Wu, H. Wu, X. Xiao, and L. Lin. ID-Aligner: Enhancing Identity-Preserving
277

Text-to-Image Generation with Reward Feedback Learning, Apr. 2024b. URL http://arxiv.
278

org/abs/2404.15449. arXiv:2404.15449 [cs].
279

J. Deng, J. Guo, N. Xue, and S. Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face
280

Recognition. pages 4690–4699, 2019. URL https://openaccess.thecvf.com/content_
281

CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_
282

Recognition_CVPR_2019_paper.html.
283

R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or. An Image
284

is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion, Aug. 2022a.
285

URL http://arxiv.org/abs/2208.01618. arXiv:2208.01618 [cs].
286

R. Gal, O. Patashnik, H. Maron, A. H. Bermano, G. Chechik, and D. Cohen-Or. StyleGAN-NADA:
287

CLIP-guided domain adaptation of image generators. ACM Trans. Graph., 41(4), July 2022b.
288

ISSN 0730-0301. doi: 10.1145/3528223.3530164. URL https://doi.org/10.1145/3528223.
289

3530164. Number of pages: 13 Place: New York, NY, USA Publisher: Association for Computing
290

Machinery tex.articleno: 141 tex.issue_date: July 2022.
291

Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai. AnimateDiff:
292

Animate your personalized text-to-image diffusion models without specific tuning. In The twelfth
293

international conference on learning representations, 2024a. URL https://openreview.net/
294

forum?id=Fx2SbBgcte.
295

Z. Guo, Y. Wu, Z. Chen, L. Chen, and Q. He. PuLID: Pure and Lightning ID Customization via Con-
296

trastive Alignment, Apr. 2024b. URL http://arxiv.org/abs/2404.16022. arXiv:2404.16022
297

[cs].
298

Y. Han, J. Zhu, K. He, X. Chen, Y. Ge, W. Li, X. Li, J. Zhang, C. Wang, and Y. Liu. Face Adapter
299

for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control, May 2024. URL
300

http://arxiv.org/abs/2405.12970. arXiv:2405.12970 [cs].
301

X. He, Q. Liu, S. Qian, X. Wang, T. Hu, K. Cao, K. Yan, and J. Zhang. ID-Animator: Zero-Shot
302

Identity-Preserving Human Video Generation, May 2024. URL http://arxiv.org/abs/2404.
303

15275. arXiv:2404.15275 [cs].
304

J. Ho, A. Jain, and P. Abbeel.
Denoising Diffusion Probabilistic Models.
In Ad-
305

vances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran
306

Associates, Inc., 2020.
URL https://proceedings.neurips.cc/paper/2020/hash/
307

4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.
308

J. Hu, L. Shen, and G. Sun. Squeeze-and-excitation networks. In 2018 IEEE/CVF conference on
309

computer vision and pattern recognition, pages 7132–7141, 2018. doi: 10.1109/CVPR.2018.00745.
310

J. Huang, X. Dong, W. Song, H. Li, J. Zhou, Y. Cheng, S. Liao, L. Chen, Y. Yan, S. Liao, and
311

X. Liang. ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving,
312

Apr. 2024. URL http://arxiv.org/abs/2404.16771. arXiv:2404.16771 [cs].
313

10


---Page Break---
T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial
314

networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition
315

(CVPR), June 2019.
316

B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani. Imagic: Text-
317

based real image editing with diffusion models. In Conference on computer vision and pattern
318

recognition 2023, 2023.
319

N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.-Y. Zhu. Multi-Concept Customization of Text-
320

to-Image Diffusion, Dec. 2022. URL http://arxiv.org/abs/2212.04488. arXiv:2212.04488
321

[cs].
322

K. Mishchenko and A. Defazio. Prodigy: An Expeditiously Adaptive Parameter-Free Learner, Mar.
323

2024. URL http://arxiv.org/abs/2306.06101. arXiv:2306.06101 [cs, math, stat].
324

A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen.
325

GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models,
326

Mar. 2022. URL http://arxiv.org/abs/2112.10741. arXiv:2112.10741 [cs].
327

F. P. Papantoniou, A. Lattas, S. Moschoglou, J. Deng, B. Kainz, and S. Zafeiriou. Arc2Face: A
328

Foundation Model of Human Faces, Mar. 2024. URL http://arxiv.org/abs/2403.11641.
329

arXiv:2403.11641 [cs].
330

W. Peebles and S. Xie. Scalable Diffusion Models with Transformers, Mar. 2023. URL http:
331

//arxiv.org/abs/2212.09748. arXiv:2212.09748 [cs].
332

D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach.
333

SDXL: Improving latent diffusion models for high-resolution image synthesis. In The twelfth
334

international conference on learning representations, 2024. URL https://openreview.net/
335

forum?id=di52zR8xgf.
336

R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-Resolution Image Syn-
337

thesis with Latent Diffusion Models, Apr. 2022. URL http://arxiv.org/abs/2112.10752.
338

arXiv:2112.10752 [cs].
339

N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. DreamBooth: Fine Tuning
340

Text-to-Image Diffusion Models for Subject-Driven Generation, Aug. 2022. URL http://arxiv.
341

org/abs/2208.12242. arXiv:2208.12242 [cs].
342

C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, R. Gontijo-Lopes,
343

B. K. Ayan, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi. Photorealistic Text-to-Image Diffusion
344

Models with Deep Language Understanding. Oct. 2022.
345

J. Shi, W. Xiong, Z. Lin, and H. J. Jung.
InstantBooth: Personalized Text-to-Image Genera-
346

tion without Test-Time Finetuning, Apr. 2023. URL http://arxiv.org/abs/2304.03411.
347

arXiv:2304.03411 [cs].
348

X. Sui, S. Li, X. Geng, Y. Wu, X. Xu, Y. Liu, R. Goh, and H. Zhu. CRAFT: Cross-attentional flow
349

transformer for robust optical flow. In 2022 IEEE/CVF conference on computer vision and pattern
350

recognition (CVPR), 2022.
351

Z. Teed and J. Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In A. Vedaldi,
352

H. Bischof, T. Brox, and J.-M. Frahm, editors, Computer vision – ECCV 2020, pages 402–419,
353

Cham, 2020. Springer International Publishing. ISBN 978-3-030-58536-5.
354

Y. Tewel, R. Gal, G. Chechik, and Y. Atzmon. Key-locked rank one editing for text-to-image
355

personalization. In ACM SIGGRAPH 2023 conference proceedings, Siggraph ’23, New York, NY,
356

USA, 2023. Association for Computing Machinery. ISBN 9798400701597. doi: 10.1145/3588432.
357

3591506. URL https://doi.org/10.1145/3588432.3591506. Number of pages: 11 Place: ,
358

Los Angeles, CA, USA, tex.articleno: 12.
359

M. Toker, H. Orgad, M. Ventura, D. Arad, and Y. Belinkov. Diffusion Lens: Interpreting Text
360

Encoders in Text-to-Image Pipelines, Mar. 2024. URL http://arxiv.org/abs/2403.05846.
361

arXiv:2403.05846 [cs] version: 1.
362

11


---Page Break---
Q. Wang, X. Bai, H. Wang, Z. Qin, A. Chen, H. Li, X. Tang, and Y. Hu. InstantID: Zero-shot Identity-
363

Preserving Generation in Seconds, Feb. 2024. URL http://arxiv.org/abs/2401.07519.
364

arXiv:2401.07519 [cs].
365

X. Wang, T. Chen, Q. Ge, H. Xia, R. Bao, R. Zheng, Q. Zhang, T. Gui, and X. Huang. Orthogonal
366

subspace learning for language model continual learning. In Findings of the association for
367

computational linguistics: EMNLP 2023, Singapore, Dec. 2023. Association for Computational
368

Linguistics.
369

Y. Wei, Y. Zhang, Z. Ji, J. Bai, L. Zhang, and W. Zuo. ELITE: Encoding Visual Concepts into Textual
370

Embeddings for Customized Text-to-Image Generation. In ICCV 2023. arXiv, Feb. 2023. doi:
371

10.48550/arXiv.2302.13848. URL http://arxiv.org/abs/2302.13848. arXiv:2302.13848
372

[cs].
373

H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text Compatible Image Prompt Adapter
374

for Text-to-Image Diffusion Models, Aug. 2023. URL http://arxiv.org/abs/2308.06721.
375

arXiv:2308.06721 [cs].
376

C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang. BiSeNet: Bilateral segmentation network for
377

real-time semantic segmentation. In V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, editors,
378

Computer vision – ECCV 2018, pages 334–349, Cham, 2018. Springer International Publishing.
379

ISBN 978-3-030-01261-8.
380

H. Zhang, J. Zhou, Y. Lu, M. Guo, P. Wang, L. Shen, and Q. Qu. The Emergence of Reproducibility
381

and Consistency in Diffusion Models, Feb. 2024. URL http://arxiv.org/abs/2310.05264.
382

arXiv:2310.05264 [cs].
383

12


---Page Break---
NeurIPS Paper Checklist
384

1. Claims
385

Question: Do the main claims made in the abstract and introduction accurately reflect the
386

paper’s contributions and scope?
387

Answer: [Yes]
388

Justification: The main claims in the abstract and introduction accurately reflect the paper’s
389

contributions and scope, as the detailed results, discussions, and conclusions align with and
390

support the initial claims.
391

Guidelines:
392

• The answer NA means that the abstract and introduction do not include the claims
393

made in the paper.
394

• The abstract and/or introduction should clearly state the claims made, including the
395

contributions made in the paper and important assumptions and limitations. A No or
396

NA answer to this question will not be perceived well by the reviewers.
397

• The claims made should match theoretical and experimental results, and reflect how
398

much the results can be expected to generalize to other settings.
399

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
400

are not attained by the paper.
401

2. Limitations
402

Question: Does the paper discuss the limitations of the work performed by the authors?
403

Answer: [Yes]
404

Justification: The limitations of the work are discussed in the "Conclusions and Discussion"
405

section.
406

Guidelines:
407

• The answer NA means that the paper has no limitation while the answer No means that
408

the paper has limitations, but those are not discussed in the paper.
409

• The authors are encouraged to create a separate "Limitations" section in their paper.
410

• The paper should point out any strong assumptions and how robust the results are to
411

violations of these assumptions (e.g., independence assumptions, noiseless settings,
412

model well-specification, asymptotic approximations only holding locally). The authors
413

should reflect on how these assumptions might be violated in practice and what the
414

implications would be.
415

• The authors should reflect on the scope of the claims made, e.g., if the approach was
416

only tested on a few datasets or with a few runs. In general, empirical results often
417

depend on implicit assumptions, which should be articulated.
418

• The authors should reflect on the factors that influence the performance of the approach.
419

For example, a facial recognition algorithm may perform poorly when image resolution
420

is low or images are taken in low lighting. Or a speech-to-text system might not be
421

used reliably to provide closed captions for online lectures because it fails to handle
422

technical jargon.
423

• The authors should discuss the computational efficiency of the proposed algorithms
424

and how they scale with dataset size.
425

• If applicable, the authors should discuss possible limitations of their approach to
426

address problems of privacy and fairness.
427

• While the authors might fear that complete honesty about limitations might be used by
428

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
429

limitations that aren’t acknowledged in the paper. The authors should use their best
430

judgment and recognize that individual actions in favor of transparency play an impor-
431

tant role in developing norms that preserve the integrity of the community. Reviewers
432

will be specifically instructed to not penalize honesty concerning limitations.
433

3. Theory Assumptions and Proofs
434

Question: For each theoretical result, does the paper provide the full set of assumptions and
435

a complete (and correct) proof?
436

13


---Page Break---
Answer: [Yes]
437

Justification: For each theoretical result, the paper provides a plenty of experimental support.
438

Guidelines:
439

• The answer NA means that the paper does not include theoretical results.
440

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
441

referenced.
442

• All assumptions should be clearly stated or referenced in the statement of any theorems.
443

• The proofs can either appear in the main paper or the supplemental material, but if
444

they appear in the supplemental material, the authors are encouraged to provide a short
445

proof sketch to provide intuition.
446

• Inversely, any informal proof provided in the core of the paper should be complemented
447

by formal proofs provided in appendix or supplemental material.
448

• Theorems and Lemmas that the proof relies upon should be properly referenced.
449

4. Experimental Result Reproducibility
450

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
451

perimental results of the paper to the extent that it affects the main claims and/or conclusions
452

of the paper (regardless of whether the code and data are provided or not)?
453

Answer: [Yes]
454

Justification: All the experimental details are clearly stated in the paper and the code will be
455

made publicly available.
456

Guidelines:
457

• The answer NA means that the paper does not include experiments.
458

• If the paper includes experiments, a No answer to this question will not be perceived
459

well by the reviewers: Making the paper reproducible is important, regardless of
460

whether the code and data are provided or not.
461

• If the contribution is a dataset and/or model, the authors should describe the steps taken
462

to make their results reproducible or verifiable.
463

• Depending on the contribution, reproducibility can be accomplished in various ways.
464

For example, if the contribution is a novel architecture, describing the architecture fully
465

might suffice, or if the contribution is a specific model and empirical evaluation, it may
466

be necessary to either make it possible for others to replicate the model with the same
467

dataset, or provide access to the model. In general. releasing code and data is often
468

one good way to accomplish this, but reproducibility can also be provided via detailed
469

instructions for how to replicate the results, access to a hosted model (e.g., in the case
470

of a large language model), releasing of a model checkpoint, or other means that are
471

appropriate to the research performed.
472

• While NeurIPS does not require releasing code, the conference does require all submis-
473

sions to provide some reasonable avenue for reproducibility, which may depend on the
474

nature of the contribution. For example
475

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
476

to reproduce that algorithm.
477

(b) If the contribution is primarily a new model architecture, the paper should describe
478

the architecture clearly and fully.
479

(c) If the contribution is a new model (e.g., a large language model), then there should
480

either be a way to access this model for reproducing the results or a way to reproduce
481

the model (e.g., with an open-source dataset or instructions for how to construct
482

the dataset).
483

(d) We recognize that reproducibility may be tricky in some cases, in which case
484

authors are welcome to describe the particular way they provide for reproducibility.
485

In the case of closed-source models, it may be that access to the model is limited in
486

some way (e.g., to registered users), but it should be possible for other researchers
487

to have some path to reproducing or verifying the results.
488

5. Open access to data and code
489

14


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
490

tions to faithfully reproduce the main experimental results, as described in supplemental
491

material?
492

Answer: [Yes]
493

Justification: All data and code will be made publicly available.
494

Guidelines:
495

• The answer NA means that paper does not include experiments requiring code.
496

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
497

public/guides/CodeSubmissionPolicy) for more details.
498

• While we encourage the release of code and data, we understand that this might not be
499

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
500

including code, unless this is central to the contribution (e.g., for a new open-source
501

benchmark).
502

• The instructions should contain the exact command and environment needed to run to
503

reproduce the results. See the NeurIPS code and data submission guidelines (https:
504

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
505

• The authors should provide instructions on data access and preparation, including how
506

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
507

• The authors should provide scripts to reproduce all experimental results for the new
508

proposed method and baselines. If only a subset of experiments are reproducible, they
509

should state which ones are omitted from the script and why.
510

• At submission time, to preserve anonymity, the authors should release anonymized
511

versions (if applicable).
512

• Providing as much information as possible in supplemental material (appended to the
513

paper) is recommended, but including URLs to data and code is permitted.
514

6. Experimental Setting/Details
515

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
516

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
517

results?
518

Answer: [Yes]
519

Justification: Please refer to the "Implementation Detail" section in the main paper.
520

Guidelines:
521

• The answer NA means that the paper does not include experiments.
522

• The experimental setting should be presented in the core of the paper to a level of detail
523

that is necessary to appreciate the results and make sense of them.
524

• The full details can be provided either with the code, in appendix, or as supplemental
525

material.
526

7. Experiment Statistical Significance
527

Question: Does the paper report error bars suitably and correctly defined or other appropriate
528

information about the statistical significance of the experiments?
529

Answer: [No]
530

Justification: We evaluated on a diverse set of 30 celebrities, each with around 50 prompts,
531

which is sufficient to reflect the model’s performance.
532

Guidelines:
533

• The answer NA means that the paper does not include experiments.
534

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
535

dence intervals, or statistical significance tests, at least for the experiments that support
536

the main claims of the paper.
537

• The factors of variability that the error bars are capturing should be clearly stated (for
538

example, train/test split, initialization, random drawing of some parameter, or overall
539

run with given experimental conditions).
540

15


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
541

call to a library function, bootstrap, etc.)
542

• The assumptions made should be given (e.g., Normally distributed errors).
543

• It should be clear whether the error bar is the standard deviation or the standard error
544

of the mean.
545

• It is OK to report 1-sigma error bars, but one should state it. The authors should
546

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
547

of Normality of errors is not verified.
548

• For asymmetric distributions, the authors should be careful not to show in tables or
549

figures symmetric error bars that would yield results that are out of range (e.g. negative
550

error rates).
551

• If error bars are reported in tables or plots, The authors should explain in the text how
552

they were calculated and reference the corresponding figures or tables in the text.
553

8. Experiments Compute Resources
554

Question: For each experiment, does the paper provide sufficient information on the com-
555

puter resources (type of compute workers, memory, time of execution) needed to reproduce
556

the experiments?
557

Answer: [Yes]
558

Justification: We use 2 A6000 GPUs, each with 48G of memory.
559

Guidelines:
560

• The answer NA means that the paper does not include experiments.
561

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
562

or cloud provider, including relevant memory and storage.
563

• The paper should provide the amount of compute required for each of the individual
564

experimental runs as well as estimate the total compute.
565

• The paper should disclose whether the full research project required more compute
566

than the experiments reported in the paper (e.g., preliminary or failed experiments that
567

didn’t make it into the paper).
568

9. Code Of Ethics
569

Question: Does the research conducted in the paper conform, in every respect, with the
570

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
571

Answer: [Yes]
572

Justification: The research conducted in the paper conforms in every respect with the
573

NeurIPS Code of Ethics.
574

Guidelines:
575

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
576

• If the authors answer No, they should explain the special circumstances that require a
577

deviation from the Code of Ethics.
578

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
579

eration due to laws or regulations in their jurisdiction).
580

10. Broader Impacts
581

Question: Does the paper discuss both potential positive societal impacts and negative
582

societal impacts of the work performed?
583

Answer: [Yes]
584

Justification: We discussed the positive impacts, including its potential use in entertainment
585

and art, video games and virtual reality. Additionally, its potential use for educational
586

purposes in historical recreation, such as recreating faces of historical figures or enhancing
587

documentaries, bringing history to life. We also pointed out potential negative impacts,
588

including privacy violations. There is a risk of creating and using images of individuals with-
589

out their consent. Moreover, misinformation and deepfakes are among the most concerning
590

impacts, with the creation of deepfake videos that could be used to spread misinformation
591

and manipulate public opinion. We also highlighted security concerns, as the technology
592

16


---Page Break---
could be used to bypass facial recognition systems for fraudulent purposes, posing significant
593

security challenges. The authors will join in the effort for possible mitigation by providing
594

gated release of models.
595

Guidelines:
596

• The answer NA means that there is no societal impact of the work performed.
597

• If the authors answer NA or No, they should explain why their work has no societal
598

impact or why the paper does not address societal impact.
599

• Examples of negative societal impacts include potential malicious or unintended uses
600

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
601

(e.g., deployment of technologies that could make decisions that unfairly impact specific
602

groups), privacy considerations, and security considerations.
603

• The conference expects that many papers will be foundational research and not tied
604

to particular applications, let alone deployments. However, if there is a direct path to
605

any negative applications, the authors should point it out. For example, it is legitimate
606

to point out that an improvement in the quality of generative models could be used to
607

generate deepfakes for disinformation. On the other hand, it is not needed to point out
608

that a generic algorithm for optimizing neural networks could enable people to train
609

models that generate Deepfakes faster.
610

• The authors should consider possible harms that could arise when the technology is
611

being used as intended and functioning correctly, harms that could arise when the
612

technology is being used as intended but gives incorrect results, and harms following
613

from (intentional or unintentional) misuse of the technology.
614

• If there are negative societal impacts, the authors could also discuss possible mitigation
615

strategies (e.g., gated release of models, providing defenses in addition to attacks,
616

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
617

feedback over time, improving the efficiency and accessibility of ML).
618

11. Safeguards
619

Question: Does the paper describe safeguards that have been put in place for responsible
620

release of data or models that have a high risk for misuse (e.g., pretrained language models,
621

image generators, or scraped datasets)?
622

Answer: [Yes]
623

Justification: The work describes basic safeguards implemented for the responsible release
624

of models, particularly focusing on preventing misuse. We have incorporated filters that
625

specifically exclude NSFW (Not Safe for Work) keywords in the generation prompts, such
626

as ’nude,’ ’naked,’ ’nsfw,’ ’topless,’ and ’bare breasts.’ This approach helps mitigate the risk
627

of generating inappropriate or sensitive content."
628

Guidelines:
629

• The answer NA means that the paper poses no such risks.
630

• Released models that have a high risk for misuse or dual-use should be released with
631

necessary safeguards to allow for controlled use of the model, for example by requiring
632

that users adhere to usage guidelines or restrictions to access the model or implementing
633

safety filters.
634

• Datasets that have been scraped from the Internet could pose safety risks. The authors
635

should describe how they avoided releasing unsafe images.
636

• We recognize that providing effective safeguards is challenging, and many papers do
637

not require this, but we encourage authors to take this into account and make a best
638

faith effort.
639

12. Licenses for existing assets
640

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
641

the paper, properly credited and are the license and terms of use explicitly mentioned and
642

properly respected?
643

Answer: [Yes]
644

17


---Page Break---
Justification: In our paper, we have ensured proper attribution for all assets used, such as
645

code, data, and models, by citing the related papers and sources from which these assets
646

were derived. Additionally, we have adhered to the licensing terms and conditions of each
647

asset, as detailed in the respective citations.
648

Guidelines:
649

• The answer NA means that the paper does not use existing assets.
650

• The authors should cite the original paper that produced the code package or dataset.
651

• The authors should state which version of the asset is used and, if possible, include a
652

URL.
653

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
654

• For scraped data from a particular source (e.g., website), the copyright and terms of
655

service of that source should be provided.
656

• If assets are released, the license, copyright information, and terms of use in the
657

package should be provided. For popular datasets, paperswithcode.com/datasets
658

has curated licenses for some datasets. Their licensing guide can help determine the
659

license of a dataset.
660

• For existing datasets that are re-packaged, both the original license and the license of
661

the derived asset (if it has changed) should be provided.
662

• If this information is not available online, the authors are encouraged to reach out to
663

the asset’s creators.
664

13. New Assets
665

Question: Are new assets introduced in the paper well documented and is the documentation
666

provided alongside the assets?
667

Answer: [Yes]
668

Justification: New assets introduced in the paper are well documented. The code is accompa-
669

nied by usage documentation and is embedded with detailed comments to ensure clarity and
670

ease of use for future researchers. Additionally, videos are provided alongside a description
671

of the files and a list of prompts used for their generation, which enhances transparency and
672

replicability of the results.
673

Guidelines:
674

• The answer NA means that the paper does not release new assets.
675

• Researchers should communicate the details of the dataset/code/model as part of their
676

submissions via structured templates. This includes details about training, license,
677

limitations, etc.
678

• The paper should discuss whether and how consent was obtained from people whose
679

asset is used.
680

• At submission time, remember to anonymize your assets (if applicable). You can either
681

create an anonymized URL or include an anonymized zip file.
682

14. Crowdsourcing and Research with Human Subjects
683

Question: For crowdsourcing experiments and research with human subjects, does the paper
684

include the full text of instructions given to participants and screenshots, if applicable, as
685

well as details about compensation (if any)?
686

Answer: [NA]
687

Justification: The paper does not involve crowdsourcing nor research with human subjects.
688

Guidelines:
689

• The answer NA means that the paper does not involve crowdsourcing nor research with
690

human subjects.
691

• Including this information in the supplemental material is fine, but if the main contribu-
692

tion of the paper involves human subjects, then as much detail as possible should be
693

included in the main paper.
694

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
695

or other labor should be paid at least the minimum wage in the country of the data
696

collector.
697

18


---Page Break---
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
698

Subjects
699

Question: Does the paper describe potential risks incurred by study participants, whether
700

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
701

approvals (or an equivalent approval/review based on the requirements of your country or
702

institution) were obtained?
703

Answer: [NA]
704

Justification: The paper does not involve crowdsourcing nor research with human subjects.
705

Guidelines:
706

• The answer NA means that the paper does not involve crowdsourcing nor research with
707

human subjects.
708

• Depending on the country in which research is conducted, IRB approval (or equivalent)
709

may be required for any human subjects research. If you obtained IRB approval, you
710

should clearly state this in the paper.
711

• We recognize that the procedures for this may vary significantly between institutions
712

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
713

guidelines for their institution.
714

• For initial submissions, do not include any information that would break anonymity (if
715

applicable), such as the institution conducting the review.
716

19


---Page Break---
