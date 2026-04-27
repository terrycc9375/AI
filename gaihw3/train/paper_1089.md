Manipulation Intention Understanding for Accurate
Zero-Shot Composed Image Retrieval

Anonymous Author(s)
Affiliation
Address
email

Abstract

Composed Image Retrieval (CIR) facilitates retrieving an image matching a refer-
1

ence image while incorporating specified textual modifications, which is crucial
2

for internet searches and e-commerce. Traditional supervised CIR methods rely
3

on annotated triplets, which are labor-intensive and limit generalizability. Recent
4

advances in Zero-Shot Composed Image Retrieval (ZS-CIR) address the challenge
5

of performing this task without annotated triplets. A key challenge in ZS-CIR
6

is training models on limited intention-relevant datasets to understand human
7

intention implicitly expressed in textual modifications for accurately retrieving
8

target images. In this paper, we introduce an image-text dataset incorporated
9

with pseudo-manipulation intentions to enhance the training of ZS-CIR models
10

in understanding human manipulation intents. Based on our dataset, we propose
11

a novel framework, De-MINDS, for capturing the intent humans aim to modify,
12

thereby enhancing the ZS-CIR model’s ability to understand human manipulation
13

descriptions. Specifically, a simple mapping network first maps image information
14

into language space and forms a target description with a manipulation descrip-
15

tion. Subsequently, De-MINDS captures intention-relevant information from tar-
16

get descriptions and converts them into several pseudo-word tokens for accurate
17

ZS-CIR. The De-MINDS model exhibits robust generalization and significant
18

improvements in performance across four ZS-CIR tasks. It achieves performance
19

improvements from 2.05% to 4.35% over the best methods and establishes new
20

state-of-the-art results with comparable inference times. Our code is available at
21

https://anonymous.4open.science/r/De-MINDS/.
22

1
Introduction
23

Composed Image Retrieval (CIR) [55] aims to retrieve an image that is visually similar to a reference
24

image while having visual modification according to the manipulation text. Different from traditional
25

image retrieval [15], CIR offers more flexibility and accuracy by enabling users to integrate both
26

visual and textual information into their search intent. This approach has gained emerging attention
27

in internet searches and e-commerce applications [12, 45]. Various supervised methods have been
28

proposed to solve CIR problem [12, 33, 19, 4], which requires a large amount of annotated triplets,
29

i.e., a reference image, a manipulated description, and a target image, for training task-specific
30

retrieval models. However, these supervised methods are labor-intensive for data annotation and tend
31

to suffer from limited generalization capabilities due to bias in human annotation. To enhance model
32

generalization and perform CIR tasks without annotated triplets, recent research [45, 3, 52, 25, 20]
33

introduce Zero-Shot Composed Image Retrieval (ZS-CIR). Existing solutions for ZS-CIR map an
34

image to the language space, combining it with text to form a query. This query retrieves target
35

images from the shared semantic space of a pre-trained vision-language model by calculating semantic
36

similarity. These methods typically involve a pre-trained mapping network that converts the reference
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
image into a pseudo-word token S∗. During retrieval, this token S∗is merged with the manipulation
38

description to construct a target description, which a pre-trained CLIP model [41] then encodes,
39

leveraging its comprehensive pre-trained knowledge across image candidates for retrieval.
40

Despite remarkable advancement, the pre-trained mapping networks are not satisfactory for CIR due
41

to the following reasons:
42

(1) There exists a discrepancy between the retrieval and pre-training stages in ZS-CIR models. During
43

retrieval, the mapping network is tasked with aligning intent-specific visual information (e.g., objects,
44

scenes, colors, and styles) in language space to form a composed image description query (e.g.,
45

change to a man playing the accordion joyfully in the street) for calculating semantic similarity with
46

the target image. However, in the pre-training phase, the mapping network aligns general visual
47

information with textual descriptions of the image content (e.g., a musician plays the piano). Without
48

intent-specific mapping, the pseudo-token S∗contains heavy information redundancy involving most
49

objects, background/foreground, color, and style, leading to inaccurate retrieval.
50

(2) Accurately understanding the intention a user intends to modify in manipulation descriptions
51

presents substantial challenges. These intentions are implicitly expressed in users’ manipulation
52

descriptions. For instance, the manipulation intention embedded in the request to “make this photo
53

feel like early fall” may involve changing colors (e.g., orange and yellow), adjusting the scene (e.g.,
54

fallen leaves), and adding specific objects (e.g., autumnal trees). However, existing ZS-CIR models
55

rely on the CLIP language encoder, which challenges capturing fine-grained/long information from
56

text [51, 58], facing difficulties in accurately understanding these manipulation intentions.
57

In this work, we introduce the intent-CC3M, an intention-based dataset for training mapping net-
58

works capable of aligning intention-relevant visual information within the language space, thus
59

addressing the gap between pre-training and retrieval in ZS-CIR models. We incorporate pseudo-
60

manipulation descriptions in CC3M [47], the widely used ZS-CIR training dataset [45, 52]. These
61

pseudo descriptions, reflecting potential user intention to manipulate images, are reasoned through
62

chain-of-thought prompting using an off-the-shelf Multi-modal Large Language Model (MLLM),
63

facilitating the learning of intent-specific mapping capabilities. Furthermore, to overcome the chal-
64

lenge of existing ZS-CIR models in understanding manipulation intention within descriptions, we
65

propose a novel unDErstanding of Manipulation INtention from target Description before Searching
66

approach, named De-MINDS. We leverage pseudo-manipulation descriptions to train De-MINDS
67

to capture manipulation intention from various aspects (e.g., objects, scenes, colors, styles) guided
68

by multiple learnable queries. This intention information is mapped to several pseudo-word tokens,
69

which are subsequently input into the CLIP language encoder, enhancing its ability to understand
70

users’ intention to modify and thereby improving the accuracy of CIR.
71

The main contributions of this work are summarized as follows: (1) We introduce intent-CC3M, a
72

novel dataset with pseudo-manipulation descriptions reasoned by an MLLM to bridge the gap between
73

pre-training and retrieval in ZS-CIR models. Our experiments demonstrate that baseline models
74

trained with our dataset are capable of aligning intention-relevant visual information, achieving
75

consistent performance improvements. (2) We propose a novel manipulation intention understanding
76

network. We extract intentions in manipulation descriptions under the guidance of learnable queries
77

and map to several pseudo-word tokens for retrieval, enhancing the CLIP’s ability to understand users’
78

intentions. It sheds new light on intention-based image retrieval. (3) Our De-MINDS are consistently
79

effective and generalizable across diverse ZS-CIR tasks. It significantly improves CIR performance
80

from 2.05% to 4.35% across four CIR tasks, establishing new state-of-the-art results with comparable
81

inference time, further impacting vision and language applications.
82

2
Related Works
83

Composed Image Retrieval. Composed Image Retrieval (CIR) integrates image and text for retrieval
84

[54]. Current models typically employ late fusion for integrating visual and language features
85

separately [4, 33, 4]. In contrast, zero-shot CIR models like Pic2Word [45], SEARLE [3], and
86

Context-I2W [52] train on image-text pairs, bypassing the need for costly CIR datasets. Pic2Word
87

aligns entire images into text features, SEARLE adds a pseudo-word token to GPT-based captions,
88

and Context-I2W employs context-dependent word mapping for accurate retrieval. However, these
89

methods rely on the pre-trained CLIP language encoder, which struggles to understand intentions
90

within manipulation descriptions. To tackle this issue, we propose a novel model that effectively
91

2


---Page Break---
A rugby player passes the 

ball with his teammate.

Input Image

Original Caption

A rugby player in a purple and orange jersey is in the 

foreground running with the ball, while three other 
players in similar jerseys are in the background running 
towards him, indicating a game in progress. The field is 

green, and the focus is on the action of the players.

Rewritten Caption

Original Caption Rewriting

Rewrite the original caption like a human 

description in: 1. Describe the object's 
appearance, position, and relationship. 2. 

Describe what is in the background and 
foreground. 3. Focusing on colors, styles, and 
materials. 4. Identify the domain of the image.

Manipulation Intention Reasoning

You are a powerful human manipulation intents 

analyst. Infer human possible manipulation 

intents in an image caption.

In a game, a rugby player in a purple and orange jersey is 

running with the ball on a green field while the other 

three in similar jerseys are running toward him. 

Pseudo-Manipulation Description

①

①

①

②

②

②

🌋LLaVA

①
②Original Caption Rewriting Process

Pseudo-Manipulation Description Reasoning Process

Figure 1: Illustration of using LLaVA to create our intent-CC3M dataset. We first use a prompt to
guide the LLaVA model in generating rewritten captions with multi-view visual descriptions. Then,
we leverage another prompt to reason pseudo-manipulation descriptions with potential intentions.

understands these intentions, thereby improving the ZS-CIR model’s ability to retrieve images based
92

on human manipulation intents accurately. Unlike CIReVL [25], which employs LLMs during
93

inference for composed retrieval, introducing non-negligible computational overhead, our model is
94

lightweight and achieves comparable inference time to recent approaches.
95

Vision and Language Pre-training Models. Vision and Language Pre-training (VLP) models, like
96

CLIP [41], leverage extensive image-text pair training to achieve implicit alignment. Recent VLP
97

advancements [60, 49] utilize static models to integrate encoded image and text features, enabling
98

various zero-shot tasks [29, 49, 48]. However, current CLIP-based zero-shot learning struggles with
99

manipulation description in CIR tasks, motivating our approach, which enhances CLIP’s capabilities
100

of understanding user intentions to modify from fine-grained/long descriptions. Moreover, recent
101

studies [1, 28, 38, 37], inspired by DETR [7], employ learnable queries to select image and text
102

information. In our work, we utilize multiple learnable queries to guide the extraction of manipulation
103

intentions from target descriptions, providing explanatory cues for more accurate ZS-CIR.
104

Image-text Dataset Enhancement. In the field of vision-language learning, various endeavors
105

[17, 27, 18, 39, 10] aim to enhance caption quality within existing image-text datasets. LaCLIP [17]
106

utilizes LLMs to refine raw captions. VeCLIP [27] integrates insights from raw and synthetic sources
107

using LLMs. The latest approach, ShareGPT4V [10], leverages MLLMs to generate descriptive
108

captions from deliberate prompts and corresponding image inputs. However, these methods ignore
109

human manipulation intentions, which are crucial for CIR tasks. To bridge this gap, we introduce a
110

novel dataset infused with pseudo-manipulation intentions reasoned by MLLMs.
111

3
Methodology
112

3.1
Preliminary
113

Given a reference image space I and a text description space T , Composed Image Retrieval (CIR)
114

involves a user manipulation text T ∈T describing hypothetical semantic changes to a reference
115

image Ir ∈I, aiming to retrieve a target image with its closest context from an image database
116

D = {Ii, . . . , In}. Zero-Shot CIR (ZS-CIR) approaches [45, 3, 52] sidestep this requirement
117

by training a mapping network to map the reference image into an associated text representation.
118

Specifically, these methods learn a mapping function fθ : I →Z, where Z is a pre-defined text-
119

token embedding space. fθ is trained using intermediate image representations from a specific image
120

encoder ΨI, often part of a pre-trained vision-language representation system. Template filling
121

around the manipulation text over the pseudo token embedding S∗= fθ(ΨI(Ir)) is then employed
122

to aggregate information into a target description P (e.g., “a photo of S∗, {T}).” This target
123

description serves as input for target image retrieval, encoding it using the associated pre-trained text
124

encoder ΨT . The respective matching score is cos_sim(ΨI(Ir), ΨT (P)) using cosine similarity.
125

3.2
Creating Intention-based Image-text Aliagment Dataset
126

To address the discrepancy between pre-training and retrieval in existing ZS-CIR models, we aim
127

to develop an intention-based image-text dataset for training mapping networks capable of aligning
128

3


---Page Break---
intent-relevant visual information within the language space. To make a fair comparison and mitigate
129

the bias in human annotation, we propose to augment the widely used ZS-CIR training image-text
130

dataset, CC3M, through LLaVA [32], an open-source, state-of-the-art Multi-modal Large Language
131

Model (MLLM) known for its robust performance in vision-language tasks. However, reasoning
132

potential manipulation intentions from image-text pairs remains a challenging task for LLaVA.
133

Recent advancements in MLLMs include the development of Chain-of-Thought (CoT) prompting
134

[56], which enables MLLMs to produce a sequence of reasoning steps, breaking down multi-step
135

problems into intermediate stages and enhancing performance in complex tasks [24]. Inspired by the
136

CoT prompting mechanism, we explore a novel multimodal CoT prompting strategy using LLaVA to
137

reason pseudo-manipulation descriptions with potential intentions from image-text pairs effectively.
138

As illustrated in Figure 1, we divide the process of reasoning pseudo-manipulation descriptions
139

into two stages: the Caption Rewriting stage rewrites the original caption with multi-view visual
140

information for CIR tasks. The Intention Reasoning stage further understands the manipulation
141

intentions from rewritten captions to reason pseudo-manipulation descriptions. Specifically, in the
142

caption rewriting stage, we utilize the i-th image Ii and its original caption T i
ori from the CC3M,
143

denoted as D = {(Ii
r, T i
ori), . . . , (In
r , T n
ori)}. We guide the LLaVA model with a prompt to generate
144

a rewritten caption T i
rew for each image. These rewritten captions, averaging 65 tokens, include
145

various aspects of visual information (e.g., object, foreground/background, color, and domain style).
146

In the intention reasoning stage, we apply an additional prompt to reason manipulation intention for
147

rewritten captions. This results in a more effective pseudo-manipulation description T i
int, averaging 27
148

tokens. The result dataset is represented as ˜D = {(Ii
r, T i
ori, T i
rew, T i
int), . . . , (In
r , T n
ori, T n
rew, T n
int)}.
149

3.3
Manipulation Intention Understanding From Descriptions Before Searching
150

Since ZS-CIR models leverage the CLIP language encoder, there is a challenge in understanding
151

manipulation intentions that are implicitly expressed in user descriptions. To address this challenge,
152

we propose a method to understand the manipulation intention before feeding into the CLIP language
153

encoder for accurate ZS-CIR in two modules: the Manipulation Intention Understanding captures
154

manipulation intentions and maps them into several pseudo tokens. The Reasoning Distillation
155

further aligns the context of desired pseudo-word tokens closely with human intention by leveraging
156

pseudo-manipulation description to enhance the models’ ability to understand human intention.
157

Image and Context Encoding. For a given sample (Ir, Tori, Trew, Tint) from intent-CC3M. Since
158

the pre-trained vision-language models are strong at modeling the cross-modal implicit alignment.
159

Initially, we employ the frozen image encoder ΨI from the CLIP model to encode the global image
160

feature of the reference image Ir as v = ΨI(Ir) = {vi}d
i=1 ∈Rd×1. Subsequently, we apply a
161

simple mapping network fθ with parameters θ to extract a pseudo token embedding S∗= fθ(v).
162

Considering our focus on manipulation intention understanding for ZS-CIR, fθ is structured as a
163

simple three-layer fully-connected network. We then construct a target description P formatted
164

as “a photo of S∗, {T}”. We consider two scenarios for manipulation intention understanding:
165

deducing intention information from concise texts (e.g., original caption) or integrating it from
166

lengthy texts(e.g., rewritten caption). Accordingly, the text T is composed randomly within a batch
167

according to the following distribution: 50% original caption Trew and 30% rewritten caption Tori to
168

learn manipulation intention understanding, 20% pseudo-manipulation description Tint to ensure
169

training stability (details are in Appendix C). We feed the target description to the language encoder
170

ΨT of frozen CLIP to represent the target description P by a set of language feature vectors T
171

={ti}m
i=1 ⊆Rd×m. t1 represents the [CLS] embedding tcls with global information of image and
172

caption, while other ones denote word embeddings ˜T ={ti}m
i=2.
173

Manipulation Intentions Understanding. Given the word embeddings of the target descriptions,
174

this module aims to capture different manipulation intentions, thereby enhancing the CLIP lan-
175

guage encoder’s capability to understand users’ intents for manipulation. To capture different
176

manipulation intentions, we introduce a set of learnable query embeddings for guidance, denoted
177

as X = {xk}n
k=1 ∈Rd×n, where d is the embedding dimension and n is the number of queries.
178

Each query xk represents a kind of manipulation intention. As depicted in Figure 2(left), we im-
179

plement cross-attention mechanisms to extract intention-relevant contextual information from the
180

word embeddings ˜T = {ti}m
i=2 using the learnable queries X. The cross-attention operation in-
181

volves three primary steps. First, we compute the query, key and value through linear projections,
182

4


---Page Break---
Image and Context Encoding

Input Image

Language

Encoder

Image
Encoder

a  photo
of  𝑺∗

Target Description

,  A rugby
…  

...

...

...

Cross-Attention

Feed Forward

𝑞

#𝑻

×N

a  
photo

of  𝑺∗
,  In
a

…  

Language

Encoder

Distill 

Loss

Mapping
Network

Manipulation Intention Understanding Training

Pseudo-Manipulation 

Description

Candidate Images

Text-to-image
Retrieval

Inference with De-MINDS

Reference Image

𝑓$

Language

Encoder

Image
Encoder

Mapping
Network

𝑓$

is
a  photo  of  
𝑺∗
smaller
and
not
eating
...
De
-MINDS

Language

Encoder

Image
Encoder
Retrieved Image

Target Description

Gate
Alignment 

Loss

Manipulation Intention

Understanding

Gate

game  

Reasoning Distillation

①
②

Manipulation Intention Understanding Process
Reasoning Distillation Process

①
②

②
①

①

①

②

①

①

①

Figure 2: An overview of our De-MINDS. Pre-training (left): Map the image to a pseudo token S∗,
and understand the intention from the target description. Inference (right): Map the inference image
to S∗to construct the target description and understand manipulation intention for ZS-CIR.

i.e., Q = XW Q, K = [X, ˜T ]W K, V = [X, ˜T ]W V . [X, ˜T ] denotes concatenating the two
183

matrices, which enhances the interaction between learnable queries and word embeddings with better
184

performance. Then, the learnable queries from the current cross-attention block Xi is calculated as:
185

Xi
att = Att(Q, K, V ) = softmax

 
QK⊤

√

d

!

V , Xi = FFW(Xi
att + Xi−1) + Xi
att
(1)

where Xi−1 are learnable queries from the previous block and FFW(·) denotes 2-layer feed-forward
186

networks. the refined query embeddings X are then fed into the frozen language encoder ΨT of
187

CLIP to extract the intention embedding as t∗= ΨT (Xn) = {ti
∗}d
i=1 ∈Rd×1 (d = 768).
188

Reasoning Distillation. Given the intention embedding t∗, the AI agent needs to further align with
189

human manipulation intention. Specifically, we aim to reduce the distance between the intention
190

embedding and the corresponding pseudo-manipulation description’s [CLS] word embedding, which
191

represents the MLLM’s intention embedding while ensuring that each embedding remains distinct
192

and discriminative. Given the intention embeddings Tint = {ti
∗}N
i=1, where N is the number of
193

images in ˜D, and the corresponding MLLM’s intention embeddings ˜t∗= ΨT (Tint) ∈˜Tint we
194

employ a symmetric contrastive loss inspired by SimCLR [11, 13, 45] as follows:
195

Ldistil = Ls2t(t∗,˜t∗) + Lt2s(˜t∗, t∗)
(2)

The two contrastive loss terms are defined as:
196

Ls2t(t∗,˜t∗) = −1

|B|

X

i∈B
log
eτ(ti
∗)T ˜ti
∗
P

j∈B eτ(ti∗)T ˜tj
∗, Lt2s(ˆt∗,˜t∗) = −1

|B|

X

i∈B
log
eτ(˜ti
∗)T ti
∗
P

j∈B eτ(˜ti
∗)T tj
∗
(3)

where B is the number of images in a batch and τ is a temperature hyper-parameter that controls the
197

strength of penalties on hard negative samples.
198

Cross-Modal Alignment. Given the embedding of user manipulation intention, this module aims
199

to form a target embedding optimized for retrieval. Since the nature of CIR, both the reference
200

image and the manipulation intention form a comprehensive context that defines the target image. To
201

dynamically control the influence of manipulation intentions on the retrieval process, we introduce a
202

learnable scalar gate that decides the contribution of the manipulation intention information t∗and
203

integrates the global information tcls to form the final target embedding ˆt as follows:
204

ˆt = tcls + gate · t∗

Then, we aim to match a target image to its paired target embedding while separating unpaired
205

ones. We minimize the symmetric contrastive loss between the image embedding v and the target
206

embedding ˆt as follows:
207

Lalign = Ls2t(ˆt, v) + Lt2s(v,ˆt)
(4)
where Ls2t and Lt2s are two contrastive loss terms as Eq.3. The final loss used to optimize is:
208

L = Ldistill + Lalign
(5)

5


---Page Break---
Inference with De-MINDS. In the inference stage, we compose the reference image with the paired
209

manipulation description and compare the composed query with candidate images for retrieval. As
210

shown in Figure 2 (right), we compose the pseudo token embedding S∗of the image from the
211

mapping network with the text description and feed it to the pre-trained language encoder of CLIP.
212

The result is embedded by the text encoder and compared to the visual features of candidate images.
213

Since we focus on studying the manipulation intention understanding searching for ZS-CIR, we utilize
214

the same prompt in the most recent works [45, 52] for a fair comparison. We show prompt examples
215

for different ZS-CIR tasks. In all examples, [*] indicates the pseudo token from the mapping
216

network: (a) Domain conversion aims to modify the domain of the reference image. The prompt
217

is defined as a [domain tag] of [*]; (b) Object composition retrieves an image that contains
218

an object in the reference image and other object tags. The prompt is in the format of a photo
219

of [*], [obj1 tag] and [obj2 tag], . . . , and [objn tag]; (c) Sentence manipulation
220

modifies the reference image based on a sentence. We simply append the sentence with the special
221

token as a photo of [*], [sentence]. More details are in Appendix D.3.
222

4
Experiments
223

Datasets. We evaluate our model on four ZS-CIR datasets, i.e., COCO [31] for object composition,
224

ImageNet [16, 21] for domain conversion, CIRR [33] for object/scene manipulation, and Fashion-IQ
225

[57] for attribute manipulation. All the dataset settings and evaluation metrics (Recall@K) follow the
226

recent works [45, 52] for a fair comparison.
227

(1) Domain conversion. This dataset comprises 16,983 images of 200 classes from four domains,
228

i.e., cartoon, origami, toy, and sculpture. We use the prompt (a) in inference. (2) Object composition.
229

The dataset contains images with corresponding lists of object labels and instance masks of query
230

images. We randomly crop one object and mask its background using its instance mask to create a
231

reference image. We use the prompt (b) in inference. (3) Object/scene manipulation. A reference
232

image is an instruction for manipulating an object or the background scene. We apply the prompt
233

(c) in inference. (4) Attribute manipulation. This dataset includes various description sentences for
234

manipulating image attributes. We utilize the prompt (c) in inference. More details in Appendix D.2.
235

Implementation Details. Generating one pseudo-manipulation description through LLaVA-1.6-13B
236

[32] for the entire Conceptual Caption dataset [47], which comprises 3M images (CC3M), requires
237

approximately 625 hours on 5 A100 (80G) GPUs. For training De-MINDS, We utilize the CC3M and
238

adopt ViT-L/14 CLIP [41] pre-trained on 400M image-text paired data. We employ AdamW [34] with
239

a learning rate of 1 × 10−6, weight decay of 0.1, and a linear warmup of 10000 steps. The number
240

of cross-attention blocks is 6. The number of learnable queries is 4. The batch size for contrastive
241

learning is 1024. To improve training stability, we initialize the learnable scalar of tanh-gating to 0
242

[2]. For training Context-I2W and SEARLE, we keep the same setting reported in their paper, only
243

replacing the original captions with our pseudo-manipulation descriptions. All models are trained on
244

4 NVIDIA A100 (80G) GPUs. To ensure reliable results, we report the performance averaged over
245

three trials. More details are in Appendix D.1.
246

4.1
Quantitative and Qualitative Results
247

We compare De-MINDS with several ZS-CIR methods, including: 1) Pic2Word [45]: Maps the
248

visual features of a reference image into a pseudo-word token within the CLIP token embedding
249

space; 2) SEARLE-XL [3]: Similar to Pic2Word, further integrating the pseudo-word token with the
250

caption generated by GPT [6] and distilled for efficiency; 3) Context-I2W [52]: Selectively extracts
251

text-relevant visual information from the reference image before mapping it into a pseudo-word
252

token; 4) CIReVL [25]: Uses LLMs to enhance the manipulation description during inference; and
253

5) LinCIR [20]: Masks subjects in captions from various image-text datasets for training. For a fair
254

comparison, we present the reported results of methods relying on the ViT-L/14 CLIP model.
255

Moreover, we compare De-MINDS with 6) SEARLE-XL* and Context-I2W*: Replace the original
256

captions with our pseudo-manipulation description, and standard ZS-CIR methods, including 7)
257

Text-only: Computes similarity based on the CLIP features of descriptions and candidate images; 8)
258

Image-only: Retrieves the most similar images to the reference image; and 9) Image + Text: Sums
259

the CLIP features of the reference image and the description.
260

6


---Page Break---
Table 1: Results on Fashion-IQ for attribute manipulation.

Dress
Shrit
TopTee
Average

Methods
Conferences
R10
R50
R10
R50
R10
R50
R10
R50

Image-only
–
5.4
13.9
9.9
20.8
8.3
17.7
7.9
17.5
Text-only
–
13.6
29.7
18.9
31.8
19.3
37.0
17.3
32.9
Image+Text
–
16.3
33.6
21.0
34.5
22.2
39.0
19.8
35.7
Pic2Word [45]
CVPR 2023
20.0
40.2
26.2
43.6
27.9
47.4
24.7
43.7
CIReVL [25]
ICLR 2024
24.6
44.8
29.5
47.4
31.4
53.7
28.6
48.6
LinCIR [20]
CVPR 2024
20.9
42.4
29.1
46.8
28.8
50.2
26.3
46.5

SEARLE-XL [3]
ICCV 2023
20.3
43.2
27.4
45.7
29.3
50.2
25.7
46.3
SEARLE-XL*
–
22.7
45.0
29.4
47.9
30.2
51.4
27.4
48.1

Context-I2W [52]
AAAI 2024
23.1
45.3
29.7
48.6
30.6
52.9
27.8
48.9
Context-I2W*
–
23.9
46.9
30.4
49.7
31.1
53.8
28.5
50.1

De-MINDS
–
25.2
48.7
31.0
51.2
32.9
55.7
29.7
51.9

Ours

is black and long 
sleeves with red 
and white designs 

at the center

Query

B00AN545PI.png

has longer sleeves 
and color is blue with 

buttoned front and 
has double pockets

is a shorter, sexier, 

tighter fit and a 
lighter color with a 

waistband

Context-I2W

Figure 3: Results on the attribute manipulation task

Ours

Origami

Toy

Query

Cartoon

Sculpture

Context-I2W

Figure 4: Results on the domain conversion task.

Tables 1 to 4 present the quantitative results, while Figures 3 to 6 display the corresponding qualitative
261

results of our model and the most recent works, CIReVL and Context-I2W. The attribute manipulation
262

task requires accurately localizing specific attributes within the entire image. As demonstrated in Table
263

1, De-MINDS outperforms existing ZS-CIR models significantly, achieving an average improvement
264

of 2.20% over the State-of-the-Art (SoTA) model, CIReVL. CIReVL’s dependency on an LLM at
265

inference introduces substantial computational overhead during retrieval. De-MINDS tackles this
266

challenge by extracting fashion-relevant intention within manipulation descriptions into a series of
267

implicit pseudo-tokens for CLIP retrieval. This approach is more efficient and suitable for models than
268

relying on explicit, often noisy, LLM analysis results. Figure 3 further illustrates how De-MINDS
269

effectively understand complex fashion-relevant attributes in manipulation descriptions, such as a
270

sexier style with a waistband (row 1), black color with a special design in the center (row 2), and
271

longer sleeves with two pockets in blue (row 3), facilitating more accurate searching.
272

We further assess De-MINDS’ capability in foreground/background differentiation and fine-grained
273

image editing through the object/scene manipulation task (Table 2). De-MINDS consistently surpasses
274

existing ZS-CIR models, achieving an average performance improvement of 2.05% over the best
275

model. This enhancement is attributed to De-MINDS’ approach of extracting human intention from
276

manipulation descriptions before searching, enhancing the ability of the CLIP language encoder
277

to understand the user’s intention to modify. In Figure 5, De-MINDS accurately understands
278

manipulation intention to change the number of an object and modify the background (row 1), alter
279

the stage and remove an overlapping object (row 2), adjust the camera focus, age of a dog, and
280

remove a specific object (row 3), and modify the style of an image with a specific design (row 4).
281

In the object composition experiments (Table 3), De-MINDS significantly outperforms the current
282

SoTA model by an average of 4.30%. These results prove the effectiveness of De-MINDS in
283

accurately mapping visual information to the language token space via bridges the gap between
284

pre-training and retrieval, which facilitates the combination of multiple objects, as shown in Figure 6.
285

Moreover, in the domain conversion results (Table 4), De-MINDS consistently outperforms existing
286

approaches and notably surpasses the SoTA Context-I2W by an average of 4.35%. As illustrated in
287

Figure 4, De-MINDS accurately maps objects within complex scenes (e.g., a saxophonist in the street,
288

a bald eagle on wood, a monkey in the forest, and a sea lion in the water). In contrast, Context-I2W
289

struggles to select the intention-relevant local visual features due to its reliance on image caption
290

without intention, whereas our pseudo-manipulation descriptions are effectively addressed.
291

7


---Page Break---
Table 2: Results on CIRR for object
manipulation task.

Methods
R1
R5
R10 R50

Image-only
7.4
23.6 34.0 57.4
Text-only
20.9 44.8 55.5 79.1
Image+Text
12.4 36.2 49.1 78.2
Pic2Word [45]
23.9 51.7 65.3 87.8
CIReVL [25]
24.6 52.3 64.9 86.3
LinCIR [20]
25.0 53.3 66.7
–

SEARLE-XL [3]
24.2 52.4 66.3 88.6
SEARLE-XL*
25.4 54.1 66.9 89.3

Context-I2W [52] 25.6 55.1 68.5 89.8
Context-I2W*
26.3 55.7 69.0 90.2

De-MINDS
27.3 57.0 71.3 91.6

Table 3: Results on COCO for object
composition task.

Methods
R1
R5
R10

Image-only
8.6
15.4
18.9
Text-only
6.1
15.7
23.5
Image+Text
10.2
20.2
26.6
Pic2Word [45]
11.5
24.8
33.4

Context-I2W [52]
13.5
28.5
38.1
Context-I2W*
14.3
29.7
40.5

De-MINDS
15.7
33.2
44.1

Ours
Context-I2W
Query

Target two animals 

resting on white 
towel rather showing 

one black

Take the picture 
closer, make the dog 
younger, and remove 

the person

dev-224-2-img1.png

Make dog sleep in 

couch or ground 
and remove objects 

from its mouth

make it a poster of 

the dog, and have 

text above and 
below the animal
Figure 5: Retrieved results on the object manipulation task

Ours

train, light, 
people, railway,

package, sky

Query

man, woman, table,

bottle, food, knife, 

fork, wine

leaves, person, food, 

chair, table, plate,

fork, bread

Context-I2W

Figure 6: Retrieved results on the object composition task.
Table 4: Results on ImageNet for domain conversion.

Cartoon
Origami
Toy
Sculpture
Average

Methods
Conferences
R10
R50
R10
R50
R10
R50
R10
R50
R10
R50

Image-only
–
0.3
4.5
0.2
1.8
0.6
5.7
0.3
4.0
0.4
4.0
Text-only
–
0.2
1.1
0.8
3.7
0.8
2.4
0.4
2.0
0.5
2.3
Image+Text
–
2.2
13.3
2.0
10.3
1.2
9.7
1.6
11.6
1.7
11.2
Pic2Word [45]
CVPR 2023
8.0
21.9
13.5
25.6
8.7
21.6
10.0
23.8
10.1
23.2

Context-I2W [52]
AAAI 2024
10.2
26.1
17.5
28.7
11.6
27.4
12.1
28.2
12.9
27.6
Context-I2W*
–
11.2
27.4
18.7
30.4
12.5
29.8
13.7
31.4
14.0
29.8

De-MINDS
–
13.3
31.2
20.3
34.5
14.7
31.7
16.5
34.7
16.2
33.0

4.2
Ablation Study
292

In Table 5, we evaluate the contributions of De-MINDS components on the CIRR and FashionIQ
293

datasets. (1) In models ‘2-3’, we assess the significance of the intent-CC3M dataset. Replacing the
294

pseudo-manipulation description with original captions (model ‘2’) results in an average performance
295

drop of 3.80%, demonstrating training with intent-CC3M benefit for aligning intention-relevant
296

visual information. Using a single prompt for pseudo-manipulation descriptions (model ‘3’) causes a
297

3.14% performance decline, indicating that CoT prompting enhances MLLM in reasoning potential
298

manipulation intention. (2) In models ‘4-6’, we evaluate key modules in the manipulation intention
299

understanding process. Without intention embeddings from De-MINDS (model ‘4’), performance
300

drops by 4.02% on average, proving De-MINDS’s importance in CIR. Removing the global feature
301

tcls (model ‘5’) leads to a 2.38% performance decline, highlighting the necessity of comprehensive
302

both global and intention information. Summing global and intention features directly (model
303

‘6’) causes a 1.64% performance drop, indicating the need for adaptive capture of complementary
304

information. (3) In models ‘7-9’, we assess De-MINDS’s training strategies. Using only original
305

captions as T (model ‘7’) reduces training stability, resulting in a 1.62% performance drop. Without
306

the distillation loss (model ‘8’) or replacing it with a cosine loss (model ‘9’) leads to performance
307

drops of 3.58% and 1.54%, respectively, indicating the necessity of symmetric contrastive loss for
308

distilling MLLM’s reasoning ability. In models ‘10-12’, we evaluate alternative solutions. Not
309

utilizing T for image-to-text mapping (model ‘10’) results in a 2.30% performance drop, confirming
310

the effectiveness of our pseudo-manipulation descriptions. Applying MiniGPT-4 [61] to generate the
311

intent-CC3M dataset (model ‘11’) results in a 1.18% performance drop, suggesting that a superior
312

MLLM model benefits pseudo-manipulation description quality. Leveraging the LLaMA [53] rewrite
313

8


---Page Break---
Table 5: Ablation study of main components
on CIRR and FashionIQ.

CIRR
Fashion-IQ

Methods
R1
R5
R10
R10
R50

1.
full model
27.3 57.0 71.3 29.7
51.9
Significant of inetent-CC3M
2.
w/o intent-CC3M
24.6 53.7 67.1 26.0
46.8
3.
w/o CoT
25.2 54.3 67.8 26.7
47.5
Key modules of De-MINDS process
4.
w/o De-MINDS
24.0 53.5 67.2 25.8
46.6
5.
w/o global feature
25.5 55.2 68.0 27.3
49.6
6.
w/o gate
25.9 55.3 69.5 27.9
50.4
Training Strategies
7.
w/o construct T
26.2 55.6 69.3 27.8
50.2
8.
w/o distil
24.8 53.9 67.3 26.3
47.0
9.
cos distll
26.2 55.5 69.7 27.9
50.2
Alternative solutions
10. a photo of S∗
25.5 55.2 67.9 27.5
49.6
11. MiniGPT4’s caption 26.4 55.7 70.2 28.2
50.8
12. LLM’s caption
25.2 53.7 67.2 26.9
47.2

Shows another room with a 

side table and chair, except 
they are each in front of two 
windows in a corner and the 

chairs have cushions.

Manipulation Description

Remove all dogs and basket, 

Add adult dog standing and 

alert, Place dog on cement 

pavement with handler 
seated behind dog's head.

Learnable Queries

Standing guinea pig on the 
background of toys instead of 

a white-red puppy sleeping

on a boot on the ground.

Reference Image
Retrieved Image

Place dog standing on hind 
legs, Add another dog, and 
Place dogs in a commercial, 
industrial setting with orange

background.

Figure 7: Visualization of the top two attention
words for each learnable query, different colors
denoting the results corresponding to each query.

CC3M dataset [17] (model ‘12’) causes a 3.40% performance drop, indicating the necessity of MLLM
314

for generating pseudo-manipulation description with multi-view supplementary image detail.
315

4.3
Analysis
316

Interpretability of Learnable Query. In Figure 7, we visualize the top two attention words of each
317

learnable query from the last block, demonstrating the distinct focus of the four queries. Specifically,
318

the first two queries mainly focus on object and attribute information, while the last two queries
319

mostly consider foreground/background and relation information. These attention maps substantiate
320

De-MINDS’s interpretability in extracting specific intention across various descriptions, supporting
321

the understanding of intention from manipulation descriptions.
322

Effectiveness and Efficiency Analysis. Our approach achieves significant improvements on four
323

widely compared ZR-CIR tasks from 2.05% to 4.35% over the SoTA models. Designed for under-
324

standing manipulation intention, the model size of De-MINDS(58.5M) is larger than the simple
325

3-layer MLP mapping (0.9M) of Pic2Word. Consequently, our training time (20 hours) is 6 hours
326

longer than Pic2Word under the same settings. Notably, our inference time (0.017s) is ×58 faster
327

than CIReVL (∼1s), which uses LLM for inference, and only 0.005s slower than Pic2Word. It’s
328

worth noting that our model using just 50% of the pre-training data achieves comparable performance
329

to SoTA models (details are in Appendix A.2).
330

Limitation. While the training process for De-MINDS does not introduce significant additional
331

memory or computational overhead, generating pseudo-manipulation descriptions using MLLMs
332

can be computationally intensive. Moreover, these pseudo descriptions are not filtered, potentially
333

introducing irrelevant details that do not align with actual human manipulation intention. Our paper
334

aims to bridge the gap between pre-training and retrieval in ZS-CIR models and introduce a novel
335

framework to enhance the model’s capability to understand user intention. Future work could explore
336

more efficient methods to generate pseudo-manipulation descriptions while maintaining performance.
337

5
Conclusion
338

In this paper, we introduce intent-CC3M, an intention-based dataset featuring pseudo-manipulation
339

descriptions reasoned through chain-of-thought prompting by an MLLM for training mapping
340

networks to align intention-relevant visual information. Leveraging intent-CC3M, we propose a
341

novel manipulation intention understanding network that employs learnable queries to enhance the
342

models’ capability to understand user intention from manipulation descriptions for accurate CIR.
343

De-MINDS shows strong generalization ability and remarkably improves the best performance of
344

existing approaches on four diverse ZS-CIR tasks with comparable inference times. Our work inspires
345

intention-based image retrieval and impacts diverse vision and language applications.
346

9


---Page Break---
References
347

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc,
348

Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi,
349

Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud,
350

Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bi´nkowski, Ricardo Barreira, Oriol Vinyals,
351

Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning.
352

In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural
353

Information Processing Systems, volume 35, pages 23716–23736, 2022.
354

[2] Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley.
355

Rezero is all you need: Fast convergence at large depth. In Uncertainty in Artificial Intelligence, pages
356

1352–1361, 2021.
357

[3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image
358

retrieval with textual inversion. arXiv:2303.15247, 2023.
359

[4] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Effective conditioned and
360

composed image retrieval combining clip-based features. In Proceedings of the IEEE/CVF Conference on
361

Computer Vision and Pattern Recognition, pages 21466–21474, June 2022.
362

[5] Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov.
363

Knowledge distillation: A good teacher is patient and consistent. In Proceedings of the IEEE/CVF
364

conference on computer vision and pattern recognition, pages 10925–10934, 2022.
365

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
366

Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
367

Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
368

Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,
369

Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models
370

are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances
371

in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.
372

[7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey
373

Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision,
374

pages 213–229, 2020.
375

[8] Akshay Chawla, Hongxu Yin, Pavlo Molchanov, and Jose Alvarez. Data-free knowledge distillation for
376

object detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,
377

pages 3289–3298, 2021.
378

[9] Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, and Manmohan Chandraker. Learning efficient object
379

detection models with knowledge distillation. In Proc. of Advances in Neural Information Processing
380

Systems (NeurIPS), volume 30, 2017.
381

[10] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin.
382

Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793,
383

2023.
384

[11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
385

contrastive learning of visual representations. In Proc. of International Conference on Machine Learning
386

(ICML), pages 1597–1607. PMLR, 2020.
387

[12] Yanbei Chen, Shaogang Gong, and Loris Bazzani. Image search with text feedback by visiolinguistic atten-
388

tion learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
389

pages 3001–3011, 2020.
390

[13] Niv Cohen, Rinon Gal, Eli A. Meirom, Gal Chechik, and Yuval Atzmon. "This is my unicorn, Fluffy":
391

Personalizing frozen vision-language representations. In Proc. of the European Conference on Computer
392

Vision (ECCV), 2022.
393

[14] Niv Cohen, Rinon Gal, Eli A. Meirom, Gal Chechik, and Yuval Atzmon. “this is my unicorn, fluffy”:
394

Personalizing frozen vision-language representations. In European conference on computer vision, pages
395

558–577, 2022.
396

[15] Ritendra Datta, Dhiraj Joshi, Jia Li, and James Z Wang. Image retrieval: Ideas, influences, and trends of
397

the new age. ACM Computing Surveys, 40(2):1–60, 2008.
398

[16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
399

image database. In Computer Vision and Pattern Recognition, pages 248–255, 2009.
400

[17] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with
401

language rewrites. Advances in Neural Information Processing Systems, 36, 2024.
402

[18] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen,
403

Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next
404

generation of multimodal datasets. Advances in Neural Information Processing Systems, 36, 2024.
405

[19] Sonam Goenka, Zhaoheng Zheng, Ayush Jaiswal, Rakesh Chada, Yue Wu, Varsha Hedau, and Pradeep
406

Natarajan. Fashionvlp: Vision language transformer for fashion retrieval with feedback. In Proceedings of
407

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14105–14115, June 2022.
408

10


---Page Break---
[20] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, , Yoohoon Kang, and Sangdoo Yun. Language-only efficient
409

training of zero-shot composed image retrieval. In Conference on Computer Vision and Pattern Recognition
410

(CVPR), 2024.
411

[21] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai,
412

Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces
413

of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF
414

International Conference on Computer Vision, pages 8340–8349, 2021.
415

[22] Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In NIPS
416

Deep Learning and Representation Learning Workshop, 2015.
417

[23] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780,
418

1997.
419

[24] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei
420

Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with
421

language models. Advances in Neural Information Processing Systems, 36, 2024.
422

[25] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. Vision-by-language
423

for training-free compositional image retrieval. In The Twelfth International Conference on Learning
424

Representations, 2024.
425

[26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept
426

customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision
427

and Pattern Recognition, pages 1931–1941, 2023.
428

[27] Zhengfeng Lai, Haotian Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan,
429

Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, et al. From scarcity to efficiency: Improving clip training via
430

visual-enriched captions. arXiv preprint arXiv:2310.07699, 2023.
431

[28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training
432

with frozen image encoders and large language models, 2023.
433

[29] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training
434

for unified vision-language understanding and generation. In Proceedings of the 39th International
435

Conference on Machine Learning, pages 12888–12900, 2022.
436

[30] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong
437

Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In
438

European Conference on Computer Vision, pages 121–137, 2020.
439

[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
440

and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla,
441

Bernt Schiele, and Tinne Tuytelaars, editors, European Conference on Computer Vision, pages 740–755,
442

2014.
443

[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural
444

information processing systems, 36, 2024.
445

[33] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life
446

images with pre-trained vision-and-language models. In Proceedings of the IEEE/CVF International
447

Conference on Computer Vision, pages 2125–2134, October 2021.
448

[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on
449

Learning Representations, 2018.
450

[35] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim
451

Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on
452

Computer Vision and Pattern Recognition, pages 14297–14306, 2023.
453

[36] Ron Mokady, Amir Hertz, and Amit H. Bermano. Clipcap: Clip prefix for image captioning, 2021.
454

[37] Muhammad Ferjad Naeem, Muhammad Gul Zain Ali Khan, Yongqin Xian, Muhammad Zeshan Afzal,
455

Didier Stricker, Luc Van Gool, and Federico Tombari. I2mvformer: Large language model generated
456

multi-view document supervision for zero-shot image classification. In Proceedings of the IEEE/CVF
457

Conference on Computer Vision and Pattern Recognition, pages 15169–15179, 2023.
458

[38] Muhammad Ferjad Naeem, Yongqin Xian, Luc V Gool, and Federico Tombari. I2dformer: Learning image
459

to document attention for zero-shot image classification. Advances in Neural Information Processing
460

Systems, 35:12283–12294, 2022.
461

[39] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving
462

multimodal datasets with image captioning. Advances in Neural Information Processing Systems, 36, 2024.
463

[40] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen,
464

Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep
465

learning library. NeurIPS, 32, 2019.
466

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
467

Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning trans-
468

ferable visual models from natural language supervision. In Proceedings of the International Conference
469

on Machine Learning, pages 8748–8763, 2021.
470

[42] Aditya Ramesh, Mukul Goyal, and Rob Fergus. Dall-e: Creating images from text. OpenAI Blog, 2021.
471

11


---Page Break---
[43] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua
472

Bengio. FitNets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.
473

[44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dream-
474

booth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the
475

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
476

[45] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister.
477

Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the
478

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19305–19314, 2023.
479

[46] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation.
480

arXiv preprint arXiv:2311.17042, 2023.
481

[47] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned,
482

hypernymed, image alt-text dataset for automatic image captioning. In Annual Meeting of the Association
483

for Computational Linguistics, pages 2556–2565, 2018.
484

[48] Jiangming Shi, Yachao Zhang, Xiangbo Yin, Yuan Xie, Zhizhong Zhang, Jianping Fan, Zhongchao Shi,
485

and Yanyun Qu. Dual pseudo-labels interactive self-training for semi-supervised visible-infrared person
486

re-identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
487

11218–11228, 2023.
488

[49] Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners:
489

Empirical studies on vqa and visual entailment, 2022.
490

[50] Derek Tam, Colin Raffel, and Mohit Bansal. Simple weakly-supervised image captioning via CLIP’s
491

multimodal embeddings. In The AAAI-23 Workshop on Creative AI Across Modalities, 2023.
492

[51] Yingtian Tang, Yutaro Yamada, Yoyo Zhang, and Ilker Yildirim. When are lemons purple? the concept
493

association bias of vision-language models. In Proceedings of the 2023 Conference on Empirical Methods
494

in Natural Language Processing, pages 14333–14348, 2023.
495

[52] Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Yue Hu, and Qi Wu. Context-i2w: Map-
496

ping images to context-dependent words for accurate zero-shot composed image retrieval. In Proceedings
497

of the AAAI Conference on Artificial Intelligence, volume 38, pages 5180–5188, 2024.
498

[53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
499

Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation
500

language models. arXiv preprint arXiv:2302.13971, 2023.
501

[54] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text
502

and image for image retrieval - an empirical odyssey. In Proceedings of the IEEE/CVF Conference on
503

Computer Vision and Pattern Recognition, pages 6439–6448, 2019.
504

[55] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and
505

image for image retrieval-an empirical odyssey. In Proceedings of the IEEE/CVF conference on computer
506

vision and pattern recognition, pages 6439–6448, 2019.
507

[56] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
508

Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural
509

information processing systems, 35:24824–24837, 2022.
510

[57] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris.
511

Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the
512

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11307–11317, 2021.
513

[58] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the
514

long-text capability of clip, 2024.
515

[59] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and
516

Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the
517

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5579–5588, 2021.
518

[60] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for
519

vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
520

Recognition, pages 16816–16825, 2022.
521

[61] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing
522

vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592,
523

2023.
524

[62] Wanrong Zhu, An Yan, Yujie Lu, Wenda Xu, Xin Eric Wang, Miguel Eckstein, and William Yang Wang.
525

Visualize before you write: Imagination-guided open-ended text generation, 2023.
526

A
Extended Analysis
527

A.1
Analysis of the number of learnable queries.
528

We conduct analysis on the number of learnable query embedding X = {xk}n
k=1 ∈Rd×n as shown
529

in Figure 8. We find that n = 2 results in not learning sufficient intentions for manipulation, but
530

12


---Page Break---
26.5

27.3

25.8

25.1

24.5

29.7

31

29.1

28.4

27.5

31.8

33

31.1
30.7

30.1

32.5

33.2

31.8

31.2

30.6

24

25

26

27

28

29

30

31

32

33

34

2 queries
4 queries
8 queries
16 queries
32 queries

CIRR R@1
Shirt R@10
ImageNet R@50
COCO R@5

Figure 8: Analysis of the number of learnable queries.

when n is added to 32, it is redundant and unhelpful for the CLIP model to understand manipulation
531

intentions. We finally choose n = 4, which gives the best result among different settings.
532

Table 6: Results on ImageNet for domain conversion.

Cartoon
Origami
Toy
Sculpture
Average

Methods
Conferences
R10
R50
R10
R50
R10
R50
R10
R50
R10
R50

Pic2Word [45]
CVPR 2023
8.0
21.9
13.5
25.6
8.7
21.6
10.0
23.8
10.1
23.2

Context-I2W [52]
AAAI 2024
10.2
26.1
17.5
28.7
11.6
27.4
12.1
28.2
12.9
27.6
Context-I2W*
–
11.2
27.4
18.7
30.4
12.5
29.8
13.7
31.4
14.0
29.8

Context-I2W(50 %)
AAAI 2024
9.0
23.0
14.3
25.6
10.7
25.0
11.0
25.5
11.3
24.8
De-MINDS(50 %)
–
11.7
28.3
19.2
30.9
12.8
30.2
14.2
32.0
14.5
30.4
De-MINDS(100 %)
–
13.3
31.2
20.3
34.5
14.7
31.7
16.5
34.7
16.2
33.0

Table 7: Results on CIRR for object manipu-
lation task.

Methods
R1
R5
R10
R50

Pic2Word [45]
23.9
51.7
65.3
87.8
CIReVL [25]
24.6
52.3
64.9
86.3
LinCIR [20]
25.0
53.3
66.7
–

SEARLE-XL [3]
24.2
52.4
66.3
88.6
SEARLE-XL*
25.4
54.1
66.9
89.3

Context-I2W [52]
25.6
55.1
68.5
89.8
Context-I2W*
26.3
55.7
69.0
90.2

Context-I2W(50%)
24.8
53.6
67.1
88.9
De-MINDS (50%)
26.5
56.0
69.3
90.5
De-MINDS
27.3
57.0
71.3
91.6

Table 8: Results on COCO for object composition
task.

Methods
R1
R5
R10

Pic2Word [45]
11.5
24.8
33.4

Context-I2W [52]
13.5
28.5
38.1
Context-I2W*
14.3
29.7
40.5

Context-I2W(50%)
12.1
25.6
34.4
De-MINDS (50%)
14.6
30.4
40.8
De-MINDS (100%)
15.7
33.2
44.1

A.2
More Effectiveness and Efficiency Analysis
533

In Table 6 to 9, we present more evidence supporting the efficacy and efficiency of our De-MINDS.
534

With only 50% of the training data, De-MINDS matches and exceeds the performance of the state-
535

of-the-art (SoTA) Context-I2W model by 0.83% to 2.20%. Remarkably, De-MINDS outperforms
536

reported results of the SoTA model by 1.98% to 4.57% under the same 50% training data, underscoring
537

our method’s superiority.
538

13


---Page Break---
Table 9: Results on Fashion-IQ for attribute manipulation.

Dress
Shrit
TopTee
Average

Methods
Conferences
R10
R50
R10
R50
R10
R50
R10
R50

Pic2Word [45]
CVPR 2023
20.0
40.2
26.2
43.6
27.9
47.4
24.7
43.7
CIReVL [25]
ICLR 2024
24.6
44.8
29.5
47.4
31.4
53.7
28.6
48.6
LinCIR [20]
CVPR 2024
20.9
42.4
29.1
46.8
28.8
50.2
26.3
46.5

SEARLE-XL [3]
ICCV 2023
20.3
43.2
27.4
45.7
29.3
50.2
25.7
46.3
SEARLE-XL*
–
22.7
45.0
29.4
47.9
30.2
51.4
27.4
48.1

Context-I2W [52]
AAAI 2024
23.1
45.3
29.7
48.6
30.6
52.9
27.8
48.9
Context-I2W*
–
23.9
46.9
30.4
49.7
31.1
53.8
28.5
50.1

Context-I2W(50%)
AAAI 2024
21.4
43.7
28.1
46.9
29.7
51.4
26.4
47.3
De-MINDS (50%)
–
24.3
47.5
30.6
50.0
31.3
54.0
28.7
50.5
De-MINDS (100%)
–
25.2
48.7
31.0
51.2
32.9
55.7
29.7
51.9

Algorithm 1 Manipulation Intention Understanding’s process.

Input: batch of word embeddings of target descriptions ˜T = {ti}m
i=1, where t1 is the global feature
tcls, Nlayer, the frozen CLIP language encoder ΨT
Parameter: a set of learnable embeddings X ∈Rd×n , 8-heads attention layer Attn, 3-layers FC
layers fM, gateα.
Output: target embedding ˆt

1: Initialize X ∈Rd×n, Attn, fM randomly.
2: Let Xi
att = {ti}m
i=2, t = 1
3: while t ≤Nlayer do
4:
Xi+1
att = Xi
att + Attnt(q=q, k=concat([Xi
att, q]), v=concat([Xi
att, q]))
5:
Xi+1
att = Xi+1
att + fMt(Xi+1
att )
6:
t = t + 1
7: end while
t∗= ΨT (Xoutput)
ˆt = tcls + tanh(gateα) · t∗
8: return ˆt

A.3
Broader Impact
539

We propose a novel image-text dataset augmentation strategy that generates diverse rewrites for
540

any given image-text pair. This approach not only bolsters the performance of vision-language
541

models but also enhances capabilities in textual inversion [44], including text-to-image generation
542

via diffusion models and personalized image retrieval. However, it is crucial to note that MLLMs are
543

trained on extensive web data, which may incorporate factual inaccuracies and hallucinatory content.
544

Consequently, the intention-infused versions of texts could inherit these flaws. We advocate for
545

the implementation of rigorous data filtering methods before these models’ deployment in practical
546

settings. Furthermore, while the MLLM-based rewriting strategy demands substantial GPU/TPU
547

computational resources, potentially increasing the carbon footprint.
548

A.4
Qualitative Results of intent-CC3M
549

Figure 9 to 10 we leverage DALL-E [42] to generate images of each caption for qualitative experiment.
550

We compare intent-CC3M with the CC3M dataset and GPT4’s rewritten captions. We found that
551

the captions of Intent-CC3M, which contain potential manipulation intentions, provide better visual
552

information compared to the original captions and those rewritten by a large language model. This
553

improvement is due to incorporating diverse visual perspectives (e.g., colors, scenes, and objects)
554

using a multi-model language model, which enhances the training of text-to-image generation tasks.
555

Notably, our pseudo-manipulation descriptions are shorter than the rewritten captions. The results
556

show that pseudo-manipulation descriptions serve as more effective prompts, enabling DALL-E to
557

generate results that are closer to the original images. This demonstrates the high quality of our
558

pseudo-manipulation descriptions.
559

14


---Page Break---
Original Caption

a street musician 

plays an accordion.

LLM’s 

Rewritten Caption

A street performer 

serenades 

passersby with the 

melodies of an 

accordion.

Generated Image
Pseudo-Manipulation 

Description

Man plays accordion 

joyfully in the street, 

with cafe background 

suggesting a lively, 

public musical 

performance.

Generated Image

A man is playing an accordion on a 

street. He is standing in the foreground, 

holding the accordion in his hands. 

There are several chairs and tables in the 

background, suggesting a cafe or 

outdoor seating area. The man is smiling, 

indicating he is enjoying his 

performance. The accordion is black and 

has a metallic finish. The street is lined 

with buildings, and there are a few other 

people in the background.      

Our Rewritten Caption
Generated Image

Automotive industry 

business now offers 

a new tuning 

package for 

automobile model.

The automotive 

industry now 

introduces a fresh 

tuning package for a 

specific car model.

Highlights Black VW 

Beetle with tuning 

package, sleek design, 

and glossy finish, 

emphasizing driving on 

rural road.

A black Volkswagen Beetle in motion on 

a road. The car is equipped with a new 

tuning package, featuring a body kit with 

a front bumper, side skirts, and a rear 

bumper, all in the same black color. The 

car's sleek and modern design has a 

glossy finish that reflects the 

surrounding environment. The 

background is a blur of greenery,, which 

contrasts with the car's urban aesthetic. 

he domain of the image is photography, 

capturing a real-life scene. 

on a sunny winter day.
Underneath the crisp 

winter sun.

A large, round, stone 

castle with multiple 

towers, highlighted by 

stands in the middle of 

a lush green field, 

surrounded by a moat, 

under a clear blue sky.

A large, round, stone castle with 

multiple towers stands in the middle of 

a lush green field. The castle is 

surrounded by a moat, and the sky is 

clear and blue. The sun is shining 

brightly, casting long shadows and 

highlighting the textures of the stone. 

The castle is the central focus of the 

image, with the open field stretching out 

around it. 

I start the season 

with a big tree.

I kick off the season 

with a towering tree.

A large tree in the center of 

a town square, surrounded 

by white buildings with 

blue accents, under a blue 

sky. The square is bustling 

with people, showcasing a 

European-influenced 

architecture.

A large, leafy tree stands prominently in 

the center of a town square, surrounded 

by white buildings with blue accents. The 

tree is the focal point, with its vibrant 

green leaves contrasting against the clear 

blue sky. The town square is bustling with 

people, adding life to the scene. The 

architecture of the buildings suggests a 

European influence, and the overall 

atmosphere is serene and picturesque.

Original Image

Figure 9: Qualitative results of our intent-CC3M dataset. We leverage DALL-E to generate images of
the captions. We compare intent-CC3M with the CC3M dataset and LLM’s rewritten captions.

15


---Page Break---
Original Caption

A toddler laughs in 

enjoyment on the 

playground.

A young child 

giggles with delight 

while playing on the 

playground.

Generated Image

A young girl is laughing 

on a swing, wearing a 

colorful dress, with two 

adults watching her. 

The swing is yellow and 

wooden. They are in a 

park-like setting.

Generated Image

A young girl is sitting on a playground 

swing, laughing and enjoying herself. 

She is wearing a colorful dress and has 

blonde hair. Two adults are standing 

nearby, watching her with smiles. The 

swing is yellow and has a wooden seat. 

In the background, there is a sandy area 

and a fence. The scene is set in a park-

like environment.

Our Rewritten Caption
Generated Image

a shepherd pictured 

with his flock of 

sheep in north.

A shepherd 

captured alongside 

his flock of sheep in 

the northern region.

Man in plaid shirt and 

flat cap stands in field 

with sheep, smiling, 

looking at camera, on 

clear, blue day.

A man in a plaid shirt and a flat cap 

stands in a field with a flock of sheep. He 

is smiling and looking directly at the 

camera. The sheep are scattered around 

him, grazing on the grass. The sky is 

clear and blue, suggesting a sunny day. 

The man's attire and the presence of the 

sheep suggest a rural or pastoral setting. 

The image captures a moment of 

peaceful coexistence between the man 

and his flock.

A child touches an ice 

sculpture.

A child reaches out to 

touch an ice sculpture.

Child reaches out to touch ice 

dragon sculpture, intricately 

designed with scales and 

spikes, standing in front of it 

on sandy ground.

A child reaches out to touch a large ice 

sculpture of a dragon, which is 

intricately designed with scales and 

spikes. The sculpture is positioned on a 

sandy ground, and the child is standing 

in front of it. The background is dark, 

highlighting the brightness of the ice 

sculpture. The dragon sculpture is made 

of ice and is the main focus of the image. 

The child's interaction with the sculpture 

suggests a sense of wonder and curiosity.

office worker sitting at 

the table and working 

on the computer .

An office worker sits 

at their desk, 

diligently working on 

their computer.

A man works on his 

computer at a desk with a 

monitor, keyboard , and 

mouse. He is dressed 

casually, wearing glasses, 

and has a slight smile. The 

background is minimalistic. 

The style is a flat illustration 

with a limited color palette.

A man is seated at a desk, engrossed in 

work on his computer. The desk is 

equipped with a monitor, keyboard, and 

mouse. The man is dressed in a casual 

outfit, wearing glasses and has a slight 

smile on his face. The background is 

minimalistic with a light beige color, and 

there's a window that lets in natural light. 

The overall style of the image is a flat 

illustration with a limited color palette, 

giving it a modern and clean look.

Original Image
Pseudo-Manipulation 

Description
LLM’s 

Rewritten Caption

Figure 10: Qualitative results of our intent-CC3M dataset. We leverage DALL-E to generate images
of the captions. We compare intent-CC3M with the CC3M dataset and LLM’s rewritten captions.

16


---Page Break---
B
Algorithm of Manipulation Intention Understanding’s Process.
560

Algorithm 1 outlines the pseudo-code for the manipulation intention understanding process. We
561

create a fixed number of learnable embeddings as latent queries to capture intentions that the user
562

aims to modify within manipulation descriptions. These learnable embeddings are then employed in
563

a Transformer to execute cross-attention with the target descriptions word embedding {ti}m
i=2. The
564

number of output tokens produced by the De-MINDS matches the count of learnable embeddings. To
565

enhance the interaction between learnable embeddings and word embeddings, we concatenate the
566

learnable embeddings with keys and values during the cross-attention process. Each learned query
567

interacts with different intentions, as shown in Figure 2. To achieve a dynamic ratio during the fusion
568

of global and intention embeddings, we utilize a tanh-gating mechanism [23].
569

Table 10: More ablation study on CIRR and FashionIQ.

CIRR
Fashion-IQ

Methods
R1
R5
R10
R10
R50

1.
100% original caption
26.2
55.5
69.5
26.8
49.9
2.
100% rewritten caption
25.8
55.4
69.0
26.5
49.6
3.
100% pseudo-manipulation description
25.3
54.5
68.0
26.9
49.7
4.
50% original, 50% rewritten
26.5
55.9
70.3
27.7
50.9
5.
50% original, 50% pseudo
25.5
55.2
68.6
27.0
50.1
6.
50% rewritten, 50% pseudo
25.9
55.8
69.7
27.4
50.5
7.
40% original , 30% rewritten , 30% pseudo
26.1
55.7
69.2
28.1
50.1
8.
50% original , 25% rewritten , 25% pseudo
26.7
56.5
70.4
29.2
51.4
9.
50% original , 30% rewritten , 20% pseudo
27.3
57.0
71.3
29.7
51.9
10.
w/o align loss
20.6
45.2
57.3
23.6
42.8

C
Further Ablation Studies on the Training Strategy
570

Table 10 details additional ablation analyses of the training strategy in De-MINDS. In model
571

‘1-10’, we evaluate the necessity of constructs T for pre-training Our method supports two
572

scenarios in manipulation intention understanding: integrating intention information from lengthy
573

texts and deducing it from concise texts. We evaluated the utility of the original caption Trew, the
574

rewritten caption Tori, and the pseudo-manipulation description Tint in fostering an understanding of
575

manipulation intentions and ensuring training stability. Our experiments led to the optimal ratio of
576

50% original caption, 30% rewritten caption, and 20% pseudo-manipulation description. Moreover,
577

in model ‘9-10’, we assess the significance of the alignment loss. The absence of alignment
578

between the original image embedding and the target embedding in pre-training results in a notable
579

decrease in average performance by 9.54%. This highlights the crucial role of aligning the original
580

image during training, as in CIR, both the reference image and the manipulation intention together
581

create a comprehensive context that defines the target image.
582

D
More Details of De-MINDS
583

D.1
More Implementation Details For Baseline Models And Mapping Network
584

Generating one intention caption through LLaVA-1.6-13B [32] for the entire Conceptual Caption
585

dataset [47], which comprises 3M images (CC3M) dataset requires approximately 625 hours on 5
586

A100 GPUs. By leveraging the capabilities of LLaVA, we ensure that each text sample within the
587

dataset is enriched with diverse and contextually intent-relevant text rewrites, significantly enhancing
588

the dataset’s utility for composed image retrieval tasks. For training De-MINDS, we utilize the CC3M
589

and adopt ViT-L/14 CLIP [41] pre-trained on 400M image-text paired data. We employ AdamW [34]
590

with a learning rate of 1 × 10−6, weight decay of 0.1, and a linear warmup of 10000 steps. The batch
591

size for contrastive learning is 1024. To improve training stability, we initialize the learnable scalar of
592

tanh-gating to 0 [2]. For training Context-I2W, we only replace the original captions of CC3M with
593

our pseudo-manipulation descriptions. Specifically, we employ AdamW [34] with a learning rate of
594

1 × 10−5, weight decay of 0.1, and a linear warmup of 10000 steps. The batch size for contrastive
595

17


---Page Break---
learning is 1024. For training SEARLE, we utilize the ImageNet1K [16] test set, which comprises
596

100K images, and leverage LLaVA to generate intention captions as detailed in Section 3.2. We
597

employ AdamW, with a learning rate of 5 × 10−5 and a batch size of 256. All models are trained
598

on 4 NVIDIA A100 (80G) GPUs. Moreover, we conduct ablation studies on CIRR test sets and
599

FashionIQ validation sets. For FashionIQ, we consider the average recall. To ensure reliable results,
600

we report the performance averaged over three trials.
601

Mapping network design. Table 11 summarizes the mapping network fθ architecture we employ.
602

Table 11: Pytorch-style[40] model description of the mapping network fθ. The output is fed into the
CLIP language encoder.

Layer
Module
Output
nn.Linear(512, 768)
ReLU2
nn.ReLU
Dropout2
nn.Dropout(0.1)
FC2
nn.Linear(512, 512)
ReLU1
nn.ReLU
Dropout1
nn.Dropout(0.1)
FC1
nn.Linear(512, 512)

D.2
More Evaluation Datasets Details of Query and Candidate Images.
603

We evaluate our model on four ZS-CIR datasets, i.e., COCO [31] for object composition, ImageNet
604

[16, 21] for domain conversion, CIRR [33] for object/scene manipulation, and Fashion-IQ [57] for
605

attribute manipulation. All the dataset settings and evaluation metrics (Recall@K) follow the recent
606

works [45, 52] for a fair comparison. The evaluation datasets are preprocessed, as explained in the
607

main paper, we describe the details of the dataset, i.e., number of query images and candidate images
608

used for evaluation.
609

Table 12: The number of images used for evaluation in each dataset.

Dataset
Query images
Candidate images

ImageNet
10,000
16,983
COCO
4,766
4,766
CIRR (test)
4,148
2,315
Fashion (Dress)
2,017
3,817
Fashion (Shirt)
2,038
6,346
Fashion (TopTee)
1,961
5,373

D.3
More Inference Details of Prompts for Different Evaluate Tasks
610

(1) Domain conversion. This setup evaluates the ability to compose real images and domain infor-
611

mation to retrieve corresponding domain-specific images. We utilize ImageNet [16] and ImageNet-R
612

[21], which comprises 200 classes with diverse domains and has domain annotations. Following
613

Pic2Word, we pick cartoon, origami, toy, and sculpture as the evaluation target to avoid noise in the
614

annotations. With this selection, we have 16,983 images as candidates. In the evaluation, given the
615

real image from ImageNet and target domain names, we compose the query following the procedure
616

in (a) in the Inference section. e.g., a cartoon of [*].
617

(2) Object composition. We evaluate the validation split (5000 images) of COCO [31], which
618

dataset contains images with corresponding lists of object classes and instance mask of query images.
619

Following Pic2Word, we randomly crop one object and mask its background using its instance mask
620

to create a query for each image. The list of object classes is used as text specification. Given the
621

reference image and class list, we compose a query by following (b) in the Inference section. e.g., a
622

photo of [*], [cat] and [dog].
623

(3) Object/scene manipulation by text description. In this setup, a reference image is provided
624

alongside a text description containing instructions for manipulating either an object or the background
625

18


---Page Break---
scene depicted in the reference image. This composition of the reference image and text description
626

enables the retrieval of manipulated images. We evaluate the test split of CIRR [33] using the standard
627

evaluation protocol following previous works [45, 3, 52], and query texts are composed following the
628

procedure in (c) of the Inference section.
629

(4) Attribute manipulation. We employ Fashion-IQ [57], which includes various modification texts
630

related to image attributes. These attribute manipulations are given as a sentence. As with CIRR, we
631

adopt the standard evaluation protocol and create query texts following the procedure provided in
632

(c) of the Inference section. In evaluation, we employ the validation set, following previous works
633

[4, 45, 3, 52].
634

E
Extended Related Works
635

Mapping Image as One Word. Several methods [30, 59] represent image regions as word tokens via
636

VLP models, which rely on object detector efficacy. However, ZR-CIR tasks extend the alignment
637

ability beyond objects to scenes, styles, attributes, ect. Our method addresses this issue by employing
638

pseudo triplet data, which maps a pseudo reference image to a pseudo word token and combines it
639

with the caption to align with the target image. PALAVRA [14] proposes personalized image retrieval
640

via cycle contrastive loss, requiring class-wise and caption annotations. In contrast, our model
641

facilitates fine-grained image-to-word mapping without additional annotations. Other approaches
642

[26, 36, 62, 50] utilize a single word token to represent multiple images of the same object for
643

text-to-image generation. Our model obviates the need for costly image-supervised training.
644

Knowledge Distillation. Knowledge distillation is a machine learning technique wherein a simpler
645

model, known as the student, learns to mimic the behavior of a more complex model, known as
646

the teacher, by learning from its predictions [22]. This approach has demonstrated efficacy across
647

various computer vision tasks, including image classification [22, 43, 5], object detection [9, 8],
648

and text-to-image synthesis [35, 46], resulting in improved model compression, computational
649

efficiency, and accuracy. In our study, we employ knowledge distillation to transfer knowledge from
650

a computationally expensive optimization method (teacher) to a more lightweight neural network
651

(student). Specifically, we train a manipulation intention understanding network to replicate the
652

reasoning ability of an MLLM using a distillation loss. Alternatively, our lightweight network can be
653

interpreted as a surrogate model of the more resource-intensive technique.
654

19


---Page Break---
NeurIPS Paper Checklist
655

1. Claims
656

Question: Do the main claims made in the abstract and introduction accurately reflect the
657

paper’s contributions and scope?
658

Answer: [Yes]
659

Justification: The abstract and introduction are include the claims made in the paper
660

Guidelines:
661

• The answer NA means that the abstract and introduction do not include the claims
662

made in the paper.
663

• The abstract and/or introduction should clearly state the claims made, including the
664

contributions made in the paper and important assumptions and limitations. A No or
665

NA answer to this question will not be perceived well by the reviewers.
666

• The claims made should match theoretical and experimental results, and reflect how
667

much the results can be expected to generalize to other settings.
668

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
669

are not attained by the paper.
670

2. Limitations
671

Question: Does the paper discuss the limitations of the work performed by the authors?
672

Answer: [Yes]
673

Justification: Our paper has limitation in our main paper.
674

Guidelines:
675

• The answer NA means that the paper has no limitation while the answer No means that
676

the paper has limitations, but those are not discussed in the paper.
677

• The authors are encouraged to create a separate "Limitations" section in their paper.
678

• The paper should point out any strong assumptions and how robust the results are to
679

violations of these assumptions (e.g., independence assumptions, noiseless settings,
680

model well-specification, asymptotic approximations only holding locally). The authors
681

should reflect on how these assumptions might be violated in practice and what the
682

implications would be.
683

• The authors should reflect on the scope of the claims made, e.g., if the approach was
684

only tested on a few datasets or with a few runs. In general, empirical results often
685

depend on implicit assumptions, which should be articulated.
686

• The authors should reflect on the factors that influence the performance of the approach.
687

For example, a facial recognition algorithm may perform poorly when image resolution
688

is low or images are taken in low lighting. Or a speech-to-text system might not be
689

used reliably to provide closed captions for online lectures because it fails to handle
690

technical jargon.
691

• The authors should discuss the computational efficiency of the proposed algorithms
692

and how they scale with dataset size.
693

• If applicable, the authors should discuss possible limitations of their approach to
694

address problems of privacy and fairness.
695

• While the authors might fear that complete honesty about limitations might be used by
696

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
697

limitations that aren’t acknowledged in the paper. The authors should use their best
698

judgment and recognize that individual actions in favor of transparency play an impor-
699

tant role in developing norms that preserve the integrity of the community. Reviewers
700

will be specifically instructed to not penalize honesty concerning limitations.
701

3. Theory Assumptions and Proofs
702

Question: For each theoretical result, does the paper provide the full set of assumptions and
703

a complete (and correct) proof?
704

Answer: [Yes]
705

20


---Page Break---
Justification: All the formulas in the paper be numbered and cross-referenced
706

Guidelines:
707

• The answer NA means that the paper does not include theoretical results.
708

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
709

referenced.
710

• All assumptions should be clearly stated or referenced in the statement of any theorems.
711

• The proofs can either appear in the main paper or the supplemental material, but if
712

they appear in the supplemental material, the authors are encouraged to provide a short
713

proof sketch to provide intuition.
714

• Inversely, any informal proof provided in the core of the paper should be complemented
715

by formal proofs provided in appendix or supplemental material.
716

• Theorems and Lemmas that the proof relies upon should be properly referenced.
717

4. Experimental Result Reproducibility
718

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
719

perimental results of the paper to the extent that it affects the main claims and/or conclusions
720

of the paper (regardless of whether the code and data are provided or not)?
721

Answer: [Yes]
722

Justification: The code and sample dataset are provided in our supplementary. We describe
723

the steps taken to make the results reproducible or verifiable.
724

Guidelines:
725

• The answer NA means that the paper does not include experiments.
726

• If the paper includes experiments, a No answer to this question will not be perceived
727

well by the reviewers: Making the paper reproducible is important, regardless of
728

whether the code and data are provided or not.
729

• If the contribution is a dataset and/or model, the authors should describe the steps taken
730

to make their results reproducible or verifiable.
731

• Depending on the contribution, reproducibility can be accomplished in various ways.
732

For example, if the contribution is a novel architecture, describing the architecture fully
733

might suffice, or if the contribution is a specific model and empirical evaluation, it may
734

be necessary to either make it possible for others to replicate the model with the same
735

dataset, or provide access to the model. In general. releasing code and data is often
736

one good way to accomplish this, but reproducibility can also be provided via detailed
737

instructions for how to replicate the results, access to a hosted model (e.g., in the case
738

of a large language model), releasing of a model checkpoint, or other means that are
739

appropriate to the research performed.
740

• While NeurIPS does not require releasing code, the conference does require all submis-
741

sions to provide some reasonable avenue for reproducibility, which may depend on the
742

nature of the contribution. For example
743

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
744

to reproduce that algorithm.
745

(b) If the contribution is primarily a new model architecture, the paper should describe
746

the architecture clearly and fully.
747

(c) If the contribution is a new model (e.g., a large language model), then there should
748

either be a way to access this model for reproducing the results or a way to reproduce
749

the model (e.g., with an open-source dataset or instructions for how to construct
750

the dataset).
751

(d) We recognize that reproducibility may be tricky in some cases, in which case
752

authors are welcome to describe the particular way they provide for reproducibility.
753

In the case of closed-source models, it may be that access to the model is limited in
754

some way (e.g., to registered users), but it should be possible for other researchers
755

to have some path to reproducing or verifying the results.
756

5. Open access to data and code
757

Question: Does the paper provide open access to the data and code, with sufficient instruc-
758

tions to faithfully reproduce the main experimental results, as described in supplemental
759

material?
760

21


---Page Break---
Answer: [Yes]
761

Justification: Our paper provides open access to the code for creating the dataset and
762

reproducing the main experimental results. We will provide the entire dataset after our paper
763

is accepted.
764

Guidelines:
765

• The answer NA means that paper does not include experiments requiring code.
766

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
767

public/guides/CodeSubmissionPolicy) for more details.
768

• While we encourage the release of code and data, we understand that this might not be
769

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
770

including code, unless this is central to the contribution (e.g., for a new open-source
771

benchmark).
772

• The instructions should contain the exact command and environment needed to run to
773

reproduce the results. See the NeurIPS code and data submission guidelines (https:
774

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
775

• The authors should provide instructions on data access and preparation, including how
776

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
777

• The authors should provide scripts to reproduce all experimental results for the new
778

proposed method and baselines. If only a subset of experiments are reproducible, they
779

should state which ones are omitted from the script and why.
780

• At submission time, to preserve anonymity, the authors should release anonymized
781

versions (if applicable).
782

• Providing as much information as possible in supplemental material (appended to the
783

paper) is recommended, but including URLs to data and code is permitted.
784

6. Experimental Setting/Details
785

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
786

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
787

results?
788

Answer: [Yes]
789

Justification: Our paper specifies all the training and test details in the main paper and
790

appendix. We also provide the pseudo-code for our method in our appendix.
791

Guidelines:
792

• The answer NA means that the paper does not include experiments.
793

• The experimental setting should be presented in the core of the paper to a level of detail
794

that is necessary to appreciate the results and make sense of them.
795

• The full details can be provided either with the code, in appendix, or as supplemental
796

material.
797

7. Experiment Statistical Significance
798

Question: Does the paper report error bars suitably and correctly defined or other appropriate
799

information about the statistical significance of the experiments?
800

Answer: [No]
801

Justification: Error bars are not reported because it would be too computationally expensive
802

for four datasets.
803

Guidelines:
804

• The answer NA means that the paper does not include experiments.
805

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
806

dence intervals, or statistical significance tests, at least for the experiments that support
807

the main claims of the paper.
808

• The factors of variability that the error bars are capturing should be clearly stated (for
809

example, train/test split, initialization, random drawing of some parameter, or overall
810

run with given experimental conditions).
811

22


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
812

call to a library function, bootstrap, etc.)
813

• The assumptions made should be given (e.g., Normally distributed errors).
814

• It should be clear whether the error bar is the standard deviation or the standard error
815

of the mean.
816

• It is OK to report 1-sigma error bars, but one should state it. The authors should
817

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
818

of Normality of errors is not verified.
819

• For asymmetric distributions, the authors should be careful not to show in tables or
820

figures symmetric error bars that would yield results that are out of range (e.g. negative
821

error rates).
822

• If error bars are reported in tables or plots, The authors should explain in the text how
823

they were calculated and reference the corresponding figures or tables in the text.
824

8. Experiments Compute Resources
825

Question: For each experiment, does the paper provide sufficient information on the com-
826

puter resources (type of compute workers, memory, time of execution) needed to reproduce
827

the experiments?
828

Answer: [Yes]
829

Justification: We indicate the type of compute workers and compute time for dataset
830

generation and training.
831

Guidelines:
832

• The answer NA means that the paper does not include experiments.
833

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
834

or cloud provider, including relevant memory and storage.
835

• The paper should provide the amount of compute required for each of the individual
836

experimental runs as well as estimate the total compute.
837

• The paper should disclose whether the full research project required more compute
838

than the experiments reported in the paper (e.g., preliminary or failed experiments that
839

didn’t make it into the paper).
840

9. Code Of Ethics
841

Question: Does the research conducted in the paper conform, in every respect, with the
842

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
843

Answer: [Yes]
844

Justification: We have reviewed the NeurIPS Code of Ethics.
845

Guidelines:
846

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
847

• If the authors answer No, they should explain the special circumstances that require a
848

deviation from the Code of Ethics.
849

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
850

eration due to laws or regulations in their jurisdiction).
851

10. Broader Impacts
852

Question: Does the paper discuss both potential positive societal impacts and negative
853

societal impacts of the work performed?
854

Answer: [Yes]
855

Justification: We discuss both potential positive societal impacts and negative societal
856

impacts of the work performed in our appendix.
857

Guidelines:
858

• The answer NA means that there is no societal impact of the work performed.
859

• If the authors answer NA or No, they should explain why their work has no societal
860

impact or why the paper does not address societal impact.
861

23


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
862

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
863

(e.g., deployment of technologies that could make decisions that unfairly impact specific
864

groups), privacy considerations, and security considerations.
865

• The conference expects that many papers will be foundational research and not tied
866

to particular applications, let alone deployments. However, if there is a direct path to
867

any negative applications, the authors should point it out. For example, it is legitimate
868

to point out that an improvement in the quality of generative models could be used to
869

generate deepfakes for disinformation. On the other hand, it is not needed to point out
870

that a generic algorithm for optimizing neural networks could enable people to train
871

models that generate Deepfakes faster.
872

• The authors should consider possible harms that could arise when the technology is
873

being used as intended and functioning correctly, harms that could arise when the
874

technology is being used as intended but gives incorrect results, and harms following
875

from (intentional or unintentional) misuse of the technology.
876

• If there are negative societal impacts, the authors could also discuss possible mitigation
877

strategies (e.g., gated release of models, providing defenses in addition to attacks,
878

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
879

feedback over time, improving the efficiency and accessibility of ML).
880

11. Safeguards
881

Question: Does the paper describe safeguards that have been put in place for responsible
882

release of data or models that have a high risk for misuse (e.g., pretrained language models,
883

image generators, or scraped datasets)?
884

Answer: [No]
885

Justification: our paper poses no such risks.
886

Guidelines:
887

• The answer NA means that the paper poses no such risks.
888

• Released models that have a high risk for misuse or dual-use should be released with
889

necessary safeguards to allow for controlled use of the model, for example by requiring
890

that users adhere to usage guidelines or restrictions to access the model or implementing
891

safety filters.
892

• Datasets that have been scraped from the Internet could pose safety risks. The authors
893

should describe how they avoided releasing unsafe images.
894

• We recognize that providing effective safeguards is challenging, and many papers do
895

not require this, but we encourage authors to take this into account and make a best
896

faith effort.
897

12. Licenses for existing assets
898

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
899

the paper, properly credited and are the license and terms of use explicitly mentioned and
900

properly respected?
901

Answer: [Yes]
902

Justification: the creators or original owners of assets are the license and terms of use
903

explicitly mentioned and properly respected.
904

Guidelines:
905

• The answer NA means that the paper does not use existing assets.
906

• The authors should cite the original paper that produced the code package or dataset.
907

• The authors should state which version of the asset is used and, if possible, include a
908

URL.
909

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
910

• For scraped data from a particular source (e.g., website), the copyright and terms of
911

service of that source should be provided.
912

24


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the
913

package should be provided. For popular datasets, paperswithcode.com/datasets
914

has curated licenses for some datasets. Their licensing guide can help determine the
915

license of a dataset.
916

• For existing datasets that are re-packaged, both the original license and the license of
917

the derived asset (if it has changed) should be provided.
918

• If this information is not available online, the authors are encouraged to reach out to
919

the asset’s creators.
920

13. New Assets
921

Question: Are new assets introduced in the paper well documented and is the documentation
922

provided alongside the assets?
923

Answer: [No]
924

Justification: Our paper does not release new assets.
925

Guidelines:
926

• The answer NA means that the paper does not release new assets.
927

• Researchers should communicate the details of the dataset/code/model as part of their
928

submissions via structured templates. This includes details about training, license,
929

limitations, etc.
930

• The paper should discuss whether and how consent was obtained from people whose
931

asset is used.
932

• At submission time, remember to anonymize your assets (if applicable). You can either
933

create an anonymized URL or include an anonymized zip file.
934

14. Crowdsourcing and Research with Human Subjects
935

Question: For crowdsourcing experiments and research with human subjects, does the paper
936

include the full text of instructions given to participants and screenshots, if applicable, as
937

well as details about compensation (if any)?
938

Answer: [No]
939

Justification: Our paper does not involve crowdsourcing nor research with human subjects.
940

Guidelines:
941

• The answer NA means that the paper does not involve crowdsourcing nor research with
942

human subjects.
943

• Including this information in the supplemental material is fine, but if the main contribu-
944

tion of the paper involves human subjects, then as much detail as possible should be
945

included in the main paper.
946

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
947

or other labor should be paid at least the minimum wage in the country of the data
948

collector.
949

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
950

Subjects
951

Question: Does the paper describe potential risks incurred by study participants, whether
952

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
953

approvals (or an equivalent approval/review based on the requirements of your country or
954

institution) were obtained?
955

Answer: [No]
956

Justification: Our paper does not involve crowdsourcing nor research with human subjects
957

Guidelines:
958

• The answer NA means that the paper does not involve crowdsourcing nor research with
959

human subjects.
960

• Depending on the country in which research is conducted, IRB approval (or equivalent)
961

may be required for any human subjects research. If you obtained IRB approval, you
962

should clearly state this in the paper.
963

25


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
964

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
965

guidelines for their institution.
966

• For initial submissions, do not include any information that would break anonymity (if
967

applicable), such as the institution conducting the review.
968

26


---Page Break---
