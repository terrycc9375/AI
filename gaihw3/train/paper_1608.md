UniEdit: A Unified Tuning-Free Framework for
Video Motion and Appearance Editing

Anonymous Author(s)
Affiliation
Address
email

Project webpage: https://uni-edit.github.io/UniEdit/
1

Figure 1: Examples edited by UniEdit. Our solution supports both video motion editing in the time
axis (i.e., from playing guitar to eating or waving) and various video appearance editing scenarios
(i.e., stylization, rigid/non-rigid object replacement, background modification). We encourage the
readers to watch the videos on our project page.

Abstract

Recent advances in text-guided video editing have showcased promising results
2

in appearance editing (e.g., stylization). However, video motion editing in the
3

temporal dimension (e.g., from eating to waving), which distinguishes video edit-
4

ing from image editing, is underexplored. In this work, we present UniEdit, a
5

tuning-free framework that supports both video motion and appearance editing by
6

harnessing the power of a pre-trained text-to-video generator within an inversion-
7

then-generation framework. To realize motion editing while preserving source
8

video content, based on the insights that temporal and spatial self-attention layers
9

encode inter-frame and intra-frame dependency respectively, we introduce auxiliary
10

motion-reference and reconstruction branches to produce text-guided motion and
11

source features respectively. The obtained features are then injected into the main
12

editing path via temporal and spatial self-attention layers. Extensive experiments
13

demonstrate that UniEdit covers video motion editing and various appearance
14

editing scenarios, and surpasses the state-of-the-art methods. Our code will be
15

publicly available.
16

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
1
Introduction
17

The advent of pre-trained diffusion-based [26, 53] text-to-image generators [49, 50, 48] has revo-
18

lutionized the fields of design and filmmaking, opening new vistas for creative expression. These
19

advancements, underpinned by seminal works in text-to-image synthesis, have paved the way for inno-
20

vative text-guided editing techniques for both images [42, 24, 4, 5] and videos [65, 6, 39, 70, 17, 46].
21

Such techniques not only enhance creative workflows but also promise to redefine content creation
22

within these industries.
23

Video editing, in contrast to image editing, introduces the intricate challenge of ensuring frame-wise
24

consistency. Efforts to address this challenge have led to the development of methods that leverage
25

shared features and structures with the source video [6, 39, 37, 70, 46, 7, 33, 62, 18] through an
26

inversion-then-generation pipeline [42, 53], exemplified by Pix2Video’s approach [6] to consistent
27

appearance editing across frames. To transfer the edited appearance from the anchor frame to the
28

remaining frames consistently, it employs a pre-trained image generator and extends the self-attention
29

layers to cross-frame attention to generate each remaining frame. Despite these advancements in
30

performing video appearance editing (e.g., stylization, object appearance replacement, etc.), these
31

methodologies fall short in editing video motion (e.g., replacing the movement of playing guitar with
32

waving), hampered by a lack of motion priors and limited control over inter-frame dependencies,
33

underscoring a critical gap in video editing capabilities.
34

Previous attempts [65, 44] at video motion editing through fine-tuning a pre-trained generator on
35

the given source video and then editing motion through text guidance. Although effective, they
36

necessitate a delicate balance between the generative prowess of the model and the preservation of
37

the source video’s content. This compromise often leads to restricted motion diversity and unwanted
38

content variations, indicating a pressing need for a more robust solution.
39

In response, our work aims to explore a tuning-free framework that adeptly navigates the complexities
40

of editing both the motion and appearance of videos. To achieve this, we identify three technical
41

challenges: 1) it is non-trivial to incorporate the text-guided motion into the source content, as directly
42

applying video appearance editing [46, 18] or image editing [5] schemes leads to undesirable results
43

(as shown in Fig. 5); 2) preserving the non-edited content of the source video; 3) inheriting the spatial
44

structure of the source video during appearance editing.
45

Our solution, UniEdit, harnesses the power of a pre-trained text-to-video generator (e.g., LaVie [63])
46

within an inversion-then-generation framework [42], tailored to overcome the identified challenges.
47

Particularly, we introduce three key innovations: 1) To inject text-guided motion into the source
48

content, we highlight the insight that the temporal self-attention layers of the generator encode
49

the inter-frame dependency. Acting in this way, we introduce an auxiliary motion-reference branch
50

to generate text-guided motion features, which are then injected into the main editing path via
51

temporal self-attention layers. 2) To preserve the non-edited content of the source video, motivated
52

by the image editing technique [5], we follow the insight that the spatial self-attention layers of the
53

generator encode the intra-frame dependency. Therefore, we introduce an auxiliary reconstruction
54

branch, and inject the features obtained from the spatial self-attention layers of the reconstruction
55

branch into the main editing path. 3) To retain the spatial structure during the appearance editing, we
56

replace the spatial attention maps of the main editing path with those in the reconstruction branch.
57

To our best knowledge, UniEdit represents a pioneering leap in text-guided, tuning-free video
58

motion editing. In addition, its unified architecture not only facilitates a wide array of video
59

appearance editing tasks, as shown in Fig. 1, but also empowers image-to-video generators for
60

zero-shot text-image-to-video generation. Through comprehensive experimentation, we demonstrate
61

UniEdit’s superior performance relative to existing state-of-the-art methods, highlighting its potential
62

to significantly advance the field of video editing.
63

2
Related Works
64

2.1
Video Generation
65

Researchers have achieved video generation with generative adversarial networks [58, 51, 61],
66

language models [69, 71], or diffusion models [28, 52, 25, 23, 3, 60, 72, 19, 63, 8, 47]. To make the
67

generation more controllable, recent endeavors have also incorporated additional structure guidance
68

(e.g., depth map) [16, 10, 74, 11, 20, 64], or conducted customized generation [65, 67, 34, 75, 59, 41].
69

2


---Page Break---
These models have generally learned real-world video distribution from large-scale data, and achieved
70

promising results on text-to-video or image-to-video generation. Based on their success, we leverage
71

the learned prior in the pre-trained model to achieve tuning-free video motion and appearance editing.
72

2.2
Video Editing
73

Video editing aims to produce a new video that is aligned with the provided editing instructions
74

(e.g., text) while maintaining the other characteristics of the source video. It can be categorized into
75

appearance and motion editing.
76

For appearance editing [70, 15, 17, 35, 12], like turn a video into the style of Van Gogh, the main
77

challenge is to achieve temporal-consistent generation across different frames. Early attempts [6, 37,
78

46, 7, 33, 62] leveraged text-to-image models with inter-frame propagation to ensure consistency.
79

For instance, Pix2Video [6] replaces the key and value of the current frame with those of the
80

first and previous frame. Video-P2P [39] achieved local editing via video-specific fine-tuning and
81

unconditional embedding optimization [43]. Follow-up studies [18, 70, 45] also leveraged the edit-
82

then-propagate framework with neatest-neighbor field [18], estimated optical flow [70], or temporal
83

deformation field [45]. Despite the promising results, due to the constraint on the source video
84

structure, these approaches are specialized in appearance editing and can not be applied to motion
85

editing directly.
86

Recent studies have also explored video motion editing with text guidance [65, 44], user-provided
87

motion [32, 54, 15], or specific motion representation [55, 36, 22]. For example, Dreamix [44]
88

proposed fine-tuning a pre-trained text-to-video model with mixed video-image reconstruction
89

objectives for each source video. Then the editing is realized by conditioning the fine-tuned model
90

on the given target prompt. MoCA [68] decoupled the video into the first-frame appearance and
91

the optical flow, and trained a diffusion model to generate video conditioned on the first frame and
92

the text. However, it struggled to preserve the non-edited motion (e.g., background dynamics) as
93

it generates the entire motion from the text. Different from the aforementioned approaches that
94

require fine-tuning or user-provided motion input, we are the first to achieve tuning-free motion and
95

appearance editing with text guidance only.
96

3
Preliminaries: Video Diffusion Models
97

Our proposed UniEdit is built upon video diffusion models. Therefore, we first recap the architecture
98

that is used in common text-guided video diffusion models [63, 2].
99

Overall Architecture
Modern text-to-video (T2V) diffusion models typically extend a pre-trained
100

text-to-image (T2I) model [49] to the video domain with the following adaptations. 1) Introducing
101

additional temporal layers by inflating 2d convolutional layers to 3d form, or adding temporal
102

self-attention layers [57] to model the correlation between video frames. 2) Due to the extensive
103

computational resources for modeling spatial-temporal joint distribution, these works typically
104

first train video generation models on low spatial and temporal resolutions, and then upsampling
105

the generated results with cascaded models. 3) Other improvements like efficiency [1], training
106

strategy [19], or additional control signals [16], etc. During inference, given standard Gaussian
107

distribution zT ∼N(0, 1), the denoising UNet is used to perform T denoising steps to obtain the
108

outputs [26, 53]. If the model is trained in latent space [49], a decoder is employed to reconstruct
109

videos from the latent domain.
110

Attention Mechanisms
In particular, for each block of the denoising UNet, there are four basic
111

modules: a convolutional module, a spatial self-attention module (SA-S), a spatial cross-attention
112

module (CA-S), and a temporal self-attention module (SA-T). Formally, the attention operation [57]
113

can be formulated as:
114

attn(Q, K, V ) = softmax(QKT

√

d
)V,
(1)

where Q (query), K (key), V (value) are derived from inputs, and d is the dimension of hidden states.
115

Intuitively, CA-S is in charge of fusing semantics from the text condition, SA-S models the intra-
116

frame dependency, SA-T models the inter-frame dependency and ensures the generated results are
117

temporally consistent. We leverage these intuitions in our designs as elaborated below.
118

3


---Page Break---
Figure 2: Overview of UniEdit. It follows an inversion-then-generation pipeline and consists of a
main editing path, an auxiliary reconstruction branch and an auxiliary motion-reference branch. The
reconstruction branch produces source features for content preservation, and the motion-reference
branch yields text-guided motion features for motion injection. The source features and motion
features are injected into the main editing path through spatial self-attention (SA-S) and temporal
self-attention (SA-T) modules respectively (Sec. 4.1). We further introduce spatial structure control
to retain the coarse structure of the source video (Sec. 4.2).

4
UniEdit
119

Method Overview.
As shown in Fig. 2, our main editing path is based on an inversion-then-
120

generation pipeline: we use the latent after DDIM inversion [53] as the initial noise zT 1, then perform
121

denoising process starting from zT with the pre-trained UNet conditioned on the target prompt Pt. For
122

motion editing, to achieve source content preservation and motion control, we propose to incorporate
123

an auxiliary reconstruction branch and an auxiliary motion-reference branch to provide desired source
124

and motion features, which are injected into the main editing path to achieve content preservation and
125

motion editing (as shown in Fig. 3). We propose the pipeline of motion editing and appearance editing
126

in Sec. 4.1 & Sec. 4.2 respectively. To further alleviate the background inconsistency, we introduce
127

a mask-guided coordination scheme in Sec. 4.3. We also extend UniEdit to text-image-to-video
128

generation (TI2V) in Sec. 4.4.
129

4.1
Tuning-Free Video Motion Editing
130

Content Preservation on SA-S Modules.
One of the key challenges of editing tasks is to inherit
131

the original content (e.g., textures and background) in the source video. To this end, we introduce
132

an auxiliary reconstruction branch. The reconstruction path starts from the same inversed latent
133

zT similar to the main editing path, and then conducts the denoising process with the pre-trained
134

UNet conditioned on the source prompt Ps to reconstruct the original frames. As verified in image
135

editing [56, 24, 5], the attention features in the denoising model during reconstruction contain the
136

content of the source video. Hence, we inject attention features of the reconstruction path into the
137

main editing path on spatial self-attention (SA-S) layers for content preservation. At denoising step t,
138

the attention operation of the l-th SA-S module in the main editing path is formulated as:
139

SA-Sl
edit :=
attn(Q, K, V r),
t < t0 and l > L,
attn(Q, K, V ),
otherwise,
(2)

where Q, K, V are features in the main editing path, V r refer to the value feature of the corresponding
140

SA-S layer in the reconstruction branch, t0 = 50 and L = 10 are hyper-parameters following previous
141

work [5]. By replacing the value of spatial features, the video synthesized by the main editing path
142

retains the non-edited characters (e.g., identity and background) of the source video, as exhibited
143

in Fig. 7a. Unlike previous video editing works [37, 29] which introduces a cross-frame attention
144

mechanism (i.e., using the key and value of the first/last frame), we implement Eq. 2 frame-wisely to
145

better tackle source video with large dynamics.
146

1For real source video, we set source prompt to null during both forward and inversion process to achieve
high-quality reconstruction [43].

4


---Page Break---
Motion Injection on SA-T Modules.
After implementing the content-preserving technique intro-
147

duced above, we can obtain an edited video with the same content in the source video. However, it
148

is observed that the output video could not follow the text prompt Pt properly. A straightforward
149

solution is to increase the value of L so that balancing between the impact of injected information and
150

the conditioned text prompt. Nevertheless, this could result in a content mismatch with the original
151

source video in terms of structures and textures.
152

To obtain the desired motion without sacrificing content consistency, we propose to guide the main
153

editing path with reference motion. Concretely, an auxiliary motion-reference branch (which also
154

starts from the inversed latent zT ) is involved during the denoising process. Different from the
155

reconstruction branch, the motion-reference branch is conditioned on the target prompt Pt, which
156

contains the description of the desired motion. To transfer the motion into the main editing path, our
157

core insight here is that temporal layers model the inter-frame dependency of the synthesized video
158

clip (as shown in Fig. 6). Motivated by the observations above, we design the attention map injection
159

on temporal self-attention layers of the main editing path:
160

SA-Tl
edit := attn(Qm, Km, V )
(3)

where Qm and Km refer to the query and key of the motion-reference branch, note that we replace
161

the query and key of SA-T modules in the main editing path with those in the motion-reference
162

branch on all layers and denoising steps. It’s observed that the injection of temporal attention maps
163

can effectively facilitate the main editing path to generate motion aligned with the target prompt.
164

To better fuse the motion with the content in the source video, we also implement spatial structure
165

control (refer to Sec. 4.2) on the main editing path and motion-reference branch in the early steps.
166

4.2
Tuning-Free Video Appearance Editing
167

Figure 3: Detailed illustration of the relation-
ship between the main editing path, the auxiliary
reconstruction branch and the auxiliary motion-
reference branch. The content preservation, motion
injection and spatial structure control are achieved
by the fusion of Q (query), K (key), V (value) fea-
tures in spatial self-attention (SA-S) and temporal
self-attention (SA-T) modules.

In Sec. 4.1, we introduce the pipeline of UniEdit
168

for video motion editing. In this subsection,
169

we aim to perform appearance editing (e.g.,
170

style transfer, object replacement, background
171

changing) via the same framework. In general,
172

there are two main differences between appear-
173

ance editing and motion editing. Firstly, ap-
174

pearance editing does not require changing the
175

inter-frame relationships. Therefore, we remove
176

the motion-reference branch and corresponding
177

motion injection mechanism from the motion
178

editing pipeline. Secondly, the main challenge
179

of appearance editing is to maintain the struc-
180

tural consistency of the source video. To address
181

this, we introduce spatial structure control be-
182

tween the main editing path and the reconstruc-
183

tion branch.
184

Spatial Structure Control on SA-S Modules.
185

Previous approaches on video appearance editing [70, 18] mainly realize spatial structure control
186

with the assistance of additional network [73]. When the auxiliary control model fails, it may result
187

in inferior performance in preserving the structure of the original video. Alternatively, we suggest
188

extracting the layout information of the source video from the reconstruction branch. Intuitively,
189

the attention maps in spatial self-attention layers encode the structure of the synthesized video, as
190

verified in Fig. 6. Hence, we replace the query and key of SA-S module in the main editing path with
191

those in the reconstruction branch:
192

SA-Sl
edit :=
attn(Qr, Kr, V ),
t < t1,
attn(Q, K, V ),
otherwise,
(4)

where Qr and Kr refer to the query and key of the reconstruction branch, t1 is used to control the
193

extent of editing. It is worth mentioning that the effect of spatial structure control is distinct from the
194

content preservation mechanism in Sec. 4.1. Take stylization as an example, the proposed structure
195

control in Eq. 4 only ensures consistency in terms of each frame’s composition, while enabling the
196

model to generate the required textures and styles based on the text prompt. On the other hand,
197

5


---Page Break---
the content preservation technique inherits the textures and style of the source video. Therefore,
198

we use structure control instead of content preservation for appearance editing. In addition, using
199

the proposed structure control technique in motion editing can make the layout of the output video
200

similar to the source video (shown in Fig. 11b in Appendix). Users have the flexibility to adjust the
201

consistency between the edited video and the source video layout based on their specific requirements.
202

4.3
Mask-Guided Coordination (Optional)
203

To further improve the editing performance, we suggest leveraging the foreground/background
204

segmentation mask M to guide the denoising process [14, 13]. There are two possible ways to obtain
205

the mask M: the attention maps of CA-S modules with a threshold [24]; or employing an off-the-shelf
206

segmentation model [38] on the source and generated videos. The obtained segmentation masks can
207

be leveraged to 1), alleviate the indistinction in foreground and background; 2), improve content
208

consistency between edited and source videos. To this end, we leverage mask-guided self-attention in
209

the main editing path to coordinate the editing process. Formally, we define:
210

m-attn(Q, K, V ; M) = softmax(QKT

√

d
+ M)V.
(5)

Then the mask-guided self-attention:
211

SAmask := m-attn(Q, K, V ; M f) ⊙Mm + m-attn(Q, K, V ; M b) ⊙(1 −Mm),
(6)

where M f, M b ∈{−∞, 0} indicate the foreground and background masks in the editing path
212

respectively, Mm ∈{0, 1} denotes the foreground mask from the motion-reference branch, and ⊙is
213

Hadamard product. In addition, we leverage the mask during the content preservation and motion
214

injection for the features obtained from the reconstruction branch and the motion-reference branch
215

(e.g., we replace Qm with Mm ⊙Qm + (1 −Mm) ⊙Q).
216

4.4
T2V Models are Zero-Shot TI2V Generators
217

To make our framework more flexible, we further derive a method to incorporate images as input
218

and synthesize high-quality video conditioned on both image and text-prompt. Different from some
219

image animation techniques [2], our method allows the user to guide the animation process with text
220

prompts. Concretely, we first achieve image-to-video (I2V) generation by: 1) transforming input
221

images with simulated camera movement to form a pseudo-video clip [44] or 2) leveraging existing
222

image animation approaches (e.g., SVD [2], AnimateDiff [21]) to synthesis a video with random
223

motion (which may not consistent with the text prompt). Then, we perform text-guided editing with
224

UniEdit on the vanilla video to obtain the final output video.
225

5
Experiments
226

5.1
Comparison with State-of-the-Art Methods
227

Implementation Details
UniEdit is not limited to specific video diffusion models. In this section,
228

we build UniEdit upon LaVie [63] as an instantiation to verify the effectiveness of our method. To
229

demonstrate the flexibility of UniEdit across different base models, we also implement the proposed
230

method on VideoCrafter2 [9] and exhibit the editing results in Appendix B.1. For each input video,
231

we follow the pre-processing step in LaVie to the resolution of 320 × 512. Then, the pre-processed
232

video is fed into the UniEdit to perform video editing. It takes 1-2 minutes to edit on an NVIDIA
233

A100 GPU for each video. More details can be found in Appendix A.
234

Baselines.
To evaluate the performance of UniEdit, we compare the editing results of UniEdit
235

with state-of-the-art motion and appearance editing approaches. For motion editing, due to the
236

lack of open-source tuning-free (zero-shot) methods, we adapt the state-of-the-art non-rigid image
237

editing technique MasaCtrl [5] to a T2V model [63] (denoted as MasaCtrl∗in Fig. 5) and a one-shot
238

video editing method Tune-A-Video (TAV) [65] as strong baselines. For appearance editing, we
239

use the latest methods with strong performance, including FateZero [46], TokenFlow [18], and
240

Rerender-A-Video (Rerender) [70] as baselines.
241

Evaluation Set.
The evaluation set consists of 100 samples, including: a) 20 randomly sampled
242

video clips from the open-source LOVEU-TGVE-2023 [66] dataset, along with their corresponding
243

80 text prompts, and b) 20 videos from online sources (www.pexels.com and www.pixabay.com),
244

with manually designed prompts, as the baseline methods do not have an open-source evaluation set.
245

6


---Page Break---
Figure 4: Examples edited by UniEdit. For each case, the upper frames come from the source video,
and the lower frames indicate the edited results with the target prompt. We encourage the readers to
watch the videos and make evaluations.

Qualitative Results.
We present editing examples of UniEdit in Fig. 1, Fig. 4 (additional examples
246

in Fig. 16-21 of Appendix B.8). Please visit our project page for more videos. UniEdit demonstrates
247

the ability to: 1) edit in various scenarios, including motion-changing, object replacement, style
248

transfer, and background modification; 2) align with the target prompt; and 3) maintain excellent
249

temporal consistency. Additionally, we compare UniEdit with state-of-the-art methods in Fig. 5
250

(further comparisons in Fig.13,14,15 of Appendix B.7). For a fair comparison, we also migrated
251

all baselines to LaVie [63], using the same base model as our method. The results are presented
252

in Fig. 15. For appearance editing, we showcase two scenarios: non-rigid object replacement and
253

stylization. In object replacement, our method outperforms baselines in terms of prompt alignment
254

and background consistency. In stylization, UniEdit excels in preserving content. For example, the
255

grassland retains its original appearance without any additional elements. In motion editing, UniEdit
256

surpasses baselines in aligning the video with the target prompt and preserving the source content.
257

Quantitative Results.
We quantitatively evaluate our method using two approaches: 1) CLIP
258

scores and user preference, as employed in previous work [65]; and 2) VBench [31] scores, a recently
259

proposed benchmark suite for T2V models. The summarized results are in Tab. 1. Following previous
260

work [65], we assess the effectiveness of our method in terms of temporal consistency and alignment
261

with the target prompt. Additionally, we conducted a user study involving 10 participants who rated
262

the edited videos on a scale of 1 to 5. We also utilize the recently proposed VBench [31] benchmark
263

to provide a more comprehensive assessment, which includes ‘Frame Quality’ metrics and ‘Temporal
264

Quality’ metrics. UniEdit outperforms the baseline methods across all metrics. Furthermore, the
265

mask-guided coordination technique introduced in Sec. 4.3 further enhances performance (see
266

Appendix B.3). For more detailed quantitative results, please refer to Appendix B.2&B.3&B.5.
267

5.2
Ablation Study and Analysis
268

How UniEdit Works?
To better understand how UniEdit works and reveal our insight on the
269

spatial and temporal self-attention layers, we visualize the features in the SA-S and SA-T modules
270

and compare them with the magnitude of optical flow between adjacent frames in Fig. 6a. It is evident
271

that, in comparison to the spatial query maps (2nd row), the temporal cross-frame attention maps (3rd
272

row) exhibit a notably higher degree of overlap with the optical flow (4th row). This indicates that the
273

temporal self-attention layers encode inter-frame dependencies and facilitate motion injection, while
274

content preservation and structure control are carried out in the spatial self-attention layers.
275

7


---Page Break---
Figure 5: Comparison with state-of-the-art methods for both video appearance and motion editing. It
shows that UniEdit achieves better source content preservation, and outperforms baselines in motion
editing by a large margin.

Table 1: Quantitative comparison with state-of-the-art video editing techniques.

Method

Frame Consistency Textual Alignment
Frame Quality
Temporal Quality

CLIP
Score
User
Pref.

CLIP
Score
User
Pref.

Aesthetic
Quality
Imaging
Quality

Subject
Consistency
Motion
Smoothness
Temporal
Flickering

TAV [65]
95.39
3.74
27.89
3.30
51.97
49.60
93.10
93.27
91.48
MasaCtrl∗[5]
97.61
4.31
25.58
3.17
54.58
58.72
93.04
95.70
94.29
FateZero [46]
96.72
4.48
27.30
3.48
53.77
56.99
93.55
94.80
93.42
Rerender [70]
97.18
4.16
27.94
3.55
54.59
57.97
93.08
95.57
94.36
TokenFlow[18]
97.02
4.50
28.58
3.34
52.60
60.65
91.97
95.04
93.50

UniEdit
98.35
4.72
31.43
4.79
58.25
62.94
95.73
97.30
96.74
UniEdit-Mask
98.36
4.73
31.50
4.90
58.77
63.12
95.86
97.28
96.79

Output Visualization of the Two Auxiliary Branches.
Recall that to perform motion editing,
276

we propose to transfer the targeted motion from the motion-reference branch and realize content
277

preservation via feature injection from the reconstruction branch. To verify the effectiveness, we
278

visualized the output of each branch in Fig. 6b. It is observed that the motion-reference branch
279

(4th row) generates video with the target motion, and effectively transfers it to the main path (3rd
280

row); meanwhile, the main path inherits the content from the reconstruction branch (2nd row), thus
281

enhancing the consistency of unedited parts.
282
Table 2: Impact of various components.

Content
Preservation
Motion
Injection
Structure
Control
Frame
Similarity
Textual
Alignment
Frame
Consistency

90.54
28.76
96.99
!
97.28
29.95
98.12
!
!
91.30
31.48
98.08
!
!
96.11
31.37
98.12
!
!
!
96.29
31.43
98.09

The Effectiveness of Each Component.
To
283

demonstrate that all the designed feature injection
284

techniques in Sec. 4.1 & 4.2 contribute to the final
285

results, we make a quantitative evaluation on 15
286

motion editing cases, as we utilize all three com-
287

ponents in motion editing. To assess the similarity
288

between the edited video and the source video (e.g.,
289

background and identity), we introduce the ‘Frame
290

Similarity’, which is the average frame cosine similarity between the source frame embedding and
291

the edited frame embedding. As shown in Tab. 2, editing with content preservation results in high
292

frame similarity, suggesting that replacing value features in SA-S modules can effectively retain the
293

content of the source video. The use of motion injection and structure control significantly enhances
294

‘Textual Alignment’, indicating successful transfer of the targeted motion to the main editing path.
295

Ultimately, the best results are achieved through the combined use of all components.
296

8


---Page Break---
(a) Visualization of attention features.
(b) Visualization of each branch’s output.
Figure 6: (6a): Visualization of spatial query in SA-S (second row), cross-frame temporal attention
maps in SA-T (third row), and the magnitude of optical flow (fourth row). (6b): Visualization of the
video output of the main editing path, the reconstruction branch and the motion-reference branch.

(a) Ablation study on t0 in Eq. 2.
(b) Ablation study on t1 in Eq. 4.
Figure 7: Ablation study on hyper-parameters.

Ablation on Hyper-parameters. We utilize content preservation in Eq. 2 to maintain the original
297

content from the source video. By varying the feature injection steps in Fig. 7a, we observe that
298

replacing the value features at a few steps introduces inconsistencies in the background (footprints
299

on the beach). In practice, we adhere to the hyper-parameter selection outlined in [5] (last row).
300

Simultaneously, we note that adjusting the blend layers and steps in Eq. 4 can effectively regulate
301

the extent to which the edited image adheres to the original image. For instance, in the stylization
302

demonstrated in Fig. 7b, injecting the attention map into fewer (15) steps yields a stylized output that
303

may not retain the same structure as the input, while injecting into all 50 steps results in videos with
304

nearly identical textures but less stylization. Users have the flexibility to adjust the blended steps to
305

achieve their preferred balance between stylization and fidelity.
306

6
Conclusion and Limitations
307

In this paper, we design a novel tuning-free framework UniEdit for both video motion and appearance
308

editing. By leveraging a motion-reference branch and a reconstruction branch and injecting features
309

into the main editing path, it is capable of performing motion editing and various appearance
310

editing. There are nevertheless some limitations. Firstly, we observe performance degradation when
311

performing both types of editing simultaneously. Secondly, since our work is based on T2V models,
312

the proposed method also inherits some of the shortcomings of the existing models, such as inferior
313

performance in understanding complex prompts. We exhibit the failure cases in Appendix B.6.
314

9


---Page Break---
References
315

[1] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat,
316

Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for
317

video generation. arXiv preprint arXiv:2401.12945, 2024.
318

[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Do-
319

minik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion:
320

Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
321

[3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja
322

Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent
323

diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
324

Recognition, pages 22563–22575, 2023.
325

[4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow
326

image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision
327

and Pattern Recognition, pages 18392–18402, 2023.
328

[5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng.
329

Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing.
330

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
331

[6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image
332

diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
333

pages 23206–23217, 2023.
334

[7] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-
335

aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on
336

Computer Vision, pages 23040–23050, 2023.
337

[8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo
338

Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for
339

high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
340

[9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying
341

Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models.
342

arXiv preprint arXiv:2401.09047, 2024.
343

[10] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, and Ming-Hsuan
344

Yang. Motion-conditioned diffusion model for controllable video synthesis. arXiv preprint
345

arXiv:2304.14404, 2023.
346

[11] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin.
347

Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint
348

arXiv:2305.13840, 2023.
349

[12] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-
350

Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-guided
351

attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023.
352

[13] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-
353

based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.
354

[14] Paul Couairon, Clément Rambour, Jean-Emmanuel Haugeard, and Nicolas Thome. Videdit:
355

Zero-shot and spatially aware text-driven video editing. arXiv preprint arXiv:2306.08707, 2023.
356

[15] Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. Dragvideo:
357

Interactive drag-style video editing. arXiv preprint arXiv:2312.02216, 2023.
358

[16] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis
359

Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings
360

of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.
361

10


---Page Break---
[17] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo
362

Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models.
363

arXiv preprint arXiv:2309.16496, 2023.
364

[18] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion
365

features for consistent video editing. In International Conference on Learning Representations
366

(ICLR), 2024.
367

[19] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh
368

Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing
369

text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709,
370

2023.
371

[20] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl:
372

Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933,
373

2023.
374

[21] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Ani-
375

matediff: Animate your personalized text-to-image diffusion models without specific tuning.
376

arXiv preprint arXiv:2307.04725, 2023.
377

[22] Tianyu He, Junliang Guo, Runyi Yu, Yuchi Wang, Jialiang Zhu, Kaikai An, Leyi Li, Xu Tan,
378

Chunyu Wang, Han Hu, et al. Gaia: Zero-shot talking avatar generation. In International
379

Conference on Learning Representations (ICLR), 2024.
380

[23] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen.
Latent video
381

diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint
382

arXiv:2211.13221, 2022.
383

[24] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or.
384

Prompt-to-prompt image editing with cross attention control. In International Conference on
385

Learning Representations (ICLR), 2023.
386

[25] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko,
387

Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High
388

definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
389

[26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
390

in neural information processing systems, 33:6840–6851, 2020.
391

[27] Jonathan Ho and Tim Salimans.
Classifier-free diffusion guidance.
arXiv preprint
392

arXiv:2207.12598, 2022.
393

[28] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J
394

Fleet. Video diffusion models. arXiv:2204.03458, 2022.
395

[29] Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, and Sibei Yang.
Free-
396

bloom: Zero-shot text-to-video generator with llm director and ldm animator. arXiv preprint
397

arXiv:2309.14494, 2023.
398

[30] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao
399

Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-
400

based image editing with multimodal large language models. arXiv preprint arXiv:2312.06739,
401

2023.
402

[31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang,
403

Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang,
404

Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video
405

generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and
406

Pattern Recognition, 2024.
407

[32] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using
408

temporal attention adaption for text-to-video diffusion models. arXiv preprint arXiv:2312.00845,
409

2023.
410

11


---Page Break---
[33] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zero-shot grounded video editing using
411

text-to-image diffusion models. arXiv preprint arXiv:2310.01107, 2023.
412

[34] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change
413

Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. arXiv
414

preprint arXiv:2312.00777, 2023.
415

[35] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. Rave:
416

Randomized noise shuffling for fast and consistent video editing with diffusion models. arXiv
417

preprint arXiv:2312.04524, 2023.
418

[36] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman.
419

Dreampose:
Fashion image-to-video synthesis via stable diffusion.
arXiv preprint
420

arXiv:2304.06025, 2023.
421

[37] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang
422

Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion
423

models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.
424

[38] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
425

Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv
426

preprint arXiv:2304.02643, 2023.
427

[39] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing
428

with cross-attention control. arXiv preprint arXiv:2303.04761, 2023.
429

[40] Qi Mao, Lan Chen, Yuchao Gu, Zhen Fang, and Mike Zheng Shou. Mag-edit: Localized
430

image editing in complex scenarios via mask-based attention-adjusted guidance. arXiv preprint
431

arXiv:2312.11396, 2023.
432

[41] Joanna Materzynska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and
433

Bryan Russell.
Customizing motion in text-to-video diffusion models.
arXiv preprint
434

arXiv:2312.04966, 2023.
435

[42] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano
436

Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In
437

International Conference on Learning Representations, 2022.
438

[43] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion
439

for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF
440

Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
441

[44] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv
442

Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors.
443

arXiv preprint arXiv:2302.01329, 2023.
444

[45] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei
445

Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally
446

consistent video processing. arXiv preprint arXiv:2308.07926, 2023.
447

[46] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng
448

Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the
449

IEEE/CVF International Conference on Computer Vision, 2023.
450

[47] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei
451

Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint
452

arXiv:2310.15169, 2023.
453

[48] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical
454

text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3,
455

2022.
456

[49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
457

resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
458

conference on computer vision and pattern recognition, pages 10684–10695, 2022.
459

12


---Page Break---
[50] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton,
460

Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al.
461

Photorealistic text-to-image diffusion models with deep language understanding. Advances in
462

Neural Information Processing Systems, 35:36479–36494, 2022.
463

[51] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with
464

singular value clipping. In Proceedings of the IEEE international conference on computer
465

vision, pages 2830–2839, 2017.
466

[52] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu,
467

Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without
468

text-video data. arXiv preprint arXiv:2209.14792, 2022.
469

[53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In
470

International Conference on Learning Representations (ICLR), 2021.
471

[54] Yao Teng, Enze Xie, Yue Wu, Haoyu Han, Zhenguo Li, and Xihui Liu. Drag-a-video: Non-rigid
472

video editing with point-based interaction. arXiv preprint arXiv:2312.02936, 2023.
473

[55] Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Mo-
474

tioneditor: Editing video motion via content-aware diffusion. arXiv preprint arXiv:2311.18830,
475

2023.
476

[56] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features
477

for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on
478

Computer Vision and Pattern Recognition, pages 1921–1930, 2023.
479

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
480

Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
481

processing systems, 30, 2017.
482

[58] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics.
483

Advances in neural information processing systems, 29, 2016.
484

[59] Cong Wang, Jiaxi Gu, Panwen Hu, Songcen Xu, Hang Xu, and Xiaodan Liang. Dreamvideo:
485

High-fidelity image-to-video generation with image retention and text guidance. arXiv preprint
486

arXiv:2312.03018, 2023.
487

[60] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang.
488

Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
489

[61] Ting-Chun Wang, Ming-Yu Liu, Andrew Tao, Guilin Liu, Bryan Catanzaro, and Jan Kautz.
490

Few-shot video-to-video synthesis. Advances in Neural Information Processing Systems, 32,
491

2019.
492

[62] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and
493

Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv
494

preprint arXiv:2303.17599, 2023.
495

[63] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang,
496

Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded
497

latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
498

[64] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying
499

Shan. Motionctrl: A unified and flexible motion controller for video generation. arXiv preprint
500

arXiv:2312.03641, 2023.
501

[65] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne
502

Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image
503

diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International
504

Conference on Computer Vision, pages 7623–7633, 2023.
505

13


---Page Break---
[66] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang,
506

Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu,
507

Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text
508

guided video editing competition, 2023.
509

[67] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan
510

Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video
511

generation using textual and structural guidance. arXiv preprint arXiv:2306.00943, 2023.
512

[68] Wilson Yan, Andrew Brown, Pieter Abbeel, Rohit Girdhar, and Samaneh Azadi. Motion-
513

conditioned image animation for video editing. arXiv preprint arXiv:2311.18827, 2023.
514

[69] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation
515

using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.
516

[70] Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender a video: Zero-shot
517

text-guided video-to-video translation. In ACM SIGGRAPH Asia 2023 Conference Proceedings,
518

2023.
519

[71] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G
520

Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video
521

transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
522

Recognition, pages 10459–10469, 2023.
523

[72] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu,
524

Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for
525

text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
526

[73] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image
527

diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer
528

Vision, pages 3836–3847, 2023.
529

[74] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and
530

Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint
531

arXiv:2305.13077, 2023.
532

[75] Yuxin Zhang, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Weiming Dong, and
533

Changsheng Xu. Motioncrafter: One-shot motion customization of diffusion models. arXiv
534

preprint arXiv:2312.05288, 2023.
535

14


---Page Break---
Supplementary Materials

We organize the Appendix as follows:
536

• Appendix A: detailed descriptions of experimental settings.
537

• Appendix B: more experimental results, including:
538

• Editing results on different T2V model (Appendix B.1).
539

• Quantitative ablation on hyper-parameter selection (Appendix B.2).
540

• Ablation study on mask-guided coordination (Appendix B.3).
541

• Observation and analysis on the proposed components (Appendix B.4).
542

• Analysis and comparison on inference time (Appendix B.5).
543

• Failure cases visualization (Appendix B.6).
544

• More Comparisons with baseline methods (Appendix B.7).
545

• More Editing results of UniEdit (Appendix B.8).
546

• Appendix C: Broader Impacts.
547

We encourage the readers to watch the videos on our project page.
548

A
Detailed Experimental Settings
549

Base T2V Model.
We instantiate the proposed method on LaVie [63], which is a pre-trained
550

text-to-video generation model that produces consistent and high-quality videos. To achieve a fair
551

comparison, we only leverage the base T2V model in LaVie and load the open-source pre-trained
552

weights for video editing tasks in the experiments. Note that the edited video clip could further be
553

seamlessly fed into the temporal interpolation model and the video super-resolution model to obtain
554

video with a longer duration and higher resolution.
555

Video Preprocessing.
For each input video, we resize it to the resolution of 320 × 512, followed by
556

normalization, which is consistent with the training configuration of LaVie. Then, the pre-processed
557

video is fed into the base model of Lavie to perform video editing. To maximize the generation power
558

of LaVie, we set all input videos to 16 frames. For a source video, it takes 1-2 minutes to edit on an
559

NVIDIA A100 GPU.
560

Configurations.
For real source videos, we inverse them with 50 DDIM inversion steps and perform
561

DDIM deterministic sampling with 50 steps for generation. For the generated videos, we use the
562

same start latent of synthesizing the source video as the initial noise zT for the main editing path and
563

two auxiliary branches. We use the commonly used classifier-free guidance technique [27] with a
564

scale of 7.5.
565

Details of User Study.
As a text-guided editing task, in addition to CLIP scores, it is crucial to
566

evaluate results through human subjective assessment. To achieve this, we utilized MOS (Mean
567

Opinion Score) as our metric and collected feedback from 10 experienced volunteers. We randomly
568

selected 20 editing samples and permuted results from different models. Volunteers were then tasked
569

to evaluate the results based on two perspectives: frame consistency and textual alignment. They
570

provided ratings for these aspects on a scale of 1-5. Specifically, frame consistency measures the
571

smoothness of the video, aiming to avoid dramatic jittering and ensure coherence between the content
572

of each frame. Textual alignment assesses whether the editing results adhere to the text guidance and
573

maintain the content of the source video. In the end, we computed the average user ratings for each
574

method as our final results.
575

15


---Page Break---
As illustrated in Tab. 1, UniEdit shows the best performance on frame consistency. Regarding textual
576

alignment, UniEdit significantly outperforms all other baselines, demonstrating its capacity to support
577

diverse editing scenarios.
578

Baselines.
We implement all baseline methods with their official repositories. For MasaCtrl [5],
579

we adapt it to video editing by first setting the base model to a T2V model [63], then performing
580

MasaCtrl on all frames of the source video. Moreover, since most baselines use StableDiffusion (SD)
581

as the base model, we resize the source video to 512 × 512 to align with the default configuration of
582

SD, then feed it into the denoising model, which can maximize the power of SD.
583

B
Additional Experimental Results and Analysis
584

B.1
Results on Different T2V Model
585

We additionally implement our method on VideoCrafter2 [9], a concurrent work on T2V generation
586

to demonstrate the flexibility of UniEdit. The results are shown in Fig. 8.
587

Figure 8: Editing results with UniEdit on VideoCrafter2 [9].

B.2
Quantitative Ablation on Hyper-parameter Selection
588

In practice, we empirically found set these values to fixed values, i.e., t0 = 50, L = 10 (same as
589

MasaCtrl [5]) and t1 = 25 can achieve satisfying results on most cases, and we further perform a
590

quantitative study when applying different hyper-parameters in Tab. 3&4.

Table 3: Quantitative comparison on hyper-parameter selection.
Metric
Frame Similarity
Textual Alignment
Frame Consistency

t0 = 20, L = 10
94.33
31.57
98.09
t0 = 50, L = 10
96.29
31.84
98.12
t0 = 50, L = 8
96.76
31.25
98.11
591

Table 4: Quantitative comparison on hyper-parameter selection.
Metric
Frame Similarity
Textual Alignment
Frame Consistency

t1 = 20
96.21
30.92
98.06
t1 = 25
96.29
31.43
98.09
t1 = 30
96.50
31.04
98.08

16


---Page Break---
B.3
Ablation Study on the Impact of Mask-Guided Coordination
592

To investigate the impact of mask-guided coordination, we begin by visualizing masks obtained
593

from 1) the attention map in CA-S modules; 2) the off-the-shelf segmentation model SAM [38],
594

followed by presenting both qualitative and quantitative results of implementing UniEdit with or
595

without mask-guided coordination.
596

As verified by previous work [24], the attention maps in CA-S modules contain correspondence
597

information between text and visual features. The underlying intuition is that the attention maps
598

between each word and the spatial features at point (i, j) indicate ‘how similar this token is to
599

the spatial feature at this location’. We visualize the text-image cross attention map alongside the
600

synthesized frame in Fig. 9. We observe spatial correspondences that align with the video output from
601

the attention map. For instance, areas with higher values of the token ‘man’ and ‘NYC’ correspond
602

to the foreground and background, respectively. We further employ a fixed threshold (0.4 in practice)
603

to derive binary segmentation maps from the attention maps. For comparison, we also display the
604

segmentation mask obtained by point prompt on SAM. It’s observed that the cross-attention mask is
605

generally accurate and could serve as a reliable proxy in practice when an external segmentor is not
606

available.
607

We examine the impact of mask-guided coordination through both qualitative and quantitative results
608

across 4 settings: {w/o UniEdit, UniEdit w/o mask, UniEdit with mask from CA-S, UniEdit with
609

mask from SAM}. Qualitatively, shown in Fig. 10, the implementation of UniEdit significantly
610

enhances the consistency between the edited videos and the original video. The application of the
611

mask-guided coordination technique further improves the consistency of unedited areas (e.g., color
612

and texture). The quantitative results in Tab. 5 align coherently with this analysis.
613

Table 5: Ablation on the proposed mask-guided coordination.
Metric
Textual Alignment
Frame Consistency

TAV
27.89
95.39
MasaCtrl∗
25.58
97.61
FateZero
27.30
96.72
Rerender
27.94
97.18
TokenFlow
28.58
97.02

UniEdit (w/o mask)
31.43
98.35
UniEdit (w CA-S mask)
31.49
98.33
UniEdit (w SAM mask)
31.50
98.36

Figure 9: Visualization of attention maps and masks in mask-guided coordination (Sec. 4.3). The top
row are attention maps corresponding to different tokens in CA-S modules, (a) is the final output
frame, (b) and (c) are the foreground/background binary mask obtained by employing a threshold on
the attention map of ‘Man’ token and point prompt segmentation with SAM, respectively.

17


---Page Break---
Figure 10: Qualitative editing results across 4 settings: w/o UniEdit (2nd row), UniEdit w/o mask
(3rd row), UniEdit with mask from CA-S (4th row), UniEdit with mask from SAM (5th row).

18


---Page Break---
B.4
More Observation and Analysis on the Proposed Components
614

Difference Between QK and V Features in SA-S Modules To comprehend why we can have
615

inhomogeneous QK and V and their differences, we visualized the results of swapping different
616

features (QK or V) in SA-S modules during style transfer tasks on the source video in Fig. 11a. As
617

can be seen, compared to editing with no feature replacement (2nd row), replacing QK in the 3rd row
618

results in the edited video adopting the same spatial structure as the source video. Simultaneously,
619

replacing V eradicates the style information in the 4th row, meaning the texture details from the
620

source video are utilized to replace the style depicted by the target prompt. To summarize, the query
621

and key features (in SA-S modules) dictate the spatial structure of the generated video, while the
622

value features tend to influence the texture, including details such as color tones.
623

Influence of Spatial Structure Control in Motion Editing We explored the role of spatial control
624

in motion editing. The proposed method synthesizes videos with larger modifications when removing
625

the spatial control mechanism on both the motion-reference branch and the main editing branch. We
626

visualized the results in Fig. 11b. It can be observed that although the motion-reference branch can
627

still generate the target motion without the control of spatial structure, the layout deviates significantly,
628

for example, the raccoon assumes a different pose and location. We regard this as a suboptimal
629

solution because, compared to the results presented in the 3rd row, the results w/o spatial structure
630

control modifies the object position of the source video, leading to a decrease in consistency between
631

the edited result and the source video.
632

(a) Replacing different features in SA-S modules.
(b) Motion editing w/ or w/o structure control.

Figure 11: Ablation on the proposed feature injection techniques. (11a): comparison of appearance
editing without feature replacement (2nd row), with QK replacement (3rd row), with V replacement
(4nd row); (11b): comparison of motion editing with and without the designed spatial structure
control mechanism.

19


---Page Break---
B.5
Analysis and Comparison on Inference Time
633

We conduct a theoretical analysis of the additional cost of UniEdit and an empirical comparison with
634

baseline methods in terms of inference speed.
635

Theoretically, our method primarily involves feature replacement operations in attention modules,
636

achieved through forward hook registration and introducing minimal additional computation. There-
637

fore, the main difference between synthesizing a video from random noise and editing a video
638

with UniEdit lies in the batch size of the denoising process (i.e., vanilla generation: batchsize=1,
639

appearance editing: batchsize=2, motion editing: batchsize=3), and this process could be further
640

accelerated through multi-GPU parallel processing techniques. Additionally, we utilize LaVie [63] as
641

the base T2V model in the paper, which takes approximately 45 seconds to synthesize a 16-frame
642

video. Our method can be even faster when adapted to more efficient base models.
643

Empirically, UniEdit demonstrates comparable speed with baseline methods. The comparison of
644

inference time on a single 16-frame source video clip with a resolution of 320x512 on 1 NVIDIA
645

A100 GPU is as follows:
646

Table 6: Quantitative comparison on inference time of editing a single 16-frame video clip.

Method
TAV
MasaCtrl∗FateZero Rerender TokenFlow
UniEdit
(appearance editing)
UniEdit
(motion editing)

Inference time ∼10min
∼90s
∼130s
∼110s
∼100s
∼95s
∼125s

B.6
Failure Cases Visualization
647

We exhibit failure cases in Fig. 12. Fig. 12a showcase when editing multiple elements simultaneously,
648

and we observe a relatively large inconsistency with the source video. A naive solution is to perform
649

editing with UniEdit multiple times. Fig. 12b visualizes the results when editing video with complex
650

scenes, and the model sometimes could not understand the semantics in the target prompt, resulting
651

in incorrect editing. This may be caused by the base model’s limited text understanding power,
652

as discussed in [30]. It could be alleviated by leveraging the reasoning power of MLLM [30], or
653

adapting approaches in complex scenario editing [40].
654

(a) Edit multiple elements simultaneously.
(b) Complex scene editing.

Figure 12: Visualization of failure cases.

B.7
More Comparison with State-of-the-Art Methods
655

Please refer to Fig. 13 and Fig. 14 for more comparison with the state-of-the-art methods. For a fair
656

comparison, we also migrated all baselines to LaVie [63], using the same base model as our method.
657

The results are presented in Fig. 15, and they are found to be inferior compared to those in Fig. 5
658

(based on Stable Diffusion).
659

B.8
More Results of UniEdit
660

More edited results of UniEdit are provided in Fig. 16-21. Examples of TI2V generation are provided
661

in Fig. 22.
662

20


---Page Break---
Figure 13: More comparison with state-of-the-art methods.

Figure 14: More comparison with state-of-the-art methods.

21


---Page Break---
Figure 15: More comparison with state-of-the-art methods. We adapt the baseline methods to the
text-to-video model LaVie [63] and compare with our method (also based on LaVie).

22


---Page Break---
Figure 16: More appearance editing results of UniEdit.

23


---Page Break---
Figure 17: More appearance editing results of UniEdit.

24


---Page Break---
Figure 18: More appearance editing results of UniEdit.

25


---Page Break---
Figure 19: More appearance editing results of UniEdit.

26


---Page Break---
Figure 20: More motion editing results of UniEdit.

27


---Page Break---
Figure 21: More motion editing results of UniEdit.

Figure 22: Results of text-image-to-video synthesis in Sec. 4.4.

28


---Page Break---
C
Broader Impacts
663

UniEdit is a tuning-free approach and is intended for advancing AI/ML research on video editing.
664

We encourage users to use the model responsibly. We discourage users from using the codes to
665

generate intentionally deceptive or untrue content or for inauthentic activities. It is suggested to add
666

watermarks to prevent misuse.
667

29


---Page Break---
NeurIPS Paper Checklist
668

1. Claims
669

Question: Do the main claims made in the abstract and introduction accurately reflect the
670

paper’s contributions and scope?
671

Answer: [Yes]
672

Justification: In this work, we present UniEdit, a tuning-free framework that supports
673

both video motion and appearance editing by harnessing the power of a pre-trained text-
674

to-video generator within an inversion-then-generation framework.Extensive experiments
675

demonstrate that UniEdit covers video motion editing and various appearance editing
676

scenarios, and surpasses the state-of-the-art method.
677

Guidelines:
678

• The answer NA means that the abstract and introduction do not include the claims
679

made in the paper.
680

• The abstract and/or introduction should clearly state the claims made, including the
681

contributions made in the paper and important assumptions and limitations. A No or
682

NA answer to this question will not be perceived well by the reviewers.
683

• The claims made should match theoretical and experimental results, and reflect how
684

much the results can be expected to generalize to other settings.
685

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
686

are not attained by the paper.
687

2. Limitations
688

Question: Does the paper discuss the limitations of the work performed by the authors?
689

Answer: [Yes]
690

Justification: We discussed the potential limitations of the method in Sec. 6 and presented
691

failed cases in Appendix B.6.
692

Guidelines:
693

• The answer NA means that the paper has no limitation while the answer No means that
694

the paper has limitations, but those are not discussed in the paper.
695

• The authors are encouraged to create a separate "Limitations" section in their paper.
696

• The paper should point out any strong assumptions and how robust the results are to
697

violations of these assumptions (e.g., independence assumptions, noiseless settings,
698

model well-specification, asymptotic approximations only holding locally). The authors
699

should reflect on how these assumptions might be violated in practice and what the
700

implications would be.
701

• The authors should reflect on the scope of the claims made, e.g., if the approach was
702

only tested on a few datasets or with a few runs. In general, empirical results often
703

depend on implicit assumptions, which should be articulated.
704

• The authors should reflect on the factors that influence the performance of the approach.
705

For example, a facial recognition algorithm may perform poorly when image resolution
706

is low or images are taken in low lighting. Or a speech-to-text system might not be
707

used reliably to provide closed captions for online lectures because it fails to handle
708

technical jargon.
709

• The authors should discuss the computational efficiency of the proposed algorithms
710

and how they scale with dataset size.
711

• If applicable, the authors should discuss possible limitations of their approach to
712

address problems of privacy and fairness.
713

• While the authors might fear that complete honesty about limitations might be used by
714

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
715

limitations that aren’t acknowledged in the paper. The authors should use their best
716

judgment and recognize that individual actions in favor of transparency play an impor-
717

tant role in developing norms that preserve the integrity of the community. Reviewers
718

will be specifically instructed to not penalize honesty concerning limitations.
719

3. Theory Assumptions and Proofs
720

30


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
721

a complete (and correct) proof?
722

Answer: [NA]
723

Justification: This paper aims to design a simple-and-effective video editing method named
724

UniEdit, without focusing on theoretical results.
725

Guidelines:
726

• The answer NA means that the paper does not include theoretical results.
727

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
728

referenced.
729

• All assumptions should be clearly stated or referenced in the statement of any theorems.
730

• The proofs can either appear in the main paper or the supplemental material, but if
731

they appear in the supplemental material, the authors are encouraged to provide a short
732

proof sketch to provide intuition.
733

• Inversely, any informal proof provided in the core of the paper should be complemented
734

by formal proofs provided in appendix or supplemental material.
735

• Theorems and Lemmas that the proof relies upon should be properly referenced.
736

4. Experimental Result Reproducibility
737

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
738

perimental results of the paper to the extent that it affects the main claims and/or conclusions
739

of the paper (regardless of whether the code and data are provided or not)?
740

Answer: [Yes]
741

Justification: This paper provides detailed information on the models, parameters, hyper-
742

parameter selection, computational resources in Sec. 5 and Appendix A to ensure repro-
743

ducibility.
744

Guidelines:
745

• The answer NA means that the paper does not include experiments.
746

• If the paper includes experiments, a No answer to this question will not be perceived
747

well by the reviewers: Making the paper reproducible is important, regardless of
748

whether the code and data are provided or not.
749

• If the contribution is a dataset and/or model, the authors should describe the steps taken
750

to make their results reproducible or verifiable.
751

• Depending on the contribution, reproducibility can be accomplished in various ways.
752

For example, if the contribution is a novel architecture, describing the architecture fully
753

might suffice, or if the contribution is a specific model and empirical evaluation, it may
754

be necessary to either make it possible for others to replicate the model with the same
755

dataset, or provide access to the model. In general. releasing code and data is often
756

one good way to accomplish this, but reproducibility can also be provided via detailed
757

instructions for how to replicate the results, access to a hosted model (e.g., in the case
758

of a large language model), releasing of a model checkpoint, or other means that are
759

appropriate to the research performed.
760

• While NeurIPS does not require releasing code, the conference does require all submis-
761

sions to provide some reasonable avenue for reproducibility, which may depend on the
762

nature of the contribution. For example
763

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
764

to reproduce that algorithm.
765

(b) If the contribution is primarily a new model architecture, the paper should describe
766

the architecture clearly and fully.
767

(c) If the contribution is a new model (e.g., a large language model), then there should
768

either be a way to access this model for reproducing the results or a way to reproduce
769

the model (e.g., with an open-source dataset or instructions for how to construct
770

the dataset).
771

(d) We recognize that reproducibility may be tricky in some cases, in which case
772

authors are welcome to describe the particular way they provide for reproducibility.
773

In the case of closed-source models, it may be that access to the model is limited in
774

31


---Page Break---
some way (e.g., to registered users), but it should be possible for other researchers
775

to have some path to reproducing or verifying the results.
776

5. Open access to data and code
777

Question: Does the paper provide open access to the data and code, with sufficient instruc-
778

tions to faithfully reproduce the main experimental results, as described in supplemental
779

material?
780

Answer: [No]
781

Justification: Due to company policy reasons, we are currently unable to upload the code.
782

The code will be publicly available after the paper is published.
783

Guidelines:
784

• The answer NA means that paper does not include experiments requiring code.
785

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
786

public/guides/CodeSubmissionPolicy) for more details.
787

• While we encourage the release of code and data, we understand that this might not be
788

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
789

including code, unless this is central to the contribution (e.g., for a new open-source
790

benchmark).
791

• The instructions should contain the exact command and environment needed to run to
792

reproduce the results. See the NeurIPS code and data submission guidelines (https:
793

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
794

• The authors should provide instructions on data access and preparation, including how
795

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
796

• The authors should provide scripts to reproduce all experimental results for the new
797

proposed method and baselines. If only a subset of experiments are reproducible, they
798

should state which ones are omitted from the script and why.
799

• At submission time, to preserve anonymity, the authors should release anonymized
800

versions (if applicable).
801

• Providing as much information as possible in supplemental material (appended to the
802

paper) is recommended, but including URLs to data and code is permitted.
803

6. Experimental Setting/Details
804

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
805

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
806

results?
807

Answer: [Yes]
808

Justification: This paper provides detailed information on the models, parameters, hyper-
809

parameter selection, computational resources in Sec. 5 and Appendix A to ensure repro-
810

ducibility.
811

Guidelines:
812

• The answer NA means that the paper does not include experiments.
813

• The experimental setting should be presented in the core of the paper to a level of detail
814

that is necessary to appreciate the results and make sense of them.
815

• The full details can be provided either with the code, in appendix, or as supplemental
816

material.
817

7. Experiment Statistical Significance
818

Question: Does the paper report error bars suitably and correctly defined or other appropriate
819

information about the statistical significance of the experiments?
820

Answer: [No]
821

Justification: The common practice in video editing does not including error bars, and we
822

follow the previous papers.
823

Guidelines:
824

• The answer NA means that the paper does not include experiments.
825

32


---Page Break---
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
826

dence intervals, or statistical significance tests, at least for the experiments that support
827

the main claims of the paper.
828

• The factors of variability that the error bars are capturing should be clearly stated (for
829

example, train/test split, initialization, random drawing of some parameter, or overall
830

run with given experimental conditions).
831

• The method for calculating the error bars should be explained (closed form formula,
832

call to a library function, bootstrap, etc.)
833

• The assumptions made should be given (e.g., Normally distributed errors).
834

• It should be clear whether the error bar is the standard deviation or the standard error
835

of the mean.
836

• It is OK to report 1-sigma error bars, but one should state it. The authors should
837

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
838

of Normality of errors is not verified.
839

• For asymmetric distributions, the authors should be careful not to show in tables or
840

figures symmetric error bars that would yield results that are out of range (e.g. negative
841

error rates).
842

• If error bars are reported in tables or plots, The authors should explain in the text how
843

they were calculated and reference the corresponding figures or tables in the text.
844

8. Experiments Compute Resources
845

Question: For each experiment, does the paper provide sufficient information on the com-
846

puter resources (type of compute workers, memory, time of execution) needed to reproduce
847

the experiments?
848

Answer: [Yes]
849

Justification: This paper provides detailed information on the computational resources in
850

Sec. 5 and Appendix A and inference time comparison in Tab. 6.
851

Guidelines:
852

• The answer NA means that the paper does not include experiments.
853

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
854

or cloud provider, including relevant memory and storage.
855

• The paper should provide the amount of compute required for each of the individual
856

experimental runs as well as estimate the total compute.
857

• The paper should disclose whether the full research project required more compute
858

than the experiments reported in the paper (e.g., preliminary or failed experiments that
859

didn’t make it into the paper).
860

9. Code Of Ethics
861

Question: Does the research conducted in the paper conform, in every respect, with the
862

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
863

Answer: [Yes]
864

Justification: The research strictly adheres to the NeurIPS Code of Ethics in every respect.
865

Guidelines:
866

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
867

• If the authors answer No, they should explain the special circumstances that require a
868

deviation from the Code of Ethics.
869

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
870

eration due to laws or regulations in their jurisdiction).
871

10. Broader Impacts
872

Question: Does the paper discuss both potential positive societal impacts and negative
873

societal impacts of the work performed?
874

Answer: [Yes]
875

Justification: The broader impacts are discussed in Appendix C.
876

33


---Page Break---
Guidelines:
877

• The answer NA means that there is no societal impact of the work performed.
878

• If the authors answer NA or No, they should explain why their work has no societal
879

impact or why the paper does not address societal impact.
880

• Examples of negative societal impacts include potential malicious or unintended uses
881

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
882

(e.g., deployment of technologies that could make decisions that unfairly impact specific
883

groups), privacy considerations, and security considerations.
884

• The conference expects that many papers will be foundational research and not tied
885

to particular applications, let alone deployments. However, if there is a direct path to
886

any negative applications, the authors should point it out. For example, it is legitimate
887

to point out that an improvement in the quality of generative models could be used to
888

generate deepfakes for disinformation. On the other hand, it is not needed to point out
889

that a generic algorithm for optimizing neural networks could enable people to train
890

models that generate Deepfakes faster.
891

• The authors should consider possible harms that could arise when the technology is
892

being used as intended and functioning correctly, harms that could arise when the
893

technology is being used as intended but gives incorrect results, and harms following
894

from (intentional or unintentional) misuse of the technology.
895

• If there are negative societal impacts, the authors could also discuss possible mitigation
896

strategies (e.g., gated release of models, providing defenses in addition to attacks,
897

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
898

feedback over time, improving the efficiency and accessibility of ML).
899

11. Safeguards
900

Question: Does the paper describe safeguards that have been put in place for responsible
901

release of data or models that have a high risk for misuse (e.g., pretrained language models,
902

image generators, or scraped datasets)?
903

Answer: [NA]
904

Justification: This paper poses no such risks.
905

Guidelines:
906

• The answer NA means that the paper poses no such risks.
907

• Released models that have a high risk for misuse or dual-use should be released with
908

necessary safeguards to allow for controlled use of the model, for example by requiring
909

that users adhere to usage guidelines or restrictions to access the model or implementing
910

safety filters.
911

• Datasets that have been scraped from the Internet could pose safety risks. The authors
912

should describe how they avoided releasing unsafe images.
913

• We recognize that providing effective safeguards is challenging, and many papers do
914

not require this, but we encourage authors to take this into account and make a best
915

faith effort.
916

12. Licenses for existing assets
917

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
918

the paper, properly credited and are the license and terms of use explicitly mentioned and
919

properly respected?
920

Answer: [Yes]
921

Justification: Yes, the creators or original owners of assets used in the paper are properly
922

credited, and the license and terms of use are explicitly mentioned and properly respected.
923

Guidelines:
924

• The answer NA means that the paper does not use existing assets.
925

• The authors should cite the original paper that produced the code package or dataset.
926

• The authors should state which version of the asset is used and, if possible, include a
927

URL.
928

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
929

34


---Page Break---
• For scraped data from a particular source (e.g., website), the copyright and terms of
930

service of that source should be provided.
931

• If assets are released, the license, copyright information, and terms of use in the
932

package should be provided. For popular datasets, paperswithcode.com/datasets
933

has curated licenses for some datasets. Their licensing guide can help determine the
934

license of a dataset.
935

• For existing datasets that are re-packaged, both the original license and the license of
936

the derived asset (if it has changed) should be provided.
937

• If this information is not available online, the authors are encouraged to reach out to
938

the asset’s creators.
939

13. New Assets
940

Question: Are new assets introduced in the paper well documented and is the documentation
941

provided alongside the assets?
942

Answer: [Yes]
943

Justification: We have uploaded the code of this paper to an anonymous repository and
944

provided the corresponding link in Appendix. The code will be made publicly available
945

after the paper is published.
946

Guidelines:
947

• The answer NA means that the paper does not release new assets.
948

• Researchers should communicate the details of the dataset/code/model as part of their
949

submissions via structured templates. This includes details about training, license,
950

limitations, etc.
951

• The paper should discuss whether and how consent was obtained from people whose
952

asset is used.
953

• At submission time, remember to anonymize your assets (if applicable). You can either
954

create an anonymized URL or include an anonymized zip file.
955

14. Crowdsourcing and Research with Human Subjects
956

Question: For crowdsourcing experiments and research with human subjects, does the paper
957

include the full text of instructions given to participants and screenshots, if applicable, as
958

well as details about compensation (if any)?
959

Answer: [NA]
960

Justification: The paper does not involve crowdsourcing nor research with human subjects.
961

Guidelines:
962

• The answer NA means that the paper does not involve crowdsourcing nor research with
963

human subjects.
964

• Including this information in the supplemental material is fine, but if the main contribu-
965

tion of the paper involves human subjects, then as much detail as possible should be
966

included in the main paper.
967

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
968

or other labor should be paid at least the minimum wage in the country of the data
969

collector.
970

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
971

Subjects
972

Question: Does the paper describe potential risks incurred by study participants, whether
973

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
974

approvals (or an equivalent approval/review based on the requirements of your country or
975

institution) were obtained?
976

Answer: [NA]
977

Justification: The paper does not involve crowdsourcing nor research with human subjects.
978

Guidelines:
979

• The answer NA means that the paper does not involve crowdsourcing nor research with
980

human subjects.
981

35


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
982

may be required for any human subjects research. If you obtained IRB approval, you
983

should clearly state this in the paper.
984

• We recognize that the procedures for this may vary significantly between institutions
985

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
986

guidelines for their institution.
987

• For initial submissions, do not include any information that would break anonymity (if
988

applicable), such as the institution conducting the review.
989

36


---Page Break---
