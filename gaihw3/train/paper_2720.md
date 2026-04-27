General Compression Framework for
Efficient Transformer Object Tracking

Anonymous Author(s)
Affiliation
Address
email

Abstract

Transformer-based trackers have established a dominant role in the field of visual
1

object tracking. While these trackers exhibit promising performance, their deploy-
2

ment on resource-constrained devices remains challenging due to inefficiencies. To
3

improve the inference efficiency and reduce the computation cost, prior approaches
4

have aimed to either design lightweight trackers or distill knowledge from larger
5

teacher models into more compact student trackers. However, these solutions
6

often sacrifice accuracy for speed. Thus, we propose a general model compression
7

framework for efficient transformer object tracking, named CompressTracker, to
8

reduce the size of a pre-trained tracking model into a lightweight tracker with
9

minimal performance degradation. Our approach features a novel stage division
10

strategy that segments the transformer layers of the teacher model into distinct
11

stages, enabling the student model to emulate each corresponding teacher stage
12

more effectively. Additionally, we also design a unique replacement training tech-
13

nique that involves randomly substituting specific stages in the student model
14

with those from the teacher model, as opposed to training the student model in
15

isolation. Replacement training enhances the student model’s ability to replicate
16

the teacher model’s behavior. To further forcing student model to emulate teacher
17

model, we incorporate prediction guidance and stage-wise feature mimicking to
18

provide additional supervision during the teacher model’s compression process.
19

Our framework CompressTracker is structurally agnostic, making it compatible
20

with any transformer architecture. We conduct a series of experiment to verify the
21

effectiveness and generalizability of CompressTracker. Our CompressTracker-4
22

with 4 transformer layers, which is compressed from OSTrack, retains about 96%
23

performance on LaSOT (66.1% AUC) while achieves 2.17× speed up.
24

1
Introduction
25

Visual object tracking is tasked with continuously localizing a target object across video frames based
26

on the initial bounding box in the first frame. Transformer-based trackers have achieved promising
27

performance on well-established benchmarks, their deployment on resource-restricted device remains
28

a significant challenge. Developing a strong tracker with high efficiency is of great significance.
29

To reduce the inference cost of models, previous works attempt to design lightweight trackers or
30

transfer the knowledge from teacher models to student trackers. Despite achieving increased speed,
31

these existing methods still exhibit notable limitations. (1) Inferior Accuracy. Certain works propose
32

lightweight tracking models [6, 10, 4, 21, 26] or employ neural architecture search (NAS) to search
33

better architecture [42]. Due to the limited number of parameters, these models often suffer from
34

underfitting and inferior performance. (2) Complex Training. Some works [15] aim to enhance the
35

accuracy of fast trackers through transferring the knowledge from a teacher tracker to a student model.
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
(a) We compress OSTrack and achieve promising performance.
(b) Performance and speed comparison
(c) FLOPs and params comparison

FLOPs
Params
FPS
AUC

Figure 1: We apply our framework to OSTrack under several different layer configurations. (a) We
implement each enhancement into our CompressTracker step by step. The training time is calculated
by using 8 NVIDIA RTX 3090 GPUs. Notably, our CompressTracker-4 accelerates OSTrack by
2.17× while preserving approximately 96% of its original accuracy, thereby demonstrating the
effectiveness of our framework. (b) Performance and speed comparison of CompressTracker variants
with different numbers of layers. CT-x refers to a version of CompressTracker with ’x’ layers. (c)
FLOPs and parameters comparison of CompressTracker variants with different numbers of layers.
Despite the improved performance, [15] introduces a complex multi-stage training strategy, which
37

is time-consuming. Any suboptimal performance in these individual stages can cumulatively result
38

in suboptimal performance in the final model. (3) Stucture Limitation. Additionally, the model
39

reduction paradigm in [15] severely restricts the structure of student models to be consistent only
40

with the teacher’s model.
41

Thus, we introduce CompressTracker, a novel and general model compression framework to enhance
42

the efficiency of transformer tracking models. The current dominant trackers are one-stream mod-
43

els [44, 15, 4, 10] characterized by a series of sequential transformer encoder layers, each designed
44

to refine the temporal matching features across frames. The output of each layer is a critical temporal
45

matching result that is refined as the layers get deeper. Given this layer-wise refinement, it becomes a
46

natural progression to consider the model not as a single entity but as a series of interconnected stages
47

and encourage student tracker to align teacher model at each stage. We propose the stage division
48

strategy, which involves partitioning the teacher model, a complex pretrained transformer-based
49

tracking model, into distinct stages that correspond to the layers of a simpler student model. This is
50

achieved by dividing the teacher model into a number of stages equivalent to the student model’s
51

layers. Each stage in the student model is then tasked with learning and replicating the functional
52

behavior of its corresponding stage in the teacher model. This division is not merely a structural
53

alteration but a strategic educational approach. By focusing each stage of the student model on
54

mimicking a specific stage of the teacher, we enable a targeted and efficient transfer of knowledge.
55

The student model learns not just the ’what’ of tracking—i.e., the raw matching of features—but also
56

the ’how’—i.e., the strategies developed by the teacher model at each layer of processing.
57

Contrary to conventional practices that isolate the training of student models, we employ a replacement
58

training methodology that strategically intertwines the teacher and student models. The core of this
59

methodology is the dynamic substitution of stages during training. we randomly select stages from
60

the student model and replace them with the corresponding stages from the teacher model. By doing
61

so, we situate the teacher model and the student model within a collaborative environment. This
62

arrangement permits the unaltered stages of the teacher model to collaboratively inform and enhance
63

the learning of the substituted stages in the student model rather than supervising the entire student
64

model as a single entity. The student model is not merely learning in parallel but is directly engaging
65

with the teacher’s learned behaviors. After training, we can just combine each stage of student model
66

for inference. The replacement training leads to a more authentic replication of the teacher’s tracking
67

strategies and helps to prevent the student model from overfitting to specific stages of the teacher
68

model, promoting a more stable training.
69

To augment the learning process, we introduce prediction guidance, which serves as a supervisory
70

signal for the student model by leveraging the teacher model’s predictions. By using the predictions
71

of the teacher model as a reference, the student model can converge more quickly. Furthermore,
72

to enhance the similarity of the temporal matching features across corresponding stages, we have
73

2


---Page Break---
developed a stage-wise feature mimicking strategy. This approach systematically aligns the feature
74

representations learned at each stage of the student model with those of the teacher model, thereby
75

promoting a more accurate and consistent learning. In Figure 1 (a), we show the procedure and the
76

results we are able to achieve with each step toward an efficient transformer tracker.
77

Compared to previous works, our CompressTracker holds many merits. (1) Enhanced Mimicking
78

and Performance. CompressTracker enables the student model to better mimic the teacher model,
79

resulting in better performance. As shown in Figure 1, our CompressTracker-4 achieves 2.17× speed
80

up while maintaining about 96% accuracy. (2) Simplified Training Process. Our CompressTracker
81

streamlines training into a single but efficient step. This simplification not only reduces the time
82

and resources required for training but also minimizes the potential for sub-optimal performance
83

associated with complex procedures. The training process for CompressTracker-4 requires merely 20
84

hours on 8 NVIDIA RTX 3090 GPUs. (3) Heterogeneous Model Compression. Our stage division
85

strategy gives a high degree of flexibility in the design of the student model. Our framework supports
86

any transformer architecture for student model, which is not restricted to the same structure of teacher
87

tracker. The number of layers and their structure are not predetermined but can be tailored to fit the
88

specific computational constraints and requirements of the deployment environment.
89

Our contribution can be summarized as follows: (1) We introduce a novel and general model
90

compression framework, CompressTracker, to facilitate the efficient transformer-based object tracking.
91

(2) We propose a stage division strategy that enables a fine-grained imitation of the teacher model at
92

the stage level, enhancing the precision and efficiency of knowledge transfer. (3) We propose the
93

replacement training to improve the student model’s capacity to replicate the teacher model’s behavior.
94

(4) We further incorporate the prediction guidance and feature mimicking to accelerate and refine
95

the learning process of the student model. (5) Our CompressTracker breaks structural limitations,
96

adapting to various transformer architectures for student model. It outperforms existing models,
97

notably accelerating OSTrack [44] by 2.17× while preserving approximately 96% accuracy.
98

2
Related Work
99

Visual Object Tracking. Visual object tracking aims to localize the target object of each frame
100

based on its initial appearance. Previous tracking methods [2, 28, 46, 3, 16, 27, 5, 23, 12, 41]
101

utilize a two-stream pipeline to decouple the feature extraction and relation modeling. Recently, the
102

one-stream pipeline hold the dominant role. [44, 14, 15, 1, 37, 8, 11, 19] combine feature extraction
103

and relation modeling into a unified process. These models are built upon vision transformer, which
104

consists of a series of transformer encoder layers. Thanks to a more adequate relationship modeling
105

between template and search frame, one-stream models achieve impressive performance. However,
106

these models suffer from low inference efficiency, which is the main obstacle to practical deployment.
107

Efficient Tracking. Some works have attempted to speed up tracking models. [42] utilizes neural
108

architecture search (NAS) to search a light Siamese network, and the searching process is complex.
109

[6, 10, 4, 26] design a lightweight tracking model, but the small number of parameters restricts the
110

accuracy to a large degree. MixFormerV2 [15] propose a complex multi-stage model reduction
111

strategy. Although MixFormerV2-S achieves real-time speed on CPU, the multi-stage training
112

strategy is time consuming, which requires about 120 hours (5 days) on 8 Nvidia RTX8000 GPUs,
113

even several times the original training time of MixFormer [14]. Any suboptimal performance
114

during these stages impact the final model’s performance negatively. Besides, the reduction paradigm
115

imposes constraints on the design of student models. To address these shortcuts, we propose the
116

general model compression framework, CompressTracker, to explore the roadmap toward an end-
117

to-end and traininig-efficient model compression for lightweight transformer-based tracker. Our
118

CompressTracker break the structure restriction and achieves balance between speed and accuracy.
119

Transformer Compression. Model compression aims to reduce the size and computational cost of
120

a large model while retaining as much performance as possible, and recently many attempts have
121

been made to speed up a large pretrained tranformer model. [18] reduced the number of parameters
122

through pruning technique, and [35] accomplished the quantization of BERT to 2-bits utilizing
123

Hessian information. [34, 36, 25, 38] leverage the knowledge distillation to transfer the knowledge
124

from teacher to student model and exploit pretrained model. Beyond language models, considerable
125

focus has also been placed on compressing vision transformer models. [33, 40, 9, 20, 7, 43, 45] utilize
126

multiple model compression techniques to compress vision transformer models. MixFormerV2 [15]
127

3


---Page Break---
Prediction

Guidance

Random 
Replacement

Template

Search Area

…
…

Student
Prediction

Stage 1
Stage 2
Stage N

Stage 1
Stage 2
Stage N

Teacher 
Prediction

Template

Search Area

…
…

Prediction
Stage 1
Stage 2

(b) Inference Phase

(a) Training Phase

Stage N

GroundTruth

GroundTruth

Supervision

Student Layer
Teacher Layer
Trainable
Frozen
Random Path

Feature
Mimicking

…
Random 
Replacement

Random 
Replacement

Feature
Mimicking

Feature
Mimicking

…

Figure 2: CompressTracker Framework. (a) In the training phase, we divide both the teacher model
and student model into an identical number of stages. We implement a series of training strategies
including replacement training, prediction guidance, and stage-wise feature mimicking, to enhance
the student model’s ability to emulate the teacher model. The dotted lines represent the randomly
selected paths for replacement training, with black dotted lines indicating the chosen path, while grey
dotted lines denote paths not selected in a specific training iteration. (b) During inference process, we
simply combine each stage of the student model for testing purposes.

proposed a two-stage model reduction paradigm to distill a lightweight tracker, relying on the complex
128

multi-stage distillation training. However, our CompressTracker propose an end-to-end and efficient
129

compression training to achieve any transformer structure compression, which speed up OSTrack
130

2.17× while maintaining about 96% accuracy.
131

3
CompressTracker
132

In this section, we will introduce our proposed general model compression framework, Com-
133

pressTracker. The workflow of our CompressTracker in illustrated in Figure 2.
134

3.1
Stage Division
135

Recently, transformer-based one-stream tracking models [8, 14, 44, 15] surpass conventional Siamese
136

trackers [2, 13, 12], becoming the dominant manner in the field of visual object tracking. These
137

models consist of several transformer encoder layers, each generating and progressively refining
138

temporal matching features. Building upon this layer-wise refinement, we introduce the stage division
139

strategy, which segments the model into a series of sequential stages. This approach encourages the
140

student model to emulate the teacher model’s behavior at each individual stage. Specifically, we
141

denote the pretrained tracker and the compressed model as teacher and student model, with Nt and
142

Ns layers, respectively. Both teacher and student models are then divided into Ns stages, where each
143

stage in the student model encompasses a single layer, and each corresponding stage in the teacher
144

model may aggregate multiple layers. For a specific stage i, we establish a correspondence between
145

the stages of the teacher and student models. The objective of stage division is to enforce each stage
146

of the student model to replicate its counterpart in the teacher model. This stage division strategy
147

breaks the traditional approach that treats the model as an indivisible whole [6, 10, 4, 15]. Instead, it
148

enables a fine-grained learning process where the student model transfers knowledge from the teacher
149

in a more detailed, stage-specific manner.
150

Unlike the reduction paradigm adopted in [15], which confines itself to pruning within identical
151

structures, our CompressTracker framework facilitates support for arbitrary transformer structures of
152

the student tracker, thanks to our innovative stage-wise division design. To align the size and channel
153

dimensions of the student model’s temporal matching features with those of the teacher model, we
154

implement input and output projection layers before and after the student layers, respectively. These
155

4


---Page Break---
projection layers serve as an adjustment mechanism to ensure compatibility between the teacher
156

and student models and allow for a broader range of architectural possibilities for the student model.
157

During the inference process, these input and output injection layers are omitted.
158

3.2
Replacement Training
159

During the training process, we adopt the replacement training to integrates teacher model and student
160

models, diverging from the conventional practice of training the student model in isolation. In a
161

specific training iteration, we implement a stochastic process to determine which stages of the student
162

model are to be replaced by the corresponding stages of the teacher model. For the specific stage
163

i, we decide whether to replace or not by random Bernoulli sampling bi with probability p, where
164

bi ∈{0, 1}. If bi equals 1, the output from the preceding stage i −1 is directed to the i student stage,
165

otherwise, we channel the output into the i frozen teacher stage. This replacement training creates
166

a collaborative learning environment where the teacher model dynamically supervises the student
167

model. The unreplaced stages of teacher provide valuable contextual supervision for a specific stage
168

in the student model. Consequently, the student model is not operating in parallel but is actively
169

engaged with and learning from the teacher’s established behaviors. For the optimization of student
170

model, we only require the groundtruth box and denote the loss as Ltrack. Upon completion of the
171

training process, the student model’s stages are harmoniously combined for inference. We show the
172

pseudocode code in Appendix A.1.
173

3.3
Prediction Guidance & Stage-wise Feature Mimicking
174

Replacement training enables the student model to learn the behavior of each individual stage,
175

resulting in enhanced performance. However, merely forcing student model to emulate teacher model
176

may be overly challenging for a smaller-sized student. Thus, we employ the teacher’s predictions to
177

further guide the learning of compressed tracker. We apply the same loss as Ltrack for prediction
178

guidance, which is denoted as Lpred. With the aid of prediction guidance, student benefits from a
179

quicker and stable learning process, assimilating knowledge from teacher model more effectively.
180

While prediction guidance accelerates the convergence, the student tracker might not entirely match
181

the complex behavior of the teacher model. We introduce the stage-wise feature mimicking to further
182

synchronize the temporal matching features between corresponding stages of the teacher and student
183

models. This alignment is quantified by calculating the L2 distance between the outputs of these
184

stages, which is referred as Lfeat. It is worth noting that any metric assessing the discrepancy in
185

feature distributions can serve as the loss function. However, we choose a simple L2 distance rather
186

than a complex loss to highlight the effectiveness of our stage division and replacement training
187

strategies. The stage-wise feature mimicking not only promotes a closer similarity in the feature
188

representations of corresponding stages but also enhances the overall coherence between the teacher
189

and student models.
190

3.4
Progressive Replacement
191

In Section 3.2, we describe the replacement training strategy. Although setting the Bernoulli sampling
192

probability p as a constant value can realize the compression, these stages have not been trained
193

together at the same time and there may be some dissonance. A further finetuning step is necessary
194

to achieve better harmony among the stages. Thus, we introduce a progressive replacement strategy
195

to bridges the gap between the two initially separate training phases, fostering an end-to-end easy-to-
196

hard learning process. By adjusting the value of p, we can control the number of stages to be replaced.
197

The value of p gradually increases from pinit to 1.0, allowing for a more incremental and coherent
198

training progression:
199

p =






pinit,
0 <= t < α1m,
pinit + pinit
t−α1m
(1−α1−α2)m,
α1m <= t <= (1 −α2)m,
1.0,
(1 −α2)m < t <= m,
(1)

where m represents the total number of training epochs, and t is a specific training epoch, α1 and
200

α2 are hyper parameters to modulate the training process. Specifically, α1 controls the duration of
201

5


---Page Break---
Method
LaSOT
LaSOText
TNL2K
TrackingNet
UAV123
FPS
AUC
PNorm
P
AUC
P
AUC
P
AUC
PNorm
P
AUC
P

OSTrack-256 [44]
69.1
78.7
75.2
47.4
53.3
54.3
-
83.1
87.8
82.0
68.3
-
105
CompressTracker-2
60.4 87%
68.5
61.5
40.4 85%
43.8
48.5 89%
45.0
78.2 94%
83.3
74.8
62.5 92%
82.5
346 3.30×
CompressTracker-3
64.9 94%
74.0
68.4
44.6 94%
49.6
52.6 97%
50.9
81.6 98%
86.7
79.4
65.4 96%
88.3
267 2.54×
CompressTracker-4
66.1 96%
75.2
70.6
45.7 96%
50.8
53.6 99%
52.5
82.1 99%
87.6
80.1
67.4 99%
88.0
228 2.17×
CompressTracker-6
67.5 98%
77.5
72.4
46.7 99%
52.5
54.7 101%
54.3
82.9 99%
87.8
81.5
67.9 99%
88.7
162 1.54×
CompressTracker-8
68.4 99%
78.0
73.1
47.2 99%
53.1
55.2 102%
54.8
83.3 101%
88.0
81.9
68.2 99%
89.0
127 1.21×
Table 1: Compress OSTrack. We compress OSTrack multiple configurations with different layer
settings. CompressTracker-x denotes the compressed student model with ’x’ layers. We report the
performance on 5 benchmarks and calculate the performance gap in comparison to the original
OSTrack. Our CompressTracker effectively achieves the balance between performance and speed.

Method
LaSOT
LaSOText
TNL2K
TrackingNet
UAV123
FPS
AUC
PNorm
P
AUC
P
AUC
P
AUC
PNorm
P
AUC
P

MixFormerV2-B [15]
70.6
80.8
76.2
50.6
56.9
57.4
58.4
83.4
88.1
81.6
69.9
92.1
165
MixFormerV2-S [15]
60.6
69.9
60.4
43.6
46.2
48.3
43.0
75.8
81.1
70.4
65.8
86.8
325
CompressTracker-M-S
62.0 88%
70.9
63.2
44.5 88%
47.1
50.2 87%
47.8
77.7 93%
82.5
73.0
66.9 96%
87.1
325 1.97×
Table 2: Compress MixFormerV2. We compress MixFormerV2 into CompressTracker-M-S with
4 layers, which is the same as MixFormerV2-S including the dimension of MLP layer. We report
the performance on 5 benchmarks and calculate the performance gap in comparison to the origin
MixFormerV2-B. Our CompressTracker-M-S outperforms MixFormerV2-S under the same setting.
warmup process, whereas α2 determines the length of final finetuning process. The mathematical
202

expectation of p for each layer is:
203

E(p) =
Z m

0
pdt = [1 + pinit

2
+ 1 −pinit

2
(α2 −α1)]m.
(2)

It is worth noting that each layer is optimized fewer times than the total iteration count, according to
204

the mathematical expectation. Through dynamically adjusting the replacement rate p, we eliminate
205

the requirement of finetuning and accomplish an end-to-end model compression.
206

3.5
Training and Inference
207

Our CompressTracker is a general framework applicable to a wide array of student model architectures.
208

For the optimization of student model, our CompressTracker solely requires an end-to-end and easy-
209

to-hand training process instead of multi-stage training methodologies. Furthermore, our approach
210

simplifies the loss function design, eliminating the need for complex formulations. During training,
211

teacher model is frozen and we only optimize student tracker. The total loss for CompressTracker is:
212

L = λtrackLtrack + λpredLpred + λfeatLfeat.
(3)

After training, the various stages of the student model are combined to create a unified model for the
213

inference phase. Consistent with previous methods [44, 14], a Hanning window penalty is adopted.
214

4
Experiments
215

4.1
Implement Details
216

Our framework CompressTracker is general and not dependent on a specific transformer structure,
217

hence we select OSTrack [44] as baseline, which is a simple and effective transformer-based tracker.
218

The training datasets consist of LaSOT [17], TrackingNet [32], GOT-10K [24], and COCO [29],
219

following OSTrack [44] and MixFormerV2 [15]. We set λtrack as 1, λpred as 1, and λfeat as 0.2.
220

The pinit is set as 0.5. We train the CompressTracker with AdamW optimizer [31], with the weight
221

decay as 10−4 and the initial learning rate of 4 × 10−5. The batch size is 128. The total training
222

epochs is 500 with 60K image pairs per epoch andthe learning rate is reduced by a factor of 10 after
223

400 epochs. α1 and α2 are set as 0.1. The search and template images are resized to resolutions
224

of 288 × 288 and 128 × 128. We initialize the CompressTracker with the pretrained parameters of
225

OSTrack. We report the inference speed on a NVIDIA RTX 2080Ti GPU.
226

6


---Page Break---
Method
LaSOT
LaSOText
TNL2K
TrackingNet
UAV123
FPS
AUC
PNorm
P
AUC
P
AUC
P
AUC
PNorm
P
AUC
P

OSTrack-256 [44]
69.1
78.7
75.2
47.4
53.3
54.3
-
83.1
87.8
82.0
68.3
-
105
SMAT [21]
61.7
71.1
64.6
-
-
-
-
78.6
84.2
75.6
64.3
83.9
158
CompressTracker-SMAT
62.8 91%
72.2
64.0
43.4 92%
46.0
49.6 91%
46.9
79.7 96%
85.0
75.4
65.9 96%
86.4
138 1.31×
Table 3: Compress OSTrack for SMAT. We compress OSTrack into CompressTracker-SAMT
with 4 SMAT layers, which is the same as SMAT. We report the performance on 5 benchmarks and
calculate the performance gap in comparison to the original OSTrack. Our CompressTracker-SAMT
outperforms SMAT under the same setting.

Method
LaSOT
LaSOText
TNL2K
TrackingNet
UAV123
FPS
AUC
PNorm
P
AUC
P
AUC
P
AUC
PNorm
P
AUC
P

CompressTracker-2
60.4
68.5
61.5
40.4
43.8
48.5
45.0
78.2
83.3
74.8
62.5
82.5
346
CompressTracker-3
64.9
74.0
68.4
44.6
49.6
52.6
50.9
81.6
86.7
79.4
65.4
88.3
267
CompressTracker-4
66.1
75.2
70.6
45.7
50.8
53.6
52.5
82.1
87.6
80.1
67.4
88.0
228
CompressTracker-6
67.5
77.5
72.4
46.7
52.5
54.7
54.3
82.9
87.8
81.5
67.9
88.7
162
CompressTracker-8
68.4
78.0
73.1
47.2
53.1
55.2
54.8
83.3
88.0
81.9
68.2
89.0
127

HiT-Base [26]
64.6
73.3
68.1
44.1
-
-
-
80.0
84.4
77.3
65.6
-
175
HiT-Samll [26]
60.5
68.3
61.5
40.4
-
-
-
77.7
81.9
73.1
63.3
-
192
HiT-Tiny [26]
54.8
60.5
52.9
35.8
-
-
-
74.6
78.1
68.8
53.2
-
204
SMAT [21]
61.7
71.1
64.6
-
-
-
-
78.6
84.2
75.6
64.3
83.9
158
MixFormerV2-S [15]
60.6
69.9
60.4
43.6
46.2
48.3
43.0
75.8
81.1
70.4
65.8
86.8
325
FEAR-L [6]
57.9
68.6
60.9
-
-
-
-
-
-
-
-
-
-
FEAR-XS [6]
53.5
64.1
54.5
-
-
-
-
-
-
-
-
-
80
HCAT [10]
59.0
68.3
60.5
-
-
-
-
76.6
82.6
72.9
63.6
-
195
E.T.Track [4]
59.1
-
-
-
-
-
-
74.5
80.3
70.6
62.3
-
150
LightTrack-LargeA [42]
55.5
-
56.1
-
-
-
-
73.6
78.8
70.0
-
-
-
LightTrack-Mobile [42]
53.8
-
53.7
-
-
-
-
72.5
77.9
69.5
-
-
120
STARK-Lightning [41]
58.6
69.0
57.9
-
-
-
-
-
-
-
-
-
200
DiMP [3]
56.9
65.0
56.7
-
-
-
-
74.0
80.1
68.7
65.4
-
77
SiamFC++ [39]
54.4
62.3
54.7
-
-
-
-
75.4
80.0
70.5
-
-
90

Table 4: State-of-the-art comparison. We compare our CompressTracker which is compressed from
OSTrack with previous light-weight tracking models. Our CompressTracker demonstrates superior
performance over previous models.
4.2
Compress Object Tracker
227

In this section, we compress the pretrained OSTrack into different layer configurations. We report
228

the performance of our CompressTracker across these configurations in Table 1. CompressTracker-
229

4 compress OSTrack from 12 layers into 4 layers, and maintain 96% and 99% performance on
230

LaSOT and TrackingNet while achieving 2.17× speed up. Furthermore, as shown in Figure 1,
231

the training process of CompressTracker-4 is notably efficient, requiring only approximately 20
232

hours using 8 NVIDIA RTX 3090 GPUs. For CompressTracker-6 and CompressTracker-8, as
233

we increase the number of layers, the performance gap between our compresstracker and OSTrack
234

diminishes. It is worth noting that our CompressTracker even outperforms the origin OSTrack on some
235

benchmarks. Specifically, CompressTracker-6 reaches 54.7% AUC on TNL2K, and CompressTracker-
236

8 achieves 55.2% AUC on TNL2K and 83.3% AUC on TrackingNet, while the origin OSTrack only
237

achieves 54.3% AUC on TNL2K and 83.1% AUC on TrackingNet. Our framework CompressTracker
238

demonstrates near lossless compression with the added benefit of increased processing speed.
239

Moreover, to affirm the generalization ability of our approach, we conduct experiments on Mix-
240

FormerV2 [15] and SMAT [21]. MixFormerV2-S is a fully transformer tracking model consisting
241

of 4 transformer layers, trained via a complex multi-stages model reduction paradigm. Following
242

MixFormerV2-S, we adopt MixFormerV2-B as teacher and compress it to a student model with
243

4 layers. The results are shown in Table 2. Our CompressTracker-M-S share the same structure
244

and channel dimension of MLP layers with MixFormerV2-S and outperforms MixFormerV2-S by
245

about 1.4% AUC on LaSOT. SMAT replace the vanilla attention in transformer layer with sepa-
246

rated attention. We compress OSTrack into a student model CompressTracker-SMAT, aligning the
247

number and structure of transformer layer with SAMT. We maintain the decoder of OSTrack for
248

CompressTracker-SMAT. CompressTracker-SMAT surpasses SMAT by 1.1% AUC on LaSOT, which
249

demonstrates that our framework is flexible and not limited by the structure of transformer layer.
250

Results in Table 1, 2, 3 verify the generalization ability and effectiveness of our framework.
251

4.3
Comparison with State-of-the-arts
252

To demonstrate the effectiveness of our CompressTracker, we compare our CompressTracker with
253

state-of-the-art efficient trackers in 5 benchmarks. As shown in Table 4, our CompressTracker
254

7


---Page Break---
#
Init. method
AUC
1
MAE-first4
59.9%
2
OSTrack-first4
62.0%
3
OSTrack-skip4
62.3%

Table 5: Backbone Initialization. ‘MAE-
first4’ denotes initializing the student model
using the first 4 layers of MAE-B. ‘OSTrack-
skip4’ represents utilizing every fourth layer
of OSTrack for the student model.

#
Init. & Opt.
AUC
1
Random & Trainable
62.3%
2
Teacher & Frozen
62.6%
3
Teacher & Trainable
62.8%

Table 6: Decoder Initialization and Optimization.
‘Random’ denotes randomly initialized decoder, and
’Teacher’ means the decoder is initialized with teacher
parameters. ‘Frozen’ represents that the decoder is
frozen, and ’Trainable’ denotes decoder is trainable.

#
Layer Split
AUC
1
Even
62.8%
2
Uneven
62.7%

Table 7: Stage Division. ‘Even’ denotes evenly divid-
ing stage strategy, and ‘Uneven’ means that the layer
number of each stage in teacher model is 2,2,6,2.

#
Epochs
AUC
1
300
65.2%
2
500
66.1%

Table 8: Training Epochs. ’300’
and ’500’ denote the total training
epochs.

Table 9: Ablation studies on LaSOT. The default choice for our model is colored in gray .

outperforms previous efficient trackers. Both HiT [26] and SMAT [21] are solely trained on the
255

groundtruth and reduce computation through specialized network architectures. MixFormerV2-S [15]
256

achieves model compression via a model reduction paradigm. Our CompressTracker-4 achieves
257

66.1% AUC on LaSOT while maintaining 228 FPS. CompressTracker-4 outperforms HiT-Base
258

by 1.5% AUC on LaSOT without any specialized model structure design. CompressTracker-4
259

achieves the balance between speed and accuracy. Meanwhile, our CompressTracker-2, with just two
260

transformer layers, maintains the highest speed at 346 FPS and also obtains competitive performance.
261

CompressTracker-2 surpasses HiT-Tiny by 5.6% AUC on LaSOT, and achieves about the same
262

performance as MixFormerV2-S with only two transformer layers. As we add more transformer
263

layers with CompressTracker-6 and CompressTracker-8, we see further improvements in performance.
264

These outcomes demonstrate the effectiveness of our CompressTracker framework.
265

#
Prediction
Guidance
Feature
Mimicking
Replacement
Traininig
AUC

1
62.8
2
✓
63.5
3
✓
63.3
4
✓
63.7
5
✓
✓
64.1
6
✓
✓
64.5
7
✓
✓
64.3
8
✓
✓
✓
65.2

Table 10: Ablation studies on LaSOT to
analyze the supervision of student model.
The default choice for our model is col-
ored in gray .

CompressTracker-2
CompressTracker-3
CompressTracker-4
CompressTracker-6
CompressTracker-8
56

58

60

62

64

66

68

AUC %

56.6

60.7

62.8

63.6

64.9

58.6

62.4

63.8

65.7

66.7

60.1

64.3

65.2

66.5

67.5

Naive Training
Distill Training
CompressTracker

Figure 3: Ablation study on training strategy.

266

4.4
Ablation Study
267

In this section, we conduct a series of ablation studies on LaSOT to explore the factors contributing to
268

the effectiveness of our CompressTracker. Unless otherwise specified, the teacher model is OSTrack,
269

and the student model has 4 encoder layers. The student model is trained for 300 epochs. Please see
270

Appendix A.2 for more analysis.
271

Backbone Initialization. We initialize the backbone of student model with different parameters and
272

only train the student model with groundtruth supervision. The results are shown in Table 5. It can
273

be observed that utilizing the knowledge from teacher model is crucial. Moreover, initializing with
274

skipped layers (#3) yields slightly better performance than continuous layers. This suggests that
275

initialization with skipped layers leads to improved representation similarity.
276

Decoder Initialization and Optimization. We investigate the influence of decoder’s initialization and
277

optimization on the accuracy of student tracker in Table 6. Initializing the decoder with parameters
278

from the teacher model (#2) results in an improvement of approximately 0.3% compared to a decoder
279

initialized randomly (#1), which underscores the benefits of transferring knowledge from the teacher
280

8


---Page Break---
model to enhance the accuracy of the student model’s decoder. Furthermore, making the decoder
281

trainable leads to an additional improvement of 0.2%.
282

Stage Division. Our stage division strategy divides the teacher model into the several stages, and we
283

explore the stage division strategy in Table 7. We design two kinds of division strategy: even and
284

uneven, For the even division, we evenly split the teacher model’s 12 layers into 4 stages, with each
285

stage comprising 3 layers. For uneven division, we follow the design manner in [22, 30] and divide
286

the 12 layers at a ratio of 1:1:3:1. Consequently, the number of layers in each stage of the teacher
287

model is 2, 2, 6, and 2, respectively. The performance of the two approaches is comparable, leading
288

us to select the equal division strategy for simplicity.
289

Analysis on Supervision. We conduct a series of experiments to comprehensively analyze the
290

supervision effects on the student model and to verify the effectiveness of our proposed training
291

strategy. Results are presented in Table 10. Our proposed replacement training approach (#4)
292

improves by 0.9 % AUC compared to singly training student model on groundtruth (#1), which
293

demonstrates that the replacement training enhances the similarity between teacher and student
294

models. Besides, prediction guidance (#5) and feature mimicking (#8) further boost the performance,
295

indicating the effectiveness of the two strategies. Compared to only training on groundtruth (#1), our
296

proposed replacement training, prediction guidance and feature mimicking collectively assist student
297

model in more closely mimicking the teacher model, resulting in a total increase of 2.4% AUC.
298

To further explore the generalization ability of our proposed training strategy, we compare the
299

performance of models with different layer numbers and training settings, as illustrated in Figure 3.
300

’Naive Training’ denotes that the student model is trained without teacher supervision and replacement
301

training. ’Distill Training’ represents that the student model is trained only with teacher supervision.
302

’CompressTracker’ refers to the same training setting in Table 10 #8. It can be observed that as the
303

number of layers increases, there is a corresponding improvement in accuracy. Our CompressTracker
304

shows a noticeable performance boost due to our proposed training strategy, which verifies the
305

effectiveness and generalization ability of our framework.
306

Training Epochs. Based on the analysis in Section 3.4, the optimization steps for each layer are
307

lower than total training steps. Thus, to ensure adequate training of each stage, we increase the
308

training epochs from 300 to 500, and show the result in Table 8. Extending the training epochs
309

ensures that student models receive comprehensive training, leading to improved accuracy.
310

5
Limitation
311

While our CompressTracker demonstrates promising performance and generalization, its training
312

is somewhat inefficient, requiring about 2× time compared to training a student model on ground
313

truth data (20h vs. 8h on 8 NVIDIA 3090 GPUs, as shown in Figure 1 (a)). Moreover, a performance
314

gap still exists between the teacher and student models suggests room for improvement in lossless
315

compression. Future efforts will focus on developing more efficient training methods to boost student
316

model accuracy and decrease training duration.
317

6
Broader Impacts
318

Our CompressTracker framework efficiently compresses object tracking models for edge device
319

deployment but poses potential misuse risks, such as unauthorized surveillance. We recommend users
320

to carefully consider the real-world implications and adopt risk mitigation strategies.
321

7
Conclusion
322

In this paper, we propose a general compression framework, CompressTracker, for visual object
323

tracking. We propose a novel stage division strategy to separate the structural dependencies between
324

the student and teacher models. We propose the replacement training to enhance student’s ability
325

to emulate the teacher model. We further introduce the prediction guidance and stage-wise feature
326

mimicking to improve performance. Extensive experiments verify the effectiveness and generalization
327

ability of our CompressTracker. Our CompressTracker is capable of accelerating tracking models
328

while preserving performance to the greatest extent possible.
329

9


---Page Break---
References
330

[1] Yifan Bai, Zeyang Zhao, Yihong Gong, and Xing Wei. Artrackv2: Prompting autoregressive tracker where
331

to look and how to describe. arXiv preprint arXiv:2312.17133, 2023.
332

[2] Luca Bertinetto, Jack Valmadre, Joao F Henriques, Andrea Vedaldi, and Philip HS Torr. Fully-convolutional
333

siamese networks for object tracking. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The
334

Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part II 14, pages 850–865. Springer, 2016.
335

[3] Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning discriminative model
336

prediction for tracking. In Proceedings of the IEEE/CVF international conference on computer vision,
337

pages 6182–6191, 2019.
338

[4] Philippe Blatter, Menelaos Kanakis, Martin Danelljan, and Luc Van Gool. Efficient visual tracking with
339

exemplar transformers. In Proceedings of the IEEE/CVF Winter conference on applications of computer
340

vision, pages 1571–1581, 2023.
341

[5] David S Bolme, J Ross Beveridge, Bruce A Draper, and Yui Man Lui. Visual object tracking using adaptive
342

correlation filters. In 2010 IEEE computer society conference on computer vision and pattern recognition,
343

pages 2544–2550. IEEE, 2010.
344

[6] Vasyl Borsuk, Roman Vei, Orest Kupyn, Tetiana Martyniuk, Igor Krashenyi, and Jiˇri Matas. Fear: Fast,
345

efficient, accurate and robust visual tracker. In European Conference on Computer Vision, pages 644–663.
346

Springer, 2022.
347

[7] Arnav Chavan, Zhiqiang Shen, Zhuang Liu, Zechun Liu, Kwang-Ting Cheng, and Eric P Xing. Vision
348

transformer slimming: Multi-dimension searching in continuous optimization space. In Proceedings of the
349

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4931–4941, 2022.
350

[8] Boyu Chen, Peixia Li, Lei Bai, Lei Qiao, Qiuhong Shen, Bo Li, Weihao Gan, Wei Wu, and Wanli Ouyang.
351

Backbone is all your need: A simplified architecture for visual object tracking. In European Conference on
352

Computer Vision, pages 375–392. Springer, 2022.
353

[9] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for
354

visual recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages
355

12270–12280, 2021.
356

[10] Xin Chen, Ben Kang, Dong Wang, Dongdong Li, and Huchuan Lu. Efficient visual tracking via hierarchical
357

cross-attention transformer. In European Conference on Computer Vision, pages 461–477. Springer, 2022.
358

[11] Xin Chen, Houwen Peng, Dong Wang, Huchuan Lu, and Han Hu. Seqtrack: Sequence to sequence learning
359

for visual object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
360

Recognition, pages 14572–14581, 2023.
361

[12] Xin Chen, Bin Yan, Jiawen Zhu, Dong Wang, Xiaoyun Yang, and Huchuan Lu. Transformer tracking. In
362

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8126–8135,
363

2021.
364

[13] Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, and Rongrong Ji. Siamese box adaptive
365

network for visual tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern
366

recognition, pages 6668–6677, 2020.
367

[14] Yutao Cui, Cheng Jiang, Limin Wang, and Gangshan Wu. Mixformer: End-to-end tracking with iterative
368

mixed attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
369

pages 13608–13618, 2022.
370

[15] Yutao Cui, Tianhui Song, Gangshan Wu, and Limin Wang. Mixformerv2: Efficient fully transformer
371

tracking. Advances in Neural Information Processing Systems, 36, 2024.
372

[16] Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, and Michael Felsberg. Atom: Accurate tracking
373

by overlap maximization. In Proceedings of the IEEE/CVF conference on computer vision and pattern
374

recognition, pages 4660–4669, 2019.
375

[17] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and
376

Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In Proceedings of the
377

IEEE/CVF conference on computer vision and pattern recognition, pages 5374–5383, 2019.
378

[18] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural
379

networks. arXiv preprint arXiv:1803.03635, 2018.
380

[19] Shenyuan Gao, Chunluan Zhou, and Jun Zhang. Generalized relation modeling for transformer tracking.
381

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18686–
382

18695, 2023.
383

[20] Chengyue Gong and Dilin Wang. Nasvit: Neural architecture search for efficient vision transformers with
384

gradient conflict-aware supernet training. ICLR Proceedings 2022, 2022.
385

[21] Goutam Yelluru Gopal and Maria A Amer. Separable self and mixed attention transformers for efficient
386

object tracking. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,
387

pages 6708–6717, 2024.
388

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
389

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
390

[23] João F Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista. High-speed tracking with kernelized
391

correlation filters. IEEE transactions on pattern analysis and machine intelligence, 37(3):583–596, 2014.
392

[24] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic
393

object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence, 43(5):1562–
394

1577, 2019.
395

10


---Page Break---
[25] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.
396

Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351, 2019.
397

[26] Ben Kang, Xin Chen, Dong Wang, Houwen Peng, and Huchuan Lu. Exploring lightweight hierarchical
398

vision transformers for efficient visual tracking. In Proceedings of the IEEE/CVF International Conference
399

on Computer Vision, pages 9612–9621, 2023.
400

[27] Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. Siamrpn++: Evolution of
401

siamese visual tracking with very deep networks. In Proceedings of the IEEE/CVF conference on computer
402

vision and pattern recognition, pages 4282–4291, 2019.
403

[28] Bo Li, Junjie Yan, Wei Wu, Zheng Zhu, and Xiaolin Hu. High performance visual tracking with siamese re-
404

gion proposal network. In Proceedings of the IEEE conference on computer vision and pattern recognition,
405

pages 8971–8980, 2018.
406

[29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
407

and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014:
408

13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages
409

740–755. Springer, 2014.
410

[30] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A
411

convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern
412

recognition, pages 11976–11986, 2022.
413

[31] Ilya Loshchilov and Frank Hutter.
Decoupled weight decay regularization.
arXiv preprint
414

arXiv:1711.05101, 2017.
415

[32] Matthias Muller, Adel Bibi, Silvio Giancola, Salman Alsubaihi, and Bernard Ghanem. Trackingnet:
416

A large-scale dataset and benchmark for object tracking in the wild. In Proceedings of the European
417

conference on computer vision (ECCV), pages 300–317, 2018.
418

[33] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient
419

vision transformers with dynamic token sparsification. Advances in neural information processing systems,
420

34:13937–13949, 2021.
421

[34] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert:
422

smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
423

[35] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and
424

Kurt Keutzer. Q-bert: Hessian based ultra low precision quantization of bert. In Proceedings of the AAAI
425

Conference on Artificial Intelligence, volume 34, pages 8815–8821, 2020.
426

[36] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for bert model compression.
427

arXiv preprint arXiv:1908.09355, 2019.
428

[37] Xing Wei, Yifan Bai, Yongchao Zheng, Dahu Shi, and Yihong Gong. Autoregressive visual tracking. In
429

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9697–9706,
430

2023.
431

[38] Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, and Ming Zhou. Bert-of-theseus: Compressing bert
432

by progressive module replacing. arXiv preprint arXiv:2002.02925, 2020.
433

[39] Yinda Xu, Zeyu Wang, Zuoxin Li, Ye Yuan, and Gang Yu. Siamfc++: Towards robust and accurate visual
434

tracking with target estimation guidelines. In Proceedings of the AAAI conference on artificial intelligence,
435

volume 34, pages 12549–12556, 2020.
436

[40] Yifan Xu, Zhijie Zhang, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng
437

Xu, and Xing Sun. Evo-vit: Slow-fast token evolution for dynamic vision transformer. In Proceedings of
438

the AAAI Conference on Artificial Intelligence, volume 36, pages 2964–2972, 2022.
439

[41] Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, and Huchuan Lu. Learning spatio-temporal transformer
440

for visual tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pages
441

10448–10457, 2021.
442

[42] Bin Yan, Houwen Peng, Kan Wu, Dong Wang, Jianlong Fu, and Huchuan Lu. Lighttrack: Finding
443

lightweight neural networks for object tracking via one-shot architecture search. In Proceedings of the
444

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15189, 2021.
445

[43] Zhendong Yang, Zhe Li, Ailing Zeng, Zexian Li, Chun Yuan, and Yu Li. Vitkd: Practical guidelines for vit
446

feature knowledge distillation. arXiv preprint arXiv:2209.02432, 2022.
447

[44] Botao Ye, Hong Chang, Bingpeng Ma, Shiguang Shan, and Xilin Chen. Joint feature learning and relation
448

modeling for tracking: A one-stream framework. In European conference on computer vision, pages
449

341–357. Springer, 2022.
450

[45] Jinnian Zhang, Houwen Peng, Kan Wu, Mengchen Liu, Bin Xiao, Jianlong Fu, and Lu Yuan. Minivit:
451

Compressing vision transformers with weight multiplexing. In Proceedings of the IEEE/CVF Conference
452

on Computer Vision and Pattern Recognition, pages 12145–12154, 2022.
453

[46] Zhipeng Zhang, Houwen Peng, Jianlong Fu, Bing Li, and Weiming Hu. Ocean: Object-aware anchor-free
454

tracking. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020,
455

Proceedings, Part XXI 16, pages 771–787. Springer, 2020.
456

11


---Page Break---
A
Appendix / supplemental material
457

Algorithm 1 Pseudocode of OSTrack in a PyTorch-like style

# z/x: RGB image of template/search region
# patch_embed: patch embedding layer,
# pos_embed_z/pos_embed_z: position embedding for template/search region
# blocks: transformer block layers
# decoder: decoder network

def forward(x, z):

# patch embedding layer
x, z = patch_embed(x), patch_embed(z)

# add position embedding
x, z = x + pos_embed_x, z + pos_embed_z

# concat
x = torch.cat([z, x], dim=1)

# transformer layers
for i, blk in enumerate(blocks):

x = blk(x)

# decode the matching result
x = decoder(x)

A.1
Replacement Training
458

We present the pseudocode for the training and testing phases of CompressTracker in Algorithm 2
459

and Algorithm 3, respectively. Additionally, the pseudocode of OSTrack [44] is also shown in
460

Algorithm 1. During training process, we employ Bernoulli sampling to implement a replacement
461

training strategy, while in the test phase, we integrate the student layers and discard the teacher layer.
462

#
Replacement
AUC
Training
Time

1
Random
65.2%
12 h

2
Decouple-300
64.6%
16 h

Table 11: Ablation study on re-
placement training.

#
Replacement
AUC

1
w/ Progressive
65.2%

2
w/o Progressive
64.8%

Table 12: Ablation study on
progressive replacement.

#
Model
Training
Time
1
CompressTracker-4
20 h
2
OSTrack
17 h
3
MixFormerV2-S
120 h

Table 13: Training Time com-
parison.

0.1
0.3
0.5
0.7
0.9
61

62

63

64

65

66

67

AUC %

61.6

62.3

63.7
63.6

63.2

62.8

64.0

64.8

65.0

64.2

63.5

65.3

66.2
66.3

66.0

CompressTracker-3
CompressTracker-4
CompressTracker-6

Figure 4: Ablation study on different replace-
ment probability.

CompressTracker-2
CompressTracker-3
CompressTracker-4
CompressTracker-6
CompressTracker-8
5

10

15

20

25

30

Training Time (h)

6

7

8

12

14
14

16

20

25

29
Naive Training
CompressTracker

Figure 5: Training Time.

12


---Page Break---
Algorithm 2 Pseudocode of CompressTracker for Training in a PyTorch-like style

# z/x: RGB image of template/search region
# patch_embed: patch embedding layer,
# pos_embed_z/pos_embed_z: position embedding for template/search region
# bernoulli_sample: bernoulli sampling function with probability of p
# n_s/n_t: layer number of student/teacher model
# teacher_blocks: transformer block layers of a pretrained teacher
# student_blocks: transformer block layers of student model
# decoder: decoder network

def forward(x, z):

# patch embedding layer
x, z = patch_embed(x), patch_embed(z)

# add position embedding
x, z = x + pos_embed_x, z + pos_embed_z

# concat
x = torch.cat([z, x], dim=1)

# replacement sampling
inference_blocks = []
for i in range(n):

if bernoulli_sample() == 1:

inference_blocks.append(student_blocks[i])
else:

for j in range(n_t//n_s):

inference_blocks.append(teacher_blocks[i*(n_t//n_s) + j])

# randomly replaced transformer layers
for i, blk in enumerate(inference_blocks):

x = blk(x)

# decode the matching result
x = decoder(x)

Algorithm 3 Pseudocode of CompressTracker for Testing in a PyTorch-like style

# z/x: RGB image of template/search region
# patch_embed: patch embedding layer,
# pos_embed_z/pos_embed_z: position embedding for template/search region
# student_blocks: transformer block layers of student model
# decoder: decoder network

def forward(x, z):

# patch embedding layer
x, z = patch_embed(x), patch_embed(z)

# add position embedding
x, z = x + pos_embed_x, z + pos_embed_z

# concat
x = torch.cat([z, x], dim=1)

# transformer layers
for i, blk in enumerate(student_blocks):

x = blk(x)

# decode the matching result
x = decoder(x)

13


---Page Break---
A.2
More Ablation Study
463

We represent more ablation studies on LaSOT to explore the factors contributing to effectiveness of
464

our CompressTracker. Unless otherwise specified, teacher model is OSTrack,and student model has 4
465

encoder layers. The student model is trained for 300 epochs, and the pinit is set as 0.5.
466

Replacement Training. To evaluate the efficiency and effectiveness of our replacement training
467

strategy, we conduct a series of experiments and results are presented in Table 11. ’Random’ denotes
468

our replacement training, and ’Decouple-300’ represents decoupling the training of each stage. Result
469

of # 1 aligns with our replacement training with 300 training epochs, while in # 2, we employ a
470

decoupled training approach for each stage. Initially, we substitute the first stage of the teacher
471

model with its counterpart in the student model, training the first stage for 75 epochs. Subsequently,
472

the first trained stage of the student model is frozen, and the second stage undergoes training for
473

an additional 75 epochs. Following this iterative process, we train the four stages cumulatively
474

over 300 epochs, with an additional 30 epochs for fine-tuning. The ’Decouple-300’ (# 2) approach
475

achieves 64.6% AUC on LaSOT with the same training epochs, marginally lower by 0.6% AUC
476

than our replacement training strategy (# 1). The ’Decouple-300’ approach (# 2) requires a complex,
477

multi-stage trainingalong with supplementary fine-tuning, while our CompressTracker operates on an
478

end-to-end, single-step basis. Besides, the ’Decouple-300’ approach may suffer from suboptimal
479

outcomes at a specific training process, but our CompressTracker can avoid this problem through its
480

unified training manner, which validates the superiority of our replacement training strategy.
481

Replacement Probability. We investigate the impact of replacement probability on the accuracy of
482

student model in Figure 4. We maintain a constant replacement probability instead of implementing
483

the progressive replacement strategy and train the student model with 300 epochs and 30 extra
484

finetuning epochs. It can be observed from Figure 4 that performance is adversely affected when
485

the replacement probability is set either too high or too low. Optimal results are achieved when the
486

replacement probability is within the range of 0.5 to 0.7. Specifically, a too low probability leads to
487

inadequate training, whereas a too high probability may result in the insufficient interaction between
488

teacher model and student tracker. Thus, we set the pinit as 0.5 based on the experiment result.
489

Progressive Replacement. In Table 12, we illustrate the impact of progressive replacement strategy.
490

The first row (# 1) corresponds to the same setting of CompressTracker, while in the second row (# 2)
491

we fix the sampling probability as 0.5 and the student model is trained with 300 epochs followed by
492

30 finetuning epochs. The absence of progressive replacement leads to a performance degradation of
493

0.4% AUC, thereby highlighting the efficacy of our progressive replacement approach.
494

Training Time. We compare the training time of CompressTracker with 500 training epochs across
495

different layers in Figure 5. ’Naive Training’ denotes solely training on groundtruth data with
496

300 epochs, and ’CompressTracker’ represents our proposed training strategy with 500 epochs.
497

The training time is recorded on 8 NVIDIA RTX 3090 GPUs. Besides, the training times of
498

our CompressTracker-4, OSTrack, and MixFormerV2-S are presented in Table 13. Although our
499

CompressTracker requires a longer training time compared to the ’Naive Training’, the increased
500

computational overhead remains within acceptable limits. Moreover, MixFormerV2-S is trained
501

on 8 Nvidia RTX8000 GPUs, and we estimate this will take roughly 80 hours on 8 NVIDIA RTX
502

3090 GPUs based on the relative computational capabilities of these GPUs. The training time of our
503

CompressTracker-4 is significantly less than that of MixFormerV2-S, which validate the efficiency
504

and effectiveness of our framework.
505

14


---Page Break---
NeurIPS Paper Checklist
506

1. Claims
507

Question: Do the main claims made in the abstract and introduction accurately reflect the
508

paper’s contributions and scope?
509

Answer: [Yes]
510

Justification: We summarize our contribution in the abstract and introduction.
511

Guidelines:
512

• The answer NA means that the abstract and introduction do not include the claims
513

made in the paper.
514

• The abstract and/or introduction should clearly state the claims made, including the
515

contributions made in the paper and important assumptions and limitations. A No or
516

NA answer to this question will not be perceived well by the reviewers.
517

• The claims made should match theoretical and experimental results, and reflect how
518

much the results can be expected to generalize to other settings.
519

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
520

are not attained by the paper.
521

2. Limitations
522

Question: Does the paper discuss the limitations of the work performed by the authors?
523

Answer: [Yes]
524

Justification: We discuss the limitations of our work in our maniscript.
525

Guidelines:
526

• The answer NA means that the paper has no limitation while the answer No means that
527

the paper has limitations, but those are not discussed in the paper.
528

• The authors are encouraged to create a separate "Limitations" section in their paper.
529

• The paper should point out any strong assumptions and how robust the results are to
530

violations of these assumptions (e.g., independence assumptions, noiseless settings,
531

model well-specification, asymptotic approximations only holding locally). The authors
532

should reflect on how these assumptions might be violated in practice and what the
533

implications would be.
534

• The authors should reflect on the scope of the claims made, e.g., if the approach was
535

only tested on a few datasets or with a few runs. In general, empirical results often
536

depend on implicit assumptions, which should be articulated.
537

• The authors should reflect on the factors that influence the performance of the approach.
538

For example, a facial recognition algorithm may perform poorly when image resolution
539

is low or images are taken in low lighting. Or a speech-to-text system might not be
540

used reliably to provide closed captions for online lectures because it fails to handle
541

technical jargon.
542

• The authors should discuss the computational efficiency of the proposed algorithms
543

and how they scale with dataset size.
544

• If applicable, the authors should discuss possible limitations of their approach to
545

address problems of privacy and fairness.
546

• While the authors might fear that complete honesty about limitations might be used by
547

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
548

limitations that aren’t acknowledged in the paper. The authors should use their best
549

judgment and recognize that individual actions in favor of transparency play an impor-
550

tant role in developing norms that preserve the integrity of the community. Reviewers
551

will be specifically instructed to not penalize honesty concerning limitations.
552

3. Theory Assumptions and Proofs
553

Question: For each theoretical result, does the paper provide the full set of assumptions and
554

a complete (and correct) proof?
555

Answer: [NA]
556

15


---Page Break---
Justification: We do not include theoretical results.
557

Guidelines:
558

• The answer NA means that the paper does not include theoretical results.
559

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
560

referenced.
561

• All assumptions should be clearly stated or referenced in the statement of any theorems.
562

• The proofs can either appear in the main paper or the supplemental material, but if
563

they appear in the supplemental material, the authors are encouraged to provide a short
564

proof sketch to provide intuition.
565

• Inversely, any informal proof provided in the core of the paper should be complemented
566

by formal proofs provided in appendix or supplemental material.
567

• Theorems and Lemmas that the proof relies upon should be properly referenced.
568

4. Experimental Result Reproducibility
569

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
570

perimental results of the paper to the extent that it affects the main claims and/or conclusions
571

of the paper (regardless of whether the code and data are provided or not)?
572

Answer: [Yes]
573

Justification: We include all the details in our paper, including datasets, model, and training
574

details. Other researchers can reproduce our result easily, and we will release our code once
575

our work is accepted.
576

Guidelines:
577

• The answer NA means that the paper does not include experiments.
578

• If the paper includes experiments, a No answer to this question will not be perceived
579

well by the reviewers: Making the paper reproducible is important, regardless of
580

whether the code and data are provided or not.
581

• If the contribution is a dataset and/or model, the authors should describe the steps taken
582

to make their results reproducible or verifiable.
583

• Depending on the contribution, reproducibility can be accomplished in various ways.
584

For example, if the contribution is a novel architecture, describing the architecture fully
585

might suffice, or if the contribution is a specific model and empirical evaluation, it may
586

be necessary to either make it possible for others to replicate the model with the same
587

dataset, or provide access to the model. In general. releasing code and data is often
588

one good way to accomplish this, but reproducibility can also be provided via detailed
589

instructions for how to replicate the results, access to a hosted model (e.g., in the case
590

of a large language model), releasing of a model checkpoint, or other means that are
591

appropriate to the research performed.
592

• While NeurIPS does not require releasing code, the conference does require all submis-
593

sions to provide some reasonable avenue for reproducibility, which may depend on the
594

nature of the contribution. For example
595

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
596

to reproduce that algorithm.
597

(b) If the contribution is primarily a new model architecture, the paper should describe
598

the architecture clearly and fully.
599

(c) If the contribution is a new model (e.g., a large language model), then there should
600

either be a way to access this model for reproducing the results or a way to reproduce
601

the model (e.g., with an open-source dataset or instructions for how to construct
602

the dataset).
603

(d) We recognize that reproducibility may be tricky in some cases, in which case
604

authors are welcome to describe the particular way they provide for reproducibility.
605

In the case of closed-source models, it may be that access to the model is limited in
606

some way (e.g., to registered users), but it should be possible for other researchers
607

to have some path to reproducing or verifying the results.
608

5. Open access to data and code
609

16


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
610

tions to faithfully reproduce the main experimental results, as described in supplemental
611

material?
612

Answer: [No]
613

Justification: We provide the core pseudocode in appendix, and we will release our code
614

after acceptance.
615

Guidelines:
616

• The answer NA means that paper does not include experiments requiring code.
617

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
618

public/guides/CodeSubmissionPolicy) for more details.
619

• While we encourage the release of code and data, we understand that this might not be
620

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
621

including code, unless this is central to the contribution (e.g., for a new open-source
622

benchmark).
623

• The instructions should contain the exact command and environment needed to run to
624

reproduce the results. See the NeurIPS code and data submission guidelines (https:
625

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
626

• The authors should provide instructions on data access and preparation, including how
627

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
628

• The authors should provide scripts to reproduce all experimental results for the new
629

proposed method and baselines. If only a subset of experiments are reproducible, they
630

should state which ones are omitted from the script and why.
631

• At submission time, to preserve anonymity, the authors should release anonymized
632

versions (if applicable).
633

• Providing as much information as possible in supplemental material (appended to the
634

paper) is recommended, but including URLs to data and code is permitted.
635

6. Experimental Setting/Details
636

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
637

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
638

results?
639

Answer: [Yes]
640

Justification: We include all the details in our paper, including datasets, model, and train-
641

ing details. Other researchers can reproduce our result easily. We also provide the core
642

pseudocode in appendix.
643

Guidelines:
644

• The answer NA means that the paper does not include experiments.
645

• The experimental setting should be presented in the core of the paper to a level of detail
646

that is necessary to appreciate the results and make sense of them.
647

• The full details can be provided either with the code, in appendix, or as supplemental
648

material.
649

7. Experiment Statistical Significance
650

Question: Does the paper report error bars suitably and correctly defined or other appropriate
651

information about the statistical significance of the experiments?
652

Answer: [No]
653

Justification: Our paper does not report error bars.
654

Guidelines:
655

• The answer NA means that the paper does not include experiments.
656

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
657

dence intervals, or statistical significance tests, at least for the experiments that support
658

the main claims of the paper.
659

17


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
660

example, train/test split, initialization, random drawing of some parameter, or overall
661

run with given experimental conditions).
662

• The method for calculating the error bars should be explained (closed form formula,
663

call to a library function, bootstrap, etc.)
664

• The assumptions made should be given (e.g., Normally distributed errors).
665

• It should be clear whether the error bar is the standard deviation or the standard error
666

of the mean.
667

• It is OK to report 1-sigma error bars, but one should state it. The authors should
668

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
669

of Normality of errors is not verified.
670

• For asymmetric distributions, the authors should be careful not to show in tables or
671

figures symmetric error bars that would yield results that are out of range (e.g. negative
672

error rates).
673

• If error bars are reported in tables or plots, The authors should explain in the text how
674

they were calculated and reference the corresponding figures or tables in the text.
675

8. Experiments Compute Resources
676

Question: For each experiment, does the paper provide sufficient information on the com-
677

puter resources (type of compute workers, memory, time of execution) needed to reproduce
678

the experiments?
679

Answer: [Yes]
680

Justification: We provide the type of GPU we used and time of execution
681

Guidelines:
682

• The answer NA means that the paper does not include experiments.
683

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
684

or cloud provider, including relevant memory and storage.
685

• The paper should provide the amount of compute required for each of the individual
686

experimental runs as well as estimate the total compute.
687

• The paper should disclose whether the full research project required more compute
688

than the experiments reported in the paper (e.g., preliminary or failed experiments that
689

didn’t make it into the paper).
690

9. Code Of Ethics
691

Question: Does the research conducted in the paper conform, in every respect, with the
692

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
693

Answer: [Yes]
694

Justification: Our research conforms with the NeurIPS Code of Ethics.
695

Guidelines:
696

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
697

• If the authors answer No, they should explain the special circumstances that require a
698

deviation from the Code of Ethics.
699

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
700

eration due to laws or regulations in their jurisdiction).
701

10. Broader Impacts
702

Question: Does the paper discuss both potential positive societal impacts and negative
703

societal impacts of the work performed?
704

Answer: [Yes]
705

Justification: We discuss the potential negative societal impacts.
706

Guidelines:
707

• The answer NA means that there is no societal impact of the work performed.
708

• If the authors answer NA or No, they should explain why their work has no societal
709

impact or why the paper does not address societal impact.
710

18


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
711

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
712

(e.g., deployment of technologies that could make decisions that unfairly impact specific
713

groups), privacy considerations, and security considerations.
714

• The conference expects that many papers will be foundational research and not tied
715

to particular applications, let alone deployments. However, if there is a direct path to
716

any negative applications, the authors should point it out. For example, it is legitimate
717

to point out that an improvement in the quality of generative models could be used to
718

generate deepfakes for disinformation. On the other hand, it is not needed to point out
719

that a generic algorithm for optimizing neural networks could enable people to train
720

models that generate Deepfakes faster.
721

• The authors should consider possible harms that could arise when the technology is
722

being used as intended and functioning correctly, harms that could arise when the
723

technology is being used as intended but gives incorrect results, and harms following
724

from (intentional or unintentional) misuse of the technology.
725

• If there are negative societal impacts, the authors could also discuss possible mitigation
726

strategies (e.g., gated release of models, providing defenses in addition to attacks,
727

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
728

feedback over time, improving the efficiency and accessibility of ML).
729

11. Safeguards
730

Question: Does the paper describe safeguards that have been put in place for responsible
731

release of data or models that have a high risk for misuse (e.g., pretrained language models,
732

image generators, or scraped datasets)?
733

Answer: [Yes]
734

Justification: We describe the safeguards in our paper.
735

Guidelines:
736

• The answer NA means that the paper poses no such risks.
737

• Released models that have a high risk for misuse or dual-use should be released with
738

necessary safeguards to allow for controlled use of the model, for example by requiring
739

that users adhere to usage guidelines or restrictions to access the model or implementing
740

safety filters.
741

• Datasets that have been scraped from the Internet could pose safety risks. The authors
742

should describe how they avoided releasing unsafe images.
743

• We recognize that providing effective safeguards is challenging, and many papers do
744

not require this, but we encourage authors to take this into account and make a best
745

faith effort.
746

12. Licenses for existing assets
747

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
748

the paper, properly credited and are the license and terms of use explicitly mentioned and
749

properly respected?
750

Answer: [Yes]
751

Justification: All the code, data and models we used are credited.
752

Guidelines:
753

• The answer NA means that the paper does not use existing assets.
754

• The authors should cite the original paper that produced the code package or dataset.
755

• The authors should state which version of the asset is used and, if possible, include a
756

URL.
757

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
758

• For scraped data from a particular source (e.g., website), the copyright and terms of
759

service of that source should be provided.
760

• If assets are released, the license, copyright information, and terms of use in the
761

package should be provided. For popular datasets, paperswithcode.com/datasets
762

has curated licenses for some datasets. Their licensing guide can help determine the
763

license of a dataset.
764

19


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
765

the derived asset (if it has changed) should be provided.
766

• If this information is not available online, the authors are encouraged to reach out to
767

the asset’s creators.
768

13. New Assets
769

Question: Are new assets introduced in the paper well documented and is the documentation
770

provided alongside the assets?
771

Answer: [NA]
772

Justification: We do not release new assets.
773

Guidelines:
774

• The answer NA means that the paper does not release new assets.
775

• Researchers should communicate the details of the dataset/code/model as part of their
776

submissions via structured templates. This includes details about training, license,
777

limitations, etc.
778

• The paper should discuss whether and how consent was obtained from people whose
779

asset is used.
780

• At submission time, remember to anonymize your assets (if applicable). You can either
781

create an anonymized URL or include an anonymized zip file.
782

14. Crowdsourcing and Research with Human Subjects
783

Question: For crowdsourcing experiments and research with human subjects, does the paper
784

include the full text of instructions given to participants and screenshots, if applicable, as
785

well as details about compensation (if any)?
786

Answer: [NA]
787

Justification: This paper does not involve crowdsourcing nor research with human subjects.
788

Guidelines:
789

• The answer NA means that the paper does not involve crowdsourcing nor research with
790

human subjects.
791

• Including this information in the supplemental material is fine, but if the main contribu-
792

tion of the paper involves human subjects, then as much detail as possible should be
793

included in the main paper.
794

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
795

or other labor should be paid at least the minimum wage in the country of the data
796

collector.
797

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
798

Subjects
799

Question: Does the paper describe potential risks incurred by study participants, whether
800

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
801

approvals (or an equivalent approval/review based on the requirements of your country or
802

institution) were obtained?
803

Answer: [NA]
804

Justification: This paper does not involve crowdsourcing nor research with human subjects.
805

• The answer NA means that the paper does not involve crowdsourcing nor research with
806

human subjects.
807

• Depending on the country in which research is conducted, IRB approval (or equivalent)
808

may be required for any human subjects research. If you obtained IRB approval, you
809

should clearly state this in the paper.
810

• We recognize that the procedures for this may vary significantly between institutions
811

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
812

guidelines for their institution.
813

• For initial submissions, do not include any information that would break anonymity (if
814

applicable), such as the institution conducting the review.
815

20


---Page Break---
