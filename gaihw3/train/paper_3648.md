End-to-End Autonomous Driving without Costly
Modularization and 3D Manual Annotation

Anonymous Author(s)
Affiliation
Address
email

Abstract

We propose UAD, a method for vision-based end-to-end autonomous driving
1

(E2EAD), achieving the best open-loop evaluation performance in nuScenes, mean-
2

while showing robust closed-loop driving quality in CARLA. Our motivation stems
3

from the observation that current E2EAD models still mimic the modular archi-
4

tecture in typical driving stacks, with carefully designed supervised perception
5

and prediction subtasks to provide environment information for oriented planning.
6

Although achieving groundbreaking progress, such design has certain drawbacks:
7

1) preceding subtasks require massive high-quality 3D annotations as supervision,
8

posing a significant impediment to scaling the training data; 2) each submodule
9

entails substantial computation overhead in both training and inference. To this end,
10

we propose UAD, an E2EAD framework with an unsupervised1 proxy to address
11

all these issues. Firstly, we design a novel Angular Perception Pretext to eliminate
12

the annotation requirement. The pretext models the driving scene by predicting the
13

angular-wise spatial objectness and temporal dynamics, without manual annota-
14

tion. Secondly, a self-supervised training strategy, which learns the consistency of
15

the predicted trajectories under different augment views, is proposed to enhance
16

the planning robustness in steering scenarios. Our UAD achieves 38.7% relative
17

improvements over UniAD on the average collision rate in nuScenes and surpasses
18

VAD for 6.40 points on the driving score in CARLA’s Town05 Long benchmark.
19

Moreover, the proposed method only consumes 44.3% training resources of UniAD
20

and runs 3.4× faster in inference. Our innovative design not only for the first time
21

demonstrates unarguable performance advantages over supervised counterparts,
22

but also enjoys unprecedented efficiency in data, training, and inference.
23

1
Introduction
24

Recent decades have witnessed breakthrough achievements in autonomous driving. The end-to-
25

end paradigm, which seeks to integrate perception, prediction, and planning tasks into a unified
26

framework, stands as a representative branch [33, 1, 39, 3, 35, 21, 22]. The latest advances in end-to-
27

end autonomous driving significantly piqued researchers’ interest [21, 22]. However, handcrafted and
28

resource-intensive supervised sub-tasks for perception and prediction, which have previously proved
29

their utility in environment modeling [35, 3, 20], continue to be indispensable, as shown in Fig. 1a.
30

Then what insights have we gained from the recent advances? It has come to our attention that one of
31

the most enlightening innovations lies in the Transformer-based pipeline, in which the queries act
32

as a connective thread, seamlessly bridging various tasks. Besides, the capability for environment
33

modeling has also seen a significant boost, primarily due to complicated interactions of supervised
34

1Following [30, 4], here we consider the methods as “unsupervised” ones as long as no manual annotation is
used and required in the target task or domain.

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
(a)
(b)
Figure 1: (a) End-to-end autonomous driving paradigms. 1) The vanilla architecture that directly
predicts control command. 2) The modularized design that combines various preceding tasks. 3)
Our proposed framework with unsupervised pretext task. (b) Comparison of training cost, inference
speed and average L2 error between our method and [21, 22] on 8 NVIDIA Tesla A100 GPUs.

sub-tasks. However, every coin has two sides. In comparison to the vanilla design [33] (see Fig. 1a),
35

modularized methods incur unavoidable computation and annotation overhead. As illustrated in
36

Fig. 1b, the training of the recent method UniAD [21] takes 48 GPU days while running at only 2.1
37

frames per second (FPS). Moreover, modules in existing perception and prediction design require large
38

quantities of high-quality annotated data. The financial overhead for human annotation significantly
39

impedes the scalability of such modularized methods with supervised subtasks to leverage massive
40

data. As proved by large foundation models [24, 31], scaling up the data volume is the key to bringing
41

the model capabilities to the next level. Thus we ask ourselves the question: Is it viable to devise an
42

efficient and robust E2EAD framework while alleviating the reliance on 3D annotation?
43

In this work, we show the answer is affirmative by proposing an innovative Unsupervised pretext task
44

for end-to-end Autonomous Driving (UAD), which seeks to efficiently model the environment. The
45

pretext task consists of an angular-wise perception module to learn spatial information by predicting
46

the objectness of each sector region in BEV space, and an angular-wise dreaming decoder to absorb
47

temporal knowledge by predicting inaccessible future states. The introduced angular queries link
48

the two modules as a whole pretext task to perceive the driving scene. Notably, our method shines
49

by completely eliminating the annotation requirement for perception and prediction. Such data
50

efficiency is not attainable for current methods with complex supervised modularization [21, 22]. The
51

supervision for learning spatial objectness is obtained by projecting the 2D region of interests (ROIs)
52

from an off-the-shelf open-set detector [28] to BEV space. While utilizing the publicly available open-
53

set 2D detector pre-trained with manual annotation from other domains (e.g. COCO [27]), we avoid
54

the need for any additional 3D labels within our paradigm and target domains (e.g. nuScenes [2] and
55

CARLA [11]), thereby creating a pragmatically unsupervised setting [30]. Furthermore, we introduce
56

a self-supervised direction-aware learning strategy to train the planning model. Specifically, the
57

visual observations are augmented with different rotation angles, and the consistency loss is applied
58

to the predictions for robust planning. Without bells and whistles, the proposed UAD outperforms
59

UniAD for 0.13m in nuScenes Avg. L2 error, and surpasses VAD [22] for 9.92 points in CARLA
60

route completion score. Such unprecedented performance gain is achieved with a 3.4× inference
61

speed, a mere 44.3% training budget of UniAD, and zero annotations, as illustrated in Fig. 1b.
62

In summary, our contributions are as follows: 1) We propose an unsupervised pretext task to discard
63

the requirement of 3D manual annotation in end-to-end autonomous driving, potentially making
64

it more feasible to scale the training data to billions level without any labeling overload; 2) We
65

introduce a novel self-supervised direction-aware learning strategy to maximize the consistency of the
66

predicted trajectories under different augment views, which enhances planning robustness in steering
67

scenarios; 3) Our method shows superiority in both open- and closed-loop evaluation compared with
68

other vision-based E2EAD methods, with much lower computation and annotation cost.
69

2
Related Work
70

2.1
End-to-End Autonomous Driving
71

End-to-end autonomous driving can be dated back to 1988, when the ALVINN [33] proposed by
72

Carnegie Mellon University could successfully navigate a vehicle over 400 meters. After that, to
73

2


---Page Break---
BEV
Encoder

BEV Feature

OS 2D 
Detector

2D Boxes

Dreaming

Decoder

BEV Object Mask

Angular 
Partition

Angular Objectness

Label

Angular BEV Feature

Angular Query

Perception

Predict
View
Transform

Objectness

Loss

Cross
Attention

Ego Query

Direction

Predict

Planning

Predict

Direction-Augmented

GT Ego Trajectory

Multi-view Images

Direction
Augment

Direction

Loss

Imitation

Loss

Turn 
Right

Driving
Command

F

F

F

F

270
180
90

270
180
90

Angular Perception Pretext
Direction Aware Planning

K

K

Figure 2: The architecture of our UAD. The inference pipeline is marked by black arrows with blue
background, which plans ego trajectory based on the input multi-view images. The training pipeline
consists of Angular Perception Pretext (orange arrows with khaki background) and Direction-Aware
Planning (orange arrows with purple background). “F” in BEV feature indicates the driving direction.

improve the robustness of E2EAD, a series of modern approaches such as NEAT [6], P3 [35],
74

MP3 [3], ST-P3 [20] introduce the design of more dedicated modularization, which integrate auxiliary
75

information such as HD maps, and additional tasks like bird’s-eye view (BEV) segmentation. Most re-
76

cently, embracing advanced architectures like Transfromer [37] and visual occupancy prediction [29],
77

UniAD [21] and VAD [22] demonstrate impressive performance in open-loop evaluation. In this work,
78

instead of integrating complex supervised modular sub-tasks, we innovatively propose another path
79

proving that an efficient unsupervised pretext task without any human annotation like 3D bounding
80

boxes and point cloud categories, can achieve even superior performance than recent state-of-the-arts.
81

2.2
World Model
82

In pursuit of understanding the dynamic changes in environments, researchers in the fields of gaming
83

and robotics have proposed various world models [13, 14, 15, 16]. Recently, the autonomous driving
84

community introduces world models for safer maneuvering [32, 18, 12, 38]. MILE [18] considers
85

the environment as a high-level embedding and tends to predict its future state with historical
86

observations. Drive-WM [38] proposes a framework to integrate world models with existing E2E
87

methods to improve planning robustness. In this work, we propose an auto-regressive mechanism,
88

tailored to our unsupervised pretext, to capture angular-wise temporal dynamics within each sector.
89

3
Method
90

3.1
Overview
91

As illustrated in Fig. 2, our UAD framework consists of two essential components: 1) the Angular
92

Perception Pretext, aims to liberate E2EAD from costly modularized tasks in an unsupervised fashion;
93

2) the Direction-Aware Planning, learns self-supervised consistency of the augmented trajectories.
94

Specifically, UAD first models the driving environment with the pretext. The spatial knowledge
95

is acquired by estimating the objectness of each sector region within the BEV space. The angular
96

queries, each responsible for a sector, are introduced to extract features and predict the objectness.
97

The supervision label is generated by projecting the 2D regions of interests (ROIs) to the BEV space,
98

which are predicted with an available open-set detector GroundingDINO [28]. This way not only
99

eliminates the 3D annotation requirement, but also greatly reduces the training budget. Moreover, as
100

driving is inherently a dynamic and continuous process, we thus propose an angular-wise dreaming
101

decoder to encode the temporal knowledge. The dreaming decoder can be viewed as an augmented
102

world model [13] capable of auto-regressively predicting the future states.
103

Subsequently, direction-aware planning is introduced to train the planning module. The raw BEV
104

feature is augmented with different rotation angles, yielding rotated BEV representations and ego
105

trajectories. We apply self-supervised consistency loss to the predicted trajectories of each augmented
106

view, which is expected to improve the robustness for directional change and input noises. The
107

learning strategy can also be regarded as a novel data augmentation technique customized for end-to-
108

end autonomous driving, which enhances the diversity of trajectory distribution.
109

3


---Page Break---
2D Boxes

BEV Object Mask

Multi-view Images
Multi-view Image Masks

Sampling
Detect

Angular Objectness

Label

K

(a)

Angular Query

Cross
Attention

μ

σ

Prior Distribution

GRU

Updated Angular Query

Distribution 

Sampling

Posterior Distribution

Dreaming Loss

Update

Output

Dreaming Layer

×Planning Steps

Angular BEV Feature

K

μ

σ

(b)
Figure 3: (a) Label generation for angular perception pretext. (b) Illustration of dreaming decoder.

3.2
Angular Perception Pretext
110

Spatial Representation Learning. Our model attempts to acquire spatial knowledge of the driving
111

scene by predicting the objectness of each sector region within the BEV space. Specifically, taking
112

multi-view images {Ii ∈RHi×Wi×3} as input, the BEV encoder [25] first extracts visual information
113

into the BEV feature Fb ∈RHb×Wb×C. Then, Fb is partitioned into K sectors with a uniform angle θ
114

centered around ego car. Each sector contains several feature points in BEV space. Denoting feature
115

of a sector as f ∈RN×C, where N is the maximum number of feature points in all sectors, we derive
116

angular BEV feature Fa ∈RK×N×C. Zero-padding is applied on sectors with fewer than N points.
117

Then why do we partition the rectangular BEV feature to angular-wise formatting? The underlying
118

reason is that, in the absence of depth information, the region in BEV space corresponding to an ROI
119

in 2D image is a sector. As illustrated in Fig. 3a, by projecting 3D sampling points to images and
120

verifying their presence in 2D ROIs, a BEV object mask M∈RHb×Wb×1 is generated, representing
121

the objectness in BEV space. Specifically, the sampling points falling within 2D ROIs are set to 1,
122

while the others are 0. It is noticed that the positive sectors are irregularly and sparsely distributed in
123

BEV space. To make the objectness label more compact, similar to the BEV feature partition, we
124

uniformly divide M into K equal parts. The segments overlapped with positive sectors are assigned
125

with 1, constituting the angular objectness label Yobj ∈RK×1. Thanks to the rapid development
126

of open-set detection, it’s now convenient to obtain 2D ROIs for the input multi-view images by
127

feeding the pre-defined prompts (e.g., vehicle, pedestrian, and barrier) to a 2D open-set detector like
128

GroundingDINO [28]. Such design is the key in reducing annotation cost and scaling up the dataset.
129

To predict the objectness score of each sector, we define angular queries Qa ∈RK×C to summarize
130

Fa. Each angular query qa ∈R1×C in Qa will interact with corresponding f by cross attention [37],
131

qa = CrossAttention(qa, f),
(1)

Finally, we map Qa to the objectness scores Pa ∈RK×1 with a linear layer, which is supervised by
132

Yobj with binary cross-entropy loss (denoted as Lspat).
133

Temporal Representation Learning. We propose to capture the temporal information of driving
134

scenarios with the angular-wise dreaming decoder. As shown in Fig. 3b, the decoder auto-regressively
135

learns transition dynamics of each sector in a similar way of world model [14]. Assuming the planning
136

module predicts the trajectories of future T steps, the dreaming decoder accordingly comprises T
137

layers, where each updates the input angular queries Qa and angular BEV feature Fa based on the
138

learned temporal dynamics. At step t, the queries Qt−1
a
first grasp environmental dynamics from the
139

observation feature Ft
a with a gated recurrent unit (GRU) [7], which generates Qt
a (hidden state),
140

Qt
a = GRU(Qt−1
a
, Ft
a),
(2)

In previous world models, the hidden state Q is solely used for perceiving observed scenes. The
141

GRU iteration thus ends at t with the final observation Ft
a. In our framework, Q is also used for
142

predicting ego trajectories in the future. Yet, the future observation, e.g., Ft+1
a
, is unavailable, as
143

the world model [14] is designed for forecasting the future with only current observation. To obtain
144

Qt+1
a
, we first propose to update Ft
a to provide pseudo observations ˆFt+1
a
,
145

ˆFt+1
a
= CrossAttention(Ft
a, Qt
a).
(3)

Then Qt+1
a
can be generated with Eq. 2 and inputs of ˆFt+1
a
and Qt
a.
146

Following the loss design in world models [14, 15, 16], we respectively map Qt−1
a
and Qt
a to distri-
147

butions of {µt−1
a
, σt−1
a
∈RK×C} and {µt
a, σt
a ∈RK×C}, and then minimize their KL divergence.
148

4


---Page Break---
For the prior distribution from Qt−1
a
, it’s regarded as a prediction of the future dynamics without
149

observation. In contrast, the posterior distribution from Qt
a represents the future dynamics with the
150

observation Ft
a. The KL divergence between the two distributions measures the gap between the
151

imagined future (prior) and the true future (posterior). We expect to enhance the capability of future
152

prediction for long-term driving safety, which is realized by optimizing the dreaming loss Ldrm,
153

Ldrm = KL({µt
a, σt
a}||{µt−1
a
, σt−1
a
}),
(4)

3.3
Direction-Aware Planning
154

Planning Head. The outputs of angular perception pretext contain a group of angular queries
155

{Qt
a (t = 1, ..., T)}. For planning, we correspondingly initialize T ego queries {Qt
ego ∈R1×C (t =
156

1, ..., T)} to extract planning-relevant information and predict the ego trajectory of each future time
157

step. The interaction between ego queries and angular queries is performed with cross attention,
158

Qt
ego = CrossAttention(Qt
ego, Qt
a).
(5)

The output ego queries {Qt
ego} are then used to predict the ego trajectories of future T steps.
159

Following previous works [21, 22], a high-level driving signal c (turn left, turn right or go straight) is
160

provided as prior knowledge. The planning head takes the concatenated ego feature Fego ∈RT ×C
161

from {Qt
ego} and the driving command c as inputs, and outputs the planning trajectory Ptraj ∈RT ×2,
162

Ptraj = PlanHead(Fego, c),
(6)
where the PlanHead is the same as UniAD [21]. We apply L1 loss to minimize the distance between
163

the predicted ego trajectory Ptraj and the ground truth Gtraj, denoted as Limi. Notably, Gtraj is easy
164

to obtain, and manual annotation is not required in practical scenarios.
165

Directional Augmentation. Observed that the training data is predominated by the go straight
166

scenarios, we propose a directional augmentation strategy to balance the distribution. As shown
167

in Fig. 4, the BEV feature Fb is rotated with different angles r ∈R = {90◦, 180◦, 270◦}, yielding
168

BEV Feature/Mask

F

Planner

F

F
F

Rotate

Planner

Pred Trajectory

Pred Trajectory

Consistency

Loss

GT Trajectory

GT Trajectory

Rotate

Direction
Augment
Imitation

Loss

Imitation

Loss

Figure 4: Illustration of direction-aware learning strategy.

the rotated representations {Fr
b}. The
169

augmented features will also be used for
170

the pretext and planning task, and super-
171

vised by the aforementioned loss func-
172

tions (e.g., Lspat). Notably, the BEV ob-
173

ject mask M and the ground truth ego
174

trajectory Gtraj are also rotated to pro-
175

vide corresponding supervision labels.
176

Furthermore, we propose an auxiliary task to enhance the steering capability. In specific, we predict
177

the planning direction that the ego car intends to maneuver (i.e., left, straight or right) based on the
178

ego query Qt
ego, which is mapped to the probabilities of three directions Pt
dir ∈R1×3. The direction
179

label Yt
dir is generated by comparing the x-axis value of ground truth Gt
traj(x) with the threshold
180

δ. Specifically, Yt
dir is assigned to straight if −δ < Gt
traj(x) < δ, otherwise Yt
dir = left/right
181

for Gt
traj(x) ⩽−δ/Gt
traj(x) ⩾δ, respectively. We use the cross-entropy loss to minimize the gap
182

between the direction prediction Pt
dir and the direction label Yt
dir, denoted as Ldir.
183

Directional Consistency. Tailored to the introduced directional augmentation, we propose a direc-
184

tional consistency loss to improve the augmented plan training in a self-supervised manner. It should
185

be noticed that the augmented trajectory predictions Pt,r
traj incorporate the same scene information as
186

the original one Pt,r=0
traj , i.e., BEV features with different rotation angles. Therefore, it’s reasonable
187

to consider the consistency among the predictions and regulate the noises caused by the rotation. The
188

planning head is expected to be more robust to directional change and input distractors. Specifically,
189

Pt,r
traj are first rotated back to the original scene direction, then L1 loss is applied with Pt,r=0
traj ,
190

Lcons =
1
T ·|R|
PT
t=1
PR
r ||Rot(Pt,r
traj) −Pt,r=0
traj ||1,
(7)

where Rot is the inverse rotation.
191

To summarize, the overall objective for our UAD contains spatial objectness loss, dreaming loss from
192

the pretext, and imitation learning loss, direction loss, consistency loss from the planning task,
193

L = ω1Lspat + ω2Ldrm + ω3Limi + ω4Ldir + ω5Lcons,
(8)
where ω1, ω2, ω3, ω4, ω5 are the weight coefficients.
194

5


---Page Break---
Table 1: Open-loop planning performance in nuScenes [2]. † indicates LiDAR-based method and ‡
denotes TemAvg evaluation protocol used in VAD and ST-P3 (see Eq. 9 for details). ⋄means using
ego status in the planning module and calculating collision rates following BEV-Planner [26].

Method
Tasks with 3D annotation
L2 (m) ↓
Collision (%) ↓
Intersection (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.
1s
2s
3s
Avg.
FPS

NMP† [39]
Det & Motion
-
-
2.31
-
-
-
1.92
-
-
-
-
-
-
SA-NMP† [39]
Det & Motion
-
-
2.05
-
-
-
1.59
-
-
-
-
-
-
FF† [19]
Occ
0.55
1.20
2.54
1.43
0.06
0.17
1.07
0.43
-
-
-
-
-
EO† [23]
Occ
0.67
1.36
2.78
1.60
0.04
0.09
0.88
0.33
-
-
-
-
-

ST-P3 [20]
Det & Map
1.72
3.26
4.86
3.28
0.44
1.08
3.01
1.51
2.53
8.17
14.4
8.37
1.8
UniAD [21]
Det&Track&Map&Motion&Occ
0.48
0.96
1.65
1.03
0.05
0.17
0.71
0.31
0.21
1.32
3.63
1.72
2.1
VAD-Tiny [22]
Det & Map & Motion
0.60
1.23
2.06
1.30
0.33
1.33
2.21
1.29
0.94
3.22
7.65
3.94
17.6
VAD-Base [22]
Det & Map & Motion
0.54
1.15
1.98
1.22
0.10
0.24
0.96
0.43
0.60
2.38
5.18
2.72
5.3
OccNet [36]
Det & Map & Occ
1.29
2.13
2.99
2.14
0.21
0.59
1.37
0.72
-
-
-
-
3.3
UAD-Tiny (Ours)
None
0.47
0.99
1.71
1.06
0.08
0.39
0.90
0.46
0.24
1.15
3.12
1.50
18.9
UAD (Ours)
None
0.39
0.81
1.50
0.90
0.01
0.12
0.43
0.19
0.13
0.88
2.66
1.22
7.2

ST-P3‡ [20]
Det & Map
1.33
2.11
2.90
2.11
0.23
0.62
1.27
0.71
2.53
8.17
14.4
8.37
1.8
UniAD‡ [21]
Det&Track&Map&Motion&Occ
0.44
0.67
0.96
0.69
0.04
0.08
0.23
0.12
0.21
1.32
3.63
1.72
2.1
VAD-Base‡ [22]
Det & Map & Motion
0.41
0.70
1.05
0.72
0.07
0.17
0.41
0.22
0.60
2.38
5.18
2.72
5.3
Drive-WM‡ [38]
Det & Map
0.43
0.77
1.20
0.80
0.10
0.21
0.48
0.26
-
-
-
-
-
UAD‡ (Ours)
None
0.28
0.41
0.65
0.45
0.01
0.03
0.14
0.06
0.13
0.88
2.66
1.22
7.2

UniAD‡⋄[21]
Det&Track&Map&Motion&Occ
0.20
0.42
0.75
0.46
0.02
0.25
0.84
0.37
0.20
1.33
3.24
1.59
2.1
VAD-Base‡⋄[22]
Det & Map & Motion
0.17
0.34
0.60
0.37
0.04
0.27
0.67
0.33
0.21
2.13
5.06
2.47
5.3
BEV-Planner‡⋄[26]
None
0.16
0.32
0.57
0.35
0.00
0.29
0.73
0.34
0.35
2.62
6.51
3.16
-
UAD‡⋄(Ours)
None
0.13
0.28
0.48
0.30
0.00
0.12
0.55
0.22
0.10
0.80
2.48
1.13
7.2

4
Experiment
195

4.1
Experimental Setup
196

We conduct experiments in nuScenes [2] for open-loop evaluation, that contains 40,157 samples,
197

of which 6,019 ones are used for evaluation. Following previous works [20, 21, 22], we adopt the
198

metrics of L2 error (in meters) and collision rate (in percentage). Notably, the intersection rate with
199

road boundary (in percentage), proposed in BEV-Planner [26], is also included for evaluation. For
200

the closed-loop setting, we follow previous works [34, 20] to perform evaluation in the Town05 [34]
201

benchmark of the CARLA simulator [11]. Route completion (in percentage) and driving score (in
202

percentage) are used as the evaluation metrics. We adopt the query-based view transformer [25] to
203

learn BEV features from multi-view images. The confidence threshold of the open-set 2D detector
204

is set to 0.35 to filter unreliable predictions. The angle θ to partition the BEV space is set to 4◦
205

(K=360◦/4◦), and the default threshold δ is 1.2m (see Sec. 3.3). The weight coefficients in Eq. 8 are
206

set to 2.0, 0.1, 1.0, 2.0, 1.0. Our model is trained for 24 epochs on 8 NVIDIA Tesla A100 GPUs with
207

a batch size of 1 per GPU. Other settings follow UniAD [21] unless otherwise specified.
208

We observed that ST-P3 [20] and VAD [22] adopt different open-loop evaluation protocols (L2 error
209

and collision rate) from UniAD in their official codes. We denote the setting in ST-P3 and VAD as
210

TemAvg and the one in UniAD as NoAvg, respectively. In specific, the TemAvg protocol calculates
211

metrics by averaging the performances from 0.5s to the corresponding timestamp. Taking the L2
212

error at 2s as an example, the calculation in TemAvg is
213

L2@2s = Avg(l20.5s, l21.0s, l21.5s, l22.0s),
(9)

where Avg is the average operation and 0.5s is the time interval between two consecutive annotated
214

frames in nuScenes [2]. For NoAvg protocol, L2@2s = l22.0s.
215

4.2
Comparison with State-of-the-arts
216

Open-loop Evaluation. Tab. 1 presents the performance comparison in terms of L2 error, collision
217

rate, intersection rate with road boundary, and FPS. Since ST-P3 and VAD adopt different evaluation
218

protocols from UniAD to compute L2 error and collision rate (see Sec. 4.1), we respectively calculate
219

the results under different settings, i.e., NoAvg and TemAvg. As shown in Tab. 1, the proposed UAD
220

achieves superior planning performance over UniAD and VAD on all metrics, while running faster.
221

Notably, our UAD obtains 39.4% and 55.2% relative improvements on Collision@3s compared
222

with UniAD and VAD under the NoAvg evaluation protocol (e.g., 39.4%=(0.71%-0.43%)/0.71%),
223

demonstrating the longtime robustness of our method. Moreover, UAD runs at 7.2FPS, which is 3.4×
224

and 1.4× faster than UniAD and VAD-Base, respectively, verifying the efficiency of our framework.
225

Surprisingly, our tiny version, UAD-Tiny, which aligns the settings of backbone, image size, and BEV
226

6


---Page Break---
Table 2: Closed-loop evaluation in the
CARLA simulator [11].
† denotes the
LiDAR-based method.

Method

Town05 Short
Town05 Long
Driving
Score ↑
Route
Completion ↑Driving
Score ↑
Route
Completion ↑

CILRS [8]
7.47
13.40
3.68
7.19
LBC [5]
30.97
55.01
7.05
32.09
Transfuser† [34]
54.52
78.41
33.15
56.36
ST-P3 [20]
55.14
86.74
11.45
83.15
VAD-Base [22]
64.29
87.26
30.31
75.20
UAD (Ours)
67.83
91.05
36.71
85.12

Table 3: Ablation on the loss functions. We evaluate the
influence of each designed module by applying corre-
sponding loss.

# Lspat Ldrm Ldir Lcons Limi
L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
-
-
-
-
✓
1.20 3.04 5.30 3.18 0.83 1.33 5.13 2.43
②
✓
-
-
-
✓
0.44 0.93 1.64 1.00 0.30 0.56 1.28 0.71
③
-
✓
-
-
✓
0.51 1.12 1.97 1.20 0.71 1.13 2.71 1.52
④
-
-
✓
-
✓
0.83 1.57 2.40 1.60 0.79 1.29 3.89 1.99
⑤
-
-
-
✓
✓
0.59 1.30 2.34 1.41 0.76 1.25 3.47 1.83
⑥
✓
✓
✓
✓
✓
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19

Table 4: Ablation on the dreaming decoder.

# Circular
Update
Dreaming
Loss

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
-
-
0.98 1.73 2.74 1.82 0.43 0.85 1.71 1.00
②
✓
-
0.50 0.98 1.87 1.12 0.27 0.60 1.37 0.75
③
-
✓
0.44 0.96 1.73 1.04 0.08 0.35 1.13 0.52
④
✓
✓
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19

Table 5: Ablation on direction-aware learning strategy.

# Directional
Augment
Directional
Consistency

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
-
-
0.42 0.88 1.61 0.97 0.05 0.18 0.73 0.32
②
✓
-
0.41 0.83 1.53 0.92 0.05 0.23 0.68 0.32
③
✓
✓
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19

resolution in VAD-Tiny, runs at the fastest speed of 18.9FPS while clearly outperforming VAD-Tiny
227

and even achieving comparable performance with VAD-Base. This again proves the superiority of
228

our design. More detailed runtime comparisons and analyses are presented in the appendix. We adopt
229

the NoAvg evaluation protocol in the following ablation experiments unless otherwise specified.
230

Recent works discuss the effect of using ego status in the planning module [22, 26]. Following this
231

trend, we also fairly compare the ego status equipped version of our model with these works. It shows
232

that the superiority of our UAD is still preserved, which also achieves the best performance against
233

the compared methods. Moreover, BEV-Planner [26] introduces a new metric named “interaction”
234

for better evaluating the performance of E2EAD methods. As shown in Tab. 1, our model obtains
235

the average interaction rate of 1.13%, obviously outperforming other methods. This again proves
236

the effectiveness of our UAD. On the other hand, this demonstrates the importance of designing a
237

suitable pretext for perceiving the environment. Only using ego status is not enough for safe driving.
238

Closed-loop Evaluation. The simulation results in CARLA [11] are shown in Tab. 2. Our UAD
239

achieves better performance compared with recent E2E planners ST-P3 [20] and VAD [22] in all
240

scenarios, proving the effectiveness. Notably, on challenging Town05 Long benchmark, UAD greatly
241

outperforms recent E2E method VAD by 6.40 points on the driving score and 9.92 points on route
242

completion, respectively. This proves the reliability of our UAD for long-term autonomous driving.
243

4.3
Component-wise Ablation
244

Loss Functions. We first analyze the influence of different loss functions that correspond to the
245

proposed pretext task and self-supervised trajectory learning strategy. The experiments are conducted
246

on the validation split of the nuScenes [2], as shown in Tab. 3. The model with single imitation
247

loss Limi is considered as the baseline (①). With the enhanced perception capability by the spatial
248

objectness loss Lspat, the average L2 error and collision rate are clearly improved to 1.00m and 0.71%
249

from 3.18m and 2.43%, respectively (②v.s. ①). The dreaming loss Ldrm, direction loss Ldir and
250

consistency loss Lcons also respectively bring considerable gains on the average L2 error for 1.98m,
251

1.58m, 1.77m over the baseline model (③,④,⑤v.s. ①). The loss functions are finally combined to
252

construct our UAD (⑥), which obtains the average L2 error of 0.90m and average collision rate of
253

0.19%. The results demonstrate the effectiveness of each proposed component.
254

Temporal Learning with Dreaming Decoder. The temporal learning with the proposed dreaming
255

decoder is realized by Circular Update and Dreaming Loss. The circular update is in charge of both
256

extracting information from observed scenes (Eq. 2) and generating pseudo observations to predict
257

the ego trajectories of future frames (Eq. 3). We study the influence of each module in Tab. 4. Circular
258

Update and Dreaming Loss respectively bring performance gains of 0.70m/0.78m on the average L2
259

error (②,③v.s.①), proving the effectiveness of our designs. Applying both two modules (④) achieves
260

the best performance, showing their complementarity for temporal representation learning.
261

Direction Aware Learning Strategy. Directional Augmentation and Directional Consistency are the
262

two core components of the proposed direction-aware learning strategy. We prove their effectiveness
263

in Tab. 5. It shows that the Directional Augmentation improves the average L2 error for considerable
264

7


---Page Break---
Table 7: Performances under different driving scenes. ∗denotes not using direction-aware learning.

Method

Perf. go straight ↓
(5309 samples)

Perf. turn left ↓
(301 samples)

Perf. turn right ↓
(409 samples)

Perf. Overall ↓
(6019 samples)
Avg. L2 (m)
Avg. Col. (%)
Avg. L2 (m)
Avg. Col. (%)
Avg. L2 (m)
Avg. Col. (%)
Avg. L2 (m)
Avg. Col. (%)

UniAD [21]
0.98
0.26
1.48
0.55
1.27
0.73
1.03
0.31
VAD-Base [22]
1.19
0.37
1.47
0.78
1.39
0.81
1.22
0.43
UAD∗(Ours)
0.89
0.28
1.55
0.43
1.51
0.65
0.97
0.32
UAD (Ours)
0.84
0.17
1.39
0.22
1.16
0.33
0.90
0.19

0.05m (②v.s.①). One interesting observation is that applying the augmentation brings more gains
265

for long-term planning than short-term ones, i.e., the L2 error of 1s/3s decreases for 0.01m/0.08m
266

compared with ①, which proves the effectiveness of our augmentation on enhancing longer temporal
267

information. The Directional Consistency further reduces the average collision rate for impressive
268

0.13% (③v.s.②), which enhances the robustness for driving directional change.
269

Angular Design. We further explore the influence of the proposed angular design by removing
270

the angular partition and angular queries. Specifically, the BEV feature is directly fed into the
271

dreaming decoder to predict pixel-wise objectness, which is supervised by the BEV object mask
272

(see Fig. 2) with binary cross-entropy loss. Besides, the ego query directly interacts with the BEV
273

feature by cross-attention to extract environmental information. The results are presented in Tab. 6.
274

Table 6: Ablation on the angular design.

#
Angular
Design

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
-
0.78 1.31 2.01 1.37 0.61 1.39 2.12 1.37
②
✓
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19

When discarding the angular design, the average L2 er-
275

ror degrades for 0.47m, and the average collision rate
276

consistently degrades for 1.18%. This demonstrates the
277

effectiveness of our angular design in perceiving com-
278

plex environments and planning robust driving routes.
279

4.4
Further Analysis
280

Planning Performance in Different Driving Scenes. The direction-aware learning strategy is
281

designed to enhance the planning performance in scenarios of vehicle steering. We demonstrate the
282

superiority of our proposed model by evaluating the metrics of different driving scenes in Tab. 7.
283

According to the given driving command (i.e., go straight, turn left and turn right), we divide the
284

6,019 validation samples in nuScenes [2] into three parts, which contain 5,309, 301 and 409 ones,
285

respectively. Not surprisingly, all methods perform better under go straight scenes than the steering
286

scenes, proving the necessity of augmenting the imbalanced training data for robust planning. When
287

applying the proposed direction-aware learning strategy, our UAD achieves considerable gains on
288

the average collision rate of turn left and turn right scenes (UAD v.s. UAD∗). Notably, our model
289

outperforms UniAD and VAD by a large margin in steering scenes, proving its effectiveness.
290

Visualization of Angular Perception and Planning. The angular perception pretext is designed
291

to perceive the objects in each sector region. We show its capability by visualizing the predicted
292

objectness in nuScenes [2] in Fig. 5a. For a better view, we transform the discrete objectness
293

scores and ground truth to a pseudo-BEV mask. It shows that our model can successfully capture
294

surrounding objects. Fig. 5a also shows the open-loop planning results of recent SOTA UniAD [21],
295

VAD [22] and our UAD, proving the effectiveness of our method to plan a more reasonable ego
296

trajectory. Fig. 5b compares the closed-loop driving routes between Transfuser [34], ST-P3 [20] and
297

our UAD in CARLA [11]. Our method successfully notices the person and drives in a much safer
298

manner, proving the reliability of our UAD in handling safe-critical issues under complex scenarios.
299

Due to limited space, we present more analyses in the appendix, including 1) the influence of partition
300

angle θ, 2) the influence of direction threshold δ, 3) different backbones and pre-trained weights, 4)
301

replacing 2D ROIs from GroundingDINO with 2D GT boxes, 5) different settings of GroundingDINO
302

to generate 2D ROIs, 6) the influence of pre-training to previous method UniAD and our UAD, 7)
303

runtime analysis of each module in our UAD and modularized UniAD, 8) more visualizations, etc.
304

4.5
Discussion
305

Ego Status and Open-loop Planning Evaluation. As revealed by [26, 40], it’s not a challenge to
306

acquire decent performance of L2 error and collision rate (the original metrics in nuScenes [2]) in
307

the open-loop evaluation of nuScenes by using ego status in the planning module (see Tab. 1). The
308

question is: is open-loop evaluation meaningless? Our answer is NO. Firstly, the inherent reason for
309

the observation is that the simple cases of go straight dominate the nuScenes testing dataset. In these
310

8


---Page Break---
(a)
(b)
Figure 5: (a) Qualitative results in nuScenes. (b) Qualitative results in CARLA.

cases, even a linear extrapolation of motion being sufficient for planning is not surprising. However,
311

as shown in Tab. 7, in more challenging cases like turn right and turn left, the open-loop metrics can
312

still clearly indicate the difficulty of steering scenarios and the differences in methods, which is also
313

proved in [26]. Therefore, open-loop evaluation is not meaningless, while the crux is the distribution
314

of the testing data and the metrics. Secondly, the advantage of open-loop evaluation is its efficiency,
315

which benefits the fast development of algorithms. This view is also revealed by a recent simulator
316

design study [9], which tries to transform the closed-loop evaluation into an open-loop fashion.
317

In our work, we thoroughly compare our model with other methods, which shows consistent improve-
318

ments against previous works under various driving scenarios (straight or steering), different usage of
319

ego status (w/. or w/o.), diverse evaluation metrics (L2 error, collision rate or intersection rate from
320

[26]), and different evaluation types (open- or closed-loop). It thus again proves the importance of
321

designing suitable pretext tasks for end-to-end autonomous driving.
322

How to Guarantee Safety in Current Auto-Drive System? Safety is the first requirement of
323

autonomous driving systems in practical products, especially for L4-level auto-vehicles. To guarantee
324

safety, offline collision check with predicted 3D boxes is an inevitable post-process under current
325

technological conditions. Then, a question naturally arises: how to safely apply our model to
326

current auto-driving systems? Before answering this question, we reaffirm our claim that we believe
327

discarding 3D labels is an efficient, attractive, and potential direction for E2EAD, but it doesn’t mean
328

we refuse to use any 3D labels if the relatively cheap ones are available in practical product engineering.
329

For instance, solely annotating bounding boxes without object identity for tracking is much cheaper
330

than labeling other elements like HD-map, and point-cloud segmentation labels for occupancy.
331

Therefore, we provide a degraded version of our method by arranging an additional 3D detection head.
332

Table 8: Ablation on the 3D detection head.

#
Detection
Head

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
-
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19
②
✓
0.37 0.86 1.57 0.93 0.02 0.17 0.55 0.25

Then our model can seamlessly integrate into auto-
333

drive products, and offline collision check is achiev-
334

able. As shown in Tab. 8, integrating the 3D detection
335

head doesn’t bring additional improvements, which
336

again proves the design of our method has sufficiently
337

encoded 3D information to the planning module.
338

In a nutshell, 1) our work can easily integrate other 3D tasks if they are inevitable under current
339

technical conditions; 2) the experiments again prove from the side that our spatial-temporal module
340

has already encoded important 3D clues for planning; 3) we hope our frontier work can eliminate
341

some inessential 3D sub-tasks for both research and engineer usage of E2EAD models. An era of
342

cheap, laboratory-affordable but robust, practical E2EAD design will eventually come!
343

5
Conclusion
344

Our work seeks to liberate E2EAD from costly modularization and 3D manual annotation. With this
345

goal, we propose the unsupervised pretext task to perceive the environment by predicting angular-
346

wise objectness and future dynamics. To improve the robustness in steering scenarios, we introduce
347

the direction-aware training strategy for planning. Experiments demonstrate the effectiveness and
348

efficiency of our method. As discussed, although the ego trajectories are easily obtained, it is almost
349

impossible to collect billion-level precisely annotated data with perception labels. This impedes the
350

further development of end-to-end autonomous driving. We believe our work provides a potential
351

solution to this barrier and may push performance to the next level when massive data are available.
352

9


---Page Break---
References
353

[1] Mariusz Bojarski, Philip Yeres, Anna Choromanska, Krzysztof Choromanski, Bernhard Firner, Lawrence
354

Jackel, and Urs Muller. Explaining how a deep neural network trained with end-to-end learning steers a
355

car. arXiv preprint arXiv:1704.07911, 2017.
356

[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan,
357

Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving.
358

In CVPR, 2020.
359

[3] Sergio Casas, Abbas Sadat, and Raquel Urtasun. Mp3: A unified model to map, perceive, predict and plan.
360

In CVPR, 2021.
361

[4] Jun Cen, Peng Yun, Junhao Cai, Michael Yu Wang, and Ming Liu. Open-set 3d object detection. In 2021
362

International Conference on 3D Vision (3DV). IEEE, 2021.
363

[5] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In Conference
364

on Robot Learning, 2020.
365

[6] Kashyap Chitta, Aditya Prakash, and Andreas Geiger. Neat: Neural attention fields for end-to-end
366

autonomous driving. In ICCV, 2021.
367

[7] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated
368

recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.
369

[8] Felipe Codevilla, Eder Santana, Antonio M López, and Adrien Gaidon. Exploring the limitations of
370

behavior cloning for autonomous driving. In ICCV, 2019.
371

[9] NAVSIM Contributors. Navsim: Data-driven non-reactive autonomous vehicle simulation. https:
372

//github.com/autonomousvision/navsim, 2024.
373

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
374

image database. In CVPR. Ieee, 2009.
375

[11] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open
376

urban driving simulator. In Conference on robot learning, 2017.
377

[12] Zeyu Gao, Yao Mu, Ruoyan Shen, Chen Chen, Yangang Ren, Jianyu Chen, Shengbo Eben Li, Ping Luo,
378

and Yanfeng Lu. Enhance sample efficiency and robustness of end-to-end urban autonomous driving via
379

semantic masked world model. arXiv preprint arXiv:2210.04017, 2022.
380

[13] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. NeurIPS, 2018.
381

[14] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning
382

behaviors by latent imagination. ICLR, 2020.
383

[15] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete
384

world models. ICLR, 2021.
385

[16] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through
386

world models. arXiv preprint arXiv:2301.04104, 2023.
387

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
388

In CVPR, 2016.
389

[18] Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zachary Murez, Corina Gurau, Hudson Yeo, Alex
390

Kendall, Roberto Cipolla, and Jamie Shotton. Model-based imitation learning for urban driving. NeurIPS,
391

2022.
392

[19] Peiyun Hu, Aaron Huang, John Dolan, David Held, and Deva Ramanan. Safe local motion planning with
393

self-supervised freespace forecasting. In CVPR, 2021.
394

[20] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end
395

vision-based autonomous driving via spatial-temporal feature learning. In ECCV. Springer, 2022.
396

[21] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei
397

Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In CVPR, 2023.
398

[22] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu,
399

Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving.
400

arXiv preprint arXiv:2303.12077, 2023.
401

10


---Page Break---
[23] Tarasha Khurana, Peiyun Hu, Achal Dave, Jason Ziglar, David Held, and Deva Ramanan. Differentiable
402

raycasting for self-supervised occupancy forecasting. In ECCV, 2022.
403

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
404

Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint
405

arXiv:2304.02643, 2023.
406

[25] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai.
407

Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal trans-
408

formers. In ECCV. Springer, 2022.
409

[26] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you
410

need for open-loop end-to-end autonomous driving? In CVPR, 2024.
411

[27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
412

and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV. Springer, 2014.
413

[28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang,
414

Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object
415

detection. arXiv preprint arXiv:2303.05499, 2023.
416

[29] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy
417

networks: Learning 3d reconstruction in function space. In CVPR, 2019.
418

[30] Mahyar Najibi, Jingwei Ji, Yin Zhou, Charles R Qi, Xinchen Yan, Scott Ettinger, and Dragomir Anguelov.
419

Unsupervised 3d perception with 2d vision-language distillation for autonomous driving. In ICCV, 2023.
420

[31] OpenAI. Chatgpt [large language model]. https://chat.openai.com, 2023.
421

[32] Minting Pan, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang. Iso-dream: Isolating and leveraging
422

noncontrollable visual dynamics in world models. NeurIPS, 2022.
423

[33] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. NeurIPS, 1988.
424

[34] Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end
425

autonomous driving. In CVPR, 2021.
426

[35] Abbas Sadat, Sergio Casas, Mengye Ren, Xinyu Wu, Pranaab Dhawan, and Raquel Urtasun. Perceive,
427

predict, and plan: Safe motion planning through interpretable semantic representations. In ECCV. Springer,
428

2020.
429

[36] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping
430

Luo, Dahua Lin, et al. Scene as occupancy. In ICCV, 2023.
431

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
432

Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.
433

[38] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future:
434

Multiview visual forecasting and planning with world model for autonomous driving. CVPR, 2024.
435

[39] Wenyuan Zeng, Wenjie Luo, Simon Suo, Abbas Sadat, Bin Yang, Sergio Casas, and Raquel Urtasun.
436

End-to-end interpretable neural motion planner. In CVPR, 2019.
437

[40] Jiang-Tian Zhai, Ze Feng, Jinhao Du, Yongqiang Mao, Jiang-Jiang Liu, Zichang Tan, Yifu Zhang, Xiaoqing
438

Ye, and Jingdong Wang. Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes.
439

arXiv preprint arXiv:2305.10430, 2023.
440

11


---Page Break---
A
Appendix
441

The appendix presents additional designing and explaining details of our Unpervised pretext task for
442

end-to-end Autonomous Driving (UAD) in the manuscript.
443

• Different Partition Angles
444

We explore the influence of different partition angles in angular pretext to learn better
445

spatio-temporal knowledge.
446

• Different Direction Thresholds
447

We explore the influence of different thresholds in direction prediction to enhance planning
448

robustness in complex driving scenarios.
449

• Different Backbones and Pre-trained Weights
450

We compare the performance of different backbones and pre-trained weights on our method.
451

• Objectness Label Generation with GT Boxes
452

We compare the generated objectness label between using the pseudo ROIs from Ground-
453

ingDINO [28] and ground-truth boxes on different backbones.
454

• Settings for ROI Generation
455

We ablate different settings for the open-set 2D detector GroundingDINO, which provides
456

ROIs for the label generation of angular perception pretext.
457

• Different Image Sizes and BEV Resolution
458

We compare the performance with different input sizes of multi-view images and BEV
459

resolutions.
460

• Runtime Analysis
461

We evaluate the runtime of each module of UAD and compare with modularized UniAD [21],
462

which demonstrates the efficiency of our method.
463

• Classification of Angular Perception
464

We evaluate the objectness prediction in the angular perception pretext, which demonstrates
465

the enhanced perception capability in complex driving scenarios.
466

• Influence of Pre-training
467

We evaluate the influence of pre-training by detailing the training losses and planning
468

performances with different pre-trained weights.
469

• More Visualizations
470

We provide more visualizations for the predicted angular-wise objectness and planning re-
471

sults in the open-loop evaluation of nuScenes [2] and closed-loop simulation of CARLA [11].
472

A.1
Different Partition Angles
473

The proposed angular perception pretext divides the BEV space into multiple sectors. We explore the
474

influence of partition angle θ in Tab 9. Experimental results show that the L2 error and inference
475

speed gradually increase with the partition angle. The model with partition angle of 1◦(①) achieves
476

the best average L2 error of 0.85m. And the partition angle of 4◦contributes to the best average
477

collision rate of 0.19% (③). This reveals that a smaller partition angle helps learn more fine-grained
478

environmental representations, eventually benefiting planning. In contrast, the model with a large
479

partition angle sparsely perceives the scene. Despite reducing the computation cost, it will also
480

degrade the safety of the end-to-end autonomous driving system.
481

Table 9: Ablation on different partition angles
in the proposed angular pretext.

# Partition
Angle

L2 (m) ↓
Collision (%) ↓
FPS
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
1◦
0.35 0.78 1.42 0.85 0.01 0.28 0.68 0.32
5.0
②
2◦
0.34 0.77 1.46 0.86 0.01 0.22 0.48 0.24
6.3
③
4◦
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19
7.2
④
8◦
0.38 0.85 1.55 0.93 0.01 0.18 0.55 0.25
7.7
⑤
15◦
0.47 0.94 1.69 1.03 0.03 0.20 0.60 0.28
8.1
⑥
30◦
0.48 1.00 1.75 1.08 0.05 0.28 0.63 0.32
8.4

Table 10: Ablation on different thresholds of direc-
tion prediction in the directional augmentation.

# Threshold
(m)

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
0.5
0.35 0.79 1.43 0.86 0.03 0.18 0.71 0.31
②
0.8
0.35 0.77 1.46 0.86 0.01 0.12 0.68 0.27
③
1.2
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19
④
1.5
0.40 0.82 1.52 0.91 0.02 0.15 0.42 0.20
⑤
2.0
0.38 0.85 1.55 0.93 0.01 0.08 0.48 0.19

12


---Page Break---
Table 11: Ablation on different backbones and pre-trained weights.

# Backbone Pretrained
Weight

L2 (m) ↓
Collision (%) ↓
FPS
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
Res50
None
0.43 0.94 1.65 1.01 0.03 0.37 0.86 0.42
9.6
②
ImageNet 0.41 0.90 1.66 0.99 0.03 0.32 0.80 0.38

③

Res101

None
0.40 0.87 1.59 0.95 0.02 0.23 0.59 0.28

7.2
④
ImageNet 0.37 0.84 1.53 0.91 0.01 0.18 0.50 0.23
⑤
COCO
0.36 0.83 1.51 0.90 0.01 0.16 0.45 0.21
⑥
NuImages 0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19

Table 12: Ablation on 2D object boxes in pretext label generation.

# Backbone 2D Object
Box

L2 (m) ↓
Collision (%) ↓
FPS
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
Res50
Pseudo
0.41 0.90 1.66 0.99 0.03 0.32 0.80 0.38
9.6
②
GT
0.41 0.87 1.61 0.96 0.03 0.30 0.71 0.35

③
Res101
Pseudo
0.39 0.81 1.50 0.90 0.01 0.12 0.43 0.19
7.2
④
GT
0.37 0.79 1.45 0.84 0.01 0.13 0.39 0.18

A.2
Different Direction Thresholds
482

The direction prediction that the ego car intends to maneuver (i.e., left, straight and right) is proposed
483

to enhance the steering capability for autonomous driving. The label is generated with the threshold δ
484

(see Eq. 7 in the manuscript), which determines the ground-truth direction of each waypoint in the
485

expert trajectory. Here we explore the influence by ablating different thresholds, as shown in Tab. 10.
486

Experimental results show that the L2 error gradually increases with the direction threshold. The
487

model with δ of 0.5m (①) achieves the lowest L2 error of 0.86m. It reveals that a smaller threshold
488

will force the planner to fit the expert navigation, leading to a closer distance between the predicted
489

trajectory and the ground truth. In contrast, the collision rate benefits more from larger thresholds.
490

The model with δ of 2.0m obtains the best collision rate at 2s of 0.08% (⑤), showing the effectiveness
491

for robust planning. Notably, the threshold of 1.2m contributes to a great balance with the average L2
492

error of 0.90m and average collision rate of 0.19%.
493

A.3
Different Backbones and Pre-trained Weights
494

As a common sense, pre-training the backbone network with fundamental tasks like image classi-
495

fication on ImageNet [10] will benefit the sub-tasks. The previous method UniAD [21] uses the
496

pre-trained weights of BEVFormer [25]. What surprised us is that when replacing the pre-trained
497

weights with the one learned on ImageNet, the performance of UniAD dramatically degraded (see
498

“Influence of Pre-training” for more details). This inspires us to explore the influence of backbone
499

settings on our framework. As shown in Tab. 11, interestingly, even without any pre-training, our
500

model still outperforms UniAD with pre-trained ResNet101 and VAD with pre-trained ResNet50.
501

This verifies the effectiveness of our unsupervised pretext task on modeling the driving scenes. We
502

also use publicly available pre-trained weights on detection datasets like COCO [27] and nuImages [2]
503

to train our model, which shows better performance. These experimental results and observations
504

demonstrate that a potentially promising topic is how to pre-train a model for end-to-end autonomous
505

driving. We leave this to future research.
506

A.4
Objectness Label Generation with GT Boxes
507

As mentioned in the manuscript, the essence of generating the angular objectness label lies in the
508

2D ROIs, which come from the open-set 2D detector GroundingDINO [28]. Here we explore the
509

influence of using the ground-truth 2D boxes as ROIs, which provide more high-quality samples
510

for the representation learning in the angular perception pretext. Tab. 12 shows that training with
511

GT boxes achieves consistent performance gains on both ResNet50 [17] and ResNet101 [17] (②,④
512

v.s. ①,③). This reveals that accurate annotation does help to learn better spatio-temporal knowledge
513

and improve ego planning. Considering the cost in real-world deployment, training with accessible
514

13


---Page Break---
Table 13: Ablation on the settings of ROI generation. The Conf. Thresh denotes the confidence
threshold in GroundingDINO [28] to filter unreliable predictions. vehicle,pedestrian,barrier represent
the used prompt words to obtain ROIs of corresponding classes. Rule Filter indicates filtering the
ROIs that are more than half of the length or width of the image.

#
Conf.
Thresh
Prompt
Words
Rule
Filter

L2 (m) ↓
Collision (%) ↓
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
0.35
{vehicle}
-
0.48
0.98
1.75
1.07
0.08
0.38
0.80
0.42
②
0.35
{vehicle,pedestrian}
-
0.47
0.94
1.69
1.03
0.04
0.27
0.71
0.34
③
0.35
{vehicle,pedestrian,barrier}
-
0.43
0.88
1.60
0.97
0.03
0.23
0.60
0.29
④
0.35
{vehicle,pedestrian,barrier}
✓
0.39
0.81
1.50
0.90
0.01
0.12
0.43
0.19
⑤
0.30
{vehicle,pedestrian,barrier}
✓
0.39
0.82
1.45
0.89
0.01
0.21
0.51
0.24
⑥
0.40
{vehicle,pedestrian,barrier}
✓
0.46
0.90
1.57
0.98
0.01
0.13
0.37
0.17

Table 14: Comparison with different backbones, image sizes and BEV resolutions.

#
Method
Backbone
Image
Size

BEV
Resolution

L2 (m) ↓
Collision (%) ↓
FPS
1s
2s
3s
Avg.
1s
2s
3s
Avg.

①
UniAD [21]
R101
1600×900
200×200
0.48
0.96
1.65
1.03
0.05
0.17
0.71
0.31
2.1

②
VAD-Tiny [22]
R50
640×360
100×100
0.60
1.23
2.06
1.30
0.33
1.33
2.21
1.29
17.6
③
VAD-Base [22]
R50
1280×720
200×200
0.54
1.15
1.98
1.22
0.10
0.24
0.96
0.43
5.3

④
UAD (Ours)
R50
640×360
100×100
0.47
0.99
1.71
1.06
0.08
0.39
0.90
0.46
18.9
⑤
UAD (Ours)
R50
1600×900
200×200
0.41
0.90
1.66
0.99
0.03
0.32
0.80
0.38
9.6
⑥
UAD (Ours)
R101
1600×900
200×200
0.39
0.81
1.50
0.90
0.01
0.12
0.43
0.19
7.2

pseudo labels is a more efficient way compared with the manual annotation, which also shows
515

comparable performance in autonomous driving (①v.s. ②and ③v.s. ④).
516

A.5
Settings for ROI Generation.
517

The quality of learned spatio-temporal knowledge highly relies on the generated ROIs by the open-set
518

2D detector GroundingDINO [28], which are then projected as the BEV objectness label for training
519

the angular perception pretext. We explore the influence of generated ROIs with different settings,
520

as shown in Tab. 13. We take the setting with the confidence score of 0.35, prompt word of vehicle
521

and without the Rule Filter, as the baseline (①). By appending more prompt words (e.g., pedestrian,
522

barrier), the planning performance gradually improves (③,②v.s.①), showing the enhanced perception
523

capability with more diversified objects. Filtering the ROIs with overlarge size (i.e., Rule Filter)
524

brings considerable gains for the average L2 error of 0.07m and average collision rate of 0.10%
525

(④v.s.③). One interesting observation is that decreasing the confidence threshold would slightly
526

improve the L2 error while causing higher collision rate (⑤v.s.④). In contrast, increasing the threshold
527

obtains lower average collision rate of 0.17% and higher average L2 error of 0.98m. This reveals the
528

importance of providing diversified ROIs for angular perception learning as well as ensuring high
529

quality. The model with the confidence score of 0.35, all prompt words and Rule Filter achieves
530

balanced performance with the average L2 error of 0.90m and average collision rate of 0.19%.
531

A.6
Different Image Sizes and BEV Resolution
532

For safe autonomous driving, increasing the input size of the multi-view images and the resolution
533

of the built BEV representation is an effective way, which provide more detailed environmental
534

information. While benefiting perception and planning, it inevitably brings heavy computation cost.
535

We then ablate the image size and BEV resolution of our UAD to find a balanced version between
536

performance and efficiency, as shown in Tab. 14. The results show that our UAD with ResNet-
537

101 [17], image size of 1600×900, BEV resolution of 200×200, achieves the best performance
538

compared with previous methods UniAD [21] and VAD-Base [22] while running faster with 7.2FPS
539

(⑥). By replacing the backbone with ResNet-50, our UAD is more efficient with little performance
540

degradation (⑤v.s. ⑥). We further align the settings of VAD-Tiny, which has an inference speed
541

of outstanding 17.6FPS (②), to explore the influence of much smaller input sizes. Tab. 14 shows
542

that our UAD still achieves excellent performance even compared with VAD-Base of high-resolution
543

inputs (④v.s. ③). Notably, our UAD of this version has the fastest inference speed of 18.9FPS. This
544

14


---Page Break---
Table 15: Module runtime comparison between
UniAD [21] and our UAD. The inference is
measured on an NVIDIA Tesla A100 GPU.

Model
Partition

UniAD
UAD (Ours)

Module
Latency
(ms)
Proportion
(%)
Module
Latency
(ms)
Proportion
(%)

Feature
Extraction

Backbone
38.1 ±0.5
8,2%
Backbone 36.0 ±0.3
26.0%
BEV
Encoder
83.4 ±0.5
17.9%
BEV
Encoder
81.5 ±0.4
58.9%

Det&Track 145.3 ±1.3
31.2%
Map
92.1 ±0.7
19.8%

Angular
Partition
1.1 ±0.1
0.8%

Motion
50.6 ±0.6
10.9%
Sub-
Task

Occupancy 45.9 ±0.4
9.9%

Dreaming

Decoder
18.2 ±0.2
13.2%

Prediction
Planning
Head
9.7 ±0.3
2.1%
Planning
Head
1.5 ±0.1
1.1%

Total
-
465.1 ±4.3
100%
-
138.3 ±1.1
100.0%

Figure 6: Visualization of the PR and ROC curves
for the angular-wise objectness prediction in differ-
ent driving scenes.

(a)
(b)
Figure 7: Optimization of UniAD (a) and our UAD (b) with different pre-trained backbone weights.

again proves the effectiveness of our method in performing fine-grained perception, as well as the
545

robustness to fit the inputs of different sizes.
546

A.7
Runtime Analysis
547

Tab. 15 compares the runtime of each module between the modularized method UniAD [21] and
548

our UAD. As we adopt the Backbone and BEV Encoder from BEVFormer [25] that are the same in
549

UniAD, the latency of feature extraction is similar with little difference due to different pre-processing.
550

The modular sub-tasks in UniAD consume most of the runtime, i.e., significant 71.8% for Det&Track
551

(31.2%), Map (19.8%), Motion (10.9%) and Occupancy (9.9%), respectively. In contrast, our UAD
552

performs simple Angular Partition and Dreaming Decoder, which take only 14.0% (19.3ms) to model
553

the complex environment. This demonstrates our insight that it’s a necessity to liberate end-to-end
554

autonomous driving from costly modularization. The downstream Planning Head takes negligible
555

1.5ms to plan the ego trajectory, compared with 9.7ms in UniAD. Finally, our UAD finishes the
556

inference with a total runtime of 138.3ms, 3.4× faster than the 465.1ms of UniAD, showing the
557

efficiency of our design.
558

A.8
Classification of Angular Perception
559

The proposed angular perception pretext learns spatio-temporal knowledge of the driving scene
560

by predicting the objectness of each sector region, which is supervised by the generated binary
561

angular-wise label. We show the perception ability by evaluating the classification metrics based on
562

the validation split of the nuScenes [2] dataset. Fig. 6 draws the Precision-Recall (PR) curve and
563

Receiver-Operating-Characteristic (ROC) curve in different driving scenes (i.e., turn left, go straight
564

and turn right). In the PR curve, our UAD achieves balanced precision and recall scores in different
565

driving scenes, showing the effectiveness of our pretext task to perceive the surrounding objects.
566

Notably, the performance of go straight scenes is slightly better than the steering ones under all
567

thresholds. This proves our insight to design tailored direction-aware learning strategy for improving
568

the safety-critical turn left and turn right scenes. The ROC curve shows the robustness of our angular
569

perception pretext to classify the objects from complex environmental observations.
570

15


---Page Break---
Figure 8: Visualization of the angular perception.

Figure 9: Visualization of the planning results. The first two rows show the success of our method in
safe planning in complex scenarios, while the third row exhibits a failure case of our planner when no
temporal information could be acquired when t=0.

A.9
Influence of Pre-training
571

Pre-training the backbone network with fundamental tasks is a commonly used metric to benefit
572

representation learning. As mentioned in “Different Backbones and Pre-trained Weights” of Sec. 4.4
573

in the manuscript, the performance of the previous SOTA method UniAD [21] dramatically degrades
574

without the pre-trained weights from BEVFormer [25]. Here we further detail the influence by
575

comparing the training losses and planning performances with different pre-trained weights in Fig. 7.
576

Fig. 7a shows that the training losses increase by about 20 on average when replaced with the
577

pre-trained weights from ImageNet [10]. Correspondingly, the average L2 error is significantly higher
578

than the one with the pre-trained weights from BEVFormer. This reveals that UniAD heavily relies
579

on the perceptive pre-training in BEVFormer to optimize modularized sub-tasks. In contrast, our
580

UAD performs comparably even without any pre-training (see Fig. 7b), proving the effectiveness of
581

our designs for robust optimization.
582

16


---Page Break---
Figure 10: Visualization of angular perception and planning in Carla.

A.10
More Visualizations
583

Open-loop Planning
We provide more visualizations about the predicted angular-wise objectness
584

and planning results on nuScenes [2]. Fig. 8 compares the discrete objectness scores and ground
585

truth, proving the effectiveness of our angular perception pretext to perceive the objects in each sector
586

region. The planning results of previous SOTA methods (i.e., UniAD [21] and VAD [22]) and our
587

UAD are shown in Fig. 9. With the designed pretext and tailored training strategy, our method could
588

plan a more reasonable ego trajectory under different driving scenarios, proving the effectiveness
589

of our work. The third row shows the failure case of our planner. In this case, the ego car is given
590

the “Turn Right” command when t = 0 (i.e., the first frame of the driving scenario), leading to
591

ineffectiveness of our planner in learning helpful temporal information. A possible solution to deal
592

with this is to apply an auxiliary trajectory prior for the first several frames, and we leave this to
593

future work.
594

Closed-loop Simulation
Fig. 10 visualizes the predicted objectness and planning results in the
595

Town05 Long benchmark of CARLA [11]. Following the setting of ST-P3 [20] in closed-loop evalua-
596

tion, we collect visual observations from the cameras of “CAM_FRONT”, “CAM_FRONT_LEFT”,
597

“CAM_FRONT_RIGHT” and “CAM_BACK”. It shows that the sector regions in which the surround-
598

ing objects exist are successfully captured by our UAD, proving the effectiveness and robustness of
599

our design. Notably, the missed objects by GroundingDINO [28], e.g., the black car in the camera of
600

“CAM_FRONT_LEFT” at t = 145, are surprisingly perceived and marked in the corresponding sector.
601

This demonstrates our method has the capability of learning perceptive knowledge in a data-driven
602

manner, even with coarse supervision by the generated 2D pseudo boxes from GroundingDINO.
603

17


---Page Break---
NeurIPS Paper Checklist
604

1. Claims
605

Question: Do the main claims made in the abstract and introduction accurately reflect the
606

paper’s contributions and scope?
607

Answer: [Yes]
608

Justification: The main claims made in the abstract and introduction accurately reflect the
609

paper’s contributions and scope, please see Sec. 4.
610

Guidelines:
611

• The answer NA means that the abstract and introduction do not include the claims
612

made in the paper.
613

• The abstract and/or introduction should clearly state the claims made, including the
614

contributions made in the paper and important assumptions and limitations. A No or
615

NA answer to this question will not be perceived well by the reviewers.
616

• The claims made should match theoretical and experimental results, and reflect how
617

much the results can be expected to generalize to other settings.
618

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
619

are not attained by the paper.
620

2. Limitations
621

Question: Does the paper discuss the limitations of the work performed by the authors?
622

Answer: [Yes]
623

Justification: The limitations of the work are discussed in this paper, please see Sec. 4.5.
624

Guidelines:
625

• The answer NA means that the paper has no limitation while the answer No means that
626

the paper has limitations, but those are not discussed in the paper.
627

• The authors are encouraged to create a separate "Limitations" section in their paper.
628

• The paper should point out any strong assumptions and how robust the results are to
629

violations of these assumptions (e.g., independence assumptions, noiseless settings,
630

model well-specification, asymptotic approximations only holding locally). The authors
631

should reflect on how these assumptions might be violated in practice and what the
632

implications would be.
633

• The authors should reflect on the scope of the claims made, e.g., if the approach was
634

only tested on a few datasets or with a few runs. In general, empirical results often
635

depend on implicit assumptions, which should be articulated.
636

• The authors should reflect on the factors that influence the performance of the approach.
637

For example, a facial recognition algorithm may perform poorly when image resolution
638

is low or images are taken in low lighting. Or a speech-to-text system might not be
639

used reliably to provide closed captions for online lectures because it fails to handle
640

technical jargon.
641

• The authors should discuss the computational efficiency of the proposed algorithms
642

and how they scale with dataset size.
643

• If applicable, the authors should discuss possible limitations of their approach to
644

address problems of privacy and fairness.
645

• While the authors might fear that complete honesty about limitations might be used by
646

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
647

limitations that aren’t acknowledged in the paper. The authors should use their best
648

judgment and recognize that individual actions in favor of transparency play an impor-
649

tant role in developing norms that preserve the integrity of the community. Reviewers
650

will be specifically instructed to not penalize honesty concerning limitations.
651

3. Theory Assumptions and Proofs
652

Question: For each theoretical result, does the paper provide the full set of assumptions and
653

a complete (and correct) proof?
654

Answer: [NA]
655

18


---Page Break---
Justification: This paper does not include theoretical results.
656

Guidelines:
657

• The answer NA means that the paper does not include theoretical results.
658

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
659

referenced.
660

• All assumptions should be clearly stated or referenced in the statement of any theorems.
661

• The proofs can either appear in the main paper or the supplemental material, but if
662

they appear in the supplemental material, the authors are encouraged to provide a short
663

proof sketch to provide intuition.
664

• Inversely, any informal proof provided in the core of the paper should be complemented
665

by formal proofs provided in appendix or supplemental material.
666

• Theorems and Lemmas that the proof relies upon should be properly referenced.
667

4. Experimental Result Reproducibility
668

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
669

perimental results of the paper to the extent that it affects the main claims and/or conclusions
670

of the paper (regardless of whether the code and data are provided or not)?
671

Answer: [Yes]
672

Justification: All the information needed to reproduce the main experimental results of the
673

paper is disclosed, please see Sec. 4.1.
674

Guidelines:
675

• The answer NA means that the paper does not include experiments.
676

• If the paper includes experiments, a No answer to this question will not be perceived
677

well by the reviewers: Making the paper reproducible is important, regardless of
678

whether the code and data are provided or not.
679

• If the contribution is a dataset and/or model, the authors should describe the steps taken
680

to make their results reproducible or verifiable.
681

• Depending on the contribution, reproducibility can be accomplished in various ways.
682

For example, if the contribution is a novel architecture, describing the architecture fully
683

might suffice, or if the contribution is a specific model and empirical evaluation, it may
684

be necessary to either make it possible for others to replicate the model with the same
685

dataset, or provide access to the model. In general. releasing code and data is often
686

one good way to accomplish this, but reproducibility can also be provided via detailed
687

instructions for how to replicate the results, access to a hosted model (e.g., in the case
688

of a large language model), releasing of a model checkpoint, or other means that are
689

appropriate to the research performed.
690

• While NeurIPS does not require releasing code, the conference does require all submis-
691

sions to provide some reasonable avenue for reproducibility, which may depend on the
692

nature of the contribution. For example
693

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
694

to reproduce that algorithm.
695

(b) If the contribution is primarily a new model architecture, the paper should describe
696

the architecture clearly and fully.
697

(c) If the contribution is a new model (e.g., a large language model), then there should
698

either be a way to access this model for reproducing the results or a way to reproduce
699

the model (e.g., with an open-source dataset or instructions for how to construct
700

the dataset).
701

(d) We recognize that reproducibility may be tricky in some cases, in which case
702

authors are welcome to describe the particular way they provide for reproducibility.
703

In the case of closed-source models, it may be that access to the model is limited in
704

some way (e.g., to registered users), but it should be possible for other researchers
705

to have some path to reproducing or verifying the results.
706

5. Open access to data and code
707

Question: Does the paper provide open access to the data and code, with sufficient instruc-
708

tions to faithfully reproduce the main experimental results, as described in supplemental
709

material?
710

19


---Page Break---
Answer: [Yes]
711

Justification: This paper provides open access to the data and code to reproduce the main
712

experimental results, please see Sec. 4.1.
713

Guidelines:
714

• The answer NA means that paper does not include experiments requiring code.
715

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
716

public/guides/CodeSubmissionPolicy) for more details.
717

• While we encourage the release of code and data, we understand that this might not be
718

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
719

including code, unless this is central to the contribution (e.g., for a new open-source
720

benchmark).
721

• The instructions should contain the exact command and environment needed to run to
722

reproduce the results. See the NeurIPS code and data submission guidelines (https:
723

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
724

• The authors should provide instructions on data access and preparation, including how
725

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
726

• The authors should provide scripts to reproduce all experimental results for the new
727

proposed method and baselines. If only a subset of experiments are reproducible, they
728

should state which ones are omitted from the script and why.
729

• At submission time, to preserve anonymity, the authors should release anonymized
730

versions (if applicable).
731

• Providing as much information as possible in supplemental material (appended to the
732

paper) is recommended, but including URLs to data and code is permitted.
733

6. Experimental Setting/Details
734

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
735

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
736

results?
737

Answer: [Yes]
738

Justification: All the training and test details are specified in this paper, please see Sec. 4.1.
739

Guidelines:
740

• The answer NA means that the paper does not include experiments.
741

• The experimental setting should be presented in the core of the paper to a level of detail
742

that is necessary to appreciate the results and make sense of them.
743

• The full details can be provided either with the code, in appendix, or as supplemental
744

material.
745

7. Experiment Statistical Significance
746

Question: Does the paper report error bars suitably and correctly defined or other appropriate
747

information about the statistical significance of the experiments?
748

Answer: [Yes]
749

Justification: This paper reports information about the statistical significance of experiments,
750

please see Sec. 4.
751

Guidelines:
752

• The answer NA means that the paper does not include experiments.
753

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
754

dence intervals, or statistical significance tests, at least for the experiments that support
755

the main claims of the paper.
756

• The factors of variability that the error bars are capturing should be clearly stated (for
757

example, train/test split, initialization, random drawing of some parameter, or overall
758

run with given experimental conditions).
759

• The method for calculating the error bars should be explained (closed form formula,
760

call to a library function, bootstrap, etc.)
761

• The assumptions made should be given (e.g., Normally distributed errors).
762

20


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error
763

of the mean.
764

• It is OK to report 1-sigma error bars, but one should state it. The authors should
765

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
766

of Normality of errors is not verified.
767

• For asymmetric distributions, the authors should be careful not to show in tables or
768

figures symmetric error bars that would yield results that are out of range (e.g. negative
769

error rates).
770

• If error bars are reported in tables or plots, The authors should explain in the text how
771

they were calculated and reference the corresponding figures or tables in the text.
772

8. Experiments Compute Resources
773

Question: For each experiment, does the paper provide sufficient information on the com-
774

puter resources (type of compute workers, memory, time of execution) needed to reproduce
775

the experiments?
776

Answer: [Yes]
777

Justification: This paper provides sufficient information on the computer resources, please
778

see Sec. 4.1.
779

Guidelines:
780

• The answer NA means that the paper does not include experiments.
781

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
782

or cloud provider, including relevant memory and storage.
783

• The paper should provide the amount of compute required for each of the individual
784

experimental runs as well as estimate the total compute.
785

• The paper should disclose whether the full research project required more compute
786

than the experiments reported in the paper (e.g., preliminary or failed experiments that
787

didn’t make it into the paper).
788

9. Code Of Ethics
789

Question: Does the research conducted in the paper conform, in every respect, with the
790

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
791

Answer: [Yes]
792

Justification: The conducted research conforms with the NeurIPS Code of Ethics.
793

Guidelines:
794

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
795

• If the authors answer No, they should explain the special circumstances that require a
796

deviation from the Code of Ethics.
797

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
798

eration due to laws or regulations in their jurisdiction).
799

10. Broader Impacts
800

Question: Does the paper discuss both potential positive societal impacts and negative
801

societal impacts of the work performed?
802

Answer: [Yes]
803

Justification: This paper discusses both potential positive societal impacts and negative
804

societal impacts of the work, please see Sec. 4.5.
805

Guidelines:
806

• The answer NA means that there is no societal impact of the work performed.
807

• If the authors answer NA or No, they should explain why their work has no societal
808

impact or why the paper does not address societal impact.
809

• Examples of negative societal impacts include potential malicious or unintended uses
810

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
811

(e.g., deployment of technologies that could make decisions that unfairly impact specific
812

groups), privacy considerations, and security considerations.
813

21


---Page Break---
• The conference expects that many papers will be foundational research and not tied
814

to particular applications, let alone deployments. However, if there is a direct path to
815

any negative applications, the authors should point it out. For example, it is legitimate
816

to point out that an improvement in the quality of generative models could be used to
817

generate deepfakes for disinformation. On the other hand, it is not needed to point out
818

that a generic algorithm for optimizing neural networks could enable people to train
819

models that generate Deepfakes faster.
820

• The authors should consider possible harms that could arise when the technology is
821

being used as intended and functioning correctly, harms that could arise when the
822

technology is being used as intended but gives incorrect results, and harms following
823

from (intentional or unintentional) misuse of the technology.
824

• If there are negative societal impacts, the authors could also discuss possible mitigation
825

strategies (e.g., gated release of models, providing defenses in addition to attacks,
826

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
827

feedback over time, improving the efficiency and accessibility of ML).
828

11. Safeguards
829

Question: Does the paper describe safeguards that have been put in place for responsible
830

release of data or models that have a high risk for misuse (e.g., pretrained language models,
831

image generators, or scraped datasets)?
832

Answer: [NA]
833

Justification: This paper poses no such risks.
834

Guidelines:
835

• The answer NA means that the paper poses no such risks.
836

• Released models that have a high risk for misuse or dual-use should be released with
837

necessary safeguards to allow for controlled use of the model, for example by requiring
838

that users adhere to usage guidelines or restrictions to access the model or implementing
839

safety filters.
840

• Datasets that have been scraped from the Internet could pose safety risks. The authors
841

should describe how they avoided releasing unsafe images.
842

• We recognize that providing effective safeguards is challenging, and many papers do
843

not require this, but we encourage authors to take this into account and make a best
844

faith effort.
845

12. Licenses for existing assets
846

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
847

the paper, properly credited and are the license and terms of use explicitly mentioned and
848

properly respected?
849

Answer: [Yes]
850

Justification: The assets used in this paper are credited and the license is respected.
851

Guidelines:
852

• The answer NA means that the paper does not use existing assets.
853

• The authors should cite the original paper that produced the code package or dataset.
854

• The authors should state which version of the asset is used and, if possible, include a
855

URL.
856

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
857

• For scraped data from a particular source (e.g., website), the copyright and terms of
858

service of that source should be provided.
859

• If assets are released, the license, copyright information, and terms of use in the
860

package should be provided. For popular datasets, paperswithcode.com/datasets
861

has curated licenses for some datasets. Their licensing guide can help determine the
862

license of a dataset.
863

• For existing datasets that are re-packaged, both the original license and the license of
864

the derived asset (if it has changed) should be provided.
865

22


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
866

the asset’s creators.
867

13. New Assets
868

Question: Are new assets introduced in the paper well documented and is the documentation
869

provided alongside the assets?
870

Answer: [NA]
871

Justification: This paper does not release new assets.
872

Guidelines:
873

• The answer NA means that the paper does not release new assets.
874

• Researchers should communicate the details of the dataset/code/model as part of their
875

submissions via structured templates. This includes details about training, license,
876

limitations, etc.
877

• The paper should discuss whether and how consent was obtained from people whose
878

asset is used.
879

• At submission time, remember to anonymize your assets (if applicable). You can either
880

create an anonymized URL or include an anonymized zip file.
881

14. Crowdsourcing and Research with Human Subjects
882

Question: For crowdsourcing experiments and research with human subjects, does the paper
883

include the full text of instructions given to participants and screenshots, if applicable, as
884

well as details about compensation (if any)?
885

Answer: [NA]
886

Justification: This paper does not involve crowdsourcing nor research with human subjects.
887

Guidelines:
888

• The answer NA means that the paper does not involve crowdsourcing nor research with
889

human subjects.
890

• Including this information in the supplemental material is fine, but if the main contribu-
891

tion of the paper involves human subjects, then as much detail as possible should be
892

included in the main paper.
893

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
894

or other labor should be paid at least the minimum wage in the country of the data
895

collector.
896

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
897

Subjects
898

Question: Does the paper describe potential risks incurred by study participants, whether
899

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
900

approvals (or an equivalent approval/review based on the requirements of your country or
901

institution) were obtained?
902

Answer: [NA]
903

Justification: This paper does not involve crowdsourcing nor research with human subjects.
904

Guidelines:
905

• The answer NA means that the paper does not involve crowdsourcing nor research with
906

human subjects.
907

• Depending on the country in which research is conducted, IRB approval (or equivalent)
908

may be required for any human subjects research. If you obtained IRB approval, you
909

should clearly state this in the paper.
910

• We recognize that the procedures for this may vary significantly between institutions
911

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
912

guidelines for their institution.
913

• For initial submissions, do not include any information that would break anonymity (if
914

applicable), such as the institution conducting the review.
915

23


---Page Break---
