DELT: A Simple Diversity-driven EarlyLate Training
for Dataset Distillation

Anonymous Author(s)
Affiliation
Address
email

Abstract

Recent advances in dataset distillation have led to solutions in two main directions.
1

The conventional batch-to-batch matching mechanism is ideal for small-scale
2

datasets and includes bi-level optimization methods on models and syntheses,
3

such as FRePo, RCIG, and RaT-BPTT, as well as other methods like distribution
4

matching, gradient matching, and weight trajectory matching. Conversely, batch-to-
5

global matching typifies decoupled methods, which are particularly advantageous
6

for large-scale datasets. This approach has garnered substantial interest within the
7

community, as seen in SRe2L, G-VBSM, WMDD, and CDA. A primary challenge
8

with the second approach is the lack of diversity among syntheses within each
9

class since samples are optimized independently and the same global supervision
10

signals are reused across different synthetic images. In this study, we propose a
11

new EarlyLate training scheme to enhance the diversity of images in batch-to-
12

global matching with less computation. Our approach is conceptually simple yet
13

effective, it partitions predefined IPC samples into smaller subtasks and employs
14

local optimizations to distill each subset into distributions from distinct phases,
15

reducing the uniformity induced by the unified optimization process. These distilled
16

images from the subtasks demonstrate effective generalization when applied to
17

the entire task. We conducted extensive experiments on CIFAR, Tiny-ImageNet,
18

ImageNet-1K, and its sub-datasets. Our empirical results demonstrate that the
19

proposed approach significantly improves over previous state-of-the-art methods
20

under various IPCs1.
21

1
Introduction
22

Prior Dataset Distillation
Our DELT Distillation

Prior
Ours

Computation
!×#
# + # −./ … + ./

Early-optimized image
Late-optimized image
IPC1
IPCN

Our DELT generated images

IPC1:N
IPC1:N
batch-to-global 

matching

batch-to-global 

matching

…

…

…

…

Full
Full

IPC1

IPC2

IPCN

IPC1

IPC2

IPCN

#!

#"

#

#

#

##

bird
bird

Figure 1: Distill datasets to IPC𝑁requires 𝑁∗𝑇
iterations in traditional distillation processes
(left) but fewer iteration processes (right).

In the era of large models and large datasets, dataset
23

distillation has emerged as a crucial strategy to en-
24

hance training efficiency and make AI technologies
25

more accessible and affordable for the general public.
26

Previous approaches [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
27

primarily employ a batch-to-batch matching tech-
28

nique, where information like features, gradients,
29

and trajectories from a local original data batch are
30

used to supervise and train a corresponding batch
31

of generated data. This method’s strength lies in its
32

ability to capture fine-grained information from the
33

original data, as each batch’s supervision signals vary.
34

However, the downside is the necessity to repeatedly
35

1represents 𝑛Images Per Class for the distilled dataset.

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Class Index

(1) SRe2L

(3) Ours [1×1 initialized]

(2) CDA

Figure 2: Left: Intra-class semantic cosine similarity after a pretrained ResNet-18 model on ImageNet-
1K dataset, lower values are better. Right: Synthetic images from SRe2L, CDA and our DELT.

input both original and generated data for each training iteration, which significantly increases memory
36

usage and computational costs. Recently, a new decoupled method [11, 12, 13] has been proposed
37

to separate the model training and data synthesis, also it leverages the batch-to-global matching to
38

avoid inputting original data during distilled data generation. This solution has demonstrated great
39

advantage on large-scale datasets like ImageNet-1K [11, 14] and ImageNet-21K [12]. However, as
40

shown in Fig. 2 right subfigure, a significant limitation of this method is its strategy of synthesizing
41

each data point individually, where supervision is repetitively applied across various synthetic images.
42

For instance, SRe2L[11] utilizes globally-counted layer-wise running means and variances from the
43

pre-trained model for supervising different intra-class image synthesis. This methodology results in a
44

pronounced lack of diversity within the same category of generated images.
45

To address this issue, previous studies such as G-VBSM [14] and RDED [15] have been conducted.
46

Specifically, G-VBSM [14] introduces a framework that utilizes a diverse set of local-match-global
47

matching signals derived from multiple backbones and statistical metrics, offering more precise and
48

effective matching than the singular model. However, as the diversity of matching models grows, the
49

overall complexity of the framework also increases, thus diminishing its conciseness. RDED [15]
50

crops each original image into multiple patches and ranks these using realism scores generated by an
51

observer model. Then it amalgamates every four chosen patches from previous stage into a single new
52

image, maintaining the resolution of the original images, and produce IPC-numbered distilled images
53

for each class. While RDED is effective for selecting and combining data, it does not enhance or
54

optimize the visual content within the distilled dataset. Thus, the diversity and richness of information
55

it encapsulates largely dependent on the distribution of the original dataset.
56

Our solution, termed the EarlyLate training scheme, is straightforward and also orthogonal to
57

these prior methods: by initializing each image in the same category at a different starting point for
58

optimization, we ensure that the final optimized results vary across images. We also use teacher-ranked
59

real image patches to initialize the synthetic images. This prevents some images from being short-
60

optimized and ensures they provide sufficient information. As shown in Fig. 1 of the computation
61

comparison, our approach not only enhances intra-class diversity but also significantly reduces the
62

computational load of the training process. Specifically, while conventional training requires 𝑇
63

optimization iterations per image or batch, in our EarlyLate scheme, the first image undergoes 𝑇1
64

iterations (where 𝑇1 = 𝑇). Subsequent batches are processed with progressively fewer iterations, such
65

as 𝑇2 (𝑇2 = 𝑇1 −RI2) for the next set, and so forth. The iterations for the final batch are reduced
66

to RI which is 1/𝑗of the standard count (where typically 𝑗= 4 or 8), meaning the total number of
67

optimization iterations required is just about 2/3 of prior batch-to-global matching methods, such as
68

SRe2L and CDA. We further visualize the average cosine similarity between each sample of 50 IPCs
69

with the associated cluster centroid within the same class on ImageNet-1K, as shown in Fig. 2 left
70

subfigure, DELT shows significantly better diversity than other counterpart methods across all classes.
71

We perform extensive experiments on datasets of CIFAR-10, Tiny-ImageNet, ImageNet-1K and its
72

subsets. On ImageNet-1K, our proposed approach achieves 66.1% under IPC 50 with ResNet-101,
73

outperforming previous state-of-the-art RDED by 4.9%. On small-scale datasets of CIFAR-10, our
74

approach also obtains 2.5% and 19.2% improvement over RDED and SRe2L using ResNet-101.
75

Our main contributions in this work are as follows:
76

• We propose a simple yet effective EarlyLate training scheme for dataset distillation to
77

enhance the intra-class diversity of synthetic images from batch-to-global matching.
78

2RI is the number of round iterations and will be introduced in Sec. 4.3.

2


---Page Break---
• We demonstrate empirically that the proposed method can generate optimized images at
79

different distances from their initializations, to enlarge informativeness among generations.
80

• We conducted extensive experiments and ablations on various datasets across different scales
81

to prove the effectiveness of the proposed approach3.
82

2
Related Work
83

Dataset Distillation. Dataset distillation or condensation [1] focuses on creating a compact yet
84

representative subset from a large original dataset. This enables more efficient model training while
85

maintaining the ability to evaluate on the original test data distribution and achieve satisfactory
86

performance. Previous works [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] mainly designed how to better match
87

the distribution between original data and generated data in a batch-to-batch manner, such as the
88

distribution of features [6], gradients [2], or the model weight trajectories [4, 8]. The primary
89

optimization method used is bi-level optimization [16, 17], which involves optimizing model
90

parameters and updating images simultaneously. For instance, using gradient matching, the process
91

can be formulated as to minimize the gradient distance:
92

min
S∈R𝑁×𝑑𝐷(∇𝜃ℓ(S; 𝜃), ∇𝜃ℓ(T; 𝜃)) = 𝐷(S, T; 𝜃)
(1)

where the function 𝐷(·, ·) is defined as a distance metric such as MSE [18], 𝜃denotes the model
93

parameters, and ∇𝜃ℓ(·; 𝜃) represents the gradient, utilizing either the original dataset T or its synthetic
94

version S. 𝑁is the number of 𝑑-dimensional synthetic data. During distillation, the synthetic dataset
95

S and model 𝜃are updated alternatively,
96

S ←S −𝜆∇S𝐷(S, T; 𝜃),
𝜃←𝜃−𝜂∇𝜃ℓ(𝜃; S),
(2)

where 𝜆and 𝜂are learning rates designated for S and 𝜃, respectively.
97

Batch-to-global matching used in [11, 14, 12, 13] tracks the distribution of BN statistics derived from
98

the original dataset for the local batch synthetic data, the formulation can be:
99

min
S∈R𝑁×𝑑(
∑︁

𝑙

𝜇𝑙(S) −BNRM
𝑙

2 +
∑︁

𝑙

𝜎2
𝑙(S) −BNRV
𝑙

2)
(3)

where 𝑙is the index of BN layer, 𝜇𝑙(S) and 𝜎2
𝑙(S) are mean and variance. BNRM
𝑙
and BNRV
𝑙
are
100

running mean and running variance in the pre-trained model at 𝑙-th layer, which are globally counted.
101

Fig. 3 illustrates the difference of batch-to-batch and batch-to-global matching mechanisms, where 𝑏
102

represents a local batch in data T and S.
103

(1) Batch–to-Batch Matching

(2) Batch–to-Global Matching

Global mean,

var statistics

Batch mean, var

statistics

weight pretraining

weight sharing

13
Gradient/Trajectory/Feature, etc.

Gradient/Trajectory/Feature, etc.

ℒ2

ℒ2

ℒ+,

03

03

Original
dataset

.

.

.1

! represents a local

batch in # and $

Figure 3: Batch–to-batch vs. batch-to-global
matching in dataset distillation. 𝜃𝑓indicates
weights are pretrained and frozen in this stage.

Moreover, for the recent advances of multi-stage
104

dataset distillation methods, MDC [10] proposes to
105

compress multiple condensation processes into a
106

single one by including an adaptive subset loss on
107

top of the basic condensation loss, so that to obtain
108

datasets with multiple sizes. PDD [9] generates mul-
109

tiple small batches of synthetic images, each batch is
110

conditioned on the accumulated data from previous
111

batches. Unlike PDD, our current synthetic batch
112

is independent with different operation iterations
113

and not relevant to any previous batches. D3 [19]
114

partitions large datasets into smaller subtasks and
115

employs locally trained experts to distill each subset
116

into distributions. These distilled distributions from
117

the subtasks demonstrate effective generalization
118

when applied to the entire task.
119

Initialization. Weight initialization [20, 21, 22, 23] is pivotal in training neural networks, significantly
120

influencing their optimization process. Proper initialization is essential for ensuring model convergence
121

and mitigating issues such as gradient vanishing. Recently, weight selection [24] introduces a strategy
122

for initializing smaller models by selecting a subset of weights from a pretrained larger model. This
123

3Our synthetic images on ImageNet-1K are available anonymously at link.

3


---Page Break---
Scheduler
+
+

+
Concatenation node

Images Per Class whose rank 

CDA Gradient updates for 
 iterations

Figure 4: The proposed DELT learning procedure via a multi-round EarlyLate scheme.

method facilitates the transfer of learned attributes from the pretrained weights, enhancing the smaller
124

model’s performance. Weight subcloning [25] involves manipulating the pretrained model to derive a
125

correspondingly scaled-down version with equivalent initialization. This involves two main steps:
126

initially, it applies a neuron importance ranking to reduce the embedding dimension per layer within
127

the pretrained model. Subsequently, it eliminates blocks from the transformer model to align with the
128

layer count of the scaled-down network.
129

This work focuses on data initialization for generation processes. Few studies have examined this
130

angle. While, PCA-K [26] appears to be the most relevant. It employs an initialization method that
131

involves drawing samples from a distribution that accurately mirrors and is easily sampled from the
132

training distribution. During training, it is possible to retrieve some details from the original image
133

using the initial noisy sample, which at best provides a blurred representation of the original image.
134

3
Our Approach
135

Preliminaries. The objective of a regular dataset distillation task is to generate a compact synthetic
136

dataset S = {( ˆ𝒙1, ˆ𝒚1) , . . . ,   ˆ𝒙|S|, ˆ𝒚|S|
} as a student dataset that captures a substantial amount of
137

the information from a larger labeled dataset T = {(𝒙1, 𝒚1) , . . . ,  𝒙| T|, 𝒚| T|
}, which serves as the
138

teacher dataset. Here, ˆ𝒚represents the soft label for the synthetic sample ˆ𝒙, and the size of S is much
139

smaller than T, yet it retains the essential information of the original dataset T. The learning goal
140

using this distilled dataset is to train a post-validation model with parameters 𝜽:
141

𝜽S = arg min
𝜽
LS(𝜽),
(4)

142

LS(𝜽) = E( ˆ𝒙, ˆ𝒚)∈S

ℓ(𝜙𝜽S ( ˆ𝒙), ˆ𝒚; 𝜽)

,
(5)
where ℓis a standard loss function such as soft cross-entropy and 𝜙𝜽S represents the model.
143

The primary aim of dataset distillation is to produce synthetic data that ensures minimal performance
144

difference between models trained on the synthetic dataset S and those trained on the original dataset
145

T using validation data 𝑉. The optimization procedure for generating S is given by:
146

arg min
S,|S|


sup
ℓ 𝜙𝜽T (𝒙𝑣𝑎𝑙), 𝒚𝑣𝑎𝑙
 −ℓ 𝜙𝜽S (𝒙𝑣𝑎𝑙), 𝒚𝑣𝑎𝑙
	

(𝒙𝑣𝑎𝑙,𝒚𝑣𝑎𝑙)∼𝑉

.
(6)

where (𝒙𝑣𝑎𝑙, 𝒚𝑣𝑎𝑙) are the sample and label pairs in the validation set of the real dataset T. The
147

learning task then focuses on the <data, label> pairs within S, maintaining a balanced representation
148

of distilled data across each class.
149

2

3

5

1

4

Teacher

Ranker

1

2

3

4

5

3

2

4

IPC initialization

Figure 5: Selection criteria with a
teach ranker.

Initialization. Previous dataset distillation methods [11, 14,
150

12] on large-scale datasets like ImageNet-1K and 21K employ
151

Gaussian noise by default for data initialization in the synthesis
152

phase. However, Gaussian noise is random and lacks any
153

semantic information. Intuitively, using real images provide a
154

more meaningful and structured starting point, and this struc-
155

tured start can lead to quicker convergence during optimization
156

because the initial data already contains useful features and
157

patterns that are closer to the target distribution, which further
158

4


---Page Break---
enhances realism, quality, and generalization of the synthesized images. As shown in Fig. 2 right
159

subfigure, our generated images exhibit both diversity and a high degree of realism in some cases.
160

Selection Criteria. Here, we introduce how to select real image patches to initialize the synthetic
161

images. In our final syntheses, a significant fraction of our data has been subject to limited optimization
162

iterations, making effective initialization crucial. A proper initialization also dramatically minimizes
163

the overall computational load required for the updating on data. Prior approach [15] has demonstrated
164

that choosing representative data patches from the original dataset without training can yield favorable
165

performance without any additional training. Our observation, however, underscores that applying
166

iterative refinement to original patches can lead to markedly improved results. As illustrated in Fig. 1,
167

our selection criterion is based on a pretrained teacher model as a ranker, we calculate all patches’
168

probabilities and sort them as the initialization pool. Then, we choose lowest, medium, or highest
169

probability patches as the initialization for our optimization.
170

Diversity-driven IPC Concatenation Training. As shown in Fig. 4, to further emphasize diversity
171

and avoid potential distribution bias from initialization, we optimize the initialized images starting
172

from different points. The motivation behind this design is that different data samples require varying
173

numbers of iterations to converge which is similar to the early stopping idea [27]. Importantly, as
174

images become easier to predict with more updates by class labels, training primarily on easy data
175

points can hinder model generalization. Therefore, our method enhances generalization by generating
176

data samples with varying difficulty levels, acting as a regularizer by limiting the optimization process
177

to a smaller volume of image pixel space. Previous work [28] studies how to perform early stopping
178

training on different layers’ weights of the model with progressive retraining to mitigate noisy labels.
179

We are pioneering to study how to leverage early-late training when optimizing data. Moreover, we
180

improve the efficiency of our approach by performing gradient updates in a single scan. Initially, we
181

conduct a single gradient loop, continually introducing new data for distillation by concatenating them
182

at different time stamps. Consequently, the 𝑀batch receives the synthetic images of all preceding
183

batches, IPC0:𝑀𝑘−1, as final generations. This process can be simplified as follows:
184

IPC0:𝑀𝑘−1 = [ ˆ𝒙0, ˆ𝒙1, . . . , ˆ𝒙𝑘−1
|              {z              }

IPC0:𝑘−1

, . . .

|                    {z                    }

...

, ˆ𝒙𝑀𝑘−1

|                               {z                               }

IPC0:𝑀𝑘−1

]
(7)

where [ ˆ𝒙0, ˆ𝒙1, . . . , ˆ𝒙𝑀𝑘−1] refers to the concatenation of the generated image. 𝑀is the number
185

of batches, 𝑘is the number of generated images in each batch. We train these different batches at
186

different starting points, each batch goes through a completed learning phase, but the total number of
187

iterations varies. Then, the multiple IPCs of ˆ𝒙are concatenated into a simple batch. Because of its
188

early-late training property, we refer to this simple training scheme as EarlyLate training.
189

Training Procedure. As illustrated in Fig. 4, our learning procedure is extremely simple using an
190

incremental learning process: We split the total IPCs to be learned into multiple batches. The training
191

begins with the first batch. Following a predefined number of iterations, the second batch commences
192

its iterative training, and this process continues sequentially with subsequent batches. Batch-to-global
193

matching algorithm [12] of Eq. 3 has been utilized between each round.
194

4
Experiments
195

4.1
Datasets and Results Details
196

We first run DELT on five standard benchmark tests including CIFAR-10 (10 classes) [29], Tiny-
197

ImageNet (200 classes) [30], ImageNet-1K (1,000 classes) [31] and it variants of ImageNette (10
198

classes) [32], and ImageNet-100 (100 classes) [33] with performances reported in Table 1 and Table 2.
199

The evaluation protocol is following prior works [15, 11]. We compare DELT to six baseline dataset
200

distillation algorithms including Matching Training Trajectories (MTT) [4], Improved Distribution
201

Matching (IDM) [34], TrajEctory Matching with Constant Memory (TESLA) [8], Squeeze-Recover-
202

Relabel (SRe2L) [11], Difficulty-Aligned Trajectory-Matching (DATM) [35], Realistic-Diverse-
203

Efficient Dataset Distillation (RDED) [15]. Following previous dataset distillation methods [2, 15,
204

11], we use ConvNet [36], ResNet-18/ResNet-101 [37], EfficientNet-B0 [38], MobileNet-V2 [39],
205

MnasNet1_3 [40], and RegNet-Y-8GF [41], as our backbone for training or post-validation. All our
206

experiments are conducted on 4 NVIDIA RTX 4090 GPUs.
207

5


---Page Break---
ResNet-18
ResNet-101
MobileNet-v2

Dataset
IPC
SRe2L [11]
RDED [15]
Ours
SRe2L [11]
RDED [15]
Ours
Ours

1
16.6 ± 0.9
22.9 ± 0.4
24.0 ± 0.8
13.7 ± 0.2
18.7 ± 0.1
20.4 ± 1.0 20.2 ± 0.4
CIFAR-10
10
29.3 ± 0.5
37.1 ± 0.3
43.0 ± 0.9
24.3 ± 0.6
33.7 ± 0.3
37.4 ± 1.2 29.3 ± 0.3
50
45.0 ± 0.7
62.1 ± 0.1
64.9 ± 0.9
34.9 ± 0.1
51.6 ± 0.4
54.1 ± 0.8 42.9 ± 2.2

1
19.1 ± 1.1
35.8 ± 1.0
24.1 ± 1.8
15.8 ± 0.6
25.1 ± 2.7
19.4 ± 1.7
19.1 ± 1.0
ImageNette
10
29.4 ± 3.0
61.4 ± 0.4
66.0 ± 1.4
23.4 ± 0.8
54.0 ± 0.4
55.4 ± 6.2 64.7 ± 1.4
50
40.9 ± 0.3
80.4 ± 0.4
88.2 ± 1.2
36.5 ± 0.7
75.0 ± 1.2
83.3 ± 1.1 85.7 ± 0.4

Tiny-ImageNet
1
2.62 ± 0.1
9.7 ± 0.4
9.3 ± 0.5
1.9 ± 0.1
3.8 ± 0.1
5.6 ± 1.0
3.5 ± 0.5
10
16.1 ± 0.2
41.9 ± 0.2
43.0 ± 0.1
14.6 ± 1.1
22.9 ± 3.3
42.8 ± 0.9 26.5 ± 0.5
50
41.1 ± 0.4
58.2 ± 0.1
55.7 ± 0.5
42.5 ± 0.2
41.2 ± 0.4
58.5 ± 0.3 51.3 ± 0.5

ImageNet-100
10
9.5 ± 0.4
36.0 ± 0.3
28.2 ± 1.5
6.4 ± 0.1
33.9 ± 0.1
22.4 ± 3.3
15.8 ± 0.2
50
27.0 ± 0.4
61.6 ± 0.1
67.9 ± 0.6
25.7 ± 0.3
66.0 ± 0.6
70.8 ± 2.3
55.0 ± 1.8
100
-
74.5 ± 0.4
75.1 ± 0.2
-
73.5 ± 0.8
77.6 ± 1.8
76.7 ± 0.3

ImageNet-1K
10
21.3 ± 0.6
42.0 ± 0.1
45.8 ± 0.1
30.9 ± 0.1
48.3 ± 1.0
48.5 ± 1.6
35.1 ± 0.5
50
46.8 ± 0.2
56.5 ± 0.1
59.2 ± 0.4
60.8 ± 0.5
61.2 ± 0.4
66.1 ± 0.5
56.2 ± 0.3
100
52.8 ± 0.3
59.8 ± 0.1
62.4 ± 0.2
62.8 ± 0.2
-
67.6 ± 0.3
58.9 ± 0.3
Table 1: Comparison with SOTA dataset distillation methods using relatively large-scale backbones
on five benchmarks across different scales. MobileNet-v2 is modified to match the low resolutions of
CIFAR-10 and Tiny-ImageNet following [42]. Due to the table space limitation, some other methods
that are weaker than RDED are not listed, such as CDA and G-VBSM. Since IPC 1 is not applicable
to use EarlyLate strategy and the single image in each class is optimized with a constant iteration.

ConvNet

Dataset
IPC
MTT [4]
IDM [34]
TESLA [8]
DATM [35]
RDED [15]
Ours

1
47.7 ± 0.9
-
-
-
33.8 ± 0.8
29.8 ± 1.4
ImageNette
10
63.0 ± 1.3
-
-
-
63.2 ± 0.7
51.7 ± 1.2
50
-
-
-
-
83.8 ± 0.2
84.5 ± 0.4

Tiny-ImageNet
1
8.8 ± 0.3
10.1 ± 0.2
-
17.1 ± 0.3
12.0 ± 0.1
12.4 ± 0.8
10
23.2 ± 0.2
21.9 ± 0.3
-
31.1 ± 0.3
39.6 ± 0.1
40.0 ± 0.4
50
28.0 ± 0.3
27.7 ± 0.3
-
39.7 ± 0.3
47.6 ± 0.2
48.6 ± 0.2

ImageNet-100
10
-
17.1 ± 0.6
-
-
29.6 ± 0.1
24.7 ± 1.5
50
-
26.3 ± 0.4
-
-
50.2 ± 0.2
51.9 ± 1.1
100
-
-
-
-
58.6 ± 0.4
61.5 ± 0.5

ImageNet-1K
1
-
-
7.7 ± 0.2
-
6.4 ± 0.1
8.8 ± 0.5
10
-
-
17.8 ± 1.3
-
20.4 ± 0.1
31.3 ± 0.8
50
-
-
27.9 ± 1.2
-
38.4 ± 0.2
41.7 ± 0.1
Table 2: Comparison with SOTA dataset distillation methods using small-scale backbone architecture
on four benchmark datasets. Following [4, 34, 15], Conv-3 is used for CIFAR-10, Conv-4 for Tiny-
ImageNet and ImageNet-1K, Conv-5 for ImageNette, and Conv-6 for ImageNet-100 and ImageNet-1K.
Entries marked with “-” are missing due to scalability issue.

As shown in Table 1, our approach establishes the new state-of-the-art accuracy in 13 out of 15 of
208

the configurations on five datasets from small-scale CIFAR-10 to large-scale ImageNet-1K using
209

relatively large backbone architecture of ResNet-101, in many cases with significant margins of
210

improvement. The results using small-scale architecture ConvNet are shown in Table 2, our approach
211

also achieves the state-of-the-art accuracy in 8 out of 12 of the configurations on four datasets.
212

4.2
Cross-architecture generalization
213

An important characteristic of distilled datasets is their effectiveness in generalizing to novel training
214

architectures. In this context, we assess the transferability of DELT’s distilled datasets tailored for
215

ImageNet-1K with 10 images per class. Following previous studies [11, 15], we test our models
216

using five distinct architectures: ResNet-18 [37], MobileNet-V2 [39], MnasNet1_3 [40], EfficientNet-
217

B0 [38], and RegNet-Y-8GF [41]. As shown in Table 4, our proposed approach demonstrates
218

significant better performance than other competitive methods on all these architectures.
219

4.3
Ablation Study
220

Mosaic splicing pattern. Mosaic stitching method [43] in RDED selects four crops from the train set
221

as the optimal hyper-parameter, and puts the contents of the four crops into a synthetic image that is
222

directly used for post-validation. In this work, considering that we use different difficulty levels of
223

6


---Page Break---
3×3
4×4
5×5
6×6
Figure 6: Mosaic splicing patterns on ImageNet-1K using real image patches as the initialization.
In each block, the left column is the starting real image initialized samples and right is the final
optimized syntheses. From top to bottom are images generated by early training and late training.

selection for initialization, we examine different strategies of the Mosaic splicing patterns, including
224

1 × 1, 2 × 2, 3 × 3, 4 × 4, and 5 × 5 patches, as illustrated in Fig. 11. The ablation results are shown in
225

Table 3, it can be observed that 1 × 1 achieves the best accuracy.
226

Initialization. We examine how different initialization strategies affect final performance, including:
227

choosing lowest probability crops, medium probability crops and highest probability crops. Our results
228

are shown in Table 3. Overall, the performance gap between different strategies is not significant, and
229

selecting the medium probability crops as the initialization achieves the best accuracy.
230

Optimization iterations. We examine two types of optimization iterations: maximum iteration
231

(MI) for the earliest batch training and round iteration (RI). MI presents the number of optimization
232

iterations that the earliest batch goes through. RI represents the number of iterations used for each
233

round in Fig. 4. It essentially indicates the iteration gap between the optimization of two adjacent
234

batches. As shown in Table 3, we test MI values of 1K, 2K, and 4K, using 500 and 1K iterations for
235

each RI. Note that when MI is set to 1K, it is not feasible to use 1K as RI. The results show that 4K
236

(same as [11, 12]) MI and 500 RI achieves the best accuracy.
237

Early-only vs. EarlyLate. Early-only is equivalent to using constant MI to optimize each image. The
238

method will transform to baseline batch-to-global matching of CDA [12] + real image initialization.
239

Our results in Table 3 clearly show that the EarlyLate training bring a significant improvement on
240

final performance. More importantly, this strategy is the key factor in enhancing generation diversity.
241

Real image stitching vs. Minimax diffusion vs. Ours. We further compare the performance of our
242

approach with real image stitching [15] and diffusion generation [44]. The results are presented in
243

Table 3d. While the first two methods produce more realistic images, each image contains limited
244

information. In contrast, our method achieves the best final performance.
245

4.4
Computational Analysis
246

For image optimization-based methods like SRe2L and CDA, the total computational cost is calculated
247

as 𝑁× 𝑇, where 𝑁is the MI. In our EarlyLate scheme, the first batch images undergo 𝑇1 iterations
248

(where 𝑇1 = 𝑇). Subsequent batches are processed with progressively fewer iterations, such as 𝑇2
249

(𝑇2 = 𝑇1 −RI) for the next set, and so forth. The iterations for the final batch are reduced to RI which
250

is 1/𝑗of the standard count (where 𝑗= 4 or 8 in our ablation), the total number of our optimization
251

iterations required is 𝑁× 𝑇−𝑗( 𝑗−1)

2
RI, which is roughly 2/3 of prior batch-to-global matching
252

methods. Our real time consumptions for data generation are shown in Table 5, note that the smaller
253

the dataset like CIFAR, the more time is spent on loading and processing the data, rather than training.
254

4.5
Visualization of DELT
255

Fig. 7 illustrates a comprehensive visual comparison between randomly selected synthetic images from
256

our distilled dataset and those from the real image patches [15], MinimaxDiffusion [44], MTT [4],
257

IDC [45], SRe2L [11], SCDD [46], CDA [12] and G-VBSM [14] distilled data. It can be observed that
258

7


---Page Break---
Table 3: Ablation experiments on various aspects of our framework with ResNet-18 on ImageNet-1K.

# Patches
Top 1 acc

1 × 1
57.57
2 × 2
56.92
3 × 3
56.62
4 × 4
56.71
5 × 5
56.51

(a) Number of patches. Ablation on initializing
different numbers of scoring patches. Results
are from ResNet-18 on ImageNet-1K for 500
iterations to synthesize 50 IPCs.

Selection criteria
Top 1 acc

Lowest probability
57.55
Medium probability
57.67
Highest probability
57.03

(b) Selection criteria. Initializing 1 × 1 images
selected according to teacher model’s probability

Iterations
Round Iterations
500
1K

1K
44.87
n/a
2K
45.61
44.40
4K
46.42
44.66

(c) Round Iterations. Top-1 acc. of our method for IPC
10 using different round iterations with ResNet-18.

Dataset
CDA [12] + Our init.
Ours

ImageNet-1K
43.5
45.8
Tiny-ImageNet
42.2
43.0
CIFAR-10
39.4
43.0
(d) Ablation on init. and EarlyLate under IPC 10.

IPC
RDED [15]
MinimaxDiffusion [44]
Ours

10
42.0
44.3
45.8
50
56.5
58.6
59.2
(e) Comparison with real and diffusion generated data.

Table 4: Cross-architecture generalization. Results are evaluated on IPC 10.

Recover \Validation
ResNet-18
EfficientNet-B0
MobileNet-V2
MnasNet1_3
RegNet-Y-8GF

ResNet-18

SRe2L [11]
41.9
41.9
33.1
39.3
51.5
CDA [12]
42.2
43.9
34.2
39.7
52.9
G-VBSM [14]
41.4
42.6
33.5
40.1
52.2
RDED [15]
42.3
42.8
34.4
40.0
54.8
Ours
46.4(+4.1)
47.1(+4.3)
36.1(+1.7)
40.7(+0.7)
57.5(+2.7)

Table 5: Actual computational consumption and analysis (hours under IPC 50) in data synthesis
with image optimization-based methods on a single NVIDIA 4090 GPU. “RI” represents round
iterations. A total 4K iterations are used for all methods and datasets to ensure fair comparisons.

Dataset (hours)

Method
ImageNet-1K
Tiny-ImageNet
CIFAR-10

G-VBSM [14]
114.1
5.5
0.195
SRe2L [11]
29.0
5.0
0.084
CDA [12]
29.0
5.0
0.084
Ours (RI = 500)
17.6(↓39.3%)
3.4(↓32.0%)
0.083(↓1.1%)
Ours (RI = 1K)
18.8(↓35.2%)
3.6(↓28.0%)
0.084(↓0.0%)

the images generated by each method have their own characteristics. MinimaxDiffusion leverages the
259

diffusion model to synthesize images which is close to the real ones. However, as in our above ablation,
260

both real and diffusion-generated data are inferior to ours. MTT results show noticeable artifacts
261

and distortions, the objects in all images are located in the middle of the generations, the diversity is
262

limited. IDC results also show distorted and less recognizable dog images, but diversity is increased.
263

SRe2L exhibits some dog features but with significant distortions and similar simple background.
264

SCDD shows more recognizable dog features but still the color is simple and monochromatic, the
265

same situation happens in CDA. G-VBSM shows more colorful patterns, possibly due to recovery
266

from multiple different networks, but all generations are in the same pattern and the diversity is
267

not large. Our approach’s synthetic images exhibit a higher degree of diversity, including both
268

compressed distorted images from long-optimized initializations and clear, recognizable dog images
269

from short-optimized initializations, a unique capability not present in other methods.
270

4.6
Application I: Data-free Network Pruning
271

Our distilled dataset acts as a multifunctional training tool and boosts the adaptability for diverse
272

downstream applications. We validate its utility in the scenario of data-free network pruning [47].
273

Table 6 shows the applicability of our dataset in this task when pruning 50% weights, where it
274

significantly surpasses previous methods such as SRe2L and RDED under IPC 10 and 50.
275

8


---Page Break---
Ours
G-VBSM
CDA
SCDD
SRe2L
MTT
IDC
Real
M-Diffusion
Figure 7: Distilled dataset visualization compared with other image optimization-based methods.

Table 6: Accuracy of data-free network pruning using slimming [48] on VGG11-BN [49].

SRe2L [11]
RDED [15]
Ours

IPC 10
12.5
13.2
17.9(+4.7)
IPC 50
31.7
42.8
44.8(+2.0)

4.7
Application II: Continual Learning
276

100
200
300
400
500
600
700
800
900
Class Number

10

15

20

25

30

35

40

Top-1 Accuracy (%)

SRe2L
G-VBSM
Ours

Figure 8: Continual learning results.

We examine the effectiveness of DELT generated images
277

in the continual learning scenario. Following the setup in
278

prior studies [11, 6], we perform 100-step class-incremental
279

experiments on ImageNet-1K, comparing our results with
280

the baselines G-VBSM and SRe2L. As shown in Fig. 8,
281

our DELT distilled dataset significantly outperforms G-
282

VBSM, with an average improvement of about 10% in
283

100-step class-incremental learning task. This highlights
284

the significant benefits of deploying DELT, particularly in
285

mitigating the challenges of continual learning.
286

5
Conclusion
287

We have introduced a new training strategy, EarlyLate, to improve image diversity in batch-to-global
288

matching scenarios for dataset distillation. The proposed approach organizes predefined IPC samples
289

into smaller, manageable subtasks and utilizes local optimizations. This strategy helps in refining
290

each subset into distributions characteristic of different phases, thereby mitigating the homogeneity
291

typically caused by a singular optimization process. The images refined through this method exhibit
292

robust generalization across the entire task. We have extensively evaluated this approach on CIFAR-10
293

and 100, Tiny-ImageNet, ImageNet-1K, and its variants. Our empirical findings indicate that our
294

approach significantly outperforms prior state-of-the-art methods across various IPC configurations.
295

Limitations. Our method effectively avoids the issue of insufficient data diversity generated by
296

batch-to-global methods and reduces the computational cost of the generation process. However,
297

there is still a performance gap when training the model on our generated data compared to training
298

on the original dataset. Additionally, our short-optimized data exhibits similar semantic information
299

to the original images, which may potentially leak the privacy of the original dataset.
300

9


---Page Break---
References
301

[1] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A Efros. Dataset distillation. arXiv preprint
302

arXiv:1811.10959, 2018.
303

[2] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching. arXiv
304

preprint arXiv:2006.05929, 2020.
305

[3] Yongchao Zhou, Ehsan Nezhadarya, and Jimmy Ba. Dataset distillation using neural feature regression.
306

Advances in Neural Information Processing Systems, 35:9813–9827, 2022.
307

[4] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu. Dataset
308

distillation by matching training trajectories. In Proceedings of the IEEE/CVF Conference on Computer
309

Vision and Pattern Recognition, pages 4750–4759, 2022.
310

[5] Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sungroh Yoon. Dataset condensation
311

with contrastive signals. In International Conference on Machine Learning, pages 12352–12364. PMLR,
312

2022.
313

[6] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In IEEE/CVF Winter
314

Conference on Applications of Computer Vision, WACV 2023, Waikoloa, HI, USA, January 2-7, 2023,
315

2023.
316

[7] Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang. Dataset distillation via factorization.
317

Advances in Neural Information Processing Systems, 35:1100–1113, 2022.
318

[8] Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet-1k with
319

constant memory. In International Conference on Machine Learning, pages 6565–6590. PMLR, 2023.
320

[9] Xuxi Chen, Yu Yang, Zhangyang Wang, and Baharan Mirzasoleiman. Data distillation can be like
321

vodka: Distilling more times for better quality. In The Twelfth International Conference on Learning
322

Representations, 2024.
323

[10] Yang He, Lingao Xiao, Joey Tianyi Zhou, and Ivor Tsang. Multisize dataset condensation. ICLR, 2024.
324

[11] Zeyuan Yin, Eric Xing, and Zhiqiang Shen. Squeeze, recover and relabel: Dataset condensation at imagenet
325

scale from a new perspective. In NeurIPS, 2023.
326

[12] Zeyuan Yin and Zhiqiang Shen. Dataset distillation in large data era. arXiv preprint arXiv:2311.18838,
327

2023.
328

[13] Haoyang Liu, Tiancheng Xing, Luwei Li, Vibhu Dalal, Jingrui He, and Haohan Wang. Dataset distillation
329

via the wasserstein metric. arXiv preprint arXiv:2311.18531, 2023.
330

[14] Shitong Shao, Zeyuan Yin, Muxin Zhou, Xindong Zhang, and Zhiqiang Shen. Generalized large-scale data
331

condensation via various backbone and statistical matching. In CVPR, 2024.
332

[15] Peng Sun, Bei Shi, Daiwei Yu, and Tao Lin. On the diversity and realism of distilled dataset: An efficient
333

dataset distillation paradigm. In CVPR, 2024.
334

[16] Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. Investigating bi-level optimization
335

for learning and vision from a unified perspective: A survey and beyond. IEEE Transactions on Pattern
336

Analysis and Machine Intelligence, 44(12):10045–10067, 2021.
337

[17] Yihua Zhang, Prashant Khanduri, Ioannis Tsaknakis, Yuguang Yao, Mingyi Hong, and Sĳia Liu. An
338

introduction to bi-level optimization: Foundations and applications in signal processing and machine
339

learning. arXiv preprint arXiv:2308.00788, 2023.
340

[18] Zhou Wang and Alan C Bovik. Mean squared error: Love it or leave it? a new look at signal fidelity
341

measures. IEEE signal processing magazine, 26(1):98–117, 2009.
342

[19] Tian Qin, Zhiwei Deng, and David Alvarez-Melis.
Distributional dataset distillation with subtask
343

decomposition. arXiv preprint arXiv:2403.00999, 2024.
344

[20] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural
345

networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics,
346

pages 249–256. JMLR Workshop and Conference Proceedings, 2010.
347

10


---Page Break---
[21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
348

human-level performance on imagenet classification. In Proceedings of the IEEE international conference
349

on computer vision, pages 1026–1034, 2015.
350

[22] Dmytro Mishkin and Jiri Matas. All you need is a good init. arXiv preprint arXiv:1511.06422, 2015.
351

[23] Trieu H Trinh, Minh-Thang Luong, and Quoc V Le. Selfie: Self-supervised pretraining for image
352

embedding. arXiv preprint arXiv:1906.02940, 2019.
353

[24] Zhiqiu Xu, Yanjie Chen, Kirill Vishniakov, Yida Yin, Zhiqiang Shen, Trevor Darrell, Lingjie Liu, and
354

Zhuang Liu. Initializing models with larger ones. In The Twelfth International Conference on Learning
355

Representations, 2024.
356

[25] Mohammad Samragh, Mehrdad Farajtabar, Sachin Mehta, Raviteja Vemulapalli, Fartash Faghri, Devang
357

Naik, Oncel Tuzel, and Mohammad Rastegari. Weight subcloning: direct initialization of transformers
358

using larger pretrained ones. arXiv preprint arXiv:2312.09299, 2023.
359

[26] Jeffrey Zhang, Shao-Yu Chang, Kedan Li, and David Forsyth. Preserving image properties through
360

initializations in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of
361

Computer Vision, pages 5242–5250, 2024.
362

[27] Lutz Prechelt. Early stopping-but when? In Neural Networks: Tricks of the trade, pages 55–69. Springer,
363

2002.
364

[28] Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu.
365

Understanding and improving early stopping for learning with noisy labels. Advances in Neural Information
366

Processing Systems, 34:24392–24403, 2021.
367

[29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
368

[30] Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.
369

[31] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
370

image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255.
371

Ieee, 2009.
372

[32] Fastai. Fastai/imagenette: A smaller subset of 10 easily classified classes from imagenet, and a little more
373

french.
374

[33] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Computer Vision–ECCV
375

2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages
376

776–794. Springer, 2020.
377

[34] Ganlong Zhao, Guanbin Li, Yipeng Qin, and Yizhou Yu. Improved distribution matching for dataset
378

condensation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
379

pages 7856–7865, 2023.
380

[35] Ziyao Guo, Kai Wang, George Cazenavette, Hui Li, Kaipeng Zhang, and Yang You. Towards lossless
381

dataset distillation via difficulty-aligned trajectory matching. In The Twelfth International Conference on
382

Learning Representations, 2024.
383

[36] Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. In Proceedings
384

of the IEEE conference on computer vision and pattern recognition, pages 4367–4375, 2018.
385

[37] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
386

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
387

[38] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In
388

International conference on machine learning, pages 6105–6114. PMLR, 2019.
389

[39] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2:
390

Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and
391

pattern recognition, pages 4510–4520, 2018.
392

[40] Mingxing Tan, Bo Chen, Ruoming Pang, Vĳay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V
393

Le. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF
394

conference on computer vision and pattern recognition, pages 2820–2828, 2019.
395

11


---Page Break---
[41] Ilĳa Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network
396

design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
397

pages 10428–10436, 2020.
398

[42] Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang. Decoupled knowledge distillation. In
399

Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11953–11962,
400

2022.
401

[43] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy
402

of object detection. arXiv preprint arXiv:2004.10934, 2020.
403

[44] Jianyang Gu, Saeed Vahidian, Vyacheslav Kungurtsev, Haonan Wang, Wei Jiang, Yang You, and Yiran
404

Chen. Efficient dataset distillation via minimax diffusion. In CVPR, 2024.
405

[45] Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, Jung-Woo
406

Ha, and Hyun Oh Song. Dataset condensation via efficient synthetic-data parameterization. In Proceedings
407

of the 39th International Conference on Machine Learning, 2022.
408

[46] Muxin Zhou, Zeyuan Yin, Shitong Shao, and Zhiqiang Shen. Self-supervised dataset distillation: A good
409

compression is all you need. arXiv preprint arXiv:2404.07976, 2024.
410

[47] Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. arXiv
411

preprint arXiv:1507.06149, 2015.
412

[48] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning
413

efficient convolutional networks through network slimming. In Proceedings of the IEEE international
414

conference on computer vision, pages 2736–2744, 2017.
415

[49] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
416

recognition. arXiv preprint arXiv:1409.1556, 2014.
417

[50] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen,
418

Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep
419

learning library. Advances in neural information processing systems, 32, 2019.
420

[51] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data
421

augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on computer vision
422

and pattern recognition workshops, pages 702–703, 2020.
423

12


---Page Break---
Appendix
424

A
Broader Impacts
425

Our dataset distillation framework can significantly reduce the computational resources required for
426

training machine learning models. This leads to lower energy consumption and cost, making AI
427

more accessible and sustainable. By generating smaller, more manageable datasets, researchers and
428

developers can iterate and experiment more quickly, accelerating the pace of innovation in various
429

AI applications. However, condensed datasets might inadvertently amplify biases present in the
430

original data. If the distillation process does not adequately address bias, it could lead to unfair
431

or discriminatory AI systems. Also, simplifying datasets may lead to a loss of important nuances
432

and context, potentially degrading the performance of models in real-world applications where such
433

details are crucial. Moreover, the models may overfit to condensed data, indicating that models trained
434

on distilled datasets might perform well on the condensed data but poorly on more diverse real-world
435

data, limiting their generalizability and robustness.
436

B
Training Details
437

Table 7: Hyper-parameter settings.

(a) Validation settings

config
value

optimizer
AdamW

base learning rate
0.001 (all)
0.0025 (MobileNet-v2)
weight decay
0.01

batch size

100 (IPC50)
50 (IPC10)
10 (IPC1)
learning rate schedule
cosine decay
training epoch
300

augmentation

RandAugment
RandomResizedCrop
RandomHorizontalFlip

(b) Recovery settings

config
value

𝛼BN
0.01
optimizer
Adam
base learning rate
0.25
momentum
𝛽1, 𝛽2 = 0.5, 0.9
batch size
100
learning rate schedule
cosine decay
recovery iteration
4,000
round iteration
500 [IPC 10, 50, 100]
initialization
top medium
augmentation
RandomResizedCrop

(c) Dataset-specific settings in recovery

config
CIFAR10
Tiny-ImageNet
ImageNette
ImageNet-100
ImageNet-1K

RandAugment (m)
5
4
6
6
6
RandAugment (n)
4
3
2
2
2
RandAugment (mstd)
1.0
1.0
1.0
1.0
1.0

IPC1 Recovery Iterations

2K (R18)
500 (R18)
1K (R18)
-
3K (Conv4)
3K (R101)
500 (R101)
1K (R101)
-
-
2K (MobileNet)
500 (MobileNet)
2K (MobileNet)
-
-
-
1K (Conv4)
4K (Conv5)
-
-

For reproducibility, we provide all our hyper-parameter settings used in our experiments in Table 7,
438

we outline such details below.
439

Squeezing and Pre-trained models. Following the previous works [11, 12, 15], we use the official
440

PyTorch [50] pre-trained ResNet-18 model for ImageNet-1K, and we use the same official Torchvision
441

[50] code to produce our pre-trained models, ResNet-18 and ConvNet, for the other datasets.
442

Ranking. A crucial part of our method is initialization, we simply use ResNet-18 pre-trained models
443

to rank and select the top-medium images as initialization for all our datasets, except for ImageNet-100
444

where we simply extracted the top-medium images based on the rankings of the original ImageNet-1K.
445

Recovery. For our synthesis, we provide the details of the general hyper-parameters used for different
446

datasets, including ImageNet-1K, ImageNet-100, ImageNette, Tiny-ImageNet, and CIFAR10, in
447

Table 7b. Because synthesizing a single image per class, i.e., IPC 1, is quite special as we cannot use
448

13


---Page Break---
Figure 9: Synthetic image visualizations on Tiny-ImageNet generated by our DELT.

rounds, we apply different numbers of iterations based on both the dataset scale and the validation
449

teacher model as outlined in Table 7c.
450

Validation. This includes both the soft-label generation, Relabel in SRe2L, and evaluation, or
451

post-training. We outline such details in Table 7a. We use timm’s version of RandAugment [51] with
452

different settings depending on the synthesized dataset being validated as outlined in Table 7c.
453

C
More Visualizations
454

We provide more visualizations on synthetic Tiny-ImageNet, ImageNette and CIFAR-10 datasets. In
455

each figure, each column represents a different class, with images progressing from long optimization
456

at the top to short optimization at the bottom.
457

14


---Page Break---
Figure 10: Synthetic image visualizations on ImageNette generated by our DELT.

15


---Page Break---
Figure 11: Synthetic image visualizations on CIFAR-10 generated by our DELT.

16


---Page Break---
NeurIPS Paper Checklist
458

1. Claims
459

Question: Do the main claims made in the abstract and introduction accurately reflect the
460

paper’s contributions and scope?
461

Answer: [Yes]
462

Justification: We have clearly stated the contributions and scope of this paper.
463

Guidelines:
464

• The answer NA means that the abstract and introduction do not include the claims made
465

in the paper.
466

• The abstract and/or introduction should clearly state the claims made, including the
467

contributions made in the paper and important assumptions and limitations. A No or
468

NA answer to this question will not be perceived well by the reviewers.
469

• The claims made should match theoretical and experimental results, and reflect how
470

much the results can be expected to generalize to other settings.
471

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
472

are not attained by the paper.
473

2. Limitations
474

Question: Does the paper discuss the limitations of the work performed by the authors?
475

Answer: [Yes]
476

Justification: The limitations have been discussed in the conclusion section.
477

Guidelines:
478

• The answer NA means that the paper has no limitation while the answer No means that
479

the paper has limitations, but those are not discussed in the paper.
480

• The authors are encouraged to create a separate "Limitations" section in their paper.
481

• The paper should point out any strong assumptions and how robust the results are to
482

violations of these assumptions (e.g., independence assumptions, noiseless settings,
483

model well-specification, asymptotic approximations only holding locally). The authors
484

should reflect on how these assumptions might be violated in practice and what the
485

implications would be.
486

• The authors should reflect on the scope of the claims made, e.g., if the approach was
487

only tested on a few datasets or with a few runs. In general, empirical results often
488

depend on implicit assumptions, which should be articulated.
489

• The authors should reflect on the factors that influence the performance of the approach.
490

For example, a facial recognition algorithm may perform poorly when image resolution
491

is low or images are taken in low lighting. Or a speech-to-text system might not be
492

used reliably to provide closed captions for online lectures because it fails to handle
493

technical jargon.
494

• The authors should discuss the computational efficiency of the proposed algorithms
495

and how they scale with dataset size.
496

• If applicable, the authors should discuss possible limitations of their approach to address
497

problems of privacy and fairness.
498

• While the authors might fear that complete honesty about limitations might be used by
499

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
500

limitations that aren’t acknowledged in the paper. The authors should use their best
501

judgment and recognize that individual actions in favor of transparency play an important
502

role in developing norms that preserve the integrity of the community. Reviewers will
503

be specifically instructed to not penalize honesty concerning limitations.
504

3. Theory Assumptions and Proofs
505

Question: For each theoretical result, does the paper provide the full set of assumptions and
506

a complete (and correct) proof?
507

Answer: [NA]
508

17


---Page Break---
Justification: The paper does not include theoretical assumptions.
509

Guidelines:
510

• The answer NA means that the paper does not include theoretical results.
511

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
512

referenced.
513

• All assumptions should be clearly stated or referenced in the statement of any theorems.
514

• The proofs can either appear in the main paper or the supplemental material, but if
515

they appear in the supplemental material, the authors are encouraged to provide a short
516

proof sketch to provide intuition.
517

• Inversely, any informal proof provided in the core of the paper should be complemented
518

by formal proofs provided in appendix or supplemental material.
519

• Theorems and Lemmas that the proof relies upon should be properly referenced.
520

4. Experimental Result Reproducibility
521

Question: Does the paper fully disclose all the information needed to reproduce the main
522

experimental results of the paper to the extent that it affects the main claims and/or conclusions
523

of the paper (regardless of whether the code and data are provided or not)?
524

Answer: [Yes]
525

Justification: We have provided all the experimental details to reproduce the results. Code is
526

also available in the supplemental materials.
527

Guidelines:
528

• The answer NA means that the paper does not include experiments.
529

• If the paper includes experiments, a No answer to this question will not be perceived well
530

by the reviewers: Making the paper reproducible is important, regardless of whether
531

the code and data are provided or not.
532

• If the contribution is a dataset and/or model, the authors should describe the steps taken
533

to make their results reproducible or verifiable.
534

• Depending on the contribution, reproducibility can be accomplished in various ways.
535

For example, if the contribution is a novel architecture, describing the architecture fully
536

might suffice, or if the contribution is a specific model and empirical evaluation, it may
537

be necessary to either make it possible for others to replicate the model with the same
538

dataset, or provide access to the model. In general. releasing code and data is often
539

one good way to accomplish this, but reproducibility can also be provided via detailed
540

instructions for how to replicate the results, access to a hosted model (e.g., in the case
541

of a large language model), releasing of a model checkpoint, or other means that are
542

appropriate to the research performed.
543

• While NeurIPS does not require releasing code, the conference does require all
544

submissions to provide some reasonable avenue for reproducibility, which may depend
545

on the nature of the contribution. For example
546

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
547

to reproduce that algorithm.
548

(b) If the contribution is primarily a new model architecture, the paper should describe
549

the architecture clearly and fully.
550

(c) If the contribution is a new model (e.g., a large language model), then there should
551

either be a way to access this model for reproducing the results or a way to reproduce
552

the model (e.g., with an open-source dataset or instructions for how to construct the
553

dataset).
554

(d) We recognize that reproducibility may be tricky in some cases, in which case authors
555

are welcome to describe the particular way they provide for reproducibility. In the
556

case of closed-source models, it may be that access to the model is limited in some
557

way (e.g., to registered users), but it should be possible for other researchers to have
558

some path to reproducing or verifying the results.
559

5. Open access to data and code
560

Question: Does the paper provide open access to the data and code, with sufficient instructions
561

to faithfully reproduce the main experimental results, as described in supplemental material?
562

18


---Page Break---
Answer: [Yes]
563

Justification: We have included the code in the supplemental materials and shared the data
564

link anonymously in the main paper.
565

Guidelines:
566

• The answer NA means that paper does not include experiments requiring code.
567

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
568

public/guides/CodeSubmissionPolicy) for more details.
569

• While we encourage the release of code and data, we understand that this might not be
570

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
571

including code, unless this is central to the contribution (e.g., for a new open-source
572

benchmark).
573

• The instructions should contain the exact command and environment needed to run
574

to reproduce the results.
See the NeurIPS code and data submission guidelines
575

(https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
576

• The authors should provide instructions on data access and preparation, including how
577

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
578

• The authors should provide scripts to reproduce all experimental results for the new
579

proposed method and baselines. If only a subset of experiments are reproducible, they
580

should state which ones are omitted from the script and why.
581

• At submission time, to preserve anonymity, the authors should release anonymized
582

versions (if applicable).
583

• Providing as much information as possible in supplemental material (appended to the
584

paper) is recommended, but including URLs to data and code is permitted.
585

6. Experimental Setting/Details
586

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
587

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
588

results?
589

Answer: [Yes]
590

Justification: We have specified all the training and test details to understand the results.
591

Guidelines:
592

• The answer NA means that the paper does not include experiments.
593

• The experimental setting should be presented in the core of the paper to a level of detail
594

that is necessary to appreciate the results and make sense of them.
595

• The full details can be provided either with the code, in appendix, or as supplemental
596

material.
597

7. Experiment Statistical Significance
598

Question: Does the paper report error bars suitably and correctly defined or other appropriate
599

information about the statistical significance of the experiments?
600

Answer: [Yes]
601

Justification: We have performed our experiments three times for each to provide the mean
602

and variance accuracy suitably and correctly in our tables.
603

Guidelines:
604

• The answer NA means that the paper does not include experiments.
605

• The authors should answer "Yes" if the results are accompanied by error bars, confidence
606

intervals, or statistical significance tests, at least for the experiments that support the
607

main claims of the paper.
608

• The factors of variability that the error bars are capturing should be clearly stated (for
609

example, train/test split, initialization, random drawing of some parameter, or overall
610

run with given experimental conditions).
611

• The method for calculating the error bars should be explained (closed form formula,
612

call to a library function, bootstrap, etc.)
613

• The assumptions made should be given (e.g., Normally distributed errors).
614

19


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error of
615

the mean.
616

• It is OK to report 1-sigma error bars, but one should state it. The authors should
617

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
618

of Normality of errors is not verified.
619

• For asymmetric distributions, the authors should be careful not to show in tables or
620

figures symmetric error bars that would yield results that are out of range (e.g. negative
621

error rates).
622

• If error bars are reported in tables or plots, The authors should explain in the text how
623

they were calculated and reference the corresponding figures or tables in the text.
624

8. Experiments Compute Resources
625

Question: For each experiment, does the paper provide sufficient information on the computer
626

resources (type of compute workers, memory, time of execution) needed to reproduce the
627

experiments?
628

Answer: [Yes]
629

Justification: We have provided the details of computer resources in the experimental section.
630

Guidelines:
631

• The answer NA means that the paper does not include experiments.
632

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
633

or cloud provider, including relevant memory and storage.
634

• The paper should provide the amount of compute required for each of the individual
635

experimental runs as well as estimate the total compute.
636

• The paper should disclose whether the full research project required more compute
637

than the experiments reported in the paper (e.g., preliminary or failed experiments that
638

didn’t make it into the paper).
639

9. Code Of Ethics
640

Question: Does the research conducted in the paper conform, in every respect, with the
641

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
642

Answer: [Yes]
643

Justification: This research conducted in the paper conforms in every respect with the
644

NeurIPS Code of Ethics.
645

Guidelines:
646

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
647

• If the authors answer No, they should explain the special circumstances that require a
648

deviation from the Code of Ethics.
649

• The authors should make sure to preserve anonymity (e.g., if there is a special
650

consideration due to laws or regulations in their jurisdiction).
651

10. Broader Impacts
652

Question: Does the paper discuss both potential positive societal impacts and negative
653

societal impacts of the work performed?
654

Answer: [Yes]
655

Justification: We have discussed both potential positive societal impacts and negative societal
656

impacts in Sec. A.
657

Guidelines:
658

• The answer NA means that there is no societal impact of the work performed.
659

• If the authors answer NA or No, they should explain why their work has no societal
660

impact or why the paper does not address societal impact.
661

• Examples of negative societal impacts include potential malicious or unintended uses
662

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
663

(e.g., deployment of technologies that could make decisions that unfairly impact specific
664

groups), privacy considerations, and security considerations.
665

20


---Page Break---
• The conference expects that many papers will be foundational research and not tied
666

to particular applications, let alone deployments. However, if there is a direct path to
667

any negative applications, the authors should point it out. For example, it is legitimate
668

to point out that an improvement in the quality of generative models could be used to
669

generate deepfakes for disinformation. On the other hand, it is not needed to point out
670

that a generic algorithm for optimizing neural networks could enable people to train
671

models that generate Deepfakes faster.
672

• The authors should consider possible harms that could arise when the technology is
673

being used as intended and functioning correctly, harms that could arise when the
674

technology is being used as intended but gives incorrect results, and harms following
675

from (intentional or unintentional) misuse of the technology.
676

• If there are negative societal impacts, the authors could also discuss possible mitigation
677

strategies (e.g., gated release of models, providing defenses in addition to attacks,
678

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
679

feedback over time, improving the efficiency and accessibility of ML).
680

11. Safeguards
681

Question: Does the paper describe safeguards that have been put in place for responsible
682

release of data or models that have a high risk for misuse (e.g., pretrained language models,
683

image generators, or scraped datasets)?
684

Answer: [NA]
685

Justification: We believe this paper poses no such risks.
686

Guidelines:
687

• The answer NA means that the paper poses no such risks.
688

• Released models that have a high risk for misuse or dual-use should be released with
689

necessary safeguards to allow for controlled use of the model, for example by requiring
690

that users adhere to usage guidelines or restrictions to access the model or implementing
691

safety filters.
692

• Datasets that have been scraped from the Internet could pose safety risks. The authors
693

should describe how they avoided releasing unsafe images.
694

• We recognize that providing effective safeguards is challenging, and many papers do
695

not require this, but we encourage authors to take this into account and make a best
696

faith effort.
697

12. Licenses for existing assets
698

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
699

the paper, properly credited and are the license and terms of use explicitly mentioned and
700

properly respected?
701

Answer: [Yes]
702

Justification: We have cited all papers and credited all code we utilized in this work.
703

Guidelines:
704

• The answer NA means that the paper does not use existing assets.
705

• The authors should cite the original paper that produced the code package or dataset.
706

• The authors should state which version of the asset is used and, if possible, include a
707

URL.
708

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
709

• For scraped data from a particular source (e.g., website), the copyright and terms of
710

service of that source should be provided.
711

• If assets are released, the license, copyright information, and terms of use in the
712

package should be provided. For popular datasets, paperswithcode.com/datasets
713

has curated licenses for some datasets. Their licensing guide can help determine the
714

license of a dataset.
715

• For existing datasets that are re-packaged, both the original license and the license of
716

the derived asset (if it has changed) should be provided.
717

21


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to the
718

asset’s creators.
719

13. New Assets
720

Question: Are new assets introduced in the paper well documented and is the documentation
721

provided alongside the assets?
722

Answer: [Yes]
723

Justification: Our code has been included in the supplemental materials and is well
724

documented, we have also shared the synthetic data in the main paper.
725

Guidelines:
726

• The answer NA means that the paper does not release new assets.
727

• Researchers should communicate the details of the dataset/code/model as part of their
728

submissions via structured templates. This includes details about training, license,
729

limitations, etc.
730

• The paper should discuss whether and how consent was obtained from people whose
731

asset is used.
732

• At submission time, remember to anonymize your assets (if applicable). You can either
733

create an anonymized URL or include an anonymized zip file.
734

14. Crowdsourcing and Research with Human Subjects
735

Question: For crowdsourcing experiments and research with human subjects, does the paper
736

include the full text of instructions given to participants and screenshots, if applicable, as
737

well as details about compensation (if any)?
738

Answer: [NA]
739

Justification: This paper does not involve crowdsourcing nor research with human subjects.
740

Guidelines:
741

• The answer NA means that the paper does not involve crowdsourcing nor research with
742

human subjects.
743

• Including this information in the supplemental material is fine, but if the main
744

contribution of the paper involves human subjects, then as much detail as possible
745

should be included in the main paper.
746

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
747

or other labor should be paid at least the minimum wage in the country of the data
748

collector.
749

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
750

Subjects
751

Question: Does the paper describe potential risks incurred by study participants, whether
752

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
753

approvals (or an equivalent approval/review based on the requirements of your country or
754

institution) were obtained?
755

Answer: [NA]
756

Justification: This paper does not involve crowdsourcing nor research with human subjects.
757

Guidelines:
758

• The answer NA means that the paper does not involve crowdsourcing nor research with
759

human subjects.
760

• Depending on the country in which research is conducted, IRB approval (or equivalent)
761

may be required for any human subjects research. If you obtained IRB approval, you
762

should clearly state this in the paper.
763

• We recognize that the procedures for this may vary significantly between institutions
764

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
765

guidelines for their institution.
766

• For initial submissions, do not include any information that would break anonymity (if
767

applicable), such as the institution conducting the review.
768

22


---Page Break---
