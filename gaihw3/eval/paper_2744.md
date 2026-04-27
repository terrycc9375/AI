Prioritize Alignment in Dataset Distillation

Anonymous Author(s)
Affiliation
Address
email

.
1

Abstract

Dataset Distillation aims to compress a large dataset into a significantly more
2

compact, synthetic one without compromising the performance of the trained mod-
3

els. To achieve this, existing methods use the agent model to extract information
4

from the target dataset and embed it into the distilled dataset. Consequently, the
5

quality of extracted and embedded information determines the quality of the dis-
6

tilled dataset. In this work, we find that existing methods introduce misaligned
7

information in both information extraction and embedding stages. To alleviate
8

this, we propose Prioritize Alignment in Dataset Distillation (PAD), which aligns
9

information from the following two perspectives. 1) We prune the target dataset
10

according to the compressing ratio to filter the information that can be extracted
11

by the agent model. 2) We use only deep layers of the agent model to perform the
12

distillation to avoid excessively introducing low-level information. This simple
13

strategy effectively filters out misaligned information and brings non-trivial im-
14

provement for mainstream matching-based distillation algorithms. Furthermore,
15

built on trajectory matching, PAD achieves remarkable improvements on vari-
16

ous benchmarks, achieving state-of-the-art performance. The code and distilled
17

datasets will be made public.
18

1
Introduction
19

Dataset Distillation (DD) [43] aims to compress a large dataset into a small synthetic dataset that
20

preserves important features for models to achieve comparable performances. Ever since being
21

introduced, DD has gained a lot of attention because of its wide applications in practical fields such
22

as privacy preservation [5, 44], continual learning [28, 35], and neural architecture search [12, 32].
23

Recently, matching-based methods [46, 42, 6] have achieved promising performance in distilling
24

high-quality synthetic datasets. Generally, the process of these methods can be summarized into two
25

steps: (1) Information Extraction: an agent model is used to extract important information from the
26

target dataset by recording various metrics such as gradients [49], distributions [48], and training
27

trajectories [1], (2) Information Embedding: the synthetic samples are optimized to incorporate the
28

extracted information, which is achieved by minimizing the differences between the same metric
29

calculated on the synthetic data and the one recorded in the previous step.
30

In this work, we first reveal both steps will introduce misaligned information, which is redundant
31

and potentially detrimental to the quality of the synthetic data. Then, by analyzing the cause of this
32

misalignment, we propose alleviating this problem through the following two perspectives.
33

Typically, in the Information Extraction step, most distillation methods allow the agent model to
34

see all samples in the target dataset. This means information extracted by the agent model comes
35

from samples with various difficulties (see Figure 1(a)). However, according to previous study
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
+

100% easy 
100% hard 

for all IPCs
IPC

100% easy 

+

100% easy 
50% hard 

+

50% easy 
100% hard 

small
medium
large

…
…

previous methods
ours

(a) Data used for distillation

IPC
small
medium
large

discard
discard
discard

agent model

shallow layer

deep layer
distill

distill

more 
semantical 
information

more 
low-level 
information

(b) Parameters used for distillation

Figure 1: (a) Compared with using all samples without differentiation in IPCs (left), PAD meticulously
selects a subset of samples for different IPCs to align the expected difficulty of information required
(right). (b) Different layers distill different patterns (left). PAD masks out (grey box) shallow-layer
parameters during metric matching in accordance with IPCs (right).

[10], information related to easy samples is only needed when the compression ratio is high. This
37

misalignment leads to the sub-optimal of the distillation performance.
38

To alleviate the above issue, we first use data selection methods to measure the difficulty of each
39

sample in the target dataset. Then, during the distillation, a data scheduler is employed to ensure only
40

data whose difficulty is aligned with the compression ratio is available for the agent model.
41

In the Information Embedding step, most distillation methods except DM [48] choose to use all
42

parameters of the agent model to perform the distillation. Intuitively, this will ensure the information
43

extracted by the agent model is fully utilized. However, we find shallow layer parameters of the
44

model can only provide low-quality, basic signals, which are redundant for dataset distillation in
45

most cases. Conversely, performing the distillation with only parameters from deep layers will yield
46

high-quality synthetic samples. We attribute this contradiction to the fact that deeper layers in DNNs
47

tend to learn higher-level representations of input data [27, 37].
48

Based on our findings, to avoid embedding misaligned information in the Information Embedding step,
49

we propose to use only parameters from deeper layers of the agent model to perform distillation, as
50

illustrated in Figure 1(b). This simple change brings significant performance improvement, showing
51

its effectiveness in aligning information.
52

Through experiments, we validate that our two-step alignment strategy is effective for distillation
53

methods based on matching gradients [49], distributions [48], and trajectories [1]. Moreover, by
54

applying our alignment strategy on trajectory matching [1, 10], we propose our novel method named
55

Prioritize Alignment in Dataset Distillation (PAD). After conducting comprehensive evaluation
56

experiments, we show PAD achieves state-of-the-art (SOTA) performance.
57

2
Misaligned Information in Dataset Distillation
58

Generally, we can summarize the distillation process of matching-based methods into the following
59

two steps: (1) Information Extraction: use an agent model to extract essential information from the
60

target dataset, realized by recording metrics such as gradients [49], distributions [48], and training
61

trajectories [1], (2) Information Embedding: the synthetic samples are optimized to incorporate the
62

extracted information, realized by minimizing the differences between the same metric calculated on
63

the synthetic data and the one recorded in the first step.
64

2


---Page Break---
65

66

67

10
15
20
25
Ratio (%)

16

17

18

19

Acc (%)

Remove easy
samples

Remove hard
samples
ipc500
ipc1
baseline

(a) Matching gradients

69

70

71

10
15
20
25
Ratio (%)

25

26

27

28

Acc (%)

(b) Matching distributions

82

83

10
15
20
25
Ratio (%)

45

46

47

Acc (%)

(c) Matching trajectories

Figure 2: Distillation performance on CIFAR-10 where data points are removed with different ratios.
Removing unnecessary data points helps to improve the performance of methods based on matching
gradients, distributions, and trajectories, both in low and high IPC cases.

65

66

0
25
50
75
Ratio (%)

27

28

29

Acc (%)

ipc500
ipc10
baseline

(a) Matching gradients

67

68

0
25
50
75
Ratio (%)

27

28

29

Acc (%)

(b) Matching distributions

83

84

0
25
50
75
Ratio (%)

65

66

67

Acc (%)

(c) Matching trajectories

Figure 3: Distillation performances on CIFAR-10 where n% (ratio) shallow layer parameters are not
utilized during distillation. Discarding shallow-layer parameters is beneficial for methods based on
matching gradients, distributions, and trajectories, both in low and high IPC cases.

In this section, through analyses and experimental verification, we show the above two steps both
65

will introduce misaligned information to the synthetic data.
66

2.1
Misaligned Information Extracted by Agent Models
67

In the information extraction step, an agent model is employed to extract information from the target
68

dataset. Generally, most existing methods [1, 6, 49, 46] allow the agent model to see the full dataset.
69

This implies that the information extracted by the agent model originates from samples with diverse
70

levels of difficulty. However, the expected difficulty of distilled information varies with changes in
71

IPC: smaller IPCs prefer easier information while larger IPCs should distill harder one [10].
72

To verify if this misalignment will influence the quality of synthetic data, we perform the distillation
73

where hard/easy samples of target dataset are removed with various ratios. As the results reported in
74

Figure 2, pruning unaligned data points is beneficial for all matching-based methods. This proves the
75

misalignment indeed will influence the distillation performance and can be alleviated by filtering out
76

misaligned data from the target dataset.
77

2.2
Misaligned Information Embedded by Metric Matching
78

Most existing methods use all parameters of the agent model to compute the metric used for matching.
79

Intuitively, this helps to improve the distillation performance, since in this way all information
80

extracted by the agent model will be embedded into the synthetic dataset. However, since shallow
81

layers in DNNs tend to learn basic distributions of data [27, 37], using parameters from these layers
82

can only provide low-level signals that turned out to be redundant in most cases.
83

3


---Page Break---
As can be observed in Figure 3, it is evident that across all matching-based methods, the removal
84

of shallow layer parameters consistently enhances performance, regardless of the IPC setting. This
85

proves employing over-shallow layer parameters to perform the distillation will introduce misaligned
86

information to the synthetic data, compromising the quality of distilled data.
87

3
Method
88

To alleviate the information misalignment issue, based on trajectory matching (TM) [1, 10], we
89

propose Prioritizing Alignment in Dataset Distillation (PAD). PAD can also be applied to methods
90

based on matching gradients [49] and distributions [48], which are introduced in Appendix A.1.
91

3.1
Preliminary of Trajectory Matching
92

Following the two-step procedure, to extract information, TM-based methods [1, 10] first train agent
93

models on the real dataset DR and record the changes of the parameters. Specifically, let {θ∗
t }N
0 be
94

an expert trajectory, which is a parameter sequence recorded during the training of agent model. At
95

each iteration of trajectory matching, θ∗
t and θ∗
t+M are randomly selected from expert trajectories as
96

the start and target parameters.
97

To embed the information into the synthetic data, TM methods minimize the distance between the
98

expert trajectory and the student trajectory. Let ˆθt denote the parameters of the student agent model
99

trained on synthetic dataset DS at timestep t. The student trajectory progresses by doing gradient
100

descent on the cross-entropy loss l for N steps:
101

ˆθt+i+1 = ˆθt+i −α∇l(ˆθt+i, DS),
(1)
Finally, the synthetic data is optimized by minimizing the distance metric, which is formulated as:
102

L = ||ˆθt+N −θ∗
t+M||
||θ∗
t+M −θ∗
t || ,
(2)

3.2
Filtering Information Extraction
103

In section 2.1, we show using data selection to filter out unmatched samples could alleviate the
104

misalignment caused in Information Extraction step. According to previous work [10], TM-based
105

methods prefer easy information and choose to match only early trajectories when IPC is small.
106

Conversely, hard information is preferred by high IPCs and they match only late trajectories. Hence,
107

we should use easy samples to train early trajectories, while late trajectories should be trained with
108

hard samples. To realize this efficiently, we first use the data selection method to measure the difficulty
109

of samples contained in the target dataset. Then, during training expert trajectories, a scheduler is
110

implemented to gradually incorporate hard samples into the training set while excluding easier ones
111

from it.
112

Difficulty Scoring Function
Identifying the difficulty of data for DNNs to learn has been well
113

studied in data selection area [29, 17, 16, 40]. For simplicity consideration, we use Error L2-Norm
114

(EL2N) score [33] as the metric to evaluate the difficulty of training examples (other metrics can also
115

be chosen, see Section 4.3.2). Specifically, let x and y denote a data point and its label, respectively.
116

Then, the EL2N score can be calculated by:
117

χt(x, y) = E||p(wt, x) −y||2,
(3)
where p(wt, x) = σ(f(wt, x)) is the output of a model f at training step t transformed into a
118

probability distribution. In consistent with [40], samples with higher EL2N scores are considered as
119

harder samples in this paper.
120

Scheduler
The scheduler can be divided into the following stages. Firstly, the hardest samples are
121

removed from the training set, ensuring that it exclusively comprises data meeting a predetermined
122

initial ratio (IR). Then, during training expert trajectories, samples are gradually added to the training
123

set in order of increasing difficulty. After incorporating all the data into the training set, the scheduler
124

will begin to remove easy samples from the target dataset. Unlike the gradual progression involved in
125

adding data, the action of reducing data is completed in a single operation, since now the model has
126

been trained on simple samples for a sufficient time.
127

4


---Page Break---
Dataset
CIFAR-10
CIFAR-100
Tiny ImageNet
IPC
1
10
50
500
1000
1
10
50
100
1
10
50
Ratio
0.02
0.2
1
10
20
0.2
2
10
20
0.2
2
10

Random
15.4±0.3
31.0±0.5
50.6±0.3
73.2±0.3
78.4±0.2
4.2±0.3
14.6±0.5
33.4±0.4
42.8±0.3
1.4±0.1
5.0±0.2
15.0±0.4
KIP [31]
49.9±0.2
62.7±0.3
68.6±0.2
-
-
15.7±0.2
28.3±0.1
-
-
-
-
-
FRePo [50]
46.8±0.7
65.5±0.4
71.7±0.2
-
-
28.7±0.1
42.5±0.2
44.3±0.2
-
15.4±0.3
25.4±0.2
-
RCIG [26]
53.9±1.0
69.1±0.4
73.5±0.3
-
-
39.3±0.4
44.1±0.4
46.7±0.3
-
25.6±0.3
29.4±0.2
-

DC [49]
28.3±0.5
44.9±0.5
53.9±0.5
72.1±0.4
76.6±0.3
12.8±0.3
25.2±0.3
-
-
-
-
-
DM [48]
26.0±0.8
48.9±0.6
63.0±0.4
75.1±0.3
78.8±0.1
11.4±0.3
29.7±0.3
43.6±0.4
-
3.9±0.2
12.9±0.4
24.1±0.3
DSA [47]
28.8±0.7
52.1±0.5
60.6±0.5
73.6±0.3
78.7±0.3
13.9±0.3
32.3±0.3
42.8±0.4
-
-
-
-
TESLA [4]
48.5±0.8
66.4±0.8
72.6±0.7
-
-
24.8±0.4
41.7±0.3
47.9±0.3
49.2±0.4
-
-
-
CAFE [42]
30.3±1.1
46.3±0.6
55.5±0.6
-
-
12.9±0.3
27.8±0.3
37.9±0.3
-
-
-
-
MTT [1]
46.2±0.8
65.4±0.7
71.6±0.2
-
-
24.3±0.3
39.7±0.4
47.7±0.2
49.2±0.4
8.8±0.3
23.2±0.2
28.0±0.3
FTD [6]
46.0±0.4
65.3±0.4
73.2±0.2
-
-
24.4±0.4
42.5±0.2
48.5±0.3
49.7±0.4
10.5±0.2
23.4±0.3
28.2±0.4
DATM [10]
46.9±0.5
66.8±0.2
76.1±0.3
83.5±0.2
85.5±0.4
27.9±0.2
47.2±0.4
55.0±0.2
57.5±0.2
17.1±0.3
31.1±0.3
39.7±0.3
PAD
47.2±0.6
67.4±0.3
77.0±0.5
84.6±0.3
86.7±0.2
28.4±0.5
47.8±0.2
55.9±0.3
58.5±0.3
17.7±0.2
32.3±0.4
41.6±0.4

Full Dataset
84.8±0.1
56.2±0.3
37.6±0.4

Table 1: Comparison with previous dataset distillation methods (bottom: matching-based, top: others)
on CIFAR-10, CIFAR-100 and Tiny ImageNet. ConvNet is used for the distillation and evaluation.
Our method consistently outperforms prior matching-based methods.

3.3
Filtering Information Embedding
128

To filter out misaligned information introduced by matching shallow-layer parameters, we propose
129

to add a parameter selection module that masks out part of shallow layers for metric computation.
130

Specifically, parameters of an agent network can be represented as a flattened array of length L that
131

stores weights of agent models ordered from shallow to deep layers (parameters within the same
132

layer are sorted in default order). The parameter selection sets a threshold ratio α such that the first
133

k = L · α parameters are not used for distillation. Then the parameters used for matching can now be
134

formulated as:
135

ˆθt+N = {ˆθ0, ˆθ1, · · · , ˆθk−1
|
{z
}
discard

, ˆθk, ˆθk+1, · · · , ˆθL
|
{z
}
used for matching

}.
(4)

In practice, the ratio α should vary with the change of IPC. For smaller IPCs, it is necessary to
136

incorporate basic information thus α should be lower. Conversely, basic information is redundant in
137

larger IPC cases, so α should be higher accordingly.
138

4
Experiments
139

4.1
Settings
140

We compare PAD with several prominent dataset distillation methods, which can be divided into two
141

categories: matching-based approaches including DC [49], DM [48], DSA [47], CAFE [42], MTT [1],
142

FTD [6], DATM [10], TESLA [4], and kernel-based approaches including KIP [31], FRePo [50],
143

RCIG [26]. The assessment is conducted on widely recognized datasets: CIFAR-10, CIFAR-100[18],
144

and Tiny ImageNet [20]. We implemented our method based on DATM [10]. In both the distillation
145

and evaluation phases, we apply the standard set of differentiable augmentations commonly used in
146

previous studies [1, 6, 10]. By default, networks are constructed with instance normalization unless
147

explicitly labeled with "-BN," indicating batch normalization (e.g., ConvNet-BN). For CIFAR-10
148

and CIFAR-100, distillation is typically performed using a 3-layer ConvNet, while Tiny ImageNet
149

requires a 4-layer ConvNet. Cross-architecture experiments also utilize LeNet [21], AlexNet [19],
150

VGG11 [39], and ResNet18 [11]. More details can be found in the appendix.
151

4.2
Main Results
152

CIFAR and Tiny ImageNet
We conduct comprehensive experiments to compare the performance
153

of our method with previous works. As the results presented in Table 1, PAD outperforms previous
154

matching-based methods on three datasets except for the case when IPC=1. When compared with
155

kernel-based methods which use a larger network to perform the distillation, our technique exhibits
156

superior performance in most cases, particularly when the compression ratio exceeds 1%. As can be
157

observed, PAD performs relatively better when IPC is high, suggesting our filtering out misaligned
158

information strategy becomes increasingly effective as IPC increases.
159

5


---Page Break---
Dataset
Ratio
Method
ConvNet
ConvNet-BN
ResNet18
ResNet18-BN
VGG11
AlexNet
LeNet
MLP
Avg.

CIFAR-10
20%

Random
78.38
80.25
84.58
87.21
80.81
80.75
61.85
50.98
75.60
Glister
62.46
70.52
81.10
74.59
78.07
70.55
56.56
40.59
66.81
Forgetting
76.27
80.06
85.67
87.18
82.04
81.35
64.59
52.21
76.17
DATM
85.50
85.23
87.22
88.13
84.65
85.14
66.70
52.40
79.37
PAD
86.90
85.67
86.95
88.09
84.34
85.83
67.28
53.62
79.84
↑
+8.52
+5.42
+2.37
+0.88
+3.53
+5.08
+5.43
+2.64
+4.24

CIFAR-100
20%

Random
42.80
46.38
47.48
55.62
42.69
38.05
25.91
20.66
39.95
Glister
35.45
37.13
42.49
46.14
43.06
28.58
23.33
17.08
34.16
Forgetting
45.52
49.99
51.44
54.65
43.28
43.47
27.22
22.90
42.30
DATM
57.50
57.75
57.98
63.34
55.10
55.69
33.57
26.39
50.92
PAD
58.50
58.66
58.15
63.17
55.02
55.93
33.87
27.12
51.30
↑
+15.70
+12.28
+10.67
+7.55
+12.33
+17.88
+7.96
+6.46
+11.35

Tiny
10%

Random
15.00
24.21
17.73
28.07
22.51
14.03
9.25
5.85
17.08
Glister
17.32
19.77
18.84
23.12
19.10
11.68
8.84
3.86
15.32
Forgetting
20.04
23.83
19.38
28.88
23.77
12.13
12.06
5.54
18.20
DATM
39.68
40.32
36.12
43.14
38.35
35.10
12.41
9.02
31.76
PAD
41.02
40.88
36.08
42.96
38.64
35.02
13.17
9.68
32.18
↑
+26.02
+16.67
+18.35
+14.89
+16.13
+20.99
+3.92
+3.83
+15.10
Table 2: Cross-architecture evaluation of distilled data on unseen networks. Results worse than
random selection are indicated with red color. ↑denotes the performance improvement brought by
our method compared with random selection. Tiny denotes Tiny ImageNet.

Method
ConvNet
ResNet18
VGG
AlexNet

Random
33.46
31.95
32.18
26.65
FTD
48.90
46.65
43.24
42.20
DATM
55.03
51.71
45.38
45.74
PAD
55.91
52.35
44.97
45.92

(a) Datasets distilled by PAD general-
ize well across various architectures.

FIEX
FIEM
Accuracy(%)

66.7
✓
66.9
✓
67.2
✓
✓
67.4

(b) Each module brings non-
trivial improvements.

IR
AEE
20
40
60

50%
66.23
66.07
65.92
75%
67.36
67.34
66.58
80%
67.26
67.08
66.47

(c) Set IR as 75% always per-
form best.

Table 3: (a) Cross-Architecture evaluation on CIFAR-100 IPC50. (b) Ablation studies on the modules
of our method on CIFAR-10 IPC10. (c) Results of different sets of data selection hyper-parameters
on CIFAR-10 IPC10.

Cross Architecture Generalization
We evaluate the generalizability of our distilled data in both
160

low and high IPC cases. As results reported in Table 3(a), when IPC is small, our distilled data
161

outperforms the previous SOTA method DATM on ResNet and AlexNet while maintaining comparable
162

accuracy on VGG. This suggests that our distilled data on high compressing ratios generalizes well
163

across various unseen networks. Moreover, as reflected in Table 2, our distilled datasets on large IPCs
164

also have the best performance on most evaluated architectures, showing good generalizability in the
165

low compressing ratio case.
166

4.3
Ablation Study
167

To validate the effectiveness of each component of our method, we conducted ablation experiments
168

on modules (section 4.3.1) and their hyper-parameter settings (section 4.3.2 and section 4.3.2).
169

4.3.1
Modules
170

Our method incorporates two separate modules to filter information extraction (FIEX) and information
171

embedding (FIEM), respectively. To verify their isolated effectiveness, we conduct an ablation study
172

by applying two modules individually. As depicted in Table 3(b), both FIEX and FIEM bring
173

improvements, implying their efficacy. By applying these two modules, we are able to effectively
174

remove unaligned information, improving the distillation performance.
175

4.3.2
Hyper-parameters of Filtering Information Extraction
176

Initial Ratio and Data Addition Epoch
To filter the information learned by agent models, we
177

initialize the training set with only easy samples, and the size is determined by a certain ratio of
178

the total size. Then, we gradually add hard samples into the training set. In practice, we use two
179

hyper-parameters to control the addition process: the initial ratio (IR) of training data for training
180

set initialization and the end epoch of hard sample addition (AEE). These two parameters together
181

control the amount of data agent models can see at each epoch and the speed of adding hard samples.
182

6


---Page Break---
Method
IPC
1
10
500

Loss
45.74
66.45
83.47
Uncertainty [3]
46.22
66.99
84.22
EL2N [33]
47.23
67.38
84.63

(a) Using EL2N to measure the diffi-
culty of samples has the best perfor-
mance.

IPC
Ratio
100%
75%
50%
25%

1
47.2
46.56
45.98
41.32
10
67.2
67.34
66.86
65.15
500
83.71
83.82
84.23
84.64

(b) As IPC increases, removing
more shallow-layer parameters
becomes more effective.

Strategy
IPC
10
50

Baseline
67.2
76.5
Loss
67.3
77.0
Depth
67.7
77.3

(c) Using layer depth to select
parameters outperforms using
matching loss.

Table 4: (a) Ablation of different difficulty scoring functions on CIFAR-10. (b) Results of masking
out different ratios of shallow-layer parameters across various IPCs on CIFAR-10. (c) Ablation on
the strategy used for parameter selection on CIFAR-10

(a) with 100% parameters
(b) with 75% parameters
(c) with 50% parameters

Figure 4: Synthetic images of CIFAR-10 IPC50 obtained by PAD with different ratios of parameter
selection. Smoother image features indicate that by removing some shallow-layer parameters during
matching, PAD successfully filters out coarse-grained low-level information.

In Table 3(c), we show the distillation results where different hyper-parameters are utilized. In
183

general, a larger initial ratio and faster speed of addition bring better performances. Although the
184

distillation benefited more from learning simpler information when IPC is small [10], our findings
185

indicate that excessively removing difficult samples (e.g., more than a quarter) early in the training
186

phase can adversely affect the distilled data. This negative impact is likely due to the excessive
187

removal leading to distorted feature distributions within each category. On the other hand, reasonably
188

improving the speed of adding hard samples allows the agent model to achieve a more balanced
189

learning of information of varying difficulty across different stages.
190

Other Difficulty Scoring Functions
Identifying the difficulty of data points is the key to filtering
191

out misaligned information in the extraction step. Here, we compare the effect of using other
192

difficulty-scoring functions to evaluate the difficulty of data. (1) prediction loss of a pre-trained
193

ResNet. (2) uncertainty score [3]. (3) EL2N [33]. As can be observed in Table 4(a), EL2N performs
194

the best across various IPCs; thus, we use it to measure how hard each data point is as default in our
195

method. Note that this can also be replaced with a more advanced data selection algorithm.
196

4.3.3
Ratios of Parameter Selection
197

It is important to find a good balance between the percentage of shallow-layer parameters removed
198

from matching and the loss of information. In Table 4(b), we show results obtained on different
199

IPCs by discarding various ratios of shallow-layer parameters. The impact of removing varying
200

proportions of shallow parameters on the distilled data and its relationship with changes in IPC
201

is consistent with prior conclusions. For small IPCs, distilled data requires more low-level basic
202

information. Thus, removing too many shallow-layer parameters causes a negative effect on the
203

classification performance. By contrast, high-level semantic information is more important when
204

it comes to large IPCs. With increasing ratios of shallow-layer parameters being discarded, we can
205

ensure that low-level information is effectively filtered out from the distilled data.
206

5
Discussion
207

5.1
Distilled Images with Filtering Information Embedding
208

To see the concrete patterns brought by removing shallow-layer parameters to perform the trajectory
209

matching, we present distilled images obtained by discarding various ratios of shallow-layer parame-
210

ters in Figure 4. As can be observed in Figure 4(a), without removing any shallow-layer parameters
211

7


---Page Break---
1
2
3
4
5
6
Layer

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

4.0

iter 0
iter 1000
iter 5000

Loss

(a) CIFAR-10 IPC1

1
2
3
4
5
6
Layer

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75
iter 0
iter 1000
iter 5000

Loss

(b) CIFAR-10 IPC10

1
2
3
4
5
6
Layer

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Loss

iter 0
iter 1000
iter 5000

(c) CIFAR-10 IPC500

Figure 5: Losses of different layers of ConvNet after matching trajectories for 0, 1000, and 5000
iterations. We notice a similar phenomenon on both small (IPC1 and IPC10) and large IPCs (IPC500):
losses of shallow-layer parameters fluctuate along the matching process, while losses of deep-layer
parameters show a clear trend of decreasing.

(a) Match shallow layers only
(b) Original
(c) Match deep layers only

Figure 6: Synthetic images visualization with parameter selection. Matching parameters in shallow
layers produces an abundance of low-level texture features, whereas patterns generated by matching
deep-layer parameters embody richer high-level semantic information.

to filter misaligned information, synthetic images are interspersed with substantial noises. These
212

noises often take the form of coarse and generic information, such as the overall color distribution
213

and edges in the image, which provides minimal utility for precise classification.
214

By contrast, images distilled by our enhanced methodology (see Figure 4(b) and Figure 4(c)), which
215

includes meticulous masking out shallow-layer parameters during trajectory matching according to the
216

compressing ratio, contain more fine-grained and smoother features. These images also encapsulate
217

a broader range of semantic information, which is crucial for helping the model make accurate
218

classifications. Moreover, we observe a clear trend: as the amount of the removed shallow-layer
219

parameters increases, the distilled images exhibit clearer and smoother features.
220

5.2
Rationale for Parameter Selection
221

In this section, we analyze from the perspective of trajectory matching why shallow-layer parameters
222

should be masked out. In Figure 5, we present the changes in trajectory matching loss across different
223

layers as the distillation progresses. Compared to the deep-layer parameters of the agent model,
224

a substantial number of shallow-layer parameters exhibit low loss values that fluctuate during the
225

matching process (see Figure 5). By contrast, the loss values of the deep layers are much higher but
226

consistently decrease as distillation continues. This suggests that matching shallow layers primarily
227

conveys low-level information that is readily captured by the synthetic data and quickly saturated.
228

Consequently, the excessive addition of such low-level information produces noise, reducing the
229

quality of distilled datasets.
230

For a concrete visualization, we provide distilled images resulting from using only shallow-layer
231

parameters or only deep-layer parameters to match trajectories in Figure 6. The coarse image features
232

depicted in Figure 6(a) further substantiate our analysis.
233

8


---Page Break---
5.3
Parameter Selection Strategy
234

In the previous section, we observed a positive correlation between the depth of the model layers
235

and the magnitude of their trajectory-matching losses. Notably, the loss in the first layer of the
236

ConvNet was higher compared to other shallow layers. Consequently, we further compared different
237

parameter alignment strategies, specifically by sorting the parameters based on their matching losses
238

and excluding a certain proportion of parameters with lower losses. Higher loss values indicate
239

greater discrepancies in parameter weights; thus, continuing to match these parameters can inject
240

more information into the synthetic data. As shown in Table 4(c), sorting by loss results in an
241

improvement compared with no parameter alignment, but filtering based on parameter depth proves
242

to be more effective.
243

6
Related Work
244

Introduced by [43], dataset distillation aims to synthesize a compact set of data that allows models to
245

achieve similar test performances compared with the original dataset. Since then, a number of studies
246

have explored various approaches. These methods can be divided into three types: kernel-based,
247

matching-based, and using generative models [45].
248

Kernel-based methods are able to achieve closed-form solutions for the inner optimization [31] via
249

kernel ridge regression with NTK [22]. FRePo [50] distills a compact dataset through neural feature
250

regression and reduces the training cost.
251

Matching-based methods first use agent models to extract information from the target dataset
252

by recording a specific metric [7, 23, 38, 24]. Representative works that design different metrics
253

include DC [49] that matches gradients, DM [48] that matches distributions, and MTT [1] that
254

matches training trajectories. Then, the distilled dataset is optimized by minimizing the matched
255

distance between the metric computed on synthetic data and the record one from the previous step.
256

Following this workflow, many works have been proposed to improve the efficacy of the distilled
257

dataset. For example, CAFE [42] preserves the real feature distribution and the discriminative power
258

of the synthetic data and achieves prominent generalization ability across various architectures.
259

DREAM [25] employs K-Means to select representative samples for distillation and improves the
260

distillation efficiency. DATM [10] proposes to match early trajectories for small IPCs and late
261

trajectories for large IPCs, achieving SOTA performances on several benchmarks. Moreover, new
262

metrics such as spatial attention maps [36, 15] have also been introduced and achieved promising
263

performance in distilling large-scale datasets.
264

Generative models such as GANs [8, 13, 14, 41] and diffusion models [34, 30, 9] can also be used to
265

distill high quality datasets. DiM [41] uses deep generative models to store information of the target
266

dataset. GLaD [2] transfers synthetic data optimization from the pixel space to the latent space by
267

employing deep generative priors. It enhances the generalizability of previous distillation methods.
268

7
Conclusion
269

In this work, we find a limitation of existing Dataset Distillation methods in that they will introduce
270

misaligned information to the distilled datasets. To alleviate this, we propose PAD, which incorporates
271

two modules to filter out misaligned information. For information extraction, PAD prunes the target
272

dataset based on sample difficulty for different IPCs so that only information with aligned difficulty
273

is extracted by the agent model. For information embedding, PAD discards part of shallow-layer
274

parameters to avoid injecting low-level basic information into the synthetic data. PAD achieves
275

SOTA performance on various benchmarks. Moreover, we show PAD can also be applied to methods
276

based on matching gradients and distribution, bringing remarkable improvements across various IPC
277

settings.
278

Limitations
Our alignment strategy could also be applied to methods based on matching gradients
279

and distributions (see Appendix A.1). However, due to the limitation of computing resources,
280

for methods based on matching distributions and gradients, we have only validated our method’s
281

effectiveness on DM [48] and DC [49] (see Table 5 and Table 6).
282

9


---Page Break---
References
283

[1] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu.
284

Dataset distillation by matching training trajectories. 2022 IEEE/CVF Conference on Computer
285

Vision and Pattern Recognition (CVPR), pages 10708–10717, 2022.
286

[2] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu.
287

Generalizing dataset distillation via deep generative prior. 2023 IEEE/CVF Conference on
288

Computer Vision and Pattern Recognition (CVPR), pages 3739–3748, 2023.
289

[3] Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis,
290

Percy Liang, Jure Leskovec, and Matei Zaharia. Selection via proxy: Efficient data selection for
291

deep learning. arXiv preprint arXiv:1906.11829, 2019.
292

[4] Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet-
293

1k with constant memory. In International Conference on Machine Learning, 2022.
294

[5] Tian Dong, Bo Zhao, and Lingjuan Lyu. Privacy for free: How does dataset condensation help
295

privacy? ArXiv, abs/2206.00240, 2022.
296

[6] Jiawei Du, Yiding Jiang, Vincent Y. F. Tan, Joey Tianyi Zhou, and Haizhou Li. Minimizing the
297

accumulated trajectory error to improve dataset distillation. 2023 IEEE/CVF Conference on
298

Computer Vision and Pattern Recognition (CVPR), pages 3749–3758, 2022.
299

[7] Jiawei Du, Qin Shi, and Joey Tianyi Zhou. Sequential subset matching for dataset distillation.
300

ArXiv, abs/2311.01570, 2023.
301

[8] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
302

Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. Communica-
303

tions of the ACM, 63:139 – 144, 2014.
304

[9] Jianyang Gu, Saeed Vahidian, Vyacheslav Kungurtsev, Haonan Wang, Wei Jiang, Yang You,
305

and Yiran Chen. Efficient dataset distillation via minimax diffusion. ArXiv, abs/2311.15529,
306

2023.
307

[10] Ziyao Guo, Kai Wang, George Cazenavette, Hui Li, Kaipeng Zhang, and Yang You. Towards
308

lossless dataset distillation via difficulty-aligned trajectory matching. ArXiv, abs/2310.05773,
309

2023.
310

[11] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image
311

recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
312

pages 770–778, 2015.
313

[12] Haifeng Jin, Qingquan Song, and Xia Hu. Auto-keras: An efficient neural architecture search
314

system. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
315

Discovery & Data Mining, 2018.
316

[13] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative
317

adversarial networks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition
318

(CVPR), pages 4396–4405, 2018.
319

[14] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila.
320

Analyzing and improving the image quality of stylegan. 2020 IEEE/CVF Conference on
321

Computer Vision and Pattern Recognition (CVPR), pages 8107–8116, 2019.
322

[15] Samir Khaki, Ahmad Sajedi, Kai Wang, Lucy Z. Liu, Yuri A. Lawryshyn, and Konstantinos N.
323

Plataniotis. Atom: Attention mixer for efficient dataset distillation, 2024.
324

[16] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, and
325

Rishabh K. Iyer. Grad-match: Gradient matching based data subset selection for efficient
326

deep model training. In International Conference on Machine Learning, 2021.
327

10


---Page Break---
[17] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Rishabh Iyer Univer-
328

sity of Texas at Dallas, Indian Institute of Technology Bombay Institution One, and IN Two.
329

Glister: Generalization based data subset selection for efficient and robust learning. In AAAI
330

Conference on Artificial Intelligence, 2020.
331

[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.
332

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep
333

convolutional neural networks. Communications of the ACM, 60:84 – 90, 2012.
334

[20] Ya Le and Xuan S. Yang. Tiny imagenet visual recognition challenge. 2015.
335

[21] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning
336

applied to document recognition. Proc. IEEE, 86:2278–2324, 1998.
337

[22] Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak,
338

Jascha Narain Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth
339

evolve as linear models under gradient descent. Journal of Statistical Mechanics: Theory and
340

Experiment, 2020, 2019.
341

[23] Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sung-Hoon Yoon. Dataset
342

condensation with contrastive signals. In International Conference on Machine Learning, 2022.
343

[24] Haoyang Liu, Tiancheng Xing, Luwei Li, Vibhu Dalal, Jingrui He, and Haohan Wang. Dataset
344

distillation via the wasserstein metric. ArXiv, abs/2311.18531, 2023.
345

[25] Yanqing Liu, Jianyang Gu, Kai Wang, Zheng Hua Zhu, Wei Jiang, and Yang You. Dream: Effi-
346

cient dataset distillation by representative matching. 2023 IEEE/CVF International Conference
347

on Computer Vision (ICCV), pages 17268–17278, 2023.
348

[26] Noel Loo, Ramin M. Hasani, Mathias Lechner, and Daniela Rus. Dataset distillation with
349

convexified implicit gradients. ArXiv, abs/2302.06755, 2023.
350

[27] Aravindh Mahendran and Andrea Vedaldi. Visualizing deep convolutional neural networks
351

using natural pre-images. International Journal of Computer Vision, 120:233 – 255, 2016.
352

[28] Wojciech Masarczyk and Ivona Tautkute. Reducing catastrophic forgetting with learning on
353

synthetic data. In CVPR Workshop, 2020.
354

[29] Baharan Mirzasoleiman, Jeff A. Bilmes, and Jure Leskovec. Coresets for data-efficient training
355

of machine learning models. In International Conference on Machine Learning, 2019.
356

[30] Brian B. Moser, Federico Raue, Sebastián M. Palacio, Stanislav Frolov, and Andreas Dengel.
357

Latent dataset distillation with diffusion models. ArXiv, abs/2403.03881, 2024.
358

[31] Timothy Nguyen, Zhourung Chen, and Jaehoon Lee. Dataset meta-learning from kernel
359

ridge-regression. ArXiv, abs/2011.00050, 2020.
360

[32] Ramakanth Pasunuru and Mohit Bansal. Continual and multi-task architecture search. ArXiv,
361

abs/1906.05226, 2019.
362

[33] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet:
363

Finding important examples early in training. In Neural Information Processing Systems, 2021.
364

[34] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
365

resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on
366

Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021.
367

[35] Andrea Rosasco, Antonio Carta, Andrea Cossu, Vincenzo Lomonaco, and Davide Bacciu.
368

Distilled replay: Overcoming forgetting through synthetic samples. In International Workshop
369

on Continual Semi-Supervised Learning, 2021.
370

[36] Ahmad Sajedi, Samir Khaki, Ehsan Amjadian, Lucy Z. Liu, Yuri A. Lawryshyn, and Kon-
371

stantinos N. Plataniotis. Datadam: Efficient dataset distillation with attention matching. In
372

Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages
373

17097–17107, October 2023.
374

11


---Page Break---
[37] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi
375

Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based
376

localization. International Journal of Computer Vision, 128:336 – 359, 2016.
377

[38] Seung-Jae Shin, Heesun Bae, DongHyeok Shin, Weonyoung Joo, and Il-Chul Moon. Loss-
378

curvature matching for dataset selection and condensation. In International Conference on
379

Artificial Intelligence and Statistics, 2023.
380

[39] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
381

image recognition. CoRR, abs/1409.1556, 2014.
382

[40] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond
383

neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information
384

Processing Systems, 35:19523–19536, 2022.
385

[41] Kai Wang, Jianyang Gu, Daquan Zhou, Zheng Hua Zhu, Wei Jiang, and Yang You. Dim:
386

Distilling dataset into generative model. ArXiv, abs/2303.04707, 2023.
387

[42] Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Hua Zhu, Shuo Yang, Shuo Wang, Guan Huang,
388

Hakan Bilen, Xinchao Wang, and Yang You. Cafe learning to condense dataset by aligning
389

features. 2022.
390

[43] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros. Dataset distillation,
391

2020.
392

[44] Qiying Yu, Yang Liu, Yimu Wang, Ke Xu, and Jingjing Liu. Multimodal federated learning via
393

contrastive representation ensemble. ArXiv, abs/2302.08888, 2023.
394

[45] Ruonan Yu, Songhua Liu, and Xinchao Wang. Dataset distillation: A comprehensive review.
395

IEEE Transactions on Pattern Analysis and Machine Intelligence, 46:150–170, 2023.
396

[46] Bo Zhao and Hakan Bilen. Dataset condensation with differentiable siamese augmentation. In
397

International Conference on Machine Learning, 2021.
398

[47] Bo Zhao and Hakan Bilen. Dataset condensation with differentiable siamese augmentation. In
399

International Conference on Machine Learning, 2021.
400

[48] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. 2023 IEEE/CVF
401

Winter Conference on Applications of Computer Vision (WACV), pages 6503–6512, 2021.
402

[49] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching.
403

ArXiv, abs/2006.05929, 2020.
404

[50] Yongchao Zhou, Ehsan Nezhadarya, and Jimmy Ba. Dataset distillation using neural feature
405

regression. ArXiv, abs/2206.00719, 2022.
406

12


---Page Break---
IPC
Ratio
Baseline
10%
15%
20%
25%

1
17.03
16.34
18.27
18.91
16.32
500
65.21
65.34
66.47
66.31
65.27

(a) Removing various ratios of hard/easy sam-
ples improves DC on small/large IPCs.

IPC
Ratio
Baseline
10%
15%
20%
25%

1
26.66
27.24
27.97
27.48
25.41
500
70.74
70.89
70.37
69.80
70.32

(b) Removing various ratios of hard/easy sam-
ples improves DM on small/large IPCs.

Table 5: Results of filtering information extraction by removing hard/easy samples in DC(a) and
DM(b) on CIFAR-10.

IPC
Ratio
Baseline
25%
50%
75%

10
29.23
28.67
27.36
28.88
500
65.88
65.97
66.24
65.39

(a) Matching gradients from deep-layer parameters
leads to improvements.

IPC
Ratio
Baseline
25%
50%
75%

10
29.23
28.67
27.36
28.88
500
67.48
67.76
68.14
67.39

(b) Matching distributions from deep-layer param-
eters leads to improvements.

Table 6: Results of filtering information embedding by masking out shallow-layer parameters for
metric computation in DC(a) and DM(b) on CIFAR-10.

A
Appendix
407

A.1
Filtering Misaligned Information in DC and DM
408

Although PAD is implemented based on trajectory matching methods, we also test our proposed
409

data alignment and parameter alignment on gradient matching and distribution matching. The
410

performances of enhanced DC and DM with each of the two modules are reported in Table 5 and
411

Tabl 6, respectively. We provide details of how we integrate these two modules into gradient matching
412

and distribution matching in the following sections.
413

Gradient Matching We use the official implementation1 of DC [49]. In the Information Extraction
414

step, DC uses an agent model to calculate the gradients after being trained on the target dataset. We
415

employ filter misaligned information in this step as follows: When IPC is small, a certain ratio of
416

hard samples is removed from the target dataset so that the recorded gradients only contain simple
417

information. Conversely, when IPC becomes large, we remove easy samples instead.
418

In the Information Embedding step, DC optimizes the synthetic data by back-propagating on the
419

gradient matching loss. The loss is computed by summing the differences in gradients between
420

each pair of model parameters. Thus, we apply parameter selection by discarding a certain ratio of
421

parameters in the shallow layers.
422

Distribution Matching We use the official implementation of DM [48], which can be accessed
423

via the same link as DC. In the Information Extraction step, DM uses an agent model to generate
424

embeddings of input images from the target dataset. Similarly, filtering information extraction is
425

applied by removing hard samples for small IPCs and easy samples for large IPCs.
426

In the Information Embedding step, since DM only uses the output of the last layer to match
427

distributions, we modify the implementation of the network such that outputs of each layer in the
428

model are returned by the forward function. Then, we perform parameter selection following the
429

same practice as before.
430

A.2
Experiment Settings
431

We use DATM [10] as the backbone TM algorithm and our proposed PAD is built upon. Thus, our
432

configurations for distillation, evaluation, and network are consistent with DATM.
433

1https://github.com/VICO-UoE/DatasetCondensation.git

13


---Page Break---
Distillation. We conduct the distillation process for 10,000 iterations to ensure full convergence of
434

the optimization. By default, ZCA whitening is applied in all the experiments.
435

Evaluation. We train a randomly initialized network on the distilled dataset and evaluate its per-
436

formance on the entire validation set of the original dataset. Following DATM [10], the evaluation
437

networks are trained for 1000 epochs to ensure full optimization convergence. For fairness, the
438

experimental results of previous distillation methods in both low and high IPC settings are sourced
439

from [10].
440

Network. We employ a range of networks to assess the generalizability of our distilled datasets.
441

For scaling ResNet, LeNet, and AlexNet to Tiny-ImageNet, we modify the stride of their initial
442

convolutional layer from 1 to 2. In the case of VGG, we adjust the stride of its final max pooling
443

layer from 1 to 2. The MLP used in our evaluations features a single hidden layer with 128 units.
444

Hyper-parameters. Hyper-parameters of our experiments on CIFAR-10, CIFAR-100, and Tiny-
445

ImageNet are reported in Table 7. Hyper-parameters can be divided into three parts including data
446

alignment (DA), parameter alignment (PA) and trajectory matching (TM). Soft labels are applied in
447

all experiments , we set its momentum to 0.9.
448

Compute resources. Our experiments are run on 4 NVIDIA A100 GPUs, each with 80 GB of
449

memory. The amount of GPU memory needed is mainly determined by the batch size of synthetic
450

data and the number of steps that the agment model is trained on synthetic data. To reduce the GPU
451

usage when IPC is large, one can apply TESLA [4] or simply reducing the synthetic steps N or the
452

synthetic batch size. However, the decrement of hyper-parameters shown in Table 7 could result in
453

performance degradation.
454

Dataset
IPC
DA
PA
TM

IR
AEE
α
N
M
T −
T
T +
Interval
Synthetic
Batch Size
Learning Rate
(Label)
Learning Rate
(Pixels)

CIFAR-10

1

0.75
20

0%
80
2
0
4
4
-
10
5
100
10
25%
80
2
0
10
20
100
100
2
100
50
25%
80
2
0
20
40
100
500
2
1000
500
50%
80
2
40
60
60
-
1000
10
50
1000
75%
80
2
40
60
60
-
1000
10
50

CIFAR-100

1

0.75
40

0%
40
3
0
10
20
100
100
10
1000
10
25%
80
2
0
20
40
100
1000
10
1000
50
50%
80
2
40
60
80
100
1000
10
1000
100
50%
80
2
40
80
80
-
1000
10
50

TI
1

0.75
40

0%
60
2
0
15
30
400
200
10
10000
10
25%
60
2
0
20
40
100
250
10
100
50
50%
80
2
20
40
60
100
250
10
100

Table 7: Hyper-parameters for different benchmarks.

14


---Page Break---
Figure 7: Distilled images of CIFAR-10 IPC10

15


---Page Break---
Figure 8: Distilled images of CIFAR-10 IPC10

16


---Page Break---
Figure 9: Distilled images of CIFAR-10 IPC10

17


---Page Break---
NeurIPS Paper Checklist
455

1. Claims
456

Question: Do the main claims made in the abstract and introduction accurately reflect the
457

paper’s contributions and scope?
458

Answer: [Yes]
459

Justification: Our main claim does accurately reflect the paper’s contributions and scope.
460

Guidelines:
461

• The answer NA means that the abstract and introduction do not include the claims
462

made in the paper.
463

• The abstract and/or introduction should clearly state the claims made, including the
464

contributions made in the paper and important assumptions and limitations. A No or
465

NA answer to this question will not be perceived well by the reviewers.
466

• The claims made should match theoretical and experimental results, and reflect how
467

much the results can be expected to generalize to other settings.
468

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
469

are not attained by the paper.
470

2. Limitations
471

Question: Does the paper discuss the limitations of the work performed by the authors?
472

Answer: [Yes]
473

Justification: We discuss limitations at the end of the paper.
474

Guidelines:
475

• The answer NA means that the paper has no limitation while the answer No means that
476

the paper has limitations, but those are not discussed in the paper.
477

• The authors are encouraged to create a separate "Limitations" section in their paper.
478

• The paper should point out any strong assumptions and how robust the results are to
479

violations of these assumptions (e.g., independence assumptions, noiseless settings,
480

model well-specification, asymptotic approximations only holding locally). The authors
481

should reflect on how these assumptions might be violated in practice and what the
482

implications would be.
483

• The authors should reflect on the scope of the claims made, e.g., if the approach was
484

only tested on a few datasets or with a few runs. In general, empirical results often
485

depend on implicit assumptions, which should be articulated.
486

• The authors should reflect on the factors that influence the performance of the approach.
487

For example, a facial recognition algorithm may perform poorly when image resolution
488

is low or images are taken in low lighting. Or a speech-to-text system might not be
489

used reliably to provide closed captions for online lectures because it fails to handle
490

technical jargon.
491

• The authors should discuss the computational efficiency of the proposed algorithms
492

and how they scale with dataset size.
493

• If applicable, the authors should discuss possible limitations of their approach to
494

address problems of privacy and fairness.
495

• While the authors might fear that complete honesty about limitations might be used by
496

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
497

limitations that aren’t acknowledged in the paper. The authors should use their best
498

judgment and recognize that individual actions in favor of transparency play an impor-
499

tant role in developing norms that preserve the integrity of the community. Reviewers
500

will be specifically instructed to not penalize honesty concerning limitations.
501

3. Theory Assumptions and Proofs
502

Question: For each theoretical result, does the paper provide the full set of assumptions and
503

a complete (and correct) proof?
504

Answer: [NA]
505

18


---Page Break---
Justification: We didn’t present any theoretical results in this paper.
506

Guidelines:
507

• The answer NA means that the paper does not include theoretical results.
508

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
509

referenced.
510

• All assumptions should be clearly stated or referenced in the statement of any theorems.
511

• The proofs can either appear in the main paper or the supplemental material, but if
512

they appear in the supplemental material, the authors are encouraged to provide a short
513

proof sketch to provide intuition.
514

• Inversely, any informal proof provided in the core of the paper should be complemented
515

by formal proofs provided in appendix or supplemental material.
516

• Theorems and Lemmas that the proof relies upon should be properly referenced.
517

4. Experimental Result Reproducibility
518

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
519

perimental results of the paper to the extent that it affects the main claims and/or conclusions
520

of the paper (regardless of whether the code and data are provided or not)?
521

Answer: [Yes]
522

Justification: All hyper-parameters and computing resources needed for experiments are
523

listed in the Appendix.
524

Guidelines:
525

• The answer NA means that the paper does not include experiments.
526

• If the paper includes experiments, a No answer to this question will not be perceived
527

well by the reviewers: Making the paper reproducible is important, regardless of
528

whether the code and data are provided or not.
529

• If the contribution is a dataset and/or model, the authors should describe the steps taken
530

to make their results reproducible or verifiable.
531

• Depending on the contribution, reproducibility can be accomplished in various ways.
532

For example, if the contribution is a novel architecture, describing the architecture fully
533

might suffice, or if the contribution is a specific model and empirical evaluation, it may
534

be necessary to either make it possible for others to replicate the model with the same
535

dataset, or provide access to the model. In general. releasing code and data is often
536

one good way to accomplish this, but reproducibility can also be provided via detailed
537

instructions for how to replicate the results, access to a hosted model (e.g., in the case
538

of a large language model), releasing of a model checkpoint, or other means that are
539

appropriate to the research performed.
540

• While NeurIPS does not require releasing code, the conference does require all submis-
541

sions to provide some reasonable avenue for reproducibility, which may depend on the
542

nature of the contribution. For example
543

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
544

to reproduce that algorithm.
545

(b) If the contribution is primarily a new model architecture, the paper should describe
546

the architecture clearly and fully.
547

(c) If the contribution is a new model (e.g., a large language model), then there should
548

either be a way to access this model for reproducing the results or a way to reproduce
549

the model (e.g., with an open-source dataset or instructions for how to construct
550

the dataset).
551

(d) We recognize that reproducibility may be tricky in some cases, in which case
552

authors are welcome to describe the particular way they provide for reproducibility.
553

In the case of closed-source models, it may be that access to the model is limited in
554

some way (e.g., to registered users), but it should be possible for other researchers
555

to have some path to reproducing or verifying the results.
556

5. Open access to data and code
557

Question: Does the paper provide open access to the data and code, with sufficient instruc-
558

tions to faithfully reproduce the main experimental results, as described in supplemental
559

material?
560

19


---Page Break---
Answer: [Yes]
561

Justification: Our code will be made public.
562

Guidelines:
563

• The answer NA means that paper does not include experiments requiring code.
564

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
565

public/guides/CodeSubmissionPolicy) for more details.
566

• While we encourage the release of code and data, we understand that this might not be
567

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
568

including code, unless this is central to the contribution (e.g., for a new open-source
569

benchmark).
570

• The instructions should contain the exact command and environment needed to run to
571

reproduce the results. See the NeurIPS code and data submission guidelines (https:
572

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
573

• The authors should provide instructions on data access and preparation, including how
574

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
575

• The authors should provide scripts to reproduce all experimental results for the new
576

proposed method and baselines. If only a subset of experiments are reproducible, they
577

should state which ones are omitted from the script and why.
578

• At submission time, to preserve anonymity, the authors should release anonymized
579

versions (if applicable).
580

• Providing as much information as possible in supplemental material (appended to the
581

paper) is recommended, but including URLs to data and code is permitted.
582

6. Experimental Setting/Details
583

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
584

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
585

results?
586

Answer: [Yes]
587

Justification: All details are listed in the Appendix.
588

Guidelines:
589

• The answer NA means that the paper does not include experiments.
590

• The experimental setting should be presented in the core of the paper to a level of detail
591

that is necessary to appreciate the results and make sense of them.
592

• The full details can be provided either with the code, in appendix, or as supplemental
593

material.
594

7. Experiment Statistical Significance
595

Question: Does the paper report error bars suitably and correctly defined or other appropriate
596

information about the statistical significance of the experiments?
597

Answer: [Yes]
598

Justification: Our experiment results are reflected by classification accuracy.
599

Guidelines:
600

• The answer NA means that the paper does not include experiments.
601

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
602

dence intervals, or statistical significance tests, at least for the experiments that support
603

the main claims of the paper.
604

• The factors of variability that the error bars are capturing should be clearly stated (for
605

example, train/test split, initialization, random drawing of some parameter, or overall
606

run with given experimental conditions).
607

• The method for calculating the error bars should be explained (closed form formula,
608

call to a library function, bootstrap, etc.)
609

• The assumptions made should be given (e.g., Normally distributed errors).
610

• It should be clear whether the error bar is the standard deviation or the standard error
611

of the mean.
612

20


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
613

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
614

of Normality of errors is not verified.
615

• For asymmetric distributions, the authors should be careful not to show in tables or
616

figures symmetric error bars that would yield results that are out of range (e.g. negative
617

error rates).
618

• If error bars are reported in tables or plots, The authors should explain in the text how
619

they were calculated and reference the corresponding figures or tables in the text.
620

8. Experiments Compute Resources
621

Question: For each experiment, does the paper provide sufficient information on the com-
622

puter resources (type of compute workers, memory, time of execution) needed to reproduce
623

the experiments?
624

Answer: [Yes]
625

Justification: All details are listed in the Appendix.
626

Guidelines:
627

• The answer NA means that the paper does not include experiments.
628

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
629

or cloud provider, including relevant memory and storage.
630

• The paper should provide the amount of compute required for each of the individual
631

experimental runs as well as estimate the total compute.
632

• The paper should disclose whether the full research project required more compute
633

than the experiments reported in the paper (e.g., preliminary or failed experiments that
634

didn’t make it into the paper).
635

9. Code Of Ethics
636

Question: Does the research conducted in the paper conform, in every respect, with the
637

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
638

Answer: [Yes]
639

Justification: We follow the code of ethics.
640

Guidelines:
641

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
642

• If the authors answer No, they should explain the special circumstances that require a
643

deviation from the Code of Ethics.
644

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
645

eration due to laws or regulations in their jurisdiction).
646

10. Broader Impacts
647

Question: Does the paper discuss both potential positive societal impacts and negative
648

societal impacts of the work performed?
649

Answer: [NA]
650

Justification: Our work doesn’t have societal impacts.
651

Guidelines:
652

• The answer NA means that there is no societal impact of the work performed.
653

• If the authors answer NA or No, they should explain why their work has no societal
654

impact or why the paper does not address societal impact.
655

• Examples of negative societal impacts include potential malicious or unintended uses
656

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
657

(e.g., deployment of technologies that could make decisions that unfairly impact specific
658

groups), privacy considerations, and security considerations.
659

• The conference expects that many papers will be foundational research and not tied
660

to particular applications, let alone deployments. However, if there is a direct path to
661

any negative applications, the authors should point it out. For example, it is legitimate
662

to point out that an improvement in the quality of generative models could be used to
663

21


---Page Break---
generate deepfakes for disinformation. On the other hand, it is not needed to point out
664

that a generic algorithm for optimizing neural networks could enable people to train
665

models that generate Deepfakes faster.
666

• The authors should consider possible harms that could arise when the technology is
667

being used as intended and functioning correctly, harms that could arise when the
668

technology is being used as intended but gives incorrect results, and harms following
669

from (intentional or unintentional) misuse of the technology.
670

• If there are negative societal impacts, the authors could also discuss possible mitigation
671

strategies (e.g., gated release of models, providing defenses in addition to attacks,
672

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
673

feedback over time, improving the efficiency and accessibility of ML).
674

11. Safeguards
675

Question: Does the paper describe safeguards that have been put in place for responsible
676

release of data or models that have a high risk for misuse (e.g., pretrained language models,
677

image generators, or scraped datasets)?
678

Answer: [NA]
679

Justification: This is not applicable to our work.
680

Guidelines:
681

• The answer NA means that the paper poses no such risks.
682

• Released models that have a high risk for misuse or dual-use should be released with
683

necessary safeguards to allow for controlled use of the model, for example by requiring
684

that users adhere to usage guidelines or restrictions to access the model or implementing
685

safety filters.
686

• Datasets that have been scraped from the Internet could pose safety risks. The authors
687

should describe how they avoided releasing unsafe images.
688

• We recognize that providing effective safeguards is challenging, and many papers do
689

not require this, but we encourage authors to take this into account and make a best
690

faith effort.
691

12. Licenses for existing assets
692

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
693

the paper, properly credited and are the license and terms of use explicitly mentioned and
694

properly respected?
695

Answer: [Yes]
696

Justification: We cite all code, data, and previous works in a proper manner.
697

Guidelines:
698

• The answer NA means that the paper does not use existing assets.
699

• The authors should cite the original paper that produced the code package or dataset.
700

• The authors should state which version of the asset is used and, if possible, include a
701

URL.
702

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
703

• For scraped data from a particular source (e.g., website), the copyright and terms of
704

service of that source should be provided.
705

• If assets are released, the license, copyright information, and terms of use in the
706

package should be provided. For popular datasets, paperswithcode.com/datasets
707

has curated licenses for some datasets. Their licensing guide can help determine the
708

license of a dataset.
709

• For existing datasets that are re-packaged, both the original license and the license of
710

the derived asset (if it has changed) should be provided.
711

• If this information is not available online, the authors are encouraged to reach out to
712

the asset’s creators.
713

13. New Assets
714

Question: Are new assets introduced in the paper well documented and is the documentation
715

provided alongside the assets?
716

22


---Page Break---
Answer: [NA]
717

Justification: This is not applicable to our work.
718

Guidelines:
719

• The answer NA means that the paper does not release new assets.
720

• Researchers should communicate the details of the dataset/code/model as part of their
721

submissions via structured templates. This includes details about training, license,
722

limitations, etc.
723

• The paper should discuss whether and how consent was obtained from people whose
724

asset is used.
725

• At submission time, remember to anonymize your assets (if applicable). You can either
726

create an anonymized URL or include an anonymized zip file.
727

14. Crowdsourcing and Research with Human Subjects
728

Question: For crowdsourcing experiments and research with human subjects, does the paper
729

include the full text of instructions given to participants and screenshots, if applicable, as
730

well as details about compensation (if any)?
731

Answer: [NA]
732

Justification: This is not applicable to our work.
733

Guidelines:
734

• The answer NA means that the paper does not involve crowdsourcing nor research with
735

human subjects.
736

• Including this information in the supplemental material is fine, but if the main contribu-
737

tion of the paper involves human subjects, then as much detail as possible should be
738

included in the main paper.
739

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
740

or other labor should be paid at least the minimum wage in the country of the data
741

collector.
742

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
743

Subjects
744

Question: Does the paper describe potential risks incurred by study participants, whether
745

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
746

approvals (or an equivalent approval/review based on the requirements of your country or
747

institution) were obtained?
748

Answer: [NA]
749

Justification: This is not applicable to our work.
750

Guidelines:
751

• The answer NA means that the paper does not involve crowdsourcing nor research with
752

human subjects.
753

• Depending on the country in which research is conducted, IRB approval (or equivalent)
754

may be required for any human subjects research. If you obtained IRB approval, you
755

should clearly state this in the paper.
756

• We recognize that the procedures for this may vary significantly between institutions
757

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
758

guidelines for their institution.
759

• For initial submissions, do not include any information that would break anonymity (if
760

applicable), such as the institution conducting the review.
761

23


---Page Break---
