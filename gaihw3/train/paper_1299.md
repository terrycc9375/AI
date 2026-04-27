Bridging Inter-task Gap of Continual Self-supervised
Learning with External Data

Anonymous Author(s)
Affiliation
Address
email

Abstract

Recent research on Self-Supervised Learning (SSL) has demonstrated its ability to
1

extract high-quality representations from unlabeled samples. However, in continual
2

learning scenarios where training data arrives sequentially, SSL’s performance
3

tends to deteriorate. This study focuses on Continual Contrastive Self-Supervised
4

Learning (CCSSL) and highlights that the absence of contrastive learning on inter-
5

task data, due to the unavailability of historical samples, leads to a significant drop
6

in performance. To tackle this issue, we introduce a simple and effective method
7

called BGE, which Bridges the inter-task Gap of CCSSL using External data from
8

publicly available datasets. BGE enables the contrastive learning of each task data
9

with external data, allowing relationships between them to be passed along the tasks,
10

thereby facilitating implicit inter-task data comparisons. To overcome the limitation
11

of the external data selection and maintain its effectiveness, we further propose
12

the One-Propose-One algorithm to collect more relevant and diverse high-quality
13

samples from the chosen external data while filtering out distractions from the out-
14

of-distribution data. Experiments show that BGE can generate better discriminative
15

representation in CCSSL, especially for inter-task data, and improve classification
16

results with various external data compositions. Additionally, the proposed method
17

can be seamlessly integrated into existing continual learning methods yielding
18

significant performance improvement.
19

1
Introduction
20

In recent years, deep neural networks [13, 22, 35] have achieved great success, but plenty of works
21

are under the assumption that all data are available simultaneously for training. In practical scenarios,
22

acquiring the entire dataset at once is often challenging due to data being constantly updated. In this
23

case, training the network continually suffers from catastrophic forgetting [38], meaning that the
24

network severely forgets old task knowledge after learning the new one. Hence, continual learning
25

investigates methods to train networks incrementally while mitigating catastrophic forgetting.
26

Although continual learning has been widely studied and numerous effective methods [32, 36, 40]
27

have been proposed, most existing research remains focused on supervised learning, with Continual
28

Contrastive Self-Supervised Learning (CCSSL) receiving relatively little attention. However, studying
29

CCSSL is equally significant.
30

To prevent catastrophic forgetting, prior CCSSL works CaSSLe [16], PFR [18], and POCON [19]
31

use knowledge distillation, while CPPF [11] incorporates prototype clustering. In this paper, we
32

highlight an important but generally overlooked issue in these works: Comparisons of inter-task
33

data are absent. Specifically, a widely accepted opinion in continual learning is that if the sum of
34

each task’s loss is minimized, then continual learning’s performance reaches its upper bound: joint
35

learning. However, in CCSSL, even if each task’s loss is minimized, there is still a gap between joint
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Compare
Compare

Task t-1
Task t

Incomparable

External Dataset

…

Compare
Compare

Bridge the 
Inter-task 
Gap
Compare

a) Fine-tune
b) CaSSLe

d) Joint
c) Ours

bottle
fox
cloud
maple

Figure 1:
Left: Overview of our method BGE. In typical CCSSL methods, the inter-task data
pairs are incomparable. We employ an external dataset to complement these missing comparisons,
effectively bridging the inter-task gap. Right: t-SNE [47] visualization of four classes belonging to
different tasks in continual learning. Compared to prior methods Fine-tune and CaSSLe [16], we
make the inter-task data more separable.

learning. Because joint learning requires any sample pair in the entire dataset to participate in the
37

contrastive loss computation. In contrast, in continual learning, inter-task data are unavailable to each
38

other, meaning this aspect of the contrastive loss is never computed and optimized. This omission
39

increases the likelihood of inter-task class confusion, as illustrated in Figure 1 Right, despite classes
40

from four different tasks having distinctly different semantics, they still show confusion in prior
41

methods Fine-tune and CaSSLe [16]. In contrast, our method and joint training consider inter-task
42

comparisons and can better distinguish them.
43

Since we could not directly use data from other tasks for inter-task comparisons, we would like to
44

compensate for these comparisons with the help of external data. Some prior works [31, 52, 56]
45

have explored using external data for continual learning. GD [31] and ZSCL [56] use external
46

data for distillation to stabilize the feature space, while requiring extensive external data and high
47

computational costs. ST [52] employs external data as additional training data, but as a supervised
48

method, it requires pseudo-labels, making it less robust to out-of-distribution (OOD) data. Tang et
49

al. [45] enhance exemplar diversity with external data. Existing methods focus on using external
50

data in supervised learning, but given that CCSSL does not require labels for training, we propose
51

using external data in CCSSL, which avoids the need for pseudo-labels and is more generalizable and
52

robust to OOD data. Besides, our motivation is to improve feature space by compensating for absent
53

comparisons rather than merely stabilizing it, and it does not require extensive external data.
54

In summary, we propose incorporating publicly available external data into training to compensate for
55

the absent inter-task comparisons, as shown in Figure 1 Left. When the external dataset is sufficiently
56

large, it is reasonable to assume a high probability that some external data share similar features with
57

the task data, even if they are in different classes. By incorporating these high-quality external data
58

into CCSSL, the data from each task can be compared with them. enables the inter-data relationship to
59

be passed along the tasks, thereby constructing implicit inter-task comparisons. Further, considering
60

that external data in open-world scenarios may contain extensive OOD data that is not beneficial for
61

task training, we propose the One-Propose-One (OPO) sampling algorithm, to sample high-quality
62

external data that are relevant to tasks and sufficiently diverse without any hyperparameters.
63

Experiments demonstrate that BGE can be seamlessly integrated into existing methods, resulting
64

in significant performance improvement. We also point out that although it may seem unsurprising
65

that network performance improves with more training data, this improvement is not due to richer
66

input features, because when we add equal external data into joint training, the performance doesn’t
67

improve even sometimes decreases. Instead, BGE compensates for the absent comparisons caused by
68

2


---Page Break---
inter-task data unavailability, which is much more meaningful in continual learning. Our contributions
69

can be summarized as follows:
70

• We point out that existing methods overlook the issue of inter-task data comparisons, and
71

propose BGE to incorporate external data into training to address this gap.
72

• We propose the One-Propose-One (OPO) sampling algorithm to sample external data that
73

are relevant to tasks and sufficiently diverse, while also filtering out OOD data that are not
74

beneficial for learning.
75

• Experiments show that BGE can be seamlessly integrated into existing CCSSL methods and
76

consistently yields significant improvement.
77

2
Related work
78

Self-Supervised Learning (SSL)
SSL trains the network without the need for supervised signals.
79

One of the prominent branches is contrastive learning [5, 8–10, 21, 23, 53]. The objective of
80

contrastive learning can be roughly explained as reducing the distance between positive pairs while
81

enlarging it between negative pairs. SimCLR [8] simply follows this objective but requires a large
82

batch size. MoCo [10, 23] introduces a momentum encoder and a negative sample dictionary to
83

solve this problem. SwAV [5] and Barlow Twins [53] introduces prototype comparisons and cross-
84

decorrelation loss, respectively. Then BYOL [21] and SimSiam [9] can conduct contrastive learning
85

without negative samples. However, all these methods assume that a large dataset is available for
86

pre-training, which is often impractical in real-world scenarios where data acquisition is incremental.
87

Therefore, we research a continual method, which is more practical.
88

Since no labeling requirement, incorporating external data into SSL is straightforward. Prior long-
89

tailed SSL works [3, 28] leverage external data to balance head and tail classes. Instead, we extend the
90

exploration to continual learning, aiming to use external data to compensate for the absent inter-task
91

comparisons while further preventing catastrophic forgetting.
92

Continual learning
Continual learning allows the network to learn from sequentially arriving data
93

and prevent catastrophic forgetting. Existing continual learning methods can be categorized into
94

three groups, which are 1) Regularization-based methods [1, 14, 29, 32, 34, 50, 54] add additional
95

regularization constraints such as knowledge distillation [14, 32, 50] or limiting important parameters
96

update [1, 29, 34, 54] to network training. 2) Replay-based methods [4, 26, 40, 43, 55] save few
97

representative data from old tasks called exemplars to recover the distribution of old data when the
98

new task is trained. 3) Architecture-based methods [15, 36, 37, 41, 51], which adjust the architecture
99

or parameters of the network during each task training. Currently, most continual learning methods
100

still focus on supervised learning. While some of them [6, 33, 44] draw on the idea of contrastive
101

learning, there are still few works consider continual learning without any supervision. Among them,
102

CaSSLe [16], PFR[18], and POCON[19] use distillation, and CPPF[11] adds clustering to form
103

a more complete framework. Sy-CON [7] also reveals the distinction between CCSSL and joint
104

training, but it only additionally passes current task data into the old network to get more diverse
105

intra-task negative features, which still fails to provide effective inter-task comparisons. Thus it
106

underperforms in most contrastive learning frameworks. Compared to them, we introduce external
107

data to facilitate implicit inter-task comparisons to solve the problem of absent inter-task comparisons.
108

3
Proposed method
109

3.1
Preliminary
110

Contrastive Self-Supervised Learning (CSSL)
In Self-Supervised Learning (SSL), the dataset D
111

contains only n image inputs {x1, x2, ..., xn} without labels. SSL trains a network fθ parameterized
112

by θ to map these inputs to embeddings {z1, z2, ..., zn}. Many well-known SSL works [5, 8, 21, 23,
113

53] use contrastive learning framework. In contrastive learning, a random augmentation function
114

A is pre-designed. Given an input x, two augmented views (xa, xb) are obtained by applying A
115

twice. Subsequently, embeddings za = fθ(xa) and zb = fθ(xb) are passed through a projector hθ′
116

parameterized by θ′ to get z′
a = hθ′(za), z′
b = hθ′(zb), which are involved in LSSL. In essence,
117

3


---Page Break---
LSSL expects the network to output similar embeddings for two views of the same input (i.e. positive
118

pair), while ensuring that embeddings from views of different inputs (i.e. negative pair) are dissimilar.
119

Continual CSSL (CCSSL)
In CCSSL setting, The overall dataset D is divided into multiple tasks.
120

Assuming that T tasks {T1, T2, ..., TT } are to be learned, D can be divided into {D1, D2, ..., DT },
121

where Di ∩Dj = ∅, ∀i, j ∈{1 : T}. Also as SSL, for each task Tt , Dt is only composed of nt
122

images {x1, x2, ..., xnt} without labels. Continual learning requires the network to learn knowledge
123

as each task’s data arrives sequentially, with dataset Di only available at Ti. The optimization
124

objective is to continually train the network parameter θ to satisfy every task, which is defined as:
125

argmin
θ

T
X

t=1
E(xa,xb)∼A(Dt)LSSL(hθ′(fθ(xa)), hθ′(fθ(xb)))
(1)

3.2
Revising and improving CCSSL via external data
126

Typical contrastive learning paradigms [8, 23, 53] can be generalized as reducing distances between
127

positive pairs and enlarging them between negative pairs on feature hyperspheres. Adjusting the
128

interrelationships of sample pairs in this way enables the network to effectively represent features
129

[27, 49]. However, in CCSSL, the data is divided by tasks. During the learning process of task Tt, data
130

from other tasks are unavailable. This prevents adequate tuning of inter-sample relationships, resulting
131

in suboptimal network training. We identify two reasons for this suboptimality: 1) The network
132

rapidly forgets knowledge about old data due to catastrophic forgetting, so their features cannot
133

be well extracted in subsequent tasks. 2) Insufficient learning about each task occurs because data
134

from one task cannot act as negative samples for another task. While prior works address problem 1
135

through techniques like distillation [16, 18, 19] and clustering [11], problem 2 remains underexplored.
136

However, we argue that this is unreasonable, and solving problem 2 is equally important.
137

Prior works [20, 32] widely agree that in the ideal case, continual learning can perform up to joint
138

learning, wherein no forgetting occurs and each task reaches optimality. However, in CSSL, even if
139

no forgetting occurs, there is still an optimization gap between continual and joint learning due to the
140

absence of inter-task data comparisons in the training objective. Unlike supervised learning which
141

guides the network through labels, CSSL relies on data interactions for network learning. When data
142

is incomplete, the training objective also becomes incomplete. For better comprehension, we can
143

decompose the joint training contrastive loss into two terms as in Eq. 2, representing the comparisons
144

of intra-task and inter-task data, denoted as Lintra and Linter, respectively. Lintra is the training
145

objective of the conventional CCSSL, also referred to as Lcontinual. However, for input x ∈Dt
146

in task Tt, negative samples come exclusively from Dt rather than the overall dataset D, making
147

direct comparisons between inter-task data infeasible. Consequently, Linter can not be computed and
148

optimized in continual learning forever, resulting in a Linter gap between Lcontinual and Ljoint.
149

Ljoint = 1

T

T
X

t=1


Lintra = Lcontinual
z
}|
{
E(xa,xb)∼A(Dt)LSSL (hθ′ (fθ (xa)) , hθ′ (fθ (xb)))

+ E
xa∼A(Dt),
xb∼A(D−Dt) LSSL (hθ′ (fθ (xa)) , hθ′ (fθ (xb)))
|
{z
}
Linter


(2)

We argue that the lack of optimization for Linter leads to confusion between inter-task data. Figure 1
150

Right compares the t-SNE visualizations of features from 4 CIFAR100 classes under joint and 10
151

tasks continual training (4 classes belong to different tasks during continual training). Compared to
152

the joint-trained network, the continually trained network shows poor clustering and severe class
153

boundary confusion. More experiments about inter-task confusion can be found at Appendix A.2.1.
154

Despite CaSSLe [16] employing distillation to consolidate old knowledge, the issue of inter-task class
155

boundary confusion remains. To address the overlooked problem of Linter, a straightforward idea
156

is to save exemplars for each task. However, this may raise serious privacy concerns. We therefore
157

explore an alternative method to optimize Linter without exemplars and protect the discriminative
158

4


---Page Break---
class boundaries. Figure 1c shows the feature distribution of our method, with all 4 inter-task classes
159

better distinguished, and the overall distribution closer to joint training.
160

To compensate for Linter, bridging the gap of inter-task comparisons is essential. This requires
161

introducing additional comparisons into each task, implying extra data incorporation. Under the
162

constraints of continual learning, simultaneous access to data from multiple tasks is infeasible.
163

Therefore, the idea emerges to incorporate publicly available external data into CCSSL to address the
164

lack of inter-task comparisons. Each task’s data can be directly compared with external data, enabling
165

relationships between data to be passed along the task sequence. Moreover, using external data better
166

protects privacy, and the costs of obtaining unlabeled data from public data sources are extremely low.
167

We thus propose our method BGE, meaning Bridging the inter-task comparison Gap with External
168

data, as shown in Figure 1 Left. BGE incorporates external data into each task’s training except
169

the first one, and resamples part of them after each task using our sampling algorithm ( detailed in
170

Section 3.3). This external data acts as a bridge for inter-task comparisons, constructing implicit
171

comparisons for inter-task data. For task Tt, with Dt−1
e
as the external data sampled after task Tt−1,
172

the training objective is defined as:
173

Lt = E(xa,xb)∼A(Dt∪Dt−1
e
)LSSL (hθ′ (fθ (xa)) , hθ′ (fθ (xb)))
(3)

Incorporating external data aligns the optimization objective of continual learning more closely with
174

Eq. 2, enhancing the mutual understanding of inter-task classes.
175

3.3
One-Propose-One (OPO) sampling
176

While abundant external data features generally cover in-task data comprehensively, incorporating all
177

external data into continual learning is impractical due to computational constraints. Additionally,
178

open-world external data may include substantial task-irrelevant out-of-distribution (OOD) data,
179

which is unhelpful for training. Therefore, a sampling algorithm is needed to select high-quality
180

external data. We observe that Linter includes comparisons of current task data Dt with both old task
181

data D1:t−1 and future task data Dt+1:T . So sampled external data should ideally proxy for both old
182

and future task data. To represent old data, sampled data should have similar features to them, while
183

representing future data requires imaginative sampling. Therefore, our sampling algorithm is based
184

on both proximity and diversity considerations, and integrates these two aspects into a single objective
185

without any hyperparameters. We noted that prior sampling algorithms [3, 28] for long-tailed learning
186

also consider proximity and diversity, but they require hyperparameters selection.
187

We measure proximity using the cosine distance between sample features. On the other hand, prior
188

work [49] indicates that to avoid collapse, contrastive learning methods tend to map all inputs to
189

a uniform distribution within the feature hypersphere (i.e. uniformity). Thus we assume that the
190

entire distribution of the current task data approximately covers the hypersphere, ensuring diversity.
191

Based on the above, we propose a sampling algorithm called One-Propose-One (OPO) as depicted
192

in Algorithm 1. After training each task Tt, OPO constructs the external dataset Dt
e, which is then
193

incorporated in training task Tt+1. Specifically, OPO considers that each in-task sample can equally
194

propose an external sample with the closest feature distance to itself and has not been proposed.
195

Given the current task budget Kt, we collect all proposed samples as a candidate set Dc, and select
196

the Kt minimum distance samples to be added to the external dataset Dt
e. We follow iCaRL [40]’s
197

exemplar update algorithm, maintaining an equal budget for each task within the total budget K.
198

OPO ensures proximity and diversity without hyperparameters, maintaining similarity to old data and
199

adequate coverage of future data features.
200

4
Experiments
201

4.1
Experimental setup
202

Dataset setup
We conduct experiments with the following datasets: 1) CIFAR100 [30], which
203

contains 100 classes, each with 500 train images and 100 test images. Each image is 32×32 pixels.
204

We follow the class incremental learning setting to split the classes equally by the number of tasks.
205

Experiments are conducted under 4 tasks and 10 tasks settings, wherein each task contains 25 classes
206

5


---Page Break---
Algorithm 1 One-Propose-One(OPO) Sampling Algorithm
Input: current task ID t, current task dataset Dt, entire external dataset Dout, last task sampled
external dataset Dt−1
e
, model f, total budget K, cosine distance metric cos(·, ·)
Output: sampled external dataset Dt
e
1: Calculate current task budget Kt = K

t , Adjust Dt−1
e
= REDUCEDATA(Dt−1
e
, Kt) [40]
2: Create candidate set Dc = {}
3: while | Dc |< Kt do
4:
for each x ∈Dt do
5:
u = argminx′∈(Dout−Dt−1
e
)cos(f(x), f(x′)), du = minxi∈Dtcos(f(xi), f(u))
6:
Dc = Dc ∪{u}, Dout = Dout −{u}
7:
end for
8: end while
9: D′
c = SORT(Dc, key = du) [: Kt], Dt
e = Dt−1
e
∪D′
c
10: return Dt
e

and 10 classes. 2) ImageNet100 [46], which consists of 100 classes selected from ImageNet [12],
207

with a total of 130K images of 224×224 pixels. It is equally split under 5 tasks and 10 tasks settings.
208

External dataset setup
For CIFAR100, the selected external datasets include CIFAR10,
209

Places365test (the test set of Places365 [57]) and ImageNet-R [24], among them, Places365test and
210

ImageNet-R are OOD for CIFAR100. CIFAR10 contains 50,000 images with 32×32 pixels in 10
211

classes. Places365 is a scene recognition dataset with its test set containing 328,500 images of various
212

scenes. ImageNet-R contains 24,000 images featuring art, cartoons, and other styles. We resize both
213

Places365test and ImageNet-R to 32×32 pixels. We consider three compositions of external datasets,
214

CIFAR (CIFAR10), CP (CIFAR10+Places365test) and CPI (CIFAR10+Places365test+ImageNet-R)
215

For ImageNet100, the external datasets include ImageNet900, Places365 and DomainNet [39].
216

ImageNet900 is all data in ImageNet excluding ImageNet100, totaling 1.1 million images. Places365
217

contains 1.8 million images, and DomainNet contains 0.6 million images of 6 domains. They are also
218

used here as OOD data. All data are 224×224 pixels. We consider three compositions of external
219

datasets, IN (ImageNet-900), INP (ImageNet900+Places365) and IND (ImageNet900+DomainNet).
220

Baselines
We compare the original performance of existing exemplar-free CCSSL methods to their
221

performance when with BGE. The methods we compare include 1) Fine-Tune (FT): Sequentially
222

training the network with data from each task without additional prevention of catastrophic forgetting.
223

2) CaSSLe [16]: Introducing a distillation loss between the current model and the old model in
224

the form of contrastive loss. 3) PFR [18]: Addressing catastrophic forgetting based on functional
225

regularization [17]. We slightly optimized its network structure and training procedure.
226

Training and evaluation setup
Unless specified otherwise, all experiments employ Barlow Twins
227

[53] as the contrastive learning framework and Resnet18 [22] as the backbone. The sampling budget
228

is uniformly set at 10K. For evaluation, we follow [16, 18, 19] to report the linear evaluation accuracy
229

of the final network across all classes as the evaluation metric. For other setups see Appendix A.1.
230

4.2
Results
231

Performance improvement on prior methods
We compare the performance improvement BGE
232

yields to the base methods when using different external data compositions. Table 1 shows that
233

on CIFAR100, BGE can consistently and significantly improve base methods. It is worth noting
234

that as the number of tasks increases, BGE yields even greater improvement, with improvement of
235

1.5%-3.5% for 4 tasks and 2.5%-7% for 10 tasks. This is also in line with our motivation, as an
236

increasing number of tasks results in more missing inter-task data comparisons.
237

Moreover, across different external dataset compositions, we observe that CIFAR yields the most
238

significant improvement. This is attributed to the CIFAR10 dataset best matches the distribution of
239

CIFAR100, thereby offering highly relevant features, even if their classes do not intersect. When in-
240

corporating datasets like Places365 or ImageNet-R, which are OOD for CIFAR100, the improvement
241

decreases. Thanks to our OPO sampling algorithm can well resist the harm of OOD data (detailed in
242

6


---Page Break---
Table 1: Comparison of BGE’s performance improvement on CIFAR100. CIFAR, CP, and CPI are
different external dataset compositions. Performance was evaluated by linear evaluation accuracy of
the final network. We equally divided classes into 4 tasks and 10 tasks. BGE consistently improves
base methods across different external dataset compositions. As for Joint training, ED represents
adding equivalent external data, which does not improve the performance.

Methods
CIFAR
CP
CPI

4tasks
10tasks
4tasks
10tasks
4tasks
10tasks

FT
56.19
49.36
56.19
49.36
56.19
49.36
FT+BGE
59.49(+3.30) 56.62(+7.26) 58.69(+2.50) 55.14(+5.78) 58.71(+2.52) 55.74(+6.38)

CaSSLe [16]
60.04
53.89
60.04
53.89
60.04
53.89
CaSSLe+BGE
62.38(+2.34) 58.14(+4.25) 61.72(+1.68) 56.92(+3.03) 61.51(+1.47) 56.36(+2.47)

PFR [18]
60.92
55.57
60.92
55.57
60.92
55.57
PFR+BGE
64.37(+3.45) 61.02(+5.45) 63.15(+2.23) 60.31(+4.74) 62.88(+1.96) 59.99(+4.42)

Joint Acc

Joint
68.09
68.09
68.09
Joint+ED
68.15(+0.06)
67.11(-0.98)
68.19(+0.10)

Table 2: Performance improvement yielded by BGE on ImageNet100. IN, INP, and IND are different
external dataset compositions. ED represents adding equivalent external data in joint training.

Methods
IN
INP
IND

5tasks
10tasks
5tasks
10tasks
5tasks
10tasks

FT
64.02
56.72
64.02
56.72
64.02
56.72
FT+BGE
68.20(+4.18) 64.16(+7.44) 67.84(+3.82) 64.08(+7.36) 69.06(+5.04) 65.00(+8.28)

CaSSLe [16]
70.02
60.68
70.02
60.68
70.02
60.68
CaSSLe+BGE
72.46(+2.44) 66.80(+6.12) 71.44(+1.42) 65.94(+5.26) 72.68(+2.66) 67.10(+6.42)

PFR [18]
70.14
63.12
70.14
63.12
70.14
63.12
PFR+BGE
72.52(+2.38) 69.28(+6.16) 72.94(+2.80) 68.40(+5.28) 72.60(+2.46) 68.94(+5.82)

Joint Acc

Joint
80.44
80.44
80.44
Joint+ED
80.24(-0.20)
79.70(-0.74)
78.88(-1.56)

Section 4.3). On ImageNet100, the performance improvement is shown in Table 2, showcasing a
243

similar improvement regularity to that observed on CIFAR100. BGE achieves 1.5%-4% improvement
244

for 5 tasks and 5%-7.5% improvement for 10 tasks. More experiments see Appendix A.2.7.
245

We also emphasize that although it might seem intuitive that network performance would improve
246

with richer data because of richer features, BGE yielded improvement does not simply stem from
247

using more data. In Table 1 and Table 2, we incorporate an equal amount of external data into
248

joint training. However, the results do not improve, and may even decrease when the external data
249

contains OOD samples. We believe this is because incorporating irrelevant external data into the
250

training process causes the model to allocate some capacity to learning these unrelated data, thereby
251

weakening its focus on the in-task data. Hence, the learning of external data can not directly contribute
252

to the learning of in-task data.
253

Long task sequence experiments
We conduct experiments with 100 tasks on CIFAR100, which
254

means one task only contains one class, to verify the effectiveness of BGE on long task sequences.
255

We set the sampling budget to 1000. Figure 2 shows the performance of different base methods
256

with or without BGE as the learned tasks increase. On one hand, BGE improves the final network
257

performance, especially evident in FT and PFR. On the other hand, the network’s performance
258

increases even more rapidly with BGE, indicating that the network’s generalization ability to unseen
259

7


---Page Break---
FT
CaSSLe
PFR

20
40
60
80
100
5

10

15

20

25

30

35

40

45

50

tasks

  w/o BGE
  w/ BGE

20
40
60
80
100
5

10

15

20

25

30

35

40

45

50

tasks

  w/o BGE
  w/ BGE

20
40
60
80
100
5

10

15

20

25

30

35

40

45

50

tasks

  w/o BGE
  w/ BGE

Accuracy(%)

Figure 2: Performance improvement of BGE at CIFAR100 100 tasks setting.

Table 3: Accuracy on CIFAR100 and ImageNet100 with different sampling algorithms. Bold
indicates better performance.

CIFAR100 FT
CIFAR100 PFR

External dataset
CP
CPI
CP
CPI

Sampling algorithm
4tasks
10tasks
4tasks
10tasks
4tasks
10tasks
4tasks
10tasks

random
57.41
52.78
57.22
52.56
62.57
59.33
62.58
58.45
OPO
58.69
55.14
58.71
55.74
63.15
60.31
62.88
59.99

ImageNet100 FT
ImageNet100 PFR

External dataset
INP
IND
INP
IND

Sampling algorithm
4tasks
10tasks
4tasks
10tasks
4tasks
10tasks
4tasks
10tasks

random
66.50
61.90
66.90
61.90
71.36
67.26
72.56
67.98
OPO
67.84
64.08
69.06
65.00
72.94
68.40
72.60
68.94

tasks is higher. This stems from BGE can both overcome catastrophic forgetting and compare with
260

future tasks it guessed, thus accumulating more knowledge in the early training stages.
261

4.3
Ablation study
262

CIFAR
CPI
0

5

10

15

20

25

30

35

40

FID Score ↓

External datasets compositions

 Random       
 OPO

Figure 3: FID score of different sam-
pling algorithms when CIFAR and
CPI as external data.

Sampling algorithm
Table 3 shows the effect of OPO sam-
263

pling compared to random sampling for FT and PFR improve-
264

ment when external datasets contain OOD data. OPO algo-
265

rithm consistently provides more improvement than random
266

sampling. However, we also observed that when all external
267

data are in-distribution (ID), the improvement from OPO algo-
268

rithm is not stable. This suggests that external data quality is
269

sufficiently high, making random sampling sufficient for our
270

needs. To validate this, we calculated the Fréchet Inception
271

Distance (FID) scores [25] between the in-task dataset and
272

external datasets obtained by different sampling algorithms
273

under CIFAR and CPI compositions, as shown in Figure 3.
274

A lower FID score indicates greater similarity between two
275

datasets, and vice versa. Figure 3 shows that with the CIFAR
276

composition, the FID score is lower, and the effect of the OPO
277

algorithm is little, indicating that this dataset is already of
278

high quality. In contrast, under CPI, the FID score is higher when random sampling, while shows a
279

significant decrease when OPO sampling. It indicates that the OPO algorithm adjusts the distribution
280

of the external dataset considerably to make it more compatible with the in-task dataset. Therefore
281

OPO algorithm will have more advantages when the external dataset contains OOD data.
282

Besides, we observed that the advantage of OPO sampling algorithm is more significant on the
283

ImageNet100 dataset. We believe this can be attributed to two factors: 1) Higher image pixels contain
284

8


---Page Break---
more information, and fewer images will satisfy the proximity. 2) With a larger quantity of external
285

data, there are more potentially high-quality data, facilitating better sampling.
286

Table 4: Comparison of additional
positive and negative pairs’ effects.

Negative
Positive
Acc

52.79
✓
53.40
✓
55.61
✓
✓
56.21

Effect of additional positive and negative pairs
We fur-
287

ther investigate whether additional positive or negative pairs
288

provided by BGE contribute more to performance improve-
289

ment. We conduct experiments based on CaSSLe [16] on the
290

CIFAR100 4 tasks setting. Because this experiment requires
291

explicitly calculating the loss incurred by each positive and
292

negative pair, we convert the framework to SimCLR [8]. We
293

masked the additional positive or negative pairs in Table 4.
294

The results show that both types of pairs improve performance
295

individually, and negative pairs yield more significant improve-
296

ment, supporting our emphasis that the impact of absent inter-task comparisons is severe but neglected.
297

But positive pairs also yield performance improvement, which is because high-quality external data
298

have feature intersections with in-task data, proving that external data can prevent catastrophic
299

forgetting as well. With the synergistic effect of both, the improvement reaches the highest.
300

Experiments with only OOD external data
In the experiments presented in Table 1 and Table 2,
301

all external data contain some amount of ID data. To assess BGE’s performance without any ID data
302

in the external dataset, we conduct experiments on CIFAR100 4 tasks based on PFR, as shown in
303

Table 5. The external dataset is only composed of ImageNet-R or Places365test. In joint training,
304

these data are detrimental. While in continual training, BGE consistently improves the base method
305

by nearly 2%, regardless of the composition of OOD data used. It indicates that the performance
306

improvement from BGE does not only come from imitating in-task data features, but also from
307

introducing similar additional comparisons into each task itself, which is beneficial for constructing
308

implicit inter-task comparisons. Even if the external data has few recognizable similar features to
309

the in-task data, the network can still try its best to mine valuable knowledge from external data to
310

compensate for inter-task comparisons.
311

Table 5: Effectiveness of BGE when external data are totally OOD.

External dataset compositions
PFR
+BGE
Joint
Joint+ED
ImageNet-R
Places365test
✓
60.92
62.85(+1.93)
68.09
68.03(-0.06)
✓
60.92
62.81(+1.89)
68.09
67.75(-0.34)
✓
✓
60.92
62.88(+1.96)
68.09
67.15(-0.94)

Table 6: Performance of BGE
when choosing more types of
datasets.

External datasets
Acc

N/A
60.92
GenImage [58]
64.37
CC3M [42]
63.53
CUB200 [48]
62.42

BGE with more types of datasets
We validate the effective-
312

ness of BGE across more aspects of external datasets. Table 6
313

presents the results when using GenImage [58], a dataset of gen-
314

erated images; CC3M [42], a dataset sourced from the Internet;
315

and CUB200 [48], a fine-grained bird dataset as external dataset.
316

Experiments with GenImage and CC3M demonstrate BGE’s effec-
317

tiveness with both model-generated and real-world Internet data,
318

demonstrating its practical value. Since CUB200 is fine-grained
319

and lacking in diversity, it is extremely unfriendly to BGE, yet
320

BGE can still improve the base method.
321

5
Conclusion
322

In this paper, we address a commonly overlooked but severe issue in Continual Contrastive Self-
323

Supervised Learning (CCSSL): the lack of inter-task comparisons. To tackle this, we propose our
324

method BGE to incorporate external data into training, bridging the inter-task gap and facilitating
325

implicit inter-task data comparisons. We also design the One-Propose-One sampling algorithm to
326

select high-quality external data and filter out irrelevant OOD data. BGE can be seamlessly integrated
327

into existing methods and yield significant improvement.
328

9


---Page Break---
References
329

[1] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars. Memory aware synapses:
330

Learning what (not) to forget. In Proceedings of the European conference on computer vision
331

(ECCV), pages 139–154, 2018.
332

[2] S. Amir, Y. Gandelsman, S. Bagon, and T. Dekel. Deep vit features as dense visual descriptors.
333

arXiv preprint arXiv:2112.05814, 2(3):4, 2021.
334

[3] J. Bai, Z. Liu, H. Wang, J. Hao, Y. Feng, H. Chu, and H. Hu. On the effectiveness of out-
335

of-distribution data in self-supervised long-tail learning. arXiv preprint arXiv:2306.04934,
336

2023.
337

[4] J. Bang, H. Kim, Y. Yoo, J.-W. Ha, and J. Choi. Rainbow memory: Continual learning with a
338

memory of diverse samples. In Proceedings of the IEEE/CVF conference on computer vision
339

and pattern recognition, pages 8218–8227, 2021.
340

[5] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin. Unsupervised learning of
341

visual features by contrasting cluster assignments. Advances in neural information processing
342

systems, 33:9912–9924, 2020.
343

[6] H. Cha, J. Lee, and J. Shin. Co2l: Contrastive continual learning. In Proceedings of the
344

IEEE/CVF International conference on computer vision, pages 9516–9525, 2021.
345

[7] S. Cha and T. Moon. Sy-con: Symmetric contrastive loss for continual self-supervised represen-
346

tation learning. arXiv preprint arXiv:2306.05101, 2023.
347

[8] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning
348

of visual representations. In International conference on machine learning, pages 1597–1607.
349

PMLR, 2020.
350

[9] X. Chen and K. He. Exploring simple siamese representation learning. In Proceedings of the
351

IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758, 2021.
352

[10] X. Chen, H. Fan, R. Girshick, and K. He. Improved baselines with momentum contrastive
353

learning. arXiv preprint arXiv:2003.04297, 2020.
354

[11] X. Chen, Z. Sun, K. Yan, S. Ding, and H. Lu. Combining past, present and future: A self-
355

supervised approach for class incremental learning. arXiv preprint arXiv:2311.08764, 2023.
356

[12] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical
357

image database. In 2009 IEEE conference on computer vision and pattern recognition, pages
358

248–255. Ieee, 2009.
359

[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani,
360

M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for
361

image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
362

[14] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle. Podnet: Pooled outputs distillation for
363

small-tasks incremental learning. In Computer vision–ECCV 2020: 16th European conference,
364

Glasgow, UK, August 23–28, 2020, proceedings, part XX 16, pages 86–102. Springer, 2020.
365

[15] C. Fernando, D. Banarse, C. Blundell, Y. Zwols, D. Ha, A. A. Rusu, A. Pritzel, and D. Wier-
366

stra. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint
367

arXiv:1701.08734, 2017.
368

[16] E. Fini, V. G. T. Da Costa, X. Alameda-Pineda, E. Ricci, K. Alahari, and J. Mairal. Self-
369

supervised models are continual learners. In Proceedings of the IEEE/CVF Conference on
370

Computer Vision and Pattern Recognition, pages 9621–9630, 2022.
371

[17] S. Garg and Y. Liang. Functional regularization for representation learning: A unified theoretical
372

perspective. Advances in Neural Information Processing Systems, 33:17187–17199, 2020.
373

10


---Page Break---
[18] A. Gomez-Villa, B. Twardowski, L. Yu, A. D. Bagdanov, and J. Van de Weijer. Continually
374

learning self-supervised representations with projected functional regularization. In Proceedings
375

of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3867–3877,
376

2022.
377

[19] A. Gomez-Villa, B. Twardowski, K. Wang, and J. van de Weijer. Plasticity-optimized comple-
378

mentary networks for unsupervised continual learning. In Proceedings of the IEEE/CVF Winter
379

Conference on Applications of Computer Vision, pages 1690–1700, 2024.
380

[20] D. Goswami, Y. Liu, B. Twardowski, and J. van de Weijer. Fecam: Exploiting the heterogeneity
381

of class distributions in exemplar-free continual learning. Advances in Neural Information
382

Processing Systems, 36, 2024.
383

[21] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch,
384

B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new ap-
385

proach to self-supervised learning. Advances in neural information processing systems, 33:
386

21271–21284, 2020.
387

[22] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In
388

Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–
389

778, 2016.
390

[23] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick. Momentum contrast for unsupervised visual
391

representation learning. In Proceedings of the IEEE/CVF conference on computer vision and
392

pattern recognition, pages 9729–9738, 2020.
393

[24] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Para-
394

juli, M. Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution
395

generalization. In Proceedings of the IEEE/CVF international conference on computer vision,
396

pages 8340–8349, 2021.
397

[25] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two
398

time-scale update rule converge to a local nash equilibrium. Advances in neural information
399

processing systems, 30, 2017.
400

[26] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin. Learning a unified classifier incrementally
401

via rebalancing. In Proceedings of the IEEE/CVF conference on computer vision and pattern
402

recognition, pages 831–839, 2019.
403

[27] W. Huang, M. Yi, X. Zhao, and Z. Jiang. Towards the generalization of contrastive self-
404

supervised learning. arXiv preprint arXiv:2111.00743, 2021.
405

[28] Z. Jiang, T. Chen, T. Chen, and Z. Wang. Improving contrastive learning on imbalanced data
406

via open-world sampling. Advances in Neural Information Processing Systems, 34:5997–6009,
407

2021.
408

[29] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan,
409

J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in
410

neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.
411

[30] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images. 2009.
412

[31] K. Lee, K. Lee, J. Shin, and H. Lee. Overcoming catastrophic forgetting with unlabeled data in
413

the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
414

312–321, 2019.
415

[32] Z. Li and D. Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and
416

machine intelligence, 40(12):2935–2947, 2017.
417

[33] H. Lin, B. Zhang, S. Feng, X. Li, and Y. Ye. Pcr: Proxy-based contrastive replay for online
418

class-incremental continual learning. In Proceedings of the IEEE/CVF Conference on Computer
419

Vision and Pattern Recognition, pages 24246–24255, 2023.
420

11


---Page Break---
[34] X. Liu, M. Masana, L. Herranz, J. Van de Weijer, A. M. Lopez, and A. D. Bagdanov. Rotate
421

your networks: Better weight consolidation and less catastrophic forgetting. In 2018 24th
422

International Conference on Pattern Recognition (ICPR), pages 2262–2268. IEEE, 2018.
423

[35] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In
424

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
425

11976–11986, 2022.
426

[36] A. Mallya and S. Lazebnik. Packnet: Adding multiple tasks to a single network by iterative
427

pruning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition,
428

pages 7765–7773, 2018.
429

[37] A. Mallya, D. Davis, and S. Lazebnik. Piggyback: Adapting a single network to multiple tasks
430

by learning to mask weights. In Proceedings of the European conference on computer vision
431

(ECCV), pages 67–82, 2018.
432

[38] M. McCloskey and N. J. Cohen. Catastrophic interference in connectionist networks: The
433

sequential learning problem. In Psychology of learning and motivation, volume 24, pages
434

109–165. Elsevier, 1989.
435

[39] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang. Moment matching for multi-source
436

domain adaptation. In Proceedings of the IEEE/CVF international conference on computer
437

vision, pages 1406–1415, 2019.
438

[40] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. icarl: Incremental classifier and
439

representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern
440

Recognition, pages 2001–2010, 2017.
441

[41] J. Serra, D. Suris, M. Miron, and A. Karatzoglou. Overcoming catastrophic forgetting with
442

hard attention to the task. In International conference on machine learning, pages 4548–4557.
443

PMLR, 2018.
444

[42] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed,
445

image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual
446

Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
447

2556–2565, 2018.
448

[43] H. Shin, J. K. Lee, J. Kim, and J. Kim. Continual learning with deep generative replay. Advances
449

in neural information processing systems, 30, 2017.
450

[44] Z. Song, Y. Zhao, Y. Shi, P. Peng, L. Yuan, and Y. Tian. Learning with fantasy: Semantic-aware
451

virtual contrastive constraint for few-shot class-incremental learning. In Proceedings of the
452

IEEE/CVF conference on computer vision and pattern recognition, pages 24183–24192, 2023.
453

[45] Y.-M. Tang, Y.-X. Peng, and W.-S. Zheng.
Learning to imagine: Diversify memory for
454

incremental learning using unlabeled data. In Proceedings of the IEEE/CVF Conference on
455

Computer Vision and Pattern Recognition, pages 9549–9558, 2022.
456

[46] Y. Tian, D. Krishnan, and P. Isola. Contrastive multiview coding. In Computer Vision–ECCV
457

2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16,
458

pages 776–794. Springer, 2020.
459

[47] L. Van der Maaten and G. Hinton. Visualizing data using t-sne. Journal of machine learning
460

research, 9(11), 2008.
461

[48] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011
462

dataset. 2011.
463

[49] T. Wang and P. Isola. Understanding contrastive representation learning through alignment
464

and uniformity on the hypersphere. In International conference on machine learning, pages
465

9929–9939. PMLR, 2020.
466

[50] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu. Large scale incremental learning.
467

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
468

374–382, 2019.
469

12


---Page Break---
[51] S. Yan, J. Xie, and X. He. Der: Dynamically expandable representation for class incremen-
470

tal learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern
471

recognition, pages 3014–3023, 2021.
472

[52] L. Yu, X. Liu, and J. Van de Weijer. Self-training for class-incremental semantic segmentation.
473

IEEE Transactions on Neural Networks and Learning Systems, 2022.
474

[53] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny. Barlow twins: Self-supervised learning via
475

redundancy reduction. In International conference on machine learning, pages 12310–12320.
476

PMLR, 2021.
477

[54] F. Zenke, B. Poole, and S. Ganguli. Continual learning through synaptic intelligence. In
478

International conference on machine learning, pages 3987–3995. PMLR, 2017.
479

[55] M. Zhai, L. Chen, and G. Mori. Hyper-lifelonggan: Scalable lifelong learning for image
480

conditioned generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and
481

Pattern Recognition, pages 2246–2255, 2021.
482

[56] Z. Zheng, M. Ma, K. Wang, Z. Qin, X. Yue, and Y. You.
Preventing zero-shot transfer
483

degradation in continual learning of vision-language models. In Proceedings of the IEEE/CVF
484

International Conference on Computer Vision, pages 19125–19136, 2023.
485

[57] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba. Places: A 10 million image
486

database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence,
487

2017.
488

[58] M. Zhu, H. Chen, Q. Yan, X. Huang, G. Lin, W. Li, Z. Tu, H. Hu, J. Hu, and Y. Wang. Genimage:
489

A million-scale benchmark for detecting ai-generated image. Advances in Neural Information
490

Processing Systems, 36, 2024.
491

13


---Page Break---
A
Appendix / supplemental material
492

A.1
Experimental details
493

We use SGD optimizer with warmup cosine scheduler to train the network with batchsize of 256. For
494

CIFAR100, we train 500 epochs per task with a learning rate of 0.3 and weight decay of 1e-4 for
495

FT and CaSSLe[16]. For PFR[18], we use the learning rate as 0.4. For ImageNet100, we train 400
496

epochs per task with a learning rate of 0.4 and weight decay of 1e-4.
497

We use one RTX 3090 for CIFAR100 experiments and one A40 for ImageNet100 experiments. For
498

CIFAR100 experiments, it takes about 5 hours in 4 tasks setting and 8 hours in 10 tasks setting. For
499

ImageNet100 experiments, it takes about 17 hours in 5 tasks setting and 27 hours in 10 tasks setting.
500

A.2
More experiments
501

A.2.1
BGE’s improvement to inter-task confusion
502

We categorize the results of classification errors into two types, inter-task confusion (the wrong
503

prediction belongs to a different task than the target) and intra-task confusion (the wrong prediction
504

belongs to the same task as the target). Under the CIFAR100 4 tasks setting, we compare the
505

probability of each of the two types of confusion occurring for the class contained in the last task for
506

the three baseline methods, as shown in Table 7. Ideally, the ratio of intra-task confusion to inter-task
507

confusion should be 1:3, since the ratio of the number of current task classes to the total number
508

of previous task classes is 1:3. However, the inter-task confusion in Table 7 is 5 to 7 times higher
509

than the intra-task confusion, suggesting that the lack of Linter optimization has a severe impact on
510

performance, while BGE improves this and decreases inter-task confusion.

Table 7: Comparison of intra-task confusion and inter-task confusion. ↓means the value is the lower
the better.

Method
Intra-task confusion↓
Inter-task confusion↓

FT
4.56%
33.48%
FT+BGE
4.60%(+0.04%)
30.12%(-3.36%)
CaSSLe
6.84%
32.08%
CaSSLe+BGE
6.08%(-0.76%)
28.52%(-3.56%)
PFR
6.32%
29.64%
PFR+BGE
6.44%(+0.12%)
27.36%(-2.28%)

511

A.2.2
Experiments on the method without negative samples
512

While the results in Table 4 indicate that the effectiveness of BGE mainly stems from additional
513

negative samples, we conducted experiments using the contrastive learning framework BYOL, which
514

calculates contrastive loss without the need of negative samples, as shown in Table 8. The results
515

indicate that our method still achieves improvement, demonstrating its applicability even in methods
516

without negative samples.

Table 8: Performance improvement yielded by BGE in BYOL.

Methods
CIFAR
CP

4tasks
10tasks
4tasks
10tasks

FT
52.36
47.97
52.36
47.97
FT+BGE
56.88(+4.52)
49.42(+1.45)
56.37(+4.01)
49.22(+1.25)

CaSSLe
57.46
52.61
57.46
52.61
CaSSLe+BGE
59.20(+1.78)
56.16(+3.55)
58.92(+1.46)
55.22(+2.61)

517

14


---Page Break---
A.2.3
Visualization of sample algorithm
518

We visualize the relationship between external and in-task samples obtained by different sampling
519

algorithms under CIFAR and CPI compositions, as shown in Figure 4. When CIFAR10 as external
520

data, the distributions of random and OPO samples are similar, both covering the entire area effectively.
521

While in the CPI setting, random sampling fails to cover the entire area, in contrast, the OPO algorithm
522

achieves superior proximity and diversity, consequently leading to greater performance improvement.
523

This observation corroborates our discussion about the sampling algorithm in Section 4.3.

CPI
CIFAR10

Random
OPO
Random
OPO
In-task data
Sampled external data

Figure 4: Comparison of external data sampled by different algorithms. When the entire external data
quality is high (CIFAR), there is little difference between random and OPO sampling. When the data
contains many OOD data (CPI), OPO outperforms random in sampling relevant and diverse samples.

524

A.2.4
Self-supervised learning feature characteristics
525

Previous work [2] points out that self-supervised trained networks map inputs together according
526

to feature characteristics rather than according to labels as supervised trained networks tend to do.
527

Inspired by them, we validate that we adopted network also has such characteristics. Table 9 shows
528

the average number of one sample’s k-nearest neighbors belonging to the class of this sample for
529

networks trained in the supervised or self-supervised manner. It is evident that supervised networks
530

consistently have more same-class neighbors, indicating that they cluster images based on labels. In
531

contrast, self-supervised networks are less influenced by image classes, which is advantageous for
532

incorporating external data.

Table 9: Statistics on how many of the k-nearest neighbors of a sample belong to the same class as
this sample in self-supervised and supervised networks.

k
3
5
10
20
30
50
100
Acc

Supervised
1.76
2.93
5.58
10.87
15.63
24.38
40.86
71.64
Self-supervised
1.36
2.25
4.14
7.24
9.96
14.53
22.00
68.09

533

Table 10 presents the class statistics of the top 100 nearest neighbors of the "willow tree" class on the
534

CIFAR100 dataset, as learned by self-supervised and supervised networks. Self-supervised learning
535

results in a lower proportion of same-class neighbors, indicating less influence from class labels.
536

Additionally, the neighbors of other classes in the self-supervised network exhibit features more
537

similar to the "willow tree" class.
538

This insight suggests that external data, despite having different actual classes with in-task data,
539

can proxy for the in-task data in self-supervised learning due to shared features. Thus giving us
540

confidence that using external data in self-supervised learning as in BGE can yield good results and
541

justify our cosine distance based sampling algorithm.
542

A.2.5
Fairness alignment
543

Introducing external data incurs additional iterations and new knowledge. To ensure fairness, we
544

train the base method PFR for more epochs and use pre-training with external data to initialize the
545

weights for in-task data training. Experimental results, as shown in Table 11, reveal that training
546

15


---Page Break---
Table 10: The class name and average number of the top 5 classes with the highest number of the top
100 neighbors of the "willow tree" class.

Supervised learning
Self-supervised learning

Neighbor class
Avg number
Neighbor class
Avg number

willow tree
48.59
willow tree
18.68
mushroom
7.85
oak tree
18.47
girl
4.19
maple tree
16.45
butterfly
3.05
pine tree
8.48
bus
2.94
forest
8.10

for more epochs and pre-training with external data do not lead to performance improvement. This
547

highlights the effectiveness of BGE under fairer conditions.

Table 11: Comparison of the performance improvement of BGE and other factors to ensure fairness.

Methods
Acc

Base
60.92
Train more epochs
61.21
Use external data to pre-train
61.28
Ours
64.37

548

A.2.6
Experiment statistical significance
549

Due to limited computational resources, we report the mean and standard deviation of three random
550

trials for only the primary experiments in Tables 12 and 13. The performance of the BGE on the three
551

base methods when using CIFAR and CPI as external dataset compositions under the CIFAR100
552

4 tasks and 10 tasks setting is shown in Table 12. Table 13 shows the performance of BGE using
553

different sampling algorithms with CPI as the external dataset, also in the CIFAR100 4 tasks and 10
554

tasks setting, across the same three baseline methods.

Table 12: Results with multiple runs.

Methods
CIFAR
CPI

4tasks
10tasks
4tasks
10tasks

FT
59.80±0.27
56.92±0.29
59.06±0.39
55.18±0.51
CaSSLe
62.39±0.41
57.99±0.28
61.86±0.36
56.52±0.21
PFR
64.13±0.24
60.01±0.02
63.12±0.33
59.94±0.05

Table 13: Results with multiple runs.

Methods
4tasks
10tasks

random
OPO
random
OPO

FT
57.61±0.42
59.06±0.39
52.81±0.23
55.18±0.51
CaSSLe
61.59±0.25
61.86±0.36
55.50±0.23
56.52±0.21
PFR
62.50±0.11
63.12±0.33
58.66±0.27
59.94±0.05

555

A.2.7
Full experiments
556

We present here the full set of experiments, encompassing various base methods, sampling bud-
557

gets, sampling methods, and compositions of external datasets, demonstrating the performance
558

improvement of BGE on CIFAR100 (Table 14) and ImageNet100 (Table 15).
559

16


---Page Break---
Table 14: Full experiment results on CIFAR100 dataset.

Methods
External Dataset
CIFAR10
CP
CPI

Budget
Sample
method
4tasks
10tasks
4tasks
10tasks
4tasks
10tasks

FT

0
-
56.19
49.36
56.19
49.36
56.19
49.36

5K
random 58.65(+2.46) 54.78(+5.42) 57.54(+1.35) 52.09(+2.73) 56.95(+0.76) 52.3(+2.94)

OPO
58.51(+2.32) 54.39(+5.03) 57.56(+1.37) 54.59(+5.23) 58.3(+2.11)
53.15(+3.79)

10K
random 60.01(+3.82) 56.56(+7.20) 57.41(+1.22) 52.78(+3.42) 57.22(+1.03) 52.56(+3.20)

OPO
59.49(+3.30) 56.62(+7.26) 58.69(+2.50) 55.14(+5.78) 58.71(+2.52) 55.74(+6.38)

CaSSLe

0
-
60.04
53.89
60.04
53.89
60.04
53.89

5K
random 61.26(+1.22) 56.72(+2.83) 60.86(+0.82) 54.47(+0.58) 61.06(+1.02) 54.52(+0.63)

OPO
61.35(+1.31) 56.63(+2.74) 61.39(+1.35) 55.24(+1.35) 61.30(+1.26) 55.77(+1.88)

10K
random 62.49(+2.45) 57.49(+3.60) 60.98(+0.94) 55.48(+1.59) 61.44(+1.40) 55.40(+1.51)

OPO
62.38(+2.34) 58.14(+4.25) 61.72(+1.68) 56.92(+3.03) 61.51(+1.47) 56.36(+2.47)

PFR

0
-
60.92
55.57
60.92
55.57
60.92
55.57

5K
random 62.84(+1.92) 60.01(+4.44) 62.39+(1.47) 58.49(+2.92) 62.16(+1.24) 57.78(+2.21)

OPO
62.79(+1.87) 59.66(+4.09) 62.16(+1.24) 59.29(+3.72) 62.87(+1.95) 58.41(+2.84)

10K
random 63.51(+2.59) 61.58(+6.01) 62.57(+1.65) 59.33(+3.76) 62.58(+1.66) 58.45(+2.88)

OPO
64.37(+3.45) 61.02(+5.45) 63.15(+2.23) 60.31(+4.74) 62.88(+1.96) 59.99(+4.42)

Table 15: Full experiment results on ImageNet100 dataset.

Methods
External Dataset
IN
INP
IND

Budget
Sample
method
5tasks
10tasks
5tasks
10tasks
5tasks
10tasks

FT

0
-
64.02
56.72
64.02
56.72
64.02
56.72

10K
random 67.66(+3.64) 63.02(+6.30) 66.50(+2.48) 61.90(+5.18) 66.90(+2.88) 61.90(+5.18)

OPO
68.20(+4.18) 64.16(+7.44) 67.84(+3.82) 64.08(+7.36) 69.06(+5.04) 65.00(+8.28)

CaSSLe

0
-
70.02
60.68
70.02
60.68
70.02
60.68

10K
random 71.52(+1.50) 65.02(+4.34) 71.04(+1.02) 64.34(+3.66) 70.98(+0.96) 65.44(+4.76)

OPO
72.46(+2.44) 66.80(+6.12) 71.44(+1.42) 65.94(+5.26) 72.68(+2.66) 67.10(+6.42)

PFR

0
-
70.14
63.12
70.14
63.12
70.14
63.12

10K
random 72.82(+2.68) 68.20(+5.08) 71.36(+1.22) 67.26(+4.14) 72.56(+2.42) 67.98(+4.86)

OPO
72.52(+2.38) 69.28(+6.16) 72.94(+2.80) 68.40(+5.28) 72.60(+2.46) 68.94(+5.82)

A.3
Limitations and future directions
560

There are still limitations to BGE, such as increased data volume for training, leading to additional
561

computational costs. For future directions, we believe BGE can inspire further research into continual
562

learning from the perspective of inter-task data relationships. Additionally, BGE’s use of external
563

data instead of exemplars to compensate for inter-task comparisons enhances privacy preservation,
564

offering a pathway for future work to address privacy concerns associated with using exemplars. We
565

research methods to allow the network to learn continually, which have no negative impact on society,
566

and at the same time, we proposed method facilitates privacy protection and has a positive impact on
567

society.
568

17


---Page Break---
NeurIPS Paper Checklist
569

1. Claims
570

Question: Do the main claims made in the abstract and introduction accurately reflect the
571

paper’s contributions and scope?
572

Answer: [Yes]
573

Justification: The abstract and introduction in Section 1 accurately reflect our contributions
574

in continual contrastive self-supervised learning.
575

Guidelines:
576

• The answer NA means that the abstract and introduction do not include the claims
577

made in the paper.
578

• The abstract and/or introduction should clearly state the claims made, including the
579

contributions made in the paper and important assumptions and limitations. A No or
580

NA answer to this question will not be perceived well by the reviewers.
581

• The claims made should match theoretical and experimental results, and reflect how
582

much the results can be expected to generalize to other settings.
583

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
584

are not attained by the paper.
585

2. Limitations
586

Question: Does the paper discuss the limitations of the work performed by the authors?
587

Answer: [Yes]
588

Justification: We discuss the limitations of our work in Appendix A.3.
589

Guidelines:
590

• The answer NA means that the paper has no limitation while the answer No means that
591

the paper has limitations, but those are not discussed in the paper.
592

• The authors are encouraged to create a separate "Limitations" section in their paper.
593

• The paper should point out any strong assumptions and how robust the results are to
594

violations of these assumptions (e.g., independence assumptions, noiseless settings,
595

model well-specification, asymptotic approximations only holding locally). The authors
596

should reflect on how these assumptions might be violated in practice and what the
597

implications would be.
598

• The authors should reflect on the scope of the claims made, e.g., if the approach was
599

only tested on a few datasets or with a few runs. In general, empirical results often
600

depend on implicit assumptions, which should be articulated.
601

• The authors should reflect on the factors that influence the performance of the approach.
602

For example, a facial recognition algorithm may perform poorly when image resolution
603

is low or images are taken in low lighting. Or a speech-to-text system might not be
604

used reliably to provide closed captions for online lectures because it fails to handle
605

technical jargon.
606

• The authors should discuss the computational efficiency of the proposed algorithms
607

and how they scale with dataset size.
608

• If applicable, the authors should discuss possible limitations of their approach to
609

address problems of privacy and fairness.
610

• While the authors might fear that complete honesty about limitations might be used by
611

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
612

limitations that aren’t acknowledged in the paper. The authors should use their best
613

judgment and recognize that individual actions in favor of transparency play an impor-
614

tant role in developing norms that preserve the integrity of the community. Reviewers
615

will be specifically instructed to not penalize honesty concerning limitations.
616

3. Theory Assumptions and Proofs
617

Question: For each theoretical result, does the paper provide the full set of assumptions and
618

a complete (and correct) proof?
619

Answer: [NA]
620

18


---Page Break---
Justification: We do not include theoretical results.
621

Guidelines:
622

• The answer NA means that the paper does not include theoretical results.
623

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
624

referenced.
625

• All assumptions should be clearly stated or referenced in the statement of any theorems.
626

• The proofs can either appear in the main paper or the supplemental material, but if
627

they appear in the supplemental material, the authors are encouraged to provide a short
628

proof sketch to provide intuition.
629

• Inversely, any informal proof provided in the core of the paper should be complemented
630

by formal proofs provided in appendix or supplemental material.
631

• Theorems and Lemmas that the proof relies upon should be properly referenced.
632

4. Experimental Result Reproducibility
633

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
634

perimental results of the paper to the extent that it affects the main claims and/or conclusions
635

of the paper (regardless of whether the code and data are provided or not)?
636

Answer: [Yes]
637

Justification: We realease our code to prove reproducibility.
638

Guidelines:
639

• The answer NA means that the paper does not include experiments.
640

• If the paper includes experiments, a No answer to this question will not be perceived
641

well by the reviewers: Making the paper reproducible is important, regardless of
642

whether the code and data are provided or not.
643

• If the contribution is a dataset and/or model, the authors should describe the steps taken
644

to make their results reproducible or verifiable.
645

• Depending on the contribution, reproducibility can be accomplished in various ways.
646

For example, if the contribution is a novel architecture, describing the architecture fully
647

might suffice, or if the contribution is a specific model and empirical evaluation, it may
648

be necessary to either make it possible for others to replicate the model with the same
649

dataset, or provide access to the model. In general. releasing code and data is often
650

one good way to accomplish this, but reproducibility can also be provided via detailed
651

instructions for how to replicate the results, access to a hosted model (e.g., in the case
652

of a large language model), releasing of a model checkpoint, or other means that are
653

appropriate to the research performed.
654

• While NeurIPS does not require releasing code, the conference does require all submis-
655

sions to provide some reasonable avenue for reproducibility, which may depend on the
656

nature of the contribution. For example
657

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
658

to reproduce that algorithm.
659

(b) If the contribution is primarily a new model architecture, the paper should describe
660

the architecture clearly and fully.
661

(c) If the contribution is a new model (e.g., a large language model), then there should
662

either be a way to access this model for reproducing the results or a way to reproduce
663

the model (e.g., with an open-source dataset or instructions for how to construct
664

the dataset).
665

(d) We recognize that reproducibility may be tricky in some cases, in which case
666

authors are welcome to describe the particular way they provide for reproducibility.
667

In the case of closed-source models, it may be that access to the model is limited in
668

some way (e.g., to registered users), but it should be possible for other researchers
669

to have some path to reproducing or verifying the results.
670

5. Open access to data and code
671

Question: Does the paper provide open access to the data and code, with sufficient instruc-
672

tions to faithfully reproduce the main experimental results, as described in supplemental
673

material?
674

19


---Page Break---
Answer: [Yes]
675

Justification: We release our code, and related information can be found at README.md in
676

our code supplemental material.
677

Guidelines:
678

• The answer NA means that paper does not include experiments requiring code.
679

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
680

public/guides/CodeSubmissionPolicy) for more details.
681

• While we encourage the release of code and data, we understand that this might not be
682

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
683

including code, unless this is central to the contribution (e.g., for a new open-source
684

benchmark).
685

• The instructions should contain the exact command and environment needed to run to
686

reproduce the results. See the NeurIPS code and data submission guidelines (https:
687

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
688

• The authors should provide instructions on data access and preparation, including how
689

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
690

• The authors should provide scripts to reproduce all experimental results for the new
691

proposed method and baselines. If only a subset of experiments are reproducible, they
692

should state which ones are omitted from the script and why.
693

• At submission time, to preserve anonymity, the authors should release anonymized
694

versions (if applicable).
695

• Providing as much information as possible in supplemental material (appended to the
696

paper) is recommended, but including URLs to data and code is permitted.
697

6. Experimental Setting/Details
698

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
699

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
700

results?
701

Answer: [Yes]
702

Justification: We specify all the training and test details in section 4.1 and Appendix A.1.
703

Guidelines:
704

• The answer NA means that the paper does not include experiments.
705

• The experimental setting should be presented in the core of the paper to a level of detail
706

that is necessary to appreciate the results and make sense of them.
707

• The full details can be provided either with the code, in appendix, or as supplemental
708

material.
709

7. Experiment Statistical Significance
710

Question: Does the paper report error bars suitably and correctly defined or other appropriate
711

information about the statistical significance of the experiments?
712

Answer: [Yes]
713

Justification: we report error bars in Appendix A.2.6.
714

Guidelines:
715

• The answer NA means that the paper does not include experiments.
716

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
717

dence intervals, or statistical significance tests, at least for the experiments that support
718

the main claims of the paper.
719

• The factors of variability that the error bars are capturing should be clearly stated (for
720

example, train/test split, initialization, random drawing of some parameter, or overall
721

run with given experimental conditions).
722

• The method for calculating the error bars should be explained (closed form formula,
723

call to a library function, bootstrap, etc.)
724

• The assumptions made should be given (e.g., Normally distributed errors).
725

20


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error
726

of the mean.
727

• It is OK to report 1-sigma error bars, but one should state it. The authors should
728

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
729

of Normality of errors is not verified.
730

• For asymmetric distributions, the authors should be careful not to show in tables or
731

figures symmetric error bars that would yield results that are out of range (e.g. negative
732

error rates).
733

• If error bars are reported in tables or plots, The authors should explain in the text how
734

they were calculated and reference the corresponding figures or tables in the text.
735

8. Experiments Compute Resources
736

Question: For each experiment, does the paper provide sufficient information on the com-
737

puter resources (type of compute workers, memory, time of execution) needed to reproduce
738

the experiments?
739

Answer: [Yes]
740

Justification: We provide sufficient information on the computer resources in Appendix A.1.
741

Guidelines:
742

• The answer NA means that the paper does not include experiments.
743

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
744

or cloud provider, including relevant memory and storage.
745

• The paper should provide the amount of compute required for each of the individual
746

experimental runs as well as estimate the total compute.
747

• The paper should disclose whether the full research project required more compute
748

than the experiments reported in the paper (e.g., preliminary or failed experiments that
749

didn’t make it into the paper).
750

9. Code Of Ethics
751

Question: Does the research conducted in the paper conform, in every respect, with the
752

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
753

Answer: [Yes]
754

Justification: We have read the NeurIPS Code of Ethics, and conduct research with it.
755

Guidelines:
756

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
757

• If the authors answer No, they should explain the special circumstances that require a
758

deviation from the Code of Ethics.
759

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
760

eration due to laws or regulations in their jurisdiction).
761

10. Broader Impacts
762

Question: Does the paper discuss both potential positive societal impacts and negative
763

societal impacts of the work performed?
764

Answer: [Yes]
765

Justification: We discuss the societal impacts in Appendix A.3.
766

Guidelines:
767

• The answer NA means that there is no societal impact of the work performed.
768

• If the authors answer NA or No, they should explain why their work has no societal
769

impact or why the paper does not address societal impact.
770

• Examples of negative societal impacts include potential malicious or unintended uses
771

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
772

(e.g., deployment of technologies that could make decisions that unfairly impact specific
773

groups), privacy considerations, and security considerations.
774

21


---Page Break---
• The conference expects that many papers will be foundational research and not tied
775

to particular applications, let alone deployments. However, if there is a direct path to
776

any negative applications, the authors should point it out. For example, it is legitimate
777

to point out that an improvement in the quality of generative models could be used to
778

generate deepfakes for disinformation. On the other hand, it is not needed to point out
779

that a generic algorithm for optimizing neural networks could enable people to train
780

models that generate Deepfakes faster.
781

• The authors should consider possible harms that could arise when the technology is
782

being used as intended and functioning correctly, harms that could arise when the
783

technology is being used as intended but gives incorrect results, and harms following
784

from (intentional or unintentional) misuse of the technology.
785

• If there are negative societal impacts, the authors could also discuss possible mitigation
786

strategies (e.g., gated release of models, providing defenses in addition to attacks,
787

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
788

feedback over time, improving the efficiency and accessibility of ML).
789

11. Safeguards
790

Question: Does the paper describe safeguards that have been put in place for responsible
791

release of data or models that have a high risk for misuse (e.g., pretrained language models,
792

image generators, or scraped datasets)?
793

Answer: [NA]
794

Justification: The paper poses no such risks.
795

Guidelines:
796

• The answer NA means that the paper poses no such risks.
797

• Released models that have a high risk for misuse or dual-use should be released with
798

necessary safeguards to allow for controlled use of the model, for example by requiring
799

that users adhere to usage guidelines or restrictions to access the model or implementing
800

safety filters.
801

• Datasets that have been scraped from the Internet could pose safety risks. The authors
802

should describe how they avoided releasing unsafe images.
803

• We recognize that providing effective safeguards is challenging, and many papers do
804

not require this, but we encourage authors to take this into account and make a best
805

faith effort.
806

12. Licenses for existing assets
807

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
808

the paper, properly credited and are the license and terms of use explicitly mentioned and
809

properly respected?
810

Answer: [Yes]
811

Justification: All existing assets we use are cited in section 4
812

Guidelines:
813

• The answer NA means that the paper does not use existing assets.
814

• The authors should cite the original paper that produced the code package or dataset.
815

• The authors should state which version of the asset is used and, if possible, include a
816

URL.
817

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
818

• For scraped data from a particular source (e.g., website), the copyright and terms of
819

service of that source should be provided.
820

• If assets are released, the license, copyright information, and terms of use in the
821

package should be provided. For popular datasets, paperswithcode.com/datasets
822

has curated licenses for some datasets. Their licensing guide can help determine the
823

license of a dataset.
824

• For existing datasets that are re-packaged, both the original license and the license of
825

the derived asset (if it has changed) should be provided.
826

22


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
827

the asset’s creators.
828

13. New Assets
829

Question: Are new assets introduced in the paper well documented and is the documentation
830

provided alongside the assets?
831

Answer: [NA]
832

Justification: The paper does not release new assets.
833

Guidelines:
834

• The answer NA means that the paper does not release new assets.
835

• Researchers should communicate the details of the dataset/code/model as part of their
836

submissions via structured templates. This includes details about training, license,
837

limitations, etc.
838

• The paper should discuss whether and how consent was obtained from people whose
839

asset is used.
840

• At submission time, remember to anonymize your assets (if applicable). You can either
841

create an anonymized URL or include an anonymized zip file.
842

14. Crowdsourcing and Research with Human Subjects
843

Question: For crowdsourcing experiments and research with human subjects, does the paper
844

include the full text of instructions given to participants and screenshots, if applicable, as
845

well as details about compensation (if any)?
846

Answer: [NA]
847

Justification: The paper does not involve crowdsourcing nor research with human subjects.
848

Guidelines:
849

• The answer NA means that the paper does not involve crowdsourcing nor research with
850

human subjects.
851

• Including this information in the supplemental material is fine, but if the main contribu-
852

tion of the paper involves human subjects, then as much detail as possible should be
853

included in the main paper.
854

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
855

or other labor should be paid at least the minimum wage in the country of the data
856

collector.
857

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
858

Subjects
859

Question: Does the paper describe potential risks incurred by study participants, whether
860

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
861

approvals (or an equivalent approval/review based on the requirements of your country or
862

institution) were obtained?
863

Answer: [NA]
864

Justification: The paper does not involve crowdsourcing nor research with human subjects.
865

Guidelines:
866

• The answer NA means that the paper does not involve crowdsourcing nor research with
867

human subjects.
868

• Depending on the country in which research is conducted, IRB approval (or equivalent)
869

may be required for any human subjects research. If you obtained IRB approval, you
870

should clearly state this in the paper.
871

• We recognize that the procedures for this may vary significantly between institutions
872

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
873

guidelines for their institution.
874

• For initial submissions, do not include any information that would break anonymity (if
875

applicable), such as the institution conducting the review.
876

23


---Page Break---
