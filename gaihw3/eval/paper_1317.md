Scalable Ensemble Diversification
for OOD Generalization and Detection

Anonymous Author(s)
Affiliation
Address
email

Abstract

Training a diverse ensemble of models has several practical application scenarios,
1

such as model selection for out-of-distribution (OOD) generalization and the
2

detection of OOD samples via Bayesian principles. Previous approaches to diverse
3

ensemble training have relied on the framework of letting the models make the
4

correct predictions for the given in-distribution (ID) data while letting them come up
5

with different hypotheses for the OOD data. As such, they require well-separated
6

ID and OOD datasets to ensure a performant and diverse ensemble and have
7

only been verified in smaller-scale lab environments where such a separation is
8

readily available. In this work, we propose a framework, Scalable Ensemble
9

Diversification (SED), for scaling up existing diversification methods to large-scale
10

datasets and tasks (e.g. ImageNet), where the ID-OOD separation may not be
11

available. SED automatically identifies OOD samples within the large-scale ID
12

dataset on the fly and encourages the ensemble to make diverse hypotheses on
13

them. To make SED more suitable for large-scale applications, we propose an
14

algorithm to speed up the expensive pairwise disagreement computation. We verify
15

the resulting diversification of the ensemble on ImageNet and demonstrate the
16

benefit of diversification on the OOD generalization and OOD detection tasks.
17

In particular, for OOD detection, we propose a novel uncertainty score estimator
18

based on the diversity of ensemble hypotheses, which lets SED surpass all the
19

considered baselines in OOD detection task. Code will be available soon.
20

1
Introduction
21

Training a diverse ensemble of models is useful in multiple applications. Diverse ensembles are used
22

to enhance out-of-distribution (OOD) generalization, where strong spurious features learned from
23

the in-distribution (ID) training data hinder generalization [30, 31, 28, 23]. By learning multiple
24

hypotheses, the ensemble is given a chance to learn causal features that are otherwise overshadowed
25

by the prominent spurious features [39, 4]. In Bayesian machine learning, diversification of the
26

posterior samples has been studied as a means to improve the precision and efficiency of sample
27

uncertainty estimates [5, 37].
28

A common strategy to train a diverse ensemble is to introduce two objectives: one for the main
29

task and one for diversification [29, 5, 28, 23]. The main task loss, such as the cross-entropy loss
30

for classification, encourages the hypotheses to solve the task on the labeled ID training set. The
31

diversification loss encourages the hypotheses to diversify the responses on an unlabelled OOD
32

dataset [28, 23] (Figure 1). The datasets for the objectives are separated to avoid contradictory
33

objectives: prediction diversification on the ID set will encourage wrong answers if there is only one
34

correct label.
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
This strategy, however, requires a separate OOD dataset where the hypotheses may make diverse
36

predictions without harming the main task performance on the ID training samples. Previous work
37

has thus been tested on hypothetical lab settings where the spurious and causal features can easily be
38

controlled to secure separate ID and OOD datasets for diverse ensemble training. It is not clear yet
39

how one could diversify an ensemble of models for realistic, uncontrolled, and large-scale applications
40

(e.g. ImageNet scale) where collecting a separate OOD dataset can be very costly, if not impossible.
41

Existing work assumes existence of ID and OOD datasets.

Our Scalable Ensemble Diversiﬁcation (SED) only requires a single ID dataset.

Diverse model ensemble

Classify

2 datasets
training objectives

Disagree

ID dataset

OOD dataset

Diverse model ensemble
1 dataset
training objectives

Classify

Disagree

Identify OOD samples

ID dataset

OOD samples

Figure 1: Existing diversification work vs SED. Unlike previous
diversification approaches that require a separate OOD dataset on
which the models are trained to diverge, our Scalable Ensemble
Diversification (SED) operates on a single ID dataset where OOD
samples are dynamically identified and are used to let the ensemble
members diverge.

To address the scalability challenge,
42

we propose a novel diversification
43

framework, Scalable Ensemble Diver-
44

sification (SED, Figure 1). We intro-
45

duce three ingredients. (1) OOD sam-
46

ples are dynamically selected from the
47

ID training samples, on which the mod-
48

els are trained to make different predic-
49

tions. (2) At each iteration, a subset of
50

model pairs are stochastically selected
51

to construct the disagreement objec-
52

tive, rather than the full list of model
53

pairs. (3) Deep networks are trained
54

to diversify only a few layers at the
55

end, rather than the full networks. This
56

framework allows scaling up existing
57

ensemble diversification methods. In
58

this work, we focus on scaling up the
59

Agree to Disagree (A2D) method [28].
60

We verify that SEDdiversifies a model
61

ensemble trained on ImageNet. We
62

demonstrate the benefit of diversifica-
63

tion on OOD generalization and OOD
64

detection tasks. For the former, we showcase the usage of SED-diversified ensemble in three variants:
65

(a) vanilla ensemble of prediction probabilities [22], (b) an average of the model weights through
66

model soup [38], and (c) the oracle selection of the individual models for each OOD test set [23, 30].
67

In all three cases, SEDachieves a superior generalization to OOD datasets like ImageNet-A/R/C,
68

OpenImages, and iNaturalist.
69

For OOD detection, we seek multiple ways to use the SED-diversified ensemble: (a) treating them as
70

samples of the Bayesian posterior and (b) using our novel OODness estimate of Predictive Diversity
71

Score (PDS) that measures the diversity of predictions from an ensemble. We show that PDS provides
72

a superior detection of OOD samples like ImageNet-A/R/C, OpenImages, and iNaturalist.
73

Our contributions are
74

1. Scalable Ensemble Diversification (SED) framework that scales up existing ensemble
75

methods;
76

2. Predictive Diversity Score (PDS) that computes the OODness score for samples based on
77

ensemble prediction diversity;
78

3. First demonstration of the ensemble diversification and its application to OOD generalization
79

and detection at ImageNet level.
80

The code will be released with the next versions of the manuscript.
81

2
Related work
82

In this section, we give a short overview of ensembling methods. At first, we speak about ensembles
83

in general and the role of diversity in them (§ 2.1), then we focus on ensembling methods for neural
84

networks and separate them into two big groups. The first group includes algorithms that use loss
85

regularizers (§ 2.2) and the second group covers works that do not modify the training loss (§ 2.3).
86

2


---Page Break---
2.1
Ensembles as a technique
87

Ensembling is a powerful technique of aggregating the outputs of multiple models to make more
88

accurate predictions and it has been around for decades [12, 21, 18, 2, 3]. It is well known that
89

diversity in ensemble members’ outputs leads to better performance of the ensemble compared to the
90

performance of a single model [21] because ensemble members make independent errors [12, 11].
91

Therefore, one way to reduce DNNs’ reliance on spurious correlations is to train multiple models
92

on the same task and make them diverse in terms of errors they make so that their ensemble is less
93

dependent on such correlations.
94

2.2
Neural network ensembles that promote diversity through loss regularizers
95

Diversity in models can be induced by supplying training loss with a suitable regularizer.
96

Such regularizers can diversify models’ weights [5, 7, 34, 6], features [39, 4], input gradients
97

[29, 30, 31, 33] and outputs [25, 5, 28, 23].
98

Notably, in [5] authors showed that regularizer of a certain structure that repulses ensemble members’
99

weights or outputs leads to ensembles that provide a better approximation of Bayesian Model
100

Averaging. This idea was later extended by works that repulse ensemble members’ features [39] and
101

input gradients [33].
102

Since the ensemble performs better due to the diversity of errors that ensemble members make
103

[21] we want those members to give pairwise different outputs for the same inputs. Unfortunately,
104

diversity in weights space, input gradient space, or features space does not guarantee such property
105

without additional assumptions due to functional symmetry which means that models can be different
106

in terms of their weights or feature maps and input gradients they produce but still give the same
107

outputs for a given input. That is why we are focused on methods that diversify models’ outputs,
108

specifically [28, 23] which are state-of-the-art according to [1] and use regularizer of repulsive nature
109

conceptually similar to [5].
110

2.3
Neural network ensembles that promote diversity without modifying loss
111

In addition to loss regularizers, there were an uncountable number of different ways to induce diversity
112

in ensembles of neural networks that did not modify the training loss. The most straightforward
113

approach of independently training multiple models of the same architecture by changing only random
114

seeds is called Deep Ensemble [22] which was extended from the Bayesian perspective in [37].
115

Another solution is to construct an ensemble from models trained with different hyperparameters [36],
116

augmentations [24], or architectures [40]. More computationally efficient direction allows training
117

only one base model inducing diversity by ensembling either checkpoints saved in different local
118

minima along the training trajectory of this base model [19] or models produced by the base model
119

after applying dropout [10] or masking [9] to it. The mixture of experts paradigm can also be viewed
120

as an ensemble diversification technique [41] where diversification happens due to assigning different
121

training samples to different ensemble members.
122

Despite their conceptual simplicity Deep Ensembles [22] and ensembles of models trained with
123

different hyperparameters [36] are strong baselines for OOD detection [27] and OOD generalization
124

tasks, especially when combined with model souping techniques [38]. That is why we selected them
125

as baselines for our experiments.
126

3
Method
127

We present our main technical contributions, Scalable Ensemble Diversification (SED, §3.2) and the
128

Predictive Diversity Score (PDS, §3.3).
129

3.1
Preliminaries
130

We cover background materials before introducing our main technical contributions. We work with
131

a training set D := {xn, yn}N
n=1, which we refer to as the in-distribution (ID) dataset. For prior
132

diversification methods, we also assume the existence of a separate, unlabeled out-of-distribution
133

3


---Page Break---
(OOD) dataset Dood := {xood
n }Nood
n=1. We write f(·, θ) for a deep neural network classifier parametrized
134

by θ. f (x; θ) ∈RC indicates the logit outputs for C classes for input x. We write p(x) :=
135

Softmax(f(x)) ∈[0, 1]C for the probability outputs. We consider an ensemble {f 1, · · · , f M} of M
136

models.
137

3.1.1
Existing ensemble diversification approach
138

We introduce an existing approach for diversifying an ensemble of models [28, 23]. Two objectives
139

are imposed upon the ensemble of models: the main task loss and the diversification regularization.
140

For the main task, the community has focused on the classification task. The cross-entropy loss
141

−log py(x; θ) is used to train the model ensemble {f 1, · · · , f M} on the ID dataset D:
142

Lmain =
1
MN

X

n

X

m
−log pm
yn(xn; θ).
(1)

This encourages each member of the ensemble to behave similarly on the ID dataset.
143

Different diversification schemes use different diversification regularization loss Ldiv applied on pairs
144

(f m, f l) of ensemble members. The diversification objective is commonly optimized on the OOD
145

dataset Dood to encourage the training of multiple hypotheses on the OOD samples while avoiding
146

clashes with the main task objective. In this work, we focus on the Agree to Disagree [28] method.
147

The diversification loss for a pair (pm, pl) is defined as:
148

A2D(pm(x), pl(x)) = −log

pm
ˆy (x) · (1 −pl
ˆy(x)) + (1 −pm
ˆy (x)) · pl
ˆy(x)

(2)

where ˆy := arg maxc pm
c (x) is the predicted class for the first model pm. One may symmetrically
149

define ˆy to be the prediction for the second model pl; in practice, it does not make a difference [28].
150

Note that the diversification loss favors pl to predict a lower likelihood for the prediction by pm,
151

pl
ˆy(x), and vice versa. For M models in an ensemble, A2D is applied on the OOD dataset Dood for
152

every pair of models (pm, pl):
153

Ldiv =
1
N ood · M(M −1)

X

n

X

m<l
A2D(pm(xood
n ), pl(xood
n )).
(3)

3.2
Scalable Ensemble Diversification (SED)
154

We present Scalable Ensemble Diversification (SED) that addresses the limitation of the existing
155

ensemble diversification framework that requires a separate OOD dataset. We introduce two main
156

components of SED: dynamic selection of OOD samples within the ID dataset (§3.2.1) and the
157

stochastic selection of pairs to diverge in the optimization iterations (§3.2.2).
158

3.2.1
Dynamic selection of OOD samples
159

If only the ID training dataset is present, it is difficult to induce diversity in ensemble members,
160

as they are uniformly incentivized to solve the main task objective: given x, predict y. Hence,
161

previous approaches have introduced a qualitatively disjoint unlabeled set, which we refer to as
162

the OOD dataset, where the ensemble members are encouraged to disagree with each other. The
163

clear separation of ID and OOD datasets for the two objectives matters for ensuring a good balance
164

between the main task performance and the diversity of hypotheses.
165

Previous works like Pagliardini et al. [28], Lee et al. [23] have performed experiments on small-scale
166

datasets where factors are well-controlled and clean versions of OOD datasets are readily available.
167

Examples include Waterbirds, Camelyon17, CelebA, MultiNLI, C-MNIST, and the Office-Home
168

datasets. For example, for Waterbirds, the ID dataset is set as the cases where the bird’s habitat
169

matches with the visual background and the OOD dataset corresponds to the complementary case.
170

While conceptually desirable, collecting a separate OOD dataset can be highly cumbersome and
171

expensive. For a large-scale dataset like ImageNet, it is highly non-obvious how one could build a
172

corresponding OOD dataset where the underlying feature-label correlations are different from the ID
173

training dataset.
174

To address this challenge, we consider dynamically identifying an OOD subset of the ID dataset and
175

letting the ensemble diverge on this subset. The desiderata for the identification of OOD samples
176

4


---Page Break---
within the ID dataset are twofold: (a) we wish to discriminate samples where the ensemble members
177

make mistakes and (b) we only trust the ensemble prediction for the OOD sample identification when
178

the ensemble is sufficiently trained.
179

We define the sample-wise weight αn on each ID sample (xn, yn) ∈D that satisfy the two conditions:
180

αn :=
CE(f 1, · · · , f M; xn, yn)

1
|B|
P

b∈B CE(f 1, · · · , f M; xb, yb)
2
(4)

where CE(f 1, · · · , f M; xn, yn) := CE( 1

M
P

m f m(xn), yn) is the loss on the logit-averaged pre-
181

diction and B is a minibatch that contains the sample (xn, yn). αn is a weight proportional to the
182

ensemble loss on the sample; we thus meet the condition (a). The normalization is designed to handle
183

the condition (b). To see this, consider the batch-wise weight
184

αB :=
1
|B|

X

b∈B
αb =
1
1
|B|
P

b CE(f 1, · · · , f M; xb, yb).
(5)

Note that αB is now inversely proportional to the average cross-entropy loss of the ensemble on
185

the batch B. Thus, the overall level of αn for n ∈B is lower for earlier iterations of the ensemble
186

training, where the predictions from the models are not trustworthy yet.
187

With this definition of sample-wise weight αn for the diversification objective, we define the SED
188

objective with the A2D loss for the diversification kernel:
189

LSED := Lmain +
λ
NM(M −1)

X

n

X

m<l
stopgrad(αn) · A2D(pm(xn), pl(xn)),
(6)

where λ > 0 controls the overall weight of the diversification term. Note that, compared to Equation
190

3, this formulation does not rely on the OOD dataset Dood. Instead, all ID samples are treated as
191

potential OOD samples, where their OODness is softly determined via αn. This enables a seamless
192

adaptation of existing ensemble diversification methods to a relaxed setting where a separate OOD
193

dataset is unavailable.
194

3.2.2
Further tricks for scalability
195

Model 1

Iteration K: 
randomly select 1 and 3

Input

Forward pass

Model 2

Model 3

Model 1

Iteration K + 1: 
randomly select 1 and 2

Model 2

Model 3

Ensemble diversification algorithms are often based on pairwise
196

similarities of the members. Pairwise similarity computation scales
197

quadratically with the size of the ensemble M. The second term of
198

Equation 6 is an example of this. This is potentially a hurdle when
199

ensemble diversification is to be applied to M ≥10, and the data
200

and parameter sizes are in the order of millions (e.g. ImageNet).
201

We address this computational challenge by computing the summa-
202

tion of pairwise distances as a stochastic sum. For every minibatch B
203

of SGD iterations, we uniformly-iid sample a subset I of {1, · · · , M}
204

to compute the diversification term in Equation 6. The procedure is
205

illustrated in the figure on the right.
206

To further speed up the SED training, we consider diversifying only
207

a subset of layers, while freezing the other layers. In our experiments,
208

ensemble members share the same frozen feature extractor of Deit3b
209

[32] pretrained on ImageNet-21k [8] and we diversify only the last
210

two layers of the models.
211

3.3
Predictive Diversity Score (PDS) for OOD Detection
212

We demonstrate several benefits of the diversified ensembles in §4. One of them is the possibility of
213

using them for detecting OOD samples through the notion of epistemic uncertainty [13]. Given an
214

ensemble of models, a simple baseline for OOD detection is to compute the predictive uncertainty of
215

the Bayesian Model Averaging (BMA) by treating the ensemble members as samples of the posterior
216

p(θ|D) [22, 37]:
217

ηBMA := max
c
1
M

X

m
pm
c (x).
(7)

5


---Page Break---
This notion of epistemic uncertainty does not directly exploit the potential diversity in individual
218

models of the ensemble because it averages out the predictions along the model index m.
219

We propose a novel measure for epistemic uncertainty, Predictive Diversity Score (PDS), that directly
220

measures the prediction diversity of the individual members. The formulation is given below:
221

ηPDS := 1

C

X

c
max
m pm
c (x).
(8)

PDS is a continuous relaxation of the number of unique argmax predictions within an ensemble
222

of models. To see this, consider the special case where pm ∈{0, 1} are one-hot vectors. Then,
223

maxm pm
c (x) is 1 if any of m predicts c and 0 otherwise. Thus, P

c maxm pm
c (x) computes the
224

number of classes that at least one of the ensemble members predicts. We show that, with our diverse
225

ensembles, PDS outperforms the DE baseline for the OOD detection task (§4.4).
226

4
Experiments
227

We verify our contributions, Scalable Ensemble Diversification (SED, §3.2) and Predictive Diversity
228

Score (PDS, §3.3), on ImageNet-scale tasks and datasets. We first verify that SED diversifies the
229

ensemble (§4.2). Then, we demonstrate the application of diversified ensemble to OOD generalization
230

(§4.3) and OOD detection (§4.4) tasks.
231

4.1
Experimental setup
232

We task the ensemble with the OOD generalization and OOD detection tasks.
233

Training settings. For both tasks, we train an ensemble of models with the SED framework with
234

the A2D [28] diversity regularization using AdamW optimizer [26]. We use the default settings of a
235

batch size of 16, learning rate 10−3, weight decay 0.01, and the number of epochs 10. The overall
236

diversity weight λ is set to 0.1 and the stochastic pairing is done for |I| = 2 models for each SGD
237

batch. We use Deit3b [32] network pretrained on ImageNet21k [8] for all the experiments. Following
238

the speed-up trick in §3.2.2, we use only the last 2 layers of the network. For the in-distribution
239

(ID) dataset where the ensemble is trained to diversify, we use the training split of ImageNet with
240

|D| = 1, 281, 167. All experiments were ran on RTX2080Ti GPUs with 12GB vRAM and 40GB
241

RAM, each experiment took from 2 to 12 hours depending on the complexity of the training.
242

Baselines. For naive ensemble training, we consider the deep ensemble [22] where each ensemble
243

member independently with different random seeds that control the weight initialization and SGD
244

batch shuffling. To match the resource usage of our SED, where we diversify only the last 2 layers
245

of the network, we consider the shallow ensemble variant, which is the deep ensemble where only
246

the last 2 layers are trained. We further consider a viable diversification scheme that performs deep
247

ensemble with varying hyperparameters [36]. In addition to that, we reimplement A2D [28] and
248

DivDis [23] algorithms and apply them without stochastic model sampling to do classification on
249

labeled samples from ImageNet-Train and disagreement on unlabeled samples from ImageNet-R.
250

For A2D we use frozen feature extractor and a parallel variant of their method which means that all
251

ensemble members are trained simultaneously and not sequentially. The computational complexity
252

of both these approaches scales quadratically with ensemble size which is why they are called Naive
253

A2D and Naive DivDis respectively.
254

Evaluation benchmarks. The generalization performances of the ImageNet-trained ensembles are
255

measured on multiple test datasets, ranging from the in-distribution validation split of ImageNet with
256

50,000 samples to OOD datasets like ImageNet-A (A [17], 7.5k images & 200 classes), ImageNet-R
257

(A [16], 30k images, 200 classes), ImageNet-C (C-i for corruption strength i [14], 50k images, 1k
258

classes). OpenImages-O (OI [35], 17k images, unlabeled), and iNaturalist (iNat [20], 10k images,
259

unlabeled). For OOD detection, we task the ensemble with the detection of the above OOD datasets
260

against the ImageNet validation split.
261

Evaluation metrics. For OOD generalization, we use the accuracy. For OOD detection, we use the
262

area under the ROC curve, following [15].
263

6


---Page Break---
GT
Cowboy hat
Sea lion
Scuba diver
Great shark
Weimaraner

SED
Cowboy hat
Sea lion
Scuba diver
Great shark
Weimaraner
Comic book
Otter
Jellyfish
Killer whale
Vizsla
PDS
0.300
0.300
0.294
0.292
0.292

GT
Pomegranate
Zebra
Pomegranate
Pomegranate
Hummingbird

SED
Pomegranate
Zebra
Pomegranate
Pomegranate
Hummingbird
PDS
0.216
0.216
0.216
0.216
0.216

Figure 2: ImageNet-R examples leading to the greatest and least disagreement. We show the 5 most
divergent and 5 least divergent samples according to the SED ensemble. We measure the prediction diversity
with the Prediction Diversity Score (PDS) in §3.3. GT refers to the ground truth category. Ensemble predictions
are shown in bold, in cases when ensemble members predict classes different from the ensemble prediction we
provide them on the next line with standard font.

4.2
Diversification
264

We start with the question of whether Scalable Ensemble Diversification (SED) truly diversify the
265

ensemble at the ImageNet scale. To measure the diversity of the ensemble, we compute the number
266

of unique predictions for each sample for the committee of models (#unique).
267

Method
C-1
C-5
iNat
OI

Deep ensemble
1.09
1.19
1.31
1.23
+Diverse hyperparams
1.11
1.32
1.48
1.33

Naive DivDis
1.04
1.14
1.19
1.16
Naive A2D
1.04
1.15
1.19
1.91

SED-A2D
5.00
5.00
4.68
4.11

Table 1: #unique for ensembles.
We report the
#unique on OOD datasets (see §4.1 for the datasets).
The ensemble size M is 5 for all methods; it is the max
possible #unique value.

Table 1 shows the #unique values for the IN-Val
268

as well as multiple OOD datasets. We observe
269

that the deep ensemble baseline does not increase
270

the diversity dramatically (e.g. 1.09 for C-1) be-
271

yond no-diversity values (1.0). Diversification
272

tricks like hyperparameter diversification (1.11
273

for C-1) or Naive A2D (1.04 for C-1) and DivDis
274

(1.04 for C-1) do not improve the prediction di-
275

versity dramatically. On the other hand, our SED
276

increases the prediction diversity across the board
277

(e.g. 5.00 for C-1).
278

Qualitative results on ImageNet-R further verify the ability of SED to diversify the ensemble (Fig-
279

ure 2). As a measure for diversity, we use the Predictive Diversity Score (PDS) in §3.3. We observe
280

that the samples inducing the highest diversity (high PDS scores) are indeed ambiguous: for the
281

first image, where the “cowboy hat” is the ground truth category, we observe that “comic book” is
282

also a valid label for the image style. On the other hand, samples with low PDS exhibit clearer
283

image-to-category relationship.
284

4.3
OOD Generalization
285

We examine the first application of diversified ensembles: OOD generalization. We hypothesize that
286

the superior diversification ability verified in §4.2 leads to greater OOD generalization due to the
287

consideration of more robust hypotheses that do not rely on obvious spurious correlations.
288

Ensemble aggregation for OOD generalization. As a means to exploit such robust hypothe-
289

ses, we consider 3 aggregation strategies.
(1) Oracle selection: the best-performing individ-
290

ual model is chosen from an ensemble [28, 30]. Final prediction is given by f(x; θm⋆) where
291

7


---Page Break---
Oracle selection
Prediction ensemble
Uniform soup

Method
M
Val IN-A IN-R C-1
C-5
Val IN-A IN-R C-1
C-5
Val IN-A IN-R C-1
C-5

Single model
1
85.4 37.9
44.7 75.6 38.5 85.4 37.9
44.7 75.6 38.5 85.4 37.9
44.7 75.6 38.5

Deep ensemble
5
85.4 37.9
44.9 75.7 38.6 85.4 39.9
46.3 75.7 38.6 85.3 36.7
44.6 75.5 38.3
+Diverse HPs
5
85.4 38.5
45.4 77.4 40.7 85.4 39.9
46.5 76.0 39.0 85.3 35.3
44.1 75.9 38.7
Naive DivDis
5
85.2 35.8
40.8 77.2 40.2 85.1 36.3
41.8 77.2 40.2 84.8 40.7
42.5 76.2 38.9
Naive A2D
5
85.2 36.6
44.3 77.3 40.4 85.1 37.8
45.2 77.2 40.3 84.5 39.3
45.1 75.5 39.1
SED-A2D
5
85.1 38.3
45.3 77.2 40.4 85.3 42.4
48.1 77.3 40.6 85.3 40.3
46.1 77.3 40.6

Deep ensemble 50 85.5 38.1
45.2 75.7 38.6 85.5 38.8
45.8 75.6 38.5 85.4 37.5
45.0 75.5 38.4
+Diverse HPs
50 85.5 38.5
45.6 77.5 40.8 85.5 42.5
48.5 76.0 39.0 85.4 36.4
44.8 75.9 38.8
SED-A2D
50 82.6 39.0
45.8 74.4 38.3 83.5 50.9
54.4 75.8 39.3 83.5 39.2
46.5 75.8 39.3

Table 2: OOD generalization of ensembles. Models are trained on the ImageNet training split. M is the
ensemble size. For Naive DivDis and A2D, we use the ImageNet-R as the OOD datasets where the respective
diversification objectives are applied.

m⋆:= arg maxm Acc(f m, Dood). (2) Prediction ensemble is a vanilla prediction ensemble where
292

the logit values are averaged:
1
M
P

m f m(x) [38]. (3) Uniform soup [38] averages the weights
293

themselves. Final prediction is given by f(x; 1

M
P
m θm).
294

SED improves OOD generalization for ensembles. We show the OOD generalization performances
295

of ensembles in Table 2, for the three ensemble prediction aggregation strategies described above. We
296

observe that our SED framework (SED-A2D) results in superior OOD generalization performances
297

for all three strategies. SED-A2D is particularly strong in prediction ensemble (e.g. 48.1% for M = 5
298

and 54.4% for M = 50 on ImageNet-R) and uniform soup (e.g. 46.1% for M = 5 and 46.5%
299

for M = 50 on ImageNet-R). We contend that the increased ensemble diversity contributes to the
300

improvements in OOD generalization. We also remark that the SED framework (SED-A2D) envelops
301

the performance of Naive A2D in this ImageNet-scale experiment. Together with the superiority of
302

computational efficiency (as discussed at the end of § 4.4) of SED-A2D over the Naive A2D, this
303

demonstrates that SED fulfills its purpose of scaling up ensemble diversification methods like A2D.
304

Deep ensemble is a strong baseline. We also note that deep ensemble, particularly with diverse
305

hyperparameters, provides a strong baseline, outperforming dedicated diversification methodologies
306

under the oracle selection strategy when M = 5. It also provides a good balance between ID
307

(ImageNet validation split) and OOD generalization.
308

4.4
OOD Detection
309

Method
η
C-1
C-5
iNat
OI

Single model
BMA
0.615
0.833
0.958
0.909

Deep Ensemble
BMA
0.619
0.835
0.958
0.911
+Diverse HPs
BMA
0.642
0.861
0.969
0.923
Naive DivDis
BMA
0.598
0.843
0.966
0.922
Naive A2D
BMA
0.594
0.835
0.966
0.916
SED-A2D
BMA
0.641
0.845
0.960
0.915

Deep Ensemble
PDS
0.565
0.625
0.592
0.589
+Diverse HPs
PDS
0.643
0.849
0.926
0.889
Naive DivDis
PDS
0.600
0.851
0.969
0.939
Naive A2D
PDS
0.599
0.850
0.971
0.939
SED-A2D
PDS
0.686
0.896
0.977
0.941

Table 3: OOD detection via ensembles. For each OOD
dataset (C-1, C-5, iNat, and OI), the ensembles are tasked
to detect the respective OOD samples among ID samples
(ImageNet validation split). We show the AUROC scores for
the OOD detection task. Ensemble size is fixed at M = 5.
η refers to the epistemic uncertainty computation framework
discussed in §3.3.

We study the impact of ensemble diversifi-
310

cation on OOD detection capabilities of an
311

ensemble. Once an ensemble is trained, we
312

compute the epistemic uncertainty, or like-
313

lihood of the sample being OOD, following
314

two schemes, ηBMA and ηPDS introduced in
315

§3.3.
316

SED and PDS together lead to superior
317

OOD detection performances. We show
318

the OOD detection results in Table 3. For
319

the BMA scores, deep ensemble remains a
320

strong baseline. In particular, when the hy-
321

perparameters are varied (“+Diverse HPs”),
322

the detection AUROC reaches the maximal
323

performances among the ensembles using
324

the BMA scores. The quality of PDS is
325

more sensitive to the ensemble diversity, as
326

seen in the jump from the deep ensemble
327

(e.g. 0.589 for OI) to the diverse-HP vari-
328

ant (0.889). However, when the ensemble
329

8


---Page Break---
is sufficiently diverse, such as when trained
330

with SED-A2D, the PDS leads to high-quality OODness scores. SED-A2D with PDS achieves the
331

best AUROC across the board, including the BMA variants.
332

Figure 3: Impact of diversity regulariser on OOD detection. We show the model answer diversity, measured
by PDS, and the OOD detection performance, measured by AUROC, against λ values, the loss weight for the
disagreement regularizer term.

Impact of diversification parameter λ. We further study the impact of ensemble diversification
333

on the OOD detection with the PDS estimator. In Figure 3, we observe that strengthening the
334

diversification objective (higher λ) indeed leads to greater diversity (higher PDS), with a jump at
335

around λ ∈[10−1, 101]. This range corresponds to the jump in the OOD detection performance
336

(higher AUROC).
337

Figure 4: Impact of ensemble size on OOD detection.

Influence of ensemble size. How ensemble size
338

influences performance of our method? We can
339

see that increasing ensemble size helps to im-
340

prove AUROC for OOD detection on C-1 (Fig-
341

ure 4).
Increasing ensemble size marginally
342

helps, but using 5 models provides already a
343

significant improvement over the smallest pos-
344

sible ensemble of size 2. It is also important to
345

mention, that SED framework is computationally
346

more efficient w.r.t. ensemble size M than Naive
347

A2D and Naive DivDis: since we train ensembles for the fixed number of epochs, training complexity
348

for SED is O(1) thanks to stochastic model pairs selection, while for Naive A2D and Naive DivDis it
349

is O(M 2).
350

5
Conclusion
351

Ensemble diversification has many implications for treating one of the ultimate goals of machine learn-
352

ing, handling out-of-distribution (OOD) samples. By training a large number of plausible hypotheses
353

on an in-distribution (ID) dataset, an OOD-generalizable hypothesis may appear. Moreover, the
354

diversity of hypotheses lets us distinguish ID samples from OOD samples by measuring the degree of
355

divergence in ensemble members’ predictions. Despite conceptual benefits, diverse-ensemble training
356

has previously remained a lab-bound concept for several reasons. First, previous approaches required
357

a separate OOD dataset that may nurture diverse hypotheses. Second, computational complexities of
358

previous pairwise diversification objectives increase quadratically with the ensemble size.
359

We have addressed the challenges through the novel Scalable Ensemble Diversification (SED)
360

framework. SED identifies the OOD-like samples from a single dataset, bypassing the need to
361

prepare a separate OOD dataset. SED also employs a stochastic pair selection algorithm which
362

reduces the quadratic complexity of previous approaches to a constant cost per SGD iteration. We
363

have demonstrated good performances by SED on the OOD generalization and detection tasks, both
364

at the ImageNet scale, a largely underexplored regime in the ensemble diversification community.
365

In particular, for OOD detection, our novel diversity measure of Predictive Diversity Score (PDS)
366

amplifies the benefits of diverse ensembles for OOD detection. The code to reproduce the results of
367

our experiments will provided with the next revision of the manuscript.
368

Limitations
369

We do not provide theoretical justification for the method. Our experiments were conducted on
370

models with a frozen feature extractor.
371

9


---Page Break---
References
372

[1] H. L. Benoit, L. Jiang, A. Atanov, O. F. Kar, M. Rigotti, and A. Zamir. Unraveling the key compo-
373

nents of OOD generalization via diversification. In The Twelfth International Conference on Learning
374

Representations, 2024. URL https://openreview.net/forum?id=Lvf7GnaLru.
375

[2] L. Breiman. Bagging predictors. Machine Learning, 24(2):123–140, Aug 1996. ISSN 1573-0565. doi:
376

10.1007/BF00058655. URL https://doi.org/10.1007/BF00058655.
377

[3] L. Breiman. Random forests. Machine Learning, 45(1):5–32, Oct 2001. ISSN 1573-0565. doi: 10.1023/A:
378

1010933404324. URL https://doi.org/10.1023/A:1010933404324.
379

[4] A. S. Chen, Y. Lee, A. Setlur, S. Levine, and C. Finn. Project and probe: Sample-efficient domain
380

adaptation by interpolating orthogonal features. arXiv preprint arXiv:2302.05441, 2023.
381

[5] F. D’Angelo and V. Fortuin. Repulsive deep ensembles are bayesian. Advances in Neural Information
382

Processing Systems, 34:3451–3465, 2021.
383

[6] A. de Mathelin, F. Deheeger, M. Mougeot, and N. Vayatis. Maximum weight entropy. arXiv preprint
384

arXiv:2309.15704, 2023.
385

[7] A. de Mathelin, F. Deheeger, M. Mougeot, and N. Vayatis. Deep anti-regularized ensembles provide
386

reliable out-of-distribution uncertainty quantification, 2023.
387

[8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image
388

database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.
389

doi: 10.1109/CVPR.2009.5206848.
390

[9] N. Durasov, T. Bagautdinov, P. Baque, and P. Fua. Masksembles for uncertainty estimation. In Proceedings
391

of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13539–13548, 2021.
392

[10] Y. Gal and Z. Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep
393

learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.
394

[11] I. Goodfellow, Y. Bengio, and A. Courville.
Deep Learning.
MIT Press, 2016.
http://www.
395

deeplearningbook.org.
396

[12] L. Hansen and P. Salamon. Neural network ensembles. IEEE Transactions on Pattern Analysis and
397

Machine Intelligence, 12(10):993–1001, 1990. doi: 10.1109/34.58871.
398

[13] J. C. Helton, J. D. Johnson, and W. L. Oberkampf. An exploration of alternative approaches to the
399

representation of uncertainty in model predictions. Reliability Engineering & System Safety, 85(1-3):
400

39–71, 2004.
401

[14] D. Hendrycks and T. Dietterich.
Benchmarking neural network robustness to common corruptions
402

and perturbations. In International Conference on Learning Representations, 2019. URL https://
403

openreview.net/forum?id=HJz6tiCqYm.
404

[15] D. Hendrycks and K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples
405

in neural networks. In International Conference on Learning Representations, 2017. URL https:
406

//openreview.net/forum?id=Hkg4TI9xl.
407

[16] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo,
408

et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings
409

of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021.
410

[17] D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples. In Proceedings
411

of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021.
412

[18] T. K. Ho. Random decision forests. In Proceedings of 3rd International Conference on Document Analysis
413

and Recognition, volume 1, pages 278–282 vol.1, 1995. doi: 10.1109/ICDAR.1995.598994.
414

[19] G. Huang, Y. Li, G. Pleiss, Z. Liu, J. E. Hopcroft, and K. Q. Weinberger. Snapshot ensembles: Train 1, get
415

m for free. arXiv preprint arXiv:1704.00109, 2017.
416

[20] R. Huang and Y. Li. Mos: Towards scaling out-of-distribution detection for large semantic space. In
417

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8710–8719,
418

2021.
419

[21] A. Krogh and J. Vedelsby.
Neural network ensembles, cross validation, and active learning.
In
420

G. Tesauro, D. Touretzky, and T. Leen, editors, Advances in Neural Information Processing Systems,
421

volume 7. MIT Press, 1994. URL https://proceedings.neurips.cc/paper_files/paper/1994/
422

file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf.
423

[22] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation
424

using deep ensembles. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan,
425

and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran As-
426

sociates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/
427

9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf.
428

10


---Page Break---
[23] Y. Lee, H. Yao, and C. Finn.
Diversify and disambiguate: Out-of-distribution robustness via dis-
429

agreement.
In The Eleventh International Conference on Learning Representations, 2023.
URL
430

https://openreview.net/forum?id=RVTOp3MwT3n.
431

[24] Z. Li, I. Evtimov, A. Gordo, C. Hazirbas, T. Hassner, C. C. Ferrer, C. Xu, and M. Ibrahim. A whac-a-mole
432

dilemma: Shortcuts come in multiples where mitigating one amplifies others. In Proceedings of the
433

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20071–20082, 2023.
434

[25] Y. Liu and X. Yao. Simultaneous training of negatively correlated neural networks in an ensemble. IEEE
435

Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 29(6):716–725, 1999.
436

[26] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on
437

Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.
438

[27] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and
439

J. Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift.
440

Advances in neural information processing systems, 32, 2019.
441

[28] M. Pagliardini, M. Jaggi, F. Fleuret, and S. P. Karimireddy. Agree to disagree: Diversity through disagree-
442

ment for better transferability. In The Eleventh International Conference on Learning Representations,
443

2023. URL https://openreview.net/forum?id=K7CbYQbyYhY.
444

[29] A. Ross, W. Pan, L. Celi, and F. Doshi-Velez. Ensembles of locally independent prediction models. In
445

Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 5527–5536, 2020.
446

[30] D. Teney, E. Abbasnejad, S. Lucey, and A. van den Hengel. Evading the simplicity bias: Training a diverse
447

set of models discovers solutions with superior ood generalization. In Proceedings of the IEEE/CVF
448

Conference on Computer Vision and Pattern Recognition (CVPR), pages 16761–16772, June 2022.
449

[31] D. Teney, M. Peyrard, and E. Abbasnejad. Predicting is not understanding: Recognizing and addressing
450

underspecification in machine learning. In S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner,
451

editors, Computer Vision – ECCV 2022, pages 458–476, Cham, 2022. Springer Nature Switzerland. ISBN
452

978-3-031-20050-2.
453

[32] H. Touvron, M. Cord, and H. Jégou. Deit iii: Revenge of the vit. In European conference on computer
454

vision, pages 516–533. Springer, 2022.
455

[33] T. Trinh, M. Heinonen, L. Acerbi, and S. Kaski. Input-gradient space particle inference for neural network
456

ensembles. In International Conference on Learning Representations, 2024.
457

[34] H. Wang and Q. Ji. Diversity-enhanced probabilistic ensemble for uncertainty estimation. In R. J. Evans
458

and I. Shpitser, editors, Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence,
459

volume 216 of Proceedings of Machine Learning Research, pages 2214–2225. PMLR, 31 Jul–04 Aug
460

2023. URL https://proceedings.mlr.press/v216/wang23c.html.
461

[35] H. Wang, Z. Li, L. Feng, and W. Zhang. Vim: Out-of-distribution with virtual-logit matching. In
462

Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4921–4930,
463

2022.
464

[36] F. Wenzel, J. Snoek, D. Tran, and R. Jenatton. Hyperparameter ensembles for robustness and uncertainty
465

quantification. Advances in Neural Information Processing Systems, 33:6514–6527, 2020.
466

[37] A. G. Wilson and P. Izmailov. Bayesian deep learning and a probabilistic perspective of generalization.
467

Advances in neural information processing systems, 33:4697–4708, 2020.
468

[38] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong,
469

A. Farhadi, Y. Carmon, S. Kornblith, and L. Schmidt. Model soups: averaging weights of multiple
470

fine-tuned models improves accuracy without increasing inference time. In K. Chaudhuri, S. Jegelka,
471

L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, Proceedings of the 39th International Conference
472

on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 23965–23998.
473

PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/wortsman22a.html.
474

[39] S. Yashima, T. Suzuki, K. Ishikawa, I. Sato, and R. Kawakami. Feature space particle inference for neural
475

network ensembles. In International Conference on Machine Learning, pages 25452–25468. PMLR, 2022.
476

[40] S. Zaidi, A. Zela, T. Elsken, C. C. Holmes, F. Hutter, and Y. Teh. Neural ensemble search for uncertainty
477

estimation and dataset shift. Advances in Neural Information Processing Systems, 34:7898–7911, 2021.
478

[41] T. Zhou, S. Wang, and J. A. Bilmes.
Diverse ensemble evolution: Curriculum data-model mar-
479

riage.
In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Gar-
480

nett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Asso-
481

ciates, Inc., 2018.
URL https://proceedings.neurips.cc/paper_files/paper/2018/file/
482

3070e6addcd702cb58de5d7897bfdae1-Paper.pdf.
483

11


---Page Break---
NeurIPS Paper Checklist
484

1. Claims
485

Question: Do the main claims made in the abstract and introduction accurately reflect the
486

paper’s contributions and scope?
487

Answer: [Yes]
488

Justification: Please refer to § 4
489

Guidelines:
490

• The answer NA means that the abstract and introduction do not include the claims
491

made in the paper.
492

• The abstract and/or introduction should clearly state the claims made, including the
493

contributions made in the paper and important assumptions and limitations. A No or
494

NA answer to this question will not be perceived well by the reviewers.
495

• The claims made should match theoretical and experimental results, and reflect how
496

much the results can be expected to generalize to other settings.
497

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
498

are not attained by the paper.
499

2. Limitations
500

Question: Does the paper discuss the limitations of the work performed by the authors?
501

Answer: [Yes]
502

Justification: Please refer to § 5
503

Guidelines:
504

• The answer NA means that the paper has no limitation while the answer No means that
505

the paper has limitations, but those are not discussed in the paper.
506

• The authors are encouraged to create a separate "Limitations" section in their paper.
507

• The paper should point out any strong assumptions and how robust the results are to
508

violations of these assumptions (e.g., independence assumptions, noiseless settings,
509

model well-specification, asymptotic approximations only holding locally). The authors
510

should reflect on how these assumptions might be violated in practice and what the
511

implications would be.
512

• The authors should reflect on the scope of the claims made, e.g., if the approach was
513

only tested on a few datasets or with a few runs. In general, empirical results often
514

depend on implicit assumptions, which should be articulated.
515

• The authors should reflect on the factors that influence the performance of the approach.
516

For example, a facial recognition algorithm may perform poorly when image resolution
517

is low or images are taken in low lighting. Or a speech-to-text system might not be
518

used reliably to provide closed captions for online lectures because it fails to handle
519

technical jargon.
520

• The authors should discuss the computational efficiency of the proposed algorithms
521

and how they scale with dataset size.
522

• If applicable, the authors should discuss possible limitations of their approach to
523

address problems of privacy and fairness.
524

• While the authors might fear that complete honesty about limitations might be used by
525

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
526

limitations that aren’t acknowledged in the paper. The authors should use their best
527

judgment and recognize that individual actions in favor of transparency play an impor-
528

tant role in developing norms that preserve the integrity of the community. Reviewers
529

will be specifically instructed to not penalize honesty concerning limitations.
530

3. Theory Assumptions and Proofs
531

Question: For each theoretical result, does the paper provide the full set of assumptions and
532

a complete (and correct) proof?
533

Answer: [NA]
534

12


---Page Break---
Justification: The paper contains no theoretical results.
535

Guidelines:
536

• The answer NA means that the paper does not include theoretical results.
537

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
538

referenced.
539

• All assumptions should be clearly stated or referenced in the statement of any theorems.
540

• The proofs can either appear in the main paper or the supplemental material, but if
541

they appear in the supplemental material, the authors are encouraged to provide a short
542

proof sketch to provide intuition.
543

• Inversely, any informal proof provided in the core of the paper should be complemented
544

by formal proofs provided in appendix or supplemental material.
545

• Theorems and Lemmas that the proof relies upon should be properly referenced.
546

4. Experimental Result Reproducibility
547

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
548

perimental results of the paper to the extent that it affects the main claims and/or conclusions
549

of the paper (regardless of whether the code and data are provided or not)?
550

Answer: [Yes]
551

Justification: Please refer to § 4
552

Guidelines:
553

• The answer NA means that the paper does not include experiments.
554

• If the paper includes experiments, a No answer to this question will not be perceived
555

well by the reviewers: Making the paper reproducible is important, regardless of
556

whether the code and data are provided or not.
557

• If the contribution is a dataset and/or model, the authors should describe the steps taken
558

to make their results reproducible or verifiable.
559

• Depending on the contribution, reproducibility can be accomplished in various ways.
560

For example, if the contribution is a novel architecture, describing the architecture fully
561

might suffice, or if the contribution is a specific model and empirical evaluation, it may
562

be necessary to either make it possible for others to replicate the model with the same
563

dataset, or provide access to the model. In general. releasing code and data is often
564

one good way to accomplish this, but reproducibility can also be provided via detailed
565

instructions for how to replicate the results, access to a hosted model (e.g., in the case
566

of a large language model), releasing of a model checkpoint, or other means that are
567

appropriate to the research performed.
568

• While NeurIPS does not require releasing code, the conference does require all submis-
569

sions to provide some reasonable avenue for reproducibility, which may depend on the
570

nature of the contribution. For example
571

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
572

to reproduce that algorithm.
573

(b) If the contribution is primarily a new model architecture, the paper should describe
574

the architecture clearly and fully.
575

(c) If the contribution is a new model (e.g., a large language model), then there should
576

either be a way to access this model for reproducing the results or a way to reproduce
577

the model (e.g., with an open-source dataset or instructions for how to construct
578

the dataset).
579

(d) We recognize that reproducibility may be tricky in some cases, in which case
580

authors are welcome to describe the particular way they provide for reproducibility.
581

In the case of closed-source models, it may be that access to the model is limited in
582

some way (e.g., to registered users), but it should be possible for other researchers
583

to have some path to reproducing or verifying the results.
584

5. Open access to data and code
585

Question: Does the paper provide open access to the data and code, with sufficient instruc-
586

tions to faithfully reproduce the main experimental results, as described in supplemental
587

material?
588

13


---Page Break---
Answer: [Yes]
589

Justification: Code will be available soon, please refer to § 4.1.
590

Guidelines:
591

• The answer NA means that paper does not include experiments requiring code.
592

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
593

public/guides/CodeSubmissionPolicy) for more details.
594

• While we encourage the release of code and data, we understand that this might not be
595

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
596

including code, unless this is central to the contribution (e.g., for a new open-source
597

benchmark).
598

• The instructions should contain the exact command and environment needed to run to
599

reproduce the results. See the NeurIPS code and data submission guidelines (https:
600

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
601

• The authors should provide instructions on data access and preparation, including how
602

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
603

• The authors should provide scripts to reproduce all experimental results for the new
604

proposed method and baselines. If only a subset of experiments are reproducible, they
605

should state which ones are omitted from the script and why.
606

• At submission time, to preserve anonymity, the authors should release anonymized
607

versions (if applicable).
608

• Providing as much information as possible in supplemental material (appended to the
609

paper) is recommended, but including URLs to data and code is permitted.
610

6. Experimental Setting/Details
611

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
612

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
613

results?
614

Answer: [Yes]
615

Justification: please refer to § 4.1.
616

Guidelines:
617

• The answer NA means that the paper does not include experiments.
618

• The experimental setting should be presented in the core of the paper to a level of detail
619

that is necessary to appreciate the results and make sense of them.
620

• The full details can be provided either with the code, in appendix, or as supplemental
621

material.
622

7. Experiment Statistical Significance
623

Question: Does the paper report error bars suitably and correctly defined or other appropriate
624

information about the statistical significance of the experiments?
625

Answer: [No]
626

Justification: Error bars are not reported because their magnitude was below the rounding
627

error or roughly around it for the majority of experiments.
628

Guidelines:
629

• The answer NA means that the paper does not include experiments.
630

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
631

dence intervals, or statistical significance tests, at least for the experiments that support
632

the main claims of the paper.
633

• The factors of variability that the error bars are capturing should be clearly stated (for
634

example, train/test split, initialization, random drawing of some parameter, or overall
635

run with given experimental conditions).
636

• The method for calculating the error bars should be explained (closed form formula,
637

call to a library function, bootstrap, etc.)
638

• The assumptions made should be given (e.g., Normally distributed errors).
639

14


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error
640

of the mean.
641

• It is OK to report 1-sigma error bars, but one should state it. The authors should
642

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
643

of Normality of errors is not verified.
644

• For asymmetric distributions, the authors should be careful not to show in tables or
645

figures symmetric error bars that would yield results that are out of range (e.g. negative
646

error rates).
647

• If error bars are reported in tables or plots, The authors should explain in the text how
648

they were calculated and reference the corresponding figures or tables in the text.
649

8. Experiments Compute Resources
650

Question: For each experiment, does the paper provide sufficient information on the com-
651

puter resources (type of compute workers, memory, time of execution) needed to reproduce
652

the experiments?
653

Answer: [Yes]
654

Justification: please refer to § 4.1.
655

Guidelines:
656

• The answer NA means that the paper does not include experiments.
657

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
658

or cloud provider, including relevant memory and storage.
659

• The paper should provide the amount of compute required for each of the individual
660

experimental runs as well as estimate the total compute.
661

• The paper should disclose whether the full research project required more compute
662

than the experiments reported in the paper (e.g., preliminary or failed experiments that
663

didn’t make it into the paper).
664

9. Code Of Ethics
665

Question: Does the research conducted in the paper conform, in every respect, with the
666

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
667

Answer: [Yes]
668

Justification: we followed the Code to the best of our knowledge.
669

Guidelines:
670

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
671

• If the authors answer No, they should explain the special circumstances that require a
672

deviation from the Code of Ethics.
673

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
674

eration due to laws or regulations in their jurisdiction).
675

10. Broader Impacts
676

Question: Does the paper discuss both potential positive societal impacts and negative
677

societal impacts of the work performed?
678

Answer: [NA]
679

Justification: We believe that this work has no societal impact.
680

Guidelines:
681

• The answer NA means that there is no societal impact of the work performed.
682

• If the authors answer NA or No, they should explain why their work has no societal
683

impact or why the paper does not address societal impact.
684

• Examples of negative societal impacts include potential malicious or unintended uses
685

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
686

(e.g., deployment of technologies that could make decisions that unfairly impact specific
687

groups), privacy considerations, and security considerations.
688

15


---Page Break---
• The conference expects that many papers will be foundational research and not tied
689

to particular applications, let alone deployments. However, if there is a direct path to
690

any negative applications, the authors should point it out. For example, it is legitimate
691

to point out that an improvement in the quality of generative models could be used to
692

generate deepfakes for disinformation. On the other hand, it is not needed to point out
693

that a generic algorithm for optimizing neural networks could enable people to train
694

models that generate Deepfakes faster.
695

• The authors should consider possible harms that could arise when the technology is
696

being used as intended and functioning correctly, harms that could arise when the
697

technology is being used as intended but gives incorrect results, and harms following
698

from (intentional or unintentional) misuse of the technology.
699

• If there are negative societal impacts, the authors could also discuss possible mitigation
700

strategies (e.g., gated release of models, providing defenses in addition to attacks,
701

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
702

feedback over time, improving the efficiency and accessibility of ML).
703

11. Safeguards
704

Question: Does the paper describe safeguards that have been put in place for responsible
705

release of data or models that have a high risk for misuse (e.g., pretrained language models,
706

image generators, or scraped datasets)?
707

Answer: [NA]
708

Justification: We believe that our paper does not pose such risks as we train models for
709

ImageNet classification.
710

Guidelines:
711

• The answer NA means that the paper poses no such risks.
712

• Released models that have a high risk for misuse or dual-use should be released with
713

necessary safeguards to allow for controlled use of the model, for example by requiring
714

that users adhere to usage guidelines or restrictions to access the model or implementing
715

safety filters.
716

• Datasets that have been scraped from the Internet could pose safety risks. The authors
717

should describe how they avoided releasing unsafe images.
718

• We recognize that providing effective safeguards is challenging, and many papers do
719

not require this, but we encourage authors to take this into account and make a best
720

faith effort.
721

12. Licenses for existing assets
722

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
723

the paper, properly credited and are the license and terms of use explicitly mentioned and
724

properly respected?
725

Answer: [No]
726

Justification: we were unable to find the license for the dataset we used.
727

Guidelines:
728

• The answer NA means that the paper does not use existing assets.
729

• The authors should cite the original paper that produced the code package or dataset.
730

• The authors should state which version of the asset is used and, if possible, include a
731

URL.
732

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
733

• For scraped data from a particular source (e.g., website), the copyright and terms of
734

service of that source should be provided.
735

• If assets are released, the license, copyright information, and terms of use in the
736

package should be provided. For popular datasets, paperswithcode.com/datasets
737

has curated licenses for some datasets. Their licensing guide can help determine the
738

license of a dataset.
739

• For existing datasets that are re-packaged, both the original license and the license of
740

the derived asset (if it has changed) should be provided.
741

16


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
742

the asset’s creators.
743

13. New Assets
744

Question: Are new assets introduced in the paper well documented and is the documentation
745

provided alongside the assets?
746

Answer: [NA]
747

Justification: the paper does not release new assets.
748

Guidelines:
749

• The answer NA means that the paper does not release new assets.
750

• Researchers should communicate the details of the dataset/code/model as part of their
751

submissions via structured templates. This includes details about training, license,
752

limitations, etc.
753

• The paper should discuss whether and how consent was obtained from people whose
754

asset is used.
755

• At submission time, remember to anonymize your assets (if applicable). You can either
756

create an anonymized URL or include an anonymized zip file.
757

14. Crowdsourcing and Research with Human Subjects
758

Question: For crowdsourcing experiments and research with human subjects, does the paper
759

include the full text of instructions given to participants and screenshots, if applicable, as
760

well as details about compensation (if any)?
761

Answer: [NA]
762

Justification: the paper does not involve crowdsourcing nor research with human subjects.
763

Guidelines:
764

• The answer NA means that the paper does not involve crowdsourcing nor research with
765

human subjects.
766

• Including this information in the supplemental material is fine, but if the main contribu-
767

tion of the paper involves human subjects, then as much detail as possible should be
768

included in the main paper.
769

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
770

or other labor should be paid at least the minimum wage in the country of the data
771

collector.
772

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
773

Subjects
774

Question: Does the paper describe potential risks incurred by study participants, whether
775

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
776

approvals (or an equivalent approval/review based on the requirements of your country or
777

institution) were obtained?
778

Answer: [NA]
779

Justification: the paper does not involve crowdsourcing nor research with human subjects.
780

Guidelines:
781

• The answer NA means that the paper does not involve crowdsourcing nor research with
782

human subjects.
783

• Depending on the country in which research is conducted, IRB approval (or equivalent)
784

may be required for any human subjects research. If you obtained IRB approval, you
785

should clearly state this in the paper.
786

• We recognize that the procedures for this may vary significantly between institutions
787

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
788

guidelines for their institution.
789

• For initial submissions, do not include any information that would break anonymity (if
790

applicable), such as the institution conducting the review.
791

17


---Page Break---
