Quantile Activation: departing from single point
estimation for better generalization across distortions

Anonymous Author(s)
Affiliation
Address
email

Abstract

A classifier is, in its essence, a function which takes an input and returns the class
1

of the input and implicitly assumes an underlying distribution. We argue in this
2

article that one has to move away from this basic tenet to obtain generalization
3

across distributions. Specifically, the class of the sample should depend on the
4

points from its “context distribution” for better generalization across distributions.
5

How does one achieve this? – The key idea is to “adapt” the outputs of each neuron
6

of the network to its context distribution. We propose quantile activation,QACT,
7

which, in simple terms, outputs the relative quantile of the sample in its context
8

distribution, instead of the actual values in traditional networks.
9

The scope of this article is to validate the proposed activation across several experi-
10

mental settings, and compare it with conventional techniques. For this, we use the
11

datasets developed to test robustness against distortions – CIFAR10C, CIFAR100C,
12

MNISTC, TinyImagenetC, and show that we achieve a significantly higher gen-
13

eralization across distortions than the conventional classifiers, across different
14

architectures. Although this paper is only a proof of concept, we surprisingly find
15

that this approach outperforms DINOv2(small) at large distortions, even though
16

DINOv2 is trained with a far bigger network on a considerably larger dataset.
17

1
Introduction
18

Deep learning approaches have significantly influenced image classification tasks on the machine
19

learning pipelines over the past decade. They can easily beat human performance on such tasks by non-
20

trivial margins by using innovative ideas such as Batch Normalization [19] and other normalization
21

techniques [3, 31], novel rectifiers such as ReLU/PReLU [26, 31, 3] and by using large datasets and
22

large models.
23

However, these classification systems do not generalize across distributions [2, 34], which leads to
24

instability when used in practice. [22] shows that deep networks with ReLU activation degrades in
25

performance under distortions. [4] observes that there exists a feature collapse which inhibits the
26

networks to be reliable.
27

Fundamental Problem of Classification:
We trace the source of the problem to the fact that – Ex-
28

isting classification pipelines necessitates single point prediction, i.e., they should allow classification
29

of a single sample given in isolation. We argue that, for good generalization, one should move away
30

from this basic tenet and instead allow the class to be dependent on the “context distribution” of the
31

sample. That is, when performing classification, one needs to know both the sample and the context
32

of the sample for classification.
33

While this is novel in the context of classification systems, it is widely prevalent in the specific domain
34

of Natural Language Processing (NLP) – The meaning of a word is dependent on the context of the
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Severity 0
Severity 1
Severity 2
Severity 3
Severity 4
Severity 5

QACT

Accuracy
88.02
82.44
76.74
71.54
68.25
65.34

ReLU

Accuracy
90.48
81.06
65.9
48.34
41.54
35.86

Figure 1: Comparing TSNE plots of QACT and ReLU activation on CIFAR10C with Gaussian
distortions. Observe that QACT maintains the class structure extremely well across distortions, while
the usual ReLU activations loses the class structure as severity increases.

word. However, to our knowledge, this has not been considered for general classification systems.
36

Even when using dominant NLP architectures such as Transformers for vision [8], the technique has
37

been to split the image into patches and then obtain the embedding for an individual image.
38

Obtaining a classification framework for incorporating context distribution:
We suspect that
39

the main reason why the context distribution is not incorporated into the classification system is
40

– The naive approach of considering a lot of samples in the pipeline to classify a single sample
41

is computationally expensive. We solve this problem by considering the context distribution of
42

each neuron specifically. We introduce Quantile Activation (QACT), which outputs a probability
43

depending upon the context of all outputs. This, however, gives rise to new challenges in training,
44

which we address in section 2.
45

Figure 1 illustrates the differences of the proposed framework with the existing framework. As
46

severity increases (w.r.t Gaussian Noise), we observe that ReLU activation loses the class structure.
47

This behaviour can be attributed to the fact that, as the input distribution changes, the activations
48

either increase/decrease, and due to the multiplicative effect of numerous layers, this leads to very
49

different features. On the other hand, the proposed QACT framework does not suffer from this, since
50

if all the pre-activations1 change in a systematic way, the quantiles adjust automatically to ensure that
51

the inputs for the next layer does not change by much. This is reflected in the fact that class structure
52

is preserved with QACT.
53

Remark:
Quantile activation is different from existing quantile neural network based approaches,
54

such as regression [30], binary quantile classification [36], Anomaly Detection [24, 33]. Our approach
55

is achieving best in-class performance by incorporating context distribution in the classification
56

paradigm. Our approach is also markedly different from Machine unlearning which is based on
57

selective forgetting of certain data points or retraining from scratch [32].
58

Contributions:
A decent amount of literature on neuronal activation is available. However, to the
59

best of our knowledge, none matches the central idea proposed in this work.
60

In [5], the authors propose an approach to calibrate a pre-trained classifier fθ(x) by extending it
61

to learn a quantile function, Q(x, θ, τ) (τ denotes the quantile), and then estimate the probabilities
62

using
R

τ I[Q(x, θ, τ) ≥0.5]dτ 2. They show that this results in probabilities which are robust to
63

distortions.
64

1. In this article, we extend this approach to the level of a neuron, by suitably deriving the
65

forward and backward propagation equations required for learning (section 2).
66

2. We then show that a suitable incorporation of our extension produces context dependent
67

outputs at the level of each neuron of the neural network.
68

1We use the following convention – “Pre-activations” denote the inputs to the activation functions and
“Activations” denote the outputs of the activation function.
2I[] denotes the indicator function

2


---Page Break---
3. Our approach contributes to achieving better generalization across distributions and is
69

more robust to distortions, across architectures. We evaluate our method using different
70

architectures and datasets, and compare with the current state-of-the-art – DINOv2. We
71

show that QACT proposed here is more robust to distortions than DINOv2, even if we
72

have considerably less number of parameters (22M for DINOv2 vs 11M for Resnet18).
73

Additionally, DINOv2 is trained on 20 odd datasets, before being applied on CIFAR10C; in
74

contrast, our framework is trained on CIFAR10, and produces more robust outcome (see
75

figures 3,5).
76

4. The proposed QACT is consistent with all the existing techniques used in DINOv2, and
77

hence can be easily incorporated into any ML framework.
78

5. We also adapt QACT to design a classifier which returns better calibrated probabilities.
79

Related Works on Domain Generalization (DG):
The problem of domain generalizations tries
80

to answer the question – Can we use a classifier trained on one domain across several other related
81

domains? The earliest known approach for this is Transfer Learning [28, 37], where a classifier
82

from a single domain is applied to a different domain with/without fine-tuning. Several approaches
83

have been proposed to achieve DG, such as extracting domain-invariant features over single/multiple
84

source domains [11, 1, 9, 29, 16], Meta Learning [17, 9], Invariant Risk Minimization [2]. Self
85

supervised learning is another proposed approach which tries to extract features on large scale datasets
86

in an unsupervised manner, the most recent among them being DINOv2 [27]. Very large foundation
87

models, such as GPT-4V, are also known to perform better with respect to distribution shifts [12].
88

Nevertheless, to the best of our knowledge, none of these models incorporates context distributions
89

for classification.
90

2
Quantile Activation
91

Rethinking Outputs from a Neuron:
To recall – if x denotes the input, a typical neuron does the
92

following – (i) Applies a linear transformation with parameters w, b, giving wtx + b as the output,
93

and (ii) applies a rectifier g, returning g(wtx + b). Typically, g is taken to be the ReLU activation -
94

grelu(x) = max(0, x). Intuitively, we expect that each neuron captures an “abstract” feature, usually
95

not understood by a human observer.
96

An alternate way to model a neuron is to consider it as predicting a latent variable y, where y = 1
97

if the feature is present and y = 0 if the feature is absent. Mathematically, we have the following
98

model:
99

z = wtx + b + ϵ
and
y = I[z ≥0]
(1)

This is very similar to the standard latent variable model for logistic regression, with the main
100

exception being, the outputs y are not known for each neuron beforehand. If y is known, it is rather
101

easy to obtain the probabilities – P(z ≥0). Can we still predict the probabilities, even when y itself
102

is a latent variable?
103

The authors in [5] propose the following algorithm to estimate the probabilities:
104

1. Let {xi} denote the set of input samples from the input distribution x and {zi} denote their
105

corresponding latent outputs, which would be from the distribution z
106

2. Assign y = 1 whenever z > (1 −τ)th quantile of z, and 0 otherwise. For a specific sample,
107

we have yi = 1 if zi > (1 −τ)th quantile of {zi}
108

3. Fit the model Q(x, τ; θ) to the dataset {((xi, τ), yi)}, and estimate the probability as,
109

P(yi = 1) =
Z 1

τ=0
I[Q(x, τ; θ) ≥0.5]dτ
(2)

The key idea:
Observe that in step 2., the labelling is done without resorting to actual ground-
110

truth labels. This allows us to obtain the probabilities on the fly for any set of parameters, only by
111

considering the quantiles of z.
112

Defining the Quantile Activation QACT
Let z denote the pre-activation of the neuron, and let
113

{zi} denote the samples from this distribution. Let Fz denote the cumulative distribution function
114

(CDF), and let fz denote the density of the distribution. Accordingly, we have that F −1
z
(τ) denotes
115

3


---Page Break---
Algorithm 1 Forward Propagation for a single neuron
Input: [zi] a vector of pre-activations, 0 < τ1 < τ2 < · · · < τnτ < 1 - a list of quantile indices at
which we compute the quantiles.

Append two large values, c and −c, to the vector [zi].
Count n+ = number of positive values, n−= number of negative values, and assign the weight
w+ = 1/n+ to the positive values, and w−= 1/n−to the negative values.
Compute weighted quantiles {qi} at each of {τi} over the set {zi} ∪{c, −c}
Compute QACT(zi) using the function,

QACT(x) = 1

nτ

X

i
I[x ≥qi]
(5)

Remember [zi], w+, w−, [QACT(zi)] for backward propagation.
return [QACT(zi)]

Algorithm 2 Backward Propagation for a single neuron
Input: grad_output, 0 < τ1 < τ2 < · · · < τnτ < 1 - a list of quantile indices at which we compute
the quantiles.
Context from Forward Propagation: [zi], w+, w−, [QACT(zi)]

Obtain a weighted sample from [zi] with weights w+, w−– (say) S.
Obtain a kernel density estimate, using points from S, at each of the points in zi – (say) ˆfz(zi)
Set,
grad_input = grad_output ⊙[ ˆfz(zi)]
(6)

return grad_input

the τ th quantile of z. Using step (2) of the algorithm above, we define,
116

QACT(z) =
Z 1

τ=0
I[z > F −1
z
(1 −τ)]dτ
Substitute
=
τ→(1−τ)

Z 1

τ=0
I[z > F −1
z
(τ)]dτ
(3)

Computing the gradient of QACT:
However, to use QACT in a neural network, we need to
117

compute the gradient which is required for back-propagation. Let τz denote the quantile at which
118

F −1
z
(τz) = z. Then we have that QACT(z) = τz since F −1
z
(τ) is an increasing function. So, we
119

have that QACT(F −1
z
(τ)) = τ. In other words, we have that QACT(z) is Fz(z), which is nothing
120

but the CDF of z. Hence, we have,
121

∂QACT(z)

∂z
= fz(z)
(4)

where fz(z) denotes the density of the distribution.
122

Grounding the Neurons:
With the above formulation, observe that since QACT is identical to
123

CDF, it follows that, QACT(z) is always a uniform distribution between 0 and 1, irrespective of the
124

distribution z. When training numerous neurons in a layer, this could cause all the neurons to learn
125

the same behaviour. Specifically, if, half the time, a particular abstract feature is more prevalent
126

than others, QACT (as presented above) would not be able to learn this feature. To correct this, we
127

enforce that positive values and negative values have equal weight. Given the input distribution z,
128

We perform the following transformation before applying QACT. Let
129

z+ =
z
if z ≥0
0
otherwise
z−=
z
if z < 0
0
otherwise
(7)

denote the truncated distributions. Then,
130

z‡ =
z+
with probability 0.5
z−
with probability 0.5
(8)

From definition of z‡, we get that the median of z‡ is 0. This grounds the input distribution to have
131

the same positive and negative weight.
132

4


---Page Break---
(a)
(b)

(c)

(d)

Figure 2: Intuition behind quantile activation. (a) shows a simple toy distribution of points (blue), it’s
distortion (orange) and a simple line (red) on which the samples are projected to obtain activations.
(b) shows the distribution of the pre-activations. (c) shows the distributions of the activations with
QACT of the original distribution (blue). (d) shows the distributions of the activations with QACT
under the distorted distribution (orange). Observe that the distributions match perfectly under small
distortions. Note that even if the distribution matches perfectly, the quantile activation is actually a
deterministic function.

Dealing with corner cases:
It is possible that during training, some neurons either only get positive
133

values or only get negative values. However, for smooth outputs, one should still only give the weight
134

of 0.5 for positive values. To handle this, we include two values c (large positive) and −c (large
135

negative) for each neuron. Since, the quantiles are conventionally computed using linear interpolation,
136

this allows the outputs to vary smoothly. We take c = 100 in this article.
137

Estimating the Density for Back-Propagation:
Note that the gradient for the back propagation is
138

given by the density of z‡ (weighted distribution). We use the Kernel Density Estimation (KDE), to
139

estimate the density. We, (i) First sample N points with weights w+, w−, and (ii) then estimate the
140

density at all the input points [zi]. This is point-wise multiplied with the backward gradient to get the
141

gradient for the input. In this article we use N = 1000, which we observe gets reasonable estimates.
142

Computational Complexity:
Computational Complexity (for a single neuron) is majorly decided
143

by 2 functions – (i) Computing the quantiles has the complexity for a vector [zi] of size n can be
144

performed in O(n log(n)). Since this is log-linear in n, it does not increase the complexity drastically
145

compared to other operations in a deep neural network. (ii) Computational complexity of the KDE
146

estimates is O(Snτ) where S is the size of sample (weighted sample from [zi]) and nτ is the number
147

of quantiles, giving a total of O(n + Snτ). In practice, we consider S = 1000 and nτ = 100 which
148

works well, and hence does not increase with the batch size. This too scales linearly with batch size
149

n, and hence does not drastically increase the complexity.
150

Remark:
Algorithms 1, and 2 provide the pseudocode for the quantile activation. For stable
151

training, in practice, we prepend and append the quantile activation with BatchNorm layers.
152

Why QACT is robust to distortions?
To understand the idea behind quantile activation, consider a
153

simple toy example in figure 2. For ease of visualization, assume that the input features (blue) are in 2
154

dimensions, and also assume that the line of the linear projection is given by the red line in figure 2a.
155

Now, assume that the blue input features are rotated, leading to a different distribution (indicated here
156

by orange). Since activations are essentially (unnormalized) signed distances from the line, we plot
157

the histograms corresponding to the two distributions in figure 2b. As expected, these distributions
158

are different. However, after performing the quantile activation in equation 3, we have that both are
159

uniform distribution. This is illustrated in figures 2c and 2d. This behaviour has a normalizing effect
160

across different distributions, and hence has better distribution generalization than other activations.
161

5


---Page Break---
3
Training with QACT
162

In the previous section, we described the procedure to adapt a single neuron to its context distribution.
163

In this section we discuss how this extends to the Dense/Convolution layers, the loss functions to
164

train the network and the inference aspect.
165

Extending to standard layers:
The extension of equation 3 to dense outputs is straightforward.
166

A typical output of the dense layer would be of the shape (B, Nc) - B denotes the batch size, Nc
167

denotes the width of the network. The principle is - The context distribution of a neuron is all the
168

values which are obtained using the same parameters. In this case, each of the values across the ‘B’
169

dimension are considered to be samples from the context distribution.
170

For a convolution layer, the typical outputs are of the form - (B, Nc, H, W) - B denotes the size of
171

the batch, Nc denotes the number of channels, H, W denotes the sizes of the images. In this case we
172

should consider all values across the 1st,3rd and 4th dimension to be from the context distribution,
173

since all these values are obtained using the same parameters. So, the number of samples would be
174

B × H × W.
175

Loss Functions:
One can use any differentiable loss function to train with quantile activation. We
176

specifically experiment with the standard Cross-Entropy Loss, Triplet Loss, and the recently proposed
177

Watershed Loss [6] (see section 4). However, if one requires that the boundaries between classes
178

adapt to the distribution, then learning similarities instead of boundaries can be beneficial. Both
179

Triplet Loss and Watershed Loss fall into this category. We see that learning similarities does have
180

slight benefits when considering the embedding quality.
181

Inference with QACT:
As stated before, we want to assign a label for classification based on the
182

context of the sample. There exist two approaches for this – (1) One way is to keep track of the
183

quantiles and the estimated densities for all neurons and use it for inference. This allows inference
184

for a single sample in the traditional sense. However, this also implies that one would not be able
185

to assign classes based on the context at evaluation. (2) Another way is to make sure that, even for
186

inference on a single sample, we include several samples from the context distribution, but only use
187

the output for a specific sample. This allows one to assign classes based on the context. In this article,
188

we follow the latter approach.
189

Quantile Classifier:
Observe that the proposed QACT (without normalization) returns the values in
190

[0, 1] which can be interpreted as probabilities. Hence, one can also use this for the classification layer.
191

Nonetheless, two changes are required – (i) Traditional softmax used in conjunction with negative-
192

log-likelihoood loss already considers “relative” activations of the classification in normalization.
193

However, QACT does not. Hence, one should use Binary-Cross-Entropy loss with QACT, which
194

amounts to one-vs-rest classification. (ii) Also, unlike a neuron in the middle layers, the bias of the
195

neuron in the classification layer depends on the class imbalance. For instance, with 10 classes, one
196

would have only 1/10 of the samples labelled 1 and 9/10 of the samples labelled 0. To address this,
197

we require that the median of the outputs be at 0.9, and hence weight the positive class with 0.9 and
198

the negative class with 0.1 respectively. In this article, whenever QACT is used, we use this approach
199

for inference.
200

We observe that (figures 13 and 14) using quantile classifier on the learned features in general
201

improves the consistency of the calibration error and also leads to the reducing the calibration error.
202

In this article, for all networks trained with quantile activation, we use quantile classifier to compute
203

the accracies/calibration errors.
204

4
Evaluation
205

To summarize, we make the following changes to the existing classification pipeline – (i) Replace the
206

usual ReLU activation with QACT and (ii) Use triplet or watershed loss instead of standard cross
207

entropy loss. We expect this framework to learn context dependent features, and hence be robust
208

to distortions. (iii) Also, use quantile classifier to train the classifier on the embedding for better
209

calibrated probabilities.
210

6


---Page Break---
(a) Accuracy
(b) Top-Label Calibration Error

Figure 3: Comparing QACT with ReLU activation and Dinov2 (small) on CIFAR10C. We observe
that, while at low severity of distortions QACT has a similar accuracy as existing pipelines, at
higher levels the drop in accuracy is substantially smaller than existing approaches. With respect to
calibration, we observe that the calibration error remains constant (up to standard deviations) across
distortions.

Evaluation Protocol:
To evaluate our approach, we consider the two datasets developed for this
211

purpose – CIFAR10C, CIFAR100C, TinyImagenetC [15], MNISTC[25]. These datasets have a
212

set of 15 distortions at 5 severity levels. To ensure diversity we evaluate our method on 4 ar-
213

chitectures – (overparametrized) LeNet, ResNet18[14] (11M parameters), VGG[35](15M param-
214

eters) and DenseNet [18](1M parameters). The code to reproduce the results can be found at
215

https://anonymous.4open.science/r/QuantAct-2B41.
216

Baselines for Comparison:
To our knowledge, there exists no other framework which proposed
217

classification based on context distribution. So, for comparison, we consider standard ReLU activation
218

[10], pReLU [13], and SELU [20] for all the architectures stated above. Also, we compare our
219

results with DINOv2 (small) [27] (22M parameters) which is current state-of-the-art for domain
220

generalization. Note that for DINOv2, architecture and datasets used for training are substantially
221

different (and substantially larger) from what we consider in this article. Nevertheless, we include the
222

results for understanding where our proposed approach lies on the spectrum. We consider the small
223

version of DINOv2 to match the number of parameters with the compared models.
224

Metrics:
We consider four metrics – Accuracy (ACC), calibration error (ECE) [23] (both marginal
225

and Top-Label) and mean average precision at K (MAP@K) to evaluate the embedding. For the case
226

of ReLU/pReLU/SELU activation with Cross-Entropy, we use the logistic regression trained on the
227

train set embeddings, and for QACT we use the calibrated linear classifier, as proposed above. We do
228

not perform any additional calibration and use the probabilities. We discuss a selected set of results
229

in the main article. Please see appendix C for more comprehensive results.
230

Calibration error measures the reliability of predicted probabilities. In simple words, if one predicts
231

100 samples with (say) probability 0.7, then we expect 70 of the samples to belong to class 1 and the
232

rest to class 0. This is measured using either the marginal or top-label calibration error. We refer the
233

reader to [23] for details, which also provides an implementation to estimate the calibration error.
234

Remark:
For all the baselines we use the standard Cross-Entropy loss for training. For inference
235

on corrupted datasets, we retrain the last layer with logistic regression on the train embedding and
236

evaluate it on test/corrupted embedding. For QACT, we as a convention use watershed loss unless
237

otherwise stated, for training. For inference, we train the Quantile Classifier on the train embedding
238

and evaluate it on test/corrupted embedding.
239

The proposed QACT approach is robust to distortions:
In fig. 3 we compare the proposed
240

QACT approach with predominant existing pipeline – ReLU+Cross-Entropy and DINOv2(small)
241

on CIFAR10C. In figure 3a we see that as the severity of the distortion increases, the accuracy of
242

ReLU and DINOv2 drops significantly. On the other hand, while at small distortions the results are
243

comparable, as severity increases QACT performs substantially better than conventional approaches.
244

7


---Page Break---
(a)
(b)
(c)

Figure 4: (a) Dependence on Loss functions. Here we compare watershed with other popular
loss functions – Triplet and Cross-Entropy when used with QACT. We see that watershed per-
forms slightly better with respect to MAP. (b) Comparing QACT with other popular activations –
ReLU/pReLU/SELU with respect to accuracy. (c) Comparing QACT with other popular activations
– ReLU/pReLU/SELU with respect to Calibration Error (Marginal). From both (b) and (c) we can
conclude that QACT is notably more robust across distortions than several of the existing activation.
All the plots use ResNet18 with CIFAR10C dataset.

(a)
(b)

Figure 5: Results on CIFAR100C/TinyImagenetC. We compare QACT+watershed to ReLU and
DinoV2 small on CIFAR100C/TinyImagenetC dataset with ResNet18. Note that the observations are
consistent with CIFAR10C. (a) shows how accuracy changes across distortions. Observe that QACT
is similar to DINOv2(s) with respect to embedding quality across all distortions, even if DINOv2
has 22M parameters as compared to Resnet18 11M parameters and is trained on larger datasets. (b)
shows how calibration error (marginal) changes across severities. While other approaches lead to an
increase in calibration error, QACT has similar calibration error across distortions.

At severity 5, QACT outperforms DINOv2. On the other hand, we observe that in figure 3b, the
245

calibration error stays consistent across distortions.
246

How much does QACT depend on the loss function?
Figure 4a compares the watershed classifier
247

with other popular losses – Triplet and Cross-Entropy. We see that all the loss functions perform com-
248

parably when used in conjunction with QACT. We observe that watershed has a slight improvement
249

when considering MAP and hence, we consider that as the default setting. However, we point out
250

that QACT is compatible with several loss functions as well.
251

QACT vs ReLU/pReLU/SELU activations:
To verify that most existing activations do not share
252

the robustness property of QACT, we compare QACT with other activations in figures 4b and 4c. We
253

observe that QACT is greatly more robust with respect to distortions in both accuracy and calibration
254

error than other activation functions.
255

Results on Larger Datasets:
To verify that our observations hold for larger datasets, we use
256

CIFAR100C/TinyImagenetC to compare the proposed QACT+watershed with existing approaches.
257

We observe on figure 5 that QACT performs comparably well as DINOv2, although DINOv2(s)
258

has 22M parameters and is trained on significantly larger datasets. Moreover, we also observe that
259

8


---Page Break---
QACT has approximately constant calibration error across distortions, as opposed to a significantly
260

increasing calibration error for ReLU or DINOv2.
261

5
Conclusion And Future Work
262

To summarize, traditional classification systems do not consider the “context distributions” when
263

assigning labels. In this article, we propose a framework to achieve this by – (i) Making the activation
264

adaptive by using quantiles and (ii) Learning a kernel instead of the boundary for the last layer. We
265

show that our method is more robust to distortions by considering MNISTC, CIFAR10C, CIFAR100C,
266

TinyImagenetC datasets across varying architectures.
267

The scope of this article is to provide a proof of concept and a framework for performing inference in
268

a context-dependent manner. We outline several potential directions for future research:
269

I. The key idea in our proposed approach is that the quantiles capture the distribution of each
270

neuron from the batch of samples, providing outputs accordingly. This poses a challenge for
271

inference, and we have discussed two potential solutions: (i) remember the quantiles and
272

density estimates for single sample evaluation, or (ii) ensure that a batch of samples from
273

the same distribution is processed together. We adopt the latter method in this article. An
274

alternative approach would be to learn the distribution of each neuron using auxiliary loss
275

functions, adjusting these distributions to fit the domain at test time. This gives us more
276

control over the network at test time compared to current workflows.
277

II. Since the aim of the article was to establish a proof-of-concept, we did not focus on scaling,
278

and use only a single GPU for all the experiments. To extend it to multi-GPU training,
279

one needs to synchronize the quantiles across GPU, in a similar manner as that for Batch-
280

Normalization. We expect this to improve the statistics, and to allow considerably larger
281

batches of training.
282

III. On the theoretical side, there is an interesting analogy between our quantile activation and
283

how a biological neuron behaves. It is known that when the inputs to a biological neuron
284

change, the neuron adapts to these changes [7]. Quantile activation does something very
285

similar, which leads to an open question – can we establish a formal link between the
286

adaptability of a biological neuron and the accuracy of classification systems?
287

IV. Another theoretical direction to explore involves considering distributions not just at the
288

neuron level, but at the layer level, introducing a high-dimensional aspect to the problem.
289

The main challenge here is defining and utilizing high dimensional quantiles, which remains
290

an open question [21].
291

Broad Impact:
In this article, we propose an approach to maintain calibration and generalization
292

across small distortions. While, we do not foresee any direct societal consequences of our work, we
293

expect the potential future consequences of the technique to reduce the bias in the following ways
294

– (i) Since we do not assume normal distribution, our approach is likely to handle long tails better
295

than existing methods. This would help in reducing the dataset bias where marginal groups are less
296

represented. (ii) Note that the output of each QACT layer is a uniform distribution. This can allow us
297

to understand the working of each layer in isolation and possibly reduce the black-box nature of the
298

current classification systems. (iii) Moreover, by directly modifying the context distribution of each
299

neuron, one can easily make the networks more reliable without resorting to expensive re-training the
300

entire network.
301

References
302

[1] Kei Akuzawa, Yusuke Iwasawa, and Yutaka Matsuo. Adversarial invariant feature learning with
303

accuracy constraint for domain generalization. In European Conf. Mach. Learning, 2019.
304

[2] Martín Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk mini-
305

mization. arXiv:1907.02893, 2019.
306

[3] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.
Layer normalization.
307

arXiv:1607.06450, 2016.
308

9


---Page Break---
[4] Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger B. Grosse, and Jörn-Henrik Jacobsen.
309

Understanding and mitigating exploding inverses in invertible neural networks. In Artificial
310

Intelligence and Statistics, 2021.
311

[5] Aditya Challa, Snehanshu Saha, and Soma Dhavala. Quantprob: Generalizing probabilities
312

along with predictions for a pre-trained classifier. arXiv:2304.12766, 2023.
313

[6] Aditya Challa, Sravan Danda, and Laurent Najman. A novel approach to regularising 1nn
314

classifier for improved generalization. arXiv:2402.08405, 2024.
315

[7] Colin WG Clifford, Michael A Webster, Garrett B Stanley, Alan A Stocker, Adam Kohn,
316

Tatyana O Sharpee, and Odelia Schwartz.
Visual adaptation: Neural, psychological and
317

computational aspects. Vision research, 2007.
318

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
319

Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
320

Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image
321

recognition at scale. In Int. Conf. on Learning Representations, 2021.
322

[9] Qi Dou, Daniel Coelho de Castro, Konstantinos Kamnitsas, and Ben Glocker. Domain gener-
323

alization via model-agnostic learning of semantic features. In Neural Inform. Process. Syst.,
324

2019.
325

[10] Kunihiko Fukushima. Correction to "visual feature extraction by a multilayered network of
326

analog threshold elements". IEEE Trans. Syst. Sci. Cybern., 1970.
327

[11] Muhammad Ghifary, W. Bastiaan Kleijn, Mengjie Zhang, and David Balduzzi.
Domain
328

generalization for object recognition with multi-task autoencoders. In Proc. Int. Conf. Comput.
329

Vision, 2015.
330

[12] Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Tailin Wu, Yilong Yin, Salman H.
331

Khan, Lina Yao, Tongliang Liu, and Kun Zhang.
How well does gpt-4v(ision) adapt to
332

distribution shifts? A preliminary investigation. arXiv:2312.07424, 2023.
333

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers:
334

Surpassing human-level performance on imagenet classification. In Proc. Int. Conf. Comput.
335

Vision, 2015.
336

[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
337

recognition. In Proc. Conf. Comput. Vision Pattern Recognition, 2016.
338

[15] Dan Hendrycks and Thomas G. Dietterich. Benchmarking neural network robustness to common
339

corruptions and perturbations. In Int. Conf. on Learning Representations, 2019.
340

[16] Shoubo Hu, Kun Zhang, Zhitang Chen, and Laiwan Chan. Domain generalization via multido-
341

main discriminant analysis. In Uncertainity in Artificial Intelligence, 2019.
342

[17] Bincheng Huang, Si Chen, Fan Zhou, Cheng Zhang, and Feng Zhang. Episodic training for
343

domain generalization using latent domains. In Int. Conf. on Cogni. Systems and Signal Process.,
344

2020.
345

[18] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected
346

convolutional networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition,
347

CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 2261–2269. IEEE Computer Society,
348

2017. doi: 10.1109/CVPR.2017.243. URL https://doi.org/10.1109/CVPR.2017.243.
349

[19] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training
350

by reducing internal covariate shift. In Int. Conf. Mach. Learning, 2015.
351

[20] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing
352

neural networks. In Neural Inform. Process. Syst., 2017.
353

[21] Roger Koenker. Quantile Regression. Econometric Society Monographs. Cambridge University
354

Press, 2005. doi: 10.1017/CBO9780511754098.
355

10


---Page Break---
[22] Agustinus Kristiadi, Matthias Hein, and Philipp Hennig. Being bayesian, even just a bit, fixes
356

overconfidence in relu networks. In Int. Conf. Mach. Learning, 2020.
357

[23] Ananya Kumar, Percy Liang, and Tengyu Ma. Verified uncertainty calibration. In Neural Inform.
358

Process. Syst., 2019.
359

[24] Zhong Li and Matthijs van Leeuwen. Explainable contextual anomaly detection using quantile
360

regression forests. Data Min. Knowl. Discov., 2023.
361

[25] Norman Mu and Justin Gilmer. MNIST-C: A robustness benchmark for computer vision.
362

arXiv:1906.02337, 2019.
363

[26] Vinod Nair and Geoffrey E. Hinton.
Rectified linear units improve restricted boltzmann
364

machines. In Int. Conf. Mach. Learning, 2010.
365

[27] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,
366

Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran,
367

Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra,
368

Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick
369

Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without
370

supervision. arXiv:2304.07193, 2023.
371

[28] Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Trans. Knowl. Data
372

Eng., 2010.
373

[29] Vihari Piratla, Praneeth Netrapalli, and Sunita Sarawagi. Efficient domain generalization via
374

common-specific low-rank decomposition. In Int. Conf. Mach. Learning, 2020.
375

[30] Tejas Prashanth, Snehanshu Saha, Sumedh Basarkod, Suraj Aralihalli, Soma S. Dhavala,
376

Sriparna Saha, and Raviprasad Aduri. Lipgene: Lipschitz continuity guided adaptive learning
377

rates for fast convergence on microarray expression data sets. IEEE ACM Trans. Comput. Biol.
378

Bioinform., 2022.
379

[31] Tim Salimans and Diederik P. Kingma. Weight normalization: A simple reparameterization to
380

accelerate training of deep neural networks. In Neural Inform. Process. Syst., 2016.
381

[32] Aditi Seetha, Satyendra Singh Chouhan, Emmanuel S Pilli, Vaskar Raychoudhury, and Snehan-
382

shu Saha. Dievd-sf: Disruptive event detection using continual machine learning with selective
383

forgetting. IEEE Transactions on Computational Social Systems, 2024.
384

[33] Hogeon Seo, Seunghyoung Ryu, Jiyeon Yim, Junghoon Seo, and Yonggyun Yu. Quantile
385

autoencoder for anomaly detection. In AAAI,Workshop on AI for Design and Manufacturing
386

(ADAM), 2022.
387

[34] Zheyan Shen, Peng Cui, Tong Zhang, and Kun Kuang. Stable learning via sample reweighting.
388

In AAAI Conf. on Artificial Intelligence, 2020.
389

[35] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
390

image recognition. In Int. Conf. on Learning Representations, 2015.
391

[36] Anuj Tambwekar, Anirudh Maiya, Soma S. Dhavala, and Snehanshu Saha. Estimation and
392

applications of quantiles in deep binary classification. IEEE Trans. Artif. Intell., 2022.
393

[37] Fuzhen Zhuang, Zhiyuan Qi, Keyu Duan, Dongbo Xi, Yongchun Zhu, Hengshu Zhu, Hui Xiong,
394

and Qing He. A comprehensive survey on transfer learning. Proc. IEEE, 2021.
395

11


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 6: Comparing QACT with ReLU activation and Dinov2 (small).

A
Experiment details for figure 1
396

We consider the features obtained from ResNet18 with both QACT and RelU activations for the
397

datasets of CIFAR10C with gaussian_noise at all the severity levels. Hence, we have 6 datasets
398

in total. To use TSNE for visualization, we consider 1000 samples from each dataset and obtain
399

the combined TSNE visualizations. Each figure shows a scatter plot of the 2d visualization for the
400

corresponding dataset.
401

B
Compute Resources and Other Experimental Details
402

All experiments were performed on a single NVidia GPU with 32GB memory with Intel Xeon CPU
403

(10 cores). For training, we perform an 80:20 split of the train dataset with seed 42 for reproducibility.
404

All networks are initialized using default pytorch initialization technique.
405

We use Adam optimizer with initial learning rate 1e −3. We use ReduceLRonPlateau learning
406

rate scheduler with parameters – factor=0.1, patience=50, cooldown=10, threshold=0.01, thresh-
407

old_mode=abs, min_lr=1e-6. We monitor the validation accuracy for learning rate scheduling. We
408

also use early_stopping when the validation accuracy does not increase by 0.001.
409

C
Extended Results Section
410

Comparing QACT + watershed and ReLU+Cross-Entropy:
Figure 6 shows the corresponding
411

results. The first experiment compares QACT + watershed with ReLU + Cross-Entropy on two
412

standard networks – ResNet18 and DenseNet. With respect to accuracy, we observe that while at
413

severity 0, ReLU + Cross-Entropy slightly outperforms QACT + watershed, as severity increases
414

QACT + watershed is far more stable. We even outperform DinoV2(small) (22M parameters) at
415

severity 5. Moreover, with respect to calibration error, we see a consistent trend across distortions.
416

As [5] argues, this helps in building more robust systems compared to one where calibration error
417

increases across distortions.
418

12


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 7: Triplet vs Watershed vs Cross-Entropy

Does loss function make a lot of difference?
Figure 7 compares three different loss functions
419

Watershed, Triplet and Cross-Entropy when used in conjunction with QACT. We observe similar
420

trends across all loss functions. However, Watershed performs better with respect to Mean Average
421

Precision (MAP) and hence we use this as a default strategy.
422

Why Mean-Average-Precision? – We argue that the key indicator of distortion invariance should
423

be the quality of embedding. While, accuracy (as measured by a linear classifier) is a good metric,
424

a better one would be to measure the Mean-Average-Precision. With respect to calibration error,
425

due to the scale on the Y-axis, the figures suggest reducing calibration error. However, the standard
426

deviations overlap, and hence, these are assumed to be constant across distortions.
427

How well does watershed perform when used with ReLU activation?
Figure 8 shows the
428

corresponding results. We observe that both the watershed loss and cross-entropy have large overlaps
429

in the standard deviations at all severity levels. So, this shows that, when used in conjunction with
430

ReLU watershed and cross-entropy loss are very similar. But in conjunction with QACT, we see that
431

watershed has a slightly higher Mean-Average-Precision.
432

What if we consider an easy classification task?
In figure 9, we perform the comparison of
433

QACT+Watershed and ReLU and cross-entropy on MNISTC dataset. Across different architectures,
434

we observe a lot less variation (standard deviation) of QACT+Watershed compared to RelU and
435

cross-entropy. This again suggests robustness against distortions of QACT+Watershed.
436

Comparing with other popular activations:
Figures 10 and 11 shows the comparison of QACT
437

with ReLU, pReLU and SeLU. We observe the same trend across ReLU, pReLU and SeLU, while
438

QACT is far more stable across distortions.
439

Results on CIFAR100/TinyImagenetC:
Figure 12 compares QACT+Watershed and ReLU+Cross-
440

Entropy on CIFAR100C dataset.
We also include the results of QACT+Cross-Entropy vs.
441

ReLU+Cross-Entropy on TinyImagenetC. The results are consistent with what we observe on
442

CIFAR10C, and hence, draw the same conclusions as before.
443

13


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 8: Watershed vs Cross-Entropy when using ReLU activation

(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 9: Results on MNIST

14


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 10: QACTvs ReLU vs pReLU vs Selu activations on ResNet18

(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 11: QACTvs ReLU vs pReLU vs Selu activations on Densenet

15


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 12: QACTvs ReLU on Resnet18+CIFAR100

(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 13: Effect of Quantile Classifier. We use ResNet18 and DinoV2 architectures on CIFAR10.

16


---Page Break---
(a) Accuracy
(b) Mean Average Precision (MAP@100)

(c) Marginal Calibration Error
(d) Top-Label Calibration Error

Figure 14: Effect of Quantile Classifier. We use ResNet18 and DinoV2 architectures on CIFAR100.

Effect of Quantile Classifier:
Figures 13 and 14 shows the effect of quantile classifier on standard
444

ResNet10/DinoV2 outputs with CIFAR10C/CIFAR100C datasets. While the accuracy values are
445

almost equivalent, we observe a “flatter” trend of the calibration errors, sometimes reducing the error
446

as in the case of CIFAR100C.
447

D
Watershed Loss
448

The authors in [6] proposed a novel classifier – watershed classifier, which works by learning
449

similarities instead of the boundaries. Below we give the brief idea of the loss function, and refer the
450

reader to the original paper for further details.
451

1. Let (xi, yi) denote the samples in each batch, and let fθ denote the embedding network.
452

fθ(xi) denotes the corresponding embedding.
453

2. Starting from randomly selected seeds in the batch, propagate the labels to all the samples.
454

Let ˆyi denote the estimated samples. For each fθ(xi) and for each label l, obtain the nearest
455

neighbour in the samples in the set,
456

Sl = {fθ(xi) | ˆyi = yi = l}
(9)

that is, all the samples of class l labelled correctly. Denote this nearest neighbour using
457

fθ(xi,l,1nn).
458

3. Then the loss is given by,
459

Watershed Loss =
−1
nsamples

nsamples
X

i=1

L
X

l=1
I[yi = l] log

 
exp (−∥fθ(xi) −fθ(xi,l,1nn)∥)
PL
j=1 exp (−∥fθ(xi) −fθ(xi,j,1nn)∥)

!

(10)

Why Watershed Loss?:
Observe that the loss in equation 10 implicitly learns representations
460

consistent with the RBF kernel, which is known to be translation invariant. Minimizing this loss
461

function, hence, will learn translation invariant kernels. This is important for obtaining networks
462

robust to distortions.
463

17


---Page Break---
If one uses (say) Cross Entropy loss, then the features learned would be such that the classes are
464

linearly separable. Contrast this with watershed, which instead learns a similarity between two points
465

in a translation invariant manner.
466

Remark:
Observe that the watershed loss is very similar to metric learning losses. The authors in
467

[6] claim that this offers better generalization, and show that this is consistent with 1NN classifier.
468

Moreover, they show that this classifier (without considering fθ) has a VC dimension which is equal
469

to the number of classes. While metric learning losses are similar, there is no such guarantee with
470

respect to classification. This motivated our choice of using watershed loss over other metric learning
471

losses.
472

18


---Page Break---
NeurIPS Paper Checklist
473

The checklist is designed to encourage best practices for responsible machine learning research,
474

addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
475

the checklist: The papers not including the checklist will be desk rejected. The checklist should
476

follow the references and precede the (optional) supplemental material. The checklist does NOT
477

count towards the page limit.
478

Please read the checklist guidelines carefully for information on how to answer these questions. For
479

each question in the checklist:
480

• You should answer [Yes] , [No] , or [NA] .
481

• [NA] means either that the question is Not Applicable for that particular paper or the
482

relevant information is Not Available.
483

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).
484

The checklist answers are an integral part of your paper submission. They are visible to the
485

reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
486

(after eventual revisions) with the final version of your paper, and its final version will be published
487

with the paper.
488

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
489

While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
490

proper justification is given (e.g., "error bars are not reported because it would be too computationally
491

expensive" or "we were unable to find the license for the dataset we used"). In general, answering
492

"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
493

acknowledge that the true answer is often more nuanced, so please just use your best judgment and
494

write a justification to elaborate. All supporting evidence can appear either in the main paper or the
495

supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
496

please point to the section(s) where related material for the question can be found.
497

IMPORTANT, please:
498

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
499

• Keep the checklist subsection headings, questions/answers and guidelines below.
500

• Do not modify the questions and only use the provided macros for your answers.
501

1. Claims
502

Question: Do the main claims made in the abstract and introduction accurately reflect the
503

paper’s contributions and scope?
504

Answer: [Yes]
505

Justification: Yes. The main contribution is a proof-of-concept that one should move away
506

from single point estimation for better generalization across distortions.
507

Guidelines:
508

• The answer NA means that the abstract and introduction do not include the claims
509

made in the paper.
510

• The abstract and/or introduction should clearly state the claims made, including the
511

contributions made in the paper and important assumptions and limitations. A No or
512

NA answer to this question will not be perceived well by the reviewers.
513

• The claims made should match theoretical and experimental results, and reflect how
514

much the results can be expected to generalize to other settings.
515

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
516

are not attained by the paper.
517

2. Limitations
518

Question: Does the paper discuss the limitations of the work performed by the authors?
519

Answer: [Yes]
520

Justification: Since, the scope is only a proof-of-concept, we have not considered scaling
521

to large datasets/models in this work. Scaling these ideas would require rethinking current
522

strategies and does not fit perfectly into existing framework. Moreover, the proposed
523

approach slightly more resource intensive than ReLU activation.
524

Guidelines:
525

19


---Page Break---
• The answer NA means that the paper has no limitation while the answer No means that
526

the paper has limitations, but those are not discussed in the paper.
527

• The authors are encouraged to create a separate "Limitations" section in their paper.
528

• The paper should point out any strong assumptions and how robust the results are to
529

violations of these assumptions (e.g., independence assumptions, noiseless settings,
530

model well-specification, asymptotic approximations only holding locally). The authors
531

should reflect on how these assumptions might be violated in practice and what the
532

implications would be.
533

• The authors should reflect on the scope of the claims made, e.g., if the approach was
534

only tested on a few datasets or with a few runs. In general, empirical results often
535

depend on implicit assumptions, which should be articulated.
536

• The authors should reflect on the factors that influence the performance of the approach.
537

For example, a facial recognition algorithm may perform poorly when image resolution
538

is low or images are taken in low lighting. Or a speech-to-text system might not be
539

used reliably to provide closed captions for online lectures because it fails to handle
540

technical jargon.
541

• The authors should discuss the computational efficiency of the proposed algorithms
542

and how they scale with dataset size.
543

• If applicable, the authors should discuss possible limitations of their approach to
544

address problems of privacy and fairness.
545

• While the authors might fear that complete honesty about limitations might be used by
546

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
547

limitations that aren’t acknowledged in the paper. The authors should use their best
548

judgment and recognize that individual actions in favor of transparency play an impor-
549

tant role in developing norms that preserve the integrity of the community. Reviewers
550

will be specifically instructed to not penalize honesty concerning limitations.
551

3. Theory Assumptions and Proofs
552

Question: For each theoretical result, does the paper provide the full set of assumptions and
553

a complete (and correct) proof?
554

Answer: [NA]
555

Justification: We do not consider theoretical aspects in this article. While there are interesting
556

theoretical connections, we leave it for future work.
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

Justification: We provide an anonymous link to generate all the results provided in the article.
574

Moreover, we describe all the hyper-parameters used in the appendix as well.
575

Guidelines:
576

• The answer NA means that the paper does not include experiments.
577

• If the paper includes experiments, a No answer to this question will not be perceived
578

well by the reviewers: Making the paper reproducible is important, regardless of
579

whether the code and data are provided or not.
580

• If the contribution is a dataset and/or model, the authors should describe the steps taken
581

to make their results reproducible or verifiable.
582

• Depending on the contribution, reproducibility can be accomplished in various ways.
583

For example, if the contribution is a novel architecture, describing the architecture fully
584

20


---Page Break---
might suffice, or if the contribution is a specific model and empirical evaluation, it may
585

be necessary to either make it possible for others to replicate the model with the same
586

dataset, or provide access to the model. In general. releasing code and data is often
587

one good way to accomplish this, but reproducibility can also be provided via detailed
588

instructions for how to replicate the results, access to a hosted model (e.g., in the case
589

of a large language model), releasing of a model checkpoint, or other means that are
590

appropriate to the research performed.
591

• While NeurIPS does not require releasing code, the conference does require all submis-
592

sions to provide some reasonable avenue for reproducibility, which may depend on the
593

nature of the contribution. For example
594

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
595

to reproduce that algorithm.
596

(b) If the contribution is primarily a new model architecture, the paper should describe
597

the architecture clearly and fully.
598

(c) If the contribution is a new model (e.g., a large language model), then there should
599

either be a way to access this model for reproducing the results or a way to reproduce
600

the model (e.g., with an open-source dataset or instructions for how to construct
601

the dataset).
602

(d) We recognize that reproducibility may be tricky in some cases, in which case
603

authors are welcome to describe the particular way they provide for reproducibility.
604

In the case of closed-source models, it may be that access to the model is limited in
605

some way (e.g., to registered users), but it should be possible for other researchers
606

to have some path to reproducing or verifying the results.
607

5. Open access to data and code
608

Question: Does the paper provide open access to the data and code, with sufficient instruc-
609

tions to faithfully reproduce the main experimental results, as described in supplemental
610

material?
611

Answer: [Yes]
612

Justification: We provide the code using an anonymous link at https://anonymous.
613

4open.science/r/QuantAct-2B41. The datasets are all public datasets which can be
614

downloaded.
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

Justification: We explain the experimental setting in complete detail in the article.
641

Guidelines:
642

• The answer NA means that the paper does not include experiments.
643

21


---Page Break---
• The experimental setting should be presented in the core of the paper to a level of detail
644

that is necessary to appreciate the results and make sense of them.
645

• The full details can be provided either with the code, in appendix, or as supplemental
646

material.
647

7. Experiment Statistical Significance
648

Question: Does the paper report error bars suitably and correctly defined or other appropriate
649

information about the statistical significance of the experiments?
650

Answer: [Yes]
651

Justification: The datasets used incorporate 15 kinds of distortions across 5 severity levels.
652

We report the error bars across the 15 kinds of distortions which should provide a good
653

picture of the reliability of the results.
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

Answer:[Yes]
680

Justification: Yes. we include the entire information about the compute resources in the
681

appendix.
682

Guidelines:
683

• The answer NA means that the paper does not include experiments.
684

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
685

or cloud provider, including relevant memory and storage.
686

• The paper should provide the amount of compute required for each of the individual
687

experimental runs as well as estimate the total compute.
688

• The paper should disclose whether the full research project required more compute
689

than the experiments reported in the paper (e.g., preliminary or failed experiments that
690

didn’t make it into the paper).
691

9. Code Of Ethics
692

Question: Does the research conducted in the paper conform, in every respect, with the
693

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
694

Answer: [Yes]
695

Justification: We have tried to maintain the double-blind policy to the maximum extent
696

possible.
697

Guidelines:
698

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
699

• If the authors answer No, they should explain the special circumstances that require a
700

deviation from the Code of Ethics.
701

22


---Page Break---
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
702

eration due to laws or regulations in their jurisdiction).
703

10. Broader Impacts
704

Question: Does the paper discuss both potential positive societal impacts and negative
705

societal impacts of the work performed?
706

Answer: [Yes]
707

Justification: In this article we propose a novel way to allow robustness against distortions.
708

We do not expect any negative societal impacts. There could be positive societal impact
709

since this can potentially stop hallucinations/bias of the ML models.
710

Guidelines:
711

• The answer NA means that there is no societal impact of the work performed.
712

• If the authors answer NA or No, they should explain why their work has no societal
713

impact or why the paper does not address societal impact.
714

• Examples of negative societal impacts include potential malicious or unintended uses
715

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
716

(e.g., deployment of technologies that could make decisions that unfairly impact specific
717

groups), privacy considerations, and security considerations.
718

• The conference expects that many papers will be foundational research and not tied
719

to particular applications, let alone deployments. However, if there is a direct path to
720

any negative applications, the authors should point it out. For example, it is legitimate
721

to point out that an improvement in the quality of generative models could be used to
722

generate deepfakes for disinformation. On the other hand, it is not needed to point out
723

that a generic algorithm for optimizing neural networks could enable people to train
724

models that generate Deepfakes faster.
725

• The authors should consider possible harms that could arise when the technology is
726

being used as intended and functioning correctly, harms that could arise when the
727

technology is being used as intended but gives incorrect results, and harms following
728

from (intentional or unintentional) misuse of the technology.
729

• If there are negative societal impacts, the authors could also discuss possible mitigation
730

strategies (e.g., gated release of models, providing defenses in addition to attacks,
731

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
732

feedback over time, improving the efficiency and accessibility of ML).
733

11. Safeguards
734

Question: Does the paper describe safeguards that have been put in place for responsible
735

release of data or models that have a high risk for misuse (e.g., pretrained language models,
736

image generators, or scraped datasets)?
737

Answer: [NA]
738

Justification: We do not use any datasets/models that have high risk for misuse.
739

Guidelines:
740

• The answer NA means that the paper poses no such risks.
741

• Released models that have a high risk for misuse or dual-use should be released with
742

necessary safeguards to allow for controlled use of the model, for example by requiring
743

that users adhere to usage guidelines or restrictions to access the model or implementing
744

safety filters.
745

• Datasets that have been scraped from the Internet could pose safety risks. The authors
746

should describe how they avoided releasing unsafe images.
747

• We recognize that providing effective safeguards is challenging, and many papers do
748

not require this, but we encourage authors to take this into account and make a best
749

faith effort.
750

12. Licenses for existing assets
751

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
752

the paper, properly credited and are the license and terms of use explicitly mentioned and
753

properly respected?
754

Answer: [Yes]
755

Justification: We have cited the original articles of all the datasets/models we use in the
756

article. We have ensured that these can be used with credit for academic purposes.
757

Guidelines:
758

• The answer NA means that the paper does not use existing assets.
759

• The authors should cite the original paper that produced the code package or dataset.
760

23


---Page Break---
• The authors should state which version of the asset is used and, if possible, include a
761

URL.
762

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
763

• For scraped data from a particular source (e.g., website), the copyright and terms of
764

service of that source should be provided.
765

• If assets are released, the license, copyright information, and terms of use in the
766

package should be provided. For popular datasets, paperswithcode.com/datasets
767

has curated licenses for some datasets. Their licensing guide can help determine the
768

license of a dataset.
769

• For existing datasets that are re-packaged, both the original license and the license of
770

the derived asset (if it has changed) should be provided.
771

• If this information is not available online, the authors are encouraged to reach out to
772

the asset’s creators.
773

13. New Assets
774

Question: Are new assets introduced in the paper well documented and is the documentation
775

provided alongside the assets?
776

Answer: [Yes]
777

Justification:
We
share
the
code
at
https://anonymous.4open.science/r/
778

QuantAct-2B41.
779

Guidelines:
780

• The answer NA means that the paper does not release new assets.
781

• Researchers should communicate the details of the dataset/code/model as part of their
782

submissions via structured templates. This includes details about training, license,
783

limitations, etc.
784

• The paper should discuss whether and how consent was obtained from people whose
785

asset is used.
786

• At submission time, remember to anonymize your assets (if applicable). You can either
787

create an anonymized URL or include an anonymized zip file.
788

14. Crowdsourcing and Research with Human Subjects
789

Question: For crowdsourcing experiments and research with human subjects, does the paper
790

include the full text of instructions given to participants and screenshots, if applicable, as
791

well as details about compensation (if any)?
792

Answer: [NA]
793

Justification: The paper does not involve crowdsourcing nor research with human subjects.
794

Guidelines:
795

• The answer NA means that the paper does not involve crowdsourcing nor research with
796

human subjects.
797

• Including this information in the supplemental material is fine, but if the main contribu-
798

tion of the paper involves human subjects, then as much detail as possible should be
799

included in the main paper.
800

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
801

or other labor should be paid at least the minimum wage in the country of the data
802

collector.
803

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
804

Subjects
805

Question: Does the paper describe potential risks incurred by study participants, whether
806

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
807

approvals (or an equivalent approval/review based on the requirements of your country or
808

institution) were obtained?
809

Answer: [NA]
810

Justification: the paper does not involve crowdsourcing nor research with human subjects.
811

Guidelines:
812

• The answer NA means that the paper does not involve crowdsourcing nor research with
813

human subjects.
814

• Depending on the country in which research is conducted, IRB approval (or equivalent)
815

may be required for any human subjects research. If you obtained IRB approval, you
816

should clearly state this in the paper.
817

24


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
818

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
819

guidelines for their institution.
820

• For initial submissions, do not include any information that would break anonymity (if
821

applicable), such as the institution conducting the review.
822

25


---Page Break---
