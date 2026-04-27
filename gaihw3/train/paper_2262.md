Robust Guided Diffusion for Offline Black-box
Optimization

Anonymous Author(s)
Affiliation
Address
email

Abstract

Offline black-box optimization aims to maximize a black-box function using an
1

offline dataset of designs and their measured properties. Two main approaches have
2

emerged: the forward approach, which learns a mapping from input to its value,
3

thereby acting as a proxy to guide optimization, and the inverse approach, which
4

learns a mapping from value to input for conditional generation. (a) Although
5

proxy-free (classifier-free) diffusion shows promise in robustly modeling the inverse
6

mapping, it lacks explicit guidance from proxies, essential for generating high-
7

performance samples beyond the training distribution. Therefore, we propose
8

proxy-enhanced sampling which utilizes the explicit guidance from a trained proxy
9

to bolster proxy-free diffusion with enhanced sampling control. (b) Yet, the trained
10

proxy is susceptible to out-of-distribution issues. To address this, we devise the
11

module diffusion-based proxy refinement, which seamlessly integrates insights from
12

proxy-free diffusion back into the proxy for refinement. To sum up, we propose
13

Robust Guided Diffusion for Offline Black-box Optimization (RGD), combining the
14

advantages of proxy (explicit guidance) and proxy-free diffusion (robustness) for
15

effective conditional generation. RGD achieves state-of-the-art results on various
16

design-bench tasks, underscoring its efficacy. Our code is here.
17

1
Introduction
18

Creating new objects to optimize specific properties is a ubiquitous challenge that spans a multitude
19

of fields, including material science, robotic design, and genetic engineering. Traditional methods
20

generally require interaction with a black-box function to generate new designs, a process that could
21

be financially burdensome and potentially perilous [1, 2]. Addressing this, recent research endeavors
22

have pivoted toward a more relevant and practical context, termed offline black-box optimization
23

(BBO) [3, 4]. In this context, the goal is to maximize a black-box function exclusively utilizing an
24

offline dataset of designs and their measured properties.
25

There are two main approaches for this task: the forward approach and the reverse approach. The
26

forward approach entails training a deep neural network (DNN), parameterized as Jϕ(·), using the
27

offline dataset. Once trained, the DNN acts as a proxy and provides explicit gradient guidance to
28

enhance existing designs. However, this technique is susceptible to the out-of-distribution (OOD)
29

issue, leading to potential overestimation of unseen designs and resulting in adversarial solutions [5].
30

The reverse approach aims to learn a mapping from property value to input. Inputting a high value
31

into this mapping directly yields a high-performance design. For example, MINs [6] adopts GAN [7]
32

to model this inverse mapping, and demonstrate some success. Recent works [4] have applied
33

proxy-free diffusion1 [8], parameterized by θ, to model this mapping, which proves its efficacy over
34

1Classifier-free diffusion is for classification and adapted to proxy-free diffusion to generalize to regression.

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
other generative models. Proxy-free diffusion employs a score predictor ˜sθ(·, ·, ω). This represents a
35

linear combination of conditional and unconditional scores, modulated by a strength parameter ω to
36

balance condition and diversity in the sampling process. This guidance significantly diverges from
37

proxy (classifier) diffusion that interprets scores as classifier gradients and thus generates adversarial
38

solutions. Such a distinction grants proxy-free diffusion its inherent robustness in generating samples.
39

2.0
1.5
1.0
0.5
0.0
0.5
1.0
1.5
2.0
xd1

2.0

1.5

1.0

0.5

0.0

0.5

1.0

1.5

2.0

xd2

Negative Rosenbrock Function
Initial points
Proxy-free diffusion
Proxy-enhanced sampling

3600

3000

2400

1800

1200

600

0

Figure 1: Motivation of explicit proxy guidance.

Nevertheless, proxy-free diffusion, initially de-
40

signed for in-distribution generation, such as
41

synthesizing specific image categories, faces
42

limitations in offline BBO. Particularly, it strug-
43

gles to generate high-performance samples that
44

exceed the training distribution due to the lack
45

of explicit guidance2. Consider, for example,
46

the optimization of a two-dimensional variable
47

(xd1, xd2) to maximize the negative Rosenbrock
48

function [9]: y(xd1, xd2) = −(1 −xd1)2 −
49

100(xd2 −x2
d1)2, as depicted in Figure 1. The
50

objective is to steer the initial points (indi-
51

cated in pink) towards the high-performance
52

region (highlighted in yellow). While proxy-
53

free diffusion can nudge the initial points closer to this high-performance region, the generated points
54

(depicted in blue) fail to reach the high-performance region due to its lack of explicit proxy guidance.
55

To address this challenge, we introduce a proxy-enhanced sampling module as illustrated in Fig-
56

ure 2(a). It incorporates the explicit guidance from the proxy Jϕ(x) into proxy-free diffusion to
57

enable enhanced control over the sampling process. This module hinges on the strategic optimization
58

of the strength parameter ω to achieve a better balance between condition and diversity, per reverse
59

diffusion step. This incorporation not only preserves the inherent robustness of proxy-free diffusion
60

but also leverages the explicit proxy guidance, thereby enhancing the overall conditional generation
61

efficacy. As illustrated in Figure 1, samples (depicted in red) generated via proxy-enhanced sampling
62

are more effectively guided towards, and often reach, the high-performance area (in yellow).
63

Forward diffusion

Reverse diffusion
Final design

Diffusion-based

proxy refinement

Proxy-enhanced sampling (optimizing    )

(a)

(b)
Diffusion distribution
Proxy

Probability flow ODE

...

Figure 2: Overall of RGD.

Yet, the trained proxy is susceptible to out-of-
64

distribution (OOD) issues. To address this, we
65

devise a module diffusion-based proxy refinement
66

as detailed in Figure 2(b). This module seamlessly
67

integrates insights from proxy-free diffusion into
68

the proxy Jϕ(x) for refinement. Specifically, we
69

generate a diffusion distribution pθ(y|ˆx) on adver-
70

sarial samples ˆx, using the associated probability
71

flow ODE 3. This distribution is derived indepen-
72

dently of a proxy, thereby exhibiting greater ro-
73

bustness than the proxy distribution on adversarial
74

samples. Subsequently, we calculate the Kullback-
75

Leibler divergence between the two distributions
76

on adversarial samples, and use this divergence
77

minimization as a regularization strategy to fortify
78

the proxy’s robustness and reliability.
79

To sum up, we propose Robust Guided Diffusion for Offline Black-box Optimization (RGD), a novel
80

framework that combines the advantages of proxy (explicit guidance) and proxy-free diffusion (ro-
81

bustness) for effective conditional generation. Our contributions are three-fold:
82

• We propose a proxy-enhanced sampling module which incorporates proxy guidance into proxy-free
83

diffusion to enable enhanced sampling control.
84

• We further develop diffusion-based proxy refinement which integrates insights from proxy-free
85

diffusion back into the proxy for refinement.
86

• RGD delivers state-of-the-art performance on various design-bench tasks, emphasizing its efficacy.
87

2Proxy-free diffusion cannot be interpreted as a proxy and thus does not provide explicit guidance [8].
3Ordinary Differential Equation

2


---Page Break---
2
Preliminaries
88

2.1
Offline Black-box Optimization
89

Offline black-box optimization (BBO) aims to maximize a black-box function with an offline dataset.
90

Imagine a design space as X = Rd, where d is the design dimension. The offline BBO [3] is:
91

x∗= arg max
x∈X J(x).
(1)

In this equation, J(·) is the unknown objective function, and x ∈X is a possible design. In this
92

context, there is an offline dataset, D, that consists of pairs of designs and their measured properties.
93

Specifically, each x denotes a particular design, like the size of a robot, while y indicates its related
94

metric, such as its speed.
95

A common approach gradient ascent fits a proxy distribution pϕ(y|x) = N(Jϕ(x), σϕ(x)) to the
96

offline dataset where ϕ denote the proxy parameters:
97

arg min
ϕ E(x,y)∈D[−log pϕ(y|x)].

= arg min
ϕ E(x,y)∈D log(
√

2πσϕ(x)) + (y −Jϕ(x))2

2σ2
ϕ(x)
.
(2)

For the sake of consistency with terminology used in the forthcoming subsection on guided diffusion,
98

we will refer to pϕ(·|·) as the proxy distribution and Jϕ(·) as the proxy. Subsequently, this approach
99

performs gradient ascent with Jϕ(x), leading to high-performance designs x∗:
100

xτ+1 = xτ + η∇xJϕ(x)|x=xτ ,
for τ ∈[0, M −1],
(3)

converging to xM after M steps. However, this method suffers from the out-of-distribution issue
101

where the proxy predicts values that are notably higher than the actual values.
102

2.2
Diffusion Models
103

Diffusion models, a type of latent variable models, progressively introduce Gaussian noise to data in
104

the forward process, while the reverse process aims to iteratively remove this noise through a learned
105

score estimator. In this work, we utilize continuous time diffusion models governed by a stochastic
106

differential equation (SDE), as presented in [10]. The forward SDE is formulated as:
107

dx = f(x, t)dt + g(t)dw.
(4)

where f(·, t) : Rd →Rd represents the drift coefficient, g(·) : R →R denotes the diffusion
108

coefficient and w is the standard Wiener process. This SDE transforms data distribution into noise
109

distribution. The reverse SDE is:
110

dx =

f(x, t) −g(t)2∇x log p(x)

dt + g(t)d ¯
w,
(5)

with ∇x log p(x) representing the score of the marginal distribution at time t, and ¯
w symbolizing the
111

reverse Wiener process. The score function ∇x log p(x) is estimated using a time-dependent neural
112

network sθ(xt, t), enabling us to transform noise into samples. For simplicity, we will use sθ(xt),
113

implicitly including the time dependency t.
114

2.3
Guided Diffusion
115

Guided diffusion seeks to produce samples with specific desirable attributes, falling into two cate-
116

gories: proxy diffusion [11] and proxy-free diffusion [8]. While these were initially termed classifier
117

diffusion and classifier-free diffusion in classification tasks, we have renamed them to proxy diffu-
118

sion and proxy-free diffusion, respectively, to generalize to our regression context. Proxy diffusion
119

combines the model’s score estimate with the gradient from the proxy distribution, providing explicit
120

guidance. However, it can be interpreted as a gradient-based adversarial attack.
121

Proxy-free guidance, not dependent on proxy gradients, enjoys an inherent robustness of the sampling
122

process. Particularly, it models the score as a linear combination of an unconditional and a conditional
123

score. A unified neural network sθ(xt, y) parameterizes both score types. The score sθ(xt, y)
124

3


---Page Break---
approximates the gradient of the log probability ∇xt log p(xt|y), i.e., the conditional score, while
125

sθ(xt) estimates the gradient of the log probability ∇xt log p(xt), i.e., the unconditional score. The
126

score function follows:
127

˜sθ(xt, y, ω) = (1 + ω)sθ(xt, y) −ωsθ(xt).
(6)

Within this context, the strength parameter ω specifies the generation’s adherence to the condition
128

y, which is set to the maximum value ymax in the offline dataset following [4]. Optimization of ω
129

balances the condition and diversity. Lower ω values increase sample diversity at the expense of
130

conformity to y, and higher values do the opposite.
131

3
Method
132

In this section, we present our method RGD, melding the strengths of proxy and proxy-free diffu-
133

sion for effective conditional generation. Firstly, we describe a newly developed module termed
134

proxy-enhanced sampling. It integrates explicit proxy guidance into proxy-free diffusion to enable
135

enhanced sampling control, as detailed in Section 3.1. Subsequently, we explore diffusion-based
136

proxy refinement which incorporates insights gleaned from proxy-free diffusion back into the proxy,
137

further elaborated in Section 3.2. The overall algorithm is shown in Algorithm 1.
138

3.1
Proxy-enhanced Sampling
139

Algorithm 1 Robust Guided Diffusion for Offline BBO
Input: offline dataset D, # of diffusion steps T.

1: Train proxy distribution pϕ(y|x) on D by Eq. (2).
2: Train proxy-free diffusion model sθ(xt, y) on D.
3: /*Diffusion-based proxy refinement */
4: Identify adversarial samples via grad ascent.
5: Compute diffusion distribution pθ(y|ˆx) by Eq. (12).
6: Compute KL divergence loss as per Eq. (13).
7: Refine proxy distribution pϕ(y|x) through Eq. (15).
8: /*Proxy-enhanced sampling */
9: Begin with xT ∼N(0, I)
10: for t = T −1 to 0 do
11:
Derive the score ˜sθ(xt+1, y, ω) from Eq. (6).
12:
Update xt+1 to xt(ω) using ω as per Eq. (7).
13:
Optimize ω to ˆω following Eq. (8).
14:
Finalize the update of xt with ˆω via Eq. (9).
15: end for
16: Return x∗= x0

As discussed in Section 2.3, proxy-
140

free diffusion trains an unconditional
141

model and conditional models. Although
142

proxy-free diffusion can generate samples
143

aligned with most conditions, it tradition-
144

ally lacks control due to the absence of
145

an explicit proxy. This is particularly sig-
146

nificant in offline BBO where we aim to
147

obtain samples beyond the training dis-
148

tribution. Therefore, we require explicit
149

proxy guidance to achieve enhanced sam-
150

pling control. This module is outlined in
151

Algorithm 1, Line 8- Line 16.
152

Optimization of ω. Directly updating
153

the design xt with proxy gradient suffers
154

from the OOD issue and determining a
155

proper condition y necessitates the man-
156

ual adjustment of multiple hyperparame-
157

ters [6]. Thus, we propose to introduce
158

proxy guidance by only optimizing the strength parameter ω within ˜sθ(xt, y, ω) in Eq. (6). As
159

discussed in Section 2.3, the parameter ω balances the condition and diversity, and an optimized ω
160

could achieve a better balance in the sampling process, leading to more effective generation.
161

Enhanced Sampling. With the score function, the update of a noisy sample xt+1 is computed as:
162

xt(ω) = solver(xt+1, ˜sθ(xt+1, y, ω)),
(7)

where the solver is the second-order Heun solver [12], chosen for its enhanced accuracy through a
163

predictor-corrector method. A proxy is then trained to predict the property of noise xt at time step t,
164

denoted as Jϕ(xt, t). By maximizing Jϕ(xt(ω), t) with respect to ω, we can incorporate the explicit
165

proxy guidance into proxy-free diffusion to enable enhanced sampling control in the balance between
166

condition and diversity. This maximization process is:
167

ˆω = ω + η ∂Jϕ(xt(ω), t)

∂ω
.
(8)

where η denotes the learning rate. We leverage the automatic differentiation capabilities of Py-
168

Torch [13] to efficiently compute the above derivatives within the context of the solver’s operation.
169

The optimized ˆω then updates the noisy sample xt+1 through:
170

xt = solver(xt+1, ˜sθ(xt+1, y, ˆω)).
(9)

4


---Page Break---
This process iteratively denoises xt, utilizing it in successive steps to progressively approach x0,
171

which represents the final high-scoring design x∗.
172

Proxy Training. Notably, Jϕ(xt, t) can be directly derived from the proxy Jϕ(x), the mean of the
173

proxy distribution pϕ(·|x) in Eq. (2). This distribution is trained exclusively at the initial time step
174

t = 0, eliminating the need for training across time steps. To achieve this derivation, we reverse the
175

diffusion from xt back to x0 using the formula:
176

x0 = xt + sθ(xt) · σ(t)2

µ(t)
,
(10)

where sθ(xt) is the estimated unconditional score at time step t, and σ(t)2 and µ(t) are the variance
177

and mean functions of the perturbation kernel at time t, as detailed in equations (32-33) in [10].
178

Consequently, we express
179

Jϕ(xt, t) = Jϕ

xt + sθ(xt) · σ(t)2

µ(t)


.
(11)

This formulation allows for the optimization of the strength parameter ω via Eq. (8). For simplicity,
180

we will refer to Jϕ(·) in subsequent discussions.
181

3.2
Diffusion-based Proxy Refinement
182

In the proxy-enhanced sampling module, the proxy Jϕ(·) is employed to update the parameter ω
183

to enable enhanced control. However, Jϕ(·) may still be prone to the OOD issue, especially on
184

adversarial samples [5]. To address this, we refine the proxy by using insights from proxy-free
185

diffusion. The procedure of this module is specified in Algorithm 1, Lines 3-7.
186

Diffusion Distribution. Adversarial samples are identified by gradient ascent on the proxy as per
187

Eq. (3) to form the distribution q(x). Consequently, these samples are vulnerable to the proxy
188

distribution. Conversely, the proxy-free diffusion, which functions without depending on a proxy,
189

inherently offers greater resilience against these samples, thus producing a more robust distribution.
190

For an adversarial sample ˆx ∼q(x), we compute pθ(ˆx), pθ(ˆx|y) via the probability flow ODE, and
191

p(y) through Gaussian kernel-density estimation. The diffusion distribution regarding y is derived as:
192

pθ(y|ˆx) = pθ(ˆx|y) · p(y)

pθ(ˆx)
,
(12)

which demonstrates inherent robustness over the proxy distribution pϕ(y|ˆx). Yet, directly applying
193

diffusion distribution to design optimization by gradient ascent is computationally intensive and
194

potentially unstable due to the demands of reversing ODEs and scoring steps.
195

Proxy Refinement. We opt for a more feasible approach: refine the proxy distribution pϕ(y|ˆx) =
196

N(Jϕ(ˆx), σϕ(ˆx)) by minimizing its distance to the diffusion distribution pθ(y|ˆx). The distance is
197

quantified by the Kullback-Leibler (KL) divergence:
198

Eq[D(pϕ||pθ)] = Eq(x)

Z
pϕ(y|ˆx) log
pϕ(y|ˆx)

pθ(y|ˆx)


dy.
(13)

We avoid the parameterization trick for minimizing this divergence as it necessitates backpropagation
199

through pθ(y|ˆx), which is prohibitively expensive. Instead, for the sample ˆx, the gradient of the KL
200

divergence D(pϕ||pθ) with respect to the proxy parameters ϕ is computed as:
201

Epϕ(y|ˆx)

d log pϕ(y|ˆx)

dϕ


1 + log pϕ(y|ˆx)

pθ(y|ˆx)


.
(14)

Complete derivations are in Appendix A. The KL divergence then acts as regularization in our loss L:
202

L(ϕ, α) = ED[−log pϕ(y|x)] + αEq(x)[D(pϕ||pθ)],
(15)

where D is the training dataset and α is a hyperparameter. We propose to optimize α based on the
203

validation loss via bi-level optimization as detailed in Appendix B.
204

4
Experiments
205

In this section, we conduct comprehensive experiments to evaluate our method’s performance.
206

5


---Page Break---
4.1
Benchmarks
207

Tasks. Our experiments encompass a variety of tasks, split into continuous and discrete categories.
208

The continuous category includes four tasks: (1) Superconductor (SuperC) 4: The objective here
209

is to engineer a superconductor composed of 86 continuous elements. The goal is to enhance the
210

critical temperature using 17, 010 design samples. This task is based on the dataset from [1]. (2) Ant
211

Morphology (Ant): In this task, the focus is on developing a quadrupedal ant robot, comprising 60
212

continuous parts, to augment its crawling velocity. It uses 10, 004 design instances from the dataset
213

in [3, 14]. (3) D’Kitty Morphology (D’Kitty): Similar to Ant Morphology, this task involves the
214

design of a quadrupedal D’Kitty robot with 56 components, aiming to improve its crawling speed
215

with 10, 004 designs, as described in [3, 15]. (4) Rosenbrock (Rosen): The aim of this task is to
216

optimize a 60-dimension continuous vector to maximize the Rosenbrock black-box function. It uses
217

50000 designs from the low-scoring part [9].
218

For the discrete category, we explore three tasks: (1) TF Bind 8 (TF8): The goal is to identify an
219

8-unit DNA sequence that maximizes binding activity. This task uses 32, 898 designs and is detailed
220

in [16]. (2) TF Bind 10 (TF10): Similar to TF8, but with a 10-unit DNA sequence and a larger pool
221

of 50, 000 samples, as described in [16]. (3) Neural Architecture Search (NAS): This task focuses
222

on discovering the optimal neural network architecture to improve test accuracy on the CIFAR-10
223

dataset, using 1, 771 designs [17].
224

Evaluation. In this study, we utilize the oracle evaluation from design-bench [3]. Adhering to this
225

established protocol, we analyze the top 128 promising designs from each method. The evaluation
226

metric employed is the 100th percentile normalized ground-truth score, calculated using the formula
227

yn =
y−ymin
ymax−ymin , where ymin and ymax signify the lowest and highest scores respectively in the
228

comprehensive, yet unobserved, dataset. In addition to these scores, we provide an overview of each
229

method’s effectiveness through the mean and median rankings across all evaluated tasks. Notably,
230

the best design discovered in the offline dataset, designated as D(best), is also included for reference.
231

For further details on the 50th percentile (median) scores, please refer to Appendix C.
232

4.2
Comparison Methods
233

Our approach is evaluated against two primary groups of baseline methods: forward and inverse
234

approaches. Forward approaches enhance existing designs through gradient ascent. This includes: (i)
235

Grad: utilizes simple gradient ascent on current designs for new creations; (ii) ROMA [18]: imple-
236

ments smoothness regularization on proxies; (iii) COMs [5]: applies regularization to assign lower
237

scores to adversarial designs; (iv) NEMO [19]: bridges the gap between proxy and actual functions
238

using normalized maximum likelihood; (v) BDI [20]: utilizes both forward and inverse mappings to
239

transfer knowledge from offline datasets to the designs; (vi) IOM [21]: ensures consistency between
240

representations of training datasets and optimized designs.
241

Inverse approaches focus on learning a mapping from a design’s property value back to its input.
242

High property values are input into this inverse mapping to yield enhanced designs. This includes: (i)
243

CbAS [22]: CbAS employs a VAE model to implicitly implement the inverse mapping. It gradually
244

tunes its distribution toward higher scores by raising the scoring threshold. This process can be
245

interpreted as incrementally increasing the conditional score within the inverse mapping framework.
246

(ii) Autofocused CbAS (Auto.CbAS) [23]: adopts importance sampling for retraining a regression
247

model based on CbAS. (iii) MIN [6]: maps scores to designs via a GAN model and explore this
248

mapping for optimal designs. (iv) BONET [24]: introduces an autoregressive model for sampling
249

high-scoring designs. (v) DDOM [4]: utilizes proxy-free diffusion to model the inverse mapping.
250

Traditional methods as detailed in [3] are also considered: (i) CMA-ES [25]: modifies the covariance
251

matrix to progressively shift the distribution towards optimal designs; (ii) BO-qEI [26]: implements
252

Bayesian optimization to maximize the proxy and utilizes the quasi-Expected-Improvement acqui-
253

sition function for design suggestion, labeling designs using the proxy; (iii) REINFORCE [27]:
254

enhances the input space distribution using the learned proxy model.
255

4Previously, the task oracle exhibited inconsistencies, producing varying outputs for identical inputs. This
issue has now been rectified by the development team.

6


---Page Break---
4.3
Experimental Configuration
256

In alignment with the experimental protocols established in [3, 20], we have tailored our training
257

methodologies for all approaches, except where specified otherwise. For methods such as BO-qEI,
258

CMA-ES, REINFORCE, CbAS, and Auto.CbAS that do not utilize gradient ascent, we base our
259

approach on the findings reported in [3]. We adopted T = 1000 diffusion sampling steps, set the
260

condition y to ymax, and initial strength ω as 2 in line with [4]. To ensure reliability and consistency in
261

our comparative analysis, each experimental setting was replicated across 8 independent runs, unless
262

stated otherwise, with the presentation of both mean values and standard errors. These experiments
263

were conducted using a NVIDIA GeForce V100 GPU. We’ve detailed the computational overhead of
264

our approach in Appendix D to provide a comprehensive view of its practicality.
265

Table 1: Results (maximum normalized score) on continuous tasks.

Method
Superconductor
Ant Morphology
D’Kitty Morphology
Rosenbrock
D(best)
0.399
0.565
0.884
0.518
BO-qEI
0.402 ± 0.034
0.819 ± 0.000
0.896 ± 0.000
0.772 ± 0.012
CMA-ES
0.465 ± 0.024
1.214 ± 0.732
0.724 ± 0.001
0.470 ± 0.026
REINFORCE
0.481 ± 0.013
0.266 ± 0.032
0.562 ± 0.196
0.558 ± 0.013
Grad
0.490 ± 0.009
0.932 ± 0.015
0.930 ± 0.002
0.701 ± 0.092
COMs
0.504 ± 0.022
0.818 ± 0.017
0.905 ± 0.017
0.672 ± 0.075
ROMA
0.507 ± 0.013
0.898 ± 0.029
0.928 ± 0.007
0.663 ± 0.072
NEMO
0.499 ± 0.003
0.956 ± 0.013
0.953 ± 0.010
0.614 ± 0.000
IOM
0.524 ± 0.022
0.929 ± 0.037
0.936 ± 0.008
0.712 ± 0.068
BDI
0.513 ± 0.000
0.906 ± 0.000
0.919 ± 0.000
0.630 ± 0.000
CbAS
0.503 ± 0.069
0.876 ± 0.031
0.892 ± 0.008
0.702 ± 0.008
Auto.CbAS
0.421 ± 0.045
0.882 ± 0.045
0.906 ± 0.006
0.721 ± 0.007
MIN
0.499 ± 0.017
0.445 ± 0.080
0.892 ± 0.011
0.702 ± 0.074
BONET
0.422 ± 0.019
0.925 ± 0.010
0.941 ± 0.001
0.780 ± 0.009
DDOM
0.495 ± 0.012
0.940 ± 0.004
0.935 ± 0.001
0.789 ± 0.003
RGD
0.515 ± 0.011
0.968 ± 0.006
0.943 ± 0.004
0.797 ± 0.011

Table 2: Results (maximum normalized score) on discrete tasks & ranking on all tasks.

Method
TF Bind 8
TF Bind 10
NAS
Rank Mean
Rank Median
D(best)
0.439
0.467
0.436
BO-qEI
0.798 ± 0.083
0.652 ± 0.038
1.079 ± 0.059
9.1/15
11/15
CMA-ES
0.953 ± 0.022
0.670 ± 0.023
0.985 ± 0.079
7.3/15
4/15
REINFORCE
0.948 ± 0.028
0.663 ± 0.034
−1.895 ± 0.000
11.3/15
14/15
Grad
0.872 ± 0.062
0.646 ± 0.052
0.624 ± 0.102
9.0/15
10/15
COMs
0.517 ± 0.115
0.613 ± 0.003
0.783 ± 0.029
10.3/15
10/15
ROMA
0.927 ± 0.033
0.676 ± 0.029
0.927 ± 0.071
6.1/15
6/15
NEMO
0.942 ± 0.003
0.708 ± 0.022
0.737 ± 0.010
5.3/15
5/15
IOM
0.823 ± 0.130
0.650 ± 0.042
0.559 ± 0.081
7.4/15
6/15
BDI
0.870 ± 0.000
0.605 ± 0.000
0.722 ± 0.000
9.6/15
9/15
CbAS
0.927 ± 0.051
0.651 ± 0.060
0.683 ± 0.079
8.7/15
8/15
Auto.CbAS
0.910 ± 0.044
0.630 ± 0.045
0.506 ± 0.074
10.3/15
10/15
MIN
0.905 ± 0.052
0.616 ± 0.021
0.717 ± 0.046
10.4/15
10/15
BONET
0.913 ± 0.008
0.621 ± 0.030
0.724 ± 0.008
7.7/15
8/15
DDOM
0.957 ± 0.006
0.657 ± 0.006
0.745 ± 0.070
4.9/15
5/15
RGD
0.974 ± 0.003
0.694 ± 0.018
0.825 ± 0.063
2.0/15
2/15

4.4
Results and Analysis
266

In Tables 1 and 2, we showcase our experimental results for both continuous and discrete tasks.
267

To clearly differentiate among the various approaches, distinct lines separate traditional, forward,
268

and inverse approaches within the tables For every task, algorithms performing within a standard
269

deviation of the highest score are emphasized by bolding following [5].
270

We make the following observations. (1) As highlighted in Table 2, RGD not only achieves the top
271

rank but also demonstrates the best performance in six out of seven tasks, emphasizing the robustness
272

and superiority of our method. (2) RGD outperforms the VAE-based CbAS, the GAN-based MIN
273

7


---Page Break---
and the Transformer-based BONET. This result highlights the superiority of diffusion models in
274

modeling inverse mappings compared to other generative approaches. (3) Upon examining TF
275

Bind 8, we observe that the average rankings for forward and inverse methods stand at 10.3 and
276

6.0, respectively. In contrast, for TF Bind 10, both methods have the same average ranking of 8.7,
277

indicating no advantage. This notable advantage of inverse methods in TF Bind 8 implies that the
278

relatively smaller design space of TF Bind 8 (48) facilitates easier inverse mapping, as opposed to the
279

more complex space in TF Bind 10 (410). (4) RGD’s performance is less impressive on NAS, where
280

designs are encoded as 64-length sequences of 5-category one-hot vectors. This may stem from
281

the design-bench’s encoding not fully capturing the sequential and hierarchical aspects of network
282

architectures, affecting the efficacy of inverse mapping modeling.
283

Table 3: Ablation studies on RGD.

Task
D
RGD
w/o proxy-e
w/o diffusion-b r
direct grad update
SuperC
86
0.515 ± 0.011
0.495 ± 0.012
0.502 ± 0.005
0.456 ± 0.002
Ant
60
0.968 ± 0.006
0.940 ± 0.004
0.961 ± 0.011
−0.006 ± 0.003
D’Kitty
56
0.943 ± 0.004
0.935 ± 0.001
0.939 ± 0.003
0.714 ± 0.001
Rosen
60
0.797 ± 0.011
0.789 ± 0.003
0.813 ± 0.005
0.241 ± 0.283
TF8
8
0.974 ± 0.003
0.957 ± 0.007
0.960 ± 0.006
0.905 ± 0.000
TF10
10
0.694 ± 0.018
0.657 ± 0.006
0.667 ± 0.009
0.672 ± 0.018
NAS
64
0.825 ± 0.063
0.745 ± 0.070
0.717 ± 0.032
0.718 ± 0.032

4.5
Ablation Studies
284

In this section, we present a series of ablation studies to scrutinize the individual contributions of
285

distinct components in our methodology. We employ our proposed approach as a benchmark and
286

methodically exclude key modules, such as the proxy-enhanced sampling and diffusion-based proxy
287

refinement, to assess their influence on performance. These variants are denoted as w/o proxy-e and
288

w/o diffusion-b r. Additionally, we explore the strategy of directly performing gradient ascent on
289

the diffusion intermediate state, referred to as direct grad update. The results from these ablation
290

experiments are detailed in Table 3.
291

Our analysis reveals that omitting either module results in a decrease in performance, thereby affirming
292

the importance of each component. The w/o diffusion-b r variant generally surpasses w/o proxy-e,
293

highlighting the utility of the proxy-enhanced sampling even with a basic proxy setup. Conversely,
294

direct grad update tends to produce subpar results across tasks, likely attributable to the proxy’s
295

limitations in handling out-of-distribution samples, leading to suboptimal design optimizations.
296

0
100
200
300
400
500
600
700
800
900
1000

Diffusion step t

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

Strength ratio 
/
0

Ant
TF10

Figure 3: Dynamics of strength ratio ω/ω0.

To further dive into the proxy-enhanced sam-
297

pling module, we visualize the strength ra-
298

tio ω/ω0—where ω0 represents the initial
299

strength—across diffusion steps t. This analysis
300

is depicted in Figure 3 for two specific tasks:
301

Ant and TF10. We observe a pattern of initial
302

decrease followed by an increase in ω across
303

both tasks. This pattern can be interpreted as
304

follows: The decrease in ω facilitates the genera-
305

tion of a more diverse set of samples, enhancing
306

exploratory capabilities. Subsequently, the in-
307

crease in ω signifies a shift towards integrating
308

high-performance features into the sample gen-
309

eration. Within this context, conditioning on
310

the maximum y is not aimed at achieving the
311

dataset’s maximum but at enriching samples with high-scoring attributes. Overall, this adjustment of
312

ω effectively balances between generating novel solutions and honing in on high-quality ones.
313

In addition, we visualize the proxy distribution alongside the diffusion distribution for a sample ˆx
314

from the Ant task in Figure 4, to substantiate the efficacy of diffusion-based proxy refinement. The
315

proxy distribution significantly overestimates the ground truth, whereas the diffusion distribution
316

closely aligns with it, demonstrating the robustness of diffusion distribution. For a more quantitative
317

8


---Page Break---
analysis, we compute the expectation of both distributions and compare them with the ground
318

truth. The mean of the diffusion distribution is calculated as Epθ(y|ˆx)[y] = Epϕ(y|ˆx)
h
pθ(y|ˆx)
pϕ(y|ˆx)y
i
.
319

0.6
0.8
1.0
1.2
Y

0

1

2

3

4

Prob Density

Proxy Distribution p (y|x)

Peak of Proxy Distribution

Diffusion Distribution p (y|x)
Peak of Diffusion Distribution
Ground-truth

Figure 4: Proxy vs. diffusion distribution.

The MSE loss for the proxy distribution is 2.88, while
320

for the diffusion distribution, it is 0.13 on the Ant
321

task. Additionally, we evaluate this on the TFB10
322

task, where the MSE loss for the proxy distribution
323

is 323.63 compared to 0.82 for the diffusion distribu-
324

tion. These results further corroborate the effective-
325

ness of our proposed module.
326

Furthermore, we (1) investigate the impact of re-
327

placing our trained proxy model with alternative ap-
328

proaches, specifically ROMA and COMs, (2) analyze
329

the performance with an optimized condition y and
330

(3) explore a simple annealing approach of ω. For
331

a comprehensive discussion on these, readers are re-
332

ferred to Appendix E.
333

4.6
Hyperparameter Sensitivity Analysis
334

This section investigates the sensitivity of RGD to various hyperparameters. Specifically, we analyze
335

the effects of (1) the number of diffusion sampling steps T, (2) the condition y, and (3) the learning
336

rate η of the proxy-enhanced sampling. These parameters are evaluated on two tasks: the continuous
337

Ant task and the discrete TFB10 task. For a detailed discussion, see Appendix F.
338

5
Related Work
339

Offline black-box optimization. A recent surge in research has presented two predominant ap-
340

proaches for offline BBO. The forward approach deploys a DNN to fit the offline dataset, subsequently
341

utilizing gradient ascent to enhance existing designs. Typically, these techniques, including COMs [5],
342

ROMA [18], NEMO [19], BDI [20, 28], IOM [29] and Parallel-mentoring [30], are designed to
343

embed prior knowledge within the surrogate model to alleviate the OOD issue. The reverse ap-
344

proach [6, 31] is dedicated to learning a mapping from property values back to inputs. Feeding a high
345

value into this inverse mapping directly produces a design of elevated performance. Additionally,
346

methods in [22, 23] progressively tailor a generative model towards the optimized design via a proxy
347

function and BONET [24] introduces an autoregressive model trained on fixed-length trajectories to
348

sample high-scoring designs. Recent investigations [4] have underscored the superiority of diffusion
349

models in delineating the inverse mapping. However, research on specialized guided diffusion for
350

offline BBO remains limited. This paper addresses this research gap.
351

Guided diffusion. Guided diffusion seeks to produce samples with specific desirable attributes.
352

Contemporary research in guided diffusion primarily concentrates on enhancing the efficiency of
353

its sampling process. [32] propose a method for distilling a classifier-free guided diffusion model
354

into a more efficient single model that necessitates fewer steps in sampling. [33] introduce an
355

operator splitting method to expedite classifier guidance by separating the update process into two
356

key functions: the diffusion function and the conditioning function. Additionally, [34] presents an
357

efficient and universal guidance mechanism that utilizes a readily available proxy to enable diffusion
358

guidance across time steps. In this work, we explore the application of guided diffusion in offline
359

BBO, with the goal of creating tailored algorithms to efficiently generate high-performance designs.
360

6
Conclusion
361

In conclusion, we propose Robust Guided Diffusion for Offline Black-box Optimization (RGD). The
362

proxy-enhanced sampling module adeptly integrates proxy guidance to enable enhanced sampling
363

control, while the diffusion-based proxy refinement module leverages proxy-free diffusion insights
364

for proxy improvement. Empirical evaluations on design-bench have showcased RGD’s outstanding
365

performance, further validated by ablation studies on the contributions of these novel components.
366

We discuss the broader impact and limitation in Appendix G.
367

9


---Page Break---
References
368

[1] Kam Hamidieh. A data-driven statistical model for predicting the critical temperature of a
369

superconductor. Computational materials science, 2018.
370

[2] Karen S Sarkisyan et al. Local fitness landscape of the green fluorescent protein. Nature, 2016.
371

[3] Brandon Trabucco, Xinyang Geng, Aviral Kumar, and Sergey Levine. Design-Bench: bench-
372

marks for data-driven offline model-based optimization. arXiv preprint arXiv:2202.08450,
373

2022.
374

[4] Siddarth Krishnamoorthy, Satvik Mehul Mashkaria, and Aditya Grover. Diffusion models for
375

black-box optimization. Proc. Int. Conf. Machine Learning (ICML), 2023.
376

[5] Brandon Trabucco, Aviral Kumar, Xinyang Geng, and Sergey Levine. Conservative objective
377

models for effective offline model-based optimization. In Proc. Int. Conf. Machine Learning
378

(ICML), 2021.
379

[6] Aviral Kumar and Sergey Levine. Model inversion networks for model-based optimization.
380

Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2020.
381

[7] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
382

Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Proc. Adv. Neur.
383

Inf. Proc. Syst (NeurIPS), 2014.
384

[8] Jonathan Ho and Tim Salimans.
Classifier-free diffusion guidance.
arXiv preprint
385

arXiv:2207.12598, 2022.
386

[9] HoHo Rosenbrock. An automatic method for finding the greatest or least value of a function.
387

The computer journal, 1960.
388

[10] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and
389

Ben Poole. Score-based generative modeling through stochastic differential equations. Proc.
390

Int. Conf. Learning Rep. (ICLR), 2021.
391

[11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Proc.
392

Adv. Neur. Inf. Proc. Syst (NeurIPS), 2021.
393

[12] Endre Süli and David F Mayers. An introduction to numerical analysis. Cambridge university
394

press, 2003.
395

[13] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
396

Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: an imperative
397

style, high-performance deep learning library. Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2019.
398

[14] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang,
399

and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.
400

[15] Michael Ahn, Henry Zhu, Kristian Hartikainen, Hugo Ponte, Abhishek Gupta, Sergey Levine,
401

and Vikash Kumar. Robel: robotics benchmarks for learning with low-cost robots. In Conf. on
402

Robot Lea. (CoRL), 2020.
403

[16] Luis A Barrera et al. Survey of variation in human transcription factors reveals prevalent DNA
404

binding changes. Science, 2016.
405

[17] Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. arXiv
406

preprint arXiv:1611.01578, 2017.
407

[18] Sihyun Yu, Sungsoo Ahn, Le Song, and Jinwoo Shin. Roma: robust model adaptation for offline
408

model-based optimization. Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2021.
409

[19] Justin Fu and Sergey Levine. Offline model-based optimization via normalized maximum
410

likelihood estimation. Proc. Int. Conf. Learning Rep. (ICLR), 2021.
411

[20] Can Chen, Yingxue Zhang, Jie Fu, Xue Liu, and Mark Coates. Bidirectional learning for offline
412

infinite-width model-based optimization. In Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2022.
413

10


---Page Break---
[21] Han Qi, Yi Su, Aviral Kumar, and Sergey Levine. Data-driven model-based optimization via
414

invariant representation learning. In Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2022.
415

[22] David Brookes, Hahnbeom Park, and Jennifer Listgarten. Conditioning by adaptive sampling
416

for robust design. In Proc. Int. Conf. Machine Learning (ICML), 2019.
417

[23] Clara Fannjiang and Jennifer Listgarten. Autofocused oracles for model-based design. Proc.
418

Adv. Neur. Inf. Proc. Syst (NeurIPS), 2020.
419

[24] Satvik Mehul Mashkaria, Siddarth Krishnamoorthy, and Aditya Grover. Generative pretraining
420

for black-box optimization. In Proc. Int. Conf. Machine Learning (ICML), 2023.
421

[25] Nikolaus Hansen. The CMA evolution strategy: a comparing review. Towards A New Evolu-
422

tionary Computation, 2006.
423

[26] James T Wilson, Riccardo Moriconi, Frank Hutter, and Marc Peter Deisenroth. The reparame-
424

terization trick for acquisition functions. arXiv preprint arXiv:1712.00424, 2017.
425

[27] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforce-
426

ment learning. Machine learning, 1992.
427

[28] Can Chen, Yingxue Zhang, Xue Liu, and Mark Coates. Bidirectional learning for offline
428

model-based biological sequence design. In Proc. Int. Conf. Machine Lea. (ICML), 2023.
429

[29] Han Qi, Yi Su, Aviral Kumar, and Sergey Levine. Data-driven model-based optimization via
430

invariant representation learning. In Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2022.
431

[30] Can Chen, Christopher Beckham, Zixuan Liu, Xue Liu, and Christopher Pal. Parallel-mentoring
432

for offline model-based optimization. In Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2023.
433

[31] Alvin Chan, Ali Madani, Ben Krause, and Nikhil Naik. Deep extrapolation for attribute-
434

enhanced generation. Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2021.
435

[32] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho,
436

and Tim Salimans. On distillation of guided diffusion models. In Proc. Comp. Vision. Pattern.
437

Rec.(CVPR), 2023.
438

[33] Suttisak Wizadwongsa and Supasorn Suwajanakorn. Accelerating guided diffusion sampling
439

with splitting numerical methods. In Proc. Int. Conf. Learning Rep. (ICLR), 2023.
440

[34] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum,
441

Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proc. Comp.
442

Vision. Pattern. Rec.(CVPR), 2023.
443

[35] Can Chen, Yingxue Zhang, Jie Fu, Xue Liu, and Mark Coates. Bidirectional learning for offline
444

infinite-width model-based optimization. Proc. Adv. Neur. Inf. Proc. Syst (NeurIPS), 2022.
445

11


---Page Break---
A
Derivation
446

This section provides a derivation of the gradient of the KL divergence. Let’s consider the KL
447

divergence term, defined as:
448

D(pϕ||pθ) =
Z
pϕ(y|ˆx) log
pϕ(y|ˆx)

pθ(y|ˆx)


dy.
(16)

The gradient with respect to the parameters ϕ is computed as follows:
449

dD(pϕ||pθ)

dϕ
=
Z dpϕ(y|ˆx)

dϕ


1 + log pϕ(y|ˆx)

pθ(y|ˆx)


dy

=
Z
pϕ(y|ˆx)d log pϕ(y|ˆx)

dϕ
(1 + log pϕ(y|ˆx)

pθ(y|ˆx) ) dy

= Epϕ(y|ˆx)

d log pϕ(y|ˆx)

dϕ


1 + log pϕ(y|ˆx)

pθ(y|ˆx)


.

(17)

B
Hyperparameter Optimization
450

We propose adjusting α based on the validation loss, establishing a bi-level optimization framework:
451

α∗= arg min
α
EDv[log pϕ∗(α)(yv|xv)],
(18)

s.t.
ϕ∗(α) = arg min
ϕ
L(ϕ, α).
(19)

Within this context, Dv represents the validation dataset sampled from the offline dataset. The inner
452

optimization task, which seeks the optimal ϕ∗(α), is efficiently approximated via gradient descent.
453

C
Evaluation of Median Scores
454

While the main text of our paper focuses on the 100th percentile scores, this section provides an
455

in-depth analysis of the 50th percentile scores. These median scores, previously explored in [3], serve
456

as an additional metric to assess the performance of our RGD method. The outcomes for continuous
457

tasks are detailed in Table 5, and those pertaining to discrete tasks, along with their respective ranking
458

statistics, are outlined in Table 6. An examination of Table 6 highlights the notable success of the
459

RGD approach, as it achieves the top rank in this evaluation. This finding underscores the method’s
460

robustness and effectiveness.
461

D
Computational Overhead
462

Table 4: Computational Overhead (in seconds).

Process
SuperC
Ant
D’Kitty
NAS
Proxy training
40.8
74.5
24.7
7.8
Diffusion training
405.9
767.9
251.1
56.0
Proxy-e sampling
30.0
29.7
29.6
31.5
Diffusion-b proxy r
3104.6
4036.7
2082.8
3096.2
Overall cost
3581.3
4908.8
2388.2
3191.5

In this section, we analyze the computational overhead of our method. RGD consists of two core
463

components: proxy-enhanced sampling (proxy-e sampling) and diffusion-based proxy refinement
464

(diffusion-b proxy r). Additionally, RGD employs a trained proxy and a proxy-free diffusion model,
465

whose computational demands are denoted as proxy training and diffusion training, respectively.
466

Table 4 indicates that experiments can be completed within approximately one hour, demonstrating ef-
467

ficiency. The diffusion-based proxy refinement module is the primary contributor to the computational
468

overhead, primarily due to the usage of a probability flow ODE for sample likelihood computation.
469

12


---Page Break---
However, as this is a one-time process for refining the proxy, its high computational cost is offset by its
470

non-recurring nature. In contexts such as robotics or bio-chemical research, the most time-intensive
471

part of the production cycle is usually the evaluation of the unknown objective function. Therefore,
472

the time differences between methods for deriving high-performance designs are less critical in
473

actual production environments, highlighting RGD’s practicality where optimization performance
474

are prioritized over computational speed. This aligns with recent literature (A.3 Computational
475

Complexity in [35] and A.7.5. Computational Cost in [28]) indicating that in black-box optimization
476

scenarios, computational time is relatively minor compared to the time and resources dedicated to
477

experimental validation phases.
478

Table 5: Results (median normalized score) on continuous tasks.

Method
Superconductor
Ant Morphology
D’Kitty Morphology
Rosenbrock
BO-qEI
0.300 ± 0.015
0.567 ± 0.000
0.883 ± 0.000
0.761 ± 0.004
CMA-ES
0.379 ± 0.003
−0.045 ± 0.004
0.684 ± 0.016
0.200 ± 0.000
REINFORCE
0.463 ± 0.016
0.138 ± 0.032
0.356 ± 0.131
0.553 ± 0.008
Grad
0.339 ± 0.013
0.532 ± 0.014
0.867 ± 0.006
0.540 ± 0.025
COMs
0.312 ± 0.018
0.568 ± 0.002
0.883 ± 0.000
0.419 ± 0.286
ROMA
0.364 ± 0.020
0.467 ± 0.031
0.850 ± 0.006
−0.121 ± 0.242
NEMO
0.319 ± 0.010
0.592 ± 0.001
0.882 ± 0.002
0.510 ± 0.000
IOM
0.343 ± 0.018
0.513 ± 0.024
0.873 ± 0.009
0.126 ± 0.443
BDI
0.412 ± 0.000
0.474 ± 0.000
0.855 ± 0.000
0.561 ± 0.000
CbAS
0.111 ± 0.017
0.384 ± 0.016
0.753 ± 0.008
0.676 ± 0.008
Auto.CbAS
0.131 ± 0.010
0.364 ± 0.014
0.736 ± 0.025
0.695 ± 0.008
MIN
0.336 ± 0.016
0.618 ± 0.040
0.887 ± 0.004
0.634 ± 0.082
BONET
0.319 ± 0.014
0.615 ± 0.004
0.895 ± 0.021
0.630 ± 0.009
DDOM
0.295 ± 0.001
0.590 ± 0.003
0.870 ± 0.001
0.640 ± 0.001
RGD
0.308 ± 0.003
0.684 ± 0.006
0.874 ± 0.001
0.644 ± 0.002

Table 6: Results (median normalized score) on discrete tasks & ranking on all tasks.

Method
TF Bind 8
TF Bind 10
NAS
Rank Mean
Rank Median
BO-qEI
0.439 ± 0.000
0.467 ± 0.000
0.544 ± 0.099
6.4/15
7/15
CMA-ES
0.537 ± 0.014
0.484 ± 0.014
0.591 ± 0.102
8.0/15
5/15
REINFORCE
0.462 ± 0.021
0.475 ± 0.008
−1.895 ± 0.000
9.7/15
9/15
Grad
0.546 ± 0.022
0.526 ± 0.029
0.443 ± 0.126
6.6/15
8/15
COMs
0.439 ± 0.000
0.467 ± 0.000
0.529 ± 0.003
7.7/15
8/15
ROMA
0.543 ± 0.017
0.518 ± 0.024
0.529 ± 0.008
7.6/15
5/15
NEMO
0.436 ± 0.016
0.453 ± 0.013
0.563 ± 0.020
8.3/15
8/15
IOM
0.439 ± 0.000
0.474 ± 0.014
−0.083 ± 0.012
9.3/15
8/15
BDI
0.439 ± 0.000
0.476 ± 0.000
0.517 ± 0.000
7.3/15
8/15
CbAS
0.428 ± 0.010
0.463 ± 0.007
0.292 ± 0.027
11.3/15
12/15
Auto.CbAS
0.419 ± 0.007
0.461 ± 0.007
0.217 ± 0.005
11.9/15
13/15
MIN
0.421 ± 0.015
0.468 ± 0.006
0.433 ± 0.000
7.0/15
7/15
BONET
0.507 ± 0.007
0.460 ± 0.013
0.571 ± 0.095
5.9/15
6/15
DDOM
0.553 ± 0.002
0.488 ± 0.001
0.367 ± 0.021
6.9/15
5/15
RGD
0.557 ± 0.002
0.545 ± 0.006
0.371 ± 0.019
4.9/15
4/15

E
Further Ablation Studies
479

In this section, we extend our exploration to include alternative proxy refinement schemes, namely
480

ROMA and COMs, to compare against our diffusion-based proxy refinement module. The objective
481

is to assess the relative effectiveness of these schemes in the context of the Ant and TFB10 tasks.
482

The comparative results are presented in Table 7. Our investigation reveals that proxies refined
483

through ROMA and COMs exhibit performance akin to the vanilla proxy and they fall short of
484

achieving the enhancements seen with our diffusion-based proxy refinement. We hypothesize that
485

the diffusion-based proxy refinement, by aligning closely with the characteristics of the diffusion
486

13


---Page Break---
model, provides a more relevant and impactful signal. This alignment improves the proxy’s ability to
487

enhance the sampling process more effectively.
488

Table 7: Comparative Results of Proxy Integration with COMs, ROMA, and ours.

Method
Ant Morphology
TF Bind 10
No proxy
0.940 ± 0.004
0.657 ± 0.006
Vanilla proxy
0.961 ± 0.011
0.667 ± 0.009
COMs
0.963 ± 0.004
0.668 ± 0.003
ROMA
0.953 ± 0.003
0.667 ± 0.003
Ours
0.968 ± 0.006
0.694 ± 0.018

Additionally, we contrast our approach, which adjusts the strength parameter ω, with the MIN method
489

that focuses on identifying an optimal condition y. The MIN strategy entails optimizing a Lagrangian
490

objective with respect to y, a process that requires manual tuning of four hyperparameters. We
491

adopt their methodology to determine optimal conditions y and incorporate these into the proxy-free
492

diffusion for tasks Ant and TF10. The normalized scores for Ant and TF10 are 0.950 ± 0.017 and
493

0.660 ± 0.027, respectively. The outcomes fall short of those achieved by our method as detailed
494

in Table 7. This discrepancy likely stems from the complexity involved in optimizing y, whereas
495

dynamically adjusting ω proves to be a more efficient strategy for enhancing sampling control.
496

Last but not least, we explore simple annealing approaches for ω. Specifically, we test two annealing
497

scenarios considering the default ω as 2.0: (1) a decrease from 4.0 to 0.0, and (2) an increase from
498

0.0 to 4.0, both modulated by a cosine function over the time step (t). We apply these strategies to
499

the Ant Morphology and TF Bind 10 tasks, and the results are as follows:

Table 8: Results of Annealing Approaches.
Method
Ant Morphology
TF Bind 10
RGD
0.968
0.694
ω = 2.0
0.940
0.657
Increase
0.948
0.654
Decrease
0.924
0.647

500

The empirical results across both strategies illustrate their inferior performance compared to our
501

approach, thereby demonstrating the efficacy of our proposed method.
502

F
Hyperparameter Sensitivity Analysis
503

RGD’s performance is assessed under different settings of T, y, and η. We experiment with T values
504

of 500, 750, 1000, 1250, and 1500, with the default being T = 1000. For the condition ratio y/ymax,
505

we test values of 0.5, 1.0, 1.5, 2.0, and 2.5, considering 1.0 as the default. Similarly, for the learning
506

rate η, we explore values of 2.5e−3, 5.0e−3, 0.01, 0.02, and 0.04, with the default set to η = 0.01.
507

Results are normalized by comparing them with the performance obtained at default values.
508

As depicted in Figures 5, 6, and 7, RGD demonstrates considerable resilience to hyperparameter
509

variations. The Ant task, in particular, exhibits a more marked sensitivity, with a gradual enhancement
510

in performance as these hyperparameters are varied. The underlying reasons for this trend include:
511

(1) An increase in the number of diffusion steps (T) enhances the overall quality of the generated
512

samples. This improvement, in conjunction with more effective guidance from the trained proxy,
513

leads to better results. (2) Elevating the condition (y) enables the diffusion model to extend its reach
514

beyond the existing dataset, paving the way for superior design solutions. However, selecting an
515

optimal y can be challenging and may, as observed in the TFB10 task, sometimes lead to suboptimal
516

results. (3) A higher learning rate (η) integrates an enhanced guidance signal from the trained proxy,
517

contributing to improved performances.
518

In contrast, the discrete nature of the TFB10 task seems to endow it with a certain robustness
519

to variations in these hyperparameters, highlighting a distinct behavioral pattern in response to
520

hyperparameter adjustments.
521

14


---Page Break---
500
750
1000
1250
1500
Diffusion step t

0.990

0.995

1.000

1.005

1.011

Score ratio

Ant
TF10

Figure 5: The ratio of the
performance of our RGD
method with T to the per-
formance with T = 1000.

0.5
1.0
1.5
2.0
2.5
Condition y/ymax

0.960

1.000
1.004
1.008
1.012

Score ratio

Ant
TF10

Figure 6: The ratio of the
performance of our RGD
method with y/ymax to the
performance with 1.0.

0.0025
0.005
0.01
0.02
0.04
Learning rate 

0.985

0.995

1.005

1.015

1.020

Score ratio

Ant
TF10

Figure 7: The ratio of the
performance of our RGD
method with η to the per-
formance with η = 0.01.

G
Broader Impact and Limitation
522

Broader impact. Our research has the potential to significantly accelerate advancements in fields such
523

as new material development, biomedical innovation, and robotics technology. These advancements
524

could lead to breakthroughs with substantial positive societal impacts. However, we recognize that,
525

like any powerful tool, there are inherent risks associated with the misuse of this technology. One
526

concerning possibility is the exploitation of our optimization techniques to design objects or entities
527

for malicious purposes, including the creation of more efficient weaponry or harmful biological agents.
528

Given these potential risks, it is imperative to enforce strict safeguards and regulatory measures,
529

especially in areas where the misuse of technology could lead to significant ethical and societal harm.
530

The responsible application and governance of such technologies are crucial to ensuring that they
531

serve to benefit society as a whole.
532

Limitation. We recognize that the benchmarks utilized in our study may not fully capture the
533

complexities of more advanced applications, such as protein drug design, primarily due to our current
534

limitations in accessing wet-lab experimental setups. Moving forward, we aim to mitigate this
535

limitation by fostering partnerships with domain experts, which will enable us to apply our method
536

to more challenging and diverse problems. This direction not only promises to validate the efficacy
537

of our approach in more complex scenarios but also aligns with our commitment to pushing the
538

boundaries of what our technology can achieve.
539

15


---Page Break---
NeurIPS Paper Checklist
540

1. Claims
541

Question: Do the main claims made in the abstract and introduction accurately reflect the
542

paper’s contributions and scope?
543

Answer: [Yes]
544

Justification: The abstract and introduction accurately reflect the paper’s contributions and
545

scope.
546

Guidelines:
547

• The answer NA means that the abstract and introduction do not include the claims
548

made in the paper.
549

• The abstract and/or introduction should clearly state the claims made, including the
550

contributions made in the paper and important assumptions and limitations. A No or
551

NA answer to this question will not be perceived well by the reviewers.
552

• The claims made should match theoretical and experimental results, and reflect how
553

much the results can be expected to generalize to other settings.
554

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
555

are not attained by the paper.
556

2. Limitations
557

Question: Does the paper discuss the limitations of the work performed by the authors?
558

Answer: [Yes]
559

Justification: We discuss the limitations in Appendix G.
560

Guidelines:
561

• The answer NA means that the paper has no limitation while the answer No means that
562

the paper has limitations, but those are not discussed in the paper.
563

• The authors are encouraged to create a separate "Limitations" section in their paper.
564

• The paper should point out any strong assumptions and how robust the results are to
565

violations of these assumptions (e.g., independence assumptions, noiseless settings,
566

model well-specification, asymptotic approximations only holding locally). The authors
567

should reflect on how these assumptions might be violated in practice and what the
568

implications would be.
569

• The authors should reflect on the scope of the claims made, e.g., if the approach was
570

only tested on a few datasets or with a few runs. In general, empirical results often
571

depend on implicit assumptions, which should be articulated.
572

• The authors should reflect on the factors that influence the performance of the approach.
573

For example, a facial recognition algorithm may perform poorly when image resolution
574

is low or images are taken in low lighting. Or a speech-to-text system might not be
575

used reliably to provide closed captions for online lectures because it fails to handle
576

technical jargon.
577

• The authors should discuss the computational efficiency of the proposed algorithms
578

and how they scale with dataset size.
579

• If applicable, the authors should discuss possible limitations of their approach to
580

address problems of privacy and fairness.
581

• While the authors might fear that complete honesty about limitations might be used by
582

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
583

limitations that aren’t acknowledged in the paper. The authors should use their best
584

judgment and recognize that individual actions in favor of transparency play an impor-
585

tant role in developing norms that preserve the integrity of the community. Reviewers
586

will be specifically instructed to not penalize honesty concerning limitations.
587

3. Theory Assumptions and Proofs
588

Question: For each theoretical result, does the paper provide the full set of assumptions and
589

a complete (and correct) proof?
590

Answer: [NA]
591

16


---Page Break---
Justification: The paper does not include theoretical results.
592

Guidelines:
593

• The answer NA means that the paper does not include theoretical results.
594

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
595

referenced.
596

• All assumptions should be clearly stated or referenced in the statement of any theorems.
597

• The proofs can either appear in the main paper or the supplemental material, but if
598

they appear in the supplemental material, the authors are encouraged to provide a short
599

proof sketch to provide intuition.
600

• Inversely, any informal proof provided in the core of the paper should be complemented
601

by formal proofs provided in appendix or supplemental material.
602

• Theorems and Lemmas that the proof relies upon should be properly referenced.
603

4. Experimental Result Reproducibility
604

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
605

perimental results of the paper to the extent that it affects the main claims and/or conclusions
606

of the paper (regardless of whether the code and data are provided or not)?
607

Answer: [Yes]
608

Justification: We provide our code link in the abstract and detail our settings in Section 4.3.
609

Guidelines:
610

• The answer NA means that the paper does not include experiments.
611

• If the paper includes experiments, a No answer to this question will not be perceived
612

well by the reviewers: Making the paper reproducible is important, regardless of
613

whether the code and data are provided or not.
614

• If the contribution is a dataset and/or model, the authors should describe the steps taken
615

to make their results reproducible or verifiable.
616

• Depending on the contribution, reproducibility can be accomplished in various ways.
617

For example, if the contribution is a novel architecture, describing the architecture fully
618

might suffice, or if the contribution is a specific model and empirical evaluation, it may
619

be necessary to either make it possible for others to replicate the model with the same
620

dataset, or provide access to the model. In general. releasing code and data is often
621

one good way to accomplish this, but reproducibility can also be provided via detailed
622

instructions for how to replicate the results, access to a hosted model (e.g., in the case
623

of a large language model), releasing of a model checkpoint, or other means that are
624

appropriate to the research performed.
625

• While NeurIPS does not require releasing code, the conference does require all submis-
626

sions to provide some reasonable avenue for reproducibility, which may depend on the
627

nature of the contribution. For example
628

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
629

to reproduce that algorithm.
630

(b) If the contribution is primarily a new model architecture, the paper should describe
631

the architecture clearly and fully.
632

(c) If the contribution is a new model (e.g., a large language model), then there should
633

either be a way to access this model for reproducing the results or a way to reproduce
634

the model (e.g., with an open-source dataset or instructions for how to construct
635

the dataset).
636

(d) We recognize that reproducibility may be tricky in some cases, in which case
637

authors are welcome to describe the particular way they provide for reproducibility.
638

In the case of closed-source models, it may be that access to the model is limited in
639

some way (e.g., to registered users), but it should be possible for other researchers
640

to have some path to reproducing or verifying the results.
641

5. Open access to data and code
642

Question: Does the paper provide open access to the data and code, with sufficient instruc-
643

tions to faithfully reproduce the main experimental results, as described in supplemental
644

material?
645

17


---Page Break---
Answer: [Yes]
646

Justification: We provide a link to our source code in the abstract and thoroughly describe
647

our experimental settings in Section 4.3.
648

Guidelines:
649

• The answer NA means that paper does not include experiments requiring code.
650

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
651

public/guides/CodeSubmissionPolicy) for more details.
652

• While we encourage the release of code and data, we understand that this might not be
653

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
654

including code, unless this is central to the contribution (e.g., for a new open-source
655

benchmark).
656

• The instructions should contain the exact command and environment needed to run to
657

reproduce the results. See the NeurIPS code and data submission guidelines (https:
658

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
659

• The authors should provide instructions on data access and preparation, including how
660

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
661

• The authors should provide scripts to reproduce all experimental results for the new
662

proposed method and baselines. If only a subset of experiments are reproducible, they
663

should state which ones are omitted from the script and why.
664

• At submission time, to preserve anonymity, the authors should release anonymized
665

versions (if applicable).
666

• Providing as much information as possible in supplemental material (appended to the
667

paper) is recommended, but including URLs to data and code is permitted.
668

6. Experimental Setting/Details
669

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
670

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
671

results?
672

Answer: [Yes]
673

Justification: We detail our setting in Section 4.3 and also discuss hyperparameter sensitivity
674

in Appendix F.
675

Guidelines:
676

• The answer NA means that the paper does not include experiments.
677

• The experimental setting should be presented in the core of the paper to a level of detail
678

that is necessary to appreciate the results and make sense of them.
679

• The full details can be provided either with the code, in appendix, or as supplemental
680

material.
681

7. Experiment Statistical Significance
682

Question: Does the paper report error bars suitably and correctly defined or other appropriate
683

information about the statistical significance of the experiments?
684

Answer: [Yes]
685

Justification: To ensure reliability and consistency in our comparative analysis, each experi-
686

mental setting was replicated across 8 independent runs, unless stated otherwise, with the
687

presentation of both mean values and standard errors.
688

Guidelines:
689

• The answer NA means that the paper does not include experiments.
690

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
691

dence intervals, or statistical significance tests, at least for the experiments that support
692

the main claims of the paper.
693

• The factors of variability that the error bars are capturing should be clearly stated (for
694

example, train/test split, initialization, random drawing of some parameter, or overall
695

run with given experimental conditions).
696

18


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
697

call to a library function, bootstrap, etc.)
698

• The assumptions made should be given (e.g., Normally distributed errors).
699

• It should be clear whether the error bar is the standard deviation or the standard error
700

of the mean.
701

• It is OK to report 1-sigma error bars, but one should state it. The authors should
702

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
703

of Normality of errors is not verified.
704

• For asymmetric distributions, the authors should be careful not to show in tables or
705

figures symmetric error bars that would yield results that are out of range (e.g. negative
706

error rates).
707

• If error bars are reported in tables or plots, The authors should explain in the text how
708

they were calculated and reference the corresponding figures or tables in the text.
709

8. Experiments Compute Resources
710

Question: For each experiment, does the paper provide sufficient information on the com-
711

puter resources (type of compute workers, memory, time of execution) needed to reproduce
712

the experiments?
713

Answer: [Yes]
714

Justification: We have discussed these in Section 4.3.
715

Guidelines:
716

• The answer NA means that the paper does not include experiments.
717

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
718

or cloud provider, including relevant memory and storage.
719

• The paper should provide the amount of compute required for each of the individual
720

experimental runs as well as estimate the total compute.
721

• The paper should disclose whether the full research project required more compute
722

than the experiments reported in the paper (e.g., preliminary or failed experiments that
723

didn’t make it into the paper).
724

9. Code Of Ethics
725

Question: Does the research conducted in the paper conform, in every respect, with the
726

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
727

Answer: [Yes]
728

Justification: We preserve anonymity and conform with the NeurIPS Code of Ethics.
729

Guidelines:
730

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
731

• If the authors answer No, they should explain the special circumstances that require a
732

deviation from the Code of Ethics.
733

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
734

eration due to laws or regulations in their jurisdiction).
735

10. Broader Impacts
736

Question: Does the paper discuss both potential positive societal impacts and negative
737

societal impacts of the work performed?
738

Answer: [Yes]
739

Justification: We discuss both potential positive and negative impacts in Appendix G.
740

Guidelines:
741

• The answer NA means that there is no societal impact of the work performed.
742

• If the authors answer NA or No, they should explain why their work has no societal
743

impact or why the paper does not address societal impact.
744

• Examples of negative societal impacts include potential malicious or unintended uses
745

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
746

(e.g., deployment of technologies that could make decisions that unfairly impact specific
747

groups), privacy considerations, and security considerations.
748

19


---Page Break---
• The conference expects that many papers will be foundational research and not tied
749

to particular applications, let alone deployments. However, if there is a direct path to
750

any negative applications, the authors should point it out. For example, it is legitimate
751

to point out that an improvement in the quality of generative models could be used to
752

generate deepfakes for disinformation. On the other hand, it is not needed to point out
753

that a generic algorithm for optimizing neural networks could enable people to train
754

models that generate Deepfakes faster.
755

• The authors should consider possible harms that could arise when the technology is
756

being used as intended and functioning correctly, harms that could arise when the
757

technology is being used as intended but gives incorrect results, and harms following
758

from (intentional or unintentional) misuse of the technology.
759

• If there are negative societal impacts, the authors could also discuss possible mitigation
760

strategies (e.g., gated release of models, providing defenses in addition to attacks,
761

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
762

feedback over time, improving the efficiency and accessibility of ML).
763

11. Safeguards
764

Question: Does the paper describe safeguards that have been put in place for responsible
765

release of data or models that have a high risk for misuse (e.g., pretrained language models,
766

image generators, or scraped datasets)?
767

Answer: [NA]
768

Justification: We do not release any datasets nor pre-trained models.
769

Guidelines:
770

• The answer NA means that the paper poses no such risks.
771

• Released models that have a high risk for misuse or dual-use should be released with
772

necessary safeguards to allow for controlled use of the model, for example by requiring
773

that users adhere to usage guidelines or restrictions to access the model or implementing
774

safety filters.
775

• Datasets that have been scraped from the Internet could pose safety risks. The authors
776

should describe how they avoided releasing unsafe images.
777

• We recognize that providing effective safeguards is challenging, and many papers do
778

not require this, but we encourage authors to take this into account and make a best
779

faith effort.
780

12. Licenses for existing assets
781

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
782

the paper, properly credited and are the license and terms of use explicitly mentioned and
783

properly respected?
784

Answer: [Yes]
785

Justification: We have duly credited all utilized assets and adhered to their respective licenses
786

and terms of use.
787

Guidelines:
788

• The answer NA means that the paper does not use existing assets.
789

• The authors should cite the original paper that produced the code package or dataset.
790

• The authors should state which version of the asset is used and, if possible, include a
791

URL.
792

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
793

• For scraped data from a particular source (e.g., website), the copyright and terms of
794

service of that source should be provided.
795

• If assets are released, the license, copyright information, and terms of use in the
796

package should be provided. For popular datasets, paperswithcode.com/datasets
797

has curated licenses for some datasets. Their licensing guide can help determine the
798

license of a dataset.
799

• For existing datasets that are re-packaged, both the original license and the license of
800

the derived asset (if it has changed) should be provided.
801

20


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
802

the asset’s creators.
803

13. New Assets
804

Question: Are new assets introduced in the paper well documented and is the documentation
805

provided alongside the assets?
806

Answer: [Yes]
807

Justification: We plan to open-source our code and have ensured thorough documentation of
808

the code.
809

Guidelines:
810

• The answer NA means that the paper does not release new assets.
811

• Researchers should communicate the details of the dataset/code/model as part of their
812

submissions via structured templates. This includes details about training, license,
813

limitations, etc.
814

• The paper should discuss whether and how consent was obtained from people whose
815

asset is used.
816

• At submission time, remember to anonymize your assets (if applicable). You can either
817

create an anonymized URL or include an anonymized zip file.
818

14. Crowdsourcing and Research with Human Subjects
819

Question: For crowdsourcing experiments and research with human subjects, does the paper
820

include the full text of instructions given to participants and screenshots, if applicable, as
821

well as details about compensation (if any)?
822

Answer: [NA]
823

Justification: This paper does not engage in crowdsourcing or involve studies with human
824

participants.
825

Guidelines:
826

• The answer NA means that the paper does not involve crowdsourcing nor research with
827

human subjects.
828

• Including this information in the supplemental material is fine, but if the main contribu-
829

tion of the paper involves human subjects, then as much detail as possible should be
830

included in the main paper.
831

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
832

or other labor should be paid at least the minimum wage in the country of the data
833

collector.
834

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
835

Subjects
836

Question: Does the paper describe potential risks incurred by study participants, whether
837

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
838

approvals (or an equivalent approval/review based on the requirements of your country or
839

institution) were obtained?
840

Answer: [NA]
841

Justification: This paper does not engage in crowdsourcing or research involving human
842

subjects.
843

Guidelines:
844

• The answer NA means that the paper does not involve crowdsourcing nor research with
845

human subjects.
846

• Depending on the country in which research is conducted, IRB approval (or equivalent)
847

may be required for any human subjects research. If you obtained IRB approval, you
848

should clearly state this in the paper.
849

• We recognize that the procedures for this may vary significantly between institutions
850

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
851

guidelines for their institution.
852

• For initial submissions, do not include any information that would break anonymity (if
853

applicable), such as the institution conducting the review.
854

21


---Page Break---
