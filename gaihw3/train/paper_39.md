Language models scale reliably with over-training and
on downstream tasks

Anonymous Author(s)
Affiliation
Address
email

Abstract

Scaling laws are useful guides for derisking expensive training runs, as they predict
1

performance of large models using cheaper, small-scale experiments. However,
2

there remain gaps between current scaling studies and how language models are
3

ultimately trained and evaluated. For instance, scaling is usually studied in the
4

compute-optimal training regime (i.e., “Chinchilla optimal” regime). In contrast,
5

models are often over-trained to reduce inference costs. Moreover, scaling laws
6

mostly predict loss on next-token prediction, but models are usually compared on
7

downstream task performance. To address both shortcomings, we create a testbed
8

of 104 models with 0.011B to 6.9B parameters trained with various numbers of
9

tokens on three data distributions. First, we fit scaling laws that extrapolate in both
10

the amount of over-training and the number of model parameters. This enables us
11

to predict the validation loss of a 1.4B parameter, 900B token run (i.e., 32× over-
12

trained) and a 6.9B parameter, 138B token run (i.e., a compute-optimal run)—each
13

from experiments that take 300× less compute. Second, we relate the perplexity of
14

a language model to its downstream task performance by proposing a power law.
15

We use this law to predict top-1 error averaged over downstream tasks for the two
16

aforementioned models, using experiments that take 20× less compute.
17

1
Introduction
18

Training large language models is expensive. Furthermore, training high-quality models requires a
19

complex recipe of algorithmic techniques and training data. To reduce the cost of finding successful
20

training recipes, researchers first evaluate ideas with small experiments and then extrapolate their
21

efficacy to larger model and data regimes via scaling laws. With reliable extrapolation, it is possible
22

to quickly iterate at small scale and still pick the method that will perform best for the final large
23

training run. Indeed, this workflow has become commonplace for training state-of-the-art language
24

models like Chinchilla 70B [45], PaLM 540B [19], GPT-4 [76], and many others.
25

Despite their importance for model development, published scaling laws differ from the goals of
26

training state-of-the-art models in important ways. For instance, scaling studies usually focus on the
27

compute-optimal training regime (“Chinchilla optimality” [45]), where model and dataset size are set
28

to yield minimum loss for a given compute budget. However, this setting ignores inference costs.
29

As larger models are more expensive at inference, it is now common practice to over-train smaller
30

models [113]. Another potential mismatch is that most scaling laws quantify model performance by
31

perplexity in next-token prediction instead of accuracy on widely used benchmark datasets. However,
32

practitioners usually turn to benchmark performance, not loss, to compare models.
33

In this paper, we conduct an extensive set of experiments to address both scaling in the over-trained
34

regime and benchmark performance prediction.
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
1016
1017
1018
1019
1020
1021
1022

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

Reducible loss: C4 eval

N = 0.011B
N = 0.079B
N = 0.154B
N = 0.411B
N = 1.4B
N = 6.9B

Prediction
Interpolation
Extrapolation
M = 20
M = 320
M = 640

2.0
2.5
3.0
3.5
4.0
4.5
5.0
5.5
6.0

Loss: C4 eval

0.4

0.5

0.6

0.7

0.8

Average top-1 error: 17-task split

1022
0.56

0.60

0.64

0.68

0.72

2.4
2.6

0.44

0.46

0.48

0.50

0.52

Figure 1: Reliable scaling with over-training and on downstream error prediction. (left) We fit
a scaling law for model validation loss, parameterized by (i) a token multiplier M = N/D, which
is the ratio of training tokens D to parameters N and (ii) the compute C in FLOPs used to train a
model, approximated by C = 6ND. Larger values of M specify more over-training. We are able
to extrapolate, in both N and M, the validation performance of models requiring more than 300×
the training compute used to construct the scaling law. (right) We also fit a scaling law to predict
average downstream top-1 error as a function of validation loss. We find that fitting scaling laws
for downstream error benefits from using more expensive models when compared to fitting for loss
prediction. We predict the average error over 17 downstream tasks for models trained with over 20×
the compute. For this figure, we train all models on RedPajama [112].

Motivated by the practice of training beyond compute-optimality, we first investigate whether scaling
36

follows reliable trends in the over-trained regime. We notice, as implied by Hoffmann et al. [45], for a
37

set of models of different sizes trained with a constant ratio of tokens to parameters, models’ reducible
38

loss L′ [43, 45] follows a power law (L′ = λ · C−η) in the amount of training compute C. We
39

find that as one increases the ratio of tokens to parameters, corresponding to more over-training, the
40

scaling exponent η remains about the same, while the scalar λ changes. We explain our observations
41

by reparameterizing existing scaling laws in relation to the amount of over-training.
42

To establish empirically that scaling extrapolates in the over-trained regime, we further experiment
43

with a testbed of 104 models, trained from scratch on three different datasets: C4 [88, 27],
44

RedPajama [112], and RefinedWeb [82]. We find that scaling laws fit to small models can accurately
45

predict the performance of larger models that undergo more over-training. Figure 1 (left) illustrates our
46

main over-training result, where we invest 2.4e19 FLOPs to extrapolate the C4 validation performance
47

of a 1.4B parameter model trained on 900B tokens, which requires 300× more compute to train.
48

In addition to over-training, we also investigate if scaling laws can predict the performance of a
49

model on downstream tasks. We establish a power law relationship between language modeling
50

perplexity and the average top-1 error on a suite of downstream tasks. While it can be difficult to
51

predict the error on individual tasks, we find it possible to predict aggregate performance from a
52

model’s perplexity among models trained on the same training data. Figure 1 (right) presents our
53

main downstream error prediction result, where we invest 2.7e20 FLOPs to predict the average top-1
54

error over a set of downstream tasks to within 1 percentage point for a 6.9B compute-optimal model,
55

which requires 20× more compute to train.
56

Our results suggest that the proposed scaling laws are promising to derisk (i) the effects of over-
57

training models and (ii) the downstream performance of scaling up training recipes. To facilitate
58

further research on reliable scaling, we will release all experiments and models.
59

2
Developing scaling laws for over-training and downstream tasks
60

In this section, we develop scaling laws to predict over-trained and downstream performance. First,
61

we provide key definitions (Section 2.1). We next present a scaling law for over-training drawing on
62

empirical observation and prior work (Section 2.2). To connect loss scaling and downstream error
63

prediction, we observe that average top-1 error decreases exponentially as a function of validation loss,
64

2


---Page Break---
1017
1019
1021

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

Reducible loss: C4 eval

Training set: C4

1017
1019
1021

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

Training set: RedPajama

1017
1019
1021

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

Training set: RefinedWeb

N = 0.011B
N = 0.079B
N = 0.154B
N = 0.411B
Interpolation
Extrapolation

10

20

40

80

160

320

640

token multiplier M

Figure 2: Scaling in the over-trained regime follows consistent power law exponents. We notice
parallel lines in the log-log plots of reducible loss vs. training compute for a range of token multipliers
M, which give the ratio of training tokens to model parameters. Larger M corresponds to more
over-training. For a power law giving reducible loss as a function of compute: L′(C) = λ · C−η, the
exponent η remains relatively constant resulting in lines with approximately fixed slope (Figure 17).
The scalar λ that determines the y-intercept, however, shifts with different token multipliers. This
suggests λ is a function of the token multiplier, while η is not.

which we formalize as a novel scaling law (Section 2.3). In later sections, we build an experimental
65

setup (Section 3) to quantify the extent to which our scaling laws extrapolate reliably (Section 4).
66

2.1
Preliminaries
67

Scaling laws for loss.
Typically, scaling laws predict model loss L as a function of the compute
68

C in FLOPs used for training. If one increases the number of parameters N in a model or the
69

number of tokens D that a model is trained on, compute requirements naturally increase. Hence, we
70

assume C is a function of N, D. Following Kaplan et al. [51], we use the approximation C = 6ND,
71

which Hoffmann et al. [45] independently verify. We consider,
72

L(C) = E + L′(C),
(1)

where E is an irreducible loss and L′ is the reducible loss. E captures the Bayes error or minimum
73

possible loss achievable on the validation domain. The L′(C) term captures what can possibly be
74

learned about the validation domain by training on a source domain. L′(C) should approach zero
75

with increased training data and model capacity. L′(C) is often assumed to follow a power law:
76

L′(C) = λ · C−η (i.a., Hestness et al. [43], OpenAI [76]). It is also often helpful to consider a power
77

law in a log-log plot, where it appears as a line with slope −η and y-intercept log (λ).
78

Token multipliers.
We define a token multiplier M = D/N as the ratio of training tokens to model
79

parameters for notational convenience. M allows us to consider fixed relationships between D and
80

N even as a model gets bigger (i.e., as N becomes larger).
81

Compute-optimal training.
Hoffmann et al. [45] establish compute-optimal training, where, for
82

any compute budget H, the allocation of parameters and tokens is given by,
83

arg min
N,D L(N, D) s.t. C(N, D) = H.
(2)

To solve for the optimal N ∗, D∗, one can sweep N, D for each compute budget, retaining the
84

best configurations. Hoffmann et al. [45] find that as the compute budget increases, N ∗and D∗
85

scale roughly evenly. Assuming equal scaling, there is a fixed compute-optimal token multiplier
86

M ∗= D∗/N ∗per training distribution.
87

Over-training.
We define over-training as the practice of allocating compute sub-optimally, so
88

smaller models train on a disproportionately large number of tokens (i.e., M > M ∗). While loss
89

should be higher than in the compute-optimal allocation for a given training budget, the resulting
90

models have fewer parameters and thus incur less inference cost.
91

2.2
Scaling laws for over-training
92

To propose a scaling law for over-trained models, we first turn to empirical observation. We train four
93

model configurations with parameter counts between 0.011B and 0.411B for token multipliers M
94

3


---Page Break---
3
4
5
6

Loss: C4 eval

0.50

0.55

0.60

0.65

0.70

0.75

0.80

Average top-1 error: 17-task split

Training set: C4

3
4
5
6

Loss: C4 eval

0.50

0.55

0.60

0.65

0.70

0.75

0.80

Training set: RedPajama

3
4
5
6

Loss: C4 eval

0.50

0.55

0.60

0.65

0.70

0.75

0.80

Training set: RefinedWeb

Model
Interpolation
Extrapolation

Figure 3: Average top-1 error scales as a function of loss. We plot models trained on three datasets
and notice an exponential decay of average top-1 error as C4 eval loss, on the x-axis, decreases. We
consider on the y-axes average error on 17 evaluations where performance is at least 10 points above
random chance for at least one 0.154B scale model. These observations suggest that average top-1
error should be predictable with reliable loss estimates.

between 20 and 640, where M = 20 points lie roughly on the compute-optimal frontier, and larger
95

M corresponds to more over-training. We defer experimental details to Section 3 to focus on our
96

observations first. In Figure 2, we show loss against compute in a log-log plot for the models trained
97

on three datasets and evaluated on the C4 eval set. We notice parallel lines when fitting power laws to
98

the reducible loss, which suggests a near-constant scaling exponent even with increased over-training.
99

This indicates that scaling behavior should be describable in the amount of over-training.
100

In search of an analytic expression for the observations in Figure 2, we consider existing scaling
101

literature. A common functional form for the risk of a model, as proposed in prior work [93, 45] is,
102

L(N, D) = E + AN −α + BD−β.
(3)
Recall from Section 2.1, N is the number of parameters and D the number of training tokens. The
103

constants E, A, α, B, β are fit from data. By fitting this parametric form, Hoffmann et al. [45]
104

find that scaling exponents α and β are roughly equal, suggesting that one should scale N and D
105

equally as compute increases. Hence, we assume α = β. With this assumption, we reparameterize
106

Equation (3) in terms of compute C = 6ND and a token multiplier M = D/N. We get,
107

L(C, M) = E +
 
aM η + bM −η
C−η,
(4)

where η = α/2, a = A(1/6)−η, b = B(1/6)−η gives the relation to Equation (3). For a complete
108

derivation, see Appendix A.
109

Equation (4) has the following interpretation: (i) The scaling exponent η is not dependent on M.
110

Thus, we always expect lines with the same slope in the log-log plot—as in Figure 2. (ii) The term
111

aM η + bM −η determines the offsets between curves with different token multipliers. Hence, we
112

expect non-overlapping, parallel lines in the log-log plot for the range of M we consider—also
113

consistent with Figure 2.
114

Recall that we make the assumption α = β, which implies equal scaling of parameters and tokens
115

as more compute is available. However, as explained in Appendix A, even if α ̸= β, we get a
116

parameterization that implies the power-law exponent remains constant with over-training.
117

2.3
Scaling laws for downstream error
118

Scaling is typically studied in the context of loss [51, 45, 72], which Schaeffer et al. [100] note
119

is smoother than metrics like accuracy. However, practitioners often use downstream benchmark
120

accuracy as a proxy for model quality and not loss on perplexity evaluation sets. To better connect
121

scaling laws and over-training to task prediction, we revisit the suite of models plotted in Figure 2. In
122

Figure 3, we plot average downstream top-1 errors over evaluations sourced from LLM-Foundry [69]
123

against the C4 eval loss. We defer details of the setup to Section 3 to focus here on a key observation:
124

average error appears to follow exponential decay as loss decreases.
125

Based on the exponential decay we observe in Figure 3, we propose the following relationship
126

between downstream average top-1 error Err and loss L,
127

Err(L) = ϵ −k · exp (−γL),
(5)

4


---Page Break---
1017
1019
1021

Compute (6ND) [FLOPs]

2

3

4

5

6

Loss: OpenLM eval

Search

1017
1019
1021

Compute (6ND) [FLOPs]

Filter

1017
1019
1021

Compute (6ND) [FLOPs]

Fit

Grid search models
Selected models
Target 1.4B model
Target 6.9B model
Interpolation
Extrapolation

1022

1.9
1.8
1.7

Figure 4: Search, filter, fit: A recipe for selecting configurations for scaling. (left) To generate the
final configurations presented in Table 3, we run a 435 model grid search over model width, hidden
dimension, number of attention heads, batch size, and warmup steps. All models are trained near
compute-optimally. (center) We plot the efficient frontier of models, which appear to follow a trend,
excluding models from 5.2 × 1016 to 5.2 × 1017, which fall below the trend. (right) We fit a power
law with irreducible error to the remaining configurations, picking four configurations that closely
track the full model suite (“Selected models”). These models extrapolate the performance of 1.4B,
6.9B target models. Shaded regions represent bootstrap 95% confidence intervals.

where ϵ, k, γ are fit from data. Equation (5) also has an interpretation in terms of model perplexity
128

PP(L) = exp (L),
129

Err(PP) = ϵ −k · PP−γ.
(6)

Namely, Err follows a power law in PP that is bounded from above by ϵ signifying arbitrarily high
130

error and from below by ϵ −k · exp(−γE), where E is the Bayes error from Equation (4).
131

Equation (5) in conjunction with Equation (4) suggests a three-step method to predict Err as a function
132

of compute and the amount of over-training. For choices of training and validation distributions, (i)
133

fit a scaling law to Equation (4) using triplets of compute C, token multiplier M, and measured loss
134

L on a validation set to yield (C, M) 7→L. (ii) Fit a scaling law to Equation (5) using pairs of loss L
135

and downstream error Err for models to get L 7→Err. (iii) Chain predictions to get (C, M) 7→Err.
136

3
Constructing a scaling testbed
137

In this section, we discuss our experimental setup to test the predictions suggested by Equations (4)
138

and (5). We first present our general language modeling setup (Section 3.1). Next, we discuss our
139

strategy for determining model configurations for our scaling investigation (Section 3.2) and fitting
140

scaling laws (Section 3.3). We then present metrics to validate how well scaling laws predict loss and
141

downstream performance (Section 3.4).
142

3.1
Training setup
143

We train transformers [116] for next token prediction, based on architectures like GPT-2 [85] and
144

LLaMA [113]. We employ GPT-NeoX [15] as a standardized tokenizer for all data. See Appendix B
145

for architecture, optimization, and hyperparameter details.
146

3.2
Model configurations
147

To get final configurations for the 0.011B to 0.411B parameter models plotted in Figures 2 and 3, we
148

first conduct a wide grid search over a total of 435 models, trained from scratch, from 0.01B to 0.5B
149

parameters (Figure 4 (left)). We train on the original OpenLM data mix [39], which largely consists
150

of RedPajama [112] and The Pile [31]. While we eventually plan to over-train models, at this step
151

we search for base configurations near compute-optimality. We train on 20 tokens per parameter
152

(M = 20), which, in early experiments, gives models near the compute-optimal frontier. This is
153

similar to findings in Hoffmann et al. [45]’s Table 3, which suggests that M = 20 is near-optimal for
154

the Chinchilla experimental setup.
155

5


---Page Break---
Table 1: Default number of parameters N and token multiplier M to fit our scaling laws. We
invest ∼100 A100 hours to fit Equation (4) and ∼1,000 A100 hours to fit Equation (5).

N
M
Used to fit Equation (4)
Used to fit Equation (5)

0.011B
20
✓
✓
0.079B
20
✓
✓
0.154B
20
✓
✓
0.411B
20
✓
✓
0.011B
320
✓
✓
1.4B
20
✗
✓

Total compute C [FLOPs]
2.4e19
2.7e20

To find maximally performant small-scale models on validation data, we tune model width, number
156

of layers, number of attention heads, warmup steps, and batch size. Our validation set, OpenLM
157

eval, contains tokens from recent arXiv papers, the OpenLM codebase itself, and news articles. We
158

find in early experiments that qk-LayerNorm makes models less sensitive to learning rate, which
159

is a phenomenon Wortsman et al. [123] report in their Figure 1. Hence, we fix the learning rate
160

(3e-3) for our sweeps. We also perform smaller grid searches over 1.4B and 6.9B parameter model
161

configurations at M = 20, retaining the best configurations.
162

At this point, we have many models, several of which give poor performance; following prior
163

work [51, 45], we want to keep only models that give best performance. Hence, in Figure 4 (center),
164

we filter out models that do not lie on the Pareto frontier. While there appears to be a general trend,
165

configurations between 5.2 × 1016 and 5.2 × 1017 FLOPs lie below the frontier established by other
166

models. We hypothesize these models over-perform as they are trained for more optimization steps
167

than their neighbors based on our power-of-two batch sizes. We provide support for this hypothesis
168

in Appendix E, but opt to remove these models from our investigation.
169

To ensure tractable compute requirements for our scaling experiments, we require a subset of models
170

that follows the trend of the entire Pareto frontier. In Figure 4 (right), we fit trends to the Pareto
171

models and to a subset of four models. We notice that the trends closely predict both the performance
172

of the 1.4B and 6.9B models, suggesting that our small-scale configurations reliably extrapolate in
173

the compute-optimal setting.
174

Moving forward, we do not tune hyperparameters for other token multipliers (i.e., M ̸= 20), on
175

other training or evaluation distributions, or on validation sets for downstream tasks. For more details
176

including specific hyperparameters, see Appendix C.
177

To create our scaling testbed, we start with the four small-scale, base configurations from our
178

grid search: N ∈{0.011B, 0.079B, 0.154B, 0.411B}. To ensure our conclusions are not particular
179

to a single training distribution, we train models on each of C4 [88, 27], RedPajama [112], and
180

RefinedWeb [82], which have 138B, 1.15T, and 600B tokens, respectively, for different token
181

multipliers M ∈{5, 10, 20, 40, 80, 160, 320, 640}. We omit runs that require more tokens than are
182

present in a dataset (i.e., N = 0.411B, M = 640 for C4). We additionally train N = 1.4B models at
183

M = 20 and at the largest token multiplier possible without repeating tokens (i.e., 80 for C4, 640 for
184

RedPajama, and 320 for RefinedWeb). We train N = 6.9B, M = 20 models on each dataset given
185

the relevance of 7B parameter models [113, 49]. In total this results in a testbed of 104 models.
186

3.3
Fitting scaling laws
187

We fit Equation (4) to approximate E, a, b, η using curve-fitting in SciPy [117] (i.e., Levenberg-
188

Marquardt to minimize non-linear least squares). We repeat this process to fit Equation (5) to
189

approximate ϵ, k, γ. We invest ∼100 A100 hours to train the models required to fit a scaling law for
190

loss and ∼1,000 A100 hours for a corresponding law for downstream error. Unless otherwise specified,
191

we fit to the N, M pairs in Table 1, which are a subset of our full testbed. Our configurations allow
192

us to test for extrapolation to the N = 1.4B, M = 640 (900B token) and the N = 6.9B, M = 20
193

(138B token) regimes.
194

6


---Page Break---
10

20

40

80

160

320

640

M

0.011B

0.079B

0.154B

0.411B

1.4B

6.9B

N

1.1% 0.0% 0.2% 0.7% 0.9% 0.0% 0.6%

2.6% 0.3% 0.2% 0.4% 0.1% 0.7% 0.8%

1.5% 0.5% 1.1% 1.1% 3.3% 2.8% 0.6%

0.5% 0.2% 0.0% 2.8% 0.2% 2.0%

0.8%
1.5%

4.3%

Train: C4
Eval: C4 eval

10

20

40

80

160

320

640

M

0.3% 0.0% 0.3% 1.7% 1.1% 0.0% 1.0%

2.2% 0.3% 0.2% 0.7% 1.4% 2.1% 2.3%

0.8% 0.5% 0.6% 0.0% 0.4% 0.4% 0.3%

0.4% 0.2% 0.1% 0.3% 0.3% 1.4% 1.1%

0.1%
0.7%

0.7%

Train: RedPajama

Eval: C4 eval

10

20

40

80

160

320

640

M

0.9% 0.0% 0.9% 1.9% 1.0% 0.0% 1.1%

2.4% 0.1% 0.0% 0.5% 1.2% 2.0% 0.9%

0.9% 0.2% 0.6% 2.8% 2.2% 0.8% 0.9%

0.2% 0.1% 0.5% 0.8% 0.9% 0.9% 0.3%

0.6%
0.0%

1.6%

Train: RefinedWeb

Eval: C4 eval

0.0%

2.0%

4.0%

6.0%

8.0%

10.0%

Relative error

Figure 5: Relative error on C4 eval for different training distributions. Boxes highlighted in
yellow correspond to pairs—number of parameters N, token multiplier M—used to fit Equation (4).
Larger values of M correspond to more over-training. The prediction error is low in both interpolation
and extrapolation ranges. Below N = 1.4B, empty squares correspond to runs that were not possible
due to the limited dataset size for single epoch training. At N = 1.4B we run at M = 20 and at the
largest possible multiplier. At N = 6.9B, we run at M = 20.

3.4
Evaluation setup
195

Evaluation datasets.
Unless otherwise stated, our default validation loss dataset is C4 eval. For
196

downstream tasks, we adopt a subset from 46 tasks from LLM-foundry [69], which includes standard
197

tasks with both zero-shot and few-shot evaluations. Specifically, we consider a 17-task subset where,
198

for each evaluation, at least one 0.154B scale model—trained with as many as 99B tokens—gets
199

10 percentage points above chance accuracy: ARC-Easy [23], BIG-bench: CS algorithms [11],
200

BIG-bench: Dyck languages [11], BIG-bench: Novel Concepts [11], BIG-bench: Operators [11],
201

BIG-bench: QA WikiData [11], BoolQ [21], Commonsense QA [107], COPA [92], CoQA [91],
202

HellaSwag (zero-shot) [126], HellaSwag (10-shot) [126], LAMBADA [77], PIQA [14], PubMed
203

QA Labeled [50], SQuAD [90], and WinoGrand [55]. For more details on evaluation datasets
204

see Appendix D. We focus on this subset to ensure we are measuring signal, not noise. Including
205

downstream tasks like MMLU [40], where performance is close to random chance, however, does
206

not invalidate our results as we show in our evaluation set ablations (Appendix E).
207

Metrics.
We consider three main metrics: Validation loss, which is the cross entropy between a
208

model’s output and the one-hot ground truth token, averaged over all tokens in a sequence and over
209

all sequences in a dataset. Average top-1 error, which is a uniform average over the 17 downstream
210

evaluations, as mentioned in the above paragraph. To measure how good a prediction ζ(C, M) is,
211

we measure Relative prediction error: |ζ(C, M) −ζGT |/ζGT , where ζ is the predicted loss L or the
212

average top-1 error Err. ζGT is the ground truth measurement to predict.
213

4
Results: Reliable extrapolation
214

In this Section, we quantify the extent to which the scaling laws developed in Section 2 extrapolate
215

larger model performance using the scaling testbed from Section 3. By default, we fit Equations (4)
216

and (5) to the configurations in Table 1, use C4 eval for loss, and the 17-task split from Section 3.4
217

for average top-1 error.
218

Over-trained performance is predictable.
We highlight our main over-training results in
219

Figure 1 (left). Namely, we are able to extrapolate both in the number of parameters N and the
220

token multiplier M to closely predict the C4 eval performance of a 1.4B parameter model trained on
221

900B RedPajama tokens (N = 1.4B, M = 640). Our prediction, which takes 300× less compute
222

to construct than the final 1.4B run, is accurate to within 0.7% relative error. Additionally, for the
223

N = 6.9B, M = 20 run, near compute-optimal, the relative error is also 0.7%.
224

These results support several key takeaways. (i) Scaling can be predictable even when one increases
225

both the model size and the amount of over-training compared to the training runs used to fit a scaling
226

law. (ii) The form presented in Equation (4) is useful in practice for predicting over-trained scaling
227

behavior. (iii) Fitting to Equation (4) gives good prediction accuracy near compute-optimal. More
228

7


---Page Break---
Table 2: Downstream relative prediction error at 6.9B parameters and 138B tokens. While
predicting accuracy on individual zero-shot downstream evaluations can be challenging (“Individual”),
predicting averages across downstream datasets is accurate (“Avg.”).

Individual top-1 error
Avg. top-1 error

Train set
ARC-E [23]
LAMBADA [77]
OpenBook QA [68]
HellaSwag [126]
17-task split

C4 [88, 27]
28.96%
15.01%
16.80%
79.58%
0.14%
RedPajama [112]
5.21%
14.39%
8.44%
25.73%
0.05%
RefinedWeb [82]
26.06%
16.55%
1.92%
81.96%
2.94%

specifically, predictions are accurate both for the 1.4B over-trained model and the 6.7B compute-
229

optimal model using a single scaling fit.
230

While Figure 1 explores a specific case of making predictions in the over-trained regime, we aim to
231

understand the error profile of our predictions across training datasets, token multipliers, and number
232

of parameters. Hence, Figure 5 shows the relative error between ground truth loss and predicted
233

loss on C4 eval for models in our testbed. We notice uniformly low prediction error suggesting that
234

predictions are accurate in many settings.
235

Average top-1 error is predictable.
Figure 1 (right) presents our main result in estimating scaling
236

laws for downstream error. Concretely, we use the models indicated in Table 1 to fit Equations (4)
237

and (5), chaining the scaling fits to predict the average top-1 error as a function of training compute
238

C and the token multiplier M. Our fits allow us to predict, using 20× less compute, the downstream
239

performance of a 6.9B model trained on 138B RedPajama tokens to within 0.05% relative error and a
240

1.4B model trained on RedPajama 900B tokens to within 3.6% relative error.
241

Table 2 additionally shows the relative error of our downstream performance predictions for models
242

trained on C4, RedPajama, and RefinedWeb, indicating that our scaling law functional forms are
243

applicable on many training datasets. We note that while average accuracy is predictable, individual
244

downstream task predictions are significantly more noisy. We report relative error for more model
245

predictions in Figures 11 and 12. We also find that if we remove the 1.4B model for the Equation (5)
246

fit, relative error jumps, for instance, from 0.05% to 10.64% on the 17-task split for the 6.9B,
247

138B token RedPajama prediction. This highlights the importance of investing more compute when
248

constructing scaling laws for downstream task prediction compared to loss prediction.
249

Under-training, out-of-distribution scaling, and compute-reliability trade-offs.
In addition to
250

our main results presented above, we include additional results in Appendix E, which we summarize
251

here. First, we notice that when token multipliers become too small (i.e., M = 5) scaling becomes
252

unreliable and lies off the trend. Additionally, multipliers other than 20, such as 10, 40, and 80, garner
253

points that are roughly on the compute optimal frontier (Figure 9). This observation suggests that the
254

compute-optimal multiplier may lie in a range rather than take a single value. To probe the limits
255

of reliable scaling, we attempt to break our scaling laws in out-of-distribution settings. We find that
256

models trained on C4—English filtered—and evaluated on next token prediction on code domains
257

have a high relative error in many cases. Perhaps surprisingly, evaluating the same models on German
258

next token prediction gives reliable loss scaling (Figure 10). We additionally examine the compute
259

necessary to create accurate scaling laws, finding that scaling laws can be constructed more cheaply
260

for loss prediction than for downstream error prediction (Figures 15 and 16).
261

5
Related work
262

We review the most closely related work in this section. For additional related work, see Appendix F.
263

Scaling laws.
Early works on scaling artificial neural networks observe predictable power-law
264

scaling in the training set size and number of model parameters [43, 44, 93]. Alabdulmohsin et al.
265

[2] stress the importance of looking at the extrapolation regime of a scaling law. Yang et al. [124]
266

prescribe architectural and hyperparameter changes when scaling model width to realize performant
267

models; Yang et al. [125] make analogous recommendations when scaling model depth. Bi et al.
268

[13] propose hyperparameter aware scaling laws. Unlike the aforementioned work, our investigation
269

focuses on over-training and predicting downstream accuracy.
270

Hoffmann et al. [45] investigate how the number of model parameters N and training tokens D
271

should be chosen to minimize loss L given a compute budget C. Hoffmann et al. [45] find that when
272

scaling up C, both N and D should be scaled equally up to a multiplicative constant (i.e., N ∝C∼0.5
273

8


---Page Break---
and D ∝C∼0.5) to realize compute-optimality. Appendix C of the Chinchilla paper additionally
274

suggests that these findings hold across three datasets. However, Hoffmann et al. [45] do not verify
275

their scaling laws for training beyond compute-optimality, or for downstream error prediction—both
276

of which are central to our work.
277

Sardana & Frankle [98] propose modifications to the Chinchilla formulation to incorporate inference
278

costs into the definition of compute-optimality and solve for various fixed inference budgets. Their
279

key finding, which is critical for our work, is that when taking into account a large enough inference
280

budget, it is optimal to train smaller models for longer than the original Chinchilla recommendations.
281

Our work presupposes that over-training can be beneficial.
Instead of solving for inference-
282

optimal schemes, we support empirically a predictive theory of scaling in the over-trained regime.
283

Additionally, we provide experiments across many validation and training sets.
284

For predicting downstream scaling beyond loss, Isik et al. [47] relate the number of pre-training tokens
285

to downstream cross-entropy and machine translation BLEU score [78] after fine-tuning. In contrast,
286

we take a holistic approach to evaluation by looking at top-1 error over many natural language tasks.
287

Schaeffer et al. [100] argue that emergent abilities [120] are a product of non-linear metrics and
288

propose smoother alternatives. As a warmup for why non-linear metrics may be hard to predict,
289

Schaeffer et al. [100] consider predicting an ℓlength sequence exactly: Err(N, ℓ) ≈1 −PP(N)−ℓ,
290

where N is the number of parameters in a model and PP is its perplexity. This is a special case of
291

our Equations (5) and (6), where the number of training tokens does not appear, ϵ = 1, k = 1, and
292

γ = ℓ. In contrast, we treat ϵ, k, γ as free parameters for a scaling law fit, finding that average error
293

over downstream tasks can make for a predictable metric.
294

Over-training in popular models.
There has been a rise in over-trained models [113, 114] and
295

accompanying massive datasets [112, 82, 104, 3]. For example, Chinchilla 70B [45] is trained with a
296

token multiplier of 20, while LLaMA-2 7B [114] uses a token multiplier of 290. In our investigation,
297

we look at token multipliers from 5 to 640 to ensure coverage of popular models and relevance for
298

future models that may be trained on even more tokens.
299

6
Limitations, future work, and conclusion
300

Limitations and future work.
We identify limitations, which provide motivation for future work.
301

• Hyperparameters. While our configurations are surprisingly amenable to reliable scaling across
302

many training and testing distributions without further tuning, there is a need to develop scaling
303

laws that do not require extensive hyperparameter sweeps.
304

• Scaling up. Validating the trends in this paper for even larger runs is a valuable direction.
305

Additionally, repeating our setup for models that achieve non-trivial performance on harder
306

evaluations like MMLU is left to future work.
307

• Scaling down. Actualizing predictable scaling with even cheaper runs is important to make this
308

area of research more accessible, especially for downstream error prediction.
309

• Failure cases. While we present a preliminary analysis of when scaling is unreliable, future work
310

should investigate conditions under which scaling breaks down.
311

• Post-training. It is common to employ fine-tuning interventions after pre-training, which we do
312

not consider. Quantifying to what degree over-training the base model provides benefits after
313

post-training is an open area of research.
314

• Individual downstream task prediction. While we find that averaging over many task error
315

metrics can make for a predictable metric, per-task predictions are left to future work.
316

• In-the-wild performance. Downstream task performance is a proxy for the in-the-wild user
317

experience. Analyzing scaling trends in the context of this experience is timely.
318

• Dataset curation. Our work only deals with existing training datasets. Exploring dataset curation
319

for improved model scaling is another promising direction.
320

Conclusion.
We show that the loss of over-trained models, trained past compute-optimality, is
321

predictable. Furthermore, we propose and validate a scaling law relating loss to average downstream
322

task performance. We hope our work will inspire others to further examine the relationship between
323

model training and downstream generalization. Our testbed will be made publicly available, and we
324

hope it will make scaling research more accessible to researchers and practitioners alike.
325

9


---Page Break---
References
326

[1] Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, and Hanie Sedghi. Exploring the limits
327

of large scale pre-training. In International Conference on Learning Representations (ICLR),
328

2022. https://arxiv.org/abs/2110.02095.
329

[2] Ibrahim Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling
330

laws in language and vision. In Advances in Neural Information Processing Systems (NeuIPS),
331

2022. https://arxiv.org/abs/2209.06640.
332

[3] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi
333

Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on
334

data selection for language models. arXiv preprint, 2024. https://arxiv.org/abs/2402.
335

16827.
336

[4] Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki,
337

Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al.
338

Santacoder: don’t reach for the stars! arXiv preprint, 2023. https://arxiv.org/abs/
339

2301.03988.
340

[5] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and
341

Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with
342

operation-based formalisms. In Conference of the North American Chapter of the Association
343

for Computational Linguistics (NACCL), 2019. https://aclanthology.org/N19-1245.
344

[6] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael
345

Voznesensky, Bin Bao, David Berard, Geeta Chauhan, Anjali Chourdia, Will Constable,
346

Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind,
347

Brian Hirsh, Sherlock Huang, Laurent Kirsch, Michael Lazos, Yanbo Liang, Jason Liang,
348

Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark
349

Saroufim, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William
350

Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan,
351

Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic
352

python bytecode transformation and graph compilation. In International Conference on
353

Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2024.
354

https://pytorch.org/blog/pytorch-2-paper-tutorial.
355

[7] Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer,
356

Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giridharan Anantharaman,
357

Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh
358

Koura, Brian O’Horo, Jeffrey Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, and
359

Veselin Stoyanov. Efficient large scale language modeling with mixtures of experts. In
360

Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022. https:
361

//aclanthology.org/2022.emnlp-main.804.
362

[8] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint,
363

2016. https://arxiv.org/abs/1607.06450.
364

[9] Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining
365

neural scaling laws. arXiv preprint, 2021. https://arxiv.org/abs/2102.06701.
366

[10] Yamini Bansal, Behrooz Ghorbani, Ankush Garg, Biao Zhang, Maxim Krikun, Colin Cherry,
367

Behnam Neyshabur, and Orhan Firat. Data scaling laws in nmt: The effect of noise and
368

architecture. In International Conference on Machine Learning (ICML), 2022. https:
369

//proceedings.mlr.press/v162/bansal22b.html.
370

[11] BIG bench authors. Beyond the imitation game: Quantifying and extrapolating the capabilities
371

of language models. In Transactions on Machine Learning Research (TMLR), 2023. https:
372

//openreview.net/forum?id=uyTL5Bvosj.
373

[12] Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On
374

the dangers of stochastic parrots: Can language models be too big? In Proceedings ACM
375

conference on fairness, accountability, and transparency (FAccT), 2021. https://dl.acm.
376

org/doi/10.1145/3442188.3445922.
377

10


---Page Break---
[13] DeepSeek-AI Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi
378

Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao,
379

Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He,
380

Wen-Hui Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng
381

Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu
382

Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui
383

Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Jun-Mei Song,
384

Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Min Tang, Bing-Li Wang, Peiyi Wang, Shiyu
385

Wang, Yaohui Wang, Yongji Wang, Tong Wu, Yu Wu, Xin Xie, Zhenda Xie, Ziwei Xie,
386

Yi Xiong, Hanwei Xu, Ronald X Xu, Yanhong Xu, Dejian Yang, Yu mei You, Shuiping Yu,
387

Xin yuan Yu, Bo Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang,
388

Minghu Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou,
389

Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek llm: Scaling open-source language
390

models with longtermism. arXiv preprint, 2024. https://arxiv.org/abs/2401.02954.
391

[14] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning
392

about physical commonsense in natural language. In Association for the Advancement of
393

Artificial Intelligence (AAAI), 2020. https://arxiv.org/abs/1911.11641.
394

[15] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding,
395

Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai
396

Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel
397

Weinbach. Gpt-neox-20b: An open-source autoregressive language model. BigScience
398

Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models,
399

2022. https://aclanthology.org/2022.bigscience-1.9.
400

[16] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
401

Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini
402

Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya
403

Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric
404

Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam
405

McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-
406

shot learners.
In Advances in Neural Information Processing Systems (NeurIPS), 2020.
407

https://arxiv.org/abs/2005.14165.
408

[17] Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. Broken neural scaling laws. In
409

International Conference on Learning Representations (ICLR), 2023. https://openreview.
410

net/forum?id=sckjveqlCZ.
411

[18] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade
412

Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling
413

laws for contrastive language-image learning. In Conference on Computer Vision and Pattern
414

Recognition (CVPR), 2023. https://arxiv.org/abs/2212.07143.
415

[19] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
416

Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh,
417

Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay,
418

Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson,
419

Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju
420

Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García,
421

Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan,
422

Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani
423

Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie
424

Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee,
425

Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason
426

Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel.
427

Palm: Scaling language modeling with pathways. In Journal of Machine Learning Research
428

(JMLR), 2022. https://arxiv.org/abs/2204.02311.
429

[20] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan
430

Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned
431

language models. arXiv preprint, 2022. https://arxiv.org/abs/2210.11416.
432

11


---Page Break---
[21] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and
433

Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In
434

Conference of the North American Chapter of the Association for Computational Linguistics
435

(NAACL), 2019. https://aclanthology.org/N19-1300.
436

[22] Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. ELECTRA:
437

Pre-training text encoders as discriminators rather than generators.
In International
438

Conference on Learning Representations (ICLR), 2020. https://openreview.net/pdf?
439

id=r1xMH1BtvB.
440

[23] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
441

and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
442

challenge. arXiv preprint, 2018. https://arxiv.org/abs/1803.05457.
443

[24] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast
444

and memory-efficient exact attention with IO-awareness. In Advances in Neural Information
445

Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/2205.14135.
446

[25] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin
447

Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al.
448

Scaling vision transformers to 22 billion parameters. In International Conference on Machine
449

Learning (ICML), 2023. https://proceedings.mlr.press/v202/dehghani23a.html.
450

[26] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training
451

of deep bidirectional transformers for language understanding. In Conference of the North
452

American Chapter of the Association for Computational Linguistics (NAACL), 2019. https:
453

//aclanthology.org/N19-1423.
454

[27] Jesse Dodge, Maarten Sap, Ana Marasovi´c, William Agnew, Gabriel Ilharco, Dirk Groeneveld,
455

Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on
456

the colossal clean crawled corpus. In Conference on Empirical Methods in Natural Language
457

Processing (EMNLP), 2021. https://aclanthology.org/2021.emnlp-main.98.
458

[28] Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu,
459

Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten
460

Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin
461

Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le,
462

Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language models
463

with mixture-of-experts. In International Conference on Machine Learning (ICML), 2022.
464

https://arxiv.org/abs/2112.06905.
465

[29] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto:
466

Model alignment as prospect theoretic optimization. arXiv preprint, 2024. https://arxiv.
467

org/abs/2402.01306.
468

[30] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao
469

Nguyen, Mitchell Wortsman Ryan Marten, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim
470

Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe,
471

Stephen Mussmann, Mehdi Cherti Richard Vencu, Ranjay Krishna, Pang Wei Koh, Olga
472

Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont,
473

Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt.
474

Datacomp: In search of the next generation of multimodal datasets. In Advances in Neural
475

Information Processing Systems (NeurIPS), 2023. https://arxiv.org/abs/2304.14108.
476

[31] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster,
477

Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy.
478

The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint, 2020.
479

https://arxiv.org/abs/2101.00027.
480

[32] Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia,
481

Ciprian Chelba, and Colin Cherry. Scaling laws for neural machine translation. arXiv preprint,
482

2021. https://arxiv.org/abs/2109.07740.
483

12


---Page Break---
[33] Mitchell A Gordon, Kevin Duh, and Jared Kaplan. Data and parameter scaling laws for neural
484

machine translation. In Conference on Empirical Methods in Natural Language Processing
485

(EMNLP), 2021. https://aclanthology.org/2021.emnlp-main.478.
486

[34] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord,
487

Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating
488

the science of language models. arXiv preprint, 2024. https://arxiv.org/abs/2402.
489

00838.
490

[35] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.
491

arXiv preprint, 2023. https://arxiv.org/abs/2312.00752.
492

[36] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher
493

Ré. Combining recurrent, convolutional, and continuous-time models with linear state space
494

layers. In Advances in Neural Information Processing Systems (NeurIPS), 2021. https:
495

//openreview.net/forum?id=yWd42CWN3c.
496

[37] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with
497

structured state spaces. In International Conference on Learning Representations (ICLR),
498

2022. https://arxiv.org/abs/2111.00396.
499

[38] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar, Teodoro Mendes, Allie Del Giorno,
500

Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi,
501

Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen
502

Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li.
Textbooks are all you
503

need. Preprint, 2023. https://www.microsoft.com/en-us/research/publication/
504

textbooks-are-all-you-need.
505

[39] Suchin Gururangan, Mitchell Wortsman, Samir Yitzhak Gadre, Achal Dave, Maciej Kilian,
506

Weijia Shi, Jean Mercat, Georgios Smyrnis, Gabriel Ilharco, Matt Jordan, Reinhard
507

Heckel, Alex Dimakis, Ali Farhadi, Vaishaal Shankar, and Ludwig Schmidt. OpenLM:
508

a minimal but performative language modeling (lm) repository, 2023. https://github.
509

com/mlfoundations/open_lm.
510

[40] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
511

Jacob Steinhardt. Measuring massive multitask language understanding. In International
512

Conference on Learning Representations (ICLR), 2021. https://arxiv.org/abs/2009.
513

03300.
514

[41] T. J. Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson,
515

Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann,
516

Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei,
517

and Sam McCandlish. Scaling laws for autoregressive generative modeling. arXiv preprint,
518

2020. https://arxiv.org/abs/2010.14701.
519

[42] Danny Hernandez, Jared Kaplan, T. J. Henighan, and Sam McCandlish. Scaling laws for
520

transfer. arXiv preprint, 2021. https://arxiv.org/abs/2102.01293.
521

[43] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Frederick Diamos, Heewoo Jun,
522

Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning
523

scaling is predictable, empirically. arXiv preprint, 2017. https://arxiv.org/abs/1712.
524

00409.
525

[44] Joel Hestness, Newsha Ardalani, and Gregory Diamos.
Beyond human-level accuracy:
526

Computational challenges in deep learning.
In Principles and Practice of Parallel
527

Programming (PPoPP), 2019. https://arxiv.org/abs/1909.01736.
528

[45] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai,
529

Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark,
530

et al. Training compute-optimal large language models. In Advances in Neural Information
531

Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/2203.15556.
532

13


---Page Break---
[46] Hakan Inan, Khashayar Khosravi, and Richard Socher.
Tying word vectors and word
533

classifiers: A loss framework for language modeling. In International Conference on Learning
534

Representations (ICLR), 2017. https://arxiv.org/abs/1611.01462.
535

[47] Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii,
536

and Sanmi Koyejo. Scaling laws for downstream task performance of large language models.
537

arXiv, 2024. https://arxiv.org/abs/2402.04177.
538

[48] Maor Ivgi, Yair Carmon, and Jonathan Berant. Scaling laws under the microscope: Predicting
539

transformer performance from small scale experiments. In Conference on Empirical Methods
540

in Natural Language Processing (EMNLP), 2022. https://aclanthology.org/2022.
541

findings-emnlp.544.
542

[49] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
543

Chaplot, Florian Bressand Diego de las Casas, Gianna Lengyel, Guillaume Lample, Lucile
544

Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut
545

Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint,
546

2023. https://arxiv.org/abs/2310.06825.
547

[50] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A
548

dataset for biomedical research question answering. In Conference on Empirical Methods in
549

Natural Language Processing (EMNLP), 2019. https://aclanthology.org/D19-1259.
550

[51] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon
551

Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural
552

language models. arXiv preprint, 2020. https://arxiv.org/abs/2001.08361.
553

[52] Tobit Klug, Dogukan Atik, and Reinhard Heckel. Analyzing the sample complexity of self-
554

supervised image reconstruction methods. arXiv preprint, 2023. https://arxiv.org/abs/
555

2305.19079.
556

[53] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu
557

Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. arXiv
558

preprint, 2019. http://arxiv.org/abs/1909.11942.
559

[54] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano,
560

Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel
561

Haziza. xformers: A modular and hackable transformer modelling library, 2022. https:
562

//github.com/facebookresearch/xformers.
563

[55] Hector Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In
564

International conference on the principles of knowledge representation and reasoning, 2012.
565

https://aaai.org/papers/59-4492-the-winograd-schema-challenge.
566

[56] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed,
567

Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer.
BART: Denoising sequence-to-
568

sequence pre-training for natural language generation, translation, and comprehension. In
569

Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
https:
570

//aclanthology.org/2020.acl-main.703.
571

[57] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao
572

Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source
573

be with you! arXiv preprint, 2023. https://arxiv.org/abs/2305.06161.
574

[58] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A
575

challenge dataset for machine reading comprehension with logical reasoning. In International
576

Joint Conference on Artificial Intelligence, 2020. https://arxiv.org/abs/2007.08124.
577

[59] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy,
578

Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT
579

pretraining approach. arXiv preprint, 2019. http://arxiv.org/abs/1907.11692.
580

14


---Page Break---
[60] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining
581

Xie. A convnet for the 2020s. Conference on Computer Vision and Pattern Recognition
582

(CVPR), 2022. https://arxiv.org/abs/2201.03545.
583

[61] Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William
584

Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, et al. The
585

data provenance initiative: A large scale audit of dataset licensing & attribution in ai. arXiv
586

preprint, 2023. https://arxiv.org/abs/2310.16787.
587

[62] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint,
588

2017. https://arxiv.org/abs/1711.05101.
589

[63] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier,
590

Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian,
591

Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov,
592

Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo,
593

Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan
594

Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang,
595

Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra,
596

Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu,
597

Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane
598

Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz
599

Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra,
600

and Harm de Vries. Starcoder 2 and the stack v2: The next generation. arXiv preprint, 2024.
601

https://arxiv.org/abs/2402.19173.
602

[64] Risto Luukkonen, Ville Komulainen, Jouni Luoma, Anni Eskelinen, Jenna Kanerva, Hanna-
603

Mari Kupari, Filip Ginter, Veronika Laippala, Niklas Muennighoff, Aleksandra Piktus, et al.
604

Fingpt: Large generative models for a small language. In Conference on Empirical Methods
605

in Natural Language Processing (EMNLP), 2023. https://aclanthology.org/2023.
606

emnlp-main.164.
607

[65] Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind
608

Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, Dirk Groenveld, Iz Beltagy,
609

Hanneneh Hajishirz, Noah A. Smith, Kyle Richardson, and Jesse Dodge. Paloma: A benchmark
610

for evaluating language model fit. arXiv preprint, 2023. https://paloma.allen.ai.
611

[66] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large
612

annotated corpus of English: The Penn Treebank.
In Computational Linguistics, 1993.
613

https://aclanthology.org/J93-2004.
614

[67] William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah A. Smith.
615

Effects of parameter norm growth during transformer training: Inductive bias from gradient
616

descent. In Conference on Empirical Methods in Natural Language Processing (EMNLP),
617

2021. https://aclanthology.org/2021.emnlp-main.133.
618

[68] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
619

electricity? a new dataset for open book question answering. In Conference on Empirical
620

Methods in Natural Language Processing (EMNLP), 2018. https://arxiv.org/abs/1809.
621

02789.
622

[69] MosaicML. Llm evaluation scores, 2023. https://www.mosaicml.com/llm-evaluation.
623

[70] Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman,
624

Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al.
625

Crosslingual generalization through multitask finetuning.
In Annual Meeting of the
626

Association for Computational Linguistics (ACL), 2022. https://aclanthology.org/
627

2023.acl-long.891.
628

[71] Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue
629

Zhuo, Swayam Singh, Xiangru Tang, Leandro Von Werra, and Shayne Longpre. Octopack:
630

Instruction tuning code large language models. arXiv preprint, 2023. https://arxiv.org/
631

abs/2308.07124.
632

15


---Page Break---
[72] Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus,
633

Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained
634

language models. In Advances in Neural Information Processing Systems (NeuIPS), 2023.
635

https://arxiv.org/abs/2305.16264.
636

[73] Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet
637

Singh, and Douwe Kiela. Generative representational instruction tuning. arXiv preprint, 2024.
638

https://arxiv.org/abs/2402.09906.
639

[74] Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig,
640

Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech
641

Kryscinski, Lidiya Murakhovs’ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng,
642

Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Rayhan
643

Joty, and Caiming Xiong. Long sequence modeling with xgen: A 7b llm trained on 8k input
644

sequence length. arXiv preprint, 2023. https://arxiv.org/abs/2309.03450.
645

[75] OpenAI. Triton, 2021. https://github.com/openai/triton.
646

[76] OpenAI. Gpt-4 technical report, 2023. https://arxiv.org/abs/2303.08774.
647

[77] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella
648

Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The
649

LAMBADA dataset: Word prediction requiring a broad discourse context. In Annual Meeting
650

of the Association for Computational Linguistics (ACL), 2016. http://www.aclweb.org/
651

anthology/P16-1144.
652

[78] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic
653

evaluation of machine translation. In Annual Meeting of the Association for Computational
654

Linguistics (ACL), 2002. https://aclanthology.org/P02-1040.
655

[79] Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana
656

Thompson, Phu Mon Htut, and Samuel Bowman. BBQ: A hand-built bias benchmark for
657

question answering. In Annual Meeting of the Association for Computational Linguistics
658

(ACL), 2022. https://aclanthology.org/2022.findings-acl.165.
659

[80] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
660

Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
661

style, high-performance deep learning library. In Advances in Neural Information Processing
662

Systems (NeurIPS), 2019. https://arxiv.org/abs/1912.01703.
663

[81] Patronus AI. EnterprisePII dataset, 2023. https://tinyurl.com/2r5x9bst.
664

[82] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro
665

Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The
666

RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web
667

data only. arXiv preprint, 2023. https://arxiv.org/abs/2306.01116.
668

[83] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman,
669

Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella,
670

Kranthi Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong,
671

Bartłomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi
672

Saito, Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Wo´zniak, Zhenyuan Zhang,
673

Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer
674

era. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.
675

https://aclanthology.org/2023.findings-emnlp.936.
676

[84] Ofir Press and Lior Wolf. Using the output embedding to improve language models. In
677

Proceedings of the Conference of the European Chapter of the Association for Computational
678

Linguistics (EACL), 2017. https://aclanthology.org/E17-2025.
679

[85] Alec Radford,
Jeff Wu,
Rewon Child,
David Luan,
Dario Amodei,
and Ilya
680

Sutskever.
Language models are unsupervised multitask learners.
Preprint, 2019.
681

https://d4mucfpksywv.cloudfront.net/better-language-models/language_
682

models_are_unsupervised_multitask_learners.pdf.
683

16


---Page Break---
[86] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis
684

Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford,
685

Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche,
686

Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl,
687

Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, Irina Higgins,
688

Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar,
689

Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini,
690

L. Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena
691

Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau,
692

Maria Tsimpoukelli, N. K. Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Tobias
693

Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi,
694

Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris
695

Jones, James Bradbury, Matthew G. Johnson, Blake A. Hechtman, Laura Weidinger, Iason
696

Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol
697

Vinyals, Kareem W. Ayoub, Jeff Stanway, L. L. Bennett, Demis Hassabis, Koray Kavukcuoglu,
698

and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training
699

gopher. arXiv preprint, 2021. https://arxiv.org/abs/2112.11446.
700

[87] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
701

Chelsea Finn. Direct preference optimization: Your language model is secretly a reward
702

model. In Advances in Neural Information Processing Systems (NeurIPS), 2023. https:
703

//arxiv.org/abs/2305.18290.
704

[88] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
705

Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified
706

text-to-text transformer. arXiv preprint, 2019. https://arxiv.org/abs/1910.10683.
707

[89] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
708

Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified
709

text-to-text transformer. In The Journal of Machine Learning Research (JMLR), 2020. https:
710

//arxiv.org/abs/1910.10683.
711

[90] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+
712

questions for machine comprehension of text. In Conference on Empirical Methods in Natural
713

Language Processing (EMNLP), 2016. https://aclanthology.org/D16-1264.
714

[91] Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: A conversational question
715

answering challenge. In Transactions of the Association for Computational Linguistics (TACL),
716

2019. https://aclanthology.org/Q19-1016.
717

[92] Melissa Roemmele, Cosmin Adrian Bejan, , and Andrew S. Gordon. Choice of plausible
718

alternatives: An evaluation of commonsense causal reasoning.
In Association for the
719

Advancement of Artificial Intelligence (AAAI) Spring Symposium, 2011. https://people.
720

ict.usc.edu/~gordon/copa.html.
721

[93] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive
722

prediction of the generalization error across scales. In International Conference on Learning
723

Representations (ICLR), 2020. https://arxiv.org/abs/1909.12673.
724

[94] Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. Gender bias
725

in coreference resolution. In Conference of the North American Chapter of the Association for
726

Computational Linguistics (NAACL), 2018. https://aclanthology.org/N18-2002.
727

[95] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An
728

adversarial winograd schema challenge at scale. arXiv preprint, 2019. https://arxiv.org/
729

abs/1907.10641.
730

[96] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled
731

version of bert: smaller, faster, cheaper and lighter. arXiv preprint, 2019. http://arxiv.
732

org/abs/1910.01108.
733

17


---Page Break---
[97] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa:
734

Commonsense reasoning about social interactions. In Empirical Methods in Natural Language
735

Processing (EMNLP), 2019. https://aclanthology.org/D19-1454.
736

[98] Nikhil Sardana and Jonathan Frankle. Beyond chinchilla-optimal: Accounting for inference
737

in language model scaling laws. In NeurIPS Workshop on Efficient Natural Language and
738

Speech Processing (ENLSP), 2023. https://arxiv.org/abs/2401.00448.
739

[99] Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari,
740

Stella Biderman, Hady Elsahar, Niklas Muennighoff, Jason Phang, et al. What language
741

model to train if you have one million gpu hours? In Conference on Empirical Methods
742

in Natural Language Processing (EMNLP), 2022. https://aclanthology.org/2022.
743

findings-emnlp.54.
744

[100] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language
745

models a mirage? In Advances in Neural Information Processing Systems (NeurIPS), 2023.
746

https://arxiv.org/abs/2304.15004.
747

[101] Utkarsh Sharma and Jared Kaplan. A neural scaling law from the dimension of the data
748

manifold. In Journal of Machine Learning Research (JMLR), 2022. https://arxiv.org/
749

abs/2004.10802.
750

[102] Noam Shazeer. Glu variants improve transformer. arXiv preprint, 2020. https://arxiv.
751

org/abs/2002.05202.
752

[103] Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F Karlsson, Abinaya Mahendiran,
753

Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, et al.
754

Aya dataset: An open-access collection for multilingual instruction tuning. arXiv preprint
755

arXiv:2402.06619, 2024. https://arxiv.org/abs/2402.06619.
756

[104] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell
757

Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open
758

corpus of three trillion tokens for language model pretraining research. arXiv preprint, 2024.
759

https://arxiv.org/abs/2402.00159.
760

[105] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond
761

neural scaling laws: beating power law scaling via data pruning. In Advances in Neural
762

Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id=
763

UmvSlP-PyV.
764

[106] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer:
765

Enhanced transformer with rotary position embedding. arXiv preprint, 2021. https://
766

arxiv.org/abs/2104.09864.
767

[107] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA:
768

A question answering challenge targeting commonsense knowledge. In Conference of the
769

North American Chapter of the Association for Computational Linguistics (NAACL), 2019.
770

https://aclanthology.org/N19-1421.
771

[108] Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won
772

Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, and Donald Metzler.
Scale
773

efficiently:
Insights from pre-training and fine-tuning transformers.
In International
774

Conference on Learning Representations (ICLR), 2022. https://openreview.net/forum?
775

id=f2OYVDyfIB.
776

[109] Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Chung, William Fedus, Jinfeng Rao,
777

Sharan Narang, Vinh Tran, Dani Yogatama, and Donald Metzler. Scaling laws vs model
778

architectures: How does inductive bias influence scaling?
In Conference on Empirical
779

Methods in Natural Language Processing (EMNLP), 2023. https://aclanthology.org/
780

2023.findings-emnlp.825.
781

[110] MosaicML NLP Team. Introducing mpt-7b: A new standard for open-source, commercially
782

usable llms, 2023. www.mosaicml.com/blog/mpt-7b.
783

18


---Page Break---
[111] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha,
784

Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee,
785

Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun,
786

Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts,
787

Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch,
788

Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel
789

Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen,
790

Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina,
791

Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm,
792

Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-
793

Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog
794

applications. arXiv preprint, 2022. https://arxiv.org/abs/2201.08239.
795

[112] Together Computer. Redpajama: an open dataset for training large language models, 2023.
796

https://github.com/togethercomputer/RedPajama-Data.
797

[113] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux,
798

Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien
799

Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and
800

Efficient Foundation Language Models. arXiv preprint, 2023. https://arxiv.org/abs/
801

2302.13971.
802

[114] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine
803

Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel,
804

Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude
805

Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman
806

Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor
807

Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne
808

Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier
809

Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton,
810

Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael
811

Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams,
812

Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie
813

Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas
814

Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint, 2023.
815

https://arxiv.org/abs/2307.09288.
816

[115] Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke
817

Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al.
Aya model:
818

An instruction finetuned open-access multilingual language model. arXiv preprint, 2024.
819

https://arxiv.org/abs/2402.07827.
820

[116] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
821

Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural
822

Information Processing Systems (NeurIPS), 2017. https://arxiv.org/abs/1706.03762.
823

[117] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David
824

Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J.
825

van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew
826

R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng, Eric W.
827

Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A.
828

Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul
829

van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific
830

Computing in Python. Nature Methods, 2020. https://rdcu.be/b08Wh.
831

[118] Siyuan Wang, Zhongkun Liu, Wanjun Zhong, Ming Zhou, Zhongyu Wei, Zhumin Chen, and
832

Nan Duan. From lsat: The progress and challenges of complex reasoning. Transactions on
833

Audio, Speech, and Language Processing, 2021. https://arxiv.org/abs/2108.00648.
834

[119] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan
835

Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In
836

19


---Page Break---
International Conference on Learning Representations (ICLR), 2022. https://openreview.
837

net/forum?id=gEZrGCozdqR.
838

[120] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani
839

Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto,
840

Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus.
Emergent abilities of large
841

language models. In Transactions on Machine Learning Research (TMLR), 2022. https:
842

//openreview.net/forum?id=yzkSU5zdwD.
843

[121] Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang,
844

Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. Ethical and social risks of
845

harm from language models. arXiv preprint, 2021. https://arxiv.org/abs/2112.04359.
846

[122] BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana
847

Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al.
848

Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint, 2022.
849

https://arxiv.org/abs/2211.05100.
850

[123] Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D
851

Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for
852

large-scale transformer training instabilities. arXiv preprint, 2023. https://arxiv.org/
853

abs/2309.14322.
854

[124] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick
855

Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs V: Tuning large
856

neural networks via zero-shot hyperparameter transfer. In Advances in Neural Information
857

Processing Systems (NeuIPS), 2021. https://arxiv.org/abs/2203.03466.
858

[125] Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Feature learning in infinite depth
859

neural networks. In International Conference on Learning Representations (ICLR), 2024.
860

https://openreview.net/forum?id=17pVDnpwwl.
861

[126] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a
862

machine really finish your sentence? In Annual Meeting of the Association for Computational
863

Linguistics (ACL), 2019. https://aclanthology.org/P19-1472.
864

[127] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer.
Scaling vision
865

transformers. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
866

https://arxiv.org/abs/2106.04560.
867

[128] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Advances in Neural
868

Information Processing Systems (NeuIPS), 2019. https://arxiv.org/abs/1910.07467.
869

[129] Biao Zhang, Ivan Titov, and Rico Sennrich. Improving deep transformer with depth-scaled
870

initialization and merged attention. In Empirical Methods in Natural Language Processing
871

(EMNLP), 2019. https://aclanthology.org/D19-1083.
872

[130] Yanli Zhao, Andrew Gu, Rohan Varma, Liangchen Luo, Chien chin Huang, Min Xu, Less
873

Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Bernard
874

Nguyen, Geeta Chauhan, Yuchen Hao, and Shen Li. Pytorch fsdp: Experiences on scaling
875

fully sharded data parallel. In Very Large Data Bases Conference (VLDB), 2023. https:
876

//dl.acm.org/doi/10.14778/3611540.3611569.
877

[131] Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun.
878

Jec-qa: A legal-domain question answering dataset. In Association for the Advancement of
879

Artificial Intelligence (AAAI), 2020. https://arxiv.org/abs/1911.12011.
880

[132] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied,
881

Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation
882

models. arXiv preprint, 2023. https://arxiv.org/abs/2304.06364.
883

[133] Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries,
884

Qian Liu, and Niklas Muennighoff. Astraios: Parameter-efficient instruction tuning code large
885

language models. arXiv preprint, 2024. https://arxiv.org/abs/2401.00788.
886

20


---Page Break---
Contents
887

1
Introduction
1
888

2
Developing scaling laws for over-training and downstream tasks
2
889

2.1
Preliminaries
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3
890

2.2
Scaling laws for over-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3
891

2.3
Scaling laws for downstream error . . . . . . . . . . . . . . . . . . . . . . . . . .
4
892

3
Constructing a scaling testbed
5
893

3.1
Training setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5
894

3.2
Model configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5
895

3.3
Fitting scaling laws . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6
896

3.4
Evaluation setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
7
897

4
Results: Reliable extrapolation
7
898

5
Related work
8
899

6
Limitations, future work, and conclusion
9
900

A Scaling-law derivations
22
901

B Additional training details
23
902

C Additional grid search details
23
903

D Evaluation dataset details
23
904

E Additional results
23
905

F
Additional related work
30
906

G Broader impact
31
907

H Licensing
31
908

21


---Page Break---
A
Scaling-law derivations
909

We first show that reparameterizing Equation (3) in terms of the compute C and token multiplier M
910

for α = β yields Equation (4). Combining C = 6ND and M = D/N yields N =
p

C/(6M) and
911

D =
p

CM/6. Inserting these into Equation (3) yields,
912

L(C, M) = E + A
 C

6M

−α

2
+ B
CM

6

−α

2
,

= E +

 

A
1

6

−α

2
M
α

2 + B
1

6

−α

2
M −α

2

!

C−α

2 .

This is equal to Equation (4), making the substitutions η = α/2, a = A(1/6)−η, b = B(1/6)−η, as
913

noted in the main body.
914

Relation to compute-optimal training.
Recall that we made the assumption α = β, which implies
915

equal scaling of parameters and tokens to realize compute-optimal models. While this assumption
916

is empirically justified [45], even if α ̸= β, we get a parameterization that implies the power law
917

exponent in Equation (4) remains constant with over-training, while the power law scalar changes.
918

To find a compute-optimal training setting, Hoffmann et al. [45] propose to minimize the right-hand
919

side of Equation (3) subject to the compute constraint C = 6ND. This yields, N ∗= γ
1
α+β (C/6)

β
α+β
920

and D∗= γ−
1
α+β (C/6)
α
α+β , where γ = αA

βB , for notational convenience. The associated risk is,
921

L(N ∗, D∗) = E +

Aγ

−α
β+α + Bγ

β
β+α
 C

6

−αβ

α+β
.

We now deviate from compute-optimal training by modifying the model size and tokens by
922

multiplication with a constant √m, according to
923

Nm =
1
√mN ∗,
Dm = √mD∗.
(7)

This modification keeps the compute constant (i.e., 6NmDm = 6N ∗D∗). The risk, then, becomes
924

L(fNm,Dm) = E +

m
α

2 Aγ

−α
β+α + m−β

2 Bγ

β
β+α

C−αβ

α+β .
(8)

We again expect the same power law exponent and changing power law scalar. Note that m in
925

Equation (8) is similar to M in Equation (4). Specifically, m is a multiple of the Chinchilla-optimal
926

token multiplier M ∗= D∗/N ∗, which is no longer fixed as a compute budget changes for α ̸= β.
927

22


---Page Break---
Table 3: Main models and hyperparameters used in our investigation. Models have number of
parameters N, with number of layers nlayers, number of attention heads nheads, model width dmodel,
and width per attention head dhead. Batch sizes are global and in units of sequences. Each sequence
has 2,048 tokens. A100 GPU hours are at M = 20, which are near compute-optimal runs. For the
1.4B scale, a batch size of 256 performs slightly better than 512.

N
nlayers
nheads
dmodel
dhead
Warmup
Learning rate
Batch size
M = 20 A100 hours

0.011B
8
4
96
24
100
3e-3
64
0.3
0.079B
8
4
512
128
400
3e-3
512
5
0.154B
24
8
576
72
400
3e-3
512
12
0.411B
24
8
1,024
128
2,000
3e-3
512
75
1.4B
24
16
2,048
128
5,000
3e-3
256
690
6.9B
32
32
4,096
128
5,000
3e-4
2,048
17,000

B
Additional training details
928

Architecture.
As stated in the main paper, we train transformers [116], based on auto-
929

regressive, decoder-only, pre-normalization architectures like GPT-2 [85] and LLaMA [113]. We
930

adopt OpenLM [39] for modeling, which utilizes PyTorch [80, 6], xformers [54], triton [75],
931

FlashAttention [24], FSDP [130], and bfloat16 automatic mixed precision. Like LLaMA, we omit
932

bias terms, but replace RMSNorm [128] with LayerNorm [8], which has readily available fused
933

implementations. Following Wortsman et al. [123], we apply qk-LayerNorm [25], which adds
934

robustness to otherwise poor hyperparameter choices (e.g., learning rate). We use SwiGLU [102]
935

activations and depth-scaled initialization [129]. We use a sequence length of 2,048, rotary positional
936

embeddings [106], and the GPT-NeoX-20B tokenizer [15], which yields a vocabulary size of 50k.
937

We do not use weight tying [84, 46]. We sample without replacement during training and employ
938

sequence packing without attention masking. We separate documents in our training corpora with
939

end-of-text tokens.
940

Objectives and optimization.
We train with a standard causal language modeling objective (i.e.,
941

next token prediction) with an additive z-loss [19] (coefficient 1e-4), which mitigates output logit
942

norm growth [67] instabilities. We use the AdamW optimizer [62] (PyTorch defaults except beta2 =
943

0.95), with independent weight decay [123] (coefficient 1e-4). For the learning rate schedule, we use
944

linear warmup and cosine decay. We cool down to a low learning rate (3e-5).
945

C
Additional grid search details
946

Final model configurations.
We present our final hyperparameters in Table 3.
947

Grid search configuration selection.
Recall in Section 3.3, we run a grid search over many
948

configurations. We present the architectures we sweep over in Table 4.
949

D
Evaluation dataset details
950

All 46 downstream evaluations are based on MosaicML’s LLM-foundry evaluation suite [69]. We
951

specifically consider the datasets given in Table 5.
Recall that we use a subset of 17 of these
952

evaluations that give signal (are above random chance) for the compute range we consider. See
953

Appendix E, where we ablate over the 17 subset design choice by including more and less evaluations.
954

E
Additional results
955

Scaling law fits.
We present specific coefficients for our fits in Table 6.
956

Small-scale experiments can predict model rank order.
We expect to be able to rank hypothetical
957

models based on their predicted performance, which is useful when deciding what large-scale runs
958

23


---Page Break---
1016
1017
1018
1019

Compute (6ND) [FLOPs]

3

4

5

Loss: OpenLM eval

1016
1017
1018
1019

Compute (6ND) [FLOPs]

3

4

5
Grid search models

1000

2000

3000

4000

5000

6000

7000

Number of optimization steps

Figure 6: Understanding over-performing models in our grid search. (left) Models trained with
5.2 × 1016 to 5.2 × 1017 FLOPs over-perform relative to their neighbors. In looking at the number
of optimization steps, we notice that the over-performing models experience more optimization steps
than their x-axis neighbors. We hypothesize that the number of optimization steps is important,
especially for smaller models, when trying to find models that lie along a trend. (right) A view of the
same phenomenon, specifically on the efficient frontier.

to train. To verify, we rank 9 testbed models with N ≥1.4B by ground-truth top-1 error and by
959

estimated top-1 error. We find high rank correlation of 0.88 for the 17-task split.
960

Over-performing grid search models experience more optimization steps.
As mentioned in
961

Section 3.3 and Figure 4, we notice that models between 0.011B to 0.079B (i.e., 5.2 × 1016 to
962

5.2 × 1017 FLOPs trained near compute-optimal) over-perform compared to the trend established by
963

other models in our initial grid searches. This results in a bump in the scaling plot. While we choose
964

to exclude this range of models for our scaling study, we additionally investigate this phenomenon.
965

In Figure 6 we color grid search configurations by the number of optimization steps (i.e., number
966

of tokens seen divided by batch size divided by sequence length). We notice that models in the
967

aforementioned range experience more optimization steps than their x-axis neighbors. For context,
968

Figure 1 (left) in Kaplan et al. [51] also shows a bump; however, there the performance is worse than
969

the general trend instead of better as in our work. We leave understanding more fully the interactions
970

between hyperparameters, scaling, and performance to future work.
971

Scaling is largely predictable in-distribution (ID).
Prior work focuses on understanding scaling
972

using ID loss, often using training loss directly [51, 45]. Hence, we also consider Paloma [65] loss
973

evaluation sets, which are designed to probe performance in specific domains. We use Paloma’s
974

C4 [88, 27], RedPajama [112], and Falcon-RefinedWeb [82] splits to probe for ID loss. As seen
975

in Figure 7, relative error is mostly low. Relative error is largest for the N = 1.4B, M = 640
976

RedPajama run at 15.4%. Examining this case specifically, we find that the model performs better
977

than the scaling law prediction. We hypothesize that as a model sees more tokens there is an increased
978

likelihood of near-duplicate sequences ID, resulting in performance that is better than predicted.
979

Relative error is stable across many choices of downstream evaluation suites.
To understand
980

how sensitive our investigation is to our choices of downstream evaluation sets, we consider several
981

other options as seen in Figure 8. We find that our prediction errors are fairly (i) low and (ii) consistent
982

for many choices of downstream evaluation sets including the whole suite of 46 evaluations.
983

Scaling can break down when under-training.
We find that when a token multiple is too small
984

(i.e., under-training regime), scaling appears unreliable. In Figure 9 we see for M = 5 the scaling
985

trend is different. We hypothesize that tuning hyperparameters (e.g., warmup, batch size) directly for
986

smaller multipliers may help mitigate the breakdown in predictability.
987

24


---Page Break---
10

20

40

80

160

320

640

M

0.011B

0.079B

0.154B

0.411B

1.4B

6.9B

N

1.1% 0.0% 0.1% 0.7% 0.9% 0.0% 0.6%

2.6% 0.3% 0.2% 0.8% 0.2% 0.3% 1.4%

2.0% 0.5% 1.3% 1.0% 3.9% 3.5% 1.3%

0.2% 0.2% 0.2% 3.5% 0.5% 2.7%

0.4%
2.3%

3.6%

Train: C4
Eval: C4 (Paloma split)

10

20

40

80

160

320

640

M

4.6% 0.0% 0.3% 2.8% 1.5% 0.0% 0.8%

0.5% 0.0% 1.1% 2.2% 3.0% 3.5% 3.3%

0.2% 0.0% 0.5% 1.3% 1.4% 1.0% 0.5%

0.1% 0.0% 0.2% 0.3% 0.7%10.0%10.3%

3.0%
15.4%

10.3%

Train: RedPajama
Eval: RedPajama (Paloma split)

10

20

40

80

160

320

640

M

1.1% 0.0% 0.9% 1.6% 0.8% 0.0% 1.1%

2.3% 0.1% 0.4% 1.0% 2.0% 3.2% 2.3%

1.1% 0.2% 0.1% 1.5% 0.8% 2.0% 2.4%

0.7% 0.1% 1.6% 2.3% 2.0% 2.3% 4.3%

1.4%
6.0%

5.6%

Train: RefinedWeb
Eval: RefinedWeb (Paloma split)

0.0%

2.0%

4.0%

6.0%

8.0%

10.0%

Relative error

Figure 7: In-distribution (ID) settings. Boxes highlighted in yellow correspond to data points used
to fit Equation (4). Relative error is generally low across interpolation and extrapolation regimes.
Relative error is largest for the RedPajama N = 1.4B, M = 640 prediction at 15.4%. In this case,
we find that our scaling law predicts the model should perform worse than it does in practice.

0
10
20
30
40
50
Inclusion threshold t (i.e., include evals where any model gets

t percentage points above random chance at 0.154B scales)

10
2

10
1

Relative prediction error

0
10
20
30
40
Number of excluded datasets (out of 46-total)

10
2

10
1

C4
RedPajama
RefinedWeb

Figure 8: Downstream evaluation set ablation for 6.9B parameter, 138B token runs. Recall that
we consider a 17 task evaluation suite created by including only test sets where any 0.154B model we
trained (for any token multiplier and training dataset) gets t = 10 percentage points above random
chance. We evaluate over this subset to make sure we are measuring signal not noise. Here, we wish
to understand how sensitive the relative prediction error is to our choice of t. (left) We see that relative
prediction error is fairly low before a threshold of t = 35 (less than 10% relative error). When too
many tasks are excluded (i.e., t ≥40) relative error spikes. Averaging over all 46 datasets (t = −5 as
some evals are worse than random chance) also makes for a predictable metric (less than 3% relative
error). (right) A parallel view, showing how many tasks are removed as t increases. 40 out of the 46
tasks can be removed and relative error is still fairly stable.

1016
1018
1020

Compute (6ND, D = MN) [FLOPs]

2

3

4

5

6

Reducible loss: C4 eval

Training set: C4

1016
1018
1020

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

6
Training set: RedPajama

1016
1018
1020

Compute (6ND, D = MN) [FLOPs]

1

2

3

4

5

Training set: RefinedWeb

N = 0.011B
N = 0.079B
N = 0.154B
N = 0.411B
Interpolation
Extrapolation

5

10

20

40

80

token multiplier M

Figure 9: Scaling with small token multipliers. For smaller multipliers (e.g., M = 5 in cyan),
scaling does not follow the same trend as that of larger multipliers. Additionally, many token
multipliers (e.g., M ∈{10, 20, 40, 80}) garner points close to the compute-optimal frontier.

25


---Page Break---
10

20

40

80

160

320

640

M

0.011B

0.079B

0.154B

0.411B

1.4B

6.9B

N

15.2%0.0% 8.4% 2.9% 0.2% 0.0% 0.2%

2.1% 0.3% 1.7% 0.2% 1.8% 3.0% 9.7%

3.3% 0.6% 0.7% 1.6% 6.4% 4.3% 2.8%

0.5% 0.3% 0.4% 9.5% 4.1%22.4%

2.5%
4.3%

3.3%

Train: C4
Eval: 100 programming languages

(Paloma split)

10

20

40

80

160

320

640

M

5.7% 0.0% 3.6% 1.1% 1.4% 0.0% 3.5%

0.0% 0.4% 1.6% 2.1% 3.0% 5.1%24.3%

0.9% 0.7% 0.7% 1.6% 1.3% 4.7% 2.0%

0.7% 0.3% 0.1% 0.5% 0.7% 3.9%

9.9%
10.0%

9.8%

Train: C4
Eval: Penn Tree Bank

(Paloma split)

10

20

40

80

160

320

640

M

4.1% 0.0% 2.1% 0.6% 1.5% 0.0% 1.2%

1.5% 0.1% 0.5% 0.2% 0.2% 1.1% 1.7%

2.3% 0.1% 0.6% 0.0% 2.2% 2.2% 0.7%

0.5% 0.0% 0.0% 2.8% 0.2% 3.1%

0.9%
7.6%

3.4%

Train: C4
Eval: C4 German eval

0.0%

2.0%

4.0%

6.0%

8.0%

10.0%

Relative error

Figure 10: Out-of-distribution (OOD) settings. Boxes highlighted in yellow correspond to data
points used to fit Equation (4). Recall that the C4 training set is English-filtered. Relative error can
spike, suggesting unreliable scaling, for (left) programming languages and (center) Penn Tree Bank,
which contains many frequently occurring, uncommon substrings. However, scaling is relatively
reliable when evaluating on (right) German. These results motivate future studies of OOD conditions
that affect scaling in the over-trained regime.

10

20

40

80

160

320

640

M

0.011B

0.079B

0.154B

0.411B

1.4B

6.9B

N

0.3% 0.2% 0.6% 0.9% 0.2% 0.3% 1.3%

1.2% 0.4% 1.0% 0.1% 0.3% 0.3% 0.0%

0.2% 0.7% 0.5% 1.2% 1.7% 1.0% 0.4%

0.6% 0.2% 0.0% 1.6% 0.0% 0.4%

0.3%
3.1%

0.1%

Train: C4,
Downstream: 46-task split

10

20

40

80

160

320

640

M

0.1% 0.4% 1.1% 1.3% 0.2% 0.6% 0.4%

0.1% 0.0% 0.3% 0.8% 1.0% 0.6% 1.3%

0.2% 0.2% 0.0% 0.6% 0.9% 0.9% 0.3%

1.3% 0.8% 1.3% 1.5% 1.0% 1.0% 1.0%

0.3%
3.4%

2.1%

Train: RedPajama,
Downstream: 46-task split

10

20

40

80

160

320

640

M

1.2% 0.1% 0.1% 0.7% 0.3% 0.8% 0.6%

0.5% 1.4% 0.4% 0.9% 0.8% 1.0% 1.2%

0.8% 0.1% 0.6% 0.3% 1.1% 0.7% 0.9%

0.5% 1.1% 1.1% 1.7% 1.6% 0.6% 0.9%

0.3%
4.3%

2.8%

Train: RefinedWeb,
Downstream: 46-task split

0%

2%

4%

6%

8%

10%

Relative error

Figure 11: Relative error on average top-1 predictions (46 task split). Boxes highlighted in yellow
correspond to data points used to fit Equation (5). Using our fits, we accurately predict downstream
average top-1 error across interpolation and extrapolation regimes. This result supports that (i)
chaining a scaling law and our proposed exponential decay function is a valid procedure and (ii)
average top-1 error can be highly predictable.

Scaling can be unpredictable out-of-distribution (OOD).
Our main result shows reliable C4 eval
988

loss predictions with models trained on RedPajama, which is an OOD evaluation setting. However,
989

both C4 and RedPajama both contain tokens sourced from CommonCrawl.
990

To further probe OOD performance, we measure the relative error of scaling laws fit to models trained
991

on C4 and evaluated on Paloma’s 100 programming languages [65], Paloma’s Penn Tree Bank (PTB)
992

split [66], and a German version of C4 [27]. Recall that the C4 training set we use has been filtered
993

for English text. Hence we expect (i) the proportion of code is minimal, (ii) the “<unk>” substrings in
994

PTB raw text do not appear frequently, and (iii) German is not prevalent. We notice that extrapolation
995

relative error tends to be high for large M, N on programming languages and PTB (Figure 10 (left,
996

center)). In contrast, for German C4, relative error is still low across the extrapolation range, with a
997

maximum relative error of 7.6% at the N =1.4B, M = 80 scale (Figure 10 (right)). We hypothesize
998

that further modifications to scaling laws are necessary to predict when scaling should be reliable as a
999

function of the training and evaluation distributions.
1000

Small-scale experiments can predict average downstream top-1 error.
To verify that chaining
1001

Equations (4) and (5) is effective in practice, we collect C4 eval loss and downstream error pairs for
1002

the configurations in Table 1. In Figure 11, we look at relative error for our scaling predictions in the
1003

context of Average top-1 error over 46 evals and in Figure 12 over the high-signal 17 eval subset. We
1004

again notice reliable scaling in interpolation and extrapolation regimes, suggesting the validity of our
1005

procedure to predict downstream average top-1 error.
1006

26


---Page Break---
10

20

40

80

160

320

640

M

0.011B

0.079B

0.154B

0.411B

1.4B

6.9B

N

1.2% 0.7% 1.2% 2.3% 1.1% 1.1% 3.1%

3.9% 1.2% 2.5% 1.0% 0.5% 0.6% 0.4%

1.3% 2.9% 1.1% 2.2% 3.9% 1.4% 0.2%

2.3% 1.3% 1.7% 2.8% 2.2% 1.8%

0.7%
9.6%

0.1%

Train: C4,
Downstream: 17-task split

10

20

40

80

160

320

640

M

0.6% 0.4% 2.1% 2.5% 0.6% 0.8% 0.8%

0.8% 0.7% 0.2% 1.3% 0.8% 0.5% 1.3%

0.9% 1.7% 0.2% 0.1% 0.2% 0.7% 0.0%

2.6% 2.0% 1.0% 2.6% 1.9% 3.4% 3.4%

0.7%
3.6%

0.0%

Train: RedPajama,
Downstream: 17-task split

10

20

40

80

160

320

640

M

2.5% 0.4% 0.7% 1.5% 1.3% 1.7% 0.9%

1.0% 2.7% 0.4% 0.4% 0.9% 0.3% 2.3%

1.5% 0.1% 0.0% 1.3% 2.1% 0.6% 0.6%

0.8% 2.2% 2.0% 3.6% 3.4% 1.2% 1.6%

0.4%
5.6%

2.9%

Train: RefinedWeb,
Downstream: 17-task split

0%

2%

4%

6%

8%

10%

Relative error

Figure 12: Relative error on average top-1 predictions (17 task split). Boxes highlighted in yellow
correspond to data points used to fit Equation (5). Using our fits, we accurately predict downstream
average top-1 error across interpolation and extrapolation regimes. This result supports that (i)
chaining a scaling law and our proposed exponential decay function is a valid procedure and (ii)
average top-1 error can be highly predictable.

2.5
3.0
3.5
4.0
4.5
5.0
5.5
6.0

Loss: C4

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

Average top-1 error: 17-task split

2
3
4
5
6
7
8

Loss: RedPajama

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

0.85

3
4
5
6

Loss: RefinedWeb

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

0.85

C4
RedPajama
RefinedWeb
Interpolation
Extrapolation

2.5
3.0
3.5
4.0
4.5
5.0
5.5
6.0

Loss: C4

0.66

0.68

0.70

0.72

0.74

0.76

Average top-1 error: 46-task split

2
3
4
5
6
7
8

Loss: RedPajama

0.66

0.68

0.70

0.72

0.74

0.76

3
4
5
6

Loss: RefinedWeb

0.66

0.68

0.70

0.72

0.74

0.76

C4
RedPajama
RefinedWeb
Interpolation
Extrapolation

Figure 13: Correlation between average top-1 error and evaluation loss. We observe that
regardless of evaluation loss distribution (x-axis), models tend to follow Equation (5). This suggests
that there can be several reasonable choices for the validation loss distribution. Additionally, ID
models trained on C4 and evaluated on a C4 validation set, perform best in terms of loss, but these
gains don’t necessarily translate to lower error downstream (e.g., (left column)). This suggests the
need to fit Equation (5) per dataset and also suggests comparing models trained on different data
distributions with a single loss evaluation can be misleading.

Loss evaluation ablations for downstream trends.
Figure 13 presents the correlation between
1007

downstream error and loss evaluated on different validation sets (C4, RedPajama, and RefinedWeb).
1008

Regardless of the validation set (x-axis), models follow the exponential decay relationship given
1009

in Equation (5), suggesting the choice of validation loss is not critical for the appearance of this
1010

phenomenon.
1011

Investing more compute in a scaling law makes it more predictive.
Thus far we have looked
1012

at standard configurations from Table 1 to construct our scaling laws, mainly to demonstrate
1013

extrapolation to larger N, M. However, for practitioners, the main constraint is often training
1014

27


---Page Break---
1018
1019
1020
1021

Compute [FLOPs] used for the scaling fit

10
4

10
3

10
2

10
1

100

Relative error: C4 eval

5
10
15
20
25
30
Number of samples used for the scaling fit

Trend
Individual estimates
Default setting from Table 2

Figure 14: Trade-offs between scaling law for loss fitting considerations and reliability.
Each red circle represents a scaling law fit to Equation (4) with as many as 29 models trained
on RedPajama. Specifically, a grid formed by N ∈{0.011B, 0.079B, 0.154B, 0.411B}, M ∈
{5, 10, 20, 40, 80, 160, 320} gives 28 models and a N = 1.4B, M = 20 run gives the last model. We
sort models by training FLOPs in increasing order and sample models uniformly from index windows
[1, 2, ..., n] for n ∈[5, 6, .., 29] to fit Equation (4). The blue star represents the default configuration
presented in Table 1. The prediction target is a N = 1.4B, M = 640 (D = 900B) model. As the
amount of compute (left) and the number of points (right) used to fit the scaling law increases, relative
error trends downwards. Our default configuration keeps compute and number of points low, while
still providing low prediction error compared to the trend.

1018
1019
1020
1021

Compute [FLOPs] used for the scaling fit

10
4

10
3

10
2

10
1

100

Relative error: C4 eval

1018
1019
1020
1021

Compute [FLOPs] used for the scaling fit

Relative error: 17-task split

Trend
Default setting from Table 2
Individual estimates

Figure 15: Compute vs. relative error for the 1.4B, 900B token RedPajama run. (left) The
compute necessary to accurately predict loss is less than that needed to accurately predict (right)
average downstream error. This claim is supported by the fact that the slope of the trend for loss is
steeper than for top-1 error. These findings corroborate Figure 16.

compute. Hence, we wish to understand the trade-offs between the amount of compute invested
1015

in creating a scaling law and the relative error of the resulting law in the over-trained regime. In
1016

Figure 14 (left), we see that as one increases the amount of compute, it is possible to get better fits
1017

with lower relative error. In Figure 14 (right), we see a similar trend as one increases the number of
1018

data points used to fit a scaling law. Blue stars indicate the configurations from Table 1, which provide
1019

accurate predictions relative to the general trends—hinting at their usefulness for our investigation.
1020

In Figures 15 and 16 we repeat the compute analysis comparing trade-offs for loss prediction and
1021

error prediction for our RedPajama 1.4B parameter, 900B token and 6.9B parameter, 138B token
1022

runs respectively. We find that less compute is generally necessary to construct a loss scaling law that
1023

achieves the same relative error as that of an error prediction scaling law.
1024

28


---Page Break---
1018
1019
1020
1021

Compute [FLOPs] used for the scaling fit

10
4

10
3

10
2

10
1

100

Relative error: C4 eval

1018
1019
1020
1021

Compute [FLOPs] used for the scaling fit

Relative error: 17-task split

Trend
Default setting from Table 2
Individual estimates

Figure 16: Compute vs. relative error for the 6.9B, 138B token RedPajama run. (left) The
compute necessary to accurately predict loss is less than that needed to accurately predict (right)
average downstream error. This claim is supported by the fact that the slope of the trend for loss is
steeper than for top-1 error. These findings corroborate Figure 15.

20
40
80
160
320
640
Token multiplier M

0.11

0.12

0.13

0.14

0.15

Scaling exponent 

C4
RedPajama
RefinedWeb
Trend

Figure 17: Scaling exponent vs. token multiplier. In Figure 2, we notice roughly parallel lines
(i.e., roughly constant scaling exponent η) in the log-log plot of loss vs. compute, even as the token
multiplier M changes. Here we plot η vs. M directly, where the shaded region gives a 95% bootstrap
confidence interval for the trend. This view supports that η is relatively constant.

On compute-optimal token multipliers.
We consider 20 tokens per parameter as close to compute-
1025

optimal for our experiments. Here we investigate, using different approaches, what the compute-
1026

optimal token multipliers are for each dataset—assuming one should scale number of parameter and
1027

training tokens equally as Hoffmann et al. [45] suggest.
1028

Turning to Figure 9, we notice that there are many multipliers, between 10 and 80 that yield models
1029

close to the frontier. Hence, empirically, it appears choices within this range should be suitable for
1030

the optimal token multiplier.
1031

We can also compute an optimal token multiplier using the coefficients in Table 6. Based on Hoffmann
1032

et al. [45]’s Equation (4) and the assumption that α = β, we write,
1033

N ∗(C) = G
C

6

 1

2
, D∗(C) = G−1
C

6

 1

2
, G =
a

b

 1

4η .
(9)

To compute M ∗= D∗/N ∗, we then have,
1034

M ∗=
 b

a

 1

2η
.
(10)

Using the values from Table 6 and plugging into Equation (10), we find M ∗
C4 = 2.87, M ∗
RedPajama =
1035

4.30, M ∗
RefinedWeb = 3.79, where the subscript gives the dataset name. These values conflict with the
1036

observation in Figure 9, which suggests M = 5 is already too small to give points on the Pareto
1037

frontier. We hypothesize this mismatch arises because we fit our scaling laws using models with
1038

M ≥20.
1039

29


---Page Break---
2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.72

0.74

0.76

0.78

0.80

0.82

0.84

Top-1 error: AGIEval LSAT AR

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.71

0.72

0.73

0.74

0.75

0.76

0.77

0.78

Top-1 error: AGIEval LSAT LR

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.72

0.74

0.76

0.78

Top-1 error: AGIEval LSAT RC

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.70

0.72

0.74

0.76

0.78

0.80

Top-1 error: AGIEval SAT English

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.625

0.650

0.675

0.700

0.725

0.750

0.775

0.800

Top-1 error: ARC-Challenge

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.3

0.4

0.5

0.6

0.7

Top-1 error: ARC-Easy

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.46

0.48

0.50

0.52

0.54

0.56

0.58

0.60

Top-1 error: BBQ

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.675

0.700

0.725

0.750

0.775

0.800

0.825

Top-1 error: BIG-bench: Conceptual combinations

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.96

0.97

0.98

0.99

1.00

Top-1 error: BIG-bench: Conlang translation

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.6

0.7

0.8

0.9

1.0

Top-1 error: BIG-bench: CS algorithms

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Top-1 error: BIG-bench: Dyck languages

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.73

0.74

0.75

0.76

0.77

Top-1 error: BIG-bench: Elementary math QA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.740

0.745

0.750

0.755

0.760

Top-1 error: BIG-bench: Language identification

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.72

0.73

0.74

0.75

0.76

0.77

Top-1 error: BIG-bench: Logical deduction

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.44

0.46

0.48

0.50

0.52

0.54

0.56

Top-1 error: BIG-bench: Misconceptions

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

0.85

Top-1 error: BIG-bench: Novel Concepts

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.80

0.85

0.90

0.95

1.00

Top-1 error: BIG-bench: Operators

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

Top-1 error: BIG-bench: QA WikiData

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.84

0.86

0.88

0.90

0.92

0.94

0.96

0.98

1.00

Top-1 error: BIG-bench: Repeat copy logic

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.425

0.450

0.475

0.500

0.525

0.550

0.575

0.600

Top-1 error: BIG-bench: Strange stories

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.44

0.46

0.48

0.50

0.52

0.54

Top-1 error: BIG-bench: Strategy QA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.68

0.70

0.72

0.74

0.76

0.78

Top-1 error: BIG-bench: Understanding fables

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.35

0.40

0.45

0.50

0.55

0.60

Top-1 error: BoolQ

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.68

0.70

0.72

0.74

0.76

0.78

0.80

0.82

Top-1 error: Commonsense QA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.2

0.3

0.4

0.5

0.6

Top-1 error: COPA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Top-1 error: CoQA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.44

0.46

0.48

0.50

0.52

0.54

0.56

0.58

Top-1 error: Enterprise PII classification

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.3

0.4

0.5

0.6

0.7

Top-1 error: HellaSwag (10-shot)

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.3

0.4

0.5

0.6

0.7

Top-1 error: HellaSwag (zero-shot)

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.6

0.7

0.8

0.9

1.0

Top-1 error: Jeopardy

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.4

0.5

0.6

0.7

0.8

0.9

1.0

Top-1 error: LAMBADA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.70

0.72

0.74

0.76

0.78

0.80

Top-1 error: LogiQA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.740

0.745

0.750

0.755

0.760

0.765

Top-1 error: MathQA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.730

0.735

0.740

0.745

0.750

0.755

0.760

0.765

0.770

Top-1 error: MMLU (5-shot)

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.73

0.74

0.75

0.76

0.77

Top-1 error: MMLU (zero-shot)

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.600

0.625

0.650

0.675

0.700

0.725

0.750

0.775

Top-1 error: OpenBook QA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.25

0.30

0.35

0.40

0.45

0.50

Top-1 error: PIQA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.5

0.6

0.7

0.8

0.9

1.0

Top-1 error: PubMed QA Labeled

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.975

0.980

0.985

0.990

0.995

1.000

Top-1 error: Simple Arithmetic: NoSpaces

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.96

0.97

0.98

0.99

1.00

Top-1 error: Simple Arithmetic: WithSpaces

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.48

0.49

0.50

0.51

0.52

Top-1 error: SIQA

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.5

0.6

0.7

0.8

0.9

1.0

Top-1 error: SQuAD

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.45

0.50

0.55

0.60

Top-1 error: WinoGender MC: Female

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.35

0.40

0.45

0.50

0.55

0.60

Top-1 error: WinoGender MC: Male

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.15

0.20

0.25

0.30

0.35

0.40

0.45

0.50

Top-1 error: WinoGrand

2.5
3.0
3.5
4.0
4.5
5.0
5.5

Loss: C4 eval

0.36

0.38

0.40

0.42

0.44

0.46

0.48

0.50

0.52

Top-1 error: WinoGrande

C4
RedPajama
RefinedWeb
Random chance

Figure 18: Downstream top-1 error vs. C4 eval loss for each of the 46 downstream evals. Here
we plot models from our testbed for each scatter plot. We see that some individual evaluations, like
ARC-Easy, follow exponential decay. Others, like BIG-bench: CS algorithms, show step function
behavior. Still others, like MathQA, hover around random chance.

F
Additional related work
1040

Language modeling.
Language models can be grouped into encoder-only [26, 53, 59, 96, 22],
1041

encoder-decoder [56, 89], and decoder-only architectures [85, 113, 114, 110, 49, 38, 74, 7, 111,
1042

28, 64, 99, 122, 4, 57, 63, 34]. Most current implementations are based on the transformer [116].
1043

However, there has been a recent resurgence in scaling language models based on non-transformer
1044

architectures [83, 36, 37, 35]. Further, there has been substantial work on adapting pre-trained
1045

language models to better follow instructions [119, 20, 70, 61, 71, 133, 87, 29, 115, 103, 73].
1046

However, following prior work [45, 72] and given their overall prevalence, we limit ourselves to
1047

GPT-style, decoder-only transformers that have solely been pre-trained.
1048

30


---Page Break---
Scaling laws.
Kaplan et al. [51] investigate scaling trends in GPT language models. Bahri et al.
1049

[9] investigate different scaling regimes theoretically, and Sharma & Kaplan [101] relate scaling
1050

coefficients to data manifold dimensions. Tay et al. [108, 109] elucidate the connection between
1051

model architecture and scaling trends, while Hernandez et al. [42], Tay et al. [108] develop scaling
1052

laws for transfer learning. Ivgi et al. [48] also consider transfer learning scaling laws and highlight
1053

the importance of hyperparameter selection in the low-compute regime. Ghorbani et al. [32], Gordon
1054

et al. [33], Bansal et al. [10] develop scaling laws for neural machine translation. Caballero et al. [17]
1055

propose a scaling law functional form, which they demonstrate is predictive in several domains.
1056

Scaling beyond language modeling.
There is a large body of work on scaling neural networks
1057

beyond language modeling, for example in computer vision [60, 127, 105, 1, 2], multimodal
1058

learning [41, 18, 30], and image reconstruction [52].
1059

Over-training in existing models.
To contextualize the extent to which we over-train, we provide
1060

token multipliers for popular models in Table 8.
1061

G
Broader impact
1062

Language models have known risks in terms harmful language, toxicity, and human automation—to
1063

name a few [121, 12]. We will include the following for our public release “WARNING: These are
1064

base models and not aligned with post-training. They are provided as is and intended as research
1065

artifacts only.” However, even as research artifacts, we recognize that models can still be misused
1066

by malicious actors or can be harmful to benevolent actors. When deciding to release our models
1067

and experiments, we considered (i) the benefit to the scientific community and (ii) the benchmark
1068

performance relative to other models that have already been released. For (i) we feel that our testbed
1069

is of use to others in the community who want to do scaling research, but do not necessarily have the
1070

means to train these model artifacts themselves. Hence, we predict (and hope) releasing all models
1071

and experiments will be helpful to others wanting to participate in scaling research. For (ii), we note
1072

that there are publicly available models [113, 114, 49], which outperform models from our testbed
1073

and that are more likely to be widely adopted. Finally, we recognize that advancing scaling science
1074

also has potential for harm. Specifically, while we are concerned with loss and downstream task
1075

performance for popular evaluation settings, it is possible that nefarious actors may use scaling laws
1076

to help design more harmful models.
1077

H
Licensing
1078

In terms of licensing, we will release our code, models, and experiments under an MIT licence, which
1079

is also attached to our supplementary submission.
1080

31


---Page Break---
Table 4: Topologies for our grid searches. We consider 130 architectures for our grid search. After
sweeping over batch size and warmup, we get a total of 435 configurations.

nlayers
nheads
dmodel
Number of
parameters [B]

4
4
96
0.010
4
12
96
0.010
12
12
96
0.011
12
4
96
0.011
8
4
96
0.011
16
4
96
0.011
16
12
96
0.011
8
12
96
0.011
24
4
96
0.012
24
12
96
0.012
4
4
192
0.021
4
8
192
0.021
4
12
192
0.021
8
8
192
0.023
8
4
192
0.023
8
12
192
0.023
12
4
192
0.025
12
8
192
0.025
12
12
192
0.025
16
4
192
0.026
16
8
192
0.026
16
12
192
0.026
24
8
192
0.030
24
4
192
0.030
24
12
192
0.030
4
12
288
0.033
4
4
288
0.033
8
12
288
0.037
8
4
288
0.037
4
4
320
0.038
4
8
320
0.038
12
12
288
0.041
12
4
288
0.041
8
8
320
0.043
8
4
320
0.043
16
4
288
0.045
16
12
288
0.045
12
4
320
0.049
12
8
320
0.049
24
4
288
0.053
24
12
288
0.053
16
8
320
0.055
16
4
320
0.055
4
12
488
0.062
4
4
512
0.065
4
16
512
0.065
4
8
512
0.065
24
8
320
0.066
24
4
320
0.066
4
4
576
0.074
4
8
576
0.074
4
12
576
0.074
8
12
488
0.075
8
4
512
0.079
8
8
512
0.079
8
16
512
0.079
4
4
640
0.085
4
16
640
0.085
4
8
640
0.085
12
12
488
0.087
8
4
576
0.090
8
12
576
0.090
8
8
576
0.090
12
16
512
0.093
12
8
512
0.093

nlayers
nheads
dmodel
Number of
parameters [B]

12
4
512
0.093
16
12
488
0.100
8
16
640
0.105
8
4
640
0.105
8
8
640
0.105
12
8
576
0.106
16
16
512
0.106
4
4
768
0.106
12
12
576
0.106
16
8
512
0.106
4
8
768
0.106
12
4
576
0.106
4
16
768
0.106
16
4
512
0.106
4
12
768
0.106
16
12
576
0.122
16
4
576
0.122
16
8
576
0.122
12
4
640
0.126
24
12
488
0.126
12
16
640
0.126
12
8
640
0.126
24
8
512
0.133
24
4
512
0.133
24
16
512
0.133
8
8
768
0.134
8
16
768
0.134
8
4
768
0.134
8
12
768
0.134
16
16
640
0.146
16
8
640
0.146
16
4
640
0.146
24
8
576
0.154
24
4
576
0.154
24
12
576
0.154
4
8
1024
0.155
4
16
1024
0.155
4
4
1024
0.155
12
8
768
0.162
12
4
768
0.162
12
12
768
0.162
12
16
768
0.162
24
16
640
0.186
24
8
640
0.186
24
4
640
0.186
16
16
768
0.191
16
4
768
0.191
16
8
768
0.191
16
12
768
0.191
8
8
1024
0.206
8
4
1024
0.206
8
16
1024
0.206
24
8
768
0.247
24
12
768
0.247
24
4
768
0.247
24
16
768
0.247
12
8
1024
0.257
12
4
1024
0.257
12
16
1024
0.257
16
8
1024
0.309
16
4
1024
0.309
16
16
1024
0.309
24
16
1024
0.412
24
8
1024
0.412
24
4
1024
0.412

32


---Page Break---
Table 5: 46 downstream tasks. All downstream tasks considered in this work, evaluated via LLM-
foundry [69]. For more information on each dataset and specifics about the LLM-foundry category
and evaluation type, please see: https://www.mosaicml.com/llm-evaluation.

Downstream task
LLM-foundry category
Evaluation type
Shots
Samples
Baseline

AGIEval LSAT AR [132, 131, 118]
symbolic problem solving
multiple choice
3
230
0.25
AGIEval LSAT LR [132, 131, 118]
reading comprehension
multiple choice
3
510
0.25
AGIEval LSAT RC [132, 131, 118]
reading comprehension
multiple choice
3
268
0.25
AGIEval SAT English [132]
reading comprehension
multiple choice
3
206
0.25
ARC-Challenge [23]
world knowledge
multiple choice
10
2376
0.25
ARC-Easy [23]
world knowledge
multiple choice
10
2376
0.25
BBQ [79]
safety
multiple choice
3
58492
0.50
BIG-bench: CS algorithms [11]
symbolic problem solving
language modeling
10
1320
0.00
BIG-bench: Conceptual combinations [11]
language understanding
multiple choice
10
103
0.25
BIG-bench: Conlang translation [11]
language understanding
language modeling
0
164
0.00
BIG-bench: Dyck languages [11]
symbolic problem solving
language modeling
10
1000
0.00
BIG-bench: Elementary math QA [11]
symbolic problem solving
multiple choice
10
38160
0.25
BIG-bench: Language identification [11]
language understanding
multiple choice
10
10000
0.25
BIG-bench: Logical deduction [11]
symbolic problem solving
multiple choice
10
1500
0.25
BIG-bench: Misconceptions [11]
world knowledge
multiple choice
10
219
0.50
BIG-bench: Novel Concepts [11]
commonsense reasoning
multiple choice
10
32
0.25
BIG-bench: Operators [11]
symbolic problem solving
language modeling
10
210
0.00
BIG-bench: QA WikiData [11]
world knowledge
language modeling
10
20321
0.00
BIG-bench: Repeat copy logic [11]
symbolic problem solving
language modeling
10
32
0.00
BIG-bench: Strange stories [11]
commonsense reasoning
multiple choice
10
174
0.50
BIG-bench: Strategy QA [11]
commonsense reasoning
multiple choice
10
2289
0.50
BIG-bench: Understanding fables [11]
reading comprehension
multiple choice
10
189
0.25
BoolQ [21]
reading comprehension
multiple choice
10
3270
0.50
COPA [92]
commonsense reasoning
multiple choice
0
100
0.50
CoQA [91]
reading comprehension
language modeling
0
7983
0.00
Commonsense QA [107]
commonsense reasoning
multiple choice
10
1221
0.25
Enterprise PII classification [81]
safety
multiple choice
10
3395
0.50
HellaSwag (10-shot) [126]
language understanding
multiple choice
10
10042
0.25
HellaSwag (zero-shot) [126]
language understanding
multiple choice
0
10042
0.25
Jeopardy [69]
world knowledge
language modeling
10
2117
0.00
LAMBADA [77]
language understanding
language modeling
0
5153
0.00
LogiQA [58]
symbolic problem solving
multiple choice
10
651
0.25
MMLU (5-shot) [40]
world knowledge
multiple choice
5
14042
0.25
MMLU (zero-shot) [40]
world knowledge
multiple choice
0
14042
0.25
MathQA [5]
symbolic problem solving
multiple choice
10
2983
0.25
OpenBook QA [68]
commonsense reasoning
multiple choice
0
500
0.25
PIQA [14]
commonsense reasoning
multiple choice
10
1838
0.50
PubMed QA Labeled [50]
reading comprehension
language modeling
10
1000
0.00
SIQA [97]
commonsense reasoning
multiple choice
10
1954
0.50
SQuAD [90]
reading comprehension
language modeling
10
10570
0.00
Simple Arithmetic: NoSpaces [69]
symbolic problem solving
language modeling
10
1000
0.00
Simple Arithmetic: WithSpaces [69]
symbolic problem solving
language modeling
10
1000
0.00
WinoGender MC: Female [94]
safety
multiple choice
10
60
0.50
WinoGender MC: Male [94]
safety
multiple choice
10
60
0.50
WinoGrande [95]
language understanding
schema
0
1267
0.50
WinoGrand [55]
language understanding
schema
0
273
0.50

Table 6: Scaling law fit parameters. Here we present our scaling coefficients fit to Equations (4)
and (5) using configurations from Table 1.

Training dataset
Fit for Equation (4): L(C, M) =
Fit for Equation (5): Err(L) =
E + (a · M η + b · M −η)Cη
ϵ −k · exp (−γL)

C4 [88, 27]
1.51 +
 
114 · M 0.242 + 190 · M −0.242
C−0.242
0.850 −2.08 · exp (−0.756 · L)
RedPajama [112]
1.84 +
 
166 · M 0.272 + 367 · M −0.272
C−0.272
0.857 −2.21 · exp (−0.715 · L)
RefinedWeb [82]
1.73 +
 
125 · M 0.254 + 246 · M −0.254
C−0.254
0.865 −2.21 · exp (−0.707 · L)

33


---Page Break---
Table 7: Downstream relative prediction error at 6.9B, 138B tokens, with and without the 1.4B
data point. Recall in Table 1, we introduce a N = 1.4B, M = 20 run to get better downstream error
predictions. Here we compare, prediction errors with and without this model for fitting the scaling
law. Note that without the model (i.e., rows with “w/o 1.4B”) average top-1 predictions, over the 17
tasks. are less accurate.

Scaling law fit
Train set
ARC-E
LAMBADA
OpenBook QA
HellaSwag
17 eval
[23]
[77]
[68]
[126]

Table 1
C4 [88, 27]
28.96%
15.01%
16.80%
79.58%
0.14%
Table 1 w/o 1.4B
C4 [88, 27]
0.92%
2.04%
96.16%
61.79%
0.42%

Table 1
RedPajama [112]
5.21%
14.39%
8.44%
25.73%
0.05%
Table 1 w/o 1.4B
RedPajama [112]
8.13%
11.07%
7.56%
30.98%
10.64%

Table 1
RefinedWeb [82]
26.06%
16.55%
1.92%
81.96%
2.94%
Table 1 w/o 1.4B
RefinedWeb [82]
15.39%
6.26%
6.79%
6.52%
15.79%

Table 8: Token multipliers of existing models. In our work, we run experiments with token
multipliers between 5 and 640 for {GPT-2 [85], LLaMA [113]}-style decoder-only architectures.

Model family
Parameters N
Training tokens D
Token multiplier M

T5 [89]
11B
34B
3.1
GPT-3 [16]
175B
300B
1.7
Gopher [86]
280B
300B
1.1
Chinchilla [45]
70B
1.4T
20.0
LLaMA [113]
7B
1T
140.0
LLaMA [113]
70B
1.4T
20.0
LLaMA-2 [114]
7B
2T
290.0
LLaMA-2 [114]
70B
2T
30.0
XGen [74]
7B
1.5T
210.0
MPT [110]
7B
1T
140.0

34


---Page Break---
NeurIPS Paper Checklist
1081

1. Claims
1082

Question: Do the main claims made in the abstract and introduction accurately reflect the
1083

paper’s contributions and scope?
1084

Answer: [Yes]
1085

Justification: The experiment section justify the claims made in the abstract and introduction,
1086

namely that the developed scaling laws for over-training and downstream task prediction are
1087

predictive in practice for larger scale runs.
1088

Guidelines:
1089

• The answer NA means that the abstract and introduction do not include the claims
1090

made in the paper.
1091

• The abstract and/or introduction should clearly state the claims made, including the
1092

contributions made in the paper and important assumptions and limitations. A No or
1093

NA answer to this question will not be perceived well by the reviewers.
1094

• The claims made should match theoretical and experimental results, and reflect how
1095

much the results can be expected to generalize to other settings.
1096

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
1097

are not attained by the paper.
1098

2. Limitations
1099

Question: Does the paper discuss the limitations of the work performed by the authors?
1100

Answer: [Yes]
1101

Justification: The final section discusses limitations, which provide motivation for future
1102

work.
1103

Guidelines:
1104

• The answer NA means that the paper has no limitation while the answer No means that
1105

the paper has limitations, but those are not discussed in the paper.
1106

• The authors are encouraged to create a separate "Limitations" section in their paper.
1107

• The paper should point out any strong assumptions and how robust the results are to
1108

violations of these assumptions (e.g., independence assumptions, noiseless settings,
1109

model well-specification, asymptotic approximations only holding locally). The authors
1110

should reflect on how these assumptions might be violated in practice and what the
1111

implications would be.
1112

• The authors should reflect on the scope of the claims made, e.g., if the approach was
1113

only tested on a few datasets or with a few runs. In general, empirical results often
1114

depend on implicit assumptions, which should be articulated.
1115

• The authors should reflect on the factors that influence the performance of the approach.
1116

For example, a facial recognition algorithm may perform poorly when image resolution
1117

is low or images are taken in low lighting. Or a speech-to-text system might not be
1118

used reliably to provide closed captions for online lectures because it fails to handle
1119

technical jargon.
1120

• The authors should discuss the computational efficiency of the proposed algorithms
1121

and how they scale with dataset size.
1122

• If applicable, the authors should discuss possible limitations of their approach to
1123

address problems of privacy and fairness.
1124

• While the authors might fear that complete honesty about limitations might be used
1125

by reviewers as grounds for rejection, a worse outcome might be that reviewers
1126

discover limitations that aren’t acknowledged in the paper. The authors should use
1127

their best judgment and recognize that individual actions in favor of transparency play
1128

an important role in developing norms that preserve the integrity of the community.
1129

Reviewers will be specifically instructed to not penalize honesty concerning limitations.
1130

3. Theory Assumptions and Proofs
1131

Question: For each theoretical result, does the paper provide the full set of assumptions and
1132

a complete (and correct) proof?
1133

35


---Page Break---
Answer: [Yes]
1134

Justification: All assumptions are clearly stated and full proofs/derivations are provided in
1135

the Appendix.
1136

Guidelines:
1137

• The answer NA means that the paper does not include theoretical results.
1138

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
1139

referenced.
1140

• All assumptions should be clearly stated or referenced in the statement of any theorems.
1141

• The proofs can either appear in the main paper or the supplemental material, but if
1142

they appear in the supplemental material, the authors are encouraged to provide a short
1143

proof sketch to provide intuition.
1144

• Inversely, any informal proof provided in the core of the paper should be complemented
1145

by formal proofs provided in appendix or supplemental material.
1146

• Theorems and Lemmas that the proof relies upon should be properly referenced.
1147

4. Experimental Result Reproducibility
1148

Question: Does the paper fully disclose all the information needed to reproduce the
1149

main experimental results of the paper to the extent that it affects the main claims and/or
1150

conclusions of the paper (regardless of whether the code and data are provided or not)?
1151

Answer: [Yes]
1152

Justification: We point to all public datasets and open source training infrastructure. We
1153

additionally specify all hyperparameters used for training.
1154

Guidelines:
1155

• The answer NA means that the paper does not include experiments.
1156

• If the paper includes experiments, a No answer to this question will not be perceived
1157

well by the reviewers: Making the paper reproducible is important, regardless of
1158

whether the code and data are provided or not.
1159

• If the contribution is a dataset and/or model, the authors should describe the steps taken
1160

to make their results reproducible or verifiable.
1161

• Depending on the contribution, reproducibility can be accomplished in various ways.
1162

For example, if the contribution is a novel architecture, describing the architecture fully
1163

might suffice, or if the contribution is a specific model and empirical evaluation, it may
1164

be necessary to either make it possible for others to replicate the model with the same
1165

dataset, or provide access to the model. In general. releasing code and data is often
1166

one good way to accomplish this, but reproducibility can also be provided via detailed
1167

instructions for how to replicate the results, access to a hosted model (e.g., in the case
1168

of a large language model), releasing of a model checkpoint, or other means that are
1169

appropriate to the research performed.
1170

• While NeurIPS does not require releasing code, the conference does require all
1171

submissions to provide some reasonable avenue for reproducibility, which may depend
1172

on the nature of the contribution. For example
1173

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
1174

to reproduce that algorithm.
1175

(b) If the contribution is primarily a new model architecture, the paper should describe
1176

the architecture clearly and fully.
1177

(c) If the contribution is a new model (e.g., a large language model), then there should
1178

either be a way to access this model for reproducing the results or a way to reproduce
1179

the model (e.g., with an open-source dataset or instructions for how to construct
1180

the dataset).
1181

(d) We recognize that reproducibility may be tricky in some cases, in which case
1182

authors are welcome to describe the particular way they provide for reproducibility.
1183

In the case of closed-source models, it may be that access to the model is limited in
1184

some way (e.g., to registered users), but it should be possible for other researchers
1185

to have some path to reproducing or verifying the results.
1186

5. Open access to data and code
1187

36


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient
1188

instructions to faithfully reproduce the main experimental results, as described in
1189

supplemental material?
1190

Answer: [Yes]
1191

Justification: We include code and data needed to reproduce all figures in the paper. Our
1192

datasets are sourced from HuggingFace and our training code utilizes OpenLM, which is
1193

open-source.
1194

Guidelines:
1195

• The answer NA means that paper does not include experiments requiring code.
1196

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
1197

public/guides/CodeSubmissionPolicy) for more details.
1198

• While we encourage the release of code and data, we understand that this might not be
1199

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
1200

including code, unless this is central to the contribution (e.g., for a new open-source
1201

benchmark).
1202

• The instructions should contain the exact command and environment needed to run to
1203

reproduce the results. See the NeurIPS code and data submission guidelines (https:
1204

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
1205

• The authors should provide instructions on data access and preparation, including how
1206

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
1207

• The authors should provide scripts to reproduce all experimental results for the new
1208

proposed method and baselines. If only a subset of experiments are reproducible, they
1209

should state which ones are omitted from the script and why.
1210

• At submission time, to preserve anonymity, the authors should release anonymized
1211

versions (if applicable).
1212

• Providing as much information as possible in supplemental material (appended to the
1213

paper) is recommended, but including URLs to data and code is permitted.
1214

6. Experimental Setting/Details
1215

Question: Does the paper specify all the training and test details (e.g., data splits,
1216

hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand
1217

the results?
1218

Answer: [Yes]
1219

Justification: We explicitly have sections and appendices that detail our experimental setup
1220

(training and evaluation) and title the sections and appendices to indicate this.
1221

Guidelines:
1222

• The answer NA means that the paper does not include experiments.
1223

• The experimental setting should be presented in the core of the paper to a level of detail
1224

that is necessary to appreciate the results and make sense of them.
1225

• The full details can be provided either with the code, in appendix, or as supplemental
1226

material.
1227

7. Experiment Statistical Significance
1228

Question: Does the paper report error bars suitably and correctly defined or other appropriate
1229

information about the statistical significance of the experiments?
1230

Answer: [Yes]
1231

Justification: When appropriate we report bootstrap 95% confidence intervals (e.g., in
1232

Figure 4 and Figure 17). We do not train models with many seeds, which is prohibitively
1233

expensive. Given the large size of the C4 validation set, we observe that bootstrap 95%
1234

confidence intervals for loss (computed over either token an sequence sampling) are close to
1235

zero.
1236

Guidelines:
1237

• The answer NA means that the paper does not include experiments.
1238

37


---Page Break---
• The authors should answer "Yes" if the results are accompanied by error bars,
1239

confidence intervals, or statistical significance tests, at least for the experiments that
1240

support the main claims of the paper.
1241

• The factors of variability that the error bars are capturing should be clearly stated (for
1242

example, train/test split, initialization, random drawing of some parameter, or overall
1243

run with given experimental conditions).
1244

• The method for calculating the error bars should be explained (closed form formula,
1245

call to a library function, bootstrap, etc.)
1246

• The assumptions made should be given (e.g., Normally distributed errors).
1247

• It should be clear whether the error bar is the standard deviation or the standard error
1248

of the mean.
1249

• It is OK to report 1-sigma error bars, but one should state it. The authors should
1250

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
1251

of Normality of errors is not verified.
1252

• For asymmetric distributions, the authors should be careful not to show in tables or
1253

figures symmetric error bars that would yield results that are out of range (e.g. negative
1254

error rates).
1255

• If error bars are reported in tables or plots, The authors should explain in the text how
1256

they were calculated and reference the corresponding figures or tables in the text.
1257

8. Experiments Compute Resources
1258

Question: For each experiment, does the paper provide sufficient information on the
1259

computer resources (type of compute workers, memory, time of execution) needed to
1260

reproduce the experiments?
1261

Answer: [Yes]
1262

Justification: We are transparent about how many GPU hours it takes to construct our scaling
1263

laws and train our models (e.g., in Table 1).
1264

Guidelines:
1265

• The answer NA means that the paper does not include experiments.
1266

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
1267

or cloud provider, including relevant memory and storage.
1268

• The paper should provide the amount of compute required for each of the individual
1269

experimental runs as well as estimate the total compute.
1270

• The paper should disclose whether the full research project required more compute
1271

than the experiments reported in the paper (e.g., preliminary or failed experiments that
1272

didn’t make it into the paper).
1273

9. Code Of Ethics
1274

Question: Does the research conducted in the paper conform, in every respect, with the
1275

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
1276

Answer: [Yes]
1277

Justification: We have reviewed the code of ethics and feel that our research abides by this
1278

code in every respect.
1279

Guidelines:
1280

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
1281

• If the authors answer No, they should explain the special circumstances that require a
1282

deviation from the Code of Ethics.
1283

• The authors should make sure to preserve anonymity (e.g., if there is a special
1284

consideration due to laws or regulations in their jurisdiction).
1285

10. Broader Impacts
1286

Question: Does the paper discuss both potential positive societal impacts and negative
1287

societal impacts of the work performed?
1288

Answer: [Yes]
1289

38


---Page Break---
Justification: This work is related to predicting the performance of language models, before
1290

they are trained. As such, it falls under the category of basic research. However, because we
1291

produce generative language model artifacts as part of our paper, we recognize that these
1292

pre-trained models can pose risk. We provide a discussion of risks in Appendix G.
1293

Guidelines:
1294

• The answer NA means that there is no societal impact of the work performed.
1295

• If the authors answer NA or No, they should explain why their work has no societal
1296

impact or why the paper does not address societal impact.
1297

• Examples of negative societal impacts include potential malicious or unintended uses
1298

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
1299

(e.g., deployment of technologies that could make decisions that unfairly impact specific
1300

groups), privacy considerations, and security considerations.
1301

• The conference expects that many papers will be foundational research and not tied
1302

to particular applications, let alone deployments. However, if there is a direct path to
1303

any negative applications, the authors should point it out. For example, it is legitimate
1304

to point out that an improvement in the quality of generative models could be used to
1305

generate deepfakes for disinformation. On the other hand, it is not needed to point out
1306

that a generic algorithm for optimizing neural networks could enable people to train
1307

models that generate Deepfakes faster.
1308

• The authors should consider possible harms that could arise when the technology is
1309

being used as intended and functioning correctly, harms that could arise when the
1310

technology is being used as intended but gives incorrect results, and harms following
1311

from (intentional or unintentional) misuse of the technology.
1312

• If there are negative societal impacts, the authors could also discuss possible mitigation
1313

strategies (e.g., gated release of models, providing defenses in addition to attacks,
1314

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
1315

feedback over time, improving the efficiency and accessibility of ML).
1316

11. Safeguards
1317

Question: Does the paper describe safeguards that have been put in place for responsible
1318

release of data or models that have a high risk for misuse (e.g., pretrained language models,
1319

image generators, or scraped datasets)?
1320

Answer: [Yes]
1321

Justification: We provide discussion of responsible release in Appendix G. Specifically,
1322

models in this release are know to be less capable than state-of-the-art, publicly available
1323

models [113, 114, 49], and, hence, we feel the risk for misuse is low.
1324

Guidelines:
1325

• The answer NA means that the paper poses no such risks.
1326

• Released models that have a high risk for misuse or dual-use should be released with
1327

necessary safeguards to allow for controlled use of the model, for example by requiring
1328

that users adhere to usage guidelines or restrictions to access the model or implementing
1329

safety filters.
1330

• Datasets that have been scraped from the Internet could pose safety risks. The authors
1331

should describe how they avoided releasing unsafe images.
1332

• We recognize that providing effective safeguards is challenging, and many papers do
1333

not require this, but we encourage authors to take this into account and make a best
1334

faith effort.
1335

12. Licenses for existing assets
1336

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
1337

the paper, properly credited and are the license and terms of use explicitly mentioned and
1338

properly respected?
1339

Answer: [Yes]
1340

Justification: We utilize data-sources publicly available on the HuggingFace platform and
1341

abide by the terms of use. For C4: Open Data Commons License Attribution family, for
1342

RedPajama: a list of licenses (found here.), for RefinedWeb: Open Data Commons License
1343

39


---Page Break---
Attribution family. We use the OpenLM repo for training and also abide by their MIT license.
1344

We cite all papers and repos in the main text.
1345

Guidelines:
1346

• The answer NA means that the paper does not use existing assets.
1347

• The authors should cite the original paper that produced the code package or dataset.
1348

• The authors should state which version of the asset is used and, if possible, include a
1349

URL.
1350

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
1351

• For scraped data from a particular source (e.g., website), the copyright and terms of
1352

service of that source should be provided.
1353

• If assets are released, the license, copyright information, and terms of use in the
1354

package should be provided. For popular datasets, paperswithcode.com/datasets
1355

has curated licenses for some datasets. Their licensing guide can help determine the
1356

license of a dataset.
1357

• For existing datasets that are re-packaged, both the original license and the license of
1358

the derived asset (if it has changed) should be provided.
1359

• If this information is not available online, the authors are encouraged to reach out to
1360

the asset’s creators.
1361

13. New Assets
1362

Question: Are new assets introduced in the paper well documented and is the documentation
1363

provided alongside the assets?
1364

Answer: [Yes]
1365

Justification: Our code release documents all new model assets under the exp_db/ folder
1366

and includes a MIT license. This is also specified in Appendix H.
1367

Guidelines:
1368

• The answer NA means that the paper does not release new assets.
1369

• Researchers should communicate the details of the dataset/code/model as part of their
1370

submissions via structured templates. This includes details about training, license,
1371

limitations, etc.
1372

• The paper should discuss whether and how consent was obtained from people whose
1373

asset is used.
1374

• At submission time, remember to anonymize your assets (if applicable). You can either
1375

create an anonymized URL or include an anonymized zip file.
1376

14. Crowdsourcing and Research with Human Subjects
1377

Question: For crowdsourcing experiments and research with human subjects, does the paper
1378

include the full text of instructions given to participants and screenshots, if applicable, as
1379

well as details about compensation (if any)?
1380

Answer: [NA]
1381

Justification: This research does not involve crowdsourcing or human subjects.
1382

Guidelines:
1383

• The answer NA means that the paper does not involve crowdsourcing nor research with
1384

human subjects.
1385

• Including this information in the supplemental material is fine, but if the main
1386

contribution of the paper involves human subjects, then as much detail as possible
1387

should be included in the main paper.
1388

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1389

or other labor should be paid at least the minimum wage in the country of the data
1390

collector.
1391

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1392

Subjects
1393

40


---Page Break---
Question: Does the paper describe potential risks incurred by study participants, whether
1394

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1395

approvals (or an equivalent approval/review based on the requirements of your country or
1396

institution) were obtained?
1397

Answer: [NA]
1398

Justification: This paper does not involve research with human subjects.
1399

Guidelines:
1400

• The answer NA means that the paper does not involve crowdsourcing nor research with
1401

human subjects.
1402

• Depending on the country in which research is conducted, IRB approval (or equivalent)
1403

may be required for any human subjects research. If you obtained IRB approval, you
1404

should clearly state this in the paper.
1405

• We recognize that the procedures for this may vary significantly between institutions
1406

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1407

guidelines for their institution.
1408

• For initial submissions, do not include any information that would break anonymity (if
1409

applicable), such as the institution conducting the review.
1410

41


---Page Break---
