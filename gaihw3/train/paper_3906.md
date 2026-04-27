Stacking Your Transformers: A Closer Look at Model
Growth for Efficient LLM Pre-Training

Wenyu Du1∗
Tongxu Luo2,3∗†
Zihan Qiu4
Zeyu Huang5
Yikang Shen6

Reynold Cheng1
Yike Guo2
Jie Fu2‡

1School of Computing and Data Science, The University of Hong Kong
2HKUST
3USTB
4Tsinghua University
5University of Edinburgh
6MIT-IBM Watson AI Lab
wydu@cs.hku.hk
tongxuluo@gmail.com
jiefu@ust.hk

Abstract

LLMs are computationally expensive to pre-train due to their large scale. Model
growth emerges as a promising approach by leveraging smaller models to accelerate
the training of larger ones. However, the viability of these model growth methods
in efficient LLM pre-training remains underexplored. This work identifies three
critical Obstacles: (O1) lack of comprehensive evaluation, (O2) untested viability
for scaling, and (O3) lack of empirical guidelines. To tackle O1, we summarize
existing approaches into four atomic growth operators and systematically evaluate
them in a standardized LLM pre-training setting. Our findings reveal that a depth-
wise stacking operator, called Gstack, exhibits remarkable acceleration in training,
leading to decreased loss and improved overall performance on eight standard
NLP benchmarks compared to strong baselines. Motivated by these promising
results, we conduct extensive experiments to delve deeper into Gstack to address
O2 and O3. For O2 (untested scalability), our study shows that Gstack is scalable
and consistently performs well, with experiments up to 7B LLMs after growth and
pre-training LLMs with 750B tokens. For example, compared to a conventionally
trained 7B model using 300B tokens, our Gstack model converges to the same loss
with 194B tokens, resulting in a 54.6% speedup. We further address O3 (lack of
empirical guidelines) by formalizing guidelines to determine growth timing and
growth factor for Gstack, making it practical in general LLM pre-training. We also
provide in-depth discussions and comprehensive ablation studies of Gstack. Our
code and pre-trained model are available at https://llm-stacking.github.io/.

1
Introduction

Emergent abilities of Large Language Models (LLMs) rely on scaling-up [1, 2]. Empirical evidence
from scaling laws [3–5] fuels the development of increasingly larger models, pushing the boundaries
of LLMs capabilities. However, pre-training these gigantic models comes at a significant cost in
terms of energy consumption and environmental impact [6] (e.g., pre-training Llama-3 [7] consumes
a total of 7.7M GPU hours and generates 2290 tons of carbon dioxide equivalent of carbon emissions).
The efficient pre-training of LLMs is thus crucial, both from a scientific and a societal perspective, to
ensure the continual growth and adoption of AI [8, 9].

One promising research direction to accelerate model training involves leveraging trained smaller
(base) models to expedite the training of larger (target) models, a technique known as model growth.

∗Equal Contributions.
† Work done during interning at HKUST.
‡ Corresponding Author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Concretely, model growth studies how to leverage the trained smaller model’s parameters Θ(s) to
initialize the larger model’s parameters Θ(l). Current popular methods generally focus on expanding
the parameters of the base model through techniques like splitting [10–12], copying [13, 14], or matrix
mapping [15]. There are also some approaches that initialize new parameters from scratch [16, 12, 17].
The primary objective is to accelerate the training of large models, and existing methods demonstrate
promising speedup results on models such as BERT [11, 14, 18, 15, 12, 13]. Despite such empirical
evidence and its alignment with the goal of efficient LLM pre-training, model growth methods are
not widely adopted in the context of LLM pre-training [7, 19]. To our best knowledge, the only LLM
that utilizes model growth for accelerating is FLM-101B [20], but it lacks a baseline LLM trained
from scratch to compare. We observe three key Obstacles that hinder LLM pre-training from using
existing model growth techniques, specifically:

• O1: Lack of comprehensive assessment. Some existing model growth methods report results on
LLM pre-training, but either lack a baseline comparison [20] or are still in exploratory stages [15, 13].
In contrast, most growth approaches are evaluated in encoder-based BERT models [14, 11, 18, 12,
13, 16, 17], which have different architecture and training configurations compared to prominent
decoder-based LLMs such as Llama [21].

• O2: The untested scalability. This scalability has two aspects: the model size and the amount of pre-
training data. Regarding the model size, the existing approaches are only evaluated on smaller-scale
BERT models or in preliminary experiments with LLMs. It is unclear whether these growth methods
will continue accelerating training when applied to large-scale LLMs with more extensive evaluation.
As for the amount of pre-training data, there are debates [22] over whether certain efficient training
strategies may initially converge faster but ultimately perform similarly or worse than vanilla training
methods when given ample computational resources (i.e., more training data).

• O3: Lack of empirical guidelines. Scaling laws [3, 4] give clear empirical guidelines on pre-training
computational-optimized LLMs, greatly stimulating and advancing the field. Yet, there is a lack of
empirical guidelines on growth techniques, discouraging LLM practitioners from adopting these
approaches, especially considering the high costs of LLM pre-training.

These three obstacles are consequential in nature. Hence, in this work, we empirically revisit the
concept of model growth as a solution to efficient LLM pre-training by tackling them one by one.

0
20
40
60
80
100
FLOPs (1e+20)

1.8

2.0

2.2

2.4

2.6

2.8

Training Loss

scratch
Gdirect(Gstack)

0
50
100
150
200
250
300
Tokens (Billions)

40
60
80
100

1.9

2.0

54.6%
0.0%

150
200
250
300

Figure 1: The training loss for two 7B LLMs,
trained from scratch and with G↑
direct (Gstack). At
300B tokens, Gstack accelerates by 54.6% com-
pared to scratch.

To tackle O1, we systematically evaluate model
growth techniques on practical LLM pre-training.
We first categorize existing growth methods and
summarize them into four atomic growth op-
erators, each of which can grow along two di-
rections: widthwise (intra-layer) and depthwise
(layer-wise).
We illustrate them in Figure 2.
These operators serve as representative choices
for evaluating the performance of model growth
techniques. We use these operators to expand
400M base models to 1.1B Llama-like LLMs and
continually pre-train them. Next, we evaluate
these growth techniques on the training loss and
eight standard NLP benchmarks from the Har-
ness toolkit [23]. We found the direct operator
that stacks depthwisely Gstack consistently out-
performs others across overall evaluation metrics,
demonstrating its potential in accelerating LLM
pre-training. This motivates us to investigate ex-
tensively by addressing O2 and O3 on Gstack.

To address O2, we investigate the Gstack opera-
tor’s scalability to larger model sizes and to more
training data. We conduct extensive experiments
by scaling model size up to 7B parameters trained with 300B tokens, and pre-training a 410M
model with over 750B training tokens. This is in contrast to the previous largest LLM pre-training
experiment that uses model growth methods and has baselines for comparison, which is reported
in Ligo [15], where a GPT2-1.5B model is trained for 15k steps (approximately 15B tokens). The

2


---Page Break---
results are encouraging, as we consistently observe significant improvements Gstack offers in both
scenarios. For example, we achieve a remarkable 54.6% speedup in pre-training for a 7B model
with 300B tokens (Figure 1). Interestingly, the loss improvement in our 750B-token experiment
aligns with a logarithmic function. We further extend this logarithmic curve and determine that the
improvement continues to be substantial even for the LLM trained with over 8T tokens. Moreover,
we summarize all our experiments by estimating the LLM scaling law for LLMs pre-trained with
Gstack. Given the same target loss value, our analysis reveals a significantly reduced computational
cost compared to the common scaling law [4].

For O3, we explore the practical guidelines for using Gstack in LLM pre-training. Given a compu-
tational budget, we determine the optimal strategy for two key factors of Gstack, growth timing d
and growth factor g. Growth timing d relates to the training tokens used for small models before
growing, and growth factor g refers to the factor between the non-embedding parameter number of
the large models and the small models. We formalize our findings into equations that offer concrete
suggestions for utilizing Gstack. We believe this work could significantly pique the interest and bolster
confidence in future LLM pre-training with model growth techniques, both in academia and industry.

To summarize, our contributions are four-fold: 1) We first systematically investigate model growth
techniques and identify four atomic model growth operators, establishing a better understanding
of the field in Section 3.1. 2) We then design a standard LLM pre-training testbed and perform
comprehensive evaluations on these operators, finding that a simple depthwise stacking Gstack exhibits
significant superiority in Section 3. 3) We further demonstrate the scalability of Gstack with experi-
ments on LLMs ranging from 410M to 7B parameters and up to 750B training tokens in Section 4.1.
4) We also provide guidelines of equations on determining growth timing and growth factors for
optimal use of Gstack in Section 4.2.

2
Related Work - Model Growth for Efficient Pre-training

The idea of growing neural networks dates back to the 1990s [24–26]. The pioneering work of
Net2Net [10] marks a milestone, for the first attempt to study model growth in deep learning era.
Net2Net expands width and depth while keeping original functions (namely function preserving) via
randomly splitting old neurons and injecting new identity layers. The widthwise splitting method of
Net2Net represents a series of works that aim to “expand” the existing neurons to the desired larger
size. Bert2Bert [11] serves as a BERT-based extension of the widthwise Net2Net. StagedGrow[13]
doubles the width by concatenating two identical layers and halves final loss to keep function-
preserving. Lemon [12] suggests integrating a parameter into the splitting of neurons in Bert2Bert,
aiming to break weight symmetry. Depthwisely, StackedBert [14] simply stacks duplicated layers
to form a deeper model. In contrast to the above direct copy/split approaches, LiGO [15]presents a
learning-based method that initializes the larger model’s parameters via learning a linear mapping
from the smaller model’s parameters.

Alongside the approaches that expand existing parameters, there are works that initialize new ones
without relying on existing ones. For instance, MSG [17] proposes a multi-staged growing strategy
that progressively expands transformer components, where the newly grown neurons are randomly
initialized using a masking mechanism to ensure function preservation. Besides, some works have
assigned specific values, like zero, to the newly initialized neurons to negate their influence [16, 12].

All the above methods are primarily explored in BERT or earlier stages of LLM pre-training. On
the other hand, our objective is to present the first systematic review of model growth techniques in
the LLMs era. To our knowledge, FLM-101B [20] is the only existing LLM that uses the growth
method [17] for accelerating billion-scale LLM pre-training. Nonetheless, this work lacks a baseline
model trained from scratch, making it difficult to assess the effectiveness of the model growth
technique. In contrast, we aim to provide a comprehensive study by establishing a standardized
testbed to compare LLMs trained from scratch and with various growth methods in LLM pre-training.

3
Systematically Assessing Model Growth for LLM Pre-Training

Existing model growth methods [14, 11, 18, 15, 12, 13, 16, 17] are mainly evaluated on BERT [27],
with limited focus on decoder-only large-scale language models such as Llama [21]. Moreover,
these growth methods are often not comparable due to different training settings [14, 11, 17, 12].

3


---Page Break---
Even some growth LLMs experiments are evaluated, their results are often incomplete [20, 15]. To
overcome these limitations, we first summarize existing works [14, 11, 18, 15, 12, 13, 16, 17] into
four atomic growth operators to represent these growth techniques. Then we build a standardized
LLMs training testbed to pre-train LLMs with four growth operators on depthwise and widthwise
directions and evaluate the results with both training loss and eight evaluation metrics in Harness [23].

3.1
Growing LLMs with Growth Operators

Recent years, researchers have focused on enhancing the efficiency of training large models by
making use of smaller pre-existing models [10, 11, 14, 18, 15, 12, 13, 16, 17]. These state-of-the-art
methods can be categorized into two distinct groups. The first group focuses on deriving new neurons
from the existing ones [10, 11, 14, 12, 15], while the second group focuses on initializing new
parameters separately [18, 13, 16, 17]. Drawing from these two lines of research, we summarize four
atomic growth operators. These operators include: (A) directly duplicating and stacking old layers
in a depthwise manner or splitting neurons in the same layer widthwisely, denoted as Gdirect, (B)
generating expanded parameters using a learnable mapping matrix to the existing parameters, denoted
as Glearn, (C) setting the new parameters to zero, denoted as Gzero, and (D) randomly initializing the
new parameters, denoted as Grandom. The illustration of four operators is shown in Figure 2. The
Gdirect and Glearn growth operators produce new neurons from the current ones, in contrast to Gzero
and Grandom which initialize new parameters independently. For the formal definitions of the operators
and the differences to the existing growth methods in design, please refer to Appendix A. Complex
growth methods, such as those involving auxiliary loss or exploring training dynamics like learning
rates [28, 29, 16] are interesting. But considering the high computational cost of LLM pre-training,
we focus on simple, universally applicable growth operators for different LLM pre-training settings.

(c) 𝑮𝒛𝒆𝒓𝒐
→

𝑾𝟎
𝑾𝟏

𝑾𝟐
𝑾𝟑

𝑹𝟎

𝑹𝟏

𝑶
𝑶
𝑹𝟐

𝑩𝒂𝒔𝒆

𝑾𝟎
𝑾𝟏

𝑾𝟐
𝑾𝟑

𝑾Old parameter

𝑫

𝑹

New parameter
from the old

New parameter
from random

Training needed
𝐻
Hyper network
𝑶New parameter 

assigned to zero

(a) 𝑮𝒅𝒊𝒓𝒆𝒄𝒕

→
and 𝑮𝒅𝒊𝒓𝒆𝒄𝒕

↑
(𝑮𝒔𝒕𝒂𝒄𝒌)

𝜶𝑾𝟎
𝑾𝟏

𝜶𝑾𝟐
𝑾𝟑

𝑫𝟎
= 𝜷𝑾𝟎

𝑫𝟏
= 𝜷𝑾𝟐

𝑫𝟒
= 𝜶𝑾𝟐

𝑫𝟑
= 𝑾𝟑

𝑫𝟐
= 𝜷𝑾𝟐

α + β = 1
Split

Copy

(d) 𝑮𝒓𝒂𝒏𝒅𝒐𝒎

→

𝑾𝟎
𝑾𝟏

𝑾𝟐
𝑾𝟑

𝑹𝟒
𝑹𝟑

Layer

Mask

0 →1
𝑹𝟎

𝑹𝟏

𝑹𝟐

(b) 𝑮𝒍𝒆𝒂𝒓𝒏

→

𝑾𝟎
𝑾𝟏

𝑾𝟐
𝑾𝟑

𝑫𝟎
𝑫𝟏

𝑫𝟑
𝑫𝟒

𝑫𝟐

𝑫𝟓

𝑫𝟔
𝑫𝟕
𝑫𝟖

𝐻

×

=

Layer

Layer

Copy

Figure 2: The simplified illustration of four growth operators Gdirect, Glearn, Gzero and Grandom, each
of which can grow along widthwise (intra-layer) G→or depthwise (layer-wise) G↑. Wn is the
parameters before growth, while Dn , Rn and O are the growth parameters derived from the old,
randomly initialized, and zero-initialized respectively. Except Gdirect, other three operators only
illustrates the widthwise growth.

To make a fair comparison of the four growth operators for LLM pre-training, we define a standardized
“one-hop” growth process that involves two training phases, small model training before growth and
large model training after growth. We first train the small LLMs with d tokens before growing. Then,
we use operator G to grow them to the target LLMs by a factor of g for non-embedding parameters

4


---Page Break---
and then continual pre-training the large LLMs for D tokens. Two key factors in the procedure are
worth noting: the growth factor g and the data for base model training d, which can be interpreted as
“growth timing”. We further evaluate each growth operator by separately examining in depthwise
(intra-layer) growth G↑and widthwise (layer-wise) growth G→. Concretely, we start with base
models (400M LLMs) trained on d = 10B tokens, apply the four operators in both directions to scale
them up to the target size of 1.1B (approximately a growth factor of g = 4), and then continue training
for an additional D = 97.5B tokens. 4 Appendix B contains the LLM’s architecture configuration
and training details.

3.2
Pre-Training 1.1B LLMs

We report results on training loss, eight standard Harness NLP benchmarks along with the average
accuracy and the speedup ratio in Figure 3. Our key discovery reveals that depthwise growth G↑
exhibits a significant acceleration over both widthwise growth G→and training models from scratch,
while surprisingly, G→does not offer any notable advantages. Among the depthwise growth operators,
G↑
direct, G↑
learn, and G↑
zero, all outperform the baseline and G↑
random. The underperformance of G↑
random
in our study may be attributed to its design for gradual “mini-step” growth [17], whereas our unified
approach uses a single step. Most notably, depthwise stacking G↑
direct emerges as the clear winner
among growth operators, surpassing its competitors in speedup, training loss and nearly every
Harness evaluation metric. For example, compared to training models from scratch for 100B tokens,
G↑
direct achieves a significant efficiency gain, increasing training speed by 49.1%. The calculation of
speedup please refer to Appendix B.2. The Appendix C presents more experiments on these operators,
including their loss training and evaluation figures.

|
|
|
|
|
|
|
|
|
48.20
48.67
44.14
48.36
46.16
44.67
44.24
45.66
47.87

29.18
28.32
28.41
27.38
28.58
26.70
27.64
26.70
27.21

54.25
51.76
52.69
51.17
51.55
49.70
53.82
50.37
48.86

28.87
27.95
25.96
28.11
27.34
25.03
26.11
26.57
25.96

71.98
71.81
70.78
71.16
69.47
69.74
70.13
69.91
69.64

81.1
81.9
77.7
80.0
81.4
76.0
79.5
79.5
76.8

56.03
56.98
53.35
54.45
54.22
54.93
52.95
53.51
54.53

52.80
52.48
50.43
51.52
51.25
49.54
50.63
50.32
50.12

16.73
17.35
17.85
16.93
18.03
18.76
18.29
18.44
17.98

2.151
2.161
2.258
2.156
2.209
2.249
2.227
2.233
2.204

49.1%
46.6%
-25.7%
48.6%
-0.7%
-17.9%
-13.8%
-15.4%
0.0%

Lambada (↑) -

ARC-c (↑) -

ARC-e (↑) -

Logiqa (↑) -

PIQA (↑) -

Sciq (↑) -

Winogrande (↑) -

Avg. (↑) -

Wikitext (↓) -

Loss (↓) -

Depth
Width

𝑠𝑐𝑟𝑎𝑡𝑐ℎ
Baseline
𝐺𝑑𝑖𝑟𝑒𝑐𝑡

↑
𝐺𝑧𝑒𝑟𝑜
↑
𝐺𝑟𝑎𝑛𝑑𝑜𝑚

↑
𝐺𝑙𝑒𝑎𝑟𝑛

↑
𝐺𝑑𝑖𝑟𝑒𝑐𝑡

→
𝐺𝑧𝑒𝑟𝑜
→
𝐺𝑟𝑎𝑛𝑑𝑜𝑚

→
𝐺𝑙𝑒𝑎𝑟𝑛

→

Speed-up (↑) -

Figure 3: We evaluate operators using training loss and Lambada [30], ARC-c [31], ARC-e [31],
Logiqa [32], PIQA [33], Sciq [34], Winogrande [35] and Wikitext PPL [36] totaling eight standard
NLP benchmarks. After 8 × 1020 FLOPs of training, G↑
direct demonstrates a significant speedup.

4
Delving Deeper Into Depthwise Stacking (Gstack)

The empirical evidence suggests that certain growth operators, most notably G↑
direct, exhibit an
impressive acceleration in LLM pre-training compared to the baseline approach of training models
from scratch. We now turn our attention to a more in-depth examination of the G↑
direct. For ease
of reference, we will henceforth denote this depthwise stacking approach as operator Gstack:

4Given growth factor g = 4, the sum of FLOPs for training d = 10B and D = 97.5B approximately equals
to consumption for training large LLMs D = 100B, which is the FLOPs of our baseline trained from scratch.

5


---Page Break---
M = M ◦M ◦· · · ◦M
|
{z
}
g×M
, where M is a small base model trained with d tokens, M is the target

model and g is the growth factor.

This section addresses the two main challenges (O2 and O3) outlined in the introduction: 1) evaluating
the performance of Gstack in scaling scenarios, i.e. larger model sizes and more training tokens; and
2) determining the hyperparameters when using Gstack, i.e., the growth timing d and growth factor g.

4.1
Scaling Gstack

0
10
20
30
40
FLOPs (1e+20)

2.0

2.2

2.4

2.6

2.8

Training Loss

scratch
Gstack

0
50
100
150
200
250
300
Tokens (Billions)

20
25
30
35
40
1.95

2.00

2.05

2.10

2.15

48.6%
0.0%

54.5%
0.0%

150
200
250

(a) Training Loss

10
20
30
40
FLOPs (1e+20)

42

44

46

48

50

52

54

56

Average Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(b) Average Accuracy

Figure 4: Training 3B LLMs with 300B tokens. Gstack significantly
outperforms scratch in (a) loss and (b) average accuracy across
NLP benchmarks. At 180B and 240B tokens, Gstack accelerates by
48.6% and 54.5% compared to scratch.

Scaling Model Sizes for Gstack.
Our scaled-up experiments in-
volve two larger model sizes:
3B and 7B. We initially train
smaller models with a layer
count that is one-quarter of
our target layers (growth factor
g = 4), utilizing 10B tokens
(d = 10B). Then, we train the
stacked models using over 300B
tokens (D = 300B) for both
sizes.
Figures 4 and 5 show
the loss, and the NLP bench-
marks average accuracy eval-
uated using the Harness eval-
uator for training 3B and 7B
LLMs with 300B tokens, re-
spectively. 5 The acceleration of
Gstack is consistent across two
models and all evaluation metrics. For instance, considering the 3B model, Figure 4 demonstrates
that Gstack achieves a 54.5% speedup in pre-training, improvement of 2.1 in NLP benchmarks average
accuracy compared to the baseline 3B model trained with 240B tokens.

When comparing the 1B, 3B, and 7B models, it is evident that the benefits of Gstack are not reduced
as the model size increases, implying that its acceleration effect can be leveraged even with larger
models. Details of the evaluation results, including evaluation with instruction tuning, can be found in
Appendix D. Appendix E compares our baselines with the open-source LLMs Pythia and tinyLlama.

0
20
40
60
80
100
FLOPs (1e+20)

1.8

2.0

2.2

2.4

2.6

2.8

Training Loss

scratch
Gstack

0
50
100
150
200
250
300
Tokens (Billions)

40
60
80
100

1.9

2.0 40.8%
0.0%
55.3%
0.0%

53.8%
0.0%

150
200
250
300

(a) Training Loss

20
40
60
80
100
FLOPs (1e+20)

40

42

44

46

48

50

52

54

Average Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(b) Average Accuracy

Figure 5: Training 7B LLMs with 300B tokens. Gstack significantly
outperforms scratch in (a) loss and (b) average accuracy across NLP
benchmarks. At 160B, 220B and 280B tokens, Gstack accelerates
by 40.8%, 55.3% and 53.8% compared to scratch.

Scaling Training Tokens for
Gstack.
We next evaluate the
scalability of the stacking opera-
tor on another dimension - train-
ing with more tokens. This is
especially important in light of
recent discussions about the va-
lidity of efficient training algo-
rithms, which have sparked de-
bate [22] over whether certain
strategies may initially learn
faster but ultimately perform
similarly or worse than vanilla
training methods when given
more training data. Hence, we
aim to pre-train a LLM using
Gstack on a substantial amount
of training tokens.

5In this study, we always calculate the consumption by combining the FLOPs required for both training small
models and large models. So given g = 4, the consumption for training small model d = 10B equals to the cost
for training D = 2.5B, so the plotted curves for Gstack actually starts at 2.5B.

6


---Page Break---
0
5
10
15
FLOPs (1e+20)

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

Training Loss

scratch
Gstack

0
200
400
600
Tokens (Billions)

6
8
10
12
14
16

2.25

2.30

2.35

53.1%
0.0%

33.7%
0.0%

31.0%
0.0%

300
400
500
600
700

(a) Training Loss

0
50
100
150
Flops (1e+20)

0.00

0.02

0.04

0.06

0.08

0.10

Training Loss Difference

scratch
Gstack

0
2000
4000
6000
8000
Tokens (Billions)

(b) Estimated Loss Difference

Figure 6: Training 410M LLMs with 750B tokens. Gstack signifi-
cantly outperforms scratch in (a) loss. At 400B tokens, we observe
a 53.1% acceleration, and even at 700B tokens, there is still a
31.0% acceleration. (b) We fit the difference between the losses
of the scratch and Gstack and find that the acceleration with Gstack
remain sustainable for longer training.

Concretely, we conduct an ex-
periment on a 410M LLM us-
ing 750B tokens. Following the
experimental setup in the previ-
ous section, we set growth ra-
tio g = 4 and growth timing
d = 10B and conduct contin-
uous pre-training on the target
410M LLMs for 750B tokens.
Compared to the chinchilla-
recommended 8B tokens [4] for
the 410M model, our experi-
mental setting also surpasses
this value by nearly 100 times,
reaching 750B tokens.

The training dynamics on Fig-
ure 6a indicate that Gstack re-
mains effective in such cases.
Details of the evaluation results
with the similar findings can be
found in Appendix D.3. Building upon the exceptional stability of LLM pre-training [37, 38], we
estimate loss improvements and plot them in Figure 6b. The fitting curve indicates Gstack will con-
tinue to exhibit acceleration effects even after 8T tokens, which is over 1000 times longer than the
recommended token number [4]. It is also notable that this loss improvement after 8T training is not
trivial for LLM pre-training, as previous studies [39] suggest that even minor improvements in the
later phase can have a relatively substantial impact on downstream performance.

From a LLM practitioner’s perspective, this is also crucial considering “overtraining”, which involves
training LLMs with significantly larger amounts of data than recommended by scaling laws [3–5], a
common practice that has become prevalent. A notable example is the training of LLama 3-8B with
15T tokens, which is nearly 100 times greater than the token count recommended by the chinchilla
scaling laws [4]. Hence, this finding provides confidence in the consistent excellent acceleration of
Gstack throughout the entire practical LLM pre-training process.

0.1
1
10
100 1000
FLOPs (1e+20)

1

2

3

4

6

9

Training Loss

scratch(410M)
Gstack(410M)

scratch(1.1B)
Gstack(1.1B)

scratch(3B)
Gstack(3B)

scratch(7B)

Gstack(7B)

Scratch Law
Stacking Law
scratch(13B)
Gstack(13B)

scratch(70B)
Gstack(70B)

Figure 7: We plot scaling law lines
based on 410M, 1.1B, 3B, 7B LLMs
and make two predictions at the
same losses of original computational-
optimized 13B and 70B LLMs.

Estimating Scaling Laws.
To further explore our find-
ings, we graph our four models (410M, 1.1B, 3B, and 7B)
on the same figure and attempt to uncover our “scaling
law” using the Gstack operator. Following [3, 4], we de-
fine the scaling power law using the equation LC = aCb,
where a and b are constants we need to fit, C represents the
FLOPs, and LC denotes the model’s final loss under this
FLOP. We use the curve_filt function in SciPy [40] to fit
both the scratch model and the Gstack model and present the
estimation scaling law in Figure 7. The figure shows that
our Gstack scaling law exhibits improved efficiency com-
pared to the scaling law estimated from baseline LLMs,
achieving the same target loss while requiring much less
computational resources. However, in light of the signifi-
cant computational resources devoted to other scaling law
studies [3, 4], we acknowledge that our Gstack scaling law
is an initial estimate subject to computation constraints, and
a comprehensive study is left for future research.

4.2
Determining Growth Timing and Growth Factor for Using Gstack

We comprehensively validate the effectiveness of the Gstack compared to training from scratch in
Section 4.1. However, to incorporate Gstack into a LLM’s pre-training process, we need to determine
two crucial hyperparameters: the growing time (d) and the growing factor (g). In our previous
experiments, we rely on ad-hoc choices for these parameters, thereby lacking a systematic approach

7


---Page Break---
to determining them when use Gstack. There exists research on investigating the growth timing [41],
but the settings are quite different from the LLM pre-training. Therefore, this section offers a clear
guide for practitioners looking to optimize using the Gstack operator in LLM pre-training processes.

We begin by offering a formal definition. When given a computational budget C, established scaling
power laws [3, 4] exist to guide the non-embedding parameters N and the number of training tokens D
to achieve the lowest model loss in the case of training from scratch. However, tuning hyperparameters
becomes more complex when the fixed budget C is allocated to find the optimal model training
strategy using the Gstack operator, which involves two training phases. Consequently, the overall
computational budget C can be expressed as the sum of the two components: C = C1 + C2.
Here, C1 and C2 represent the flops required to train the initial small models C1 = FLOPs(n, d),
and the large model C2 = FLOPs(N, D) respectively, where n and d denote the parameters and
training tokens of the small model, and N and D represent the parameters and training tokens of
the large model. Since the large model is grown by a factor of g such that N = gn, we have
C = C1 + C2 = FLOPs(g, N, d) + FLOPs(N, D) = FLOPs(g, N, d, D).

0
1
5
10
20
50
Growth Timing (Billions of tokens)

2.4

2.5

2.6

2.7

Training Loss

0.8

1.0

1.2

1.4

1.6

1.8

2.0

2.2

FLOPs (1e+20)

(a) IsoFLOP on 410M

0
1
5
10
20
50
Growth Timing (Billions of tokens)

2.2

2.3

2.4

2.5

2.6

Training Loss

2

3

4

5

6

7

FLOPs (1e+20)

(b) IsoFLOP on 1.1B

0
1
5
10
20
50
Growth Timing (Billions of tokens)

2.2

2.3

2.4

2.5

2.6

2.7

Training Loss

4

5

6

7

8

9

10

11

12

FLOPs (1e+20)

(c) IsoFLOP on 3B

Figure 8: In 410M, 1.1B, and 3B LLMs, we plot smoothed loss curves for different growth timing d
given a set of FLOPs to form IsoFLOP figures. We find a clear valley in loss, indicating that for a
given FLOP budget, there exists an optimal growth timing d for the Gstack operation.

So when given a budget C, our objective is to identify the optimized values D, N, d, g that
minimize the loss L(D, N, d, g). However, simultaneously optimizing the above four variables can
be computationally expensive. Therefore, instead of searching for global optimals, we separately
determine two factors closely related to the Gstack: the training tokens for the small model (growth
timing) d and the growth factor g:

arg min
f,h
L(D, N, d, g),
where d = f(D, N), g = h(D, N)
(1)

0
5
10
15
FLOPs (1e+20)

5

10

15

20

25

30

35

Growth Timing (Billions of tokens)

1

2

3

4

5

Parameters (Billions)

Figure 9: We fit a contour figure
for predicting d given C and N.
These optimal growth timing d fit
the figure well.

Determining Growth Timing: d.
We first explore the effect
of growth timing, i.e. the training token d for the small model.
Particularly, we apply the Gstack operator to a series of small
models trained with d = 0B, 1B, 5B, 10B, 20B, 50B tokens.
Subsequently, we stack them to the target layers with growth
factor g = 4 and train for a fixed set of computational FLOPs.
We replicate the above experiments using three target model sizes
N = 410M, 1.1B, 3B and plot each set of IsoFLOP points in
Figure 8a, 8b and 8c. Surprisingly, even a small model trained
with just 1B tokens exhibits a significant speedup compared to
the directly stacking small random initialized models (represented
as “0B”). While 0B’s performance is similar to models trained
from scratch, implying stacking itself does not serve as an effec-
tive initialization method. Furthermore, by applying smoothing
techniques to model IsoFLOP curves as parabolas, we identify
the optimized value of d that minimizes loss for each FLOP count,
leading us to hypothesize the existence of a logarithmic equation involving N, C, and d:

8


---Page Break---
log10(d) = a log10(N) +
b
log10(C) + c
(2)

After fitting, we obtain a = 0.88, b = 163.27 and c = −5.74 and we plot the contour figure in
Figure 9. It can be observed that our estimated curves align well with the actual optimal points.

1
2
4
8
24
Growth Factor

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

Training Loss

2

3

4

5

6

7

FLOPs (1e+20)

(a) IsoFLOP on 1.1B

1
4
8
16
32
Growth Factor

2.4

2.6

2.8

3.0

3.2

3.4

Training Loss

2

3

4

5

6

7

FLOPs (1e+20)

(b) IsoFLOP on 3B

Figure 10: In 1.1B, and 3B LLMs, we plot smoothed loss curves
for different growth factor g given a set of FLOPs as IsoFLOP
figures. The optimal g falls between 2 and 4.

Determining Growth Factor:
g.
Another factor we deter-
mine is the growth factor g.
As models with 3B and 7B pa-
rameters have identical depths,
we run experiments using two
model sizes: 1.1B (24 layers)
and 3B (32 layers).
Specifi-
cally, we vary the stack fac-
tors to g = 2, 4, 8, 24 for the
1.1B model and g = 4, 8, 16, 32
for the 3B model while keep-
ing the base models trained with
d = 10B tokens. The smoothed
IsoFLOP curves are plotted in
Figure 10. Interestingly, even
with a relatively shallow 2-layer
base model and a growth factor
of g = 16, we observe a remarkable improvement compared to the baseline 3B model (g = 1).
However, when using a 1-layer base model, Gstack underperforms compared to the baseline. Our
curves indicate that the optimal growth factor g lies between 2 and 4.

However, unlike determining training token d, we cannot generate sufficient data to estimate the
relationship between N, C, and g, due to computational constraints. Thus, this work suggests a
constant growth factor of g = 4. We also include our preliminary estimated equation and contour
figure for g in the Appendix F. All evaluation results of Section 4.2 are listed in Appendix G.

5
Ablation and Discussion

To further give insights into adopting model growth techniques in LLM pre-training, we ablate
variances for Gstack and discuss function preserving in general model growth techniques.

5.1
Ablation: How to Stack?

It is worth noting that Gstack differs from the algorithm proposed in StackedBERT [14], which
utilizes a gradually stacking strategy. Hence, we compare our “one-hop” Gstack and their gradual
stacking approach. Following the methodology introduced in StackBERT, we employ a two-step
stack strategy. Given our target model size of 1.1B with 24 layers, we start with a 6-layer model.
Subsequently, we train it on 10B tokens and double the model’s depth through stacking, repeating
this step twice (train-stack-train-stack) to achieve the desired scale. Our experiments demonstrate that
Gstack outperforms gradual stacking approaches on loss and downstream evaluations. For example,
the evaluation results show that Gstack achieves a 2.4 higher average accuracy and 0.6 better Wikitext
PPL than gradual stacking when pre-training large models for 100B tokens. The results can be found
in Appendix H.1. We further compare other stacking variations, such as stacking via interpolation
and partial stacking of certain layers which are also adopted in LlamaPro [42] and Solar [43], and
leave our detailed findings in the Appendix H.2 and H.3.

5.2
Discussion: Why Does Function Preserving Fail?

Function preservation (FP) is a key concept that underlies most model growth approaches [10–12, 17].
The idea is intuitive that a larger model should initialize parameters that can represent the same

9


---Page Break---
function as the ones in the smaller model, i.e. ∀x, f(x; Θ(s)) = f(x; Θ(l)
init), where x is the input.
We give a mathematical definition of FP in the Appendix I.1.

We find it intriguing that our Gstack approach, which violates FP, emerges as the most effective in our
study. To further investigate, we conduct a simple ablation study to break FP by introducing noise
on the strict-FP operator G→
direct. We initialize the new neurons by a weighted combination of two
sets of parameters: those from G→
direct and those from random initialization. The weighting factor is
controlled by a noise ratio. Our findings are intriguing. After 40B tokens training, adding 20% noise
outperforms original G→
direct by 0.27 on the Wikitext PPL and 0.41 on the average accuracy score.

We also add noise for Gstack. When we add 20% noise, our LLM performs slightly better than
the no-noise model. However, when the noise level exceeds 20%, the performance significantly
deteriorates. These results indicate that function preservation may not be the sole determining factor
for model growth. In other words, exploring ways to accelerate the training of larger models and
strict preserving function during growth might represent two overlapping yet distinct research
directions. The experimental details are provided in the Appendix I.2.

6
Conclusion

This work empirically explores model growth approaches for efficient LLM pre-training. We address
three key challenges of current model growth research for efficient LLM pre-training. We first
comprehensively evaluate model growth techniques into four atomic operators and explore depthwise
growth Gstack beats all other methods and baselines in various evaluations. We next address concerns
about the scalability of Gstack by extending the model and training data scales. Furthermore, we
systematically analyze the usage of the Gstack operator, focusing on growth timing and growth factor.
Based on this analysis, we formalize a set of guidelines for effectively utilizing the Gstack operator. In
addition, we provide in-depth discussions and comprehensive ablation studies of Gstack, shedding
light on the broader implications of our work.

7
Limitations

While our work has demonstrated remarkable potential, four limitations deserve further attention.
One limitation is the constraint of computation resources. For example, we only compare two sets
of growth factor d configurations, which limits the capacity to derive a formula for determining the
optimal growth factor d. Another limitation of our work is the focus on relatively simple operator
choices, where we prioritize simplicity over exploring more sophisticated strategies. For instance, we
do not extensively investigate the multi-step growth or dynamic modifications to the training process,
such as adjusting the learning rate during continual pre-training. The third limitation involves the
incomplete cosine learning rate schedule during training. This also arises from the resource-intensive
nature of pre-training LLMs and the constraints on available computational resources. Therefore, we
adopt a strategy where we initially set a large number of training tokens and then we pre-train LLMs
until the training runs are interrupted by tasks with higher priority. Lastly, although this study’s scope
is an empirical exploration and the content is self-contained, there is a lack of theoretical insights
into the success of Gstack in LLM pre-training.6 Nonetheless, we will release all LLM checkpoints to
facilitate the community’s investigation into the theoretical principles behind our observations.

8
Acknowledgments

We thank all constructive comments from anonymous reviewers. Reynold Cheng and Wenyu Du were
supported by the Hong Kong Jockey Club Charities Trust (Project 260920140), the University of
Hong Kong (Project 109000579), the HKU Outstanding Research Student Supervisor Award 2022-23,
and the HKU Faculty Exchange Award 2024 (Faculty of Engineering).

6A very recent paper indicates training LLMs via stacking may improve in reasoning [44].

10


---Page Break---
References

[1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz
Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
Cited on page 1.

[2] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani
Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto,
Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language
models, 2022. Cited on page 1.

[3] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models, 2020. Cited on pages 1, 2, 7, and 8.

[4] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom
Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia
Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent
Sifre. Training compute-optimal large language models, 2022. Cited on pages 2, 3, 7, and 8.

[5] Ibrahim Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws
in language and vision, 2022. Cited on pages 1 and 7.

[6] Mengwei Xu, Wangsong Yin, Dongqi Cai, Rongjie Yi, Daliang Xu, Qipeng Wang, Bingyang
Wu, Yihao Zhao, Chen Yang, Shihe Wang, Qiyang Zhang, Zhenyan Lu, Li Zhang, Shangguang
Wang, Yuanchun Li, Yunxin Liu, Xin Jin, and Xuanzhe Liu. A survey of resource-efficient llm
and multimodal foundation models, 2024. Cited on page 1.

[7] AI@Meta. Llama 3 model card, 2024. Cited on pages 1, 2, and 30.

[8] Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan
Maeng, Gloria Chang, Fiona Aga Behram, James Huang, Charles Bai, Michael Gschwind,
Anurag Gupta, Myle Ott, Anastasia Melnikov, Salvatore Candido, David Brooks, Geeta
Chauhan, Benjamin Lee, Hsien-Hsin S. Lee, Bugra Akyildiz, Maximilian Balandat, Joe Spisak,
Ravi Jain, Mike Rabbat, and Kim Hazelwood. Sustainable ai: Environmental implications,
challenges and opportunities, 2022. Cited on page 1.

[9] Alex de Vries. The growing energy footprint of artificial intelligence. Joule, 7(10):2191–2194,
2023. Cited on page 1.

[10] Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowl-
edge transfer. arXiv preprint arXiv:1511.05641, 2015. Cited on pages 2, 3, 4, 9, and 16.

[11] Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, Zhi Wang, Xiao
Chen, Zhiyuan Liu, and Qun Liu. bert2bert: Towards reusable pretrained language models.
arXiv preprint arXiv:2110.07143, 2021. Cited on pages 2, 3, 4, and 16.

[12] Yite Wang, Jiahao Su, Hanlin Lu, Cong Xie, Tianyi Liu, Jianbo Yuan, Haibin Lin, Ruoyu Sun,
and Hongxia Yang. Lemon: Lossless model expansion, 2023. Cited on pages 2, 3, 4, 9, and 16.

[13] Sheng Shen, Pete Walsh, Kurt Keutzer, Jesse Dodge, Matthew Peters, and Iz Beltagy. Staged
training for transformer language models. In International Conference on Machine Learning,
pages 19893–19908. PMLR, 2022. Cited on pages 2, 3, 4, and 16.

[14] Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training
of bert by progressively stacking. In International conference on machine learning, pages
2337–2346. PMLR, 2019. Cited on pages 2, 3, 4, 9, and 16.

11


---Page Break---
[15] Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid Karlinsky,
Rogerio Feris, David Daniel Cox, Zhangyang Wang, and Yoon Kim. Learning to grow pretrained
models for efficient transformer training. arXiv preprint arXiv:2303.00980, 2023. Cited on
pages 2, 3, 4, and 16.

[16] Utku Evci, Bart van Merrienboer, Thomas Unterthiner, Max Vladymyrov, and Fabian Pe-
dregosa. Gradmax: Growing neural networks using gradient information. arXiv preprint
arXiv:2201.05125, 2022. Cited on pages 2, 3, 4, and 16.

[17] Yiqun Yao, Zheng Zhang, Jing Li, and Yequan Wang. Masked structural growth for 2x faster
language model pre-training, 2024. Cited on pages 2, 3, 4, 5, 9, and 16.

[18] Cheng Yang, Shengnan Wang, Chao Yang, Yuechuan Li, Ru He, and Jingqiao Zhang. Progres-
sively stacking 2.0: A multi-stage layerwise training method for bert training speedup. arXiv
preprint arXiv:2011.13635, 2020. Cited on pages 2, 3, 4, and 16.

[19] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut
Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. Cited on
page 2.

[20] Xiang Li, Yiqun Yao, Xin Jiang, Xuezhi Fang, Xuying Meng, Siqi Fan, Peng Han, Jing Li,
Li Du, Bowen Qin, Zheng Zhang, Aixin Sun, and Yequan Wang. Flm-101b: An open llm and
how to train it with $100k budget, 2023. Cited on pages 2, 3, and 4.

[21] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas
Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes,
Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony
Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian
Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut
Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov,
Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta,
Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiao-
qing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng
Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien
Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation
and fine-tuned chat models, 2023. Cited on pages 2, 3, and 30.

[22] Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt J. Kusner. No train
no gain: Revisiting efficient training algorithms for transformer-based language models, 2023.
Cited on pages 2 and 6.

[23] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas
Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron,
Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework
for few-shot language model evaluation, 12 2023. Cited on pages 2 and 4.

[24] Scott Fahlman and Christian Lebiere. The cascade-correlation learning architecture. In D. Touret-
zky, editor, Advances in Neural Information Processing Systems, volume 2. Morgan-Kaufmann,
1989. Cited on page 3.

[25] Scott E. Fahlman. The recurrent cascade-correlation architecture. In Neural Information
Processing Systems, 1990. No citations.

[26] Steven Gutstein, Olac Fuentes, and Eric A. Freudenthal. Knowledge transfer in deep convolu-
tional neural nets. In Int. J. Artif. Intell. Tools, 2007. Cited on page 3.

[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of
deep bidirectional transformers for language understanding, 2019. Cited on page 3.

12


---Page Break---
[28] Lemeng Wu, Bo Liu, Peter Stone, and Qiang Liu. Firefly neural architecture descent: a general
approach for growing neural networks, 2021. Cited on page 4.

[29] Xin Yuan, Pedro Savarese, and Michael Maire. Accelerated training via incrementally growing
neural networks using variance transfer and learning rate adaptation, 2023. Cited on page 4.

[30] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi,
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset:
Word prediction requiring a broad discourse context, 2016. Cited on page 5.

[31] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge, 2018. Cited on page 5.

[32] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A
challenge dataset for machine reading comprehension with logical reasoning, 2020. Cited on
page 5.

[33] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning
about physical commonsense in natural language, 2019. Cited on page 5.

[34] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science
questions, 2017. Cited on page 5.

[35] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An
adversarial winograd schema challenge at scale, 2019. Cited on page 5.

[36] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture
models, 2016. Cited on page 5.

[37] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen,
Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and
Ji-Rong Wen. A survey of large language models, 2023. Cited on page 7.

[38] Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, Yangrui Chen, Zhi Zhang, Yanghua Peng,
Xiang Li, Cong Xie, Shibiao Nong, Yulu Jia, Sun He, Hongmin Chen, Zhihao Bai, Qi Hou,
Shipeng Yan, Ding Zhou, Yiyao Sheng, Zhuo Jiang, Haohan Xu, Haoran Wei, Zhang Zhang,
Pengfei Nie, Leqi Zou, Sida Zhao, Liang Xiang, Zherui Liu, Zhe Li, Xiaoying Jia, Jianxi Ye,
Xin Jin, and Xin Liu. Megascale: Scaling large language model training to more than 10,000
gpus, 2024. Cited on page 7.

[39] Zhengxiao Du, Aohan Zeng, Yuxiao Dong, and Jie Tang. Understanding emergent abilities of
language models from the loss perspective, 2024. Cited on page 7.

[40] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cour-
napeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der
Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nel-
son, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng, Eric W. Moore, Jake
VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero,
Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mul-
bregt, Aditya Vijaykumar, Alessandro Pietro Bardelli, Alex Rothberg, Andreas Hilboll, Andreas
Kloeckner, Anthony Scopatz, Antony Lee, Ariel Rokem, C. Nathan Woods, Chad Fulton,
Charles Masson, Christian Häggström, Clark Fitzgerald, David A. Nicholson, David R. Hagen,
Dmitrii V. Pasechnik, Emanuele Olivetti, Eric Martin, Eric Wieser, Fabrice Silva, Felix Lenders,
Florian Wilhelm, G. Young, Gavin A. Price, Gert-Ludwig Ingold, Gregory E. Allen, Gregory R.
Lee, Hervé Audren, Irvin Probst, Jörg P. Dietrich, Jacob Silterra, James T Webber, Janko Slaviˇc,
Joel Nothman, Johannes Buchner, Johannes Kulick, Johannes L. Schönberger, José Vinícius
de Miranda Cardoso, Joscha Reimer, Joseph Harrington, Juan Luis Cano Rodríguez, Juan
Nunez-Iglesias, Justin Kuczynski, Kevin Tritz, Martin Thoma, Matthew Newville, Matthias
Kümmerer, Maximilian Bolingbroke, Michael Tartre, Mikhail Pak, Nathaniel J. Smith, Nikolai
Nowaczyk, Nikolay Shebanov, Oleksandr Pavlyk, Per A. Brodtkorb, Perry Lee, Robert T.

13


---Page Break---
McGibbon, Roman Feldbauer, Sam Lewis, Sam Tygier, Scott Sievert, Sebastiano Vigna, Ste-
fan Peterson, Surhud More, Tadeusz Pudlik, Takuya Oshima, Thomas J. Pingel, Thomas P.
Robitaille, Thomas Spura, Thouis R. Jones, Tim Cera, Tim Leslie, Tiziano Zito, Tom Krauss,
Utkarsh Upadhyay, Yaroslav O. Halchenko, and Yoshiki Vázquez-Baeza. Scipy 1.0: fundamen-
tal algorithms for scientific computing in python. Nature Methods, 17(3):261–272, February
2020. Cited on page 7.

[41] Haihang Wu, Wei Wang, Tamasha Malepathirana, Damith Senanayake, Denny Oetomo, and
Saman Halgamuge. When to grow? a fitting risk-aware policy for layer growing in deep neural
networks, 2024. Cited on page 8.

[42] Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, and Ying
Shan. Llama pro: Progressive llama with block expansion. arXiv preprint arXiv:2401.02415,
2024. Cited on pages 9 and 37.

[43] Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeon-
woo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, et al. Solar 10.7 b: Scaling large language
models with simple yet effective depth up-scaling. arXiv preprint arXiv:2312.15166, 2023.
Cited on pages 9 and 37.

[44] Nikunj Saunshi, Stefani Karp, Shankar Krishnan, Sobhan Miryoosefi, Sashank J. Reddi, and
Sanjiv Kumar. On the inductive bias of stacking towards improving reasoning, 2024. Cited on
page 10.

[45] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small
language model, 2024. Cited on page 22.

[46] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast
and memory-efficient exact attention with io-awareness, 2022. Cited on page 22.

[47] Daria
Soboleva,
Faisal
Al-Khateeb,
Robert
Myers,
Jacob
R
Steeves,
Joel
Hestness,
and
Nolan
Dey.
SlimPajama:
A
627B
token
cleaned
and
dedu-
plicated
version
of
RedPajama.
https://www.cerebras.net/blog/
slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama,
2023. Cited on page 22.

[48] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a
machine really finish your sentence?, 2019. Cited on page 27.

[49] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic
human falsehoods, 2022. Cited on page 27.

[50] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding, 2021. Cited on page
27.

[51] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien,
Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward
Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In
International Conference on Machine Learning, pages 2397–2430. PMLR, 2023. Cited on page
29.

[52] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason
Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse
text for language modeling. arXiv preprint arXiv:2101.00027, 2020. Cited on pages 29 and 39.

[53] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord,
Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David
Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar,
Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff,
Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander,
Dustin Schwenk, Saurabh Shah, Will Smith, Nishant Subramani, Mitchell Wortsman, Pradeep
Dasigi, Nathan Lambert, Kyle Richardson, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A.

14


---Page Break---
Smith, and Hannaneh Hajishirzi. Olmo: Accelerating the science of language models. Preprint,
2024. Cited on page 39.

[54] Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao,
Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao
Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen,
Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim
Baldwin, and Eric P. Xing. Llm360: Towards fully transparent open-source llms, 2023. Cited
on page 39.

[55] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Rus-
sell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann,
Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob
Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha
Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind
Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk
Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an Open Corpus of Three Trillion Tokens for
Language Model Pretraining Research. arXiv preprint, 2024. Cited on page 39.

15


---Page Break---
A
Details of Growth Operators

A.1
Four Growth Operators

A.1.1
Operator Gdirect: Direct Derivation of Grown Parameters From Old Parameters

One intuitive strategy for expanding neural networks involves directly duplicating or splitting existing
neurons. [14, 11, 12]. Unlike other growth operators, we distinguish between growth in terms of
depth and width.

For width-wise expansion, the Net2Net technique and its transformer implementations [10, 11]
involve splitting old neurons into two or more parts, with each splitting step achieving a=b+c.
Depending on the specific splitting mechanism, there are two variations: even splitting and uneven
splitting. The latter is proposed to address symmetry issues that arise when neurons are evenly split.
In this paper, we adopt the approach of uneven splitting.

In the context of depth-wise expansion, a common practice is to duplicate layers, often referred to as
“stacking” [14]. Therefore, we use the term Gstack to represent this operator. While this approach may
appear to deviate from function preservation, it surprisingly yields a strong baseline.

A.1.2
Operator Glearn: Generation of New Parameters through Matrix Transformation

Glearn is an operator that learns a matrix transformation function to map small models to a larger
one [15]. This operator is applicable to both width and depth expansion. Considering the original
model f with parameters θ, the target model F with parameters Θ, and Glearn as the hypernetwork
for meta-learning, the training corpus is denoted as D, and the language model loss is denoted as L.
Then, we optimize the following objective:

arg min
Glearn
Ex∼D L(x; FΘ),
where Θ = Glearn(θ)
(3)

A.1.3
Operator Gzero: Setting New Parameters to 0

Setting new parameters to zero is often considered a simple method to achieve function preservation.
However, optimizing networks with a significant number of zeros can present challenges. To
tackle this issue, we adopt current practices that selectively zero out either the fan-in or fan-out
parameters [13, 16, 12]. Specifically, for operator Gzero, during width growing, we zero out only the
set of fan-out parameters for new neurons and randomly initialize the remaining ones. In the case of
depthwise expansion, we zero out the final output layer of the newly-duplicated transformer blocks’
MultiHead Attention and MLP.

A.1.4
Operator Grandom: Random Initialization of New Parameters

This group follows the common practice of randomly initializing new parameters. In earlier attempts,
old neurons were frozen after the growth process [18, 17]. However, to ensure function preservation, a
recent study introduces a mask for new neurons after expansion [17]. This mask is gradually removed
during ongoing training. We refer to this new approach as the growth operator Grandom.

A.2
Difference of Our Operators and Base Methods

The operators G→
direct shares a similar setting to Lemon with minor variances due to Llama achitectures.
Glearn is consistent with the methods LiGO, but with our own implementation. For Gzero, our approach
aligns with Lemon in terms of depth, but differs from stagedTraining in width. Unlike stagedTraining,
we do not double the width and assign zeros to the off-diagonal entries. Instead, our approach is more
flexible; by zeroing out the submatrix in the bottom-left corner, we can extend it to any dimension.
Our Grandom does not exhibit the “multi-hop” growth like MSG, instead, it grows “one-hop” directly
to the target size. Our implementation of G↑
direct (Gstack) differs from the algorithm employed in
stackedBert. In stackedBert, a gradual growing technique is utilized, whereas our operator follows a
more direct approach.

16


---Page Break---
A.3
Details of Gdirect

Embedding
Consider E ∈RV ×d, and our goal is to expand it to E′ ∈RV ×D, Gdirect just copy
some columns:

E′
=
Gdirect(E)
(4)
=
ER
(5)

=
E

I
I

|{z}
d

I



(6)

where R ∈Rd×D is used to copy the embedding matrix E.

Linear
Consider W ∈Rdout×din, target parameter W ′ ∈RDout×Din, where dout ≤Dout, din ≤
Din, Gdirect is defined as:

W ′
=
Gdirect(W)
(7)
=
LWR
(8)

=
dout

 "I
I
I

#

W

α
β

|{z}
din

I



(9)

where R ∈Rdin×Din is used for expanding the fan-in and L ∈RDout×dout is used for expanding the
fan-out. To satisfy function preserving, we ensure that α + β = I.

RMSNorm
For RMSNorm, a similar approach is adopted, consider parameter µ ∈Rd, expanded
parameter µ′ =
√

d
√

D[µ, µ0,D−d] ∈RD:

RMSNorm′(x′) =
x′
q

1
D
PD
i=1 x′2
i
⊙µ′
(10)

= [

v
u
u
t

Pd
i=1 x2
i
PD
i=1 x′2
i
× RMSNorm(x), ζ]
(11)

Therefore, using the Gdirect, it is not possible to achieve function preservation for RMSNorm

Depth (Gstack)
Consider a transformer with l layers represented as F = f0 ◦f1 ◦· · · ◦fl. Our
objective is to expand it to L layers, where L is a multiple of l. We have various stacking forms for
this purpose, such as (a) direct stacking: F ′ = F ◦F ◦· · · ◦F.

Algorithm 1 Operator Gstack
Input: Base model M l
k with l layers trained using dataset dk where k is iteration steps. Growth
factor g.
Output: Target Model Mgl
0 with gl layers
Ml
0=M l
k
for t = 2 to g do
▷Model Stacking
Mtl
0 = M(t−1)l
0
◦M l
k
end

17


---Page Break---
A.4
Details of Gzero

Embedding
Consider an embedding matrix E ∈RV ×d. The Gzero operator expands it to E′ ∈
RV ×D with O, where d ≤D. Formally:

E′ = [E, O]
(12)

Therefore, give a token x, the expanded embedding can be expressed as:

Embedding′(x) = 1xE′ = [Embedding(x), 0D−d]
(13)

Linear
Consider parameter W ∈Rdout×din. Gzero expand it to W ′ ∈RDout×Din, where dout ≤
Dout and din ≤Din. Formally:

W ′ =

W
A
O
C


(14)

where A, C are randomly initialized new parameters. Considering the input token x ∈Rdin before
expansion, and the input after expansion x′ ∈RDin:

x′ = [x, 0Din−din]
(15)

Linear′(x′)
=
x′W ′T
(16)

=
[x, 0Din−din]

W T
O
AT
CT



(17)

=
[xW T , 0Dout−dout]
(18)
=
[Linear(x), 0Dout−dout]
(19)

RMSNorm
Considering the parameter µ ∈Rd, Gzero expand it to µ′ = [αµ, ξ] like Grandom in
Appendix A.5, because the input must be x′ = [x, 0D−d] ∈RD.

Depth
In depth, by retaining only the residual part and initializing the MHA and SwiGLU final
linear projections to zero, the MHA and SwiGLU layers can achieve function preservation.

A.5
Details of Grandom

Embedding
Consider an embedding matrix E ∈RV ×d. The goal of Grandom is to expand it to
E′ ∈RV ×D, where d ≤D. Formally:

E′ = [E, E]
(20)

where E ∈RV ×(D−d) represents randomly initialized new parameters. We use a mask c ∈RD to
mask out the randomly initialized parts:

c = [1d, 0D−d] →[1d, 1D−d]
(21)

Therefore, for a token x, the masked embedding can be expressed as:

Embedding′(x) = 1xE′ ⊙c = [Embedding(x), 0D−d]
(22)

18


---Page Break---
Linear
Consider parameter W ∈Rdout×din. Our goal is to expand it to W ′ ∈RDout×Din, where
dout ≤Dout and din ≤Din. Formally:

W ′ =

W
A
B
C


(23)

where A, B, C are randomly initialized new parameters. Considering the input token x ∈Rdin before
expansion, and the input after expansion x′ ∈RDin:

x′ = [x, 0Din−din]
(24)

x′W ′T
=
[x, 0Din−din]

W T
BT

AT
CT



(25)

=
[xW T , xBT ]
(26)

To ensure that the expanded part of x′ starts with zeros, we still utilize a mask:

c = [1dout, 0Dout−dout] →[1dout, 1Dout−dout]
(27)

Linear′(x′) = x′W ′T ⊙c = [Linear(x), 0Dout−dout]
(28)

RMSNorm
Considering the parameter µ ∈Rd, our objective is to expand it to µ′ = [αµ, ξ] ∈RD,
where α is an undetermined coefficient and ξ is a randomly initialized new parameter. Let the input
be x′ = [x, 0D−d] ∈RD, then we have:

D
X

i=0
x′2 =

d
X

i=0
x2
(29)

RMSNorm′(x′)
=
x′
q

1
D
PD
i=0 x′
i
2 ⊙µ′
(30)

=
[x, 0D−d]
q

1
D
Pd
i=0 xi2
⊙[αµ, ξ]
(31)

=




√

D
√

d
x
q

1
d
Pd
i=0 xi2
⊙αµ, 0D−d




(32)

By observing equation 32, we can conclude that, to achieve function preservation, α =
√

d
√

D. Finally,
we can conclude:

RMSNorm′(x′) = [RMSNorm(x), 0D−d]
(33)

Depth
In depth, preserving only the residual part and masking the MHA and SwiGLU layers can
achieve function preservation:

Y = X + MHA(RMSNorm(X)) ⊙c
(34)
Y = X + SwiGLU(RMSNorm(X)) ⊙c
(35)
c = 0D →1D
(36)

19


---Page Break---
A.6
Details of Glearn

Using Glearn for width expansion, for the embedding layer E ∈RV ×d, the parameter Bemb ∈RD×d
is defined as follows:

E′ = EBT
emb
(37)

For Attention layer, where WQ, WK, WV , and WO ∈Rd×d, and RMSNorm µ1 ∈Rd, the parameters
BQ, BK, and BV ∈RD×d, we have:












W ′
Q
=
BQWQBT
emb
W ′
K
=
BKWKBT
emb
W ′
V
=
BV WV BT
emb
W ′
O
=
BembWOBT
V
µ′
1
=
Bembµ1

(38)

For MLP, where Wup, Wgate ∈Rdmlp×d, Wdown ∈Rd×dmlp, RMSNorm µ2 ∈Rd, the parameter
Bmlp ∈RDmlp×dmlp, we have:










W ′
up
=
BmlpWupBT
emb
W ′
down
=
BembWmlpBT
mlp
W ′
gate
=
BmlpWgateBT
emb
µ′
2
=
Bembµ2

(39)

For the output head Whead ∈RV ×d, we have:

W ′
head = WheadBemb
(40)

Using Glearn for depth expansion, consider a transformer model with L1 layers, we use Glearn to
expand it to L2 layers. For l ∈{1, 2, · · · , L2}:


















W Q
l
′
=
PL1
j=1 DQ
l,jW Q
j
W K
l
′
=
PL1
j=1 DK
l,jW K
j
W V
l
′
=
PL1
j=1 DV
l,jW V
j
W O
l
′
=
PL1
j=1 DO
l,jW O
j
µ(ln1)
l

′
=
PL1
j=1 D(ln1)
l,j
µ(ln1)
j

(41)

where DQ,K,V,O,ln1 ∈RL2×L1 represents learnable parameters. These parameters are used to
expand the MHA vertically in depth. Similarly, for SwiGLU, we also perform expansion using a
similar method. Formally, this can be written as:














W up
l
′
=
PL1
j=1 Dup
l,jW up
j
W down
l
′
=
PL1
j=1 Ddown
l,j
W down
j
W gate
l
′
=
PL1
j=1 Dgate
l,j W gate
j
µ(ln2)
l

′
=
PL1
j=1 D(ln2)
l,j
µ(ln2)
j

(42)

where Dup,down,gate,ln2 ∈RL2×L1 represents learnable parameters used for expanding SwiGLU in
the depth.

20


---Page Break---
B
LLMs Framework and Training Details

Embedding
Consider a vocabulary size V and embedding size d. Then, the embedding matrix
E ∈RV ×d, and the one-hot vector for input tokens X is denoted as 1X ∈RT ×V , where T is the
sequence length. Formally, it can be written as:

Embedding(X) = 1XE
(43)

for i, v ∈[V ], where i ̸= j, it is guaranteed that Ei ̸= Ej.

Multi-Head Attention
Multi-Head Attention (MHA) consists of multiple attention heads, each of
which computes its own self-attention. The results of these attention heads are then concatenated and
projected to obtain the following output:

Qi, Ki, Vi = XW Q
i , XW K
i , XW V
i
Hi = softmax( QiKT
i
√dh )Vi
MHA(X) = Concat(H1, · · · , Hn)W O
(44)

here, the input X ∈RT ×d, parameters W Q
i
∈Rd×dh, W K
i
∈Rd×dh, W V
i
∈Rd×dh, and W O ∈
Rd×d, where n × dh = d.

Feed Forward Network
The Feed Forward Network (FFN) consists of two linear layers and the
activation function GeLU. Typically, the two linear layers first perform an up-projection to dF F N
and then down-project back to the dimension d. Therefore, FFN is defined as:

FFN(X) = GeLU(XWup)Wdown
(45)

where the input X ∈RT ×d, parameter Wup ∈Rd×dF F N and Wdown ∈RdF F N×d.

SwiGLU
LLaMA replaces the original FFN in the Transformer Decoder with SwiGLU, resulting
in improved performance. SwiGLU consists of three linear layers and the swiglu activation function.
It can be defined as:

SwiGLU(X) = (XWgate ⊙swiglu(XWup))Wdown
(46)

where ⊙means the element-wise multiplication, the input X ∈RT ×d, parameter Wup ∈Rd×dF F N ,
Wgate ∈Rd×dF F N and Wdown ∈RdF F N×d.

RMSNorm
Before MHA, FFN, or SwiGLU, there is a layer of RMSNorm to enhance the stability
of the model. Compared to LayerNorm, RMSNorm is simpler in form. Formally, it can be written as:

RMSNorm(X) =
X
q

1
d
Pd
i=1 X2
i
⊙µ
(47)

where X ∈RT ×d, parameter µ ∈Rd.

21


---Page Break---
B.1
LLMs Training with Growth Operator

Algorithm 2 LLMs Training with Growth Operator
Input: Growth operator G, Loss function L, Iterative optimizer A. Dataset {d1, d2, · · · , dk} for base
model. Dataset {D1, D2, · · · , DK} for target model.
Output: Target Model MK
Initial Phase: Initialize a base model M0 from scratch.
for t = 1 to k do
▷Base Model Training
loss = L(Mt−1, dt)
Mt ←A(Mt−1, loss)
end
M0 = G(Mk)
for t = 1 to K do
▷Target Model Training
loss = L(Mt−1, Dt)
Mt ←A(Mt−1, loss)
end

B.2
Details of Speedup Calculation

We calculate speedup sp between operator G and scratch model pre-training by:

sp = FLOPsscratch

FLOPsG
−1
(48)

where FLOPsscratch and FLOPsG represent the FLOPs required by the scratch model and the G
model, respectively, to achieve the same loss.

B.3
Details of Training Settings

We use TinyLlama 7 [45] as our pre-training codebase. We employ FSDP (Fully Sharded DataParallel)
along with FlashAttention [46] 2.0, and other acceleration techniques. We use the open-source dataset
Slimpajama-627B 8 [47] for pre-training. The hyperparameters used for each model size are listed in
Table 1. Our 7B model is trained over around 100B tokens per day on an NVIDIA Hopper cluster.

Table 1: Hyperparameters

Size
Context Length
Batch Size
max-LR
min-LR
Warmup Steps
LR Scheduler
410M
2048
2M tokens
6e-4
6e-5
3000
cosine
1.1B
2048
2M tokens
3e-4
3e-5
3000
cosine
3B
2048
2M tokens
1.6e-4
1.6e-5
3000
cosine
7B
2048
2M tokens
1e-4
1e-5
3000
cosine

C
Training Loss and Evaluation Results of Four Operators in both Depth and
Width growth

We have two small (base) models, one trained with token count d = 10B and another trained with
token count d = 50B.

7Apache-2.0 license
8The license of Slimpajama-627B includes: Common Crawl Foundation Terms of Use; C4 license; GitHub
was limited to MIT, BSD, or Apache licenses only; Books: the_pile_books3 license and pg19 license; ArXiv
Terms of Use; Wikipedia License; StackExchange license on the Internet Archive

22


---Page Break---
0
2
4
6
8
FLOPs (1e+20)

2.1

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch

Gdirect
Glearn

Grandom
Gzero

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

70
80

(a) Growing in depth from small model (10B)

0
2
4
6
8
FLOPs (1e+20)

2.1

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch

Gdirect
Glearn

Grandom
Gzero

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

70
80

(b) Growing in depth from small model (50B)

0
2
4
6
8
FLOPs (1e+20)

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch
Gdirect
Glearn

Grandom
Gzero

0
20
40
60
80
Tokens (Billions)

6
7
2.20

2.25

2.30

70
80

(c) Growing in width from small model (10B)

0
1
2
3
4
5
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

3.2

Training Loss

scratch
Gdirect
Glearn

Grandom
Gzero

0
10
20
30
40
50
60
Tokens (Billions)

4.0
4.5

2.30

2.35

2.40

45
50
55
60

(d) Growing in width from small model (50B)

Figure 11: Training Loss on Slimpajama.

23


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

42

44

46

48

50

52

54

56

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

29

30

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

65

66

67

68

69

70

71

72

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

68

70

72

74

76

78

80

82

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

48

50

52

54

56

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0
17.0

17.5

18.0

18.5

70
80

(h) Wikitext (ppl ↓)

Figure 12: Evaluation results on growth in depth from small model (10B) by four operators.

2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

42

44

46

48

50

52

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

29

30

31

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

65

66

67

68

69

70

71

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

68

70

72

74

76

78

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

50

51

52

53

54

55

56

57

58

Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0

17.5

18.0

18.5

19.0

70
80

(h) Wikitext (ppl ↓)

Figure 13: Evaluation results on growth in depth from small model (50B) by four operators.

24


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

33

36

39

42

45

48

51

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

20

25

30

35

40

45

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

58

60

62

64

66

68

70

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

64

66

68

70

72

74

76

78

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

49

50

51

52

53

54

55

56

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

20

25

30

35

40

45

50

Word Perplexity

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0

20

22

70
80

(h) Wikitext (ppl ↓)

Figure 14: Evaluation results on growth in width from small model (10B) by four operators.

2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

33

36

39

42

45

48

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

10

15

20

25

30

35

40

45

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

24

25

26

27

28

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

56

58

60

62

64

66

68

70

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

55

60

65

70

75

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

49

50

51

52

53

54

55

Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

15

20

25

30

35

40

45

50

Word Perplexity

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0

16

18

20

70
80

(h) Wikitext (ppl ↓)

Figure 15: Evaluation results on growth in width from small model (50B) by four operators.

25


---Page Break---
2
4
6
8
FLOPs (1e+20)

44

46

48

50

52

Average Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(a) Growing in depth from
small model (10B)

2
4
6
8
FLOPs (1e+20)

44

46

48

50

Average Accuracy

scratch

Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(b) Growing in depth from
small model (50B)

2
4
6
8
FLOPs (1e+20)

38

40

42

44

46

48

50

Average Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(c) Growing in width from
small model (10B)

2
4
6
8
FLOPs (1e+20)

36

38

40

42

44

46

48

50

Average Accuracy

scratch
Gdirect
Glearn

Grandom
Gzero

20
40
60
80
Tokens (Billions)

(d) Growing in width from
small model (50B)

Figure 16: Average accuracy of seven standard NLP benchmarks.

D
Evaluation Results of Scaling Gstack

D.1
3B

10
20
30
40
FLOPs (1e+20)

42

44

46

48

50

52

54

56

Average Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

Figure 17: Average accuracy of standard NLP benchmarks at 3B size.

26


---Page Break---
10
20
30
40
FLOPs (1e+20)

24

26

28

30

32

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(a) ARC-c (Acc ↑)

10
20
30
40
FLOPs (1e+20)

42

45

48

51

54

57

60

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(b) ARC-e (Acc ↑)

10
20
30
40
FLOPs (1e+20)

30

35

40

45

50

55

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(c) Lambada (Acc ↑)

10
20
30
40
FLOPs (1e+20)

25

26

27

28

29

30

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(d) Logiqa (Acc ↑)

10
20
30
40
FLOPs (1e+20)

62

64

66

68

70

72

74

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(e) PIQA (Acc ↑)

10
20
30
40
FLOPs (1e+20)

65

70

75

80

85

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(f) Sciq (Acc ↑)

10
20
30
40
FLOPs (1e+20)

50

52

54

56

58

60

62

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(g) Winogrande (Acc ↑)

10
20
30
40
FLOPs (1e+20)

15

18

21

24

27

30

Word Perplexity

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

30
35
40

14

15

16

220
240
260
280

(h) Wikitext (ppl ↓)

Figure 18: Evaluation results on scratch model and Gstack model at 3B size.

D.2
7B

20
40
60
80
100
FLOPs (1e+20)

24

26

28

30

32

34

36

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(a) ARC-c (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

40

45

50

55

60

65

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(b) ARC-e (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

30

35

40

45

50

55

60

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(c) Lambada (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

26

27

28

29

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(d) Logiqa (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

66

68

70

72

74

76

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(e) PIQA (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

70

75

80

85

90

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(f) Sciq (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

52

54

56

58

60

62

64

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(g) Winogrande (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

35

40

45

50

55

60

65

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(h) Hellaswag [48] (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

35

36

37

38

39

40

41

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(i) TruthfulQA[49] (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

23

24

25

26

Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(j) MMLU [50] (Acc ↑)

20
40
60
80
100
FLOPs (1e+20)

12

15

18

21

24

27

Word Perplexity

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

80
90

12

13

14

220
240
260

(k) Wikitext (ppl ↓)

20
40
60
80
100
FLOPs (1e+20)

40

42

44

46

48

50

52

54

Average Accuracy

scratch
Gstack

50
100
150
200
250
300
Tokens (Billions)

(l) Avg. (Acc ↑)

Figure 19: Evaluation results on scratch model and Gstack model at 7B size.

27


---Page Break---
D.3
410M

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

42

43

44

45

46

47

48

49

Average Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

Figure 20: Average accuracy of standard NLP benchmarks at 410M size.

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

23

24

25

26

27

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(a) ARC-c (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

38

40

42

44

46

48

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(b) ARC-e (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

28

30

32

34

36

38

40

42

44

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(c) Lambada (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

24

25

26

27

28

29

30

31

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(d) Logiqa (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

61

62

63

64

65

66

67

68

69

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(e) PIQA (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

64

66

68

70

72

74

76

78

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(f) Sciq (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

50

51

52

53

54

55

Accuracy

scratch
Gstack

200
400
600
Tokens (Billions)

(g) Winogrande (Acc ↑)

2.5
5.0
7.5
10.0
12.5
15.0
FLOPs (1e+20)

21

24

27

30

33

36

Word Perplexity

scratch
Gstack

200
400
600
Tokens (Billions)

12
14
19

20

21

500
550
600
650

(h) Wikitext (ppl ↓)

Figure 21: Evaluation results on scratch model and Gstack model at 410M size.

D.4
Instruction Tuning Results on 3B

Table 2: Evaluation Results after Instruction-Tuning (Higher better)

Method
Tokens
Tuning
lambada
arc-c
arc-e
logiqa
piqa
sciq
winogrande
avg

scratch
400B
é
54.07
28.84
55.35
26.88
73.94
82.0
59.43
54.36
Ë
60.35
31.48
56.1
27.04
74.32
81.2
60.14
55.8

Gstack
290B
é
55.04
32.34
58.08
28.88
73.88
79.6
61.8
55.66
Ë
61.34
34.98
59.97
29.65
75.14
80.1
60.22
57.34

28


---Page Break---
E
Compare with Other Opensource LLMs

In Table 3, we compare the harness evaluation results after training the Gstack model and the scratch
model (Baseline) for 100B tokens with Pythia-1B [51] and TinyLlama-1.1B, which are trained on
the same number of tokens. The comparative results indicate that our baseline performs normally,
comparable to pythia-1B. Meanwhile, the Gstack model significantly outperforms both the baseline
and pythia-1B, demonstrating the acceleration effect of Gstack on the pre-training process.

Table 3: Compare with opensource LLMs on 1B

Pythia-1B
TinyLlama-1.1B
Gstack-1.1B
Baseline-1.1B
Datasets
Pile-300B [52]
Slimpajama-627B& Starcoder
Slimpajama-627B
Slimpajama-627B
Tokens
100B
103B
100B
100B
lambada
53.52
-
48.20
47.87
ARC-c
25.59
24.32
29.18
27.21
ARC-e
47.26
44.91
54.25
48.86
piqa
69.31
67.30
71.98
69.64
logiqa
29.49
-
28.87
25.96
sciq
77.3
-
81.1
76.8
winogrande
51.22
53.28
56.03
54.53
Avg.
50.53
-
52.80
50.09

F
Fitting Results for the Growth Factor g

Although due to computational resource limitations, we only explore predicting g given N and C on
the 1.1B and 3B models, we still attempted to fit using equation:

log10(g) = a log10(N) +
b
log10(C) + c
(49)

In the equation 49, N represents the number of target parameters, g represents the growth factor. The
fitting result is as follows:

log10(g) = 1.01 log10(N) −
29.88
log10(C) −7.36
(50)

We also visualize the fitted curves in Figure 22, but the results were mediocre due to the lack of data.

0
5
10
15
FLOPs (1e+20)

0

2

4

6

8

10

12

Growth Factor

0

1

2

3

4

5

Parameters (Billions)

Figure 22: Visualization of the Equation 50.

29


---Page Break---
F.1
Stacking Law Guidelines For Llama Families

We give an example of empirical usage of Gstack by using the configurations of Llama2 and Llama3
families [21, 7] to show the estimated optimal base model training tokens d and growth factor g in
Table 4.

Table 4: “Stacking Law” Guidelines

Model
N
D
d
g
Llama3-8B
8B
15T
6.58B
4
Llama2-7B
7B
2T
11.11B
4
Llama2-13B
13B
2T
15.84B
4
Llama2-70B
70B
2T
42.48B
4

G
Training Loss and Evaluation Results of “growth timing” and “growth
factor”

G.1
“Growth Timing” d

0.0
0.5
1.0
1.5
2.0
FLOPs (1e+20)

2.4

2.6

2.8

3.0

3.2

3.4

3.6

Training Loss

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

0
20
40
60
80
Tokens (Billions)

1.4
1.6
1.8

2.4

2.5

70
80

(a) Training Loss

0.5
1.0
1.5
2.0
FLOPs (1e+20)

42

43

44

45

46

47

Average Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 23: Training loss and standard NLP benchmarks average accuracy of 410M.

30


---Page Break---
0.5
1.0
1.5
2.0
FLOPs (1e+20)

22

23

24

25

26

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

38

39

40

41

42

43

44

45

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

26

28

30

32

34

36

38

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

26

27

28

29

30

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

61

62

63

64

65

66

67

68

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

62

64

66

68

70

72

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

50

51

52

53

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

0.5
1.0
1.5
2.0
FLOPs (1e+20)

22

24

26

28

30

32

34

36

Word Perplexity

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

1.6
1.8

23

24

70
80

(h) Wikitext (ppl ↓)

Figure 24: Evaluation results on 410M.

0
2
4
6
8
FLOPs (1e+20)

2.1

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
FLOPs (1e+20)

44

46

48

50

52

Average Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 25: Training loss and standard NLP benchmarks average accuracy of 1.1B.

31


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

40

42

44

46

48

50

52

54

56

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

29

30

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

68

70

72

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

68

70

72

74

76

78

80

82

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

51

52

53

54

55

56

57

Accuracy

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch
Gstack(6L; 1B)

Gstack(6L; 5B)

Gstack(6L; 10B)

Gstack(6L; 20B)

Gstack(6L; 50B)

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0
17.0

17.5

18.0

18.5

70
80

(h) Wikitext (ppl ↓)

Figure 26: Evaluation results on 1.1B.

0.0
2.5
5.0
7.5
10.0
12.5
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

Training Loss

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

0
20
40
60
80
Tokens (Billions)

10
11
12

2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
10
12
14
FLOPs (1e+20)

42

44

46

48

50

52

Average Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 27: Training loss and standard NLP benchmarks average accuracy of 3B.

32


---Page Break---
2
4
6
8
10
12
14
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

40

42

44

46

48

50

52

54

56

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

30

35

40

45

50

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

25

26

27

28

29

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

62

64

66

68

70

72

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

66

69

72

75

78

81

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

50

51

52

53

54

55

56

57

58

Accuracy

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

16

18

20

22

24

26

28

30

Word Perplexity

scratch
Gstack(8L; 1B)

Gstack(8L; 5B)

Gstack(8L; 10B)

Gstack(8L; 20B)

Gstack(8L; 50B)

Gstack(8L; 100B)

20
40
60
80
Tokens (Billions)

10
11
12

16

17

18

70
80

(h) Wikitext (ppl ↓)

Figure 28: Evaluation results on 3B.

G.2
“Growth Factor” g

0
2
4
6
8
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

Training Loss

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

2.30

70
80

(a) Training Loss

2
4
6
8
FLOPs (1e+20)

40

42

44

46

48

50

52

Average Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 29: Training loss and standard NLP benchmarks average accuracy of 1.1B.

33


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

39

42

45

48

51

54

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

30

35

40

45

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

24

25

26

27

28

29

30

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

62

64

66

68

70

72

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

60

65

70

75

80

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

49

50

51

52

53

54

55

56

57

Accuracy

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

21

24

27

30

33

Word Perplexity

scratch
Gstack(12L × 2)

Gstack(6L × 4)

Gstack(3L × 8)

Gstack(1L × 24)

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0

18

19

70
80

(h) Wikitext (ppl ↓)

Figure 30: Evaluation results on 1.1B.

0.0
2.5
5.0
7.5
10.0
12.5
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

Training Loss

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

0
20
40
60
80
Tokens (Billions)

10
11
12

2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
10
12
14
FLOPs (1e+20)

38

40

42

44

46

48

50

52

Average Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 31: Training loss and standard NLP benchmarks average accuracy of 3B.

34


---Page Break---
2
4
6
8
10
12
14
FLOPs (1e+20)

22

24

26

28

30

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

36

39

42

45

48

51

54

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

15

20

25

30

35

40

45

50

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

24

25

26

27

28

29

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

60

62

64

66

68

70

72

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

60

65

70

75

80

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

50

51

52

53

54

55

56

57

58

Accuracy

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

15

20

25

30

35

40

45

50

Word Perplexity

scratch
Gstack(8L × 4)

Gstack(4L × 8)

Gstack(2L × 16)

Gstack(1L × 32)

20
40
60
80
Tokens (Billions)

10
11
12

16

18

20

70
80

(h) Wikitext (ppl ↓)

Figure 32: Evaluation results on 3B.

H
Discussion on “How to stack?” and Evaluation Results

H.1
Training Loss and Evaluation Results of Gradual Stack

0
2
4
6
8
FLOPs (1e+20)

2.1

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch
Gstack(6L; 10B)

Ggradual(6L; 50B)

Ggradual(12L; 50B)

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
FLOPs (1e+20)

44

46

48

50

52

Average Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 33: Training loss and standard NLP benchmarks average accuracy of scratch, Gstack and
Ggradual.

35


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

42

44

46

48

50

52

54

56

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

29

30

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

65

66

67

68

69

70

71

72

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

68

70

72

74

76

78

80

82

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

52

53

54

55

56

57

Accuracy

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch
Gstack

Ggradual

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0
17.0

17.5

18.0

18.5

70
80

(h) Wikitext (ppl ↓)

Figure 34: Evaluation results on scratch, Gstack and gradual stacking in StackBert.

H.2
Ablation: f2 ◦f1 ◦f0 ◦f2 ◦f1 ◦f0 or f2 ◦f2 ◦f1 ◦f1 ◦f0 ◦f0 (interpolation)

To investigate whether the connections between layers affect the performance of stacking, we
conduct a comparison of two approaches for stacking small models into larger ones. We explore
two approaches for stacking small models into larger ones. The first approach involves taking the
entire small model as a unit and directly stacking it, which can retain the connections between most
layers. The second approach involves replicating and interleaving each layer in the small model,
which almost break the connections. To measure the degree of retention of inter-layer connections
after stacking, we define the connection rate Rc:

Rc = Conr

Conall
(51)

where the Conr is number of retained connections, the Conall is number of all connections.

For example, if we had a small model with three layers, denoted as f2 ◦f1 ◦f0, and desired a model
depth of 6, the first approach would result in f2 ◦f1 ◦f0 ◦f2 ◦f1 ◦f0, where its Rc = 80%. The
second approach would result in f2 ◦f2 ◦f1 ◦f1 ◦f0 ◦f0, where its Rc = 40%.

In our experiments, we stack a small model with 8 layers to a 24 layers target model. The growth
timing d is 10B tokens and growing factor s is 3. The Rc of Gstack is 91.3% and the Rc of Ginterpolate
is 30.4%. We report the training loss and standard NLP benchmarks average accuracy in Figure 35.
At the beginning of training, interpolated stacking perform as well as stacking entire small model.
However, as the training continues, the performance of interpolated stacking deteriorates.

Therefore, we can conclude that the higher the connection rate of stacking, the better the effect of
stacking. In Appendix H.3, we continue to validate this conclusion.

36


---Page Break---
0.0
2.5
5.0
7.5
10.0
12.5
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

Training Loss

scratch
Ginterpolate

Gstack

0
20
40
60
80
Tokens (Billions)

10
11
12

2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
10
12
14
FLOPs (1e+20)

42

44

46

48

50

52

Average Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 35: Training loss and standard NLP benchmarks average accuracy of scratch, Gstack and
interpolation.

We also report the details of evaluation results about 8 standard NLP benchmarks.

2
4
6
8
10
12
14
FLOPs (1e+20)

24

25

26

27

28

29

30

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

42

44

46

48

50

52

54

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

30

35

40

45

50

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

25

26

27

28

29

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

62

64

66

68

70

72

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

66

69

72

75

78

81

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

51

52

53

54

55

56

57

58

Accuracy

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
10
12
14
FLOPs (1e+20)

16

18

20

22

24

26

28

30

Word Perplexity

scratch
Ginterpolate

Gstack

20
40
60
80
Tokens (Billions)

10
11
12

16

17

18

70
80

(h) Wikitext (ppl ↓)

Figure 36: Evaluation results on scratch, Gstack and interpolation.

H.3
Ablation: Partial Stacking

Partial stacking has been explored in LLMs like LlamaPro [42], Solar [43]. But their goal is to stack
an off-the-shelf LLMs such as Llama2, while our aim is to accelerate LLM pre-training process.

To explore stacking which layers of the small model can achieve the best performance, we con-
duct experiments on partial stacking. In our experiments, we stack a small model with 6 lay-
ers ({L1, L2, · · · , L6}) to a 24 layers target model. We set growth timing d = 10B tokens and
growth factor g = 4. For simplicity, we use a format such as 1-234*7-56 to denote stacking 234
layers 7 times.

37


---Page Break---
0
2
4
6
8
FLOPs (1e+20)

2.1

2.2

2.3

2.4

2.5

2.6

2.7

2.8

2.9

3.0

Training Loss

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)
Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)
Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)
Gstack(1234
56 * 10)

0
20
40
60
80
Tokens (Billions)

6
7
2.15

2.20

2.25

70
80

(a) Training Loss

2
4
6
8
FLOPs (1e+20)

44

46

48

50

52

Average Accuracy

scratch
Gstack(123456 * 4)

Gstack(123 * 7
456)
Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)
Gstack(12
3456 * 5
56)

Gstack(12
34 * 10
56)
Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 37: Training loss and standard NLP benchmarks average accuracy of scratch, Gstack and other
partial stacking.

We report the training loss and standard NLP benchmarks average accuracy in Figure 37. By
observing the loss curves in Figure 37a, we can find that the eight partial stacking methods are clearly
divided into three groups based on their loss. The first group, {123456*4, 12-3456*5-56, 12-345*7-6,
123-456*7}, achieves the best performance. The second group consisting of {1234-56*10, 12-34*10-
56, 1-234*7-56}, performs just so-so. The third group, {123*7-456}, performs poorly, even worse
than the baseline.

In Table 5, we summarize the eight partial stacking and calculate the Rc of each partial stacking
methods based on Equation 51.

For partial stacking, we conclude that: all > middle ≈back ≫front. Meanwhile, when the stacked
parts are the same, the larger the Rc, the better the performance.

Table 5: Rc and stacked parts of each partial stacking method

Group
Method
Stacked parts
Rc

First

123456*4
all
87.0%
12-3456*5-56
middle-back
78.3%
12-345*7-6
middle-back
74.0%
123-456*7
back
74.0%

Second
1234-56*10
back
60.7%
12-34*10-56
middle
60.7%
1-234*7-56
front-middle
74.0%
Third
123*7-456
front
74.0%

Then, we report the evaluation results here.

38


---Page Break---
2
4
6
8
FLOPs (1e+20)

23

24

25

26

27

28

29

30

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)

Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

2
4
6
8
FLOPs (1e+20)

42

44

46

48

50

52

54

56

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)

Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

2
4
6
8
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch
Gstack(123456 * 4)

Gstack(123 * 7
456)

Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)

Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

2
4
6
8
FLOPs (1e+20)

25

26

27

28

29

30

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)

Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

2
4
6
8
FLOPs (1e+20)

64

66

68

70

72

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)
Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

2
4
6
8
FLOPs (1e+20)

66

69

72

75

78

81

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)
Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

2
4
6
8
FLOPs (1e+20)

50

51

52

53

54

55

56

57

Accuracy

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

2
4
6
8
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch
Gstack(123456 * 4)
Gstack(123 * 7
456)

Gstack(1
234 * 7
56)
Gstack(12
345 * 7
6)

Gstack(123
456 * 7)

Gstack(12
3456 * 5
56)
Gstack(12
34 * 10
56)

Gstack(1234
56 * 10)

20
40
60
80
Tokens (Billions)

5.5
6.0
6.5
7.0

17.0

17.5

18.0

18.5

70
80

(h) Wikitext (ppl ↓)

Figure 38: Evaluation results on scratch, Gstack and other partial stacking.

H.4
Compare with Pythia, OLMo and Amber on 7B Size

Table 6: Compare with opensource 7B LLMs on 130B tokens.

Pythia-6.9B
OLMo-7B [53]
Amber-7B [54]
Gstack-7B
Datasets
Pile-300B [52]
Dolma [55]
Amber
Slimpajama-627B
Tokens
130B
133B
132B
130B
ARC-c
33.28
28.58
29.01
35.24
ARC-e
59.81
51.60
55.05
63.64
boolq
63.39
55.05
60.18
66.45
hellaswag
60.03
54.52
61.21
65.85
lambada
65.11
49.91
57.13
57.93
logiqa
28.88
28.42
26.73
26.88
obqa
37.20
33.60
37.40
36.40
piqa
75.03
74.43
76.01
76.82
sciq
82.7
74.4
82.0
85.9
winogrande
60.14
53.75
56.83
62.75
Avg.
56.56
50.43
54.16
57.79
Wikitext
13.3340
18.4690
15.6202
12.5635

I
Details of Function Preserving

I.1
Function Preserving

Function preservation is a key concept that underlies diverse model growth approaches. It entails
ensuring consistent output from a model, regardless of its expansion. Mathematically, let us define a
function as F and a growth operator as G. The ultimate aim is to apply the operator G to the function
F, thereby obtaining the target function denoted as F. The core objective here is to maintain the
model’s function to generate the same output for a given input. Formally,

∀x, F(x) = F(x), where F = G(F)
(52)

39


---Page Break---
I.2
Breaking Function Preserving by Adding Noise

For the down projection in SwiGLU and the output projection in MultiHeadAttention, we apply
noise:

Wnoise ←(1 −α)W + αϵ
where ϵ ∼N(0,
1
d × l2 )
(53)

For the Embedding Layer and other Linear Layers, we apply noise:

Wnoise ←(1 −α)W + αϵ
where ϵ ∼N(0, 2

5d)
(54)

Adding Noise on Gdirect to Break FP

0
1
2
3
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

3.2

3.4

Training Loss

scratch
Gdirect(0%noise)

Gdirect(20%noise)

0
10
20
30
40
Tokens (Billions)

2.50
2.75
3.00

2.30

2.35

2.40

2.45

35
40

(a) Training Loss

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

43

44

45

46

47

48

49

Average Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(b) Average Accuracy

Figure 39: Training loss and standard NLP benchmarks average accuracy of scratch, G→
direct and
G→
direct with 20% noise.

40


---Page Break---
0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

23

24

25

26

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(a) ARC-c (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

41

42

43

44

45

46

47

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(b) ARC-e (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

30

32

34

36

38

40

42

44

46

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(c) Lambada (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

25

26

27

28

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(d) Logiqa (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

65

66

67

68

69

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(e) PIQA (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

66

68

70

72

74

76

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(f) Sciq (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

52

53

54

Accuracy

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

(g) Winogrande (Acc ↑)

0.5
1.0
1.5
2.0
2.5
3.0
3.5
FLOPs (1e+20)

20

22

24

26

28

Word Perplexity

scratch
Gdirect(0%noise)

Gdirect(20%noise)

10
20
30
40
Tokens (Billions)

2.6
2.8
3.0
3.2

19.5

20.0

20.5

35
40

(h) Wikitext (ppl ↓)

Figure 40: Evaluation results on scratch, G→
direct and G→
direct with 20% noise.

Training Loss And Evaluation Results on Adding Noise G→
direct

Adding Noise on Gstack
Since adding noise actually improve the Gdirect performance, we also add
noise on Gstack.

We stack an 8 layers small model to 24 layers, and then add noise with α = 0.2. We report training
loss and standard NLP benchmarks average accuracy in Figure 41. Adding noise demonstrates an
advantage in Training loss.

0
1
2
3
4
5
6
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

Training Loss

scratch
Gstack(0%noise)

Gstack(20%noise)

0
20
40
60
80
Tokens (Billions)

4.5
5.0
5.5
2.15

2.20

2.25

2.30

60
70
80

(a) Training Loss

1
2
3
4
5
6
FLOPs (1e+20)

44

46

48

50

Average Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(b) Average Accuracy

Figure 41: Training loss and standard NLP benchmarks average accuracy of scratch, Gstack and Gstack
with 20% noise.

Details of the evaluation results are as follows:

41


---Page Break---
1
2
3
4
5
6
FLOPs (1e+20)

23

24

25

26

27

28

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(a) ARC-c (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

42

44

46

48

50

52

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(b) ARC-e (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

30

33

36

39

42

45

48

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(c) Lambada (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

25

26

27

28

29

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(d) Logiqa (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

65

66

67

68

69

70

71

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(e) PIQA (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

66

68

70

72

74

76

78

80

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(f) Sciq (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

50

51

52

53

54

55

56

Accuracy

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

(g) Winogrande (Acc ↑)

1
2
3
4
5
6
FLOPs (1e+20)

18

20

22

24

26

28

Word Perplexity

scratch
Gstack(0%noise)

Gstack(20%noise)

20
40
60
80
Tokens (Billions)

4.5
5.0
5.5

18.0

18.5

19.0

70
80

(h) Wikitext (ppl ↓)

Figure 42: Evaluation results on scratch, Gstack and Gstack with 20% noise.

J
Results on Samba

We utilize the codebase from Samba9, which implements a hybrid State Space Model using the
Slimpajama dataset for LM. In this experiment, we follow the guidelines outlined in the main paper
to guide our stacking process. With a parameter size of 410M and training on 100B tokens, we set
the growth timing to 8B and the growth factor to 3. We opted for 3 instead of 4 because Samba is an
interleaving of Mamba and self-attention layers. Since the target model has 12 layers, we can only
stack even layers, leading us to select a 4-layer base model (Mamba-SA-Mamba-SA).

Our experiments results on loss curves 43 and downstream tasks 7 indicate stacking also works
beyond Transformer-based LLMs. Please note that in Table 7, we select stack with 47B rather than
50B to count the additional consumption required to train the base model on 8B tokens.

0.0
0.5
1.0
1.5
2.0
2.5
3.0
FLOPs (1e+20)

2.4

2.6

2.8

3.0

3.2

3.4

Training Loss

scratch
Gstack

0
20
40
60
80
100
Tokens (Billions)

1.0
1.5
2.0
2.5

2.4

2.5

61.7%
0.0%
61.5%
0.0%
58.2%
0.0%

40
60
80
100

Figure 43: The training loss for two Samba LLMs, trained from scratch and with Gstack. At loss=2.48,
2.45, 2.42, Gstack accelerates by 61.7%, 61.5% and 58.2% compared to scratch.

9https://github.com/microsoft/Samba

42


---Page Break---
Table 7: Evaluation Results on Samba LLMs

Method
Tokens
lambada
arc-c
arc-e
logiqa
piqa
sciq
avg
scratch
50B
36.41
25.34
43.77
27.50
67.36
70.00
45.06
Gstack
47B
38.44
26.19
44.95
26.88
67.95
72.80
46.20

K
Loss Spikes

Figure 44 illustrates the loss spikes that occur right after stacking.

0
1
2
3
4
FLOPs (1e+20)

2.2

2.4

2.6

2.8

3.0

3.2

3.4

3.6

3.8

Training Loss

scratch
Gstack

Grandom
base model

0
10
20
30
40
Tokens (Billions)

Figure 44: Loss Spikes in Gstack (Non-FP) and G↑
random (FP)

L
Societal Impacts

As a successful exploration for efficient LLM pre-training, our work has great potential to give
positive societal impact towards sustainable AI. Nevertheless, as a common drawback for LLMs,
there are also chances that our LLMs might be misused intentionally or uniintentionally.

43


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: In the Abstract, we clearly elucidate our contributions, and at the end of
Section 1 Introduction, we further detail our contributions and scope.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In Section 7, we discuss the limitations of our work.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

44


---Page Break---
Answer: [NA]

Justification: Our study is empirical exploration.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We report our detailed training settings in Appendix B.3.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

45


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We use open-source dataset Slimpajama-627B for pre-training, we report this
in Appendix B.3. We have submitted our code on OpenReview and will open-source it on
GitHub in the final version.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: We report the detailed settings in Appendix B.3

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [No]

Justification: LLMs pre-training consumes a significant amount of computational resources,
making it impractical to conduct multiple experiments to obtain error bars.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

46


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: We report the needed Compute Resources in Appendix B.3 and required
FLOPs of each experiments in each Figures.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have read this code.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

Justification: We have a section in the Appendix L to discuss societal impacts.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.

47


---Page Break---
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [No]

Justification: Our study is an empirical exploration. The dataset we use is a open-source
high-quality corpus, and the models we release are intended solely for further research and
are not meant for direct industrial application.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]

Justification: Please refer to Appendix B.3.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

48


---Page Break---
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has
curated licenses for some datasets. Their licensing guide can help determine the license
of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [Yes]

Justification: All codes and models are will be full released under the license of CC-BY 4.0.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

Justification: This work does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?

Answer: [NA]

Justification: This work does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

49


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

50


---Page Break---
