Linear Mode Connectivity in
Differentiable Tree Ensembles

Anonymous Author(s)
Affiliation
Address
email

Abstract

Linear Mode Connectivity (LMC) refers to the phenomenon that performance
1

remains consistent for linearly interpolated models in the parameter space. For
2

independently optimized model pairs from different random initializations, achiev-
3

ing LMC is considered crucial for validating the stable success of the non-convex
4

optimization in modern machine learning models and for facilitating practical
5

parameter-based operations such as model merging. While LMC has been achieved
6

for neural networks by considering the permutation invariance of neurons in each
7

hidden layer, its attainment for other models remains an open question. In this
8

paper, we first achieve LMC for soft tree ensembles, which are tree-based differen-
9

tiable models extensively used in practice. We show the necessity of incorporating
10

two invariances: subtree flip invariance and splitting order invariance, which do
11

not exist in neural networks but are inherent to tree architectures, in addition to
12

permutation invariance of trees. Moreover, we demonstrate that it is even possible
13

to exclude such additional invariances while keeping LMC by designing decision
14

list-based tree architectures, where such invariances do not exist by definition. Our
15

findings indicate the significance of accounting for architecture-specific invariances
16

in achieving LMC.
17

1
Introduction
18

A non-trivial empirical characteristic of modern machine learning models trained using gradient
19

methods is that models trained from different random initializations could become functionally
20

almost equivalent, even though their parameter representations differ. If the outcomes of all training
21

sessions converge to the same local minima, this empirical phenomenon can be understood. However,
22

considering the complex non-convex nature of the loss surface, the optimization results are unlikely to
23

converge to the same local minima. In recent years, particularly within the context of neural networks,
24

the transformation of model parameters while preserving functional equivalence has been explored by
25

considering the permutation invariance of neurons in each hidden layer [1, 2]. Notably, only a slight
26

performance degradation has been observed when using weights derived through linear interpolation
27

between permuted parameters obtained from different training processes [3, 4]. This demonstrates
28

that the trained models reside in different, yet functionally equivalent, local minima. This situation is
29

referred to as Linear Mode Connectivity (LMC) [5]. From a theoretical perspective, LMC is crucial
30

for supporting the stable and successful application of non-convex optimization. In addition, LMC
31

also holds significant practical importance, enabling techniques such as model merging [6, 7] by
32

weight-space parameter averaging.
33

Although neural networks are most extensively studied among the models trained using gradient
34

methods, other models also thrive in real-world applications. A representative is tree ensemble models,
35

such as random forests [8]. While they are originally trained by not gradient but greedy algorithms,
36

differentiable soft tree ensembles, which learn parameters of the entire model through gradient-based
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
optimization, have recently been actively studied. Not only empirical studies regarding accuracy
38

and interpretability [9–11], but also theoretical analyses have been performed [12, 13]. Moreover,
39

the differentiability of soft trees allows for integration with various deep learning methodologies,
40

including fine-tuning [14], dropout [15], and various stochastic gradient descent methods [16, 17].
41

Furthermore, the soft tree represents the most elementary form of a hierarchical mixture of experts [18–
42

20]. Investigating soft tree models not only advances our understanding of this particular structure
43

but also contributes to broader research into essential technological components critical for the
44

development of large-scale language models [21].
45

Only Permutation

Ours

Target

Origin

Figure 1: A representative experimental result on
the MiniBooNE [22] dataset (left) and conceptual
diagram of the LMC for tree ensembles (right).

A research question that we tackle in this paper
46

is: “Can LMC be achieved for soft tree ensem-
47

bles?”. Our empirical results, which are high-
48

lighted with a green line in the top left panel
49

of Figure 1, clearly show that the answer is
50

“Yes”. This plot shows the variation in test accu-
51

racy when interpolating weights of soft oblivi-
52

ous trees, perfect binary soft trees with shared
53

parameters at each depth, trained from differ-
54

ent random initializations. The green line is
55

obtained by our method introduced in this pa-
56

per, where there is almost zero performance
57

degradation. Furthermore, as shown in the bot-
58

tom left panel of Figure 1, the performance can
59

even improve when interpolating between mod-
60

els trained on split datasets.
61

The key insight is that, when performing interpolation between two model parameters, considering
62

only tree permutation invariance, which corresponds to the permutation invariance of neural networks,
63

is not sufficient to achieve LMC, as shown in the orange lines in the plots. An intuitive understanding
64

of this situation is also illustrated in the right panel of Figure 1. To achieve LMC, that is, the green
65

lines, we show that two additional invariances beyond tree permutation, subtree flip invariance and
66

splitting order invariance, which inherently exist for tree architectures, should be accounted for.
67

Moreover, we demonstrate that it is possible to exclude such additional invariances while preserving
68

LMC by modifying tree architectures. We realize such an architecture based on a decision list, a
69

binary tree structure where branches extend in only one direction. By designating one of the terminal
70

leaves as an empty node, we introduce a customized decision list that omits both subtree flip invariance
71

and splitting order invariance, and empirically show that this can achieve LMC by considering only
72

tree permutation invariance. Since incorporating additional invariances is computationally expensive,
73

we can efficiently perform weight-space averaging in model merging on our customized decision
74

lists.
75

Our contributions are summarized as follows:
76

• First achievement of LMC for tree ensembles with accounting for additional invariances beyond
77

tree permutation.
78

• Development of a decision list-based tree architecture that does not involve the additional invari-
79

ances.
80

• A thorough empirical investigation of LMC across various tree architectures, invariances, and
81

real-world datasets.
82

2
Preliminary
83

We prepare the basic concepts of LMC and soft tree ensembles.
84

2.1
Linear Mode Connectivity
85

Let us consider two models, A and B, that have the same architecture. In the context of evaluating
86

LMC, the concept of a “barrier” is frequently used [4, 23]. Let ΘA, ΘB ∈RP be vectorized
87

parameters of models A and B, respectively, for P parameters. Assume that C : RP →R measures
88

the performance of the model, such as accuracy, given its parameter vector. If higher values of C(·)
89

2


---Page Break---
mean better performance, the barrier between two parameter vectors ΘA and ΘB is defined as:
90

B(ΘA, ΘB) = sup
λ∈[0,1]
[ λC(ΘA) + (1 −λ)C(ΘB) −C(λΘA + (1 −λ)ΘB) ] .
(1)

We can simply reverse the subtraction order if lower values of C(·) mean better performance like loss.
91

Several techniques have been developed to reduce barriers by transforming parameters while pre-
92

serving functional equivalence. Two main approaches are activation matching (AM) and weight
93

matching (WM). AM takes the behavior of model inference into account, while WM simply com-
94

pares two models using their parameters. The validity of both AM and WM has been theoretically
95

supported [24]. Numerous algorithms are available for implementing AM and WM. For instance, [4]
96

uses a formulation based on the Linear Assignment Problem (LAP) to find suitable permutations,
97

while [23] employs a differentiable formulation that allows for the optimization of permutations using
98

gradient-based methods.
99

Existing research has focused exclusively on neural network architectures such as multi-layer per-
100

ceptrons (MLP) and convolutional neural networks (CNN). No study has been conducted from the
101

perspective of linear mode connectivity for soft tree ensembles.
102

2.2
Soft Tree Ensemble
103

Unlike typical hard decision trees, which explicitly determine the data flow to the right or left at each
104

splitting node, soft trees represent the proportion of data flowing to the right or left as continuous
105

values between 0 and 1. This approach enables a differentiable formulation.
106

We use a sigmoid function, σ : R →(0, 1) to formulate a function µm,ℓ(xi, wm, bm) : RF ×
107

RF ×N × R1×N →(0, 1) that represents the proportion of the ith data point xi flowing to the ℓth
108

leaf of the mth tree as a result of soft splittings:
109

µm,ℓ(xi, wm, bm)=

N
Y

n=1
σ(w⊤
m,nxi + bm,n)
|
{z
}
flow to the left

1ℓ↙n(1 −σ(w⊤
m,nxi + bm,n))
|
{z
}
flow to the right

1n↘ℓ,
(2)

where N denotes the number of splitting nodes in each tree. The parameters wm,n ∈RF and
110

bm,n ∈R correspond to the feature selection mask and splitting threshold value for nth node in a
111

mth tree, respectively. The expression 1ℓ↙n (resp. 1n↘ℓ) is an indicator function that returns 1 if the
112

ℓth leaf is positioned to the left (resp. right) of a node n, and 0 otherwise.
113

If parameters are shared across all splitting nodes at the same depth, such perfect binary trees are
114

called oblivious trees. Mathematically, wm,n = wm,n′ and bm,n = bm,n′ for any nodes n and n′ at
115

the same depth in an oblivious tree. Oblivious trees can significantly reduce the number of parameters
116

from an exponential to a linear order of the tree depth, and they are actively used in practice [9, 11].
117

To classify C categories, the output of the mth tree is computed by the function fm : RF ×
118

RF ×N × R1×N × RC×L →RC as sum of the leaf parameters πm,ℓweighted by the outputs of
119

µm,ℓ(xi, wm, bm):
120

fm(xi, wm, bm, πm) =

L
X

ℓ=1
πm,ℓµm,ℓ(xi, wm, bm),
(3)

where L is the number of leaves in a tree. By combining this function for M trees, we realize the
121

function f : RF × RM×F ×N × RM×1×N × RM×C×L →RC as an ensemble model consisting of
122

M trees:
123

f(xi, w, b, π) =

M
X

m=1
fm(xi, wm, bm, πm),
(4)

with the parameters w = (w1, . . . , wM), b = (b1, . . . , bM), and π = (π1, . . . , πM) being ran-
124

domly initialized.
125

3


---Page Break---
(a)
(b)

Reordering

Leaf Swap
Subtree Flip

Inequality Sign Flip

Figure 2: (a) Subtree flip invariance. (b) Splitting order invariance for an oblivious tree.

Despite the apparent differences, there are correspondences between MLPs and soft tree ensemble
126

models. The formulation of a soft tree ensemble with D = 1 is:
127

f(xi, w, b, π) =

M
X

m=1


σ(w⊤
m,1xi + bm,1)πm,1 + (1 −σ(w⊤
m,1xi + bm,1))πm,2


=

M
X

m=1


(πm,1 −πm,2)σ(w⊤
m,1xi + bm,1) + πm,2

.
(5)

When we consider the correspondence between πm,1 −πm,2 in tree ensembles and second layer
128

weights in the two-layer perceptron, the tree ensembles model matches to the two-layer perceptron. It
129

is clear from the formulation that the permutation of hidden neurons in a neural network corresponds
130

to the rearrangement of trees in a tree ensemble.
131

3
Invariances Inherent to Tree Ensembles
132

In this section, we discuss additional invariances inherent to trees (Section 3.1) and introduce a
133

matching strategy specifically for tree ensembles (Section 3.2). We also show that the presence of
134

additional invariances varies depending on the tree structure, and we present tree structures where no
135

additional invariances beyond tree permutation exist (Section 3.3).
136

3.1
Parameter modification processes that maintains functional equivalence in tree ensembles
137

First, we clarify what invariances should be considered for tree ensembles, which are expected to
138

reduce the barrier significantly if taken into account. When we consider perfect binary trees, there are
139

three types of invariance:
140

• Tree permutation invariance. In Equation (4), the behavior of the function does not change even
141

if the order of the M trees is altered. This corresponds to the permutation of internal nodes in
142

neural networks, which has been a subject of active interest in previous studies on LMC.
143

• Subtree flip invariance. When the left and right subtrees are swapped simultaneously with the
144

inversion of the inequality sign at the split, the functional behavior remains unchanged, which we
145

refer to subtree flip invariance. Figure 2(a) presents a schematic diagram of this invariance, which
146

is not found in neural networks but is unique to binary tree-based models. Since σ(−c) = 1 −σ(c)
147

for c ∈R due to the symmetry of sigmoid, the inversion of the inequality is achieved by inverting
148

the signs of wm,n and bm,n. [25] also focused on the sign of weights, but in a different way from
149

ours. They pay attention to the amount of change from the parameters at the start of fine-tuning,
150

rather than discussing the sign of the parameters.
151

• Splitting order invariance. Oblivious trees share parameters at the same depth, which means
152

that the decision boundaries are straight lines without any bends. With this characteristic, even if
153

the splitting rules at different depths are swapped, functional equivalence can be achieved if the
154

positions of leaves are also swapped appropriately as shown in Figure 2(b). This invariance does
155

not exist for non-oblivious perfect binary trees without parameter sharing, as the behavior of the
156

decision boundary varies depending on the splitting order.
157

4


---Page Break---
Note that MLPs also have an additional invariance beyond just permutation. Particularly in MLPs
158

that employ ReLU as an activation function, the output of each layer changes linearly with a zero
159

crossover. Therefore, it is possible to modify parameters without changing functional behavior by
160

multiplying the weights in one layer by a constant and dividing the weights in the previous layer by
161

the same constant. However, since the soft tree is based on the sigmoid function, this invariance does
162

not apply. Previous studies [3, 4, 23] have consistently achieved significant reductions in barriers
163

without accounting for this scale invariance. One potential reason is that changes in parameter scale
164

are unlikely due to the nature of optimization via gradient descent. Conversely, when we consider
165

additional invariances inherent to trees, the scale is equivalent to the original parameters.
166

3.2
Matching Strategy
167

8

4
4

2
2
2
2

4

3

2

Parameter Sharing
Parameter Sharing

Figure 3: Weighting strategy.

Here, we propose a matching strategy for bi-
168

nary trees. When considering invariances, it
169

is necessary to compare multiple functionally
170

equivalent trees and select the most suitable one
171

for achieving LMC. Although comparing tree
172

parameters is a straightforward approach, since
173

the contribution of all the parameters in a tree is
174

not equal, we apply weighting for each node for
175

better matching. By interpreting a tree as a rule
176

set with shared parameters as shown in Figure 3,
177

we determine the weight of each splitting node
178

by counting the number of leaves to which the node affects. For example, in the case of the left
179

example in Figure 3, the root node affects eight leaves, nodes at depth 2 affect four leaves, and nodes
180

at depth 3 affect two leaves. This strategy can apply to even trees other than perfect binary trees. For
181

example, in the right example of Figure 3, the root node affects four leaves, a node at depth 2 affects
182

three leaves, and a node at depth 3 affects two leaves.
183

In this paper, we employ the LAP, which is used as a standard benchmark [4] for matching algorithms.
184

The procedures for AM and WM are as follows. Detailed algorithms (Algorithms 1 and 2) are
185

described in Section A in the supplementary material.
186

• Activation Matching (Algorithm 1). In trees, there is nothing that directly corresponds to the
187

activations in neural networks. However, by treating the output of each individual tree as an
188

activation value of a neural network, it is possible to optimize the permutation of trees while
189

examining their output similarities. Regarding subtree flip and splitting order invariances, it is
190

possible to find the optimal pattern from all the possible patterns of flips and changes in the splitting
191

order. Since the tree-wise output remains unchanged, the similarity between each tree, generated
192

by considering additional invariances, and the target tree is evaluated based on the inner product of
193

parameters while applying node-wise weighting.
194

• Weight Matching (Algorithm 2). Similar to AM, WM also involves applying weighting while
195

extracting the optimal pattern by exploring possible flipping and ordering patterns. Although it is
196

necessary to solve the LAP multiple times for each layer in MLPs [4], tree ensembles require only
197

a single run of the LAP since there are no layers.
198

The time complexity of solving the LAP is O(M 3) using a modified Jonker-Volgenant algorithm
199

without initialization [26], implemented in SciPy [27], where M is the number of trees. If only
200

considering tree permutation, this process needs to be performed only once in both WM and AM.
201

However, when considering additional invariances, we need to solve the LAP for each pattern
202

generated by considering these additional invariances. In a non-oblivious perfect binary tree with
203

depth D, there are 2D −1 splitting nodes, leading to 22D−1 possible combinations of sign flips.
204

Additionally, in the case of oblivious trees, there are D! different patterns of splitting order invariance.
205

Therefore, for large values of D, conducting a brute-force search becomes impractical.
206

In Section 3.3, we will discuss methods to eliminate additional invariance by adjusting the tree
207

structure. This enables efficient matching even for deep models. Additionally, in Section 4.2, we
208

will present numerical experiment results and discuss that the practical motivation to apply these
209

algorithms is limited when targeting deep perfect binary trees.
210

5


---Page Break---
3.3
Architecture-dependency of the Invariances
211

Invariance exists
Empty Node

Figure 4: Tree architecture where neither subtree
flip invariance nor splitting order invariance exists.

In previous subsections, tree architectures are
212

fixed to perfect binary trees as they are most
213

commonly and practically used in soft trees.
214

However, tree architectures can be flexible as
215

we have shown in the right example in Figure 3,
216

and here we show that we can specifically de-
217

sign tree architecture that has neither the subtree
218

flip nor splitting order invariances. This allows
219

efficient matching as considering such two in-
220

variances is computationally expensive.
221

Table 1: Invariances inherent to each model archi-
tecture.

Perm
Flip
Order

Non-Oblivious Tree
✓
✓
×
Oblivious Tree
✓
✓
✓
Decision List
✓
(✓)
×
Decision List (Modified)
✓
×
×

Our idea is to modify a decision list shown on
222

the left side of Figure 4, which is a tree structure
223

where branches extend in only one direction.
224

Due to this asymmetric structure, the number of
225

parameters does not increase exponentially with
226

the depth, and the splitting order invariance does
227

not exist. Moreover, subtree flip invariance also
228

does not exist for any internal nodes except for
229

the terminal splitting node, as shown in the left
230

side of Figure 4. To completely remove this invariance, we virtually eliminate one of the terminal
231

leaves by leaving the node empty, that is, a fixed prediction value of zero, as shown on the right
232

side of Figure 4. Therefore only permutation invariance exists for our proposed architecture. We
233

summarize invariances inherent to each model architecture in Table 1.
234

4
Experiment
235

We empirically evaluate barriers in soft tree ensembles to examine LMC.
236

4.1
Setup
237

Datasets.
In our experiments, we employed Tabular-Benchmark [28], a collection of tabular
238

datasets suitable for evaluating tree ensembles. Details of datasets are provided in Section B in the
239

supplementary material. As proposed in [28], we randomly sampled 10, 000 instances for train and
240

test data from each dataset. If the dataset contains fewer than 20, 000 instances, they are randomly
241

divided into halves for train and test data. We applied quantile transformation to each feature and
242

standardized it to follow a normal distribution.
243

Hyperparameters. We used three different learning rates η ∈{0.01, 0.001, 0.0001} and adopted the
244

one that yields the highest training accuracy for each dataset. The batch size is set at 512. It is known
245

that the optimal settings for the learning rate and batch size are interdependent [29]. Therefore, it is
246

reasonable to fix the batch size while adjusting the learning rate. During AM, we set the amount of
247

data used for random sampling to be the same as the batch size, thus using 512 samples to measure the
248

similarity of the tree outputs. As the number of trees M and their depths D vary for each experiment,
249

these details will be specified in the experimental results section. During training, we minimized
250

cross-entropy using Adam [16] with its default hyperparameters1. Training is conducted for 50
251

epochs. To measure the barrier using Equation (1), experiments were conducted by interpolating
252

between two models with λ ∈{0, 1/24, . . . , 23/24, 1}, which has the same granularity as in [4].
253

Randomness. We conducted experiments with five different random seed pairs: rA ∈{1, 3, 5, 7, 9}
254

and rB ∈{2, 4, 6, 8, 10}. As a result, the initial parameters and the contents of the data mini-batches
255

during training are different in each training. In contrast to spawning [5] that branches off from the
256

exact same model partway through, we used more challenging practical conditions. The parameters
257

w, b, and π were randomly initialized using a uniform distribution, identical to the procedure for a
258

fully connected layer in the MLP2.
259

1https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
2https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

6


---Page Break---
Naive

Perm

Perm&Flip

0

5

10

15

20

Averaged Accuracy Barrier (Non-Oblivious)

M=256 (Train)

Naive

Perm

Perm&Flip

0

5

10

15

20
D=2 (Train)

Naive
Perm

Perm&OrderPerm&Flip

Perm&Flip&Order

0

5

10

15

20

Averaged Accuracy Barrier (Oblivious)

Naive
Perm

Perm&OrderPerm&Flip

Perm&Flip&Order

0

5

10

15

20

Naive

Perm

Perm&Flip

0

5

10

15

20
M=256 (Test)

Naive

Perm

Perm&Flip

0

5

10

15

20
D=2 (Test)

Naive
Perm

Perm&OrderPerm&Flip

Perm&Flip&Order

0

5

10

15

20

D=1, WM
D=2, WM
D=3, WM

D=1, AM
D=2, AM
D=3, AM

Naive
Perm

Perm&OrderPerm&Flip

Perm&Flip&Order

0

5

10

15

20

M=64, WM
M=256, WM
M=1024, WM

M=64, AM
M=256, AM
M=1024, AM

Figure 5: Barriers averaged across 16 datasets with respect to considered invariances for non-
oblivious (top row) and oblivious (bottom row) trees. The error bars show the standard deviations of
5 executions.

Interpolation

70

75

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation
55

60

65

Higgs

Interpolation

80

85

MagicTelescope

Interpolation

60

80

MiniBooNE

Interpolation

60

70

bank-marketing

Interpolation

75

80

85

california

Interpolation
50

60

70

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation

60

70

80

electricity

Interpolation

54

56

58

eye_movements

Interpolation

65

70

heloc

Interpolation

75

80

85

house_16H

Interpolation

65

70

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 6: Interpolation curves of test accuracy for oblivious trees on 16 datasets from Tabular-
Benchmark [28]. Two model pairs are trained with on the same dataset. The error bars show the
standard deviations of 5 executions. We used M = 256 trees with a depth D = 2.

Resources. All experiments were conducted on a system equipped with an Intel Xeon E5-2698 CPU
260

at 2.20 GHz, 252 GB of memory, and Tesla V100-DGXS-32GB GPU, running Ubuntu Linux (version
261

4.15.0-117-generic). The reproducible PyTorch [30] implementation is provided in the supplementary
262

material.
263

4.2
Results for Perfect Binary Trees
264

Figure 5 shows how the barrier between two perfect binary tree model pairs changes in each operation.
265

The vertical axis of each plot in Figure 5 shows the averaged barrier over datasets for each considered
266

invariance. The results for both the oblivious and non-oblivious trees are plotted separately in a
267

vertical layout. The panels on the left display the results when the depth D of the tree varies, keeping
268

M = 256 constant. The panels on the right show the results when the number of trees M varies, with
269

D fixed at 2. For both oblivious and non-oblivious trees, we observed that the barrier significantly
270

decreases as the considered invariances increase. Focusing on the test data results, after accounting for
271

various invariances, the barrier is nearly zero, indicating that LMC has been achieved. In particular,
272

the difference between the case of only permutation and the case where additional invariances are
273

considered tends to be larger in the case of AM. This is because parameter values are not used during
274

the rearrangement of the tree in AM. Additionally, it has been observed that the barrier increases as
275

trees become deeper, and the barrier decreases as the number of trees increases. These behaviors
276

correspond to the changes observed in neural networks when the depth varies or when the width of
277

hidden layers increases [3, 4]. Figure 6 shows interpolation curves when using AM in oblivious trees
278

7


---Page Break---
Interpolation

60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

75

80

85MagicTelescope

Interpolation

70

80

90

MiniBooNE

Interpolation

70

75

bank-marketing

Interpolation
75

80

85

california

Interpolation
50

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation
65

70

75

electricity

Interpolation

52.5

55.0

57.5

eye_movements

Interpolation

60

70

heloc

Interpolation

80

85

house_16H

Interpolation

65

70

jannis

Interpolation

70

80

90

pol

Naive
Tree Permutation
Ours

Figure 7: Interpolation curves of test accuracy for oblivious trees on 16 datasets from Tabular-
Benchmark [28]. Two model pairs are trained on split datasets with different class ratios. The error
bars show the standard deviations of 5 executions. We used M = 256 trees with a depth D = 2.

with D = 2 and M = 256. Other detailed results, such as performance for each dataset, are provided
279

in Section C in the supplementary material.
280

Furthermore, we conducted experiments with split data following the protocol in [4, 31], where
281

the initial split consists of randomly sampled 80% negative and 20% positive instances, and the
282

second split inverts these ratios. There is no overlap between the two split datasets. We trained two
283

model pairs using these separately split datasets and observed an improvement in performance by
284

interpolating their parameters. Figure 7 illustrates the interpolation curves under AM in oblivious
285

trees with parameters D = 2 and M = 256. We can observe that considering additional invariances
286

improves performance after interpolation. Note that the data split is configured to remain consistent
287

even when the training random seeds differ. Detailed results for each dataset using WM or AM are
288

provided in Section C of the supplementary material.
289

Table 2: Barriers, accuracies, and model sizes for
MLP, non-oblivious trees, and oblivious trees.

MLP

Barrier
Depth

Naive
Perm [4]
Accuracy
Size

1
8.755 ± 0.877
0.491 ± 0.062 76.286 ± 0.094
12034
2
15.341± 1.125 2.997 ± 0.709 75.981 ± 0.139
77826
3
15.915 ± 2.479 5.940 ± 2.153 75.935 ± 0.117 143618

Non-Oblivious Tree

Barrier
Depth

Naive
Perm
Ours
Accuracy
Size

1
8.965 ± 0.963 0.449 ± 0.235 0.181 ± 0.078 76.464 ± 0.167 12544
2
6.801 ± 0.464 0.811 ± 0.333 0.455 ± 0.105 76.631 ± 0.052 36608
3
5.602 ± 0.926 1.635 ± 0.334 0.740 ± 0.158 76.339 ± 0.115 84736

Oblivious Tree

Barrier
Depth

Naive
Perm
Ours
Accuracy
Size

1
8.965 ± 0.963 0.449 ± 0.235 0.181 ± 0.078 76.464 ± 0.167 12544
2
7.881 ± 0.866 0.918 ± 0.092 0.348 ± 0.172 76.623 ± 0.042 25088
3
7.096 ± 0.856 1.283 ± 0.139 0.484 ± 0.049 76.535 ± 0.063 38656

Table 2 compares the average test barriers of an
290

MLP with a ReLU activation function, whose
291

width is equal to the number of trees, M = 256.
292

The procedure for MLPs follows that described
293

in Section 4.1. The permutation for MLPs is
294

optimized using the method described in [4].
295

Since [4] indicated that WM outperforms AM
296

in neural networks, WM was used for the com-
297

parison. Overall, tree models exhibit smaller
298

barriers compared to MLPs while keeping sim-
299

ilar accuracy levels. It is important to note that
300

MLPs with D > 1 tend to have more parameters
301

at the same depth compared to trees, leading to
302

more complex optimization landscapes. Nev-
303

ertheless, the barrier for the non-oblivious tree
304

at D = 3 is smaller than that for the MLP at
305

D = 2, even with more parameters. Further-
306

more, at the same depth of D = 1, tree models
307

have a smaller barrier. Here, the model size is
308

evaluated using F = 44, the average input fea-
309

ture size of 16 datasets used in the experiments.
310

In Section 3.2, we have shown that considering additional invariances for deep perfect binary trees
311

is computationally challenging, which may suggest developing heuristic algorithms for deep trees.
312

However, we consider it is rather a low priority, supported by our observations that the barrier tends
313

to increase as trees deepen even if we consider invariances. This trend indicates that deep models are
314

fundamentally less important for model merging considerations. Furthermore, deep perfect binary
315

trees are rarely used in practical scenarios. [12] have demonstrated that generalization performance
316

degrades with increasing depth in perfect binary trees due to the degeneracy of the Neural Tangent
317

Kernel (NTK) [32]. This evidence further supports the preference for shallow perfect binary trees,
318

and increasing the number of trees can enhance the expressive power while reducing barriers.
319

8


---Page Break---
Table 3: Barriers averaged for 16 datasets under WM with D = 2 and M = 256.

Train
Test

Barrier
Barrier
Architecture

Naive
Perm
Ours
Accuracy
Naive
Perm
Ours
Accuracy

Non-Oblivious Tree
13.079 ± 0.755 4.707 ± 0.332 3.303 ± 0.104 85.646 ± 0.090 6.801 ± 0.464 0.811 ± 0.333 0.455 ± 0.105 76.631 ± 0.052
Oblivious Tree
14.580 ± 1.108 4.834 ± 0.176 2.874 ± 0.108 85.808 ± 0.146 7.881 ± 0.866 0.919 ± 0.093 0.348 ± 0.172 76.623 ± 0.042
Decision List
13.835 ± 0.788 3.687 ± 0.230
—
85.337 ± 0.134 7.513 ± 0.944 0.436 ± 0.120
—
76.629 ± 0.119
Decision List (Modified) 12.922 ± 1.131 3.328 ± 0.204
—
85.563 ± 0.141 6.734 ± 1.096 0.468 ± 0.150
—
76.773 ± 0.051

Table 4: Barriers averaged for 16 datasets under AM with D = 2 and M = 256.

Train
Test

Barrier
Barrier
Architecture

Naive
Perm
Ours
Accuracy
Naive
Perm
Ours
Accuracy

Non-Oblivious Tree
13.079 ± 0.755 14.963 ± 1.520 4.500 ± 0.527 85.646 ± 0.090 6.801 ± 0.464
8.631 ± 1.444
0.943 ± 0.435 76.631 ± 0.052
Oblivious Tree
14.580 ± 1.108 17.380 ± 0.509 3.557 ± 0.201 85.808 ± 0.146 7.881 ± 0.866 10.349 ± 0.476 0.395 ± 0.185 76.623 ± 0.042
Decision List
13.835 ± 0.788 12.785 ± 1.924
—
85.337 ± 0.134 7.513 ± 0.944
7.452 ± 1.840
—
76.629 ± 0.119
Decision List (Modified) 12.922 ± 1.131
6.364 ± 0.194
—
85.563 ± 0.141 6.734 ± 1.096
2.114 ± 0.243
—
76.773 ± 0.051

4.3
Results for Decision Lists
320

2
4
8
D

0.0

2.5

5.0

7.5

10.0

Averaged Accuracy Barrier

WM

Oblivious
Decision List
Decision List (Modified)

2
4
8
D

5

10

15

20

AM

Figure 8: Averaged barrier for 16 datasets as a
function of tree depth. The error bars show the
standard deviations of 5 executions. The solid line
represents the barrier in train accuracy, while the
dashed line represents the barrier in test accuracy.

We present empirical results of the original de-
321

cision lists and our modified decision lists, as
322

shown in Figure 4. As we have shown in Table 1,
323

they have fewer invariances.
324

Figure 8 illustrates barriers as a function of
325

depth, considering only permutation invariance,
326

with M fixed at 256. In this experiment, we
327

have excluded non-oblivious trees from compar-
328

ison as the number of their parameters exponen-
329

tially increases as trees deepen, making them
330

infeasible computation. Our proposed modified
331

decision lists reduce the barrier more effectively
332

than both oblivious trees and the original de-
333

cision lists. However, barriers of the modified
334

decision lists are still larger than those obtained by considering additional invariances with perfect
335

binary trees. Tables 3 and 4 show the averaged barriers for 16 datasets, with D = 2 and M = 256.
336

Although barriers of modified decision lists are small when considering only permutations (Perm),
337

perfect binary trees such as oblivious trees with additional invariances (Ours) exhibit smaller barriers,
338

which supports the validity of using oblivious trees as in [9, 11]. To summarize, when considering
339

the practical use of model merging, if the goal is to prioritize efficient computation, we recommend
340

using our proposed decision list. Conversely, if the goal is to prioritize barriers, it would be preferable
341

to use perfect binary trees, which have a greater number of invariant operations that maintain the
342

functional behavior.
343

5
Conclusion
344

We have presented the first investigation of LMC for soft tree ensembles. We have identified additional
345

invariances inherent in tree architectures and empirically demonstrated the importance of considering
346

these factors. Achieving LMC is crucial not only for understanding the behavior of non-convex
347

optimization from a learning theory perspective but also for implementing practical techniques such as
348

model merging. By arithmetically combining parameters of differently trained models, a wide range
349

of applications such as task-arithmetic [33], including unlearning [34] and continual-learning [35],
350

have been explored. Our research extends these techniques to soft tree ensembles that began training
351

from entirely different initial conditions. We will leave these empirical investigations for future work.
352

This study provides a fundamental analysis of ensemble learning, and we believe that our discussion
353

will not have any negative societal impacts.
354

9


---Page Break---
References
355

[1] Robert Hecht-Nielsen. On the algebraic structure of feedforward network weight spaces. In
356

Advanced Neural Computers. 1990.
357

[2] An Mei Chen, Haw-minn Lu, and Robert Hecht-Nielsen. On the Geometry of Feedforward
358

Neural Network Error Surfaces. Neural Computation, 1993.
359

[3] Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The Role of Permutation
360

Invariance in Linear Mode Connectivity of Neural Networks. In International Conference on
361

Learning Representations, 2022.
362

[4] Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git Re-Basin: Merging Models
363

modulo Permutation Symmetries. In The Eleventh International Conference on Learning
364

Representations, 2023.
365

[5] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear Mode
366

Connectivity and the Lottery Ticket Hypothesis. In Proceedings of the 37th International
367

Conference on Machine Learning, 2020.
368

[6] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes,
369

Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig
370

Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy
371

without increasing inference time. In Proceedings of the 39th International Conference on
372

Machine Learning, 2022.
373

[7] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task Arithmetic in the
374

Tangent Space: Improved Editing of Pre-Trained Models. In Thirty-seventh Conference on
375

Neural Information Processing Systems, 2023.
376

[8] Leo Breiman. Random Forests. In Machine Learning, 2001.
377

[9] Sergei Popov, Stanislav Morozov, and Artem Babenko. Neural Oblivious Decision Ensembles
378

for Deep Learning on Tabular Data. In International Conference on Learning Representations,
379

2020.
380

[10] Hussein Hazimeh, Natalia Ponomareva, Petros Mol, Zhenyu Tan, and Rahul Mazumder. The
381

Tree Ensemble Layer: Differentiability meets Conditional Computation. In Proceedings of the
382

37th International Conference on Machine Learning, 2020.
383

[11] Chun-Hao Chang, Rich Caruana, and Anna Goldenberg. NODE-GAM: Neural generalized
384

additive model for interpretable deep learning. In International Conference on Learning
385

Representations, 2022.
386

[12] Ryuichi Kanoh and Mahito Sugiyama. A Neural Tangent Kernel Perspective of Infinite Tree
387

Ensembles. In International Conference on Learning Representations, 2022.
388

[13] Ryuichi Kanoh and Mahito Sugiyama. Analyzing Tree Architectures in Ensembles via Neural
389

Tangent Kernel. In International Conference on Learning Representations, 2023.
390

[14] Guolin Ke, Zhenhui Xu, Jia Zhang, Jiang Bian, and Tie-Yan Liu. DeepGBM: A Deep Learning
391

Framework Distilled by GBDT for Online Prediction Tasks. In Proceedings of the 25th ACM
392

SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019.
393

[15] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
394

Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine
395

Learning Research, 2014.
396

[16] Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International
397

Conference on Learning Representations, 2015.
398

[17] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware Mini-
399

mization for Efficiently Improving Generalization. In International Conference on Learning
400

Representations, 2021.
401

10


---Page Break---
[18] M.I. Jordan and R.A. Jacobs. Hierarchical mixtures of experts and the EM algorithm. In
402

Proceedings of International Conference on Neural Networks, 1993.
403

[19] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E.
404

Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-
405

Experts Layer. In International Conference on Learning Representations, 2017.
406

[20] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,
407

Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling Giant Models with
408

Conditional Computation and Automatic Sharding. In International Conference on Learning
409

Representations, 2021.
410

[21] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
411

Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
412

Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut
413

Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B, 2023.
414

[22] Byron Roe. MiniBooNE particle identification. UCI Machine Learning Repository, 2010.
415

[23] Fidel A. Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric
416

Granger, and Marco Pedersoli. Re-basin via implicit Sinkhorn differentiation. In IEEE/CVF
417

Conference on Computer Vision and Pattern Recognition, 2023.
418

[24] Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, and Wei Hu. Going Beyond Linear
419

Mode Connectivity: The Layerwise Linear Feature Connectivity. In Thirty-seventh Conference
420

on Neural Information Processing Systems, 2023.
421

[25] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-merging:
422

Resolving interference when merging models. In Thirty-seventh Conference on Neural Informa-
423

tion Processing Systems, 2023.
424

[26] David F. Crouse. On implementing 2D rectangular assignment algorithms. IEEE Transactions
425

on Aerospace and Electronic Systems, 2016.
426

[27] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David
427

Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J.
428

van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew
429

R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng, Eric W.
430

Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A.
431

Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul
432

van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific
433

Computing in Python. Nature Methods, 2020.
434

[28] Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. Why do tree-based models still
435

outperform deep learning on typical tabular data?
In Thirty-sixth Conference on Neural
436

Information Processing Systems Datasets and Benchmarks Track, 2022.
437

[29] Samuel L. Smith, Pieter-Jan Kindermans, and Quoc V. Le. Don’t Decay the Learning Rate,
438

Increase the Batch Size. In International Conference on Learning Representations, 2018.
439

[30] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
440

Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas
441

Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
442

Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style,
443

High-Performance Deep Learning Library. In Advances in Neural Information Processing
444

Systems, 2019.
445

[31] Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur. REPAIR:
446

REnormalizing permuted activations for interpolation repair. In The Eleventh International
447

Conference on Learning Representations, 2023.
448

[32] Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural Tangent Kernel: Convergence and
449

Generalization in Neural Networks. In Advances in Neural Information Processing Systems,
450

2018.
451

11


---Page Break---
[33] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Ha-
452

jishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International
453

Conference on Learning Representations, 2023.
454

[34] Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Am-
455

manabrolu, and Yejin Choi. QUARK: Controllable text generation with reinforced unlearning.
456

In Advances in Neural Information Processing Systems, 2022.
457

[35] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Dilan Gorur, Razvan Pascanu, and Hassan
458

Ghasemzadeh. Linear Mode Connectivity in Multitask and Continual Learning. In Inter-
459

national Conference on Learning Representations, 2021.
460

12


---Page Break---
A
Detailed Algorithms
461

We present pseudo-code of algorithms for activation matching (Algorithm 1) and weight matching
462

(Algorithm 2). In these algorithms, if there is only one possible pattern for U ∈N, which represents
463

the number of possible operations, and the corresponding operation does nothing in particular, it
464

becomes equivalent to simply considering tree permutations.

Algorithm 1: Activation matching for soft trees

1 ACTIVATIONMATCHING(ΘA ∈RM×PTree, ΘB ∈RM×PTree, xsampled ∈RF ×Nsampled)

2
Initialize OA ∈RM×Nsampled×C and OB ∈RM×Nsampled×C to store outputs

3
for m = 1 to M do

4
for i = 1 to Nsampled do

5
Set the output of the mth tree with ΘA[m] using xsampled[:, i] to OA[m, i].

6
Set the output of the mth tree with ΘB[m] using xsampled[:, i] to OB[m, i].

7
Initialize similarity matrix S ∈RM×M

8
for mA = 1 to M do

9
for mB = 1 to M do

10
S[mA, mB] ←FLATTEN(OA[mA]) · FLATTEN(OB[mB])

11
p ←LINEARSUMASSIGNMENT(S)
// p ∈NM: Optimal assignments

12
ΘA, ΘB ←WEIGHTING(ΘA, ΘB)

13
Initialize operation indices q ∈NM

14
for m = 1 to M do

15
for u = 1 to U do
// U ∈N: Number of possible operations

16
u′ ←UPDATEBESTOPERATION(ADJUSTTREE(ΘA[m], u) · ΘB[m], u)

17
Append u′ to q
// q ∈NM: Optimal operations

18
return p, q

Algorithm 2: Weight matching for soft trees

1 WEIGHTMATCHING(ΘA ∈RM×PTree, ΘB ∈RM×PTree)

2
ΘA, ΘB ←WEIGHTING(ΘA, ΘB)

3
Initialize similarity matrix for each operation S ∈RU×M×M

4
for u = 1 to U do

5
for mA = 1 to M do

6
θ ←ADJUSTTREE(ΘA[mA], u)
// θ ∈RPTree: Adjusted tree-wise parameters

7
for mB = 1 to M do

8
S[u, mA, mB] ←θ · ΘB[mB]

9
S′ ←max(S, axis=0)
// S′ ∈RM×M: Similarity matrix between trees

10
p ←LINEARSUMASSIGNMENT(S′)
// p ∈NM: Optimal assignments

11
q ←argmax(S, axis=0)[p]
// q ∈NM: Optimal operations

12
return p, q

465

Here, we describe the specifications of the notations and functions used in Algorithms 1 and 2. In
466

Section 2.1, ΘA and ΘB are initially defined as vectors. However, for ease of use, in Algorithms 1
467

and 2, ΘA and ΘB are represented as matrices of size RM×PTree, where PTree denotes the number of
468

parameters in a single tree. Multidimensional array elements are accessed using square brackets [·].
469

For example, for G ∈RI×J, G[i] refers to the ith slice along the first dimension, and G[:, j] refers
470

to the jth slice along the second dimension, with sizes RJ and RI, respectively. Furthermore, it can
471

also accept a vector v ∈Nl as an input. In this case, G[v] ∈Rl×J. The FLATTEN function converts
472

multidimensional input into a one-dimensional vector format. As the LINEARSUMASSIGNMENT
473

13


---Page Break---
function, scipy. optimize. linear_sum_assignment3 is used to solve the LAP. In the ADJUSTTREE
474

function, the parameters of a tree are modified according to the uth pattern among the enumerated U
475

patterns. Additionally, in the WEIGHTING function, parameters are multiplied by the square root
476

of their weights defined in Section 3.2 to simulate the process of assessing a rule set. If the first
477

argument for the UPDATEBESTOPERATION function, the input inner product, is larger than any
478

previously input inner product values, then u′ is updated with u, the second argument. If not, u′
479

remains unchanged.
480

B
Dataset
481

Table 5: Summary of the datasets used in the experiments.
Dataset
N
F
Link

Bioresponse
3434
419
https://www.openml.org/d/45019
Diabetes130US
71090
7
https://www.openml.org/d/45022
Higgs
940160
24
https://www.openml.org/d/44129
MagicTelescope
13376
10
https://www.openml.org/d/44125
MiniBooNE
72998
50
https://www.openml.org/d/44128
bank-marketing
10578
7
https://www.openml.org/d/44126
california
20634
8
https://www.openml.org/d/45028
covertype
566602
10
https://www.openml.org/d/44121
credit
16714
10
https://www.openml.org/d/44089
default-of-credit-card-clients
13272
20
https://www.openml.org/d/45020
electricity
38474
7
https://www.openml.org/d/44120
eye_movements
7608
20
https://www.openml.org/d/44130
heloc
10000
22
https://www.openml.org/d/45026
house_16H
13488
16
https://www.openml.org/d/44123
jannis
57580
54
https://www.openml.org/d/45021
pol
10082
26
https://www.openml.org/d/44122

C
Additional Empirical Results
482

Tables 6, 7, 8 and 9 present the barrier for each dataset with D = 2 and M = 256. By incorporating
483

additional invariances, it has been possible to consistently reduce the barriers.
484

Tables 10 and 11 detail the characteristics of the barriers in the decision lists for each dataset with
485

D = 2 and M = 256. The barriers in the modified decision lists tend to be smaller.
486

Tables 12 and 13 show the barrier for each model when only considering permutations with D = 2
487

and M = 256. It is evident that focusing solely on permutations leads to smaller barriers in the
488

modified decision lists compared to other architectures.
489

Figures 9, 10, 11, 12, 13, 14, 15 and 16 show the interpolation curves of oblivious trees with D = 2
490

and M = 256 across various datasets and configurations. Significant improvements are particularly
491

noticeable in AM, but improvements are also observed in WM. These characteristics are also apparent
492

in the non-oblivious trees, as shown in Figures 17, 18, 19, 20, 21, 22, 23 and 24. Regarding split data
493

training, the dataset for each of the two classes is initially complete (100%). It is then divided into
494

splits of 80% and 20%, and 20% and 80%, respectively. Each model is trained using these splits.
495

Figures 13, 15, 21, and 23 show the training accuracy evaluated using the full dataset (100% for each
496

class).
497

3https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_
assignment.html

14


---Page Break---
Table 6: Accuracy barrier for non-oblivious trees with WM.

Train
Test
Dataset
Naive
Perm
Perm&Flip
Naive
Perm
Perm&Flip

Bioresponse
18.944 ± 10.076
5.876 ± 1.477
4.132 ± 0.893
8.235 ± 6.456
1.285 ± 0.635
0.314 ± 0.432
Diabetes130US
2.148 ± 0.601
1.388 ± 1.159
0.947 ± 0.888
1.014 ± 0.959
0.540 ± 0.999
0.784 ± 0.840
Higgs
27.578 ± 1.742
18.470 ± 0.769
14.772 ± 1.419
4.055 ± 1.089
0.662 ± 0.590
0.292 ± 0.421
MagicTelescope
2.995 ± 1.198
0.576 ± 0.556
0.307 ± 0.346
2.096 ± 1.055
0.361 ± 0.618
0.229 ± 0.348
MiniBooNE
18.238 ± 4.570
2.272 ± 0.215
1.506 ± 0.211
12.592 ± 4.190
0.231 ± 0.314
0.000 ± 0.000
bank-marketing
13.999 ± 4.110
2.711 ± 1.183
1.521 ± 0.463
13.593 ± 4.567
1.843 ± 1.001
0.953 ± 0.688
california
6.396 ± 2.472
0.873 ± 0.551
0.520 ± 0.327
5.226 ± 2.377
0.224 ± 0.248
0.206 ± 0.131
covertype
16.823 ± 4.159
1.839 ± 0.336
0.914 ± 0.546
14.900 ± 4.016
1.035 ± 0.106
0.376 ± 0.333
credit
7.317 ± 2.425
3.172 ± 2.636
2.615 ± 0.831
5.861 ± 2.064
2.202 ± 3.103
1.830 ± 0.588
default-of-credit-card-clients
14.318 ± 4.509
5.419 ± 1.318
3.273 ± 0.793
6.227 ± 4.205
0.937 ± 1.036
0.243 ± 0.172
electricity
10.090 ± 2.930
1.035 ± 0.543
0.221 ± 0.192
9.422 ± 2.795
0.771 ± 0.478
0.130 ± 0.071
eye_movements
18.743 ± 1.994
11.605 ± 1.927
7.866 ± 1.301
1.495 ± 0.467
0.463 ± 0.183
0.180 ± 0.206
heloc
4.434 ± 1.611
1.652 ± 0.475
1.012 ± 0.481
0.830 ± 0.727
0.475 ± 0.447
0.322 ± 0.338
house_16H
8.935 ± 2.504
3.362 ± 0.482
2.660 ± 1.208
4.230 ± 2.189
0.219 ± 0.224
0.404 ± 0.782
jannis
17.756 ± 3.322
10.442 ± 1.404
7.362 ± 0.219
3.205 ± 2.849
0.029 ± 0.064
0.007 ± 0.016
pol
20.542 ± 2.873
4.612 ± 0.912
3.225 ± 1.080
15.830 ± 2.562
1.708 ± 0.599
1.012 ± 0.859

Table 7: Accuracy barrier for non-oblivious trees with AM.

Train
Test
Dataset
Naive
Perm
Perm&Flip
Naive
Perm
Perm&Flip

Bioresponse
18.944 ± 10.076
14.066 ± 7.045
5.710 ± 0.915
8.235 ± 6.456
5.037 ± 3.141
0.966 ± 0.316
Diabetes130US
2.148 ± 0.601
3.086 ± 2.566
0.574 ± 0.365
1.014 ± 0.959
1.936 ± 2.878
0.105 ± 0.152
Higgs
27.578 ± 1.742
30.704 ± 2.899
18.435 ± 1.599
4.055 ± 1.089
7.272 ± 1.089
1.044 ± 0.483
MagicTelescope
2.995 ± 1.198
3.309 ± 1.486
0.778 ± 0.515
2.096 ± 1.055
2.693 ± 1.190
0.428 ± 0.327
MiniBooNE
18.238 ± 4.570
34.934 ± 8.157
2.332 ± 0.383
12.592 ± 4.190
28.721 ± 7.869
0.074 ± 0.081
bank-marketing
13.999 ± 4.110
13.598 ± 7.638
3.098 ± 0.539
13.593 ± 4.567
12.810 ± 7.605
2.643 ± 0.704
california
6.396 ± 2.472
5.800 ± 2.036
0.697 ± 0.535
5.226 ± 2.377
4.858 ± 2.017
0.261 ± 0.285
covertype
16.823 ± 4.159
19.708 ± 6.392
1.420 ± 0.619
14.900 ± 4.016
17.765 ± 6.400
0.758 ± 0.540
credit
7.317 ± 2.425
10.556 ± 8.753
3.640 ± 1.624
5.861 ± 2.064
9.378 ± 9.083
2.551 ± 1.987
default-of-credit-card-clients
14.318 ± 4.509
14.166 ± 2.297
4.247 ± 1.678
6.227 ± 4.205
6.514 ± 2.049
0.885 ± 1.852
electricity
10.090 ± 2.930
12.955 ± 4.558
0.762 ± 0.332
9.422 ± 2.795
12.261 ± 4.554
0.499 ± 0.260
eye_movements
18.743 ± 1.994
18.757 ± 1.273
10.957 ± 1.019
1.495 ± 0.467
1.583 ± 1.011
0.146 ± 0.167
heloc
4.434 ± 1.611
6.564 ± 2.404
1.774 ± 0.672
0.830 ± 0.727
2.179 ± 2.100
0.385 ± 0.370
house_16H
8.935 ± 2.504
10.184 ± 2.667
3.908 ± 0.863
4.230 ± 2.189
5.664 ± 2.461
1.056 ± 0.693
jannis
17.756 ± 3.322
19.004 ± 1.246
9.890 ± 1.036
3.205 ± 2.849
4.047 ± 1.415
0.346 ± 0.443
pol
20.542 ± 2.873
16.267 ± 3.914
7.967 ± 3.208
15.830 ± 2.562
12.863 ± 3.983
4.539 ± 2.727

Table 8: Accuracy barrier for oblivious trees with WM.

Train
Test
Dataset
Naive
Perm
Perm&Order&Flip
Naive
Perm
Perm&Order&Flip

Bioresponse
16.642 ± 4.362
4.800 ± 0.895
3.289 ± 0.680
7.165 ± 2.547
1.069 ± 1.020
0.299 ± 0.247
Diabetes130US
3.170 ± 3.304
1.120 ± 1.123
0.246 ± 0.177
2.831 ± 3.476
0.882 ± 1.309
0.181 ± 0.155
Higgs
28.640 ± 0.914
19.754 ± 1.023
13.689 ± 0.814
4.648 ± 0.966
1.270 ± 0.808
0.266 ± 0.232
MagicTelescope
2.659 ± 1.637
0.473 ± 0.632
0.077 ± 0.110
2.012 ± 1.343
0.534 ± 0.565
0.093 ± 0.144
MiniBooNE
22.344 ± 7.001
2.388 ± 0.194
1.628 ± 0.208
16.454 ± 6.706
0.075 ± 0.086
0.012 ± 0.019
bank-marketing
13.512 ± 6.416
2.998 ± 1.582
0.925 ± 0.688
12.856 ± 6.609
2.324 ± 1.618
0.634 ± 0.433
california
8.281 ± 4.253
0.874 ± 0.524
0.351 ± 0.267
6.578 ± 4.264
0.342 ± 0.209
0.034 ± 0.024
covertype
23.977 ± 2.565
2.073 ± 0.657
0.976 ± 0.523
21.790 ± 2.253
0.992 ± 0.496
0.422 ± 0.319
credit
6.912 ± 4.083
2.369 ± 0.887
0.662 ± 0.606
5.739 ± 4.502
1.324 ± 0.674
0.350 ± 0.522
default-of-credit-card-clients
16.301 ± 4.462
4.512 ± 1.033
2.902 ± 0.620
7.618 ± 3.873
0.728 ± 0.331
0.531 ± 0.557
electricity
8.835 ± 1.824
1.060 ± 0.684
0.279 ± 0.266
7.952 ± 1.995
0.731 ± 0.383
0.285 ± 0.200
eye_movements
22.604 ± 1.486
12.687 ± 1.645
7.826 ± 1.822
2.884 ± 1.646
0.825 ± 0.711
0.607 ± 0.259
heloc
6.282 ± 2.351
2.517 ± 1.156
1.507 ± 0.498
1.625 ± 1.480
0.869 ± 0.957
0.727 ± 0.785
house_16H
13.600 ± 5.135
3.302 ± 0.376
1.950 ± 0.346
8.055 ± 4.429
0.330 ± 0.441
0.158 ± 0.098
jannis
19.390 ± 1.013
11.358 ± 0.377
7.140 ± 0.538
1.999 ± 1.237
0.305 ± 0.409
0.214 ± 0.235
pol
20.125 ± 2.902
5.059 ± 1.482
2.544 ± 1.005
15.887 ± 3.061
2.100 ± 1.358
0.751 ± 0.892

Table 9: Accuracy barrier for oblivious trees with AM.

Train
Test
Dataset
Naive
Perm
Perm&Order&Flip
Naive
Perm
Perm&Order&Flip

Bioresponse
16.642 ± 4.362
19.033 ± 8.533
6.358 ± 1.915
7.165 ± 2.547
6.904 ± 5.380
1.038 ± 0.591
Diabetes130US
3.170 ± 3.304
5.473 ± 3.260
0.703 ± 0.517
2.831 ± 3.476
5.290 ± 3.486
0.390 ± 0.291
Higgs
28.640 ± 0.914
33.234 ± 3.164
15.678 ± 0.713
4.648 ± 0.966
8.113 ± 2.614
0.415 ± 0.454
MagicTelescope
2.659 ± 1.637
3.902 ± 1.931
0.224 ± 0.256
2.012 ± 1.343
3.687 ± 1.876
0.334 ± 0.434
MiniBooNE
22.344 ± 7.001
41.022 ± 3.398
2.184 ± 0.425
16.454 ± 6.706
34.452 ± 3.161
0.033 ± 0.056
bank-marketing
13.512 ± 6.416
12.248 ± 6.748
1.330 ± 0.806
12.856 ± 6.609
11.356 ± 7.168
0.695 ± 0.464
california
8.281 ± 4.253
9.539 ± 4.798
0.371 ± 0.365
6.578 ± 4.264
8.354 ± 4.648
0.112 ± 0.181
covertype
23.977 ± 2.565
27.590 ± 2.172
1.051 ± 0.407
21.790 ± 2.253
25.289 ± 1.787
0.403 ± 0.236
credit
6.912 ± 4.083
9.839 ± 6.698
1.169 ± 0.839
5.739 ± 4.502
8.291 ± 7.268
0.549 ± 0.751
default-of-credit-card-clients
16.301 ± 4.462
21.746 ± 7.075
3.646 ± 0.520
7.618 ± 3.873
12.183 ± 5.954
0.285 ± 0.372
electricity
8.835 ± 1.824
18.177 ± 5.979
0.472 ± 0.507
7.952 ± 1.995
17.396 ± 5.809
0.405 ± 0.356
eye_movements
22.604 ± 1.486
23.221 ± 3.024
8.588 ± 2.248
2.884 ± 1.646
2.761 ± 1.628
0.398 ± 0.435
heloc
6.282 ± 2.351
9.074 ± 3.894
2.541 ± 0.471
1.625 ± 1.480
3.891 ± 2.655
0.485 ± 0.397
house_16H
13.600 ± 5.135
17.963 ± 5.099
2.841 ± 0.543
8.055 ± 4.429
12.192 ± 4.635
0.292 ± 0.157
jannis
19.390 ± 1.013
22.482 ± 3.113
9.570 ± 0.316
1.999 ± 1.237
4.292 ± 2.509
0.069 ± 0.154
pol
20.125 ± 2.902
19.558 ± 5.785
3.056 ± 0.510
15.887 ± 3.061
14.858 ± 5.523
0.961 ± 0.722

15


---Page Break---
Table 10: Accuracy barrier for decision lists with WM.

Train
Test
Dataset
Naive
Perm
Naive (Modified) Perm (Modified)
Naive
Perm
Naive (Modified) Perm (Modified)

Bioresponse
21.323 ± 6.563 4.259 ± 0.698
14.578 ± 3.930
4.641 ± 0.918
9.325 ± 3.988 0.346 ± 0.277
7.346 ± 4.261
1.309 ± 0.827
Diabetes130US
5.182 ± 3.745
1.483 ± 1.006
2.754 ± 1.098
1.088 ± 0.608
4.910 ± 4.244 1.293 ± 1.332
1.476 ± 1.308
0.849 ± 0.885
Higgs
27.778 ± 1.036 16.110 ± 0.518
28.915 ± 1.314
14.071 ± 0.395
4.777 ± 0.803 0.106 ± 0.203
5.136 ± 0.946
0.039 ± 0.083
MagicTelescope
4.855 ± 3.388
0.355 ± 0.682
5.138 ± 2.655
0.182 ± 0.141
4.137 ± 3.763 0.280 ± 0.519
4.534 ± 2.588
0.157 ± 0.162
MiniBooNE
23.059 ± 1.479 1.911 ± 0.138
14.916 ± 3.616
1.580 ± 0.178
17.248 ± 1.683 0.025 ± 0.036
9.340 ± 3.585
0.035 ± 0.042
bank-marketing
11.952 ± 3.794 0.979 ± 0.478
11.589 ± 2.167
0.373 ± 0.448
11.387 ± 4.113 0.536 ± 0.472
10.540 ± 2.067
0.349 ± 0.348
california
6.522 ± 3.195
0.621 ± 0.363
8.435 ± 3.273
0.538 ± 0.214
5.167 ± 2.962 0.236 ± 0.146
6.844 ± 3.087
0.151 ± 0.147
covertype
13.408 ± 3.839 1.341 ± 0.433
11.114 ± 2.689
1.257 ± 0.904
11.162 ± 3.620 0.472 ± 0.340
8.826 ± 2.729
0.477 ± 0.889
credit
11.238 ± 8.115 1.968 ± 0.990
14.626 ± 5.448
1.390 ± 0.423
10.880 ± 9.040 1.421 ± 1.046
13.667 ± 5.951
0.940 ± 0.612
default-of-credit-card-clients 12.513 ± 5.116 3.107 ± 1.123
11.378 ± 2.123
3.793 ± 0.881
5.161 ± 4.304 0.328 ± 0.512
3.197 ± 1.916
0.666 ± 0.651
electricity
6.524 ± 1.863
0.725 ± 0.451
9.101 ± 2.685
0.944 ± 0.557
5.834 ± 1.838 0.420 ± 0.354
8.487 ± 2.460
0.543 ± 0.511
eye_movements
19.125 ± 1.791 9.433 ± 1.385
19.738 ± 1.490
8.755 ± 1.391
1.990 ± 1.623 0.329 ± 0.102
1.916 ± 1.492
0.277 ± 0.302
heloc
4.513 ± 1.826
1.564 ± 0.617
5.116 ± 0.793
1.574 ± 0.154
0.725 ± 0.598 0.155 ± 0.190
1.263 ± 0.711
0.359 ± 0.346
house_16H
9.195 ± 2.408
2.520 ± 0.446
8.693 ± 1.302
2.222 ± 0.730
4.629 ± 2.314 0.063 ± 0.129
4.192 ± 1.517
0.185 ± 0.296
jannis
20.766 ± 2.097 9.484 ± 0.371
20.520 ± 1.017
7.400 ± 0.324
3.947 ± 2.605 0.006 ± 0.013
4.451 ± 1.300
0.004 ± 0.009
pol
23.401 ± 5.448 3.137 ± 1.038
20.137 ± 4.200
3.435 ± 0.675
18.933 ± 5.249 0.952 ± 0.925
16.522 ± 3.502
1.143 ± 0.565

Table 11: Accuracy barrier for decision lists with AM.

Train
Test
Dataset
Naive
Perm
Naive (Modified) Perm (Modified)
Naive
Perm
Naive (Modified) Perm (Modified)

Bioresponse
21.323 ± 6.563 13.349 ± 5.943
14.578 ± 3.930
10.363 ± 7.256
9.325 ± 3.988
4.817 ± 2.825
7.346 ± 4.261
3.871 ± 4.608
Diabetes130US
5.182 ± 3.745
5.590 ± 3.328
2.754 ± 1.098
1.371 ± 0.507
4.910 ± 4.244
4.926 ± 3.796
1.476 ± 1.308
0.694 ± 0.649
Higgs
27.778 ± 1.036 28.910 ± 2.132
28.915 ± 1.314
20.131 ± 1.693
4.777 ± 0.803
6.722 ± 1.231
5.136 ± 0.946
1.755 ± 1.403
MagicTelescope
4.855 ± 3.388
3.349 ± 3.273
5.138 ± 2.655
1.451 ± 0.705
4.137 ± 3.763
3.001 ± 3.478
4.534 ± 2.588
1.090 ± 0.437
MiniBooNE
23.059 ± 1.479 18.149 ± 7.500
14.916 ± 3.616
3.870 ± 1.168
17.248 ± 1.683 13.868 ± 7.222
9.340 ± 3.585
0.797 ± 0.860
bank-marketing
11.952 ± 3.794 9.782 ± 6.722
11.589 ± 2.167
2.815 ± 0.957
11.387 ± 4.113
9.151 ± 7.204
10.540 ± 2.067
2.521 ± 1.055
california
6.522 ± 3.195
5.812 ± 2.365
8.435 ± 3.273
2.254 ± 0.813
5.167 ± 2.962
4.899 ± 2.018
6.844 ± 3.087
1.186 ± 0.643
covertype
13.408 ± 3.839 14.727 ± 7.029
11.114 ± 2.689
4.036 ± 1.450
11.162 ± 3.620 13.352 ± 7.056
8.826 ± 2.729
2.656 ± 1.302
credit
11.238 ± 8.115 18.620 ± 9.806
14.626 ± 5.448
8.979 ± 6.919
10.880 ± 9.040 18.606 ± 10.015
13.667 ± 5.951
8.113 ± 6.633
default-of-credit-card-clients 12.513 ± 5.116 12.880 ± 5.070
11.378 ± 2.123
6.055 ± 1.178
5.161 ± 4.304
6.465 ± 5.062
3.197 ± 1.916
0.533 ± 0.239
electricity
6.524 ± 1.863
4.988 ± 2.732
9.101 ± 2.685
3.041 ± 0.676
5.834 ± 1.838
4.361 ± 2.532
8.487 ± 2.460
2.637 ± 0.730
eye_movements
19.125 ± 1.791 18.694 ± 1.774
19.738 ± 1.490
13.408 ± 1.196
1.990 ± 1.623
3.046 ± 1.625
1.916 ± 1.492
1.807 ± 1.312
heloc
4.513 ± 1.826
5.504 ± 1.650
5.116 ± 0.793
3.287 ± 0.758
0.725 ± 0.598
1.711 ± 1.278
1.263 ± 0.711
0.528 ± 0.147
house_16H
9.195 ± 2.408
8.591 ± 3.370
8.693 ± 1.302
3.937 ± 0.816
4.629 ± 2.314
4.547 ± 2.726
4.192 ± 1.517
0.751 ± 0.508
jannis
20.766 ± 2.097 20.768 ± 2.200
20.520 ± 1.017
12.008 ± 0.892
3.947 ± 2.605
6.472 ± 2.342
4.451 ± 1.300
0.106 ± 0.162
pol
23.401 ± 5.448 17.384 ± 6.441
20.137 ± 4.200
10.339 ± 2.743
18.933 ± 5.249 13.285 ± 5.863
16.522 ± 3.502
6.492 ± 2.536

Table 12: Training accuracy barrier for permuted models with WM. The numbers in parentheses
represent the original accuracy.

Dataset
Non-Oblivious Tree
Oblivious Tree
Decision List
Decision List (Modified)

Bioresponse
5.876 ± 1.477 (93.005)
4.800 ± 0.895 (91.753)
4.259 ± 0.698 (91.771)
4.641 ± 0.918 (90.489)
Diabetes130US
1.388 ± 1.159 (60.686)
1.120 ± 1.123 (60.567)
1.483 ± 1.006 (60.425)
1.088 ± 0.608 (61.178)
Higgs
18.470 ± 0.769 (97.232)
19.754 ± 1.023 (97.616) 16.110 ± 0.518 (95.838) 14.071 ± 0.395 (95.831)
MagicTelescope
0.576 ± 0.556 (84.963)
0.473 ± 0.632 (84.460)
0.355 ± 0.682 (84.999)
0.182 ± 0.141 (85.411)
MiniBooNE
2.272 ± 0.215 (99.980)
2.388 ± 0.194 (99.980)
1.911 ± 0.138 (99.977)
1.580 ± 0.178 (99.976)
bank-marketing
2.711 ± 1.183 (79.490)
2.998 ± 1.582 (79.351)
0.979 ± 0.478 (79.166)
0.373 ± 0.448 (79.709)
california
0.873 ± 0.551 (87.897)
0.874 ± 0.524 (87.909)
0.621 ± 0.363 (88.012)
0.538 ± 0.214 (88.054)
covertype
1.839 ± 0.336 (79.445)
2.073 ± 0.657 (79.754)
1.341 ± 0.433 (79.618)
1.257 ± 0.904 (79.550)
credit
3.172 ± 2.636 (78.679)
2.369 ± 0.887 (78.231)
1.968 ± 0.990 (78.166)
1.390 ± 0.423 (78.905)
default-of-credit-card-clients
5.419 ± 1.318 (78.017)
4.512 ± 1.033 (78.657)
3.107 ± 1.123 (77.315)
3.793 ± 0.881 (78.308)
electricity
1.035 ± 0.543 (80.375)
1.060 ± 0.684 (80.861)
0.725 ± 0.451 (80.396)
0.944 ± 0.557 (80.651)
eye_movements
11.605 ± 1.927 (81.693)
12.687 ± 1.645 (83.730)
9.433 ± 1.385 (81.075)
8.755 ± 1.391 (81.451)
heloc
1.652 ± 0.475 (77.430)
2.517 ± 1.156 (78.370)
1.564 ± 0.617 (77.968)
1.574 ± 0.154 (78.550)
house_16H
3.362 ± 0.482 (93.093)
3.302 ± 0.376 (93.351)
2.520 ± 0.446 (92.783)
2.222 ± 0.730 (93.058)
jannis
10.442 ± 1.404 (100.000) 11.358 ± 0.377 (100.000) 9.484 ± 0.371 (100.000) 7.400 ± 0.324 (100.000)
pol
4.612 ± 0.912 (98.348)
5.059 ± 1.482 (98.340)
3.137 ± 1.038 (97.883)
3.435 ± 0.675 (97.881)

Table 13: Training accuracy barrier for permuted models with AM. The numbers in parentheses
represent the original accuracy.

Dataset
Non-Oblivious
Oblivious
Decision List
Decision List (Modified)

Bioresponse
14.066 ± 7.045 (93.005)
19.033 ± 8.533 (91.753)
13.349 ± 5.943 (91.771)
10.363 ± 7.256 (90.489)
Diabetes130US
3.086 ± 2.566 (60.686)
5.473 ± 3.260 (60.567)
5.590 ± 3.328 (60.425)
1.371 ± 0.507 (61.178)
Higgs
30.704 ± 2.899 (97.232)
33.234 ± 3.164 (97.616)
28.910 ± 2.132 (95.838)
20.131 ± 1.693 (95.831)
MagicTelescope
3.309 ± 1.486 (84.963)
3.902 ± 1.931 (84.460)
3.349 ± 3.273 (84.999)
1.451 ± 0.705 (85.411)
MiniBooNE
34.934 ± 8.157 (99.980)
41.022 ± 3.398 (99.980)
18.149 ± 7.500 (99.977)
3.870 ± 1.168 (99.976)
bank-marketing
13.598 ± 7.638 (79.490)
12.248 ± 6.748 (79.351)
9.782 ± 6.722 (79.166)
2.815 ± 0.957 (79.709)
california
5.800 ± 2.036 (87.897)
9.539 ± 4.798 (87.909)
5.812 ± 2.365 (88.012)
2.254 ± 0.813 (88.054)
covertype
19.708 ± 6.392 (79.445)
27.590 ± 2.172 (79.754)
14.727 ± 7.029 (79.618)
4.036 ± 1.450 (79.550)
credit
10.556 ± 8.753 (78.679)
9.839 ± 6.698 (78.231)
18.620 ± 9.806 (78.166)
8.979 ± 6.919 (78.905)
default-of-credit-card-clients 14.166 ± 2.297 (78.017)
21.746 ± 7.075 (78.657)
12.880 ± 5.070 (77.315)
6.055 ± 1.178 (78.308)
electricity
12.955 ± 4.558 (80.375)
18.177 ± 5.979 (80.861)
4.988 ± 2.732 (80.396)
3.041 ± 0.676 (80.651)
eye_movements
18.757 ± 1.273 (81.693)
23.221 ± 3.024 (83.730)
18.694 ± 1.774 (81.075)
13.408 ± 1.196 (81.451)
heloc
6.564 ± 2.404 (77.430)
9.074 ± 3.894 (78.370)
5.504 ± 1.650 (77.968)
3.287 ± 0.758 (78.550)
house_16H
10.184 ± 2.667 (93.093)
17.963 ± 5.099 (93.351)
8.591 ± 3.370 (92.783)
3.937 ± 0.816 (93.058)
jannis
19.004 ± 1.246 (100.000) 22.482 ± 3.113 (100.000) 20.768 ± 2.200 (100.000) 12.008 ± 0.892 (100.000)
pol
16.267 ± 3.914 (98.348)
19.558 ± 5.785 (98.340)
17.384 ± 6.441 (97.883)
10.339 ± 2.743 (97.881)
16


---Page Break---
Interpolation
70

80

90

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation
60

80

100
Higgs

Interpolation

80

85

MagicTelescope

Interpolation

60

80

100

MiniBooNE

Interpolation
60

70

80

bank-marketing

Interpolation
75

80

85

california

Interpolation

60

80

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

70

80
default-of-credit-card-clients

Interpolation

60

70

80

electricity

Interpolation
60

70

80

eye_movements

Interpolation

70

80
heloc

Interpolation

80

90

house_16H

Interpolation

80

100

jannis

Interpolation

80

90

100
pol

Naive
Tree Permutation
Ours

Figure 9: Interpolation curves of train accuracy for oblivious trees with AM.

Interpolation

70

75

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation
55

60

65

Higgs

Interpolation

80

85

MagicTelescope

Interpolation

60

80

MiniBooNE

Interpolation

60

70

bank-marketing

Interpolation

75

80

85

california

Interpolation
50

60

70

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation

60

70

80

electricity

Interpolation

54

56

58

eye_movements

Interpolation

65

70

heloc

Interpolation

75

80

85

house_16H

Interpolation

65

70

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 10: Interpolation curves of test accuracy for oblivious trees with AM.

Interpolation

80

90

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation
70

80

90

Higgs

Interpolation

82

84

86

MagicTelescope

Interpolation

80

100

MiniBooNE

Interpolation
60

70

80

bank-marketing

Interpolation

80

85

california

Interpolation

60

70

80

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

70

80
default-of-credit-card-clients

Interpolation
70

75

80

electricity

Interpolation
60

70

80

eye_movements

Interpolation
70

75

heloc

Interpolation

80

90

house_16H

Interpolation
80

90

100

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 11: Interpolation curves of train accuracy for oblivious trees with WM.

Interpolation

70

75

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation

62.5

65.0

Higgs

Interpolation

82

84

MagicTelescope

Interpolation
70

80

90

MiniBooNE

Interpolation

60

70

bank-marketing

Interpolation

80

85

california

Interpolation

60

70

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

65

70
default-of-credit-card-clients

Interpolation
70

75

80

electricity

Interpolation

54

56

58

eye_movements

Interpolation

66

68

70

heloc

Interpolation
75

80

85

house_16H

Interpolation

70

72

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 12: Interpolation curves of test accuracy for oblivious trees with WM.

17


---Page Break---
Interpolation
60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

65

70

75
Higgs

Interpolation

75

80

85

MagicTelescope

Interpolation

70

80

90

MiniBooNE

Interpolation

70

75

bank-marketing

Interpolation
75

80

85

california

Interpolation
50

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation

60

70

default-of-credit-card-clients

Interpolation

70

80
electricity

Interpolation

55

60

65

eye_movements

Interpolation
60

70

heloc

Interpolation

80

85

90
house_16H

Interpolation

75

80

jannis

Interpolation

70

80

90

pol

Naive
Tree Permutation
Ours

Figure 13: Interpolation curves of train accuracy for oblivious trees with AM by use of split dataset.

Interpolation

60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

75

80

85MagicTelescope

Interpolation

70

80

90

MiniBooNE

Interpolation

70

75

bank-marketing

Interpolation
75

80

85

california

Interpolation
50

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation
65

70

75

electricity

Interpolation

52.5

55.0

57.5

eye_movements

Interpolation

60

70

heloc

Interpolation

80

85

house_16H

Interpolation

65

70

jannis

Interpolation

70

80

90

pol

Naive
Tree Permutation
Ours

Figure 14: Interpolation curves of test accuracy for oblivious trees with AM by use of split dataset.

Interpolation
60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

65

70

Higgs

Interpolation

75

80

85

MagicTelescope

Interpolation

80

90

MiniBooNE

Interpolation

70

75

bank-marketing

Interpolation

80

85

california

Interpolation

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation
60

65

70

default-of-credit-card-clients

Interpolation

70

80
electricity

Interpolation

55

60

65

eye_movements

Interpolation
60

70

heloc

Interpolation
80

85

house_16H

Interpolation

75

80

jannis

Interpolation
70

80

90

pol

Naive
Tree Permutation
Ours

Figure 15: Interpolation curves of train accuracy for oblivious trees with WM by use of split dataset.

Interpolation

60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

75

80

85MagicTelescope

Interpolation

80

90

MiniBooNE

Interpolation

70

75

bank-marketing

Interpolation

80

85

california

Interpolation

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation
65

70

75

electricity

Interpolation

52.5

55.0

57.5

eye_movements

Interpolation

60

70

heloc

Interpolation

80

85

house_16H

Interpolation

65

70

jannis

Interpolation
70

80

90

pol

Naive
Tree Permutation
Ours

Figure 16: Interpolation curves of test accuracy for oblivious trees with WM by use of split dataset.

18


---Page Break---
Interpolation

70

80

90

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation

70

80

90

Higgs

Interpolation
80.0

82.5

85.0

MagicTelescope

Interpolation

80

100

MiniBooNE

Interpolation
60

70

80

bank-marketing

Interpolation

80

85

california

Interpolation

60

80

covertype

Interpolation

60

70

80

Accuracy

credit

Interpolation
60

70

80
default-of-credit-card-clients

Interpolation

70

80

electricity

Interpolation

70

80

eye_movements

Interpolation

70

75

heloc

Interpolation

85

90

house_16H

Interpolation
80

90

100

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 17: Interpolation curves of train accuracy for non-oblivious trees with AM.

Interpolation

65

70

75

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

82

84

86

MagicTelescope

Interpolation

70

80

90

MiniBooNE

Interpolation

60

70

bank-marketing

Interpolation

80

85

california

Interpolation
50

60

70

covertype

Interpolation

60

70

Accuracy

credit

Interpolation

60

65

70
default-of-credit-card-clients

Interpolation

70

80

electricity

Interpolation

56

58

eye_movements

Interpolation

65

70

heloc

Interpolation
80

85

house_16H

Interpolation

70

75
jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 18: Interpolation curves of test accuracy for non-oblivious trees with AM.

Interpolation

70

80

90

Accuracy

Bioresponse

Interpolation
58

60

62

Diabetes130US

Interpolation
70

80

90

Higgs

Interpolation
80.0

82.5

85.0

MagicTelescope

Interpolation

80

90

100

MiniBooNE

Interpolation

70

80

bank-marketing

Interpolation

80

85

california

Interpolation
60

70

80

covertype

Interpolation

70

75

80

Accuracy

credit

Interpolation
60

70

80
default-of-credit-card-clients

Interpolation

70

80

electricity

Interpolation

70

80

eye_movements

Interpolation
72.5

75.0

77.5

heloc

Interpolation

85

90

house_16H

Interpolation
80

90

100

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 19: Interpolation curves of train accuracy for non-oblivious trees with WM.

Interpolation

70

75

Accuracy

Bioresponse

Interpolation

55

60

Diabetes130US

Interpolation

62.5

65.0

Higgs

Interpolation

82

84

MagicTelescope

Interpolation
70

80

90

MiniBooNE

Interpolation

60

70

bank-marketing

Interpolation

80

85

california

Interpolation

60

70

covertype

Interpolation

70

75

Accuracy

credit

Interpolation

60

65

70
default-of-credit-card-clients

Interpolation
70

75

80

electricity

Interpolation

54

56

58

eye_movements

Interpolation

66

68

70

heloc

Interpolation
75

80

85

house_16H

Interpolation

70

72

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 20: Interpolation curves of test accuracy for non-oblivious trees with WM.

19


---Page Break---
Interpolation

60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

65

70

75
Higgs

Interpolation
70

80

MagicTelescope

Interpolation
70

80

90

MiniBooNE

Interpolation

70

80

bank-marketing

Interpolation
75

80

85

california

Interpolation

60

70

covertype

Interpolation
50

60

70

Accuracy

credit

Interpolation

60

70

default-of-credit-card-clients

Interpolation

70

80
electricity

Interpolation

55

60

65

eye_movements

Interpolation

65

70

75
heloc

Interpolation
84

86

88

house_16H

Interpolation

75

80

jannis

Interpolation
70

80

90

pol

Naive
Tree Permutation
Ours

Figure 21: Interpolation curves of train accuracy for non-oblivious trees with AM by use of split
dataset.

Interpolation

60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

70

80

MagicTelescope

Interpolation
70

80

90

MiniBooNE

Interpolation

65

70

75

bank-marketing

Interpolation
75

80

85

california

Interpolation

60

70

covertype

Interpolation

60

80

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation

65

70

75

electricity

Interpolation
50

55

eye_movements

Interpolation

60

70

heloc

Interpolation
82

84

86

house_16H

Interpolation
65

70

jannis

Interpolation
70

80

90

pol

Naive
Tree Permutation
Ours

Figure 22: Interpolation curves of test accuracy for non-oblivious trees with AM by use of split
dataset.

Interpolation
60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

70

75
Higgs

Interpolation
70

80

MagicTelescope

Interpolation
80

90

MiniBooNE

Interpolation

70

80bank-marketing

Interpolation

80

85

california

Interpolation
60

70

covertype

Interpolation
50

60

70

Accuracy

credit

Interpolation

60

70

default-of-credit-card-clients

Interpolation

70

80

electricity

Interpolation

55

60

65

eye_movements

Interpolation

65

70

75
heloc

Interpolation

86

88

house_16H

Interpolation

75

80

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 23: Interpolation curves of train accuracy for non-oblivious trees with WM by use of split
dataset.

Interpolation
60

70

Accuracy

Bioresponse

Interpolation
50

55

60

Diabetes130US

Interpolation

60

65

Higgs

Interpolation

70

80

MagicTelescope

Interpolation
80

90

MiniBooNE

Interpolation

65

70

75

bank-marketing

Interpolation

80

85

california

Interpolation
60

70

covertype

Interpolation
50

60

70

Accuracy

credit

Interpolation

60

70
default-of-credit-card-clients

Interpolation
65

70

75

electricity

Interpolation

52.5

55.0

57.5

eye_movements

Interpolation

60

70

heloc

Interpolation

82.5

85.0

87.5
house_16H

Interpolation
65

70

jannis

Interpolation

80

90

pol

Naive
Tree Permutation
Ours

Figure 24: Interpolation curves of test accuracy for non-oblivious trees with WM by use of split
dataset.

20


---Page Break---
NeurIPS Paper Checklist
498

1. Claims
499

Question: Do the main claims made in the abstract and introduction accurately reflect the
500

paper’s contributions and scope?
501

Answer: [Yes]
502

Justification: The abstract and introduction consistently present our research on tree ensem-
503

bles from LMC perspectives.
504

Guidelines:
505

• The answer NA means that the abstract and introduction do not include the claims
506

made in the paper.
507

• The abstract and/or introduction should clearly state the claims made, including the
508

contributions made in the paper and important assumptions and limitations. A No or
509

NA answer to this question will not be perceived well by the reviewers.
510

• The claims made should match theoretical and experimental results, and reflect how
511

much the results can be expected to generalize to other settings.
512

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
513

are not attained by the paper.
514

2. Limitations
515

Question: Does the paper discuss the limitations of the work performed by the authors?
516

Answer: [Yes]
517

Justification: In Section 3.2, we have discussed the limitations.
518

Guidelines:
519

• The answer NA means that the paper has no limitation while the answer No means that
520

the paper has limitations, but those are not discussed in the paper.
521

• The authors are encouraged to create a separate "Limitations" section in their paper.
522

• The paper should point out any strong assumptions and how robust the results are to
523

violations of these assumptions (e.g., independence assumptions, noiseless settings,
524

model well-specification, asymptotic approximations only holding locally). The authors
525

should reflect on how these assumptions might be violated in practice and what the
526

implications would be.
527

• The authors should reflect on the scope of the claims made, e.g., if the approach was
528

only tested on a few datasets or with a few runs. In general, empirical results often
529

depend on implicit assumptions, which should be articulated.
530

• The authors should reflect on the factors that influence the performance of the approach.
531

For example, a facial recognition algorithm may perform poorly when image resolution
532

is low or images are taken in low lighting. Or a speech-to-text system might not be
533

used reliably to provide closed captions for online lectures because it fails to handle
534

technical jargon.
535

• The authors should discuss the computational efficiency of the proposed algorithms
536

and how they scale with dataset size.
537

• If applicable, the authors should discuss possible limitations of their approach to
538

address problems of privacy and fairness.
539

• While the authors might fear that complete honesty about limitations might be used by
540

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
541

limitations that aren’t acknowledged in the paper. The authors should use their best
542

judgment and recognize that individual actions in favor of transparency play an impor-
543

tant role in developing norms that preserve the integrity of the community. Reviewers
544

will be specifically instructed to not penalize honesty concerning limitations.
545

3. Theory Assumptions and Proofs
546

Question: For each theoretical result, does the paper provide the full set of assumptions and
547

a complete (and correct) proof?
548

Answer: [NA]
549

21


---Page Break---
Justification: We do not provide theoretical results in this paper.
550

Guidelines:
551

• The answer NA means that the paper does not include theoretical results.
552

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
553

referenced.
554

• All assumptions should be clearly stated or referenced in the statement of any theorems.
555

• The proofs can either appear in the main paper or the supplemental material, but if
556

they appear in the supplemental material, the authors are encouraged to provide a short
557

proof sketch to provide intuition.
558

• Inversely, any informal proof provided in the core of the paper should be complemented
559

by formal proofs provided in appendix or supplemental material.
560

• Theorems and Lemmas that the proof relies upon should be properly referenced.
561

4. Experimental Result Reproducibility
562

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
563

perimental results of the paper to the extent that it affects the main claims and/or conclusions
564

of the paper (regardless of whether the code and data are provided or not)?
565

Answer: [Yes]
566

Justification: The experimental setup is detailed in Section 4.1.
567

Guidelines:
568

• The answer NA means that the paper does not include experiments.
569

• If the paper includes experiments, a No answer to this question will not be perceived
570

well by the reviewers: Making the paper reproducible is important, regardless of
571

whether the code and data are provided or not.
572

• If the contribution is a dataset and/or model, the authors should describe the steps taken
573

to make their results reproducible or verifiable.
574

• Depending on the contribution, reproducibility can be accomplished in various ways.
575

For example, if the contribution is a novel architecture, describing the architecture fully
576

might suffice, or if the contribution is a specific model and empirical evaluation, it may
577

be necessary to either make it possible for others to replicate the model with the same
578

dataset, or provide access to the model. In general. releasing code and data is often
579

one good way to accomplish this, but reproducibility can also be provided via detailed
580

instructions for how to replicate the results, access to a hosted model (e.g., in the case
581

of a large language model), releasing of a model checkpoint, or other means that are
582

appropriate to the research performed.
583

• While NeurIPS does not require releasing code, the conference does require all submis-
584

sions to provide some reasonable avenue for reproducibility, which may depend on the
585

nature of the contribution. For example
586

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
587

to reproduce that algorithm.
588

(b) If the contribution is primarily a new model architecture, the paper should describe
589

the architecture clearly and fully.
590

(c) If the contribution is a new model (e.g., a large language model), then there should
591

either be a way to access this model for reproducing the results or a way to reproduce
592

the model (e.g., with an open-source dataset or instructions for how to construct
593

the dataset).
594

(d) We recognize that reproducibility may be tricky in some cases, in which case
595

authors are welcome to describe the particular way they provide for reproducibility.
596

In the case of closed-source models, it may be that access to the model is limited in
597

some way (e.g., to registered users), but it should be possible for other researchers
598

to have some path to reproducing or verifying the results.
599

5. Open access to data and code
600

Question: Does the paper provide open access to the data and code, with sufficient instruc-
601

tions to faithfully reproduce the main experimental results, as described in supplemental
602

material?
603

22


---Page Break---
Answer: [Yes]
604

Justification: Reproducible source code is provided in the supplementary material.
605

Guidelines:
606

• The answer NA means that paper does not include experiments requiring code.
607

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
608

public/guides/CodeSubmissionPolicy) for more details.
609

• While we encourage the release of code and data, we understand that this might not be
610

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
611

including code, unless this is central to the contribution (e.g., for a new open-source
612

benchmark).
613

• The instructions should contain the exact command and environment needed to run to
614

reproduce the results. See the NeurIPS code and data submission guidelines (https:
615

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
616

• The authors should provide instructions on data access and preparation, including how
617

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
618

• The authors should provide scripts to reproduce all experimental results for the new
619

proposed method and baselines. If only a subset of experiments are reproducible, they
620

should state which ones are omitted from the script and why.
621

• At submission time, to preserve anonymity, the authors should release anonymized
622

versions (if applicable).
623

• Providing as much information as possible in supplemental material (appended to the
624

paper) is recommended, but including URLs to data and code is permitted.
625

6. Experimental Setting/Details
626

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
627

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
628

results?
629

Answer: [Yes]
630

Justification: The experimental setup is detailed in Section 4.1.
631

Guidelines:
632

• The answer NA means that the paper does not include experiments.
633

• The experimental setting should be presented in the core of the paper to a level of detail
634

that is necessary to appreciate the results and make sense of them.
635

• The full details can be provided either with the code, in appendix, or as supplemental
636

material.
637

7. Experiment Statistical Significance
638

Question: Does the paper report error bars suitably and correctly defined or other appropriate
639

information about the statistical significance of the experiments?
640

Answer: [Yes]
641

Justification: We conducted experiments multiple times with different random seeds and
642

have reported the results, including the variability.
643

Guidelines:
644

• The answer NA means that the paper does not include experiments.
645

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
646

dence intervals, or statistical significance tests, at least for the experiments that support
647

the main claims of the paper.
648

• The factors of variability that the error bars are capturing should be clearly stated (for
649

example, train/test split, initialization, random drawing of some parameter, or overall
650

run with given experimental conditions).
651

• The method for calculating the error bars should be explained (closed form formula,
652

call to a library function, bootstrap, etc.)
653

• The assumptions made should be given (e.g., Normally distributed errors).
654

23


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error
655

of the mean.
656

• It is OK to report 1-sigma error bars, but one should state it. The authors should
657

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
658

of Normality of errors is not verified.
659

• For asymmetric distributions, the authors should be careful not to show in tables or
660

figures symmetric error bars that would yield results that are out of range (e.g. negative
661

error rates).
662

• If error bars are reported in tables or plots, The authors should explain in the text how
663

they were calculated and reference the corresponding figures or tables in the text.
664

8. Experiments Compute Resources
665

Question: For each experiment, does the paper provide sufficient information on the com-
666

puter resources (type of compute workers, memory, time of execution) needed to reproduce
667

the experiments?
668

Answer: [Yes]
669

Justification: The computational resources used in our experiment is described in Section 4.1.
670

Guidelines:
671

• The answer NA means that the paper does not include experiments.
672

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
673

or cloud provider, including relevant memory and storage.
674

• The paper should provide the amount of compute required for each of the individual
675

experimental runs as well as estimate the total compute.
676

• The paper should disclose whether the full research project required more compute
677

than the experiments reported in the paper (e.g., preliminary or failed experiments that
678

didn’t make it into the paper).
679

9. Code Of Ethics
680

Question: Does the research conducted in the paper conform, in every respect, with the
681

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
682

Answer: [Yes]
683

Justification: We reviewed the NeurIPS Code of Ethics and conducted our research in
684

accordance with it.
685

Guidelines:
686

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
687

• If the authors answer No, they should explain the special circumstances that require a
688

deviation from the Code of Ethics.
689

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
690

eration due to laws or regulations in their jurisdiction).
691

10. Broader Impacts
692

Question: Does the paper discuss both potential positive societal impacts and negative
693

societal impacts of the work performed?
694

Answer: [Yes]
695

Justification: We have addressed societal impact in Section 5.
696

Guidelines:
697

• The answer NA means that there is no societal impact of the work performed.
698

• If the authors answer NA or No, they should explain why their work has no societal
699

impact or why the paper does not address societal impact.
700

• Examples of negative societal impacts include potential malicious or unintended uses
701

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
702

(e.g., deployment of technologies that could make decisions that unfairly impact specific
703

groups), privacy considerations, and security considerations.
704

24


---Page Break---
• The conference expects that many papers will be foundational research and not tied
705

to particular applications, let alone deployments. However, if there is a direct path to
706

any negative applications, the authors should point it out. For example, it is legitimate
707

to point out that an improvement in the quality of generative models could be used to
708

generate deepfakes for disinformation. On the other hand, it is not needed to point out
709

that a generic algorithm for optimizing neural networks could enable people to train
710

models that generate Deepfakes faster.
711

• The authors should consider possible harms that could arise when the technology is
712

being used as intended and functioning correctly, harms that could arise when the
713

technology is being used as intended but gives incorrect results, and harms following
714

from (intentional or unintentional) misuse of the technology.
715

• If there are negative societal impacts, the authors could also discuss possible mitigation
716

strategies (e.g., gated release of models, providing defenses in addition to attacks,
717

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
718

feedback over time, improving the efficiency and accessibility of ML).
719

11. Safeguards
720

Question: Does the paper describe safeguards that have been put in place for responsible
721

release of data or models that have a high risk for misuse (e.g., pretrained language models,
722

image generators, or scraped datasets)?
723

Answer: [NA]
724

Justification: We provide the source code as supplementary material; however, since the
725

experiments concern the fundamental nature of machine learning models, we believe there
726

are no risks involved.
727

Guidelines:
728

• The answer NA means that the paper poses no such risks.
729

• Released models that have a high risk for misuse or dual-use should be released with
730

necessary safeguards to allow for controlled use of the model, for example by requiring
731

that users adhere to usage guidelines or restrictions to access the model or implementing
732

safety filters.
733

• Datasets that have been scraped from the Internet could pose safety risks. The authors
734

should describe how they avoided releasing unsafe images.
735

• We recognize that providing effective safeguards is challenging, and many papers do
736

not require this, but we encourage authors to take this into account and make a best
737

faith effort.
738

12. Licenses for existing assets
739

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
740

the paper, properly credited and are the license and terms of use explicitly mentioned and
741

properly respected?
742

Answer: [Yes]
743

Justification: We have used open datasets, citing them in accordance with their license
744

information.
745

Guidelines:
746

• The answer NA means that the paper does not use existing assets.
747

• The authors should cite the original paper that produced the code package or dataset.
748

• The authors should state which version of the asset is used and, if possible, include a
749

URL.
750

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
751

• For scraped data from a particular source (e.g., website), the copyright and terms of
752

service of that source should be provided.
753

• If assets are released, the license, copyright information, and terms of use in the
754

package should be provided. For popular datasets, paperswithcode.com/datasets
755

has curated licenses for some datasets. Their licensing guide can help determine the
756

license of a dataset.
757

25


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
758

the derived asset (if it has changed) should be provided.
759

• If this information is not available online, the authors are encouraged to reach out to
760

the asset’s creators.
761

13. New Assets
762

Question: Are new assets introduced in the paper well documented and is the documentation
763

provided alongside the assets?
764

Answer: [NA]
765

Justification: We do not provide any new assets.
766

Guidelines:
767

• The answer NA means that the paper does not release new assets.
768

• Researchers should communicate the details of the dataset/code/model as part of their
769

submissions via structured templates. This includes details about training, license,
770

limitations, etc.
771

• The paper should discuss whether and how consent was obtained from people whose
772

asset is used.
773

• At submission time, remember to anonymize your assets (if applicable). You can either
774

create an anonymized URL or include an anonymized zip file.
775

14. Crowdsourcing and Research with Human Subjects
776

Question: For crowdsourcing experiments and research with human subjects, does the paper
777

include the full text of instructions given to participants and screenshots, if applicable, as
778

well as details about compensation (if any)?
779

Answer: [NA]
780

Justification: This paper neither engages in crowdsourcing nor research involving human
781

subjects.
782

Guidelines:
783

• The answer NA means that the paper does not involve crowdsourcing nor research with
784

human subjects.
785

• Including this information in the supplemental material is fine, but if the main contribu-
786

tion of the paper involves human subjects, then as much detail as possible should be
787

included in the main paper.
788

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
789

or other labor should be paid at least the minimum wage in the country of the data
790

collector.
791

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
792

Subjects
793

Question: Does the paper describe potential risks incurred by study participants, whether
794

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
795

approvals (or an equivalent approval/review based on the requirements of your country or
796

institution) were obtained?
797

Answer: [NA]
798

Justification: This paper neither engages in crowdsourcing nor research involving human
799

subjects.
800

Guidelines:
801

• The answer NA means that the paper does not involve crowdsourcing nor research with
802

human subjects.
803

• Depending on the country in which research is conducted, IRB approval (or equivalent)
804

may be required for any human subjects research. If you obtained IRB approval, you
805

should clearly state this in the paper.
806

• We recognize that the procedures for this may vary significantly between institutions
807

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
808

guidelines for their institution.
809

26


---Page Break---
• For initial submissions, do not include any information that would break anonymity (if
810

applicable), such as the institution conducting the review.
811

27


---Page Break---
