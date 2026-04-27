Parameter-efficient Fine-tuning in Hyperspherical
Space for Open-vocabulary Semantic Segmentation

Anonymous Author(s)
Affiliation
Address
email

Abstract

Open-vocabulary semantic segmentation seeks to label each pixel in an image
1

with arbitrary text descriptions. Vision-language foundation models, especially
2

CLIP, have recently emerged as powerful tools for acquiring open-vocabulary
3

capabilities. However, fine-tuning CLIP to equip it with pixel-level prediction
4

ability often suffers three issues: 1) high computational cost, 2) misalignment
5

between the two inherent modalities of CLIP, and 3) degraded generalization ability
6

on unseen categories. To address these issues, we propose H-CLIP, a symmetrical
7

parameter-efficient fine-tuning (PEFT) strategy conducted in hyperspherical space
8

for both of the two CLIP modalities. Specifically, the PEFT strategy is achieved
9

by a series of efficient block-diagonal learnable transformation matrices and a
10

dual cross-relation communication module among all learnable matrices. Since
11

the PEFT strategy is conducted symmetrically to the two CLIP modalities, the
12

misalignment between them is mitigated. Furthermore, we apply an additional
13

constraint to PEFT on the CLIP text encoder according to the hyperspherical energy
14

principle, i.e., minimizing hyperspherical energy during fine-tuning preserves the
15

intrinsic structure of the original parameter space, to prevent the destruction of
16

the generalization ability offered by the CLIP text encoder. Extensive evaluations
17

across various benchmarks show that H-CLIP achieves new SOTA open-vocabulary
18

semantic segmentation results while only requiring updating approximately 4% of
19

the total parameters of CLIP.
20

1
Introduction
21

The aim of open-vocabulary semantic segmentation is to create a segmentation model capable of
22

labeling each pixel in an image with categories that are not limited to a specific closed set according to
23

text descriptions. Vision-language foundation models [43, 5, 34, 39, 11, 17, 26, 21, 27, 13, 18, 29, 10,
24

28, 45], especially CLIP [39], are often utilized to endow open-vocabulary recognition capabilities.
25

Consequently, open-vocabulary semantic segmentation essentially boil down to transferring these
26

vision-language foundation models, originally trained with image-level supervision, to perform
27

pixel-level predictions.
28

To this end, current methods [52, 48, 7, 50] typically fine-tune CLIP on a benchmark dataset with
29

segmentation annotations, i.e., COCO [2], to equip it with the segmentation ability. However, this
30

often leads to three main issues. First, fine-tuning CLIP on limited categories would affect its
31

generalization ability, resulting in significant performance degradation on unseen categories. Second,
32

current fine-tuning strategies are usually asymmetrical, which inevitably causes a misalignment
33

between the two inherent modalities of CLIP, i.e., image and text [52], which may lead to sub-
34

optimal performance. Third, although remarkable performance gains, these approaches often rely on
35

computationally extensive full fine-tuning, which raises concerns about scalability and affordability.
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
To address these issues, we propose a symmetric parameter-efficient fine-tuning (PEFT) strategy
37

for CLIP, dubbed H-CLIP. Specifically, we implement this PEFT through a partial orthogonal fine-
38

tuning (POF) strategy, which introduces a series of efficient block-diagonal learnable transformation
39

matrices into the hyperspherical space. Then, to preserve CLIP’s generalization ability, we leverage
40

the hyperspherical energy principle [32, 38], which suggests that maintaining the same hyperspherical
41

energy during fine-tuning preserves the intrinsic structure, i.e., generalization ability. In light of this,
42

we upgrade our POF by incorporating orthogonal constraints in the learnable matrices for updating
43

CLIP’s text encoder, as orthogonal transformations keep the hyperspherical energy unchanged during
44

fine-tuning. Subsequently, we introduce a dual cross-relation communication (DCRC) module to
45

explicitly encourage cross-modal and cross-layer communications within all learnable matrices.
46

This communication not only preserves the hyperspherical energy but also further mitigating the
47

misalignment problem.
48

Extensive results demonstrate that H-CLIP achieves new state-of-the-art open-vocabulary semantic
49

segmentation results across three benchmarks by fine-tuning CLIP with approximately 4% of the
50

total parameters of CLIP.
51

2
Related Work
52

2.1
Open-vocabulary Semantic Segmentation
53

Prior open-vocabulary semantic segmentation works typically perform this task through leveraging
54

CLIP [39]. initial efforts like [56] directly fine-tune CLIP on mainstream segmentation datasets, e.g.,
55

COCO [2]. However, they claim that fine-tuning CLIP’s encoder significantly reduces its ability
56

to generalize to unseen classes. To address this issue, some methods [15, 8, 51, 49] swing to the
57

opposite extreme, fine-tuning an additional mask generator [6] for segmentation while keeping CLIP
58

frozen to maintain generalization-oriented recognition. However, this frozen parameter space lacks
59

segmentation awareness, resulting in a misalignment between regions and text descriptions [30].
60

Other studies [52, 50, 7] propose an advanced solution that fine-tunes only selected parameters, e.g.,
61

certain layers of CLIP, to enable pixel-level predictions while keeping most of CLIP’s parameters
62

fixed, thus minimizing losing of generalization. Although the advantages are remarkable, these
63

methods often work with a very small learning rate, implicitly encouraging a small deviation from
64

the pre-trained CLIP, limiting the segmentation performance. In a nutshell, the trade-off between
65

preserving CLIP’s generalization and learning segmentation knowledge persists, hindering the final
66

performance. Based on the paradigm of existing fine-tuning-based methods, our method explores a
67

better trade-off from a fresh viewpoint: hyperspherical space.
68

2.2
Large-scale Model Fine-tuning
69

Along with the improvement of large-scale foundation models [26, 34, 28, 23, 53, 42, 41, 40, 60],
70

e.g., segment anything model [23], numerous fine-tuning works [37, 36, 4, 57, 58, 14, 54, 47, 31, 61]
71

are proposed to adapt these models to various downstream scenarios. The core of these approaches
72

lies in updating only limited parameters to capture the specific characteristics of different scenarios,
73

while keeping most parameters fixed to maintain generalization. In contrast, fine-tuning CLIP
74

for open-vocabulary semantic segmentation often meets a dilemma. On the one hand, limited
75

parameters typically fall short in facilitating the transition from a classification model, i.e., CLIP, to a
76

segmentation task. On the other hand, directly increasing the number of trainable parameters risks
77

undermining CLIP’s ability to generalize to unseen classes, as experimented in CAT-Seg [7]. Most
78

methods [52, 48] solve this issue by simply freezing CLIP’s text encoder and fine-tuning its image
79

encoder, inevitably causing misalignment between the two modalities of CLIP. In this paper, we shed
80

light on how to preserve generalization in a symmetric parameter-efficient fine-tuning manner and
81

strive to explore an appropriate fine-tuning method for open-vocabulary semantic segmentation.
82

3
Preliminaries
83

3.1
Hyperspherical Energy
84

Existing fine-tuning methods implicitly assume that a smaller Euclidean distance between the fine-
85

tuned model and the pre-trained model indicates better preservation of the pre-trained ability. However,
86

2


---Page Break---
the Euclidean difference is unable to fully capture the degree of semantic preservation. According to
87

the inspiration from Thomson problem[44] which is to determine the minimum electrostatic potential
88

energy configuration of N mutually-repelling electrons on the surface of a unit sphere, we adopt the
89

Hyperspherical Energy to characterize the diversity of the model. The hyperspherical energy function
90

of a fully connected layer W is defined as HE(W ):=P

i̸=j ∥ˆwi −ˆwj∥−1, where ˆwi :=wi/∥wi∥
91

denotes the i-th normalized neuron. The power of the model representation can be characterized by the
92

hyperspherical energy of its neurons. Higher energy implies higher redundancy, while lower energy
93

indicates that these neurons of the model are more diverse. For the original semantic information not
94

to be destroyed in the case of fine-tuning, we hypothesize that a good fine-tuning model should have
95

a minimal difference in hyperspherical energy compared to the pre-trained model:
96

min
W
HE(W ) −HE(W 0)

⇔
min
W


X

i̸=j
∥ˆwi −ˆwj∥−1 −
X

i̸=j
∥ˆw0
i −ˆw0
j∥−1
.
(1)

One can easily observe that the attainable minimum is zero for Eq. (1). In this case, the hyperspherical
97

energy should satisfy an invariance property (the application of the same orthogonal transformation
98

for all neurons demonstrates the pairwise hyperspherical similarity). Based on the hyperspherical
99

energy invariance property, the minimum of zero can be achieved as long as W and W 0 differ
100

only up to a rotation or reflection, i.e., W =RW 0 in which R∈Rd×d is an orthogonal matrix (The
101

determinant 1 or −1 means rotation or reflection, respectively).
102

3.2
Notation of Tensor Product
103

In this section, we introduce the fundamental concept underlying our DCRC (Sec. 4.3): tensor product.
104

A p-order tensor is indexed by p indices and can be represented as a multidimensional array of data.
105

Formally, a p-order tensor A can be written as A = (ai1,i2,··· ,ip) ∈Rn1×n2×···np. Slices of a tensor
106

are matrices defined from the tensor by holding all but two indices constant. For a 3-order tensor,
107

A(:, :, k) corresponds the kth frontal slice. For p-order tensors, matrix slices of p-order tensors can be
108

referenced using linear indexing by reshaping the tensor into an n1 × n2 × n3n4 · · · np 3-order tensor
109

and referring to the kth frontal slice as A(:, :, k). Ai ∈Rn1×n2×···np−1 for i = 1, · · · , np denotes the
110

(p −1)-order tensor created by holding the pth index of A fixed at i. It is possible to create a tensor
111

in a block circulant pattern, where each block is a tensor of (p −1)-order:
112

circ(A) =





A1
Anp
Anp−1
· · ·
A2
A2
A1
Anp
· · ·
A3
...
...
...
...
...
Anp
Anp−1
Anp−2
· · ·
A1



,

where circ(·) creates a block circulant tensor and the size of circ(A) is (n1np×n2np×· · ·×np−2np×
113

np−1). define unfold(·) to take an n1 × · · · × np tensor A and return an n1np × n2 × · · · np−1 block
114

tensor in the following way:
115

unfold(A) =
A1
A2
· · ·
Anp
T .

The operation that takes unfold back to tensor form is the “fold” command. Specially, fold(·, np)
116

takes an n1np × n2 × · · · × np−1 block tensor and returns an n1 × · · · × np tensor, defined as:
117

fold(unfold(A), np) = A.

4
Methodology
118

4.1
Overview of H-CLIP
119

Fig. 2 illustrates the proposed H-CLIP framework, which is based on two core components: (1) POF
120

updates the pre-trained parameter space of CLIP using a series of block-diagonal transformation
121

matrices. According to analysis in Sec. 1, each parameter matrix in CLIP’s text encoder is orthogonal
122

to preserve generalization. (2) DCRC incorporates cross-modal and cross-layer communication
123

within all tunable matrices, facilitating alignment between different modalities.
124

3


---Page Break---
…

Dual Cross Relation 

Communication

Decoder

Image

Text

Parameter Space

 Fine-tuning

 Orthogonal 

Matrix

 Frozen Pre-trained Weight

…

����

…

����

����

����

…

����

����

����

…

����

����

…

����

…

����

����

����

…

����

����

����

…

����

����

…

…

q

q

q

q

d

d

Figure 1: A schematic representation of H-CLIP. In the H-CLIP framework, we propose a partial
orthogonal fine-tuning strategy, where each pre-trained weight matrix is paired with a tuned block-
diagonal transformation matrix, some of which are orthogonal to preserve generalization. Then, we
introduce a dual cross-relation communication mechanism to facilitate communication among all
matrices, enabling alignment between different modalities.

4.2
Partial Orthogonal Fine-tuning
125

The core idea of partial orthogonal fine-tuning (POF) is to introduce the concept of hyperspherical
126

space for fine-tuning CLIP. In this hyperspherical space, we fine-tune CLIP’s text encoder under an
127

orthogonality design principle from OFT [38] to preserve the hyperspherical energy of the pre-trained
128

parameter space. Similarly, we use Cayley parameterization [3] to ensure a tunable matrix R is
129

strictly orthogonal, formally as:
130

R = (I + Q)(I −Q)−1,
(2)

where Q is skew-symmetric. Here, for R in CLIP’s image encoder, we remove the orthogonality
131

constraint, defined as:
132

R⊤R = RR⊤= I,
(3)
where I is an identity matrix. Considering the relatively large dimension d of the pre-trained matrix,
133

for better efficiency, we introduce a block-diagonal structure by parameterizing R with b blocks,
134

formally as:
135

R = diag(R1, R2, · · · Ri, · · · , Rb) =





R1
...
Rb



,
(4)

where Ri ∈Rd/b×d/b.
Specifically, denote RV
= {Rv1, · · · , Rvℓ, · · · , RvL} and RE =
136

{Re1, · · · , Reℓ, · · · , ReL} as the sets of block-diagonal matrices in CLIP’s image encoder and
137

text encoder, respectively, where L is its number of Transformer layers, Rvℓ∈Rdv×dv, and
138

Reℓ∈Rde×de. For simplicity, we set dv = de = d. Overall, we develop a H-CLIP framework,
139

and for an input feature map Mℓin the ℓth Transformer layer of CLIP, the right branch produces the
140

adjusted feature map via H-CLIP, ˜Mℓ, formally via:
141

˜Mℓ=
Fℓ(Mℓ; RℓWℓ),
if Rℓ∈RV

Fℓ(Mℓ; RℓWℓ),
s.t. R⊤
ℓRℓ= RℓR⊤
ℓ= I
otherwise,
(5)

where Wℓis a pre-trained weight matrix in ℓth layer of CLIP’s encoder, and Fℓrepresents ℓth layer of
142

CLIP’s encoder. During the fine-tuning phase, H-CLIP is fine-tuned in conjunction with the original
143

parameter space of CLIP, which is loaded from the pre-trained checkpoint and remains frozen.
144

4.3
Dual Cross Relation Communication
145

Although in POF, we relax the orthogonal constraint for CLIP’s image encoder to learn segmentation
146

knowledge, each layer of the image encoder still incorporates a limited number of parameters, which
147

largely restricts the flexibility of the projection adjustment due to the limitation of Hidden Markov
148

4


---Page Break---
Chain along layers [24, 46, 36]. To address this limitation, one might consider fully fine-tuning
149

instead of using a small number of parameters. However, this approach can cause a misalignment
150

between image and text features in CLIP, resulting in sub-optimal performance [52]. Based on
151

the above analysis, we introduce Dual Cross-Relation Communication (DCRC), which facilitates
152

interaction among different layers and modalities (i.e., text and image). DCRC explicitly enhances
153

the flexibility of fine-tuned projection adjustments and prevents misalignment issues.
154

DCRC introduces cross-layer and cross-modality communication among different block-diagonal
155

matrices, achieved through two relation projections. To do this, we first treat all blocks in ℓth layer as
156

an individual slice in this 3-order tensor Tℓ, which is derived as follows:
157

Tℓ= [Rvℓ1, Reℓ1, · · · , Rvℓi, Reℓi, · · · , Rvℓb, Reℓb] ∈Rq×q×(b+b),
(6)

Where q = d/b. Then, we treat the tensor Tℓas an individual slice within a 4-order tensor T , defined
158

as follows:
159

T = [T1, T2, · · · , Tℓ, · · · , TL] ∈Rq×q×(b+b)×L.
(7)

Initially, according to the characteristics of gradient propagation in deep learning theory, i.e., chain
160

rule, each frontal slice R·ℓi ∈{Rq×q}(b+b)×L is updated sequentially in CLIP’s encoder. As a result,
161

updating the T lacks cross-frontal-slice communication, limiting the flexibility of adjusting fine-tuned
162

projection. To address this, we introduce two special tensor products, i.e., 3-order T-product and
163

Higher-order T-product.
164

Definition 4.1(3-order T-product) For A ∈Rn1×n2×n3 and B ∈Rn2×l×n3, the 3-order T-product
165

C ∈Rn1×l××n3 = A ∗B is defined as:
166

C = A ∗B = fold(circ(A) · unfold(B)),
(8)

where “·” represents standard matrix product.
167

Definition 4.2(Higher-order T-product) For A ∈Rn1×n2×n3···×np and B ∈Rn2×l×n3×···×np, the
168

High-order T-product C ∈Rn1×l×n3···×np = A ∗B is defined as:
169

C = A ∗B = fold(circ(A) ∗unfold(B)).
(9)

If A ∈Rn1×n2×n3, according to the 3-order T-product, there is an invertible transform S3(·) :
170

Rn1×n2×n3 →Rn1×n2×n3 in third dimension and it transform the Eq. (8) as:
171

C = S−1
3 (S3(A) ⊙S3(B)) = S−1
3 ( ¯
A ⊙¯B) = S−1
3 ( ¯C),
(10)

where ¯C = ¯
A ⊙¯B denotes the frontal-slice-wise product (Definition 2.1 refers to [19]) ¯C(; , ; , i) =
172

¯
A(; , ; , i) · ¯B(; , ; , i), i = 1, 2, · · · , n3 and S−1
3 (·) is the inverse transform of S3(·). According to the
173

definition of the frontal-slice-wise product, the invertible transform S3(·) is formulated as:
174

¯
A = S3(A) = A ×3 S3,
(11)

where “×3” denotes the mode-3 product and S3 ∈Rn3×n3 is an arbitrary invertible matrix. Similarly,
175

the inverse transform of Eq. (11) is derived as:
176

A = S−1
3 ( ¯A) = ¯
A ×3 S−1
3 .
(12)

Similarly, if A ∈Rn1×n2×···×np, according to the Higer-order T-product, there are invertible
177

transform Si(·) : Rn1×n2×···×np →Rn1×n2×···×np, i = 3, 4, · · · , p in ith dimension and they
178

transform the Eq. (9) as:
179

C = ˜S−1( ˜S(A) ⊙˜S(B)) = ˜S−1( ¯
A ⊙¯B) = ˜S−1( ¯C),
(13)

where ˜S(A) = Sp(Sp−1(· · · S3(A) · · · )), ¯C =
¯
A ⊙¯B denotes the frontal-slice-wise product
180

¯C(; , ; , i) =
¯
A(; , ; , i) · ¯B(; , ; , i), i = 1, 2, · · · , n3n4 · · · np and ˜S−1(·) is the inverse transform
181

of ˜S(·). Similarly, the inverse transform ˜S(·) is formulated as:
182

¯
A = ˜S(A) = A ×3 S3 ×4 S4 · · · ×p Sp,
(14)

and its inverse transform is derived as:
183

A = ˜S−1( ¯
A) = ¯
A ×3 S−1
3
×4 S−1
4
· · · ×p S−1
p .
(15)

5


---Page Break---
Model
VLM
Additional Backbone A-847 PC-459 A-150 PC-59 PAS-20 PAS-20b

Traditional Fine-Tuning
ZS3Net [1]
-
ResNet-101
-
-
-
19.4
38.3
-
LSeg [25]
CLIP ViT-B/32
ResNet-101
-
-
-
-
47.4
-
ZegFormer [8] CLIP ViT-B/16
ResNet-101
4.9
9.1
16.9
42.8
86.2
62.7
ZSseg [51]
CLIP ViT-B/16
ResNet-101
7.0
-
20.5
47.7
88.4
-
OpenSeg [15]
ALIGN
ResNet-101
4.4
7.9
17.5
40.1
-
63.8
OVSeg [30]
CLIP ViT-B/16
ResNet-101c
7.1
11.0
24.8
53.3
92.6
-
ZegCLIP [59]
CLIP ViT-B/16
-
-
-
-
41.2
93.6
-
CAT-Seg [7]
CLIP ViT-B/16
-
12.0
19.0
31.8
57.5
94.6
77.3
Parameter-efficient Fine-Tuning
SAN [50]
CLIP ViT-B/16
-
10.1
12.6
27.5
53.8
94.0
-
Ours
CLIP ViT-B/16
-
12.4
19.3
32.4
57.9
95.2
78.2

Traditional Fine-Tuning
LSeg [25]
CLIP ViT-B/32
ViT-L/16
-
-
-
-
52.3
-
OpenSeg [15]
ALIGN
Eff-B7
8.1
11.5
26.4
44.8
-
70.2
OVSeg [30]
CLIP ViT-L/14
Swin-B
9.0
12.4
29.6
55.7
94.5
-
SAN [50]
CLIP ViT-L/14
-
12.4
15.7
32.1
57.7
94.6
-
ODISE [49]
CLIP ViT-L/14
Stable Diffusion
11.1
14.5
29.9
57.3
-
-
CAT-Seg [7]
CLIP ViT-L/14
-
16.0
23.8
37.9
63.3
97.0
82.5
Parameter-efficient Fine-Tuning
SAN [50]
CLIP ViT-L/14
-
12.4
15.7
32.1
57.7
94.6
-
Ours
CLIP ViT-L/14
-
16.5
24.2
38.4
64.1
97.7
83.2

Table 1: Comparison with state-of-the-art methods on standard benchmarks.
The best-
performing results are presented in bold, while the second-best results are underlined. “VLM”:
visual language model.

Derivation. please refer to supplementary material.
■
184

According to Eqs. (29), (30) and (31), we adopt its idea and design arbitrary invertible relation matrix
185

S3 ∈R(b+b)×(b+b) and S4 ∈RL×L to capture the cross-modality and cross-layer information in T .
186

Then the updated tensor Tw is formulated as:
187

Tw = T ×3 S3 ×4 S4 ∈Rq×q×(b+b)×L,
(16)

where the relation matrix S3 and S4 are learnable. To better capture the nonlinear interactions inside
188

the whole parameter space, we further adopt k layers deep neural network (DNN) f3(·) and f4(·) to
189

replace the transform ×3S3 and ×4S4, respectively, and the DNN f3(·) is formulated as:
190

f3(T ) = σ(σ(· · · σ(σ(A ×3 W1) ×3 W2) · · · ) × Wk−1) × Wk,
(17)

where σ(·) is a nonlinear scalar function and matrices {Wj ∈R(b+b)}k
j=1. The DNN f4(·) is similar.
191

Finally, the T is updated by T = T + αTw, where α ∈R(b+b)×L is a learnable parameter.
192

5
Experiments
193

5.1
Experimental Setup
194

Datasets. Following previous studies [7, 48], we utilizes the COCO-Stuff dataset [2] as our training
195

set. This dataset comprises approximately 118,000 densely annotated images across 171 distinct
196

semantic categories. During inference, we carry out comparisons with state-of-the-art methods across
197

several semantic segmentation datasets, including ADE20K [55], PASCAL VOC [12], and PASCAL-
198

Context [35]. ADE20K [55] is a classical semantic segmentation dataset comprising around 20,000
199

training images and 2,000 validation images. Besides, it includes two different test sets: A-150 and
200

A-847. The test set A-150 has 150 common categories, while the test set A-847 has 847 categories.
201

PASCAL VOC [12] is a small dataset for semantic segmentation, which includes 1464 training
202

images and 1449 validation images. The dataset contains 20 different foreground categories. We
203

name it as PAS-20. In line with [7], we also report a score on PAS-20b, which involves “background”
204

as the 21st category. PASCAL-Context [35] is upgraded from the original PASCAL VOC dataset.
205

It includes two different test sets: PC-59 and PC-459 for evaluation. The test set PC-59 has 59
206

categories, while the test set PC-459 has 459 categories.
207

6


---Page Break---
Image
Ground truth
CAT-Seg [7]
Ours

Figure 2: Comparison of qualitative reults on ADE20K [55] with 150 categories. we compare Our
method with CAT-Seg [7].

Evaluation metric. Following prior works [7, 48], we adopt mean Intersection over Union (mIoU)
208

to evaluate the semantic segmentation performance on the three benchmarks.
209

Implementation Details. We implement our method using the Transformer-based CLIP model.
210

Following the protocol established in [7], we evaluate our results on two versions of the CLIP model:
211

ViT-B/16 and ViT-L/14. For training, we use the Adam optimizer [22] with an initial learning rate
212

of 5 × 10−6 for CLIP, and a weight decay of 10−4. Training is conducted with one image per
213

mini-batch. We set q = 128 for balancing efficiency and performance. The function f3(·) and f4(·)
214

are implemented using two 2-layer MLPs. We act the cost-based approach provided in [7] as our
215

decoder. All models are trained over 80,000 iterations on 4 NVIDIA RTX 3090 GPUs.
216

5.2
Main Results
217

Methods
OVSeg [30] CAT-Seg [7] SAN [50] Ours

Param. (M)
147.2
25.6
8.4
5.6

Table 2: Efficiency comparison in terms of learnable pa-
rameters.

Comparing to SOTAs. Here, we compare
218

our proposed H-CLIP with several state-of-
219

the-art methods, as shown in Table 1, using
220

six test sets across three benchmarks. Over-
221

all, we achieve the best results. Most exist-
222

ing open-vocabulary semantic segmentation
223

methods employ traditional fine-tuning approaches, i.e., full or partial fine-tuning (tuning certain lay-
224

ers of CLIP). While these methods offer sufficient flexibility for learning new knowledge, they often
225

result in a significant performance drop on unseen classes, as observed with OVSeg [30]. Among
226

these methods, CAT-Seg [7] achieves performance comparable to ours. However, its fine-tuning
227

scheme is manually controlled through different layer combinations, necessitating a careful design to
228

balance generalization and flexibility, while ours does not suffer from such an issue. Then, compared
229

to SAN [50], another parameter-efficient fine-tuning method that introduces only a limited number of
230

tunable parameters, our approach significantly outperforms it, achieving improvements of 6.6% on
231

the PC-459 dataset and 3.9% on the PC-59 dataset with ViT-B/16 as the base model. These results
232

demonstrate the effectiveness of our method in preserving generalization while learning segmentation
233

knowledge.
234

Qualitative results. Here, we visualize our method’s representative example segmentation results
235

against prevailing methods, e.g., CAT-Seg [7] in the PC-459 dataset. As shown in Figs. 2 - 4, we
236

observe that our approach is able to generalize on diverse scenarios and produce more accurate
237

results.
238

7


---Page Break---
Image
Ground truth
CAT-Seg [7]
Ours

Figure 3: Comparison of qualitative reults on VOC2010 [12] with 59 categories.

Image
Ground truth
CAT-Seg [7]
Ours

Figure 4: Comparison of qualitative reults on ADE20K [55] with 847 categories.

Method
POF DCRC Param. (M) A-847 PC-459 A-150 PC-59 PAS-20 PAS-20b

Freeze
✗
✗
0
4.4
6.6
24.8
49.4
92.5
71.9
LoRA [16]
✗
✗
7.5
11.4
17.6
28.6
55.1
94.2
76.7

✓
✗
5.62
12.3
19.0
31.6
56.4
94.6
76.3
H-CLIP
✗
✓
0.01
7.6
10.9
26.8
53.6
92.7
74.5
✓
✓
5.63
12.4
19.3
32.4
57.9
95.2
78.2

Table 3: Ablation study on the components of H-CLIP. “LoRA”: a mainstream parameter-efficient
tuning method with a comparable number of parameters for comparison. “POF”: Partial Orthogonal
Fine-tuning. “DCRC”: Dual Cross Relation Communication. The base model is ViT-B/16.

Efficiency comparison. We compare the efficiency of our method with other approaches, including
239

OVSeg [30], CAT-Seg [7], and SAN [50], all of which utilize CLIP ViT models. The comparison,
240

8


---Page Break---
Block dimension q
Param. (M) A-847 PC-459 A-150 PC-59 PAS-20 PAS-20b

256 × 256
22.52
12.4
19.2
32.7
57.6
95.4
77.9
(a)
128 × 128
5.63
12.4
19.3
32.4
57.9
95.2
78.2
64 × 64
1.41
11.7
18.4
31.7
56.9
95.0
76.4

Orthogonal Constraint Param. (M) A-847 PC-459 A-150 PC-59 PAS-20 PAS-20b

w/o
7.51
11.9
18.5
32.2
57.5
95.3
76.9
(b)
with
3.76
12.2
19.1
31.4
57.1
94.3
76.8
POF
5.63
12.4
19.3
32.4
57.9
95.2
78.2

Table 4: Ablation study on different designs in POF. We show the impact of (a) different block
dimensions q and (b) orthogonal constraints. The base model is ViT-B/16.

summarized in Table 2, shows that our method employs the fewest trainable parameters while
241

balancing the generalization of the pre-trained model and the flexibility for learning new knowledge.
242

Additionally, since we introduce a lightweight architecture for calculating relations, specifically two
243

relation matrices, the inference overhead is negligible during the inference phase.
244

5.3
Ablative Studies
245

Ablation of Main Components. Here, we conduct an ablation study to demonstrate the benefits
246

of each component of our proposed H-CLIP: partial orthogonal fine-tuning (POF) and dual cross-
247

relation communication (DCRC). We use the ViT-B/16 [9] version of CLIP as the baseline, shown in
248

row 1 of Table 3. Additionally, we implement a mainstream parameter-efficient fine-tuning (PEFT)
249

method, LoRA [16], for comparison with a similar number of learnable parameters, as shown in row
250

2. Note that LoRA can improve performance compared to the baseline, demonstrating that PEFT is a
251

viable approach for this task. Then, comparing row 5 to row 2, we observe significant performance
252

gains, indicating that our results are driven by our targeted solution rather than merely the number of
253

parameters. Moreover, row 3 shows that using only POF preserves generalization on unseen classes,
254

particularly in the A-847 dataset. Meanwhile, solely adapting DCRC shows limited improvement, as
255

it only enhances communication among frozen weight matrices. Finally, integrating DCRC with POF
256

yields clear performance gains, e.g., a 12.6% improvement on the PC-459 dataset.
257

Different Design of POF. Table 4 presents experiments introducing different designs into POF. The
258

design of POF is related to (1) block dimension, i.e., q, and (2) how orthogonality constraints are
259

applied. In (a), the results show that larger In (a), the results show that larger q generally performs
260

better than smaller q. However, we find a good trade-off between performance and parameter
261

efficiency, with q = 128 working well across datasets and tasks. Therefore, we maintain this setting
262

in other experiments. In (b), we show that both blindly applying orthogonality constraints to the
263

learnable matrices of all layers and not using any constraints at all can degrade performance on most
264

test sets, demonstrating the value of our analysis with the hyperspherical energy principle.
265

6
Conclusion
266

In this paper, we propose a H-CLIP framework to address three issues: 1) high computational cost, 2)
267

misalignment between the two inherent modalities of CLIP, and 3) degraded generalization ability
268

on unseen categories when equipping CLIP with pixel-level prediction ability for open-vocabulary
269

semantic segmentation. Specifically, we propose a symmetrical parameter-efficient fine-tuning (PEFT)
270

strategy conducted in hyperspherical space for both of the two CLIP modalities. Specifically, the
271

PEFT strategy is achieved by a series of efficient block-diagonal learnable transformation matrices and
272

a dual cross-relation communication module among all learnable matrices to mitigate misalignment
273

between different modalities. Furthermore, we apply an additional constraint to PEFT on the CLIP
274

text encoder according to the hyperspherical energy principle, i.e., minimizing hyperspherical energy
275

during fine-tuning preserves the intrinsic structure of the original parameter space, to prevent the
276

destruction of the generalization ability offered by the CLIP text encoder. Extensive experiments
277

demonstrate that the proposed H-CLIP framework generalized improves segmentation performance
278

across several benchmarks while introducing approximately 4% of CLIP’s total parameters. We hope
279

our approach will provide a new direction and inspire future research in this field.
280

9


---Page Break---
References
281

[1] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick Pérez.
Zero-shot semantic
282

segmentation. Advances in Neural Information Processing Systems, 32, 2019.
283

[2] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in
284

context. In Proceedings of the IEEE conference on computer vision and pattern recognition,
285

pages 1209–1218, 2018.
286

[3] Arthur Cayley. Sur quelques propriétés des déterminants gauches. 1846.
287

[4] Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Shangzhan Zhang, Yan Wang, Zejian
288

Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam fails to segment anything? – sam-adapter:
289

Adapting sam in underperformed scenes: Camouflage, shadow, and more, 2023.
290

[5] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng,
291

and Jingjing Liu. Uniter: Universal image-text representation learning. In European conference
292

on computer vision, pages 104–120. Springer, 2020.
293

[6] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar.
294

Masked-attention mask transformer for universal image segmentation. In Proceedings of the
295

IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.
296

[7] Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seun-
297

gryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation, 2024.
298

[8] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmenta-
299

tion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
300

pages 11583–11592, 2022.
301

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
302

Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
303

An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
304

arXiv:2010.11929, 2020.
305

[10] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng
306

Liu, Ce Liu, Yann LeCun, Nanyun Peng, et al. Coarse-to-fine vision-language pre-training with
307

fusion in the backbone. Advances in neural information processing systems, 35:32942–32956,
308

2022.
309

[11] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang
310

Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end
311

vision-and-language transformers. In Proceedings of the IEEE/CVF Conference on Computer
312

Vision and Pattern Recognition, pages 18166–18176, 2022.
313

[12] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman.
314

The pascal visual object classes (voc) challenge. International journal of computer vision,
315

88:303–338, 2010.
316

[13] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Large-scale
317

adversarial training for vision-and-language representation learning. Advances in Neural
318

Information Processing Systems, 33:6616–6628, 2020.
319

[14] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li,
320

and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International
321

Journal of Computer Vision, 132(2):581–595, 2024.
322

[15] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmen-
323

tation with image-level labels. In European Conference on Computer Vision, pages 540–557.
324

Springer, 2022.
325

[16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
326

Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In
327

International Conference on Learning Representations, 2022.
328

10


---Page Break---
[17] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu.
329

Seeing out of the box: End-to-end pre-training for vision-language representation learning. In
330

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
331

12976–12985, 2021.
332

[18] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas
333

Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings
334

of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021.
335

[19] Eric Kernfeld, Misha Kilmer, and Shuchin Aeron. Tensor–tensor products with invertible linear
336

transforms. Linear Algebra and its Applications, 485:545–570, 2015.
337

[20] Misha E Kilmer and Carla D Martin. Factorization strategies for third-order tensors. Linear
338

Algebra and its Applications, 435(3):641–658, 2011.
339

[21] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without
340

convolution or region supervision. In International conference on machine learning, pages
341

5583–5594. PMLR, 2021.
342

[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
343

arXiv:1412.6980, 2014.
344

[23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
345

Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In
346

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026,
347

2023.
348

[24] Jungbeom Lee, Jooyoung Choi, Jisoo Mok, and Sungroh Yoon. Reducing information bottleneck
349

for weakly supervised semantic segmentation. Advances in Neural Information Processing
350

Systems, 34:27408–27421, 2021.
351

[25] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-
352

driven semantic segmentation. In International Conference on Learning Representations,
353

2022.
354

[26] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
Blip: Bootstrapping language-
355

image pre-training for unified vision-language understanding and generation. In International
356

conference on machine learning, pages 12888–12900. PMLR, 2022.
357

[27] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven
358

Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum
359

distillation. Advances in neural information processing systems, 34:9694–9705, 2021.
360

[28] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu
361

Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image
362

pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
363

Recognition, pages 10965–10975, 2022.
364

[29] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang,
365

Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for
366

vision-language tasks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow,
367

UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 121–137. Springer, 2020.
368

[30] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang,
369

Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted
370

clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
371

pages 7061–7070, 2023.
372

[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances
373

in neural information processing systems, 36, 2024.
374

[32] Weiyang Liu, Rongmei Lin, Zhen Liu, Lixin Liu, Zhiding Yu, Bo Dai, and Le Song. Learning
375

towards minimum hyperspherical energy. Advances in neural information processing systems,
376

31, 2018.
377

11


---Page Break---
[33] Canyi Lu, Xi Peng, and Yunchao Wei. Low-rank tensor completion with a new tensor nuclear
378

norm induced by invertible linear transforms. In Proceedings of the IEEE/CVF conference on
379

computer vision and pattern recognition, pages 5996–6004, 2019.
380

[34] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic
381

visiolinguistic representations for vision-and-language tasks. Advances in neural information
382

processing systems, 32, 2019.
383

[35] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler,
384

Raquel Urtasun, and Alan Yuille.
The role of context for object detection and semantic
385

segmentation in the wild. In Proceedings of the IEEE conference on computer vision and
386

pattern recognition, pages 891–898, 2014.
387

[36] Zelin Peng, Zhengqin Xu, Zhilin Zeng, Lingxi Xie, Qi Tian, and Wei Shen. Parameter efficient
388

fine-tuning via cross block orchestration for segment anything model. In IEEE/CVF Conference
389

on Computer Vision and Pattern Recognition (CVPR), 2024.
390

[37] Zelin Peng, Zhengqin Xu, Zhilin Zeng, Xiaokang Yang, and Wei Shen. Sam-parser: Fine-tuning
391

sam efficiently by parameter space reconstruction. In Proceedings of the AAAI Conference on
392

Artificial Intelligence, volume 38, pages 4515–4523, 2024.
393

[38] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian
394

Weller, and Bernhard Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning.
395

Advances in Neural Information Processing Systems, 36:79320–79362, 2023.
396

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
397

Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
398

models from natural language supervision. In International conference on machine learning,
399

pages 8748–8763. PMLR, 2021.
400

[40] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan,
401

Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models
402

for code. arXiv preprint arXiv:2308.12950, 2023.
403

[41] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba,
404

Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment
405

model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
406

tion, pages 15638–15650, 2022.
407

[42] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert:
408

Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530,
409

2019.
410

[43] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from
411

transformers. arXiv preprint arXiv:1908.07490, 2019.
412

[44] JJ Tomson. On the structure of the atom: an investigation of the stability and periods of
413

osciletion of a number of corpuscles arranged at equal intervals around the circumference of
414

a circle; with application of the results to the theory atomic structure. Philos. Mag. Series 6,
415

7(39):237, 1904.
416

[45] Jianfeng Wang, Xiaowei Hu, Zhe Gan, Zhengyuan Yang, Xiyang Dai, Zicheng Liu, Yumao
417

Lu, and Lijuan Wang. Ufo: A unified transformer for vision-language representation learning.
418

arXiv preprint arXiv:2111.10023, 2021.
419

[46] Zifeng Wang, Xi Chen, Rui Wen, Shao-Lun Huang, Ercan Kuruoglu, and Yefeng Zheng.
420

Information theoretic counterfactual learning from missing-not-at-random feedback. Advances
421

in Neural Information Processing Systems, 33:1854–1864, 2020.
422

[47] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-llama: Further
423

finetuning llama on medical papers. arXiv preprint arXiv:2304.14454, 2023.
424

[48] Bin Xie, Jiale Cao, Jin Xie, Fahad Shahbaz Khan, and Yanwei Pang. Sed: A simple encoder-
425

decoder for open-vocabulary semantic segmentation, 2023.
426

12


---Page Break---
[49] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello.
427

Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings
428

of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966,
429

2023.
430

[50] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network
431

for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on
432

Computer Vision and Pattern Recognition, pages 2945–2954, 2023.
433

[51] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A
434

simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language
435

model. In European Conference on Computer Vision, pages 736–753. Springer, 2022.
436

[52] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Convolutions die
437

hard: Open-vocabulary segmentation with single frozen convolutional clip. In NeurIPS, 2023.
438

[53] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Li, Xiyang Dai,
439

Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization
440

and vision-language understanding. Advances in Neural Information Processing Systems,
441

35:36067–36080, 2022.
442

[54] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan
443

Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with
444

zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
445

[55] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio
446

Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal
447

of Computer Vision, 127:302–321, 2019.
448

[56] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European
449

Conference on Computer Vision, pages 696–712. Springer, 2022.
450

[57] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learn-
451

ing for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern
452

Recognition (CVPR), 2022.
453

[58] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for
454

vision-language models. International Journal of Computer Vision (IJCV), 2022.
455

[59] Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, and Yifan Liu.
Zegclip: Towards
456

adapting clip for zero-shot semantic segmentation. Proceedings of the IEEE/CVF Conference
457

on Computer Vision and Pattern Recognition (CVPR), 2023.
458

[60] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: En-
459

hancing vision-language understanding with advanced large language models. arXiv preprint
460

arXiv:2304.10592, 2023.
461

[61] Xiangyang Zhu, Renrui Zhang, Bowei He, Aojun Zhou, Dong Wang, Bin Zhao, and Peng Gao.
462

Not all features matter: Enhancing few-shot clip with adaptive prior refinement. In Proceedings
463

of the IEEE/CVF International Conference on Computer Vision, pages 2605–2615, 2023.
464

13


---Page Break---
465

Appendix of H-CLIP

A
Derivation of the Definition
466

In this section, we provide derivations of definitions in the main paper.
467

Definition 4.1(3-order T-product) For A ∈Rn1×n2×n3 and B ∈Rn2×l×n3, the 3-order T-product
468

C ∈Rn1×l××n3 = A ∗B is defined as:
469

C = A ∗B = fold(circ(A) · unfold(B)),
(18)

where “·” represents standard matrix product.
470

Definition 4.2(Higher-order T-product) For A ∈Rn1×n2×n3···×np and B ∈Rn2×l×n3×···×np, the
471

High-order T-product C ∈Rn1×l×n3···×np = A ∗B is defined as:
472

C = A ∗B = fold(circ(A) ∗unfold(B)).
(19)

Derivation. According to [20], if A is n1 × n2 × n3, A can be block diagonalized by using Discrete
473

Fourier Transformer (DFT) matrix Fn3 ∈Rn3×n3 as:
474

(Fn3 ⊗In1) · circ(unfold(A)) · (F∗
n3 ⊗In2) = D =





D1
...
Dn3



∈Rn1n3×n2n3,
(20)

where “⊗” denotes the Kernecker product, “F∗
n3” denotes the conjugate transpose of Fn3, “·” means
475

standard matrix product and D is a block diagonal matrix. In fact, the i-th block matrix Di of D can
476

be computed by applying DFT of A along 3-rd dimension. The 3-order T-product in Eq. (18) can
477

be computed as:
478

(F∗
n3 ⊗In1) · ((Fn3 ⊗In1) · circ(unfold(A)) · (F∗
n3 ⊗In2)) · (Fn3 ⊗In2) · unfold(B).
(21)

It is readily shown that (Fn3 ⊗In2)unfold can be computed by applying DFT of B along 3-rd
479

dimension: the result called ¯B. Thus, Eq. (21) remains to multiply each block matrix Di of D with
480

each block matrix Bi of ¯B, then take an inverse DFT along the 3-rd dimension of the result. Hence,
481

the 3-order T-product in Eq. (18) can be re-formulated as:
482

C = DFT−1
3 (DFT3(A) ⊙DFT3(B)) = DFT−1
3 ( ¯
A ⊙¯B) = DFT−1
3 ( ¯C),
(22)

where DFT3(·) is DFT along 3-rd dimension and DFT−1
3 (·) is the inverse DFT along 3-rd dimension.
483

In mathematics, the DFT of A along 3-rd dimension is formulated as:
484

¯
A = DFT3(A) = A ×3 Fn3.
(23)

Similarly, the inverse DFT of ¯
A along 3-rd dimension is derived as:
485

A = DFT−1
3 ( ¯
A) = ¯
A ×3 F−1
n3 .
(24)

By the detailed theoretical analysis in [33], the DFT has been extended to a general invertible
486

transform S with an invertible transform matrix S. In mathematics, the invertible transform of A
487

along 3-rd dimension is formulated as:
488

¯
A = S3(A) = A ×3 Sn3.
(25)

Similarly, the inverse transform of ¯
A along 3-rd dimension is derived as:
489

A = S−1
3 ( ¯
A) = ¯
A ×3 S−1
n3 .
(26)

Similarly, if A ∈Rn1×n2×···×np, A can be block diagonalized by using a sequence of DFT matrices
490

Fni ∈Rni×ni, i = 3, 4, ·, p as:
491

(Fnp ⊗Fnp−1 ⊗· · · ⊗Fn3 ⊗In1) · ˜
A · (F∗
np ⊗F∗
np−1 ⊗· · · ⊗F∗
n3 ⊗In2) = D,
(27)

14


---Page Break---
where ˜
A = circ(unfold(A)) ∈Rn1n3n4···np×n2n3···np. Since the matrix D is block diagonal with
492

n3n4 · · · np blocks each of size n1 × n2, the Higher-order T-product in Eq. (19) can be computed
493

as:
494

(˜F∗⊗In1) · ((˜F ⊗In1) · ˜
A · (˜F∗⊗In2)) · (˜Fn3 ⊗In2) · ˜B,
(28)

where ˜F = Fnp ⊗Fnp−1 ⊗· · · ⊗Fn3. Using the DEF, it is straightforward to show that the block
495

diagonal matrix D in Eq. (27) can be obtained by repeated DFTs of A along each dimension expect
496

for 1-st and 2-nd dimension. Similarly, by using a sequence invertible transform Sj(·), i = 3, 4, ·, p
497

with invertible transform matrix Si, the Higher-order T-product in Eq. (19) can be re-formulated as:
498

C = ˜S−1( ˜S(A) ⊙˜S(B)) = ˜S−1( ¯
A ⊙¯B) = ˜S−1( ¯C),
(29)

where ˜S(A) = Sp(Sp−1(· · · S3(A) · · · )), ¯C =
¯
A ⊙¯B denotes the frontal-slice-wise product
499

¯C(; , ; , i) =
¯
A(; , ; , i) · ¯B(; , ; , i), i = 1, 2, · · · , n3n4 · · · np and ˜S−1(·) is the inverse transform
500

of ˜S(·). The inverse transform ˜S(·) is formulated as:
501

¯
A = ˜S(A) = A ×3 S3 ×4 S4 · · · ×p Sp,
(30)

and its inverse transform is derived as:
502

A = ˜S−1( ¯
A) = ¯
A ×3 S−1
3
×4 S−1
4
· · · ×p S−1
p .
(31)

■
503

15


---Page Break---
NeurIPS Paper Checklist
504

1. Claims
505

Question: Do the main claims made in the abstract and introduction accurately reflect the
506

paper’s contributions and scope?
507

Answer: [Yes]
508

Justification: The abstract clearly states the following claims about the paper’s contribu-
509

tions and scope. We aim to address a significant challenge in open-vocabulary semantic
510

segmentation: fine-tuning CLIP to achieve per-pixel predictions without compromising
511

its generalization capabilities. To this end, we propose a novel H-CLIP, which introduces
512

a partial orthogonal fine-tuning strategy that prevents a drop in hyperspherical energy,
513

thereby preserving generalization. Subsequently, H-CLIP employs dual cross-relation com-
514

munication to increase projection flexibility, facilitating the acquisition of segmentation
515

knowledge.
516

2. Limitations
517

Question: Does the paper discuss the limitations of the work performed by the authors?
518

Answer: [NA]
519

Justification: Those are not discussed in the paper.
520

Guidelines:
521

• The answer NA means that the paper has no limitation while the answer No means that
522

the paper has limitations, but those are not discussed in the paper.
523

• The authors are encouraged to create a separate "Limitations" section in their paper.
524

• The paper should point out any strong assumptions and how robust the results are to
525

violations of these assumptions (e.g., independence assumptions, noiseless settings,
526

model well-specification, asymptotic approximations only holding locally). The authors
527

should reflect on how these assumptions might be violated in practice and what the
528

implications would be.
529

• The authors should reflect on the scope of the claims made, e.g., if the approach was
530

only tested on a few datasets or with a few runs. In general, empirical results often
531

depend on implicit assumptions, which should be articulated.
532

• The authors should reflect on the factors that influence the performance of the approach.
533

For example, a facial recognition algorithm may perform poorly when image resolution
534

is low or images are taken in low lighting. Or a speech-to-text system might not be
535

used reliably to provide closed captions for online lectures because it fails to handle
536

technical jargon.
537

• The authors should discuss the computational efficiency of the proposed algorithms
538

and how they scale with dataset size.
539

• If applicable, the authors should discuss possible limitations of their approach to
540

address problems of privacy and fairness.
541

• While the authors might fear that complete honesty about limitations might be used by
542

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
543

limitations that aren’t acknowledged in the paper. The authors should use their best
544

judgment and recognize that individual actions in favor of transparency play an impor-
545

tant role in developing norms that preserve the integrity of the community. Reviewers
546

will be specifically instructed to not penalize honesty concerning limitations.
547

3. Theory Assumptions and Proofs
548

Question: For each theoretical result, does the paper provide the full set of assumptions and
549

a complete (and correct) proof?
550

Answer: [Yes]
551

Justification: All assumptions are proved clearly in the main paper or the supplemental
552

material.
553

Guidelines:
554

• The answer NA means that the paper does not include theoretical results.
555

16


---Page Break---
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
556

referenced.
557

• All assumptions should be clearly stated or referenced in the statement of any theorems.
558

• The proofs can either appear in the main paper or the supplemental material, but if
559

they appear in the supplemental material, the authors are encouraged to provide a short
560

proof sketch to provide intuition.
561

• Inversely, any informal proof provided in the core of the paper should be complemented
562

by formal proofs provided in appendix or supplemental material.
563

• Theorems and Lemmas that the proof relies upon should be properly referenced.
564

4. Experimental Result Reproducibility
565

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
566

perimental results of the paper to the extent that it affects the main claims and/or conclusions
567

of the paper (regardless of whether the code and data are provided or not)?
568

Answer: [Yes]
569

Justification: The detailed experimental settings and information are provided in Sec. 5, and
570

this information is sufficient to reproduce the main experimental results.
571

Guidelines:
572

• The answer NA means that the paper does not include experiments.
573

• If the paper includes experiments, a No answer to this question will not be perceived
574

well by the reviewers: Making the paper reproducible is important, regardless of
575

whether the code and data are provided or not.
576

• If the contribution is a dataset and/or model, the authors should describe the steps taken
577

to make their results reproducible or verifiable.
578

• Depending on the contribution, reproducibility can be accomplished in various ways.
579

For example, if the contribution is a novel architecture, describing the architecture fully
580

might suffice, or if the contribution is a specific model and empirical evaluation, it may
581

be necessary to either make it possible for others to replicate the model with the same
582

dataset, or provide access to the model. In general. releasing code and data is often
583

one good way to accomplish this, but reproducibility can also be provided via detailed
584

instructions for how to replicate the results, access to a hosted model (e.g., in the case
585

of a large language model), releasing of a model checkpoint, or other means that are
586

appropriate to the research performed.
587

• While NeurIPS does not require releasing code, the conference does require all submis-
588

sions to provide some reasonable avenue for reproducibility, which may depend on the
589

nature of the contribution. For example
590

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
591

to reproduce that algorithm.
592

(b) If the contribution is primarily a new model architecture, the paper should describe
593

the architecture clearly and fully.
594

(c) If the contribution is a new model (e.g., a large language model), then there should
595

either be a way to access this model for reproducing the results or a way to reproduce
596

the model (e.g., with an open-source dataset or instructions for how to construct
597

the dataset).
598

(d) We recognize that reproducibility may be tricky in some cases, in which case
599

authors are welcome to describe the particular way they provide for reproducibility.
600

In the case of closed-source models, it may be that access to the model is limited in
601

some way (e.g., to registered users), but it should be possible for other researchers
602

to have some path to reproducing or verifying the results.
603

5. Open access to data and code
604

Question: Does the paper provide open access to the data and code, with sufficient instruc-
605

tions to faithfully reproduce the main experimental results, as described in supplemental
606

material?
607

Answer: [No]
608

Justification: The code will be made public after acceptance.
609

17


---Page Break---
Guidelines:
610

• The answer NA means that paper does not include experiments requiring code.
611

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
612

public/guides/CodeSubmissionPolicy) for more details.
613

• While we encourage the release of code and data, we understand that this might not be
614

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
615

including code, unless this is central to the contribution (e.g., for a new open-source
616

benchmark).
617

• The instructions should contain the exact command and environment needed to run to
618

reproduce the results. See the NeurIPS code and data submission guidelines (https:
619

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
620

• The authors should provide instructions on data access and preparation, including how
621

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
622

• The authors should provide scripts to reproduce all experimental results for the new
623

proposed method and baselines. If only a subset of experiments are reproducible, they
624

should state which ones are omitted from the script and why.
625

• At submission time, to preserve anonymity, the authors should release anonymized
626

versions (if applicable).
627

• Providing as much information as possible in supplemental material (appended to the
628

paper) is recommended, but including URLs to data and code is permitted.
629

6. Experimental Setting/Details
630

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
631

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
632

results?
633

Answer: [Yes]
634

Justification: All the experimental settings are clarified in Sec. 5. (and Appendix).
635

Guidelines:
636

• The answer NA means that the paper does not include experiments.
637

• The experimental setting should be presented in the core of the paper to a level of detail
638

that is necessary to appreciate the results and make sense of them.
639

• The full details can be provided either with the code, in appendix, or as supplemental
640

material.
641

7. Experiment Statistical Significance
642

Question: Does the paper report error bars suitably and correctly defined or other appropriate
643

information about the statistical significance of the experiments?
644

Answer: [No]
645

Justification: error bars are not reported because it would be too computationally expensive.
646

Guidelines:
647

• The answer NA means that the paper does not include experiments.
648

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
649

dence intervals, or statistical significance tests, at least for the experiments that support
650

the main claims of the paper.
651

• The factors of variability that the error bars are capturing should be clearly stated (for
652

example, train/test split, initialization, random drawing of some parameter, or overall
653

run with given experimental conditions).
654

• The method for calculating the error bars should be explained (closed form formula,
655

call to a library function, bootstrap, etc.)
656

• The assumptions made should be given (e.g., Normally distributed errors).
657

• It should be clear whether the error bar is the standard deviation or the standard error
658

of the mean.
659

• It is OK to report 1-sigma error bars, but one should state it. The authors should
660

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
661

of Normality of errors is not verified.
662

18


---Page Break---
• For asymmetric distributions, the authors should be careful not to show in tables or
663

figures symmetric error bars that would yield results that are out of range (e.g. negative
664

error rates).
665

• If error bars are reported in tables or plots, The authors should explain in the text how
666

they were calculated and reference the corresponding figures or tables in the text.
667

8. Experiments Compute Resources
668

Question: For each experiment, does the paper provide sufficient information on the com-
669

puter resources (type of compute workers, memory, time of execution) needed to reproduce
670

the experiments?
671

Answer: [Yes]
672

Justification: The computing requirements are provided in Sec. 5.
673

• The answer NA means that the paper does not include experiments.
674

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
675

or cloud provider, including relevant memory and storage.
676

• The paper should provide the amount of compute required for each of the individual
677

experimental runs as well as estimate the total compute.
678

• The paper should disclose whether the full research project required more compute
679

than the experiments reported in the paper (e.g., preliminary or failed experiments that
680

didn’t make it into the paper).
681

9. Code Of Ethics
682

Question: Does the research conducted in the paper conform, in every respect, with the
683

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
684

Answer: [Yes]
685

Justification: We ensure our research adheres to the guidelines.
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

Answer: [NA]
695

Justification: there is no societal impact of the work performed.
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

19


---Page Break---
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

Justification: the paper poses no such risks.
725

Guidelines:
726

• The answer NA means that the paper poses no such risks.
727

• Released models that have a high risk for misuse or dual-use should be released with
728

necessary safeguards to allow for controlled use of the model, for example by requiring
729

that users adhere to usage guidelines or restrictions to access the model or implementing
730

safety filters.
731

• Datasets that have been scraped from the Internet could pose safety risks. The authors
732

should describe how they avoided releasing unsafe images.
733

• We recognize that providing effective safeguards is challenging, and many papers do
734

not require this, but we encourage authors to take this into account and make a best
735

faith effort.
736

12. Licenses for existing assets
737

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
738

the paper, properly credited and are the license and terms of use explicitly mentioned and
739

properly respected?
740

Answer: [NA]
741

Justification: All the assets used are properly credited and are the license and terms of use
742

explicitly mentioned and properly respected.
743

• The answer NA means that the paper does not use existing assets.
744

• The authors should cite the original paper that produced the code package or dataset.
745

• The authors should state which version of the asset is used and, if possible, include a
746

URL.
747

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
748

• For scraped data from a particular source (e.g., website), the copyright and terms of
749

service of that source should be provided.
750

• If assets are released, the license, copyright information, and terms of use in the
751

package should be provided. For popular datasets, paperswithcode.com/datasets
752

has curated licenses for some datasets. Their licensing guide can help determine the
753

license of a dataset.
754

• For existing datasets that are re-packaged, both the original license and the license of
755

the derived asset (if it has changed) should be provided.
756

• If this information is not available online, the authors are encouraged to reach out to
757

the asset’s creators.
758

13. New Assets
759

Question: Are new assets introduced in the paper well documented and is the documentation
760

provided alongside the assets?
761

Answer: [NA]
762

Justification: The paper does not release new assets.
763

• The answer NA means that the paper does not release new assets.
764

• Researchers should communicate the details of the dataset/code/model as part of their
765

submissions via structured templates. This includes details about training, license,
766

limitations, etc.
767

20


---Page Break---
• The paper should discuss whether and how consent was obtained from people whose
768

asset is used.
769

• At submission time, remember to anonymize your assets (if applicable). You can either
770

create an anonymized URL or include an anonymized zip file.
771

14. Crowdsourcing and Research with Human Subjects
772

Question: For crowdsourcing experiments and research with human subjects, does the paper
773

include the full text of instructions given to participants and screenshots, if applicable, as
774

well as details about compensation (if any)?
775

Answer: [NA]
776

Justification: This paper does not involve crowdsourcing nor research with human subjects.
777

• The answer NA means that the paper does not involve crowdsourcing nor research with
778

human subjects.
779

• Including this information in the supplemental material is fine, but if the main contribu-
780

tion of the paper involves human subjects, then as much detail as possible should be
781

included in the main paper.
782

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
783

or other labor should be paid at least the minimum wage in the country of the data
784

collector.
785

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
786

Subjects
787

Question: Does the paper describe potential risks incurred by study participants, whether
788

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
789

approvals (or an equivalent approval/review based on the requirements of your country or
790

institution) were obtained?
791

Answer: [NA]
792

Justification: This paper does not involve crowdsourcing nor research with human subjects.
793

Guidelines:
794

• The answer NA means that the paper does not involve crowdsourcing nor research with
795

human subjects.
796

• Depending on the country in which research is conducted, IRB approval (or equivalent)
797

may be required for any human subjects research. If you obtained IRB approval, you
798

should clearly state this in the paper.
799

• We recognize that the procedures for this may vary significantly between institutions
800

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
801

guidelines for their institution.
802

• For initial submissions, do not include any information that would break anonymity (if
803

applicable), such as the institution conducting the review.
804

21


---Page Break---
