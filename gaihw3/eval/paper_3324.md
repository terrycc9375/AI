Training Dynamics of Transformers to Recognize
Word Co-occurrence via Gradient Flow Analysis

Hongru Yang∗
The University of Texas at Austin
& Princeton University
hy6385@utexas.edu

Bhavya Kailkhura
Lawrence Livermore National Laboratory
kailkhura1@llnl.gov

Zhangyang Wang
The University of Texas at Austin
atlaswang@utexas.edu

Yingbin Liang
The Ohio State University
liang.889@osu.edu

Abstract

Understanding the training dynamics of transformers is important to explain the
impressive capabilities behind large language models. In this work, we study the
dynamics of training a shallow transformer on a task of recognizing co-occurrence
of two designated words. In the literature of studying training dynamics of trans-
formers, several simplifications are commonly adopted such as weight reparameter-
ization, attention linearization, special initialization, and lazy regime. In contrast,
we analyze the gradient flow dynamics of simultaneously training three attention
matrices and a linear MLP layer from random initialization, and provide a frame-
work of analyzing such dynamics via a coupled dynamical system. We establish
near minimum loss and characterize the attention model after training. We discover
that gradient flow serves as an inherent mechanism that naturally divide the training
process into two phases. In Phase 1, the linear MLP quickly aligns with the two tar-
get signals for correct classification, whereas the softmax attention remains almost
unchanged. In Phase 2, the attention matrices and the MLP evolve jointly to enlarge
the classification margin and reduce the loss to a near minimum value. Technically,
we prove a novel property of the gradient flow, termed automatic balancing of
gradients, which enables the loss values of different samples to decrease almost at
the same rate and further facilitates the proof of near minimum training loss. We
also conduct experiments to verify our theoretical results.

1
Introduction

Ever since the invention of self-attention [VSP+17], transformers have become a dominat-
ing backbone architecture in many machine learning applications such as computer vision
[DBK+20, LLC+21] and natural language processing [DCLT18].
Nowadays, ChatGPT and
GPT-4 [Ope23] have demonstrated astonishing abilities in many areas such as language under-
standing, mathematics and coding, which have sparked artificial general intelligence [BCE+23].
In the meantime, there has been a burgeoning development of large language models (LLMs)
[TLI+23, MH23, ADF+23] as well as multi-modal models [Tea23].

Despite the huge empirical success, theoretical understanding of why a pre-trained language model
can possess such impressive performance has been significantly lagging behind. Some previous

∗Work done while doing a internship at Lawrence Livermore National Laboratory and visiting Princeton
University.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
efforts have been made in understanding the capacity and representational power of transformers
[EGKZ22, LAG+22, ZPGA23, SHT23, BCW+23]. However, most results of this type of works are
existential and rely on manual construction of the weights. It is unclear whether the constructed
weights are the actual solutions after training transformers. In order to understand the mechanism
behind those pre-trained language models, a line of studies have aimed to open the black box of
optimization via studying training dynamics of transformers and explaining why transformers can be
trained to perform well [JSL22, LWLC22, TWCD23, TWZ+23, LLR23, TLZO23, TLTO23, ZFB24,
HCL23]. However, those previous works often relied on various simplifications in their analysis such
as weight reparameterization, attention linearization, special initialization, lazy regime, etc. One goal
of this paper is to take a further step to demystify the training dynamics of transformers and consider
more practical training setup, thus better capturing the actual training process.

Our study of transformers’ training dynamics will focus on a basic problem of recognizing co-
occurrence of words under a binary classification setup, which is an important ability of LLMs
to perform many tasks correctly in natural language processing (NLP). For example, the classical
n-gram model [MS99, Dam18] predicts the next word based on co-occurrence of multiple words.
Consider the following scenario: if the task for the language model is to read a paragraph describing
a children and then answer some questions, say, “Is Bob eating a banana?”. In order to answer the
question correctly, the model must be able to detect the co-occurrence of the two words “Bob” and
“banana” in the paragraph. Motivated by this, we study the problem of detecting co-occurrence of
two target words via the model of a one-layer transformer with a self-attention module followed by a
linear multi-layer perceptrons (MLP) layer. Our goal is to characterize the dynamics of the training
process via the gradient flow analysis, thus providing a theory to explain how transformers can be
trained to perform well.

Our contribution is summarized below:

• We study the gradient flow dynamics of detecting word co-occurrence. The training starts with
random initialization and then simultaneously updates four weight matrices (including key, query,
and value matrices and a linear MLP) in the transformer architecture via gradient flow. We show
that gradient flow can achieve small loss although the loss function is highly nonconvex. We
further characterize the explicit form of attention matrices after training, which captures the strong
positive correlation between the two target signals and strong negative correlation between one
target signal and the common token, both leading to large classification margin.
• We characterize the training process into two phases. In Phase 1 (alignment of MLP for correct
classification), we show that the linear MLP of the transformer quickly aligns with the two target
word tokens whereas all other variables in the dynamical system stay almost unchanged from their
initialization values. All training samples are correctly classified at the end of Phase 1, but the loss
value is still large due to small classification margin. In Phase 2 (evolution of attention and MLP
for large classification margin), along with the continual evolution of MLP, correct classification
by MLP also encourages the gradients of attention matrices to learn. Specifically, the softmax
probability increases if the key and query tokens correspond to the two target words, and the
value transform of the two words becomes more positively correlated, both leading to enlarge the
classification margin. Thus, the training and test loss values both are driven down to nearly zero.
• Technically, our proof techniques do not rely on several commonly used assumptions in the
literature such as weight reparameterization, attention linearization, special initialization, lazy
regime, etc. Our main idea is to treat the problem as a coupled dynamical system with six different
types of dynamic variables, for which we provide an articulated analysis on the gradient flow
dynamics. In particular, we prove a novel property of the gradient flow, termed automatic balancing
of gradients, which shows that the ratio of several important gradients will evolve closely within
the same range during training. This enables us to show that the losses of all training samples
can decrease almost at the same rate, and is also a key component in proving the near minimum
training loss as well as analyzing the changes of softmax.

1.1
Related Work

Transformer representational power. Several previous works have studied the expressiveness of
transformers. One line of work was from a universal approximation perspective and thus provided
the existential results [YBR+19, WCM22, PBM21, ZPGA23, LAG+22, BCW+23]. As a separate
view, [EGKZ22] showed that a single attention head can represent a sparse function over the input

2


---Page Break---
sequence with sample complexity much smaller than the context length. [ZZYW23] studied the
approximation and generalization performance of transformers in in-context learning. [SHT23]
proved that transformers can represent certain functions more efficiently than MLPs.

Training transformers.
Various settings of training transformers have been studied recently.
[WXM21] studied the impact of head and prompt tuning of transformer on the downstream learning
tasks. [JSL22] proved that transformers can learn spatial structures. [LWLC22] studied how a
shallow transformer learns a dataset with both label-relevant and label-irrelevant tokens. [TWCD23]
studied a next-token prediction problem and showed that self-attention behaves like a discriminating
scanning algorithm. [LLR23] analyzed a layer-wise optimization scheme on how transformers learn
topic structures. [TLZO23, TLTO23, VDT24] studied a setting where transformers can learn a
SVM solution. [LWM+23] provided analysis of training graph transformers for node classification
tasks. [Thr24] studied the implicit bias in the next-token prediction problem. For in-context linear
regression, [VONR+23] constructed transformer weights to solve this task and showed empirically
that this is similar to what the transformer learned by gradient descent, [ACDS24] proved that the
critical points of the training objective of linear transformers implement a pre-conditioned gradient
descent, [ZFB24] provided the training dynamics of linear attention models, [HCL23] characterized
the training dynamics of softmax transformers, and [CSWY24] studied a multi-task linear regression
problem with a multi-headed softmax transformer. Further, [LWL+24] focused on nonlinear self-
attention and nonlinear MLP over classification tasks in in-context learning. [WLCC23] proved the
convergence of transformers via neural tangent kernel. [NDL24] showed that two-layer transformers
can learn causal structure via gradient descent. [CL24] developed algorithms for provably learning a
multi-head attention layer. [HWCL24] studied how transformers learn feature-position correlation.

This paper studies a different problem of detecting co-occurrence of words via transformers. Such a
setting has not been considered in the literature. More importantly, the previous studies of training
dynamics of transformers have adopted various assumptions/simplifications such as weight reparame-
terization, special initialization, attention linearization, lazy regime, etc. In contrast, our analysis here
based on gradient flow does not rely on those simplifications, which can be of independent interest
for studying transformers in other settings.

2
Problem Setting

Notations. For a vector v ∈Rd, we use diag(v) to denote a diagonal matrix with v being the diagonal
entries. When we subtract the vector v by a scalar a, we subtract each entry of v by a, i.e., v −a ∈Rd

and (v −a)i = vi −a. We use eΩ, eΘ, eO to hide polylogarithmic factors.

2.1
Data Model

Definition 2.1 (Data distribution). Given a set of orthonormal vectors {µi}d
i=1 as word embedding,
let µ1, µ2 ∈Rd be two target signals whose co-occurrence needs to be detected by the model,
and let µ3 ∈Rd be a common token vector. A data entry (X, y) ∈Rd×L × {±1}, where X =
[x1, x2, . . . , xL] consists of L tokens, is generated by the distribution D as follows:

1. Uniformly randomly select an index i3 ∈[L] and set xi3 = µ3.

2. Then, one of the following cases occurs:

• With probability 1/2, set y = 1 and uniformly randomly select two indices i1 ̸= i2 ∈[L] \ {i3}
and set xi1 = µ1, xi2 = µ2. For i ∈[L] \ {i1, i2, i3}, set xi = Uniform({µi}d
i=4).
• With probability 1/6, set y = −1 and uniformly randomly select one index i1 ∈[L] \ {i3} and
set xi1 = µ1. For i ∈[L] \ {i3, i1}, we set xi = Uniform({µi}d
i=4).
• With probability 1/6, set y = −1 and uniformly randomly select one index i2 ∈[L] \ {i3} and
set xi2 = µ2. For i ∈[L] \ {i3, i2}, we set xi = Uniform({µi}d
i=4).
• With probability 1/6, set y = −1. For all i ∈[L] \ {i3}, we set xi = Uniform({µi}d
i=4).

In summary, there are 4 types of data: (1) both µ1, µ2 appear, (2) only µ1 appears, (3) only µ2
appears, and (4) neither µ1 nor µ2 appears. We denote the set of indices of the above 4 different types
of data by I1, I2, I3, I4 ⊆[n]. We further define R = {µi}d
i=4. For simplicity, our data distribution
assumes µ1, µ2, µ3 appear only once in a data entry. The occurrence probability of each type of

3


---Page Break---
data is chosen in the above way to make the distribution label-balanced. We assume there is a fixed
set of orthonormal vectors as word embedding, which is analogous to the one-hot embedding of a
set of vocabularies. Furthermore, in our daily language, there are some words appearing in almost
every sentence such as “a” and “the”. Thus, to model those words, we include a common token in
every data entry. Finally, notice that if we ignore the common token and random tokens, the data
distribution simplifies to a logical AND problem.
Remark 2.2. Recognizing co-occurrence of words is an important ability for language models
to perform many NLP tasks correctly. Consider the example of a language model first reading a
paragraph describing a children and then answering the question “Is Bob eating a banana?” If the
description is “Bob is watching a television while eating a banana”, then the model should answer
“Yes”. If the description is “Bob is playing computer games”, then the model should answer “No”.
Thus, the model needs to recognize the co-occurrence of “Bob” and “banana”.

For simplicity of our analysis, we make the following assumption on our training data set.

Assumption 2.3. The training set satisfies: (i) |I1|

n
= 1

2 and |I2|

n
= |I3|

n
= |I4|

n
= 1

6; and (ii) for
all i1, i2 ∈[n], l1, l2 ∈[L], if X(i1)
l1
, X(i2)
l2
/∈{µ1, µ2, µ3}, then X(i1)
l1
̸= X(i2)
l2
, i.e., all irrelevant
words are different.

The first assumption can be approximately satisfied with high probability given the total number n of
samples is large enough. Such an assumption can be removed by applying the standard concentration
theorems. The second assumption implicitly assumes nL ≤d. If the irrelevant words are uniformly
sampled from a large entire vocabulary, then each irrelevant word appears only very few times in the
training set. Thus, letting irrelevant words appear only once in the entire training set is a reasonable
way to simplify our analysis.

2.2
Transformer Architecture and Training

Consider a training set {(X(i), yi)}n
i=1 with n training samples. Each data point X(i) ∈Rd×L

contains L tokens, i.e., X(i) = [x(i)
1 , x(i)
2 , . . . , x(i)
L ]. We consider the transformer model with a
self-attention module followed by a linear MLP:

F(X; W, WV , WK, WQ) =

L
X

l=1

m1
X

j=1
aj


w⊤
j WV X · Softmax
X⊤W ⊤
KWQxl
√m


(1)

where the query matrix WQ ∈Rm×d, the key matrix WK ∈Rm×d, the value matrix WV ∈Rm×d,
the hidden-layer MLP weights W ∈Rm1×m (with w⊤
j being the j-th row of W), and the output-layer
weights of the MLP a ∈Rm1. We define the linear MLP function of the transformer to be G(µ) =
Pm1
j=1 ajw⊤
j WV µ. We now introduce some shorthand notations K = WKX, Q = WQX, V =
WV X and let kl = WKxl. Notice that K = [k1, k2, . . . , kL]. We further extend this shorthand to
ql and vl. We also define functions k(µ) = WKµ, q(µ) = WQµ, v(µ) = WV µ. We introduce the

shorthand for the score vector sl := X⊤W ⊤
K WQxl
√m
and the attention vector pl := Softmax(sl). For the

attention vector, if µ, ν ∈X(i), let l(i, µ), l(i, ν) be the indices such that X(i)
l(i,µ) = µ, X(i)
l(i,ν) = ν,

and we define p(i)
q←µ,k←ν := p(i)
q←l(i,µ),k←l(i,ν) := Softmax

X(i)⊤W ⊤
K WQµ
√m


l(i,ν).

Initialization. We initialize aj
i.i.d.
∼
Uniform(±1) and the value of a is fixed during training.

The trainable parameters are [W, WV , WK, WQ]. We initialize [W, WV , WK, WQ] by Wi,j
i.i.d.
∼

N(0, σ2
1) and (WV )i,j, (WK)i,j, (WQ)i,j
i.i.d.
∼N(0, σ2
0).

Training. We adopt the cross-entropy loss l(x) = log(1 + exp(−x)). The gradient of the cross-
entropy loss is given by l′(x) = −
1
1+exp(x) and we define g(x) =
1
1+exp(x). The model is trained by
gradient flow to minimize the following empirical loss:

bL(W, WV , WK, WQ) = 1

n

n
X

i=1
l(yiF(X(i); W, WV , WK, WQ)).
(2)

4


---Page Break---
Similarly, we define the generalization loss L := E(X,y)∼D ℓ(yF(X)). We introduce the parameter
condition that we take throughout the entire analysis and proofs.

Condition 1. We make the following parameter choices in our analysis:

• The embedding dimension and network width satisfy m ≥eΩ(max{m1, L2}) and m1 ≥eΩ(1).

• The network weight initialization variance satisfies σ0 =
1
eΘ(
√

Lm); and σ1 =
1
eΘ(√m1).

• The number of training samples and tokens satisfy n ≥eΩ(L2) and L ≥eΩ(1).

• The failure probability satisfies 1/δ ≤poly(m).

3
Main Results

Challenges. The essential goal is to derive the gradient flow update for each weight matrix (see
the gradient expressions for all weight matrices in Appendix B). However, directly analyzing the
dynamics of those weight matrices is extremely challenging, because: (i) keeping track of how the
column and row spaces of each weight matrix change during training is difficult; and (ii) all attention
and MLP weight matrices are affecting each other, leading to highly coupled dynamics.

Our General Idea. To overcome the above challenges, we first note that rather than tracking
W, WV , WK, WQ directly, it is sufficient to analyzing their impact on inputs, i.e., X⊤W ⊤
Q WKX and
a⊤WWV X, which are sufficient to compute F(X; W, WV , WK, WQ). Based on this observation,
we formulate two differential equations to keep track of w(t)⊤
j
W (t)
V µ and ν⊤W (t)⊤
K
W (t)
Q µ (with
respect to t) for all µ, ν ∈{µi}d
i=1 (See Equation (6) and Equation (7) in Appendix C). We further

include additional equations to keep track of
D
w(t)
j1 , w(t)
j2
E
, ν⊤W (t)⊤
V
W (t)
V µ, ν⊤W (t)⊤
K
W (t)
K µ, and

ν⊤W (t)⊤
Q
W (t)
Q µ to complete the system. Intuitively, the additional equations keep track of the shape
of the neurons and the word embedding after WV , WK, WQ transform. Although the dynamical
system does not directly track the softmax, the softmax probability can be calculated via the scores of
ν⊤W (t)⊤
K
W (t)
Q µ. The full dynamical system is presented in Appendix C. Then the training dynamics
can be characterized by analyzing these differential equations (see a proof outline in Section 4).

In the next two theorems, we present our characterization of the training process into two phases.

Theorem 3.1 (Phase 1). With probability at least 1 −δ over the randomness of weight initialization,
there exists a time T1 = eO(1/m) such that

• The linear MLP functions satisfy: G(T1)(µ1) ≥Ω(1), G(T1)(µ2) ≥Ω(1), G(T1)(µ3) ≤−Ω(1).

• All training samples are correctly classified: yiF (T1)
i
> 0 for all i ∈[n].

• For t ∈[0, T1], all dynamical variables
D
w(t)
j1 , w(t)
j2
E
, ν⊤W (t)⊤
K
W (t)
Q µ, ν⊤W (t)⊤
V
W (t)
V µ,

ν⊤W (t)⊤
K
W (t)
K µ, and ν⊤W (t)⊤
Q
W (t)
Q µ are close to their initialization values.

• Training loss is still large: bL(T1) = Θ(1).

In Theorem 3.1, item 1 implies that in a short time, the linear MLP function G(T1)(·) positively aligns
with the two target signals µ1 and µ2, but negatively aligns with the common token µ3. This further
guarantees item 2 of Theorem 3.1 that all training samples are classified correctly. Further, item 3 of
Theorem 3.1 indicates that the attention matrices are still close to their initialization values, and hence
have not started to learn any knowledge yet. This results in item 4 of Theorem 3.1, which shows that
the training loss is still large.

Theorem 3.2 (Phase 2). With probability at least 1 −δ over the randomness of weight initialization,
there exists a time range (T1, T2) with T2 = poly(m) such that for all t ∈(T1, T2)

• µ⊤
2 W (t)⊤
K
W (t)
Q µ1
and
µ⊤
1 W (t)⊤
K
W (t)
Q µ2
increase,
whereas
µ⊤
3 W (t)⊤
K
W (t)
Q µ1
and

µ⊤
3 W (t)⊤
K
W (t)
Q µ2 decrease.

5


---Page Break---
• µ⊤
1 W (t)⊤
V
W (t)
V µ2 increases, whereas µ⊤
1 W (t)⊤
V
W (t)
V µ3 and µ⊤
2 W (t)⊤
V
W (t)
V µ3 decrease.

• Linear MLP functions satisfy: G(t)(µ1) ≥Ω(1), G(t)(µ2) ≥Ω(1), −G(t)(µ3) ≤Ω(1).

• G(t)(µ1) + G(t)(µ2) + G(t)(µ3) ≥Ω(1), G(t)(µ1) + G(t)(µ3) ≤−Ω(1) and G(t)(µ2) +
G(t)(µ3) ≤−Ω(1).

In Theorem 3.2, item 1 indicates that, during Phase 2, gradient flow drives the self-attention module
to weigh more between the two target signals µ1 and µ2, and to weigh less between one of these
signals and the common token µ3. Item 2 indicates that gradient flow drives the value matrix WV
to positively align the two target signals µ1 and µ2, but negatively align one target signal (µ1 or µ2)
with the common token µ3. Further, the last two items indicate that the MLP continue to classify
correctly and further enlarge the classification margin. Hence, all items in Theorem 3.2 collectively
indicate that attention and MLP evolve jointly to enlarge the classification margin and hence drive the
loss value to decrease in Phase 2.
Theorem 3.3 (Near Minimum Training Loss and Attention). With probability at least 1 −δ, there
exists a time T ⋆= Θ(poly(m)) such that

• The training and generalization losses satisfy bL(T ⋆) ≤1/poly(m) and L(T ⋆) ≤1/poly(m).

• The attention matrices satisfies:

W (T ⋆)⊤
K
W (T ⋆)
Q
= W (0)⊤
K
W (0)
Q
+ P

i1,i2∈[d] C(T ⋆)
i1,i2 µi1µ⊤
i2,
(3)

where C(T ⋆)
1,2 , C(T ⋆)
2,1 , −C(T ⋆)
3,1 , −C(T ⋆)
3,2
=
Θ

σ2
0m
√mLσ2
1mm1 +
σ2
0
√m
√mσ2
1mm1


and C(T ⋆)
i1,i2
≤

eO

σ2
0m
n√mLσ2
1mm1 +
σ2
0
√m
√mσ2
1mm1


if one of i1, i2 ∈[d] \ [3].

Theorem 3.3 indicates that both training and test losses converge nearly to zero as long as the
embedding dimension m is sufficiently large, because both the attention and MLP matrices are
trained towards enlarging the classification margin in Phase 2. Theorem 3.3 also provides the explicit
form of the attention matrix in Equation (3), in which the second term captures the learned information
of the self-attention module. It can be seen that the large coefficients C(T ⋆)
1,2
and C(T ⋆)
2,1
capture strong

coupling of the two target signals µ1 and µ2, and the large negative coefficients C(T ⋆)
3,1
and C(T ⋆)
3,2
encourages strong negative coupling of one target signal µ1 or µ2 and the common token µ3. All these
attention terms contribute to enlarge correct classification margin. Further, the coefficients between
all other random tokens are order-level smaller and hence do not corrupt the correct classification.

Synthetic Experiment: We next verify our theory and the two-phase characterization of the training
process via synthetic experiments (see the experiment setup in Appendix A).






	
	

	
	



	







	



  	   

  	   


  
   

  
   	

Phase 1
Phase 2






	
	

	
	





	
















	 
 

	 


 



Phase 1
Phase 2

(a) Attention score correlation
(b) Training loss

Figure 1: Synthetic experiments with illustration of two training phases. The detailed experiment
setup can be found in Appendix A.

Figure 1 (a) shows how the attention score correlation µ⊤
i W (t)⊤
K
W (t)
Q µj evolves during the training.
It is clear that these scores do not change significantly in Phase 1, verifying Theorem 3.1. In Phase
2, the score correlation between two target signals µ1 and µ2 increases, and the score between one
target signal and the common token decreases, verifying Theorem 3.2.

6


---Page Break---
Figure 1 (b) plots how the training loss changes during the two phases of training. The blue curve
(indexed by ‘loss’) represents the overall training loss of all samples. The other curves correspond to
the training loss of four types of samples as indicated in the legend. In Phase 1, the training loss for
samples with both target signals (i.e., orange curve) decreases because the linear MLP layer aligns
with the target signals (verifying Lemma 4.1 in Section 4.1). The training loss for samples with
one target signal and the common token (i.e., green or red curves) first increases because the linear
MLP layer initially has not aligned negatively enough with the common token yet (as captured by
Lemma 4.2 in Section 4.1), and then decreases in the later stage of Phase 1 when the MLP layer aligns
negatively with the common token (as captured by Lemma 4.3 in Section 4.1). All loss functions
decrease in Phase 2 because all attention matrices and linear MLP jointly enlarge the classification
margin, verifying Theorem 3.2.

4
Proof Outline: Two-phase Gradient Flow Analysis

4.1
Phase 1: Alignment of Linear MLP for Correct Classification

In Phase 1, the linear MLP quickly aligns with the two target word tokens while all attention matrices
stay roughly unchanged from their initialization values. We show that the linear MLP functions
|G(t)(µ1)|, |G(t)(µ2)|, |G(t)(µ3)| become sufficiently large (larger than some constant threshold) so
that all training samples are correctly classified at the end of Phase 1.

We first analyze the dynamical system at the initialization. In particular, the following lemma shows
that at the initialization, the linear MLP layer receives a sufficiently large gradient from the two target
signals, and hence samples with the two target signals will be classified correctly as co-occurrence
soon afte the training starts.
Lemma 4.1 (Same as Lemma E.4). With probability at least 1 −δ over the weight initialization,

∀µ ∈{µ1, µ2} :
∂ajw(0)
j
W (0)
V
µ
∂t
= Θ((σ2
0 + σ2
1)m).

Further, by the definition of Phase 1 (see Definition E.2 for a formal definition), the gradients of
the attention matrices in the dynamical system are much smaller than that of linear MLP given
in Lemma 4.1. This implies that during Phase 1, mainly the linear MLP is performing learning,
whereas all the attention matrices are changing slowly from their initialization. Based on this, we have
∂
∂tG(t)(µ1) = Θ((σ2
0 + σ2
1)mm1) which implies that it takes only O(1/(σ2
0 + σ2
1)mm1) iterations
for G(t)(µ1) to reach a certain constant magnitude.

Lemma 4.1 indicates that the samples with co-occurrence of the two target signals are classified
correctly. The following lemma shows that the initial gradient from the common token, i.e., the
gradient of G(t)(µ3), is much smaller than the gradient from the two target signals, which implies
that the samples with only one target signal may be classified incorrectly as co-occurrence (since the
network in this case will output a positive value). This is verified empirically by our experiments in
Figure 1 (b), where the loss function corresponding to only one target signal and the common token
first increases in Phase 1.
Lemma 4.2 (Same as Lemma E.16). Let F = maxi |F (0)
i
|. With probability at least 1 −δ over the
weight initialization,
∂ajw(0)
j
W (0)
V
µ3
∂t

 = eO

σ2
1
√mm1 + σ2
0
√

mL + σ2
1mF

.

Notice that the model output F depends on the weight initialization scale and can be made small.

We next show in the following lemma that the gradient ∂G(t)(µ3)

∂t
of the common token will quickly
become negative soon after the training begins, which drives the transformer model to output a
negative value when it sees those types of samples. This implies that negative samples (without
co-occurrence of two target tokens) will be classified correctly towards the end of Phase 1. This is
also verified empirically by our experiments in Figure 1 (b), where the loss function corresponding to
only one target signal and the common token descreases towards the end of Phase 1.
Lemma 4.3 (Abbreviated from Theorem E.19). There exists a time T0.5 ≤T1 and a constant C such
that for all t ∈[T0.5, T1]

(1 + C) max

∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t

≤−∂G(t)(µ3)

∂t
≤(1 −C)

∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t

.

7


---Page Break---
Proof Intuition of Lemma 4.3. We first note that the term

1
n
Pn
i=1 g(0)
i
yi
PL
l2=1
Pm1
j2=1 ∥w(0)
j2 ∥2
2(X(i)p(0,i)
l2
)⊤µ

makes the major contribution to the gradient ∂G(t)(µ3)

∂t
. Such a term is small at the initialization due
to the cancellation effect from positive and negative yi’s. However, since the linear MLP G will
positively align the two target signals at the beginning, for the samples with positive labels, g(t)
i
will

decrease, whereas for samples with only one target signal, g(t)
i
will increase. Hence,
∂ajw(t)
j
W (t)
V
µ3
∂t
will become negative. This trend will continue until the gradient from µ3 starts to match the gradients
from µ1, µ2, which is what we establish in Lemma 4.3.

Using Lemma 4.3, we can show that all training samples are correctly classified at end of Phase 1.

4.2
Phase 2: Evolution of Attention and MLP for Large Classification Margin

In Phase 2, both attention and MLP matrices evolve towards enlarging the classification margin, thus
driving the loss value small.

We now analyze what happens in Phase 2. Let T2 denote the end of Phase 2. Recall that at the end of
Phase 1, we have G(t)(µ1), G(t)(µ2), −G(t)(µ3) ≥Ω(1). We will mainly need to show that such a
condition continues to hold in Phase 2, so that attention matrices will evolve with MLP to learn better
classifiers. To this end, we exam the following gradient flow in the dynamical system:

∂G(t)(µ)

∂t
= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1
p(t,i2)
q←l2,k←µ ·

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
(4)

+ m1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ.

It has been proved that at the end of Phase 1, for µ ∈{µ1, µ2, µ3}, we have

P

j1

∂w(t)⊤
j1
∂t
W (t)
V µ
 ≪

P

j1 w(t)⊤
j1
∂W (t)
V
∂t
µ
 since the magnitude of Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
is large. Assume this

can hold for long enough (which we can indeed prove later). Then, we only need to focus on the
first term in the sum on the right-hand side in Equation (4). On the other hand, from the dynamical
system, we can calculate

∂
∂t

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= 2m1

n

n
X

i=1
g(t)
i yiF (t)
i
.
(5)

Thus, if yiF (t)
i
> 0 for all i ∈[n], then Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
is always increasing.

Thus, ∂G(t)(µ)

∂t
mainly depends on the behavior of 1

n
P
i2: µ∈X(i2) g(t)
i2 yi2
PL
l2=1 p(t,i2)
q←l2,k←µ. Fur-

ther, this is also a key quantity we need to analyze ∂ν⊤W (t)⊤
V
W (t)
V
µ
∂t
and
∂ν⊤W (t)⊤
K
W (t)
Q µ
∂t
. Note
that 1

n
P

i2: µ∈X(i2) g(t)
i2 yi2
PL
l2=1 p(t,i2)
q←l2,k←µ ≈1

n
P

i2: µ∈X(i2) g(t)
i2 yi2 if p(t,i2)
q←l2,k←µ ≈1/L which
holds at the beginning of Phase 2. Later, we are going to prove convergence of the training loss via
the following: (1) the training loss can decrease if the softmax probability is uniform; (2) even though
the softmax probability will deviate from uniform distribution during training, we can bound such
deviation and the loss value can still decrease.

Automatic balancing of gradients. As argued above, our main focus is on analyzing the behavior of
1
n
P

i2: µ∈X(i2) g(t)
i2 yi2. This consists of two parts: (i) Lemma 4.4, which shows that the two groups

of samples with only the presence of one target signal have gradients P

i∈I2 g(t)
i
and P

i∈I3 g(t)
i
close

to each other during training; and (ii) Lemma 4.5, which shows that the gradient gaps P

i∈I1 g(t)
i
−
P

i∈I2 g(t)
i
and P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
are not too small compared with P

i∈[n] g(t)
i . Both

8


---Page Break---
Lemmas 4.4 and 4.5 establish that the ratio of those important gradients are kept within certain ranges
during training. We call such a key property as automatic balancing of gradients, which is further
used for proving that the gradient flow can drive the training loss small.
Lemma 4.4 (Same as Lemma F.5). For t ∈[T1, T2], there exists a small constant C ≪1 such that
P

i∈I2 g(t)
i
−P

i∈I3 g(t)
i


min(P

i∈I2 g(t)
i , P

i∈I3 g(t)
i )
≤C.

Proof Intuition of Lemma 4.4. The intuition behind the result is as follows. If P

i∈I2 g(t)
i
becomes

much bigger than P

i∈I3 g(t)
i
during the training, then P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
is much smaller than
P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
which makes ∂G(t)(µ1)

∂t
< ∂G(t)(µ2)

∂t
. It is not hard to show that random

tokens make negligible contributions to the gradient. Thus, for i ∈I2, we have ∂F (t)
i
∂t
≈∂G(t)(µ1)

∂t
+

∂G(t)(µ3)

∂t
. By the chain rule, we have ∂g(t)
i
∂t
= g′(yiF (t)
i
)yi
∂F (t)
i
∂t . Since ∂G(t)(µ3)

∂t
< 0, if ∂G(t)(µ1)

∂t
<

∂G(t)(µ2)

∂t
, then P

i∈I2 g(t)
i
will drop faster than P

i∈I3 g(t)
i , i.e., −∂

∂t
P

i∈I2 g(t)
i
> −∂

∂t
P

i∈I3 g(t)
i .

In Appendix, we formally prove Lemma 4.4 by analyzing the ratio P
i∈I2 g(t)
i / P

i∈I3 g(t)
i , and show
that this ratio hangs over around 1 during training.

Lemma 4.5 (Abbreviated from Lemma F.6). For t ∈[T1, T2], the gradient satisfies that
P

i∈[n] g(t)
i
P
i∈I2∪I3∪I4 g(t)
i
−P
i∈I1 g(t)
i
= O(1),

P

i∈[n] g(t)
i
P
i∈I1 g(t)
i
−P
i∈I2 g(t)
i
= O(1).

Further, for some constant C, we have

(1 + C) max
∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t


≤−∂G(t)(µ3)

∂t
≤(1 −C)
∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t


.

Proof Sketch of Lemma 4.5. The proof of Lemma 4.5 relies on analyzing how the ratio between
∂G(t)(µ1)

∂t
and −∂G(t)(µ3)

∂t
changes. We show that this ratio will hang over around some range. Recall

the relationship that ∂G(t)(µ)

∂t
≈1

n
P
i2: µ∈X(i2) g(t)
i2 yi2 · Pm1
j1=1
Pm1
j2=1 aj1aj2
D
w(t)
j1 , w(t)
j2
E
. It is not
hard to show that

−∂G(t)(µ3)

∂t
∂G(t)(µ1)

∂t
≈−P

i∈I1 g(t)
i
+ P

i∈I2∪I3∪I4 g(t)
i
P
i∈I1 g(t)
i
−P
i∈I2 g(t)
i
.

Define R(t) :=
−P

i∈I1 g(t)
i
+P

i∈I2∪I3∪I4 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
. Solving when ∂

∂tR(t) ≥0 yields a quadratic inequal-

ity, and further analysis shows that the root is contractive and is within some specific range.

Utilizing the gradient automatic balancing properties, the following corollary characterizes how the
attention matrices in the dynamical system change in Phase 2. In particular, we can show that after
WV -transform, µ1 and µ2 become more positively correlated whereas µ1 and µ3 (also µ2 and µ3)
become negatively correlated. This is a direct result following from updates of the dynamical system.
Corollary 4.6 (Abbreviated from Corollary F.13). For t ∈[T1, T2],

∂
∂tµ⊤
2 W (t)⊤
V
W (t)
V µ1 > 0,
∂
∂tµ⊤
1 W (t)⊤
V
W (t)
V µ3 < 0.

Since we have analyzed how G(t)(·) will change in stage 2, we can utilize this information to analyze
the change of softmax attention via the following relationship: by Appendix C, we can derive

∂µ⊤
1 W (t)⊤
K
W (t)
Q µ2
∂t

9


---Page Break---
=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1
µ⊤
1 W (t)⊤
K
K(t,i) · diag

G(t)(X(i)) −(G(t)(X(i)))⊤p(t,i)
l

p(t,i)
l
x(i)⊤
l
µ2

+
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1
µ⊤
2 W (t)⊤
Q
q(t,i)
l
p(t,i)⊤
l
· diag

G(t)(X(i)) −(G(t)(X(i)))⊤p(t,i)
l

X(i)⊤µ1.

The following lemma shows that the attention score between the two target signals µ1 and µ2
increases, whereas that between one target signal µ1 or µ2 and the common token µ3 decreases.
Lemma 4.7 (Abbreviated from Lemma F.16). For µ, ν ∈{µ1, µ2}, µ ̸= ν, and for t ∈[T1, T2],
∂
∂tν⊤W (t)⊤
K
W (t)
Q µ =
1
√m
eΘ(bL(t)σ2
0m) 1

L,
∂
∂tµ⊤
3 W (t)⊤
K
W (t)
Q µ = −1
√m
eΘ(bL(t)σ2
0m) 1

L.

Lemma 4.7 is keeping track of the attention coefficients C(t)
i1,i2 in Theorem 3.3 via gradient flow,
which proves the second item of Theorem 3.3.

5
Discussion and Future Directions

In this work, we developed a novel gradient flow based framework for analyzing the training
dynamics of a one-layer transformer to recognize co-occurring tokens. We provided a two-phase
characterization of the training process. In Phase 1, the linear MLP layer is trained to classify samples
correctly, with attention weights almost unchanged. In Phase 2, both attention matrices and the linear
MLP jointly evolve to enlarge the classification margin, thus reducing the loss to near minimum.

As future work, it will be interesting to analyze more general transformer architectures such as
multi-headed attention, multi-layer transformer, etc. Further, it is of interest to study the dynamics of
more advanced gradient descent algorithms such as gradient descent with adaptive learning rate, with
momentum, etc., and explore how the hyperparameters will affect the training dynamics. Another
direction is to study more practical language sequences where tokens are generated in a correlated
fashion. Then the next token prediction becomes an intriguing problem.

Acknowledgement

H. Yang would like to thank Jason D. Lee and Yunwei Ren for insightful discussion and suggestions.

This work was performed under the auspices of the U.S. Department of Energy by the Lawrence
Livermore National Laboratory under Contract No. DE-AC52-07NA27344 and supported by the
LLNL-LDRD Program under Project No. 22-SI-004 and 24-ERD-010. The work of Y. Liang was
supported in part by the U.S. National Science Foundation under the grants ECCS-2113860 and
DMS-2134145. The work of Z. Wang was in part supported by an NSF Scale-MoDL grant (award
number: 2133861) and the CAREER Award (award number: 2145346).

References

[ACDS24] Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn
to implement preconditioned gradient descent for in-context learning. Advances in
Neural Information Processing Systems, 36, 2024.

[ADF+23] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre
Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2
technical report. arXiv preprint arXiv:2305.10403, 2023.

[BCE+23] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric
Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al.
Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint
arXiv:2303.12712, 2023.

[BCW+23] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as
statisticians: Provable in-context learning with in-context algorithm selection. arXiv
preprint arXiv:2306.04637, 2023.

10


---Page Break---
[CL24] Sitan Chen and Yuanzhi Li. Provably learning a multi-head attention layer. arXiv
preprint arXiv:2402.04084, 2024.

[CSWY24] Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of
multi-head softmax attention for in-context learning: Emergence, convergence, and
optimality. arXiv preprint arXiv:2402.19442, 2024.

[Dam18] Friederick J Damerau. Markov models and linguistic theory, volume 95. Walter de
Gruyter GmbH & Co KG, 2018.

[DBK+20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua
Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition
at scale. In International Conference on Learning Representations, 2020.

[DCLT18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-
training of deep bidirectional transformers for language understanding. arXiv preprint
arXiv:1810.04805, 2018.

[EGKZ22] Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases
and variable creation in self-attention mechanisms. In International Conference on
Machine Learning, pages 5793–5831. PMLR, 2022.

[HCL23] Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers.
arXiv preprint arXiv:2310.05249, 2023.

[HWCL24] Yu Huang, Zixin Wen, Yuejie Chi, and Yingbin Liang.
Transformers provably
learn feature-position correlations in masked image modeling.
arXiv preprint
arXiv:2403.02233, 2024.

[JSL22] Samy Jelassi, Michael Sander, and Yuanzhi Li. Vision transformers provably learn
spatial structure. Advances in Neural Information Processing Systems, 35:37822–
37836, 2022.

[LAG+22] Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang.
Transformers learn shortcuts to automata. In The Eleventh International Conference
on Learning Representations, 2022.

[LLC+21] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and
Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows.
In Proceedings of the IEEE/CVF international conference on computer vision, pages
10012–10022, 2021.

[LLR23] Yuchen Li, Yuanzhi Li, and Andrej Risteski. How do transformers learn topic structure:
Towards a mechanistic understanding. arXiv preprint arXiv:2303.04245, 2023.

[LWL+24] Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, and Pin-Yu Chen. Training
nonlinear transformers for efficient in-context learning: A theoretical learning and
generalization analysis. arXiv preprint arXiv:2402.15607, 2024.

[LWLC22] Hongkang Li, Meng Wang, Sijia Liu, and Pin-Yu Chen. A theoretical understanding of
shallow vision transformers: Learning, generalization, and sample complexity. In The
Eleventh International Conference on Learning Representations, 2022.

[LWM+23] Hongkang Li, Meng Wang, Tengfei Ma, Sijia Liu, Zaixi Zhang, and Pin-Yu Chen. What
improves the generalization of graph transformer? a theoretical dive into self-attention
and positional encoding. NeurIPS 2023 Workshop: New Frontiers in Graph Learning,
2023, 2023.

[MH23] James Manyika and Sissie Hsiao. An overview of bard: an early experiment with
generative ai. AI. Google Static Documents, 2, 2023.

[MS99] Christopher Manning and Hinrich Schutze. Foundations of statistical natural language
processing. MIT press, 1999.

11


---Page Break---
[NDL24] Eshaan Nichani, Alex Damian, and Jason D Lee. How transformers learn causal
structure with gradient descent. arXiv preprint arXiv:2402.14735, 2024.

[Ope23] OpenAI. Gpt-4 technical report, 2023.

[PBM21] Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing complete. The
Journal of Machine Learning Research, 22(1):3463–3497, 2021.

[SHT23] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Representational strengths and
limitations of transformers. arXiv preprint arXiv:2306.02896, 2023.

[Tea23] Gemini Team. Gemini: a family of highly capable multimodal models. arXiv preprint
arXiv:2312.11805, 2023.

[Thr24] Christos Thrampoulidis. Implicit bias of next-token prediction. arXiv:2402.18551,
2024.

[TLI+23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux,
Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar,
et al.
Llama: Open and efficient foundation language models.
arXiv preprint
arXiv:2302.13971, 2023.

[TLTO23] Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak.
Transformers as support vector machines. In NeurIPS 2023 Workshop on Mathematics
of Modern Machine Learning, 2023.

[TLZO23] Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, and Samet Oymak. Max-
margin token selection in attention mechanism. In Thirty-seventh Conference on Neural
Information Processing Systems, 2023.

[TWCD23] Yuandong Tian, Yiping Wang, Beidi Chen, and Simon Du. Scan and snap: Understand-
ing training dynamics and token composition in 1-layer transformer. arXiv preprint
arXiv:2305.16380, 2023.

[TWZ+23] Yuandong Tian, Yiping Wang, Zhenyu Zhang, Beidi Chen, and Simon Shaolei Du.
Joma: Demystifying multilayer transformers via joint dynamics of mlp and attention.
In The Twelfth International Conference on Learning Representations, 2023.

[VDT24] Bhavya Vasudeva, Puneesh Deora, and Christos Thrampoulidis. Implicit bias and fast
convergence rates for self-attention. arXiv preprint arXiv:2402.05738, 2024.

[VONR+23] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexan-
der Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn
in-context by gradient descent. In International Conference on Machine Learning,
pages 35151–35174. PMLR, 2023.

[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in
neural information processing systems, 30, 2017.

[WCM22] Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a
case study on approximating turing machines with transformers. Advances in Neural
Information Processing Systems, 35:12071–12083, 2022.

[WLCC23] Yongtao Wu, Fanghui Liu, Grigorios Chrysos, and Volkan Cevher. On the conver-
gence of encoder-only shallow transformers. In Thirty-seventh Conference on Neural
Information Processing Systems, 2023.

[WXM21] Colin Wei, Sang Michael Xie, and Tengyu Ma. Why do pretrained language models
help in downstream tasks? an analysis of head and prompt tuning. Advances in Neural
Information Processing Systems, 34:16158–16170, 2021.

[YBR+19] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv
Kumar. Are transformers universal approximators of sequence-to-sequence functions?
In International Conference on Learning Representations, 2019.

12


---Page Break---
[ZFB24] Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear
models in-context. Journal of Machine Learning Research, 25(49):1–55, 2024.

[ZPGA23] Haoyu Zhao, Abhishek Panigrahi, Rong Ge, and Sanjeev Arora. Do transformers parse
while predicting the masked word? arXiv preprint arXiv:2303.08117, 2023.

[ZZYW23] Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. What and how
does in-context learning learn? bayesian model averaging, parameterization, and
generalization. arXiv preprint arXiv:2305.19420, 2023.

13


---Page Break---
Contents

1
Introduction
1

1.1
Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

2
Problem Setting
3

2.1
Data Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3

2.2
Transformer Architecture and Training . . . . . . . . . . . . . . . . . . . . . . . .
4

3
Main Results
5

4
Proof Outline: Two-phase Gradient Flow Analysis
7

4.1
Phase 1: Alignment of Linear MLP for Correct Classification . . . . . . . . . . . .
7

4.2
Phase 2: Evolution of Attention and MLP for Large Classification Margin . . . . .
8

5
Discussion and Future Directions
10

A Setup of Synthetic Experiment
16

B
Gradient Flow Update for Weight Matrices
16

C Gradient Flow Dynamical System
16

C.1
Derivation of the Dynamical System . . . . . . . . . . . . . . . . . . . . . . . . .
17

D Initialization
21

E Training Dynamics: Phase 1
23

E.1
Initial Gradients . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

E.2
Maximum Perturbation of Neuron Outputs . . . . . . . . . . . . . . . . . . . . . .
27

E.3
Perturbation Term Involving Correlation of Value-transformed Data
. . . . . . . .
27

E.4
Perturbation Term Involving Correlation of Neurons . . . . . . . . . . . . . . . . .
29

E.5
Neuron Weights Align with Signal Value . . . . . . . . . . . . . . . . . . . . . . .
30

E.6
Alignment of Common Token
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
32

E.7
Small Score Movement in Phase 1 . . . . . . . . . . . . . . . . . . . . . . . . . .
37

E.8
All Variables are within Range in Definition of Phase 1 . . . . . . . . . . . . . . .
43

F
Training Dynamics: Phase 2
44

F.1
Automatic Balancing of Gradients . . . . . . . . . . . . . . . . . . . . . . . . . .
48

F.2
How Fast the Loss Decreases . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54

F.3
Growth of Neuron Correlation . . . . . . . . . . . . . . . . . . . . . . . . . . . .
55

F.4
Growth of Correlation of Value-Transformed Data . . . . . . . . . . . . . . . . . .
56

F.5
Change of Random-Token Sub-Network . . . . . . . . . . . . . . . . . . . . . . .
57

F.6
Change of Score and Softmax Probability . . . . . . . . . . . . . . . . . . . . . .
59

F.7
Change of Self-Correlation of Key/Query-Transformed data
. . . . . . . . . . . .
61

14


---Page Break---
F.8
Small Loss is Achieved . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
62

F.9
Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
63

F.10 Proof of Theorem 3.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
63

G Auxiliary Results
63

H Probability
64

15


---Page Break---
A
Setup of Synthetic Experiment

We conduct synthetic experiment to verify our theoretical results. We create a dataset following our
data distribution in Definition 2.1 with 60 training samples: 30 samples have both µ1 and µ2 in it, 10
samples have only µ1, 10 samples have only µ2, and 10 samples have neither µ1 nor µ2. Each data
consists of 5 patches and each patch has dimension 64. The embedding dimension m is set to be 128
and the number of neurons is set to be 256. We use Kaiming initialization to initialize the transformer
weights. The transformer is trained by gradient descent with learning rate 0.01 for 30000 epochs.

B
Gradient Flow Update for Weight Matrices

We provide the gradient flow update for each weight matrix as follows.

∂w(t)
j
∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
ajV (t,i)p(t,i)
l
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
aj

L
X

h=1
v(t,i)
h
p(t,i)
l,h

∂W (t)
V
∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajw(t)
j

X(i)p(t,i)
l
⊤

∂W (t)
K
∂t
=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
aj

L
X

h=1
w(t)⊤
j
v(t,i)
h

L
X

h′=1

∂p(t,i)
l,h
∂sl,h′ q(t,i)
l
x(i)⊤
h′

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
aj

L
X

h=1
w(t)⊤
j
v(t,i)
h

L
X

h′=1
p(t,i)
l,h (I(h = h′) −p(t,i)
l,h′ )q(t,i)
l
x(i)⊤
h′

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
aj

−w(t)⊤
j
V (t,i)p(t,i)
l
q(t,i)
l
(X(i)p(t,i)
l
)⊤+ q(t,i)
l
w(t)⊤
j
V (t,i)diag(p(t,i)
l
)X(i)⊤

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajq(t,i)
l

−w(t)⊤
j
V (t,i)p(t,i)
l
p(t,i)⊤
l
+ w(t)⊤
j
V (t,i)diag(p(t,i)
l
)

X(i)⊤

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajq(t,i)
l
p(t,i)⊤
l
diag

w(t)⊤
j
V (t,i) −w(t)⊤
j
V (t,i)p(t,i)
l

X(i)⊤

∂W (t)
Q
∂t
=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
aj

L
X

h=1
w(t)⊤
j
v(t,i)
h

L
X

h′=1
p(t,i)
l,h (I(h = h′) −p(t,i)
l,h′ )k(t,i)
h′
x(i)⊤
l

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
aj

−w(t)⊤
j
V (t,i)p(t,i)
l
(K(t,i)p(t,i)
l
) + K(t,i)diag(p(t,i)
l
)V (t,i)⊤w(t)
j

x(i)⊤
l

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajK(t,i) 
−w(t)⊤
j
V (t,i)p(t,i)
l
p(t,i)
l
+ diag(p(t,i)
l
)V (t,i)⊤w(t)
j

x(i)⊤
l

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajK(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l

p(t,i)
l
x(i)⊤
l

C
Gradient Flow Dynamical System

We first provide our complete dynamical system. The derivation of each equation is provided in
Appendix C.1.

∂w(t)⊤
j1
W (t)
V µ
∂t

16


---Page Break---
= 1

n

n
X

i2=1
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E 
X(i2)p(t,i2)
l2

⊤
µ + 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ

(6)

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
K(t,i) · diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l

p(t,i)
l
x(i)⊤
l
µ

+
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)⊤
l
· diag

w(t)⊤
j
V (t,i) −w(t)⊤
j
V (t,i)p(t,i)
l

X(i)⊤ν

(7)

∂
D
w(t)
j1 , w(t)
j2
E

∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
aj2w(t)⊤
j1
V (t,i)p(t,i)
l
+ 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
aj1w(t)⊤
j2
V (t,i)p(t,i)
l

∂ν⊤W (t)⊤
V
W (t)
V µ
∂t

= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ + 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν

∂ν⊤W (t)⊤
Q
W (t)
Q µ

∂t

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
ajν⊤W (t)⊤
Q
K(t,i) · diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
l(i,µ)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

m1
X

j=1
ajµ⊤W (t)⊤
Q
K(t,i) · diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,ν)

p(t,i)
l(i,ν)

∂ν⊤W (t)⊤
K
W (t)
K µ
∂t

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←µ ·

w(t)⊤
j
v(t,i)(µ) −w(t)⊤
j
V (t,i)p(t,i)
l


+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←ν ·

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l


C.1
Derivation of the Dynamical System

Lemma C.1. Let µ ∈{µi}d
i=1. For all j ∈[m], we have

∂w(t)⊤
j1
W (t)
V µ
∂t
= 1

n

n
X

i2=1
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E 
X(i2)p(t,i2)
l2

⊤
µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ

= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1
aj

∥w(t)
j ∥2
2 + ∥v(t)(µ)∥2
2

p(t,i)
q←l,k←µ + ε,

17


---Page Break---
where

ε = 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ

+ 1

n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ)
E
I(v(t,i)
l2
̸= v(t)(µ)).

Proof. Let l(i, µ) denote the index such that X(i)
l(i,µ) = µ. By the gradient flow update, we have

∂w(t)⊤
j1
W (t)
V µ
∂t
= 1

n

n
X

i2=1
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E 
X(i2)p(t,i2)
l2

⊤
µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ

= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E

= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1



aj1∥w(t)
j1 ∥2
2 +
X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E


p(t,i2)
q←l2,k←µ

+ 1

n

X

i1:µ/∈X(i1)
g(t)
i1 yi1

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E

+ 1

n

X

i1:µ∈X(i1)
g(t)
i1 yi1

L
X

l1=1
aj1



∥v(t)(µ)∥2
2p(t,i1)
q←l1,k←µ +
X

l2̸=l(i1,µ)

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E




= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1
aj1

∥w(t)
j1 ∥2
2 + ∥v(t)(µ)∥2
2

p(t,i)
q←l,k←µ

+ 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ

+ 1

n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ)
E
I(v(t,i)
l2
̸= v(t)(µ)).

Lemma C.2. The following equation on the gradient flow holds:

∂
D
w(t)
j1 , w(t)
j2
E

∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1





X

l′:X(i)
l′ ∈U
aj2w(t)⊤
j1
V (t,i)
l′
p(t,i)
l,l′ + aj1w(t)⊤
j2
V (t,i)
l′
p(t,i)
l,l′






+ 1

n

n
X

i=1
g(t)
i yi

L
X

l=1





X

l′:X(i)
l′ /∈U
aj2w(t)⊤
j1
V (t,i)
l′
p(t,i)
l,l′ + aj1w(t)⊤
j2
V (t,i)
l′
p(t,i)
l,l′




.

18


---Page Break---
Proof. By gradient flow update, we have

∂
D
w(t)
j1 , w(t)
j2
E

∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
aj2w(t)⊤
j1
V (t,i)p(t,i)
l
+ 1

n

n
X

i=1
g(t)
i yi

L
X

l=1
aj1w(t)⊤
j2
V (t,i)p(t,i)
l

= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1





X

l′:X(i)
l′ ∈U
aj2w(t)⊤
j1
V (t,i)
l′
p(t,i)
l,l′ + aj1w(t)⊤
j2
V (t,i)
l′
p(t,i)
l,l′






+ 1

n

n
X

i=1
g(t)
i yi

L
X

l=1





X

l′:X(i)
l′ /∈U
aj2w(t)⊤
j1
V (t,i)
l′
p(t,i)
l,l′ + aj1w(t)⊤
j2
V (t,i)
l′
p(t,i)
l,l′




.

Lemma C.3. Let µ, ν ∈{µi}d
i=1. Then the following equation on gradient flow holds:

∂νW (t)⊤
V
W (t)
V µ
∂t
= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ

+ 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν.

Proof. The gradient flow update can be derived as follows:

∂νW (t)⊤
V
W (t)
V µ
∂t
= 1

n

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j

X(i)p(t,i)
l
⊤
µ

+ 1

n

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j

X(i)p(t,i)
l
⊤
ν

= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ

+ 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν.

Lemma C.4. Let µ, ν ∈{µi}d
i=1. Then the following equations on gradient flow hold.

∂ν⊤W (t)⊤
Q
W (t)
Q µ

∂t

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
ajν⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
l(i,µ)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

m1
X

j=1
ajµ⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,ν)

p(t,i)
l(i,ν),

∂ν⊤W (t)⊤
K
W (t)
K µ
∂t

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←µ

w(t)⊤
j
v(t,i)(µ) −w(t)⊤
j
V (t,i)p(t,i)
l


19


---Page Break---
+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

,

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

=
1
n√m

X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2

v(t)⊤(ν)w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←ν

+
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
aj

L
X

l=1
ν⊤W (t)⊤
K
K(t,i)
l

V (t,i)⊤
l
w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←lI(K(t,i)
l
̸= k(t)(ν))

+
1
n√m

X

i:ν,µ∈X(i)
g(t)
i yi

m1
X

j=1
aj∥q(t)(µ)∥2
2p(t,i)
q←µ,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)


+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

I(q(t,i)
l
̸= q(t)(µ)).

Proof. To prove the first result, we have

∂ν⊤W (t)⊤
Q
W (t)
Q µ

∂t

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l

p(t,i)
l
x(i)⊤
l
µ

+
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l

p(t,i)
l
x(i)⊤
l
ν

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
ajν⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
l(i,µ)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

m1
X

j=1
ajµ⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,ν)

p(t,i)
l(i,ν).

To prove the second result, we have

∂ν⊤W (t)⊤
K
W (t)
K µ
∂t

=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
q(t,i)
l
p(t,i)⊤
l
diag

w(t)⊤
j
V (t,i) −w(t)⊤
j
V (t,i)p(t,i)
l

X(i)⊤µ

+
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
K
q(t,i)
l
p(t,i)⊤
l
diag

w(t)⊤
j
V (t,i) −w(t)⊤
j
V (t,i)p(t,i)
l

X(i)⊤ν

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←µ

w(t)⊤
j
v(t,i)(µ) −w(t)⊤
j
V (t,i)p(t,i)
l


+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
K
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

.

To prove the third result, we have

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

20


---Page Break---
=
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
K
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l

p(t,i)
l
x(i)⊤
l
µ

+
1
n√m

n
X

i=1
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)⊤
l
diag

w(t)⊤
j
V (t,i) −w(t)⊤
j
V (t,i)p(t,i)
l

X(i)⊤ν

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
ajν⊤W (t)⊤
K
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
l(i,µ)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l


=
1
n√m

X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2

v(t)⊤(ν)w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←ν

+
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
aj

L
X

l=1
ν⊤W (t)⊤
K
K(t,i)
l

V (t,i)⊤
l
w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←lI(K(t,i)
l
̸= k(t)(ν))

+
1
n√m

X

i:ν,µ∈X(i)
g(t)
i yi

m1
X

j=1
aj∥q(t)(µ)∥2
2p(t,i)
q←µ,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)


+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

I(q(t,i)
l
̸= q(t)(µ)).

D
Initialization

Lemma D.1. With probability at least 1 −δ over the randomness of the initialization of WK and
WQ, for any l1, l2 ∈[d], we have


D
W (0)
K µl1, W (0)
Q µl2
E ≤σ2
0m

 r

4
m log 2d

δ + 4

m log 2d

δ

!

,


D
W (0)
K µl1, W (0)
K µl2
E ≤σ2
0m

 r

4
m log 2d

δ + 4

m log 2d

δ

!

,
l1 ̸= l2


D
W (0)
Q µl1, W (0)
Q µl2
E ≤σ2
0m

 r

4
m log 2d

δ + 4

m log 2d

δ

!

,
l1 ̸= l2

and for any l ∈[d],

∥W (0)
K µl∥2
2 = σ2
0m

 

1 ±

 r

4
m log 2d

δ + 4

m log 2d

δ

!!

,

∥W (0)
Q µl∥2
2 = σ2
0m

 

1 ±

 r

4
m log 2d

δ + 4

m log 2d

δ

!!

.

Proof. Note that W (0)
K µl1, W (0)
Q µl2 ∼N(0, σ2
0I). The rest of proof applies Lemma H.2.

Corollary D.2. For all i ∈[n], l, k ∈[L], we have

p(0,i)
l,k
= 1

L ± eO
 1

Lm


.

21


---Page Break---
Proof. Following from Lemma G.2 and from the first-order Taylor approximation on the softmax
function from 0, we have

p(0,i)
l,k
= 1

L ± O

 
σ2
0m
L√m

 r

4
m log 2d

δ + 4

m log 2d

δ

!!

.

The corollary then follows from Condition 1.

Lemma D.3. With probability at least 1 −δ over the randomness of the initialization of W and WV ,
then for l1 ̸= l2 ∈[d], we have


D
W (0)
V µl1, W (0)
V µl2
E ≤σ2
0m

 r

4
m log 2d

δ + 4

m log 2d

δ

!

,

for j1 ̸= j2 ∈[m1], we have


D
w(0)
j1 , w(0)
j2
E ≤σ2
1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

!

,

and for all j ∈[m1], l ∈[d], we have


D
w(0)
j , W (0)
V µl
E ≤σ0σ1m

 r

4
m log 2m1d

δ
+ 4

m log 2m1d

δ

!

∥wj∥2
2 = σ2
1m

 

1 ±

 r

4
m log 2m1

δ
+ 4

m log 2m1

δ

!!

∥W (0)
V µl∥2
2 = σ2
0m

 

1 ±

 r

4
m log 2d

δ + 4

m log 2d

δ

!!

.

Proof. The proof is similar to that for Lemma D.1 and is omitted.

Lemma D.4. Conditioned on the success of the event in Corollary D.2, for all i ∈[n], l ∈[L], with
probability at least 1 −δ over the randomness in the initialization of WV ,

∥W (0)
V X(i)p(0,i)
l
∥2
2 = σ2
0m

L

 

1 ±

r

4
m log 2nL

δ
± 4

m log 2nL

δ

!

Proof. First of all, by Corollary D.2 and Assumption 2.3,

E
h
∥W (0)
V X(i)p(0,i)
l
∥2
2
i
= E




m
X

j=1


W (0)
V


j , X(i)p(0,i)
l

2


= σ2
0m∥X(i)p(0,i)
l
∥2
2 = σ2
0m

L .

Finally, applying Bernstein’s inequality and taking a union bound over [n] and [L] we finish the
proof.

Corollary D.5. Conditioned on the success of Lemma D.4, with probability at least 1 −δ over the
randomness of W, for all j ∈[m1], i ∈[n], l ∈[L], we have
w(0)⊤
j
V (0,i)p(0,i)
l
 ≤σ1σ0

r

m

L log m1nL

δ
.

Proof. Conditioned on V (0,i)p(0,i)
l
, we have w(0)⊤
j
V (0,i)p(0,i)
l
∼N(0, σ2
1∥V (0,i)p(0,i)
l
∥2
2). Thus, by
Gaussian tail bound and a union bound over j ∈[m1], i ∈[n], l ∈[L], with probability at least
1 −δ, we have
w(0)⊤
j
V (0,i)p(0,i)
l
 ≤σ1σ0

r

m

L log m1nL

δ
.

22


---Page Break---
Lemma D.6. With probability at least 1 −δ over the randomness in the initialization of a, we have

|S+1| = m1



1

2 ±

s

2 log(4/δ)

m1



,

|S−1| = m1



1

2 ±

s

2 log(4/δ)

m1



.

Proof. The proof follows by applying Hoeffding’s inequality.

Lemma D.7 (Initial sub-network output). Assume that the success of the events in Lemma D.3 holds.
For all i ∈[d], with probability at least 1 −δ over the randomness in the weight initialization, we
have

|G(0)(µi)| ≤eO(σ1σ0
√mm1).

Proof. Consider a fixed i ∈[d]. By Lemma D.3, we have ∥W (0)
V µi∥2
2 = Θ(σ2
0m). Thus, conditioned
on W (0)
V µi, we have Pm1
j=1 w(0)
j W (0)
V µi ∼N(0, σ2
1σ2
0mm1). Thus, by Gaussian concentration

bound, we have |G(0)(µi)| ≤eO(σ1σ0√mm1).

Lemma D.8 (Initial network output). Assume that the success of the events in Lemma D.3 and
Lemma D.4 holds. For all i ∈[n], with probability at least 1 −δ over the randomness in the weight
initialization, we have

|F (0)(X(i))| ≤2σ0σ1
p

2Lm1m log(2Ln/δ) ≤0.01.

Proof. For fixed l ∈L, j ∈[m], i ∈[n], by Corollary D.5, we have

w(0)⊤
j
V (0,i)p(0,i)
l
 ≤σ1σ0

r

m

L log m1nL

δ
.

Thus, this implies that ajw(0)⊤
j
V (0,i)p(0,i)
l
is a sub-Gaussian random variable with variance proxy
σ2
0σ2
1
m

L log m1nL

δ
. Therefore, the following inequality holds.

P







m1
X

j=1
ajw(0)⊤
j
V (0,i)p(0,i)
l


≥2σ0σ1

s

2m1m(log m1nL

δ
) log(2/δ)
L



≤δ.

Taking a union bound over i ∈[n], l ∈[L], with probability at least 1 −δ, for all i ∈[n], we have

|F (0)(X(i))| ≤2σ0σ1

r

2Lm1m(log m1nL

δ
) log(2Ln/δ).

Finally, by Condition 1, we can make |F (0)(X(i))| ≤0.01.

E
Training Dynamics: Phase 1

During Phase 1 of training, the linear layer quickly aligns with the target signals and all the remaining
quantities stay roughly the same. The analysis need to keep track of the evolution of the above
quantities with respect to the two signals µ1, µ2, the common token µ3 and the random tokens.

Definition E.1 (Radius of keys and queries). Define the radius of keys and queries RK, RQ respec-
tively to be

RK := max
i,j∈[d]

µ⊤
i W (t)⊤
K
W (t)
K µj −µ⊤
i W (0)⊤
K
W (0)
K µj
 ,

RQ := max
i,j∈[d]

µ⊤
i W (t)⊤
Q
W (t)
Q µj −µ⊤
i W (0)⊤
Q
W (0)
Q µj
 .

23


---Page Break---
Definition E.2 (Phase 1). Define the range of Phase 1 to be [0, T1],
where T1
=
min{t′, CT1/(σ2
1mm1)} for some sufficiently large constant CT1 and t′ is defined to be the maximum
time such that for all t ≤t′, all of the following hold:

1. maxj∈[m],µ∈{µi}3
i=1

w(t)
j W (t)
V µ −w(0)
j W (0)
V µ
 ≤R where R < O(1/m1);

2. maxj∈[m],µ/∈{µi}3
i=1

w(t)
j W (t)
V µ −w(0)
j W (0)
V µ
 ≤O(R/n + R/√m);

3. maxµ,ν∈{µ}d
i=1

µ⊤W (t)⊤
Q
W (t)
K ν −µ⊤W (0)⊤
Q
W (0)
K ν
 ≤RS where RS ≤O(1/(m√m));

4. RK, RQ ≤eO(σ2
0
√m).

Based on this definition, we can further obtain the maximum softmax probability change as follows.
Proposition E.3. Define

RP :=
max
i∈[n], µ,ν∈X(i)

p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν
 .

Then

RP = O

1
√mL + L

m


.

Proof. By Lemma G.2, we have RP ≤O(RS/L + R2
SL) = O

1
√mL + L

m

.

Initially, the loss for the samples with one signal will increase.

E.1
Initial Gradients

Lemma E.4 (Signal updates, same as Lemma 4.1). At t = 0, for µ ∈{µ1, µ2}, we have

∂ajw(0)
j W (0)
V µ
∂t
= Θ((σ2
0 + σ2
1)m).

Proof. Take µ = µ1. First of all, by the gradient flow update in Lemma C.1, we have

∂w(t)⊤
j1
W (t)
V µ
∂t
= 1

n

n
X

i2=1
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E 
X(i2)p(t,i2)
l2

⊤
µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ

= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E

= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1



aj1∥w(t)
j1 ∥2
2 +
X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E


p(t,i2)
q←l2,k←µ

+ 1

n

X

i1:µ/∈X(i1)
g(t)
i1 yi1

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E

+ 1

n

X

i1:µ∈X(i1)
g(t)
i1 yi1

L
X

l1=1
aj1



∥v(t)(µ)∥2
2p(t,i1)
q←l1,k←µ +
X

l2̸=l(i1,µ)

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E




24


---Page Break---
= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1
aj1

∥w(t)
j1 ∥2
2 + ∥v(t)(µ)∥2
2

p(t,i)
q←l,k←µ

+ 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ

|
{z
}
ε1

+ 1

n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ)
E
I(v(t,i)
l2
̸= v(t)(µ))

|
{z
}
ε2

.

Now, by Lemma D.3, Corollary D.2 and Lemma D.8, we have

aj1
1
n

X

i:µ1∈X(i)
g(0)
i
yi

L
X

l=1
aj1

∥w(0)
j1 ∥2
2 + ∥v(0)(µ)∥2
2

p(0,i)
q←l,k←µ1

= 1

n

X

i∈I1
g(0)
i

L
X

l=1


∥w(0)
j ∥2
2 + ∥v(0)(µ1)∥2
2

p(0,i)
q←l,k←µ1 −1

n

X

i∈I2
g(0)
i

L
X

l=1


∥w(0)
j ∥2
2 + ∥v(0)(µ1)∥2
2

p(0,i)
q←l,k←µ1

=
1

3 ± 0.01

L ·
 
σ2
1m + σ2
0m

 

1 ±

 r

4
m log 2d

δ + 4

m log 2d

δ

!!
1
L(1 + o(1)).

On the other hand, by Proposition E.5, we have

|ε1| =



1
n

X

i:µ1∈X(i)
g(0)
i
yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ1



≤σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log m1

δ ;

|ε2| =


1
n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ1)
E
I(v(0,i)
l2
̸= v(0)(µ1))



≤σ2
0m
√

L

 r

4
m log 2nL

δ
+ 4

m log 2nL

δ

!

.

If m ≥Cm1 log2 m1

δ in Condition 1 for some sufficiently large C, then |ε1| ≤0.01σ2
1m; and if
m ≥C′L log 2nL

δ
for some sufficiently large C′, then |ε2| ≤0.01σ2
0m.

Proposition E.5. Assume the events in Lemma D.3 and Corollary D.2 succeed. With probability at
least 1 −δ over the randomness in the weight initialization, for all j1 ∈[m1], we have


X

j2:j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E

≤σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log 4m1

δ
.

Further, for all µ ∈{µi}d
i=1, we have


1
n

X

i:µ∈X(i)
g(0)
i
yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ



≤|i : µ ∈X(i)|

n
σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log m1

δ ,

and

1
n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ)
E
I(v(0,i)
l2
̸= v(0)(µ))



25


---Page Break---
≤σ2
0m
√

L

 r

4
m log 2nLd

δ
+ 4

m log 2nLd

δ

!

.

Proof. First, fix i, l, and consider the randomness of a. By Lemma D.3, aj2
D
w(0)
j1 , w(0)
j2
E
is a sub-

Gaussian random variable with variance proxy σ4
1m2
q

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

2
. This implies

that with probability at least 1 −δ/2, for all j1 ∈[m1], we have


X

j2:j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E

≤σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log 4m1

δ
.

Thus, by Corollary D.2, for all µ ∈{µi}d
i=1, we have


1
n

X

i:µ1∈X(i)
g(0)
i
yi

L
X

l=1

X

j2:j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ



≤|i : µ ∈X(i)|

n
σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log m1

δ .

We next derive the second inequality. Consider the randomness in W (0)
V . Note that

L
X

l2=1
p(0,i)
l1,l2v(0,i)
l2
I(v(0,i)
l2
̸= v(0)(µ)) ∼N

 

0, σ2
0

L
X

l2=1
(p(0,i)
l1,l2)2I(v(0,i)
l2
̸= v(0)(µ))I

!

.

Thus, by Lemma H.2 and Corollary D.2 and taking a union bound over i ∈[n], l1 ∈[L], µ ∈
{µi}d
i=1, we have with probability at least 1 −δ/2,


* L
X

l2=1
p(0,i)
l1,l2v(0,i)
l2
I(v(0,i)
l2
̸= v(0)(µ)), v(0)(µ)

+ ≤σ2
0m 1
√

L

 r

4
m log 2nLd

δ
+ 4

m log 2nLd

δ

!

.

Therefore,

1
n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ)
E
I(v(0,i)
l2
̸= v(0)(µ))



≤σ2
0m
√

L

 r

4
m log 2nLd

δ
+ 4

m log 2nLd

δ

!

.

Lemma E.6 (Random token updates). For µ ∈{µi}d
i=4, we have

∂ajw(0)
j W (0)
V µ
∂t
= O
 1

n(σ2
0 + σ2
1)m + σ2
0
√

mL

.

Proof. Following the proof of Lemma E.4, we have

∂ajw(0)
j W (0)
V µ
∂t
= 1

n

n
X

i2=1
g(0)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj2
D
w(0)
j1 , w(0)
j2
E 
X(i2)p(0,i2)
l2

⊤
µ

+ 1

n

n
X

i1=1
g(0)
i1 yi1

L
X

l1=1
aj1p(0,i1)⊤
l1
V (0,i1)⊤W (0)
V µ

= 1

n

X

i:µ∈X(i)
g(0)
i
yi

L
X

l=1
aj1

∥w(0)
j1 ∥2
2 + ∥v(0)(µ)∥2
2

p(0,i)
q←l,k←µ

26


---Page Break---
+ 1

n

X

i:µ∈X(i)
g(0)
i
yi

L
X

l=1

X

j2:j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ

|
{z
}
ε1

+ 1

n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ)
E
I(v(0,i)
l2
̸= v(0)(µ))

|
{z
}
ε2

By Proposition E.5 and the fact that only one X(i) satisfies µ ∈X(i), we have

1
n

X

i:µ∈X(i)
g(0)
i
yi

L
X

l=1
aj1

∥w(0)
j1 ∥2
2 + ∥v(0)(µ)∥2
2

p(0,i)
q←l,k←µ = Θ
 1

n(σ2
0 + σ2
1)m

,

and

|ε1| ≤1

nσ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log 4m1

δ
,

|ε2| ≤σ2
0m
√

L

 r

4
m log 2nLd

δ
+ 4

m log 2nLd

δ

!

.

E.2
Maximum Perturbation of Neuron Outputs

Lemma E.7. For all t ≤T1, for all i ∈[n], l ∈[L], we have
w(t)
j W (t)
V X(i)p(t,i)
l
−w(0)
j W (0)
V X(i)p(0,i)
l
 ≤eO(LRP σ0σ1
√m) + 4R/L.

Proof. By Definition E.2, we have
w(t)
j V (t,i)p(t,i)
l
−w(0)
j V (0,i)p(0,i)
l


≤

L
X

l′=1

w(t)
j V (t,i)
l′
p(t,i)
l,l′ −w(0)
j V (0,i)
l′
p(0,i)
l,l′


≤
X

l′:Vl′∈v(µ1,µ2,µ3)
(RP eO(σ0σ1
√m) + R/L) +
X

l′:Vl′ /∈v(µ1,µ2,µ3)
(RP eO(σ0σ1
√m) + R/(nL))

≤eO(RP σ0σ1
√m + R/L) + eO(LRP σ0σ1
√m + R/n)

= eO(LRP σ0σ1
√m) + 4R/L.

By our choice of parameters in Condition 1 and Definition E.2, we have eO(LRP σ0σ1
√m)+4R/L <
σ0σ1
p

m/L.

E.3
Perturbation Term Involving Correlation of Value-transformed Data

Proposition E.8. With probability at least 1 −δ, for all µ ∈{µi}d
i=1, we have


m1
X

j=1
ajµ⊤W (0)⊤
V
w(0)
j


≤eO(σ0σ1
√mm1).

Proof. The proof is similar to Proposition E.22, and is omitted.

27


---Page Break---
Lemma E.9 (Value correlation change). For all µ, ν ∈{µi}d
i=1, we have

∂

v(t)(µ), v(t)(ν)


∂t

 ≤


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
O(1),

and thus,

D
v(t)(µ), v(t)(ν)
E
−
D
v(0)(µ), v(0)(ν)
E ≤t


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
O(1)

for all t ≤T1. Thus, for µ ̸= ν, we have

v(t)(µ), v(t)(ν)
 ≤eO(σ2
0
√m) and ∥v(t)(µ)∥2
2 =
Θ(σ2
0m) for t ≤T1.

Proof. By Lemma C.3, we have

∂νW (t)⊤
V
W (t)
V µ
∂t
= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ

+ 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν.

Further,


m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j
−

m1
X

j=1
ajν⊤W (0)⊤
V
w(0)
j


≤m1R.

Thus, by Proposition E.8, we have


m1
X

j=1
ajw(t)⊤
j
W (t)
V ν


≤



m1
X

j=1
aj ˙σ(0)
i,l,jw(0)⊤
j
W (0)
V ν


+



m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j
−

m1
X

j=1
ajν⊤W (0)⊤
V
w(0)
j



≤eO(σ0σ1
√mm1) + m1R,
which implies
1
n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ

+ 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν



≤1

n


n
i : µ ∈X(i)o +

n
i : ν ∈X(i)o
 
eO(σ0σ1
√mm1) + m1R


≤


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
O(1)

where the last inequality applies Lemma E.7.

Corollary E.10. For all t ≤T1, we have

1
n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ1)
E
I(v(t,i)
l2
̸= v(t)(µ1))

 ≤L · eO
 
σ2
0
√m

.

Proof. We derive the following bound:

1
n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ)
E
I(v(t,i)
l2
̸= v(t)(µ))



≤

L
X

l1=1

L
X

l2=1
p(t,i)
l1,l2


D
v(t,i)
l2
, v(t)(µ)
E I(v(t,i)
l2
̸= v(t)(µ))

≤L · eO
 
σ2
0
√m

,
where the last inequality follows from Lemma E.9 and Lemma D.3.

28


---Page Break---
E.4
Perturbation Term Involving Correlation of Neurons

Lemma E.11. For all t ≤T1, we have


1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ1


≤|{i : µ1 ∈X(i)}|

n
eO(σ2
1
√mm1).

Proof. We first derive:


1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ1



≤



1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E

· max
i,l p(t,i)
q←l,k←µ1

≤



1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E

· max
i,l p(t,i)
q←l,k←µ1

|
{z
}
(1)

+



1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
−
D
w(0)
j1 , w(0)
j2
E

· max
i,l p(t,i)
q←l,k←µ1

|
{z
}
(2)

.

For term (1), by Proposition E.5, we have


1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E

≤|{i : µ1 ∈X(i)}|

n
eO
 
σ2
1
√mm1L

.

For term (2), note that


1
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
−
D
w(0)
j1 , w(0)
j2
E


≤|{i : µ1 ∈X(i)}|

n
Lm1 · O
 1

m


,

where the inequality is by Lemma D.3, Proposition E.12.

Finally, since p(t,i)
q←l,k←µ1 ≤1

L +
1
L2 + RP , combining the upper bound for both terms (1) and (2),
we have


α
n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2 ˙σ(t)
i,l,j2

D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ1



≤α|{i : µ1 ∈X(i)}|

n
· eO
 
σ2
1
√mm1L

.

Proposition E.12 (Neuron correlation change, Phase 1). For t ≤T1, for all j1, j2 ∈[m1], we have


∂
D
w(t)
j1 , w(t)
j2
E

∂t


≤2L

 

σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L

!

,

and thus,

D
w(t)
j1 , w(t)
j2
E
−
D
w(0)
j1 , w(0)
j2
E ≤t2L

 

σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L

!

.

29


---Page Break---
Proof. By the gradient flow update in Lemma C.2, we have

∂
D
w(t)
j1 , w(t)
j2
E

∂t
= 1

n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj2w(t)⊤
j1
V (t,i′)p(t,i′)
l′
+ 1

n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj1w(t)⊤
j2
V (t,i′)p(t,i′)
l′

= 1

n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj2

w(0)⊤
j1
V (0,i′)p(0,i′)
l′
+ (w(t)⊤
j1
V (t,i′)p(t,i′)
l′
−w(0)⊤
j1
V (0,i′)p(0,i′)
l′
)


+ 1

n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj1

w(0)⊤
j2
V (0,i′)p(0,i′)
l′
+ (w(t)⊤
j2
V (t,i′)p(t,i′)
l′
−w(0)⊤
j2
V (0,i′)p(0,i′)
l′
)

.

Now, since by Lemma E.7 we have
w(t)
j W (t)
V X(i)p(t,i)
l
−w(0)
j W (0)
V X(i)p(0,i)
l
 ≤eO(LRP σ0σ1
√m) + 4R/L,

by Definition E.2 and Corollary D.5, we have

1
n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj2

w(0)⊤
j1
V (0,i′)p(0,i′)
l′
+ (w(t)⊤
j1
V (t,i′)p(t,i′)
l′
−w(0)⊤
j1
V (0,i′)p(0,i′)
l′
)


≤L

 

σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L

!


1
n

n
X

i′=1
g(t)
i′ yi′

L
X

l′=1
aj1

w(0)⊤
j2
V (0,i′)p(0,i′)
l′
+ (w(t)⊤
j2
V (t,i′)p(t,i′)
l′
−w(0)⊤
j2
V (0,i′)p(0,i′)
l′
)


≤L

 

σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L

!

.

Thus,

D
w(t)
j1 , w(t)
j2
E
−
D
w(0)
j1 , w(0)
j2
E

≤
Z t

τ=0



∂
D
w(τ)
j1 , w(τ)
j2
E

∂τ


≤t2L

 

σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L

!

.

Corollary E.13 (Neuron Norm Change, Phase 1). For all t ≤T1 and all j ∈[m], we have

∥w(t)
j ∥2
2 −∥w(0)
j ∥2
2
 ≤t2L(σ1σ0

r

m

L log m1nL

δ
+ eO(LRP σ0σ1
√m) + 4R/L).

E.5
Neuron Weights Align with Signal Value

Theorem E.14 (Signal correlation growth, phase 1). For t ≤T1, for µ ∈{µ1, µ2},

∂ajw(t)
j W (t)
V µ
∂t
= 1

n

 
X

i:µ∈X(i),
yi=1

g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ −
X

i:µ∈X(i),
yi=−1

g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ

!

Θ((σ2
0 + σ2
1)m) + ε

where

|ε| ≤eO(Lσ2
0
√m + σ2
1
√mm1).

Proof. We take µ = µ1 and the proof is similar for µ = µ2. By Lemma C.1, we have

∂ajw(t)
j W (t)
V µ
∂t

30


---Page Break---
= 1

n

X

i:µ1∈X(i),yi=1
g(t)
i

L
X

l=1


∥w(t)
j ∥2
2 + ∥v(t)(µ1)∥2
2

p(t,i)
q←l,k←µ1

−1

n

X

i:µ1∈X(i),yi=−1
g(t)
i

L
X

l=1


∥w(t)
j ∥2
2 + ∥v(t)(µ1)∥2
2

p(t,i)
q←l,k←µ1 + ε

=

∥w(t)
j ∥2
2 + ∥v(t)(µ1)∥2
2
 1

n

 
X

i:µ1∈X(i),
yi=1

g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1 −
X

i:µ1∈X(i),
yi=−1

g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1

!

+ ε

where

ε = 1

n

X

i:µ1∈X(i)
g(t)
i yi

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i)
q←l,k←µ1

+ 1

n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ1)
E
I(v(t,i)
l2
̸= v(t)(µ1)).

We can now bound the magnitude of ε by Corollary E.10, Lemma E.11 and Proposition E.5 and
obtain:

|ε| ≤L · eO
 
σ2
0
√m

+ eO(σ2
1
√mm1).

Finally, by Lemma D.3 and Corollary E.13, we have ∥w(t)
j ∥2
2 = Θ(σ2
1m) and by Lemma E.9, we
have ∥v(t)(µ1)∥2
2 = Θ(σ2
0m). The proof is completed.

Theorem E.15 (Random token growth, Phase 1). For t ≤T1, for µ ∈R, we have

∂ajw(t)
j W (t)
V µ
∂t
= O
 1

n(σ2
0 + σ2
1)m

+ eO(σ2
0L√m).

Proof. Fix a µ ∈R. By our Assumption 2.3, µ appears at most once in the training data set. Now,
assume µ is in the training set and let i⋆be the index of the sample containing µ. Applying Lemma C.1
on the random token µ, we have

∂ajw(t)
j W (t)
V µ
∂t
= 1

ng(t)
i⋆yi⋆

L
X

l=1
aj

∥w(t)
j ∥2
2 + ∥v(t)(µ)∥2
2

p(t,i⋆)
q←l,k←µ + ε,

where

ε = 1

ng(t)
i⋆yi⋆

L
X

l=1

X

j2̸=j1
aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i⋆)
q←l,k←µ

+ 1

n

n
X

i=1
g(t)yi

L
X

l1=1
aj1

L
X

l2=1

D
p(t,i)
l1,l2v(t,i)
l2
, v(t)(µ)
E
I(v(t,i)
l2
̸= v(t)(µ)).

We can now bound the magnitude of ε by Corollary E.10 and Lemma E.11 as follows:

|ε| ≤L · eO
 
σ2
0
√m

+ 1

n
eO(σ2
1
√mm1).

Finally, by Lemma D.3 and Corollary E.13, we have

∂ajw(t)
j W (t)
V µ
∂t
= O
 1

n(σ2
0 + σ2
1)m

+ eO(σ2
0L√m).

31


---Page Break---
E.6
Alignment of Common Token

Lemma E.16 (Initial Per Neuron Gradient of Common Token, same as Lemma 4.2). Let F =
maxi F (0)
i
. We have

∂ajw(0)
j W (0)
V µ3
∂t

 = eO

σ2
1
√mm1 + σ2
0
√

mL + (σ2
0 + σ2
1)mF

.

Proof. Following from the proof of Lemma E.4, we have

∂aj1w(0)
j1 W (0)
V µ3
∂t
= 1

n

X

i:µ3∈X(i)
g(0)
i
yi

L
X

l=1


∥w(0)
j1 ∥2
2 + ∥v(0)(µ1)∥2
2

p(0,i)
q←l,k←µ3

+ aj1
1
n

X

i:µ3∈X(i)
g(0)
i
yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ3

|
{z
}
ε1

+ aj1
1
n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ3)
E
I(v(0,i)
l2
̸= v(0)(µ3))

|
{z
}
ε2

.

For the first term, we have


1
n

X

i:µ3∈X(i)
g(0)
i
yi

L
X

l=1


∥w(0)
j1 ∥2
2 + ∥v(0)(µ3)∥2
2

p(0,i)
q←l,k←µ3



=


1
n

X

yi=1
g(0)
i

L
X

l=1


∥w(0)
j1 ∥2
2 + ∥v(0)(µ3)∥2
2

p(0,i)
q←l,k←µ3

−1

n

X

yi=−1
g(0)
i

L
X

l=1


∥w(0)
j1 ∥2
2 + ∥v(0)(µ3)∥2
2

p(0,i)
q←l,k←µ3



= 1

n


∥w(0)
j1 ∥2
2 + ∥v(0)(µ3)∥2
2
 

X

yi=1
g(0)
i

L
X

l=1
p(0,i)
q←l,k←µ3 −
X

yi=−1
g(0)
i

L
X

l=1
p(0,i)
q←l,k←µ3

 .

By Lemma D.8, and Corollary D.2, we have

1
n



X

yi=1
g(0)
i

L
X

l=1
p(0,i)
q←l,k←µ3 −
X

yi=−1
g(0)
i

L
X

l=1
p(0,i)
q←l,k←µ3



≤1

n

 X

yi=1

1

2 + F

L
X

l=1

 1

L + 1

L2


−
X

yi=−1

1

2 −F

L
X

l=1

 1

L −1

L2

!

≤
1

2 + F
  1

L +
1
Lm

 L

2 −
1

2 −F
  1

L −
1
Lm

 L

2
≤2F + O(1/m).

By Lemma D.3, we have ∥w(0)
j1 ∥2
2 = Θ(σ2
1m) and ∥v(0)(µ3)∥2
2 = Θ(σ2
0m), and thus,


1
n

X

i:µ3∈X(i)
g(0)
i
yi

L
X

l=1
aj1

∥w(0)
j1 ∥2
2 + ∥v(0)(µ3)∥2
2

p(0,i)
q←l,k←µ3


≤eO(α(σ2
0 + σ2
1)mF).

Further, by Proposition E.5, we have

|ε1| =



1
n

X

i:µ3∈X(i)
g(0)
i
yi

L
X

l=1

X

j2̸=j1
aj2
D
w(0)
j1 , w(0)
j2
E
p(0,i)
q←l,k←µ3



32


---Page Break---
≤σ2
1
√m1m

 r

4
m log 2m2
1
δ
+ 4

m log 2m2
1
δ

! r

log m1

δ .

|ε2| =


1
n

n
X

i=1
g(0)
i
yi

L
X

l1=1
aj1

L
X

l2=1

D
p(0,i)
l1,l2v(0,i)
l2
, v(0)(µ3)
E
I(v(0,i)
l2
̸= v(0)(µ3))



≤σ2
0m
√

L

 r

4
m log 2nL

δ
+ 4

m log 2nL

δ

!

.

Finally, we have

∂aj1w(0)
j1 W (0)
V µ3
∂t

 ≤eO

σ2
1
√mm1 + σ2
0
√

mL + (σ2
0 + σ2
1)mF

.

Lemma E.17 (Per neuron gradient of common token). For t ≤T1, we have

∂ajw(0)
j W (0)
V µ3
∂t

 ≤eO(σ2
1m)

Proof. The proof is similar to that for Theorem E.14 and we omit it here.

Definition E.18 (sub-network). Define the sub-network structure as

G(µ) =

m1
X

j=1
ajwjWV µ.

We can compute the gradient of the sub-network as

∂G(µ)

∂t
=

m1
X

j=1

∂ajw(t)
j W (t)
V µ
∂t
=

m1
X

j=1
ajw(t)
j
∂W (t)
V µ
∂t
+ aj
∂w(t)
j
∂t W (t)
V µ

= 1

n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ

+ 1

n

n
X

i1=1
g(t)
i1 yi1m1

L
X

l1=1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E
.

Theorem E.19 (Complete version of Lemma 4.3). There exists a T0.5 ≤T1 and a constant 0 < C < 1
such that for all T0.5 ≤t ≤T1,

(1 + C) max
∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t


≤−∂G(t)(µ3)

∂t
≤(1 −C)
∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t


.

Further,

∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t
≥Ω((σ2
0 + σ2
1)mm1).

Proof. First of all, we have

∂F (t)
i
∂t
=

L
X

l1=1

m1
X

j=1

L
X

l2=1

 
∂ajw(t)
j W (t)
V Xl2
∂t
p(t,i)
l1,l2 + ajw(t)
j W (t)
V Xl2
∂p(t,i)
l1,l2
∂t

!

.

By Lemma E.4 and Lemma E.16, we have

∂ajw(0)
j W (0)
V µ1
∂t
−
∂ajw(0)
j W (0)
V µ3
∂t
= Θ((σ2
0 + σ2
1)m),

33


---Page Break---
∂ajw(0)
j W (0)
V µ2
∂t
−
∂ajw(0)
j W (0)
V µ3
∂t
= Θ((σ2
0 + σ2
1)m).

By Theorem E.15 and Corollary E.24, we have

L
X

l1=1

m1
X

j=1

L
X

l2=1

 
∂ajw(t)
j W (t)
V Xl2
∂t
I(Xl2 /∈{µi}3
i=1)p(t,i)
l1,l2 + ajw(t)
j W (t)
V Xl2
∂p(t,i)
l1,l2
∂t

!

= O
L

n (σ2
0 + σ2
1)mm1


.

Thus, for all i ∈I1 ∪I2 ∪I3, ∂F (0)
i
∂t
> 0, which implies ∂g(0)
i
∂t
< 0 for i ∈I1 and ∂g(0)
i
∂t
> 0 for
i ∈I2 ∪I3.

Next, we show that there must exist a time such that −∂G(t)(µ3)

∂t
= max( ∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t
).

Without loss of generality, assume ∂G(t)(µ1)

∂t
> ∂G(t)(µ2)

∂t
. We now analyze the condition when the
magnitude of the update of the common token is less than the magnitude of the update of the signals.
By Lemma C.1, we have

m1
X

j=1

∂ajw(t)
j W (t)
V µ1
∂t
≥−

m1
X

j=1

∂ajw(t)
j W (t)
V µ3
∂t

⇔1

n


∥W (t)∥2
F + m1∥v(t)(µ1)∥2
2
  X

i∈I1
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1 −
X

i∈I2
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1

!

+

m1
X

j=1
ajε(t)(µ1)

≥1

n


∥W (t)∥2
F + m1∥v(t)(µ3)∥2
2
  
X

i∈I2∪I3∪I4
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ3 −
X

i∈I1
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ3

!

+

m1
X

j=1
ajε(t)(µ3)

⇔1

n

X

i∈I1
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1 + 1

n
∥W (t)∥2
F + m1∥v(t)(µ3)∥2
2
∥W (t)∥2
F + m1∥v(t)(µ1)∥2
2

X

i∈I1
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ3

≥1

n
∥W (t)∥2
F + m1∥v(t)(µ3)∥2
2
∥W (t)∥2
F + m1∥v(t)(µ1)∥2
2

X

i∈I2∪I3∪I4
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ3 + 1

n

X

i∈I2
g(t)
i

L
X

l=1
p(t,i)
q←l,k←µ1

+

Pm1
j=1 ajε(t)(µ3)

∥W (t)∥2
F + m1∥v(t)(µ1)∥2
2
−

Pm1
j=1 ajε(t)(µ1)

∥W (t)∥2
F + m1∥v(t)(µ1)∥2
2

⇔1

n(1 + o(1))
X

i∈I1
g(t)
i
+ 1

n(1 + o(1))
X

i∈I1
g(t)
i

≥1

n(1 + o(1))
X

i∈I2∪I3∪I4
g(t)
i
+ 1

n(1 + o(1))
X

i∈I2
g(t)
i
± O
rm1

m



where the last equality applies Corollary E.24. If m is sufficiently larger than m1 (by some large
constant factor C), then the last term is negligible. Note that the above implies that if ∂G(t)(µ1)

∂t
≥

∂G(t)(µ3)

∂t
then P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= Ω(1) since P

i∈I3∪I4 g(t)
i
= Ω(1) for all t ≤T1. Now, by

Proposition E.21, if the initialization level is small enough, then P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
≥Ω(1).

Thus, if ∂G(t)(µ1)

∂t
≥−∂G(t)(µ3)

∂t
, then 1

n
P
i∈I1
∂g(t)
i
∂t
= −Θ((σ2
0 + σ2
1)mm1). Therefore, there

must exist a time t′ = Θ(1/m) such that ∂G(t′)(µ1)

∂t
= −∂G(t′)(µ3)

∂t
. Further, for t ≥Θ(F/m),
we have yiF (t)
i
> 0 for all i ∈I4 and P

i∈I4 g(t)
i
< min(P

i∈I2 g(t)
i , P

i∈I3 g(t)
i ), where F =

maxi∈[n] |F (0)
i
|.

34


---Page Break---
Now, consider a time point t′ when
∂G(t′)(µ1)

∂t
=
−∂G(t′)(µ3)

∂t
.
At t′, we must have

min(P

i∈I2 g(t′)
i
, P

i∈I3 g(t′)
i
) −P

i∈I4 g(t′)
i
= Ω(1). Thus,

2
X

i∈I1
g(t′)
i
−2
X

i∈I4
g(t′)
i
= Ω(1),

which implies

∂
∂F 2
X

i∈I1
g(t′)
i
−∂

∂F

X

i∈I4
g(t′)
i
= Ω(1).

For i ∈I1, we have

∂F (t′)
i
∂t
= ∂G(t′)(µ1)

∂t
+ ∂G(t′)(µ2)

∂t
+ ∂G(t′)(µ3)

∂t
+ O
L

n σ2
1mm1



= ∂G(t′)(µ2)

∂t
+ O
L

n σ2
1mm1


.

For i ∈I2,

∂F (t)
i
∂t
= ∂G(t′)(µ1)

∂t
+ ∂G(t′)(µ3)

∂t
+ O
L

n σ2
1mm1


= O
L

n σ2
1mm1


.

For i ∈I3, by Proposition E.21,

∂F (t)
i
∂t
= ∂G(t′)(µ2)

∂t
+ ∂G(t′)(µ3)

∂t
+ O
L

n σ2
1mm1


= O
 
Fσ2
1mm1

,

and for i ∈I4,

∂F (t)
i
∂t
= ∂G(t′)(µ3)

∂t
+ O
L

n σ2
1mm1


.

Thus, by chain rule ∂g(t)
i
∂t
= ∂gi

∂Fi
∂F (t)
i
∂t , we have

∂
∂t

 

2
X

i∈I1
g(t′)
i
−2
X

i∈I2
g(t′)
i
−
X

i∈I3∪I4
g(t′)
i

!

= −Θ(σ2
1mm1).
(8)

This implies that there exists a constant L such that

∂
∂t

 X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
+ (1 + L)

 X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i

!!

= −Θ(σ2
1mm1),

which implies that there must exist a time T0.5 ≤O(1/m) such that for all t ≥T0.5,

−∂G(t)(µ3)

∂t
≥(1 + L)∂G(t)(µ1)

∂t
.

Moreover, by Theorem E.26, if CT1 is sufficiently large, we have T0.5 < T1.

Finally, consider the time when ∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t
> −∂G(t)(µ3)

∂t
. During this time, note that we
have

∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t
≥Θ(σ2
1mm1).

Further, if ∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t
= −∂G(t)(µ3)

∂t
, then ∂

∂tg(t)
i
= −Θ(σ2
1mm1) for all i ∈I2 ∪I3 ∪I4.
Thus, there must exist a constant U such that

−∂G(t)(µ3)

∂t
≤(1 −U)
∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t



for all t ≤T1.

35


---Page Break---
Corollary E.20. There exists a small positive constant C such that if we define T ′ := min{t :
mini∈I2∪I3 F (t)
i
≥C}, then T ′ ≤O(1/m).

Proposition E.21. Let F = maxi∈[n] |F (0)
i
|. For t ≤T1, we have

1
n

X

i∈I2
g(t)
i
−1

n

X

i∈I3
g(t)
i

 ≤O (F) .

Proof. First of all, by Lipschitzness of g, we have

1
n

X

i∈I2
g(t)
i
−1

n

X

i∈I3
g(t)
i

 ≤|G(t)(µ1) −G(0)(µ1) −(G(t)(µ2) −G(0)(µ2))| + 2 max
i∈[n]

F (0)
i
 + o(1).

Without loss of generality, assume G(t)(µ1) > G(t)(µ2) for all t ≤T1; otherwise, we can break the
interval [0, T1] into sub-intervals by the time points when G(t)(µ1) −G(t)(µ2) changes its sign and
then apply the analysis below to each sub-interval. We first derive

∂G(t)(µ1)

∂t
−∂G(t)(µ2)

∂t

=

m1
X

j1=1

 
1
n

X

i2∈I1∪I2
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ1 + 1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ1

−1

n

X

i2∈I1∪I3
g(t)
i2 yi2

L
X

l2=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ2 −1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ2

!

=

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E  
1
n

X

i∈I1
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ1 −1

n

X

i∈I1
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ2

!

+

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E  
1
n

X

i∈I2
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ1 −1

n

X

i∈I3
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ2

!

+ m1

 
1
n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ1 −1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ2

!

=

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E  
1
n

X

i∈I3
g(t)
i
−1

n

X

i∈I2
g(t)
i
+ o(1)

!

+ m1

 
1
n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ1 −1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ2

!

≤

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E  

G(t)(µ1) −G(0)(µ1) −(G(t)(µ2) −G(0)(µ2))

+ G(0)(µ1) + G(0)(µ2) + O

max
i∈[n] |F (0)
i
|

+ o(1)

!

+ m1

 
1
n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ1 −1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ2

!

.

By Theorem E.14, we have

m1

Z T

0

1
n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ1 −1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
aj1p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ2 dt

≤T · eO(Lσ2
0
√mm1),

36


---Page Break---
Z T

0


G(0)(µ1) + G(0)(µ2) + O

max
i∈[n] |F (0)
i
|

+ o(1)
 m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
dt

≤O

G(0)(µ1) + G(0)(µ2) + max
i∈[n] |F (0)
i
|

Tσ2
1mm1


,

and

exp




Z T

0

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
dt



= O(1).

By Grönwall’s inequality, we have

G(T )(µ1) −G(0)(µ1) −(G(T )(µ2) −G(0)(µ2))

≤T · eO(Lσ2
0
√mm1) + O

G(0)(µ1) + G(0)(µ2) + max
i∈[n] |F (0)
i
|

Tσ2
1mm1



+
Z T

0


t · eO(Lσ2
0
√m) + O

G(0)(µ1) + G(0)(µ2) + max
i∈[n] |F (0)
i
|

tσ2
1m

· O(1) dt

≤O

G(0)(µ1) + G(0)(µ2) + max
i∈[n] |F (0)
i
|

.

This implies that

1
n

X

i∈I2
g(t)
i
−1

n

X

i∈I3
g(t)
i

 ≤O

G(0)(µ1) + G(0)(µ2) + max
i∈[n] |F (0)
i
|

.

Thus, if the initialization scale is sufficiently small, 1

n
P
i∈I2 g(t)
i
and 1

n
P
i∈I3 g(t)
i
are only differed
by a small constant.

E.7
Small Score Movement in Phase 1

Proposition E.22. Conditioned on the success of Lemma D.1 and Lemma D.3, with probability at
least 1 −δ over the randomness of a, for all i ∈[n] and µ, ν ∈X(i), we have


m1
X

j=1
aj∥k(0)(µ)∥2
2w(0)⊤
j
v(0)(ν)


≤O

 

σ2
0mσ1σ0

r

mm1 log m1d

δ

r

log nd

δ

!

,



m1
X

j=1
aj∥q(0)(µ)∥2
2w(0)⊤
j
v(0)(ν)


≤O

 

σ2
0mσ1σ0

r

mm1 log m1d

δ

r

log nd

δ

!

and for all l, l′ ∈[L] with K(0,i)
l
̸= k(0)(ν) and q(0,i)
l
̸= Q(0)(ν), we have


m1
X

j=1
ajν⊤W (0)⊤
K
K(0,i)
l
V (0,i)⊤
l′
w(0)
j


≤eO(σ2
0
√mσ0σ1
√mm1),



m1
X

j=1
ajν⊤W (0)⊤
Q
q(0,i)
l
V (0,i)⊤
l′
w(0)
j


≤eO(σ2
0
√mσ0σ1
√mm1).

Proof. Fix i ∈[n] and µ, ν ∈X(i). Consider the randomness of a. By Lemma D.1, Corollary D.2
and Lemma D.3, aj∥k(0)(µ)∥2
2w(0)⊤
j
v(0)(ν) is a sub-Gaussian random variable with variance proxy
O((σ2
0mσ1σ0
√m
p

log(dm1/δ))2). Then the following inequality holds.


m1
X

j=1
aj∥k(0)(µ)∥2
2w(0)⊤
j
v(0)(ν)


≤O

 

σ2
0mσ1σ0

r

mm1 log m1d

δ

r

log 2

δ

!

.

37


---Page Break---
Finally, take a union bound over i ∈[n], µ, ν ∈{µi}d
i=1. The analysis for the second term is similar.

Next, note that ajν⊤W (0)⊤
K
K(0,i)
l
V (0,i)⊤
l′
w(0)
j
is a sub-Gaussian random variable with variance proxy
eO((σ2
0
√mσ0σ1
√m)2). Thus,


m1
X

j=1
ajν⊤W (0)⊤
K
K(0,i)
l
V (0,i)⊤
l′
w(0)
j


≤eO(σ2
0
√mσ0σ1
√mm1).

Lemma E.23 (Score change). For all t ≤T1, for all ν, µ ∈{µi}d
i=1, we have

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

 ≤
1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO
 1

L +
1
√m



and thus,
ν⊤W (t)⊤
K
W (t)
Q µ −ν⊤W (0)⊤
K
W (0)
Q µ
 ≤t 1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO
 1

L +
1
√m


.

Proof. First of all, by Lemma C.4, we expand the per step gradient descent update as follows:

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

=
1
n√m

X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2

v(t)⊤(ν)w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←ν

|
{z
}
(1)

+
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
aj

L
X

l=1
ν⊤W (t)⊤
K
K(t,i)
l

V (t,i)⊤
l
w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←lI(K(t,i)
l
̸= k(t)(ν))

|
{z
}
(2)

+
1
n√m

X

i:ν,µ∈X(i)
g(t)
i yi

m1
X

j=1
aj∥q(t)(µ)∥2
2p(t,i)
q←µ,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)


|
{z
}
(3)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

I(q(t,i)
l
̸= q(t)(µ))

|
{z
}
(4)

Analysis of (1): By triangle inequality, we have

|(1)| ≤
1
n√m



X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2v(t)⊤(ν)w(t)
j p(t,i)
q←µ,k←ν


|
{z
}
(a)

+
1
n√m



X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)p(t,i)
q←µ,k←ν


|
{z
}
(b)

.

To analyze (a), since (a + ε1)(b + ε2) = ab + aε2 + bε1 + ε1ε2, we have


m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν)p(t,i)
q←µ,k←ν −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)p(0,i)
q←µ,k←ν



38


---Page Break---
≤



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν) −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)


p(t,i)
q←µ,k←ν

+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)


·
p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν
 .

Further, by Lemma D.1, Lemma D.3, Definition E.1 and Definition E.2, we have


m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν) −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)



≤max
ν,j

∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν) −∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)


≤m1

RK eO(σ0σ1
√m) + R eO(σ2
0m) + RKR

.
(9)

Combining with Proposition E.22, this implies


m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν)p(t,i)
q←µ,k←ν −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)p(0,i)
q←µ,k←ν



≤m1 · max
ν,j

∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν) −∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)
 p(t,i)
q←µ,k←ν

+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)


·
p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν


≤m1

RK eO(σ0σ1
√m) + R eO(σ2
0m) + RKR
  2

L + RP


+ eO
 
RP σ2
0mσ1σ0
√mm1

.

Thus,

|(a)| =



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν)p(t,i)
q←µ,k←ν



≤



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
v(t)(ν)p(t,i)
q←µ,k←ν −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)p(0,i)
q←µ,k←ν



+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
v(0)(ν)p(0,i)
q←µ,k←ν



≤m1

RK eO(σ0σ1
√m) + R eO(σ2
0m) + RKR
  2

L + RP



+ eO
 
RP σ2
0mσ1σ0
√mm1

+ eO

σ2
0mσ1σ0
√mm1
1
L


.

On the other hand, to analyze (b), we have


m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)p(t,i)
q←µ,k←ν −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)p(0,i)
q←µ,k←ν



≤



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ) −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)


· p(t,i)
q←µ,k←ν

+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)


·
p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν
 .
(10)

39


---Page Break---
Note that


m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ) −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)



=



L
X

l=1

m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)
l
p(t,i)
l(i,µ),l −

L
X

l=1

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l
p(0,i)
l(i,µ),l



≤

L
X

l=1



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)
l
p(t,i)
l(i,µ),l −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l
p(0,i)
l(i,µ),l



≤

L
X

l=1



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)
l
−

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l


p(t,i)
l(i,µ),l

+

L
X

l=1



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l


·
p(t,i)
l(i,µ),l −p(0,i)
l(i,µ),l


≤max
l∈[L]



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)
l
−

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l



+ LRP max
l∈[L]



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l


,
(11)

which implies

|(b)|

=



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)p(t,i)
q←µ,k←ν



≤



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)p(t,i)
q←µ,k←ν −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)p(0,i)
q←µ,k←ν



+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)p(0,i)
q←µ,k←ν



(i)
≤



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)p(t,i)
l(i,µ) −

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)


· p(t,i)
q←µ,k←ν

+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)


·
p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν


+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)p(0,i)
q←µ,k←ν



(ii)
≤

 

max
l∈[L]



m1
X

j=1
aj∥k(t)(ν)∥2
2w(t)⊤
j
V (t,i)
l
−

m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l



+ LRP max
l∈[L]



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)
l



!

· p(t,i)
q←µ,k←ν

40


---Page Break---
+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)


·
p(t,i)
q←µ,k←ν −p(0,i)
q←µ,k←ν


+



m1
X

j=1
aj∥k(0)(ν)∥2
2w(0)⊤
j
V (0,i)p(0,i)
l(i,µ)p(0,i)
q←µ,k←ν



(iii)
≤

 

m1

RK eO(σ0σ1
√m) + R eO(σ2
0m) + RKR


+ LRP eO(σ2
0mσ0σ1
√mm1)

!

·
 1

L + RP


+ eO

σ2
0mσ0σ1
√mm1

 1

L + RP



where (i) follows from Equation (10), (ii) follows from Equation (11) and (iii) follows from
Equation (9) and Proposition E.22. Combining the upper bound for both (a) and (b), we obtain

|(1)| ≤
1
√m


i : µ, ν ∈X(i)	

n
eO

  

m1

RK eO(σ0σ1
√m) + R eO(σ2
0m) + RKR


+ LRP eO(σ2
0mσ0σ1
√mm1)

!

·
 1

L + RP


+ σ2
0mσ0σ1
√mm1

 1

L + RP

 !

=
1
√m


i : µ, ν ∈X(i)	

n
eO(1/L).

Analysis of (2): (2) can be analyzed similarly as (1) with ∥k(0)(ν)∥2
2 replaced by ν⊤W (t)
K K(t,i)
l
.
Thus, we only need to replace σ2
0m with σ2
0
√m and then take a sum over l. We have

|(2)| ≤
1
√m


i : µ ∈X(i)	

n
eO

  

m1

RK eO(σ0σ1
√m) + R eO(σ2
0
√m) + RKR


+ LRP eO(σ2
0
√mσ0σ1
√mm1)

!

· (1 + LRP ) + σ2
0
√mσ0σ1
√mm1 (1 + LRP )

!

=
1
√m


i : µ ∈X(i)	

n
eO(1/√m).

Analysis of (3): (3) can be analyzed similarly to (1), and we obtain

|(3)| ≤
1
√m


i : µ, ν ∈X(i)	

n
eO(1/L).

Analysis of (4): (4) can be analyzed similarly to (2), and we obtain

|(4)| ≤
1
√m


i : ν ∈X(i)	

n
eO(1/√m).

Finally, combining the bounds on (1) −(4), we have

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

 ≤
1
√m


i : µ, ν ∈X(i)	

n
eO(1/L) +
1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO(1/√m)

≤
1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO(1/L + 1/√m).

Corollary E.24 (Softmax change). For t ≤T1, we have the following:

41


---Page Break---
• if both X(i)
l1 , X(i)
l2 ∈R, then

∂p(t,i)
l1,l2
∂t

 ≤eO

1
L(L2 + n)√m


,

• otherwise,

∂p(t,i)
l1,l2
∂t

 ≤eO

1
L2√m


.

Thus,

• if both X(i)
l1 , X(i)
l2 ∈R, then
p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤eO
 1

L + 1

n


1
Lm2

 1

L +
1
√m


,

• otherwise,
p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤eO

1
Lm2

 1

L +
1
√m


.

Proof. Consider fixed i, l1, l2. By Lemma G.2, we have

∂p(t,i)
l1,l2
∂t

 ≤p(t,i)
l1,l2


∂s(t,i)
l1,l2
∂t

 + p(t,i)
l1,l2

p(t,i)⊤
l1
∂s(t,i)
l1
∂t

 .

By Lemma E.23, we have the following cases:

• if both X(i)
l1 , X(i)
l2 ∈R, then

∂s(t,i)
l1,l2
∂t

 ≤eO
 1

nm

 1

L +
1
√m


;

• otherwise,

∂s(t,i)
l1,l2
∂t

 ≤eO
 1

m

 1

L +
1
√m


.

Next, during Phase 1, we have p(t,i)
l1,l2 ≤1

L +
1
Lm + RP . Therefore,

• if X(i)
l1 ∈R, then
p(t,i)⊤
l1
∂s(t,i)
l1
∂t

 ≤eO
 1

L + 1

n

 1

m

 1

L +
1
√m


;

• otherwise,
p(t,i)⊤
l1
∂s(t,i)
l1
∂t

 ≤eO
 1

m

 1

L +
1
√m


.

Thus,

• if both X(i)
l1 , X(i)
l2 ∈R, then

∂p(t,i)
l1,l2
∂t

 ≤eO
 1

L + 1

n


1
Lm

 1

L +
1
√m


;

42


---Page Break---
• otherwise,

∂p(t,i)
l1,l2
∂t

 ≤eO
 1

Lm

 1

L +
1
√m


.

This implies that

• if both X(i)
l1 , X(i)
l2 ∈R, then
p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤eO
 1

L + 1

n


1
Lm2

 1

L +
1
√m


;

• otherwise,
p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤eO

1
Lm2

 1

L +
1
√m


.

Lemma E.25 (K, Q self-correlation change). For t ≤T1, for µ, ν ∈{µi}d
i=1, we have
µ⊤W (t)⊤
K
W (t)
K ν −µ⊤W (0)⊤
K
W (0)
K ν
 ≤t 1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO
 1
√m


,

µ⊤W (t)⊤
Q
W (t)
Q ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 ≤t 1
√m


i : µ ∈X(i)	 +

i : ν ∈X(i)	

n
eO
 1
√m


.

Proof. By Lemma C.4, we only need to replace eO(σ2
0m) by eO(σ2
0
√m) in the proof of Lemma E.23.
Thus, we omit the proof here.

E.8
All Variables are within Range in Definition of Phase 1

Finally, we prove that at the end of Phase 1, all the variables in Definition E.2 stay in the range.
Theorem E.26. For t ≤CT1/(σ2
1mm1) (where the constant CT1 is from the definition of T1 in
Definition E.2), all of the following hold:

1. maxj∈[m],µ∈{µi}3
i=1

w(t)
j W (t)
V µ −w(0)
j W (0)
V µ
 ≤R, where R = O(1/m1);

2. maxj∈[m],µ/∈{µi}3
i=1

w(t)
j W (t)
V µ −w(0)
j W (0)
V µ
 ≤O(R/n + R/√m);

3. maxµ,ν∈{µi}d
i=1

µ⊤W (t)⊤
Q
W (t)
K ν −µ⊤W (0)⊤
Q
W (0)
K ν
 ≤RS, where RS ≤O(1/m√m);

4. maxµ,ν∈{µi}d
i=1

µ⊤W (t)⊤
Q
W (t)
Q ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 ≤RQ;

5. maxµ,ν∈{µi}d
i=1

µ⊤W (t)⊤
K
W (t)
K ν −µ⊤W (0)⊤
K
W (0)
K ν
 ≤RK.

Thus, T1 = CT1/(σ2
1mm1).

Proof. The first two results are proved by Theorem E.14, Theorem E.15, Lemma E.17. The third
result is proved by Lemma E.23, The fourth and fifth results are proved by Lemma E.25.

Theorem E.27 (End of Phase 1). At t = T1, we have yiF (T1)
i
= Θ(1) for all i ∈[n], and

G(T1)(µ1) = Θ(1),
G(T1)(µ2) = Θ(1),
−G(T1)(µ3) = Θ(1),

∀µ ∈R : G(T1)(µ) = eO(σ0σ1
√mm1).
Further,
m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E
= Θ(σ2
1mm1).

43


---Page Break---
Proof. By Theorem E.26, we have yiF (T1)
i
= O(1) and

G(T1)(µ1) = O(1),
G(T1)(µ2) = O(1),
−G(T1)(µ3) = O(1).

Further, by Theorem E.14 and Theorem E.19, we have yiF (T1)
i
≥Ω(1) for i ∈I1 and

G(T1)(µ1) ≥Ω(1),
G(T1)(µ2) ≥Ω(1).

Finally, by Theorem E.26, we have that T1 = CT1/m. And note that if the constant CT1 in
Definition E.2 is sufficiently large, then by Corollary E.20, we have T ′ ≤T1. Thus, we have
yiF (T1)
i
≥Ω(1) for i ∈I2 ∪I3 ∪I4 and −G(T1)(µ3) ≥Ω(1).

By Lemma D.7 and Theorem E.15, we have

∀µ ∈R : G(T1)(µ) = eO(σ0σ1
√mm1).

Finally, by Lemma D.3, Proposition E.5 and Proposition E.12, we have

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E
= Θ(σ2
1mm1).

Theorem E.28 (Phase 1, formal restatement of Theorem 3.1). With probability at least 1 −δ over
the randomness of weight initialization, there exists a time T1 = eO(1/m) such that

• G(T1)(µ1) ≥Ω(1), G(T1)(µ2) ≥Ω(1), G(T1)(µ3) ≤−Ω(1).

• All the training samples are correctly classified: yiF (T1)
i
= Ω(1) for all i ∈[n].

• For t ∈[0, T1],

D
w(t)
j1 , w(t)
j2
E
−
D
w(0)
j1 , w(0)
j2
E ≤eO
 1

m




D
v(t)(µ), v(t)(ν)
E
−
D
v(0)(µ), v(0)(ν)
E ≤eO
 1

m



ν⊤W (t)⊤
K
W (t)
Q µ −ν⊤W (0)⊤
K
W (0)
Q µ
 ≤eO

1
m3/2

 1

L +
1
√m



µ⊤W (t)⊤
K
W (t)
K ν −µ⊤W (0)⊤
K
W (0)
K ν
 ≤eO
 1

m2



µ⊤W (t)⊤
Q
W (t)
Q ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 ≤eO
 1

m2



• The training loss satisfies bL(T1) = Θ(1).

Proof. The first two results are proved in Theorem E.27. The third result is proved by Lemma E.9,
Proposition E.12, Lemma E.23, and Lemma E.25. The result on the training loss is a direct conse-
quence of Definition E.2.

F
Training Dynamics: Phase 2

The idea of the proof is to first define conditions for Phase 2, which guarantees that the small training
loss can be achieved. Then we will show that those conditions can be satisfied starting from the end
of Phase 1 and up to at least Ω(poly(m)) time, which will serve as the end of Phase 2.

Definition F.1. We define Phase 2 of the training to be t ∈[T1, T2] such that

44


---Page Break---
• The change of K, Q self-correlation is small:

max
µ,ν

µ⊤W (t)⊤
K
W (t)
K ν −µ⊤W (0)⊤
K
W (0)
K ν
 = eO(σ2
0
√m)

max
µ,ν

µ⊤W (t)⊤
Q
W (t)
Q ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 = eO(σ2
0
√m)

• The change of softmax probability satisfies:

max
l1,l2∈[L], i∈[n]

p(t,i)
l1,l2 −p(0,i)
l1,l2

 < O(1/L2)

• The sum of neuron correlation satisfies

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
≤O(σ2
1mm1).

• The gradient of W, WV satisfies

maxµ∈{µi}d
i=1


Pm1
j=1 aj
∂w(t)
j
∂t W (t)
V µ

Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1/L) 1

n

X

i∈[n]
g(t)
i

maxi∈[n]


PL
l1=1
PL
l2=1 G(t)(X(i)
l2 )
∂p(t,i)
l1,l2
∂t


Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1) 1

n

X

i∈[n]
g(t)
i

• The gradients P

i∈I g(t)
i
for I ∈{I1, I2, I3, I4} satisfies

X

i∈I4
gi ≤min

 X

i∈I2
g(t)
i ,
X

i∈I3
g(t)
i

!

≤max

 X

i∈I2
g(t)
i ,
X

i∈I3
g(t)
i

!

≤
X

i∈I1
g(t)
i
≤
X

i∈I2∪I3∪I4
g(t)
i

and

1
2 ≤

P
i∈I2 g(t)
i
P
i∈I3 g(t)
i
≤2.

• yiF (t)
i
> C for all i ∈[n] for some fixed constant C.

• |G(t)(µ)| ≤O(log m) for µ ∈{µi}3
i=1.

• PL
l1=1
P

l2: X(i)
l2 ∈{µk}d
k=4 G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2 ≤O(1).

• T2 ≤O(poly(m)).
Corollary F.2. For t ∈[T1, T2], if i, j ∈I where I ∈{I1, I2, I3, I4}, then

max
i,j∈I
g(t)
i
g(t)
j
≤O(1).

Proof. Take I = I1 and the proof is similar for the remaining cases. For fixed i, j ∈I1, we have

g(t)
i
g(t)
j
=
1 + exp(yjF (t)
j )

1 + exp(yiF (t)
i
)
≤2
exp(yjF (t)
j )

exp(yiF (t)
i
)
,

where the inequality is due to yiF (t)
i
≥C in Definition F.1. Now we consider

exp(yjF (t)
j )

exp(yiF (t)
i
)

45


---Page Break---
=
exp(yj
PL
l1=1
PL
l2=1 G(t)(X(j)
l2 )p(t,j)
q←l1,k←l2)

exp(yi
PL
l1=1
PL
l2=1 G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2)

= exp








L
X

l1=1

X

µ∈{µk}3
k=1

G(t)(µ)p(t,j)
q←l1,k←µ



−




L
X

l1=1

X

µ∈{µk}3
k=1

G(t)(µ)p(t,i)
q←l1,k←µ









|
{z
}
(1)

· exp











L
X

l1=1

X

l2: X(j)
l2 ∈{µk}d
k=4

G(t)(X(j)
l2 )p(t,j)
q←l1,k←l2




−






L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2











|
{z
}
(2)

.

By Definition F.1, it is easy to see that |(2)| ≤O(1). For (1), we have





L
X

l1=1

X

µ∈{µk}3
k=1

G(t)(µ)p(t,j)
q←l1,k←µ



−




L
X

l1=1

X

µ∈{µk}3
k=1

G(t)(µ)p(t,i)
q←l1,k←µ







≤



X

µ∈{µk}3
k=1

G(t)(µ)(1 ± o(1)) −
X

µ∈{µk}3
k=1

G(t)(µ)(1 ± o(1))


≤O(1).

Thus, we have

exp(yjF (t)
j )

exp(yiF (t)
i
)
≤O(1)

for t ∈[T1, T2].

Lemma F.3. Phase 2 in Definition F.1 is well-defined.

Proof. We need to show that the definition is valid at t = T1 with conditions satisfied with strict
inequality. Then since everything changes continuously, there naturally exists a T2 > T1 such that
Definition F.1 is well-defined.

First of all, by Lemma E.25, for µ, ν ∈{µi}d
i=1, we have

max
µ,ν

µ⊤W (T1)⊤
K
W (T1)
K
ν −µ⊤W (0)⊤
K
W (0)
K ν
 = eO(σ2
0
√m),

max
µ,ν

µ⊤W (T1)⊤
Q
W (T1)
Q
ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 = eO(σ2
0
√m).

Next, by Proposition E.12, Proposition E.5 and Lemma D.3, we have

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E
= Θ(σ2
1mm1).

Recall that 1

n
Pn
i=1 g(T1)
i
= Θ(1). On the other hand, by Corollary E.10, Lemma D.3, we have

max
µ∈{µi}d
i=1



m1
X

j=1
aj
∂w(T1)
j
∂t
W (T1)
V
µ


= O(σ2
0mm1)

=⇒
maxµ∈{µi}d
i=1


Pm1
j=1 aj
∂w(T1)
j
∂t
W (T1)
V
µ

Pm1
j1=1
Pm1
j2=1
D
aj1w(T1)
j1
, aj2w(T1)
j2
E
≤O
σ2
0
σ2
1


= eO
 m1

Lm


.

46


---Page Break---
Also, by Corollary E.24, we have

max
i∈[n]



L
X

l1=1

L
X

l2=1
G(T1)(X(i)
l2 )
∂p(T1,i)
l1,l2
∂t

 = eO
 1
√m



=⇒
maxi∈[n]


PL
l1=1
PL
l2=1 G(T1)(X(i)
l2 )
∂p(T1,i)
l1,l2

∂t


Pm1
j1=1
Pm1
j2=1
D
aj1w(T1)
j1
, aj2w(T1)
j2
E
≤eO

1
m3/2


≤o(1)
X

i∈[n]
g(T1)
i
.

Further, by Corollary E.24, we have

max
l1,l2∈[L], i∈[n]

p(T1,i)
l1,l2
−p(0,i)
l1,l2

 < O(1/L2).

Now, by Proposition E.21, if F is small enough (which can be achieved by making the initialization
scale small enough), then

1
2 <

P

i∈I2 g(T1)
i
P

i∈I3 g(T1)
i
< 2.

Lastly, by Theorem E.27, we have yiF (T1)
i
≥Ω(1). And it is straightforward to see that |G(T1)(µ)| ≤
O(log m) for µ ∈{µi}3
i=1.

Finally, we prove PL
l1=1
P

l2: X(i)
l2 ∈{µk}d
k=4 G(T1)(X(i)
l2 )p(T1,i)
q←l1,k←l2 ≤O(1). A simple corollary

from Lemma D.7 and Lemma D.8 is that


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(0)(X(i)
l2 )p(0,i)
q←l1,k←l2


≤O(1).

Next, we have


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 )p(T1,i)
q←l1,k←l2 −

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(0)(X(i)
l2 )p(0,i)
q←l1,k←l2



≤

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

(G(T1)(X(i)
l2 ) −G(0)(X(i)
l2 ))
 p(0,i)
q←l1,k←l2

+

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(0)(X(i)
l2 )
p(T1,i)
q←l1,k←l2 −p(0,i)
q←l1,k←l2



+

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 ) −G(0)(X(i)
l2 )

p(T1,i)
q←l1,k←l2 −p(0,i)
q←l1,k←l2



≤O(1)

which proves that

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 )p(T1,i)
q←l1,k←l2 ≤O(1).

In Phase 2, we will analyze the dynamical system in a different way since all the variables now might
change dramatically from their values at initialization.

47


---Page Break---
Lemma F.4. For t ∈[T1, T2], we have

∂
∂t

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= 2m1

n

n
X

i=1
g(t)
i yiF (t)
i
> 0

and thus,

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= Ω(σ2
1mm1).

Proof. By Lemma C.2, we obtain

∂
∂t

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E

=

m1
X

j1=1

m1
X

j2=1

 
1
n

n
X

i=1
g(t)
i yi

L
X

l1=1
aj1w(t)⊤
j1
V (t,i)p(t,i)
l1
+ 1

n

n
X

i=1
g(t)
i yi

L
X

l2=1
aj2w(t)⊤
j2
V (t,i)p(t,i)
l2

!

= 2m1

n

n
X

i=1
g(t)
i yiF (t)
i
.

Then, note that the final term is positive since by Definition F.1, we have yiF (t)
i
> 0 for all i ∈[n].
Finally, by Theorem E.27, we have for all t ∈[T1, T2],

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
≥Ω(σ2
1mm1).

F.1
Automatic Balancing of Gradients

Lemma F.5 (Same as Lemma 4.4). For t ∈[T1, T2], there exists a small constant C ≪1 such that

P

i∈I2 g(t)
i
−P

i∈I3 g(t)
i


min(P

i∈I2 g(t)
i , P

i∈I3 g(t)
i )
≤C.

Proof. Without loss of generality, assume P

i∈I2 g(t)
i
> P

i∈I3 g(t)
i . Then, we have

P

i∈I2 g(t)
i
−P

i∈I3 g(t)
i


min(P

i∈I2 g(t)
i , P

i∈I3 g(t)
i )
=

P

i∈I2 g(t)
i
−P

i∈I3 g(t)
i
P

i∈I3 g(t)
i
=

P

i∈I2 g(t)
i
P

i∈I3 g(t)
i
−1.

Now by the quotient rule, we have ∂

∂t

 P

i∈I2 g(t)
i
P

i∈I3 g(t)
i


≤0 if and only if

∂
∂t

 X

i∈I2
g(t)
i

!  X

i∈I3
g(t)
i

!

−

 X

i∈I2
g(t)
i

!
∂
∂t

 X

i∈I3
g(t)
i

!

≤0

⇔

 X

i∈I2
g′(yiF (t)
i
)∂yiF (t)
i
∂t

!  X

i∈I3
g(t)
i

!

−

 X

i∈I2
g(t)
i

!  X

i∈I3
g′(yiF (t)
i
)∂yiF (t)
i
∂t

!

≤0

⇔

 X

i∈I2
g(t)
i

!  X

i∈I3
g(t)
i

!  
1
P

i∈I2 g(t)
i

X

i∈I2
g′(yiF (t)
i
)∂yiF (t)
i
∂t
−
1
P

i∈I3 g(t)
i

X

i∈I3
g′(yiF (t)
i
)∂yiF (t)
i
∂t

!

≤0

⇔

 
1
P

i∈I2 g(t)
i

X

i∈I2
g′(yiF (t)
i
)∂yiF (t)
i
∂t
−
1
P

i∈I3 g(t)
i

X

i∈I3
g′(yiF (t)
i
)∂yiF (t)
i
∂t

!

≤0

48


---Page Break---
⇔

 
1
P

i∈I2 g(t)
i

X

i∈I2

−1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

∂yiF (t)
i
∂t

!

−

 
1
P

i∈I3 g(t)
i

X

i∈I3

−1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

∂yiF (t)
i
∂t

!

≤0

⇔

 
1
P

i∈I3 g(t)
i

X

i∈I3

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

∂yiF (t)
i
∂t

!

≤

 
1
P

i∈I2 g(t)
i

X

i∈I2

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

∂yiF (t)
i
∂t

!

.
(12)

Note that
P

i∈I g(t)
i
1 + mini∈I exp(−yiF (t)
i
)
≥
X

i∈I

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))
≥
P
i∈I g(t)
i
1 + maxi∈I exp(−yiF (t)
i
)
.

By Definition F.1, we have that for i′ ∈I2,

∂yi′F (t)
i′
∂t

= yi′

L
X

l1=1

m1
X

j=1

L
X

l2=1



∂ajw(t)
j W (t)
V X(i′)
l2
∂t
p(t,i′)
l1,l2 + ajw(t)
j W (t)
V X(i′)
l2
∂p(t,i′)
l1,l2
∂t





= yi′

L
X

l2=1

∂G(t)(X(i′)
l2 )
∂t
(1 + o(1)) +

L
X

l1=1

L
X

l2=1
yi′G(t)(X(i′)
l2 )
∂p(t,i′)
l1,l2
∂t

= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



.

Similarly, for i′ ∈I3, we have

∂yi′F (t)
i′
∂t

= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I3
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



.

Now, we analyze Equation (12). Note that for t ∈[T1, T2], if P

i∈I2 g(t)
i
is sufficiently larger

than P

i∈I3 g(t)
i
(i.e., P

i∈I2 g(t)
i / P

i∈I3 g(t)
i
is bigger than some threshold), then ∂G(t)(µ1)

∂t
<

∂G(t)(µ2)

∂t
and mini∈I2
∂yiF (t)
i
∂t
> maxi∈I3
∂yiF (t)
i
∂t
.
Thus, the ratio will decrease.
Similarly,
if P

i∈I2 g(t)
i / P

i∈I3 g(t)
i
is too small, the ratio will increase. Next, we compute a bound on
P

i∈I2 g(t)
i / P

i∈I3 g(t)
i
so that
∂
∂t

 P

i∈I2 g(t)
i
P

i∈I3 g(t)
i


= 0. Substituting ∂yiF (t)
i
∂t
in Equation (12), we

have
 
1
P

i∈I3 g(t)
i

X

i∈I3

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

!

·



−

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E


X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
± o(1)
X

i∈[n]
g(t)
i









=

 
1
P

i∈I2 g(t)
i

X

i∈I2

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

!

49


---Page Break---
·



−

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E


X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I3
g(t)
i
± o(1)
X

i∈[n]
g(t)
i









⇒1 + mini∈I2 exp(−yiF (t)
i
)

1 + maxi∈I3 exp(−yiF (t)
i
)



−
X

i∈I1
g(t)
i
+
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I1
g(t)
i
+
X

i∈I2
g(t)
i
± o(1)
X

i∈[n]
g(t)
i





=



−
X

i∈I1
g(t)
i
+
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I1
g(t)
i
+
X

i∈I3
g(t)
i
± o(1)
X

i∈[n]
g(t)
i





⇒

P

i∈I2 g(t)
i
P

i∈I3 g(t)
i
= 1 +

 
1
P

i∈I3 g(t)
i

P
i∈I3
1
(1+exp(yiF (t)
i
))(1+exp(−yiF (t)
i
))

1
P

i∈I2 g(t)
i

P

i∈I2
1
(1+exp(yiF (t)
i
))(1+exp(−yiF (t)
i
))

·
−P

i∈I1 g(t)
i
+ P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
+ P

i∈I2 g(t)
i
± o(1) P

i∈[n] g(t)
i
P

i∈I3 g(t)
i

!

.

Since

1
P

i∈I g(t)
i

X

i∈I

1

(1 + exp(yiF (t)
i
))(1 + exp(−yiF (t)
i
))

∈

"
1

1 + maxi∈I exp(−yiF (t)
i
)
,
1

1 + mini∈I exp(−yiF (t)
i
)

#

,

we have

1
P

i∈I3 g(t)
i

P

i∈I3
1
(1+exp(yiF (t)
i
))(1+exp(−yiF (t)
i
))

1
P

i∈I2 g(t)
i

P
i∈I2
1
(1+exp(yiF (t)
i
))(1+exp(−yiF (t)
i
))

∈

"
1 + mini∈I2 exp(−yiF (t)
i
)

1 + maxi∈I3 exp(−yiF (t)
i
)
, 1 + maxi∈I2 exp(−yiF (t)
i
)

1 + mini∈I3 exp(−yiF (t)
i
)

#

.

By Definition F.1, we have


−P

i∈I1 g(t)
i
+ P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
+ P

i∈I2 g(t)
i
± o(1) P

i∈[n] g(t)
i
P

i∈I3 g(t)
i

 ≤6.

Thus,
P

i∈I2 g(t)
i
P

i∈I3 g(t)
i
∈1 ± 6 ·

 "
1 + mini∈I2 exp(−yiF (t)
i
)

1 + maxi∈I3 exp(−yiF (t)
i
)
, 1 + maxi∈I2 exp(−yiF (t)
i
)

1 + mini∈I3 exp(−yiF (t)
i
)

#

−1

!

.

Lemma F.6 (Complete version of Lemma 4.5). For t ∈[T1, T2], we have
P

i∈[n] g(t)
i
P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
= O(1),

P

i∈[n] g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= O(1)

P

i∈[n] g(t)
i
P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
= O(1),

∂G(t)(µ1)

∂t
∂G(t)(µ2)

∂t
= Θ(1).

Further, there exists a constant C such that

(1 + C) max
∂G(t)(µ1)

∂t
, ∂G(t)(µ2)

∂t


≤−∂G(t)(µ3)

∂t
≤(1 −C)
∂G(t)(µ1)

∂t
+ ∂G(t)(µ2)

∂t


.

50


---Page Break---
Proof. Without loss of generality, assume ∂G(t)(µ1)

∂t
> ∂G(t)(µ2)

∂t
. First of all, by Definition F.1, we
have

∂G(t)(µ1)

∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
± o(1)
X

i∈[n]
g(t)
i



,

∂G(t)(µ2)

∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I3
g(t)
i
± o(1)
X

i∈[n]
g(t)
i



,

∂G(t)(µ3)

∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
± o(1)
X

i∈[n]
g(t)
i



.

This implies that

∂G(t)(µ1)

∂t
∂G(t)(µ2)

∂t
=

P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
± o(1) P

i∈[n] g(t)
i
P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
± o(1) P

i∈[n] g(t)
i
,

−∂G(t)(µ3)

∂t
∂G(t)(µ1)

∂t
=
−P
i∈I1 g(t)
i
+ P
i∈I2∪I3∪I4 g(t)
i
± o(1) P
i∈[n] g(t)
i
P
i∈I1 g(t)
i
−P

i∈I2 g(t)
i
± o(1) P

i∈[n] g(t)
i
.

We next analyze ∂

∂t
−∂G(t)(µ3)

∂t
∂G(t)(µ1)

∂t
. We first define the ratio R(t) =

P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
. Since

the dependence of R on t is clear and for the ease of notation, we omit this dependence below.
Rearranging this definition, we obtain

(1 + R)

 X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i

!

=
X

i∈I3∪I4
g(t)
i .
(13)

We consider the range of R ∈[1, 3]. By Definition F.1, we have
P

i∈[n] g(t)
i
P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
= O(1),

P

i∈[n] g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= O(1),
(14)

where the second equation follows from Lemma F.5. This implies that

∂G(t)(µ1)

∂t
∂G(t)(µ2)

∂t
= Θ(1),

P

i∈[n] g(t)
i
P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
= O(1),

which proves the first result and

−∂G(t)(µ3)

∂t
∂G(t)(µ1)

∂t
= −P

i∈I1 g(t)
i
+ P
i∈I2∪I3∪I4 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
+ o(1).

Thus, to analyze ∂

∂t
−∂G(t)(µ3)

∂t
∂G(t)(µ1)

∂t
, we can instead analyze

∂
∂t

P

i∈I2∪I3∪I4 g(t)
i
−P

i∈I1 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
≥0

⇔∂

∂t

 
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I1
g(t)
i

!  X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i

!

−

 
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I1
g(t)
i

!
∂
∂t

 X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i

!

≥0

51


---Page Break---
⇔
X

i∈I2∪I3∪I4

∂g(t)
i
∂t
+ R
X

i∈I2

∂g(t)
i
∂t
≥(1 + R)
X

i∈I1

∂g(t)
i
∂t .
(15)

Recall that by Definition F.1, we have for i1 ∈I1,

∂yi1F (t)
i1
∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



3
X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I2∪I3
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

for i2 ∈I2,

∂yi2F (t)
i2
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

for i3 ∈I3,

∂yi3F (t)
i3
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I3
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

and for i4 ∈I4,

∂yi4F (t)
i4
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



.

The above implies that for i1 ∈I1,

∂yi1F (t)
i1
∂t
P
i∈I1 g(t)
i
−P
i∈I2 g(t)
i
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n

 

1 +

P
i∈I1 g(t)
i
−P

i∈I3 g(t)
i
P
i∈I1 g(t)
i
−P
i∈I2 g(t)
i
−R + o(1)

!

;

for i2 ∈I2,

∂yi2F (t)
i2
∂t
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n (1 −R + o(1)) ;

for i3 ∈I3,

∂yi3F (t)
i3
∂t
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n

 P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
−R + o(1)

!

;

and for i4 ∈I4,

∂yi1F (t)
i1
∂t
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n (−R + o(1)) .

Substituting
the
above
into
Equation
(15)
and
divide
both
sides
by

1
n
Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
, we have

(1 + R)
X

i2∈I2
g′(F (t)
i2 )(R −1 + o(1)) +
X

i3∈I3
g′(F (t)
i3 )

 

R −

P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
+ o(1)

!

+
X

i4∈I4
g′(F (t)
i4 )(R + o(1))

≥(1 + R)
X

i1∈I1
g′(F (t)
i1 )

 

1 +

P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
−R + o(1)

!

.
(16)

52


---Page Break---
By Lemma F.5, we obtain


P

i∈I1 g(t)
i
−P

i∈I3 g(t)
i
P

i∈I1 g(t)
i
−P

i∈I2 g(t)
i
−1

 ≤ε,

where we use ε to denote the small deviation. Note that Equation (16) is a quadratic inequality in R
and can be rearranged as aR2 + bR + c ≥0, where

a =
X

i1∈I1
g′(F (t)
i1 ) +
X

i2∈I2
g′(F (t)
i2 ),

b =
X

i3∈I3
g′(F (t)
i3 ) +
X

i4∈I4
g′(F (t)
i4 ) −
X

i1∈I1
g′(F (t)
i1 ),

c = (1 + o(1))

 

−
X

i2∈I2
g′(F (t)
i2 ) −(1 ± ε)
X

i3∈I3
g′(F (t)
i3 ) −(2 ± ε)
X

i1∈I1
g′(F (t)
i1 )

!

.

Now we analyze the equality condition in Equation (16) where we calculate the root of the equation
aR2 + bR + c = 0. We have

b
a =

P

i3∈I3 g′(F (t)
i3 ) + P

i4∈I4 g′(F (t)
i4 ) −P

i1∈I1 g′(F (t)
i1 )
P

i1∈I1 g′(F (t)
i1 ) + P

i2∈I2 g′(F (t)
i2 )
,

c
a =
(1 + o(1))

−P

i2∈I2 g′(F (t)
i2 ) −(1 ± ε) P

i3∈I3 g′(F (t)
i3 ) −(2 ± ε) P

i1∈I1 g′(F (t)
i1 )


P

i1∈I1 g′(F (t)
i1 ) + P

i2∈I2 g′(F (t)
i2 )
.

Note that
X

i∈I
g(t)
i
min
i∈I
1

1 + exp(yiF (t)
i
)
≤−
X

i∈I
g′(F (t)
i
) ≤
X

i∈I
g(t)
i
max
i∈I
1

1 + exp(yiF (t)
i
)
.

By Equation (14), we have −b

2a ≥−1/4 and

b
a

 ≤(1 + O(max
i∈[n] exp(−yiF (t)
i
))) max

 

P

i1∈I1 g(F (t)
i1 ) −P

i3∈I3 g(F (t)
i3 )
P
i1∈I1 g(F (t)
i1 ) + P

i2∈I2 g(F (t)
i2 )

 ,



P

i4∈I4 g(F (t)
i4 )
P
i1∈I1 g(F (t)
i1 ) + P

i2∈I2 g(F (t)
i2 )



!

≤C < 1,
and
 c

a

 = (1 ± O(ε) ± O(max
i∈[n] exp(−yiF (t)
i
))) · 2.

Recall that we are considering the case of R ∈[1, 3].
Thus, we only need to consider the

root that is positive and we can calculate the root R⋆= −b

2a +
q

b2
4a2 −c

a ∈(1, 2) if ε and

maxi∈[n] exp(−yiF (t)
i
) are both suffiently small.

Next, since g′(F (t)
i
) < 0, the root R⋆is contractive (i.e., if R(t) > R⋆then R(t) is decreasing and if
R(t) < R⋆then R(t) is increasing).

Finally, the result at the end of Phase 1 implies that R(T1) ∈[1, 3], which completes the proof.

Corollary F.7. For t ∈[T1, T2], we have

G(t)(µ1)
G(t)(µ2), G(t)(µ1)

−G(t)(µ3), G(t)(µ2)

−G(t)(µ3) = Θ(1).

Proof. We first prove G(t)(µ1)

G(t)(µ2) = Θ(1). Following from Lemma F.6, we have

∂G(t)(µ1)

∂t
∂G(t)(µ2)

∂t
= Θ(1). By

Theorem E.27, we have G(T1)(µ1)

G(T1)(µ2) = Θ(1). Thus, for t ∈[T1, T2], we obtain

G(t)(µ1)
G(t)(µ2) =
G(T1)(µ1) +
R t
T1
∂G(τ)(µ1)

∂τ
dτ

G(T1)(µ2) +
R t
T1
∂G(τ)(µ2)

∂τ
dτ
= Θ(1).

53


---Page Break---
Note that Lemma F.6 and Definition F.1 imply that

∂G(t)(µ1)

∂t
−∂G(t)(µ3)

∂t
= Θ(1). Since
G(T1)(µ1)
−G(T1)(µ3) = Θ(1),

similarly, we have
G(t)(µ1)
−G(t)(µ3) = Θ(1). Finally, G(t)(µ1)

G(t)(µ2) = Θ(1) and
G(t)(µ1)
−G(t)(µ3) = Θ(1) imply that

G(t)(µ2)
−G(t)(µ3) = Θ(1).

F.2
How Fast the Loss Decreases

Theorem F.8. For t ∈[T1, T2], we have

bL(t) =
1

Θ(σ2
1mm1)(t −T1) + (1/bL(T1))
.

Proof. First of all, the gradient flow update for the empirical loss is given by
∂bL

∂t
=
Pn
i=1 ℓ′(yiF (t)
i
) ∂yiF (t)
i
∂t
. By Definition F.1, we have for i1 ∈I1,

∂yi1F (t)
i1
∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



3
X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
−
X

i∈I2∪I3
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

for i2 ∈I2,

∂yi2F (t)
i2
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I2
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

for i3 ∈I3,

∂yi3F (t)
i3
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+
X

i∈I1
g(t)
i
−
X

i∈I3
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



;

and for i4 ∈I4,

∂yi4F (t)
i4
∂t
= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E 1

n



X

i∈I1
g(t)
i
−
X

i∈I2∪I3∪I4
g(t)
i
+ o(1)
X

i∈[n]
g(t)
i



.

By Lemma F.6, we have

∂yiF (t)
i
∂t
=

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
Θ



1

n

X

i∈[n]
g(t)
i





for all i ∈[n]. Therefore,

∂bL

∂t = 1

n

n
X

i=1
ℓ′(yiF (t)
i
)∂yiF (t)
i
∂t

= 1

n

n
X

i=1
−g(t)
i

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
Θ



1

n

X

i′∈[n]
g(t)
i′





= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
Θ








1

n

X

i∈[n]
g(t)
i





2




= −

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
Θ

bL2
,

54


---Page Break---
where the last equality follows from the property of binary cross-entropy loss that ℓ(x) = Θ(−ℓ′(x))
for x > 0. By Definition F.1 and Lemma F.4, we have for all t ∈[T1, T2],

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= Θ(σ2
1mm1).

Thus, we have

∂bL

∂t = −Θ(σ2
1mm1)bL2.

Now, consider the differential equation dL

dt = −C1L2. Note that this is a separable differential
equation in t and we can solve it by

1
L2
dL

dt + C1 = 0
⇒
d
dt
 
C1t −L−1 + C2

= 0
⇒
L(t) =
1
C1t + C2
.

This implies for t ∈[T1, T2], we have

bL(t) =
1

Θ(σ2
1mm1)(t −T1) + (1/bL(T1))

Corollary F.9. The following bound holds:


m1
X

j1=1

m1
X

j2=1

D
aj1w(T2)
j1
, aj2w(T2)
j2
E
−

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E

≤eO
m1

m


.

Proof. This is a direct consequence of Lemma F.4 and Theorem F.8.

F.3
Growth of Neuron Correlation

Lemma F.10. For t ∈[T1, T2], we have

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
−

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E

= O
m1 log m log t

σ2
1mm1


= O
m1 log2 m

σ2
1mm1



and thus,

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= Θ(σ2
1mm1).

Proof. By Lemma F.4, we have

∂
∂t

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
= 2m1

n

n
X

i=1
g(t)
i yiF (t)
i
.

By Theorem F.8, we have 1

n
Pn
i=1 g(t)
i
= O(bL(t)) = O

1
(σ2
1mm1)(t−T1)+1/bL(T1)


. Further, by

Definition F.1, we have |F (t)
i
| ≤O(log m). Thus, for t ∈[T1, T2], we obtain

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
−

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E

= 2m1

Z t

T1

1
n

n
X

i=1
g(τ)
i
yiF (τ)
i
dτ

55


---Page Break---
≤O(m1 log m)
Z t

T1
O

 
1

(σ2
1mm1)(τ −T1) + 1/bL(T1)

!

dτ

= O
m1 log m log t

σ2
1mm1


= O
m1 log2 m

σ2
1mm1



where the last line follows because T2 ≤O(poly(m)) in Definition F.1. Finally, by Theorem E.27
we have

m1
X

j1=1

m1
X

j2=1

D
aj1w(T1)
j1
, aj2w(T1)
j2
E
= Θ(σ2
1mm1).

F.4
Growth of Correlation of Value-Transformed Data

We now analyze the correlation term with value-transformed data.

Lemma F.11 (Growth of correlation of value-transformed data). For µ, ν ∈{µi}d
i=1, we have

∂
∂tµ⊤W (t)⊤
V
W (t)
V ν = G(t)(ν) 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ + G(t)(µ) 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←ν.

Thus, for t ∈[T1, T2], we have

max
µ,ν

µ⊤W (t)⊤
V
W (t)
V ν −µ⊤W (T1)⊤
V
W (T1)
V
ν
 ≤O
log m log t

σ2
1mm1


.

Proof. By the gradient flow update, we have

∂µW (t)⊤
V
W (t)
V ν
∂t

= 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajν⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←µ + 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
V
w(t)
j p(t,i)
q←l,k←ν

= G(t)(ν) 1

n

X

i:µ∈X(i)
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←µ + G(t)(µ) 1

n

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1
p(t,i)
q←l,k←ν.

Thus, we obtain
µW (τ)⊤
V
W (τ)
V ν −µW (T1)⊤
V
W (T1)
V
ν


≤
Z τ

T1
|G(t)(ν)|



1
n

X

i:µ∈X(i)
g(t)
i yi





L
X

l=1
p(t,i)
q←l,k←µ

 + |G(t)(µ)|



1
n

X

i:ν∈X(i)
g(t)
i yi





L
X

l=1
p(t,i)
q←l,k←ν

 dt.

By Definition F.1, for t ∈[T1, T2], we have |G(t)(µ)| ≤O(log m) and PL
l=1 p(t,i)
q←l,k←µ ≤O(1).
Further, by the property of the cross-entropy loss, we have

1
n

X

i:µ∈X(i)
g(t)
i yi = O(bL(t)).

Therefore, by Theorem F.8, we obtain
µW (τ)⊤
V
W (τ)
V ν −µW (T1)⊤
V
W (T1)
V
ν
 ≤O(log m)
Z τ

T1
bL(t)dt ≤O(log m) · O
 log τ

σ2
1mm1


.

56


---Page Break---
Corollary F.12. For t ∈[T1, T2], we have

maxµ∈{µi}d
i=1


Pm1
j=1 aj
∂w(t)
j
∂t W (t)
V µ

Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1/L) 1

n

X

i∈[n]
g(t)
i .

Proof. By Lemma E.9 and and Lemma F.11, for all µ ̸= ν ∈{µi}d
i=1, we have |µ⊤W (t)⊤
V
W (t)
V ν| ≤
eO(σ2
0
√m + 1/m) and ∥W (t)
V ν∥2
2 = eO(σ2
0m + 1/m). Thus, by Lemma F.11, we have

maxµ,ν
µ⊤W (t)⊤
V
W (t)
V ν

Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E ≤eO
σ2
0
√m + 1/m

σ2
1mm1


,

maxν ∥W (t)
V ν∥2
2
Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E ≤eO
σ2
0m + 1/m

σ2
1mm1


.

Recall that

m1
X

j=1
aj
∂w(t)
j
∂t W (t)
V µ = m1

n

n
X

i1=1
g(t)
i1 yi1

L
X

l1=1
p(t,i1)⊤
l1
V (t,i1)⊤W (t)
V µ.

This implies that

maxµ∈{µi}d
i=1


Pm1
j=1 aj
∂w(t)
j
∂t W (t)
V µ

Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤eO
σ2
0
√mL + L/m + σ2
0m + 1/m
σ2
1m


· 1
n

n
X

i=1
g(t)
i .

Corollary F.13 (Complete version of Corollary 4.6). For t ∈[T1, T2], we have

∂
∂tµ⊤
2 W (t)⊤
V
W (t)
V µ1 > 0,
∂
∂tµ⊤
1 W (t)⊤
V
W (t)
V µ1 > 0,
∂
∂tµ⊤
2 W (t)⊤
V
W (t)
V µ2 > 0

∂
∂tµ⊤
1 W (t)⊤
V
W (t)
V µ3 < 0,
∂
∂tµ⊤
2 W (t)⊤
V
W (t)
V µ3 < 0.

Proof. This is a direct consequence of Lemma F.11, Lemma F.6 and Definition F.1.

F.5
Change of Random-Token Sub-Network

Lemma F.14. For t ∈[T1, T2] and µ ∈{µi}d
i=4, we have

∂G(t)(µ)

∂t

 ≤O
 1

n
bL(t)σ2
1mm1


+ eO(bL(t)m1(L(σ2
0
√m + 1/m) + σ2
0m)).

Thus,
G(t)(µ) −G(T1)(µ)
 ≤eO
 1

n + m1(L(σ2
0
√m + 1/m) + σ2
0m)
σ2
1mm1


.

Proof. By Lemma C.1 and Definition F.1, we have


∂G(t)(µ)

∂t

 =



m1
X

j=1
ajw(t)
j
∂W (t)
V µ
∂t
+ aj
∂w(t)
j
∂t W (t)
V µ



≤



1
n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ



57


---Page Break---
+


1
n

n
X

i1=1
g(t)
i1 yi1m1

L
X

l1=1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E .

By Definition F.1 and Corollary F.2,


1
n

X

i2: µ∈X(i2)
g(t)
i2 yi2

L
X

l2=1

m1
X

j1=1

m1
X

j2=1
aj1aj2
D
w(t)
j1 , w(t)
j2
E
p(t,i2)
q←l2,k←µ


≤O
 1

n
bL(t)σ2
1mm1


.

By Lemma E.9 and and Lemma F.11, we have |µ⊤W (t)⊤
V
W (t)
V ν| ≤eO(σ2
0
√m + 1/m) for µ ̸= ν
and ∥W (t)
V µ∥2
2 = O(σ2
0m). Thus,

1
n

n
X

i1=1
g(t)
i1 yi1m1

L
X

l1=1

L
X

l2=1

D
p(t,i1)
l1,l2 v(t,i1)
l2
, v(t)(µ)
E ≤eO(bL(t)m1(L(σ2
0
√m + 1/m) + σ2
0m)).

Thus, by Theorem F.8, for t ∈[T1, T2], we have

G(t)(µ) −G(T1)(µ)
 =


Z t

T1

∂G(τ)(µ)

∂τ
dτ
 ≤

O
σ2
1mm1

n


+ eO(m1L(σ2
0
√m + 1/m))
 Z t

T1
bL(τ) dτ

≤eO
 1

n + m1(L(σ2
0
√m + 1/m) + σ2
0m)
σ2
1mm1


.

Corollary F.15. For t ∈[T1, T2], we have


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2


≤O(1).

Proof. By Lemma F.3, we have


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 )p(T1,i)
q←l1,k←l2


≤O(1)

By Lemma F.14, for t ∈[T1, T2], G(t)(µ) = O(1) for µ ∈{µi}d
i=4. On the other hand, by the
triangle inequality, we have


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 )p(T1,i)
q←l1,k←l2 −

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2



≤

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

(G(T1)(X(i)
l2 ) −G(t)(X(i)
l2 ))
 p(t,i)
q←l1,k←l2

+

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )
p(T1,i)
q←l1,k←l2 −p(t,i)
q←l1,k←l2



+

L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(T1)(X(i)
l2 ) −G(t)(X(i)
l2 )

p(T1,i)
q←l1,k←l2 −p(t,i)
q←l1,k←l2



≤O(1),

58


---Page Break---
where the last inequality applies Definition F.1. This implies that, for t ∈[T1, T2], by Lemma F.14,
we have


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2


≤O(1).

F.6
Change of Score and Softmax Probability

Lemma F.16 (Change of score, complete version of Lemma 4.7). For t ∈[T1, T2], the attention
scores are changing in the following way:

• For µ, ν ∈{µ1, µ2}, µ ̸= ν, the query-key-correlation score between the two target signals
increases, while the query-key-correlation score between one target signal and the common
token decreases, i.e.,
∂
∂tν⊤W (t)⊤
K
W (t)
Q µ =
1
√m
eΘ(bL(t)σ2
0m) 1

L,

∂
∂tµ⊤
3 W (t)⊤
K
W (t)
Q µ = −1
√m
eΘ(bL(t)σ2
0m) 1

L.

• The change of score satisfies:

max
µ,ν∈{µi}3
i=1

ν⊤W (t)
K W (t)
Q µ −ν⊤W (T1)
K
W (T1)
Q
µ
 = Θ

σ2
0m
√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


.

• For all µ ∈{µ1, µ2, µ3}, γ ∈{µi}d
i=4, the query-key-correlation score changes as follows:

∂
∂tµ⊤W (t)⊤
K
W (t)
Q γ
 =
1
n√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

,

and
µ⊤W (t)⊤
K
W (t)
Q γ −µ⊤W (T1)⊤
K
W (T1)
Q
γ
 ≤eO

σ2
0m
n√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


,

γ⊤W (t)⊤
K
W (t)
Q µ −γ⊤W (T1)⊤
K
W (T1)
Q
µ
 ≤eO

σ2
0m
n√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


.

Proof. By Lemma C.4, we have

∂ν⊤W (t)⊤
K
W (t)
Q µ

∂t

=
1
n√m

X

i:µ,ν∈X(i)
g(t)
i yi

m1
X

j=1
aj∥k(t)(ν)∥2
2

v(t)⊤(ν)w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←ν

+
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
aj

L
X

l=1
ν⊤W (t)⊤
K
K(t,i)
l

V (t,i)⊤
l
w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
q←µ,k←lI(K(t,i)
l
̸= k(t)(ν))

+
1
n√m

X

i:ν,µ∈X(i)
g(t)
i yi

m1
X

j=1
aj∥q(t)(µ)∥2
2p(t,i)
q←µ,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)


+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

L
X

l=1

m1
X

j=1
ajµ⊤W (t)⊤
Q
q(t,i)
l
p(t,i)
q←l,k←ν

w(t)⊤
j
v(t,i)(ν) −w(t)⊤
j
V (t,i)p(t,i)
l

I(q(t,i)
l
̸= q(t)(µ)).

Now, we take µ = µ1, ν = µ2. By Theorem E.27, we have G(T1)(µ2) ≥Ω(1). A consequence of

Definition F.1 and Lemma F.6 is that ∂G(t)(µ2)

∂t
> 0 for t ∈[T1, T2]. Thus, we have G(t)(µ2) ≥Ω(1).

59


---Page Break---
Further by Theorem E.27, we have G(T1)(µ) = eO(σ0σ1√mm1) for µ ∈R and then by Lemma F.14,

we have G(t)(µ) = eO(σ0σ1√mm1) + eO

1
n + m1(L(σ2
0
√m+1/m)+σ2
0m)
σ2
1mm1


for t ∈[T1, T2]. Also,

Definition F.1 implies that G(t)(µ2) −Pm1
j=1 w(t)
j V (t,i)p(t,i)
l(i,µ) ≥Ω(1). Now, this yields

∂µ⊤
2 W (t)⊤
K
W (t)
Q µ1
∂t
=
1
√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

.

On the other hand, by the analysis similar to the above, we obtain

∂µ⊤
3 W (t)⊤
K
W (t)
Q µ1
∂t
= −1
√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

.

Next, to prove the maximum change of the score, we have

max
µ,ν


∂ν⊤W (t)
K W (t)
Q µ

∂t

 ≤
1
√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

.

By Theorem F.8, we have
ν⊤W (t)⊤
K
W (t)
Q µ −ν⊤W (T1)⊤
K
W (T1)
Q
µ
 ≤
Z t

T1

1
√m
eΘ(bL(τ)σ2
0m) 1

L + eO
 1
√m
bL(τ)σ2
0
√m

dτ

≤eO

σ2
0m
√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


.

Finally, for γ ∈{µi}d
i=4, µ ∈{µi}d
i=1, we have

∂
∂tµ⊤W (t)⊤
K
W (t)
Q γ
 =
1
n√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

,

which implies that
µ⊤W (t)⊤
K
W (t)
Q γ −µ⊤W (T1)⊤
K
W (T1)
Q
γ
 ≤eO

σ2
0m
n√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


.

Corollary F.17 (Change of softmax). For t ∈[T1, T2], the softmax probability is changing in the
following way:

• For µ, ν ∈{µ1, µ2}, µ ̸= ν, the softmax probability between the two target signals
increases, whereas the softmax probability between one target signal and the common token
decreases, i.e.,

∂
∂tp(t,i)
q←µ,k←ν =
1
√m
eΘ(bL(t)σ2
0m) 1

L2 ,

∂
∂tp(t,i)
q←µ,k←µ3 = −1
√m
eΘ(bL(t)σ2
0m) 1

L2 .

• For all µ ∈{µ1, µ2}, γ ∈{µi}d
i=4, the softmax probability between one target signal and
a random token changes as follows:

∂
∂tp(t,i)
q←µ,k←γ

 ≤
1
n√m
eΘ(bL(t)σ2
0m) 1

L2 + eO

1
L√m
bL(t)σ2
0
√m

.

Furthermore, we have

max
i∈[n],l1,l2∈[L]

p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤O(1/L2).

60


---Page Break---
Proof. Take µ = µ1, ν = µ2. First of all, by Definition F.1 and Lemma F.16, we have
p(t,i)⊤
q←µ1
∂X(i)⊤W (t)⊤
K
W (t)
Q µ1
∂t

 ≤
1
√mO(bL(t)σ2
0m) 1

L2 +
1
n√m
eΘ(bL(t)σ2
0m) 1

L + eO
 1
√m
bL(t)σ2
0
√m

.

Thus, by Lemma G.1 and Lemma F.16, for i ∈I1, we obtain

∂
∂tp(t,i)
q←µ1,k←µ2 = 1

m
eΘ(bL(t)σ2
0m) 1

L2 ,

∂
∂tp(t,i)
q←µ1,k←µ3 = −1

m
eΘ(bL(t)σ2
0m) 1

L2 .

On the other hand, for γ ∈{µi}d
i=4, we have

∂
∂tp(t,i)
q←µ,k←γ

 ≤
1
nm
eΘ(bL(t)σ2
0m) 1

L2 + eO
 1

Lm
bL(t)σ2
0
√m

.

Finally, by Lemma F.16, we have

max
µ,ν

ν⊤W (t)
K W (t)
Q µ −ν⊤W (T1)
K
W (T1)
Q
µ
 ≤eO

σ2
0m
√mLσ2
1mm1
+
σ2
0
√m
√mσ2
1mm1


.

Thus, by Lemma G.2, we obtain

max
i,l1,l2

p(t,i)
l1,l2 −p(T1,i)
l1,l2



≤eO

 
σ2
0m
mL2σ2
1mm1
+
σ2
0
√m
mσ2
1mm1L + L

σ2
0m
mLσ2
1mm1
+
σ2
0
√m
mσ2
1mm1

2!

= O(1/L2).

Corollary F.18. For t ∈[T1, T2], we have

maxi∈[n]


PL
l1=1
PL
l2=1 G(t)(X(i)
l2 )
∂p(t,i)
l1,l2
∂t


Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1) 1

n

X

i∈[n]
g(t)
i .

Proof. This is a direct consequence of Definition F.1, Lemma F.10 and Corollary F.17.

F.7
Change of Self-Correlation of Key/Query-Transformed data

Lemma F.19 (Change of self-correlation). For µ, ν ∈{µi}d
i=1, we have
ν⊤W (t)⊤
Q
W (t)
Q µ −ν⊤W (T1)⊤
Q
W (T1)
Q
µ
 ≤eO

σ2
0
√m
√mσ2
1mm1


,

ν⊤W (t)⊤
K
W (t)
K µ −ν⊤W (T1)⊤
K
W (T1)
K
µ
 ≤eO

σ2
0
√m
√mσ2
1mm1


.

Proof. We prove the result for ν⊤W (t)⊤
Q
W (t)
Q µ and the proof for ν⊤W (t)⊤
K
W (t)
K µ is similar. By
Lemma C.4, we have

∂ν⊤W (t)⊤
Q
W (t)
Q µ

∂t

=
1
n√m

X

i:µ∈X(i)
g(t)
i yi

m1
X

j=1
ajν⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,µ)

p(t,i)
l(i,µ)

+
1
n√m

X

i:ν∈X(i)
g(t)
i yi

m1
X

j=1
ajµ⊤W (t)⊤
Q
K(t,i)diag

V (t,i)⊤w(t)
j
−w(t)⊤
j
V (t,i)p(t,i)
l(i,ν)

p(t,i)
l(i,ν).

61


---Page Break---
A simple result from Lemma F.16 is that |ν⊤W (t)⊤
K
W (t)
Q µ| ≤eO(σ2
0
√m) for t ∈[T1, T2] and
µ, ν ∈{µi}d
i=1. Therefore, by Definition F.1,

∂ν⊤W (t)⊤
Q
W (t)
Q µ

∂t

 ≤eO
 1
√m
bL(t)σ2
0
√m log m

,

which implies
ν⊤W (t)⊤
Q
W (t)
Q µ −ν⊤W (T1)⊤
Q
W (T1)
Q
µ
 ≤
Z t

T1
eO
 1
√m
bL(t)σ2
0
√m log m

dτ ≤eO

σ2
0
√m
√mσ2
1mm1


,

where the inequality follows from Theorem F.8 and Definition F.1.

F.8
Small Loss is Achieved

Theorem F.20. Define T ⋆= mint{t : bL(t) = Θ(1/poly(m))}. Then, T ⋆∈[T1, T2].

Proof. The following results altogether show that Phase 2 can last for at least Θ(poly(m)) time:

• Lemma F.19 proves that the change of K, Q self-correlation as follows:

max
µ,ν

µ⊤W (t)⊤
K
W (t)
K ν −µ⊤W (0)⊤
K
W (0)
K ν
 = eO(σ2
0
√m),

max
µ,ν

µ⊤W (t)⊤
Q
W (t)
Q ν −µ⊤W (0)⊤
Q
W (0)
Q ν
 = eO(σ2
0
√m).

• Corollary F.9 proves that

m1
X

j1=1

m1
X

j2=1

D
aj1w(t)
j1 , aj2w(t)
j2
E
≤O(σ2
1mm1).

• Corollary F.12 proves that

maxµ∈{µi}d
i=1


Pm1
j=1 aj
∂w(t)
j
∂t W (t)
V µ

Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1/L) 1

n

X

i∈[n]
g(t)
i .

• Corollary F.15 proves that


L
X

l1=1

X

l2: X(i)
l2 ∈{µk}d
k=4

G(t)(X(i)
l2 )p(t,i)
q←l1,k←l2


≤O(1).

• Corollary F.17 proves that

max
i∈[n],l1,l2∈[L]

p(t,i)
l1,l2 −p(0,i)
l1,l2

 ≤O(1/L2).

• Corollary F.18 proves that

maxi∈[n]


PL
l1=1
PL
l2=1 G(t)(X(i)
l2 )
∂p(t,i)
l1,l2
∂t


Pm1
j1=1
Pm1
j2=1
D
aj1w(t)
j1 , aj2w(t)
j2
E
≤o(1) 1

n

X

i∈[n]
g(t)
i .

• Lemma F.5 implies that

1
2 ≤

P

i∈I2 g(t)
i
P

i∈I3 g(t)
i
≤2.

62


---Page Break---
• Lemma F.6 implies that the gradients P

i∈I g(t)
i
for I ∈{I1, I2, I3, I4} satisfies

X

i∈I4
gi ≤min

 X

i∈I2
g(t)
i ,
X

i∈I3
g(t)
i

!

≤max

 X

i∈I2
g(t)
i ,
X

i∈I3
g(t)
i

!

≤
X

i∈I1
g(t)
i
≤
X

i∈I2∪I3∪I4
g(t)
i .

• Lemma F.6 and Corollary F.17 proves that yiF (t)
i
≥yiF (T1)
i
≥C.

The above shows that all the requirements needed to satisfy the definition of Phase 2 (Definition F.1)
can hold for at least Ω(poly(m)) time. Thus, T2 −T1 ≥Ω(poly(m)). Further, by Theorem F.8, we
have that bL(t) ≤O(1/poly(m)) implies t = Θ(poly(m)).

F.9
Proof of Theorem 3.2

Proof. This is proved in Corollary F.13 and Lemma F.16.

F.10
Proof of Theorem 3.3

Proof. This is proved as a direct consequence of Lemma F.6 and Theorem F.20.

For the generalization loss, since the training loss satisfies bL(T ⋆) ≤1/poly(m), for each class I ∈
{I1, I2, I3, I4} there exists a sample X⋆
I such that ℓ(X⋆
I ) ≤1/poly(m). Note that by Definition F.1
the random tokens only contributes to O(1) in F (T ⋆)(X). Thus, given a fixed new sample X ∼D,
we have |F (T ⋆)(X) −F (T ⋆)(X⋆
I )| ≤O(1) which implies ℓ(yF (T ⋆)(X)) ≤1/poly(m). Since this
holds for all X ∼D, we have L(T ⋆) ≤1/poly(m).

G
Auxiliary Results

The gradient of p(x)i = Softmax(x)i for x ∈Rn:

∂p(x)i

∂xj
=
∂
∂xj

exp(xi)
Pn
k=1 exp(xk) =






−
exp(xi)
P

k exp(xk)
exp(xj)
P

k exp(xk) = −p(x)ip(x)j
i ̸= j

exp(xi)
P

k exp(xk) −

exp(xi)
P

k exp(xk)
2
= p(x)i(1 −p(x)i)
i = j

= p(x)i(I(i = j) −p(x)j)

⇒J(p(x)) = diag(p(x)) −p(x)p(x)⊤.
(17)

Lemma G.1 (Gradient of softmax). Let s(t) ∈Rl be differentiable in t and p(s) = Softmax(s).
Denote pi(s) = Softmax(s)i. Then

∂p(s(t))

∂t
= ∂p(s)

∂s
∂s(t)

∂t
= (diag(p(s)) −p(s)p(s)⊤)∂s(t)

∂t .

Proof. By the chain rule and the gradient of softmax in Equation (17), we obtain

∂p(s(t))

∂t
= ∂p(s)

∂s
∂s(t)

∂t
= J(p(s))∂s(t)

∂t
= (diag(p(s)) −p(s)p(s)⊤)∂s(t)

∂t .

Lemma G.2 (Perturbation of softmax). Let s ∈Rl and p(s) = Softmax(s). Denote pi(s) =
Softmax(s)i. Consider a small perturbation ε ∈Rl to s. Then

p(s + ε) −p(s) = (diag(p(s)) −p(s)p(s)⊤)ε + ξ,

where ∥ξ∥∞= O(∥ε∥2
2).

Proof. By Taylor’s expansion theorem on softmax and the gradient of softmax in Equation (17), we
have

p(s + ε) −p(s) = J(p(s))ε + ξ = (diag(p(s)) −p(s)p(s)⊤)ε + ξ,

where ∥ξ∥∞= O(∥ε∥2
2).

63


---Page Break---
H
Probability

Lemma H.1 (Bernstein’s inequality for bounded random variables). Assume Z1, . . . , Zn are n i.i.d.
random variables with E[Zi] = 0 and |Zi| ≤M for all i ∈[n] almost surely. Let Z = Pn
i=1 Zi.
Then, for all t > 0,

P[Z > t] ≤exp

 

−
t2/2
Pn
j=1 E[Z2
j ] + Mt/3

!

≤exp

 

−min

(
t2

2 Pn
j=1 E[Z2
j ],
t
2M

)!

,

which implies with probability at least 1 −δ,

Z ≤

v
u
u
t2

n
X

j=1
E[Z2
j ] log 1

δ + 2M log 1

δ .

Lemma H.2. For w1, w2 ∈Rm with w1, w2
i.i.d.
∼N(0, Im/m), we have

P

"
∥w1∥2
2 −1
 ≥

r

4
m log 2

δ + 4

m log 2

δ

#

≤δ,

P

"

|⟨w1, w2⟩| ≥

r

4
m log 2

δ + 4

m log 2

δ

#

≤δ.

Proof. We first have

E

∥w1∥2
2

= E

" m
X

i=1
w2
1,i

#

= 1.

Note that w2
1,i is a sub-Gamma random variable with parameters ( 4

m2 , 4

m). Thus, by Bernstein’s
inequality,

P

"
∥w1∥2
2 −E

∥w1∥2
2
 ≥

r

4
m log 2

δ + 4

m log 2

δ

#

≤δ.

Next,

E[⟨w1, w2⟩] = E

" m
X

i=1
w1,iw2,i

#

= 0

By Bernstein’s inequality, we obtain

P

"

|⟨w1, w2⟩| ≥

r

4
m log 2

δ + 4

m log 2

δ

#

≤δ.

64


---Page Break---
NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and precede the (optional) supplemental material. The checklist does NOT
count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: [NA]
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
Justification: [NA]

65


---Page Break---
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

Answer: [Yes]

Justification: [NA]

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

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not include experiments.

66


---Page Break---
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

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justification: [NA]

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

67


---Page Break---
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: The experiments are done with synthetic data of small scale and the behavior
of the experiment results are pretty consistent.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
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
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.

68


---Page Break---
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: [NA]
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
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
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper poses no such risks.

69


---Page Break---
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

Answer: [NA]

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: [NA]

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

Justification: [NA]

70


---Page Break---
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
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

71


---Page Break---
