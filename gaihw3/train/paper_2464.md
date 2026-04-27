Transformers on Markov data: Constant depth suffices

Nived Rajaraman ˚
UC Berkeley
Marco Bondaschi
EPFL
Kannan Ramchandran
UC Berkeley
Michael Gastpar
EPFL

Ashok Vardhan Makkuva
EPFL

Abstract

Attention-based transformers have been remarkably successful at modeling gen-
erative processes across various domains and modalities. In this paper, we study
the behavior of transformers on data drawn from kth-order Markov processes,
where the conditional distribution of the next symbol in a sequence depends on
the previous k symbols observed. We observe a surprising phenomenon empir-
ically which contradicts previous findings: when trained for sufficiently long, a
transformer with a fixed depth and 1 head per layer is able to achieve low test
loss on sequences drawn from kth-order Markov sources, even as k grows. Fur-
thermore, this low test loss is achieved by the transformer’s ability to represent
and learn the in-context conditional empirical distribution. On the theoretical
side, our main result is that a transformer with a single head and three layers can
represent the in-context conditional empirical distribution for kth-order Markov
sources, concurring with our empirical observations. Along the way, we prove that
attention-only transformers with Oplog2pkqq layers can represent the in-context
conditional empirical distribution by composing induction heads to track the pre-
vious k symbols in the sequence. These results provide more insight into our
current understanding of the mechanisms by which transformers learn to capture
context, by understanding their behavior on Markov sources. Code is available at:
https://github.com/Bond1995/Constant-depth-Transformers.

1
Introduction

Attention-based transformers have revolutionized the field of natural language processing (NLP) [1, 2]
and beyond [3, 4], achieving significant performance gains across tasks like machine translation, text
generation, and sentiment analysis. A key factor in their success is their ability to model sequences
far more efficiently, and the ability to learn in-context [5, 6].

To understand this capability, a canonical approach is to sample the input from a kth-order Markov
process, where the next symbol’s conditional distribution depends only on the previous k symbols.
Recent studies [7, 6, 8] have investigated the ability of transformers to learn Markov processes
and establish that learning happens in phases. The transformer eventually learns to represent the
conditional k-gram model, which is the in-context MLE of the Markov process.

The results in [6, 8] seem to suggest that for low depth transformers to learn Markov processes
of order k, it is essential that the number of heads scale linearly in k. At first glance, this is a bit
concerning - real world data generating processes often contain long-range dependencies. How is it
that transformers succeed at capturing these kinds of long-range dependencies, while at the same time
requiring so many heads to be able to capture the necessary context for kth-order Markov sources?

˚Correspondence to nived.rajaraman@berkeley.edu.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
¨ ¨ ¨
2
0
1
1
Xn`1
1
0
1
0

Figure 1: kth-order Markov process for k “ 4.
The symbol Xn`1 is sampled from the distri-
bution Pp¨|Xn, Xn´1, Xn´2, Xn´3q which only
depends on the last 4 symbols (marked in red).

Attention-only
Standard
L
2
rlog2pk ` 1qs
3
H
k
1
1

Table 1: Each column in this table indicates
that there is a transformer with L layers and
H heads in the first layer which can represent
the conditional k-gram model.2

To understand the nature of this phenomenon, we train low-depth transformers on kth-order Markov
sources. These experiments result in two surprising empirical phenomena that seem to contradict
previous findings: when trained for sufficiently long, piq a 2-layer, 1-head transformer can learn
kth-order Markov processes for k as large as 4, piiq a 3-layer, 1-head transformer is able to achieve
low test loss on sequences drawn from kth-order Markov sources, even as k grows to be as large as 8
(Fig. 3). In both cases, the values of k for which the models appear to learn kth-order Markov sources
are much higher than those predicted in prior experiments [6, 8]. This discrepancy shows that our
understanding of the mechanisms used by transformers to learn kth-order Markov processes is not
complete and raises a broader question:

What is the interplay between depth, number of heads and non-linearity in learning
kth-order Markov processes?

In this paper, we approach this question from the point of view of representation power, and provide
some partial explanations toward the phenomena illustrated previously.

Our main contributions are as follows:

1. We show, rather surprisingly, that the standard transformer architecture with 3 layers and 1
head per layer is capable of representing the conditional k-gram model (Definition 1), and
thereby learn kth-order Markov models in-context.
2. Along the way to building up to this result, we consider the simpler family of attention-
only transformers and show that they can represent the conditional k-gram model with
rlog2pk ` 1qs layers.
3. Under a natural assumption on the nature of the attention patterns learnt by the transformer,
we then argue that for k ě 3 attention-only transformers need at least r1 ` log2pk ´ 2qs
layers to represent a “kth-order induction head” (Definition 2). Empirically, transformers are
observed to learn kth-order induction heads whenever they achieve small test error [6].

The last result is a consequence of a more general tradeoff between the number of layers, L, and heads
per layer, H, an attention-only transformer requires to represent a kth-order induction head, under a
natural assumption on the learnt attention patterns. In conjunction, these results also reveal the role of
non-linearities (aside from the softmax in the attention) in the transformer architecture. In particular,
it appears that layer normalization plays a critical role in the ability of constant-depth transformers to
learn the conditional k-gram model. Together with the experimental results mentioned previously,
these results paint a more comprehensive picture about the representation landscape of transformers
in the context of kth-order Markov processes.

Notation.
Scalars are denoted by italic lower case letters like x, y and Euclidean vectors and
matrices in bold x, y, M, etc. The notation 0pˆq (resp. 1pˆq) refers to the all-zero (resp. all-
one) matrix. When it is clear from the context, we omit the dimensions of a matrix. Define
rSs fi t1, 2, . . . , Su for S P N. Ip¨q denotes the indicator function and UnifpSq denotes the uniform
distribution over a set S.

1.1
Related work

There is a large body of active research focused on studying different aspects of transformer models
[9, 10, 11, 12]. Our work closely relates to the aspects of understanding the representation power of

2The requisite embedding dimension and bit-precision to achieve a target additive approximation is discussed
in more detail in Sections 4 and 5

2


---Page Break---
transformers, and in-context learning. [13, 14, 15] study the representation capabilities of transformers
and show properties such as universal approximation and Turing-completeness. Viewing transformers
as sequence to sequence models, [16, 17] study their ability to model formal languages and automata.
Along more related lines to our work, [18, 19] present logarithmic depth transformer constructions
for representing a k-hop generalization of the notion of an induction head [20]. On the other hand the
theoretical and mechanistic understanding of in-context learning [21] has received much attention
lately [22, 23, 24, 25], focusing on different operating regimes and phases of learning. There are a
few recent papers which study the behavior of transformers when trained on data generated from
Markov processes, and generalizations thereof [5, 26]. In particular, [7, 8] study the optimization
landscape of gradient descent in learning generalizations of Markov processes, and [6] present a
study of how transformers learn to represent in-context k-gram models, focusing on different phases
of learning.

2
Preliminaries

We provide the necessary background for Markov processes, the conditional k-gram model, and the
transformer architecture.

2.1
Markov processes

Markov processes are one of the widely used models in sequence modeling [27]. The characterizing
property of these processes is that at any time step, the future evolution is only influenced by the most
recent states. More formally, a sequence pXnqně1 is a kth-order Markov process on a finite state
space rSs with the transition kernel P, if surely,

P
`
Xn`1 | X1, ¨ ¨ ¨ , Xn
˘
“ P
`
Xn`1 | Xn´k`1, ¨ ¨ ¨ , Xn
˘

This property allows us to capture the conditional distribution at any position using only its previous
k symbols. This motivates the notion of a conditional k-gram, its empirical counterpart, defined for
any sequence px1, . . . , xnq.
Definition 1 (Conditional k-gram model). Given a sequence px1, ¨ ¨ ¨ , xnq of length n in rSsn,
the conditional k-gram model x
Prkp¨ | x1, ¨ ¨ ¨ , xnq corresponds to the in-context estimate of the
distribution over symbols conditioned on the last k symbols, i.e. for x P rSs,

x
Prkpx | x1, ¨ ¨ ¨ , xnq fi

řn
i“k`1 Ipxi “ x, xi´1 “ xn, . . . , xi´k “ xn´k`1q
řn
i“k`1 Ipxi´1 “ xn, . . . , xi´k “ xn´k`1q
,

which is defined only so long as the denominator is non-zero. This structure is illustrated in Figure 2a.
It is well known that the conditional k-gram in Eq. (1) with Laplace smoothing corresponds to the
Bayes optimal estimate of the next symbol probability, when the data is drawn from fixed Markov
process sampled from a prior distribution [27].

In our experiments, we will consider kth-order Markov kernels sampled from a Dirichlet prior with
parameter 1. Namely, the transition Pp¨|X1 “ i1, ¨ ¨ ¨ , Xk “ ikq is sampled independently and
uniformly on the S-dimensional simplex ∆S
1 , for each tuple pi1, ¨ ¨ ¨ , ikq.

2.2
Transformer architecture

In this paper, we will consider variants of the standard transformer architecture in Figure 2b introduced
in [1], with the goal to understand the role of depth and the non-linearities in the architecture. The
simplest variant removes all the layer normalization and the (non-linear) feedforward layer, and is
referred to as an attention-only transformer. The L-layer 1-head attention-only transformer with
relative position encodings, operating on a sequence of length T is defined in Architecture 1.

The attention scores in layer ℓ, tattpℓq
n,i : i ď nu, are computed as Softmax
`␣@
W pℓq
K pxpℓq
j
`

ppℓq,K
n´j q, W pℓq
Q xpℓq
j
D
: j P rns
(˘
. The superscript pℓq indicates the layer index, and the matrices

W pℓq
K , W pℓq
Q , W pℓq
V
P Rdˆd capture the key, query and value matrices in layer ℓ. Note that the
attention-only transformer may include a feedforward layer with linear activations, i.e. a linear

3


---Page Break---
Input:
0
1
0
1
0
3
2
3
¨ ¨ ¨
0
1

match final
k symbols
Uniform
distribution

(a) Conditional k-gram model. The conditional k-gram
is the in-context estimate of the Markov process and
is realized in two steps. The first step is to find the
locations in the sequence (marked red) which match
the final k symbols (functionally, a kth-order induc-
tion head). The conditional k-gram model returns the
uniform distribution over the next symbol at these loca-
tions (marked blue).

Transformer layer

Layer norm

xpℓq

Multi-head
Attention
+ POS

rxpℓq

Layer norm

FFN

xpℓ`1q

Layer 1

...

Layer ℓ

...

Layer L

xp1q

Linear +
Prediction

Prp¨ | x1, ¨ ¨ ¨ , xT q

Emb
Emb
Emb
¨ ¨ ¨

x1
x2
xT

(b) Transformer architecture. POS refers to the
relative position encodings.

transformation. For representation purposes, this linear transformation can be combined with the
projection matrix in the attention layer, allowing the feedforward layer to be omitted from the model.
In the attention layer, we consider relative position encodings (the terms labeled in blue), which
translates the key and value vectors depending on the relative position of the embedded symbol.

for n “ 1, 2, ¨ ¨ ¨ , T; xp1q
n
“ Embpxnq P Rd.
(Input embeddings)
for ℓ“ 1, 2, ¨ ¨ ¨ , L, do

for n “ 1, 2, ¨ ¨ ¨ , T, do

rxpℓq
n “
ÿ

iPrns attpℓq
n,i ¨W pℓq
V
´
xpℓq
i
` ppℓq,V
n´i

¯
P Rd,
(Attention)

xpℓ`1q
n
“ xpℓq
n ` rxpℓq
n ,
(Residual)

Ź Here, attpℓq
n,i “ Softmaxi
´!A
W pℓq
K pxpℓq
i
` ppℓq,K
n´i q, W pℓq
Q xpℓq
n
E
: i P rSs
)¯
.

logitT “ AxpL`1q
T
` b
P RS,
(Linear)

Prθ p¨ | x1, ¨ ¨ ¨ , xT q “ f plogitT q P RS.
(Prediction)

Architecture 1: Attention-only transformer.

The extension to H heads is straightforward, where in each transformer layer there are H attention
layers in parallel, resulting in ypℓ,1q
n
, ¨ ¨ ¨ , ypℓ,Hq
n
P Rd for each n. These vectors are concatenated and
passed through a linear transformation W pℓq
O
: RdH Ñ Rd which is the output of the attention layer.
Finally, the output of the model after L layers is passed through a linear layer, which projects the
d-dimensional embeddings back into RS and the resulting vector is passed through a non-linearity f,
usually a softmax, to result in the model’s prediction of the next symbol probabilities. The theoretical
results in this paper will choose f “ ReLUp¨q.

3
Understanding the empirical behavior of transformers

The motivation for the present work comes from a series of experimental results, which challenge our
current understanding of transformers in the context of learning Markov processes. Several works in
the literature [7, 8, 6] have studied the ability of transformer models to learn kth-order Markov pro-
cesses. The experimental results present in the literature suggest that in order for a 2 layer transformer
model to be able to learn a randomly sampled Markov process of order k, it is crucial for the number
of heads in the first attention layer to scale linearly with the order, k. In particular, the authors of [6]
claim that in their experiments, “Single attention headed models could not achieve better performance
than bigram (models)” in learning random kth-order Markov processes in-context. Similarly, the

4


---Page Break---
0
5000
10000
15000

0.1

0.2

0.3

0.4

          
 

Iteration

Test loss gap from the optimal

k “ 2
k “ 4

0
5000
10000
15000
20000
25000

0.0

0.1

0.2

0.3

0.4

0.5

          
 
 

Iteration

Test loss gap from the optimal

k “ 2
k “ 4
k “ 8

Figure 3: Gap with the optimal test loss for paq a 2-layer, 1-head transformer model (above), and
pbq a 3-layer, 1-head transformer (below), averaged over 3 runs for each k. The models learn the
conditional k-gram model for randomly sampled k-th order Markov processes, even for large k.

authors of [8] study a generalization of learning kth-order Markov processes to learning causal
processes on degree k graphs. The theory and experiments pertain to 2-layer k head transformers.

In Figure 3, we train 2 and 3-layer transformers with a single head on data drawn from random
Markov processes of various orders drawn from a Dirichlet prior. With 2 layers and a single head,
we see that the model is able to learn even order-4 Markov processes, and go beyond the simple
order-1 processes which were projected to be the limit of its ability to learn. Likewise, with 3 layers,
transformers are able to go much further and learn order-8 Markov processes, which was the largest
value of k we evaluated on. These results contrast with our current understanding of how induction
heads are realized in the parameter space [6, 8] - existing constructions which realize these attention
patterns require k heads when the number of layers is 2, and it’s unclear how to implement them with
fewer heads. At a high level, each of the k heads play a critical role - where, loosely speaking, the
ith-head looks back i positions in the sequence.

Building up to our main results, in the sequel, we study the simpler case of attention-only transformers
where the feedforward layers and layer normalization are removed.

4
Warming up: Attention-only transformers

The study of attention-only transformers trained on Markov processes has garnered some attention
in the prior literature. Notably, the authors of [6] study 2-layer 1-head attention-only transformers
trained on data drawn from 1st-order Markov processes whose parameters are drawn from a Dirichlet
prior. The model is observed to learn a very specific behavior, known as an “induction head” [20],
which in this setting is able to represent the conditional 1-gram (Eq. (1)).

The induction head mechanism is composed of two layers where the first layer learns the attention
pattern attp1q
n,i “ Ipi “ n ´ 1q, thereby allowing the model to capture information about the symbol
at position n ´ 1 in the embedding vector at time n. In the second layer, the attention layer picks
out those indices n where xn´1 “ xT , the final symbol in the sequence. At these positions, since
xn´1 “ xT , one would expect that the next symbol xn is a good predictor of xT `1, and the model
uses this information to predict the next symbol xT `1 according to its conditional empirical estimate,
x
Pr1pxT `1|x1, ¨ ¨ ¨ , xT q, i.e. the conditional 1-gram model.

Theorem 1. The conditional 1-gram model can be represented by a 2-layer and 1-head attention-only
transformer with embedding dimension d “ 3S ` 2.

Although a version of this result is proved in [6], we include a proof in Appendix A for completeness.

Remark 1. In Theorem 1 and other results to follow, we de-emphasize the role of the bit-precision to
which the transformer is implemented. That said, note that when the constructions in Theorems 1
to 3 are implemented to OplogpTqq bits of precision, the representation results are realized up to an
additive Op1{Tq error.

The ideas in Theorem 1 readily extend to representing the conditional k-gram model, by instead
using k heads in the first layer. The jth head learns the attention pattern attp1q
n,i “ Ipi “ n ´ jq and

5


---Page Break---
concatenating the outputs of the heads, the model learns to aggregate information about xn, ¨ ¨ ¨ , xn´k
in the embedding vector at time n. The second layer realizes what is best described as a “kth-order ”
induction head, where the model learns to pick out those positions n where for every j P rks,
xn´j “ xT ´j`1, i.e. the history of length k at those positions match the final k symbols in the input
sequence ( see Figure 4). This mechanism is also referred to as a long-prefix induction head [28].

Definition 2 (Higher-order induction head). A 1-head attention layer is said to realize a kth-order in-
duction head if on any sequence px1, ¨ ¨ ¨ , xT q P rSsT , for any fixed n ď T, as a function of the input
sequence, attn,T is maximized if and only if xn´j “ xT ´j`1 for every j P rks.

Input:

n “

0
1
0
1
0
3
2
3

1
2
3
6
4
5
7
8
9 10

10
1
2
3
4
5
6
7
8
9
0

0.5

1

n

attT,n

0
1

Figure 4: kth-order induction head for k “ 2. The at-
tention pattern attT,n is maximized for those values of
n at which xT ´j`1 “ xn´j for all j P rks. These are
the positions where the k-length prefix at those posi-
tions matches with the last k symbols in the sequence.

kth-order induction heads generalize the
concept of an induction head [20], and keep
track of the positions i ď n where there is
a perfect occurrence of the final k symbols
in the sequence. Such attention patterns
are immediately useful in representing the
conditional k-gram - increasing the temper-
ature within the softmax of this attention
layer results in an attention pattern which
converges to the uniform distribution over
those positions where the final k symbols
xT ´k`1, ¨ ¨ ¨ , xT are seen previously in the
sequence. Loosely, this allows the model
to “condition” on the last k symbols in the
sequence. With k heads, the model can ag-
gregate information from the previous k po-
sitions and implement a kth-order induction
head, which leads to the following result. A
full proof is discussed in Appendix A.1.

Theorem 2. The conditional k-gram model can be represented by an attention-only transformer with
2 layers, k heads and embedding dimension d “ pk ` 2qS ` k ` 1.

While this result is positive, it suggests that a 2-layer transformer requires approximately k times as
many parameters to be able to represent the conditional k-gram model. The first result we prove is
that increasing the depth of the model is exponentially more beneficial, in that a transformer with
Oplogpkqq depth can estimate in-context k-grams.

Theorem 3. The conditional k-gram model can be represented by an attention-only transformer with
relative position encodings, with L “ rlog2pk ` 1qs layers and 1 head per layer. The embedding
dimension is ď 2kpS ` 1q ` S.

With 2 layers and k heads, the transformer aggregates information about each of the previous k
positions one step at a time through the k heads. However, with Ωplogpkqq layers, the same task
can be done far more efficiently. In the first attention layer, the model aggregates information about
the current and previous position. Namely, using the relative position embeddings, xp2q
n
is chosen
as a linear combination of xp1q
n
“ Embpxnq and xp1q
n´1 “ Embpxn´1q. This allows the embedding at
position n to aggregate information about xn and xn´1. In the same vein, in the second attention
layer, the model aggregates information from xp2q
n
and xp2q
n´2 in xp3q
n ; the former has information
about xn and xn´1, and the latter has information about xn´2 and xn´3. This expands the “window”
of xi’s on which xn depends on to size 4. In the ℓth layer, the model aggregates information from xpℓq
n
and xpℓq
n´2ℓwhich allows xpℓ`1q
n
to effectively depend on the xi’s in a window of size 2ℓ`1 starting

at position n, namely xn, ¨ ¨ ¨ , xn´2ℓ`1`1. In the final layer, the embedding at position i, xpLq
i
for
L “ rlog2pk ` 1qs depends on xn, xn´1, ¨ ¨ ¨ , xn´k. In the last layer, the model can realize the
dot-product
A
W pLq
K xpLq
n , W pLq
Q xpLq
T
E
“ řk
j“1 Ipxn´j “ xT ´j`1q by choosing the key and query
vectors appropriately. By increasing the temperature in the attention softmax, the attention pattern
realized is the uniform distribution on values of n such that xn´j “ xT ´j`1 for every j P rks, i.e., a
kth-order induction head. The full proof of this result is provided in Appendix B.

6


---Page Break---
Multi-head
Attention
+ POS

xpℓq

Ă
xypℓq

FFN

Layer norm

xpℓ`1q

(a) Rearranged transformer layer with
layer normalization and FFN.

Layers 1 ` 2

»

——–

Embpxnq
EmbpxT q

¨ ¨ ¨
un
}un}2
¨ ¨ ¨
uT
}uT }2
vn
}vn}2

vT
}vT }2

fi

ffiffifl

Layer 3

r ¨ ¨ ¨
Embpxiq
¨ ¨ ¨
EmbpxT q s

attT,i 9 exp
´
κ
xvi,uT y
}vi}2}uT }2

¯

(realizes a kth-order induction head)

Emb

x1, ¨ ¨ ¨ , xT

(b) Realizing a kth-order induction head in a 3-layer transformer
following the architecture in Figure 5a.

Figure 5: Disassembling the constant-depth construction. The first two layers are critical in the
model’s ability to capture information from the previous k positions. Layer normalization plays a
critical role in in the 3rd layer which realizes a kth-order induction head.

While this is a promising step toward understanding the behavior transformers exhibit in Figure 3,
showing that depth plays an important role in their ability to represent conditional k-gram models,
the picture is still not complete. The experimental results in Section 3 do not preclude the possibility
that a transformer might not even require logarithmic depth to be able to learn kth-order Markov
processes approximately. In the next section, we will study constant-depth transformers and establish
a rather surprising positive result about the representation power of this class of models in capturing
conditional k-grams.

5
Understanding the role of non-linearity: Constant-depth constructions

In the previous section, we saw how the transformer uses the power of depth to learn conditional
k-grams far more efficiently. In particular, every additional attention layer effectively doubles the
window of positions i “ n ´ 1, n ´ 2, ¨ ¨ ¨ which the model has access to information about at the
current time n. By composing L “ Ωplogpkqq attention layers, the model is able to collect enough
information within the output embedding xpL`1q
n
to be able to realize a kth-order induction head in
the next layer. In this section, we prove that adding non-linearity to the architecture, in the form of
layer normalization, can significantly change the mechanism in which the transformer realizes this
kth-order head. In particular, there are constant depth architectures which allow a kth-order induction
head to be realized, surpassing the logarithmic depth attention-only constructions.

Modification to the standard transformer architecture.
To simplify the proof of our main result,
we will consider a subtle modification to the standard transformer architecture, which is presented in
Architecture 2 and Figure 5a. We will remove the first layer norm prior to the multi-head attention
and move the second layer norm to after the feed-forward network. It is important to note that
Theorem 4 holds even for the architecture presented in Figure 2b, which is the architecture we
evaluate empirically. The modification we present in Figure 5a allows the construction to be simpler
and makes it much easier to convey the key intuition. The main difference compared to the attention-
only design presented in Architecture 1 is the addition of layer normalization and a feedforward layer
in the for-loop over n P rTs for each transformer layer ℓ. The differences between Architectures 2
and 1 are emphasized in blue.

Theorem 4. Conditional k-grams can be represented by a transformer with 3 layers, 1 head per
layer, relative position encodings and layer normalization. The embedding dimension is OpSq.

Remark 2. Although the proof stated does not bound the approximation error arising from a finite
bound on the bit precision of the transformer, in theory, it should suffice to have ΩplogpTq ` kq bits
per parameter for the statement of Theorem 4 to go through with an Op1{Tq additive approximation
error. The main point is that none of the weights of the model exceed exppkq and with logpTq
additional bits per parameter, the approximation error scales as Op1{Tq.

7


---Page Break---
rxpℓq
n “ xpℓq
n `
ÿ

iPrns attpℓq
n,i ¨W pℓq
V
´
xpℓq
i
` ppℓq,V
n´i
¯
P Rd,
(Attention + Residual1)

ypℓq
n
“ W pℓq
2
ReLU
´
W pℓq
1
rxpℓq
n
¯
P Rd,
(FFN)

lpℓq
n
“ ypℓq
n ´ µ1dˆ1

σ
P Rd,
(LN)

xpℓ`1q
n
“ lpℓq
n ` rxpℓq
n P Rd,
(Residual2)

Architecture 2: Modified transformer architecture. The computations above are carried out for each
n P rTs in each layer ℓP rLs. In the layer normalization step (LN), the feature mean µ is defined as,
Ei„Unifprdsq
“@
ed
i ypℓq
n
D‰
and the feature variance σ2 “ Ei„Unifprdsq
“@
ed
i , ypℓq
n
D2‰
´ µ2.

5.1
Proof sketch

In the attention-only transformer with 2 layers and k heads, the model is able to keep track of where
the final k symbols in the sequence appeared previously (i.e., a kth-order induction head) by, loosely,
using each head to keep track of the occurrences of one of the final k symbols. On the other hand,
with the benefit of more depth, with L “ Ωplogpkqq layers, the model is able to collect enough
information within the output embedding xpL`1q
n
to be able to realize the same behavior. However,
neither of these constructions scale down to the case when the depth and number of heads of the
transformer are both constants independent of k. We provide a brief sketch of the construction below.

Recall that a kth-order induction head keeps track of the indices i such that @j P rks, xi´j “
xn´j`1. Defining zi fi řk
j“1 2jexi´j`1, notice that the condition t@j P rks, xi´j “ xn´j`1u
can equivalently be captured by writing tzi´1 “ znu. This true because of the fact that the binary
representation of any integer is unique. Furthermore, these vectors, up to scaling, can be realized by
softmax attention (namely, attn,n´i 9 2i for 1 ď i ď k).

With this step, finding occurrences of the last k symbols in the input sequence boils down to realizing
an attention pattern in the second layer, attp2q
n,i, which is maximized whenever zi´1 “ zn. While
dot-product attention naively encourages those values of i for which zi´1 and zn are “similar” to
each other, a qualitative statement is lacking. In general, it will turn out to that a different measure of
similarity is necessary within the softmax to be able to encourage those values of i for which these
vectors match. This is where the role of layer-normalization comes in.

Instead of the usual dot-product, suppose the attention mechanism in the second layer was,

attp2q
n,i 9 exp

˜

´κ
››››
zi´1
}zi´1}2
´
zn
}zn}2

››››

2

2

¸

,
(1)

where κ is the temperature parameter. Then, as the temperature κ grows, the attention pattern
essentially focuses on those values of i for which zi{}zi´1}2 “ zn{}zn}2. With this attention
pattern, we are thus very close to the statement we wanted to check, (zi´1
?“ zn). As it turns out, for
the special structure in the zi’s considered (dyadic sums of one-hot vectors), we may write down,

zi´1 “ zn ðñ zi´1{}zi´1}2 “ zn{}zn}2.

A quantifiable equivalence is provided in Lemma 1.

Realizing L2-norm attention (eq. (1)).
Observe the equivalence,
B zi´1

}zi´1}2
,
zn
}zn}2

F
“ 1 ´ 1

2

››››
zi´1
}zi´1}2
´
zn
}zn}2

››››

2

2
(2)

Taking a softmax on both sides, notice that the RHS (up to an additive constant) is the L2-norm based
attention, while the LHS is the usual dot-product attention between zi´1{}zi´1}2 and zn{}zn}2.
Thus on unit-normalized vectors, L2-norm attention and dot product attention are but the same.

While the first layer of the transformer computes the zi’s by a weighted summation, layer normaliza-
tion fills in the last missing piece of the puzzle which is to normalize them to unit norm. This is a
consequence of defining the embedding vectors appropriately, as we discuss more in Appendix C.1.

8


---Page Break---
From this step, realizing the actual conditional k-gram model follows readily. In particular, as the
temperature κ in the attention grows, the attention pattern zooms in on indices i P In fi tk ` 1 ď
i ď n : @j P rks, xi´j “ xn´j`1u in the last layer. The value vectors at this step are the one-hot
encoding of xi; putting everything together, the logits realized by the transformer are,

logitT pxT `1q “
1
|In|

ÿ

iPIn
Ipxi “ xT q,
(3)

which is the conditional k-gram model (eq. (1)).

While the transformer construction described above only requires two layers, the actual construction
we propose differs slightly and has an additional layer. The first two layers of the transformer
respectively compute zi and zi´1 which are added to the embedding vector at time i. This is
important because we need to test whether zi´1
?“ zn and not whether zi
?“ zn or zi´1
?“ zn.

Summary.
The construction can be summarized as follows: the first layer computes zn “
řk
j“1 2j´1 ¨ exn´j by choosing appropriate value vectors and relative position embeddings to realize
the attention pattern attnn,n´i 9 2iIp1 ď i ď kq. The layernorm that follows subsequently can be
replaced by RMSnorm, by a simple trick which we discuss in Appendix C.1, resulting in zn{}zn}2
to be appended to the embedding at time n. Using a very similar construction, layer 2 computes
zn´1{}zn´1}2, which is added to the embedding at time n. Finally, in the last layer, the dot-product
xzi´1{}zi´1}2, zn{}zn}2y defines the attention score, and as the temperature κ grows, the pattern
converges to UnifpInq. Choosing the value vectors in this layer appropriately gives eq. (3).

6
Lower bounds on transformer size

In this section, we study the limits of how shallow a transformer can be made while still capturing
conditional k-grams. The first result we establish in this vein is a lower bound against 1-layer
transformers showing that their expressive power is too limited unless the embedding dimension or
number of heads scale near-linearly in T.

Theorem 5. Consider any 1-layer transformer with layer normalization and feedforward layers,
where all the coordinates of the embedding vectors and unnormalized attention scores are computed
with p bits of precision. If the transformer is able to compute the conditional 3-gram on inputs drawn
from t0, 1, 2uT to within an additive error of 1{3T, then 2pH ` dp ` 2 ě T{3.

Choosing the bit precision to be p “ OplogpTqq, this implies that for transformers with 1 layer, the
sum of the number of heads and the embedding dimension must be at least ΩpT{ logpTqq, in order to
represent conditional 3-grams to within an additive error of 1{3T.

6.1
Conditional lower bounds on attention-only transformers

While the previous section shows that 1-layer transformers have fairly limited representation power,
it is not immediately clear how whether any of these issues are present with transformers with more
layers. Indeed, as we discussed in Section 4, an attention-only transformer with Oplog2pkqq layers
and 1 head per layer can represent conditional k-grams on its input sequences. With the addition
of non-linearities, Theorem 4 shows that the model can represent conditional k-grams using just a
constant number of layers. In this section, we try to understand the gap between these two results
and prove conditional lower bounds on the size of attention-only transformers which do not have
non-linearities arising from layer normalization.

We prove conditional lower bounds under some natural assumptions on the nature of the attention
patterns learnt by the transformer. To motivate these assumptions, consider the experiment in Figure 6,
where we train an attention-only transformer with 2 layers and 1 head, on order-1 Markov processes.
At test-time, we plot the attention patterns learnt in the first layer of the model on test sequences.
Notice that the attention pattern learnt by the model at layer 1 is largely independent of the input
sequences themselves and only depends on the position.

Assumption 1. In an L-layer attention-only transformer with H heads per layer, assume that layers
ℓ“ 1, 2, ¨ ¨ ¨ , L ´ 1 and heads h P rHs realize an attention pattern where attpℓ,hq
n,i
only depends on
the positions n and i and on ℓand h, but not on the input sequence x1, ¨ ¨ ¨ , xT .

9


---Page Break---
0
5
10
15
20
25
30

0

5

10

15

20

25

30

0.0

0.2

0.4

0.6

0.8

1.0

Index i

Index n

0
5
10
15
20
25
30

0

5

10

15

20

25

30

0.00

0.05

0.10

0.15

0.20

0.25

0.30

Index i

Index n

0
2
4
6
8
10
0.0

0.2

0.4

0.6

0.8

1.0

Sequence index piq

Attention weight

Figure 6: Attention matrix of the first attention layer, for a 2-layer 1-head transformer model trained
on an order-1 Markov process, averaged across 100 input sequences of length 128. (a) and (b) plot
the mean and standard deviation of the first 32 rows and columns of the attention matrix, while (c)
zooms in on the column n “ 10 and plots the mean attention for this column. (a) and (c) show that
for almost all indices n, the attention layer focuses only on the previous symbol xn´1. (b) shows
that the attention pattern does not vary much with the input sequence considered, thereby providing
evidence toward Assumption 1. More discussion in Appendix G.

Rather than proving the size lower bound depending on the transformers ability to represent the
conditional k-gram itself, we consider a simplification and assume that the goal of the model is to
represent a kth-order induction head (Definition 2) in the last layer. Although learning a kth-order
induction head is not strictly necessary for the transformer to be able to represent conditional k-
grams, note that every construction we have considered so far (cf. Theorems 1 to 4) go through this
mechanism to realize the conditional k-gram model. Likewise, for other related problems, such as the
causal learning task in [8], the causal structure is captured by an extension of the kth-order induction
head to general causal graphs. Our main lower bound is the following result.
Theorem 6. Consider an L-layer transformer with hℓheads in layer L. Assuming the transformer
satisfies Assumption 1, if śL´1
ℓ“1 pHℓ` 1q ď k ´ 2, the attention pattern in layer L cannot represent
a kth-order induction head.

While this lower bound is not unconditional, meaning that it does not directly imply that the trans-
former cannot represent conditional k-grams, it is important to understand the interpretation of
this result: attention-only transformers which somehow break through this barrier need to use a
significantly different mechanism to realize the conditional k-gram model.

Theorem 6 implies that under Assumption 1, a 2-layer attention-only transformer with 1 head cannot
realize a kth-order induction head for any k ě 4. Likewise, under the same assumption, a 3-layer
attention-only transformer with 1 head cannot realize a kth-order induction head for any k ě 6. These
results give more weight to the experiment in Figure 3 where we observe that a 2-layer transformer
learns a kth-order Markov process for k “ 4 and a 3-layer transformer learns a kth-order Markov
process for k “ 8, and show that non-linearities in the architecture allow the transformer to break
past the size barriers in Theorem 4.

7
Conclusion

We observe empirically that 2 and 3 layer transformers are able to learn kth-order Markov chains for
much higher values of k than previously anticipated. We show there are Oplogpkqq-layer constructions
of attention-only transformers which are able to learn the conditional k-gram model, which is the
in-context MLE of the Markov model. With non-linearities in the model, we show that a 3-layer
1-head transformer is capable of representing the same. We show that 1-layer transformers cannot
represent conditional k-grams for any k ě 3 unless the number of heads or embedding dimension
scale almost linearly in T. We also prove a conditional lower bound on the depth and number of
heads of attention-only transformers to represent kth-order induction heads, under an assumption on
the realized attention patterns.

Acknowledgments and Disclosure of Funding

The work was partially supported by NSF Grant CCF-2211209 and Swiss NSF Grant 200364.

10


---Page Break---
References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
In Advances in Neural
Information Processing Systems, pages 5998–6008, 2017.

[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[3] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image
recognition at scale, 2021.

[4] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked
autoencoders are scalable vision learners, 2021.

[5] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth
of a transformer: A memory viewpoint. In Thirty-seventh Conference on Neural Information
Processing Systems, 2023.

[6] Benjamin L. Edelman, Ezra Edelman, Surbhi Goel, Eran Malach, and Nikolaos Tsilivis. The
evolution of statistical induction heads: In-context learning markov chains, 2024.

[7] Ashok Vardhan Makkuva, Marco Bondaschi, Adway Girish, Alliot Nagle, Martin Jaggi, Hyeji
Kim, and Michael Gastpar. Attention with markov: A framework for principled analysis of
transformers via markov chains. arXiv preprint arXiv:2402.04161, 2024.

[8] Eshaan Nichani, Alex Damian, and Jason D. Lee. How transformers learn causal structure with
gradient descent, 2024.

[9] Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In International

Conference on Machine Learning, pages 11080–11090, 2021.

[10] Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris
Papailiopoulos. Looped transformers as programmable computers. In Proceedings of the 40th
International Conference on Machine Learning, pages 11398–11442, 23–29 Jul 2023.

[11] Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, and Christos Thrampoulidis. On
the role of attention in prompt-tuning. In Proceedings of the 40th International Conference on
Machine Learning, 2023.

[12] Hongkang Li, Meng Wang, Sijia Liu, and Pin-Yu Chen.
A theoretical understanding of
shallow vision transformers: Learning, generalization, and sample complexity. In The Eleventh
International Conference on Learning Representations, 2023.

[13] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar.
Are transformers universal approximators of sequence-to-sequence functions? In International
Conference on Learning Representations, 2020.

[14] Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is Turing-complete. Journal of

Machine Learning Research, 22(75):1–35, 2021.

[15] Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study
on approximating Turing machines with transformers. In Advances in Neural Information
Processing Systems, volume 35, pages 12071–12083, 2022.

[16] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transform-
ers learn shortcuts to automata, 2023.

[17] Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of trans-
formers to recognize formal languages, 2020.

11


---Page Break---
[18] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and
logarithmic depth, 2024.

[19] Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations
of transformers. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
editors, Advances in Neural Information Processing Systems, volume 36, pages 36677–36707.
Curran Associates, Inc., 2023.

[20] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom
Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain,
Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson
Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan,
Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022.

[21] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems, 35:24824–24837, 2022.

[22] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection, 2023.

[23] Ziqian Lin and Kangwook Lee. Dual operating modes of in-context learning, 2024.

[24] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning
algorithm is in-context learning? investigations with linear models, 2023.

[25] Jesse Hoogland, George Wang, Matthew Farrugia-Roberts, Liam Carroll, Susan Wei, and Daniel
Murfet. The developmental landscape of in-context learning. arXiv preprint arXiv:2402.02364,
2024.

[26] Nived Rajaraman, Jiantao Jiao, and Kannan Ramchandran. Toward a theory of tokenization in
llms, 2024.

[27] J. R. Norris. Markov Chains. Cambridge Series in Statistical and Probabilistic Mathematics.
Cambridge University Press, 1997.

[28] Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. Localizing model
behavior with path patching. arXiv preprint arXiv:2304.05969, 2023.

[29] Andrew Chi-Chih Yao. Some complexity questions related to distributive computing (prelimi-
nary report). In Proceedings of the eleventh annual ACM symposium on Theory of computing,
pages 209–213, 1979.

[30] Matteo
Pagliardini.
GPT-2
modular
codebase
implementation.
https://github.com/epfml/llm-baselines, 2023.

12


---Page Break---
Appendix

Table of Contents

A
Proof of Theorem 1
13
A.1
Extension to k-heads: Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . .
14

B
Proof of Theorem 3
15

C
Proof of Theorem 4
18
C.1
Modifying the definition of layer normalization . . . . . . . . . . . . . . . . .
18
C.2
Notation and supplementary lemmas . . . . . . . . . . . . . . . . . . . . . . .
18
C.3
Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

D
Representation lower bounds for 1-layer transformers: Proof of Theorem 5
22

E
Lower bounds on representing kth-order induction heads: Proof of Theorem 6
23
E.1
Lower bounds on 2-layer 1-head attention-only transformers
. . . . . . . . . .
23
E.2
L-layer attention-only transformers with 1 head per layer: Proof of Corollary 1 .
26
E.3
The general case: Transformers with Hℓheads in layer ℓ: Proof of Theorem 6 .
27

F
Model architecture and hyper-parameters
28

G
Additional experimental results
28

Notation.
The notation ed1
i P Rd1 refers to the one-hot encoding of i in d1 dimensions. In other
words it is the ith standard basis vector in d1 dimensions. The notation BlkdiagptA1, A1, ¨ ¨ ¨ , Amuq
refers to the block diagonal matrix with ith block as Ai.

A
Proof of Theorem 1

We will first prove Theorem 1. In the first layer, choose the embeddings as,

xp1q
n
“ Embpxnq “ κ
“
11ˆ2
eS
xn
01ˆ2S
‰T P Rd.
(4)
for a constant κ ą 0 to be chosen later and d “ 2S ` 2. The relative position encodings will
essentially be supported on the first two coordinates, the middle S coordinates are a one-hot encoding
of the symbol xn and the last 2S coordinates are 0. The relative position encodings in the first layer
are chosen to be pp1q,K
n´i
“ κ p´1 ` Ipn ´ i “ 1uqq ed
1 P Rd and pp1q,V
n´i
“ 0 P Rd. Choose W p1q
K
and W p1q
Q
to be ed
1ped
1qT P Rdˆd. With this choice,
A
W p1q
K
`
xp1q
i
` pp1q,K
n´i
˘
, W p1q
Q xp1q
n
E
“ κIpn ´ i “ 1q
(5)

As κ Ñ 8, the attention pattern (which takes the softmax over of these inner products over i P rns)
computes,

attp1q
n,i “ Ipi “ n ´ 1q
(6)
for any n ą 1. Choose the value matrix as,

W p1q
V
“

«0p2`Sqˆ2
0
0
0
ISˆS
0
0
0
0

ff

P Rdˆd
(7)

And with this choice and the residual connection, we get,
xp2q
n
“ κ
“11ˆ2
eS
xn
eS
xn´1
0‰
P Rd
(8)

which serves as the input to the 2nd transformer layer.

13


---Page Break---
Layer 2.
In layer 2, the relative position encodings pK,p2q
n´i
and pV,p2q
n´i are all set as 0. The key

matrix picks out the eS
xn block out of xp2q
n
and the query vector picks out the eS
xi´1 block out of xp2q
i´1.
In particular, these matrices are chosen so that,

W p2q
K xp2q
i
“ κ
“11ˆ2
eS
xi´1
0‰T P Rd,

W p2q
Q xp2q
n
“ κ
“
11ˆ2
eS
xn
0
‰T P Rd
(9)

Taking the inner product of these vectors, and taking κ Ñ 8, observe that the attention pattern
concentrates on the uniform distribution over all coordinates i such that xi´1 “ xn. More formally,
the attention pattern for any n ą 1 is,

attp2q
n,i “
Ipxi´1 “ xnq
řn
i“2 Ipxi´1 “ xnq,
(10)

assuming řn
i“2 Ipxi´1 “ xnq ą 0. Having realized this attention pattern, may choose the value and
subsequent linear layer appropriately. The value matrix simply picks out the eS
xi block from xp2q
i
and

places it into the last S coordinates of xp3q
i , and the linear layer simply extracts this block and outputs
it (after scaling down by a factor of κ), realizing the logits,

logitn “
1
řn
i“2 Ipxi´1 “ xnq

n
ÿ

i“2
Ipxi´1 “ xnq ¨ eS
xi.
(11)

if řn
i“2 Ipxi´1 “ xnq ą 0. In particular, under the same condition,

logitT pxT `1q “
řT
n“2 Ipxn “ xT `1, xn´1 “ xT q
řn
i“2 Ipxn´1 “ xT q
(12)

assuming řn
i“2 Ipxn´1 “ xT q, which is the conditional 1-gram model.

A.1
Extension to k-heads: Proof of Theorem 2

In the first layer, the embeddings are chosen to be,

xp1q
n
“ Embpxnq “ κ
“
01ˆk
1
eS
xn
01ˆpk`1qS
‰T P Rd
(13)

With d “ pk ` 1qpS ` 1q ` S. The relative position encodings are chosen as pK,p1q
i
“
“
ek
i
0
‰T for

1 ď i ď k and pK,p1q
i
“ 0 otherwise. Similarly, pV,p1q
i
“ 0 for every i. The hth head has key and
query matrices,

W p1,hq
Q
“
„
01ˆk
1
0
0
0
0

ȷ

W p1,hq
K
“
„
01ˆph´1q
1
0
0
0
0

ȷ
(14)

With these choices, and letting κ Ñ 8, the hth layer computes the attention pattern,

attp1,hq
n,i
“ Ipi “ n ´ hq.
(15)

Choose the corresponding value matrix as,

W p1,hq
V
“
„
0p2`hSqˆ2
0
0
0
ISˆS
0

ȷ
(16)

choosing the projection matrix appropriately, the output of the transformer after the first residual
connection is,

xp2q
n
“ κ
“ 01ˆk
1
eS
xn
¨ ¨ ¨
eS
xn´k
‰T .
(17)

14


---Page Break---
Layer 2.
In this layer, the relative position encodings pK,p2q
n´i
and pV,p2q
n´i are all set as 0. The key
and query matrices are chosen as,

W p2q
Q
“
„
0Skˆk
IpSk`1qˆpSk`1q
0
0
0
0

ȷ

W p2q
K
“
„
0Skˆpk`Sq
IpSk`1qˆpSk`1q
0
0

ȷ
.
(18)

With this choices, we have that,

A
W p2q
K xp2q
i , W p2q
Q xp2q
n
E
“ κ

kÿ

j“1
Ipxi´j “ xn´j`1q.
(19)

Taking κ Ñ 8, observe that the attention pattern concentrates on the uniform distribution over all
coordinates i such that xi´j “ xn´j`1 for all j P rks. More formally, if řn
i“2 Ipxi´1 “ xnq ą 0,
the attention pattern for any n ą 1 is,

attp2q
n,i “
Ip@j P rks, xi´j “ xn´j`1q
řn
i“k`1 Ip@j P rks, xi´j “ xn´j`1q.
(20)

The value matrix picks out eS
xi from the embedding xp2q
i
(Equation (17)) and places it in the last S
coordinates. The subsequent linear layer picks out the last S coordinates, resulting in the logits,

logitn “

n
ÿ

i“k`1

Ip@j P rks, xi´j “ xn´j`1q
řn
i“k`1 Ip@j P rks, xi´j “ xn´j`1qeS
xi,
(21)

assuming that řn
i“k`1 Ip@j P rks, xi´j “ xn´j`1q ą 0. In particular,

logitT pxT `1q “

řT
n“k`1 Ip@0 ď j ď k, xn´j “ xT ´j`1q
řT
n“k`1 Ip@1 ď j ď k, xn´j “ xT ´j`1q
,
(22)

assuming řT
n“k`1 Ip@1 ď j ď k, xn´j “ xT ´j`1q ą 0, i.e., the conditional k-gram model.

B
Proof of Theorem 3

Define k‹ “ 2rlog2pk`1qs by rounding k ` 1 up to the nearest power of 2 and ℓ‹ “ log2pk‹q. In the
setting of relative position encodings, given the sequence x1, ¨ ¨ ¨ , xn, while generating the output of
the attention + feedforward layer for the symbol xn, the embeddings xi “ Embpxnq ` pn´i are used
for i P rns. In other words, the position encoding vector is taken relative to the end of the sequence,
rather than the start of the sequence. Consider the embedding of x as,

xp1q
n
“ Embpxnq “
“
01ˆℓ‹
1
eS
xn
01ˆpk‹´1qS
01ˆS
‰T P Rpk‹`1qS`ℓ‹`1
(23)

where ed1
i P RS is the standard basis vector in d1 dimensions. And the relative position encoding for
the keys as,

pp1q,K
i
“

$
’
’
&

’
’
%

“
11ˆℓ‹
0
‰T ,
if i “ 0,
”
eℓ‹
1`log2piq
0
ıT
if i P t1, 2, 4, ¨ ¨ ¨ , k‹{2u

0dˆ1
otherwise.

(24)

And for the value vectors, pV
i “ 0 for all i.

For the first layer and first head, we will describe the value, key and query matrices. Choose,

W p1q
K
“ ?κ
„
1
0
0
0

ȷ
, and,

W p1q
Q
“ ?κ
„
01ˆl‹
1
0
0
0
0

ȷ
.
(25)

15


---Page Break---
Then, observe that for i ě 1,
A
W p1q
K
`
xn´i ` pp1q,K
i
˘
, W p1q
Q xn
E
“ κIpi “ 1q

and for i “ 0,
A
W p1q
K
`
xn ` pp1q,K
0
˘
, W p1q
Q xn
E
“ κ

In particular, letting κ Ñ 8, the attention pattern is,

attp1q
n,n´i “ 1

2Ipi “ 0q ` 1

2Ipi “ 1q.
(26)

Choose the value matrix as,

W p1q
V
“
„
0pℓ‹`Sqˆℓ‹
0
0
2I

ȷ

together with the residual connection, we get,

xp2q
n
“
“ 01ˆℓ‹
1
eS
xn
eS
xn ` eS
xn´1
01ˆpk‹´2qS
01ˆS
‰T
(27)

Layer ℓ` 1.
By induction, assume that the output of the ℓth transformer layer is of the form,

xpℓ`1q
n
“
“ 01ˆℓ‹
1
vn
01ˆpk‹´2ℓqS
01ˆS
‰T
(28)

for some vector vn P R2ℓS. We will show that with appropriately chosen key, query and value vectors
in the pℓ` 1qth layer, the output of this layer is,

xpℓ`2q
n
“
“ 01ˆℓ‹
1
vn
vn ` vn´2ℓ
01ˆpk‹´2ℓ`1qS
01ˆS
‰T
(29)

We will consider the same relative position encodings and query matrix in this layer as in the first
layer (Equations (24) and (25)). Consider a key matrix of the form,

W pℓ`1q
K
“
„
01ˆℓ
?κ
0
0
0
0

ȷ

With this choice, observe that for i ě 1,
A
W pℓ`1q
K
`
xpℓ`1q
n´i
` ppℓ`1q,K
i
˘
, W pℓ`1q
Q
xpℓ`1q
n
E
“ κ ¨ Ipi “ 2ℓq

and for i “ 0,
A
W pℓ`1q
K
`
xpℓ`1q
n
` ppℓ`1q,K
0
˘
, W pℓ`1q
Q
xpℓ`1q
n
E
“ κ

In particular, letting κ Ñ 8, the attention pattern is,

attpℓ`1q
n,n´i “ 1

2Ipi “ 0q ` 1

2Ipi “ 2ℓq.
(30)

Choosing the value matrix as,

W pℓ`1q
V
“
„
0pℓ‹`2ℓSqˆℓ‹
0
0
2I

ȷ
,

we get,

xpℓ`2q
n
“
“ 01ˆℓ‹
1
vn
vn ` vn´2ℓ
01ˆpk‹´2ℓ`1qS
01ˆS
‰T
(31)

16


---Page Break---
Final last transformer layer (ℓ“ ℓ‹).
The output of the second last transformer layer, indexed
ℓ‹ ´ 1 is,

zpℓ‹q
n
fi xpℓ‹q
n
“
”
01ˆℓ‹
1
vpℓ‹´1q
n
vpℓ‹´1q
n
` vpℓ‹´1q
n´2ℓ‹´1
01ˆS
ıT

“
”
01ˆℓ‹
1
vpℓ‹´1q
n
vpℓ‹´1q
n
` vpℓ‹´1q
n´ k‹

2
01ˆS
ıT
,

which follows by plugging in the definition of k‹. Note that there exists a linear transformation Lpℓ‹q
such that,

zpℓ‹´1q
n
fi Lpℓ‹qxpℓ‹q
n
“
”
01ˆℓ‹
1
vpℓ‹´1q
n
vpℓ‹´1q
n´ k‹

2
01ˆS
ıT

This can be further decomposed as,

zpℓ‹´1q
n

“
”
01ˆℓ‹
1
vpℓ‹´2q
n
vpℓ‹´2q
n
` vpℓ‹´2q
n´2ℓ‹´2
vpℓ‹´2q
n´ k‹

2
vpℓ‹´2q
n´ k‹

2
` vpℓ‹´2q
n´ k‹

2 ´2ℓ‹´2
01ˆS
ıT

And yet again there exists a linear transformation Lpℓ‹´1q which transforms this as,

zpℓ‹´2q
n
fi Lpℓ‹´1qzpℓ‹´1q
n

“
”
01ˆℓ‹
1
vpℓ‹´2q
n
vpℓ‹´2q
n´2ℓ‹´2
vpℓ‹´2q
n´ k‹

2 ´2ℓ‹´2
vpℓ‹´2q
n´ k‹

2 ´2ℓ‹´2
01ˆS
ıT

“
”
01ˆℓ‹
1
vpℓ‹´2q
n
vpℓ‹´2q
n´ k‹

4
vpℓ‹´2q
n´ k‹

2
vpℓ‹´2q
n´ 3k‹

4
01ˆS
ıT
(32)

By recursing this argument and composing all the linear transformations, up to a global permutation,
we get that,

ℓ‹
ź

ℓ“1
Lpℓqxpℓ‹q
n
“
”
01ˆℓ‹
1
vp1q
n
vp1q
n´1
¨ ¨ ¨
vp1q
n´pk‹´1q
01ˆS
ıT

“
”
01ˆℓ‹
1
eS
xn
¨ ¨ ¨
eS
xn´pk‹´1q
01ˆS
ıT
(33)

In the final layer, we will right multiply the key, query and value matrices by L‹ “ śℓ‹

ℓ“1 Lpℓq. The
effect can be interpreted as operating the original key, query and value matrices on the embedding
vectors in Equation (33). In the final layer, we will set all the position encodings to be 0 and consider
the key and query matrices,

W pℓ‹q
K
“ ?κ
„
0Skˆpℓ‹`1`Sq
ISkˆSk
0
0
0
0

ȷ

W pℓ‹q
Q
“ ?κ
„
0Skˆpℓ‹`1q
ISkˆSk
0
0
0
0

ȷ
(34)

Then,

A
W pℓ‹q
K
L‹xpℓ‹q
n´i, W pℓ‹q
Q
L‹xpℓ‹q
n
E
“ κ

k´1
ÿ

j“0
Ipxn´j “ xi´1´jq
(35)

Where we must be careful to note that the input xpℓ‹q
n
contains copies of exn, exn´1, ¨ ¨ ¨ , exn´k since
k‹ ě k ` 1 by definition.

Letting κ Ñ 8, if there exists i such that řk´1
j“0 Ipxn´j “ xi´j´1q ą 0, for n ě k, the attention
pattern is,

attpℓ‹q
n,i “
Ipxi´1 “ xn, xi´2 “ xn´1, ¨ ¨ ¨ , xi´k “ xn´k`1q
řn
i“k Ipxi´1 “ xn, xi´2 “ xn´1, ¨ ¨ ¨ , xi´k “ xn´k`1q
(36)

17


---Page Break---
Finally, choose,

W pℓ‹`2q
V
“
„
0pd´Sqˆpℓ‹`1q
0
0
0Sˆpℓ‹`1q
ISˆS
0

ȷ
,
(37)

we get,

xpℓ‹`1q
n
`

n
ÿ

i“k

Ipxi´1 “ xn, xi´2 “ xn´1, ¨ ¨ ¨ , xi´k “ xn´k`1q
řn
i“k Ipxi´1 “ xn, xi´2 “ xn´1, ¨ ¨ ¨ , xi´k “ xn´k`1q

„
0pd´Sqˆ1
exi

ȷ
(38)

Choosing the subsequent linear layer as,

A “
“0Sˆpd´Sq
ISˆS
‰
(39)

b “ 0Sˆ1
(40)

Results in the output,

logitT pxT `1q “

Tÿ

n“k

Ipxn “ xT `1, xn´1 “ xT , xn´2 “ xT ´1, ¨ ¨ ¨ , xn´k “ xT ´k`1q
řT
n“k Ipxn´1 “ xT , xn´2 “ xT ´1, ¨ ¨ ¨ , xn´k “ xT ´k`1q
(41)

which is precisely the in-context conditional k-gram.

C
Proof of Theorem 4

C.1
Modifying the definition of layer normalization

In every layer, we will perform a simple transformation which is to double the hidden dimension d
and add a copy of ´xpℓq
n into the last d coordinates. This is possible by modifying the weights of
the transformer appropriately as discussed below. A consequence of this transformation is that the
feature mean of the xn’s is µn “ 0, and therefore the standard deviation σn simply normalizes by
the L2-norm of the features. In order to avoid having to explicitly state this transformation at each
layer, we will simply redefine the layer norm LN to output v{}v}2 for the input vector v, which is
realized on the first d coordinates of the transformed embeddings.

This transformation can be realized automatically by redefining the initial embeddings Embpxnq,
and modifying the weights of the attention and feedforward subnetworks as follows: The input
embeddings are changed to rEmbpxnq
´EmbpxnqsT P R2d. The key and query matrices are
chosen to be 0 on the last d coordinates in every layer; the value matrix for i ě 1 is transformed
to Blkdiag
`␣
W pℓq
V , W pℓq
V
(˘
, and likewise changing the feedforward layer to the block diagonal

matrices Blkdiag
`␣
W pℓq
1 , W pℓq
1
(˘
and Blkdiag
`␣
W pℓq
2 , W pℓq
2
(˘
. This transformation adds a copy

of ´xpℓq
n into the last d coordinates of the corresponding embeddings.

C.2
Notation and supplementary lemmas

For each i P rTs, define,

vi “ exi´1 ` 3 ¨ exi´2 ` ¨ ¨ ¨ ` 3k´1 ¨ exi´k
(42)

ui “ exi ` 3 ¨ exi´1 ` ¨ ¨ ¨ ` 3k´1 ¨ exi´k`1
(43)

Note that although vi “ ui´1, we make the distinction between the two to avoid any confusion in
what is stored in the embedding vector at time i and at time i ´ 1. Furthermore, define,

In “ tk ` 1 ď i ď n : @j P rks, xi´j “ xn´j`1u.
(44)

Lemma 1. If i P In, zi “ zn´1. However, if i ě k ` 1 but i R In, then,
››
vi
}vi}2 ´
un
}un}2
››
2 ě 3´k.

Let j‹ P t0, 1, ¨ ¨ ¨ , k ´ 1u denote the largest index j such that xn´j ‰ xi´j´1. Consider the
coordinates a “ xn´j‹ P rSs and b “ xi´j‹´1 P rSs. Then,

xvn, eay ´ xui, eay ě 3j ´

j‹´1
ÿ

j“0
3j “ 3j‹

2 ,
(45)

18


---Page Break---
xui, eby ´ xvn, eby ě 3j‹

2
(46)

If }vn}2 ě }ui}2, then,
B ui

}ui}2
, eb

F
´
B vn

}vn}2
, eb

F
ě xui, eby ´ xvn, eby

maxt}ui}2, }vn}2u ě
3j‹

2 ¨ 3k

2
“ 3j‹´k
(47)

This uses the fact that ui and vn are coordinate-wise non-negative. On the other hand, if }vn}2 ď
}ui}2, using a similar analysis,
B ui

}ui}2
, ea

F
´
B vn

}vn}2
, ea

F
ě 3j‹´k.
(48)

In either case, there is a coordinate (a or b) such that, ui{}ui}2 and vn{}vn}2 differ by at least 3j‹´k.
This implies the lower bound on the L2 norm of the difference of the vectors.

C.3
Proof of Theorem 4

Choose the input embeddings as,

xp1q
n
“ Embpxnq “
“
01ˆ3
eS
x
01ˆ5S
‰T P R6S`3
(49)

In the first two layers we will use the same relative position embeddings, in particular,

pp1q,K
i
“ pp2q,K
i
“

$
’
&

’
%

a

logp3q ¨
“
1
0
‰T ,
if i “ 0,

pi ` 1q
a

logp3q ¨
“
0
1
0
‰T ,
if i P t1, 2, ¨ ¨ ¨ , k ´ 1u,

pk ` 1q
a

logp3q ¨
“
0
0
1
0
‰T ,
if i “ k.

,
(50)

and the value embeddings,

pp1q,V
i
“ pp2q,V
i
“

#
3i “
1
0
‰T
for i ď k
0
i ą k.
(51)

In the final layer, we will drop all position-related information and choose pp3q,K
i
“ pp3q,V
i
“ 0 for
all i.

Layer 1.
Consider the key and query matrices,

W p1q
K
“ ?κ ¨
„
11ˆ2
0
0
0

ȷ

W p1q
Q
“ ?κ ¨
„
01ˆ3
11ˆS
0
0
0
0

ȷ
(52)

Then, observe that,
A
W p1q
K
`
Embpxn´iq ` pp1q,K
i
˘
, W p1q
Q Embpxnq
E
“ κpi ` 1q logp3q ¨ Ip0 ď i ď mintn, ku ´ 1q

Letting κ Ñ 8, this results in the attention pattern,

attp1q
n,n´i “ 3iIp0 ď i ď mintn, ku ´ 1q

řmintn,ku´1
i1“0
3i1
(53)

Choose the value matrix as,

W p1q
V
“
„
0pS`3qˆ3
0
0
I

ȷ

The output of the attention layer (with the residual connection) is,

rxp1q
n
“
“
01ˆ3
eS
xn
un
01ˆ3S
‰T , where, un “

mintn,ku´1
ÿ

i“0
attn,n´i eS
xn´i.
(54)

19


---Page Break---
In the feedforward layer to follow, we will choose,

W p1q
1
“ I

W p1q
2
“

«0p3`2Sqˆp3`Sq
0
0
0
ISˆS
0
0
0
0

ff
(55)

Which simply extracts un from rxp1q
n . With the subsequent layer norm and residual connection, the
output of the first layer is,

xp2q
n
“
”
01ˆ3
eS
xn
un
un
}un}2
01ˆ3S
ıT
(56)

Layer 2.
In this layer, the relative position encodings and query matrix are the same as in layer 1
but the key matrix is chosen as,

W p2q
K
“ ?κ
„
0
11ˆ2
0
0
0
0

ȷ
(57)

With this choice, observe that,
A
W p2q
K pxp2q
n´i ` pp1q,K
i
q, W p2q
Q xp2q
n
E
“ κpi ` 1q logp3q ¨ Ip1 ď i ď kq
(58)

As before, since κ Ñ 8, this results in the attention pattern,

attp2q
n,n´i “ 3iIp1 ď i ď mintk, n ´ 1uq

řmintk,n´1u
i1“1
3i1
(59)

which is similar, but subtly different from the attention pattern in the first layer (Equation (53)). The
first layer focuses on indices n ´ i such that 0 ď i ď k ´ 1, while this layer focuses on 1 ď i ď k.
Choosing the value and projection matrices as,

W p2q
V
“

»

—–

I3ˆ3
0
0
03Sˆ3
0
0
0
ISˆS
0
0
0
0

fi

ffifl
(60)

The output of the attention layer (with the first residual connection) is,

rxp2q
n
“
”
Zn
01ˆ2
eS
xn
un
un
}un}2
vn
01ˆ2S
ıT
,

where, vn “

mintk,n´1u
ÿ

i“1
attn,n´i eS
xn´i,

and, Zn “

mintk,n´1u
ÿ

i“1
attn,n´i 3i,

(61)

It is a short calculation to see that Zn “ 3k`1{5 if n ě k ` 1 and otherwise, Zn ď 3k{5. This
will be useful later, since the value of Zn can be used to determine whether n ě k ` 1 or n ď k
which will allow the the next layer to avoid calculating the attention at i ď k, where the evaluation
xn “ xi´1, ¨ ¨ ¨ , xn´k`1 “ xi´k is not well defined. In the subsequent FFN layer, we will choose,

W p2q
1
“ I

W p2q
2
“

«0p3`4Sqˆp3`3Sq
0
0
0
ISˆS
0
0
0
0Sˆ2S

ff
(62)

Which extracts vn from the embedding rxp2q
n . With the layer norm and adding the final residual
connection, the output of this layer is,

xp3q
n
“
”
Zn
02ˆ1
eS
xn
un
un
}un}2
vn
vn
}vn}2
0Sˆ1
ıT
(63)

20


---Page Break---
Layer 3.
In this layer, all the relative position encodings are set as 0 and instead,

W p3q
Q
“
?

2κ

«1
0
0
0
0
0Sˆp2`3Sq
ISˆS
0
0
0
0
0

ff

W p3q
K
“
?

2κ ¨

«1
0
0
0
0
0Sˆp2`4Sq
ISˆS
0
0
0
0
0

ff
(64)

With these choices,
A
W p3q
K xp3q
i , W p3q
Q xp3q
n
E
“ 2κZiZn `
2κxvi, uny
}vi}2 ¨ }un}2

“ 2κZiZn ` 2κ ´ κ
››››
vi
}vi}2
´
un
}un}2

››››

2
(65)

The resulting attention scores are,

attp3q
n,i 9 exp

˜

´κ
››››
vi
}vi}2
´
un
}un}2

››››

2
` 2κZiZn

¸

(66)

Recall that In “ tk ` 1 ď i ď n : @j P rks, xn´j`1 “ xi´ju. Then for any i P In, vi “ un, and
by Lemma 1, for any i ě k ` 1 but not in In,
››››
vi
}vi}2
´
un
}un}2

››››
2
ě 1

3k .

Note that this gap is small but non-zero. Furthermore, recall that Zi “ 3k`1{5 if i ě k and otherwise
Zi ď 3k{5. Thus the attention prefers values of i such that vi “ un and such that i ě k ` 1. In
particular, as κ Ñ 8, the resulting attention pattern is,

attp3q
n,¨ “ UnifpInq.
(67)

Choosing,

W p3q
V
“
„
0
0
0
0Sˆ3
ISˆS
0

ȷ
.

We get that,

rxp3q
n
“ xp3q
n
`

n
ÿ

i“1
attp3q
n,i

„ 0
eS
xi

ȷ
“ xp3q
n
`
1
|In|

ÿ

iPIn

„ 0
eS
xi

ȷ
.

The feedforward layer is chosen to have W p3q
1
“ W p3q
2
“ 0, and the overall output of the final
transformer layer is therefore just rxp3q
n . In the output linear layer, choose,

A “
“0Sˆpd´Sq
ISˆS
‰

b “ 0
(68)

which results in,

logitn “
1
|In|

ÿ

iPIn
eS
xi “

n
ÿ

i“k`1

Ip@1 ď j ď k, xi´j “ xn´j`1q
řn
i1“k`1 Ip@1 ď j ď k, xi1´j “ xn´j`1q ¨ exi

In particular,

logitT pxT `1q “

řT
n“k`1 Ip@0 ď i ď k, xn´i “ xT ´i`1q
řT
n“k`1 Ip@1 ď i ď k, xn´i “ xT ´i`1q
(69)

which is the conditional k-gram.

21


---Page Break---
D
Representation lower bounds for 1-layer transformers: Proof of Theorem 5

We prove this lower bound by a reduction to communication complexity, and specifically to the set
disjointness problem.

Suppose Alice and Bob are given strings a, b P t0, 1un which are indicator vectors of sets A and B.
Their goal is to jointly compute DISpa, bq “ IpDi : ai “ bi “ 1q, which indicates whether A and B
intersect or not. Alice and Bob may send a single bit message to the other party over a sequence of
communication rounds. The following seminal result by [29] asserts a lower bound on amount of
communication required between Alice and Bob to carry out this task.
Theorem 7 ([29]). Any deterministic protocol for computing DISpa, bq requires at least n rounds of
communication.

We show that a 1-layer transformer with sufficiently small embedding dimension / number of heads
can be used to simulate a two-way communication protocol between Alice and Bob to solve DISpa, bq
in a way which contradicts Yao’s lower bound in Theorem 7.

With m “ T{3´1, suppose Alice and Bob have length m bit strings a, b P t0, 1um. The transformer’s
input will be a sequence of the form,

2, a1, b1, 2, a2, b2, ¨ ¨ ¨ , 2, am, bm, 2, 1,
(70)

of length 3m ` 2 “ T ´ 1. The input basically contains a repeating motif, composed of the symbol
2 followed by one of Alice’s bits, and then one of Bob’s bits. The last 2 symbols are 2 and 1. We
will consider the empirical conditional 3-gram probability the transformer associates with the symbol
xT “ 2. Noting that xT ´1 “ 1 and xT ´2 “ 1, the conditional 3-gram is computed to be,
řT ´1
i“3 Ipxi “ 1, xi´1 “ 1, xi´2 “ 2q

řT ´1
i“3 Ipxi´1 “ 1, xi´2 “ 2q
(71)

Note that if xi´2 “ 2, then i must be of the form 3j for j “ 1, ¨ ¨ ¨ , n, and we may rewrite the sum
as,
řm
j“1 Ipx3j “ 1, x3j´1 “ 1q
řm
j“1 Ipx3j´1 “ 1q
“ |A X B|

|B|
(72)

Now, let us use the transformer to construct a deterministic communication protocol between Alice and
Bob. Alice is given px2, x5, ¨ ¨ ¨ , x3m´1q “ pa1, a2, ¨ ¨ ¨ , amq and Bob is given px3, x6, ¨ ¨ ¨ , x3mq “
pb1, b2, ¨ ¨ ¨ , bmq.

In the first round, Alice computes the normalization in the softmax of the attention which comes from
the set of inputs she holds. For simplifying notation define,

scorephqpiq “ exp
´A
W phq
K pEmbpxiq ` piq, W phq
Q EmbpxT ´1q
E¯
(73)

In particular, for each head h P rHs, she computes,

Zphq
Alice “ log
´ÿm

j“1 scorephqp3j ´ 1q
¯
(74)

Assuming that the transformer uses p bits of precision, Alice communicates Zphq
Alice for each h, which
corresponds to pH bits of communication. With this information, Bob completes the rest of the
normalization term (again up to p bits of precision) and computes,

Zphq “ log
´
Zphq
Alice ` Zphq
Bob ` Zphq
common
¯
,
(75)

where Zphq
Bob “ log
´ÿm

j“1 scorephqp3jq
¯
(76)

and Zphq
common “ log
´ÿm

j“1 scorephqp3j ´ 2q ` scorephqpT ´ 2q ` scorephqpT ´ 1q
¯
(77)

which is the overall normalization term in the softmax. This is communicated back to Alice, using
another pH bits of communication. Next using this information, Alice computes the output of the

22


---Page Break---
attention layer, taking the convex combination corresponding to the inputs she knows. In particular,
for each h P rHs she computes,

n
ÿ

j“1

scorephqp3j ´ 1q

exppZphqq
Embpx3j´1q P Rd.
(78)

across all the heads. Rather than transmitting everything, she concatenates the outputs of the heads,
and multiplies them by the value and projection matrices to result in the output yAlice which is
d-dimensional. This is sent to Bob using dp bits of communication. Subsequently, Bob computes the
terms in the attention corresponding to the inputs he knows as well as the public inputs (all the 2’s at
positions 3j ´ 2 as well as the last two symbols). In particular,

m
ÿ

j“1

scorephqp3jq

exppZphqq Embpx3jq `

m`1
ÿ

j“1

scorephqp3j ´ 2q

exppZphqq
Embp2q ` scorephqpT ´ 1q

exppZphqq
Embp1q
(79)

These are yet again concatenated across all the heads and multiplied by the value and projection
matrices to result in the output yBob which is added to yAlice to result in y. Bob passes y through
the residual connection, layer norm, and feedforward layers, and subsequently through the linear
layer and softmax of the model to result in the output of the model. By assumption, the output of the
model approximately captures the conditional 3-gram, which by Equation (72) equals |A X B|{|B|.
Note that if |A X B|{|B| is non-zero, it must be at least 1{T. This means, if the transformer is able to
compute the conditional 3-gram to within an additive error of 1{3T, then Bob can simply threshold
the output of the transformer to decide whether A X B “ H or not, thereby solving DISpa, bq.

Since this communication protocol is deterministic, by Yao’s lower bound in Theorem 7, the number
of bits communicated between Alice and Bob must be at least m “ T{3 ´ 1. The total number
of bits of communication in the protocol is 2pH ` dp ` 1 (the last 1 comes from Bob having to
communicate the answer to Alice), completing the proof.

E
Lower bounds on representing kth-order induction heads: Proof of
Theorem 6

In this section we prove the size-lower bound on attention-only transformers representing kth-order
induction heads in Theorem 6. To enable this result to be better interpreted, we will break it down
into two corollaries.
Corollary 1. Consider an L-layer attention-only transformer with 1 head per layer and relative
position encodings, which satisfies Assumption 1. If L ď 1 ` log2pk ´ 2q, the attention pattern in
layer L of the transformer cannot represent a kth-order induction head.
Corollary 2. Consider an 2-layer attention-only transformer with H heads in the first layer and
relative position encodings, and assume that Assumption 1 is satisfied. If H ď k ´ 3, the attention
pattern in the 2nd layer cannot represent a kth-order induction head.

We will first prove the result for the case L “ 2 and H “ 1, which falls in the intersection of both
of these corollaries. We will show that these models cannot represent kth-order induction heads for
k ą 3, under Assumption 1. We subsequently extend it to the general L-layer transformer (i.e.,
Corollary 1) in Appendix E.2 and to the general case with Hℓheads in layer ℓP rLs in Appendix E.3.

E.1
Lower bounds on 2-layer 1-head attention-only transformers

In this section we show that under Assumption 1, a 2-layer 1-head attention-only transformer cannot
represent kth-order induction heads for any k ě 4. We will prove lower bounds on the transformer
when the input is binary, i.e., S “ t0, 1u. With relative position embeddings, observe that the first
layer of the transformer model learns representations of the form,

xp2q
n
“ Embpxnq `
ÿ

iďn
attp1q
n,i W p1q
V Embpxiq `
ÿ

iďn
W p1q
V pV,p1q
n´i
(80)

where note that the attention pattern only depends on n and i and not on xi or xn. These representa-
tions are input into the second layer, which realizes the attention pattern attp2q
n,i, which is proportional

23


---Page Break---
to,

exp
´A
W p2q
K
`
xp2q
i
` pK,p2q
n´i
˘
, W p2q
Q xp2q
n
E¯
.
(81)

We need this function to be maximized uniquely when xi´1 “ xn, ¨ ¨ ¨ , xi´k “ xn´k`1. Denoting
ϕp0q “ W p1q
V Embp0q and ϕp1q “ W p1q
V Embp1q,

xp2q
n
“ Embpxnq `
ÿ

iďn
attp1q
n,i W p1q
V Embpxiq `
ÿ

iďn
W p1q
V pp1q,V
n´i
(82)

“ xnEmbp1q ` p1 ´ xnqEmbp0q `
ÿ

iďn
attp1q
n,i
`
xi ¨ ϕp1q ` p1 ´ xiq ¨ ϕp0q
˘
`
ÿ

iďn
W p1q
V pp1q,V
n´i

(83)

“
ˆEmbp1q ` Embp0q

2
` x1
n ¨ Embp1q ´ Embp0q

2

˙
`
ÿ

iďn
attp1q
n,i

ˆϕp1q ` ϕp0q

2
` x1
i ¨ ϕp1q ´ ϕp0q

2

˙

`
ÿ

iďn
W p1q
V pp1q,V
n´i
(84)

where x1
i Ð 2xi ´ 1. We can write this down as,

xp2q
n
“ mp1q
n
` M p1q
n
rx1
n
x1
n´1
¨ ¨ ¨
x1
1sT
(85)

where M p1q
n
is a matrix of rank at most 2 and of the form,

M p1q
n
“
ˆϕp1q ´ ϕp0q

2

˙ ”
attp1q
n,n
¨ ¨ ¨
attp1q
n,1
ı
`
ˆEmbp1q ´ Embp0q

2

˙
r1
0
¨ ¨ ¨
0s
(86)

which is independent of x1
1, ¨ ¨ ¨ , x1
n. Likewise mp1q
n
collects all the vectors in the sum that don’t
depend on x1
1, ¨ ¨ ¨ , x1
n. Now, observe that in the next layer, we wish to show that an induction head
cannot be realized by attp2q
n,i for each i ď n. We will show this for any value of i ď n ´ k.

In the second layer, we may write down the key vectors as,

W p2q
K
´
xp2q
i
` pp2q,K
n´i
¯
“ W p2q
K mp1q
i
` W p2q
K M p1q
i
rx1
i
x1
i´1
¨ ¨ ¨
x1
1sT ` W p2q
K pp2q,K
n´i . (87)

Again, defining the vector mp1q
i
and the matrix M
p1q
i
appropriately (having rank at most 2), this
equals,

mp1q
i ptx1
iu Y tx1
´k´1, ¨ ¨ ¨ , x1
1uq ` M
p1q
i y
(88)

where y fi
“x1
i´1
¨ ¨ ¨
x1
i´k
‰T and the vector mp1q
i
depends on x1
i as well as the inputs
x1
i´k´1, ¨ ¨ ¨ , x1
1, which in this context, are treated as nuisance variables since they do not inter-
sect with tx1
i´1, ¨ ¨ ¨ , x1
i´ku Y txn, ¨ ¨ ¨ , xn´k`1u. Henceforth we will avoid explicitly stating the

dependency of mp1q
i
on the xj’s. Similarly, the query vector can be written down as,

W p2q
Q xp2q
n
“ x
mp1q
n
` |
M p1q
n x ` x
M p1q
n y
(89)

where x
mp1q
n , |
M p1q
n
and x
M p1q
n
are defined appropriately, with x
M p1q
n
and x
M p1q
n
of rank at most 2, and
x is defined as
“x1
n
¨ ¨ ¨
x1
n´k`1
‰T . For an appropriate matrix M ˆ
n,i, vectors mˆ
n,i and Ă
mˆ
n,i and
scalar mˆ
n,i, the dot-product of the key and query vectors can be written as,
A
W p2q
K
`
xp2q
i
` pp2q,K
n´i
˘
, W p2q
Q xp2q
n
E

“ xT M ˆ
n,iy ` yT `

M
ˆ
n,i
˘
y ` pmˆ
n,iqT x ` pmˆ
n,iqT y ` mˆ
n,i fi fn,ipx, yq,
(90)

Which is a linear function in x and quadratic in y, both of which lie on t˘1uk. Note that the matrix

M ˆ
n,i has rank at most 2 since it is a product of M
p1q
i
and |
M p1q
n , each with rank at most 2. Next we
introduce a lemma showing that if M ˆ
n,i is inherently low rank, the quadratic form in Equation (90)
which captures the dot-product between the key and value vectors cannot satisfy the property that
for every y, the function is uniquely maximized at x “ y. In particular, this means that for any
i ď n ´ k, there is some choice of xn, xn´1, ¨ ¨ ¨ , xn´k`1 such that there are xi´1, ¨ ¨ ¨ , xi´k such
that for at least one j P rks, xi´j and xn´j´1 are not equal, but the attention score is larger than the
case when xi´j were equal to xn´j´1 for each j P rks.

24


---Page Break---
Lemma 2. If M ˆ
n,i has rank ď k ´ 2, it is impossible for fn,ipx, yq to satisfy the property that for
every y P t˘1uk, the maximizer is uniquely x “ y.

The proof is almost complete: if k ě 4, then the rank of M ˆ
n,i, which is at most 2, does not exceed
k ´ 2. This means that when k ě 4, any attention pattern realized in the second layer must satisfy
the property that there exists a string such that the attention is no longer uniquely maximized when
xn “ xi´1, ¨ ¨ ¨ , xn´k`1 “ xi´k.

Proof. For the purpose of brevity, define Hk “ t˘1uk. First consider the reparameterization,

rx “ Ă
M ˆ
n,ix, where Ă
M ˆ
n,i “
„pM ˆ
n,iqT

pmˆ
n,iqT

ȷ
.
(91)

Then, the dot-product of the key and query matrices can be written as,
“
yT
1
‰
rx ` yT `

M
ˆ
n,i
˘
y ` pmˆ
n,iqT y ` mˆ
n,i
(92)

Note that this function is linear in rx and therefore must be maximized on a vertex of the convex hull
of the domain, Ă
M ˆ
n,iHk fi
␣Ă
M ˆ
n,ih : h P Hk
(
. If M ˆ
n,i has rank at most k ´ 2, the rank of Ă
M ˆ
n,i is
at most k ´ 1 and cannot be full rank. We show that this must imply that there is a vertex v P Hk
such that Ă
M ˆ
n,iv is not a unique vertex of the convex hull of Ă
M ˆ
n,iHk. This means that v cannot be a
unique maximizer for rx when maximizing over all strings in Equation (92), and specifically y “ v is
a witness to Lemma 2.

Below we discuss how to find such a vector v. Note that Ă
M ˆ
n,i is not full rank, which implies that

there exists a vector n such that Ă
M ˆ
n,in “ 0. Without loss of generality, let n1 be the smallest
non-zero coordinate of n in absolute value. Then the vector n´1
1 n has no non-zero coordinates in
the interval p´1, 1q. We will show that signpn´1
1 nq is a good choice for v.

Consider two cases,

Case I.
Every non-zero coordinate of n´1
1 n is in t˘1u. Consider any x P Hk which matches with
n on the non-zero coordinates. Consider x1 which is the same as x, except a negation is taken on
the coordinates where n is non-zero. Note that Ă
M ˆ
n,ix “ Ă
M ˆ
n,ix1, for the same value of x. This
means that for any y. In particular, from Equation (92), both x and x1 are maximizers, showing that
Lemma 2 is true in this case. We circumvent having to find such a vector v in this case.

Case II.
n´1
1 n has non-zero coordinates which are not all in t˘1u. In particular, at least one
coordinate where this vector is strictly less than ´1 or strictly greater than `1. In this case, observe
that the sign vector rn “ signpn´1
1 nq P Hk lies within, but is not a vertex of the convex hull of
the set Hk Y tn´1
1 nu. The reason for this is simple to see when we assume that n´1
1 n has only
one coordinate which is not in r´1, 1s, say, the coordinate j “ 2: here, rn can be written down as a
convex combination (with non-zero coefficients) of n´1
1 n and rnp2q; the latter vector is obtained by
flipping coordinate 2 of rn. When there is more than one coordinate not in r´1, 1s, we can peel away
these large coordinates in n´1
1 n by taking a convex combination of this vector with the vectors rnpjq

for the appropriate values of j, to return the sign vector rn. Here, rnpjq is the version of rn where the
jth-coordinate is flipped. This results in the following claim.

Claim 1. The sign vector rn lies within the convex hull of the points Hk Ytn´1
1 nu, but is not a vertex
of this set.

In particular, we may write,

rn “ α0n´1
1 n `
ÿ

jPrns
αj rnpjq.
(93)

where α0 ą 0 and řn
j“0 αi “ 1. By left-multiplying this on both sides by Ă
M ˆ
n,i and noting that n
lies in the null-space of this matrix, we get,
Ă
M ˆ
n,irn “
ÿ

jPrns
αj Ă
M ˆ
n,irnpjq
(94)

25


---Page Break---
where note that ř
jPrns αj is strictly less than 1, since α0 ą 0. We may write this vector as,

Ă
M ˆ
n,irn “ α00 `
ÿ

jPrns
αj Ă
M ˆ
n,irnpjq

“ α0

2k
ÿ

hPHk

Ă
M ˆ
n,ih `
ÿ

jPrns
αj Ă
M ˆ
n,irnpjq
(95)

Since α0 ą 0, this equation implies that the image of rn under Ă
M ˆ
n,i itself falls within convp Ă
M ˆ
n,iHkq,
but is itself not a vertex of this set. This means that rn can never be a maximizer of fn,i
`
¨, y
˘
for any
y, and in particular when y “ rn, thereby proving Lemma 2.

E.2
L-layer attention-only transformers with 1 head per layer: Proof of Corollary 1

Proof. The proof largely tracks the 2-layer case, with the main exception that we keep track of how
the maximum possible rank of the matrix M ˆ
n,i grows as a function of the depth of the transformer.
In the case the 2-layer transformer, we show that it cannot exceed 2. With the addition of more layers,
we show that it cannot exceed 2L´1.

Recall from the notation in Equation (85) that the output of the first attention layer is,

xp2q
n
“ mp1q
n
` M p1q
n
rx1
n
x1
n´1
¨ ¨ ¨
x1
1sT
(96)

where M p1q
n
P Rdˆn has rank at most 2. Let us rewrite this as,

xp2q
n
“ mp1q
n
` M
p1q
n
“x1
T
x1
T ´1
¨ ¨ ¨
x1
1
‰T
(97)

where M p1q
n
P RdˆT is causally masked to be 0’s when it operates on xi for all indices i ą n. Note
that even with this causal masking, M
p1q
n
has rank at most 2, as discussed in Equation (85).

By induction, assume that the output of the pℓ´ 1qth attention layer is of the form,

xpℓq
n “ mpℓ´1q
n
` M
pℓ´1q
n
x1:T
(98)

where x1:T fi
“x1
T
x1
T ´1
¨ ¨ ¨
x1
1
‰T . Passing xpℓq
n through the ℓth attention layer, we get,

xpℓ`1q
n
“ xpℓq
n `
ÿ

iďn
attpℓq
n,i W pℓq
V
´
xpℓq
i
` ppℓq,V
n´i
¯
(99)

“ mpℓ´1q
n
` M
pℓ´1q
n
x1:T `
ÿ

iďn
attpℓq
n,i W pℓq
V mpℓ´1q
i
`
ÿ

iďn
attpℓq
n,i W pℓq
V M
pℓ´1q
i
x1:T

`
ÿ

iďn
attpℓq
n,i W pℓq
V ppℓq,V
n´i
(100)

Define,

mpℓq
n “ mpℓ´1q
n
`
ÿ

iďn
attpℓq
n,i W pℓq
V mpℓ´1q
i
`
ÿ

iďn
attpℓq
n,i W pℓq
V ppℓq,V
n´i , and,
(101)

M
pℓq
n “ M
pℓ´1q
n
`
ÿ

iďn
attpℓq
n,i W pℓq
V M
pℓ´1q
i
(102)

Then, we can write down,

xpℓ`1q
n
“ mpℓq
n ` M
pℓq
n x1:T
(103)

We also inductively assume that for every i ď n,

piq M
pℓ´1q
i
has rank R ď 2ℓ´1, and,

piiq M
pℓ´1q
i
can be factorized in the form řR
r“1 ur ¨ vT
i,r, where only the vi,r’s depend on i, but
the ur’s do not depend on i.

26


---Page Break---
Both of these conditions are true when ℓ´ 1 “ 1 as evidenced by the structure of M p1q
i
in Equa-

tion (86) and noting that M
p1q
i
is obtained from M p1q
i
by right multiplying by a diagonal mask
matrix. Using the recursion in Equation (101), we prove that the induction hypotheses piq and piiq
are true at layer ℓas well. In particular using the decomposition in piiq, observe that,

M
pℓq
n “

R
ÿ

r“1
ur ¨ vT
n,r `
ÿ

iďn
attpℓq
n,i W pℓq
V

R
ÿ

r“1
ur ¨ vT
i,r
(104)

“

R
ÿ

r“1
ur ¨ vT
n,r `

R
ÿ

r“1
W pℓq
V ur ¨
´ÿ

iďn attpℓq
n,i vi,r
¯T
(105)

“

2R
ÿ

r“1
ur ¨ vT
n,r
(106)

where for r1 P rRs, uR`r1 fi W pℓq
V ur and vn,r1 fi ř

iďn attpℓq
n,i vi,r. Since M pℓq
n
is the sum of
2R rank 1 matrices and therefore has rank at most 2R ď 2ℓ, proving both parts of the induction
hypothesis.

By induction, at the end of the pL ´ 1qth layer, we have an output which looks like,

xpLq
n
“ mpL´1q
n
` M
pL´1q
n
x1:T
(107)

where M pL´1q
n
has rank at most 2L´1. More importantly, note that by the causal masking, even
though it appears to depend on the whole input sequence through x1:T , note that xpLq
n
only depends
on x1, ¨ ¨ ¨ , xn and not on the future inputs to this time n. In particular, by a similar argument as
in the 2-layer case (cf. Equation (85) to Equation (90)), for any i ď n ´ k we can decompose the
dot-product of the key and query vectors at the Lth layer as a bilinear form which looks like,
A
W pLq
K
`
xpLq
i
` ppLq,K
n´i
˘
, W pLq
Q xpLq
n
E

“ xT M ˆ
n,iy ` yT `

M
ˆ
n,i
˘
y ` pmˆ
n,iqT x ` pmˆ
n,iqT y ` mˆ
n,i fi f pL`1q
n,i
px, yq (108)

where x and y are defined as
“x1
n
¨ ¨ ¨
x1
n´k`1
‰T and
“x1
i´1
¨ ¨ ¨
x1
i´k
‰T respectively, and

M ˆ
n,i has rank at most that of M pL´1q
n
, which is 2L´1. In particular, if 2L´1 ď k ´ 2, by Lemma 2
the proof concludes.

E.3
The general case: Transformers with Hℓheads in layer ℓ: Proof of Theorem 6

The hth head of the first layer of the attention-only transformer learns patterns of the form,

rxp1,hq
n
“
ÿ

iďn
attp1,hq
n,i
W p1,hq
V
Embpxiq `
ÿ

iďn
W p1,hq
V
pV,p1,hq
n´i
(109)

“
ÿ

iďn
attp1,hq
n,i

ˆϕhp0q ` ϕhp1q

2
` x1
i ¨ ϕhp1q ´ ϕhp0q

2

˙
`
ÿ

iďn
W p1,hq
V
pV,p1,hq
n´i
(110)

where the last equation assumes a binary input sequence, defines x1
i “ 2xi ´ 1 and uses the notation
ϕhp0q “ W p1,hq
V
Embp0q and ϕhp1q “ W p1,hq
V
Embp1q. We can further rewrite this as,

rxp1,hq
n
“ mp1,hq
n
` M p1,hq
n
x1:T
(111)

where each M p1,hq
n
P RdˆT is rank 1 and applies a causal mask on the inputs xi for i ą n. Recall
that the output of the first attention layer applies a projection matrix on the concatentation of rxp1,hq
n
across h P rH1s and then adds a residual connection. The output can be written down as,

rxp2q
n
“ Embpxnq ` W p1q
O

»

—–

mp1,1q
n
...
mp1,H1q
n

fi

ffifl ` W p1q
O

»

—–

M p1,1q
n
...
M p1,H1q
n

fi

ffifl x1:T
(112)

27


---Page Break---
“ mp1q
n
` M p1q
n x1:T ,
(113)

where,

M p1q
n
“
ˆEmbp1q ` Embp0q

2

˙
eT
n ` W p1q
O

»

—–

M p1,1q
n
...
M p1,H1q
n

fi

ffifl , and,
(114)

mp1q
n
“
ˆEmbp1q ´ Embp0q

2

˙
` W p1q
O

»

—–

mp1,1q
n
...
mp1,H1q
n

fi

ffifl
(115)

Notice that the rank of the matrix M p1q
n
is at most H1`1. This is because the concatenation operation
can increase the rank at most additively, and since each of the M p1,hq
n
matrices are rank at most 1.

Following through the proof in Appendix E.2 for the L-layer case, we can prove inductively that at
any layer ℓ, the output looks like,

xpℓq
n “ mpℓq
n ` M pℓq
n x1:T
(116)

where the rank of M pℓq
n
is śℓ
i“1pHi ` 1q. Invoking Lemma 2, if śL´1
i“1 pHi ` 1q ď k ´ 2, the
attention-only transformer cannot realize a kth-order induction head at layer L.

F
Model architecture and hyper-parameters

The experiments were run on one 8 ˆ A100 GPU node.

Parameter
Matrix shape

transformer.wte
2 ˆ d
transformer.wpe
N ˆ d
transformer.h.ln_1 pˆℓq
d ˆ 1
transformer.h.attn.c_attn pˆℓq
3d ˆ d
transformer.h.attn.c_proj pˆℓq
d ˆ d
transformer.h.ln_2 pˆℓq
d ˆ 1
transformer.h.mlp.c_fc pˆℓq
4d ˆ d
transformer.h.mlp.c_proj pˆℓq
d ˆ 4d
transformer.ln_f
d ˆ 1

Table 2: Parameters in the transformer architecture with their shape.

G
Additional experimental results

Assumption 1 suggests that the attention patterns attpℓq
n,i in layers ℓ“ 1, 2, ¨ ¨ ¨ , L ´ 1, as learnt by an
L-layer attention-only transformers may only be a function of only the position indices n, i. In this
section we run some additional experiments to test this conjecture. We train a 2 layer attention-only
transformer with k heads in the first layer, on data drawn from a randomly sampled kth-order Markov
process, and focus on the learnt attention patterns as a function of in the input sequence. Figure 7
plots the results of this experiment for k “ 2 and Figure 8 for k “ 3. While in both cases there is
some variance in the attention patterns learnt by the transformer in some of the heads, we believe
that this is a consequence of the iteration budget of the transformer, and specifically the fact that
even if the test loss appears to have converged, the transformer may still continue changing in the
parameter space. Furthermore, when the attention patterns have some non-zero but small variance as
a function of the input, a relaxation of Assumption 1, we also believe that the results we proved in
Corollaries 1 and 2 and Theorem 6 should carry over approximately and leave this as an interesting
question for future work. Conditional lower bounds of this nature, reliant on structural assumptions
the transformer appears to demonstrate in practice are an interesting area of future research.

28


---Page Break---
Dataset
k-th order binary Markov source
Architecture
Based on the GPT-2 architecture as implemented in [30]

Batch size
Grid-searched in t8, 16u
Accumulation steps
1

Optimizer
AdamW (β1 “ 0.9, β2 “ 0.95)
Learning rate
0.001
Scheduler
Cosine
# Iterations
Up to 25000
Weight decay
1 ˆ 10´3

Dropout
0
Sequence length
Grid-searched in t32, 64, 128, 256, 512, 1024u
Embedding dimension
Grid-searched in t16, 32, 64u
Transformer layers
Between 1 and 8
Attention heads
Up to k

Repetitions
3

Table 3: Settings and parameters for the transformer model used in the experiments.

0
2
4
6
8
10
0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

Sequence index

Attention weight

(a) First head

0
2
4
6
8
10

0.00

0.05

0.10

0.15

0.20

0.25

0.30

Sequence index

(b) Second head

Figure 7: Mean attention for column n “ 10 of the two heads of the first attention layer, for a 2-layer
2-head transformer model trained on an order-3 Markov process, averaged across 100 input sequences
of length 128.

0
2
4
6
8
10
0.00

0.05

0.10

0.15

0.20

0.25

Sequence index

Attention weight

(a) First head

0
2
4
6
8
10
0.00

0.02

0.04

0.06

0.08

0.10

0.12

0.14

Sequence index

(b) Second head

0
2
4
6
8
10
0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

0.40

Sequence index

(c) Third head

Figure 8: Mean attention for column n “ 10 of the three heads of the first attention layer, for a
2-layer 3-head transformer model trained on an order-3 Markov process, averaged across 100 input
sequences of length 128.

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The paper presents experimental evidence about kth-order Markov processes
being effectively learnable by small-depth transformers. We also prove theorems about the
representation power of low-depth transformers, which are stated in the main paper.
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
Justification: We state that our lower bound in Corollary 1 is a conditional result in Sec-
tion 6.1, and that it is an important open problem to see whether the conditional statements
can be removed. While our work talks about the representational power of transformers, we
also mention that learning dynamics of gradient descent is an important direction that future
work needs to address.
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

30


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: All theorems in the paper are stated and cross-referenced with proofs in the
appendix.

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

Justification: Experiment hyperparameters are presented in Tables 2 and 3 and the code has
been provided along with the supplementary material.

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

31


---Page Break---
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: All the code has been open sourced; hyperparameter choices have been
provided in Tables 2 and 3.
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
Justification: Tables 2 and 3 cover this information.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: Standard error bars are provided on all plots.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

32


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
Justification: Details in Appendix F.
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
Justification: No NeurIPS code of ethics were violated.
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
Justification: This work is a primarily theoretical study on the behavior of tokenization on
toy problems (learning Markov chains). The societal impact of this research is not likely to
be significant.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.

33


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

Answer: [NA]

Justification: No models with a high risk for misuse were trained or released.

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

Justification: Code has been properly credited, via citing the relevant papers.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

34


---Page Break---
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

Justification: No new assets released.

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

Justification: No crowdsourcing or research with human subjects.

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

Justification: No crowdsourcing or research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

35


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

36


---Page Break---
