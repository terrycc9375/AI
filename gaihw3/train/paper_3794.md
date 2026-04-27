ALPINE: Unveiling The Planning Capability of
Autoregressive Learning in Language Models

Siwei Wang1∗Yifei Shen1∗Shi Feng2 Haoran Sun3 Shang-Hua Teng4† Wei Chen1
1Microsoft Research Asia ({siweiwang, yifeishen, weic}@microsoft.com)
2Harvard University (shifeng@fas.harvard.edu)
3Peking University (sunhaoran0301@stu.pku.edu.cn)
4University of Southern California (shanghua@usc.edu)

Abstract

Planning is a crucial element of both human intelligence and contemporary large
language models (LLMs). In this paper, we initiate a theoretical investigation
into the emergence of planning capabilities in Transformer-based LLMs via their
next-word prediction mechanisms. We model planning as a network path-finding
task, where the objective is to generate a valid path from a specified source node
to a designated target node. Our mathematical characterization shows that Trans-
former architectures can execute path-finding by embedding the adjacency and
reachability matrices within their weights. Furthermore, our theoretical analysis
of gradient-based learning dynamics reveals that LLMs can learn both the adja-
cency and a limited form of the reachability matrices. These theoretical insights
are then validated through experiments, which demonstrate that Transformer ar-
chitectures indeed learn the adjacency and an incomplete reachability matrices,
consistent with our theoretical predictions. When applying our methodology to the
real-world planning benchmark Blocksworld, our observations remain consistent.
Additionally, our analyses uncover a fundamental limitation of current Transformer
architectures in path-finding: these architectures cannot identify reachability re-
lationships through transitivity, which leads to failures in generating paths when
concatenation is required. These findings provide new insights into how the in-
ternal mechanisms of autoregressive learning facilitate intelligent planning and
deepen our understanding of how future LLMs might achieve more advanced and
general planning-and-reasoning capabilities across diverse applications.

1
Introduction

Large language models (LLMs) such as ChatGPT have impressed many with their powerful capabili-
ties across a wide range of tasks, including language processing, knowledge extraction, reasoning,
planning, coding, tool use, and more [1]. However, we continue to be intrigued by the underlying
mechanisms that fuel the power of LLMs. While all current LLMs are built on the Transformer
architecture, which uses autoregressive learning to predict the next word in a language sequence, the
overarching question remains:

Why does the process of next-word prediction give rise to intelligence?

There is no definite answer to this question yet, but researchers are approaching the problem from
various angles, aiming to characterize the power and limitations of LLMs, as well as to capture
their underlying acquisition, abstraction, generalization, adaptation, and reasoning mechanisms.

∗denotes equal contributions. Corresponding author: Wei Chen (weic@microsoft.com)
†Supported by a Simons Foundation Investigator Award.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Recently, the mechanisms of grammar learning, knowledge manipulation, scaling laws, and arithmetic
operations have been empirically uncovered [4, 3, 5, 2, 31, 11]. Furthermore, theoretical analyses have
been conducted on in-context learning, chain-of-thought, and other forms of reasoning [30, 8, 7, 27].
Beyond these, LLMs’ capability for planning—a fundamental component of human intelligence—has
also drawn considerable attention. Planning is involved in nearly every aspect of our daily life, such as
organizing a task at work, planning a trip, or seeking a mathematical proof of a theorem. Additionally,
task planning plays a pivotal role in state-of-the-art LLM-empowered autonomous agents, such as
HuggingGPT [18], Voyager [23], and Reflection [19]. Understanding how LLMs complete a planning
task can shed light on how the seemingly low-level statistical task of next-word prediction transforms
into a high-level intelligent process. Several prior studies have empirically evaluated the planning
capabilities of LLMs, yielding both positive and negative results [16, 22]. However, the current
results are incomplete and do not fully explain why LLMs can or cannot successfully accomplish
specific planning tasks (see Appendix A for a detailed discussion of related works).

Given that planning often involves making sequential selections of next steps to achieve a desired
goal, it naturally relates to the path-finding task in networks. For example, autonomous agents (e.g.,
HuggingGPT [18]) for scheduling API calls can be likened to finding a call path in the API call graph;
a mathematical proof can be seen as a proof path from the axioms to the final theorem [21]; and a
step-by-step solution to a grade-school math problem can be viewed as a path in the dependency
graph among the variables [28, 29]. Many previous studies on LLM planning capabilities are related
to path-finding. e.g., an LLM planning benchmark called Blocksworld [22] can be viewed as path-
finding from the initial state of the blocks to the final state in a state transition graph. Furthermore, in
neuroscience, planning is often evaluated through path-finding in a maze [26]. Consequently, in this
paper, we abstract planning in LLM learning as the following path-finding task: given an underlying
directed graph, a Transformer architecture is provided with training data consisting of a collection of
paths that specify the source node, the target node, and a path from the source to the target. The task
of the language model is then to generate a path for a new source-target pair. In addition to measuring
the performance of the trained model, we examine the internal weighting mechanism and the learning
dynamics of the Transformer architecture during the learning and problem-solving process. This
research is part of our broader project, ALPINE (Autoregressive Learning for Planning In NEtworks),
which aims to answer the overarching question on the connection between the process of next-word
prediction and the emergence of intelligence through the lens of planning.

Our Contributions: Our project initiates a theoretical investigation into the development of planning
capabilities in Transformer-based language models by focusing on characterizing their expressiveness
and learning dynamics in the path-finding task. First, in Theorem 2, we present a mathematical
construction of a Transformer that encodes both the adjacency and reachability matrices of the
network, thereby establishing that Transformer architectures possess the expressive capacity to
complete the path-finding task. Then, in Theorem 3, we prove that when applying gradient descent
to minimize the cross-entropy loss on the training data, a model based on a simplified Transformer
architecture can extract the adjacency matrix and a limited form of the reachability matrix, using
them to mimic human-like intelligence in path-finding. Our theoretical analysis further reveals
a fundamental limitation of current Transformer architectures: they do not learn certain types of
reachability, particularly transitive reachability, resulting in an incomplete ability to reason about
future steps when planning. To validate our theoretical findings, we conduct extensive experiments
training Transformers on the path language using autoregressive learning. First, these experiments
demonstrate that Transformers achieve high accuracy in the path-finding task (Figure 3). Second, we
show that it is indeed possible to extract both the adjacency and a limited form of the reachability
matrices from the Transformers’ weights (Figures 1,2,5,6(a)). Third, we observe a significant drop
in test accuracy when the source and target nodes are connected only through concatenated path
segments in the training data (Figure 6). These findings align with our theoretical analysis, confirming
that current Transformers have limitations in learning transitive reachability relationships, unlike
human intelligence. Finally, we validate these results on a real-world task planning benchmark,
Blocksworld [22], which directly corresponds to the path-finding problem (see Appendix F).

2
Preliminaries

Throughout this paper, we use the following notations for matrices and vectors: a and A stand for a
column vector and a matrix, respectively. Notations a(i) and A(i,j) denote the ith entry of vector a
and the (i, j)th entry in matrix A, respectively. We also denote the ith row of matrix A by A(i,:).

2


---Page Break---
2.1
Autoregressive Transformer Architecture and Loss Function
In this paper, we adopt the standard GPT architecture [17]. We use the following notation for the
architecture and loss function in our analysis. Let N denote the sequence length, d the embedding
size, H the number of heads, dk = d/H the embedding size per head, and M the vocabulary size.
One key component of the architecture is the attention mechanism, which is formulated as:

Attention(Q, K, V ) = softmax
QK⊤

√dk


V
(1)

where Q ∈RN×dk, K ∈RN×dk, V ∈RN×dk are the query, key, and value matrices, respec-
tively. Denoting X ∈RN×d as input, the multi-head attention is computed as MHA(X) =
Concati∈[H](Attention(XW Q
i , XW K
i , XW V
i )), where W Q
i
∈Rd×dk, W K
i
∈Rd×dk, W V
i
∈
Rd×dk are the learnable weight matrices for the query, key, and value matrices of the i-th head.

The feed-forward layer is a two-layer multi-layer perceptron (MLP) defined as follows:

FFN(X) = max(0, XW1 + 1N×1b⊤
1 )W2 + 1N×1b⊤
2
(2)

where W1 ∈Rd×4d, W2 ∈R4d×d, b1 ∈R4d, and b2 ∈Rd are the learnable weight matrices and
1N×x is the all-one matrix with dimension N × x. Finally, one Transformer layer is defined as:

Transformer(X) = FFN(LN2(MHA(LN1(X)) + X)) + MHA(LN1(X)) + X
(3)

where LN1 and LN2 are two layer normalizations.

With these essential components in place, we proceed to introduce the procedures of GPT. The training
data consists of many sequences of tokens, where each sequence is expressed as u = (u1, · · · , uN),
in which un denotes the token id for the n-th token in sequence u. We first represent the tokens in
u by a one-hot embedding matrix U ∈RN×M, where U(n,un) = 1 and 0 elsewhere. Then there
are learnable token embedding matrix Wt ∈RM×d and positional embedding matrix Wp ∈RN×d,
and the input H0 = UWt + Wp ∈RN×d. This input H0 is fed into an L-layer Transformer, i.e.,
Hl = Transformer(Hl−1) ∈RN×d for l = 1, · · · , L.

Finally, the output embedding goes through another layer normalization LNt, and then it is multiplied
by a learnable output weight matrix Wo ∈Rd×M to convert back to probability weights over all
possible tokens. We calculate the output probability vector at position n, denoted by ˆu(n+1), using
ˆu(n+1) = softmax((LNt(HL))(n,:)Wo), 1 ≤n < N. This probability vector is used to predict the
token for position n + 1, which reflects the autoregressive learning paradigm.
The adopted loss function is the cross-entropy loss for the next token prediction, given by:

ℓ(u) = −

N−1
X

n=1

M
X

k=1
U(n+1,k) log ˆu(n+1),k
(4)

2.2
Path-Planning Dataset: Syntax and Data Sources

The dataset is designed to evaluate GPT’s path-planning capability on simple graphs. It is generated
from a directed graph G = (V, E), where V is the set of nodes, and E is the set of edges. For any two
nodes u, v ∈V, (u, v) ∈E means that there is a directed edge from u to v in G. A pair of source
node s and target node t is considered a valid pair if G contains at least one path from s to t.

The training dataset D contains sequences in the format “s t s a b c t \n”, where s represents the
source node token, t the target node token, s a b c t are tokens for nodes in a valid path from s to t,
and \n indicates the end of the sequence. During testing, we provide valid pairs of source and target
nodes in the format “s t”. The model is tasked with completing the remaining tokens in the sequence.
The completion is considered correct if the model generates a valid path with the correct syntax.

3
Expressiveness and Learning Dynamics of Transformer Models

3.1
Expressiveness

3


---Page Break---
Algorithm 1 A handcrafted path-finding algorithm

1: Input: Adjacency matrix A, reachability matrix
R, source node s, target node t
2: Set path P = [s t s] and set current node i = s
3: while i ̸= t do
4:
Obtain S = {k|A(i,k) = 1 and R(t,k) = 1}
5:
Randomly sample next node k from S
6:
Append k to path P, and set i = k
7: end while
8: output path P

In our path-finding task, the essential step
for completing a path is to predict the next
node based on the current information. It is
evident that to predict the subsequent node
on the path, only information related to the
current node and the target node is neces-
sary. Algorithm 1 introduces a idealized
method that utilizes both the adjacency and
reachability matrices of the graph. The true
adjacency matrix Atrue and the true reach-
ability matrix Rtrue are defined as:

Atrue
(i,k) =
1,
if (i, k) ∈E,
0,
otherwise.
Rtrue
(t,k) =
1,
if k can reach t in G,
0,
otherwise.

Fact 1. Assuming that t is reachable from s, then Algorithm 1 is guaranteed to output a valid path
with input A = Atrue and R = Rtrue.

To illustrate the expressive capacities of the Transformer model, we first demonstrate how to manually
construct a Transformer that can perform the path-finding task by simulating Algorithm 1.
Theorem 2. Given a graph G (with adjacency matrix Atrue and reachability matrix Rtrue), for every
ε > 0, there exists a 1-layer, 1-head, and O(|V|)-embedding-size Transformer model that generates
a valid path for every valid source-target pair (s, t) with probability at least 1 −ε.

The proof involves encoding the adjacency and reachability matrices into the weights of the FFN and
attention layers, respectively, while mimicking the computation of Algorithm 1 (see Appendix B).

3.2
Learning Dynamics

Having established the mathematical existence of a Transformer model capable of accomplishing
path-finding in a given network, as demonstrated in Theorem 2, we now shift our focus to the
following fundamental question: Can the Transformer architecture, trained on sufficient path data
with an autoregressive loss as in Equation (4) and using the gradient descent (GD) method, learn the
adjacency and reachability matrices and carry out path-finding similar to the idealized Algorithm 1?

Theoretically, we notice that the Transformer may not be able to learn the true adjacency and
reachability matrices for the underlying graph. Instead, it can only learn the relevant information that
is directly encoded in the observed training data D. Therefore, we define the observed adjacency and
reachability matrices based on the training data D as follows.

Aobs
(i,k)(D) =
1,
if ∃u ∈D, n ∈[3, N −1] s.t. un = i, un+1 = k
0,
otherwise

Robs
(t,k)(D) =
1,
if ∃u ∈D, n ∈[4, N] s.t. u2 = t, un = k
0,
otherwise.

Naturally, the observed adjacency matrix Aobs(D) only records the edges (i, k) that appear in some
path within the training data D. On the other hand, the observed reachability matrix Robs(D) exhibits
more nuanced distinctions from the true reachability matrix. It only records that node t is reachable
from node k, if the training data D contains a path (sequence) whose target node is t and k appears
as a non-source node on the path. We call such pairs (t, k) observed reachable pairs. Noticeably,
reachability through transitivity, i.e., through concatenation of path segments in D, is not observed.

Here we consider the following simplified 1-layer and 1-head Transformer structure: a) The at-
tention weight is only on the target node (the second token), i.e., we manually set every row in
softmax

QK⊤

√dk


in Eq. (1) to be a one-hot vector with the second coordinate being 1 (this is
validated in our experiments shown in Figure 4), and set the positional embedding matrix Wp = 0;
b) We remove all the layer normalizations, and use FFN(X) = XW M instead of Eq.
(2),
Transformer(X) = FFN(X) + MHA(X) instead of Eq. (3); c) The token embedding matrix
Wt and the output weight matrix Wo are set to be identity. The embedding size is the same as the
vocabulary size (d = M), and we only consider the cross-entropy loss of predicting the next node.

4


---Page Break---
Since there is only one layer and one head, we use W V to represent the weight of the value matrix in
the attention layer. Under the above Transformer structure,

(HL)(n,:)Wo = (UWtW M+αUWtW V )(n,:)Wo = (UW M+αUW V )(n,:) = W M
(un,:)+W V
(u2,:),

where α is the manually set attention weight matrix (every row is a one-hot vector with the second
coordinate being 1). Therefore, the probability vector when predicting the (n + 1)-th token is
softmax(W M
(un,:) + W V
(u2,:)), and the prediction probability ˆun+1,k equals

ˆun+1,k =
exp(W M
(un,k) + W V
(u2,k))
P

ℓexp(W M
(un,ℓ) + W V
(u2,ℓ)).
(5)

Let Ni,j,k be the number of times in D that: a) the current node is i; b) the target node is j; c) the
next node is k, and let Ni,j = P
k Ni,j,k, then we have the following theorem.

Theorem 3. Under the cross-entropy loss ℓ(D), for all (i, k) pairs, i) if P

j Ni,j = 0, then
∂ℓ(D)
∂W M
(i,k) is

always 0; ii) if P
j Ni,j > 0 but P
j Ni,j,k = 0, then
∂ℓ(D)
∂W M
(i,k) is always positive; iii) if P
j Ni,j,k > 0,

then
∂ℓ(D)
∂W M
(i,k) is negative when W M
(i,k) →−∞. Similarly, for all (j, k) pairs, i) if P

i Ni,j = 0, then

∂ℓ(D)
∂W V
(j,k) is always 0; ii) if P

i Ni,j > 0 but P

i Ni,j,k = 0, then
∂ℓ(D)
∂W V
(j,k) is always positive; iii) if
P
i Ni,j,k > 0, then
∂ℓ(D)
∂W V
(j,k) is negative when W V
(j,k) →−∞.

Proof Sketch. By the definition of the cross-entropy loss in Eq.(4), and the prediction probability
vector in Eq.(5), the total cross-entropy loss of the model (with matrices W M, W V ) is

ℓ(D) = −
X

i,j,k
Ni,j,k(W M
(i,k) + W V
(j,k)) +
X

i,j
Ni,j log

 X

ℓ
exp(W M
(i,ℓ) + W V
(j,ℓ))

!

.

Then we can get that: (the proof for the W V part is similar)

∂ℓ(D)
∂W M
(i,k)
= −
X

j
Ni,j,k +
X

j
Ni,j
exp(W M
(i,k) + W V
(j,k))
P

ℓexp(W M
(i,ℓ) + W V
(j,ℓ)).
(6)

In case i), P

j Ni,j = 0 implies that P

j Ni,j,k = 0. Hence
∂ℓ(D)
∂W M
(i,k) is always 0.

In case ii), P

j Ni,j > 0 implies that the second term in Eq. (6) is positive, while P

j Ni,j,k = 0

implies that the first term in Eq. (6) is 0. Hence
∂ℓ(D)
∂W M
(i,k) is always positive.

In case iii), when P

j Ni,j > 0 and W M
(i,k) →−∞, the second term in Eq. (6) converges to 0, and it

is smaller than P

j Ni,j,k > 0. Hence,
∂ℓ(D)
∂W M
(i,k) is negative when W M
(i,k) →−∞.

The above technical theorem directly leads to a theoretical explanation on how the model learns the
adjacency and reachability information, as explained below.

Learning the adjacency matrix.
Let E(D) denote the set of edges appearing in the training
dataset D, which corresponds to the observed adjacency matrix Aobs(D). For any (i, k) ∈E(D),
P

j Ni,j,k > 0, and for any (i′, k′) /∈E(D), P

j Ni′,j,k′ = 0. Then from the above theorem, under
the gradient descent learning paradigm, W M
(i′,k′) will keep decreasing (since its gradient is always
positive), while W M
(i,k) will not (since its gradient becomes negative when its value is sufficiently
negative). This tends to make W M
(i,k) higher than W M
(i′,k′) after training. In this way, the Transformer
model learns the information about the observed adjacency matrix with matrix W M.

To facilitate comprehension, we conducted a simple experiment on the simplified Transformer, and
present the results in Figure 1, In this experiment, we generate a 10-node graph, and use 3 different
training datasets D1, D2, D3 based on this graph. D1 contains all the paths with length 1; D2 contains

5


---Page Break---
(a) True adjacency Atrue
(b) W M under D1
(c) W M under D2
(d) W M under D3

Figure 1: Empirical verification regarding the learning of the adjacency matrix.

(a) True reachability Rtrue

(b) Robs(D1)
(c) Robs(D2)
(d) Robs(D3)

(e) W V under D1
(f) W V under D2
(g) W V under D3

Figure 2: Empirical verification regarding the learning of the observed reachability matrix.

all the paths with length 1 and 20% of the paths with length higher than 1; and D3 contains all the
possible paths. Figure 1(a) is the true adjacency matrix of the graph, which is also the observed
adjacency matrix for the three datasets. Figure 1(b), 1(c), 1(d) are the W M matrices with training
datasets D1, D2, D3, respectively. Upon observation, it becomes evident that these W M matrices all
successfully capture the structural information from the adjacency matrix.

Learning the reachability matrix.
Similar to the process of learning the adjacency matrix, since
only observed reachable pairs (j, k) have P

i Ni,j,k > 0, the gradient descent learning paradigm
tends to make the W V
(j,k) terms corresponding to observed reachable pairs (j, k) higher than the
W V
(j′,k′) terms corresponding to non-observed reachable pairs (j′, k′) (which is either not reachable
or not observed) after the training. In this way, the Transformer model captures the structural
information of observed reachability matrix with weight matrix W V .

Figure 2 shows the correlation between W V and the observed reachabilities under different dataset
D’s in the above experiment. Figure 2(a) is the real reachability matrix of the graph; Figure 2(b), 2(c),
2(d) are the observed reachability matrices in datasets D1, D2, D3, respectively; and Figure 2(e), 2(f),
2(g) are the W V matrices with training datasets D1, D2, D3, respectively. These illustrations show
that all the weight matrices W V can satisfactorily learn the information of the observed reachabilities
present in the training datasets, but cannot deduce any non-observed reachabilities.

Predicting the next node on a path.
From Eq.(5), the probability vector for predicting the next
node is softmax(W M
(un,:) + W V
(u2,:)), where un represents the current node, and u2 represents the
target node. This resembles the procedure in Algorithm 1: it predicts the next node k such that both
W M
(un,k) is high (corresponding to A(un,k) = 1) and W V
(u2,k) is high (corresponding to R(u2,k) = 1).

6


---Page Break---
(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes
(d) 400 Nodes

Figure 3: Accuracy on the test datasets with embedding size d = 120.

In summary, our theoretical analysis demonstrates that a simplified one-layer, one-head autoregressive
Transformer (with perfect attention) can effectively learn crucial adjacency and reachability informa-
tion from the training data through gradient descent training. Moreover, it can utilize this learned
information to predict the next node akin to the decision-making process of a human algorithm
designer in similar scenarios. This suggests that, when confronted with the path-finding or more
general planning task with a given goal, the Transformer learns the structural information to associate
the next step with both the current step and the goal, enabling it to generate the subsequent task step.
Nevertheless, the Transformer’s limitation in learning only the observed reachability matrix—without
deducing the complete reachability matrix—hints at potential constraints on the goal-oriented infor-
mation it can acquire. This limitation may result in the Transformer failing to grasp novel reachability
relationships derived from the transitivity of reachability relations, unlike human intelligence.

4
Empirical Evaluations: Peeking into a Trained Transformer

In this section, we conduct extensive experiments on the path-finding task using the general Trans-
former architecture as described in Section 2.1. The datasets are generated as described below.

The DAG is generated randomly based on two parameters: the number of nodes n, and the probability
of edge p = 0.1: For any 1 ≤i < j ≤n, there is an edge (i, j) ∈E with probability p. Given the
DAG, we first find all the possible reachable pairs (s, t). Then these reachable pairs are separated
into the training set (w.p. 0.5) and the test set (w.p. 0.5), but if edge (s, t) ∈E, we always put (s, t) in
the training set. For a reachable pair (s, t) in the training set, we generate m = 20 random paths that
start at s and end at t, and put these m paths into the training dataset. When generating the random
path, at each current node i, we find all the possible k ∈V such that Atrue
(i,k) = 1 and Rtrue
(t,k) = 1 (i.e.,
there is an edge (i, k) ∈E, and k could also reach the target node t), and uniformly choose a random
one in them. Moreover, if (s, t) ∈E, we always put the one-edge path “s t s t \n” in the training
dataset to guarantee that all edges appear at least once in the training data.

4.1
Accuracy on Test Dataset

We train Transformer models on the aforementioned training dataset and subsequently evaluate the
performance of these models using the pairs in the test dataset. The correctness of a model’s output is
determined based on its validity in terms of syntax and whether it corresponds to a valid path from s
to t. In our experiments, we employ Transformer models with an embedding size of d = 120. We
conduct tests using various configurations, ranging from 1-layer and 1-head to 6-layer and 6-head,
while considering different graph sizes, with number of nodes n ranging from 100 to 500. The
accuracy results take average over 10000 trials, and are presented in Figure 3 (due to space limits,
some results are deferred to Appendix E, and all of them are consistent with our conclusions). From
these results, we make the following observations: a) When comparing the figures, the accuracy tends
to decrease as the number of nodes increases; b) When examining each row, the accuracy remains
relatively stable even as the number of attention heads increases; c) When examining each column,
the accuracy shows at most a slight improvement as the number of layers increases.

The above observations suggest that the embedding size is the most important hyperparameter that
affects the accuracy of the model. On the one hand, when the embedding size is sufficiently large
compared to the graph size, even 1-layer and 1-head models perform well. This coincides with our
theoretical analysis, which shows that when the embedding size equals to the graph size, the 1-layer
and 1-head structure is enough to predict the next nodes accurately. On the other hand, when the

7


---Page Break---
(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes
(d) 400 Nodes

Figure 4: The average attention in the 1-layer and 1-head Transformers.

(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes
(d) Average Weight Gap

Figure 5: The first 20 rows and columns of W M ′ (the red boxes correspond to 1’s in the adjacency
matrix), and the average weight gap between edge terms and non-edge terms in W M ′.

embedding size is small compared to the graph size, even 6-layer and 6-head Transformers cannot
achieve good performance. Because of this, in the following, we concentrate on the explainability of
the 1-layer and 1-head Transformer models.

4.2
Peeking into a Trained Transformer

Attention.
In our analysis, we assume that the attention is fixed on the target node. Is this true
for the Transformer models learned from real data? The corresponding results are shown in Figure
4. These results are obtained by looking into the attention mechanism of the 1-layer and 1-head
Transformer models, and showing the average (taking on the test dataset) matrix of softmax

QK⊤

√dk


,

of which the n-th row represents the attention vector for predicting the (n + 1)-th token.

Note that the second column in these figures represents the attention weights on the second token,
which corresponds to the target node in our test data. We can see that, when predicting the next
tokens, almost all the attention weights are concentrated on this column, especially for those models
with higher accuracy (Figure 4(a) for n = 100 and Figure 4(b) for n = 200). This demonstrates
that indeed the Transformer model learns the correct attention for the path-finding task, and our
assumption on the attention weights for the theoretical analysis is reasonable.

Adjacency Matrix.
In the 1-layer and 1-head Transformers, let W M ′ (shown in Figure 5) be the
matrix whose i-th row is FFN
 
e⊤
i Wt

Wo + (e⊤
i Wt)Wo, where ei is the one-hot column vector
that represents the token for node i. Based on the Transformer computation, intuitively this matrix
is one of the components in the output that contains information related to the current node. The
detailed reason for choosing this matrix is explained in Appendix D.

In Figure 5(a), the W M ′ matrix and the adjacency matrix are highly aligned: all large entries in the
W M ′ matrix correspond to real edges, and all real edges correspond to large entries in the W M ′

matrix. This high accuracy is because the embedding size d = 120 is higher than the number of
nodes n = 100. If the embedding size is lower than the graph size (Figures 5(b), 5(c)), we inevitably
lose some accuracy when approximating the adjacency matrix by the product of matrices with rank
smaller than the graph size. Even so, there is still high relevance between W M ′ and the adjacency
matrix: almost all real edges correspond to large entries in the W M ′ matrix.

In Figure 5(d), we show the gap between the average weight corresponding to edges (i.e., the average
of W M ′
(i,j)’s with i < j and (i, j) ∈E) and the average weight corresponding to non-edges (i.e.,

8


---Page Break---
the average of W M ′
(i,j)’s with i < j and (i, j) /∈E) during the training process. These gaps keep
increasing until convergence, suggesting that weights between edges and non-edges are more easily
separated as the learning process proceeds.

Reachability Matrix.
In the 1-layer and 1-head Transformers, let W V ′ be the matrix whose i-th
row is (e⊤
i Wt)W V Wo + FFN
 
(e⊤
i Wt)W V 
Wo, where ei is the one-hot column vector that
represents the token for node i. Intuitively, this matrix is the remaining component in the output that
contains information related to the target node. The detailed reason is also explained in Appendix D.

In Figure 6(a), we show the average weights of three different sets in the graphs: “obs” corresponds
to the W V ′
(t,k)’s with t ≥k and Robs
(t,k) = 1; “real\obs” corresponds to the W V ′
(t,k)’s with t ≥k,

Robs
(t,k) = 0 but Rreal
(t,k) = 1; and “non” corresponds to the W V ′
(t,k)’s with t ≥k and Rreal
(t,k) = 0. Here
we only show the results of graphs with 100 nodes and 200 nodes, since their accuracy is high enough
and their attention is quite close to being concentrated on the target node. When there are more
nodes, the ability to approximate the reachability matrix is not enough for us to distinguish it. From
these average weights, we can see that the Transformer learns Robs quite well, as for those terms in
“real\obs”, their weights are almost the same as those in “non”. This echoes our analysis.

To further demonstrate that Rreal is not learned as good as Robs, we divide the source-target node
pairs (s, t) in the test dataset into four categories: a) degree 0: Robs
(t,s) = 1; b) degree 1: (s, t) is not of
degree 0, while s has at least one out-neighbor node u such that (u, t) is of degree 0, i.e. Robs
(t,u) = 1;
c) degree 2: (s, t) is not of degree 0 and 1, while s has at least one out-neighbor node u such that
(u, t) is of degree 1; d) degree 3 or more: the remaining (s, t) pairs in the test dataset. Roughly
speaking, in our analysis, for (s, t) pairs of degree 0 or 1, we know that there is a node u such that
Aobs
(s,u) = 1 and Robs
(t,u) = 1. Then node u will have a large weight, indicating a high accuracy. As for
(s, t) pairs of degree 2 or more, there is no node u such that both Aobs
(s,u) = 1 and Robs
(t,u) = 1. In this
case, the high-weight entry when predicting the next node of s is either an adjacent node of s or a
recorded node that can reach t. This should reduce the accuracy.

To see this, we check the accuracy of the Transformers on the (s, t) pairs of the four different
categories. The results are shown in Figure 6 (b)-(d). In these figures, each row of the accuracy
matrix is further divided into four sub-rows corresponding to the accuracy of degree-0 pairs, degree-1
pairs, degree-2 pairs, and degree-3 or more pairs respectively (in the graph with 100 nodes, there are
no test (s, t) pairs in the degree-3 or more category). From these results, we can see that the accuracy
for degree-2 pairs and degree-3 or more pairs is much lower than the two other categories in most
cases. It indicates that, even with more parameters and a more complex structure (e.g. a 6-layer
and 6-head Transformer), the Transformer model has a fundamental difficulty in generating paths
for high-degree source-target pairs, namely those pairs that can only be connected by concatenating
several path segments in the training dataset. This result demonstrates the validity of our theoretical
analysis, i.e., after training with gradient descent on cross-entropy loss, the Transformer can only
learn observed reachability, and will miss those unobserved reachability deduced from the transitivity
of the reachability relation.

In summary, our extensive empirical evaluation leads to the following conclusions about the Trans-
former model in achieving the path-finding task: (a) With large enough embedding size, the model
can achieve high accuracy in general; (b) The model achieves its performance by concentrating
attention on the target nodes as intended, and learning the information on adjacency and reachability
matrices, just as what a human would do and as predicted by our theoretical analysis; and (c) The
model may have limitations and fail to learn high-order reachability relations through transitivity, and
thus fail to generate paths derived from high-order reachability.

5
Discussion and Future Work

In summary, this paper and Project ALPINE more broadly conceptualize planning as path-finding
in networks, and combine theoretical analysis of the Transformer architecture and autoregressive
loss with empirical validation. Our aim is to uncover the mechanisms by which intelligence may
emerge from autoregressive Transformer architectures. We analytically demonstrate that Transformers
possess the expressiveness required to perform path-finding tasks and that gradient descent on cross-
entropy loss enables them to learn necessary—but incomplete—graph information for path-finding.

9


---Page Break---
(a) Average Weights
(b) 100 Nodes
(c) 200 Nodes
(d) 300 Nodes

Figure 6: The average weights in W V ′, and the accuracy for (s, t)’s with different degrees.

Additionally, we reveal both analytically and empirically that autoregressive training of language
models has inherent limitations in the path-finding task.

Practical Implications: Our findings in LLMs for path planning may have practical implications for
the training, testing, and enhancement of language models. In particular, the limitations we identified
in current Transformer architectures for transitive reasoning suggest several directions for enhancing
LLM frameworks to achieve more advanced and general planning-and-reasoning capabilities across
diverse applications. For instance, in data generation for training, creating more diversified datasets
that explicitly cover more reachability relationships may help the model achieve a higher accuracy.
When evaluating a language model’s planning capability, it may be beneficial to test for higher-order
relationships not directly encoded in the training data but requiring chaining and concatenation to
assess whether the model can perform transitive planning. Furthermore, by highlighting limitations in
current language models, our study motivates future research into improved Transformer architectures,
including incorporating transitivity directly into the model structure.

Challenges in Reasoning about Unobserved Reachability: Technically, the challenge in learning
unobserved reachability with current Transformer architectures stems from the nature of next-token
prediction loss: learning unobserved reachability incurs a higher training loss. Specifically, when
predicting the next token for a given current node i and target node j, the optimal distribution for
minimizing training loss should align with the observed distribution in the training dataset, i.e.,
Pr[next node = k|current node = i and target node = j] = Ni,j,k

Ni,j
(as explained in Section 3.2).
Learning unobserved reachabilities requires deviating from the distribution defined by the train-
ing data, which leads to a higher training loss. Consequently, with the current training loss and
Transformer architecture, the model cannot learn unobserved reachabilities, such as transitive reacha-
bility. To enable the model to learn transitivity, we may need alternative training objectives, such as
path accuracy, or structural improvements to the Transformer that allow it to ‘deduce’ unobserved
reachabilities. Conceptually, the current training data and loss objective do not provide sufficient
information to teach the model transitivity or other derived relationships. Therefore, enhancing
transitivity and similar capabilities may require enriching the training data, modifying the objective
function, or incorporating new components into the model architecture.

Future Directions: Our investigation opens several promising directions for future research: (a)
Extending our study to hyper-graphs and hyper-paths, where a hyper-edge represents scenarios
requiring multiple preconditions to be met simultaneously in order to carry out the next step, as often
seen in task planning and mathematical proofs. (b) Addressing the limitations of Transformers in
path-finding and other planning tasks by exploring richer path-finding languages, fine-tuning, or
architectural improvements to LLMs. (c) Examining connections between the abstract path-finding
task and concrete planning tasks (e.g., block manipulation in Blocksworld) to understand whether,
and how, Transformers abstract these tasks into path-finding frameworks. (d) Investigating in-context
path-finding capabilities, where training data includes different graphs with corresponding paths,
to see how Transformers learn to find new paths in new graphs. (e) Exploring the integration of
chain-of-thought and backtracking capabilities into Transformers for path-finding, which may offer
crucial insights into enabling these features for general search and planning tasks.

In our ongoing project ALPINE, we plan to deepen our investigation into all the aforementioned fronts.
We also hope that our work will inspire more researchers to study LLMs through combined theoretical
and empirical analysis, with the ultimate goal of enhancing their capabilities and understanding how
human-like intelligence can be achieved through statistical learning and AI mechanisms.

10


---Page Break---
References

[1] O. J. Achiam, S. Adler, and et al. GPT-4 technical report. 2023.

[2] Z. Allen-Zhu and Y. Li. Physics of language models: Part 1, context-free grammar. arXiv
preprint arXiv:2305.13673, 2023.

[3] Z. Allen-Zhu and Y. Li. Physics of language models: Part 3.1, knowledge storage and extraction.
arXiv preprint arXiv:2309.14316, 2023.

[4] Z. Allen-Zhu and Y. Li. Physics of language models: Part 3.2, knowledge manipulation. arXiv
preprint arXiv:2309.14402, 2023.

[5] Z. Allen-Zhu and Y. Li. Physics of language models: Part 3.3, knowledge capacity scaling laws.
arXiv preprint arXiv:2404.05405, 2024.

[6] Z. Chai, T. Zhang, L. Wu, K. Han, X. Hu, X. Huang, and Y. Yang. GraphLLM: Boosting graph
reasoning ability of large language model. arXiv preprint arXiv:2310.05845, 2023.

[7] G. Feng, B. Zhang, Y. Gu, H. Ye, D. He, and L. Wang. Towards revealing the mystery behind
chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems,
36, 2023.

[8] A. Giannou, S. Rajput, J.-y. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos. Looped transformers
as programmable computers. In International Conference on Machine Learning, pages 11398–
11442. PMLR, 2023.

[9] J. Guo, L. Du, and H. Liu. Gpt4graph: Can large language models understand graph structured
data? an empirical evaluation and benchmarking. arXiv preprint arXiv:2305.15066, 2023.

[10] W. Huang, P. Abbeel, D. Pathak, and I. Mordatch. Language models as zero-shot planners:
Extracting actionable knowledge for embodied agents. In International Conference on Machine
Learning, pages 9118–9147. PMLR, 2022.

[11] N. Lee, K. Sreenivasan, J. D. Lee, K. Lee, and D. Papailiopoulos. Teaching arithmetic to small
transformers. arXiv preprint arXiv:2307.03381, 2023.

[12] X. Liu, Z. Peng, X. Yi, X. Xie, L. Xiang, Y. Liu, and D. Xu. ToolNet: Connecting large language
models with massive tools via tool graph. arXiv preprint arXiv:2403.00839, 2024.

[13] Z. Liu, Z. Lai, Z. Gao, E. Cui, Z. Li, X. Zhu, L. Lu, Q. Chen, Y. Qiao, J. Dai, et al. ControlLLM:
Augment language models with tools by searching on graphs. arXiv preprint arXiv:2310.17796,
2023.

[14] Z. Luo, X. Song, H. Huang, J. Lian, C. Zhang, J. Jiang, X. Xie, and H. Jin. Graphinstruct:
Empowering large language models with graph understanding and reasoning capability. arXiv
preprint arXiv:2403.04483, 2024.

[15] W. Merrill and A. Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers.
Transactions of the Association for Computational Linguistics, 11:531–545, 2023.

[16] I. Momennejad, H. Hasanbeig, F. Vieira Frujeri, H. Sharma, N. Jojic, H. Palangi, R. Ness, and
J. Larson. Evaluating cognitive maps and planning in large language models with CogEval.
Advances in Neural Information Processing Systems, 36, 2023.

[17] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding
by generative pre-training. 2018.

[18] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang. HuggingGPT: Solving AI tasks with
ChatGPT and its friends in Huggingface. Advances in Neural Information Processing Systems,
36, 2023.

[19] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents
with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36,
2024.

11


---Page Break---
[20] J. Tang, Y. Yang, W. Wei, L. Shi, L. Su, S. Cheng, D. Yin, and C. Huang. GraphGPT: Graph
instruction tuning for large language models. arXiv preprint arXiv:2310.13023, 2023.

[21] T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. Solving Olympiad geometry without human
demonstrations. Nature, 625(7995):476–482, 2024.

[22] K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati. On the planning abilities
of large language models-a critical investigation. Advances in Neural Information Processing
Systems, 36, 2023.

[23] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager:
An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291,
2023.

[24] H. Wang, S. Feng, T. He, Z. Tan, X. Han, and Y. Tsvetkov. Can language models solve graph
problems in natural language? Advances in Neural Information Processing Systems, 36, 2023.

[25] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, et al.
A survey on large language model based autonomous agents. Frontiers of Computer Science,
18(6):1–26, 2024.

[26] J. C. Whittington, D. McCaffary, J. J. Bakermans, and T. E. Behrens. How to build a cognitive
map: insights from models of the hippocampal formation. arXiv preprint arXiv:2202.01682,
2022.

[27] K. Yang, J. Ackermann, Z. He, G. Feng, B. Zhang, Y. Feng, Q. Ye, D. He, and L. Wang. Do
efficient transformers really save computation? arXiv preprint arXiv:2402.13934, 2024.

[28] T. Ye, Z. Xu, Y. Li, and Z. Allen-Zhu. Physics of language models: Part 2.1, grade-school math
and the hidden reasoning process. arXiv preprint arXiv:2407.20311, 2024.

[29] T. Ye, Z. Xu, Y. Li, and Z. Allen-Zhu. Physics of language models: Part 2.2, how to learn from
mistakes on grade-school math problems. arXiv preprint arXiv:2408.16293, 2024.

[30] R. Zhang, S. Frei, and P. L. Bartlett. Trained transformers learn linear models in-context. arXiv
preprint arXiv:2306.09927, 2023.

[31] Y. Zhang, A. Backurs, S. Bubeck, R. Eldan, S. Gunasekar, and T. Wagner. Unveiling Trans-
formers with LEGO: a synthetic reasoning task. arXiv preprint arXiv:2206.04301, 2022.

12


---Page Break---
A
Related Works

A.1
LLMs for Planning

Several recent studies have empirically evaluated the planning abilities of large language models.
For instance, CogEval has introduced a variety of planning tasks set in mazes and graphs, ultimately
finding no evidence to suggest LLMs grasp the intricacies of the maps or the planning tasks themselves
[16]. Similarly, another study explored the Blocksworld game, a planning challenge where humans
typically achieve success rates above 70%, in stark contrast to GPT-3’s mere 5% success rate [22].
Our paper also proposes a novel approach by formulating a class of planning problems as path-finding
on graphs, applying this model to the Blocksworld game and uncovering significant insights, as
detailed in Appendix F.

Despite these seemingly negative evaluations, LLMs have shown remarkable aptitude in executing
real-world planning tasks, creating the field of autonomous agents [25]. Certain applications of
autonomous agents feature explicit graphs. In the tool agent HuggingGPT [18], LLMs are deployed
to trigger a sequence of external APIs in response to user requests. Here, APIs are conceptualized as
graph nodes, with their interrelations represented as edges, and the selection process akin to identify-
ing a path or subgraph that aligns with user demands. This scenario is an extension of the settings
discussed in this paper, where the graph is text-attributed and the objective function is evaluated
through textual analysis. In addition, the application of graph search techniques has been shown to
enhance the performance of tool agents significantly [13, 12]. This demonstrates that our approach of
abstracting planning as path-finding in graphs is reasonable. The math agent AlphaGeometry utilizes
LLMs to solve geometry problems [21]. By treating lemmas as nodes and their interdependencies
as edges, the process of finding a proof of a theorem is analogous to finding a path to the theorem
node in the above graph formed by possible lemma nodes and their interdependency edges. However,
[21] only focuses on using LLMs to generate auxiliary constructions, and the reasoning tasks are
done by a non-LLM engine. This is very different from our approach. There are no explicit graphs
in other agents, such as game agents [23], embodied agents [10], and code agents [19]. The core
strategy in these domains is to employ verbal reinforcement learning within LLMs. However, it is
noteworthy that any dynamic programming problem characterized by deterministic state transitions
can be reformulated as a shortest path problem on a graph, with states and transitions represented as
nodes and edges, respectively. As a result, the area of autonomous agents is also closely related to the
path-finding task investigated in this paper.

A.2
LLMs for Graphs

GPT4Graph [9] and NLGraph [24] have developed extensive frameworks for assessing LLMs in
the context of graph tasks. These frameworks encompass a broad spectrum of challenges, including
classic graph problems (e.g., connectivity, cycle detection, and topological sorting), graph neural
network (GNN) tasks (e.g., node and graph classification), and semantic graph question answering
(e.g., knowledge graph inquiries). They also explore various input formats, such as adjacency lists,
edge lists, GML, and GraphML, alongside innovative prompting techniques such as few-shot, role
prompting, chain-of-thought, and algorithmic prompting (e.g., stating “we are using DFS”). These
studies demonstrate that LLMs possess basic graph processing capabilities, and the choice of prompts
and formats significantly influences the performance. Yet, they also reveal the models’ susceptibility
to spurious correlations within graphs. GPT-4, for instance, only achieves around 50% accuracy on
shortest path tasks, even when utilizing complex prompts. To our knowledge, our paper presents the
first theoretical analysis that identifies and explains the spurious correlations learned by transformers,
partially supporting some of the negative outcomes reported in these studies.

There has also been a surge in efforts aiming at bolstering LLMs’ performance on graph tasks.
Innovations such as GraphGPT [20] and GraphLLM [6], which incorporate an additional GNN
encoder, have shown notable improvements across the aforementioned graph tasks. GraphInstruct
[14] seeks to enhance LLMs’ capabilities using pure LLM approaches. This involves meticulously
documenting the steps of classical algorithms (e.g., BFS and DFS) and fine-tuning LLMs to learn
these graph algorithms. This method of procedural supervision has extended the capacity of LLMs in
graph tasks from the complexity class TC0 to P/poly [7]. However, while this approach has yielded
performance improvements in simpler tasks such as topological sorting and connectivity, it has proven
less effective for more complex challenges, e.g., finding Hamiltonian Paths.

13


---Page Break---
A.3
Algorithm Simulation with Transformers

Recent theoretical investigations have shed light on the capability of the Transformer to simulate
algorithms, a topic that has garnered considerable interest. From the view of discrete algorithms,
Transformer models are likened to parallel circuits characterized by polynomial width and constant
depth, which places them within the TC0 complexity class (note that TC0 ⊆NC1 ⊆P). On
the other hand, despite their impressive expressiveness, the Transformer is theoretically incapable
of addressing a range of P-complete problems, including the testing of Context-Free Grammar
Membership [15]. However, the advent of chain-of-thought prompting has enabled the Transformer
to sequentially simulate algorithms, thereby equipping them to tackle P-complete problems in
domains such as arithmetic and decision-making [7]. The exploration then extends to continuous
algorithms, where it has been demonstrated that the Transformer can approximate functions such as
matrix inversion, Stochastic Gradient Descent (SGD), and power iterations [8]. Our study specifically
applies Transformer models to simulate path-finding algorithms, presenting evidence that their
expressiveness is sufficient for such tasks (Theorem 2). Nevertheless, the usage of autoregressive loss
and gradient descent introduces certain limitations, which have not been studied in existing works.

A.4
Mechansims of LLMs

LLMs have demonstrated capabilities that exceed the theoretically predicted lower bounds of expres-
siveness. To demystify this paradox, numerous studies have employed experimental methodologies
akin to those used in the physical and biological sciences. Their aim is to decode the mechanisms of
LLMs. The foundational strategy is to generate controlled synthetic datasets to analyze how trans-
formers (not necessarily LLMs) complete various tasks. Standard methods for this analysis include
visualizing attention patterns to examine computational properties (such as locality and time invari-
ance) and employing linear probing on the hidden states to determine the extent of learning. Given that
the training data is synthetic and the ground-truth mappings are generally known, it becomes feasible
to isolate the influence of various factors (e.g., prompting strategies, chain-of-thought reasoning, and
data formatting). For example, a dataset designed for learning group operations, as detailed in [31],
facilitates the exploration of how pretraining, data composition, and neural architecture influence
reasoning tasks within LLMs. Similarly, the generation of synthetic context-free grammar (CFG)
data, as described in [2], enables training GPT-2 models, uncovering their capacity to learn dynamic
programming algorithms for parsing CFGs. Synthetic datasets focusing on biographical knowledge,
presented in [3, 4, 5], probe into the mechanisms of knowledge storage, retrieval, manipulation, and
the implications of scaling laws. Moreover, the work in [11] introduces synthetic datasets that aim
to understand how smaller LLMs tackle basic arithmetic operations, e.g., addition, and examines
the effects of few-shot prompting, pretraining, and model scaling [11]. Our work builds upon these
investigations by conducting controlled experiments with a path-finding dataset, thereby shedding
light on the complexities and challenges of planning in language models.

B
Proof of Theorem 2

Theorem 2. Given a graph G (with adjacency matrix Atrue and reachability matrix Rtrue), for every
ε > 0, there exists a 1-layer, 1-head, and O(|V|)-embedding-size Transformer model that generates
a valid path for every valid source-target pair (s, t) with probability at least 1 −ε.

Proof. For simplicity, we omit all layer normalizations in this construction. Suppose the input token
sequence is “s t s u1 u2 . . . uk” with k ≥0, where s (= u0) and t are the tokens of the source and
target nodes, respectively, and nodes s, u1, · · · , uk form a path that can reach node t in graph G. Our
objective is to construct a 1-layer and 1-head Transformer model that generates an out-neighbor uk+1
of uk such that there exists at least one path from uk+1 to t in G.

In essence, we utilize the attention layer to attend the output solely to the target node t. This approach
allows the distribution of next token uk+1 to become a function of both the current node uk and the
target node t (as formulated in Section 2). Then, by integrating the adjacency matrix Atrue into the
FFN layer and the reachability matrix Rtrue into the matrix W V in the attention layer, we extract
row vectors Rtrue
(t,:) and Atrue
(uk,:) from Rtrue and Atrue, respectively, corresponding to the target node t
and current node uk. By selecting proper coefficients, we can let the output be the sum of Rtrue
(t,:) and

14


---Page Break---
Atrue
(uk,:). Following the softmax layer, the non-negligible entries in the final vector correspond to the
feasible next nodes. With this encoding, the Transformer serves as a simulator of Algorithm 1 with
input A = Atrue and R = Rtrue.

Following our notation in Section 2.1, we adopt d = |V| + 2, M = |V| + 1 and N = k + 3. In
the Transformer, there are M = |V| + 1 tokens representing the |V| nodes and the end-of-line
“\n”. Hence, the input tokens can be represented by the one-hot embedding matrix U ∈RN×M.
We let Wt = (IM×M | 0M×1) ∈RM×d and Wp =
 0(k+3)×(|V|+1) | c0 · e2

∈RN×d, here e2
represents the second unit column vector of dimension k + 3, (A | B) is the notation for matrix
concatenation by column, and c0 is a positive parameter to be decided. According to the definition of
the Transformer, we now have a matrix H0 such that the first |V| + 1 columns are the tokens of nodes
in the sequence and the last column indicates the positions of the target node t. More specifically, we
have

H0 =










e⊤
s
0
e⊤
t
c0
e⊤
s
0
e⊤
u1
0
· · ·
· · ·
e⊤
uk
0









∈RN×d,

here eu represents the one-hot token vector for node u (with dimension M = |V| + 1).

Then we construct the attention layer of the Transformer. We only have one head and let W K =
 0(|V|+2)×(|V|+1) | 1(|V|+2)×1
⊤∈Rd×d and W Q =
√

d·Id×d. Then we can compute H0W K =
 0(|V|+2)×1 | c0 · 1(|V|+2)×1 | 0(|V|+2)×(k+1)
⊤, i.e., second rows are all c0’s and other rows are all
0’s, and H0W Q =
√

d · H0.

Therefore,

(H0W Q)(H0W K)⊤

√

d
=




0
c0
01×(k+1)
0
c2
0 + c0
01×(k+1)
0(k+1)×1
c0 · 1(k+1)×1
0(k+1)×(k+1)



∈RN×N.

And we can compute the first part of the attention layer as

softmax
(H0W Q)(H0W K)⊤

√

d


=







1
k+2+ec0
ec0
k+2+ec0
1
k+2+ec0 · 11×(k+1)

1

k+2+ec2
0+c0
ec2
0+c0

k+2+ec2
0+c0
1

k+2+ec2
0+c0 · 11×(k+1)

1
k+2+ec0 · 1(k+1)×1
ec0
k+2+ec0 · 1(k+1)×1
1
k+2+ec0 · 1(k+1)×(k+1)





∈RN×N.

By setting c0 →+∞, we obtain:

softmax
(H0W Q)(H0W K)⊤

√

d


→




0
1
01×(k+1)
· · ·
· · ·
· · ·
0
1
01×(k+1)



.

Furthermore, we set W V =

c1 · Rtrue
0|V|×2
02×|V|
02×2


, where c1 > 0 is also a parameter to be decided

later. Then after the attention layer, we have a matrix as

lim
c0→+∞MHA(H0) = c1 ·




Rtrue
(t,:)
0
0
· · ·
· · ·
· · ·
Rtrue
(t,:)
0
0



∈RN×d.

Now we construct the feed-forward layer, which is a two-layer MLP.

For the first layer, the weight matrix W1 is set to be,

W1 =
 I(|V|+2)×(|V|+2)
0(|V|+2)×3(|V|+2)

∈Rd×4d.

and the bias b1 = −c1 · 14d×1, which implies that 1N×1b⊤
1
= −c1 · 1N×4d.
When c0
is large enough, the (k + 3)th row of the matrix max
 
0, (MHA(H0) + H0)W1 + 1N×1b⊤
1


15


---Page Break---
is max
 
0, c1 ·
 R(t,:) | 01×(3|V|+8)

+
 
e⊤
uk | 01×(3|V|+7)

−c1 · 11×4(|V|+2)

.
Since uk can
reach t, in c1 ·
 R(t,:) | 01×(3|V|+8)

+
 
e⊤
uk | 01×(3|V|+7)

, only the entry for node uk is
c1 + 1 while all other entries are 0 or c1.
Therefore, the (k + 3)th row of the matrix
max
 
0, (MHA(H0) + H0)W1 + 1N×1b⊤
1

can be arbitrarily close to
 
e⊤
uk | 01×(3|V|+7)

. Here
eu represents the one-hot token vector for node u (with dimension M = |V| + 1).

For the second layer, we set

W2 =

c2 · A
0|V|×2
0(3|V|+8)×|V|
0(3|V|+8)×2


∈R4d×d,

where c2 is a positive parameter to be decided, and b2 = 0. By this way, we have

lim
c0→∞(FFN(MHA(H0) + H0))(k+3,:) →
 c2 · A(uk,:) | 01×2

∈R|V|+2.

Therefore,

lim
c0→∞(H1)(k+3,:) →
 c1 · R(t,:) + c2 · A(uj,:) | 01×2

+ (euk | 0) ∈R|V|+2,

where eu represents the one-hot token vector for node u (with dimension M = |V| + 1).

Then we fix c1 = c2 and let them be large enough. In this case, the dominant entries in (H1)(k+3,:)
represent the nodes that are both the out-neighbor of uj and reachable to t, since those entries will
have the value of 2c1 while other entries are at most c1 +1. This means that (H1)(k+3,:) can correctly

indicates the next node uk+1. Specifically, let Wo =
 I(|V|+1)×(|V|+1) | 0(|V|+1)×1
⊤∈Rd×M.
Then the final output approaches the following vector

lim
c0,c1=c2→∞ˆuk+1 =
lim
c0,c1=c2→∞softmax((H1)(k+3,:)Wo)

= 1

C ·
 I[A(uk,1) = 1 ∧R(t,1) = 1], · · · , I[A(uk,|V|) = 1 ∧R(t,|V|) = 1], 0
,

where C is the number of nodes that are both the out-neighbor of uk and reachable to t. Thus, this
encoding guarantees that for any ε > 0 and Q > 0, we can always construct a 1-layer, 1-head, and
(|V| + 2)-embedding-size Transformer that provides the correct next token with probability at least
1 −
ε
2Q by selecting large enough parameters c0, c1, c2.

Then we prove that there exists a Q such that this Transformer can output a correct path for every
valid source and target node pair with probability at least 1 −ε. Suppose we can output all the nodes
that are both the out-neighbor of the current node and reachable to the target node with the same
probability in each round without any error. Then, whatever the current node is, there is a probability
of at least
1
|V||V| that the target node is reached within the next |V| generated nodes. Therefore, the

target node is reached in c3 · |V| steps with probability at least 1 −

1 −
1
|V||V|
c3
, where c3 ∈N is a
positive integer. We let c3 = log1−
1
|V||V|
ϵ
2 and Q = c3 · |V|. Then, according to the Union bound,

the Transformer can output a correct path in Q steps with an error rate of at most
ε
2Q · Q + ε

2 = ε.

Finally, there are two different rules (other than output a correct next node): i) when the input
sequence is only “s t”, the prediction of the next token should be the source node s; ii) when the
input sequence is “s t s a b c t”, the prediction of the next token should be \n. Case i) can be solved
using the Transformer architecture utilizing the position information and attention to the first position;
and case ii) can be solved by using the Transformer architecture utilizing the position information
and attention to the second position. To maintain focus on the main construction corresponding to
Algorithm 1, we omit the detailed construction for these two boundary cases.

C
Proof of Theorem 3

Theorem 3. Under the cross-entropy loss ℓ(D), for all (i, k) pairs, i) if P

j Ni,j = 0, then
∂ℓ(D)
∂W M
(i,k) is

always 0; ii) if P

j Ni,j > 0 but P

j Ni,j,k = 0, then
∂ℓ(D)
∂W M
(i,k) is always positive; iii) if P

j Ni,j,k > 0,

then
∂ℓ(D)
∂W M
(i,k) is negative when W M
(i,k) →−∞. Similarly, for all (j, k) pairs, i) if P

i Ni,j = 0, then

16


---Page Break---
∂ℓ(D)
∂W V
(j,k) is always 0; ii) if P

i Ni,j > 0 but P

i Ni,j,k = 0, then
∂ℓ(D)
∂W V
(j,k) is always positive; iii) if
P

i Ni,j,k > 0, then
∂ℓ(D)
∂W V
(j,k) is negative when W V
(j,k) →−∞.

Proof. We only prove the first part of this theorem, since the proof of the second part is almost
identical.

By the definition of the cross-entropy loss in Eq.(4), and the prediction weight vector in Eq.(5) for
our simplified model, the total cross-entropy loss of the model (with matrices W M, W V ) is

ℓ(D)
=
−
X

u∈D

X

n≥3

X

k
U(n+1,k) log ˆu(n+1),k

=
−
X

u∈D

X

n≥3

X

k
U(n+1,k) log
exp(W M
(un,k) + W V
(u2,k))
P

ℓexp(W M
(un,ℓ) + W V
(u2,ℓ))

=
−
X

u∈D

X

n≥3

X

k
U(n+1,k)
X

i,j
I[un = i, u2 = j] log
exp(W M
(i,k) + W V
(j,k))
P

ℓexp(W M
(i,ℓ) + W V
(j,ℓ))

=
−
X

i,j,k
Ni,j,k log
exp(W M
(i,k) + W V
(j,k))
P

ℓexp(W M
(i,ℓ) + W V
(j,ℓ))

=
−
X

i,j,k
Ni,j,k(W M
(i,k) + W V
(j,k)) +
X

i,j,k
Ni,j,k log

 X

ℓ
exp(W M
(i,ℓ) + W V
(j,ℓ))

!

=
−
X

i,j,k
Ni,j,k(W M
(i,k) + W V
(j,k)) +
X

i,j
Ni,j log

 X

ℓ
exp(W M
(i,ℓ) + W V
(j,ℓ))

!

.

Then we have that

∂ℓ(D)
∂W M
(i,k)
= −
X

j
Ni,j,k +
X

j
Ni,j
exp(W M
(i,k) + W V
(j,k))
P

ℓexp(W M
(i,ℓ) + W V
(j,ℓ)).
(7)

In case i), P

j Ni,j = 0 implies that P

j Ni,j,k = 0. Hence
∂ℓ(D)
∂W M
(i,k) is always 0.

In case ii), P

j Ni,j > 0 implies that the second term in Eq. (7) is positive, while P

j Ni,j,k = 0

implies that the first term in Eq. (7) is 0. Hence
∂ℓ(D)
∂W M
(i,k) is always positive.

In case iii), when P

j Ni,j > 0 and W M
(i,k) converges to −∞, then the second term in Eq. (7)

converges to 0, and it is smaller than P

j Ni,j,k > 0. Hence,
∂ℓ(D)
∂W M
(i,k) is negative when W M
(i,k)
converges to −∞.

D
The Reason of Choosing W M′ and W V ′

Note that in the Transformer layer, the output can be written as3

FFN

softmax
QK⊤

√dk


V + X

+ softmax
QK⊤

√dk


V + X.

Also noting that we have verified that the attention is concentrated at the second token, then we
let X2 = U(2,:)Wt, representing the token embedding of the target node, and Xn = U(n,:)Wt,
representing the token embedding of the current node.

3For simplicity, though the layer normalizations LN1,LN2 and LNt are used in our experiments, we omit
them in the equations in this section.

17


---Page Break---
(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes

(d) 400 Nodes
(e) 500 Nodes

Figure 7: Accuracy on the test datasets with embedding size d = 120.

Then we know that
ˆu(n+1) ≈
 
FFN
 
X2W V + Xn

+ X2W V + Xn

Wo.

Table 1: Cosine Similarity of FFN
 
X2W V + Xn

Wo and FFN
 
X2W V 
Wo + FFN (Xn) Wo

Graph
100 Nodes
200 Nodes
300 Nodes
400 Nodes
500 Nodes
Average Cosine Similarity
0.926
0.924
0.901
0.870
0.889

It is straightforward that XnWo contains the information of the current node, and X2W V Wo
contains the information of the target node. As for FFN
 
X2W V + Xn

Wo, we choose to use
its linear approximation as FFN
 
X2W V + Xn

Wo ≈FFN
 
X2W V 
Wo + FFN (Xn) Wo. As
shown in Table 1 (which takes average over all possible X2’s and Xn’s), this is a good approximation.
Then we can treat FFN
 
X2W V 
Wo as the information of the target node, and FFN (Xn) Wo as
the information of the current node.

Because of this, we let W M ′ be the matrix whose i-th row is FFN
 
e⊤
i Wt

Wo + (e⊤
i Wt)Wo,
where ei represents the one-hot column vector for node i (with dimension M = |V| + 1).
Note that in the simplified Transformer model of Theorem 3, there is no XnWo term, and
W M ′ is the same as matrix W M.
Similarly, we let W V ′ be the matrix whose ith row is
(e⊤
i Wt)W V Wo + FFN
 
(e⊤
i Wt)W V 
Wo, where ei represents the one-hot column vector for
node i (with dimension M = |V| + 1). Note that in the simplified Transformer model of Theorem 3,
there is no FFN
 
X2W V 
Wo term, and W V ′ is the same as matrix W V .

E
Complete Experimental Results on the Synthetic Datasets

In this section, we state the complete experimental results on the synthetic datasets. All these results
are conducted on a single A100 GPU.

• The accuracy results on all these tests are presented in Figure 7.
• The attention results on all the 1-layer and 1-head Transformers are presented in Figure 8.

• The results of W M ′’s are shown in Figure 9.
• The accuracy of the Transformers on the (s, t) pairs of the four different categories are
shown in Figure 10.

As we can see, all these results are consistent with our conclusions.

18


---Page Break---
(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes

(d) 400 Nodes
(e) 500 Nodes

Figure 8: The average attention in 1-layer and 1-head Transformers.

(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes

(d) 400 Nodes
(e) 500 Nodes

Figure 9: The first 20 rows and columns of W M ′ matrix in the 1-layer and 1-head Transformers (the
red boxes correspond to 1’s in the adjacency matrix A).

19


---Page Break---
(a) 100 Nodes
(b) 200 Nodes
(c) 300 Nodes

(d) 400 Nodes
(e) 500 Nodes

Figure 10: The accuracy for (s, t)’s with different degrees.

F
Path-planning in Blocksworld

To further validate the theoretical results in Section 3 and the practicability of the proposed path-
finding task, we consider Blocksworld benchmark [22]. Blocksworld is a scenario consisting of a set
of blocks identified by different colors. The blocks are either placed on table or on top of another
block and the task is to plan a block manipulation from the source state to the target state.

We formulate Blocksworld as a path-finding task. Here we construct a graph GBW for the case with
4 blocks, where each node represents a state of the blocks. For example, node 0 refers to the state
that “the red block is on top of the blue block, the blue block is on top of the orange block, the orange
block is on top of the yellow block, and the yellow block is on the table”. GBW is a directed graph
with 73 nodes, and the adjacency matrix of GBW is presented in Figure 11(a).

In the original Blocksworld task, the answer is a sequence of actions, which is equivalent to the
notion of edges in GBW . We reformulated it to let the model output a path from the given source
state to the given target state, only consisting of the nodes. This can be seen as a simplified version
and a pure planning task. We randomly select 80% of all node pairs for training and the rest 20% for
testing, and generate 50000 training sequences in the same format as introduced in Section 2. We
mainly use Transformers with 1 layer and 1 head for the convenience of visualization.

F.1
Results

We first present the accuracy results during training when using different embedding sizes d ∈
{30, 60, 120}. As shown in Figure 11(d), although a smaller embedding size results in a longer time
to converge, all models reach an accuracy near 100% at the end of the training.

Then, we use the same method introduced in Section 4 to visualize the attention map and the W M ′

matrix for the model with d = 120 after the entire iterations. In Figure 11(c), we can see that when
predicting the tokens on the path, almost all the attention weights are on the second token which
represents for the target node, demonstrating the capability of the model to learn a correct attention.
For adjacency matrix, we find that the W M ′ matrix in Figure 11(b) almost perfectly aligns to the real
adjacency matrix of GBW . And the weight gap (average edge weight minus average non-edge weight)
for all models keeps increasing in the training process until convergence, as shown in Figure 11(e).

In addition, we present the results related to observed reachability matrix in Figure 12. Figure 12(a)
shows the observed reachability in the training dataset. Although GBW is fully-connected, some
reachability are not observed since we request that all training data has no cycle. Specifically, in this

20


---Page Break---
(a) Adjacency Matrix of GBW

0
10
20
30
40
50
60
70
0

10

20

30

40

50

60

70

7.5
5.0
2.5
0.0
2.5
5.0
7.5

(b) The W M′ Matrix

0
2
4
6
8

0

2

4

6

8

0.0
0.2
0.4
0.6
0.8
1.0

(c) The Average Attention

0
500
1000
1500
2000
2500
3000
Iterations

0.0

0.2

0.4

0.6

0.8

1.0

Accuravy on Test Set

d=120
d=60
d=30

(d) Accuracy Curve

0
500
1000
1500
2000
2500
3000
Iterations

0

1

2

3

4

5

6

7

Average Weight Gap

d=120
d=60
d=30

(e) Average Weight Gap

Figure 11: Accuracy, attention, and adjacency matrix results for the experiment on Blocksworld
benchmark.

(a) Observed Reachability

0
10
20
30
40
50
60
70
0

10

20

30

40

50

60

70

5.0
2.5
0.0
2.5
5.0
7.5
10.0

(b) The W V ′ Matrix

0
500
1000
1500
2000
2500
3000
Iterations

1

0

1

2

3

4

Average Weight Gap

d=120
d=60
d=30

(c) Average Weight Gap

Figure 12: Experiment for reachability on Blocksworld benchmark.

case, each of the first 24 nodes is not observed reachable to any nodes other than itself. To validate
whether the Transformer has captured this information, we construct W V ′ matrix through the same
method presented in Section 4. As shown in Figure 12(b), the first 24 columns of the W V ′ matrix are
noticeably darker, which aligns with the observed reachability matrix in Figure 12(a). Furthermore,
we plot the gap between the average weight of W V ′ on observed reachability and the average weight
of W V ′ on non-observed reachability in Figure 12(c), and find that this gap keeps increasing for all
models. Since there does not exist any test pairs with degree 2 or more (as defined in Section 4), we
do not compare the accuracy between different degrees in Blocksworld.

In summary, our experimental results on the Blocksworld benchmark confirm our theoretical analyses
(Theorem 3) and empirical results on the synthetic data, and they at least partially explain the planning
capability of the Transformer on the Blocksworld scenario.

21


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: These claims can accurately reflect the paper’s contributions and scope. Details
are in Section 3 and Section 4.

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
Justification: We discuss some future research directions in Section 5. These are potential
limitations of our current work.

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

22


---Page Break---
Answer: [Yes]
Justification: All the assumptions are provided, and the complete proofs are stated in
Appendix B and Appendix C.
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
Justification: The code is uploaded in the supplementary material. All the information
needed to reproduce the main experimental results are provided.
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

23


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: The code is uploaded in the supplementary material.
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
Justification: Most of these details are explained in Section 2.
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
Justification: It is explained that all the accuracy results take average over 10000 trials.
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

24


---Page Break---
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
Justification: The computer resources is explained in Appendix E.
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
Justification: The research conducted in our paper conforms with the NeurIPS Code of
Ethics.
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
Justification: Our work mainly contributes to the theory of interpretability and explainability
in Transformers. We believe the societal impacts are limited.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

25


---Page Break---
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

Justification: We believe this paper poses no such risks.

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

Answer: [NA]

Justification: The paper does not use existing assets.

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

26


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.
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

27


---Page Break---
