Transformers Learn to Achieve Second-Order
Convergence Rates for In-Context Linear Regression

Deqing Fu
Tian-Qi Chen
Robin Jia
Vatsal Sharan
Department of Computer Science
University of Southern California
{deqingfu,tchen939,robinjia,vsharan}@usc.edu

Abstract

Transformers excel at in-context learning (ICL)—learning from demonstrations
without parameter updates—but how they do so remains a mystery. Recent work
suggests that Transformers may internally run Gradient Descent (GD), a first-order
optimization method, to perform ICL. In this paper, we instead demonstrate that
Transformers learn to approximate second-order optimization methods for ICL.
For in-context linear regression, Transformers share a similar convergence rate as
Iterative Newton’s Method; both are exponentially faster than GD. Empirically,
predictions from successive Transformer layers closely match different iterations of
Newton’s Method linearly, with each middle layer roughly computing 3 iterations;
thus, Transformers and Newton’s method converge at roughly the same rate. In
contrast, Gradient Descent converges exponentially more slowly. We also show
that Transformers can learn in-context on ill-conditioned data, a setting where
Gradient Descent struggles but Iterative Newton succeeds. Finally, to corroborate
our empirical findings, we prove that Transformers can implement k iterations of
Newton’s method with k + O(1) layers.

1
Introduction

Transformer neural networks [Vaswani et al., 2017] have become the default architecture for natural
language processing [Devlin et al., 2019, Brown et al., 2020, OpenAI, 2023]. As first demonstrated
by GPT-3 [Brown et al., 2020], Transformers excel at in-context learning (ICL)—learning from
prompts consisting of input-output pairs, without updating model parameters. Through in-context
learning, Transformer-based Large Language Models (LLMs) can achieve state-of-the-art few-shot
performance across a variety of downstream tasks [Rae et al., 2022, Smith et al., 2022, Thoppilan
et al., 2022, Chowdhery et al., 2022].

Given the importance of Transformers and ICL, many prior efforts have attempted to understand how
Transformers perform in-context learning. Prior work suggests Transformers can approximate various
linear functions well in-context [Garg et al., 2022]. Specifically to linear regression tasks, prior work
has tried to understand the ICL mechanism, and the dominant hypothesis is that Transformers learn
in-context by running optimization internally through gradient-based algorithms [von Oswald et al.,
2022, 2023, Ahn et al., 2023, Dai et al., 2023, Mahankali et al., 2024].

This paper presents strong evidence for a competing hypothesis: Transformers trained to perform
in-context linear regression learn a strategy much more similar to a second-order optimization method
than a first-order method like Gradient Descent (GD). In particular, Transformers approximately
implement a second-order method with a convergence rate very similar to Newton-Schulz’s Method,
also known as the Iterative Newton’s Method, which iteratively improves an estimate of the inverse of

Our codes are available at https://github.com/DeqingFu/transformers-icl-second-order.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the data matrix to compute the optimal weight vector. Across many Transformer layers, subsequent
layers approximately compute more and more iterations of Newton’s Method, with increasingly better
predictions; both eventually converge to the optimal minimum-norm solution found by ordinary least
squares (OLS). Interestingly, this mechanism is specific to Transformers: LSTMs do not learn these
same second-order methods, as their predictions do not even improve across layers.

We present both empirical and theoretical evidence for our claims. Empirically, Transformer layers
demonstrate a similar rate of convergence to the OLS solution as second-order methods such as
Iterative Newton, which is substantially faster than the rate of convergence of GD (Figure 2). The
predictions made by the Transformer at successive layers closely match the predictions made by
Iterative Newton after a proportional number of iterations, showing that they progress in similar
ways at the same rate. In contrast, to match the Transformer’s predictions after k layers, GD
would have to run for exponential in k many steps (Figure 3). Some individual Transformer layers
make progress equivalent to hundreds of GD steps: these layers must be doing something more
sophisticated than GD. Furthermore, a crucial aspect of second-order methods is that they can
handle ill-conditioned problems by correcting the curvature. We find that the convergence rate of
Transformers is not significantly affected by ill-conditioning, which again matches Iterative Newton
but not GD. To provide theoretical grounding to our empirical results, we show that Transformer
circuits can efficiently implement Iterative Newton: one transformer layer can compute one Newton
iteration (given O(1) pre/post-processing layers), and requires hidden states of dimension O(d) for a
d-dimensional linear regression problem. Overall, our work provides a mechanistic account of how
Transformers perform ICL that explains model behavior better than previous hypotheses, and hints at
why Transformers are well-suited for ICL relative to other architectures.

2
Related Work

In-context learning by large language models.
GPT-3 [Brown et al., 2020] first showed that
Transformer-based large language models can “learn” to perform new tasks from in-context demon-
strations (i.e., input-output pairs). Since then, a large body of work in NLP has studied in-context
learning, for instance by understanding how the choice and order of demonstrations affects results
[Lu et al., 2022, Liu et al., 2022, Rubin et al., 2022, Su et al., 2023, Chang and Jia, 2023, Nguyen and
Wong, 2023], studying the effect of label noise [Min et al., 2022c, Yoo et al., 2022, Wei et al., 2023],
and proposing methods to improve ICL accuracy [Zhao et al., 2021, Min et al., 2022a,b].

In-context learning beyond natural language.
Inspired by the phenomenon of ICL by large
language models, subsequent work has studied how Transformers learn in-context beyond NLP tasks.
Garg et al. [2022] first investigated Transformers’ ICL abilities for various classical machine learning
problems, including linear regression. We largely adopt their linear regression setup in this work.
Li et al. [2023] formalize in-context learning as an algorithm learning problem. Han et al. [2023]
suggests that Transformers learn in-context by performing Bayesian inference on prompts, which can
be asymptotically interpreted as kernel regression. Other work has analyzed how Transformers do
in-context classification [Tarzanagh et al., 2023a,b, Zhang et al., 2023], the role of pertaining data
[Raventós et al., 2023], and the relationship between model architecture and ICL [Lee et al., 2023].

Do Transformers implement Gradient Descent?
A growing body of work has suggested that
Transformers learn in-context by implementing gradient descent within their internal representations.
Akyürek et al. [2022] summarize operations that Transformers can implement, such as multiplication
and affine transformations, and show that Transformers can implement gradient descent for linear
regression using these operations. Concurrently, von Oswald et al. [2022] argue that Transformers
learn in-context via gradient descent, where one layer performs one gradient update. In subsequent
work, von Oswald et al. [2023] further argue that Transformers are strongly biased towards learning
to implement gradient-based optimization routines. Ahn et al. [2023] extend the work of von Oswald
et al. [2022] by showing Transformers can learn to implement preconditioned Gradient Descent,
where the pre-conditioner can adapt to the data. Bai et al. [2023] provide detailed constructions for
how Transformers can implement a range of learning algorithms via gradient descent. Finally, Dai
et al. [2023] conduct experiments on NLP tasks and conclude that Transformer-based language models
performing ICL behave similarly to models fine-tuned via gradient descent; however, concurrent work
[Shen et al., 2023b] argues that real-world LLMs do not perform ICL via gradient descent. Mahankali
et al. [2024] showed that implementing gradient descent is a global minima for single layer linear
self-attention. However, we study deeper models in this work, which can behave differently from

2


---Page Break---
single-layer models. In this paper, we argue that Transformers actually learn to perform in-context
learning by implementing a second-order optimization method, not gradient descent1.

Mechanistic interpretability for Transformers. Our work attempts to understand the mechanism
through which Transformers perform in-context learning. Prior work has studied other aspects of
Transformers’ internal mechanisms, including reverse-engineering language models [Wang et al.,
2022], the grokking phenomenon [Power et al., 2022, Nanda et al., 2023], manipulating attention
maps [Hassid et al., 2022], and circuit finding [Conmy et al., 2023].

Theoretical Expressivity of Transformers. Giannou et al. [2023] provide a construction of looped
transformers to implement Iterative Newton’s method for solving pseudo-inverse, and each Newton
iteration can be implemented by 13 looped Transformer layers. In contrast, our construction needs
only one Transformer layer to compute one Newton iteration.

3
Problem Setup

Transformers

𝑥!

!

𝑥!

"

⋮
𝑥!

#

………

!𝑦!
!𝑦"
!𝑦#
!𝑦#$!

In-Context 

Examples

𝑥"

!

𝑥"

"

⋮
𝑥"

#

𝑥$

!

𝑥$

"

⋮
𝑥$

#

𝑥$%!

!

𝑥$%!

"

⋮
𝑥$%!

#

0

⋮
0
𝑦!

0

⋮
0
𝑦"

0

⋮
0
𝑦#

0

⋮
0
𝑦#$!

Figure 1: Illustration of how Transformers are
trained to do in-context linear regression.

In this paper, we focus on the following linear regres-
sion task. The task involves n examples {xi, yi}n
i=1
where xi ∈Rd and yi ∈R. The examples are
generated from the following data generating distri-
bution PD, parameterized by a distribution D over
(d × d) positive semi-definite matrices. For each se-
quence of n in-context examples, we first sample a
ground-truth weight vector w⋆i.i.d.
∼N(0, I) ∈Rd

and a matrix Σ
i.i.d.
∼D. For i ∈[n], we sample each
xi
i.i.d.
∼N(0, Σ). The label yi for each xi is given by
yi = w⋆⊤xi. Note that for much of our experiments
D is only supported on the identity matrix I and hence Σ = I, but we also consider some distribu-
tions over ill-conditioned matrices, which give rise to ill-conditioned regression problems. Most of
our results are on this noiseless setup and results with the noisy setup are in Appendix A.3.2.

3.1
Standard Methods for Solving Linear Regression

Our central research question is:

What convergence rate does the algorithm Transformers learn for linear regression achieve?

To investigate this question, we first discuss various known algorithms for linear regression. We then
compare them with Transformers empirically in §4 and theoretically in §5, to evaluate if Transformers
are more similar to first-order or second-order methods. We care particularly about algorithms’
convergence rates (the number of steps required to reach an ϵ error).

For any time step t, let X(t) = [x1
· · ·
xt]⊤be the data matrix and y(t) = [y1
· · ·
yt]⊤be
the labels for all the datapoints seen so far. Note that since t can be smaller than the data dimension d,
X(t) can be singular. We now consider various algorithms for making predictions for xt+1 based on
X(t) and y(t). When it is clear from context, we drop the superscript and refer to X(t) and y(t) as
X and y, where X and y correspond to all the datapoints seen so far.

Ordinary Least Squares. This method finds the minimum-norm solution to the objective:

L(w | X, y) = 1

2n∥y −Xw∥2
2.
(1)

The Ordinary Least Squares (OLS) solution has a closed form given by the Normal Equations:

ˆwOLS = (X⊤X)†X⊤y
(2)

1After an initial version of this paper, Vladymyrov et al. [2024] found that a variant of Gradient Descent
can mimic Iterative Newton by approximating the inverse implicitly and getting second-order rates, which also
supports our claim.

3


---Page Break---
where S := X⊤X and S† is the pseudo-inverse [Moore, 1920] of S.

Gradient Descent. Gradient descent (GD) is a first-order method which finds the weight vector ˆwGD
with initialization ˆwGD
0
= 0 using the iterative update rule:

ˆwGD
k+1 = ˆwGD
k
−η∇wL( ˆwGD
k
| X, y).
(3)

It is known that GD requires O (κ(S) log(1/ϵ)) steps to converge to an ϵ error where κ(S) = λmax(S)

λmin(S)
is the condition number. Thus, when κ(S) is large, GD converges slowly [Boyd and Vandenberghe,
2004].

Online Gradient Descent. While GD computes the gradient with respect to the full data matrix X at
each iteration, Online Gradient Descent (OGD) is an online algorithm that only computes gradients
on the newly received data point {xk, yk} at step k:

ˆwOGD
k+1 = ˆwOGD
k
−ηk∇wL( ˆwOGD
k
| xk, yk).
(4)

Picking ηk =
1
∥xk∥2
2 ensures that the new weight vector ˆwOGD
k+1 makes zero error on {xk, yk}.

Iterative Newton’s Method. This is a second-order method which finds the weight vector ˆwNewton
by iteratively apply Newton’s method to finding the pseudo inverse of S = X⊤X [Schulz, 1933,
Ben-Israel, 1965].

M0 = αS, where α =
2
∥SS⊤∥2
,
ˆwNewton
0
= M0X⊤y,

Mk+1 = 2Mk −MkSMk,
ˆwNewton
k+1
= Mk+1X⊤y.
(5)

This computes an approximation of the psuedo inverse using the moments of S. In contrast to GD,
the Iterative Newton’s method only requires O(log κ(S) + log log(1/ϵ)) steps to converge to an
ϵ error [Soderstrom and Stewart, 1974, Pan and Schreiber, 1991]. Note that this is exponentially
faster than the convergence rate of GD. We discuss additional algorithms such as Conjugate Gradient,
BFGS, and L-BFGS in the Appendix A.2.3.

3.2
Solving Linear Regression with Transformers

We will use neural network models such as Transformers to solve this linear regression task. As
shown in Figure 1, at time step t + 1, the model sees the first t in-context examples {xi, yi}t
i=1, and
then makes predictions for xt+1, whose label yt+1 is not observed by the Transformers model.

We randomly initialize our models and then train them on the linear regression task to make predictions
for every number of in-context examples t, where t ∈[n]. Training and test data are both drawn
from PD. To make the input prompts contain both xi and yi, we follow same the setup as Garg
et al. [2022]’s to zero-pad yi’s, and use the same GPT-2 model [Radford et al., 2019] with softmax
activation and causal attention mask (discussed later in Definition 3.1).

We now present the key mathematical details for the Transformer architecture, and how they can be
used for in-context learning. First, the causal attention mask enforces that attention heads can only
attend to hidden states of previous time steps, and is defined as follows.

Definition 3.1 (Causal Attention Layer). A causal attention layer with M heads and activation
function σ is denoted as Attn on any input sequence H = [h1, · · · , hN] ∈RD×N, where D is the
dimension of hidden states and N is the sequence length. In the vector form,

˜ht = [Attn(H)]t = ht +

M
X

m=1

t
X

j=1
σ (⟨Qmht, Kmhj⟩) · Vmhj.
(6)

Vaswani et al. [2017] originally proposed the Transformer architecture with the Softmax activation
function for the attention layers. Later works have found that replacing Softmax(·) with 1

t ReLU(·)
does not hurt model performance [Cai et al., 2022, Shen et al., 2023a, Wortsman et al., 2023]. The
Transformers architecture is defined by putting together attention layers with feed forward layers:

4


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

10
4

10
3

10
2

10
1

100

Errors

Transformer Errors v.s. # Layers

# In-Context Examples = 05
# In-Context Examples = 10
# In-Context Examples = 15
# In-Context Examples = 20
# In-Context Examples = 22
# In-Context Examples = 25
# In-Context Examples = 30
# In-Context Examples = 35

(a) Transformers

1
3
5
7
9
11
13
15
17
19
21
23
Iterative Newton Steps

10
4

10
3

10
2

10
1

100

Errors

Iterative Newton Errors v.s. # Steps

# In-Context Examples = 05
# In-Context Examples = 10
# In-Context Examples = 15
# In-Context Examples = 20
# In-Context Examples = 22
# In-Context Examples = 25
# In-Context Examples = 30
# In-Context Examples = 35

(b) Iterative Newton’s Method

1
30
80
160
Gradient Descent Steps

10
4

10
3

10
2

10
1

100

Errors

Gradient Descent Errors v.s. # Steps

# In-Context Examples = 05
# In-Context Examples = 10
# In-Context Examples = 15
# In-Context Examples = 20
# In-Context Examples = 22
# In-Context Examples = 25
# In-Context Examples = 30
# In-Context Examples = 35

(c) Gradient Descent

Figure 2: Convergence of Algorithms. Similar to Iterative Newton and GD, Transformer’s perfor-
mance improve over the layer index ℓ. When n > d, the Transformer model, from layers 3 to 8,
demonstrates a superlinear convergence rate, similar to Iterative Newton, while GD, with fixed step
size, is sublinear. Later layers of Transformers show a slower convergence rate, and we hypothesize
they have little incentive to implement the algorithm precisely since the error is already very small. A
24-layer Transformer model exhibits the same superlinear convergence (Figure 25 in §A.4.2).

Definition 3.2 (Transformers). An L-layer decoder-based transformer with Causal Attention Layers
is denoted as TFθ and is a composition of a MLP Layer (with a skip connection) and a Causal
Attention Layers. For input sequence H(0), the transformers ℓ-th hidden layer is given by

TFℓ
θ(H(0)) := H(ℓ) = MLPθ(ℓ)
mlp


Attnθ(ℓ)
attn(H(ℓ−1))

.

where θ = {θ(ℓ)
mlp, θ(ℓ)
attn}L
ℓ=1 and θ(ℓ)
attn = {Q(ℓ)
m , K(ℓ)
m , V (ℓ)
m }M
m=1 has M heads at layer ℓ.

In particular for the linear regression task, Transformers perform in-context learning as follows
Definition 3.3 (Transformers for Linear Regression). Given in-context examples {x1, y1, . . . , xt, yt},
Transformers make predictions on a query example xt+1 through a readout layer parameterized as
θreadout = {u, v}, and the prediction ˆyTF
t+1 is given by

ˆyTF
t+1 := ReadOut
h
TFL
θ ({x1, y1, · · · , xt, yt, xt+1})
|
{z
}
H(L)

i
= u⊤H(L)
:,2t+1 + v.

To compare the rate of convergence of iterative algorithms to that of Transformers, we treat the layer
index ℓof Transformers as analogous to the iterative step k of algorithms discussed in §3.1. Note
that for Transformers, we need to re-train the ReadOut layer for every layer index ℓso that they can
improve progressively (see §4.1 and for experimental details) for linear regression tasks.

3.3
Measuring Algorithmic Similarity

We propose two metrics to measure the similarity between linear regression algorithms.

Similarity of Errors. This metric aims to measure similarity of algorithms through comparing
prediction errors. For a linear regression algorithm A, let A(xt+1 | {xi, yi}t
i=1) denote its prediction
on the (t + 1)-th in-context example xt+1 after observing the first t examples (see Figure 1). We
write A(xt+1) := A(xt+1 | {xi, yi}t
i=1) for brevity. Errors (i.e., residuals) on the sequence are:2

E(A | {xi, yi}n+1
i=1 ) =
h
A(x2) −y2, · · · , A(xn+1) −yn+1
i⊤
.

The similarity of errors for two algorithms Aa and Ab is the expected cosine similarity of their errors
on a randomly sampled data sequence:

SimE(Aa, Ab) =
E
{xi,yi}n+1
i=1 ∼PD

"

C

E(Aa|{xi, yi}n+1
i=1 ), E(Ab|{xi, yi}n+1
i=1 )
#

.

2the indices start from 2 to n + 1 because we evaluate all cases where t can choose from 1, · · · , n.

5


---Page Break---
Here C(u, v) =
⟨u,v⟩
∥u∥2∥v∥2 is the cosine similarity, n is the total number of in-context examples, and
PD is the data generation process discussed previously.

Similarity of Induced Weights. All standard algorithms for linear regression estimate a weight
vector ˆw. While neural ICL models like Transformers do not explicitly learn such a weight vector,
similar to Akyürek et al. [2022], we can induce an implicit weight vector ˜w learned by any algorithm
A by fitting a weight vector to its predictions. We can then measure similarity of algorithms by
comparing the induced ˜w. To do this, for any fixed sequence of t in-context examples {xi, yi}t
i=1,
we sample T ≫d query examples ˜xk
i.i.d.
∼N(0, Σ), where k ∈[T]. For this fixed sequence of
in-context examples {xi, yi}t
i=1, we create T in-context prediction tasks and use the algorithm A to
make predictions A(˜xk | {xi, yi}t
i=1). We define the induced data matrix and labels as

˜
X =





˜x⊤
1...
˜x⊤
T




˜Y =





A(˜x1 | {xi, yi}t
i=1)
...
A(˜xT | {xi, yi}t
i=1)



.
(7)

The induced weight vector for A and these t examples is:

˜wt(A) := ˜wt(A | {xi, yi}t
i=1) = ( ˜
X⊤˜
X)−1 ˜
X⊤˜Y .
(8)

The similarity of induced weights between two algorithms Aa and Ab is the expected average cosine
similarity3 of induced weights ˜wt(Aa) and ˜wt(Ab) over all possible 1 ≤t ≤n, on a randomly
sampled data sequence:

SimW(Aa, Ab) =
E
{xi,yi}n
i=1∼PD

"
1
n

n
X

t=1
C

˜wt(Aa|{xi, yi}t
i=1), ˜wt(Ab|{xi, yi}t
i=1))
#

.

Matching steps between algorithms. Each algorithm converges to its predictions after several steps
— for example the number of iterations for Iterative Newton and GD, and the number of layers for
Transformers (see Section 4.1). When comparing two algorithms, given a choice of steps for the first
algorithm, we match it with the steps for the second algorithm that maximize similarity.
Definition 3.4 (Best-matching Steps). Let M be the metric for evaluating similarities between two
algorithms Aa and Ab, which have steps pa ∈[0, Ta] and pb ∈[0, Tb], respectively. For a given
choice of pa, we define the best-matching number of steps of algorithm Ab for Aa as:

pM
b (pa) := arg max
pb∈[0,Tb]
M(Aa(· | pa), Ab(· | pb)).
(9)

In our experiments, we chose Ta, Tb be large enough integers so the algorithms converge. The
matching processes can be visualized as heatmaps as shown in Figure 3, where best-matching steps
are highlighted. This enables us to compare the rate of convergence of algorithms. In particular, if
two algorithms converge at the same rate, the best matching steps between the two algorithms should
follow a linear trend. We will discuss these results in §4. See Figure 26 on how best-matching steps
help compare the convergence rates.

4
Experimental Evidence

We primarily study the Transformers-based GPT-2 model with 12 layers and 8 heads per layer.
Alternative configurations with fewer heads per layer, or with more layers, also support our findings;
we defer them to §A.4.1 and §A.4.2. We initially focus on isotropic cases where Σ = I and later
consider ill-conditioned Σ in §4.3. Our training setup is exactly the same as Garg et al. [2022]:
models are trained with at most n = 40 in-context examples for d = 20 (with the same learning rate,
batch size etc.).

We claim that Transformers learn high-order optimization methods in-context. We provide evidence
that Transformers improve themselves with more layers in §4.1; Transformers share the same rate of
convergence as Iterative Newton, exponentially faster than that of GD, in §4.2; and they also perform
well on ill-conditioned problems in §4.3. Finally, we contrast Transformers with LSTMs in §4.5.

3Alternative metrics such as ℓ2 distance gives the same observation. Here cosine similarity is better since
errors usually have small magnitudes, and directions of induced weights are meaningful.

6


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.920
.920
.912

.876
.876
.929

.927

.916

.923

.916

.926

.927

.919
.949

.954

.953

.979

.980

.979

.988

.988

.988

.992

.993
.993
.993
.993

.992
.993
.994
.994

.993
.993
.993

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.954
.953
.870
.770
.692
.645
.578
.577
.733
.831
.905
.946
.795
.883
.944

.973
.974
.974

.982
.982
.982

.986
.986
.986
.986
.986
.986
.987
.987
.986
.986
.986
.986

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 3: Heatmaps of Similarity. The best matching steps are highlighted in yellow. Transformers
layers show a linear trend with Iterative Newton steps but an exponential trend with GD. This suggests
Transformers and Iterative Newton have the same convergence rate that is exponentially faster than
GD. See Figure 10 for an additional heatmap where GD’s steps are shown in log scale: on that plot
there is a linear correspondence between Transformers and GD’s steps. This further strengthens the
claim that Transformers have an exponentially faster rate of convergence than GD.

4.1
Transformers improve progressively over layers

Many known algorithms for linear regression, including GD, OGD, and Iterative Newton, are iterative:
their performance progressively improves as they perform more iterations, eventually converging to
a final solution. How can a Transformer implement such an iterative algorithm? von Oswald et al.
[2022] propose that deeper layers of the Transformer may correspond to more iterations; in particular,
they show that there exist Transformer parameters such that each attention layer performs one step of
GD.

Following this intuition, we first investigate whether the predictions of a trained Transformer improve
as the layer index ℓincreases. For each layer of hidden states H(ℓ) (see Definition 3.2), we re-train
the ReadOut to predict yt for each t; the new predictions are given by ReadOut(ℓ) 
H(ℓ)
. Thus
for each input prompt, there are L Transformer predictions parameterized by layer index ℓ. All
parameters besides the ReadOut layer parameters are kept frozen.

As shown in Figure 2(a) (and Figure 7(a) in the Appendix), as we increase the layer index ℓ, the
prediction performance improves progressively. Hence, Transformers progressively improve their
predictions over layers ℓ, similar to how iterative algorithms improve over steps. Such observations
are consistent with language tasks where Transformers-based language models also improve their
predictions along with layer progressions [Geva et al., 2022, Chuang et al., 2023].

4.2
Transformers are more similar to second-order methods, such as Iterative Newton

We now test the more specific hypothesis that the iterative updates performed across Transformer
layers are similar to the iterative updates for known iterative algorithms. First, Figure 2 shows that
the middle layers of Transformers converge at a rate similar to Iterative Newton, and faster than GD.
In particular, the Transformer and Iterative Newton both converge at a superlinear rate, while GD
converges at a sublinear rate.

Next, we analyze whether each layer ℓof the Transformer corresponds to performing k steps of
some iterative algorithm, for some k depending on ℓ. We focus here on GD and Iterative Newton’s
Method; we will discuss online algorithms in Section 4.5, and additional optimization methods in
Appendix A.2.3. We will discuss results on noisy linear regression tasks in Appendix A.3.2.

For each layer ℓof the Transformer, we measure the best-matching similarity (see Def. 3.4) with
candidate iterative algorithms with the optimal choice of the number of steps k. As shown in Figure 3,

7


---Page Break---
100
101
102
103
Steps

10
4

10
3

10
2

10
1

100

Errors

Convergence on Ill-Conditioned Data

Transformer
Iterative Newton
Gradient Descent

Figure 4: Transformers perfor-
mance on ill-conditioned data.
Given 40 in-context examples,
Transformers and Iterative New-
ton converge similarly and they
both can converge to the OLS so-
lution quickly whereas GD suf-
fers.

0
5
10
15
20
25
30
35
40
Number of In-Context Examples

0.0

0.2

0.4

0.6

0.8

1.0

Mean Square Errors

OGD
Transformers
LSTM
Newton's Method (5 steps)

19
15
11
7
3
Time Stamp Gap

0.00

0.05

0.10

0.15

0.20

0.25

Mean Square Errors

OGD
Transformers
LSTM
Newton's Method (5 steps)

Figure 5: In the left figure, we measure model predictions with
normalized MSE. Though LSTM is seemingly most similar to
Newton’s Method with only 5 steps, neither algorithm converges
yet. OGD also has a similar trend as LSTM. In the right figure,
we measure the model’s error rate on example xn−g after seeing
n examples, for different values of the time stamp gap g (see
Appendix A.6), and find both Transformers and not-converged
Newton have better memorization than LSTM and OGD.

the Transformer has very high error similarity with Iterative Newton’s method at all layers. Moreover,
we see a clear linear trend between layer 3 and layer 9 of the Transformer, where each layer appears
to compute roughly 3 additional iterations of Iterative Newton’s method. This trend only stops at the
last few layers because both algorithms converge to the OLS solution; Newton is known to converge
to OLS (see §3.1), and we verify in Appendix A.2 that the last few layers of the Transformer also
basically compute OLS (see Figure 14 in the Appendix). We observe the same trends when using
similarity of induced weights as our similarity metric (see Figure 9 in the Appendix). Figure 11 in the
Appendix shows that there is a similar linear trend between Transformer and BFGS, an alternative
quasi-Newton method. This is perhaps not surprising, given that BFGS also gets a superlinear
convergence rate for linear regression Nocedal and Wright [1999]. Thus, we do not claim that
Transformers specifically implement Iterative Newton, only that they (approximately) implement
some second-order method.

In contrast, even though GD has a comparable similarity with the Transformers at later layers, their
best matching follows an exponential trend. As discussed in the Section 3.1, for well-conditioned
problems where κ ≈1, to achieve ϵ error, the rate of convergence of GD is O(log(1/ϵ)) while the rate
of convergence of Iterative Newton is O(log log(1/ϵ)). Therefore the rate of convergence of Iterative
Newton is exponentially faster than GD. Transformer’s linear correspondence with Iterative Newton
and its exponential correspondence with GD provides strong evidence that the rate of convergence of
Transformers is similar to Iterative Newton, i.e., O(log log(1/ϵ)). We also note that it is not possible
to significantly improve GD’s convergence rate without using second-order methods: Nemirovski
and Yudin [1983] showed a Ω
 
log(1/ϵ)

lower bound on the convergence rate of gradient-based
methods for smooth and strongly convex problems, and Arjevani et al. [2016] shows a similar lower
bound specifically for quadratic problems. In the Appendix, we show that limited-memory BFGS
Liu and Nocedal [1989] and conjugate gradient (see Figure 12), which do not use full-second order
information, also converge slower than Transformers. This provides further evidence for the usage of
second-order information by Transformers. We also show more evidence by investigating alternative
function classes such as linear regression with noises in Appendix A.3.2 and 2-layer neural network
with ReLU or Tanh activation function in Appendix A.3.3.

Overall, we conclude that a Transformer trained to perform in-context linear regression learns to
implement an algorithm that is very similar to second-order methods, such as Iterative Newton’s
method, not GD. Starting at layer 3, subsequent layers of the Transformer compute more and more
iterations of Iterative Newton’s method. This algorithm successfully solves the linear regression
problem, as it converges to the optimal OLS solution in the final layers.

4.3
Transformers perform well on ill-conditioned data

We repeat the same experiments with data xi
i.i.d.
∼N(0, Σ) sampled from an ill-condition covariance
matrix Σ with condition number κ(Σ) = 100, and eigenbasis chosen uniformly at random. The first

8


---Page Break---
0
5
10
15
20
25
30
35
40
in-context examples

0.0

0.2

0.4

0.6

0.8

1.0

1.2

squared error

Transformers with Various Hidden Sizes

Transformers (Hidden Size=8)
Transformers (Hidden Size=16)
Transformers (Hidden Size=32)
Transformers (Hidden Size=64)
Least Squares

Figure 6: Ablation on Transformer’s Hidden Size. For linear regression problems with d = 20,
Transformers need O(d) hidden dimension to mimic OLS solutions.

d/2 eigenvalues of Σ are 100, and the last d/2 are 1. Note that choosing the eigenbasis uniformly at
random for each sequence ensures that there is a different covariance matrix Σ for each sequence of
datapoints.

As shown in Figure 4, the Transformer model’s performance still closely matches Iterative Newton’s
Method with 21 iterations, same as when Σ = I (see layer 10-12 in Figure 3). The convergence of
second-order methods has a mild logarithmic dependence on the condition number since they correct
for the curvature. On the other hand, GD’s convergence is affected polynomially by conditioning.
As κ(Σ) increase from 1 to 100, the number steps required for GD’s convergence increases signifi-
cantly (see Fig. 4 where GD requires 2,000 steps to converge), making it impossible for a 12-layer
Transformers to implement these many gradient updates. We also note that preconditioning the
data by (X⊤X)† can make the data well-conditioned, but since the eigenbasis is chosen uniformly
at random, with high probability there is no sparse pre-conditioner or any fixed pre-conditioner
which works across the data distribution. Computing (X⊤X)† appears to be as hard as computing
the OLS solution (Eq. 1)—in fact Sharan et al. [2019] conjecture that first-order methods such as
gradient descent and its variants cannot avoid polynomial dependencies in condition number in the
ill-conditioned case.4 See Appendix A.3.1 for detailed experiments on ill-conditioned problems.
These experiments further strengthen our thesis that Transformers learn to perform second-order
optimization methods in-context, not first-order methods such as GD.

4.4
Transformers Require O(d) Hidden Dimension

We ablate 12-layer 1-head Transformers with various hidden sizes on d = 20 problems. As shown
in Figure 6, we observe that Transformers can mimic OLS solution when the hidden size is 32 or
64, but fail with smaller sizes. This resonates with our theoretical results on O(d) hidden dimension
in Theorem 5.1, and in this case, the theorem ensures a construction of transformers to implement
Iterative Newton’s method.

4.5
LSTM is more similar to OGD than Transformers

As discussed in §A.1, LSTM is an alternative auto-regressive model widely used before the introduc-
tion of Transformers. Thus, a natural research question is: If Transformers can learn in-context, can
LSTMs do so as well? If so, do they learn the same algorithms? To answer this question, we train a
LSTM model in an identical manner to the Transformers studied in the previous sections.

Figure 5 plots the error of Transformers, LSTMs, and other standard methods as a function of the
number of in-context (i.e., training) examples provided. While LSTMs can also learn linear regression
in-context, they have much higher mean-squared error than Transformers. Their error rate is similar
to Iterative Newton’s Method after only 5 iterations, a point where it is far from converging to the
OLS solution. Finally, we show that LSTMs behave more like an online learning algorithm than
Transformers. In particular, its predictions are biased towards getting more recent training examples
correct, as opposed to earlier examples, as shown in Figure 5. This property makes LSTMs similar to

4Regarding preconditioning, we also note that—even for well-conditioned instances—preconditioned GD
still gets a linear rate of convergence, whereas Transformers and Iterative Newton get superlinear rates.

9


---Page Break---
online GD. In contrast, five steps of Newton’s method has the same error on average for recent and
early examples, showing that the LSTM implements a very different algorithm from a few iterations
of Newton. We hypothesize that since LSTMs have limited memory, they must learn in a roughly
online fashion; in contrast, Transformer’s attention heads can access the entire sequence of past
examples, enabling it to learn more complex algorithms. See §A.1 for more discussions.

5
Theoretical Justification

Our empirical evidence demonstrates that Transformers behave much more similarly to Iterative
Newton’s than to GD. Iterative Newton is a second-order optimization method, and is algorithmically
more involved than GD. We begin by first examining this difference in complexity. As discussed in
Section 3, the updates for Iterative Newton are of the form,

ˆwNewton
k+1
= Mk+1X⊤y
where Mk+1 = 2Mk −MkSMk
(10)
and M0 = αS for some α > 0. We can express Mk in terms of powers of S by expanding iteratively,
for example M1 = 2αS −4α2S3, M2 = 4αS −12α2S3 + 16α3S5 −16α4S7, and in general
Mk = P2k+1−1
s=1
βsSs for some βs ∈R (see Appendix B.3 for detailed calculations). Note that k
steps of Iterative Newton’s requires computing Ω(2k) moments of S. Let us contrast this with GD.
GD updates for linear regression take the form,

ˆwGD
k+1 = ˆwGD
k
−η(S ˆwGD
k
−X⊤y).
(11)

Like Iterative Newton, we can express ˆwGD
k
in terms of powers of S and X⊤y. However, after k
steps of GD, the highest power of S is only O(k). This exponential separation is consistent with the
exponential gap in terms of the parameter dependence in the convergence rate—O (κ(S) log(1/ϵ))
for GD vs. O(log κ(S) + log log(1/ϵ)) for Iterative Newton. Therefore, a natural question is
whether Transformers can actually as complicated of a method such as Iterative Newton with only
polynomially many layers? Theorem 5.1 shows that this is indeed possible.
Theorem 5.1. For any k, there exist Transformer weights such that on any set of in-context examples
{xi, yi}n
i=1 and test point xtest, the Transformer predicts on xtest using x⊤
test ˆwNewton
k
. Here
ˆwNewton
k
are the Iterative Newton updates given by ˆwNewton
k
= MkX⊤y where Mj is updated as
Mj = 2Mj−1 −Mj−1SMj−1, 1 ≤j ≤k,
M0 = αS,

for some α > 0 and S = X⊤X. The dimensionality of the hidden layers is O(d), and the number
of layers is k + 8. One transformer layer computes one Newton iteration. 3 initial transformer layers
are needed for initializing M0 and 5 layers at the end are needed to read out predictions from the
computed pseudo-inverse Mk.

We note that our proof uses full attention instead of causal attention and ReLU activations for the
self-attention layers. The definitions of these and the full proof appear in Appendix B.

6
Conclusion and Discussion

In this work, we studied how Transformers perform in-context learning for linear regression. In
contrast with the hypothesis that Transformers learn in-context by implementing gradient descent,
our experimental results show that different Transformer layers match iterations of Iterative Newton
linearly and Gradient Descent exponentially. This suggests that Transformers share a similar rate
of convergence to Iterative Newton but not to Gradient Descent. Moreover, Transformers can
perform well empirically on ill-conditioned linear regression, whereas first-order methods such as
Gradient Descent struggle. This empirical evidence — when combined with existing lower bounds in
optimization — suggests that Transformers use second-order information for solving linear regression,
and we also prove that Transformers can indeed represent second-order methods.

An interesting direction is to explore a wider range of second-order methods that Transformers can
implement. It also seems promising to extend our analysis to classification problems, especially
given recent work showing that Transformers resemble SVMs in classification tasks [Li et al.,
2023, Tarzanagh et al., 2023a]. Finally, a natural question is to understand the differences in
the model architecture that make Transformers better in-context learners than LSTMs. Based on
our investigations with LSTMs, we hypothesize that Transformers can implement more powerful
algorithms because of having access to a longer history of examples. Investigating the role of this
additional memory in learning appears to be an intriguing direction.

10


---Page Break---
Acknowledgement

We would like to thank the USC NLP Group and Center for AI Safety for providing compute
resources. DF would like to thank Oliver Liu and Ameya Godbole for their extensive discussions. DF
and RJ were supported by a Google Research Scholar Award. RJ was also supported by an Open
Philanthropy research grant. VS was supported by NSF CAREER Award CCF-2239265 and an
Amazon Research Award.

References

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement
preconditioned gradient descent for in-context learning. ArXiv, abs/2306.00297, 2023. 1, 2

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning
algorithm is in-context learning? investigations with linear models. ArXiv, abs/2211.15661, 2022.
2, 3.3, B.1, B.4, B.2, B.5

Yossi Arjevani, Shai Shalev-Shwartz, and Ohad Shamir. On lower and upper bounds in smooth and
strongly convex optimization. Journal of Machine Learning Research, 17(126):1–51, 2016. URL
http://jmlr.org/papers/v17/15-106.html. 4.2

Yu Bai, Fan Chen, Haiquan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection. ArXiv, abs/2306.04637, 2023. 2

Adi Ben-Israel. An iterative method for computing the generalized inverse of an arbitrary matrix.
Mathematics of Computation, 19(91):452–455, 1965. ISSN 00255718, 10886842. URL http:
//www.jstor.org/stable/2003676. 3.1

Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

3.1

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jef-
frey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Ben-
jamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and
Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell,
M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33,
pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/
paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
1,
2

Han Cai, Chuang Gan, and Song Han. Efficientvit: Enhanced linear attention for high-resolution
low-computation visual recognition. ArXiv, abs/2205.14756, 2022. 3.2

Ting-Yun Chang and Robin Jia. Data curation alone can stabilize in-context learning. In Proceedings
of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 8123–8144, Toronto, Canada, July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.acl-long.452. URL https://aclanthology.org/2023.acl-long.
452. 2

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh,
Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam
Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James
Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Lev-
skaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin
Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret
Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick,
Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica
Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan

11


---Page Break---
Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck,
Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022.
1

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola:
Decoding by contrasting layers improves factuality in large language models, 2023. 4.1

Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-
Alonso. Towards automated circuit discovery for mechanistic interpretability, 2023. 2

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. Why can gpt learn in-context?
language models secretly perform gradient descent as meta-optimizers. ArXiv, abs/2212.10559,
2023. 1, 2

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep
bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota,
June 2019. Association for Computational Linguistics.
doi: 10.18653/v1/N19-1423.
URL
https://aclanthology.org/N19-1423. 1

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn
in-context? a case study of simple function classes. ArXiv, abs/2208.01066, 2022. 1, 2, 3.2, 4,
A.3.3, D

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build
predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing, pages 30–45, Abu Dhabi, United Arab
Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.
emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main.3. 4.1

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris
Papailiopoulos. Looped transformers as programmable computers. In Andreas Krause, Emma
Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors,
Proceedings of the 40th International Conference on Machine Learning, volume 202 of Pro-
ceedings of Machine Learning Research, pages 11398–11442. PMLR, 23–29 Jul 2023. URL
https://proceedings.mlr.press/v202/giannou23a.html. 2, B.5

Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-context learning of large language models explained
as kernel regression, 2023. 2

Michael Hassid, Hao Peng, Daniel Rotem, Jungo Kasai, Ivan Montero, Noah A. Smith, and Roy
Schwartz. How much does attention actually attend? questioning the importance of attention in
pretrained transformers, 2022. 2

Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory. Neural Computation, 9
(8):1735–1780, 11 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL https:
//doi.org/10.1162/neco.1997.9.8.1735. A.1

Ivan Lee, Nan Jiang, and Taylor Berg-Kirkpatrick. Exploring the relationship between model
architecture and in-context learning ability, 2023. 2

Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers
as algorithms: Generalization and stability in in-context learning. In International Conference on
Machine Learning, 2023. 2, 6

Dong C. Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization.
Mathematical Programming, 45:503–528, 1989. URL https://api.semanticscholar.org/
CorpusID:5681609. 4.2

12


---Page Break---
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen.
What makes good in-context examples for GPT-3?
In Proceedings of Deep Learning In-
side Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for
Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online, May 2022. As-
sociation for Computational Linguistics.
doi: 10.18653/v1/2022.deelio-1.10.
URL https:
//aclanthology.org/2022.deelio-1.10. 2

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered
prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of
the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.acl-long.556. 2

Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably
the optimal in-context learner with one layer of linear self-attention. In The Twelfth International
Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=
8p3fu56lKc. 1, 2

Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language
model prompting for few-shot text classification. In Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pages 5316–5330, Dublin,
Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.
365. URL https://aclanthology.org/2022.acl-long.365. 2

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in
context. In Proceedings of the 2022 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United
States, July 2022b. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.
201. URL https://aclanthology.org/2022.naacl-main.201. 2

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke
Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In
Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages
11048–11064, Abu Dhabi, United Arab Emirates, December 2022c. Association for Computational
Linguistics. doi: 10.18653/v1/2022.emnlp-main.759. URL https://aclanthology.org/2022.
emnlp-main.759. 2

E.H Moore. On the reciprocal of the general algebraic matrix. Bulletin of American Mathematical
Society, 26:394–395, 1920. 3.1

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures
for grokking via mechanistic interpretability, 2023. 2

A.S. Nemirovski and D.B Yudin. Problem complexity and method efficiency in optimization. 1983.

4.2

Tai Nguyen and Eric Wong.
In-context example selection with influences.
arXiv preprint
arXiv:2302.11042, 2023. 2

Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 1999. 4.2, A.2.3

OpenAI. Gpt-4 technical report, 2023. URL http://arxiv.org/abs/2303.08774v3. 1

Victor Y. Pan and Robert S. Schreiber. An improved newton iteration for the generalized inverse of a
matrix, with applications. SIAM J. Sci. Comput., 12:1109–1130, 1991. 3.1

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward
Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang,
Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning
library, 2019. D

13


---Page Break---
Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: General-
ization beyond overfitting on small algorithmic datasets, 2022. 2

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
models are unsupervised multitask learners. 2019. 3.2

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John
Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan,
Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks,
Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron
Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu,
Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen
Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro,
Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch,
Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux,
Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume,
Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas,
Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger,
Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol
Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu,
and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher,
2022. 1

Allan Raventós, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the
emergence of non-bayesian in-context learning for regression, 2023. 2

Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context
learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United
States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.
191. URL https://aclanthology.org/2022.naacl-main.191. 2

Günther Schulz. Iterative berechung der reziproken matrix. Zeitschrift für Angewandte Mathematik
und Mechanik (Journal of Applied Mathematics and Mechanics), 13:57–59, 1933. 3.1

Vatsal Sharan, Aaron Sidford, and Gregory Valiant. Memory-sample tradeoffs for linear regression
with small error. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of
Computing, pages 890–901, 2019. 4.3

Kai Shen, Junliang Guo, Xu Tan, Siliang Tang, Rui Wang, and Jiang Bian. A study on relu and
softmax in transformer, 2023a. 3.2

Lingfeng Shen, Aayush Mishra, and Daniel Khashabi. Do pretrained transformers really learn
in-context by gradient descent?, 2023b. 2

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared
Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon
Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He,
Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron to train
megatron-turing nlg 530b, a large-scale generative language model, 2022. 1

Torsten Soderstrom and G. W. Stewart. On the numerical properties of an iterative method for
computing the moore- penrose generalized inverse. SIAM Journal on Numerical Analysis, 11(1):
61–74, 1974. ISSN 00361429. URL http://www.jstor.org/stable/2156431. 3.1

Hongjin Su, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari
Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. Selective annotation makes lan-
guage models better few-shot learners. In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.net/forum?id=qY1hlv7gwg. 2

Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak. Transformers as
support vector machines. ArXiv, abs/2308.16898, 2023a. 2, 6

14


---Page Break---
Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, and Samet Oymak. Max-margin token
selection in attention mechanism, 2023b. 2

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze
Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven
Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin,
James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent
Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh
Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi,
Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran,
Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee,
Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton,
Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak,
Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. 1

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
In I. Guyon, U. Von
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, edi-
tors, Advances in Neural Information Processing Systems, volume 30. Curran Associates,
Inc., 2017.
URL https://proceedings.neurips.cc/paper_files/paper/2017/file/
3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 1, 3.2

Max Vladymyrov, Johannes von Oswald, Mark Sandler, and Rong Ge. Linear transformers are
versatile in-context learners, 2024. 1

Johannes von Oswald, Eyvind Niklasson, E. Randazzo, João Sacramento, Alexander Mordvintsev,
Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In
International Conference on Machine Learning, 2022. 1, 2, 4.1

Johannes von Oswald, Eyvind Niklasson, Maximilian Schlegel, Seijin Kobayashi, Nicolas Zucchet,
Nino Scherrer, Nolan Miller, Mark Sandler, Blaise Agüera y Arcas, Max Vladymyrov, Razvan
Pascanu, and Joao Sacramento. Uncovering mesa-optimization algorithms in transformers. ArXiv,
abs/2309.05858, 2023. 1, 2

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Inter-
pretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022. 2

Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao
Liu, Da Huang, Denny Zhou, and Tengyu Ma. Larger language models do in-context learning
differently, 2023. 2

Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Replacing softmax with relu
in vision transformers, 2023. 3.2

Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee,
Sang-goo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label
demonstrations.
In Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, pages 2422–2437, Abu Dhabi, United Arab Emirates, December 2022.
Association for Computational Linguistics.
doi: 10.18653/v1/2022.emnlp-main.155.
URL
https://aclanthology.org/2022.emnlp-main.155. 2

Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context.
ArXiv, abs/2306.09927, 2023. 2

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving
few-shot performance of language models. In International Conference on Machine Learning,
pages 12697–12706. PMLR, 2021. 2

15


---Page Break---
Appendix

A Additional Experimental Results
16

A.1
Contrast with LSTMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

A.2 Additional Results on Isotropic Data without Noise . . . . . . . . . . . . . . . . .
17

A.2.1
Progression of Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . .
17

A.2.2
Heatmaps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17

A.2.3
Comparison with Other Second-Order Methods . . . . . . . . . . . . . . .
20

A.2.4
Additional Results on Comparison over Transformer Layers . . . . . . . .
22

A.2.5
Additional Results on Similarity of Induced Weights . . . . . . . . . . . .
22

A.3 Varying Data Distribution or Function Class . . . . . . . . . . . . . . . . . . . . .
23

A.3.1
Experiments on Ill-Conditioned Problems . . . . . . . . . . . . . . . . . .
23

A.3.2
Experiments with Noisy Linear Regression . . . . . . . . . . . . . . . . .
25

A.3.3
Experiments with a Non-Linear Function Class (2-Layer MLP)
. . . . . .
25

A.4 Varying Transformer Architecture
. . . . . . . . . . . . . . . . . . . . . . . . . .
27

A.4.1
Experiments on Transformers of Fewer Heads . . . . . . . . . . . . . . . .
27

A.4.2
Experiments on Transformers with More Layers
. . . . . . . . . . . . . .
28

A.5
Heatmaps with Best-Matching Steps Help Compare Convergence Rates . . . . . .
29

A.6
Definitions for Evaluating Forgetting . . . . . . . . . . . . . . . . . . . . . . . . .
29

B
Detailed Proofs for Section 5
30

B.1
Helper Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

B.2
Proof of Theorem 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31

B.3
Iterative Newton as a Sum of Moments Method . . . . . . . . . . . . . . . . . . .
34

B.4
Estimated weight vectors lie in the span of previous examples . . . . . . . . . . . .
35

C Computes
36

D License
36

E
Limitations
36

F
Broader Impacts
36

A
Additional Experimental Results

A.1
Contrast with LSTMs

While our primary goal is to analyze Transformers, we also consider LSTMs [Hochreiter and
Schmidhuber, 1997] to understand whether Transformers learn different algorithms than other neural
sequence models trained to do linear regression. In particular, we train a unidirectional L-layer
LSTM, which generates a sequence of hidden states H(ℓ) for each layer ℓ, similarly to an L-layer
Transformer. As with Transformers, we add a readout layer that predicts the ˆyLSTM
t+1
from the final
hidden state at the final layer, H(L)
:,2t+1.

16


---Page Break---
Transformers
LSTM

Newton
0.991
0.920
GD
0.957
0.916
OGD
0.806
0.954
Table 1: Similarity of errors between algorithms. Transformers are more similar to full-observation
methods such as Newton and GD; and LSTMs are more similar to online methods such as OGD.

We train a 10-layer LSTM model, with 5.3M parameters, in an identical manner to the Transformers
(with 9.5M parameters) studied in the previous sections.5

LSTMs’ inferior performance to Transformers can be explained by the inability of LSTMs to use
deeper layers to improve their predictions. Figure 7 shows that LSTM performance does not improve
across layers—a readout head fine-tuned for the first layer makes equally good predictions as the
full 10-layer model. Thus, LSTMs seem poorly equipped to fully implement iterative algorithms.
Similarly, Table 1 shows that LSTMs are more similar to OGD than Transformers are, whereas
Transformers are more similar to Newton and GD than LSTMs.

A.2
Additional Results on Isotropic Data without Noise

A.2.1
Progression of Algorithms

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
# In-Context Examples

10
3

10
2

10
1

100

Errors

Errors v.s. # In-Context Examples

Transformers Layer #01
Transformers Layer #02
Transformers Layer #03
Transformers Layer #04
Transformers Layer #05
Transformers Layer #06
Transformers Layer #07
Transformers Layer #08
Transformers Layer #09
Transformers Layer #10
Transformers Layer #11
Transformers Layer #12

(a) Transformers

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
# In-Context Examples

10
3

10
2

10
1

100

Errors

Errors v.s. # In-Context Examples

Iterative Newton #01 (1 steps)
Iterative Newton #02 (1 steps)
Iterative Newton #03 (3 steps)
Iterative Newton #04 (5 steps)
Iterative Newton #05 (8 steps)
Iterative Newton #06 (10 steps)
Iterative Newton #07 (14 steps)
Iterative Newton #08 (17 steps)
Iterative Newton #09 (20 steps)
Iterative Newton #10 (21 steps)
Iterative Newton #11 (21 steps)
Iterative Newton #12 (21 steps)

(b) Iterative Newton’s Method

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
# In-Context Examples

10
3

10
2

10
1

100

Errors

Errors v.s. # In-Context Examples

LSTM Layer #01
LSTM Layer #02
LSTM Layer #03
LSTM Layer #04
LSTM Layer #05
LSTM Layer #06
LSTM Layer #07
LSTM Layer #08
LSTM Layer #09
LSTM Layer #10

(c) LSTM

Figure 7: Progression of Algorithms. (a) Transformer’s performance improves over the layer index
ℓ. (b) Iterative Newton’s performance improves over the number of iterations k, in a way that closely
resembles the Transformer. We plot the best-matching k to Transformer’s ℓfollowing Definition 3.4.
(c) In contrast, LSTM’s performance does not improve from layer to layer.

A.2.2
Heatmaps

We present heatmaps with all values of similarities.

5While the LSTM has fewer parameters than the Transformer, we found in preliminary experiments that
increasing the size of the LSTM would not substantively change our results.

17


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.920
.920
.912
.816
.716
.662
.634
.623
.618
.613
.620
.616

.876
.876
.929
.858
.760
.702
.672
.660
.655
.651
.656
.652

.829
.828
.927
.893
.805
.745
.713
.700
.694
.690
.695
.692

.781
.781
.911
.916
.848
.789
.755
.741
.735
.732
.735
.733

.734
.734
.883
.923
.886
.834
.798
.783
.777
.774
.777
.774

.691
.691
.850
.916
.912
.875
.840
.824
.817
.814
.817
.815

.652
.652
.814
.898
.926
.910
.878
.862
.855
.852
.854
.852

.619
.619
.779
.874
.927
.935
.911
.895
.888
.885
.886
.885

.591
.591
.748
.849
.919
.949
.937
.921
.915
.913
.913
.913

.569
.569
.723
.826
.907
.954
.956
.942
.936
.934
.935
.934

.552
.552
.703
.807
.894
.953
.968
.958
.953
.950
.951
.950

.539
.539
.688
.792
.882
.949
.976
.969
.965
.962
.962
.962

.530
.530
.677
.780
.871
.944
.979
.977
.973
.971
.971
.971

.524
.524
.669
.771
.863
.938
.980
.983
.980
.978
.978
.978

.520
.519
.664
.765
.857
.933
.979
.986
.985
.983
.983
.983

.517
.517
.660
.760
.852
.929
.977
.988
.988
.987
.986
.987

.515
.515
.657
.757
.848
.926
.975
.988
.990
.989
.989
.989

.513
.513
.655
.754
.846
.924
.973
.988
.992
.991
.991
.991

.512
.512
.653
.752
.843
.921
.972
.988
.992
.992
.992
.993

.511
.511
.652
.751
.842
.920
.970
.987
.993
.993
.993
.993

.511
.511
.651
.750
.840
.918
.969
.986
.992
.993
.994
.994

.510
.510
.649
.749
.839
.917
.967
.984
.991
.993
.993
.993

.508
.508
.646
.746
.835
.913
.963
.981
.988
.989
.990
.990

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.954
.953
.870
.770
.692
.645
.620
.610
.606
.600
.607
.603
.578
.577
.733
.831
.905
.946
.954
.946
.941
.939
.939
.938
.543
.543
.694
.795
.883
.944
.970
.967
.963
.961
.962
.961
.531
.531
.679
.781
.871
.939
.973
.974
.972
.970
.970
.969
.525
.524
.672
.772
.863
.935
.974
.977
.976
.974
.974
.974
.521
.521
.667
.767
.858
.932
.974
.979
.978
.977
.977
.977
.518
.518
.664
.763
.855
.929
.973
.980
.980
.979
.979
.979
.516
.516
.661
.761
.852
.927
.973
.981
.981
.980
.980
.980
.515
.515
.660
.759
.850
.926
.972
.982
.983
.981
.981
.981
.514
.514
.658
.757
.849
.924
.972
.982
.983
.982
.982
.982
.514
.514
.657
.756
.847
.923
.971
.982
.984
.983
.983
.983
.512
.512
.656
.755
.846
.922
.970
.982
.984
.983
.983
.983
.512
.512
.655
.753
.845
.921
.970
.982
.984
.983
.984
.984
.512
.511
.655
.753
.844
.921
.970
.982
.985
.984
.984
.984
.511
.511
.654
.752
.844
.920
.969
.982
.985
.984
.985
.985
.511
.510
.653
.752
.843
.920
.969
.982
.985
.985
.985
.985
.510
.510
.652
.751
.842
.919
.968
.982
.985
.984
.985
.985
.510
.510
.652
.750
.841
.919
.968
.982
.985
.985
.985
.985
.510
.510
.652
.750
.841
.918
.968
.982
.986
.985
.986
.986
.509
.509
.652
.750
.841
.918
.968
.982
.986
.986
.986
.986
.509
.508
.651
.749
.840
.917
.967
.982
.986
.985
.986
.986
.509
.509
.651
.749
.840
.917
.967
.981
.986
.985
.986
.986
.510
.509
.651
.749
.840
.916
.967
.981
.986
.986
.986
.986
.509
.508
.650
.748
.839
.916
.966
.981
.986
.986
.986
.986
.508
.508
.650
.748
.839
.916
.966
.981
.986
.986
.986
.986
.508
.508
.650
.748
.839
.916
.966
.981
.986
.986
.987
.987
.508
.508
.650
.748
.838
.915
.966
.981
.986
.986
.986
.986

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 8: Similarity of Errors. The best matching steps are highlighted in yellow.

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

-.000
.001
.859
.811
.742
.719
.714
.711
.711
.711
.711
.712

.000
.001
.872
.856
.795
.769
.763
.760
.760
.760
.760
.761

.001
.001
.870
.890
.844
.816
.809
.806
.806
.805
.806
.806

.002
.001
.857
.909
.883
.857
.849
.845
.845
.845
.845
.846

.003
.000
.838
.915
.911
.889
.881
.877
.877
.876
.877
.877

.004
-.000
.819
.912
.928
.914
.906
.902
.902
.901
.902
.902

.005
-.000
.801
.903
.937
.932
.926
.922
.921
.921
.922
.922

.005
-.001
.785
.893
.939
.944
.941
.937
.936
.936
.937
.937

.005
-.001
.773
.883
.936
.951
.952
.948
.947
.947
.948
.948

.006
-.001
.763
.874
.932
.953
.959
.955
.955
.955
.956
.956

.006
-.000
.756
.867
.927
.954
.963
.961
.961
.960
.961
.961

.006
-.000
.750
.862
.923
.953
.966
.965
.965
.964
.965
.965

.006
-.000
.747
.858
.920
.952
.967
.967
.968
.967
.968
.968

.006
.000
.744
.855
.918
.950
.968
.969
.969
.969
.970
.970

.006
.000
.742
.853
.916
.949
.967
.970
.971
.970
.971
.971

.006
.000
.741
.851
.914
.948
.967
.970
.972
.971
.972
.972

.007
.000
.740
.850
.913
.947
.966
.970
.972
.972
.973
.973

.007
.000
.739
.849
.912
.946
.966
.970
.973
.972
.973
.974

.007
.000
.739
.849
.911
.945
.966
.970
.973
.973
.974
.974

.007
.000
.738
.848
.911
.945
.965
.970
.973
.973
.974
.974

.007
.000
.738
.848
.911
.944
.965
.970
.973
.973
.974
.974

.007
.000
.738
.848
.910
.944
.965
.970
.973
.973
.974
.974

.007
.000
.738
.848
.910
.944
.965
.970
.973
.973
.974
.974

Similarity of Induced Weight w (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.069
-.002
.771
.731
.695
.683
.674
.671
.671
.669
.670
.670
.020
.004
.772
.880
.934
.958
.965
.963
.964
.962
.964
.964
.020
.005
.757
.866
.927
.959
.971
.970
.971
.969
.971
.971
.020
.005
.752
.861
.923
.958
.972
.973
.974
.972
.974
.974
.020
.005
.749
.858
.921
.957
.973
.974
.975
.973
.975
.975
.020
.005
.748
.856
.919
.956
.973
.974
.976
.974
.976
.976
.019
.005
.747
.855
.918
.955
.973
.975
.976
.974
.976
.976
.020
.005
.746
.854
.917
.954
.973
.975
.976
.975
.977
.977
.020
.005
.745
.854
.917
.954
.972
.975
.977
.975
.977
.977
.020
.005
.745
.853
.916
.953
.972
.975
.977
.975
.977
.977
.020
.005
.744
.853
.916
.953
.972
.975
.977
.976
.977
.977
.020
.005
.744
.852
.915
.953
.972
.975
.977
.976
.977
.977
.019
.005
.744
.852
.915
.953
.972
.975
.977
.976
.978
.977
.020
.005
.744
.852
.915
.952
.972
.975
.977
.976
.978
.978
.020
.005
.743
.851
.915
.952
.972
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.952
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.952
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.952
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.952
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.952
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.951
.971
.975
.977
.976
.978
.978
.020
.005
.743
.851
.914
.951
.971
.975
.978
.976
.978
.978
.020
.005
.742
.850
.914
.951
.971
.975
.978
.976
.978
.978
.020
.005
.742
.851
.914
.951
.971
.975
.978
.976
.978
.978
.020
.005
.742
.850
.913
.951
.971
.975
.978
.976
.978
.978
.020
.005
.742
.850
.913
.951
.971
.975
.978
.976
.978
.978
.019
.005
.742
.850
.913
.951
.971
.975
.978
.976
.978
.978

Similarity of Induced Weight w (Transformers v.s. Gradient Descent)

Figure 9: Similarity of Induced Weight Vectors. The best matching steps are highlighted in yellow.

18


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

4

8

16

32

64

128

256

512

1024

2048

4096

 Gradient descent Steps

.953
.953
.870
.771
.692
.645
.620
.609
.605
.599
.606
.602

.910
.910
.903
.826
.750
.703
.676
.665
.660
.655
.661
.657

.842
.841
.913
.878
.816
.773
.746
.733
.728
.724
.728
.725

.759
.759
.886
.905
.876
.846
.820
.807
.801
.798
.801
.799

.678
.677
.831
.895
.910
.903
.886
.873
.867
.865
.867
.865

.610
.610
.768
.858
.914
.938
.934
.924
.918
.916
.917
.916

.563
.563
.717
.817
.897
.947
.961
.954
.950
.948
.948
.947

.536
.535
.685
.786
.875
.941
.972
.971
.968
.966
.967
.966

.521
.521
.666
.766
.858
.932
.973
.979
.978
.977
.977
.977

.513
.513
.656
.755
.847
.923
.971
.982
.984
.982
.983
.983

.509
.509
.652
.749
.840
.917
.967
.982
.986
.985
.986
.986

.507
.507
.648
.745
.836
.913
.964
.980
.986
.986
.987
.987

.506
.505
.646
.744
.834
.911
.962
.979
.985
.987
.988
.988

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 10: Similarity of Errors of Gradient Descent in Log Scale. The best matching steps are
highlighted in yellow. Putting the number of steps of Gradient Descent in log scale further verifies
the claim that Transformer’s rate of covergence is exponentially faster than that of Gradient Descent.

19


---Page Break---
A.2.3
Comparison with Other Second-Order Methods

In this section, we ablate with alternative second-order methods, such as Conjugate Gradient, BFGS,
and its limited memory variant, L-BFGS.

Conjugate Gradient Method. For linear regression problems, the Conjugate Gradient (CG) method
solves the linear system
(X⊤X)
|
{z
}
S

w −X⊤y = 0

CG finds the weight vector ˆwCG with initialization w0 by maintain a set of conjugate gradient
{∆w1, · · · , ∆wk}. It follows the iterative update rule

dk = −∇L(wk)

∆wk = dk −

k−1
X

i=0

d⊤
k S∆wi
∆w⊤
i S∆wi
∆wi

αk = arg min
α L(wk + α∆wk)

wk+1 = wk + αk∆wk

(12)

The conjugate Gradient method requires O (√κ log(1/ϵ)) steps to converge to an ϵ error on quadratic
objectives such as linear regression.

BFGS. Broyden– Fletcher–Goldfarb–Shanno (BFGS) is a Quasi-Newton method, designed to
approximate the inverse Hessian Bk :≈∇2L(wk)−1. The BFGS updates are given by

wk+1 = wk −αkBk∇L(wk)
(13)

where
sk = wk+1 −wk
yk = ∇L(wk+1) −∇L(wk)

Bk+1 = Bk −Bkyky⊤
k Bk
y⊤
k Bkyk
+ sks⊤
k
y⊤
k sk
When k is large, Bk approximates the inverse Hessian well.

L(imited-memory)-BFGS. L-BFGS is a limited-memory version of BFGS. Instead of the inverse
Hessian Bk, L-BFGS maintains a history of past m updates (where m is usually small). Recall the
iterative update rule of Bk in BFGS

Bk+1 = Bk −Bkyky⊤
k Bk
y⊤
k Bkyk
+ sks⊤
k
y⊤
k sk
(14)

Unlike BFGS, which recursively unroll to an initialization B0, L-BFGS only unroll to Bk−m but
replacing Bk−m with Binit. In this regard, running n steps of L-BFGS only requires O(mn) memory,
which is more memory-efficient than BFGS who requires O(n2) memory. The trade-off is that L-
BFGS won’t have a good estimate of the inverse Hessian when m < d, where d is the dimensionality
of the quadratic problem. In this regard, it will converge slower than full BFGS.

In Figure 11 and Figure 12, we compare Transformers with BFGS, L-BFGS, and Conjugate Gradient
method on the metric of similarity of errors. We find that Transformers have a similar linear
correspondence with BFGS. This is perhaps not surprising, given that BFGS also gets a superlinear
convergence rate for linear regression Nocedal and Wright [1999]. Meanwhile, Transformers show a
substantially faster convergence rate than L-BFGS and CG.

20


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40

 BFGS Steps

.977
.977
.863
.759
.675
.628
.601
.587
.583
.578
.586
.581
.911
.911
.826
.825
.779
.735
.706
.690
.684
.679
.686
.682
.801
.801
.719
.783
.817
.801
.775
.760
.754
.750
.755
.752
.687
.687
.616
.706
.805
.829
.816
.802
.796
.793
.797
.794
.585
.585
.537
.633
.767
.834
.839
.829
.823
.820
.823
.822
.505
.505
.498
.591
.730
.829
.859
.852
.848
.847
.848
.847
.467
.467
.512
.602
.722
.828
.876
.876
.873
.872
.872
.872
.471
.471
.558
.647
.744
.838
.892
.895
.893
.892
.892
.892
.491
.491
.608
.698
.781
.858
.907
.912
.910
.908
.908
.908
.512
.512
.650
.742
.818
.880
.921
.926
.925
.923
.923
.923
.526
.526
.676
.771
.846
.900
.934
.939
.938
.935
.936
.936
.532
.532
.687
.784
.864
.917
.946
.951
.950
.947
.948
.948
.530
.530
.687
.787
.871
.930
.958
.963
.962
.959
.960
.960
.525
.525
.681
.782
.872
.936
.968
.973
.972
.970
.970
.970
.518
.518
.671
.772
.864
.936
.973
.979
.979
.977
.978
.978
.511
.511
.662
.762
.857
.933
.976
.984
.984
.982
.983
.983
.506
.506
.656
.756
.850
.929
.976
.986
.987
.985
.986
.986
.504
.504
.653
.753
.847
.927
.976
.987
.989
.988
.988
.988
.504
.504
.653
.752
.846
.925
.975
.988
.991
.990
.990
.990
.504
.504
.652
.751
.846
.924
.974
.989
.993
.992
.992
.992
.503
.503
.652
.751
.845
.923
.973
.989
.993
.993
.994
.994
.503
.503
.651
.750
.843
.922
.972
.988
.994
.994
.994
.995
.502
.502
.650
.748
.842
.920
.971
.987
.994
.995
.995
.996
.501
.501
.649
.747
.841
.919
.969
.987
.994
.995
.996
.996
.500
.500
.648
.747
.840
.918
.969
.986
.993
.995
.996
.996
.500
.500
.648
.746
.839
.918
.968
.986
.993
.995
.996
.996
.500
.500
.647
.746
.839
.917
.968
.985
.993
.995
.996
.996
.500
.500
.647
.745
.839
.917
.967
.985
.993
.995
.996
.996
.500
.500
.647
.745
.839
.917
.967
.985
.992
.995
.996
.996
.500
.500
.647
.745
.839
.917
.967
.985
.992
.995
.996
.996
.500
.500
.647
.745
.839
.917
.967
.985
.992
.995
.996
.996
.500
.500
.647
.745
.838
.917
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.917
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.917
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.995
.996
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.994
.995
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.994
.995
.996
.500
.500
.647
.745
.838
.916
.967
.984
.992
.994
.995
.996

Similarity of Errors  (Transformers v.s. Bfgs)

(a) Transformer v.s. BFGS

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40

 L-BFGS Steps

.979
.979
.863
.762
.678
.629
.604
.591
.588
.581
.589
.585
.783
.783
.924
.913
.846
.797
.767
.753
.747
.742
.747
.744
.722
.722
.894
.928
.893
.851
.821
.807
.801
.796
.800
.798
.626
.626
.806
.899
.929
.916
.891
.877
.870
.867
.869
.868
.594
.594
.766
.870
.930
.940
.922
.908
.901
.899
.900
.899
.589
.589
.754
.858
.929
.949
.936
.923
.917
.914
.915
.915
.579
.579
.739
.841
.921
.956
.951
.939
.933
.930
.932
.931
.565
.565
.722
.824
.911
.959
.963
.952
.947
.945
.946
.945
.549
.549
.705
.807
.899
.959
.973
.964
.960
.957
.958
.958
.537
.537
.691
.794
.888
.955
.978
.972
.968
.965
.966
.966
.532
.532
.685
.788
.882
.953
.980
.976
.973
.970
.971
.971
.528
.528
.681
.784
.878
.951
.982
.980
.976
.974
.975
.974
.525
.525
.677
.779
.874
.947
.982
.983
.980
.978
.978
.978
.522
.522
.673
.774
.869
.944
.982
.985
.983
.981
.981
.981
.519
.519
.669
.771
.865
.941
.982
.987
.985
.983
.984
.984
.517
.517
.667
.768
.863
.939
.982
.988
.987
.985
.985
.985
.516
.516
.665
.766
.860
.937
.981
.988
.988
.986
.986
.986
.514
.514
.663
.764
.859
.936
.980
.989
.989
.987
.988
.988
.513
.513
.662
.763
.857
.934
.980
.989
.990
.988
.989
.989
.512
.512
.661
.762
.856
.933
.979
.989
.991
.989
.989
.989
.511
.511
.660
.761
.855
.932
.978
.990
.991
.990
.990
.990
.511
.511
.659
.760
.854
.931
.977
.990
.992
.991
.991
.991
.510
.510
.658
.759
.853
.930
.977
.989
.992
.991
.992
.992
.509
.509
.657
.758
.852
.929
.976
.989
.993
.992
.992
.992
.509
.509
.657
.757
.851
.928
.975
.989
.993
.992
.993
.993
.508
.508
.656
.756
.850
.927
.975
.989
.993
.993
.993
.993
.508
.508
.656
.756
.849
.927
.974
.989
.993
.993
.993
.994
.507
.507
.655
.755
.849
.926
.974
.989
.993
.994
.994
.994
.507
.507
.655
.754
.848
.925
.973
.989
.993
.994
.994
.994
.507
.507
.654
.754
.848
.925
.973
.988
.994
.994
.994
.994
.506
.506
.654
.754
.847
.924
.973
.988
.994
.994
.994
.995
.506
.506
.654
.753
.847
.924
.972
.988
.993
.994
.995
.995
.506
.506
.653
.753
.846
.924
.972
.988
.993
.994
.995
.995
.506
.506
.653
.753
.846
.923
.972
.987
.993
.995
.995
.995
.506
.505
.653
.752
.846
.923
.971
.987
.993
.995
.995
.995
.505
.505
.653
.752
.845
.923
.971
.987
.993
.995
.995
.996
.505
.505
.652
.752
.845
.922
.970
.987
.993
.995
.995
.996
.505
.505
.652
.751
.845
.922
.970
.986
.993
.995
.995
.996
.505
.505
.652
.751
.845
.922
.970
.986
.993
.995
.995
.996
.505
.504
.652
.751
.845
.922
.970
.986
.993
.995
.996
.996

Similarity of Errors  (Transformers v.s. L-Bfgs)

(b) Transformer v.s. L-BFGS

Figure 11: Similarity of Errors between Transformers and BFGS or L-BFGS. The best matching
steps are highlighted in yellow. We find that Transformer, from layers 6 to 11, has a linear correspon-
dence with BFGS. For L-BFGS, due to its limited memory, it approximates second-order information
more slowly and results in a slower convergence rate than Transformers.

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40

 Conjugate Gradient Steps

.841
.841
.937
.858
.767
.718
.689
.675
.670
.665
.670
.667
.731
.731
.893
.923
.871
.827
.795
.779
.772
.768
.772
.770
.665
.665
.835
.914
.917
.890
.860
.844
.837
.833
.836
.834
.618
.619
.787
.885
.930
.927
.904
.888
.881
.878
.880
.878
.586
.586
.751
.855
.925
.947
.934
.919
.912
.909
.910
.910
.565
.565
.726
.831
.914
.955
.952
.939
.932
.930
.931
.930
.552
.552
.711
.816
.904
.957
.964
.952
.947
.944
.945
.944
.544
.544
.701
.806
.896
.956
.971
.962
.956
.954
.955
.954
.538
.538
.694
.798
.890
.954
.976
.969
.964
.961
.962
.962
.534
.534
.689
.793
.885
.952
.979
.974
.969
.967
.967
.967
.530
.530
.685
.788
.881
.950
.980
.977
.974
.971
.972
.971
.527
.527
.681
.784
.877
.948
.982
.980
.977
.974
.975
.975
.524
.524
.678
.780
.874
.946
.982
.982
.979
.977
.978
.978
.522
.522
.675
.777
.871
.945
.982
.984
.982
.979
.980
.980
.520
.520
.672
.774
.868
.943
.982
.985
.983
.981
.982
.982
.518
.518
.670
.772
.866
.941
.982
.986
.985
.983
.983
.983
.517
.517
.668
.770
.864
.940
.982
.987
.986
.984
.984
.984
.515
.515
.667
.768
.863
.939
.982
.988
.987
.985
.985
.985
.514
.514
.666
.767
.861
.938
.981
.988
.988
.986
.986
.986
.513
.513
.664
.765
.860
.937
.981
.989
.989
.987
.987
.987
.512
.513
.663
.764
.859
.936
.981
.989
.989
.988
.988
.988
.512
.512
.662
.763
.858
.935
.980
.989
.990
.988
.989
.989
.511
.511
.661
.762
.857
.934
.980
.989
.990
.989
.989
.989
.510
.510
.660
.761
.856
.933
.979
.990
.991
.990
.990
.990
.510
.510
.660
.760
.855
.932
.979
.990
.991
.990
.990
.990
.509
.509
.659
.759
.854
.932
.979
.990
.992
.990
.991
.991
.509
.509
.658
.759
.853
.931
.978
.990
.992
.991
.991
.991
.508
.508
.658
.758
.852
.930
.978
.990
.992
.991
.991
.992
.508
.508
.657
.757
.852
.930
.978
.990
.992
.991
.992
.992
.508
.508
.657
.757
.851
.929
.977
.990
.993
.992
.992
.992
.507
.507
.656
.756
.851
.929
.977
.990
.993
.992
.992
.992
.507
.507
.656
.756
.850
.928
.977
.990
.993
.992
.992
.993
.507
.507
.655
.755
.850
.928
.976
.990
.993
.992
.993
.993
.506
.506
.655
.755
.849
.927
.976
.990
.993
.993
.993
.993
.506
.506
.655
.754
.849
.927
.976
.989
.993
.993
.993
.993
.506
.506
.654
.754
.848
.927
.975
.989
.993
.993
.993
.994
.505
.505
.654
.754
.848
.926
.975
.989
.994
.993
.993
.994
.505
.505
.654
.753
.847
.926
.975
.989
.994
.993
.994
.994
.505
.505
.654
.753
.847
.926
.975
.989
.994
.993
.994
.994
.505
.505
.653
.753
.847
.925
.974
.989
.994
.994
.994
.994

Similarity of Errors  (Transformers v.s. Conjugate Gradient)

Figure 12: Similarity of Errors between Transformers and Conjugate Gradient. Transformer’s
convergence rate is still faster than conjugate gradient methods.

21


---Page Break---
A.2.4
Additional Results on Comparison over Transformer Layers

2
4
6
8
10
12
Layer Index

0.5

0.6

0.7

0.8

0.9

1.0

Cosine Similarity

SimE(Transformers, Newton)
SimE(Transformers, GD)
SimE(Transformers, OLS)

(a) Similarity of Errors

2
4
6
8
10
12
Layer Index

0.0

0.2

0.4

0.6

0.8

1.0

Cosine Similarity

SimW(Transformers, Newton)
SimW(Transformers, GD)
SimW(Transformers, OLS)

(b) Similarity of Induced Weights

Figure 13: Similarities between Transformer and candidate algorithms. Transformers resemble
Iterative Newton’s Method the most.

A.2.5
Additional Results on Similarity of Induced Weights

We present more details line plots for how the similarity of weights changes as the models see more
in-context observations {xi, yi}n
i=1, i.e., as n increases. We fix the number of Transformers layers ℓ
and compare with other algorithms with their best-match steps to ℓin Figure 14.

0
5
10
15
20
25
30
35
40
Number of In-Context Examples

0.2

0.0

0.2

0.4

0.6

0.8

1.0

Cosine Similarity

Layer 2

CosSim(wTF, wNewton)

CosSim(wTF, wGD)
CosSim(wTF, wOLS)

0
5
10
15
20
25
30
35
40
Number of In-Context Examples

0.2

0.0

0.2

0.4

0.6

0.8

1.0

Cosine Similarity

Layer 3

CosSim(wTF, wNewton)

CosSim(wTF, wGD)

CosSim(wTF, wOLS)

0
5
10
15
20
25
30
35
40
Number of In-Context Examples

0.2

0.0

0.2

0.4

0.6

0.8

1.0

Cosine Similarity

Layer 12

CosSim(wTF, wNewton)

CosSim(wTF, wGD)
CosSim(wTF, wOLS)

Figure 14: Similarity of induced weights over varying number of in-context examples, on three layer
indices of Transformers, indexed as 2, 3 and 12. We find that initially at layer 2, the Transformers
model hasn’t learned so it has zero similarity to all candidate algorithms. As we progress to the
next layer number 3, we find that Transformers start to learn, and when provided few examples,
Transformers are more similar to OLS but soon become most similar to the Iterative Newton’s
Method. Layer 12 shows that Transformers in the later layers converge to the OLS solution when
provided more than 1 example. We also find there is a dip around n = d for similarity between
Transformers and OLS but not for Transformers and Newton, and this is probably because OLS has a
more prominent double-descent phenomenon than Transformers and Newton.

22


---Page Break---
A.3
Varying Data Distribution or Function Class

A.3.1
Experiments on Ill-Conditioned Problems

In this section, we repeat the same experiments as we did on isotropic data in the main text and in
Appendix A.2, and we change the covariance matrix to be ill-conditioned such that κ(Σ) = 100.

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
# In-Context Examples

10
3

10
2

10
1

100

Errors

Errors v.s. # In-Context Examples

Transformers Layer #01
Transformers Layer #02
Transformers Layer #03
Transformers Layer #04
Transformers Layer #05
Transformers Layer #06
Transformers Layer #07
Transformers Layer #08
Transformers Layer #09
Transformers Layer #10
Transformers Layer #11
Transformers Layer #12

(a) Transformers

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
# In-Context Examples

10
3

10
2

10
1

100

Errors

Errors v.s. # In-Context Examples

Iterative Newton #01 (1 steps)
Iterative Newton #02 (1 steps)
Iterative Newton #03 (3 steps)
Iterative Newton #04 (5 steps)
Iterative Newton #05 (8 steps)
Iterative Newton #06 (10 steps)
Iterative Newton #07 (14 steps)
Iterative Newton #08 (17 steps)
Iterative Newton #09 (20 steps)
Iterative Newton #10 (21 steps)
Iterative Newton #11 (21 steps)
Iterative Newton #12 (21 steps)

(b) Iterative Newton’s Method

Figure 15: Progression of Algorithms on Ill-Conditioned Data. Transformer’s performance still
improves over the layer index ℓ; Iterative Newton’s Method’s performance improves over the number
of iterations t and we plot the best-matching t to Transformer’s ℓfollowing Definition 3.4.

We also present the heatmaps to find the best-matching steps and conclude that Transformers are
similar to Newton’s method than GD in ill-conditioned data.

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.885
.886
.829
.713
.598
.557
.535
.529
.528
.530
.532
.529

.814
.814
.848
.780
.662
.615
.593
.587
.585
.587
.589
.586

.736
.736
.842
.838
.733
.679
.656
.650
.649
.650
.652
.650

.661
.662
.811
.878
.805
.745
.722
.716
.714
.715
.716
.715

.593
.593
.765
.893
.867
.808
.783
.777
.775
.775
.777
.775

.536
.537
.715
.887
.913
.862
.834
.828
.825
.826
.827
.826

.493
.494
.672
.868
.940
.903
.873
.866
.864
.864
.865
.864

.464
.464
.640
.847
.951
.933
.902
.894
.892
.893
.894
.893

.444
.445
.617
.828
.953
.953
.923
.915
.913
.913
.914
.913

.431
.432
.601
.812
.948
.966
.938
.930
.928
.928
.929
.928

.422
.423
.590
.800
.942
.973
.949
.940
.939
.939
.939
.939

.416
.416
.582
.791
.935
.976
.958
.949
.947
.947
.948
.948

.411
.412
.576
.784
.928
.977
.965
.956
.954
.954
.955
.956

.407
.408
.572
.778
.923
.976
.971
.963
.961
.962
.962
.963

.404
.404
.567
.772
.916
.973
.976
.970
.968
.968
.969
.970

.400
.400
.563
.766
.910
.970
.980
.975
.974
.974
.975
.976

.397
.397
.559
.760
.904
.966
.981
.979
.978
.979
.979
.980

.394
.394
.555
.756
.898
.962
.982
.983
.982
.982
.983
.984

.392
.392
.552
.752
.894
.958
.981
.985
.984
.985
.986
.986

.390
.390
.549
.748
.890
.954
.979
.985
.985
.986
.987
.988

.389
.389
.548
.746
.887
.951
.977
.985
.985
.986
.987
.988

.387
.388
.545
.743
.883
.947
.973
.983
.983
.984
.985
.986

.384
.385
.538
.733
.872
.935
.962
.972
.972
.973
.974
.975

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
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
1100
1200
1300
1400
1500
1600
1700
1800
1900
2000
2100
2200
2300
2400
2500
2600
2700
2800
2900
3000

 Gradient descent Steps

.990
.990
.709
.548
.469
.440
.420
.413
.413
.416
.418
.413
.502
.503
.686
.870
.941
.921
.896
.889
.886
.887
.887
.886
.451
.451
.633
.839
.953
.958
.936
.929
.927
.927
.927
.926
.433
.433
.612
.821
.950
.970
.952
.945
.943
.943
.943
.943
.422
.423
.600
.809
.945
.975
.960
.954
.952
.952
.952
.952
.417
.418
.593
.802
.941
.977
.966
.960
.958
.958
.958
.958
.413
.413
.588
.796
.937
.978
.970
.964
.962
.962
.962
.962
.410
.410
.584
.791
.933
.978
.973
.967
.965
.965
.966
.966
.408
.408
.581
.788
.930
.978
.975
.970
.968
.968
.968
.968
.405
.406
.578
.785
.927
.977
.977
.972
.970
.970
.970
.971
.404
.405
.576
.782
.925
.977
.978
.974
.972
.972
.972
.972
.402
.403
.574
.780
.923
.976
.979
.975
.974
.974
.974
.974
.401
.402
.573
.778
.921
.975
.980
.976
.975
.975
.975
.976
.400
.400
.572
.776
.919
.975
.981
.977
.976
.976
.976
.977
.399
.400
.571
.775
.918
.974
.981
.978
.977
.977
.977
.978
.399
.400
.570
.774
.917
.974
.982
.980
.978
.979
.979
.979
.398
.398
.569
.772
.915
.973
.982
.980
.979
.979
.979
.980
.397
.398
.568
.771
.913
.972
.982
.981
.979
.980
.980
.980
.397
.397
.567
.770
.913
.971
.983
.982
.980
.981
.981
.981
.396
.396
.567
.769
.912
.971
.983
.982
.981
.981
.981
.982
.395
.396
.566
.768
.910
.970
.983
.982
.981
.982
.982
.982
.395
.395
.565
.767
.909
.970
.983
.983
.982
.982
.982
.983
.394
.394
.564
.766
.908
.969
.983
.983
.982
.982
.983
.983
.394
.395
.564
.766
.908
.969
.984
.984
.982
.983
.983
.984
.393
.393
.563
.765
.907
.968
.983
.984
.983
.983
.983
.984
.393
.394
.563
.765
.907
.968
.984
.985
.984
.984
.984
.985
.393
.394
.563
.764
.905
.967
.984
.985
.984
.984
.984
.985
.393
.394
.562
.763
.905
.967
.984
.985
.984
.984
.984
.985
.392
.392
.562
.763
.904
.966
.983
.985
.984
.984
.984
.985
.392
.392
.561
.762
.903
.965
.983
.985
.984
.984
.985
.985
.391
.392
.561
.762
.903
.965
.984
.985
.984
.985
.985
.986

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 16: Similarity of Errors on Ill-Conditioned Data. The best matching steps are highlighted
in yellow.

23


---Page Break---
1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.003
.023
.646
.739
.747
.721
.650
.626
.617
.615
.612
.608

.003
.024
.659
.778
.793
.765
.690
.664
.654
.653
.649
.645

.003
.024
.662
.808
.834
.805
.726
.699
.688
.687
.683
.679

.002
.024
.655
.827
.868
.838
.755
.728
.717
.715
.711
.707

.002
.024
.644
.836
.893
.864
.779
.751
.740
.738
.734
.729

.002
.024
.632
.838
.907
.881
.795
.766
.755
.753
.749
.744

.002
.023
.622
.835
.915
.892
.805
.777
.765
.764
.760
.755

.001
.023
.615
.831
.919
.900
.814
.785
.773
.772
.768
.763

.002
.023
.610
.827
.920
.906
.820
.792
.780
.779
.775
.770

.002
.023
.606
.824
.920
.911
.828
.800
.788
.787
.783
.778

.002
.023
.603
.821
.919
.917
.837
.810
.798
.797
.793
.789

.002
.023
.599
.816
.917
.924
.851
.826
.813
.812
.809
.804

.002
.023
.592
.807
.910
.930
.868
.846
.834
.832
.829
.825

.002
.022
.579
.791
.896
.932
.885
.868
.856
.855
.853
.849

.002
.021
.562
.768
.873
.926
.897
.889
.878
.877
.876
.873

.002
.021
.544
.744
.849
.914
.904
.906
.895
.894
.895
.893

.002
.020
.528
.722
.826
.900
.905
.918
.909
.908
.910
.909

.002
.020
.515
.704
.807
.886
.902
.925
.918
.918
.922
.921

.003
.019
.505
.690
.792
.873
.897
.929
.924
.925
.929
.929

.003
.019
.498
.680
.781
.863
.891
.931
.926
.928
.933
.933

.003
.019
.492
.672
.772
.854
.885
.930
.927
.929
.935
.935

.003
.019
.488
.666
.766
.848
.880
.929
.926
.929
.935
.935

.003
.019
.485
.662
.761
.843
.877
.927
.925
.927
.934
.935

Similarity of Induced Weight w (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
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
1100
1200
1300
1400
1500
1600
1700
1800
1900
2000
2100
2200
2300
2400
2500
2600
2700
2800
2900
3000

 Gradient descent Steps

.010
-.062
.292
.337
.346
.333
.294
.287
.280
.284
.273
.274
.009
.010
.625
.829
.913
.907
.831
.807
.798
.796
.790
.787
.008
.011
.611
.821
.916
.924
.858
.839
.830
.828
.822
.820
.008
.012
.602
.812
.912
.931
.874
.858
.850
.848
.843
.841
.008
.012
.595
.804
.906
.934
.884
.873
.864
.862
.858
.857
.008
.012
.589
.796
.899
.935
.892
.883
.875
.873
.870
.868
.008
.013
.583
.789
.893
.934
.897
.892
.884
.882
.879
.878
.008
.013
.577
.782
.886
.932
.901
.898
.891
.889
.886
.885
.008
.013
.572
.775
.880
.930
.904
.904
.896
.895
.892
.892
.008
.014
.568
.769
.875
.928
.906
.908
.901
.899
.898
.897
.008
.014
.564
.764
.870
.926
.908
.912
.905
.903
.902
.901
.007
.014
.560
.759
.865
.924
.909
.915
.908
.907
.906
.905
.007
.014
.557
.755
.860
.922
.910
.918
.911
.910
.909
.909
.007
.014
.554
.750
.856
.919
.910
.920
.914
.912
.912
.911
.007
.014
.551
.747
.852
.917
.911
.922
.916
.914
.914
.914
.007
.014
.548
.743
.848
.915
.911
.923
.918
.916
.916
.916
.007
.014
.546
.740
.845
.913
.911
.925
.919
.918
.918
.918
.007
.015
.544
.737
.842
.911
.911
.926
.921
.920
.920
.920
.007
.015
.542
.734
.839
.909
.911
.927
.922
.921
.921
.922
.008
.015
.540
.731
.836
.907
.910
.928
.923
.922
.923
.923
.007
.015
.538
.729
.834
.906
.910
.929
.924
.923
.924
.924
.007
.015
.536
.726
.831
.904
.910
.930
.925
.924
.925
.926
.007
.015
.534
.724
.829
.902
.910
.931
.926
.925
.926
.927
.007
.015
.533
.722
.827
.901
.909
.931
.927
.926
.927
.928
.007
.015
.531
.720
.825
.900
.909
.932
.928
.927
.928
.929
.007
.015
.530
.718
.822
.898
.909
.932
.928
.927
.929
.929
.007
.015
.529
.716
.821
.897
.908
.933
.929
.928
.929
.930
.007
.015
.528
.715
.819
.896
.908
.933
.929
.929
.930
.931
.007
.015
.527
.713
.817
.894
.908
.934
.930
.929
.931
.932
.007
.015
.525
.712
.816
.893
.907
.934
.930
.929
.931
.932
.007
.015
.524
.710
.814
.892
.907
.934
.931
.930
.932
.933

Similarity of Induced Weight w (Transformers v.s. Gradient Descent)

Figure 17: Similarity of Induced Weights on Ill-Conditioned Data. The best matching steps are
highlighted in yellow.

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40

 BFGS Steps

.925
.857
.725
.619
.562
.542
.533
.530
.531
.529
.527
.528
.819
.774
.794
.727
.661
.638
.627
.625
.625
.623
.621
.621
.720
.660
.757
.782
.726
.700
.689
.688
.687
.686
.685
.684
.650
.586
.705
.801
.768
.743
.733
.732
.731
.729
.729
.728
.592
.544
.667
.804
.805
.782
.772
.771
.770
.769
.769
.768
.546
.528
.649
.800
.839
.819
.810
.809
.807
.806
.806
.805
.507
.534
.657
.803
.870
.857
.848
.847
.845
.844
.844
.844
.473
.548
.679
.813
.889
.890
.882
.882
.879
.878
.878
.878
.451
.565
.711
.833
.905
.917
.911
.910
.907
.907
.906
.907
.440
.582
.733
.854
.919
.938
.935
.934
.931
.930
.930
.931
.433
.593
.747
.871
.934
.954
.952
.952
.949
.948
.948
.949
.427
.593
.750
.878
.946
.968
.967
.967
.964
.964
.964
.965
.423
.593
.749
.878
.953
.977
.978
.978
.976
.975
.975
.976
.419
.588
.743
.874
.954
.982
.984
.985
.983
.982
.982
.983
.416
.585
.739
.869
.952
.984
.988
.989
.987
.987
.987
.987
.414
.583
.738
.868
.950
.985
.990
.991
.990
.989
.989
.990
.413
.582
.736
.866
.948
.984
.991
.992
.992
.991
.991
.992
.412
.580
.734
.864
.946
.983
.992
.993
.993
.992
.993
.994
.411
.578
.731
.861
.944
.982
.992
.993
.993
.993
.994
.994
.410
.577
.730
.860
.942
.980
.992
.993
.994
.994
.994
.995
.409
.577
.729
.858
.941
.979
.991
.993
.994
.994
.994
.995
.409
.576
.728
.858
.940
.979
.991
.992
.994
.994
.994
.995
.409
.576
.728
.857
.940
.978
.990
.992
.993
.994
.994
.995
.408
.575
.728
.857
.939
.978
.990
.992
.993
.993
.994
.995
.408
.575
.728
.857
.939
.978
.990
.991
.993
.993
.994
.995
.408
.575
.727
.856
.939
.977
.990
.991
.993
.993
.994
.995
.408
.575
.727
.856
.939
.977
.990
.991
.993
.993
.994
.995
.408
.575
.727
.856
.938
.977
.989
.991
.993
.993
.993
.994
.408
.575
.727
.856
.938
.977
.989
.991
.993
.993
.993
.994
.408
.575
.727
.856
.938
.977
.989
.991
.992
.993
.993
.994
.408
.575
.727
.856
.938
.976
.989
.991
.992
.993
.993
.994
.408
.575
.727
.855
.938
.976
.989
.990
.992
.993
.993
.994
.408
.575
.727
.855
.938
.976
.989
.990
.992
.992
.993
.994
.408
.575
.727
.855
.938
.976
.989
.990
.992
.992
.993
.994
.408
.574
.726
.855
.938
.976
.989
.990
.992
.992
.993
.994
.408
.574
.726
.855
.937
.976
.989
.990
.992
.992
.993
.994
.408
.574
.726
.855
.937
.976
.988
.990
.992
.992
.993
.994
.408
.574
.726
.855
.937
.976
.988
.990
.992
.992
.993
.994
.408
.574
.726
.855
.937
.976
.988
.990
.992
.992
.993
.994
.408
.574
.726
.855
.937
.976
.988
.990
.992
.992
.993
.994

Similarity of Errors  (Transformers v.s. Bfgs)

(a) Transformer v.s. BFGS

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40

 L-BFGS Steps

.926
.856
.723
.617
.561
.541
.532
.529
.530
.528
.526
.526
.699
.881
.873
.779
.712
.689
.678
.676
.675
.674
.672
.674
.607
.832
.906
.852
.784
.758
.747
.745
.744
.743
.741
.742
.527
.747
.898
.916
.857
.828
.817
.816
.814
.813
.813
.813
.494
.694
.869
.939
.898
.870
.859
.858
.856
.855
.855
.856
.484
.671
.848
.945
.923
.895
.884
.883
.881
.880
.880
.881
.481
.662
.832
.945
.941
.914
.904
.904
.901
.900
.900
.901
.472
.652
.816
.939
.955
.931
.921
.920
.918
.917
.917
.917
.458
.637
.801
.930
.965
.946
.936
.935
.933
.932
.932
.933
.447
.625
.787
.919
.970
.957
.948
.947
.945
.944
.944
.945
.439
.616
.779
.912
.973
.965
.957
.956
.953
.953
.952
.953
.436
.611
.773
.906
.973
.971
.964
.963
.960
.959
.959
.960
.433
.607
.769
.901
.972
.975
.969
.968
.965
.965
.965
.965
.432
.604
.765
.897
.970
.978
.973
.972
.970
.969
.969
.970
.430
.602
.761
.893
.969
.980
.976
.975
.972
.972
.972
.973
.429
.600
.758
.890
.967
.982
.978
.978
.975
.974
.974
.975
.427
.598
.755
.887
.965
.983
.980
.980
.977
.976
.976
.977
.424
.595
.752
.884
.963
.984
.982
.982
.979
.978
.978
.979
.423
.594
.750
.882
.962
.985
.984
.983
.981
.980
.980
.981
.422
.592
.749
.880
.960
.986
.985
.985
.982
.981
.981
.982
.421
.591
.747
.878
.959
.986
.986
.986
.983
.983
.983
.984
.420
.589
.746
.876
.958
.986
.987
.987
.985
.984
.984
.985
.418
.588
.744
.874
.956
.986
.988
.988
.986
.985
.986
.986
.417
.587
.742
.873
.954
.986
.989
.989
.987
.987
.987
.987
.417
.586
.741
.871
.953
.986
.989
.990
.988
.987
.987
.988
.416
.585
.740
.870
.952
.986
.990
.990
.988
.988
.988
.989
.415
.584
.739
.869
.951
.985
.990
.991
.989
.989
.989
.990
.414
.584
.738
.868
.951
.985
.991
.991
.990
.989
.989
.990
.414
.583
.737
.867
.950
.985
.991
.992
.990
.990
.990
.991
.413
.582
.736
.866
.949
.985
.991
.992
.991
.990
.990
.991
.413
.582
.736
.866
.948
.984
.991
.992
.991
.991
.991
.992
.412
.581
.735
.865
.948
.984
.991
.992
.991
.991
.991
.992
.412
.581
.735
.865
.947
.984
.992
.992
.992
.991
.992
.992
.412
.581
.734
.864
.947
.984
.992
.993
.992
.992
.992
.993
.412
.580
.734
.863
.946
.983
.992
.993
.992
.992
.992
.993
.411
.580
.733
.863
.946
.983
.992
.993
.992
.992
.992
.993
.411
.579
.733
.863
.945
.983
.992
.993
.993
.992
.993
.994
.411
.579
.732
.862
.945
.983
.992
.993
.993
.993
.993
.994
.411
.579
.732
.862
.944
.982
.992
.993
.993
.993
.993
.994
.411
.579
.732
.861
.944
.982
.992
.993
.993
.993
.993
.994

Similarity of Errors  (Transformers v.s. L-Bfgs)

(b) Transformer v.s. L-BFGS

Figure 18: Similarity of Errors on Ill-Conditioned Data with Quasi-Newton Methods. The best
matching steps are highlighted in yellow. Transformer also matches BFGS linearly, from layers 4
to 11. L-BFGS still suffers due to its limited memory but still better than Gradient Descentbecause
L-BFGS also attempts to approximate second-order information.

24


---Page Break---
A.3.2
Experiments with Noisy Linear Regression

We repeat the same experiments on noisy linear regression tasks with y = w⊤x + ε where ε ∼
N(0, σ2) with noise level σ = 0.1. As shown in Figure 19, Transformers still show superlinear
convergence on noisy linear regression tasks. Since the predictor is ˆw =
 
X⊤X + λI
† X⊤y for
some λ, the iterative newton’s method is applied to S = X⊤X + λI. Iterative Newton’s method
still keeps the same superlinear convergence rates. As it’s also shown in Figure 19, Transformers and
Iternative Newton’s rates match linearly, as in the noiseless linear regression tasks.

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

10
4

10
3

10
2

10
1

100

Errors

Transformer Errors v.s. # Layers

# In-Context Examples = 05
# In-Context Examples = 10
# In-Context Examples = 15
# In-Context Examples = 20
# In-Context Examples = 22
# In-Context Examples = 25
# In-Context Examples = 30
# In-Context Examples = 35

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 iterative newton Steps

.957
.903
.786
.670
.615
.586
.589
.576
.590
.583
.586
.584

.922
.921
.829
.712
.651
.620
.622
.611
.623
.617
.619
.618

.878
.925
.869
.761
.691
.659
.660
.650
.661
.655
.658
.657

.829
.914
.899
.812
.736
.702
.701
.692
.701
.696
.699
.698

.778
.890
.915
.861
.781
.745
.742
.735
.743
.739
.741
.740

.729
.858
.917
.901
.826
.788
.784
.777
.784
.781
.783
.782

.685
.822
.905
.929
.867
.829
.824
.818
.824
.821
.823
.822

.646
.786
.884
.944
.904
.867
.861
.856
.860
.859
.860
.859

.611
.752
.859
.946
.933
.901
.893
.889
.893
.892
.893
.892

.582
.722
.833
.939
.953
.929
.921
.917
.920
.919
.920
.920

.558
.696
.809
.926
.964
.951
.942
.939
.941
.941
.941
.941

.540
.676
.789
.912
.967
.966
.958
.955
.957
.957
.957
.957

.527
.662
.773
.899
.967
.976
.969
.967
.968
.968
.968
.969

.518
.652
.763
.889
.964
.982
.977
.976
.976
.976
.976
.977

.512
.645
.755
.881
.960
.985
.983
.982
.982
.982
.982
.983

.507
.640
.749
.875
.956
.986
.986
.986
.986
.987
.986
.987

.504
.636
.744
.870
.952
.985
.989
.989
.989
.990
.989
.990

.502
.633
.741
.866
.949
.984
.990
.990
.991
.991
.991
.992

.501
.631
.738
.863
.946
.983
.990
.991
.992
.992
.992
.993

.499
.629
.736
.861
.944
.981
.990
.991
.992
.993
.992
.993

.498
.628
.735
.859
.942
.980
.990
.991
.992
.993
.992
.993

.498
.627
.734
.858
.941
.979
.989
.991
.991
.992
.992
.993

.497
.626
.733
.857
.940
.978
.989
.990
.991
.992
.992
.993

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1

2

4

8

16

32

64

128

256

512

1024

2048

4096

 gradient descent (log scale) Steps

.962
.887
.788
.692
.638
.607
.610
.598
.611
.604
.607
.606

.918
.906
.839
.757
.699
.667
.668
.657
.669
.662
.665
.665

.848
.903
.885
.832
.774
.740
.739
.730
.740
.734
.737
.737

.761
.868
.905
.897
.850
.817
.813
.807
.814
.810
.812
.812

.677
.810
.889
.936
.912
.885
.880
.875
.880
.878
.879
.879

.607
.747
.850
.940
.952
.936
.929
.926
.929
.928
.929
.929

.558
.696
.807
.922
.967
.966
.961
.959
.960
.960
.961
.961

.529
.664
.775
.900
.967
.981
.978
.977
.978
.978
.978
.978

.513
.646
.756
.882
.960
.986
.987
.986
.987
.987
.987
.988

.505
.637
.745
.871
.953
.985
.990
.990
.991
.991
.991
.992

.501
.631
.739
.864
.947
.983
.991
.992
.992
.993
.993
.993

.498
.628
.735
.860
.943
.980
.990
.991
.992
.993
.993
.993

.497
.626
.733
.857
.940
.978
.989
.990
.991
.992
.992
.993

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 19: Experiment results on Noisy Linear Regression. (Top) Transformers have superlinear
convergence rate. (Bottom) Transformers match Iterative Newton’s rate and are exponentially faster
than Gradient Descent.

A.3.3
Experiments with a Non-Linear Function Class (2-Layer MLP)

To extend our experiments to non-linear cases, we adopt the same 2-layer ReLU neural network
studied by Garg et al. [2022]: see Fig. 5(c) in their paper. For any prompt (x1, y1, · · · , xt, yt),
instead of generating labels yk = w⋆⊤x as mainly studied in the paper, we study a 2-layer neural
network function class parameterized by W ∈Rdhidden×d, v ∈Rdhidden, a ∈Rdhidden, and b ∈R,
so that
yk = fW ,v,a,b(xk) = a⊤ReLU

W xk + v

+ b
(15)

Then we repeat the same probing experiments as in the main paper. As shown in Figure 20, even
on 2-layer neural network tasks with ReLU activation, Transformer shows superlinear convergence
rates. Transformer shows an exponentially faster convergence rate than Gradient Descent’s, because

25


---Page Break---
Gradient Descent’s steps are shown in log scale and the trend is linear – similar to Figure 9 in the
main paper.

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

10
1

100

Errors

2-Layer MLP with ReLU activation

# In-Context Examples = 25
# In-Context Examples = 40
# In-Context Examples = 80
# In-Context Examples = 100

Transformer Errors v.s. # Layers

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1

2

4

8

16

32

64

128

256

512

 gradient descent (log scale) Steps

.850
.851
.850
.858
.675
.662
.661
.619
.617
.620
.627
.612

.860
.861
.860
.867
.684
.671
.669
.626
.624
.627
.633
.619

.881
.883
.882
.890
.706
.691
.688
.647
.644
.646
.654
.638

.913
.916
.916
.922
.750
.729
.723
.684
.682
.682
.690
.673

.944
.949
.949
.952
.804
.777
.767
.732
.725
.722
.731
.714

.941
.947
.947
.955
.870
.838
.826
.796
.788
.783
.790
.777

.892
.897
.898
.915
.920
.901
.893
.864
.856
.852
.856
.845

.801
.805
.805
.834
.913
.929
.934
.917
.916
.914
.915
.910

.733
.734
.734
.764
.873
.919
.936
.938
.943
.942
.945
.948

.699
.701
.702
.738
.856
.905
.925
.935
.947
.947
.949
.955

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 20: Empirical Results on 2-Layer Neural Network Regression with ReLU activation function.
Transformers have superlinear convergence rates and match Gradient Descent’s convergence rate
exponentially

It would be interesting to ablate the activation function used in Equation (16). We further consider
the case when it’s using the Tanh activation instead of ReLU, i.e.

yk = fW ,v,a,b(xk) = a⊤Tanh

W xk + v

+ b
(16)

Repeating the same experiments as before, as shown in Figure 21, we find that Transformers use
the entire first 5 layers to pre-process and then only in the next few layers show exponentially faster
convergence rate compared to Gradient Descent. We further note that in both Figure 20 and Figure 21,
the cosine similarities between Transformers and Gradient Descent are significantly lower than the
experiments with linear regression tasks. This might due to the over-parameterization of the function
class and Transformers and Gradient Descent may arrive at different optima.

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

10
2

10
1

2 × 10
2

3 × 10
2

4 × 10
2

6 × 10
2

Errors

2-Layer MLP with Tanh activation

# In-Context Examples = 25
# In-Context Examples = 40
# In-Context Examples = 80
# In-Context Examples = 100

Transformer Errors v.s. # Layers

1
2
3
4
5
6
7
8
9
10
11
12
Transformer Layer Index

1

1

3

5

10

18

34

61

110

198

357

642

1156

2082

3748

 gradient descent (log scale) Steps

.980
.980
.980
.980
.978
.739
.675
.666
.663
.667
.659
.662

.980
.980
.980
.980
.978
.739
.675
.666
.663
.667
.659
.662

.982
.982
.982
.982
.981
.744
.680
.671
.669
.672
.665
.667

.981
.981
.981
.981
.979
.745
.679
.670
.668
.672
.665
.667

.981
.981
.981
.981
.979
.750
.683
.675
.672
.675
.668
.669

.980
.980
.980
.980
.977
.761
.698
.688
.684
.687
.680
.681

.973
.973
.973
.973
.970
.772
.712
.703
.699
.703
.696
.697

.962
.962
.962
.962
.959
.791
.736
.728
.722
.725
.718
.718

.943
.943
.943
.943
.940
.821
.771
.762
.756
.758
.751
.750

.889
.889
.889
.888
.886
.859
.831
.824
.818
.820
.814
.812

.807
.807
.808
.808
.807
.877
.876
.873
.869
.870
.864
.863

.717
.717
.717
.717
.716
.851
.888
.894
.892
.893
.890
.889

.662
.661
.661
.662
.661
.824
.880
.894
.895
.898
.898
.898

.618
.618
.618
.618
.617
.795
.863
.883
.885
.888
.888
.887

.616
.616
.616
.616
.615
.792
.853
.872
.876
.880
.879
.879

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 21: Empirical Results on 2-Layer Neural Network Regression with Tanh activation function.
Transformers have superlinear convergence rates and match Gradient Descent’s convergence rate
exponentially

It would be interesting for future research to explore further this function class of 2-layer MLP to
understand fully how Transformer solve the regression problem in-context and whether it achieves a
different optimum compared to alternative algorithms such as (Stochastic) Gradient Descent.

26


---Page Break---
A.4
Varying Transformer Architecture

A.4.1
Experiments on Transformers of Fewer Heads

In this section, we present experimental results from an alternative model configurations than the
main text. We show in the main text that Transformers learn second-order optimization methods
in-context where the experiments are using a GPT-2 model with 12 layers and 8 heads per layer. In
this section, we present experiments with a GPT-2 model with 12 layers but only 1 head per layer.

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.920
.920
.911
.909
.861
.785
.707
.671
.647
.631
.626
.619

.876
.876
.892
.912
.879
.823
.749
.709
.685
.667
.663
.655

.829
.829
.864
.901
.887
.859
.791
.750
.726
.706
.702
.694

.780
.780
.829
.877
.884
.887
.832
.792
.768
.746
.743
.735

.733
.733
.791
.845
.872
.906
.867
.832
.810
.787
.784
.776

.690
.690
.753
.811
.853
.913
.896
.869
.849
.825
.823
.816

.654
.654
.719
.777
.829
.910
.916
.900
.884
.861
.860
.852

.624
.624
.688
.746
.805
.900
.927
.924
.912
.894
.893
.885

.598
.598
.661
.719
.780
.885
.930
.939
.934
.920
.920
.913

.576
.576
.637
.695
.757
.867
.926
.947
.947
.941
.942
.935

.559
.559
.619
.676
.738
.851
.918
.948
.955
.956
.957
.951

.546
.546
.605
.662
.723
.837
.910
.947
.958
.966
.968
.963

.537
.537
.595
.651
.712
.826
.903
.944
.959
.973
.976
.972

.530
.530
.587
.644
.704
.817
.896
.940
.958
.977
.981
.979

.525
.525
.582
.638
.698
.810
.890
.936
.957
.980
.984
.984

.522
.522
.578
.634
.693
.806
.886
.933
.955
.981
.986
.987

.519
.519
.576
.631
.690
.802
.882
.930
.953
.981
.987
.989

.518
.517
.573
.629
.688
.799
.880
.928
.951
.981
.987
.991

.516
.516
.572
.627
.686
.798
.878
.926
.949
.980
.986
.992

.515
.515
.571
.626
.685
.796
.876
.924
.948
.979
.986
.992

.514
.514
.570
.625
.684
.795
.874
.923
.946
.978
.985
.992

.513
.513
.569
.624
.683
.793
.872
.920
.945
.976
.983
.991

.510
.510
.565
.620
.679
.787
.865
.914
.938
.969
.976
.984

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.954
.955
.915
.885
.840
.757
.685
.655
.630
.615
.609
.604
.584
.585
.645
.703
.764
.870
.923
.943
.945
.943
.944
.939
.552
.552
.610
.668
.729
.841
.911
.945
.955
.964
.966
.962
.539
.539
.596
.653
.713
.826
.902
.941
.956
.970
.973
.971
.532
.532
.588
.645
.705
.819
.897
.938
.955
.973
.977
.975
.528
.528
.583
.640
.700
.813
.892
.936
.954
.974
.979
.978
.525
.525
.581
.637
.697
.810
.889
.934
.953
.975
.980
.979
.522
.522
.578
.635
.694
.807
.887
.932
.952
.975
.980
.980
.520
.520
.576
.632
.692
.804
.885
.931
.951
.976
.981
.982
.519
.520
.575
.631
.691
.803
.884
.930
.950
.976
.981
.983
.519
.519
.574
.630
.689
.801
.882
.928
.950
.976
.981
.983
.518
.518
.573
.629
.688
.801
.881
.928
.949
.976
.981
.983
.517
.517
.572
.628
.687
.799
.880
.927
.948
.976
.982
.984
.516
.516
.572
.628
.687
.799
.880
.926
.948
.976
.982
.985
.516
.516
.571
.627
.686
.798
.879
.926
.948
.976
.981
.985
.516
.516
.570
.626
.686
.797
.878
.925
.947
.976
.981
.985
.516
.516
.571
.626
.685
.797
.878
.925
.947
.976
.982
.985
.515
.515
.570
.626
.685
.796
.877
.924
.947
.976
.982
.985
.514
.514
.569
.625
.684
.795
.876
.924
.946
.976
.982
.985
.513
.514
.568
.625
.684
.795
.876
.923
.946
.976
.981
.986
.514
.514
.569
.624
.683
.795
.876
.923
.946
.976
.982
.986
.513
.513
.568
.624
.683
.795
.875
.923
.946
.976
.982
.986
.513
.513
.568
.624
.683
.794
.875
.922
.945
.975
.981
.986
.513
.513
.568
.624
.683
.794
.875
.922
.945
.975
.981
.986
.513
.513
.567
.623
.682
.794
.874
.922
.945
.975
.981
.986
.513
.513
.567
.623
.682
.794
.875
.922
.945
.975
.981
.986
.513
.513
.568
.623
.682
.793
.874
.921
.944
.975
.981
.986

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 22: Similarity of Errors on an alternative Transformers Configuration. The best matching
steps are highlighted in yellow.

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.000
-.003
.581
.756
.740
.757
.734
.717
.713
.710
.711
.712

.001
-.003
.590
.767
.770
.802
.782
.766
.762
.759
.760
.761

.002
-.002
.588
.764
.789
.840
.827
.811
.807
.805
.806
.807

.002
-.002
.580
.751
.797
.868
.864
.850
.847
.844
.845
.846

.003
-.002
.567
.734
.796
.885
.891
.881
.878
.876
.877
.877

.003
-.002
.553
.716
.789
.893
.911
.905
.903
.901
.902
.902

.003
-.001
.541
.700
.780
.894
.923
.922
.921
.921
.922
.922

.003
-.001
.531
.686
.770
.891
.930
.934
.935
.936
.937
.937

.003
-.001
.522
.675
.761
.886
.932
.941
.944
.947
.948
.948

.002
-.000
.515
.666
.753
.880
.932
.945
.949
.954
.955
.955

.003
-.000
.510
.660
.746
.875
.930
.946
.952
.959
.961
.961

.003
.000
.506
.655
.741
.870
.928
.947
.954
.963
.964
.965

.003
.000
.503
.652
.738
.867
.926
.946
.954
.965
.967
.968

.003
.001
.501
.650
.736
.864
.924
.945
.954
.966
.968
.969

.003
.001
.500
.648
.734
.862
.922
.944
.954
.967
.969
.971

.003
.001
.499
.647
.732
.861
.921
.943
.953
.968
.970
.972

.003
.001
.498
.646
.731
.860
.920
.942
.953
.968
.970
.972

.003
.001
.498
.645
.731
.859
.919
.942
.952
.968
.970
.973

.003
.001
.498
.645
.730
.858
.919
.941
.952
.968
.970
.973

.003
.001
.497
.644
.730
.858
.918
.941
.951
.967
.970
.973

.003
.001
.497
.644
.729
.858
.918
.941
.951
.967
.970
.973

.003
.001
.497
.644
.729
.857
.918
.940
.951
.967
.970
.973

.003
.001
.497
.644
.729
.857
.918
.940
.951
.967
.969
.973

Similarity of Induced Weight w (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9
10
11
12
Trasformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.001
.119
.522
.684
.689
.702
.688
.680
.673
.669
.668
.669
.003
.020
.517
.675
.758
.885
.936
.951
.955
.961
.961
.963
.002
.019
.508
.662
.746
.876
.933
.952
.959
.968
.968
.970
.002
.019
.503
.657
.741
.871
.930
.952
.959
.970
.971
.973
.003
.019
.502
.655
.739
.869
.928
.951
.959
.971
.972
.974
.003
.019
.501
.653
.737
.867
.927
.950
.959
.971
.972
.975
.002
.019
.500
.652
.736
.866
.927
.950
.958
.971
.972
.975
.003
.019
.499
.652
.735
.865
.926
.950
.958
.972
.973
.976
.003
.019
.499
.651
.735
.865
.925
.949
.958
.972
.973
.976
.003
.019
.498
.651
.734
.864
.925
.949
.958
.972
.973
.976
.003
.019
.498
.650
.734
.864
.925
.949
.958
.972
.973
.976
.003
.019
.498
.650
.734
.863
.924
.948
.957
.972
.973
.976
.003
.019
.498
.650
.733
.863
.924
.948
.957
.972
.973
.976
.002
.019
.498
.650
.733
.863
.924
.948
.957
.972
.973
.977
.003
.019
.497
.649
.733
.863
.924
.948
.957
.972
.973
.977
.003
.019
.497
.649
.733
.862
.923
.948
.957
.972
.973
.977
.003
.019
.497
.649
.733
.862
.923
.948
.957
.972
.973
.977
.003
.019
.497
.649
.733
.862
.923
.948
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.923
.948
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.923
.947
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.923
.947
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.922
.947
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.923
.947
.957
.972
.973
.977
.003
.019
.497
.649
.732
.862
.923
.947
.956
.972
.973
.977
.003
.019
.496
.648
.732
.861
.922
.947
.956
.972
.973
.977
.003
.019
.497
.648
.732
.861
.922
.947
.956
.972
.973
.977
.003
.019
.496
.648
.732
.861
.922
.947
.956
.972
.973
.977

Similarity of Induced Weight w (Transformers v.s. Gradient Descent)

Figure 23: Similarity of Induced Weights on an alternative Transformers Configuration. The
best matching steps are highlighted in yellow.

We conclude that our experimental results are not restricted to a specific model configurations, smaller
models such as GPT-2 with 12 layers and 1 head each layer also suffice in implementing the Iterative
Newton’s method, and more similar than gradient descents, in terms of rate of convergence.

27


---Page Break---
A.4.2
Experiments on Transformers with More Layers

In this section, we investigate whether deeper models would behave similarly or differently. We work
on Transformers with 24 layers and 8 heads each.

1
2
3
4
5
6
7
8
9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
Trasformer Layer Index

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 Iterative newton Steps

.918
.918
.918
.918
.915

.872
.873
.873
.873
.922
.920

.913
.931

.926
.919

.924
.928

.915
.930
.921

.920
.928
.936

.924
.939

.933
.952

.955

.953
.964
.969

.966
.972

.965
.972

.981

.981

.981
.987

.987

.987
.988
.992

.988
.993
.993
.992
.992
.991
.991
.992
.993

.988
.993
.993
.992
.992
.992
.992
.992
.993

.992
.992
.992
.991
.991
.992
.993

Similarity of Errors  (Transformers v.s. Iterative Newton)

1
2
3
4
5
6
7
8
9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
Trasformer Layer Index

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

 Gradient descent Steps

.953
.953
.952
.953
.886
.840
.771
.744
.702
.681
.644

.577
.577
.577
.577
.712
.776
.833
.866
.899
.920
.948

.796
.833
.874
.899
.945
.961
.966

.961
.967

.960
.966

.976

.976

.976

.982

.982

.982

.984

.984
.987
.987
.986
.986
.986
.986
.986
.987

.984
.987
.987
.986
.986
.986
.986
.986
.987

Similarity of Errors  (Transformers v.s. Gradient Descent)

Figure 24: Similarity of Errors on a 24-layer Transformers Configuration. The best matching
steps are highlighted in yellow.

1
3
5
7
9
11
13
15
17
19
21
23
Transformer Layer Index

10
4

10
3

10
2

10
1

100

Errors

Transformer Errors v.s. # Layers

# In-Context Examples = 05
# In-Context Examples = 10
# In-Context Examples = 15
# In-Context Examples = 20
# In-Context Examples = 22
# In-Context Examples = 25
# In-Context Examples = 30
# In-Context Examples = 35

Figure 25: Transformers with 24 layers also converge superlinearly, similar to Iterative Newton.

28


---Page Break---
A.5
Heatmaps with Best-Matching Steps Help Compare Convergence Rates

In this section, we show the heatmaps with best-matching steps among known algorithms.

1
2
3
4
5
6
7
8
9
10 11 12 13 14 15 16 17 18 19 20 21 22 23
Iterative Newton Steps

1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
1000
1050
1100
1150
1200
1250
1300

Gradient Descent Steps

.974 .955 .923 .883 .838 .794 .754 .719 .691 .670

.697 .738 .782 .828 .873 .915 .948 .971 .984 .987 .985 .979

.830 .874 .913 .945 .967 .980 .987 .989

.982 .988 .989

.989 .989

.988 .990

.989 .990

.990

.989

.990

.990

.989 .990

.990

.990

.990

.990

.990

.990 .990 .990 .989 .984

.991 .990 .990 .989 .985

.990 .990 .990 .989 .984
(a) Iterative Newton v.s. Gradient Descent

1
2
3
4
5
6
7
8
9
10 11 12 13 14 15 16 17 18 19 20 21 22 23
Iterative Newton Steps

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

Iterative Newton Steps

1.000
.991

.991
1.000
.990

.990
1.000
.989

.989
1.000
.989

.989
1.000
.990

.990
1.000
.990

.990
1.000
.992

.992
1.000
.993

.993
1.000
.995

.995
1.000
.996

.996
1.000
.997

.997
1.000
.998

.998
1.000
.998

.998
1.000
.999

.999
1.000
.999

.999
1.000
1.000

1.000
1.000
1.000

1.000
1.000
1.000

1.000
1.000
1.000

1.000
1.000
1.000

1.000
1.000
1.000

1.000
1.000
.999

.999
1.000

(b) Iterative Newton v.s. Iterative Newton

1
2
3
4
5
6
7
8
9
10 11 12 13 14 15 16 17 18 19 20 21 22 23
Iterative Newton Steps

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

BFGS Steps

.985 .966 .934 .896 .858 .822

.925 .916 .906 .896 .885 .871

.817 .838

.852

.857 .883 .903

.855 .883 .908 .927

.877 .906 .929 .946 .958

.928 .949 .964

.945 .964 .978 .988

.979 .989 .994

.979 .989 .995 .997

.994 .997 .998 .999

.997 .998 .999 .999 .999

.998 .999 .999 1.0001.000

.999 1.0001.0001.000

1.0001.0001.000

1.0001.000

(c) Iterative Newton v.s. BFGS

Figure 26: Best-Matching Steps on Similarity of Residuals Help Compare Convergence Rates. (a:
top-left) When comparing Iterative Newton and Gradient Descent, there is an exponential trend –
showing Iterative Newton converges exponentially faster than Gradient Descent. (b: top-right) When
Iterative Newton is compared with itself in sub-figure, there is a linear trend – showing they have
the same convergence rate. (c: bottom) When Iterative Newton is compared to BFGS in sub-figure,
there a linear trend after there are enough steps for BFGS to approximate second-order information –
showing Iterative Newton and BFGS share a similar convergence rate after sufficient BFGS steps.

A.6
Definitions for Evaluating Forgetting

We measure the phenomenon of model forgetting by reusing an in-context example within {xi, yi}n
i=1
as the test example xtest. In experiments of Figure 5, we fix n = 20 and reuse xtest = xi. We
denote the “Time Stamp Gap” as the distance the reused example index i from the current time stamp
n = 20. We measure the forgetting of index i as

Forgetting(A, i) =
E
{xi,yi}n
i=1∼PD
MSE

A(xi | {xi, yi}n
i=1), yi

(17)

Note: the further away i is from n, the more possible algorithm A forgets.

29


---Page Break---
B
Detailed Proofs for Section 5

In this section, we work on full attention layers with normalized ReLU activation σ(·) = 1

nReLU(·)
given n examples.
Definition B.1. A full attention layer with M heads and ReLU activation is also denoted as Attn on
any input sequence H = [h1, · · · , hN] ∈RD×N, where D is the dimension of hidden states and N
is the sequence length. In the vector form,

˜ht = [Attn(H)]t = ht + 1

n

M
X

m=1

n
X

j=1
ReLU (⟨Qmht, Kmhj⟩) · Vmhj
(18)

Remark B.2. This is slightly different from the causal attention layer (see Definition 3.1) in that
at each time stamp t, the attention layer in Definition B.1 has full information of all hidden states
j ∈[n], unlike causal attention layer which requires j ∈[t].

B.1
Helper Results

We begin by constructing a useful component for our proof, and state some existing constructions
from Akyürek et al. [2022].
Lemma B.3. Given hidden states {h1, · · · , hn}, there exists query, key and value matrices Q, K, V
respectively such that one attention layer can compute Pn
j=1 hj.

Proof. We can pad each hidden state by 1 and 0’s such that h′
t ←

"ht
1
0d

#

∈R2d+1 . We con-

struct two heads where Q1 = K1 = Q2 =

"Od×d
Od×1
Od×d
O1×d
1
O1×d
Od×d
Od×1
Od×d

#

and K2 = −K1. Then

"Od×d
Od×1
Od×d
O1×d
1
O1×d
Od×d
Od×1
Od×d

#

h′
t =

"0d
1
0d

#

.

Let V1 = V2 =

O(d+1)×d
O(d+1)×(d+1)
nId×d
Od×(d+1)


so that Vm

"hj
1
0d

#

=

0d+1
nhj


.

We apply one attention layer to these 1-padded hidden states and we have

˜ht = h′
t + 1

n

2
X

m=1

n
X

j=1
ReLU
 
Qmh′
t, Kmh′
j

· Vmh′
j

= h′
t + 1

n

n
X

j=1

h
ReLU(1) + ReLU(−1)
i
·

0d+1
nhj



=

"ht
1
0d

#

+

0d+1
Pn
j=1 hj


=




ht
1
Pn
j=1 hj





(19)

Proposition B.4 (Akyürek et al., 2022). Each of mov, aff, mul, div can be implemented by a single
transformer layer. These four operations are mappings RD×N →RD×N, expressed as follows,

mov(H; s, t, i, j, i′, j′): selects the entries of the s-th column of H between rows i and j, and copies
them into the t-th column (t ≥s) of H between rows i′ and j′.

mul(H; a, b, c, (i, j), (i′, j′), (i′′, j′′)): in each column h of H, interprets the entries between i and
j as an a × b matrix A1, and the entries between i′ and j′ as a b × c matrix A2, multiplies these
matrices together, and stores the result between rows i′′ and j′′, yielding a matrix in which each
column has the form [h:i′′−1, A1A2, hj′′:]⊤. This allows the layer to implement inner products.

30


---Page Break---
div(H; (i, j), i′, (i′′, j′′)): in each column h of H, divides the entries between i and j by the
absolute value of the entry at i′, and stores the result between rows i′′ and j′′, yielding a matrix in
which every column has the form [h:i′′−1, hi:j/|hi′|, hj′′:]⊤.

aff(H; (i, j), (i′, j′), (i′′, j′′), W1, W2, b): in each column h of H, applies an affine transforma-
tion to the entries between i and j and i′ and j′, then stores the result between rows i′′ and j′′,
yielding a matrix in which every column has the form [h:i′′−1, W1hi:j + W2hi′:j′ + b, hj′′:]⊤. This
allows the layer to implement subtraction by setting W1 = I and W2 = −I.

B.2
Proof of Theorem 5.1

Theorem 5.1. For any k, there exist Transformer weights such that on any set of in-context examples
{xi, yi}n
i=1 and test point xtest, the Transformer predicts on xtest using x⊤
test ˆwNewton
k
. Here
ˆwNewton
k
are the Iterative Newton updates given by ˆwNewton
k
= MkX⊤y where Mj is updated as

Mj = 2Mj−1 −Mj−1SMj−1, 1 ≤j ≤k,
M0 = αS,

for some α > 0 and S = X⊤X. The dimensionality of the hidden layers is O(d), and the number
of layers is k + 8. One transformer layer computes one Newton iteration. 3 initial transformer layers
are needed for initializing M0 and 5 layers at the end are needed to read out predictions from the
computed pseudo-inverse Mk.

Proof. We break the proof into parts.

Transformers Implement Initialization T (0) = αS. Given input sequence H := {x1, · · · , xn},
with xi ∈Rd, we first apply the mov operations given by Proposition B.4 (similar to Akyürek et al.
[2022], we show only non-zero rows when applying these operations):

x1
· · ·
xn

mov
−→

x1
· · ·
xn
x1
· · ·
xn


(20)

We call each column after mov as hj. With an full attention layer, one can construct two heads with

query and value matrices of the form Q⊤
1 K1 = −Q⊤
2 K2 =

Id×d
Od×d
Od×d
Od×d


such that for any

t ∈[n], we have

2
X

m=1
ReLU (⟨Qmht, Kmhj⟩) = ReLU(x⊤
t xj) + ReLU(−x⊤
t xj) = ⟨xt, xj⟩
(21)

Let all value matrices Vm = nα

Id×d
Od×d
Od×d
Od×d


for some α ∈R. Combining the skip connections,

we have

˜ht =

xt
xt


+ 1

n

n
X

j=1
⟨xt, xj⟩nα

xj
0


=

xt
xt


+

"
α
Pn
j=1 xjx⊤
j

xt
0

#

=

xt + αSxt
xt


(22)

Now we can use the aff operator to make subtractions and then

xt + αSxt
xt


aff
−→

(xt + αSxt) −xt
xt


=

αSxt
xt


(23)

We call this transformed hidden states as H(0) and denote T (0) = αS:

H(0) =
h
h(0)
1
· · ·
h(0)
n
i
=

T (0)x1
· · ·
T (0)xn
x1
· · ·
xn


(24)

Notice that S is symmetric and thereafter T (0) is also symmetric.

Transformers implement Newton Iteration. Let the input prompt be the same as Equation (24),

H(0) =
h
h(0)
1
· · ·
h(0)
n
i
=

T (0)x1
· · ·
T (0)xn
x1
· · ·
xn


(25)

31


---Page Break---
We claim that the ℓ’s hidden states can be of the similar form

H(ℓ) =
h
h(ℓ)
1
· · ·
h(ℓ)
n
i
=

T (ℓ)x1
· · ·
T (ℓ)xn
x1
· · ·
xn


(26)

We prove by induction that assuming our claim is true for ℓ, we work on ℓ+ 1:

Let Qm = ˜Qm


Od
−n

2 Id
Od
Od



|
{z
}
G

, Km = ˜
Km


Id
Od
Od
Od



|
{z
}
J

where ˜Q⊤
1 ˜
K1 := I, ˜Q⊤
2 ˜
K2 := −I and

V1 = V2 =

Id
Od
Od
Od



|
{z
}
J

. A 2-head self-attention layer, with ReLU attentions, can be written has

h(ℓ+1)
t
= [Attn(H(ℓ))]t = h(ℓ)
t
+ 1

n

2
X

m=1

n
X

j=1
ReLU
D
Qmh(ℓ)
t , Kmh(ℓ)
j
E
· Vmh(ℓ)
j
(27)

where

2
X

m=1
ReLU
D
Qmh(ℓ)
t , Kmh(ℓ)
j
E
· Vmh(ℓ)
j

=
h
ReLU

(Gh(ℓ)
t )⊤˜Q⊤
1 ˜
K1
| {z }
I

(Jh(ℓ)
j )

+ ReLU

(Gh(ℓ)
t )⊤˜Q⊤
2 ˜
K2
| {z }
−I

(Jh(ℓ)
j )
i
· (Jh(ℓ)
j )

=
h
ReLU((Gh(ℓ)
t )⊤(Jh(ℓ)
j )) + ReLU(−(Gh(ℓ)
t )⊤(Jh(ℓ)
j ))
i
· (Jh(ℓ)
j )

= (Gh(ℓ)
t )⊤(Jh(ℓ)
j )(Jh(ℓ)
j )

= (Jh(ℓ)
j )(Jh(ℓ)
j )⊤(Gh(ℓ)
t )

(28)

Plug in our assumptions that h(ℓ)
j
=

T (ℓ)xj
xj


, we have Jh(ℓ)
j
=

T (ℓ)xj
0d


and Gh(ℓ)
t
=

−n

2 xt
0d


,

we have

h(ℓ+1)
t
=

T (ℓ)xt
xt


+ 1

n

n
X

j=1


T (ℓ)xj
0d

 
T (ℓ)xj
0d

⊤
−n

2 xt
0d



=

T (ℓ)xt −1

2
Pn
j=1(T (ℓ)xj)(T (ℓ)xj)⊤xt
xt



=

"
T (ℓ)xt −1

2T (ℓ) Pn
j=1 xjx⊤
j

T (ℓ)⊤xt
xt

#

=

"
T (ℓ) −1

2T (ℓ)ST (ℓ)⊤
xt
xt

#

(29)

Now we pass over an MLP layer with

h(ℓ+1)
t
←h(ℓ+1)
t
+

Id
Od
Od
Od


h(ℓ+1)
t
=

"
2T (ℓ) −T (ℓ)ST (ℓ)⊤
xt
xt

#

(30)

Now we denote the iteration
T (ℓ+1) = 2T (ℓ) −T (ℓ)ST (ℓ)⊤
(31)

We find that T (ℓ+1)⊤= T (ℓ+1) since T (ℓ) and S are both symmetric. It reduces to

T (ℓ+1) = 2T (ℓ) −T (ℓ)ST (ℓ)
(32)

This is exactly the same as the Newton iteration.

Transformers can implement ˆwTF
ℓ
= T (ℓ)X⊤y. Going back to the empirical prompt format
{x1, y1, · · · , xn, yn}. We can let parameters be zero for positions of y’s and only rely on the skip

32


---Page Break---
connection up to layer ℓ, and the H(ℓ) is then




T (ℓ)xj
0
xj
0
0
yj





n

j=1

. We again apply operations from

Proposition B.4:




T (ℓ)xj
0
xj
0
0
yj





n

j=1

mov
−→




T (ℓ)xj
T (ℓ)xj
xj
0
0
yj





n

j=1

mul
−→





T (ℓ)xj
T (ℓ)xj
xj
0
0
yj
0
T (ℓ)yjxj





n

j=1

(33)

Now we apply Lemma B.3 over all even columns in Equation (33) and we have

Output =

n
X

j=1





T (ℓ)xj
0
yj
T (ℓ)yjxj



=

ξ
T (ℓ) Pn
j=1 yjxj


=

ξ
T (ℓ)X⊤y


(34)

where ξ denotes irrelevant quantities. Note that the resulting T (ℓ)X⊤y is also the same as Iterative
Newton’s predictor ˆwk = MkX⊤y after k iterations. We denote ˆwTF
ℓ
= T (ℓ)X⊤y.

Transformers can make predictions on xtest by

 ˆwTF
ℓ
, xtest

.

Now we can make predictions on text query xtest:

 ξ
xtest
ˆwTF
ℓ
xtest


mov
−→




ξ
xtest
ˆwTF
ℓ
xtest
0
ˆwTF
ℓ



mul
−→





ξ
xtest
ˆwTF
ℓ
xtest
0
ˆwTF
ℓ
0

 ˆwTF
ℓ
, xtest





(35)

Finally, we can have an readout layer βReadOut = {u, v} applied (see Definition 3.3) with u =
[03d
1]⊤and v = 0 to extract the prediction

 ˆwTF
ℓ
, xtest

at the last location, given by xtest. This
is exactly how Iterative Newton makes predictions.

To Perform k steps of Newton’s iterations, Transformers need O(k) layers.

Let’s count the layers:

• Initialization: mov needs O(1) layer; gathering αS needs O(1) layer; and aff needs O(1)
layer. In total, Transformers need O(1) layers for initialization.

• Newton Iteration: each exact Newton’s iteration requires O(1) layer. Implementing k
iterations requires O(k) layers.

• Implementing ˆwTF
ℓ
: We need one operation of mov and mul each, requiring O(1) layer
each. Apply Lemma B.3 for summation also requires O(1) layer.

• Making prediction on test query: We need one operation of mov and mul each, requiring
O(1) layer each.

Hence, in total, Transformers can implement k-step Iterative Newton and make predictions accord-
ingly using O(k) layers.

Remark B.5. We note that Giannou et al. [2023] used 13 layers to compute one Newton Iteration,
and in our construction, we need only one Transformer layer (with one attention layer and one MLP
layer) to compute one Newton Iteration. At the same time, we didn’t use Akyürek et al. [2022] for
constructing Newton Iterations. Akyürek et al. [2022] is applied to initialize Newton and for reading
out the prediction.

In our construction, only the initialization and read-out prediction components use causal attention
and softmax because Akyürek et al. [2022]’s construction is applied. To be more specific, those are
the first 3 layers in initializing Iterative Newton and the last 5 layers in reading out the predictions
from the computed pseudo-inverse. All the layers corresponding to the Iterative Newton updates are
using full attention and normalized ReLU activations.

33


---Page Break---
Remark B.6. We note that our proof can be extended to causal attention for n sufficiently larger than
d. Under causal attention (see Definition 3.1) with normalized ReLU activation, Equation (29) can be

rewritten as follows, given t > d, we first choose G =

Od
−1

2Id
Od
Od


, where the coefficient on the

upper right block is −1

2 instead of −n

2 originally. Then

h(ℓ+1)
t
=

T (ℓ)xt
xt


+ 1

t

t
X

j=1


T (ℓ)xj
0d

 
T (ℓ)xj
0d

⊤
−1

2xt
0d



=

T (ℓ)xt −1

2
1
t
Pt
j=1(T (ℓ)xj)(T (ℓ)xj)⊤xt
xt



=

"
T (ℓ)xt −1

2T (ℓ) 
1
t
Pt
j=1 xjx⊤
j

T (ℓ)⊤xt
xt

#

=

"
T (ℓ) −1

2T (ℓ) ˆΣT (ℓ)⊤
xt
xt

#

(36)

where ˆΣ = 1

t
Pt
j=1 xjx⊤
j is the estimate of the covariance matrix given seen in-context examples
{xj, yj}t
j=1 so far. Since t > d, ˆΣ is an unbiased estimate for Σ ≈1

nS if n is sufficiently large. The
rest of the proof follows similarly, up to the perturbation introduced by the error in the estimate of ˆΣ.

We also note when t < d, the estimate ˆΣ = 1

t
Pt
j=1 xjx⊤
j is no longer a valid covariance matrix
since it’s singular. Then this gives different T (ℓ+1) for different time stamp t < d and such error may
propagate in our proof. Hence, a formal extension to causal models requires extensive analysis of the
error bounds and it is beyond the scope of this work. Nonetheless, we provide a plausible direction of
such an extension.

B.3
Iterative Newton as a Sum of Moments Method

Recall that Iterative Newton’s method finds S† as follows

M0 =
2
∥SS⊤∥2
|
{z
}
α

S⊤,
Mk = 2Mk−1 −Mk−1SMk−1, ∀k ≥1.
(37)

We can expand the iterative equation to moments of S as follows.

M1 = 2M0 −M0SM0 = 2αS⊤−4α2S⊤SS⊤= 2αS −4α2S3.
(38)

Let’s do this one more time for M2.

M2 = 2M1 −M1SM1 = 2(2αS −4α2S3) −(2αS −4α2S3)S(2αS −4α2S3)

= 4αS −8α2S3 −4α2S3 + 16α3S5 −16α4S7

= 4αS −12α2S3 + 16α3S5 −16α4S7.

(39)

We can see that Mk are summations of moments of S, with respect to some pre-defined coefficients
from the Newton’s algorithm. Hence Iterative Newton is a special of an algorithm which computes
an approximation of the inverse using second-order moments of the matrix,

Mk =

2k+1−1
X

s=1
βsSs
(40)

with coefficients βs ∈R.

We note that Transformer circuits can represent other sum of moments other than Newton’s method.
We can introduce different coefficients βi than in the proof of Theorem 5.1 by scaling the value
matrices or through the MLP layers.

34


---Page Break---
B.4
Estimated weight vectors lie in the span of previous examples

What properties can we infer and verify for the weight vectors which arise from Newton’s method? A
straightforward one arises from interpreting any sum of moments method as a kernel method.

We can expand Ss as follows

Ss =

 
t
X

i=1
xix⊤
i

!s

=

t
X

i=1




X

j1,··· ,js−1
⟨xi, xj1⟩

s−2
Y

v=1


xjv, xjv+1



xix⊤
js−1.
(41)

Then we have

ˆwt = MtX⊤y =

2t+1−1
X

s=1
βsSsX⊤y

=

2t+1−1
X

s=1
βs






t
X

i=1




X

j1,··· ,js−1
⟨xi, xj1⟩

s−2
Y

v=1


xjv, xjv+1



xix⊤
js−1






(
t
X

i=1
yixi

)

=

2t+1−1
X

s=1
βs




t
X

i=1



X

j1,··· ,js
yjs ⟨xi, xj1⟩

s−1
Y

v=1


xjv, xjv+1



xi





=

t
X

i=1




2t+1−1
X

s=1

X

j1,··· ,js
βsyjs ⟨xi, xj1⟩

s−1
Y

v=1


xjv, xjv+1





|
{z
}
ϕt(i|X,y,β)

xi

=

t
X

i=1
ϕt(i | X, y, β) xi

(42)

where X is the data matrix, β are coefficients of moments given by the sum of moments method and
ϕt(·) is some function which assigns some weight to the i-th datapoint, based on all other datapoints.
Therefore if the Transformer implements a sum of moments method (such as Newton’s method),
then its induced weight vector ˜wt(Transformers | {xi, yi}t
i=1) after seeing in-context examples
{xi, yi}t
i=1 should lie in the span of the examples {xi}t
i=1:

˜wt(Transformers | {xi, yi}t
i=1)
?= Span{x1, · · · , xt} =

t
X

t=1
aixi
for coefficients ai.
(43)

We test this hypothesis. Given a sequence of in-context examples {xi, yi}t
i=1, we fit coefficients
{ai}t
i=1 in Equation (43) to minimize MSE loss:

{ˆai}t
i=1 =
arg min
a1,a2,··· ,at∈R

 ˜wt(Transformers | {xi, yi}t
i=1) −

t
X

t=1
aixi



2

2
.
(44)

We then measure the quality of this fit across different number of in-context examples t, and visualize
the residual error in Figure 27. We find that even when t < d, Transformers’ induced weights still
lie close to the span of the observed examples xi’s. This provides an additional validation of our
proposed mechanism.

35


---Page Break---
1
5
10
15
20
25
30
35
40
# of In-Context Examples

0.00000

0.00002

0.00004

0.00006

0.00008

0.00010

0.00012

Linearity Error (MSE)

Linearity Error  vs. # In-Context Examples

Transformers
OLS

Figure 27: Verification of hypothesis that the Transformers induced weight vector w lies in the span
of observed examples {xi}.

C
Computes

All experiments involving fine-tuning GPT2 models to learn in-context linear regressions are trained
on one NVIDIA A6000. Linear probing experiments also used one NVIDIA A6000.

D
License

We used PyTorch Paszke et al. [2019] as our code framework and we used PyTorch implementation
of LSTMs. PyTorch is licensed under the Modified BSD license.

We used GPT-2 Model as our backbone, and it’s released under MIT License. We used trained GPT-2
checkpoints for linear regression by Garg et al. [2022] and it’s released under MIT License.

E
Limitations

In this work, our analyses of Transformers are mostly based on only one simple task: linear regression.
It might not be able to extrapolate to any arbitrary algorithmic tasks. It would be interesting for future
work to extend such analysis to an extensive class of problems.

F
Broader Impacts

This paper presents work whose goal is to advance the field of Machine Learning. Through a
mechanistic understanding of Transformers, the backbone of modern large language models (LLMs),
this work can help advance building safe and trustworthy models.

36


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

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

Justification: The limitations is discussed in Section E.

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

Justification: We provide a theorem in Section 5 with its proof in Appendix B.

37


---Page Break---
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
Justification: We provide the details of experimental settings in §4.
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

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer:[Yes]

38


---Page Break---
Justification: We will release codes and data generation processes.

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

Answer:[Yes]

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

39


---Page Break---
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
Justification: The detail of the computing resourse is provided at §C
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
Justification: The authors have read the NeurIPS Code of Ethics and made sure the paper
follows the NeurIPS Code of Ethics in every aspect.
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
Justification: The potential societal impact is discussed in §F.
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

40


---Page Break---
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

Justification: Our paper works on simple linear regression tasks. We believe there is no such
risk.
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
Justification: See §D for details.
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

41


---Page Break---
Justification: We did not introduce any new assets in this paper.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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

42


---Page Break---
