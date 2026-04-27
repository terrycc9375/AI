Universal In-Context Approximation
By Prompting Fully Recurrent Models

Aleksandar Petrov, Tom A. Lamb, Alasdair Paren, Philip H.S. Torr, Adel Bibi
Department of Engineering Science
University of Oxford
aleks@robots.ox.ac.uk

Abstract

Zero-shot and in-context learning enable solving tasks without model fine-tuning,
making them essential for developing generative model solutions. Therefore, it is
crucial to understand whether a pretrained model can be prompted to approximate
any function, i.e., whether it is a universal in-context approximator. While it was
recently shown that transformer models do possess this property, these results rely
on their attention mechanism. Hence, these findings do not apply to fully recurrent
architectures like RNNs, LSTMs, and the increasingly popular SSMs. We demon-
strate that RNNs, LSTMs, GRUs, Linear RNNs, and linear gated architectures such
as Mamba and Hawk/Griffin can also serve as universal in-context approximators.
To streamline our argument, we introduce a programming language called LSRL
that compiles to these fully recurrent architectures. LSRL may be of independent
interest for further studies of fully recurrent models, such as constructing inter-
pretability benchmarks. We also study the role of multiplicative gating and observe
that architectures incorporating such gating (e.g., LSTMs, GRUs, Hawk/Griffin)
can implement certain operations more stably, making them more viable candidates
for practical in-context universal approximation.

1
Introduction

Until recently, solving a task with machine learning required training or fine-tuning a model on a
dataset matching the task at hand. However, large foundation models exhibit the ability to solve new
tasks without being specifically fine-tuned or trained for them: often it is sufficient to simply prompt
them in the right way.
Prompting has been especially successful because of in-context learning:
the ability to modify the model’s behavior with information provided within the input sequence,
without changing the underlying model parameters (Brown et al., 2020). Yet, we know little about
the theoretical properties of prompting. It is not even clear if there are limits to what can be achieved
with prompting or, conversely, whether it is possible to prompt your way into any behaviour or task.

This can be framed as a universal approximation question. Classically, universal approximation
results show how a class of tractable functions, such as neural networks, approximates another class of
concept functions, e.g., all continuous functions on a bounded domain, with arbitrary accuracy. This
is often done by showing that one can choose model parameters that approximate the target function.
However, in-context learning poses a different challenge as the model parameters are fixed. Instead,
a part of the input (the prompt) is modified to cause the model to approximate the target function.
Hence, we define universal in-context approximation to be the property that there exist fixed weights
such that the resulting model can be prompted to approximate any function from a concept class.
Understanding whether a model can be a universal in-context approximator is especially important as
most commercial models are accessible exclusively via a prompting interface (La Malfa et al., 2023).

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
In-context learning has been almost exclusively studied in conjunction with the transformer archi-
tecture (Vaswani et al., 2017). This is likely because in-context abilities appear once the models are
large enough (Wei et al., 2021) and most large models have been transformer-based. On the subject of
universal in-context approximation, Wang et al. (2023) were first to show that a transformer possesses
this property by discretising and memorising all possible functions in the model weights. Memo-
risation is not needed, though, and even small transformers can be universal approximators when
prompted Petrov et al. (2024). Both results, however, critically depend on the attention mechanism of
the transformer architecture (Bahdanau et al., 2015).

Still, generative models are not restricted to attention-based architectures: there are the “classic”
recurrent neural networks (RNNs, Amari, 1972), long short-term memory models (LSTMs, Hochreiter
and Schmidhuber, 1997) and gated recurrent units (GRUs, Cho et al., 2014). Recently, Linear RNN
models (also known as state-space models or SSMs) were proposed as a scalable alternative to
the transformer architecture (Orvieto et al., 2023; Fu et al., 2023a) and have started to outperform
similarly-sized transformers when multiplicative gating is added (Gu and Dao, 2023; De et al., 2024;
Botev et al., 2024). Furthermore, despite in-context learning being associated with the transformer,
recent empirical results show in-context learning in SSMs, RNNs, LSTMs and even convolutional
models (Xie et al., 2022; Akyürek et al., 2024; Lee et al., 2024).

Yet, despite their ability to be in-context learners, there is little known about the theoretical properties
of these fully recurrent architectures. As these architectures become more and more widely used,
understanding their in-context approximation abilities is increasingly more important for their safety,
security and alignment. We show that, in fact, many of these architectures, similarly to transformers,
can be universal in-context approximators. Concretely, our contributions are as follows:

i. We develop Linear State Recurrent Language (LSRL): a programming language that com-
piles to different fully recurrent models. Programming in LSRL is akin to “thinking like a
recurrent model”. LSRL programs can then be implemented exactly as model weights.
ii. Using LSRL, we construct Linear RNN models that can be prompted to act as any token-
to-token function over finite token sequences, or to approximate any continuous function.
These results also hold for RNNs, LSTMs, GRUs and Hawk/Griffin models (De et al., 2024).
iii. We present constructions with and without multiplicative gating. However, we observe that
the constructions without these gates depend on numerically unstable conditional logic.
iv. Nevertheless, we show that multiplicative gates lead to more compact and numerically stable
models, making it more likely that universal in-context approximation properties arise in
models utilising them, such as LSTMs, GRUs and the latest generation of Linear RNNs.

2
Preliminaries

Fully recurrent architectures.
In this work, we focus exclusively on fully recurrent neural network
architectures. Recurrent models operate over sequences. Concretely, consider an input sequence
(x1, . . . , xN) with xt ∈X, X being some input space. We will refer to the elements of the input
sequence as tokens even if they are real-valued vectors. A recurrent model g : X ⋆→Y maps a
sequence of inputs to an output in some output space Y. These models are always causal, namely:
yt = g(x1, . . . , xt).
(1)
We will abuse the notation and refer to (y1, ..., yt)=(g(x1), ..., g(x1, ..., xt)) as simply g(x1, ..., xt).
We will also separate the input sequence into a query (q1, ..., qn) and a prompt (p1, ..., pN). The
prompt specifies the target function f that we approximate while the query designates the input at
which we evaluate it. Contrary to the typical setting, we will place the query before the prompt.1

There are various neural network architectures that fall under the general framework of Eq. (1). The
quintessential one is the RNN. It processes inputs one by one with only a non-linear state being
passed from one time step to the other. A model g can thus be stacked RNN layers, each one being:
st = σ(Ast−1 + Bxt + b),
yt = ϕ(st),
(Classic RNN)
(2)

1That is necessitated by the limited capacity of the state variables. As the model is fixed, in order to increase
the precision of the approximation, we can only increase the prompt length. If the prompt is before the query, it
would have to be compressed into a fixed-size state, limiting the approximation precision even with increased
prompt lengths. But if the query has a fixed size, it can be stored in a fixed-size state variable exactly.

2


---Page Break---
with A, B, b and the initial state value s0 being model parameters, σ a non-linear activation function
and ϕ a multi-layer perceptron (MLP) with ReLU activations. We assume that σ is always a ReLU
to keep the analysis simpler. The non-linearity in the state update can make the model difficult to
train (vanishing and exploding gradients, Bengio et al., 1994). Therefore, Linear RNNs have been
proposed as regularizing the eigenvalues of A can stabilise the training dynamics (Orvieto et al.,
2023). Linear RNNs also admit a convolutional representation, making them trainable in parallel (Gu
et al., 2021; Fu et al., 2023a). Linear RNNs drop the non-linearity from the state update in Eq. (2):
st = Ast−1 + Bxt + b,
yt = ϕ(st).
(Linear RNN)
(3)

The fully linear state updates do not affect the expressivity of the models, as non-linear activations
are nevertheless present in the MLP layers ϕ between the linear state update layers (Wang and
Xue, 2023; Boyd and Chua, 1985). The state-of-the-art Linear RNN models also utilise some form
of multiplicative gating (Gu and Dao, 2023; De et al., 2024; Botev et al., 2024). While specific
implementations can differ, we can abstract it as the following Gated Linear RNN architecture:
st = Ast−1 + Bxt + b,
yt = γ(xt) ⊙ϕ(st),
(Gated Linear RNN)
(4)

with γ being another MLP and ⊙being the element-wise multiplication operation (Hadamard product).
Eq. (4) encompasses a range of recently proposed models. For example, one can show that any model
consisting of L stacked Gated Linear RNN layers, with γ and ϕ with k layers, can be represented as
a L(k+2)-layer Hawk or Griffin model (De et al., 2024). The conversions are described in detail in
App. E. We can similarly add multiplicative gating to the classic RNN architecture:
st = σ(Ast−1 + Bxt + b),
yt = γ(xt) ⊙ϕ(st),
(Gated RNN)
(5)

Eq. (5) may appear unusual but it is related to the well-known GRU (Cho et al., 2014) and LSTM
(Hochreiter and Schmidhuber, 1997) architectures. Same as the case with Griffin/Hawk, any Gated
RNN can be represented as a L(k+2)-layer GRU or LSTM model (details in Apps. C and D). As a
result, if there exists a Gated RNN model that is a universal in-context approximator (which we later
show to be the case), then there also exist GRU and LSTM models with the same property.

Theoretical understanding of in-context learning.
Beyond the question of universal in-context
approximation, there have been attempts to theoretically understand in-context learning from various
perspectives. The ability to learn linear functions and perform optimization in-context has been
extensively explored in the context of linear regression (Garg et al., 2022; Akyürek et al., 2022;
von Oswald et al., 2023a; Fu et al., 2023b; Zhang et al., 2023; Ahn et al., 2023), kernel regression
(Han et al., 2023) and dynamical systems (Li et al., 2023). Furthermore, studies have explored
how in-context learning identifies and applies the appropriate pretraining skill (Xie et al., 2022;
Coda-Forno et al., 2023; Bai et al., 2023). It has also been shown that transformers can construct
internal learning objectives and optimize them during the forward pass (von Oswald et al., 2023b;
Dai et al., 2023). However, these studies almost exclusively focus on the transformer architecture,
and the applicability of their findings to fully recurrent models remains unclear.

Approximation theory.
Let X and Y be normed vector spaces. Take a set of functions C ⊆YX
from X to Y called a concept space. Take also a set of nicely behaved functions H ⊂YX , called
hypothesis space. H could be any set that we have tools to construct and analyse, e.g., all polynomials
or all neural networks of a particular architectural type. Approximation theory is concerned with how
well functions in H approximate functions in C. We say that H universally approximates C over a
compact domain D (or that H is dense in C) if for every f∈C and ϵ>0 there exist a h∈H such that
supx∈D |f(x)-h(x)|≤ϵ. There is a long history of studying the concept class of continuous functions
and hypothesis classes of single hidden layer neural networks (Cybenko, 1989; Barron, 1993) or
deeper models (Hornik et al., 1989; Telgarsky, 2015). The concept class of sequence-to-sequence
functions has been shown to be universally approximated with the hypothesis classes of transformers
(Yun et al., 2019), RNNs (Schäfer and Zimmermann, 2006) and Linear RNNs (Wang and Xue, 2023).

The hypothesis spaces in this work are different. The model is fixed and only the prompt part of
the input is changed, i.e., all learnable parameters are in the prompt. Take a recurrent model g as in
Eq. (1) with fixed model parameters and a query length n. The hypothesis class is all functions that
result by calling g on the user query followed by the prompt and taking the last n′ outputs:

HDn
g
= {(q1, . . . , qn) 7→g(q1, . . . , qn, p1, . . . , pN)[-n′:] | ∀pi ∈D, N > 0}.
(6)

3


---Page Break---
Original
DAG
Simpliﬁed
Path Graph

Linear
ReLU
Input
LinState
Concat

Node types

Compiled Linear RNN
LSRL program

Example input and output

ctr0s+ctr1s
(1→2)

Linear
(2→2)

Linear
(2→1)

ReLU
(2→2)

input
(1)

is_zero
(1→1)

ctr0s
(1→1)

ctr1s
(1→1)

Concat
((1,1)→2)

Concat
((1,1)→2)

output
(2→1)

ReLU
(1→1)

ReLU
(1→1)

Substract
(2→1)

Substract
(1→1)

Scale
(1→1)

Scale
(1→1)

input
(1)

ReLU

0
1
1
0
1
1
0
1
1
0

0
0
1
0
1
1
1
1
1
1

Figure 1: Compilation of an LSRL program to a Linear RNN. An example of a simple LSRL
program that takes a sequence of 0s and 1s as an input and outputs 1 if there have been more 1s than
0s and 0 otherwise. The LSRL compiler follows the rules in App. A to simplify the computation
DAG into a path graph. The resulting path graph can be represented as a Linear RNN with one layer.

1 ForEach:

2 input = Input(dim =1)

3 ctr1s=LinState(input ,

4
A=[[1]] , B=[[1]] ,

5
init_state =[[0]])

6 # equivalent to 1-input

7 is_zero = f_not(input)

8 ctr0s = LinState( is_zero ,

9
A=[[1]] , B=[[1]] ,

10
init_state =[[0]])

11 # equiv. to f_step(ctr1s -ctr0s)

12 output=f_larger(

13
ctr1s ,ctr0s ,mu=10)

14 return output

The domain D of pi and qi can be continuous embeddings in Rd or discrete tokens V = {1, ..., V }.

Note that each h∈Hg is identified by a prompt (p1, ..., pN) but is a function with domain all possible
queries (q1, ..., qn). Therefore, finding a hypothesis h∈Hg that approximates a target function f is
equivalent to finding the prompt of that hypothesis. The approximation properties of Hg in Eq. (6)
depend on the architecture of g, as well as its specific parameters.

We study the recurrent architectures in Eqs. (2) to (5) and their ability to approximate contin-
uous functions over real-valued vectors and to represent discrete maps over tokens (which cor-
responds to how language models are used in practice). We consider the following classes of
functions. Cvec=(Rdout)[0,1]din contains all continuous functions from the unit hypercube to Rdout,
while Ctok={h∈(Vl)Vl | h causal} all causal functions from l tokens to l tokens. The hypothesis
classes are Hvec(g) corresponding to Eq. (6) with D=[0, 1]din, n=n′=1 and g some fixed model of
one of the four architectures in Eqs. (2) to (5), and Htok(g) with D=V and n=n′=l.

3
Linear State Recurrent Language (LSRL)

We can construct the weights for universal in-context models with the architectures in Eqs. (2) to (5)
by hand but this is labour-intensive, error-prone, difficult to interpret, and the specific weights would
be architecture-dependent. Working at such a low level of abstraction can also obfuscate common
mechanisms and design patterns, making it more difficult to appreciate both the capabilities and
the constraints of fully recurrent architectures. Instead, we propose a new programming language:
Linear State Recurrent Language (LSRL).2 LSRL programs compile to the four architectures in
Eqs. (2) to (5). Conversely, any Linear RNN can be represented as an LSRL program, making LSRL
a versatile tool for studying the capabilities of recurrent models. Later, in Secs. 4 to 6 we make use of
LSRL to develop programs that are universal approximators for Cvec and Ctok, thus showing that all
four architectures can be universal in-context approximators.

LSRL syntax.
An LSRL program specifies how a single element is processed and how the recurrent
states are updated for the next element. LSRL programs always start with an Input(x) = x
with an x of a fixed dimension. Only one Input can be declared in a program. Linear layers
and ReLUs are also supported: Lin[A, b](x) := Ax + b, ReLU(x) := max(0, x). The unique
component of LSRL, however, is its LinState operation implementing the linear state update in
Linear RNNs (Eq. (3)): LinState[A, B, b, s0](xt) := Ast−1 + Bxt + b, where the state st−1
is the output of the call this node at step t −1. LinState is the only way information can be
passed from previous tokens to the current one. We also provide a Concat operation that combines

2Our implementation of LSRL is available at https://github.com/AleksandarPetrov/LSRL

4


---Page Break---
1
ForEach:
2
input = Input(dim=1+ d_in+d_out)

3
# counter needed to know whether we are looking at the query or the prompt
4
const_1 = f_constant(input , 1)
5
counter_vector = LinState(input=const_1 , A=ones(d_in ,d_in), B=ones(d_in ,1), init_state=zeros(d_in ,1))

6
# copy the query in a state (only when the counter is 1)
7
q_update = f_ifelse(cond=f_smaller(counter_vector , 1.5), t=input[: d_in], f=input[: d_in ]*0)
8
q = LinState(input=q_update , A=eye(d_in), B=eye(d_in), init_state=zeros(d_in ,1))

9
# the following operations will only change the output when counter > 1
10
# the step size is the first element of every prompt element
11
step_size = Linear(input=input[0], A=ones(d_in ,1), b=zeros(d_in ,1))

12
# using it we can compute the upper bounds of the current prompt cell
13
lb = input[1 : 1 + d_in]
14
ub = lb + step_size

15
# now check if q is in this cell (the bump should be 1 on all dimensions)
16
q_in_bump_componentwise = f_bump(q, lb, ub)
17
bump_sum = Linear(input=q_in_bump_componentwise , A=ones(1,d_in), b=zeros (1,1))
18
in_cell = f_larger(bump_sum , d_in - 0.5)
19
in_and_processing = f_and(in_cell , f_larger(counter , 0.5))

20
# if counter >1 and this cell contains q, add the value to the output state
21
update = f_ifelse(cond=f_larger(in_and_processing ,0.5), t=input[-d_out:], f=input[-d_out :]*0)
22
y = LinState(input=update , A=eye(d_out), B=eye(d_out), init_state=zeros(d_out ,1))
23
return y

Listing 1: LSRL program for universal approximation in-context for continuous functions. The
inputs are q = [q′⊤, 0⊤
dout+1]⊤with q′ ∈[0, 1]din being the query value at which we want to evaluate
the function, then followed by prompts describing the target function as in Eq. (8).

variables: Concat(x, y) := (x1, ..., x|x|, y1, ..., y|y|). Finally, to support gating architectures we
also implement a rudimentary Multi operation that splits its input into two sub-arrays and returns
their element-wise multiplication: Multi(x) := x[ : |x|/2] ⊙x[|x|/2 : ]. Naturally, Multi requires
that x has even length. These six operations can be composed into a direct acyclic graph (DAG) with
a single source node (the Input variable) and a single sink node (marked with a return statement).

Such a program operates over a single token xt passed to Input, while a recurrent model needs to
operate over sequences. Thus, we wrap the program into a ForEach loop that passes each element
individually for the DAG to output a variable denoted by a return clause. Each element is processed
by the exact same program, with the only difference being that the state of the LinState variables is
changing between iterations. You can see an example of a small LSRL program in Fig. 1.

Expressiveness limitations.
ForEach does not behave like the typical for loop: only the states
are accessible between iterations, i.e., you cannot use the output of a linear layer at step t in any
computation at step t + 1. Furthermore, as the program is a DAG and only states of LinState nodes
are passed between iterations, variables computed in latter operations of a previous time step are not
accessible as inputs in earlier layers (with respect to the topological sorting of the computation graph).
This leads to a key programming paradigm in LSRL: a LinState update cannot depend non-linearly
on its own state. That includes it depending on a variable that depends on the LinState itself and
conditional updates to the state. Such a dependency would break the DAG property of the program.3
This poses serious limitations on what algorithms can be expressed in a Linear RNN and makes
programming them challenging. Still, in Sec. 4 we show how carefully constructing state updates and
auxiliary variables can nevertheless allow to program some limited conditional behaviours.

Compilation.
Any LSRL program without Multi nodes can be compiled to a Linear RNN (Eq. (3))
or to a Gated Linear RNN (Eq. (4)). If the program has Multi nodes, then it cannot be compiled to a
Linear RNN as the multiplicative gating cannot be implemented exactly. However, it can be compiled
to a Gated Linear RNN. To compile an LSRL program to a Linear (Gated) RNN, we first parse the
program to build a computation graph. This is a DAG with a single source (the Input node) and a
single sink (the return statement of the ForEach loop). At the same time, a Linear (Gated) RNN
can be represented as a path graph (no branching) with the six basic operations as nodes. Therefore,
the compilation step needs to transform this DAG into a path graph. We achieve that by iterativly
collapsing the first branching point into a single node. The exact rules that achieve that are described
in App. A. Later, in Sec. 6, we will show how any Linear (Gated) RNN can be converted into a
non-linear (Gated) RNN, hence, how we can compile LSRL programs to these architectures as well.

3For example, we cannot implement an operation that adds one to the state and squares it at each time step:
st+1 = (st + 1)2 or an operation that performs conditional assignment st+1 = 0 if (st > 5) else st.

5


---Page Break---
Target function
Approximated function

q:

y:

(0,0)

0

(q1,q2)

0

(q1,q2)

0

(q1,q2)

0

(q1,q2)
(q1,q2)

Model

Model

Model

Model

Model

States:

Inputs:

1.0

0.0
0.0

1.0

0.0
1.0

1.0
1.0

0.0
0.0

0.0
1.0

Figure 2: Intuition behind the LSRL program for universal in-context approximation for
continuous functions in Lst. 1. Our target function f has input dimension din = 2 and output
dimension dout = 1. Each input dimension is split into two parts, hence δ = 1/2. We illustrated an
example input sequence of length 5: one for the query and four for the prompt tokens corresponding
to each of the discretisation cells. The query (q1, q2) falls in the cell corresponding to the third prompt
token. We show how the two LinState variables in the program are updated after each step. Most
notably, how the state holding the output y is updated after p3 is processed.

Syntactic sugar.
To make programming easier, we define several convenience functions. For
instance, we can Slice variables x[l:u] via sparse Lin layers. We can also sum variables and element-
wise multiplication with scalars (implemented as Lin layers). For logical operations we also need step
functions which can be approximated with ReLUs: f_step[µ](x) := ReLU(µx) −µReLU(x −1/µ),
where µ is a positive constant controlling the quality of the approximation. We can also approximate
bump functions (1 between l and u and 0 otherwise): f_bump[l, u, µ](x) := f_step[µ](x −l) −
f_step[µ](x −u). Similarly, we can approximate conjunction (f_and), disjunction (f_or), negation
(f_not), and comparison operators (f_larger and f_smaller). See App. F for the definitions.

Critically, we need also a conditional operator that assigns a value t(x) if a certain condition is met
and another value f(x) otherwise. One way to implement this is:

f_ifelse[cond, t, f, λ](x) := ReLU(-λ cond(x)+f(x)) + ReLU(-λ f_not(cond(x))+t(x))
−ReLU(-λ cond(x)-f(x)) −ReLU(-λ f_not(cond(x))-t(x)), (7)

where λ is a constant that is larger than any absolute value that t(x) and f(x) can attain. This
construction, however, is not numerically stable and we will study alternatives in Sec. 5. We provide
both numerical (SciPy.sparse, Virtanen et al. 2020) and symbolic (SymPy, Meurer et al. 2017)
backends with the second being crucial for programs that are not numerically stable.

Prior work on encoding algorithms in model weights.
A similar approach to developing a pro-
gramming language that compiles to model weights was already done for the transformer architecture
with the RASP language (Weiss et al., 2021) and the Tracr compiler (Lindner et al., 2023). They were
predominantly created as a tool for interpretability research. In a sense, RASP is to a transformer as
LSRL is to a (Linear) (Gated) RNN. Hence, can be used to develop benchmarks for interpretabil-
ity methods for fully-recurrent architectures. However, while RASP can only express a subset of
transformer models, LSRL is isomorphic to the set of all (Gated) Linear RNNs (though not to the
non-linear ones). That means that any (Gated) Linear RNN can be represented and analysed as an
LSRL program and vice versa. Hence, the limitations of what you can express in LSRL are also
limitations of what a Linear (Gated) RNN can do. Namely: (i) we cannot have exact multiplicative
interactions between inputs without multiplicative gates, and (ii) we cannot have state variable updates
depending non-linearly on their previous iterations or in any way on a variable that depends on them.

6


---Page Break---
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
O
T
T
O
T
T
O
T
T
O
T
T
O
T
T
output

output_regs
??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? O?? OT? OTT OTT OTT OTT OTT OTT OTT OTT OTT OTT OTT OTT OTT
t_updates
??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? O?? ?T? ??T ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ???
copy_and_on
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
1
1
1
0
0
0
0
0
0
0
0
0
0
0
0
started_on
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
18
18
18
18
18
18
18
18
18
18
18
18
18
18
18
all_matching
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
0
0
0
0
0
matching
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
1
1
1
0
0
0
0
0
1
0
0
0
0
0
0
0
0
0
q
C?? CA? CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN
global_ctr
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

input
C
A
N
A
U
S
V
I
E
B
U
L
S
O
F
C
A
N
O
T
T
D
E
N
C
O
P
E
N
G
L
O
N
query
prompt

final outputs
discarded outputs

Figure 3: Intuition behind the LSRL program for universal in-context approximation for
discrete functions in Lst. 2. Our keys and values have length n=3 and represent countries and
capitals, e.g., AUStria7→VIEnna, BULgaria7→SOFia, and so on. The query is CAN for Canada and the
final n outputs are OTT (Ottawa). We show the values of some of the variables in Lst. 2 at each step,
with the LinState variables being marked with arrows. For cleaner presentation we are tokenizing
letters as 07→?, 17→A, 27→B, etc. Vertical separators are for illustration purposes only.

4
Universal In-Context Approximation with Linear RNNs

We proceed with building LSRL programs that are universal in-context approximators: one for
approximating continuous functions (Cvec), and one for maps between token sequences (Ctok).

4.1
Approximating continuous functions in Cvec

The idea behind the approximation for continuous functions is to discretise the domain into a grid and
approximate the function as constant in each cell of the grid. This technique is commonly used for
showing universal approximation using the step activation function (Blum and Li, 1991; Scarselli and
Tsoi, 1998). However, it is not obvious how to implement this approach in-context when information
across input tokens can be only combined linearly. Consider a target function f : [0, 1]din→[0, 1]dout
and a discretization step δ. Our approach is to describe the value of f in each of the discretization
cells as a single prompt token. For the cell with lower bounds l1, . . . , ldin and their respective upper
bounds l1+δ, ..., ldin+δ, the corresponding prompt token is a (din+dout+1)-dimensional vector:

p = [δ, l1, . . . , ldin, ¯y1, . . . ¯ydout]⊤,
(8)

where ¯y is the value of f at the centre of that cell: ¯y = f(l1+δ/2, ..., ldin+δ/2). Each prompt token
describes the size of the cell (the discretisation step δ), its starting lower bound, and the value of the
target function at the centre of the cell. Thus, ⌈1/δ⌉din such tokens, one for each cell, are sufficient to
describe the piece-wise constant approximation of f. A query q′ ∈[0, 1]din can fall in only one of the
cells. We pad it with zeros and encode it as the first input element: q = [q′⊤, 0⊤
dout+1]⊤, followed by
the prompt. Our program will extract and save q′ to a state and then process the prompt tokens one
at a time until it finds the one whose cell contains q′. The target function value for this cell will be
added to an accumulator state. If the current cell does not contain q′, then 0 is instead added.Hence,
the accumulator’s final value corresponds to the value of f at the centre of the cell containing q′. The
full LSRL program is provided in Lst. 1 and an illustration for din = 2, dout = 1, δ = 1/2 is shown in
Fig. 2. The prompt length required to approximate an L-Lipschitz function f (w.r.t. the ℓ2 norm) to
precision ϵ is N = (2ϵ/L√din)-din = O(ϵ-din) (see App. B for the proof). Asymptotically, this is as
good as one can hope without further assumptions on the target function. This is also better than the
best known result for the same problem for transformers: O(ϵ-10-14din-4d2
in) in Petrov et al. 2024.

4.2
Approximating functions over token sequences in Ctok

Sec. 4.1 focused on continuous functions but recurrent architectures are often used to model natural
language whose domain is tokens. Thus, we also look at modelling maps over a discrete domain.
Any function from n tokens to n tokens taking values in V = {1, . . . , V } can be represented as
a dictionary whose keys and values are in Vn. Therefore, a simple way to represent this function
in-context is to first provide the n tokens corresponding to the query and then a sequence of 2n tokens
corresponding to key and value pairs (see Fig. 3 for an illustration of the setup). The model stores the

7


---Page Break---
1
ForEach:
2
input = Input(dim=1)

3
# counter needed to know whether we are looking at the query or the prompt
4
const_1 = f_constant(input , 1)
5
global_ctr = LinState(input=const_1 , A=[[1]] , B=[[1]] , init_state =[[ -1]])

6
# counters mod[n] and mod[2n]
7
mod_n_ctr = f_modulo_counter(input , n)
8
mod_2n_ctr = f_modulo_counter(input , 2*n)

9
# which mode are we in (looking at the query , comparing query with key , or copying value to state)
10
is_prompt = f_larger(global_ctr , n-0.5)
11
is_compare_mode = f_larger(mod_2n_ctr , n-0.5)
12
is_copy_mode = f_and(is_prompt , f_not(is_compare_mode))
13
is_first_token_for_copy = f_and(is_copy_mode ,f_smaller(mod_n_ctr , 0.5))

14
# update the state holding the query if this is one of the first n tokens
15
tq=f_ifelse(f_smaller(is_prompt , 0.5), t=input , f=input *0)
16
tqs=[ f_ifelse(f_and(f_larger(mod_n_ctr ,i-0.5), f_smaller(mod_n_ctr ,i+0.5)),t=tq,f=tq*0) for i in 1..n]
17
q = LinState(input=Concat(tqs), A=eye(n), B=eye(n), init_state=zeros(n,1)) # query

18
# if we are in compare mode (looking at keys), check if this token matches the corresponding one in the query
19
qs=[ f_ifelse(f_and(f_larger(mod_n_ctr ,i-0.5),f_smaller(mod_n_ctr ,i+0.5)),t=q[i],f=q[i]*0) for i in 1..n]
20
cor_q_el = Linear(input=Concat(qs), A=ones(1,n), b=zeros (1,1))
21
matching = f_and(f_and(f_larger(input , cor_q_el -0.5),f_smaller(input , cor_q_el +0.5)),is_compare_mode)

22
# keep a buffer of the last last n+1 match values , the +1 because we can only read the buffer after writing
23
buffer = LinState(input=matching , A=shift_matrix , B=[[0] for _ in 1..n], [[1]]) , init_state=zeros(n+1,1))
24
buffer_sum = Linear(input=buffer , A=[[1 for _ in 1..n], [0]], b=zeros(1, 1))
25
all_matching = f_larger(buffer_sum , n-0.5)

26
# if all are matching and it 's the first token in the value part of the (key , value) pair , then mark this as

the iterations when we start copying to state
27
matching_and_first_for_copy = f_and(all_matching , is_first_token_for_copy)
28
t_started_on_update = f_ifelse(matching_and_first_for_copy ,t=global_ctr , f=global_ctr *0)
29
started_on = LinState(input=t_started_on_update , A=eye(1), B=eye(1), init_state=zeros(1, 1))

30
# copying to state for n iterations after started_on
31
copy_and_on = f_and(is_copy_mode , f_smaller(global_ctr , started_on+n))
32
mod_n_eq_i = [ f_and(f_larger(mod_n_ctr ,i-0.5), f_smaller(mod_n_ctr ,i+0.5)) for i in 1..n ]
33
t_updates_should_update = [f_and(copy_and_on , mod_n_eq_i[i]) for i in 1..n]
34
t_updates = [f_ifelse(f_larger(t_updates_should_update[i], 0.5), t=input , f=input *0) for i in 1..n]
35
output_regs = [LinState(input=update , A=eye(1), B=eye(1), init_state=zeros (1,1)) for update in t_updates]

36
# finally , read out the value from the corresponding output register in order to output from the model
37
t_outputs = [f_ifelse(f_larger(mod_n_eq_i[i], 0.5), t=output_regs[i], f=output_regs[i]*0) for i in 1..n]

38
return Linear(input=Concat(t_outputs), A=ones(1,n), b=zeros (1,1))

Listing 2: LSRL program for universal in-context approximation of discrete functions. The
inputs are q1, ..., qn (the query tokens), followed by pairs of keys and values from the map we are
approximating. The last n outputs are the value corresponding to the key matching the query.

query in a state and processes the key-value pairs one by one by comparing the key (the first n tokens)
with the query. If they match, then the value (the next n tokens) is copied into a state that keeps it and
repeatedly outputs it. This continues until the end of the prompt, at which point the last n outputted
tokens will be the value corresponding to the key matching the query. This is essentially a dictionary
lookup. However, as shown in Lst. 2, implementing dictionary lookup in a linear recurrent model is
much less straightforward than executing dict[key] in a general-purpose programming language.

Lst. 2 can appear daunting at first so we would like to clarify the non-trivial aspects. First, we need to
count how far we are into every set of n or 2n tokens. This can be done with mod n and mod 2n
operations but implementing modulo for arbitrary large inputs is not possible with ReLU MLPs
(Ziyin et al., 2020). Therefore, we implement this with LinState as f_modulo_counter which has
a unit-length state that is rotated 1/n or 1/2n revolutions per iteration, with the angle corresponding
to the modulo value (App. F.7). Second, we need to do dynamic indexing to copy the i-th input in
a subsequence to the i-th element of a state and vice-versa. Dynamic indexing, however, cannot
be succinctly represented in a Linear RNN. We work around this with temporary variables that are
non-zero only at the i-th coordinates (see Lines 16, 17, 19, 20, 32 to 35, 37 and 38). Finally, in order
to compare whether all n elements in the query and the key match, we need to remember whether
the previous n pairs were matching. As RNNs do not have attention, we implement this short-term
memory buffer as a LinState with a shift matrix (Line 23).

5
Stable Universal In-Context Approximation with Gated Linear RNNs

The ReLU-based conditional operator is not numerically stable.
The LSRL programs in Lsts. 1
and 2 for approximating functions in respectively Cvec and Ctok rely on the f_ifelse conditional
assignment operator in Eq. (7) in order to implement different behaviours depending on whether

8


---Page Break---
No noise
Amount of noise added to the parameters

Wrong tokens [%]

10-9
10-7
10-5
10-3

100

80

60

40

20

0

Functions over token sequences 
Continous functions

Average difference

No noise
Amount of noise added to the parameters
10-9
10-7
10-5

10-1

105

102

108

1011

1014

1017

10-3

Implementation of f_ifelse

Original

With
gating
Without
gating

Optimized
Step-based (optimized)
Multiplicative
Multiplicative optimized

Figure 4: Robustness of the various f_ifelse implementations to model parameter noise. We
show how the performance of the two universal approximation programs in Lsts. 1 and 2 deteriorates
as we add Gaussian noise of various magnitudes to the non-zero weights of the resulting compiled
models. As expected, the original f_ifelse implementation in Eq. (7) exhibits numerical precision
errors at the lowest noise magnitude. For the token sequence case, numerical precision errors are
present in all samples even in the no-noise setting. Hence, the original f_ifelse implementation
is less numerically robust while the implementations with multiplicative gating are the most robust.
For Lst. 1 (approximating Cvec) we report the Euclidean distance between the target function value
and the estimated one over 10 queries for 25 target functions. For Lst. 2 we report the percentage of
wrong token predictions over 5 queries for 25 dictionary maps. Lower values are better in both cases.

we are processing the query or specific parts of the prompt. This operator is not numerically stable.
The first term in Eq. (7) relies on cond(x) being exactly zero if the condition is not met. In this
way, multiplying it with −λ would be 0 and f(x) would be returned. However, if cond(x) is not
identically 0 but has a small positive value, then −λcond(x) can “overpower” f(x) resulting in the
ReLU output being 0. In our experience, this is not a problem when processing inputs through the
LSRL program step-by-step. However, de-branching the DAG into a path graph —which is necessary
in order to uncover the equivalent Linear RNN— appears to introduce such numerical instabilities
which occasionally result in wrong outputs as conditional assignments will be 0 when they should
not. This problem is more prominent in Lst. 2 which is longer (more debranching steps) and has more
f_ifelse operations: it gets most tokens wrong because of that instability (see Original, No noise in
Fig. 4). To this end, we support LSRL with a symbolic backend (based on SymPy) that performs the
debranching steps exactly. Using it, both programs always produce the correct output.

This numerical instability highlights a critical practical limitations of the universal approximation
results in Sec. 4: if the models are not numerically stable, it is unlikely that they occur in practice by
training models using gradient descent. This section shows how to improve the numerical stability of
Eq. (7) and obtain more realistic recurrent models that are universal approximators in-context.

Removing unnecessary terms in Eq. (7).
Eq. (7) has 4 separate ReLU terms. The first two handle
the cases when t(x) and f(x) are positive and the second two when they are negative. Therefore, if
we know that one or both of these will always be non-negative, we can drop the corresponding terms.
Additionally, if f(x) is always 0, then the first and third terms can be safely dropped. Similarly,
the second and fourth are unnecessary if f(x) ≡0. All f_ifelse in Lsts. 1 and 2 fall in this case
and hence can be simplified. We will refer to this f_ifelse implementation that is aware of the
attainable values of t(x) and f(x) as optimized. As it reduces the number of numerically unstable
ReLU operations in the model, we expect that it will improve the stability of the compiled models.
We experimented with adding various levels of noise to the non-zero model parameters, and, as the
results in Fig. 4 show, optimized is indeed more numerically robust than original.

Step-based implementation.
We can get rid of the input sensitivity of Eq. (7) using f_step:

f_ifelse[cond, t, f, λ](x) := ReLU(-λ+λf_step(1/2-cond(x))+f(x)) + ReLU(-λ+λf_step(cond(x)-1/2)+t(x))
−ReLU(-λ+λf_step(1/2-cond(x))-f(x)) −ReLU(-λ+λf_step(cond(x)-1/2)-t(x)). (9)

We can also apply the optimisation strategy here. While this implementation is robust to noise in the
input it appears to be more sensitive to parameter noise, as shown in Fig. 4.

9


---Page Break---
Numerically stable f_ifelse with multiplicative gates.
Removing the unused ReLU terms in the
original f_ifelse reduces the opportunities for numerical precision issues to creep in but does not
solve the underlying problem. The multiplicative gating present in the Linear Gated RNN (Eq. (4))
and Gated RNN models (Eq. (5)) can help via implementing a numerically stable conditional operator:

f_ifelse[cond, t, f](x) := cond(x) ⊙t(x) + f_not(cond(x)) ⊙f(x),
(10)

where the element-wise product is implemented in LSRL with Concat and Multi. We will refer to the
implementation of f_ifelse in Eq. (10) as multiplicative. Similarly to original implementation of
f_ifelse in Eq. (7), we can drop the t(x) and f(x) term if they are equal to zero (multiplicative
optimized). If cond(x) is not exactly zero, cond(x) ⊙t(x) will result in a small error to the output
but, in contrast to the original implementation, is not going to cause a discontinuity in the output of
the operation. Therefore, Eq. (10) should be more robust to numerical precision issues than Eq. (7).
Fig. 4 shows that this is the case in practice with Lsts. 1 and 2 being more robust to parameter noise
when using multiplicative gates compared to the ReLU-based implementations. Therefore, Linear
Gated RNNs (Eq. (4)) —to which models with multiplicative gates can be compiled— are more
likely than Linear RNNs (Eq. (3)) to exhibit universal approximation properties in practice.

6
Universal In-context Approximation with Non-linear (Gated) RNNs

Secs. 4 and 5 showed how universal approximation of continuous and token-to-token functions can
be implemented in LSRL and compiled to respectively Linear RNNs and Linear Gated RNNs. This
section aims to address the situation with non-linear state updates, that is, the cases of classic and
gated RNNs (Eqs. (2) and (5)). Concretely, we show how every linear (Gated) RNN can be converted
to a non-linear (Gated) RNN. The key idea is that the ReLU applied to the state updates in the
non-linear architectures is an identity operation if its inputs are positive. Hence, we can split the states
in positive and negative components, flip the sign of the negative component, pass them separately
through the ReLU—which will act as an identity as all elements will be non-negative— and then fuse
the positive and negative components back together in the A matrix at the next time step:

st = Ast−1 + Bxt + b
yt = ϕ(st).
≡


s+
t
s−
t


= ReLU

A
−A
−A
A

 
s+
t
s−
t


+

B
−B


xt +

b
−b



yt = ϕ

[I
−I]

s+
t
s−
t


.

(11)

Using Eq. (11) we can compile any LSRL program to an RNN (Eq. (2)) or a Gated RNN (Eq. (5)).
This includes Lsts. 1 and 2. Hence, RNNs and Gated RNNs can be universal in-context approximators
for continuous and token-to-token functions. As any Gated RNN can be represented as a GRU model
(App. C) or an LSTM (App. D), these models are too universal in-context approximators.

7
Discussion and Conclusions

We developed LSRL: a programming language for specifying programs expressible with recurrent
neural architectures. We then used LSRL to show that various architectures —from the humble RNN
to the state-of-the-art Linear Gated RNNs— can all be universal approximators in-context.

Safety and security implications.
If a model can be prompted to approximate any function, then
preventing it from exhibiting undesirable behaviours (i.e., alignment) might be impossible. Therefore,
it is important to further study the safety and security implications of these properties.

Limitations.
In this work we provide constructive existence results: that is, we show that there
can exist models with various recurrent architectures that are universal in-context approximators.
However, the present theory is not sufficient to analyse whether a given model has this property.
That is a much more difficult question that would require a very different approach. We also
assume no restrictions on the A matrix in the state update equations. However, many state-of-the-art
models impose structural constraints on A (e.g., it being diagonal) for the sake of fast training and
inference (Gu et al., 2020, 2021; Gupta et al., 2022). It is not directly obvious whether such structural
restrictions would affect the universal in-context approximation abilities of these architectures. In
practice, however, the compiled matrices are very sparse and often diagonal. Therefore, it is highly
likely that our results translate to models with structural restrictions.

10


---Page Break---
Acknowledgements

We would like to thank Simon Schug for pointing us to relevant works on fully recurrent models.
We are also extremely grateful to Juuso Haavisto for his insight on building compilers. This work
is supported by a UKRI grant Turing AI Fellowship (EP/W002981/1) and the EPSRC Centre for
Doctoral Training in Autonomous Intelligent Machines and Systems (EP/S024050/1). Adel Bibi
acknowledges funding from the KAUST Office of Sponsored Research (OSR-CRG2021-4648) and
support from Google Cloud through the Google Gemma 2 Academic Program GCP Credit Award.

References

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. 2023. Transformers learn to imple-
ment preconditioned gradient descent for in-context learning. In Advances in Neural Information
Processing Systems.

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2022. What learning
algorithm is in-context learning? Investigations with linear models. In The Eleventh International
Conference on Learning Representations.

Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. 2024. In-context language learning:
Arhitectures and algorithms. arXiv preprint arXiv:2401.12973.

Shun-ichi Amari. 1972. Learning patterns and pattern sequences by self-organizing nets of threshold
elements. IEEE Transactions on Computers, C-21(11):1197–1206.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by
jointly learning to align and translate. In International Conference on Learning Representations.

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. 2023. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection. In Advances in neural information
processing systems.

Andrew R Barron. 1993. Universal approximation bounds for superpositions of a sigmoidal function.
IEEE Transactions on Information Theory, 39(3):930–945.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer.
arXiv preprint arXiv:2004.05150.

Yoshua Bengio, Patrice Simard, and Paolo Frasconi. 1994. Learning long-term dependencies with
gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166.

Edward K Blum and Leong Kwan Li. 1991. Approximation theory and feedforward networks. Neural
networks, 4(4):511–515.

Aleksandar Botev, Soham De, Samuel L Smith, Anushan Fernando, George-Cristian Muraru, Ruba
Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, et al. 2024.
RecurrentGemma: Moving past transformers for efficient open language models. arXiv preprint
arXiv:2404.07839.

Stephen Boyd and Leon Chua. 1985. Fading memory and the problem of approximating nonlinear
operators with Volterra series. IEEE Transactions on Circuits and Systems, 32(11):1150–1161.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models
are few-shot learners. In Advances in Neural Information Processing Systems.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder–decoder
for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods
in Natural Language Processing (EMNLP).

Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, and Eric Schulz. 2023.

Meta-in-context learning in large language models. In Advances in Neural Information Processing
Systems, pages 65189–65201.

11


---Page Break---
George Cybenko. 1989. Approximation by superpositions of a sigmoidal function. Mathematics of
control, signals and systems, 2(4):303–314.

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. 2023. Why
can GPT learn in-context? Language models secretly perform gradient descent as meta-optimizers.
In Findings of the Association for Computational Linguistics: ACL 2023.

Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert
Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. 2024. Griffin:
Mixing gated linear recurrences with local attention for efficient language models. arXiv preprint
arXiv:2402.19427.

Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re. 2023a.

Hungry Hungry Hippos: Towards language modeling with state space models. In International
Conference on Learning Representations.

Deqing Fu, Tian-Qi Chen, Robin Jia, and Vatsal Sharan. 2023b. Transformers learn higher-order
optimization methods for in-context learning: A study with linear models.
arXiv preprint
arXiv:2310.17086.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. 2022. What can transformers
learn in-context? A case study of simple function classes. In Advances in Neural Information
Processing Systems.

Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. 2000. Learning to forget: Continual prediction
with lstm. Neural computation, 12(10):2451–2471.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces.
arXiv preprint arXiv:2312.00752.

Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. 2020. HiPPO: Recurrent
memory with optimal polynomial projections. In Advances in Neural Information Processing
Systems.

Albert Gu, Karan Goel, and Christopher Re. 2021. Efficiently modeling long sequences with
structured state spaces. In International Conference on Learning Representations.

Ankit Gupta, Albert Gu, and Jonathan Berant. 2022. Diagonal state spaces are as effective as
structured state spaces. In Advances in Neural Information Processing Systems.

Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. 2023. In-context learning of large language models
explained as kernel regression. arXiv preprint arXiv:2305.12766.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural Computation,
9(8):1735–1780.

Kurt Hornik, Maxwell Stinchcombe, and Halbert White. 1989. Multilayer feedforward networks are
universal approximators. Neural networks, 2(5):359–366.

Emanuele La Malfa, Aleksandar Petrov, Simon Frieder, Christoph Weinhuber, Ryan Burnell, Raza
Nazar, Anthony G. Cohn, Nigel Shadbolt, and Michael Wooldridge. 2023. Language Models as a
Service: Overview of a new paradigm and its challenges. arXiv preprint arXiv:2309.16573.

Ivan Lee, Nan Jiang, and Taylor Berg-Kirkpatrick. 2024.
Exploring the relationship between
model architecture and in-context learning ability. In International Conference on Learning
Representations.

Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. 2023. Trans-
formers as algorithms: Generalization and stability in in-context learning.
In International
Conference on Machine Learning.

David Lindner, János Kramár, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, and Vladimir
Mikulik. 2023. Tracr: Compiled transformers as a laboratory for interpretability. In Advances in
Neural Information Processing Systems.

12


---Page Break---
Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondˇrej ˇCertík, Sergey B. Kirpichev, Matthew
Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean
Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik
Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štˇepán Rouˇcka, Ashutosh Saboo,
Isuru Fernando, Sumith Kulal, Robert Cimrman, and Anthony Scopatz. 2017. SymPy: Symbolic
computing in Python. PeerJ Computer Science, 3.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu,
and Soham De. 2023. Resurrecting recurrent neural networks for long sequences. In International
Conference on Machine Learning.

Aleksandar Petrov, Philip HS Torr, and Adel Bibi. 2024. Prompting a pretrained transformer can be a
universal approximator. In International Conference on Machine Learning.

Franco Scarselli and Ah Chung Tsoi. 1998. Universal approximation using feedforward neural
networks: A survey of some existing methods, and some new results. Neural networks, 11(1):15–
37.

Anton Maximilian Schäfer and Hans Georg Zimmermann. 2006. Recurrent neural networks are uni-
versal approximators. In Artificial Neural Networks–ICANN 2006: 16th International Conference,
Athens, Greece, September 10-14, 2006. Proceedings, Part I 16, pages 632–640. Springer.

Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. arXiv preprint
arXiv:1911.02150.

Matus Telgarsky. 2015. Representation benefits of deep feedforward networks. arXiv preprint
arXiv:1509.08101.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
Processing Systems.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau,
Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt,
Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric
Jones, Robert Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas,
Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris,
Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0
Contributors. 2020. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature
Methods, 17:261–272.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev,
Andrey Zhmoginov, and Max Vladymyrov. 2023a. Transformers learn in-context by gradient
descent. In International Conference on Machine Learning.

Johannes von Oswald, Eyvind Niklasson, Maximilian Schlegel, Seijin Kobayashi, Nicolas Zuc-
chet, Nino Scherrer, Nolan Miller, Mark Sandler, Max Vladymyrov, Razvan Pascanu, and João
Sacramento. 2023b. Uncovering mesa-optimization algorithms in transformers. arXiv preprint
arXiv:2309.05858.

Shida Wang and Beichen Xue. 2023. State-space models with layer-wise nonlinearity are universal
approximators with exponential decaying memory. In Advances in Neural Information Processing
Systems.

Yihan Wang, Jatin Chauhan, Wei Wang, and Cho-Jui Hsieh. 2023. Universality and limitations of
prompt tuning. Advances in Neural Information Processing Systems.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du,
Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. In
International Conference on Learning Representations.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2021. Thinking like transformers. In International
Conference on Machine Learning.

13


---Page Break---
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2022. An explanation of
in-context learning as implicit Bayesian inference. In International Conference on Learning
Representations.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. 2019.

Are transformers universal approximators of sequence-to-sequence functions? In International
Conference on Learning Representations.

Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. 2023. Trained transformers learn linear models
in-context. arXiv preprint arXiv:2306.09927.

Liu Ziyin, Tilman Hartwig, and Masahito Ueda. 2020. Neural networks fail to learn periodic functions
and how to fix it. In Advances in Neural Information Processing Systems.

14


---Page Break---
A
Computation Graph Debranching Rules

We convert the computation DAG resulting from the LSRL program into a path program by attending
to the first node whose output is the input for multiple other nodes, i.e., the first branching node.

Preparation step.
Before we even start debranching we first pre-process the graph by fusing
consecutive nodes of the same type together. The specific rules are:

• If a Lin node is followed by a single other Lin node, then fuse them together. This follows
directly from the classical result that composing linear functions is a linear function.

• If a ReLU node is followed by another ReLU node, we can drop one of them as ReLU is
idempotent.

• If a Lin is followed by a LinState, we can subsume the weight matrix A of the linear
node in the B matrix of the LinState, and the bias b of the Lin node in the bias b of the
LinState.

• If all inputs of a Concat node are the same, then this node only duplicates the input and
hence can be safely replaced with a Lin layer.

The debranching process goes through the following cases in order. And iterates until there are no
branching nodes left, in other words, until the graph has become a path graph. We will refer to the
nodes whose input is the branching node as subsequent nodes.

Case 1A: If all subsequent nodes are Multi.
As all Multi nodes that have the same input (the
branching node) they must all be producing the exact same output. Hence, only one can be kept. This
removes one branch.

Case 1B: If subsequent nodes are a combination of Multi and other nodes.
We add a single Lin
layer that acts as a bypass for the non-Multi nodes using the fact that multiplicatin by 1 is identity.
This is followed by a single Multi layer. We then add Slice operators between the new Lin layer
and the non-Multi nodes. This keeps the number of branches unchanged but removes the Multi
node and the new branch can be handled by the other rules.

Case 2: All subsequent nodes are LinState.
LinState nodes can be fused into a single LinState
node by combining their states and update matrices. As each LinState may have different subsequent
nodes itself, we add Slice nodes to extract the respective subspaces of the state. This keeps the
number of branches unchanged but puts the graph into Case 5A.

Case 3: All subsequent nodes are ReLU.
We can replace them by a single ReLU node. This removes
one branch.

Case 4: All subsequent nodes are Concat.
One complication is that Concat nodes can depend
on other Concat nodes. So, we will restrict ourselves at this step by only treating the Concat nodes
that depend only on the branch node directly by replacing them with a single Lin node. The rest will
be handled by the Lin and Concat case (Case 10) or the only Lin case (Cases 5A and 5B). See the
following example:

BranchNode

Concat

Concat

=⇒

BranchNode

Lin

Concat

Hence, this operation either reduces the number of branches by one or will be followed by a case that
reduces the number of branches.

15


---Page Break---
Case 5A: Only Lin nodes and they are all Slices.
This is one of the more challenging cases.
While the Slice nodes are simply Lin nodes with special structure, we cannot treat them like standard
Lin nodes (see Case 5B). While we can merge them into a single Lin node, we will then need further
Slices to extract the relevant subspaces for the subsequent nodes. Therefore, we would be simply
replacing Slice nodes with Slice nodes. Instead, we use the observation that Slice nodes can be
fused with subsequent Lin and LinState nodes and can be pushed after ReLU and Concat nodes.
Therefore we treat each subsequent node differently, depending on its type:

• If there are Multi nodes after any of the Slice nodes, they can all be fused into a single
Lin node followed by a single Multi node.

• If there are Lin or LinState nodes after any of the Slice nodes, the Slices can be fused
with the A matrix of the Lin nodes and the B matrix of the LinState nodes. This uses the
fact that composing linear functions results in a linear function.

• If there is a ReLU after a Slice node, their position can be switched without changing the
nodes. That is because ReLU commutes with linear operations with b = 0 and A with
non-negative eigenvalues as is the case for Slice nodes.

• If there is a Concat node after a Slice node, we can similarly push the Slice as a new Lin
node after the Concat.

This step does not reduce the number of branching nodes but prepares the graph for a removal, with
the specific case depending on the remaining nodes.

Case 5B: Only Lin nodes and they are not all Slices.
We can combine them into a single Lin
node and then add Slices to extract the relevant subspaces for the subsequent nodes. These Slices
can then be pushed into the next operations using Case 5A.

Case 6: Both LinState nodes and other nodes.
If both LinState nodes and other nodes are
present, we can pass through the other variables with dummy LinState variables using zero matrices
for A and identities for B. Then, Case 2 can be used to fuse all the LinState variables together.

Case 7A: Only Lin and ReLU nodes where all Lin nodes are followed by only one node which is
a ReLU.
If we add Lin bypasses to the ReLUs we will have only Lin nodes left. Each one of them
would be followed by a ReLU. Hence, Case 5B can be first applied, followed by Case 3.

Case 7B: Only Lin and ReLU nodes where some Lin nodes are not followed by only one node
which is a ReLU.
In this case we cannot apply the above strategy. Instead, we fuse the ReLUs by
placing ReLU-based bypasses before the Lin nodes. We do this in a similar spirit to Eq. (11), by
splitting the positive and negative components and treating them separately. See App. F.6 for the
LSRL implementation. Our DAG will then be in Case 7A first, then Case 5B, and, finally, in Case 3.

Case 8: Only Lin and Concat nodes.
We add Lin bypasses for the Concat nodes which can then
be merged using Case 5B and then Case 5A.

Case 9: Only ReLU and Concat nodes.
Same strategy as for Case 8 but with ReLU bypasses.

Case 10: Only Lin, ReLU or Concat nodes.
We introduce ReLU bypasses to all Concat nodes and
to the Lin branches which are not immediately followed by a ReLU. This will be followed by applying
Case 5B and then Case 3.

The above 13 cases cover all possible branching configurations. After repeated application, they
reduce any DAG corresponding to an LSRL program to a path graph that can be compiled to one of
the recurrent models in Sec. 2.

B
Error Bound on the Approximation Scheme for Continuous Functions

In Sec. 4.1 we outlined a strategy to perform universal in-context approximation for continuous
functions with Linear RNNs. The full program is in Lst. 1 and an illustration of the scheme is

16


---Page Break---
presented in Fig. 2. In Sec. 4.1 we claimed that the prompt length required to approximate an
L-Lipschitz function f (w.r.t. the ℓ2 norm) to precision ϵ is N = (2ϵ/L√din)-din = O(ϵ-din). The
present appendix offers the formal proof of this claim.

The program in Lst. 1 approximates the value of a function y = f(q) with the value ¯y at the centre
c of the cell that contains q. Therefore, the error of our approximation is the maximum difference
between f(q) and f(c): ∥f(q) −f(c)∥2. First, as the length of each side of the cell is δ, that means
that ∥q −c∥∞≤δ/2. Thus, ∥q −c∥2 ≤
√dinδ/2. Therefore, thanks to f being L-Lipschitz we get:

∥f(q) −f(c)∥2 ≤δL√din

2
.

If we want to upper bound this approximation error by ϵ, we need to have δ small enough:

δ ≤
2ϵ
L√din
.

Finally, as the number of cells we need to cover the whole domain is N = (1/δ)din, this corresponds
to us needing sufficiently long prompt:

N ≥
1

δ

din
≥
L√din

2ϵ

din
.

Therefore, if we want our approximation to have error at most ϵ anywhere in the domain, we need a
prompt of length at least (L√din/2ϵ)din.

C
Gated RNNs are GRU models

A GRU layer (Cho et al., 2014) with input at ∈Rdin and hidden state ht−1 ∈Rdhidden, and output
ht ∈Rdhidden can be described as follows:
zt = Sigmoid(Wzat + Uzht−1 + bz),
(update gate vector)
(12)
rt = Sigmoid(Wrat + Urht−1 + br),
(reset gate vector)
(13)
ˆht = tanh(What + Uh(rt ⊙ht−1) + bh),
(candidate activation vector)
(14)

ht = (1 −zt) ⊙ht−1 + zt ⊙ˆht,
(output vector)
(15)

In this section, we show a conversion of a single Gated RNN layer (Eq. (5)) to k + 2 GRU layers.
Here, k is the number of layers in the γ and h MLPs in Eq. (5). We first show that a single GRU layer
can be used to compute the updated state st and the output of the first layer of γ when applied to
xt. Then, every pair of single layers of γ(xt) and ϕ(st) can be represented as an individual GRU
layer. Finally, a single layer can be used to compute the element-wise multiplication γ(xt) ⊙ϕ(st).
For simplicity, we assume the Sigmoid and tanh nonlinearities are replaced by ReLUs. If not, they
can each be approximated with MLPs and hence also with additional GRU layers. Additionally, for
convenience we will assume din = dhidden.

C.1
Representing the state update as a GRU layer

For this layer we set bz = 1, Wz = 0 , Uz = 0 giving zt = 1. Similarly, we set br = 1, Wr = 0 ,
Ur = 0 giving rt = 1. Thus, Eq. (14) reduces to:
ˆht = σ(What + Uhht−1 + bh),
(16)

Setting at =

0
xt


, where xt ∈Rdin/2, ht−1 =

st−1
0


, where st−1 ∈Rdhidden/2, Wh =

0
B
0
I


,

Uh =

A
0
0
0


, bh =

b
−klb


, where klb is a vector where every element in k is a lower bound on

xt. results in Eq. (15) becoming:

ht = σ

0
B
0
I

 
0
xt


+

A
0
0
0

 
st−1
0


+

b
−klb


=

σ(Ast−1 + Bxt + b)
σ(xt −klb)


=

σ(st)
xt −klb


.

(17)
Note: if we do not want to assume a compact domain for xt, it would be possible to use the same
trick as in Equation (11) rather than subtracting k in this layer and adding in the next. However, we
omit this approach for clarity of presentation.

17


---Page Break---
C.2
Representing each MLP layer as a GRU layer

In these layers, similarly to the recurrent layer, we set bz = 1, Wz = 0 , Uz = 0 giving zt = 1.
In the same way, we set br = 1, Wr = 0 , Ur = 0 giving rt = 1. Here, however, we set

Wh =

Whi
0
0
Wγi


, Uh = 0 and bh =

bhi
bγi


, except for the first of such layer where bh =

bhi
bγi + Wγiklb


. Thus, for an input at =

a1,t
a2,t


the layer output (Eq. (15)) for layer i is:

ht = σ

Wϕi
0
0
Wγi

 
a1,t
a2,t


+

bϕi
bγi


=

ϕi(a1,t)
γi(a1,t)


.
(18)

Here, ϕi and γi are the i-th layers (including the ReLU) of respectively ϕ and γ in Eq. (5).

C.3
Representing the multiplicative gating with a single GRU layer

The only thing left is to model the element-wise multiplication of the outputs of ϕ and γ in Eq. (5). We

do this using a GRU layer with bz = 0, Wz = 0, Uz =

0
0
0
I


. We set br = 0, Wr = 0 , Ur = 0

giving rt = 0. We also set bh = 0, Wh =

0
0
I
0


, Uh = 0. Thus, for an input at =

a1,t
a2,t


, the

output ht (Eq. (15)) of this GRU layer becomes:

ht = σ

0
0
I
0

 
a1,t
a2,t


⊙

0
0
0
I

 
a1,t
a2,t


=

0
σ(a1,t) ⊙a2,t


.
(19)

If at is the output of a GRU layer constructed as in Eq. (18) (as is in our case), then it must be
non-negative. This is due to the ReLU application in Eq. (18). Hence, the application of another ReLU
to a1,t in Eq. (19) can be safely removed as ReLU is idempotent and Eq. (19) simplifies to

ht =

0
a1,t ⊙a2,t


.
(20)

Thus, this construction computes element-wise multiplication of a1,t and a2,t.

C.4
Composing the operations to model a single Gated RNN layer

In order to represent Eq. (5), we use one GRU layer for the recurrence (as described in App. C.1),
followed by k GRU layers modelling a pair of the k MLP layers of ϕ and γ (App. C.2), completed
with a single mixing layer (App. C.3). This stack of k + 2 layers models exactly the Gated RNN
layer (Eq. (5)):

st = σ

A

0
st−1


+ B

xt
0


+ b


yt =

0
γ(xt) ⊙ϕ(st)


,

With this, we have shown that any Gated RNN (Eq. (5)) can be expressed as a GRU-based model.
Hence, the two universal approximation programs in Lsts. 1 and 2 can be implemented also in
GRU-based models. Thus, the GRU architecture can also be a universal in-context approximator.

D
Gated RNNs are LSTMs

A single LSTM layer (Hochreiter and Schmidhuber, 1997; Gers et al., 2000) with input at ∈Rdin,
hidden state ht−1 ∈Rdhidden, candidate memory cell ˜ct ∈Rdhidden, memory cell ct ∈Rdhidden and layer

18


---Page Break---
output ht ∈Rdhidden can be expressed as:

ft = Sigmoid(Wfat + Ufht−1 + bf),
(forget gate vector)
(21)
it = Sigmoid(Wiat + Uiht−1 + bi),
(input gate vector)
(22)
ot = Sigmoid(Woat + Uoht−1 + bo),
(output gate vector)
(23)
˜ct = tanh(Wcat + Ucht−1 + bc),
(candidate cell vector)
(24)
ct = ft ⊙ct−1 + it ⊙˜ct,
(memory cell vector)
(25)
ht = ot ⊙tanh(ct),
(output vector)
(26)

where h0 = 0 and c0 = 0 .

In a way analogous to App. C, we show that a single layer of a gated RNN (Eq. (5)) can be expressed
using k + 2 LSTM layers, where k is the maximum depth of either of the MLP networks ϕ or γ. We
again follow the setup of replacing all Sigmoid and tanh activation functions with ReLU activations
which we denote σ and we again assume that din = dhidden. The set up follows the same structure as
in App. C. First, we show that the non-linear state update computing st can be expressed as a single
LSTM layer. We then show that we can represent the layers in MLP networks γ(xt) and ϕ(st) using
single LSTM layers. Finally, a single layer can compute the Hadamard product between γ(xt) and
ϕ(st). Therefore, any Gated RNN with ReLU activations can be expressed as a LSTM with ReLU
activations.

For clarity of the exposition, we once again assume that our inputs belong to a compact domain X
of real vectors. This implies that the set is bounded and, in particular, that we can find a vector klb
such that klb,i ≤(xt)i for i ∈[din] for all xt ∈X. In other words, we have (xt −klb)i ≥0 for for
i ∈1, . . . , din. We will make use of this fact several times when dealing with ReLU activations.

D.1
Representing the state update as an LSTM layer

We first represent the non-linear state update in Eq. (5) using a single layer of an LSTM. In particular,
we set Wf = 0, Uf = 0 and bf = 0 so that ft = 0. We also set Wi = 0, Ui = 0, bi = 1 and
Wc = 0, Uc = 0, bc = 1. This results in it = 1 and ˜ct = 1. We see from this that the LSTM layer
with these weight settings reduces to

ht = ot = σ(Woat + Uoht−1 + bo).
(27)

We now set at =

0
xt


, where xt ∈Rdin/2, ht−1 =

st−1
0


, where st−1 ∈Rdhidden/2, Wo =

0
B
0
I


, Uo =

A
0
0
0


, bo =

b
−klb


so that

ht = σ

0
B
0
I

 
0
xt


+

A
0
0
0

 
st−1
0


+

b
−klb


=

σ(Ast−1 + Bxt + b)
σ(xt −klb)


=

st
xt −klb


.

(28)

D.2
Representing each MLP layer as an LSTM layer

Now we want to use an LSTM layers to model the MLP layers of both γ and ϕ simultaneously. We
set Wf = 0, Uf = 0, bf = 0 and Wi = 0, Ui = 0, bi = 1 and Wc = 0, Uc = 0, bc = 1 as

before. We make a change for these LSTM layers by setting Wo =

Wϕi
0
0
Wγi


, Uo = 0 and

bo =

bϕi
bγi


, except for the first layer where bϕ =

bϕ1
bγ1 + Wγ1k


. Thus, for an input at =

a1,t
a2,t



the layer output is:

ht = σ

Wϕi
0
0
Wγi

 
a1,t
a2,t


+

bϕi
bγi


=

ϕi(a1,t)
γi(a2,t)


.
(29)

Here, ϕi and γi again refer to the i-th layers (including the ReLU) of respectively ϕ and γ in Eq. (5).

Note that, without a loss of generality, if we have that ϕ has m layers whereas γ has k with m < k,
then we can also model this by simply adding additional layers to model additional layers for γ whilst

19


---Page Break---
simply passing on ϕ unchanged. Specifically, we set set the weights to ensure that ft = 0 and that it

and ˜ct are 1 so that ht = ot. The input to this layer for i > k is then given as at =

ϕ(st)
a2,t


. The we

set the weights to compute ot as

ot = σ

I
0
0
Wγi

 
ϕ(st)
a2,t


+

0
bϕi


=

ϕ(st)
γi(a2,t)


.
(30)

D.3
Representing the multiplicative gating with an LSTM layer

Finally, we model the element-wise multiplication of the outputs of ϕ and γ in Eq. (5). To do this we
set the weights of the input gate and candidate cell vectors for the final layers of of γ and ϕ to be as
follows:

it = σ

0
0
I
0

 
a1,t
a2,t


+

0
0


=

0
a1,t


(31)

and

˜c = σ

0
0
0
I

 
a1,t
a2,t


+

0
0


=

0
a2,t


.
(32)

Then by setting Wf = 0, Uf = 0, bf = 0 and Wo = 0, Uo = 0, bo = 1 to force ft = 0 and
ot = 1, we get

yt = σ(ct) =

σ(0 ⊙0)
σ(a1,t ⊙a2,t)


=

0
σ(a1,t ⊙a2,t)


.
(33)

D.4
Composing the operations to model a single Gated RNN layer

To model the gated RNN described in Eq. (5), we again follow the same lines as described in App. C.
In particular, we use one LSTM layer for the recurrent state updated as described in App. D.1.
We then stack k LSTM layers as described in App. D.2 to model the k MLP layers of ϕ and γ in
parallel. We then use one final layer to both give the final MLP layer of ϕ and γ and to compute their
Hadamard product as set out in App. D.3 in order to match the output of the gated RNN in Eq. (5).
Now, since we are working with σ = ReLU, both γ(xt) and ϕ(st) are positive and therefore so is
their product. Hence, applying σ to the product components in Eq. (33) leaves the the components
invariant. Therefore, we output is

yt =

0
γ(xt) ⊙ϕ(st)


,
(34)

as required.

Hence, we have shown that a single layer of a gated RNN as described by Eq. (5) can be represented
using k + 2 LSTM layers where k is the maximum depth of ϕ and γ. Therefore, once again, the
two universal approximation programs in Lsts. 1 and 2 can also be implemented for LSTMs. Hence,
LSTM models are also universal approximators in the sense described in Sec. 4.

E
Gated Linear RNNs are Hawk/Griffin Models

A single residual block of a Hawk/Griffin model (De et al., 2024) consists of two components, a
recurrent block for temporal mixing which makes use of a one-dimensional temporal convolution, as
well as real-gated linear recurrent unit (RG-LRU) and a gated MLP block. Specifically, we consider
an input at ∈Rdin, inputs to the blocks of dimensions din and outputs from each block of dimensions
din. Within blocks, all vectors have dimensionality dhidden = Edin, where E is denotes an expansion
factor. Below, we formally describe the form of the recurrent and gated MLP blocks which are the
two main components making up the residual blocks used for Hawk and Griffin.

Recurrent block. The recurrent block consists of two branches. The first applies a one-dimensional
temporal convolution followed by a RG-LRU. The second branch simply performs a linear transfor-
mation followed by a non-linearity, i.e. applies a single layer of an MLP.

20


---Page Break---
Consider the first branch of the recurrent block with an input at. The one-dimensional temporal
convolution can be written as:
a′
t = Waat,
(35)
gt = GeLU(Wgat + bg),
(36)

Mt =
a′
t−(dconv−1), . . . , a′
t−2, a′
t−1, a′
t

,
(37)

zt =

dconv-1
X

i=0
WM[i]Mt[t −i] + bconv
(convolution with window size dconv),
(38)

where bconv is a bias vector and WM =
 ˜
B, ˜
A ˜
B, ˜
A2 ˜
B, · · · , ˜
At ˜
B, · · ·

is the convolutional kernel
for the one-dimensional temporal convolution.

The output of this convolution is then fed into a RG-LRU. We can write this down concretely using
as an input zt from the one-dimensional convolution and with recurrent state ht ∈Rdmodel:
rt = Sigmoid(Wrzt + br),
(39)
it = Sigmoid(Wizt + bi),
(40)
a = Sigmoid(Λ),
(Λ a learnable parameter)
(41)
at = acrt,
(c = 8 fixed scalar constant)
(42)

ht = at ⊙ht−1 +
q

1 −a2
t ⊙(it ⊙zt).
(43)

Now consider the second branch of the recurrent block. This performs a linear transformation
followed by a non-linear activation:
gt = GeLU(Wgat + bg).
(44)

To get the final output of the recurrent block, we multiply the components of the vectors computed
from each branch within the recurrent block and then perform a non-linear transformation:
h′
t = gt ⊙ht,
(45)

ot = Woh′
t + bo.
(46)

Gated MLP block. After passing through the recurrent block, we pass the output ot into a gated
MLP block. Again we have two branches, the first where we linearly transform the input to this block
et = Weot + be,
(47)
and the second performs a single layer MLP transformation as
ft = GeLU(Wfot + bf).
(48)
These are then combined through a Hadamard product and linear transformation as
e′
t = et ⊙ft,
(49)

mt = Wme′
t + bm.
(50)
We then have that the vector mt acts as the output of the residual block given the input at.

Distinction between the Griffin and Hawk models. Hawk is the more simple of the two architectures
proposed in (De et al., 2024). Here, residual blocks using the recurrent block described above are
simply stacked on top of each other to form the Hawk architecture. Griffin, on the other hand, mixes
recurrent blocks and local attention. In particular, two residual blocks with recurrent blocks are
followed by one residual block using local MQA attention (Beltagy et al., 2020; Shazeer, 2019).

Simplifying Assumptions. We again follow the setup of replacing all Sigmoid and tanh activation
functions with ReLU activations which we denote σ. Furthermore, we assume for simplicity that
din = dhidden by choosing E = 1. Moreover, the Hawk and Griffin architecture contains residual
connections and normalising layers which we omit.4 We again assume compactness of the input
domain X and denote a vector of finite values klb, such that klb,i ≤(xt)i for i ∈[din] and all
xt ∈X, just as before. Finally, we assume that dconv = T where T is the maximum sequence length.

4We will force a lot of our recurrent blocks to implement the identity function. So instead of this, we could
implement the 0 function in the recurrent block and use a residual connection between the residual block input
and the output of the recurrent block to achieve the same identity function. However, for clarity we ignore
residual connections in our derivations.

21


---Page Break---
E.1
Representing the state update using a recurrent block

Starting with the input to the Hawk model, which we denote at, we define this to be a function of the

input to the Gated RNN xt as at =

0
xt


. First, we set Wa = I so that a′
t = at. Next we choose

matrices ˜
A =

0
A
0
0


and ˜
B =

0
B
0
0


which we then use, with a convolutional window size

of dconv = T to form the convolutional kernel WM =
 ˜
B, ˜
A ˜
B, ˜
A2 ˜
B, · · · , ˜
At ˜
B, · · ·

. Setting the

convolutional bias as bconv =

0
1


gives

zt =

t−1
X

i=0
WM[i]Mt[t −i] + bconv,
(51)

= ˜
Bat + ˜
A ˜
Bat−1 + · · · + ˜
At−1 ˜
Ba1 +

0
1


(52)

=

st
1


.
(53)

Now, we pass zt through the RG-LRU. We set Λ = 0 so that at = 0. We also define Wi = 0 and
bi = 1 so that it = 1. This gives us ht = zt, so that we pass the output of the one-dimensional
convolution through he RG-LRU.

Next, let’s focus on the second branch. Making use of the lower bound klb on the domain X, we set

Wg = I and bg =

1
−klb


so that

gt = σ

I

0
xt


+

1
−klb


=

σ(1)
σ(xt −klb)


=

1
xt −klb


,
(54)

where we used that (xt −klb)i ≥0 for every i. Combining the two branches gives

h′
t =

1
xt −klb


⊙

st
1


=

st
xt −klb


.
(55)

We finally get the output of the recurrent block by defining Wo = I and b0 =

0
klb


so that

ot =

st
xt


.
(56)

E.2
Representing the identity function using a recurrent block

We now show that we can pass an input unchanged through a recurrent block. Assume that the input to

the recurrent block is at =

a1,t
a2,t


with Wa = I so that a′
t = at. Then we define matrices ˜
A = 0 and

˜
B = I which we then use to form the convolutional kernel WM =
 ˜
B, ˜
A ˜
B, ˜
A2 ˜
B, · · · , ˜
At ˜
B, · · ·

.
Finally, setting the convolutional bias as bconv = 0 results in zt = at. From here, we can again set
Λ = 0, Wi = 0 and bi = 1 so that ht = zt. Looking at the second branch and setting Wg = 0 and
bg = 1 so that h′
t = ht. Finally, we can simply output the input to the recurrent block by setting
Wo = I and bo = 0 so that ot = ht which means that ot = at.

E.3
Representing each MLP layer as a gated MLP block

We can represent the MLP layers of the networks ϕ(st) and γ(xt) as described in Eq. (4) using Gated
MLP blocks. We again denote the i-th layer of ϕ and γ as ϕi and γi. Assume that the input to the

gated MLP block is at =

a1,t
a2,t


. Then, on the first purely linear branch, let us define We = I and

22


---Page Break---
be = 1 so that et = 1. On the second non-linear branch, we can define Wf =

Wϕi
0
0
Wγi


and

bf =

bϕi
bγi


. This results in

ft = σ

Wϕi
0
0
Wγi

 
a1,t
a2,t


+

bϕi
bγi


=

ϕi(a1,t)
γi(a2,t)


.
(57)

Due to our setting of et, we get e′
t = ft. Further, defining Wm = I and bm = 0 makes the output
of the MLP block be

mt =

ϕi(a1,t)
γi(a2,t)


.
(58)

Emulating the layers of only one the two networks. Suppose without loss of generality (WLOG)
that ϕ has m layers and γ has n layers where m < n. Suppose also that our input to the MLP block

is at =

ϕ(xt)
a2,t


. Again, on the first purely linear branch, let us define We = I and be = 1 so that

et = 1. Now we modify the weights on the second non-linear branch by defining Wf =

I
0
0
Wγi



and bf =

0
bγi


. This gives us

ft = σ

I
0
0
Wγi

 
ϕ(xt)
a2,t


+

0
bγi


=

σ(ϕ(xt))
γi(a2,t)


=

ϕ(xt)
γi(a2,t)


,
(59)

where we have used that since ϕ(xt) is a ReLU network whose final activation is a ReLU, we have
that ϕ(xt) = σ(ϕ(xt)). Hence, if our networks have different depths and we have fully emulated
one of the networks, we can continue to emulate the remaining layers of the other network while
keeping the fully emulated network fixed and unchanged.

E.4
Representing the identify function using a gated MLP block

In this section we show that we can represent an identity function using a gated MLP block. This
can be simply done by setting Wf = 0, bf = 1, We = I, be = 0, Wm = I and bm = 0. This then

gives us that for an input at =

a1,t
a2,t


to the gated MLP block, the output of the gated MLP block is

mt = at. Thus, we pass the input through the gated MLP unchanged.

E.5
Representing multiplicative gating with a gated MLP block

The final thing we need to do is to compute an element-wise product of two vectors in order to match
the output in Eq. (4). In other words, to match the ϕ(xt) ⊙γ(st) operation.

Again, assume that the input to the gated MLP block is at =

a1,t
a2,t


. Working with the first linear

branch, we define We =

0
0
I
0


and be = 0, so that

et =

0
0
I
0

 
a1,t
a2,t


+ 0 =

0
a1,t


.
(60)

Next, we define Wf = I and be = 0 so that

ft =

σ(a1,t)
σ(a2,t)


.
(61)

Setting Wm = I and bm = 0 gives the output of the gated MLP as

mt =

0
a1,t ⊙σ(a2,t)


.
(62)

23


---Page Break---
E.6
Composing the operations to model a single gated linear-RNN layer

Now that we have all the individual layers, we can combine them so that we can use a Hawk model
to emulate a single Gated RNN layer.

First we start by taking the input of the form at =

0
xt


. We use a residual block that consists of

a recurrent block computing the state update as descried in App. E.1 and then a gated MLP block
that computes the identity function as demonstrated in App. E.4. This gives an output from this first

recurrent block as ot =

st
xt


.

Next, we emulate the MLP layers of the networks ϕ and γ in parallel. Suppose WLOG that ϕ and γ
have m and n MLP layers respectively, where m ≤n. We stack m residual blocks using recurrent
blocks that implement the identity function as described in App. E.2 followed by MLP blocks that
apply the MLP layers of ϕ and γ as described in App. E.3. Stacking m such residual blocks results in

the output mt =

γm(st)
ϕ(xt)


, where we can fully emulate the shallower network ϕ(xt).

Now, for the remaining k −m layers for the network γ(xt), we stack residual blocks with recurrent
blocks implementing the identity function as described in App. E.2 and MLP blocks that leave ϕ(xt)
unchanged whilst applying the additional layers needed to emulate γ(st) as described at the end
of App. E.3. After stacking k −m additional residual layers in this fashion, the output of the final

residual block will now be mt =

γ(st)
ϕ(xt)


, which fully reconstructs the MLP networks γ and ϕ.

Finally, we utilise a residual block with a recurrent block that implements the identity function as
described in App. E.2 followed by a gated MLP block that applies multiplicative gating as described

in App. E.5. This then gives as an output of this final residual block mt =

0
γ(st) ⊙σ(ϕ(xt))


.

Since ϕ(xt) is a MLP network with the final activation function being a ReLU activation, we have
that σ(ϕ(xt)) = ϕ(xt), giving the required final output from the stacked block of residual blocks as

mt =

0
γ(st) ⊙ϕ(xt)


.
(63)

Hence, we have shown that a single layer of a gated RNN as described by Eq. (5) can be represented
using k + 2 Hawk residual blocks where k is the maximum depth of ϕ and γ. Once again, the two
universal approximation programs in Lsts. 1 and 2 can also be applied to Hawk models as they can
represent Gated Linear RNNs. Therefore, Hawk models are also universal approximators in the sense
described in Sec. 4.

Gated Linear-RNNs are Griffin models too. The above argument extends to the Griffin architecture
which uses stacks of two residual blocks with recurrent blocks followed by a residual block with
attention. The only thing that changes is that for every third residual block, which in our argument
will be used to compute the MLP layers of ϕ and γ in parallel, the recurrent block is now replaced
with a local MQA block.

We can set the key query and values matrices to implement the identity function which is to act
input to the block. Hence, as a corollary of the above argument, we can also show that the universal
approximation programs in Lsts. 1 and 2 can also be implemented as Griffin models. Therefore,
Griffin models can also be universal approximators in the sense described in Sec. 4.

F
Definitions for some helper functions in LSRL

F.1
f_not

This is a convenience function that creates a NOT function block. It assumes that x is 0 or 1. Works
with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.

1
not_x = 1 - x

24


---Page Break---
F.2
f_and

This is a convenience function that creates an AND function block. It assumes that x and y are 0 or 1.
Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise. mu is the
approximation parameter µ for f_step as described in Sec. 3.

1
and_x_y = ReLU(f_step(x, mu) + f_step(y, mu) - 1)

F.3
f_or

This is a convenience function that creates an OR function block. It assumes that x and y are 0 or 1.
Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise. mu is the
approximation parameter µ for f_step as described in Sec. 3.

1
or_x_y = f_step(x + y, mu=mu)

F.4
f_smaller

This is a convenience function that a less than comparison block. Works with scalar and vector-valued
inputs. With vector-valued inputs, it acts element-wise. mu is the approximation parameter µ for
f_step as described in Sec. 3.

1
smaller_x_y = f_step(y - x, mu=mu)

F.5
f_larger

This is a convenience function that a more than comparison block. Works with scalar and vector-
valued inputs. With vector-valued inputs, it acts element-wise. mu is the approximation parameter µ
for f_step as described in Sec. 3.

1
larger_x_y = f_step(x - y, mu=mu)

F.6
f_relu_identity

Identity operation using ReLUs. This is useful for debranching when some of the branches have ReLUs
but the other don’t. We can add this as a bypass for the ones that do not and can then merge the ReLUs
together (see App. A for details).

1
positive_part = ReLU(x)
2
negative_part = ReLU(
3
Linear(
4
input=x,
5
A=-1 * eye(x.dim),
6
b=zeros(x.dim , 1),
7
)
8
)
9
both = Concat ([ positive_part , negative_part ])
10
relu_identity = Linear(
11
input=both ,
12
A=hstack(eye(x.dim), -1 * eye(x.dim)),
13
b=zeros(x.dim , 1),
14
)

F.7
f_modulo_counter

Computes the x mod divisor where x is a counter starting from zero. The idea is that we rotate
a unit vector so that it makes a full revolution every divisor rotations. dummy_input can be any
variable, we use it only to construct a constant.

1
angle = 2 * pi / divisor
2
R = [[cos(angle), sin(angle)], [sin(angle), cos(angle)]]

3
unit_vector = [[1], [0]]

4
# we first rotate , then output so if we want the first output to be 0 we need to have the init_state one step

before that
5
init_state = R.inv() @ unit_vector

25


---Page Break---
6
# this rotates a 2D vector 1/ divisor revolutions at a time
7
cycler = LinState(
8
input=dummy_input ,
9
A=R,
10
B=zeros(2, dummy_input.dim),
11
init_state=init_state ,
12
)

13
# we now need to extract the position of the cycler
14
extractor_matrix = vstack (*[(R^i * unit_vector).T) for i in range(divisor)])
15
indicator = Linear(
16
input=cycler ,
17
A=extractor_matrix ,
18
b=zeros(divisor , 1)
19
)

20
# the dot product with the row of extractor_matrix corresponding to the current position of the cycler is 1
21
# the dot product with the second highest is cos(angle)
22
# thus , we can threshold at 1-cos(angle /2) to get a one hot encoding of the current position of the cycler
23
one_hot = f_larger(indicator , cos(angle / 2))

24
# and to get an integer value we need one final linear layer
25
mod_value = Linear(
26
one_hot ,
27
A=[[i for i in range(divisor)]],
28
b=zeros(1, 1)
29
)

26


---Page Break---
NeurIPS Paper Checklist

i. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract and the introduction clearly state all the contributions of the
paper and clearly differentiate the theoretical results which hold in general and the empirical
phenomena that we observe, which may not generalize to all settings.
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

ii. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The paper discusses the limitations of the present work. There is a dedi-
cated Limitations section in Sec. 7 that addresses the fact that we only provide constructive
existence results but not necessary and sufficient conditions for universal in-context approxi-
mation to arise. We also highlight that our results might not hold to models with structural
constraints on their parameters. Moreover, we have a dedicated section (Sec. 5) which
addresses some of the limitations of constructing universal in-context approximators with
fully recurrent architectures in practice. This section proposes solutions and demonstrates
that they result in more numerically stable models which are more likely to occur in practice.
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

27


---Page Break---
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

iii. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The paper has two main theoretical results: the constructions of universal in-
context approximators for continuous and for discrete functions. Both results are presented
as LSRL programs which compile to the architectures considered in this work. Furthermore,
these programs have been implemented in Python, their correctness has been tested and they
are available in the supplementary materials.
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

iv. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: There are two experimental aspects to this work. First, there is the implemen-
tation of LSRL and the two universal approximation programs in Lsts. 1 and 2. The most
critical aspect of implementing LSRL is the debranching algorithm which is described in
detail in App. A. Additionally, the two programs are described in full in their correspond-
ing listings. We also provide Python implementation for the LSRL compiler and the two
programs.
Second, there is the study of how affected by parameter noise are the different implementa-
tions of the conditional assignment operator f_ifelse which was presented in Sec. 6. The
details of this experiment are described in Fig. 4 and we also provide the code with which
we did the experiment and our plots.
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

28


---Page Break---
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

v. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We are providing code that includes an implementation of LSRL, the two
universal in-context approximation programs in Lsts. 1 and 2 and everything needed to
reproduce the experiments in this work.
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

vi. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [NA]
Justification: The experiments in our work are based on constructed models rather than
trained models. Therefore, considerations such as dataset, optimizers and hyperparameters
do not apply.
Guidelines:

• The answer NA means that the paper does not include experiments.

29


---Page Break---
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
vii. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: For this work, uncertainty quantification could only make sense in the context
of Fig. 4. However, the behaviour we observe, especially for the continuous case, is bimodal.
As bimodal distributions cannot be properly captured with error bars we decided against
using them. Furthermore, we are studying whether a phenomenon occurs, rather than
quantifying it. Therefore, we decided to instead use a strip plot instead as it explicitly shows
all our results unabridged, clearly indicates the bimodal nature of the results, and distinctly
showcases the noise robustness trends of the different approaches we consider.
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
viii. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [No]
Justification: Our experiments were ran on a single machine and using only CPU com-
pute. Therefore, the compute required is negligible for the contemporary machine learning
standards.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
ix. Code Of Ethics

30


---Page Break---
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: This is a theoretical work with no human participants, datasets, or potential
societal impact or harmful consequences. Therefore, the present work has no moral or
ethical relevance or implications.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

x. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: As mentioned above, this is a theoretical work which establishes theoretical
properties of mathematical objects that are already used in practice. However, we do
discuss the implications of our findings, namely that if models are universal in-context
approximators, then it might be difficult to ensure that they are aligned and cannot be
misused. Nevertheless, we only show that this is a property already present in existing
models, and hence our work does not introduce new attack or misuse vectors. On the
contrary, we hope that us highlighting this issues will help the community to develop safer
and more secure generative AI systems.
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

xi. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: We release no data or models.
Guidelines:

31


---Page Break---
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
xii. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [NA]
Justification: The paper does not use existing assets beyond common Python libraries.
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
xiii. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The only new asset arising from this work is the LSRL code base which we
have ensured to be well-documented, accompanied by unit and integration tests, and with
illustrative Jupyter notebooks.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
xiv. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]

32


---Page Break---
Justification: There were no human participants involved in any part of this work.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
xv. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: This is a purely theoretical work and as such no IRB approval or equivalent
was necessary or appropriate.
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

33


---Page Break---
