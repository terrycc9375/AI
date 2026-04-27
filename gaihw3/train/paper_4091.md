An exactly solvable model for emergence and scaling
laws in the multitask sparse parity problem

Yoonsoo Nam*a, Nayara Fonseca*a, Seok Hyeong Leeb, Chris Mingarda c, and Ard A. Louisa

aRudolf Peierls Centre for Theoretical Physics, University of Oxford
bCenter for Quantum Structures in Modules and Spaces, Seoul National University
cPhysical and Theoretical Chemistry Laboratory, University of Oxford

Abstract

Deep learning models can exhibit what appears to be a sudden ability to solve a
new problem as training time, training data, or model size increases, a phenomenon
known as emergence. In this paper, we present a framework where each new ability
(a skill) is represented as a basis function. We solve a simple multi-linear model
in this skill-basis, finding analytic expressions for the emergence of new skills, as
well as for scaling laws of the loss with training time, data size, model size, and
optimal compute. We compare our detailed calculations to direct simulations of a
two-layer neural network trained on multitask sparse parity, where the tasks in the
dataset are distributed according to a power-law. Our simple model captures, using
a single fit parameter, the sigmoidal emergence of multiple new skills as training
time, data size or model size increases in the neural network.

1
Introduction

Emergence in large language models (LLMs) has attracted a lot of recent attention [1–4]. It motivates
the costly drive to train ever larger models on ever larger datasets, in the hope that new skills will
emerge. While the concept of emergence has been critiqued on the grounds that the sharpness of the
transition to acquiring a new skill may be sensitive to the measure being used [5], the observation
that important new skills are learned for larger models raises many challenging questions: when
the skills emerge and what drives the emergence. These questions are complicated by difficulties in
formally defining skills or capabilities [6], and by our general limited understanding of the internal
representations of deep neural networks [7].

Another widely observed property of deep learning models is that the loss improves predictably as a
power-law in the number of data points or the number of model parameters or simply in the amount
of compute thrown at a problem. These neural scaling laws [8, 9] have been widely observed across
different architectures and datasets [10–16]. While the scaling exponents can depend on these factors,
the general phenomena of scaling appear to be remarkably robust. This raises many interesting
questions such as: What causes the near-universal scaling behavior? How does the continuous scaling
of the loss relate to the discontinuous emergence of new skills?

A challenge in answering the questions raised by the phenomena of emergence and scaling laws arises
from the enormous scale and expense of training cutting-edge modern LLMs, which are optimized
for commercial applications, and not for answering scientific questions about how they work. One
way that progress can be made is to study simpler dataset/architecture combinations that are more
tractable. The current paper is inspired in part by recent work in this direction that proposed studying
emergence in learning the sparse parity problem [17, 18], which is easy to define, but known to be

*These authors contributed equally; {yoonsoo.nam,nayara.fonsecadesa}@physics.ox.ac.uk.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
computationally hard. In particular, Michaud et al. [18] introduce the multiple unique sparse parity
problem – where tasks are distributed in the data through a power-law distribution of frequencies – as
a proxy for studying emergence and neural scaling in LLMs. For this data set, the authors empirically
measure the scaling laws of a 2-layer multilayer perceptron (MLP) as a function of training steps (T),
parameters (N), and training samples (D). Based on their quanta model of abrupt skill acquisition,
they schematically derive neural scaling laws as a sum of emergences of new skills. However, no link
was established between the neural network dynamics and the quanta model.

In this paper, we introduce an analytically tractable model by defining a basis of orthogonal functions
for the multitask sparse parity problem. Each basis function corresponds to a skill that can be learned,
and their respective frequencies are distributed following a power-law with exponent α + 1. We
then propose a simple multilinear expansion in these orthogonal functions that introduces a layered
structure reminiscent of neural networks (NNs) and gives rise to the stage-like training dynamics [19].
With our simple model, we can analytically calculate full scaling laws, including pre-factors, as a
function of data exponents α, T, D, N, and optimal compute C. Our simple model can, with just one
parameter calibrated to the emergence of the first skill, predict the ordered emergence of multiple
skills in a 2-layer MLP. We summarize our contributions as follows:

1. Skills as basis functions. We establish a framework for investigating emergence by repre-
senting skills as orthogonal functions that form a basis in function space (Section 2). We
apply our methods to controlled experiments on the multitask sparse parity dataset.
2. Multilinear model. We propose an analytically tractable model that is expanded in the basis
of skill functions, and is multilinear with respect to its parameters so that it possesses a
layerwise structure (Section 3). The multilinear nature of the model produces non-linear
dynamics, and the orthogonal basis decouples the dynamics of each skill.
3. Scaling laws. We derive scaling laws for our multilinear model, including the prefactor
constants, which relate the model’s performance to training time (T), dataset size (D),
number of parameters (N), and optimal compute (C = N × T), see Section 4. We show
that the scaling exponents for these factors are −α/(α + 1), −α/(α + 1), −α, −α/(α + 2),
respectively, where α + 1 is the exponent of the power-law input data.
4. Predicting emergence. We demonstrate that our multilinear model captures the skill emer-
gence of an MLP with 2 layers for varying training time, dataset size, and number of
trainable parameters. Our results show that the multilinear model, calibrated only on the
first skill, can predict the emergence of subsequent skills in the 2-layer MLP, see Fig. 1
and Section 5. We obtain an equivalent result on the time emergence for a transformer
architecture (Fig. 4).

103
104

T

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength  Rk/S

(a) Time emergence

103
104

D

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength  Rk/S

(b) Data emergence

100
101

N

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength  Rk/S

(c) Parameter emergence

k = 1
k = 2
k = 3
k = 4
k = 5
k = 1
k = 2
k = 3
k = 4
k = 5

Figure 1: Predicting emergence. The skill strength Rk, defined as the kth coefficient if a model is
expanded in the basis of the skill functions (gk), measures how well the kth skill is learned, and is
plotted against (a) time T, (b) data set size D, and (c) number of parameters N (width of the hidden
layer). Rk is normalized by the target scale S such that Rk/S = 1 means zero skill loss. The dashed
lines show the abrupt growth – emergence – of 5 skills for a 2-layer MLP (Appendix K) trained on
the multitask sparse parity problem with data power-law exponent α = 0.6 (shaded area indicate
1-standard deviation over at least 10 runs). Solid lines are the predictions (Eqs. (14), (17) and (21),
respectively) from our multilinear model calibrated on the first skill (blue) only.

2


---Page Break---
Table 1: Multitask sparse parity dataset and skill basis functions. The control bits are ns-
dimensional one-hot vectors encoding specific parity tasks, indexed in the first column. The frequency
of the distinct parity tasks follows a rank-frequency distribution with an inverse power law relation
(Eq. (1)). The skill bits are binary strings with m = 3 relevant sparse bits (highlighted in colors)
with their locations varying by skill. The y column shows the target scale S multiplied by the parity
computed from the relevant bit set M(i, x). The last columns show the values of the skill basis
functions gk(i, x), defined in Eq. (2).

Skill idx (I)
Control bits
Skill bits (X)
y
M(i, x) g1(i, x)
g2(i, x)
. . .
gns(i, x)

1
1000000
110110000100
S
[1,1,0]
1
0
. . .
0
1
1000000
100101010001
−S
[0,1,0]
−1
0
. . .
0
...
...
...
...
...
...
...
...
...
2
0100000
001001011011
−S
[0,0,1]
0
−1
. . .
0
...
...
...
...
...
...
...
...
...
ns
0000001
001010100110
−S
[1,1,1]
0
0
. . .
−1

2
Setup

In this section, we define the multitask sparse parity problem under the mean-squared error (MSE)
loss. We represent skills as orthogonal functions and measure their strength in a model by calculating
the linear correlation between the model output and the skill basis functions. For a comprehensive
list of notations, refer to the glossary in Appendix A. Our code is also available online.1

Multitask sparse parity problem.
In the sparse parity problem, nb skill bits are presented to the
model. The target function is a parity function applied to a fixed subset of the input bits. The model
must detect the relevant m < nb sparse bits and return the parity function on this subset (M(i, x),
see Table 1). Michaud et al. [18] introduced the multitask sparse parity problem by introducing ns
unique sparse parity variants – or skills – with different sparse bits (for a representation, see Table 1).
Each skill is represented in the ns control bits as a one-hot string, and the model must solve the
specific sparse parity task indicated by the control bits (for more details, see Appendix B.1).

The ns skills (random variable I ∈{1, 2, . . . , ns}) follow a power law distribution Ps, and the skill
bits (random variable X ∈{0, 1}nb) are uniformly distributed. Because Ps and Pb are independent,
the input distribution P(I, X) follows a product of two distributions:

Ps(I = i) :=
i−(α+1)
Pns
j j−(α+1) ,
Pb(X = x) := 2−nb,
P(I, X) := Ps(I)Pb(X).
(1)

We denote A =
Pns
j=1 j−(α+1)−1
so that Ps(i) := Ai−(α+1).

Skill basis functions.
We represent the kth skill as a function gk : {0, 1}ns+nb →{−1, 0, 1} that
returns the parity ({−1, 1}) on the kth skill’s sparse bits if i = k, but returns 0 if the control bit
mismatches that of the kth skill (i ̸= k):

gk(i, x) :=

(−1)
P

j Mj(i,x)
if i = k
0
otherwise ,
(2)

where M : {0, 1}ns+nb →{0, 1}m is the map that selects the relevant sparse bits for the ith

skill (Table 1) and Mj(i, x) is the jth entry of M(i, x). Note that different skill functions have 0
correlation as the supports of skills functions are mutually exclusive:

gk(i, x)gk′(i, x) = δi,kδk,k′.
(3)

1https://github.com/yoonsoonam119/Skill_Eigenmode.git

3


---Page Break---
The target function.
The target function is a sum over ns skill functions multiplied by a target
scale S:

f ∗(i, x) := S

ns
X

k=1
gk(i, x).
(4)

The target scale S is the norm of the target function (EI,X [f ∗(I, X)f ∗(I, X)] = S2). Note that the
skill functions serve as ‘features’ or countable basis for describing the target function as in Hutter
[20].

MSE loss.
We use MSE loss for analytic tractability:

L := 1

2EX,I
h
(f ∗(I, X) −f(I, X))2i
,
(5)

where f is the function expressed by a given model. We define the skill loss Lk as the loss when only
the kth skill is given, which can be weighted by their skill frequencies to express the total loss:

Lk := 1

2EX
h
(f ∗(I = k, X) −f(I = k, X))2i
,
L =

ns
X

k=1
Ps(I = k)Lk.
(6)

Skill strength.
The skill strength or the linear correlation between the kth skill (gk) and a function
expressed by the model at time T (fT ) is

Rk(T) := EX [gk(I = k, X)fT (I = k, X)] .
(7)

The skill strength Rk is the kth coefficient if a model is expanded in the basis of the skill functions
(gk). The skill strength, like the test loss, can be accurately approximated by a sum (see Appendix K.3).
The skill loss Lk (Eq. (6)) can be expressed by the skill strength and the norm of the learned function
for I = k:

Lk(T) = 1

2
 
S2 + EX

fT (I = k, X)2
−2SRk(fT )

.
(8)

The skill loss becomes 0 if and only if fT (I = k, X) = Sgk(I = k, X).

Experimental setting.
We use a 2-layer MLP that receives the ns + nb bits as inputs and outputs a
scalar ({0, 1}ns+nb →R). In most of the experiments, the NN is trained with stochastic gradient
descent (SGD) with width 1000, using ns = 5, m = 3, and nb = 32, unless otherwise stated. A
decoder transformer is also used for the time emergent experiments. See Appendix K for details.

3
Multilinear model

We propose a simple multilinear model – multilinear with respect to the parameters – with the first N
most frequent skill functions gk(i, x) as the basis functions (features):

fT (i, x; a, b) =

N
X

k=1
ak(T)bk(T)gk(i, x),
(9)

where a, b ∈RN are the parameters. The model has built-in skill functions gk – which transform
control bits and skill bits into the parity outputs of each skill – so the model only needs to scale the
parameters to akbk = S.

The multilinear structure (product of ak, bk) is analogous to the layered structure of NNs and
results in emergent dynamics (Fig. 1(a)) different from a linear model with the same basis functions
(Appendix H). A similar model has been studied by Saxe et al. [19] in the context of linear neural
networks (Appendix B.2).

For the multilinear model, note that ak(T)bk(T) is the skill strength Rk (Eq. (7)) and the skill loss
(Eq. (6)) is a function of S and Rk only:

ak(T)bk(T) = Rk(T),
Lk(T) = 1

2(S −Rk(T))2 .
(10)

4


---Page Break---
Assuming that we are training the model on D samples from P(I, X), the empirical loss decomposes
into a sum of empirical skill losses because gk’s supports are mutually exclusive. This decouples the
dynamics of each skill (Rk(T)), which is analytically solvable under gradient flow (Appendix C.1).

L(D)(T) =
1
2D

ns
X

k=1
dk(S −Rk(T))2,
Rk(T)

S
=
1

1 +

S
Rk(0) −1

e−2η
dk

D ST ,
(11)

where dk is the number of samples of the kth skill (i.e., number of samples (i, x) with gk(i, x) ̸= 0),
η is the learning rate, and 0 < Rk(0) < S is the skill strength at initialization.

4
Scaling laws

Recent literature has extensively explored scaling laws; see Section 7 for an overview. In this section,
we derive the scaling laws of our multilinear model (Section 3) for time (T), data (D), parameters
(N) and optimal compute (C). We define compute as C := T × N [21].

Table 2 shows our analytical scaling laws including their prefactor constants (Appendix J) and Fig. 2
compares the simulation of our model with our scaling law predictions. For the scaling law exponents,
we achieve the same exponent as in Hutter [20] for D and in Michaud et al. [18] for T, D, and N.
Assuming 0 < α < 1, the exponents are consistent with the small power-law exponents reported in
large-scale experiments, see, e.g., [9, 14, 22].

Using Eqs. (6), (10) and (11), we derive the loss as a function of time (T), data (D), parameters (N),
and the number of observations for each skill [d1, · · · , dns]:

L = S2

2

N
X

k=1
Ps(k)
1

1 +

S
Rk(0) −1
−1
e2η
dk

D ST
2 + S2

2

ns
X

k=N+1
Ps(k).
(12)

Under suitable assumptions (e.g., for the T scaling law, we take D, N →∞and dk/D →Ps(k)),
we can use Eq. (12) to derive the scaling laws. For T, D, and N, we used Eq. (11) – decoupled
dynamics induced the basis functions gk – to decouple the evolution of each skill loss:

1. For the time scaling law, each Lk shares the same dynamics with T scaled by Ps(k).

2. For the data scaling law, each Lk depends only on the observation the kth skill (dk > 0).
3. For the parameter scaling law, each Lk depends on whether the model has gk as a basis
function.

For the optimal compute scaling law, we show in Corollary 4 (Appendix J) that the optimal tradeoff
between T and N for given C is when T is large enough to fit the N th skill (Fig. 3). In Appendix J, we
show rigorous derivations of all scaling laws, including the prefactors, error bounds, and conditions
(e.g., how large N must be compared to T to be treated as infinity). For simplified derivations for the
exponents only, see Appendix E. For an intuitive derivation (stage-like training) and connection to
Michaud et al. [18], see Appendix D.

5
Predicting emergence

The literature on emergence has rapidly expanded lately; for a review of these developments, see
Section 7. In this section, we analyze the emergence of a 2-layer NN (Section 2) and discuss to
what degree the emergence in NNs can be described with our model. At initialization, NNs lack the
information about the data and must ‘discover’ each gk. To take this effect into account in our model,
we add an extra parameter which we calibrate (fit) on an NN trained on one skill (ns = 1) system
and use it to predict the emergence of subsequent skills for the ns = 5 setup (Fig. 1).

5.1
Time emergence

In our multilinear model, the layerwise structure – the product of parameters akbk – leads to a
sigmoidal saturation where an update of one layer hastens the update of the other layer. Feature

5


---Page Break---
100
101
102
103
104

T

10
2

10
1

L

(a) Time scaling

100
101
102
103

D

10
1

L

(b) Data scaling

100
101
102

N

10
2

10
1

L

(c) Parameter scaling

α = 0.3
α = 0.6
α = 0.9

L = ATT −α/(α + 1)

D, N →∞

L = ADD −α/(α + 1)

N, T →∞

L = ANN −α
T, D →∞

α = 0.3
α = 0.6
α = 0.9

Figure 2: Scaling laws. The learning curve (L is the MSE loss) of the multilinear model (solid) and
the theoretical power-law (dotted) for (a) time T, (b) data D, and (c) parameters N. Lower left legends
show the condition (top) and the scaling law (bottom) where α + 1 is the exponent of the power-law
input data (Eq. (1)). See the appendices for 1) rigorous derivations of the theoretical scaling laws
including the exponents, prefactors (e.g., AN for L = ANN −α), and conditions (Appendix J); 2)
simplified derivations of the exponent only (Appendix E); 3) details of the experiment (Appendix K.4).

Table 2: Summary of the scaling laws for the multilinear model. The leftmost column indicates
the bottleneck resource while the next two columns are the conditions for the ‘large resources’ – large
enough to be treated as infinity. The fourth column is the bottleneck resource’s scaling law exponent
for the loss. The last two columns show the statement for the prefactor constant and the scaling law
(with the assumptions and explicit error terms) in Appendix J.

Bottleneck
Condition 1
Condition 2
Exponent
Prefactor
Scaling law

Time (T)
D ≫NT 2, T 3
N α+1 ≫T
−α/(α+1)
Thm.4
Thms.2,3
Data (D)
T ≫D(log D)1+ϵ
N α+1 ≫D
−α/(α+1)
Thm.5
Thm.5
Parameter (N)
D ≫T 3
N α+1 = o(T)
−α
Thm.1
Thm.1
Compute (C)
D ≫T 3
N α+1 ≈T
−α/(α+2)
Cor. 5
Cor. 4

102
104
106
108

C

100

L

102
104
106
108

C

10
1

100

L

102
104
106
108

C

10
2

10
1

100

L

α = 0.3
α = 0.6
α = 0.9
α = 0.3
α = 0.6
α = 0.9

Figure 3: Scaling law for optimal compute. The solid lines are the learning curves of the multilinear
model as a function of compute C = T × N with varying parameters N from 101 (top plateau) to
104 (bottom plateau). The dotted lines are optimal compute scaling laws with exponent −α/(α + 2)
(Appendix E.4) and calculated prefactor constants (Appendix J). See Appendix K.4 for details of the
experiment. For a given C, we achieve the optimal tradeoff when T is large enough to fit all N skills
(i.e. when the solid lines plateau). For the case α = 0.3, the optimal C for the model decays faster
than the power-law, see Appendix E.1.

6


---Page Break---
learning dynamics in a 2-layer MLP shares the positive feedback between the layers but require a
non-trivial update of parameters to express gk.

Extended model.
Given that feature learning, though nonlinear, involves parameter updates, we
compensate for the additional delay in feature-learning by multiplying gk by a calibration constant
0 < B < 1:

fT (i, x; a, b) =

N
X

k=1
ak(T)bk(T)Bgk(i, x),
0 < B < 1.
(13)

The calibration constant B rescales the dynamics in T (Eq. (11)):

Rk(T)

S
=
1

1 +

S
Rk(0) −1

e−2ηPs(k)B2ST ,
(14)

where dk/D →Ps(k) because we assume D →∞. We observe that B2 = 1/22 fits the NN trained
on one skill (see Fig. 11 in Appendix I), and the calibrated model predicts emergence in the ns = 5
system (Fig. 1(a)): suggesting that the dynamics of feature-learning gk in 2-layers NNs is similar to
that of parameter learning (akbk) in a simple multilinear model. For further intuition of the extended
model, see an example of time emergence in an NN in Appendix G.

5.2
Data point emergence

Our multilinear model can learn the kth skill with a single observation of the skill because the skill
functions gk are built in (see Corollary 1 in Appendix C.2). NNs, without the fixed basis functions,
must ‘discover’ each gk, which requires multiple samples from the kth skill.

Extended model.
To make our model a Dc-shot learner, we extend it by replacing gk with the ek,l
basis:

fT (i, x; a, B) =

N
X

k=1
ak(T)

Dc
X

l=1
Bk,l(T)ek,l(i, x),
(15)

where the matrix B ∈RN×Dc is an extension of b ∈RN in Eq. (9), Dc is a fixed scalar, and
ek,l(i, x) : {0, 1}ns+nb →R are functions with the following properties:

EX|I=k [ek,lek,l′] = δll′,
ek,l(I ̸= k, x) = 0,

Dc
X

l=1

1
√Dc
ek,l = gk.
(16)

The first property states that ek’s, when I = k, are orthonormal in X. The second property asserts
that, similar to gk (Eq. (2)), ek,l is non-zero only when I = k, and fitting of the kth skill only occurs
among ek,l’s, keeping the skills decoupled. The third property states that gk can be expressed using
ek,l.

For the kth skill, the extended model overfits gk when there are fewer observations (dk) than the
dimension of the ek,l basis (Dc), and fits gk when dk ≥Dc, making our model a Dc shot learner.

Dc shot learner.
If we initialize the extended model in Eq. (15) with sufficiently small initialization
and if the conditions in Eq. (16) are satisfied, then the skill strength after training (T →∞) on D
datapoints is

Rk(∞) =

(
S

1 −
p

1 −dk/Dc

: dk < Dc
S
: dk ≥Dc.
(17)

The number dk is the number of samples in the training set for the kth skill (i.e., datapoints with
gk(i, x) ̸= 0).

Proof See Appendix F.3.
■

Using Eq. (17), we can calculate the emergence of Rk/S as a function of D. Note that Eq. (17)
is similar to the model in Michaud et al. [18] in that, to learn a skill, the model requires a certain
number of samples from the skill.

7


---Page Break---
The derivation of Eq. (17) follows trivially from the dynamics of the extended model (Eq. (15))
and well-known results in linear/kernel regression [23–27]. To be more specific, the model finds
the minimum norm solution as if we performed ridgeless regression on gk with basis functions
[ek,1, · · · ek,Dc]. See Appendix F.3 for details.

We observe that Dc = 800 approximates the data emergence for the ns = 1 system (see Fig. 11 in
Appendix I) and also the emergence for ns = 5 system (Fig. 1(b)), suggesting that the NN discovers
gk when it observes Dc samples from the kth skill.

5.3
Parameter emergence

Since our multilinear model has gk’s as the basis functions, it requires only one basis function (2
parameters) to express a skill (see Corollary 2 in Appendix C.3). A 2-layer NN cannot express a skill
with a single hidden node (i.e., a hidden layer with width 1); it requires multiple hidden nodes to
express a single skill.

Extended model.
To compensate for the need for multiple hidden nodes in expressing one skill,
we extend our model similarly to Eq. (15). Because the number of parameters is now a bottleneck,
we ensure the model has N basis functions (ek,l’s):

fT (i, x; a, B) =

q−1
X

k=1

Nc
X

l=1
ak(T)Bk,l(T)ek,l(i, x) +

r
X

l′=1
aq(T)Bq,l′(T)eq,l′(i, x),
(18)

where Nc is the number of basis functions needed to express a skill, quotient q is ⌊(N −1)/Nc⌋+ 1
and remainder r is such that (q −1)Nc + r = N. In short, the N basis functions are
[e1,1, · · · , e1,Nc,
e2,1, · · · , e2,Nc
· · ·
eq,1, · · · , eq,r].
(19)
Similar to Eq. (16), the basis functions satisfy the following properties

EX|I=k [ek,lek,l′] = δll′,
ek,l(I ̸= k, x) = 0,

Nc
X

l=1

1
√Nc
ek,l = gk.
(20)

Nc basis functions for a skill.
For the extended model in Eq. (18), the skill strength at T, D →∞
for a given N becomes

Rk(∞) =






0
: k > q
S r

Nc
: k = q
S
: k < q .
(21)

Proof See Appendix F.4.
■

The model can express the kth skill based on the number of available basis functions for the given
skill (Eq. (21)). For example, skills with k < q have all Nc basis functions [ek,1, · · · , ek,Nc] to
express the kth skill (Eq. (20)), while for k = q, only r of the Nc basis functions are available.

We observe that Nc = 4 fits the parameter emergence for the ns = 1 system (see Fig. 11 in
Appendix I) and also the emergence for the ns = 5 system (Fig. 1(c)), suggesting that the NN
requires 4 nodes in expressing gk. The results also suggest that an NN, while lacking the ordering
of basis functions (Eq. (19)), prefers to use the hidden neuron in fitting more frequent skills. The
‘preference’ toward frequent skills agrees with Fig. 1(a) where the NN learns more frequent skills first.
Note that for the parameter emergence experiment, Adam [28] was used, instead of SGD, to increase
the chance of escaping the near-flat saddle points induced by an insufficient number of parameters.

5.4
Time emergence in a transformer

To test whether our conceptual framework extends to other architectures, we perform a time emergence
experiment with a transformer (Fig. 4). Note that the emergent time τemerge – when the skill strength
is sufficiently larger than 0 – follows the same power-law relationship as Eq. (11): τemerge(k) ∝kα+1
(see Fig. 6 in Appendix D for a discussion on emergent time). This suggests that, in the multitask
sparse parity setup, other architectures may follow similar decoupled dynamics (Eq. (11)) and the
consequent scaling laws (Section 4) and emergence (Section 5). An in-depth study of these findings
across different architectures is left for future work.

8


---Page Break---
103
104
105
106

T

0.0

0.5

1.0

Rk/S

k = 1
k = 2
k = 3
k = 4
k = 5

1
2
3
4
5
Skill index k

104

105

τemerge(k)

τemerge(k) ∝k α + 1

Figure 4: Transformer on multitask sparse parity task. We trained a transformer on the multitask
sparse parity task with α = 0.9; see Appendix K for details. Left: An example of the time emergence
(measued in steps) for the transformer in the ns = 5 setup. See Appendix I for enlarged plots showing
the saturation of each skill in linear scale. Right: The kth skill’s emergent time τemerge(k) (i.e.
Rk(τemerge(k))/S = 0.05) as a function of k (error bars indicate 1-standard deviation over 5 runs).
The emergent times follow a power law of kα+1, following the same relationship in the multilinear
model (Eq. (11)).

5.5
Limitations of the multilinear model

The strength of our extended multilinear model comes from the decoupled dynamics for each skill:
leading to the prediction of the time, data, and parameter emergence with a single calibration. The
weakness of our model is that it simplifies the more complex dynamics of NNs.
Time emergence.
We note that the NN and the multilinear model emerge at similar instances, but
the NN takes longer to saturate fully. This is because, for a given skill, the dynamics of the NN is not
one sigmoidal saturation but a sum of multiple sigmoidal dynamics with different saturation times.
To express the parity function, the NN must use multiple hidden neurons, and the skill strength can
be divided into the skill strength from each neuron whose dynamics follow a sigmoidal saturation.
Because of the non-linearity and the function it expresses, each neuron is updated at different rates,
and the slowly saturating neurons result in a longer tail compared to our multilinear model. For an
example, see Fig. 8 in Appendix G.
Data point emergence.
Our extended model (Eq. (17)) deviates from NNs when dk ≪Dc and
NNs show a more abrupt change in Rk as a function of D. This is because our model asserts strict
decoupling among the skills: even a few dk will contribute to learning gk from ek,l. This differs
from the NN, which lacks strict decoupling among the samples from different skills. We speculate
that because NNs can perform benign [29] or tempered [30] overfitting, they treat a few data points
from less frequent skills as ‘noise’ from more frequent skills: requiring more samples to learn the
infrequent skills.
Parameter emergence.
Note that Fig. 1(c) has high variance compared to other emergence plots
in Fig. 1; this is because the NN sparsely, over many repeated trials, uses the hidden neurons to
learn less frequent skills over more frequent ones (see Table 5 in Appendix I for an example of such
outliers). Because NNs are less strictly biased toward frequent skills than our model, we speculate
that initial conditions favoring less frequent skills may contribute to the outliers.

6
Discussion and conclusion

This work demonstrated scaling laws and predicted emergence in a 2-layer MLP using a tractable
multilinear model. We found that representing skills as mutually exclusive functions leads to the
decoupled dynamics, resulting in the scaling laws observed in a 2-layer MLP. The layerwise structure
leads to emergent (sigmoidal) saturation of the skill strength, similar to what is observed in 2-layer
MLPs.

Despite lacking explicit skill functions, NNs exhibit similar emergence patterns. We speculate that
the model’s layerwise structure and power-law frequencies of the skills induce stage-like dynamics

9


---Page Break---
(Appendix D) in NNs. The parameters relevant for expressing more frequent skills are updated
significantly faster than those for less frequent skills. When skill ‘discovery’ operates on different
time scales with minimal interaction, the skill dynamics effectively become decoupled, justifying
our model setup.

Our results suggest a link between feature learning and emergence [6] driven by decoupled, stage-like
dynamics. The layerwise dynamics leading to sigmoidal saturation may also disentangle the problem
into skills (features) of varying importance (frequencies). Then feature learning, or discovering
the basis functions that describe the target function [31, 32] (for recent studies, see [33–38]), likely
occurs in stages. Investigating this connection through layerwise dynamics is left for future work.

Similar to many prior works (see, e.g., [20, 18]), we studied a simple model on an idealized power-
law distributed dataset. Also, our model cannot capture the complex non-linear interactions among
multiple skills but can express any linear superposition of skills. In future work, we will explore
‘complex skills’ in language as a superposition of linearly independent skills. By validating our
findings in language tasks, we aim to contribute to a broader understanding of how neural networks
acquire and exhibit complex behaviors.

7
Related works

In this section, we review the literature on scaling laws and emergence in NNs. Focusing on data
scaling, Hutter [20] develops a model with a discrete set of features. Under the assumption of a
power-law distribution of features, this model demonstrates that the error decreases as a power law
with increasing data size. In a related vein, Michaud et al. [18] propose a model of neural scaling laws
in which the loss is decomposed into a sum over ‘quanta’. Their model aims to reconcile the apparent
discrepancy between loss metrics’ regular power-law scaling and the abrupt development of novel
capabilities in large-scale models. Various other models for neural scaling laws have been proposed
in recent research, including connecting neural scaling exponents to the data manifold’s dimension
[39] and their relation with kernels [40], proposing solvable random-feature models [41, 21], and
developing data scaling models using kernel methods [42, 43, 25].

Closely related to the study of neural scaling laws is the understanding of emergent abilities in large
language models. Several studies [1–4] document examples of such emergent abilities. Arora and
Goyal [44] propose a framework for the emergence of tuples of skills in language models, in which
the task of predicting text requires combining different skills from an underlying set of language
abilities. Okawa et al. [45] demonstrate that a capability composed of smoothly scaling skills will
exhibit emergent scaling due to the multiplicative effect of the underlying skills’ performance. Other
works related to the skill acquisition include Yu et al. [46], who introduce a new evaluation to measure
the ability to combine skills and develop a methodology for grading such evaluations, and Chen et al.
[47], who formalize the notion of skills and their natural acquisition order in language models.

10


---Page Break---
Acknowledgements

NF acknowledges the UKRI support through the Horizon Europe guarantee Marie Skłodowska-Curie
grant (EP/X036820/1). SL was supported by the National Research Foundation of Korea (NRF) grant
funded by the Korean government (MSIT) (No.2020R1A5A1016126). We thank Charles London,
Eric Michaud, Zohar Ringel, and Shuofeng Zhang for their helpful comments.

References

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[2] Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom
Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, et al. Predictability and surprise in large
generative models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability,
and Transparency, pages 1747–1764, 2022.

[3] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid,
Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al.
Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.
arXiv preprint:2206.04615, 2022.

[4] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani
Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large
language models. arXiv preprint: 2206.07682, 2022.

[5] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language
models a mirage? Advances in Neural Information Processing Systems, 36, 2023.

[6] Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase,
Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman,
Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric
Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó
hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards,
Yoshua Bengio, Danqi Chen, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer,
He He, Atoosa Kasirzadeh, Yejin Choi, and David Krueger. Foundational challenges in assuring
alignment and safety of large language models. arXiv preprint: 2404.09932, 2024.

[7] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Con-
erly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu,
Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex
Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter,
Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language
models with dictionary learning. Transformer Circuits Thread, 2023. https://transformer-
circuits.pub/2023/monosemantic-features/index.html.

[8] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan
Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is
predictable, empirically. arXiv preprint:1712.00409, 2017.

[9] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models. arXiv preprint:2001.08361, 2020.

[10] Jonathan S Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive
prediction of the generalization error across scales. arXiv preprint: 1909.12673, 2019.

[11] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson,
Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive
generative modeling. arXiv preprint:2010.14701, 2020.

11


---Page Break---
[12] Mitchell A Gordon, Kevin Duh, and Jared Kaplan. Data and parameter scaling laws for
neural machine translation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and
Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing, pages 5915–5922, Online and Punta Cana, Dominican Republic,
November 2021. Association for Computational Linguistics.

[13] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transform-
ers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 12104–12113, 2022.

[14] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
Training compute-optimal large language models. arXiv preprint:2203.15556, 2022.

[15] Vivien Cabannes, Elvis Dohmatob, and Alberto Bietti. Scaling laws for associative memories.
arXiv preprint arXiv:2310.02984, 2023.

[16] Gregor Bachmann, Sotiris Anagnostidis, and Thomas Hofmann. Scaling mlps: A tale of
inductive bias. Advances in Neural Information Processing Systems, 36, 2024.

[17] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang.
Hidden progress in deep learning: Sgd learns parities near the computational limit. Advances in
Neural Information Processing Systems, 35:21750–21764, 2022.

[18] Eric Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural
scaling. Advances in Neural Information Processing Systems, 36, 2023.

[19] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dy-
namics of learning in deep linear neural networks. Proceedings of the International Conference
on Learning Representations 2014, 2014. arXiv:1312.6120.

[20] Marcus Hutter. Learning curve theory. arXiv preprint:2102.04074, 2021.

[21] Blake Bordelon, Alexander Atanasov, and Cengiz Pehlevan. A dynamical model of neural
scaling laws. arXiv preprint:2402.01092, 2024.

[22] Tamay Besiroglu, Ege Erdil, Matthew Barnett, and Josh You. Chinchilla scaling: A replication
attempt. arXiv preprint:2404.10102, 2024.

[23] Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Spectral bias and task-model
alignment explain generalization in kernel regression and infinitely wide neural networks.
Nature communications, 12(1):2914, 2021.

[24] Arthur Jacot, Berfin Simsek, Francesco Spadaro, Clément Hongler, and Franck Gabriel. Kernel
alignment risk estimator: Risk prediction from training data. Advances in Neural Information
Processing Systems, 33:15568–15578, 2020.

[25] Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová. Generalization error rates
in kernel regression: The crossover from the noiseless to noisy regime. Advances in Neural
Information Processing Systems, 34:10131–10143, 2021.

[26] Ouns El Harzli, Bernardo Cuenca Grau, Guillermo Valle-Pérez, and Ard A Louis. Double-
descent curves in neural networks: a new perspective using gaussian processes. In Proceedings
of the AAAI Conference on Artificial Intelligence, pages 11856–11864, 2024.

[27] James B Simon, Madeline Dickens, Dhruva Karkada, and Michael R DeWeese. The eigenlearn-
ing framework: A conservation law perspective on kernel regression and wide neural networks.
Transactions on Machine Learning Research, 2023. arXiv:2110.03922.

[28] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv
preprint:1412.6980, 2014.

[29] Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in
linear regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070,
2020.

12


---Page Break---
[30] Neil Mallinar, James Simon, Amirhesam Abedsoltan, Parthe Pandit, Misha Belkin, and Preetum
Nakkiran.
Benign, tempered, or catastrophic: Toward a refined taxonomy of overfitting.
Advances in Neural Information Processing Systems, 35:1182–1195, 2022.

[31] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and
new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–
1828, 2013.

[32] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436,
2015.

[33] Greg Yang and Edward J Hu. Tensor programs iv: Feature learning in infinite-width neural
networks. In International Conference on Machine Learning, pages 11727–11737. PMLR,
2021.

[34] Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners:
The silent alignment effect. arXiv preprint arXiv:2111.00034, 2021.

[35] Arthur Jacot, Eugene Golikov, Clément Hongler, and Franck Gabriel. Feature learning in l_2-
regularized dnns: Attraction/repulsion and sparsity. Advances in Neural Information Processing
Systems, 35:6763–6774, 2022.

[36] Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution
in wide neural networks. Advances in Neural Information Processing Systems, 35:32240–32256,
2022.

[37] Inbar Seroussi, Gadi Naveh, and Zohar Ringel. Separation of scales and a thermodynamic
description of feature learning in some cnns. Nature Communications, 14(1):908, 2023.

[38] Hugo Cui, Luca Pesce, Yatin Dandi, Florent Krzakala, Yue M Lu, Lenka Zdeborová, and Bruno
Loureiro. Asymptotics of feature learning in two-layer networks after one gradient-step. arXiv
preprint arXiv:2402.04980, 2024.

[39] Utkarsh Sharma and Jared Kaplan. Scaling laws from the data manifold dimension. Journal of
Machine Learning Research, 23(9):1–34, 2022. arXiv:2004.10802.

[40] Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining
neural scaling laws. arXiv preprint:2102.06701, 2021.

[41] Alexander Maloney, Daniel A Roberts, and James Sully. A solvable model of neural scaling
laws. arXiv preprint:2210.16859, 2022.

[42] Stefano Spigler, Mario Geiger, and Matthieu Wyart. Asymptotic learning curves of kernel
methods: empirical data versus teacher–student paradigm. Journal of Statistical Mechanics:
Theory and Experiment, 2020(12):124001, 2020.

[43] Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning
curves in kernel regression and wide neural networks. In International Conference on Machine
Learning, pages 1024–1034. PMLR, 2020.

[44] Sanjeev Arora and Anirudh Goyal. A theory for emergence of complex skills in language
models. arXiv preprint:2307.15936, 2023.

[45] Maya Okawa, Ekdeep S Lubana, Robert Dick, and Hidenori Tanaka. Compositional abilities
emerge multiplicatively: Exploring diffusion models on a synthetic task. Advances in Neural
Information Processing Systems, 36, 2024.

[46] Dingli Yu, Simran Kaur, Arushi Gupta, Jonah Brown-Cohen, Anirudh Goyal, and Sanjeev
Arora. Skill-mix: A flexible and expandable family of evaluations for ai models. arXiv
preprint:2310.17567, 2023.

[47] Mayee Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christo-
pher Ré. Skill-it! a data-driven skills framework for understanding and training language
models. Advances in Neural Information Processing Systems, 36, 2023.

13


---Page Break---
[48] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking:
Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177, 2022.

[49] Irina Gennad’evna Shevtsova. Sharpening of the upper bound of the absolute constant in the
berry–esseen inequality. Theory of Probability and Its Applications, 51(3):549–553, 2007.

[50] Hugh L Montgomery and Robert C Vaughan. Multiplicative Number Theory I: Classical Theory.
Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2007.

14


---Page Break---
A
Glossary

A
Normalization constant for Ps such that Ps(k) = Ak−(α+1)
T
Time or step
D
Number of data points
N
Number of parameters (skill basis functions in the model for the
multilinear model; the width of hidden layer for MLP)
C
The computation cost T × N
ns
The number of skills in the multitask sparse parity problem
I
Random variable of the control bits
X
Random variable of the skill bits
Ps
Probability of skills (control bits)
Pb
Probability of skill bits
S
The target scale or the norm of the target function
Rk
Skill strength of the kth skill (Eq. (7))
L
Total (generalization) loss
L(D)
Empirical loss for D samples
Lk
Skill loss of the kth skill (Eq. (6))
dk
Number of observation of the kth skill (i.e. number of training points
(i, x) with gk(i, x) ̸= 0)
f ∗
Target function f ∗: {0, 1}ns+nb →{−S, S} (Eq. (4))
gk
The kth skill basis function gk : {0, 1}ns+nb →{−1, 0, 1} (Eq. (2))

15


---Page Break---
Table 3: Representation of the multitask sparse parity as presented in [18]. The control bits are
one-hot vectors encoding a specific parity task. The frequency of the different tasks follows a power-
law distribution. In this example, there are ns = 10 tasks, and skill bits are length nb = 15. The
y column is the resulting parity computed from m = 3 bits (highlighted in colors). The multitask
dataset provides a controlled experimental setting designed to investigate skills.

Control bits
Skill bits
y

10000000000
110001000001010
1
01000000000
010100100001000
0
00100000000
001101010110101
1
...
...
...
00000000001
100010001001100
1

B
Background

In this section, we review the multitask sparse parity dataset, as described by Michaud et al. [18] and
discuss the nonlinear dynamics of two-layer linear networks, following the work of Saxe et al. [19].

B.1
Multitask sparse parity

The sparse parity task can be stated as follows: for a bit string of length nb, the goal is to determine
the parity (sum mod 2) of a predetermined subset of m bits within that string. The multitask sparse
parity [18] extends this problem by introducing ns unique sparse parity variants in the dataset. The
input bit strings have a length of ns + nb. The first ns bits function as indicators by assigning a
specific task. The frequency of the distinct parity tasks follows a rank-frequency distribution with an
inverse power law relation (power-law distribution). The last nb bits are uniformly distributed. This
sets a binary classification problem {0, 1}ns+nb →{0, 1} where only a single bit of the initial ns
bits is nonzero. In Table 3, the many distinct parity tasks represent different skills. 2

The proposal in [18] aims to reconcile the regularity of scaling laws with the emergence of abilities
with scale using three key hypotheses: (i) skills, represented as a finite set of computations, are
distinct and separate; (ii) these skills differ in their effectiveness, leading to a ranking based on their
utility to reduce the loss; and (iii) the pattern of how frequently these skills are used in prediction
follows a power-law distribution. Interestingly, the multitask problem has a consistent pattern across
scaling curves: each parity displays a distinct transition, characterized by a sharp decrease in loss at a
specific scale of parameters, data, or training step. Such a sudden shift occurs after an initial phase of
no noticeable improvement, leading to reverse sigmoid-shaped learning curves. Michaud et al. [18]
empirically show that for a one-hidden-layer neural network with ReLU activation, trained using
cross-entropy loss and the Adam optimizer, these transitions happen at different scales for distinct
tasks. This results in a smooth decrease in the overall loss as the number of skill levels increases.

B.2
Nonlinear dynamics of linear neural network

Saxe et al. [19] have solved the exact dynamics for two-layer linear neural networks with gradient
descent under MSE loss (Fig. 5(a)).3 The dynamics decompose into independent modes that show
sigmoidal growth at different timescales (Fig. 5(c)). The setup assumes orthogonal input features
X ∈Rd1 and input-output correlation matrix Σ ∈Rd1×d3 for target output f ∗(X) ∈Rd3:

EX [XiXj] = δij,
Σ = EX

Xf ∗T (X)

(22)

2Note that here we follow the even/odd parity convention used in [18], i.e., {0, 1}, instead of {1, −1} as
used in the main text.
3To be specific, it is under gradient flow or the continuous limit of full batch gradient descent.

16


---Page Break---
...

...

(a) Linear neural network

...

...

(b) Independent modes

0
25
50
75
100
T

0.0

0.2

0.4

0.6

0.8

1.0

akbk/λk

λ1 = 1.0
λ2 = 0.5
λ3 = 0.25

(c) Dynamics of modes

Figure 5: Nonlinear dynamics of linear neural networks. (a): A two-layer undercomplete linear
neural network, which is a multiplication of two matrices, where d2 < d1 and d2 < d3. (b): The d2
independent modes of dynamics for linear neural network (Eq. (24)). The product of parameters akbk
are learnable parameters and vectors uk, vk are obtained from SVD of the input-output correlation
matrix Σ (Eq. (22)). (c): The temporal evolution of akbk under gradient descent, which follows
a sigmoidal growth (Eq. (25)). Note that smaller λk – the singular value of Σ – results in a more
delayed saturation of akbk.

By performing SVD (singular value decomposition) on input-output correlation matrix Σ = UΛV ,
the target function f ∗: Rd1 →Rd3 becomes:

f ∗(x) =

d2
X

k=1
vkλkuT
k x,
U T ΛV = EX

Xf ∗(X)T 
(23)

where uk ∈Rd1,vk ∈Rd3 are the row vectors of U, V and λk ∈R are the singular values of Λ.

Saxe et al. [19] have shown that the dynamics of a two-layer (one-hidden-layer) undercomplete (the
width of the hidden layer is smaller than the width of the input and output) linear neural network
decomposes into that of the following ‘modes’:

vT
k f(x; a, b) = akbkuT
k x
k ∈{1, 2, · · · , d2}.
(24)

where ak, bk ∈R are the parameters. Note that Eq. (24) are d2 decoupled functions vT
k f(x) : Rd1 →
R (Fig. 5(b)). Assuming small and positive initialization (0 < ak(0)bk(0) ≪λk), the dynamics
of Eq. (24) under gradient descent with learning rate η can be solved analytically; the product of
parameters akbk grows sigmoidally with saturation time proportional to λ−1
k
(Fig. 5(c)):

ak(T)bk(T)

λk
=
1

1 +

λk
ai(0)bi(0) −1

e−2ηλkt .
(25)

Using the analytic equation of the multilinear model, Saxe et al. [19] have empirically demonstrated
that the dynamics of both linear and nonlinear neural networks closely resemble that of the multilinear
model (Eq. (25)).

17


---Page Break---
C
Derivation of the multilinear model

In this section, we provide derivations of how the skill loss of our multilinear model evolves with
a given resource: time (Lemma 1), data (Corollary 1), and parameters (Corollary 2). Note that
two corollaries for data and parameters (Corollaries 1 and 2) follow from the decoupled dynamics
(Lemma 1).

C.1
Decoupled dynamics of the multilinear model

Lemma 1. Let the multilinear model Eq. (9) be trained with gradient flow on D i.i.d samples for the
setup in Section 2 (input distribution: Eq. (1), target function: Eq. (4), and MSE loss: Eq. (5)). Let
k ≤N be a skill index in the multilinear model and the input distribution (k ≤ns). Then assuming
the following initialization ak(0) = bk(0) and 0 < ak(0)bk(0) < S, the dynamics of the kth skill
strength (Rk) is

Rk(T) =
S

1 +

S
Rk(0) −1

e−2ηS
dk

D T
(26)

and the skill loss is

Lk(T) =
S2

2

1 +

S
Rk(0) −1
−1
e2ηS
dk

D T
2 ,
(27)

where η is the learning rate and dk is the number of observations with gk(I = k, x(jk)) ̸= 0.

Proof For j = 1, · · · , D, denote (i(j), x(j)) be the jth data point in the training set. Then the
empirical loss for D datapoints is given as

L(D) =
1
2D

D
X

j=1


f ∗(i(j), x(j)) −f(i(j), x(j))
2
.
(28)

We note that


f ∗(i(j), x(j)) −f(i(j), x(j))
2
=

 ns
X

k=1
(S −akbk)gk(i(j), x(j))

!2

= (S −ai(j)bi(j))2gi(j)(i(j), x(j))2

= (S −ai(j)bi(j))2,

as gi(i, j) ∈{1, −1} and gk(i, j) = 0 for i ̸= k. So if we denote dk the number of data points with
i(j) = k, then we can conclude

L(D) =
1
2D

D
X

j=1
(S −ai(j)bi(j))2 =
1
2D

ns
X

k=1
dk(S −akbk)2,
(29)

which is the decoupled loss in the main text (Eq. (11)). Using the gradient descent equation and
Eq. (29), we obtain

dak

dt = −η dLD

dak
(30)

= −η dk

D bk(akbk −S).
(31)

Likewise, we can obtain the equation for bk as

dbk

dt = −η dk

D ak(akbk −S).
(32)

18


---Page Break---
Because of symmetry between a and b (See Appendix B.2 or [19]), assuming ak(0) = bk(0), and
ak(0)bk(0) > 0 results in ak(T) = bk(T) for all T. The equation for Rk = akbk is

dRk

dt
= −η dak

dt bk + ak
dbk

dt = −η dk

D (b2
k + a2
k)(akbk −S)
(33)

= −2η dk

D Rk(Rk −S).
(34)

Assuming ak(0)bk(0) < S, we can solve the differential equation to obtain

Rk(T) =
S

1 +

S
Rk(0) −1

e−2ηS
dk

D T .
(35)

The equation for Lk follows from Eq. (10).
■

C.2
One-shot learner

Corollary 1. For the setup in Lemma 1, the kth skill loss (Lk) at T, N →∞is

Lk(∞) =

0
: dk > 0
(S −Rk(0))2/2 ≈S2/2
: dk = 0,
(36)

where dk is the number of kth skill’s observations.

Proof The corollary follows directly from Lemma 1. By taking T, N →∞,

Rk(∞) =

S
: dk > 0
Rk(0)
: dk = 0
(37)

We obtain the result by using the relationship between Rk and Lk in Eq. (10).
■

C.3
Equivalence between a basis function and a skill

Corollary 2. Let the multilinear model Eq. (9) be trained with gradient flow on D i.i.d samples for
the setup in Section 3 (input distribution: Eq. (1), target function: Eq. (4), and MSE loss: Eq. (5)).
Assume ak(0) = bk(0), 0 < ak(0)bk(0) < S, and that the model has the N most frequent skills as
basis functions. Then Rk for the kth ≤ns skill at T, D →∞is

Lk(∞) =

0
: k ≤N
S2/2
: k > N
(38)

Proof The corollary follows directly from Lemma 1. By taking T, D →∞,

Rk(∞) =

S
: k ≤N
Rk(0)
: k > N
(39)

We obtain the result by using the relationship between Rk and Lk in Eq. (10) and Rk(0) ≪S.
■

19


---Page Break---
D
Stage-like training: intuitive derivation of the scaling laws

Even though we provide more detailed (Appendix E) and rigorous (Appendix J) derivation of the
scaling laws, a less general yet more intuitive solution aids in understanding the scaling laws of our
model and NNs. In this section, we define stage-like training – one skill is completely learned before
the next skill initiates learning (Fig. 6(a)) – and state the conditions for it to occur. We provide an
example of how stage-like training results in the time scaling law and explain how the model in
Michaud et al. [18] may arise from the NN dynamics. Finally, we discuss the stage-like training’s
role in emergence in NNs.

0
50
100
150
200
T

0.0

0.2

0.4

0.6

0.8

1.0

Rk(T)/S

ϵ

1 −ϵ

τ(e)
1 (ϵ)

τ(s)
1 (ϵ)

τ(e)
2 (ϵ)

k = 1
k = 2

(a) Emergent and saturation time

0
50
100
150
200
T

10.5

11.0

11.5

12.0

12.5

13.0

L

1
2S 2
N
X

k = 1
Ps(I = k)

∆τ(e)(ϵ)

1
2S 2Ps(1) + O(ϵ)

(b) Loss change between emergences

Figure 6: Stage-like training. The multilinear model is trained on the multitask sparse parity problem
with α = 0.6 and S = 5. (a): Skill strength of the model as a function of time. The emergent
time τ (e)
k (ϵ) is the time required for the kth skill to reach Rk/S = ϵ. The saturation time τ (s)
k (ϵ)
is the time required for Rk/S to saturate from ϵ to 1 −ϵ. The model shows stage-like training if
the emergent time interval τ (e)
k+1(ϵ) −τ (e)
k (ϵ) is larger than the saturation time τ (s)
k (ϵ) for sufficiently
small ϵ (0.05 in the figure). (b): The loss as a function of time for the same system as (a). For
stage-like training, the change in the loss for the kth emergence is Ps(k)Lk + O(ϵ) and the interval
for the next emergence is ∆τ (e)(ϵ) = τ (e)
k+1(ϵ) −τ (e)
k (ϵ).

D.1
Stage-like training

When a model exhibits an emergence behavior – when saturation of skill occurs abruptly after a delay
– and the intervals between each emergence are sufficiently large, the model admits stage-like training.
The multilinear model (sigmoidal saturation of skills strength, Eq. (11)) in the multitask sparse parity
dataset (power-law decay of skill frequencies, Eq. (1)) can satisfy such conditions: In Fig. 6(a), we
observe the stage-like training in time in which one skill saturates (reaches Rk/S ≈1) before the
next skill initiates its emergence. To quantify this behavior, we define two intervals for each skill (see
Fig. 6(a)):

• The emergent time τ (e)
k (ϵ): the time for Rk/S to reach ϵ;

• The saturation time τ (s)
k (ϵ): the time for Rk/S to saturate from ϵ to 1 −ϵ.

Using the dynamics equation (Eq. (11)) and that dk/D →Ps(k), the emergent time and saturation
time of the kth skill becomes

τ (e)
k (ϵ) =
1
2ηPs(k)S ln

 
S
Rk(0) −1

1
ϵ −1

!

∝kα+1,
τ (s)
k (ϵ) =
1
ηPs(k)S ln
1

ϵ −1

∝kα+1.

(40)
For sufficiently small initialization (Rk(0) ≪S), we get a stage-like training:

τ (s)
k (ϵ) < τ (e)
k+1(ϵ) −τ (e)
k (ϵ),
ϵ ≪1.
(41)

where the model finishes learning (saturating) the kth skill before starting to learn (emerging) the
next skill.

20


---Page Break---
D.2
Time scaling law from stage-like training

Assuming our model satisfies the stage-like training for all k of interest, we can derive the time
scaling law from the stage-like training.

At τ (e)
k (ϵ), because of stage-like training, all skills with index up to but not including k have saturated
(Ri<k ≈S), or equivalently Li<k ≈0 (Eq. (10)). The total loss, the sum of Lj weighted by
Ps(j) ∝j−(α+1) (Eq. (6)), becomes P∞
j=k Ps(I = j)S2/2 (Fig. 6(b)). The saturation of the kth

skill results in a loss difference of Ps(I = k)S2/2. Thus, we obtain

∆L

L
≈
Ps(I = k)
P∞
j=k Ps(I = j) = −
k−(α+1)
P∞
j=k j−(α+1) ≈−
k−(α+1)
R ∞
k j−(α+1)dj
(42)

= −αk−1 + O(k−2).
(43)

Accordingly, the emergent interval between the k and k + 1 skills relative to the τ (e)
k (ϵ) is

∆T

T
= τ (e)
k+1(ϵ) −τ (e)
k (ϵ)

τ (e)
k (ϵ)
= (k + 1)α+1 −kα+1

kα+1
(44)

= (α + 1)k−1 + O(k−2).
(45)

Assuming k ≫1 and combining Eq. (43) and Eq. (45) to the largest order, we have the equation for
the power-law with exponent −α/(α + 1) in Fig. 2(a):

∆L

L
= −
α
α + 1
∆T

T .
(46)

If the stage-like training holds for any resource (e.g., time, data, or parameters), the scaling law can
be derived using the ratio of change in loss per skill (Eq. (43)) and the ratio of change with respect to
the resource (given by the emergent time in Eq. (45)). The quanta model in Michaud et al. [18] is an
example where the stage-like training holds for all resources.

D.3
Discussion on the effective decoupling of skills in neural networks

In Section 5, we have empirically demonstrated that the multilinear model predicts the emergence of a
2-layer NN (Fig. 1). In Section 6, we briefly discussed why NNs, despite their lack of the decoupling
among the skills, behave similarly to the decoupled model with gks as fixed basis functions: the
stage-like training in NNs – induced by the model’s layerwise structure and power-law frequencies
of the skills – effectively decouples the skills. In this subsection, we extend the discussion in more
detail.

In NNs, even though gks are ‘discovered’ (feature learned) by non-tractable dynamics, we speculate
that similar stage-like dynamics also hold in ‘discovering’ (feature learning) gks: parameters ‘useful’
for expressing more frequent skills will be updated significantly faster than parameters useful for
expressing less frequent skills.

If skill discovery and saturation dynamics operate at different time scales (stages), with negligible
interaction among the skills, the skill dynamics become effectively decoupled. Because the dynamics
are decoupled in stages, NNs repeat the feature learning process – using the limited resource (time,
data, parameters) to express the skill – for all skills with each iteration varying only in the scale of the
resource (e.g. training time, number of observations, and number of hidden layer neurons): resulting
in a similar emergence to our multilinear model.

A more concrete understanding of our speculation that feature learning also occurs in stages due to a
layerwise structure is left for future work.

E
Derivation of the scaling law exponents

This section provides a detailed derivation of the scaling laws up to a rigor common in physics and
engineering. For example, we approximate the Riemann sum as integral or treat k, the number of

21


---Page Break---
skills, as a differentiable parameter. For more general and rigorous derivations including the prefactor
constants, see Appendix J. Instead, for more intuition and the relationship to the quanta model in
Michaud et al. [18], see Appendix D.

Table 4: Summary of the scaling laws. The leftmost column shows the bottleneck of the scaling
law. The middle three columns show the resource values in terms of the bottleneck (either taken to
infinity or proportional to the bottleneck). The last column shows the scaling exponent for the loss as
power-law of the bottleneck where α + 1 is the exponent of the Zipfian input data (Eq. (1)).

Bottleneck
Time
Data
Parameter
Exponent

Time (T)
T
∞
∞
−α/(α + 1)
Data (D)
∞
D
∞
−α/(α + 1)
Parameter (N)
∞
∞
N
−α
Compute (C)
C(α+1)/(α+2)
∞
C1/(α+2)
−α/(α + 2)

E.1
Time scaling law exponent

To derive the time scaling law exponent, we assume the time as the bottleneck and take N, D →∞.
By using the decoupled dynamics of each skill loss (Lemma 1),

Lk =
S2

2

1 +

S
Rk(0) −1
−1
e2η
dk

D ST
2 .
(47)

Noting that dk/D →Ps(k) as D →∞, where Ps(k) = Ak−(α+1), we have

Lk =
S2

2

1 +

S
Rk(0) −1
−1
e2ηAk−(α+1)ST
2 .
(48)

This is a function of k−(α+1)T only, suggesting the decoupling dynamics for each skill. Thus,
dLk

dT = −
k
(α + 1)T
dLk

dk .
(49)

Using Eq. (6) and taking N, ns →∞at the same rate,4 we can approximate the loss as an integral
instead of a sum over k:

L ≈lim
N→∞

Z N

1
Ak−(α+1)Lkdk,
(50)

where A is the normalization constant for Ps. We can differentiate the loss and use Eq. (49) to express
the equation in terms of k:

dL
dT = lim
N→∞

Z N

1
Ak−(α+1) dLk

dT dk = −lim
N→∞
1
(α + 1)T

Z N

1
Ak−α dLk

dk dk.
(51)

Integrating by parts, we obtain

dL
dT = −lim
N→∞
1
(α + 1)T

Ak−αLk
N
1 −lim
N→∞
α
(α + 1)T

Z N

1
Ak−(α+1)Lkdk
(52)

= −lim
N→∞O

N −α 1

T


+ O
 1

TeT


−
α
(α + 1)T L.
(53)

The first term goes to 0 as N →∞and the second term goes to 0 exponentially faster compared to
the last term for T ≫1, which leads to the scaling law with exponent −α/(α + 1):

dL(T)

L(T) = −
α
α + 1
dT

T .
(54)

4We take N and ns to ∞at the same rate since we do not want the number of parameters to be a bottleneck
in this setup.

22


---Page Break---
Finite N correction for small α.
In Fig. 7, we observe that our model with α = 0.1 deviates
from the expected power-law with exponent −α/(α + 1). The deviation can be explained by the
antiderivative term in Eq. (52):

100
101
102
103

T

100

101

L

simulation
power law(corrected)

power law

α = 0.1
α = 0.3
α = 0.5
α = 0.7

Figure 7: Scaling law and corrected predictions. A simulation of our multilinear model with
N = 50, 000 (solid), a scaling law with exponent −α/(α + 1) (dotted), and a corrected scaling
law considering finite N (dashed, Eq. (56)). The finite N corrected scaling law better predicts the
dynamics, especially for smaller α.

lim
N→∞




1
2(α + 1)
S2A

1 +
1
S/Rk(0)−1e2ηSAk−(α+1)T
2
k−α

T





N

1

= lim
N→∞


O

N −α 1

T


−O
 1

TeT


.

(55)

The second term (k = 1) goes to 0 faster than O(T −1) for sufficiently larger T but the first term
(k = N) may not decay fast enough for finite N and sufficiently small α. For example, N = 50, 000
and α = 0.1 leads to N −α ≈0.3, which is not negligibly small.

Assuming finite N and small α such that the first term in Eq. (55) is non-negligible, we can rewrite
Eq. (52) as

dL
dT ≈−
α
(α + 1)
L + LC

T
,
LC ≈S2AN −α/2α,
(56)

where we assumed a small initialization S/Rk(0) ≫1 and sufficiently large number of parameters
N α+1 ≫T to approximate LC. Because the total loss at initialization is L(0) = S2/2, LC is
non-negligible compared to the loss for sufficiently small α. Thus considering LC, we obtain the
corrected power-law which better approximates the time scaling law (dashed lines in Fig. 7). For
a rigorous and comprehensive analysis of the time scaling law, see Theorem 2 and Theorem 3 in
Appendix J.

E.2
Data scaling law exponent

In this section, we derive the data scaling law exponent. The data scaling law assumes T →∞
and N →∞with data as the bottleneck. From the decoupled dynamics of the multilinear model
(Lemma 1), we can show that our model is a one-shot learner (Corollary 1):

One shot learner.
Given that N > k, T →∞, and dk is the number of samples from the training
set with gk(i, x) ̸= 0, the kth skill loss after training is

Lk(∞) =

0
: dk > 0
(S −Rk(0))2/2 ≈S2/2
: dk = 0.
(57)

Proof See Appendix C.2.
■

23


---Page Break---
Our model requires only one sample from the kth skill to learn such a skill, similar to how language
models are few-shot learners at inference.5 The model can one-shot learn a skill since it has gk as the
basis functions, and the dynamics among different skills are decoupled. A similar one-shot learner
has been studied in Hutter [20] where the error depends on a single ‘observation’ of a feature.

Because the kth skill loss only depends on dk (number of observations for the kth skill), we can
calculate the expectation of the skill loss for D data points from Pobserved(k|D) or the probability
that dk > 0:
Pobserved(k|D) = 1 −(1 −Ps(k))D .
(58)
Using the one-shot learning property (Eq. (57)), the probability of observing the kth skill (Eq. (58)),
and the decomposition of the loss into skill losses (Eq. (6)), the expected loss for D datapoints is

ED [L] = 1

2

∞
X

k=1
S2Ps(k)(1 −Pobserved(k))
(59)

= 1

2S2A

∞
X

k=1
k−(α+1) (1 −Ps(k))D
(60)

≈1

2S2A
Z ∞

1
k−(α+1) 
1 −Ak−(α+1)D
dk,
(61)

where the expectation ED is over all possible training sets of size D, and A is the normalization
constant such that P(k) = Ak−(α+1). The difference in the loss ∆L = ED+1 [L] −ED [L] is

∆L = 1

2S2A
Z ∞

1
k−(α+1) 
1 −Ak−(α+1)D 
1 −Ak−(α+1)
−1

dk
(62)

= −1

2S2A2
Z ∞

1
k−2(α+1) 
1 −Ak−(α+1)D
dk.
(63)

We can integrate ∆L by parts.

∆L = 1

2


−
S2Ak−α

(α + 1)(D + 1)


1 −Ak−(α+1)D+1∞

1

−
S2Aα
2(α + 1)(D + 1)

Z ∞

1
k−(α+1) 
1 −Ak−(α+1)D+1
dk

≈O
 
(1 −Ps(1))D+1
−
S2Aα
2(α + 1)(D + 1)

Z ∞

1
k−(α+1) 
1 −Ak−(α+1)D 
1 −Ak−(α+1)
dk

≈−
α
(α + 1)(D + 1)ED [L] +
α
(α + 1)(D + 1)∆L.

In the second line, the first term goes to 0 for D ≫1. In the last line, we used the expression for ∆L
(Eq. (62)) and ED [L] (Eq. (59)). Rearranging the equation above and using that D ≫1, we obtain
the scaling law with exponent −α/(α + 1):
∆L
ED [L] = −
α
1 + (α + 1)D ≈−
α
(α + 1)
1
D
(64)

= −
α
(α + 1)
∆D

D .
(65)

where in the last line, ∆D/D = 1/D as the change in the number of data points relative to D is one.

E.3
Parameter scaling law exponent

The parameter scaling law assumes T →∞and D →∞, with the parameters N < ns as the
bottleneck. Because our model is a one-shot learner (Eq. (57)), learning of the kth skill only depends
on the existence of gk in the model; the model with [g1, · · · , gN] will learn all k ≤N skills with
Lk = 0.

The Lk dependence on gk is formalized in Corollary 2, which we repeat here.

5Few-shot learning is typically discussed in the context of models that have undergone pre-training (see, e.g.
[1]). We speculate that expanding in the basis gk in our framework can model aspects of the pre-training process.

24


---Page Break---
Equivalence between a basis function and a skill.
Given T, D →∞and if the multilinear model
has the N most frequent skill functions as a basis,

Lk(∞) =

0
: k ≤N
S2/2
: k > N.
(66)

Proof See Appendix C.3.
■

Using Eq. (66) and Eq. (6), we can express the total loss as function of N:

L ≈S2

2

Z ∞

N+1
Ak−(α+1)dk ∝(N + 1)−α.
(67)

By approximating N ≈N + 1 for N ≫1, we obtain the power-law with exponent −α.

E.4
Optimal compute scaling law

For analytical tractability, we define compute as C := T × N. We start from Eq. (12) with D →∞

L ≈
Z N

1
Ak−(α+1)Lkdk + lim
ns→∞
S2

2

Z ns

N
Ak−(α+1)dk.
(68)

We can use Eq. (56) to calculate the first term and integrate the last term to get

L ≈(L(0) + LC)T −α/(α+1) −Lc + S2A

2α N −α
(69)

≈O(T −α/(α+1)) + O(N −α),
(70)

where we used that L(0) ≫LC and S2A/(2α) −LC > 0. Intuitively, the approximation shows
the tradeoff between T – when increased, decreases the loss of the first N skills – and N – when
increased, decreases the loss at sufficiently large T – for fixed compute C. For a comprehensive
analysis of the approximation above, see Appendix J.

Removing the irrelevant constant terms,

L = T −α/(α+1) + N −α.
(71)

We can use the method of Lagrangian multiplier to obtain

−
α
α + 1T −α/(α+1)−1 + λN = 0,
(72)

−αN −(α+1) + λT = 0,
(73)
NT −C = 0,
(74)

where λ is the Lagrange multiplier and C is compute. We can solve the above set of equations to
obtain T α+1 ∝N or equivalently

T ∝C(α+1)/(α+2),
N ∝C1/(α+2).
(75)

We can plug it in Eq. (71) to get
L ∝C−α/(α+2).
(76)
This derivation is similar to that of Bordelon et al. [21] (see Appendix N: Compute Optimal Scaling
from Sum of Power-Laws in [21]). For a rigorous derivation of the optimal compute scaling law, see
Corollary 4 and Appendix J.

25


---Page Break---
F
Derivation of the extended multilinear model

In this section, we show the derivation for the extended multilinear model.

F.1
Gradient flow in the extended multilinear model

Lemma 2. Let the extended multilinear model Eq. (15) be trained with gradient flow on D i.i.d
samples for the setup in Section 2 (input distribution: Eq. (1), target function: Eq. (4), and MSE loss:
Eq. (5)). For the skill index k ≤N be a skill index in the multilinear model, let the feature matrix
Φ ∈RDc×dk for the kth skill be

Φlj = ek,l(i(j) = k, x(j)),
(77)

and SVD on Φ = USV . Assuming that the system is overparametrized (dk < Dc), the gradient
on ⃗Bk ∈RDc ([Bk,1, · · · , Bk,Dc]) is contained in the column space of semi-orthogonal matrix
U ∈RDc×dk:

UU T d ⃗Bk

dt
= d ⃗Bk

dt .
(78)

Proof Similar to Lemma 1, the total loss can be decomposed into each skill such that the dynamics
of Bk,l relies only on dk observations of the kth skill:

LD =
1
2D

ns
X

k=1

D
X

j=1


f ∗(i(j), x(j)) −f(i(j), x(j))
2
(79)

=
1
2D

ns
X

k=1

dk
X

jk=1

 

Sgk(k, x(jk)) −

Dc
X

l=1
akBk,lek,l(k, x(jk))

!2

(80)

=
1
2D

ns
X

k=1

dk
X

jk=1

 Dc
X

l=1
(
S
√Dc
−akBk,l)ek,l(k, x(jk))

!2

.
(81)

In the second line, we used Eq. (16) that ek,l(I ̸= k, x) = 0 and the orthogonality of gk (Eq. (3)). In
the last line, we used Eq. (16) that gk = Dc
−1/2 P

l ek,l. We can find the gradient descent equation
of Bk,l from Eq. (81):

dBk,l

dt
= −η

dk
X

j=1

1
D

"

akek,l(k, x(j))

Dc
X

l′=1
(akBk,l′ −
S
√Dc
)ek,l′(k, x(j))

#

,
(82)

which in the matrix form is

d ⃗Bk

dt
= −ηak

D ΦΦT
 

Bkak −
⃗S
√Dc

!

,
(83)

where Dc dimensional vectors ⃗Bk and ⃗S are [Bk,1, · · · , Bk,Dc] and [S, · · · , S] respectively. It
illustrates that dBk

dt is contained in im(Φ), which is contained in im(U) (immediate from Φ = USV ).

As UU T (Uz) = U(U T U)z = Uz, UU T acts as identity on image of U, showing that UU T d ⃗Bk

dt =

d ⃗Bk

dt .

■

F.2
Conserved quantity of extended multilinear model

Lemma 3. In the setup of Lemma 2, a2
k −| ⃗Bk|2 is conserved over time.

Proof We can use Eq. (81) to find the equation for ak:

dak

dt = −η

dk
X

j=1

1
D

"X

l=1
Bk,lek,l(k, x(j))

Dc
X

l′=1
(akBk,l′ −
S
√Dc
)ek,l′(k, x(j))

#

,
(84)

26


---Page Break---
which in the matrix form is

dak

dt = −η

D
⃗BT
k ΦΦT
 
⃗Bkak −
⃗S
√Dc

!

.
(85)

Then

ak
dak

dt = −ηak

D
⃗BT
k ΦΦT
 
⃗Bkak −
⃗S
√Dc

!

(86)

= ⃗BT
k
d ⃗Bk

dt ,
(87)

where we used Eq. (83) in the last line. Thus, a2
k −| ⃗Bk|2 is conserved during the dynamics.
■

F.3
Dc shot learner

Proposition 1. Let the setup be as that in Lemma 2. Suppose that ak(T) is eventually bounded away
from zero, i.e. there exists δ > 0 and M > 0 such that T > M ⇒|ak(T)| ≥δ. Also assume that
U ⊥-component of ⃗Bk(0)ak(0) and ⃗Bk(0)S is negligible. Then the skill strength Rk is

Rk(∞) =

(
dk < Dc :
S

1 −
p

1 −dk/Dc


dk ≥Dc :
S
(88)

Proof First, we show that dLk

dt ≤0 with equality only holding when the gradient is 0.

dLk

dt
= dLk

dak

dak

dt +

Dc
X

i

dLk
dBk,i

dBk,i

dt
(89)

= −η dk

D

 
dLk

dak

dLk

dak
+

Dc
X

i

dLk
dBk,i

dLk
dBk,i

!

≤0.
(90)

The equality holds only when

dLk

dak
= dak

dt = 0
and
dLk
dBk,i
= dBk,i

dt
= 0 .
(91)

We show that both ak and ⃗Bk are bounded throughout whole dynamics. As

Lk =

Φ

 
⃗Bkak −
⃗S
√Dc

!

2

≥σ2
UU T
 
⃗Bkak −
⃗S
√Dc

!

2

(92)

for σ2 the smallest nonzero eigenvalue of ΦΦT , where Φ = USV . This shows that

UU T
 
⃗Bkak −
⃗S
√Dc

!

(93)

is bounded, so UU T ⃗Bkak is bounded. Meanwhile, in Lemma 2, we showed that (1−UU T ) d ⃗Bk

dt = 0,
so (1 −UU T ) ⃗Bkak is bounded. This shows that ⃗Bkak is bounded. As a2
k −| ⃗Bk|2 is constant
(Lemma 3) and | ⃗Bkak| = |ak|| ⃗Bk| is bounded, this shows that both ak and | ⃗Bk| are bounded.

The dynamics moving in some bounded region always has at least one accumulation point, which
we denote as p. We will show that dLk

dt = 0 at p. The function Lk(t) in t is a decreasing differential

function which is positive. We also note that d2Lk(t)

dt2
is globally bounded, as it can be expressed in
polynomial expression in (ak, ⃗Bk) and we showed that (ak(t), ⃗Bk(t)) is bounded. From Taylor’s
theorem, one can obtain

inf Lk(t) ≤Lk(t1 + t2) ≤Lk(t1) + t2
dLk

dt (t1) + t2
2
2 M
(94)

27


---Page Break---
for M = sup | d2Lk(t)

dt2
|. Choosing t2 = −dLk

dt (t1)M −1 shows that

Lk(t1) −
1
2M

dLk

dt (t1)
2
≥inf Lk(t)
(95)

and letting t1 →∞here gives

lim
t1→∞
1
2M

dLk

dt (t1)
2
≤lim
t1→∞(Lk(t1) −inf Lk(t)) = 0
(96)

so dLk

dt →0 as t →∞. Meanwhile, as p is accumulation point of (ak, Bk), dLk

dt (p) is accumulation
point of dLk

dt (ak(t), ⃗Bk(t)). As limt→∞
dLk

dt (t) = 0, the only accumulation point of dLk

dt (t) is zero,
which shows that dLk

dt (p) = 0.

We have seen that a2
k−| ⃗Bk|2 and (I −UU T ) ⃗Bk are conserved in our dynamics. A quantity conserved
in dynamics should also be conserved at p, so p = (a, ⃗B) should satisfy the following conditions:

• a2 −| ⃗B|2 = ak(0)2 −| ⃗Bk(0)|2 (Lemma 3);

• (I −UU T ) ⃗B = (I −UU T ) ⃗Bk(0) (Lemma 2);

•
dLk

dt (a, ⃗B) = 0, or equivalently the gradient is 0 at p.

We will solve for p satisfying those three conditions. The third condition is equivalent to that

aUU T
 
⃗Ba −
⃗S
√Dc

!

= 0.
(97)

As ak(T) is eventually bounded away from zero, we have a ̸= 0, so

UU T
 
⃗Ba −
⃗S
√Dc

!

= 0.
(98)

It follows that

⃗B = UU T ⃗B + (I −UU T ) ⃗B = UU T
⃗S
√Dc
a−1 + (I −UU T ) ⃗Bk(0)
(99)

and substituting to first condition gives

a2 −1

a2

UU T
⃗S
√Dc



2

−
(I −UU T ) ⃗Bk(0)

2
= ak(0)2 −| ⃗Bk(0)|2.
(100)

This is equivalent to a quadratic equation in a2, and has a following solution of

a2 =

v
u
u
t

UU T
⃗S
√Dc



2

+ (ak(0)2 −|UU T ⃗Bk(0)|2)2

4
+ ak(0)2 −|UU T ⃗Bk(0)|2

2
.
(101)

This shows that there are two candidates for p, with a given as two square roots of Eq. (101) and B
determined from a by Eq. (99). It is impossible for Lk(t) to have accumulation points both in regions
a > 0 and a < 0, as it would imply ak(t) = 0 happens infinitely many often, contradicting that ak
is eventually bounded away from zero. Thus it follows that Lk(t) can only have one accumulation
point. As dynamics having unique accumulation point should converge, it follows that

(a, ⃗B) = (ak(∞), ⃗Bk(∞)).
(102)

One can check that the U ⊥-component of ⃗Bk(∞)ak(∞) is given as

(I −UU T ) ⃗Bk(∞)ak(∞) = (I −UU T ) ⃗Bk(0)ak(0)
(103)

and this is bounded by |(1 −UU T )Bk(0)|(S + ak(0)), so by our assumption this is negligible.
Thus, we find that ⃗Bk(∞)ak(∞) is the pseudo-inverse solution, which is also found by the linear

28


---Page Break---
model with ek,l as basis functions. We can calculate Lk(∞) using the result from kernel (linear)
regression [23–27] (for a summary, see tables 1 and 2 in appendix A of [27]). Using the terminology
in table 1 of [27], the sample size is dk; the number of parameters is Dc; ridge and noise are absent;
the eigenfunctions are [ek,1, · · · , ek,Dc]; the eigen coefficients are EX[ek,i(x)Sgk(x)] = SD−1/2
c
(Eq. (16)); eigenvalues are uniform; the learnability is dk/Dc for all i; and the overfitting coefficient
is (1 −dk/Dc)−1. Taking into account that we have halved the MSE loss (Eq. (5)), the test loss is

Lk(∞) = S2

2


1 −dk

Dc


.
(104)

We obtain the result by using Eq. (10).
■

F.4
Nc basis functions for a skill

Proposition 2. Let the extended multilinear model Eq. (18) be trained with gradient flow on D →∞
i.i.d samples for the setup in Section 3 with ns →∞(input distribution: Eq. (1), target function:
Eq. (4), and MSE loss: Eq. (5), initialization: that of Proposition 1). For a model with the following
finite N basis functions
[e1,1, · · · , e1,Nc, e2,1, · · · , eq,r],
(105)
where quotient q = ⌊(N −1)/Nc⌋+ 1 and remainder r is such that (q −1)Nc + r = N. The skill
strength at T →∞becomes

Rk(∞) =






k > q :
0
k = q :
S r

Nc
k < q :
S.
(106)

Proof Because we have D →∞and because [ek,1, · · · ek,Nc] can express gk (Eq. (20)), it is trivial
to show that Rk(∞) = S for k < q. For k = q, the gradient descent dynamics (Eq. (83)) leads to

d ⃗Bk

dt
= −ηak

D ΦΦT
 
⃗Bkak −
⃗S
√Nc

!

(107)

where the matrix Φ ∈Rr×dk and vector ⃗Bk ∈Rr are the feature matrix(Eq. (77)) and parameters for
the kth skill respectively. As D →∞, the matrix ΦΦT becomes a rank r identity matrix scaled by
the frequency of the skill:

lim
D→∞
1
D(ΦΦT )ll′ = EI,X [ek,l(k, X)ek,l′(k, X)] = P(k)δl,l′.
(108)

Plugging in ΦΦT ,

dBk,l

dt
= −ηP(k)ak


Bk,lak −
S
√Nc


.
(109)

Assuming the initialization in Proposition 1, we can show that ak(∞)Bk,l(∞) = S/√Nc for l ≤r.
From Eq. (7), the skill strength Rk(∞) is

Rk(∞) =

r
X

l=1

S
√Nc
EX [ek,l(k, X)gk(k, X)]
(110)

= S r

Nc
,
(111)

where we used Eq. (20) for the linear correlation between ek,l and gk.
■

29


---Page Break---
G
Time emergence example in NN

In this section, we discuss an example for the time emergence case (Fig. 1(a)) in which the saturation
of skill in an NN consists of multiple saturating ‘modes’ as in Fig. 8.

(a) neuron modes for a parity function

0
250
500
750
1000
T

0.0

0.2

0.4

0.6

0.8

1.0

R/S

mode 1
mode 2
mode 3
NN
MulLin

(b) mode/skill strength

Figure 8: Modes in NN. A 2-layer MLP with ReLU activations with a width of 3 and weight sharing
(Eq. (114)) is trained to fit the parity function. (a): The skill strength R, because of the last layer’s
linearity, can be decomposed into skill strength from each hidden neuron or each ‘mode’ (shown in
different colors, Eq. (119)). (b): The skill strength for each mode follows a near-sigmoidal curve
with different emergent/saturation times (colors) whose sum results in the total skill strength (solid
black). Note that different saturation times of each mode result in a deviation from the prediction of
the multilinear model with B2 = 1/3 (dashed black).

Task.
We assume an input X ∈R3×8 (note that we are not using X as a random variable) that is
all 8 possible inputs for bits with dimension 3. The target Y is the parity function scaled by S.

X =
  0 0 0 0 1 1 1 1
0 0 1 1 0 0 1 1
0 1 0 1 0 1 0 1

,
Y =
 
S −S −S S −S S S −S 
(112)

NN.
We assume a 2-layer width 3 NN with ReLU activation with the input dimension 3 (Fig. 8(a)).
The NN has 16 parameters, but to simplify the argument, we use weight sharing so NN has only 4
parameters:
f(x; α, β, γ, c) = wT σ(Wx + b) + c
(113)

where σ is the ReLU activation and W, b, w are

W =
  −α
α
−α
−β
β
−β
γ
−γ
γ

,
b =
  0
β
−γ

,
w =
  −2α
β
γ

.
(114)

Modes.
It is easy to see that α = β = γ =
√

2S and c = −S leads to the target parity function.
We note that one parameter except c (i.e. α, β, γ) maps to one neuron or a mode (colors in Fig. 8(a)).
We define the first mode f (1) as

f (1)(x) = w1σ(W T
1 x + b1) = −2α2σ(x2 −x1 −x3)
(115)

= −2α2h1(x),
h1(x) := σ(x2 −x1 −x3),
(116)

where w1, b1 are the first entry of w, b respectively and W1 is the first row of W. Note that f (1)(x)
takes a form similar to the multilinear model (Eq. (9)) but with h1 as the respective basis. We define
f (2), f (3) similarly, and the sum of modes becomes the NN:

f(x) =

3
X

q=1
f (i)(x) + c,
(117)

which resembles the multilinear model with different skills.

30


---Page Break---
Mode strength.
Analogous to the skill strength in Eq. (7), we define mode q’s strength R(q) as

R(q) =
1
8S2 Y T f (q)(X),
(118)

where f (q)(X) = [f (q)(X1), · · · , f (q)(X8)] and Xj are the jth column of X. By the linearity of the
expectation,

R =

3
X

q=1
R(q).
(119)

Note that constant c always has zero correlation (inner product) to the target (Y ).

Analysis.
The dynamics of each mode R(q)(x) differs from that of the multilinear model (Eq. (11))
because hq(x) often depends on the parameter, and the dynamics are no longer decoupled among
each mode. Nevertheless, each mode follows a sigmoid-like growth (Fig. 8(b)). We note that each
mode has a different saturation time scale or is updated at different frequencies. A mode with a longer
time scale leads to a longer ‘tail’ of saturation as discussed in the main text.

Update frequency.
Because of the non-linearity, each mode differs in the gradients it receives. We
can explicitly calculate the gradient for each parameter as:

dα2

dt = 2ηα2(−S −(−2α2 + 2β2 + c))
(120)

dβ2

dt = −ηβ2(S −(−2α2 + 5β2 + 5c))
(121)

dγ2

dt = −ηγ2(S −(γ2 + c))
(122)

dc
dt = −η(2α2 −5β2 −γ2 −8c).
(123)

We immediately notice that c will grow the fastest for small initialization (α, β, γ, c ≪1) because it
saturates exponentially while other parameters saturate sigmoidally. Considering that S is always
the largest term and c saturate to S quickly, we notice that the saturation is in the order of α2
(≈2S + 2c ≈4S), β2 (≈−S + 5c ≈4S), and γ2(≈2S). We observe that our crude approximation
holds in Fig. 8(b): the first (α) and the second (β) modes saturate at similar timescale, while the third
mode (γ) requires approximately twice the time for saturation.

31


---Page Break---
H
Details of the multilinear model

The multilinear model (Fig. 9(a)) has two identifying properties: 1) the layerwise structure and 2) gk
as the basis functions. In this section, we discuss the role of each property in more detail.

...

(a) Multilinear model illustration

0
25
50
75
100
T

0.0

0.2

0.4

0.6

0.8

1.0

Rk/S

d1/D = 0.5
d2/D = 0.2
d3/D = 0.1

(b) Decoupled dynamics

Figure 9: Multilinear model. (a): An illustration of the multilinear model which is multilinear in
terms of parameters, generating a layerwise structure. The model also has the skill functions gks as
basis functions. (b): The dynamics of the multilinear model are decoupled and each skill strength
(Rk) shows a sigmoidal growth in time. Note that less frequent skills have a more delayed growth.

Multilinearity.
The product of two parameters (akbk) creates the layerwise structure (Fig. 9(a))
that gives rise to the emerging dynamics (sudden saturation or sigmoidal growth) in Fig. 9(b). The
time emergence of NN is well-described by the sigmoidal dynamics (Fig. 1(a)); a non-sigmoidal
saturation dynamics, for example, that of linear models (Fig. 10(a)), would inadequately describe the
time emergence. Such dynamics have first been studied by Saxe et al. [19] (See Appendix B.2 for an
overview).

Assuming a sufficiently fast decay of dk for the skills, the sigmoidal growth results in a stage-like
training (Appendix D) where one skill fully saturates before the next skill emerges. In Appendix D,
we discuss how the stage-like training can describe the quanta model [18] and how NNs, without
explicit gks, decouple each skill.

Finally, note that even though sigmoidal saturation has a resemblance to the test accuracy in grokking
[48], our model is irrelevant to grokking because Rk – which is defined over the expectation over the
kth skill (Eq. (7)) – appears both in the empirical loss (Eq. (11)) and the test loss: failing to describe
the discrepancy between train and test accuracy in grokking.

Connection to linear models.
In Section 4 and Appendix E, we have shown how the scaling laws
follow from the basis functions gk that decouples the loss. To analyze the role of gk, we can ask
whether a simpler linear model with gk as basis functions (Eq. (124)) also recovers the scaling laws.
The answer is yes and we outline how a linear model can recover all scaling laws. In addition, we
also outline how extended linear models – extended similar to Section 5 such that skills are decoupled
– can recover all emergence behaviors shown in Appendix F except the time emergence.

By replacing akbk with wk, we obtain the linear model with skill basis functions:

fT (i, x; w) =

N
X

k=1
wk(T)gk(i, x).
(124)

The dynamics of the linear model under gradient flow is

Rk(T) = wk(T) = S(1 −e−η
dk

D T ),
(125)

where we assumed wk(0) = 0. The linear model follows an exponential saturation of the skill
strength in contrast to the sigmoidal saturation of the multilinear model (Fig. 10).

32


---Page Break---
0
50
100
T

0.0

0.5

1.0

Rk/S

(a) Linear

0
50
100
T

0.0

0.5

1.0

Rk/S

(b) Multi Linear

d1/D = 0.5
d2/D = 0.2
d3/D = 0.1

Figure 10: Dynamics of linear and multilinear model. (a): Skill strength dynamics of the linear
model (Eq. (125)) (b): Skill strength dynamics of the multilinear model (Eq. (11)). For the linear
model, Rk emerges from T = 0 for all dk/D > 0: obstructing the stage-like training. For the
multilinear model, Rk shows a delayed emergence depending on dk/D: allowing the stage-like
training and describing the sigmoidal time emergence in Fig. 1(a).

Nevertheless, the linear model Eq. (125) results in the same scaling laws in Section 4. For the time
scaling law, we recover the relationship between dLk/dT and dLk/dk in Appendix E.1 because
Rk(T) is a function of dk

D T only (where dk/D = Ps(k) for D →∞). For the data scaling law,
we recover Corollary 1 because each wk (i.e. Rk) is decoupled. For the parameter scaling law, we
recover Corollary 2 trivially as the linear model shares the same basis functions.

The data and parameter emergence in Section 5 can be obtained from the linear model in Eq. (124) if
we extend the model analogous to Eqs. (15) and (18). For example, we can extend the model for data
emergence as

fT (i, x; W) =

N
X

k=1

Dc
X

l=1
Wk,l(T)ek,l(i, x),
(126)

where the matrix W ∈RN×Dc is an extension of w ∈RN in Eq. (124), Dc is a fixed scalar, and
ek,l(i, x) : {0, 1}ns+nb →R are functions with the following properties:

EX|I=k [ek,lek,l′] = δll′,
ek,l(I ̸= k, x) = 0,

Dc
X

l=1

1
√Dc
ek,l = gk.
(127)

The equivalence can be shown by Lemma 2 which states that the multilinear model finds the minimum
norm solution: the solution that the linear model finds in a ridgeless regression setup.

Thus, for our setup, the basis functions play a critical role in the scaling laws and data/parameter
emergences. The choice of basis functions, also known as the task-model alignment (see [23, 27]),
determines the linear model’s scaling laws and emergence behaviors. See Bordelon et al. [21] for a
study of the scaling laws in linear models.

33


---Page Break---
I
Additional plots and tables

0
1000
2000
T

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength Rk/S

(a) Time calibration

1000
2000
D

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength Rk/S

(b) Data calibration

0
5
10
N

0.0

0.2

0.4

0.6

0.8

1.0

Skill strength Rk/S

(c) Parameter calibration

NN
extended model
NN
extended model

Figure 11: Calibration and prediction on emergence. The calibration of the extended multilinear
model (solid) on the 2-layer NN (dashed) for ns = 1 system. For the calibrated parameters, we have
B2 = 1/22 for time (Eq. (14)), Dc = 800 for data (Eq. (17)), and Nc = 4 for hidden layer width
(Eq. (21)).

Table 5: Samples of skill strength Rk/S. The table shows the skill strength at N = 10 for 10
different runs of the parameter emergence experiment (Fig. 1(c)). Note that the variance of Rk/S
is amplified by the outliers – shaded columns – that learn a less frequent skill at the cost of a more
frequent skill (second column) or fail to learn a skill (seventh column).

k = 1
0.98
0.98
0.98
0.98
0.98
0.98
0.98
0.98
0.98
0.98
k = 2
4.5
0.95
0.95
0.95
0.96
0.96
0.04
0.96
0.96
0.95
k = 3
0.6
0.0
0.72
0.90
0.92
0.64
0.88
0.8
0.58
0.52
k = 4
0.0
0.78
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
k = 5
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0

Figure 12: Enlarged emergence. Enlarged view of skill emergence from Fig. 4, showing that
saturations also follow a sigmoidal pattern. The x-axis is measured in steps.

34


---Page Break---
J
Rigorous derivation of the scaling laws

In Appendix E, we discussed the scaling laws in simplified settings, favoring intuition over mathe-
matical rigor. Building upon the intuitive understanding developed in Appendix E, we now turn our
attention to a rigorous analysis of the scaling laws. In this section, we will derive general scaling
laws by considering a comprehensive set of parameters and variables. Our goal is to establish the
conditions under which these scaling laws hold and to quantify the associated error terms. By
explicitly analyzing the error terms, this section aims to provide a rigorous assessment of the validity
and limitations of our scaling law estimates.

Table 6: Scaling laws and their conditions. The leftmost column indicates the condition for the
‘large resource’ – large enough to be treated as infinity, while the second column is the condition
between the other two resources for the scaling law (third column). The last two columns show where
the statement for the prefactor constant (e.g. AN for scaling law L = ANN −α) and the scaling law
(with the assumptions and explicit error terms) are given. Note that whenever T appears in theorems
and corollaries, ηS is multiplied to make it dimensionless.

Large resource
Condition
Scaling law
Constant
Statement

D ≫T 3
N α+1 = o(T)
L = ANN −α
Theorem 1
Theorem 1
D ≫NT 2, T 3
N α+1 ≫T
L = AT T −α/(α+1)
Theorem 4
Theorems 2 and 3
D ≫T 3
N α+1 ≈T
L = ACC−α/(α+2)
Corollary 5
Corollary 4
T ≫D(log D)1+ϵ
N α+1 = o(D)
L = ANN −α
Theorem 5
Theorem 5
T ≫D(log D)1+ϵ
N α+1 ≫D
L = ADD−α/(α+1)
Theorem 5
Theorem 5

J.1
General set up, repeated

We go back to the most general settings possible. Our starting point is Eq. (27), which describes the
dynamics of Rk and Lk valid for k ≤N:

Lk =
S2

2

1 +

S
Rk(0) −1
−1
e2η
dk

D ST
2
(27)

We do not use skills for indices k > N in our model, but we can still denote

Rk = 0
and
Lk = S2

2 .
(128)

For Ps(k) = Ak−α−1, the total loss is given as

L =

ns
X

k=1
Ps(k)Lk =

N
X

k=1
Ps(k)Lk +

ns
X

k=N+1
Ps(k)S2

2 .
(129)

When ns, N, T are all set, their dependency with the data is only determined by the statistics dk,
the number of data with i(j) = k. We assumed that (i, x) ∈I × {0, 1}nd was collected as random
samples with i following the Zipfian distribution of size ns and exponent α + 1, or equivalently
P(i = k) = Ps(k) = Ak−α−1 for 1 ≤k ≤ns. Then (d1, · · · , dns) is a vector denoting the number
of occurrences in D independent sampling from that distribution. It follows that di follows binomial
distribution B(D, Ps(k)).

In this complete perspective, our loss is dependent on all of those parameters and variables

L = L(nS, D, Rinit, N, T)
(130)

where Rinit = (R1(0), · · · , RN(0)) denotes the vector representing initial condition. We will also
simply denote rk = Rk(0). We will not assume much on rk, but we absolutely need 0 < rk < S for
dynamics to hold, and we also should have

ns
X

k=1
Ps(k)r2
k = E[f(0)2] ≪S2.
(131)

35


---Page Break---
We will not impose any particular distribution on Rinit. Instead, we will try to identify sufficient
conditions on rk for our desired result to hold, and those conditions will differ by the situation we are
considering. For example, in Theorems 2 and 3 where we prove time scaling law L = Θ(T −α/(α+1))
for large enough D and bottleneck T, we only require ϵ < rk < S/2 for some ϵ > 0. However, the
exact constant depends on the distribution of rk, and figuring out the explicit constant seems to be
only feasible when we fix rk = r as in Theorem 4.

J.2
Estimates for large D

We will first consider the situation where D becomes the ‘large resource’ so that its effect on the
loss function is negligible. The number of data dk follows binomial distribution B(D, Ps(k)), so
dk/D converges to Ps(k) for large enough D. So taking the limit of L when we let D →∞has the
effect of replacing dk/D by Ps(k) in the expression of L. We will establish an explicit inequality
comparing the difference between L and this limit.
Lemma 4. For a function F : R →R with its total variation V (F) bounded, we have
ED


F(dk

D )

−Ez∼N(Ps(k),Ps(k)(1−Ps(k))/D) [F(z)]
 <
V (F)
√

D
p

Ps(k)(1 −Ps(k))
(132)

where N(µ, σ2) denotes normal distribution of mean µ and variance σ2.

Proof This is just an application of the Berry-Esseen inequality (with constant 1, see [49] for modern
treatment) applied to dk following binomial distribution B(D, Ps(k)).
■
Lemma 5. Let F : R →R be a C2 function such that F ′′ is bounded. Then we have
Ez∼N(Ps(k),Ps(k)(1−Ps(k))/D) [F(z)] −F(Ps(k))
 ≤Ps(k)(1 −Ps(k))

2D
sup|F ′′|.
(133)

Proof First, we apply Taylor’s theorem to show that

|F(z) −F(Ps(k)) −F ′(Ps(k))(z −Ps(k))| ≤(z −Ps(k))2

2
sup |F ′′|.
(134)

Taking expectation when z follows normal distribution N(Ps(k), Ps(k)(1−Ps(k))

D
) gives

|Ez [F(z) −F(Ps(k))]| = |Ez [F(z) −F(Ps(k)) −F ′(Ps(k))(z −Ps(k))]|
(135)

≤Ez [|F(z) −F(Ps(k)) −F ′(Ps(k))(z −Ps(k))|]
(136)

≤Ez

(z −Ps(k))2

2
sup |F ′′|

(137)

=Ps(k)(1 −Ps(k))

2D
sup|F ′′|.
(138)

■
Proposition 3. We have


ED [Lk] −
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2



<
2αS2
p

DPs(k)
+ 4S4η2T 2Ps(k)

D
.
(139)

Proof Consider the function F : R →R given as

F(z) =
S2

2

1 +

S
rk −1
−1
e2ηST z
2 .
(140)

This function is monotone decreasing and C2 on the whole domain, and its supremum and infimum
are given as

sup F =
lim
z→−∞F(z) = S2

2
and
inf F = lim
z→∞F(z) = 0.
(141)

36


---Page Break---
This implies that

V (F) = sup F −inf F = S2

2 .
(142)

Also, we will show that F ′′ is globally bounded. We first calculate

F ′′(z) = −4S3rk(1 −rk

S )2η2T 2 e2ηST z(1 −rk

S −2rk

S e2ηST z)
 
1 −rk

S + rk

S e2ηST z4
.
(143)

We consider the following inequalities

e2ηST z ≤S

rk


1 −rk

S + rk

S e2ηST z
(144)
1 −rk

S −2rk

S e2ηST z
 ≤
1 −rk

S

 + 2rk

S e2ηST z < 2

1 + rk

S (e2ηST z −1)

(145)

to show that

|F ′′(z)| < 4S3rk(1 −rk

S )2η2T 2
2S

rk
 
1 −rk

S + rk

S e2ηST z2

 
1 −rk

S + rk

S e2ηST z4
< 8S4η2T 2
(146)

for all z. Thus we can apply both Lemma 4 and Lemma 5 to this function F and we have
ED


F(dk

D )

−F(Ps(k))
 <
V (F)
√

D
p

Ps(k)(1 −Ps(k))
+ Ps(k)(1 −Ps(k))

2D
sup|F ′′|

<
S2

2
√

D
p

Ps(k)(1 −Ps(k))
+ 4Ps(k)S4η2T 2

D

<
2αS2
p

DPs(k)
+ 4Ps(k)S4η2T 2

D
(147)

where the last line follows from that we always have

1 −Ps(k) ≥1 −Ps(1) =
2−(α+1) + · · · + n−(α+1)
s
1 + 2−(α+1) + · · · + n−(α+1)
s
>
2−(α+1)

1 + 2−(α+1) >
1
22(α+1) .
(148)

■
Lemma 6. For any integer N and σ ≥1/2 and σ ̸= 1, we have

N
X

k=1
k−σ = ζ(σ) + N 1−σ

1 −σ + O(N −σ)
(149)

where ζ is the Riemann zeta function (defined over the whole complex plane except 1 via analytic
continuation). In addition,
N
X

k=1
k−1 = log N + γ + O(N −1)
(150)

where γ = 0.5772156649... is Euler’s constant.

Proof See Corollary 1.15 of [50], or other analytic number theory textbooks.
■
Proposition 4. (Large D approximation) We have

ED[L] −

N
X

k=1
Ps(k)
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2 −

ns
X

k=N+1
Ps(k)S2

2
(151)

=O

S2D−1/2fα(N) + S4η2T 2D−1
(152)

where

fα(N) =






1
if α > 1
log N
if α = 1
N (1−α)/2
if α < 1.
(153)

The constant on the O term only depends on α.

37


---Page Break---
Proof From the description of L in Eq. (129), we have

ED[L] −

N
X

k=1
Ps(k)
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2 −

ns
X

k=N+1
Ps(k)S2

2
(154)

=

N
X

k=1
Ps(k)






ED[Lk] −
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2






.
(155)

We apply Proposition 3 to give

N
X

k=1
Ps(k)






ED[Lk] −
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2






<

N
X

k=1
Ps(k)

 
2αS2
p

DPs(k)
+ 4S4η2T 2Ps(k)

D

!

.

(156)
Each of these sum involving Ps(k) is bounded as

N
X

k=1
Ps(k)2 <

 N
X

k=1
Ps(k)

!2

< 1
(157)

and
N
X

k=1

p

Ps(k) <

N
X

k=1
k−(α+1)/2 = O(fα(N))
(158)

which follows from Lemma 6. Combining those two gives

N
X

k=1
Ps(k)

 
2αS2
p

DPs(k)
+ S4η2T 2Ps(k)

D

!

= O

S2D−1/2fα(N) + S4η2T 2D−1
.
(159)

■

While Proposition 4 holds for any D, it becomes only meaningful if the resulting error terms are less
than the main term we desire. We will revisit this when the exact main term is found, and determine
the sufficient size of D for error terms to become small enough.

J.3
Estimates for not too small ns

We next discuss the effect of ns. When ns →∞heuristically, then intuitively we have Ps(k) →
k−(α+1)/ζ(α + 1). We will discuss the difference between when we regard ns as ∞and when we
do not.

Proposition 5. The following equations hold:

A−1 =

ns
X

k=1
k−(α+1) = ζ(α + 1) −n−α
s
α
+ O(n−α−1
s
)
(160)

Ps(k) =
k−α−1

ζ(α + 1)


1 +
n−α
s
αζ(α + 1)O(n−α−1
s
)

(161)

ns
X

k=N+1
Ps(k) = N −α −n−α
s
αζ(α + 1) + O(N −min(α+1,2α))
(162)

All implied constants on O only depend on α.

38


---Page Break---
Proof The first statement Eq. (160) follows from substituting σ = α + 1 in Lemma 6. As Ps(k) =
Ak−(α+1), the second statement Eq. (161) immediately follows. If we substitute ns = N into
Eq. (160) and calculate differences between them, we obtain

ns
X

k=N+1
k−α−1 = N −α −n−α
s
α
+ O(N −α−1).
(163)

Thus we have

ns
X

k=N+1
Ps(k) = A

ns
X

k=N+1
k−(α+1) = N −α −n−α
s
αζ(α + 1) + O
 
N −α−1 + (N −α −n−α
s
)n−α
s

. (164)

Regardless of the size of ns, We always have

(N −α −n−α
s
)n−α
s
≤
N −α

2

2
= N −2α

4
(165)

so the third statement Eq. (162) follows.
■

We go back to the description of total loss given in Eq. (129) as

L =

N
X

k=1
Ps(k)Lk +

ns
X

k=N+1
Ps(k)S2

2
(129)

and we take its expectation in D. Proposition 4 suggests that its limit when D →∞is given as

lim
D→∞ED[L] =

N
X

k=1
Ps(k)
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2 +

ns
X

k=N+1
Ps(k)S2

2 .
(166)

Denote

L1 =

N
X

k=1
Ps(k)
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2
(167)

L2 =

ns
X

k=N+1
Ps(k)S2

2 .
(168)

We discuss the effect of ns in L1 and L2, by comparing limit of L1 and L2 when ns →∞and
their original values.

• For the term L1, the change of letting ns as finite value from ns →∞has effect of
multiplying T by 1+n−α
s
/(αζ(α+1)), and multiplying whole L1 by 1+n−α
s
/(αζ(α+1)).
It can be equivalently put as

L1(ns, N, T) =

1 +
n−α
s
αζ(α + 1) + O(n−α−1
s
)

L1


∞, N, T

1 +
n−α
s
αζ(α + 1) + O(n−α−1
s
)

.

(169)
We always have ns > N and N →∞eventually, so if dependency of L1 with respect
to T is at most polynomial order, then change of main term of L1 is negligible. We can’t
establish exact statements yet without the descriptions of size of L1.
• The term L2 only depends on N and ns, not on T. Applying Proposition 5 (especially
Eq. (162)) gives

L2(ns, N, T) = N −α −n−α
s
αζ(α + 1)
S2

2 + O(N −min(α+1,2α)S2)
(170)

When ns grows faster than N then n−α
s
part is totally negligible, and when ns has same
order as N then n−α
s
affects the constant for main term of L2. Things might get little
complicated when ns = N + o(N), where N −α −n−α
s
= o(N −α) can happen then.

39


---Page Break---
• Comparing size of L1 and L2 mainly depends on time. The term L2 is fixed, and L1
decreases as T increases. For T = ∞we have L1 = 0, so L2 having order N −α dominates
(this proves scaling law for N of exponent α), so restriction on ns becomes quite substantial.
For small T and large N where the size of L2 is small, we can expect the restriction on ns
to be less substantial. For example, in the extreme case N = ∞, we have L2 = 0, and ns
does not matter at all (except that, of course, it should satisfy ns ≥N).

For such reasons, it is hard to quantify exact conditions for ns such that error terms are controlled,
unless we specify relative growth of (N, T). However, ns = ω(N) suffices to assure that setting
ns = ∞has zero effect on the main term. We will not worry about ns in this setting anymore too,
and come back to this at the very end to determine enough ns.

J.4
Estimating main terms

We assume D = ∞and ns = ∞– virtually implying that dk/D = Ps(k) and Ps(k) =
k−α−1/ζ(α + 1) (calculated by rule of ns = ∞). We decomposed our main term into

lim
ns→∞lim
D→∞ED[L] = L1 + L2
(171)

where

L1 =

N
X

k=1
Ps(k)
S2

2

1 +

S
rk −1
−1
e2ηPs(k)ST
2
(172)

and

L2 =

∞
X

k=N+1
Ps(k)S2

2 .
(173)

By Proposition 5, L2 is determined almost completely as

L2 =
S2N −α

2αζ(α + 1) + O(N −α−1).
(174)

Now focus on L1. For

F(z) =
S2

2

1 +

S
rk −1
−1
e2ηST z
2
(175)

(note: it really depends on rk so it is correct to write Fk, but for convenience we will keep using F.)
one can express L1 as

L1 =

N
X

k=1
Ps(k)F(Ps(k)).
(176)

Lemma 7. Let F(z) be defined as Eq. (175).

1. (Estimate for large z) We have

0 ≤F(z) ≤(S −rk)2

2
min

1, S2

r2
k
e−4ηST z

.
(177)

2. (Estimate for small z) For z ≥0, we have

(S −rk)2

2
−8ηS3T

27
z ≤F(z) ≤(S −rk)2

2
.
(178)

Proof

1. The left side is obvious. For the right side, F(z) ≤(S −rk)2/2 follows from noting that
F(0) = (S−rk)2

2
and proving F ′(z) ≤0, and F(z) ≤(S−rk)2

2
S2

r2
k e−4ηST z follows from

just replacing 1 +

S
rk −1
−1
e2ηST z in the denominator of F by

S
rk −1
−1
e2ηST z.

40


---Page Break---
2. For the left side, it suffices to show −F ′(z) ≤8ηS3T

27
. One can calculate

F ′(z) = −2S2rk(1 −rk

S )2ηT
e2ηST z
 
1 + rk

S (e2ηST z −1)
3
(179)

and

F ′′(z) = −4S3rk(1 −rk

S )2η2T 2 e2ηST z(1 −rk

S −2rk

S e2ηST z)
 
1 + rk

S (e2ηST z −1)
4
(180)

so F has unique inflection point at

1 −rk

S −2rk

S e2ηST z = 0
⇒
e2ηST Z = 1

2

 S

rk
−1

(181)

and this point is where −F ′(z) obtains maximum. Substituting this to the expression of
F ′(z) gives −F ′(z) = 8ηS3T

27
.

■

Our threshold for distinguishing two approximation methods will be set as z = z0 = (ζ(α +
1)ηST)−1, where both two error terms are bounded by O(S2). The constant ζ(α + 1) is set to make
later calculations much easier. Applying Lemma 7 gives

L1 =

N
X

k=1
Ps(k)F(Ps(k))
(182)

=
X

1≤k≤N,Ps(k)<z0

(S −rk)2

2
Ps(k)

+ O



ηS3T
X

1≤k≤N,Ps(k)<z0
Ps(k)2 + S2
X

1≤k≤N,Ps(k)>z0
Ps(k)min

1, S2

r2
k
e−4ηST Ps(k)


.

(183)

Denote

M =
X

1≤k≤N,Ps(k)<z0

(S −rk)2

2
Ps(k)
(184)

E1 = ηS3T
X

1≤k≤N,Ps(k)<z0
Ps(k)2
(185)

E2 = S2
X

1≤k≤N,Ps(k)>z0
Ps(k)min

1, S2

r2
k
e−4ηST Ps(k)

.
(186)

Proposition 6. Suppose that there exists 0 < r <
√

S such that r ≤rk < S/2 for all k. In the
decomposition of
lim
ns→∞lim
D→∞ED[L] = M + L2 + O(E1 + E2)
(187)

given as above, we have the following bound.

1. If (ηST)1/(α+1) > N, then

L2 =
S2N −α

2αζ(α + 1) + O(S2N −α−1)
(188)

M = E1 = 0
(189)

E2 = O

S2(log(S/r))α/(α+1)(ηST)−α/(α+1)
(190)

41


---Page Break---
2. If (ηST)1/(α+1) < N, then

L2 + M = Θ



S2
X

k>(ηST )1/(α+1)
Ps(k)



= Θ(S2(ηST)−α/(α+1))
(191)

E1 = O

S2(ηST)−α/(α+1)
(192)

E2 = O

S2(log(S/r))α/(α+1)(ηST)−α/(α+1)
(193)

Here all constants in O and Θ terms are absolute with respect to η, S, T, N. (They may depend on
α.)

Proof We first note that the condition Ps(k) < z0 = (ζ(α + 1)ηST)−1 is equivalent to

Ps(k) < z0 = (ζ(α + 1)ηST)−1 ⇔k−α−1 <
1
ηST ⇔k > (ηST)1/(α+1).
(194)

Thus we can rephrase the descriptions of terms as

M =
X

(ηST )1/(α+1)<k≤N

(S −rk)2

2
Ps(k)
(195)

E1 = ηS3T
X

(ηST )1/(α+1)<k≤N
Ps(k)2
(196)

E2 = S2
X

k≤min((ηST )1/(α+1),N)
Ps(k)min

1, S2

r2
k
e−4ηST Ps(k)

.
(197)

Applying Proposition 5 easily shows that

L2 =
S2N −α

2αζ(α + 1) + O(S2N −α−1).
(198)

For M and E1, we will consider them by dividing two cases depending on whether (ηST)1/(α+1) >
N or (ηST)1/(α+1) < N. If (ηST)1/(α+1) > N, then the condition (ηST)1/(α+1) < k ≤N is
never satisfied, so M = E1 = 0. Now suppose (ηST)1/(α+1) < N. We first note that

L2 + M =
X

(ηST )1/(α+1)<k≤N

(S −rk)2

2
Ps(k) +
X

k>N

S2

2 Ps(k).
(199)

As (S −rk)2 = Θ(S2), we can let

L2 + M = Θ



S2
X

k>(ηST )1/(α+1)
Ps(k)




(200)

and using Proposition 5 gives the desired estimate L2 + M = Θ(S2(ηST)−α/(α+1)). For E1,
estimating sum of Ps(k)2 using Lemma 6 gives

E1 = O



ηS3T
X

k>(ηST )1/(α+1)
k−2(α+1)



= O

S2(ηST)−α/(α+1)
.
(201)

For E2 we always have

E2 ≤S2
X

k≤(ηST )1/(α+1)
Ps(k)min

1, S2

r2 e−4ηST Ps(k)

(202)

regardless of the size of N, so it suffices to bound this sum. If we denote l = (ηST)1/(α+1) and
define

F2(z) = min

1, S2

r2 e−4ηST z

,
(203)

42


---Page Break---
it suffices to show the bound
X

k≤l
Ps(k)F2(Ps(k)) = O

(log(S/r))α/(α+1)(ηST)−α/(α+1)
.
(204)

We will approximate this sum as
X

k≤l
Ps(k)F2(Ps(k)) =
X

k≤l
(Ps(k) −Ps(k + 1))
Ps(k)
Ps(k + 1) −Ps(k)F2(Ps(k))
(205)

=
X

k≤l
(Ps(k) −Ps(k + 1))
k−α−1

(α + 1)k−α−2(1 + O(k−1))F2(Ps(k))
(206)

=O



X

k≤l
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)F2(Ps(k))



.
(207)

to obtain the form of Riemann sum approximation for the integral of
Z ∞

z=Ps(l)
z−1/(α+1)F2(z)dz
(208)

at Ps(l) < Ps(l −1) < · · · < Ps(1). As F2(z) is decreasing function, this Riemann sum is always
less than the integral, so we obtain

X

k≤l
Ps(k)F2(Ps(k)) = O

 Z ∞

z=Ps(l)
z−1/(α+1)F2(z)dz

!

.
(209)

We note that Ps(l) = (ζ(α + 1)ηST)−1. The threshold for F2(z) to become 1 is given at

S2

r2 e−4ηST z = 1
⇔
z =
1
2ηST log S

r .
(210)

As r <
√

S, this value is always greater than Ps(l). Thus we can divide our integral as
Z ∞

(ζ(α+1)ηST )−1 z−1/(α+1)F2(z)dz
(211)

=
Z (2ηST )−1 log(S/r)

(ζ(α+1)ηST )−1
z−1/(α+1)dz +
Z ∞

(2ηST )−1 log(S/r)
z−1/(α+1) S2

r2 e−4ηST zdz.
(212)

The first part is bounded by
Z (2ηST )−1 log(S/r)

(ζ(α+1)ηST )−1
z−1/(α+1)dz = O
 
(2ηST)−1 log(S/r)
α/(α+1)
(213)

which can be shown to be O

(log(S/r))α/(α+1)(ηST)−α/(α+1)
. For the second part, we apply
substitution of w = 4ηSTz to show
Z ∞

(2ηST )−1 log(S/r)
z−1/(α+1) S2

r2 e−4ηST zdz =S2

r2 (4ηST)−α/(α+1)
Z ∞

2 log(S/r)
w−1/(α+1)e−wdw

(214)

=S2

r2 (4ηST)−α/(α+1)Γ

α
α + 1, 2 log S

r


(215)

and applying the asymptotic Γ(s, x) = O(xs−1e−x) suggests that this is bounded by

≪S2

r2 (4ηST)−α/(α+1)

log S

r

−1/(α+1)
e−2 log(S/r) = O

(ηST)−α/(α+1)
.
(216)

■

43


---Page Break---
Theorem 1. (Parameter scaling law) Assume the following conditions: ns > N with lim(N/ns) =
γ < 1 (γ can be zero), and there exists 0 < r <
√

S such that r < Rk(0) < S/2 for all k. If
N, T →∞while satisfying N α+1 = o(T), the expected loss ED[L] for all datasets D of size D
satisfies

ED[L] = S2(1 −γα)

2αζ(α + 1)N −α

+ O

S2N −min(α+1,2α) + S2 (log(S/r))α/(α+1) (ηST)−α/(α+1)

+ O

S2D−1/2fα(N) + S4η2T 2D−1
,
(217)

where

fα(N) =






1
if α > 1
log N
if α = 1
N (1−α)/2
if α < 1.
(218)

The constant on the O term only depends on α. When D ≫T 3, then all the error terms involving D
are negligible.

Proof In the situation ns = ∞and D = ∞, Proposition 6 shows that

ED[L] =
S2

2αζ(α + 1)N −α + O

S2N −(α+1) + S2 (log(S/r))α/(α+1) (ηST)−α/(α+1)
. (219)

We consider the effect of ns first. As L1 becomes an error term in this estimation, letting ns as a
finite value has no effect on overall estimation. The term L2 accounts for the main term, and letting
ns as finite value changes it to
N −α −n−α
s
αζ(α + 1)
S2

2 + O(N −min(α+1,2α)S2).
(220)

This accounts for the factor (1 −γα) on the main term and O(N −min(α+1,2α)S2) added to the
error term. The effect of D is exactly described in Proposition 4, contributing the error term of
O
 
S2D−1/2fα(N) + S4η2T 2D−1
. Regarding the sufficient condition for D, if D ≫T 3 then we
have
S4ηT 2D−1 ≪T −α/(α+1),
S2D−1/2fα(N) ≪T −3/2N 1/2 ≪T −1
(221)
so all error terms involving D are less than O(T −α/(α+1)).
■

For the situation T = O(N α+1) however, the error terms E1 and E2 are of same size, so we can only
say that the main term is of O(S2(ηST)−α/(α+1)).
Theorem 2. (Upper bound for the time scaling law) Assume the following conditions: ns > N, and
there exists there exists 0 < r <
√

S such that r < Rk(0) < S/2 for all k. If N, T →∞while
satisfying ηST = O(N α+1), the expected loss ED[L] is

ED[L] = O

S2 (log(S/r))α/(α+1) (ηST)−α/(α+1) + S2D−1/2fα(N) + S4η2T 2D−1
(222)

with constant on O only depending on α and lim sup((ηST)1/(α+1)/N), with fα defined as in
Theorem 1. If D ≫NT 2 and D ≫T 3, then all the error terms involving D are negligible.

Proof The error term regarding D can be obtained in the same way as Theorem 1, so we will let
D = ∞for the rest of the proof. Also, we can let ns = ∞, as we observed that it contributes at most
to the constant factor of the upper bound and does not change the scaling.

In the decomposition of Proposition 6, we always have

E2 = O

S2 (log(S/r))α/(α+1) (ηST)−α/(α+1)
(223)

and
E1 = O

S2(ηST)−α/(α+1)
(224)

holding regardless of N, so it only remains to consider L2 + M . If (ηST)1/(α+1) < N, then
L2 + M is of size O
 
S2(ηST)−α/(α+1)
. If (ηST)1/(α+1) ≥N, then N and (ηST)1/(α+1) has
same order, so L2 + M = L2 = Θ(S2N −α) is O
 
S2(ηST)−α/(α+1)
. Thus in either cases we
have the desired bound.
■

44


---Page Break---
Theorem 3. (Lower bound for the time scaling law) Assume the following conditions: ns > N and
0 < Rk(0) < S/2. If N, T →∞while satisfying (8ζ(α + 1)−1ηST)1/(α+1) < N, the expected
loss ED[L] is

ED[L] ≥κS2(ηST)−α/(α+1) + O

η−1ST −1 + S2D−1/2fα(N) + S4η2T 2D−1
(225)

for κ and constant on O only depending on α, with fα defined as in Theorem 1. If D ≫NT 2 and
D ≫T 3, then all the error terms involving D are negligible.

Proof The error term regarding D can be obtained in the same way as Theorem 1, so we will let
D = ∞for the rest of the proof. We only show the lower bound for L1, holding regardless of N and
ns. In Lemma 7 (Eq. (178)) we have

F(z) ≥(S −rk)2

2
−8ηS3T

27
z ≥S2

8 −8ηS3T

27
z
(226)

for z ≥0, so if z ≤(4ηST)−1 then F(z) ≥S2/8 −2S2/27 > S2/20.
The condi-
tion Ps(k) ≤(4ηST)−1 is equivalent to that k ≥(4ζ(α + 1)−1ηST)1/(α+1). In evaluating
L1 = PN
k=1 Ps(k)F(Ps(k)), we will only add over k in range of

(4ζ(α + 1)−1ηST)1/(α+1) < k < (8ζ(α + 1)−1ηST)1/(α+1).
(227)
From the assumption, this interval sits inside 1 < k < N. For such k we use upper bound of
F(Ps(k)) > S2/20. Then by using Proposition 5 we can obtain

L1 ≥S2

20

X

(4ζ(α+1)−1ηST )1/(α+1)<k<(8ζ(α+1)−1ηST )1/(α+1)
Ps(k)
(228)

= S2

20

(ζ(α + 1)−1ηST)−α/(α+1)

αζ(α + 1)
(4−α/(α+1) −8−α/(α+1)) + O
 
(ηST)−1
.
(229)

The possible effect of ns on the main term is to multiply both the main term by and T by (1 + n−α
s
),
so it increases the bound.
■

The condition (8ζ(α + 1)−1ηST)1/(α+1) < N is not absolutely necessary for lower bound. The
condition (ηST)1/(α+1) = Θ(N) and ns ≥2N would suffice and one can formulate a similar
theorem, although the constant of lower bound might be much smaller if (ηST)1/(α+1)/N is small.

Lastly, we provide a simpler version of those results combined and discuss the special case where the
optimal compute C = NT, or the given engineering budget, is specified.
Corollary 3. (Summary of the large data estimation) Assuming D ≫NT 2, T 3 and ns ≫N 1+ϵ
such that effects of ns and D are negligible, then for N, T →∞we have

ED[L] = Θη,S,r

max(N −α, T −α/(α+1))

,
(230)

where Θη,S,r denotes that the implied constant depends on η, S, α and r = min Rk(0) > 0. In
particular, we have
N α+1 = O(T)
⇒
ED[L] = Θη,S,r(N −α)
(231)
and
T = O(N α+1)
⇒
ED[L] = Θη,S,r(T −α/(α+1)).
(232)

Proof Apply Theorem 1 if N α+1 = o(T) and Theorem 2 and Theorem 3 if N α+1 ≫T.
■
Corollary 4. (The ‘computationally optimal’ case) Denote C = NT and assume the conditions in
Corollary 3. Then we have
ED[L] ≫C−α/(α+2).
(233)
When N = Θ(C1/(α+2)) and T = Θ(C(α+1)/(α+2)), we achieve ED[L] = Θ(C−α/(α+2)). (Its
implied constant may depend on implied constant for growth of N and T.)

Proof The first part follows from

ED[L] ≫max(N −α, T −α/(α+1))
(234)
and
max(N −α, T −α/(α+1)) ≥(N −α)1/(α+2)(T −α/(α+1))(α+1)/(α+2) = (NT)−α/(α+2).
(235)
The second part can be checked by substituting (N, T) = (C1/(α+2), C(α+1)/(α+2)) (or their
constant multiples) to Corollary 3.
■

45


---Page Break---
J.5
Computing the constant for time scaling law

While we have found the time scaling law E[L] = O(T −α/(α+1)) holding for T = O(N α+1),
bounds in Theorem 2 and Theorem 3 were chosen rather lazily and do not depict the correct picture.
We will find the constant using a more refined estimation, but we require additional assumptions
on parameters. We will focus on the setting where D and ns are large enough to be negligible,
Rk(0) = r is fixed, and T = O(N α+1) with fixed constant such that time scaling law holds.
Theorem 4. (Constant for time scaling law) Denote L∞as the loss when D, ns →∞so that their
effect is negligible:

L∞= L∞(T, N) =

N
X

k=1
Ps(k)
S2

2

1 +
  S

r −1
−1 e2ηPs(k)ST
2 +
S2N −α

2αζ(α + 1).
(236)

When T, N →∞and lim N/(ηST)1/(α+1) = λ for a fixed constant λ ∈(0, ∞], the following limit
exists:
A(λ) =
lim
T,N→∞(ηST)α/(α+1)L∞(T, N).
(237)

The prefactor constant A as the a function of λ (when λ = ∞then let λ−α = λ−(α+1) = 0) is

A(λ) = ζ(α + 1)−1/(α+1)

α + 1

Z ∞

λ−(α+1)/ζ(α+1)
u−1/(α+1)ΦS,r(u)du +
S2

2αζ(α + 1)λ−α,
(238)

where

ΦS,r(u) =
S2

2

1 +
  S

r −1
−1 e2u
2 .
(239)

Proof We first observe

L∞=

N
X

k=1
Ps(k)ΦS,r(ηSTPs(k)) +
S2N −α

αζ(α + 1).
(240)

We will seek to convert it into Riemann sum form of certain integral. We start by noting that

Ps(k) = (Ps(k) −Ps(k + 1))
k
α + 1(1 + O(k−1))
(241)

= ζ(α + 1)−1/(α+1)

α + 1
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)(1 + O(k−1))
(242)

Denote uk = ηSTPs(k), then the sum can be approximated to
X

k
Ps(k)ΦS,r(ηSTPs(k))
(243)

≈
X

k
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)ΦS,r(ηSTPs(k))
(244)

=(ηST)−α/(α+1) X

k
(uk −uk+1)u−1/(α+1)
k
ΦS,r(uk)
(245)

if we ignore small k. As ΦS,r is decreasing, this corresponds to Riemann sum taking minimum in the
interval [uk+1, uk]. So integral provides an upper bound for this sum. Similarly, we can approximate
it with Riemann sum taking maximum in [uk, uk−1] if we use

Ps(k) = ζ(α + 1)−1/(α+1)

α + 1
(Ps(k −1) −Ps(k))Ps(k −1)−1/(α+1)(1 + O(k−1))
(246)

instead. As ΦS,r shows exponential decay, we can ignore values at small k, so this shows

(ηST)−α/(α+1) X

k
(uk −uk+1)u−1/(α+1)
k
ΦS,r(uk) ≈
Z ∞

uN
u−1/(α+1)ΦS,r(u)du
(247)

46


---Page Break---
and from that
uN = ηSTN −(α+1)ζ(α + 1)−1 = λ−(α+1)ζ(α + 1)−1
(248)
we obtain our desired result.
■

Theorem 4 basically tells that for N = λ(ηST)1/(α+1) and D, ns large enough, we have

L ∼A(λ)(ηST)−α/(α+1)
(249)
with A(λ) given as Eq. (238), thus specifying the constant for time scaling law. For finite λ, this
theorem covers the computationally optimal case of (N, T) = (λ1C1/(α+2), λ2C(α+1)/(α+2)) for
some nonzero constant λ1, λ2. For λ = ∞, it describes the case T = o(N α+1) where effect of N is
negligible.

Corollary 5. Denote L∞as L∞as the loss when D, ns →∞same as Eq. (236). Denote C = NT
and suppose that
(N, ηST) = (λ(ηSC)1/(α+2), λ−1(ηSC)(α+1)/(α+2))
(250)
for a fixed constant 0 < λ < ∞. Then as C →∞, we have

L∞= A

λ(α+2)/(α+1)
λα/(α+1) (ηSC)−α/(α+2) (1 + o(1))
(251)

where A is given as Eq. (238) of Theorem 4.

Proof As lim N/(ηST)1/(α+1) = λ(α+2)/(α+1) under above conditions, we can apply Theorem 4
and substituting Eq. (250) into Eq. (249) gives the desired result.
■

Technically we can optimize L∞for a given fixed value of C = NT by letting λ as argument of
minimum of A
 
λ(α+2)/(α+1)
λ−α/(α+1), although it seems almost impossible to obtain any form
of formula for such λ.

Lastly, we provide the following estimate for the time scale constant (A(λ)) when r is small,
especially the first term in Eq. (238).
Proposition 7. As r →0, we have (Λ > 0 fixed)
Z ∞

Λ
u−1/(α+1)ΦS,r(u)du ≈

log S −r

r

α/(α+1) 21/(α+1)S2(α + 1)

4α
.
(252)

Proof Denote M = ( S

r −1), and replace u by (log M)v. Then we have
Z ∞

Λ
u−1/(α+1)ΦS,r(u)du = (log M)α/(α+1) S2

2

Z ∞

Λ/ log M

v−1/(α+1)dv

(1 + M 2v−1)2
(253)

= (log M)α/(α+1) S2

2

Z ∞

0
1v≥Λ/ log M
v−1/(α+1)dv

(1 + M 2v−1)2 .
(254)

As M →∞, the integrand converges to

lim
M→∞1v≥Λ/ log M
v−1/(α+1)dv

(1 + M 2v−1)2 =
v−1/(α+1)
if v ≤1/2
0
if v > 1/2.
(255)

The integrand is bounded by v−1/(α+1) if v ≤1/2 and v−1/(α+1)e−2(2v−1) if v > 1/2, those of
which are all integrable. So we can apply Lebesgue’s dominated convergence theorem to show

lim
M→∞

Z ∞

Λ/ log M

v−1/(α+1)dv

(1 + M 2v−1)2 =
Z ∞

0

 

lim
M→∞1v≥Λ/ log M
v−1/(α+1)dv

(1 + M 2v−1)2

!

(256)

=
Z 1/2

0
v−1/(α+1)dv.
(257)

Thus we have

lim
r→0


log S −r

r

−α/(α+1) Z ∞

Λ
u−1/(α+1)ΦS,r(u)du = S2

2

Z 1/2

0
v−1/(α+1)dv
(258)

= 21/(α+1)S2(α + 1)

4α
(259)

which can be observed to be equivalent to the desired expression of Eq. (252).
■

47


---Page Break---
J.6
Estimates for large T and threshold between data/parameter scaling

The estimates for small D require different techniques from estimates for large D. We will consider
the situation T grows much faster than D and N, and discuss when data scaling law of L =
Θ(D−α/(α+1)) happens. We will consider a simpler setting of ’ns = ∞’ or equivalently that effects
of ns are negligible (ns = ω(N) seems to suffice) and Rk(0) = r < S is fixed, although it won’t be
impossible to discuss their subtle effects.

First we single out effect of T by comparing L(T) and L(∞). We remind

Lk(T) =
S2

2

1 +
  S

r −1
−1 e2ηdkST/D
2
(27)

and its limit when T →∞is given as

Lk(∞) = lim
T →∞Lk(T) =

(
(S−r)2

2
if dk = 0
0
if dk > 0.
(260)

Proposition 8. Suppose that Rk(0) = r < S is fixed. For large T, we have

ED[L(T)] −ED[L(∞)] = O

S4r−2De−4ηST/D
.
(261)

Proof As Lk(T) is decreasing in T, we always have Lk(T) ≥Lk(∞) so therefore

ED[L(T)] −ED[L(∞)] ≥0.
(262)

So we only need to establish an upper bound for Lk(T) −Lk(∞). We note that Lk(T) −Lk(∞)
when dk = 0, so one can write

Lk(T) −Lk(∞) = 1dk>0Lk(T)
(263)

where 1dk>0 denotes the characteristic function

1dk>0 =
1
if dk > 0
0
if dk = 0.
(264)

We use simple bound of

Lk(T) <
S2

2
  S

r −1
−1 e2ηdkST/D
2 < S4

2 r−2e−4ηdkST/D.
(265)

As dk follows binomial distribution B(D, Ps(k)), considering its moment generating function gives

Edk[e−4ηdkST/D] =

1 −Ps(k) + Ps(k)e−4ηST/DD
(266)

so thus

Edk[1dk>0e−4ηdkST/D] =

1 −Ps(k) + Ps(k)e−4ηST/DD
−(1 −Ps(k))D.
(267)

Meanwhile, for 0 ≤u, v ≤1 real numbers, we have

|uD −vD| = |u −v||uD−1 + uD−2v + · · · + vD−1| ≤D|u −v|
(268)

so, applying this inequality to above gives

Edk[1dk>0e−4ηdkST/D] ≤DPs(k)e−4ηST/D.
(269)

Thus, we can deduce

Edk[Lk(T)] −Edk[Lk(∞)] = Edk[1dk>0Lk(T)]
(270)

< S4r−2

2
Edk[1dk>0e−4ηdkST/D]
(271)

≤S4r−2

2
De−4ηST/DPs(k)
(272)

48


---Page Break---
and thus

0 ≤ED[L(T)] −ED[L(∞)] < S4r−2

2
De−4ηST/D
∞
X

k=1
Ps(k)2 = O

S4r−2De−4ηST/D
.

(273)

■

This provides an almost complete account for the effect of very large T. We will let T = ∞from
this point. We have

ED[L(∞)] = (S −r)2

2

N
X

k=1
Ps(k)(1 −Ps(k))D + S2

2

∞
X

k=N+1
Ps(k).
(274)

Applying Lemma 6 gives

∞
X

k=N+1
Ps(k) =
N −α

αζ(α + 1) + O(N −α−1)
(275)

so it suffices to focus on the first sum. We will divide the range of k into two 1 ≤k ≤M and
M < k ≤N. For the sum over 1 ≤k ≤M, we will apply the following simple bound (in the last
part, we used 1 −x ≤e−x)

0 ≤

M
X

k=1
Ps(k)(1 −Ps(k))D ≤(1 −Ps(M))D ≤e−Ps(M)D.
(276)

For the sum over M < k ≤N, we will approximate the sum into some integral, which happens to be
incomplete gamma function.
Proposition 9. For 2 < M < N integers, we have

N
X

k=M+1
Ps(k)(1 −Ps(k))D
(277)

=D−α/(α+1) ζ(α + 1)−1/(α+1)

α + 1


Γ

α
α + 1, DPs(N)

−Γ

α
α + 1, DPs(M)

(278)

+ O

D−(2α+1)/(α+1) + D−α/(α+1)M −1
.
(279)

Here Γ denotes the incomplete gamma function

Γ (s, x) =
Z ∞

x
ys−1e−ydy.
(280)

Proof Consider the interval [Ps(N), Ps(M)] and its partition P = {Ps(N) < Ps(N −1) < · · · <
Ps(M)}. For a function f(x) = x−1/(α+1)(1 −x)D, we will consider its upper and lower Darboux
sums with respect to P. As f is decreasing in (0, 1], its upper and lower Darboux sums are given
respectively as

U(f, P) =

N−1
X

k=M
(Ps(k) −Ps(k + 1))Ps(k + 1)−1/(α+1)(1 −Ps(k + 1))D
(281)

L(f, P) =

N−1
X

k=M
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)(1 −Ps(k))D.
(282)

and those give bound of the integral of f as

L(f, P) ≤
Z Ps(M)

Ps(N)
f(x)dx ≤U(f, P).
(283)

Meanwhile, by noting that

Ps(k) = ζ(α + 1)−1/(α+1)

α + 1
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)(1 + O(k−1))
(284)

49


---Page Break---
one can show

N
X

k=M
Ps(k)(1 −Ps(k))D
(285)

=ζ(α + 1)−1/(α+1)

α + 1

 N−1
X

k=M
(Ps(k) −Ps(k + 1))Ps(k)−1/(α+1)(1 −Ps(k))D
!

(1 + O(M −1))

(286)

=ζ(α + 1)−1/(α+1)

α + 1
L(f, P)(1 + O(M −1)).
(287)

Applying a similar argument for upper Darboux sum gives

N
X

k=M
Ps(k)(1 −Ps(k))D = ζ(α + 1)−1/(α+1)

α + 1
U(f, P)(1 + O(M −1))
(288)

and from Eq. (283) it follows

N
X

k=M
Ps(k)(1−Ps(k))D = ζ(α + 1)−1/(α+1)

α + 1

 Z Ps(M)

Ps(N)
x−1/(α+1)(1 −x)Ddx

!

(1+O(M −1)).

(289)
From now we will estimate the integral
Z Ps(M)

Ps(N)
x−1/(α+1)(1 −x)Ddx.
(290)

We replace x = y/D in the integral inside, then it becomes
Z Ps(M)

Ps(N)
x−1/(α+1)(1 −x)Ddx = D−α/(α+1)
Z DPs(M)

DPs(N)
y−1/(α+1) 
1 −y

D

D
dy.
(291)

We want to approximate
 
1 −y

D
D by e−y, so we will estimate difference between them. We have

D log(1 −y/D) = −y −

∞
X

k=2

yk

kDk−1
(292)

so if D > 2y then

−y > D log(1 −y/D) = −y −1

D

∞
X

k=2

yk

kDk−2 > −y −1

D

∞
X

k=2

yk

2(2y)k−2 = −y −y2

D
(293)

so

e−y

1 −y2

D


< e−ye−y2/D <

1 −y

D

D
< e−y,
(294)

where we used the inequality 1 −x ≤e−x. As Ps(M) < 1/2 if M > 2 (obvious from Ps(M) <
(Ps(1) + Ps(2))/2 < 1/2), any y in the interval [DPs(N), DPs(M)] satisfies D > 2y. So, we can
apply this approximation in every y. It follows that
Z DPs(M)

DPs(N)
y−1/(α+1) 
1 −y

D

D
dy
(295)

=
Z DPs(M)

DPs(N)
y−1/(α+1)e−ydy + O

 Z DPs(M)

DPs(N)
y−1/(α+1)e−y y2

D dy

!

(296)

=
Z DPs(M)

DPs(N)
y−1/(α+1)e−ydy + O

D−1
Z ∞

0
y−1/(α+1)e−yy2dy

(297)

=Γ

α
α + 1, DPs(N)

−Γ

α
α + 1, DPs(M)

+ O(D−1).
(298)

Combining this with Eq. (289) and Eq. (291) gives the desired result.
■

We combine Proposition 8 and Proposition 9 together to obtain this final estimation result.

50


---Page Break---
Theorem 5. (Scaling laws for large time estimation) Suppose that N, D →∞and ns ≫N 1+ϵ for
some ϵ > 0 so that effect of ns is negligible. Suppose that Rk(0) = r for all 1 ≤k ≤N.

1. (Parameter scaling law) If N = o(D1/(α+1)), then we have

ED[L] =
S2

2αζ(α + 1)N −α + O

S2D−α/(α+1) + S2N −α−1 + S4r−2De−4ηST/D
.

(299)

2. (Data scaling law) If D = O(N α+1) and µ = lim(D/N α+1) exists (it can be zero), then

ED[L] = D−α/(α+1)
(S −r)2ζ(α + 1)−1/(α+1)

2(α + 1)
Γ

α
α + 1,
D
N α+1ζ(α + 1)


+ S2(D/N α+1)α/(α+1)

2αζ(α + 1)



+ O

S2D−(2α+1)/(2α+2) + S4r−2De−4ηST/D
(300)

Here Γ denotes the incomplete gamma function

Γ (s, x) =
Z ∞

x
ys−1e−ydy.
(301)

In particular, if D = o(N α+1) such that µ = 0, we have

ED[L] = D−α/(α+1) (S −r)2ζ(α + 1)−1/(α+1)

2(α + 1)
Γ

α
α + 1


(1 + o(1))

+ O

S4r−2De−4ηST/D
.
(302)

In either case, T ≫D(log D)1+ϵ for some ϵ > 0 implies that error terms involving T are negligible.

Proof Proposition 8 states

ED[L(T)] −ED[L(∞)] = O

S4r−2De−4ηST/D
(261)

and we showed

ED[L(∞)] = (S −r)2

2

N
X

k=1
Ps(k)(1 −Ps(k))D + S2

2

∞
X

k=N+1
Ps(k)
(274)

and
∞
X

k=N+1
Ps(k) =
N −α

αζ(α + 1) + O(N −α−1).
(275)

For the sum of Ps(k)(1 −Ps(k))D over 1 ≤k ≤N, we use the estimate (see Eq. (276)) of

M
X

k=1
Ps(k)(1 −Ps(k))D = O

e−Ps(M)D
(303)

and the estimate of Proposition 9. Combining all those gives

ED[L]
(304)

=
S2N −α

2αζ(α + 1)
(305)

+D−α/(α+1) (S −r)2ζ(α + 1)−1/(α+1)

2(α + 1)


Γ

α
α + 1, DPs(N)

−Γ

α
α + 1, DPs(M)


(306)

+O

S2(D−(2α+1)/(α+1) + D−α/(α+1)M −1 + N −α−1 + e−Ps(M)D) + S4r−2e−4ηST/D
.

(307)

We will prove our main statement by choosing appropriate M depending on size comparison between
D and N.

51


---Page Break---
1. If N = o(D1/(α+1)), then we let M = 3, and also regard all incomplete gamma function
values as O(1). Then it follows

ED[L] =
S2N −α

2αζ(α + 1) + O

S2D−α/(α+1) + S2N −α−1 + S4r−2e−4ηST/D
(308)

and thus obtaining the parameter scaling law.
2. Suppose D = O(N α+1) and µ = lim(D/N α+1) exists. We want

D−α/(α+1) (S −r)2ζ(α + 1)−1/(α+1)

2(α + 1)
Γ

α
α + 1, DPs(N)

+
S2N −α

2αζ(α + 1)
(309)

to be our main term, and set M < N such that the term

S2D−α/(α+1)Γ

α
α + 1, DPs(M)

(310)

and error terms not depending on T given as

O

S2(D−(2α+1)/(α+1) + D−α/(α+1)M −1 + N −α−1 + e−Ps(M)D)

(311)

are all bounded by O(D−(2α+1)/(2α+2)).
Set M
= D1/(2α+2).
Then Ps(M) =
D−1/2/ζ(α + 1), so applying the asymptotic Γ(s, x) = O(xs−1e−x) gives

Γ

α
α + 1, DPs(M)

= O

D−1/2(α+1)e−
√

D/ζ(α+1)
.
(312)

This term and e−Ps(M)D
=
e−
√

D/ζ(α+1) are less than D−α/(α+1)M −1
=
O(D−(2α+1)/(2α+2)), and obviously D−(2α+1)/(α+1) is less than D−(2α+1)/(2α+2). Thus
it follows that

ED[L] = D−α/(α+1) (S −r)2ζ(α + 1)−1/(α+1)

2(α + 1)
Γ

α
α + 1,
D
N α+1ζ(α + 1)


+
S2N −α

2αζ(α + 1)

+ O

S2D−(2α+1)/(2α+2) + S4r−2De−4ηST/D
.
(313)

Regarding the final statement regarding sufficient condition for large T, T ≫D(log D)1+ϵ implies

De−4ηST/D < De−4ηS(log D)1+ϵ < D · D−4ηS(log D)ϵ ≪D−K
(314)

for any K > 0, showing that the error term O
 
S4r−2De−4ηST/D
is negligible compared to all
other error terms of Eq. (299) and Eq. (300).
■

We also provide a summary of all large time estimation results.
Corollary 6. (Summary of large time estimation) Assuming T ≫D(log D)1+ϵ and ns ≫N 1+ϵ
such that effects of ns and T are negligible, and Rk(0) = r for all 1 ≤k ≤N. Then for D, N →∞,
we have
ED[L] = Θη,S,r

max(N −α, D−α/(α+1))

,
(315)

where Θη,S,r denotes that the implied constant depends on η, S, r and α. In particular, we have

N α+1 = O(D)
⇒
ED[L] = Θη,S,r(N −α)
(316)

and
D = O(N α+1)
⇒
ED[L] = Θη,S,r(D−α/(α+1)).
(317)

Proof Just summarize the results of Theorem 5.
■

52


---Page Break---
K
Methods

In this section, we present the methods used in our experiments.

K.1
2-layer MLP

We trained a 2-layer fully connected neural network (MLP) with ReLU activations. All parameters
of the MLP were initialized with a Gaussian distribution with a standard deviation of 0.001. The
input dimension of the model was ns + nb = 5 + 32 where ns is the length of control bits (number
of skills) and nb is the length of the skill bits. Each skill has m = 3 mutually exclusive sparse bits
that are used to express the skill function. The target scale was S = 5. The model was trained with
SGD without momentum and no weight decay (the exception is the parameter emergence experiment
where Adam with learning rate 0.001 and weight decay of 5 × 10−5 was used to escape the local
minima).6 For the data emergence experiment, the learning rate was halved every 50, 000 step.

The skill strength Rk(T) (Eq. (7)) was measured using 20, 000 i.i.d samples from the kth skill.7 For
the time emergence, the skill strengths were measured every 50 steps, while for other experiments,
they were measured after training. To mimic the infinite parameter N →∞, we used the model of
width 1000 (for the hidden layer). To mimic the infinite time T →∞, we trained for 5 × 105 steps
(3 × 104 steps for time emergence) where each step had the batch size of 4000 (2000 for the data
emergence experiment). To mimic D →∞, we sampled new data points for every batch. The details
are given in the following table.

Name
Values

width
1000
learning rate
0.05
initialization standard deviation
0.01
activation
ReLU
batch size
4000
steps
500,000
target scale
5
number of skill bits
32
number of skills
5

K.2
Transformer

This section outlines the transformer architecture used in Fig. 4. Data is encoded as for the 2-layer
MLP, but with one-hot positional encoding appended to the data. We use a basic decoder transformer
with 1 block, an initial embedding layer with output dimension 512, and a final linear layer. For the
attention mechanism, we used 4 attention heads. For non-linearity, we used ReLU. A batch size of
5000 was used with a target scale S = 1 and default Pytorch initialization. The model was trained
with SGD with a learning rate of 5 × 10−5, weight decay of 10−5, and momentum with β = 0.9. At
every 100 steps, the skill strength Rk(T) (Eq. (7)) was measured using 20, 000 i.i.d samples from
the kth skill.

K.3
Measurement of skill strength

The skill strength Rk is a simple linear correlation between the learned function f – function
expressed by NN – and gk for Pb given I = k. We approximate the expectation over X by taking the
mean over 20, 000 i.i.d samples from Pb for the kth skill:

Rk = EX[f(k, X)gk(k, X)] ≈
1
20000

20000
X

j=1
f(k, x(j))gk(k, x(j)),
(318)

6We are free to choose any optimizer as long as it preserves the order in which the skills are learned.
Additionally, the parameter emergence experiment uses infinite data; we expect the same solution for Adam and
SGD.
7Note that except the data scaling law experiment, the training set size is infinite.

53


---Page Break---
where the notation x(j) denotes the jth sample.

K.4
Details of the scaling law experiment

For the loss of the model (solid lines) in Fig. 2, we used the analytic equation for the model (Eq. (12))
under suitable assumptions such as sufficiently large ns (Table 2). For the scaling laws (dotted lines)
in Fig. 2, we used the exponents from Appendix E or Appendix J and the prefactor constants from
Theorem 4 (time scaling law), Theorem 5 (data scaling law), and Theorem 1 (parameter scaling law).
For the hyperparameters of the simulation, we used ns = 105 such that ns is large compared to other
resources; S = 1 and Rk(0) = 0.01 such that S −Rk(0) ≈S; and η = 1.

Time scaling law.
The total loss as a function of T for D, N →∞(Fig. 2(a), solid) is

L = S2

2

ns
X

k=1
Ps(k)
1

1 +

S
Rk(0) −1
−1
e2ηPs(k)ST
2 ,
(319)

which follows by taking D →∞and N = ns on Eq. (12). The scaling law (Fig. 2(a), dotted) is

L = AT T −α/(α+1),
(320)

where the exponent is derived in Appendix E.1 or Theorems 2 and 3. The prefactor constant is

At = S2

2
ζ(α + 1)−1/(α+1)

(α + 1)(ηS)α/(α+1)

Z ∞

0

u−1/(α+1)

1 +
  S

r −1
−1 e2u
2 du,
(321)

which we obtained by taking D →∞on Eq. (238).

Data scaling law.
The total loss as a function of D when N, T →∞(Fig. 2(b), solid) is

ED [L] = S2

2

ns
X

k=1
(1 −Ps(k))D Ps(k),
(322)

which follows from Eq. (58). The scaling law (Fig. 2(b), dotted) is

L = ADD−α/(α+1),
(323)

where the exponent follows from Appendix E.2 or Theorem 5. The prefactor constant is

AD = S2

2
ζ(α + 1)−1/(α+1)

α + 1
Γ

α
α + 1


(324)

which we obtained by taking N →∞in Eq. (302).

Parameter scaling law.
The total loss as a function of N when T, D →∞(Fig. 2(c), solid) is

L = S2

2

ns
X

k=N+1
Ps(k),
(325)

which follows from taking T, D →∞on Eq. (12). The scaling law (Fig. 2(c), dotted) is

L = ANN −α,
(326)

where the exponent follows from Theorem 1. The prefactor constant is

AN = S2

2 ,
(327)

which we obtained by taking D, T →∞, N/ns →0, and ζ(α + 1) ≈α−1 in Eq. (217).

54


---Page Break---
Compute scaling law.
The total loss as a function of T and N for D →∞(Fig. 3, solid) is

L = S2

2

N
X

k=1
Ps(k)
1

1 +

S
Rk(0) −1
−1
e2ηPs(k)ST
2 +

ns
X

k=N+1
Ps(k),
(328)

which follows by taking D
→
∞in Eq. (12).
In Fig. 3, we plotted for N
∈
{10, 20, 50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 10000} and T ∈[1, 1000] as examples of dif-
ferent tradeoff between T and N for fixed C.

The scaling law (Fig. 3, dotted) is
L = AcC−α/(α+2),
(329)
where the exponent is derived in Appendix E.4 or Corollary 4. Using Corollary 5, the prefactor
constant is
Ac = A

λ(α+2)/(α+1)
λα/(α+1) (ηS)−α/(α+2)
(330)

where A : R →R is defined in Eq. (238). We used the minimum value of Ac for λ ∈(0, ∞].

K.5
Estimates of the compute use

On CPU, our emergence experiments on the 2-layer MLP (Fig. 1) take 2 ∼5 hours for a single run
of time emergence experiments and 20 ∼40 hours for a single run of other experiments depending
on the CPU. All experiments were repeated 10 times (except for parameter emergence where we
repeated the experiment 50 times). Each experiment requires memory of at most 5GB. The CPU
cluster in which we experimented contained the following CPUs: Intel(R) Core(TM) i5-7500, i7-
9700K, i7-8700; and Intel(R) Xeon(R) Silver 4214R, Gold 5220R, Silver 4310, Gold 6226R, E5-2650
v2, E5-2660 v3, E5-2640 v4, Gold 5120, Gold 6132. The transformer experiment (Fig. 4) takes
48 ∼72 hours for each run; we used an RTX4090 with 24GB RAM, with 1 CPU from the list above.

55


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The claims in the abstract and introduction accurately reflect the paper’s
contributions, as evidenced by the contribution list in the introduction section, which
references the sections presenting each result.
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
Justification: The paper discusses the limitations of the multilinear model in Section 5.5
and the general limitations regarding the assumptions about the framework in Section 6
(Discussion and Conclusion). These sections address the robustness of the results and the
scope of the claims made.
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

56


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: All theoretical results are accompanied by intuitive explanations in the main
text, with detailed derivations (Appendices C and E) and rigorous proofs (Appendix J) in
the supplemental material. In addition, an alternative derivation of the scaling laws via
stage-like training is given in Appendix D.
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
Justification: We have provided details – architecture, initialization, learning setup, and
measurements – of our experiments in Appendix K including the details of NN for emergence
experiment (Fig. 1) in Appendix K.1, the details of scaling law experiment (Figs. 2 and 3) in
Appendix K.4, the details of the transformer experiment (Fig. 4) in Appendix K.2.
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

57


---Page Break---
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We provide a link to our source code in the main text.

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

Justification: We specify the details in Appendix K.1 (2-layer NN) and Fig. 4 (transformer).

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

Justification: Yes, all our figures in the main text, which are not simulations (Figs. 1 and 4),
have 1-standard deviation error bars.

Guidelines:

• The answer NA means that the paper does not include experiments.

58


---Page Break---
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

Answer: [Yes]

Justification: We provide the compute resource details in Appendix K.5.

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

Justification: The research conducted in this paper adheres to the principles outlined in the
NeurIPS Code of Ethics.

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

59


---Page Break---
Justification: This research aims to deepen the fundamental understanding of emergence
phenomena and scaling laws in deep learning. As our theoretical and empirical investigations
are conducted in a carefully controlled, idealized environment, we do not anticipate any
immediate societal consequences arising directly from the findings of this particular study.
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
Justification: The research presented in this paper does not involve the release of data or
models that pose a risk for misuse.
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
Justification: This paper does not use any existing assets from other sources.
Guidelines:

• The answer NA means that the paper does not use existing assets.

60


---Page Break---
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
Justification: This paper does not involve any crowdsourcing experiments or research with
human subjects.
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
Justification: No IRB approvals or equivalent reviews were required.

61


---Page Break---
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

62


---Page Break---
