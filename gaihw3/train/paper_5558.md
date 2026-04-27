Don’t Compress Gradients in Random Reshuffling:
Compress Gradient Differences

Abdurakhmon Sadiev1,2∗, Grigory Malinovsky1, Eduard Gorbunov1,2,3,4†‡,
Igor Sokolov1, Ahmed Khaled5, Konstantin Burlachenko1, Peter Richtárik1

1King Abdullah University of Science and Technology, Saudi Arabia
2Moscow Institute of Physics and Technology, Russian Federation
3Mohamed bin Zayed University of Artificial Intelligence, UAE
4Mila, Université de Montréal, Canada
5Princeton University, USA

Abstract

Gradient compression is a popular technique for improving communication com-
plexity of stochastic first-order methods in distributed training of machine learning
models. However, the existing works consider only with-replacement sampling of
stochastic gradients. In contrast, it is well-known in practice and recently confirmed
in theory that stochastic methods based on without-replacement sampling, e.g.,
Random Reshuffling (RR) method, perform better than ones that sample the gradi-
ents with-replacement. In this work, we close this gap in the literature and provide
the first analysis of methods with gradient compression and without-replacement
sampling. We first develop a distributed variant of random reshuffling with gradient
compression (Q-RR), and show how to reduce the variance coming from gradient
quantization through the use of control iterates. Next, to have a better fit to Feder-
ated Learning applications, we incorporate local computation and propose a variant
of Q-RR called Q-NASTYA. Q-NASTYA uses local gradient steps and different local
and global stepsizes. Next, we show how to reduce compression variance in this
setting as well. Finally, we prove the convergence results for the proposed methods
and outline several settings in which they improve upon existing algorithms.

1
Introduction

Distributed learning plays a crucial role in the training of modern Deep Learning (DL) models since
distributed approaches are able to significantly reduce training time [Goyal et al., 2017, You et al.,
2019]. Moreover, distributed methods are mandatory for such applications as Federated learning
(FL) [Koneˇcný et al., 2016, McMahan et al., 2017], where multiple nodes connected over a network
collaborate on a learning task. Each node possesses its own dataset and cannot share this data with
other nodes or a central server. As a result, algorithms for federated learning often rely on local
computation and lack access to the entire dataset of training examples. Federated learning finds
applications in diverse fields, including language modeling for mobile keyboards [Liu et al., 2021],

∗Part of the work was done while A. Sadiev was a research intern at KAUST, working in the Optimization
and Machine Learning Lab of P. Richtárik.
†Part of the work was done when E. Gorbunov was a researcher at MIPT and Mila & UdeM and also a
visiting researcher at KAUST, in the Optimization and Machine Learning Lab of P. Richtárik.
‡Corresponding author, eduard.gorbunov@mbzuai.ac.ae

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
healthcare [Antunes et al., 2022], and wireless communications [Yang et al., 2022]. Its applications
extend to various other domains [Kairouz et al., 2019].

Distributed learning tasks are often solved through empirical-risk minimization (ERM), where the
m-th device contributes an empirical loss function fm(x) representing the average loss of model x
on its local dataset, and our goal is to then minimize the average loss over all the nodes:

min
x∈Rd

"

f(x)
def
=
1
M

M
X

m=1
fm(x)

#

,
(1)

where the function f represents the average loss. Every fm is an average of sample loss functions f i
m
each representing the loss of model x on the i-th datapoint on the m-th clients’ dataset: that is for
each m ∈{1, 2, . . . , M} we have

fm(x)
def
=
1
nm

nm
X

i=1
f i
m(x).

For simplicity we shall assume that the datasets on all clients are of equal size: n1 = n2 = . . . = nM,
though this assumption is only for convenience and our results easily extend to the case when clients
have datasets of unequal sizes. Thus our optimization problem is

min
x∈Rd

"

f(x) =
1
nM

M
X

m=1

n
X

i=1
f i
m(x)

#

.
(2)

Because d is often very large in practice, the dominant paradigm for solving (2) relies on first-order
(gradient) information. Federated learning algorithms have access to two key primitives: (a) local
computation, where for a given model x ∈Rd we can compute stochastic gradients ∇f i
m(x) locally
on client m, and (b) communication, where the different clients can exchange their gradients or
models with a central server.

1.1
Communication compression

In practice, communication is more expensive than local computation [Kairouz et al., 2019], and as
such one of the chief concerns of algorithms for distributed learning is communication efficiency.
Algorithms for distributed/federated learning have thus focused on achieving communication effi-
ciency, with one common ingredient being the use of gradient compression, where each client sends
a compressed or quantized version of their update instead of the full update vector, potentially saving
communication bandwidth by sending fewer bits over the network. There are many operators that can
be used for compressing the update vectors: stochastic quantization [Alistarh et al., 2017], random
sparsification [Wangni et al., 2018, Stich et al., 2018], and others [Tang et al., 2020]. In this work we
consider compression operators satisfying the following assumption:
Assumption 1. A compression operator is an operator Q : Rd →Rd such that for some ω > 0, the
relations

E [Q(x)] = x
and
E
h
∥Q(x) −x∥2i
≤ω∥x∥2
hold for x ∈Rd.

Unbiased compressors can reduce the number of bits clients communicate per round, but also increases
the variance of the stochastic gradients used slowing down overall convergence, see e.g. [Khirirat
et al., 2018, Theorem 5.2] and [Stich, 2020, Theorem 1]. By using control iterates, Mishchenko et al.
[2019b] developed DIANA—an algorithm that can reduce the variance due to gradient compression
with unbiased compression operators, and thus ensure fast convergence. DIANA has been extended
and analyzed in many settings [Horváth et al., 2019, Stich, 2020, Safaryan et al., 2021] and forms an
important tool in our arsenal for using gradient compression.

1.2
Random Reshuffling

Despite the importance of addressing the communication bottleneck, local computations also signifi-
cantly affect the training. For simplicity, consider the 1-node scenario. In this case, the update rule
of the standard work-horse method in stochastic optimization – stochastic gradient descent (SGD)

2


---Page Break---
Table 1: Summary of known and new complexity results for solving distributed finite-sum optimization problem (2). Column “Complexity"
indicates the number of communication rounds to find a solution with accuracy ε > 0. Column “RR?" shows whether an algorithm uses
Random Reshuffling, “C?" indicates whether a method applies the compression of gradients or difference between the gradients and also whether
methods for communication, “H?" means independence from the constant of data heterogeneity in the complexity, “CVX?" indicates whether
each loss on the i-th datapoint on the m-th client is convex, but not strongly convex. Notation: κ = Lmax/µ and eκ = Lmax/eµ are conditional
number of problem (2), where Lmax = Lipschitz constant, µ and eµ are the strong convexity constants of f and f i
m respectively; variances

at the solution point x⋆: σ2
∗=
1
Mn

M
P

m=1

n
P

i=1
∥∇f i
m(x⋆) −∇fm(x⋆)∥2 and σ2
∗,n =
1
n

n
P

i=1
∥∇f i(x⋆)∥2; heterogeneity constant

ζ2
⋆=
1
M

M
P

m=1
∥∇fm(x⋆)∥2. The results of this paper are highlighted in blue.

Method
Complexity
RR?
C?
H?
CVX?
SGD
[Gower et al., 2019]
κ +
σ2
⋆,n
µ2ε
✗
✗
✗(1)
✓

RR
[Mishchenko et al., 2020]
eκ + σ⋆,n

eµ

q

eκn

ε
✓
✗
✗(1)
✗

RR
[Mishchenko et al., 2020]
nκ + σ⋆,n

eµ
p κn

ε
✓
✗
✗(1)
✓

QSGD
[Gorbunov et al., 2020]
 
1 + ω

M

κ + ω

M
σ2
⋆+ζ2
⋆
µ2ε
+
σ2
⋆
Mµ2ε

(2)
✗
✓
✗
✓

Q-RR
Corollary 1

 
1 + ω

M

eκ + ω

M
σ2
⋆+ζ2
⋆
eµ2ε
+ σ⋆,n

eµ

q

eκn

ε
✓
✓
✗
✗

Q-RR
Corollary 6

 
n + ω

M

κ + ω

M
σ2
⋆+ζ2
⋆
µ2ε
+ ρ⋆

µ
p κn

ε

(3)
✓
✓
✗
✓

DIANA
[Mishchenko et al., 2019a]
 
1 + ω

M

κ + ω

M
σ2
⋆
µ2ε +
σ2
⋆
Mµ2ε
✗
✓
✓
✓

DIANA-RR

Corollary 2
n(1 + ω) +
 
1 + ω

M

eκ + σ⋆,n

eµ

q

eκn

ε
✓
✓
✓
✗

DIANA-RR

Corollary 8
n(1 + ω) +
 
n + ω

M

κ + σ⋆,n

µ
p κn

ε
✓
✓
✓
✓

(1) In the case of SGD, RR we use ✗in “H?" to show that the complexity of these methods is provided in the non-distributed setup.
(2) The following inequality is useful for the comparison of complexities: σ2
⋆,n ≤σ2
⋆.
(3) We denote ρ2
⋆=
ω
M (σ2
⋆+ ζ2
⋆) + σ2
⋆,n.

[Robbins and Monro, 1951] – can be written as follows: xt+1 = xt −γ∇f j(xt), where j is sampled
from {1, . . . , n} uniformly at random. This procedure thus uses with-replacement sampling in order
to select the stochastic gradient used at each step from the dataset. However, in the training of DL
models, without-replacement sampling is used much more often: that is, at the beginning of each
epoch we choose a permutation π1, π2, . . . , πn of {1, 2, . . . , n} and do the i-th update using the πi-ith
gradient: xi+1
t
= xi
t −γ∇f πi(xi
t). Without-replacement sampling SGD, also known as Random
Reshuffling (RR) [Bottou, 2009], typically achieves better asymptotic convergence rates compared
to with-replacement SGD and can improve upon it in many settings as shown by recent theoretical
progress [Mishchenko et al., 2020, Ahn et al., 2020, Rajput et al., 2020, Safran and Shamir, 2021].
While with-replacement SGD achieves an error proportional to O
  1

T

after T steps [Stich, 2019],
Random Reshuffling achieves an error of O
  n

T 2

after T steps, faster than SGD when the number of
steps T is large.

1.3
Can Communication Compression and Random Reshuffling be Friends?

As we described earlier, Random Reshuffling and communication compression are two important
tools for training modern DL models, and both techniques are relatively well understood. However,
there are no papers that study Random Reshuffling and communication compression in combination.
This leads us to the natural question: how these techniques should be combined to improve the
convergence speed of existing distributed methods?

1.4
Contributions

In this paper, we aim to develop methods for Distributed and Federated Learning that combine
gradient compression and random reshuffling. While each of these techniques can aid in reducing the
communication complexity of distributed optimization, their combination is under-explored. Thus
our goal is to design methods that improve upon existing algorithms in convergence rates and in
practice. We summarize our contributions as follows.

3


---Page Break---
⋄The issue: naïve combination has no improvements. As a natural step towards our goal, we
propose and study a new algorithm, Q-RR (Algorithm 1), that combines random reshuffling with
gradient compression at every communication round. However, for Q-RR our theoretical results do
not show any improvement upon QSGD when the compression level is reasonable (see Table 1).
Moreover, we observe similar performance of Q-RR and QSGD in various numerical experiments.
Therefore, we conclude that this phenomenon is not an artifact of our analysis but rather an issue of
Q-RR: communication compression adds an additional noise that dominates the one coming from
the stochastic gradients sampling.
⋄The remedy: reduction of compression variance. To remove the additional variance added
by the compression and unleash the potential of Random Reshuffling in distributed learning
with compression, we propose DIANA-RR (Algorithm 2), a combination of Q-RR and the DIANA
algorithm. We derive the convergence rates of the new method and show that it improves upon the
convergence rates of Q-RR, QSGD, and DIANA (see Table 1). We point out that to achieve such
results we use n shift-vectors per worker in DIANA-RR unlike DIANA that uses only 1 shift-vector.
⋄Extensions to the local steps. Inspired by the NASTYA algorithm of Malinovsky et al. [2022], we
propose a variant of NASTYA, Q-NASTYA (Algorithm 3), that naïvely mixes quantization, local
steps with random reshuffling, and uses different local and server stepsizes. Although it improves
in per-round communication cost over NASTYA but, similar to Q-RR, we show that Q-NASTYA
suffers from added variance due to gradient quantization. To overcome this issue, we propose
another algorithm, DIANA-NASTYA (Algorithm 4), that adds DIANA-style variance reduction to
Q-NASTYA and removes the additional variance.

Finally, to illustrate our theoretical findings we conduct experiments on federated logistic regression
tasks and on distributed training of neural networks.

2
Algorithms and convergence theory

We will primarily consider the setting of strongly-convex and smooth optimization. We assume that
the average function f is strongly convex:
Assumption 2. Function f : Rd →R is µ-strongly convex, i.e., for all x, y ∈Rd,

f(x) −f(y) −⟨∇f(y), x −y⟩≥µ

2 ∥x −y∥2,
(3)

and functions f i
1, f i
2, . . . , f i
M : Rd →R are convex for all i = 1, . . . , n.

Examples of objectives satisfying Assumption 2 include ℓ2-regularized linear and logistic regression.
Throughout the paper, we assume that f has the unique minimizer x⋆∈Rd. We also use the
assumption that each individual loss f i
m is smooth, i.e. has Lipschitz-continuous first-order derivatives:

Assumption 3. Function f i
m : Rd →R is Li,m-smooth for every i ∈[n] and m ∈[M], i.e., for all
x, y ∈Rd and for all m ∈[M] and i ∈[n],

∥∇f i
m(x) −∇f i
m(y)∥≤Li,m∥x −y∥.
(4)

We denote the maximal smoothness constant as Lmax
def
= maxi,m Li,m.

For some methods, we shall additionally impose the assumption that each function is strongly convex:
Assumption 4. Each function f i
m : Rd →R is eµ-strongly convex.

The Bregman divergence associated with a convex function h is defined for all x, y ∈Rd as

Dh(x, y)
def
= h(x) −h(y) −⟨∇h(y), x −y⟩.

Note that the inequality (3) defining strong convexity can be written as Df(x, y) ≥µ

2 ∥x −y∥2.

2.1
Algorithm Q-RR

The first method we introduce is Q-RR (Algorithm 1). Q-RR is a straightforward combination of
distributed random reshuffling and gradient quantization. This method can be seen as the stochastic
without-replacement analogue of the distributed quantized gradient method of Khirirat et al. [2018].

4


---Page Break---
Algorithm 1 Q-RR: Distributed Random Reshuffling with Quantization

Input: x0 – starting point, γ > 0 – stepsize

1: for t = 0, 1, . . . , T −1 do
2:
Receive xt from the server and set x0
t,m = xt
3:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
4:
for i = 0, 1, . . . , n −1 do
5:
for m = 1, . . . , M in parallel do

6:
Receive xi
t from the server, compute and send Q

∇f πi
m
m (xi
t)

back

7:
end for
8:
Compute and send xi+1
t
= xi
t −γ 1

M
PM
m=1 Q

∇f πi
m
m (xi
t)

to the workers

9:
end for
10:
xt+1 = xn
t
11: end for
Output: xT

We shall use the notion of shuffling radius defined by Mishchenko et al. [2021] for the analysis of
distributed methods with random reshuffling:

Definition 2.1. Define the iterate sequence xi+1
⋆
= xi
⋆−γ

M
PM
m=1 ∇f πi
m
m (x⋆). Then the shuffling
radius is the quantity

σ2
rad
def
= max
i

(
1
γ2M

M
X

m=1
EDf πi
m (xi
⋆, x⋆)

)

.

We provide clarifications regarding this term in Appendix C.1. To compare our subsequent results
with known ones, we introduce bounds on the shuffling radius. The following lemma demonstrates
that these bounds are independent of the stepsize γ, even though γ is used in Definition 2.1.
Lemma 2.1 ([Mishchenko et al., 2020]). Let Assumptions 3, 4 hold. Then the shuffling radius σ2
rad
satisfies the following inequlity
eµn

8 σ2
⋆,n ≤σ2
rad ≤Lmaxn

4
σ2
⋆,n,

where σ2
⋆,n
def
=
1
n

nP

i=1
∥∇f i(x⋆)∥2, and f i =
1
M

M
P

m=1
f i
m.

We now state the main convergence theorem for Algorithm 1:
Theorem 2.1. Let Assumptions 1, 3, 4 hold and let the stepsize satisfy 0 < γ ≤
1
(1+2 ω

M )Lmax . Then,

for all T ≥0 the iterates produced by Q-RR (Algorithm 1) satisfy

E∥xT −x⋆∥2 ≤(1 −γeµ)nT ∥x0 −x⋆∥2 + 2γ2σ2
rad
eµ
+ 2γω

eµM (ζ2
⋆+ σ2
⋆),
(5)

where ζ2
⋆
def
=
1
M

M
P

m=1
∥∇fm(x⋆)∥2, and σ2
⋆
def
=
1
Mn

M
P

m=1

nP

i=1
∥∇f i
m(x⋆) −∇fm(x⋆)∥2.

All proofs are relegated to the appendix. By choosing the stepsize γ properly, we can obtain the
communication complexity (number of communication rounds) needed to find an ε-approximate
solution as follows:
Corollary 1. Under the same conditions as Theorem 2.1 and for Algorithm 1, there exists a stepsize
γ > 0 such that the number of communication rounds nT to find a solution with accuracy ε > 0

(i.e. E∥xT −x⋆∥2 ≤ϵ) is equal to e
O
 
1 + ω

M
 Lmax

eµ
+ ω(ζ2
⋆+σ2
⋆)
M eµ2ε
+
σrad
√

eµ3ε


, where e
O(·) hides

constants and logarithmic factors.

The
complexity
of
Quantized
SGD
(QSGD)
is
[Gorbunov
et
al.,
2020]:

e
O
 
1 + ω

M
 Lmax

µ
+ (ωζ2
⋆+(1+ω)σ2
⋆)
Mµ2ε


. For simplicity, let us neglect the differences between

5


---Page Break---
Algorithm 2 DIANA-RR

Input: x0 – starting point, {hi
0,m}M,n
m,i=1,1 – initial shift-vectors, γ > 0 – stepsize, α > 0 – stepsize
for learning the shifts
1: for t = 0, 1, . . . , T −1 do
2:
Receive xt from the server and set x0
t,m = xt
3:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
4:
for i = 0, 1, . . . , n −1 do
5:
for m = 1, 2, . . . , M in parallel do

6:
Receive xi
t from the server, compute and send Q

∇f πi
m
m (xi
t) −hπi
m
t,m

back

7:
Set ˆgπi
m
t,m = hπi
m
t,m + Q

∇f πi
m
m (xi
t,m) −hπi
m
t,m


8:
Set hπi
m
t+1,m = hπi
m
t,m + αQ

∇f πi
m
m (xi
t,m) −hπi
m
t,m


9:
end for
10:
Compute xi+1
t
= xi
t −γ 1

M
PM
m=1 ˆgπi
m
t,m and send xi+1
t
to the workers
11:
end for
12:
xt+1 = xn
t
13: end for
Output: xT

µ and eµ. First, when ω = 0 we recover the complexity of FedRR [Mishchenko et al., 2021]
which is known to be better than the one of SGD as long as ε is sufficiently small as we have
nµσ2
⋆,n/8 ≤σ2
rad ≤nLσ2
⋆,n/4 from Lemma 2.1. Next, when M = 1 and ω = 0 (single node, no
compression) our results recovers the rate of RR [Mishchenko et al., 2020].

However, it is more interesting to compare Q-RR and QSGD when M > 1 and ω > 1, which is
typically the case. In these settings, Q-RR and QSGD have the same complexity since the O(1/ε)
term dominates the O(1/√ε) one if ε is sufficiently small. That is, the derived result for Q-RR has no
advantages over the known one for QSGD unless ω is very small, which means that there is almost
no compression at all. We also observe this phenomenon in the experiments.

The main reason for that is the variance appearing due to compression. Indeed, even if the current
point is the solution of the problem (xi
t = x∗), the update direction −γ 1

M
PM
m=1 Q

∇f πi
m
m (xi
t)


has the compression variance

EQ

"
γ
M

M
X

m=1


Q(∇f πi
m
m (x⋆)) −∇f πi
m
m (x⋆)


2#

≤γ2ω

M 2

M
X

m=1
∥∇f πi
m
m (x⋆)∥2.

This upper bound is tight and non-zero in general. Moreover, it is proportional to γ2 that creates the
term proportional to γ in (5) like in the convergence results for QSGD/SGD, while the RR-variance is
proportional to γ2 in the same bound. Therefore, during the later stages of the convergence Q-RR
behaves similarly to QSGD when we decrease the stepsize.

2.2
Algorithm DIANA-RR

To reduce the additional variance caused by compression, we apply DIANA-style shift sequences
[Mishchenko et al., 2019b, Horváth et al., 2019]. Thus we obtain DIANA-RR (Algorithm 2), which
applies compression to the differences between the gradients and learnable shifts. Since the shifts are
updated using the past gradients information, one can see DIANA-RR as a method with compression
of gradient differences. We notice that unlike DIANA, DIANA-RR has n shift-vectors on each node.
Theorem 2.2. Let Assumptions 1, 3, 4 hold and suppose that the stepsizes satisfy γ
≤

min

α
2neµ,
1
(1+ 6ω

M )Lmax


, and α ≤
1
1+ω. Define the following Lyapunov function for every t ≥0

Ψt+1
def
= ∥xt+1 −x⋆∥2 + 4ωγ2

αM 2

M
X

m=1

n−1
X

j=0
(1 −γµ)j∥∆j
t+1,m∥2,
(6)

6


---Page Break---
Algorithm 3 Q-NASTYA

Input: x0 – starting point, γ > 0 – local stepsize, η > 0 – global stepsize

1: for t = 0, 1, . . . , T −1 do
2:
for m ∈[M] in parallel do
3:
Receive xt from the server and set x0
t,m = xt
4:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
5:
for i = 0, 1, . . . , n −1 do

6:
Set xi+1
t,m = xi
t,m −γ∇f πi
m
m (xi
t,m)
7:
end for
8:
Compute gt,m =
1
γn
 
xt −xn
t,m

and send Qt(gt,m) to the server
9:
end for
10:
Compute gt =
1
M
PM
m=1 Qt(gt,m)
11:
Compute xt+1 = xt −ηgt and send xt+1 to the workers
12: end for
Output: xT

where ∆j
t+1,m = hπj
m
t+1,m −∇f πj
m
m (x⋆) Then, for all T ≥0 the iterates produced by DIANA-RR
(Algorithm 2) satisfy

E [ΨT ] ≤(1 −γeµ)nT Ψ0 + 2γ2σ2
rad
eµ
Corollary 2. Under the same conditions as Theorem 2.2 and for Algorithm 2, there exists stepsizes
γ, α > 0 such that the number of communication rounds nT to find a solution with accuracy ε > 0 is

e
O

n(1 + ω) +
 
1 + ω

M
 Lmax

eµ
+
σrad
√

εeµ3


.

Unlike Q-RR/QSGD/DIANA, DIANA-RR does not have a e
O(1/ε)-term, which makes it superior to
Q-RR/QSGD/DIANA for small enough ε. However, the complexity of DIANA-RR has an additive
e
O(n(1 + ω)) term arising due to learning the shifts {hi
t,m}m∈[M],i∈[n]. Nevertheless, this additional
term is not the dominating one when ε is small enough. Next, we elaborate a bit more on the
comparison between DIANA and DIANA-RR. That is, DIANA has e
O
 
1 + ω

M
 Lmax

µ
+ (1+ω)σ2
⋆
Mµ2ε


complexity [Gorbunov et al., 2020]. Neglecting the differences between µ and eµ, we observe
a similar relation between DIANA-RR and DIANA as between RR and SGD: instead of the term
O((1+ω)σ2
⋆/(Mµ2ε)) appearing in the complexity of DIANA, DIANA-RR has O(σrad/√

εeµ3) term much
better depending on ε. To the best of our knowledge, our result is the only known one establishing
the theoretical superiority of RR to regular SGD in the context of distributed learning with gradient
compression. Moreover, when ω = 0 (no compression) we recover the rate of FedRR and when
additionally M = 1 (single worker) we recover the rate of RR.

2.3
Algorithms with Local Steps

In this subsection, we study a new variant of NASTYA, Q-NASTYA (Algorithm 3), that unifies
quantization, local steps with random reshuffling, and uses different local and server stepsizes.
Although it improves in per-round communication cost over NASTYA but, similar to Q-RR, we show
that Q-NASTYA suffers from added variance due to gradient quantization. To overcome this issue, we
propose another algorithm, DIANA-NASTYA (Algorithm 4), that adds DIANA-style variance reduction
to Q-NASTYA and removes the additional variance.

Theorem 2.3. Let Assumptions 1, 2, 3 hold. Let the stepsizes γ, η satisfy 0 < η ≤
1
16Lmax(1+ ω

M ),

0 < γ ≤
1
5nLmax . Then, for all T ≥0 the iterates produced by Q-NASTYA (Algorithm 3) satisfy

E

∥xT −x⋆∥2
≤

1 −ηµ

2

T
∥x0 −x⋆∥2 + 8 ηω

µM ζ2
⋆+ 9

2
γ2nLmax

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

.

Corollary 3. Under the same conditions as Theorem E.1 and for Algorithm 3, there exist stepsizes
γ = η/n and η > 0 such that the number of communication rounds T to find a solution with accuracy

7


---Page Break---
Algorithm 4 DIANA-NASTYA

Input: x0 – starting point, {h0,m}M
m=1 – initial shift-vectors, γ > 0 – local stepsize, η > 0 – global
stepsize, α > 0 – stepsize for learning the shifts
1: for t = 0, 1, . . . , T −1 do
2:
for m = 1, . . . , M in parallel do
3:
Receive xt from the server and set x0
t,m = xt
4:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
5:
for i = 0, 1, . . . , n −1 do

6:
Set xi+1
t,m = xi
t,m −γ∇f πi
m
m (xi
t,m)
7:
end for
8:
Compute gt,m =
1
γn
 
xt −xn
t,m

and send Qt (gt,m −ht,m) to the server
9:
Set ht+1,m = ht,m + αQt (gt,m −ht,m)
10:
Set ˆgt,m = ht,m + Qt (gt,m −ht,m)
11:
end for
12:
ht+1 =
1
M
PM
m=1 ht+1,m = ht + α

M
PM
m=1 Qt (gt,m −ht,m)

13:
ˆgt =
1
M
PM
m=1 ˆgt,m = ht +
1
M
PM
m=1 Qt (gt,m −ht,m)
14:
xt+1 = xt −ηˆgt
15: end for
Output: xT

ε > 0 is e
O

Lmax

µ
 
1 + ω

M

+ ω

M
ζ2
⋆
εµ3 +
q

Lmax

εµ3
q

ζ2⋆+ σ2⋆

n


. If γ →0, one can choose η > 0 such

that the above complexity bound improves to e
O

Lmax

µ
 
1 + ω

M

+ ω

M
ζ2
⋆
εµ3

.

We emphasize several differences with the known theoretical results. First, the FedCOM method
of Haddadpour et al. [2021] was analyzed in the homogeneous setting only, i.e., fm(x) = f(x)
for all m ∈[M], which is an unrealistic assumption for FL applications. In contrast, our result
holds in the fully heterogeneous case. Next, the analysis of FedPAQ of Reisizadeh et al. [2020]
uses a bounded variance assumption, which is also known to be restrictive. Nevertheless, let us
compare to their result. Reisizadeh et al. [2020] derive the following complexity for their method:
e
O

Lmax

µ
 
1 + ω

M

+ ω

M
σ2
µ2ε +
σ2
Mµ2ε

. This result is inferior to the one we show for Q-NASTYA:

when ω is small, the main term in the complexity bound of FedPAQ is e
O (1/ε), while for Q-NASTYA
the dominating term is of the order e
O (1/√ε) (when ω and ε are sufficiently small). We also highlight
that FedCRR [Malinovsky and Richtárik, 2022] does not converge if ω > M 2γµε/(2∥xn
∗,m∥
2), while
Q-NASTYA does for any ω ≥0. Finally, when ω = 0 (no compression) we recover NASTYA as a
special case, and using γ = η/n, we recover the rate of FedRR [Mishchenko et al., 2021].

Theorem 2.4. Let Assumptions 1, 2, 3 hold. Suppose the stepsizes γ, η, α satisfy 0 < γ ≤
1
16Lmaxn,

0 < η ≤min

α
2µ,
1
16Lmax(1+ 9ω

M )


, and α ≤
1
1+ω. Define the following Lyapunov function:

Ψt+1
def
= ∥xt+1 −x⋆∥2 + 8ωη2

αM 2

M
X

m=1
∥ht+1,m −h⋆
m∥2.
(7)

Then, for all T ≥0 the iterates produced by DIANA-NASTYA (Algorithm 4) satisfy

E [ΨT ] ≤

1 −ηµ

2

T
Ψ0 + 9

2
γ2nL

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

.
(8)

Corollary 4. Under the same conditions as Theorem E.2 and for Algorithm 4, there exist stepsizes
γ = η/n, η > 0, α > 0 such that the number of communication rounds T to find a solution with

accuracy ε > 0 is e
O

ω + Lmax

µ
 
1 + ω

M

+
q

Lmax

εµ3
q

ζ2⋆+ σ2⋆

n


. If γ →0, one can choose η > 0

such that the above complexity bound improves to e
O

ω + Lmax

µ
 
1 + ω

M

.

8


---Page Break---
1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

mushrooms; Rand-2

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

w8a; Rand-6

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

a9a; Rand-2

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

Figure 1: Non-local methods

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

mushrooms; Rand-2

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

w8a; Rand-6

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

a9a; Rand-2

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

Figure 2: Local methods

Figure 3: The comparison of the proposed methods (Q-NASTYA, DIANA-NASTYA, Q-RR, DIANA-RR),
DIANA-RR-1S (a modification of DIANA-RR), and existing baselines (QSGD, DIANA, FedCOM,
FedPAQ). All methods use tuned stepsizes and the Rand-k compressor.

In contrast to Q-NASTYA, DIANA-NASTYA does not suffer from the e
O(1/ε) term in the complexity
bound. This shows the superiority of DIANA-NASTYA to Q-NASTYA. Next, FedCRR-VR [Malinovsky

and Richtárik, 2022] has the rate e
O

(ω+1)(1−1

κ)
n

(1−(1−1

κ)
n)
2 +
√κ(ζ⋆+σ⋆)

µ√ε


, which depends on e
O (1/√ε).

However, the first term is close to e
O
 
(ω + 1)κ2
for a large condition number. FedCRR-VR-2
utilizes variance reduction technique from Malinovsky et al. [2021] and it allows to get rid of

permutation variance. This method has e
O



(ω+1)

1−
1
κ√κn

 n

2

1−

1−
1
κ√κn

 n

2
2 +
√κζ⋆

µ√ε



complexity, but it requires

additional assumption on number of functions n and thus not directly comparable with our result.
Note that if we have no compression (ω = 0), DIANA-NASTYA recovers rate of NASTYA.

In Appendix J, we provide versions of Q-NASTYA and DIANA-NASTYA with partial participation of
clients, which is another important aspect of FL, and derive the convergence results for them.

3
Experiments

We evaluated our methods for solving logistic regression problems and training neural networks
in three parts: (i) Comparison of the proposed non-local methods with existing baselines; (ii)
Comparison of the proposed local methods with existing baselines; (iii) Comparison of the proposed
non-local methods in training ResNet-18 on CIFAR10.

Logistic Regression.
To confirm our theoretical results we conducted several numerical experiments
on binary classification problem with L2 regularized logistic regression of the form

min
x∈Rd

"

f(x)
def
=
1
M

M
X

m=1

1
nm

nm
X

i=1
fm,i

#

,
(9)

where fm,i
def
= log
 
1 + exp(−ymia⊤
mix)

+ λ∥x∥2
2 (ami, ymi) ∈Rd× ∈{−1, 1}, i = 1, . . . , nm
are the training data samples stored on machines m = 1, . . . , M, and λ > 0 is a regularization
parameter. In all experiments, for each method, we used the largest stepsize allowed by its theory

9


---Page Break---
multiplied by some individually tuned constant multiplier. For better parallelism, each worker m
uses mini-batches of size ≈0.1nm. In all algorithms, as a compression operator Q, we use Rand-k
[Beznosikov et al., 2020] with fixed compression ratio k/d ≈0.02, where d is the number of features
in the dataset.

In our first experiment (see Figure 1), we compare Q-RR, DIANA-RR, and DIANA-RR-1S with
classical baselines (QSGD [Alistarh et al., 2017], DIANA [Mishchenko et al., 2019b]) that use a with-
replacement mini-batch SGD estimator. DIANA-RR-1S is a memory-friendly version of DIANA-RR

that stores and uses a single shift ht,m on the worker side rather than n individual shifts hπi
m
t,m. Figure 1
illustrates that Q-RR exhibits similar behavior to QSGD, with both methods being slower than DIANA
methods across all considered datasets. DIANA-RR-1S and DIANA show comparable convergence
rates, indicating that random reshuffling alone, without introducing additional shifts, does not make
a significant difference. Finally, DIANA-RR achieves the best rate among all considered non-local
methods, efficiently reducing the variance and reaching the lowest functional sub-optimality tolerance.
These experimental results align perfectly with our theoretical analysis.

The second experiment shows that DIANA-based method can significantly outperform in practice
when one applies it to local methods as well. In particular, whereas Q-NASTYA shows comparative
behavior as existing methods FedCOM [Haddadpour et al., 2021], FedPAQ [Reisizadeh et al., 2020]
in all considered datasets, DIANA-NASTYA noticeably outperforms other methods.

0
2
4
6
8
#bits/n
1e7

20

40

60

Top1 acc (validation)

QSGD =  0.001
QSGD-RR =  0.001

0
1
2
3
#bits/n
1e11

25

50

75

Top1 acc (validation)

DIANA = 1.0
DIANA-RR = 1.0

Figure 4: The comparison of Q-RR, QSGD, DIANA, and DIANA-RR on the task of training ResNet-18
on CIFAR-10 with n = 10 workers. Top-1 accuracy on test set is reported. Stepsizes were tuned and
workers used Rand-k compressor with k/d ≈0.05.

Training Deep Neural Network model: ResNet-18 on CIFAR-10.
Since random reshuffling is a
very popular technique in training neural networks, it is natural to test the proposed methods on such
problems. Therefore, in the second set of experiments, we consider training ResNet-18 [He et al.,
2016] model on the CIFAR10 dataset Krizhevsky and Hinton [2009]. To conduct these experiments
we use FL_PyTorch simulator [Burlachenko et al., 2021].

The main goal of this experiment is to verify the phenomenon observed in Experiment 1 on the
training of a deep neural network. That is, we tested Q-RR, QSGD, DIANA, and DIANA-RR in
the distributed training of ResNet-18 on CIFAR10, see Figure 4. As in the logistic regression
experiments, we observe that (i) Q-RR and QSGD behave similarly and (ii) DIANA-RR outperforms
DIANA. For further experimental results and details, we refer to Appendix B.

4
Conclusion

In this work, we provide the first study of distributed random reshuffling with communication
compression. Our theoretical and empirical findings illustrate the inefficiency of naïve combination
of random reshuffling and communication compression. We also show how this issue can be resolved
via the usage of shifts for communication compression. Finally, we develop and analyze methods
with random reshuffling, communication compression, and local steps. It is worth mentioning that
although our theoretical results are obtained for strongly convex problems, the considered methods
perform well in the experiments on non-convex tasks like training neural networks.

10


---Page Break---
Acknowledgements

The research reported in this publication was supported by funding from King Abdullah University of
Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) Center of Excellence
for Generative AI, under award number 5940, iii) SDAIA-KAUST Center of Excellence in Artificial
Intelligence and Data Science.

The work of A. Sadiev and E. Gorbunov (while affiliated with MIPT) was supported by a grant
for research centers in the field of artificial intelligence, provided by the Analytical Center for the
Government of the Russian Federation in accordance with the subsidy agreement (agreement identifier
000000D730324P540002) and the agreement with the Moscow Institute of Physics and Technology
dated November 1, 2021 No. 70-2021-00138.

References

Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. SGD with shuffling: optimal rates without component
convexity and large epoch requirements. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell,
Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020,
December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/
hash/cb8acb1dc9821bf74e6ca9068032d623-[]Abstract.html.

Dan Alistarh, Demjan Grubic, Jerry Z. Li, Ryota Tomioka, and Milan Vojnovic.
QSGD:
Communication-efficient SGD via gradient quantization and encoding. In Proceedings of the 31st
International Conference on Neural Information Processing Systems, NIPS’17, page 1707–1718,
Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Rodolfo Stoffel Antunes, Cristiano André da Costa, Arne Küderle, Imrana Abdullahi Yari, and
Björn Eskofier. Federated learning for healthcare: Systematic review and architecture proposal.
ACM Trans. Intell. Syst. Technol., 13(4), 2022. ISSN 2157-6904. doi: 10.1145/3501813. URL

https://doi.org/10.1145/3501813.

Debraj Basu, Deepesh Data, Can Karakus, and Suhas Diggavi. Qsparse-local-SGD: Distributed SGD
with quantization, sparsification and local computations. volume 32, 2019.

Aleksandr Beznosikov, Samuel Horváth, Peter Richtárik, and Mher Safaryan. On biased compression
for distributed learning. arXiv preprint arXiv:2002.12410, abs/2002.12410, 2020. URL https:
//arXiv.org/abs/2002.12410.

Léon Bottou.
Curiously fast convergence of some stochastic gradient descent algorithms.
In
Proceedings of the symposium on learning and data science, Paris, volume 8, pages 2624–2633.
Citeseer, 2009.

Konstantin Burlachenko, Samuel Horváth, and Peter Richtárik. Fl_pytorch: optimization research
simulator for federated learning. In Proceedings of the 2nd ACM International Workshop on
Distributed Machine Learning, pages 1–7, 2021.

Chih-Chung Chang and Chih-Jen Lin. LIBSVM: a library for support vector machines. ACM
Transactions on Intelligent Systems and Technology (TIST), 2(3):1–27, 2011.

Ilyas Fatkhullin, Igor Sokolov, Eduard Gorbunov, Zhize Li, and Peter Richtárik.
Ef21 with
bells & whistles: Practical algorithmic extensions of modern error feedback. arXiv preprint
arXiv:2110.03294, 2021.

Margalit R. Glasgow, Honglin Yuan, and Tengyu Ma. Sharp bounds for federated averaging (local
SGD) and continuous perspective. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera,
editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics,
volume 151 of Proceedings of Machine Learning Research, pages 9050–9090. PMLR, 2022. URL
https://proceedings.mlr.press/v151/glasgow22a.html.

Eduard Gorbunov, Filip Hanzely, and Peter Richtárik. A unified theory of SGD: variance reduc-
tion, sampling, quantization and coordinate descent. In International Conference on Artificial
Intelligence and Statistics, pages 680–690. PMLR, 2020.

11


---Page Break---
Eduard Gorbunov, Filip Hanzely, and Peter Richtárik. Local SGD: Unified theory and new efficient
methods. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International
Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine
Learning Research, pages 3556–3564. PMLR, 13–15 Apr 2021. URL https://proceedings.
mlr.press/v130/gorbunov21a.html.

Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter
Richtárik. Sgd: General analysis and improved rates. In International conference on machine
learning, pages 5200–5209. PMLR, 2019.

Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola,
Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet
in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, and Mehrdad Mahdavi. Federated
learning with compression: Unified analysis and sharp guarantees. In Arindam Banerjee and Kenji
Fukumizu, editors, The 24th International Conference on Artificial Intelligence and Statistics,
AISTATS 2021, April 13-15, 2021, Virtual Event, volume 130 of Proceedings of Machine Learning
Research, pages 2350–2358. PMLR, 2021. URL http://proceedings.mlr.press/v130/
haddadpour21a.html.

K. He et al. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification. In Proceedings of the IEEE international
conference on computer vision, pages 1026–1034, 2015.

Samuel Horváth, Dmitry Kovalev, Konstantin Mishchenko, Sebastian Stich, and Peter Richtárik.
Stochastic distributed learning with gradient quantization and variance reduction. arXiv preprint
arXiv:1904.05115, abs/1904.05115, 2019. URL https://arXiv.org/abs/1904.05115.

Samuel Horváth, Maziar Sanjabi, Lin Xiao, Peter Richtárik, and Michael Rabbat. FedShuffle:
Recipes for better use of local work in federated learning. arXiv preprint arXiv:2204.13169,
abs/2204.13169, 2022. URL https://arXiv.org/abs/2204.13169.

Kun Huang, Xiao Li, Andre Milzarek, Shi Pu, and Junwen Qiu. Distributed random reshuffling over
networks. arXiv preprint arXiv:2112.15287, abs/2112.15287, 2021. URL https://arXiv.org/
abs/2112.15287.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. In International conference on machine learning, pages 448–456.
PMLR, 2015.

Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin
Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L.
D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett,
Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang
He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi,
Mikhail Khodak, Jakub Koneˇcný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo,
Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus
Pagh, Mariana Raykova, Hang Qi, Daniel Ramage, Ramesh Raskar, Dawn Song, Weikang Song,
Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma,
Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao. Advances
and open problems in federated learning. arXiv preprint arXiv:1912.04977, abs/1912.04977, 2019.
URL https://arXiv.org/abs/1912.04977.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and
Ananda Theertha Suresh. SCAFFOLD: stochastic controlled averaging for federated learning. In
Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July
2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5132–5143.
PMLR, 2020. URL http://proceedings.mlr.press/v119/karimireddy20a.html.

12


---Page Break---
Sarit Khirirat, Hamid Reza Feyzmahdavian, and Mikael Johansson. Distributed learning with
compressed gradients. arXiv preprint arXiv:1806.06573, abs/1806.06573, 2018. URL https:
//arXiv.org/abs/1806.06573.

Jakub Koneˇcný, H. Brendan McMahan, Felix Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave
Bacon. Federated learning: strategies for improving communication efficiency. In NIPS Private
Multi-Party Machine Learning Workshop, 2016.

Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images.
Technical Report 0, University of Toronto, Toronto, Ontario, 2009.

Ming Liu, Stella Ho, Mengqi Wang, Longxiang Gao, Yuan Jin, and He Zhang. Federated learning
meets natural language processing: a survey. arXiv preprint arXiv:2107.12603, abs/2107.12603,
2021. URL https://arXiv.org/abs/2107.12603.

Grigory Malinovsky and Peter Richtárik. Federated random reshuffling with compression and
variance reduction. arXiv preprint arXiv:2205.03914, abs/2205.03914, 2022. URL https:
//arXiv.org/abs/2205.03914.

Grigory Malinovsky, Alibek Sailanbayev, and Peter Richtárik. Random reshuffling with variance
reduction: New analysis and better rates. arXiv preprint arXiv:2104.09342, 2021.

Grigory Malinovsky, Konstantin Mishchenko, and Peter Richtárik. Server-side stepsizes and sampling
without replacement provably help in federated optimization. arXiv preprint arXiv:2201.11066,
abs/2201.11066, 2022. URL https://arXiv.org/abs/2201.11066.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-efficient learning of deep networks from decentralized data. In Artificial intelli-
gence and statistics, pages 1273–1282. PMLR, 2017.

Konstantin Mishchenko, Eduard Gorbunov, Martin Takáˇc, and Peter Richtárik. Distributed learning
with compressed gradient differences. arXiv preprint arXiv:1901.09269, 2019a.

Konstantin Mishchenko, Eduard Gorbunov, Martin Takác, and Peter Richtárik. Distributed learning
with compressed gradient differences. arXiv preprint arXiv:1901.09269, abs/1901.09269, 2019b.
URL https://arXiv.org/abs/1901.09269.

Konstantin Mishchenko, Ahmed Khaled, and Peter Richtárik. Random reshuffling: Simple analysis
with vast improvements. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina
Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33:
Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December
6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/
c8cc6e90ccbff44c9cee23611711cdc4-[]Abstract.html.

Konstantin Mishchenko, Ahmed Khaled, and Peter Richtárik. Proximal and federated random
reshuffling. arXiv preprint arXiv:2102.06704, abs/2102.06704, 2021. URL https://arXiv.
org/abs/2102.06704.

Aritra Mitra, Rayana Jaafar, George J. Pappas, and Hamed Hassani.
Linear convergence
in federated learning:
Tackling client heterogeneity and sparse gradients.
In M. Ran-
zato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Ad-
vances in Neural Information Processing Systems, volume 34, pages 14606–14619. Cur-
ran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/file/
7a6bda9ad6ffdac035c752743b7e9d0e-[]Paper.pdf.

Tomoya Murata and Taiji Suzuki. Bias-variance reduced local SGD for less heterogeneous federated
learning.
In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International
Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139
of Proceedings of Machine Learning Research, pages 7872–7881. PMLR, 2021. URL http:
//proceedings.mlr.press/v139/murata21a.html.

Jose Javier Gonzalez Ortiz, Jonathan Frankle, Mike Rabbat, Ari Morcos, and Nicolas Ballas. Trade-
offs of local SGD at scale: an empirical study. arXiv preprint arXiv:2110.08133, abs/2110.08133,
2021. URL https://arXiv.org/abs/2110.08133.

13


---Page Break---
Shashank Rajput, Anant Gupta, and Dimitris S. Papailiopoulos. Closing the convergence gap of
SGD without replacement. In Proceedings of the 37th International Conference on Machine
Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine
Learning Research, pages 7964–7973. PMLR, 2020. URL http://proceedings.mlr.press/
v119/rajput20a.html.

Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani.
Fedpaq: A communication-efficient federated learning method with periodic averaging and quanti-
zation. In Silvia Chiappa and Roberto Calandra, editors, The 23rd International Conference on
Artificial Intelligence and Statistics, AISTATS 2020, 26-28 August 2020, Online [Palermo, Sicily,
Italy], volume 108 of Proceedings of Machine Learning Research, pages 2021–2031. PMLR, 2020.
URL http://proceedings.mlr.press/v108/reisizadeh20a.html.

Peter Richtárik, Elnur Gasanov, and Konstantin Burlachenko. Error feedback reloaded: From
quadratic to arithmetic mean of smoothness constants. arXiv preprint arXiv:2402.10774, 2024.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical
statistics, pages 400–407, 1951.

Mher Safaryan, Filip Hanzely, and Peter Richtárik.
Smoothness matrices beat smoothness
constants: Better communication compression techniques for distributed optimization.
In
M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors,
Advances in Neural Information Processing Systems, volume 34, pages 25688–25702. Cur-
ran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/file/
d79c6256b9bdac53a55801a066b70da3-[]Paper.pdf.

Itay Safran and Ohad Shamir. Random shuffling beats SGD only after many epochs on ill-conditioned
problems. arXiv preprint arXiv:2106.06880, abs/2106.06880, 2021. URL https://arXiv.org/
abs/2106.06880.

Sebastian U. Stich. Unified optimal analysis of the (stochastic) gradient method. arXiv preprint
arXiv:1907.04232, abs/1907.04232, 2019. URL https://arXiv.org/abs/1907.04232.

Sebastian U. Stich. On communication compression for distributed optimization on heterogeneous
data. arXiv preprint arXiv:2009.02388, abs/2009.02388, 2020. URL https://arXiv.org/abs/
2009.02388.

Sebastian U. Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In
Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi,
and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual
Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8,
2018, Montréal, Canada, pages 4452–4463, 2018. URL https://proceedings.neurips.cc/
paper/2018/hash/b440509a0106086a67bc2ea9df0a1dab-[]Abstract.html.

Zhenheng Tang, Shaohuai Shi, Xiaowen Chu, Wei Wang, and Bo Li. Communication-efficient dis-
tributed deep learning: a comprehensive survey. arXiv preprint arXiv:2003.06307, abs/2003.06307,
2020. URL https://arXiv.org/abs/2003.06307.

Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-
efficient distributed optimization. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-
Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31.
Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/file/
3328bdf9a4b9504b9398284244fe97c2-[]Paper.pdf.

Blake Woodworth, Brian Bullins, Ohad Shamir, and Nathan Srebro. The min-max complexity
of distributed stochastic convex optimization with intermittent communication. arXiv preprint
arXiv:2102.01583, abs/2102.01583, 2021. URL https://arXiv.org/abs/2102.01583.

Blake E. Woodworth, Kumar Kshitij Patel, and Nati Srebro. Minibatch vs local SGD for heteroge-
neous distributed learning. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina
Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33:
Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December
6-12, 2020, virtual, 2020a. URL https://proceedings.neurips.cc/paper/2020/hash/
45713f6ff2041d3fdfae927b82488db8-[]Abstract.html.

14


---Page Break---
Blake E. Woodworth, Kumar Kshitij Patel, Sebastian U. Stich, Zhen Dai, Brian Bullins, H. Brendan
McMahan, Ohad Shamir, and Nathan Srebro. Is local SGD better than minibatch SGD? In
Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July
2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 10334–
10343. PMLR, 2020b. URL http://proceedings.mlr.press/v119/woodworth20a.html.

Zhaohui Yang, Mingzhe Chen, Kai-Kit Wong, H. Vincent Poor, and Shuguang Cui. Federated learning
for 6g: Applications, challenges, and opportunities. Engineering, 8:33–41, 2022. ISSN 2095-
8099. doi: https://doi.org/10.1016/j.eng.2021.12.002. URL https://www.sciencedirect.
com/science/article/pii/S2095809921005245.

Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan
Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep
learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019.

Chulhee Yun, Shashank Rajput, and Suvrit Sra. Minibatch vs local SGD with shuffling: Tight
convergence bounds and beyond. arXiv preprint arXiv:2110.10342, abs/2110.10342, 2021. URL
https://arXiv.org/abs/2110.10342.

15


---Page Break---
Contents

1
Introduction
1

1.1
Communication compression . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

1.2
Random Reshuffling
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

1.3
Can Communication Compression and Random Reshuffling be Friends? . . . . . .
3

1.4
Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3

2
Algorithms and convergence theory
4

2.1
Algorithm Q-RR . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
4

2.2
Algorithm DIANA-RR . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6

2.3
Algorithms with Local Steps . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
7

3
Experiments
9

4
Conclusion
10

A Extra Related Works
18

B
Experimental details
18

B.1
Logistic Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

B.1.1
Experiment 1: Comparison of the Proposed Non-Local Methods with Exist-
ing Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

B.1.2
Experiment 2: Comparison of the Proposed Local Methods with Existing
Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

B.1.3
Experiment 3: Comparison of DIANA-RR with EF21 and DIANA . . . . .
22

B.2
Training Deep Neural Network model: ResNet-18 on CIFAR-10 . . . . . . . . . .
22

B.2.1
Computing Environment . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

B.2.2
Loss Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

B.2.3
Dataset and Metric . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

B.2.4
Tuning Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

B.2.5
Optimization-Based Fine-Tuning for Pretrained ResNet-18. . . . . . . . .
26

B.2.6
Experiments
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

B.3
Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27

C Missing Proofs for Q-RR
29

C.1
Shuffle Radius Clarification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

C.2
Proof of Theorem 2.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

C.3
Non-Strongly Convex Summands
. . . . . . . . . . . . . . . . . . . . . . . . . .
32

D Missing Proofs for DIANA-RR
40

D.1
Proof of Theorem 2.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
40

D.2
Non-Strongly Convex Summands
. . . . . . . . . . . . . . . . . . . . . . . . . .
44

16


---Page Break---
E Theoretical Results for Q-NASTYA and DIANA-NASTYA
52

F
Missing Proofs for Q-NASTYA
53

G Missing Proofs for DIANA-NASTYA
56

H Alternative Analysis of Q-NASTYA
61

I
Alternative Analysis of DIANA-NASTYA
64

J
Partial Participation for Method with Local Steps
67

J.1
Analysis of Q-NASTYA with Partial Participation . . . . . . . . . . . . . . . . . .
67

J.2
Analysis of DIANA-NASTYA with Partial Participation . . . . . . . . . . . . . . .
72

17


---Page Break---
A
Extra Related Works

Federated optimization has been the subject of intense study, with many open questions even in the
setting when all clients have identical data [Woodworth et al., 2020b,a, 2021]. The FedAvg algorithm
(also known as Local SGD) has also been a subject of intense study, with tight bounds obtained only
very recently by Glasgow et al. [2022]. It is now understood that using many local steps adds bias to
distributed SGD, and hence several methods have been developed to mitigate it, e.g. [Karimireddy
et al., 2020, Murata and Suzuki, 2021], see the work of Gorbunov et al. [2021] for a unifying lens on
many variants of Local SGD. Note that despite the bias, even vanilla FedAvg/Local SGD still reduces
the overall communication overhead in practice [Ortiz et al., 2021].

The success of RR in the single-machine setting has inspired several recent methods that use it as a
local update method as part of distributed training: Mishchenko et al. [2021] developed a distributed
variant of random reshuffling, FedRR. FedRR uses RR as a local client update method in lieu of SGD.
They show that FedRR can improve upon the convergence of Local SGD when the number of local
steps is fixed as the local dataset size, i.e. when H = n. Yun et al. [2021] study the same method
under the name Local RR under a more restrictive assumption of bounded inter-machine gradient
deviation and show that by varying H to be smaller than n better rates can be obtained in this setting
than the rates of Mishchenko et al. [2021]. Other work has explored more such combinations between
RR and distributed training algorithms [Huang et al., 2021, Malinovsky et al., 2022, Horváth et al.,
2022].

There are several methods that combine compression or quantization and local steps: both Basu et al.
[2019] and Reisizadeh et al. [2020] combined Local SGD with quantization and sparsification, and
Haddadpour et al. [2021] later improved their results using a gradient tracking method, achieving
linear convergence under strong convexity. In parallel, Mitra et al. [2021] also developed a variance-
reduced method, FedLin, that achieves linear convergence under strong convexity despite using local
steps and compression. The paper most related to our work is [Malinovsky and Richtárik, 2022]
in which the authors combine iterate compression, random reshuffling, and local steps. We study
gradient compression instead, which is a more common form of compression in both theory and
practice [Kairouz et al., 2019]. We compare our results against [Malinovsky and Richtárik, 2022] and
show we obtain better rates compared to their work.

B
Experimental details

In this section, we provide missing details on the experimental setting from Section 3. The codes
are provided in the following anonymous repository: https://anonymous.4open.science/r/
diana_rr-[]B0A5.

B.1
Logistic Regression

To confirm our theoretical results we conducted several numerical experiments on binary classification
problem with L2 regularized logistic regression of the form

min
x∈Rd

"

f(x)
def
=
1
M

M
X

m=1

1
nm

nm
X

i=1
fm,i

#

,
(10)

where fm,i
def
= log
 
1 + exp(−ymia⊤
mix)

+ λ∥x∥2
2 (ami, ymi) ∈Rd× ∈{−1, 1}, i = 1, . . . , nm
are the training data samples stored on machines m = 1, . . . , M, and λ > 0 is a regularization
parameter. In all experiments, for each method, we used the largest stepsize allowed by its theory
multiplied by some individually tuned constant multiplier. For better parallelism, each worker m
uses mini-batches of size ≈0.1nm. In all algorithms, as a compression operator Q, we use Rand-k
[Beznosikov et al., 2020] with fixed compression ratio k/d ≈0.02, where d is the number of features
in the dataset.

Hardware and Software.
All algorithms were written in Python 3.8. We used three different CPU
cluster node types:

1. AMD EPYC 7702 64-Core;

18


---Page Break---
2. Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz;

3. Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz.

Datasets.
The datasets were taken from open LibSVM library Chang and Lin [2011], sorted in
ascending order of labels, and equally split among 20 machines \clients\workers. The remaining
part of size N −20 · ⌊N/20⌋was assigned to the last worker, where N = PM
m=1 nm is the total size
of the dataset. A summary of the splitting and the data samples distribution between clients can be
found in Tables 2, 3, 4, 5.

Table 2: Summary of the datasets and splitting of the data samples among clients.
Dataset
M
N (dataset size)
d (# of features)
nm (# of datasamples per client)

mushrooms
20
8120
112
406
w8a
20
49749
300
2487
a9a
20
32560
123
1628

Table 3: Partition of the mushrooms dataset among clients.
Client’s №
# of datasamples of class "-1"
# of datasamples of class "+1"

1 – 9
406
0
10
262
144
11 – 19
0
406
20
0
410

Table 4: Partition of the w8a dataset among clients.
Client’s №
# of datasamples of class "-1"
# of datasamples of class "+1"

1 – 19
2487
0
20
1017
1479

Table 5: Partition of the a9a dataset among clients.
Client’s №
# of datasamples of class "-1"
# of datasamples of class "+1"

1 – 14
1628
0
15
1328
300
16 – 19
0
1628
20
0
1629

Hyperparameters.
Regularization parameter λ was chosen individually for each dataset to guar-
antee the condition number L/µ to be approximately 104, where L and µ are the smoothness and
strong-convexity constants of function f. For the chosen logistic regression problem of the form
(10), smoothness and strong convexity constants L, Lm, Li,m, µ, eµ of functions f, fm and f i
m were
computed explicitly as

19


---Page Break---
L
=
λmax

 
1
M

M
X

m=1

1
4nm
A⊤
mAm + 2λI

!

Lm
=
λmax


1
4nm
A⊤
mAm + 2λI


Li,m
=
λmax

1

4amia⊤
mi + 2λI


µ
=
2λ
eµ
=
2λ,

where Am is the dataset associated with client m, and ami is the i-th row of data matrix Am. In
general, the fact that f is L-smooth with

L ≤1

M

M
X

m=1

1
nm

nm
X

i=1
Li,m

follows from the Li,m-smoothness of f i
m (see Assumption 3).

In all algorithms, as a compression operator Q, we use Rand-k as a canonical example of unbiased
compressor with relatively bounded variance, and fix the compression parameter k = ⌊0.02d⌋, where
d is the number of features in the dataset.

In addition, in all algorithms, for all clients m = 1, . . . , M, we set the batch size for the SGD
estimator to be bm = ⌊0.1nm⌋, where nm is the size of the local dataset.

The summary of the values L, Lm, Li,m Lmax, µ, bm and k for each dataset can be found in Table 6.

Table 6: Summary of the hyperparameters.
Dataset
L
Lmax
µ
λ
k
bm (batchsize)

mushrooms
2.59
5.25
2.58 · 10−4
1.29 · 10−4
2
40
w8a
0.66
28.5
6.6 · 10−5
3.3 · 10−5
6
248
a9a
1.57
3.5
1.57 · 10−4
7.85 · 10−5
2
162

In all experiments, we follow constant stepsize strategy within the whole iteration procedure. For each
method, we set the largest possible stepsize predicted by its theory multiplied by some individually
tuned constant multiplier. For a more detailed explanation of the tuning routine, see Sections B.1.1
and B.1.2.

SGD implementation.
We considered two approaches to minibatching: random reshuffling and
with-replacement sampling. In the first, all clients m = 1, . . . , M independently permute their local
datasets and pass through them within the next subsequent ⌊nm

bm ⌋steps. In our implementations
of Q-RR, Q-NASTYA and DIANA-NASTYA, all clients permuted their datasets in the beginning of
every new epoch, whereas for the DIANA-RR method they do so only once in the beginning of the
iteration procedure. Second approach of minibatching is called with-replacement sampling, and it
requires every client to draw bm data samples from the local dataset uniformly at random. We used
this strategy in the baseline algorithms (QSGD, DIANA, FedCOM and FedPAQ) we compared our
proposed methods to.

Experimental setup.
To compare the performance of methods within the whole optimization
process, we track the functional suboptimality metric f(xt) −f(x⋆) that was recomputed after each
epoch. For each dataset, the value f(x⋆) was computed once at the preprocessing stage with 10−16
tolerance via conjugate gradient method. We terminate our algorithms after performing 5000 epochs.

B.1.1
Experiment 1: Comparison of the Proposed Non-Local Methods with Existing Baselines

In our first experiment (see Figure 1), we compare Q-RR, DIANA-RR, and DIANA-RR-1S with
classical baselines (QSGD [Alistarh et al., 2017], DIANA [Mishchenko et al., 2019b]) that use

20


---Page Break---
1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

mushrooms; Rand-2

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

w8a; Rand-6

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

1000
3000
5000
Data passes

10 7

10 5

10 3

10 1

f(xt)
f(x )

a9a; Rand-2

QSGD
Q-RR
DIANA
DIANA-RR
DIANA-RR-1S

Figure 5: Non-local methods

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

mushrooms; Rand-2

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

w8a; Rand-6

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

1000
3000
5000
Data passes

10
3

10
1

f(xt)
f(x )

a9a; Rand-2

Q-NASTYA
DIANA-NASTYA
FedCOM
FedPAQ

Figure 6: Local methods

Figure 7: The comparison of the proposed methods (Q-NASTYA, DIANA-NASTYA, Q-RR, DIANA-RR),
DIANA-RR-1S (a modification of DIANA-RR), and existing baselines (QSGD, DIANA, FedCOM,
FedPAQ). All methods use tuned stepsizes and the Rand-k compressor.

a with-replacement mini-batch SGD estimator. DIANA-RR-1S is a memory-friendly version of
DIANA-RR that stores and uses a single shift ht,m on the worker side rather than n individual shifts

hπi
m
t,m. Figure 1 illustrates that Q-RR exhibits similar behavior to QSGD, with both methods be-
ing slower than DIANA methods across all considered datasets. DIANA-RR-1S and DIANA show
comparable convergence rates, indicating that random reshuffling alone, without introducing ad-
ditional shifts, does not make a significant difference. Finally, DIANA-RR achieves the best rate
among all considered non-local methods, efficiently reducing the variance and reaching the lowest
functional sub-optimality tolerance. These experimental results align perfectly with our theoret-
ical analysis. For each of the considered non-local methods, we take the stepsize as the largest
one predicted by the theory premultiplied by the individually tuned constant factor from the set
{0.000975, 0.00195, 0.0039, 0.0078, 0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64,
128, 256, 512, 1024, 2048, 4096} .

Therefore, for each local method on every dataset, we performed 20 launches to find the stepsize
multiplier showing the best convergence behavior (the fastest reaching the lowest possible level of
functional suboptimality f(xt) −f(x⋆)).

Theoretical stepsizes for methods Q-RR and DIANA-RR are provided by the Theorems 2.1 and 2.2,
whereas stepsizes for QSGD and DIANA were taken from the paper Gorbunov et al. [2020].

B.1.2
Experiment 2: Comparison of the Proposed Local Methods with Existing Baselines

The second experiment shows that DIANA-based method can significantly outperform in practice
when one applies it to local methods as well. In particular, whereas Q-NASTYA shows comparative
behavior as existing methods FedCOM [Haddadpour et al., 2021], FedPAQ [Reisizadeh et al., 2020]
in all considered datasets, DIANA-NASTYA noticeably outperforms other methods.

In this set of experiments, we tuned stepsizes similarly to the non-local methods. However, for
algorithms Q-NASTYA, DIANA-NASTYA, and FedCOM we needed to independently adjust the client
and server stepsizes, leading to a more extensive tunning routine.

As before, for each local method on every dataset, tuned client and server stepsizes are defined by the
theoretical one and adjusted constant multiplier. Theoretical stepsizes for methods Q-NASTYA and
DIANA-NASTYA are given by the Theorems E.1 and E.2, whereas FedCOM and FedPAQ stepsizes
were taken from the papers by Haddadpour et al. [2021] and Reisizadeh et al. [2020] respectively.

21


---Page Break---
We now list all the considered multipliers of client and server stepsizes for every method (i.e. γ and η
respectively):

• Q-NASTYA:

– Multipliers for γ : {0.000975, 0.00195, 0.0039, 0.0078, 0.0156, 0.0312, 0.0625, 0.125,
0.25, 0.5, 1, 2, 4, 8, 16, 32, 64,128};
– Multipliers for η : {0.0039, 0.0078, 0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,
16, 32, 64, 128}.

• DIANA-NASTYA:

– Multipliers for γ and η : {0.000975, 0.00195, 0.0039, 0.0078, 0.0156, 0.0312, 0.0625,
0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64,128};

• FedCOM:

– Multipliers for γ : {0.0312, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256,
512,1024, 2048, 4096, 8192, 16384, 32768 };
– Multipliers for η : {0.000975, 0.00195, 0.0039, 0.0078, 0.0156, 0.0312, 0.0625, 0.125,
0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128}.

• FedPAQ:

– Multipliers for γ : {0.00195, 0.0039, 0.0078, 0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.5,
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
131072, 262144, 524288, 1048576 }.

For example, to find the best pair (γ, η) for FedCOM method on each dataset, we performed 378
launches. A similar subroutine was executed for all algorithms on all datasets independently.

B.1.3
Experiment 3: Comparison of DIANA-RR with EF21 and DIANA

In our third experiment (see Figure 8), we compared DIANA-RR with DIANA and EF21-SGD
[Fatkhullin et al., 2021]. The EF21-SGD is the state-of-the-art algorithm for contractive compressors
in distributed non-convex settings. All compared algorithms used a with-replacement mini-batch
SGD estimator, consistent with the setup in Section B.1.1. However, in this experiment contrast with
Section B.1.1, we employed reliable theoretical step sizes that ensure guaranteed convergence.

The EF21 algorithm family is designed for usage with contraction compressors in non-convex
optimization for L-smooth objective functions in for of Equation 1. For scenarios involving un-
biased compressors such as Rand-k, the EF21-SGD can be adapted through scaling [Fatkhullin
et al., 2021]. More specifically, an unbiased compressor C(x) : Rd →Rd which satisfied Assump-
tion 1 via applying the transformation C′(x)
def
= (ω + 1)−1 · C(x) yields a contraction compressor
E

∥C′(x) −x∥2
≤(1 −α)∥x∥2, ∀x ∈Rd with α = 1/ω+1. In particular, this procedure makes
Rand-k compatible with the EF21 algorithm family. In this experiment, we implemented EF21-SGD
following the refined analysis from [Richtárik et al., 2024], which offers a stricter better convergence
guarantee through improved bounds on the theoretical step size compared to [Fatkhullin et al., 2021].

As shown in Figure 8 EF21-SGD does not perform fast enough in scenarios involving using unbiased
compressors for strongly-convex optimization problems compared to DIANA-RR and DIANA.

B.2
Training Deep Neural Network model: ResNet-18 on CIFAR-10

Since random reshuffling is a very popular technique in training neural networks, it is natural to test
the proposed methods on such problems. Therefore, in the second set of experiments, we consider
training ResNet-18 [He et al., 2016] model on the CIFAR10 dataset Krizhevsky and Hinton [2009].
To conduct these experiments we use FL_PyTorch simulator [Burlachenko et al., 2021].

The main goal of this experiment is to verify the phenomenon observed in Experiment 1 on the
training of a deep neural network. That is, we tested Q-RR, QSGD, DIANA, and DIANA-RR in
the distributed training of ResNet-18 on CIFAR10, see Figure 9. As in the logistic regression
experiments, we observe that (i) Q-RR and QSGD behave similarly and (ii) DIANA-RR outperforms
DIANA.

22


---Page Break---
0
1000
2000
3000
4000
5000
Data passes

0.01

0.1

1

f(xt) −f ∗

mushrooms; RandK[K=2]

DIANA
DIANA-RR
EF21-SGD

0
1000
2000
3000
4000
5000
Data passes

0.01

0.1

1

f(xt) −f ∗

a9a; RandK[K=2]

DIANA
DIANA-RR
EF21-SGD

Figure 8: The comparison of the proposed variance-reduced DIANA-RR and baselines DIANA, EF21-
SGD. All algorithms use theoretical step-sizes, Rand-k compressor, number of workers is 20.

0
2
4
6
8
#bits/n
1e7

20

40

60

Top1 acc (validation)

QSGD =  0.001
QSGD-RR =  0.001

0
1
2
3
#bits/n
1e11

25

50

75

Top1 acc (validation)

DIANA = 1.0
DIANA-RR = 1.0

Figure 9: The comparison of Q-RR, QSGD, DIANA, and DIANA-RR on the task of training ResNet-18
on CIFAR-10 with n = 10 workers. Top-1 accuracy on test set is reported. Stepsizes were tuned and
workers used Rand-k compressor with k/d ≈0.05.

To illustrate the behavior of the proposed methods in training Deep Neural Networks (DNN), we
consider the ResNet-18 [He et al., 2016] model. This model is used for image classification,
feature extraction for image segmentation, object detection, image embedding, and image captioning.
We train all layers of ResNet-18 model meaning that the dimension of the optimization problem
equals d = 11, 173, 962. During the training, the ResNet-18 model normalizes layer inputs via
exploiting 20 Batch Normalization [Ioffe and Szegedy, 2015] layers that are applied directly before
nonlinearity in the computation graph of this model. Batch normalization (BN) layers add 9600
trainable parameters to the model. Besides trainable parameters, a BN layer has its internal state
that is used for computing the running mean and variance of inputs due to its own specific regime of
working. We use He initialization [He et al., 2015].

B.2.1
Computing Environment

We performed numerical experiments on a server-grade machine running Ubuntu 18.04 and Linux
Kernel v5.4.0, equipped with 16-cores (2 sockets by 16 cores per socket) 3.3 GHz Intel Xeon, and
four NVIDIA A100 GPU with 40GB of GPU memory. The distributed environment is simulated
in Python 3.9 via using the software suite FL_PyTorch [Burlachenko et al., 2021] that serves for
carrying complex Federate Learning experiments. FL_PyTorch allowed us to simulate the distributed
environment in the local machine. Besides storing trainable parameters per client, this simulator
stores all not trainable parameters including BN statistics per client.

B.2.2
Loss Function

Training of ResNet-18 can be formalized as problem (1) with the following choice of f i
m

fm(x) =
1
|nm|

|nm|
X

j=1
CE(b(j), g(a(j), x)),
(11)

23


---Page Break---
where CE(p, q)
def
= −P#classes
k=1
pi·log(qi) with agreement 0·log(0) = 0 is a standard cross-entropy
loss, function g : R28×28 × Rd →[0, 1]#classes is a neural network taking image a(j) and vector
of parameters x as an input and returning a vector in probability simplex, and nm is the size of the
dataset on worker m.

B.2.3
Dataset and Metric

In our experiments, we used CIFAR10 dataset Krizhevsky and Hinton [2009]. The dataset consists of
input variables ai ∈R28×28×3, and response variables bi ∈{0, 1}10 and is used for training 10-way
classification. The sizes of training and validation set are 5 × 104 and 104 respectively. The training
set is partitioned heterogeneously across 10 clients. To measure the performance, we evaluate the loss
function value f(x), norm of the gradient ∥∇f(x)∥2 and the Top-1 accuracy of the obtained model
as a function of passed epochs and the normalized number of bits sent from clients to the server.

B.2.4
Tuning Process

In this set of experiments, we tested QSGD [Alistarh et al., 2017], Q-RR (Algorithm 1), DI-
ANA [Mishchenko et al., 2019a] and DIANA-RR (Algorithm 2) algorithms. For all algorithms,
we tuned the strategy ∈{A, B, C} of decaying stepsize model via selecting the best in terms of the
norm of the full gradient on the train set in the final iterate produced after 20000 rounds. The stepsize
policies are described below.

A. Stepsizes decaying as inverse square root of the number epochs

γe =





γinit ·
1
√e −s + 1,
if e ≥s,

γinit,
if e < s,

where γe denotes the stepsize used during epoch e + 1, s is a fixed shift.

B. Stepsizes decaying as inverse of number epochs

γ =





γinit ·
1
e −s + 1,
if e ≥s,

γinit,
if e < s.

C. Fixed stepsize

γ = γinit.

We say that the algorithm passed e epochs if the total number of computed gradient oracles lies
between e PM
m=1 nm and (e + 1) PM
m=1 nm. For each algorithm the used stepsize γinit and shift
parameter s were tuned via selecting from the following sets:

γinit ∈γset
def
= {4.0, 3.75, 3.00, 2.5, 2.00, 1.25, 1.0, 0.75, 0.5, 0.25,
0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0006}.

s ∈sset
def
= {50, 100, 200, 500, 1000}.

In all tested methods, clients independently apply Rand-k compression with carnality k = ⌊0.05d⌋.
Computation for all gradient oracles is carried out in single precision float (FP64) arithmetic.

24


---Page Break---
0
200
400
600
800
1000
Data passes

20

40

60

80

Top1 acc (validation)

QSGD = 3.0
QSGD-RR = 3.0

0
200
400
600
800
1000
Data passes

100

102

|| f(xt)|| (train)

QSGD = 3.0
QSGD-RR = 3.0

0
200
400
600
800
1000
Data passes

100

101

102

f(xt) (train)

QSGD = 3.0
QSGD-RR = 3.0

0.0
0.5
1.0
1.5
2.0
#bits/n
1e11

20

40

60

80

Top1 acc (validation)

QSGD = 3.0
QSGD-RR = 3.0

0.0
0.5
1.0
1.5
2.0
#bits/n
1e11

10
1

100

101

|| f(xt)|| (train)

QSGD = 3.0
QSGD-RR = 3.0

0.0
0.5
1.0
1.5
2.0
#bits/n
1e11

100

101

102

f(xt) (train)

QSGD = 3.0
QSGD-RR = 3.0

Figure 10

Figure 11: Comparison of QSGD and Q-RR in the training of ResNet-18 on CIFAR-10, with n = 10
workers. Here (a) and (d) show Top-1 accuracy on test set, (b) and (e) – norm of full gradient on the
train set, (c) and (f) – loss function value on the train set. Stepsizes and decay shift has been tuned
from sset and γset based on minimum achievable value of loss function on the train set.

25


---Page Break---
0
100
200
300
400
Data passes

25

50

75

Top1 acc (validation)

DIANA = 1.0
DIANA-RR = 1.0

0
100
200
300
400
500
600
Data passes

100

101

|| f(xt)|| (train)

DIANA = 1.0
DIANA-RR = 1.0

0
200
400
600
Data passes

10
1

100

101

f(xt) (train)

DIANA = 1.0
DIANA-RR = 1.0

0
1
2
3
#bits/n
1e11

25

50

75

Top1 acc (validation)

DIANA = 1.0
DIANA-RR = 1.0

0
1
2
3
#bits/n
1e11

100

101

|| f(xt)|| (train)

DIANA = 1.0
DIANA-RR = 1.0

0
1
2
3
#bits/n
1e11

10
1

100

101

f(xt) (train)

DIANA = 1.0
DIANA-RR = 1.0

Figure 12: Comparison of DIANA and DIANA-RR in the training of ResNet-18 on CIFAR-10, with
n = 10 workers. Here (a) and (d) show Top-1 accuracy on test set, (b) and (e) – norm of full gradient
on the train set, (c) and (f) – loss function value on the train set. Stepsizes and decay shift has been
tuned from sset and γset based on minimum achievable value of loss function on the train set. For
both algorithms stepsize is fixed. For both algorithms stepsize is decaying according to srategy B.

B.2.5
Optimization-Based Fine-Tuning for Pretrained ResNet-18.

In this setting, we trained ResNet-18 image classification in a distributed way across n = 10 clients.
In this experiment, we have trained only the last linear layer.

Next, we have turned off batch normalization. Turning off batch normalization implies that the
computation graph of NN g(a, x) with weights of NN denoted as x is a deterministic function and
does not include any internal state.

The loss function is a standard cross-entropy loss augmented with extra ℓ2-regularization α∥x∥2/2
with α = 0.0001. Initially used weights of NN are pretrained parameters after training the model on
ImageNet.

The dataset distribution across clients has been set in a heterogeneous manner via presorting dataset
D by label class and after this, it was split across 10 clients.

The comparison of stepsizes policies used in QSGD and Q-RR is presented in Figure 14. The behavior
of the algorithms with best tuned step sizes is presented in Figure 13. These results demonstrate that
in this setting there is no real benefit of using Q-RR in comparison to QSGD.

B.2.6
Experiments

The comparison of QSGD and Q-RR is presented in Figure 11. In particular, Figure 9 shows that in
terms of the convergence to stationary points both algorithms exhibit similar behavior. However, Q-
RR has better generalization and in fact, converges to the better loss function value. This experiment
demonstrates that Q-RR with manually tuned stepsize can be better compared to QSGD in terms
of the final quality of obtained Deep Learning model. For QSGD the tuned meta parameters are:

26


---Page Break---
γinit = 3.0,s = 200, strategy = B. For QSGD-RR tuned meta parameters are: γinit = 3.0, s = 1000,
strategy = B.

The results of comparison of DIANA and DIANA-RR are presented in Figure 12. For DIANA the tuned
meta parameters are: γinit = 1.0,s = 0, strategy = C and for DIANA-RR tuned meta parameters are:
γinit = 1.0, s = 0, strategy = C. These results show that DIANA-RR outperforms DIANA in terms of
all reported metrics.

B.3
Discussion

More about used arithmetics.
We used FP64 (IEEE 754) due to its superior numerical stability
compared to FP32, FP16, and BFloat16. While FP32 and FP16 are commonly used for inference
tasks, the choice of precision for training depends on the specific requirements of the task. In certain
cases, FP32 may be sufficient, but for others, FP64 is necessary to ensure stability.

The performance gain from switching from FP64 to FP32 can indeed vary based on the GPU model.
For instance, the NVIDIA A100 40GB GPU used in our experiments offers approximately a two-fold
increase in computational throughput with FP32 compared to FP64. The specific architecture of
the GPU influences the choice of precision, and these characteristics can differ across various GPU
models and updates.

The computational burden.
The primary focus of the paper is to highlight the fundamental
complexities and limits of algorithmic behavior. The experiments presented in our paper are intended
for illustrative purposes.

The computational demands of our work are significant.
Performing experiments beyond
ResNet-18/CIFAR-10/FP64 with 10 clients is near the limit of what is feasible with our com-
putational resources. In our simulation involving 10 clients sharing a common dataset, we ran 2000
rounds/epochs for fine-tuning. Based on an estimate of 2 minutes per epoch, the total computation
time would be approximately 66 hours per run (2 minutes/epoch × 2000 epochs = 66 hours). Taking
into account the grid search with 18 preset learning rates, 5 sets of decay parameters, and 4 algorithms,
the total estimated computation time would be around 23760 hours (66 hours × 18 × 5 × 4). This
represents a substantial amount of computation time. Therefore, conducting a comprehensive compar-
ison involving four algorithms with an extensive grid of hyperparameters is already challenging for
models larger than ResNet-18 on CIFAR10. To cover 23760 hours of training would indeed require
approximately 40 GPUs running continuously for about 25 days. Nonetheless, we have conducted
numerous experiments to ensure a thorough and fair comparison.

Training in overparameterized regime.
During training image classification Convolution Neural
Networks, we got two results for QSGD-RR as an improvement of QSGD. During training only the
last layer (see Fig. 13) there are no benefits QSGD-RR, but QSGD does not behave worse.

When training the whole network (Fig. 11), the results suggest that Q-RR is much better than Q-SGD.
Although we do not have formal proof explaining this phenomenon, we conjecture that this can be
related to the significant overparameterization occurring during the training of a large model on a
relatively small dataset. That is, the model can almost perfectly fit the training data on all clients,
leading to a decrease in the heterogeneity parameter. In this case, there is no need for shifts since
the variance coming from compression naturally goes to zero, and the complexities of QSGD and
DIANA match (see Table 1). In this situation, Q-RR performs better than QSGD since the compression
does not spoil the convergence of RR. Therefore, DIANA-type shifts are not always necessary to get
improvements. Nevertheless, we conjecture that they are necessary when the datasets are larger and
more complex.

27


---Page Break---
0
2
4
6
8
#bits/n
1e7

20

40

60

Top1 acc (validation)

QSGD =  0.001
QSGD-RR =  0.001

0
2
4
6
8
#bits/n
1e7

1

2

|| f(xt)|| (train)

QSGD =  0.001
QSGD-RR =  0.001

0
2
4
6
8
#bits/n
1e7

2.0

2.5

f(xt) (train)

QSGD =  0.001
QSGD-RR =  0.001

Figure 13: Comparison of QSGD and Q-RR in the training of the last linear layer of ResNet-18
on CIFAR-10, with n = 10 workers. Here (a) shows Top-1 accuracy on test set, (b) – norm of full
gradient on the train set, (c) – loss function value on the train set. Stepsizes and decay shift has been
tuned from sset and γset based on minimum achievable value of loss function on the train set. Both
algorithms used fixed stepsize during training.

0
1
2
3
4
5
6
7
8
#bits/n
1e7

10

20

30

40

50

Top1 acc (validation)

QSGD ( = 0.0006)

QSGD ( = 0.003)

QSGD ( = 0.001)
QSGD ( = 0.01)

QSGD ( = 0.03)
QSGD ( = 0.06)

QSGD ( = 0.5)

QSGD ( = 0.1)
QSGD ( = 0.2)

QSGD ( = 0.25)
QSGD ( = 0.75)

QSGD ( = 1.0)

QSGD ( = 1.25)
QSGD ( = 2.00)

QSGD ( = 2.5)

QSGD ( = 3.00)
QSGD ( = 3.75)

QSGD ( = 4.00)

0
1
2
3
4
5
6
7
8
#bits/n
1e7

10

20

30

40

50

Top1 acc (validation)

QSGD-RR ( = 0.0006) =  0.0006

QSGD-RR ( = 0.003) =  0.003

QSGD-RR ( = 0.001) =  0.001
QSGD-RR ( = 0.01) =  0.01

QSGD-RR ( = 0.03) =  0.03
QSGD-RR ( = 0.06) =  0.06

QSGD-RR ( = 0.1) =  0.1

QSGD-RR ( = 0.2) =  0.2
QSGD-RR ( = 0.25) =  0.25

QSGD-RR ( = 0.75) =  0.75

QSGD-RR ( = 1.0) =  1.0
QSGD-RR ( = 1.25) =  1.25

QSGD-RR ( = 2.00) =  2.0
QSGD-RR ( = 2.5) =  2.5

QSGD-RR ( = 3.00) =  3.0

QSGD-RR ( = 3.75) =  3.75
QSGD-RR ( = 4.00) =  4.0

0
1
2
3
4
5
6
7
8
#bits/n
1e7

101

102

f(xt) (train)

QSGD ( = 0.0006)

QSGD ( = 0.003)
QSGD ( = 0.001)

QSGD ( = 0.01)

QSGD ( = 0.03)
QSGD ( = 0.06)

QSGD ( = 0.5)
QSGD ( = 0.1)

QSGD ( = 0.2)

QSGD ( = 0.25)
QSGD ( = 0.75)

QSGD ( = 1.0)

QSGD ( = 1.25)
QSGD ( = 2.00)

QSGD ( = 2.5)

QSGD ( = 3.00)

QSGD ( = 3.75)
QSGD ( = 4.00)

0
1
2
3
4
5
6
7
8
#bits/n
1e7

101

102

f(xt) (train)

QSGD-RR ( = 0.0006) =  0.0006

QSGD-RR ( = 0.003) =  0.003
QSGD-RR ( = 0.001) =  0.001

QSGD-RR ( = 0.01) =  0.01
QSGD-RR ( = 0.03) =  0.03

QSGD-RR ( = 0.06) =  0.06

QSGD-RR ( = 0.1) =  0.1
QSGD-RR ( = 0.2) =  0.2

QSGD-RR ( = 0.25) =  0.25

QSGD-RR ( = 0.75) =  0.75
QSGD-RR ( = 1.0) =  1.0

QSGD-RR ( = 1.25) =  1.25
QSGD-RR ( = 2.00) =  2.0

QSGD-RR ( = 2.5) =  2.5
QSGD-RR ( = 3.00) =  3.0

QSGD-RR ( = 3.75) =  3.75

QSGD-RR ( = 4.00) =  4.0

0
1
2
3
4
5
6
7
8
#bits/n
1e7

100

|| f(xt)|| (train)

QSGD ( = 0.0006)

QSGD ( = 0.003)

QSGD ( = 0.001)
QSGD ( = 0.01)

QSGD ( = 0.03)

QSGD ( = 0.06)

QSGD ( = 0.5)

QSGD ( = 0.1)
QSGD ( = 0.2)

QSGD ( = 0.25)

QSGD ( = 0.75)

QSGD ( = 1.0)

QSGD ( = 1.25)

QSGD ( = 2.00)
QSGD ( = 2.5)

QSGD ( = 3.00)

QSGD ( = 3.75)

QSGD ( = 4.00)

0
1
2
3
4
5
6
7
8
#bits/n
1e7

100

|| f(xt)|| (train)

QSGD-RR ( = 0.0006) =  0.0006

QSGD-RR ( = 0.003) =  0.003

QSGD-RR ( = 0.001) =  0.001
QSGD-RR ( = 0.01) =  0.01

QSGD-RR ( = 0.03) =  0.03

QSGD-RR ( = 0.06) =  0.06

QSGD-RR ( = 0.1) =  0.1

QSGD-RR ( = 0.2) =  0.2
QSGD-RR ( = 0.25) =  0.25

QSGD-RR ( = 0.75) =  0.75

QSGD-RR ( = 1.0) =  1.0

QSGD-RR ( = 1.25) =  1.25

QSGD-RR ( = 2.00) =  2.0

QSGD-RR ( = 2.5) =  2.5
QSGD-RR ( = 3.00) =  3.0

QSGD-RR ( = 3.75) =  3.75

QSGD-RR ( = 4.00) =  4.0

Figure 14: Comparison of QSGD and Q-RR in the training of the last linear layer of ResNet-18 on
CIFAR-10, with n = 10 workers. Here (a) and (b) show Top-1 accuracy on test set, (c) and (d) – loss
function value on the train set, (e) and (f) – norm of full gradient on the train set. Stepsizes and decay
shift has been tuned from sset and γset based on minimum achievable value of loss function on the
train set. During training stepsize was fixed. Batch Normalization was turned off.

28


---Page Break---
C
Missing Proofs for Q-RR

In the main part of the paper, we introduce Assumptions 3 and 4 for the analysis of Q-RR and
DIANA-RR. These assumptions can be refined as follows.

Assumption 5. Function f πi =
1
M
PM
i=1 f πi
m
m
: Rd →R is eL-smooth for all sets of permutations
π = (π1, . . . , πm) from [n] and all i ∈[n], i.e.,

max
i∈[n],π ∥∇f πi(x) −∇f πi(y)∥≤eL∥x −y∥
∀x, y ∈Rd.

Assumption 6. Function f πi =
1
M
PM
i=1 f πi
m
m
: Rd →R is eµ-strongly convex for all sets of
permutations π = (π1, . . . , πm) from [n] and all i ∈[n], i.e.,

min
i∈[n],π

n
f πi(x) −f πi(y) −⟨∇f i,π(y), x −y⟩
o
≥eµ

2 ∥x −y∥2
∀x, y ∈Rd.

Moreover, functions f i
1, f i
2, . . . , f i
M : Rd →R are convex for all i = 1, . . . , n.

We notice that Assumptions 3 and 4 imply Assumptions 5 and 6. In the proofs of the results for
Q-RR and DIANA-RR, we use Assumptions 5 in addition to Assumptions 3 and we use Assumption 6
instead of Assumption 4.

C.1
Shuffle Radius Clarification

Our results depend on the so-called shuffling radius proposed by Mishchenko et al. [2021]:

σ2
rad
def
= max
i

(
1
γ2M

M
X

m=1
EDf πi
m (xi
⋆, x⋆)

)

,

where xi+1
⋆
= xi
⋆−γ

M
PM
m=1 ∇f πi
m
m (x⋆).

One can think of the shuffling radius as a counterpart to the variance term in SGD. Both concepts
measure how much the algorithm’s performance can fluctuate near the optimal solution, but the
cause of these fluctuations is different: in SGD, it is due to random sampling, and in RR, it is due
to reshuffling. Additionally, Lemma 2.1 provides bounds for the shuffling radius — showing the
maximum and minimum possible values — based on the variance at the optimum, reinforcing the
shuffling radius as a useful way to understand how RR behaves. This relationship helps clarify how
the reshuffling process influences the algorithm’s path and its efficiency in reaching an optimal point.

C.2
Proof of Theorem 2.1

For convenience, we restate the theorem below.
Theorem C.1 (Theorem 2.1). Let Assumptions 1, 3, 5, 6 hold and 0 < γ ≤
1
eL+2 ω

M Lmax . Then, for

all T ≥0 the iterates produced by Q-RR satisfy

E∥xT −x⋆∥2 ≤(1 −γeµ)nT ∥x0 −x⋆∥2 + 2γ2σ2
rad
eµ
+ 2γω

eµM
 
ζ2
⋆+ σ2
⋆

,

where ζ2
⋆=
1
M

M
P

m=1
∥∇fm(x⋆)∥2, and σ2
⋆=
1
Mn

M
P

m=1

nP

i=1
∥∇f i
m(x⋆) −∇fm(x⋆)∥2.

Proof. Using xi+1
⋆
= xi
⋆−γ

M
PM
m=1 ∇f πi
m
m (x⋆) and line 7 of Algorithm 1, we get

∥xi+1
t
−xi+1
⋆
∥2
=

xi
t −xi
⋆−γ 1

M

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (x⋆)


2

=
xi
t −xi
⋆
2 −2γ

*
1
M

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (x⋆)

, xi
t −xi
⋆

+

+γ2

1
M

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (x⋆)


2

.

29


---Page Break---
Taking the expectation w.r.t. Q, we obtain

EQ

∥xi+1
t
−xi+1
⋆
∥2
=
xi
t −xi
⋆
2 −2γ

*
1
M

M
X

m=1


∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)

, xi
t −xi
⋆

+

+γ2EQ






1
M

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (x⋆)


2

.

In view of Assumption 1 and Eξ∥ξ −c∥2 = Eξ∥ξ −Eξξ∥2 + ∥Eξξ −c∥2, we have

EQ

∥xi+1
t
−xi+1
⋆
∥2
=
xi
t −xi
⋆
2 −2γ

M

M
X

m=1

D
∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆), xi
t −xi
⋆
E

+γ2EQ






1
M

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (xi
t)


2



+γ2

1
M

M
X

m=1


∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)


2

≤
xi
t −xi
⋆
2 −2γ

M

M
X

m=1

D
∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆), xi
t −xi
⋆
E

+γ2

1
M

M
X

m=1


∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)


2

+γ2ω

M 2

M
X

m=1

∇f πi
m
m (xi
t)

2
,

where in the last step we apply independence of Q

∇f πi
m
m (xi
t)

for m ∈[M]. Next, we use

three-point identity4 and obtain

EQ

∥xi+1
t
−xi+1
⋆
∥2
≤
xi
t −xi
⋆
2

−2γ

M

M
X

m=1


D
f
πim
m (xi
⋆, xi
t) + D
f
πim
m (xi
t, x⋆) −D
f
πim
m (xi
⋆, x⋆)


+γ2

1
M

M
X

m=1


∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)


2

+γ2ω

M 2

M
X

m=1

∇f πi
m
m (xi
t)

2
.

4For any differentiable function f : Rd →Rd we have: ⟨∇f(x)−∇f(y), x−z⟩= Df(z, x)+Df(x, y)−
Df(z, y).

30


---Page Break---
Applying eL-smoothness and convexity of
1
M
Pm
m=1 f πi
m
m , eµ-strong convexity of
1
M
Pm
m=1 f πi
m
m , and
Lmax-smoothness and convexity of f i
m, we get

EQ

∥xi+1
t
−xi+1
⋆
∥2
≤
(1 −γeµ)
xi
t −xi
⋆
2 −2γ

1 −eLγ
 1

M

M
X

m=1
D
f
πim
m (xi
t, x⋆)

+2γ 1

M

M
X

m=1
D
f
πim
m (xi
⋆, x⋆) + γ2ω

M 2

M
X

m=1

∇f πi
m
m (xi
t)

2

≤
(1 −γeµ)
xi
t −xi
⋆
2 −2γ

1 −eLγ
 1

M

M
X

m=1
D
f
πim
m (xi
t, x⋆)

+2γ 1

M

M
X

m=1
D
f
πim
m (xi
⋆, x⋆) + 2γ2ω

M 2

M
X

m=1

∇f πi
m
m (x⋆)

2

+2γ2ω

M 2

M
X

m=1

∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)

2
.

So, we get

EQ

∥xi+1
t
−xi+1
⋆
∥2
≤
(1 −γeµ)
xi
t −xi
⋆
2 + 2γ2ω

M 2

M
X

m=1

∇f πi
m
m (x⋆)

2

+2γ

M

M
X

m=1
D
f
πim
m (xi
⋆, x⋆)

−2γ

1 −γ

eL + 2ωLmax

M

 1

M

M
X

m=1
D
f
πim
m (xi
t, x⋆).

Taking the full expectation and using a definition of shuffle radius, 0 < γ ≤
1
(eL+2 ω

M Lmax), and

D
f
πim
m (xi
t, x⋆) ≥0, we obtain

E

∥xi+1
t
−xi+1
⋆
∥2
≤
(1 −γeµ) E
hxi
t −xi
⋆
2i
+ 2γ3σ2
rad + 2γ2ω

M 2

M
X

m=1
E
∇f πi
m
m (x⋆)

2

=
(1 −γeµ) E
hxi
t −xi
⋆
2i
+ 2γ3σ2
rad + 2γ2ω

M 2n

M
X

m=1

n
X

j=1

∇f j
m(x⋆)
2

≤
(1 −γeµ) E
hxi
t −xi
⋆
2i
+ 2γ3σ2
rad + 2γ2ω

M
 
ζ2
⋆+ σ2
⋆

.

Unrolling the recurrence in i, we derive

E

∥xt+1 −x⋆∥2
≤
(1 −γeµ)n E
h
∥xt −x⋆∥2i
+ 2γ3σ2
rad

n−1
X

j=0
(1 −γeµ)j

+2γ2ω

M
 
ζ2
⋆+ σ2
⋆
 n−1
X

j=0
(1 −γeµ)j.

Unrolling the recurrence in t, we derive

E

∥xT −x⋆∥2
≤
(1 −γeµ)nT ∥x0 −x⋆∥2 + 2γ3σ2
rad

T −1
X

t=0
(1 −γeµ)nt
n−1
X

j=0
(1 −γeµ)j

+2γ2ω

M
 
ζ2
⋆+ σ2
⋆
 nT −1
X

j=0
(1 −γeµ)nt
n−1
X

j=0
(1 −γeµ)j.

Since PnT −1
j=0 (1 −γeµ)j ≤
1
γeµ, we get the result.

31


---Page Break---
Corollary 5. Let the assumptions of Theorem C.1 hold and

γ = min

(
1
eL + 2 ω

M Lmax
,

s

εeµ
6σ2
rad
,
εeµM
6ω (ζ2⋆+ σ2⋆)

)

.
(12)

Then, Q-RR finds a solution with accuracy ε > 0 after the following number of communication
rounds:

e
O

 eL

eµ + ω

M
Lmax

eµ
+ ω

M
ζ2
⋆+ σ2
⋆
εeµ2
+
σrad
p

εeµ3

!

.

Proof. Theorem C.1 implies

E∥xT −x⋆∥2 ≤(1 −γeµ)nT ∥x0 −x⋆∥2 + 2γ2σ2
rad
eµ
+ 2γω

eµM
 
ζ2
⋆+ σ2
⋆

.
(13)

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper-bound each term from the right-hand side by ε/3. Thus, we get additional conditions on
γ:
2γ2σ2
rad
eµ
< ε

3,
2γω

eµM
 
ζ2
⋆+ σ2
⋆

< ε

3
and also the upper bound on the number of communication rounds nT

nT = e
O
 1

γeµ


.

Substituting (12), we get a final result.

C.3
Non-Strongly Convex Summands

In this section, we provide the analysis of Q-RR without using Assumptions 4, 6. Before we move
one to the proofs, we would like to emphasize that

xi+1
t
= xi
t −γ 1

M

M
X

m=1
Q

∇f πi
m
m (xi
t)

.

Then we have

xt+1 = xt −γ

n−1
X

i=0

1
M

M
X

m=1
Q

∇f πi
m
m (xi
t)

= xt −τ
1
Mn

n−1
X

i=0

M
X

m=1
Q

∇f πi
m
m (xi
t)

,

where τ = γn. For convenience, we denote

gt =
1
Mn

n−1
X

i=0

M
X

m=1
Q

∇f πi
m
m (xi
t)


allowing to write the update rule as xt+1 = xt −τgt.

Lemma C.1 (Lemma 1 from [Malinovsky et al., 2022]). For any k ∈[n], let ξπ1, . . . , ξπk be
sampled uniformly without replacement from a set of vectors {ξ1, . . . , ξn} and ¯ξπ be their average.
Then, it holds

E¯ξπ = ¯ξ,
E

∥¯ξπ −¯ξ∥2
=
n −k
k(n −1)σ2,
(14)

where ¯ξ = 1

n
Pn
i=1 ξi, ¯ξπ = 1

k
Pk
i=1 ξπi, σ2 = 1

n
Pn
i=1 ∥ξi −¯ξ∥2

Lemma C.2. Under Assumptions 1, 2, 3, 5, the following inequality holds

EQ [−2τ⟨gt, xt −x⋆⟩] ≤−τµ

2 ∥xt −x⋆∥2 −τ(f(xt) −f(x⋆)) + τ eL

n

n−1
X

i=0
∥xi
t −xt∥2.

32


---Page Break---
Proof. Using that EQ [gt] =
1
Mn
Pn−1
i=0
PM
m=1 ∇f πi
m
m (xi
t) and definition of h⋆, we get

−2τEQ [⟨gt, xt −x⋆⟩]
=
−1

Mn

n−1
X

i=0

M
X

m=1

D
∇f πi
m
m (xi
t), xt −x⋆
E

=
−1

Mn

M
X

m=1

n−1
X

i=0

D
∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆), xt −x⋆
E
.

Using three-point identity, we obtain

−2τEQ [⟨gt, xt −x⋆⟩]
=
−2τ

Mn

M
X

m=1

n−1
X

i=0


D
f
πim
m (xt, x⋆) + D
f
πim
m (x⋆, xi
t) −D
f
πim
m (xt, xi
t)


=
−2τDf(xt, x⋆) −2τ

n

n−1
X

i=0
Df πi(x⋆, xi
t) + 2τ

n

n−1
X

i=0
Df πi(xt, xi
t)

≤
−2τDf(xt, x⋆) + τ eL

n

n−1
X

i=0
∥xi
t −xt∥2,

where in the last inequality we apply eL-smoothness and convexity of each function f πi. Finally,
using µ-strong convexity of f, we finish the proof of the lemma.

Lemma C.3. Under Assumptions 1, 2, 3, 5, the following inequality holds

EQ

∥gt∥2
≤
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2
+ 4ω

Mn
 
ζ2
⋆+ σ2
⋆


+8

eL +
ω
MnLmax

(f(xt) −f(x⋆)).

Proof. Taking the expectation w.r.t. Q and using variance decomposition E

∥ξ∥2
=
E

∥ξ −E [ξ] ∥2
+ ∥Eξ∥2, we get

EQ

∥gt∥2
=
EQ






1
Mn

n−1
X

i=0

M
X

m=1
Q

∇f πi
m
m (xi
t)


2



=
EQ






1
Mn

n−1
X

i=0

M
X

m=1


Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (xi
t)


2



+


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xi
t)



2

.

33


---Page Break---
Next, Assumption 1 and conditional independence of Q

∇f πi
m
m (xi
t)

for m = 1, . . . , M, i =
0, . . . , n −1 imply

EQ

∥gt∥2
=
1
M 2n2

n−1
X

i=0

M
X

m=1
EQ

Q

∇f πi
m
m (xi
t)

−∇f πi
m
m (xi
t)

2

+


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xi
t)



2

≤
ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xi
t)

2
+


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xi
t)



2

≤
2ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xi
t) −∇f πi
m
m (xt)

2
+
2ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xt)

2

+2


1
Mn

n−1
X

i=0

M
X

m=1


∇f πi
m
m (xi
t) −∇f πi
m
m (xt)


2

+2


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xt)



2

.

Using Lmax-smoothness and convexity of f i
m and eL-smoothness and convexity of f πi
=

1
M
PM
m=1 f πi
m
m , we derive

EQ

∥gt∥2
≤
4ω
M 2n2 Lmax

n−1
X

i=0

M
X

m=1
D
f
πim
m (xi
t, xt) +
2ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xt)

2

+4eL 1

n

n−1
X

i=0
Df πi(xi
t, xt) + 2 ∥∇f(xt)∥2

≤
4

eL +
ω
MnLmax
 1

n

n−1
X

i=0
Df πi(xi
t, xt) +
4ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (x⋆)

2

+
4ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xt) −∇f πi
m
m (x⋆)

2
+ 2 ∥∇f(xt) −∇f(x⋆)∥2

≤
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0

xi
t −xt
2 +
4ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (x⋆)

2

+
8ω
M 2n2 Lmax

n−1
X

i=0

M
X

m=1
D
f
πim
m (xt, x⋆) + 4eL (f(xt) −f(x⋆)) .

Taking the full expectation, we obtain

E

∥gt∥2
≤
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E
hxi
t −xt
2i
+
4ω
M 2n2

n−1
X

i=0

M
X

m=1
E
∇f πi
m
m (x⋆)

2

+

4eL + 8ω

MnLmax


E [f(xt) −f(x⋆)]

=
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E
hxi
t −xt
2i
+ 4ω

Mn
 
ζ2
⋆+ σ2
⋆


+

4eL + 8ω

MnLmax


E [f(xt) −f(x⋆)] .

34


---Page Break---
Lemma C.4. Let Assumptions 1, 2, 3, 5 hold and τ ≤
1

2
q

eL(eL+
ω
Mn Lmax). Then, the following

inequality holds

1
n

n−1
X

i=0
E

∥xi
t −xt∥2
≤
24τ 2 
eL +
ω
MnLmax

E [f(xt) −f(x⋆)]

+8τ 2 ω

Mn
 
ζ2
⋆+ σ2
⋆

+ 8τ 2 σ2
⋆,n
n ,

where σ2
⋆,n = 1

n
Pn
i=1 ∥∇f i(x⋆)∥2, f i(x) =
1
M
PM
m=1 f i
m(x), i ∈[n].

Proof. Since xi
t = xt −
τ
Mn
PM
m=1
Pi−1
j=0 Q

∇f πj
m
m (xj
t)

, we have

EQ

∥xi
t −xt∥2
=
τ 2EQ







1
Mn

M
X

m=1

i−1
X

j=0
Q

∇f πj
m
m (xj
t)



2



=
τ 2EQ







1
Mn

M
X

m=1

i−1
X

j=0


Q

∇f πj
m
m (xj
t)

−∇f πj
m
m (xj
t)



2



+τ 2



1
Mn

M
X

m=1

i−1
X

j=0
∇f πj
m
m (xj
t)



2

≤
τ 2

M 2n2

M
X

m=1

i−1
X

j=0
EQ

Q

∇f πj
m
m (xj
t)

−∇f πj
m
m (xj
t)

2

+τ 2



1
Mn

M
X

m=1

i−1
X

j=0
∇f πj
m
m (xj
t)



2

.

35


---Page Break---
Using Assumption 1, eL-smoothness and convexity of f πi =
1
M
PM
m=1 f πi
m
m
and Lmax-smoothness
and convexity of f i
m, we obtain

EQ

∥xi
t −xt∥2
≤
τ 2ω
M 2n2

M
X

m=1

i−1
X

j=0

∇f πj
m
m (xj
t)

2
+ τ 2



1
Mn

M
X

m=1

i−1
X

j=0
∇f πj
m
m (xj
t)



2

≤
2τ 2ω
M 2n2

M
X

m=1

i−1
X

j=0

∇f πj
m
m (xj
t) −∇f πj
m
m (xt)

2
+ 2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+2τ 2



1
n

i−1
X

j=0


∇f πj(xj
t) −∇f πj(xt)



2

(15)

+ 2τ 2ω

M 2n2

M
X

m=1

i−1
X

j=0

∇f πj
m
m (xt)

2

≤
4τ 2ω
M 2n2

M
X

m=1

n−1
X

j=0
LmaxD
f πj
m
m (xj
t, xt) + 2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+4eLτ 2 1

n

n−1
X

j=0
Df πj (xj
t, xt) + 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0

∇f πj
m
m (xt)

2

=
4τ 2 
eL +
ω
MnLmax
 1

n

n−1
X

j=0
Df πj (xj
t, xt)

+2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+ 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0

∇f πj
m
m (xt)

2
.
(16)

Next, we need to estimate the second term from the previous inequality. Taking the full expectation
and using Lemma C.1 and using new notation σ2
t = 1

n
Pn
j=1 E[∥∇f j(xt) −∇f(xt)∥2], we get

E







1
n

i−1
X

j=0
∇f πj(xt)



2


=
i2

n2 E

∥∇f(xt)∥2
+ i2

n2 E







1

i

i−1
X

j=0


∇f πj(xt) −∇f(xt)



2



≤
i2

n2 E

∥∇f(xt)∥2
+ i2

n3
n −i
i(n −1)

n
X

j=1
E

∥∇f j(xt) −∇f(xt)∥2

≤
E

∥∇f(xt)∥2
+ 1

nσ2
t .
(17)

Taking the full expectation from (16) and using (17), we obtain

E

∥xi
t −xt∥2
≤
4τ 2 
eL +
ω
MnLmax
 n−1
X

j=0
E
h
Df πj (xj
t, xt)
i

+2τ 2E

∥∇f(xt)∥2
+ 2τ 2

n σ2
t + 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (xt)

2
.

Using eL-smoothness of f πj, we get

E

∥xi
t −xt∥2
≤
2eLτ 2 
eL +
ω
MnLmax
 n−1
X

j=0
E
h
∥xj
t −xt∥2i

+2τ 2E

∥∇f(xt)∥2
+ 2τ 2

n σ2
t + 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (xt)

2
.

36


---Page Break---
Since τ ≤
1

2
q

eL(eL+
ω
Mn Lmax), we have

E

∥xi
t −xt∥2
≤
2

1 −2eLτ 2 
eL +
ω
MnLmax
 n−1
X

j=0
E
h
∥xj
t −xt∥2i

≤
4τ 2E

∥∇f(xt)∥2
+ 4τ 2

n σ2
t + 4τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (xt)

2

≤
8τ 2ω
M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (xt) −∇f πj
m
m (x⋆)

2

+ 8τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (x⋆)

2
+ 4τ 2E

∥∇f(xt) −∇f(x⋆)∥2

+4τ 2

n



1

n

n
X

j=1
E

∥∇f j(xt)∥

−E

∥∇f(xt)∥2




≤
8τ 2ω
M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (xt) −∇f πj
m
m (x⋆)

2

+ 8τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
∇f πj
m
m (x⋆)

2
+ 8τ 2E

∥∇f(xt) −∇f(x⋆)∥2

+8τ 2

n2

n
X

j=1
E

∥∇f j(xt) −∇f j(x⋆)∥2
+ 8τ 2

n2

n
X

j=1
E

∥∇f j(x⋆)∥2
.

Summing from i = 0 to n −1 and using eL-smoothness of f i and Lmax-smoothness of f i
m, we obtain

1
n

n−1
X

i=0
E

∥xi
t −xt∥2
≤
16τ 2ω

Mn LmaxE [f(xt) −f(x⋆)] + 16τ 2

n
eLE [f(xt) −f(x⋆)]

+8τ 2ω

Mn
 
ζ2
⋆+ σ2
⋆

+ 8τ 2

n σ2
⋆,n + 8τ 2eLE [f(xt) −f(x⋆)] .

Theorem C.2. Let Assumptions 1, 2, 3, 5 hold and stepsize γ satisfy

0 < γ ≤
1

16n

eL +
ω
MnLmax
.
(18)

Then, for all T ≥0 the iterates produced by Q-RR satisfy

E

∥xT −x⋆∥2
≤

1 −nγµ

2

T
∥x0 −x⋆∥2 + 18γ2neL

µ

 ω

M (ζ2
⋆+ σ2
⋆) + σ2
⋆,n


+8 γω

µM (ζ2
⋆+ σ2
⋆),

where

σ2
⋆,n = 1

n

n
X

i=1
∥∇f i(x⋆)∥2.
(19)

37


---Page Break---
Proof. Taking expectation w.r.t. Q and using Lemma C.3, we get

EQ

∥xt+1 −x⋆∥2
=
∥xt −x⋆∥2 −2τEQ [⟨gt, xt −x⋆⟩] + τ 2EQ

∥gt∥2

≤
∥xt −x⋆∥2 −2τEQ

gt, xt −x⋆


+2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2

+8τ 2 
eL +
ω
MnLmax

(f(xt) −f(x⋆)) + 4τ 2ω

Mn (ζ2
⋆+ σ2
⋆).

Using Lemma C.2, we obtain

EQ

∥xt+1 −x⋆∥2
≤
∥xt −x⋆∥2

−τµ

2 ∥xt −x⋆∥2 −τ(f(xt) −f(x⋆)) + τ eL

n

n−1
X

i=0
∥xi
t −xt∥2

+2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2

+8τ 2 
eL +
ω
MnLmax

(f(xt) −f(x⋆)) + 4τ 2ω

Mn (ζ2
⋆+ σ2
⋆)

≤

1 −τµ

2


∥xt −x⋆∥2

−τ

1 −8τ

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+τ eL

1 + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2

+4τ 2ω

Mn (ζ2
⋆+ σ2
⋆).

Next, we take the full expectation and apply Lemma C.4:

E

∥xt+1 −x⋆∥2
≤

1 −τµ

2


E

∥xt −x⋆∥2

−τ

1 −8τ

eL +
ω
MnLmax

E [f(xt) −f(x⋆)]

+24τ 3eL

1 + 2τ

eL +
ω
MnLmax
 
eL +
ω
MnLmax

(f(xt) −f(x⋆))

+8τ 3eL

1 + 2τ

eL +
ω
MnLmax
  
ω
Mn(ζ2
⋆+ σ2
⋆) + σ2
⋆,n
n

!

+ 4τ 2ω

Mn (ζ2
⋆+ σ2
⋆).

Using (18), we derive

E

∥xt+1 −x⋆∥2
≤

1 −τµ

2


E

∥xt −x⋆∥2

+9τ 3eL

 
ω
Mn(ζ2
⋆+ σ2
⋆) + σ2
⋆,n
n

!

+ 4τ 2ω

Mn (ζ2
⋆+ σ2
⋆)

Recursively unrolling the inequality, substituting τ = nγ and using
+∞
P

t=0

 
1 −τµ

2
t ≤
2
µτ , we get the

result.

Corollary 6. Let the assumptions of Theorem C.2 hold and

γ = min





1

16n

eL +
ω
MnLmax
,
r εµ

82neL

 ω

M ∆2
⋆+ σ2
⋆,n
−1

2 , εµM

24ω∆2⋆




,
(20)

38


---Page Break---
where ∆2
⋆= ζ2
⋆+ σ2
⋆. Then, Q-RR finds a solution with accuracy ε > 0 after the following number
of communication rounds:

e
O



neL

µ + ω

M
Lmax

µ
+ ω

M
ζ2
⋆+ σ2
⋆
εµ2
+

s

neL
εµ3

r ω

M (ζ2⋆+ σ2⋆) + σ2⋆,n



.

Proof. Theorem C.2 implies

E

∥xT −x⋆∥2
≤

1 −nγµ

2

T
∥x0 −x⋆∥2 + 18γ2neL

µ

 ω

M
 
ζ2
⋆+ σ2
⋆

+ σ2
⋆,n


+8 γω

µM
 
ζ2
⋆+ σ2
⋆

.

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper bound each term from the right-hand side by ε/3. Thus, we get additional conditions on
γ:

18γ2neL

µ

 ω

M
 
ζ2
⋆+ σ2
⋆

+ σ2
⋆,n

< ε

3,
8 γω

µM
 
ζ2
⋆+ σ2
⋆

< ε

3,

and also the upper bound on the number of communication rounds nT

nT = e
O
 1

γµ


.

Substituting (20) in the previous equation, we get the result.

39


---Page Break---
D
Missing Proofs for DIANA-RR

D.1
Proof of Theorem 2.2

Lemma D.1. Let Assumptions 1, 3, 5, 6 hold and α ≤
1
1+ω. Then, the iterates of DIANA-RR satisfy

1
M

M
X

m=1
EQ
h
∥hπi
m
t+1,m −∇f πi
m
m (x⋆)∥2i
≤
1 −α

M

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+2αLmax

M

M
X

m=1
D
f
πim
m (xi
t, x⋆).

Proof. Taking expectation w.r.t. Q, we obtain

EQ
h
∥hπi
m
t+1,m −∇f πi
m
m (x⋆)∥2i
=
EQ
h
∥hπi
m
t,m + αQ(∇f πi
m
m (xi
t) −hπi
m
t,m) −∇f πi
m
m (x⋆)∥2i

=
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+2αEQ
hD
Q(∇f πi
m
m (xi
t) −hπi
m
t,m), hπi
m
t,m −∇f πi
m
m (x⋆)
Ei

+α2EQ
h
∥Q(∇f πi
m
m (xi
t) −hπi
m
t,m)∥2i

=
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+2α
D
∇f πi
m
m (xi
t) −hπi
m
t,m, hπi
m
t,m −∇f πi
m
m (x⋆)
E

+α2EQ
h
∥Q(∇f πi
m
m (xi
t) −hπi
m
t,m)∥2i
.

Assumption 1, Lmax-smoothness and convexity of f i
m and α ≤1/(1+ω) imply

EQ
h
∥hπi
m
t+1,m −∇f πi
m
m (x⋆)∥2i
≤
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+2α
D
∇f πi
m
m (xi
t) −hπi
m
t,m, hπi
m
t,m −∇f πi
m
m (x⋆)
E

+α2(1 + ω)∥∇f πi
m
m (xi
t) −hπi
m
t,m∥2

≤
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+α
D
∇f πi
m
m (xi
t) −hπi
m
t,m, hπi
m
t,m + ∇f πi
m
m (xi
t) −2∇f πi
m
m (x⋆)
E

≤
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+α∥∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)∥2 −α∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

≤
(1 −α)∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+α∥∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)∥2
(21)

≤
(1 −α)∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2 + 2αLmaxD
f
πim
m (xi
t, x⋆).

Summing up the above inequality for m = 1, . . . , M, we get the result.

Theorem D.1. Let Assumptions 1, 3, 5, 6 hold and 0 < γ ≤min
n
α
2neµ,
1
eL+ 6ω

M Lmax

o
, α ≤
1
1+ω.

Then, for all T ≥0 the iterates produced by DIANA-RR satisfy

E [ΨT ] ≤(1 −γeµ)nT Ψ0 + 2γ2σ2
rad
eµ
,

where Ψt is defined in (6).

40


---Page Break---
Proof. Using xi+1
⋆
= xi
⋆−γ

M
PM
m=1 ∇f πi
m
m (x⋆) and line 9 of Algorithm 2, we derive

∥xi+1
t
−xi+1
⋆
∥2
=

xi
t −xi
⋆−γ 1

M

M
X

m=1


ˆgπi
m
t,m −∇f πi
m
m (x⋆)


2

=
xi
t −xi
⋆
2 −2γ

M

M
X

m=1

D
ˆgπi
m
t,m −∇f πi
m
m (x⋆)

, xi
t −xi
⋆
E

+γ2

1
M

M
X

m=1


ˆgπi
m
t,m −∇f πi
m
m (x⋆)


2

.

Taking expectation w.r.t. Q and using E∥ξ −c∥2 = E∥ξ −Eξ∥2 + ∥Eξ −c∥2, we obtain

EQ

∥xi+1
t
−xi+1
⋆
∥2
=
xi
t −xi
⋆
2 −2γ

M

M
X

m=1

D
∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆), xi
t −xi
⋆
E

+γ2EQ






1
M

M
X

m=1


Q

∇f πi
m
m (xi
t) −hπi
m
t,m

+ hπi
m
t,m −∇f πi
m
m (x⋆)


2



≤
xi
t −xi
⋆
2 −2γ

M

M
X

m=1

D
∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆), xi
t −xi
⋆
E

+γ2EQ






1
M

M
X

m=1


Q

∇f πi
m
m (xi
t) −hπi
m
t,m

−∇f πi
m
m (xi
t) + hπi
m
t,m


2



+γ2

1
M

M
X

m=1


∇f πi
m
m (x⋆) −∇f πi
m
m (xi
t)


2

.

Independence of Q

∇f πi
m
m (xi
t) −hπi
m
t,m

, m ∈[M], assumption 1, and three-point identity imply

EQ

∥xi+1
t
−xi+1
⋆
∥2
≤
xi
t −xi
⋆
2

−2γ

M

M
X

m=1


D
f
πim
m (xi
⋆, xi
t) + D
f
πim
m (xi
t, x⋆) −D
f
πim
m (xi
⋆, x⋆)


+γ2ω

M 2

M
X

m=1

∇f πi
m
m (xi
t) −hπi
m
t,m

2

+γ2

1
M

M
X

m=1


∇f πi
m
m (x⋆) −∇f πi
m
m (xi
t)


2

≤
xi
t −xi
⋆
2

−2γ

M

M
X

m=1


D
f
πim
m (xi
⋆, xi
t) + D
f
πim
m (xi
t, x⋆) −D
f
πim
m (xi
⋆, x⋆)


+2γ2ω

M
1
M

M
X

m=1

∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)

2

+γ2

1
M

M
X

m=1


∇f πi
m
m (x⋆) −∇f πi
m
m (xi
t)


2

+2γ2ω

M 2

M
X

m=1

hπi
m
t,m −∇f πi
m
m (x⋆)

2
.

41


---Page Break---
Using Lmax-smoothness and µ-strong convexity of functions f i
m and eL-smoothness and eµ-strong

convexity of f πi =
1
M
PM
i=1 f πi
m
m , we obtain

EQ

∥xi+1
t
−xi+1
⋆
∥2
≤
(1 −γeµ)
xi
t −xi
⋆
2

−2γ

1 −γ

eL + 2ω

M Lmax

 1

M

M
X

m=1
D
f
πim
m (xi
t, x⋆)

+2γ

M

M
X

m=1
D
f
πim
m (xi
⋆, x⋆) + 2γ2ω

M 2

M
X

m=1

hπi
m
t,m −∇f πi
m
m (x⋆)

2
.

Taking the full expectation and using Defenition 2, we derive

E

∥xi+1
t
−xi+1
⋆
∥2
≤
(1 −γeµ)E
hxi
t −xi
⋆
2i

−2γ

1 −γ

eL + 2ω

M Lmax

 1

M

M
X

m=1
E
h
D
f
πim
m (xi
t, x⋆)
i

+2γ3σ2
rad + 2γ2ω

M 2

M
X

m=1
E
hπi
m
t,m −∇f πi
m
m (x⋆)

2
.

Recursively unrolling the inequality, we get

E

∥xt+1 −x⋆∥2
≤
(1 −γeµ)nE
h
∥xt −x⋆∥2i

+2γ2ω

M 2

M
X

m=1

n−1
X

j=0
(1 −γeµ)jE
hπi
m
t,m −∇f πi
m
m (x⋆)

2

−2γ

1 −γ

eL + 2ω

M Lmax

 1

M

M
X

m=1

n−1
X

j=0
(1 −γeµ)jE
h
D
f
πim
m (xi
t, x⋆)
i

+2γ3σ2
rad

n−1
X

j=0
(1 −γeµ)j.

Next, we apply (6) and Lemma D.1:

E [Ψt+1] ≤(1 −γeµ)nE
h
∥xt −x⋆∥2i
+ 2γ3σ2
rad

n−1
X

j=0
(1 −γeµ)j

+

c(1 −α) + 2ω

M

 γ2

M

M
X

m=1

n−1
X

j=0
(1 −γeµ)jE
hπi
m
t,m −∇f πi
m
m (x⋆)

2

−2γ

1 −cγαLmax −γ

eL + 2ω

M Lmax

 1

M

M
X

m=1

n−1
X

j=0
(1 −γeµ)jE
h
D
f
πim
m (xi
⋆, x⋆)
i
,

42


---Page Break---
where c =
4ω
αM 2 . Using α ≤
1
1+ω and γ ≤min

α
2nµ,
1
(eL+6ω/MLmax)


, we obtain

E [Ψt+1]
≤
(1 −γeµ)nE
h
∥xt −x⋆∥2i

+

1 −α

2

 4ωγ2

αM 2

M
X

m=1

n−1
X

j=0
(1 −γeµ)jE
hπi
m
t,m −∇f πi
m
m (x⋆)

2

+2γ2σ3
rad

n−1
X

j=0
(1 −γeµ)j

≤
max
n
(1 −γeµ)n,

1 −α

2

o
E [Ψt]

+2γ2σ3
rad

n−1
X

j=0
(1 −γeµ)j

≤
(1 −γeµ)nE [Ψt] + 2γ3σ2
rad

n−1
X

j=0
(1 −γeµ)j.

Recursively rewriting the inequality, we obtain

E [ΨT ]
≤
(1 −γeµ)nT Ψ0 + 2γ3σ2
rad

T −1
X

t=0
(1 −γeµ)tn
n−1
X

j=0
(1 −γeµ)j

≤
(1 −γeµ)nT Ψ0 + 2γ3σ2
rad

nT −1
X

k=0
(1 −γeµ)k

Using that
+∞
P

k=0


1 −γeµ

2
k
≤
2
eµγ , we finish proof.

Corollary 7. Let the assumptions of Theorem D.1 hold, α =
1
1+ω and

γ = min

(
α
2neµ,
1
eL + 6ω

M Lmax
,

p

εeµ
2σrad

)

.
(22)

Then DIANA-RR finds a solution with accuracy ε > 0 after the following number of communication
rounds:

e
O

 

n(1 + ω) +
eL
eµ + ω

M
Lmax

eµ
+
σrad
p

εeµ3

!

.

Proof. Theorem D.1 implies

E [ΨT ] ≤(1 −γeµ)nT Ψ0 + 2γ2σ2
rad
eµ
.

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper bound each term from the right-hand side by ε

2. Thus, we get an additional condition
on γ:
2γ2σ2
rad
eµ
< ε

2,

and also the upper bound on the number of communication rounds nT

nT = e
O
 1

γµ


.

Substituting (22) in the previous equation, we get the result.

43


---Page Break---
D.2
Non-Strongly Convex Summands

In this section, we provide the analysis of DIANA-RR without using Assumptions 4, 6. We emphasize
that xi+1
t
= xi
t −γ 1

M
PM
m=1 ˆgπi
m
t,m. Then we have

xt+1 = xt −γ

n−1
X

i=0

1
M

M
X

m=1
ˆgπi
m
t,m = xt −τ
1
Mn

n−1
X

i=0

M
X

m=1
ˆgπi
m
t,m.

We denote ˆgt =
1
Mn
Pn−1
i=0
PM
m=1 ˆgπi
m
t,m.
Lemma D.2. Let Assumptions 1, 2, 3, 5 hold. Then, the following inequality holds

−2τEQ [⟨ˆgt −h⋆, xt −x⋆⟩] ≤−τµ

2 ∥xt −x⋆∥2 −τ (f(xt) −f(x⋆)) + τ eL 1

n

n−1
X

i=1
∥xt −xi
t∥2,

where h⋆= ∇f(x⋆) = 0.

Proof. Since h⋆= ∇f(x⋆) = 0, the proof of Lemma D.2 is identical to the proof of Lemma C.2.

Lemma D.3. Let Assumptions 1, 2, 3, 5 hold. Then, the following inequality holds

EQ

∥ˆgt −h⋆∥2
≤
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2 + 8

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+
4ω
M 2n2

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

Proof. Taking expectation w.r.t. Q, we get

EQ
h
∥ˆgt −h⋆∥2i
=
EQ






1
Mn

n−1
X

i=0

M
X

m=1
ˆgπi
m
t,m −h⋆



2



=
EQ






1
Mn

n−1
X

i=0

M
X

m=1


hπi
m
t,m + Q

∇f πi
m
m (xi
t) −hπi
m
t,m

−h⋆



2



=
EQ






1
Mn

n−1
X

i=0

M
X

m=1


hπi
m
t,m −∇f πi
m
m (xi
t) + Q

∇f πi
m
m (xi
t) −hπi
m
t,m


2



+


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xi
t) −h⋆



2

.

Independence of Q

∇f πi
m
m (xi
t) −hπi
m
t,m

, m ∈[M] and Assumption 1 imply

EQ

∥ˆgt −h⋆∥2
=
1
M 2n2

n−1
X

i=0

M
X

m=1
EQ

hπi
m
t,m −∇f πi
m
m (xi
t) + Q

∇f πi
m
m (xi
t) −hπi
m
t,m

2

+


1
Mn

n−1
X

i=0

M
X

m=1
∇f πi
m
m (xi
t) −h⋆



2

≤
ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xi
t) −hπi
m
t,m

2
+


1
n

n−1
X

i=0
∇f πi(xi
t) −h⋆



2

≤
2ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xi
t) −∇f πi
m
m (xt)

2
+ 2

n

n−1
X

i=0

∇f πi(xi
t) −∇f πi(xt)

2

+
2ω
M 2n2

n−1
X

i=0

M
X

m=1

hπi
m
t,m −∇f πi
m
m (xt)

2
+ 2


1
n

n−1
X

i=0
∇f πi(xt) −h⋆



2

.

44


---Page Break---
Using Lmax-smoothness and convexity of f i
m and eL-smoothness and convexity of f πi, we obtain

EQ

∥ˆgt −h⋆∥2
≤
4ωLmax

M 2n2

n−1
X

i=0

M
X

m=1
D
f
πim
m (xi
t, xt) + 4eL

n

n−1
X

i=0
Df πi(xi
t, xt)

+
4ω
M 2n2

n−1
X

i=0

M
X

m=1

hπi
m
t,m −∇f πi
m
m (x⋆)

2
+ 4eL (f(xt) −f(x⋆))

+
4ω
M 2n2

n−1
X

i=0

M
X

m=1

∇f πi
m
m (xt) −∇f πi
m
m (x⋆)

2

≤
2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2 + 4eL (f(xt) −f(x⋆))

+ 8ω

MnLmax
1
Mn

n−1
X

i=0

M
X

m=1
D
f
πim
m (xt, x⋆)

+
4ω
M 2n2

n−1
X

i=0

M
X

m=1

hπi
m
t,m −∇f πi
m
m (x⋆)

2
.

Lemma D.4. Let α ≤
1
1+ω and Assumptions 1, 2, 3, 5 hold. Then, the iterates produced by DIANA-RR
satisfy

1
Mn

n−1
X

i=0

M
X

m=1
EQ
h
∥hπi
m
t+1,m −∇f πi
m
m (x⋆)∥2i
≤
1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+2αeLLmax

n

n−1
X

i=0
∥xi
t −xt∥2

+4αLmax (f(xt) −f(x⋆)) .

Proof. Fist of all, we introduce new notation: Ht =
1
Mn
Pn−1
i=0
PM
m=1 EQ
h
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2i
.
Using (21) and summing it up for i = 0, . . . , n −1, we obtain

Ht+1
≤
1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2 +
α
Mn

n−1
X

i=0

M
X

m=1
∥∇f πi
m
m (xi
t) −∇f πi
m
m (x⋆)∥2

≤
1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2 + 2α

Mn

n−1
X

i=0

M
X

m=1
∥∇f πi
m
m (xi
t) −∇f πi
m
m (xt)∥2

+ 2α

Mn

n−1
X

i=0

M
X

m=1
∥∇f πi
m
m (xt) −∇f πi
m
m (x⋆)∥2.

Next, we apply Lmax-smoothness and convexity of f i
m and eL-smoothness and convexity of f πi:

Ht+1
≤
1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2 + 4α

MnLmax

n−1
X

i=0

M
X

m=1
D
f
πim
m (xi
t, xt)

+ 4α

MnLmax

n−1
X

i=0

M
X

m=1
D
f
πim
m (xt, x⋆)

≤
1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2 + 2α

n
eLLmax

n−1
X

i=0
∥xi
t −xt∥2

+ 4α

MnLmax

n−1
X

i=0

M
X

m=1
D
f
πim
m (xt, x⋆).

45


---Page Break---
Lemma D.5. Let Assumptions 1, 2, 3, 5 and τ ≤
1

2
q

eL(eL+
ω
Mn Lmax). Then, the following inequality

holds

1
n

n−1
X

i=0
E

∥xi
t −xt∥2
≤
24τ 2 
eL +
ω
MnLmax

E [f(xt) −f(x⋆)] + 8τ 2 σ2
⋆,n
n

+8 τ 2ω

M 2n2

n−1
X

i=0

M
X

m=1
E
h
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2i
,

where σ2
⋆,n = 1

n
Pn
i=1 ∥∇f i(x⋆)∥2.

Proof. Since xi
t = xt −
τ
Mn
PM
m=1
Pi−1
j=0

hπj
m
t,m + Q

∇f πj
m
m (xj
t) −hπi
m
t,m

, we have

EQ

∥xi
t −xt∥2
=
τ 2EQ







1
Mn

M
X

m=1

i−1
X

j=0


hπj
m
t,m + Q

∇f πj
m
m (xj
t) −hπj
m
t,m



2



=
τ 2EQ







1
Mn

M
X

m=1

i−1
X

j=0


hπj
m
t,m −∇f πj
m
m (xj
t) + Q

∇f πi
m
m (xj
t) −hπj
m
t,m



2



+τ 2



1
Mn

M
X

m=1

i−1
X

j=0
∇f πj
m
m (xj
t)



2

.

Independence of Q

∇f πi
m
m (xj
t) −hπj
m
t,m

, m ∈[M] and Assumption 1 imply

EQ

∥xi
t −xt∥2
=
τ 2

M 2n2

M
X

m=1

i−1
X

j=0
EQ

hπj
m
t,m −∇f πj
m
m (xj
t) + Q

∇f πi
m
m (xj
t) −hπj
m
t,m

2

+τ 2



1
Mn

M
X

m=1

i−1
X

j=0
∇f πj
m
m (xj
t)



2

≤
τ 2ω
M 2n2

M
X

m=1

i−1
X

j=0

∇f πj
m(xj
t) −hπj
m
t,m

2
+ τ 2



1
n

i−1
X

j=0
∇f πj(xj
t)



2

≤
2τ 2ω
M 2n2

M
X

m=1

n−1
X

j=0

∇f πj
m
m (xj
t) −∇f πj
m
m (xt)

2
+ 2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+ 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0

hπj
m
t,m −∇f πj
m
m (xt)

2
+ 2τ 2

n

n−1
X

j=0

∇f πj(xj
t) −∇f πj(xt)

2
.

46


---Page Break---
Using Lmax-smoothness and convexity of f i
m and eL-smoothness and convexity of f πj, we obtain

EQ

∥xi
t −xt∥2
≤
4τ 2ω
M 2n2 Lmax

M
X

m=1

n−1
X

j=0
D
f πj
m
m (xj
t, xt) + 2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+ 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0

hπj
m
t,m −∇f πj
m
m (xt)

2
+ 2τ 2eL2

n

n−1
X

j=0

xj
t −xt

2

≤
2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

j=0
∥xj
t −xt∥2 + 2τ 2



1
n

i−1
X

j=0
∇f πj(xt)



2

+ 2τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0

hπj
m
t,m −∇f πj
m
m (xt)

2
.

Taking the full expectation and using (17), we derive

E

∥xi
t −xt∥2
≤
2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

j=0
E
h
∥xj
t −xt∥2i
+ 2τ 2E

∥∇f(xt)∥2

+ 4τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
hπj
m
t,m −∇f πj
m
m (x⋆)

2
+ 2τ 2

n E

σ2
t


+ 8τ 2ω

M 2n2 Lmax

M
X

m=1

n−1
X

j=0
E

D
f πj
m
m (xt, x⋆)

.

Using Lmax-smoothness and convexity of f i
m and eL-smoothness and convexity of f πj, we obtain

E

∥xi
t −xt∥2
≤
2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

j=0
E
h
∥xj
t −xt∥2i

+ 4τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
hπj
m
t,m −∇f πj
m
m (x⋆)

2
+ 2τ 2

n E

σ2
t


+4τ 2

eL +
2ω
M 2n2 Lmax


E [f(xt) −f(x⋆)] .

Now we need to estimate 2τ 2

n E

σ2
t

. Due to E

σ2
t

≤1

n
Pn
i=1 E

∥∇f i(xt)∥2
, we get

2τ 2

n E

σ2
t

≤
2τ 2

n2

n
X

j=1
E

∥∇f j(xt)∥2

≤
4τ 2

n2

n
X

j=1
E

∥∇f j(xt) −∇f j(x⋆)∥2
+ 4τ 2

n2

n
X

j=1
E

∥∇f j(x⋆)∥2

≤
8τ 2

n2 eL

n
X

j=1
E

Df j(xt, x⋆)

+ 4τ 2

n2

n
X

j=1
σ2
n,⋆.

47


---Page Break---
Combining two previous inequalities, we get

E

∥xi
t −xt∥2
≤
2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

j=0
E
h
∥xj
t −xt∥2i

+ 4τ 2ω

M 2n2

M
X

m=1

n−1
X

j=0
E
hπj
m
t,m −∇f πj
m
m (x⋆)

2

+4τ 2

eL +
2ω
M 2n2 Lmax


E [f(xt) −f(x⋆)]

+8τ 2

n
eLE [f(xt) −f(x⋆)] + 4τ 2

n2

n
X

j=1
σ2
n,⋆.

Summing from i = 0 to n −1 and using τ ≤
1

2
q

eL(eL+
ω
Mn Lmax), we obtain

1
n

n−1
X

i=0
E

∥xi
t −xt∥2
≤
2

1 −2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2

≤
8τ 2ω
M 2n2

M
X

m=1

n−1
X

j=0
E
hπj
m
t,m −∇f πj
m
m (x⋆)

2

+8τ 2

eL +
2ω
M 2n2 Lmax


E [f(xt) −f(x⋆)]

+16τ 2

n
eLE [f(xt) −f(x⋆)] + 8τ 2

n2

n
X

j=1
σ2
n,⋆.

We consider the following Lyapunov function:

Ψt+1 = ∥xt+1 −x⋆∥2 + cτ 2

Mn

M
X

m=1

n−1
X

j=0

hπi
m
t+1,m −∇f πi
m
m (x⋆)

2
.
(23)

Theorem D.2. Let Assumptions 1, 2, 3, 5 hold and

γ ≤min





α
nµ,
1

12n

eL + 11ω

MnLmax





,
α ≤
1
1 + ω ,
c = 10ω

αMn.

Then, for all T ≥0 the iterates produced by DIANA-RR satisfy

E [ΨT ] ≤

1 −nγµ

2

T
Ψ0 + 20γ2neL

µ
σ2
⋆,n.

Proof. Taking expectation w.r.t. Q and using Lemma D.2, we get

EQ

∥xt+1 −x⋆∥2
=
∥xt −τˆgt −x⋆+ τh⋆∥2

=
∥xt −x⋆∥2 −2τEQ [⟨ˆgt −h⋆, xt −x⋆⟩] + τ 2EQ

∥ˆgt −h⋆∥2

≤
∥xt −x⋆∥2 −τµ

2 ∥xt −x⋆∥2 + τ 2EQ

∥ˆgt −h⋆∥2

−τ (f(xt) −f(x⋆)) + τ eL 1

n

n−1
X

i=1
∥xt −xi
t∥2.

48


---Page Break---
Next, due to Lemma D.3 we have

EQ

∥xt+1 −x⋆∥2
≤

1 −τµ

2


∥xt −x⋆∥2 −τ (f(xt) −f(x⋆)) + τ eL 1

n

n−1
X

i=1
∥xt −xi
t∥2

+2τ 2eL

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2

+8τ 2 
eL +
ω
MnLmax

(f(xt) −f(x⋆))

+ 4ωτ 2

M 2n2

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

≤

1 −τµ

2


∥xt −x⋆∥2 + 4ωτ 2

M 2n2

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

−τ

1 −8τ

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+τ eL

1 + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2.

Using (23), we obtain

EQ [Ψt+1]
≤

1 −τµ

2


∥xt −x⋆∥2 + 4ωτ 2

M 2n2

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

−τ

1 −8τ

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+τ eL

1 + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2

+ cτ 2

Mn

M
X

m=1

n−1
X

j=0
E
hπi
m
t+1,m −∇f πi
m
m (x⋆)

2
.

To estimate the last term in the above inequality, we apply Lemma D.4:

EQ [Ψt+1]
≤

1 −τµ

2


∥xt −x⋆∥2 + 4ωτ 2

M 2n2

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

−τ

1 −8τ

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+τ eL

1 + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2

+cτ 2 1 −α

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

+cτ 2 2αeLLmax

n

n−1
X

i=0
∥xi
t −xt∥2 + 4cτ 2αLmax (f(xt) −f(x⋆))

≤

1 −τµ

2


∥xt −x⋆∥2 +

1 −α +
4ω
cMn

 cτ 2

Mn

n−1
X

i=0

M
X

m=1
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2

−τ

1 −4cταLmax −8τ

eL +
ω
MnLmax

(f(xt) −f(x⋆))

+τ eL

1 + 2cταLmax + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
∥xi
t −xt∥2.

49


---Page Break---
Let Ht =
cτ 2
Mn
Pn−1
i=0
PM
m=1 E
h
∥hπi
m
t,m −∇f πi
m
m (x⋆)∥2i
. Taking the full expectation and using
Lemma D.5, we get

E [Ψt+1]
≤

1 −τµ

2


E

∥xt −x⋆∥2
+

1 −α +
4ω
cMn


Ht

−τ

1 −4cταLmax −8τ

eL +
ω
MnLmax

E [f(xt) −f(x⋆)]

+τ eL

1 + 2cταLmax + 2τ

eL +
ω
MnLmax
 1

n

n−1
X

i=0
E

∥xi
t −xt∥2

≤

1 −τµ

2


E

∥xt −x⋆∥2
+

1 −α +
4ω
cMn


Ht

−τ

1 −4cταLmax −8τ

eL +
ω
MnLmax

E [f(xt) −f(x⋆)]

+24τ 3eL

1 + 2cταLmax + 2τ

eL +
ω
MnLmax
 
eL +
ω
MnLmax

E [f(xt) −f(x⋆)]

+8τ 3eL

1 + 2cταLmax + 2τ

eL +
ω
MnLmax
 σ2
⋆,n
n

+8τ eLω

cMn


1 + 2cταLmax + 2τ

eL +
ω
MnLmax

Ht.

Selecting c =
Aω
αMn, where A is a positive number to be specified later, we have

1 + 2cταLmax + 2τ

eL +
ω
MnLmax

= 1 + 2τ

eL + (A + 1)ω

Mn
Lmax


,

1 −4cταLmax −8τ

eL +
ω
MnLmax

≥1 −8τ

eL + (A + 1)ω

Mn
Lmax


.

Then, we have

E [Ψt+1]
≤

1 −τµ

2


E

∥xt −x⋆∥2
+

1 −α + 4α

A


Ht

−τ

1 −8τ

eL + (A + 1)ω

Mn
Lmax


E [f(xt) −f(x⋆)]

+24τ 3eL

eL +
ω
MnLmax
 
1 + 2τ

eL + (A + 1)ω

Mn
Lmax


E [f(xt) −f(x⋆)]

+8τ 3eL

1 + 2τ

eL + (A + 1)ω

Mn
Lmax

 σ2
⋆,n
n

+8α

A τ eL

1 + 2τ

eL + (A + 1)ω

Mn
Lmax


Ht.

Taking τ =
1
B(eL+ (A+1)ω

Mn
Lmax), where B is some positive constant, we obtain

E [Ψt+1]
≤

1 −τµ

2


E

∥xt −x⋆∥2
+

1 −α + 4α

A + 8α

A τ eL

1 + 2

B


Ht

−τ

1 −8

B −24

B2


1 + 2

B


E [f(xt) −f(x⋆)]

+8τ 3eL

1 + 2

B

 σ2
⋆,n
n .

Choosing A = 10, B = 12, τ ≤α

µ, we have

E [Ψt+1]
≤

1 −min
nτµ

2 , α

2

o
E [Ψt] + 10τ 3eLσ2
⋆,n
n

≤

1 −τµ

2


E [Ψt] + 10τ 3eLσ2
⋆,n
n

50


---Page Break---
Recursively unrolling the inequality, substituting τ = nγ and using
+∞
P

t=0

 
1 −τµ

2
t ≤
2
µτ , we finish

proof.

Corollary 8. Let the assumptions of Theorem D.2 hold, α =
1
1+ω, and

γ = min





α
2nµ,
1

12n

eL + 11ω

MnLmax
,
s

εµ

40neLσ2⋆,n




.
(24)

Then, DIANA-RR finds a solution with accuracy ε > 0 after the following number of communication
rounds:

e
O



n(1 + ω) + neL

µ + ω

M
Lmax

µ
+

s

neL
εµ3 σ⋆,n



.

Proof. Theorem D.2 implies

E [ΨT ] ≤(1 −γµ)nT Ψ0 + 20γ2neL

µ
σ2
⋆,n.

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper bound each term from the right-hand side by ε

2. Thus, we get an additional condition
on γ:

20γ2neL

µ
σ2
⋆,n < ε

2,

and also the upper bound on the number of communication rounds nT

nT = e
O
 1

γµ


.

Substituting (24) in the previous equation, we obtain the result.

51


---Page Break---
E
Theoretical Results for Q-NASTYA and DIANA-NASTYA

Theorem E.1. Let Assumptions 1, 2, 3 hold. Let the stepsizes γ, η satisfy 0 < η ≤
1
16Lmax(1+ ω

M ),

0 < γ ≤
1
5nLmax . Then, for all T ≥0 the iterates produced by Q-NASTYA (Algorithm 3) satisfy

E

∥xT −x⋆∥2
≤

1 −ηµ

2

T
∥x0 −x⋆∥2 + 8 ηω

µM ζ2
⋆+ 9

2
γ2nLmax

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

.

Corollary 9. Under the same conditions as Theorem E.1 and for Algorithm 3, there exist stepsizes
γ = η/n and η > 0 such that the number of communication rounds T to find a solution with accuracy

ε > 0 is e
O

Lmax

µ
 
1 + ω

M

+ ω

M
ζ2
⋆
εµ3 +
q

Lmax

εµ3
q

ζ2⋆+ σ2⋆

n


. If γ →0, one can choose η > 0 such

that the above complexity bound improves to e
O

Lmax

µ
 
1 + ω

M

+ ω

M
ζ2
⋆
εµ3

.

We emphasize several differences with the known theoretical results. First, the FedCOM method
of Haddadpour et al. [2021] was analyzed in the homogeneous setting only, i.e., fm(x) = f(x)
for all m ∈[M], which is an unrealistic assumption for FL applications. In contrast, our result
holds in the fully heterogeneous case. Next, the analysis of FedPAQ of Reisizadeh et al. [2020]
uses a bounded variance assumption, which is also known to be restrictive. Nevertheless, let us
compare to their result. Reisizadeh et al. [2020] derive the following complexity for their method:
e
O

Lmax

µ
 
1 + ω

M

+ ω

M
σ2
µ2ε +
σ2
Mµ2ε

. This result is inferior to the one we show for Q-NASTYA:

when ω is small, the main term in the complexity bound of FedPAQ is e
O (1/ε), while for Q-NASTYA
the dominating term is of the order e
O (1/√ε) (when ω and ε are sufficiently small). We also highlight
that FedCRR [Malinovsky and Richtárik, 2022] does not converge if ω > M 2γµε/(2∥xn
∗,m∥
2), while
Q-NASTYA does for any ω ≥0. Finally, when ω = 0 (no compression) we recover NASTYA as a
special case, and using γ = η/n, we recover the rate of FedRR [Mishchenko et al., 2021].
Theorem E.2. Let Assumptions 1, 2, 3 hold. Suppose the stepsizes γ, η, α satisfy 0 < γ ≤
1
16Lmaxn,

0 < η ≤min

α
2µ,
1
16Lmax(1+ 9ω

M )


, and α ≤
1
1+ω. Define the following Lyapunov function:

Ψt+1
def
= ∥xt+1 −x⋆∥2 + 8ωη2

αM 2

M
X

m=1
∥ht+1,m −h⋆
m∥2.
(25)

Then, for all T ≥0 the iterates produced by DIANA-NASTYA (Algorithm 4) satisfy

E [ΨT ] ≤

1 −ηµ

2

T
Ψ0 + 9

2
γ2nL

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

.
(26)

Corollary 10. Under the same conditions as Theorem E.2 and for Algorithm 4, there exist stepsizes
γ = η/n, η > 0, α > 0 such that the number of communication rounds T to find a solution with

accuracy ε > 0 is e
O

ω + Lmax

µ
 
1 + ω

M

+
q

Lmax

εµ3
q

ζ2⋆+ σ2⋆

n


. If γ →0, one can choose η > 0

such that the above complexity bound improves to e
O

ω + Lmax

µ
 
1 + ω

M

.

In contrast to Q-NASTYA, DIANA-NASTYA does not suffer from the e
O(1/ε) term in the complexity
bound. This shows the superiority of DIANA-NASTYA to Q-NASTYA. Next, FedCRR-VR [Malinovsky

and Richtárik, 2022] has the rate e
O

(ω+1)(1−1

κ)
n

(1−(1−1

κ)
n)
2 +
√κ(ζ⋆+σ⋆)

µ√ε


, which depends on e
O (1/√ε).

However, the first term is close to e
O
 
(ω + 1)κ2
for a large condition number. FedCRR-VR-2
utilizes variance reduction technique from Malinovsky et al. [2021] and it allows to get rid of

permutation variance. This method has e
O



(ω+1)

1−
1
κ√κn

 n

2

1−

1−
1
κ√κn

 n

2
2 +
√κζ⋆

µ√ε



complexity, but it requires

additional assumption on number of functions n and thus not directly comparable with our result.
Note that if we have no compression (ω = 0), DIANA-NASTYA recovers rate of NASTYA.

52


---Page Break---
F
Missing Proofs for Q-NASTYA

We start with deriving a technical lemma along with stating several useful results from [Malinovsky
et al., 2022]. For convenience, we also introduce the following notation:

gt,m = 1

n

n−1
X

i=0
∇f πi
m
m (xi
t,m).

Lemma F.1. Let Assumptions 1, 2, 3 hold. Then, for all t ≥0 the iterates produced by Q-NASTYA
satisfy

EQ

∥gt∥2
≤2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2+8Lmax

1 + ω

M


(f(xt) −f(x⋆))+4ω

M ζ2
⋆,

where EQ is expectation w.r.t. Q, and ζ2
⋆=
1
M
PM
m=1 ∥∇fm(x⋆)∥2.

Proof. Using the variance decomposition E

∥ξ∥2
= E

∥ξ −E [ξ] ∥2
+ ∥Eξ∥2, we obtain

EQ

∥gt∥2
=
1
M 2

M
X

m=1
EQ





Q

 
1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m)

!

−1

n

n−1
X

i=0
∇f πi
m
m (xi
t,m)



2



+


1
Mn

M
X

m=1

n−1
X

i=0
∇f πi
m
m (xi
t,m)



2

Asm.1

≤
ω
M 2

M
X

m=1


1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m)



2

+


1
Mn

M
X

m=1

n−1
X

i=0
∇f πi
m
m (xi
t,m)



2

.

Next, we use ∇fm(xt) = 1

n
Pn−1
i=0 ∇f πi
m
m (xt) and ∥a + b∥2 ≤2∥a∥2 + 2∥b∥2:

EQ

∥gt∥2
≤
2ω
M 2

M
X

m=1


1
n

n−1
X

i=0


∇f πi
m
m (xi
t,m) −∇f πi
m
m (xt)


2

+ 2ω

M 2

M
X

m=1
∥∇fm(xt)∥2

+2


1
Mn

M
X

m=1

n−1
X

i=0


∇f πi
m
m (xi
t,m) −∇f πi
m
m (xt)


2

+ 2


1
M

M
X

m=1
∇fm(xt)



2

≤
2
 
1 + ω

M


M

M
X

m=1


1
n

n−1
X

i=0


∇f πi
m
m (xi
t,m) −∇f πi
m
m (xt)


2

+ 2ω

M 2

M
X

m=1
∥∇fm(xt)∥2 + 2 ∥∇f(xt)∥2 .

Using Li,m-smoothness of f i
m and f and also convexity of fm, we obtain

EQ

∥gt∥2
≤
2
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

∇f πi
m
m (xi
t,m) −∇f πi
m
m (xt)

2
+ 4ω

M 2

M
X

m=1
∥∇fm(xt) −∇fm(x⋆)∥2

+ 4ω

M 2

M
X

m=1
∥∇fm(x⋆)∥2 + 2 ∥∇f(xt) −∇f(x⋆)∥2

≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 8Lmax
 
1 + ω

M


M

M
X

m=1
Dfm(xt, x⋆) + 4ω

M ζ2
⋆.

53


---Page Break---
Lemma F.2 (see [Malinovsky et al., 2022]). Under Assumptions 1, 2, 3, it holds

−1

Mn

M
X

m=1

n−1
X

i=0

D
f πi
m
m (xi
t,m), xt −x⋆
E
≤−µ

4 ∥xt−x⋆∥2−1

2 (f(xt) −f(x⋆))+Lmax

2Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 .

Lemma F.3 (see [Malinovsky et al., 2022]). Under Assumptions 1, 2, 3 and γ ≤
1
2Lmaxn, it holds

1
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 ≤8γ2n2Lmax (f(xt) −f(x⋆)) + 2γ2n
 
σ2
⋆+ (n + 1)ζ2
⋆

.

Theorem F.1. Let Assumptions 1, 2, 3 hold and stepsizes γ, η satisfy

0 < η ≤
1
16Lmax
 
1 + ω

M
,
0 < γ ≤
1
5nLmax
.
(27)

Then, for all T ≥0 the iterates produced by Q-NASTYA satisfy

E

∥xT −x⋆∥2
≤

1 −ηµ

2

T
∥x0 −x⋆∥2 + 9

2
γ2nLmax

µ
 
σ2
⋆+ (n + 1)ζ2
⋆

+ 8 ηω

µM ζ2
⋆.

Proof. Taking expectation w.r.t. Q and using Lemma F.1, we get

EQ

∥xt+1 −x⋆∥2
=
∥xt −x⋆∥2 −2ηEQ [⟨gt, xt −x⋆⟩] + η2EQ

∥gt∥2

≤
∥xt −x⋆∥2 −2ηEQ

"*
1
M

M
X

m=1
Q

 
1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m)

!

, xt −x⋆

+#

+2η2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+8η2Lmax

1 + ω

M


(f(xt) −f(x⋆)) + 4η2 ω

M ζ2
⋆

≤
∥xt −x⋆∥2 −2η 1

Mn

M
X

m=1

n−1
X

i=0

D
∇f πi
m
m (xi
t,m), xt −x⋆
E

+2η2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+8η2Lmax

1 + ω

M


(f(xt) −f(x⋆)) + 4η2 ω

M ζ2
⋆.

Next, Lemma F.2 implies

EQ

∥xt+1 −x⋆∥2
≤
∥xt −x⋆∥2 −ηµ

2 ∥xt −x⋆∥2 −η (f(xt) −f(x⋆))

+8η2Lmax

1 + ω

M


(f(xt) −f(x⋆)) + ηLmax

Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+2η2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 4η2 ω

M ζ2
⋆

≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηLmax

1 + ω

M


(f(xt) −f(x⋆))

+ηLmax
 
1 + 2ηLmax
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 4η2 ω

M ζ2
⋆.

54


---Page Break---
Using Lemma F.3, we get

EQ

∥xt+1 −x⋆∥2
≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηL

1 + ω

M


(f(xt) −f(x⋆))

+ηLmax

1 + 2ηLmax

1 + ω

M


· 8γ2n2Lmax (f(xt) −f(x⋆))

+ηLmax

1 + 2ηLmax

1 + ω

M


· 2γ2n
 
σ2
⋆+ (n + 1)ζ2
⋆


+4η2 ω

M ζ2
⋆.

In view of (27), we have

EQ

∥xt+1 −x⋆∥2
≤

1 −ηµ

2


∥xt −x⋆∥2 + 4η2 ω

M ζ2
⋆

−η

1 −8ηLmax

1 + ω

M


−8γ2n2L2
max

1 + 2Lmaxη

1 + ω

M


(f(xt) −f(x⋆))

+2γ2nηLmax

1 + 2ηL

1 + ω

M

  
σ2
⋆+ nζ2
⋆


≤

1 −ηµ

2


∥xt −x⋆∥2 + 4η2 ω

M ζ2
⋆+ 9

4ηLmaxγ2n
 
σ2
⋆+ (n + 1)σ2
⋆

.

Recursively unrolling the inequality and using
+∞
P

t=0

 
1 −ηµ

2
t ≤
2
µη, we get the result.

Corollary 11. Let the assumptions of Theorem E.1 hold, γ = η/n, and

η = min

(
1
16Lmax
 
1 + ω

M
,
r εµn

9Lmax

 
(n + 1)ζ2
⋆+ σ2
⋆
−1/2 , εµM

24ωζ2⋆

)

.
(28)

Then, Q-NASTYA finds a solution with accuracy ε > 0 after the following number of communication
rounds:

e
O

 
Lmax

µ


1 + ω

M


+ ω

M
ζ2
⋆
εµ3 +

s

Lmax

εµ3

q

ζ2⋆+ σ2
⋆/n

!

.

If γ →0, one can choose η = min

1
16Lmax(1+ ω

M ), εµM

24ωζ2⋆


such that the above complexity bound

improves to

e
O
Lmax

µ


1 + ω

M


+ ω

M
ζ2
⋆
εµ3


.

Proof. Theorem E.1 implies

E

∥xT −x⋆∥2
≤

1 −ηµ

2

T
∥x0 −x⋆∥2 + 9

2
γ2nLmax

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

+ 8 ηω

µM ζ2
⋆.

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper bound each term from the right-hand side by ε/3. Thus, we get additional conditions on
η:
9
2
η2Lmax

nµ
 
(n + 1)ζ2
⋆+ σ2
⋆

< ε

3,
8 ηω

µM ζ2
⋆< ε

3
and also the upper bound on the number of communication rounds T

T = e
O
 1

ηµ


.

Substituting (31) in the previous equation, we get the first part of the result. When γ →0, the proof
follows similar steps.

55


---Page Break---
G
Missing Proofs for DIANA-NASTYA

Lemma G.1. Under Assumptions 1, 2, 3, the iterates produced by DIANA-NASTYA satisfy

−EQ

"
1
M

M
X

m=1
⟨ˆgt,m −h⋆, xt −x⋆⟩

#

≤
−µ

4 ∥xt −x⋆∥2 −1

2 (f(xt) −f(x⋆))

−1

Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m)

+Lmax

2Mn

M
X

m=1

n−1
X

i=0
∥xt −xi
t,m∥2,

where h⋆= ∇f(x⋆).

Proof. Using that EQ [ˆgt,m] = gt,m and definition of h⋆, we get

−EQ

"
1
M

M
X

m=1
⟨ˆgt,m −h⋆, xt −x⋆⟩

#

=
−1

M

M
X

m=1
⟨gt,m −h⋆, xt −x⋆⟩

=
−1

Mn

M
X

m=1

n−1
X

i=0

D
∇f πi
m
m (xi
t,m) −∇f πi
m
m (x⋆), xt −x⋆
E
.

Next, three-point identity and Lmax-smoothness of each function f i
m imply

−EQ

"
1
M

M
X

m=1
⟨ˆgt,m −h⋆, xt −x⋆⟩

#

=
−1

Mn

M
X

m=1

n−1
X

i=0


D
f
πim
m (xt, x⋆) + D
f
πim
m (x⋆, xi
t,m) −D
f
πim
m (xt, xi
t,m)


≤
−Df(xt, x⋆) −
1
Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m)

+Lmax

2Mn

M
X

m=1

n−1
X

i=0
∥xt −xi
t,m∥2

Finally, using µ-strong convexity of f, we finish the proof of lemma.

Lemma G.2. Under Assumptions 1, 2, 3, the iterates produced by DIANA-NASTYA satisfy

EQ

∥ˆgt −h⋆∥2
≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 8Lmax

1 + ω

M


(f(xt) −f(x⋆))

+ 4ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2.

Proof. Since gt =
1
M
PM
m=1 gt,m and E∥ξ −c∥2 = E∥ξ −Eξ∥2 + E∥Eξ −c∥2, we have

EQ

∥ˆgt −h⋆∥2
=
EQ






1
M

M
X

m=1
(ht,m + Q (gt,m −ht,m) −h⋆
m)



2



=
EQ






1
M

M
X

m=1
(ht,m + Q (gt,m −ht,m)) −gt



2

+ ∥gt −h⋆∥2 .

56


---Page Break---
Next, independence of Q (gt,m −ht,m), m ∈M, Assumption 1, and Lmax-smoothness and convex-
ity of each function f i
m imply

EQ

∥ˆgt −h⋆∥2
≤
ω
M 2

M
X

m=1
∥gt,m −ht,m∥2 + ∥gt −h⋆∥2

≤
2ω
M 2

M
X

m=1


1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m) −∇fm(xt)



2

+ 2ω

M 2

M
X

m=1
∥∇fm(xt) −ht,m∥2

+2 ∥gt −∇f(xt)∥2 + 2 ∥∇f(xt) −h⋆∥2

≤
2ω
M 2

M
X

m=1


1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m) −∇fm(xt)



2

+ 2ω

M 2

M
X

m=1
∥∇fm(xt) −ht,m∥2

+2


1
M

M
X

m=1

 
1
n

n−1
X

i=0
∇f πi
m
m (xi
t,m) −∇fm(xt)

!

2

+ 2 ∥∇f(xt) −h⋆∥2

≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 2ω

M 2

M
X

m=1
∥∇fm(xt) −ht,m∥2

+2 ∥∇f(xt) −h⋆∥2 .

Using Lmax-smoothness and convexity of fm, we get

EQ

∥ˆgt −h⋆∥2
≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 2ω

M 2

M
X

m=1
∥∇fm(xt) −ht,m∥2

+4Lmax (f(xt) −f(x⋆))

≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 4ω

M 2

M
X

m=1
∥∇fm(xt) −h⋆
m∥2

+ 4ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2 + 4Lmax (f(xt) −f(x⋆))

≤
2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 8Lmaxω

M 2

M
X

m=1
Dfm(xt, x⋆)

+ 4ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2 + 4Lmax (f(xt) −f(x⋆)) .

Lemma G.3. Under Assumptions 1, 2, 3, and α ≤
1
1+ω, the iterates produced by DIANA-NASTYA
satisfy

1
M

M
X

m=1
EQ

∥ht+1,m −h⋆
m∥2
≤
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + 2αL2
max
Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2

+4αLmax (f(xt) −f(x⋆)) .

57


---Page Break---
Proof. Taking expectation w.r.t. Q and using Assumption 1, we obtain

1
M

M
X

m=1
EQ

∥ht+1,m −h⋆
m∥2
=
1
M

M
X

m=1
EQ

∥ht,m + αQ(gt,m −ht,m) −h⋆
m∥2

≤
1
M

M
X

m=1

 
∥ht,m −h⋆
m∥2 + 2αEQ [⟨Q(gt,m −ht,m), ht,m −h⋆
m⟩]


+α2 1

M

M
X

m=1
EQ

∥Q(gt,m −ht,m)∥2

≤
1
M

M
X

m=1

 
∥ht,m −h⋆
m∥2 + 2α ⟨gt,m −ht,m, ht,m −h⋆
m⟩


+α2(1 + ω) 1

M

M
X

m=1
∥gt,m −ht,m∥2

Using α ≤
1
1+ω, we get

1
M

M
X

m=1
EQ

∥ht+1,m −h⋆
m∥2
≤
1
M

M
X

m=1

 
∥ht,m −h⋆
m∥2 + α ⟨gt,m −ht,m, ht,m + gt,m −2h⋆
m⟩


≤
1
M

M
X

m=1

 
∥ht,m −h⋆
m∥2 + α∥gt,m −h⋆
m∥2 −α∥ht,m −h⋆
m∥2

≤
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + α

M

M
X

m=1
∥gt,m −h⋆
m∥2.

Finally, Lmax-smoothness and convexity of fm imply

1
M

M
X

m=1
EQ

∥ht+1,m −h⋆
m∥2
≤
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2

+2α

M

M
X

m=1

 
∥gt,m −∇fm(xt)∥2 + ∥∇fm(xt) −h⋆
m∥2

≤
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + 4Lmaxα

M

M
X

m=1
Dfm(xt, x⋆)

+2α

M

M
X

m=1


1
n

n−1
X

i=0
(∇f πi
m
m (xi
t,m) −∇f i
m(xt))



2

≤
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + 4Lmaxα (f(xt) −f(x⋆))

+2L2
maxα
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 .

Theorem G.1. Let Assumptions 1, 2, 3 hold and stepsizes γ, η, α satisfy

0 < γ ≤
1
16Lmaxn,
0 < η ≤min

(
α
2µ,
1
16Lmax
 
1 + 9ω

M


)

,
α ≤
1
1 + ω .
(29)

Then, for all T ≥0 the iterates produced by DIANA-NASTYA satisfy

E [ΨT ] ≤

1 −ηµ

2

T
Ψ0 + 9

2
γ2nL

µ
 
σ2
⋆+ (n + 1)ζ2
⋆

.
(30)

58


---Page Break---
Proof. We have

∥xt+1 −x⋆∥2
=
∥xt −ηˆgt −x⋆+ ηh⋆∥2

=
∥xt −x⋆∥2 −2η⟨ˆgt −h⋆, xt −x⋆⟩+ η2∥ˆgt −h⋆∥2.

Taking expectation w.r.t. Q and using Lemma G.1, we obtain

EQ

∥xt+1 −x⋆∥2
=
∥xt −x⋆∥2 −2ηEQ [⟨ˆgt −h⋆, xt −x⋆⟩] + η2EQ

∥ˆgt −h⋆∥2

≤

1 −ηµ

2


∥xt −x⋆∥2 −η(f(xt) −f(x⋆)) −2η

Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m)

+Lmaxη

Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + η2EQ

∥ˆgt −h⋆∥2
.

Next, Lemma G.2 implies

EQ

∥xt+1 −x⋆∥2
≤

1 −ηµ

2


∥xt −x⋆∥2 −η(f(xt) −f(x⋆)) −2η

Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m)

+Lmaxη

Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 2η2L2
max
 
1 + ω

M


Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2

+η2
 

8Lmax

1 + ω

M


(f(xt) −f(x⋆)) + 4ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2
!

≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηLmax

1 + ω

M


(f(xt) −f(x⋆))

+Lmaxη

1 + 2ηLmax

1 + ω

M


1
Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2

−2η

Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m) + 4η2ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2.

Using (6) and Lemma G.3, we get

EQ [Ψt+1]

≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηLmax

1 + ω

M


(f(xt) −f(x⋆))

+Lmaxη

1 + 2ηLmax

1 + ω

M


1
Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2

−2η

Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m (x⋆, xi
t,m) + 4η2ω

M 2

M
X

m=1
∥ht,m −h⋆
m∥2

+cη2
 
1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + 2αL2
max
Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2 + 4αLmax (f(xt) −f(x⋆))

!

≤

1 −ηµ

2


∥xt −x⋆∥2 + η2

c(1 −α) + 4ω

M

 1

M

M
X

m=1
∥ht,m −h⋆
m∥2

−η

1 −8ηLmax

1 + ω

M


−4αηcLmax

(f(xt) −f(x⋆))

+Lη

1 + 2ηLmax

1 + ω

M


+ 2αηcLmax

1
Mn

M
X

m=1

n−1
X

i=0
∥xi
t,m −xt∥2.

59


---Page Break---
Taking the full expectation, we derive

E [Ψt+1]
≤

1 −ηµ

2


E

∥xt −x⋆∥2
+ η2

c(1 −α) + 4ω

M

 1

M

M
X

m=1
E

∥ht,m −h⋆
m∥2

−η

1 −8ηLmax

1 + ω

M


−4αηcL

E [f(xt) −f(x⋆)]

+Lmaxη

1 + 2ηLmax

1 + ω

M


+ 2αηcLmax

1
Mn

M
X

m=1

n−1
X

i=0
E

∥xi
t,m −xt∥2
.

Using Lemma F.3, we get

E [Ψt+1]
≤

1 −ηµ

2


E

∥xt −x⋆∥2
+ η2

c(1 −α) + 4ω

M

 1

M

M
X

m=1
E

∥ht,m −h⋆
m∥2

−η

1 −8ηLmax

1 + ω

M


−4αηcLmax

E [f(xt) −f(x⋆)]

+8γ2n2L2
maxη

1 + 2ηLmax

1 + ω

M


+ 2αηcLmax

E [f(xt) −f(x⋆)]

+2γ2nLmaxη

1 + 2ηLmax

1 + ω

M


+ 2αηcLmax
  
σ2
⋆+ (n + 1)ζ2
⋆

.

In view of (29), we have

E [Ψt+1]
≤

1 −ηµ

2


E

∥xt −x⋆∥2
+

1 −α

2

 cη2

M

M
X

m=1
E

∥ht,m −h⋆
m∥2

+9

4γ2nLmaxη
 
σ2
⋆+ (n + 1)ζ2
⋆


Using definition of Lyapunov function and using
+∞
P

t=0

 
1 −ηµ

2
t ≤
2
µη, we get the result.

Corollary 12. Let the assumptions of Theorem E.2 hold, γ = η/n, α =
1
1+ω, and

η = min

(
α
2µ,
1
16Lmax
 
1 + 9ω

M
,
r εµn

9Lmax

 
(n + 1)ζ2
⋆+ σ2
⋆
−1/2
)

.
(31)

Then, DIANA-NASTYA finds a solution with accuracy ε > 0 after the following number of communi-
cation rounds:

e
O

 

ω + Lmax

µ


1 + ω

M


+

s

Lmax

εµ3

q

ζ2⋆+ σ2
⋆/n

!

.

If γ →0, one can choose η = min

α
2µ,
1
16Lmax(1+ 9ω

M )


such that the number of communication

rounds T to find solution with accuracy ε > 0 is

e
O

ω + Lmax

µ


1 + ω

M


.

Proof. Theorem E.2 implies

E [ΨT ] ≤

1 −ηµ

2

T
Ψ0 + 9

2
γ2nLmax

µ
 
(n + 1)ζ2
⋆+ σ2
⋆

.

To estimate the number of communication rounds required to find a solution with accuracy ε > 0, we
need to upper bound each term from the right-hand side by ε

2. Thus, we get an additional restriction
on η:
9
2
η2Lmax

nµ
 
(n + 1)ζ2
⋆+ σ2
⋆

< ε

2,

and also the upper bound on the number of communication rounds T

T = e
O
 1

ηµ


.

Substituting (31) in the previous equation, we get the first part of the result. When γ →0, the proof
follows similar steps.

60


---Page Break---
H
Alternative Analysis of Q-NASTYA

In this analysis, we will use additional sequence:

xi
⋆,m = x⋆−γ

i−1
X

j=0
∇fm(x⋆).
(32)

Theorem H.1. Let Assumptions 1, 3, 4 hold. Moreover, we assume that (1 −γµ)n ≤
9/10−1/C

1+1/C
=
bC < 1 for some numerical constant C > 1. Also let β =
η
γn ≤
1
3C ω

M +1 and γ ≤
1
Lmax . Then, for all
T ≥0 the iterates produced by Q-NASTYA satisfy

≤max

1 −β

10, 1 −α

2


Ψt + 2

µβγ2ˆσ2
rad
(33)

E

∥xT −x⋆∥2
≤

1 −β

10


∥xt −x∗∥2 + 4

µβγ2ˆσ2
rad + 3β2 ω

M
1
M
ˆ∆⋆,

where ˆ∆⋆=
1
M
PM
m=1 ∥xn
⋆,m −x⋆∥2 and ˆσ2
rad ≤Lmax
 
ζ2
⋆+ nσ2
⋆/4

.

Proof. The update rule for one epoch can be rewritten as

xt+1 = xt −η 1

M

m=1
X

M
Q
xt −xn
t,m
γn


.

Using this, we derive

∥xt+1 −x∗∥2 =

xt −η 1

M

M
X

m=1
Q
xt −xn
t,m
γn


−x∗



2

= ∥xt −x∗∥2 −2η

*

xt −x∗, 1

M

M
X

m=1
Q
xt −xn
t,m
γn

+

+ η2

1
M

M
X

m=1
Q
xt −xn
t,m
γn



2

.

Taking conditional expectation w.r.t. the randomness comming from compression, we get

EQ∥xt+1 −x∗∥2 = ∥xt −x∗∥2 −2η

*

xt −x∗, 1

M

M
X

m=1

xt −xn
t,m
γn

+

+ η2EQ


1
M

M
X

m=1
Q
xt −xn
t,m
γn



2

.

Next, we use the definition of quantization operator and independence of Q
 xt−xn
t,m
γn

, m ∈[M]:

EQ∥xt+1 −x∗∥2 ≤∥xt −x∗∥2 −2η

*

xt −x∗, 1

M

M
X

m=1

xt −xn
t,m
γn

+

+ η2



ω

M
1
M

M
X

m=1


xt −xn
t,m
γn



2
+


1
M

M
X

m=1

xt −xn
t,m
γn



2

.

61


---Page Break---
Since β =
η
γn, we obtain

EQ∥xt+1 −x∗∥2 ≤∥xt −x∗∥2 −2β

*

xt −x∗, 1

M

M
X

m=1

 
xt −xn
t,m

+

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2 + β2

1
M

M
X

m=1

 
xt −xn
t,m



2

= ∥xt −x∗∥2 + 2β

*

xt −x∗, 1

M

M
X

m=1

 
xn
t,m −xt

+

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2 + β2

1
M

M
X

m=1

 
xn
t,m −xt



2

=

xt −x∗+ β

 
1
M

M
X

m=1

 
xn
t,m −xt

!

2

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2

=

(1 −β)(xt −x∗) + β

 
1
M

M
X

m=1

 
xn
t,m

−x∗

!

2

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2 .

Using the condition that x∗=
1
M
PM
m=1 xn
∗,m we have:

EQ∥xt+1 −x∗∥2 ≤

(1 −β)(xt −x∗) + β

 
1
M

M
X

m=1

 
xn
t,m −xn
∗,m

!

2

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2 .

Convexity of squared norm and Jensen’s inequality imply

EQ∥xt+1 −x∗∥2 ≤(1 −β)∥xt −x∗∥2 + β


1
M

M
X

m=1

 
xn
t,m −xn
∗,m



2

+ β2 ω

M
1
M

M
X

m=1

xt −xn
t,m
2 .

Next, from Young’s inequality we get

EQ∥xt+1 −x∗∥2 ≤(1 −β)∥xt −x∗∥2 + β


1
M

M
X

m=1

 
xn
t,m −xn
∗,m



2

+ 3β2 ω

M ∥xt −x∗∥2

+ 3β2 ω

M
1
M

M
X

m=1
∥xn
t,m −xn
∗,m∥2 + 3β2 ω

M
1
M

M
X

m=1
∥xn
∗,m −x∗∥2.

Theorem 4 from [Mishchenko et al., 2021] gives

E

"
1
M

M
X

m=1
∥xn
t,m −xn
∗,m∥2
#

≤(1 −γµ)n h
∥xt −x⋆∥2i
+ 2γ3ˆσ2
rad




n−1
X

j=0
(1 −γµ)j





= (1 −γµ)n h
∥xt −x⋆∥2i
+ 2γ2ˆσ2
rad
1
γµ.

62


---Page Break---
It leads to

E∥xt+1 −x∗∥2 ≤(1 −β)∥xt −x∗∥2 + β

(1 −γµ)n h
∥xt −x⋆∥2i
+ 2γ3ˆσ2
rad
1
γµ



+ 3β2 ω

M ∥xt −x∗∥2 + 3β2 ω

M


(1 −γµ)n h
∥xt −x⋆∥2i
+ 2γ3ˆσ2
rad
1
γµ



+ 3β2 ω

M
1
M

M
X

m=1
∥xn
∗,m −x∗∥2

≤

1 −β + β(1 −γµ)n + 3β2 ω

M + 3β2 ω

M (1 −γµ)n
∥xt −x∗∥2

+ 2βγ3ˆσ2
rad
1
γµ


1 + 3β ω

M


+ 3β2 ω

M
1
M

M
X

m=1
∥xn
∗,m −x∗∥2.

Using (1 −γµ)n ≤
9/10−1/C

1+1/C , we have

(1 −γµ)n ≤
9/10 −1/C

1 + 1/C

(1 −γµ)n

1 + 1

C


≤9

10 −1

C

−9

10β + β(1 −γµ)n + β

C + β

C (1 −γµ)n ≤0

1 −β + β(1 −γµ)n + β

C + β

C (1 −γµ)n ≤1 −β

10.

Next, applying β ≤
1
1+3C ω

M , we derive

1 −β + β(1 −γµ)n + 3β2 ω

M + 3β2 ω

M (1 −γµ)n ≤1 −β

10.

Finally, we have

E∥xt+1 −x⋆∥2 ≤

1 −β

10


∥xt −x∗∥2 + 2βγ2ˆσ2
rad
1
µ


1 + 1

C



+ 3β2 ω

M
1
M

M
X

m=1
∥xn
∗,m −x∗∥2

≤

1 −β

10


∥xt −x∗∥2 + 4

µβγ2ˆσ2
rad

+ 3β2 ω

M
1
M

M
X

m=1
∥xn
∗,m −x∗∥2.

63


---Page Break---
I
Alternative Analysis of DIANA-NASTYA

Theorem I.1. Let Assumptions 1, 3, 4 hold. Moreover, we assume that (1−γµ)n ≤
9/10−1/B

1+1/B
= bB < 1
for some numerical constant B > 1. Also let β =
η
γn ≤
1
12B ω

M +1 and γ ≤
1
Lmax and also α ≤
1
ω+1.
Then, for all T ≥0 the iterates produced by DIANA-NASTYA satisfy

EΨT ≤max

1 −β

10, 1 −α

2

T
Ψ0 +
2
µ min( β

10, α

2 )
βγ2ˆσ2
rad.
(34)

Proof. We start with expanding the square:

∥xt+1 −x∗∥2 = ∥xt −ηˆgt −x∗∥2

=

xt −η 1

M

M
X

m=1
(ht,m + Q(gt,m −ht,m)) −x∗



2

= ∥xt −x∗∥2 −2η

*
1
M

M
X

m=1
(ht,m + Q(gt,m −ht,m)) , xt −x∗

+

+ η2

1
M

M
X

m=1
(ht,m + Q(gt,m −ht,m))



2

.

Taking the expectation w.r.t. Q, we get

EQ∥xt+1 −x∗∥2 = ∥xt −x∗∥2 −2η

*
1
M

M
X

m=1
gt,m, xt −x∗

+

+ η2EQ


1
M

M
X

m=1
(ht,m + Q(gt,m −ht,m))



2

= ∥xt −x∗∥2 −2η

*
1
M

M
X

m=1
gt,m, xt −x∗

+

+ η2EQ


1
M

M
X

m=1
(ht,m + Q(gt,m −ht,m) −gt,m)



2

+ η2

1
M

M
X

m=1
gt,m



2

≤∥xt −x∗∥2 −2η

*
1
M

M
X

m=1
gt,m, xt −x∗

+

+ η2 ω

M 2

M
X

m=1
∥gt,m −ht,m∥2 + η2

1
M

M
X

m=1
gt,m



2

≤∥xt −x∗∥2 −2η

*
1
M

M
X

m=1
gt,m, xt −x∗

+

+ η2 2ω

M 2

M
X

m=1
∥gt,m −h∗,m∥2 + η2 2ω

M 2

M
X

m=1
∥ht,m −h∗,m∥2 + η2

1
M

M
X

m=1
gt,m



2

.

64


---Page Break---
Next, using definition of gt,m, we obtain

E∥xt+1 −x∗∥2 ≤∥xt −x∗∥2 −2η

*
1
M

M
X

m=1

xt −xn
t,m
γn
, xt −x∗

+

+ η2

1
M

M
X

m=1

xt −xn
t,m
γn



2

+ η2 2ω

M 2

M
X

m=1
∥gt,m −h∗,m∥2 + η2 2ω

M 2

M
X

m=1
∥ht,m −h∗,m∥2

= ∥xt −x∗∥2 + 2α

*
1
M

M
X

m=1

 
xn
t,m −xt

, xt −x∗

+

+ α2

1
M

M
X

m=1

 
xn
t,m −xt



2

+ η2 2ω

M 2

M
X

m=1
∥gt,m −h∗,m∥2 + η2 2ω

M 2

M
X

m=1
∥ht,m −h∗,m∥2

=

xt −x∗+ α 1

M

M
X

m=1

 
xn
t,m −xt



2

+ η2 2ω

M 2

M
X

m=1
∥gt,m −h∗,m∥2 + η2 2ω

M 2

M
X

m=1
∥ht,m −h∗,m∥2

=

(1 −β)(xt −x∗) + β

 
1
M

M
X

m=1

 
xn
t,m −xn
∗,m

!

2

≤(1 −β)∥xt −x∗∥2 + β 1

M

M
X

m=1
∥xn
t,m −xn
∗,m∥2

+ η2 2ω

M 2

M
X

m=1
∥gt,m −h∗,m∥2 + η2 2ω

M 2

M
X

m=1
∥ht,m −h∗,m∥2.

Let us consider recursion for control variable:

∥ht+1,m −h∗,m∥2 = ∥ht,m + αQ(gt,m −ht,m) −h∗,m∥2

= ∥ht,m −h∗,m∥2 + α ⟨Q(gt,m −ht,m), ht,m −h∗,m⟩+ α2∥Q(gt,m −ht,m)∥2.

Taking the expectation w.r.t. Q, we have

EQ∥ht+1,m −h∗,m∥2 ≤∥ht,m −h∗,m∥2 + 2α ⟨gt,m −ht,m, ht,m −h∗,m⟩+ α2 (ω + 1) ∥gt,m −ht,m∥2 .

Using α ≤
1
ω+1 we have

E∥ht+1,m −h∗,m∥2 ≤∥ht,m −h∗,m∥2

+ 2α ⟨gt,m −ht,m, ht,m −h∗,m⟩+ α ∥gt,m −ht,m∥2

= ∥ht,m −h∗,m∥2

+ 2α ⟨gt,m −ht,m, ht,m −h∗,m⟩+ α ⟨gt,m −ht,m, gt,m −ht,m⟩

= ∥ht,m −h∗,m∥2

+ α ⟨gt,m −ht,m, gt,m −ht,m + 2ht,m −2h∗,m⟩

= ∥ht,m −h∗,m∥2

+ α ⟨gt,m −ht,m, gt,m + ht,m −2h∗,m⟩

= ∥ht,m −h∗,m∥2

+ α ⟨gt,m −ht,m −h∗,m + h∗,m, gt,m + ht,m −2h∗,m⟩

= ∥ht,m −h∗,m∥2

+ α ⟨gt,m −h∗,m −(ht,m −h∗,m), (gt,m −h∗,m) + (ht,m −h∗,m)⟩

= ∥ht,m −h∗,m∥2 + α∥gt,m −h∗,m∥2 −α∥ht,m −h∗,m∥2

= (1 −α)∥ht,m −h∗,m∥2 + α∥gt,m −h∗,m∥2.

65


---Page Break---
Using this bound we get that

1
M

M
X

m=1
EQ∥ht+1,m −h∗,m∥2 ≤(1 −α) 1

M

M
X

m=1
∥ht,m −h∗,m∥2 + α 1

M

M
X

m=1
∥gt,m −h∗,m∥2.

Let us consider Lyapunov function:

Ψt = ∥xt −x∗∥2 + 4ωη2

αM
1
M

M
X

m=1
∥ht,m −h∗,m∥2.

Using previous bounds and Theorem 4 from [Mishchenko et al., 2021] we have

EΨt+1 ≤(1 −β)∥xt −x∗∥2 + β

(1 −γµ)nE∥xt −x∗∥2 + γ3 1

γµ ˆσ2
rad



+ η2 2ω

M
1
M

M
X

m=1
E∥gt,m −h∗,m∥2 + η2 2ω

M
1
M

M
X

m=1
E∥ht,m −h∗,m∥2

+ (1 −α)4ωη2

αM
1
M

M
X

m=1
E∥ht,m −h∗,m∥2 + α4ωη2

αM
1
M

M
X

m=1
E∥gt,m −h∗,m∥2

≤

1 −α

2

 4ωη2

αM
1
M

M
X

m=1
E∥ht,m −h∗,m∥2 + η2 6ω

M
1
M

M
X

m=1
E∥gt,m −h∗,m∥2

+ (1 −β)E∥xt −x∗∥2 + β

(1 −γµ)nE∥xt −x∗∥2 + γ3 1

γµ ˆσ2
rad.


Let us consider

η2 1

M

M
X

m=1
E∥gt,m −h∗,m∥2 = η2 1

M

M
X

m=1
E

xt −xn
t,m
γn
−x∗−xn
∗,m
γn



2

≤2η2 1

M

M
X

m=1
E

xt −x∗

γn



2
+ 2η2 1

M

M
X

m=1
E

xn
t,m −xn
∗,m
γn



2

≤2β2 1

M

M
X

m=1
E ∥xt −x∗∥2 + 2β2 1

M

M
X

m=1
E
xn
t,m −xn
∗,m
2

≤2β2E ∥xt −x∗∥2 + 2β2 1

M

M
X

m=1
E
xn
t,m −xn
∗,m
2 .

Putting all the terms together and using (1 −γµ)n ≤
9/10−1/B

1+1/B
= bB < 1, β ≤
1
12B ω

M +1 we have

EΨt+1 ≤

1 −β + 12 ω

M β2 + 12 ω

M β2(1 −γµ)n + β(1 −γµ)n
E∥xt −x∗∥2 + βγ3 1

γµ ˆσ2
rad

+ 2β2 6ω

M γ3 1

γµ ˆσ2
rad +

1 −α

2

 4ωη2

αM
1
M

M
X

m=1
E∥ht,m −h∗,m∥2

≤

1 −β

10


E∥xt −x⋆∥2 + 2

µβγ2ˆσ2
rad +

1 −α

2

 4ωη2

αM
1
M

M
X

m=1
E∥ht,m −h∗,m∥2

≤max

1 −β

10, 1 −α

2


Ψt + 2

µβγ2ˆσ2
rad.

Unrolling this recursion we get the final result.

66


---Page Break---
Algorithm 5 Q-NASTYA-PP

Input: x0 – starting point, γ > 0 – local stepsize, η > 0 – global stepsize

1: for t = 0, 1, . . . , T −1 do
2:
Sample a cohort St with cardinality C uniformly
3:
for m ∈St in parallel do
4:
Receive xt from the server and set x0
t,m = xt
5:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
6:
for i = 0, 1, . . . , n −1 do

7:
Set xi+1
t,m = xi
t,m −γ∇f πi
m
m (xi
t,m)
8:
end for
9:
Compute gt,m =
1
γn
 
xt −xn
t,m

and send Qt(gt,m) to the server
10:
end for
11:
Compute gt = 1

C
P

m∈St Qt(gt,m)
12:
Compute xt+1 = xt −ηgt and send xt+1 to the workers
13: end for
Output: xT

J
Partial Participation for Method with Local Steps

J.1
Analysis of Q-NASTYA with Partial Participation

Lemma J.1. Let Assumptions 1, 2, 3 hold. Then, for all t ≥0 the iterates produced by Q-NASTYA-PP
(Algorithm 5) satisfy

EQ,St
h
∥gt∥2i
≤2L2
max
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 8Lmax

1 + ω

C


(f (xt) −f (x⋆))

+ 4
 ω

C +
M −C
C max M −1, 1


σ2
⋆,

where EQ,St is expectation w.r.t. Q, St and σ2
⋆=
1
M
PM
m=1 ∥∇fm (x⋆)∥2.

Proof. E

∥ξ∥2
= E

∥ξ −E[ξ]∥2
+ ∥Eξ∥2, we obtain

EQ

∥gt∥2

=EQ






1
C

X

m∈St

 

Q

 
1
n

n−1
X

i=0
∇f
πi
m
m

xi
t,m
!

−1

n

n−1
X

i=0
∇f
πi
m
m

xi
t,m
!

+
1
Cn

X

m∈St

n−1
X

i=0
∇f
πi
m
m

xi
t,m


2



= 1

C2 EQ∥
X

m∈St
(

 
1
n

n−1
X

i=0
∇f
πi
m
m

xi
t,m
!

−1

n

n−1
X

i=0
∇f
πi
m
m

xi
t,m


|
{z
}
=ξm

∥2]

+


1
Cn

X

m∈St

n−1
X

i=0
∇f
πi
m
m

xi
t,m


2

= 1

C2 EQ



X

m∈St
∥ξm∥2 +
X

m,l∈St:m̸=l
2 ⟨ξm, ξl⟩



+


1
Cn

X

m∈St

n−1
X

i=0
∇f
πi
m
m

xi
t,m


2

.

67


---Page Break---
Using independence between ξm and ξl for different m, l and Using (2), (3), we get

EQ
h
∥gt∥2i
= 1

C2
X

m∈St
EQ





Q

 
1
n

n−1
X

i=0
∇f πi
m
m
 
xi
t,m

!

−1

n

n−1
X

i=0
∇f πi
m
m
 
xi
t,m



2



+


1
Cn

X

m∈St

n−1
X

i=0
∇f πi
m
m
 
xi
t,m



2

≤ω

C2
X

m∈St


1
n

n−1
X

i=0
∇f πi
m
m
 
xi
t,m



2

+


1
Cn

X

m∈St

n−1
X

i=0
∇f πi
m
m
 
xi
t,m



2

.

Rewriting previous inequality and using ∇fm(x) = 1

n

n−1
X

i=0
∇f πi
m
m (xt) , we have

EQ
h
∥gt∥2i
≤2ω

C2
X

m∈St


1
n

n−1
X

i=0


∇f πi
m
m
 
xi
t,m

−∇f πi
m
m (xt)


2

+ 2ω

C2
X

m∈St
∥∇fm (xt)∥2

+ 2


1
Cn

X

m∈St

n−1
X

i=0


∇f πi
m
m
 
xi
t,m

−∇f πi
m
m (xt)


2

+ 2


1
C

X

m∈St
∇fm (xt)



2

≤2
 
1 + ω

C


C

X

m∈St


1
n

n−1
X

i=0


∇f πi
m
m
 
xi
t,m

−∇f πi
m
m (xt)


2

+ 2ω

C2
X

m∈St
∥∇fm (xt)∥2 + 2


1
C

X

m∈St
∇fm (xt)



2

Using L-smoothness of f i
m and f and also convexity of fm, we obtain

EQ
h
∥gt∥2i
≤2
 
1 + ω

C


Cn

X

m∈St

n−1
X

i=0

∇f πi
m
m
 
xi
t,m

−∇f πi
m
m (xt)

2

+ 4ω

C2
X

m∈St
∥∇fm (xt) −∇fm (x⋆)∥2

+ 4ω

C2
X

m∈St
∥∇fm (x⋆)∥2 + 4


1
C

X

m∈St
(∇fm (xt) −∇fm (x⋆))



2

+ 4


1
C

X

m∈St
∇fm (x⋆)



2

≤2L2
max
 
1 + ω

C


Cn

X

m∈St

n−1
X

i=0

xi
t,m −xt
2 + 8Lmax
 
1 + ω

C


C

X

m∈St
Dfm (xt, x⋆)

+ 4ω

C2
X

m∈St
∥∇fm (x⋆)∥2 + 4


1
C

X

m∈St
∇fm (x⋆)



2

.

Taking expectation w.r.t. St and using uniform sampling, we receive

68


---Page Break---
EQ,St
h
∥gt∥2i
≤2L2
max
 
1 + ω

C


n
ESt

"
1
C

X

m∈St

n−1
X

i=0
∥xi
t,m −xt|2
#

+ 8Lmax

1 + ω

C


ESt

"
1
C

X

m∈St
Dfm (xt, x⋆)

#

+ 4ω

C ESt

"
1
C

X

m∈St
∥∇fm (x⋆)∥2
#

+ 4ESt






1
C

X

m∈St
∇fm (x⋆)



2



≤2L2
max
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 8Lmax
 
1 + ω

C


M

M
X

m=1
Dfm (xt, x⋆)

+ 4ω

C
1
M

M
X

m=1
∥∇fm (x⋆)∥2 + 4
M −C
MC max M −1, 1

M
X

m=1
∥∇fm (x⋆)∥2 .

Theorem J.1. Let step sizes η, γ satisfy the following equations

η =
1
16Lmax
 
1 + ω

C
,
γ =
1
5nLmax

Under the Assumptions 1, 2, 3 iterates of Q-NASTYA-PP (Algorithm 5) satisfy

E
h
∥xT −x⋆∥2i
≤

1 −ηµ

2

T
∥x0 −x⋆∥2 + 9

2
γ2nLmax

µ

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+ 8 η

µ

 ω

C σ2
⋆+
M −C
C max(M −1, 1)σ2
⋆


,

where

σ2
⋆= 1

M

M
X

m=1
∥∇fm (x⋆)∥2 ,
σ2
⋆,m = 1

n

∇f i
m (x⋆)
2

As we can see, there is an additional error term proportional to
M−C
C max(M−1,1) that arises due to client
sampling in the partial participation setting. Note that when C = M (all clients are participating),
this error term vanishes, allowing us to recover the previous result for the full participation case. This
shows the consistency of our theoretical framework across different participation scenarios.

69


---Page Break---
Proof.

Taking expectation w.r.t. Q, St and using Lemma J.1 updated, we get

EQ,St
h
∥xt+1 −x⋆|2i

=∥xt −x⋆|2 −2ηEQ,St [⟨gt, xt −x⋆⟩] + η2EQ,St
hgt2i

≤∥xt −x⋆∥2 −2ηEQ,St

"*
1
C

X

m∈St
Q

 
1
n

n−1
X

i=0
∇f πi
m
m
 
xi
t,m

!

, xt −x⋆
+#

+ 2η2L2
max
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 8η2Lmax

1 + ω

C


(f (xt) −f (x⋆))

+ 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆

≤∥xt −x⋆∥2 −2η 1

Mn

M
X

m=1

n−1
X

i=0


∇f πi
m
 
xi
t,m

, xt −x⋆

+ 2η2L2
max
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 8η2Lmax

1 + ω

C


(f (xt) −f (x⋆))

+ 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆.

Using Lemma F.2, we obtain

EQ,St
h
∥xt+1 −x⋆∥2i

≤∥xt −x⋆∥2 −ηµ

2 ∥xt −x⋆∥2 −η (f (xt) −f (x⋆))

+ 8η2Lmax

1 + ω

C


(f (xt) −f (x⋆)) + ηLmax

Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+ 2η2L2
max
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆

≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηLmax

1 + ω

C


(f (xt) −f (x⋆))

+ ηLmax
 
1 + 2ηLmax
 
1 + ω

C


Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 +

4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆.

Using Lemma F.3, we have

EQ,St
h
∥xt+1 −x⋆∥2i
≤

1 −ηµ

2


∥xt −x⋆∥2 −η

1 −8ηL

1 + ω

C


(f (xt) −f (x⋆))

+ ηLmax

1 + 2ηLmax

1 + ω

C


· 8γ2n2Lmax (f (xt) −f (x⋆))

+ ηLmax

1 + 2ηLmax

1 + ω

C


· 2γ2n

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+ 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆.

Finally, we receive

70


---Page Break---
EQ,St
h
∥xt+1 −x⋆∥2i

≤

1 −ηµ

2


∥xt −x⋆∥2 + 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆

−η

1 −8ηLmax

1 + ω

C


−8γ2n2L2
max

1 + 2Lmaxη

1 + ω

C


(f (xt) −f (x⋆))

+ 2γ2nηLmax

1 + 2ηL

1 + ω

C

  
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

≤

1 −ηµ

2


∥xt −x⋆∥2 + 4η2
 ω

C +
M −C
C max M −1, 1


σ2
⋆

+ 9

4ηLmaxγ2n

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

Recursively rewriting the inequality and using P+∞
t=0
 
1 −ηµ

2
t ≤
2
µη, we finish proof.

71


---Page Break---
Algorithm 6 DIANA-NASTYA-PP

Input: x0 – starting point, {h0,m}M
m=1 – initial shift-vectors, γ > 0 – local stepsize, η > 0 – global
stepsize, α > 0 – stepsize for learning the shifts
1: for t = 0, 1, . . . , T −1 do
2:
Sample a cohort St with cardinality C uniformly
3:
for m ∈St in parallel do
4:
Receive xt from the server and set x0
t,m = xt
5:
Sample random permutation of [n]: πm = (π0
m, . . . , πn−1
m
)
6:
for i = 0, 1, . . . , n −1 do

7:
Set xi+1
t,m = xi
t,m −γ∇f πi
m
m (xi
t,m)
8:
end for
9:
Compute gt,m =
1
γn
 
xt −xn
t,m

and send Qt (gt,m −ht,m) to the server
10:
Set ht+1,m = ht,m + αQt (gt,m −ht,m)
11:
Set ˆgt,m = ht,m + Qt (gt,m −ht,m)
12:
end for
13:
ht+1 = 1

C
P

m∈St ht+1,m = ht + α

C
P

m∈St Qt (gt,m −ht,m)
14:
ˆgt = 1

C
P

m∈St ˆgt,m = ht + 1

C
P

m∈St Qt (gt,m −ht,m)
15:
xt+1 = xt −ηˆgt
16: end for
Output: xT

J.2
Analysis of DIANA-NASTYA with Partial Participation

Theorem J.2. Let step sizes η, γ satisfy the following equations

η = min

 
1
80Lmax
 
1 + ω

C
,
C
µ(1 + ω)M

!

,
γ =
1
5nLmax

Define the Lyapunov function:

Ψt = ∥xt −x⋆∥2 + A

M

M
X

m=1
∥ht,m −h⋆
m∥2 ,

where A = λη2. Selecting parameters α =
1
1+ω; λ =
8ω
αM , γ =
1
5nLmax , also using η ≤

min

C
µ(1+ω)M ,
1
80Lmax(1+ ω

C )


Under the Assumptions 1, 2, 3 iterates of DIANA-NASTYA-PP (Algo-

rithm 6) satisfy

E [ΨT ] ≤

1 −ηµ

2

T
E [Ψ0] + 3γ2n2L2
max
µ

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+
2η(M −C)
µC max(1, M −1)σ2
⋆.

Note that we eliminate the variance term proportional to ω : 8 η

µ
ω
C σ2
⋆. In the Partial Participation

regime, we have a variance term proportional to
(M−C)
C max(1,M−1), which equals zero if C = M. This
term decreases as O
  1

C

, so we achieve the expected linear speedup.

72


---Page Break---
Proof. STEP 1: we need to estimate inner product. By ˆgt = 1

C
P
m∈St ˆgt,m, we have

−Et

"*
1
C

X

m∈St
ˆgt,m, xt −x⋆
+#

= −

*
1
C Et

" X

m∈St
ˆgt,m

#

, xt −x⋆
+

= −

*
1
M

M
X

m=1
Et [ˆgt,m] , xt −x⋆

+

= −1

M

M
X

m=1
⟨gt,m, xt −x⋆⟩

= −1

M

M
X

m=1
⟨gt,m −h⋆
m, xt −x⋆⟩

≤−µ

4 ∥xt −x⋆∥2 −1

2 (f (xt) −f (x⋆))

−
1
Mn

M
X

m=1

n−1
X

i=0
D
f
πim
m
m
 
x⋆, xi
t,m


+ Lmax

2Mn

M
X

m=1

n−1
X

i=0

xt −xi
t,m
2 .

STEP 2: We need to bound E ∥ˆgt∥2 . By ˆgt = 1

C

X

m∈St
ˆgt,m, we have

EQ
h
∥ˆgt∥2i
=EQ






1
C

X

m∈St
(ht,m + Q (gt,m −ht,m) −gt,m + gt,m)



2



=EQ






1
C

X

m∈St
(ht,m + Q (gt,m −ht,m) −gt,m)



2

+


1
C

X

m∈St
gt,m



2

= 1

C2
X

m∈St
EQ
h
∥ht,m + Q (gt,m −ht,m)∥2i
+


1
C

X

m∈St
gt,m



2

≤ω

C2
X

m∈St
∥gt,m −ht,m∥2 +


1
C

X

m∈St
gt,m



2

≤2ω

C2
X

m∈St
∥gt,m −∇fm (xt)∥2 + 2ω

C2
X

m∈St
∥∇fm (xt) −ht,m∥2

+ 2


1
C

X

m∈St
gt,m −h⋆
m



2

+ 2


1
C

X

m∈St
h⋆
m



2

Taking expectation by subsamling, we have

73


---Page Break---
EQ,St
h
∥ˆgt∥2i
≤2ω

C
1
M

M
X

m=1
∥gt,m −∇fm (xt)∥2 + 2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2

+ 2

M

M
X

m=1
∥gt,m −h⋆
m∥2 +
2(M −C)
C(M −1)M

M
X

m=1
∥h⋆
m∥2

≤2ω

C
1
M

M
X

m=1
∥gt,m −∇fm (xt)∥2 + 2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2

+ 4

M

M
X

m=1
∥gt,m −∇fm (xt)∥2 + 4

M

M
X

m=1
∥∇fm (xt) −h⋆
m∥2

+
2(M −C)
C(M −1)M

M
X

m=1
∥h⋆
m∥2

≤4

1 + ω

C

 L2
max
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2

+ 8Lmax

M

M
X

m=1
Dfm (xt, x⋆) +
2(M −C)
C(M −1)M

M
X

m=1
∥h⋆
m∥2

=

1 + ω

C

 L2
max
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2

+ 8Lmax (f (xt) −f (x⋆)) +
2(M −C)
C(M −1)M

M
X

m=1
∥h⋆
m∥2

Thus, we have

EQ,St
h
∥xt+1 −x⋆∥2i
≤

1 −ηµ

2


∥xt −x⋆∥2 −η (1 −4Lmaxη) (f (xt) −f (x⋆))

+ ηLmax

1 + 4

1 + ω

C


Lmaxη

1
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+ 2η2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2 + 2η2(M −C)

C(M −1)M

M
X

m=1
∥h⋆
m∥2 .

STEP 3: Note that

1
M

M
X

m=1
∥ht+1,m −h⋆
m∥2 = C

M
1
C

X

m∈St
∥ht+1,m −h⋆
m∥2+M −C

M
1
M −C

X

m/∈St
∥ht+1,m −h⋆
m∥2 .

Taking expectation by compression, we have

74


---Page Break---
EQ

"
1
C

X

m∈St
∥ht+1,m −h⋆
m∥2
#

= EQ

"
1
C

X

m∈St
∥ht,m + αQ (gt,m −ht,m) −h⋆
m∥2
#

= 1

C

X

m∈St


∥ht,m −h⋆
m∥2 + 2α ⟨gt,m −ht,m, ht,m −h⋆
m⟩+ α2(1 + ω) ∥gt,m −ht,m∥2

α≤1/1+ω
≤
1
C

X

m∈St


∥ht,m −h⋆
m∥2 + 2α ⟨gt,m −ht,m, ht,m −h⋆
m⟩+ α ∥gt,m −ht,m∥2

= 1 −α

C

X

m∈St
∥ht,m −h⋆
m∥2 + α

C

X

m∈St
∥gt,m −ht,m∥2 .

Taking expectation by subsampling, we have

EQ,St

"
1
C

X

m∈St
∥ht+1,m −h⋆
m∥2
#

≤ESt

"
1 −α

C

X

m∈St
∥ht,m −h⋆
m∥2 + α

C

X

m∈St
∥gt,m −h⋆
m∥2
#

= 1 −α

M

M
X

m=1
∥ht,m −h⋆
m∥2 + α

M

M
X

m=1
∥gt,m −h⋆
m∥2 .

Thus, we have

ESt,Qt

"
1
M

M
X

m=1
∥ht+1,m −h⋆
m∥2
#

=(1 −α)C

M 2

M
X

m=1
∥ht,m −h⋆
m∥2 + αC

M 2

M
X

m=1
∥gt,m −h⋆
m∥2

+ M −C

M
ESt,Qt




1
M −C

X

m/∈St
∥ht,m −h⋆
m∥2





=(1 −α)C

M 2

M
X

m=1
∥ht,m −h⋆
m∥2 + αC

M 2

M
X

m=1
∥gt,m −h⋆
m∥2

+ M −C

M
1
M

M
X

m=1
∥ht,m −h⋆
m∥2

≤

1 −αC

M

 1

M

M
X

m=1
∥ht,m −h⋆
m∥2

+ 2αL2
maxC
M 2n

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+ 4LmaxαC

M 2

M
X

m=1
Dfm (xt, x⋆) .

STEP 4: Defining Lyapunov function as follows

Ψt = ∥xt −x⋆∥2 + A

M

M
X

m=1
∥ht,m −h⋆
m∥2 ,

75


---Page Break---
we have

EQ,St [Ψt+1] ≤

1 −ηµ

2


∥xt −x⋆∥2 −η (1 −4Lmaxη) (f (xt) −f (x⋆))

+ ηLmax

1 + 4

1 + ω

C


Lmaxη

1
Mn

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2

+ 2η2ω

C
1
M

M
X

m=1
∥∇fm (xt) −ht,m∥2 + 2η2(M −C)

C(M −1)M

M
X

m=1
∥h⋆
m∥2

+

1 −αC

M

 A

M

M
X

m=1
∥ht,m −h⋆
m∥2

+ 2αL2
maxAC
M 2n

M
X

m=1

n−1
X

i=0

xi
t,m −xt
2 + 4LmaxαAC

M
(f (xt) −f (x⋆)) .

Setting A = λη2 and using Lemma F.3, we have

E [Ψt+1] ≤

1 −ηµ

2


E
h
∥xt −x⋆∥2i
+

1 −αC

M + 4ω

λC

 λη2

M

M
X

m=1
E
h
∥ht,m −h⋆
m∥2i

−η

1 −8ηLmax

1 + ω

C


−4ηLmaxαλ C

M


E [f (xt) −f (x⋆)]

+ 8γ2n2L2
maxη

1 + 4ηLmax

1 + ω

C


+ 2ηLmaxαλ C

M


E [f (xt) −f (x⋆)]

+ 2γ2n2L2
maxη

1 + 4ηLmax

1 + ω

C


+ 2ηLmaxαλ C

M

  
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+ 2η2(M −C)

C(M −1) σ2
⋆.

Selecting α =
1
1+ω;
λ =
8ω
αM ;
η
≤
C
µ(1+ω)M , also using η
=
1
80Lmax(1+ ω

C ),
γ
=

1
5nLmax and applying previous steps we obtain

E [Ψt+1] ≤

1 −ηµ

2


E [Ψk] + 3γ2n2L2
maxη

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+ 2η2(M −C)

C(M −1) σ2
⋆

−η
1

2 −10γ2n2L2
max


E [f (xt) −f (x⋆)]

≤

1 −ηµ

2


E [Ψk] + 3γ2n2L2
maxη

 
1
M

M
X

m=1
σ2
⋆,m + nσ2
⋆

!

+ 2η2(M −C)

C(M −1) σ2
⋆,

76


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We clearly outline our contributions in the abstract and introduction, and we
also include a dedicated Contributions section.
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
Justification: We clearly highlight all assumptions and limitations in the text.
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

77


---Page Break---
Justification: The main contribution of the paper is its theoretical analysis. We support the
paper with necessary definitions, assumptions, and lemmas.

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
Justification: The paper is supported by reproducible experiments, with all stochastic ele-
ments from pseudo-random generators fixed in advance. For details, see the supplementary
materials.

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

78


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We aim to make the paper and all source code for experiments open-sourced
to accelerate scientific findings in the field of Federated Learning and Machine Learning in
general. For details, see the supplementary materials.

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

Justification: We provide detailed guidelines for experiments setup in Appendix and in the
folder with experiment source code. For details, see the supplementary materials.

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

Justification: We provide detailed guidelines for experiments setup in Appendix and in the
folder with experiment source code. For details, see the supplementary materials.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

79


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
Justification: The paper provides information on the computer resources in Appendix.
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
Justification: Our research focuses on mathematical objects and does not involve human
subjects or participants. The data used for our experiments consists of publicly available
datasets. Our work does not explicitly address or examine the social implications of applying
this research in practice.
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
Justification: Our work operates on mathematical objects, and the essence of our work
provides a new optimization algorithm. Because of theoretical nature of our work the impact
discussion is not applied.

80


---Page Break---
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
Justification: Our paper provides an optimization algorithm with the required theory. The
data or models are not output assets of our work.
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
Justification: We provide references for used datasets and deep learning models used in
experiments.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.

81


---Page Break---
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

Answer: [Yes]

Justification: The output assets of our paper is Algorithm and Source code for experiments.

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

82


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

83


---Page Break---
