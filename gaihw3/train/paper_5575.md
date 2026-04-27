Byzantine Robustness and Partial Participation Can
Be Achieved at Once: Just Clip Gradient Differences

Grigory Malinovsky∗

KAUST†, MBZUAI‡
Peter Richt´arik
KAUST
Samuel Horv´ath
MBZUAI
Eduard Gorbunov§
MBZUAI

Abstract

Distributed learning has emerged as a leading paradigm for training large machine
learning models. However, in real-world scenarios, participants may be unreliable
or malicious, posing a significant challenge to the integrity and accuracy of the
trained models. Byzantine fault tolerance mechanisms have been proposed to
address these issues, but they often assume full participation from all clients, which
is not always practical due to the unavailability of some clients or communication
constraints. In our work, we propose the first distributed method with client
sampling and provable tolerance to Byzantine workers. The key idea behind the
developed method is the use of gradient clipping to control stochastic gradient
differences in recursive variance reduction. This allows us to bound the potential
harm caused by Byzantine workers, even during iterations when all sampled clients
are Byzantine. Furthermore, we incorporate communication compression into the
method to enhance communication efficiency. Under general assumptions, we
prove convergence rates for the proposed method that match the existing state-of-
the-art (SOTA) theoretical results. We also propose a heuristic on adjusting any
Byzantine-robust method to a partial participation scenario via clipping.

1
Introduction

Distributed optimization problems are a cornerstone of modern machine learning research. They
naturally arise in scenarios where data is distributed across multiple clients; for instance, this is typical
in Federated Learning (FL) (Koneˇcn´y et al., 2016; Kairouz et al., 2021). Such problems require
specialized algorithms adapted to the distributed setup. Additionally, the adoption of distributed
optimization methods is motivated by the sheer computational complexity involved in training modern
machine learning models. Many models deal with massive datasets and intricate architectures,
rendering training infeasible on a single machine (Li, 2020). Distributed methods, by parallelizing
computations across multiple machines, offer a pragmatic solution to accelerate training and address
these computational challenges, thus pushing the boundaries of machine learning capabilities.

To make distributed training accessible to the broader community, collaborative learning approaches
have been actively studied in recent years (Kijsipongse et al., 2018b; Ryabinin and Gusev, 2020;
Atre et al., 2021; Diskin et al., 2021a). In such applications, there is a high risk of the occurrence
of so-called Byzantine workers (Lamport et al., 1982; Su and Vaidya, 2016)—participants who can
violate the prescribed distributed algorithm/protocol either intentionally or simply because they are
faulty. In general, such workers may even have access to some private data of certain participants and
may collude to increase their impact on the training. Since the ultimate goal is to achieve robustness

∗Part of the work was done when G. Malinovsky was visiting MBZUAI.
†King Abdullah University of Science and Technology.
‡Mohamed bin Zayed University of Artificial Intelligence.
§Corresponding author: eduard.gorbunov@mbzuai.ac.ae.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
in the worst case, many papers in the field make no assumptions limiting the power of Byzantine
workers. Clearly, in this scenario, standard distributed methods based on the averaging of received
information (e.g., stochastic gradients) are not robust, even to a single Byzantine worker. Such a
worker can send an arbitrarily large vector that can shift the method arbitrarily far from the solution.
This aspect makes it non-trivial to design methods with provable robustness to Byzantines (Baruch
et al., 2019; Xie et al., 2020a). Despite all the challenges, multiple methods are developed/analyzed
in the literature (Alistarh et al., 2018; Allen-Zhu et al., 2021; Wu et al., 2020; Zhu and Ling, 2021;
Karimireddy et al., 2021, 2022; Gorbunov et al., 2022, 2023; Allouah et al., 2023).

However, literally, all existing methods with provable Byzantine robustness require the full (or close
to full) participation of clients or rely on extra assumptions. The requirement of full participation
is impractical for modern distributed learning problems since they can have millions of clients
(Bonawitz et al., 2017; Niu et al., 2020). In such scenarios, it is more natural to use sampling of
clients to speed up the training. Moreover, some clients can be unavailable at certain moments,
e.g., due to a poor connection, low battery, or simply because of the need to use the computing
power for some other tasks. Although partial participation of clients is a natural attribute of large-
scale collaborative training, it is not studied under the presence of Byzantine workers. Moreover,
this question is highly non-trivial: the existing methods can fail to converge if combined na¨ıvely
with partial participation since Byzantine can form a majority during particular rounds and, thus,
destroy the whole training with just one round of communications. Therefore, the field requires the
development of new distributed methods that are provably robust to Byzantine attacks and can work
with partial participation even when Byzantine workers form a majority during some rounds.

Our Contributions
We develop Byzantine-tolerant Variance-Reduced MARINA with Partial Par-
ticipation (Byz-VR-MARINA-PP, Algorithm 1) – the first distributed method having Byzantine
robustness and allowing partial participation of clients without strong additional assumptions. Our
method uses variance reduction to handle Byzantine workers and clipping of stochastic gradient
differences to bound the potential harm of Byzantine workers even when they form a majority during
particular rounds of communication. To make the method even more communication efficient, we
add communication compression. We prove the convergence of Byz-VR-MARINA-PP for general
smooth non-convex functions and Polyak-Łojasiewicz functions. In the special case of full participa-
tion, our complexity bounds recover the ones for Byz-VR-MARINA (Gorbunov et al., 2023) that are
the current SOTA convergence results. Moreover, we prove that in some cases, partial participation is
theoretically beneficial for Byz-VR-MARINA-PP. We also propose a simplified version of Byz-VR-
MARINA-PP with better neighborhood term in the convergence bounds (Byz-VR-MARINA-PP+,
Algorithm 3) and a heuristic on how to use clipping to adapt any Byzantine-robust method to the
partial participation setup and illustrate its performance in experiments.

1.1
Related Work

Below, we overview closely related works. Additional discussion is deferred to Appendix A.

Byzantine robustness. The primary vulnerability of standard distributed methods to Byzantine
attacks lies in the aggregation rule: even one worker can arbitrarily distort the average. Therefore,
many papers on Byzantine robustness focus on the application of robust aggregation rules, such as
the geometric median (Pillutla et al., 2022), coordinate-wise median, trimmed median (Yin et al.,
2018), Krum (Blanchard et al., 2017), and Multi-Krum (Damaskinos et al., 2019). However, simply
robustifying the aggregation rule is insufficient to achieve provable Byzantine robustness, as illustrated
by Baruch et al. (2019) and Xie et al. (2020a), who design special Byzantine attacks that can bypass
standard defenses. This implies that more significant algorithmic changes are required to achieve
Byzantine robustness, a point also formally proven by Karimireddy et al. (2021), who demonstrate that
permutation-invariant algorithms – i.e., algorithms independent of the order of stochastic gradients at
each step – cannot provably converge to any predefined accuracy in the presence of Byzantines.

Wu et al. (2020) are the first who exploit variance reduction to tolerate Byzantine attacks. They
propose and analyze the method called Byrd-SAGA, which uses SAGA-type (Defazio et al., 2014)
gradient estimators on the good workers and geometric median for the aggregation. Gorbunov
et al. (2023) develop another variance-reduced method called Byz-VR-MARINA, which is based on
(conditionally biased) GeomSARAH/PAGE-type (Horv´ath et al., 2023; Li et al., 2021) gradient
estimator and any robust aggregation in the sense of the definition from (Karimireddy et al., 2021,

2


---Page Break---
2022), and derive the improved convergence guarantees that are the current SOTA in the literature.
There are also many other approaches and we discuss some of them in Appendix A.

Partial participation and client sampling. In the context of Byzantine-robust learning, there exists
several works that develop and analyze methods with partial participation (Data and Diggavi, 2021;
El-Mhamdi et al., 2021; Boubouh et al., 2022; Allouah et al., 2024a). However, these works rely
on the restrictive assumption that the number of participating clients at each round is larger than
the number of Byzantine workers. In this case, Byzantines cannot form a majority, and standard
methods can be applied without any changes. In contrast, our method converges in more challenging
scenarios, e.g., Byz-VR-MARINA-PP provably converges even when the server samples one client,
which can be Byzantine. If the number of participating clients is such that Byzantine clients can form
majority, these methods have a certain probability of divergence and this probability grows with each
communication round. We provide a more detailed discussion in Appendix A.

2
Preliminaries

In this section, we formally introduce the problem, main definition, and assumptions used in the
analysis. That is, we consider finite-sum distributed optimization problem5

minx∈Rd

f(x) := 1

G
P

i∈G fi(x)
	
,
fi(x) := 1

m
Pm
j=1 fi,j(x)
∀i ∈G,
(1)

where G is a set of regular clients of size G := |G|. In the context of distributed learning, fi : Rd →R
corresponds to the loss function on the data of client i, and fi,j : Rd →R is the loss computed on the
j-th sample from the dataset of client i. Next, we assume that the set of all clients taking part in the
training is [n] = {1, 2, . . . , n} and G ⊆[n]. The remaining clients B := [n] \ G are Byzantine ones.
We assume that B := |B| := δrealn ≤δn, where δreal is an exact ratio of Byzantine workers and δ is
a known upper bound for δreal. We also assume that 0 ≤δreal ≤δ < 1/2 since otherwise Byzantine
workers form a majority and problem (1) becomes impossible to solve in general.

Notation. We use a standard notation for the literature on distributed stochastic optimization.
Everywhere in the text ∥x∥denotes a standard ℓ2-norm of x ∈Rd, ⟨a, b⟩refers to the standard
inner product of vectors a, b ∈Rd. The clipping operator is defined as follows: clipλ(x) :=
min{1, λ/∥x∥}x for x ̸= 0 and clipλ(0) := 0. Finally, Prob{A} denotes the probability of event A,
E[ξ] is the full expectation of random variable ξ, E[ξ | A] is the expectation of ξ conditioned on the
event A. We also sometimes use Ek[ξ] to denote an expectation of ξ w.r.t. the randomness coming
from step k.

Robust aggregator. We follow the definition from (Gorbunov et al., 2023) of (δ, c)-robust aggrega-
tion, which is a generalization of the definitions proposed by Karimireddy et al. (2021, 2022).
Definition 2.1 ((δ, c)-Robust Aggregator). Assume that {x1, x2, . . . , xn} is such that there exists
a subset G ⊆[n] of size |G| = G ≥(1 −δ)n for δ ≤δmax < 0.5 and there exists σ ≥0 such
that
1
G(G−1)
P

i,l∈G E
h
∥xi −xl∥2i
≤σ2 where the expectation is taken w.r.t. the randomness
of {xi}i∈G. We say that the quantity bx is (δ, c)-Robust Aggregator (δ, c)-RAgg) and write bx =
RAgg (x1, . . . , xn) for some c > 0, if the following inequality holds:

E

∥bx −¯x∥2
≤cδσ2,
(2)

where ¯x :=
1
|G|
P

i∈G xi. If additionally bx is computed without the knowledge of σ2, we say that bx
is (δ, c)-Agnostic Robust Aggregator (δ, c)-ARAgg and write bx = ARAgg (x1, . . . , xn).

One can interpret the definition as follows. Ideally, we would like to filter out all Byzantine workers
and compute just an average ¯x over the set of good clients. However, this is impossible in general
since we do not know apriori who are Byzantine workers. Instead of this, it is natural to expect that
the aggregation rule approximates the ideal average up in a certain sense, e.g., in terms of the expected
squared distance to ¯x. As Karimireddy et al. (2021) formally show, in terms of such criterion (E[∥bx −
¯x∥2]), the definition of (δ, c)-RAgg cannot be improved (up to the numerical constant). Moreover,

5For simplicity, we assume that all regular workers have the same size of local datasets. Our analysis can
be easily generalized to the case of different sizes of local datasets: this will affect only the value of L± from
Assumption D.3 for some sampling strategies.

3


---Page Break---
standard aggregators such as Krum (Blanchard et al., 2017), geometric median, and coordinate-wise
median do not satisfy Definition 2.1 (Karimireddy et al., 2021), though another popular standard
aggregation rule called coordinate-wise trimmed mean (Yin et al., 2018) satisfies Definition 2.1 as
shown by Allouah et al. (2023) through the more general definition of robust aggregation. To address
this issue, Karimireddy et al. (2021) develop the aggregator called CenteredClip and prove that it fits
the definition of (δ, c)-RAgg. Karimireddy et al. (2022) propose a procedure called Bucketing that
fixes Krum, geometric median, and coordinate-wise median, i.e., with Bucketing Krum, geometric,
and coordinate-wise median become (δ, c)-ARAgg, which is important for our algorithm since the
variance of the vectors received from regular workers changes over time in our method. We notice
here that δ is a part of the input that should satisfy δreal ≤δ ≤δmax.

Compression operators. In our work, we use standard unbiased compression operators with
relatively bounded variance (Khirirat et al., 2018; Horv´ath et al., 2023).

Definition 2.2 (Unbiased compression). Stochastic mapping Q : Rd →Rd is called unbiased
compressor/compression operator if there exists ω ≥0 such that for any x ∈Rd E[Q(x)] =
x,
E

∥Q(x) −x∥2
≤ω∥x∥2. For the given unbiased compressor Q(x), one can define the
expected density6 as ζQ := supx∈Rd E [∥Q(x)∥0], where ∥y∥0 is the number of non-zero components
of y ∈Rd.

In this definition, parameter ω reflects how lossy the compression operator is: the larger ω the
more lossy the compression. For example, this class of compression operators includes random
sparsification (RandK) (Stich et al., 2018) and quantization (Goodall, 1951; Roberts, 1962; Alistarh
et al., 2017). For RandK compression ω = d

K −1, ζQ = K and for ℓ2-quantization ω =
√

d−1, ζQ =
√

d, see the proofs in (Beznosikov et al., 2020).

Assumptions. Up to a couple of assumptions that are specific to our work, we use the same
assumptions as in (Gorbunov et al., 2023). We start with two new assumptions.

Assumption 2.3 (Bounded ARAgg). We assume that the server applies aggregation rule A such that
A is (δ, c)-ARAgg and there exists constant FA > 0 such that for any inputs x1, . . . , xn ∈Rd the
norm of the aggregator is not greater than the maximal norm of the inputs: ∥A (x1, . . . , xn)∥≤
FA maxi∈[n] ∥xi∥.

The above assumption is satisfied for popular (δ, c)-robust aggregation rules presented in the literature
(Karimireddy et al., 2021, 2022). Therefore, this assumption is more a formality than a real limitation:
it is needed to exclude some pathological examples of (δ, c)-robust aggregation rules, e.g., for any A
that is (δ, c)-RAgg one can construct unbounded (δ, 2c)-RAgg as A = A + X, where X is a random
sample from the Gaussian distribution N(0, cδσ2).

Next, for part of our results, we also make the following assumption.

Assumption 2.4 (Bounded compressor (optional)). We assume that workers use compression operator
Q satisfying Definition 2.2 and bounded as follows: ∥Q(x)∥≤DQ∥x∥
∀x ∈Rd.

For example, RandK and ℓ2-quantization meet this assumption with DQ =
d
K and DQ =
√

d
respectively. In general, constant DQ can be large (proportional to d). However, in practice, one can
use RandK with K =
d
100 and, thus, have moderate DQ = 100. We also have the results without
Assumption 2.4, but with worse dependence on some other parameters, see Section 4.

Next, we assume that good workers have ζ2-heterogeneous local loss functions.

Assumption 2.5 (ζ2-heterogeneity). We assume that good clients have ζ2-heterogeneous local loss
functions for some ζ ≥0, i.e., 1

G
P

i∈G ∥∇fi(x) −∇f(x)∥2 ≤ζ2
∀x ∈Rd.

The above assumption is quite standard for the literature on Byzantine robustness (Wu et al., 2020;
Karimireddy et al., 2022; Gorbunov et al., 2023; Allouah et al., 2023). Moreover, some kind of
a bound on the heterogeneity of good clients is necessary since otherwise Byzantine robustness
cannot be achieved in general. In the appendix, all proofs are given under a more general version of

6This quantity is well-suited for sparsification-type compression operators like random sparsification (Stich
et al., 2018) and 1-level ℓ2-quantization (Alistarh et al., 2017). For other compressors, such as quantization with
more than one level (Goodall, 1951; Roberts, 1962), ζQ is not the main characteristic describing their properties.

4


---Page Break---
Algorithm 1 Byz-VR-MARINA-PP: Byzantine-tolerant VR-MARINA with Partial Participation

1: Input: vectors x0, g0 ∈Rd, stepsize γ, mini-batch size b, probability p ∈(0, 1], number of
iterations K, (δ, c)-ARAgg, clients’ sample size 1 ≤C ≤bC ≤n, clipping coefficients {αk}k≥1
2: for k = 0, 1, . . . , K −1 do
3:
Get a sample from Bernoulli distribution with parameter p: ck ∼Be(p)
4:
Sample the set of clients Sk ⊆[n], |Sk| = C if ck = 0; otherwise |Sk| = bC
5:
Broadcast gk, ck to all workers
6:
for i ∈G ∩Sk in parallel do
7:
xk+1 = xk −γgk and λk+1 = αk+1∥xk+1 −xk∥

8:
Set gk+1
i
=

(
∇fi(xk+1),
if ck = 1,

gk + clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise,

where b∆i(xk+1, xk) is a mini-batched estimator of ∇fi(xk+1) −∇fi(xk), Q(·) for i ∈
G ∩Sk are computed independently
9:
end for

10:
gk+1 =






ARAgg
 
{gk+1
i
}i∈Sk

,
if ck = 1,

gk + ARAgg
n
clipλk+1

Q

b∆i(xk+1, xk)
o

i∈Sk


,
otherwise

11: end for

Assumption 2.5, see Assumption D.5. Finally, the case of homogeneous data (ζ = 0) is also quite
popular for collaborative learning (Diskin et al., 2021b; Kijsipongse et al., 2018a).

The following assumption is classical for the literature on non-convex optimization.

Assumption 2.6 (Smoothness (simplified)). We assume that for all i ∈G and j ∈[m] there exists
L ≥0 such that fi,j is L-smooth, i.e., for all x, y ∈Rd

∥∇fi,j(x) −∇fi,j(y)∥≤L∥x −y∥.
(3)

Moreover, we assume that f is uniformly lower bounded by f ∗∈R, i.e., f ∗:= infx∈Rd f(x).

For the sake of simplicity, we do not differentiate between various notions of smoothness in the
main text. However, our analysis takes into account the differences between smoothness constants,
similarity of local functions, and sampling strategy (see Appendix D.1).

Finally, we also consider functions satisfying Polyak-Łojasiewicz (PŁ) condition (Polyak, 1963;
Łojasiewicz, 1963). This assumption belongs to the class of assumptions on the structured non-
convexity that allows achieving linear convergence (Necoara et al., 2019).

Assumption 2.7 (PŁ condition (optional)). We assume that function f satisfies Polyak-Łojasiewicz
(PŁ) condition with parameter µ > 0, i.e., for all x ∈Rd there exists f ∗:= infx∈Rd f(x) such that
∥∇f(x)∥2 ≥2µ (f(x) −f ∗) .

3
New Method: Byz-VR-MARINA-PP

We propose a new method called Byzantine-tolerant Variance-Reduced MARINA with Partial Partici-
pation (Byz-VR-MARINA-PP, Algorithm 1). Our method extends Byz-VR-MARINA (Gorbunov
et al., 2023) to the partial participation case via the proper usage of the clipping operator. To illustrate
how Byz-VR-MARINA-PP works, we first consider a special case of full participation.

Special case: Byz-VR-MARINA. If all clients participate at each round (Sk ≡[n]) and clipping
is turned off (λk ≡+∞), then Byz-VR-MARINA-PP reduces to Byz-VR-MARINA that works
as follows. Consider the case when no compression is applied (Q(x) = x) and b∆i(xk+1, xk) =
∇fi,jk(xk+1) −∇fi,jk(xk), where jk is sampled uniformly at random from [m], i ∈G. Then,
regular workers compute GeomSARAH/PAGE gradient estimator at each step: for i ∈G

gk+1
i
=
∇fi(xk+1),
with probability p,
gk + ∇fi,jk(xk+1) −∇fi,jk(xk),
otherwise

5


---Page Break---
With small probability p, good workers compute full gradients, and with larger probability 1 −p they
update their estimator via adding stochastic gradient difference. To balance the oracle cost of these
two cases, one can choose p ∼1/m (for b-size mini-batched estimator – p ∼b/m). Such estimators
are known to be optimal for finding stationary points in the stochastic first-order optimization (Fang
et al., 2018; Arjevani et al., 2023). Next, good workers send gk+1
i
or ∇fi,jk(xk+1) −∇fi,jk(xk) to
the server who robustly aggregate the received vectors. Since estimators are conditionally biased,
i.e., E[gk+1
i
| xk+1, xk] ̸= ∇fi(xk+1), the additional bias coming from the aggregation does not
cause significant issues in the analysis or practice. Moreover, the variance of {gk+1
i
}i∈G w.r.t. the
sampling of the stochastic gradients is proportional to ∥xk+1 −xk∥2 →0 with probability 1 −p
(due to Assumption D.3) that progressively limits the effect of Byzantine attacks. For a more detailed
explanation of why recursive variance reduction works better than SAGA/SVRG-type variance
reduction, we refer to (Gorbunov et al., 2023). Arbitrary sampling allows the improvement of the
dependence on the smoothness constants. Unbiased communication compression also naturally fits
the framework since it is applied to the stochastic gradient difference, meaning that the variance of
{gk+1
i
}i∈G w.r.t. the sampling of the stochastic gradients and compression remains proportional to
∥xk+1 −xk∥2 with probability 1 −p.

New ingredients: client sampling and clipping. The algorithmic novelty of Byz-VR-MARINA-PP
in comparison to Byz-VR-MARINA is twofold: with (typically large) probability 1 −p only C
clients sampled uniformly at random from the set of all clients participate at each round, and clipping
is applied to the compressed stochastic gradient differences. With a small probability p, a larger
number7 of clients bC ≤n takes part in the communication. The main role of clipping is to ensure
that the method can withstand the attacks of Byzantines when they form a majority or, more precisely
when there are more than δC Byzantine workers among the sampled ones. Indeed, without clipping
(or some other algorithmic changes) such situations are critical for convergence: Byzantine workers
can shift the method arbitrarily far from the solution, e.g., they can collectively send some vector
with the arbitrarily large norm. In contrast, Byz-VR-MARINA-PP tolerates any attacks even when
all sampled clients are Byzantine workers since the update remains bounded due to the clipping. Via
choosing λk+1 ∼∥xk+1 −xk∥we ensure that the norm of transmitted vectors decreases with the
same rate as it does in Byz-VR-MARINA with full client participation. Finally, with probability 1 −p
regular workers can transmit just compressed vectors and leave the clipping operation to the server
since Byzantines can ignore clipping operation.

4
Convergence Results

We define Gk
C = G ∩Sk and Gk
C = |Gk
C| and
 n
k

=
n!
k!(n−k)! represents the binomial coefficient. We
also use the following probabilities:

pG := Prob

Gk
C ≥(1 −δ) C
	
= P

⌈(1−δ)C⌉≤t≤C
(
G
t)(
n−G
C−t)
(
n
C)
,

PGk
C := Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
=
C
npG
· P

⌈(1−δ)C⌉≤t≤C
(
G−1
t−1)(
n−G
C−t)
(
n−1
C−1)
.

These probabilities naturally appear in the analysis and statements of the theorems. When ck = 0,
then server samples C clients, and two situations can appear: either Gk
C is at least (1 −δ) C meaning
that the aggregator can ensure robustness according to Definition 2.1 or Gk
C < (1 −δ) C. Probability
pG is the probability of the first event, and the second event implies that the aggregation can be spoiled
by Byzantine workers (but clipping bounds the “harm”). Finally, we use PGk
C in the computation of
some conditional expectations when the first event occurs. The mentioned probabilities can be easily
computed for some special cases. For example, if C = 1, then pG = G/n and PGk
C = 1/G; if C = 2,
then pG = G(G−1)/n(n−1) and PGk
C = 2/G; finally, if C = n, then pG = 1 and PGk
C = 1.

The next theorem is our main convergence result for general unbiased compression operators.

7As one can see from our analysis, it is sufficient to take bC ≥max{1, δrealn/δ} similarly to (Data and Diggavi,
2021). However, in contrast to the approach from Data and Diggavi (2021), Byz-VR-MARINA-PP requires
such communications only with small probability p.

6


---Page Break---
Theorem 4.1. Let Assumptions 2.3, 2.5, 2.6 hold, λk+1
=
2L
xk+1 −xk, and bC
≥
max{1, δrealn/δ}. Assume that 0 < γ ≤1/L(1+
√

A), where constant A is defined as

A
=
32pGGPGk
C
p2(1 −δ)C (30ω + 11) (1 + 2cδ) + 16(1 −pG)(1 + 4F 2
A)
p2
.
(4)

Then for all K ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

E
h∇f
 
bxK2i
≤
2Φ0
γ(K+1) + 4 b
Dζ2

p
,
(5)

where bD =
2δPGk
b
C
1−δ

6cG

b
C + p

+ eD, where eD = 0 when bC = n, eD =
PGk
b
C
G

(1−δ) b
C when bC = n, and bxK

is chosen uniformly at random from x0, x1, . . . , xK, and Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.
If, in addition, Assumption 2.7 holds and 0 < γ ≤1/L(1+
√

2A), then for all K ≥0 the iterates
produced by Byz-VR-MARINA-PP (Algorithm 1) with ρ = min

γµ, p

8
	
satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 4 b
Dζ2γ

pρ
,
(6)

where Φ0 = f
 
x0
−f ∗+ 4γ

p
g0 −∇f
 
x02.

The above theorem establishes similar guarantees to the current SOTA ones obtained for Byz-VR-
MARINA. That is, in the general non-convex case, we prove O(1/K) rate, which is optimal (Arjevani
et al., 2023), and for PŁ-functions we derive linear convergence result to the neighborhood depending
on the heterogeneity. The size of this neighborhood matches the one derived for Byz-VR-MARINA
by Gorbunov et al. (2023). However, since our result is obtained considering the challenging scenario
of partial participation of clients, the maximal theoretically allowed stepsize in our analysis of
Byz-VR-MARINA-PP is smaller than the one from (Gorbunov et al., 2023).

In particular, the second term in the constant A appears due to the partial participation, and the
whole expression for A is proportional to 1/p2. In contrast, a similar constant A from the result for
Byz-VR-MARINA is proportional to 1/p, which can be noticeably smaller than 1/p2. Indeed, to make
the expected number of clients participating in the communication round equal to O(C), to make
the expected number of stochastic oracle calls equal to O(b), and to make the expected number
of transmitted components for each worker taking part in the communication round equal O(ζQ),
parameter p should be chosen as p = min{C/n, b/m, ζQ/d}, where the latter term in the minimum
often equals to Θ(1/(ω+1)) (Gorbunov et al., 2021). Therefore, in some scenarios, p can be small.

Next, in the special case of full participation, we have C = bC = n, pG = PGk
C = 1, meaning
that A = Θ((1+ω)(1+cδ)/p2) for Byz-VR-MARINA-PP. In contrast, the corresponding constant for
Byz-VR-MARINA is of the order Θ((1+ω)/pn + (1+ω)cδ/p2), which is strictly better than our bound.
In this special case, we do not recover the result for Byz-VR-MARINA.

Such a complexity deterioration can be explained as follows: the presence of clipping introduces
additional technical difficulties in the analysis, resulting in a reduced step size compared to Byz-VR-
MARINA, even when C = bC = n. To achieve a more favorable convergence rate, particularly in
scenarios of complete participation, we also establish the results under Assumption 2.4.

Theorem 4.2. Let Assumptions 2.3, 2.4, 2.5, 2.6 hold, λk+1 = DQL
xk+1 −xk, and bC ≥
max{1, δrealn/δ}. Assume that 0 < γ ≤1/L(1+
√

A), where constant A equals

A
=
4pGGPGk
C
p(1 −δ)C

 3ω + 2

(1 −δ)C + 8(5ω + 4)cδ

p


+
8(1 −pG)(2 + F 2
AD2
Q)
p2
.
(7)

Then for all K ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

E
h∇f
 
bxK2i
≤
2Φ0
γ(K+1) + 2 b
Dζ2

p
,
(8)

where bD =
2δPGk
b
C
1−δ

6cG

b
C + p

+ eD, where eD = 0 when bC = n, eD =
PGk
b
C
G

(1−δ) b
C when bC = n, and bxK

is chosen uniformly at random from x0, x1, . . . , xK, and Φ0 = f
 
x0
−f ∗+ γ

p
g0 −∇f
 
x02.

7


---Page Break---
If, in addition, Assumption 2.7 holds and 0 < γ ≤1/L(1+
√

2A), then for all K ≥0 the iterates
produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy with ρ = min

γµ, p

4
	

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 2 b
Dζ2γ

pρ
,
(9)

where Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.

With Assumption 2.4, vectors {Q(b∆i(xk+1, xk))}i∈Gk
C can be upper bounded by DQL
xk+1 −xk.
Using this fact, one can take the clipping level sufficiently large such that it is turned off for the
regular workers. This allows us to simplify the proof and remove 1/p factor in front of the terms not
proportional to δ or to 1 −pG in the expression for A that can make the stepsize larger. However,
the second term in (7) can be larger than (4), since it depends on potentially large constant DQ.
Therefore, the rates of convergence from Theorems 4.1 and 4.2 cannot be compared directly. We also
highlight that the clipping level from Theorem 4.2 is in general larger than the clipping level from
Theorem 4.1 and, thus, it is expected that with full participation Theorem 4.2 gives better results
than Theorem 4.1: the bias introduced due to the clipping becomes smaller with the increase of the
clipping level. However, in the partial participation regime, the price for this is a decrease of the
stepsize to compensate for the increased harm from Byzantine clients in situations when they form a
majority. Further discussion of the technical challenges we overcame is deferred to Appendix E.3.

Nevertheless, in the case of full participation, we have C = bC = n, pG = PGk
C = PGk
b
C = 1,
meaning that A = Θ((1+ω)/pn + (1+ω)cδ/p2) in Theorem 4.2. That is, in this case, we recover the
result of Byz-VR-MARINA. More generally, if pG = 1, which is equivalent to C ≥max{1, δrealn/δ},
then PGk
C = Prob{i ∈Gk
C} = min{1, C/G}, PGk
b
C = Prob{i ∈Gk
b
C} = min{1, b
C/G} and we
have A = Θ((1+ω)/pC + (1+ω)cδ/p2). Here, the first term in A is n/C worse than the corresponding
term for Byz-VR-MARINA. However, the second term in A matches the corresponding term for
Byz-VR-MARINA. Moreover, this term is the main one if cδ ≥p/C, which is typically the case since
parameter p is often small (p = min{C/b
C, b/m, ζQ/d}). In such cases, Byz-VR-MARINA-PP has the
same rate of convergence as Byz-VR-MARINA while utilizing, on average, just O(C) workers at
each step in contrast to Byz-VR-MARINA that uses n workers at each step. That is, in some cases,
partial participation is provably beneficial for Byz-VR-MARINA-PP.

Byz-VR-MARINA+: simplified version of Byz-VR-MARINA.
In Appendix F, we propose a
simplified version of Byz-VR-MARINA called Byz-VR-MARINA+ (Algorithm 3). The only dif-
ference is related to Line 10 of the method: when ck = 0, Byz-VR-MARINA+ computes just the
average of
n
clipλk+1

Q

b∆i(xk+1, xk)
o

i∈Sk
instead of robust aggregation, while keeps using

ARAgg when ck = 1. Of course, when ck = 0 and at least one Byzantine worker is sampled, then
the step can be useless, but the “harm” of this step is bounded due to the clipping. However, in
certain regimes (e.g., when C is small enough and the number of Byzantine workers is much smaller
than the number of regular workers), the probability of sampling only regular workers is larger than
sampling at least one Byzantine worker when ck = 0, meaning that with high enough probability the
resulting estimator has no additional bias coming from the robust aggregation. We formally analyze
Byz-VR-MARINA+ and show that such a modification of the method leads to better theoretical
results (especially when C is small). In particular, in the settings of Theorem 4.2, we prove that
Byz-VR-MARINA+ exhibits the same O(1/K) rate but converges to O(1/p) smaller neighborhood
when bC = n, i.e., the neighborhood term for Byz-VR-MARINA+ is optimal (Karimireddy et al.,
2022; Allouah et al., 2024b). Moreover, our results for Byz-VR-MARINA+ allow larger stepsizes
when C is small enough. For further details and complete proofs, we refer to Appendix F.

Extensions without full-batch gradient computations.
The proposed methods – Byz-VR-
MARINA and Byz-VR-MARINA+ – have a common limitation related to the full-batch gradient
computation with probability p. Although this probability is typically small, even one full-gradient
computation can be very expensive for certain problems. To address this issue, we propose the modifi-
cations of Byz-VR-MARINA and Byz-VR-MARINA+ without full-batch gradient computations at all
(see Algorithms 4 and 5 in Appendix G). That is, these modifications differ from Byz-VR-MARINA
and Byz-VR-MARINA+ in Line 8 only: when ck = 1, every good worker i from Sk computes
and sends to the server b′-size mini-batched stochastic gradient estimator e∇fi(xk+1) of ∇fi(xk+1).

8


---Page Break---
0.0
0.5
1.0
1.5
2.0
epochs

10−6

10−4

10−2

f(x) −f *

CM | SHB

Byz-VR-MARINA-PP

Byz-VR-MARINA

0
2
4
6
8
10
epochs

10−6

10−4

10−2

f(x) −f *

CM | SHB

Partial (20% clients)
Full

0.0
0.5
1.0
1.5
2.0
epochs

10−6

10−4

10−2

f(x) −f *

CM | SHB

clip mult. = 0.01
clip mult. = 0.1
clip mult. = 1.0
clip mult. = 10.0
clip mult. = None

Figure 1: The optimality gap f(xk) −f(x∗) for 3 different scenarios. We use coordinate-wise mean
with bucketing equal to 2 as an aggregation and shift-back as an attack. We use the a9a dataset,
where each worker accesses the full dataset with 15 good and 5 Byzantine workers. We do not use
any compression. In each step, we sample 20% of clients uniformly at random to participate in the
given round unless we specifically mention that we use full participation. Left: Linear convergence
of Byz-VR-MARINA-PP with clipping versus non-convergence without clipping. Middle: Full
versus partial participation, showing faster convergence with clipping. Right: Clipping multiplier λ
sensitivity, demonstrating consistent linear convergence across varying λ values.

Under the additional assumption that the variance of e∇fi(xk+1) is uniformly bounded by σ2/b′, which
is a standard assumption for variance-reduced methods without full-batch gradient computations
(Fang et al., 2018; Cutkosky and Orabona, 2019; Li et al., 2021; Gorbunov et al., 2021), we prove that
both methods converge similarly as in the case of the (periodical) full-batch gradient computations
but to the neighborhood having an additional term proportional to

cδ + PGk
b
C
G/b
C2

σ2/b′. For further
details and complete proofs, we refer to Appendix G.

Heuristic extension of Byz-VR-MARINA-PP.
In this short remark, we illustrate how the proposed
clipping technique can be applied to a general class of Byzantine-robust methods to adapt them to
the case of partial participation. Consider the methods having the following update rule: xk+1 =
xk −γ · Agg({gk
i }i∈[n]), where {gk
i }i∈[n] are the vectors received from workers at iteration k and
Agg is some aggregation rule. A vast majority of existing Byzantine-robust methods fit this scheme.
In the case of partial participation of clients, we propose to modify the scheme as follows:

xk+1 = xk −γgk,
where gk := gk−1 + Agg

clipλk(gk
i −gk−1)
	

i∈Sk


,
(10)

where Sk ⊆[n] is a subset of clients participating in round k and {λk}k≥0 is sequence of clipping
parameters specified by the server. In particular, Byz-VR-MARINA-PP can be seen as an application
of scheme (10) to Byz-VR-MARINA (up to a minor modification when ck = 1 in Byz-VR-MARINA)
with λk+1 = λ∥xk+1 −xk∥. We suggest to use λk+1 = λ∥xk+1 −xk∥with tunable parameter
λ > 0 for other methods as well.

5
Numerical Experiments

Firstly, we showcase the benefits of employing clipping to remedy the presence of Byzantine workers
and partial participation. For this task, we consider the standard logistic regression model with
ℓ2-regularization, i.e., fi,j(x) = −yi,j log(h(x, ai,j)) −(1 −yi,j) log(1 −h(x, ai,j)) + η∥x∥2,
where yi,j ∈{0, 1} is the label, ai,j ∈Rd represents the feature vector, η is the regularization
parameter, and h(x, a) = 1/(1+e−a⊤x). This objective is smooth, and for η > 0, it is also strongly
convex, satisfying the PŁ-condition. We consider the a9a LIBSVM dataset (Chang and Lin, 2011)
and set η = 0.01. In the experiments, we focus on an important feature of Byz-VR-MARINA-PP: it
has linear convergence for homogeneous datasets across clients even in the presence of Byzantine
workers and partial participation, as shown in Theorems 4.1 and 4.2.

To demonstrate this experimentally, we consider the setup with 15 good workers and 5 Byzantines,
each worker can access the entire dataset, and the server uses coordinate-wise median with bucketing
as the aggregator (see also Appendix C). For the attack, we propose a new attack that we refer to as
the shift-back attack, which acts in the following way. If Byzantine workers are in the majority in the
current round k, then each Byzantine worker sends x0 −xk. Otherwise, they follow protocol and act
as benign workers. Further experimental details are deferred to Appendix H.

9


---Page Break---
0
1
2
3
4
epochs

10−1

100

f(x)

CM | BF

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

CM | SHB

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

RFA | BF

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

RFA | SHB

W/ Clip
W/O Clip

Figure 2: Training loss of 2 aggregation rules (CM, RFA) under 2 attacks (BF, SHB) on the MNIST
dataset under heterogeneous data split with 20 clients, 5 of which are malicious. Additional experi-
ments on CIFAR10 are provided in Appendix H.

We compare our Byz-VR-MARINA-PP with its version without clipping. We note that the setup
that we consider is the most favorable in terms of minimized variance in terms of data and gradient
heterogeneity. We show that even in this simplest setup, the method without clipping does not
converge since there is no method that can withstand the Byzantine majority. Therefore, any more
complex scenario would also fall short using our simple attack. On the other hand, we show that
once clipping is applied, Byz-VR-MARINA-PP is able to converge linearly to the exact solution,
complementing our theoretical results.

Figure 1 showcases these observations. On the left, we can see Byz-VR-MARINA-PP converges
linearly to the optimal solution, while the version without clipping remains stuck at the starting point
since Byzantines are always able to push the solution back to the origin since they can create the
majority in some rounds. In the middle plot, we compare the full participation scenario in which
all the clients participate in each round, which does not require clipping since, in each step, we are
guaranteed that Byzantines are not in the majority, to partial participation with clipping. We can see,
when we compare the total number of computations (measured in epochs), Byz-VR-MARINA-PP
leads to faster convergence even though we need to employ clipping. Finally, in the right plot, we
measure the sensitivity of clipping multiplier λ. We can see that Byz-VR-MARINA-PP is not very
sensitive to λ in terms of convergence, i.e., for all the values of λ, we still converge linearly. However,
the suboptimal choice of λ leads to slower convergence.

Furthermore, we also realize that other attacks and more complicated experiments could potentially
damage clipping more than methods not using clipping. Therefore, we provide additional experiments
with neural networks and different attacks in heterogeneous settings. For our experimental setup,
we follow (Karimireddy et al., 2021). However, when working with neural networks, the choice of
standard variance reduction is not effective (Defazio and Bottou, 2019). Therefore, we use Byzantine
Robust Momentum SGD (Karimireddy et al., 2021) as an underlying optimization method; see (10).

We consider the MNIST dataset (LeCun and Cortes, 1998) with heterogeneous splits with 20 clients,
5 of which are malicious. For the attacks, we consider A Little is Enough (ALIE) (Baruch et al.,
2019), Bit Flipping (BF), and aforementioned Shift-Back (SHB). For the aggregations, we consider
coordinate median (CM) (Chen et al., 2017) and robust federated averaging (RFA) (Pillutla et al.,
2022) with bucketing.

From Figure 2, we can see that clipping does not lead to performance degradation. On the contrary,
clipping performs on par or better than its variant without clipping. Furthermore, we can see that no
robust aggregator is able to withstand the shift-back attack without clipping.

6
Conclusion and Future Work

This work makes an important step in the direction of achieving Byzantine robustness under the
partial participation of clients. However, some important questions remain open. First of all, it will
be interesting to understand whether the derived bounds can be further improved in terms of the
dependence on ω, m, and C. Next, it would be interesting to rigorously prove that our heuristic works
for SGD with client momentum (Karimireddy et al., 2021, 2022) and other Byzantine-robust methods.
Finally, studying other participation patterns (non-uniform sampling/arbitrary client participation) is
also a very prominent direction for future research.

10


---Page Break---
Acknowledgements

The work of G. Malinovsky and P. Richt´arik was supported by funding from King Abdullah University
of Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) Center of Excellence
for Generative AI, under award number 5940, iii) SDAIA-KAUST Center of Excellence in Artificial
Intelligence and Data Science.

References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. (2016).
Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on
Computer and Communications Security, pages 308–318. (Cited on page 20)

Agarwal, A. and Duchi, J. C. (2011). Distributed delayed stochastic optimization. Advances in neural
information processing systems, 24. (Cited on page 20)
Ajalloeian, A. and Stich, S. U. (2020). On the convergence of SGD with biased gradients. arXiv
preprint arXiv:2008.00051. (Cited on page 20)

Alistarh, D., Allen-Zhu, Z., and Li, J. (2018). Byzantine stochastic gradient descent. In Advances in
Neural Information Processing Systems, pages 4618–4628. (Cited on pages 2 and 19)
Alistarh, D., Grubic, D., Li, J., Tomioka, R., and Vojnovic, M. (2017). QSGD: Communication-
efficient SGD via gradient quantization and encoding. In Advances in Neural Information Process-
ing Systems, volume 30. (Cited on pages 4 and 20)
Allen-Zhu, Z., Ebrahimian, F., Li, J., and Alistarh, D. (2021). Byzantine-resilient non-convex
stochastic gradient descent. In International Conference on Learning Representations. (Cited on
pages 2 and 19)

Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R., Rizk, G., and Voitovych, S. (2024a).
Tackling byzantine clients in federated learning. arXiv preprint arXiv:2402.12780. (Cited on page 3)

Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R., and Stephan, J. (2023). Fixing by
mixing: A recipe for optimal byzantine ml under heterogeneity. In International Conference on
Artificial Intelligence and Statistics, pages 1232–1300. (Cited on pages 2, 4, and 19)

Allouah, Y., Guerraoui, R., Gupta, N., Pinot, R., and Rizk, G. (2024b). Robust distributed learning:
tight error bounds and breakdown point under data heterogeneity. Advances in Neural Information
Processing Systems, 36. (Cited on pages 8 and 63)

Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. (2023). Lower
bounds for non-convex stochastic optimization. Mathematical Programming, 199(1-2):165–214.
(Cited on pages 6 and 7)
Atre, M., Jha, B., and Rao, A. (2021). Distributed deep learning using volunteer computing-like
paradigm. arXiv preprint arXiv:2103.08894. (Cited on page 1)
Baruch, G., Baruch, M., and Goldberg, Y. (2019). A little is enough: Circumventing defenses for
distributed learning. In Advances in Neural Information Processing Systems, volume 32. (Cited on
pages 2, 10, and 74)

Basu, D., Data, D., Karakus, C., and Diggavi, S. (2019). Qsparse-local-SGD: Distributed SGD with
quantization, sparsification and local computations. In Advances in Neural Information Processing
Systems, volume 32. (Cited on page 20)
Bernstein, J., Zhao, J., Azizzadenesheli, K., and Anandkumar, A. (2019). signSGD with majority
vote is communication efficient and fault tolerant. In International Conference on Learning
Representations. (Cited on page 20)
Beznosikov, A., Horv´ath, S., Richt´arik, P., and Safaryan, M. (2020). On biased compression for
distributed learning. arXiv preprint arXiv:2002.12410. (Cited on pages 4 and 20)
Blanchard, P., El Mhamdi, E. M., Guerraoui, R., and Stainer, J. (2017). Machine learning with
adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing
Systems, volume 30. (Cited on pages 2 and 4)

Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal,
A., and Seth, K. (2017). Practical secure aggregation for privacy-preserving machine learning. In

11


---Page Break---
Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security,
pages 1175–1191. (Cited on page 2)

Boubouh, K., Boussetta, A., Gupta, N., Maurer, A., and Pinot, R. (2022). Democratizing machine
learning: Resilient distributed learning with heterogeneous participants. In 2022 41st International
Symposium on Reliable Distributed Systems (SRDS), pages 94–120. IEEE. (Cited on page 3)

Chang, C.-C. and Lin, C.-J. (2011). Libsvm: a library for support vector machines. ACM transactions
on intelligent systems and technology (TIST), 2(3):1–27. (Cited on page 9)

Chen, L., Wang, H., Charles, Z., and Papailiopoulos, D. (2018). Draco: Byzantine-resilient distributed
training via redundant gradients. In International Conference on Machine Learning, pages 903–912.
(Cited on page 19)

Chen, X., Wu, S. Z., and Hong, M. (2020). Understanding gradient clipping in private SGD: A
geometric perspective. In Advances in Neural Information Processing Systems, volume 33, pages
13773–13782. (Cited on page 20)

Chen, Y., Su, L., and Xu, J. (2017). Distributed statistical machine learning in adversarial settings:
Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing
Systems, 1(2):1–25. (Cited on pages 10 and 74)

Cutkosky, A. and Orabona, F. (2019). Momentum-based variance reduction in non-convex SGD. In
Advances in Neural Information Processing Systems, volume 32. (Cited on pages 9, 19, and 66)

Damaskinos, G., El-Mhamdi, E.-M., Guerraoui, R., Guirguis, A., and Rouault, S. (2019). Aggregathor:
Byzantine machine learning via robust gradient aggregation. Proceedings of Machine Learning
and Systems, 1:81–106. (Cited on page 2)

Damaskinos, G., Guerraoui, R., Patra, R., Taziki, M., et al. (2018). Asynchronous byzantine machine
learning (the case of sgd). In International Conference on Machine Learning, pages 1145–1154.
PMLR. (Cited on page 20)

Data, D. and Diggavi, S. (2021). Byzantine-resilient high-dimensional SGD with local iterations on
heterogeneous data. In International Conference on Machine Learning, pages 2478–2488. PMLR.
(Cited on pages 3, 6, 19, and 20)

Defazio, A., Bach, F., and Lacoste-Julien, S. (2014). SAGA: A fast incremental gradient method
with support for non-strongly convex composite objectives. In Advances in Neural Information
Processing Systems, volume 27. (Cited on pages 2 and 19)

Defazio, A. and Bottou, L. (2019). On the ineffectiveness of variance reduced optimization for deep
learning. Advances in Neural Information Processing Systems, 32. (Cited on page 10)

Demidovich, Y., Malinovsky, G., Sokolov, I., and Richt´arik, P. (2023). A guide through the Zoo of
biased SGD. In Advances in Neural Information Processing Systems, volume 36. (Cited on page 20)

Diskin, M., Bukhtiyarov, A., Ryabinin, M., Saulnier, L., Lhoest, Q., Sinitsin, A., Popov, D., Pyrkin,
D., Kashirin, M., Borzunov, A., del Moral, A. V., Mazur, D., Kobelev, I., Jernite, Y., Wolf, T., and
Pekhimenko, G. (2021a). Distributed deep learning in open collaborations. In Advances in Neural
Information Processing Systems, volume 34, pages 7879–7897. (Cited on page 1)

Diskin, M., Bukhtiyarov, A., Ryabinin, M., Saulnier, L., Sinitsin, A., Popov, D., Pyrkin, D. V.,
Kashirin, M., Borzunov, A., Villanova del Moral, A., et al. (2021b). Distributed deep learning in
open collaborations. In Advances in Neural Information Processing Systems, volume 34, pages
7879–7897. (Cited on page 5)

El-Mhamdi, E. M., Farhadkhani, S., Guerraoui, R., Guirguis, A., Hoang, L.-N., and Rouault, S. (2021).
Collaborative learning in the jungle (decentralized, byzantine, heterogeneous, asynchronous and
nonconvex learning). Advances in neural information processing systems, 34:25044–25057. (Cited
on page 3)

Fang, C., Li, C. J., Lin, Z., and Zhang, T. (2018). Spider: Near-optimal non-convex optimization via
stochastic path-integrated differential estimator. In Advances in Neural Information Processing
Systems, volume 31. (Cited on pages 6, 9, and 71)

Fang, M., Liu, J., Gong, N. Z., and Bentley, E. S. (2022). AFLGuard: Byzantine-robust asynchronous
federated learning. In Proceedings of the 38th Annual Computer Security Applications Conference,
pages 632–646. (Cited on page 20)

12


---Page Break---
Fatkhullin, I., Sokolov, I., Gorbunov, E., Li, Z., and Richt´arik, P. (2021). EF21 with bells & whistles:
Practical algorithmic extensions of modern error feedback. arXiv preprint arXiv:2110.03294.
(Cited on page 20)
Ghadimi, S. and Lan, G. (2013). Stochastic first-and zeroth-order methods for nonconvex stochastic
programming. SIAM journal on optimization, 23(4):2341–2368. (Cited on page 66)
Ghosh, A., Maity, R. K., Kadhe, S., Mazumdar, A., and Ramchandran, K. (2021). Communication-
efficient and byzantine-robust distributed learning with error feedback. IEEE Journal on Selected
Areas in Information Theory, 2(3):942–953. (Cited on page 20)
Ghosh, A., Maity, R. K., and Mazumdar, A. (2020). Distributed newton can communicate less and
resist Byzantine workers. In Advances in Neural Information Processing Systems, volume 33,
pages 18028–18038. (Cited on page 20)

Goodall, W. (1951). Television by pulse code modulation. Bell System Technical Journal, 30(1):33–49.
(Cited on page 4)
Gorbunov, E., Borzunov, A., Diskin, M., and Ryabinin, M. (2022). Secure distributed training at
scale. In International Conference on Machine Learning. (arXiv preprint arXiv:2106.11257, 2021).
(Cited on pages 2 and 19)
Gorbunov, E., Burlachenko, K. P., Li, Z., and Richt´arik, P. (2021). MARINA: Faster non-convex
distributed learning with compression. In International Conference on Machine Learning, pages
3788–3798. (Cited on pages 7, 9, 20, 66, and 71)
Gorbunov, E., Danilova, M., and Gasnikov, A. (2020). Stochastic optimization with heavy-tailed
noise via accelerated gradient clipping. In Advances in Neural Information Processing Systems,
volume 33, pages 15042–15053. (Cited on pages 20, 26, and 57)
Gorbunov, E., Horv´ath, S., Richt´arik, P., and Gidel, G. (2023). Variance reduction is an antidote to
Byzantines: Better rates, weaker assumptions and communication compression as a cherry on the
top. International Conference on Learning Representations. (Cited on pages 2, 3, 4, 5, 6, 7, 19, 20, 25,
26, and 74)

Gower, R. M., Schmidt, M., Bach, F., and Richt´arik, P. (2020). Variance-reduced methods for machine
learning. Proceedings of the IEEE, 108(11):1968–1983. (Cited on page 19)
Haddadpour, F., Kamani, M. M., Mokhtari, A., and Mahdavi, M. (2021). Federated learning with
compression: Unified analysis and sharp guarantees. In International Conference on Artificial
Intelligence and Statistics, pages 2350–2358. (Cited on page 20)
He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778.
(Cited on page 74)
He, L., Karimireddy, S. P., and Jaggi, M. (2022). Byzantine-robust decentralized learning via
ClippedGossip. arXiv preprint arXiv:2202.01545. (Cited on page 19)
Horv´ath, S., Kovalev, D., Mishchenko, K., Stich, S., and Richt´arik, P. (2023). Stochastic distributed
learning with gradient quantization and variance reduction. Optimization Methods and Software,
38(1). (arXiv preprint arXiv:1904.05115, 2019). (Cited on pages 2, 4, 19, and 20)
Islamov, R., Qian, X., and Richt´arik, P. (2021). Distributed second order methods with fast rates and
compressed communication. In International Conference on Machine Learning, pages 4617–4628.
(Cited on page 20)

Johnson, R. and Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance
reduction. In Advances in Neural Information Processing Systems, volume 26. (Cited on page 19)
Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K.,
Charles, Z., Cormode, G., Cummings, R., et al. (2021). Advances and open problems in federated
learning. Foundations and Trends® in Machine Learning, 14(1–2):1–210. (Cited on page 1)
Karimireddy, S. P., He, L., and Jaggi, M. (2021). Learning from history for Byzantine robust
optimization. In International Conference on Machine Learning, pages 5311–5319. (Cited on
pages 2, 3, 4, 10, 19, 20, and 74)

Karimireddy, S. P., He, L., and Jaggi, M. (2022). Byzantine-robust learning on heterogeneous datasets
via bucketing. International Conference on Learning Representations. (Cited on pages 2, 3, 4, 8, 10,
19, 23, and 26)

13


---Page Break---
Khirirat, S., Feyzmahdavian, H. R., and Johansson, M. (2018). Distributed learning with compressed
gradients. arXiv preprint arXiv:1806.06573. (Cited on pages 4 and 20)
Kijsipongse, E., Piyatumrong, A., et al. (2018a). A hybrid gpu cluster and volunteer computing
platform for scalable deep learning. The Journal of Supercomputing, 74(7):3236–3263. (Cited on
page 5)
Kijsipongse, E., Piyatumrong, A., and U-ruekolan, S. (2018b). A hybrid gpu cluster and volunteer
computing platform for scalable deep learning. The Journal of Supercomputing. (Cited on page 1)

Koneˇcn´y, J., McMahan, H. B., Yu, F., Richt´arik, P., Suresh, A. T., and Bacon, D. (2016). Federated
learning: strategies for improving communication efficiency. In NIPS Private Multi-Party Machine
Learning Workshop. (Cited on page 1)
Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images.
(Cited on page 74)

Lamport, L., Shostak, R., and Pease, M. (1982). The Byzantine generals problem. ACM Transactions
on Programming Languages and Systems, 4(3):382–401. (Cited on page 1)
LeCun,
Y.
and
Cortes,
C.
(1998).
The
mnist
database
of
handwritten
digits.
http://yann.lecun.com/exdb/mnist/. (Cited on pages 10 and 74)

Li, C. (2020). Demystifying gpt-3 language model: A technical overview. ”https://lambdalabs.
com/blog/demystifying-gpt-3”. (Cited on page 1)
Li, Z., Bao, H., Zhang, X., and Richt´arik, P. (2021). PAGE: A simple and optimal probabilistic
gradient estimator for nonconvex optimization. In International Conference on Machine Learning,
pages 6286–6295. (Cited on pages 2, 9, 19, 26, 66, and 71)

Li, Z., Kovalev, D., Qian, X., and Richt´arik, P. (2020). Acceleration for compressed gradient descent
in distributed and federated optimization. In International Conference on Machine Learning, pages
5895–5904. (Cited on page 20)

Łojasiewicz, S. (1963). A topological property of real analytic subsets. Coll. du CNRS, Les ´equations
aux d´eriv´ees partielles, 117:87–89. (Cited on page 5)

Lyu, L., Yu, H., Ma, X., Sun, L., Zhao, J., Yang, Q., and Yu, P. S. (2020). Privacy and robustness in
federated learning: Attacks and defenses. arXiv preprint arXiv:2012.06337. (Cited on page 19)
Mai, V. V. and Johansson, M. (2021). Stability and convergence of stochastic gradient clipping:
Beyond Lipschitz continuity and smoothness. In International Conference on Machine Learning,
pages 7325–7335. (Cited on page 20)
Mishchenko, K., Gorbunov, E., Tak´aˇc, M., and Richt´arik, P. (2019). Distributed learning with
compressed gradient differences. arXiv preprint arXiv:1901.09269. (Cited on page 20)
Nazin, A. V., Nemirovsky, A. S., Tsybakov, A. B., and Juditsky, A. B. (2019). Algorithms of
robust stochastic optimization based on mirror descent method. Automation and Remote Control,
80:1607–1627. (Cited on page 20)
Necoara, I., Nesterov, Y., and Glineur, F. (2019). Linear convergence of first order methods for
non-strongly convex optimization. Mathematical Programming, 175:69–107. (Cited on page 5)

Nedi´c, A., Bertsekas, D. P., and Borkar, V. S. (2001). Distributed asynchronous incremental subgradi-
ent methods. Studies in Computational Mathematics, 8(C):381–407. (Cited on page 20)

Nemirovski, A. S., Juditsky, A. B., Lan, G., and Shapiro, A. (2009). Robust stochastic approximation
approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574–1609. (Cited on
page 66)
Nguyen, L. M., Liu, J., Scheinberg, K., and Tak´aˇc, M. (2017). Sarah: A novel method for machine
learning problems using stochastic recursive gradient. In International Conference on Machine
Learning, pages 2613–2621. (Cited on page 19)
Nguyen, T. D., Ene, A., and Nguyen, H. L. (2023). Improved convergence in high probability of
clipped gradient methods with heavy tails. arXiv preprint arXiv:2304.01119. (Cited on page 20)

Niu, C., Wu, F., Tang, S., Hua, L., Jia, R., Lv, C., Wu, Z., and Chen, G. (2020). Billion-scale federated
learning on mobile clients: A submodel design with tunable privacy. In Proceedings of the 26th
Annual International Conference on Mobile Computing and Networking, pages 1–14. (Cited on
page 2)

14


---Page Break---
Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the difficulty of training recurrent neural
networks. In International Conference on Machine Learning, pages 1310–1318. (Cited on page 20)
Pillutla, K., Kakade, S. M., and Harchaoui, Z. (2022). Robust aggregation for federated learning.
IEEE Transactions on Signal Processing, 70:1142–1154. (Cited on pages 2, 10, and 74)
Polyak, B. T. (1963). Gradient methods for the minimisation of functionals. USSR Computational
Mathematics and Mathematical Physics, 3(4):864–878. (Cited on page 5)
Qian, X., Richt´arik, P., and Zhang, T. (2021). Error compensated distributed SGD can be accelerated.
In Advances in Neural Information Processing Systems, volume 34. (Cited on page 20)
Rajput, S., Wang, H., Charles, Z., and Papailiopoulos, D. (2019). Detox: A redundancy-based
framework for faster and more robust gradient aggregation. In Advances in Neural Information
Processing Systems, volume 32. (Cited on page 19)
Regatti, J., Chen, H., and Gupta, A. (2020). ByGARS: Byzantine SGD with arbitrary number of
attackers. arXiv preprint arXiv:2006.13421. (Cited on page 19)
Richt´arik, P., Sokolov, I., and Fatkhullin, I. (2021). EF21: A new, simpler, theoretically better,
and practically faster error feedback. In Advances in Neural Information Processing Systems,
volume 34. (Cited on pages 20 and 22)
Roberts, L. (1962). Picture coding using pseudo-random noise. IRE Transactions on Information
Theory, 8(2):145–154. (Cited on page 4)

Rodr´ıguez-Barroso, N., Mart´ınez-C´amara, E., Luz´on, M., Seco, G. G., Veganzones, M. ´A., and
Herrera, F. (2020). Dynamic federated learning model for identifying adversarial clients. arXiv
preprint arXiv:2007.15030. (Cited on page 19)
Ryabinin, M. and Gusev, A. (2020). Towards crowdsourced training of large neural networks using
decentralized mixture-of-experts. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and
Lin, H., editors, Advances in Neural Information Processing Systems, volume 33, pages 3659–3672.
Curran Associates, Inc. (Cited on page 1)
Sadiev, A., Danilova, M., Gorbunov, E., Horv´ath, S., Gidel, G., Dvurechensky, P., Gasnikov, A.,
and Richt´arik, P. (2023). High-probability bounds for stochastic optimization and variational
inequalities: the case of unbounded variance. arXiv preprint arXiv:2302.00999. (Cited on page 20)
Sadiev, A., Malinovsky, G., Gorbunov, E., Sokolov, I., Khaled, A., Burlachenko, K., and Richt´arik,
P. (2022). Federated optimization algorithms with random reshuffling and gradient compression.
arXiv preprint arXiv:2206.07021. (Cited on page 20)
Safaryan, M., Islamov, R., Qian, X., and Richt´arik, P. (2022). FedNL: Making Newton-type methods
applicable to federated learning. In International Conference on Machine Learning. (arXiv preprint
arXiv:2106.02969, 2021). (Cited on page 20)
Schmidt, M., Le Roux, N., and Bach, F. (2017). Minimizing finite sums with the stochastic average
gradient. Mathematical Programming, 162(1):83–112. (Cited on page 19)
Seide, F., Fu, H., Droppo, J., Li, G., and Yu, D. (2014). 1-bit stochastic gradient descent and its
application to data-parallel distributed training of speech dnns. In Fifteenth Annual Conference of
the International Speech Communication Association. Citeseer. (Cited on page 20)
Stich, S. U., Cordonnier, J.-B., and Jaggi, M. (2018). Sparsified SGD with memory. In Advances in
Neural Information Processing Systems, volume 31. (Cited on pages 4 and 20)
Su, L. and Vaidya, N. H. (2016). Fault-tolerant multi-agent optimization: optimal iterative distributed
algorithms. In Proceedings of the 2016 ACM Symposium on Principles of Distributed Computing,
pages 425–434. (Cited on page 1)
Szlendak, R., Tyurin, A., and Richt´arik, P. (2022). Permutation compressors for provably faster
distributed nonconvex optimization. In International Conference on Learning Representations.
(Cited on page 25)
Vaswani, S., Bach, F., and Schmidt, M. (2019). Fast and faster convergence of SGD for over-
parameterized models and an accelerated perceptron. In International Conference on Artificial
Intelligence and Statistics, pages 1195–1204. (Cited on page 26)
Vogels, T., Karimireddy, S. P., and Jaggi, M. (2019).
Powersgd: Practical low-rank gradient
compression for distributed optimization. In Advances in Neural Information Processing Systems,
volume 32. (Cited on page 20)

15


---Page Break---
Wen, W., Xu, C., Yan, F., Wu, C., Wang, Y., Chen, Y., and Li, H. (2017). Terngrad: Ternary
gradients to reduce communication in distributed deep learning. In Advances in Neural Information
Processing Systems, volume 30. (Cited on page 20)
Wu, Z., Ling, Q., Chen, T., and Giannakis, G. B. (2020). Federated variance-reduced stochastic
gradient descent with robustness to byzantine attacks. IEEE Transactions on Signal Processing,
68:4583–4596. (Cited on pages 2 and 4)

Xie, C., Koyejo, O., and Gupta, I. (2020a). Fall of empires: Breaking byzantine-tolerant sgd by inner
product manipulation. In Uncertainty in Artificial Intelligence, pages 261–270. (Cited on page 2)

Xie, C., Koyejo, S., and Gupta, I. (2020b). Zeno++: Robust fully asynchronous sgd. In International
Conference on Machine Learning, pages 10495–10503. PMLR. (Cited on page 20)
Xu, X. and Lyu, L. (2020). Towards building a robust and fair federated learning system. arXiv
preprint arXiv:2011.10464. (Cited on page 19)
Yang, Y.-R. and Li, W.-J. (2023). Buffered asynchronous sgd for byzantine learning. Journal of
Machine Learning Research, 24(204):1–62. (Cited on page 20)

Yin, D., Chen, Y., Kannan, R., and Bartlett, P. (2018). Byzantine-robust distributed learning: Towards
optimal statistical rates. In International Conference on Machine Learning, pages 5650–5659.
(Cited on pages 2 and 4)
Zhang, J., He, T., Sra, S., and Jadbabaie, A. (2020a). Why gradient clipping accelerates training: A
theoretical justification for adaptivity. In International Conference on Learning Representations.
(arXiv preprint arXiv:1905.11881, 2019). (Cited on page 20)
Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S., Kumar, S., and Sra, S. (2020b). Why
are adaptive methods good for attention models? In Advances in Neural Information Processing
Systems, volume 33, pages 15383–15393. (Cited on pages 20 and 57)

Zhu, H. and Ling, Q. (2021). Broadcast: Reducing both stochastic and compression noise to robustify
communication-efficient federated learning. arXiv preprint arXiv:2104.06685. (Cited on pages 2
and 20)

16


---Page Break---
Contents

1
Introduction
1

1.1
Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

2
Preliminaries
3

3
New Method: Byz-VR-MARINA-PP
5

4
Convergence Results
6

5
Numerical Experiments
9

6
Conclusion and Future Work
10

A Extra Related Work
19

B
Useful Facts
22

C Justification of Assumption 2.3
23

D General Analysis
25

D.1
Refined Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

D.2 Technical Lemmas
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

D.3
Main Results
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42

E Analysis for Bounded Compressors
46

E.1
Technical Lemmas
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46

E.2
Main Results
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
56

E.3
On the Technical Non-Triviality of the Analysis . . . . . . . . . . . . . . . . . . .
57

F
Byz-VR-MARINA-PP+: Simplified Version of Byz-VR-MARINA-PP
58

F.1
Analysis for Bounded Compressors
. . . . . . . . . . . . . . . . . . . . . . . . .
58

F.2
Discussion of the Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
63

G Analysis without Full-Batch Gradient Computations
65

G.1
New Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
66

G.2
Main Results for Byz-VR-MARINA without Full-Batch Gradient Computations . .
70

G.2.1
General Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
70

G.2.2
Results for Bounded Compressors . . . . . . . . . . . . . . . . . . . . . .
71

G.3
Main Results for Byz-VR-MARINA+ without Full-Batch Gradient Computations .
72

G.3.1
Results for Bounded Compressors . . . . . . . . . . . . . . . . . . . . . .
72

H Experimental Details and Extra Experiments
74

H.1
Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
74

17


---Page Break---
H.2
Extra Experiments
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
74

18


---Page Break---
A
Extra Related Work

Further Comparison with Data and Diggavi (2021).
As we mention in the main text, Data and
Diggavi (2021) assume that 3B is smaller than C. More precisely, Data and Diggavi (2021) assume
that B ≤ϵC, where ϵ ≤1

3 −ϵ′ for some parameter ϵ′ > 0 that will be explained later. That is, the
results from Data and Diggavi (2021) do not hold when C is smaller than 3B, and, in particular,
their algorithm cannot tolerate the situation when the server samples only Byzantine workers at some
particular communication round. We also notice that when C ≥4B, then existing methods such as
Byz-VR-MARINA (Gorbunov et al., 2023) or Client Momentum (Karimireddy et al., 2021, 2022)
can be applied without any changes to get a provable convergence.

Next, Data and Diggavi (2021) derive the upper bounds for the expected squared distance to the
solution (in the strongly convex case) and the averaged expected squared norm of the gradient (in
the non-convex case), where the expectation is taken w.r.t. the sampling of stochastic gradients
only and the bounds itself hold with probability at least 1 −K

H exp

−ϵ′2(1−ϵ)C

16

, where H is the
number of local steps. For simplicity consider the best-case scenario: H = 1 (local steps deteriorate
the results from Data and Diggavi (2021)). Then, the lower bound for this probability becomes
negative when either C is not large enough or when K is large or when ϵ is close to 1

3, e.g., for
K = 106, ϵ = ϵ′ = 1

6, C = 5000 this lower bound is smaller than −720, meaning that in this case,
the result does not guarantee convergence. In contrast, our results have classical convergence criteria,
where the expectations are taken w.r.t. the all randomness.

Finally, the bounds from Data and Diggavi (2021) have non-reduceable terms even for homogeneous
data case: these terms are proportional to σ2

b , where σ2 is the upper bound for the variance of the
stochastic estimator on regular clients and b is the batchsize. In contrast, our results have only
decreasing terms in the upper bounds when the data is homogeneous.

Byzantine robustness.
There exist various approaches to achieving Byzantine robustness (Lyu
et al., 2020). Alistarh et al. (2018); Allen-Zhu et al. (2021) rely on the concentration inequalities for
the stochastic gradients with bounded noise to iteratively remove them from the training. Karimireddy
et al. (2021) formalize the definition of robust aggregation and propose the first provably robust
aggregation rule called CenteredClip and the first provably Byzantine robust method under bounded
variance assumption for homogeneous problems, i.e., when all good workers share one dataset. In
particular, the method from (Karimireddy et al., 2021) uses client momentum on the clients that helps
to memorize previous steps for good workers and withstand time-coupled attacks. This approach is
extended by He et al. (2022) to the setup of decentralized learning. Allouah et al. (2023) develop
an alternative definition for robust aggregation and propose a new aggregation rule satisfying their
definition. Karimireddy et al. (2022) generalize these results to the heterogeneous data case and derive
lower bounds for the optimization error that one can achieve in the heterogeneous case. Based on the
formalism from Karimireddy et al. (2021), Gorbunov et al. (2022) propose a server-free approach
that uses random checks of computations and bans of peers. This trick allows the elimination of all
Byzantine workers after a finite number of steps on average. There are also many other approaches,
e.g., one can use redundant computations of the stochastic gradients (Chen et al., 2018; Rajput et al.,
2019) or introduce reputation metrics (Rodr´ıguez-Barroso et al., 2020; Regatti et al., 2020; Xu and
Lyu, 2020) to achieve some robustness, see also a recent survey by Lyu et al. (2020).

Variance reduction.
The literature on variance-reduced methods is very rich (Gower et al., 2020).
The first variance-reduced methods are designed to fix the convergence of standard Stochastic Gradient
Descent (SGD) and make it convergent to any predefined accuracy even with constant stepsizes. Such
methods as SAG (Schmidt et al., 2017), SVRG (Johnson and Zhang, 2013), SAGA (Defazio et al.,
2014) are developed mainly for (strongly) convex smooth optimization problems, while methods like
SARAH (Nguyen et al., 2017), STORM (Cutkosky and Orabona, 2019), GeomSARAH (Horv´ath
et al., 2023), PAGE (Li et al., 2021) are designed for general smooth non-convex problems. In
this paper, we use GeomSARAH/PAGE-type variance reduction as the main building block of the
method that makes the method robust to Byzantine attacks.

Partial participation and client sampling.
In the context of Byzantine-robust learning, there exists
one work that develops and analyzes the method with partial participation (Data and Diggavi, 2021).
However, this work relies on the restrictive assumption that the number of participating clients at each

19


---Page Break---
round is at least three times larger than the number of Byzantine workers. In this case, Byzantines
cannot form a majority, and standard methods can be applied without any changes. In contrast, our
method converges in more challenging scenarios, e.g., Byz-VR-MARINA-PP provably converges
even when the server samples one client, which can be Byzantine. The results from Data and Diggavi
(2021) have some other noticeable limitations that we discuss in Appendix A.

Communication compression.
The literature on communication compression can be roughly
divided into two huge groups. The first group studies the methods with unbiased communication
compression. Different compression operators in the application to Distributed SGD/GD are studied
in (Alistarh et al., 2017; Wen et al., 2017; Khirirat et al., 2018). To improve the convergence rate by
fixing the error coming from the compression Mishchenko et al. (2019) propose to apply compression
to the special gradient differences. Multiple extensions and generalizations of mentioned techniques
are proposed and analyzed in the literature, e.g., see (Horv´ath et al., 2023; Gorbunov et al., 2021; Li
et al., 2020; Qian et al., 2021; Basu et al., 2019; Haddadpour et al., 2021; Sadiev et al., 2022; Islamov
et al., 2021; Safaryan et al., 2022).

Another large part of the literature on compressed communication is devoted to biased compression
operators (Ajalloeian and Stich, 2020; Demidovich et al., 2023). Typically, such compression
operators require more algorithmic changes than unbiased compressors since na¨ıve combinations
of biased compression with standard methods (e.g., Distributed GD) can diverge (Beznosikov et al.,
2020). Error feedback is one of the most popular ways of utilizing biased compression operators in
practice (Seide et al., 2014; Stich et al., 2018; Vogels et al., 2019), see also (Richt´arik et al., 2021;
Fatkhullin et al., 2021) for the modern version of error feedback with better theoretical guarantees for
non-convex problems.

In the context of Byzantine robustness, methods with communication compression are also studied.
The existing approaches are based on aggregation rules based on the norms of the updates (Ghosh
et al., 2020, 2021), SignSGD and majority vote (Bernstein et al., 2019), SAGA-type variance
reduction coupled with unbiased compression (Zhu and Ling, 2021), and GeomSARAH/PAGE-type
variance reduction combined with unbiased compression (Gorbunov et al., 2023).

Gradient clipping.
Gradient clipping has multiple useful properties and applications. Originally it
was used by Pascanu et al. (2013) to reduce the effect of exploding gradients during the training of
RNNs. Gradient clipping is also a popular tool for achieving provable differential privacy (Abadi
et al., 2016; Chen et al., 2020), convergence under generalized notions of smoothness (Zhang et al.,
2020a; Mai and Johansson, 2021) and better (high-probability) convergence under heavy-tailed
noise assumption (Zhang et al., 2020b; Nazin et al., 2019; Gorbunov et al., 2020; Sadiev et al.,
2023; Nguyen et al., 2023). In the context of Byzantine-robust learning, gradient clipping is also
utilized to design provably robust aggregation (Karimireddy et al., 2021). Our work proposes a novel
useful application of clipping, i.e., we utilize clipping to achieve Byzantine robustness with partial
participation of clients.

Byzantine-robust asynchronous methods.
Byzantine-robust asynchronous methods are also
very relevant to the problem of partial participation in the Byzantine-robust learning. Indeed, the
asynchronous methods like Asynchronous SGD (Agarwal and Duchi, 2011; Nedi´c et al., 2001)
naturally have partial participation since whenever some worker finishes the computation (of the
stochastic gradients), this worker immediately sends the update to the server and the server applies
this update without waiting all other clients. However, without extra assumptions asynchronous
methods cannot be tolerate Byzantine attacks: Byzantine clients could immediately send any vector
to the server to guarantee that their update is received earlier than the updates from regular clients.
Clearly, such a behavior of Byzantine workers leads to the divergence of the method unless the server
has additional information that can be used for acceptance/rejection of the update or some other
alternation of the communication protocol preventing the situations when some client updates the
model too many times in a row is applied.

Therefore, the existing approaches addressing this important problem rely on extra assumptions.
Damaskinos et al. (2018) propose to use Lipschitz filter and frequency filters in order to filter out
Byzantine workers. Next, Xie et al. (2020b); Fang et al. (2022) use additional validation data on the
server to decide whether to accept the update from workers. This assumption is restrictive for many
FL applications when the data on clients is private and is not available on the server. Yang and Li

20


---Page Break---
(2023) propose so-called BASGD (and its momentum version) where the key idea is to split workers
into the buffers and wait until each buffer gets at least one gradient update. In the case when the
number of buffers is sufficiently large (at least 2B, where B is the number of Byzantine workers), the
authors show that BASGD converges. However, this means that to make the step BASGD requires
to collect sufficiently large number of gradients such that the good buffers form majority, which is
closer to full participation than to the partial participation in the worst case.

We emphasize that in our work we consider a different setup of synchronous communications with
partial participation. The approaches discussed in the above paragraph cannot be directly applied to
the problem considered in this paper without extra assumptions.

21


---Page Break---
B
Useful Facts

For all a, b ∈Rd and α > 0, p ∈(0, 1] the following relations hold:

2⟨a, b⟩= ∥a∥2 + ∥b∥2 −∥a −b∥2
(11)

∥a + b∥2 ≤(1 + α)∥a∥2 +
 
1 + α−1
∥b∥2
(12)

−∥a −b∥2 ≤−
1
1 + α∥a∥2 + 1

α∥b∥2,
(13)

(1 −p)

1 + p

2


≤1 −p

2,
p ≥0
(14)

(1 −p)

1 + p

2

 
1 + p

4


≤1 −p

4
p ≥0.
(15)

Lemma B.1. (Lemma 5 from (Richt´arik et al., 2021)). Let a, b > 0. If 0 ≤γ ≤
1
√a+b, then

aγ2 + bγ ≤1. The bound is tight up to the factor of 2 since
1
√a+b ≤min
n
1
√a, 1

b
o
≤
2
√a+b.

22


---Page Break---
C
Justification of Assumption 2.3

Algorithm 2 Bucketing Algorithm (Karimireddy et al., 2022)

1: Input: {x1, . . . , xn}, s ∈N – bucket size, Aggr – aggregation rule
2: Sample random permutation π = (π(1), . . . , π(n)) of [n]

3: Compute yi = 1

s
Pmin{si,n}
k=s(i−1)+1 xπ(k) for i = 1, . . . , ⌈n/s⌉

4: Return: bx = Aggr(y1, . . . , y⌈n/s⌉)

Krum and Krum ◦Bucketing.
Krum aggregation rule is defined as

Krum(x1, . . . , xn) =
argmin
xi∈{x1,...,xn}

X

j∈Si
∥xj −xi∥2,

where Si ⊂{x1, . . . , xn} is the subset of n −B −2 closest vectors to xi.
By definition,
Krum(x1, . . . , xn) ∈{x1, . . . , xn} and, thus ∥Krum(x1, . . . , xn)∥≤maxi∈[n] ∥xi∥, i.e., As-
sumption 2.3 holds with FA = 1. Since Krum ◦Bucketing applies Krum aggregation to av-
erages yi over the buckets and ∥yi∥≤
1
s
Pmin{si,n}
k=s(i−1)+1 ∥xπ(k)∥≤maxi∈[n] ∥xi∥, we have that
∥Krum ◦Bucketing(x1, . . . , xn)∥≤maxi∈[n] ∥xi∥.

Geometric median (GM) and GM ◦Bucketing.
Geometric median is defined as follows:

GM(x1, . . . , xn) = argmin
x∈Rd

n
X

i=1
∥x −xi∥.
(16)

One can show that GM(x1, . . . , xn)
∈
Conv(x1, . . . , xn)
:=
{x
∈
Rd
|
x
=
Pn
i=1 αixi for some α1, . . . , αn ≥1 such that Pn
i=1 αi = 1}, i.e., geometric median belongs to the
convex hull of the inputs. Indeed, let GM(x1, . . . , xn) = x = ˆx + ˜x, where ˆx is the projection of x
on Conv(x1, . . . , xn) and ˜x = x −ˆx. Then, the optimality condition implies that ⟨ˆx −x, y −ˆx⟩≥0
for all y ∈Conv(x1, . . . , xn). In particular, for all i ∈[n] we have ⟨ˆx −x, xi −ˆx⟩≥0. Since

⟨ˆx −x, xi −ˆx⟩
=
⟨˜x, ˆx −xi⟩= 1

2∥˜x + ˆx −xi∥2 −1

2∥˜x∥2 −1

2∥ˆx −xi∥2

=
1
2∥x −xi∥2 −1

2∥˜x∥2 −1

2∥ˆx −xi∥2

≤
1
2∥x −xi∥2 −1

2∥ˆx −xi∥2,

we get that ∥x −xi∥≥∥ˆx −xi∥for all i ∈[n] and the equality holds if and only if ˜x = 0.
Therefore, argmin from (16) is achieved for x such that x = ˆx, meaning that GM(x1, . . . , xn) ∈
Conv(x1, . . . , xn). Therefore, there exist some coefficients α1, . . . , αn ≥0 such that Pn
i=1 αi = 1
and GM(x1, . . . , xn) = Pn
i=1 αixi, implying that

∥GM(x1, . . . , xn)∥≤

n
X

i=1
αi∥xi∥≤max
i∈[n] ∥xi∥.

That is, GM satisfies Assumption 2.3 with FA = 1. Similarly to the case of Krum ◦Bucketing, we
also have ∥GM ◦Bucketing(x1, . . . , xn)∥≤maxi∈[n] ∥xi∥.

Coordinate-wise median (CM) and CM ◦Bucketing.
Coordinate-wise median (CM) is formally
defined as

CM(x1, . . . , xn) = argmin
x∈Rd

n
X

i=1
∥x −xi∥1,
(17)

where ∥· ∥1 denotes ℓ1-norm. This is equivalent to geometric median/median applied to vectors
x1, . . . , xn component-wise. Therefore, from the above derivations for GM we have

∥CM(x1, . . . , xn)∥∞
≤
max
i∈[n] ∥xi∥∞,

∥CM ◦Bucketing(x1, . . . , xn)∥∞
≤
max
i∈[n] ∥xi∥∞,

23


---Page Break---
where ∥· ∥∞denotes ℓ∞-norm. Therefore, due to the standard relations between ℓ2- and ℓ∞-norms,
i.e., ∥a∥∞≤∥a∥≤
√

d∥a∥∞for any a ∈Rd, we have

∥CM(x1, . . . , xn)∥
≤
√

d max
i∈[n] ∥xi∥,

∥CM ◦Bucketing(x1, . . . , xn)∥
≤
√

d max
i∈[n] ∥xi∥,

i.e., Assumption 2.3 is satisfied with FA =
√

d.

24


---Page Break---
D
General Analysis

D.1
Refined Assumptions

For simplicity, in the main part of our paper, we present simplified versions of our main results.
However, our analysis works under more general assumptions presented in this section.

Assumption on bC.
In all the results of this paper, we assume that n ≥bC ≥max{1, δrealn/δ}. This
condition ensures that the robust aggregation makes sense when ck = 1, i.e., at least 1 −δ proportion
of sampled workers are not Byzantine ones when ck = 1.

Refined smoothness.
The following assumption is classical for the literature on non-convex
optimization.
Assumption D.1 (L-smoothness). We assume that function f : Rd →R is L-smooth, i.e., for all
x, y ∈Rd we have
∥∇f(x) −∇f(y)∥≤L∥x −y∥.
(18)
Moreover, we assume that f is uniformly lower bounded by f ∗∈R, i.e., f ∗:= infx∈Rd f(x). In
addition, we assume that fi is Li-smooth for all i ∈G, i.e., for all x, y ∈Rd

∥∇fi(x) −∇fi(y)∥≤Li∥x −y∥.
(19)

We notice here that (19) implies L-smoothness of f with L ≤1

G
P

i∈G Li, i.e., smoothness constant
of f can be better than the averaged smoothness constant of the local loss functions on the regular
clients.

Following Gorbunov et al. (2023), we consider refined assumptions on the smoothness.
Assumption D.2 (Global Hessian variance assumption (Szlendak et al., 2022)). We assume that
there exists L± ≥0 such that for all x, y ∈Rd

1
G

X

i∈G
∥∇fi(x) −∇fi(y)∥2 −∥∇f(x) −∇f(y)∥2 ≤L2
±∥x −y∥2.
(20)

We notice that (19) implies (20) with L± ≤maxi∈G Li. Szlendak et al. (2022) prove that L± satisfies
the following relation: L2
avg −L2 ≤L2
± ≤L2
avg, where L2
avg :=
1
G
P

i∈G L2
i . In particular, it is
possible that L± = 0 even if the data on the good workers is heterogeneous.
Assumption D.3 (Local Hessian variance assumption (Gorbunov et al., 2023)). We assume that there
exists L± ≥0 such that for all x, y ∈Rd

1
G

X

i∈G
E
b∆i(x, y) −∆i(x, y)

2
≤L2
±
b ∥x −y∥2,
(21)

where ∆i(x, y) := ∇fi(x)−∇fi(y) and b∆i(x, y) is an unbiased mini-batched estimator of ∆i(x, y)
with batch size b.

This assumption incorporates considerations for the smoothness characteristics inherent in all func-
tions {fi,j}i∈G,j∈[m], the sampling policy, and the similarity among the functions {fi,j}i∈G,j∈[m].
Gorbunov et al. (2023) have demonstrated that, assuming smoothness of {fi,j}i∈G,j∈[m], Assump-
tion D.3 holds for various standard sampling strategies, including uniform and importance samplings.

For part of our results, we also need to assume smoothness of all {fi,j}i∈G,j∈[m] explicitly.
Assumption D.4 (Smoothness of fi,j (optional)). We assume that for all i ∈G and j ∈[m] there
exists Li,j ≥0 such that fi,j is Li,j-smooth, i.e., for all x, y ∈Rd

∥∇fi,j(x) −∇fi,j(y)∥≤Li,j∥x −y∥.
(22)

Refined heterogeneity.
Instead of Assumption 2.5, we consider a more generalized one.
Assumption D.5 ((B, ζ2)-heterogeneity). We assume that good clients have
 
B, ζ2
-heterogeneous
local loss functions for some B ≥0, ζ ≥0, i.e.,
1
G

X

i∈G
∥∇fi(x) −∇f(x)∥2 ≤B∥∇f(x)∥2 + ζ2
∀x ∈Rd

25


---Page Break---
When B = 0, the above assumption recovers Assumption 2.5. However, it also covers some situations
when the model is over-parameterized (Vaswani et al., 2019) and can hold with smaller values of ζ2.
This assumption is also used in (Karimireddy et al., 2022; Gorbunov et al., 2023).

D.2
Technical Lemmas

Lemma D.6. Let X be a random vector in Rd and e
X = clipλ(X). Assume that E[X] = x ∈Rd
and ∥x∥≤λ/2, then

E
h
∥e
X −x∥2i
≤10E ∥X −x∥2 .

Proof. The proof follows a similar procedure to that presented in Lemma F.5 from (Gorbunov et al.,
2020). To commence the proof, we introduce two indicator random variables:

χ = I{X:∥X∥>λ} =

1,
if ∥X∥> λ,
0,
otherwise
, η = I{X:∥X−x∥> λ

2 } =
1,
if ∥X −x∥> λ

2
0,
otherwise
.

Moreover, since ∥X∥≤∥x∥+ ∥X −x∥
∥x∥≤λ/2
≤
λ
2 + ∥X −x∥, we have χ ≤η. Using that we get

e
X = min

1,
λ
∥X∥


X = χ λ

∥X∥X + (1 −χ)X.

By Markov’s inequality,

E[η] = P

∥X −x∥> λ

2


= P

∥X −x∥2 > λ2

4


≤4

λ2 E

∥X −x∥2
.
(23)

Using ∥e
X −x∥≤∥e
X∥+ ∥x∥≤λ + λ

2 = 3λ

2 , we obtain

E
h
∥e
X −x∥2i
= E
h
∥e
X −x∥2χ + ∥e
X −x∥2(1 −χ)
i

= E

"

χ

λ
∥X∥X −x


2
+ ∥X −x∥2(1 −χ)

#

≤E

"

χ

λ
∥X∥X
 + ∥x∥
2
+ ∥X −x∥2(1 −χ)

#

∥x∥≤λ

2
≤

 

E

"

χ
3λ

2

2
+ ∥X −x∥2
#!

,

where in the last inequality we applied 1 −χ ≤1. Using (23) and χ ≤η we get

E
h
∥e
X −x∥2i
≤9λ2

4

 2

λ

2
E

∥X −x∥2
+ E

∥X −x∥2

≤10E

∥X −x∥2
.

Lemma D.7 (Lemma 2 from Li et al. (2021)). Assume that function f is L-smooth (Assumption D.1)
and xk+1 = xk −γgk. Then

f
 
xk+1
≤f
 
xk
−γ

2

∇f
 
xk2 −
 1

2γ −L

2

 xk+1 −xk2 + γ

2

gk −∇f
 
xk2 .

Lemma D.8. Let Assumptions D.1, D.2, D.3 hold and the Compression Operator satisfy Definition 2.2.
Let us define ”ideal” estimator:

gk+1 =

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

1
Gk
C
P

i∈Gk
C
∇fi(xk+1),
cn = 1,
[1]

gk + ∇f
 
xk+1
−∇f
 
xk
,
cn = 0 and Gk
C < (1 −δ)C,
[2]

gk +
1
Gk
C
P

i∈Gk
C
clipλ

Q

b∆i
 
xk+1, xk
,
cn = 0 and Gk
C ≥(1 −δ)C.
[3]

26


---Page Break---
Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

A1 = E
hgk+1 −∇f
 
xk+12i

≤(1 −p)

1 + p

4


E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG


1 + 4

p

 2 · PGk
C n

C

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i,

where pG = Prob

Gk
C ≥(1 −δ)C
	
and PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

Proof. Let us examine the expected value of the squared difference between ideal estimator and full
gradient:

A1 = E
hgk+1 −∇f
 
xk+12i

= E
h
Ek
hgk+1 −∇f
 
xk+12ii

= (1 −p) pGE



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∇f
 
xk+1


2

| [3]





+ (1 −p)(1 −pG)E
h
Ek
hgk −∇f(xk)
2i
| [2]
i
+ pE







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

.

Using (12) and ∇f
 
xk
−∇f
 
xk
= 0 we obtain

B1 = E



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∇f
 
xk+1


2

| [3]





= E



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−∇f

xk+1
+ ∇f

xk
−∇f

xk


2

| [3]





(12)
≤

1 + p

4


E
hgk −∇f
 
xk2i

+

1 + 4

p


E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−

∇f(xk+1) −∇f(xk)



2

| [3]





=

1 + p

4


E
hgk −∇f(xk)
2i

+

1 + 4

p


E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]



.

Let us consider the last part of the inequality:

B′
1 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]





= E



ESk



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]







.

27


---Page Break---
Note that Gk
C ≥(1 −δ)C in this case:

B′
1 ≤
1
C(1 −δ)E



ESk



X

i∈Gk
C

Ek

clipλ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]









≤
1
C(1 −δ)E

"X

i∈G
ESk
h
IGk
C

i
Ek

clipλ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

=
1
C(1 −δ)E

"X

i∈G
PGk
C · Ek

clipλ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

,
(24)

where IGk
C is an indicator function for the event

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
and PGk
C
=

Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
is probability of such event. Note that ESk
h
IGk
C

i
= PGk
C. In
case of uniform sampling of clients we have

∀i ∈G
PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	

=
C
npG
·
X

(1−δ)C≤t≤C

 
G
t

 
n −G
C −t

 
n
C

−1!

,

pG =
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n −1
C −1

−1!

Now we can continue with inequalities:

B′
1 ≤
PGk
C
C(1 −δ)E

"X

i∈G
Ek

clipλ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

≤
PGk
C
C(1 −δ)E

"X

i∈G
Ek


EQ

clipλ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

(12)
≤
PGk
C
C(1 −δ)E

"X

i∈G
2Ek


EQ

clipλ

Q

b∆i

xk+1, xk
−∆i

xk+1, xk
2
| [3]

#

+
PGk
C
C(1 −δ)E

"X

i∈G
2Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

.

Using Lemma D.6 we have

B′
1

Lemma D.6

≤
PGk
C
C(1 −δ)E

"X

i∈G
20Ek


EQ

Q

b∆i

xk+1, xk
−∆i

xk+1, xk
2
| [3]

#

+
PGk
C
C(1 −δ)E

"X

i∈G
2Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

≤
20 · PGk
C
C(1 −δ)E

"X

i∈G
Ek


EQ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G
Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

≤
20 · PGk
C
C(1 −δ)E

"X

i∈G
Ek


EQ

Q

b∆i

xk+1, xk
2
−
X

i∈G

∆i

xk+1, xk
2
| [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G
Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

.

28


---Page Break---
Applying Definition 2.2 of Unbiased Compressor we have

B′
1 ≤
20 · PGk
C
C(1 −δ)E

"X

i∈G
(1 + ω)Ek
b∆i
 
xk+1, xk
2
−
X

i∈G

∆i
 
xk+1, xk2 | [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

≤
20 · PGk
C
C(1 −δ)E

"X

i∈G
(1 + ω)Ek
b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
#

+
20 · PGk
C
C(1 −δ)E

"X

i∈G
(1 + ω)Ek
∆i
 
xk+1, xk2 −
X

i∈G
Ek
∆i
 
xk+1, xk2 | [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

.

Now we combine terms and have

B′
1 ≤
20 · PGk
C
C(1 −δ)(1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
20 · PGk
C
C(1 −δ)ωE

"X

i∈G

∆i
 
xk+1, xk2 | [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

=
20 · PGk
C
C(1 −δ)(1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
20 · PGk
C
C(1 −δ)ωE

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 + ∥∆
 
xk+1, xk
∥2 | [3]

#

+
2 · PGk
C
C(1 −δ)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

.

Rearranging terms leads to

B′
1 ≤
20 · PGk
C
C(1 −δ)(1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
2 · PGk
C
C(1 −δ)(10ω + 1)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

+
20 · PGk
C
C(1 −δ)ωE

"X

i∈G

∆
 
xk+1, xk2 | [3]

#

.

Now we apply Assumptions D.1, D.2, D.3:

B′
1 ≤
20 · PGk
C
C(1 −δ)(1 + ω)E

GL2
±
b ∥xk+1 −xk∥2


+
2 · PGk
C
C(1 −δ)(10ω + 1)E

GL2
±∥xk+1 −xk∥2

+
20 · PGk
C
C(1 −δ)ωE
h
GL2 xk+1 −xk2i
.

29


---Page Break---
Finally, we have

B′
1 ≤
2 · PGk
C · G

C(1 −δ)


10ωL2 + (10ω + 1)L2
± + 10(ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

Let us plug obtained results:

B1 ≤

1 + p

4


E
hgk −∇f(xk)
2i

+

1 + 4

p

 2 · PGk
C · G

C(1 −δ)


10ωL2 + (10ω + 1)L2
± + 10(ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

Let us consider the term E







1
Gk
b
C

P

i∈Gk
b
C
∇fi(xk+1) −∇f(xk+1)



2

:

E







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

≤E



1

Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)
2





≤
1

(1 −δ) bC
E




X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)
2





=
1

(1 −δ) bC
E

"X

i∈G
IGk
b
C
∇fi(xk+1) −∇f(xk+1)
2
#

Using definition of PGk
C we get

E







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

≤
PGk
b
C
(1 −δ) bC
E

"X

i∈G

∇fi(xk+1) −∇f(xk+1)
2
#

≤
G · PGk
b
C
(1 −δ) bCG
E

"X

i∈G

∇fi(xk+1) −∇f(xk+1)
2
#

Using Assumption D.5 we get

E







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

≤
G · PGk
b
C
(1 −δ) bC
E

B∥∇f(x)∥2 + ζ2

≤
δrealn · PGk
b
C
(1 −δ) δrealn

δ
E

B∥∇f(x)∥2 + ζ2

=
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2
(25)

30


---Page Break---
Also, we have

A1 = E
hgk+1 −∇f(xk+1)
2i

≤(1 −p)pGB1 + (1 −p)(1 −pG)E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

≤(1 −p)pG

1 + p

4


E
hgk −∇f(xk)
2i

+ (1 −p)pG


1 + 4

p

 2 · PGk
C · G

C(1 −δ)

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i

+ (1 −p)(1 −pG)E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2
.

To simplify the bound we use
 
1 + p

4 > 1

and obtain

A1 ≤(1 −p)pG

1 + p

4


E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG


1 + 4

p

 2 · PGk
C · G

C(1 −δ)

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i

+ (1 −p)(1 −pG)E
hgk −∇f(xk)
2i

≤(1 −p)pG

1 + p

4


E
hgk −∇f(xk)
2i

+ (1 −p)pG


1 + 4

p

 2 · PGk
C · G

C(1 −δ)

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i

+ (1 −p)(1 −pG)

1 + p

4


E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

≤(1 −p)

1 + p

4


E
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG


1 + 4

p

 2 · PGk
C n

C

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i.

Lemma D.9. Let us define ”ideal” estimator:

gk+1 =

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

1
Gk
C
P

i∈Gk
C
∇fi(xk+1),
cn = 1,
[1]

gk + ∇f
 
xk+1
−∇f
 
xk
,
cn = 0 and Gk
C < (1 −δ)C,
[2]

gk +
1
Gk
C
P

i∈Gk
C
clipλ

Q

b∆i
 
xk+1, xk
,
cn = 0 and Gk
C ≥(1 −δ)C.
[3]

Also let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

A2 = E
hgk+1 −gk+12i

≤pE
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

+ (1 −p)pGE



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









+ (1 −p)(1 −pG)E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]

,

31


---Page Break---
where pG = Prob

Gk
C ≥(1 −δ)C
	
.

Proof. Using conditional expectations we have

A2 = E
h
Ek
hgk+1 −gk+12ii

= pE
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

+(1 −p)pGE



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−

gk + ARAggk+1
Q



2

| [3]





+ (1 −p)(1 −pG)E

Ek

gk + ∇f(xk+1) −∇f(xk) −

gk + ARAggk+1
Q

2
| [2]

.

After simplification, we get the following bound:

A2 ≤pE
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

+ (1 −p)pGE



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









+ (1 −p)(1 −pG)E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]

.

Lemma D.10. Let Assumptions D.1 and D.5 hold and Aggregation Operator (ARAgg) satisfy Def-
inition 2.1. Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1)
satisfy

T1 = E
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2,

where eB := 0 and eζ2 := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eζ2 :=
PGk
b
C
Gζ2

(1−δ) b
C when bC < n.

Proof. Using the definition of aggregation operator, we have

T1 = E
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

(12)
≤E



Ek






ARAgg
 
{gk+1
i
}i∈Sk

−
1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1)



2

| [1]





+ E



Ek







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

| [1]



.

32


---Page Break---
Since
1
Gk
b
C

P

i∈Gk
b
C
∇fi(xk+1) = ∇f(xk+1) with probability 1 when bC = n, we can estimate the last

term as

E



Ek







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

| [1]





≤










0,
if bC = n

E



1

Gk
b
C

P

i∈Gk
b
C
Ek
h∇fi(xk+1) −∇f(xk+1)
2i
| [1]



,
if bC < n

≤






0,
if bC = n
PGk
b
C
(1−δ) b
C
P

i∈G
E
h∇fi(xk+1) −∇f(xk+1)
2i
,
if bC < n

(As. D.5)
≤






0,
if bC = n
PGk
b
C
G

(1−δ) b
C
 
BE

∥∇f(xk+1)∥2
+ ζ2
,
if bC < n
= eBE

∥∇f(xk+1)∥2
+ eζ2,

where

eB :=






0,
if bC = n,
PGk
b
C
GB

(1−δ) b
C ,
if bC < n,
and
eζ2 :=






0,
if bC = n,
PGk
b
C
Gζ2

(1−δ) b
C ,
if bC < n.

33


---Page Break---
Using the above bound, we continue the estimation of T1 as follows:

T1

(Def. 2.1)
≤
E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

Ek
h∇fi
 
xk+1
−∇fl
 
xk+12 | [1]
i





+ eBE

∥∇f(xk+1)∥2
+ eζ2

(12)
≤E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E
h
2
∇fi
 
xk+1
−∇f
 
xk+12 | [1]
i





+ E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E
h
2
∇fl
 
xk+1
−∇f
 
xk+12 | [1]
i





+ eBE

∥∇f(xk+1)∥2
+ eζ2

= E



cδ

Gk
b
C

X

i∈Gk
b
C

4Ek
h∇fi
 
xk+1
−∇f
 
xk+12 | [1]
i


+ eBE

∥∇f(xk+1)∥2
+ eζ2

≤
PGk
b
Ccδ

(1 −δ) bC

X

i∈G
4Ek
h∇fi
 
xk+1
−∇f
 
xk+12i
+ eBE

∥∇f(xk+1)∥2
+ eζ2

(As. D.5)
≤

 4GPGk
b
CcδB

(1 −δ) bC
+ eB

!

E
h∇f
 
xk+12i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2

(12)
≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 +
∇f
 
xk+1
−∇f
 
xk2i

+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2

≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2,

which concludes the proof.

Lemma D.11. Let Assumptions D.1, D.2, D.3 hold and the Compression Operator satisfy Defini-
tion 2.2. Also let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤
8GPGk
C
(1 −δ)C


10(1 + ω)L2
±
b
+ (10ω + 1)L2
± + 10ωL2

cδE

∥xk+1 −xk∥2
,

where PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

34


---Page Break---
Proof. By the definition of robust aggregation, we have

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤E




cδ
D2

X

i,l∈Gk
C
i̸=l

Ek

clipλ

Q

b∆i

xk+1, xk
−clipλ

Q

b∆l

xk+1, xk
2 | [3]




,

where D2 = Gk
C(Gk
C −1). Next, we consider pair-wise differences:

T ′
2(i, l) = Ek

clipλ

Q

b∆i
 
xk+1, xk
−clipλ

Q

b∆l
 
xk+1, xk
2
| [3]


(12)
≤
2Ek

clipλ

Q

b
∆i

xk+1, xk
−∆i

xk+1, xk
+ ∆l

xk+1, xk
−clipλ

Q

b
∆l

xk+1, xk
2 | [3]


+ 2Ek
h∆i
 
xk+1, xk
−∆l
 
xk+1, xk2 | [3]
i

(12)
≤4Ek

clipλ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]


+ 4Ek

∆l
 
xk+1, xk
−clipλ

Q

b∆l
 
xk+1, xk
2
| [3]


+ 2Ek
h∆l
 
xk+1, xk
−∆i
 
xk+1, xk2 | [3]
i
]

(12)
≤4Ek

clipλ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]


+ 4Ek

∆l
 
xk+1, xk
−clipλ

Q

b∆l
 
xk+1, xk
2
| [3]


+ 4Ek
h∆l
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i

+ 4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i
.

35


---Page Break---
Now we can combine all parts together:

bT2 = E




1
Gk
C(Gk
C −1)

X

i,l∈Gk
C
i̸=l

T ′
2(i, l)





≤E




1
D2

X

i,l∈Gk
C
i̸=l

4Ek

clipλ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]





+ E




1
D2

X

i,l∈Gk
C
i̸=l

4Ek

∆l
 
xk+1, xk
−clipλ

Q

b∆l
 
xk+1, xk
2
| [3]





+ E




1
D2

X

i,l∈Gk
C
i̸=l

4Ek
h∆l
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i





+ E




1
D2

X

i,l∈Gk
C
i̸=l

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i



.

Rearranging the terms, we obtain

bT2 ≤E




1
D2

X

i,l∈Gk
C
i̸=l

8Ek

clipλ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]





+ E




1
D2

X

i,l∈Gk
C
i̸=l

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i



.

It leads to

bT2 ≤E



1

Gk
C

X

i∈Gk
C

8Ek

clipλ

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




Lemma D.6

≤
E



1

Gk
C

X

i∈Gk
C

80Ek

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

36


---Page Break---
Using variance decomposition we get

bT2 ≤E



1

Gk
C

X

i∈Gk
C

80Ek

Q

b∆i
 
xk+1, xk
2
| [3]




−E



1

Gk
C

X

i∈Gk
C

80Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Using properties of unbiased compressors (Definition 2.2) we have

bT2 ≤E



1

Gk
C

X

i∈Gk
C

80(1 + ω)Ek

b∆i
 
xk+1, xk
2
| [3]




−E



1

Gk
C

X

i∈Gk
C

80Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Also we have

bT2 ≤E



1

Gk
C

X

i∈Gk
C

80(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

80(1 + ω)Ek
h∆i
 
xk+1, xk2 | [3]
i




−E



1

Gk
C

X

i∈Gk
C

80Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Let us simplify the inequality:

bT2 ≤E



1

Gk
C

X

i∈Gk
C

80(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

80ωEk
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

37


---Page Break---
Using a variance decomposition once again, we get

bT2 ≤E



1

Gk
C

X

i∈Gk
C

80(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

80ωEk
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

80ωEk
h∆
 
xk+1, xk2 | [3]
i


.

Using a similar argument to the one used in the previous lemma, we obtain

bT2 ≤E

"
PGk
C
(1 −δ)C

X

i∈G
80(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]
#

+ E

"
PGk
C
(1 −δ)C

X

i∈G
80ωEk
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i#

+ E

"
PGk
C
(1 −δ)C

X

i∈G
8Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i#

+ E

"
PGk
C
(1 −δ)C

X

i∈G
80ωEk
h∆
 
xk+1, xk2 | [3]
i#

.

Using Assumptions D.1, D.2, D.3:

bT2 ≤E

"
80(1 + ω)GPGk
CL2
±
(1 −δ)Cb
∥xk+1 −xk∥2
#

+ E

"
8(10ω + 1)GPGk
CL2
±
(1 −δ)C
∥xk+1 −xk∥2
#

+ E

"
80GPGk
CωL2

(1 −δ)C
∥xk+1 −xk∥2
#

.

Finally, we obtain

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤
8GPGk
C
(1 −δ)C


10(1 + ω)L2
±
b
+ (10ω + 1)L2
± + 10ωL2

cδE

∥xk+1 −xk∥2
.

Lemma D.12. Let Assumptions 2.3 and D.1 hold. Also let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Assume that λk+1 = αλk+1∥xk+1 −xk∥. Then for all k ≥0 the iterates produced by Byz-VR-
MARINA-PP (Algorithm 1) satisfy

T3 = E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]


≤2(L2 + F 2
Aα2
λk+1)E
hxk+1 −xk2i

38


---Page Break---
Proof.

T3 = E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]


(12)
≤E

Ek


2
∇f(xk+1) −∇f(xk)
2 + 2
ARAggk+1
Q

2
| [2]


Using L-smoothness and Assumption 2.3 we have

T3

(12)
≤E
h
Ek
h
2L2 xk+1 −xk2 + 2F 2
Aλ2
k+1 | [2]
ii

≤E
h
Ek
h
2L2 xk+1 −xk2 + 2F 2
Aα2
λk+1∥xk+1 −xk∥2 | [2]
ii

≤2(L2 + F 2
Aα2
λk+1)E
hxk+1 −xk2i
.

Lemma D.13. Let Assumptions 2.3, D.1, D.2, D.3, D.5 hold and Compression Operator satisfy
Definition 2.2. Also let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

4


E
hgk −∇f
 
xk2i

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

where

A = 4

p

 
80

p

pGPGk
Cn

C
ω + 24
GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 160

p pG
GPGk
C
(1 −δ)C cδω

!

L2

+ 4

p

8

p

pGPGk
Cn

C
(10ω + 1) + 16

p pG
GPGk
C
(1 −δ)C cδ(10ω + 1)

L2
±

+ 4

p

160

p pG
GPGk
C
(1 −δ)C (1 + ω)cδ + 80

p pGPGk
C(1 + ω) n

C

 L2
±
b

+ 4

p

4

p(1 −pG)F 2
Aα2
λk+1


,

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

and where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n,

pG = Prob

Gk
C ≥(1 −δ)C
	
, and PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

39


---Page Break---
Proof. Let us combine bounds for A1 and A2 together:

A0 = E
hgk+1 −∇f
 
xk+12i

≤

1 + p

2


E
hgk+1 −∇f
 
xk+12i
+

1 + 2

p


E
hgk+1 −gk+12i

≤

1 + p

2


A1 +

1 + 2

p


A2

≤

1 + p

2


(1 −p)

1 + p

4


E
hgk −∇f
 
xk2i

+

1 + p

2


(1 −p)pG


1 + 4

p

 2 · PGk
C n

C

 

10ωL2 + (10ω + 1)L2
± +
10(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i

+

1 + p

2


p

 δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2
!

+

1 + 2

p


pE
h
Ek
hARAgg
 
∇f1(xk+1), . . . , ∇f b
C(xk+1)

−∇f(xk+1)
2i
| [1]
i

+

1 + 2

p


(1 −p)pGE



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−ARAggk+1
Q



2

| [3]









+

1 + 2

p


(1 −p)(1 −pG)E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]

.

Finally, we obtain the following bound:

A0

(12)
≤

1 −p

4


E
hgk −∇f
 
xk2i

+ 8

p

PGk
Cn

C
pG


10ωL2 + (10ω + 1)L2
± + 10(ω + 1)L2
±
b


E

∥xk+1 −xk∥2

+ 2p

 δ · PGk
b
C
1 −δ E

B∥∇f(x)∥2 + ζ2
!

+ (p + 2) E
h
Ek
hARAgg
 
∇f1(xk+1), . . . , ∇fn(xk+1)

−∇f(xk+1)
2i
| [1]
i

+ 2

ppGE



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









+ 2

p(1 −pG)E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]


40


---Page Break---
Now, we can apply Lemmas D.10, D.11, D.12:

A0 = E
hgk+1 −∇f
 
xk+12i

≤

1 −p

4


E
hgk −∇f
 
xk2i

+ 8

p

PGk
Cn

C
pG


10ωL2 + (10ω + 1)L2
± + 10(ω + 1)L2
±
b


E

∥xk+1 −xk∥2

+ (p + 2)

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i

+ 4 (p + 2)
GPGk
b
Ccδ

(1 −δ) bC
ζ2 + (p + 2)eζ2

+ 2

ppGE

80(1 + ω)
GPGk
C
(1 −δ)C
L2
±
b cδ∥xk+1 −xk∥2


+ 2

ppGE

8(10ω + 1)
GPGk
C
(1 −δ)C L2
±cδ∥xk+1 −xk∥2


+ 2

ppGE

80
GPGk
C
(1 −δ)C ωL2cδ∥xk+1 −xk∥2


+ 2

p(1 −pG)2(L2 + F 2
Aα2
λk+1)E
hxk+1 −xk2i

+ 2p
δPGk
b
C
1 −δ E

B∥∇f(x)∥2 + ζ2
.

Finally, we have

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

4


E
hgk −∇f
 
xk2i

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 E

∥xk+1 −xk∥2
,

where

A = 32pG

p2
PGk
Cn

C


10ωL2 + (10ω + 1)L2
± + 10(ω + 1)L2
±
b



+ 8

p2
GPGk
C
(1 −δ)C pGcδ

80(1 + ω)L2
±
b
+ 8(10ω + 1)L2
± + 80ωL2


+ 4

p

 24GPGk
b
CcδB

(1 −δ) bC
+ 6 eB

!

L2 + 16(1 −pG)

p2
(L2 + F 2
Aα2
λk+1),

and

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

where eD := 0 when bC = n, and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n. Once we simplify the equation, we
obtain

A = 4

p

 
80

p

pGPGk
Cn

C
ω + 24
GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 160

p pG
GPGk
C
(1 −δ)C cδω

!

L2

+ 4

p

8

p

pGPGk
Cn

C
(10ω + 1) + 16

p pG
GPGk
C
(1 −δ)C cδ(10ω + 1)

L2
±

+ 4

p

160

p pG
GPGk
C
(1 −δ)C (1 + ω)cδ + 80

p pGPGk
C(1 + ω) n

C

 L2
±
b

+ 4

p

4

p(1 −pG)F 2
Aα2
λk+1


.

41


---Page Break---
D.3
Main Results

Theorem D.14.
Let Assumptions 2.3,
D.1,
D.2,
D.3,
D.5 hold.
Setting λk+1
=
2 maxi∈G Li
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where

A = 4

p

 
80

p

pGPGk
Cn

C
ω + 24
GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 160

p pG
GPGk
C
(1 −δ)C cδω

!

L2

+ 4

p

8

p

pGPGk
Cn

C
(10ω + 1) + 16

p pG
GPGk
C
(1 −δ)C cδ(10ω + 1)

L2
±

+ 4

p

160

p pG
GPGk
C
(1 −δ)C (1 + ω)cδ + 80

p pGPGk
C(1 + ω) n

C

 L2
±
b

+ 4

p

4

p(1 −pG)F 2
Aα2
λk+1


,

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

and where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n,
and

PGk
C =
C
npG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n
C

−1!

,

pG = Prob

Gk
C ≥(1 −δ) C
	

=
X

⌈(1−δ)C⌉≤t≤C

 
G
t

 
n −G
C −t

 
n −1
C −1

−1!

.

Then for all K ≥0 the iterates produced by Byz-VR-MARINA (Algorithm 1) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 4 bDζ2

p −4 bB
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

2γ

p
g0 −∇f
 
x02 .

42


---Page Break---
Proof of Theorem D.14. For all k ≥0 we introduce Φk = f
 
xk
−f ∗+ 2γ

p
gk −∇f
 
xk2.
Using the results of Lemmas D.13 and D.7, we derive

E

Φk+1 (D.7)
≤
E

f
 
xk
−f ∗−
 1

2γ −L

2

 xk+1 −xk2 + γ

2

gk −∇f
 
xk2

−γ

2 E
h∇f
 
xk2i
+ 2γ

p E
hgk+1 −∇f
 
xk+12i

(D.13)
≤
E

f
 
xk
−f ∗−
 1

2γ −L

2

 xk+1 −xk2 + γ

2

gk −∇f
 
xk2

−γ

2 E
h∇f
 
xk2i
+ 2γ

p


1 −p

4


E
hgk −∇f
 
xk2i

+ 2γ

p


bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2


= E

f
 
xk
−f ∗
+ 2γ

p


1 −p

4


+ p

4


E
hgk −∇f
 
xk2i
+ 2 bDζ2γ

p

+ 1

2γ
 
1 −Lγ −Aγ2
E

∥xk+1 −xk∥2
−γ

2

 

1 −4 bB

p

!

E
h∇f
 
xk2i

= E

Φk
+ 2 bDζ2γ

p
+ 1

2γ
 
1 −Lγ −Aγ2
E

∥xk+1 −xk∥2

−γ

2

 

1 −4 bB

p

!

E
h∇f
 
xk2i
.

Using choice of stepsize and second condition: 0 < γ ≤
1
L+
√

A, 4 bB < p and lemma B.1 we have

E

Φk+1
≤E

Φk
+ 2 bDζ2γ

p
−γ

2

 

1 −4 bB

p

!

E
h∇f
 
xk2i

Next, we have γ

2

1 −4 b
B
p

> 0 and Φk+1 ≥0. Therefore, summing up the above inequality for
k = 0, 1, . . . , K and rearranging the terms, we get

1
K + 1

K
X

k=0
E
h∇f
 
xk2i
≤
2

γ

1 −4 b
B
p

(K + 1)

K
X

k=0

 
E

Φk
−E

Φk+1

+ 4 bDζ2

p −4 bB

=2
 
E

Φ0
−E

Φk+1

γ

1 −4 b
B
p

(K + 1)
+ 4 bDζ2

p −4 bB

≤
2E

Φ0

γ

1 −4 b
B
p

(K + 1)
+ 4 bDζ2

p −4 bB
.

Theorem D.15. Let Assumptions 2.3,
D.1,
D.2,
D.3,
D.5,
2.7 hold.
Set λk+1
=
maxi∈G Li
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

2A
,
8 bB < p

43


---Page Break---
where

A = 4

p

 
80

p

pGPGk
Cn

C
ω + 24
GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 160

p pG
GPGk
C
(1 −δ)C cδω

!

L2

+ 4

p

8

p

pGPGk
Cn

C
(10ω + 1) + 16

p pG
GPGk
C
(1 −δ)C cδ(10ω + 1)

L2
±

+ 4

p

160

p pG
GPGk
C
(1 −δ)C (1 + ω)cδ + 80

p pGPGk
C(1 + ω) n

C

 L2
±
b

+ 4

p

4

p(1 −pG)F 2
Aα2
λk+1


,

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

and where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n,
and

PGk
C =
C
npG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n
C

−1!

,

pG = Prob

Gk
C ≥(1 −δ) C
	

=
X

⌈(1−δ)C⌉≤t≤C

 
G
t

 
n −G
C −t

 
n −1
C −1

−1!

.

Then for all K ≥0 the iterates produced by Byz-VR-MARINA (Algorithm 1) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 4 bDγζ2

pρ
,

where ρ = min
h
γµ

1 −8 b
B
p

, p

8
i
and Φ0 = f
 
x0
−f ∗+ 4γ

p
g0 −∇f
 
x02.

Proof. For all k ≥0 we introduce Φk = f
 
xk
−f ∗+ 4γ

p
gk −∇f
 
xk2. Using the results of
Lemmas D.13 and D.7, we derive

E

Φk+1 (D.7)
≤
E

f
 
xk
−f ∗−
 1

2γ −L

2

 xk+1 −xk2 + γ

2

gk −∇f
 
xk2

−γ

2 E
h∇f
 
xk2i
+ 4γ

p E
hgk+1 −∇f
 
xk+12i

(D.13)
≤
E

f
 
xk
−f ∗−
 1

2γ −L

2

 xk+1 −xk2 + γ

2

gk −∇f
 
xk2

−γ

2 E
h∇f
 
xk2i
+ 4γ

p


1 −p

4


E
hgk −∇f
 
xk2i

+ 4γ

p


bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2


= E

f
 
xk
−f ∗
+ 4γ

p


1 −p

4


+ p

8


E
hgk −∇f
 
xk2i
+ 4 bDζ2γ

p

+ 1

2γ
 
1 −Lγ −2Aγ2
E

∥xk+1 −xk∥2
−γ

2

 

1 −8 bB

p

!

E
h∇f
 
xk2i
.

44


---Page Break---
Using Assumption 2.7 we obtain

E

Φk+1
≤E

f
 
xk
−f ∗
+

1 −p

8

 4γ

p E
hgk −∇f
 
xk2i
+ 4 bDζ2γ

p

+ 1

2γ
 
1 −Lγ −2Aγ2
E

∥xk+1 −xk∥2

−γµ

 

1 −8 bB

p

!

E

f
 
xk
−f ∗
.

Finally, we have

E

Φk+1
≤

 

1 −min

"

γµ

 

1 −
bB
p

!

, p

8

#!

E

Φk
+ 4 bDζ2γ

p
.

Unrolling the recurrence with ρ = min
h
γµ

1 −8 b
B
p

, p

8
i
, we obtain

E

Φk
≤(1 −ρ)K E

Φ0
+ 4 bDζ2γ

p

K−1
X

k=0
(1 −ρ)k

≤(1 −ρ)K E

Φ0
+ 4 bDζ2γ

p

∞
X

k=0
(1 −ρ)k

= (1 −ρ)K E

Φ0
+ 4 bDγζ2

pρ

Taking into account Φk ≥f
 
xk
−f (x∗), we get the result.

45


---Page Break---
E
Analysis for Bounded Compressors

E.1
Technical Lemmas

Lemma E.1. Let Assumptions D.1, D.2, D.3 and 2.4 hold and the Compression Operator satisfy
Definition 2.2. We set λk+1 = DQ maxi,j Li,j. Let us define ”ideal” estimator:

gk+1 =

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

1
Gk
b
C

P

i∈Gk
b
C
∇fi(xk+1),
cn = 1,
[1]

gk + ∇f
 
xk+1
−∇f
 
xk
,
cn = 0 and Gk
C < (1 −δ)C,
[2]

gk +
1
Gk
C
P

i∈Gk
C
clipλ

Q

b∆i
 
xk+1, xk
,
cn = 0 and Gk
C ≥(1 −δ)C.
[3]

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

A1 = E
hgk+1 −∇f
 
xk+12i

≤(1 −p)E
hgk −∇f(xk)
2i
+ p
δPGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG
PGk
CG

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

where pG = Prob

Gk
C ≥(1 −δ)C
	
and PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

Proof. Similarly to general analysis, we start from conditional expectations:

A1 = E
hgk+1 −∇f
 
xk+12i

= E
h
Ek
hgk+1 −∇f
 
xk+12ii

= (1 −p) pGE



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∇f
 
xk+1


2

| [3]





+ (1 −p)(1 −pG)E
h
Ek
hgk −∇f(xk)
2i
| [2]
i

+ pE







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

.
(26)

Using (12) and ∇f
 
xk
−∇f
 
xk
= 0 we obtain

B1 = E



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−∇f
 
xk+1


2

| [3]





= E



Ek






gk +
1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−∇f

xk+1
+ ∇f

xk
−∇f

xk


2

| [3]





46


---Page Break---
Using λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥we can guarantee that clipping operator becomes
identical since we have
Q

b∆i
 
xk+1, xk ≤DQ
b∆i
 
xk+1, xk

≤DQ



1
b

X

j∈m
∇fi,j(xk+1) −∇fi,j(xk)



≤DQ
1
b

X

j∈m

∇fi,j(xk+1) −∇fi,j(xk)


≤DQ max
j
Li,j
xk+1 −xk

≤DQ max
i,j Li,j
xk+1 −xk .

Therefore, we can continue as follows

B1 = E



Ek






gk +
1
Gk
C

X

i∈Gk
C

Q

b∆i
 
xk+1, xk
−∇f
 
xk+1


2

| [3]





= E



Ek






gk +
1
Gk
C

X

i∈Gk
C

Q

b∆i

xk+1, xk
−∇f

xk+1
+ ∇f

xk
−∇f

xk


2

| [3]



.

Moreover, we can avoid application of Young’s inequality and use variance decomposition instead:

B1 ≤E
hgk −∇f
 
xk2i

+ E



Ek







1
Gk
C

X

i∈Gk
C

Q

b∆i

xk+1, xk
−

∇f(xk+1) −∇f(xk)



2

| [3]





≤E
hgk −∇f(xk)
2i

+ E



Ek







1
Gk
C

X

i∈Gk
C

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]



.
(27)

Let us consider the last part of the inequality. Note that Gk
C ≥(1 −δ)C in this case and

B′
1 = E



Ek







1
Gk
C

X

i∈Gk
C

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]





= E



ESk



Ek







1
Gk
C

X

i∈Gk
C

Q

b∆i
 
xk+1, xk
−∆
 
xk+1, xk


2

| [3]









≤
1
C2(1 −δ)2 E



ESk



X

i∈Gk
C

Ek

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]









≤
1
C2(1 −δ)2 E

"X

i∈G
ESk
h
IGk
C

i
Ek

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

=
1
C2(1 −δ)2 E

"X

i∈G
PGk
C · Ek

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

,
(28)

where IGk
C is an indicator function for the event

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
and PGk
C
=

Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
is probability of such event. Note that ESk
h
IGk
C

i
= PGk
C. In the

47


---Page Break---
case of uniform sampling of clients, we have

∀i ∈G
PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	

= C

n
1
pG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n −1
C −1

−1!

.

Now, we can continue with inequalities:

B′
1 ≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek


EQ

Q

b∆i

xk+1, xk
−∆

xk+1, xk
2
| [3]

#

≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek


EQ

Q

b∆i

xk+1, xk
−∆i

xk+1, xk
2
| [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

.

Using variance decomposition, we have

B′
1 ≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek


EQ

Q

b∆i

xk+1, xk
2
−
X

i∈G

∆i

xk+1, xk
2
| [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G
Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2i
| [3]

#

.

Applying the definition of unbiased compressor, we get

B′
1 ≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
(1 + ω)Ek
b∆i
 
xk+1, xk
2
−
X

i∈G

∆i
 
xk+1, xk2 | [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

≤
PGk
C
C2(1 −δ)2 E

"X

i∈G
(1 + ω)Ek
b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G
(1 + ω)Ek
∆i
 
xk+1, xk2 −
X

i∈G
Ek
∆i
 
xk+1, xk2 | [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

.

48


---Page Break---
Next, we rearrange terms and derive

B′
1 ≤
PGk
C
C2(1 −δ)2 (1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
PGk
C
C2(1 −δ)2 ωE

"X

i∈G

∆i
 
xk+1, xk2 | [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

=
PGk
C
C2(1 −δ)2 (1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
PGk
C
C2(1 −δ)2 ωE

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 + ∥∆
 
xk+1, xk
∥2 | [3]

#

+
PGk
C
C2(1 −δ)2 E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

.

Rearranging terms leads to

B′
1 ≤
PGk
C
C2(1 −δ)2 (1 + ω)E

"X

i∈G
Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]

#

+
PGk
C
C2(1 −δ)2 (ω + 1)E

"X

i∈G

∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]

#

+
PGk
C
C2(1 −δ)2 ωE

"X

i∈G

∆
 
xk+1, xk2 | [3]

#

.

Now we apply Assumptions D.1, D.2, D.3:

B′
1 ≤
PGk
C
C2(1 −δ)2 (1 + ω)E

GL2
±
b ∥xk+1 −xk∥2


+
PGk
C
C2(1 −δ)2 (ω + 1)E

GL2
±∥xk+1 −xk∥2
+
PGk
C
C2(1 −δ)2 ωE
h
GL2 xk+1 −xk2i
.

Finally, we have

B′
1 ≤
PGk
C · G

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

Let us plug the obtained results in (27):

B1 ≤E
hgk −∇f(xk)
2i

+
PGk
C · G

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

49


---Page Break---
Also, we have

A1 = E
hgk+1 −∇f(xk+1)
2i

(26),(25)
≤
(1 −p)pGB1 + (1 −p)(1 −pG)E
hgk −∇f(xk)
2i

+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

≤(1 −p)pGE
hgk −∇f(xk)
2i
+ p
δ · PGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG
PGk
C · G

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E
h
∥xk+1 −xk∥2i

+ (1 −p)(1 −pG)E
hgk −∇f(xk)
2i
.

Rearranging the terms, we get

A1 ≤(1 −p)E
hgk −∇f(xk)
2i
+
δPGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+ (1 −p)pG
PGk
CG

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E

∥xk+1 −xk∥2
.

Lemma E.2. Let Assumptions D.1, D.2, D.3, D.4, 2.4 hold and the compression operator satisfy
Definition 2.2. Also, let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤4
GPGk
C
C(1 −δ)cδ

(1 + ω)L2
±
b
+ (ω + 1)L2
± + ωL2

E

∥xk+1 −xk∥2
,

where PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

Proof. By definition of the robust aggregation, we have

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤E




cδ
D2

X

i,l∈Gk
C
i̸=l

Ek

clipλ

Q

b∆i

xk+1, xk
−clipλ

Q

b∆l

xk+1, xk
2 | [3]




,

where D2 = Gk
C(Gk
C −1).

50


---Page Break---
Using λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥we can guarantee that clipping operator becomes
identical since we have ∀i ∈G

Q

b∆i
 
xk+1, xk ≤DQ
b∆i
 
xk+1, xk

≤DQ



1
b

X

j∈m
∇fi,j(xk+1) −∇fi,j(xk)



≤DQ
1
b

X

j∈m

∇fi,j(xk+1) −∇fi,j(xk)


≤DQ max
j
Li,j
xk+1 −xk .
(29)

Let us consider pair-wise differences: ∀i, l ∈G

T ′
2(i, l) = Ek

clipλ

Q

b∆i
 
xk+1, xk
−clipλ

Q

b∆l
 
xk+1, xk
2
| [3]


= Ek

Q

b∆i
 
xk+1, xk
−Q

b∆l
 
xk+1, xk
2
| [3]


= Ek

Q

b
∆i

xk+1, xk
−∆i

xk+1, xk
+ ∆l

xk+1, xk
−Q

b
∆l

xk+1, xk
2 | [3]


+ Ek
h∆i
 
xk+1, xk
−∆l
 
xk+1, xk2 | [3]
i

(12)
≤2Ek

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]


+ 2Ek

∆l
 
xk+1, xk
−Q

b∆l
 
xk+1, xk
2
| [3]


+ Ek
h∆l
 
xk+1, xk
−∆i
 
xk+1, xk2 | [3]
i
]

(12)
≤2Ek

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]


+ 2Ek

∆l
 
xk+1, xk
−Q

b∆l
 
xk+1, xk
2
| [3]


+ 2Ek
h∆l
 
xk+1, xk
−∆
 
xk+1, xk2 +
∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i
.

51


---Page Break---
Now we can combine all the parts together:

bT2 = E




1
Gk
C(Gk
C −1)

X

i,l∈Gk
C
i̸=l

T ′
2(i, l)





≤E




1
D2

X

i,l∈Gk
C
i̸=l

2Ek

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]





+ E




1
D2

X

i,l∈Gk
C
i̸=l

2Ek

∆l
 
xk+1, xk
−Q

b∆l
 
xk+1, xk
2
| [3]





+ E




1
D2

X

i,l∈Gk
C
i̸=l

2Ek
h∆l
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i





+ E




1
D2

X

i,l∈Gk
C
i̸=l

2Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i



.

Rearranging the terms, we get

bT2 ≤E



4

Gk
C

X

i∈Gk
C

Ek

Q

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



4

Gk
C

X

i∈Gk
C

Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Using variance decomposition, we get

bT2 ≤E



1

Gk
C

X

i∈Gk
C

4Ek

Q

b∆i
 
xk+1, xk
2
| [3]




−E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

52


---Page Break---
Using the properties of unbiased compressors, we obtain

bT2 ≤E



1

Gk
C

X

i∈Gk
C

4(1 + ω)Ek

b∆i
 
xk+1, xk
2
| [3]




−E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




≤E



1

Gk
C

X

i∈Gk
C

4(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

4(1 + ω)Ek
h∆i
 
xk+1, xk2 | [3]
i




−E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Let us simplify the inequality:

bT2 ≤E



1

Gk
C

X

i∈Gk
C

4(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

4ωEk
h∆i
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i


.

Using variance decomposition once again, we get

bT2 ≤E



1

Gk
C

X

i∈Gk
C

4(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]




+ E



1

Gk
C

X

i∈Gk
C

4ωEk
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i




+ E



1

Gk
C

X

i∈Gk
C

4ωEk
h∆
 
xk+1, xk2 | [3]
i


.

53


---Page Break---
Then, we apply similar arguments to the ones used in deriving (28):

bT2 ≤E

"
PGk
C
C(1 −δ)

X

i∈G
4(1 + ω)Ek

b∆i
 
xk+1, xk
−∆i
 
xk+1, xk
2
| [3]
#

+ E

"
PGk
C
C(1 −δ)

X

i∈G
4ωEk
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i#

+ E

"
PGk
C
C(1 −δ)

X

i∈G
4Ek
h∆i
 
xk+1, xk
−∆
 
xk+1, xk2 | [3]
i#

+ E

"
PGk
C
C(1 −δ)

X

i∈G
4ωEk
h∆
 
xk+1, xk2 | [3]
i#

.

Using Assumptions D.1, D.2, D.3:

bT2 ≤E

4(1 + ω)
GPGk
C
C(1 −δ)
L2
±
b ∥xk+1 −xk∥2

+ E

4(ω + 1)
GPGk
C
C(1 −δ)ωL2
±∥xk+1 −xk∥2


+ E

4
GPGk
C
C(1 −δ)ωL2∥xk+1 −xk∥2

.

Finally, we obtain

T2 = E



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i
 
xk+1, xk
−ARAggk+1
Q



2

| [3]









≤4
GPGk
C
C(1 −δ)cδ

(1 + ω)L2
±
b
+ (ω + 1)L2
± + ωL2

E

∥xk+1 −xk∥2
.

Lemma E.3. Let Assumptions 2.3, D.1, D.2, D.3, D.4, D.5, 2.4 hold and the compression operator
satisfy Definition 2.2. We set λk+1 = DQ maxi,j Li,j∥xk+1−xk∥. Also, let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP (Algorithm 1) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

2


E
hgk −∇f
 
xk2i

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

with

A = 4

p

 
pGPGk
CG

C2(1 −δ)2 ω +
8GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 8

ppG
GPGk
C
C(1 −δ)cδω

!

L2

+ 4

p

 pGPGk
CG

C2(1 −δ)2 (ω + 1) + 8

ppG
GPGk
C
C(1 −δ)cδ(ω + 1)
 
L2
± + L2
±
b



+ 16

p2 (1 −pG)F 2
A


DQ max
i,j Li,j

2

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n,

pG = Prob

Gk
C ≥(1 −δ)C
	
and PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
.

54


---Page Break---
Proof. Let us combine bounds for A1 and A2 together:

A0 = E
hgk+1 −∇f
 
xk+12i

≤

1 + p

2


E
hgk+1 −∇f
 
xk+12i
+

1 + 2

p


E
hgk+1 −gk+12i

≤

1 + p

2


A1 +

1 + 2

p


A2

≤

1 + p

2


(1 −p)E
hgk −∇f(xk)
2i
+

1 + p

2


p
δPGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+

1 + p

2


(1 −p)pG
PGk
C G

C2(1 −δ)2

 

ωL2 + (ω + 1)L2
± +
(ω + 1)L2
±
b

!

E
h
∥xk+1 −xk∥2i

+

1 + 2

p


pE
h
Ek
hARAgg
 
∇f1(xk+1), . . . , ∇fn(xk+1)

−∇f(xk+1)
2i
| [1]
i

+

1 + 2

p


(1 −p)pGE



Ek







1
Gk
C

X

i∈Gk
C

clipλ

Q

b∆i

xk+1, xk
−ARAggk+1
Q



2

| [3]









+

1 + 2

p


(1 −p)(1 −pG)E

Ek

∇f(xk+1) −∇f(xk) −ARAggk+1
Q

2
| [2]

.

Using Lemma E.2 and lemmas from General Analysis (Lemmas D.10 and D.12) we have

A0 = E
hgk+1 −∇f
 
xk+12i

≤

1 −p

2


E
hgk −∇f
 
xk2i
+ 2p
δPGk
b
C
(1 −δ)E

B∥∇f(x)∥2 + ζ2

+

1 −p

2


pG
PGk
CG

C2(1 −δ)2


ωL2 + (ω + 1)L2
± + (ω + 1)L2
±
b


E

∥xk+1 −xk∥2

+ (p + 2)

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i

+ 4 (p + 2)
GPGk
b
Ccδ

(1 −δ) bC
ζ2 + (p + 2)eζ2

+ 2

ppGE

4(1 + ω)
GPGk
C
C(1 −δ)cδ L2
±
b ∥xk+1 −xk∥2


+ 2

ppGE

4(ω + 1)
GPGk
C
C(1 −δ)cδL2
±∥xk+1 −xk∥2


+ 2

ppGE

4ω
GPGk
C
C(1 −δ)cδL2∥xk+1 −xk∥2

+ 2

p(1 −pG)2(L2 + F 2
Aα2
λk+1)E
hxk+1 −xk2i
.

Finally, we have

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

2


E
hgk −∇f
 
xk2i

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

where

A = 4

p

 
pGPGk
CG

C2(1 −δ)2 ω +
8GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 8

ppG
GPGk
C
C(1 −δ)cδω

!

L2

+ 4

p

 pGPGk
CG

C2(1 −δ)2 (ω + 1) + 8

ppG
GPGk
C
C(1 −δ)cδ(ω + 1)
 
L2
± + L2
±
b



+ 16

p2 (1 −pG)F 2
A


DQ max
i,j Li,j

2

55


---Page Break---
and

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD.

E.2
Main Results

Theorem E.4. Let Assumptions 2.3, D.1, D.2, D.3, D.4, D.5, 2.4 hold.
Setting λk+1 =
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where

A = 4

p

 
pGPGk
CG

C2(1 −δ)2 ω +
8GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 8

ppG
GPGk
C
C(1 −δ)cδω

!

L2

+ 4

p

 pGPGk
CG

C2(1 −δ)2 (ω + 1) + 8

ppG
GPGk
C
C(1 −δ)cδ(ω + 1)
 
L2
± + L2
±
b



+ 16

p2 (1 −pG)F 2
A


DQ max
i,j Li,j

2

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n, and

PGk
C =
C
npG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n
C

−1!

,

pG = Prob

Gk
C ≥(1 −δ) C
	

=
X

⌈(1−δ)C⌉≤t≤C

 
G
t

 
n −G
C −t

 
n
C

−1!

.

Then for all K ≥0 the iterates produced by Byz-VR-MARINA (Algorithm 1) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 2 bDζ2

p −4 bB
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

γ
p
g0 −∇f
 
x02 .

Proof. The proof is analogous to the proof of Theorem D.14.

Theorem E.5. Let Assumptions 2.3, 2.4, D.1, D.2, D.3, D.4, D.5, 2.7 hold. Setting λk+1 =
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

2A
,
8 bB < p,

where

A = 4

p

 
pGPGk
CG

C2(1 −δ)2 ω +
8GPGk
b
Ccδ

(1 −δ) bC
B + 6 eB + 4

p(1 −pG) + 8

ppG
GPGk
C
C(1 −δ)cδω

!

L2

+ 4

p

 pGPGk
CG

C2(1 −δ)2 (ω + 1) + 8

ppG
GPGk
C
C(1 −δ)cδ(ω + 1)
 
L2
± + L2
±
b



+ 16

p2 (1 −pG)F 2
A


DQ max
i,j Li,j

2

56


---Page Break---
bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB,
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD,

where eB := 0 and eD := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eD :=
PGk
b
C
G

(1−δ) b
C when bC < n, and

where pG = Prob

Gk
C ≥(1 −δ)C
	
and PGk
C = Prob

i ∈Gk
C | Gk
C ≥(1 −δ) C
	
. Then for all
K ≥0 the iterates produced by Byz-VR-MARINA (Algorithm 1) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 2 bDζ2

pρ ,

where ρ = min
h
γµ

1 −8 b
B
p

, p

4
i
and Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.

Proof. The proof is analogous to the proof of Theorem D.15.

E.3
On the Technical Non-Triviality of the Analysis

As we explain in the main part of the paper, the main reason why we propose to use clipping
is to handle the situations when Byzantine workers form a majority during some communication
rounds since the existing approaches are vulnerable to such scenarios. However, the introduction
of the clipping does not come for free: if the clipping level is too small, clipping can create a
noticeable bias to the updates. Because of this issue, existing works such as (Zhang et al., 2020b;
Gorbunov et al., 2020) use non-trivial policies for the choice of the clipping level, and the analysis in
these works differs significantly from the existing analysis for the methods without clipping. The
analysis of Byz-VR-MARINA is based on the unbiasedness of vectors Q( ˆ∆i(xk+1, xk)), i.e., on
the following identity: E[Q( ˆ∆i(xk+1, xk)) | xk+1, xk] = ∆i(xk+1, xk) = ∇fi(xk+1) −∇fi(xk).
Since E[clipλk+1(Q( ˆ∆i(xk+1, xk))) | xk+1, xk] ̸= ∇fi(xk+1) −∇fi(xk) in general, to analyze
Byz-VR-MARINA-PP we also use a special choice of the clipping level: λk+1 = αk+1∥xk+1 −xk∥.
To illustrate the main reasons for that, let us consider the case of uncompressed communication
(Q(x) ≡x). In this setup, for large enough αk+1 we have clipλk+1 ˆ∆i(xk+1, xk) = ˆ∆i(xk+1, xk)
for all i ∈G (due to Assumption 2.6), which allows us using a similar proof to the one for Byz-VR-
MARINA when good workers form a majority in a round. Moreover, when Byzantine workers form
a majority, our choice of the clipping level allows us to bound the second moment of the shift from
the Byzantine workers as ∼∥xk+1 −xk∥2 (see Lemmas D.9 and D.12), i.e., the second moment
of the shift is of the same scale as the variance of {gi}i∈G, which goes to zero. Next, to properly
analyze these two situations, we overcame another technical challenge related to the estimation of the
conditional expectations and probabilities of corresponding events (see Lemmas D.9 and D.10 and
formulas for pG and PGk
C at the beginning of Section 4). In particular, the derivation of formula (24)
is quite non-standard for stochastic optimization literature: there are two sources of stochasticity –
one comes from the sampling of clients, and the other one comes from the sampling of stochastic
gradients and compression. This leads to the estimation of variance of the average of the random
number of random vectors, which is novel on its own. In addition, when the compression operator is
used, the analysis becomes even more involved since one cannot directly apply the main property
of unbiased compression (Definition 2.2), and we use Lemma D.6 in the proof to address this issue.
It is also worth mentioning that in contrast to Byz-VR-MARINA, our method does not require full
participation even with a small probability p. Instead, it is sufficient for Byz-VR-MARINA-PP to
sample a large enough cohort of bC clients with probability p to ensure that Byzantine workers form a
minority in such rounds.

57


---Page Break---
F
Byz-VR-MARINA-PP+: Simplified Version of Byz-VR-MARINA-PP

In this section, we present a simplified version of Byz-VR-MARINA-PP called Byz-VR-MARINA-
PP+ (see Algorithm 3). The only difference between the two methods is in Line 10: Byz-VR-
MARINA+ does not apply robust aggregation when ck = 0 and just averages the clipped vectors
received from the set of clients Sk. Nevertheless, when ck = 1, i.e., a large cohort of clients is
sampled, the method still uses robust aggregation.

Algorithm 3 Byz-VR-MARINA-PP+: Simplified Byz-VR-MARINA-PP

1: Input: vectors x0, g0 ∈Rd, stepsize γ, mini-batch size b, probability p ∈(0, 1], number of
iterations K, (δ, c)-ARAgg, clients’ sample size 1 ≤C ≤bC ≤n, clipping coefficients {αk}k≥1
2: for k = 0, 1, . . . , K −1 do
3:
Get a sample from Bernoulli distribution with parameter p: ck ∼Be(p)
4:
Sample the set of clients Sk ⊆[n], |Sk| = C if ck = 0; otherwise |Sk| = bC
5:
Broadcast gk, ck to all workers
6:
for i ∈G ∩Sk in parallel do
7:
xk+1 = xk −γgk and λk+1 = αk+1∥xk+1 −xk∥

8:
Set gk+1
i
=

(
∇fi(xk+1),
if ck = 1,

gk + clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise,

where b∆i(xk+1, xk) is a mini-batched estimator of ∇fi(xk+1) −∇fi(xk), Q(·) for i ∈
G ∩Sk are computed independently
9:
end for

10:
gk+1 =






ARAgg
 
{gk+1
i
}i∈Sk

,
if ck = 1,

gk + 1

C
P

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise

11: end for

The key idea behind this modification can be explained as follows. For simplicity, let us assume that
C is small and δreal is also small. Then, for the communication rounds with ck = 0, with a large
probability, only good clients will be sampled. In this case, the method can use just an average of the
received vectors and benefit from the lack of bias appearing due to the robust aggregation. Moreover,
when ck = 0 and at least one of the sampled clients is Byzantine, the method will tolerate due to
the clipping. That is, when C is small, the method can potentially benefit from the lack of robust
aggregation when ck = 0. However, for the rounds with ck = 1, in the worst case, bC = n, meaning
that all Byzantines workers are guaranteed to be sampled. To tolerate such situations, we keep the
robust aggregation in the method when ck = 1.

F.1
Analysis for Bounded Compressors

For simplicity, we analyze Byz-VR-MARINA-PP+ for bounded compressors only. The analysis
is very similar to the one we provide for Byz-VR-MARINA-PP, but several steps are significantly
simpler. In particular, the central part in the analysis of Byz-VR-MARINA-PP is in deriving a
good recursive inequality for E[∥gk −∇f(xk)∥2], which requires several quite technical steps. For
Byz-VR-MARINA-PP+, one can obtain a similar inequality much easier as shown in the next lemma.

Lemma F.1. Let Assumptions D.1, D.2, D.3, D.4, D.5, 2.4 hold and the compression operator satisfy
Definition 2.2. Assume that C ≤G. We set λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥. Then for all k ≥0
the iterates produced by Byz-VR-MARINA-PP+ (Algorithm 3) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

2


E
hgk −∇f
 
xk2i
(30)

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

58


---Page Break---
with

A = 4

p

 8pBGPGk
b
Ccδ

(1 −δ) bC
+ 6p eB + (1 −p)pk
Gω
C
+ 6(1 −p)(1 −pk
G)
p

!

L2

+ 4(1 −p)pk
G
p


1 + ω

C


L2
± + 4(1 −p)pk
G(1 + ω)
pC
L2
±
b

+ 24(1 −p)(1 −pk
G)
p2


DQ max
i,j Li,j

2
,

bB =
8GPGk
b
CcδBp

(1 −δ) bC
+ 6p eB,
bD =
4GPGk
b
Ccδp

(1 −δ) bC
+ p eD,

where eB, eD, pk
G, PGk
b
C are defined in Lemma D.13.

Proof. From the update rule of gk+1, we have

E
hgk+1 −∇f(xk+1)
2i
= p E
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2 | ck = 1
i

|
{z
}
T1

(31)

+ (1 −p) E





gk + 1

C

X

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)

−∇f(xk+1)



2

| ck = 0





|
{z
}
T2

.

Next, we bound T1 and T2 separately. From Lemma D.10, we have

T1 ≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2.

As for T2, we consider two possible situations: either Sk ∩B = ∅(no Byzantine workers are among
sampled ones) or Sk ∩B ̸= ∅(at least one Byzantine worker is sampled). Then, T2 equals

T2 = pk
G E





gk + 1

C

X

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)

−∇f(xk+1)



2

| ck = 0, Sk ∩B = ∅





|
{z
}
b
T2

+ (1 −pk
G) E





gk + 1

C

X

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)

−∇f(xk+1)



2

| ck = 0, Sk ∩B ̸= ∅





|
{z
}
e
T2

,

where

pk
G := Prob{Sk ∩B = ∅| ck = 0} =

 G
C

 n
C
 = (G −C + 1)(G −C + 2) · . . . · (n −C)

(G + 1)(G + 2) · . . . · n
.

The choice of the clipping level λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥and inequality (29) imply that

clipλk+1

Q

b∆i(xk+1, xk)

for all i ∈G. Therefore, for bT2, we have

bT2 = E





gk + 1

C

X

i∈Sk
Q

b∆i(xk+1, xk)

−∇f(xk+1)



2

| ck = 0, Sk ∩B = ∅





= E

∥gk −∇f(xk)∥2

+ E






1
C

X

i∈Sk
Q

b∆i(xk+1, xk)

−(∇f(xk+1) −∇f(xk))



2

| ck = 0, Sk ∩B = ∅



,

59


---Page Break---
where we use that E
h
1
C
P

i∈Sk Q

b∆i(xk+1, xk)

| ck = 0, Sk ∩B = ∅
i
=
∇f(xk+1) −

∇f(xk).
Moreover,
since
E
h
1
C
P

i∈Sk Q

b∆i(xk+1, xk)

| ck = 0, Sk ∩B = ∅, Sk
i
=

1
C
P

i∈Sk(∇fi(xk+1) −∇fi(xk)) =:
1
C
P

i∈Sk ∆i(xk+1, xk), we can decompose the last term
in the upper-bound for bT2 as follows:

bT2 = E

∥gk −∇f(xk)∥2

+ E



E






1
C

X

i∈Sk


Q

b∆i(xk+1, xk)

−∆i(xk+1, xk)


2

| Sk



| ck = 0, Sk ∩B = ∅





+ E






1
C

X

i∈Sk
∆i(xk+1, xk) −(∇f(xk+1) −∇f(xk))



2

| ck = 0, Sk ∩B = ∅



.

Since the compression operator computations are independent on each client, we have

bT2 = E

∥gk −∇f(xk)∥2

+ 1

C2 E

"

E

" X

i∈Sk

Q

b∆i(xk+1, xk)

−∆i(xk+1, xk)

2
| Sk

#

| ck = 0, Sk ∩B = ∅

#

+ E






1
C

X

i∈Sk
∆i(xk+1, xk)



2

| ck = 0, Sk ∩B = ∅



−E
h∇f(xk+1) −∇f(xk)
2i

≤E

∥gk −∇f(xk)∥2

+ 1

C2 E

"

E

" X

i∈Sk

Q

b∆i(xk+1, xk)

−b∆i(xk+1, xk)

2
| Sk

#

| ck = 0, Sk ∩B = ∅

#

+ 1

C2 E

"

E

" X

i∈Sk

b∆i(xk+1, xk) −∆i(xk+1, xk)

2
| Sk

#

| ck = 0, Sk ∩B = ∅

#

+ E






1
C

X

i∈Sk
∆i(xk+1, xk)



2

| ck = 0, Sk ∩B = ∅



−E
h∇f(xk+1) −∇f(xk)
2i

(Def. 2.2)
≤
E

∥gk −∇f(xk)∥2

+ ω

C2 E

" X

i∈Sk

b∆i(xk+1, xk)

2
| ck = 0, Sk ∩B = ∅

#

+ 1

C2 E

" X

i∈Sk

b∆i(xk+1, xk) −∆i(xk+1, xk)

2
| ck = 0, Sk ∩B = ∅

#

+ 1

C E

" X

i∈Sk

∆i(xk+1, xk)
2 | ck = 0, Sk ∩B = ∅

#

−E
h∇f(xk+1) −∇f(xk)
2i

= E

∥gk −∇f(xk)∥2
+ ω

CG

X

i∈G
E
b∆i(xk+1, xk)

2

+
1
CG

X

i∈G
E
b∆i(xk+1, xk) −∆i(xk+1, xk)

2

+ 1

G

X

i∈G
E
h∆i(xk+1, xk)
2i
−E
h∇f(xk+1) −∇f(xk)
2i
.

60


---Page Break---
Using E
b∆i(xk+1, xk)

2
= E
b∆i(xk+1, xk) −∆i(xk+1, xk)

2
+ E
h∆i(xk+1, xk)
2i
,

we continue the derivation as follows:

bT2 ≤E

∥gk −∇f(xk)∥2
+ 1 + ω

CG

X

i∈G
E
b∆i(xk+1, xk) −∆i(xk+1, xk)

2

+

1 + ω

C

 1

G

X

i∈G
E
h∆i(xk+1, xk)
2i
−E
h∇f(xk+1) −∇f(xk)
2i

(21)
≤E

∥gk −∇f(xk)∥2
+ (1 + ω)L2
±
bC
E

∥xk+1 −xk∥2

+

1 + ω

C


E

"
1
G

X

i∈G

∆i(xk+1, xk)
2 −
∇f(xk+1) −∇f(xk)
2
#

+ ω

C E
h∇f(xk+1) −∇f(xk)
2i

(20),(18)
≤
E

∥gk −∇f(xk)∥2
+
 ω

C L2 +

1 + ω

C


L2
± + (1 + ω)L2
±
bC


E

∥xk+1 −xk∥2
.

Next, we estimate eT2 using Young’s inequality and the choice of the clipping level:

eT2 ≤(1 + β)E
hgk −∇f(xk)
2i
+ 2(1 + β−1)E
h∇f(xk+1) −∇f(xk)
2i

+ 2(1 + β−1)E






1
C

X

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)


2

| ck = 0, Sk ∩B ̸= ∅





(18)
≤(1 + β)E
hgk −∇f(xk)
2i
+ 2(1 + β−1)
 
L2E

∥xk+1 −xk∥2
+ E[λ2
k+1]


= (1 + β)E
hgk −∇f(xk)
2i
+ 2(1 + β−1)

L2 + D2
Q max
i,j L2
i,j


E

∥xk+1 −xk∥2
,

where β > 0 will be specified later in the proof. Combining the derived upper bounds for bT2 and eT2,
we get

T2 ≤
 
pk
G + (1 −pk
G)(1 + β)

E

∥gk −∇f(xk)∥2

+ pk
G

 ω

C L2 +

1 + ω

C


L2
± + (1 + ω)L2
±
bC


E

∥xk+1 −xk∥2

+ 2(1 −pk
G)(1 + β−1)

L2 + D2
Q max
i,j L2
i,j


E

∥xk+1 −xk∥2
.

Plugging the obtained bounds for T1 and T2 into (31), we obtain

E

∥gk+1 −∇f(xk+1)∥2
≤(1 −p)
 
pk
G + (1 −pk
G)(1 + β)

E

∥gk −∇f(xk)∥2

+ p

  8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2
!

+ (1 −p)pk
G

 ω

C L2 +

1 + ω

C


L2
± + (1 + ω)L2
±
bC


E

∥xk+1 −xk∥2

+ 2(1 −p)(1 −pk
G)(1 + β−1)

L2 + D2
Q max
i,j L2
i,j


E

∥xk+1 −xk∥2
.

Taking

β :=

(
p
2(1−pk
G),
if pk
G < 1,

1,
if pk
G = 1,

we ensure that pk
G + (1 −pk
G)(1 + β) ≤1 + p

2 and (1 −pk
G)(1 + β−1) ≤(1−pk
G)(p+2(1−pk
G))
p
≤

3(1−pk
G)
p
. Using these inequalities and (1 −p)
 
1 −p

2

≤1 −p

2, we simplify the upper bound for

61


---Page Break---
E

∥gk+1 −∇f(xk+1)∥2
as follows:

E

∥gk+1 −∇f(xk+1)∥2
≤

1 −p

2


E

∥gk −∇f(xk)∥2

+ p

  8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i
+
4GPGk
b
Ccδζ2

(1 −δ) bC
+ eζ2
!

+ (1 −p)pk
G

 ω

C L2 +

1 + ω

C


L2
± + (1 + ω)L2
±
bC


E

∥xk+1 −xk∥2

+ 6(1 −p)(1 −pk
G)
p


L2 + D2
Q max
i,j L2
i,j


E

∥xk+1 −xk∥2
.

Rearranging the terms, we get (35).

Then, similarly to the analysis of Byz-VR-MARINA, we get the following result.
Theorem F.2.
Let Assumptions D.1,
D.2,
D.3,
D.4,
D.5,
2.4 hold.
Set λk+1
=
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where

A = 4

p

 8pBGPGk
b
Ccδ

(1 −δ) bC
+ 6p eB + (1 −p)pk
Gω
C
+ 6(1 −p)(1 −pk
G)
p

!

L2

+ 4(1 −p)pk
G
p


1 + ω

C


L2
± + 4(1 −p)pk
G(1 + ω)
pC
L2
±
b

+ 24(1 −p)(1 −pk
G)
p2


DQ max
i,j Li,j

2
,

bB =
8GPGk
b
CcδBp

(1 −δ) bC
+ 6p eB,
bD =
4GPGk
b
Ccδp

(1 −δ) bC
+ p eD,

and

PGk
C =
C
npG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n
C

−1!

,

pk
G = Prob{Sk ∩B = ∅| ck = 0} = (G −C + 1)(G −C + 2) · . . . · (n −C)

(G + 1)(G + 2) · . . . · n
.

Then for all K ≥0 the iterates produced by Byz-VR-MARINA+ (Algorithm 3) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 2 bDζ2

p −4 bB
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

γ
p
g0 −∇f
 
x02 .

Proof. The proof is analogous to the proof of Theorem D.14.

Theorem F.3. Let Assumptions 2.4, D.1, D.2, D.3, D.4, D.5, 2.7 hold.
Set λk+1
=
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤min

1
L +
√

2A


,
8 bB < p,

62


---Page Break---
where

A = 4

p

 8pBGPGk
b
Ccδ

(1 −δ) bC
+ 6p eB + (1 −p)pk
Gω
C
+ 6(1 −p)(1 −pk
G)
p

!

L2

+ 4(1 −p)pk
G
p


1 + ω

C


L2
± + 4(1 −p)pk
G(1 + ω)
pC
L2
±
b

+ 24(1 −p)(1 −pk
G)
p2


DQ max
i,j Li,j

2
,

bB =
8GPGk
b
CcδBp

(1 −δ) bC
+ 6p eB,
bD =
4GPGk
b
Ccδp

(1 −δ) bC
+ p eD,

and

PGk
C =
C
npG
·
X

(1−δ)C≤t≤C

 
G −1
t −1

 
n −G
C −t

 
n
C

−1!

,

pk
G = Prob{Sk ∩B = ∅| ck = 0} = (G −C + 1)(G −C + 2) · . . . · (n −C)

(G + 1)(G + 2) · . . . · n
.

Then for all K ≥0 the iterates produced by Byz-VR-MARINA+ (Algorithm 3) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 2 bDζ2

pρ ,

where ρ = min
h
γµ

1 −8 b
B
p

, p

4
i
and Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.

Proof. The proof is analogous to the proof of Theorem D.15.

F.2
Discussion of the Results

Improved neighborhood term and bound on δ.
The key property of Byz-VR-MARINA+ is its
better neighborhood terms, and maximal allowed fraction of Byzantine workers δ in comparison
to Byz-VR-MARINA. To illustrate it, consider the non-PŁsetting (the discussion for the PŁ case is
similar). For both algorithms, the neighborhood term in the convergence bounds equals O

b
Dζ2

p−4 b
B


,

but corresponding constants bB and bD are different:

bB = 2
δPGk
b
C
1 −δ B
12cG

bC
+ p

+ 6 eB
and
bD = 2
δPGk
b
C
1 −δ

6cG

bC
+ p

+ eD
for
Byz-VR-MARINA,

bB =
8GPGk
b
CcδBp

(1 −δ) bC
+ 6p eB
and
bD =
4GPGk
b
Ccδp

(1 −δ) bC
+ p eD
for
Byz-VR-MARINA+.

For the simplicity of the comparison, consider the case of bC = n. Then, PGk
b
C = 1 and

bB = Θ (cδB)
and
bD = Θ(cδ)
for
Byz-VR-MARINA,
bB = Θ (cδBp)
and
bD = Θ(cδp)
for
Byz-VR-MARINA+,
implying that the neighborhood term for Byz-VR-MARINA+ is 1/p times smaller than the neigh-
borhood term for Byz-VR-MARINA. Moreover, the restriction 4 bB < p used in the analysis of both
methods implies

cδ = O
 1

Bp


for
Byz-VR-MARINA,

cδ = O
 1

B


for
Byz-VR-MARINA+,

i.e., the result for Byz-VR-MARINA+ allows (1/p)-times more Byzantine workers when B > 0. We
emphasize that the neighborhood term and the bound on δ in the results for Byz-VR-MARINA+
cannot be improved up to the numerical factors (Allouah et al., 2024b).

63


---Page Break---
Comparison of stepsizes when bC = n and C = 1.
For simplicity, to compare the stepsize
restrictions for Byz-VR-MARINA and Byz-VR-MARINA+, we consider the case when bC = n and
C = 1. Moreover, let us assume that b = 1 and let us ignore the differences between smoothness
constants and replace them with their upper bound L from Assumption 2.6. Then, for both methods,
the results in the non-PŁsetting (the discussion for the PŁ case is similar) with B = 0 hold for
0 < γ ≤1/L(1+
√

A), where

A = Θ

 
1
p


1 + ω + (1 + ω)cδ

p


+
δreal(1 + F 2
AD2
Q)
p2

!

for
Byz-VR-MARINA,

A = Θ

 
1 + ω

p
+
δrealD2
Q
p2

!

for
Byz-VR-MARINA+,

where we use pG = G/n = 1 −δreal, PGk
C = 1/G, pk
G = G/n = 1 −δreal. That is, the result for
Byz-VR-MARINA+ allows to use larger stepsizes than in Byz-VR-MARINA (though the methods
are equivalent when C = 1). A similar comparison holds for small enough C as well. Therefore,
we recommend using Byz-VR-MARINA+ instead of Byz-VR-MARINA when C is small. We also
highlight that the result for Byz-VR-MARINA+ does not require Assumption 2.3.

64


---Page Break---
G
Analysis without Full-Batch Gradient Computations

In this section, we consider versions of Byz-VR-MARINA-PP and Byz-VR-MARINA-PP+ that
do not use full-batch gradient computations at all – see Algorithms 4 and 5. These variants of
Byz-VR-MARINA-PP and Byz-VR-MARINA-PP+ use b′-size mini-batched estimator e∇fi(xk+1)
when ck = 1 for every i ∈G ∩Sk in line 8 and are identical to their original versions in all other
steps/computations. This modification reduces the computation cost of iterations when ck = 1,
making the methods more practical.

Algorithm 4 Byz-VR-MARINA-PP without full-batch gradient computations

1: Input: vectors x0, g0 ∈Rd, stepsize γ, mini-batch size b, mini-batch size b′, probability
p ∈(0, 1], number of iterations K, (δ, c)-ARAgg, clients’ sample size 1 ≤C ≤bC ≤n, clipping
coefficients {αk}k≥1
2: for k = 0, 1, . . . , K −1 do
3:
Get a sample from Bernoulli distribution with parameter p: ck ∼Be(p)
4:
Sample the set of clients Sk ⊆[n], |Sk| = C if ck = 0; otherwise |Sk| = bC
5:
Broadcast gk, ck to all workers
6:
for i ∈G ∩Sk in parallel do
7:
xk+1 = xk −γgk and λk+1 = αk+1∥xk+1 −xk∥

8:
Set gk+1
i
=

(e∇fi(xk+1),
if ck = 1,

gk + clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise,

where e∇fi(xk+1) is a b′-size mini-batched estimator of ∇fi(xk+1), b∆i(xk+1, xk) is a
b-size mini-batched estimator of ∇fi(xk+1) −∇fi(xk), Q(·) for i ∈G ∩Sk are computed
independently
9:
end for

10:
gk+1 =






ARAgg
 
{gk+1
i
}i∈Sk

,
if ck = 1,

gk + ARAgg
n
clipλk+1

Q

b∆i(xk+1, xk)
o

i∈Sk


,
otherwise

11: end for

Algorithm 5 Byz-VR-MARINA-PP+: without full-batch gradient computations

1: Input: vectors x0, g0 ∈Rd, stepsize γ, mini-batch size b, mini-batch size b′, probability
p ∈(0, 1], number of iterations K, (δ, c)-ARAgg, clients’ sample size 1 ≤C ≤bC ≤n, clipping
coefficients {αk}k≥1
2: for k = 0, 1, . . . , K −1 do
3:
Get a sample from Bernoulli distribution with parameter p: ck ∼Be(p)
4:
Sample the set of clients Sk ⊆[n], |Sk| = C if ck = 0; otherwise |Sk| = bC
5:
Broadcast gk, ck to all workers
6:
for i ∈G ∩Sk in parallel do
7:
xk+1 = xk −γgk and λk+1 = αk+1∥xk+1 −xk∥

8:
Set gk+1
i
=

(e∇fi(xk+1),
if ck = 1,

gk + clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise,

where e∇fi(xk+1) is a b′-size mini-batched estimator of ∇fi(xk+1), b∆i(xk+1, xk) is a
b-size mini-batched estimator of ∇fi(xk+1) −∇fi(xk), Q(·) for i ∈G ∩Sk are computed
independently
9:
end for

10:
gk+1 =






ARAgg
 
{gk+1
i
}i∈Sk

,
if ck = 1,

gk + 1

C
P

i∈Sk
clipλk+1

Q

b∆i(xk+1, xk)

,
otherwise

11: end for

However, our analysis of Byz-VR-MARINA-PP/Byz-VR-MARINA-PP+ without full-batch gradient
computations requires the following additional assumption.

65


---Page Break---
Assumption G.1. We assume that there exist σ ≥0 such that for all x ∈Rd and i ∈[n]

E
h
∥e∇fi(x) −∇fi(x)∥2i
≤σ2

b′ ,
(32)

where e∇fi(x) is an unbiased b′-size mini-batched estimator of ∇fi(x).

In particular, when 1

m
Pm
j=1 ∥∇fi,j(x)−∇fi(x)∥2 ≤σ, which is a standard assumption for variance-
reduced methods without full-batch gradient computations (Cutkosky and Orabona, 2019; Li et al.,
2021; Gorbunov et al., 2021), estimator e∇fi(x) =
1
b′
Pb′

j=1 ∇fi,ξj
i (x) with {ξj
i }i∈[n],j∈[m] being
i.i.d. samples from the uniform distribution over [m] satisfies (32). Assumption G.1 is also standard
for general stochastic optimization (Nemirovski et al., 2009; Ghadimi and Lan, 2013).

G.1
New Lemma

The main change in the analysis is related to Lemma D.10 since it is the only lemma that relies on the
full-batch gradient computation. Nevertheless, it can be easily generalized to the case of Algorithms 4
and 5, as shown in the next result.
Lemma G.2. Let Assumptions D.1, D.5, G.1 hold and Aggregation Operator (ARAgg) satisfy
Definition 2.1. Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP/Byz-VR-MARINA-
PP+ (Algorithms 4 and 5) satisfy

T1 = E
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i

+
4GPGk
b
CcδB

(1 −δ) bC
ζ2 + eζ2 +

 
PGk
b
CG

(1 −δ)2 bC2 + 4cδ

!
σ2

b′ ,

where eB := 0 and eζ2 := 0 when bC = n, and eB :=
PGk
b
C
GB

(1−δ) b
C and eζ2 :=
PGk
b
C
Gζ2

(1−δ) b
C when bC < n.

Proof. Using the definition of aggregation operator, we have

T1 = E
h
Ek
hARAgg
 
{gk+1
i
}i∈Sk

−∇f(xk+1)
2i
| [1]
i

(12)
≤E



Ek






ARAgg
 
{gk+1
i
}i∈Sk

−
1
Gk
b
C

X

i∈Gk
b
C

e∇fi(xk+1)



2

| [1]





+ E



Ek







1
Gk
b
C

X

i∈Gk
b
C

e∇fi(xk+1) −∇f(xk+1)



2

| [1]



.
(33)

To proceed, we estimate the second term in the right-hand side of the above inequality first. From
variance decomposition, we have

E



Ek







1
Gk
b
C

X

i∈Gk
b
C

e∇fi(xk+1) −∇f(xk+1)



2

| [1]





= E



Ek







1
Gk
b
C

X

i∈Gk
b
C

(e∇fi(xk+1) −∇fi(xk+1))



2

| [1]





+ E



Ek







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

| [1]



.
(34)

66


---Page Break---
The choice of bC implies that Gk
b
C ≥(1 −δ) bC. Moreover, due to the independence of stochastic
gradient computations on different workers, we have

E



Ek







1
Gk
b
C

X

i∈Gk
b
C

(e∇fi(xk+1) −∇fi(xk+1))



2

| [1]





≤
1

(1 −δ)2 bC2 E



Ek







X

i∈Gk
b
C

(e∇fi(xk+1) −∇fi(xk+1))



2

| [1]





=
1

(1 −δ)2 bC2 E




X

i∈Gk
b
C

Ek

e∇fi(xk+1) −∇fi(xk+1)

2
| [1]





=
PGk
b
C
(1 −δ)2 bC2

X

i∈G
E
e∇fi(xk+1) −∇fi(xk+1)

2 (32)
≤
PGk
b
CGσ2

(1 −δ)2 bC2b′ .

Next, since
1
Gk
b
C

P

i∈Gk
b
C
∇fi(xk+1) = ∇f(xk+1) with probability 1 when bC = n, we can estimate the

last term in (34) as

E



Ek







1
Gk
b
C

X

i∈Gk
b
C

∇fi(xk+1) −∇f(xk+1)



2

| [1]





≤










0,
if bC = n

E



1

Gk
b
C

P

i∈Gk
b
C
Ek
h∇fi(xk+1) −∇f(xk+1)
2i
| [1]



,
if bC < n

≤






0,
if bC = n
PGk
b
C
(1−δ) b
C
P

i∈G
E
h∇fi(xk+1) −∇f(xk+1)
2i
,
if bC < n

(As. D.5)
≤






0,
if bC = n
PGk
b
C
G

(1−δ) b
C
 
BE

∥∇f(xk+1)∥2
+ ζ2
,
if bC < n
= eBE

∥∇f(xk+1)∥2
+ eζ2,

where

eB :=






0,
if bC = n,
PGk
b
C
GB

(1−δ) b
C ,
if bC < n,
and
eζ2 :=






0,
if bC = n,
PGk
b
C
Gζ2

(1−δ) b
C ,
if bC < n.

Plugging the derived bounds in (34), we get

E



Ek







1
Gk
b
C

X

i∈Gk
b
C

e∇fi(xk+1) −∇f(xk+1)



2

| [1]



≤
PGk
b
CGσ2

(1 −δ)2 bC2b′

+ eBE

∥∇f(xk+1)∥2
+ eζ2.

67


---Page Break---
Using the above bound in (33), we continue the estimation of T1 as follows:

T1

(Def. 2.1)
≤
E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

Ek

e∇fi
 
xk+1
−e∇fl
 
xk+1
2
| [1]






+
PGk
b
CGσ2

(1 −δ)2 bC2b′ + eBE

∥∇f(xk+1)∥2
+ eζ2

= E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

Ek
h∇fi
 
xk+1
−∇fl
 
xk+12 | [1]
i





+ E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

Ek

e∇fi
 
xk+1
−∇fi(xk+1) −e∇fl
 
xk+1
+ ∇fl
 
xk+1
2
| [1]






+
PGk
b
CGσ2

(1 −δ)2 bC2b′ + eBE

∥∇f(xk+1)∥2
+ eζ2,

68


---Page Break---
where in the last inequality, we use the conditional independence of {e∇fi(xk+1)}i ∈Gk
b
C for fixed

xk+1. Next, using Young’s inequality, we derive

T1

(12)
≤E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E
h
2
∇fi
 
xk+1
−∇f
 
xk+12 | [1]
i





+ E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E
h
2
∇fl
 
xk+1
−∇f
 
xk+12 | [1]
i





+ E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E

2
e∇fi
 
xk+1
−∇fi
 
xk+1
2
| [1]






+ E




cδ
Gk
b
C(Gk
b
C −1)

X

i,l∈Gk
b
C
i̸=l

E

2
e∇fl
 
xk+1
−∇fl
 
xk+1
2
| [1]






+
PGk
b
CGσ2

(1 −δ)2 bC2b′ + eBE

∥∇f(xk+1)∥2
+ eζ2

(32)
≤E



cδ

Gk
b
C

X

i∈Gk
b
C

4Ek
h∇fi
 
xk+1
−∇f
 
xk+12 | [1]
i


+ 4cδσ2

b′

+
PGk
b
CGσ2

(1 −δ)2 bC2b′ + eBE

∥∇f(xk+1)∥2
+ eζ2

≤
PGk
b
Ccδ

(1 −δ) bC

X

i∈G
4Ek
h∇fi
 
xk+1
−∇f
 
xk+12i
+ 4cδσ2

b′

+
PGk
b
CGσ2

(1 −δ)2 bC2b′ + eBE

∥∇f(xk+1)∥2
+ eζ2

(As. D.5)
≤

 4GPGk
b
CcδB

(1 −δ) bC
+ eB

!

E
h∇f
 
xk+12i
+
4GPGk
b
CcδB

(1 −δ) bC
ζ2 + eζ2

+

 
PGk
b
CG

(1 −δ)2 bC2 + 4cδ

!
σ2

b′

(12)
≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 +
∇f
 
xk+1
−∇f
 
xk2i

+
4GPGk
b
CcδB

(1 −δ) bC
ζ2 + eζ2 +

 
PGk
b
CG

(1 −δ)2 bC2 + 4cδ

!
σ2

b′

≤

 8GPGk
b
CcδB

(1 −δ) bC
+ 2 eB

!

E
h∇f
 
xk2 + L2 xk+1 −xk2i

+
4GPGk
b
CcδB

(1 −δ) bC
ζ2 + eζ2 +

 
PGk
b
CG

(1 −δ)2 bC2 + 4cδ

!
σ2

b′ ,

69


---Page Break---
which concludes the proof.

G.2
Main Results for Byz-VR-MARINA without Full-Batch Gradient Computations

G.2.1
General Results

All the lemmas derived in Appendix D.2 hold for Algorithm 4 as well except Lemma D.10, which
can be replaced with Lemma G.2, and Lemma D.13 that has the following analog.
Lemma G.3. Let Assumptions 2.3, D.1, D.2, D.3, D.5, G.1 hold and Compression Operator satisfy
Definition 2.2. Also, let us introduce the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP without full-batch gradient
computations (Algorithm 4) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

4


E
hgk −∇f
 
xk2i
+

 
3PGk
b
CG

(1 −δ)2 bC2 + 12cδ

!
σ2

b′

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

where A, bB, bD, pG, PGk
C are defined in Lemma D.13.

Proof. Up to the replacement of the bound from Lemma D.10 with the bound from Lemma G.2, the
proof of the result is identical to the proof of Lemma D.13.

Theorem G.4. Let Assumptions 2.3,
D.1,
D.2,
D.3,
D.5,
G.1 hold.
Set λk+1
=
2 maxi∈G Li
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where A and bB are defined in Theorem D.14. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA without full-batch gradient computations (Algorithm 4) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 4 bDζ2

p −4 bB
+

 12PGk
b
CG

(1 −δ)2 bC2 + 48cδ

!
σ2

b′(p −4 bB)
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

2γ

p
g0 −∇f
 
x02 .

Proof. The proof is identical to the proof of Theorem D.14 up to the replacement of Lemma D.13
with Lemma G.3.

Theorem G.5.
Let Assumptions 2.3,
D.1,
D.2,
D.3,
D.5,
2.7 hold.
Set λk+1
=
maxi∈G Li
xk+1 −xk. Assume that

0 < γ ≤min

1
L +
√

2A


,
8 bB < p

where A and bB are defined in Theorem D.15. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA without full-batch gradient computations (Algorithm 4) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 4 bDγζ2

pρ
+

 12PGk
b
CG

(1 −δ)2 bC2 + 48cδ

!
γσ2

b′pρ,

where ρ = min
h
γµ

1 −8 b
B
p

, p

8
i
and Φ0 = f
 
x0
−f ∗+ 4γ

p
g0 −∇f
 
x02.

70


---Page Break---
Proof. The proof is identical to the proof of Theorem D.15 up to the replacement of Lemma D.13
with Lemma G.3.

In contrast to their counterparts for Byz-VR-MARINA-PP with (periodical) full-batch gradient
computations (Theorems D.14 and D.15), the above results have additional terms proportional to σ2

b′
in the upper bounds. These terms cannot be reduced with the decrease of the stepsize but can be made
smaller via the increase of b′. A similar phenomenon appears in the analysis of the methods with
recursive variance reduction even in Byzantine-free case (Fang et al., 2018; Li et al., 2021; Gorbunov
et al., 2021), and to address it, b′ is typically chosen to be large.

G.2.2
Results for Bounded Compressors

Similarly to the previous section, we start with an adaptation of Lemma E.3 to the case without
full-batch gradient computations.

Lemma G.6. Let Assumptions 2.3, D.1, D.2, D.3, D.4, D.5, G.1, 2.4 hold and the compression
operator satisfy Definition 2.2. We set λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥. Also, let us introduce
the notation

ARAggk+1
Q
= ARAgg

clipλk+1

Q

b∆1(xk+1, xk)

, . . . , clipλk+1

Q

b∆C(xk+1, xk)

.

Then for all k ≥0 the iterates produced by Byz-VR-MARINA-PP without full-batch gradient
computations (Algorithm 4) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

2


E
hgk −∇f
 
xk2i
+

 
3PGk
b
CG

(1 −δ)2 bC2 + 12cδ

!
σ2

b′

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,

where A, bB, bD, pG, PGk
C are defined in Lemma D.13.

Proof. Up to the replacement of the bound from Lemma D.10 with the bound from Lemma G.2, the
proof of the result is identical to the proof of Lemma E.3.

Theorem G.7. Let Assumptions 2.3, D.1, D.2, D.3, D.4, D.5, G.1, 2.4 hold. Setting λk+1 =
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where A and bB are defined in Theorem E.4. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA without full-batch computations (Algorithm 4) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 2 bDζ2

p −4 bB
+

 
6PGk
b
CG

(1 −δ)2 bC2 + 24cδ

!
σ2

b′(p −4 bB)
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

γ
p
g0 −∇f
 
x02 .

Proof. The proof is identical to the proof of Theorem E.4 up to the replacement of Lemma E.3 with
Lemma G.6.

Theorem G.8. Let Assumptions 2.3, D.1, D.2, D.3, D.4, D.5, G.1, 2.4, 2.7 hold. Setting λk+1 =
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

2A
,
8 bB < p,

71


---Page Break---
where A and bB are defined in Theorem E.5. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA without full-batch computations (Algorithm 4) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 2 bDζ2

pρ
+

 
6PGk
b
CG

(1 −δ)2 bC2 + 24cδ

!
γσ2

b′pρ,

where ρ = min
h
γµ

1 −8 b
B
p

, p

4
i
and Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.

Proof. The proof is identical to the proof of Theorem E.5 up to the replacement of Lemma E.3 with
Lemma G.6.

G.3
Main Results for Byz-VR-MARINA+ without Full-Batch Gradient Computations

G.3.1
Results for Bounded Compressors

Similarly to the analysis of Byz-VR-MARINA without full-batch gradient computations, we start
with the adaptation of Lemma F.1 to the no-full-batch gradient computations case.
Lemma G.9. Let Assumptions D.1, D.2, D.3, D.4, D.5, G.1, 2.4 hold and the compression operator
satisfy Definition 2.2. Assume that C ≤G. We set λk+1 = DQ maxi,j Li,j∥xk+1 −xk∥. Then for
all k ≥0 the iterates produced by Byz-VR-MARINA-PP+ without full-batch gradient computations
(Algorithm 5) satisfy

E
hgk+1 −∇f
 
xk+12i
≤

1 −p

2


E
hgk −∇f
 
xk2i
+

 
PGk
b
CG

(1 −δ)2 bC2 + 4cδ

!
pσ2

b′

+ bBE
h∇f
 
xk2i
+ bDζ2 + pA

4 ∥xk+1 −xk∥2,
(35)

where A, bB, bD, pG, PGk
C are defined in Lemma F.1.

Proof. Up to the replacement of the bound from Lemma D.10 with the bound from Lemma G.2, the
proof of the result is identical to the proof of Lemma F.1.

Theorem G.10. Let Assumptions D.1, D.2, D.3, D.4, D.5, G.1, 2.4 hold.
Set λk+1
=
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤
1
L +
√

A
,
4 bB < p,

where A and bB are defined in Theorem F.2. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA+ without full-batch gradient computations (Algorithm 5) satisfy

E
h∇f
 
bxK2i
≤
2Φ0

γ

1 −4 b
B
p

(K + 1)
+ 2 bDζ2

p −4 bB
+

 
2PGk
b
CG

(1 −δ)2 bC2 + 8cδ

!
pσ2

b′(p −4 bB)
,

where bxK is chosen uniformly at random from x0, x1, . . . , xK, and Φ0
= f
 
x0
−f ∗+

γ
p
g0 −∇f
 
x02 .

Proof. The proof is analogous to the proof of Theorem D.14.

Theorem G.11. Let Assumptions 2.4, D.1, D.2, D.3, D.4, D.5, G.1 2.7 hold.
Set λk+1 =
maxi,j Li,j
xk+1 −xk. Assume that

0 < γ ≤min

1
L +
√

2A


,
8 bB < p,

where A and bB are defined in Theorem F.3. Then for all K ≥0 the iterates produced by Byz-VR-
MARINA+ without full-batch gradient computations (Algorithm 5) satisfy

E

f
 
xK
−f (x∗)

≤(1 −ρ)K Φ0 + 2 bDζ2

pρ
+

 
2PGk
b
CG

(1 −δ)2 bC2 + 8cδ

!
γσ2

b′ρ ,

where ρ = min
h
γµ

1 −8 b
B
p

, p

4
i
and Φ0 = f
 
x0
−f ∗+ 2γ

p
g0 −∇f
 
x02.

72


---Page Break---
Proof. The proof is analogous to the proof of Theorem D.15.

As in the case of Byz-VR-MARINA, the above upper bounds for Byz-VR-MARINA+ without full-
batch gradient computations have additional terms proportional to σ2

b′ . In contrast to the results for
Byz-VR-MARINA+ without full-batch gradient computations, these terms for Byz-VR-MARINA+
are 1/p times smaller.

73


---Page Break---
H
Experimental Details and Extra Experiments

H.1
Experimental Details

For each experiment, we tune the step size using the following set of candidates {0.1, 0.01, 0.001}.
The step size is fixed. We do not use learning rate warmup or decay. We use batches of size 32 for all
methods. For partial participation, in each round, we sample 20% of clients uniformly at random.
For λk = λ∥xk −xk−1∥used for clipping, we select λ from {0.1, 1., 10.}. Each experiment is run
with three varying random seeds, and we report the mean optimality gap with one standard error. The
optimal value is obtained by running gradient descent (GD) on the complete dataset for 1000 epochs.
Our implementation of attacks and robust aggregation schemes is based on the public implementation
from (Gorbunov et al., 2023).
H.2
Extra Experiments
Below we provide the missing neural network experiments from the main paper. We consider the
MNIST dataset (LeCun and Cortes, 1998) and CIFAR10 (Krizhevsky et al., 2009) (as in (Karimireddy
et al., 2021)) with 20 clients, 5 of which are malicious, and 4 clients are sampled in each step. For
the attacks, we consider A Little is Enough (ALIE) (Baruch et al., 2019) and the aforementioned
Shift-Back (SHB). For the aggregations, we consider coordinate median (CM) (Chen et al., 2017)
and robust federated averaging (RFA) (Pillutla et al., 2022) with bucketing. For the MNIST dataset,
we use a simple neural network with two convolution layers followed by two fully connected. For
CIFAR 10, we use ResNet18 (He et al., 2016) architecture with layer norm. One can note that the
results are consistent with the ones provided in the main paper, i.e., clipping performs on par or better
than its variant without clipping, and no robust aggregator is able to withstand the shift-back attack
without clipping. Our implementation is available at https://github.com/SamuelHorvath/VR_
Byzantine/tree/partial_participation.

0
1
2
3
4
epochs

10−1

100

f(x)

CM | ALIE

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

CM | SHB

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

RFA | ALIE

W/ Clip
W/O Clip

0
1
2
3
4
epochs

10−1

100

f(x)

RFA | SHB

W/ Clip
W/O Clip

0
1
2
3
4
epochs

20

40

60

80

100

Accuracy

CM | ALIE

W/ Clip
W/O Clip

0
1
2
3
4
epochs

0

20

40

60

80

100

Accuracy

CM | SHB

W/ Clip
W/O Clip

0
1
2
3
4
epochs

20

40

60

80

100

Accuracy

RFA | ALIE

W/ Clip
W/O Clip

0
1
2
3
4
epochs

20

40

60

80

100

Accuracy

RFA | SHB

W/ Clip
W/O Clip

Figure 3: Training loss (top) and test accuracy (bottom) of 2 aggregation rules (CM, RFA) under 4
attacks (BF, LF, ALIE, SHB) on the MNIST dataset under heterogeneous data split with 20 clients, 5
of which are malicious, 4 clients sampled per round.

0
10
20
30
40
50
60
epochs

100

f(x)

CM | ALIE

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

100

f(x)

CM | SHB

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

100

f(x)

RFA | ALIE

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

100

101

f(x)

RFA | SHB

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

20

40

60

80

Accuracy

CM | ALIE

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

20

40

60

80

Accuracy

CM | SHB

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

20

40

60

80

Accuracy

RFA | ALIE

W/ Clip
W/O Clip

0
10
20
30
40
50
60
epochs

20

40

60

80

Accuracy

RFA | SHB

W/ Clip
W/O Clip

Figure 4: Training loss (top) and test accuracy (bottom) of 2 aggregation rules (CM, RFA) under 4
attacks (BF, LF, ALIE, SHB) on the CIFAR10 dataset under heterogeneous data split with 20 clients,
5 of which are malicious, 4 clients sampled per round.

74


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: see Sections 1, 3, 4, 5

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

Justification: see Sections 2, 3, 4

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
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

75


---Page Break---
Justification: see Section 4 and the appendix
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
Justification: see Section 5
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

76


---Page Break---
Answer: [Yes]
Justification: see Section 5
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
Justification: see Section 5
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
Justification: see Section 5
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
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

77


---Page Break---
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

Answer: [No]

Justification: We do not provide specific information about the computer resources used for
our experiments. However, none of our experiments require significant computational power.
Each experiment can be run in less than 5 minutes on a Quadro RTX 6000 NVIDIA GPU.

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

Justification: our work follows the NeurIPS Code of Ethics

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

Justification: theoretical paper

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

78


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

Justification: we do not train new models

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

Justification: we provide citations when necessary

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

79


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: the paper does not release new assets
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
Justification: the paper does not involve crowdsourcing nor research with human subjects
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
Justification: the paper does not involve crowdsourcing nor research with human subjects
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

80


---Page Break---
