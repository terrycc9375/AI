Efficient Federated Learning against Heterogeneous
and Non-stationary Client Unavailability

Ming Xiang1
Stratis Ioannidis1
Edmund Yeh1
Carlee Joe-Wong2
Lili Su1

1Northeastern University, Boston, MA
2Carnegie Mellon University, Pittsburgh, PA
{xiang.mi,l.su}@northeastern.edu
{ioannidis,eyeh}@ece.neu.edu
cjoewong@andrew.cmu.edu

Abstract

Addressing intermittent client availability is critical for the real-world deployment
of federated learning algorithms. Most prior work either overlooks the potential
non-stationarity in the dynamics of client unavailability or requires substantial
memory/computation overhead. We study federated learning in the presence of
heterogeneous and non-stationary client availability, which may occur when the
deployment environments are uncertain, or the clients are mobile. The impacts of
heterogeneity and non-stationarity on client unavailability can be significant, as we
illustrate using FedAvg, the most widely adopted federated learning algorithm. We
propose FedAWE, which includes novel algorithmic structures that (i) compensate
for missed computations due to unavailability with only O(1) additional memory
and computation with respect to standard FedAvg, and (ii) evenly diffuse local
updates within the federated learning system through implicit gossiping, despite
being agnostic to non-stationary dynamics. We show that FedAWE converges to a
stationary point of even non-convex objectives while achieving the desired linear
speedup property. We corroborate our analysis with numerical experiments over
diversified client unavailability dynamics on real-world data sets.

1
Introduction

Federated learning is a distributed machine learning approach that enables training global models
without disclosing raw local data [31, 20]. It has been adopted in commercial applications such as
autonomous vehicles [6, 69, 40], the Internet of things [38], and natural language processing [62, 42].

Heterogeneous data and massive client populations are two of the defining characteristics of cross-
device federated learning systems [31, 20]. Despite intensive efforts [31, 28, 67, 44, 20], several key
challenges that arise from the involvement of large-scale client populations are often overlooked
in the existing literature [41]. One of the primary hurdles is the issue of client unavailability.
Intuitively, more active clients drive the global model to their local optima by overfitting their local
data, which biases the training. In addition, the higher the uncertainty in client unavailability, the
larger the performance degradation. Concrete examples that confirm these intuitions in the context
of FedAvg - the most widely adopted federated learning algorithm - can be found in Section 4.
Client unavailability issues can arise from internal factors such as different working schedules and
heterogeneous hardware/software constraints. External factors, such as poor network coverage
and frequent handovers of base stations due to fast movements, only exacerbate these problems
[49, 56, 63, 3, 20]. The intricate interplay of internal and external factors results in the non-stationarity
and heterogeneity of client unavailability.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Figure 1: Client i’s available proba-
bilities pt
i’s are heterogeneous and are
subject to non-stationary dynamics.

Most prior work either assumes exact knowledge of the clients’
available dynamics or requires their dynamics to be benignly
stationary [31, 26, 41, 54, 53]. A related line of work studies
asynchronous federated learning wherein clients are vulnera-
ble to delays in message transmission and the reported model
updates may be stale [58, 37, 48, 24]. The proposed methods
therein assume the availability of all clients or uniformly sam-
pled clients, making them inapplicable to our settings. A few
recent works [43, 57] study non-stationary dynamics. Ribero et
al. [43] consider the settings where the available probabilities
follow a homogeneous Markov chain. Xiang et al. [57] require
that clients be capable of continuous local optimization regardless of communication failures. A
handful of other works [13, 59] memorize the old gradients of the unavailable clients to compensate
for their unavailability. However, the added memory burdens the federated learning system with
substantial memory proportional to the product of the number of clients and the model dimension.

Contributions. In this work, we focus on stochastic client unavailability, where client i is available
for federated learning model training with probability pt
i at any time t. An illustration can be found
in Fig. 1. Our contributions are four-fold:

• In Section 4, via constructing concrete examples, we demonstrate that both heterogeneity and
non-stationarity of pt
i will result in bias and thus significant performance degradation of FedAvg.
• In Section 5, we propose an algorithm named FedAWE, which features computational and memory
efficiency: only O(1) additional computation and memory per client will be used when compared
with FedAvg. The design of FedAWE introduces two novel algorithmic structures: adaptive inno-
vation echoing and implicit gossiping. At a high level, these novel algorithmic structures (i) help
clients catch up on the missed computation, and (ii) simultaneously enable a balanced information
mixture through implicit client-client gossip, which ultimately corrects the remaining bias. Notably,
no direct neighbor information exchanges are used, and the client unavailability dynamics remains
unknown to all clients and the parameter server.
• In Section 6, we show that FedAWE converges to a stationary point of even non-convex global
objective and achieves the linear speedup property without conditions on second-order partial
derivatives of the loss function in analysis.
• In Section 7, we validate our analysis with numerical experiments over diversified client unavail-
ability dynamics on real-world data sets.

2
Related Work

Dynamical client availability. There is a recent surge of efforts to study time-varying client
availability [44, 43, 7, 53, 43, 41, 57], which can be roughly classified into two categories depending
on whether the parameter server can unilaterally determine the participating clients.

(i) Controllable participation. Earlier research [31, 28] presumes that, in each round, the parameter
server could select a small set of clients either uniformly at random or in proportion to the volume
of local data held by clients. More recently, Cho et al. [10] design adaptive and non-uniform client
sampling to accelerate learning convergence, albeit at the cost of introducing a non-zero residual error.
In another work, Cho et al. [8] study the convergence of FedAvg with cyclic client participation. Yet,
the set of available clients is sampled uniformly at random per cyclic round and is decided unilaterally
by the parameter server. Perazzone et al. [41] consider heterogeneous and time-varying response
rates pt
i under the assumptions that pt
i is known a priori and that the stochastic gradients are bounded
in expectation. Furthermore, the dynamics of pt
i are determined by the parameter server by solving a
stochastic optimization problem. Chen et al. [7] propose a client sampling scheme wherein only the
clients with the most “important” updates communicate back to the parameter server. This sampling
method can achieve performance comparable to that of full client participation, provided that pt
i is
globally known to both the parameter server and the clients. Departing from this line of literature, our
setup neither assumes any side information or prior knowledge of the response rates pt
i nor assumes
that the parameter server has any influence on pt
i.

(ii) Uncontrollable participation. There is a handful of work on building resilience against arbitrary
client availability [43, 53, 59, 13, 61, 54]. Ribero et al. [43] consider random client availability

2


---Page Break---
whose underlying response rates are also heterogeneous and time-varying with unknown dynamics.
However, the underlying dynamics of pt
i in [43] are assumed to follow a homogeneous Markov chain.
Wang et al. [53] propose a generalized FedAvg that amplifies parameter updates every P rounds
for some carefully tuned P. Despite its elegant unified analysis and potential to accommodate non-
independent unavailability dynamics, to reach a stationary point, pt
i needs to satisfy some assumptions
to ensure roughly equal availability of all clients over every P rounds. Yang et al. [61] analyze
a setting where clients participate in the training at their will. Yet, their convergence is shown to
be up to a non-zero residual error. The algorithms proposed in [13, 59] share the same idea of
using the memorized latest updates from unavailable clients for global aggregation. Despite superior
numerical performance, both algorithms demand a substantial amount of additional memory [54].
For non-convex objectives, both [59] and [13] require an absolute bounded inactive period, and share
similar technical assumptions such as almost surely bounded stochastic gradients [59] or Lipschitz
Hessian [13]. Though bounded inactive periods are relevant for applications wherein the sensors
wake up on a periodic schedule, this assumption is not satisfied even for the simple stochastic setting
when clients are selected uniformly at random. Wang and Ji consider unknown heterogeneous pi’ in
a concurrent work [54]; however, pi’s are assumed to be fixed over time.

Asynchronous federated learning. Another related line of work is asynchronous federated learning.
To the best of our knowledge, Xie et al. [58] initialize the study of asynchronous federated learning,
wherein the parameter server revises the global model every time it receives an update from a client.
Convergence is shown under some technical assumptions such as weakly-convex global objectives,
bounded delay, and bounded stochastic gradients. Zakerinia et al. [68] propose QuAFL which is
shown to be resilient to computation asynchronicity and quantized communication yet under the
bounded and stationary delay assumption. Nguyen et al. [37] propose FedBuff, which uses additional
memory to buffer asynchronous aggregation to achieve scalability and privacy. Convergence is shown
under bounded gradients and bounded staleness assumptions. In fact, most convergence guarantees in
the asynchronous federated learning literature rely on bounded staleness [58, 37, 48, 24], or bounded
gradients [58, 37, 24]. Recently, arbitrary delay is considered in the context of distributed SGD with
bounded stochastic gradients and (0, ζ)-bounded inter-client heterogeneity [32] (see Assumption 4
for the definition). The convergence suffers from a non-zero residual term O(ζ2). In contrast, our
convergence guarantee is free from non-zero residual terms and does not require gradients to be
bounded.

3
Problem Formulation

A federated learning system consists of a parameter server and m clients that collaboratively minimize

min
x∈Rd F(x) ≜1

m

m
X

i=1
Fi(x),
(1)

where Fi(x) ≜Eξi∼Di [ℓi(x; ξi)] is the local objective and can be non-convex, Di is the local
distribution, ξi is a stochastic sample that client i has access to, ℓi is the local loss function, and d is
the model dimension.

We use Assumption 1 to capture the uncertain non-stationary dynamics and heterogeneity. Let At
denote the set of active clients, 1{·} an indicator function, T the number of total training rounds.

Assumption 1. There exists a δ ∈(0, 1] such that pt
i ≜E[1{i∈At}] ≥δ, where the events {i ∈At}
are independent across clients i and across rounds t ∈[T].

Assumption 1 subsumes uniform availability [26, 61] and stationary availability considered in [54].
Independent client unavailability is widely adopted by federated learning research [26, 28, 22, 60, 61,
54]. Analyzing non-independent unavailability, together with uncertain and non-stationary dynamics
in Assumption 1, is in general challenging. Specifically, the involved entanglement of stochastic
gradient and availability statistics fundamentally complicates the theoretical analysis. However, we
conjecture that independence and strictly positive probabilities are only necessary for the technical
convenience of our analysis. Our experiments in Section 7 suggest that our algorithm offers notable
improvement even in the presence of non-independent and occasionally zero-valued probabilities.
Future work will investigate how to provably accommodate correlated or zero-valued probabilities of
arbitrary probabilistic trajectories.

3


---Page Break---
4
Heterogeneity and Non-stationarity May Lead to Significant Bias

0.0
0.2
0.4
0.6
0.8
1.0
p1

0.0

0.2

0.4

0.6

0.8

1.0

p2

0

10

20

30

40

50

||xoutput
x ||

Figure 2: Let xoutput ≜limt→∞E

xt.
Under most of the choices of p1, p2, xoutput
is far from x∗.

In this section, we illustrate the impacts of heterogeneity
and non-stationarity of client availability under the clas-
sic FedAvg. We use two examples to showcase the signifi-
cant bias incurred.

Example 1 (Heterogeneity). Suppose that m = 2 and pt
i =
pi for i ∈[2]. Let Fi (x) ≜∥x −ui∥2
2 /2, where x, ui ∈R.
The global objective (1) is

F (x) = 1

2(∥x −u1∥2
2 + ∥x −u2∥2
2),
(2)

with unique minimizer x⋆= (u1 + u2)/2. Let u1 = 0 and
u2 = 100. Fig. 2 illustrates how the heterogeneity in pi
affects the expected output of FedAvg.

Example 1 matches [54, Theorem 1], which shows that FedAvg leads to a biased global objective (3)
under heterogeneous pi’s, and that (3) may be significantly away from (1) depending on pi’s.

eF(x) ≜

m
X

i=1

pi
Pm
j=1 pj
Fi(x).
(3)

When the probabilistic dynamics of pt
i’s is non-stationary, obtaining an exact biased objective similar
to (3) in a neat analytical form becomes challenging, if not impossible, due to the unstructured non-
stationary dynamics. Fortunately, Example 2 helps us confirm that the complex interplay between
pt
i’s across rounds and clients will inevitably further degrade the performance of FedAvg algorithm.

Example 2 (Non-stationarity). In Fig. 3, a total of m = 100 clients perform an image classification
task on the SVHN dataset [36] under the FedAvg algorithm, whose local dataset distribution follows
Dirichlet(0.1) [16]. Clients become available with probability pt
i = p·[γ·sin(0.1π·t)+(1−γ)], ∀i ∈
[m]. The hyperparameter details are deferred to Appendix J. Observations can be found in the caption.

= 0.1
= 0.2
= 0.3
= 0.5
Non-stationary Degree 

68%

70%

73%

75%

78%

80%

82%

Train Acc

p = 0.1

p = 0.3

p = 0.5

p = 1.0

(a) Train accuracy.

= 0.1
= 0.2
= 0.3
= 0.5
Non-stationary Degree 

68%

70%

73%

75%

78%

80%

82%

Test Acc

p = 0.1

p = 0.3

p = 0.5

p = 1.0

(b) Test accuracy
Figure 3: Train and test accuracy results in percentage (%). In particular, the parameter γ signifies the
degree of non-stationary. Notice that, as the client availability becomes more non-stationary (a larger γ),
FedAvg experiences a significant drop in accuracy. For example, both the train and test accuracies drop by over
10% when p = 0.1, and γ increases from 0.1 to 0.5.

5
Federated Agile Weight Re-Equalization (FedAWE)

To minimize (1), one natural idea is to have the entire client population performs the same number of
local updates and mixes these updates carefully to ensure they are weighted equally. Unfortunately,
when clients are available only intermittently, they will miss some rounds. A naive approach to
equalizing the number of local updates is to have clients catch up by performing their missed local
computations immediately when they become available. However, this approach requires a daunting
amount of resources and may not be possible due to hardware/software constraints. Formally, recall
that At is the set of available clients at time t. Let τi(t) ≜{t′ : t′ < t and i ∈At′} denote the most
recent (with respect to time t) round that client i is available. Compared with standard FedAvg, the

4


---Page Break---
Algorithm 1: FedAWE

1 Inputs: T, s, ηl, ηg, x0.

2 for i ∈[m] do x0
i ←x0 and τi(0) ←−1 ;

3 for t = 0, · · · , T −1 do

4
for i ∈At do

5
x(t,0)
i
←xt
i;

6
for k = 0, · · · , s −1 do

7
x(t,k+1)
i
←

8
x(t,k)
i
−ηl∇ℓi(x(t,k)
i
; ξ(t,k)
i
);

9
end

10
Gt
i ←xt
i −x(t,s)
i
;

11
xt†
i ←x(t,0)
i
−ηg(t −τi(t))Gt
i;

12
τi(t + 1) ←t;

13
Report xt†
i to the parameter server;

14
end

15
xt+1 ←
1
|At|
P
i∈At xt†
i ;

16
for i ∈[m] do

17
if i ∈At then

18
xt+1
i
←xt+1;

19
else if i /∈At then

20
xt+1
i
←xt
i;

21
τi(t + 1) ←τi(t);

22
end

23 end

naive “catch-up” procedure will consume (t −τi(t) −1) · s local stochastic gradient descent updates
and (t −τi(t) −1) additional stochastic samples, where s is the number of local updates per round
when a client is available in standard FedAvg.

In this work, we target computation-light algorithms that, compared with FedAvg, only take O(1)
additional computation without additional stochastic samples. We propose Federated Agile Weight
Re-Equalization (FedAWE), which is formally described in Algorithm 1. It involves two novel algo-
rithmic structures: adaptive innovation echoing and implicit gossiping. At a high level, these novel
algorithmic structures (i) help clients catch up on the missed computation, and (ii) simultaneously
enable a balanced information mixture through implicit client-client gossip, which ultimately corrects
the remaining bias.

In Algorithm 1, each client keeps two local variables xi and τi, along with a few auxiliary variables
used in updating xi and τi. The algorithm inputs are rather standard: total training rounds T, local
and global learning rates ηl and ηg, the number of local updates per round s, and the initial model x0.
In each round t, similar to FedAvg, an available client i ∈At performs s steps of stochastic gradient
descent on its local model xt
i (lines 5-8), where ∇ℓi(·; ξ(t,k)
i
) is the stochastic gradient of sample
ξ(t,k)
i
. Next, we describe the two novel algorithmic structures used in FedAWE.

Adaptive innovation echoing. Departing from FedAvg wherein the local estimate xt
i is updated as
xt†
i ←x(t,0)
i
−ηgGt
i. In FedAWE (lines 10-11), we “echo” the local innovation Gt
i by multiplying
it by (t −τi(t)). Intuitively, this simple echoing helps us approximately equalize the number of
local improvements, as formally stated in Proposition 1. It says that the total numbers of innovations
echoing are the same for all active clients for any given round and allows the unavailable clients to
catch up to the missed computations when they become available.

Proposition 1. If 1{i∈AR−1} = 1, it holds that PR−1
t=0 1{i∈At} (t −τi(t)) = R, ∀R ≥1.

W (t)
ij ≜








1
|At|, if i, j ∈At;
1, if i = j and i /∈At;
0, otherwise.
(4)

Implicit gossiping. In FedAWE, the parameter server
does not send the most recent global model to the ac-
tive clients at the beginning of a round. Instead, the
parameter server aggregates the locally updated models
xt†
i and sends the new global model xt+1 to all active
clients At (lines 14-15). By postponing multicasting the shared global model, the active clients
in At implicitly gossip their updated local models with each other through the parameter server
[57]. Though the postponed multi-cast brings in staleness, simple coupling argument show that
the staleness is bounded (Lemma 2). In addition, our empirical results (Table 8 in Appendix J)
suggest that there is no significant slowdown when compared to vanilla FedAvg. Gossip-type algo-
rithms were originally proposed for peer-to-peer networks and are well-known for their agility to
communication failures and asynchronous information exchange in achieving average consensus

5


---Page Break---
[12, 4, 23, 15, 30, 35]. Intuitively, the clients’ local estimates are eventually equally weighted in the
final algorithm output. Note that, departing from the standard gossiping protocols therein [23, 45],
information exchange in FedAWE does not involve client-client communication. The information
mixing matrix under FedAWE is defined in (4), which is doubly stochastic. Let M (t) ≜E[(W (t))2],
ρ(t) ≜λ2(M (t)), J = 11⊤/m, and ρ ≜maxt ρ(t), where λ2(·) denotes the second largest
eigenvalue. We next characterize the information mixing error, i.e., consensus error in Lemma 1.

Lemma 1 ([34, 33, 50]). For any matrix B ∈Rd×m, it holds that EW [∥B
Qt
r=1 W (r) −J

∥2
F] ≤

ρt∥B∥2
F, where the expectation is taken with respect to randomness in W matrices.

6
Convergence Analysis

In this section, we analyze the convergence of FedAWE. All missing proofs and intermediate results
are deferred to the Appendix. Details can be found in Table of Contents.

6.1
Assumptions

We start by stating regulatory assumptions that are common in federated learning analysis [26, 51, 22].

Assumption 2. Each local objective function ∇Fi(x) is L-Lipschitz, i.e.,

∥∇Fi(x1) −∇Fi(x2)∥2 ≤L ∥x1 −x2∥2 , ∀x1, x2, and ∀i ∈[m].

Assumption 3. Stochastic gradients ∇ℓi(x; ξ) are unbiased with bounded variance, i.e.,

E [∇ℓi(x; ξ) | x] = ∇Fi(x) and E
h
∥∇ℓi(x; ξ) −∇Fi(x)∥2
2 | x
i
≤σ2, ∀i ∈[m].

Assumption 4. The divergence between local and global gradients is bounded for β, ζ ≥0 such that

1
m

m
X

i=1
∥∇Fi(x) −∇F(x)∥2
2 ≤β2 ∥∇F(x)∥2
2 + ζ2.
(5)

When the local data sets are homogeneous, ∇Fi(x) = ∇F(x) holds for any client i ∈[m], resulting
in β = ζ = 0. Assumption 4 and its variants in Table 1 are often referred to as bounded gradient
dissimilarity assumption to account for data heterogeneity across clients. It can be easily checked
that our Assumption 4 is more relaxed or equivalent to the variants therein.

Table 1: Popular variant assumptions on gradient dissimilarity.
Bounded Gradient Dissimilarity
References

maxx ∥∇Fi(x)∥2
2 ≤ζ2, ∀i ∈[m]
[28, 65, 9, 10, 59]

1
m
Pm
i=1 ∥∇Fi(x)∥2
2 ≤β2 ∥∇F(x)∥2
2
[26, 27]

1
m
Pm
i=1 ∥∇Fi(x) −∇F(x)∥2
2 ≤ζ2
[52, 64, 17, 55, 1, 21, 53, 61]

1
m
Pm
i=1 ∥∇Fi(x)∥2
2 ≤β2 ∥∇F(x)∥2
2 + ζ2
[22, 67, 51, 50, 13]

6.2
Auxiliary/Imaginary update sequence construction.

Directly analyzing the evolution of xt and xt
i is challenging due to the fact that different clients
update at different rounds, and that different active clients echo their local innovation Gt
i (line 9
in Algorithm 1) with different strength (t −τi). As such, we construct an auxiliary/imaginary update
sequence zt
i for client i ∈[m], whose evolution is closely coupled with xt and xt
i but is easier to
analyze. Note that the auxiliary/imaginary update sequence is never actually computed by clients but
acts as a necessary tool in building up the analysis.

Definition 1. The auxiliary sequence {zt
i} of client i ∈[m] is defined as

zt
i ≜xt
i −ηlηgs(t −τi(t) −1)∇Fi(xτi(t)+1
i
), ∀i ∈[m].
(6)

6


---Page Break---
Recall that τi(0) = −1. Thus, by definition, z0
i = x0
i according to (6). For general t, when client
i ∈At−1, we simply have τi(t) = t −1 and thus t −1 −τi(t) = t −1 −(t −1) = 0. That is, the
auxiliary model zt
i and the real model xt
i are identical whenever the client i becomes available in the
previous round.

• When i ∈At−1, the iterate of zi is a bit more involved:

zt
i
(7.a)
= xt
i
(7.b)
=

P

j∈At−1
|At−1|




zt−1
j
+ (xt−1
j
−zt−1
j
)
|
{z
}
(7.c)

−ηlηg(t −1 −τj(t −1))Gt−1
j




,
(7)

where (7.a) holds because of Definition 1 and i ∈At−1, (7.b) because of line 10 in Algorithm 1,
addition and subtraction. (7.c) can be expanded by (6). We defer the simplified form of (7) to (18)
in Appendix C for a tidy presentation.
• When i /∈At−1, zt
i has a simple iterative relation:

zt
i = zt−1
i
−ηlηgs∇Fi(xτi(t−1)+1
i
).
(8)

At a high level, the sequence zt
i approximately mimics the ideal descent evolution at a client as if
the client performs local optimizations on its local model xi per round regardless of its availability.
Mathematically, the idea is that, if the progress per iteration of the auxiliary sequence zt
i is bounded,
we can show the convergence of xt
i when xt
i and zt
i are close to each other.

It is worth noting that auxiliary sequences are used in peer-to-peer distributed learning literature
[46, 2, 29, 66, 47, 33]. Yet, existing constructions are not applicable to our problem due to (1) the
non-convexity of the global objectives, (2) multiple local updates per round, (3) possibly unbounded
gradients, and (4) the general form of bounded gradient dissimilarity. Departing from the use
of staled stochastic gradients for auxiliary updates therein, we adopt the true gradient ∇Fi(·) to
avoid the complications from the involved interplay between randomness in stochastic samples
and randomness in τi(t). On the technical front, it follows from Definition 1 that ∥xt
i −zt
i∥2
2 ≤
η2
l η2
gs2(t −τi(t) −1)2∥∇Fi(xτi(t)+1
i
)∥2
2, whose bound appears to be quite challenging to derive
due to the coupling of different realizations of τi(t) and gradients. As such, we bound the average of
∥xt
i −zt
i∥2 across clients and rounds in Proposition 2.
Lemma 2 (Unavailability statistics). Under Assumption 1 and δ defined therein. It holds for t ≥0
that E [t −τi(t)] ≤1/δ and E
h
(t −τi(t))2i
≤2/δ2.

Lemma 2 yields an upper bound on the first and second moments of a client i’s unavailable duration
despite the unstructured nature of clients’ non-stationary and heterogeneous unavailability. In the
special case where we have clients available with the same probability δ, the duration simply follows a
homogeneous geometric distribution. It can be easily checked that our bounds trivially hold. However,
the duration becomes a more challenging non-homogeneous geometric random variable under our
non-stationary unavailability dynamics. Lemma 2 can be derived by using a simple coupling argument
and by using tools from probability theory [14].

6.3
Main results.

Let ¯zt ≜1

m
Pm
i=1 zt
i, F ⋆≜minx F(x), and δmax ≜maxi∈[m],t∈[T ] pt
i.

Lemma 3 (Descent Lemma). Let Ft define the sigma algebra generated by randomness up to round
t. Suppose Assumptions 2, 3 hold and ηlηg ≤9/(100sL), it holds that

E

F(¯zt+1) −F(¯zt) | Ft
≤−ηlηgs

4

∇F(¯zt)
2
2

+ 2ηlηgsLσ2  
ηlηgδmax + 4.5mη2
l sL


m2

m
X

i=1
(t −τi(t))2

+ 35ηgη3
l s3L2

m

m
X

i=1
(t −τi(t))2 ∇Fi(xτi(t)+1
i
)

2

2

+ 2.2ηlηgsL2

m

m
X

i=1

xt
i −zt
i
2
2
|
{z
}
Approximation Error

+ηlηgsL2

2m

m
X

i=1

zt
i −¯zt2
2
|
{z
}
Consensus Error

.

7


---Page Break---
The proof of Lemma 3 follows from the standard analysis for non-convex smooth objectives but with
non-trivial adaptation to account for adaptive innovation echoing and implicit gossiping. In particular,
it highlights two terms unique in our derivation: the approximation error from the auxiliary sequence
and the consensus error from the implicit gossiping procedure.
Proposition 2 (Approximation error). Given Assumptions 2 and 4, it holds that

1
mT

T −1
X

t=0

m
X

i=1
E
hxt
i −zt
i
2

2

i
≤6η2
l η2
gs2

δ2
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2

2

i
+ 6η2
l η2
gs2

δ2
ζ2

+6L2η2
l η2
gs2

δ2
1
m

m
X

i=1

1
T

T −1
X

t=0
E
hzt
i −¯zt2

2

i
.
(9)

The proof of Proposition 2 starts from Definition 1. Although in general it is difficult to bound the
error, Assumptions 2 and 4 allow us to break down the problem into bounding the averaged gradient
norm of ¯zt and the consensus error over all randomness instead. Next, we analyze the consensus error.
Note that although implicit gossiping takes place in Algorithm 1 for xt
i, its analysis is technically
challenging as discussed before. So, we adopt the auxiliary zt
i as an intermediary and apply Young’s
inequality to bound the actual consensus error. Details will be specified next. Formally, the auxiliary
models can be expressed in a compact matrix form as Z(t) ≜[zt
1, . . . , zt
m]. Their local parameter
innovation matrix f
Gt is formulated by combing (7) and (8). We refer the interested readers to (19)
in Appendix C for the exact formula. Unrolling the recursion, the consensus error can be expanded as

1
m



Z(t−1) −ηlηg e
G(t−1)
W (t−1) (I −J)

2

F

(10.a)
=
η2
l η2
g
m



t−1
X

q=0
e
G(q)




t−1
Y

l=q
W (q) −J







2

F

,

(10)

where equality (10.a) holds because all clients are initiated at the same weight.

Lemma 4 ([57]). Under Assumption 1, it holds that ρ ≤1 −δ4(1−(1−δ)m)2

8
.

Recall that ρ bounds the expected spectral norm of the information mixing matrix W (t). It is
important to have ρ < 1 for an exponential decay of the consensus error (see Lemma 1). We now
proceed to present the convergence rates. In the sequel, we assume it holds for ηg and ηl that

ηlηg ≤

 
1 −√ρ

δ

80s(L + 1)
 √ρ + 1
 p

(β2 + 1) (1 + L2)
; ηl ≤
δ
200sL
p

(β2 + 1) (1 + L2)
.
(11)

The proof of the consensus error borrows insights from the analysis of the gossip algorithm [34, 52]
but with substantial adaptation to accommodate the novel auxiliary formulation and multi-step local
updates. Under the learning rate conidtions in (11) and Assumptions 1, 2, 3 and 4, we can show that

1
mT

T −1
X

t=0

m
X

i=1
E
hxt
i −zt
i
2

2

i
≍
1
mT

T −1
X

t=0

m
X

i=1
E
hzt
i −¯zt2

2

i
≍1

T

T −1
X

t=0
E
h∇F(¯zt)
2

2

i
.
(12)

It remains to bound the full convergence error of zt
i, which is presented in Theorem 1.
Theorem 1 (Convergence error of zt
i). Suppose that Assumptions 1, 2, 3 and 4 hold. Choose learning
rates ηl and ηg such that the conditions in (11) are met for T ≥1, it holds that

1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
≲

 
F(¯z0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2
+ η2
l η2
gs2L2

σ2 + ζ2

δ2(1 −√ρ)2


. (13)

By addition, subtraction, and Young’s inequality, (14) and (15) hold under Assumption 2.

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2

2

i
≍1

T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −zt
i
2

2

i
+ 1

T

T −1
X

t=0

1
m

m
X

i=1
E
hzt
i −¯zt2

2

i
; (14)

1
T

T −1
X

t=0
E
h∇F(¯xt)
2

2

i
≍1

T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −zt
i
2

2

i
+ 1

T

T −1
X

t=0
E
h∇F(¯zt)
2

2

i
.
(15)

8


---Page Break---
Moreover, from (12), (14) and (15), it can be seen that (16) holds.

1
mT

T −1
X

t=0

m
X

i=1
E
hxt
i −¯xt2

2

i
≍1

T

T −1
X

t=0
E
h∇F(¯zt)
2

2

i
≍1

T

T −1
X

t=0
E
h∇F(¯xt)
2

2

i
.
(16)

Combining (12), (13), (14) and (15), we are ready for Corollary 1.

Corollary 1 (Convergence rate of xt
i). Suppose that Assumptions 1, 2, 3 and 4 hold. Choose learning
rates as ηl =
1
√

T sL, ηg =
√

sδm such that the conditions in (11) are met for T ≥1, it holds that

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≲L
 
F(¯x0) −F ⋆

√

sδmT
+
δmax
δ
3
2 √

smT
σ2 + sm

T


σ2 + ζ2

δ(1 −√ρ)2


.
(17)

Corollary 1 establishes the full convergence rate for FedAWE algorithm. It can be seen that the first
and second terms dominate when T is sufficiently large, which relate to initial suboptimality gap
and stochastic gradient noise σ2, respectively. The non-stationary client unavailability results in the
third term, which relates to gradient divergence ζ2 and also to σ2. The proof of Corollary 1 follows
from (15) by plugging in Proposition 2 and Theorem 1. In the special case where k clients participate
uniformly at random, we simply have δmax = δ = k/m. Our convergence bound attains the rate
of O(1/
√

skT). In other words, we achieve the desired linear speedup property with respect to
the number of local steps s and the number of active clients k, matching the established literature
[60, 53, 64, 65]. The linear speedup property enables a large cross-device federated learning system
to take advantage of a massive scale of parallelism. Notice that the consensus error (16) and the
convergence rate (17) have the same asymptotic order with respect to the parameters therein. Hence,
the consensus error also enjoys the desired linear speedup property when T is sufficiently large.

7
Numerical Experiments

Overview. In this section, we evaluate FedAWE on real-world data sets to corroborate our analysis and
compare it with the other state-of-the-art algorithms. The missing specifications and additional results
can be found in Appendix J. Specifically, we consider a federated learning system of one parameter
server and m = 100 clients, wherein clients become available intermittently. The image classification
tasks use CNNs and are based on SVHN [36], CIFAR-10 [25] and CINIC-10 [11] data sets. All of
them include 10 classes of images of different categories. To emulate a highly heterogeneous local
data distribution, the image class distribution νi ∼Dirichlet(α = 0.1) at client i [16, 53, 54].

Non-stationary client unavailability. A total of four unavailable dynamics are evaluated in Table 2,
including stationary and non-stationary with staircase, sine and interleaved sine trajectories, with their
visualizations available in the same table. The classification tasks become more challenging as the list
progresses due to the growing complexity in the non-stationary dynamics. Furthermore, our choices
of the non-stationary dynamics are motivated by real-world federated learning participation statistics,
for example, sine trajectory [3], and by generalizing the existing participation patterns such as cyclic
participation [8, 54]. In particular, the interleaved sine dynamics is more challenging than the vanilla
cyclic availability dynamics since clients become available during each active period with probability
that is less than 1 and non-stationary simultaneously. Formally, client i’s dynamics is defined as
pt
i = pi · fi(t), where fi(t) is a time-dependent function under non-stationary dynamics but fi(t) = 1
when stationary, and pi = ⟨νi, ϕ⟩. ϕ characterizes the unbalanced contribution of different image
classes to the generated probabilities. Each element of [ϕ]c is drawn from Uniform(0, Φc), where a
smaller Φc leads to a less significant contribution of that image class.

Correlating the local data distribution and the probability of client availability is a common practice
in the prior literature. For example, Gu et al. in [13] experiment with a formula for pi so that
clients that hold images of smaller digits participate less frequently. Wang and Ji in [54] construct
pi as an inner product of the clients’ local data distribution νi and an external distribution Φ′. It
is immediately clear that the coupling of local data distribution (νi ∼Dirichlet(α = 0.1)) and
class contribution ϕ leads to non-independent pi’s. In addition, Assumption 1 will not hold in the
case of interleaved sine non-stationary dynamics since pt
i’s occasionally reach 0. Although being
agnostic to the challenging client unavailability dynamics not covered by our analysis, we observe
that FedAWE retains its outperformance. Comparisons will be specified next.

9


---Page Break---
Table 2: Results and comparisons on real-world datasets in the form of mean accuracy ± standard deviation
and are obtained over 3 repetitions in different random seeds. Results are averaged over the last 50 rounds. The
total number of global rounds is 2000 for SVHN, CIFAR-10 and CINIC-10. Algorithms are categorized into two
groups: (1) ones not aided by memory or known statistics; (2) ones assisted by memory or known statistics. For
a fair competition, we boldface the best accuracy in the first group, while the second best is underlined.

Unavailable
Dynamics

Datasets
SVHN
CIFAR-10
CINIC-10
Algorithms
Train
Test
Train
Test
Train
Test

Stationary

pi

0

FedAWE (ours)
86.5 ± 0.7 %
86.1 ± 0.7 %
68.1 ± 1.4 %
66.3 ± 1.1 %
47.9 ± 2.1 %
47.3 ± 2.0 %
FedAvg over active
82.6 ± 1.0 %
82.4 ± 1.1 %
64.1 ± 1.9 %
62.9 ± 1.4 %
43.6 ± 2.4 %
43.1 ± 2.4 %
FedAvg over all
76.1 ± 2.1 %
76.1 ± 2.4 %
55.8 ± 2.1 %
55.4 ± 1.8 %
38.4 ± 2.1 %
38.0 ± 2.1 %
FedAU
83.4 ± 1.0 %
83.2 ± 1.0 %
65.4 ± 1.4 %
64.1 ± 1.0 %
45.6 ± 1.5 %
45.2 ± 1.5 %
F3AST
83.2 ± 0.7 %
83.2 ± 0.7 %
64.4 ± 1.1 %
63.5 ± 0.9 %
45.3 ± 1.2 %
44.8 ± 1.2 %
FedAvg with known pi’s
86.1 ± 0.5 %
85.6 ± 0.5 %
65.4 ± 1.0 %
63.1 ± 0.9 %
45.0 ± 1.2 %
44.6 ± 1.1 %
MIFA (memory aided)
84.2 ± 0.5 %
84.1 ± 0.6 %
66.6 ± 0.8 %
65.3 ± 0.6 %
47.5 ± 0.5 %
46.9 ± 0.5 %
FedVARP (memory aided)
84.6 ± 0.2 %
84.3 ± 0.1 %
67.5 ± 0.2 %
66.3 ± 0.3 %
47.8 ± 0.2 %
47.2 ± 0.2 %

Non-stationary
(Staircase)

pt
i
0

FedAWE (ours)
85.9 ± 0.8 %
85.6 ± 1.0 %
67.7 ± 1.3 %
66.0 ± 1.2 %
47.5 ± 2.0 %
46.9 ± 2.0 %
FedAvg over active
82.5 ± 1.0 %
82.4 ± 0.9 %
64.2 ± 1.8 %
63.0 ± 1.4 %
43.7 ± 2.0 %
42.3 ± 2.2 %
FedAvg over all
75.9 ± 2.1 %
75.9 ± 2.3 %
55.7 ± 2.1 %
55.4 ± 1.8 %
38.4 ± 2.0 %
37.9 ± 2.0 %
FedAU
83.6 ± 0.8 %
83.4 ± 0.8 %
65.2 ± 1.7 %
63.9 ± 1.5 %
45.7 ± 1.5 %
45.1 ± 1.5 %
F3AST
83.1 ± 0.6 %
83.1 ± 0.6 %
64.3 ± 1.1%
63.3 ± 0.9 %
45.2 ± 1.2 %
44.8 ± 1.2 %
FedAvg with known pt
i’s
85.8 ± 0.8 %
85.2 ± 0.9 %
68.0 ± 1.6 %
66.1 ± 1.8 %
45.0 ± 1.1 %
44.7 ± 1.0 %
MIFA (memory aided)
84.2 ± 0.5 %
84.0 ± 0.5 %
66.7 ± 0.7 %
65.3 ± 0.5 %
47.5 ± 0.5 %
46.9 ± 0.5 %
FedVARP (memory aided)
84.6 ± 0.2 %
84.3 ± 0.3 %
67.3 ± 0.3 %
66.1 ± 0.3 %
47.7 ± 0.2 %
47.2 ± 0.1 %

Non-stationary
(Sine)

pt
i
0

FedAWE (ours)
85.7 ± 0.9 %
85.6 ± 0.9 %
64.9 ± 1.9 %
63.5 ± 2.0 %
46.4 ± 2.4 %
45.8 ± 2.4 %
FedAvg over active
82.1 ± 1.1 %
82.0 ± 1.3 %
63.3 ± 1.9 %
62.1 ± 1.8 %
43.1 ± 2.5 %
42.6 ± 2.5 %
FedAvg over all
71.3 ± 2.5 %
71.3 ± 2.8 %
52.2 ± 2.4 %
52.1 ± 2.2 %
36.4 ± 2.0 %
36.0 ± 1.9 %
FedAU
82.5 ± 1.4 %
82.5 ± 1.3 %
64.2 ± 2.3 %
63.0 ± 1.9 %
44.4 ± 2.1 %
43.9 ± 2.1 %
F3AST
82.3 ± 1.0 %
82.3 ± 1.0 %
63.1 ± 1.7 %
62.3 ± 1.5 %
44.1 ± 1.6 %
43.7 ± 1.6 %
FedAvg with known pt
i’s
86.3 ± 1.0 %
86.0 ± 1.0 %
69.1 ± 1.2 %
67.3 ± 1.3 %
47.9 ± 1.5 %
47.4 ± 1.1 %
MIFA (memory aided)
84.2 ± 0.4 %
84.1 ± 0.4 %
66.6 ± 0.8 %
65.5 ± 0.6 %
47.4 ± 0.5 %
46.9 ± 0.4 %
FedVARP (memory aided)
84.5 ± 0.2 %
84.3 ± 0.1 %
67.4 ± 0.2 %
66.0 ± 0.3 %
47.7 ± 0.1 %
47.1 ± 0.2 %

Non-stationary
(Interleaved Sine)

pt
i
0

1

FedAWE (ours)
85.2 ± 1.6 %
84.6 ± 1.6 %
64.8 ± 3.1 %
63.3 ± 2.7 %
47.1 ± 2.7 %
46.6 ± 2.7 %
FedAvg over active
80.9 ± 1.7 %
80.7 ± 1.7 %
61.9 ± 2.4 %
60.7 ± 2.0 %
41.9 ± 2.7 %
41.5 ± 2.7 %
FedAvg over all
69.5 ± 3.4 %
69.5 ± 4.1 %
51.3 ± 2.7 %
51.3 ± 2.7 %
35.9 ± 2.0 %
35.6 ± 2.0 %
FedAU
82.6 ± 1.3 %
82.4 ± 1.1 %
63.9 ± 2.2 %
62.8 ± 1.8 %
44.2 ± 2.2 %
43.8 ± 2.1 %
F3AST
81.3 ± 1.2 %
81.3 ± 1.2 %
62.2 ± 2.1 %
61.3 ± 1.7 %
43.1 ± 2.2 %
42.7 ± 2.2 %
FedAvg with known pt
i’s
85.8 ± 1.2 %
85.2 ± 1.3 %
68.7 ± 2.1 %
66.5 ± 2.4 %
47.2 ± 2.3 %
46.8 ± 2.2 %
MIFA (memory aided)
83.8 ± 0.9 %
83.7 ± 0.8 %
65.8 ± 1.9 %
64.6 ± 1.6 %
46.5 ± 1.8 %
45.9 ± 1.7 %
FedVARP (memory aided)
84.5 ± 0.3 %
84.1 ± 0.5 %
67.3 ± 0.3 %
65.7 ± 0.2 %
47.7 ± 0.5 %
47.2 ± 0.3 %

Benchmark algorithms and discussions. We compare FedAWE with six baseline algorithms, includ-
ing FedAvg over active clients [31], FedAvg over all clients, FedAU [54], F3AST [43], FedAvg with
known pt
i’ [41], MIFA [13] and FedVARP [19]. The details of the algorithm and the additional results
are deferred to Appendix J. It is observed that FedAWE consistently outperforms the algorithms not
aided by memory or known statistics. Surprisingly, FedAWE occasionally beats MIFA and FedVARP,
which are memory-heavy. We attribute it to reuse of stored gradients from the unavailable clients.
Although FedAWE brings in stalenss due to implicit gossiping, our results (Table 8 in Appendix J)
indicate that there is no significant slowdown for FedAWE when compared to vanilla FedAvg, where
we study the first round to achieve a targeted accuracy by different algorithms. In addition, FedAWE at-
tains competitive or even better performance than FedAvg with known probability, yet unknown to
the underlying dynamics in client unavailability.

8
Conclusion

In this paper, we have shown that the impacts of heterogeneous and non-stationary client unavailability
can be significant through concrete examples on FedAvg. To address this, we have proposed an
algorithm FedAWE, which provably converges by adaptively echoing clients’ local improvement and
by evenly diffusing local updates through implicit gossiping. Theoretically, it achieves the desired
linear speedup property. Experiments have validated the superiority of FedAWE over state-of-the-art
algorithms under diversified non-stationary dynamics. Future work will investigate how to extend our
analysis to broader unavailability dynamics such as non-independent and non-stationary unavailability
and how to incorporate our findings into federated learning algorithms of different local optimization
methods.

10


---Page Break---
Acknowledgments and Disclosure of Funding

We gratefully acknowledge the support from the National Science Foundation under grants 2106891,
2107062, the National Science Foundation CAREER award under grant 2340482, and the Sony
Faculty Innovation Award. The research was sponsored by the Army Research Laboratory under
Cooperative Agreement Number W911NF-23-2-0014. The views and conclusions contained in this
document are those of the authors and should not be interpreted as representing the official policies,
either expressed or implied, of the Army Research Laboratory, the National Science Foundation,
or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints
for Government purposes notwithstanding any copyright notation herein. We also thank Connor J.
McLaughlin for valuable discussions and feedback on this work.

References

[1] Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafa¨el Pinot, and
John Stephan. Fixing by mixing: A recipe for optimal byzantine ml under heterogeneity. In
International Conference on Artificial Intelligence and Statistics, pages 1232–1300. PMLR,
2023.

[2] Dmitrii Avdiukhin and Shiva Kasiviswanathan. Federated learning under arbitrary communi-
cation patterns. In International Conference on Machine Learning, pages 425–435. PMLR,
2021.

[3] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir
Ivanov, Chloe Kiddon, Jakub Koneˇcn`y, Stefano Mazzocchi, Brendan McMahan, et al. Towards
federated learning at scale: System design. Proceedings of Machine Learning and Systems,
1:374–388, 2019.

[4] Stephen Boyd, Arpita Ghosh, Balaji Prabhakar, and Devavrat Shah. Randomized gossip
algorithms. IEEE Transactions on Information Theory, 52(6):2508–2530, 2006.

[5] Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh
Dhekane, Xavier Suau Cuadros, and Russell Webb. How to scale your ema. Advances in Neural
Information Processing Systems, 36, 2024.

[6] Jin-Hua Chen, Min-Rong Chen, Guo-Qiang Zeng, and Jia-Si Weng. Bdfl: a byzantine-fault-
tolerance decentralized federated learning method for autonomous vehicle. IEEE Transactions
on Vehicular Technology, 70(9):8639–8652, 2021.

[7] Wenlin Chen, Samuel Horv´ath, and Peter Richt´arik. Optimal client sampling for federated
learning. Transactions on Machine Learning Research, 2022.

[8] Yae Jee Cho, Pranay Sharma, Gauri Joshi, Zheng Xu, Satyen Kale, and Tong Zhang. On the
convergence of federated averaging with cyclic client participation. In Andreas Krause, Emma
Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors,
Proceedings of the 40th International Conference on Machine Learning, volume 202, pages
5677–5721. PMLR, 23–29 Jul 2023.

[9] Yae Jee Cho, Jianyu Wang, Tarun Chirvolu, and Gauri Joshi. Communication-efficient and
model-heterogeneous personalized federated learning via clustered knowledge transfer. IEEE
Journal of Selected Topics in Signal Processing, 2023.

[10] Yae Jee Cho, Jianyu Wang, and Gauri Joshi. Towards understanding biased client selection in
federated learning. In International Conference on Artificial Intelligence and Statistics, pages
10351–10375. PMLR, 2022.

[11] Luke N Darlow, Elliot J Crowley, Antreas Antoniou, and Amos J Storkey. Cinic-10 is not
imagenet or cifar-10. arXiv preprint arXiv:1810.03505, 2018.

[12] Morris H DeGroot. Reaching a consensus. Journal of the American Statistical association,
69(345):118–121, 1974.

11


---Page Break---
[13] Xinran Gu, Kaixuan Huang, Jingzhao Zhang, and Longbo Huang. Fast federated learning in the
presence of arbitrary device unavailability. Advances in Neural Information Processing Systems,
34:12052–12064, 2021.

[14] Allan Gut and Allan Gut. Probability: a graduate course, volume 200. Springer, 2006.

[15] John Hajnal and MS Bartlett. Weak ergodicity in non-homogeneous markov chains.
In
Mathematical Proceedings of the Cambridge Philosophical Society, volume 54, pages 233–246.
Cambridge Univ Press, 1958.

[16] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical
data distribution for federated visual classification. arXiv preprint arXiv:1909.06335, 2019.

[17] Xinmeng Huang, Yiming Chen, Wotao Yin, and Kun Yuan. Lower bounds and nearly optimal
algorithms in distributed learning with communication compression. In Alice H. Oh, Alekh
Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information
Processing Systems, 2022.

[18] Mark Jerrum and Alistair Sinclair. Conductance and the rapid mixing property for markov
chains: the approximation of permanent resolved. In Proceedings of the Twentieth Annual ACM
Symposium on Theory of Computing, pages 235–244, 1988.

[19] Divyansh Jhunjhunwala, Pranay Sharma, Aushim Nagarkatti, and Gauri Joshi. Fedvarp: Tack-
ling the variance due to partial client participation in federated learning. In Uncertainty in
Artificial Intelligence, pages 906–916. PMLR, 2022.

[20] Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aur´elien Bellet, Mehdi Bennis, Ar-
jun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings,
Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner,
Zachary Garrett, Adri`a Gasc´on, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Har-
chaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara
Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecn´y, Aleksandra Korolova, Farinaz Koushanfar,
Sanmi Koyejo, Tancr`ede Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock,
Ayfer ¨Ozg¨ur, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn
Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tram`er,
Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu,
and Sen Zhao. Advances and open problems in federated learning. Foundations and Trends®
in Machine Learning, 14(1–2):1–210, 2021.

[21] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Byzantine-robust learning on heteroge-
neous datasets via bucketing. In International Conference on Learning Representations. PMLR,
2022.

[22] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pages 5132–5143. PMLR, 2020.

[23] David Kempe, Alin Dobra, and Johannes Gehrke. Gossip-based computation of aggregate
information. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003.
Proceedings., pages 482–491. IEEE, 2003.

[24] Anastasiia Koloskova, Sebastian U Stich, and Martin Jaggi. Sharper convergence guarantees
for asynchronous sgd for distributed and federated learning. Advances in Neural Information
Processing Systems, 35:17202–17215, 2022.

[25] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.

[26] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia
Smith. Federated optimization in heterogeneous networks. Proceedings of Machine Learning
and Systems, 2:429–450, 2020.

12


---Page Break---
[27] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia
Smithy. Feddane: A federated newton-type method. In 2019 53rd Asilomar Conference on
Signals, Systems, and Computers, pages 1227–1231. IEEE, 2019.

[28] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence
of fedavg on non-iid data. In International Conference on Learning Representations, 2020.

[29] Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized
algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic
gradient descent. Advances in Neural Information Processing Systems, 30, 2017.

[30] Nancy A. Lynch. Distributed Algorithms. Morgan Kaufmann Publishers Inc., San Francisco,
CA, USA, 1996.

[31] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-efficient learning of deep networks from decentralized data. In Artificial
Intelligence and Statistics, pages 1273–1282. PMLR, 2017.

[32] Konstantin Mishchenko, Francis Bach, Mathieu Even, and Blake E Woodworth. Asynchronous
sgd beats minibatch sgd under arbitrary delays. Advances in Neural Information Processing
Systems, 35:420–433, 2022.

[33] Angelia Nedi´c, Alex Olshevsky, and Michael G Rabbat. Network topology and communication-
computation tradeoffs in decentralized optimization. Proceedings of the IEEE, 106(5):953–976,
2018.

[34] Angelia Nedic, Alex Olshevsky, and Wei Shi. Achieving geometric convergence for distributed
optimization over time-varying graphs. SIAM Journal on Optimization, 27(4):2597–2633, 2017.

[35] Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent opti-
mization. IEEE Transactions on Automatic Control, 54(1):48–61, 2009.

[36] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng.
Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on
Deep Learning and Unsupervised Feature Learning, 2011.

[37] John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek,
and Dzmitry Huba. Federated learning with buffered asynchronous aggregation. In International
Conference on Artificial Intelligence and Statistics, pages 3581–3607. PMLR, 2022.

[38] Thien Duc Nguyen, Samuel Marchal, Markus Miettinen, Hossein Fereidooni, N Asokan, and
Ahmad-Reza Sadeghi. D¨ıot: A federated self-learning anomaly detection system for iot. In
2019 IEEE 39th International Conference on Distributed Computing Systems (ICDCS), pages
756–767. IEEE, 2019.

[39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
style, high-performance deep learning library. Advances in Neural Information Processing
Systems, 32, 2019.

[40] Muzi Peng, Jiangwei Wang, Dongjin Song, Fei Miao, and Lili Su. Privacy-preserving and
uncertainty-aware federated trajectory prediction for connected autonomous vehicles. In The
2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2023).
IEEE/RSJ, 2023.

[41] Jake Perazzone, Shiqiang Wang, Mingyue Ji, and Kevin S Chan. Communication-efficient
device scheduling for federated learning using stochastic optimization. In IEEE INFOCOM
2022-IEEE Conference on Computer Communications, pages 1449–1458. IEEE, 2022.

[42] Swaroop Ramaswamy, Rajiv Mathews, Kanishka Rao, and Franc¸oise Beaufays. Federated
learning for emoji prediction in a mobile keyboard. arXiv preprint arXiv:1906.04329, 2019.

[43] M´onica Ribero, Haris Vikalo, and Gustavo De Veciana. Federated learning under intermittent
client availability and time-varying communication constraints. IEEE Journal of Selected Topics
in Signal Processing, 17(1):98–111, 2022.

13


---Page Break---
[44] Yichen Ruan, Xiaoxi Zhang, Shu-Che Liang, and Carlee Joe-Wong. Towards flexible device
participation in federated learning. In International Conference on Artificial Intelligence and
Statistics, pages 3403–3411. PMLR, 2021.

[45] Devavrat Shah et al. Gossip algorithms. Foundations and Trends® in Networking, 3(1):1–125,
2009.

[46] Artin Spiridonoff, Alex Olshevsky, and Ioannis Ch Paschalidis. Robust asynchronous stochastic
gradient-push: Asymptotically optimal and network-independent performance for strongly
convex functions. Journal of Machine Learning Research, 21(58), 2020.

[47] Sebastian U. Stich. Local SGD converges fast and communicates little. In International
Conference on Learning Representations, 2019.

[48] Mohammad Taha Toghani and C´esar A Uribe. Unbounded gradients in federated learning with
buffered asynchronous aggregation. In 2022 58th Annual Allerton Conference on Communica-
tion, Control, and Computing (Allerton), pages 1–8. IEEE, 2022.

[49] David Tse and Pramod Viswanath. Fundamentals of wireless communication. Cambridge
university press, 2005.

[50] Jianyu Wang and Gauri Joshi. Cooperative sgd: A unified framework for the design and analysis
of local-update sgd algorithms. The Journal of Machine Learning Research, 22(1):9709–9758,
2021.

[51] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective
inconsistency problem in heterogeneous federated optimization. Advances in Neural Information
Processing Systems, 33:7611–7623, 2020.

[52] Jianyu Wang, Anit Kumar Sahu, Gauri Joshi, and Soummya Kar. Matcha: A matching-based
link scheduling strategy to speed up distributed optimization. IEEE Transactions on Signal
Processing, 70:5208–5221, 2022.

[53] Shiqiang Wang and Mingyue Ji. A unified analysis of federated learning with arbitrary client
participation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors,
Advances in Neural Information Processing Systems, 2022.

[54] Shiqiang Wang and Mingyue Ji. A lightweight method for tackling unknown participation
statistics in federated averaging. In The Twelfth International Conference on Learning Repre-
sentations, 2024.

[55] Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He,
and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems.
IEEE journal on selected areas in communications, 37(6):1205–1221, 2019.

[56] Ming Wen, Chengchang Liu, and Yuedong Xu. Communication efficient distributed new-
ton method over unreliable networks. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 15832–15840, 2024.

[57] Ming Xiang, Stratis Ioannidis, Edmund Yeh, Carlee Joe-Wong, and Lili Su. Towards bias
correction of fedavg over nonuniform and time-varying communications. In 2023 62nd IEEE
Conference on Decision and Control (CDC), pages 6719–6724, 2023.

[58] Cong Xie, Sanmi Koyejo, and Indranil Gupta. Asynchronous federated optimization. arXiv
preprint arXiv:1903.03934, 2019.

[59] Yikai Yan, Chaoyue Niu, Yucheng Ding, Zhenzhe Zheng, Shaojie Tang, Qinya Li, Fan Wu,
Chengfei Lyu, Yanghe Feng, and Guihai Chen. Federated optimization under intermittent client
availability. INFORMS Journal on Computing, 2023.

[60] Haibo Yang, Minghong Fang, and Jia Liu. Achieving linear speedup with partial worker partici-
pation in non-IID federated learning. In International Conference on Learning Representations,
2021.

14


---Page Break---
[61] Haibo Yang, Xin Zhang, Prashant Khanduri, and Jia Liu. Anarchic federated learning. In
International Conference on Machine Learning, pages 25331–25363. PMLR, 2022.

[62] Timothy Yang, Galen Andrew, Hubert Eichner, Haicheng Sun, Wei Li, Nicholas Kong, Daniel
Ramage, and Franc¸oise Beaufays. Applied federated learning: Improving google keyboard
query suggestions. arXiv preprint arXiv:1812.02903, 2018.

[63] Hao Ye, Le Liang, and Geoffrey Ye Li. Decentralized federated learning with unreliable
communications. IEEE Journal of Selected Topics in Signal Processing, 16(3):487–500, 2022.

[64] Hao Yu, Rong Jin, and Sen Yang. On the linear speedup analysis of communication efficient
momentum sgd for distributed non-convex optimization. In International Conference on
Machine Learning, pages 7184–7193. PMLR, 2019.

[65] Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less
communication: Demystifying why model averaging works for deep learning. In Proceedings
of the AAAI Conference on Artificial Intelligence, volume 33, pages 5693–5700, 2019.

[66] Kun Yuan, Qing Ling, and Wotao Yin. On the convergence of decentralized gradient descent.
SIAM Journal on Optimization, 26(3):1835–1854, 2016.

[67] Xiaotong Yuan and Ping Li.
On convergence of fedprox: Local dissimilarity invariant
bounds, non-smoothness and beyond. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave,
and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[68] Hossein Zakerinia, Shayan Talaei, Giorgi Nadiradze, and Dan Alistarh. Communication-efficient
federated learning with data and client heterogeneity. In International Conference on Artificial
Intelligence and Statistics, pages 3448–3456. PMLR, 2024.

[69] Tengchan Zeng, Omid Semiari, Mingzhe Chen, Walid Saad, and Mehdi Bennis. Federated
learning on the road autonomous controller design for connected and autonomous vehicles.
IEEE Transactions on Wireless Communications, 21(12):10407–10423, 2022.

15


---Page Break---
Appendices

Here, we provide an overview of the Appendix. In particular, the proofs of the main results are
presented and backed by supporting lemmas and propositions.

A Limitations
17

B
Broader Impacts
17

C Nomenclatures
17

D Useful Inequalities
19

E
Descent Lemma (Lemma 3)
20

E.1
Multi-step perturbation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

E.2
Descent lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

F
Intermediate Results
26

F.1
Bounding local and global dissimilarity
. . . . . . . . . . . . . . . . . . . . . . .
26

F.2
Weight re-equalization (Proposition 1) . . . . . . . . . . . . . . . . . . . . . . . .
26

F.3
Unavailable statistics (Lemma 2) . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

F.4
Auxiliary sequence construction and properties (Proposition 2) . . . . . . . . . . .
27

F.5
Consensus error of the auxiliary sequence . . . . . . . . . . . . . . . . . . . . . .
29

F.6
Spectral norm upper bound (Lemma 4) . . . . . . . . . . . . . . . . . . . . . . . .
34

G Convergence Error of ¯zt (Theorem 1)
35

H Convergence Rate of ¯xt (Corollary 1)
38

H.1
Convergence error of Algorithm 1
. . . . . . . . . . . . . . . . . . . . . . . . . .
38

H.2
Convergence rate of Algorithm 1 . . . . . . . . . . . . . . . . . . . . . . . . . . .
39

I
Additional Results and Interpretations
40

I.1
Consensus error of Algorithm 1
. . . . . . . . . . . . . . . . . . . . . . . . . . .
40

I.2
Orders of the asymptotic rates
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
41

J
Numerical Experiments
42

J.1
Code . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42

J.2
Experimental setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42

J.3
Non-stationary client unavailability dynamics . . . . . . . . . . . . . . . . . . . .
43

J.4
Additional results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44

16


---Page Break---
A
Limitations

The limitations of our work are two-fold:

1. The client unavailability dynamics are assumed to be independent and strictly positive across
clients and rounds. While deriving guarantees is generally challenging without assuming indepen-
dence and positivity (see Section 3), it is interesting to explore how to relax the client unavailability
dynamics, where the probabilities can potentially have arbitrary trajectories.

2. Our study focuses on heterogeneous and non-stationary client unavailability in federated learn-
ing, which may vary greatly due to its inherent uncontrollable nature.
Although we have
shown FedAWE provably converges to a stationary point of even non-convex objectives, an inter-
esting yet challenging future direction is to incorporate variance reduction techniques for a more
robust update.

B
Broader Impacts

Federated learning has become the main trend for distributed learning in recent years and has
empowered commercial industries such as autonomous vehicles, the Internet of Things, and natural
language processing. Our paper focuses on the practical implementation of federated learning systems
in the real world and has significantly advanced the theory and algorithms for federated learning
by bringing together insights from statistics, optimization, distributed computing and engineering
practices. In addition, our research is important for federated learning systems to expand their
outreach to more undesirable deployment environments. We are unaware of any potential negative
social impacts of our work.

C
Nomenclatures

In this section, we provide the notations and nomenclatures used throughout our proofs for a
comprehensive presentation. However, it is worth noting that all notations have been properly
introduced before their first use. We next articulate the missing definitions and equation formulas.

Table 3: Notation table
∥v∥2
The l2 norm of a given vector v.

∥A∥F
The Frobenius norm of a given matrix A.

Ft
The sigma algebra generated by randomness up to round t.

λ2(A)
The second largest eigenvalue of a square matrix A.

Rd
A d-dimensional vector space, where d denotes the dimension.

[m]
A set {k | k ∈N, k ∈[1, m]}.

1{E}
An indicator function of event E, i.e., 1{E} = 1 when event E occurs, but 1{E} = 0
otherwise.

≲
f(n) ≲g(n), if there exists a constant co > 0 and an integer n0 ∈N, f(n) ≤cog(n)
for all n ≥n0.

≍
f(n) ≍g(n), if there exists a constant cΘ > 0 and an integer n0 ∈N, f(n) = cΘg(n)
for all n ≥n0.

Missing definitions and equation formulas.

17


---Page Break---
Table 4: Algorithmic nomenclature table
At
The set of active clients in round t.

W t
A doubly stochastic matrix to capture the information mixing error. Its definition can
be found in (4).

τi(t)
τi(t) ≜sup{t′ | t′ < t, i ∈At′} defines client i’s most recent active round. In
particular, τi(0) = −1 for all i ∈[m].

xt
i
The real model at client i at the beginning of round t in Algorithm 1.

zt
i
The auxiliary model at client i at the beginning of round t. Refer to Definition 1 for
more details. The sequence is for analysis only and is not computed by any clients.

xt
The aggregated real model at the end of round t −1 in Algorithm 1.

zt
The auxiliary model at the end of round t −1.

xt†
i ,
zt†
i

The real model of an active client i, and auxiliary model of an active client i after s-step
local computation in round t, respectively. Refer to Algorithm 1 for more details.

x(t,r)
i
The real model at client i after r-step local computation.

¯xt, ¯zt
The real and auxiliary model mean over all clients in a distributed system and in round
t, respectively.

Fi(x)
The local objective function at client i, which is assumed to be non-convex.

F(x)
The global objective function defined in (1): F(x) ≜Pm
i=1 Fi(x)/m.

∇ℓi(x)
The local stochastic gradient function at client i taken with respect to x.

∇Fi(x) The local true gradient function at client i taken with respect to x.

Di
Client i’s local data distribution.

ξi
An independent stochastic sample drawn from client i’s local distribution Di.

Table 5: Variable table
L
Lipschitz constant in Assumption 2.

σ2
The upper bound of the stochastic gradient variance.

(β, ζ)
Parameters that capture the averaged gradient dissimilarity between global and local
objectives.

ρ
The spectral norm of a stochastic matrix in expectation.

s
The number of local computation steps.

m
The number of clients in the federated learning system.

The iterate of zi when i ∈At−1.

zt
i =
1
|At−1|

X

j∈At−1

 

zt−1
j
−ηlηg

s−1
X

r=0
∇ℓj(x(t−1,r)
j
; ξ(t,r)
i
)

!

+
ηlηg
|At−1|

X

j∈At−1
(t −2 −τj(t −1))

s−1
X

r=0


∇Fj(xτj(t−1)+1
j
) −∇ℓj(x(t−1,r)
j
; ξ(t,r)
i
)

. (18)

18


---Page Break---
Local parameter innovation e
Gt of the auxiliary sequence.

e
Gt
i ≜1{i∈At}

"

(t −τi(t))

s−1
X

r=0
∇ℓi(x(t,r)
i
) −s (t −1 −τi(t)) ∇Fi(xτi(t)+1
i
)

#

+ 1{i/∈At}s∇Fi(xτi(t)+1
i
)

= 1{i∈At} (t −τi(t))

s−1
X

r=0


∇ℓi(x(t,r)
i
) −∇Fi(xt
i)

+ s∇Fi(xt
i),
(19)

where the last equality holds because xt
i = xτi(t)+1
i
and re-grouping.

Decomposition in the Proof of Lemma 6.
The local parameter innovation of the auxiliary sequence
e
Gt can be decomposed as e
Gt ≜e∆t + ∆t + s∇F t
x. Detailed definitions can be found below.

• [e∆t]i ≜1{i∈At}(t −τi(t)) Ps−1
r=0

∇ℓi(x(t,r)
i
; ξ(t,r)
i
) −∇Fi(x(t,r)
i
)

;

• [∆t]i ≜1{i∈At}(t −τi(t)) Ps−1
r=0

∇Fi(x(t,r)
i
) −∇Fi(xt
i)

;

• [∇F t
x]i ≜∇Fi(xt
i).

D
Useful Inequalities

For completeness and for ease of exposition, we present some common inequalities that will be
frequently used in our proofs.

The followings hold for any ai ∈Rd and any i ∈[m].

1. Jensen’s inequality.

1
m

m
X

i=1
ai



2

2
≤1

m

m
X

i=1
∥ai∥2
2
and



m
X

i=1
ai



2

2
≤m

m
X

i=1
∥ai∥2
2 .
(20)

2. Young’s inequality (a.k.a. Peter-Paul inequality).

⟨a1, a2⟩≤∥a1∥2
2
2ϵ
+ ϵ ∥a2∥2
2
2
,
for any ϵ > 0.
(21)

Equivalently, we have

∥a1 + a2∥2
2 = ∥a1∥2
2 + ∥a2∥2
2 + 2 ⟨a1, a2⟩

≤

1 + 1

ϵ


∥a1∥2
2 + (1 + ϵ) ∥a2∥2
2 ,
for any ϵ > 0.
(22)

3. Smoothness corollary. Given Assumption 2, it holds that

F(a1) −F(a2) =

a1 −a2,
Z 1

0
∇F(a2 + τ(a1 −a2))dτ


= ⟨∇F(a2), a1 −a2⟩+
Z 1

0
⟨a1 −a2, ∇F(a2 + τ(a1 −a2)) −∇F(a2)⟩dτ

(a)
≤⟨∇F(a2), a1 −a2⟩+ L
Z 1

0
τ ∥a1 −a2∥2 ∥(a1 −a2)∥2 dτ

≤⟨∇F(a2), a1 −a2⟩+ L

2 ∥a1 −a2∥2
2 ,
(23)

where (a) follows from Cauchy-Schwartz inequality and Assumption 2.

19


---Page Break---
E
Descent Lemma (Lemma 3)

In this section, we first present a bound on multi-step local computation. Then, we apply the bound
to the analysis of descent lemma.

E.1
Multi-step perturbation

Lemma 5. For s ≥1 and under Assumption 2, 3 and ηl ≤1/(4sL) , we have

E







s−1
X

r=0
∇Fi(x(t,r)
i
) −∇Fi(xt
i)



2

2

 Ft



≤4η2
l s3L2σ2 + 16η2
l s4L2 ∇Fi(xt
i)
2
2

Proof of Lemma 5. The proof shares a similar road map to [60, Lemma 2], but the objective is
instead to show an upper bound with respect to ∥∇Fi(xt
i)∥2
2.

For s ≥1, it holds that

E







s−1
X

r=0
∇Fi(x(t,r)
i
) −∇Fi(xt
i)



2

2

 Ft




(a)
≤s

s−1
X

r=0
E
∇Fi(x(t,r)
i
) −∇Fi(xt
i)

2

2

 Ft


(b)
≤sL2
s−1
X

r=0
E
x(t,r)
i
−xt
i

2

2

 Ft

,
(24)

where inequality (a) holds because of Jensen’s inequality, inequality (b) holds because of Assump-
tion 2. It remains to bound E[∥x(t,r)
i
−xt
i∥2 | Ft]. In what follows, we use ∇ℓ(t,k)
i
to denote
∇ℓi(x(t,k)
i
) and ∇F (t,k)
i
as ∇Fi(x(t,k)
i
), respectively, for ease of presentation.

E
x(t,r)
i
−xt
i

2

2

 Ft

= E
x(t,r−1)
i
−xt
i −ηl∇ℓ(t,r−1)
i

2

2

 Ft


= E
−ηl

∇ℓ(t,r−1)
i
−∇F (t,r−1)
i

+ x(t,r−1)
i
−xt
i −ηl

∇F (t,r−1)
i
−∇F t
i + ∇F t
i

2

2

 Ft


(c)
= η2
l E
∇ℓ(t,r−1)
i
−∇F (t,r−1)
i

2

2

 Ft

+ E
x(t,r−1)
i
−xt
i −ηl

∇F (t,r−1)
i
−∇F t
i + ∇F t
i

2

2

 Ft


(d)
≤η2
l E
∇ℓ(t,r−1)
i
−∇F (t,r−1)
i

2

2

 Ft


+

1 +
1
2s −1


E
x(t,r−1)
i
−xt
i

2

2

 Ft

+ 2sη2
l E
∇F (t,r−1)
i
−∇F t
i + ∇F t
i

2

2

 Ft


≤η2
l E
∇ℓ(t,r−1)
i
−∇F (t,r−1)
i

2

2

 Ft


+

1 +
1
2s −1


E
x(t,r−1)
i
−xt
i

2

2

 Ft

+ 4sη2
l E
∇F (t,r−1)
i
−∇F t
i

2

2

 Ft

+ 4sη2
l
∇F t
i
2
2

(e)
≤η2
l σ2 + 4sη2
l
∇F t
i
2
2

+

1 +
1
2s −1


E
x(t,r−1)
i
−xt
i

2

2

 Ft

+ 4sL2η2
l E
x(t,r−1)
i
−xt
i

2

2

 Ft


= η2
l σ2 + 4sη2
l
∇F t
i
2
2 +

1 +
1
2s −1 + 4sL2η2
l


E
x(t,r−1)
i
−xt
i

2

2

 Ft

,

where equality (c) holds because ∇ℓ(t,k)
i
is an unbiased estimator of ∇F (t,r)
i
, inequality (d) holds
because of Young’s inequality, inequality (e) holds because of Assumption 2.

By ηl ≤
1
4sL, it holds that

1
2s −1 + 4sL2η2
l ≤
1
2s −1 + 1

4s ≤
2
2s −1.

20


---Page Break---
Unroll the recursion, we have

E
x(t,r)
i
−xt
i

2

2

 Ft

≤

r−1
X

k=0


1 +
2
2s −1

k 
η2
l σ2 + 4sη2
l
∇F t
i
2
2



≤

s−1
X

k=0


1 +
2
2s −1

k 
η2
l σ2 + 4sη2
l
∇F t
i
2
2



= 2s −1

2

"
1 +
2
2s −1

s−1

2 
1 +
2
2s −1

 1

2
−1

# 
η2
l σ2 + 4sη2
l
∇F t
i
2
2



(f)
≤

s −1

2

 h√

3e −1
i 
η2
l σ2 + 4sη2
l
∇F t
i
2
2



(g)
≤4sη2
l σ2 + 16s2η2
l
∇F t
i
2
2 ,

where inequality (f) holds because of (1 + 1/x)x < exp(1), inequality (g) holds because of
√

3 exp(1) −1 < 4. Plug it back into (24), we have the desired result

E







s−1
X

r=0
∇Fi(x(t,r)
i
) −∇Fi(xt
i)



2

2

 Ft



≤4η2
l s3L2σ2 + 16η2
l s4L2 ∇Fi(xt
i)
2
2 .

E.2
Descent lemma

Proof of Lemma 3. By Assumption 2 and inequality (23), we have

F(¯zt+1) −F(¯zt) ≤

∇F(¯zt), ¯zt+1 −¯zt

|
{z
}
(A)

+ L

2

¯zt+1 −¯zt2
2
|
{z
}
(B)

.

The one-round innovation of ¯z can be rewritten as

¯zt+1 −¯zt = 1

m

X

i∈At


zt†
i −zt
i

+ 1

m

X

i/∈At

 
zt+1
i
−zt
i


= 1

m

m
X

i=1
1{i∈At}



ηlηgs

t−1
X

k=τi(t)+1
∇Fi(xk
i ) −ηlηg(t −τi(t))

s−1
X

r=0
∇ℓi(x(t,r)
i
; ξ(t,r)
i
)





−ηlηgs

m

m
X

i=1
1{i/∈At}∇Fi(xt
i)

(a)
= 1

m

m
X

i=1
1{i∈At}ηlηgs(t −1 −τi(t))∇Fi(xt
i) −1

m

m
X

i=1
1{i∈At}ηlηg(t −τi(t))

s−1
X

r=0
∇ℓi(x(t,r)
i
; ξ(t,r)
i
)

−ηlηgs

m

m
X

i=1
1{i/∈At}∇Fi(xt
i)

(b)
= ηlηg

m

m
X

i=1
1{i∈At}(t −τi(t))

s−1
X

r=0


∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)


+ ηlηg

m

m
X

i=1
1{i∈At}(t −τi(t))

s−1
X

r=0


∇Fi(xt
i) −∇Fi(x(t,r)
i
)


−ηlηgs

m

m
X

i=1
∇Fi(xt
i),

21


---Page Break---
where equality (a) using the fact that xk
i = xt
i for all k such that τi(t) + 1 ≤k ≤t, and equality (b)
is obtained by adding and subtracting ∇ℓi(xt
i; ξ(t,r)
i
) and by the fact that
 
1{i∈At} + 1{i/∈At}

= 1.

Bounding (A).

(A) =

∇F(¯zt), ¯zt+1 −¯zt

= ηlηg

*

∇F(¯zt), 1

m

m
X

i=1
1{i∈At}

t−1
X

p=−1
1{τi(t)=p}(t −p)

s−1
X

r=0


∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)
+

|
{z
}
(A.I)

+ ηlηg

m

m
X

i=1
1{i∈At}

t−1
X

p=−1
1{τi(t)=p}

*

∇F(¯zt), (t −p)

s−1
X

r=0


∇Fi(xt
i) −∇Fi(x(t,r)
i
)
+

|
{z
}
(A.II)

+ ηlηgs

m

m
X

i=1


∇F(¯zt), ∇Fi(zt
i) −∇Fi(xt
i)


|
{z
}
(A.III)

−ηlηgs

*

∇F(¯zt), 1

m

m
X

i=1
∇Fi(zt
i)

+

|
{z
}
(A.IV)

.

Bounding (A.I)

E
h
(A.I)
F ti

(a)
= ηlηgE

"

E

"*

∇F(¯zt), 1

m

m
X

i=1
1{i∈At}

t−1
X

p=−1
1{τi(t)=p}(t −p)

s−1
X

r=0


∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)
+ x(t,r)
i
, F t
# F t
#

(b)
= ηlηg

∇F(¯zt),

1
m

m
X

i=1
E
h
1{i∈At}
F ti
t−1
X

p=−1
1{τi(t)=p}(t −p)

s−1
X

r=0
E
h
E
h
∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)
 x(t,r)
i
, F ti F ti+

= 0,

where equality (a) holds because of the law of total expectation, equality (b) holds because 1{i∈At}
is by definition independent of others and Assumption 3.

Bounding (A.II)

(A.II)
(c)
≤ηlηg

m

m
X

i=1
1{i∈At}

t−1
X

p=−1
1{τi(t)=p}



s

8

∇F(¯zt)
2
2 + 2(t −p)2

s



s−1
X

r=0
∇Fi(xt
i) −∇Fi(x(t,r)
i
)



2

2





= ηlηgs

8m

m
X

i=1
1{i∈At}
∇F(¯zt)
2
2

+ ηlηg

m

m
X

i=1
1{i∈At}

t−1
X

p=−1
1{τi(t)=p}
2(t −p)2

s



s−1
X

r=0
∇Fi(xt
i) −∇Fi(x(t,r)
i
)



2

2
,

22


---Page Break---
where inequality (c) holds because of Young’s inequality. It follows that

E
h
(A.II)
Fti (d)
≤ηlηgs

8

∇F(¯zt)
2
2 + 8ηgη3
l s2L2σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 32ηgη3
l s3L2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xt
i)
2
2

= ηlηgs

8

∇F(¯zt)
2
2 + 8ηgη3
l s2L2σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 32ηgη3
l s3L2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2 ,

where inequality (d) holds because of Lemma 5, the last equality using the fact that xk
i = xt
i for all
k such that τi(t) + 1 ≤k ≤t.

Bounding (A.III).

(A.III) = ηlηgs

m

m
X

i=1


∇F(¯zt), ∇Fi(zt
i) −∇Fi(xt
i)
 (e)
≤ηlηgs

8

∇F(¯zt)
2
2 + 2ηlηgsL2

m

m
X

i=1

zt
i −xt
i
2
2 ,

where inequality (e) follows from Young’s inequality and Assumption 2. It holds that,

E
h
(A.III)
Fti
≤ηlηgs

8

∇F(¯zt)
2
2 + 2ηlηgsL2

m

m
X

i=1

zt
i −xt
i
2
2 .

Bounding (A.IV)

(A.IV) = ηlηgs

2



∇F(¯zt)
2
2 +


1
m

m
X

i=1
∇Fi(zt
i)



2

2
−

∇F(¯zt) −1

m

m
X

i=1
∇Fi(zt
i)



2

2



,

where the equality follows from the identity in Appendix D (3). It holds that

E
h
(A.IV)
Fti
= ηlηgs

2



∇F(¯zt)
2
2 +


1
m

m
X

i=1
∇Fi(zt
i)



2

2
−


1
m

m
X

i=1
∇Fi(¯zt) −1

m

m
X

i=1
∇Fi(zt
i)



2

2





≥ηlηgs

2



∇F(¯zt)
2
2 +


1
m

m
X

i=1
∇Fi(zt
i)



2

2
−L2

m

m
X

i=1

¯zt −zt
i
2
2



.

Putting (A) together,

E
h
(A)
Fti
≤−ηlηgs

4

∇F(¯zt)
2
2 + 8ηgη3
l s2L2σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 2ηlηgsL2

m

m
X

i=1

xt
i −zt
i
2
2 + ηlηgsL2

2m

m
X

i=1

¯zt −zt
i
2
2

−ηlηgs

2


1
m

m
X

i=1
∇Fi(zt
i)



2

2
+ 32ηgη3
l s3L2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2 .

23


---Page Break---
Bounding (B).

(B) ≤2Lη2
l η2
g
m2



m
X

i=1
1{i∈At}(t −τi(t))

s−1
X

r=0


∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)


2

2
|
{z
}
(B.I)

+ 2Lη2
l η2
g
m2 m

m
X

i=1
1{i∈At}(t −τi(t))2


s−1
X

r=0


∇Fi(xt
i) −∇Fi(x(t,r)
i
)


2

2
|
{z
}
(B.II)

+ 2Lη2
l η2
gs2

m2
m

m
X

i=1

∇Fi(xt
i) −∇Fi(zt
i)
2
2
|
{z
}
(B.III)

+ 2Lη2
l η2
gs2

1
m

m
X

i=1
∇Fi(zt
i)



2

2
|
{z
}
(B.IV)

Bounding (B.I)
Recall that δmax ≜supi∈[m],t∈[T ] pt
i. It holds that,

E
h
(B.I)
F ti (f)
= 2Lη2
l η2
g
m2

m
X

i=1
E
h
1{i∈At}
F ti
(t −τi(t))2
s−1
X

r=0
E

E
∇Fi(x(t,r)
i
) −∇ℓi(x(t,r)
i
; ξ(t,r)
i
)

2

2

x(t,r)
i
, F t
 F t


(g)
≤2η2
l η2
gsLδmaxσ2

m2

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2,

where equality (f) holds by the law of total expectation and by the independence of event {i ∈At},
inequality (g) holds because of Assumption 3 and by definition pt
i ≤δmax.

Bounding (B.II)
We have,

E
h
(B.II)
Fti
≤2Lη2
l η2
g
m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)24η2
l s3L2σ2

+ 2Lη2
l η2
g
m

m
X

i=1
1{τi(t)=p}

t−1
X

p=−1
(t −p)216η2
l s4L2 ∇Fi(xt
i)
2
2

= 8η2
gη4
l s3L3σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 + 32η2
gη4
l s4L3

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2 ,

where the last equality using the fact that xk
i = xt
i for all k such that τi(t) + 1 ≤k ≤t.

Bounding (B.III).
E
h
(B.III)
Fti
≤
2η2
l η2
gs2L3

m
Pm
i=1 ∥xt
i −zt
i∥2
2 .

Putting (B) together, we get

E
h
(B)
Fti
≤2η2
l η2
gsLδmaxσ2

m2

t−1
X

p=−1
1{τi(t)=p}(t −p)2 + 8η2
gη4
l s3L3σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 32η2
gη4
l s4L3

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2

+ 2η2
l η2
gs2L3

m

m
X

i=1

xt
i −zt
i
2
2 + 2Lη2
l η2
gs2

1
m

m
X

i=1
∇Fi(zt
i)



2

2
.

24


---Page Break---
Now, everything:

E
h
F(¯zt+1) −F(¯zt)
Fti
≤−ηlηgs

4

∇F(¯zt)
2
2

−ηlηgs

2
(1 −4Lηlηgs)


1
m

m
X

i=1
∇Fi(zt
i)



2

2

+ 2η2
l η2
gsLδmaxσ2

m2

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 8ηgη3
l s2L2 (1 + ηgηlsL) σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 2ηlηgsL2 (1 + ηlηgsL) 1

m

m
X

i=1

xt
i −zt
i
2
2 + ηlηgsL2

2m

m
X

i=1

zt
i −¯zt2
2

+ 32ηgη3
l s3L2 (1 + ηgηlsL) 1

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2

≤−ηlηgs

4

∇F(¯zt)
2
2 + 2η2
l η2
gsLδmaxσ2

m2

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 9ηgη3
l s2L2σ2

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2

+ 2.2ηlηgsL2 1

m

m
X

i=1

xt
i −zt
i
2
2 + ηlηgsL2

2m

m
X

i=1

zt
i −¯zt2
2

+ 35ηgη3
l s3L2 1

m

m
X

i=1

t−1
X

p=−1
1{τi(t)=p}(t −p)2 ∇Fi(xp+1
i
)

2

2 ,

where the last inequality holds because ηlηg ≤
9
100sL and that
 1

m
Pm
i=1 ∇Fi(zt
i)
2
2 ≥0.

25


---Page Break---
F
Intermediate Results

In this section, we present the intermediate results that serve as handy tools in building up our proofs
afterwards.

F.1
Bounding local and global dissimilarity

Proposition 3. For any t, it holds that

1
m

m
X

i=1

∇Fi(zt
i)
2
2 ≤3L2

m

m
X

i=1

zt
i −¯zt2
2 + 3
 
β2 + 1
 ∇F(¯zt)
2
2 + 3ζ2.

Proof of Proposition 3.

1
m

m
X

i=1

∇Fi(zt
i)
2
2 = 1

m

m
X

i=1

∇Fi(zt
i) −∇Fi(¯zt) + ∇Fi(¯zt) −∇F(¯zt) + ∇F(¯zt)
2
2

≤3

m

m
X

i=1

∇Fi(zt
i) −∇Fi(¯zt)
2
2 + 3

m

m
X

i=1

∇Fi(¯zt) −∇F(¯zt)
2
2 + 3
∇F(¯zt)
2
2

(a)
≤3L2

m

m
X

i=1

zt
i −¯zt2
2 + 3β2 ∇F(¯zt)
2
2 + 3ζ2 + 3
∇F(¯zt)
2
2

= 3L2

m

m
X

i=1

zt
i −¯zt2
2 + 3
 
β2 + 1
 ∇F(¯zt)
2
2 + 3ζ2,

where inequality (a) follows from Assumptions 2 and 4.

F.2
Weight re-equalization (Proposition 1)

Proof of Proposition 1. We show Proposition 1 by induction.

When T = 1 and i ∈A0, we have P0
t=0 1{i∈At} (t −τi(t)) = 1{i∈A0} (0 −τi(0)) = 1. Therefore,
the base case holds.

The induction hypothesis is that PK−1
t=0 1{i∈At} (t −τi(t)) = K holds for i ∈AK−1. Next, we
focus on K + 1:
K
X

t=0
1{i∈At} (t −τi(t)) =

K−1
X

t=0
1{i∈At} (t −τi(t)) + 1{i∈AK} (K −τi(K)) .
(25)

Now, we have two cases:

• Suppose i ∈AK−1, then we simply have τi(K) = K −1. It follows that Eq. (25)
(a)
= K + 1,
where (a) follows from induction hypothesis.
• Suppose i /∈AK−1,

K
X

t=0
1{i∈At} (t −τi(t))
(b)
=

τi(K)
X

t=0
1{i∈At} (t −τi(t)) + 1{i∈AK} (K −τi(K))

= τi(K) + 1 + (K −τi(K)) = K + 1,
where (b) follows because 1{i∈At} = 0 for τi(K) ≤t ≤K −1 and induction hypothesis that
Pτi(K)
t=0
1{i∈At} (t −τi(t)) = τi(K) + 1 for i ∈Aτi(K).

F.3
Unavailable statistics (Lemma 2)

Proof of Lemma 2.

E [t −τi(t)] =

t
X

r=0
P {t −τi(t) > r} =

t
X

r=0

t−1
Y

r1=t−r
(1 −pr1
i ) ≤

t
X

r=0
(1 −δ)r ≤1

δ .

26


---Page Break---
From [14, Section 12, Theorem 12.3 (i)], we know that

E [g(X)] = g(0) +
Z ∞

0
g′(x)P {X > x} dx,

where X is a non-negative random variable, and g a non-negative strictly increasing differentiable
function. It follows that,

E

X2
≤0 + 2
Z ∞

0
xP {X > x} dx = 2

∞
X

n=1

Z n

n−1
xP {X > x} dx

(a)
≤2

∞
X

n=1
n
Z n

n−1
P {X > x} dx

(b)
≤2

∞
X

n=1
nP {X > n −1}
Z n

n−1
dx = 2

∞
X

n=1
nP {X > n −1} ,

where inequality (a) holds because x ≤n, ∀x ∈(n −1, n], inequality (b) holds because
CCDF P {X > x} is non-increasing.
In particular, for a discrete random variable, we have
P {X > n −1} = P {X ≥n}.

Therefore,

E
h
(t −τi(t))2i
≤2

∞
X

n=1
nP {t −τi(t) ≥n} ≤2

∞
X

n=1
n(1 −δ)n−1 ≤2

δ2 .

F.4
Auxiliary sequence construction and properties (Proposition 2)

Proposition 4. For any t ≥0, when i /∈At, it holds that xt+1
i
−zt+1
i
= ηlηgs(t −τi(t +
1))∇Fi(xτi(t+1)+1
i
); when i ∈At, it holds that zt†
i = xt†
i , zt+1 = xt+1, and zt+1
i
= xt+1
i
.

Proof of Proposition 4. The proof is divided into two parts: i /∈At and i ∈At,

When i /∈At.
It holds that

xt+1
i
−zt+1
i
= xτi(t+1)+1
i
−



zτi(t+1)+1
i
−ηlηgs

t
X

k=τi(t+1)+1
∇Fi(xk
i )





(a)
= xτi(t+1)+1
i
−



xτi(t+1)+1
i
−ηlηgs

t
X

k=τi(t+1)+1
∇Fi(xτi(t+1)+1
i
)





= ηlηgs(t −τi(t + 1))∇Fi(xτi(t+1)+1
i
),

where equality (a) follows from Definition 1 for inactive clients.

When i ∈At.
Note that if zt++
i
= xt++
i
for each i ∈At, then by the aggregation rules,
we know xt+1 = (1/|At|) P

i∈At xt++
i
= (1/|At|) P

i∈At zt++
i
= zt+1. Then, we know that
xt+1
i
= zt+1
i
, ∀i ∈At. Hence, to show the Proposition, it is sufficient to show zt++
i
= xt++
i
holds
for i ∈At, which can be shown by induction.

When t = 0,

z0++
i
= z0
i + 0 −

x(0,0)
i
−x(0,s)
i

= x0
i −

x(0,0)
i
−x(0,s)
i

= x0++
i
.

27


---Page Break---
Thus, the base case holds. The induction hypothesis is that zt++
i
= xt++
i
, ∀i ∈At is true for all
t ≥0. Now, we focus on t + 1.

z(t+1)++
i
= zt+1
i
+ ηlηgs

t
X

k=τi(t+1)+1
∇Fi(xk
i ) −(t + 1 −τi(t + 1))

x(t+1,0)
i
−x(t+1,s)
i


= zt+1
i
+ ηlηgs(t −τi(t + 1))∇Fi(xτi(t+1)+1
i
) −(t + 1 −τi(t + 1))

x(t+1,0)
i
−x(t+1,s)
i


(a)
= zτi(t+1)+1
i
−ηlηgs(t −τi(t + 1) −1 + 1)∇Fi(xτi(t+1)+1
i
)

+ ηlηgs(t −τi(t + 1))∇Fi(xτi(t+1)+1
i
) −(t + 1 −τi(t + 1))

x(t+1,0)
i
−x(t+1,s)
i


= zτi(t+1)+1
i
−(t + 1 −τi(t + 1))

x(t+1,0)
i
−x(t+1,s)
i


(b)
= xτi(t+1)+1
i
−(t + 1 −τi(t + 1))

x(t+1,0)
i
−x(t+1,s)
i


= x(t+1)++
i
,

where equality (a) follows from the auxiliary updates zi, and equality (b) holds because of the
induction hypothesis and the fact that τi(t + 1) < t + 1 and i ∈Aτi(t+1).

Proof of Proposition 2. From Propositions 4, we have
xt
i −zt
i
2
2 ≤
ηlηgs (t −τi(t) −1) ∇Fi(xt
i)
2
2

= η2
l η2
gs2
t−1
X

p=−1
1{τi(t)=p} (t −p −1)2 ∇Fi(xp+1
i
)

2

2 .

Take expectation over all the randomness

E
hxt
i −zt
i
2
2

i (a)
≤η2
l η2
gs2
t−1
X

p=−1
E

1{τi(t)=p}

(t −p −1)2 E
∇Fi(xp+1
i
)

2

2



(b)
≤η2
l η2
gs2
t−1
X

p=−1
(t −p −1)2 P {τi(t) = p} · E
∇Fi(zp+1
i
)

2

2


,

where inequality (a) follows because by definition 1{τi(t)=p} is independent of
∇Fi(xp+1
i
)

2

2,

inequality (b) follows because xp+1
i
= zp+1
i
from Proposition 4.

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −zt
i
2
2

i
= η2
l η2
gs2 1

T

T −1
X

t=0

1
m

m
X

i=1

t−1
X

p=−1
P {τi(t) = p} (t −p −1)2 E
∇Fi(zp+1
i
)

2

2



(c)
≤η2
l η2
gs2 1

m

m
X

i=1

1
T

T −1
X

t=0
E
h∇Fi(zt
i)
2
2

i 
E
h
(t −τi(t))2i

(d)
≤η2
l η2
gs2
 2

δ2

 1

m

m
X

i=1

1
T

T −1
X

t=0
E
h∇Fi(zt
i)
2
2

i

≤3η2
l η2
gs2
 2

δ2

  
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 3η2
l η2
gs2
 2

δ2


ζ2

+ 3η2
l η2
gs2L2
 2

δ2

 1

m

m
X

i=1

1
T

T −1
X

t=0
E
hzt
i −¯zt2
2

i
,

where inequality (c) follows from re-indexing, inequality (d) from Lemma 2.

28


---Page Break---
F.5
Consensus error of the auxiliary sequence

Lemma 6 (Consensus error of zt
i). Assuming that ηl
≤
δ/(20sL), and ηlηg
≤
δ(1 −
√ρ)/(10sL(√ρ + 1)), under Assumption 2, 3 and 4, it holds that

1
m

m
X

i=1

1
T

T −1
X

t=0

m
X

i=1
E
hzt
i −¯zt2
2

i
≤
3ρsη2
l η2
g
(1 −√ρ)2δ2 σ2

+ 40ρs2η2
l η2
g
(1 −√ρ)2 ζ2

+ 40ρs2η2
l η2
g
 
β2 + 1


(1 −√ρ)2
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
.

Proof of Lemma 6. When t = 0, Z0 = [z0, · · · , z0], which immediately leads to

Z0 (I −J) = [z0, · · · , z0] −[z0, · · · , z0] = 0.

For t ≥1, recall that W (t) is a doubly stochastic matrix to characterize the information mixture, and
e
Gt, defined in (19), captures the local parameter changes in each round. It can be seen that

Z(t) =

Z(t−1) −ηlηg e
Gt−1
W (t−1).

Expanding Z, we get

Z(t) (I −J) = (Z(t−1) −ηlηg e
Gt−1)W (t−1) (I −J)

= Z0
t−1
Y

ℓ=0
W ℓ(I −J) −ηlηg

t−1
X

q=0
e
Gq
t−1
Y

ℓ=q
W (ℓ) (I −J) .

where the last follows from the fact that all clients are initiated at the same weights. Note that
Qt−1
ℓ=q W (ℓ)I = Qt−1
ℓ=q W (ℓ) and Qt−1
ℓ=q W (ℓ)J = J. Thus,

Z(t) (I −J) = Z0
 t−1
Y

ℓ=0
W ℓ−J

!

−ηlηg

t−1
X

q=0
e
Gq




t−1
Y

ℓ=q
W (ℓ) −J



= −ηlηg

t−1
X

q=0
e
Gq




t−1
Y

ℓ=q
W (ℓ) −J



,

where the last equality holds because that Z0 = [z0, · · · , z0], which immediately leads to

Z0
 t−1
Y

ℓ=0
W ℓ−J

!

= [z0, · · · , z0] −[z0, · · · , z0] = 0.

Let matrix notations e∆t, ∆t and ∇F t
x define as follows:

Gq
i = 1{i∈At}(t −τi(t))

s−1
X

r=0


∇ℓi(x(t,r)
i
; ξ(t,r)
i
) −∇Fi(x(t,r)
i
)


|
{z
}

[e∆t]i

+ 1{i∈At}(t −τi(t))

s−1
X

r=0


∇Fi(x(t,r)
i
) −∇Fi(xt
i)


|
{z
}
[∆t]i

+ s ∇Fi(xt
i)
|
{z
}
[∇F t
x]i

.

29


---Page Break---
It follows that

∥Z(t) (I −J) ∥2
F = ∥

t−1
X

q=0


e∆q + ∆q + ∇F q
x




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F

= ∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F + ∥

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F

+ 2

*t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



,

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J





+

F

.

Take expectation with respect to randomness in stochastic gradients, denote by Eξ [·]:

Eξ
h
∥Z(t) (I −J) ∥2
F
i
= Eξ



∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



+ Eξ



∥

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F





+ 2Eξ





*t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



,

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J





+

F





= Eξ



∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



+ Eξ



∥

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F





+ 2

*t−1
X

q=0
Eξ
h
e∆qi



t−1
Y

ℓ=q
W (ℓ) −J



,

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J





+

F

≤Eξ



∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



+ Eξ



∥

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



,

where the last inequality holds because Eξ
h
e∆qi
= 0. Next, we take expectation over the remaining
randomness.

E
h
∥Z(t) (I −J) ∥2
F
i
≤E



∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



+ E



∥

t−1
X

q=0
(∆q + ∇F q
x)




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F





≤η2
l η2
g ∥

t−1
X

q=0
e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F

|
{z
}
(I)

+ 2η2
l η2
g ∥

t−1
X

q=0
∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F

|
{z
}
(II)

+ 2η2
l η2
gs2 ∥

t−1
X

q=0
∇F q
x




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F

|
{z
}
(III)

.
(26)

30


---Page Break---
Bounding E [(I)]

E [(I)] =

t−1
X

q=0
E



∥e∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F





+

t−1
X

q=0

t−1
X

p=0,p̸=q
E





*
e∆p




t−1
Y

ℓ=p
W (ℓ) −J



, e∆q




t−1
Y

ℓ=q
W (ℓ) −J





+



(a)
≤

t−1
X

q=0
ρt−qE
h
∥e∆q∥2
F
i
,
(27)

where inequality (a) holds because of Assumption 3. It remains to bound E
h
∥e∆q∥2
F
i
.

∥e∆q∥2
F =

m
X

i=1
1{i∈Aq}



q−1
X

p=−1
1{τi(t)=p}(q −p)

s−1
X

r=0


∇ℓi(x(q,r)
i
; ξ(q,r)
i
) −∇Fi(x(q,r)
i
)


2

2
.

Eξ
h
∥e∆q∥2
F
i
=

m
X

i=1
1{i∈Aq}

q−1
X

p=−1
1{τi(t)=p}(q −p)2
s−1
X

r=0
Eξ

∇ℓi(x(q,r)
i
; ξ(p,r)
i
) −∇Fi(x(q,r)
i
)

2

2



≤sσ2
m
X

i=1
1{i∈Aq}

q−1
X

p=−1
1{τi(t)=p}(q −p)2.

Take expectation over the remaining randomness:

E
h
∥e∆q∥2
F
i
= E
h
Eξ
h
∥e∆q∥2
F
ii
≤sσ2
m
X

i=1
E

1{i∈Aq}
 q−1
X

p=−1
E

1{τi(t)=p}

(q −p)2 ≤2msσ2

δ2

Therefore,

1
mT

m
X

i=1

T −1
X

t=0
E [(I)] ≤
sρ
(1 −ρ)

 2

δ2


σ2.

Bounding E [(II)]

E [(II)] = E



∥

t−1
X

q=0
∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F





=

t−1
X

q=0
E



∥∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



+

t−1
X

q=0

t−1
X

p=0,p̸=q
E





*

∆p




t−1
Y

ℓ=p
W (ℓ) −J



, ∆q




t−1
Y

ℓ=q
W (ℓ) −J





+



≤

t−1
X

q=0
ρt−qE

∥∆q∥2
F

+

t−1
X

q=0

t−1
X

p=0,p̸=q
E



∥∆p




t−1
Y

ℓ=p
W (ℓ) −J



∥F∥∆q




t−1
Y

ℓ=q
W (ℓ) −J



∥F





≤

t−1
X

q=0
ρt−qE

∥∆q∥2
F

+

t−1
X

q=0

t−1
X

p=0,p̸=q
E
ρt−p

2ϵ ∥∆p∥2
F + ϵρt−q

2
∥∆q∥2
F


,

31


---Page Break---
Next, we bound the second term, choose ϵ = ρ
q−p

2 ,

t−1
X

q=0

t−1
X

p=0,p̸=q

√ρ2t−p−q

2
E

∥∆p∥2
F + ∥∆q∥2
F

≤

t−1
X

q=0

t−1
X

p=0

√ρ2t−p−q

2
E

∥∆p∥2
F + ∥∆q∥2
F


=

t−1
X

p=0

√ρt−p

2
E

∥∆p∥2
F
 t−1
X

q=0

√ρt−q +

t−1
X

q=0

√ρt−q

2
E

∥∆q∥2
F
 t−1
X

p=0

√ρt−p

=
√ρ −√ρt+1

1 −√ρ

t−1
X

q=0

√ρt−qE

∥∆q∥2
F

.
(28)

Plugging the upper bound in (28) into (27), we get

E [(II)] ≤

t−1
X

q=0

"
√ρt−q +
√ρ −√ρt+1

1 −√ρ

#
√ρt−qE

∥∆q∥2
F
 (b)
≤

t−1
X

q=0

√ρ + √ρ

1 −√ρ

 √ρt−qE

∥∆q∥2
F


≤
2√ρ
1 −√ρ

t−1
X

q=0

√ρt−qE

∥∆q∥2
F

,
(29)

where inequality (b) follows because that √ρt−q ≤√ρ for any q ≤t −1, and that √ρt+1 ≥0. It
remains to bound E

∥∆q∥2
F

. Take expectation with respect to randomness in stochastic gradients:

Eξ

∥∆q∥2
F

≤4η2
l s3L2
m
X

i=1

q−1
X

p=−1
1{τi(q)=p}(q −p)2σ2

+ 16η2
l s4L2
m
X

i=1

q−1
X

p=−1
1{τi(q)=p}(q −p)2 ∥∇Fi(xq
i )∥2
2 ,

where the inequality holds due to Lemma 5. Next, we take expectation over the remaining randomness
and plug back into (29):

E [(II)] ≤
2√ρ
1 −√ρ

t−1
X

q=0

√ρt−qE

∥∆q∥2
F


≤
8ρ
 
1 −√ρ
2

 2

δ2


η2
l s3L2mσ2

+ 32√ρ

1 −√ρ

 2

δ2


η2
l s4L2
m
X

i=1

t−1
X

q=0
E
h
∥∇Fi(xq
i )∥2
2
i T −1−t
X

k=1

√ρk

≤
8ρ
 
1 −√ρ
2

 2

δ2


η2
l s3L2mσ2 +
32ρ
 
1 −√ρ
2

 2

δ2


η2
l s4L2
m
X

i=1

t−1
X

q=0
E
h
∥∇Fi(xq
i )∥2
2
i
,

where the last inequality holds because of re-index and grouping. Therefore,

1
mT

T −1
X

t=1
E [(II)] ≤
8ρ
 
1 −√ρ
2

 2

δ2


η2
l s3L2σ2

+
32ρ
 
1 −√ρ
2

 2

δ2


η2
l s4L2 1

T

T −1
X

t=1

1
m

m
X

i=1
E
h∇Fi(xt
i)
2
2

i

≤
8ρ
 
1 −√ρ
2

 2

δ2


η2
l s3L2σ2 +
64ρ
 
1 −√ρ
2

 2

δ2


η2
l s4L4 1

T

T −1
X

t=1

1
m

m
X

i=1
E
hxt
i −zt
i
2
2

i

+
64ρ
 
1 −√ρ
2

 2

δ2


η2
l s4L2 1

T

T −1
X

t=1

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

32


---Page Break---
Bounding E [(III)]
Use a similar trick as in bounding E [(II)] , and we get

E [(III)] = E



∥

t−1
X

q=0
∇F q
x




t−1
Y

ℓ=q
W (ℓ) −J



∥2
F



≤
2√ρ
1 −√ρ

t−1
X

q=0

√ρt−qE

∥∇F q
x∥2
F

,

so that

1
mT

T −1
X

t=0
E [(III)] ≤
2√ρ
mT
 
1 −√ρ


T −1
X

t=0
E

∥∇F t
x∥2
F
 T −1−t
X

q=1

√ρq

≤
2ρ
 
1 −√ρ
2
1
mT

T −1
X

t=0

m
X

i=1
E
h∇Fi(xt
i)
2
2

i

≤
4ρL2
 
1 −√ρ
2
1
mT

T −1
X

t=0

m
X

i=1
E
hxt
i −zt
i
2
2

i
+
4ρ
 
1 −√ρ
2
1
mT

T −1
X

t=0

m
X

i=1
E
h∇Fi(zt
i)
2
2

i
.

Putting them together

1
mT

T −1
X

t=0
E
h
∥Z(t) (I −J) ∥2
F
i
≤
sρη2
l η2
g
(1 −√ρ)2

 2

δ2

  
1 + 16η2
l s2L2
σ2

+ 8ρs2L2η2
l η2
g
(1 −√ρ)2


1 + 16η2
l s2L2
 2

δ2

 1

T

T −1
X

t=1

1
m

m
X

i=1
E
hxt
i −zt
i
2
2

i

+ 8ρs2η2
l η2
g
(1 −√ρ)2


1 + 16η2
l s2L2
 2

δ2

 1

T

T −1
X

t=1

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i
.

Plug in Proposition 2.

1
mT

T −1
X

t=0
E
h
∥Z(t) (I −J) ∥2
F
i
≤
sρη2
l η2
g
(1 −√ρ)2

 2

δ2

  
1 + 20η2
l s2L2
σ2

+ 8ρs2η2
l η2
g
(1 −√ρ)2


1 + 16η2
l s2L2
 2

δ2

 
1 + η2
l η2
gs2L2
 2

δ2

 1

T

T −1
X

t=1

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

≤1.05ρsη2
l η2
g
(1 −√ρ)2

 2

δ2


σ2 + 9ρs2η2
l η2
g
(1 −√ρ)2
1
T

T −1
X

t=1

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i
,

where the last inequality holds because ηl ≤δ/(20sL) and ηlηg ≤δ/(10sL). Next, plug in
Proposition 3.

1
mT

T −1
X

t=0
E
h
∥Z(t) (I −J) ∥2
F
i
≤1.05ρsη2
l η2
g
(1 −√ρ)2

 2

δ2


σ2 + 27ρs2η2
l η2
g
(1 −√ρ)2 ζ2

+ 27ρs2η2
l η2
g
 
β2 + 1


(1 −√ρ)2
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 27ρs2L2η2
l η2
g
(1 −√ρ)2
1
T

T −1
X

t=0

1
m

m
X

i=1
E
hzt
i −¯zt2
2

i
.

It follows that

1
mT

T −1
X

t=0
E
h
∥Z(t) (I −J) ∥2
F
i
≤
3ρsη2
l η2
g
(1 −√ρ)2δ2 σ2

+ 40ρs2η2
l η2
g
(1 −√ρ)2 ζ2

+ 40ρs2η2
l η2
g
 
β2 + 1


(1 −√ρ)2
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
.

which is due to the fact that ηlηg ≤
1−√ρ
10sL(√ρ+1).

33


---Page Break---
F.6
Spectral norm upper bound (Lemma 4)

Lemma 4 adapts from [57], we present its proof here for completeness.

Proof of Lemma 4. For ease of exposition, in this proof we drop time index t. We first get the
explicit expression for E

W 2
jj′ | A ̸= ∅

. Suppose that A ̸= ∅. We have

W 2
jj′ =

m
X

k=1
WjkWj′k = WjjWj′j + Wjj′Wj′j′ +
X

k∈[m]\{j,j′}
WjkWj′k.

When k ̸= j and k ̸= j′, we have

WjkWj′k =
1
|A|2 1{j∈A}1{j′∈A}1{k∈A}.

In addition, we have WjjWj′j =
1
|A|2 1{j∈A}1{j′∈A}, and Wj′j′Wjj′ =
1
|A|2 1{j∈A}1{j′∈A}. Thus,

• For j ̸= j′, we have

W 2
jj′ =

m
X

k=1
WjkWj′k =
1
|A|1{j∈A}1{j′∈A};

• For j = j′, we have

W 2
jj =
1
|A|1{j∈A} +
 
1 −1{j∈A}

.

In the special case where A = ∅, we simply have W = I by the algorithmic clauses. Therefore,
E [Wjj′ | A = ∅] ≥0 holds for any pair of j, j′ ∈[m]. It follows, by the law of total expectation and
for all j, j′ ∈[m], that
E [Wjj′] = E [Wjj′ | A = ∅] P {A = ∅} + E [Wjj′ | A ̸= ∅] P {A ̸= ∅}
≥E [Wjj′ | A ̸= ∅] P {A ̸= ∅} .
• For j ̸= j′, it holds that

E

W 2
jj′ | A ̸= ∅

= E
 1

|A|1{j∈A}1{j′∈A}
A ̸= ∅
 (a)
≥E
 1

m1{j∈A}1{j′∈A}
A ̸= ∅

= pjpj′

m
≥δ2

m,

where inequality (a) holds because |A| ≤m ;
• For j = j′, it holds that

E

W 2
jj | A ̸= ∅

= E
 1

|A|1{j∈A} +
 
1 −1{j∈A}
 A ̸= ∅


≥E
 1

m

1{j∈A} +
 
1 −1{j∈A}
 A ̸= ∅

= 1

m ≥δ2

m.

Recall that M = E

W 2
. Next, we show that each element of M is lower bounded.

Mjj′ ≥E

W 2
jj′ | A ̸= ∅

P {A ̸= ∅} ≥δ2

m [1 −(1 −δ)m] .

We note that ρ(t) = λ2(M), where λ2 is the second largest eigenvalue of matrix M. A Markov chain
with M as the transition matrix is ergodic as the chain is (1) irreducible: Mjj′ ≥δ2

m [1 −(1 −c)m] >
0 for j, j′ ∈[m] and (2) aperiodic (it has self-loops). In addition, W matrix is by definition doubly-
stochastic. Hence, M has a uniform stationary distribution π = 1⊤/m. Furthermore, the irreducible
Markov chain is reversible since it holds for all the states that πiMij = πjMji. The conductance Φ
of a reversible Markov chain [18] with a transition matrix M can be bounded by

Φ(M) =
min
P

i∈S πi≤1

2

P

i∈S,j /∈S πiMij
P

i∈S πi
≥

  δ

m
2 [1 −(1 −δ)m] |S|
 ¯S


|S|

m
= δ2 [1 −(1 −δ)m]

m

 ¯S
 ,

where
 ¯S
 = m −|S| ≥
m

2 . From Cheeger’s inequality, we know that 1−λ2

2
≤Φ(M) ≤
p

2 (1 −λ2). Finally, we have

Φ(M) ≥δ2 [1 −(1 −δ)m]

m

 ¯S
 ≥δ2 [1 −(1 −δ)m]

2
.

Thus, ρ(t) = λ2 ≤1 −Φ2(M)

2
≤1 −δ4[1−(1−δ)m]2

8
.

34


---Page Break---
G
Convergence Error of ¯zt (Theorem 1)

In the sequel, we recall and assume the following learning rate conditions in (11):

ηlηg ≤

 
1 −√ρ

δ

80s(L + 1)
 √ρ + 1
 p

(β2 + 1) (1 + L2)
; ηl ≤
δ

200sL
p

(β2 + 1) (1 + L2)
.

Recall that δmax ≜maxi∈[m],t∈[T ] pt
i and F ⋆≜minx F(x).

Proof of Theorem 1. Take expectation over all the randomness, plug in Lemma 6 and Proposition 2.
By telescoping sum, it holds that

E

F ⋆−F(¯z0)


T
≤−ηlηgs

4
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 2η2
l η2
gsLδmaxσ2

m2T

T −1
X

t=0

m
X

i=1

t−1
X

p=−1
E

1{τi(t)=p}

(t −p)2

+ 9ηgη3
l s2L2σ2

mT

T −1
X

t=0

m
X

i=1

t−1
X

p=−1
E

1{τi(t)=p}

(t −p)2

+ 2.2ηlηgsL2 1

mT

T −1
X

t=0

m
X

i=1
E
hxt
i −zt
i
2
2

i
(30)

+ ηlηgsL2

2mT

T −1
X

t=0

m
X

i=1
E
hzt
i −¯zt2
2

i
(31)

+ 35ηgη3
l s3L2

mT

T −1
X

t=0

m
X

i=1

t−1
X

p=−1
E

1{τi(t)=p}

(t −p)2E
∇Fi(xp+1
i
)

2

2


.
(32)

Next, we bound (30), (31) and (32), respectively. First, we show that

1
mT

T −1
X

t=0

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

≤3ζ2 + 3
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 3L2

mT

T −1
X

t=0

m
X

i=1
E
hzt
i −¯zt2
2

i

≤3

"

1 + 40ρs2η2
l η2
gL2

(1 −√ρ)2

#

ζ2 + 3
 
β2 + 1

"

1 + 40ρs2η2
l η2
gL2

(1 −√ρ)2

#
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 9ρsη2
l η2
gL2

(1 −√ρ)2δ2 σ2,

(33)
where the last inequality follows from Lemma 6.

For (30), we have

2.2ηlηgsL2 1

T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −zt
i
2
2

i
≤4.4η3
l η3
gs3L2

δ2
1
T

T −1
X

t=0

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

≤s2η3
l η3
gL2

2δ2
σ2 + 14η3
l η3
gs3L2

δ2

 

1 + 40η2
l η2
gρs2L2

(1 −√ρ)2

!

ζ2

+ 14η3
l η3
gs3L2

δ2

"
 
β2 + 1

+ 40η2
l η2
gρs2L2

(1 −√ρ)2

#
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
,

where the last inequality holds due to (33). For (31), we similarly have

ηlηgsL2

2mT

T −1
X

t=0

m
X

i=1
E
hzt
i −¯zt2
2

i
≤1.5ρs2η3
l η3
gL2

(1 −√ρ)2δ2 σ2 + 20ρs3η3
l η3
gL2

(1 −√ρ)2 ζ2

+ 20ρs3η3
l η3
gL2  
β2 + 1


(1 −√ρ)2
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
.

35


---Page Break---
For (32), we have

35ηgη3
l s3L2 1

T

T −1
X

t=0

1
m

m
X

i=1

t−1
X

p=−1
E

1{τi(t)=p}

(t −p)2E
∇Fi(xp+1
i
)

2

2



≤70ηgη3
l s3L2

mTδ2

T −1
X

t=0

m
X

i=1
E
h∇Fi(xt
i)
2
2

i

≤140ηgη3
l s3L4

mTδ2

T −1
X

t=0

m
X

i=1
E
hxt
i −zt
i
2
2

i
+ 140ηgη3
l s3L2

mTδ2

T −1
X

t=0

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

≤

 

1 + 2η2
l η2
gs2L2

δ2

!  2

δ2

 70ηgη3
l s3L2

mT

T −1
X

t=0

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

(a)
≤
 2

δ2

 71ηgη3
l s3L2

mT

T −1
X

t=0

m
X

i=1
E
h∇Fi(zt
i)
2
2

i

(b)
≤426ηgη3
l s3L2

δ2

"

1 + 40ρs2η2
l η2
gL2

(1 −√ρ)2

#

ζ2 + 426ηgη3
l s3L2

δ2
 
β2 + 1

"

1 + 40ρs2η2
l η2
gL2

(1 −√ρ)2

#
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ ηgη3
l s3L2σ2

2δ2
,

where inequality (a) holds because of (11), inequality (b) holds because of (33).

Putting (30), (31) and (32) together and plugging them back into the telescoping sum, it holds that

E

F ⋆−F(¯z0)


T

≤−

 
ηlηgs

4
−14
 
β2 + 1

η3
l η3
gs3L2  
1 + L2

δ2
−20ρs3η3
l η3
gL2  
β2 + 1


(1 −√ρ)2

!
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

−

 

−426ηgη3
l s3L2  
β2 + 1
  
1 + L2

δ2

!
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 4η2
l η2
gsLδmaxσ2

mδ2
+

 
η3
l η3
gs2L2σ2

2δ2
+ 1.5ρs2η3
l η3
gL2

(1 −√ρ)2δ2 σ2 + ηgη3
l s3L2σ2

2δ2

!

+ 15η3
l η3
gs3L2ζ2

δ2
+ 20ρs3η3
l η3
gL2

(1 −√ρ)2 ζ2 + 430ηgη3
l s3L2ζ2

δ2

≤−ηlηgs

6
1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 4η2
l η2
gsLδmaxσ2

mδ2
+

 
η3
l η3
gs2L2σ2

2δ2
+ 1.5ρs2η3
l η3
gL2

(1 −√ρ)2δ2 σ2 + ηgη3
l s3L2σ2

2δ2

!

+ 15η3
l η3
gs3L2ζ2

δ2
+ 20ρs3η3
l η3
gL2

(1 −√ρ)2 ζ2 + 430ηgη3
l s3L2ζ2

δ2
,

where the last inequality holds because of (11).

36


---Page Break---
Combining the above and rearranging the terms, we get

1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
≤6
 
F(¯z0) −F ⋆

ηlηgsT

+ 24ηlηgLδmaxσ2

mδ2
+

 
3η2
l η2
gsL2σ2

δ2
+ 9ρsη2
l η2
gL2

(1 −√ρ)2δ2 σ2 + 3η2
l s2L2σ2

δ2

!

+ 90η2
l η2
gs2L2ζ2

δ2
+ 120ρs2η2
l η2
gL2

(1 −√ρ)2
ζ2 + 2580η2
l s2L2ζ2

δ2

≤6
 
F(¯z0) −F ⋆

ηlηgsT
+ 24ηlηgLδmaxσ2

mδ2
+ 15η2
l η2
gs2L2σ2

(1 −√ρ)2δ2
+ 2800η2
l η2
gs2L2ζ2

δ2(1 −√ρ)2
,

where the last inequality holds because ρ < 1. In terms of asymptotics, we have

1
T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
≲

 
F(¯z0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2
+ η2
l η2
gs2L2
 
σ2 + ζ2

δ2  
1 −√ρ
2

!

,

where we use the convention that ηg ≥1 for ease of presentation.

37


---Page Break---
H
Convergence Rate of ¯xt (Corollary 1)

H.1
Convergence error of Algorithm 1

Corollary 2 (Convergence error of xt
i). Suppose learning rates conditions in (11) are met for ηl and
ηg, and Assumptions 1, 2, 3 and 4 hold for T ≥1, it holds that

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≲

 
F(¯x0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2
+ η2
l η2
gs2L2
 
σ2 + ζ2

δ2  
1 −√ρ
2

!

,

Proof of Corollary 2.

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≤3

T

T −1
X

t=0
E
h∇F(¯xt) −∇F(¯zt)
2
2

i
+ 3

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

(a)
≤3L2

T

T −1
X

t=0
E
h¯xt −¯zt2
2

i
+ 3

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

(b)
≤3L2

T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −zt
i
2
2

i
+ 3

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

≤3
 2

δ2

 η2
l η2
gs2L2

T

T −1
X

t=0

1
m

m
X

i=1
E
h∇Fi(zt
i)
2
2

i
+ 3

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
,

where inequality (a) follows from Appendix D 2, inequality (b) follows from Assumption 2.

Further plug in Proposition 3,

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≤3

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 9η2
l η2
gs2L2
 2

δ2

  
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 9η2
l η2
gs2L4
 2

δ2

 1

T

T −1
X

t=0

1
m

m
X

i=1
E
hzt
i −¯zt2
2

i
+ 9η2
l η2
gs2L2
 2

δ2


ζ2.

Finally, plug in Lemma 6.

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≤
3

2 + 9η2
l η2
gs2L2
 2

δ2

  
β2 + 1
 90

802

 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 9 × 8

802 η2
l η2
gsL2
 2

δ2


σ2 + 9η2
l η2
gs2L2
 2

δ2


ζ2 + 9 × 90

2002 η2
l η2
gs2L2ζ2

≤2

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ sL2η2
l η2
g
δ2
σ2 + 9η2
l η2
gs2L2

δ2
ζ2 + s2L2η2
l η2
gζ2

≤12
 
F(¯z0) −F ⋆

ηlηgsT
+ 48ηlηgLδmaxσ2

mδ2
+ 31η2
l η2
gs2L2σ2

(1 −√ρ)2δ2
+ 5600η2
l η2
gs2L2ζ2

(1 −√ρ)2δ2
,

where the last inequality holds because ρ < 1. In terms of asymptotics, we have

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≲

 
F(¯x0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2
+ η2
l η2
gs2L2

σ2 + ζ2

δ2(1 −√ρ)2


,

where we use the convention that ηg ≥1 for ease of presentation.

38


---Page Break---
H.2
Convergence rate of Algorithm 1

Proof of Corollary 1. Choose step-size as ηl =
1
√

T sL, ηg =
√

sδm such that learning rate condi-
tions in (11) are met, it holds that

1
T

T −1
X

t=0
E
h∇F(¯xt)
2
2

i
≲L
 
F(¯x0) −F ⋆

√

sδmT
+
δmax
δ
3
2 √

smT
σ2 + sm

T


σ2 + ζ2

δ(1 −√ρ)2


.

39


---Page Break---
I
Additional Results and Interpretations

I.1
Consensus error of Algorithm 1

Corollary 3 (Consensus error of xt
i). Suppose learning rates conditions are met in (11) for ηl and
ηg, and Assumptions 1, 2, 3 and 4 hold for T ≥1, it holds that

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2
2

i
≲

 
F(¯x0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2

+ η2
l η2
gs2L2
σ2 + ζ2

δ2

 "

1 +
ρ
 
1 −√ρ
2

#

,

Proof of Corollary 3.

1
T

T −1
X

t=0

1
m

m
X

i=1

xt
i −¯xt2
2 = 1

T

T −1
X

t=0

1
m

m
X

i=1

xt
i −zt
i + zt
i −¯zt + ¯zt −¯xt2
2

(a)
≤1

T

T −1
X

t=0

3
m

m
X

i=1

xt
i −zt
i
2
2 + 1

T

T −1
X

t=0

3
m

m
X

i=1

zt
i −¯zt2
2 + 1

T

T −1
X

t=0
3
¯zt −¯xt2
2

(b)
≤1

T

T −1
X

t=0

3
m

m
X

i=1

xt
i −zt
i
2
2 + 1

T

T −1
X

t=0

3
m

m
X

i=1

zt
i −¯zt2
2 + 1

T

T −1
X

t=0

3
m

m
X

i=1

zt
i −xt
i
2
2

= 1

T

T −1
X

t=0

6
m

m
X

i=1

xt
i −zt
i
2
2 + 1

T

T −1
X

t=0

3
m

m
X

i=1

zt
i −¯zt2
2 ,

where inequalities (a) and (b) follow from Jensen’s inequality. Plug in Proposition 2 and take
expectation over all the randomness, we get

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2
2

i
≤36η2
l η2
gs2

δ2
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 36η2
l η2
gs2

δ2
ζ2 +

 

3 + 36η2
l η2
gs2L2

δ2

!
1
m

m
X

i=1

1
T

T −1
X

t=0
E
hzt
i −¯zt2
2

i

≤36η2
l η2
gs2

δ2
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 36η2
l η2
gs2

δ2
ζ2

+ 4

m

m
X

i=1

1
T

T −1
X

t=0
E
hzt
i −¯zt2
2

i
,

where the last inequality holds because of learning rate condition in (11). Next, plug in Lemma 6:

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2
2

i
≤36η2
l η2
gs2

δ2
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 36η2
l η2
gs2

δ2
ζ2 + 4

m

m
X

i=1

1
T

T −1
X

t=0
E
hzt
i −¯zt2
2

i

≤36η2
l η2
gs2

δ2
 
β2 + 1
 1

T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+ 1

4T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i

+ 36η2
l η2
gs2

δ2
ζ2 +
12ρsη2
l η2
g
(1 −√ρ)2δ2 σ2 + 160ρs2η2
l η2
g
(1 −√ρ)2 ζ2

≤1

2T

T −1
X

t=0
E
h∇F(¯zt)
2
2

i
+
12ρsη2
l η2
g
(1 −√ρ)2δ2 σ2 + 36η2
l η2
gs2

δ2
ζ2 + 160ρs2η2
l η2
g
(1 −√ρ)2 ζ2.

40


---Page Break---
Finally, we plug in Theorem 1

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2
2

i
≤3
 
F(¯x0) −F ⋆

ηlηgsT
+ 12ηlηgLδmaxσ2

mδ2
+ 28s2η2
l η2
gL2

δ2(1 −√ρ)2 σ2 + 1600η2
l η2
gs2L2

δ2(1 −√ρ)2 ζ2,

where we use the fact that ¯z0 = ¯x0 and ρ < 1, and the convention that ηg ≥1 and L ≥1 for ease of
presentation.

In terms of asymptotics, we have

1
T

T −1
X

t=0

1
m

m
X

i=1
E
hxt
i −¯xt2
2

i
≲

 
F(¯x0) −F ⋆

ηlηgsT
+ ηlηgLσ2

m
δmax

δ2
+ η2
l η2
gs2L2

σ2 + ζ2

δ2(1 −√ρ)2


.

I.2
Orders of the asymptotic rates

From Theorem 1, Corollary 2, Corollary 3, it is easy to see from the theorem statements that they are
all of the same asymptotic order, i.e.,

1
T

T −1
X

t=0
E[∥∇F(¯xt)∥2
2] ≍1

T

T −1
X

t=0

1
m

m
X

i=1
E[∥xt
i −¯xt∥2
2] ≍1

T

T −1
X

t=0
E[∥∇F(¯zt)∥2
2].

In addition, by applying learning rate conditions in (11) to Lemma 6 and Proposition 2, we can also
see that

1
T

T −1
X

t=0

1
m

m
X

i=1
E[∥xt
i −zt
i∥2
2] ≍1

T

T −1
X

t=0

1
m

m
X

i=1
E[∥zt
i −¯zt∥2
2] ≍1

T

T −1
X

t=0
E[∥∇F(¯zt)∥2
2].

Therefore, we conclude that (12), (14) and (15) hold.

41


---Page Break---
Table 6: Neural network architecture, loss function, learning rate scheduling, training steps and batch size
specifications

Datasets
SVHN
CIFAR-10
CINIC-10
Neural network
CNN
CNN
CNN

Model architecture∗
C(3,32) – R – M –
C(32,32) – R – M
– L(128) – R –
L(10)

C(3,32) – R – M –
C(32,32) – R – M
– L(256) – R –
L(64) – R –
L(10)

C(3,32) – R – M –
C(32,32) – R – M
– D – L(512) – R –
D – L(256) – R –
D – L(10)

Loss function
Cross-entropy loss

Local learning rate ηl
scheduling
ηl =
η0
√

t/10+1, where t denotes the global round.

Number of local steps s
10

Number of global rounds T
2000

Batch size
128

∗C(# in-channel, # out-channel): a 2D convolution layer (kernel size 3, stride 1, padding 1); R: ReLU
activation function; M: a 2D max-pool layer (kernel size 2, stride 2); L: (# outputs): a fully-connected
linear layer; D: a dropout layer (probability 0.2).

J
Numerical Experiments

J.1
Code

The code for reproducing our experiments is available at https://github.com/mingxiang12/
FedAWE.

J.2
Experimental setups

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
Data (label) class

0

2

4

6

8

10

12

14

16

18

Client Index

Figure 4: An example of data het-
erogeneity using Dirichlet(α = 0.1)
distribution with 20 clients. x-axis de-
notes the categories of images, while
y-axis denotes the client index. The
size of a circle refers to the proportion
of pictures in a given class. The color
of a circle distinguishes images with
different categories.

Hardware and Software Setups.

• Hardware. The simulations are performed on a private clus-
ter with 64 CPUs, 500 GB RAM and 8 NVIDIA A5000 GPU
cards.
• Software. We code the experiments based on PyTorch 1.13.1
[39] and Python 3.7.16.

Neural Network and Hyper-parameter Specifications. Ta-
ble 6 specifies details of the structures of the convolu-
tional neural network and training.
We initialize CNNs
using the Kaiming initialization.
The initial local learn-
ing rate η0 and the global learning rate ηg are searched,
based on the best performance after 500 global rounds,
over two grids {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005} and
{0.5, 1, 1.5, 5, 10, 50}, respectively. The results are presented
in Table 7.

The
difference
between
FedAvg
over
active
clients
and FedAvg over all clients is that the latter counts the
contributions of unavailable clients as 0’s. We set β = 0.001
for F3AST [43], which is tuned over a grid of {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005}. In addition, as
recommended by [54], we choose K = 50 in FedAU without further specification. We train CNNs
on all datasets for 2000 rounds. Fig. 3 adopts the same hyperparameter setups, yet with only 1000
training rounds.

Datasets and Data Heterogeneity.

Datasets. All the datasets we evaluate contain 10 classes of images. Some data enhancement tricks
that are standard in training image classifiers are applied during training. Specifically, we apply

42


---Page Break---
Table 7: Initial learning rate η0 and global learning rate ηg

Algorithms
FedAvg
active
FedAvg
known
FedAvg
all
FedAU
F3AST
FedAWE
MIFA
FedVARP

SVHN
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0

CIFAR-10
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0

CINIC-10
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
η0
ηg
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0
0.05
1.0
0.1
1.0
0.05
1.0
0.05
1.0

random cropping and gradient clipping with a max norm of 0.5 to all dataset trainings. Furthermore,
random horizontal flipping is applied to CIFAR-10 and CINIC-10.

One full set of experiments takes about 6 hours on SVHN and CIFAR-10 datasets, while about 10
hours on CINIC-10 dataset.

• SVHN [36]. The dataset contains 32×32 colored images of 10 different number digits. In total,
there are 73257 train images and 26032 test images.
• CIFAR-10 [25]. The dataset contains 32×32 colored images of 10 different objects. In total, there
are 50000 train images and 10000 test images.
• CINIC-10[11]. The dataset contains 32×32 colored images of 10 different objects. In total, there
are 90000 train images and 90000 test images.

Data heterogeneity. Fig. 4 visualizes an example of 20 clients, the size of each circle corresponds to
the relative proportion of images from a specific class. The larger the circle, the greater the share of
images associated with that particular class. Moreover, α controls the heterogeneity of the data such
that a greater α entails a more non-i.i.d. local data distribution and vice versa.

J.3
Non-stationary client unavailability dynamics

0.2
0.4
0.6
0.8
1.0
Probability pi

0

2

4

6

8

10

12

14

16

Count

Figure 5: A histogram of one generated pi’s
example with a total of m = 100 clients.
It can be seen that the majority of pi’s are
below 0.5.

Client unavailability dynamics and visualizations. As
specified in Section 7, we consider a total of four client
unavailable dynamics in the form of pt
i = pi · fi(t), where
pi = ⟨νi, ϕ⟩, νi ∼Dirichlet(α) and ϕ is the distribution
to characterize the uneven contributions of each image
class. In detail, each element [ϕ]c is drawn from a uniform
distribution Uniform(0, Φc). We set Φc = 1 for the first
five image classes and Φc′ = 0.5 for the remaining five
image classes. Fig. 5 plots one resulting pi’s example,
wherein pi’s are heterogeneous across clients.

Next, we formally introduce fi(t)’s under each dynamic.

• Stationary: fi(t) ≜1;
• Non-stationary with staircase trajectory:

fi(t) ≜1{t∈[t0,t0+P/2)} + 0.4 · 1{t∈[t0+P/2,t0+P )},

where P defines a period, t0 ∈{0, P, 2P, 3P, . . .}.
• Non-stationary with sine trajectory:

fi(t) ≜γ sin(2π/P · t) + (1 −γ),

where γ signifies the degree of non-stationary.
• Non-stationary with interleaved sine trajectory:

fi(t) ≜gi(t) · 1{pi·gi(t)≥δ0},

where gi(t) ≜γ sin(2π/P ·t)+(1−γ) and δ0 = 0.1 defines a cutting-off lower bound. Specifically,
δ0 cuts off the sine curve and brings in a period of zero-valued probabilities. As different clients
have different pi’s, the cut-off points are not synchronized among clients, leading to additional
availability heterogeneity.

43


---Page Break---
0
10
20
30
40
50
60
70
80
Global round t

0

1

pt
i

0.1
0.5
0.9

0
10
20
30
40
50
60
70
0.1
0.5
0.9

(a) Stationary

0
10
20
30
40
50
60
70
80
Global round t

0

1

pt
i

0.1
0.5
0.9

0
10
20
30
40
50
60
70
0.1
0.5
0.9

(b) Non-stationary with staircase trajectory

0
10
20
30
40
50
60
70
80
Global round t

0

1

pt
i

0.1
0.5
0.9

0
10
20
30
40
50
60
70
0.1
0.5
0.9

(c) Non-stationary with sine trajectory

0
10
20
30
40
50
60
70
80
Global round t

0

1

pt
i

0.1
0.5
0.9

0
10
20
30
40
50
60
70
0.1
0.5
0.9

(d) Non-stationary with interleaved sine trajectory

Figure 6: Examples of client unavailability with probabilistic trajectories. The first row in each
sub-figure plots the probabilistic trajectory of each dynamics. The second row visualizes the simulated
client availability by using a colored box to denote a client is available in that round. The y-axis is
the base probability pi to construct pt
i. In other words, more blank space means that a client is more
scarcely available. We simulate the cases where pi ∈{0.1, 0.5, 0.9}. The detailed construction of pt
i
can be found in Appendix J.3

Table 8: The first round to reach a targeted test accuracy under non-stationary of sine trajectory over 3 random
seeds. We study the first round to reach 1/4, 1/2, 3/4 and 1 of the best test accuracy of each dataset in Table 2,
which is rounded up to the nearest 10% below for ease of presentation. In addition, we sample the mean of
test accuracy every 20 global rounds to mitigate noisy progress. Some algorithms may never attain the targeted
accuracy due to their inferior performance, where we use “–” as a placeholder.

Datasets
SVHN
CIFAR10
CINIC10

Quarters
1/4
1/2
3/4
1
1/4
1/2
3/4
1
1/4
1/2
3/4
1

Test accuracy
20%
40%
60%
80%
15%
30%
45%
60%
10%
20%
30%
40%

FedAWE (ours)
40
120
200
820
20
60
200
1360
0
20
120
540
FedAvg over active clients
20
80
160
900
10
20
120
1060
0
20
40
800
FedAvg over all clients
100
420
960
–
20
60
520
–
0
20
200
–
FedAU
60
100
160
840
10
20
100
960
0
20
80
460
F3AST
40
120
200
1080
20
40
160
1300
0
20
60
540

FedAvg with known pt
i’s
20
40
100
320
10
20
140
620
0
20
40
400
MIFA (memory aided)
20
80
140
600
10
20
80
700
0
20
40
240

We choose γ = 0.3 and P = 20 for all non-stationary dynamics. Next, we visualize the probability
trajectories along with sampled client availability in Fig. 6. The plots confirm the intuition that
interleaved dynamics is the most difficult one, e.g., no clients are available in the case of 0.1 therein.

J.4
Additional results

Staleness studies. Table 8 illustrates the first round to reach a targeted test accuracy under non-
stationary client availability with sine trajectory. Specifications can be found in the caption. It
can be easily checked that, during the initial stage (the first three quarters), FedAWE slightly lags
behind FedAvg over active clients. However, when reaching the final stage (the last quarter),
FedAWE attains the target accuracy in a comparable or lower number of rounds to FedAvg over
active clients in the evaluations on SVHN and CINIC-10 datasets. The slowdown of FedAWE on
CIFAR-10 dataset is worth further investigation. In general, we arrive numerically at the conclusion
that the staleness incurred by implicit gossiping in FedAWE is mild.

Training curves.
In this part,
we show the training curves of FedAvg over active
clients, FedAWE and MIFA. In particular, the presented results of FedAWE are after exponential moving

44


---Page Break---
0
250
500
750
1000 1250 1500 1750 2000

10
2

2 × 10
3

3 × 10
3

4 × 10
3

6 × 10
3

train loss

FedAWE
FedAvg
MIFA

0
250
500
750
1000 1250 1500 1750 2000

0.2

0.4

0.6

0.8

test accuracy

FedAWE
FedAvg
MIFA

(a) Evaluation results on SVHN dataset without exponential moving average

0
250
500
750
1000 1250 1500 1750 2000

10
2

2 × 10
3

3 × 10
3

4 × 10
3

6 × 10
3

train loss

FedAWE
FedAvg
MIFA

0
250
500
750
1000 1250 1500 1750 2000

0.2

0.4

0.6

0.8

test accuracy

FedAWE
FedAvg
MIFA

(b) Evaluation results on SVHN dataset

0
250
500
750
1000 1250 1500 1750 2000

10
2

4 × 10
3

6 × 10
3

train loss

FedAWE
FedAvg
MIFA

0
250
500
750
1000 1250 1500 1750 2000

0.2

0.4

0.6

test accuracy

FedAWE
FedAvg
MIFA

(c) Evaluation results on CIFAR10 dataset

0
250
500
750
1000 1250 1500 1750 2000

10
2

6 × 10
3

7 × 10
3

8 × 10
3

9 × 10
3

train loss

FedAWE
FedAvg
MIFA

0
250
500
750
1000 1250 1500 1750 2000

0.1

0.2

0.3

0.4

0.5

test accuracy

FedAWE
FedAvg
MIFA

(d) Evaluation results on CINIC10 dataset

Figure 7: Missing training curves under non-stationary client unavailability dynamics with sine curve

average [5] under a parameter 0.99. Note that this is to ease down the noisy progress, and for a
neat presentation only, the reported results in the main text and ablation studies are all from raw
data. Fig. 7a plots the train loss and test accuracy from raw data. For example, when compared
with Fig. 7b, EMA eases down the fluctuations but does not change either the trend or the order of
algorithm performance results. All train losses are plotted on a logarithmic scale. The results are
consistent with Table 2.

Impact of system-design parameters. In this part, we study the impact of system-design parameter
including the degree of non-stationarity γ and data heterogeneity α under non-stationary with sine

45


---Page Break---
Table 9: Results after different parameter γ. pt
i = pi · (γ sin(2π/P · t) + (1 −γ)).

Unavailable
Dynamics

Datasets
γ = 0.3
γ = 0.2
γ = 0.1
Algorithms
Train
Test
Train
Test
Train
Test

Non-stationary
(Sine)

0

pt
i

FedAWE (ours)
85.7 ± 0.9 %
85.6 ± 0.9 %
85.7 ± 0.5 %
85.7 ± 0.5 %
85.8 ± 0.6 %
85.7 ± 0.7 %
FedAvg over active
82.1 ± 1.1 %
82.0 ± 1.3 %
82.0 ± 1.2 %
81.9 ± 1.2 %
82.3 ± 0.9 %
82.2 ± 1.0 %
FedAvg over all
71.3 ± 2.5 %
71.3 ± 2.8 %
73.2 ± 2.5 %
73.2 ± 2.8 %
74.0 ± 2.1 %
74.9 ± 2.4 %
FedAU
82.5 ± 1.4 %
82.5 ± 1.3 %
83.5 ± 0.3 %
83.4 ± 0.4 %
83.7 ± 0.3 %
83.6 ± 0.3 %
F3AST
82.3 ± 1.0 %
82.3 ± 1.0 %
82.3 ± 0.9 %
82.6 ± 0.8 %
82.9 ± 0.7 %
82.9 ± 0.6 %
FedAvg with known pt
i’s
86.3 ± 1.0 %
86.0 ± 1.0 %
86.2 ± 1.2 %
86.0 ± 1.4 %
86.4 ± 0.9 %
86.0 ± 0.8 %
MIFA (memory aided)
84.2 ± 0.4 %
84.1 ± 0.4 %
84.6 ± 0.1 %
84.5 ± 0.1 %
84.6 ± 0.1 %
84.4 ± 0.1 %

Table 10: Results after different Dirichlet parameter α. pt
i = pi(γ sin(2π/P · t) + (1 −γ)).

Unavailable
Dynamics

Datasets
α = 0.05
α = 0.1
α = 1.0
Algorithms
Train
Test
Train
Test
Train
Test

Non-stationary
(Sine)

0

pt
i

FedAWE (ours)
82.5 ± 2.1 %
82.5 ± 2.4 %
85.7 ± 0.9 %
85.6 ± 0.9 %
90.6 ± 0.2 %
89.7 ± 0.3 %
FedAvg over active
78.9 ± 1.6 %
78.5 ± 1.8 %
82.1 ± 1.1 %
82.0 ± 1.3 %
88.3 ± 0.1 %
87.5 ± 0.1 %
FedAvg over all
58.5 ± 3.0 %
58.5 ± 3.8 %
71.3 ± 2.5 %
71.3 ± 2.8 %
82.0 ± 0.7 %
81.9 ± 0.6 %
FedAU
79.5 ± 1.6 %
79.5 ± 1.7 %
82.5 ± 1.4 %
82.5 ± 1.3 %
88.4 ± 0.1 %
87.6 ± 0.2 %
F3AST
78.9 ± 1.3 %
78.9 ± 1.3 %
82.3 ± 1.0 %
82.3 ± 1.0 %
87.6 ± 0.1 %
87.0 ± 0.1 %
FedAvg with known pt
i’s
84.2 ± 1.0 %
83.5 ± 1.0 %
86.3 ± 1.0 %
86.0 ± 1.0 %
91.5 ± 0.3 %
90.5 ± 0.1 %
MIFA (memory aided)
82.6 ± 0.1 %
82.6 ± 0.0 %
84.2 ± 0.4 %
84.1 ± 0.4 %
88.4 ± 0.1 %
87.5 ± 0.1 %

trajectory. The results are in Table 9 and Table 10. Overall, FedAWE keeps outperforming the
algorithms not assisted by memories or known statistics.

In Table 10, clients’ local data becomes more heterogeneous when α increases. We can see a
clear increase trend in accuracy. However, FedAWE remains to attain the best accuracies both train
and test when compared to the algorithms not aided by memory or known statistics. Moreover, it
outperforms MIFA, which consumes a lot of storage space, when α = 0.1 and 1.0. The observations
confirm the practicality of FedAWE.

46


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]

Justification: We have faithfully stated our contributions in both the abstract and introduction.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Please refer to Appendix A for details.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The regulatory assumptions are stated in Section 6. Due to space limitations,
we are unable to present all the missing proofs and intermediate results in the main text.
They are deferred to Appendix. Please refer to Table of Contents for details.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]

Justification: We provide detailed experimental and the hyperparameter setups in Section 7
and Appendix J.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: Our evaluations are based on open-accessed datasets that are publically avail-
able. An official implementation code is provided through a GitHub link.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]

Justification: Experimental setting/details are important parts of reproducing our results. We
provide the details in Section 7 and Appendix J to the best of our ability.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]

Justification: Our results are averaged over multiple random seeds and accompanied by error
bars
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

47


---Page Break---
Answer: [Yes]
Justification: Please find the software/hardware specifications in Appendix J.2.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The NeurIPS code of ethics is strictly enforced throughout our research.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: We have discussed broader impacts in Appendix B. We are unaware of any
negative impacts.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risks
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]

Justification: The existing assets used in this paper has been adequately cited or credited to.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]

Justification: We have documented the experiment details in Section 7 and Appendix J.2. In
addition, we provide our code with clear details and examples.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects

48


---Page Break---
