Does Worst-Performing Agent Lead the Pack?
Analyzing Agent Dynamics in Unified Distributed SGD

Jie Hu
Yi-Ting Ma
Do Young Eun
Department of Electrical and Computer Engineering
North Carolina State University
{jhu29, yma42, dyeun}@ncsu.edu

Abstract

Distributed learning is essential to train machine learning algorithms across hetero-
geneous agents while maintaining data privacy. We conduct an asymptotic analysis
of Unified Distributed SGD (UD-SGD), exploring a variety of communication pat-
terns, including decentralized SGD and local SGD within Federated Learning (FL),
as well as the increasing communication interval in the FL setting. In this study,
we assess how different sampling strategies, such as i.i.d. sampling, shuffling, and
Markovian sampling, affect the convergence speed of UD-SGD by considering
the impact of agent dynamics on the limiting covariance matrix as described in
the Central Limit Theorem (CLT). Our findings not only support existing theories
on linear speedup and asymptotic network independence, but also theoretically
and empirically show how efficient sampling strategies employed by individual
agents contribute to overall convergence in UD-SGD. Simulations reveal that a few
agents using highly efficient sampling can achieve or surpass the performance of
the majority employing moderately improved strategies, providing new insights
beyond traditional analyses focusing on the worst-performing agent.

1
Introduction

Distributed learning deals with the training of models across multiple agents over a communication
network in a distributed manner, while addressing the challenges of privacy, scalability, and high-
dimensional data [11, 55]. Each agent i ∈[N] holds a private dataset Xi and an agent-specified
loss function Fi : Rd × Xi →R that depends on the model parameter θ ∈Rd and a data point
X ∈Xi. The goal is then to find a local minima θ∗of the objective function f(θ) ≜1

N
PN
i=1 fi(θ),
where agent i’s loss function fi(θ) ≜EX∼Di[Fi(θ, X)] and Di represents the target distribution
of data for agent i.1 Each agent i can locally compute the gradient ∇Fi(θ, X) ∈Rd w.r.t. θ for
every sampled data point X ∈Xi. Due to the distributed nature, {Di}i∈[N] and {Xi}i∈[N] are not
necessarily identically distributed over [N] so that the minima of each local function fi(θ) can be far
away from L. This is particularly relevant in decentralized training data, e.g., Federated Learning
(FL) with heterogeneous data across data centers or devices [81, 31].

In this paper, we focus on Unified Distributed SGD (UD-SGD), where each agent i ∈[N] updates its
model parameter θi
n+1 in a two-step process:
Local update: θi
n+1/2 = θi
n −γn+1∇Fi(θi
n, Xi
n),
(1a)

1Throughout the paper we don’t impose convexity assumption on f(θ). For convex f(θ), L is the global
minima. For non-convex f(θ), L represents the collection of local minima, which is of great interest in
neural network training for sufficiently good performance [20, 19]. With an additional condition such as the
Polyak-Lojasiewicz inequality, non-convex f(θ) is ensured to have a unique minima [1, 75, 78].

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Aggregation: θi
n+1 = PN
j=1 wn(i, j)θj
n+1/2,
(1b)

where γn denotes the step size, Xi
n is the data sampled by agent i at time n (i.e., agent dynamics), and
Wn =[wn(i, j)]i,j∈[N] represents the doubly-stochastic communication matrix satisfying wn(i, j) ≥
0 and 1T Wn =1T , Wn1=1. In the special case of N = 1, (1) simplifies to the vanilla SGD where
Wn = 1 for all n. UD-SGD covers a wide range of distributed algorithms, e.g., decentralized SGD
(DSGD) [71, 80, 61, 68], distributed SGD with changing topology (DSGD-CT) [24, 43], local SGD
(LSGD) in FL [55, 76], and its variant aimed at reducing communication costs (LSGD-RC) [51].

𝒊

𝑿𝒏𝒊

𝑿𝒏+𝟏

𝒊

𝒋
𝒘𝒏(𝒊, 𝒋)

𝒘𝒏(𝒋, 𝒊)

𝑿𝒏

𝒋
𝑿𝒏+𝟏

𝒋

dataset 𝒳𝒊
dataset 𝒳𝒋

Communication 

pattern 𝑾𝑛

Heterogeneous 

sampling 
strategy 𝑋𝑖, 𝑋𝑗

Figure 1: GD-SGD algorithm with a communi-
cation network of N = 5 agents, each holding
potentially distinct datasets; e.g., agent j (in blue)
samples Xj i.i.d. and agent i (in red) samples Xi
via Markovian trajectory.

Versatile Communication Patterns {Wn}:
For visualization, we depict the scenarios of
UD-SGD (1) in Figure 1. In DSGD, each agent
(node) in the graph communicates with its neigh-
bors after each SGD computation via Wn, rep-
resenting the underlying network topology. As
a special case, central server-based aggregation,
forming a fully connected network, translates
Wn into a rank-1 matrix Wn = 11T /N. To
minimize communication expenses, FL variants
allow each agent to perform multiple SGD steps
before aggregation [55, 67, 76], resulting in a
communication interval of length K and a con-
sistent pattern Wn = W for n = mK, ∀m ∈
N, and Wn = IN otherwise. In particular, i)
W=11T /N corresponds to LSGD with full agent participation (LSGD-FP) [76, 42, 51]; ii) W is a
random matrix generated by partial agent participation (LSGD-PP) [55, 18, 74]; iii) W is generated
by Metropolis-Hasting algorithm in decentralized setting, e.g., hybrid LSGD (HLSGD) [37, 32] and
decentralized FL (DFL) [46, 77, 16]. We defer further discussion of W to Appendix F.1.

Markovian vs i.i.d. Sampling: Agents typically employ i.i.d. or Markovian sampling, as illustrated
in the bottom brown box of Figure 1. In cases where agents have full access to their data, DSGD with
i.i.d sampling has been extensively studied [60, 43, 61, 47]. In FL, many application-oriented LSGD
variants have been investigated [51, 18, 77, 32, 37, 53]. However, these works solely focus on i.i.d.
sampling, restricting their applicability to Markovian sampling scenarios.

Markovian sampling, which has received increased attention in limited settings (see Table 1), is
vital where agents lack independent data access. For instance, in statistical applications, agents with
an unknown a priori distribution often use Markovian sampling over i.i.d. sampling [40, 63]. In
HLSGD across device-to-device (D2D) networks [32, 37], random walks reduce communication costs
compared to the frequent aggregations required by Gossip algorithms [38, 28, 4]. For single-agent
scenarios, vanilla SGD with Markovian noise, as applied in a D2D network, has shown improved
communication efficiency and privacy [69, 28, 35]. In contrast, for agents with full data access,
Markov Chain Monte Carlo (MCMC) methods can be more efficient than i.i.d. sampling, especially
in high-dimensional spaces with constraints [27, 40], where acceptance-rejection methods [12] lead to
computational inefficiency (e.g., wasted samples) due to multiple rejections before obtaining a sample
that satisfies constraints [26, 69]. In addition, shuffling methods can be considered as high-order
Markov chains [38], which achieves faster convergence than i.i.d. sampling [1, 79, 78].

Limitations of Non-Asymptotic Analysis on Agent’s Sampling Strategy: Recent studies on the
non-asymptotic behavior of DSGD and LSGD variants under Markovian sampling, as summarized in
Table 1, have made significant strides. However, these works often fall short in accurately revealing
the statistical influence of each agent dynamics {Xi
n} on the performance of UD-SGD. For instance,
[71, 68] proposed the error bound O( 1/ log2(1/ρ)

n1−a
), where a ∈(0.5, 1] and ρ denotes the identical
mixing rate for all agents, overlooking agent heterogeneity in sampling strategy. A similar assumption
to ρ is also evident in [42]. More recent contributions from [80, 72] have attempted to relax these
constraints by considering a finite-time bound of O(τ 2
mix/(n + 1)), where τmix is the mixing time
of the slowest agent. This approach, however, inherently focuses on the worst-performing agent,
neglecting how other agents with faster mixing rates might positively influence the system.2 Such
an analysis fails to capture the collective impact of other agents on the overall system performance,

2Although improving the finite-time upper bound to distinguish each agent may not be the focus of the
aforementioned works, their analyses require every Markov chain to be close to some neighborhood of its

2


---Page Break---
Table 1: Comparison of recent works in distributed learning: We classify the communication patterns
into seven categories, i.e., DSGD, DSGD-CT, LSGD-FP, LSGD-PP, LSGD-RC, HLSGD and DFL. We
mark ‘UD-SGD’ when all aforementioned patterns are included and the detailed discussion on {Wn}
is referred to Appendix F.1. Abbreviations: ‘Asym.’ = ‘Asymptotic’, ‘D.A.B’ = ‘Differentiating
Agent Behavior’, ‘L.S.’ = ‘Linear Speedup’, ‘A.N.I.’ = ‘Asymptotic Network Independence’.

Reference
Analysis
Sampling
Communication Pattern
D.A.B.
L.S.
A.N.I.

[58]
Asym.
i.i.d.
DSGD
✓
✓
✓

[51]
Asym.
i.i.d.
LSGD-RC
✓
✓
N/A

[43, 47]
Non-Asym.
i.i.d.
DSGD-CT
×
✓
×

[61]
Non-Asym.
i.i.d.
DSGD
×
×
✓

[18, 53]
Non-Asym.
i.i.d.
LSGD-PP
×
✓
N/A

[37, 32]
Non-Asym.
i.i.d.
HLSGD
×
✓
×

[77, 16]
Non-Asym.
i.i.d.
DFL
×
✓
×

[71, 80, 68]
Non-Asym.
Markov
DSGD
×
×
×

[42, 72]
Non-Asym.
Markov
LSGD-FP
×
✓
N/A

[69, 4, 28]
Non-Asym.
Markov
N/A (single agent)
N/A
N/A
N/A

[38, 52]
Asym.
Markov
N/A (single agent)
N/A
N/A
N/A

Our Work
Asym.
Markov
UD-SGD
✓
✓
✓

a crucial aspect in large-scale applications where identifying and managing the worst-performing
agent is challenging due to privacy concerns or sporadic unreachability. Since agents in distributed
learning have the freedom to choose their sampling strategies, it’s vital to understand how each
agent’s improved sampling approach contributes to the overall convergence speed of the UD-SGD
algorithm. This understanding is key to enhancing system performance, particularly in large-scale
machine learning scenarios where agent heterogeneity is a defining feature.

Rationale for Asymptotic Analysis: Recent trends in convergence analysis have leaned towards
non-asymptotic methods, yet it’s crucial to recognize the complementary role of asymptotic analysis
for a better understanding of convergence behaviors, as highlighted in [9, 56, 25, 39]. For vanilla SGD,
[59, 17] emphasized that central limit theorem (CLT) is far less asymptotic than it may appear under
both i.i.d. and Markovian sampling. Notably, the limiting covariance matrix, a key statistical feature
in vanilla SGD’s CLT, also prominently features in high-probability bound [59], explicit finite-time
bound [17] and 1-Wasserstein distance in the non-asymptotic CLT [66]. [38] further underscored
this by numerically showing that the limiting covariance matrix provides a more precise depiction of
convergence than the mixing rates often used in finite-time upper bounds [26, 69]. Moreover, they
argued that finite-time analysis may not suitably apply to certain efficient high-order Markov chains,
due to the lack of comparative mixing-rate metrics.

Our Contributions: We present an asymptotic analysis of the UD-SGD algorithm (1) under hetero-
geneous agent dynamics {Xi
n} and a large family of communication patterns {Wn}. Specifically,

• Under appropriate assumptions, all agents performing (1) asymptotically reach the consensus and
find θ∗: ∀i ∈[N], θn ≜1

N
PN
i=1 θi
n denotes the average model parameter among all agents, we have

lim
n→∞∥θi
n−θn∥=0,
lim
n→∞∥θn−θ∗∥=0 a.s.
(2)

Moreover, we derive the CLT of UD-SGD in the form of

γ−1/2
n
(θn −θ∗)
dist.
−−−−→
n→∞N (0, V) .
(3)

Our framework addresses technical challenges in quantifying consensus error under various commu-
nication patterns and slowly increasing communication interval. This shows a substantial extension
compared to previous studies [58, 43, 51], particularly in regulating the growth of communication

stationary distribution. This naturally incurs a maximum operator, and thus convergence is strongly influenced
by the slowest mixing rate, i.e., the worst-performing agent.

3


---Page Break---
intervals (Assumption 2.3-ii) and in proving the scaled consensus error’s boundedness (Lemma B.1).
Furthermore, we reformulate UD-SGD as a stochastic approximation-like iteration and tackle the
Markovian noise term using the Poisson equation, a technique previously confined only to vanilla
SGD with Markovian sampling [17, 38, 52]. The key here is to devise the noise decomposition that
separates the consensus error among all agents from the error caused by the bias from the Markov
chain, which aligns with the target distribution only asymptotically at infinity, not at finite times.

• In analyzing (3), we derive the exact form of V as
1
N2
PN
i=1 Vi. Here, Vi is the limiting covariance
matrix of agent i, which depends mainly on its sampling strategy {Xi
n}. This allows us to show
that improving individual agents’ sampling strategy can reduce the covariance in CLT, which in turn
implies a smaller mean-square error (MSE) for large time n. This is a significant advancement over
previous finite-sample bounds that only account for the worst-performing agent and do not fully
capture the effect of individual agent dynamics on overall system performance. Our CLT result (3)
also treats recent findings in [38] as a very special case with N = 1, where the relationship therein
between the sampling efficiency of the Markov chain and the limiting covariance matrix in the CLT
of vanilla SGD, can carry over to our UD-SGD.

• We demonstrate that our analysis supports recent findings from studies such as [42], which exhibited
linear speedup scaling with the number of agents under LSGD-FP with Markovian sampling; and
[62, 61], which examined the notion of ‘asymptotic network independence’ for DSGD with i.i.d.
sampling, where the convergence of the algorithm (1) at large time n depends solely on the left
eigenvector of Wn ( 1

N 1 considered in this paper) rather than the specific communication network
topology encoded in Wn, but now under Markovian sampling. We extend these findings in view of
CLT to a broader range of communication patterns {Wn} and general sampling strategies {Xi
n}.

• We conduct numerical experiments using logistic regression and neural network training with
several choices of agents’ sampling strategies, including a recently proposed one via nonlinear
Markov chain [25]. Our results uncover a key phenomenon: a handful of compliant agents adopting
highly efficient sampling strategies can match or exceed the performance of the majority using
moderately improved strategies. This finding is crucial for practical optimization in large-scale
learning systems, moving beyond the current literature that only considers the worst-performing
agent in more restrictive settings.

2
Preliminaries

Basic Notations: We use ∥v∥to indicate the Euclidean norm of a vector v ∈Rd and ∥M∥to indicate
the spectral norm of a matrix M ∈Rd×d. The identity matrix of dimension d is denoted by Id, and
the all-one (resp. all-zero) vector of dimension N is denoted by 1 (resp. 0). Let J ≜11T /N. The
diagonal matrix with the entries of v on the main diagonal is written as diag(v). We also use ‘⪰’ for
Loewner ordering such that A ⪰B is equivalent to xT (A −B)x ≥0 for any x ∈Rd.

Asymptotic Covariance Matrix: Asymptotic variance is a widely used metric for evaluating the
second-order properties of Markov chains associated with a scalar-valued test function in the MCMC
literature, e.g., Chapter 6.3 [12], and asymptotic covariance matrix is its multivariate version for a
vector-valued function. Specifically, we consider a finite, irreducible, aperiodic and positive recurrent
(ergodic) Markov chain {Xn}n≥0 with transition matrix P and stationary distribution π, and the
estimator ˆµn(g) ≜1

n
Pn−1
s=0 g(Xs) for any vector-valued function g : [N] →Rd. According to
the ergodic theorem [12, 13], we have limn→∞ˆµn(g) = Eπ(g) a.s.. As defined in [13, 38], the
asymptotic covariance matrix ΣX(g) for a vector-valued function g(·) is given by

ΣX(g)≜lim
n→∞n · Var(ˆµn(g))= lim
n→∞
1
n · E

∆n∆T
n
	
,
(4)

where ∆n ≜Pn−1
s=0 (g(Xs) −Eπ(g)). By following the algebraic manipulations in [12, Theorem
6.3.7] for asymptotic variance (univariate version), we can rewrite (4) in a matrix form such that

ΣX(g) = GT diag(π)
 
Z −IN + 1πT 
G,
(5)

where G ≜[g(1), · · · , g(N)]T ∈RN×d and Z ≜[IN −P + 1πT ]−1. This matrix form explicitly
shows the dependence on the transition matrix P and its stationary distribution π, and will be utilized
in our Theorem 3.3.

4


---Page Break---
Model Description: The UD-SGD in (1) can be expressed in a compact iterative form, i.e., we have

θi
n+1 = PN
j=1 wn(i, j)(θj
n −γn+1∇Fj(θj
n, Xj
n)),
(6)

at each time n, where each agent i samples according to its own Markovian trajectory {Xi
n}n≥0
with stationary distribution πi such that EX∼πi[Fi(θ, X)]=fi(θ). Let Kl denote the communication
interval between the (l −1)-th and l-th aggregation among N agents, and nl ≜Pl
m=1 Km be the
time instance for the l-th aggregation. We also define τn ≜minl{l : nl ≥n} as the index of the
upcoming aggregation at time n such that Kτn indicates the communication interval for the τn-th
aggregation, or more precisely, the length of the communication interval that includes the time index
n. The communication pattern follows that Wn = In if n ̸= nl and Wn = W otherwise for
l ≥1, where the examples of W will be discussed in Appendix F.1. Note that i) when Kl = 1, (6)
reduces to DSGD; ii) when Kl = K > 1, (6) becomes the local SGD in FL. iii) When Kl increases
with l, we recover some choices of Kl studied in [51] beyond LSGD-RC with i.i.d. sampling. This
increasing communication interval aims to further reduce the frequency of aggregation among agents
for lower communication costs, but now under a Markovian sampling setting and a wider range of
communication patterns. We below state the assumptions needed for the main theoretical results.
Assumption 2.1 (Regularity of the gradient). For each i ∈[N] and X ∈X i, the function Fi(θ, X)
is L-smooth in terms of θ, i.e., for any θ1, θ2 ∈Rd,
∥∇Fi(θ1, X) −∇Fi(θ2, X)∥≤L∥θ1 −θ2∥.
(7)

In addition, we assume that the objective function f is twice continuously differentiable and µ-strongly
convex only around the local minima θ∗∈L, i.e.,

H ≜∇2f(θ∗) ⪰µId.
(8)

Assumption 2.1 imposes the regularity conditions on the gradient ∇Fi(·, X) and Hessian matrix
of the objective function f(·), as is commonly assumed in [10, 45, 29, 38]. Note that (7) requires
per-sample Lipschitzness of ∇Fi and is stronger than the Lipschitzness of its expected version ∇fi,
which is commonly assumed under i.i.d sampling setting [73, 50, 30]. However, we remark that this
is in line with previous work on DSGD and LSGD-FP under Markovian sampling as well [71, 42, 80],
because ∇Fi(θ, X) is no longer the unbiased stochastic version of ∇fi(θ) and the effect of {Xi
n}
has to be taken into account in the analysis. The local strong convexity at the minimizer is commonly
assumed to analyze the convergence of the algorithm under both asymptotic and non-asymptotic
analysis [10, 29, 38, 45, 52, 80].
Assumption 2.2 (Ergodicity of Markovian sampling). {Xi
n}n≥0 is an ergodic Markov chain
with stationary distribution πi such that EX∼πi[Fi(θ, X)] = fi(θ), and is independent from
{Xj
n}n≥0, j ̸= i.

The ergodicity of the underlying Markov chains, as stated in Assumption 2.2, is commonly assumed
in the literature [26, 69, 80, 42, 38]. This assumption ensures the asymptotic unbiasedness of the loss
function Fi(θ, ·), which takes i.i.d. sampling as a special case.
Assumption 2.3 (Decreasing step size and slowly increasing communication interval). i) For bounded
communication interval Kτn ≤K, ∀n, we assume the polynomial step size γn = 1/na and a ∈
(0.5, 1]; Or ii) If Kτn →∞as n →∞, we assume γn = 1/n and define ηn = γnKL+1
τn
, where the
sequence {Kl}l≥0 satisfies P

n η2
n < ∞, Kτn = o(γ−1/2(L+1)
n
), and liml→∞ηnl+1/ηnl+1+1 = 1.

In Assumption 2.3, the polynomial step size γn is standard in the literature and it has the property
P

n γn = ∞, P

n γ2
n < ∞[17, 38]. Inspired by [51], we introduce ηn to control the step size
within each l-th communication interval with length Kl to restrict the growth of Kl. Specifically,
P

n η2
n < ∞and Kτn = o(γ−1/2(L+1)
n
) ensure that ηn →0 and Kτn does not increase too fast
in n. liml→∞ηnl+1/ηnl+1+1 = 1 sets the restriction on the increment from nl to nl+1. Several
practical forms of Kl suggested by [51], including Kl ∼log(l) and Kl ∼log log(l), also satisfy
Assumption 2.3-ii). We defer to Appendix A the mathematical verification of these two types of Kl,
together with the practical implications of increasing communication interval Kl.
Remark 1. In Assumption 2.3, we incorporate an increasing communication interval along with
a step size γn = 1/n. This complements the choice of step size γn in [51, Assumption 3.3], where
γn = 1/na for a ∈(0.5, 1). It is important to note, however, that the increasing communication
interval specified in [51, Assumption 3.2] is applicable only in i.i.d sampling. Under the Markovian

5


---Page Break---
sampling framework, the expression ∇Fi(θ, X)−∇fi(θ) loses its unbiased and Martingale difference
properties. Consequently, the Martingale CLT application as utilized by [51] does not directly extend
to Markovian sampling. To address this, we adapted techniques from [29, 58] to accommodate the
increasing communication interval within the Markovian sampling setting and various communication
patterns. This adaptation necessitates γn = 1/n, a specification not covered in [51]. Exploring more
general forms of Kl that could relax this assumption is outside the scope of our current study.
Assumption 2.4 (Stability on model parameter). We assume supn ∥θi
n∥< ∞almost surely ∀i ∈[N].

Assumption 2.4 claims that the sequence of {θi
n} always remains in a path-dependent compact set. It
is to ensure the stability of the algorithm that serves the purpose of analyzing the convergence, which
is often assumed under the asymptotic analysis of vanilla SGD with Markovian noise [23, 29, 52]. As
mentioned in [58, 70], checking Assumption 2.4 is challenging and requires case-by-case analysis,
even under i.i.d. sampling. Only recently the stability of SGD under Markovian sampling has been
studied in [9], but the result for UD-SGD remains unknown in the literature. Thus, we analyze each
agent’s sampling strategy in the asymptotic regime under this stability condition.
Assumption 2.5 (Contraction property of communication matrix). i). {Wn}n≥0 is independent of
the sampling strategy {Xi
n}n≥0 for all i ∈[N] and is assumed to be doubly-stochastic for all n; ii).
At each aggregation step nl, Wnl is independently generated from some distribution Pnl such that
∥EW∼Pnl[WT W]−J∥≤C1 <1 for some constant C1.

The doubly-stochasticity of Wn in Assumption 2.5-i) is widely assumed in the literature [54, 24,
43, 80]. Assumption 2.5-ii) is a contraction property to ensure that agents employing UD-SGD will
asymptotically achieve the consensus, which is also common in [7, 24, 80]. Examples of W that
satisfy Assumption 2.5-ii), e.g., Metropolis-Hasting matrix, partial agent participation in FL, are
deferred to Appendix F.1 due to space constraint.

3
Asymptotic Analysis of UD-SGD

Almost Sure Convergence: Let θn ≜1

N
PN
i=1 θi
n represent the consensus among all the agents at
time n, we establish the asymptotic consensus of the local parameters θi
n, as stated in Lemma 3.1.
Lemma 3.1. Under Assumptions 2.1, 2.3, 2.4 and 2.5, the consensus error θi
n−θn diminishes to zero
at the rate specified below: Almost surely, for every agent i ∈[N],
θi
n−θn
=
O(γn)
under Assum. 2.3-i),
O(ηn)
under Assum. 2.3-ii).
(9)

Lemma 3.1 indicates that all agents asymptotically reach consensus at a rate of O(γn) (or O(ηn)).
This finding extends the scope of [58, Proposition 1], incorporating considerations for Markovian
sampling, FL settings, and increasing communication interval Kl. The proof, detailed in Appendix B,
primarily tackles the challenge of establishing the boundedness of the sequences {γ−1
n (θi
n −θn)} (or
{η−1
n (θi
n −θn)}) almost surely for all i ∈[N]. This is specifically analyzed in Lemma B.1. Next,
with additional Assumption 2.2, we are able to obtain the almost sure convergence of θn to θ∗∈L.
Theorem 3.2. Under Assumptions 2.1 - 2.5, the consensus θn converges to L almost surely, i.e.,
lim supn infθ∗∈L ∥θn −θ∗∥= 0
a.s.
(10)
Theorem 3.2 is achieved by decomposing the Markovian noise term ∇Fi(θi
n, Xi
n) −∇fi(θi
n), using
the Poisson equation technique as discussed in [6, 29, 17], into a Martingale difference noise term,
along with additional noise terms. We then reformulate (6) into an iteration akin to stochastic
approximation, as depicted in (56). The subsequent step involves verifying the conditions on these
noise terms under our stated assumptions. Crucially, this theorem also establishes that UD-SGD
ensures an almost sure convergence of each agent to a local minimum θ∗∈L, even in scenarios
where the communication interval Kl gradually increases, in accordance with Assumption 2.3-ii).
The detailed proof of this theorem is provided in Appendix C.

Central Limit Theorem: Let Ui ≜ΣXi(∇Fi(θ∗, ·)) represent the asymptotic covariance matrix
(defined in (5)) associated with each agent i ∈[N], given their sampling strategy {Xi
n} and function
∇Fi(θ∗, ·). Define U ≜
1
N2
PN
i=1 Ui. We assume the polynomial step-size γn ∼γ⋆/na, a∈(0.5, 1]
and γ⋆> 0. In the case of a = 1, we further assume γ⋆> 1/2µ, where µ is defined in (8).
For notational simplicity, and without loss of generality, our remaining CLT result is stated while
conditioning on the event that {θn →θ∗} for some θ∗∈L.

6


---Page Break---
Theorem 3.3. Let Assumptions 2.1 - 2.5 hold. Then,

γ−1/2
n
(θn −θ∗)
dist.
−−−−→
n→∞N(0, V),
(11)

where the limiting covariance matrix V is in the form of

V =
R ∞
0
eMtUeMtdt.
(12)

Here, we have M = −H if a ∈(0.5, 1), or M = Id/2γ⋆−H if a = 1, where H is defined in (8).

Moreover, let ¯θn = 1

n
Pn−1
s=0 θs and V′ = H−1UH−1. For a ∈(0.5, 1), we have
√n(¯θn −θ∗)
dist.
−−−−→
n→∞N(0, V′),
(13)

The proof, presented in Appendix D, addresses the technical challenges in deriving the CLT for
UD-SGD, specifically the second-order conditions in decomposing the Markovian noise term, which
is not present in the i.i.d. sampling case [58, 43, 51]. We decompose ∇Fi(θn, Xi
n)−∇fi(θn) into
three parts in (48) using Poisson equation: ei
n+1, νi
n+1, ξi
n+1. The consensus error θi
n −θn embedded
in noise terms ei
n+1 and ξi
n+1 is a new factor, whose characteristics have been quantified in our
Lemma 3.1 but are not present in the single-agent scenario analyzed as an application of stochastic
approximation in [22, 29]. The specifics of this analysis are expanded upon in Appendices D.1 to
D.3. We require γ⋆>1/2µ for a=1 to ensure that the largest eigenvalue of M is negative, as this
is a necessary condition for the existence of V in (12) (otherwise integration diverges). In the case
where there is only one agent (N =1), V and V′ reduce to the matrices specified in the CLT result
of vanilla SGD [29, 38, 52]. In addition, for a special case of constant communication interval in
Assumption 2.3-i) and i.i.d. sampling as shown in Table 1, we recover the CLT of LSGD-RC in [51].
See Appendix E for detailed discussions.

Theorem 3.3 has significant implications for the MSE of {θn} for large time n, i.e., E[∥θn−θ∗∥2]=
Pd
i=1 eT
i E[(θn−θ∗)(θn−θ∗)T ]ei ≈γn
Pd
i=1 eT
i Vei = γnTr(V), where ei is the d-dimensional
vector of all zeros except 1 at the i-th entry. This indicates that a smaller limiting covariance matrix
V, according to the Loewner order, results in a smaller trace of V and consequently in a reduced
MSE for large n. Consideration for smaller V will be presented in the next section, where agents
have the opportunity to improve their sampling strategies.
Remark 2. Studies by [62, 61] have shown that in DSGD with a fixed doubly-stochastic matrix
W, the influence of communication topology diminishes after a transient period. Our Theorem 3.3
extends these findings to Markovian sampling and a broader spectrum of communication patterns as
in Table 1. This extension is based on the fact that the consensus error, impacted primarily by the
communication pattern, decreases faster than the CLT scale O(√γn) and is thus not the dominant
factor in the asymptotic regime, as suggested by Lemma 3.1.
Remark 3. Recent studies have highlighted linear speedup with increasing number of agents N in
the dominant term of their finite-sample error bounds under DSGD-CT with i.i.d. sampling [43] and
LSGD-FC with Markovian sampling [42]. However, our Theorem 3.3 demonstrates this phenomenon
under more diverse communication patterns and Markovian sampling in Table 1 via the leading term
V in our CLT. Specifically, it scales with 1/N, i.e. V= ¯V/N, where ¯V= 1

N
PN
i=1 Vi denotes the
average limiting covariance matrices across all N agents and Vi =
R ∞
0
eMtUieMtdt, suggesting
that the MSE E[∥θn −θ∗∥2] will be improved by 1/N. A similar argument also applies to V′ in (13),
i.e., V′ = ¯V′/N, where ¯V′ = 1

N
PN
i=1 V′
i and V′
i = H−1UiH−1.

Impact of Agent’s Sampling Strategy: In the literature, the mixing time-based technique has
been widely used in the non-asymptotic analysis in SGD, DSGD and various LSGD variants in FL
[26, 69, 68, 80, 42], i.e., for each agent i ∈[N] and some constant C,

∥∇Fi(θ, Xi
n) −∇fi(θ)∥≤C∥θ∥ρn
i ,
(14)

where ρi is the mixing rate of the underlying Markov chain. However, typical non-asymptotic analyses
often rely on ρ ≜maxi ρi among N agents, i.e., the worst-performing agent in their finite-time
bounds [80, 72], or assume an identical mixing rate across all N agents [42, 68].

In contrast, Remark 3 highlights that each agent holds its own limiting covariance matrices Vi and
V′
i, which are predominantly governed by the matrix Ui, capturing the agent’s sampling strategy

7


---Page Break---
{Xi
n} and contributing equally to the overall performance of UD-SGD. For each agent i, denote by
UX
i and UY
i the asymptotic covariance matrices associated with two candidate sampling strategies
{Xi
n} and {Y i
n}, respectively. Let VX and VY be the limiting covariance matrices of the distributed
system in (12), where agent i employs {Xi
n} and {Y i
n}, respectively, while keeping other agents’
sampling strategies unchanged. Then, we have the following result.
Corollary 3.4. For agent i, if there exist two sampling strategies {Xi
n}n≥0 and {Y i
n}n≥0 such that
UX
i ⪰UY
i , we have VX ⪰VY .
Corollary 3.4 directly follows from the definition of Loewner ordering, and Loewner ordering being
closed under addition (i.e., A⪰B implies A+C⪰B+C). It demonstrates that even a single agent
improves its sampling strategy from {Xi
n} to {Y i
n}, it leads to an overall reduction in V (in terms
of Loewner ordering), thereby decreasing the MSE and benefiting the entire group of N agents.
The subsequent question arises: How do we identify an improved sampling strategy {Y i
n} over the
baseline {Xi
n}?

This question has been partially addressed by [57, 48, 38], which qualitatively investigates the
‘efficiency ordering’ of two sampling strategies. In particular, [38, Theorem 3.6 (i)] shows that
sampling strategy {Yn} is more efficient than {Xn} if and only if ΣX(g) ⪰ΣY (g) for any
vector-valued function g(·) ∈Rd. Consequently, in the UD-SGD framework, employing a more
efficient sampling strategy {Y i
n} over the baseline {Xi
n} by agent i leads to ΣXi(∇Fi(θ∗, ·)) ⪰
ΣY i(∇Fi(θ∗, ·)), thus satisfying UX
i ⪰UY
i . This finding, as per Corollary 3.4, implies an overall
improvement in UD-SGD.

For illustration purposes, we list a few examples where two competing sampling strategies follow
efficiency ordering: i) When an agent has complete access to the entire dataset (e.g., deep learning),
shuffling techniques like single shuffling and random reshuffling are more efficient than i.i.d. sampling
[38, 78]; ii) When an agent works with a graph-like data structure and employs a random walk, e.g.,
agent i in Figure 1, using non-backtracking random walk (NBRW) is more efficient than simple
random walk (SRW) [48]. iii) A recently proposed self-repellent random walk (SRRW) is shown to
achieve near-zero sampling variance, indicating even higher sampling efficiency than NBRW and
SRW [25].3 This random-walk-based sampling finds a particular application in large-scale FL within
D2D networks (e.g., mobile networks, wireless sensor networks), where each agent acts as an edge
server or access point, gathering information from the local D2D network [37, 32]. Employing a
random walk over local D2D network for each agent constitutes the sampling strategy.

Theorem 3.3 and Corollary 3.4 not only qualitatively compare these sampling strategies but also
allow for a quantitative assessment of the overall system enhancement. Since every agent contributes
equally to the limiting covariance matrix V of the distributed system as in Remark 3, a key application
scenario is to encourage a subset of compliant agents to adopt highly efficient strategies like SRRW,
potentially yielding better performance than universally upgrading to slightly improved strategies like
NBRW. This approach, more feasible and impactful in large-scale machine learning scenarios where
some agents cannot freely modify their sampling strategies, is a unique aspect of our framework not
addressed in previous works focusing on the worst-performing agent [80, 42, 68, 72].

4
Experiments

In this section, we empirically evaluate the effect of agents’ sampling strategies under various
communication patterns in UD-SGD. We consider the L2-regularized binary classification problem

min
θ
f(θ) ≜1

N

N
X

i=1
fi(θ), with fi(θ)= 1

B

B
X

j=1
log

1+eθT xi,j
−yi,j
 
θT xi,j

+ κ

2 ∥θ∥2,
(15)

where the feature vector xi,j and its corresponding label yi,j are held by agent i, with a penalty
parameter κ set to 1. We use the ijcnn1 dataset [14] with 22 features in each data point and 50k
data points in total, which is evenly distributed to two groups with 50 agents each (N = 100 agents

3Note that SRRW is a nonlinear Markov chain that depends on the relative visit counts of each node in the
graph. While its application in single-agent optimization has been studied in [39], expanding the theoretical
examination of SRRW to multi-agent scenarios is beyond the scope of this paper. However, we can still
numerically evaluate the performance of UD-SGD with multiple agents on general communication matrices
using SRRW as a highly efficient sampling strategy in Section 4.

8


---Page Break---
10000
12500
15000
17500
20000
22500
25000
27500
30000
Number of steps (n)

1.0

1.5

2.0

2.5

3.0

MSE 
n
*
2

1e
5

50 SRW
50 NBRW
40 SRW, 10 SRRW
30 SRW, 20 SRRW

101
102
103

Number of steps (n)

10
5

10
4

10
3

10
2

10
1

100

MSE 
n
*
2

Centralized Learning
DSGD-VT

LSGD-FP
DFL

Figure 2: Binary classification problem. From left to right: (a) Impact of efficient sampling strategies
on convergence. (b) Performance gains from partial adoption of efficient sampling. (c) Comparative
advantage of SRRW over NBRW in a small subset of agents. (d) Asymptotic network independence
of four algorithms under UD-SGD framework with fixed sampling strategy (shuffling, SRRW). (e)
Different sampling strategies in the DSGD algorithm with time-varying topology (DSGD-VT). (f)
Different sampling strategies in the DFL algorithm with increasing communication interval.

in total) and each agent holds B = 500 distinct data points. Each agent in the first group has full
access to its entire dataset, and thus can employ i.i.d. sampling (baseline) or single shuffling. On
the other hand, each agent in the other group has a graph-like structure and uses SRW (baseline),
NBRW or SRRW with reweighting to sample its local dataset with uniform weight. In this simulation,
we assume that agents can only communicate through a communication network using the DSGD
algorithm. This scenario with heterogeneous agents, as depicted in Figure 1, is of great interest in
large-scale machine learning [37, 32]. In addition, we employ a decreasing step size γn = 1/n in our
UD-SGD framework (1) because it is typically used for the strongly convex objective function and is
tested to have the fastest convergence in this simulation setup. Due to space constraints, we defer
detailed simulation setup, including the introduction of SRW, NBRW, and SRRW, to Appendix G.1.

The simulation results are obtained through 120 independent trials. In Figure 2(a), we assume that
the first group of agents perform either i.i.d. sampling or shuffling method, while the other group of
agents all change their sampling strategies from baseline SRW to NBRW and SRRW, as shown in the
legend. This plot shows that improved sampling strategy leads to overall convergence speedup since
NBRW and SRRW are more efficient than SRW [38, 25]. Furthermore, it illustrates that SRRW is
significantly more efficient than NBRW in this simulation setup, i.e., SRRW ≫NBRW > SRW
in terms of sampling efficiency. While keeping the second group of agents unchanged, we can see
that shuffling method outperforms i.i.d. sampling with smaller asymptotic MSE. However, shuffling
method may not perform perfectly for small time n due to slow mixing behavior in the initial period,
which is also observed in the single-agent scenario in [65, 1, 38]. The error bar therein also indicates
that the random-walk sampling strategy has a significant impact on the overall system performance
and SRRW has smaller variance than NBRW and SRW.

In Figure 2(b), we let the first group of agents perform i.i.d. sampling while only changing a portion
of agents in the second group to upgrade from SRW to SRRW, e.g., 30 SRW 20 SRRW in the legend
means that there are 30 agents using SRW while the rest 20 agents in the second group upgrade
to SRRW. We observe that more agents willing to upgrade from SRW to SRRW lead to smaller
asymptotic MSE, as predicted by Theorem 3.3 and Remark 3. This improvement in MSE reduction
doesn’t scale linearly with more agents adopting SRRW because each agent holds its own dataset that
are not necessarily identical, resulting in different individual limiting covariance matrices Vi ̸=Vj.

While maintaining i.i.d. sampling for the first group of agents, we compare the performance when the
second group of agents in Figure 2(c) employ NBRW or SRRW. Remarkably, the case with only 10
agents out of 50 agents in the second group adopting far more efficient sampling strategy (40 SRW,
10 SRRW) through incentives or compliance already produces a smaller MSE than all 50 agents
using slightly better strategy (50 NBRW). The performance gap becomes even more pronounced

9


---Page Break---
when 20 agents upgrade from SRW to SRRW (30 SRW, 20 SRRW). We show that the performance
of a distributed system can be improved significantly when a small proportion of agents adopt highly
efficient sampling strategies.

Figure 2(d) empirically illustrates the asymptotic network independence property via four algorithms
under our UD-SGD framework: Centralized SGD (communication interval K = 1, communication
matrix W = 11T /N); LSGD-FP (FL with full client participation, K = 5, W = 11T /N); DSGD-
VT (DSGD with time-varying topologies, randomly chosen from 5 doubly stochastic matrices);
DFL (decentralized FL with fixed MH-generated W and increasing communication interval Kl =
max{1, log(l)} after l-th aggregation). We fix the sampling strategy (shuffling, SRRW) throughout
this plot. All four algorithms overlap around 1000 steps, implying that they have entered the
asymptotic regime with similar performance where the CLT result dominates, implying the asymptotic
network independence in the long run.

Figure 2(e) and 2(f) show the performance of different sampling strategies in DSGD-VT and DFL
algorithms in terms of MSE. Both plots consistently demonstrate that improving agent’s sampling
strategies (e.g., shuffling > iid sampling, and SRRW > NBRW > SRW) leads to faster convergence
with smaller MSE, supporting our theory.

Furthermore, in Appendix G.2, we simulate an image classification task with CIFAR-10 dataset [44]
by training a 5-layer CNN and ResNet-18 model collaboratively through a 10-agent network. The
result is illustrated in Figure 3, where SRRW outperforms NBRW and SRW as expected. In summary,
we find that upgrading even a small portion of agents to efficient sampling strategies (e.g., shuffling
method, NBRW, SRRW under different dataset structures) improves system performance in UD-SGD.
These results are consistent in binary and image classification tasks, underscoring that every agent
matters in distributed learning.

5
Conclusion

In this work, we develop an UD-SGD framework that establishes the CLT of various distributed algo-
rithms with Markovian sampling. We overcome technical challenges such as quantifying consensus
error under very general communication patterns and decomposing Markovian noise through the
Poisson equation, which extends the analysis beyond the single-agent scenario. We demonstrate that
even if only a few agents optimize their sampling strategies, the entire distributed system will benefit
with a smaller limiting covariance in the CLT, suggesting a reduced MSE. This finding challenges the
current established upper bounds where the worst-performing agent leads the pack. Future studies
could pivot towards developing fine-grained finite-time bounds to individually characterize each
agent’s behavior, and theoretically analyze the effect of SRRW in UD-SGD.

6
Acknowledgments and Disclosure of Funding

We thank the anonymous reviewers for their constructive comments. This work was supported in part
by National Science Foundation under Grant Nos. CNS-2007423, IIS-1910749, and IIS-2421484.

References

[1] Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. Sgd with shuffling: optimal rates without
component convexity and large epoch requirements. In Proceedings of the 34th International
Conference on Neural Information Processing Systems, pages 17526–17535, 2020.

[2] Noga Alon, Itai Benjamini, Eyal Lubetzky, and Sasha Sodin. Non-backtracking random walks
mix faster. Communications in Contemporary Mathematics, 9(04):585–603, 2007.

[3] Christophe Andrieu, Éric Moulines, and Pierre Priouret. Stability of stochastic approximation
under verifiable conditions. SIAM Journal on control and optimization, 44(1):283–312, 2005.

[4] Ghadir Ayache, Venkat Dassari, and Salim El Rouayheb. Walk for learning: A random walk
approach for federated learning from heterogeneous data. IEEE Journal on Selected Areas in
Communications, 41(4):929–940, 2023.

10


---Page Break---
[5] Anna Ben-Hamou, Eyal Lubetzky, and Yuval Peres. Comparing mixing times on sparse
random graphs. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete
Algorithms, pages 1734–1740. SIAM, 2018.

[6] Albert Benveniste, Michel Métivier, and Pierre Priouret. Adaptive algorithms and stochastic
approximations, volume 22. Springer Science & Business Media, 2012.

[7] Pascal Bianchi, Gersende Fort, and Walid Hachem. Performance of a distributed stochastic
approximation algorithm. IEEE Transactions on Information Theory, 59(11):7405–7418, 2013.

[8] Patrick Billingsley. Convergence of probability measures. John Wiley & Sons, 2013.

[9] Vivek Borkar, Shuhang Chen, Adithya Devraj, Ioannis Kontoyiannis, and Sean Meyn. The ode
method for asymptotic statistics in stochastic approximation and reinforcement learning. arXiv
preprint arXiv:2110.14427, 2021.

[10] V.S. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint: Second Edition. Texts
and Readings in Mathematics. Hindustan Book Agency, 2022.

[11] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein, et al. Distributed opti-
mization and statistical learning via the alternating direction method of multipliers. Foundations
and Trends® in Machine learning, 3(1):1–122, 2011.

[12] Pierre Brémaud. Markov chains: Gibbs fields, Monte Carlo simulation, and queues, volume 31.
Springer Science & Business Media, 2013.

[13] Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng. Handbook of markov chain
monte carlo. CRC press, 2011.

[14] Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector machines. ACM
transactions on intelligent systems and technology (TIST), 2(3):1–27, 2011.

[15] VijaySekhar Chellaboina and Wassim M Haddad. Nonlinear dynamical systems and control: A
Lyapunov-based approach. Princeton University Press, 2008.

[16] Vishnu Pandi Chellapandi, Antesh Upadhyay, Abolfazl Hashemi, and Stanislaw H Zak. On the
convergence of decentralized federated learning under imperfect information sharing. arXiv
preprint arXiv:2303.10695, 2023.

[17] Shuhang Chen, Adithya Devraj, Ana Busic, and Sean Meyn. Explicit mean-square error bounds
for monte-carlo and linear stochastic approximation. In International Conference on Artificial
Intelligence and Statistics, pages 4173–4183. PMLR, 2020.

[18] Wenlin Chen, Samuel Horváth, and Peter Richtárik. Optimal client sampling for federated
learning. Transactions on Machine Learning Research, 2022.

[19] Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun.
The loss surfaces of multilayer networks. In Artificial intelligence and statistics, pages 192–204,
2015.

[20] Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and
Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-
convex optimization. In Advances in neural information processing systems, volume 27, 2014.

[21] Burgess Davis. On the intergrability of the martingale square function. Israel Journal of
Mathematics, 8:187–190, 1970.

[22] Bernard Delyon. Stochastic approximation with decreasing gain: Convergence and asymptotic
theory. Technical report, 2000.

[23] Bernard Delyon, Marc Lavielle, and Eric Moulines. Convergence of a stochastic approximation
version of the em algorithm. Annals of statistics, pages 94–128, 1999.

11


---Page Break---
[24] Thinh Doan, Siva Maguluri, and Justin Romberg. Finite-time analysis of distributed td (0)
with linear function approximation on multi-agent reinforcement learning. In International
Conference on Machine Learning, pages 1626–1635. PMLR, 2019.

[25] Vishwaraj Doshi, Jie Hu, and Do Young Eun. Self-repellent random walks on general graphs–
achieving minimal sampling variance via nonlinear markov chains. In International Conference
on Machine Learning. PMLR, 2023.

[26] John C Duchi, Alekh Agarwal, Mikael Johansson, and Michael I Jordan. Ergodic mirror descent.
SIAM Journal on Optimization, 22(4):1549–1578, 2012.

[27] Martin Dyer, Alan Frieze, Ravi Kannan, Ajai Kapoor, Ljubomir Perkovic, and Umesh Vazirani.
A mildly exponential time algorithm for approximating the number of solutions to a multi-
dimensional knapsack problem. Combinatorics, Probability and Computing, 2(3):271–284,
1993.

[28] Mathieu Even. Stochastic gradient descent under markovian sampling schemes. In International
Conference on Machine Learning, 2023.

[29] Gersende Fort. Central limit theorems for stochastic approximation with controlled markov
chain dynamics. ESAIM: Probability and Statistics, 19:60–80, 2015.

[30] Yann Fraboni, Richard Vidal, Laetitia Kameni, and Marco Lorenzi. A general theory for client
sampling in federated learning. In International Workshop on Trustworthy Federated Learning
in conjunction with IJCAI, 2022.

[31] Avishek Ghosh, Jichan Chung, Dong Yin, and Kannan Ramchandran. An efficient framework
for clustered federated learning. In Proceedings of the 34th International Conference on Neural
Information Processing Systems, pages 19586–19597, 2020.

[32] Yuanxiong Guo, Ying Sun, Rui Hu, and Yanmin Gong. Hybrid local SGD for federated learning
with heterogeneous communications. In International Conference on Learning Representations,
2022.

[33] P. Hall, C.C. Heyde, Z.W. Birnbauam, and E. Lukacs. Martingale Limit Theory and Its
Application. Communication and Behavior. Elsevier Science, 2014.

[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 770–778, 2016.

[35] Hadrien Hendrikx. A principled framework for the design and analysis of token algorithms. In
International Conference on Artificial Intelligence and Statistics, pages 470–489. PMLR, 2023.

[36] Roger A. Horn and Charles R. Johnson. Topics in Matrix Analysis. Cambridge University Press,
1991.

[37] Seyyedali Hosseinalipour, Sheikh Shams Azam, Christopher G Brinton, Nicolo Michelusi,
Vaneet Aggarwal, David J Love, and Huaiyu Dai. Multi-stage hybrid federated learning over
large-scale d2d-enabled fog networks. IEEE/ACM Transactions on Networking, 2022.

[38] Jie Hu, Vishwaraj Doshi, and Do Young Eun. Efficiency ordering of stochastic gradient descent.
In Advances in Neural Information Processing Systems, 2022.

[39] Jie Hu, Vishwaraj Doshi, and Do Young Eun. Accelerating distributed stochastic optimiza-
tion via self-repellent random walks. In The Twelfth International Conference on Learning
Representations, 2024.

[40] Mark Jerrum and Alistair Sinclair. The markov chain monte carlo method: an approach to
approximate counting and integration. Approximation Algorithms for NP-hard problems, PWS
Publishing, 1996.

[41] Thomas Kailath, Ali H Sayed, and Babak Hassibi. Linear estimation. Prentice Hall, 2000.

12


---Page Break---
[42] Sajad Khodadadian, Pranay Sharma, Gauri Joshi, and Siva Theja Maguluri. Federated rein-
forcement learning: Linear speedup under markovian sampling. In International Conference on
Machine Learning, pages 10997–11057, 2022.

[43] Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A
unified theory of decentralized sgd with changing topology and local updates. In International
Conference on Machine Learning, pages 5381–5393, 2020.

[44] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images.
Technical report, 2009.

[45] Harold Kushner and G George Yin. Stochastic approximation and recursive algorithms and
applications, volume 35. Springer Science & Business Media, 2003.

[46] Anusha Lalitha, Shubhanshu Shekhar, Tara Javidi, and Farinaz Koushanfar. Fully decentralized
federated learning. In Advances in neural information processing systems, 2018.

[47] Batiste Le Bars, Aurélien Bellet, Marc Tommasi, Erick Lavoie, and Anne-Marie Kermarrec.
Refined convergence and topology learning for decentralized sgd with heterogeneous data. In
International Conference on Artificial Intelligence and Statistics, pages 1672–1702. PMLR,
2023.

[48] Chul-Ho Lee, Xin Xu, and Do Young Eun. Beyond random walk and metropolis-hastings
samplers: why you should not backtrack for unbiased graph sampling. ACM SIGMETRICS
Performance evaluation review, 40(1):319–330, 2012.

[49] Jure Leskovec and Andrej Krevl. Snap datasets: Stanford large network dataset collection,
2014.

[50] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence
of fedavg on non-iid data. In International Conference on Learning Representations, 2020.

[51] Xiang Li, Jiadong Liang, Xiangyu Chang, and Zhihua Zhang. Statistical estimation and online
inference via local sgd. In Proceedings of Thirty Fifth Conference on Learning Theory, volume
178 of Proceedings of Machine Learning Research, pages 1613–1661, 02–05 Jul 2022.

[52] Xiang Li, Jiadong Liang, and Zhihua Zhang. Online statistical inference for nonlinear stochastic
approximation with markovian data. arXiv preprint arXiv:2302.07690, 2023.

[53] Yunming Liao, Yang Xu, Hongli Xu, Lun Wang, and Chen Qian. Adaptive configuration for
heterogeneous participants in decentralized federated learning. In IEEE INFOCOM 2023. IEEE,
2023.

[54] Adwaitvedant S Mathkar and Vivek S Borkar. Nonlinear gossip. SIAM Journal on Control and
Optimization, 54(3):1535–1557, 2016.

[55] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-efficient learning of deep networks from decentralized data. In Artificial
intelligence and statistics, pages 1273–1282. PMLR, 2017.

[56] Sean Meyn. Control systems and reinforcement learning. Cambridge University Press, 2022.

[57] Antonietta Mira. Ordering and improving the performance of monte carlo markov chains.
Statistical Science, pages 340–350, 2001.

[58] Gemma Morral, Pascal Bianchi, and Gersende Fort. Success and failure of adaptation-diffusion
algorithms with decaying step size in multiagent networks. IEEE Transactions on Signal
Processing, 65(11):2798–2813, 2017.

[59] Wenlong Mou, Chris Junchi Li, Martin J Wainwright, Peter L Bartlett, and Michael I Jordan. On
linear stochastic approximation: Fine-grained polyak-ruppert and non-asymptotic concentration.
In Conference on Learning Theory, pages 2947–2997. PMLR, 2020.

13


---Page Break---
[60] Giovanni Neglia, Chuan Xu, Don Towsley, and Gianmarco Calbi. Decentralized gradient
methods: does topology matter? In International Conference on Artificial Intelligence and
Statistics, pages 2348–2358. PMLR, 2020.

[61] Alex Olshevsky. Asymptotic network independence and step-size for a distributed subgradient
method. Journal of Machine Learning Research, 23(69):1–32, 2022.

[62] Shi Pu, Alex Olshevsky, and Ioannis Ch Paschalidis. Asymptotic network independence in
distributed stochastic optimization for machine learning: Examining distributed and centralized
stochastic gradient descent. IEEE signal processing magazine, 37(3):114–122, 2020.

[63] Christian P Robert, George Casella, and George Casella. Monte Carlo statistical methods,
volume 2. Springer, 1999.

[64] Sheldon M Ross, John J Kelly, Roger J Sullivan, William James Perry, Donald Mercer, Ruth M
Davis, Thomas Dell Washburn, Earl V Sager, Joseph B Boyce, and Vincent L Bristow. Stochastic
processes, volume 2. Wiley New York, 1996.

[65] Itay Safran and Ohad Shamir. How good is sgd with random shuffling? In Conference on
Learning Theory, pages 3250–3284. PMLR, 2020.

[66] R. Srikant. Rates of Convergence in the Central Limit Theorem for Markov Chains, with an
Application to TD Learning. arXiv e-prints, page arXiv:2401.15719, January 2024.

[67] Sebastian U Stich. Local sgd converges fast and communicates little. In International Conference
on Learning Representations, 2018.

[68] Tao Sun, Dongsheng li, and Bao Wang. On the decentralized stochastic gradient descent with
markov chain sampling. IEEE Transactions on Signal Processing, PP, 07 2023.

[69] Tao Sun, Yuejiao Sun, and Wotao Yin. On markov chain gradient descent. In Advances in
neural information processing systems, volume 31, 2018.

[70] M Vidyasagar. Convergence of stochastic approximation via martingale and converse lyapunov
methods. arXiv preprint arXiv:2205.01303, 2022.

[71] Hoi-To Wai. On the convergence of consensus algorithms with markovian noise and gradient
bias. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 4897–4902. IEEE,
2020.

[72] Han Wang, Aritra Mitra, Hamed Hassani, George J Pappas, and James Anderson. Federated tem-
poral difference learning with linear function approximation under environmental heterogeneity.
arXiv preprint arXiv:2302.02212, 2023.

[73] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor. Tackling the
objective inconsistency problem in heterogeneous federated optimization. In Advances in
Neural Information Processing Systems, volume 33, pages 7611–7623, 2020.

[74] Shiqiang Wang and Mingyue Ji. A unified analysis of federated learning with arbitrary client
participation. In Advances in neural information processing systems, 2022.

[75] Stephan Wojtowytsch. Stochastic gradient descent with noise of machine learning type part i:
Discrete time analysis. Journal of Nonlinear Science, 33(3):45, 2023.

[76] Blake Woodworth, Kumar Kshitij Patel, Sebastian Stich, Zhen Dai, Brian Bullins, Brendan
Mcmahan, Ohad Shamir, and Nathan Srebro. Is local sgd better than minibatch sgd?
In
International Conference on Machine Learning, pages 10334–10343. PMLR, 2020.

[77] Hao Ye, Le Liang, and Geoffrey Ye Li. Decentralized federated learning with unreliable
communications. IEEE Journal of Selected Topics in Signal Processing, 16(3):487–500, 2022.

[78] Chulhee Yun, Shashank Rajput, and Suvrit Sra. Minibatch vs local SGD with shuffling: Tight
convergence bounds and beyond. In International Conference on Learning Representations,
2022.

14


---Page Break---
[79] Chulhee Yun, Suvrit Sra, and Ali Jadbabaie. Open problem: Can single-shuffle sgd be better
than reshuffling sgd and gd? In Proceedings of Thirty Fourth Conference on Learning Theory,
volume 134, pages 4653–4658. PMLR, Aug 2021.

[80] Sihan Zeng, Thinh T Doan, and Justin Romberg. Finite-time convergence rates of decentralized
stochastic approximation with applications in multi-agent and multi-task learning.
IEEE
Transactions on Automatic Control, 2022.

[81] Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated
learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018.

15


---Page Break---
A
Discussion of Assumption 2.3-ii)

A.1
Suitable choices of Kl

When wet let Kl ∼log(l) (resp. Kl ∼log log(l)), as suggested by [51], it trivially satisfies
Kτn = o(γ−1/2(L+1)
n
) = o(n1/2(L+1)) since by definition Kτn < Kn ∼log(n) (resp. log log(n)),
and log(n) = o(nϵ) (resp. log log(n) = o(nϵ)) for any ϵ > 0. Besides, P

n η2
n = P

n γ2
nK2(L+1)
τn
≲
P

n n−2n2(L+1)ϵ = P

n n2(L+1)ϵ−2. To ensure P
n η2
n < ∞, it is sufficient to have 2(L+1)ϵ−2 <
−1, or equivalently, ϵ < 1/2(L + 1). Since ϵ can be arbitrarily small to satisfy the condition,
P

n η2
n < ∞is satisfied. When Kl ∼log(l), we can rewrite the last condition as

ηnl+1
ηnl+1+1
=
γnl+1
γnl+1+1

KL+1
l
KL+1
l+1
=
nl+1 + 1

nl + 1

 log(l + 1) + 1

log(l) + 1

L+1

=

1 + Kl+1

nl + 1

 log(l + 1) + 1

log(l) + 1

L+1
,

(16)

where we have nl ∼log(l!) such that Kl+1/nl = log(l+1)/ log(l!) →0 and log(l+1)/ log(l) →1
as l →∞, which leads to limn→∞ηnl+1/ηnl+1+1 = 1. Similarly, for Kl ∼log log(l), we
have nl ∼log(Ql
s=1 log(s)) such that Kl+1/nl ∼log log(l + 1)/ log log(Ql
s=1 log(s)) →0 and
log log(l + 1)/ log log(l) →1 as l →∞, which also leads to limn→∞ηnl+1/ηnl+1+1 = 1.

A.2
Practical implications of Assumption 2.3-ii)

In this assumption, we allow the number of local iterations to go to infinity asymptotically. In
distributed learning environments such as mobile, IoT, and wireless sensor networks, where nodes
are often constrained by battery life, increasing communication interval in Assumption 2.3-ii) plays
a crucial role in balancing energy costs with communication effectiveness. It allows agents to
communicate more frequently early on, leading to a faster initial convergence to the neighborhood of
θ∗. Then, we slow down the communication frequency between agents to conserve energy, leveraging
the diminishing returns on accuracy improvements from additional communications.

Consider the scenario where devices across multiple clusters collaborate on a distributed optimization
task, utilizing local datasets. Devices within each cluster form a communication network that allows
a virtual agent to perform a heterogeneous Markov chain trajectory via random walk, or an i.i.d.
sequence in a complete graph with self-loops, depending on the application context. Each cluster
features an edge server that supports the exchange of model estimates with neighboring clusters. By
performing K local updates before uploading these to the cluster’s edge server, the model benefits
from reduced communication overhead. As the frequency of updates between devices and edge
servers decreases — optimized by gradually increasing K — we effectively lower communication
costs, particularly as the model estimation θn is close to θ∗.

B
Proof of Lemma 3.1

Let J⊥≜IN −J ∈RN×N and J⊥≜J⊥⊗Id ∈RNd×Nd, where ⊗is the Kronecker product.
Let Θn = [(θ1
n)T , · · · , (θN
n )T ]T ∈RNd. Then, motivated by [58], we define a sequence ϕn ≜
η−1
n+1J⊥Θn ∈RNd in the increasing communication interval case (resp. ϕn ≜γ−1
n+1J⊥Θn in the
bounded communication interval case), where ηn+1 is defined in Assumption 2.3-ii). J⊥Θn =
Θn −1

N (11T ⊗Id)Θn represents the consensus error of the model.

We first give the following lemma that shows the pathwise boundedness of ϕn.
Lemma B.1. Let Assumptions 2.1, 2.3, 2.4 and 2.5 hold. For any compact set Ω⊂RNd, the
sequence ϕn satisfies supn E[∥ϕn∥21∩j≤n−1{Θj∈Ω}] < ∞.

Lemma B.1 and Assumption 2.3-ii) imply that for any n ≥0, E[∥J⊥Θn∥21∩j≤n−1{Θj∈Ω}] =
η2
n+1E[∥ϕn∥21∩j≤n−1{Θj∈Ω}] ≤Cη2
n+1 for some constant C that depends on C1 and Ω. Along
with Assumption 2.4 such that ∥J⊥Θn∥is always bounded per each trajectory, it means

∥J⊥Θn∥1∩j≤n−1{Θj∈Ω} = O(ηn)
a.s.

16


---Page Break---
Let {Ωm}m≥0 be a sequence of increasing compact subset of RNd such that S
m Ωm = RNd. Then,
we know that for any m ≥0,

∥J⊥Θn∥1∩j≤n−1{Θj∈Ωm} = O(ηn)
a.s.
(17)

(17) indicates either one of the following two cases:

• there exists some trajectory-dependent index m′ such that each trajectory {Θn}n≥0 is
always within the compact set Ωm′, i.e., 1∩j≤n{Θj∈Ωm′} = 1 (satisfied by the construction
of increasing compact sets {Ωm}m≥0 and Assumption 2.4), and we have ∥J⊥Θn∥= O(ηn)
such that limn→∞J⊥Θn = 0;

• Θn will escape the compact set Ωm eventually for any m ≥0 in finite time such that
1∩j≤n−1{Θj∈Ωm} = 0 when n is large enough.

We can see the second case contradicts Assumption 2.4 because we assume every trajectory {Θn}n≥0
is within some compact set. Therefore, (17) for any m ≥0 is equivalent to showing ∥J⊥Θn∥=
O(ηn) and limn→∞J⊥Θn = 0. Under Assumption 2.3-i) we can obtain similar result ∥J⊥Θn∥=
O(γn) by following the same steps as above, which completes the proof of Lemma 3.1.

Proof of Lemma B.1. We begin by rewriting (6) in the matrix form,

Θn+1 = Wn (Θn −γn+1∇F(Θn, Xn)) ,
(18)

where Xn ≜(X1
n, X2
n, · · · , XN
n ) and ∇F(Θn, Xn) ≜[∇F1(θ1
n, X1
n)T , · · · , ∇FN(θN
n , XN
n )T ]T ∈
RNd. Recall θn ≜1

N
PN
i=1 θi
n ∈Rd and we have [θT
n , · · · , θT
n ]T = 1

N (11T ⊗Id)Θn ∈RNd.

Case 1 (Increasing communication interval Kτn): By left multiplying (18) with 1

N (11T ⊗Id),
along with γn+1 = ηn+1/KL+1
τn+1 in Assumption 2.3-ii), we have the following iteration

1
N (11T ⊗Id)Θn+1 = 1

N (11T ⊗Id)Θn −ηn+1
1
N (11T ⊗Id)∇F(Θn, Xn)

KL+1
τn+1
,
(19)

where the equality comes from 1

N (11T ⊗Id)Wn = 1

N (11T Wn ⊗Id) = 1

N (11T ⊗Id). With (18)
and (19), we have

Θn+1 −1

N (11T ⊗Id)Θn+1

=

Wn −1

N (11T ⊗Id)

Θn −ηn+1


Wn −1

N (11T ⊗Id)
 ∇F(Θn, Xn)

KL+1
τn+1

=(J⊥Wn ⊗Id)J⊥Θn −ηn+1(J⊥Wn ⊗Id)∇F(Θn, Xn)

KL+1
τn+1

=ηn+1(J⊥Wn ⊗Id)

 

η−1
n+1J⊥Θn −∇F(Θn, Xn)

KL+1
τn+1

!

,

(20)

where the second equality comes from Wn −1

N (11T ⊗Id) = (Wn −1

N 11T ) ⊗Id = J⊥Wn ⊗Id
and (J⊥Wn ⊗Id)J⊥= J⊥WnJ⊥⊗Id = J⊥Wn ⊗Id. Let an ≜ηn/ηn+1, dividing both sides
of (20) by ηn+2 gives

ϕn+1 = an+1(J⊥Wn ⊗Id)

 

ϕn −∇F(Θn, Xn)

KL+1
τn+1

!

.
(21)

17


---Page Break---
Define the filtration {Fn}n≥0 as Fn ≜σ{Θ0, X0, W0, Θ1, X1, W1, · · · , Xn−1, Wn−1, Θn, Xn}.
Recursively computing (21) w.r.t the time interval [nl, nl+1] gives

ϕnl+1 =

"
nl+1
Y

k=nl+1
ak

#  "

J⊥

nl+1−1
Y

k=nl
Wk

#

⊗Id

!

ϕnl

−

nl+1−1
X

k=nl

" nl+1
Y

i=k+1
ai

#  "

J⊥

nl+1−1
Y

i=k
Wi

#

⊗Id

!
∇F(Θk, Xk)

KL+1
l+1

=
ηnl+1
ηnl+1+1
(J⊥Wnl ⊗Id) ϕnl −

nl+1−1
X

k=nl

ηnl+1

ηk+2
(J⊥Wnl ⊗Id) ∇F(Θk, Xk)

KL+1
l+1
,

(22)

where Q is the backward multiplier, the second equality comes from J⊥WnJ⊥= J⊥Wn
and Wk
= IN for k
/∈{nl}.
In Assumption 2.5, we have ∥EW∼Pnl [WT J⊥W]∥=
∥EW∼Pnl [WT W −J]∥≤C1 < 1. Then,

E[∥ϕnl+1∥2|Fnl]

=
 ηnl+1

ηnl+1+1

2
ϕT
nlEWnl∼Pnl

h
(J⊥Wnl ⊗Id)T (J⊥Wnl ⊗Id)
i
ϕnl

−2E

" nl+1−1
X

k=nl

η2
nl+1
ηnl+1+1ηk+2
ϕT
nl (J⊥Wnl ⊗Id)T (J⊥Wnl ⊗Id) ∇F(Θk, Xk)

KL+1
l+1

 Fnl

#

+ E







nl+1−1
X

k=nl

ηnl+1

ηk+2
(J⊥Wnl ⊗Id) ∇F(Θk, Xk)

KL+1
l+1



2
Fnl





≤
 ηnl+1

ηnl+1+1

2
ϕT
nlEWnl∼Pnl
 
WT
nlJ⊥Wnl ⊗Id

ϕnl

−2
 ηnl+1

ηnl+1+1

2
E

" nl+1−1
X

k=nl
ϕT
nl
 
WT
nlJ⊥Wnl ⊗Id
 ∇F(Θk, Xk)

KL+1
l+1

 Fnl

#

+
 ηnl+1

ηnl+1+1

2
E





(J⊥Wnl ⊗Id)

nl+1−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1



2
Fnl





≤
 ηnl+1

ηnl+1+1

2
C1∥ϕnl∥2 + 2
 ηnl+1

ηnl+1+1

2
C1∥ϕnl∥E

"

nl+1−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1



 Fnl

#

+
 ηnl+1

ηnl+1+1

2
C1E







nl+1−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1



2
Fnl



,

(23)

where the first inequality comes from JT
⊥J⊥= J⊥and ηk+2 ≥ηnl+1+1 for k ∈[nl, nl+1 −1].
Then, we analyze the norm of the gradient ∥∇F(Θk, Xk)∥in the second term on the RHS of (23)
conditioned on Fnl. By Assumption 2.4, we assume Θnl is within some compact set Ωat time
nl such that supi∈[N],Xi∈Xi ∇Fi(θi
nl, Xi) ≤CΩfor some constant CΩ. For n = nl + 1 and any
X ∈X1 × X2 × · · · × XN, we have

∥∇F(Θnl+1, X)∥≤∥∇F(Θnl+1, X) −∇F(Θnl, X)∥+ ∥∇F(Θnl, X)∥.

Considering
∥∇F(Θnl, X)∥,
we
have
supX ∥∇F(Θnl, X)∥2
≤
PN
i=1 supXi∈Xi ∥∇Fi(θi
nl, Xi)∥2 ≤NC2
Ωsuch that ∥∇F(Θnl, X)∥≤
√

NCΩ. In addition, we

18


---Page Break---
have

∥∇F(Θnl+1, X) −∇F(Θnl, X)∥2 =

N
X

i=1
∥∇Fi(θi
nl+1, Xi) −∇Fi(θi
nl, Xi)∥2

≤

N
X

i=1
L2∥θi
nl+1 −θi
nl∥2

≤

N
X

i=1
γ2
nl+1L2∥∇Fi(θi
nl, Xi
nl)∥2

≤γ2
nl+1C2
ΩNL2

(24)

such that ∥∇F(Θnl+1, X) −∇F(Θnl, X)∥≤γnl+1CΩ
√

NL. Thus, for any X,

∥∇F(Θnl+1, X)∥≤(1 + γnl+1L)
√

NCΩ.
(25)

For n = nl + 2 and any X, we have

∥∇F(Θnl+2, X)∥≤∥∇F(Θnl+2, X) −∇F(Θnl+1, X)∥+ ∥∇F(Θnl+1, X)∥.

Similar to the steps in (24), we have

∥∇F(Θnl+2, X) −∇F(Θnl+1, X)∥2 ≤

N
X

i=1
γ2
nl+2L2∥∇Fi(θi
nl+1, Xi
nl+1)∥2

=γ2
nl+2L2∥∇F(Θnl+1, Xnl+1)∥2.

(26)

Then, ∥∇F(Θnl+2, X)∥≤(1 + γnl+2L) supX ∥∇F(Θnl+1, X)∥and, together with (25), we have

∥∇F(Θnl+2, X)∥≤(1 + γnl+2L)(1 + γnl+1L)
√

NCΩ.
(27)

By induction, ∥∇F(Θnl+m, X)∥≤Qm
s=1(1 + γnl+sL)
√

NCΩfor m ∈[1, Kl+1 −1].

The next step is to analyze the growth rate of Qm
s=1(1 + γnl+sL). By 1 + x ≤ex for x ≥0, we have

m
Y

s=1
(1 + γnl+sL) ≤eL Pm
s=1 γnl+s.

For step size γn = 1/n, we have L Pm
s=1 γnl+s = L Pm
s=1 1/(nl+s) < L Pm
s=1 1/s < L(log(m)+
1) such that Qm
s=1(1 + γnl+sL) < (em)L. Then,


nl+1−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1

 ≤
1
KL+1
l+1

nl+1−1
X

k=nl
∥∇F(Θk, Xk)∥≤
1
KL+1
l+1

√

NeLCΩ

Kl+1−1
X

m=0
mL

≤
√

NeLCΩ,

(28)

where the last inequality comes from PKl+1−1
m=0
mL < Kl+1(Kl+1 −1)L < KL+1
l+1 . We can see the
sum of the norm of the gradients are bounded by
√

NeLCΩ, which only depends on the compact set
Ωat time n = nl.

Let δ1 ∈(C1, 1). Since from Assumption 2.3-ii), liml→∞ηnl+1/ηnl+1+1 = 1, there exists some
large enough l0 such that (
ηnl+1
ηnl+1+1 )2C1 < δ1 < δ2 := (δ1 + 1)/2 < 1 for any l > l0. Note that δ1

depends only on C1 and is independent of Fn. Then, let ˜CΩ:=
√

NeLCΩ, we can rewrite (23) as

E[∥ϕnl+1∥2|Fnl] ≤δ1∥ϕnl∥2 + 2δ1 ˜CΩ∥ϕnl∥+ δ1 ˜C2
Ω
≤δ2∥ϕnl∥2 + MΩ,
(29)

where MΩsatisfies MΩ> 8 ˜C2
Ω/(1 −δ1) + δ1 ˜C2
Ω, which is derived from rearranging (29) as
MΩ≥(δ1 −δ2)∥ϕnl∥2 + 2δ1 ˜CΩ∥ϕnl∥+ δ1 ˜C2
Ωand upper bounding the RHS. Upon noting that
1∩j≤nl{Θj∈Ω} ≤1∩j≤nl−1{Θj∈Ω}, we obtain

E
h
∥ϕnl+1∥21∩j≤nl{Θj∈Ω}
i
≤δ2E
h
∥ϕnl∥21∩j≤nl−1{Θj∈Ω}
i
+ MΩ.
(30)

19


---Page Break---
The induction leads to E[∥ϕnl+1∥21∩j≤nl{Θj∈Ω}] ≤δ
nl+1−nl0
2
E[∥ϕnl0∥21∩j≤nl0−1{Θj∈Ω}] +
M/(1 −δ2) < ∞for any l ≥l0. Besides, for m ∈(nl, nl+1), by following the above steps
(23) applied to (21), we have

E[∥ϕm∥2|Fnl] ≤
ηnl+1

ηm+1

2
∥ϕnl∥2 + 2
ηnl+1

ηm+1

2
∥ϕnl∥E

"

m−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1



 Fnl

#

+
ηnl+1

ηm+1

2
E







m−1
X

k=nl

∇F(Θk, Xk)

KL+1
l+1



2
Fnl



.

(31)

By (28) we already show that ∥Pnl+1−1
k=nl
∇F(Θk,Xk)

KL+1
l+1
∥< ∞conditioned on Fnl.
Therefore,

E[∥ϕm∥21∩j≤nl{Θj∈Ω}] < ∞for m ∈(nl, nl+1). This completes the boundedness analysis of
E[∥ϕn∥21∩j≤n−1{Θj∈Ω}].

Case 2 (Bounded communication interval Kτn ≤K): In this case, we do not need the auxiliary
step size ηn and can directly work on γn = 1/na for a ∈(0.5, 1]. Similar to (20), we have

Θn+1 −1

N (11T ⊗Id)Θn+1 = γn+1(J⊥Wn ⊗Id)
 
γ−1
n+1J⊥Θn −∇F(Θn, Xn)

,
(32)

and let bn ≜γn/γn+1, dividing both sides of above equation by γn+2 gives

ϕn+1 = bn+1(J⊥Wn ⊗Id) (ϕn −∇F(Θn, Xn)) .
(33)

Then, by following the similar steps in (22) and (23), we obtain

E[∥ϕnl+1∥2|Fnl] ≤
 γnl+1

γnl+1+1

2
C1

 

∥ϕnl∥2 + 2∥ϕnl∥E

"

nl+1−1
X

k=nl
∇F(Θk, Xk)



 Fnl

#

+ E







nl+1−1
X

k=nl
∇F(Θk, Xk)



2
Fnl





!

.

(34)

Also similar to (25) - (28), we can bound the sum of the norm of the gradients as


nl+1−1
X

k=nl
∇F(Θk, Xk)

 ≤

nl+1−1
X

k=nl

"
k
Y

s=nl
(1 + γs+1L)

#
√

NCΩ.
(35)

Now that Kl is bounded above by K, Qk
s=nl(1 + γs+1L) ≤eL Pk
s=nl γs+1 < eL PK−1
s=0 γs+1 := CK.
Then, we further bound (35) as


nl+1−1
X

k=nl
∇F(Θk, Xk)

 ≤
√

NKCKCΩ.
(36)

The subsequent proof is basically a replication of (29) - (31) and is therefore omitted.

C
Proof of Theorem 3.2

We focus on analyzing the convergence property of θ, which is obtained by left multiplying (18) with
1
N (1T ⊗Id), i.e.,

θn+1 = 1

N (1T ⊗Id)θn+1

= θn −γn+1
1
N (1T ⊗Id)∇F(Θn, Xn).
(37)

where the second equality comes from Wn being doubly stochastic and
1
N (1T ⊗Id)Wn =
1
N (1T Wn ⊗Id) = 1

N (1T ⊗Id).

For self-contained purpose, we first give the almost sure convergence result for the stochastic
approximation that will be used in our proof.

20


---Page Break---
Theorem C.1 (Theorem 2 [23]). Consider the stochastic approximation in the form of

θn+1 = θn + γn+1h(θn) + γn+1en+1 + γn+1rn+1.
(38)

Assume that

C1. w.p.1, the closure of {θn}n≥0 is a compact subset of Rd;

C2. {γn} is a decreasing sequence of positive number such that P

n γn = ∞;

C3. w.p.1, limp→∞
Pp
n=1 γnen exists and is finite. Moreover, limn→∞rn = 0.

C4. vector-valued function h is continuous on Rd and there exists a continuously differentiable
function V : Rd →R such that ⟨∇V (θ), h(θ)⟩≤0 for all θ ∈Rd. Besides, the interior of
V (L) is empty where L ≜{θ ∈Rd : ⟨∇V (θ), h(θ)⟩= 0}.

Then, w.p.1, lim supn d(θn, L) = 0.

We can rewrite (37) as

θn+1 =θn −γn+1
1
N (1T ⊗Id)∇F(Θn, Xn)

=θn −γn+1∇f(θn) −γn+1

 
1
N

N
X

i=1
∇fi(θi
n) −∇f(θn)

!

−γn+1

 
1
N

N
X

i=1
∇Fi(θi
n, Xi
n) −1

N

N
X

i=1
∇fi(θi
n)

!

,

(39)

and work on the converging behavior of the third and fourth term. By definition of function ∇f(·),
we have

rn ≜1

N

N
X

i=1
∇fi(θi
n) −∇f(θn) = 1

N

N
X

i=1


∇fi(θi
n) −∇fi(θn)

.
(40)

By the Lipschitz continuity of function ∇Fi(·, X) in (7), we have

∥rn∥≤1

N

N
X

i=1
L∥θi
n −θn∥≤
L
√

N

Θn −1

N (11T ⊗Id)Θn

 =
L
√

N
∥J⊥Θn∥,
(41)

where the second inequality comes from the Cauchy-Schwartz inequality. In Appendix B, we have
shown limn J⊥Θn = 0 almost surely such that limn→∞rn = 0 almost surely.

Next, we further decompose the fourth term in (39). For an ergodic transition matrix P and a function
v associated with the same state space X, define the operator Pkv(x) ≜P

y∈X Pk(x, y)v(y) for the
k-step transition probability Pk(x, y). Denote by P1, · · · , PN the underlying transition matrices
of all N agents with corresponding stationary distribution π1, · · · , πN. Then, for every function
∇Fi(θi, ·) : Xi →Rd, there exists a corresponding function mθi(·) : Xi →Rd such that

mθi(x) −Pimθi(x) = ∇Fi(θi, x) −∇fi(θi).
(42)

The solution of the Poisson equation (42) has been studied in the literature, e.g., [17, 38]. For
self-contained purpose, we derive the closed-form mθi(x) from scratch. First of all, we can obtain
function mθi(x) in the recursive form as follows,

mθi(x) = ∇Fi(θi, x)−∇fi(θi)+Pi[∇Fi(θi, ·)−∇fi(θi)](x)+P2
i [∇Fi(θi, ·)−∇fi(θi)](x)+· · · .
(43)
It is not hard to check that (43) satisfies (42). Note that by induction we get

Pk
i −1(πi)T =
 
Pi −1(πi)T k , ∀k ∈N, k ≥1.
(44)

21


---Page Break---
Then, we can further simplify (43), and the closed-form expression of mθi(x) is given as

mθi(x) =
X

y∈Xi


Pi −1(πi)T 0 (x, y)(∇Fi(θi, y) −∇fi(θi))

+
X

y∈X i


P1
i −1(πi)T 
(x, y)(∇Fi(θi, y) −∇fi(θi)) + · · ·

=
X

y∈Xi

" ∞
X

k=0


Pi −1(πi)T k
#

(x, y)(∇Fi(θi, y) −∇fi(θi))

=
X

y∈Xi

 
I −Pi + 1(πi)T −1 (x, y)(∇Fi(θi, y) −∇fi(θi)),

(45)

where the fourth equality comes from (44). Note that the so-called ‘fundamental matrix’ (I −Pi +
1(πi)T )−1 exists for every ergodic Markov chain Xi from Assumption 2.2. Since function ∇Fi is
Lipschitz continuous, we have the following lemma.
Lemma C.2. Under assumption (A1), functions mθi(x) and Pimθi(x) are both Lipschitz continuous
in θi for any x ∈Xi.

Proof. By (45), for any θi
1, θi
2 ∈Rd and x ∈Xi, we have

mθi
1(x) −mθi
2(x)
 ≤



X

y∈Xi

 
I −Pi + 1(πi)T −1 (x, y)

∇Fi(θi
1, y) −∇Fi(θi
2, y)



+
∇fi(θi
1) −∇fi(θi
2)


≤Ci max
y∈Xi

∇Fi(θi
1, y) −∇Fi(θi
2, y)
 +
∇fi(θi
1) −∇fi(θi
2)


≤(CiL + 1)∥θi
1 −θi
2∥,

(46)

where the second inequality holds for a constant Ci that is the largest absolute value of the entry
in the matrix (I −Pi + 1(πi)T )−1. Therefore, mθi(x) is Lipschitz continuous in θi. Moreover,
following the similar steps as above, we have
Pimθi
1(x) −Pimθi
2(x)
 =



X

y∈Xi
Pi(x, y)mθi
1(y) −
X

y∈Xi
Pi(x, y)mθi
2(y)



=



X

y∈Xi
Pi(x, y)

mθi
1(y) −mθi
2(y)



≤
X

y∈X i
Pi(x, y)
mθi
1(y) −mθi
2(y)


≤|Xi|
mθi
1(y) −mθi
2(y)


≤|Xi|(CiL + 1)∥θi
1 −θi
2∥

(47)

such that Pimθi(x) is also Lipschitz continuous in θi, which comletes the proof.

Now with (42) we can decompose ∇Fi(θi
n, Xi
n) −∇fi(θi
n) as
∇Fi(θi
n, Xi
n) −∇fi(θi
n) =mθin(Xi
n) −Pimθin(Xi
n)

= mθin(Xi
n) −Pimθin(Xi
n−1)
|
{z
}
ei
n+1
+ Pimθin(Xi
n−1)
|
{z
}
νin

−Pimθi
n+1(Xi
n)
|
{z
}
νi
n+1
+ Pimθi
n+1(Xi
n) −Pimθin(Xi
n)
|
{z
}
ξi
n+1

.

(48)

22


---Page Break---
Here {γnei
n} is a Martingale difference sequence and we need the martingale convergence theorem
in Theorem C.3.

Theorem C.3 (Theorem 6.4.6 [64]). For an Fn-Martingale Sn, set Xn−1 = Sn −Sn−1. If for some
1 ≤p ≤2,

∞
X

n=1
E[∥Xn−1∥p|Fn−1] < ∞
a.s.
(49)

then Sn converges almost surely.

We want to show that P

n γ2
n+1E[∥ei
n+1∥2|Fn] < ∞such that P

n γnei
n converges almost surely
by Theorem C.3. As we can see in (45), with Lemma C.2 and Assumption 2.4, for a sample
path (Θn within a compact set Ω), supn ∥mθin(x)∥< ∞and supn ∥Pimθin(x)∥< ∞almost
surely for all x ∈Xi. This ensures that ei
n+1 is an L2-bounded martingale difference sequence, i.e.,
supn ∥ei
n+1∥≤supn(∥mθi
n(Xi
n+1)∥+∥Pimθi
n(Xi
n)∥) ≤DΩ< ∞. Together with Assumption 2.3,
we get
X

n
γ2
n+1E[∥ei
n+1∥2|Fn] ≤DΩ
X

n
γ2
n+1 < ∞
a.s.
(50)

and thus P

n γnei
n converges almost surely.

Next, for the term νi
n we have

p
X

k=0
γk+1(νi
k −νi
k+1) =

p
X

k=0
(γk+1 −γk)νi
k + γ0νi
0 −γp+1νi
p+1.
(51)

As is shown before, for a given sample path, ∥Pimθin(x)∥is bounded almost surely for all n and
x ∈X i such that supn ∥νi
n∥< ∞almost surely. Since limn→∞(γn+1 −γn) = 0, we have
limn→∞(γn+1 −γn)νi
n = 0. Note that there exists a path-dependent constant C (that bounds ∥νi
n∥)
such that for any n ≥m,



n
X

k=m
(γk+1 −γk)νi
k

 ≤C

n
X

k=m
(γk −γk+1) = C(γm −γn+1) < Cγm.
(52)

Since limn→∞γn = 0, there exists a positive integer M such that for all n ≥m ≥M, γm < ϵ/C
and ∥Pn
k=m(γk+1 −γk)νi
k∥< ϵ for every ϵ > 0. Therefore, {Pp
k=0(γk+1 −γk)νi
k}p≥0 is a Cauchy
sequence and P∞
k=0(γk+1 −γk)νi
k converges by Cauchy convergence criterion. The last term of
(51) tends to zero. Therefore, P∞
k=0 γk+1(νi
k −νi
k+1) converges and is finite.

For the last term ξi
n, Lemma C.2 leads to

1
N

N
X

i=1

ξi
n+1
 ≤C′

N

N
X

i=1
∥θi
n+1 −θi
n∥≤C′

√

N
∥Θn+1 −Θn∥.
(53)

for the Lipschitz constant C′ of Pimθi(x). However, the relationship between θn and θn+1 is
not obvious in the D-SGD and FL setting due to the update rule (18) with communication matrix
Wn, unlike the classical stochastic approximation shown in (38). We come up with the novel
decomposition of ξi
n, which takes the consensus error into account, to solve this issue, i.e.,

ξi
n+1 =
h
Pimθi
n+1(Xi
n) −Pimθn+1(Xi
n)
i
+

Pimθn(Xi
n) −Pimθin(Xi
n)


+

Pimθn+1(Xi
n) −Pimθn(Xi
n)

.
(54)

23


---Page Break---
Using the Lipschitzness property of Pimθ(X) in Lemma C.2, we have

1
N

N
X

i=1

ξi
n+1
 ≤C′

N

N
X

i=1

 θi
n+1 −θn+1
 + ∥θn+1 −θn∥+
θn −θi
n


≤C′

√

N

Θn+1 −1

N
 
11T ⊗Id

Θn+1

 + C′

√

N

Θn −1

N
 
11T ⊗Id

Θn



+ C′ ∥θn+1 −θn∥

= C′

√

N
(∥J⊥Θn+1∥+ ∥J⊥Θn∥) + C′ ∥θn+1 −θn∥

= C′

√

N
(∥J⊥Θn+1∥+ ∥J⊥Θn∥) + C′γn+1


1
N (1T ⊗Id)∇F(Θn, Xn)
 .

(55)

In Appendix B we have shown limn→∞J⊥Θn = 0 almost surely.
Moreover, ∥1

N (1T ⊗
Id)∇F(Θn, Xn)∥is bounded per sample path. Therefore, limn→∞1

N
PN
i=1 ∥ξi
n+1∥= 0 such
that limn→∞1

N
PN
i=1 ξi
n+1 = 0 almost surely.

To sum up, we decompose (39) into

θn+1 = θn −γn+1∇f(θn) −γn+1rn −γn+1
1
N

N
X

i=1

 
ei
n+1 + νi
n −νi
n+1 + ξi
n+1

.
(56)

Now that limp→∞
Pp
n=1
1
N
PN
i=1 γnei
n and limp→∞
Pp
n=0
1
N
PN
i=1 γn+1(νi
n −νi
n+1) converge
and are finite, limn→∞rn = 0, limn→∞1

N
PN
i=1 ξi
n = 0 , all the conditions of C3 in Theorem
C.1 are satisfied. Additionally, Assumption 2.4 corresponds to C1, Assumption 2.3 meets C2, and
C4 is automatically satisfied when we choose the lyapunov function V (θ) = f(θ). Therefore,
lim supn infθ∗∈L ∥θn −θ∗∥= 0.

D
Proof of Theorem 3.3

To obtain Theorem 3.3, we need to utilize the existing CLT result for general SA in Theorem D.1 and
check all the necessary conditions therein.
Theorem D.1 (Theorem 2.1 [29]). Consider the stochastic approximation iteration (38), assume

C1. Let θ∗be the root of function h, i.e., h(θ∗) = 0, and assume limn→∞θn = θ∗. Moreover,
assume the mean field h is twice continuously differentiable in a neighborhood of θ∗, and
the Jacobian H ≜∇h(θ∗) is Hurwitz, i.e., the largest real part of its eigenvalues B < 0;

C2. The step size P

n γn = ∞, P

n γ2
n < ∞, and either (i). log(γn−1/γn) = o(γn), or (ii).
log(γn−1/γn) ∼γn/γ⋆for some γ⋆> 1/2|B|;

C3. supn ∥θi
n∥< ∞almost surely for any i ∈[N];

C4.
(a) {en}n≥0 is an Fn-Martingale difference sequence, i.e., E[en|Fn−1] = 0, and there
exists τ > 0 such that supn≥0 E[∥en∥2+τ|Fn−1] < ∞;

(b) E[en+1eT
n+1|Fn] = U + D(A)
n
+ D(B)
n
, where U is a symmetric positive semi-definite
matrix and
(
D(A)
n
→0 almost surely,

limn γnE
h
Pn
k=1 D(B)
k

i
= 0.
(57)

C5. Let rn = r(1)
n
+ r(2)
n , rn is Fn-adapted, and





r(1)
n
 = o(√γn)
a.s.
√γn

Pn
k=1 r(2)
k
 = o(1)
a.s.
(58)

24


---Page Break---
Then,
1
√γn
(θn −θ∗)
dist.
−−−−→
n→∞N(0, V),
(59)

where

VHT + HV = −U
in case C2 (i),
V(Id + 2γ⋆HT ) + (Id + 2γ⋆H)V = −2γ⋆U
in case C2 (ii).
(60)

Note that the matrix U in the condition C4(b) of Theorem D.1 was assumed to be positive definite in
the original Theorem 2.1 [29]. It was only to ensure that the solution V to the Lyapunov equation (60)
is positive definite, which was only used for the stability of the related autonomous linear ODE (e.g.,
Theorem 3.16 [15] or Theorem 2.2.3 [36]). However, in this paper, we do not need strict positive
definite matrix V. Therefore, we extend U to be positive semi-definite such that V is also positive
semi-definite (see Lemma D.2 for the closed form of matrix V). Such kind of extension does not
change any of the proof steps in [29].

D.1
Discussion about C1-C3

Our Assumption 2.1 corresponds to C1 by letting function h(θ) = −∇f(θ) therein. We can also let
γ⋆in Theorem 3.3 large enough to satisfy C2. The typical form of step size, also indicated in [29], is
polynomial step size γn ∼γ⋆/na for a ∈(0.5, 1]. Note that a ∈(0.5, 1) satisfies C2 (i) and a = 1
satisfies C2 (ii). Assumption 2.4 corresponds to C3.4

D.2
Analysis of C4

To check condition C4, we need to analyze the Martingale difference sequence {ei
n}. Recall
ei
n+1 = mθin(Xi
n) −Pimθin(Xi
n−1) such that there exists a constant C,

E
hei
n+1
2+τ |Fn
i
≤CE
hmθin(Xi
n)
2+τ +
Pimθin(Xi
n−1)
2+τ Fn
i

= C
X

Y ∈X i
Pi(Xi
n−1, Y )
mθin(Y )
2+τ + C
Pimθin(Xi
n−1)
2+τ .
(61)

Since ∥mθin(Y )∥< ∞almost surely by Assumption 2.4 and X i is a finite state space, at all time n,
we have
X

Y ∈X i
Pi(Xi
n−1, Y )
mθin(Y )
2+τ < ∞
a.s.
(62)

and there exists another constant C′ such that by definition of Pimθin(Xi
n−1), we have
Pimθin(Xi
n−1)
2+τ ≤C′ X

Y ∈X i
Pi(Xi
n−1, Y )
mθin(Y )
2+τ < ∞
a.s.
(63)

Therefore, E[∥ei
n+1∥2+τ|Fn] < ∞a.s. for all n and C4.(a) is satisfied.

We now turn to C4.(b). Note that for any i ̸= j, we have E[ei
n+1(ej
n+1)T |Fn] = E[ei
n+1|Fn] ·
E[(ej
n+1)T |Fn] = 0 due to the independence between agent i and j, and E[ei
n+1|Fn] = 0. Then, we
have

E





 
1
N

N
X

i=1
ei
n+1

!  
1
N

N
X

i=1
ei
n+1

!T 
Fn



=
1
N 2

N
X

i=1
E

ei
n+1(ei
n+1)T  Fn

.
(64)

The analysis of E[ei
n+1(ei
n+1)T |Fn] is inspired by Section 4 [29] and Section 4.3.3 [22], where they
constructed another Poisson equation to further decompose the noise terms therein.5 Here, expanding

4Theorem D.1 is slightly modified in terms of condition C3, which is mentioned as a special case in Section
2.2 [29]. For the sake of mathematical simplicity, we stick to condition C3 in the proof.
5However, we note that [29, 22] considered the Lipschitz continuity of function F i
θi(x) defined in (68) as an
assumption instead of a conclusion, where we give a detailed proof for this. We also obtain matrix Ui in an
explicit form, which coincides with the definition of asymptotic covariance matrix and was not simplified in
[29]. The discussion on the improvement of Ui is outlined in Section 3.2, which was not the focus of [29, 22]
and was not covered therein.

25


---Page Break---
E[ei
n+1(ei
n+1)T |Fn] gives

E

ei
n+1(ei
n+1)T  Fn

=E[mθin(Xi
n)mθin(Xi
n)T |Fn] + Pimθin(Xi
n−1)
 
Pimθin(Xi
n−1)
T

−E[mθin(Xi
n)|Fn]
 
Pimθin(Xi
n−1)
T−Pimθin(Xi
n−1)E[mθin(Xi
n)T |Fn]

=
X

y∈Xi
Pi(Xn−1, y)mθin(y)mθin(y)T −Pimθin(Xi
n−1)
 
Pimθin(Xi
n−1)
T .

(65)
Denote by

Gi(θi, x) ≜
X

y∈Xi
Pi(x, y)mθi(y)mθi(y)T −Pimθi(x) (Pimθi(x))T ,
(66)

and let its expectation w.r.t the stationary distribution πi be gi(θi) ≜Ex∼πi[Gi(θi, x)], we can
construct another Poisson equation, i.e.,

E

ei
n+1(ei
n+1)T  Fn

−
X

Xin∈Xi
π(Xi
n)E

ei
n+1(ei
n+1)T  Fn


=Gi(θi
n, Xi
n−1) −gi(θi
n)

=φi
θin(Xi
n−1) −Piφi
θin(Xi
n−1),

(67)

for some matrix-valued function φi : Rd × Xi →Rd×d. Following the similar steps shown in (42) -
(45), we can obtain the closed-form expression

φi
θi(x) =
X

y∈Xi

 
I −Pi + 1(πi)T −1 (x, y)Gi(θi, x) −gi(θi).
(68)

Then, we can decompose (65) into
Gi(θi
n, Xi
n−1) = gi(θ∗)
| {z }
Ui

+ gi(θi
n) −gi(θ∗)
|
{z
}

D(1)
i,n

+ φi
θin(Xi
n) −Piφi
θin(Xi
n−1)
|
{z
}

D(2,a)
i,n

+ φi
θin(Xi
n−1) −φi
θin(Xi
n)
|
{z
}

D(2,b)
i,n

.

(69)
Let U ≜
1
N2
PN
i=1 Ui, D(1)
n
≜
1
N 2
PN
i=1 D(1)
1,n, D(2,a)
n
≜
1
N2
PN
i=1 D(2,a)
i,n , and D(2,b)
n
≜

1
N2
PN
i=1 D(2,b)
i,n , we want to prove that D(1)
n
satisfies the first condition in C4, and D(2,a)
n
, D(2,b)
n
meet the second condition in C4.

We now show that for all i, Gi(θi, x) is Lipschitz continuous in θi ∈Ωfor some compact subset
Ω⊂Rd. For any x ∈Xi and θi
1, θi
2 ∈Ω, we can get

∥mθi
1(x)mθi
1(x)T −mθi
2(x)mθi
2(x)T ∥

=∥mθi
1(x)(mθi
1(x) −mθi
2(x))T −(mθi
1(x) −mθi
2(x))mθi
2(x)T ∥

≤∥mθi
1(x) −mθi
2(x)∥(mθi
1(x)∥+ ∥mθi
2(x)∥)

≤C∥θi
1 −θi
2∥,

(70)

for some constant C, where the last inequality comes from ∥mθi
1(x)∥< ∞since θi
1 ∈Ωand the
Lipschitz continuous function mθi(x). Similarly, we can get ∥Pimθi
1(x)−Pimθi
2(x)∥≤C∥θi
1−θi
2∥.
Therefore, Gi(θi, x) and gi(θi) are Lipschitz continuous in θi ∈Ωfor any x ∈Xi.

For the sequence {D(1)
i,n}n≥0, by applying Theorem 3.2 and conditioned on limn→∞θn = θ∗for
an optimal point θ∗∈L, we have limn→∞∥gi(θi
n) −gi(θ∗)∥≤limn→∞C∥θi
n −θ∗∥= 0. This
implies D(1)
i,n →0 for every i ∈[N] and thus D(1)
n
→0 as n →∞almost surely, which satisfies the
first condition in (57).

For the Martingale difference sequence {D(2,a)
i,n }n≥0, we use Burkholder inequality (e.g., Theorem
2.10 [33], [21]) such that for p ≥1 and some constant Cp,

E

"

n
X

i=1
D(2,a)
i,n



p#

≤CpE





 n
X

i=1

D(2,a)
i,n

2
!p/2

.
(71)

26


---Page Break---
By the definition (66) and Assumption 2.4, for a sample path, supn ∥Gi(θi
n, x)∥< ∞for any x ∈Xi,
as well as supn ∥gi(θi
n)∥< ∞, which leads to supn ∥φi
θin(x)∥< ∞for any x ∈Xi because of (68).

Then, we have supn ∥D(2,a)
i,n ∥≤C < ∞for the path-dependent constant C. Taking p = 1 and we
have

lim
n→∞γnCp

v
u
u
t

n
X

i=1

D(2,a)
i,n

2
≤lim
n→∞CpCγn
√n = 0
a.s.
(72)

Thus, Lebesgue dominated convergence theorem gives

lim
n→∞γnCpE





v
u
u
t

n
X

i=1
∥D(2,a)
i,n ∥2



= E



lim
n→∞γnCp

v
u
u
t

n
X

i=1
∥D(2,a)
i,n ∥2



= 0

and we have limn→∞γnE[∥Pn
i=1 D(2,a)
i,n ∥] = 0.

For the sequence {D(2,b)
i,n }n≥0, we have

n
X

k=1
D(2,b)
i,k
=

n
X

k=1


φi
θi
k(Xi
k−1) −φi
θi
k−1(Xi
k−1)

+ φi
θi
0(Xi
0) −φi
θi
n(Xi
n)

=

n
X

k=1


φi
θi
k(Xi
k−1)−φi
θk(Xi
k−1)+φi
θk(Xi
k−1)−φi
θk−1(Xi
k−1)+φi
θk−1(Xi
k−1)−φi
θi
k−1(Xi
k−1)


+ φi
θi
0(Xi
0) −φi
θin(Xi
n).
(73)

Since Gi(θi, x) and gi(θi) are Lipschitz continuous in θi ∈Ω, φi
θi(x) is also Lipschitz continuous
in θi ∈Ωand is bounded. We have


n
X

k=1
D(2,b)
i,k

 ≤



n
X

k=1
φi
θi
k(Xi
k−1) −φi
θi
k−1(Xi
k−1)

 +
φi
θi
0(Xi
0)
 +
φi
θin(Xi
n)


≤



n
X

k=1
φi
θi
k(Xi
k−1) −φi
θi
k−1(Xi
k−1)

 + D1

≤

n
X

k=1
D2DΩγk + D1

(74)

where ∥φi
θi
0(Xi
0)∥+ ∥φi
θin(Xi
n)∥≤D1 for a given sample path, D2 is the Lipschitz constant of

φi
θi(x), and ∥∇Fi(xi, Xi)∥≤DΩfor any xi ∈Ωand Xi ∈X i. Then,

γn



n
X

k=1
D(2,b)
i,k

 ≤D2DΩγn

n
X

k=1
γk + γnD1 →0
as n →∞
(75)

because γn
Pn
k=1 γk = O(n1−2a) by assumption 2.3. Therefore, the second condition of C4 is
satisfied.

D.3
Analysis of C5

We now analyze condition C5. The decreasing rate of each term in (56) has been proved in Appendix C.
Specifically, by assumption 2.4, there exists a compact subset for a given sample path, and

• we have shown that
r(A)
n
 = O(ηn) a.s., which implies
r(A)
n
 = o(√γn) a.s.

• For 1
N
PN
i=1 ξi
n, in the case of increasing communication interval, 1

N
PN
i=1 ξi
n = O(γn +
ηn), by Assumption 2.3-ii), we know (γn + ηn)/√γn = √γn + √γnKL+1
τn
= o(1) such
that ∥1

N
PN
i=1 ξi
n∥= o(√γn) almost surely. On the other hand, in the case of bounded
communication interval, 1

N
PN
i=1 ξi
n = O(γn) such that ∥1

N
PN
i=1 ξi
n∥= o(√γn) a.s.

27


---Page Break---
• Since supn ∥νi
n∥< ∞almost surely, we have supp ∥1

N
PN
i=1
Pp
k=0(νi
k −νi
k+1)∥=
supp ∥1

N
PN
i=1(νi
0 −νi
p+1)∥< ∞almost surely.
Then, √γp∥1

N
PN
i=1
Pp
k=0(νi
k −
νi
k+1)∥= O(√γp) leads to √γp∥1

N
PN
i=1
Pp
k=0(νi
k −νi
k+1)∥= o(1) a.s.

Let r(1)
n
≜r(A)
n
+ 1

N
PN
i=1 ξi
n and r(2)
n
≜1

N
PN
i=1(νi
k −νi
k+1). From above, we can see that C5 in
Theorem D.1 is satisfied and we show that all the conditions in Theorem D.1 have been satisfied.

D.4
CLosed Form of Limitimg Covariance Matrix

Lastly, we need to analyze the closed-form expression of U as in C4 (b) of Theorem D.1. Recall
that U =
1
N2
PN
i=1 Ui and Ui = gi(θ∗) in (69). We now give the exact form of function gi(θ∗) as
follows:

gi(θ∗) =
X

x∈Xi
πi(x)



mθ∗(x)mθ∗(x)T −



X

y∈Xi
Pi(x, y)mθ∗(y)







X

y∈Xi
Pi(x, y)mθ∗(y)





T 



= E





 ∞
X

s=0
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!  ∞
X

s=0
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!T 



−E





 ∞
X

s=1
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!  ∞
X

s=1
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!T 



= E
h 
∇Fi(θ∗, Xi
0) −∇fi(θ∗)
  
∇Fi(θ∗, Xi
0) −∇fi(θ∗)
T i

+ E



 
∇Fi(θ∗, Xi
0) −∇fi(θ∗)

 ∞
X

s=1
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!T 



+ E

" ∞
X

s=1
[∇Fi(θ∗, Xs) −∇fi(θ∗)]

!
 
∇Fi(θ∗, Xi
0) −∇fi(θ∗)
T
#

= Cov(∇Fi(θ∗, X0), ∇Fi(θ∗, X0))

+

∞
X

s=1
[Cov(∇Fi(θ∗, X0), ∇Fi(θ∗, Xs)) + Cov(∇Fi(θ∗, Xs), ∇Fi(θ∗, X0))] ,

= Σ(∇F(θ∗, ·)).
(76)

where the second equality comes from the recursive form of mθi(x) in (45), and that the process
{Xn}n≥0 is in its stationary regime, i.e., X0 ∼πi from the beginning. The last equality comes
from rewriting Cov(∇Fi(θ∗, Xi), ∇Fi(θ∗, Xj)) in a matrix form. Note that gi(θ∗) is exactly the
asymptotic covariance matrix of the underlying Markov chain {Xi
n}n≥0 associated with the test
function ∇Fi(θ∗, ·). By utilizing the following lemma, we can obtain the explicit form of V as
defined in (60).
Lemma D.2 (Lemma D.2.2 [41]). If all the eigenvalues of matrix M have negative real part, then for
every positive semi-definite matrix U there exists a unique positive semi-definite matrix V satisfying
U + MV + VMT = 0. The explicit solution V is given as

V =
Z ∞

0
eMtUe(MT )tdt.
(77)

D.5
CLT of Polyak-Ruppert Averaging

We now consider the CLT result of Polyak-Ruppert averaging ¯θn = 1

n
Pn−1
k=0 θk. The steps follow
similar way by verifying that the conditions in the related CLT of Polyak-Ruppert averaging for the
stochastic approximation are satisfied. The additional assumption is given below.

28


---Page Break---
C6. For the sequence {rn} in (38), n−1/2 Pn
k=0 r(1)
k
→0 with probability 1.

Then, the CLT of Polyak-Ruppert averaging is as follows.

Theorem D.3 (Theorem 3.2 of [29]). Consider the iteration (38), assume C1, C3, C4, C5 in
Theorem D.1 are satisfied. Moreover, assume C6 is satisfied. Then, with step size γn ∼γ⋆/na for
a ∈(0.5, 1), we have
√n(¯θn −θ∗)
dist.
−−−−→
n→∞N(0, V′),
(78)

where V′ = H−1UH−T .

Discussion about C1 and C3 can be found in Section D.1. Condition C4 has been analyzed in Section
D.2 and condition C5 has been examined in Section D.3. The only condition left to analyze is C6,
which is based on the results obtained in Section D.3. In view of (56), r(1)
n
= r(A)
n
+ 1

N
PN
i=1 ξi
n+1,
so C6 is equivalent to

n−1/2
n
X

k=1

"

r(A)
k
+ 1

N

N
X

i=1

 
ξi
k+1

#

→0
w.p.1.
(79)

In Section D.3, we have shown that
r(A)
n
 = O(ηn), 1

N
PN
i=1 ξi
n = O(γn). Note that by Assump-

tion 2.3, we consider bounded communication interval for step size γn ∼γ⋆/na for a ∈(0.5, 1), and

hence, ηn = O(γn) such that
r(A)
n
 = O(γn). We then know that

n
X

k=1

r(A)
n
 = O(n1−a),

n
X

k=1
∥1

N

N
X

i=1
ξi
n∥= O(n1−a),
(80)

such that

n−1/2
n
X

k=1

r(A)
k
+ 1

N

N
X

i=1

 
ξi
k+1

 = O(n1/2−a) = o(1),
(81)

which proved (79) and C6 is verified. Therefore, Theorem D.3 is proved under our Assumptions 2.1 -
2.5.

E
Discussion on the comparison of Theorem 3.3 to the CLT result in [51]

As a byproduct of our Theorem 3.3, we have the following corollary.

Corollary E.1. Under Assumptions 2.1 - 2.5, for the sub-sequence {nl}l≥0 where Kl = K for all l,
we have
1
√nl

l
X

k=1
(¯θnk −θ∗)
dist.
−−−→
l→∞N(0, V′)
(82)

Proof. Since Kl = K for all l, we have nl = Kl. There is an existing result showing the CLT result
of the partial sum of a sub-sequence (after normalization) has the same normal distribution as the
partial sum of the original sequence.

Theorem E.2 (Theorem 14.4 of [8]). Given a sequence of random variable θ1, θ2, · · · with partial
sum Sn ≜Pn
k=1 θk such that
1
√nSn
dist.
−−−−→
n→∞N(0, V). Let nl be some positive random variable

taking integer value such that θnl is on the same space as θn. In addition, for some sequence {bl}l≥0
going to infinity, nl/bl →c for a positive constant c. Then,
1
√nl Snl
dist.
−−−→
l→∞N(0, V).

From Theorem E.2 and our Theorem 3.3, we have
1
√nl
Pl
k=1(¯θnk −θ∗)
dist.
−−−→
l→∞N(0, V′).

29


---Page Break---
Recently, [51] studied the CLT result under the LSGD-FP algorithm with i.i.d sampling (with slightly
different setting of the step size). We are able recover Theorem 3.1 of [51] under the constant
communication interval while adjusting their step size to make a fair comparison. We state their
algorithm below for self-contained purpose. During each communication interval n ∈(nl, nl+1],

θi
n+1 =

(
θi
n −γl∇Fi(θi
n, Xi
n)
if n ∈(nl, nl+1),
1
N
PN
i=1(θi
n −γl∇Fi(θi
n, Xi
n))
if n = nl+1.
(83)

The CLT result associated with (83) is given below.

Theorem E.3 (Theorem 3.1 of [51]). Under LSGD-FP algorithm with i.i.d. sampling, we have

√nl

l

l
X

k=1

 ¯θnk −θ∗
dist.
−−−→
l→∞N(0, νV′),
(84)

where ν ≜liml→∞1

l2 (Pl
k=1 Kl)(Pl
k=1 K−1
l
).

Note that ν = 1 for constant K. We can rewrite (84) as
√nl

l

l
X

k=1

 ¯θnk −θ∗
=
√nl
√

l
1
√

l

l
X

k=1
(¯θnk −θ∗) =
√

K 1
√

l

l
X

k=1
(¯θnk −θ∗)
(85)

such that
1
√

l

l
X

k=1
(¯θnk −θ∗)
dist.
−−−→
l→∞N(0, 1

K V′).
(86)

Note that the step size in (83) keeps unchanged during each communication interval, while ours in
(1) keeps decreasing even in the same communication interval. This makes our step size decreasing
faster than theirs. To make a fair comparison, we only choose a sub-sequence {nKl}l≥0 in (86) such
that it is ‘equivalent’ to see that our step sizes become the same at each aggregation step. In this case,
we again use Theorem E.2 to obtain

1
√

Kl

l
X

s=1
(¯θnKs −θ∗)
dist.
−−−→
l→∞N(0, 1

K V′),
(87)

such that
1
√

l

l
X

s=1
(¯θnKs −θ∗) =
√

K
1
√

Kl

l
X

s=1
(¯θnKs −θ∗)
dist.
−−−→
l→∞N(0, V′).
(88)

Therefore, our Corollary E.1 also recovers Theorem 3.1 of [51] under the constant communication
interval K, but with more general communication patterns and Markovian sampling.

F
Discussion on Communication Patterns

F.1
Examples of Communication Matrix W

Metropolis Hasting Algorithm: In the decentralized learning such as D-SGD, HLSGD and DFL,
W at the aggregation step can be generated locally using the Metropolis Hasting algorithm based on
the underlying communication topology, and is deterministic [62, 43, 80]. Specifically, each agent i
exchanges its degree di with its neighbors j ∈N(i), forming the weight W(i, j) = min{1/di, 1/dj}
for j ∈N(i) and W(i, i) = 1 −P
j̸=N(i) W(i, j). In this case, W is doubly stochastic and
symmetric. By Perron-Frobenius theorem, its SLEM λ2(W) < 1 . Then, ∥WT W −J∥=
∥W2 −J∥= λ2
2(W) < 1, which satisfies Assumption 2.5-ii). It is worth noting that this algorithm
is robust to time-varying communication topologies.

Client Sampling in FL: For LSGD-FP studied in [67, 76, 42], W = 11T /N trivially satisfies
Assumption 2.5-ii). For LSGD-PP on the other hand, only a small fraction of agents participate in
each aggregation step for consensus [50, 30]. Denote by S a randomly selected set of agents (without
replacement) of fixed size |S| ∈{1, 2, · · · , N} at time n and WS plays a role of aggregating θi
n

30


---Page Break---
for agent i ∈S. Additionally, the central server needs to broadcast updated parameter θn+1 to
the newly selected set S′ with the same size, which results in a bijective mapping σ (for S →S′
and [N]/S →[N]/S′) and a corresponding permutation matrix TS→S′. Then, the communication
matrix becomes W = TS→S′WS.6 Specifically, TS→S′(i, j) = 1 if j = σ(i) and TS→S′(i, j) = 0
otherwise. Besides, WS(i, j) = 1/|S| for i, j ∈S, WS(i, i) = 1 for i /∈S, and WS(i, j) = 0
otherwise. Note that WS is now a random matrix, since S is a randomly chosen subset of size
|S|. Clearly, for each choice of S, WS is doubly stochastic, symmetric and W2
S = WS. Taking
the expectation of WS w.r.t the randomly selected set S gives ES[WS](i, i) = 1 −(|S| −1)/N
for i ∈[N] and ES[WS](i, j) = (|S| −1)/N(N −1) for i ̸= j. Note that ES[WS] has all
positive entries. Therefore, we use the fact TT T = I for permutation matrix T such that ∥E[W] −
J∥= ∥ES,S′[WT
S TT
S,S′TS,S′WS] −J∥= ∥ES[WT
S WS] −J∥= ∥ES[WS] −J∥< 1 by
Perron–Frobenius theorem and eigendecomposition, which satisfies Assumption 2.5-ii).

F.2
Discussion on partial client sampling

The commonly used partial client sampling algorithm in the FL literature [50, 30] is FedAvg as
follows:

1. At time n, the central server updates its global parameter θn =
1
|S|
P

i∈S θi
n from the
agents in the previous set S. Then, the central server selects a new subset of agents S′ and
broadcasts θn to agent i ∈S′, i.e., θi
n = θn;
2. Each selected agent i computes K steps of SGD locally and consecutively updates its local
parameter θi
n+1, · · · , θi
n+K according to (1a);

3. Each selected agent i ∈S′ uploads θi
n+K to the central server.

Then, the central server repeats the above three steps with θn+K and a new set of selected agents.

In our client sampling scheme, at the aggregation step n, the design of WS results in ˜θi
n =
1
|S|
P

j∈S θj
n for a selected agent i ∈S, and ˜θi
n = θi
n for an unselected agent i /∈S. Mean-

while, the central server updates the global parameter ˜θn = ˜θi
n for i ∈S. Then, the permutation
matrix TS→S′ ensures that the newly selected agent i ∈S′ will use ˜θn as the initial point for its subse-
quent SGD iterations. Consequently, from the selected agents’ perspective, the communication matrix
W = TS→S′WS corresponds to step 1 in FedAvg. As we can observe, both algorithms update
the global parameter identically from the central server’s viewpoint, rendering them mathematically
equivalent regarding the global parameter update.

We acknowledge that under the i.i.d sampling strategy, the behavior of unselected agents in our
algorithm differs from FedAvg. Specifically, unselected agents are idle in FedAvg, while they
continue the SGD computation in our algorithm (despite not contributing to the global parameter
update). Importantly, when an unselected agent is later selected, the central server overwrites its local
parameter during the broadcasting process. This ensures that the activities of agents when they are
unselected have no impact on the global parameter update.

To our knowledge, the FedAvg algorithm under the Markovian sampling strategy remains unexplored
in the FL literature. Extrapolating the behavior of unselected agents in FedAvg from i.i.d sampling to
Markovian sampling suggests that unselected agents would remain idle. In contrast, our algorithm
enables unselected agents to continue evolving Xi
n. These additional transitions contribute to faster
mixing of the Markov chain for each unselected agent and a smaller bias of Fi(θ, Xi
n) relative to its
mean field fi(θ), potentially accelerating the convergence.

G
Additional Simulation

G.1
Simulation Setup in Section 4

This simulation is performed on a PC with an AMD R9 5950X, RTX 3080 and 128 GB RAM. In
this simulation, we assume that agents follow the DSGD algorithm (1). In Figure 2(a) - 2(c), each

6In Appendix F.2, we will discuss the mathematical equivalence between our client sampling strategy and
the commonly used one in the FL literature [50, 30].

31


---Page Break---
agent holds a disjoint local dataset (non-overlapping data points for every agent), while we distribute
the ijcnn1 dataset [14] with more varied distribution among 100 agents by leveraging Dirichlet
distribution with the default alpha value of 0.5 in Figure 2(d) - 2(f).

Moreover, we assume that all agents are distributed over a communication network. In order to create
this network among 100 agents and the graph-like dataset structure held by each agent, we utilize
connected sub-graphs from the real-world graph Facebook in SNAP [49]. All 100 agents collaborate
together to generate a deterministic communication matrix W = [Wij] with Metropolis Hasting
algorithm of the following form: For i, j ∈[N], we have

Wij =

(
min
n
1
di , 1

dj

o
if agent j is the neighbor of agent i,

0
otherwise,

Wii = 1 −
X

j∈[N]
Wij,

where di represents the degree of agent i in the graph. The communication interval K is set to 1, as
is the usual choice in DSGD [71, 80, 61, 68].

For the first group of agents, we assume they have full access to their datasets, thus performing i.i.d.
sampling or single shuffling. In particular,

• i.i.d. sampling employed by agent i means that the data point Xi
n is independently and
uniformly sampled from its dataset Xi at each time n.
• Single shuffling, by its name, only shuffles the dataset once and adheres to that specific order
throughout the training process.

On the other hand, within the second group of agents, we assume that they hold graph-like datasets.
Now, we introduce simple random walk (SRW), non-backtracking random walk (NBRW), and
self-repellent random walk (SRRW) in order:

• SRW refers to the walker that chooses one of the neighboring nodes uniformly at random.
• NBRW, as studied in [2, 48, 5], is a variation of SRW, which selects one of the neighbors
uniformly at random, with the exception of the one visited in the last step.
• SRRW, recently proposed by [25], is designed with a nonlinear transition kernel K[x] ∈
[0, 1]N×N of the following form:

Kij[x] ≜
Pij(xj/µj)−α
P

k∈[N] Pik(xk/µk)−α ,
∀i, j ∈[N],
(89)

where matrix P = [Pij] is the transition kernel of the baseline Markov chain and µ = [µi] is
its corresponding stationary distribution. Additionally, α denotes the force of self repellence,
and larger α leads to stronger force of self repellence, thus higher sampling efficiency
[25, Corollary 4.3]. Moreover, vector x ∈RN is in the interior of probability simplex,
representing the empirical distribution, in other words, the visit frequency of each node in
the graph. The update rule of this empirical distribution is in the following form:

xn+1 = xn + βn+1(δXn+1 −xn),
(90)

where βn ≜(n + 1)−b is the step size of SRRW iterates. b = 1 was original proposed in
[25] and is recently extended to b ∈(0.5, 1) in [39]. In this simulation, we use SRW as the
baseline Markov chain of SRRW, and in turn µ is proportional to the degree distribution.
We also assume x0 = 1/N, i.e., each node has been visited once, and choose the step
size βn = (n + 1)−0.8, force of self repellence α = 20 according to the suggestion in [39,
Section 4].

Since SRW, NBRW, and SRRW all admit the stationary distribution that is proportional to degree
distribution, in order to obtain unfirom target in (15), we need to reweight the gradient computed by
each agent i in order to maintain an asymptotic unbiased gradient. Thus, agent i should modify its
SGD update from (1a) to the following:

θi
n+1/2 = θi
n −γn+1∇Fi(θi
n, Xi
n)/d(Xi
n).
(91)

32


---Page Break---
0
25
50
75
100
125
150
175
200
Number of rounds (n)

0.6

0.8

1.0

1.2

1.4

1.6

1.8

2.0

2.2

training loss

Shuffling, SRW
Shuffling, NBRW
Shuffling, SRRW

0
25
50
75
100
125
150
175
200
Number of rounds (n)

0.0

0.5

1.0

1.5

2.0

training loss

Shuffling, SRRW with 10 clients
Shuffling, SRRW with 20 clients
Shuffling, SRRW with 30 clients
Shuffling, SRRW with 40 clients

0
25
50
75
100
125
150
175
200
Number of rounds (n)

0.10

0.15

0.20

0.25

0.30

0.35

0.40

0.45

0.50

training loss

Shuffling, SRW
Shuffling, NBRW
Shuffling, SRRW(
= 10)

Figure 3: Image classification experiment. From left to right: (a) Comparison of various sampling
strategies in image classification problem using 5-layer neural network. (b) Train a 5-layer CNN
model with different number of total agents (clients) to show the linear speedup effect. (c) Train
ResNet-18 model with different sampling strategies among 10 agents with participation ratio 0.4.

G.2
Image Classification Task

In this part, we perform the image classification task through a 5-layer neural network, where the
CIFAR10 dataset [44] with 50k image data is evenly distributed to 10 agents. Each agent possesses
5k images, which are further divided into 200 batches, each batch with 25 images.

The Convolutional Neural Network (CNN) model used in this simulation encompasses:

• Two convolutional layers (i.e., nn.Conv2d(3, 6, 5) and nn.Conv2d(6, 16, 5)), each followed by
ReLU activation functions to introduce non-linearity and max pooling (i.e., nn.MaxPool2d(2,
2)) to reduce spatial dimensions.
• Three fully connected (linear) layers, concluding with a softmax output to handle the
multi-class classification problem.

Similar to the simulation setup in Section 4, among the 10 participating agents, five have unrestricted
access to their respective data allocations, enabling them to utilize the shuffling method to iterate
through their batches. The other five agents are designed to simulate limited data access scenarios.
Their data access is structured using five distinct graph topologies extracted from the SNAP dataset
collection [49], each graph simulating a unique communication pattern among the batches (nodes) of
data. Within these topologies, agents adopt one of three random walk strategies — SRW, NBRW, and
SRRW, all with importance reweighting — to sample the batches for training.

Local model training is conducted for five epochs at each agent before aggregating the updates at a
central server — a process repeated for a total of 200 communication rounds. Each epoch consists of a
full traversal of the local dataset of agents in the first group, or 200 batches sampled for training in the
second group of agents. To mimic realistic conditions, we also introduce partial agent participation
where only 40% of agents are selected randomly to transmit their updates in each round, reflecting
the intermittent communication in real-world FL deployments. Lastly, the selection of the step size
βn for SRRW iterates (90) is a critical aspect of our experiments. In this simulation, we experiment
with various values of b ∈{0.501, 0.6, 0.7, 0.8, 0.9} to determine the most advantageous setting for
maximizing the benefits of the SRRW strategy. Based on our findings, the best choice for the SRRW
step size is b = 0.501, in other words, βn = (n + 1)−0.501.

The simulation result is quantified by averaging the training loss across ten repeated trials for each
configuration. As depicted in Figure 3(a), the training results are consistent with our previous findings
in Figure 2(a) in the context of the FL framework and the training of neural networks: the use of a
more effective sampling approach, even for a portion of the agents, results in significant enhancements
in the overall training of the model, and this improvement is further enhanced through the highly
efficient sampling strategy SRRW.

In Figure 3(b) and 3(c), we perform image classification experiments in the FL setting with partial
client participation. Only 4 random agents will participate in the training process at each aggregation
phase. In Figure 3(b), we fix the sampling strategy (shuffling, SRRW with α = 10) and test
the linear speedup effect for the 5-layer CNN model by duplicating 10 agents to N agents with
N ∈{10, 20, 30, 40}, keeping the same participation ratio 0.4. As can be seen from the plot, the
training loss is inversely proportional to the number of agents, i.e., at 200 rounds, the training loss
is 0.52 for 10 agents, 0.23 for 20 agents, 0.18 for 30 agents, and 0.12 for 40 agents. In Figure 3(c),
we extend the current simulation from 5-layer CNN model to ResNet-18 model [34] in order to

33


---Page Break---
numerically test the performance of different sampling strategies in a more complex neural network
training task. By fixing the shuffling in the first group of agents, we observe that improving Markovian
sampling from SRW to NBRW, then to SRRW, gives accelerated training process with smaller training
loss.

H
Limitations

Our study provides crucial insights into the identification of nuanced agents’ sampling behaviors in
UD-SGD, where improving each agent’s sampling strategy speeds up overall system performance
without additional computational burden except the additional storage for the visit counts used for
sampling their datasets. Our UD-SGD is scalable in terms of larger datasets as the sampling strategy
(i.e., random walk) utilized by each agent leverages only local information for its dataset. However,
our work has two limitations that should be acknowledged.

1. Assumption 2.4 posits that the parameter trajectory {θn} is almost surely bounded, which is
a strong assumption. This is crucial for guaranteeing the well-defined nature of all related
quantities. Some mechanisms such as projections onto a compact subset [45, Chapter 5.1],
or truncation-based method with expanding compact subsets can do the trick to ensure that
the iteration is always bounded [3]. As mentioned in our discussion after Assumption 2.4,
only recently the stability of SGD under Markovian sampling has been guaranteed to hold
for some class of objective function f [9], while the discussion on stability issue under
multi-agent scenario with Markovian sampling remains an open problem and we do not
pursue to remove this stability assumption in this paper.
2. Asymptotic analysis: The main results of our work, i.e., almost sure convergence and
central limit theorem in distributed optimization, are based on asymptotic analysis and
might not accurately represent the finite-sample performance of each contributing agent
in the system. The state-of-the-art finite-sample analysis in the literature only focuses on
the worst-performing agent that cannot capture the nuanced differences between agents’
sampling strategies, with the explanation detailed in Footnote 2. Thus, a finite-sample error
bound that distinguish every agent’s dynamics is still unknown and regarded as a future
direction.

Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract and the introduction clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. The claims
made match theoretical and experimental results, and reflect how much the results can be
expected to generalize to other settings.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The ‘Limitations’ section is provided in Appendix H.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: All the theorems, formulas, and proofs in the paper have been numbered and
cross-referenced. All assumptions have been clearly stated or referenced in the statement of
any theorems. The complete and correct proofs have been provided in appendices.
4. Experimental Result Reproducibility

34


---Page Break---
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The detailed simulation setups have been provided in Appendix G.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [No]

Justification: We do not provide clean source codes, but detailed instructions to perform the
experiment have been provided in Appendix G.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: The full details have been provided in Appendix G.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]
Justification: We plot the error bars in Figure 2(a). However, error bars are intentionally
omitted in Figure 2 in the main body to avoid a cluttered plot and deliver the main message.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: We specify the platform that runs the simulation in Appendix G.1.

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We fully obey the NeurIPS Code of Ethics.

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [NA]

Justification: This work mainly aims to demonstrate how the improvement of partial agents’
sampling strategies can accelerate the overall system’s performance. It advocates for
improving common wealth and has no negative social impact.

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

35


---Page Break---
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]

Justification: We have cited correctly the datasets ijcnn1 and CIFAR-10 used for experiments
in our paper.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.

36


---Page Break---
