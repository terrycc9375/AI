Is O(log N) practical?
Near-Equivalence Between Delay Robustness and
Bounded Regret in Bandits and RL

Enoch H. Kang∗
Foster School of Business
University of Washington
Seattle, WA 98195

P. R. Kumar
Electrical & Computer Engineering
Texas A&M University
College Station, TX 77843

Abstract

Interactive decision making, encompassing bandits, contextual bandits, and rein-
forcement learning, has recently been of interest to theoretical studies of experi-
mentation design and recommender system algorithm research. One recent finding
in this area is that the well-known Graves-Lai constant being zero is a necessary
and sufficient condition for achieving bounded (or constant) regret in interactive
decision-making. As this condition may be a strong requirement for many appli-
cations, the practical usefulness of pursuing bounded regret has been questioned.
In this paper, we show that the condition of the Graves-Lai constant being zero
is also necessary for a consistent algorithm to achieve delay model robustness
when reward delays are unknown (i.e., when feedback is anonymous). Here, model
robustness is measured in terms of ϵ-robustness, one of the most widely used and
one of the least adversarial robustness concepts in the robust statistics literature.
In particular, we show that ϵ-robustness cannot be achieved for a consistent (i.e.,
uniformly sub-polynomial regret) algorithm, however small the nonzero ϵ value is,
when the Grave-Lai constant is not zero. While this is a strongly negative result, we
also provide a positive result for linear rewards models (contextual linear bandits,
reinforcement learning with linear MDP) that the Grave-Lai constant being zero
is also sufficient for achieving bounded regret without any knowledge of delay
models, i.e., the best of both the efficiency world and the delay robustness world.

1
Introduction

We consider the cost of addressing stochastic and anonymous delayed rewards in Decision-Making
with Structured Observations (DMSO) [1, 2], which generalizes interactive decision-making problems,
such as structured bandits, contextual bandits, and reinforcement learning2. In many real-life
applications of interactive decision-making problems, stochastic and unknown delays in reward make
it challenging to attribute the sequence of observed outcomes to previous decisions. In medical
treatments, for example, a doctor cannot easily be sure whether a medical outcome is due to the effect
of current treatment or due to some other previously taken treatment’s delayed effect. This type of
reward delay in decisions is called an ‘unknown reward delay’ [3] or ‘delayed anonymous feedback’
[4, 5]3. Under this setting, the decision maker never observes the period information for which each

∗ehwkang@uw.edu
2One can refer to Appendix A (or [2, 1] for a more comprehensive, detailed description) to see how bandit,
contextual bandit, and episodic reinforcement learning problems can be described as DMSO problems.
3Most previous studies focus on delayed, anonymous, and aggregated (DAAF) feedback, where only the
observe sum of the rewards arriving at each episode is observed. Here, we consider impossibility results for

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Figure 1: Examples of misspecification of the reward delay model of decisions (π1, π2, π3)

reward corresponds to, even after it receives the delayed reward at the later time step. As it is not
obvious which decision caused each observed reward, reward attribution becomes a challenge.

Some knowledge (e.g., mean) of the probabilistic distribution of each decision’s reward delay,
combined with the careful design of algorithms, may help to resolve this reward attribution problem
under stochastic and anonymous delayed rewards [4]. However, those delay models themselves
may be misspecified [6]. Therefore, whether we can design an algorithm that is robust to model
misspecification becomes a main concern in the problems with stochastic and anonymous delayed
rewards.

One of the most widely used concepts of model misspecification in the robust statistics literature is
ϵ-robustness [7]. Given a parameter ϵ > 0 and true distribution D, a model distribution bD is called
an ϵ-(general) contamination of D if dT V (D, bD) ≤ϵ, where dT V denotes the total variation distance
function4. Figure 1 illustrates some examples of ϵ-contamination of the delay models. As ϵ-robustness
is also one of the weakest (i.e., least adversarial) and most elementary notion of robustness [8], the
first question on an algorithm’s delay robustness will be, “up to which ϵ the algorithm’s properties are
robust to ϵ-contamination of delay model misspecification?”.

In this paper, we prove that no consistent (i.e., uniformly sub-polynomial regret) algorithm for DMSO
can be designed to be robust to ϵ-contamination of delay model misspecification unless DMSO’s
Graves-Lai constant [9, 2, 1] being zero. While this is a strong negative result, we also provide a
positive result for linear DMSO problems (linear contextual bandit, reinforcement learning with
linear MDPs), showing that the Graves-Lai constant being zero [9, 10, 11] is sufficient for achieving
bounded regret without any knowledge of delay models. As the Graves-Lai constant being zero holds
if and only if we can achieve bounded regret [2, 1, 10, 11], the results in this paper strongly motivate
the practical usefulness of designing learning systems where we can achieve bounded regret.

1.1
Related work

While no previous work has studied the link between consistent algorithms and robustness to delay
distribution misspecification when reward delays are unknown (=anonymous) in problems related
to DMSO (e.g, bandit problems and reinforcement learning problems), there has been other work
looking at different flavors of delayed anonymous rewards. There were prior studies on delayed
rewards [12, 13, 14, 15], but [4] was the first to formalize the unknown stochastic reward delays
assumption in interactive decision making problems, which led to the literature on stochastic delayed,
anonymous and aggregated feedback (DAAF). While [4] provides a consistent algorithm for stochastic
bandits that does not require any knowledge of delay distributions, it requires a strong assumption that
the mean of delay distribution is precisely known, which cannot be achieved under ϵ-contamination
however small ϵ is5 (In Section 4, we show that no algorithm can be consistent when ϵ > 0). [16]
provides another consistent algorithm for stochastic bandits that also does not require any knowledge
of delay distributions and improves [4], but it requires a different assumption that the delayed reward
feedback exactly associates the reward and the arm and therefore rewards are not anonymous; [17,
18] make similar assumption for episodic reinforcement learning problems with stochastic delays.

the delayed, anonymous, and non-aggregated feedback, where we observe each delayed anonymous reward
separately.
4The total variation distance dTV(ν, υ) is defined as 1

2∥ν −υ∥1 = supE∈Σ |ν(E) −υ(E)|, where Σ stands
for the measurable sets on which two distributions ν and υ are defined.
5For example, for the family of distributions with k-th moment bounded by 1 for k ≥2, ϵ-contamination in
delay distribution, i.e., dT V

D, ˆD

> ϵ, implies
E [D] −E
h
ˆD
i > kϵ1−1/k (See Assumption 4.2 for more).

2


---Page Break---
While our work centers on delayed anonymous rewards—where the learner cannot associate delayed
rewards with the actions that generated them—there exists a parallel line of research addressing
non-anonymous delays, in which such associations are possible. In this context, several studies have
proposed algorithms that account for unrestricted or unbounded delays. [19] developed algorithms
for nonstochastic multiarmed bandits with unrestricted delays, achieving robust regret bounds by
employing a skipping strategy to manage excessively delayed feedback. [20] adapted Thompson
sampling to handle multiarmed bandits with unrestricted delays, extending its applicability to delayed
feedback settings without assuming bounded delays. [21] derived near-optimal regret bounds for
adversarial MDPs with delayed bandit feedback, addressing the challenges posed by feedback delays
in adversarial environments. [22] proposed an optimal algorithm for adversarial bandits experiencing
arbitrary delays, establishing regret bounds that hold even when delays are extensive. Building
on this foundation, [23] introduced a "best-of-both-worlds" algorithm that improves upon [22] by
providing both adversarial guarantees and near-optimal stochastic performance without requiring
prior knowledge of the maximal delay.

2
Preliminaries

2.1
Decision-Making with Structured Observations

The DMSO problem framework generalizes many problems such as bandit problems, contextual
bandit problems, and episodic reinforcement learning problems. DMSO is characterized by the
environment and its learning protocol, which is described as follows:

◦The environment of a DMSO problem framework is specified as a tuple (Π, R, O, F), where Π
denotes the decision space, R denotes the reward space, O denotes the observation space, and
F = Q

π∈Π Fπ denotes the model class where Fπ ⊆△R×O (Here, △E notation means the space
of all possible probability distributions over a set E).6 We use fπ to refer to an element of Fπ, with
fπ being the π-coordinate of f ∈F. A ground-truth model f ⋆∈F governs the rewards and the
observations based on the decisions made in the rounds. While f ⋆is unknown to the learner, it
is typically assumed that a set F that includes f ⋆is known to the learner. Formally, we make the
following assumption, which is often called the realizability assumption [24, 25, 26].
Assumption 2.1 (Realizability). The learner has access to the model class F containing the ground-
truth model f ⋆.

◦The learning protocol for the DMSO problem consists of n rounds. In round k ≤n,
1. The learner makes a decision πk ∈Π.
2. A reward rk ∈R and an observation ok ∈O are generated, where (rk, ok) ∼f ⋆
πk ∈Fπk
3. Learner observes ok. If there are reward delays, the learner observes Rk, the set of rewards that
arrive at the round k. Rk is equivalent to rk only if there are no reward delays.

2.2
Learning Algorithm for DMSO

Given that we characterized the DMSO problem framework, we can now describe a learning algorithm
for it. Let hk be the history up to round k, i.e., hk = {(πj, Rj, oj)}k−1
j=1 where πj ∈Π, Rj ∈R, oj ∈
O and H be the set of all possible histories of rounds for k ≥1. A learning algorithm A is defined as
an element of A ⊆(H 7→△Π), which is a subset of the set of all possible mappings from the history
space H to the set of all possible distributions over Π. That is, at each round k, given the history
hk ∈H, a learning algorithm A ∈A chooses pk = A(hk) ∈△Π. The decision at round k, πk, is
sampled from pk. Note that f ∈F, A ∈A and the round n completely determine the stochastic
behavior of the learning protocol up to round n, i.e., they induce a probability distribution we call
Pf,n,A[·] over the set of all histories up to round n. We also denote the respective expectation by
Ef,n,A[·]. When the meaning is clear from the context, we use Pf,n[·] and Ef,n[·] instead of Pf,n,A[·]
and Ef,n,A[·].

Given (r, o) ∼fπ, we denote µfπ := Efπ[r]. Let πf ∈arg maxπ∈Π µfπ denote an optimal decision
for the model f. The sub-optimality gap of decision π for model f is defined as ∆f(π) := µfπf −µfπ.
When the ground truth model is f, choosing πf at each round until round n yields the largest value of

6This model class definition is equivalent to the definition used by Wagenmaker and Foster [1] because if A
is a set, countable or not, the Cartesian product XA is defined to be the collection of all functions f : A →X.

3


---Page Break---
total reward until round n. Therefore, we can measure the optimality of an algorithm A until round n
in terms of regret, which is defined by

Regf,A(n) := EA,f,n

" n
X

k=1
∆f(πk)

#

.
(1)

2.3
Consistent Learning Algorithm’s Instance-Dependent Regret Lower Bound

Regf,A(n) is a quantity that is dependent on the true model instance f. Since the true model f
is unknown to the learner a priori, a good learning algorithm must be able to perform well for all
possible f ∈F. Therefore, one might want to define the goodness of an algorithm by its capability to
achieve the minimal value of Regf,A(n) among all possible algorithms for all the instances f ∈F.
However, this is not achievable; a bespoke algorithm that always chooses πf will outperform all
possible algorithms for the instance f, while suffering linear regret for the instances in F \ f.

Since many problems have algorithms that incur sub-polynomial regret for all instances, it is common
to exclude algorithms that suffer polynomially increasing regret in some instances. This idea is
formalized in the following definition that restricts the space of ‘interesting’ algorithms.
Definition 2.2 (Graves and Lai (1997) [9]). A learning algorithm A is called consistent if
Regf,A(n) = o (np) holds for every p > 0 and f ∈F.

For DMSO problems, it has been recently shown that any consistent algorithm’s instance-dependent
regret must satisfy the asymptotic lower bound described in the following theorem [2].
Theorem 2.3 (Dong and Ma (2022) [2]). Suppose that there are no reward delays. Then for every
instance f ∈F, the expected regret of any consistent algorithm A satisfies

lim sup
n→∞

Regf,A(n)

ln n
≥C(f) = lim
n→∞C(f, n),
(2)

where C(f, n) is the solution to the optimization equation

C(f, n) ≜min
w∈R|Π|
+

X

π∈Π
wπ∆f(π)

s.t.
X

π∈Π
wπDKL(fπ∥gπ) ≥1 , ∀g ∈F(f)c

∥w∥∞≤n,

(3)

where DKL is the KL divergence and F(f) := {g ∈F | πg = πf}.
Corollary 2.4. Let f ⋆∈F be the ground-truth model. For a consistent algorithm to achieve
sub-logarithmic regret, C(f ⋆) = 0 must hold. That is, achievement of sub-logarithmic regret for all
possible instances of F can be assured a priori only if C(f) = 0 for f ∈F.
Theorem 2.5 (Wagenmaker and Foster [1]). Expected regret of order C(f ⋆) ln n can be achieved for
DMSO problems without delays. That is, bounded regret can be assured if C(f) = 0 for f ∈F.

2.4
ϵ-contamination and Total Variation Distance

In robust statistics, one of the oldest and the most commonly used concepts for modeling contaminated
data is the concept of ϵ-contamination [7]. Given a parameter 0 < ϵ < 1 and original distribution D, a
distribution X is called an ϵ-additive contamination (or Huber contamination) of D if X is a mixture
distribution of D and an unknown arbitrary distribution E, with their selection probabilities being
(1 −ϵ) for D and ϵ for E. Furthermore, A distribution X is called an ϵ-subtractive contamination of
D if it is equivalent to an arbitrary ϵ-probability removal from D (and normalization). Finally, we
say that a distribution X is a (general) ϵ-contamination of D if X can be constructed by removing
ϵ-probability from D and replacing that ϵ with equal mass from some arbitrary distribution E.

As discussed earlier, the concept of ϵ-contamination is closely related to the concept of total variation
distance. Given a space of distributions, the total variation distance, denoted as dT V (ν, υ), is defined
as dTV(ν, υ) := 1

2∥ν −υ∥1 = supE∈Σ |ν(E) −υ(E)|, where Σ stands for the measurable sets on
which ν and υ are defined. It is well known that dT V is a metric that satisfies interesting properties

4


---Page Break---
such as 1) dT V = 0 if and only if ν = υ and 2) dT V = 1 if and only if ν and υ are singular, i.e. there
exists E such that ν(E) = 1 and υ(E) = 0. It is also well known that the concept of ϵ-contamination
is equivalent to the concept of total variation distance [8]; we separately state this property as the
following lemma.
Lemma 2.6 ([8]). Given a parameter 0 < ϵ < 1, a distribution X is an ϵ-contamination of D (and
vice versa) if and only if dT V (X, D) = ϵ.

3
Main Model: ϵ-delay Robustness

Denote the true reward delay distributions of each decision π ∈Π by Dπ. Every time πk is determined
at each round k, dk ∼Dπk is generated along with the generation of (rk, ok) ∼fπ. While ok is
observed immediately at round k, rk is scheduled to arrive at round dk + k. The order of reward
arrivals does not necessarily match the order of the reward generation.

As in [3], we assume the unknown delays setting throughout the paper. In the unknown delays setting,
the delay dk is not observed, and therefore attributing rewards to the previous decisions becomes a
nontrivial problem. One can only guess from which decision the reward just arrived came based on
the history of previous decisions and some information about reward delay distributions.

As reward delays are not observed, information we know about reward delay distributions is likely
to be misspecified. We model this misspecification as ϵ-contamination of the true delay distri-
bution models {Dπ}π∈Π resulting in information about { ˆDπ}π∈Π instead, where ˆDπ is an out-
come of ϵ-contamination of the delay distribution model Dπ, i.e., dT V (Dπ, ˆDπ) ≤ϵ. Note that
ϵ-contamination of delay distribution models encompasses many possible misspecification of informa-
tion about delay distribution. For example, it implies misspecification of mean of delay distribution
as dT V (Dπ, ˆDπ) > ϵ implies |E[D] −E[ ˆD]| > 0 (See discussions in Assumption 4.2 for details).

We now propose the formal definition of robustness in terms of delay distribution knowledge.
Definition 3.1. We say that a consistent algorithm is ϵ-delay robust if it is consistent when the given
delay distributions are ϵ-contaminations of the true delay distributions.

4
Main Results

4.1
Negative Result: Delay Robustness Requires C(f) = 0 for all f ∈F

Given the definition of ϵ-delay robustness provided in Section 3, the main question is when a
consistent, ϵ-delay robust algorithm exists. The answer is quite negative: unless C(f ⋆) = 0 holds, no
consistent algorithm can be ϵ-delay robust, however small ϵ > 0 is.
Theorem 4.1. Under minor technical assumptions (see Assumptions 4.2-4.5 below), regardless of
how small ϵ > 0 is, a consistent learning algorithm can be ϵ-delay robust only if C(f ⋆) = 0.

Theorem 4.1 implies that the concept of consistent algorithm fails even with a very small misspecifi-
cation of the delay model unless the Graves-Lai constant C(f ⋆) satisfies C(f ⋆) = 0. Since we want
to design a learning system with existence of a consistent algorithm that works for all instances of
f ∈F, we need C(f) = 0 for f ∈F, which was the necessary and sufficient condition for achieving
bounded regret when there were no reward delays (Corollary 2.4 and Theorem 2.5)

The intuition behind the proof of Theorem 4.1 (See Appendix B for details), is as follows. When the
reward delay model is precisely known, i.e., when the reward delay model is not contaminated, we
might be able to address this challenge by designing a good algorithm that makes the probability of
confusion in reward attribution as small as we want. However, in the case of ϵ-contamination of delay
models, under minor technical assumptions below (Assumptions 4.2-4.5), we can always provide a
delay model contamination that makes the precision of any consistent algorithm’s reward attribution
no better than 1 −δ for some δ > 0. This leads to reward distribution suffering δ-contamination,
which makes impossible to design a consistent algorithm.

The assumptions required for Theorem 4.1 are as follows.
Assumption 4.2. For the family of distributions Dr|o := {fπ(· | o) | f ∈F, π ∈Π, o ∈O}, there
exists a function q(δ) s.t. for D1, D2 ∈Dr|o, |E[D1] −E[D2]| ≤q(δ) implies dT V (D1, D2) ≤δ.

5


---Page Break---
For some special families of reward distributions, such q that satisfies Assumption 4.2 is known [8].
(Let k be a constant in what follows)

• For the family of Gaussian distributions with standard deviation 1, q(δ) = kδ.
• For the family of log-concave distributions with standard deviation 1, q(δ) = kδ log(1/δ).
• For the family of distributions with kth moment bounded by 1 for k ≥2, q(δ) = kδ1−1/k.
Assumption 4.3 expresses the conditional unimodality in likelihood functions in terms of rewards.
Assumption 4.3. Given ground truth f ∈F and g1, g2 ∈F, for every π ∈Π, Eg1π[r|o] ≤
Eg2π[r|o] ≤Efπ[r|o] or Eg1π[r|o] ≥Eg2π[r|o] ≥Efπ[r|o] implies DKL(fπ(· | o), g2
π(· | o)) ≤
DKL(fπ(· | o), g1
π(· | o)) almost everywhere (a.e.).

Assumptions 4.4 and 4.5 exclude trivial cases where the reward information is not at all needed for
the inference of the ground-truth model f.
Assumption 4.4 (Density of Fπ for every π ∈Π). For every π ∈Π, {gπ ∈Fπ | µgπ ≥µfπf } ∩
{gπ ∈Fπ | |Efπ(r | o) −Egπ(r | o)| ≤q(δ) a.e.} is nonempty given δ > 0.

Intuitively, {gπ ∈Fπ | µgπ ≥µfπf } is the set of hypotheses in Fπ we need to reject, and
{gπ ∈Fπ | |Efπ(r | o) −Egπ(r | o)| ≤q(δ) a.e.} is the set of hypothesis we cannot reject under
contamination of outcomes from decision π. Note that {gπ ∈Fπ | |Efπ(r | o) −Egπ(r | o)| ≤
q(δ) a.e.} ⊆{gπ ∈Fπ | |µgπ −µfπ| ≤q(δ)}.
Assumption 4.5. Let go
π be the marginal distribution of the observation of gπ ∈Fπ. There exists
ro > 0 such that for every π ∈Π, |µfπ −µgπ| ≤ro implies f o
π = go
π a.e..

Note that f o
π = go
π a.e. if and only if DKL(f o
π∥go
π) = 0 holds. If DKL(f o
π∥go
π) > 0, no information
on the rewards will be required to reject gπ under fπ, the true hypothesis for the decision π. On
the other hand, in the reinforcement learning problems where reward functions are parametrized
independent of the transition model parameters, r0 in the Assumption 4.5 is +∞.

4.2
Positive Result: C(f) = 0 for all f ∈F enables Super-Robust Bounded Regret

In the previous section, we saw in Theorem 4.1 that C(f) = 0 for f ∈F is required to assure the
existence of a delay-robust consistent algorithm. We also saw that C(f) = 0 for f ∈F is required to
assure sub-logarithmic regret for all possible instances of true f (Theorem 2.3).

Here, we try to answer when it is sufficient to assure best of both worlds, i.e., bounded regret and
robustness to any delay-model miss-specification at the same time. Before answering this question,
we need to define and explore two new concepts: cross-informativeness and max-contamination.

4.2.1
Cross-informativeness

Recall that we denote by Pf,n,A the distribution of the outcomes of algorithm A for the model
instance f ∈F by the nth round. Let us denote the algorithm that always chooses decision π ∈Π as
π. Then Lemma 4.6 motivates the concept of cross-informativeness.
Lemma 4.6. Suppose that f ∈F is the ground-truth model instance. Then C(f) = 0 implies that
DKL
 
Pf,n,πf ∥Pg,n,πf

= Ω(n) holds for g ∈F(f)c.

Proof. Suppose that C(f) = 0. According to equation (3), this implies that there exist wπf
and 0 ≤n0 < ∞such that for all n ≥n0, wπf DKL(fπf ∥gπf ) ≥1, ∀g ∈F(f)c. Therefore,
DKL
 
Pf,n,πf ∥Pg,n,πf

= nDKL(fπf ∥gπf ) ≥
1
wπf n holds ∀g ∈F(f)c for n ≥n0.

Lemma 4.6 provides an intuitive and straightforward connection between C(f) = 0 and bounded
regret: we can reject all the hypotheses that need a rejection to conclude that f is indeed the ground
truth hypothesis (F(f)c), simply by exclusively playing πf forever, while incurring zero regret.
Lemma 4.6 shows how informative πf is when the true hypothesis is f. When the true hypothesis is
not f, πf can be arbitrarily uninformative.

The natural question now arises is how much cross-informativeness (informativeness of πh ∈Π
for the ground truth f ∈F when h ̸= f) is sufficient for us to achieve bounded regret. The key

6


---Page Break---
assumption used in this paper is Assumption 4.7, which is later shown to be satisfied for the linear
systems (contextual linear bandits, linear MDP) when the conditions implied by C(f) = 0 for f ∈F
are satisfied (Section 5).

Assumption 4.7 (Cross-informativeness). Suppose that f ∈F is the ground-truth model. Then for
any g, h ∈F, DKL (Pf,n,πh∥Pg,n,πh) = ω(ln n) holds.

Note that the cross-informativeness lower bound rate of ω(ln n) in Assumption 4.7 is a much weaker
rate than the lower bound rate Ω(n) in Lemma 4.6.

4.2.2
Max-contamination

Whatever true delay distribution the reward delays follow, the maximum number of reward ar-
rivals from π by the round k is Nπ(k), the total number of π decisions by the round k. The max-

contamination of the decision π′ ∈Π at round k is defined as δmax
π′
(k) := min(

P

π∈Π\π′ Nπ(k)

e
N(k)
, 1),

where e
N(k) stands for the total number of reward arrivals by round k. Note that the contamination
of reward arrival at k is bounded by the max-contamination δmax
π′
(k), as the delay distributions of
decisions are stationary, i.e., they do not change over time.

4.2.3
Algorithm Simply-Test-To-Commit (ST2C)

Assumption 4.8. For all f, g ∈F, π ∈Π, and o ∈O,
ln fπ

gπ

 < c for some c > 0. This implies

DKL(fπ(· | o)∥gπ(· | o)) < ∞.

Assumption 4.8 excludes trivially informative cases where f and g are almost immediately dis-
tinguished given the observation o ∈O.
Under Assumption 4.8, we can well-define β :=
(supg∈Π,π∈Π,E∈Eπ
dfπ(·|o)

dgπ(·|o)(E))−1 (where Eπ denote the collection of measurable sets for fπ(· | o)

and gπ(· | o)), as Assumption 4.8 holds if and only if the log-likelihood ratio ln fπ(·|o)

gπ(·|o) is well-defined
on the support of gπ and is finite a.e..

Let P c
g,k,π indicate the likelihood of g ∈F that is computed as if all reward arrivals by the round
k are from the decision π. (Note that this is actually not true, as we allow decision transitions in
Algorithm 1.) Note that ln
P c
f,k,π
P c
g,k,π = Pn
k=1 ln f c
π(k)
gcπ(k), where f c
π(k) and gc
π(k) are likelihood of each
data assuming that the data is from π.

We now describe the algorithm Simply-Test-to-Commit (ST2C) as the Algorithm 1 below.

Algorithm 1 Simply-Test-to-Commit (ST2C) Algorithm

1: Choose any h ∈F, set bf = h
2: for n = 1, 2, . . . do
3:
Choose π b
f as the decision at period n
4:
Observe on and Rn, newly compute δmax
π b
f (n)

5:
if Fn := {g ∈F | Pn
k=1 ln
gc
π ˆ
f
(k)

ˆ
fcπ ˆ
f
(k) ≥2 ln n + Pn
k=1
2
√β δmax
π b
f (k)} ̸= ∅then

6:
Choose any g ∈Fn
7:
Set ˆf = g
8:
end if
9: end for

4.2.4
Analysis of algorithm ST2C

Later in Section 5, we will show that C(f) = 0 for all f ∈F, which is a design feature of a
learning problem we decide a priori, is sufficient to show that Assumption 4.7 indeed holds for
some representative linear problems. In this section, we show that Assumption 4.7 (combined with a
technical minor Assumption 4.8) is sufficient to allow Algorithm 1 to achieve bounded regret without
any knowledge of delay distribution model.

7


---Page Break---
The following Lemma 4.9 shows that ˆf stays at incorrect instances for only finite time, except for the
periods ˆf arrives at incorrect instances.

Lemma 4.9. Under Assumption 4.7 and 4.8, total number of periods ˆf stays in F \ f ⋆is finite in
expectation, except for the periods wrong transition (transition to F \ f ⋆) happens.

Lemma 4.10 shows that the total number of wrong transitions from the correct inferences is finite in
expectation.

Lemma 4.10. Under Assumption 4.8, the number of rounds Algorithm 1 satisfies the event { bf =

f ⋆}∩{∃g ∈F(f ⋆)c s.t. Pn
k=1 ln
gc
π ˆ
f (k)

ˆ
f c
π ˆ
f (k) ≥2 ln k+Pn
k=1
2
√β δmax
π b
f (k)} holds is finite in expectation.

Theorem 4.11. Under Assumption 4.7 and 4.8, the algorithm ST2C (Algorithm 1), which does not
require any knowledge of the delay distribution model, achieves a bounded regret ∆(1+5 4c4e−2

W(2c2)2 ) π2

6 ,
where W is the principal branch Lambert W function [27], ∆is the maximum per-period mean
reward difference among decisions, and c is from Assumption 4.8.

Proof. Combining Lemmas 4.9 and 4.10, we can conclude that bf /∈F(f ⋆) holds only for a finite
number of rounds in expectation. That is, regret is bounded in expectation. For detailed derivation of
the bound, see Appendix C.3.

5
Equivalence of bounded regret and delay robustness in linear systems

As discussed in Section 4.2, satisfying the cross-informativeness condition introduced in Assumption
4.7 is the key assumption that enables Algorithm 1 to achieve bounded regret with super-robustness
to delay. In this section, we show that linear learning problems such as contextual linear bandit and
reinforcement learning (RL) with linear MDP indeed satisfies the cross-informativeness condition if
C(f) = 0 for f ∈F. That is, for those problems, the condition ‘C(f) = 0 for f ∈F’ is not only
necessary (Section 4.1), but also sufficient for achieving bounded regret under any level of delay
model misspecification. In other words, we can conclude that achieving bounded regret is equivalent
to achieving any level of delay robustness for such linear problems discussed in Section 5.1 and 5.2.

5.1
Contextual linear bandit problem

Hao, Lattimore, and Szepesvari [10] was the first to characterize when C(f) = 0 holds for contextual
linear bandit problems. In this paper, we follow the notations and settings of [10] as follows: Let’s
consider the stochastic M-armed contextual linear bandit with a horizon of n rounds with M arms
and a finite A-size set of k-dimensional possible contexts X = {xj}j∈[A]. At each round, a context
is sampled according to the unknown distribution p over X and then observed. Every time a context
is sampled, an arm choice (a decision in MDSO framework) happens. When the sampled context
is xj and its chosen arm is m, we receive ϕm(xj)′θ + ϵ, where {ϕm : Rk 7→Rd}m∈[M] are linear
representation functions that are assumed to be precisely known, θ is a parameter vector of dimension
d that is shared across the arms, and ϵ is an i.i.d. random noise that follows a sub-Gaussian distribution
with variance proxy σ2.

Let Θ be the set of all parameter vectors, and let θ⋆∈Θ be the unknown true parameter. Suppose
that C(θ) = 0 for θ ∈Θ. Let mjθ be an optimal arm for context j ∈[A] when the true parameter is
θ, i.e., mjθ ∈argmaxm∈[M] ϕm(xj)′θ. The following Theorem 5.1 characterizes previous results
on when bounded regret can be achieved when there are no reward delays.

Theorem 5.1 ([10, 28]). Given linear contextual bandit setting described above, when there
are no reward delays, bounded regret can be a priori guaranteed to be achieved if and only if

ϕmjθ(xj) | j ∈A
	
spans Rd for all θ ∈Θ.

Note that the condition that ‘

ϕmjθ(xj) | j ∈A
	
spans Rd for all θ ∈Θ’ in Theorem 5.1 is easily
satisfied a priori when we are given rich enough context set [29]. How does this easily satisfied
condition work when there are anonymous delayed rewards? The following theorem of ours shows
that ‘

ϕmjθ(xj) | j ∈A
	
spans Rd for all θ ∈Θ’ implies that the Assumption 4.7 is satisfied.

8


---Page Break---
Theorem 5.2. Given contextual linear bandit setting described above,
Assumption 4.7
(DKL (Pθ⋆,n,πθ∥Pθ′,n,πθ) = Ω(n) holds for θ′ ∈F(θ⋆)c) is satisfied if

ϕmjθ(xj) | j ∈A
	
spans
Rd for all θ ∈Θ.

See Appendix D for the proof of Theorem 5.2. As Assumption 4.8 trivially holds for linear problems
[2], Theorem 4.11 (which requires Assumption 4.7 and 4.8 to hold) conclude that our Algorithm 1
achieves bounded regret without any knowledge of delay distribution model if

ϕmjθ(xj) | j ∈A
	

spans Rd for all θ ∈Θ (which is easily satisfied when the context space is rich enough [29]).

For contextual linear bandit setting described above, it has been shown that ’

ϕmjθ(xj) | j ∈A
	

spans Rd for all θ ∈Θ’ is equivalent to ‘C(θ) = 0 for all θ ∈Θ’ [26, 2, 1]. Since C(θ) = 0 for all
θ ∈Θ is a necessary condition for a-priori assurance of any-level robustness of a consistent algorithm
(Theorem 4.1), we have the following Corollary 5.3, which is a reminiscent of Theorem 5.1 above.
Corollary 5.3. Given contextual linear bandit setting described above, under any anonymous delayed
rewards, bounded regret can be a priori guaranteed to be achieved if and only if

ϕmjθ(xj) | j ∈A
	

spans Rd for all θ ∈Θ.

Note that Corollary 5.3 strongly motivates the practical usefulness of bounded regret algorithm design
in contextual linear bandit problems, as it is also a necessary and sufficient condition for achieving
any level of delay robustness. This condition is indeed not hard to satisfy in the real world; for
example, in Spotify, million daily users can be considered a rich enough context for exploring 60,000
new songs uploaded daily [29].

5.2
Reinforcement learning with Linear MDP

Papini et al. [11] was the first to characterize the condition for achieving bounded regret for some
popular classes of reinforcement learning with episodic Linear Markov Decision Process (MDP). In
this paper, we follow the notations of [11], which are as follows: we are given a time-inhomogenous
MDP M =

S, A, H, {rh}H
h=1 , P, µ

, where S is finite state space, A is a finite action space, H
is the length of each episode, {rh} are the reward functions where rh(s, a) the expected reward of
a pair (s, a) ∈S × A at time-step h, P := {ph} are the transition kernels, and µ is the initial state
distribution. A policy π = (π1, . . . , πH) ∈Π is a sequence of per-time-step policies πh : S →A.
For every h ∈[H] := {1, . . . , H}, we define the state-action value function of a policy π as

Qπ
h(s, a) = rh(s, a) + Eπ
hPH
i=h+1 ri (si, ai)
i
and Qπ⋆
h (s, a) := Q⋆
h(s, a) = supπ Qπ
h(s, a) =

LhQ⋆
h+1(s, a) where LhQ⋆
h+1(s, a) := rh(s, a) + Es′∼ph(s,a)

maxa′ Q⋆
h+1 (s′, a′)

.

As in [11], we focus on episodic Linear MDP setting with Bellman closure [30], which is more
general than many popular Linear MDP settings such as low-rank Linear MDPs [30].
Definition 5.4 (Linear MDP with Bellman closure (completeness)[30]). Suppose that we are given a
feature map ϕh : S × A →Rdh, possibly different at any h ∈[H], mapping state-action pair (s, a)
into a dh-dimensional vector ϕh(s, a). For the set of bounded value function Qh = {Qh | θh ∈
Θh : Qh(s, a) = ϕh(s, a)⊤θh, ∀(s, a)
	
and the associated parameter space Θh =

θh ∈Rd :
ϕh(s, a)⊤θh
 ≤D
	
. An MDP is said to satisfy zero Inherent Bellman Error (IBE) (or satisfy
Bellman closure) if ∀h ∈[H], supQh+1∈Qh+1 infQh∈Qh ∥Qh −LhQh+1∥∞= 0 holds.

The following Theorem 5.5 summarizes a previous result that characterizes when bounded regret can
be achieved when there are no reward delays.
Theorem 5.5 (Papini et al. [11]). Denote the optimal policy as π⋆and ϕ⋆
h(s) := ϕh (s, π⋆
h(s)). In
Linear MDPs satisfying Bellman closure, the condition that ‘span{ϕ⋆
h(s) | ∀s ∈S, π⋆visits s at h
with positive probability} = Rd for all h ∈[H]’ is sufficient for achieving bounded regret in high
probability when there is no unknown reward delay.
Theorem 5.6. Given episodic Linear MDP with Bellman closure described above, Assumption 4.7
(DKL (Pθ⋆,n,πθ∥Pθ′,n,πθ) = Ω(n) holds for θ′ ∈F(θ⋆)c) is satisfied if span{ϕ⋆
h(s) | ∀s ∈S, π⋆

visits s at h with positive probability} = Rd for all h ∈[H].

Again, as Assumption 4.8 trivially holds for linear problems [2], Theorem 4.11 (which requires
Assumption 4.7 and 4.8 to hold) conclude that our Algorithm 1 achieves bounded regret without

9


---Page Break---
any knowledge of delay distribution model if span{ϕ⋆
h(s) | ∀s ∈S, π⋆visits s at h with positive
probability} = Rd for all h ∈[H]. Therefore, we have the following Corollary 5.7.
Corollary 5.7. Given episodic Linear MDP with Bellman closure setting described above, under
any anonymous delayed rewards, bounded regret can be a priori guaranteed to be achieved if
span{ϕ⋆
h(s) | ∀s ∈S, π⋆visits s at h with positive probability} = Rd for all h ∈[H].

Again, note that Corollary 5.7 strongly motivates the practical usefulness of bounded regret algorithm
design, as the sufficient condition for it is also sufficient for achieving any level of delay robustness.

6
Conclusions

In this paper, we characterize the link between consistent algorithms and delay robustness in inter-
active decision-making. The first main result states that, for consistent algorithms, the necessary
condition for achieving any (small) level of robustness against delay model misspecifications is also
sufficient for achieving bounded regret. Viewed from another perspective, this result urges us to
revisit the practicality of the instance-dependent regret minimizing algorithm design regime [9, 1] for
real-world problems with anonymous delayed rewards. The second main result states vice versa for
linear problems, showing that the well-known necessary (and sufficient) condition for bounded regret
is also sufficient for designing a consistent algorithm that achieves any (large) level of robustness
against delay model misspecifications and bounded regret at the same time. An interesting future
research direction raised by our paper is whether it is possible to achieve our second main result
without restricting to linear problems.

Acknowledgements

This material is based upon work partially supported by the US Army Contracting Command under
W911NF-22-1- 0151 and USARO under W911NF2120064, the US National Science Foundation
under CNS-2328395 and CMMI-2038625, and the US Office of Naval Research under N00014-
24-1-2615 and N00014-21-1-2385. This work was also partially conducted with support from the
Bertauche Transportation Endowment and the Edna Benson PhD Fellowship.

References

[1]
Andrew Wagenmaker and Dylan J Foster. “Instance-Optimality in Interactive Decision Making:
Toward a Non-Asymptotic Theory”. In: arXiv preprint arXiv:2304.12466 (2023).
[2]
Kefan Dong and Tengyu Ma. Asymptotic Instance-Optimal Algorithms for Interactive Decision Making.
2023. arXiv: 2206.02326 [cs.LG].
[3]
Bingcong Li, Tianyi Chen, and Georgios B Giannakis. “Bandit online learning with unknown
delays”.
In:
The 22nd International Conference on Artificial Intelligence and Statistics.
PMLR. 2019, pp. 993–1002.
[4]
Ciara Pike-Burke et al. “Bandits with delayed, aggregated anonymous feedback”. In:
International Conference on Machine Learning. PMLR. 2018, pp. 4105–4113.
[5]
Nicolo Cesa-Bianchi, Claudio Gentile, and Yishay Mansour. “Nonstochastic bandits with
composite anonymous feedback”. In: Conference On Learning Theory. PMLR. 2018, pp. 750–
773.
[6]
Siwei
Wang,
Haoyun
Wang,
and
Longbo
Huang.
“Adaptive
algorithms
for
multi-armed
bandit
with
composite
and
anonymous
feedback”.
In:
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. 11. 2021, pp. 10210–
10217.
[7]
Peter J Huber. Robust statistics. Vol. 523. John Wiley & Sons, 2004.
[8]
Ilias Diakonikolas and Daniel M Kane. Algorithmic high-dimensional robust statistics. Cam-
bridge University Press, 2023.
[9]
Todd L Graves and Tze Leung Lai. “Asymptotically efficient adaptive choice of control
laws incontrolled markov chains”. In: SIAM journal on control and optimization 35.3 (1997),
pp. 715–743.

10


---Page Break---
[10]
Botao Hao, Tor Lattimore, and Csaba Szepesvari. “Adaptive exploration in linear contextual
bandit”. In: International Conference on Artificial Intelligence and Statistics. PMLR. 2020,
pp. 3536–3545.
[11]
Matteo Papini et al. “Reinforcement learning in linear mdps: Constant regret and representation
selection”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 16371–
16383.
[12]
Gergely Neu et al. “Online Markov decision processes under bandit feedback”. In:
Advances in Neural Information Processing Systems 23 (2010).
[13]
Pooria
Joulani,
Andras
Gyorgy,
and
Csaba
Szepesvári.
“Delay-tolerant
online
convex
optimization:
Unified
analysis
and
adaptive-gradient
algorithms”.
In:
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. 1. 2016.
[14]
Nicolo Cesa-Bianchi, Claudio Gentile, and Yishay Mansour. “Delay and cooperation in
nonstochastic bandits”. In: Journal of Machine Learning Research 20.17 (2019), pp. 1–38.
[15]
Claire Vernade, Olivier Cappé, and Vianney Perchet. “Stochastic bandit models for delayed
conversions”. In: arXiv preprint arXiv:1706.09186 (2017).
[16]
Tal Lancewicki et al. “Stochastic multi-armed bandits with unrestricted delay distributions”.
In: International Conference on Machine Learning. PMLR. 2021, pp. 5969–5978.
[17]
Benjamin Howson, Ciara Pike-Burke, and Sarah Filippi. “Optimism and delays in episodic
reinforcement learning”. In: International Conference on Artificial Intelligence and Statistics.
PMLR. 2023, pp. 6061–6094.
[18]
Nikki Lijing Kuang et al. “Posterior sampling with delayed feedback for reinforcement learning
with linear function approximation”. In: Advances in Neural Information Processing Systems
36 (2023), pp. 6782–6824.
[19]
Tobias Sommer Thune, Nicolò Cesa-Bianchi, and Yevgeny Seldin. “Nonstochastic multiarmed
bandits with unrestricted delays”. In: Advances in Neural Information Processing Systems 32
(2019).
[20]
Han Wu and Stefan Wager. “Thompson sampling with unrestricted delays”. In:
Proceedings of the 23rd ACM Conference on Economics and Computation. 2022, pp. 937–
955.
[21]
Tiancheng Jin et al. “Near-optimal regret for adversarial mdp with delayed bandit feedback”.
In: Advances in Neural Information Processing Systems 35 (2022), pp. 33469–33481.
[22]
Julian Zimmert and Yevgeny Seldin. “An optimal algorithm for adversarial bandits with
arbitrary delays”. In: International Conference on Artificial Intelligence and Statistics. PMLR.
2020, pp. 3285–3294.
[23]
Saeed Masoudian, Julian Zimmert, and Yevgeny Seldin. “A Best-of-both-worlds Al-
gorithm for Bandits with Delayed Feedback with Robustness to Excessive Delays”. In:
ICML 2024 Workshop: Foundations of Reinforcement Learning and Control–Connections and Perspectives.
2024.
[24]
Alekh Agarwal et al. “Contextual bandit learning with predictable rewards”. In:
Artificial Intelligence and Statistics. PMLR. 2012, pp. 19–26.
[25]
Simon Du et al. “Bilinear classes: A structural framework for provable generalization in rl”.
In: International Conference on Machine Learning. PMLR. 2021, pp. 2826–2836.
[26]
Dylan J Foster et al. “The statistical complexity of interactive decision making”. In:
arXiv preprint arXiv:2112.13487 (2021).
[27]
Robert
M
Corless
et
al.
“On
the
Lambert
W
function”.
In:
Advances in Computational mathematics 5 (1996), pp. 329–359.
[28]
Matteo Papini et al. “Leveraging good representations in linear contextual bandits”. In:
International Conference on Machine Learning. PMLR. 2021, pp. 8371–8380.
[29]
Enoch
Hyunwook
Kang
and
PR
Kumar.
“Bounded
(o
(1))
re-
gret
recommendation
learning
via
synthetic
controls
oracle”.
In:
2023 59th Annual Allerton Conference on Communication, Control, and Computing (Allerton).
IEEE. 2023, pp. 1–7.
[30]
Andrea Zanette et al. “Learning near optimal policies with low inherent bellman error”. In:
International Conference on Machine Learning. PMLR. 2020, pp. 10978–10989.
[31]
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.

11


---Page Break---
[32]
Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999.
[33]
Sergio Verdú. “Total variation distance and the distribution of relative information”. In:
2014 Information Theory and Applications Workshop (ITA). IEEE. 2014, pp. 1–3.

12


---Page Break---
A
Special Cases of DMSO Framework

- In finite-armed bandit problems, each round is an arm pull. Π is the arm space, and R is
the space of rewards from arms. Since there is no observation space O, the model class
degenerates to F ⊆(Π 7→∆R).
- In contextual bandit problems, each round is an arm pull. Π is the set (X 7→A) of all
policies, where X is the context space and the A is the arm space. The reward space R
is the space of rewards from arms. The observation space O is X, where the kth round’s
observation ok ∈O (which results from πk) is the k + 1th round’s context. Since the
future context arrival is not affected by previous decisions, the model class degenerates to
F ⊆(Π 7→∆R).
- In episodic reinforcement learning problems, each round is an episode. Π is the set (S 7→A)
of all policies, where S is the space of all possible states and A is the action space. The
reward space R is the space of value functions at each initial state, and the observation
space O is the set of all possible sequences of action choices, state transitions, and received
rewards in one episode. The model class F is characterized jointly by the initial state
distribution and the transition kernel, which are shared across all the episodes.

B
Proof of Theorem 4.1

Recall that Pf,n,A[·] denotes the distribution of outcomes of algorithm A on the true model instance
f by the round n. We further denote the marginal distribution of Pf,n,A[·] in terms of decision π’s
rewards and outcomes by P π
f,n,A[·].

Lemma B.1. Suppose that the ground-truth model is f ∈F. Then a consistent algorithm must
satisfy (1 + o(1)) ln n ≤P

π∈Π DKL

P π
f,n,A∥P π
g,n,A

for g ∈F(f)c.

Proof.
According to Dong and Ma (2022) [2], any consistent algorithm A must satisfy (1 +
o(1)) ln n ≤DKL (Pf,n,A∥Pg,n,A) for g ∈F(f)c. Since the terms involving A (the algorithm
used to collect the data) cancel out and the outcomes of decisions are independent of each other,
Pf,n,A
Pg,n,A = Q

π∈Π
P π
f,n,A
P π
g,n,A holds. Therefore, the condition of Dong and Ma (2022) becomes (1 +

o(1)) ln n ≤P
π∈Π DKL

P π
f,n,A∥P π
g,n,A

for g ∈F(f)c.

Lemma B.2. For any ϵ > 0, ϵ-contamination in the delay model of π⋆makes the rewards of decisions
π ̸= π⋆suffer δ-contamination for some δ > 0 under consistency.

See Section B.1 for the proof of Lemma B.2.

Lemma B.3 shows that, under δ-reward contaminations in reward distributions of all π ̸= π⋆,
choosing optimal decision π⋆alone must be enough to satisfy the condition described in Lemma B.1
and otherwise, we cannot satisfy it.
Lemma B.3. If the rewards of decisions π ∈Π \ πf suffer δ-contamination for some δ > 0,

consistency of the algorithm requires (1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

for all g ∈F(f)c.

See Section B.2 for the proof of Lemma B.3.

The rest of the proof of Theorem 4.1 is immediate from the derivation of [2]’s Theorem 2.3, which is
as follows: from the chain rule of divergence, DKL

P π
f,n∥P π
g,n

= Ef,n [Nπ] DKL(fπ∥gπ) holds
for π ∈Π. Defining wπ := Ef [Nπ] /((1 + o(1)) ln n), Lemma B.3 implies that C(f) = 0 by the
definition of C(f) in the equation (3). Since we don’t know the ground truth f a priori, designing a
learning system that assures the existence of a robust algorithm requires C(f) = 0 for all f ∈F.

B.1
Proof of Lemma B.2

Denote by N [a,b]
π
the random variable that counts the number of decisions of π between rounds a and
b. For consistency, for any small enough p > 0, for any r > 0, for some m, there must exist a constant

13


---Page Break---
nr,p such that for all intervals [a, b] with b −a ≥nr,p and a, b > m, E[N [a,b]
π⋆] ≥(b −a) −(b −a)pr
holds. Recall that we denote by Dπ the true delay model for the decision π ∈Π, and by ˆDπ the
given model for Dπ. Note that ϵ-contamination means we can arbitrarily choose {Dπ}π∈Π as long as
dT V (Dπ, ˆDπ) ≤ϵ. Consider the case when Dπ⋆=
ˆ
Dπ⋆+ ϵDa −ϵDc where P(Da = k) =
1
nr,p
for 0 ≤k ≤nr,p −1 and 0 for elsewhere, and Dc is an arbitrary distribution. For π ∈Π \ π⋆,
consider Dπ = ˆ
Dπ. Then for k ≥max(nr,p, m),

P({A reward arrival at k is not from π⋆})

=
Pk
i=1 P({πi’s reward arrives at k and πi ̸= π⋆})

Pk
i=1 P({πi’s reward arrives at k})

≤
|Π| −1

|Π| −1 + Pk
i=1 P({πi’s reward arrives at k and πi = π⋆})

(4)

≤
|Π| −1

|Π| −1 +
ϵ
nr,p
Pk
i=k−nr,p E[1πi=π⋆]

≤
|Π| −1
|Π| −1 +
ϵ
nr,p (nr,p −np
r,pr) =
|Π| −1
|Π| −1 + ϵ(1 −np−1
r,p r)

where the inequality in the equation (4) follows from the fact that the delay distribution of each

decision sums to one. Therefore, P({A reward arrival at k is from π⋆}) > δ :=
ϵ(1−np−1
r,p r)

|Π|−1+ϵ(1−np−1
r,p r).

Denote the rewards distributions associated with Dπ⋆,
ˆ
Dπ⋆, Da, and Dc as Rπ⋆, ˆ
Rπ⋆, Ra, and Rc
each. Then Rπ⋆=
ˆ
Rπ⋆+ ϵRa −ϵRc must hold, where the mean of Rπ⋆and Rπ⋆are supposed to
be the same. Since the choice of Ra can be arbitrary by choosing Rc accordingly, we can conclude
that the reward distributions of decisions π ̸= π⋆indeed suffer δ-contamination.

B.2
Proof of Lemma B.3

Recall that we use fπ to refer to an element of Fπ, while fπ also denotes the π-coordinate of some
f ∈F. Suppose that the ground-truth model is f ∈F. For g ∈F(f)c, a consistent algorithm must
satisfy

(1 + o(1)) ln n ≤DKL (Pf,n∥Pg,n) =
X

π∈Π
DKL
 
P π
f,n∥P π
g,n

.
(5)

= DKL

P πf
f,n∥P πf
g,n

+
X

π∈Π\πf
Ef,n [Nπ] DKL(fπ(r, o)∥gπ(r, o))
(6)

= DKL

P πf
f,n∥P πf
g,n

+
X

π∈Π\πf
Ef,n [Nπ]

DKL(f o
π(o)∥go
π(o)) + Efπ


log fπ(r|o)

gπ(r|o)



(7)

Above,

• The inequality in the equation (5) is from Dong and Ma (2022) [2].

• The equality in the equation (5) follows from the fact that algorithm-related terms cancel
out.

• The inequality in the equation (6) is from the Divergence decomposition Lemma [31]

• The inequality in the equation (7) follows from the chain rule of KL divergence [32].

Let min(q(δ), ro) = q′(δ), where ro is defined as in Assumption 4.5. By Assumption 4.4, for every
π ̸= πf, there exists a non-empty set Eπ := {lπ ∈Fπ | |Efπ[r|o]−Elπ[r|o]| ≤q′(δ) a.e.}∩{µlπ ≥
µfπf }. Hense we can construct E(π) := {l ∈F | lπ ∈Eπ, lπ′ = fπ′ for π′ ̸= π}. Note that
E(π) ⊆F(f)c := {g ∈F | πg ̸= πf} = {g ∈F | ∃π ∈Π s.t. µgπ ≥µfπf }. Therefore, for every

14


---Page Break---
π ̸= πf,

(1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

+
X

π∈Π\πf
Ef,n [Nπ(n)] Efπ


log fπ(r|o)

gπ(r|o)


for g ∈E(π)
(8)

(⇒) (1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

+ Ef,n [Nπ(n)] Efπ


log fπ(r|o)

gπ(r|o)


for g ∈E(π) (9)

(⇒) (1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

for g ∈E(π)
(10)

(⇒) (1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

for g ∈E′(π)
(11)

(⇒) (1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

for g ∈{g ∈F | µgπ ≥µfπf }.
(12)

where E′
π := {lπ ∈Fπ | µfπf ≤µlπ} and E′(π) := {l ∈F | lπ ∈E′
π, lπ′ = fπ′ for π′ ̸= π}.
Above,

• Equation (8) follows from Assumption 4.5 and equation (7).
• The logical implication in equation (9) follows from the definition of E(π).
• The logical implication in equation (10) follows from Assumptions 4.2, 4.3 and 4.4:
|Efπ[r|o] −Egπ[r|o]| ≤q′(δ) ≤q(δ) a.e. implies dT V (fπ(· | o), gπ(· | o)) < δ a.e.
from Assumption 4.2; therefore, under some δ-contamination of fπ(· | o), the con-
taminated Efπ[r|o] can be farther from the true Efπ[r|o] than Egπ[r|o].
Therefore,

DKL (fπ(· | o) ∥gπ(· | o)) ≤0 a.e. due to Assumption 4.3, and so Efπ
h
log fπ(r|o)

gπ(r|o)
i
=

Ef o
π
h
Efπ(r|o)
h
log fπ(r|o)

gπ(r|o)
ii
= Ef o
π [DKL (fπ(· | o) ∥gπ(· | o))] ≤0.

• The logical implication in equation (11) follows from Assumption 4.3:
Define ˆEπ := {|µfπ −µgπ| ≤q′(δ)}∩{µgπ ≥µfµf } and ˆE(π) := {l ∈F | lπ ∈ˆEπ, lπ′ =

fπ′ for π′ ̸= π}. Note that ˆE(π) ⊆E(π). Then for g′ ∈E′(π) \ ˆE(π) and g ∈ˆE(π)
with (µgπ −µfπ)(µg′π −µfπ) ≥0, DKL

P π
f,n∥P π
g,n

≤DKL

P π
f,n∥P π
g′,n

due to the
monotonicity assumption of Assumption 4.3.
• The logical implication in equation (12) follows from the fact that any element in {g ∈F |
µgπ ≥µfπf } has an element in E′ that is strictly closer to f.

Since F(f)c := {g ∈F | πg ̸= πf} = {g ∈F | ∃π ∈Π s.t. µgπ ≥µfπf }, we immediately get

(1 + o(1)) ln n ≤DKL

P πf
f,n∥P πf
g,n

for g ∈F(f)c.

15


---Page Break---
C
Proof of Lemma 4.9, Lemma 4.10 and Theorem 4.11

C.1
Proof of Lemma 4.9

Lemma C.1 (Upper bounding lemma). k =
4c4e−2

W(2c2)2 satisfies e−n(ln n)2

2c2
≤k 1

n2 for all n ≥1, where
W is the principal branch Lambert W function [27].

Proof.

y = x2 · e−x(ln x)2

2c2

= x2 · e−f(x) (by setting f(x) = x(ln x)2

2c2
dy
dx = d

dx
 
x2
· e−f(x) + x2 · d
dx


e−f(x)

= 2x · e−f(x) + x2 · e−f(x)

−
(ln x)2

2c2
+ ln x

c2



(∵d

dx(e−f(x)) = e−f(x) · (−f ′(x)) and

f ′(x) = d

dx

x(ln x)2

2c2


=
1
2c2


(ln x)2 + 2x ln x · 1

x


= (ln x)2

2c2
+ ln x

c2 )

= 2xe−f(x) −x2e−f(x)
(ln x)2

2c2
+ ln x

c2



= e−f(x)

2x −x2
(ln x)2

2c2
+ ln x

c2



Set the derivative to zero to find the critical points:

2x −x2
(ln x)2

2c2
+ ln x

c2


= 0

2 = x
(ln x)2

2c2
+ ln x

c2


if x > 0

4c2 = x(ln x)2 + 2x ln x

This is the uni-modal function with a maximum, as the derivatives are positive on the left side of the
critical point and negative on the right side of the critical point. Also, note that xmax ln xmax ≤2c2,
as x(ln x)2 ≥0. This implies that xmax ≤
2c2
W(2c2), where W denote the principal branch Lambert
W function. Now note that

ymax = x2
max · e−xmax(ln xmax)2

2c2
(13)

= x2
max · e−(2+ xmax

c2
ln xmax)
(14)

≤e−2 · x2
max
(15)

≤e−2

2c2

W (2c2)

2
=
4c4e−2

W (2c2)2
(16)

Therefore, k =
4c4e−2

W(2c2)2 satisfies e−n(ln n)2

2c2
≤k 1

n2 for all n ≥1.

16


---Page Break---
Proof of Lemma 4.9. Let f be the ground truth model. After each of ˆf’s transition to g ∈F(f)c, at
period n,

P({

n
X

k=1
ln
f c
πg(k)

gcπg(k) ≤2 ln n +

n
X

k=1

2
√β δmax
πg (k)})

= P({

n
X

k=1
(ln fπg

gπg
+ ln
f c
πg(k)

fπg
+ ln
gπg
gcπg(k)) ≤2 ln n +

n
X

k=1

2
√β δmax
πg (k)})

≤P({

n
X

k=1
(ln fπg

gπg
+ ln
f c
πg(k)

fπg
+ ln
gπg
gcπg(k)) ≤2 ln n + 2C
√β ln n}) for some C
(17)

≤P({

n
X

k=1


ln fπg

gπg
−DKL(fπg, gπg)

+

n
X

k=1

 

ln
f c
πg(k)

fπg

!

+

n
X

k=1

 

ln
gπg
gcπg(k)

!

≤−ln n}) for n ≥n0 for some n0 < ∞
(18)

≤P({

n
X

k=1


ln fπg

gπg
−DKL(fπg, gπg)

+

n
X

k=1

 

ln
f c
πg(k)

fπg
−DKL(fπg, f c
πg(k))

!

+

n
X

k=1

 

ln
gπg
gcπg(k) −DKL(gπg, gc
πg(k))

!

≤−ln n}) for n ≥n0
(19)

≤P({

n
X

k=1


ln fπg

gπg
−DKL(fπg, gπg)

≥−ln n,

n
X

k=1

 

ln
f c
πg(k)

fπg
−DKL(fπg, f c
πg(k))

!

≥−ln n,

n
X

k=1

 

ln
gπg
gcπg(k) −DKL(gπg, gc
πg(k))

!

≥−ln n}c) for n ≥n0

= P({

n
X

k=1


ln fπg

gπg
−DKL(fπg, gπg)

≤−ln n} ∪{

n
X

k=1

 

ln
f c
πg(k)

fπg
−DKL(fπg, f c
πg(k))

!

≤−ln n} ∪{

n
X

k=1

 

ln
gπg
gcπg(k) −DKL(gπg, gc
πg(k))

!

≤−ln n}) for n ≥n0

≤3e−n(ln n)2

2c2
(20)

≤12c4e−2

W (2c2)2
1
n2
(21)

Above,

• Equation 17 follows from the fact that δmax
πg (k) decreases with the rate 1/n.

• Equation 18 follows from the fact that nDKL(fπg, gπg) = DKL(Pf,n,πg∥Pg,n,πg) =
ω(ln n) from the Assumption 4.7.

• Equation 19 follows from the fact that substracting positive value on the left does not change
the inequality.

• Equation 20 follows from the fact that the log-likelihood ratios are bounded by constant c
due to Assumption 4.8, and thus sub-gaussian random variables with σ2 = c2

4 .

• Equation 21 is from Lemma C.1.

Therefore, after each time a bad transition to g ∈F(f)c happens, the event {Pn
k=1 ln
f c
πg (k)

gc
πg (k) ≤

2 ln n+Pn
k=1
2
√β δmax
πg (k)} happens only finite many times (more precisely, smaller than 3· 4c4e−2

W(2c2)2 ·

π2

6 = 2c4e−2π2

W(2c2)2 ) in expectation by the Borel-Cantelli lemma, which implies that the inference will
arrive at the correct instance within finite expected time.

17


---Page Break---
C.2
Proof of Lemma 4.10

Proof of Lemma 4.10. Suppose that f is the ground truth model. For any g ∈F \ f ⋆, when ˆf = f,

P({

n
X

k=1
ln
gc
πf (k)

f cπf (k) ≥2 ln n +

n
X

k=1

2
√β δmax
πf (k)})

= P({

n
X

k=1
(ln gπf

fπf
+ ln
fπf
f cπf (k) + ln
gc
πf (k)

gπf
) ≥2 ln n +

n
X

k=1

2
√β δmax
πf (k)})

≤P({

n
X

k=1


ln gπf

fπf


+

n
X

k=1

 

ln
fπf
f cπf (k) −DKL(fπf , f c
πf (k))

!

+

n
X

k=1

 

ln
gc
πf (k)

gπf
−DKL(gc
πf (k), gπf )

!

≥2 ln n})
(22)

≤P({

n
X

k=1


ln gπf

fπf


≤2 ln n,

n
X

k=1

 

ln
fπf
f cπf (k) −DKL(fπf , f c
πf (k))

!

≤2 ln n,

,

n
X

k=1

 

ln
gc
πf (k)

gπf
−DKL(gc
πf (k), gπf )

!

≤2 ln n}c)

≤P({

n
X

k=1


ln gπf

fπf


≥2 ln n} ∪{

n
X

k=1

 

ln
fπf
f cπf (k) −DKL(fπf , f c
πf (k))

!

≥2 ln n}

, ∪{

n
X

k=1

 

ln
gc
πf (k)

gπf
−DKL(gc
πf (k), gπf )

!

≥2 ln n})

≤1

n2 + 2 4c4e−2

W (2c2)2
1
n2
(23)

Above,

• Equation 22 follows from the reverse Pinsker’s inequality [33] (total variation distance
smaller than δ implies KL divergence smaller than
1
√β δ)

• Equation 23 follows from Lemma 4.3 of Dong and Ma (2022) [2], which says that
PQ
nPm
i=1 ln Pi

Qi ≥c
o
≤exp(−c), and the fact that the log-likelihood ratios are
bounded due to Assumption 4.8, and thus sub-gaussian random variables.

Therefore, the event holds in total only for finite rounds of k in expectation (more precisely, bounded
by (1 +
8c4e−2

W(2c2)2 ) π2

6 ) by the Borel-Cantelli lemma.

C.3
Proof of Theorem 4.11

Combining Lemmas 4.9 and 4.10, we can conclude that bf /∈F(f ⋆) holds only for a finite number of
rounds in expectation. That is, regret is bounded in expectation with value ∆(1 + 5 4c4e−2

W(2c2)2 ) π2

6 .

18


---Page Break---
D
Proof of Theorem 5.2 (Linear contextual bandit case)

Let Θ be the set of all parameters, and let θ⋆∈Θ be the unknown true parameter. Suppose that
C(θ) = 0 for θ ∈Θ. By Theorem 2.3 and Theorem 5.1,

ϕmjθ(xj) | j ∈A
	
spans Rd for θ ∈Θ.
Denote Tx(n) be the number of arrivals of context x ∈X.

Then for any ˜θ ∈Θ \ {θ⋆},

DKL

Pθ⋆,n,πθ∥P˜θ,n,πθ


= 1

2

X

x∈A
E [Tx(n)] ⟨x, θ⋆−eθ⟩2
(24)

= 1

2(θ⋆−eθ)⊤E

"X

x∈A
Tx(n)xx⊤
#

(θ⋆−eθ)

= 1

2(θ⋆−eθ)⊤nE

"X

x∈A

Tx(n)

n
xx⊤
#

(θ⋆−eθ)

≥1

2∥θ⋆−eθ∥2nλmin
(25)

= Ω(n)
(26)

Above,

• The equality in equation (24) is from the divergence decomposition lemma Lattimore and
Szepesvári [31]
• λmin of equation (25) denotes the smallest eigenvalue for Exj∼p

ϕmjθ(xj)ϕmjθ(xj)⊤

• The inequality of equation (25) is from the fact that xT Ax
xT x is larger than the smallest
eigenvalue of A.
• The equality of equation (26) comes from the fact that λmin > 0 is equivalent to

ϕmjθ(xj) | j ∈A
	
spanning Rd [28].

E
Proof of Theorem 5.6 (Reinforcement learning with Linear MDP case)

It is straightforward that the proof of Theorem 5.6 is almost equivalent to the proof of Theorem 5.2,
except that the problem here is inferring θh separately for each h ∈[H].

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope?
Answer: [Yes]
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?
Answer: [Yes]
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA]
Justification: Theory paper.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions
to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [NA]
Justification: Theory paper.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparame-
ters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [NA]
Justification: Theory paper.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: Theory paper.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the
experiments?
Answer: [NA]
Justification: Theory paper.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?
Answer: [NA]
Justification: Learning theory paper that is not bound to practical implications.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release
of data or models that have a high risk for misuse (e.g., pretrained language models, image
generators, or scraped datasets)?
Answer: [NA]
Justification: Theory paper.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the
paper, properly credited and are the license and terms of use explicitly mentioned and properly
respected?
Answer: [NA]
Justification: Theory paper.
13. New Assets

20


---Page Break---
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: Theory paper.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justification: Theory paper.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or
an equivalent approval/review based on the requirements of your country or institution) were
obtained?
Answer: [NA]
Justification: Theory paper.

21


---Page Break---
