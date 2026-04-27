Almost Minimax Optimal Best Arm Identification in
Piecewise Stationary Linear Bandits

Yunlong Hou
Department of Mathematics
National University of Singapore
yhou@u.nus.edu

Vincent Y. F. Tan
Department of Mathematics
Department of Electrical and Computer Engineering
National University of Singapore
vtan@nus.edu.sg

Zixin Zhong
Data Science and Analytics Thrust
Hong Kong University of Science and Technology (Guangzhou)
zixinzhong@hkust-gz.edu.cn

Abstract

We propose a novel piecewise stationary linear bandit (PSLB) model, where the
environment randomly samples a context from an unknown probability distribution
at each changepoint, and the quality of an arm is measured by its return averaged
over all contexts. The contexts and their distribution, as well as the changepoints are
unknown to the agent. We design Piecewise-Stationary ε-Best Arm Identification+
(PSεBAI+), an algorithm that is guaranteed to identify an ε-optimal arm with
probability ≥1 −δ and with a minimal number of samples. PSεBAI+ consists
of two subroutines, PSεBAI and NAÏVE ε-BAI (NεBAI), which are executed in
parallel. PSεBAI actively detects changepoints and aligns contexts to facilitate
the arm identification process. When PSεBAI and NεBAI are utilized judiciously
in parallel, PSεBAI+ is shown to have a finite expected sample complexity. By
proving a lower bound, we show the expected sample complexity of PSεBAI+ is
optimal up to a logarithmic factor. We compare PSεBAI+ to baseline algorithms
using numerical experiments which demonstrate its efficiency. Both our analytical
and numerical results corroborate that the efficacy of PSεBAI+ is due to the
delicate change detection and context alignment procedures embedded in PSεBAI.

1
Introduction

In stochastic multi-armed bandits (MABs), an agent interacts with the environment at each time step.
The agent pulls an arm and observes the corresponding return provided by the environment. The
classical MAB framework assumes a stationary environment where the expected return of each arm
remains unchanged over time. However, we usually face ever-changing environments in real life. For
instance, in investment option selection and portfolio management, fund managers want to select a
subset of good candidate portfolios. However, the market may be bullish, bearish, or in some other
state. The transition between these states can be well-modelled as being stochastic. We wish to select
portfolios that yield the best long-term option under such a piecewise stationary environment. Further
examples such as one based on agriculture in the face of stochastically changing weather patterns
are discussed in detail in Appendix A. These motivate us to formulate and investigate a piecewise
stationary linear bandit (PSLB) model.

Our PSLB model is equipped with an arm set X, a context set Θ and a deterministic but unknown
sequence of changepoints C. At each changepoint, the environment samples a context θ ∈Θ from an

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
unknown probability distribution Pθ, and the returns of arms may change when the context changes.
The return of each arm under each context is determined by its feature x ∈X and the context θ. In
particular, the expected return of an arm is the weighted sum µx = EPθ[x⊤θ]. While the sequence
of changepoints, as well as the distribution and latent vectors of contexts are not known, the agent
samples an arm and observes the corresponding return at each time step so that it can identify the
best arm arg maxx µx up to some tolerance ε with probability ≥1 −δ and with as few samples as
possible. The agent’s behavior does not affect the sequence of contexts that is drawn from Pθ.

Main Contributions. We are the first to study the fixed-confidence best arm identification (BAI)
problem in piecewise stationary bandits (PSB). Given δ > 0, we say the arm with the highest
expected return µ∗is the best, and an arm is ε-optimal if its expected return is at least µ∗−ε. We
seek to design an (ε, δ)-PAC algorithm which can identify an ε-optimal arm with probability ≥1 −δ
in as few time steps as possible, i.e., with minimal sample complexity.

Our first contribution concerns the formulation of a novel PSLB model, where we measure the quality
of an arm x according to its expected return µx = Eθ∼Pθ[x⊤θ] for the following reasons. Consider
that an arm is measured by its average return across time, which is a generalization of the definition in
stationary bandits (SB). A notable feature of PSB models is that the context changes as time evolves,
and hence the arm’s average return across time also changes, in general. Hence, we aim to identify
an arm whose average return across contexts is high, and benefits the agent for interacting with the
environment in the long run after the arm identification task. We are thus inspired to introduce the
distribution of contexts under the PSLB model, define the expected return µx for each arm, and use
this ensemble (non-time varying) statistic as the benchmark for what we seek to learn. The BAI task
using this statistic is meaningful but challenging, as the agent needs to reliably estimate the context
vectors, changepoints, and context distribution.

Secondly, we propose PIECEWISE-STATIONARY ε-BAI+ (PSεBAI+), an algorithm designed to
tackle the BAI problem in our PSLB model. We prove that it is (ε, δ)-PAC and bound its sample
complexity. PSεBAI+ samples arms according to a suitably defined G-optimal allocation, and runs
two algorithms: NAÏVE ε-BAI (NεBAI) and PSεBAI as subroutines in parallel to achieve efficiency
and attain a bound on the sample complexity in expectation.
• Being a baseline but naïve algorithm, the complexity of NεBAI grows linearly with the maximum
length between two changepoints Lmax, motivating us to design a more efficient algorithm, PSεBAI,
to reduce the impact of Lmax.
• PSεBAI is equipped with two delicately designed subroutines LINEAR-CHANGE DETECTION
(LCD) and LINEAR-CONTEXT ALIGNMENT (LCA) to actively detect changepoints and align
contexts with those observed in the previous time steps respectively. Concretely, in terms of the
design, PSεBAI determines whether samples from two intervals are under the identical context via a
sliding window mechanism, and detects changepoints and aligns contexts accordingly; this facilitates
the estimation of context vectors and their distribution. Combining these elements into the design
of PSεBAI and analyzing them requires some care. On the theoretical side, we prove PSεBAI
identifies an ε-optimal arm faster than NεBAI with high probability. The success of PSεBAI relies
on the LCD and LCA subroutines, while a minor drawback is that they require a non-vanishing
failure probability budget which does not allow us to bound the sample complexity of PSεBAI in
expectation. To achieve a complete theoretical understanding, we delicately design the PSεBAI+
algorithm whose efficiency is inherited via the LCD and LCA procedures in PSεBAI as well as the
effective utilization of running PSεBAI and NεBAI in parallel.

Thirdly, we derive a lower bound on the complexity of any (ε, δ)-PAC algorithm in PSLB models.
To derive this bound, we first lower bound the complexity of an algorithm when the contextual
information (and changepoints) are available, and then quantify the number of arm samples (and
realized contexts) required to reliably infer an ε-best arm.We compare the upper bound of PSεBAI+
and this generic lower bound in several instances. The matching (up to logarithmic terms) of bounds
illustrate that our PSεBAI+ algorithm is almost asymptotically optimal.

Lastly, we demonstrate the efficiency of PSεBAI+ with numerical experiments. The first half of
our experiment shows that PSεBAI+ is (ε, δ)-PAC and with significantly lower sample complexity
compared to NεBAI, corroborating our theoretical findings. In the second half, we compare PSεBAI+
to NεBAI, and two other benchmarks DISTRIBUTION ε-BAI (DεBAI) and DεBAIβ.While contexts
and changepoints are not available to PSεBAI+ and NεBAI, they are observed by DεBAI and
DεBAIβ. Nevertheless, PSεBAI+ is still competitive compared to DεBAI and DεBAIβ in our

2


---Page Break---
empirical experiments. Hence, both experiments justify the necessity of the change detection and
context alignment procedures for boosting the learning of contexts and their distributions, as well as
the identification of the best arm. We also show empirically that misspecifications to the knowledge
of Lmin and Lmax do not affect the performance of our algorithm significantly.

Related work. Best arm identification (BAI) and regret minimization (RM) are two fundamental
problems in multi-armed bandits. In stationary linear bandits, [1, 2] focus on BAI and [3, 4] aim to
solve the RM problem. An efficient algorithm can choose the G-optimal allocation or XY-allocation
rule to quickly identify a good arm [1, 5]. Besides, [6, 7] focus on ε-optimal arm identification, which
is a generalization of the standard BAI problem.

The BAI and RM problems are also studied in thr non-stationary bandits (NSB), where in contrast to
the SB model, the context varies with time [8–10]. NSB can be largely divided into two classes: the
drifting bandit (DB) model, where the context changes at each step [8], and PSB, where the context
changes less frequently [10]. [11] provides an extensive discussion on the definition of NSB models.

On one hand, the RM problem have been investigated extensively in DB models [12, 13]. On the
other hand, concerning the BAI task in DB models, [14] investigated BAI with a fixed-horizon, where
the best arm has the highest average return over this horizon; [15] assumes the best arm remains
unchanged after certain time step and explores the fixed-confidence setting. Besides, when the
contextual information in NSB models is available, NSB models are known as contextual bandit (CB)
models [16–18]. [16] showed that the contextual information accelerates the best arm identification
process. More discussions on DB and CB models are presented in Appendix C.

Moreover, the context can drift dramatically in a DB model while it remains unvarying in a SB model.
Straddling between the DB and SB models, PSB models assume there is an interim stationary interval
between each two consecutive changepoints, where the context remains unchanged. The context
changes can be characterized in different ways and affect the performance of proposed algorithms in
PSB models. For instance, a changepoint signals the return of at least one arm shift as in [10], and
indicates the best arm changes as in [19].

In PSB models, a large body of works focus on RM. While [10, 20–22] equip their algorithms with
changepoint detection techniques to handle the context changes, [23] actively checks the quality of
each arm. However, there is no existing work on the fixed-confidence BAI problem in a PSB model.
To fill this gap in the literature, we design PSεBAI+ for ε-optimal arm identification in our proposed
PSLB model. We show PSεBAI+ is almost asymptotically optimal by comparing its complexity to a
generic lower bound of all algorithms, and validate its efficiency through numerical experiments.

2
Problem Setup

For m ∈N, let [m] := {1, 2, . . . , m}. For a finite set S, let ∆S denote the |S|-dim probability
simplex on S. Let A(q) := P
x∈X qxxx⊤be the matrix induced by q ∈∆X with X ⊂Rd. An
instance of piecewise stationary linear bandit is a tuple Λ = (X, Θ, Pθ, C). Specifically, x ∈Rd is
an arm (vector) and the arm set X ⊂Rd is composed of |X| = K arms that spans Rd. The latent
vector matrix Θ = (θ∗
1, . . . , θ∗
N) ∈Rd×N contains N latent column vectors where the jth column
θ∗
j is associated with context j ∈[N]. For the sake of normalization, we assume |x⊤θ∗
j | ≤1 for
all x ∈X, j ∈[N]. Let Pθ denote the distribution (probability mass function) of the latent vectors
and pj = Pθ[θ∗
j ]. We represent the probabilities of latent vectors as p = (p1, . . . , pN) ∈∆N. The
fixed but unknown sequence of changepoints C := (c1, c2, . . .) is an sequence of increasing positive
integers 1 = c1 < c2 < . . ., characterizing all the changepoints (time steps).

The return of arm x under latent context j is a random variable Y = x⊤θ∗
j + η, where η is
a zero-mean random variable (noise) supported on [−1, 1], and the expected return of arm x is
µx := Eθ∼Pθ[x⊤θ] = PN
j=1 Pθ[θ∗
j ]x⊤θ∗
j . The best arm, which we assume is unique, is denoted as
x∗:= arg maxx∈X µx with mean µ∗:= µx∗. Given a slackness parameter ε > 0, we define the
set of ε-best arms Xε := {x ∈X : µx ≥µ∗−ε}. For each pair of arms (x, ˜x) ∈X 2, define the
contextual mean gap between x and ˜x under latent context j as ∆j(x, ˜x) := (x −˜x)⊤θ∗
j and the
mean gap between x and ˜x as ∆(x, ˜x) := µx −µ˜x.

Given l ∈N, the interval (cl, . . . , cl+1 −1) is known as the lth stationary segment and its length is
Ll := cl+1 −cl. We assume Lmin ≤Ll ≤Lmax. Let lt := max{l : cl ≤t} denote the number of

3


---Page Break---
stationary segments up to time step t. At time step t ∈[T] (see Dynamics 1),
(i) If t ∈C, the environment samples a latent vector θ∗
jt according to Pθ, that is, it generates a
(latent) context sample with Pθ; otherwise the latent vector remains unchanged, i.e., θ∗
jt = θ∗
jt−1. The
contexts {θ∗
jcl }l∈N are sampled i.i.d. from Pθ at each changepoint {cl}l∈N.

(ii) The agent pulls an arm xt and observes the stochastic return Yt,xt = x⊤
t θ∗
jt + ηt, where ηt is
drawn independently from a distribution supported on [−1, 1].

The agent uses an online algorithm π := {(πt, τt, rt)}t∈N to actively interact with the instance Λ
and only has access to the arm set X, number of latent vectors N, the bounds on the length of each
segment Lmin and Lmax, the slackness parameter ε, and the confidence parameter δ.
• sampling rule πt : Hπ
t :=
 
(xπ
s , Ys,xπ
s )


s∈[t−1] →X samples an arm xπ
t based on the observation
history Hπ
t and observe the corresponding random return Yt,xπ
t ;
• stopping rule τt : Hπ
t+1 →{STOP, CONTINUE} decides whether to stop or continue to execute
given the observation history Hπ
t+1. The stopping time under algorithm π is denoted as τ π;
• recommendation rule rτ : Hπ
τ+1 →X recommends an arm ˆxπ based on Hπ
τ+1 upon termination.
The stopping time τ π is the sample complexity of the algorithm π under instance Λ. The expected
sample complexity is E[τ π], where the expectation is taken w.r.t. the random returns, the realization
of the contexts governed by the latent vector distribution Pθ, and the randomness of the algorithm π.

An algorithm π is (ε, δ)-PAC (Probably Approximately-Correct) if

P[ˆxπ ∈Xε] ≥1 −δ.

Our overarching goal in this paper is to devise an (ε, δ)-PAC algorithm that minimizes τ π with high
probability (w.h.p.) and in expectation.

3
Algorithms

3.1
A Naïve Baseline: NεBAI

We first devise the NAÏVE ε-BEST ARM IDENTIFICATION (or NεBAI) algorithm (presented in
Algorithm 2). In the design of NεBAI, only the choice of confidence radius ˜ρt takes the potential
context changes into consideration. Even though NεBAI does not attempt to detect potential changes
in the context, it can identify an ε-optimal arm w.h.p. and is with a finite expected sample complexity.

Proposition 3.1. Let ∆min = minx̸=x∗∆(x∗, x),

T N
V =
d

(ε + ∆min)2 ln 1

δ
and
T N
D =
Lmax
(ε + ∆min)2 ln 1

δ .

The NεBAI algorithm is (ε, δ)-PAC and its expected sample complexity is ˜O(T N
V + T N
D ).

The upper bound in Proposition 3.1 (also see Appendix F) consists of two main terms. (i) As NεBAI
samples arms according to the G-optimal allocation (see Appendix D), the amount of samples needed
to estimate the average of latent vectors Pt
s=1 θ∗
js/t contributes to T N
V . (ii) T N
D quantifies how fast
Pt
s=1 θ∗
js/t converges to the expectation of context vectors Pt
j=1 pjθ∗
j .

The sample complexity of NεBAI grows linearly with Lmax, but we surmise that the sample
complexity of a close-to-optimal algorithm should have a reduced dependence on Lmax.

3.2
Piecewise-Stationary ε-Best Arm Identification

The algorithm PIECEWISE-STATIONARY ε-BEST ARM IDENTIFICATION (or PSεBAI) is presented
in Algorithm 1. By using a sliding window mechanism, PSεBAI actively detects the change-
points and aligns the current latent context with contexts observed in the previous time steps via
LINEAR-CHANGE DETECTION (or LCD) and LINEAR-CONTEXT ALIGNMENT (or LCA), which
are presented in Algorithms 3 and 4 (see App. D.2.2), respectively. PSεBAI consists of three phases:
(i) Exploration phase (Exp): Estimate latent vectors and their distribution Pθ (Lines 8 to 11 and 25);
(ii) Change Detection phase (CD): Detect changepoints (Lines 12 to 16);
(iii) Context Alignment phase (CA): Evaluate the current context and align it with the contexts
observed in previous time steps (Lines 17 to 21).

4


---Page Break---
At time t, we estimate Θ and p with ˆΘt = (ˆθt,1, . . . , ˆθt,N) and ˆpt = (ˆpt,1, . . . , ˆpt,N)⊤, respectively.1

We denote the empirical mean gap between x and ˜x under context j as ˆ∆t,j(x, ˜x) := (x −˜x)⊤ˆθt,j.

PSεBAI first computes the G-optimal allocation [1] λ∗on the arm set X and its maximum possible
stopping time τ ∗(Line 2). It initializes CDsample and CAid. CDsample collects samples to detect
changepoints and CAid maintains a dictionary of {latent context index : identification samples}
pairs (Line 3);2 CAid[j] is the sequence of CD samples used to identify latent context j. It also
initializes Tt,j, the collection of time indices in [t] in the Exp phases under estimated context j.
Define

Tt =
[

j∈[N]
Tt,j,
Tt = |Tt|,
Tt,j = |Tt,j|,
∀j ∈[N].
(3.1)

Algorithm 1 PIECEWISE-STATIONARY ε-BEST ARM IDENTIFICATION (PSεBAI)

1: Input: arm set X, size of the set of latent
vectors N, bounds on the segment lengths
Lmin and Lmax, slackness parameter ε, con-
fidence parameter δ, sampling parameter γ
and window size w, threshold b.
2: Initialize: Compute the G-optimal allocation
λ∗and τ ∗= 38400 ln(80)NLmax

ε2
ln N2KLmax

δε2
.

3: Set CDsample = [ ], CAid = { }.
Set
tCD = +∞.
4: Set Tt,j
= ∅and initialize Tt, Tt,j, Tt
with (3.1) for all t ≤τ ∗, j ∈[N].

5: Sample
w

2 arms {xs}

w

2
s=1 ∼λ∗and ob-

serve the associated returns {Ys,xs}

w

2
s=1, t =
w

2 , tCA = w

2 .

6: CAid = {1 : [(xs, Ys,xs)]

w

2
s=1}, ˆjt = 1.
7: while t ≤τ ∗
do
8:
t = t + 1
9:
Sample an arm xt ∼λ∗and observe return
Yt,xt.
10:
if
mod (t −tCA, γ) ̸= 0 then
11:
Update ˆjt = ˆjt−1, Tt,ˆjt = Tt−1,ˆjt ∪
{t}, Tt,j = Tt−1,j for j ̸= ˆjt.
12:
else
13:
CDsample = CDsample + [(xt, Yt,xt)].

14:
Update ˆjt = ˆjt−1, Tt,j = Tt−1,j for all
j ∈[N].
15:
if |CDsample| ≥w then
16:
if LCD(X, w, b, CDsample[−w : ])
then
17:
CDsample = [ ].
18:
t = t + w

2 , tCA = t, tCD = +∞.

19:
ˆjt, CAid = LCA(X, w, b, CAid).

20:
if ˆjt = N + 1 then break.
21:
Revert Tt,j = Tt−w(γ+1)

2
,j for all
j ∈[N].
22:
end if
23:
end if
24:
end if
25:
Update the estimates using
(3.1), (3.2)
and (3.3).
26:
if Condition (3.4) is met and tCD = +∞
then
27:
Record ˆxε = arg maxx∈X x⊤ˆΘtˆpt.
28:
tCD = |CDsample|.
29:
else if tCD = |CDsample| −w

2 then
30:
Recommend arm ˆxε.
31:
break
32:
end if
33: end while

It then collects w

2 samples and stores them in CAid, which is then used to identify the first latent
context (Lines 5 to 6).

In the Exp phase, PSεBAI firstly samples an arm xt with λ∗and observes the return Yt,xt =
x⊤
t θ∗
jt + ηt (Line 9). It then updates the estimated context index and time collectors (Line 11). It
also updates the estimates of value and probability of each context j (Line 25) with

ˆθt,j =
1
Tt,j

X

s∈Tt,j
A(λ∗)−1xsYs,xs
and
ˆpt,j = Tt,j

Tt
,
(3.2)

1The empirical latent vector-probability pairs [(ˆθt,j, ˆpt,j)]N
j=1 can only approximate the unknown pairs
[(θ∗
σ(j), pσ(j))]N
j=1 up to a permutation σ : [N] →[N], which is determined by the occurrence order of latent
vectors. Thus we assume the latent vectors appear in order of increasing indices.
2A sample in CDsample is a CD sample; A dictionary has a pairing structure {key:value} and
dictionary[key] = value. [ai]n
i=1 denotes a sequence of elements a1, . . . , an.

5


---Page Break---
where ˆθt,j = 0 if Tt,j = 0. We define αt, ξt, βt,j, and ˆ∆clip2
t,j
(x, ˜x) in Appendix D.2.1. For each
pairs of arms (x, ˜x), the confidence radius of ∆(x, ˜x) at time step t is

ρt(x, ˜x) := 2(αt + ξt) +

N
X

j=1
βt,j| ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)|;
(3.3)

PSεBAI actively enters the CD phase every γ time steps (Line 12). It firstly adds a CD sample
to CDsample (Line 13). Next, if there are sufficient CD samples (Line 15), the LCD subroutine
(presented in Algorithm 3) is called and utilizes the most recent w CD samples to check whether a
changepoint just occurred (Line 16). PSεBAI steps into the CA phase if a changepoint is detected,
and skips the CA phase otherwise, which is illustrated by Figures 1(b) and 1(a) respectively.

Context

 active CD samples

 

Step 1: No change alarm is raised
Step 2: Check stopping rules
Step 3: Start a new Exp phase

Time step

Change Detection Phases
Exploration Phases
Active CD samples
Inactive CD samples

 samples
in an Exp phase

(a)

Context

Step 3: Revert the statistics

Step 1: LCD raises a change alarm

Step 2: LCA aligns the context

Context Alignment Phases

Step 4: Start a new Exp phase 

Time step

Change Detection Phases
Exploration Phases
Active CD samples
Inactive CD samples

(b)

Figure 1: (a) No change alarm is raised during a stationary segment. The active CD samples are the
input to the LCD subroutine at current time step t. (b) A changepoint is detected by LCD, followed
by a CA phase and a statistics reversion step.

In the CA phase, PSεBAI starts by resetting CDsample (Line 17), updating the count of time steps
and recording the ending time of this CA phase (Line 18). Thereafter, the CA subroutine (presented
in Algorithm 4) is invoked, which estimates the current latent context index ˆjt and updates CAid
(Line 19). If ˆjt = N + 1, i.e., PSεBAI identifies N + 1 latent contexts, which is incorrect under
instance Λ, it terminates and fails to identify an ε-optimal arm (Line 20). Lastly, all empirical
statistics are reverted to those from (w(γ + 1)/2) time steps ago, i.e., the most recent (wγ/2)
samples in the Exp phases are abandoned (Line 21).

The stopping rule is described in Lines 26 to 32. (I) If the following condition is satisfied (Line 26):

min
x:x̸=x∗
t
ˆ∆t(x∗
t , x) −ρt(x∗
t , x) ≥−ε
and
Tt ≥2Lmax

9
ln

2
δd,Tt


(3.4)

where the empirical mean gap ˆ∆t(x∗
t , x) := (x∗
t −x)⊤ˆΘtˆpt and x∗
t := arg maxx∈X x⊤ˆΘtˆpt,
PSεBAI records the arm with the highest empirical mean as ˆxε and the number of CD samples tCD
(Lines 27 and 28) but does not terminate immediately. Besides, a mild forced arm pull procedure is in
the second line of (3.4), which is inspired by Lemma E.1 and to ensure the performance of PSεBAI.
(II) PSεBAI will execute for another (wγ/2) time steps in which w/2 CD samples are collected;
if no changepoint is detected with these w/2 CD samples, the recorded arm ˆxε is recommended
and PSεBAI terminates (Lines 29 to 30). Part (II) of the stopping rule assures PSεBAI does not
terminate when a changepoint has occurred but has not been detected, as PSεBAI may fail to identify
an ε-optimal arm otherwise.

We remark that even though PSεBAI uses the knowledge of Lmax, our experiments show that
the performance of PSεBAI is robust to small misspecifications in Lmax (see Appendix O.2).
Furthermore, the computational complexity of PSεBAI is computed in detail in Appendix D.4. The
derived computational complexity indicates the proposed algorithm depends in a natural manner on
the problem parameters such as d, K, N, and γ. Lastly, thanks to the LCD and LCA subroutines, a
slightly modified variant of PSεBAI can also solve the “ε-Best Arm Tuple identification problem”,
which aims to identify an ε-best arm under each context; see Appendix Q for details.

6


---Page Break---
3.2.1
Theoretical guarantee of PSεBAI

To facilitate the analysis of PSεBAI, we propose the following assumptions. Note that our PSεBAI
algorithm may still succeed to identify an ε-optimal arm w.h.p. when the assumptions do not hold.

Assumption 1 (Distinguishability Condition). The agent can choose w, γ and b such that (1) 2b ≤∆c
where ∆c := minθ∗
j ̸=θ∗
˜j maxx∈X |x⊤(θ∗
j −θ∗
˜j )| is the minimum gap between two contexts; and (2)
3wγ ≤Lmin. A possible choice is

b= 8d

3w ln
2
δFAE
+

r 8d

3w ln
2
δFAE

2
+ 24d

w ln
2
δFAE
where
δFAE =
γδ
4(τ ∗)2K .
(3.5)

This assumption guarantees (i) PSεBAI will not abandon all samples during the reversion procedure
(Line 21 of Algorithm 1); (ii) each two latent vectors can be distinguished if the window size w is
sufficiently large (e.g., Lmin/6). We clarify that this assumption is only for the rigor of theoretical
guarantees and it holds provided that each stationary segment is sufficiently long; this is a feature of
PSB models and similar assumptions are also present in existing works for their analyses [21, 10, 22].
We demonstrate the robustness of PSεBAI to these parameters using experiments in Section 6.

Theorem 3.2. Define the context distribution estimation (DE) hardness parameter

HDE(xε, x) :=
Lmax
(∆(x∗, x) + ε)2 ¯H(xε, x)

where ¯H(xε, x) :=
  PN
j=1
p

min {16pj, 1/4}|∆j(xε, x) + ε|
2. Under Assumption 1, with proba-
bility at least 1 −δ, PSεBAI identifies an ε-optimal arm and its sample complexity is

˜O

 

max
xε∈Xε
x̸=xε,x∗

d

(∆(x∗, x) + ε)2 ln 1

δ
|
{z
}
TV(x)

+ HDE(xε, x) ln 1

δ
|
{z
}
TD(xε,x)

+
NLmax
∆(x∗, x) + ε ln 1

δ
|
{z
}
TR(x)

!

.
(3.6)

The upper bound comprises three terms which serve distinct purposes:
(i) Latent vector estimation (VE): ˜O (TV(x)) quantifies the bulk of samples needed to obtain a good
estimate of latent context vectors such that the returns of xε and x can be distinguished, where xε
is an ε-best arm and x /∈Xε is a suboptimal arm. TV(x) recovers the sample complexity in the
stationary linear bandits in [1], indicating that PSεBAI estimates latent vectors efficiently.
(ii) Context distribution estimation (DE): ˜O (TD(xε, x)) characterizes the bulk of samples needed to
learn the distribution of latent context vectors.
(iii) Residual estimation (RE): ˜O (TR(x)) counts the remaining samples needed for VE and DE, in
addition to ˜O (TV + TD).
Besides, the max operator is applied to exclude all suboptimal arms. We also see that TV(x) and
TD(xε, x) are similar to T N
V and T N
D in Proposition 3.1 respectively.

Firstly, the bound in (3.6) implies that, in an instance with smaller relaxed mean gap ∆(x∗, x) + ε,
PSεBAI terminates after a larger number of time steps; in other words, it is more difficult to identify
an ε-optimal arm. In difficult instances with small ∆(x∗, x) + ε, the different orders of this term in
TV(x), TD(xε, x) and TR(x) indicate that, TR(x) is small compared to TV(x) and TD(xε, x).

Secondly, DE solely utilizes context samples generated with Pθ and they are generated only at
changepoints in C, while all the observations in Exp phases facilitate VE. From this perspective, there
are less samples that can be used for DE than for VE as PSεBAI processes, and hence TD(xε, x) is
supposed to be with larger order than TV(x).

Moreover,
for
the
purpose
of
DE,
PSεBAI
needs
to
observe
context
samples
at
˜O
 
¯
H(xε,x)
(∆(x∗,x)+ε)2 ln 1

δ

= ˜O
  HDE(xε,x)

Lmax
ln 1

δ

changepoints where Lmax is the maximum length of a

stationary segment, leading us to TD(xε, x). Close examination of the definition of ¯H(xε, x) reveals
that both the vectors and their probabilities influence the number of samples needed for DE. The
comparison between TD(xε, x) and T N
D in Proposition 3.1 clearly indicates that PSεBAI mitigates
the influence of Lmax by detecting changepoints and aligning the detected context with observed
ones, while NεBAI does not do so.

7


---Page Break---
3.3
PSεBAI+ = PSεBAI ∪NεBAI

We have provided a high-probability result for PSεBAI in Theorem 3.2. The design of PSεBAI
(Line 7 of Algorithm 1) indicates that PSεBAI will not recommend any arm if it does not terminate
at time τ ∗. This result is nontrivial, as the high-probability result in Theorem 3.2 depends on the
success of change detection (Algorithm 3) and context alignment (Algorithm 4), which requires a
non-vanishing failure probability (e.g., δ/2). Thus, we cannot derive an upper bound on the expected
sample complexity of PSεBAI. We devise a solution by designing the Piecewise-Stationary ε-Best
Arm Identification+ (PSεBAI+) algorithm with a simple but effective trick.

The PSεBAI+ algorithm samples one arm with the G-optimal allocation λ∗at each time step, with
which Algorithms 1 and 2 are executed in parallel (detailed in Algorithm 5). This is feasible since
PSεBAI and NεBAI algorithms have the same sampling rule.
Theorem 3.3. The PSεBAI+ algorithm is (ε, δ)-PAC and its expected sample complexity is

˜O

min

max
xε∈Xε,x̸=xε,x∗TV(x) + TD(xε, x) + TR(x), T N
V + T N
D


.

PSεBAI+ inherits the superiority of PSεBAI to adapt to the piecewise stationary environment, and
employs the stopping rule of NεBAI to maintain a finite expected sample complexity. As a result, the
expected complexity of PSεBAI+ in Theorem 3.3 is of the same order as the high-probability one
of PSεBAI in Theorem 3.2 and is not larger than the complexity of NεBAI in Proposition 3.1. We
show how our results particularize to the stationary linear bandits BAI problem, as well as additional
discussions on the upper bound, in Appendix P.

4
Lower Bound on the Sample Complexity

Given Λ = (X, Θ, Pθ, C), define the alternative instance Λ′ = (X, Θ′, Pθ′, C) w.r.t. Λ, where Θ′ =
(θ′
1, . . . , θ′
n) ∈Rd×N, Pθ′[θ′
j] = Pθ[θ∗
j ], and there exists x ∈X \ Xε, such that x⊤
ε Eθ′∼Pθ′[θ′] <
x⊤Eθ′∼Pθ′[θ′] −ϵ for all xε ∈Xε. Let AltΘ(Λ) be set of all alternative instances (w.r.t. Λ).
Theorem 4.1. For all (ε, δ)-PAC algorithm π, there exists an instance Λ = (X, Θ, Pθ, C) such that

E[τπ] ≥max

Tε(Λ) ln
1
2.4δ , cNC


,

where

Tε(Λ)−1 := max
{vj}N
j=1
min
Λ′∈AltΘ(Λ)

X

j,x
pjvj,x
(x⊤(θ∗
j −θ′
j))2

2
,
and

NC := max
x̸=x∗

P

j pj(∆j(x∗, x)+ε)2

(∆(x∗, x) + ε)2
ln 1

4δ .

Recall that cNC is the NC-th changepoint in the changepoint sequence C, which is lower bounded by
NCLmin and is NCLmax in the worst case. To derive the lower bound in Theorem 4.1, we investigate
two environments different from the one defined in Section 2 (and as in Dynamics 1):
• Dynamics 2: the agent observes the index of current context jt (i.e., contextual linear bandits);
• Dynamics 3: the agent observes the changepoints in C and context vector θ∗
jt’s, and hence she
solely needs to estimate the distribution of contexts.
We bound the sample complexity of an (ε, δ)-BAI algorithm in Dynamics 2 and 3 respectively, which
when combined, yield the lower bound in Theorem 4.1; this is detailed in Appendix M.

Note that Tε(Λ)−1 in the lower bound generalizes [16] to the setting of linear bandits. In addition,
Theorem 4.1 can be reduced to a bound in stationary linear bandits with one latent vector [24] (see
the discussion leading to (M.15)).

5
On the Asymptotic Optimality of PSεBAI+

To illustrate the efficiency of our PSεBAI+ algorithm, we compare the upper bound on its expected
sample complexity in Theorem 3.3 and the generic lower bound in Theorem 4.1 under specific
instances below and in Appendix N. We also gain further insight into our PSεBAI+ algorithm.

8


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

106

108

1010

(a) Sample complexity of PSεBAI+ vs. NεBAI.

0
10
20
30
40
50
60
70
80
0

100

200

300

400

500

600

700

800

900

Number of Context Samples

1
2
3
4
5
6

10

20

30

0
20
40
60
80
103

104

105

(b) Number of context samples needed for BAI.

Figure 2: Experimental results

Example 1. Instance Λ = (X, Θ, Pθ, C) is with (i) 2d −1 arms: x(1) = e1, x(i) = ei, x(d+i−1) =
e1 cos ϕ+ei sin ϕ for all i ∈{2, . . . , d} where ϕ ∈[0, π/4), (ii) 2d−2 contexts: θ∗
j± = e1 cos ϕ±
ej+1 sin ϕ for all j ∈[d −1], (iii) Context distribution: pj = 1/N for all j ∈[N].

Under the instance defined in Example 1, x(i) for all i ̸= 1 is inferior to x(1) under all contexts and
x(i+d) for all i ∈[d −1] is marginally better than x(1) by 1 −cos ϕ only under context θ∗
i+ and
∆(x(1), x(i+d)) = cos ϕ −cos2 ϕ. We expect PSεBAI+ to discover this feature of the instances and
quickly identify an ε-optimal arm with a course estimation of the context distribution.

Corollary 5.1. For the instance defined in Example 1, we have HDE(xε, x) = ˜O(NLmax) for all
(xε, x) ∈Xε × (X \ Xε). In addition, if ε < (cos ϕ)(1 −cos ϕ), we have

E[τ]∗

ln(1/δ) ∈˜Θ

(1+f(ϕ)) ·
d
(∆x(1),x(d+1) +ε)2


,
(5.1)

where E[τ]∗is the minimal expected sample complexity over all (ε, δ)-PAC algorithms and f : R →R
satisfies f(ϕ) →0 as ϕ →0+. The upper bound in (5.1) is achieved by PSεBAI+.

The order of HDE in Corollary 5.1 indicates that TD(x(1), x(d+1)) = ˜O(NLmax ln(1/δ)) is not
dominating the sample complexity of PSεBAI+, suggesting that a coarse estimation of the context
distribution is sufficient when ϕ is small. In other words, PSεBAI+ exploits the feature of instances
and utilize samples mostly for estimate context vectors, which is again expected.

Corollary 5.1 implies that under such instances, the upper and lower bounds on the sample complexity
of PSεBAI+ match up to logarithmic factors, that is, the performance PSεBAI+ is near optimal.
Besides, the bound of NεBAI in Proposition 3.1 is with an extra additive term Lmax compared to
the lower bound in Corollary 5.1, illustrating that NεBAI is suboptimal and again emphasizing the
significance of detecting changes and aligning contexts for PSεBAI+ to reduce the impact of Lmax.

6
Numerical Experiments

We now evaluate the empirical performance of PSεBAI+. We utilize the instance defined in
Example 1 with d = 2, ϕ = π/8, We generate a changepoint sequence C such that cl+1 = cl + Ll
with Lmin = 3 × 104, Lmax = 5 × 104, P[Ll = Lmin] = 0.8, P[Ll = Lmax] = 0.2, and fix it
throughout the whole set of experiments. We set the confidence parameter δ = 0.05 and vary the
slackness parameter ε from 0.04 to 0.6 (i.e., ε = 0.03 × 1.35k for k ∈[12]). We set γ = 6, the
window size w = Lmin/(3γ) and compute b via (3.5) in Assumption 1.3 For each choice of algorithm
and instance, we run 20 independent trials. All the code to reproduce our experiments can be found
at https://github.com/Y-Hou/BAI-in-PSLB.git.

We first compare PSεBAI+ and NεBAI. Both algorithms succeed to identify an ε-optimal arm,
while empirically, the complexity of PSεBAI+ is ≤1% of that of NεBAI. The empirical averages
and standard deviations of the sample complexities of both algorithms are presented in Figure 2(a).

3We clarify that (3.5) in Assumption 1 is only for our theoretical guarantees. In practice, our algorithm has
shown robustness w.r.t. the parameters and we can safely neglect the constants in the formula.

9


---Page Break---
Figure 2(a) illustrates that empirically, the termination and arm recommendation of PSεBAI+ are
determined by the execution of PSεBAI as a subroutine, suggesting that in Theorem 3.3, the first
term resulting from PSεBAI actually determines the complexity of PSεBAI+.

Next, we test the efficacy of PSεBAI+ to learn and exploit the latent vectors and the distribution
of contexts. Specifically, we run PSεBAI+ and NεBAI under Dynamics 1 where neither the index
nor the vector of current context is visible to the agent. We also run benchmark algorithms: DεBAI
and its variant DεBAIβ under Dynamics 3 where context vectors and changepoints are all observed;
these two algorithms are detailed and analyzed in Appendix O.4.

As the changepoint sequence C is fixed in a given instance and t/Lmax ≤lt ≤t/Lmin for all t ∈N,
we regard the number of context samples lτ as a proxy of the sample complexity τ. We present the
number of context samples need by PSεBAI+, NεBAI, DεBAI and DεBAIβ for arm identification
w.r.t. 1/(∆min + ε)2 in Figure 2(b).

Figure 2(b) contains three messages. First, the complexity of PSεBAI+ scales as 1/(∆min + ε)2,
corroborating Theorem 3.3. Second, although PSεBAI+ has access to neither context vectors
nor changepoints, it needs roughly the same number of context samples as DεBAI and DεBAIβ,
suggesting that it is competitive compared to these algorithms that have oracle information about the
environment. Third, DεBAIβ uses the confidence radius in (3.3) and terminates with fewer context
samples compared to DεBAI, implying that the confidence radius is well-designed.

Furthermore, when ε decreases from 0.03 × 1.3512 to 0.03 × 1.359, the complexity of PSεBAI
almost remains unchanged while that of NεBAI increases rapidly as presented in instances 9 to 12 in
Figure 2(a). Meanwhile, the number of context samples need by two algorithms are shown to be with
the same pattern in Figure 2(b). This contrast indicates that the cost of distribution estimation (TD
in (3.6)) for PSεBAI+ has been significantly minimized compared to NεBAI.

To summarize, we emphasize that the empirical superiority of PSεBAI+ over NεBAI implies that
the efficacy of PSεBAI+ is inherited from PSεBAI. Our experiments show that actively exploiting the
context information, via changepoint detection and context alignment (as in PSεBAI+ and PSεBAI)
facilitates identifying the ε-optimal arm efficiently.

Similar to many existing algorithms in piecewise-stationary bandits [21, 10, 22], our algorithm
requires Assumption 1 and the knowledge of Lmax. These may not be available in practice. Thus, we
conduct more experiments in Appendix O.2 to exhibit the robustness of PSεBAI+. Specifically, in
Appendix O.2, we conduct experiments for the case in which Lmax is misspecified. In Appendix O.3,
we alter the change detection frequency γ so that w and b change accordingly. In both sets of
experiments, the overall sample complexity of PSεBAI+ does not vary significantly and retains its
superiority over NεBAI. We conclude that PSεBAI+ is robust to slight misspecifications in these
parameters, as long as Assumption 1 is not severely violated. Please refer to Appendix O for further
details and experiments.

7
Conclusion and Future Work

We proposed a novel PSLB model and designed the PSεBAI+ algorithm to identify an ε-optimal
arm with probability ≥1 −δ. The efficacy of PSεBAI+ has been demonstrated both empirically
and theoretically. We argued that this is due to the embedded change detection and context alignment
procedures. There are several directions for further exploration.

Firstly, our PSεBAI+ algorithm provides a fairly general framework for algorithm design. For
instance, in addition to utilization the G-optimal allocation to sample arms as in PSεBAI+, the
XY-allocation and adaptive XY-allocation [1] can also be considered. In other words, our PSεBAI+
algorithm can be generalized to form an entire class of algorithms for BAI in PSLB models. In
addition, deriving instance-dependent guarantees is also of great interest.

Secondly, most of the literature on piecewise-stationary bandits [21, 10, 22] make assumptions to
provide theoretical guarantees. It would be interesting to remove or reduce these assumptions under
our ε-BAI problem setup, and yet still be able to provide similar theoretical guarantees.

Finally, we believe that it is possible to adapt our PSεBAI+ algorithm to the fixed-budget setting,
i.e., to identify an ε-optimal arm with high probability in a fixed time horizon in PSLB models.

10


---Page Break---
Acknowledgements:
This work is funded by the Singapore Ministry of Education AcRF Tier 2
grant (A-8000423-00-00) and Tier 1 grants (A-8000189-01-00 and A-8000980-00-00).

References

[1] Marta Soare, Alessandro Lazaric, and Rémi Munos. Best-arm identification in linear bandits.
In Proceedings of the 27th Advances in Neural Information Processing Systems, volume 27,
pages 828–836, 2014.

[2] Junwen Yang and Vincent Tan. Minimax optimal fixed-budget best arm identification in
linear bandits. Proceedings of the 36th Advances in Neural Information Processing Systems,
35:12253–12266, 2022.

[3] Yasin Abbasi-yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. In Proceedings of the 24th Advances in Neural Information Processing
Systems, volume 24. Curran Associates, Inc., 2011.

[4] Marc Abeille and Alessandro Lazaric. Linear thompson sampling revisited. In Proceedings
of the 20th International Conference on Artificial Intelligence and Statistics, pages 176–184.
PMLR, 2017.

[5] Tanner Fiez, Lalit Jain, Kevin G Jamieson, and Lillian Ratliff. Sequential experimental design
for transductive linear bandits. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc,
E. Fox, and R. Garnett, editors, Proceedings of the 32nd Advances in Neural Information
Processing Systems, volume 32. Curran Associates, Inc., 2019.

[6] Marc Jourdan, Rémy Degenne, and Emilie Kaufmann. An ε-best-arm identification algorithm
for fixed-confidence and beyond, November 2023. arXiv:2305.16041.

[7] Rémy Degenne and Wouter M Koolen. Pure exploration with multiple correct answers. Pro-
ceedings of the 32nd Advances in Neural Information Processing Systems, 32, 2019.

[8] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with
non-stationary rewards. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q.
Weinberger, editors, Proceedings of the 27th Advances in Neural Information Processing
Systems, volume 27. Curran Associates, Inc., 2014.

[9] Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit
problems. In Jyrki Kivinen, Csaba Szepesvári, Esko Ukkonen, and Thomas Zeugmann, editors,
The 22nd International conference on Algorithmic learning theory, pages 174–188, Berlin,
Heidelberg, 2011. Springer Berlin Heidelberg.

[10] Yang Cao, Zheng Wen, Branislav Kveton, and Yao Xie. Nearly optimal adaptive procedure
with change detection for piecewise-stationary bandit. In Proceedings of the 22nd International
Conference on Artificial Intelligence and Statistics, pages 418–427. PMLR, 2019.

[11] Yueyang Liu, Xu Kuang, and Benjamin Van Roy. A definition of non-stationary bandits, Feb
2023. arXiv:2302.12202.

[12] Wang Chi Cheung, David Simchi-Levi, and Ruihao Zhu. Learning to optimize under non-
stationarity. In The 22nd International Conference on Artificial Intelligence and Statistics, pages
1079–1087. PMLR, 2019.

[13] Jonathan Shapiro, Carlos M Carvalho, and Pradeep Ravikumar. Thompson sampling in switch-
ing environments with bayesian online change point detection. In Proceedings of the 16th
International Conference on Artificial Intelligence and Statistics, pages 442–450. Microtome
Publishing, 2013.

[14] Zhihan Xiong, Romain Camilleri, Maryam Fazel, Lalit Jain, and Kevin Jamieson. A/B testing
and best-arm identification for linear bandits with robustness to non-stationarity. In Proceedings
of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238, pages
1585–1593. PMLR, 2023.

11


---Page Break---
[15] Robin Allesiardo, Raphaël Féraud, and Odalric-Ambrym Maillard. The non-stationary stochastic
multi-armed bandit problem. International Journal of Data Science and Analytics, 3:267–283,
2017.

[16] Masahiro Kato and Kaito Ariu. The role of contextual information in best arm identification,
2021. arXiv:2106.14077.

[17] Yoan Russac, Christina Katsimerou, Dennis Bohle, Olivier Cappé, Aurélien Garivier, and
Wouter M Koolen. A/B/n testing with control in the presence of subpopulations. Proceedings
of the 34th Advances in Neural Information Processing Systems, 34:25100–25110, 2021.

[18] Andrea Zanette, Kefan Dong, Jonathan N Lee, and Emma Brunskill. Design of experiments for
stochastic contextual linear bandits. Proceedings of the 34th Advances in Neural Information
Processing Systems, 34:22720–22731, 2021.

[19] Yasin Abbasi-Yadkori, András György, and Nevena Lazi´c. A new look at dynamic regret for
non-stationary stochastic bandits. Journal of Machine Learning Research, 24(288):1–37, 2023.

[20] Jia Yuan Yu and Shie Mannor. Piecewise-stationary bandit problems with side observations.
In Proceedings of the 26th International Conference on Machine Learning, ICML ’09, page
1177–1184. Association for Computing Machinery, 2009.

[21] Fang Liu, Joohyun Lee, and Ness Shroff. A change-detection based framework for piecewise-
stationary multi-armed bandit problem. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 32, 2018.

[22] Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard, and Julien Seznec. Efficient
change-point detection for tackling piecewise-stationary bandits. The Journal of Machine
Learning Research, 23(1):3337–3376, 2022.

[23] Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an
unknown number of distribution changes. In Proceedings of the 32nd Conference on Learning
Theory, pages 138–158. PMLR, 2019.

[24] Yassir Jedra and Alexandre Proutiere. Optimal best-arm identification in linear bandits. In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Proceedings of the
34th Advances in Neural Information Processing Systems, volume 33, pages 10007–10017.
Curran Associates, Inc., 2020.

[25] Peng Zhao, Lijun Zhang, Yuan Jiang, and Zhi-Hua Zhou. A simple approach for non-stationary
linear bandits. In The 23rd International Conference on Artificial Intelligence and Statistics,
pages 746–755. PMLR, 2020.

[26] Masahiro Kato, Masaaki Imaizumi, Takuya Ishihara, and Toru Kitagawa.
Asymptoti-
cally optimal fixed-budget best arm identification with variance-dependent bounds, 2023.
arXiv:2302.02988.

[27] Chao Qin and Daniel Russo. On adaptivity and confounding in contextual bandit experiments.
In NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications, 2021.

[28] Yasin Abbasi-Yadkori, Peter Bartlett, Victor Gabillon, Alan Malek, and Michal Valko. Best
of both worlds: Stochastic & adversarial best-arm identification. In Proceedings of the 31st
Conference on Learning Theory, pages 918–949. PMLR, 2018.

[29] Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[30] Kwang-Sung Jun, Kevin G Jamieson, Robert D Nowak, and Xiaojin Zhu. Top arm identification
in multi-armed bandits with batch arm pulls. In Proceedings of the 19th International Conference
on Artificial Intelligence and Statistics, pages 139–148, 2016.

[31] Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best arm
identification in multi-armed bandit models. Journal of Machine Learning Research, 17:1–42,
2016.

12


---Page Break---
[32] Marc Jourdan and Rémy Degenne. Choosing answers in epsilon-best-answer identification for
linear bandits. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang
Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine
Learning, volume 162 of Proceedings of Machine Learning Research, pages 10384–10430.
PMLR, 17–23 Jul 2022.

[33] Michael J. Todd. Minimum-Volume Ellipsoids: Theory and Algorithms. SIAM, 2016.

13


---Page Break---
Appendices

The contents of the appendices are organized as follows:

• In Appendix A, we provide further motivating examples for our problem.
• In Appendix B, we discuss the limitations of our method.
• In Appendix C, we review more related works on drifting and contextual bandits.
• In Appendix D, we provide more details about our algorithms
– Appendix D.1: pseudo-code of NεBAI in Algorithm 2.
– Appendix D.2: more details of PSεBAI including the precise definition of the confidence radius
ρt and details of LCD and LCA subroutines.
– Appendix D.3: pseudo-code of PSεBAI+ in Algorithm 5.
– Appendix D.4: the computational complexity of PSεBAI in Algorithm 1.
• In Appendix E: we provide a useful lemma for estimating the expected return of any arm in linear
bandits when the sampling rule is according to the G-optimal allocation.
• In Appendix F: we proof the upper bound on the expected sample complexity of NεBAI.
• In Appendices G to K: we detail the analysis of PSεBAI, i.e., we provide the proof of the upper
bound on its complexity in Theorem 3.2.
– Appendix G: outline of the proof.
– Appendix H: analysis of the Change Detection (CD) and Context Alignment (CA) procedures.
– Appendix I: analysis of the estimation error by decomposing it into three terms: Vector-
Estimation Error (VE), Distribution-Estimation Error (DE), and Residual Estimation Error
(RE).
– Appendix J: proof of Theorem 3.2 based on the analysis above.
– Appendix K: proof of technical lemmas that are utilized in the analysis of PSεBAI.
• In Appendix L: we prove the upper bound on the expected sample complexity of PSεBAI+ based
on the analysis of NεBAI and PSεBAI.
• In Appendix M: we derive the lower bound on the expected sample complexity of any algorithm,
i.e., proof Theorem 4.1.
• In Appendix N: we provide more examples to compare the derived upper and lower bounds on the
expected sample complexity, and illustrate the efficacy of our PSεBAI+ algorithm.
• In Appendix O: we provide more details of numerical experiments.
• In Appendix P: we provide more discussions on
– the related methods for BAI in nonstationary bandits,
– the instance-dependent upper bound,
– the connection between the piecewise-stationary linear bandits model to the stationary linear
bandits model,
– the special case where N = 1.
• In Appendix Q: we provide analytical results on the “Best Arm Tuple Identification Problem”.

A
Further Motivating Examples

We elaborate on the some concrete real-life examples that motivate our problem setup of identifying
the ensemble best arm in piecewise-stationary linear bandits.

In scenarios such as investment option selection and portfolio management also mentioned by [10, 20],
there is a multitude of options for fund managers to choose from and typically, they want to find,
in the initial pure exploration process, a small subset of candidate portfolios (or even the “best”
portfolio) based on various economic indicators and the market performance of individual stocks
before further exploitation. In a bearish market, more portfolios tend to incur losses; while in a bullish
market, more portfolios tend to generate gains. The transition between these two contexts can be
effected by stochastic factors, e.g., the weather, or the outbreak of a pandemic, making the market
conditions (contexts) stochastic. In the face of these uncertainties (in the contexts and rewards), we
wish to design and analyze algorithms that selected portfolio to yield the best long-term option under
such a piecewise-stationary environment.

Crop rotation is another example. Since crop yields can be influenced by various factors, such as
weather conditions (analogous to our stochastically generated contexts), selecting the most suitable
crop to grow and harvest from is crucial. Given several candidate crops, crops of similar types (e.g.,
potatoes and sweet potatoes) are correlated as they tend to favor similar conditions, thus, they can be

14


---Page Break---
modelled by bandits with a linear reward structure. Contextual factors, like weather conditions, are
well-modelled as being stochastic. A fixed weather condition will last a period of time and it will not
change suddenly. Domain knowledge from historical data/records provides us with prior knowledge
on Lmin and Lmax. These observations dovetail with our model. Our objective is to choose the crop
that offers the near-highest yield potential over a long time period (an ensemble ε-best arm) and is
adaptable to local environment factors, such as weather patterns.

B
Limitations

Similar to existing works on piecewise stationary bandits [21, 10, 22], we introduced some assump-
tions to provide theoretical guarantees for PSεBAI+ although PSεBAI+ has shown to be robust even
in the absence of these assumptions. Since an algorithm may not need to differentiate two contexts
with close latent vectors for identifying an ε-optimal arm, we surmise it is possible to weaken these
assumptions. For this purpose, we will consider clustering the contexts into a few classes based on
the distances between their latent vectors and design an algorithm that only aims to detect the change
of context class, instead of the change of context.

C
More Related Work

In drifting bandits (DB), the regrets of algorithms are affected by the level of nonstationarity of the
environment, which can be measured by various quantities, such as the total variation of the context
sequence and the number of time steps when the return of at least one arm changes [23, 25].

In contextual bandits (CB), where the contextual information is visible to the agent, [16, 26, 17, 27]
aim to identify the best arm with the assumption that the context changes at every time step according
to a fixed distribution, and the return of an arm is averaged across all contexts. We see that the context
distribution is involved to measure the quality of arms. However, while the agent in CB models can
observe the context information, the agent has no access to the contexts but still aims to identify an
ε-optimal arm in our piecewise stationary linear bandit (PSLB) model.

In adversarial bandits, existing works pertaining to the BAI problem only explored the fixed-horizon
setting [28]. Due to the difference between the fixed-confidence and fixed-horizon setting and the
difference between adversarial and piecewise stationary bandits, their results cannot be trivially
extended to solve the fixed-confidence BAI problem in piecewise stationary bandits.

D
More Details of Algorithms NεBAI, PSεBAI and PSεBAI+

All proposed algorithms make use of the well known G-optimal allocation (design) [1, 29], which
is widely used in the linear bandits literature. The G-optimal allocation minimizes the maximal
mean-squared prediction error in all directions [1]. Given an arm set X, the G-optimal allocation
λ∗is a distribution over the arm set, which is the minimizer of g(λ) = maxx∈X ∥x∥2
A(λ)−1, where
A(λ) = P

x∈X λ(x)xx⊤and λ ∈∆X . Interested readers may refer to Chapter 21 in [29].

D.1
The NAÏVE ε-BEST ARM IDENTIFICATION (NεBAI) Algorithm

As presented in Algorithm 2, NεBAI samples an arm with the G-optimal allocation at each time
steps; its stopping rule is grounded on the property of G-optimal allocation (see Lemma E.1) and
affected by the maximum length of a single stationary segment Lmax.

15


---Page Break---
Algorithm 2 NAÏVE ε-BEST ARM IDENTIFICATION (NεBAI)

1: Input: the arm set X, the phase length bounds Lmin, Lmax, the slackness parameter ε, the
confidence parameter δ.
2: Initialize: Compute the G-optimal allocation λ∗.
3: Compute C3 = P∞
n=1 n−3 and t∗= 3d ln(6dKC3/δ).
4: Sample t∗arms {xs}t∗
s=1 ∼λ∗and observe the associated returns {Ys,xs}t∗
s=1. Let t = t∗.
5: Compute

˜θt = 1

t

t
X

s=1
A(λ∗)−1xsYs,xs,
˙xt = arg max
x∈X
x⊤˜θt,

˜ρt =

r

8Lmax

t
ln 4KC3t3

δ
+ 5

r

d

t ln 4KC3t3

δ
.

(D.1)

6: while ˙x⊤
t ˜θt −˜ρt + ε < maxx̸= ˙xt x⊤˜θt + ˜ρt do
7:
Sample an arm xt ∼λ∗and observe return Yt,xt and let t = t + 1.
8:
Update ˜θt, ˙xt and ˜ρt with (D.1).
9: end while
10: Recommend arm ˙xε = ˙xt.

D.2
Details about PSεBAI

D.2.1
Confidence radius utilized in PSεBAI

For each pair of arms (x, ˜x), the confidence radius of ∆(x, ˜x) at time step t is

ρt(x, ˜x) := 2(αt + ξt) +

N
X

j=1
βt,j| ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)|,

where aclip2 := min{max{a, −2}, 2} denotes the value of a that is clipped to the interval [−2, 2]
and

αt := 5

s

d
Tt
ln
2
δv,Tt
, ξt := 25
√

2NLmax

Tt
ln
2
δm,Tt
,

βt,j := min
5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
, 1

,

ϕt,j := min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4


,

δv,Tt =
δ
15KT 3
t
, δm,Tt =
δ
15KNT 3
t
, δd,Tt =
δ
15NT 3
t
.

ζt(x, ˜x) minimizes the last summation. In the final theoretical upper bound on the sample complexity,
we take ζt(x, ˜x) = ε for simplicity. In the experiment, we utilize

ζt(x, ˜x) = arg min
ζt(x,˜x)∈R

N
X

j=1
βt,j( ˆ∆t,j(x, ˜x) + ζt(x, ˜x))2 = −

PN
j=1 βt,j ˆ∆t,j(x, ˜x)

PN
j=1 βt,j

for a simple and effective analytic expression in the experiment.

D.2.2
Subroutines of PSεBAI

In the LCD subroutine (Algorithm 3), two estimates ˜θ1 and ˜θ2 of the latent vectors are independently
computed from the first and second halves of the input CD samples (Line 2). LCD only raises an
alarm of changepoint (Line 4) when the difference between ˜θ1 and ˜θ2 is sufficiently large (Line 3)
and indicates a changepoint occurs w.h.p.

The LCA subroutine (Algorithm 4) estimates the latent vector of current context and updates the
dictionary CAid. Specifically, it firstly samples w/2 samples with λ∗(Line 2). It then checks

16


---Page Break---
Algorithm 3 LINEAR-CHANGE DETECTION (LCD)

1: Input: arm set X, detection window w, threshold b and w arms with the associated observations
[(˜x1, Ys,˜x1), . . . , (˜xw, Yw,˜xw)].

2: Compute ˜θ1 = 2

w
P w

2
s=1 A(λ∗)−1˜xsYs,˜xs and ˜θ2 = 2

w
Pw
s= w

2 +1 A(λ∗)−1˜xsYs,˜xs.

3: if ∃x ∈X, s.t.|x⊤(˜θ2 −˜θ1)| > b then
4:
Return True
5: else
6:
Return False
7: end if

Algorithm 4 LINEAR-CONTEXT ALIGNMENT (LCA)

1: Input: arm set X, detection window w, threshold b and context ids CAid.

2: Sample w

2 arms {˜xs}

w

2
s=1 ∼λ∗and observe the associated returns {Ys,˜xs}

w

2
s=1 , where

{(xs, Ys,xs)}t
s=t−w

2 +1 = {˜xs, Ys,˜xs}

w

2
s=1.
3: for j ∈CAid
do

4:
if not LCD(X, w, b, [(˜xs, Ys,˜xs)]

w

2
s=1 + CAid[j]) then

5:
CAid[j] = [(˜xs, Ys,˜xs)]

w

2
s=1.
6:
Return j, CAid.
7:
end if
8: end for
9: index = len(CAid) + 1.

10: CAid[index] = [(˜xs, Ys,˜xs)]

w

2
s=1.
11: Return index, CAid.

whether the current latent context has been visited in previous time steps by scanning through CAid
(Line 3). In order to learn if the current context can be aligned with context j, the LCD subroutine is
called with the w/2 recent samples and CAid[j] as input:
(i) If the current context is aligned with CAid[j] (Line 4), the current latent context is j w.h.p. Thus
the LCA subroutine updates CAid[j] and returns the index j (Lines 5 to 6), which would be ˆjt in
Line 19 of Algorithm 1. Note that in Lines 10 and 11 of Algorithm 1, all the collected samples will
be assigned to context ˆjt (until the next changing alarm) and will be used to estimate θ∗
ˆjt and all pj’s.
Therefore, aligning contexts allows PSεBAI to make good use of observation history.
(ii) If the current context is not aligned with any observed context, i.e., it has not been visited, CAid
gets extended with a new index-samples pair and is returned along with the new index (Lines 9 to 11).

17


---Page Break---
D.3
The PIECEWISE-STATIONARY ε-BEST ARM IDENTIFICATION+ (PSεBAI+) Algorithm

To help understand PSεBAI+ algorithm, we highlight the differences between PSεBAI and
PSεBAI+, and the differences between subroutines LINEAR-CONTEXT ALIGNMENT (LCA) and
LINEAR-CONTEXT ALIGNMENT+ (LCA+).

Algorithm 5 PIECEWISE-STATIONARY ε-BEST ARM IDENTIFICATION+ (PSεBAI+)

1: Input: arm set X, number of latent vectors N, bounds on the segment lengths Lmin and Lmax,
slackness parameter ε, confidence parameter δ, sampling parameter γ, window size w and
threshold b.
2: Initialize: Compute the G-optimal allocation λ∗and τ ∗= 38400 ln(80)NLmax

ε2
ln N 2KLmax

δε2
.

3: Set CDsample = [ ], CAid = { }, tCD = +∞. Set Tt,j = ∅and initialize Tt, Tt,j, Tt with (3.2)
for all t ≤τ ∗, j ∈[N].
4: Compute C3 = P∞
n=1 n−3 and t∗= 3d ln(6dKC3/δ).

5: Sample w/2 arms {xs}w/2
s=1 ∼λ∗and observe the associated returns {Ys,xs}w/2
s=1.

6: Set t = w

2 , tCA = w/2, CAid = {1 : [(xs, Ys,xs)]w/2
s=1}, ˆjt = 1.

7: Set ˙xt = 0, ˜ρt = 2ε and ˙xε = −∞. Compute ˜θt with (D.1).
8: while True do
9:
t = t + 1
10:
Sample an arm xt ∼λ∗and observe return Yt,xt.
11:
Compute ˜θt, ˙xt, ˜ρt = EU(t, ˜θt−1, t∗, C3).
12:
if ˙x⊤
t ˜θt −˜ρt + ε ≥maxx̸= ˙xt x⊤˜θt + ˜ρt then
13:
˙xε = ˙xt.
14:
Break
15:
end if
16:
if t > τ ∗then
17:
Continue
18:
end if
19:
if
mod (t −tCA, γ) ̸= 0 then
20:
Update ˆjt = ˆjt−1, Tt,ˆjt = Tt−1,ˆjt ∪{t}, Tt,j = Tt−1,j for j ̸= ˆjt.
21:
else
22:
CDsample = CDsample + [(xt, Yt,xt)].
23:
if |CDsample| ≥w then
24:
if LCD(X, w, b, CDsample[−w : ]) then
25:
CDsample = [ ].
26:
t = t + w

2 , tCA = t, tCD = +∞.

27:
ˆjt, CAid = LCA+(X, w, b, CAid)
28:
if ˆjt = N + 1 then break.
29:
Revert Tt,j = Tt−w(γ+1)

2
,j for all j ∈[N].

30:
end if
31:
end if
32:
end if
33:
if ˙xε ̸= −∞then
34:
Break
35:
end if
36:
Update the empirical estimates by (3.2) and (3.3).
37:
if condition (3.4) is met and tCD = +∞then
38:
Record ˆxε = arg maxx∈X x⊤ˆΘtˆpt.
39:
tCD = |CDsample|.
40:
else if tCD = |CDsample| −w

2 then
41:
˙xε = ˆxε
42:
Break
43:
end if
44: end while
45: Recommend ˚xε = ˙xε.

18


---Page Break---
Algorithm 6 ESTIMATE UPDATE (EU)

1: Input: time step t, vector ˜θt−1, threshold t∗and constant C3.

2: Compute ˜θt = 1

t

h
(t −1)˜θt−1 + A(λ∗)−1xtYs,xt
i
, ˙xt = 0, ˜ρt = 2ε.

3: if t ≥t∗then

4:
Compute ˙xt = arg max
x∈X
x⊤˜θt, ˜ρt =

r

8Lmax

t
ln 4KC3t3

δ
+ 5

r

d

t ln 4KC3t3

δ
.

5: end if
6: Return ˜θt, ˙xt, ˜ρt.

Algorithm 7 LINEAR-CONTEXT ALIGNMENT+ (LCA+)

1: Input: arm set X, detection window w, threshold b, context ids CAid, time step t, vector ˜θt,
threshold t∗and constant C3.
2: Set s = 0, ˙xε = −∞.
3: for s ≤w/2 do
4:
s = s + 1, t = t + 1.
5:
Sample an arm ˜xs ∼λ∗and observe return Ys,˜xs.
6:
Set (xt, Yt,xt) = (˜xs, Ys,˜xs).
7:
Compute ˜θt, ˙xt, ˜ρt = EU(t, ˜θt−1, t∗, C3).
8:
if ˙x⊤
t ˜θt −˜ρt + ε ≥maxx̸= ˙xt x⊤˜θt + ˜ρt then
9:
˙xε = ˙xt, index = len(CAid) + 1.
10:
Return index, CAid, ˙xε, ˜θt.
11:
end if
12: end for
13: for j ∈CAid do

14:
if not LCD(w, b, [(˜xs, Ys,˜xs)]

w

2
s=1, CAid[j]) then

15:
CAid[j] = [(˜xs, Ys,˜xs)]

w

2
s=1.
16:
Return j, CAid.
17:
end if
18: end for
19: index = len(CAid) + 1.

20: CAid[index] = [(˜xs, Ys,˜xs)]

w

2
s=1.
21: Return index, CAid, ˙xε, ˜θt.

D.4
Computational Complexity of PSεBAI

We provide analysis for the computational complexity of PSεBAI in this subsection.

We consider the number of these operations: arithmetic (addition and multiplication) operations, logic
operations and comparison operations and we also regard ln(·), √· and sampling from a distribution
as one step of operation.

We decompose the main loop of Algorithm 1 as follows, where the lines with O(1) operations are
omitted:

• Exploration phase (Lines 8 to 11): O(1).
• Change Detection phase (Lines 12 to 16):
– The LCD subroutine in Line 16 needs O(wd2 + Kd) operations.
• Context Alignment phase (Lines 17 to 21):
– The LCA subroutine in Line 19 needs O((wd2 + Kd)N) operations, as the LCD subroutine
will be invoked N times in the worst case in Line 4 of LCA.
– The reversion procedure in Line 21 needs O(γwd) operations.
• The updating procedure and the stopping rule checking (Lines 25 to 32):
– Updating ˆθt,j needs O(d2) operations, as we need to incorporate the latest sample into the
estimate.

19


---Page Break---
– Updating ˆpt,j, j ∈[N] requires N operations.
– Updating the confidence radius and the empirical best arm need O(KN) operations.
– The stopping condition in Equation (3.4) requires O(K) operations.

We remark that some intermediate results can be stored to avoid repeated computations and we
believe the algorithm is efficient overall. In particular, since the reward structure is linear, there are
K arms, and N contexts, O(d2), O(K), and O(N) operations probably cannot be be avoided.

From the above analysis, in the decreasing order of the number of operations:

• The Change Detection phase requires the most budget as it will be invoked every γ time steps.
• The Context Alignment phase requires the second many operations. While it requires many
operations when it is implemented, it will not be called during a stationary segment.
• The updating procedure uses small portion of the operations, as w is usually much greater than K
and N.
• The exploration phase demands constant operations in each loop.

E
Auxiliary results

Lemma E.1. Let Xs be the arm drawn with the G-optimal allocation λ∗at time step s and Ys,xs =
x⊤
s θ∗
js + ηs be the corresponding return with Eηs = 0, |x⊤
s θ∗
js| ≤1 and |ηs| ≤1. Then we have

P

"
1
n



n
X

s=1
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js

 ≥ϵ

#

≤δ

for all ϵ ≥d

n ln 2

δ +
q  d

n ln 2

δ
2 + 4d

n ln 2

δ . In addition, if n ≥d

4 ln 2

δ , we have

P

"
1
n



n
X

s=1
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js

 ≥5

r

d
n ln 2

δ

#

≤δ

Proof. Note that the arms are selected according to the G-optimal design, and thus Ex∼λ∗[xx⊤] =
A(λ∗). Therefore, for all s ∈[n],

E

x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js

= 0
(E.1)

and
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js

(E.2)

≤
x⊤A(λ∗)−1xsx⊤
s θ∗
js
 +
x⊤A(λ∗)−1xsηs
 + 1

≤
x⊤A(λ∗)−1xs
x⊤
s θ∗
js
 +
x⊤A(λ∗)−1xs
ηs
 + 1

≤2
x⊤A(λ∗)−1xs
 + 1

≤2∥x∥A(λ∗)−1∥xs∥A(λ∗)−1 + 1

≤3d

where we make use of the property of the G-optimal allocation in the last inequality

∥x∥2
A(λ∗)−1 ≤d, ∀x ∈X.
(E.3)

Additionally,

E
h 
x⊤A(λ∗)−1xsYs,xs
2i
(E.4)

= E
h 
x⊤A(λ∗)−1xs(x⊤
s θ∗
js + ηs)
2i

= E
h 
x⊤A(λ∗)−1xsx⊤
s θ∗
js
2i
+ E
h 
x⊤A(λ∗)−1xsηs
2i

+ 2E

x⊤A(λ∗)−1xsx⊤
s θ∗
js · x⊤A(λ∗)−1xsηs


(a)
= E
h 
x⊤A(λ∗)−1xs
2  
x⊤
s θ∗
js
2i
+ E
h
η2
s
 
x⊤A(λ∗)−1xs
2i

20


---Page Break---
(b)
≤2E
h 
x⊤A(λ∗)−1xs
2i

(c)
= 2 x⊤A(λ∗)−1x

(d)
≤2d
where we make use of the fact that ηt is zero-mean and is independent of other random variables
in (a); |x⊤
s θ∗
js| ≤1 and P

η2
s ≤1

= 1 in (b); xs ∼λ∗in (c); and the property of the G-optimal
allocation (E.3) in (d).

According to the Bernstein’s inequality with (E.1), (E.2) and (E.4),

P

"
1
n



n
X

s=1
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js
 ≥ϵ

#

≤2 exp

−

1
2(nϵ)2

n · 2d + dnϵ



= 2 exp

−
nϵ2

4d + 2dϵ


.

In order to upper bound the error probability by δ, we need

2 exp

−
nϵ2

4d + 2dϵ


≤δ

⇒
ϵ ≥d

n ln 2

δ +

s d

n ln 2

δ

2
+ 4d

n ln 2

δ .

In addition, if n ≥d

4 ln 2

δ , we have

5

r

d
n ln 2

δ = 5

2 max

(
d
n ln 2

δ ,

r

4d

n ln 2

δ

)

≥d

n ln 2

δ +

s d

n ln 2

δ

2
+ 4d

n ln 2

δ .

This finishes the proof.

F
Analysis of NεBAI

To analyze the theoretical performance of NεBAI, we first show that it can identify an ε-optimal arm
with probability 1 −δ and derive a high-probability upper bound on its stopping time in Lemma F.1.

Lemma F.1 (High-probability upper bound of NεBAI). With probability 1 −δ, the NεBAI algorithm
identifies an ε-optimal arm after at most

˜O

 
Lmax + d

(ε + ∆min)2 ln 1

δ

!

time steps, where ∆min = min
i̸=i∗∆(x∗−xi).

Next, we prove that after a sufficiently large number of time steps, the probability that NεBAI does
not terminate is small in Lemma F.2.

Lemma F.2. Let

T0 = 768(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ

with C3 = P∞
n=1 n−3. For t ≥T0, the probability that NεBAI does not terminate before t time
steps is
δ
(α−1)C3(t/2)2 .

Lastly, we apply Lemmas F.1 and F.2 to prove Proposition 3.1.

21


---Page Break---
Proposition 3.1. Let ∆min = minx̸=x∗∆(x∗, x),

T N
V =
d

(ε + ∆min)2 ln 1

δ
and
T N
D =
Lmax
(ε + ∆min)2 ln 1

δ .

The NεBAI algorithm is (ε, δ)-PAC and its expected sample complexity is ˜O(T N
V + T N
D ).

The detailed analysis is presented as below.

F.1
Detailed analysis of NεBAI

Proof of Lemma F.1. The result is to be proven with the following procedures.

First step: Prove
x⊤Eθ∼Pθθ −x⊤˜θt
 can be bounded with high probability.

Let ¯θt = 1

t
Pt
s=1 θ∗
js. (i) For all ε > 0, we have,

P
 x⊤Eθ∼Pθθt −x⊤¯θ
 > ε

= P

" tx⊤Eθ∼Pθθ −x⊤
 Nt
X

l=1
Llθ∗jcl

! > tε

#

= P

" 

Nt
X

l=1
Ll
 
x⊤Eθ∼Pθθ −Llx⊤θ∗jcl

 > tε

#
(a)
≤2 exp

 

−
t2ε2

2 PNt
l=1(2Ll)2

!

(b)
≤2 exp

−
tε2

8Lmax


.

Since |x⊤Eθ∼Pθθ| ≤1 and x⊤θ∗jcl| ≤1 for all l, we obtain (a) by applying Hoeffding’s inequality.
We obtain (b) from the fact that PNt
l=1 Ll = t and Ll ≤Lmax for all l. In other words, for all δ > 0,

P

"
x⊤Eθ∼Pθθ −x⊤¯θ
 >

r

8Lmax

t
ln 2

δ

#

≤δ.

(ii) Besides, according to Lemma E.1, if t ≥d

4 ln 2

δ , we have

P

"x⊤˜θt −x⊤¯θt
 ≥5

r

d

t ln 2

δ

#

= P

"
1

t



t
X

s=1
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
js

 ≥5

r

d

t ln 2

δ

#

≤δ.

(iii) For all t > 0, all x ∈X, since
x⊤Eθ∼Pθθ −x⊤˜θt
 ≤
x⊤Eθ∼Pθθ −x⊤¯θ
 +
x⊤˜θt −x⊤¯θt
 ,

we have

P
hx⊤Eθ∼Pθθ −x⊤˜θt
 > ˜ρt(α)
i
≤
δ
KCαtα
when
t ≥d

4 ln 4KCαtα

δ
,

where

˜ρt(α) =

r

8Lmax

t
ln 4KCαtα

δ
+ 5

r

d

t ln 4KCαtα

δ
,
Cα =

∞
X

n=1
n−α,
α > 2.

For simplicity, we set α = 3 and write ˜ρt(3) as ˜ρt in the following analysis. We now show that
t ≥d

4 ln 4KCαtα

δ
holds when t ≥3d ln 6dKC3

δ
with the following lemma (the proof of which is
presented by the end of Appendix F.1).

Lemma F.3. Let a, b, c > 0 and ac ≥1/2, then

t ≥max{2a ln b, 4ac ln(2ac)} ⇒t ≥a ln(btc).

With

a = d

4,
b = 4KCα

δ
= 4KC3

δ
,
c = α = 3,

22


---Page Break---
Lemma F.3 implies that

t ≥4ac ln(2abc) = 3d ln 6dKC3

δ
⇒t ≥max{2a ln b, 4ac ln(2ac)} ⇒t ≥a ln(btc).

Define event ˜Et = T

x∈X

nx⊤Eθ∼Pθθ −x⊤˜θt
 ≤˜ρt(α)
o
for t ≥t∗and ˜E = T

t≥t∗
˜Et. We have

P
h
˜Ec
t
i
≤
X

x∈X

δ
KCαtα =
δ
Cαtα ,
P
h
˜Eci
≤

∞
X

t=t∗
P
h
˜Ec
t
i
≤

∞
X

t=1

δ
Cαtα = δ.

We assume ˜E holds in the second and third steps in this proof.

Second step: Prove NεBAI recommends an ε-optimal arm when it stops.

If the algorithm terminates and returns ˙xε ̸= x∗, we have

˙x⊤
ε ˜θt −˜ρt + ε ≥max
x̸= ˙xε x⊤˜θt + ˜ρt.

Since

˙x⊤
ε Eθ∼Pθθ + ε ≥˙x⊤
ε ˜θt −˜ρt + ε
and
max
x̸= ˙xε x⊤˜θt + ˜ρt ≥max
x̸= ˙xε x⊤Eθ∼Pθθ ≥x∗⊤Eθ∼Pθθ,

we have

˙x⊤
ε Eθ∼Pθθ + ε ≥x∗⊤Eθ∼Pθθ,

implying that ˙xε ∈Xε. Hence, we always have ˙xε ∈Xε.

Third step: Derive the stopping time of NεBAI.

(i) Let x(2) = arg maxx̸=x∗x⊤Eθ∼Pθθ and ∆min = minx̸=x∗∆(x∗−x) = (x∗−x(2))⊤Eθ∼Pθθ.
We apply the following lemma to show that NεBAI stops when ˜ρt is sufficiently small.

Lemma F.4 ([30, Lemma 3]). Denote by ˆi the index of the item with empirical mean is i-th largest:
i.e., ˆw(ˆ1) ≥. . . ≥ˆw(ˆL). Assume that the empirical means of the arms are controlled by ε : i.e.,
∀i, | ˆw(i) −w(i)| < ε. Then,

∀i, w(i) −ε ≤ˆw(ˆi) ≤w(i) + ε.

Lemma F.4 implies that

˙x⊤
t ˜θt −˜ρt + ε ≥x∗⊤Eθ∼Pθθ −2˜ρt + ε = x⊤
(2)Eθ∼Pθθ −2˜ρt + ε + ∆min
and

max
x̸= ˙xt x⊤˜θt + ˜ρt ≤x⊤
(2)Eθ∼Pθθ + 2˜ρt.

When ˜ρt ≤(ε + ∆min)/4, we have

˙x⊤
t ˜θt −˜ρt + ε ≥x⊤
(2)˜θt + ε + ∆min

2
≥x⊤
(2)˜θt + 2˜ρt ≥max
x̸= ˙xt x⊤˜θt + ˜ρt,

which will lead to the termination of NεBAI.

(ii) According to the definition of ˜ρt, we have

˜ρt(α) ≤(ε + ∆min)/4

⇔

r

8Lmax

t
ln 4KCαtα

δ
+ 5

r

d

t ln 4KCαtα

δ
≤(ε + ∆min)/4

⇔

r

1

t ln 4KCαtα

δ

p

8Lmax + 5
√

d

≤(ε + ∆min)/4

⇔t ≥

√8Lmax + 5
√

d
2

(ε + ∆min)2 /16
ln 4KCαtα

δ

⇐t ≥32(8Lmax + 25d)

(ε + ∆min)2
ln 4KCαtα

δ

23


---Page Break---
⇐t ≥192(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ
.

We apply Lemma F.3 to invert the last line above. With

a = 32(8Lmax + 25d)

(ε + ∆min)2
,
b = 4KCα

δ
,
c = α = 3,

Lemma F.3 indicates that

t ≥4ac ln(2abc) = 384(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ
⇒t ≥max{2a ln b, 4ac ln(2ac)} ⇒t ≥a ln(btc).

Altogether, we show that with probability 1 −δ, NεBAI identifies an ε-optimal arm after at most

384(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ
time steps.

Proof of Lemma F.2. To begin with, let

T0 = 768(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ

For any T ≥T0, let ¯T = ⌈T/2⌉.

(I) If NεBAI terminates within ¯T time steps, there is nothing to prove.

(II) Assume NεBAI does not terminates within ¯T time steps. According to the analysis in the proof
of Lemma F.1, if NεBAI does not terminate before T time steps, i.e., the stopping time τ satisfies

that τ ≥¯T, then event
T −1
T

t= ¯T +1
˜Et does not hold. This indicates

P

τ ≥T|τ ≥¯T

≤P




T −1
[

t= ¯T +1

˜Ec
t



≤

T −1
X

t= ¯T
P
h
˜Ec
t
i
≤

T −1
X

t= ¯T

δ
Cαtα ≤δ

Cα

Z T

¯T
t−α dt

≤
δ
(α −1)Cα


1
( ¯T)α−1 −1

T α

 (a)
≤
δ
(α −1)Cα(T/2)α−1 .

(a) results from the fact that ¯T ≥T/2.

Altogether, for T ≥T0, the probability NεBAI does not terminate before T time steps is
δ
(α−1)Cα(T/2)α−1 .

Proof of Proposition 3.1. First, Tonelli’s Theorem implies that

E[τ] = E
Z τ

0
1 dx

= E
Z +∞

0
I(τ > x) dx

=
Z +∞

0
E [I(τ > x)] dx =
Z +∞

0
P(τ > x) dx.

Next, we apply Lemma F.2 to show

Eτ ≤T0 + E[τ|τ ≥T0 + 1] · P[τ ≥T0 + 1] ≤T0 +
Z +∞

T0
P(τ ≥x) dx

≤T0 +
Z +∞

T0

δ
(α −1)Cα(x/2)α−1 dx = T0 +
δ
(α −1)(α −2)(T0/2)α−2 .
(F.1)

Besides, Lemma F.1 indicates that NεBAI can identify an ε-optimal arm with probability 1 −δ.

24


---Page Break---
Proof of Lemma F.3. For x > 0, let

f(x) = x −a ln(bxc) = x −a ln b −ac ln x = x

1 −a ln b

x
−ac ln x

x


.

Since

f(x) ≥0 ⇐










a ln b

x
≤1

2
ac ln x

x
≤1

2

⇔





x ≥2a ln b

x ≥2ac ln x
.

Let d = 2ac, x = d ln(dy), then

x ≥2ac ln x = d ln x ⇔d ln(dy) ≥d ln(d ln(dy)) ⇔ln(dy) ≥ln(d ln(dy)) ⇔y ≥ln(dy).

Since z0.4 ≥ln z for all z ≥1, we have ln(dy) ≤(dy)0.4 ≤√dy and

y ≥ln(dy) ⇐y ≥
p

dy ⇐y ≥d
when yd ≥1. Hence, when ac ≥1/2, y ≥d = 2ac ≥1, we have y ≥ln(dy). Furthermore, for x
such that x ≥max{2a ln b, 4ac ln(2ac)}, we have f(x) ≥0.

G
Proof Outline of Theorem 3.2

A proof outline of Theorem 3.2 is provided in this section. It consists of three steps:
Step 1: PSεBAI (Algorithm 1) depends on the success of change detection and context alignment
(Algorithm 3 and 4). We firstly upper bound the failure probability of these two subroutines via
Lemma G.1, Lemma G.2, Lemma G.3 and summarized in Lemma G.4. More details about these
two subroutines are provided in Appendix D.2.2 and the proof of the Lemmas are postponed to
Appendix H.
Step 2: Subsequently, conditioned on their success, we provide a theoretical guarantee for the choice
of the confidence radius ρt(x, ˜x) for ∆(x, ˜x) at time step t in Lemma G.5. The proof is detailed in
Appendix I.
Step 3: Lastly, utilizing the above elements, we provide a sufficient condition for the stopping rule,
and upper bound the number of time steps in Exp phases Tτ via Lemma G.6 whose proof is presented
in Appendix J. As the total number of time steps τ is upper bounded by a constant multiple of Tτ,
the high-probability upper bound on the stopping time τ is obtained. This finishes the proof of
Theorem 3.2 (please refer to Appendix J).

Step 1: We borrow the terminology in hypothesis testing to define the errors. Let the null hypothesis
be: the algorithm has undergone a changepoint within the last w CD samples. For Algorithm 3, we
will characterize the type I error: the algorithm has experienced a changepoint within the last w CD
samples but it fails to raise a changing alarm, and the type II error: a changepoint has not occurred
but Algorithm 3 raises a changing alarm. We refer to the event Failed Alarm (FA) and the event False
Alarm Error (FAE) as that a type I error occurs and that a type II error occurs, respectively:

FAl := {a type I error occurs at changepoint cl}, l ∈N,
FAEt := {a type II error occurs time step t}, t ∈TFAE,

FA :=
[

{l:cl≤τ}
FAl
and
FAE :=
[

t∈TFAE
FAEt,

where TFAE := {t : t in CD phase, t ≤τ, [t −wγ, t] ∩C = ∅} and τ is the stopping time. In
term of Algorithm 4, define the event MIl := {Algorithm 4 misidentifies θ∗
jt}, l ∈N and MI :=
S

{l:cl≤τ} MIl. Define the good event

Good := {FAc ∩FAEc ∩MIc}.
(G.1)

We upper bound the failure probability of Goodin Lemma G.1, Lemma G.2, Lemma G.3 and conclude
the results in Lemma G.4.

25


---Page Break---
Lemma G.1. For any δFAE ∈(0, 1), with b ≥8d

3w ln
2
δFAE +

r
8d
3w ln
2
δFAE

2
+ 24

w d ln
2
δFAE , LCD

makes no false alarm before the stopping time τ with probability at least

P[FAEc] ≥1 −τ

γ KδFAE.

Lemma G.1 can also be stated as, fix b,

P [FAEc] ≥1 −τ

γ 2K exp

−min
3wb

20d , wb2

150d


,

which indicates we can always decrease the error probability by enlarging the window size w.

Assume cl is observable, i.e., θ∗
jcl ̸= θ∗
jcl+1 , and denote ˆcl as the alarm time of cl from Algorithm 3.

Lemma G.2. Conditional on FAEc, for any δFA
∈
(0, 1), if
∆c−b

2
≥
d
w ln
2
δFA +
r
d
w ln
2
δFA

2
+ 4d

w ln
2
δFA ,

P
h
cl ≤ˆcl ≤cl + wγ

2

ˆcl ≥cl
i
≥1 −δFA.

In addition, P

FA
FAEc
≤lτδFA.

Lemma G.2 guarantees a prompt change alarm will be raised within wγ

2 time steps after a changepoint
occurs. This also explains why wγ

2 samples are abandoned at line 16 of Algorithm 1 so that Tt,j only
keeps samples from context j.
Lemma G.3. Conditional on FAEc and FAc, based on the conditions in Lemma G.1 and Lemma G.2,

P [MI|FAEc, FAc] ≤lτ [(N −1)δFA + KδFAE] .

The intuition of Lemma G.3 is simple: change alarm should not be raised when the samples are from
the same context and, change alarms should be raised when the samples are from the other (N −1)
contexts. The error made by Algorithm 3 and 4 can be concluded as follows
Lemma G.4. Assume the instance satisfies Assumption 1,

P [Good] ≥1 −
δ
2τ ∗≥1 −δ

2.

Step 2: To give an upper bound on Tτ, we firstly prove that the estimate of ∆(x, ˜x) at time t is within
distance ρt(x, ˜x) from the ground truth w.h.p. Define the event CI as the estimates of all the mean
gaps locate in the high-probability Confidence Interval

CIt := {
 ˆ∆t(x, ˜x) −∆(x, ˜x)
 ≤ρt(x, ˜x), ∀x, ˜x ∈X}

and CI := T

t CIt, where ρt is defined in (3.3).

Lemma G.5. If Tt ≥2Lmax

9
ln
2
δd,Tt ,

P

CIt
Good

≥1 −(Kδv,Tt + Nδd,Tt + KNδm,Tt) .

In addition,

P

CI
Good

≥1 −δ

2.

In addition to an estimation over the latent vector which is sufficient under the stationary case (αt
in ρt), we also need to control the derivation between the empirical distribution ˆpt and the ground
truth p (βt in ρt), as well as the interactions between the latent vectors and the distribution (ξt in ρt).
This is reflected by Kδv,Tt, Nδd,Tt and KNδm,Tt which bound the the failure probability of Vector
Estimation, Distribution Estimation and Residual Estimation, respectively.

Step 3:

Lemma G.6. Conditional on Good and CI, the recommended arm ˆxε ∈Xε and when Algorithm 1
terminates, the order of Tt is upper bounded by (3.6).

The detailed proof is presented in Appendix J.

26


---Page Break---
H
Analysis of PSεBAI: Change Detection and Context Alignment

As the output of Algorithms 3 and 4 depends on random samples, the output may not meet our
expectation with some probabilities. We will characterize the probabilities of the three “bad” events:
FAE in Appendix H.1, FA in Appendix H.2 and MI in Appendix H.3; and finally upper bound the
failure probability of the good event Good in this section.

Lemma G.4. Assume the instance satisfies Assumption 1,

P [Good] ≥1 −
δ
2τ ∗≥1 −δ

2.

Some notations are introduced first

• θ1:∞:= (θ∗
jt)∞
t=1 is the latent vector sequence.
• ˆcl indicates the time step when we raise alarm for the lth changepoint.

According to the dynamics (see Dynamics 1) of the problem, at a changepoint cl ∈C, the next latent
vector will be sampled from Pθ. Therefore, it may happen that θ∗
jcl = θ∗
jcl−1, i.e, this changepoint cl
is hidden and these two consecutive stationary segments are observed as one stationary segment. In
order to upper bound the errors made by Algorithm 3, we make the following assumption, which
yields the worst-case scenario in terms of the error probability.

Assumption 2. Given θ1:∞and the changepoint sequence C, θ∗
jcl ̸= θ∗
jcl+1, ∀l ∈N.

In this section, we do not consider the randomness of θ1:∞, i.e., we consider a realization of the
sequence.4 The results do not involve any parameter depending on the specific sequence and thus can
be applied to any latent vector sequence.

H.1
False Alarm Error

Lemma G.1. For any δFAE ∈(0, 1), with b ≥8d

3w ln
2
δFAE +

r
8d
3w ln
2
δFAE

2
+ 24

w d ln
2
δFAE , LCD

makes no false alarm before the stopping time τ with probability at least

P[FAEc] ≥1 −τ

γ KδFAE.

Proof of Lemma G.1. Note that within a stationary segment of length t, the observations are i.i.d.,
and thus the time segments between two consecutive false alarm time {ˆcl −ˆcl−1}t
l=1 (where ˆc0 = 0)
are i.i.d. i.e., it is a renewal process. Thus, it is sufficient to bound the error probability under the
stationary case, i.e. C = ∅and θ1:∞= (θ∗
jt = θ∗
j1)∞
t=1.

Assume that we are under a stationary segment of length t. We wish to upper bound the error
P[FAE] = P[ˆc1 ≤t]. This is given by Lemma H.1.

Lemma H.1. Given w CD samples from the same context [(˜xs, Ys,˜xs)]w
s=1 in a CD phase at time
step t, Algorithm 3 makes a false alarm error with probability upper bounded by KδFAE, i.e.

P [FAEt] ≤KδFAE

if b ≥8d

3w ln
2
δFAE +

r
8d
3w ln
2
δFAE

2
+ 24

w d ln
2
δFAE .

Assume γ divides t. In the case of a stationary segment, the input samples to Algorithm 3 are

[(xsγ, Ysγ,xsγ)]k+w−1
s=k
	 t

γ
k=1 if no changing alarm is raised (We ignore the initialization at Line 5
in Algorithm 1 for simplicity). Denote Γi(t) := {kγ : k ≡i mod w, kγ ≤t}, i = 0, 1, . . . , w −1.
Given any i = 0, . . . , w−1, for any k1 ̸= k2 ∈Γi(t), the two list of samples, [(xsγ, Ysγ,xsγ)]k1+w−1
s=k1
and [(xsγ, Ysγ,xsγ)]k2+w−1
s=k2
, do not overlap (thus, they are independent). Therefore, P[k] := P[ˆc1 =

4This can also be taken as, we are conditioning on θ1:∞, as the realization of the sequence is independent
from the behavior of any agent/algorithm.

27


---Page Break---
kγ + i, ˆc1 ∈Γi(t)] is a geometric distribution with parameter upper bounded by KδF AE. We have

P [ˆc1 ∈Γi(t)] ≤1 −(1 −KδFAE)|Γi(t)| .

Hence, the cumulative false alarm error is

P[FAE] = P [ˆc1 ≤t]

=

w−1
X

i=0

h
1 −(1 −KδFAE)|Γi(t)|i

≤

w−1
X

i=0
|Γi(t)|KδFAE

≤t

γ KδFAE.

Proof of Lemma H.1. According to Algorithm 3, given w CD samples from the same context
[(˜xs, Ys,˜xs)]w
s=1 and x ∈X, we need to bound

P






w

2
X

s=1
x⊤A(λ∗)−1˜xsYs,˜xs −

w
X

s= w

2 +1
x⊤A(λ∗)−1˜xsYs,˜xs
 ≥w

2 b



.

The proof is similar to Lemma E.1. For any s ∈[ w

2 ] and ˜s = s + w

2
E

x⊤A(λ∗)−1˜xsYs,˜xs −x⊤A(λ∗)−1˜x˜sY˜s,˜x˜s

= 0

andx⊤A(λ∗)−1˜xsYs,˜xs −x⊤A(λ∗)−1˜x˜sY˜s,˜x˜s


≤
x⊤A(λ∗)−1˜xs˜x⊤
s θ∗
js
 +
x⊤A(λ∗)−1˜xsηs
 +
x⊤A(λ∗)−1˜x˜s˜x⊤
˜s θ∗
j˜s
 +
x⊤A(λ∗)−1˜x˜sη˜s


≤4d.

By making use of (E.4),

E
h 
x⊤A(λ∗)−1˜xsYs,˜xs −x⊤A(λ∗)−1˜x˜sY˜s,˜x˜s
2i

= E
h 
x⊤A(λ∗)−1˜xsYs,˜xs
2i
+ E
h 
x⊤A(λ∗)−1˜x˜sY˜s,˜x˜s
2i

−2E

x⊤A(λ∗)−1˜xsYs,˜xs · x⊤A(λ∗)−1˜x˜sY˜s,˜x˜s


≤4d + 2E

|x⊤θ∗
js · x⊤θ∗
j˜s|


≤6d.

According to the Bernstein’s inequality,

P






w

2
X

s=1


x⊤A(λ∗)−1˜xsYs,˜xs −x⊤A(λ∗)−1˜xs+ w

2 Ys+ w

2 ,˜xs+ w

2

  ≥w

2 ϵ





≤2 exp

 

−

1
2
  w

2 ϵ
2

w

2 · 6d + 4d

3
w

2 ϵ

!

= 2 exp

 

−

w

2 ϵ2

12d + 8d

3 ϵ

!

.

In order to upper bound the error probability by δFAE, we need

2 exp

 

−

w

2 ϵ2

12d + 8d

3 ϵ

!

≤δF AE

28


---Page Break---
⇒
ϵ ≥
4d
3 · w

2
ln
2
δFAE
+

s 4d

3 · w

2
ln
2
δFAE

2
+ 12 · 2

wd ln
2
δFAE
.

By the choice of b ≥8d

3w ln
2
δFAE +

r
8d
3w ln
2
δFAE

2
+ 24

w d ln
2
δFAE , we have

P




 2

w

w

2
X

s=1
x⊤A(λ∗)−1˜xsYs,˜xs −2

w

w
X

s= w

2 +1
x⊤A(λ∗)−1˜xsYs,˜xs
 ≥b



≤δFAE.

A union bound over the K arms yields the final result.

H.2
Failed Alarm

Lemma G.2. Conditional on FAEc, for any δFA
∈
(0, 1), if
∆c−b

2
≥
d
w ln
2
δFA +
r
d
w ln
2
δFA

2
+ 4d

w ln
2
δFA ,

P
h
cl ≤ˆcl ≤cl + wγ

2

ˆcl ≥cl
i
≥1 −δFA.

In addition, P

FA
FAEc
≤lτδFA.

Proof of Lemma G.2. Conditioned on FAEc, the detection at the changepoints is independent. Thus
we can assume there is only one changepoint c1 within a certain number of consecutive time steps.

Algorithm 3 is given w CD samples which are collected under two different context. Without loss of
generality, we assume the sample selected at time c1 is among the CD samples (otherwise, we can
regard the time step of the first sample from the second context as c1). We wish to bound

P
h
c1 ≤ˆc1 ≤c1 + γw

2

ˆc1 ≥c1
i

which is the probability of the event that, after a changepoint occurs, a changing alarm will be raised
within wγ

2 time steps (or w

2 CD samples). Here, ˆc1 ≥c1 can be guaranteed as we are conditioning on
that there is no false alarm error (FAEc).

The event c1 ≤ˆc1 ≤c1 + γw

2 indicates, at least one of the CD sample list in
nh
(xc1+(s−k)γ, Yc1+(s−k)γ,xc1+(s−k)γ)
iw

s=1

ow

k= w

2 +1
will trigger the changing alarm in Algorithm 3. In particular, in Lemma H.2, we consider the failed
arm probability when the CD samples are composed by exactly half samples from each contexts,
h
(xc1+(s−1−w

2 )γ, Yc1+(s−1−w

2 )γ,xc1+(s−1−w

2 )γ)
iw

s=1.

Lemma H.2. Given w CD samples from the two different contexts [(˜xs, Ys,˜xs)]w
s=1 in a CD phase,
where [(˜xs, Ys,˜xs)]

w

2
s=1 is from latent vector θ∗
j and [(˜xs, Ys,˜xs)]w
s= w

2 +1 is from latent vector θ∗
˜j ,
Algorithm 3 raises a changing alarm with probability lower bounded by

1 −δFA

if ∆c−b

2
≥
d
w ln
2
δFA +

r
d
w ln
2
δFA

2
+ 4d

w ln
2
δFA where ∆c := maxx∈X |x⊤(θ∗
j −θ∗
˜j )| and it is

assumed to be greater than b.

By making use of Lemma H.2,

P
h
c1 ≤ˆc1 ≤c1 + wγ

2

ˆc1 ≥c1
i

= P
h
∃k ∈{w

2 + 1, . . . , w},

LCD

w, b, [(xc1+(s−k)γ, Yc1+(s−k)γ,xc1+(s−k)γ)]w
s=1

= True
ˆc1 ≥c1
i

29


---Page Break---
≥P
h
LCD

w, b, [(xc1+(s−1−w

2 )γ, Yc1+(s−1−w

2 )γ,xc1+(s−1−w

2 )γ)]w
s=1

= True
ˆc1 ≥c1
i

≥1 −δFA

where ∆c = maxx∈X |x⊤(θ∗
j −θ∗
˜j )| and θ∗
j , θ∗
˜j are the latent vectors. Hence,

P

FA
FAEc
= 1 −P

FAcFAEc
≤1 −(1 −δFA)lτ ≤lτδFA

Proof of Lemma H.2. According to the design of LCD, there exists an x ∈X,

P







2
w

w

2
X

s=1
x⊤A(λ∗)−1˜xsYs,˜xs −2

w

w
X

s= w

2 +1
x⊤A(λ∗)−1˜xsYs,˜xs


≥b





≥1 −P

" 

w

2
X

s=1
x⊤A(λ∗)−1˜xsYs,˜xs −

w
X

s= w

2 +1
x⊤A(λ∗)−1˜xsYs,˜xs −w

2 x⊤(θ∗
j −θ∗
˜j )



≥
w

2 b −w

2 x⊤(θ∗
j −θ∗
˜j )


#

= 1 −P

" 

w

2
X

s=1

 
x⊤A(λ∗)−1˜xsYs,˜xs −x⊤θ∗
j

−

w
X

s= w

2 +1


x⊤A(λ∗)−1˜xsYs,˜xs −x⊤θ∗
˜j



≥
w

2 b −w

2 x⊤(θ∗
j −θ∗
˜j )


#

≥1 −δFA

where the last inequality holds as ∆c−b

2
≥d

w ln
2
δFA +

r
d
w ln
2
δFA

2
+ 4d

w ln
2
δFA .

H.3
Context Alignment

Lemma G.3. Conditional on FAEc and FAc, based on the conditions in Lemma G.1 and Lemma G.2,

P [MI|FAEc, FAc] ≤lτ [(N −1)δFA + KδFAE] .

Proof of Lemma G.3. The error of the context alignment procedure can be derived from the FAE and
FA analyses. Conditioned on FAEc, FAc and that the previous l −1 contexts are correctly identified,
i.e., Tl−1
k=1 MIc
k, we have the following statements.

Firstly, according to Lemma H.1, the change alarm at Line 4 in Algorithm 4 will not be triggered
with probability at least (1 −KδFAE) if the current context is context j. This error has been taken
into account in the FAE (see Remark H.3).

Secondly, if the current context is not j (which will occur at most N −1 times), a change alarm will
be raised with probability at least 1 −δFA by Lemma G.2.

Therefore, the error probability at cl is upper bounded by

P

MIl|FAEc, FAc, ∩l−1
k=1MIc
k

≤(N −1)δFA + KδFAE
and by a union bound, the cumulative error probability bounded by

P [MI|FAEc, FAc] = 1 −P [MIc|FAEc, FAc]
(H.1)

≤1 −(1 −(N −1)δFA −KδFAE)lτ

≤lτ ((N −1)δFA + KδFAE) .

Remark H.3. Note that (1) we will only bound the FAE when the CD samples are from the same
context; (2) in the analysis of FAE, we bound the error probability on the whole horizon up to time t

30


---Page Break---
as we assume there is no changepoint. In particular, there will be unused and redundant w

2 KδFAE
errors budget for FAE at each changepoint, which accumulates to lτ w

2 KδFAE before time step τ.
Therefore, the second error term in (H.1), lτKδFAE, can be covered by the unused lτ w

2 KδFAE error
budget from FAE, and thus it can be neglected in (H.1).

Proof of Lemma G.4. According to Lemma G.1, G.2, G.3, Assumption 1 and Remark H.3, given a
time τ, the total failure probability is upper bounded by

P [Goodc] = P [FAE ∪FA ∪MI]
(H.2)

≤τ

γ KδFAE + lτδFA + lτ · (N −1)δFA

= τ

γ KδFAE + lτNδFA.

In particular, when τ is upper bounded by τ ∗in Line 2 of Algorithm 1

τ ≤τ ∗= c0
NLmax

ε2
ln N 2KLmax/ε2

δ
where c0 = 3 · 6400 ln 6400. The number of changepoints till τ is upper bounded by lτ ∗≤
τ ∗
Lmin .

By Assumption 1, when b =
8d
3w ln
2
δFAE +

r
8d
3w ln
2
δFAE

2
+ 24

w d ln
2
δFAE and δFAE =
γδ
4(τ ∗)2K ,

the conditions of Lemma G.1 are met. And we can upper bound τ

γ KδFAE by
δ
4τ ∗≤δ

4.

By making use of Assumption 1 and setting δFA =
δ
4Nlτ∗τ ∗, we have δFA > δFAE and

∆c −b

2
≥b

2 ≥4d

3w ln
2
δFAE
+

s 4d

3w ln
2
δFAE

2
+ 6

wd ln
2
δFAE

≥d

w ln 2

δFA
+

s d

w ln 2

δFA

2
+ 4d

w ln 2

δFA
.

Thus, Lemma G.2 can be applied. lτ ∗NδFA can be upper bounded by
δ
4τ ∗≤δ

4.

Therefore, according to (H.2), P[Good] ≥1 −
δ
2τ ∗≥1 −δ

2.

I
Analysis of PSεBAI: Estimation Error

Lemma G.5 is proved in this section. We will upper bound these three error terms (see (I.1)): VE
error in App I.1, DE error in App I.2, RE error in App I.3, and finally we will prove Lemma G.5
which upper bounds the failure probability of CI at the end of this section.

Lemma G.5. If Tt ≥2Lmax

9
ln
2
δd,Tt ,

P

CIt
Good

≥1 −(Kδv,Tt + Nδd,Tt + KNδm,Tt) .

In addition,

P

CI
Good

≥1 −δ

2.

Given any two arms x, ˜x ∈X, by the triangular inequality, the deviation between ˆ∆t(x, ˜x) and
∆(x, ˜x) can be upper bounded by three terms: the Vector-Estimation Error (VE) term, the Distribution-
Estimation Error (DE) term, and the Residual Estimation Error (RE) term:

| ˆ∆t(x, ˜x) −∆(x, ˜x)|
(I.1)

= |(x −˜x)⊤ˆΘtˆpt −(x −˜x)⊤Θp|

31


---Page Break---
≤


N
X

j=1


ˆ∆t,j(x, ˜x) −∆j(x, ˜x)

ˆpt,j


|
{z
}
VE term

+


N
X

j=1
ˆ∆clip2
t,j
(x, ˜x)(ˆpt,j −pj)


|
{z
}
DE term

+


N
X

j=1


∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

(ˆpt,j −pj)


|
{z
}
RE term

,

where aclip2 := clip2(a) := min{max{a, −2}, 2} is a shorthand notation for the value of a that
is clipped to the interval [−2, 2]. The reason the value ˆ∆t,j(x, ˜x) is clipped is the ground truth
|∆t,j(x, ˜x)| ≤2.

Recall the event CI

CIt := {
 ˆ∆t(x, ˜x) −∆(x, ˜x)
 ≤ρt(x, ˜x), ∀x, ˜x ∈X},

CI :=
\

t
CIt,

and the confidence radius

ρt(x, ˜x) := 2(αt + ξt) +

N
X

j=1
βt,j| ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)|,
(I.2)

where
αt := 5

s

d
Tt
ln
2
δv,Tt
,
βt,j := 5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
,

ξt := 25
√

2NLmax

Tt
ln
2
δm,Tt
,
ϕt,j := min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4


,

and ζt(x, ˜x)
∈
R can be any value.
In particular, it can be the value that minimizes
PN
j=1 βt,j| ˆ∆t,j(x, ˜x) + ζt(x, ˜x)| or it can be taken as ε. For simplicity, we will take ζt(x, ˜x) =

arg minζt(x,˜x)∈R
PN
j=1 βt,j( ˆ∆t,j(x, ˜x)+ζt(x, ˜x))2 = −

PN
j=1 βt,j ˆ∆t,j(x,˜x)
PN
j=1 βt,j
for a simple and effective

analytic expression in the algorithm.

I.1
Vector-Estimation Error

For the VE term
 PN
j=1

ˆ∆t,j(x, ˜x) −∆j(x, ˜x)

ˆpt,j
, note that



N
X

j=1


ˆ∆t,j(x, ˜x) −∆j(x, ˜x)

ˆpt,j
 ≤|x⊤(ˆΘt −Θ)ˆpt| + |˜x⊤(ˆΘt −Θ)ˆpt|

Lemma I.1. Given x ∈X and Tt ≥d

4 ln
2
δv,Tt ,

P
h
|x⊤(ˆΘt −Θ)ˆpt| ≥αt
Good
i
≤δv,Tt

Proof of Lemma I.1. By the definition of the estimators in (3.2),

x⊤(ˆΘt −Θ)ˆpt =

N
X

j=1
x⊤(ˆθt,j −θ∗
j )ˆpt,j

=

N
X

j=1
x⊤



1

Tt,j

X

s∈Tt,j
A(λ∗)−1xsYs,xs −θ∗
j



Tt,j

Tt

= 1

Tt

N
X

j=1

X

s∈Tt,j
x⊤A(λ∗)−1xsYs,xs −x⊤θ∗
j

32


---Page Break---
By Lemma E.1,

P
h
|x⊤(ˆΘt −Θ)ˆpt| ≥αt
θ1:∞, Good
i
≤δv,Tt

By the property of conditional probability, we have the desired result.

Therefore, conditional on Good, with probability at least 1 −Kδv,Tt, VE ≤2αt for any x, ˜x ∈X.

I.2
Distribution-Estimation Error

Lemma I.2. Given Good and Tt ≥2Lmax

9
ln
2
δd,Tt , for any j ∈[N],

P

|ˆpt,j −pj| ≥βt,j
Good

≤δd,Tt.

Additionally, with probability 1 −Nδd,Tt,


N
X

j=1
ˆ∆clip2
t,j
(x, ˜x)(ˆpt,j −pj)
 ≤

N
X

j=1
βt,j| ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)|

where ζt(x, ˜x) can be any value.

Proof of Lemma I.2. Given any j ∈[N], and the stationary segment l, we denote Xj,l := ˆLl1{θ∗
jl =
θ∗
j } −pj ˆLl, where ˆLl is the total length of the Exp phases in the lth stationary segment. Note that
E[Xj,l|{ˆLl}lt
l=1, Good] = 0, |Xj,l| ≤Lmax a.s. and Var[Xj,l|{ˆLl}lt
l=1, Good] = pj(1 −pj)ˆL2
l ≤
pj(1 −pj)ˆLlLmax. By Bernstein’s inequality,

P

"

|

lt
X

l=1
Xj,l| ≥ϵ
{ˆLl}lt
l=1, Good

#

≤2 exp

 

−
ϵ2

2 Plt
l=1 Var[Xj,l|Good] + 2

3Lmaxϵ

!

⇒
P

"

| 1

Tt

lt
X

l=1
Xj,l| ≥ϵ
{ˆLl}lt
l=1, Good

#

≤2 exp

 

−
T 2
t ϵ2

2 Plt
l=1 Var[Xj,l|Good] + 2

3LmaxTtϵ

!

As Tt,j = Plt
l=1 ˆLl1{θ∗
jl = θ∗
j }, Tt = Plt
l=1 ˆLl and Plt
l=1 ˆLl = Tt, we have

P
h
|ˆpt,j −pj| ≥ϵ
{ˆLl}lt
l=1, Good
i
≤2 exp

 

−
T 2
t ϵ2

2 Plt
l=1 Var[Xj,l|{ˆLl}lt
l=1, Good] + 2

3LmaxTtϵ

!

≤2 exp

 

−
T 2
t ϵ2

2 Plt
l=1 pj(1 −pj)ˆLlLmax + 2

3LmaxTtϵ

!

≤2 exp

−
Ttϵ2

2pj(1 −pj)Lmax + 2

3Lmaxϵ



As the last bound is independent of {ˆLl}lt
l=1, we have

P

|ˆpt,j −pj| ≥ϵ
Good

≤2 exp

−
Ttϵ2

2pj(1 −pj)Lmax + 2

3Lmaxϵ



If we want to upper bound the above by δd,Tt ∈(0, 1), i.e.,

2 exp

−
Ttϵ2

2pj(1 −pj)Lmax + 2

3Lmaxϵ


≤δd,Tt

we require

ϵ ≥

1
3Lmax ln
2
δd,Tt +

r
1
3Lmax ln
2
δd,Tt

2
+ 2Ttpj(1 −pj)Lmax ln
2
δd,Tt
Tt
(I.3)

33


---Page Break---
In particular,

ϵ = ˜βt,j := 5

2 max

(
Lmax

3Tt
ln
2
δd,Tt
,

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

)

satisfies the condition, we have

P
h
|ˆpt,j −pj| ≥˜βt,j
Good
i
≤δd,Tt.
(I.4)

A problem here is we do not have access to pj during the dynamics, therefore, we adopt Lemma K.1
to further upper bound ˜βt,j by βt,j.

Lemma K.1. Given any j ∈[N] and Tt ≥2Lmax

9
ln
2
δd,Tt , ˜βt,j ≤βt,j.

Therefore,

P

|ˆpt,j −pj| ≥βt,j
Good

≤δd,Tt.

In addition, with probability at least 1 −Nδd,Tt, we can upper bound DE term as:


N
X

j=1
ˆ∆clip2
t,j
(x, ˜x)(ˆpt,j −pj)
 =


N
X

j=1


ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)

(ˆpt,j −pj)

(I.5)

≤

N
X

j=1
βt,j
 ˆ∆clip2
t,j
(x, ˜x) + ζt(x, ˜x)


where ζt(x, ˜x) ∈R can be any value.

I.3
Residual-Estimation Error

RE is composed by the product of two deviations, i.e.,

∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

and (ˆpt,j −pj),
as the time step t becomes large, we expect it will converge to zero fast. Thus, it is sufficient to have
a coarse estimation of it.

Lemma I.3. For any x ∈X, conditional on that |ˆpt,j −pj| ≤βt,j, ∀j ∈[N],

P






N
X

j=1


∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

(ˆpt,j −pj)
 ≥ξt

Good



≤Nδm,Tt

where ξt := 25
√

2 NLmax

Tt
ln
2
δm,Tt .

Proof of Lemma I.3. According to Lemma E.1, for any x ∈X, j ∈[N] we have

P

"

|x⊤(θ∗
j −ˆθt,j)| ≥5

s

d
Tt,j
ln
2
δm,Tt

Good

#

≤δm,Tt
(I.6)

if Tt,j ≥d

4 ln
2
δm,Tt . Note the fact that ∆j(x, ˜x) ∈[−2, 2], hence,
∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)
 ≤
∆j(x, ˜x) −ˆ∆t,j(x, ˜x)
 ≤|x⊤(θ∗
j −ˆθt,j)| + |˜x⊤(θ∗
j −ˆθt,j)|. (I.7)

And |x⊤(θ∗
j −ˆθt,j)| ≤3d with probability 1. Denote

ψt,j,1 := max

ˆpt,j, d

4Tt
ln
2
δm,Tt


, ψt,j,2 := max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


.
(I.8)

34


---Page Break---
We have5



N
X

j=1


∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

(ˆpt,j −pj)


≤
X

j:ψt,j,1>ˆpt,j



∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

βt,j
 +
X

j:ψt,j,1=ˆpt,j,
ψt,j,2>ˆpt,j



∆j(x, ˜x) −ˆ∆clip2
t,j,2 (x, ˜x)

βt,j


+
X

j:ψt,j,2=ˆpt,j



∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

βt,j


(a)
≤
X

j:ψt,j,1>ˆpt,j



∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)

βt,j
 +
X

j:ψt,j,1=ˆpt,j,
ψt,j,2>ˆpt,j



∆j(x, ˜x) −ˆ∆clip2
t,j,2 (x, ˜x)

βt,j


+
X

j:ψt,j,2=ˆpt,j
|x⊤(θ∗
j −ˆθt,j)βt,j| + |˜x⊤(θ∗
j −ˆθt,j)βt,j|
(I.9)

(b)
≤
X

j:ψt,j,1>ˆpt,j
4 · 5

2 · 5
√

2
2
Lmax

Tt
ln
2
δd,Tt

+
X

j:ψt,j,1=ˆpt,j,
ψt,j,2>ˆpt,j

min

(

4, 2 · 5

s

d
Tt,j
ln
2
δm,Tt

)

· 5
2 · 5
√

2
2
Lmax

Tt
ln
2
δd,Tt

+
X

j:ψt,j,3=ˆpt,j
2 · 5

s

d
Tt,j
ln
2
δm,Tt
· 5
2

s

8ˆpt,jLmax

Tt
ln
2
δd,Tt

≤
X

j:ψt,j,1>ˆpt,j
25
√

2Lmax

Tt
ln
2
δd,Tt

+
X

j:ψt,j,1=ˆpt,j,
ψt,j,2>ˆpt,j

min

(

4, 10

s

d
Tt,j
ln
2
δm,Tt

)

· 25
√

2
4
Lmax

Tt
ln
2
δd,Tt

+
X

j:ψt,j,2=ˆpt,j
50
√

2
√dLmax

Tt
ln
2
δm,Tt
(I.10)

≤50
√

2NLmax

Tt
ln
2
δm,Tt
= 2ξt

where (a) adopts (I.7) and (b) is obtained by the following derivations:
(1) ψt,j,1
>
ˆpt,j indicates ˆpt,j
<
d
4Tt ln
2
δm,Tt
<
25

4
Lmax

Tt
ln
2
δd,Tt , and thus βt,j
≤
5
2 ·

5
√

2
2
Lmax

Tt
ln
2
δd,Tt .In addition,6,
∆j(x, ˜x) −ˆ∆clip2
t,j
(x, ˜x)
 ≤4.

(2) ψt,j,1 = ˆpt,j indicates ˆpt,j ≤
d
4Tt ln
2
δm,Tt and ψt,j,2 > ˆpt,j implies ˆpt,j < 25

4
Lmax

Tt
ln
2
δd,Tt , thus

(I.6) can be applied and βt,j ≤5

2 · 5
√

2
2
Lmax

Tt
ln
2
δd,Tt .

5The last inequality can be loose. When each context has been visited at least once, the first summation
in (I.10) can be ignored. For empirical performance, (I.10) should be used, while ξ is adopted for theoretical
guarantees.
6This demonstrates the usefulness of the clipping technique. Without the clipping technique, we can only get
an upper bound of 6d by utilizing (E.2) which introduces another d factor in ξt.

35


---Page Break---
(3) ψt,j,2 = ˆpt,j indicates ˆpt,j ≥
25
16
Lmax

Tt
ln
2
δd,Tt ≥
d
4Tt ln
2
δm,Tt , thus (I.6) can be applied and

βt,j ≤5

2

q

8ˆpt,jLmax

Tt
ln 2

δ .
Therefore, conditional on Good, with probability at least 1 −KNδm,Tt, RE ≤2ξt.

Proof of Lemma G.5. By Lemma I.1, I.2 and I.3, conditional on Good, with probability at least
1 −(Kδv,Tt + Nδd,Tt + KNδm,Tt),
 ˆ∆t(x, ˜x) −∆(x, ˜x)
 ≤ρt(x, ˜x). This finishes the proof of
the first result in Lemma G.5.

We then accumulates all the error probabilities.
By the choices of δv,Tt =
δ
15KT 3
t , δd,Tt =

δ
15NT 3
t , δm,Tt =
δ
15KNT 3
t ,

∞
X

Tt=1
Kδv,Tt + Nδd,Tt + KNδm,Tt =

∞
X

Tt=1
K ·
δ
15KT 3
t
+ N ·
δ
15NT 3
t
+ KN ·
δ
15KNT 3
t

=

∞
X

Tt=1

3δ
15T 3
t

≤δ

4
Note that there is an reversion step at Line 18 Algorithm 1, for each T ∈N, there are at most two
time steps t1 < t2 with Tt1 = Tt2. Therefore,

∞
X

t=1
Kδv,Tt + Nδd,Tt + KNδm,Tt ≤2

∞
X

Tt=1
Kδv,Tt + Nδd,Tt + KNδm,Tt ≤δ

2

This proves the second statement.

J
Upper Bound of PSεBAI: Proof of Theorem 3.2

Theorem 3.2 is proved in this section by three steps.

• Firstly, we show that the recommended arm ˆxε is an ε-best arm upon the termination of the
algorithm in Lemma J.1.
• Secondly, we present a sufficient condition for the termination of the algorithm in terms of Tt in
Lemma G.6.
• Lastly, we show that Tτ is bounded by a constant fraction of τ and thus, obtain an upper bound on
τ.

Step 1:

According to Lemma G.5, CIt holds with probability 1−(Kδv,Tt +Nδd,Tt +KNδm,Tt). Conditional
on the good event Good in (G.1) and the event CIt (the mean gaps are well approximated at time step
t), we expect the recommended arm will be an ε-optimal arm. This is formalized in the following
lemma.

Lemma J.1. Conditional on Good and CIt, if the algorithm stops at time step t, the recommended
arm ˆxε = x∗
t is an ε-best arm.

Proof of Lemma J.1. If ˆxε = x∗, the lemma holds.

If ˆxε ̸= x∗, according to the termination condition (3.4)

min
x:x̸=ˆxε
ˆ∆t(ˆxε, x) −ρt(ˆxε, x) > −ε

we have

∆(ˆxε, x∗) ≥ˆ∆t(ˆxε, x∗) −ρt(ˆxε, x∗) ≥min
x:x̸=ˆxε
ˆ∆t(ˆxε, x) −ρt(ˆxε, x) > −ε

where the first inequality holds due to CIt. This indicates ∆(x∗, ˆxε) ≤ε and thus the recommended
arm ˆxε = x∗
t is an ε-best arm.

36


---Page Break---
Step 2:

Lemma G.6. Conditional on Good and CI, the recommended arm ˆxε ∈Xε and when Algorithm 1
terminates, the order of Tt is upper bounded by (3.6).

Proof of Lemma G.6. By Lemma J.1, ˆxε ∈Xε. Note that for any x ̸= ˆxε, when
2ρt(x, ˆxε) ≤∆(x, ˆxε) + ε,
ˆxε ̸= x∗and x = x∗,
ρt(ˆxε, x) + ρt(x∗, x) ≤∆(x∗, x) + ε,
otherwise.
(J.1)

By utilizing CIt, we have











ˆ∆t(ˆxε, x∗) −ρt(ˆxε, x∗) ≥ˆ∆t(x∗, ˆxε) −ρt(ˆxε, x∗) ≥∆t(x∗, ˆxε) −2ρt(x∗, ˆxε) ≥−ε,
ˆxε ̸= x∗and x = x∗,
ˆ∆t(ˆxε, x) −ρt(ˆxε, x) ≥ˆ∆t(x∗, x) −ρt(ˆxε, x) ≥∆(x∗, x) −ρt(x∗, x) −ρt(ˆxε, x) ≥−ε,
otherwise.

Therefore, if (J.1) hold for all x ̸= ˆxε, the algorithm must have stopped, i.e., it is sufficient to have
the following for any x ̸= ˆxε:







ρt(x∗, ˆxε) ≤∆(x∗, ˆxε) + ε

2
,
ˆxε ̸= x∗and x = x∗,

ρt(ˆxε, x) ≤∆(x∗, x) + ε

2
, ρt(x∗, x) ≤∆(x∗, x) + ε

2
,
otherwise.

Lemma J.2 gives an upper bound on the total number of time steps in Exp phases at which (J.1) must
hold for any two arms xε ∈Xε, x ∈X \ {xε}.

Lemma J.2. For any two arms (xε, x), where x ̸= xε, x∗, when

Tt = 6400 ln 6400 · d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ

+ 6400 ln 6400 · HDE(xε, x) ln NHDE(xε, x)

δ

+ 3200
√

2 ln 3200
√

2 · NLmax
∆(x∗, x) + ε
ln

KNLmax
∆(x∗,x)+ε

δ
(J.2)

= ˜O

 
d

(∆(x∗, x) + ε)2 ln 1

δ + HDE(xε, x) ln 1

δ +
NLmax
∆(x∗, x) + ε ln 1

δ

!

where

HDE(xε, x) :=
Lmax
(∆(x∗, x) + ε)2




N
X

j=1

s

min

16pj, 1

4


|∆j(xε, x) + ε|





2

,

we must have

ρt(xε, x) ≤∆(x∗, x) + ε

2

By taking the maximum of (J.2) over all ε-best arms and the suboptimal arms, we get

˜T :=
max
xε∈Xε,x̸=xε,x∗
6400 ln 6400 · d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ

+ 6400 ln 6400 · HDE(xε, x) ln NHDE(xε, x)

δ

+ 3200
√

2 ln 3200
√

2 · NLmax
∆(x∗, x) + ε
ln

KNLmax
∆(x∗,x)+ε

δ
(J.3)

= ˜O

 

max
xε∈Xε,x̸=xε,x∗
d

(∆(x∗, x) + ε)2 ln 1

δ + HDE(xε, x) ln 1

δ +
NLmax
∆(x∗, x) + ε ln 1

δ

!

.

37


---Page Break---
Proof of Lemma J.2. Let cv, cd, cm ∈(0, 1) be constants, satisfying cv + cd + 2cm = 1. By the
definition of ρt(xε, x) in (3.3), it is sufficient to have



















2αt ≤cv
∆(x∗, x) + ε

2
N
X

j=1
βt,j| ˆ∆clip2
t,j
(xε, x) + ζt(xε, x)| ≤(cd + cm)∆(x∗, x) + ε

2

2ξt ≤cm
∆(x∗, x) + ε

2
where we take ζt(xε, x)
=
ε for theoretical simplicity,
but we adopt ζt(x, ˜x)
=

arg minζt(x,˜x)∈R
PN
j=1 βt,j( ˆ∆t,j(x, ˜x) + ζt(x, ˜x))2 = −

PN
j=1 βt,j ˆ∆t,j(x,˜x)
PN
j=1 βt,j
in the algorithm, which

still enjoys the theoretical guarantee.

VE Term According to the definition of αt, we need to bound:

αt ≤cv
∆(x∗, x) + ε

4

⇔
5

s

d
Tt
ln
2
δv,Tt
≤cv
∆(x∗, x) + ε

4

By a coarse estimation, it is sufficient to have

Tt ≥
¯cvd

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ
= O

 
d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ

!

where ¯cv = 6400

c2v ln 6400

c2v .

DE Term The difficulty lies at the DE term.

N
X

j=1
βt,j| ˆ∆clip2
t,j
(xε, x) + ζt(xε, x)|

≤

N
X

j=1
βt,j| ˆ∆clip2
t,j
(xε, x) + ε|

≤
X

j:ψt,j>ˆpt,j
βt,j| ˆ∆clip2
t,j
(xε, x) + ε|

+
X

j:ψt,j=ˆpt,j
βt,j|∆j(xε, x) + ε| + βt,j| ˆ∆clip2
t,j
(xε, x) −∆j(xε, x)|

≤
X

j:ψt,j>ˆpt,j
βt,j(2 + ε) +
X

j:ψt,j=ˆpt,j
βt,j| ˆ∆clip2
t,j
(xε, x) −∆j(xε, x)|

+
X

j:ψt,j=ˆpt,j
βt,j|∆j(xε, x) + ε|

≤2ξt +
X

j:ψt,j=ˆpt,j
βt,j|∆j(xε, x) + ε|

We will upper bound 2ξt by cm
∆(x∗,x)+ε

2
first; the second term will be included in the RE term and
analyzed later. Therefore, it is sufficient to have
X

j:ψt,j=ˆpt,j
βt,j|∆j(xε, x) + ε| ≤cd
∆(x∗, x) + ε

2
(J.4)

38


---Page Break---
By the definition of βt,j in (I.2) and Lemma K.3, we get

βt,j = 5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
≤5

2

s

2 min

16pj, 1

4
	
Lmax
Tt
ln
2
δd,Tt
(J.5)

Lemma K.3. If ψt,j = ˆpt,j, then

ϕt,j ≤min

16pj, 1

4


.

Therefore, in order for (J.4) to hold, it is sufficient to have

X

j:ψt,j,2=ˆpt,j

5
2

s

2 min

16pj, 1

4
	
Lmax
Tt
ln
2
δd,Tt
|∆j(xε, x) + ε| ≤cd
∆(x∗, x) + ε

2

⇔
5
2

s

2Lmax

Tt
ln
2
δd,Tt

X

j:ψt,j,2=ˆpt,j

s

min

16pj, 1

4


|∆j(xε, x) + ε| ≤cd
∆(x∗, x) + ε

2

⇐
5
2

s

2Lmax

Tt
ln
2
δd,Tt

N
X

j=1

s

min

16pj, 1

4


|∆j(xε, x) + ε| ≤cd
∆(x∗, x) + ε

2

⇐
Tt ≥¯cdHDE(xε, x) ln NHDE(xε, x)

δ
= O

HDE(xε, x) ln NHDE(xε, x)

δ



where ¯cd = 100

c2
d ln 100

c2
d .

RE Term By the definition of ξt in (I.2), it is sufficient to have

25
√

2NLmax

Tt
ln
2
δm,Tt
≤cm
∆(x∗, x) + ε

2

⇐
Tt ≥¯cm · NLmax

∆(x∗, x) + ε ln

KNLmax
∆(x∗,x)+ε

δ
= O

 
NLmax
∆(x∗, x) + ε ln

KNLmax
∆(x∗,x)+ε

δ

!

where ¯cm = 400
√

2
cm
ln 400
√

2
cm .

By taking cv = 3

4, cd = 1

8 and cm = 1

8, the upper bound on Tt can be concluded as

Tt = 6400 ln 6400 · d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ
+ 6400 ln 6400 · HDE(xε, x) ln NHDE(xε, x)

δ

+ 3200
√

2 ln 3200
√

2 · NLmax
∆(x∗, x) + ε
ln

KNLmax
∆(x∗,x)+ε

δ

= O

 
d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ

+HDE(xε, x) ln NHDE(xε, x)

δ
+
NLmax
∆(x∗, x) + ε ln

KNLmax
∆(x∗,x)+ε

δ

!

= ˜O

 
d

(∆(x∗, x) + ε)2 ln 1

δ + HDE(xε, x) ln 1

δ +
NLmax
∆(x∗, x) + ε ln 1

δ

!

.

Step 3:

Theorem 3.2. Define the context distribution estimation (DE) hardness parameter

HDE(xε, x) :=
Lmax
(∆(x∗, x) + ε)2 ¯H(xε, x)

39


---Page Break---
where ¯H(xε, x) :=
  PN
j=1
p

min {16pj, 1/4}|∆j(xε, x) + ε|
2. Under Assumption 1, with proba-
bility at least 1 −δ, PSεBAI identifies an ε-optimal arm and its sample complexity is

˜O

 

max
xε∈Xε
x̸=xε,x∗

d

(∆(x∗, x) + ε)2 ln 1

δ
|
{z
}
TV(x)

+ HDE(xε, x) ln 1

δ
|
{z
}
TD(xε,x)

+
NLmax
∆(x∗, x) + ε ln 1

δ
|
{z
}
TR(x)

!

.
(3.6)

Proof of Theorem 3.2. By Assumption 1, and the choice of the parameters, Lemma G.4 indicates
Good is guaranteed with probability at least 1 −δ

2.

Note that some arm pulls are not counted in Tt:

• In every γ time steps, Algorithm 1 enters the CD phase to select a CD sample.
• When a changepoint is detected, Algorithm 1 steps into the CA phase, where w
2 arms are sampled
in Algorithm 4.
• During the CA phase, we abandon wγ
2 samples.

Therefore, when t is large (or after the first stationary segment)

t ≤Tt ·
γ
γ −1
Lmin
Lmin −wγ −w

2
≤Tt ·
γ
γ −1
Lmin
Lmin −3wγ

2
.
(J.6)

As γ ≥2 and wγ are set to be upper bounded by Lmin

3 , so the fractions above is upper bounded by an
absolute constant 4, e.g., with γ = 2 and w = Lmin

6 , the constant is 4. Therefore, t is of the same
order as Tt. By Lemma G.6, (3.6) holds with probability 1 −δ.

K
Analysis of PSεBAI: Technical Lemmas

Lemma K.1. Given any j ∈[N] and Tt ≥2Lmax

9
ln
2
δd,Tt , ˜βt,j ≤βt,j.

Proof of Lemma K.1. Recall

ϕt,j = min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4



The proof is scheduled in 2 steps.

Step 1: Upper bound
q

2pj(1−pj)Lmax

Tt
ln
2
δd,Tt .

When Lmax

3Tt ln
2
δd,Tt ≤
q

2pj(1−pj)Lmax

Tt
ln
2
δd,Tt , we have

˜βt,j = 5

2

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

As pj(1 −pj) ≤1

4, ∀pj ∈(0, 1), this nai¨ve bound gives ˜βt,j ≤5

2

r

2· 1

4 ·Lmax

Tt
ln
2
δd,Tt .

If Tt,j > 0, according to Lemma K.2,

˜βt,j = 5

2

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

≤5

2

s

2pjTt,jLmax

TtTt,j
ln
2
δd,Tt

≤5

2

s

2ˆpt,jLmax

Tt
· 4 max

1, 25

4
Lmax

Tt,j
ln
2
δd,Tt


ln
2
δd,Tt
,

40


---Page Break---
thus we have

˜βt,j ≤5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
(K.1)

If Tt,j = 0, i.e., the latent context j has never been observed and ˆpt,j = 0, we have

˜βt,j = 5

2

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

≤5

2

s

2˜βt,j(1 −˜βt,j)Lmax

Tt
ln
2
δd,Tt

⇒
˜βj(t, δ) ≤

25

8 Lmax ln
2
δd,Tt
Tt + 25

8 Lmax ln
2
δd,Tt
≤25

8
Lmax

Tt
ln
2
δd,Tt

Take the trivial bound into consideration and observe that ϕt,j = min{ 25Lmax

Tt
ln
2
δd,Tt , 1

4} and that

25

8
Lmax

Tt
ln
2
δd,Tt
≤25
√

2Lmax
2Tt
ln
2
δd,Tt
= 5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
.

Thus, (K.1) also holds for Tt,j = 0.

To conclude, we have

˜βt,j = 5

2 max

(
Lmax

3Tt
ln
2
δd,Tt
,

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

)

≤5

2 max

(
Lmax

3Tt
ln
2
δd,Tt
,

s

2ϕt,jLmax

Tt
ln
2
δd,Tt

)

Step 2: Simplify the bound. Recall that

ϕt,j = min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4



As Tt ≥2Lmax

9
ln
2
δd,Tt , we have

Lmax

3Tt
ln
2
δd,Tt
≤

s

2 · 1

4Lmax

Tt
ln
2
δd,Tt

According to the definition of ϕ, if ˆpt,j ≥25

4
Lmax

Tt
ln
2
δd,Tt , we have

Lmax

3Tt
ln
2
δd,Tt
≤5
√

2Lmax

Tt
ln
2
δd,Tt
≤

s

2 · 4ˆpt,jLmax

Tt
ln
2
δd,Tt

If ˆpt,j ≤25

4
Lmax

Tt
ln
2
δd,Tt , we have

Lmax

3Tt
ln
2
δd,Tt
≤25
√

2Lmax
Tt
ln
2
δd,Tt
=

s

2Lmax

Tt
ln
2
δd,Tt
· 25Lmax
Tt
ln
2
δd,Tt
Thus

max

(
Lmax

3Tt
ln
2
δd,Tt
,

s

2ϕt,jLmax

Tt
ln
2
δd,Tt

)

=

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
This gives us the desired result.

Lemma K.2. When ˜βt,j = 5

4
q

2pj(1−pj)Lmax

Tt
ln
2
δd,Tt and Tt,j > 0, we have

pj
Tt,j
≤4

Tt
max

1, 25

16
Lmax

Tt,j
ln
2
δd,Tt



41


---Page Break---
Proof of Lemma K.2.

pj
Tt,j
= pjTt

Tt,jTt

≤
ˆpt,jTt + 5

4
q

2pj(1 −pj)LmaxTt ln
2
δd,Tt
Tt,jTt

= 1

Tt
+ 5

4

r pj

Tt,j
·

s

1
Tt,j
−pj

Tt,j
·

s

2Lmax

Tt
ln
2
δd,Tt

⇒
pj
Tt,j
≤4

Tt
max

1, 25

16
Lmax

Tt,j
ln
2
δd,Tt



Lemma K.3. If ψt,j = ˆpt,j, then

ϕt,j ≤min

16pj, 1

4


.

Proof of Lemma K.3. Recall to the definition of ϕt,j in (I.2) and ψt,j in (I.8),

ϕt,j = min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4


, ψt,j = max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


,

we have

ˆpt,j ≥25

4
Lmax

Tt
ln
2
δd,Tt
,
ϕt,j = min

4ˆpt,j, 1

4



We only need to upper bound ˆpt,j.

By (I.4),

ψt,j = ˆpt,j ≤pj + 5

2 max

(
Lmax

3Tt
ln
2
δd,Tt
,

s

2pj(1 −pj)Lmax

Tt
ln
2
δd,Tt

)

⇒
ψt,j ≤pj + 5

2 max

(
4
75ψt,j,

r

pj(1 −pj) 8

25ψt,j

)

= pj + max
 4

15ψt,j,
q

2pj(1 −pj)ψt,j



⇒
ψt,j ≤4pj

Thus ˆpt,j = ψt,j ≤4pj, and ϕt,j ≤min

16pj, 1

4
	
.

L
Upper Bound of PSεBAI+: Proof of Theorem 3.3

Theorem 3.3. The PSεBAI+ algorithm is (ε, δ)-PAC and its expected sample complexity is

˜O

min

max
xε∈Xε,x̸=xε,x∗TV(x) + TD(xε, x) + TR(x), T N
V + T N
D


.

Proof. We let τ1 denote the stopping time of PSεBAI, τ2 denote the stopping time of NεBAI, and τ
denote the stopping time of PSεBAI+when an identical sequence of samples and returns are applied
to them. By the design of these algorithms, we have

τ ≤min{τ1, τ2}.

Part I: Prove that P(˚xε /∈Xε) ≤δ.

Let event E1 = 1{τ1 < τ2} ∩1{PSεBAI returns arm xε when it terminates}. Recall that ˙xε denotes
the recommended arm of NεBAI when it terminates and ˚xε denotes the recommended arm of

42


---Page Break---
PSεBAI+ when it terminates. The design of algorithms indicates that:

˚xε = xε when E1 occurs, and ˚xε = ˙xε when Ec
1 occurs.
Moreover, with the performance guarantees of NεBAI and PSεBAI(shown in Proposition 3.1 and
Theorem 3.2 individually), we have

P[˚xε /∈Xε] = P[˚xε /∈Xε|E1] · P[E1] + P(˚xε /∈Xε|Ec
1] · P[Ec
1]
= P[xε /∈Xε|E1] · P[E1] + P[ ˙xε /∈Xε|Ec
1] · P[Ec
1]
≤δ · P[E1] + δ · P[Ec
1] = δ · (P[E1] + P[Ec
1]) = δ.

Part II: Derive the expected sample complexity of PSεBAI+.

We first derive the conditional expectation of the stopping time of PSεBAI. By the same argument
as in (J.6), Tt ≤t ≤4Tt for any t. Therefore, it is sufficient to upper bound E[Tτ1] and E[τ1] can be
upper bounded by 4E[Tτ1].

According to Lemma G.6 and Lemma G.6, when both events Good and CI occur, PSεBAI terminates
with Tτ1 satisfying Tτ1 ≤˜T, where ˜T is defined in (J.3) and as below:

˜T :=
max
xε∈Xε,x̸=xε,x∗
6400 ln 6400 · d

(∆(x∗, x) + ε)2 ln Kd/ (∆(x∗, x) + ε)2

δ

+ 6400 ln 6400 · HDE(xε, x) ln NHDE(xε, x)

δ

+ 3200
√

2 ln 3200
√

2 · NLmax
∆(x∗, x) + ε
ln

KNLmax
∆(x∗,x)+ε

δ
.

Given time step t with Tt ≥2 ˜T (thus t ≥8 ˜T), we will upper bound P[τ1 > t] in the following. We
firstly upper bound P [Tτ1 > Tt]. Note that

P [Tτ1 > Tt] = P

Tτ1 > Tt|Tτ1 ≥Tt

2 + 1

P

Tτ1 ≥Tt

2 + 1


+ P

Tτ1 > Tt|Tτ1 < Tt

2 + 1

P

Tτ1 < Tt

2 + 1


≤P

Tτ1 > Tt|Tτ1 ≥Tt

2 + 1

.

If PSεBAI fails to terminate with Tτ1 ≤Tt

2 , it implies that event Good T  T

s∈I CIs

fails, where
I = {s : Tt

2 + 1 ≤Ts ≤Tt}. Hence,

P [Tτ1 > Tt] ≤P

Tτ1 > Tt

Tτ1 ≥Tt

2 + 1


≤P

"

Goodc [  [

s∈I
CIc
s

!#

≤P [Goodc] + P

"

Good
\  [

s∈I
CIc
s

!#

≤P [Goodc] + P

" [

s∈I
CIc
s

! Good

#

(a)
≤
δ
2τ ∗+ 2

Tt
X

Ts=Tt/2+1
(Kδv,Ts + Nδd,Ts + KNδm,Ts)

(b)
≤
δ
2τ ∗+ 2

Tt
X

Ts=Tt/2+1

δ
5T 3s
≤
δ
2τ ∗+ 2
Z Tt

Tt/2

δ
5x3 dx =
δ
2τ ∗+ 3δ

5T 2
t
,

43


---Page Break---
where (a) is obtained by applying Lemma G.5 to time steps in I ∩Tt (i.e., the Exp time steps in I)
and note that there are at most two time steps t1 < t2 with Tt1 = Tt2 due to the reversion step; (b) is
derived by substituting δv,Tt =
δ
15KT 3
t , δd,Tt =
δ
15NT 3
t , δm,Tt =
δ
15KNT 3
t which are used to define

the confidence radius ρ in (3.3). By using the fact that 4Tτ1 ≥τ1, for any Tt ≥2 ˜T, we have

P [τ1 > 4Tt] ≤P [Tτ1 > Tt] ≤
δ
2τ ∗+ 3δ

5T 2
t

⇒
P [τ1 > 4Tt + i] ≤
δ
2τ ∗+ 3δ

5T 2
t
,
i = 0, 1, 2, 3

⇒
P [τ1 > t] ≤
δ
2τ ∗+
48δ
5(t −4)2 ,
∀8 ˜T ≤t ≤τ ∗.

This indicates for t ≥8 ˜T, the probability that PSεBAI does not stop after t time steps is
δ
2τ ∗+
48δ
5(t−4)2 .
Therefore, by Tonelli’s Theorem, we have

E[τ1|8 ˜T < τ1 ≤τ ∗] ≤

τ ∗−1
X

t=8 ˜T
P[τ1 > t] ≤

τ ∗−1
X

t=8 ˜T

 δ

2τ ∗+
48δ
5(t −4)2



≤1 +
Z τ ∗−1

8 ˜T −1

48δ
5(t −4)2 dt ≤2.
(L.1)

Next, we bound Eτ, the expected sample complexity of PSεBAI+. Eτ can be decomposed as
below:

Eτ ≤8 ˜T + E[τ|8 ˜T < τ ≤τ ∗] + E[τ|τ > τ ∗].
(L.2)

Since τ = min{τ1, τ2} and P[min{τ1, τ2} > t] ≤P[τ1 > t] for all t, we have

E[τ|8 ˜T < τ ≤τ ∗] =

τ ∗−1
X

t=8 ˜T
P[τ > t] =

τ ∗−1
X

t=8 ˜T
P[min{τ1, τ2} > t]

≤

τ ∗−1
X

t=8 ˜T
P[τ1 > t] = E[τ1|8 ˜T < τ1 ≤τ ∗].
(L.3)

Besides, since PSεBAI will terminate after τ ∗time steps, i.e., P(τ1 ≤τ ∗) = 1. We have

E[τ|τ ≥τ ∗] =

∞
X

t=τ ∗
P[τ ≥t] ≤1 +

∞
X

t=τ ∗+1
P[τ2 ≥t] = 1 + E[τ2|τ2 > τ ∗].
(L.4)

Lemma F.1 indicates that P[τ2 ≥t] ≤
δ
(α−1)C3(t/2)2 for all t ≥T0, where

T0 = 768(8Lmax + 25d)

(ε + ∆min)2
ln 768KC3(8Lmax + 25d)

(ε + ∆min)2 δ
,
C3 =

∞
X

n=1
n−3.

Since the order of T0 is apparently larger than τ ∗, we have T0 ≤τ ∗. Then, with the same method
as in the proof of Proposition 3.1, the conditional sample complexity of NεBAI can be bounded as
follows:

E[τ2|τ2 > τ ∗] ≤
Z +∞

τ ∗
P(τ2 ≥x) dx ≤
Z +∞

τ ∗
δ
(α −1)C3(x/2)2 dx =
δ
C3τ ∗.
(L.5)

Substituting terms in (L.2) with (L.1), (L.3), (L.4) and (L.5), we have

Eτ ≤8 ˜T + 3 +
δ
C3τ ∗.

Besides,

Eτ ≤Eτ2

(a)
≤T0 +
δ
(α −1)(α −2)(T0/2)α−2 ,

Where (a) is shown in (F.1).

44


---Page Break---
Altogether, we have

Eτ ≤min

8 ˜T + 3 +
δ
C3τ ∗, 8 ˜T + 3 +
δ
C3τ ∗


.

M
Analysis of Lower Bound: Proof of Theorem 4.1

The dynamics for under our PSLB model is displayed in Dynamics 1.

Dynamics 1 Dynamics for piecewise-stationary linear bandits

1: The instance: Λ = (X, Θ, Pθ, C)
2: while the algorithm does not stop at time step t do
3:
if t ∈C then
4:
The environment samples θ∗
jt ∼Pθ
5:
else
6:
θ∗
jt = θ∗
jt−1 (the environment does not change)
7:
end if
8:
The agent samples an arm xt based on the history up to time t −1.
9:
The reward Yt,xt = x⊤
t θ∗
jt + ηt is revealed to the agent.
10: end while
11: Recommend an ε-best arm ˆxε.

To derive the lower bound in Theorem 4.1, we investigate two environments different from the one
defined in Section 2 (and as in Dynamics 1):
• Dynamics 2: the agent observes the index of current context jt, and the environment reduces to
contextual linear bandits; the definition of Dynamics 2 and the lower bound under it are detailed in
Appendix M.1.
• Dynamics 3: the agent observes the changepoints in C and context vector θ∗
jt’s, and hence she
solely needs to estimate the distribution of contexts; the definition of Dynamics 3 and the lower
bound under it are detailed in Appendix M.2.

We first derive a lower bound for (ε, δ)-BAI algorithms in Dynamics 2, that is, in contextual linear
bandits.
Corollary M.1. For any (ε, δ)-PAC algorithm π, there exists an instance Λ = (X, Θ, Pθ, C) with
Dynamics 2 such that

E[τ] ≥Tε(Λ) log
1
2.4δ .

where
Tε(Λ)−1 =
max
{vj∈∆X }N
j=1
min
Λ′∈AltΘ(Λ)

N
X

j=1
pj
X

x∈X
vj,x
(x⊤(θ∗
j −θ′
j))2

2

is defined in Theorem 4.1. In addition, when |Xε| = 1,

Tε(Λ) =
min
{vj∈∆X }N
j=1
max
x̸=x∗

PN
j=1 pj∥x∗−x∥2
A(vj)−1

(∆(x∗, x) + ε)2
.

This lower bound generalizes the result of [16] to the linear bandit setting.

We next study Dynamics 3. In this setting, the agent solely needs to estimate the distribution of
contexts Pθ with context samples. Once the agent obtains a good estimate of Pθ, she can identify
an ε-optimal arm w.h.p. Hence, the lower bound on the complexity of an (ε, δ)-BAI algorithm is
the product of the minimum length of a stationary segment and the minimum number of context
samples/changepoints needed for distribution estimation.
Corollary M.2. For any (ε, δ)-PAC algorithm π, there exists an instance Λ = (X, Θ, Pθ, C) with
Dynamics 3 such that Eτ ≥cNC, where cNC is the N th
C changepoint in the changepoint sequence C.

45


---Page Break---
Altogether, the sample complexity of an (ε, δ)-BAI algorithm in Dynamics 2 and 3 build up the
lower bound in Theorem 4.1.

M.1
Lower Bound for ε-BAI in Contextual Linear Bandits

In this section, we consider a sub-problem where the index of the current context jt is revealed at
each time step t. The original piecewise-stationary problem becomes a contextual problem whose
dynamics is presented in Algorithm 2.
As more information is provided to the agent, the lower

Dynamics 2 Dynamics for contextual linear bandits

1: The instance: Λ = (X, Θ, Pθ, C)
2: while the algorithm does not stop at time step t do
3:
if t ∈C then
4:
The environment samples θ∗
jt ∼Pθ
5:
else
6:
θ∗
jt = θ∗
jt−1 (the environment does not change)
7:
end if
8:
Reveal jt to the agent.
9:
The agent samples an arm xt based on the history up to time t −1.
10:
The reward Yt,xt = x⊤
t θ∗
jt + ηt is revealed to the agent.
11: end while
12: Recommend an ε-best arm ˆxε.

bound for this sub-problem is smaller than the one for the original problem.
Corollary M.1. For any (ε, δ)-PAC algorithm π, there exists an instance Λ = (X, Θ, Pθ, C) with
Dynamics 2 such that

E[τ] ≥Tε(Λ) log
1
2.4δ .

where
Tε(Λ)−1 =
max
{vj∈∆X }N
j=1
min
Λ′∈AltΘ(Λ)

N
X

j=1
pj
X

x∈X
vj,x
(x⊤(θ∗
j −θ′
j))2

2

is defined in Theorem 4.1. In addition, when |Xε| = 1,

Tε(Λ) =
min
{vj∈∆X }N
j=1
max
x̸=x∗

PN
j=1 pj∥x∗−x∥2
A(vj)−1

(∆(x∗, x) + ε)2
.

For simplicity, we consider the noise model is the Clipped Gaussian Distribution CN(1), i.e.,
η ∼CN(1) or Pη = CN(1).

Definition M.3. A random variable x follows the Clipped Gaussian Distribution with parameter σ,
denoted by x ∼CN(σ), if it has the probability distribution function

f(x) =
1

(2Φ(σ) −1) ·
√

2πσ2 exp

−x2

2σ2


,
∀x ∈[−1, 1]

where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.

Some notations are introduced here:

• Given an instance Λ = (X, Θ, Pθ, C), define the alternative instance Λ′ = (X, Θ′, Pθ′, C) with
respect to Λ, where Θ′ = (θ′
1, . . . , θ′
n) ∈Rd×N and Pθ′[θ′
j] = Pθ[θ∗
j ], s.t. there ∃x ∈X \
Xε, ∀xε ∈Xε, s.t., x⊤
ε Eθ′∼Pθ′ < x⊤Eθ′∼Pθ′ −ε. We denote the set containing all the alternative
instance (w.r.t. Λ) as AltΘ(Λ).
• Ht = (xs, Ys,xs, js)t−1
s=1 is the observation history up to but not include time t.
• Nj,x(t) = Pt
s=1 1{xs = x, js = j} is the number of times arm in which x is sampled under
context j. And Nj(t) := P

x∈X Nj,x(t) is the number of times in which context j appears.
• kl(p, q) = KL(Bern(p), Bern(q)) is the KL divergence between two Bernoulli distributions with
parameters p and q.

46


---Page Break---
Therefore, the probability of the observation history Ht+1 = (xs, Ys,xs, js)t
s=1 is

PπΛ
h
(xs, Ys,xs, js)t
s=1
i
=

lt
Y

l=1
Pθ[θ∗
jcl ]

Ll−1
Y

s=0
π(xcl+s|Hcl+s)Pη(Ycl+s,xcl+s|xcl+s, θ∗
jcl )

Then the log-likelihood between the two instance up to time t, given the observed data, is

Lt
h
(xs, Ys,xs, js)t
s=1
i

= log
PπΛ
h
(xs, Ys,xs, js)t
s=1
i

PπΛ′
h
(xs, Ys,xs, js)t
s=1
i

(a)
= log




Qlt
l=1 Pθ[θ∗
jcl ] QLl−1
s=0 π(xcl+s|Hcl+s)P(Ycl+s,xcl+s|xcl+s, θ∗
jcl )
Qlt
l=1 Pθ′[θ′
jcl ] QLl−1
s=0 π(xcl+s|Hcl+s)P(Ycl+s,xcl+s|xcl+s, θ′
jcl )





= log




Qlt
l=1
QLl−1
s=0 Pη(Ycl+s,xcl+s −x⊤
cl+sθ∗
jcl )
Qlt
l=1
QLl−1
s=0 Pη(Ycl+s,xcl+s −x⊤
cl+sθ′
jcl )





=

lt
Y

l=1

Ll−1
X

s=0
log
 exp(−η2
cl+s/2)
exp(−(η′
cl+s)2/2)



(b)
=

lt
X

l=1

Ll−1
X

s=0

−(Ycl+s,xcl+s −x⊤
cl+sθ∗
jcl )2 + (Ycl+s,xcl+s −x⊤
cl+sθ′
jcl )2

2

(c)
=

lt
X

l=1

Ll−1
X

s=0

x⊤
cl+sϑjcl (2ηcl+s + x⊤
cl+sϑjcl )

2

where (a) utilizes Pθ[θ∗
jcl ] = Pθ′[θ′
jcl ], (b) makes use of the relationship between the arm and
observation and in (c) ϑjcl := θ∗
jcl −θ′
jcl. Thus, the expectation of the log-likelihood is (in the
following, the expectation is taken under instance Λ and algorithm π)

E[Lt] = E

" lt
X

l=1

Ll−1
X

s=0

x⊤
cl+sϑjcl (2ηcl+s + x⊤
cl+sϑjcl )

2

#

= E

" lt
X

l=1

Ll−1
X

s=0

ϑ⊤
jcl xcl+sx⊤
cl+sϑjcl
2

#

= 1

2E

" lt
X

l=1
ϑ⊤
jcl

 Ll−1
X

s=0
xcl+sx⊤
cl+s

!

ϑjcl

#

= 1

2E



X

x∈X
1{xcl+s = x}

N
X

j=1
1{jcl = j}

lt
X

l=1
ϑ⊤
jcl

 Ll−1
X

s=0
xcl+sx⊤
cl+s

!

ϑjcl





= 1

2

X

x∈X

N
X

j=1
E [Nj,x(t)] ϑ⊤
j xx⊤ϑj

= 1

2E[t]

N
X

j=1

E[Nj(t)]

E[t]

X

x∈X

E [Nj,x(t)]

E[Nj(t)] ϑ⊤
j xx⊤ϑj
(M.8)

As the lengths of all the stationary phases are upper bounded, i.e., Ll ≤Lmax, ∀l ∈N, then by
Wald’s Lemma,

E[Nj(t)]

E[t]
= E[t]Pθ[θ∗
j ]
E[t]
= Pθ[θ∗
j ] = pj
(M.9)

47


---Page Break---
The above also holds for t = τ where τ is a stopping time. According to Lemma 19 in [31],

E[Lτ] ≥sup
E∈Fτ
kl(PπΛ[E], PπΛ′[E])
(M.10)

where Fτ = σ(Hτ+1). In addition, let E = {the recommended arm ˆxε /∈Xε}, as the algorithm π is
δ-PAC, then

kl(PπΛ[E], PπΛ′[E]) ≥kl(1 −δ, δ) ≥log
1
2.4δ
(M.11)

From (M.8), (M.9), (M.10) and (M.11), we conclude that

1
2E[τ]

N
X

j=1
pj
X

x∈X

E [Nj,x(τ)]

E[Nj(τ)] ϑ⊤
j xx⊤ϑj ≥log
1
2.4δ

⇒
min
Λ′∈AltΘ(Λ)
1
2E[τ]

N
X

j=1
pj
X

x∈X

E [Nj,x(τ)]

E[Nj(τ)] ϑ⊤
j xx⊤ϑj ≥log
1
2.4δ

⇒
max
{vj∈∆X }N
j=1
min
Λ′∈AltΘ(Λ)
1
2E[τ]

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj ≥log
1
2.4δ

⇔
E[τ] ≥Tε(Λ) log
1
2.4δ
where

Tε(Λ)−1 =
max
{vj∈∆X }N
j=1
min
Λ′∈AltΘ(Λ)
1
2

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj.

The solution to the above optimization problem is in general intractable, even for the stationary
case [32]. We can establish a connection of the above problem to the stationary ε-best identification
problem in linear bandits when we assume the change of the latent vectors are the same, i.e.,
ϑ1 = . . . = ϑN.

M.1.1
Connection with the Stationary Case

In the alternative instance AltΘ(Λ), ∃x ∈X \ Xε, for any arm xε ∈Xε, s.t.

N
X

j=1
pjx⊤
ε θ′
j + ε <

N
X

j=1
pjx⊤θ′
j
(M.12)

⇔

N
X

j=1
pj(xε −x)⊤θ′
j + ε < 0

⇔
−

N
X

j=1
pj(xε −x)⊤θ′
j +

N
X

j=1
pj(xε −x)⊤θ∗
j >

N
X

j=1
pj(xε −x)⊤θ∗
j + ε

⇔

N
X

j=1
pj(xε −x)⊤ϑj >

N
X

j=1
pj(xε −x)⊤θ∗
j + ε = ∆(xε, x) + ε

Thus,

AltΘ(Λ)

=




(X, Θ′, Pθ, C) : θ′
j = θ∗
j −ϑj, ∃x ∈X \ Xε,

N
X

j=1
pj(xε −x)⊤ϑj > ∆(xε, x) + ε, ∀xε ∈Xε




.

Define Alt(Λ)restricted ⊂AltΘ(Λ), with the additional constraint that ϑ1 = ϑ2 = · · · = ϑN. Then
we have

Alt(Λ)restricted

48


---Page Break---
=




(X, Θ′, Pθ, C) : θ′
j = θ∗
j −ϑ1, ∃x ∈X \ Xε,

N
X

j=1
pj(xε −x)⊤ϑj > ∆(xε, x) + ε, ∀xε ∈Xε




.

Note that in stationary linear bandits, the instance can be characterized by the arm set X ⊂Rd
and the latent vector θ ∈Rd. Define the alternative instance in linear bandits [32] for the instance
(X, Eθ∼Pθθ)

Alt((X, Eθ∼Pθθ))stationary
=

(X, θ′) : θ′ = Eθ∼Pθθ −ϑ1, (xε −x)⊤ϑ1 > (xε −x)⊤Eθ∼Pθ + ε, ∀xε ∈Xε
	
.

Note that

max
{vj∈∆X }N
j=1
min
Λ′∈AltΘ(Λ)
1
2E[τ]

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj

≤
max
{vj∈∆X }N
j=1
min
Λ′∈Alt(Λ)restricted
1
2E[τ]ϑ⊤
1



X

x∈X




N
X

j=1
pjvj,x



xx⊤



ϑ1

(a)
= max
¯v∈∆X
min
Λ′∈Alt(Λ)restricted
1
2E[τ]ϑ⊤
1

 X

x∈X
¯vixx⊤
!

ϑ1

= max
¯v∈∆X
min
Λ′∈Alt((X,Eθ∼Pθ θ))stationary
1
2E[τ]ϑ⊤
1

 X

x∈X
¯vixx⊤
!

ϑ1

where in (a) we denote ¯v := PN
j=1 pjvj as a mixture of {vj}N
j=1. In other words, the max-min
problem becomes the one for the ε-best arm identification problem in stationary linear bandits.
According to [32], the solution to the last optimization problem above is in general intractable.

However, the optimization problem can be simplified under some simple cases, e.g., the set of ε-best
arm is a singleton [32]. In the next subsection, a lower bound for the original problem with Xε being
a singleton will be derived.

M.1.2
A Simple Case: |Xε| = 1

Assume that the set of ε-best arm is a singleton, i.e., Xε = {x∗}, we will solve the original
optimization problem:

min
Λ′∈AltΘ(Λ)
1
2

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj

we extend the procedures in [1] to the piecewise-stationary setup.
Lemma M.4.

min
Λ′∈AltΘ(Λ)
1
2

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj ≥1

2 min
x̸=x∗
(∆(x∗, x))2
PN
j=1 pj∥x∗−x∥2
A(vj)−1

Proof. Note that Λ′ differs from Λ in the context matrix Θ. By the derivations in (M.12), there exists
arm x ̸= x∗, s.t.,
N
X

j=1
pj(x∗−x)⊤ϑj > ∆(xε, x) + ε

Therefore, the optimization problem becomes

min
ϑ1,...,ϑN
1
2

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj

s.t.
∃x ∈X \ {x∗},

N
X

j=1
pj(x∗−x)⊤ϑj ≥∆(x∗, x) + ε + a =: ∆a+ε(x∗, x)
(M.13)

49


---Page Break---
where a > 0. The Lagrangian function for this problem is

L(ϑ1, . . . , ϑN, λ) = 1

2

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj + λ(−

N
X

j=1
pj(x∗−x)⊤ϑj + ∆a+ε(x∗, x))

Then
∂L
∂ϑj
= 0 ⇒pj
X

x∈X
vj,xxx⊤ϑj −λpj(x∗−x) = 0 ⇒A(vj)1/2ϑj = λA(vj)−1/2(x∗−x)

∂L
∂λ = 0 ⇒

N
X

j=1
pj(x∗−x)⊤ϑj = ∆a+ε(x∗, x)

This indicates that

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj =

N
X

j=1
pj∥ϑj∥2
A(vj)

≥

PN
j=1 pj∥ϑj∥A(vj)∥x∗−x∥A(vj)−1
2

PN
j=1 wj∥x∗−x∥2
A(vj)−1

=

PN
j=1 pjϑ⊤
j (x∗−x)
2

PN
j=1 pj∥x∗−x∥2
A(vj)−1

=
(∆a+ε(x∗, x))2
PN
j=1 pj∥x∗−x∥2
A(vj)−1

Let a →0, we have

N
X

j=1
pj
X

x∈X
vj,xϑ⊤
j xx⊤ϑj ≥
(∆(x∗, x) + ε)2
PN
j=1 pj∥x∗−x∥2
A(vj)−1

Due to (M.13), we only require there exists x ̸= x∗, such that the constraint is satisfied, therefore,

1
2

N
X

j=1
pj∥ϑj∥2
A(vj) ≥1

2 min
x̸=x∗
(∆(x∗, x) + ε)2
PN
j=1 pj∥x∗−x∥2
A(vj)−1

By Lemma M.4, the stopping time can be lower bounded as

E[τ] ≥2 log
1
2.4δ
min
{vj∈∆X }N
j=1
max
x̸=x∗

PN
j=1 pj∥x∗−x∥2
A(vj)−1

(∆(x∗, x) + ε)2

This lower bound indicates that, (1) a good algorithm should actively detects and makes use of the
contextual information to facilitate the arm identification process. (2) our lower bound extends the
result of [16] to the linear bandits case.

The above lower bound can be further lower bounded if we restrict v1 = . . . = vN.

Lemma M.5. Let SPD(d) := {A : A ∈Rd×d, A > 0} denote the set of SPD matrices of dimension
d × d. Given any x ∈Rd, define the function f : SPD(d) →R, f(A) = x⊤A−1x, then f is convex.

Proof of Lemma M.5. Given any A, B ∈SPD(d), define

g : {t ∈R : A + tB ∈SPD(d)} →R, g(t) = x⊤(A + tB)−1x

It is suffice to prove g is convex.

g′(t) = −x⊤(A + tB)−1B(A + tB)−1x

g′′(t) = 2x⊤(A + tB)−1B(A + tB)−1B(A + tB)−1x

50


---Page Break---
= 2
 
x⊤(A + tB)−1B

(A + tB)−1  
B(A + tB)−1x

≥0

Therefore, g is convex.

Given Lemma M.5, we have

N
X

j=1
pj∥x∗−x∥2
A(vj)−1 = (x∗−x)⊤




N
X

j=1
pjA(vj)−1



(x∗−x)

≥(x∗−x)⊤




N
X

j=1
pjA(vj)





−1

(x∗−x)

= (x∗−x)⊤




N
X

j=1
pj
X

x∈X
vj,xxx⊤





−1

(x∗−x)

= (x∗−x)⊤



X

x∈X




N
X

j=1
pjvj,x



xx⊤





−1

(x∗−x)

= (x∗−x)⊤A(¯v)−1(x∗−x)

where ¯v := PN
j=1 pjvj ∈∆X and the inequality becomes equality when v1 = · · · = vn. Thus,

E[τ] ≥2 log
1
2.4δ
min
{vj∈∆X }N
j=1
max
x̸=x∗

PN
j=1 pj∥x∗−x∥2
A(vj)−1

(∆(x∗, x) + ε)2

≥2 log
1
2.4δ min
¯v∈∆X max
x̸=x∗
∥x∗−x∥2
A(¯v)−1

(∆(x∗, x) + ε)2 .
(M.14)

The last term mimics the lower bound in the stationary linear bandits with the latent vector Eθ∼Pθθ.

In addition, if we let p1 = 1 and pj = 0 for all j ̸= 1 in the instance, our lower bound can be
simplified to the lower bound in stationary linear bandits with latent vector θ∗
1 [24]

E[τ] ≥2 log
1
2.4δ
min
{vj∈∆X }N
j=1
max
x̸=x∗

PN
j=1 pj∥x∗−x∥2
A(vj)−1

(∆(x∗, x) + ε)2

= 2 log
1
2.4δ
min
v1∈∆X max
x̸=x∗
∥x∗−x∥2
A(v1)−1

(∆1(x∗, x) + ε)2 .
(M.15)

M.2
Lower Bound on the Number of changepoints

In this section, we consider an even easier problem: all the contextual information is known to the
agent, except for the distribution Pθ. The dynamics is displayed in Dynamics 3.

Dynamics 3 Dynamics for an easier problem

1: The instance: Λ = (X, Θ, Pθ, C)
2: while the algorithm does not stop at time step t do
3:
if t ∈C then
4:
The agent acknowledges t ∈C.
5:
The environment samples θ∗
jt ∼Pθ
6:
else
7:
θ∗
jt = θ∗
jt−1 (the environment does not change)
8:
end if
9:
The agent observes θ∗
jt
10: end while
11: Recommend an ε-best arm ˆxε.

51


---Page Break---
We are going to consider the alternative instances with respect to the distribution on θ, i.e., Λ′ =
(X, Θ, P ′
θ, C). where ∃x ∈X \ Xε, ∀xε ∈Xε, s.t., x⊤
ε Eθ∼P ′
θ < x⊤Eθ∼P ′
θ −ε. Denote the set
of all the alternative instance (w.r.t. Λ) as AltP (Λ). According to the Pinsker’s inequality, let
E := {the recommended arm ˆxε /∈Xε}, then for any δ-PAC algorithm π,

PπΛ[E] + PπΛ′[Ec] ≥1

2 exp (−KL(PπΛ, PπΛ′)) ,
∀Λ′ ∈AltP (Λ)

⇒
4δ ≥exp (−KL(PπΛ, PπΛ′)) ,
∀Λ′ ∈AltP (Λ)

⇒
KL(PπΛ, PπΛ′) ≥ln 1

4δ ,
∀Λ′ ∈AltP (Λ)

⇒
min
Λ′∈AltP (Λ) KL(PπΛ, PπΛ′) ≥ln 1

4δ

⇒
E[lt]
min
Λ′∈AltP (Λ) KL(Pθ, P ′
θ) ≥ln 1

4δ

⇒
E[lt] ≥
1
minΛ′∈AltP (Λ) KL(Pθ, P ′
θ) ln 1

4δ
(M.16)

We will give an upper bound on the denominator. Note that

min
Λ′∈AltP (Λ) KL(Pθ, P ′
θ) =
min
x∈X\Xε min
Λx KL(Pθ, P x
θ )
(M.17)

where Λx is an alternative instance with distribution P x
θ s.t. x⊤Eθ∼P x
θ θ−ε > x⊤
ε Eθ∼P x
θ θ, ∀xε ∈Xε.

Given any x /∈Xε, we denote the shorthand notation qj = P x
θ [θ∗
j ]. We have

N
X

j=1
qj∆j(x, xε) ≥ε + a, for any a > 0, xε ∈Xε

Fix a > 0 which is sufficiently small, we have the following optimization problem:

min
q∈∆N

N
X

j=1
pj ln pj

qj

s.t.

N
X

j=1
qj = 1

N
X

j=1
qj∆j(xε, x) + ε + a ≤0, ∀xε ∈Xε

If such alternative distribution q exists, let L denote the augmented Lagrangian function

L(q, λ, {λxε}xε∈Xε) =

N
X

j=1
pj ln pj

qj
+ λ
 N
X

j=1
qj −1

+
X

xε∈Xε
λxε(qj∆j(xε, x) + ε + a)

By the KKT conditions, we have:
∂L
∂qj
= −pj

qj
+ λ +
X

xε∈Xε
λxε∆j(xε, x) = 0, ∀j ∈[N]
(M.18)

∂L
∂λ =

N
X

j=1
qj −1 = 0

∂L
∂λxε
=

N
X

j=1
qj∆j(xε, x) + ε + a = 0, ∀xε ∈Xε

or λxε = 0 for some xε ∈Xε, which indicates some conditions are not satisfied. The equations
above give

qj =
pj
1 + P

xε∈Xε λxε(∆j(xε, x) + ε + a), ∀j ∈[N]
(M.19)

52


---Page Break---
By solving

N
X

j=1

pj
1 + P

xε∈Xε λxε(∆j(xε, x) + ε + a) −1 = 0
(M.20)

which is a polynomial with N −1 degree. Thus, an explicit solution for any instance is not applicable.
However, there are some cases where we can get an estimate of λxε
Lemma M.6. Denote j0 = arg minj∈[N] ∆j(x∗, x) and j1 = arg maxj∈[N] ∆j(x∗, x). When

|Xε| = 1, −∆j0(x∗, x) −2ε > ∆j1(x∗, x) > ε and PN
j=1 pj(∆j(x∗, x) + ε + a)3 ≤0, the solution
to (M.20) is upper bounded by

λi∗≤
∆(x∗, x) + ε + a
PN
j=1 pj(∆j(x∗, x) + ε + a)2

Proof of Lemma M.6. When the ε-best arm set is a singleton, (M.20) becomes (where a is sufficiently
small)

N
X

j=1

pj
1 + λx∗(∆j(x∗, x) + ε + a) −1 = 0

As (M.19) is a probability distribution, we need 1 + λx∗(∆j0(x∗, x) + ε + a) ≥0, which indicates
λx∗≤
−1
∆j(x∗,x)+ε+a. By the condition on j0 and j1, 0 < λx∗(∆j(x∗, x) + ε + a) < 1, ∀j ∈[N].

Therefore, by using
1
1+x = 1 −x + x2 −x3 +
x4
1+x for |x| < 1, we can expand the equation as
follows

1 =

N
X

j=1

pj
1 + λx∗(∆j(x∗, x) + ε + a)

≥

N
X

j=1
pj
  
1 −λx∗(∆j(x∗, x) + ε + a) + (λx∗(∆j(x∗, x) + ε + a))2

−(λx∗(∆j(x∗, x) + ε + a))3

⇔
0 ≥−λ2
x∗

N
X

j=1
pj(∆j(x∗, x) + ε + a)3 + λx∗

N
X

j=1
pj(∆j(x∗, x) + ε + a)2

−

N
X

j=1
pj(∆j(x∗, x) + ε + a)

⇒
λx∗≤

PN
j=1 pj(∆j(x∗, x) + ε + a)
PN
j=1 pj(∆j(x∗, x) + ε + a)2 =
∆(x∗, x) + ε + a
PN
j=1 pj(∆j(x∗, x) + ε + a)2 .

In general, when we get the {λxε}xε∈Xε and plug it in (M.19), the alternative distribution q is
obtained. Finally, we let a →0.

This gives the lower bound for the number of changepoints that need to be observed. A coarse
estimation of the KL divergence without solving (M.20) can be done as follows

N
X

j=1
pj ln pj

qj
=

N
X

j=1
pj ln

 

1 +
X

xε∈Xε
λxε(∆j(xε, x) + ε)

!

≤

N
X

j=1
pj
X

xε∈Xε
λxε(∆j(xε, x) + ε)

=
X

xε∈Xε
λxε(∆(xε, x) + ε).

53


---Page Break---
Note that the solution of λxε depends on ∆(xε, x) (e.g. Lemma M.6), so the final solution is of order
(∆(xε, x) + ε)2 for a given x /∈X. This is the solution to the inside minimization problem in (M.17).
The final lower bound will be

min
Λ′∈AltP (Λ) KL(Pθ, P ′
θ) =
min
x∈X\Xε min
Λx KL(Pθ, P x
θ )

≤min
x/∈Xε

X

xε∈Xε
λxε(∆(xε, x) + ε).

The lower bound is

E[lt] ≥max
x/∈Xε
1
P

xε∈Xε λxε(∆(xε, x) + ε) ln 1

4δ

where λxε is the solution to (M.20).

With the setup in Lemma M.6, we have

min
Λ′∈AltP (Λ) KL(Pθ, P ′
θ) =
min
x∈X\Xε min
Λx KL(Pθ, P x
θ )

≤min
x̸=xε
(∆(x∗, x) + ε)2
PN
j=1 pj(∆j(x∗, x) + ε)2

and the lower bound is

E[lt] ≥max
x̸=xε

PN
j=1 pj(∆j(x∗, x) + ε)2

(∆(x∗, x) + ε)2
ln 1

4δ .

Remark M.7. We give some comments on the existence and uniqueness of the solution to (M.20).

Existence:

• It is possible that (M.20) does not have a solution. For instance, consider a three-arm instance:
x(1) = (1, 0.5), x(2) = (0.5, 1), x(3) = (0.6, 0.6), θ∗
1 = (1, 0), θ∗
2 = (0, 1), Pθ = (0.5, 0.5), ε =
0.1. We have x(1)⊤Eθ∼Pθ = x(2)⊤Eθ∼Pθ = 0.75 and x(3)⊤Eθ∼Pθ = 0.6, thus x(3) is not an
ε-best arm. Furthermore, there does not exist an alternative distribution q, such that x(3) is the
best arm and neither x(1), x(2) is ε-best. Under such case, the lower bound on E[lt] is 0 (we regard
minx∈∅f(x) = +∞by convention).
• It is possible that (M.20) does not have a solution and it is unnecessary to estimate Pθ. For instance,
consider a two-arm instance: x(1) = (1, 0.5), x(2) = (0.5, 0.1), θ∗
1 = (1, 0), θ∗
2 = (0, 1), Pθ =
(0.5, 0.5), ε = 0.1. Arm x(1) is better than arm x(2) under all contexts. Therefore, no matter what
Pθ is, arm x(1) is the ε-best arm. Under such cases, there may exist an algorithm and it is sufficient
for it to determine the best arm if the context vectors are well-approximated. There is no need to
estimate Pθ.
• A necessary condition for the existence of the solution of (M.20) is: there exists x /∈Xε, for any
xε ∈Xε, there ∃j(xε) ∈[N], s.t.∆j(xε)(xε, x) < −ε. In other words, for each ε-best arm xε,
there is at least one context in which the alternative arm x is better than xε by at least ε.
• A sufficient condition for the existence of the solution of (M.20) is: there exists x /∈Xε and j ∈[N],
s.t.∆j(xε, x) < −ε, ∀xε ∈Xε. In other words, x is better than any ε-best arm under context j. In
the alternative instance, we can lift P ′
θ[θ∗
j ] close to 1 so that arm x becomes the ε-best arm and
X ′
ε ∩Xε = ∅.

Uniqueness:

• The uniqueness of the solution is not guaranteed, as the KKT conditions (M.18) is only a necessary
condition for the solution. We need to look for the solution that minimizes the KL divergence.
• A sufficient condition for the uniqueness of the solution is (if it exists, which indicates ∃j ∈
[N], s.t.∆j(x∗, x) < −ε): |Xε| = 1. Specifically, denote f(λx∗) := PN
j=1
pj
1+λx∗(∆j(x∗,x)+ε+a),
we have

∂f
∂λx∗=

N
X

j=1

−pj(1 + λx∗(∆j(x∗, x) + ε + a))

1 + λx∗(∆j(x∗, x) + ε + a)

54


---Page Break---
∂2f
∂λ2
x∗=

N
X

j=1

2pj(1 + λx∗(∆j(x∗, x) + ε + a))2

1 + λx∗(∆j(x∗, x) + ε + a)
> 0

As f(0) = 1,
∂f
∂λx∗(0) = −1 and
∂2f
∂λ2
x∗> 0, so there is exactly 1 solution λx∗> 0.

N
More Examples and Details

In this section, we firstly provide one more example to illustrate the tightness of our derived upper
bound lower bounds, indicating the efficiency of our PSεBAI+ algorithm. In addition, we can
observe how the upper and lower bounds are affected by the level of piecewise non-stationarity and
whether our PSεBAI+ algorithm can reduce the influence manifested by Lmax. After that, we
present a proof sketch of the results in Corollaries 5.1 and N.1 in Appendix N.1.

Example 2. Instance Λ = (X, Θ, Pθ, C) is with

• d arms: x(i) = ei, i ∈[d].
• N = d contexts: θ∗
1 = (a, 0, 0, . . . , 0)⊤, θ∗
2 = (a, b, b, . . . , b)⊤−bej, j ≥2, where b > a >
ε, b −a > ε
• Context distribution: pj = p, j ≥2 and p1 = 1 −(N −1)p, where p ∈(0,
a−ε
(N−2)b).

Under Example 2, we have

∆(x(1), x(i)) = (1 −(N −2)p) · a −(N −2)p · (b −a) > ε

∆j(x(1), x(i)) = −b + a < −ε,
i ≥2, j ̸= 1, i

Thus, (1) x(1) is the unique ε-best arm; (2) {x(i)}i≥2 are equivalent, and ∆min := ∆(x(1), x(i)); (3)
for any i ≥2, x(i) can be an ε-optimal arm under some alternative distributions.

Corollary N.1. Firstly, for the instances defined in Example 2, we have

HDE ≤16(N −2)Lmax

(∆min + ε)2

 

(a + ε)2 + (b −a −ε)2
!

,

and the sample complexity of the PSεBAI+ is tight up to (NLmax/Lmin) and logarithmic factors.
We also further observe some specific instances: (i) when p →0+, with ∆min = minx̸=x∗∆(x∗, x),
we have

E[τ]∗

ln(1/δ) ∈˜O

 

min

(
d

(∆min + ε)2 + NLmax

∆min + ε,
Lmax + d

(∆min + ε)2

)!

\
Ω

(

max

(
d

(∆min + ε)2 , Lmin(b −a −ε)

∆min + ε

))

.

(ii) When p →

a−ε)
(N−2)b
−
and (a + ε)2 + (b −a −ε)2 = Ω(1), we have HDE =
NLmax
(∆(x(1),x(i))+ε)
2

and
E[τ]∗

ln(1/δ) ∈˜O

min

HDE, d + Lmax

ε2

 \
Ω
d + Lmin

ε2


.

The upper bounds are achieved by the PSεBAI+ algorithm.

We can observe from Corollary N.1 that

• In case (i), when p →0+, p1 →1 −(N−1)(a−ε)
(N−2)b
and pj →

a−ε)
(N−2)b

, and thus the instance tends
to be non-stationary and ∆min →ε. We will obtain

Eτ
ln(1/δ) ∈˜Θ

d
(∆min + ε)2


,

indicating that our algorithm can also reduce the impact of Lmax.

55


---Page Break---
• In case (ii), the upper and lower bounds are with the same order, and the difference is solely
manifested by a additive term Lmax −Lmin, suggesting that PSεBAI+ is near optimal and again,
PSεBAI+ mitigates the impact of Lmax.

N.1
Analysis of examples

Recall the instance Λ = (X, Θ, Pθ, C) in Example 1:

Example 1. Instance Λ = (X, Θ, Pθ, C) is with (i) 2d −1 arms: x(1) = e1, x(i) = ei, x(d+i−1) =
e1 cos ϕ+ei sin ϕ for all i ∈{2, . . . , d} where ϕ ∈[0, π/4), (ii) 2d−2 contexts: θ∗
j± = e1 cos ϕ±
ej+1 sin ϕ for all j ∈[d −1], (iii) Context distribution: pj = 1/N for all j ∈[N].

Corollary 5.1. For the instance defined in Example 1, we have HDE(xε, x) = ˜O(NLmax) for all
(xε, x) ∈Xε × (X \ Xε). In addition, if ε < (cos ϕ)(1 −cos ϕ), we have

E[τ]∗

ln(1/δ) ∈˜Θ

(1+f(ϕ)) ·
d
(∆x(1),x(d+1) +ε)2


,
(5.1)

where E[τ]∗is the minimal expected sample complexity over all (ε, δ)-PAC algorithms and f : R →R
satisfies f(ϕ) →0 as ϕ →0+. The upper bound in (5.1) is achieved by PSεBAI+.

Proof of Corollary 5.1. As µx(1) = cos ϕ, µx(i) = 0, µx(d+i) = cos2 ϕ for i = 2, . . . , d and ε ∈(1−
cos ϕ, cos ϕ), x(1) is the best arm and x(i), i = 2, . . . , d are not ε-best arms and x(d+i), i = 2, . . . , d
are ε-best arms. ∆min = cos ϕ −cos2 ϕ.

When ε < cos ϕ −cos2 ϕ = ∆min, x(1) is the unique ε-best arm. By solving (M.14), we see that
there exists a continuous function f(ϕ) such that f(ϕ) →0 as ϕ →0 and we can get the lower bound
for the VE term as

Ω

(1 + f(ϕ)) ·
d
(∆(x(1), x(d+1)) + ε)2 ln 1

δ


.
(N.1)

As (x(1) −x(i))⊤θ∗
j± > 0 for i = 2, . . . , d, j ∈[d −1], x(i) cannot be an ε-best arm under any
alternative distribution, we only need to consider x(d+i), i = 2, . . . , d, which are equivalent. Given
any x(d+i), by solving (M.20), we can upper bound

λx(1) ≤
∆min + ε + sin2 ϕ
−(∆i+(x(1), x(d+i)) + ε)(∆i−(x(1), x(d+i)) + ε),

and thus the number of change points or context samples can be lower bounded as

E[lτ] ≥−(∆i+(x(1), x(d+i)) −ε)(∆i−(x(1), x(d+i)) + ε)

(∆min + ε + sin2 ϕ)(∆min + ε)

= (1 −cos ϕ −ε)(1 + cos ϕ −2 cos2 ϕ + ε)

(∆min + ε + sin2 ϕ)(∆min + ε)
= 1 −cos ϕ −ε

∆min + ε
= O(1).

According to Theorem 3.3, the upper bound is

˜O

d
(∆(x(1), x(d+1)) + ε)2 ln 1

δ + HDE(∆(x(1), x(d+1)) ln 1

δ +
NLmax
∆(x(1), x(d+1)) + ε ln 1

δ



= ˜O

d
(∆min + ε)2 ln 1

δ + HDE(∆(x(1), x(d+1)) ln 1

δ + NLmax

∆min + ε ln 1

δ



where

HDE(∆(x(1), x(d+1)) =






16NLmax,
ε ≥1 −cos ϕ

16NLmax(∆min + ε + 2(1 −cos ϕ)/N)2

(∆min + ε)2
,
ε < 1 −cos ϕ

As ∆min = cos ϕ −cos2 ϕ = cos ϕ(1 −cos ϕ), thus 1 −cos ϕ =
∆min

cos ϕ
≤
√

2∆min, and
HDE(∆(x(1), x(d+1)) < 144NLmax for any choice of ε.

56


---Page Break---
Hence, the upper bound is

E[τ] = ˜O

d
(∆min + ε)2 ln 1

δ + NLmax ln 1

δ + NLmax

∆min + ε ln 1

δ



= ˜O

d
(∆min + ε)2 ln 1

δ + NLmax

∆min + ε ln 1

δ


.

As ϕ →0, the first term (the VE term) dominates and it matches the lower bound in (N.1). Therefore,
the upper bound is asymptotically tight up to logarithmic terms.

Corollary N.1. Firstly, for the instances defined in Example 2, we have

HDE ≤16(N −2)Lmax

(∆min + ε)2

 

(a + ε)2 + (b −a −ε)2
!

,

and the sample complexity of the PSεBAI+ is tight up to (NLmax/Lmin) and logarithmic factors.
We also further observe some specific instances: (i) when p →0+, with ∆min = minx̸=x∗∆(x∗, x),
we have

E[τ]∗

ln(1/δ) ∈˜O

 

min

(
d

(∆min + ε)2 + NLmax

∆min + ε,
Lmax + d

(∆min + ε)2

)!

\
Ω

(

max

(
d

(∆min + ε)2 , Lmin(b −a −ε)

∆min + ε

))

.

(ii) When p →

a−ε)
(N−2)b
−
and (a + ε)2 + (b −a −ε)2 = Ω(1), we have HDE =
NLmax
(∆(x(1),x(i))+ε)
2

and
E[τ]∗

ln(1/δ) ∈˜O

min

HDE, d + Lmax

ε2

 \
Ω
d + Lmin

ε2


.

The upper bounds are achieved by the PSεBAI+ algorithm.

Proof of Corollary N.1. Under Example 2, we have

∆(x(1), x(i)) = (1 −(N −2)p) · a −(N −2)p · (b −a) > ε

∆j(x(1), x(i)) = −b + a < −ε,
i ≥2, j ̸= 1, i

Thus, (1) x(1) is the unique ε-best arm; (2) {x(i)}i≥2 are equivalent, and ∆min := ∆(x(1), x(i)); (3)
for any i ≥2, x(i) can be an ε-best arm under some alternative distributions.

For any i ≥2, by solving (M.20) with xε = x(1), x = x(i), we obtain λx(1) =
∆(x(1),x(i))+ε

(a+ε)(b−a−ε) and the
alternative distribution
P ′
θ[θ∗
j ] =
pj
1 + λx(1)(∆j(x(1), x(i)) + ε).

The lower bound on the expected number of changepoints is

E[lτ] ≥(a + ε)(b −a −ε)

 
∆(x(1), x(i)) + ε
2 ln 4

δ .

Thus the time complexity is lower bounded by

E[τ] ≥Lmin · (a + ε)(b −a −ε)

 
∆(x(1), x(i)) + ε
2 ln 4

δ .

Furthermore, by solving (M.14), we obtain the lower bound on the expected sample complexity when
the context index jt is revealed:

E[τ] ≥2 log
1
2.4δ min
¯v∈∆K max
i̸=1

1
¯v1 + 1

¯vi
 
∆(x(1), x(i)) + ε
2 ≥2 log
1
2.4δ
(
√

d −1 + 1)2
 
∆(x(1), x(i)) + ε
2

≥
d
 
∆(x(1), x(i)) + ε
2 · 2 log
1
2.4δ .

57


---Page Break---
We get the lower bound on the expected sample complexity:

E[τ] ≥max

(
d

(∆min + ε)2 · 2 log
1
2.4δ , Lmin(a + ε)(b −a −ε)

(∆min + ε)2
ln 4

δ

)

.
(N.2)

According to Theorem 3.3, the upper bound on the expected sample complexity is

˜O

 

min

(
d

(∆min + ε)2 ln 1

δ + HDE ln 1

δ + NLmax

∆min + ε ln 1

δ ,
Lmax + d

(∆min + ε)2 ln 1

δ

)!

(N.3)

≤˜O

 

min

(
d

(∆min + ε)2 ln 1

δ +
NLmax

(a + ε)2 + (b −a −ε)2

(∆min + ε)2
ln 1

δ

+ NLmax

∆min + ε ln 1

δ ,
Lmax + d

(∆min + ε)2 ln 1

δ

)!

where we utilize

HDE =
Lmax
(∆min + ε)2

 s

min

16pj, 1

4


|a + ε| +

s

min

16p, 1

4


|a + ε|

+ (N −2)

s

min

16p, 1

4


|b −a −ε|

!2

≤16(N −2)Lmax

(∆min + ε)2

 

(a + ε)2 + (b −a −ε)2
!

.
(N.4)

By comparing the lower bound in (N.2) and the upper bound (N.3), we conclude that

• the sample complexity of PSεBAI+ is tight up to NLmax
Lmin
and logarithmic factors.
• When the mean gap is small, the sample complexity of PSεBAI+ is dominated by the former term,
i.e., the design of PSεBAI.
• When p →0+, then p1 →1, pj →0 for j ≥2, so the instance tends to be stationary and
∆min →a. The lower bound in (N.2) becomes

E[τ] ≥max

(
d

(∆min + ε)2 · 2 log
1
2.4δ , Lmin(b −a −ε)

∆min + ε
ln 4

δ

)

and the upper bound in (N.3) turns into

˜O

 

min

(
d

(∆min + ε)2 ln 1

δ + NLmax

∆min + ε ln 1

δ ,
Lmax + d

(∆min + ε)2 ln 1

δ

)!

.

The vector estimation term dominates the sample complexity and our upper bound is tight.

• When p →

a−ε)
(N−2)b
−
, then p1 →1 −(N−1)(a−ε

(N−2)b
and pj →

a−ε
(N−2)b

, so the instance tends to
be non-stationary and ∆min →ε. The lower bound in (N.2) becomes

E[τ] ≥max
 d

4ε2 · 2 log
1
2.4δ , Lmin(a + ε)(b −a −ε)

4ε2
ln 4

δ



and the upper bound in (N.3) turns into

˜O

 

min

(
d
4ε2 ln 1

δ + HDE · ln 1

δ + NLmax

2ε
ln 1

δ , Lmax + d

4ε2
ln 1

δ

)!

where HDE is upper bounded by (N.4).
(1) If

(a + ε)2 + (b −a −ε)2
= O(4ε2 + NLmax), DE is independent of ε and VE increases
as ε decreases, thus the expected sample complexity is dominated by vector estimation term when
ε is small;
(2) If

(a + ε)2 + (b −a −ε)2
= Ω(1), the expected sample complexity is dominated by

58


---Page Break---
distribution estimation term.
Under both scenarios, our upper bound is tight.

O
Experimental Details

For the computation of the G-optimal allocation, we adopt the Wolfe–Atwood Algorithm with the
Kumar–Yildirim start introduced in [33], where the input to the function is the arm set and the output
is the G-optimal allocation. All experiments are conducted via MATLAB R2021b on a MacBook Pro
with Apple M1 Pro chip and 16 GB memory.

To shorten the execution time,

• Before PSεBAI stops, PSεBAI and NεBAI are conducted in parallel (i.e., we run PSεBAI+).
After PSεBAI stops, NεBAI continues and is run in a batch manner, i.e., we sample Lmin samples
according to the G-optimal allocation one time and update the statistics. As the sample complexity
of NεBAI is of order at least 107 and each stationary segment is of order 104, the effect of this
batch sampling procedure can be largely ignored.
• DεBAI and DεBAIβ are both conducted in segments, because the latent context vector can be
observed by the two algorithms and the latent vector does not change within a stationary segment.
• As the experiment in Section 6 illustrates that PSεBAI outperforms NεBAI and dominates the
PSεBAI+ algorithm, we run PSεBAI instead of PSεBAI+ for the addition experiments in
Section O.2 and Section O.3.

To increase the robustness of the algorithm, the window size for LCD is doubled. Note that this will
only influence an absolute constant in the sample complexity of the proposed algorithm and the order
of the sample complexity remains.

O.1
Modification of Confidence Radii in PSεBAI and PSεBAI+

During the proof of the upper bound, we have relaxed the absolute constants in the confidence radii to
simplify the proof and increase the readability. In the experiments, we utilize the tighter confidence
radii to gain better empirical performance. Note that when these tighter confidence radii are utilized
by our algorithm, it still enjoys the current theoretical guarantee. The choice of confidence radii are
as follows:

• αt: according to Lemma I.1 and Lemma E.1, α can be tightened to be

αalg
t
= d

Tt
ln
2
δv,Tt
+

s d

Tt
ln
2
δv,Tt

2
+ 4d

Tt
ln
2
δv,Tt
≤5

s

d
Tt
ln
2
δv,Tt
.

• βt,j: according to (I.3) and Lemma K.1, βt,j can be tightened to be

βalg
t,j = min










1
3Lmax ln
2
δd,Tt +

r
1
3Lmax ln
2
δd,Tt

2
+ 2ϕt,jLmax

Tt
ln
2
δd,Tt
Tt
, 1









,

where ϕt,j := min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4


.

• ξt: instead of using 25
√

2 NLmax

Tt
ln
2
δm,Tt , we turn to bound the residual error by (I.9):

ξalg
t
=
X

j:ψt,j,1>ˆpt,j
4 · βalg
t,j +
X

j:ψt,j,1=ˆpt,j,
ψt,j,2>ˆpt,j

min
n
4, 2˜ξalg
t,j
o
· βalg
t,j

+
X

j:ψt,j,3=ˆpt,j
10

s

d
Tt,j
ln
2
δm,Tt
· βalg
t,j

59


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

106

108

1010

Figure 3: Misspeficied Lmax and Lmin.

where

ψt,j,1 := max

ˆpt,j, d

4Tt
ln
2
δm,Tt


, ψt,j,2 := max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


,

˜ξalg
t,j =
d
Tt,j
ln
2
δm,Tt
+

s d

Tt,j
ln
2
δm,Tt

2
+ 4d

Tt,j
ln
2
δm,Tt
≤5

s

d
Tt,j
ln
2
δm,Tt
.

˜ξalg
t,j is obtained in a similar manner as αalg
t . ξalg
t
characterizes the confidence radii of each context
at a finer level.

These finer confidence bounds can save a constant of 2.5 when t is large.

O.2
Misspecified Lmax and Lmin

As PSεBAI+ requires the knowledge of Lmax, we empirically test the robustness of the algorithm
towards Lmax on the instances in section 6. We run PSεBAI with ˜Lmin = νLmin and ˜Lmax = νLmax
where ν = 0.8 or 1.2. An ε-best arm is recommended in all experiments. The sample complexities
are presented in Figure 3. The result indicates that PSεBAI (thus PSεBAI+) is robust towards the
knowledge of Lmax and its superiority over NεBAI is maintained.

O.3
Robustness towards w and b

According to the distinguishability condition (Assumption 1), point (2) indicates we can set w = Lmin

3γ ,
where γ is the change detection frequency, and (3.5) indicates w and b are coupled. We denote
˜w := Lmin

18 .

To exam the robustness towards w and b, we choose to vary the choice of γ, thus, w and b will change
accordingly. Specifically, we select γ ∈{2, 3, 6, 12} and the corresponding w ∈{3 ˜w, 2 ˜w, ˜w, ˜
w

2 }.
The other parameters remain unchanged. The experiment result is presented in Figure 4. When
γ = 18, w =
˜
w

3 , Assumption 1 is severely violated and the result is not desirable.

The result indicates that, while smaller γ and greater w can result in slightly greater sample complexity,
the overall sample complexity does not vary much and the superiority of our algorithm over the naive
uniform sampling algorithm NεBAI is maintained. We conclude that our algorithm is robust against
these choices, as long as Assumption 1 is not severely violated.

60


---Page Break---
5
6
7
8
9
10
11
12
105

106

107

108

109

1010

Figure 4: Robustness towards w and b.

O.4
Benchmarks: DεBAIand DεBAIβ

We present the Distribution ε-Best Arm Identification (DεBAI) in Algorithm 8 and its variant
DεBAIβ in Algorithm 9. According to Dynamics 3, the agent has access to the context vector θ∗
jt
and the changepoint t ∈C. Thus only the distribution Pθ remains unknown and the agent needs to
estimate it via the observed contexts.

Algorithm 8 DISTRIBUTION ε-BEST ARM IDENTIFICATION (DεBAI)

1: Input: the arm set X, the latent vector matrix Θ, the slackness parameter ε, the confidence
parameter δ.
2: Initialize: Compute the G-optimal allocation λ∗.
3: Compute C3 = P∞
n=1 n−3.
4: Observe θ∗
j1.
5: Compute

¨pt,j =

lt
X

ls=1

1{cls = cj}

lt
,
¨xt = arg max
x∈X
x⊤Θ¨pt
(O.1)

¨βt =

s

1
2lt
ln 2C3Nl3
t
δ
,
¨ρt(¨xt, x) = ¨βt
min
ζ(¨xt,x)∈R

N
X

j=1
|∆j(¨xt, x) + ζ(¨xt, x)|

6: while minx̸=¨xt(¨xt −x)⊤Θ¨pt −¨ρt < −ε or t /∈C do
7:
Observe θ∗
jt and 1{t ∈C} update t = t + 1.
8:
Update ¨xt and ¨ρt with (O.1) and update lt if t ∈C.
9: end while
10: Recommend arm ¨xε = ¨xt.

The design is straightforward except for ζ(¨xt, x).
It minimizes the summation ζ(¨xt, x) :=
arg miny∈R
PN
j=1 |∆j(¨xt, x) + ζ(¨xt, x)|. This trick has also been utilized in the design of PSεBAI
(see (I.5)). It helps to better exploit the structure of the latent vectors Θ and facilitates the identification
process. For easy implementation, we choose a proxy ζ(¨xt, x) = arg miny∈R
PN
j=1(∆j(¨xt, x) +

ζ(¨xt, x))2 = −1

N
PN
j=1 ∆j(¨xt, x)

61


---Page Break---
Algorithm 9 DISTRIBUTION ε-BEST ARM IDENTIFICATION-β (DεBAIβ)

1: Input: the arm set X, the latent vector matrix Θ, the slackness parameter ε, the confidence
parameter δ.
2: Initialize: Compute the G-optimal allocation λ∗.
3: Compute C3 = P∞
n=1 n−3.
4: Observe θ∗
j1.
5: Compute

◦pt,j =

lt
X

ls=1

Lls1{cls = cj}

t
,
◦xt = arg max
x∈X
x⊤Θ
◦pt,
(O.2)

◦
βt,j := min
 1

3Lmax ln 2

δt +

r
1
3Lmax ln 2

δt

2
+ 2ϕt,jLmax

t
ln 2

δt
t
, 1

,

◦
ϕt,j := min

4 max

◦pt,j, 25
4
Lmax

t
ln 2

δt


, 1

4


, δt =
δ
C3Nt3 ,

◦ρt(¨xt, x) =
min
ζ(
◦xt,x)∈R

N
X

j=1
|∆j(
◦xt, x) + ζ(
◦x, x)|
◦
βt,j.

6: while minx̸=
◦xt(
◦xt −x)⊤Θ
◦pt −
◦ρt < −ε do

7:
Observe θ∗
jt and 1{t ∈C} and let t = t + 1.

8:
Update
◦xt and
◦ρt with (O.2).
9: end while
10: Recommend arm
◦xε =
◦xt.

DεBAIβ utilizes the techniques we have used to bound the deviation between the true dis-
tribution pj and the estimated ones
◦pt,j for all j
∈
[N].
Similarly, we let ζ(
◦x, x)
=

arg minζ(
◦x,x)∈R
PN
j=1

◦
βt,j(∆j(
◦xt, x) + ζ(
◦x, x))2 = −

PN
j=1

◦
βt,j∆j(
◦xt,x)
PN
j=1

◦
βt,j
for efficient computing.

As the theoretical guarantee of DεBAIβ can be derived following a similar manner as the proof of the
DE term of PSεBAI in Appendix I.2 and in the proof of Lemma G.6, for simplicity, we just present
the result and omit the proof here here.

Theorem O.1. DεBAIβ identifies an ε-best arm within

˜O




max
xε∈X,x̸=xε,x∗
Lmax
◦
H

2
(xε, x)
(∆min + ε)2
ln 1

δ



,

time steps with probability at least 1 −δ and in expectation, where

◦
H(xε, x) :=
min
ζ(
◦xε,x)∈R

N
X

j=1

r

min{16pj, 1

4}|∆j(
◦xε, x) + ζ(
◦xε, x)|.

We present a theorem, along with a proof sketch, for the theoretical guarantee of DεBAI below.

Theorem O.2. DεBAI identifies an ε-best arm within

˜O

 

max
xε∈X,x̸=xε,x∗
Lmax ¨H2(xε, x)

(∆min + ε)2
ln 1

δ

!

,

time steps with probability at least 1 −δ and in expectation,
where
¨H(xε, x)
:=
minζ(¨xε,x)∈R
PN
j=1 |∆j(¨xε, x) + ζ(¨xε, x)|.

62


---Page Break---
Proof. For the sake of conciseness and simplicity, only a proof sketch is provided for this Theorem.
Similar results can be found in the referred contents. The proof is composed by 4 steps.

• Step 1: bound the deviation of Pθ[θ∗
j ] by Lemma O.3 and Remark O.4.
• Step 2: prove the recommended arm is an ε-best arm. Conditional on Step 1, we can prove this
following the procedures in Lemma J.1.
• Step 3: give a sufficient condition and then obtain a high probability upper bound. This can be seen
from (J.1) and by solving

¨ρt(¨xt, x) = ¨βt

N
X

j=1
|∆j(¨xt, x) + ζ(¨xt, x)| ≤∆min + ε

2

⇒
lt = ˜O

 
¨H2(xε, x)
(∆min + ε)2 ln 1

δ

!

.

⇒
t = ˜O

 
Lmax ¨H2(xε, x)

(∆min + ε)2
ln 1

δ

!

.

The high probability result is obtained by maximizing the above over xε ∈X, x ̸= xε, x∗.
• Step 4: the expected result can be derived in a similar method as in the proof of NεBAI in
Appendix F. The expected sample complexity and the high-probability sample complexity is of the
same order.

Lemma O.3. With probability at least 1 −δ, supj∈[N] |pj −¨pt,j| ≤¨βt for all t ∈N, where
¨βt =
q

2
lt ln
2
δlt and δlt =
δ
C3l3
t .

Proof. We may define the cumulative distribution function (CDF) and the empirical CDF for Pθ as
Fθ(j) = Pj
k=1 Pθ[θ∗
j ] and ¨Ft(j) = Plt
ls=1 1{cls = cj, ls ≤j} respectively for j ∈[N]. According
to the DKW inequality, we have

P

"

sup
j∈[N]

 ¨Ft(j) −F(j)
 ≥ϵ

#

≤2 exp
 
−2ltϵ2
.

By the implication

ϵ ≤|pj −¨pt,j| = |F(j) −F(j −1) −¨Ft(j) + ¨Ft(j −1)|

≤|F(j) −¨Ft(j)| + |F(j −1) −¨Ft(j −1)|

⇒
|F(j) −¨Ft(j)| ≥ϵ

2
or
|F(j −1) −¨Ft(j −1)| ≥ϵ

2,

we have that

P

"

sup
j∈[N]

pj −¨pt,j
 ≥ϵ

#

≤P

"

sup
j∈[N]

 ¨Ft(j) −F(j)
 ≥ϵ

2

#

≤2 exp

−2lt
 ϵ

2

2

= 2 exp

−ltϵ2

2


.

This is equivalent to

P

"

sup
j∈[N]

pj −¨pt,j
 ≥¨βt

#

≤δlt.

A union bound gives an upper bound of the failure probability P∞
lt=1 δlt = δ.

Remark O.4. The use of the DKW inequality gives a union bound over the deviation of the distribution
estimation all contexts. This avoid the N factor in the logarithm in ¨βt and is beneficial when the
number of contexts N is large.

63


---Page Break---
When there are a few contexts, it is also possible to directly bound the deviation for each context via
Azuma-Hoeffding’s inequality, i.e.,

P
h
|pj −¨pt,j| ≤¨βt
i
≤δlt

where ¨βt :=
q

1
2lt ln 2C3Nl3
t
δ
. A union bound gives that with probability at least 1−δ, |pj −¨pt,j| ≤¨βt
for all t ∈N, j ∈[N]. As this new confidence radius is the same as the one in Lemma O.3 up to
constant and logarithmic terms, the sample complexity should be of the same order.

We adopt this confidence radius ¨βt :=
q

1
2lt ln 2C3Nl3
t
δ
in the experiment.

P
Additional Discussions

In this section, we provide additional discussions on

• the related methods for BAI in the nonstationary bandits literature in subsection P.1,
• the upper on the sample complexity in Theorem 3.2 and Theorem 3.3 in subsection P.2,
• the connection between the piecewise-stationary linear bandits model to the stationary linear bandits
model in subsection P.3,
• the special case where the number of context N = 1 in subsection P.4,

P.1
Discussion on the Related Methods in Nonstationary Bandits

To the best of our knowledge, there is limited literature investigating the BAI in the nonstationary
bandits setup (there is comparatively a much richer literature for regret minimization in non-stationary
environments): [14] studies BAI in the fixed-horizon setup and [15] assumes the best arm remains
unchanged after certain time step and explores the fixed-confidence setting. Both of the works are
not directly comparable to our proposed piecewise-stationary setup.

Additionally, we provide strong baselines algorithms DεBAI and DεBAIβ in Section 6, which
are detailed in Section O.4. These two baselines are given oracle information about the context
(see Dynamics 3), while PSεBAI+ is not. As indicated by Figure 2(b), PSεBAI+ is competitive
compared to these strong baselines and is much better than the naive approach.

P.2
More Discussions on the Upper Bound

Firstly, we emphasize that the sample complexity in Theorem 3.2 and Theorem 3.3 are instance-
dependent, in particular, the term HDE characterizes the difficulty in estimating the distribution of the
context. Furthermore, the presence of the gaps ∆(x, x∗) also underscores that our upper bounds are
functions of the instance.

Secondly, our algorithm adopts the G-optimal design, which is minimax optimal for the BAI in
standard linear bandits problem [1, 2]. Although we have not proved it, we believe that the sample
complexity of the proposed algorithm may not be instance-dependent optimal.

Lastly, we provide some remarks on the instance-dependent optimal algorithms:
(1) The G-optimal design is the cornerstone for the more efficient and adaptive rules like XY-
allocation and XY-Adaptive Allocation in [1]. Therefore, our algorithm provides a framework
for more sophisticated algorithms in the piecewise-stationary linear bandits problem with the BAI
task. Empirically, one can attempt to replace the G-optimal design by the XY-allocation during
implementation, which can possibly yield empirical benefits.
(2) The current lower bound is established with two simpler problems (see Section 4), which is
sufficient to show our algorithm is minimax optimal. However, a tighter instance-dependent lower
bound based on the original problem may be required if one wishes to show an algorithm is instance-
dependent optimal. This can be challenging because the distribution of the arms, the contexts and the
changepoints are unknown, and the characterization of the "alternative instance" (i.e., the ε-best arm
set is changed) is difficult.

To conclude, we believe an instance-dependent optimal algorithm is appealing and can lead to future
research.

64


---Page Break---
P.3
Connections to the Stationary Bandits

Firstly, we would like to clarify that, in the piecewise-stationary linear bandits model, the instance
tends towards a stationary one when maxj∈[N] pj →1, instead of the scenario in which Lmax →∞.

When Lmin and Lmax are large, estimating each latent vector becomes easier since there are more
observations in a single stationary segment. However, the overall reward of an arm also depends
on Pθ, the distribution of latent context vectors, which is also assumed to be unknown in the setup.
In order to estimate Pθ, a learning agent can get one (unobservable) sample from Pθ only at the
(unobservable) change points. In other words, Lmax charaterizes the “sparsity” of the context samples:
the larger Lmax is, the sparser the context samples are. A larger Lmax indicates that samples from
Pθ are likely to be generated less frequently, which will result in the increase of the overall sample
complexity. Consider an extreme example of Dynamics 3 where the context vector is revealed at
every time step and Lmin and Lmax are very large. If at least lτ context samples from Pθ are required
to identify the best arm, then a sample complexity of lτLmin is unavoidable in the our setup under
this extreme case.

Secondly, while maxj∈[N] pj is important in characterizing the (non)stationarity of the instance,
we would like to emphasize that both the distribution Pθ and latent context vectors are essential in
characterizing the sample complexity, as shown in the term ¯H(xϵ, x) in Theorem 3.2. To further
illustrate this, we provide a simple but illuminating instance as follows: The instance is composed
by N = 3 contexts and K = 2 arms, and ∆j denotes the mean gap between arm 1 and arm 2 under
context j ∈[N] (to be specified below) and ∆:= P

j∈[N] pj∆j denotes the weighted mean gap
between arm 1 and arm 2. The probability of context 1 is p1 = maxj∈[N] pj = 1 −p2 −p3, and p1
is close to 1. Let ˆx denote the empirical version of the statistics x in the following arguments.

In such a case, consider this question: if an algorithm has only detected context 1 in the first t time
steps and t is large, should it stop or not?

As no change point has occurred, the current dynamics behaves similarly to that of a stationary bandit
environment up till now. Thus the empirical probabilities of the 3 contexts are ˆp1 = 1, ˆp2 = ˆp3 = 0
and the empirical mean gap under context 1 ˆ∆1 is close to ∆1. Consider the following two cases:

• Case 1: ∆1 = p2 and ∆2 = ∆3 = −p1. The true weighted mean gap between arm 1 and arm
2 is ∆= P

j∈[N] pj∆j = −p1p3 < 0, and thus arm 2 is the best arm. This indicates that, in
order to estimate ∆2 and ∆3, an algorithm needs to observe contexts 2 and 3 despite their small
probabilities. But currently ˆp2 = ˆp3 = 0, ˆ∆= ˆ∆1 ≈∆1 > 0, so the algorithm should not
terminate.
• Case 2: ∆1 = 1 and ∆2, ∆3 ∈[−2, 2]. Thus ˆ∆≈∆≥p1 −2p2 −2p3 > 0. This indicates
there is no need to observe contexts 2 and 3, because arm 1 is the best even in the worst case
(∆2 = ∆3 = −2). In this case, as long as the algorithm gets a good estimate of ∆1, it can
confidently terminate.

As these two cases indicate, in addition to Pθ, the latent vectors (which determine the means of the
arms) are equally important for the stopping time. Our algorithm takes both factors into account, as
reflected by the summation term in equation (3.4) and ˆ∆t(x∗
t , x) in (3.5) for the algorithm design,
and by ¯H(xϵ, x) for the theoretical upper bound.

We conclude that

• The unknown distribution Pθ and the latent vectors jointly determine the sample complexity.
¯H(xϵ, x) in Theorem 3.2 characterizes their roles, where the contexts with larger probabilities have
commensurately greater influence on the sample complexity of the algorithm.
• When maxj∈[N] pj →1 with other parameters fixed, the piecewise-stationary linear bandit instance
reduces to a stationary one. HDE(xϵ, x) in Theorem 3.2 becomes Lmax/4, which implies that a
constant number of change points need to be observed, and thus TD(xϵ, x) can be regarded as a
constant. The dominant term in our upper bound, the TV (x) term, recovers the upper bound in the
stationary bandits, that is, maxx̸=x∗
d
(∆(x∗,x)+ε)2 ln(1/δ) [32].

65


---Page Break---
P.4
Special Case: N = 1

We only consider N > 1 as the bandit model is only truely piecewise-stationary when N > 1.
Thus, we did not specifically derive upper and lower bounds for the extreme case where N = 1.
Nevertheless, our analysis can cover this case due to the following reasons:

• The (minimax) lower bound in Theorem 4.1 is not directly applicable since it is not derived for this
extreme instance. However, if we step back to the analysis equation (M.16), the feasible set in the
optimization problem is empty, and thus the lower bound on E[lt] (NC in Theorem4.1) is 0. In this
case, the lower bound in Theorem4.1 reduces to the lower bound in the stationary bandits.
• Regarding the upper bound in Theorem 3.2, when N = 1, we have p1 = 1 and HDE = Lmax
4
. In
this case, as HDE does not depend on the mean gaps and the mean gaps are among the fundamental
quantities to characterize the difficulty of an instance, the TD term can be regarded as a constant.
Therefore, the dominant term in the upper bound is TV . This recovers the bound in the stationary
setup.
• In addition, as we assume N is an input to our algorithm, when we are aware N = 1, we can
actually adopt the algorithms for the stationary setting, e.g., the G-allocation or XY-allocation
rule [1].

Q
ε-Best Arm Tuple Identification Problem

In the main paper, we consider the identification of an ε-best arm in terms of the “ensemble” quality
µx := Eθ∼Pθ[x⊤θ] = PN
j=1 Pθ[θ∗
j ]x⊤θ∗
j . The curious reader may wonder whether we can identify
the ε-best arm tuple X tuple
ε
:= (xε
1, . . . , xε
N), where xε
j is an ε-best arm under context j, i.e.,
xε
j ∈{x : x⊤θ∗
j ≥maxx∈X x⊤θ∗
j −ε}. Given the tools we developed in this manuscript, we answer
this problem in the affirmative.

Intuitions: Let x∗
j := arg maxx∈X x⊤θ∗
j and ∆∗
j := minx̸=x∗
j ∆j(x∗
j, x). We expect to have an
upper bound taking the form of

˜O

max
j∈[N]
Lmax

pj
·
d
 
∆∗
j + ε
2 Lmin
ln 1

δ



where O(
d
(∆∗
j +ε)
2Lmin ln 1

δ ) is an upper bound on the number of context samples j and context j will

occur once among every
1
pj contexts in expectation.

Analysis of the problem: The change detection and context alignment subroutines are only effective
within τ ∗time steps (Line 2 in Algorithm 1). However, if a context is with small occurrence
probability pj, it may not appear before τ ∗time steps.

Regarding this scenario, we only expect to obtain a high-probability upper bound on the sample
complexity, whereas the expected sample complexity requires more techniques beyond our parallel
execution trick (which is used to design PSεBAI+) and is an interesting direction left for future
research.

Besides, it is more feasible to consider the identification of ε-best arms under contexts with high
occurrence probability, i.e., we aim to identify

X tuple
ε,¯p
:= {xε
j : pj > ¯p}

where ¯p is a threshold on the occurrence probability of contexts. Let Θ¯p = {θ∗
j : pj > ¯p} denote
those context with high occurrence probability.

Goal: We aim to devise an algorithm with the minimum sample complexity (arm pulls) to ascertain
either (1) an ε-best arm under context j, or (2) θ∗
j /∈Θ¯p.

With the above goal, we expect to identify an ε-best arm for all j with θ∗
j ∈Θ¯p; for those j with
θ∗
j /∈Θ¯p, either an ε-best arm is identified or θ∗
j /∈Θ¯p is ascertained.

We propose Algorithm 10 for this problem, which is quite similar to Algorithm 1, except for a few
changes:

66


---Page Break---
Algorithm 10 PIECEWISE-STATIONARY ε-BEST ARM TUPLE IDENTIFICATION

1: Input: arm set X, size of the set of latent vectors N, bounds on the segment lengths Lmin and
Lmax, slackness parameter ε, confidence parameter δ, sampling parameter γ and window size w,
threshold b, probability threshold ¯p.

2: Initialize: Compute the G-optimal allocation λ∗and τ ∗= 38400 ln(80)NLmax

ε2
ln N2KLmax

δε2
and
Flag = [N] and Hold = [ ] and Output = [ ].
3: Set CDsample = [ ], CAid = { }. Set tCD = +∞.
4: Set Tt,j = ∅and initialize Tt, Tt,j, Tt with (3.1) for all t ≤τ ∗, j ∈[N].

5: Sample w

2 arms {xs}

w

2
s=1 ∼λ∗and observe the associated returns {Ys,xs}

w

2
s=1, t = w

2 , tCA = w

2 .

6: CAid = {1 : [(xs, Ys,xs)]

w

2
s=1}, ˆjt = 1.
7: while t ≤τ ∗and Flag ̸= ∅do
8:
t = t + 1
9:
Sample an arm xt ∼λ∗and observe return Yt,xt.
10:
if
mod (t −tCA, γ) ̸= 0 then
11:
Update ˆjt = ˆjt−1, Tt,ˆjt = Tt−1,ˆjt ∪{t}, Tt,j = Tt−1,j for j ̸= ˆjt.
12:
else
13:
CDsample = CDsample + [(xt, Yt,xt)].
14:
Update ˆjt = ˆjt−1, Tt,j = Tt−1,j for all j ∈[N].
15:
if |CDsample| ≥w then
16:
if LCD(X, w, b, CDsample[−w : ]) then
17:
CDsample = [ ].
18:
t = t + w

2 , tCA = t, tCD = +∞.

19:
ˆjt, CAid = LCA(X, w, b, CAid).
20:
if ˆjt = N + 1 then break.
21:
Revert Tt,j = Tt−w(γ+1)

2
,j for all j ∈[N].

22:
end if
23:
end if
24:
end if
25:
Update the estimates with (3.1), (3.2) and (Q.1).
26:
if ∃j ∈Flag condition (Q.2) is met for j and tCD = +∞then
27:
for j ∈Flag do
28:
if minx:x̸=x∗
t,j ˆ∆t,j(x∗
t,j, x) −αt,j ≥−ε then

29:
Record j and ˆxj,ε := arg maxx∈X x⊤ˆθt,j in Hold.
30:
Flag = Flag \ {j} and tCD = |CDsample|.
31:
else if ˆpj + βt,j < ¯p then
32:
Record j and ˆxj,ε = NAN in Hold.
33:
Flag = Flag \ {j} and tCD = |CDsample|.
34:
end if
35:
end for
36:
else if tCD = |CDsample| −w

2 then
37:
for (j, ˆxj,ε) in Hold do
38:
Output[j] = ˆxj,ε
39:
end for
40:
end if
41: end while
42: Recommend Output

• On Line 7, the algorithm stops when an ε-best arm or pj ≤¯p is identified for all contexts.
• On Line 25, it computes the confidence radii for the arms in context jt as well as the confidence
radii for the occurrence probabilities

αt,j = d

n ln
2
δv,Tt
+

s d

n ln
2
δv,Tt

2
+ 4d

n ln
2
δv,Tt
,
(Q.1)

67


---Page Break---
βt,j := min
5

2

s

2ϕt,jLmax

Tt
ln
2
δd,Tt
, 1

,

where ϕt,j := min

4 max

ˆpt,j, 25

4
Lmax

Tt
ln
2
δd,Tt


, 1

4


,

• On Line 26 to 40, the stopping rule is changed:
• Stopping condition (I) on Line 26

min
x:x̸=x∗
t,j
ˆ∆t,j(x∗
t,j, x) −αt,j ≥−ε
or
ˆpj + βt,j < ¯p


and
Tt ≥(2Lmax/9) ln(2/δd,Tt)
(Q.2)

where x∗
t,j := arg maxx∈X x⊤ˆθt,j.
• Lines 27 to 35: identify an ε-best arm or ascertain pj ≤¯p among the remaining contexts, and
record these observations in Hold for easy access in stopping condition (II).
• Lines 36 to 40: the recommended ε-best arm is recorded, where we adopt NAN to flag those
contexts with small occurrence probabilities.

Theorem Q.1. Given an instance Λ, with probability at least 1 −δ, Algorithm 10 can recommend an
ε-best arm for all context j ∈Θ¯p with sample complexity

max

(
˜O

 

max
θ∗
j ∈Θ¯
p max

(

Lmax,
d
 
∆∗
j + ε
2

)

· ln(1/δ)
pj

!

, ˜O

 

max
θ∗
j /∈Θ¯
p/2

min

pj, 1

64
	
Lmax ln(1/δ)
(pj −¯p)2

!

,

˜O

 

max
θ∗
j ∈Θ¯
p/2\Θ¯
p min

(

max

(

Lmax,
d
 
∆∗
j + ε
2

)
ln(1/δ)

pj
, min

pj, 1

64
	
Lmax ln(1/δ)
(pj −¯p)2

)! )

,

which can be simplified as

˜O

max

Lmax, d

ε2


· ln(1/δ)
¯p


.

Proof of Theorem Q.1. We provide a concise proof sketch for this theorem.

By adapting the stopping rule for best arm identification in stationary linear bandits to ε-best arm
identification, we observe that the number of arm pulls needed for ε-best arm identification under
context j is

Tt,j = ˜O

d
 
∆∗
j + ε
2 ln 1

δ


.

The remaining problem is to determine how many context samples/changepoints are needed such that
the above number of arm samples can be achieved.

Recall that ˆpt,j = Tt,j

Tt (3.2) and Lemma I.2, we have

P

|Tt,j −Ttpj| ≥Ttβt,j
Good

≤δd,Tt.

In addition, we have an upper bound on βt,j as in (J.5), so there would be sufficient arm samples for
ε-best arm identification under context j if

Tt ·



pj −2 · 5

2

s

2 min

16pj, 1

4
	
Lmax
Tt
ln
2
δd,Tt



≥˜O

d
 
∆∗
j + ε
2 ln 1

δ



By solving this inequality in terms of Tt, we obtain

Tt = ˜O

 

max

(

Lmax,
d
 
∆∗
j + ε
2

)

· ln(1/δ)
pj

!

.

68


---Page Break---
As we aim to identify the ε-best arms under each context θ∗
j ∈Θ¯p, the upper bound is at least

Tt = ˜O

 

max
θ∗
j ∈Θ¯
p max

(

Lmax,
d
 
∆∗
j + ε
2

)

· ln(1/δ)
pj

!

(Q.3)

even if the set Θ¯p is given.

Similarly, for those contexts with small occurrence probability θ∗
j /∈Θ¯p, by Lemma I.2 and (J.5), we
can identify them when

pj + 2 · 5

2

s

2 min

16pj, 1

4
	
Lmax
Tt
ln
2
δd,Tt
≤¯p

⇒
Tt = ˜O

 
min

pj, 1

64
	
Lmax ln(1/δ)
(pj −¯p)2

!

.

Careful readers may notice that the denominator depends on a “probability gap”, which can be very
small. In practice, if we can actually identify the ε-best arm in those contexts, it is also acceptable.
Therefore, we instead only choose not to identify the ε-best arm in those contexts θ∗
j ∈Θ¯p/2, which
yields an upper bound

Tt = ˜O

 

max
θ∗
j /∈Θ¯
p/2

min

pj, 1

64
	
Lmax ln(1/δ)
(pj −¯p)2

!

.
(Q.4)

In this case, the bound is at most ˜O

Lmax ln(1/δ)

¯p

. For the rest contexts θ∗
j /∈Θ¯p/2 \ Θ¯p, either
identifying the ε-best arm or ascertaining its small occurrence probability suffices, that is,

Tt = ˜O

 

max
θ∗
j ∈Θ¯
p/2\Θ¯
p min

(

max

(

Lmax,
d
 
∆∗
j + ε
2

)
ln(1/δ)

pj
, min

pj, 1

64
	
Lmax ln(1/δ)
(pj −¯p)2

)!

.

(Q.5)

The above bound is upper bounded by ˜O

max

Lmax, d

ε2
	
· ln(1/δ)
¯p

.

By taking the maximum of (Q.3), (Q.4) and (Q.5), we can obtain a high-probability problem-
dependent upper bound on the sample complexity. In addition, we can get a high-probability
problem-independent upper bound

˜O

max

Lmax, d

ε2


· ln(1/δ)
¯p


.

By setting the threshold ¯p carefully, the algorithm is guaranteed to terminate before τ ∗given the good
event Good (G.1).

69


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract and introduction accurately reflect the paper’s contributions and
scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made
in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or NA
answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much
the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]

Justification: Empirical experiments are conducted to show that Assumption 1 is needed for
theoretical proof and the proposed algorithm is robust to mild violations of the assumptions.
The limitations are discussed in Appendix B.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings, model
well-specification, asymptotic approximations only holding locally). The authors should
reflect on how these assumptions might be violated in practice and what the implications
would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only
tested on a few datasets or with a few runs. In general, empirical results often depend on
implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is
low or images are taken in low lighting. Or a speech-to-text system might not be used
reliably to provide closed captions for online lectures because it fails to handle technical
jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and
how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important
role in developing norms that preserve the integrity of the community. Reviewers will be
specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]

70


---Page Break---
Justification: The problem setting is clearly introduced in Section 2 and all theoretical results
are accompanied with proofs or proof outlines.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they
appear in the supplemental material, the authors are encouraged to provide a short proof
sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The procedures of the algorithms are clearly presented. All experiment details
are provided in Section 6 and Appendix O.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well
by the reviewers: Making the paper reproducible is important, regardless of whether the
code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to
make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may be
necessary to either make it possible for others to replicate the model with the same dataset,
or provide access to the model. In general. releasing code and data is often one good
way to accomplish this, but reproducibility can also be provided via detailed instructions
for how to replicate the results, access to a hosted model (e.g., in the case of a large
language model), releasing of a model checkpoint, or other means that are appropriate to
the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to
reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct the
dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors
are welcome to describe the particular way they provide for reproducibility. In the
case of closed-source models, it may be that access to the model is limited in some
way (e.g., to registered users), but it should be possible for other researchers to have
some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

71


---Page Break---
Answer: [Yes]
Justification: The code is released. Please refer to Section 6.
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
• The authors should provide instructions on data access and preparation, including how to
access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized ver-
sions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: Please refer to Section 6 and Appendix O.
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
Justification: Error bars (standard deviations) are included in all experiment results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the main
claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall run
with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call
to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of
the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably
report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality
of errors is not verified.

72


---Page Break---
• For asymmetric distributions, the authors should be careful not to show in tables or figures
symmetric error bars that would yield results that are out of range (e.g. negative error
rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they
were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: Please see Appendix O.

Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or
cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than
the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t
make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The authors conform with the NeurIPS Code of Ethics.

Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration
due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [NA]

Justification: The work is mainly theoretical, thus there is no identifiable societal impact.

Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact
or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g.,
deployment of technologies that could make decisions that unfairly impact specific groups),
privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to
particular applications, let alone deployments. However, if there is a direct path to any
negative applications, the authors should point it out. For example, it is legitimate to point
out that an improvement in the quality of generative models could be used to generate
deepfakes for disinformation. On the other hand, it is not needed to point out that a
generic algorithm for optimizing neural networks could enable people to train models that
generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being
used as intended and functioning correctly, harms that could arise when the technology is

73


---Page Break---
being used as intended but gives incorrect results, and harms following from (intentional
or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks, mecha-
nisms for monitoring misuse, mechanisms to monitor how a system learns from feedback
over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: The work is mainly theoretical.

Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not
require this, but we encourage authors to take this into account and make a best faith
effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]

Justification: The MATLAB code for the computation of G-optimal allocation used existing
code, and has been mentioned and cited in Appendix O.

Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service
of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a
dataset.
• For existing datasets that are re-packaged, both the original license and the license of the
derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the
asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:
• The answer NA means that the paper does not release new assets.

74


---Page Break---
• Researchers should communicate the details of the dataset/code/model as part of their sub-
missions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset
is used.
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
• Including this information in the supplemental material is fine, but if the main contribution
of the paper involves human subjects, then as much detail as possible should be included
in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or
other labor should be paid at least the minimum wage in the country of the data collector.
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

75


---Page Break---
