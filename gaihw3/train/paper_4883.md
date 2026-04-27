Worst-Case Offline Reinforcement Learning
with Arbitrary Data Support

Kohei Miyaguchi∗
IBM Research – Tokyo
Tokyo, Japan
koheimiyaguchi@gmail.com

Abstract

We propose a method of offline reinforcement learning (RL) featuring the per-
formance guarantee without any assumptions on the data support. Under such
conditions, estimating or optimizing the conventional performance metric is gener-
ally infeasible due to the distributional discrepancy between data and target policy
distributions. To address this issue, we employ a worst-case policy value as a new
metric and constructively show that the sample complexity bound of O(ϵ−2) is
attainable without any data-support conditions, where ϵ > 0 is the policy subopti-
mality in the new metric. Moreover, as the new metric generalizes the conventional
one, the algorithm can address standard offline RL tasks without modification. In
this context, our sample complexity bound can be seen as a strict improvement on
the previous bounds under the single-policy concentrability and the single-policy
realizability.

1
Introduction

Offline reinforcement learning (RL) (Levine et al., 2020; Prudencio et al., 2023) is a framework for
learning decision-making policies while constrained to a fixed batch of data, preventing the learner
from acquiring new information about the environment during training.

The primary challenges of offline RL are thus originated from the discrepancy between the state-
action distribution of the batch data µ(s)β(a|s) and the visitation distribution of the trained policy
dπ(s)π(a|s). Most of the previous studies have avoided directly dealing with this discrepancy by
posing the assumption known as concentrability (Munos and Szepesvári, 2008; Antos et al., 2008;
Chen and Jiang, 2019; Xie et al., 2022). Roughly speaking, the condition asserts that the ratio between
these two distributions dππ/µβ is well-defined and uniformly bounded over the entire state-action
space. This, in turn, constrains the trained policy π to strictly stay inside the state space covered by
the data support.

However, concetrability may be impractical in real-world applications for several reasons. First,
one often ends up with a poor coverage of the state-action space when exhaustive data collection
is expensive or practically infeasible as in the domains of autonomous driving (Fang et al., 2022),
healthcare (Yu et al., 2021) and public policy-making (Abe et al., 2010). Moreover, the precise shape
of the partial coverage is unknown if the considerations making it partial are not well-documented or
disclosed. On the other hand, it is generally difficult to accurately predict if a policy will visit a given
state or not based only on the knowledge of the policy and the batch dataset. As a result, the set of
concentrable policies in a hypothesis space may be too small to achieve reasonable performance or

∗The author is affiliated with LY Corporation at the time of publication.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: Assumptions and sample complexity bounds of related work. π∗and ˜π∗denote optimal
policies in the conventional and worst-case offline RL, respectively. πn denotes a sequence of policies
indexed with the sample size n. The realizability of π means that π-associated model-free parameters
(e.g., value functions, visitation weight functions and the policy itself) are realizable. ϵ > 0 is the
policy suboptimality given in Problem 4.1 (or equivalently in Problem 3.1, see Corollary 4.2 for the
equivalence). 0 < δ < 1 denotes the confidence parameter. H denotes the time horizon and roughly
comparable to (1 −γ)−1. Cgap and βgap denote the minimum and the lower-tail exponent of the
action value gaps, respectively. N denotes the cardinality of the function classes. The improvements
made by our result are emphasized. See Appendix A for more details.

Method
Assumptions
Sample complexity bound
Concentrability
Realizability

Zhan et al. (2022)
π∗
πn
ϵ−6(1 −γ)−4ln(N/δ)
Chen and Jiang (2022)
π∗
π∗
ϵ−2H5C−2
gapln(N/δ)
Ozdaglar et al. (2023)
π∗
π∗
ϵ−2(1 −γ)−6C−2
gapln(N/δ)
Uehara et al. (2023)
π∗
π∗
ϵ−2−4/βgap(1 −γ)−6−4/βgapln(N/δ)

Ours (Corollary 6.3)
—
˜π∗
ϵ−2(1 −γ)−4ln(N/δ)

even empty.2 Therefore, for applying offline RL in such domains, we need a method that works well
without concentrability or any coverage-related conditions.

To tackle with this issue, we study offline RL with arbitrary data support. We present two major
results in this paper.

i) We develop worst-case offline RL (Problem 4.1), a new offline RL framework for handling
poor state-action coverage, which can be seen as a natural generalization of conventional
offline RL (Corollary 4.2).
ii) We develop worst-case minimax RL (WMRL, Section 6.3), a model-free algorithm address-
ing worst-case offline RL (Corollary 6.3). The resulting sample complexity bound improves
the previous state of the art in terms of both the weakness of the assumptions and the strength
of the bound (Table 1).

The rest of the paper is organized as follows. In Section 2, we review the previous work in the literature
of offline RL, centered around theoretical studies on the role of concentrability. In Section 3, we
introduce some preliminaries around Markov decision process (MDP), offline RL and concentrability.
Then, in Section 4, with the observation that offline RL is ill-posed without concentrability, we
introduce worst-case offline RL as a natural generalization and discuss some of its properties useful
in our subsequent analysis. In Section 5, we establish the connection between worst-case offline RL
and the Lagrangians derived from the saddle-point formulation of offline RL. In Section 6, exploiting
the connection established earlier, we construct a method for solving worst-case offline RL with
polynomial sample complexity. Finally, we discuss the limitation and the future work in Section 7.

2
Related Work

The notion of concentrability is introduced by Munos (2003); Munos and Szepesvári (2008); Antos
et al. (2008) to analyze the value/policy iteration algorithms, not necessarily in the context of offline
RL. Recently, it has been increasingly gaining traction as one of the key characteristics of the difficulty
of offline RL (Chen and Jiang, 2019) due to the distribution mismatch. In its original definition,
concentrability requires the norm of the density ratio ∥dππ/µβ∥∞to be bounded uniformly for all the
policies π. Liu et al. (2020) showed that this uniform boundedness can be relaxed to the single-policy
boundedness with the principle of pessimism in the face of uncertainty (PFU). Considering the case
where the single-policy concentrability is even slightly violated, Xie et al. (2021) further analyzed the

2There are two typical examples for the empty case: i) when the time horizon of the policy evaluation is
(even ever so slightly) longer than the episode length of the collected data in time-inhomogeneous environments,
and ii) when there are undocumented restrictions on the actions taken by the behavior policy, but the trained
policy is modeled with distributions with full action supports such as Gaussian distributions.

2


---Page Break---
performance degradation caused by the lack of concentrability. Finally, we completely remove the
concentrability assumption by incorporating it into a new performance metric.

The removal of concentrability is useful not only for widening the applicability of offline RL, but
also strengthening the sample complexity bound by streamlining the analysis. Previously, Zhan
et al. (2022) established polynomial sample complexity bounds under the weakest known model-free
assumptions, yet being unable to achieving the statistically reasonable rate O(ϵ−2). Also, Chen and
Jiang (2022); Ozdaglar et al. (2023); Uehara et al. (2023) gave improved rates additionally assuming
that the minimum action value gap is bounded away from zero. On the other hand, under a set of
assumptions as weak as Zhan et al. (2022), our sample complexity bound achieves the rate of O(ϵ−2).
See Appendix A for more detailed discussions.

Algorithmically, the PFU principle is often materialized as the pessimistic or behavioral regular-
ization (Kumar et al., 2020; Fujimoto and Gu, 2021; Yu et al., 2020). Previous analyses are often
sensitive to the hyperparameters controlling the degree of such regularization, such as the truncation
threshold b in Liu et al. (2020), the Bellman consistency threshold ε in Xie et al. (2021); Chen
and Jiang (2022) and the regularization weight in Zhan et al. (2022); Uehara et al. (2023). On the
other hand, our method has no hyperparameter other than the choice of the function approximators.
One may see the root cause of this difference in that the PFU principle is built into our single new
performance metric, whereas the previous studies adopt it as an additional objective, resulting in
bi-objective optimizations.

3
Preliminaries

We denote the set of nonnegative real numbers by R+ = [0, ∞) and the uniform norm of a function
g over its domain by ∥g∥∞:= supz∈dom(g) |g(z)|. We also denote by ∆(X) the set of (generalized)
probability density functions on X relative to a suitable base measure,3 such as the counting measure
and the Lebesgue measure.

Markov decision process (MDP) and RL.
Let M = (S, A, R, T) be an MDP consisting of the
state space S, the action space A, the reward function R : S × A →∆([0, 1]) and the transition
probability T : S × A →∆(S). We assume both S and A are finite sets for simplicity. The goal of
RL in general is to optimizing policy π : S →∆(A) in terms of the policy value,

J(π) = J(π|M) := (1 −γ)Eπ



X

t≥0
γtrt



,

with a discount factor γ ∈(0, 1). Here, the expectation Eπ is taken with respect to the Markov chain
generated with at ∼π(·|st), rt ∼R(st, at) and st+1 ∼T(st, at), t ≥0, starting from a known
initial-state distribution s0 ∼p0(s).

Offline constraint.
In maximizing J(π), the offline constraint prohibits us to access the environ-
ment M except through the offline dataset D := {(si, ai, ri, s′
i)}n
i=1. We assume the dataset is
sampled from a fixed distribution pM
data ∈∆(S × A × [0, 1] × S)n such that

pM
data(D) =

n
Y

i=1
µ(si) β(ai|si) R(ri|si, ai) T(s′
i|si, ai),

where µ ∈∆(S) and β : S →∆(A) are the behavior state distribution and the behavior policy,
respectively. Typically, pM
data(D) represents the distribution of the past observational data. The
problem of offline RL is now formally given as follows.
Problem 3.1 (Offline RL). Given the offline dataset D and a small number ϵ > 0, find a policy π
achieving J∗−J(π) ≤ϵ, where J∗:= maxπ:S→∆(A) J(π).

Value, visitation and weight functions.
Let r(s, a) := Ey∼R(s,a) [y] be the expected reward func-
tion and rπ(s) := P

a r(s, a)π(a|s) be its marginalization with respect to policy π. Let T , T π and

3Examples of the generalized probability density functions include the Dirac’s delta function.

3


---Page Break---
T π
∗be the raw transition operator, the policy transition operator and its adjoint given by T v(s, a) =
P

s′ v(s′)T(s′|s, a), T πv(s) = P

a T v(s, a)π(a|s) and T π
∗d(s) = P

s′,a′ d(s′)π(a′|s′)T(s|s′, a′),
respectively. Then, the state value function vπ : S →R, the action value function qπ : S × A →R
and the state visitation distribution dπ ∈∆(S) are given by

vπ = (I −γT π)−1rπ,
qπ = r + γT vπ,
dπ = (1 −γ)(I −γT π
∗)−1p0,

as well as the state weight function wπ : supp(µ) →R and the action weight function f π :
supp(µβ) →R by

wπ(s) := dπ(s)

µ(s) ,
f π(s, a) := wπ(s) ρπ(s, a),

where supp(g) := {x ∈dom(g) | g(x) ̸= 0} denotes the support of function g and ρπ(s, a) :=
π(a|s)
β(a|s) is the density ratio of π to β. We also define the optimal value functions by v∗(s) :=
maxπ vπ(s) and q∗(s, a) = maxπ qπ(s, a) as well as the set of the optimal policies Π∗:=
{π : S →∆(A) : vπ = v∗}, which by definition all attain J∗. See Table 2 for the summary of
the notation introduced above.

Concentrability.
A policy π is said to be concentrable (or satisfying concentrability) if its state-
action visitation is contained in the data support, supp(dππ) ⊂supp(µβ). We denote the set of all
the concentrable policies by ΠCC := {π : S →∆(A) : supp(dππ) ⊂supp(µβ)}.

4
Worst-Case Offline Reinforcement Learning

In offline RL (Problem 3.1), the information on M is restricted by the data support supp(µβ). In
such situations, one cannot know about the transition probability T(s, a) and the reward probability
R(s, a) for (s, a) ̸∈supp(µβ). Consequently, the accurate estimation of J(π) is infeasible (even
with n = ∞) for unconcentrable policies, and thus previous analyses on Problem 3.1 often require
that there exists at least one concentrable optimal policy, i.e., ΠCC ∩argmaxπ J(π) ̸= ∅.

To remove such dependency on concentrability, we introduce a performance metric alternative
to J(π). Let U := {M′ : pM′
data = pM
data} be the set of the environments indistinguishable from
the true environment M with respect to the resulting data distribution pM
data. Noting that U is the
information-theoretic limit of the uncertainty on M under the offline constraint, we follow the
pessimism-in-the-face-of-uncertainty principle and consider the worst case within U,

˜J(π) :=
inf
M′∈U J(π|M′),
(1)

which we refer to as worst-case policy value. Replacing J(π) with ˜J(π) in Problem 3.1, we arrive at
the following problem.

Problem 4.1 (Worst-case offline RL). Given the offline dataset D and a small number ϵ > 0, find a
policy π achieving ˜J∗−˜J(π) ≤ϵ, where ˜J∗:= maxπ:S→∆(A) ˜J(π).

To facilitate the subsequent analysis on Problem 4.1, we next introduce the notion of truncated
environment, which is similar to, yet different from those previously considered by Liu et al. (2020);
Yin and Wang (2021) as it is based on the true and unknown data support rather than the empirical
one. The truncation is useful for characterizing the worst-case policy value ˜J(π).

Definition 4.1 (Truncated environment). The truncation of M with respect to µ and β is given by
˜
M = ( ˜S, A, ˜T, ˜R), where ˜S = S ∪{⊥} with ⊥being an absorbing state with reward zero and

˜R(r|s, a) := χµ,β(s, a) R(r|s, a) + (1 −χµ,β(s, a)) δ0(r),
(2)
˜T(s′|s, a) := χµ,β(s, a) T(s′|s, a) + (1 −χµ,β(s, a)) δ⊥(s′),
(3)

for s ∈S and a ∈A, where χµ,β(s, a) := I {µ(s) > 0} I {β(a|s) > 0} is the indicator function of
the support of µ(s) β(a|s) and δx ∈∆(X) is the Dirac’s delta function located at x ∈X.

Theorem 4.1. We have ˜J(π) = J(π| ˜
M) for all π : S →∆(A).

4


---Page Break---
Proof (sketch). It suffices to show J(π| ˜
M) ≤˜J(π) ≤J(π| ˜
M), where the first inequalilty follows
from J(π| ˜
M) ≤J(π|M′) for all M′ ∈U and the second inequality follows from ˜
M ∈U. See
Appendix D.1 for the complete proof.

In other words, worst-case offline RL is nothing but offline RL with the truncated environment
˜
M. Thus, in principle, one can exploit conventional offline RL methods to solve Problem 4.1. We
hereafter refer to the truncated counterparts (those defined by replacing M with ˜
M) of vπ, v∗, qπ,
q∗, Π∗, dπ, wπ and f π as ˜vπ, ˜v∗, ˜qπ, ˜q∗, ˜Π∗, ˜dπ, ˜wπ and ˜f π, respectively. Likewise, let ˜r, ˜rπ, ˜T ,
˜T π and ˜T π
∗be the truncated counterparts of r, rπ, T , T π and T π
∗, respectively.

Let us remark several key implications of Theorem 4.1. First, since the unknown parameters ˜R and ˜T
of the truncated environment ˜
M are only nontrivial on the data support, it is intuitively obvious that
˜J(π) can be accurately estimated even without the concetrability, given sufficiently large n. Thus, it
is reasonable to expect that Problem 4.1 does not require any concetrabilities to be well-posed, unlike
Problem 3.1. Second, the constructive existence of ˜
M makes the relationship between J(π) and
˜J(π) clearer, as stated in the following corollary.

Corollary 4.1. We have ˜J(π) = J(π) if π ∈ΠCC and ˜J(π) ≤J(π) otherwise.

Proof. See Appendix D.2.

According to Corollary 4.1, the pessimism introduced by the truncation is mild in the sense that it
conserves the values of concentrable policies. Finally, it also clarifies the relationship between the
suboptimality metrics of the conventional and the worst-case problems.

Corollary 4.2. For all π : S →∆(A), we have

J∗−J(π) ≤˜J∗−˜J(π)
(4)

if ΠCC ∩argmaxπ J(π) ̸= ∅. Moreover, the equality is attained if in addition π ∈ΠCC.

Proof. Trivial from Corollary 4.1.

In other words, solutions of the worst-case problem are also valid as solutions of the conventional
problem under the standard assumption, while the two solution concepts are identical if only concen-
trable policies are concerned. In this sense, worst-case offline RL is a natural generalization of the
conventional offline RL for handling arbitrary data distributions.

Finally, we conclude this section by showing a useful property of the worst-case optimal policies.
Let Πβ := {π : S →∆(A) | supp(µπ) ⊂supp(µβ)} be the set of the on-support policies, i.e., the
policies with the support covered by the behavior policy. The following lemma allows us to limit the
scope of policy optimization to Πβ without sacrificing the optimality in terms of ˜
M. The proof is
relegated to Appendix D.3.

Lemma 4.1. There is at least one worst-case optimal policy that is on-support, i.e., ˜Π∗∩Πβ ̸= ∅.

5
Lagrangians for Worst-Case Offline Reinforcement Learning

In this section, we set up theoretical foundation of worst-case offline RL. Specifically, in Section 5.1,
we show a connection between ˜J(π) and the Lagrangian of RL (Puterman, 2014). However, since
the Lagrangian in its original form is unstable to the function approximation error (Section 5.2), we
further introduce a regularized variant of it (Section 5.3).

5.1
Unregularized Lagrangian

Consider the following functional of v : S →R+ and f : supp(µβ) →R+,

L(v, f) := (1 −γ)Ep0 [v(s)] + Eµ,β

f(s, a) δTDv(s, a)

,
(5)

5


---Page Break---
where Ep0 and Eµ,β are the expectation operators with respect to s ∼p0(s) and (s, a) ∼µ(s)β(a|s),
respectively, and δTD : RS →RS×A is the time-difference error operator given by δTDv(s, a) =
r(s, a) + γT v(s, a) −v(s).

We refer to L(v, f) as the (unregularized) Lagrangian since it has been known as the Lagrangian of
the linear-programming-based formulations of RL (Puterman, 2014; Chen and Wang, 2016; Nachum
et al., 2019; Zhang et al., 2021; Zhan et al., 2022). The following theorem reveals that, perhaps
surprisingly, it is also connected with worst-case offline RL.

Theorem 5.1. For all π ∈Πβ ∩˜Π∗, (˜v∗, ˜f π) is a saddle point of L(v, f) in RS
+ × RS×A
+
.

Proof (sketch). The key of the proof is the following identity of Lagrangian.

Lemma 5.1. For all π ∈Πβ ∩˜Π∗, we have

L(v, f) = ˜J∗−Eµ,β
h
(f −˜f π)(s, a) (I −γ ˜T )(v −˜v∗)(s, a)
i
+ Dπ
V(v) −D∗
F(f),
(6)

where Dπ
V(v) := P

s̸∈supp(µ) ˜dπ(s) v(s) and D∗
F(f) := Eµ,β [f(s, a) {˜v∗(s) −˜q∗(s, a)}] .

Now, observe that the third term Dπ
V(v) and the fourth term D∗
F(f) in Eq. (6) are nonnegative since
all of v, f, ˜dπ and ˜v∗−˜q∗are nonnegative. Therefore, Lemma 5.1 implies that Lagrangian is bounded
from below with L(v, f) ≥˜J∗taking f = ˜f π, while bounded from above with L(v, f) ≤˜J∗taking
v = ˜v∗. Combining these two inequalities, a class of the saddle points of Lagrangian is identified as
desired. The full proof is found in Appendix E.2.

Note that the previous studies on the LP-based formulation of RL typically consider the saddle points
in a different domain, RS × RS×A
+
, where the domain of the primal variable v is unconstrained,
unlike our setting with the constraint v ≥0. This constraint is the key to establish the connection
with the worst-case environment ˜
M.

Theorem 5.1 superficially suggests that finding the saddle points of L(v, f) is a reasonable way of
finding the optimal policies with respect to ˜
M. However, in the next section, we show that it is
unstable and easily breaks down by the function approximation error.

5.2
Instability of Unregularized Lagrangian

When the state space S is large, it is practically infeasible to find a saddle point of L(v, f) naïvely
searching over the whole space RS
+ × RS×A
+
. Therefore, one may introduce compact function classes
V ⊂RS
+ and F ⊂RS×A
+
and limit the scope of the search to these classes. Since we do not know
the saddle points (which motivates us to find one), such function classes likely incur the function
approximation error. Thus, it is likely that the saddle points (˜v∗, ˜f π) may sit near the search space
V × F, but not exactly included in the space, (˜v∗, ˜f π) ̸∈V × F.

In this context, we show even a tiny function approximation error can completely disrupt the
connection established in Theorem 5.1. Consider the function classes Vϵ = {v ∈RS
+ | v ≥˜v∗+ ϵ}
and Fϵ = {f ∈RS×A
+
| ∃π ∈Πβ ∩˜Π∗s.t. f ≤˜f π −ϵ} with a small constant ϵ > 0. Then, even
though the function approximation error is small (ϵ for both V and F in terms of the L∞-norm), the
saddle point is collapsed to zero under the approximations with Vϵ and Fϵ. The proof is relegated to
Appendix E.3.
Corollary 5.1. Suppose Vϵ and Fϵ are nonempty. Then, the saddle points of L(v, f) in Vϵ × RS×A
+
must satisfy f = 0. Moreover, the saddle points of L(v, f) in RS
+ × Fϵ must satisfy v = 0.

5.3
Regularized Lagrangian

Corollary 5.1 shows that the saddle points of Lagrangian cannot be used as a reliable way to find
the optimal policies with the function approximation. As a workaround, we introduce a regularized
Lagrangian,

K(v, f) := (1 −γ)Eµ [v(s)] + Eµ,β

f(s, a) δTDv(s, a)

+ (1 −γ)2

2
∥v∥2
2,¯µ ,
(7)

6


---Page Break---
where ¯µ := µ + γT β
∗µ and ∥v∥p,¯µ := {P

s ¯µ(s)vp(s)}1/p denotes the Lp(¯µ)-norm of the functions
over S. Then, it is shown that the regularized Lagrangian is also connected with worst-case offline
RL through its saddle points. To see this, let us define the regularized counterparts of ˜dπ, ˜wπ and ˜f π

with ˘dπ := (1 −γ)(I −γ ˜T π
∗)−1˘pπ, ˘wπ := ˘dπ/µ, ˘f π := ˘wπρπ, obtained by substituting the initial
state distribution p0 with ˘pπ := µ + (1 −γ)˜vπ ¯µ.

Theorem 5.2. For all π ∈Πβ ∩˜Π∗, (˜v∗, ˘f π) is a saddle point of K(v, f) in RS
+ × RS×A
+
. Moreover,
the primal solution ˜v∗is unique on supp(¯µ).

Proof (sketch). Similarly as the proof of Theorem 5.1, the key is the following identity.

Lemma 5.2. There exist U ∗∈R such that, for all π ∈Πβ ∩˜Π∗,

K(v, f) = U ∗−Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜v∗)(s, a)
i
+ ˘Dπ
V(v) −D∗
F(f),
(8)

where ˘Dπ
V(v) := P
s̸∈supp(µ) ˘dπ(s) v(s) + (1−γ)2

2
∥v −˜vπ∥2
2,¯µ.

The first claim of Theorem 5.2 then follows from the fact ˘Dπ
V(v) is nonnegative, and the second
claim follows from the strong convexity of K(v, f) with respect to v on supp(¯µ). The full proof is
in Appendix E.5.

Comparing Theorems 5.1 and 5.2, it turns out that the regularization does not alter the primal part
of the saddle points ˜v∗. Moreover, since K(v, f) is strongly convex in terms of v, the regularized
solution is more stable against the function approximation error as opposed to the unregularized
solution. To see this, denote the regularized primal solution under the function approximation by

˜v∗
♯∈argmin
v∈V
max
f∈F K(v, f),
(9)

where V ⊂RS
+ and F ⊂RS×A
+
are compact function classes. Also denote the individual function
approximation errors of V and F by

ϵV := min
v∈V ∥v −˜v∗∥1,¯µ ,
ϵF :=
min
f∈F,π∈Πβ

n
¯BV∥f −˘f π∥1,µβ + 2(1 −γ) ∥˜vπ −˜v∗∥1,¯µ
o
, (10)

where ¯BV := max {1 + γBV, BV} and BV := maxv∈V ∥v∥∞are scale factors of V. Then, the
following lemma shows the stability of the approximate solution ˜v∗
♯in terms of the aggregated
function approximation error

εapp,V(V, F) :=

p

2 (2 + BF) ϵV + 4√ϵF

1 −γ
= O
√BF ϵV + ϵF

1 −γ


,

where BF := maxf∈F ∥f∥∞is the scale factor of F. The proof is relegated to Appendix E.6.
Lemma 5.3 (Stability of the regularized primal solution). We have ∥˜v∗
♯−˜v∗∥2,¯µ ≤εapp,V(V, F).

Note that the approximation error of F is trivially bounded by a simpler error term

ϵF ≤¯BV
min
f∈F,π∈Πβ∩˜Π∗∥f −˘f π∥1,µβ,

i.e., the L1-error with respect to the function ˘f π of the optimal on-support policy π ∈Πβ ∩˜Π∗.
Our definition of the error is weaker than that, measuring the error with respect to that of the
possibly suboptimal on-support policy π ∈Πβ in exchange for the additional suboptimality cost
2(1 −γ) ∥˜vπ −˜v∗∥1,¯µ. This is beneficial if the optimal policies are difficult to approximate, like
deterministic policies in a continuous action space, yet some near-optimal policies such as the
soft-optimal policies π(a|s) ∝exp{−η˜q∗(s, a)} are easy to approximate.

We also note that the idea of stabilizing the saddle-point-based policy optimization via a strongly
convex regularizer is not new (Nachum et al., 2019; Lee et al., 2021; Zhan et al., 2022; Uehara et al.,
2023). The major difference here (other than the truncation) is that we regularize the value function
v (like Uehara et al. (2023)) but extract the information of the optimal policy from f (like Nachum
et al. (2019); Lee et al. (2021); Zhan et al. (2022)), which, combined with our worst-case framework,
results in a striking improvement in the sample complexity.

7


---Page Break---
6
Worst-Case Minimax Reinforcement Learning

Now, we present a method to solve worst-case offline RL with the saddle points of K(v, f). We
first introduce a method of extracting policy from the dual variable f (Section 6.1), then show the
suboptimality bound of the extracted policy (Section 6.2), which is our main result, and finally show
the sample complexity bound taking into account the finite sample approximation (Section 6.3).

6.1
Policy Extraction

Motivated by Theorem 5.2, we propose a method of extracting the worst-case optimal policy π∗from
the saddle point of K(v, f). Specifically, we consider minimizing the loss function given by
DΞ(f; w, π) := max
ξ∈Ξ {Eµ,β [f(s, a) ξ(s, a)] −Eµ,π [w(s) ξ(s, a)]} ,
(11)

where w : S →R is an auxiliary weight function, w(s) := max {1 −γ, w(s)} is its lower clipping
and Ξ ⊂RS×A is a class of discriminator functions. Note that DΞ(f; w, π) is the integral probability
metric (IPM) (Sriperumbudur et al., 2009) between f(s, a) µ(s) β(a|s) and w(s) µ(s) π(a|s) with
respect to the discriminators Ξ. With a sufficiently rich Ξ, this implies DΞ(f; w, π) attains its
minimum value (i.e., zero) only if π = πf,4 thereby informally justifies the minimization of Eq. (11)
as a way of policy extraction.

This approach introduces additional (functional) variables to be optimized, w : S →R and π : S →
∆(A). To simplify the notation, consider a parameter space Θ and suppose f, w and π share the
same parameter space Θ, i.e., there exists a mapping Θ ∋θ 7→(fθ, wθ, πθ), and redefine the dual
space with F = F(Θ) := {fθ | θ ∈Θ}.5 We define the associated function approximation error with

ϵΘ := min
π∈Πβ

n
¯BV,ΞϵΘ(π) + 2(1 −γ) ∥˜v∗−˜vπ∥1,¯µ
o
,
(12)

where ¯BV,Ξ := max{ ¯BV, BΞ, ∥˜ξ∗∥∞}, BΞ := maxξ∈Ξ ∥ξ∥∞and

ϵΘ(π) := min
θ∈Θ

fθ −˘f π
1,µβ + ∥wθ −˘wπ∥1,µ + BW ∥πθ −π∥TV,µ


(13)

denotes the π-specific function approximation error of Θ. Here, BW := maxθ∈Θ ∥wθ∥∞is the
boundedness of wθ(s) and ∥· ∥TV,µ is the mean total variation (TV) distance with respect to µ, given
by ∥π −π′∥TV,µ := Eµ
P

a |π(a|s) −π′(a|s)|.

Finally, we conclude this section by introducing key quantities of the policy extraction for the
subsequent analysis. Let BΠ := maxθ∈Θ ∥πθ/π0∥∞denote the size of the policy class with respect
to some fixed base policy π0. Also let ϵΞ := minξ∈Ξ ∥ξ−˜ξ∗∥1,µ(β+π0) be the function approximation
error of Ξ, where ˜ξ∗(s, a) := ˜q∗(s, a) −˜v∗(s) is the optimal advantage function.

6.2
The Suboptimality Bound

Unifying the saddle-point problem and the policy extraction problem, we arrive at the aggregated
loss function
L(θ) := LSP(θ) + LX(θ),
(14)
where LSP(θ) := −minv∈V K(v, fθ) is the loss of fθ as a dual solution (cf. Eq. (7)) and LX(θ) :=
DΞ(fθ; wθ, πθ) is the loss of the policy extraction from fθ to πθ (cf. Eq. (11)). Let us denote the
corresponding estimation error by
ϵest(θ) := L(θ) −min
θ∈Θ L(θ).
(15)

We also define the aggregated function approximation error with
εapp,Π(V, Θ, Ξ) := (2 + 3BF) εapp,V(V, F(Θ)) + 3ϵΘ + {BF + (1 −γ)BΠ} ϵΞ.
(16)
The following theorem establishes an upper bound on the policy suboptimality in terms of ϵest(θ) and
εapp,Π(V, Θ, Ξ).

4The lower clipping w(s) ≥1 −γ > 0 plays a crucial role here to exclude the trivial minima f(s, ·) =
w(s) = 0 for every s ∈S.
5There is no loss of generality due to the coupling introduced by the parameter sharing, since one can take it
as a product space Θ = ΘF × ΘW × ΘΠ.

8


---Page Break---
Theorem 6.1. For all θ ∈Θ, we have

˜J∗−˜J(πθ) ≤∥˜wπθ∥∞

1 −γ
{ϵest(θ) + εapp,Π(V, Θ, Ξ)} .
(17)

Proof (sketch). At the heart of the proof is the following inequalities:

Γ(πθ) ≤D∗
F(fθ) + LX(θ) + B′ϵΞ

1 −γ
≤ϵest(θ) + εapp,Π(V, Θ, Ξ)

1 −γ
,

where Γ(πθ) := Eµ,πθ [˜v∗(s) −˜q∗(s, a)] denotes the average action value gap with policy πθ and
B′ := BF + (1 −γ)BΠ. Here, the lower clipping of w (Eq. (11)) and the stability of the primal
solution (Lemma 5.3) are essential for deriving the first and the second inequality, respectively.
Then, the proof is completed by bounding ˜J∗−˜J(πθ) in terms of Γ(πθ) invoking the performance
difference lemma in the worst-case environment. See Appendix F.1 for the full proof.

Theorem 6.1 suggests that one can minimize the policy suboptimality up to the function approximation
error on two conditions, i.e., the weight factor ∥˜wπθ∥∞is appropriately bounded and the loss function
L(θ) is minimized. In the following, we first discuss how to satisfy the first condition.

A trivial way of bounding ∥˜wπθ∥∞is uniformly bounding it with respect to all θ ∈Θ. Define ˜C∞:=
maxθ∈Θ ∥˜wπθ∥∞, which we refer to as the uniform truncated concentrability (UTC) coefficient.
Since ∥˜wπθ∥∞≤˜C∞, we get the following simple suboptimality bound.
Corollary 6.1. For all θ ∈Θ, we have

˜J∗−˜J(πθ) ≤
˜C∞
1 −γ {ϵest(θ) + εapp,Π(V, Θ, Ξ)}
(18)

Note that ˜C∞is always finite because of the compactness of the whole policy space ∆(A)S and
the continuity and well-definedness of π 7→∥˜wπ∥∞. Thus, Eq. (18) is non-vacuous for arbitrary
data distributions as opposed to the conventional concentrability-based results. Moreover, if the
conventional bounds are non-vacuous, ˜C∞recovers the conventional concentrability coefficient as
˜wπ = wπ.

Eq. (18) can be further refined using the localized variants of the uniform coefficient ˜C∞. The results
are presented as Corollaries F.2 and F.3 in Appendix F.3 due to space limitation.

6.3
Sample Complexity Analysis

Now that given Corollary 6.1 (and Corollaries F.2 and F.3 as well) bounds the weight factor ∥˜wπθ∥∞
with some milder variants of the concentrability coefficient, the remaining task, minimizing L(θ), is
handled within the framework of the statistical learning, leading to a sample complexity bound. Let

ˆL(θ) := max
v∈V max
ξ∈Ξ
1
n

X

z∈D
ˆLz(θ; v, ξ),
(19)

be the empirical loss function where

ˆLz(θ; v, ξ) := −

(1 −γ) v(s) + fθ(s, a) {r + γv(s′) −v(s)} + (1 −γ)2 (v2(s) + γv2(s′))

2



+ fθ(s, a)ξ(s, a) −wθ(s)Ea′∼πθ(s) [ξ(s, a′)]
(20)
is the one-sample loss function, z ≡(s, a, r, s′) ∈S × A × [0, 1] × S denotes a transition record.
Note that Eq. (20) is an unbiased estimator of the objective function L(θ).6 Therefore, it is expected
that the oracle loss L(θ) can be approximated with the empirical loss ˆL(θ) and hence the oracle
estimation error ϵest(θ) can be approximated with the empirical estimation error

ˆϵest(θ) := ˆL(θ) −min
θ∈Θ
ˆL(θ).
(21)

Formalizing such an intuition, the following corollary shows the empirical counterpart of Corollary 6.1.
See Appendix F.5 for the proof.

6Note that the expectation with respect to πθ can be computed exactly since the action space is finite. For the
continuous case, one can replace the expectation with the Monte-Carlo approximation without corrupting the
unbiasedness.

9


---Page Break---
Corollary 6.2. Let H ≡H(V, Θ, Ξ) := {z 7→ˆLz(θ; v, ξ) | θ ∈Θ, v ∈V, ξ ∈Ξ} be the class of the
one-sample loss functions and Rn(H) be its Rademacher complexity (Definition B.1). Then, for all
θ ∈Θ and δ ∈(0, 1), with probability 1 −δ, we have

˜J∗−˜J(πθ) ≤
˜C∞
1 −γ

(

ˆϵest(θ) + εapp,Π(V, Θ, Ξ) + 4Rn(H) + Ball

r

2 ln(2/δ)

n

)

,

where Ball := (1 −γ)BV + BF ¯BV + (1 −γ)2B2
V + (BF + BW)BΞ is the aggregated scale factor.

The corollary above implies that minimizing Eq. (19), which is possible with the minimax optimizers
such as the one developed by Thekumparampil et al. (2019), gives a near-optimal policy in terms
of the worst-case environment ˜
M, up to the error proportional to the sum of the optimization error
ˆϵest(θ), the approximation error εapp,Π(V, Θ, Ξ) and the statistical error O(Rn(H) + n−1/2). We
refer to this method as worst-case minimax reinforcement learning (WMRL).

The next corollary gives the sample complexity of WMRL in the simplified case where the function
approximators V, Θ and Ξ are all finite sets.
Corollary 6.3. Suppose V, F and Ξ are all finite sets and ˆϵest(θ) = εapp,Π(V, Θ, Ξ) = 0. Take any
ϵ > 0 and 0 < δ < 1. Then, we have ˜J∗−˜J(πθ) ≤ϵ with probability 1 −δ if

n = Ω

 
B2
all ˜C2
∞
ϵ2(1 −γ)2 ln N

δ

!

,
(22)

where N := |V| |Θ| |Ξ| denote the product of the cardinalities of the function approximators.

Proof. It follows from Corollary 6.2 with Massart’s lemma (Lemma B.2).

A few remarks follow. First, we can replace the UTC coefficient ˜C∞with the localized vari-
ants (Definitions F.1 and F.2) to obtain tighter bounds, by making the same argument starting from
Corollaries F.2 and F.3 instead of Corollary 6.1, respectively. Second, there are implicit dependencies
BV ≥∥˜v∗∥∞and BΞ ≥∥˜ξ∗∥∞due to the realizability ˜v∗∈V and ˜ξ∗∈Ξ that follows from
εapp,Π(V, Θ, Ξ) = 0. Hence, Eq. (22) has an implicit γ-dependency through the scale factor Ball,
which brings an extra Θ((1 −γ)−2) factor in the worst case. Table 1 adopts this form for the fairness
of comparison.

7
Conclusion

To develop an offline RL method for challenging data distributions, we have introduced and studied a
generalization of the conventional framework called worst-case offline RL. As a result, we have shown
it is possible to learn a worst-case optimal policy without any data-support conditions. Moreover,
the presented sample complexity bound strictly improves the previous state of the art under the
single-policy realizability and the single-policy concentrability, suggesting the utility of the proposed
method even with non-challenging data distributions.

We anticipate the presented results are readily extendable to continuous state-action spaces, except
that the truncated concentrability coefficients are not unconditionally finite anymore. The results in
Appendix F.3 are particularly useful in this context, yet the complete picture on the conditions of
their boundedness largely remains to be studied in future work.

Acknowledgments and Disclosure of Funding

The author is grateful to LY Corporation for providing travel funding.

References

Abe, N., Melville, P., Pendus, C., Reddy, C. K., Jensen, D. L., Thomas, V. P., Bennett, J. J., Anderson,
G. F., Cooley, B. R., Kowalczyk, M., Domick, M., and Gardinier, T. (2010). Optimizing debt

10


---Page Break---
collections using constrained reinforcement learning. In Proceedings of the 16th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining, KDD ’10, page 75–84, New
York, NY, USA. Association for Computing Machinery.

Antos, A., Szepesvári, C., and Munos, R. (2008). Learning near-optimal policies with bellman-
residual minimization based fitted policy iteration and a single sample path. Machine Learning,
71:89–129.

Chen, J. and Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning.
In International Conference on Machine Learning, pages 1042–1051. PMLR.

Chen, J. and Jiang, N. (2022). Offline reinforcement learning under value and density-ratio realizabil-
ity: the power of gaps. In Uncertainty in Artificial Intelligence, pages 378–388. PMLR.

Chen, Y. and Wang, M. (2016). Stochastic primal-dual methods and sample complexity of reinforce-
ment learning. arXiv preprint arXiv:1612.02516.

Fang, X., Zhang, Q., Gao, Y., and Zhao, D. (2022). Offline reinforcement learning for autonomous
driving with real world driving data. In 2022 IEEE 25th International Conference on Intelligent
Transportation Systems (ITSC), pages 3417–3422.

Fujimoto, S. and Gu, S. S. (2021). A minimalist approach to offline reinforcement learning. Advances
in neural information processing systems, 34:20132–20145.

Jiang, N. and Huang, J. (2020). Minimax value interval for off-policy evaluation and policy opti-
mization. In Proceedings of the 34th International Conference on Neural Information Processing
Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020).
Conservative q-learning for offline
reinforcement learning. Advances in Neural Information Processing Systems, 33:1179–1191.

Lee, J., Jeon, W., Lee, B., Pineau, J., and Kim, K.-E. (2021). Optidice: Offline policy optimization
via stationary distribution correction estimation. In International Conference on Machine Learning,
pages 6120–6130. PMLR.

Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review,
and perspectives on open problems. arXiv preprint arXiv:2005.01643.

Liu, Y., Swaminathan, A., Agarwal, A., and Brunskill, E. (2020). Provably good batch off-policy
reinforcement learning without great exploration. Advances in neural information processing
systems, 33:1264–1274.

Munos, R. (2003). Error bounds for approximate policy iteration. In Proceedings of the Twentieth
International Conference on International Conference on Machine Learning, ICML’03, page
560–567. AAAI Press.

Munos, R. and Szepesvári, C. (2008). Finite-time bounds for fitted value iteration. Journal of
Machine Learning Research, 9(5).

Nachum, O., Dai, B., Kostrikov, I., Chow, Y., Li, L., and Schuurmans, D. (2019). Algaedice: Policy
gradient from arbitrary experience. arXiv preprint arXiv:1912.02074.

Ozdaglar, A. E., Pattathil, S., Zhang, J., and Zhang, K. (2023). Revisiting the linear-programming
framework for offline RL with general function approximation. In International Conference on
Machine Learning, pages 26769–26791. PMLR.

Prudencio, R. F., Maximo, M. R., and Colombini, E. L. (2023). A survey on offline reinforcement
learning: Taxonomy, review, and open problems. IEEE Transactions on Neural Networks and
Learning Systems.

Puterman, M. L. (2014). Markov decision processes: discrete stochastic dynamic programming. John
Wiley & Sons.

11


---Page Break---
Rashidinejad, P., Zhu, H., Yang, K., Russell, S., and Jiao, J. (2023). Optimal conservative offline
RL with general function approximation via augmented lagrangian. In The Eleventh International
Conference on Learning Representations.

Shalev-Shwartz, S. and Ben-David, S. (2014). Understanding machine learning: From theory to
algorithms. Cambridge university press.

Sriperumbudur, B. K., Fukumizu, K., Gretton, A., Schölkopf, B., and Lanckriet, G. R. (2009).
On integral probability metrics,\phi-divergences and binary classification.
arXiv preprint
arXiv:0901.2698.

Thekumparampil, K. K., Jain, P., Netrapalli, P., and Oh, S. (2019). Efficient algorithms for smooth
minimax optimization. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E.,
and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 32. Curran
Associates, Inc.

Uehara, M., Kallus, N., Lee, J. D., and Sun, W. (2023). Offline minimax soft-q-learning under
realizability and partial coverage. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M.,
and Levine, S., editors, Advances in Neural Information Processing Systems, volume 36, pages
12797–12809. Curran Associates, Inc.

Uehara, M. and Sun, W. (2022). Pessimistic model-based offline reinforcement learning under partial
coverage. In International Conference on Learning Representations.

Xie, T., Cheng, C.-A., Jiang, N., Mineiro, P., and Agarwal, A. (2021). Bellman-consistent pessimism
for offline reinforcement learning. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W.,
editors, Advances in Neural Information Processing Systems.

Xie, T., Foster, D. J., Bai, Y., Jiang, N., and Kakade, S. M. (2022). The role of coverage in online
reinforcement learning. arXiv preprint arXiv:2210.04157.

Yin, M. and Wang, Y.-X. (2021). Towards instance-optimal offline reinforcement learning with
pessimism.
In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W.,
editors, Advances in Neural Information Processing Systems, volume 34, pages 4065–4078.
Curran Associates, Inc.

Yu, C., Liu, J., Nemati, S., and Yin, G. (2021). Reinforcement learning in healthcare: A survey. ACM
Comput. Surv., 55(1).

Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., Finn, C., and Ma, T. (2020). MOPO:
Model-based offline policy optimization. Advances in Neural Information Processing Systems,
33:14129–14142.

Zanette, A. (2023). When is realizability sufficient for off-policy reinforcement learning?
In
International Conference on Machine Learning, pages 40637–40668. PMLR.

Zhan, W., Huang, B., Huang, A., Jiang, N., and Lee, J. (2022). Offline reinforcement learning
with realizability and single-policy concentrability. In Conference on Learning Theory, pages
2730–2775. PMLR.

Zhang, J., Hong, M., Wang, M., and Zhang, S. (2021). Generalization bounds for stochastic saddle
point problems. In International Conference on Artificial Intelligence and Statistics, pages 568–576.
PMLR.

A
Additional Discussion on Table 1

In Table 1, we compare our result with the previous results on the sample complexity of the conven-
tional offline RL under the weakest known assumptions. The weakest known means that it relies only
on the assumptions of the single-policy realizability and the single-policy concentrability. Specifi-
cally, they do not assume the Bellman completeness (Munos and Szepesvári, 2008; Xie et al., 2021;
Chen and Jiang, 2022) or its variants (Zanette, 2023; Rashidinejad et al., 2023), the model-based
realizability (Uehara and Sun, 2022) or the all-policy realizability (Jiang and Huang, 2020), which

12


---Page Break---
Meaning
Symbol
Type
Definition

Expectation w.r.t. action
Pπ
RS×A →RS
Pπq(s) = P

a q(s, a) π(a|s)
Extension of action
Pπ
∗
RS →RS×A
Pπ
∗d(s, a) = d(s) π(a|s)

Backward transition
T
RS →RS×A
T v(s, a) = P

s′ v(s′) T(s′|s, a)
— with policy
T π
RS →RS
PπT
Forward transition
T∗
RS×A →RS
T∗c(s′) = P

s,a T(s′|s, a) c(s, a)
— with policy
T π
∗
RS →RS
T∗Pπ
∗
Marginal reward func.
rπ
S →R
Pπr
State value func.
vπ
S →R
(I −γT π)−1rπ
Optimal —
v∗
S →R
v∗(s) = maxπ vπ(s)
Action value func.
qπ
S × A →R
r + γT vπ
Optimal —
q∗
S × A →R
q∗(s, a) = maxπ qπ(s, a)
Optimal policies
Π∗
2∆(A)S
{π : S →∆(A) : vπ = v∗}

State visitation dist.
dπ
S →R
(1 −γ)(I −γT π
∗)−1p0
State weight func.
wπ
supp(µ) →R
dπ/µ
Policy ratio
ρπ
supp(µβ) →R
π/β
Action weight func.
f π
supp(µβ) →R
wπ ρπ

Policy value
J(π)
R
(1 −γ)Ep0 [vπ(s)]
Optimal —
J∗
R
maxπ J(π)
Table 2: Basic Notation

are strictly stronger than the single-policy realizability and deemed to be rather stringent (Chen and
Jiang, 2019). Below, we discuss each of the methods listed in the table.

Leveraging the Lagrangian-based formulation of RL, Zhan et al. (2022) showed the polynomial
sample complexity only with the single-policy realizability and the single-policy concentrability for
the first time. However, the order of the sample complexity bound O(ϵ−6) is significantly looser than
the standard statistical rate O(ϵ−2). Moreover, their realizability assumption requires the function
approximators to include the value/weight functions associated with a policy πn depending on the
sample size n. In particular, the resulting realizability condition is difficult to interpret since there is
no explicit characterization of πn.

Chen and Jiang (2022) took a different approach, the pessimistic value learning, achieving the
statistically reasonable rate O(ϵ−2). One of the main drawbacks of their result is, however, that
the sample complexity bound blows up if the action value gap Cgap is near zero. Here, the action
value gap is defined as the minimum gap in the values of the best action and the second best action,
Cgap = mins∈S maxa∈A{q∗(s, a) −maxa′̸=a q∗(s, a′)}, which becomes (near) zero if there exist
two actions that are (near) optimal for even one state. In addition, it only competes with the optimal
policy π∗and requires the data to cover the corresponding visitation distribution dπ∗. Finally, the
time-horizon dependency O(H5) is a bit worse than the other results in the table.

Ozdaglar et al. (2023) showed two distinct results: one requiring a completeness-type assumption
and the other requiring realizability and action-value-gap assumptions, in addition to concentrability.
We included the latter to the table. Roughly speaking, their result is similar to that of Chen and Jiang
(2022) except with the difference in the infinite/finite time horizons. Consequently, it also requires
the action gap to be bounded away from zero. We note that their algorithm relies on a constrained LP,
where the number of the constraints is equal to the size of S, making it possibly difficult to scale to
practical problems.

Uehara et al. (2023) also proposed two distinct methods: one establishes slower O(ϵ−8) rate with the
entropy regularization method, and the other establishes a sample complexity depending on so-called
the soft action value gap. We only shows the latter in Table 1. For the latter result, the new condition
on the soft action value gap relaxes those imposed on the ordinary action value gap by Chen and Jiang
(2022); Ozdaglar et al. (2023). The order of the resulting sample complexity bound is O(ϵ−2+4/βgap),
where βgap > 0 corresponds to the lower-tail exponent of the distribution of the state-wise action

13


---Page Break---
value gaps. We also note that these bounds are explicitly depending on the size of the action space A,
which could be a potential drawback of the entropy-based method.

Compared to these results, our sample complexity bound has the following advantages. First of all, it
gives the performance guarantees even if there is no concentrable policies. Second, it only requires
the model-free realizability with respect to a (fixed) worst-case optimal policy ˜π∗. Third, it achieves
the statistically reasonable rate O(ϵ−2) without any dependencies on the action value gap. Finally,
it has no explicit dependency in the algorithm on the size of S and A, even in the policy extraction
process.

B
Rademacher Complexity and Uniform Convergence

In this section, we introduce the Rademacher complexity and its properties as well as the celebrated
uniform convergence theorem. Below, let Z be a sample space, p ∈∆(Z) be a probability distribution
on it, and G ⊂RZ be a set of functions from Z to R.

Definition B.1 (Rademacher complexity). The Rademacher complexity of G with the sample size
n ≥1 is given by

Rn(G) := Eσn,zn

"

sup
g∈G

1
n

n
X

i=1
σig(zi)

#

,
(23)

where Eσn,zn denotes the expectation with respect to samples σn = {σi}n
i=1 and zn = {zi}n
i=1
drawn from Uniformn({−1, +1}) and pn, respectively.

Lemma B.1 (Uniform convergence theorem). Suppose ∥g∥∞≤c for all g ∈G. Then, for all
δ ∈(0, 1), we have

sup
g∈G


1
n

n
X

i=1
[g(zi)] −E [g(z1)]

 ≤2Rn(G) + c

r

2 ln(2/δ)

n

with probability 1 −δ on the draw of zn ∼pn.

Proof. Refer to Claim 1, Theorem 26.5, Shalev-Shwartz and Ben-David (2014) for one-
side high-probability bound and apply it to both of supg∈G
 1

n
Pn
i=1 [g(zi)] −E [g(z1)]
	
and
supg∈G(−1)
 1

n
Pn
i=1 [g(zi)] −E [g(z1)]
	
setting the confidence parameter to δ/2. The proof is
completed by taking the union of the events that these high-probability bounds do not hold.

The following is another well-known result of the Rademacher complexity.

Lemma B.2 (Massart’s lemma). For a finite set G, we have

Rn(G) ≤M(G)

r

2 ln |G|

n
,

where

M(G) :=
sup
g,g′∈G, z∈Z
|g(z) −g′(z)| .

Proof. Refer to Shalev-Shwartz and Ben-David (2014), Lemma 26.8.

C
Basic Properties of Regularized Lagrangian

In this section, we show basic property of regularized Lagrangian (Eq. (7)).

Lemma C.1 (Primal Lipschitz continuity). For all v, v′ : S × A →[0, (1 −γ)−1] and f ∈F, we
have

|K(v, f) −K(v′, f)| ≤(2 + BF) ∥v −v′∥1,¯µ

14


---Page Break---
Proof. Observe that by Eq. (7)

|K(v, f) −K(v′, f)| ≤(1 −γ) ∥v −v′∥1,µ +
f · (δTDv −δTDv′)

1,µβ + (1 −γ) ∥v −v′∥1,¯µ .

The second term on the RHS is further bounded as
f · (δTDv −δTDv′)

1,µβ = ∥f · (I −γT )(v −v′)∥1,µβ

(a)
≤BF ∥(I −γT )(v −v′)∥1,µβ

(b)
≤BF ∥v −v′∥1,¯µ ,

where (a) follows from Hölder’s inequality and (b) follows from the triangle inequality. We obtain
the desired result by summing up both sides of the two inequalities since µ ≤¯µ.

Lemma C.2 (Stability of minimax value against dual error). For all v ≥0 and π ∈Πβ, we have

K(v, f) ≥U ∗−¯BV
f −˘f π
1,µβ −2(1 −γ) ∥˜v∗−˜vπ∥1,¯µ ,

where ¯BV := max {1 + γBV, BV}.

Proof. Observe that

K(v, f)
(a)
≥K(v, f) −Dπ
V(v)

(b)= U(π) −Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
−Dπ
F(f)

(c)= U(π) −Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i

−Eµ,β
h
(f −˘f π)(s, a) {˜vπ(s) −˜qπ(s, a)}
i

(d)= U(π) + Eµ,β
h
(f −˘f π)(s, a)
n
r(s, a) + γ ˜T v(s, a) −v(s)
oi

(e)
≥U(π) −¯BV
f −˘f π
1,µβ ,

where (a) follows from the nonegativity of Dπ
V(v), (b) from Lemma E.3, (c) from Dπ
F( ˘f π) = 0, (d)
from ˜qπ(s, a) −γ ˜T v(s, a) = r(s, a) and (e) from Hölder’s inequality with |r + γ ˜T v −v| ≤¯BV.
The claim is then proved by the continuity of U(π),

U ∗−U(π) = (1 −γ) ∥˜v∗−˜vπ∥1,µ + (1 −γ)2

2
E¯µ [(˜v∗+ ˜vπ)(˜v∗−˜vπ)]

≤(1 −γ) ∥˜v∗−˜vπ∥1,µ + (1 −γ) ∥˜v∗−˜vπ∥1,¯µ
≤2(1 −γ) ∥˜v∗−˜vπ∥1,¯µ .

D
Proofs of Section 4

D.1
Proof of Theorem 4.1

Proof. The claim
˜
M ∈U is trivial from Definition 4.1. The other claim, J(π| ˜
M) ≤J(π|M′),
follows from that i) ˜r(s, a) ≤r(s, a) with ˜r(s, a) = Ey∼R(y|s,a) [y] for all s ∈S and a ∈A and ii)
the truncated transition probability ˜T(·|s, a), when in conflict with T(·|s, a), leads to the absorbing
state ⊥, which has the lowest possible cumulative discounted value, zero.

D.2
Proof of Corollary 4.1

Proof. Since the inequality is trivial from Eq. (1), we show the equality. Observe that the state-action
pair (st, at) at time t ≥0 stays inside supp(µβ) almost surely for all t ≥0 if supp(dππ) ⊂
supp(µβ). Hence, the law of the reward sequence {rt}t≥0 generated with π is the same under M
and ˜
M, leading to J(π| ˜
M) = J(π|M).

15


---Page Break---
D.3
Proof of Lemma 4.1

Proof. We show denying the conclusion results in contradiction.
Take a policy π
∈
˜Π∗.
Let χβ(s, a) := I {(s, a) ∈supp(β)} be the indicator function of supp(β).
Let zπ(s) :=
P

a χβ(s, a) π(a|s) be the mass of π(·|s) inside the support of β, which satisfies 0 ≤zπ(s) ≤1 for
all s ∈S.

Let π′ : S →∆(A) be a policy proportional to π with the support restricted to supp(β), i.e.,
zπ(s) π′(a|s) = χβ(s, a) π(a|s) for all s ∈S and a ∈A.

By the definitions of ˜rπ and ˜T π, we now have

˜rπ(s) =
X

a
χβ(s, a) π(a|s) r(s, a)

=
X

a
zπ(s) π′(a|s) r(s, a)

= zπ(s)
X

a
χβ(s, a) π′(a|s) r(s, a)

= zπ(s) ˜rπ′(s) ≤˜rπ′(s)

and

˜T πv(s) =
X

a,s′
π(a|s) χµ,β(s, a) T(s′|s, a) v(s′)

=
X

a,s′
zπ(s) π′(a|s) T(s′|s, a) v(s′)

= zπ(s)
X

a,s′
π′(a|s) χµ,β(s, a) T(s′|s, a) v(s′)

= zπ(s) ˜T π′v(s) ≤˜T π′v(s)

for all v : S →[0, ∞) and s ∈S. Therefore, we have for all s ∈S

˜v∗(s) = ˜vπ(s) = (I −γ ˜T π)−1rπ(s) =
X

t≥0
(γ ˜T π)trπ(s) ≤
X

t≥0
(γ ˜T π′)trπ′(s) = ˜vπ′(s),

which leads to the strong optimality ˜vπ′ = ˜v∗. However, we also have π′ ∈Πβ by the definition,
contradicting with the assumption ˜Π∗∩Πβ = ∅.

E
Proofs of Section 5

E.1
Proof of Lemma 5.1

The proof relies on two lemmas (Lemmas E.1 and E.2), where the second one is built on top of the
first one. Then, Lemma 5.1 is immediately proved as a special case of Lemma E.2 with π being
restricted to ˜Π∗.

Lemma E.1 gives a saddle-point decomposition of Lagrangian ignoring the offline constraint.

Lemma E.1 (Incomplete saddle-point decomposition of Lagrangian). For all π : S →∆(A),

L(v, f) = J(π) +
X

s,a
(fµβ −dππ)(s, a) δTDv(s, a).
(24)

Proof. Comparing the LHS and the RHS, it suffices to show

J(π) −(1 −γ)Ep0 [v(s)] =
X

s,a
dπ(s) π(a|s) δTDv(s, a).

16


---Page Break---
This is seen by simplifying the RHS of the above equation as follows,
X

s,a
dπ(s) π(a|s) δTDv(s, a)
(a)=
X

s
dπ(s) (I −γT π)(vπ −v)

(b)=
X

s
(I −γT π
∗)dπ(s) (vπ −v)

(c)= (1 −γ)
X

s
p0(s) (vπ −v)

(d)= J(π) −(1 −γ)Ep0 [v(s)] .

Here, (a) follows from P

a π(a|s) δTDv(s, a) = (rπ + γT πv −v)(s) and ˜rπ = (I −γT π)vπ, (b)
from (I −γT π
∗) being the adjoint operator of (I −γT π), (c) from (1 −γ)p0 = (I −γT π
∗)dπ, and
(d) from the definition of J(π).

Note that the second term of Eq. (24) may have no root with respect to f if the data support does not
cover the visitation distribution dπ, hence incomplete. Lemma E.2 gives a modification of Eq. (24) to
fix this problem.
Lemma E.2 (Generalized saddle-point decomposition of Lagrangian). For all π ∈Πβ, we have

L(v, f) = ˜J(π) −Eµ,β
h
(f −˜f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
+ Dπ
V(v) −Dπ
F(f),
(25)

where Dπ
V(v) := P

s̸∈supp(µ) ˜dπ(s) v(s) and Dπ
F(f) := Eµ,β [f(s, a) {˜vπ(s) −˜qπ(s, a)}] .

Proof. Since Lagrangian is defined with r(s, a) and T(·|s, a) only on the support of the offline data
distribution supp(µβ), Lemma E.1 together with the indistinguishability of ˜
M and M gives

L(v, f) = ˜J(π) +
X

s,a
(fµβ −˜dππ)(s, a) ˜δTDv(s, a),

where ˜δTDv := ˜r + γ ˜T v −v. The second term of the RHS of the above equation further evaluated
by separating the summation to the on-support and off-support terms
X

s,a
(fµβ −˜dππ)(s, a) ˜δTDv(s, a)

=




X

(s,a)∈supp(µβ)
+
X

(s,a)̸∈supp(µβ)



(fµβ −˜dππ)(s, a) ˜δTDv(s, a)

(a)= Eµ,β
h
(f −˜f π)(s, a) ˜δTDv(s, a)
i
+
X

(s,a)̸∈supp(µβ)

˜dπ(s) π(a|s) v(s)

(b)= −Eµ,β
h
(f −˜f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
−Dπ
F(f)

+




X

s̸∈supp(µ),a∈A
+
X

s∈supp(µ),a̸∈supp(β(s))



˜dπ(s) π(a|s) v(s)

(c)= −Eµ,β
h
(f −˜f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
−Dπ
F(f) + Dπ
V(v),

where (a) follows from that ˜f π(s, a)µ(s)β(a|s) =
˜dπ(s)π(a|s) if (s, a) ∈supp(µβ) and
δTDv(s, a) = −v(s) if (s, a) ̸∈supp(µβ), (b) from transforming ˜r in the first term, within ˜δTD, with
˜r = (I −γ ˜T )˜vπ +(˜qπ −˜vπ) and simplify the resulting (˜qπ −˜vπ) with Ea∼β(a|s)[ ˜f π(s, a)(˜qπ(s, a)−
˜vπ(s))] = 0, and (c) from evaluating the last summation as zero with the fact supp(µπ) ⊂
supp(µβ).

Finally, Lemma 5.1 is shown by taking π such that π ∈Πβ ∩˜Π∗.

17


---Page Break---
E.2
Proof of Theorem 5.1

Proof. First, see Appendix E.1 for the proof of Lemma 5.1. Then, recall that (˜v∗, ˜f π) is a saddle point
of L(v, f) if L(˜v∗, f) ≤L(˜v∗, ˜f π) ≤L(v, ˜f π) for all v, f ≥0. By Eq. (6) with the nonnegativity
of Dπ
V(v) and D∗
F(f), it suffices to show Dπ
V(˜v∗) = 0 and D∗
F( ˜f π) = 0. The former follows from
˜v∗(s) = 0 for all s ̸∈supp(µ) and the latter follows from Ea∼β(a|s)[ ˜f π(s, a) {˜v∗(s) −˜q∗(s, a)}] =
˜wπ(s)

˜v∗(s) −Ea∼π(a|s)[˜q∗(s, a)]
	
= 0 for all π ∈Πβ ∩˜Π∗.

E.3
Proof of Corollary 5.1

Proof. Note that the existence of the saddle points is always ensured since the (restricted) domains
of v and f are all convex. Therefore, the first claim is reduced to 0 = argmaxf≥0 minv∈Vϵ L(v, f),
which is shown by Lemma 5.1 and the fact (I −γ ˜T )(v −˜v) ≥(1 −γ) ϵ > 0 for all v ∈Vϵ. Besides,
the second claim is reduced to 0 = argminv≥0 maxf∈Fϵ L(v, f), which is also shown by Lemma 5.1
and the fact Eµ,β[(f −˜f π)(s, a) (I −γ ˜T )(v −˜v∗)(s, a)] ≥(1 −γ) ϵ E ˜T π
∗µ [(v −˜v∗)(s)], which
attains the minimum uniquely with v = 0.

E.4
Proof of Lemma 5.2

Define

U(π) := (1 −γ)Eµ [˜vπ(s)] + (1 −γ2)

2
∥˜vπ∥2
2,¯µ .

Then, Lemma 5.2 is proved as a special case of the following lemma with the restriction π ∈Πβ ∩˜Π∗

and U ∗:= (1 −γ)Eµ [˜v∗(s)] + (1−γ2)

2
∥˜v∗∥2
2,¯µ.

Lemma E.3 (Generalized saddle-point decomposition of regularized Lagrangian). For all π ∈Πβ,

K(v, f) = U(π) −Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
+ ˘Dπ
V(v) −Dπ
F(f).
(26)

Proof. Observe that

v2 = (v −˜vπ)2 + 2˜vπv −(˜vπ)2.

Thus, multiplying both sides with (1 −γ)2/2 and plugging to Eq. (7), we get

K(v, f) = (1 −γ)2

2


∥v −˜vπ∥2
2,¯µ −∥˜vπ∥2
2,¯µ


+ (1 −γ)
X

s
{µ(s) + (1 −γ) ˜vπ(s) ¯µ(s)} v(s) + Eµ,β

f(s, a) δTDv(s, a)

.

Now, applying Lemma E.2 on the last two terms of the RHS with the formal substitution p0 ←˘pπ,
we have

K(v, f) = (1 −γ)2

2


∥v −˜vπ∥2
2,¯µ −∥˜vπ∥2
2,¯µ


+
X

s
˘dπ(s) ˜rπ(s) −Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i

+
X

s̸∈supp(µ)

˘dπ(s) v(s) −Dπ
F(f)

=
X

s
˘dπ(s) ˜rπ(s) −(1 −γ)2

2
∥˜vπ∥2,¯µ

−Eµ,β
h
(f −˘f π)(s, a) (I −γ ˜T )(v −˜vπ)(s, a)
i
+ ˘Dπ
V(v) −Dπ
F(f),

where ˜J(π), ˜dπ(s) and ˜f π(s, a) in Lemma 5.1 are replaced with P

s ˘dπ(s) rπ(s), ˘dπ(s) and ˘f π(s, a),
respectively, due to the substitution. Finally, the proof is concluded by simplifying the first term on

18


---Page Break---
the RHS
X

s
˘dπ(s) ˜rπ(s) = (1 −γ)
X

s
(I −γ ˜T π
∗)−1 {µ + (1 −γ)˜vπ ¯µ} (s) ˜rπ(s)

(a)= (1 −γ)
X

s
{µ + (1 −γ)˜vπ ¯µ} (I −γ ˜T π)−1˜rπ(s)

(b)= U(π) + (1 −γ)2

2
∥˜vπ∥2,¯µ ,

where (a) follows from the fact (I −γ ˜T π
∗)−1 is the adjoint operator of (I −γ ˜T π)−1 and (b) is owing
to ˜vπ = (I −γ ˜T π)˜rπ.

E.5
Proof of Theorem 5.2

Proof. First, see Appendix E.4 for the proof of Lemma 5.2. Then, note that the third term ˘Dπ
V(v)
is nonegative for all v ≥0 since ˘dπ ≥0. Thus, we have K(v, f) ≥U ∗taking f = ˘f π and
K(v, f) ≤U ∗taking v = ˜v∗. The first claim is then proved by combining these two inequalities, in
the same manner as the proof of Theorem 5.1. The second claim, the uniqueness of ˜v∗, follows from
the strong convexity of K(·, f).

E.6
Proof of Lemma 5.3

Proof. Denote the primal excess risk function relative to F by

εPER(v) := max
f∈F K(v, f) −min
v≥0 max
f∈F K(v, f)

and its minimizer by v∗:= argminv≥0 maxf∈F K(v, f). Now, the strong convexity of K(·, f)
implies the strong convexity of εPER and thus, for all v : S →R,

(1 −γ)2

2
∥v −v∗∥2
2,¯µ ≤εPER(v).

We utilize this bound via the triangle inequality

˜v∗
♯−˜v∗
2,¯µ ≤
˜v∗
♯−v∗
2,¯µ + ∥v∗−˜v∗∥2,¯µ ≤

q

2εPER(˜v∗
♯) +
p

2εPER(˜v∗)

1 −γ
.

Then, each term on the RHS is bounded by

εPER(˜v∗) ≤2ϵF,
(27)
εPER(˜v∗
♯) −εPER(˜v∗) ≤(2 + BF)ϵV,
(28)

completing the proof. The proofs of Eqs. (27) and (28) are separately given below.

Proof of Eq. (27). Recall that by definition

εPER(˜v∗) = max
f∈F K(˜v∗, f) −argmin
v≥0
max
f∈F K(v, f).

Then, on the RHS, the first term is upper bounded with U ∗by Theorem 5.2 and the second term
(without the negative sign) is lower bounded by Lemma C.2, leading to the inequality

εPER(˜v∗) ≤min
f∈F


¯BV
f −˘f π
1,µβ + 2(1 −γ) ∥˜v∗−˜vπ∥1,¯µ



for all π ∈Πβ. The proof is completed by taking the minimum with respect to π.

Proof of Eq. (28). It is immediately seen from

εPER(˜v∗
♯) −εPER(˜v∗) = min
v∈V max
f∈F K(v, f) −max
f∈F K(˜v∗, f)

≤min
v∈V max
f∈F {K(v, f) −K(˜v∗, f)} .

≤(2 + BF) min
v∈V ∥v −˜v∗∥1,¯µ ,

where the last inequality follows from Lemma C.1.

19


---Page Break---
F
Proofs of Section 6

F.1
Proof of Theorem 6.1

Recall that the average action value gap is given by

Γ(π) := Eµ,π [˜v∗(s) −˜q∗(s, a)] .

The following lemma establishes the connection of Γ(πθ) and ϵest(θ).
Lemma F.1. For all θ ∈Θ, we have

Γ(πθ) ≤ϵest(θ) + εapp,Π(V, Θ, Ξ)

1 −γ
.

Proof. Refer to Appendix F.2.

Then, Theorem 6.1 is proved by combining Lemma F.1 with the following lemma.
Lemma F.2. For all π : S →∆(A), we have

˜J∗−˜J(π) ≤∥˜wπ∥∞Γ(π).

Proof. It follows directly from Hölder’s inequality with the performance difference lemma for the
truncated environment (Lemma F.3).

F.2
Proof of Lemma F.1

The proof relies on the following variant of the performance difference lemma adopted for the
worst-case environment ˜
M.
Lemma F.3 (Worst-case performance difference lemma). For all π : S →∆(A), we have

˜J∗−˜J(π) = Eµ,π [ ˜wπ(s) {˜v∗(s) −˜q∗(s, a)}] .

Consequently, for all π ∈Πβ,

˜J∗−˜J(π) = Eµ,β
h
˜f π(s, a) {˜v∗(s) −˜q∗(s, a)}
i
.

Proof. Observe that

˜J∗= (1 −γ)
X

s
p0(s) ˜v∗(s) =
X

s
˜dπ(s) (I −γ ˜T π)˜v∗(s) = Eµ
h
˜wπ(s) (I −γ ˜T π)˜v∗(s)
i
,

where the second equality follows from (I −γ ˜T π
∗) ˜dπ = p0 and the third equality follows from
˜v∗(s) = 0 and ˜T πv(s) = 0 for all s ̸∈supp(µ) and π : S →∆(A). Thus,

˜J∗−J(π) = Eµ
h
˜wπ(s) (I −γ ˜T π)˜v∗(s)
i
−Eµ [ ˜wπ(s) ˜rπ(s)]

= Eµ [ ˜wπ(s) {˜v∗(s) −Pπ˜q∗(s)}]
= Eµ,π [ ˜wπ(s) {˜v∗(s) −˜q∗(s, a)}]

where the second equality follows from ˜rπ + γ ˜T π˜v∗= Pπ˜q∗. This proves the first claim.

The second claim follows from the definition of ˜f π.

Let ˘p∗:= µ + (1 −γ)˜v∗¯µ be an alternative (unnormalized) initial state distribution and

˘f π,∗(s, a) := (1 −γ)(I −γT π
∗)˘p∗(s)
µ(s)
ρπ(s, a)
(29)

be the corresponding action visitation weight function. Applying Lemma F.3 to this setting, we obtain
the following corollary.

20


---Page Break---
Corollary F.1. Define

˜U(π) := (1 −γ)
X

s
˘p∗(s) ˜vπ(s)

and let ˜U ∗:= (1 −γ) P

s ˘p∗(s) ˜v∗(s) be its maximum. Then, we have

˜U ∗−˜U(π) = Eµ,β
h
˘f π,∗(s, a) {˜v∗(s) −˜q∗(s, a)}
i
.

Now we are prepared to prove Lemma F.1.

Proof of Lemma F.1. Let ξ∗
♯∈argminξ∈Ξ ∥ξ −˜ξ∗∥1,µβ0. Now, observe that

Γ(πθ) = Eµ,πθ
h
−˜ξ∗(s, a)
i

(a)
≤Eµ,πθ

−ξ∗
♯(s, a)

+ BΠϵΞ

(b)
≤
1
1 −γ Eµ,πθ

wθ(s)

−ξ∗
♯(s, a)
	
+ BΠϵΞ

(c)
≤
1
1 −γ

Eµ,β

fθ(s, a)

−ξ∗
♯(s, a)
	
+ DΞ(fθ; wθ, πθ)
	
+ BΠϵΞ

(d)
≤
1
1 −γ

n
Eµ,β
h
fθ(s, a)
n
−˜ξ∗(s, a)
oi
+ BFϵΞ + DΞ(fθ; wθ, πθ)
o
+ BΠϵΞ

(e)=
1
1 −γ {D∗
F(fθ) + LX(θ) + BFϵΞ} + BΠϵΞ
(30)

where (a) follows from ∥ξ∗
♯−˜ξ∗∥1,µπθ ≤BΠϵΞ, (b) from Hölder’s inequality with −˜ξ∗(s, a) ≥0
and wθ ≥1 −γ, and (c) from Eq. (11), (d) from ∥ξ∗
♯−˜ξ∗∥1,µβ ≤BΠϵΞ, and (e) from the definition
of D∗
F(f) (Lemma 5.1) and LX(θ). In the rest of the proof, we bound D∗
F(fθ) + LX(θ) with ϵest(θ).

Fix any θ′
∈
Θ and π
∈
Πβ.
Let ˜v∗
♯
∈
argminv∈V maxf∈F K(v, f) and
˘f ∗
♯
∈
argmaxf∈F K(˜v∗
♯, f). Also let ¯K∗:= K(˜v∗
♯, ˘f ∗
♯) = minv∈V maxf∈F K(v, f) be the correspond-
ing minimax value. Then, ¯K∗+ LSP(θ) is lower-bounded with D∗
F(fθ) −D∗
F(fθ′) up to an error
term,

¯K∗+ LSP(θ) = max
v∈V

n
K(˜v∗
♯, ˘f ∗
♯) −K(v, fθ)
o

(a)
≥K(˜v∗
♯, ˘f ∗
♯) −K(˜v∗
♯, fθ)

(b)
≥K(˜v∗
♯, fθ′) −K(˜v∗
♯, fθ)

(c)= D∗
F(fθ) −D∗
F(fθ′) + Eµ,β
h
(fθ −fθ′)(s, a) (I −γ ˜T )(˜v∗
♯−˜v∗)(s, a)
i

(d)
≥D∗
F(fθ) −D∗
F(fθ′) −2BF
˜v∗
♯−˜v∗
1,¯µ

(e)
≥D∗
F(fθ) −D∗
F(fθ′) −2BF εapp,V(V, F),
(31)

where (a) follows from compromising the maximum with v = ˜v∗
♯∈V, (b) from the definition of ˘f ∗
♯
above, (c) from Lemma 5.2, (d) from ∥fθ∥∞, ∥fθ′∥∞≤BF and (e) from Lemma 5.3.

Moreover, let

Φ(θ′) := min
π∈Πβ


¯BV,Ξ
fθ′ −˘f π
1,µβ + 2(1 −γ) ∥˜v∗−˜vπ∥1,¯µ



21


---Page Break---
be the intrinsic error of θ′ and fix arbitrary π∗∈Πβ ∩˜Π∗. Then, ¯K∗+ LSP(θ′) is upper-bounded
with Φ(θ′) up to another error term,

¯K∗+ LSP(θ′) = ¯K∗−U ∗+ U ∗−min
v∈V K(v, fθ′)

(a)= K(˜v∗
♯, ˘f ∗
♯) −K(˜v∗, ˘f π∗) + U ∗−min
v∈V K(v, fθ′)

(b)
≤K(˜v∗
♯, ˘f ∗
♯) −K(˜v∗, ˘f ∗
♯) + U ∗−min
v∈V K(v, fθ′)

(c)
≤(2 + BF)
˜v∗
♯−˜v∗
1,¯µ + Φ(θ′)

(d)
≤(2 + BF)εapp,V(V, F) + Φ(θ′),
(32)

where (a) follows from Lemma 5.2, (b) from Theorem 5.2, (c) from Lemma C.1 and Lemma C.2 and
(d) from Lemma 5.3.

Subtracting both sides of Eq. (31) from those of Eq. (32), we get

D∗
F(fθ) ≤LSP(θ) −LSP(θ′) + D∗
F(fθ′) + Φ(θ′) + (2 + 3BF)εapp,V(V, F).

which, summed with LX(θ) ≤LX(θ) + ϵest(θ′) = ϵest(θ) −LSP(θ) + L(θ′) on both sides, yields

D∗
F(fθ) + LX(θ) ≤ϵest(θ) + D∗
F(fθ′) + LX(θ′) + Φ(θ′) + (2 + 3BF)εapp,V(V, F).
(33)

The remaining task is to choose θ′ ∈Θ such that D∗
F(fθ′) + LX(θ′) + Φ(θ′) is nicely bounded. Now,
observe that

D∗
F(fθ′) = Eµ,β [fθ′(s, a) {˜v∗(s) −˜q∗(s, a)}]

(a)
≤min
π∈Πβ

n
Eµ,β
h
˘f π,∗(s, a) {˜v∗(s) −˜q∗(s, a)}
i
+ Eµ,β
h
(fθ′ −˘f π)(s, a) {˜v∗(s) −˜q∗(s, a)}
io

(b)= min
π∈Πβ

n
˜U ∗−˜U(π) −Eµ,β
h
(fθ′ −˘f π)(s, a) ˜ξ∗(s, a)
io

(c)
≤min
π∈Πβ


(1 −γ) ∥˜v∗−˜vπ∥1,˘p∗+
˜ξ∗
∞

fθ′ −˘f π
1,µβ



(d)
≤Φ(θ′),
(34)

where (a) follows from ˘f π,∗≥˘f π (see Eq. (29) for the definition of ˘f π,∗), (b) from Lemma F.3, (c)
from Hölder’s inequality and (d) from ˘p∗≤2¯µ and ∥˜ξ∗∥∞≤¯BV,Ξ. Moreover, for all π ∈Πβ, we
have

LX(θ′) = max
ξ∈Ξ {Eµ,β [fθ(s, a)ξ(s, a)] −Eµ,πθ [wθ(s)ξ(s, a)]}

(a)
≤max
ξ∈Ξ Eµ,β
h
(fθ −˘f π)(s, a)ξ(s, a)
i
+ max
ξ∈Ξ Eµ,π [( ˘wπ −wθ)(s)ξ(s, a)]

+ max
ξ∈Ξ {Eµ,π [wθ(s)ξ(s, a)] −Eµ,πθ [wθ(s)ξ(s, a)]}

(b)
≤BΞ

fθ −˘f π
1,µβ + ∥wθ −˘wπ∥1,µ + BW ∥πθ −π∥TV,µ


,
(35)

where (a) follows from the telescoping and (b) from ∥ξ∥∞≤BΞ for all ξ ∈Ξ and ∥˘wπ −wθ∥1,µ ≤
∥˘wπ −wθ∥1,µ since ˘wπ ≥1 −γ. Adding both sides of Eqs. (34) and (35) and taking π and θ′ as the
minimizers of Eqs. (12) and (13), respectively, we arrive at

D∗
F(fθ′) + LX(θ′) + Φ(θ′) ≤2Φ(θ′) + BΞϵΘ(π) ≤3ϵΘ
(36)

where the last inequality follows from Φ(θ′) ≤ϵΘ (Eq. (12)). The proof is concluded by adding both
sides of Eqs. (30), (33) and (36) and simplifying the error terms with Eq. (16).

22


---Page Break---
F.3
Extensions of Corollary 6.1

The suboptimality bound established by Corollary 6.1 requires the uniform boundedness of ∥˜wπθ∥∞.
With more careful analysis, however, we can obtain policy suboptimality bounds with milder condi-
tions. We present two of such upper bounds below.

To this end, we introduce two types of local truncated concentrability. Let Π ≡Π(Θ) := {πθ | θ ∈Θ}
be the set of all the policy candidates.

Definition F.1. Let ˜Cϵ := max{∥˜wπ∥∞| ˜J∗−˜J(π) ≤ϵ, π ∈Π} be the ϵ-weakly local truncated
concentrability (ϵ-WLTC) coefficient for ϵ > 0. We also define the limit WLTC coefficient as
˜C0 := limϵ→0+ ˜Cϵ.
Definition F.2. Let ˜cϵ := max{∥˜wπ∥∞| max(s,a)∈supp(µπ) {˜v∗(s) −˜q∗(s, a)} ≤ϵ, π : S →
∆(A)} be the ϵ-strongly local truncated concentrability (ϵ-SLTC) coefficient for ϵ > 0. We also
define the limit SLTC coefficient as ˜c0 := limϵ→0+ ˜cϵ.

Intuitively, both the WLTC and SLTC coefficients bound the norm of ˜wπ locally for near-optimal
policies π. The difference is the ways they measure the locality: WLTC uses the policy suboptimality
and SLTC uses the maximum action value gap. Note that WLTC dominates SLTC, ˜cϵ ≤˜Cϵ ≤˜C∞, if
Π(Θ) covers the entire policy space ∆(A)S. In general, there is no particular order between WLTC
and SLTC and we just have ˜Cϵ ≤˜C∞.

The following lemma is the foundation of our local concentrability results.

Lemma F.4. For any π ∈Π, we have

˜J∗−˜J(π) ≤

1 + ˜cϵ0(π)

˜c0


ϵ0(π),

where ϵ0(π) :=
p

˜c0Γ(π)/(1 −γ).

Proof. Refer to Appendix F.4.

Combining it with Lemma F.1 and discarding the non-asymptotic term for the simplicity, we get the
first bound as the following corollary.

Corollary F.2. For all θ ∈Θ, we have

˜J∗−˜J(πθ) ≲2

s

˜c0
1 −γ {ϵest(θ) + εapp,Π(V, Θ, Ξ)},
(37)

where a ≲b means lim supb→0+ a/b ≤1.

In words, Eq. (37) allows us to replace the uniform concentrability coefficient ˜C∞with the limit
SLTC coefficient ˜c0 at the cost of the slower convergence rate due to the square root.

Moreover, a faster bound can be obtained exploiting the limit WLTC coefficient ˜C0 instead of ˜c0,
leading to our second bound.

Corollary F.3. For all θ ∈Θ, we have

˜J∗−˜J(πθ) ≲
˜C0
1 −γ {ϵest(θ) + εapp,Π(V, Θ, Ξ)} .
(38)

Proof. Observe that Corollary F.2 implies ∥˜wπθ∥∞is bounded with ˜Cϵ, where ϵ is taken as the
RHS of Eq. (37). The claim thus follows from Theorem 6.1 with ∥˜wπθ∥∞≤˜Cϵ →˜C0 as ϵest(θ) +
εapp,Π(V, Θ, Ξ) →0.

Similarly as Corollary F.2, in comparison to Theorem 6.1, the coefficient of the upper bound is
improved from ˜C∞to ˜C0, meaning that asymptotically we only need the weakly local, not uniform,
concentrability.

23


---Page Break---
F.4
Proof of Lemma F.4

Let us denote the set of the ϵ-strongly near-optimal policies by

˜Π∗
ϵ :=
n
π : S →∆(A)
 supp(π(s)) ⊂supp( ˜
A∗
ϵ(s)), s ∈supp(µ)
o
,
ϵ ≥0,

where ˜
A∗
ϵ(s) := {a ∈A | ˜v∗(s) −˜q∗(s, a) ≤ϵ} is the ϵ-optimal action subset for s ∈S.7 Note that
by definition ˜Π∗
0 = ˜Π∗and ˜Π∗
ϵ is monotone nondecreasing with respect to ϵ, reaching the set of the
all policies with ϵ ≥1/(1 −γ). Then, the SLTC coefficient can be written in terms of ˜Π∗
ϵ,

˜cϵ = max
π∈˜Π∗ϵ
∥˜wπ∥∞.

Now, to prove Lemma F.4, we show that there exists a strongly optimal policy ¯π ∈˜Π∗
ϵ that approxi-
mates the target policy π if Γ(π) is small.

Lemma F.5. For all π : S →∆(A) and ϵ > 0, we have

min
¯π∈˜Π∗ϵ
∥π −¯π∥TV,µ ≤2Γ(π)

ϵ
.

Proof. Take π′ ∈˜Π∗
ϵ as a projection of π onto ˜Π∗
ϵ, i.e.,

π′(a|s) = 1{a ∈˜
A∗
ϵ(s)}π(a|s) + c(s) π0(a|s),
s ∈S,

with arbitrary π0 ∈˜Π∗
ϵ and c(s) := 1 −P

a∈˜
A∗ϵ (s) π(a|s). Observe that, by the triangle inequality,

∥π −π′∥TV,µ = Eµ
X

a
|π(a|s) −π′(a|s)|

≤Eµ
X

a

n
(1 −1{a ∈˜
A∗
ϵ(s)})π(a|s) + c(s) π0(a|s)
o

= 2Eµ [c(s)] .

Then, we have

min
¯π∈˜Π∗ϵ
∥π −¯π∥TV,µ ≤2Eµ [c(s)] = 2Eµ,π
h
1
n
a ̸∈˜
A∗
ϵ(s)
oi
.

Now, plugging

1
n
a ̸∈˜
A∗
ϵ(s)
o
= 1 {˜v∗(s) −˜q∗(s, a) > ϵ} ≤1

ϵ {˜v∗(s) −˜q∗(s, a)}

on the RHS, we arrive at the desired inequality

min
¯π∈˜Π∗ϵ
∥π −¯π∥TV,µ ≤2

ϵ Eµ,π [{˜v∗(s) −˜q∗(s, a)}] = 2

ϵ Γ(π).

As a corollary, we can also bound the difference of their policy values based on the SLTC coefficient.

Corollary F.4. For all π : S →∆(A) and ϵ > 0, we have

min
¯π∈˜Π∗ϵ

 ˜J(π) −˜J(¯π)
 ≤
2˜cϵ
1 −γ
Γ(π)

ϵ
.

7It is referred to as the strong near-optimality since it implies the near-optimality in the usual sense, i.e.,
˜J∗−˜J(π) ≤ϵ for all π ∈˜Π∗
ϵ.

24


---Page Break---
Proof. Take arbitrary ¯π ∈˜Π∗
ϵ and observe that

˜dπ(s) −˜d¯π(s) = (1 −γ)
n
(I −γ ˜T π
∗)−1 −(I −γ ˜T ¯π
∗)−1o
p0(s)

= (1 −γ)(I −γ ˜T π
∗)−1 n
(I −γ ˜T ¯π
∗) −(I −γ ˜T π
∗)
o
(I −γ ˜T ¯π
∗)−1p0(s)

= γ(1 −γ)(I −γ ˜T π
∗)−1 
˜T π
∗−˜T ¯π
∗

(I −γ ˜T ¯π
∗)−1p0(s)

= γ(I −γ ˜T π
∗)−1 ˜T∗
 
Pπ
∗−P ¯π
∗
 ˜d¯π(s).

Thus, we have
 ˜wπ −˜w¯π
1,µ =
X

s∈supp(µ)

 ˜dπ(s) −˜d¯π(s)


=
X

s∈supp(µ)

γ(I −γ ˜T π
∗)−1 ˜T∗
 
Pπ
∗−P ¯π
∗
 ˜d¯π(s)


(a)
≤
γ
1 −γ

X

(s,a)∈supp(µβ)


 
Pπ
∗−P ¯π
∗
 ˜d¯π(s, a)


=
γ
1 −γ Eµ



˜d¯π(s)

µ(s)

X

a∈supp(β(s))
|π(a|s) −¯π(a|s)|





(b)
≤
γ˜cϵ
1 −γ Eµ

"X

a
|π(a|s) −¯π(a|s)|

#

=
γ˜cϵ
1 −γ ∥π −¯π∥TV,µ ,
(39)

where (a) follows from the fact ˜T π
∗and ˜T∗can be identified as non-expansive mappings of types
L1(supp(µ)) →L1(supp(µ)) and L1(supp(µβ)) →L1(supp(µ)), respectively, and (b) follows
from ¯π ∈˜Π∗
ϵ. Finally, observe that
 ˜J(π) −˜J(¯π)
 =
Eµ

( ˜wπ˜rπ −˜w¯π˜r¯π)(s)


(a)
≤Eµ
( ˜wπ −˜w¯π)˜rπ (s) +
 ˜w¯π(˜rπ −˜r¯π)
 (s)


(b)
≤
 ˜wπ −˜w¯π
µ,1 + ˜cϵ ∥π −¯π∥TV,µ

(c)
≤
˜cϵ
1 −γ ∥π −¯π∥TV,µ ,

where (a) follows from the triangle inequality, (b) from the fact |˜r(s, a)| ≤1 and (c) from Eq. (39).
The proof is concluded by applying Lemma F.5 on the RHS.

We now arrive at the following corollary, which immediately implies Lemma F.4 as a special case
with the substitution ϵ = ϵ0(π).
Corollary F.5. For all ϵ > 0, we have

˜J∗−˜J(π) ≤ϵ +
2˜cϵ
1 −γ
Γ(π)

ϵ
.

Proof. By Lemma F.3, we have

˜J∗−˜J(¯π) = Eµ,π [ ˜wπ(s) {˜v∗(s) −˜q∗(s, a)}] ≤ϵEµ [ ˜wπ(s)] ≤ϵ

for all ¯π ∈˜Π∗
ϵ. Thus, decomposing the suboptimality as

˜J∗−˜J(¯π) ≤˜J∗−˜J(¯π) +
 ˜J(π) −˜J(¯π)


and applying Corollary F.4 results in the desired result.

25


---Page Break---
F.5
Proof of Corollary 6.2

Proof. The celebrated uniform convergence theorem (Lemma B.1) with the boundedness
 ˆLz(θ; v, ξ)
 ≤Ball implies that, for all δ ∈(0, 1),

max
θ∈Θ,v∈V,ξ∈Ξ

 ˆL(θ; v, ξ) −E [L(θ; v, ξ)]
 ≤2Rn(H) + Ball

r

ln(2/δ)

2n
(40)

with probability 1 −δ. Now, by definition, ϵest(θ) is uniformly approximated with ˆϵest(θ) up to as
twice as the statistical error given by Eq. (40). Plugging this into Eq. (18), we obtain the desired
bound.

26


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The worst-case policy value is introduced in Section 4 and the sample complex-
ity bound is given by Corollary 6.3, wherein the realizability condition (εapp,Π(V, Θ, Ξ) = 0)
is the only major assumption. The improvements over the previous sample complexity
bounds are summarized in Table 1 and discussed in Section 2 and Appendix A.
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
Justification: The limitations are discussed in Section 7.
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

27


---Page Break---
Answer: [Yes]

Justification: All the assumptions are explicit in the main text. The complete proof is
presented in Appendix.

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

Answer: [NA]

Justification: No experimental result is presented.

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

28


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: No experimental result is presented.
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
Answer: [NA]
Justification: No experimental result is presented.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: No experimental result is presented.
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

29


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
Answer: [NA]
Justification: No experimental result is presented.
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
Justification: This paper is purely theoretical.
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
Justification: There is no immediate societal impact to be considered of the present work
since the results are purely theoretical.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

30


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

Justification: No dataset or model is disclosed.

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

31


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

32


---Page Break---
