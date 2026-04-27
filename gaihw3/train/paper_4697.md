Smoothed Online Classification can be Harder than
Batch Classification

Vinod Raman ∗
Department of Statistics
University of Michigan
Ann Arbor, MI 48104
vkraman@umich.edu

Unique Subedi ∗
Department of Statistics
University of Michigan
Ann Arbor, MI 48104
subedi@umich.edu

Ambuj Tewari
Department of Statistics
University of Michigan
Ann Arbor, MI 48104
tewaria@umich.edu

Abstract

We study online classification under smoothed adversaries. In this setting, at each
time point, the adversary draws an example from a distribution that has a bounded
density with respect to a fixed base measure, which is known apriori to the learner.
For binary classification and scalar-valued regression, previous works [Haghtalab
et al., 2020, Block et al., 2022] have shown that smoothed online learning is as
easy as learning in the iid batch setting under PAC model. However, we show that
smoothed online classification can be harder than the iid batch classification when
the label space is unbounded. In particular, we construct a hypothesis class that is
learnable in the iid batch setting under the PAC model but is not learnable under the
smoothed online model. Finally, we identify a condition that ensures that the PAC
learnability of a hypothesis class is sufficient for its smoothed online learnability.

1
Introduction

Classification is a canonical machine learning task where the goal is to classify examples in X into
one of the possible classes in Y. There are two common classification settings based on how the
data is available to the learner: batch and online. In the batch setting, the learner is provided with
a fixed set of training samples that are used to train a classifier, which is then deployed to make
predictions on new, real-world examples [Vapnik and Chervonenkis, 1971, Natarajan, 1989]. On
the other hand, data arrives sequentially in the online setting and predictions need to be made in
each round [Littlestone, 1987, Daniely et al., 2011]. The batch setting is often studied under the iid
assumption, whereas the stream can be fully adversarial in the online setting.

For binary classification (i.e. |Y| = 2), the batch learnability of a hypothesis class H ⊆YX under the
PAC model [Valiant, 1984] is characterized in terms of a combinatorial parameter called the Vapnik-
Chervonenkis (VC) dimension of H. On the other hand, the Littlestone dimension characterizes the
learnability of H under the adversarial online model [Littlestone, 1987, Ben-David et al., 2009]. As
Haghtalab et al. [2020] remark, the latter characterization is often interpreted as an impossibility
result because even simple classes like 1-dimensional thresholds have infinite Littlestone dimension.
This hardness result arises mainly because the adversary can deterministically choose hard examples,
even possibly adapting to the learner’s strategy. One way to circumvent this hardness result is to
consider a smoothed online model, where the adversary has to choose and draw examples from
sufficiently anti-concentrated distributions [Rakhlin et al., 2011, Haghtalab, 2018, Haghtalab et al.,
2020, Block et al., 2022]. This idea is inspired from the seminal work by Spielman and Teng [2004],
who showed that the smoothed analysis of the simplex method yields a polynomial time complexity
in the input size, instead of the known worst-case exponential time complexity.

∗Equal Contribution

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
In smoothed online classification, a learner plays a game with the adversary over T ∈N rounds.
Before the game begins, the adversary reveals a base measure µ over X and an anti-concentration
parameter σ > 0 to the learner. The distribution µ can be fairly non-informative such as uniform when
applicable for X. Then, in each round t ∈[T], the adversary picks a labeled sample (xt, yt) ∈X ×Y,
where xt is drawn from a σ-smooth distribution νt with respect to µ. That is, νt(E) ≤µ(E)/σ
for all measurable subsets E in X. The adversary then reveals xt to the learner, who makes a
prediction ˆyt ∈Y. Finally, the adversary reveals the true label yt ∈Y and the learner suffers the loss
1{ˆyt ̸= yt}. Given a hypothesis class H ⊆YX , the goal of the learner is to minimize the regret, the
difference between its cumulative loss and the best possible cumulative loss over hypotheses in H.

The smoothed online model interpolates between the iid setting (σ = 1) and the adversarial setting
(σ = 0). When |Y| = 2, Haghtalab [2018] and Haghtalab et al. [2020] showed that all VC classes
are learnable in the smoothed setting with the regret O(
p

T VC(H) log (T/σ)). Extending this
result to real-valued regression, where Y = [−1, 1], with the absolute-value loss, Block et al. [2022]
showed that the finiteness of the fat-shattering dimension [Bartlett et al., 1996, Alon et al., 1997] of
H ⊆YX is a sufficient condition for smoothed online learnability. Since the finiteness of VC and
fat-shattering dimensions are characterizations of learnability under the PAC model, these results
suggest that smoothed online learning may be as easy as batch learning.

In this work, we study smoothed online classification for arbitrary label spaces Y under oblivious
adversary – one that picks σ-smooth distirbutions ν1, .., νT , samples x1 ∼ν1, ..., xT ∼νT indepen-
dently, and finally picks the labels y1, ..., yT , all before the game begins. This obvlious model has
been studied in the past Wu et al. [2023], but is slightly different than the one studied by [Haghtalab,
2018, Haghtalab et al., 2020, Block et al., 2022] (see Section 2.1 for more details). For this model, we
show that smoothed online classification continues to be as easy as batch classification when |Y| < ∞.
However, when |Y| is not finite, we show that smoothed online classification can be harder than batch
classification. We note that there has been recent interest in studying multiclass learnability when
|Y| is unbounded [Brukhim et al., 2022, Hanneke et al., 2023, Pabbaraju, 2024]. Studying infinite
label spaces is important for understanding when one can establish learning guarantees independent
of the label size. This is quite a practical question as many modern machine learning paradigms
have massive label space, such as in face recognition and protein structure prediction, where the
dependence of label size in learning bounds would be undesirable.

Theorem (Informal) Let X = [0, 1]. Then, there exists H ⊆YX that is PAC learnable but not
learnable in the smoothed online setting with µ = Uniform(X) and σ = 1.

We also provide a quantitative version of this theorem that shows a regret lowerbound linear in T
even when |Y| < ∞but bigger than 2T log(T ). Note that this is tight up to a factor of log T because
we prove a sublinear upperbound as long as |Y| = 2o(T ) in Section 4. To prove these results, we
exploit the large size of the label space to construct a hypothesis class H ⊆YX such that for every
h ∈H, its output on a finite subset of X effectively reveals its identity. We then show that such a
hypothesis class has a sample compression scheme of size 1. Then, the result “compression implies
learning” by David et al. [2016] shows that H is PAC learnable. However, we show that even the
adversary that generates iid samples from Uniform(X) can construct a difficult stream for the learner.
Our construction is inspired by the hypothesis class from [Hanneke et al., 2024, Claim 5.4]. However,
a key challenge in our construction is the fact that the adversary does not have full control over
the sequence of examples the learner will observe (due to σ-smoothness) whereas the adversary in
Hanneke et al. [2024] can pick hard examples deterministically.

In light of this hardness result, we identify a sufficiency condition for smoothed online
learnability.
To do so, let B(µ, σ) denote the set of all σ-smooth distributions with re-
spect to µ.
For any x1, . . . , xn
∈X, let us define an empirical metric on dn on H as
dn(h1, h2) = n−1 Pn
i=1 1{h1(xi) ̸= h2(xi)}. Define N(ε, H, dn) to be the covering number of
H under metric dn. Then, H is said to have uniformly bounded expected empirical metric entropy
(UBEME) if supn∈N supν1:n∈B(µ,σ) Ex1:n∼ν1:n [N(ε, H, dn)] < ∞for every fixed ε > 0. We show
that if H has the UBEME property, then it online learnable under smoothed adversaries.

Theorem (Informal) If supn∈N supν1:n∈B(µ,σ) Ex1:n∼ν1:n [N(ε, H, dn)] < ∞for every fixed ε, σ >
0, then H is smoothed online learnable under the base measure µ.

2


---Page Break---
When |Y| = 2, Haussler’s packing lemma implies that N(ε, H, dn) ≤
 
41 ε−1VC(H) [Haussler,
1995]. That is, every class with finite VC satisfies UBEME, and thus our sufficiency condition
recovers the result from Haghtalab [2018] on the smoothed learnability of VC classes. For |Y| < ∞,
we generalize the packing lemma to show that N(ε, H, dn) ≤( 22 |Y|

ε
)G(H), where G(H) is the graph
dimension of H. This inequality shows that PAC learnability of H is sufficient for its smoothed
online learnability when |Y| < ∞.

A key contribution of our sufficiency result is going beyond VC and Graph dimension and giving
the weaker sufficiency condition than the worst-case empirical entropy. Indeed, in Section 4, we
show that our sufficiency condition still provides meaningful upperbounds even when the VC and
Graph dimensions are infinite. To prove our sufficiency result, we show that UBEME implies
the bounded metric entropy of H with respect to the base measure µ. That is, for dµ(h1, h2) =
Px∼µ[h1(x) ̸= h2(x)], we have N(ε, H, dµ) < ∞for every fixed ε > 0. Then, we use algorithmic
ideas from Haghtalab [2018] that involve running multiplicative weights using the cover of H under
dµ. Unfortunately, when |Y| is unbounded, the sufficiency condition (the UBEME of H) is not
necessary. This is demonstrated by constant functions H = {x 7→a : a ∈N} that is easy to learn but
N(ε, H, d1) = ∞.

Given our separation and sufficiency results, it is natural to ask for a characterization of learnability
for smoothed online classification. Any meaningful characterization must be a joint property of both
H and µ. This is because choosing µ to be a Dirac distribution will make any H, even the set of
all measurable functions from X to Y, trivially learnable. Since the most natural joint complexity
measure of H and µ is N(ε, H, dµ), one might ask whether N(ε, H, dµ) < ∞for every ε > 0
is necessary and sufficient for smoothed online learnability of H under µ. Surprisingly, we show
that this condition is neither necessary nor sufficient (see Theorem 4.3). These results highlight the
difficulty of characterizing learnability in the smoothed setting.

2
Preliminaries

Let X denote the instance space and Y denote the label space. We assume that (X, Σ) is a measurable
space and let Π(X) denote the set of all probability measures on (X, Σ). Let H ⊆YX denote an
arbitrary hypothesis class consisting of predictors h : X →Y. For any T ∈N, we use the notation
z1:T to denote the sequence {zt}T
t=1. Finally, we let [N] := {1, 2, . . . , N}.

2.1
Smoothed Online Learning

In the smoothed online model, an adversary plays a sequential game with the learner over T rounds.
Before the game begins, the adversary reveals a base measure µ ∈Π(X) and a scalar σ > 0 to
the learner. As mentioned before, µ can be non-informative measures such as uniform if X is
totally-bounded. Then, in each round t ∈[T], an adversary picks a labeled sample (xt, yt) ∈X × Y,
where xt is drawn from a distribution νt ∈Π(X) that satisfies νt(E) ≤µ(E)

σ
for every E ∈Σ. The
adversary then reveals xt to the learner A. Using all the past examples (x1, y1), . . . , (xt−1, yt−1),
the learner then makes a potentially randomized prediction A(xt). The adversary then reveals the
true label yt ∈Y and the learner suffers the loss 1{A(xt) ̸= yt}. Given a hypothesis class H ⊆YX ,
the goal of the learner is to output predictions A(xt) that minimizes the regret, which is the difference
between its cumulative loss and the best possible cumulative loss over hypotheses in H. To formally
define the regret, let B(µ, σ) := {ν ∈Π(X) : ν(E) ≤µ(E)/σ
∀E ∈Σ} denote the set of all
σ-smooth distributions on X with respect to µ. Given H ⊆YX , the worst-case expected regret of an
algorithm A is defined as

Rµ,σ
A (T, H) :=
sup
ν1,...,νT ∈B(µ,σ)
E
x1:T ∼ν1:T

"

sup
y1:T

 T
X

t=1
E
A [1{A(xt) ̸= yt}] −inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

!#

.

Note that as σ →0, the set B(µ, σ) contains all Dirac distributions on X. This amounts to replacing
supν1,...,νT ∈B(µ,σ) Ex1:T ∼ν1:T [·] operator in the definition of regret above by supx1:T , which yields
the expected regret of A in the fully adversarial model under an oblivious adversary. Thus, our
adversary is a generalization of the online oblivious adversary for the smoothed setting. Given
this definition of expected regret, we adopt the minimax perspective to define the learnability of a
hypothesis class.

3


---Page Break---
Definition 1 (Smoothed Online Learnability). The class H ⊆YX is learnable in the smoothed online
setting if and only if for every σ > 0 and µ ∈Π(X), we have
inf
A Rµ,σ
A (T, H) = o(T).

Our worst-case expected regret is defined with respect to an oblivious adversary that picks the
entire stream (x1, y1), . . . , (xT , yT ) before the game begins. Moreover, the sequence of distributions
ν1, . . . , νT has to be chosen upfront before the sampling step (x1, . . . , xT ) ∼(ν1, . . . , νT ). That is,
the distribution νt cannot depend on the realization of previous instances x1, . . . , xt−1. This ensures
that the instances x1, . . . , xT are independent random variables. One can also consider an oblivious
adversary, where νt can depend on the past instances (x1, . . . , xt−1) sampled from (ν1, . . . , νt−1).
Since the primary contribution of this work is the hardness result in Section 3, we only focus on
the case where ν1, . . . , νT are chosen upfront. As the first adversary is a special case of the second
adversary, our hardness result also applies for the second adversary. The fully general setting of
adaptive adversaries where νt can depend on the entire history of the game up to time point t −1 has
been studied extensively in [Haghtalab et al., 2020, Block et al., 2022], where the dependence among
x1, . . . , xT is handled through coupling.

There are three natural choices for how yt may be selected by an oblivious adversary. In the first
choice, yt may depend only on the xt ∼νt. In the second choice, yt may depend on prefix on x1 ∼
ν1, ..., xt ∼νt. Finally, in the last choice, yt may depend on the entire sample x1 ∼ν1, ..., xT ∼νT .
The first choice is considered in Haghtalab [2018]and the second by Haghtalab et al. [2020], Block
et al. [2022]. In this work, we focus on the third choice, which has been considered by Wu et al.
[2023]. This choice is natural because σ-smoothness is just a property of the instances and should
not impact how the labels are selected. Since the third choice is the strongest, our sufficiency result in
Section 4 holds for the first two choices. However, establishing the separation result for the first two
choices remains an open question.

2.2
PAC Learning and Sample Compression Schemes

In contrast to existing work, we establish a separation between smoothed online learnability and
batch learnability. The notion of batch learnability we consider is PAC learnability, a canonical model
in statistical learning theory. See Appendix A for a complete definition. To prove the agnostic PAC
learnability of hypothesis classes, we use the relationship between learnability and the existence of
sample compression schemes.

A compression scheme (κ, ρ) consists of a compression function κ and a reconstruction func-
tion ρ. The compression function κ : ∪i≥1(X × Y)i →∪i≥1(X × Y)i maps a sample S =
{(x1, y1), . . . , (xn, yn)} to a subsample S′ ⊆S. The reconstruction function ρ : ∪i≥1(X × Y)i →
YX takes S′ as input and outputs a function f ∈YX . We define the size of the compression scheme
(κ, ρ) on a sample S to be |S′|, where κ(S) = S′. We let the quantity k(n) denote the maximum size
of the compression scheme on all samples S such that |S| = n. A hypothesis class H ⊆YX has a
compression scheme (κ, ρ) of size k(n) if for every sample S = {(x1, h(x1)), . . . , (xn, h(xn))} for
some h ∈H, we have f = ρ(κ(S)) such that f(xi) = h(xi) for all i ∈[n].

The compression scheme in David et al. [2016] is slightly more general as their compression function
κ can output (S′, b) where b is a finite bitstring. However, the restricted notion of a compression
scheme without b is sufficient for our purpose. The following Theorem shows that the existence of
sample compression schemes for H implies agnostic PAC learnability of H.
Theorem 2.1 (Compression
=⇒
Learnability [David et al., 2016]). Let (κ, ρ) be a sample
compression scheme for H ⊆YX of size k(n) and define fS = ρ(κ(S)) for any S ∈(X × Y)n.
Then, for every D on X × Y and n ∈N such that k(n) ≤n/2, with probability at least 1 −δ over
S ∈Dn, we have

E
(x,y)∼D[1{fS(x) ̸= y}] ≤inf
h∈H
E
(x,y)∼D[1{h(x) ̸= y}] + 100

s

k(n) log
n
k(n) + k(n) + log 1

δ
n
.

2.3
Covering Numbers, Metric Entropy, and Complexity Measures

In Section 4, we provide conditions for which a hypothesis class H is online learnable under smoothed
adversaries. While sufficient conditions for learnability are typically established via combinatorial

4


---Page Break---
dimensions, our sufficient conditions will be in terms of covering/packing numbers of H using a
distance metric that depends on the base measure µ. This discrepancy with existing literature is due
to a simple observation: any parameter of H alone cannot characterize smoothed online classification.
Indeed, if one takes the base measure µ to be a Dirac measure, then every H ⊆YX is trivially online
learnable under a smoothed adversary. Accordingly, any meaningful characterization of smoothed
online classification must be in terms of both H and µ.

To start, we first define ε-covering numbers for generic metric spaces (G, d).
Definition 2 (Covering Number). Let (G, d) be a bounded metric space. A subset G′ ⊆G is an
ε-cover for G with respect to d if for every g ∈G, there exists an g′ ∈G′ such that d(g, g′) ≤ε. The
covering number of G at scale ε, denoted N(ε, G, d), is the smallest n ∈N such that there exists an
ε-cover of G with cardinality n. That is, N(ε, G, d) := inf{|G′| : G′ is an ε-cover for G}.

The metric entropy for (G, d) at scale ε > 0 is defined as log N(ε, G, d). In this paper, we consider the
metric space (H, dµ) where H ⊆YX is a hypothesis class and dµ(h1, h2) = Px∼µ [h1(x) ̸= h2(x)]
for some µ ∈Π(X). The key complexity measure in this work is

Cε,σ(H, µ) := sup
n∈N
sup
ν1:n∈B(µ,σ)
E
x1:n∼ν1:n [N(ε, H, dˆµn)] ,

where ˆµn denotes the empirical measure over x1:n. At a high-level, Cε,σ(H, µ) measures the com-
plexity of H in terms of its average empirical covering number, where the average is taken over
processes from B(µ, σ). Using Cε,σ(H, µ), we define the property of uniformly bounded empirical
metric entropy.

Definition 3 (Uniformly Bounded Empirical Metric Entropy). A hypothesis class H ⊆YX has the
Uniformly Bounded Empirical Metric Entropy (UBEME) property with respect to µ if Cε,σ(H, µ) <
∞for every ε, σ > 0.

In Theorem 4.1, we show that H is online learnable under smoothed adversaries if H enjoys the
UBEME property with respect to the base measure µ.

3
PAC Learnability is Not Sufficient for Smoothed Online Learnability

In Section 4, we show that the PAC learnability of H is sufficient for its smoothed online learnability
when |Y| < ∞. Here, we show that this is not the case when |Y| is unbounded by constructing a PAC
learnable hypothesis class that is not smoothed online learnable. In fact, we prove a stronger result.

Theorem 3.1. There exists a hypothesis class H ⊆YX such that following holds:

(i) H has a compression scheme of size 1.

(ii) For µ = Uniform(X) and σ = 1, we have infA Rµ,σ
A (T, H) ≥T

2 .

The part (i) of Theorem 3.1, together with Theorem 2.1, shows that H is agnostic PAC learnable with

error rate ε(δ, n) = O
q

log n + log(1/δ)

n


. On the other hand, based on Definition 1, part (ii) shows

that H is not smoothed online learnable. Together, we infer the following corollary.
Corollary 3.2 (Agnostic PAC Learnability ⇏Smoothed Online Learnability). There exists H ⊆
YX such that H is agnostic PAC learnable but not learnable in the smoothed setting under µ =
Uniform(X) and σ = 1.

When |Y| < ∞, the existence of a O(1)-size compression schemes and agnostic PAC learnability are
equivalent [David et al., 2016]. Thus, there is no qualitative difference between Theorem 3.1 and
Corollary 3.2. However, when |Y| = ∞, a recent work has shown that the multiclass PAC learnability
does not imply the existence of O(1)-size compression schemes [Pabbaraju, 2024]. Thus, Theorem
3.1 is a qualitatively stronger result than Corollary 3.2. Moreover, our proof of Theorem 3.1 below
provides an explicit PAC learner for H.

Proof. (of Theorem 3.1) Let X = [0, 1]. Given a bitstring θ = (θ1, . . . , θn) ∈{0, 1}n, define
θ≤t := (θ1, . . . , θt) and θ<t := (θ1, . . . , θt−1) for any t ∈{1, . . . , n}. Fix an n ∈N and ordered

5


---Page Break---
sequence (x1, x2, . . . , xn) ∈X n such that xi ̸= xj for all i ̸= j. For every θ ∈{0, 1}n, define

hθ
(x1,...,xn)(x) :=
((x1, . . . , xn), θ≤t)
if ∃t ∈[n] such that x = xt
((x1, . . . , xn), ⋆)
otherwise

Let On := {(x1, x2, . . . , xn) ∈X n : xi ̸= xj} be the set of all ordered sequences of length n with
distinct elements. Then, we define our hypothesis class to be

H :=
[

n∈N

[

(x1,...,xn)∈On

[

θ∈{0,1}n

n
hθ
(x1,...,xn)
o
.

Here, the label space is Y := ∪h∈H{image(h)}. For any y ∈Y, let us define y[1] and y[2] to be
the first and the second entry of the tuple y respectively. Note that y[1] ∈Om for some m ∈N and
y[2] ∈{0, 1}t for some t ≤m.

Proof of (i). We now define a compression scheme (κ, ρ) of size 1 for H.

• Define a compression function κ : ∪i≥1(X × Y)i →X × Y as follows. Given any
realizable sample S = {(x1, y1), . . . , (xn, yn)} of size n ∈N, the function κ outputs
κ(S) = {(x1, y1)} if yi[2] = ⋆for all i ∈[n]. On the other hand, if there exists a yi such
that yi[2] ∈{0, 1}t for some t ∈N, then κ(S) = {(xℓ, yℓ)}. Here, ℓ∈[n] is the index such
that yℓ[2] is the longest binary string among all i ∈[n] for which yi[2] ̸= ⋆.

• Define a reconstruction function ρ : X × Y →H as follows. Given an output of the
compression function {(x, y)}, the reconstruction function outputs ρ({(x, y)}) = h0
y[1] ∈H
if y[2] = ⋆. Here, 0 is all 0’s bitstring of length equal to that of the tuple y[1]. On the other
hand, if y[2] ∈{0, 1}t for some t ≤|y[1]|, then output ρ({(x, y)}) = hθ
y[1] ∈H, where θ
is an arbitrary bitstring of length y[1] such that θ≤t = y[2].

Next, we show that (κ, ρ) is a valid compression scheme for H. Let S = {(x1, y1), . . . , (xn, yn)} ∈
(X × Y)n denote any sample of size n that is realizable by H. We want to show that f = ρ(κ(S))
satisfies f(xi) = yi for all i ∈[n]. Since S is realizable by H, there exists m ∈N, (z1, . . . , zm) ∈
X m such that zi ̸= zj for all i, j ∈[m] and β ∈{0, 1}m such that hβ
(z1,...,zm)(xi) = yi
for all i ∈
[n]. By definition of H, for all i ∈[n], we have yi[1] = (z1, . . . , zm) and yi[2] ∈{⋆, β≤t} for some
t ≤m. Given a realizable sample S, there are two cases to consider: (a) yi[2] = ⋆for all i ∈[n] and
(b) there exists i ∈[n] such that yi[2] ̸= ⋆.

If we are in case (a), then we know that xi /∈{z1, . . . , zm} for all i ∈[n]. Moreover, we have
κ(S) = {(x1, y1)} where y1[1] = (z1, . . . , zm) and y1[2] = ⋆. By definition of the reconstruction
function, ρ({(x1, y1)}) = h0
(z1,...,zm). Since xi /∈{z1, . . . , zm}, by definition of h0
(z1,...,zm), we
have h0
(z1,...,zm)(xi) =
 
(z1, . . . , zm), ⋆

= yi for all i ∈[n]. Thus, (κ, ρ) is a valid compression
scheme for H in this case.

Suppose (b) is true. Define IS := {i ∈[n] : yi[2] ̸= ⋆}. Since hβ
(z1,...,zm) is consistent with the
sample S, we must have yi[2] = β≤|yi[2]| for each i ∈IS. Here, |yi[2]| is the length of bitstring yi[2].
Let ℓ∈IS such that |yℓ[2]| ≥|yi[2]| for all i ∈IS. By definition of the compression function κ,
we have κ(S) = {(xℓ, yℓ)}, where yℓ[1] = (z1, . . . , zm) and yℓ[2] = β≤|yℓ[2]|. Let θ ∈{0, 1}m be
the completion of yℓ[2] such that the reconstruction function returns ρ({(xℓ, yℓ)}) = hθ
(z1,...,zm). By
definition of ρ, we have β≤t = θ≤t for t = |yℓ[2]|. To complete our proof, it suffices to show that
hθ
(z1,...,zm)(xi) = yi for all i ∈[n]. There are two cases to consider: i ∈IS and i /∈IS. When i /∈IS,
we have xi /∈{z1, . . . , zm} and thus hθ
(z1,...,zm)(xi) = ((z1, . . . , zm), ⋆) = yi. As for the index
i ∈IS, we must have that xi ∈{z1, . . . , zm}. Otherwise, yi[2] would be equal to ⋆, contradicting
the fact that i ∈IS. In fact, by definition of hβ
(z1,...,zm), we have xi = z|yi[2]|. This implies
that, for all i ∈IS, we have hθ
(z1,...,zm)(xi) = hθ
(z1,...,zm)(z|yi[2]|) = ((z1, . . . , zm), θ≤|yi[2]|) =
((z1, . . . , zm), β≤|yi[2]|) = yi. Here, we use that |yi[2]| ≤|yℓ[2]| for all i ∈IS and the fact that
θ≤t = β≤t for all t ≤|yℓ[2]|. Therefore, we have shown that hθ
(z1,...,zm)(xi) = yi for all i ∈[n].

Proof of (ii). Let µ = Uniform([0, 1]). We now specify the stream {(xt, yt)}T
t=1 to be observed by
the learner. For the instances, we take x1, . . . , xT ∼µ to be iid samples from µ. Note that x1, . . . , xT

6


---Page Break---
are distinct with probability 1. Moreover, as all the instances are drawn from the same distribution µ,
this adversary is σ-smooth for σ = 1. To specify y1, . . . , yT , we first draw θ ∼Uniform({0, 1}T )
and define yt = ((x1, . . . , xT ), θ≤t) for all t ∈[T]. Given distinct x1, . . . , xT , for any algorithm A,
we first show that

E
θ∼Uniform({0,1}T )

 T
X

t=1
E
A [1{A(xt) ̸= yt}] −inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

!

≥T

2 .
(1)

The probabilistic method implies the existence of a θ ∈{0, 1}T such that the claimed bound of T/2
holds. This subsequently implies that, for a distinct x1, . . . , xT , we have

sup
y1:T

 T
X

t=1
E
A [1{A(xt) ̸= yt}] −inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

!

≥T

2 .

Finally, using the fact that x1, . . . , xT ∼µ are distinct with probability 1, we obtain the bound

inf
A Rµ,σ
A (T, H) ≥
E
x1,...,xT ∼µ

"

sup
y1:T

 T
X

t=1
E
A [1{A(xt) ̸= yt}] −inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

!#

≥T

2 .

To complete our proof, it now suffices to prove Equation (1). Fixing distinct x1, . . . , xT , the
lowerbound on the expected cumulative loss of the algorithm A is

T
X

t=1
E
θ∼Unif({0,1}T )

h
E
A [1{A(xt) ̸= yt}]
i
=

T
X

t=1
E

E
θt


1{A(xt) ̸= ((x1, . . . , xT ), θ≤t)}

| A, θ<t



=

T
X

t=1
E
1

21

A(xt) ̸=
 
(x1:T ), (θ<t, 0)
	
+ 1

21

A(xt) ̸=
 
x1:T , (θ<t, 1)
	

≥

T
X

t=1

1
2 = T

2 .

At a high-level, we use the fact that θt is sampled uniformly at random from {0, 1} and is independent
of A as well as θi for all i ̸= t. Thus, on each round, the algorithm cannot do any better than
randomly guessing the value of θt. Next, we upperbound the expected loss of the best-fixed function
in hindsight. Given distinct x1, . . . , xT and θ ∈{0, 1}T , we can pick the hypothesis hθ
(x1,...,xT ). By
definition of this hypothesis, we have hθ
(x1,...,xT )(xt) = ((x1, . . . , xT ), θ≤t) = yt. Thus, for every
distinct x1, . . . , xT , we have

E
θ∼Unif({0,1}T )

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

≤
E
θ∼Unif({0,1}T )

" T
X

t=1
1{hθ
(x1,...,xT )(xt) ̸= yt}

#

= 0.

Finally, equation (1) follows upon combining the lowerbound on the cumulative loss of A and the
upperbound on the cumulative loss of the optimal hypothesis in hindsight.

To prove the qualitative separation between PAC and smoothed online learnability in Theorem 3.1, we
required |Y| to be unbounded. The following theorem, proved in Appendix C, shows the quantitative
dependence of the regret on |Y| when it is bounded.

Theorem 3.3. For every K ∈N, there exists H ⊆YX with |Y| = K such that H has a compression
scheme of 1, but infA Rµ,σ
A (T, H) ≥
log K
24 log log K for µ = Uniform(X) and σ = 1.

Theorem 3.3 shows that one can get quantitative separation whenever K ≥2T log T .

4
A Sufficient Condition for Smoothed Online Classification

In this section, we provide a sufficient condition for smoothed online classification. Our main result
provides a quantitative upperbound on the expected regret in terms of Cε,σ(H, µ).

7


---Page Break---
Theorem 4.1. For every H ⊆YX , µ ∈Π(X) and σ > 0, we have that

inf
A Rµ,σ
A (T, H) ≤6 inf
ε>0

(
εT

σ +
q

T log(Cε2,σ(H, µ))

)

.

Theorem 4.1 shows that as long as H satisfies the UBEME condition with respect to µ, it is online
learnable under a smoothed adversary. As a corollary, we also establish the following sufficient
condition in terms of the Graph dimension, a combinatorial dimension characterizing PAC learnability
when |Y| < ∞(see Appendix A for a complete definition) [Natarajan, 1989].
Corollary 4.2. For every H ⊆YX , µ ∈Π(X) and σ > 0, we have that

inf
A Rµ,σ
A (T, H) ≤6 inf
ε>0

(
εT

σ +

r

T G(H) log
41|Y|

ε2

)

≤12

r

T G(H) log
41 T |Y|

σ2


,

where G(H) denotes the Graph dimension of H.

Corollary 4.2, whose proof is in Appendix E, shows that PAC learnability of H is sufficient for
smoothed online classification whenever |Y| < ∞. When |Y| = 2, this bounds, up to a constant
factor, recovers that from Haghtalab [2018]. We now proceed with the proof of Theorem 4.1.

Proof. (of Theorem 4.1) Let ν1, . . . , νT ∈B(µ, σ) denote the sequence of σ-smooth distributions
picked by the adversary. Fix a ε > 0. Then, by Lemma B.2, we have that N(ε, H, dµ) ≤C ε

2 ,σ(H, µ).
Let H′ ⊂H denote an ε-cover with respect to dµ of size at most C ε

2 ,σ(H, µ). Let A denote the
online learner that runs the Randomized Exponential Weights Algorithm (REWA) on the the data
stream (x1, y1), ..., (xT , yT ) using H′ as its set of experts. By the guarantees of REWA,

E
A

" T
X

t=1
1{A(xt) ̸= yt}

#

≤inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= yt} +
p

2T log(|H′|)

≤inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= yt} +
q

2T log(C ε

2 ,σ(H, µ))

≤inf
h∈H

T
X

t=1
1{h(xt) ̸= yt} + sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)} +
q

2T log(C ε

2 ,σ(H, µ))

where the expectation is only taken with respect to the randomness of the REWA and the last
inequality follows by the triangle inequality. Taking an outer expectation with respect to the process
x1:T ∼ν1:T , we have E
h PT
t=1 1{A(xt) ̸= yt}
i
is at most

E
x1:T ∼ν1:T

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

+
E
x1:T ∼ν1:T

"

sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

+
q

2T log(C ε

2 ,σ(H, µ)).

It remains to bound E
h
suph∈H infh′∈H′ PT
t=1 1{h′(xt) ̸= h(xt)}
i
. We provide a sketch of the

proof here and defer the full details to Appendix D. Consider the class G = {x 7→1{h′(x) ̸= h(x)} :
h ∈H}, where h′ ∈H′ denotes the ε-cover of h with respect to dµ, and note that

E
x1:T ∼ν1:T

"

sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

≤
E
x1:T ∼ν1:T

"

sup
g∈G

T
X

t=1
g(xt)

#

.

By standard symmetrization arguments, we get

E
x1:T ∼ν1:T

"

sup
g∈G

T
X

t=1
g(xt)

#

≤sup
g∈G
E
x1:T ∼ν1:T

" T
X

t=1
g(x′
t)

#

+ 2T
E
x1:T ∼ν1:T

h
ˆR(G, x1:T )
i

where ˆR(G, x1:T ) is the Rademacher complexity of G (see Appendix A). Note that G ⊆H∆H where
H∆H := {x 7→1{h1(x) ̸= h2(x)} : h1, h2 ∈H}, and thus ˆR(G, x1:T ) ≤ˆR(H∆H, x1:T ). Using

8


---Page Break---
the discretization-based upperbound (Lemma A.1) on the empirical Rademacher complexity and a
relation between the covering numbers of H and H∆H (Lemma B.3), we have

ˆR(H∆H, x1:T ) ≤ε+

r

2 log N(ε, H∆H, ρµT )

T
≤ε+

r

2 log N(ε2, H∆H, dµT )

T
≤ε+2

s

log N( ε2

2 , H, dµT )

T
.

Plugging in the upperbound on the Rademacher complexity and using the change of measure,
σ-smoothness, and the definition of H′ on the first term gives

E
x1:T ∼ν1:T

"

sup
g∈G

T
X

t=1
g(xt)

#

≤εT

σ + 2ε T + 4

s

T log

E
x1:T ∼ν1:T


N
ε2

2 , H, dµT


.

Using the definition of Cε,σ(H, µ) to get

E
x1:T ∼ν1:T

"

sup
h∈H
inf
h∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

≤εT

σ + 2ε T + 4
q

T log C ε2

2 ,σ(H, µ),

substituting into the regret bound for A, and doing some algebra completes the proof sketch.

Our upperbounds in terms of expected empirical covering numbers can be meaningful even when
VC and Graph dimension of H is infinity. As a simple example, let X = [0, 1], µ = Uniform(X),
and consider the binary hypothesis class H = {x 7→1{x ∈A} : A ⊂Q, |A| < ∞}. It’s not too
hard to see that VC(H) = ∞. On the other hand, Cε,σ(H, µ) = 1 since for every n ∈N, the sample
x1:n ∼µ does not lie in Q almost surely, and when x1:n /∈Q, dˆµn(h1, h2) = 0 for all h1, h2 ∈H.

To prove Theorem 4.1, we show that the UBEME implies a bound on the metric entropy N(ε, H, dµ).
It is natural ask whether the finiteness of N(ε, H, dµ) for every ε > 0 alone is necessary and sufficient
for smoothed online learnability. Unfortunately, it is neither sufficient nor necessary.
Theorem 4.3. Let X = [0, 1] and µ = Uniform(X). Then,

(i) There exists a H ⊆{0, 1}X such that N(ε, H, dµ) = 1 for every ε > 0 but Rµ,σ
A (T, H) ≥T

2
for every σ > 0.

(ii) There exists H ⊆NX such that N(ε, H, dµ) = ∞for every ε > 0 but infA Rµ,σ
A (T, H) =
O(
p

T log(T)) for every σ > 0.

Proof. We first prove (i). Consider the hypothesis class H = {x 7→1{x ∈S} : S ⊂X, |S| < ∞}.
Note that for every h1, h2 ∈H, we have that dµ(h1, h2) = Px∼µ [h1(x) ̸= h2(x)] = 0 since h1 and
h2 disagree on at most a finite number of points in X. Thus, for every ε > 0, H is trivially coverable
using exactly one hypothesis in H. To show that Rµ,σ
A (T, H) ≥T

2 for every σ > 0, consider the
adversary that picks νt = µ for all t ∈[T]. The process x1:T ∼ν1:T is then an iid draw from µ of
length T. Consider the data stream (x1, y1), ..., (xT , yT ) where x1:T ∼µT and yt ∼Unif({0, 1}) for
every t ∈[T]. Such a stream is realizable by H almost surely since the the sequence of instances x1:T
are all distinct with probability 1 and there can be at most a finite number of timepoints where yt = 1.
On the other hand, any learning algorithm A must make at least T/2 mistakes in expectation (with
respect to all sources of randomness), since the labels yt ∼Unif({0, 1}). Thus, by the probabilistic
method, for every learning algorithm A, there must exist a sequence of labels y1:T , such that A’s
expected regret is T/2.

We now prove (ii). Let H = {x 7→a : a ∈N} be the class of constant functions. Note that
for every h1, h2 ∈H, we have that dµ(h1, h2) = 1 since h1 and h2 disagree everywhere on X.
Thus, for every ε < 1, we have that N(ε, H, dµ) = ∞since |H| = ∞. On the other hand, the
Littlestone dimension of H is 1. Thus, by Theorem 4 from Hanneke et al. [2023], we get that
infA Rµ,σ
A (T, H) = O(√T log T) for every σ > 0.

5
Discussion

In this work, we show a separation between the learnability of H ⊆YX in the PAC setting and the
smoothed online setting when |Y| is unbounded. We also provide a sufficient condition for smoothed

9


---Page Break---
online learnability under any base measure µ. However, as noted in Section 4, our sufficient condition
is not necessary for the smoothed learnability of H. Thus, an important open question is to find a
condition that is both necessary and sufficient for smoothed online learnability. Traditionally, in
learning theory, learnability is characterized in terms of a combinatorial property of just the hypothesis
class H ⊆YX . However, the property of H alone cannot provide a characterization of learnability in
the smoothed online setting. Choosing µ to be a Dirac distribution will make any H, even the set of
all measurable functions from X to Y, trivially learnable. Thus, the characterization of learnability
must necessarily be a property of the tuple (H, µ). To that end, we pose the following question.

Given (H, µ), is there a complexity measure that characterizes the smoothed online learnability of H
with base measure µ?

Acknowledgments and Disclosure of Funding

VR acknowledges the support from the NSF Graduate Research Fellowship Program.

References

Noga Alon, Shai Ben-David, Nicolo Cesa-Bianchi, and David Haussler. Scale-sensitive dimensions,
uniform convergence, and learnability. Journal of the ACM (JACM), 44(4):615–631, 1997.

Martin Anthony and Peter L. Bartlett. Neural Network Learning: Theoretical Foundations. Cambridge
University Press, 1999. doi: 10.1017/CBO9780511624216.

Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and
structural results. Journal of Machine Learning Research, 3(Nov):463–482, 2002.

Peter L. Bartlett, Philip M. Long, and Robert C. Williamson. Fat-shattering and the learnability of
real-valued functions. Journal of Computer and System Sciences, 52(3):434–452, 1996.

Shai Ben-David, Dávid Pál, and Shai Shalev-Shwartz. Agnostic online learning. In Conference on
Learning Theory, volume 3, page 1, 2009.

Adam Block, Yuval Dagan, Noah Golowich, and Alexander Rakhlin. Smoothed online learning is as
easy as statistical learning. In Conference on Learning Theory, pages 1716–1786. PMLR, 2022.

Nataly Brukhim, Daniel Carmon, Irit Dinur, Shay Moran, and Amir Yehudayoff. A characterization
of multiclass learnability. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer
Science (FOCS), pages 943–955. IEEE, 2022.

Amit Daniely, Sivan Sabato, Shai Ben-David, and Shai Shalev-Shwartz. Multiclass learnability and
the erm principle. In Conference on Learning Theory, volume 19, pages 207–232. PMLR, 2011.

Ofir David, Shay Moran, and Amir Yehudayoff. On statistical learning via the lens of compression.
In Proceedings of the 30th International Conference on Neural Information Processing Systems,
pages 2792–2800, 2016.

Nika Haghtalab. Foundation of Machine Learning, by the People, for the People. PhD thesis, Carnegie
Mellon University, 2018.

Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. Smoothed analysis of online and differ-
entially private learning. Advances in Neural Information Processing Systems, 33:9203–9215,
2020.

Steve Hanneke, Shay Moran, Vinod Raman, Unique Subedi, and Ambuj Tewari. Multiclass online
learning and uniform convergence. Conference on Learning Theory, 2023.

Steve Hanneke, Shay Moran, and Jonathan Shafer. A trichotomy for transductive online learning.
Advances in Neural Information Processing Systems, 36, 2024.

David Haussler. Sphere packing numbers for subsets of the boolean n-cube with bounded vapnik-
chervonenkis dimension. Journal of Combinatorial Theory, Series A, 69(2):217–232, 1995.

10


---Page Break---
Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold
algorithm. Machine Learning, 2:285–318, 1987.

B. K. Natarajan. On learning sets and functions. Machine Learning, 4(1):67–97, 1989.

Chirag Pabbaraju. Multiclass learnability does not imply sample compression. In Proceedings of The
35th International Conference on Algorithmic Learning Theory, 2024.

Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Online learning: Stochastic, constrained,
and smoothed adversaries. Advances in neural information processing systems, 24, 2011.

Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to
Algorithms. Cambridge University Press, USA, 2014.

Daniel A Spielman and Shang-Hua Teng. Smoothed analysis of algorithms: Why the simplex
algorithm usually takes polynomial time. Journal of the ACM (JACM), 51(3):385–463, 2004.

Leslie G. Valiant. A theory of the learnable. In Symposium on the Theory of Computing, 1984.

Vladimir Naumovich Vapnik and Aleksei Yakovlevich Chervonenkis. On uniform convergence of the
frequencies of events to their probabilities. Teoriya Veroyatnostei i ee Primeneniya, 16(2):264–279,
1971.

Changlong Wu, Mohsen Heidari, Ananth Grama, and Wojciech Szpankowski. Expected worst case
regret via stochastic sequential covering. Transactions on Machine Learning Research, 2023.

A
PAC Learnability, Combinatorial Dimensions, Complexity Measures

We begin by defining the agnostic PAC framework, a canonical learning model in the batch setting.

Definition 4 (Agnostic PAC Learnability). A hypothesis class H ⊆YX is agnostic PAC learnable if
there exists a function m : (0, 1)2 →N and a learning algorithm A : ∪i≥1(X × Y)i →YX with
the following property: for every ε, δ ∈(0, 1) and for every distribution D on X × Y, A running on
n ≥m(ε, δ) i.i.d. samples from D outputs a predictor AS such that with probability at least 1 −δ
over S ∼Dn,
E
(x,y)∼D [1{AS(x) ̸= y}] ≤inf
h∈H
E
(x,y)∼D [1{h(x) ̸= y}] + ε.

The VC and Graph dimension characterize PAC learnability when |Y| = 2 and |Y| < ∞respectively.

Definition 5 (Vapnik-Chervonenkis (VC) Dimension). A set S = {x1, . . . , xd} ⊂X is shattered by
a binary function class H ⊆{0, 1}X if for every τ ∈{0, 1}d, there exists a hypothesis hτ ∈H such
that for all i ∈[d], we have hτ(xi) = τi. The VC dimension of H, denoted VC(H), is the size of the
largest shattered set S ⊆X. If H can shatter arbitrarily large sets, we say that VC(H) = ∞.

Definition 6 (Graph Dimension). For any multiclass hypothesis class H ⊆YX , let ℓ◦H :=
{(x, y) 7→1{h(x) ̸= y} : h ∈H} ⊆{0, 1}(X×Y) denotes its loss class. The Graph dimension of H
is defined as G(H) := VC(ℓ◦H).

Our proof of this result relies on bounding the Rademacher complexity, a canonical complexity
measure used to establish generalization bounds in the batch setting [Bartlett and Mendelson, 2002].

Definition 7 (Empirical Rademacher Complexity). Let F ⊆RX and x1, ..., xn ∈X n. The empirical
Rademacher complexity of F with respect to x1:n is defined as

ˆR(F, x1:n) = 1

n E
τ

"

sup
f∈F

 n
X

i=1
τif(xi)

!#

where τ1, ..., τn are independent Rademacher random variables.

The following upperbound on the empirical Rademacher complexity follows by a simple application
of Massart’s lemma [Shalev-Shwartz and Ben-David, 2014, Lemma 28.6].

11


---Page Break---
Lemma A.1 (Discretization Bound). For every F ⊆RX and x1, ..., xn ∈X n, we have that

ˆR(F, x1:n) ≤inf
ε>0

(

ε +

 

sup
f∈F

r

E
ˆµn [f 2]

!r

2 log N(ε, F, ρˆµn)

n

)

where ˆµn denotes the empirical measure on the sample x1:n and ρµ is the distance metric defined as

ρˆµn(f1, f2) :=
r

E
x∼ˆµn [(f1(x) −f2(x))2].

Finally, we define the packing number for generic metric spaces.
Definition 8 (Packing Number). Let (G, d) be a bounded metric space. A subset G′ ⊆G is an
ε-packing with respect to d if for every gi, gj ∈G′ we have that d(gi, gj) > ε. The packing number
of G at scale ε, denoted M(ε, G, d), is the largest n ∈N such that there exists an ε-packing of G
with cardinality n. That is, M(ε, G, d) := sup{|G′| : G′ is an ε-packing for G}.

B
Helper Lemmas

Lemma B.1 (Covering-Packing duality [Anthony and Bartlett, 1999]). For any metric space (G, d)
and ε > 0, we have that

M(2ε, G, d) ≤N(ε, G, d) ≤M(ε, G, d).

Using the Covering-Packing duality, we prove the following technical Lemma.
Lemma B.2 (UBEME =⇒Bounded Metric Entropy with respect to µ). For any µ ∈Π(X),
hypothesis class H ⊆YX , and ε > 0, we have that

N(ε, H, dµ) ≤sup
m∈N
E
x1:m∼µm

h
N
ε

2, H, dˆµm
i
.

Proof. Fix ε > 0. By the Covering-Packing duality, we have that

sup
m∈N
E
x1:m∼µm [M(ε, H, dˆµm)] ≤sup
m∈N
E
x1:m∼µm

h
N
ε

2, H, dˆµm
i
.

Thus, it suffices to show that N(ε, H, dµ) ≤supm∈N Ex1:m∼µm [M(ε, H, dˆµm)]. Suppose, for the
sake of contradiction, we have that N(ε, H, dµ) > supm∈N Ex1:m∼µm [M(ε, H, dˆµm)]. Then by the
Covering-Packing duality, we have that M(ε, H, dµ) > supm∈N Ex1:m∼µm [M(ε, H, dˆµm)] =: c.
By the definition of ε-packing, we can find n > c hypothesis h1, ..., hn ∈H and δ > 0 such that
dµ(hi, hj) > ε + δ for every i ̸= j.

Fix m ∈N and consider a sample x1:m ∼µm. Let Sij = Pm
t=1 1{hi(xt) ̸= hj(xt)} denote
the random variable counting the number of samples on which hi and hj differ. Note that Sij ∼
Binom(m, dµ(hi, hj)). Let Em be the event that Sij > ε m for all i < j. Then,

Px1:m∼µm
h
Em
i
= 1 −Px1:m∼µm
h
∃i < j such that Sij ≤ε m
i

≥1 −
X

i<j
Px1:m∼µm
h
Sij ≤ε m
i

≥1 −
X

i<j
exp
n
−2m (dµ(hi, hj) −ε)2o

≥1 −
X

i<j
exp{−2m δ2}

≥1 −n2 exp{−2m δ2},

12


---Page Break---
where the second inequality follows by Hoeffding’s inequality. Moreover, under event Em, we have
that M(ε, H, dˆµm) ≥n, where dˆµm is the empirical measure on x1:m. Finally, note that

sup
m∈N
E
x1:m∼µm [M(ε, H, dˆµm)] ≥sup
m∈N
E
x1:m∼µm [M(ε, H, dˆµm)|Em] Px1:m∼µm
h
Em
i

≥n sup
m∈N
Px1:m∼µm
h
Em
i

≥n sup
m∈N


1 −n2 exp{−2m δ2}


= n.

Since n > c and c = supm∈N Ex1:m∼µm [M(ε, H, dˆµm)], we arrive at a contradiction.

Lemma B.3 (Covering Number of Symmetric Differences). For any H ⊆YX , ε > 0, and sequence
x1, ..., xn ∈X n, we have that

N(ε, H∆H, ˆµn) ≤

N
ε

2, H, ˆµn
 2

where ˆµn is the empirical measure on x1:n and H∆H :=
n
x 7→1{h1(x) ̸= h2(x)} : h1, h2 ∈H
o
.

Proof. Fix ε > 0. Let H′ be an ε-cover for H with respect to dˆµn. It suffices to show that H′∆H′ is
an 2ε-cover for H∆H with respect to dˆµn. To see this, pick a g ∈H∆H. Then by definition, we can
decompose g(x) = 1{h1(x) ̸= h2(x)} for some h1, h2 ∈H. Let h′
1, h′
2 ∈H′ be the elements that
cover h1, h2 respectively. Then, observe that we have

1
n

n
X

i=1
1{1{h′
1(xi) ̸= h′
2(xi)} ̸= 1{h1(xi) ̸= h2(xi)}} ≤1

n

 n
X

i=1
1{h1(xi) ̸= h′
1(xi)} +

n
X

i=1
1{h2(xi) ̸= h′
2(xi)}


≤2ε.

Thus, x 7→1{h′
1(x) ̸= h′
2(x)} is 2ε-close to x 7→1{h1(x) ̸= h2(xi)}. Since x 7→1{h′
1(x) ̸=
h′
2(x)} ∈H′∆H′ and h1, h2 ∈H were chosen arbitrarily, it follows that H′∆H′ is a 2ε-cover
for H∆H with respect to dˆµn. Finally, note that |H′∆H′| ≤|H′|2. Since ε > 0 is arbitrary, this
completes the proof.

C
Proof of Theorem 3.3

The proof here is similar to the proof of Theorem 3.1. Thus, we only provide a high-level sketch of
the arguments here.

Proof. Fix m ∈N, and take X = {1, 2, . . . , m2}. For every ordered sequence (x1, x2, . . . , xm) ∈
X m such that xi ̸= xj for all i ̸= j and a bitstring θ ∈{0, 1}m, define a hypothesis

hθ
(x1,...,xm)(x) :=
((x1, . . . , xm), θ≤t)
if ∃t ∈[m] such that x = xt
((x1, . . . , xm), ⋆)
otherwise

Let Om := {(x1, x2, . . . , xm) ∈X m : xi ̸= xj} be the set of all ordered sequences of length m
with distinct elements. Then, we define our hypothesis class to be

H :=
[

(x1,...,xm)∈Om

[

θ∈{0,1}m

n
hθ
(x1,...,xm)
o
.

Here, the label space is Y := ∪h∈H{image(h)}. Thus, the size of the label space is

|Y| =
m2

m


m! 2m =
m2!
(m2 −m)!2m = m2(m2−1) . . . (m2−(m−1)) 2m ≤(m2)m2m ≤m3m,

13


---Page Break---
when m ≥2. This implies that

m ≥1

3
log |Y|
log log |Y|.

Note that the Proof of (i) in Theorem 3.1 can be used verbatim to show that H has a compression

scheme of size 1. Thus, H is PAC learnable with error rate O
q

log n + log(1/δ)

n


.

Therefore, we now focus on establishing regret lowerbound for H. Let µ = Uniform({1, 2, . . . , m2}).
Let m ≤T. We now specify the stream {(xt, yt)}T
t=1 to be observed by the learner. For the instances,
we take x1, . . . , xm ∼µ to be iid samples from µ. Note that x1, . . . , xm are distinct with probability

≥

1 −m

m2

m
=

1 −1

m

m
≥1

4
∀m ≥2.

Moreover, as all the instances are drawn from the same distribution µ, this adversary is σ-
smooth for σ = 1. To specify y1, . . . , ym, we first draw θ ∼Uniform({0, 1}m) and define
yt = ((x1, . . . , xm), θ≤t) for all t ∈[m]. Conditioned on the fact that x1, . . . , xm are distinct,
there will be a hypothesis class h∗
θ ∈H such that h∗
θ(xt) = yt for all t ∈[m]. For m ≤t ≤T,
sample xt ∼µ and define yt = h∗
θ(xt). Note that for any x1, . . . , xm, we have

E
θ∼Uniform({0,1}m)

"
T
X

t=1
E
A [1{A(xt) ̸= yt}]

#

≥
E
θ∼Uniform({0,1}m)

" m
X

t=1
E
A [1{A(xt) ̸= yt}]

#

≥m

2 .

Here, we use the fact that bitstrings θt are sampled uniformly randomly, so no algorithm can do better
than randomly guessing. Moreover, conditioned on the event that x1, . . . , xm are distinct, we have

E
θ∼Uniform({0,1}m)

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

≤
E
θ∼Uniform({0,1}m)

" T
X

t=1
1{h∗
θ(xt) ̸= yt}

#

= 0.

Since x1, . . . , xm are distinct with probability at least 1/8, we obtain

inf
A Rµ,σ
A (T, H) ≥m

8 ≥1

24
log |Y|
log log |Y|.

D
Proof of Theorem 4.1

Proof. Let ν1, . . . , νT ∈B(µ, σ) denote the sequence of σ-smooth distributions picked by the
adversary. Fix a ε > 0. Then, by Lemma B.2, we have that N(ε, H, dµ) ≤C ε

2 ,σ(H, µ). Let
H′ ⊂H denote an ε-cover with respect to dµ of size at most C ε

2 ,σ(H, µ). Let A denote the online
learner that runs the Randomized Exponential Weights Algorithm (REWA) on the data stream
(x1, y1), ..., (xT , yT ) using H′ as its set of experts. By the guarantees of REWA,

E
A

" T
X

t=1
1{A(xt) ̸= yt}

#

≤inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= yt} +
p

2T log(|H′|)

≤inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= yt} +
q

2T log(C ε

2 ,σ(H, µ))

≤inf
h∈H

T
X

t=1
1{h(xt) ̸= yt} + sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)} +
q

2T log(C ε

2 ,σ(H, µ))

where the expectation is only taken with respect to the randomness of the MWA and the last inequality
follows by the triangle inequality. Taking an outer expectation with respect to the process x1:T ∼ν1:T ,

14


---Page Break---
we get that

E
h
T
X

t=1
1{A(xt) ̸= yt}
i

≤E

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

+ E

"

sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

+
q

2T log(C ε

2 ,σ(H, µ)).

It remains to bound E
h
suph∈H infh′∈H′ PT
t=1 1{h′(xt) ̸= h(xt)}
i
. To do so, consider the class

G = {x 7→1{h′(x) ̸= h(x)} : h ∈H}, where h′ ∈H′ denotes the ε-cover with respect to dµ of h,
and note that

E

"

sup
h∈H
inf
h′∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

≤E

"

sup
g∈G

T
X

t=1
g(xt)

#

.

By standard symmetrization arguments, we get that

E
x1:T ∼ν1:T

"

sup
g∈G

 T
X

t=1
g(xt) −Ex′
1:T ∼ν1:T

" T
X

t=1
g(x′
t)

# !#

≤
E
x1:T ,x′
1:T ∼ν1:T

"

sup
g∈G

T
X

t=1


g(xt) −g(x′
t)
#

=
E
x1:T ,x′
1:T ∼ν1:T

"

Eθ1:T

"

sup
g∈G

T
X

t=1
θt

g(xt) −g(x′
t)
##

≤2
E
x1:T ∼ν1:T

"

E
θ1:T

"

sup
g∈G

T
X

t=1
θt g(xt)

##

≤2T
E
x1:T ∼ν1:T

h
ˆR(G, x1:T )
i

where θ1:T are independent Rademacher random variables. Note that G ⊆H∆H where H∆H :=
{x 7→1{h1(x) ̸= h2(x)} : h1, h2 ∈H}, and thus ˆR(G, x1:T ) ≤ˆR(H∆H, x1:T ). Using Lemma
A.1 we can pointwise upperbound

ˆR(H∆H, x1:T ) ≤ε+

r

2 log N(ε, H∆H, ρµT )

T
≤ε+

r

2 log N(ε2, H∆H, dµT )

T
≤ε+2

s

log N( ε2

2 , H, dµT )

T
.

The first inequality follows by taking ρµT (g1, g2) =
q

1
T
PT
t=1 1{g1(xt) ̸= g2(xt)} for any two
functions g1, g2 ∈H∆H. The second inequality follows from the fact that N(ε, H∆H, ρµT ) ≤
N(ε2, H∆H, dµT ). The last inequality follows after using Lemma B.3. Plugging in the upperbound
on the Rademacher complexity, we get that

E
x1:T ∼ν1:T

"

sup
g∈G

T
X

t=1
g(xt)

#

≤sup
g∈G
E
x′
1:T ∼ν1:T

" T
X

t=1
g(x′
t)

#

+ 2ε T + 4
E
x1:T ∼ν1:T

"s

T log N
ε2

2 , H, dµT

#

≤sup
g∈G

T
X

t=1
E
x′
t∼νt [g(x′
t)] + 2ε T + 4

s

T log
E
x1:T ∼ν1:T


N
ε2

2 , H, dµT



≤sup
g∈G

T
X

t=1
E
x′
t∼µ

g(x′
t)
σ


+ 2ε T + 4

s

T log
E
x1:T ∼ν1:T


N
ε2

2 , H, dµT



≤εT

σ + 2ε T + 4

s

T log
E
x1:T ∼ν1:T


N
ε2

2 , H, dµT


,

15


---Page Break---
where the third and fourth inequality follow by change of measure, σ-smoothness, and the definition
H′ respectively. By the definition of Cε,σ(H, µ), we get that

E

"

sup
h∈H
inf
h∈H′

T
X

t=1
1{h′(xt) ̸= h(xt)}

#

≤εT

σ + 2ε T + 4
q

T log C ε2

2 ,σ(H, µ),

implying that

E

" T
X

t=1
1{A(xt) ̸= yt}

#

≤E

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

+ εT

σ + 2ε T + 4
q

T log C ε2

2 ,σ(H, µ) +
q

2T log C ε

2 ,σ(H, µ)

≤E

"

inf
h∈H

T
X

t=1
1{h(xt) ̸= yt}

#

+ 3εT

σ
+ 6
q

T log C ε2

2 ,σ(H, µ).

Since ν1, . . . , νT ∈B(µ, σ) and ε > 0 were chosen arbitrarily, we have that

Rµ,σ
A (T, H) ≤inf
ε>0

3εT

σ
+ 6
q

T log C ε2

2 ,σ(H, µ)

≤6 inf
ε>0

εT

σ +
q

T log Cε2,σ(H, µ)

.

E
Proof of Corollary 4.2

The following lemma from Haghtalab [2018], which uses the seminal packing lemma by Haussler
[1995], will be useful.

Lemma E.1 (Haussler [1995], Haghtalab [2018]). For any H ⊆{0, 1}X and µ ∈Π(X), we have
that

N(ε, H, dµ) ≤
41

ε

VC(H)
.

Lemma E.1 together with Definition 6 implies that for any multiclass hypothesis class H ⊆YX , we
have that

sup
˜µ∈Π(X×Y)
N(ε, ℓ◦H, d˜µ) ≤
41

ε

G(H)
,

where ℓ◦H := {(x, y) 7→1{h(x) ̸= y} : h ∈H} ⊆{0, 1}(X×Y) denotes the loss class of H. We
now begin the proof of Corollary 4.2.

Proof. (of Corollary 4.2) It suffices to show that

Cε,σ(H, µ) ≤
sup
˜µ∈Π(X×Y)
N
 2ε

|Y|, ℓ◦H, d˜µ


,

for every ε, σ > 0. Fix ε, σ > 0. Recall that

Cε,σ(H, µ) := sup
n∈N
sup
ν1:n∈B(µ,σ)
E
x1:n∼ν1:n [N(ε, H, dˆµn)] ,

where ˆµn denotes the empirical measure over x1:n. We will actually show something stronger, that is

16


---Page Break---
sup
n∈N
sup
x1:n∈X n N(ε, H, dˆµn) ≤
sup
˜µ∈Π(X×Y)
N
 2ε

|Y|, ℓ◦H, d˜µ


.
(2)

Fix n ∈N and let c := sup˜µ∈Π(X×Y) N( 2ε

|Y|, ℓ◦H, d˜µ). To see why Inequality (2) is true, consider
a sequence x1:n ∈X n and let ˆµn denote the empirical measure on x1:n. Let ˜µn ∈Π(X × Y) be
the joint measure over X × Y defined procedurally by first sampling x ∈ˆµn and then sampling the
label y ∼Uniform(Y). Since ˜µn ∈Π(X × Y), by definition of c, there exists a subset H′ ⊆H of
size at most c such that ℓ◦H′ is an 2ε

|Y|-cover for ℓ◦H with respect to ˜µn. It suffices to show that
H′ is an ε-cover for H with respect to ˆµn. Fix a h ∈H and let h′ be the element in H′ such that
d˜µn(ℓ◦h, ℓ◦h′) ≤2ε

|Y|. Then, by definition, we have that

2ε
|Y| ≥d˜µn(ℓ◦h, ℓ◦h′)

=
E
(x,y)∼˜µn [1{ℓ◦h(x, y) ̸= ℓ◦h′(x, y)}]

= 1

n

n
X

i=1

1
|Y|

|Y|
X

j=1
1{1{h(xi) ̸= j} ̸= 1{h′(xi) ̸= j}}

=
2
n|Y|

n
X

i=1
1{h(xi) ̸= h′(xi)}

=
2
|Y|dˆµn(h, h′).

Therefore h′ is ε-close to h with respect to dˆµn. Since h was chosen arbitrarily, this is true for all
h ∈H. Accordingly, H′ is an ε-cover for H with respect to dˆµn, implying that N(ε, H, dˆµn) ≤c.
Since n and x1:n was also chosen arbitrarily, we have that supn∈N supx1:n∈X n N(ε, H, dˆµn) ≤c,
completing the proof.

Finally, upperbound 6 infε>0

(

εT

σ +
r

T G(H) log

41|Y|

ε2
)

≤12
r

T G(H) log

41 T |Y|

σ2

fol-

lows by picking ε =
σ
√

T .

17


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Our abstract and introduction state that (i) we show that smoothed online
classification can be harder than batch classification and (ii) we provide a sufficiency
condition for smoothed online classification. The first claim is proven in Section 3 of the
paper and the sufficiency condition is proven in Section 4.
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
Justification: In the introduction, we discuss that our work does not provide a complete
characterization of learnability in the smoothed setting. We reiterate this limitation in the
Discussion section of the paper, highlighting difficulties for complete characterization in
the smoothed online model. Finally, we pose an open question regarding the complete
characterization.
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

18


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: Although our Theorems do not have any assumptions, our results hold under
formal learning models, namely the PAC model and the smoothed online model. Both
are standard frameworks in learning theory, and we define them in the Preliminaries and
Appendix. Moreover, we provide detailed proofs of all our theoretical claims.

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

Justification: The paper does not include experiments.

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

19


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

Answer: [NA]

Justification: The paper does not include experiments requiring code.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.

20


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
Answer: [NA]
Justification: The paper does not include experiments.
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
Justification: Our work conforms with NeurIPS Code of Ethics.
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
Justification: Our work is theoretical and contributes to our understanding of machine learn-
ing algorithms. Beyond theoretical insights that can be used to develop better algorithms,
our work does not have direct societal impacts.

21


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

Justification: The paper poses no risks.

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

22


---Page Break---
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

23


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

24


---Page Break---
