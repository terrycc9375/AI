On the Efficiency of ERM in Feature Learning

Ayoub El Hanchi
University of Toronto &
Vector Institute
aelhan@cs.toronto.edu

Chris J. Maddison
University of Toronto &
Vector Institute
cmaddis@cs.toronto.edu

Murat A. Erdogdu
University of Toronto &
Vector Institute
erdogdu@cs.toronto.edu

Abstract

Given a collection of feature maps indexed by a set T , we study the performance
of empirical risk minimization (ERM) on regression problems with square loss
over the union of the linear classes induced by these feature maps. This setup
aims at capturing the simplest instance of feature learning, where the model is
expected to jointly learn from the data an appropriate feature map and a linear
predictor. We start by studying the asymptotic quantiles of the excess risk of
sequences of empirical risk minimizers. Remarkably, we show that when the set
T is not too large and when there is a unique optimal feature map, these quantiles
coincide, up to a factor of two, with those of the excess risk of the oracle procedure,
which knows a priori this optimal feature map and deterministically outputs an
empirical risk minimizer from the associated optimal linear class. We complement
this asymptotic result with a non-asymptotic analysis that quantifies the decaying
effect of the global complexity of the set T on the excess risk of ERM, and relates
it to the size of the sublevel sets of the suboptimality of the feature maps. As an
application of our results, we obtain new guarantees on the performance of the best
subset selection procedure in sparse linear regression under general assumptions.

1
Introduction

A central idea in modern machine learning is that of data-driven feature learning. Specifically,
instead of performing linear prediction on top of handcrafted features, the current dominant paradigm
suggests to use models that select useful features for linear prediction in a data-dependent way [e.g.
KSH12; LBH15; He+16; Vas+17]. Of course, by putting the burden of picking a feature map on the
model and data, we should expect that the resulting learning problem will require more samples to be
solved. But just how many more samples do we need to learn such feature-learning-based models?

In this paper, we investigate this question in a general setting. We study the performance of empirical
risk minimization (ERM) on regression tasks with square loss and over model classes induced by
arbitrary collections of features maps. More precisely, let X be the random input taking value in a set
X, and let (ϕt)t∈T , ϕt : X →Rd, be a collection of feature maps indexed by a set T . For a given
regression task and i.i.d. samples, our aim is to understand the performance of ERM over the class of
predictors ∪t∈T

x 7→⟨w, ϕt(x)⟩| w ∈Rd	
as a function of the sample size, the distribution of the
data, and relevant properties of the collection of feature maps (ϕt)t∈T .

Classical uniform-convergence-based analyses would suggest that the performance of ERM in this
setting is determined by the size of the model class, appropriately measured. The main message
of this paper is that in this case, this is wrong in a strong sense. Specifically, we prove an upper
bound on the excess risk of ERM on this problem whose dependence on the size of the model class
decays monotonically with the sample size, and eventually depends only on the size of the model
class induced by the collection of optimal feature maps, which is typically much smaller.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Formal setup. We briefly formalize our problem here. Let X be the random input taking value
in a set X, and let (ϕt)t∈T , ϕt : X →Rd, be a collection of feature maps indexed by a set T .1
Let Y ∈R be the output random variable, jointly distributed with the input X. Our goal is to
learn to predict the output Y given the input X as well as possible within the class of predictors

x 7→⟨w, ϕt(x)⟩| (t, w) ∈T × Rd	
. We evaluate the quality of a single prediction ˆy given the
ground truth y through the loss function ℓ(ˆy, y) := (ˆy −y)2/2, and the overall quality of a predictor
(t, w) ∈T × Rd through its risk

R(t, w) := E[ℓ(⟨w, ϕt(X)⟩, Y )],
R∗:=
inf
(t,w)∈T ×Rd R(t, w).

We assume that we have access to n i.i.d. samples (Xi, Yi)n
i=1 with the same distribution as (X, Y ),
and perform empirical risk minimization

(ˆtn, ˆwn) ∈
argmin
(t,w)∈T ×Rd Rn(t, w)
where
Rn(t, w) := n−1
n
X

i=1
ℓ(⟨w, ϕt(Xi)⟩, Yi).

Our goal is to characterize the excess risk E(ˆtn, ˆwn) := R(ˆtn, ˆwn) −R∗.

Related work. The study of upper bounds on the excess risk of ERM in a general setting is a classical
topic. It was initiated by Vapnik and Chervonenkis [VC74] who established a link between the excess
risk of ERM and the uniform convergence of the underlying empirical process. More recently, and
fuelled by the development of Talagrand’s concentration inequality [Tal96] and its refinements [e.g.
BLM00; Bou02], a literature emerged that provided more fine-grained control of the excess risk
of ERM [e.g. BBM05; Kol06; BM06]. A key idea emerging from this line of work is localization.
This concept, and in particular the iterative localization method of Koltchinskii [Kol06], plays an
important role in our development. We refer the reader to the books [Kol11; Wai19], as well as the
recent articles [LRS15; KRV22] for more on this idea.

Focusing on the task of regression with square loss, upper bounds on the excess risk of ERM are
available for many classes of predictors, including finite [e.g. Aud07; JRT08; LM09], linear [e.g.
LM16b; Oli16; Mou22], and convex classes [e.g. LM16a; Men14; LRS15]. A key development
in this area over the last decade has been the realization that such bounds can be obtained under
much weaker assumptions than previously thought, owing to the fact that only one-sided control
of a certain empirical process is needed, and which can be obtained under very weak assumptions
[Men14; KM15; Oli16]. The line of work most closely related to ours is the one on random-design
linear regression [AC11; HKZ12; Oli16; LM16b; Sau18; Mou22; EE23], and we view our work as
an extension of this literature. We review these results in more detail in Section 2.

Finally, and on a more conceptual level, our work is related to the recent effort to understand the effect
of feature learning on the performance of neural networks [e.g. Bac17; Gho+20; Ba+22]. Beyond this
conceptual connection however, our work is quite distinct from this literature. Among other things,
our setting is more general since we consider arbitrary features maps. In the same vein, it is worth
mentioning the line of work on multiple kernel learning [e.g. Lan+04; GA11; SD16], although we
are not aware of results from this literature that are directly relevant to our setup.

Challenges. Our class of predictors is somewhat unstructured (e.g. it is in general non-convex),
so that off-the-shelf results from the above literature are not directly applicable. Nevertheless, the
analysis of the performance of ERM on linear classes provides a good starting point as we review in
Section 2. Compared to that setting however, we are faced with two additional challenges. First, we
need to control an additional source of error arising from the fact that ERM might select a suboptimal
feature map. Second, we are lead to study the suprema of certain T -indexed empirical processes,
which in the linear setting reduce to single random variables that are easily dealt with.

Organization. The rest of the paper is organized as follows. In Section 2, we review known results
on the excess risk of linear regression under square loss. In Section 3, we state our main results that
hold for the excess risk of ERM for general index sets T . In Section 4, we specialize our analysis to
the case where the index set T is finite, obtain more explicit guarantees, and discuss their implications
on the sparse linear regression problem. We conclude in Section 5 with a brief discussion.

1We assume without loss of generality that if t, s ∈T with t ̸= s, then ϕt and ϕs induce different linear
classes of functions, i.e. there is no matrix A such that ϕt(x) = Aϕs(x) for all x ∈X.

2


---Page Break---
2
Background

The goal of this section is to provide more context for our results. We review known results on the
excess risk of ERM over linear classes, which corresponds in our setting to the special case where the
set T indexing the feature maps is a singleton. As such, to avoid introducing further notation, we use
the one from the previous section, while dropping the dependence on t whenever it occurs.

In the setting of linear regression with square loss, and when the sample covariance matrix of the
feature map is invertible, there is a unique empirical risk minimizer and its excess risk admits an
explicit expression. Specifically, define

Σ := E

ϕ(X)ϕ(X)T 
,
Σn := 1

n

n
X

i=1
ϕ(Xi)ϕ(Xi)T ,

and let w∗denote the unique minimizer of the risk R(w).2 Then, an elementary calculation shows
that when Σn is invertible, there is a unique empirical risk minimizer and it satisfies

ˆwn = w∗−Σ−1
n ∇Rn(w∗).
(1)

Furthermore, since the risk is a quadratic function of w whose gradient at w∗vanishes, replacing
R( ˆwn) by the equivalent exact second order Taylor expansion around w∗yields

R( ˆwn) −R(w∗) = 1

2∥ˆwn −w∗∥2
Σ = 1

2

Σ−1
n ∇Rn(w∗)
2
Σ.
(2)

While exact, this expression is not readily interpretable. For example, how fast does this excess risk
go to 0 as a function of the sample size? The following classical result from asymptotic statistics [e.g.
Whi82; LC06; Vaa98] makes this rate more explicit. To state it, we define

g(X, Y ) := ∇wℓ(⟨w∗, ϕ(X)⟩, Y ),
G := E

g(X, Y )g(X, Y )T 
.

Theorem 1. Assume that for all j ∈[d], E

ϕ2
j(X)

< ∞, E

Y 2
< ∞, and E[∥g(X, Y )∥2
Σ−1] < ∞.
Then, as n →∞,

n · E( ˆwn)
d→1

2 · ∥Z∥2
2,

where Z ∼N(0, Σ−1/2GΣ−1/2). In particular, for any δ ∈(0, 0.1),

lim
n→∞n · QE( ˆ
wn)(1 −δ) ≍E[∥g(X, Y )∥2
Σ−1] + 2λmax(Σ−1/2GΣ−1/2) log(1/δ),

where QX(p) := inf{x ∈R | P(X ≤x) ≥p} is the quantile function of a random variable X, and
where we write a ≍b to mean that there exists absolute constants C, c such that c · b ≤a ≤C · b. In
the above statement, they can be taken as C = 1 and c = 1/32.

We provide a proof in Appendix A for completeness. For our purposes, this theorem is most easily
interpreted as follows: for large enough n and small enough δ, if the excess risk of ERM is bounded
by some quantity with probability at least 1 −δ, then this quantity is, up to a constant, at least as
large as the right-hand side of the second displayed equation divided by n. While our primary interest
is in non-asymptotic bounds, this asymptotic result, by virtue of its exactness, provides us with
a benchmark against which such bounds can be compared. In particular, it identifies the quantity
E[∥g(X, Y )∥2
Σ−1] as an intrinsic parameter determining the excess risk of ERM on this problem.

For large enough n, Theorem 1 gives an interpretable expression for the excess risk. However, it
says nothing about how large n needs to be for this expression to be accurate. This motivates a
non-asymptotic analysis of the excess risk of ERM, which has been carried out numerous times in
recent years [e.g. Oli16; LM16b; EE23]. A goal of this literature has been to obtain upper bounds on
the excess risk of ERM that hold in probability under weak moment assumptions, building on the
observation that this is indeed possible [Men14]. The following theorem is comparable to the best
known result in this area. We leave the proof to Appendix B. To state it, we define

V := E

Σ−1/2ϕ(X)ϕ(X)T Σ−1/2 −I
2
,
L :=
sup
v∈Sd−1 E

⟨v, Σ−1/2ϕ(X)⟩2 −1
2
.

2Throughout, we assume without loss of generality that the support of the distribution of ϕ(X) is not
contained in any hyperplane, which implies the invertibility of Σ and the uniqueness of w∗[cf. Mou22].

3


---Page Break---
Theorem 2. Assume that for all j ∈[d], E

ϕ4
j(X)

< ∞, E

Y 2
< ∞, and E[∥g(X, Y )∥2
Σ−1] < ∞.
Let δ ∈(0, 1). If

n ≥(512λmax(V ) + 6) log(ed) + (128L + 11) log(2/δ),

then with probability at least 1 −δ,

E( ˆwn) ≤4 · (nδ)−1 · E[∥g(X, Y )∥2
Σ−1].

At a high-level, this result says that above a certain explicit minimal sample size, the asymptotic
expression of the excess risk of Theorem 1 is correct, up to a significantly worse dependence on δ.
The restriction on the sample size is almost the best one can hope for. To see why, note that to get
guarantees on the excess risk of any empirical risk minimizer, we need at least that Σn is invertible,
otherwise there exists an empirical risk minimizer arbitrarily far away from w∗. To get quantitative
guarantees, we need slightly more control in the form of a lower bound on λmin(Σ−1/2ΣnΣ−1/2).
We refer the reader to a more detailed discussion in [EME24, Section 5].

This result has two key qualities, which we aim to reproduce in our results. First, it is assumption-
lean, requiring nothing more than a fourth moment assumption on the coordinates of the feature
map compared to Theorem 1. Second, it recovers the right dependence on the intrinsic parameter
E[∥g(X, Y )∥2
Σ−1] identified in Theorem 1. A downside of this generality is the bad dependence on δ.
Without further assumptions, this cannot be improved; we refer the reader to the recent literature on
robust linear regression for more on this issue [e.g. LM19; LL20; EME24].

3
Main Results

In this section we state our main results. They are most easily seen as extensions of Theorems 1 and 2
for general index sets T . In Section 3.1, we study the asymptotics of the excess risk of ERM in our
setting, and in Section 3.2, we present a non-asymptotic upper bound on the excess risk.

To state our results, we require additional definitions and notation. We start with the population and
the sample covariance matrices

Σ(t) := E

ϕt(X)ϕt(X)T 
,
Σn(t) := n−1
n
X

i=1
ϕt(Xi)ϕt(Xi)T .

We define the following collection of minimizers,

w∗(t) := argmin
w∈Rd R(t, w),
T∗:= argmin
t∈T
R(t, w∗(t)),

the first is uniquely defined, while the second is set-valued in general. We define the gradient of the
loss at these minimizers and their corresponding covariance matrices

g(t, (X, Y )) := ∇wℓ(⟨w∗(t), ϕt(X)⟩, Y ),
G(t, s) := E

g(t, (X, Y ))g(s, (X, Y ))T 
.

Finally, we introduce the following processes which play a key role in our development

Λn(t) := √n·λmax(I −Σ−1/2(t)Σn(t)Σ−1/2(t)),
Gn(t) := √n·∥∇wRn(t, w∗(t))∥Σ−1(t), (3)

as well as, for t∗∈T∗and t ∈T \ T∗,

∆n(t, t∗) := √n ·

1 −Rn(t, w∗(t)) −Rn(t∗, w∗(t∗))

R(t, w∗(t)) −R∗


.
(4)

We note that the process (∆n(t, t∗))t∈T \T∗is an empirical process (see [VW96] for an introduction),
while (Λn(t))t∈T and (Gn(t))t∈T are partial suprema of empirical processes. In the sequel, we will
slightly abuse this terminology, and call all of these empirical processes, with the understanding
that they can be viewed as one with more indexing. We will further assume that these processes are
separable; see [BLM13, p.305-306] for a definition. This covers a wide range of applications, while
avoiding delicate measurability issues. The suprema of such separable processes, which is the only
way they enter our results, can be studied by taking the supremum over a countable dense subset of
the index set. Therefore, without loss of generality, we assume that T is countable.

4


---Page Break---
Finally, in line with the literature on the theory of empirical processes [VW96], we say that a
sequence of empirical processes is Glivenko–Cantelli if, when rescaled by n−1/2, the supremum of
their absolute value taken over their index set converges to zero in probability as n →∞. In other
words, the weak law of large numbers holds uniformly over the index set. Similarly, we say that a
sequence of empirical processes is Donsker if it converges in distribution to its limiting Gaussian
process.3 In other words, the central limit theorem holds uniformly over the index set.

3.1
Asymptotic result

Our first main result is an asymptotic characterization of the quantiles of the excess risk of any
sequence of empirical risk minimizers in our setting, which vastly generalizes that of Theorem 1.
Theorem 3. Assume that T∗̸= ∅and for some t∗∈T∗, assume that the empirical processes
(Λn(t))t∈T , (∆(t, t∗))t∈T \T∗and (Gn(t))t∈T are Glivenko-Cantelli. Then, for all ε > 0,

lim
n→∞P
 
R(ˆtn, w∗(ˆtn)) −R∗> ε

= 0.

Furthermore, if the sequence of processes (Gn(t))t∈T is Donsker, then for any δ ∈(0, 1),

1
2 · QZ−(1 −δ) ≤lim inf
n→∞n · QE(ˆtn, ˆ
wn)(1 −δ) ≤lim sup
n→∞n · QE(ˆtn, ˆ
wn)(1 −δ) ≤QZ+(1 −δ),

where Z−:= infs∈T∗∥Z(s)∥2
2, Z+ := sups∈T∗∥Z(s)∥2
2, and (Z(t))t∈T is a mean-zero Gaussian
process with covariance function E[Z(t)Z(s)T ] = Σ−1/2(t)G(t, s)Σ−1/2(s) for all t, s ∈T .

We note that, up to a factor of two in the upper bound on the asymptotic quantiles, Theorem 3
reduces to Theorem 1 when T is a singleton, with the exact same assumptions. We are not aware of
comparable results in the literature. The proof of Theorem 3 can be found in Appendix D.
Remark 1. For small δ, the upper bound admits the more interpretable expression

QZ+(1 −δ) ≍E

sup
s∈T∗
∥Z(s)∥2
2


+ 2 log(1/δ) sup
s∈T∗
λmax(Σ−1/2(s)G(s, s)Σ−1/2(s)).
(5)

Furthermore, if T∗is finite, the first term can be upper bounded as

E

max
s∈T∗∥Z(s)∥2
2


≤80 · (1 + log|T∗|) · max
s∈T∗E[∥g(s, (X, Y ))∥2
Σ−1(s)].
(6)

To see why Theorem 3 is surprising, let us first focus on the case where T∗has a unique element
t∗, so that Z+ = Z−
d= ∥Z∥2
2 where Z ∼N(0, Σ−1/2(t∗)G(t∗, t∗)Σ−1/2(t∗)). Now consider the
oracle procedure, which knows beforehand what the optimal feature map t∗is, and outputs t∗and
a minimizer of Rn(t∗, w). Theorem 3 says that, up to a factor of two, the asymptotic quantiles of
the excess risk of ERM, which needs to learn over the large class ∪t∈T

x 7→⟨w, ϕt(x)⟩| w ∈Rd	
,
coincide with those of the oracle procedure (by Theorem 1), which only needs to learn over the linear
class

x 7→⟨w, ϕt∗(x)⟩| w ∈Rd	
!

More generally, Theorem 3 establishes that asymptotically, any ERM picks a near-optimal feature
map with probability one. It furthers shows that the asymptotic quantiles of the excess risk of
any sequence of ERMs is controlled from above and below by those of the extrema of the limiting
Gaussian process of (Gn(t))t∈T on the set of optimal feature maps T∗. This is surprising, as it implies
that asymptotically, and outside of its role in determining whether the assumptions of Theorem 3
hold, the global complexity of the set T is irrelevant to the excess risk of ERM.

Finally, we note that the Glivenko-Cantelli and Donsker assumptions in Theorem 3 can equivalently
be viewed as restrictions on the size of T , for distribution and process dependent notions of size. We
refer the reader to the books [VW96; GN15] for more on this connection. With this observation, the
main takeaway from Theorem 3 can be stated as follows.

Asymptotically, if T is not too large, the excess risk of ERM depends, at worst, only on the complexity
of the set of optimal feature maps T∗, and is independent of the global complexity of T .

3See [VW96, Section 2.1] or the proof of Theorem 3 for a more precise definition.

5


---Page Break---
3.2
Non-asymptotic result

The result in Theorem 3 hints at a dramatic localization phenomenon, whereby the influence of
the size and complexity of the collection of feature maps (ϕt)t∈T on the excess risk of ERM
vanishes as n →∞under appropriate assumptions. The root of this localization phenomenon is
the first statement of Theorem 3: eventually, ERM picks near-optimal feature maps with probability
approaching one. For small enough sample sizes however, it is clear that ERM is likely to select
suboptimal feature maps, so that this localization phenomenon cannot hold uniformly over n. This
raises a host of questions: (i) How fast, as measured by the sample size, does ERM learn the optimal
feature map? (ii) What is the effect of this localization on the rate of decay of the excess risk of ERM
non-asymptotically? (iii) What properties of the feature maps (ϕt)t∈T influence these rates?

Our answers to these questions in this very general setting are formally expressed in Theorem 4
below. To state it, we define the following parameter

L := sup E
hX

t∈T
⟨vt, Σ−1/2(t)ϕt(X)⟩2 −1
2i
,

where the supremum is taken over vectors (vt)t∈T such that P
t∈T ∥vt∥2
2 = 1. For n ∈N and
δ ∈(0, 1), we define the set function Fn,δ, for any subset S ⊂T , by

Fn,δ(S) :=

t ∈T
 R(t, w∗(t)) −R∗≤2 · (nδ)−1 · E[sup
s∈S
G2
n(s)]

(7)

This map acts as a contraction as shown in the next lemma, whose proof is deferred to Appendix E.
For a function f, we use f k to denote f k(x) := f(f k−1(x)) with f 0(x) := x.
Lemma 1. Let n ∈N, δ ∈(0, 1), and assume that T∗̸= ∅. Then for all k ∈N ∪{0},

• F k+1
n,δ (T ) ⊆F k
n,δ(T ).

• If ∃n0, B such that E[supt∈T G2
n(t)] ≤B for all n ≥n0, then T

n≥1 F k
n,δ(T ) = T∗.

With these definitions, we now state the second main result of the paper. A proof is in Appendix F.
Theorem 4. Assume that T∗̸= ∅, E[Y 2] < ∞, ∀(t, j) ∈T × [d], E[ϕ2
t,j(X)] < ∞, and
E[∥g(t, (X, Y ))∥2
Σ−1(t)] < ∞. Let δ ∈(0, 1) and k ∈N. If, for some t∗∈T∗, n satisfies

n ≥64 E[sup
t∈T
Λn(t)] + (128L + 11) log(6/δ) + 6 · δ−2 · E[ sup
t∈T \T∗
∆n(t, t∗)],

then, with probability at least 1 −δ,
ˆtn ∈F k
n,δ/2k(T ) =: Sn,δ,k,

and
E(ˆtn, ˆwn) ≤24 · (nδ)−1 · E[ sup
s∈Sn,δ,k
G2
n(s)],

where the processes Λn, ∆n, and Gn are as in (3) and (4).

We make a few remarks before interpreting the content of the theorem. First, we note that when
the index set T is a singleton, the last term in the sample size restriction vanishes, while the first
matches the sample size restriction from Theorem 2 after an application of Lemma 3 below; further
taking k = 1 in Theorem 4 recovers the upper bound on the excess risk of Theorem 2 up to a constant
factor. Theorem 4 may therefore be viewed as a broad generalization of Theorem 2. Second, under
Assumption 1 below, and by the second item of Lemma 1, the upper bound on the excess risk in
Theorem 4 eventually matches the main term in the asymptotic bound of Theorem 3 as can be seen
from (5), in the same way that Theorem 2 achieves this when compared with Theorem 1. Finally,
the statement of Theorem 4 is very general, and in fact, too general for us to be able to interpret it
precisely. As such, we will discuss it in the context of the following assumption.
Assumption 1. There exists constants CΛ, C∆, and CG independent of the sample size, but possibly
dependent on the remaining parameters of the problem, such that for all n ∈N,

E[sup
t∈T
Λn(t)] ≤CΛ,
E

"

sup
t∈T \T∗
∆n(t, t∗)

#

≤C∆,
E[sup
t∈T
G2
n(t)] ≤CG,

where Λn, ∆n, and Gn are as in (3) and (4).

6


---Page Break---
These assumptions can be equivalently viewed as a restriction on the appropriately measured size of
the index set T [VW96; GN15], and are slightly stronger than the assumptions of Theorem 3. They
always hold for finite index sets, and we will derive in Section 4 explicit estimates of the constants in
Assumption 1 in terms of moments of the feature maps and target as well as the cardinality of T .

Let us now interpret the content of Theorem 4, which comes with a free parameter k, in the context
of Assumption 1. We fix k here, and discuss its choice below. First, recalling the definition of Fn,δ,
this result says that above a certain sample size, both the suboptimality of the feature map picked
by ERM and its excess risk decay at the fast rate n−1, answering the first question we raised at the
beginning of the section. Second, this result provides an upper bound on the excess risk of ERM that
depends on the index set T only through the size of shrinking subsets Sn,δ,k, which might be large
for small n, but which by Lemma 1 converge to the set of optimal feature maps T∗as n →∞. This
transparently shows the effect of the localization phenomenon on the rate of decay of the excess risk
of ERM, answering the second question we raised. Finally, looking at the definition of Sn,δ,k, this
result identifies the size of the sublevel sets of the suboptimality function R(t, w∗(t)) −R∗defined
over feature maps as a relevant property of the collection of feature maps (ϕt)t∈T that influences the
rate of convergence of the excess risk of ERM in this setting, answering the final question we raised.

Finally, let us turn to the choice of k. Practically, we select the one that minimizes the bound on the
excess risk. Looking at the first item of Lemma 1, this optimal k balances the following trade-off: on
the one hand, for small k, applications of Fn,δ/2k constrain the input set more severely, but only a
few iterations are performed; on the other hand, larger values of k allow more iterations, but at the
cost of more weakly constraining the input set per application.

Stepping back, there are two main takeaways from Theorem 4. Firstly, and on a conceptual level, it
shows that feature learning is easy when the suboptimality function R(t, w∗(t)) −R∗, defined over
the set of features maps, has small sublevel sets. Secondly, and on a technical level, it provides a
template which can be used to derive more explicit excess risk bounds on ERM given estimates on the
expected suprema of the relevant empirical processes. Deriving such accurate estimates for infinite T
is a highly non-trivial task, and cannot be done at the level of generality we have been operating at.
The case of finite T however is tractable in a general setting as we discuss in the next section.

4
Case study: Finite index sets

In this section, we focus on the case where the index set T is finite, and aim, among other things,
at establishing explicit estimates on the various expected suprema appearing in Theorem 4 in terms
of moments of the feature maps and of the target. This problem becomes tractable in the case of
finite T because, roughly speaking, a worst-case analysis still yields non-trivial upper bounds. This
is decidedly not the case when T is infinite, in which case these expected suprema can be infinite.

We start with a slight strengthening of Theorem 3, whose assumptions reduce to simple moments
conditions when T is finite. The straightforward proof can be found in Appendix H.
Corollary 1. Assume that T is finite, for all (t, j) ∈T × [d], E

ϕ2
t,j(X)

< ∞, E

Y 2
< ∞, and
for all t ∈T , E[∥g(t, (X, Y ))∥2
Σ−1(t)] < ∞. Then

lim
n→∞P
 ˆtn /∈T∗

= 0.

Furthermore, for any δ ∈(0, 1),

1
2 · QZ−(1 −δ) ≤lim inf
n→∞n · QE(ˆtn, ˆ
wn)(1 −δ) ≤lim sup
n→∞n · QE(ˆtn, ˆ
wn)(1 −δ) ≤1

2 · QZ+(1 −δ),

where Z−:= mins∈T∗∥Zs∥2
2, Z+ := maxs∈T∗∥Zs∥2
2, and the random vectors (Zt)t∈T are jointly
Gaussian with mean zero and covariance E[ZtZT
s ] = Σ−1/2(t)G(t, s)Σ−1/2(s) for all t, s ∈T . In
particular, if T∗= {t∗}, then

n · E(ˆtn, ˆwn)
d→1

2 · ∥Z∥2
2,

where Z ∼N(0, Σ−1/2(t∗)G(t∗, t∗)Σ−1/2(t∗)).

The conclusions of Corollary 1 differ from those of Theorem 3 in two aspects. First, the feature map
picked by ERM is guaranteed to be optimal rather than near-optimal with probability converging to

7


---Page Break---
one. Second, the upper bound on the asymptotic quantiles is improved by a factor of two, yielding
the exact distribution of the rescaled excess risk when T∗is a singleton.

Making Theorem 4 more explicit is a more laborious task. We recall here two known results that
allow us to accomplish this. We start with the following bounds on the expectation of the supremum
of a finitely-indexed empirical process, which we will later use to bound the suprema of the processes
(Gn(s))s∈S and (∆n(t, t∗))t∈T \T∗appearing in Theorem 4. A proof can be found in Appendix G.

Lemma 2. Let n, d ∈N, and let Z be a random element taking value in a set Z, and let (Zi)n
i=1 be
i.i.d. samples with the same distribution as Z. Let F be a finite collection of Rd-valued measurable
functions. Define

σ2(F) := max
f∈F E

∥f(Z) −E[f(Z)]∥2
2

,
rn(F) := E

max
(i,f)∈[n]×F∥f(Zi) −E[f(Z)]∥2
2

1/2
,

and let En(f) := √n · (n−1 Pn
i=1 f(Zi) −E[f(Z)]). Then, we have

1
2 · σ(F) + 1

4 · rn(F)
√n
≤E

max
f∈F ∥En(f)∥2
2

1/2
≤c(|F|) · σ(F) + c2(|F|) · rn(F)
√n ,

where c(m) := 5√1 + log m.

Lemma 2 allows us to compute the expected supremum of a finitely-indexed empirical process, up
to log factors in the size of the index set. It is known that these factors cannot be removed from the
upper bound nor added to the lower bound without more assumptions, we refer the reader to a related
discussion in [Tro16]. Finally, while the term rn(F) might grow with n, by bounding the maximum
with the sum, it grows at most as √n. In many applications however, the random vectors f(Z) are
bounded almost surely, so that rn(F) is of order one, which justifies our presentation choice.

The second result we recall is the expectation version of a one sided Matrix Bernstein inequality due
to Tropp [Tro15]. We use it below to bound the supremum of the process (Λn(t))t∈T appearing in
Theorem 4. We do not known of a matching non-asymptotic lower bound, but an asymptotic one is
known [EME24, Proposition 17]. Upper and lower bounds similar to those of Lemma 2 hold if one
considers the expected operator norm instead of only the maximum eigenvalue [Tro16, Section 7].
Lemma 3 ([Tro15], Theorem 6.6.1.). Let n, d ∈N and for each i ∈[n], let Zi ∈Rd×d be i.i.d.
positive semi-definite matrices with the same distribution as Z. Define

V := E
h
(E[Z] −Z)2i
,
Wn := √n ·

E[Z] −1

n

n
X

i=1
Zi

.

Then, we have

E[λmax(Wn)] ≤
p

2λmax(V ) log(ed) + λmax(E[Z]) log(ed)

3√n
.

Equipped with these estimates, we may now control the expected suprema of the empirical processes
appearing in Theorem 4. To apply Lemma 2, define the following classes, for S ⊂T and t∗∈T∗

G(S) :=
n
(x, y) 7→Σ−1/2(s)g(s, (x, y))
 s ∈S
o
,

D(t∗) :=

(x, y) 7→ℓ(⟨w∗(t), ϕt(x)⟩, y) −ℓ(⟨w∗(t∗), ϕt∗(x)⟩, y)

R(t, w∗(t)) −R∗

 t ∈T \ T∗


.

Applying Lemma 2 on G(S) bounds the expected supremum of the process (Gn(s))s∈S while
applying it on D(t∗) bounds that of (∆n(t, t∗))t∈T \T∗. To control the supremum of (Λn(t))t∈T , the
key idea is to notice that it can be expressed as the maximum eigenvalue of a block diagonal matrix
whose blocks are √n(I −Σ−1/2(t)Σn(t)Σ−1/2(t)). Looking at Lemma 3, the relevant parameter is
therefore a block diagonal matrix V with the following blocks

V (t) := E

Σ−1/2(t)ϕt(X)ϕt(X)T Σ−1/2(t) −I
2
.

As the bound in Lemma 3 depends only on the maximum eigenvalue of V , the ordering of the blocks
does not matter. Putting together these estimates, we arrive at a fully explicit version of Theorem 4.

8


---Page Break---
Corollary 2. Assume that T is finite and that for all (t, j) ∈T ×[d], E

ϕ4
t,j(X)

< ∞, E

Y 4
< ∞.
Let δ ∈(0, 1), k ∈[1 + |T \ T∗|], and c(·), σ2(·), rn(·) as in Lemma 2. If, for some t∗∈T∗,

n ≥(512λmax(V ) + 6) log(ed|T |) + (128L + 11) log(6/δ)

+ 24 · δ−1 · c(|T |)σ2(D(t∗)) + 10 · δ−1/2 · c2(|T |)rn(D(t∗)),
(8)

then, with probability at least 1 −δ

ˆtn ∈eF k
n,δ/2k(T ) =: eSn,δ,k,

and
E(ˆtn, ˆwn) ≤24 · (nδ)−1 · A( eSn,δ,k),
where, for S ⊂T ,

A(S) := c2(S) ·

σ(G(S)) + c(S) · rn(G(S))
√n
2
,

and eFn,δ(S) is the same as Fn,δ(S) defined in (7) but with A(S) replacing E[sups∈S G2
n(s)].

We make a few remarks about Corollary 2; a proof sketch is in Appendix I. The set function A(S)
controlling the contraction rate of the map eFn,δ as well as the excess risk, has a pleasantly simple
form. To first order, and ignoring constants, it is given by

(1 + log|S|) · max
s∈S E
h
∥g(s, (X, Y ))∥2
Σ−1(s)
i
.

As such, as n →∞and by Lemma 1, the upper bound on the excess risk in Corollary 2 matches the
main term in the asymptotic rate derived in Theorem 3, as can be seen from (6). As the sets eSn,δ,k
are shrinking with n, the above expression clearly shows the decaying effect of the global complexity
of T on the excess risk. Finally, we note that the restriction on k in Corollary 2 is there only because
after at most that many iterations, a fixed point is reached, and further iterations worsen the bound.
We conclude this section with an example of an application of our results.
Example 1 (Sparse linear regression). Consider the sparse linear regression problem, and in particular
the best subset selection (BSS) procedure [Mil02; HTF09]. This procedure corresponds to ERM over
the restricted linear class {x 7→⟨w, ϕ(x)⟩| ∥w∥0 ≤s} in the linear regression setup of Section 2,
where ∥w∥0 is the number of non-zero entries of w and s ∈[d] is a user-chosen sparsity level.

The problem of computing the BSS procedure has attracted a lot of attention recently. While NP-hard
and therefore difficult in the worst case [Nat95], Bertsimas et al. [BKM16] showed that it can be
tractable on practical instances of moderate size. Since then, a rich literature has emerged that devises
increasingly efficient methods [e.g. Hua+18; BP20; HMS22; Guy+24]. By comparison, the statistical
performance of the BSS procedure is not yet completely understood as we discuss below.

To see how the sparse linear regression problem fits in our feature learning setting, notice that
{x 7→⟨w, ϕ(x)⟩| ∥w∥0 ≤s} = {x 7→⟨v, ϕt(x)⟩| (t, v) ∈T × Rs} where T is the set of all sub-
sets of [d] of size s, and ϕt(x) := (ϕj1(x), ϕj2(x), . . . , ϕjs(x)) ∈Rs where (j1, j2, . . . , js) are the
elements of t in increasing order. As such, Corollaries 1 and 2 are immediately applicable and provide
general statements on the performance of an arbitrary BSS procedure. To simplify the discussion, we
assume for the rest of the example that there is a unique risk minimizer w∗satisfying ∥w∗∥0 = s.

On the recovery side, the first item of Corollary 1 guarantees that we asymptotically exactly recover
the support of w∗. Non-asymptotically, the first item of Corollary 2 shows that if n further satisfies

n > min
k∈[(
d
s)]

n
4k · (γδ)−1 · A( eF k−1
n,δ/2k(T ))
o
where
γ :=
min
t∈T \T∗{R(t, w∗(t)) −R∗},

then with probability at least 1−δ, the BSS procedure recovers the support of w∗. Equivalently, these
two statements say that for large enough n, the BSS procedure coincides with the oracle procedure
which knows the support of w∗a priori and outputs an ERM from the optimal linear class.

In practice however, the interesting regime is when n is only moderately large. Corollary 2 provides
our guarantee in this case, and as such, we turn our attention to the sample size restriction (8).
Typically, we expect the main restriction to come from the first term, which in this case is given
by λmax(V ) · s log(d/s), up to constants and lower order terms. This is because if an intercept is

9


---Page Break---
included, i.e. ϕ1(X) = 1, then λmax(V ) ≥s −1, so the first term scales as s2 log(d/s) at least,
while the remaining terms typically grow more slowly with s. As a concrete example, when ϕ(X)
is a Gaussian vector, λmax(V ) = s + 1, so in this case the estimate s2 log(d/s) is tight. Under this
sample size restriction, and if ε := Y −⟨w∗, ϕ(X)⟩satisfies E

ε2 | X

≤σ2, Corollary 2 upper
bounds the excess risk by (σ2s/n) · an for a sequence of decreasing distribution-dependent constants
an converging to one as n →∞, ignoring the dependence on δ and absolute constants.

The closest existing result in the literature we are aware of is due to Shen et al. [She+13], who arrived
at comparable conclusions but in a substantially different setting. In particular, their result was
obtained in the setting n, d →∞, with an implicit assumption on the distribution of ϕ(X) [She+13,
Equation 2], and dealt with the in-sample prediction risk instead of the excess risk. Another closely
related result is due to Raskutti et al. [RWY11] who showed that the minimax expected excess risk in
a well-specified fixed-design setting is, up to constants, σ2s log(d/s)/n; see also [Bac24, Chapter 8].
Our results show that for moderate n, in the random-design setting, and when focusing on a single
instance, the log(d/s) factor can be replaced with another factor that decays to one as n →∞.

Coming back to the sample size restriction discussion, we strongly suspect that the factor s log(d/s)
is suboptimal, but we are unsure what the correct dependence is, even under Gaussian ϕ(X). Indeed,
this factor comes from the logarithmic factor in Lemma 3, when applied to the block diagonal matrix
with blocks √n
 
I −Σ−1/2(t)Σn(t)Σ−1/2(t)

. One can improve this factor by instead using versions
of this inequality based on the intrinsic dimension [Tro15, Chapter 7]. However, this is also unlikely
to be tight. Roughly speaking, this is because such logarithmic factors are tight only when the
eigenvalues of the random matrix are near-independent. This is certainly not the case for the block
diagonal matrix we are considering, since its blocks are sample covariance matrices of sub-vectors of
the same random vector ϕ(X). Capturing this dependence is beyond our reach and likely requires
new tools; we refer the interested reader to the recent articles [vHan17; LvHY18; BBvH23].

5
Conclusion

Broadly speaking, there are two main conclusions one can draw from this work. Firstly, in the large
sample regime, and if the set of candidate feature maps is not too large under an appropriate measure
of size, asking a model to additionally pick a feature map on top of learning a linear predictor has a
negligible effect on the excess risk of ERM on regression problems with square loss. Secondly, for
moderate sample sizes, the magnitude of this effect depends on the appropriately measured size of
the sublevel sets of the suboptimality function t 7→R(t, w∗(t)) −R∗. Plainly, learning feature maps
is easy when only a small subset of them is good, as the bad ones can be quickly discarded.

The most tantalizing aspect of our results is their potential in explaining the experiments in [Zha+21].
It was shown there that complex neural networks trained by ERM were able to achieve good
performance despite being expressive enough to fit random labels. This is paradoxical if one assumes
that the performance of ERM is driven by the complexity of the model class. Our results refute this
assumption for a generic collection of feature-learning-based models. While there are many works
offering explanations for this apparent paradox (see e.g. [BMR21] for a survey), we are not aware of
one that shows the vanishing influence of the size of the model class on the excess risk as Theorems 3
and 4 show. Formally connecting our statements to these experiments is beyond what we achieved
here, yet, we believe that the new perspective we took might generate useful insights in this area.

We conclude by outlining a few limitations of our work. Firstly, we do not deal with the question
of how to solve the ERM problem. Our focus is on understanding its statistical performance, and
our setting is so general that such a question cannot be meaningfully tackled. Continuing on this last
point, while the generality of our results is desirable in some aspects, it is detrimental in others. As an
example, it would be desirable to specialize our results from Section 3 to specific infinite collections
of feature maps used in practice. Let us also mention that it is a priori unclear whether ERM is an
optimal procedure, in a minimax sense, for the model classes we consider; we suspect that recently
developed tools might be relevant to address this question [Mou22]. Finally, while we focused on the
case of regression with square loss, this was mostly done to simplify the presentation. Indeed, the
only property of the loss used in the proofs is the exactness of its second order Taylor expansion. This
is however not required if one can control the error term from above and below. It is known how to do
this for many loss functions [e.g. OB21; EE23], and most importantly for logistic regression [Bac10;
Bac14]. We have purposefully selected generic notation to make translating such arguments easier.

10


---Page Break---
Acknowledgments and Disclosure of Funding

Resources used in preparing this research were provided in part by the Province of Ontario, the
Government of Canada through CIFAR, and companies sponsoring the Vector Institute. CM acknowl-
edges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC),
RGPIN-2021-03445. MAE was partially supported by NSERC Grant [2019-06167], CIFAR AI
Chairs program, and CIFAR AI Catalyst grant.

References

[AC11]
J.-Y. Audibert and O. Catoni. “Robust Linear Least Squares Regression”. In: The Annals
of Statistics (2011). URL.
[Aud07]
J.-y. Audibert. “Progressive Mixture Rules Are Deviation Suboptimal”. In: Advances in
Neural Information Processing Systems. 2007. URL.
[Ba+22]
J. Ba, M. A. Erdogdu, T. Suzuki, Z. Wang, D. Wu, and G. Yang. “High-Dimensional
Asymptotics of Feature Learning: How One Gradient Step Improves the Representation”.
In: Advances in Neural Information Processing Systems (Dec. 6, 2022). URL.
[Bac10]
F. Bach. “Self-Concordant Analysis for Logistic Regression”. In: Electronic Journal of
Statistics (Jan. 2010). DOI: 10.1214/09-EJS521.
[Bac14]
F. Bach. “Adaptivity of Averaged Stochastic Gradient Descent to Local Strong Convexity
for Logistic Regression”. In: Journal of Machine Learning Research (2014). URL.
[Bac17]
F. Bach. “Breaking the Curse of Dimensionality with Convex Neural Networks”. In:
Journal of Machine Learning Research (2017). URL.
[Bac24]
F. Bach. Learning Theory from First Principles. Dec. 24, 2024.
[BBM05]
P. L. Bartlett, O. Bousquet, and S. Mendelson. “Local Rademacher Complexities”. In:
The Annals of Statistics (Aug. 2005). DOI: 10.1214/009053605000000282.
[BBvH23]
A. S. Bandeira, M. T. Boedihardjo, and R. van Handel. “Matrix Concentration In-
equalities and Free Probability”. In: Inventiones mathematicae (Oct. 1, 2023). DOI:
10.1007/s00222-023-01204-6.
[BKM16]
D. Bertsimas, A. King, and R. Mazumder. “Best Subset Selection via a Modern Opti-
mization Lens”. In: The Annals of Statistics (Apr. 2016). DOI: 10.1214/15-AOS1388.
[BLM00]
S. Boucheron, G. Lugosi, and P. Massart. “A Sharp Concentration Inequality with
Applications”. In: Random Structures & Algorithms (2000). DOI: 10.1002/(SICI)
1098-2418(200005)16:3<277::AID-RSA4>3.0.CO;2-1.
[BLM13]
S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic
Theory of Independence. Feb. 7, 2013.
[BM06]
P. L. Bartlett and S. Mendelson. “Empirical Minimization”. In: Probability Theory and
Related Fields (July 1, 2006). DOI: 10.1007/s00440-005-0462-3.
[BMR21]
P. L. Bartlett, A. Montanari, and A. Rakhlin. “Deep Learning: A Statistical Viewpoint”.
In: Acta Numerica (May 2021). DOI: 10.1017/S0962492921000027.
[Bou02]
O. Bousquet. “A Bennett Concentration Inequality and Its Application to Suprema
of Empirical Processes”. In: Comptes Rendus Mathematique (Jan. 1, 2002). DOI: 10.
1016/S1631-073X(02)02292-6.
[BP20]
D. Bertsimas and B. V. Parys. “Sparse High-Dimensional Regression: Exact Scalable
Algorithms and Phase Transitions”. In: The Annals of Statistics (Feb. 2020). DOI:
10.1214/18-AOS1804.
[BPV20]
D. Bertsimas, J. Pauphilet, and B. Van Parys. “Sparse Regression: Scalable Algorithms
and Empirical Performance”. In: Statistical Science (2020). URL.
[COB19]
L. Chizat, E. Oyallon, and F. Bach. “On Lazy Training in Differentiable Programming”.
In: Advances in Neural Information Processing Systems. 2019. URL.
[DI17]
G. David and Z. Ilias. “High Dimensional Regression with Binary Coefficients. Estimat-
ing Squared Error and a Phase Transtition”. In: Proceedings of the 2017 Conference on
Learning Theory. June 18, 2017. URL.
[EE23]
A. El Hanchi and M. A. Erdogdu. “Optimal Excess Risk Bounds for Empirical Risk
Minimization on p-Norm Linear Regression”. In: Advances in Neural Information
Processing Systems (Dec. 15, 2023). URL.

11


---Page Break---
[EME24]
A. El Hanchi, C. Maddison, and M. Erdogdu. “Minimax Linear Regression under the
Quantile Risk”. In: Proceedings of Thirty Seventh Conference on Learning Theory.
June 30, 2024. URL.
[GA11]
M. Gönen and E. Alpaydin. “Multiple Kernel Learning Algorithms”. In: Journal of
Machine Learning Research (2011). URL.
[Gho+19]
B. Ghorbani, S. Mei, T. Misiakiewicz, and A. Montanari. “Limitations of Lazy Training
of Two-layers Neural Network”. In: Advances in Neural Information Processing Systems.
2019. URL.
[Gho+20]
B. Ghorbani, S. Mei, T. Misiakiewicz, and A. Montanari. “When Do Neural Networks
Outperform Kernel Methods?” In: Advances in Neural Information Processing Systems.
2020. URL.
[GN15]
E. Giné and R. Nickl. Mathematical Foundations of Infinite-Dimensional Statistical
Models. 2015. DOI: 10.1017/CBO9781107337862.
[GR04]
E. Greenshtein and Y. Ritov. “Persistence in High-Dimensional Linear Predictor Selec-
tion and the Virtue of Overparametrization”. In: Bernoulli (Dec. 2004). DOI: 10.3150/
bj/1106314846.
[Gre06]
E. Greenshtein. “Best Subset Selection, Persistence in High-Dimensional Statistical
Learning and Optimization under L1 Constraint”. In: The Annals of Statistics (Oct.
2006). DOI: 10.1214/009053606000000768.
[Guy+24]
T. Guyard, C. Herzet, C. Elvira, and A.-N. Arslan. “A New Branch-and-Bound Pruning
Framework for ℓ0-Regularized Problems”. In: Proceedings of the 41st International
Conference on Machine Learning. July 8, 2024. URL.
[He+16]
K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition”.
In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). June
2016. DOI: 10.1109/CVPR.2016.90.
[HKZ12]
D. Hsu, S. M. Kakade, and T. Zhang. “Random Design Analysis of Ridge Regression”.
In: Proceedings of the 25th Annual Conference on Learning Theory. June 16, 2012.
URL.
[HMS22]
H. Hazimeh, R. Mazumder, and A. Saab. “Sparse Regression at Scale: Branch-and-
Bound Rooted in First-Order Optimization”. In: Mathematical Programming (Nov. 1,
2022). DOI: 10.1007/s10107-021-01712-4.
[HTF09]
T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning: Data
Mining, Inference, and Prediction, Second Edition. Aug. 26, 2009.
[Hua+18]
J. Huang, Y. Jiao, Y. Liu, and X. Lu. “A Constructive Approach to L0 Penalized
Regression”. In: Journal of Machine Learning Research (2018). URL.
[JRT08]
A. Juditsky, P. Rigollet, and A. B. Tsybakov. “Learning by Mirror Averaging”. In: The
Annals of Statistics (Oct. 2008). DOI: 10.1214/07-AOS546.
[Kel+23]
J. Kelner, F. Koehler, R. Meka, and D. Rohatgi. “Feature Adaptation for Sparse Linear
Regression”. In: Advances in Neural Information Processing Systems (Dec. 15, 2023).
URL.
[KM15]
V. Koltchinskii and S. Mendelson. “Bounding the Smallest Singular Value of a Random
Matrix Without Concentration”. In: International Mathematics Research Notices (Jan. 1,
2015). DOI: 10.1093/imrn/rnv096.
[Kol06]
V. Koltchinskii. “Local Rademacher Complexities and Oracle Inequalities in Risk
Minimization”. In: The Annals of Statistics (Dec. 2006).
DOI: 10 . 1214 /
009053606000001019.
[Kol11]
V. Koltchinskii. Oracle Inequalities in Empirical Risk Minimization and Sparse Recovery
Problems: École D’Été de Probabilités de Saint-Flour XXXVIII-2008. July 29, 2011.
[KRV22]
V. Kanade, P. Rebeschini, and T. Vaskevicius. Exponential Tail Local Rademacher
Complexity Risk Bounds Without the Bernstein Condition. Feb. 23, 2022. DOI: 10.
48550/arXiv.2202.11461.
[KSH12]
A. Krizhevsky, I. Sutskever, and G. E. Hinton. “ImageNet Classification with Deep Con-
volutional Neural Networks”. In: Advances in Neural Information Processing Systems.
2012. URL.

12


---Page Break---
[Lan+04]
G. R. G. Lanckriet, N. Cristianini, P. Bartlett, L. E. Ghaoui, and M. I. Jordan. “Learning
the Kernel Matrix with Semidefinite Programming”. In: Journal of Machine Learning
Research (2004). URL.
[LBH15]
Y. LeCun, Y. Bengio, and G. Hinton. “Deep Learning”. In: Nature (May 28, 2015). DOI:
10.1038/nature14539.
[LC06]
E. L. Lehmann and G. Casella. Theory of Point Estimation. May 2, 2006.
[LL20]
G. Lecué and M. Lerasle. “Robust Machine Learning by Median-of-Means: Theory and
Practice”. In: The Annals of Statistics (Apr. 2020). DOI: 10.1214/19-AOS1828.
[LM09]
G. Lecué and S. Mendelson. “Aggregation via Empirical Risk Minimization”. In: Proba-
bility Theory and Related Fields (Nov. 1, 2009). DOI: 10.1007/s00440-008-0180-8.
[LM16a]
G. Lecué and S. Mendelson. Learning Subgaussian Classes : Upper and Minimax
Bounds. Sept. 17, 2016. DOI: 10.48550/arXiv.1305.4825.
[LM16b]
G. Lecué and S. Mendelson. “Performance of Empirical Risk Minimization in Linear
Aggregation”. In: Bernoulli (Aug. 2016). DOI: 10.3150/15-BEJ701.
[LM19]
G. Lugosi and S. Mendelson. “Risk Minimization by Median-of-Means Tournaments”.
In: Journal of the European Mathematical Society (Dec. 16, 2019). DOI: 10.4171/
jems/937.
[LRS15]
T. Liang, A. Rakhlin, and K. Sridharan. “Learning with Square Loss: Localization
through Offset Rademacher Complexity”. In: Proceedings of The 28th Conference on
Learning Theory. June 26, 2015. URL.
[LvHY18]
R. Latała, R. van Handel, and P. Youssef. “The Dimension-Free Structure of Nonho-
mogeneous Random Matrices”. In: Inventiones mathematicae (Dec. 1, 2018). DOI:
10.1007/s00222-018-0817-x.
[Men14]
S. Mendelson. “Learning without Concentration”. In: Proceedings of The 27th Confer-
ence on Learning Theory. May 29, 2014. URL.
[Mil02]
A. Miller. Subset Selection in Regression. Apr. 14, 2002.
DOI: 10 . 1201 /
9781420035933.
[Mou22]
J. Mourtada. “Exact Minimax Risk for Linear Least Squares, and the Lower Tail of
Sample Covariance Matrices”. In: The Annals of Statistics (Aug. 2022). DOI: 10.1214/
22-AOS2181.
[Nat95]
B. K. Natarajan. “Sparse Approximate Solutions to Linear Systems”. In: SIAM Journal
on Computing (Apr. 1995). DOI: 10.1137/S0097539792240406.
[OB21]
D. M. Ostrovskii and F. Bach. “Finite-Sample Analysis of M-Estimators Using Self-
Concordance”. In: Electronic Journal of Statistics (Jan. 2021). DOI: 10.1214/20-
EJS1780.
[Oli16]
R. I. Oliveira. “The Lower Tail of Random Quadratic Forms with Applications to
Ordinary Least Squares”. In: Probability Theory and Related Fields (Dec. 1, 2016). DOI:
10.1007/s00440-016-0738-9.
[PG99]
V. de la Peña and E. Giné. Decoupling: From Dependence to Independence. 1999.
[PWE15]
M. Pilanci, M. J. Wainwright, and L. El Ghaoui. “Sparse Learning via Boolean Relax-
ations”. In: Mathematical Programming (June 1, 2015). DOI: 10.1007/s10107-015-
0894-1.
[RWY11]
G. Raskutti, M. J. Wainwright, and B. Yu. “Minimax Rates of Estimation for High-
Dimensional Linear Regression Over ℓq-Balls”. In: IEEE Transactions on Information
Theory (Oct. 2011). DOI: 10.1109/TIT.2011.2165799.
[Sau18]
A. Saumard. “On Optimality of Empirical Risk Minimization in Linear Aggregation”.
In: Bernoulli (2018). URL.
[SD16]
A. Sinha and J. C. Duchi. “Learning Kernels with Random Features”. In: Advances in
Neural Information Processing Systems. 2016. URL.
[She+13]
X. Shen, W. Pan, Y. Zhu, and H. Zhou. “On Constrained and Regularized High-
Dimensional Regression”. In: Annals of the Institute of Statistical Mathematics (Oct. 1,
2013). DOI: 10.1007/s10463-012-0396-3.
[SPZ12]
X. Shen, W. Pan, and Y. Zhu. “Likelihood-Based Selection and Sharp Parameter Es-
timation”. In: Journal of the American Statistical Association (June 11, 2012). DOI:
10.1080/01621459.2011.645783.

13


---Page Break---
[Tal96]
M. Talagrand. “New Concentration Inequalities in Product Spaces”. In: Inventiones
mathematicae (Nov. 1, 1996). DOI: 10.1007/s002220050108.
[Tro15]
J. A. Tropp. “An Introduction to Matrix Concentration Inequalities”. In: Found. Trends
Mach. Learn. (May 1, 2015). DOI: 10.1561/2200000048.
[Tro16]
J. A. Tropp. “The Expected Norm of a Sum of Independent Random Matrices: An
Elementary Approach”. In: High Dimensional Probability VII. Ed. by C. Houdré, D. M.
Mason, P. Reynaud-Bouret, and J. Rosi´nski. 2016. DOI: 10.1007/978-3-319-40519-
3_8.
[Vaa98]
A. W. van der Vaart. Asymptotic Statistics. 1998. DOI: 10.1017/CBO9780511802256.
[Vas+17]
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. ukasz Kaiser,
and I. Polosukhin. “Attention Is All You Need”. In: Advances in Neural Information
Processing Systems. 2017. URL.
[VC74]
V. Vapnik and A. Chervonenkis. Theory of Pattern Recognition. 1974.
[vHan17]
R. van Handel. “Structured Random Matrices”. In: Convexity and Concentration. 2017.
DOI: 10.1007/978-1-4939-7005-6_4.
[VW96]
A. van der Vaart and J. A. Wellner. Weak Convergence and Empirical Processes: With
Applications to Statistics. Mar. 14, 1996.
[Wai19]
M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. 2019.
DOI: 10.1017/9781108627771.
[Whi82]
H. White. “Maximum Likelihood Estimation of Misspecified Models”. In: Econometrica
(1982). DOI: 10.2307/1912526.
[Zha+21]
C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. “Understanding Deep Learning
(Still) Requires Rethinking Generalization”. In: Commun. ACM (Feb. 22, 2021). DOI:
10.1145/3446776.
[Zhu+20]
J. Zhu, C. Wen, J. Zhu, H. Zhang, and X. Wang. “A Polynomial Algorithm for Best-
Subset Selection Problem”. In: Proceedings of the National Academy of Sciences
(Dec. 29, 2020). DOI: 10.1073/pnas.2014241117.
[ZWJ14]
Y. Zhang, M. J. Wainwright, and M. I. Jordan. “Lower Bounds on the Performance of
Polynomial-time Algorithms for Sparse Linear Regression”. In: Proceedings of The
27th Conference on Learning Theory. May 29, 2014. URL.
[ZWJ17]
Y. Zhang, M. J. Wainwright, and M. I. Jordan. “Optimal Prediction for Sparse Linear
Models? Lower Bounds for Coordinate-Separable M-estimators”. In: Electronic Journal
of Statistics (Jan. 2017). DOI: 10.1214/17-EJS1233.
[ZZ12]
C.-H. Zhang and T. Zhang. “A General Theory of Concave Regularization for High-
Dimensional Sparse Estimation Problems”. In: Statistical Science (Nov. 2012). DOI:
10.1214/12-STS399.

14


---Page Break---
A
Proof of Theorem 1

Let An denote the event that Σn is invertible. By the weak law of large numbers, Σn converges to Σ
in probability so that limn→∞P(Ac
n) = 0. Now on the event An, we have by (1)
√n · ( ˆwn −w∗) = Σ−1
n
· (√n · ∇Rn(w∗)).

By the continuous mapping theorem, Σ−1
n
converges to Σ−1 in probability and by the central limit
theorem
√n · ∇Rn(w∗)
d→N(0, G).

Therefore, by Slutsky’s theorem

√n · ( ˆwn −w∗)
d→N(0, Σ−1GΣ−1).

Now since the risk is quadratic and the gradient vanishes at w∗,

n · [R( ˆw) −R(w∗)] = 1

2 · ∥√n · ( ˆw −w∗)∥2
Σ
d→1

2∥Z∥2
2,

where Z is as in the theorem, and where the last statement follows by the continuous mapping theorem.
This proves the first statement. The bounds on the quantiles are a consequence of concentration
bounds for the norm of Gaussian vectors [e.g. EME24, Corollary 33].

B
Proof of Theorem 2

Denote by An the event that

λmin

Σ−1/2ΣnΣ−1/2
≥1

2.

We show that under the sample size restriction, P(An) ≥1 −δ/2. Indeed we have the variational
representation

λmax(I −Σ−1/2ΣnΣ−1/2) =
sup
v∈Sd−1
1
n

n
X

i=1
1 −⟨v, Σ−1/2ϕ(Xi)⟩2.

Each element in the sum is upper bounded by 1, and the variance parameter in Bousquet’s concentra-
tion inequality [Bou02] is given by the parameter L in the statement of the theorem. Applying this
inequality yields that with probability at least 1 −δ/2

λmax(I −Σ−1/2ΣnΣ−1/2) ≤2 E
h
λmax(I −Σ−1/2ΣnΣ−1/2)
i
+

r

2L log(2/δ)

n
+ 4 log(2/δ)

3n
.

Using Lemma 3 to upper bound the above expectation, and replacing the sample size n in the resulting
inequality with the minimal allowed by the theorem proves that P(An) ≥1 −δ/2 for all sample
sizes allowable by the theorem. Now on this event we have, using (2),

R( ˆwn) −R(w∗) = 1

2 · ∥Σ−1
n ∇Rn(w∗)∥2
Σ ≤2 · ∥∇Rn(w∗)∥2
Σ−1.

An elementary calculation shows

E

∥∇Rn(w∗)∥2
Σ−1

= n−1 E

∥g(X, Y )∥2
Σ−1

,

so that an application of Markov’s inequality yields that there is an event Bn that holds with probability
at least 1 −δ/2 and on which

∥∇Rn(w∗)∥2
Σ−1 ≤2 · (nδ)−1 · E

∥g(X, Y )∥2
Σ−1

.

The union bound P(An ∩Bn) = 1 −P(An ∪Bn) ≥1 −δ finishes the proof.

15


---Page Break---
C
Main Lemma

We state here a core lemma, which we use in many of our proofs. To state it, we define, for a function
F : S →R on a subset S ⊆T ,

∥F∥∞:= sup
s∈S
|F(s)|,
∥F∥∞,−:= sup
s∈S
{−F(s)},
∥F∥∞,+ := sup
s∈S
F(s),

where the first quantity is the ℓ∞norm of the function F, and the remaining are one-sided variants of
it. The processes appearing in the next statement are defined in (3) and (4).

Lemma 4. Assume that T∗̸= ∅and let t∗∈T∗. On the event that ∥n−1/2∆n(·, t∗)∥∞,+ < 1 and
∥n−1/2Λn∥∞,+ < 1, we have

R(ˆtn, w∗(ˆtn)) −R∗≤1

2 ·
1
1 −∥n−1/2∆n(·, t∗)∥∞,+
·
1
1 −∥n−1/2Λn∥∞,+
· (n−1G2
n(ˆtn)),

and

1
2 ·
n−1G2
n(ˆtn)
(1 + ∥n−1/2Λn∥∞,−)2 ≤R(ˆtn, ˆwn) −R(ˆtn, w∗(ˆtn)) ≤1

2 ·
n−1G2
n(ˆtn)
(1 −∥n−1/2Λn∥∞,+)2 .

Proof. To lighten the notation, we drop the dependence on n, and write ˆt instead of ˆtn. We start with
the first statement. First, we note that if ˆt ∈T∗, then the statement holds trivially as the left-hand side
is zero, so we only consider the other case in what follows. For any t ∈T , define

ˆw(t) ∈argmin
w∈Rd Rn(t, w),

where the choice of minimizer is arbitrary. With this definition, we have ˆwn = ˆw(ˆt). Now, by
definition of ERM,
Rn(ˆt, ˆw(ˆt)) −Rn(t∗, w∗(t∗)) ≤0.
(9)
On the other hand, for any t ∈T \ T∗, we have the decomposition

Rn(t, ˆw(t)) −Rn(t∗, w∗) = [Rn(t, ˆw(t)) −Rn(t, w∗(t))] + [Rn(t, w∗(t)) −Rn(t∗, w∗(t∗))].
(10)
We study each of the terms of (10) separately, and we start with the first. Note that since we are in the
event
inf
t∈T λmin(Σ−1/2(t)Σn(t)Σ−1/2(t)) = 1 −∥n−1/2Λn∥∞,+ > 0,

the sample covariance matrices Σn(t) are invertible for all t ∈T , so that ˆw(t) is uniquely defined
and satisfies
ˆw(t) = w∗(t) −Σ−1
n (t)∇wRn(t, w∗(t)).
(11)
Furthermore, since the function w 7→Rn(t, w) is quadratic in w and its gradient vanishes at its
minimizer ˆw(t), we have

Rn(t, ˆw(t)) −Rn(t, w∗(t)) = −1

2∥ˆw(t) −w∗(t)∥2
Σn(t) = −1

2∥∇wRn(t, w∗(t))∥2
Σ−1
n (t),
(12)

where the last equality follows from (11). To bound this last term, define

eΣn(t) := Σ−1/2(t)Σn(t)Σ−1/2(t).

Then we have,

∥∇wRn(t, w∗(t))∥2
Σ−1
n (t) =
n
Σ−1/2(t)∇wRn(t, w∗(t))
oT eΣ−1
n (t)
n
Σ−1/2(t)∇wRn(t, w∗(t))
o

≤λmax(eΣ−1
n (t)) · ∥∇wRn(t, w∗(t))∥2
Σ−1(t)

=
1

1 −λmax(I −eΣn(t))
· (n−1G2
n(t))

≤
1
1 −∥n−1/2Λn∥∞,+
· (n−1G2
n(t)).
(13)

16


---Page Break---
Finally, the second term of (10) is lower bounded by

Rn(t, w∗(t)) −Rn(t∗, w∗(t∗))) = (1 −n−1/2∆n(t, t∗))[R(t, w∗(t)) −R∗]

≥(1 −∥n−1/2∆n(·, t∗)∥∞,+)[R(t, w∗(t)) −R∗]
(14)

Combining (13) and (12) lower bounds the first term of (10), while (14) lower bounds the second.
Combining the resulting lower bound on (10) with (9) and rearranging yields the first statement.

For the upper bound in the second statement, note that for all t ∈T ,

R(t, ˆw(t)) −R(t, w∗(t)) = 1

2 · ∥ˆw(t) −w∗(t)∥2
Σ(t)

= 1

2 · ∥Σ−1
n (t)∇wRn(t, w∗(t))∥2
Σ(t)

≤1

2 · λmax(eΣ−2
n (t)) · ∥∇wRn(t, w∗(t))∥2
Σ−1(t)

= 1

2 ·
1
(1 −∥n−1/2Λn∥∞,+)2 · (n−1G2
n(t)).

where the second line follows from (11). In particular the inequality holds for ˆt. The lower bound
holds by a similar argument.

D
Proof of Theorem 3

Consistency of ˆtn. We want to show that, as n →∞,

R(ˆtn, w∗(ˆtn)) −R∗
p→0.
(15)

Using the notation introduced in Appendix C, the Glivenko-Cantelli assumptions in Theorem 3
amount to the statements that, for some t∗∈T∗, and as n →∞,

∥n−1/2Λn∥∞
p→0,
∥n−1/2∆n(·, t∗)∥∞
p→0,
∥n−1/2Gn∥∞
p→0.
(16)

Let An denote the event that both ∥n−1/2Λn∥∞< 1 and ∥n−1/2∆n(·, t∗)∥∞< 1. The union bound
and (16) show that
lim
n→∞P(Ac
n) = 0

Furthermore, on the event An, the first bound of Lemma 4 holds, and bounding n−1G2
n(ˆtn) by
∥n−1/2Gn∥2
∞yields that on An

R(ˆtn, w∗(ˆtn)) −R∗≤1

2 ·
1
1 −∥n−1/2∆n(·, t∗)∥∞
·
1
1 −∥n−1/2Λn∥∞
· ∥n−1/2Gn∥2
∞
(17)

Now let ε > 0, and denote by Bn(ε) the event that the right hand side of (17) is strictly larger
than ε.
Then the statements (16) together with the continuous mapping theorem show that
limn→∞P(Bn(ε)) = 0. Therefore, again by (17), we have

P
 
R(ˆtn, w∗(ˆtn)) −R∗> ε

≤P(Bn(ε)) + P(Ac
n),

and taking n →∞proves (15).

Asymptotic quantiles. We start with the upper bound. We have the simple decomposition

n ·

R(ˆtn, ˆwn) −R∗

= n

R(ˆtn, ˆwn) −R(ˆtn, w∗(ˆtn))

+ n

R(ˆtn, w∗(ˆtn)) −R∗

.
(18)

Now on the event An defined above, we have, by an application of Lemma 4, combining the two
bounds in the lemma along with (18), that the rescaled excess risk is upper bounded by

1
2 ·
1
1 −∥n−1/2Λn∥∞
·

1
1 −∥n−1/2∆n(·, t∗)∥∞
+
1
1 −∥n−1/2Λn∥∞


· G2
n(ˆtn).
(19)

17


---Page Break---
From the Glivenko-Cantelli assumptions (16), the first three factors converge in probability to 1. Our
aim will be to bound the upper tail of the last factor, which will imply a bound on the upper tail of the
rescaled excess risk.

We briefly make explicit the Donsker assumption before deriving this bound. Both define and note

Gn(t, v) := √n · ⟨v, Σ−1/2(t)∇wRn(t, w∗(t))⟩,
Gn(t) =
sup
v∈Sd−1 Gn(t, v),

where Sd−1 is the Euclidean unit sphere in Rd. As pointed out in Section 3, the processes Gn(t) are
partial suprema of the empirical processes Gn(t, v). The Donsker assumption of the theorem states
that the empirical processes Gn(t, v) take value in the space of bounded functions on T × Sd−1,
equipped with the ℓ∞(T × Sd−1) norm and the metric it induces, and converge weakly to their
unique Gaussian limit (G(t, v))(t,v)∈T ×Sd−1 as n →∞. By inspecting their finite dimensional

distributions, it is straightforward to verify that (G(t, v))(t,v)∈T ×Sd−1
d= (⟨v, Z(t)⟩)(t,v)∈T ×Sd−1
where (Z(t))t∈T is the Rd-valued Gaussian process defined in the statement of Theorem 3. Finally,
we define G(t) := supv∈Sd−1 G(t, v) in analogy with the definition of Gn(t).

We now upper bound the upper tail of G2
n(ˆtn) in (19). Let (εk)∞
k=1 be a decreasing sequence of
positive numbers such that εk →0 as k →∞, and define the sets
T∗(ε) := {t ∈T | R(t, w∗(t)) −R∗≤ε}
(20)

as well as the function Fk : ℓ∞(T × Sd−1) →R by
Fk(z) :=
sup
s∈T∗(εk)
sup
v∈Sd−1 z(s, v).

Note on the one hand that ∩k≥1T∗(εk) = T∗, and on the other that Fk is continuous for all k ∈N,
and in fact Lipschitz. Indeed, let z, z′ ∈ℓ∞(T × Sd−1). Then

|Fk(z) −Fk(z′)| =


sup
s∈T∗(εk)
sup
v∈Sd−1 z(s, v) −
sup
s∈T∗(εk)
sup
v∈Sd−1 z′(s, v)

 ≤∥z −z′∥∞

Now let k ∈N and x ∈[0, ∞). Then

P
 
G2
n(ˆtn) > x

= P
 
G2
n(ˆtn) > x
	
∩
ˆtn ∈T∗(εk)
	
+ P
 
G2
n(ˆtn) > x
	
∩
ˆtn /∈T∗(εk)
	

≤P
 
G2
n(ˆtn) > x
	
∩
ˆtn ∈T∗(εk)
	
+ P
 ˆtn /∈T∗(εk)
	

≤P

 

sup
s∈T∗(εk)
G2
n(s) > x

!

+ P
 
R(ˆtn, w∗(ˆtn)) −R∗> εk


= P
 
F 2
k (Gn) > x

+ P
 
R(ˆtn, w∗(ˆtn)) −R∗> εk


taking the limit as n →∞, the first term converges, by the continuous mapping theorem, to the
probability of the event

F 2
k (G) > x
	
, where G is the limiting Gaussian process discussed above,
while the second term vanishes by the first part of Theorem 3. Therefore, for all k ∈N,

lim sup
n→∞P
 
G2
n(ˆtn) > x

≤P

 

sup
s∈T∗(εk)
G2(s) > x

!

Taking the limit as k →∞, noticing that the events
(

sup
s∈T∗(εk)
G2(s) > x

)

are nested, using the continuity of probability from above, and recalling that ∩k≥1T∗(εk) = T∗gives

lim sup
n→∞P
 
G2
n(ˆtn) > x

≤P

sup
s∈T∗
G2(s) > x

.

Using properties of the quantile function (e.g. [EME24, Lemma 20]) finishes the proof of the upper
bound. For the lower bound, we make a similar argument. We have, by an application of Lemma 4,

n · [R(ˆtn, ˆwn) −R∗] ≥n · [R(ˆtn, ˆwn) −R(ˆtn, w∗(ˆtn))]

≥1

2 ·
1
(1 + ∥n−1/2Λn∥∞,−)2 · G2
n(ˆtn).

18


---Page Break---
By the Glivenko-Cantelli assumption on Λn, the first two factors converge to 1/2. For the third, we
will lower bound its upper tails, which will imply a lower bound on the upper tails of the rescaled
excess risk. We let (εk)∞
k=1 be a decreasing sequence of positive numbers such that εk →0 as
k →∞, and define Hk : ℓ∞(T × Sd−1) →R by
Hk(z) :=
inf
t∈T∗(εk)
sup
v∈Sd−1 z(t, v),

where the subsets T∗(εk) are as defined in (20). Clearly, for z, z′ ∈ℓ∞(T × Sd−1),
|Hk(z) −Hk(z′)| ≤∥z −z′∥∞
so Hk is Lipschitz and therefore continuous. Now

P
 
G2
n(ˆtn) > x

≥P
 
G2
n(ˆtn) > x
	
∩
ˆtn ∈T∗(εk)
	

≥P

inf
s∈T∗(εk) G2
n(s) > x

−P
 ˆtn /∈T∗(εk)
	

= P
 
H2
k(Gn) > x

−P
 
R(ˆtn, w∗(ˆtn)) −R∗> εk


By the same argument as above, we obtain, as n →∞, and for all k ∈N,

lim inf
n→∞P
 
G2
n(ˆtn) > x

≥P

inf
s∈T∗(εk) G2(s) > x


Taking the limit as k →∞, and noticing that
[

k≥1


inf
s∈T∗(εk) G2(s) > x

=

inf
s∈T∗G2(s) > x


proves that

lim inf
n→∞P
 
G2
n(ˆtn) > x

≥P

inf
s∈T∗G2(s) > x

.

Using properties of the quantile function (e.g. [EME24, Lemma 20]) finishes the proof of the lower
bound. The estimates on the quantiles of Z+ in Remark 1 are a consequence of standard Gaussian
concentration, see [e.g. EME24, Appendix A.3]. Finally, for the second statement in Remark 1,

E

max
s∈T∗∥Z(s)∥2
2


≤E





 X

s∈T∗
∥Z(s)∥2p
2

!1/p



≤

 X

s∈T∗
E
h
∥Z(s)∥2p
2
i!1/p

≤32 · p ·

 X

s∈T∗
E

∥Z(s)∥2
2
p
!1/p

,

where the last estimate follows from Gaussian concentration. Taking p = 1 + log|T∗|, and recalling
that for x ∈Rd, ∥x∥p ≤d1/p∥x∥∞yields the result.

E
Proof of Lemma 1

We prove the first statement by induction. For k = 0, this follows directly from the fact that by
definition F 0
n,δ(T ) = T and Fn,δ(T ) ⊆T . Now let k ∈N and assume that the statement holds for
k −1. Let s ∈F k+1
n,δ (T ). Then by definition

R(s, w∗(s)) −R∗≤2 · (nδ)−1 · E[
sup
s∈F k
n,δ(T )
G2
n(s)] ≤2 · (nδ)−1 · E[
sup
s∈F k−1
n,δ (T )
G2
n(s)].

where the second inequality follows from the fact that by the induction hypothesis, F k
n,δ(T ) ⊆
F k−1
n,δ (T ), and that the supremum is increasing. Therefore s ∈F k
n,δ(T ) since the last inequality
is the defining inequality for F k
n,δ(T ). We now turn to the second statement. Fix k and δ. On the
one hand, T∗⊆T

n≥1 F k
n,δ(T ). On the other, for any t ∈T

n≥1 F k
n,δ(T ), we have for all n ≥n0,
R(t, w∗(t)) −R∗≤2B · (nδ)−1. Therefore R(t, w∗(t)) −R∗= 0, and hence t ∈T∗.

19


---Page Break---
F
Proof of Theorem 4

Recall the notation introduced in Appendix C. Let An(t∗) be the event that:

∥n−1/2Λn∥∞,+ ≤1/2,
and
∥n−1/2∆n(·, t∗)∥≤1/2.

We start by showing that under the sample size inequality stated in the theorem, there exists a t∗∈T∗
such that P(An(t∗)) ≥1 −δ/3. Indeed, we have

∥n−1/2Λn∥∞,+ =
sup
(t,v)∈(T ×Sd−1)

1
n

n
X

i=1

n
1 −⟨v, Σ−1/2(t)ϕt(Xi)⟩2o

The elements of this sum are bounded by 1, and the variance parameter of Bousquet’s inequality
[Bou02] is given by L as defined in Section 3.2. Applying this inequality yields that with probability
at least 1 −δ/6

∥n−1/2Λn∥∞,+ ≤
2
n1/2 · E

sup
t∈T
Λn(t)

+

r

2L log(6/δ)

n
+ 4 log(6/δ)

3n

Furthermore, by Markov’s inequality, with probability at least 1 −δ/6

∥n−1/2∆(·, t∗)∥∞,+ ≤
6 · E[supt∈T \T∗∆(t, t∗)]

n1/2 · δ
Hence, when the inequality on the sample size stated in the theorem holds for some t∗, the event
An(t∗) holds with probability at least 1−δ/3. Now on this event, the first bound of Lemma 4 applies,
and we have
R(ˆtn, w∗(ˆtn)) −R∗≤2 · n−1 · G2
n(ˆtn).
(21)
Now we use the iterative localization method of Koltchinskii [Kol06]. Initially, we have no informa-
tion about where ˆtn is located aside from belonging to T , so we start with the bound

R(ˆtn, w∗(ˆtn)) −R∗≤2 · n−1 · sup
t∈T
G2
n(t).
(22)

Using Markov’s inequality, we have on an event Bn,1 which holds with probability at least 1 −δ/2k

sup
t∈T
G2
n(t) ≤2k · δ−1 · E[sup
t∈T
G2
n(t)].

Replacing in (22) yields that on the event An(t∗) ∩Bn,1,

R(ˆtn, w∗(ˆtn)) −R∗≤4k · (nδ)−1 · E[sup
t∈T
G2
n(t)],

which shows that on this event, ˆtn ∈Fn,δ/2k(T ), by definition of the map Fn,δ/2k. With this
knowledge, we now reuse the bound (21) to obtain that on An(t∗) ∩Bn,1

R(ˆtn, w∗(ˆtn)) −R∗≤2 · n−1 ·
sup
t∈Fn,δ/2k(T )
G2
n(t).

Iterating the procedure we just described k times, we obtain that on an event An(t∗) ∩(∩k
j=1Bn,j),
where P(Bn,j) ≥1 −δ/2k for all j ∈[k]
ˆtn ∈F k
n,δ/2k(T ) = Sn,δ,k.
(23)

Another application of Markov’s inequality yields that on an event C which holds with probability at
least 1 −δ/6
sup
t∈Sn,δ,k
G2
n(t) ≤6 · δ−1 · E[ sup
t∈Sn,δ,k
G2
n(t)]
(24)

Since

P(An(t∗) ∩(∩k
j=1Bn,k) ∩C) ≥1 −δ/3 −

k
X

j=1
δ/2k −δ/6 = 1 −δ,

equation (23) proves the first statement of the theorem. For the second statement, we have on the
same event An(t∗) ∩(∩k
j=1Bn,k) ∩C , and combining the two upper bounds from Lemma 4,

E(ˆtn, ˆwn) ≤4 · n−1 · G2
n(ˆtn) ≤4 · n−1 ·
sup
t∈Sn,δ,k
G2
n(t) ≤24 · (nδ)−1 · E[ sup
t∈Sn,δ,k
G2
n(t)],

where we used (23) and (24) in the above inequalities, concluding the proof.

20


---Page Break---
G
Proof of Lemma 2

We prove a slightly more general result, from which Lemma 2 can be immediately deduced.
Lemma 5. Let n, d ∈N and let T be a finite set. For each (i, t) ∈[n] × T , let Zi,t ∈Rd be random
vectors such that for each t ∈T , (Zi,t)n
i=1 are i.i.d. with the same distribution as Zt. For all t ∈T ,
assume that E[Zt] = 0, and define

σ2(T ) := sup
t∈T
E

∥Zt∥2
2

,
rn(T ) := E

"

sup
(i,t)∈[n]×T
∥Zi,t∥2
2

#1/2

.

Then

1
2 · σ(T )

n1/2 + 1

4 · rn(T )

n
≤E



sup
t∈T


1
n

n
X

i=1
Zi,t



2

2





1/2

≤C(T ) · σ(T )

n1/2 + C2(T ) · rn(T )

n
,

where C(T ) := 5
p

1 + log|T |.

To prove Lemma 5, we need to recall a few preliminary results. The first is a classical symmetrization
inequality, see e.g. [BLM13, Lemma 11.4] or [Wai19, Proposition 4.11] for a proof.
Lemma 6. For each (i, t) ∈[n] × T , let Wi,t ∈Rd be random vectors such that for each t ∈T ,
(Wi,t)n
i=1 are i.i.d. with the same distribution as Wt. Let (εi)n
i=1 be independent Rademacher random
variables, independent of the collection of random vectors Wi,t. Define Wi,t := Wi,t −E[Wt]. Then

1
2 E



sup
t∈T


1
n

n
X

i=1
εiWi,t



2

2





1/2

≤E



sup
t∈T


1
n

n
X

i=1

Wi,t



2

2





1/2

≤2 E



sup
t∈T


1
n

n
X

i=1
εiWi,t



2

2





1/2

.

The second result we recall is the Khinchin-Kahane inequality, the specific form we require is
obtained from Peña and Giné [PG99, Theorem 1.3.1] by setting q = p and p = 2 in that theorem, see
also Boucheron et al. [BLM13, page 141].
Lemma 7. For i ∈[n], let zi ∈Rd be fixed vectors. Let (εi)n
i=1 be independent Rademacher random
variables. Then for all p ≥2,

E

"

n
X

i=1
εizi



p

2

#1/p

≤
p

p −1 ·

 n
X

i=1
∥zi∥2
2

!1/2

.

A straightforward consequence of Lemma 7 is the following result, which follows from the elementary
observation that for a vector x ∈Rd, ∥x∥∞≤∥x∥p ≤d1/p∥x∥∞.
Lemma 8. For (i, t) ∈[n]×T , let zi,t ∈Rd be fixed vectors. Let (εi)n
i=1 be independent Rademacher
random variables. Then

E



sup
t∈T



n
X

i=1
εizi,t



2

2





1/2

≤5

2

p

1 + log|T | ·

 

sup
t∈T

n
X

i=1
∥zi,t∥2
2

!1/2

.

Proof. Let p ≥1. Then, by Jensen’s inequality and Lemma 7

E



sup
t∈T



n
X

i=1
εizi,t



2

2



≤E







X

t∈T



n
X

i=1
εizi,t



2p

2





1/p



≤



X

t∈T
E







n
X

i=1
εizi,t



2p

2









1/p

≤(2p −1) ·

 X

t∈T

( n
X

i=1
∥zi,t∥2
2

)p!1/p

.

Recalling that ∥x∥p ≤d1/p∥x∥∞for all x ∈Rd and taking p := 1 + log|T | yields the result.

21


---Page Break---
Finally, we need the following consequence of Lemmas 6 and 8. The proof idea is taken from [Tro16].

Lemma 9. For each (i, t) ∈[n] × T , let Wi,t ∈R be random variables such that for each t ∈T ,
(Wi,t)n
i=1 are i.i.d. with the same distribution as Wt, with Wt ≥0 almost surely. Then

E

"

sup
t∈T

n
X

i=1
Wi,t

#1/2

≤

 

sup
t∈T

n
X

i=1
E[Wi,t]

!1/2

+ 5
p

1 + log|T | · E

"

sup
(i,t)∈[n]×T
Wi,t

#1/2

.

Proof. We have by Jensen’s inequality and Lemma 6,

E

"

sup
t∈T

n
X

i=1
Wi,t

#

≤E

"

sup
t∈T



n
X

i=1
Wi,t −E[Wi,t]



#

+ sup
t∈T

n
X

i=1
E[Wi,t],

≤2 E



sup
t∈T



n
X

i=1
εiWi,t



2



1/2

+ sup
t∈T

n
X

i=1
E[Wi,t].
(25)

Conditioning on the random vectors Wi,t, we have by Lemma 8 and the assumption Wi,t ≥0 a.s.

2 E



sup
t∈T



n
X

i=1
εiWi,t



2



1/2

≤5
p

(1 + log|T |) ·

 

sup
t∈T

n
X

i=1
W 2
i,t

!1/2

,

≤5
p

1 + log|T | ·

 

sup
(i,t)∈[n]×T
Wi,t

!1/2

·

 

sup
t∈T

n
X

i=1
Wi,t

!1/2

.

Taking expectation with respect to Wi,t, and using the Cauchy-Schwartz inequality yields

2 E



sup
t∈T



n
X

i=1
εiWi,t



2



1/2

≤
p

6(1 + log|T |) · E

"

sup
(i,t)∈[n]×T
Zi,t

#1/2

· E

"

sup
t∈T

n
X

i=1
Wi,t

#1/2

.

Replacing in (25) and solving the resulting quadratic inequality yields the result.

Equipped with these results, we now prove Lemma 1. The proof idea is taken from [Tro16].

Proof of Lemma 1. We start with the lower bound. We have on the one hand

E



sup
t∈T


1
n

n
X

i=1
Zi,t



2

2



≥sup
t∈T
E






1
n

n
X

i=1
Zi,t



2

2



= σ2(T ).
(26)

On the on other hand, by Lemma 6, we have

E



sup
t∈T


1
n

n
X

i=1
Zi,t



2

2





1/2

≥1

2 E



sup
t∈T


1
n

n
X

i=1
εiZi,t



2

2





1/2

.

Define the random index
I ∈argmax
i∈[n]
max
t∈T ∥Zi,t∥2
2.

Conditioning on Zi,t, we have by Jensen’s inequality

E



sup
t∈T


1
n

n
X

i=1
εiZi,t



2

2



≥sup
t∈T
E





E

"
1
n

n
X

i=1
εiZi,t

#

2

2



= sup
t∈T

∥ZI,t∥2
2
n2
=
sup
(i,t)∈[n]×T

∥Zi,t∥2
2
n2
,

22


---Page Break---
where in the inequality, the outer expectation is with respect to εI, and the inner one is with respect
to (εi)i̸=I. Taking expectation with respect to Zi,t gives

E



sup
t∈T


1
n

n
X

i=1
Zi,t



2

2





1/2

≥1

2 · rn(T )

n
(27)

Averaging the lower bounds (26) and (27) yields the desired lower bound. We now turn to the upper
bound. We have by Lemmas 6 and 8.

E



sup
t∈T


1
n

n
X

i=1
Zi,t



2

2





1/2

≤2 E



sup
t∈T


1
n

n
X

i=1
εiZi,t



2

2





1/2

≤5
p

1 + log|T | · E

"

sup
t∈T

n
X

i=1


1
nZi,t



2

2

#1/2

Applying Lemma 9 on the last term yields the desired upper bound.

H
Proof of Corollary 1

The Glivenko-Cantelli and Donsker assumptions of Theorem 3 follow directly from the moment
assumptions of the corollary, the weak law of large numbers, and the central limit theorem, and
therefore the conclusions of Theorem 3 hold. For the first statement of the corollary, we may assume
without loss of generality that T∗̸= T , otherwise the statement holds trivially. Define

ε :=
min
t∈T \T∗{R(t, w∗(t)) −R∗}.

Then ε > 0, and by the first item of Theorem 3,

lim
n→∞P
 ˆtn /∈T∗

≤lim
n→∞P
 
R(ˆtn, w∗(ˆtn)) −R∗> ε/2

= 0.
(28)

It remains to prove the improved upper bound on the asymptotic quantiles. For this, referring to the
proof of Theorem 3, and in particular to (18), it is enough to show that

lim
n→∞P
 
n ·

R(ˆtn, w∗(ˆtn)) −R∗)

> 0

= 0,

but this follows directly from (28).

I
Proof of Corollary 2

The statement follows from the same argument as Theorem 4 with only a few simple modifications. As
explained in the main text, we use Lemma 3 to bound the quantity E[maxt∈T Λn(t)] by constructing
a block diagonal matrix. We use Lemma 2 to control, for any subset S, E

maxs∈S G2
n(s)

. The only
minor deviation from Theorem 4 is that we bound the second moment

E

"

sup
t∈T \T∗
∆2
n(t, t∗)

#

instead of the first. This explains the slightly better dependence on δ in the sample size restriction of
Corollary 2 compared to Theorem 4.

23


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]
Justification: The main claims made in the abstract and introduction are supported by our
main results of Section 3 and 4 directly.

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

Justification: The last paragraph of the conclusion explicitly discusses the limitations of this
work.

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

24


---Page Break---
Answer: [Yes]

Justification: The proofs of all the statements we claimed are new can be found in the
appendix. For known statements, we provided direct references to them.

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
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

25


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: The paper does not include experiments.
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
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)

26


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
Justification: We have reviewed the guidelines and confirm that our work adheres to them.
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
Justification: Our paper is heavily theoretical, and is quite far removed from direct applica-
tions, which is why do not foresee any direct societal impact.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

27


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

Justification: This paper poses no such risks.

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
Justification: We have cited all the relevant work we are aware of, but we do not use any
existing code, data, or models in this work.

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

28


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not release new assets.
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
Justification: Not applicable.
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
Justification: Not applicable.
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

29


---Page Break---
