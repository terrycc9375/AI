Recursive PAC-Bayes: A Frequentist Approach to
Sequential Prior Updates with No Information Loss

Yi-Shan Wu
University of South Denmark
yswu@imada.sdu.dk

Yijie Zhang
University of Copenhagen & Novo Nordisk A/S
yizh@di.ku.dk

Badr-Eddine Chérief-Abdellatif
CNRS, LPSM, Sorbonne Université, Université Paris Cité
badr-eddine.cherief-abdellatif@cnrs.fr

Yevgeny Seldin
University of Copenhagen
seldin@di.ku.dk

Abstract

PAC-Bayesian analysis is a frequentist framework for incorporating prior knowl-
edge into learning. It was inspired by Bayesian learning, which allows sequential
data processing and naturally turns posteriors from one processing step into priors
for the next. However, despite two and a half decades of research, the ability to
update priors sequentially without losing conﬁdence information along the way
remained elusive for PAC-Bayes. While PAC-Bayes allows construction of data-
informed priors, the ﬁnal conﬁdence intervals depend only on the number of points
that were not used for the construction of the prior, whereas conﬁdence information
in the prior, which is related to the number of points used to construct the prior, is
lost. This limits the possibility and beneﬁt of sequential prior updates, because the
ﬁnal bounds depend only on the size of the ﬁnal batch.
We present a novel and, in retrospect, surprisingly simple and powerful PAC-
Bayesian procedure that allows sequential prior updates with no information loss.
The procedure is based on a novel decomposition of the expected loss of random-
ized classiﬁers. The decomposition rewrites the loss of the posterior as an excess
loss relative to a downscaled loss of the prior plus the downscaled loss of the prior,
which is bounded recursively. As a side result, we also present a generalization of
the split-kl and PAC-Bayes-split-kl inequalities to discrete random variables, which
we use for bounding the excess losses, and which can be of independent interest. In
empirical evaluation the new procedure signiﬁcantly outperforms state-of-the-art.

1
Introduction

PAC-Bayesian analysis was born from an attempt to derive frequentist generalization guarantees
for Bayesian-style prediction rules (Shawe-Taylor and Williamson, 1997, McAllester, 1998). The
motivation was to provide a way to incorporate prior knowledge into the frequentist analysis of gener-
alization. PAC-Bayesian bounds provide high-probability generalization guarantees for randomized
classiﬁers. A randomized classiﬁer is deﬁned by a distribution ρ on a set of prediction rules H, which
is used to sample a prediction rule each time a prediction is to be made. Bayesian posterior is an
example of a randomized classiﬁer, whereas PAC-Bayesian bounds hold generally for all randomized
classiﬁers. Prior knowledge is encoded through a prior distribution π on H, and the complexity
of a posterior distribution ρ is measured by the Kullback-Leibler (KL) divergence from the prior,
KL(ρ∥π). PAC-Bayesian generalization guarantees are optimized by posterior distributions ρ that
optimize a trade-off between empirical data ﬁt and divergence from the prior in the KL sense.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Selection of a “good” prior plays an important role in the PAC-Bayesian bounds. If one manages
to foresee which prediction rules are likely to produce low prediction error and allocate a higher
prior mass for them, then the bounds are tighter, because the posterior only needs to make a small
deviation from the prior. But if the prior mass on well-performing prediction rules is small, the
bounds are loose. A major technique to design good priors is to use part of the data to estimate a
good prior and the rest of the data to compute a PAC-Bayes bound. It is known as data-dependent or
data-informed priors (Ambroladze et al., 2007). However, all existing approaches to data-informed
priors have three major disadvantages. The ﬁrst is that the bounds are computed on “the rest of the
data” that were not used in construction of the prior. Thus, the sample size in the bounds is only a
fraction of the total sample size. Therefore, empirically data-informed priors are not always helpful.
In many cases starting with an uninformed prior and using all the data to compute the posterior and
the bound turns to be superior to sacriﬁcing part of the data for prior construction (Ambroladze et al.,
2007, Mhammedi et al., 2019). The second disadvantage is that all the conﬁdence information about
the prior is lost in the process. In particular, a prior trained on a few data points is treated in the
same way as a prior trained on a lot of data. And a third related disadvantage is that sequential data
processing provides no beneﬁt, because the bounds only depend on the size of the last chunk and all
the conﬁdence information from processing earlier chunks is lost in the process.

Our main contribution is a new (and simple) way of decomposing the loss of a randomized classiﬁer
deﬁned by the posterior. We write it as an excess loss relative to a downscaled loss of the randomized
classiﬁer deﬁned by the prior plus the downscaled loss of the randomized classiﬁer deﬁned by the
prior. The excess loss can be bounded using PAC-Bayes-Empirical-Bernstein-style inequalities
(Tolstikhin and Seldin, 2013, Mhammedi et al., 2019, Wu et al., 2021, Wu and Seldin, 2022), whereas
the loss of the randomized classiﬁer deﬁned by the prior can be bounded recursively. The recursive
bound can both use the data used for construction of the prior and “the rest of the data”, and thereby
preserves conﬁdence information on the prior. Our contribution stands out relative to all prior work
on PAC-Bayes, and in fact all prior work on frequentist generalization bounds, because it makes
sequential data processing and sequential prior updates meaningful and beneﬁcial.

We note that while several recent papers experimented with sequential posterior updates by using
martingale-style analysis, in all these works the prior remained ﬁxed and only the posterior was
changing (Chugg et al., 2023, Biggs and Guedj, 2023, Rodríguez-Gálvez et al., 2024). The work on
sequential posterior updates is orthogonal to our contribution and can be combined with it. Namely,
it is possible to apply sequential posterior updates in-between sequential prior updates. Another line
of work used tools from online learning to derive PAC-Bayesian bounds (Jang et al., 2023), and in
this context Haddouche and Guedj (2023) have used sequential prior updates, but their bounds hold
for a uniform aggregation of sequentially constructed posteriors, which is different from standard
posteriors studied in our work. The conﬁdence bounds in their work come primarily from aggregation
rather than conﬁdence in individual posteriors in the sequence (the denominator of their bounds
depends on the number of aggregated posteriors). The need to construct and maintain a large number
of posteriors has a negative impact on the computational efﬁciency. Our work is the ﬁrst one allowing
sequential prior updates without loss of conﬁdence information.

An additional side contribution of independent interest is a generalization of the split-kl and PAC-
Bayes-split-kl inequalities of Wu and Seldin (2022) from ternary to general discrete random variables.
It is based on a novel representation of discrete random variables as a superposition of Bernoulli
random variables.

The paper is organized in the following way. In Section 2 we brieﬂy survey the evolution of data-
informed priors in PAC-Bayes and present our main idea behind Recursive PAC-Bayes; in Section 3
we present our generalization of the split-kl and PAC-Bayes-split-kl inequalities, which are later used
to bound the excess losses; in Section 4 we present the Recursive PAC-Bayes bound; in Section 5 we
present an empirical evaluation; and in Section 6 we conclude with a discussion.

2
The evolution of data-informed priors and the idea of Recursive PAC-Bayes

In this section we brieﬂy survey the evolution of data-informed priors, and then present our con-
struction of Recursive PAC-Bayes. We consider the standard classiﬁcation setting, with X being
a sample space, Y a label space, H a set of prediction rules h : X →Y, and ℓ(h(X), Y ) =
1(h(X) ̸= Y ) the zero-one loss function, where 1(·) denotes the indicator function. We let D

2


---Page Break---
denote a distribution on X × Y and S = {(X1, Y1), . . . , (Xn, Yn)} an i.i.d. sample from D. Let
L(h) = E(X,Y )∼D[ℓ(h(X), Y )] be the expected and ˆL(h, S) = 1

n
Pn
i=1 ℓ(h(Xi), Yi) the empirical
loss.

Let ρ be a distribution on H. A randomized classiﬁer associated with ρ samples a prediction rule h
according to ρ for each sample X ∈X, and applies it to make a prediction h(X). The expected loss
of such randomized classiﬁer, which we call ρ, is Eh∼ρ[L(h)] and the empirical loss is Eh∼ρ[ˆL(h, S)].
For brevity we use Eρ[·] to denote Eh∼ρ[·].

We use KL(ρ∥π) to denote the Kullback-Leibler divergence between two probability distributions, ρ
and π (Cover and Thomas, 2006). For p, q ∈[0, 1] we further use kl(p∥q) = KL((1−p, p)∥(1−q, q))
to denote the Kullback-Leibler divergence between two Bernoulli distributions with biases p and q.

The goal of PAC-Bayes is to bound Eρ[L(h)]. Below we present how the bounds on Eρ[L(h)] have
evolved. In Appendix A we also provide a graphical illustration of the evolution.

Uninformed priors
Early work on PAC-Bayes used uninformed priors (McAllester, 1998). An
uniformed prior π is a distribution on H that is independent of the data S. A classical, and still one of
the tightest bounds, is the following.
Theorem 1 (PAC-Bayes-kl Inequality, Seeger, 2002, Maurer, 2004). For any probability distribution
π on H that is independent of S and any δ ∈(0, 1):

P

∃ρ ∈P : kl

Eρ[ˆL(h, S)]
Eρ [L(h)]

≥KL(ρ∥π) + ln(2√n/δ)

n


≤δ,

where P is the set of all probability distributions on H, including those dependent on S.

A posterior ρ that minimizes Eρ[L(h)] has to balance between allocating higher mass to prediction
rules h with small ˆL(h, S) and staying close to π in the KL(ρ∥π) sense. Since π has to be independent
of S, typical uninformed priors aim “to leave maximal options open” for ρ by staying close to uniform.

Data-informed priors
Ambroladze et al. (2007) proposed to split the data S into two disjoint sets,
S = S1∪S2, and use S1 to construct a data-informed prior π and compute a bound on Eρ[L(h)] using
π and S2. Since in this approach π is independent of S2, Theorem 1 can be applied. The advantage is
that π can use S1 to give higher mass to promising classiﬁers, thus relaxing the regularization pressure
KL(ρ∥π) and making it easier for ρ to allocate even higher mass to well-performing classiﬁers (those
with small ˆL(h, S2)). The disadvantage is that the sample size in the bound (the n in the denominator)
decreases from the size of S to the size of S2. Indeed, Ambroladze et al. observed that the sacriﬁce
of S1 for prior construction does not always pay off.

Data-informed priors + excess loss
Mhammedi et al. (2019) observed that if we have already
sacriﬁced S1 for the construction of π, we could also use it to construct a reference prediction rule
h∗, typically an Empirical Risk Minimizer (ERM) on S1. They then employed the decomposition

Eρ[L(h)] = Eρ[L(h) −L(h∗)] + L(h∗)

and used S2 to give a PAC-Bayesian bound on Eρ[L(h) −L(h∗)] and a single-hypothesis bound on
L(h∗). The quantity Eρ[L(h) −L(h∗)] is known as excess loss. The advantage of this approach
is that when L(h∗) is a good approximation of Eρ[L(h)], the excess loss has lower variance than
the plain loss Eρ[L(h)] and, therefore, is more efﬁcient to bound, whereas the single-hypothesis
bound on L(h∗) does not involve the KL(ρ∥π) term. Therefore, it is generally beneﬁcial to use
excess losses in combination with data-informed priors. However, as with the previous approach,
sacriﬁcing S1 to learn π and h∗means that the denominator in the bounds (n in Theorem 1) reduces
to the size of S2, and it does not always pay off. (We note that the excess loss is not binary and
not in the [0, 1] interval, and in order to exploit small variance it is actually necessary to apply a
PAC-Bayes-Empirical-Bernstein-style inequality (Tolstikhin and Seldin, 2013, Mhammedi et al.,
2019, Wu et al., 2021) or the PAC-Bayes-split-kl inequality (Wu and Seldin, 2022) rather than
Theorem 1, but the point about reduced sample size still applies.)

Recursive PAC-Bayes (new)
We introduce the following decomposition of the loss

Eρ[L(h)] = Eρ[L(h) −γEπ[L(h′)]] + γEπ[L(h′)].
(1)

3


---Page Break---
As before, we decompose S into two disjoint sets S = S1 ∪S2. We make the following major
observations:

• The quantity Eπ[L(h′)] on the right is “of the same kind” as Eρ[L(h)] on the left.

• We can take an uninformed prior π0 and apply Theorem 1 (or any other suitable PAC-Bayes
bound) to bound Eπ[L(h′)]. (The KL term in the bound on Eπ[L(h′)] will be KL(π∥π0).)

• We can restrict π to depend only on S1, but still use all the data S in calculation of the PAC-
Bayes bound on Eπ[L(h′)], because π is a posterior relative to π0, and a posterior is allowed
to depend on all the data, and in particular on any subset of the data. Therefore, the empirical
loss Eπ[ˆL(h′, S)] can be computed on all the data S, and the denominator of the bound in
Theorem 1 can be the size of S, and not the size of S2. This is what we call preservation of
conﬁdence information on π, because all the data S are used to construct a conﬁdence bound
on Eπ[L(h′)], and not just S2. This is in contrast to the bound on L(h∗) in the approach of
Mhammedi et al. (2019), which only allows to use S2 for bounding L(h∗). Note that while
we use all the data S in calculation of the bound, we only use S1 and Eπ[ˆL(h′, S1)] in the
construction of π. Nevertheless, we can still use the knowledge that we will have n samples
when we reach the estimation phase, i.e., when constructing π we can leave the denominator of
the bound at n, allowing more aggressive deviation from π0.

• If we restrict π to depend only on S1, then it is a valid prior for estimation of any posterior
quantity Eρ[·] based on S2. Thus, if we also restrict γ to depend only on S1, we can use any PAC-
Bayes-Empirical-Bernstein-style inequality or the PAC-Bayes-split-kl inequality to estimate the
excess loss Eρ[L(h) −γEπ[L(h′)]] based on S2, i.e., based on Eρ[ˆL(h, S2) −γEπ[ˆL(h′, S2)]].
If γEπ[L(h′)] is a good approximation of Eρ[L(h)] and Eρ[L(h)] is not close to zero, then the
excess loss Eρ[L(h) −γEπ[L(h′)]] is more efﬁcient to bound than the plain loss Eρ[L(h)].

• In general, since Eρ[L(h)] is expected to improve on Eπ[L(h′)], it is natural to set γ < 1.
However, γ is not allowed to depend on S2, because otherwise ˆL(h, S2) −γEπ[ˆL(h′, S2)]
becomes a biased estimate of L(h) −γEπ[L(h′)]. We discuss the choice of γ in more detail
when we present the bound and the experiments.

• Biggs and Guedj (2023) have proposed a sequential martingale-style evaluation of a martingale
version of Eρ[L(h) −L(h∗)] and L(h∗) in the approach of Mhammedi et al., but it has not
been shown to yield signiﬁcant improvements yet. The same “martingalization” can be directly
applied to our decomposition, but to keep things simple we stay with the basic decomposition.

• Finally, we note that we can split S1 further and apply (1) recursively to bound Eπ[L(h′)].

To set notation for recursive decomposition, we use π0, π1, . . . , πT to denote a sequence of distribu-
tions on H, where π0 is an uninformed prior and πT = ρ is the ﬁnal posterior. We use γ2, . . . , γT to
denote a sequence of coefﬁcients. For t ≥2 we then have the recursive decomposition

Eπt[L(h)] = Eπt[L(h) −γtEπt−1[L(h)]] + γtEπt−1[L(h)].
(2)

To construct π1, . . . , πT we split the data S into T non-overlapping subsets, S = S1 ∪· · · ∪ST . We
restrict πt to depend on U train
t
= St
s=1 Ss only, and we use U val
t
= ST
s=t Ss to estimate (recursively)
Eπt[L(h)] (see Figure 1 in Appendix A). Note that St is used both for construction of πt and for
estimation of Eπt[L(h)] (it is both in U train
t
and U val
t
), resulting in efﬁcient use of the data. It
is possible to use any standard PAC-Bayes bound, e.g., Theorem 1, to bound Eπ1[L(h)], and any
PAC-Bayes-Empirical-Bernstein-style bound or the PAC-Bayes-split-kl bound to bound the excess
losses Eπt[L(h) −γtEπt−1[L(h)]]. The excess losses take more than three values, so in the next
section we present a generalization of the PAC-Bayes-split-kl inequality to general discrete random
variables, which may be of independent interest. The Recursive PAC-Bayes bound is presented in
Section 4.

3
Split-kl and PAC-Bayes-split-kl inequalities for discrete random variables

The kl inequality is one of the tightest concentration of measure inequalities for binary random vari-
ables. Letting kl−1,+(ˆp, ε) := max {p : p ∈[0, 1] and kl(ˆp∥p) ≤ε} denote the upper inverse of kl
and kl−1,−(ˆp, ε) := min {p : p ∈[0, 1] and kl(ˆp∥p) ≤ε} the lower inverse, it states the following.

4


---Page Break---
Theorem 2 (kl Inequality (Langford, 2005, Foong et al., 2021, 2022)). Let Z1, · · · , Zn be indepen-
dent random variables bounded in the [0, 1] interval and with E [Zi] = p for all i. Let ˆp = 1

n
Pn
i=1 Zi
be the empirical mean. Then, for any δ ∈(0, 1):

P

p ≥kl−1,+

ˆp, 1

n ln 1

δ


≤δ
;
P

p ≤kl−1,−

ˆp, 1

n ln 1

δ


≤δ.

While the kl inequality is tight for binary random variables, it is loose for random variables taking
more than two values due to its inability to exploit small variance. To address this shortcoming Wu
and Seldin (2022) have presented the split-kl and PAC-Bayes-split-kl inequalities for ternary random
variables. Ternary random variables naturally appear in a variety of applications, including analysis
of excess losses, certain ways of analysing majority votes, and in learning with abstention. The
bound of Wu and Seldin is based on decomposition of a ternary random variable into a pair of binary
random variables and application of the kl inequality to each of them. Their decomposition yields a
tight bound in the binary and ternary case, but loose otherwise. The same decomposition was used by
Biggs and Guedj (2023) to derive a slight variation of the inequality, with the same limitations. We
present a novel decomposition of discrete random variables into a superposition of binary random
variables. Unlike the decomposition of Wu and Seldin, which only applies in the ternary case, our
decomposition applies to general discrete random variables. By combining it with kl bounds for
the binary elements we obtain a tight bound. The decomposition is presented formally below and
illustrated graphically in Figure 2 in Appendix A.

3.1
Split-kl inequality

Let Z ∈{b0, . . . , bK} be a (K + 1)-valued random variable with b0 < b1 < · · · < bK. For
j ∈{1, . . . , K} deﬁne Z|j = 1(Z ≥bj) and αj = bj −bj−1. Then Z = b0 + PK
j=1 αjZ|j. For
a sequence Z1, . . . , Zn of (K + 1)-valued random variables with the same support, let Zi|j =
1(Zi ≥bj) denote the elements of binary decomposition of Zi.
Theorem 3 (Split-kl inequality for discrete random variables). Let Z1, . . . , Zn be i.i.d. random
variables taking values in {b0, . . . , bK} with E [Zi] = p for all i. Let ˆp|j = 1

n
Pn
i=1 Zi|j. Then for
any δ ∈(0, 1):

P



p ≥b0 +

K
X

j=1
αj kl−1,+

ˆp|j, 1

n ln K

δ



≤δ.

Proof. Let p|j = E

ˆp|j

, then p = b0 + PK
j=1 αjp|j and

P



p ≥b0 +

K
X

j=1
αj kl−1,+

ˆp|j, 1

n ln K

δ



≤P

∃j : p|j ≥kl−1,+

ˆp|j, 1

n ln K

δ


≤δ,

where the ﬁrst inequality is by the decomposition of p and the second inequality is by the union
bound and Theorem 2.

3.2
PAC-Bayes-Split-kl inequality

Let f : H × Z →{b0, . . . , bK} be a (K + 1)-valued loss function. (To connect it to the earlier
examples, in the binary prediction case we would have Z = X × Y with elements Z = (X, Y )
and f(h, Z) = ℓ(h(X), Y ), but we will need a more general space Z later.) For j ∈{1, . . . , K}
let f|j(·, ·) = 1(f(·, ·) ≥bj). Let DZ be an unknown distribution on Z. For h ∈H let F(h) =
EDZ[f(h, Z)] and F|j(h) = EDZ[f|j(h, Z)]. Let S = {Z1, . . . , Zn} be an i.i.d. sample according
to DZ and ˆF|j(h, S) = 1

n
Pn
i=1 f|j(h, Zi).
Theorem 4 (PAC-Bayes-Split-kl Inequality). For any distribution π on H that is independent of S
and any δ ∈(0, 1):

P



∃ρ ∈P : Eρ[F(h)] ≥b0 +

K
X

j=1
αj kl−1,+
 

Eρ[ ˆF|j(h, S)], KL(ρ∥π) + ln 2K√n

δ
n

!

≤δ,

where P is the set of all possible probability distributions on H that can depend on S.

5


---Page Break---
Proof. We have f(·, ·) = b0 + PK
j=1 αjf|j(·, ·) and F(h) = b0 + PK
j=1 αjF|j(h). Therefore,

P



∃ρ ∈P : Eρ[F(h)] ≥b0 +

K
X

j=1
αj kl−1,+
 

Eρ[ ˆF|j(h, S)], KL(ρ∥π) + ln 2K√n

δ
n

!



≤P

 

∃ρ ∈P and ∃j : Eρ[F|j(h)] ≥kl−1,+
 

Eρ[ ˆF|j(h, S)], KL(ρ∥π) + ln 2K√n

δ
n

!!

≤δ,

where the ﬁrst inequality is by the decomposition of F and the second inequality is by the union
bound and application of Theorem 1 to F|j (note that f|j is a zero-one loss function).

4
Recursive PAC-Bayes bound

Now we derive a Recursive PAC-Bayes bound based on the loss decomposition in equation (2). We
aim to bound Eπt[L(h) −γtEπt−1[L(h′)]], which we denote by

Fγt,πt−1(h) = L(h) −γtEπt−1[L(h′)] = ED×πt−1[ℓ(h(X), Y ) −γtℓ(h′(X), Y )],

where D × πt−1 is a product distribution on X × Y × H and h′ ∈H is sampled according to πt−1.
We further deﬁne

fγt(h, (X, Y, h′)) = ℓ(h(X), Y ) −γtℓ(h′(X), Y ) ∈{−γt, 0, 1 −γt, 1} ,

then Fγt,πt−1(h) = ED×πt−1[fγt(h, (X, Y, h′))]. In order to apply Theorem 4, we represent fγt as a
superposition of binary functions. For this purpose we let

bt|0, bt|1, bt|2, bt|3
	
= {−γt, 0, 1 −γt, 1}
and deﬁne fγt|j(h, (X, Y, h′))
=
1
 
fγt(h, (X, Y, h′)) ≥bt|j

.
We let Fγt,πt−1|j(h)
=
ED×πt−1[fγt|j(h, (X, Y, h′))], then Fγt,πt−1(h) = −γt + P3
j=1(bt|j −bt|j−1)Fγt,πt−1|j(h).

Now we construct an empirical estimate of Fγt,πt−1|j(h). We ﬁrst let ˆπt−1 =

hπt−1
1
, hπt−1
2
, . . .
	
be
a sequence of prediction rules sampled independently according to πt−1. We deﬁne U val
t
◦ˆπt−1 =
 
Xi, Yi, hπt−1
i

: (Xi, Yi) ∈U val
t
	
.
In words, for every sample (Xi, Yi) ∈U val
t
we sample
a prediction rule hπt−1
i
according to πt−1 and put the triplet (Xi, Yi, hπt−1
i
) in U val
t
◦ˆπt−1.
The triplets (Xi, Yi, hπt−1
i
) correspond to the random variables Z in Theorem 4.
We note
that |U val
t
| = |U val
t
◦ˆπt−1|, and we let nval
t
= |U val
t
|. We deﬁne the empirical estimate of
Fγt,πt−1|j(h) as ˆFγt|j(h, U val
t
◦ˆπt−1) =
1
nval
t
P

(X,Y,h′)∈U val
t
◦ˆπt−1 fγt|j(h, (X, Y, h′)). Note that

ED×πt−1[ ˆFγt|j(h, U val
t
◦ˆπt−1)] = Fγt,πt−1|j(h), therefore, we can use Theorem 4 to bound
Eπt[Fγt,πt−1(h)] using its empirical estimates. We are now ready to state the bound.
Theorem 5 (Recursive PAC-Bayes Bound). Let S = S1 ∪· · · ∪ST be an i.i.d. sample split in an
arbitrary way into T non-overlapping subsamples, and let U train
t
= St
s=1 Ss and U val
t
= ST
s=t Ss.
Let nval
t
= |U val
t
|. Let π∗
0, π∗
1, . . . , π∗
T be a sequence of distributions on H, where π∗
t is allowed to
depend on U train
t
, but not the rest of the data. Let γ2, . . . , γT be a sequence of coefﬁcients, where
γt is allowed to depend on U train
t−1 , but not the rest of the data. For t ∈{1, . . . , T} let Pt be a set of
distributions on H, which are allowed to depend on U train
t
. Then for any δ ∈(0, 1):

P(∃t ∈{1, . . . , T} and πt ∈Pt : Eπt[L(h)] ≥Bt(πt)) ≤δ,

where Bt(πt) is a PAC-Bayes bound on Eπt[L(h)] deﬁned recursively as follows. For t = 1

B1(π1) = kl−1,+
 

Eπ1[ˆL(h, S)], KL(π1∥π∗
0) + ln 2T √n

δ
n

!

.

For t ≥2 we let Et(πt, γt) denote a PAC-Bayes bound on Eπt[L(h) −γtEπ∗
t−1[L(h′)]] given by

Et(πt, γt) = −γt+

3
X

j=1
(bt|j−bt|j−1) kl−1,+




Eπt
h
ˆFγt|j(h, U val
t
◦ˆπ∗
t−1)
i
, KL(πt∥π∗
t−1) + ln
6T
q

nval
t
δ
nval
t






and then
Bt(πt) = Et(πt, γt) + γt Bt−1(π∗
t−1).
(3)

6


---Page Break---
Proof. By Theorem 1 we have P(∃π1 ∈P1 : Eπ1[L(h)] ≥B1(π1)) ≤δ

T . Further, by Theorem 4

for t ∈{2, . . . , T} we have P

∃πt ∈Pt : Eπt[L(h) −γtEπ∗
t−1[L(h′)]] ≥Et(πt, γt)

≤
δ
T . The
theorem follows by a union bound and the recursive decomposition of the loss (2).

Discussion

• Note that π∗
1, . . . , π∗
T can be constructed sequentially, but π∗
t can only be constructed
based on the data in U train
t
, meaning that in the construction of π∗
t we can only rely on
Eπt
h
ˆFγt|j(h, St ◦ˆπt−1)
i
, but not on Eπt
h
ˆFγt|j(h, U val
t
◦ˆπt−1)
i
. Also note that St is part

of both U train
t
and U val
t
(see Figure 1 in Appendix A for a graphical illustration). In other words,
when we evaluate the bounds we can use additional data. And even though the additional data
can only be used in the evaluation stage, we can still use the knowledge that we will get more
data for evaluation when we construct π∗
t . For example, we can take

π∗
1 = arg min
π kl−1,+
 

Eπ[ˆL(h, S1)], KL(π∥π∗
0) + ln 2T √n

δ
n

!

(4)

and for t ≥2

π∗
t = arg min
π

3
X

j=1
(bt|j−bt|j−1) kl−1,+




Eπ
h
ˆFγt|j(h, St ◦ˆπ∗
t−1)
i
, KL(π∥π∗
t−1) + ln
6T
q

nval
t
δ
nval
t




.

(5)
The empirical losses above are calculated on St corresponding to π∗
t , but the sample sizes nval
t
correspond to the size of the validation set U val
t
rather than the size of St. This allows to be more
aggressive in deviating with π∗
t from π∗
t−1 by sustaining larger KL(π∗
t ∥π∗
t−1) terms.

• Similarly, γ2, . . . , γT can also be constructed sequentially, as long as γt only depends on U train
t−1
(otherwise ˆFγt|j(h, St ◦ˆπ∗
t−1) becomes a biased estimate of Fγt,π∗
t−1|j(h)).

• We naturally want to have improvement over recursion steps, meaning Bt(π∗
t ) < Bt−1(π∗
t−1).
Plugging this into (3), we obtain E(π∗
t , γt) + γtBt−1(π∗
t−1) < Bt−1(π∗
t−1), which implies that
we want γt to be sufﬁciently small to satisfy γt < 1 −
Et(π∗
t ,γt)
Bt−1(π∗
t−1). At the same time, γt should
be non-negative. Therefore, improvement over recursion steps can only be maintained as long as
Et(π∗
t , γt) < Bt−1(π∗
t−1). We note that γt Bt−1(π∗
t−1) term in (3) is linearly increasing in γt,
whereas E(π∗
t , γt) is decreasing in γt. The value of γt that minimizes the trade-off depends on
the data. Even though it is not allowed to use U val
t
for tuning γt, it is possible to take a grid of
values of γt and a union bound over the grid, and then select the best value from the grid based
the value of the bound evaluated on U val
t
.

5
Experiments

In this section, we provide an empirical comparison of our Recursive PAC-Bayes (RPB) procedure to
the following prior work: i) Uninformed priors (Uninformed), (Dziugaite and Roy, 2017); ii) Data-
informed priors (Informed) (Ambroladze et al., 2007, Pérez-Ortiz et al., 2021); iii) Data-informed
prior + excess loss (Informed + Excess) (Mhammedi et al., 2019, Wu and Seldin, 2022). All the
experiments were run on a laptop. The source code for replicating the experiments is available at
Github1.

We start with describing the details of the optimization procedure, and then present the results.

5.1
Details of the optimization and evaluation procedure

We constructed π∗
1, . . . , π∗
T sequentially using the optimization objective, (4) for π∗
1 and (5) for π∗
2 to
π∗
T , and computed the bound using the recursive procedure in Theorem 5. There are a few technical
details concerning convexity of the optimization procedure and inﬁnite size of the set of prediction
rules H that we address next.

1https://github.com/pyijiezhang/rpb

7


---Page Break---
5.1.1
Convexiﬁcation of the loss functions

The functions fγt|j(h, (X, Y, h′)) deﬁned in Section 4 are non-convex and non-differentiable:
fγt|j(h, (X, Y, h′)) = 1
 
fγt(h, (X, Y, h′)) ≥bt|j

= 1
 
ℓ(h(X), Y ) −γtℓ(h′(X), Y ) ≥bt|j

. In
order to facilitate optimization, we approximate the external indicator function 1(z ≥z0) by a sig-
moid function ω(z; c1, z0) = (1 + exp(c1(z −z0)))−1 with a ﬁxed parameter c1 > 0 speciﬁed in
Appendix B.3.

Furthermore, since the zero-one loss ℓ(h(X), Y ) is also non-differentiable, we adopt the cross-entropy
loss, as in most modern training procedures (Pérez-Ortiz et al., 2021). Speciﬁcally, for a k-class
classiﬁcation problem, let h : X →Rk represent the function implemented by the neural network,
assigning each class a real value. Let u = h(X) be the assignment, with ui being the i-th value of
the vector. To convert this real-valued vector into a probability distribution over classes, we apply
the softmax function σ : Rk →∆k−1, where σ(u)i = exp(c2ui)/ P

j exp(c2uj) for some c2 > 0
for each entry. The cross-entropy loss ℓce : Rk × [k] →R is deﬁned by ℓce(u, Y ) = −log(σ(u)Y ).
However, since this loss is unbounded, whereas the PAC-Bayes-kl bound requires losses within
[0, 1], we enforce a [0, 1]-valued cross-entropy loss by mixing the output distribution with a uniform
distribution σ(u), i.e., ˜σ(u)i = (1 −pmin)σ(u)i + pmin/k for all i ∈[k] and for some pmin > 0,
and then rescaling it to [0, 1] by taking ˜ℓce(u, Y ) = −log(˜σ(u)Y )/ log(k/pmin).

We emphasize that in the evaluation of the bound (using Theorem 5), we directly compute the
zero-one loss and the fγt|j functions without employing the approximations.

5.1.2
Relaxation of the PAC-Bayes-kl bound

The PAC-Bayes-kl bound is often criticized for being unfriendly to optimization (Rodríguez-Gálvez
et al., 2024). Therefore, several relaxations have been proposed, including the PAC-Bayes-classic
bound (McAllester, 1999), the PAC-Bayes-λ bound (Thiemann et al., 2017), and the PAC-Bayes-
quadratic bound (Rivasplata et al., 2019, Pérez-Ortiz et al., 2021), among others. In our optimization
we have adopted the bound of McAllester (1999) instead of the kl-based bounds in Equation (5).

We again emphasize that in the evaluation of the bound we used the kl-based bounds in Theorem 5.

5.1.3
Estimation of Eπ[·]

Due to the inﬁnite size of H and lack of a closed-form expression for Eπ1[ˆL(h, S)] and
Eπt[ ˆFγt|j(h, U val
t
◦ˆπ∗
t−1)] appearing in Theorem 5, we approximate them by sampling (Pérez-
Ortiz et al., 2021). For optimization, we sample one classiﬁer for each mini-batch during stochastic
gradient descent. For evaluation, we sample one classiﬁer for each data in the corresponding evalua-
tion dataset. Due to approximation of the empirical quantities the ﬁnal bound in Theorem 5 requires
an additional concentration bound. (We note that the extra bound is only required for computation
of the ﬁnal bound, but not for optimization of ˆπ∗
t .) Speciﬁcally, let ˆπ∗
t = {hπ1
1 , hπt
2 , . . . , hπt
m} be m
prediction rules sampled independently according to πt. Then for any function f(h) taking values in
[0, 1] (which is the case for ˆL(h, S) and ˆFγt|j(h, U val
t
◦ˆπ∗
t−1)) and δ′ ∈(0, 1) we have

P

 

Eπ∗
t [f(h)] ≥kl−1,+
 
1
m

m
X

i=1
f(hπ∗
t
i ), 1

m log 1

δ′

!!

≤δ′.

It is worth noting that Eπ∗
t [f(h)] is evaluated for a ﬁxed π∗
t , meaning that there is no selection
involved, and therefore no KL term appears in the bound above. We, of course, take a union bound
over all the quantities being estimated.

5.2
Experimental results

We evaluated our approach and compared it to prior work using multi-class classiﬁcation tasks on
MNIST (LeCun and Cortes, 2010) and Fashion MNIST (Xiao et al., 2017) datasets, both with 60000
training data. The experimental setup was based on the work of Dziugaite and Roy (2017) and
Pérez-Ortiz et al. (2021). Similar to them we used Gaussian distributions for all the priors and
posteriors, modeled by probabilistic neural networks. Technical details are provided in B.

8


---Page Break---
The empirical evaluation is presented in Table 1. For the Uninformed approach, we trained and
evaluated the bound using the entire training dataset directly. For the other two baseline methods,
Informed and Informed + Excess Loss, we used half of the training data to train the informed prior
and an ERM h∗for the excess loss, and the other half to learn the posterior. For our Recursive
PAC-Bayes, we chose γt = 1/2 for all t, and conducted experiments with T = 2, 4, 6, 8 to study
the impact of recursion depth. (Each value of T corresponded to a separate run of the algorithm
and a separate evaluation of the bound, i.e., they should not be seen as successive reﬁnements.)
We applied a geometric data split. Speciﬁcally, for T = 2 the split was (30000, 30000) points;
for T = 4, it was (7500, 7500, 15000, 30000); for T = 6 it was (1875, 1875, 3750, 7500, 15000,
30000); and for T = 8, it was (469, 469, 937, 1875, 3750, 7500, 15000, 30000). This approach
allowed the early recursion steps, which had fewer data points, to efﬁciently learn the prior, while
preserving enough data for ﬁne-tuning in the later steps. Note that with this approach the value of
nval
t
= |U val
t
| = PT
s=t |Ss|, which is in the denominator of the bounds in Theorem 5, is at least n

2 .

Table 1: Comparison of the classiﬁcation loss of the ﬁnal posterior ρ on the entire training data,
Eρ[ˆL(h, S)] (Train 0-1), and on the testing data, Eρ[ˆL(h, Stest)] (Test 0-1), and the corresponding
bounds for each method on MNIST and Fashion MNIST. We report the mean and one standard
deviation over 5 repetitions. “Unif.” abbreviates the Uniform approach, “Inf.” the Informed, “Inf. +
Ex.” the Informed + Excess Loss, and “RPB” the Recursive PAC-Bayes.

MNIST
Fashion MNIST
Train 0-1
Test 0-1
Bound
Train 0-1
Test 0-1
Bound
Uninf.
.343 (2e-3)
.335 (3e-3)
.457 (2e-3)
.382 (2e-3)
.384 (2e-3)
.464 (2e-3)
Inf.
.377 (8e-4)
.371 (6e-3)
.408 (9e-4)
.412 (1e-3)
.413 (6e-3)
.440 (1e-3)
Inf. + Ex.
.157 (2e-3)
.151 (3e-3)
.192 (2e-3)
.280 (4e-3)
.285 (5e-3)
.342 (6e-3)
RPB T = 2
.143 (2e-3)
.139 (3e-3)
.321 (3e-3)
.257 (3e-3)
.266 (5e-3)
.404 (3e-3)
RPB T = 4
.112 (1e-3)
.109 (1e-3)
.203 (8e-4)
.203 (2e-3)
.213 (3e-3)
.293 (1e-3)
RPB T = 6
.103 (1e-3)
.101 (1e-3)
.166 (1e-3)
.186 (4e-4)
.198 (1e-3)
.255 (1e-3)
RPB T = 8
.101 (1e-3)
.097 (2e-3)
.158 (2e-3)
.181 (1e-3)
.192 (3e-3)
.242 (1e-3)

Table 1 shows that even with only T = 2, which corresponds to the data split used in the Informed
and the Informed + Excess Loss approaches, RPB achieves better test performance than prior work.
As the recursion deepens, further improvements in both the test error and the bound are observed. We
note that while the bound for T = 2 is looser compared to the Informed + Excess Loss method, deeper
recursion yields bounds that are tighter. Overall, deep recursion provides substantial improvements in
the bound and the test error relative to prior work.

Tables 2 and 3 provide a glimpse into the training progress of RPB with T = 8 by showing the
evolution of the key quantities along the recursive process. Similar tables for other values of T
are provided in Appendix B.4, along with training details for other methods. The tables show an
impressive reduction of the KL term and signiﬁcant improvement of the bound as the recursion
proceeds, demonstrating effectiveness of the approach.

Table 2: Insight into the training process of the Recursive PAC-Bayes for T = 8 on MNIST. The
table shows the evolution of Et(π∗
t , γt), Bt(π∗
t ), and other quantities as the training progresses with t.
We deﬁne ˆFγt(h, U val
t
◦ˆπt−1) = −γt + P3
j=1(bt|j −bt|j−1) ˆFγt|j(h, U val
t
◦ˆπt−1).

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.009 (3e-4)
.612 (9e-3)
.532 (.011)
2
59532
-0.046 (4e-3)
.031 (1e-3)
.114 (2e-3)
.421 (5e-3)
.215 (7e-3)
3
59063
.040 (3e-3)
.013 (9e-4)
.125 (3e-3)
.336 (2e-3)
.146 (3e-3)
4
58125
.049 (1e-3)
.005 (3e-4)
.099 (1e-3)
.267 (7e-4)
.120 (2e-3)
5
56250
.052 (4e-4)
.002 (1e-4)
.083 (1e-3)
.217 (1e-3)
.111 (2e-3)
6
52500
.051 (1e-3)
.001 (4e-5)
.076 (1e-3)
.185 (1e-3)
.104 (2e-3)
7
45000
.050 (1e-3)
8e-4 (6e-5)
.073 (1e-3)
.166 (1e-3)
.099 (1e-3)
8
30000
.050 (1e-3)
6e-4 (4e-5)
.074 (1e-3)
.158 (2e-3)
.097 (2e-3)

9


---Page Break---
Table 3: Insight into the training process of the Recursive PAC-Bayes for T = 8 on Fashion MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.003 (7e-5)
.733 (7e-3)
.686 (8e-3)
2
59532
-0.043 (8e-3)
.023 (6e-4)
.104 (9e-3)
.470 (8e-3)
.309 (7e-3)
3
59063
.083 (4e-3)
.008 (3e-4)
.161 (3e-3)
.396 (4e-3)
.242 (1e-3)
4
58125
.090 (3e-3)
.004 (5e-4)
.142 (4e-3)
.341 (5e-4)
.216 (5e-3)
5
56250
.093 (3e-3)
.001 (2e-4)
.126 (3e-3)
.297 (4e-3)
.204 (4e-3)
6
52500
.090 (1e-3)
6e-4 (6e-5)
.117 (1e-3)
.265 (1e-3)
.195 (3e-3)
7
45000
.090 (1e-3)
4e-4 (2e-5)
.115 (1e-3)
.248 (1e-3)
.195 (5e-4)
8
30000
.090 (1e-3)
4e-4 (1e-5)
.117 (1e-3)
.242 (1e-3)
.192 (3e-3)

6
Discussion

We have presented the ﬁrst PAC-Bayesian bound that supports sequential prior updates and preserves
conﬁdence information on the prior. The work closes a long-standing gap between Bayesian and
Frequentist learning by making sequential data processing and sequential updates of prior knowledge
meaningful and beneﬁcial in the frequentist framework, as it has always been in the Bayesian
framework. We have shown that apart from theoretical beauty the approach is highly beneﬁcial in
practice.

The Recursive PAC-Bayes framework is extremely rich and powerful, and leads to numerous direc-
tions for future research, some of which we brieﬂy sketch next.

• The decomposition in (2) applies to any loss function, including unbounded losses. It would be
interesting to ﬁnd additional applications to it.
• While we have restricted ourselves to the zero-one loss function to illustrate the use of PAC-
Bayes-split-kl, the results can be directly generalized to any bounded loss function by replacing
PAC-Bayes-split-kl with PAC-Bayes-Empirical-Bernstein or PAC-Bayes-Unexpected-Bernstein,
and deriving the corresponding analogue of Theorem 5 (which is straightforward).
• We have shown that the bound works well with geometric split of the data, but there are many
other ways to split the data which could be studied.
• There is also a lot of space for experimentation with optimization of γt.
• It would be interesting to study how the bound will perform in sequential learning settings, where
the data arrives sequentially, and thus the partition is dictated externally.
• There are many interesting research directions from the computational perspective. We note that
for base models with linear computational complexity (e.g., neural networks) the overhead of
recursion is relatively small and optimization time of Recursive PAC-Bayes is comparable to
processing all data at once or in two chunks (as in data-dependent priors). For base models with
superlinear computational complexity (e.g., kernel SVMs) sequential training of several small
models in the recursion may actually be cheaper than training a big model based on all the data.
Moreover, since the bound in Theorem 5 holds for any sequence of distributions π∗
0, π∗
1, . . . , π∗
T ,
the optimization in equation (5) is allowed to be approximate. Considering that the improvement
of the bounds and the test loss relative to prior work was very signiﬁcant, there is space to look at
the trade-off between statistical power and computational complexity. Namely, it may potentially
be possible to relax the approximation of arg min in equation (5) to gain computational speed-up
at the cost of only a small compromise on the bounds and test losses.
• We note that it is possible to start the recursion at π0. Namely, it is possible to use, for example,
Theorem 2 to bound Eπ0[L(h)] using all the data, and apply the recursive decomposition (2)
starting from π1. Whether this would yield an advantage relative to starting the recursion at π1,
as we did, remains to be studied.

Acknowledgments and Disclosure of Funding

YW acknowledges support from the Novo Nordisk Foundation, grant number NNF21OC0070621.
YZ acknowledge Ph.D. funding from Novo Nordisk A/S. BECA acknowledges funding from the
ANR grant project BACKUP ANR-23-CE40-0018-01.

10


---Page Break---
References

Amiran Ambroladze, Emilio Parrado-Hernández, and John Shawe-Taylor. Tighter PAC-Bayes bounds.
In Advances in Neural Information Processing Systems (NeurIPS), 2007.

Felix Biggs and Benjamin Guedj. Tighter PAC-Bayes generalisation bounds by leveraging example
difﬁculty. In Proceedings on the International Conference on Artiﬁcial Intelligence and Statistics
(AISTATS), 2023.

Ben Chugg, Hongjian Wang, and Aaditya Ramdas. A uniﬁed recipe for deriving (time-uniform)
PAC-Bayes bounds. Journal of Machine Learning Research, 24(372), 2023.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley Series in Telecommu-
nications and Signal Processing, 2nd edition, 2006.

Gintare Karolina Dziugaite and Daniel M. Roy. Computing nonvacuous generalization bounds for
deep (stochastic) neural networks with many more parameters than training data. In Proceedings
of the Conference on Uncertainty in Artiﬁcial Intelligence (UAI), 2017.

Andrew Foong, Wessel Bruinsma, David Burt, and Richard Turner. How tight can pac-bayes be in
the small data regime? In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Andrew Y. K. Foong, Wessel P. Bruinsma, and David R. Burt. A note on the chernoff bound for
random variables in the unit interval. arXiv preprint arXiv.2205.07880, 2022.

Maxime Haddouche and Benjamin Guedj. PAC-Bayes generalisation bounds for heavy-tailed losses
through supermartingales. Transactions on Machine Learning Research Journal, 2023.

Kyoungseok Jang, Kwang-Sung Jun, Ilja Kuzborskij, and Francesco Orabona. Tighter PAC-Bayes
bounds through coin-betting. In Proceedings of the Conference on Learning Theory (COLT), 2023.

John Langford. Tutorial on practical prediction theory for classiﬁcation. Journal of Machine Learning
Research, 6, 2005.

Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL http://yann.
lecun.com/exdb/mnist/.

Andreas Maurer. A note on the PAC-Bayesian theorem. arXiv preprint cs/0411099, 2004.

David McAllester. Some PAC-Bayesian theorems. In Proceedings of the Conference on Learning
Theory (COLT), 1998.

David McAllester. Some PAC-Bayesian theorems. Machine Learning, 37, 1999.

Zakaria Mhammedi, Peter Grünwald, and Benjamin Guedj. PAC-Bayes un-expected Bernstein
inequality. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

María Pérez-Ortiz, Omar Rivasplata, John Shawe-Taylor, and Csaba Szepesvári. Tighter risk certiﬁ-
cates for neural networks. Journal of Machine Learning Research, 2021.

Omar Rivasplata, Vikram M Tankasali, and Csaba Szepesvari. Pac-bayes with backprop. arXiv
preprint arXiv:1908.07380, 2019.

Borja Rodríguez-Gálvez, Ragnar Thobaben, and Mikael Skoglund. More PAC-Bayes bounds: From
bounded losses, to losses with general tail behaviors, to anytime validity. Journal of Machine
Learning Research, 25(110), 2024.

Matthias Seeger. PAC-Bayesian generalization error bounds for Gaussian process classiﬁcation.
Journal of Machine Learning Research, 3, 2002.

John Shawe-Taylor and Robert C. Williamson. A PAC analysis of a Bayesian estimator. In Proceed-
ings of the Conference on Learning Theory (COLT), 1997.

Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly quasiconvex
PAC-Bayesian bound. In Proceedings of the International Conference on Algorithmic Learning
Theory (ALT), 2017.

11


---Page Break---
Ilya Tolstikhin and Yevgeny Seldin. PAC-Bayes-Empirical-Bernstein inequality. In Advances in
Neural Information Processing Systems (NeurIPS), 2013.

Yi-Shan Wu and Yevgeny Seldin. Split-kl and PAC-Bayes-split-kl inequalities for ternary random
variables. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Yi-Shan Wu, Andres Masegosa, Stephan Sloth Lorenzen, Christian Igel, and Yevgeny Seldin.
Chebyshev-cantelli pac-bayes-bennett inequality for the weighted majority vote. In Advances in
Neural Information Processing Systems (NeurIPS), 2021.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for bench-
marking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

12


---Page Break---
A
Illustrations

In this appendix we provide graphical illustrations of the basic concepts presented in the paper.

Figure 1: Evolution of PAC-Bayes. The ﬁgure shows how data are used by different methods.
Dark yellow shows data used directly for optimization of the indicated quantities. Light yellow
shows data involved indirectly through dependence on the prior. Light green shows data used for
estimation of the indicated quantities. In Recursive PAC-Bayes data are released and used sequentially
chunk-by-chunk, as indicated by the dashed lines. For example, in the T = 4 case Eπ1[L(h)] is ﬁrst
evaluated on S1 to construct π1 and γ2, then in the ﬁrst recursion step on S1 ∪S2, in the second step
on S1 ∪S2 ∪S3, and in the last step on all S.

13


---Page Break---
Figure 2: Decomposition of a discrete random variable into a superposition of binary random
variables. The ﬁgure illustrates a decomposition of a discrete random variable Z with domain
of four values b0 < b1 < b2 < b3 into a superposition of three binary random variables, Z =
b0 + P3
j=1 αjZ|j. A way to think about the decomposition is to compare it to a progress bar. In
the illustration Z takes value b2, and so the random variables Z|1 and Z|2 corresponding to the ﬁrst
two segments “light up” (take value 1), whereas the random variable Z|3 corresponding to the last
segment remains “turned off” (takes value 0). The value of Z equals the sum of the lengths αj of the
“lighted up” segments.

B
Experimental details

In this section, we provide the details of the datasets in Appendix B.1, our neural network architectures
in Appendix B.2, and other details in Appendix B.3. We provide further statistics for all the methods
on both datasets in Appendix B.4.

B.1
Datasets

We perform our evaluation on two datasets, MNIST (LeCun and Cortes, 2010) and Fashion MNIST
(Xiao et al., 2017). We will introduce these two datasets in the following.

B.1.1
MNIST

The MNIST (Modiﬁed National Institute of Standards and Technology) dataset is one of the most
renowned and widely used datasets in the ﬁeld of machine learning, particularly for training and
testing in the domain of image processing and computer vision. It consists of a large collection of
handwritten digit images, spanning the numbers 0 through 9.

The MNIST dataset comprises a total of 70,000 grayscale images of handwritten digits, where the
training set has 60,000 images and the test set has 10,000 images. Each image in the dataset is 28x28
pixels, resulting in a total of 784 pixels per image. The images are in grayscale, with pixel values
ranging from 0 (black) to 255 (white). Each image is associated with a label from 0 to 9, indicating
the digit that the image represents. The images are typically stored in a single ﬂattened array of 784
elements, although they can also be represented in a 28x28 matrix format.

B.1.2
Fashion MNIST

The Fashion MNIST dataset is a contemporary alternative to the traditional MNIST dataset, created
to provide a more challenging benchmark for machine learning algorithms. It consists of images
of various clothing items and accessories, offering a more complex and varied dataset for image
classiﬁcation tasks.

The Fashion MNIST dataset contains a total of 70,000 grayscale images, where the training set has
60,000 images and the test set has 10,000 images. Each image in the dataset is 28x28 pixels, resulting

14


---Page Break---
in a total of 784 pixels per image. The images are in grayscale, with pixel values ranging from 0
(black) to 255 (white). Each image is associated with one of 10 categories, representing different
types of fashion items. The categories are: 1. T-shirt/top 2. Trouser 3. Pullover 4. Dress 5. Coat
6. Sandal 7. Shirt 8. Sneaker 9. Bag 10. Ankle boot. Similar to MNIST, the images are stored in a
single ﬂattened array of 784 elements but can also be represented in a 28x28 matrix format.

B.2
Neural network architectures

For all methods, we adopt a family of factorized Gaussian distributions to model both priors and
posteriors, characterized by the form π = N(w, σI) where w ∈Rd denotes the mean vector, and σ
represents the scalar variance. We use feedforward neural networks for the MNIST dataset (LeCun
and Cortes, 2010), while using convolutional neural networks for the Fashion MNIST dataset (Xiao
et al., 2017).

Both our feedforward neural network and convolutional neural network are probabilistic, and each
layer has a factorized (i.e. mean-ﬁeld) Gaussian distribution.

Our feedforward neural network has the following architecture:

1. Input layer. Input size: 28 × 28 (ﬂattened to 784 features).
2. Probabilistic linear layer 1. Input features: 784, output features: 600, activation: ReLU.
3. Probabilistic linear layer 2. Input features: 600, output features: 600, activation: ReLU.
4. Probabilistic linear layer 3. Input features: 600, output features: 600, activation: ReLU.
5. Probabilistic linear layer 4. Input features: 600, output features: 10, activation: Softmax.

Our convolutional neural network has the following architecture:

1. Input layer. Input size: 1 × 28 × 28.
2. Probabilistic convolutional layer 1. Input channels: 1, output channels: 32, kernel size: 3x3,
activation: ReLU.
3. Probabilistic convolutional layer 2. Input channels: 32, output channels: 64, kernel size: 3x3,
activation: ReLU.
4. Max pooling layer. Pooling size: 2x2.
5. Flattening layer. Flattens the output from the previous layers into a single vector.
6. Probabilistic linear layer 1. Input features: 9216, output features: 128, activation: ReLU.
7. Probabilistic linear layer 2 (output layer). Input features: 128, output features: 10, activation:
Softmax.

B.3
Other details in the experiments

General for all methods
The methods in comparisons are trained and evaluated using the procedure
described in Section 2 and visually illustrated in Figure 1. We will provide some further details for
each method later in the following. For all methods in comparison, we apply the optimization and
evaluation method described in Section 5.1. For the approximation described in Section 5.1.1, we set
the parameters c1 = c2 = 5. The lower bound for the prediction pmin = 1e −5. The δ in our bound
and all the other methods is selected to be δ = 0.025. As mentioned in Section 5.1.2, we use the
PAC-Bayes-classic bound by McAllester in replacement of PAC-Bayes-kl when doing optimization.
Note that for all methods, we also have to estimate the empirical loss of the posterior Eπ[·] described
in Section 5.1.3. We also allocate the budget for the union bound for the estimation such that these
estimations in the bound are controlled with probability at least 1 −δ′, where we chose δ′ = 0.01.
Therefore, the ultimate bounds for all methods hold with probability at least 1 −δ −δ′. Note that we
do not consider such bounds during optimization but only when estimating the bounds.

For all methods, we adopt a family of factorized Gaussian distributions to model both priors and
posteriors of all the learnable parameters of the classiﬁers, characterized by the form π = N(w, σI)
where w ∈Rd denotes the mean vector, and σ represents the scalar variance. For all methods, we
initialize an uninformed prior π0 = N(w0, σ0I) that is independent of data, where the mean is
randomly initialized, and the variance σ0 is initialized to 0.03 (Pérez-Ortiz et al., 2021).

15


---Page Break---
In the training process of all methods in our experiments, we set the batch size to 250, the number
of training epochs to 200, and use stochastic gradient descent with a learning rate of 0.005 and a
momentum of 0.95.

Uninformed priors
We take π0 deﬁned above as the uninformed prior. We then learn the posterior
ρ from the prior using the entire training dataset S, applying a PAC-Bayes bound. We evaluate the
bound using, again, the entire training dataset S.

Data-informed priors
We start with the same π0 as the uninformed prior. We train the informed
prior π1 using S1 with |S1| = |S|/2 by minimizing a PAC-Bayes bound. The posterior ρ is then
learned using the informed prior π1 and the subset S2 with |S2| = |S|/2, again by minimizing a
PAC-Bayes bound. The bound is evaluated using S2.

Data-informed priors + excess loss
We train the informed prior π1 and the reference classiﬁer
h∗using S1 that contains half of the training dataset. π1 is obtained by minimizing a PAC-Bayes
bound with the uninformed prior π0, while the reference classiﬁer h∗is obtained by an empirical risk
minimizer (ERM). The posterior ρ is obtained by minimizing a PAC-Bayes bound on the excess loss
between ρ and h∗. The prior used in the bound for both training and evaluation is the data-informed
prior π1. Therefore, the data for both training and evaluation of ρ must be the other half of data S2.

B.4
Further results for the experiments

In this section, we report some more statistics for all methods.

For all methods, to calculate the classiﬁcation loss of ρ on the testing data, Eρ[ˆL(h, Stest)] (Test 0-1),
we sample one classiﬁer for each data. The train 0-1 loss for all methods is computed on the entire
training dataset S, while the test 0-1 loss for all methods is computed on the test dataset Stest.

B.4.1
Recursive PAC-Bayes

We report the additional results of Recursive PAC-Bayes on MNIST with T = 2 in Table 4, T = 4 in
Table 5, and T = 6 in Table 6. We report Recursive PAC-Bayes on Fashion MNIST with T = 2 in
Table 7, T = 4 in Table 8, and T = 6 in Table 9.

Table 4: Insight into the training process of the Recursive PAC-Bayes for T = 2 on MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.024 (3e-5)
.370 (1e-3)
.254 (2e-3)
2
30000
.013 (3e-3)
.024 (1e-4)
.136 (3e-3)
.321 (3e-3)
.139 (3e-3)

Table 5: Insight into the training process of the Recursive PAC-Bayes for T = 4 on MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.023 (8e-5)
.374 (1e-3)
.258 (1e-3)
2
52500
-4e-4 (1e-3)
.025 (3e-4)
.118 (1e-3)
.305 (1e-3)
.126 (2e-3)
3
45000
.053 (1e-3)
.002 (9e-5)
.087 (2e-3)
.240 (2e-3)
.114 (1e-3)
4
30000
.054 (1e-3)
.001 (2e-5)
.083 (1e-3)
.203 (8e-4)
.109 (1e-3)

Table 6: Insight into the training process of the Recursive PAC-Bayes for T = 6 on MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.019 (7e-5)
.425 (1e-3)
.311 (3e-3)
2
58125
-0.013 (1e-3)
.032 (6e-4)
.128 (2e-3)
.341 (3e-3)
.139 (1e-3)
3
56250
.050 (1e-3)
.003 (1e-4)
.093 (6e-4)
.264 (1e-3)
.117 (2e-3)
4
52500
.051 (1e-3)
.001 (6e-5)
.080 (9e-4)
.212 (5e-4)
.108 (2e-3)
5
45000
.051 (1e-3)
9e-4 (3e-5)
.076 (2e-3)
.182 (1e-3)
.104 (6e-4)
6
30000
.049 (1e-3)
7e-4 (3e-5)
.074 (1e-3)
.166 (1e-3)
.101 (1e-3)

16


---Page Break---
Table 7: Insight into the training process of the Recursive PAC-Bayes for T = 2 on Fashion MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.011 (3e-5)
.466 (1e-3)
.389 (5e-3)
2
30000
.064 (3e-3)
.013 (2e-4)
.171 (4e-3)
.404 (3e-3)
.266 (5e-3)

Table 8: Insight into the training process of the Recursive PAC-Bayes for T = 4 on Fashion MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
.011 (1e-4)
.476 (4e-3)
.397 (6e-3)
2
52500
.032 (6e-4)
.017 (9e-4)
.147 (2e-3)
.386 (3e-3)
.240 (4e-3)
3
45000
.100 (3e-3)
3e-3 (1e-4)
.138 (5e-3)
.331 (4e-3)
.222 (5e-3)
4
30000
.095 (2e-3)
7e-4 (5e-5)
.128 (2e-3)
.293 (1e-3)
.213 (3e-3)

Table 9: Insight into the training process of the Recursive PAC-Bayes for T = 6 on Fashion MNIST.

t
nval
t
Eπt[ ˆFγt(h, U val
t
◦ˆπt−1)]
KL(π∗
t ∥π∗
t−1)
nval
t
Et(π∗
t , γt)
Bt(π∗
t )
Test 0-1
1
60000
9e-3 (8e-5)
.534 (4e-3)
.462 (5e-3)
2
58125
.013 (6e-3)
.023 (1e-3)
.151 (4e-3)
.418 (5e-3)
.254 (6e-3)
3
56250
.091 (3e-3)
3e-3 (4e-4)
.141 (2e-3)
.350 (3e-3)
.223 (1e-3)
4
52500
.090 (2e-3)
9e-4 (9e-5)
.121 (1e-3)
.296 (1e-3)
.207 (1e-3)
5
45000
.090 (1e-3)
6e-4 (3e-5)
.117 (2e-3)
.265 (2e-3)
.199 (2e-3)
6
30000
.093 (1e-3)
5e-4 (2e-5)
.122 (1e-3)
.255 (1e-3)
.198 (1e-3)

B.4.2
Uninformed priors

We report the additional results of uninformed priors (McAllester, 1998) on MNIST and Fashion
MNIST in Table 10. As described earlier in Section 2, Section 5, and Section B.3, we evaluate the
bound using the entire training set.

Table 10: Further details to compute the bound for the uninformed prior approach on MNIST and
Fashion MNIST.

Eρ[ˆL(h, S)]
KL(ρ∥π0)

n
Bound
Test 0-1
MNIST
.343 (2e-3)
.023 (4e-5)
.457 (2e-3)
.335 (3e-3)
F-MNIST
.382 (2e-3)
.011 (8e-6)
.464 (2e-3)
.384 (5e-3)

B.4.3
Data-informed priors

We report the additional results of data-informed priors (Ambroladze et al., 2007) on MNIST and
Fashion MNIST in Table 11. As described earlier in Section 2, Section 5, and Section B.3, we
evaluate the bound using S2 that is independent of the data-informed prior π1.

Table 11: Further details to compute the bound for the data-informed prior on MNIST and Fashion
MNIST.

Eρ[ˆL(h, S2)]
KL(ρ∥π0)

|S2|
Bound
Test 0-1
MNIST
.376 (8e-4)
8e-4 (9e-6)
.408 (9e-4)
.371 (6e-3)
F-MNIST
.412 (1e-3)
4e-4 (7e-6)
.440 (1e-3)
.413 (6e-3)

B.4.4
Data-informed priors + excess loss

We report the additional results of data-informed priors + excess loss (Mhammedi et al., 2019) on
MNIST and Fashion MNIST in Table 12 and 13. As described earlier in Section 2, Section 5, and
Section B.3, we evaluate the bound using S2 that is independent of the data-informed prior π1 and
the reference prediction rule h∗. The bound is composed of two parts: a bound on the excess loss of
ρ with respect to h∗(Excess bound) and a single hypothesis bound on h∗(h∗bound). We report the

17


---Page Break---
two components of the bound in Table 12. We provide further details to compute these bounds from
the losses of their corresponding quantities in Table 13.

Table 12: Details to compute the bound for the data-informed prior and excess loss on MNIST and
Fashion MNIST. The table shows the bound on the excess loss of ρ with respect to h∗(Excess bound)
and a single hypothesis bound on h∗(h∗bound).

Ex. Bound
h∗Bound
Bound
Test 0-1
MNIST
.162 (1e-3)
.029 (4e-4)
.192 (2e-3)
.151 (3e-3)
F-MNIST
.196 (5e-3)
.145 (1e-3)
.342 (6e-3)
.285 (5e-3)

Table 13: Further details to compute the bound for the data-informed prior and excess loss on MNIST
and Fashion MNIST. The table shows the empirical excess loss Eρ[ ˆ∆(h, h∗, S2)], where we deﬁne
∆(h, h∗, S2) = ˆL(h, S2) −ˆL(h∗, S2), and its bound (Excess Bound). It also shows the empirical
loss of the reference prediction rule ˆL(h∗, S2) and its bound. The computation of such bound does
not involve the KL term.

Eρ[ ˆ∆(h, h∗, S2)]
KL(ρ∥π1)

|S2|
Ex. Bound
ˆL(h∗, S2)
h∗Bound
Bound
MNIST
-0.011 (3e-3)
.035 (5e-4)
.162 (1e-3)
.026 (4e-4)
.029 (4e-4)
.192 (2e-3)
F-MNIST
.104 (6e-3)
.018 (5e-4)
.196 (5e-3)
.112 (1e-3)
.145 (1e-3)
.342 (6e-3)

18


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?
Answer: [Yes]
Justiﬁcation: We tried to write an abstract and an introduction that reﬂected the rest of the
paper as accurately as possible.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reﬂect how
much the results can be expected to generalize to other settings.
• It is ﬁne to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justiﬁcation: We have tried to be as honest as possible in setting the framework of this
theoretical work and have explicitly mentioned the limiting assumptions of the paper, e.g.
i.i.d. data.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-speciﬁcation, asymptotic approximations only holding locally). The authors
should reﬂect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reﬂect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reﬂect on the factors that inﬂuence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efﬁciency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be speciﬁcally instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

19


---Page Break---
Answer: [Yes]
Justiﬁcation: We have clearly stated the assumptions made in the paper, and our theoretical
results are accompanied by detailed proofs.
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
Justiﬁcation: The datasets are available online and we have tried to provide as much detail
as possible to make our experiments transparent and reproducible.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or veriﬁable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation, it may
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

20


---Page Break---
Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justiﬁcation: The code has been uploaded, the datasets are open source.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so No is an acceptable answer. Papers cannot be rejected simply for not
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

Justiﬁcation: We have introduced many different notations to accurately specify the different
data splits and parameters used in our framework, and have tried to detail the optimization
procedure to make our results as understandable as possible.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?

Answer: [Yes]

Justiﬁcation: We provided for each experiment the corresponding standard deviations over
different runs of the experiments. Please refer to the tables in the paper for more details.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
the main claims of the paper.

21


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
of Normality of errors is not veriﬁed.
• For asymmetric distributions, the authors should be careful not to show in tables or
ﬁgures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding ﬁgures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justiﬁcation: As mentioned in the paper, all the experiments were run on a personal laptop.
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
Justiﬁcation: We have not violated any of the points in the NeurIPS Code of Ethics.
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
Justiﬁcation: This is a theoretical paper with no speciﬁc societal impact.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

22


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact speciﬁc
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
feedback over time, improving the efﬁciency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justiﬁcation: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety ﬁlters.
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

Justiﬁcation: The open source datasets we used in the paper were properly cited, namely
MNIST (LeCun and Cortes, 2010) and Fashion MNIST (Xiao et al., 2017).

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

23


---Page Break---
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

Justiﬁcation: We have not released any new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip ﬁle.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

Justiﬁcation: Our paper did not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is ﬁne, but if the main contribu-
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

Justiﬁcation: Our paper did not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

24


---Page Break---
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

25


---Page Break---
