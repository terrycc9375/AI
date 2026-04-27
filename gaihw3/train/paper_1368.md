Theoretical Analysis of Weak-to-Strong Generalization

Hunter Lang
MIT CSAIL
David Sontag
MIT CSAIL
Aravindan Vijayaraghavan
Northwestern University

Abstract

Strong student models can learn from weaker teachers: when trained on the pre-
dictions of a weaker model, a strong pretrained student can learn to correct the
weak model’s errors and generalize to examples where the teacher is not confident,
even when these examples are excluded from training. This enables learning from
cheap, incomplete, and possibly incorrect label information, such as coarse logical
rules or the generations of a language model. We show that existing weak supervi-
sion theory fails to account for both of these effects, which we call pseudolabel
correction and coverage expansion, respectively. We give a new bound based
on expansion properties of the data distribution and student hypothesis class that
directly accounts for pseudolabel correction and coverage expansion. Our bounds
capture the intuition that weak-to-strong generalization occurs when the strong
model is unable to fit the mistakes of the weak teacher without incurring additional
error. We show that these expansion properties can be checked from finite data and
give empirical evidence that they hold in practice.

1
Introduction

Weakly-supervised learning allows practitioners to train models with possibly-incorrect, easy-to-
obtain pseudolabels instead of accurate and expensive ground-truth labels. For example, suppose the
goal is to classify documents based on whether they have positive or negative sentiment. Instead of
employing humans to label examples xi with positive/negative sentiment labels yi, weak supervision
enables models to learn from simple rules, such as: if ‘incredible’ ∈xi, sentiment =
positive. Called programmatic weak supervision, a wealth of literature has shown how to aggregate
rules, often called “labeling functions”, into individual pseudolabels that can be used to train a model
[e.g., 51, 49, 63, 21]. In typical pipelines, this consists of fine-tuning a pre-trained neural network
[71]. This method has met with a huge amount of empirical success in natural language processing
[51, 43, 28, 36, 70] computer vision [74, 62], and verticals such as healthcare [25, 26, 20, 67].

Another emerging trend in weak supervision is to use the zero-shot or few-shot outputs of a large
language model (LLM) as pseudolabels for training another language model—the student model
often outperforms its noisy “teacher”, and this technique even works when the teacher is a less
powerful model than the student, distinguishing it from classical knowledge distillation [65, 17, 18,
35, 2, 24, 33, 66, 39, 10]. Good data selection can be critical, and several approaches carefully choose
a “confident” subset of the pseudolabels [e.g. 35, 18, 39, 66]. The increasing prevalence of LLM use
in crowdwork [64] highlights the importance of better understanding this learning method.

In both cases, the pseudolabels ˜y are given by the pseudolabeler or teacher model, which is some
function of the input example x. The pseudolabeler may make errors (˜y(x) ̸= y(x)), and it may
not cover every point—there are some points where the teacher abstains from providing a weak label.
In the examples above, these are points not covered by the rules and points where the teacher LLM is
not confident, respectively. A priori, it seems that a powerful enough classifier should exactly fit the
pseudolabeler on the covered data and have trivial performance on the uncovered data. However, this
is not what happens in practice—the empirical success of weak supervision is due to two surprising,
related phenomena that together comprise weak-to-strong generalization: (a) Pseudolabel correction:

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
The performance of the model exceeds the performance of the pseudolabels used to train it; and
(b) Coverage expansion: The model performs well even on the portion of example space X that is
not covered by pseudolabels. These empirical outcomes are key to the success of weak supervision.

Surprisingly, the existing theoretical literature on programmatic weak supervision does not address
either phenomenon—the majority of weak supervision theory literature either focuses on how to
adeptly combine the outputs of multiple “weak rules” into a single pseudolabel ˜yi, [51, 49, 50, 21],
or treats learning from weak supervision as learning from noisy labels [e.g. 46, 59, 15], but as we
discuss in Section 2, this framing does not capture the setting we consider, where the pseudolabels
used to train one model are the outputs of another model.

In this work, we give a theoretical analysis of weakly-supervised learning that provably accounts for
the effects of pseudolabel correction and coverage expansion. Our results use a natural expansion
condition on the population data distribution. Informally, expansion implies that “bad” points (points
with incorrect pseudolabels, or points with no pseudolabel at all) have many “good” neighbors (points
with correct pseudolabels). If the learned student model is relatively robust on the neighborhoods
of interest, then making a mistake on a bad point means making many mistakes on good points as
well. This allows us to prove a relationship between the student model’s error on the weak labels
(the training objective) and the student model’s error on the true labels (the desired objective). Our
assumptions and bounds in Section 4 formalize this intuition. Section 5 details a procedure for
checking our expansion conditions from finite data, and in Section 6 and Appendix E, we give
empirical evidence that these conditions hold on real data.

To the best of our knowledge, our results provide the first error bounds for programmatic weak
supervision with realistic assumptions. We show that our bounds generalize and connect several
existing results from the co-training, self-training, and distribution shift literature [e.g., 6, 3, 69, 12]
and adapt them to the weak supervision setting. For example, we show in Appendix C.1 that Theorem
4.2 generalizes the co-training results of Blum and Mitchell [6]. We discuss these generalizations in
detail in Sections 2 and 4 and Appendix C. Our result in Section 5 is the first among these works to
prove that the expansion assumptions can, in principle, be checked using finite data. Unlike most prior
work in this space, our experiments in Section 6 attempt to check whether the expansion assumptions
hold in practice. While our experiments are limited in scope, our attempt to systematically check
expansion in a practical scenario is a major departure from previous work.

Finally, prior work with expansion assumptions similar to ours (in the co-training [6, 3], self-training
[69], and distribution shift [12] literature) requires that the classifiers are either perfectly robust [6, 3]
or adversarially robust [69, 12] for their bounds to apply. Empirical results suggest that adversarial
training has fairly limited value for improving coverage expansion and pseudolabel correction [38],
and that these two effects still occur for student models that are not adversarially trained [71, 10].
To close this gap, we make a connection to the literature on robust expansion [27, 34, 42] and prove
error bounds for student models that are merely “robust on average.” Unlike prior work, these bounds
allow for the presence of adversarial examples for every input point.

2
Related Work

Ratner et al. [51, 49, 50], Fu et al. [21] focus on how to combine the outputs of multiple “weak
rules” into a single pseudolabel ˜y(x) for each covered example, a problem with a long history in
the crowdsourcing literature [e.g. 16, 30, 29]. However, empirical results indicate that this is not the
important aspect of weak supervision: most methods for combining weak rules fail to significantly
outperform majority vote once the final classifier is trained [72]. Works in this literature that do
provide error bounds for the student (e.g., Fu et al. [21], Ratner et al. [51]) either fail to capture weak-
to-strong generalization effects, as shown in Section 3, or make difficult-to-justify assumptions—for
example, Ratner et al. [51] assumes that y(x) and x are conditionally independent given ˜y(x). If
this were true, there would be no gain from training a classifier, since ˜y(x) already captures all
the information that x contains about y(x). Work that treats learning from weak supervision as a
noisy label learning problem [e.g. 45, 46, 59, 15] does not capture the types of weak supervision we
consider. When the supervision comes from weaker model (be it rule-based or a weaker LLM), there
is no exogenous noise process that corrupts the training labels. There are simply some points that
deterministically get the wrong labels and some points with no label. This rules out common noise
models like class-conditional noise [46] and Tsybakov noise [60], and is arguably not appropriate to
model as instance-dependent noise [59, 15], since for each x the noise is deterministically 0 or 1.

2


---Page Break---
Burns et al. [10] conducted a large empirical study showing widespread weak-to-strong generalization
effects when training a strong language model on the generations of a weaker model. Our results
are a step toward a theoretical understanding of these effects. Several works have used expansion to
give provable guarantees in other settings where models are learning from each other. Balcan et al.
[3] use expansion to analyze co-trained [6] classifiers. Our expansion assumption is similar to their
“left-right” expansion, but we generalize beyond the multi-view setup of co-training and account
for error propagation. Wei et al. [69] give provable guarantees for self-training under expansion
assumptions similar to ours. We provide a different pseudolabel correction bound, tighter guarantees
for coverage expansion under weaker assumptions, and generalize both results to classifiers that are
not adversarially robust. Cai et al. [12] use expansion to prove general guarantees for pseudolabel
correction, semi-supervised learning, and unsupervised domain adaptation, but their results require
the student to be very adversarially robust. Compared to all these expansion works, we also outline a
rigorous theoretical framework for checking expansion on finite data and provide more empirical
evidence for our assumptions. We discuss more related work in Appendix A.

3
Setup and Shortcomings of Existing Bounds

Notation. x refers to a random variable with distribution D and italicized letters x refer to realizations
of x. We will assume for ease of exposition that the input space X is a discrete1 (but possibly very
large) set, such as all vectors in Rd up to a fixed numerical precision. For A ⊂X we use sA = X \ A.
We assume there is a ground-truth function of interest, y : X →Y = {1, . . . , k}, and a pseudolabeler
˜y : X →Y ∪{∅}, which assigns to each point x either a label in Y or the special “abstention”
symbol ∅. The function ˜y can also be thought of as the teacher model, but we are primarily concerned
with instances where the teacher is much less capable than the “student” it will be used to train.

Define S = {x|˜y(x) ̸= ∅} to be the covered subset of X, i.e., the subset of X that has a pseudolabel,
and let T = {x|˜y(x) = ∅} = X \ S be the uncovered set. This notation serves to emphasize
that training occurs on a (pseudolabeled) source subset S, and then evaluation occurs on the union
of S and the (uncovered) target T. Let {Xi} be a partition of X such that within each Xi, the
ground-truth label is constant. For example, we could set Xi = {x|y(x) = i} to be the set of
points with ground-truth label i. We will use this definition of Xi for convenience, but all our results
hold for more general partitions. Each of S and T can further be partitioned as Si = S ∩Xi,
Ti = T ∩Xi. Finally, each Si can be further partitioned into the correctly-pseudolabeled examples
Sgood
i
= {x ∈Si|˜y(x) = y(x)} and the incorrectly-pseudolabeled examples Sbad
i
= Si \ Sgood
i
.
Let αi := P(Sbad
i
|Si) be the error rate of ˜y on Si. We assume 0 < αi < 1

2 for all i.

Problem Setup. For two classifiers f, g : X →Y and a set U ⊂X, we use err(f, g|U) to represent
P(f(x) ̸= g(x)|x ∈U), their probability of disagreement conditioned on x falling in the set U. Here
the probability is over x ∼D; this will often be omitted for notational convenience. We will be par-
ticularly interested in classifiers obtained by minimizing the error on the non-abstaining weak labels
over the strong model hypothesis class F, i.e., (approximate) solutions to arg minf∈F err(f, ˜y|S).
The ultimate goal is to obtain upper bounds on the error err(f, y|X) for such classifiers. That is,
we want to upper bound the error of a classifier f on the true labels over the entire input space X.

There are two key challenges. First, the classifier is trained using ˜y, not y, and ˜y may have arbitrary
errors that are not captured by any well-studied noise model such as class-conditional or Tsybakov
noise [60]—we’ve assumed ˜y and the true labels y are both deterministic functions of the input, so
there is no exogenous noise process that corrupts the training labels.

Second, we care about the performance of f on the entire space X, but we only train on the covered
samples from S ⊂X. Again, since ˜y is an arbitrary deterministic function, our samples from S are
not distributed according to D, and S and T have no overlap, ruling out approaches like importance-
weighting. The following example elaborates on the issues at play and illustrates the shortcomings of
existing bounds in the weak supervision literature.

3.1
Shortcomings of Existing Bounds: Illustrative Example

A special case of weak-to-strong generalization is training a strong pretrained model on the outputs
of very coarse rules. Following the example from Section 1, suppose our goal is to obtain a sentiment

1This is done to simplify some of the arguments, as in HaoChen et al. [22]. As shown there, the results can
be generalized to continuous X with additional regularity conditions. Our bounds have no dependence on |X|.

3


---Page Break---
classifier, so X is the space of text documents, Y = {−1, 1} and ˜y is given by the following rules: if
‘incredible’ ∈x, ˜y(x) = +1. If ‘horrible’ ∈x, ˜y(x) = −1. Otherwise, ˜y(x) = ∅.

Assume for simplicity that “incredible” and “horrible” never co-occur, so ˜y is well-defined. This
example shows that the student model hypothesis class F and the training procedure both play a vital
role in weak-to-strong generalization. Suppose F is the class of bag-of-words classifiers, and we
obtain a student f by minimizing err(f, ˜y|S). Without modifications to the training procedure (such
as L2 regularization), f may place a large positive weight on “incredible”, a large negative weight
on “horrible”, and zero weight on all other tokens. This model has zero error on the weak labels, so
it exactly minimizes the training objective. It reproduces the pseudolabels on the covered set and
has trivial performance on the uncovered set, so there is no weak-to-strong generalization. On the
other hand, if we were to instead train a linear probe on top of a SentenceBERT [52] representation,
we would obtain a model that improves over ˜y on S (the covered set of documents containing either
“horrible” or “incredible”) and has reasonable performance on T (the uncovered set). Section 6

contains precise results for this example, but the critical (seemingly obvious) aspect is that the student
representation and the training details matter for achieving weak-to-strong generalization.

The following proposition (proven in Appendix B.3) illustrates how existing error bounds in the
programmatic weak supervision literature, which do not account for training details and the student
hypothesis class, are unable to capture pseudolabel correction and coverage expansion.
Proposition 3.1. Suppose the label marginals for the above example satisfy P(y = y) = 1

2 for
y ∈{−1, 1}, and assume that the weak label error rates α−1 = α1 = α, and that the weak labels
cover each class equally often: P(˜y = ∅|y = y) = P(˜y = ∅). Let ef = minf∈F err(f, ˜y|S) be the
classifier minimizing the weak label error on the covered set. Then the bound from Fu et al. [21,
Theorem 3] simplifies (in our notation) to: err( ˜f, y) ≤P(S) · 4α(1 −α) + P(T).

The first term accounts for the error of ˜f on the covered set S. The weak labels themselves have error
α on S, but the bound for ˜f is 4α(1−α) > α whenever α < 3

4, so Fu et al. [21]’s bound does not allow
for pseudolabel correction in this example. The second term accounts for the error of ˜f on the uncov-
ered set T. A random guess achieves error 1

2 on T, but the bound charges every point in T as an error,
so it also does not account for coverage expansion or even the performance of random guessing on T.

Expansion assumptions similar to ours have also been studied in the context of self-training [69] and
domain adaptation [12]. The following proposition shows that while the results of Wei et al. [69] can
be adapted to weakly-supervised learning, our bounds capture the full weak supervision setup better,
since Wei et al. [69, Theorem 4.3] was not designed to deal with partial coverage P(x ∈S) < 1, so
applying it to weakly-supervised learning still requires fairly large coverage.
Proposition 3.2 (informal). Suppose the coverage P(x ∈S) in the example above is less than 2/3.
Then the bound from Wei et al. [69, Theorem 4.3] does not apply since directly adapting it to the
weak supervision setting requires P(x ∈S) ≥2/3.

Empirically, coverage expansion and pseudolabel correction can both occur in the low-coverage
regime [38]. Finally, as mentioned in Section 1, Wei et al. [69] assumes the classifier is adversarially
robust. In contrast, we provide bounds that directly account for coverage expansion, place no
restrictions on the amount of weak label coverage, and allow for the presence of many adversarial
examples. As the example in this section suggests is necessary, the model hypothesis class and the
training details (in particular, the robustness of the model) play a central role in our bounds. The
following definitions attempt to capture these properties.

3.2
Definitions

Definition 1 (Neighborhood). Let N be a neighborhood function that maps each point x to a set
of points N(x) ⊂X that we call the neighborhood of x. We will assume that N satisfies x ∈
N(x′) ⇐⇒x′ ∈N(x), i.e., that the neighborhoods are symmetric. We can extend N to a function
of sets as N(A) = S

x∈A N(x). Examples to keep in mind are N(x) = {x′ : ||φ(x)−φ(x′)|| ≤r}
for some representation φ : X →Rd, or, in the case of text inputs x, the set of fluent paraphrases of
x. However, our results work with any definition of N.

Definition 2 (η-robust). For an arbitrary classifier f and point x, define r(f, x) = P(f(x′) ̸=
f(x)|x′ ∈N(x)) as the probability f gives different labels to x and a random neighbor x′ of x.

4


---Page Break---
A

 >  
N(U)
U

•
U
N(U)
c

B
A
B

Figure 1: Relative expansion (Definition 3) on the sets (A, B). Expansion requires that certain
subsets U ⊂B have neighborhoods N(U) such that P(N(U)|A) ≥cP(U|B). These probabilities
are represented graphically on the right-hand-side as the fractions |N(U) ∩A|/|A| and |U|/|B|.

A classifier f : X →Y is said to be η-robust at a point x if r(f, x) ≤η. Define Rη(f) = {x :
r(f, x) ≤η} as the set of η-robust points for f.

If η = 0, f is η-robust at x if and only if f is adversarially robust over N(x), so this definition
generalizes adversarial robustness. By Markov’s inequality, any classifier f with:

Ex∼D,x′∼D|N(x)[f(x) ̸= f(x′)] ≤γ
(1)

is η-robust on a set of probability at least 1 −γ/η (see Lemma B.1). The requirement (1) is
significantly more natural than adversarial robustness: a classifier satisfies (1) whenever it gives most
points the same labels as most of their neighbors. We refer to (1) as “average-case robustness”.
Definition 3 (Expansion). Fix sets A, B ⊂X. We say the distribution Px satisfies (c, q)-expansion
on (A, B) if for all sets U ⊂B with P(U|B) > q, P(N(U)|A) > cP(U|B).

Figure 1 shows examples of Definition 3 graphically. Intuitively, a pair of sets A and B satisfy this
definition if large subsets of B correspond/expand (via the neighborhood N) to large subsets of A.
Definition 3 requires all sets U with large enough probability in B to expand to A. However, this is
unnecessarily strong. Our theorems will only need certain structured sets, corresponding to elements
of the student hypothesis class, to expand. This is captured by Definition 4 and is the key to our
results in Section 5 on checking the expansion property.
Definition 4 (Expansion of a set collection). Fix sets A, B ⊂X and suppose M is a collection
of subsets of B. Then we say M satisfies (c, q)-expansion on (A, B) if all sets U ∈M with
P(U|B) > q satisfy P(N(U)|A) > cP(U|B).

4
Error Bounds for Weakly-Supervised Classifiers

In this section, we upper bound the gold error of f on the covered and uncovered sets—err(f, y|S)
and err(f, y|T)—with expressions involving the weak error err(f, ˜y|S) of f on the covered set,
expansion parameters, and robustness parameters. These bounds give a theoretical justification for
why empirical risk minimization using the weak labels leads to weak-to-strong generalization. We
condition on subsets Si ⊂S (for pseudolabel correction) or pairs of subsets Si ⊂S, Ti ⊂T (for
coverage expansion). This subset-wise conditioning allows for different parts of the distribution to
expand in different amounts, yielding tighter bounds, an approach also followed by Cai et al. [12].

4.1
Adversarially Robust Models
We begin by stating our theorems for the case when the classifier of interest is η-robust with η = 0
on most points x. That is, we assume that for most points x, the classifier is adversarially robust over
N(x). This follows the assumptions in related work from other domains [69, 12]. We describe our
results for the significantly more general average-case-robust classifiers in Section 4.2.

Expanding Families. We first define the families M and M′ of sets that must expand according
to Definition 4. Let F be the hypothesis class of the strong model and for each f ∈F, define
R(f) = R0(f) = {x : r(f, x) = 0} to be the set of adversarially robust points for f. For B ⊂X
and f ∈F, define U(B, f) = {x ∈B : f(x) ̸= y(x)} as the set of points in B where f makes a
mistake on the true label y. Now define M(B, F) to be the class of robust mistakes sets: M(B, F) =
{U(B, f) ∩R(f) : f ∈F}. Similarly, define M′(B, F) = {(B \ U(B, f)) ∩R(f) : f ∈F} as
the family of robust non-mistakes on B.

5


---Page Break---
Pseudolabel Correction. Here we relate the gold error of the student model f on a covered subset,
err(f, y|Si), to the weak error of f on that set, err(f, ˜y|Si). The goal is to allow for the correction of
some of the incorrect weak labels (Sbad
i
): we want our bounds for err(f, y|Si) to be less than αi, the
error rate of the weak labels. Expansion between the points with correct pseudolabels, Sgood
i
, and
points with incorrect pseudolabels, Sbad
i
, implies that there are many bad points with good neighbors.
If the classifier is suitably robust on the neighborhoods, pseudolabel correction can occur. The bound
in this section makes this intuition quantitative.

Theorem 4.1 (Pseudolabel correction). Suppose M′(Sgood
i
, F) satisfies (c, q)-expansion on the
sets (Sbad
i
, Sgood
i
) for q < 3

4(1 −2α). Consider an arbitrary classifier f ∈F such that P(f(x) ̸=
˜y(x) or f not robust at x|Si) ≤1−α+3cα

4
. Then the true error of f on Si satisfies:

err(f, y|Si) ≤
2αi
1 −2αi
P(Ę
R(f)|Si) + err(f, ˜y|Si) + αi


1 −3

2c

.

The expansion condition intuitively states that “good” sets (elements of M ′(Sgood
i
, F)) must have
suitably many neighbors with the wrong pseudolabel (elements of Sbad
i
). The trivial error bound
obtained via the triangle inequality is err(f, y|Si) ≤err(f, ˜y|Si)+αi. The bound in Theorem 4.1 has
almost the same form, but the multiplicative term on αi allows it to be much tighter than the trivial
bound. Theorem 4.1 allows for pseudolabel correction because the right-hand-side can be much less
than αi (the error of the weak teacher) when c is large and err(f, ˜y|Si) and P(Ę
R(f)|Si) are small.
While Wei et al. [69, Theorem 4.3] also gives pseudolabel correction guarantees for adversarially
robust classifiers, Theorem 4.1 is a different bound with several desirable properties that Wei et al.
[69, Theorem 4.3] lacks—we compare the two in detail in Appendix C.2 and also show how to
generalize Wei et al. [69]’s results to average-case-robustness.

The following proposition shows that the expansion conditions in Theorem 4.1 rule out strong models
that can exactly fit the weak model.

Proposition 4.1. Suppose there exists f ∈F such that for all x ∈S, f(x) = ˜y(x). Then the
(c, q)-expansion conditions of Theorem 4.1 are not satisfied.

Proof. The family of sets that must satisfy (c, q)-expansion between Sbad
i
and Sgood
i
is M′ =
{R(f) ∩(Sgood
i
\ mistakes(f)) : f ∈F}. Choose f such that f = ˜y when restricted to S,
since we assumed this is a valid choice. Sgood
i
is defined as the set where ˜y makes no mistakes
(and the true label is i), so Sgood
i
\ mistakes(f) = Sgood
i
, and thus R(f) ∩Sgood
i
∈M′. Let
N = {x ∈Sgood
i
: N(x) ∩Sbad
i
̸= ∅} be the subset of Sgood
i
with neighbors in Sbad
i
. Consider
an arbitrary point x ∈N, so there exists x′ ∈N(x) ∩Sbad
i
. Since x and x′ are both in Si,
y(x) = y(x′). Then we have f(x) = ˜y(x) = y(x) = y(x′) ̸= ˜y(x′) = f(x′), so f(x) ̸= f(x′).
Thus x ̸∈R(f). This shows N ⊂Ę
R(f). Hence

R(f) ∩Sgood
i
= R(f) ∩

(Sgood
i
∩N) ∪(Sgood
i
∩s
N)


⊂R(f) ∩

(Sgood
i
∩Ę
R(f)) ∪(Sgood
i
∩s
N)


= R(f) ∩(Sgood
i
∩s
N).

The latter set is made up entirely of points with no neighbors in Sbad
i
, so

P(N(R(f) ∩Sgood
i
)|Sbad
i
) ≤P(N(R(f) ∩(Sgood
i
∩s
N))|Sbad
i
) = 0.

But for expansion to hold, we needed P(N(R(f) ∩Sgood
i
)|Sbad
i
) > cP(R(f) ∩Sgood
i
|Sgood
i
).

Proposition 4.1 shows the strong model hypothesis class enters our error bounds indirectly via the
(data-dependent) expansion parameter. Using a richer class for the strong model may decrease the
amount of expansion, since it may make it easier to exactly fit the weak labels. At the same time, the
error of the strong model on the weak labels also appears as a term in the bounds (err(f, ˜y|Si)), and a
richer class might decrease this term, so these two terms capture a tradeoff. This makes our bounds

6


---Page Break---
flexible enough to capture empirical results showing that whether a richer class or more restricted
class works best for the strong model depends on the problem [73].

Coverage Expansion. In this section, we relate the error of f on an uncovered subset, err(f, y|Ti), to
the weak error of f on the corresponding covered subset, err(f, ˜y|Si). The goal is to give a nontrivial
error bound on these points even though we see none of them during training. Expansion from Ti to
Sgood
i
implies that subsets of Ti have enough correctly-pseudolabeled neighbors. If the student model
is robust on N, this is already enough to prove an error bound for Ti. However, we also assume that
subsets of Ti have enough incorrectly-pseudolabeled neighbors. Intuitively, this means that subsets
of Ti have the “correct” number of neighbors in Sgood
i
and the “correct” number of neighbors in
Sbad
i
. This implies a regular structure in the Si–Ti neighborhood connections that allows us to prove
a much tighter error bound. Our empirical results suggest that this structure is present in real-world
examples. We prove a weaker bound that only assumes expansion from Ti to Sgood
i
in Appendix B.
Theorem 4.2 (Error bound for uncovered points). Suppose M(Ti, F) satisfies (c, q)-expansion on
(Sgood
i
, Ti), and M′(Ti, F) satisfies (c, q)-expansion on (Sbad
i
, Ti). Consider an arbitrary classifier
f ∈F that fits the weak labels well on Si and is fairly robust on Ti: err(f, ˜y|Si) + P(Ę
R(f)|Ti) <
c(1 −q −αi) Then the true error of f on Ti satisfies:

err(f, y|Ti) ≤

1 +
αi
1 −2αi


P(Ę
R(f)|Ti) + max

q, err(f, ˜y|Si) −cαi

c(1 −2αi)


.

To qualify for the bound, f must fit the weak labels well on Si, so err(f, ˜y|Si) is small, and be
adversarially robust at most points on Ti, so P(Ę
R(f)|Ti) is small. We show in Appendix C that the
original co-training setup of Blum and Mitchell [6] satisfies the assumptions of Theorem 4.2 with
c = 1, q = 0, and P(Ę
R(f)|Ti) = 0. Theorem 4.2 exactly recovers the bounds of Blum and Mitchell
[6], Lang et al. [37] in this case, so Theorem 4.2 is a direct generalization of Blum and Mitchell [6]
but without the restrictive assumptions regarding multi-view data and conditional independence. In
Appendix B, we prove a generalization of Theorem 4.2 that allows Ti to expand to Sgood
i
and Sbad
i
at
different rates (i.e., different expansion parameters).

It is not immediately clear from Theorem 4.2 how the coverage P(S)—the probability that the weak
labeler does not abstain—affects the bounds. As with the role of strong model hypothesis class,
this enters the picture implicitly through the parameter measuring expansion between Si and Ti.
Informally, for a fixed neighborhood N, it may be easier to have expansion from Ti to Si when Si
is larger, since for each uncovered point in Ti, there are more possible covered neighbors in Si. On
the other hand, increasing the coverage by including more points in Si might also affect the weak
label accuracy parameters αi, which also appear in the bounds. In practice, there is not a consistent
tradeoff between coverage and performance, as explored recently in, e.g., Lang et al. [37], so our
bounds do not prescribe a functional dependence of the error on the amount of coverage and instead
allow that dependence to enter through data-dependent parameters.

4.2
Relaxing Robustness Requirements

The previous bounds in this section assumed the student is adversarially robust at most points. Here,
we considerably generalize this requirement so our results apply to any classifier f that satisfies (1),
i.e., any classifier that gives most points the same label as most of their neighbors. This requires several
additional definitions and goes beyond the assumptions made in other work with expansion-based
error bounds, which assume adversarial robustness at most [69, 12] or (effectively) all [6] points.

To allow for this relaxed assumption on the classifier, we assume a more robust version of expansion,
aptly called robust expansion [27, 34, 42]. To define robust expansion, we start by defining a graph
over examples with edges induced by the neighborhood N and weights given by the underlying
probability measure. HaoChen et al. [22] study this graph in the context of contrastive pretraining.
Definition 5 (Example graph). Let G = (X, E) be a graph with one node for each element of X (we
assumed X is a possibly very large, but finite, set), and connect two nodes (x, x′) if x ∈N(x′) or,
equivalently, if x′ ∈N(x), with an edge weight of w(x, x′) := P(x)P(x′)1[x ∈N(x′)].

Definition 3 (regular, non-robust expansion) is very sensitive to removal of a few edges from the
example graph. A set U ⊂B may have P(N(U)|A) large, but only because a small fraction of the

7


---Page Break---
edges (by probability mass) are connected to many x ∈A with P(x|A) large. If we ignored these
small-probability edges, the neighborhood would be much smaller. Figure 2 (appendix) shows an
example. The robust neighborhood tries to address this issue:
Definition 6 (η-robust neighborhood size). Let A, U ⊂X. The size of the η-robust neighborhood of
U in A is: P1−η(U, A) := minV ⊂X {P(V |A) : w(V, U) ≥(1 −η)w(N(U), U)}.

P1−η(U, A) is the probability of the “smallest” subset of A that still captures at least a 1 −η
fraction of the edge weight incident on U. When η = 0, we have P1(U, A) = P(N(U)|A), so this
recovers the size of the non-robust neighborhood. In the pathological example described above, we
would have P(N(U)|A) large, but P1−η(U, A) small for some η > 0. We are now ready to give
a “robustified” definition of expansion, which is identical to Definition 4 except that it requires the
robust neighborhood, rather than the regular neighborhood, to be large.
Definition 7 (Robust expansion). Fix sets A, B ⊂X and suppose M is a collection of subsets
of B. M satisfies (c, q, η)-robust expansion on (A, B) if for all U ∈M with P(U|B) > q,
P1−η(U, A) > cP(U|B). This exactly recovers Definition 3 when η = 0.

The following (informal) theorem shows that Theorems 4.1 and 4.2 hold for average-case-robust
classifiers when we replace expansion with robust expansion and R(f) with Rη(f). We state and
prove formal versions of Theorems 4.1 and 4.2 for average-case-robust classifiers in Appendix B.
Theorem 4.3 (Informal). Theorems 4.1 and 4.2 hold exactly with (c, q, η)-expansion instead of
(c, q)-expansion and Rη(f) instead of R(f).

By Markov’s inequality, for any η > 0 a classifier f with Ex∼D|A,x′∼D|N(x)[f(x) ̸= f(x′)] ≤γ has
P(Ğ
Rη(f)|A) ≤γ

η . Theorem 4.3 shows that by assuming the data distribution follows a slightly more
“regular” structure (robust expansion), we can give guarantees for average-case-robust classifiers.
This generalization is important since it matches with empirical results: adversarial training and
adversarial robustness are not required for weak-to-strong generalization to occur [73, 38, 10], and
most empirical work on weak supervision does not include adversarial training in the pipeline [71].

5
Checking Expansion

We now outline a statistical theory for checking the expansion properties of the population distribution
from finite data. This is possible because, as described in Section 4, our results do not actually require
all sets to expand—rather, they only require expansion for a class of sets that is generated by the
student hypothesis class. This means we can check expansion on a finite dataset and control the
generalization of our estimate using the complexity of the hypothesis class. The purpose of checking
expansion is not (currently) algorithmic—the goal of the procedures described in this section is to
give empirical evidence that our assumptions hold in the real world and that our bounds correlate
with actual occurrences of pseudolabel correction and coverage expansion. Exactly checking the
expansion of all subsets is coNP-complete [7]; whether our notion of expansion with respect to a
certain family of sets M can be checked efficiently is an interesting direction for future research. We
show that expansion can at least be checked statistically (i.e., from finite data), if not efficiently.

For a fixed choice of q, the (non-robust) expansion of a set family M between sets A and B is:
c = minU∈M: P(U|B)>q
P(N(U)|A)

P(U|B) . Suppose we have two samples SA = {(xi, y(xi))}nA
i=1 with
x ∼P(·|A), and SB = {(xi, y(xi))}nB
i=1 with x ∼P(·|B). For a fixed U, the denominator
is straightforward to estimate using SB as: P(U|B) ≈
1
nB
P

xi∈SB 1[xi ∈U]. Estimating the
numerator is less straightforward: due to finite sampling, SA × SB may contain no pairs (x, x′)
with x ∈N(x′). That is, the empirical neighbor graph may be empty even when the population
distribution expands (see Wei et al. [69] for a more thorough discussion). This is a major difference
between our assumptions and similar work that uses expansion-like assumptions to analyze the
performance of label-propagation algorithms that use the empirical graph, such as Pukdee et al. [48].
To overcome this, we assume we have access to a neighborhood oracle n : A →B that for each
x ∈A returns a point n(x) ∈B such that n(x) ∈N(x). We assume nothing about the distribution
of n(x) values (i.e., we do not assume that they are drawn from P(·|B), merely that P(n(x)|B) > 0).
We describe how to construct n in a practical scenario in Section 6.

The neighborhood oracle makes estimating the expansion numerator more straightforward, since if
n(x) ∈U, then by construction, x ∈N(U). Formally, P(N(U)|A) ≥P(n(x) ∈U|A), where the

8


---Page Break---
Table 1: Measured expansion and error bounds for the covered sets Si. Expansion values for the
family of sets M′(Sgood
i
, F) are measured using the heuristic described in Section 5 and shown in the
(Sbad
i
, Sgood
i
) exp. column. This column shows our heuristic finds expansion in practice. Pseudolabel
error αi = P(˜y ̸= y|Si). Worst-case error of trained classifier f on the weak labels ˜y, err(f, ˜y|Si),
across 5 independent training runs. This column shows the student can’t exactly fit the teacher labels
using this representation. Value of the error upper bound in Theorem 4.1 (specifically, the tighter
version, B.1), computed using the numbers from the other columns (details in Appendix E). For label
i = 0, the bound being strictly less than the error αi of the teacher ˜y suggests pseudolabel correction
may occur. Finally, the actual worst-case error of trained classifier f on the true labels y, err(f, y|Si),
across 5 independent training runs, shows pseudolabel correction does occur for label i = 0.

Model
i
(Sbad
i
, Sgood
i
) exp.
αi
err(f, ˜y|Si)
Bound val
err(f, y|Si)

SentenceBERT
0
0.848
0.11
0.12
0.05
0.04
1
0.497
0.33
0.29
0.37
0.35

quality of n(x) determines the tightness of this bound. This inequality is valid for any n : A →B
as long as x ∈N(n(x)). Now we can estimate: P(n(x) ∈U|A) ≈
1
nA
P

xi∈SA 1[n(xi) ∈
U]. Putting it all together, we can form our empirical estimate of the expansion by solving ˆc =

minU∈M

1
nA
P

xi∈SA 1[n(xi)∈U]

1
nB
P

xi∈SB 1[xi∈U]
subject to:
1
nB
P

xi∈SB 1[xi ∈U] ≥q −ϵ, where ϵ is chosen

appropriately to account for empirical error in estimating the probability P(U|B). The following
theorem, proven in Appendix D, shows that the expansion on the population distribution can’t be too
much smaller than the expansion on the empirical distribution.
Theorem 5.1 (Expansion generalization, informal). For arbitrary U ∈M, define the population
and empirical expansion estimates as: c(U) := P(n(x) ∈U|x ∈A)/P(x ∈U|x ∈B) and
ˆc(U) :=
1
nA
PnA
i=1 1[n(xi) ∈U]/ 1

nB
PnB
j=1 1[xi ∈U]. Then for any δ ∈(0, 1], with probability at

least 1 −δ, supU∈M ˆc(U) −c(U) ≤e
O(
p

VC(M)/nA), where e
O hides constants and log factors
in VC(M), nA + nB, and 1/δ.

Heuristic approximation.
While Theorem 5.1 gives a rigorous statistical theory for checking
expansion from finite data, it is unfortunately still intractable to compute the set with the worst
expansion on the empirical data, i.e., to solve ˆU = arg minU∈M ˆc(U). Instead, our experiments in
Section 6 use a simple randomized heuristic for approximating this minimization. If the learning
algorithm A : Sm →F is deterministic conditioned on the observed training data, we can simplify
our hypothesis class F of interest to those f ∈F such that there exists a training sample S ⊂S with
f = A(S). Since each f ∈F generates a set U(f) ∈M, given a training sample S, we can compute
f = A(S), then use our “test” sample SA, SB to compute ˆc(U(f)). Repeating this procedure for
many samples S and choosing the smallest value approximates minU∈M ˆc(U). Table 1 shows the
expansion measurements for different hypothesis classes on a weakly-supervised classification task
inspired by the example in Section 3. Appendix 6 contains more results and a much more detailed
description of the setup and process of checking expansion, but these results indicate that expansion
is present and correlates with performance.

6
Experiments

Setup. We explore training linear classifiers on top of the contrastively-fine-tuned SentenceBERT
embeddings2 [52]. As shown in Muennighoff et al. [44], training simple classifiers on top of these com-
plex pretrained representations leads to very competitive performance. We study binary sentiment pre-
diction for movie reviews on the IMDb dataset [41], continuing with the example from Section 3. For
the teacher model, we use a very coarse rule [49] based on the presence of the unigrams “incredible”
and “horrible”. Let C(w, x) be the number of times word w appears in input x. The weak label ˜y(x)
is 1 when C(incredible, x) > C(horrible, x), 0 when C(horrible, x) > C(incredible, x),
and ∅otherwise. This counts the occurrences of “horrible” and “incredible” and assigns the binary
label corresponding to the word that occurs strictly more often, and abstains otherwise.

2HuggingFaceHub model ID sentence-transformers/all-mpnet-base-v2

9


---Page Break---
Neighborhood function and oracle. We set N(x) to be the examples obtainable from x by sampling
from a pretrained paraphrase model. As described in Section 5, to measure expansion between sets A
and B, we need a neighborhood oracle n : A →B that, given x ∈A, returns a point x′ ∈N(x)∩B.
Our results require us to measure expansion between (Sgood
i
, Ti), (Sbad
i
, Ti), and (Sbad
i
, Sgood
i
). For
x ∈Sgood
i
(resp. x ∈Sbad
i
), we generate a target point x′ ∈N(x) ∩T by rejection sampling from
a pretrained paraphrase model. Because ˜y takes a simple form, we can efficiently approximate this
step by setting the logits of tokens “horrible” and “incredible” to −∞during decoding so they are
never generated. For x ∈Sbad
i
, to generate a neighbor x′′ ∈N(x) ∩Sgood
i
, we prompt GPT-4
to paraphrase a randomly chosen sentence from x′ ∈N(x) ∩Ti and include the correct word in
its paraphrase, so that x′′ ∈Sgood
i
. We rejection sample until this constraint is satisfied. Figure 3
(appendix) shows examples of these procedures.

Expansion results. Table 1 measures the expansion of the set family M′(Sgood
i
, F) on the sets
(Sbad
i
, Sgood
i
) using the procedure from Section 5. Theorem 4.1 shows this is related to pseudolabel
correction. For the SentenceBERT model, the measured expansion is high and the student fits the
weak labels well, but doesn’t overfit to the teacher labels (i.e., err(f, ˜y|Si) > 0). For label 0, our
bound indicates that pseudolabel correction should be present, since the value for our error bound
is less than αi. There is indeed pseudolabel correction on this label: err(f, y|S0) < α0. For label
1, our bound indicates that pseudolabel correction may not occur—the bound value is greater than
αi since the the measured expansion is lower for this label and the error of the student on the weak
labels is higher. As suggested by the bound, there is no pseudolabel correction: err(f, y|S1) > α1.
Our expansion-based theory can therefore differentiate between cases where pseudolabel correction
does and does not occur. Table 4 (appendix) shows the measured expansion values between the set
pairs (Sgood
i
, Ti) and (Sbad
i
, Ti), which Theorem 4.2 shows are related to the amount of coverage
expansion. For example, for label 1, Ti expands to both Sgood
i
(c = 0.75) and to Sbad
i
(c = 0.55).
The fact that both expansions are nonzero gives evidence for the structure assumed Theorem 4.2. In
this case, our coverage expansion bounds show that the student model has nontrivial performance
on the uncovered sets Ti—for example, for label 1, the worst-case value of err(f, y|T1) in all the
training runs is 0.29, and the value of our bound is 0.33. Appendix E contains more details on how
the models are trained and the bounds are computed.

7
Limitations & Conclusion

In this work, we proved error bounds based on expansion properties of the data distribution and
student hypothesis class that directly allow for weak-to-strong generalization, gave a statistical theory
for checking these expansion properties, and gave empirical evidence that they hold in practice.
Our empirical procedure for finding the worst-expanding set generated by our hypothesis class is
ultimately still a heuristic, and our experiments are limited in scope. However, Sections 5 and 6 go
beyond prior work by testing our assumptions more carefully. Finally, this work contains no new weak
supervision algorithms (e.g., new training methods) for improving weak-to-strong generalization.
While our work does not propose new weak supervision algorithms, we believe our theory suggests
a framework for encouraging weak-to-strong generalization effects: find a neighborhood structure
and student hypothesis class pair that expands, then find the student model f ∈F that minimizes the
error on the weak labels while staying as robust as possible on the neighborhoods. Both expansion
and contrastive pre-training are related to spectral properties of the underlying neighborhood graph
[22]. Can we improve the performance of weakly-supervised learning by imbuing the contrastive
pretraining objective with knowledge of the pseudolabeler ˜y? Burns et al. [10] show that weak-to-
strong generalization effects are much stronger in classification problems than in other settings such
as reward modeling. Adapting our analysis techniques to reward modeling and using our results to
design algorithms for weak-to-strong reward model training are interesting directions for future work.

Acknowledgments

Thanks to Hussein Mozannar and Ilker Demirel for feedback on drafts of this paper. DS and HL
were partially supported by NSF AitF award CCF-1723344, AV was supported by NSF grants
EECS2216970 and CCF-2154100, and HL was supported by the NDSEG fellowship. Finally, this
project was partially supported by an OpenAI “Superalignment Fast” grant.

10


---Page Break---
References

[1] Abbe, E., Bengio, S., Lotfi, A., and Rizk, K. (2023). Generalization on the unseen, logic
reasoning and degree curriculum. In International Conference on Machine Learning, pages 31–60.
PMLR.

[2] Agrawal, M., Hegselmann, S., Lang, H., Kim, Y., and Sontag, D. (2022). Large language models
are few-shot clinical information extractors. In Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages 1998–2022.

[3] Balcan, M.-F., Blum, A., and Yang, K. (2004). Co-training and expansion: Towards bridging
theory and practice. Advances in neural information processing systems, 17.

[4] Ben-David, S., Blitzer, J., Crammer, K., and Pereira, F. (2006). Analysis of representations for
domain adaptation. Advances in neural information processing systems, 19.

[5] Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., and Wortman, J. (2007). Learning bounds for
domain adaptation. Advances in neural information processing systems, 20.

[6] Blum, A. and Mitchell, T. (1998). Combining labeled and unlabeled data with co-training. In
Proceedings of the eleventh annual conference on Computational learning theory, pages 92–100.

[7] Blum, M., Karp, R. M., Vornberger, O., Papadimitriu, C. H., and Yannakakis, M. (1981). The
complexity of testing whether a graph is a superconcentrator. Information Processing Letters,
13(4-5):164–167.

[8] Bousquet, O., Boucheron, S., and Lugosi, G. (2003). Introduction to statistical learning theory.
In Summer school on machine learning, pages 169–207. Springer.

[9] Buciluˇa, C., Caruana, R., and Niculescu-Mizil, A. (2006). Model compression. In Proceedings
of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining,
pages 535–541.

[10] Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y.,
Ecoffet, A., Joglekar, M., Leike, J., et al. (2023). Weak-to-strong generalization: Eliciting strong
capabilities with weak supervision. arXiv preprint arXiv:2312.09390.

[11] Cabannnes, V. A., Bach, F., and Rudi, A. (2021). Disambiguation of weak supervision leading
to exponential convergence rates. In International Conference on Machine Learning, pages
1147–1157. PMLR.

[12] Cai, T., Gao, R., Lee, J., and Lei, Q. (2021). A theory of label propagation for subpopulation
shift. In International Conference on Machine Learning, pages 1170–1182. PMLR.

[13] Chen, M. F., Fu, D. Y., Adila, D., Zhang, M., Sala, F., Fatahalian, K., and Ré, C. (2022). Shoring
up the foundations: Fusing model embeddings and weak supervision. In Uncertainty in Artificial
Intelligence, pages 357–367. PMLR.

[14] Chen, Y., Wei, C., Kumar, A., and Ma, T. (2020). Self-training avoids using spurious features
under domain shift. Advances in Neural Information Processing Systems, 33:21061–21071.

[15] Chiang, C.-K. and Sugiyama, M. (2023). Unified risk analysis for weakly supervised learning.
arXiv preprint arXiv:2309.08216.

[16] Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates
using the em algorithm. Journal of the Royal Statistical Society: Series C (Applied Statistics),
28(1):20–28.

[17] Ding, B., Qin, C., Liu, L., Chia, Y. K., Li, B., Joty, S., and Bing, L. (2023). Is GPT-3 a good
data annotator? In Rogers, A., Boyd-Graber, J., and Okazaki, N., editors, Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
11173–11195, Toronto, Canada. Association for Computational Linguistics.

11


---Page Break---
[18] Eisenstein, J., Andor, D., Bohnet, B., Collins, M., and Mimno, D. (2022). Honest students
from untrusted teachers: Learning an interpretable question-answering pipeline from a pretrained
language model. In Workshop on Trustworthy and Socially Responsible Machine Learning,
NeurIPS 2022.

[19] Frei, S., Zou, D., Chen, Z., and Gu, Q. (2022). Self-training converts weak learners to strong
learners in mixture models. In International Conference on Artificial Intelligence and Statistics,
pages 8003–8021. PMLR.

[20] Fries, J. A., Varma, P., Chen, V. S., Xiao, K., Tejeda, H., Saha, P., Dunnmon, J., Chubb,
H., Maskatia, S., Fiterau, M., et al. (2019). Weakly supervised classification of aortic valve
malformations using unlabeled cardiac mri sequences. Nature communications, 10(1):3111.

[21] Fu, D., Chen, M., Sala, F., Hooper, S., Fatahalian, K., and Ré, C. (2020). Fast and three-rious:
Speeding up weak supervision with triplet methods. In International Conference on Machine
Learning, pages 3280–3291. PMLR.

[22] HaoChen, J. Z., Wei, C., Gaidon, A., and Ma, T. (2021). Provable guarantees for self-supervised
deep learning with spectral contrastive loss. Advances in Neural Information Processing Systems,
34:5000–5011.

[23] HaoChen, J. Z., Wei, C., Kumar, A., and Ma, T. (2022). Beyond separability: Analyzing the
linear transferability of contrastive representations to related subpopulations. Advances in neural
information processing systems, 35:26889–26902.

[24] He, X., Lin, Z., Gong, Y., Jin, A., Zhang, H., Lin, C., Jiao, J., Yiu, S. M., Duan, N., Chen,
W., et al. (2023). Annollm: Making large language models to be better crowdsourced annotators.
arXiv preprint arXiv:2303.16854.

[25] Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B.,
Ball, R., Shpanskaya, K., et al. (2019). Chexpert: A large chest radiograph dataset with uncertainty
labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence,
volume 33, pages 590–597.

[26] Johnson, A. E., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-y.,
Mark, R. G., and Horng, S. (2019). Mimic-cxr, a de-identified publicly available database of chest
radiographs with free-text reports. Scientific data, 6(1):317.

[27] Kannan, R., Lovász, L., and Montenegro, R. (2006). Blocking conductance and mixing in
random walks. Combinatorics, Probability and Computing, 15(4):541–570.

[28] Karamanolakis, G., Mukherjee, S., Zheng, G., and Hassan, A. (2021). Self-training with
weak supervision. In Proceedings of the 2021 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, pages 845–863.

[29] Karger, D., Oh, S., and Shah, D. (2011). Iterative learning for reliable crowdsourcing systems.
Advances in neural information processing systems, 24.

[30] Khetan, A., Lipton, Z. C., and Anandkumar, A. (2018). Learning from noisy singly-labeled
data. In International Conference on Learning Representations.

[31] Kifer, D., Ben-David, S., and Gehrke, J. (2004). Detecting change in data streams. In VLDB,
volume 4, pages 180–191. Toronto, Canada.

[32] Kumar, A., Ma, T., and Liang, P. (2020). Understanding self-training for gradual domain
adaptation. In International conference on machine learning, pages 5468–5479. PMLR.

[33] Kuzman, T., Mozetic, I., and Ljubešic, N. (2023). Chatgpt: Beginning of an end of man-
ual linguistic data annotation?
use case of automatic genre identification.
arXiv preprint
arXiv:2303.03953.

[34] Kwok, T. C., Lau, L. C., and Lee, Y. T. (2016). Improved cheeger’s inequality and analysis
of local graph partitioning using vertex expansion and expansion profile. In Proceedings of the
Twenty-Seventh Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1848–1861. SIAM.

12


---Page Break---
[35] Lang, H., Agrawal, M. N., Kim, Y., and Sontag, D. (2022a). Co-training improves prompt-based
learning for large language models. In International Conference on Machine Learning, pages
11985–12003. PMLR.

[36] Lang, H. and Poon, H. (2021). Self-supervised self-supervision by combining deep learning and
probabilistic logic. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pages 4978–4986.

[37] Lang, H., Vijayaraghavan, A., and Sontag, D. (2022b). Training subset selection for weak
supervision. Advances in Neural Information Processing Systems, 35:16023–16036.

[38] Li, J., Zhang, J., Schmidt, L., and Ratner, A. (2023a). Characterizing the impacts of semi-
supervised learning for weak supervision. In Thirty-seventh Conference on Neural Information
Processing Systems.

[39] Li, X., Yu, P., Zhou, C., Schick, T., Zettlemoyer, L., Levy, O., Weston, J., and Lewis, M. (2023b).
Self-alignment with instruction backtranslation. arXiv preprint arXiv:2308.06259.

[40] Loshchilov, I. and Hutter, F. (2018). Decoupled weight decay regularization. In International
Conference on Learning Representations.

[41] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. (2011). Learning
word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association
for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon,
USA. Association for Computational Linguistics.

[42] Makarychev, K., Makarychev, Y., Shan, L., and Vijayaraghavan, A. (2023). Higher-order
cheeger inequality for partitioning with buffers. arXiv preprint arXiv:2308.10160.

[43] Meng, Y., Shen, J., Zhang, C., and Han, J. (2018). Weakly-supervised neural text classifica-
tion. In proceedings of the 27th ACM International Conference on information and knowledge
management, pages 983–992.

[44] Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2022). Mteb: Massive text embedding
benchmark. arXiv preprint arXiv:2210.07316.

[45] Natarajan, N., Dhillon, I. S., Ravikumar, P. K., and Tewari, A. (2013). Learning with noisy
labels. Advances in neural information processing systems, 26.

[46] Northcutt, C., Jiang, L., and Chuang, I. (2021). Confident learning: Estimating uncertainty in
dataset labels. Journal of Artificial Intelligence Research, 70:1373–1411.

[47] Oymak, S. and Cihad Gulcu, T. (2021). A theoretical characterization of semi-supervised
learning with self-training for gaussian mixture models. In Banerjee, A. and Fukumizu, K., editors,
Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume
130 of Proceedings of Machine Learning Research, pages 3601–3609. PMLR.

[48] Pukdee, R., Sam, D., Ravikumar, P. K., and Balcan, N. (2022). Label propagation with weak
supervision. In The Eleventh International Conference on Learning Representations.

[49] Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., and Ré, C. (2017). Snorkel: Rapid train-
ing data creation with weak supervision. In Proceedings of the VLDB Endowment. International
Conference on Very Large Data Bases, volume 11, page 269. NIH Public Access.

[50] Ratner, A., Hancock, B., Dunnmon, J., Sala, F., Pandey, S., and Ré, C. (2019). Training complex
models with multi-task weak supervision. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pages 4763–4771.

[51] Ratner, A. J., De Sa, C. M., Wu, S., Selsam, D., and Ré, C. (2016). Data programming: Creating
large training sets, quickly. Advances in neural information processing systems, 29.

[52] Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. In Inui, K., Jiang, J., Ng, V., and Wan, X., editors, Proceedings of the 2019
Conference on Empirical Methods in Natural Language Processing and the 9th International Joint
Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong,
China. Association for Computational Linguistics.

13


---Page Break---
[53] Robinson, J., Jegelka, S., and Sra, S. (2020). Strength from weakness: Fast learning using weak
supervision. In International Conference on Machine Learning, pages 8127–8136. PMLR.

[54] Rühling Cachay, S., Boecking, B., and Dubrawski, A. (2021). End-to-end weak supervision.
Advances in Neural Information Processing Systems, 34:1845–1857.

[55] Sam, D. and Kolter, J. Z. (2023). Losses over labels: Weakly supervised learning via direct loss
construction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages
9695–9703.

[56] Sauer, N. (1972). On the density of families of sets. Journal of Combinatorial Theory, Series A,
13(1):145–147.

[57] Shelah, S. (1972). A combinatorial problem; stability and order for models and theories in
infinitary languages. Pacific Journal of Mathematics, 41(1):247–261.

[58] Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., and Wilson, A. G. (2021). Does
knowledge distillation really work? Advances in Neural Information Processing Systems, 34:6906–
6919.

[59] Sugiyama, M., Bao, H., Ishida, T., Lu, N., Sakai, T., and Niu, G. (2022). Machine Learning
from Weak Supervision: An Empirical Risk Minimization Approach. MIT Press.

[60] Tsybakov, A. B. (2004). Optimal aggregation of classifiers in statistical learning. The Annals of
Statistics, 32(1):135–166.

[61] Vapnik, V. (1971). On the uniform convergence of relative frequencies of events to their
probabilities. Theory of Probability and its Applications, 16(2):264–281.

[62] Varma, P. and Ré, C. (2018). Snuba: Automating weak supervision to label training data.
In Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases,
volume 12, page 223. NIH Public Access.

[63] Varma, P., Sala, F., He, A., Ratner, A., and Ré, C. (2019). Learning dependency structures for
weak supervision models. In International Conference on Machine Learning, pages 6418–6427.
PMLR.

[64] Veselovsky, V., Ribeiro, M. H., Cozzolino, P., Gordon, A., Rothschild, D., and West, R. (2023).
Prevalence and prevention of large language model use in crowd work.

[65] Wang, S., Liu, Y., Xu, Y., Zhu, C., and Zeng, M. (2021). Want to reduce labeling cost? gpt-3
can help. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages
4195–4205.

[66] Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. (2023).
Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 13484–13508, Toronto, Canada. Association for Computational Linguistics.

[67] Wang, Y., Sohn, S., Liu, S., Shen, F., Wang, L., Atkinson, E. J., Amin, S., and Liu, H. (2019).
A clinical text classification paradigm using weak supervision and deep representation. BMC
medical informatics and decision making, 19:1–13.

[68] Wei, C. and Ma, T. (2019). Improved sample complexities for deep neural networks and robust
classification via an all-layer margin. In International Conference on Learning Representations.

[69] Wei, C., Shen, K., Chen, Y., and Ma, T. (2020). Theoretical analysis of self-training with deep
networks on unlabeled data. In International Conference on Learning Representations.

[70] Yu, Y., Zuo, S., Jiang, H., Ren, W., Zhao, T., and Zhang, C. (2021). Fine-tuning pre-trained
language model with weak supervision: A contrastive-regularized self-training approach. In
Proceedings of the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pages 1063–1077.

14


---Page Break---
[71] Zhang, J., Hsieh, C.-Y., Yu, Y., Zhang, C., and Ratner, A. (2022a). A survey on programmatic
weak supervision. arXiv preprint arXiv:2202.05433.

[72] Zhang, J., Wang, H., Hsieh, C.-Y., and Ratner, A. J. (2022b). Understanding programmatic
weak supervision via source-aware influence function. Advances in Neural Information Processing
Systems, 35:2862–2875.

[73] Zhang, J., Yu, Y., Li, Y., Wang, Y., Yang, Y., Yang, M., and Ratner, A. (2021). WRENCH: A
comprehensive benchmark for weak supervision. In Thirty-fifth Conference on Neural Information
Processing Systems Datasets and Benchmarks Track.

[74] Zhang, Y., Bai, Y., Ding, M., Li, Y., and Ghanem, B. (2018). W2f: A weakly-supervised
to fully-supervised framework for object detection. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 928–936.

15


---Page Break---
A
Additional Related Work

Programmatic Weak Supervision. Pukdee et al. [48] study the performance of label propagation
for weak supervision. Unlike our work, they assume expansion with respect to the empirical sample
graph and study the performance of a particular learning algorithm (label propagation). Chen et al.
[13] show how to propagate weak labels using the embeddings of a strong pretrained model to
provably improve performance, but they also focus on the empirical graph, and not on guarantees
for the classifiers trained on the weak labels. Unlike most other work on weak supervision, which
tries to give separate algorithms and guarantees for (i) the procedure for creating a pseudolabel out
of multiple labeling functions and (ii) the student training, Sam and Kolter [55], Rühling Cachay
et al. [54] consider direct end-to-end weak supervision. However, they do not give theoretical error
guarantees for the student models. Cabannnes et al. [11] give theoretical guarantees for a different
flavor of weak supervision, when the “teacher” gives a set of labels that contains the ground-truth but
may also contain other incorrect labels. Robinson et al. [53] show that when only a few gold labels
are available, weak labels from a teacher model can speed up the learning process.

Domain Adaptation. What we call “coverage expansion” is very related to some work on domain
adaptation that can still apply when the source and target distributions do not overlap, such as Ben-
David et al. [4], Blitzer et al. [5]. Our expansion assumptions for coverage expansion, which require
the expansion of certain families of sets (generated by the student hypothesis class), are qualitatively
very related to the F∆F distance [4]. Assuming the F∆F distance is small essentially says that
the mistake set of any hypothesis in F must have similar probability in both the source and target
distribution, so the target error can’t be much higher than the source error. Kifer et al. [31] showed
that this distance can be statistically estimated from a finite sample. In an imprecise sense, our results
suggest that the F∆F distance is small when every hypothesis f ∈F is robust on the neighborhoods
N and the distribution has good expansion. We also have one unified set of assumptions that leads
to guarantees for both coverage expansion/domain adaptation and pseudolabel correction, and our
coverage expansion bounds account for the error of the teacher model on the source domain. Abbe
et al. [1] also give theoretical guarantees for learning boolean functions when part of the data domain
is completely unseen during training.

Knowledge Distillation, Pseudolabel correction. The observation that a distilled model can
outperform its teacher (i.e., what Burns et al. [10] calls weak-to-strong generalization) goes back
at least to Buciluˇa et al. [9]. In the context of knowledge distillation, Stanton et al. [58] show that
even when student models have the capacity to match the teacher, they don’t match them exactly.
They give empirical examples of this phenomenon and argue that it can be due to optimization issues
during student training. Clearly, this effect is critical for weak-to-strong generalization, and our
expansion and robustness assumptions effectively try to capture the structure (whether implicit, due
to the optimization process, or explicit, due to the choice of student hypothesis class) that rules
out this exact-teacher-fitting behavior. In the context of self-training, Chen et al. [14], Kumar et al.
[32], Oymak and Cihad Gulcu [47], Frei et al. [19] all (effectively) give pseudolabel correction
guarantees under certain distributional assumptions, such as when the data are distributed according
to a Gaussian Mixture Model [47, 19]. In contrast, like other work with expansion assumptions
[3, 69, 12, 22], our results do not assume the input data follows any specific distributional form.
Indeed, Wei et al. [69] showed that distributions similar to the Gaussian Mixture Models studied in
these works satisfy a notion of expansion very similar to the one we study here.

B
Error bound proofs

We directly prove the “average-case-robust” versions of Theorems 4.1 and 4.2. Theorems 4.1 and
4.2 directly follow from the equivalence between expansion and robust expansion, and adversarial
robustness and η-robustness, when η = 0. For convenience, we reproduce the definitions of the
example graph, η-robustness, and robust expansion here. Figure 2 shows examples of good and bad
robust expansion.

We begin by proving some lemmas that will be useful for both the pseudolabel correction and
coverage expansion bounds.
Definition (Example graph). Let G = (X, E) be a graph with one node for each element of X (we
assumed X is a possibly very large, but finite, set), and connect two nodes (x, x′) if x ∈N(x′) or,
equivalently, if x′ ∈N(x), with an edge weight of w(x, x′) := P(x)P(x′)1[x ∈N(x′)].

16


---Page Break---
•
•
•
•

• •

•
•

••
••
•
•

•
•

•

neighborhood

robust 
neighborhood

U
N(U)

V

neighborhood
•
•
•
•

• •

•
•

••
••
•
•

•
•

•

robust 
neighborhood

U

N(U)

V

Figure 2: Examples of good (left) and bad (right) robust expansion. In both cases, there is a core
subset V ⊂N(U) that accounts for most of the edge weight incident on U (at least a 1 −η fraction,
for some small η). The robust expansion is good when every such subset has large probability.

Definition (η-robust). For an arbitrary classifier f and point x, define r(f, x) = P(f(x′) ̸=
f(x)|x′ ∈N(x)) as the probability f gives different labels to x and a random neighbor x′ of x. A
classifier f : X →Y is said to be η-robust at a point x if r(f, x) ≤η. For an arbitrary classifier f,
let Rη(f) = {x : r(f, x) ≤η} be the set of η-robust points for f.
Definition (η-robust neighborhood size). Let A, U ⊂X. The size of the η-robust neighborhood of U
in A is: P1−η(U, A) := minV ⊂X {P(V |A) : w(V, U) ≥(1 −η)w(N(U), U)}.
Definition (Robust expansion). Fix sets A, B ⊂X and suppose M is a collection of subsets
of B. M satisfies (c, q, η)-robust expansion on (A, B) if for all U ∈M with P(U|B) > q,
P1−η(U, A) > cP(U|B). This exactly recovers Definition 4 when η = 0.

The following lemma shows that a classifier that is robust on average must also be η-robust on a set
of large probability.
Lemma B.1. Fix a set A ⊂X and a classifier f, and suppose

Ex∼D|A,x′∼N(x)[f(x) ̸= f(x′)] ≤γ

for some γ > 0. Then for any η > 0, P(Ğ
Rη(f)|A) ≤γ

η .

Proof. Rewriting the condition on f slightly,

Ex∼D|A[Px′(f(x) ̸= f(x′)|x′ ∈N(x))] ≤γ.

Recall that r(x, f) = Px′(f(x) ̸= f(x′)|x′ ∈N(x)). So we have Ex∼D|A[r(x, f)] ≤γ. Markov’s
inequality implies that for any η > 0,

P(r(x, f) > η|A) ≤Ex∼D|A[r(x, f)]

η
≤γ

η .

Since Rη(f) = {x : r(x, f) ≤η}, we thus have

P(Ğ
Rη(f)|A) ≤γ

η ,

which concludes the proof.

Good and bad edges.
Consider an arbitrary classifier f and arbitrary set U ⊂X and fix x′ ∈U,
x ∈N(U). We say the pair (x, x′) is bad if f(x) ̸= f(x′); otherwise the pair is good. Let e
N(U) be
the subset of N(U) reachable by good edges. Formally, e
N(x′) = {x ∈N(x′) : (x, x′) good} and
e
N(A) = ∪x′∈A e
N(x′). We are suppressing the dependence of e
N on the classifier f for notational
convenience. The following lemma guarantees that if f is η-robust on all points in U, bad edges do
not account for much of the weight between N(U) and U in the example graph (Definition 5). This
is the key to our average-case-robustness results, since it implies that if the robust expansion is large
and f is η-robust on U, the neighborhood e
N(U) of points reachable by good edges must be large.
Lemma B.2. Consider an arbitrary set U ⊂X and suppose that for all x ∈U, f is such that
r(f, x) ≤η. I.e., U ⊂Rη(f). Then:

w( e
N(U), U) ≥(1 −η)w(N(U), U).

17


---Page Break---
Proof. Since every x′ ∈U is in Rη(f), we have:

η ≥r(f, x′) = Px(f(x) ̸= f(x′)|x ∈N(x′))

= Px(f(x) ̸= f(x′), x ∈N(x′))

P(x ∈N(x′))

=
P

x∈X 1[f(x) ̸= f(x′)]1[x ∈N(x′)]P(x)
P

x∈X 1[x ∈N(x′)]P(x)

=
P

x∈X 1[f(x) ̸= f(x′)]1[x ∈N(x′)]P(x)P(x′)
P

x∈X 1[x ∈N(x′)]P(x)P(x′)

=
P

x∈X 1[f(x) ̸= f(x′)]w(x, x′)
P

x∈X w(x, x′)
,

so for all x′ ∈U we have:X

x∈X
1[(x, x′) bad]w(x, x′) ≤η
X

x∈X
w(x, x′).

Because w(x, x′) > 0 ⇐⇒x ∈N(x′), we can replace the summations over all of X with the sum
over N(x′), so:
X

x∈N(x′)
1[(x, x′) bad]w(x, x′) ≤η
X

x∈N(x′)
w(x, x′).
(2)

Now we can simplify:

w( e
N(U), U) =
X

x′∈U

X

x∈e
N(x′)
w(x, x′)

≥
X

x′∈U

X

x∈N(x′)
1[(x, x′) good]w(x, x′),

where the inequality is because some elements of e
N(U) might be reachable by a mixture of good
and bad edges. The left-hand-side counts both, and the right-hand-side only counts the contribution
of the good edges. Note that 1[(x, x′) good] = 1 −1[(x, x′) bad]. Plugging this in gives:

w( e
N(U), U) ≥
X

x′∈U

X

x∈N(x′)
(1 −1[(x, x′) bad])w(x, x′)

=
X

x′∈U

X

x∈N(x′)
w(x, x′) −
X

x′∈U

X

x∈N(x′)
1[(x, x′) bad]w(x, x′)

≥
X

x′∈U

X

x∈N(x′)
w(x, x′) −η
X

x′∈U

X

x∈N(x′)
w(x, x′)

= (1 −η)
X

x′∈U

X

x∈N(x′)
w(x, x′)

= (1 −η)w(N(U), U).

where we used (2) in the second inequality.

Expanding set family.
Now we construct the set families that must expand for our results in this
section. Let F be the hypothesis class for the strong model, y the ground-truth function, and B ⊂X
an arbitrary set. For each f ∈F let U(B, f) = {x ∈B : f(x) ̸= y(x)} be the set of f’s mistakes
on y in B. Then we define the family:

Mη(B, F) = {Rη(f) ∩U(B, f) : f ∈F}.

That is, Mη(B, F) is the family of η-robust (Definition 2) mistake sets of F on the set B. This
exactly agrees with the definition of M(B, F) from Section 4 when η = 0. Similarly, we define:

M′
η(B, F) = {Rη(f) ∩(B \ U(B, f)) : f ∈F}

as the family of η-robust non-mistake sets. Again, this exactly agrees with the definition of M′(B, F)
from Section 4 when η = 0.

18


---Page Break---
B.1
Pseudolabel correction

We directly prove the “average-case-robust” version of Theorem 4.1. Theorem 4.1 then follows from
the equivalence between expansion and robust expansion, and adversarial robustness and η-robustness,
when η = 0.

Theorem B.1 (Pseudolabel correction). Suppose M ′
η(Sgood
i
, F) satisfies (c, q, η)-robust expansion
on (Sbad
i
, Sgood
i
) for some c > 0 and η ≥0. Consider an arbitrary classifier f such that P(f(x) ̸=
˜y(x) or f not η-robust at x|Si) ≤1 −q −αi. Then f satisfies the following error bound:

err(f, y|Si) ≤err(f, ˜y|Si) −αi(2c′ −1) + 2c′αiP(Ğ
Rη(f)|Si)
1 −2c′αi

where c′ = c/(1 −αi + cαi).

Corollary B.1 (Simplified bound, average-case-robust version of Theorem 4.1). Suppose the condi-
tions of Theorem B.1 hold. For any ∆such that err(f, ˜y|Si) ≤cαi∆+ (1 −αi)(1 −∆), the error
of f satisfies:

err(f, y|Si) ≤
2αi
1 −2αi
P(Ğ
Rη(f)|Si) + err(f, ˜y|Si) + αi(1 −2c∆).

Moreover, if f satisfies
Ex∼D|Si,x′∼D|N(x)[f(x) ̸= f(x′)] ≤γ,

then

err(f, y|Si) ≤
2αi
1 −2αi

γ
η + err(f, ˜y|Si) + αi(1 −2c∆).

Proof. The first part directly follows from Theorem B.1 by substituting the value of c′, using c ≤1,
and using the form of ∆to simplify the bound. Theorem 4.1 follows from taking ∆= 3/4, which
was small enough for the error condition to hold empirically. The condition on q in Theorem 4.1
implies that if P(f(x) ̸= ˜y(x) or f not η-robust at x|Si) ≤1−αi+3cαi

4
, then

P(f(x) ̸= ˜y(x) or f not η-robust at x|Si) ≤1 −q −αi,

so the conditions of Theorem B.1 hold. This upper bound on q could be replaced by instead assuming

P(f ̸= ˜y(x) or f not η-robust at x|Si) ≤min

1 −q −αi, 1 −αi + 3cαi

4


.

The second part follows directly from Lemma B.1.

Proof of Theorem B.1. Let Mi = {x ∈Si : f(x) ̸= y(x)} be the set of mistakes of f on the true
labels in Si. Similarly, let Di = {x ∈Si : f(x) ̸= ˜y(x)} be the set of mistakes of f on the weak
labels in Si. Define Ui = Si \ Mi and let Vi = Rη(f) ∩Ui ∩Sgood
i
. Note that Vi ∈M′(Sgood
i
, F),
so if it’s large enough, it expands according to our expansion assumption. The following lemma
shows that Vi is large enough to expand.

Lemma B.3. P(Vi|Sgood
i
) > q.

Proof. Suppose for a contradiction that P(Vi|Sgood
i
) ≤q. Then by definition of Vi,

P(Vi|Sgood
i
) = 1 −P(Vi|Sgood
i
)

= 1 −P(Vi ∩Sgood
i
|Sgood
i
)

= 1 −P(((Si \ Mi) ∩Rη(f)) ∩Sgood
i
|Sgood
i
)

= 1 −P((Mi ∩Sgood
i
) ∪Rη(f)|Sgood
i
)

19


---Page Break---
Fix an arbitrary x ∈Mi ∩Sgood
i
. By definition of Mi, f(x) ̸= y(x). By definition of Sgood
i
,
˜y(x) = y(x). Hence f(x) ̸= ˜y(x), so x ∈Di, and therefore Mi ∩Sgood
i
⊂Di. Then:

q ≥P(Vi|Sgood
i
) ≥1 −P(Di ∪Rη(f)|Sgood
i
)

= 1 −(1 −αi)P((Di ∪Rη(f)) ∩Sgood
i
|Si)

≥1 −(1 −αi)P((Di ∪Rη(f))|Si)
= 1 −(1 −αi)P(f(x) ̸= ˜y(x) or f not η-robust at x|Si)
≥1 −(1 −αi)(1 −q −αi)
= q + αi(2 −q −αi)
> q
The second-to-last inequality used the assumption on P(f(x) ̸= ˜y(x) or f not η-robust at x|Si) and
the final inequality used q < 1 and 0 < αi < 1/2. Assuming P(Vi|Sgood
i
) ≤q thus leads to q > q, a
contradiction. So P(Vi|Sgood
i
) > q.

Recall that for a set A ⊂X, we define e
N(A) ⊂N(A) as the subset of points reachable from A by a
good edge f(x) = f(x′). By Lemma B.2, since Vi ⊂Rη(f),

w( e
N(Vi), Vi) ≥(1 −η)w(N(Vi), Vi).

Then by Lemma B.3, since (Sbad
i
, Sgood
i
) satisfy (c, q, η)-robust expansion,

P( e
N(Vi)|Sbad
i
) ≥P1−η(Vi, Sbad
i
) ≥cP(Vi|Sgood
i
).

Fix an arbitrary x ∈e
N(Vi). By definition of e
N, there exists x′ ∈Vi such that f(x) = f(x′). Since
x′ ∈Vi ⊂Ui, we have that f(x) = f(x′) = y(x′) (since x′ ̸∈Mi), and y(x′) = y(x), since x and
x′ are both in Si. This shows f(x) = y(x) and therefore x ̸∈Mi, so x ∈Ui = Si \ Mi. Since x
was arbitrary, e
N(Vi) ⊂Ui. Thus,

P(Ui|Sbad
i
) ≥cP(Vi|Sgood
i
).
(3)
Points x ∈Ui are correctly labeled by f, and points x ∈Sbad
i
are incorrectly labeled by the
pseudolabeler ˜y. This inequality thus guarantees that there must be some points that f gets correct
and ˜y gets incorrect. It also gives a quantitative lower bound on how much probability is assigned to
these points. Intuitively, (3) is already a “pseudolabel correction” result, and the remainder of the
proof is deriving a final bound on the error from (3).

The following lemma converts (3) into a relationship between P(Ui|Sbad
i
) and P(Ui ∩Rη(f)|Si).

Lemma B.4. P(Ui|Sbad
i
) ≥c′P(Ui ∩Rη(f)|Si) + c′αiP(Ui ∩Rη(f)|Sbad
i
), where c′ = c/(1 −
αi + cαi).

Proof.

P(Ui|Sbad
i
) ≥cP(Vi|Sgood
i
)
(by (3))

=
c
1 −αi
P(Vi ∩Sgood
i
|Si)

=
c
1 −αi
P(Ui ∩Rη(f) ∩Sgood
i
|Si)

=
c
1 −αi

 
P(Ui ∩Rη(f)|Si) −P(Ui ∩Rη(f) ∩Sbad
i
|Si)


Rearranging,
c
1 −αi
P(Ui ∩Rη(f)|Si) ≤P(Ui|Sbad
i
) +
c
1 −αi
P(Ui ∩Rη(f) ∩Sbad
i
|Si)

= P(Ui|Sbad
i
) +
cαi
1 −αi
P(Ui ∩Rη(f)|Sbad
i
)

= P(Ui|Sbad
i
) +
cαi
1 −αi


P(Ui|Sbad
i
) −P(Ui ∩Rη(f)|Sbad
i
)


= P(Ui|Sbad
i
)

1 +
cαi
1 −αi


−
cαi
1 −αi
P(Ui ∩Rη(f)|Sbad
i
)

20


---Page Break---
And finally,

c
1 −αi + cαi
P(Ui ∩Rη(f)|Si) +
cαi
1 −αi + cαi
P(Ui ∩Rη(f)|Sbad
i
) ≤P(Ui|Sbad
i
).

The following lemma decomposes the disagreement set Di into its components in Sgood
i
and Sbad
i
.

Lemma B.5. Di ⊃(Mi ∩Sgood
i
) ∪(Ui ∩Sbad
i
).

Proof. First, fix x ∈Mi ∩Sgood
i
. Since x ∈Mi, f(x) ̸= y(x). Since x ∈Sgood
i
, ˜y(x) = y(x).
Thus f(x) ̸= ˜y(x), so x ∈Di. Second, fix x ∈Ui ∩Sbad
i
. Since x ∈Ui = Si \ Mi, f(x) = y(x).
Since x ∈Sbad
i
, ˜y(x) ̸= y(x). Thus f(x) ̸= ˜y(x), so x ∈Di.

By Lemma B.5 and using P(Sgood
i
|Si) = 1 −P(Sbad
i
|Si) = 1 −αi,

P(Di|Si) ≥P(Mi|Sgood
i
)(1 −αi) + P(Ui|Sbad
i
)αi

Applying Lemma B.4,

P(Di|Si) ≥(1 −αi)P(Mi|Sgood
i
) + c′αi

P(Ui ∩Rη(f)|Si) + αiP(Ui ∩Rη(f)|Sbad
i
)

.
(4)

Applying Lemma B.4 again,

P(Ui|Si) = αiP(Ui|Sbad
i
) + (1 −αi)P(Ui|Sgood
i
)

≥c′αi

P(Ui ∩Rη(f)|Si) + αiP(Ui ∩Rη(f)|Sbad
i
)

+ (1 −αi)P(Ui|Sgood
i
),

so we have

P(Ui|Sgood
i
) ≤
1
1 −αi


P(Ui|Si) −c′αi

P(Ui ∩Rη(f)|Si) + αiP(Ui ∩Rη(f)|Sbad
i
)


Combining this with Ui = Si \ Mi,

P(Mi|Sgood
i
) = 1 −P(Ui|Sgood
i
)

≥1 −
1
1 −αi


P(Ui|Si) −c′αi

P(Ui ∩Rη(f)|Si) + αiP(Ui ∩Rη(f)|Sbad
i
)


Plugging this into (4) gives:

P(Di|Si) ≥(1 −αi) −P(Ui|Si) + 2c′αi

P(Ui ∩Rη(f)|Si) + αiP(Ui ∩Rη(f)|Sbad
i
)


≥(1 −αi) −(1 −P(Mi|Si)) + 2c′αi

1 −P(Mi ∪Rη(f)|Si) + P(Ui ∩Rη(f) ∩Sbad
i
|Si)


= P(Mi|Si) + αi(2c′ −1) + 2c′αi

P(Ui ∩Rη(f) ∩Sbad
i
|Si) −P(Mi ∪Rη(f)|Si)


≥P(Mi|Si) + αi(2c′ −1) −2c′αiP(Mi ∪Rη(f)|Si)

Let p = P(Mi|Si). Then, using the union bound for the last term,

P(Di|Si) ≥p + αi(2c′ −1) −2c′αi(p + P(Rη(f)|Si))

= (1 −2c′αi)p + αi(2c′ −1) −2c′αiP(Rη(f)|Si)

Since P(Di|Si) = err(f, ˜y|Si), rearranging terms we get:

err(f, y|Si) = P(Mi|Si) ≤err(f, ˜y|Si) −αi(2c′ −1) + 2c′αiP(Rη(f)|Si)

1 −2c′αi
.

21


---Page Break---
B.2
Coverage expansion

We directly prove the “average-case-robust” version of Theorem 4.2. Theorem 4.2 then follows from
the equivalence between expansion and robust expansion, and adversarial robustness and η-robustness,
when η = 0. The following theorem also allows for Ti to expand to Sgood
i
and Sbad
i
in different
amounts, controlled by the two expansion parameters c1 and c2. This allows for empirically tighter
bounds and is used in our experiments in Section 6. Theorem 4.2 is a special case of Theorem B.2
taking η = 0 and c1 = c2.
Theorem B.2 (Average-case-robust version of Theorem 4.2). Suppose there exists c1, c2, q, η >
0 such that Mη(Ti, F, y) satisfies (c1, q, η)-expansion on (Sgood
i
, Ti) and M ′
η(Ti, F, y) satisfies
(c2, q, η)-expansion on (Sbad
i
, Ti). Fix an arbitrary classifier f : X →Y that satisfies:

err(f, ˜y|Si) + P(Rη(f)|Ti) < c1(1 −q −αi)
Let ¯c = max{c1, c2} and c = min{c1, c2}. Then the error of f on Ti is at most:

err(f, y|Ti) ≤

1 +
αi
c/¯c −2αi


P(Rη(f)|Ti) + max

q, err(g, ˜y|Si) −c2αi

c1 −(c1 + c2)αi)


.

Moreover, if f satisfies
Ex∼D|Ti,x′∼D|N(x)[f(x) ̸= f(x′)] ≤γ,
then the error of f on Ti is bounded by:

err(f, y|Ti) ≤

1 +
αi
c/¯c −2αi

 γ

η + max

q, err(g, ˜y|Si) −c2αi

c1 −(c1 + c2)αi)


.

Proof. Let Mi = {x ∈Ti : f(x) ̸= y(x)} and Ci = Ti \ Mi = {x ∈Ti : f(x) = y(x)}.
Define their robust subsets as Ui = Mi ∩Rη(f) and Vi = Ci ∩Rη(f). We have Ui ∈Mη(Ti, F)
and Vi ∈M′
η(Ti, F). Recall that for a set A, e
N(A) is the subset of N(A) consisting of neighbors
reachable by good edges (edges where f assigns the same label to both endpoints). Since Ui ⊂Rη(f),
Lemma B.2 implies that e
N(Ui) has enough weight to qualify as one of the sets in the definition
of the robust neighborhood size P1−η(U, Sgood
i
). In particular, it implies that P( e
N(Ui)|Sgood
i
) ≥
P1−η(Ui, Sgood
i
). Similarly, e
N(Vi) satisfies P( e
N(Vi)|Sbad
i
) ≥P1−η(Vi, Sbad
i
). There are now three
cases to consider:

Case 1: P(Ui|Ti) > q, P(Vi|Ti) > q. In this case, the expansion assumptions imply:

P( e
N(Ui)|Sgood
i
) ≥c1P(Ui|Ti)

P( e
N(Vi)|Sbad
i
) ≥c2P(Vi|Ti)

For all x ∈e
N(Ui) ∩Sgood
i
there exists x′ ∈Ui with f(x) = f(x′). Then:

f(x) = |
{z
}
x′∈Ui

f(x′) ̸=

Def. of Si,Ti
z
}|
{
y(x′) = y(x) = ˜y(x),
|
{z
}

x∈Sgood
i

so f(x) ̸= ˜y(x). Similarly, for all x ∈e
N(Vi) ∩Sbad
i
there exists x′ ∈Vi with f(x) = f(x′). Then

f(x) = |
{z
}
x′∈Vi

f(x′) =

Def. of Si,Ti
z
}|
{
y(x′) = y(x) ̸= ˜y(x),
|
{z
}
x∈Sbad
i

so f(x) ̸= ˜y(x).

Let Di = {x ∈Si : f(x) ̸= ˜y(x)} be the set of points in Si where f and the teacher ˜y disagree.
These arguments showed that e
N(Ui) ∩Sgood
i
⊂Di and e
N(Vi) ∩Sbad
i
⊂Di. Since these sets are
disjoint,

P(Di|Si) ≥P( e
N(Ui) ∩Sgood
i
|Si) + P( e
N(Ui) ∩Sbad
i
|Si)

= P( e
N(Ui)|Sgood
i
)(1 −αi) + P( e
N(Ui)|Sbad
i
)αi
≥c1P(Ui|Ti)(1 −αi) + c2P(Vi|Ti)αi.

22


---Page Break---
Since Ui ∪Vi = Rη(f) ∩Ti and Ui and Vi are disjoint subsets of Ti,
P(Vi|Ti) = P(Rη(f)|Ti) −P(Ui|Ti).
Plugging this into the previous inequality and simplifying gives:

P(Ui|Ti) ≤P(Di|Si) −c2αiP(Rη(f)|Ti)

c1 −(c1 + c2)αi
= P(Di|Si) −c2αi + c2αiP(Rη(f)|Ti)

c1 −(c1 + c2)αi

≤P(Di|Si) −c2αi

c1 −(c1 + c2)αi
+
αi
c/¯c −2αi
P(Rη(f)|Ti).

We can then bound P(Mi|Ti) = P(Ui|Ti) + P(Mi ∩Rη(f)|Ti) ≤P(Ui|Ti) + P(Rη(f)|Ti), so all
combined, we have:

P(Mi|Ti) ≤

1 +
αi
c/¯c −2αi


P(Rη(f)|Ti) + P(Di|Si) −c2αi

c1 −(c1 + c2)αi

Case 2: P(Ui|Ti) ≤q. Here we directly upper-bound P(Ui|Ti) with q and P(Mi|Ti) ≤P(Ui|Ti) +
P(Rη(f)|Ti) ≤q + P(Rη(f)|Ti).

Case 3: P(Ui|Ti) > q, P(Vi|Ti) ≤q. Since P(Ui|Ti) + P(Vi|Ti) = P(Rη(f)|Ti), in this case we
have P(Ui|Ti) ≥P(Rη(f)|Ti) −q. Since P(Ui|Ti) > q, Ui expands, so as in Case 1 we have:

P( e
N(Ui)|Sgood
i
) ≥c1P(Ui|Ti),
and therefore:
P(Di|Si) ≥P( e
N(Ui)|Sgood
i
)(1 −αi) ≥c1(1 −αi)P(Ui|Ti) ≥c1(1 −αi)(P(Rη(f)|Ti) −q).
But P(Di|Si) = err(g, ˜y|Si) and we assumed:

err(g, ˜y|Si) + P(Rη(f)|Ti) < c1(1 −q −αi) ≤c1(1 −q)(1 −αi),
since c1 ≤1, this implies
err(g, ˜y|Si) + c1(1 −αi)P(Rη(f)|Ti) < c1(1 −q)(1 −αi),
so:
err(g, ˜y|Si) < c1(1 −P(Rη(f)|Ti) −q)(1 −αi)
= c1(P(Rη(f)|Ti) −q)(1 −αi).
So Case 3 leads to a contradiction. Combining Cases 1 and 2 yields the final bound. The second part
follows directly from Lemma B.1.

If c1 = c2 ≡c, we have c/¯c = 1 and c1 −(c1 + c2)αi = c(1 −2αi), which gives a simpler bound
for coverage expansion of average-case-robust classifiers. When η = 0, P(Rη(f)|Ti) := P(R(f)|Ti)
by definition, and (c, q, η)-expansion is equivalent to (c, q)-expansion. These two simplifications
directly yield Theorem 4.2 for coverage expansion of adversarially robust classifiers.

The following theorem gives a more loose bound that only assumes expansion from Ti to Sgood
i
.
Theorem B.3 (Coverage expansion, weaker assumptions). Suppose Mη(Ti, F) satisfies (c, q, η)-
robust expansion on (Sgood
i
, Ti) for some c > 0. Fix an arbitrary classifier f : X →Y. The error of
f on Ti is bounded by:

err(f, y|Ti) ≤P(Rη(f)|Ti) + max

q, err(f, ˜y|Si)

c(1 −αi)


.

Proof. Define Mi = {x : f(x) ̸= y(x)} ∩Ti as the set of mistakes of f in Ti, and let Ui =
Mi ∩Rη(f). Let Di = {x ∈Si : f(x) ̸= ˜y(x)} be the set of points in Si where f disagrees
with the weak labels. Note that err(f, ˜y|Si) = P(Di|Si) and err(f, y|Ti) = P(Mi|Ti) ≤P(Ui|Ti) +
P(Rη(f)|Ti). So the goal is to bound P(Ui|Ti). Recall that e
N(U) is the subset of N(U) consisting
of neighbors reachable by good edges (edges where f assigns the same label to both endpoints).
Since Ui ⊂Rη(f), Lemma B.2 implies that e
N(Ui) has enough weight to qualify as one of the sets
V in the definition of the robust neighborhood size P1−η(U, Sgood
i
). In particular, it implies that
P( e
N(Ui)|Sgood
i
) ≥P1−η(U, Sgood
i
). Note that Ui ∈Mη(Ti, F) since it is a mistake set intersected
with a robust set. Then, if P(Ui|Ti) > q, (c, q, η)-robust expansion implies that

P( e
N(Ui)|Sgood
i
) ≥P1−η(U, Sgood
i
) > cP(Ui|Ti).
We now proceed in two cases.

23


---Page Break---
Case 1: P(Ui|Ti) > q.
In this case, because (Sgood
i
, Ti) satisfy (c, q, η)-robust expansion, as
discussed above, we know that P( e
N(Ui)|Sgood
i
) ≥cP(Ui|Ti), and thus

P( e
N(Ui) ∩Sgood
i
|Si) ≥c(1 −αi)P(Ui|Ti).

Fix x ∈e
N(Ui) ∩Sgood
i
. By the definition of e
N(Ui), there must be a point x′ ∈Ui reachable from
x by a good edge. That is, there is a point x′ ∈Ui such that f(x) = f(x′). Then the following set
of inequalities holds:

x∈Sgood
i
z
}|
{
˜y(x) = |
{z
}
Defn of Si, Ti

y(x) =

x′∈Mi
z
}|
{
y(x′) ̸= f(x′) = f(x).
|
{z
}

Defn of e
N

This shows f(x) ̸= ˜y(x), so x ∈Di. Hence e
N(Ui) ∩Sgood
i
⊂Di and we have:

err(f, ˜y|Si) = P(Di|Si) ≥P( e
N(Ui) ∩Sgood
i
|Si) ≥c(1 −αi)P(Ui|Ti),

so

err(f, y|Ti) ≤P(Rη(f)|Ti) + err(f, ˜y|Si)

c(1 −αi) .

Case 2: P(Ui|Ti) ≤q.
In this case we directly use the bound P(Ui|Ti) ≤q, so we get
err(g, y|Ti) ≤P(Rη(f)|Ti) + q. Combining the two cases yields the theorem.

B.3
Proposition 3.1

To prove Proposition 3.1, we reproduce (a tighter, intermediate version of) Theorem 3 from Fu et al.
[21] and translate it to our notation, then simplify the bound in the setting of the example from Section
3.

The goal of Fu et al. [21] is to estimate a set of graphical model parameters µ such that, given a vector
of labeling function outputs λ(x), the graphical model Pµ(y|λ(x)) approximates P(y|x). A classifier
is then trained on weak labels sampled from Pµ(y|λ(x)). Define err(f) = P(x,y)∼D[f(x) ̸= y] and
errµ(f) = E(x,y)∼D[P˜y∼Pµ(·|λ(x))[f(x) ̸= ˜y]].

Theorem B.4. Fu et al. [21, Theorem 3, intermediate step, p.39] Let f : X →Y be an arbitrary
classifier. Then:

err(f) ≤errµ(f) + E(x′,y′)∼D

"X

y
|P(y = y|x = x′) −Pµ(y = y|λ(x) = λ(x′))|

#

.
(5)

The remainder of the theorem accounts for finite-sample issues and estimation error for the optimal
graphical model parameters µ, but this is the fundamental relationship proven between a classifier’s
clean error and weak-label error. The second term in the RHS of (5) is exactly twice the total variation
distance between P(y|x) and Pµ(y|λ(x)). For the final theorem statement, Pinsker’s inequality is
used to upper bound this term using the KL divergence between the true probability P(y|x) and the
estimated graphical model probability Pµ(y|λ(x)), but the TV version is tighter, so we proceed by
analyzing the intermediate bound (5).

In the example from Section 3, there is only one labeling function:

λ(x) = ˜y(x) =






1
good ∈x
0
bad ∈x
∅
otherwise,

and we have assumed that the true label y is a deterministic function of x, i.e., y = y(x). We
can lower-bound (5) by using errµ(f) ≥0 and exactly computing the total-variation distance term
assuming the optimal graphical model parameters µ are known, in which case the joint distribution

24


---Page Break---
Pµ(y, λ(x)) exactly matches the true joint distribution P(y, λ(x)):

Pµ(y = 0, λ(x) = 0) = P(y = 0, λ(x) = 0) = P(y = 0)P(λ(x) = 0|y = 0) = 1

2(1 −α)

Pµ(y = 0, λ(x) = 1) = P(y = 0, λ(x) = 1) = P(y = 0)P(λ(x) = 1|y = 0) = 1

2α

Pµ(y = 1, λ(x) = 1) = P(y = 1, λ(x) = 1) = P(y = 1)P(λ(x) = 1|y = 1) = 1

2(1 −α)

Pµ(y = 1, λ(x) = 0) = P(y = 1, λ(x) = 0) = P(y = 1)P(λ(x) = 0|y = 1) = 1

2α

Pµ(y = 0, λ(x) = ∅) = P(y = 0, λ(x) = ∅) = P(y = 0)P(λ(x) = ∅|y = 0) = 1

2P(λ(x) = ∅)

Pµ(y = 1, λ(x) = ∅) = P(y = 1, λ(x) = ∅) = P(y = 1)P(λ(x) = ∅|y = 1) = 1

2P(λ(x) = ∅),

where we used the assumptions from Proposition 3.1 that P(˜y = 0|y = 1) = P(˜y = 1|y = 0) = α
(the error rates of the weak labels are equal for both classes) and P(λ(x) = ∅|y = y) = P(λ(x) = ∅)
(the weak labels cover each class equally often).

Now we can use this joint distribution to compute the value of the total variation distance term in (5).
There are six cases.

Case 1: x ∈Sgood
0
.
Here x is a true negative (x ∈S0) so y(x) = 0. Since x ∈Sgood
0
(i.e., the
pseudolabel agrees with the true label), λ(x) = 0. Here we have:

P(y = 1|x = x) = 0
P(y = 0|x = x) = 1

Pµ(y = 0|λ(x) = 0) =
Pµ(y = 0, λ(x) = 0)
Pµ(y = 0, λ(x) = 0) + Pµ(y = 1, λ(x) = 0) =

1
2(1 −α)

1
2(1 −α) + 1

2α = 1 −α

Pµ(y = 1|λ(x) = 0) =
Pµ(y = 1, λ(x) = 0)
Pµ(y = 0, λ(x) = 0) + Pµ(y = 1, λ(x) = 0) =

1
2α

1
2(1 −α) + 1

2α = α

And so the (doubled) total variation distance for this x is

|1 −Pµ(y = 0|λ(x) = 0)| + |0 −Pµ(y = 1|λ(x) = 0)| = |1 −(1 −α)| + |0 −α| = 2α.

Case 2: x ∈Sgood
1
.
Here x is a true positive (x ∈S1) so y(x) = 1. Since x ∈Sgood
1
(i.e., the
pseudolabel agrees with the true label), λ(x) = 1. By the analogous argument to Case 1, the doubled
TV distance term for this x is 2α.

Case 3: x ∈Sbad
0
.
Here x is a true negative (x ∈S0) so y(x) = 0. Since x ∈Sbad
0
(i.e., the
pseudolabel disagrees with the true label), λ(x) = 1. Here we have:

P(y = 1|x = x) = 0
P(y = 0|x = x) = 1

Pµ(y = 0|λ(x) = 1) =
Pµ(y = 0, λ(x) = 1)
Pµ(y = 0, λ(x) = 1) + Pµ(y = 1, λ(x) = 1) =

1
2α

1
2α + 1

2(1 −α) = α

Pµ(y = 1|λ(x) = 1) =
Pµ(y = 1, λ(x) = 1)
Pµ(y = 0, λ(x) = 1) + Pµ(y = 1, λ(x) = 1) = 1 −α

And so the (doubled) total variation distance for this x is

|1 −Pµ(y = 0|λ(x) = 1)| + |0 −Pµ(y = 1|λ(x) = 1)| = |1 −α| + |0 −(1 −α)| = 2(1 −α).

Case 4: x ∈Sbad
1
.
Here x is a true positive (x ∈S1) so y(x) = 1. Since x ∈Sbad
1
(i.e., the
pseudolabel disagrees with the true label), λ(x) = 0. By symmetry with Case 3, the (doubled) TV
distance for this x is 2(1 −α).

25


---Page Break---
Case 5: x ∈T0.
Here x is a true negative by definition of T0, so y(x) = 0. Thus P(y = 0|x =
x) = 1. Since x ∈T, the pseudolabeler abstains, i.e. λ(x) = ∅. The (doubled) total variation
distance for this x is thus:

|1 −Pµ(y = 0|λ(x) = ∅)| + |0 −Pµ(y = 1|λ(x) = ∅)| = |1 −1

2| + |0 −1

2| = 1

Case 6: x ∈T1.
By symmetry with Case 5, in this case the doubled TV term is 1.

Finishing the proof.
Finally, we can simplify (5). For any partition {Vi} of X,

E(x′,y′)∼D

"X

y
|P(y = y|x = x′) −Pµ(y = y|λ(x) = λ(x′))|

#

=
X

i
P(x ∈Vi)E(x′,y′)

"X

y
|P(y = y|x = x′) −Pµ(y = y|λ(x) = λ(x′))|

x′ ∈Vi

#

Applying this with our 6 cases above, we obtain:

E(x′,y′)∼D

"X

y
|P(y = y|x = x′) −Pµ(y = y|λ(x) = λ(x′))|

#

= P(Sgood
0
) · 2α

+ P(Sgood
1
) · 2α

+ P(Sbad
0
) · 2(1 −α)

+ P(Sbad
1
) · 2(1 −α)
+ P(T0)
+ P(T1).

Now we can group terms together using

P(Sgood
y
) = P(Sy)P(Sgood
y
|Sy) = P(Sy)(1 −α)

P(Sbad
y
) = P(Sy)P(Sbad
y
|Sy) = P(Sy)α

P(S0) + P(S1) = P(S)
P(T0) + P(T1) = P(T)

for y ∈{0, 1} to obtain:

E(x′,y′)∼D

"X

y
|P(y = y|x = x′) −Pµ(y = y|λ(x) = λ(x′))|

#

= P(S)4α(1 −α) + P(T).

Since we ignored the nonnegative errµ(f) term in (5), the following bound is tighter than Fu et al.
[21, Theorem 3]:
err(f) ≤P(S)4α(1 −α) + P(T).
However, this is looser than α on S (since α < 3/4), so it can never account for pseudolabel
correction, and charges every point in T as an error, so it can’t account for coverage expansion.

B.4
Proposition 3.2

Here we reproduce Theorem 4.3 from Wei et al. [69] in our notation and show how to apply their
bound (initially designed for the full-coverage P(S) = 1 pseudolabel correction setting) to the weak
supervision setup. The essential idea of the reduction is to treat points x ∈T, where ˜y(x) = ∅, as
mistakes from the teacher. Unfortunately, this effectively limits the application of the bound to cases
where P(S) is large.

Definitions.
For a classifier g, let Mi(g) = {x ∈Xi : g(x) ̸= y(x)} be the mistakes of g on Xi
(with respect to the true labels y(x)). In particular, Mi(g) = {x ∈Xi : g(x) ̸= i}. Recall that the
pseudolabeler is a classifier ˜y : X →Y ∪{∅}. In our notation, Mi(˜y) = Ti ∪Sbad
i
. Mi(˜y) contains
Ti because for all x ∈Ti, ˜y(x) = ∅̸= y(x).

26


---Page Break---
Definition (Expansion, Wei et al. [69]). P(·|Xi) satisfies (c, a)-expansion if for all U ⊂Mi(˜y) with
P(U|Xi) ≤a, P(N(U)|Xi) ≥min{cP(U|Xi), 1}.
Theorem (Wei et al. [69, Theorem 4.3], our notation). Let ¯a = maxi P(Mi(˜y)|Xi) and suppose
that ¯a < 1/3 and for all i, P(·|Xi) satisfies (c, ¯a)-expansion for c > 3. Suppose the classifier ˆg
minimizes:

L(g, ˜y) := c + 1

c −1err(g, ˜y) +
2c
c −1P[∃x′ ∈N(x) : g(x) ̸= g(x′)].

Then ˆg satisfies the following error bound:

err(ˆg, y) ≤
2
c −1err(˜y, y) +
2c
c −1P[∃x′ ∈N(x) : y(x) ̸= y(x′)].

This bound shows that when the expansion (according to their definition) is large and the ground-truth
classifier ˜y is adversarially robust over N on most points, a classifer ˆg that minimizes L(g, ˜y) enjoys a
good upper bound on it error. In particular, this bound can be lower than the error of the pseudolabels
err(˜y, y). However, the assumptions require that P(Mi(˜y)|Xi) < 1/3 for all i. Since Ti ⊂Mi(˜y),
this requires that P(Ti|Xi) < 1/3 for all i.

In addition to requiring high coverage, this application of Wei et al. [69, Theorem 4.3] also gives
one unified bound for the classifier error on the full input space X instead of dealing with coverage
expansion and pseudolabel correction effects separately. Empirically, in real weak supervision
settings, the two effects don’t always occur together, so ideally a theory for weak supervision would
treat the two effects separately. We take this approach with our bounds.

B.5
Note on finite-sample error bounds

Frei et al. [19] give a finite-sample analysis that shows self-training can give weak-to-strong general-
ization effects under more strict distributional assumptions than what we consider here. Wei et al.
[69] and Cai et al. [12] both provide finite-sample error bounds for the student model in addition
to using expansion to prove relationships between the weak label error and true label error on the
population distribution. They follow a modular analysis framework that is also followed (for example)
by Blum and Mitchell [6], Balcan et al. [3], Lang et al. [37]: first, the population error on the clean
labels is related via structural assumptions to the population error on the weak labels. Second, and
almost orthogonally, off-the-shelf concentration/finite-sample arguments are applied to the weak
label population error. This approach is modular in the sense that once the population error relation
is established, the finite-sample issues can be dealt with by purely considering err(f, ˜y|S), and the
remainder of the analysis is almost unrelated to each paper’s specific setup of self-training, co-training,
domain adaptation, or weak supervision. For example, Wei et al. [69], Cai et al. [12] both apply an
off-the-shelf generalization result for deep networks from Wei and Ma [68] to err(f, ˜y|S) in this step.
The key novelty in each paper is thus to establish reasonable structural assumptions that relate the
clean and weak errors on the population data, so this is also the focus of our results. Other work with
expansion assumptions, such as HaoChen et al. [23], which gives error bounds for linear classifiers
trained on top of contrastive representations, also focuses exclusively on the population case because
of the same argument.

C
Connections and comparisons to existing bounds

C.1
Co-training with conditionally independent views

Suppose that X = X1 × X2, so each example x = (x1, x2) has two “views”. Assume that the weak
labels ˜y can be written as a function of x1 alone, and the hypothesis class F is such that each f ∈F
is a function of x2 alone. Finally, suppose the distribution of x is such that for all A ⊂X1, B ⊂X2,
and i ∈Y,

P[x1 ∈A, x2 ∈B|y(x) = i] = P[x1 ∈A|y(x) = i]P[x2 ∈B|y(x) = i].

That is, the two views are conditionally independent given the (unobserved) true label.

This is precisely the conditionally independent view setup of Blum and Mitchell [6]. They prove the
following relationship between the clean error and weak label error of any classifier.

27


---Page Break---
Theorem C.1 (Blum and Mitchell [6], rephrased). Fix f : X2 →Y. Let αi = P(Sbad
i
|Si). In the
conditionally independent view setting, the errors of f satisfy:

err(f, y|Si) ≤err(f, ˜y|Si) −αi

(1 −2αi)
err(f, y|Ti) ≤err(f, ˜y|Si) −αi

(1 −2αi)

This exactly matches our pseudolabel correction result in Theorem B.1 and our coverage expansion
result from Theorem 4.2 when c = 1, q = 0, and P(R(f)|Ti) = 0. We now show how to set N such
that the conditionally independent view distribution provably expands with these parameters and
any classifier has P(R(f)|Ti) = 0. Our results thus directly generalize those of Blum and Mitchell
[6] in three directions, allowing for the views to be dependent, allowing small sets to not have good
expansion, and allowing for the case when the classifier is not perfectly robust over N.
Lemma C.1. Suppose the hypothesis class G is such that g : X2 →Y for all g ∈G. That
is, the hypotheses are functions of x2 alone. In the conditionally-independent view setup, the
appropriate families of sets M(·, G) and M′(·, G) satisfy (c, q)-expansion on the pairs of sets
(Sgood
i
, Ti), (Sbad
i
, Ti), (Sgood
i
, Sbad
i
), and (Sbad
i
, Sgood
i
), all with c = 1 and q = 0. Additionally,
any classifier g ∈G is adversarially robust at all x ∈X, so P(R(g)|A) = 0 for all g ∈G and all
A ⊂X. Plugging these coefficients into our Theorems B.1 and 4.2 yields Theorem C.1. Our bounds
therefore exactly recover the Blum and Mitchell [6]’s bounds in the conditionally-independent-view
setting.

Proof. Let N(x) = N(x1, x2) := {(x′, x2)} be the set of all points in X that share the same value
of x2 as x. That is, x′ ∈N(x) iff x′
2 = x2. By construction, any g : X2 →Y has P(R(g)) = 1:
all points in the same neighborhood have the same value of x2, so g must be constant on every
neighborhood. This proves r(f, x) = 0 for all x.

We will show c = 1 for the pair (Sgood
i
, Ti). Completely analogous arguments hold for the three
other pairs. The expansion amount c between Sgood
i
and Ti for set size q ≥0 is given by:

c =
min
U:P(U|Ti)>q
P(N(U)|Sgood
i
)
P(U|Ti)
= P(N(U), Sgood
i
|y = i)

P(Sgood
i
|y = i)

P(Ti|y = i)
P(U, Ti|y = i)

Fix an arbitrary g ∈G and let set U ⊂Ti be the set of x ∈Ti such that g(x) ̸= y(x). A point
x ∈Sgood
i
is in N(U) iff there exists x′ ∈U with x ∈N(x′). By our neighborhood construction,
this is equivalent to x2 = x′
2. Let U2 = {x2 : x ∈U} and consider a point x ∈Sgood
i
. Then
x ∈Sgood
i
satisfies x ∈N(U) iff x2 ∈U2. Likewise, a point x ∈Ti is in U iff x2 ∈U2, since the
errors of g (and thus, membership in U) only depend on x2. Now consider membership in Sgood
i
:
x ∈Sgood
i
iff ˜y(x1) = i. Conditioned on y = i, x ∈Ti iff ˜y(x1) = ∅. We can thus rewrite the
expression above as:

P(N(U), Sgood
i
|y = i)

P(Sgood
i
|y = i)

P(Ti|y = i)
P(U, Ti|y = i) = P(x2 ∈U2, ˜y(x1) = i|y = i)

P(˜y(x1) = i|y = i)
P(˜y(x1) = ∅|y = i)
P(x2 ∈U2, ˜y(x1) = ∅|y = i)

By the conditional independence assumption on the two views, the numerator of the first term and the
denominator of the second term factor and cancel out their denominator and numerator, respectively.
This yields:
P(x2 ∈U2|y = i)

1
1
P(x2 ∈U2|y = i) = 1.

Since U was arbitrary, this shows that in the conditionally-independent view setup has (c, q)-
expansion for the pair (Sgood
i
, Ti) with c = 1 and q = 0. The argument for the remaining three pairs
of sets is completely analogous.

C.2
Wei et al. [69]’s pseudolabel correction bound

Proposition 3.2 shows that Wei et al. [69]’s error bound can’t easily be ported to the full weak
supervision setting (i.e., pseudolabel correction and coverage expansion) because it requires high
coverage P(S). In this section, we show that even for pseudolabel correction, our bound is signifi-
cantly different from Wei et al.’s bound and has three desirable properties that Wei et al. [69, Theorem

28


---Page Break---
A.2] (the more general version of Wei et al. [69, Theorem 4.3]) lacks: (a) our bound more directly
generalizes the Blum and Mitchell [6] bounds in the conditionally-independent-view setting from
Section C.1, (b) our expansion parameter naturally appears in our bound, making it clear that more
expansion yields a tighter bound, and (c) our bound empirically applies to more classifiers since it
places less strict conditions on the student model.

First, we prove a direct analogue of Wei et al. [69, Theorem A.2] using our multiplicative expansion
assumptions, since the original uses a slightly different additive expansion assumption. The proofs
are almost identical.

Theorem C.2 (Wei et al. [69] Theorem A.2, restated). Suppose M(Sbad
i
, F) satisfies (c, q)-expansion
on (Sgood
i
, Sbad
i
) for some c > αi/(1−αi). Let ˜c = c 1−αi

αi
and suppose that the classifier f satisfies:

P(f(x) ̸= ˜y(x) or f not robust at x|Si) ≤αi(1 + q(˜c −1))

Then:
err(f, y|Si) ≤2

qαi + P(R(f)|Si)

+ err(f, ˜y|Si) −αi.

Note: The following result shows our robust expansion technique can also be used to directly
generalize this result to average-case-robust classifiers, whereas Wei et al. [69, Theorem A.2] only
applies to adversarially robust classifiers. Theorem C.2 is a special case of Theorem C.3 with η = 0.

Theorem C.3. Suppose Mη(Sbad
i
, F) satisfies (c, q, η)-expansion on (Sgood
i
, Sbad
i
) for some c >
αi/(1 −αi). Let ˜c = c 1−αi

αi
and additionally suppose:

P(f(x) ̸= ˜y(x) or f not η-robust at x|Si) ≤αi(1 + q(˜c −1))

Then:
err(f, y|Si) ≤2

qαi + P(Rη(f)|Si)

+ err(f, ˜y|Si) −αi.

Furthermore, if f satisfies:

Ex′∼D|Si,x∼D|N(x′)[f(x) ̸= f(x′)] ≤γ

for some γ ≥0, then P(Rη(f)|Si) ≤γ

η .

Proof. We directly prove Theorem C.3 since Theorem C.2 is a special case with η = 0. The proof is
largely similar to Wei et al. [69, Theorem A.2] but with our multiplicative version of the expansion
assumptions and our generalization to robust expansion. Let Mi = {x ∈Si|f(x) ̸= y(x)} be the
set of mistakes of f in Si; note that err(f, y|Si) = P(Mi|Si). Let Di = {x ∈Si|f(x) ̸= ˜y(x)} be
the set of disagreements between f and ˜y in Si; note that err(f, ˜y|Si) = P(Di|Si).

We can partition Mi into three sets:

A1 = (Si \ Di) ∩Sbad
i
A2 = Di ∩Mi ∩Sbad
i
A3 = Di ∩Sgood
i

Points x ∈A1 must be in Mi since f(x) = ˜y(x) ̸= y(x) by the definitions of Si \ Di and Sbad
i
,
respectively, so A1 = A1 ∩Mi. Points x ∈A3 must be in Mi since f(x) ̸= ˜y(x) = y(x) by the
definitions of Di and Sgood
i
, respectively, so A3 ⊂Mi ∩Sgood
i
. Finally, any point in Mi ∩Sgood
i
must
be in Di since otherwise we’d have f(x) = ˜y(x) = y(x) and thus x ̸∈Mi, so Mi ∩Sgood
i
⊂A3.
Together, this indicates that {A1, A2, A3} is a partition of Mi, since we’ve shown:

A1 ∪A2 ∪A3 =
(Si \ Di) ∩Mi ∩Sbad
i
∪Di ∩Mi ∩Sbad
i
∪Di ∩Mi ∩Sgood
i
= Mi ∩Sbad
i
∪Mi ∩Sgood
i
= Mi

29


---Page Break---
Now define Ui = Mi ∩Sbad
i
∩Rη(f). Note that A1 ∩Rη(f) and A2 ∩Rη(f) are disjoint subsets
of Ui. Since Ui ⊂Rη(f), Lemma B.2 implies that w( e
N(Ui), Ui) ≥(1 −η)w(N(Ui), Ui), so
P( e
N(Ui)|Sgood
i
) ≥P1−η(Sgood
i
, Ui). Since Ui is a mistake set intersected with a robust set, Ui ∈
M(Sbad
i
, F, y).

Suppose for the sake of contradiction that P(Ui|Sbad
i
) > q. By the expansion assumption, we have:

P( e
N(Ui) ∩Sgood
i
|Si) = (1 −αi)P( e
N(Ui)|Sgood
i
) ≥(1 −αi)P1−η(Sgood
i
, Ui)

> c(1 −αi)P(Ui|Sbad
i
) = c(1 −αi)

αi
P(Ui|Si).

Since we assumed c >
αi
1−αi , for some ˜c > 1 we have

P( e
N(Ui) ∩Sgood
i
|Si) > ˜c · P(Ui|Si).

Now we decompose Si into three disjoint sets (note the differences between Vj’s and Aj’s):

V1 = Si \ Di
V2 = Di ∩Sbad
i
V3 = Di ∩Sgood
i

Consider a point x ∈e
N(Ui) ∩Sgood
i
. By definition of e
N, there exists x′ ∈Ui reachable from x
by a good edge, so f(x) = f(x′). Since x and x′ are both in Si, y(x) = y(x′). Since Ui ⊂Mi,
f(x′) ̸= y(x′), so we must have f(x) ̸= y(x) = ˜y(x). Hence x ∈Di ∩Sgood
i
. This shows that
e
N(Ui) ∩Sgood
i
⊂V3. Using the decomposition above, we have:

1 = P(Si|Si) = P(V1|Si) + P(V2|Si) + P(V3|Si)

≥P(V1|Si) + P(V2|Si) + P( e
N(Ui) ∩Sgood
i
|Si)
> P(V1|Si) + P(V2|Si) + ˜cP(Ui|Si).
(6)

Lemma C.2.

αi ≤P(V2|Si) + P(Ui|Si) + P(Sbad
i
∩V1 ∩Rη(f)|Si)

Proof.

αi = P(Sbad
i
|Si) = P(Sbad
i
∩Di|Si) + P(Sbad
i
∩(Si \ Di)|Si)

= P(V2|Si) + P(Sbad
i
∩V1 ∩Rη(f)|Si) + P(Sbad
i
∩V1 ∩Rη(f)|Si).

Consider x ∈Sbad
i
∩V1 ∩Rη(f). Since x ∈Sbad
i
, ˜y(x) ̸= f(x). Since x ∈V1 = Si \ Di,
f(x) = ˜y(x). Thus f(x) ̸= y(x) and hence x ∈Mi. Because Ui = Mi ∩Sbad
i
∩Rη(f), this shows
that Sbad
i
∩V1 ∩Rη(f) ⊂Ui. The lemma follows.

Plugging Lemma C.2 into (6) yields:

1 > P(V1|Si) + αi + (˜c −1)P(Ui|Si) −P(V1 ∩Sbad
i
∩Rη(f)|Si)

= P(V1 ∩(Sbad
i
∩Rη(f))|Si) + αi + (˜c −1)P(Ui|Si)

= P(V1 ∩(Sgood
i
∪Rη(f))|Si) + αi + (˜c −1)P(Ui|Si)
≥P(V1 ∩Rη(f)|Si) + αi + (˜c −1)P(Ui|Si).
(7)

Now recall that V1 = Si \ Di, so:

P(V1 ∩Rη(f)|Si) = 1 −P(Di ∩Rη(f)|Si)
= 1 −P(f(x) ̸= ˜y(x) or f not η-robust at x|Si)

30


---Page Break---
We assumed in the theorem statement that:

P(f(x) ̸= ˜y(x) or f not η-robust at x|Si) ≤αi(1 + q(˜c −1))

≤αi + (˜c −1)αiP(Ui|Sbad
i
)
= αi + (˜c −1)P(Ui|Si).

And hence (7) gives 1 > 1, a contradiction. It must therefore be the case that P(Ui|Sbad
i
) < q. Since
A1 ∩Rη(f) and A2 ∩Rη(f) are disjoint subsets of Ui, we have that P(A1 ∩Rη(f)|Si) + P(A2 ∩
Rη(f)|Si) ≤qαi.

We can now finish the bound.

err(f, y|Si) = P(Mi|Si) = P(A1 ∪A2 ∪A3|Si)

= P((A1 ∪A2 ∪A3) ∩Rη(f)|Si) + P((A1 ∪A2 ∪A3) ∩Rη(f)|Si)

≤P((A1 ∪A2 ∪A3) ∩Rη(f)|Si) + P(Rη(f)|Si)

≤qαi + P(A3 ∩Rη(f)|Si) + P(Rη(f)|Si).

Since (A3 ∪(Si \ Di)) ∩Rη(f) = (A1 ∪Sgood
i
) ∩Rη(f), we have:

(A3 ∩Rη(f)) ∪(Si \ Di) ∩Rη(f) = A1 ∩Rη(f) ∪Sgood
i
∩Rη(f).

All sets that are arguments to the unions above are disjoint, so:

P(A3 ∩Rη(f)|Si) + P(Di ∩Rη(f)|Si) = P(A1 ∩Rη(f)|Si) + P(Sgood
i
∩Rη(f)|Si)

Using P(Di ∩Rη(f)|Si) = 1 −P(Di ∪Rη(f)|Si), P(Sgood
i
∩Rη(f)) ≤1 −αi, and P(A1 ∩
Rη(f)|Si) ≤qαi, we obtain:

P(A3 ∩Rη(f)|Si) ≤P(Di ∪Rη(f)|Si) + qαi −αi.

Plugging this in to the earlier bound gives:

err(g, y|Si) ≤P(Rη(f)|Si) + 2qαi + P(Di ∪Rη(f)|Si) −αi,

which we can leave as-is:

err(f, y|Si) ≤P(Rη(f)|Si) + 2qαi + P(f(x) ̸= y(x) or f not η-robust at x|Si) −αi,

or union-bound to obtain:

err(f, y|Si) ≤2

qαi + P(Rη(f)|Si)

+ err(f, ˜y|Si) −αi.

The second part of the theorem follows directly from Lemma B.1.

Comparing Theorems 4.1 and C.2 in the Co-Training setting.
In Section C.1 we argued that the
conditionally-independent-view setup of Blum and Mitchell [6] satisfies our expansion assumptions
with c = 1 and q = 0, and any model is inherently perfectly robust because it’s only trained on one
of the views, so P(R(f)|Si) = 0 for all f ∈F. We can plug these values into the conditions of
Theorem C.2. In particular, Theorem C.2 only applies to classifiers f that satisfy:

err(f, ˜y|Si) ≤αi (1 + q(˜c −1)) = αi,

But in the conditionally independent view setting, minf∈F err(f, ˜y|Si) = αi. In this case, the error
bound in Theorem C.2 simplifies to:

err(f, y|Si) ≤err(f, ˜y|Si) −αi = 0

Thus, in this setting, Theorem C.2 only applies to classifiers that minimize the weak error as much as
possible, and therefore have 0 error on the true labels. In contrast, our Theorem B.1 applies to any f
with err(f, ˜y|Si) ≤1−q−αi = 1−αi. When c = 1, the value of c′ in Theorem B.1 reduces to c′ = 1,
and our bound exactly recovers the Blum and Mitchell bound for any classifier with error at most
1−αi. This relaxation of which classifiers qualify for the bound would be important for finite-sample

31


---Page Break---
Table 2: Comparison of measured values of the amount of expansion c (“exp. c val.”) on the data
described in Section 6. Theorem B.1 (our pseudolabel correction result) requires M′(Sgood
i
, F) to
expand on the pair (Sbad
i
, Sgood
i
) for some c > 0. That is, it requires robust non-mistakes to expand to
points with the wrong pseudolabels. We call this “good-to-bad” or G2B expansion. On the other hand,
Theorem C.2 requires M(Sbad
i
, F) to expand on the pair (Sgood
i
, Sbad
i
) for some c > αi/(1 −αi).
In other words, it requires robust mistakes to expand to points with the correct pseudolabel. We call
this “bad-to-good” or B2G expansion. These results show that empirically, we may have the G2B
c > 0, so our bounds apply, but the B2G c < αi/(1 −αi), so Theorem C.2 does not apply. This is
the case for the i = 1 values.

i
αi
αi/(1 −αi)
(Sbad
i
, Sgood
i
)
(G2B) exp. c val.
(Sgood
i
, Sbad
i
)
(B2G) exp. c val.

0
0.11
0.12
0.85
0.17
1
0.33
0.49
0.50
0.32

guarantees, since Theorem C.2 only applies to f for which err(f, ˜y|Si) −minf ′∈F err(f ′, ˜y|Si) = 0.
It is not possible to guarantee that we can obtain such an f from a finite sample using standard
improper PAC learning techniques. In contrast, our bound still applies to student models f with
err(f, ˜y|Si) −minf ′∈F err(f ′, ˜y|Si) > 0.

Our result can also recover the same bound as Theorem C.2 in this setting: when c = 1, we can take
∆= 1 in Corollary B.1. The condition on the classifier then becomes err(f, ˜y|Si) ≤αi, and in that
case the bound in Corollary B.1 simplifies to

err(f, y|Si) ≤err(f, ˜y|Si) + αi(1 −2c∆)
= err(f, ˜y|Si) −αi = 0

so Theorem B.1 can exactly match Theorem C.2 in this case. This discussion shows that our result
can match the bound from Theorem C.2 in this setting, but is more stable, since it applies to classifiers
that do not attain the exact optimal weak label error on the population.

Non-Robust coefficient.
The bound in Theorem B.1 has a coefficient of
2αi
1−2αi on the probability

of non-robust points P(R(f)|Si). The bound in Theorem C.3 has a coefficient of 2. Our dependence
on the probability of non-robust points is better when α < 1/3.

Expansion term not in the bound.
Theorem C.2 does not have the amount of expansion c in the
error bound—instead, the expansion is only present in the pre-conditions that the classifier needs
to meet for the bound to apply. In Wei et al. [69], the authors avoid this by introducing a loss
function that contains the expansion c and assuming the classifier f minimizes that loss, essentially
re-introducing a dependence on c. However, achieving this bound (Wei et al. [69, Theorem 4.3])
therefore requires knowing the expansion up-front in order to minimize the suitably-scaled loss
function. In contrast, our pseudolabel correction bound directly has the amount of expansion c on the
right-hand-side.

Empirically less sensitive.
Theorem C.2 requires c > αi/(1 −αi) for it to apply. This is not an
artifact of our conversion from Wei et al. [69]’s additive expansion to our multiplicative expansion, as
shown by the following lemma.
Lemma C.3. Suppose that the distribution P(·|Si) satisfies (q, δ)-additive expansion on Sbad
i
[69,
Definition A.1] for some δ > 0. Then P(·|Si) satisfies (c, q

αi )-expansion on (Sgood
i
, Sbad
i
) with
c > αi/(1 −αi).

Proof. (q, δ)-additive expansion implies that for any U ⊂Sbad
i
with P(U|Si) > q,

P(N(U) \ Sbad
i
|Si) ≥P(U|Si) + δ

P(N(U) ∩Sgood
i
|Si) ≥P(U ∩Sbad
i
|Si) + δ

P(N(U)|Sgood
i
)(1 −αi) ≥P(U|Sbad
i
)αi + δ,

32


---Page Break---
so P(N(U)|Sgood
i
) >
αi
1−αi P(U|Sbad
i
). We assumed U ⊂Sbad
i
was an arbitrary set with P(U|Si) =
P(U|Sbad
i
)αi > q, so P(U|Sbad
i
) > q/αi. Hence this example satisfies (c, q/αi)-expansion on the
pair of sets (Sgood
i
, Sbad
i
).

Lemma C.3 shows that if Wei et al. [69]’s additive expansion assumptions hold, our (Sgood
i
, Sbad
i
)
expansion assumption holds with c > αi/(1 −αi). However, our experiments suggest that this
amount of expansion may not be present empirically for the error sets of actual trained classifiers. In
contrast, our pseudolabel correction result only requires c > 0 and the bounds computed using our
empirical measurements are close to the true error of the classifiers. Table 2 shows an example of this
on the data from Section 6. For label 1, αi/(1 −αi) = 0.49 but the measured value of the expansion
between (Sgood
i
, Sbad
i
) is only 0.32, so Theorem C.2 does not apply. Of course, this does not rule out
the possibility that C.2 may apply with a different choice of the neighborhood N, but for at least one
natural setup (N the set of paraphrases of the text input x), there is not enough measured expansion
for it to apply.

D
Checking expansion: Statistical theory

Theorem D.1 (Theorem 5.1, formal). For U ∈M, define:

c(U) := P(n(x) ∈U|x ∈A)

P(x ∈U|x ∈B)

and its empirical version:

ˆc(U) :=

1
nA
PnA
i=1 1[n(xi) ∈U]

1
m
Pm
j=1 1[xi ∈U]

Suppose there is a lower bound ¯q > 0 such that q(U) ≥¯q for all U ∈M. Define γ = nB ¯q/nA and
let m = nA + nB. Then for any δ ∈(0, 1], with probability at least 1 −δ over the sampling of SA
and SB,

sup
U∈M
ˆc(U) −c(U) ≤4(4 + √γ)

s

VC(M) log
2em
VC(M) + log 8

δ
nA¯q2
.

Proof of Theorem D.1. We start off with a lemma providing concentation results for a fixed set
U ∈M.

Lemma D.1 (Concentration for a fixed U). Fix an arbitrary set U ∈M. Define:

p(U) := P(n(x) ∈U|x ∈A)
q(U) := P(x ∈U|x ∈B)

And their empirical versions:

ˆp(U) := 1

nA

X

xi∈SA
1[n(xi) ∈U]
ˆq(U) := 1

nB

X

xi∈SB
1[xi ∈U].

Suppose there is a lower bound ¯q > 0 such that q(U) ≥¯q for all U ∈M. Define γ = nB ¯q/nA.
Then for any δ ∈(0, 1],

P
 ˆp(U)

ˆq(U) −p(U)

q(U) > δ

≤2 exp

 

−2nAδ2 ·
γ¯q2
 
4 + √γ
2

!

,

where the probability is with respect to the sampling of SA and SB.

Proof. Fix ϵ > 0 and let E be the event that (1 −ϵ)q(U) ≤ˆq(U). Let W refer to the event that
ˆp(U)
ˆq(U) −p(U)

q(U) > δ. Then

P[W] = P[W|E]P[E] + P[W|E]P[E]

≤P[W|E] + P[E].

33


---Page Break---
By the multiplicative Chernoff bound,

P[E] = P[ˆq(U) < (1 −ϵ)q(U)] ≤exp(−ϵ2nB · q(U)/2).

Now the goal is to bound P[W|E].

P[W|E] = P
 ˆp(U)

ˆq(U) −p(U)

q(U) > δ
E

≤P

ˆp(U)
(1 −ϵ)q(U) −p(U)

q(U) > δ
E


= P

ˆp(U)
(1 −ϵ)q(U) −p(U)

q(U) > δ


= P [ˆp > p + qδ −ϵ(p + qδ)]
≤P[ˆp > p + ¯qδ −2ϵ],

where we used in the second line that the samples from A and B are independent (and thus, ˆp(U) and
ˆq(U) are independent), and in the last line that q ≥¯q, 0 ≤p ≤1, 0 ≤q ≤1, and δ ∈(0, 1]. Setting
ϵ′ = ¯qδ −2ϵ, and using the Chernoff-Hoeffding theorem, we obtain:

P [W|E] ≤P[ˆp > E[ˆp] + ϵ′] ≤exp(−2(ϵ′)2 · nA)

Combining results, we have that for any ϵ > 0:

P[W] ≤exp(−2(ϵ′)2 · nA) + exp(−ϵ2nB · ¯q/2).

Let γ = nB ¯q/nA. Setting ϵ =
2¯qδ
4+√γ (this is valid since ϵ > 0) yields ϵ′ =
¯qδ√γ
4+√γ , so we obtain

P[W] ≤2 exp

 

−2nAδ2 ·
γ¯q2
 
4 + √γ
2

!

.

Corollary D.1. An almost exactly symmetric argument yields:

P
p(U)

q(U) −ˆp(U)

ˆq(U) > δ

≤2 exp

 

−nAδ2 ·
γ¯q2
 
2 + √γ
2

!

,

and thus:

P

p(U)
q(U) −ˆp(U)

ˆq(U) > δ



≤4 exp

 

−nAδ2 ·
γ¯q2
 
4 + √γ
2

!

Now we show how to apply this deviation bound for ˆp(U)/ˆq(U) in place of Hoeffding’s inequality in
the symmetrization argument from Bousquet et al. [8].

Lemma D.2 (Symmetrization). Suppose we have a ghost sample of nA additional points x′
i drawn
i.i.d. from P(·|A) and nB additional points z′
i drawn i.i.d. from P(·|B). Let ˆc′(U) = ˆp′(U)/ˆq′(U)
denote the empirical expansion of set U on the ghost sample, and let ˆc(U) = ˆp(U)/ˆq(U) denote
the empirical expansion of set U on the original sample. Then for any t > 0 such that nA · t2 ≥
(4+√γ)2 log 16

γ¯q2
:

P

sup
U∈M
ˆc(U) −c(U) > t

≤2P

sup
U∈M
ˆc(U) −ˆc′(U) > t/2

.

Proof of Lemma D.2. This follows Bousquet et al. [8] exactly, except we replace the application of
one inequality with the deviation bound derived above. Let Un be the set achieving the supremum on
the left-hand-side. This depends on the sample (x1, . . . , xnA+nB).

1ˆc(Un)−c(Un)>t1ˆc′(Un)−c(Un)<t/2 = 1ˆc(Un)−c(Un)>t∧c(Un)−ˆc′(Un)≥−t/2
≤1ˆc(Un)−ˆc′(Un)>t/2

Taking the expectation over the ghost sample (x′
1, . . . , x′
nA+nB),

1ˆc(Un)−c(Un)>tP′[ˆc′(Un) −c(Un) < t/2] ≤P′[ˆc(Un) −ˆc′(Un) > t/2]

34


---Page Break---
From Lemma D.1,

P ′[ˆc(Un) −c(Un) ≥t/2] ≤2 exp

 

−nAt2

2
·
γ¯q2
 
4 + √γ
2

!

≤1

2
by the condition on nA · t2. Hence:
1ˆc(Un)−c(Un)>t ≤2P′[ˆc(hn) −ˆc′(hn) > t/2],
and taking the expectation over the original sample (x1, . . . , xnA+nB) finishes the proof.

Recall that the (deterministic) neighborhood oracle n is fixed ahead of training and thus does not
depend on the random sample(s). For xi ∈SA, and x′
i ∈S′
A, we can redefine xi ←n(xi)—that is,
we apply the neighborhood oracle as part of the sampling process for points in A. Let m = nA + nB
and define Mx1,...,xm = {(1x1∈U, . . . 1xn∈U) : U ∈M}. Recall that the growth function of
class M is defined as SM(m) = sup(x1,...,xm) |Mx1,...,xm|. Now to finish the proof, observe that
the sup in the right-hand-side of the Lemma D.2 result only depends on the finite set of vectors
Mx1,...,xm,x′
1,...,x′
m, due to our simplification xi ←n(xi) for xi ∈SA, S′
A. That is,

P

sup
U∈M
ˆc(U) −c(U) > t

≤2P

"

sup
U∈Mx1,...,xm,x′
1,...,x′m
ˆc(U) −ˆc′(U) > t/2

#

≤
X

U∈Mx1,...,xm,x′
1,...,x′m

P[ˆc(U) −ˆc′(U) > t/2]

≤2
X

U∈Mx1,...,xm,x′
1,...,x′m

P[|ˆc(U) −c(U)| > t/4]

≤8SH(2m) exp

−nA · t2
γ¯q2

16(4 + √γ)2


.

The first line simply applies the union bound over Mx1,...,xm,x′
1,...,x′m. The second line uses the fact
that if a −b > c then for any real x, either |a −x| or |b −x| or both are larger than c/2, applies a
union bound, and uses that ˆc(U) and ˆc′(U) are identically distributed. The last line applies Corollary
D.1 and uses the definition of the growth function. The Sauer-Shelah lemma [61, 56, 57] implies that
for any class H with VC(H) = d, SH(m) ≤
  em

d
d. Then setting:

t ≥4(4 + √γ)

s

VC(M) log
2em
VC(M) + log 8

δ
nA¯q2

guarantees that

P

sup
U∈M
ˆc(U) −c(U) > t

≤δ

and completes the proof.

The following proposition shows the complexity of the class of mistake sets generated by f ∈F is
bounded by complexity of the hypothesis class F itself.
Proposition D.1. Suppose F is a binary hypothesis class and y : X →{0, 1} is the ground-truth
classifier (we do not assume y ∈F). Fix a set A ⊂X and for each f ∈F let U(A, f, y) = {x ∈A :
f(x) ̸= y(x)} be f’s mistakes on A. Define M = {U(A, f, y) : f ∈F}. Then VC(M) ≤VC(F).

Proof. Fix a finite set of points V ⊂A with |V | = d, so V = (x1, . . . , xd). Suppose that M shatters
V , so:
|{U ∩V : U ∈M}| = 2d

Fix an arbitrary labeling y′ : X →{0, 1}. Let W = {xi ∈V : y′(xi) ̸= y(xi)}. Since W ∈M,
by definition of M there exists f ∈F such that f(xi) ̸= y(xi) if and only if xi ∈W. If xi ∈W,
then f(xi) ̸= y(xi) and y′(xi) ̸= y(xi); since there are only two labels, it must be the case that
f(xi) = y′(xi). If xi ̸∈W, then f(xi) = y(xi) = y′(xi). This shows that for any labeling y′ of
the points in V , there exists f ∈F that has zero error on that labeling. Hence any set shattered by
M is shattered by F, so VC(M) ≤VC(F).

35


---Page Break---
E
Experiment Details

Dataset details. We train on the IMDb dataset of movie reviews [41] (HuggingFaceHub ID
stanfordnlp/imdb), which has 25000 training examples and 25000 test examples, each with
exactly 50/50 positive/negative split, and an unspecified license. For the teacher model described
in Section 6, based on the presence of unigrams ‘horrible’ and ‘incredible’, the coverage rate is
P(x ∈S) = 0.06. The fully-supervised (gold) error of a linear classifier trained on top of the
SentenceBERT representation we use is 10.6% (when trained using the optimization procedure and
hyperparameters described below; we did not perform hyperparameter tuning, since this number is
merely used as an ideal lower bound for the performance of a weakly-supervised classifier). For
i ∈{0, 1} and points x ∈Sgood
i
, we obtained neighbors in Ti and Sbad
i
using the oracle procedure
described in Section 6 and shown graphically in Figure 3. In each neighbor sampling step, we
enforce the constraint y(x′) = i using the model trained on the gold labels to perform rejection
sampling. This ensures (as much as possible) that all neighbors have the same ground-truth label, as
required by the definitions. We include these sampled neighbors in the supplementary material for
reproducability.

Model training. We train the linear classifiers using the AdamW optimizer [40] with global learning
rate 0.01 and a weight decay of 0.1, and linear learning rate decay over 500 optimizer steps. For
the heuristic in Section 5, we retrain a model 5 times on 5 different random subsets of the covered
training samples, each 80% of the original, and use the other 20% of covered samples as a validation
set to perform early stopping with the weak label. We do not perform early stopping with the true
gold label, which is common practice in many works on programmatic weak supervision and gives a
large performance gain [73, 10]. Since the goal of our experiments is to give evidence that expansion
holds in practice, we do not tune the learning rate or weight decay parameters at all, since the initial
values we chose already led to weak-to-strong generalization effects.

Compute. We used an internal machine with 4xA100 80GB GPUs to extract all deep network
embeddings and to train the linear classifiers on top of those embeddings. Embeddings were only
extracted once. Classifiers were trained 5 times with different random seeds and data subsets to
perform the heuristic procedure in Section 5. This step was repeated several times as we developed
different versions of our bounds and recomputed different notions of expansion w.r.t. the mistake sets
of the trained classifiers.

Results. Table 3 shows the results of training using the SentenceBERT representations on the
pseudolabels described in Section 6. The table also shows the accuracy of the weak labels themselves
in the ˜y(x) row. Pseudolabel correction and coverage expansion occur in different amounts depending
on the class. For example, the student consistently improves over ˜y on S0 but not on S1. This justifies
our choice to analyze these effects separately for different classes. As discussed in Section 6, Tables
1 and 4 show the measured values of expansion, the worst-case weak label error of the student across
the 5 training runs, and the values of our bounds from Theorems B.1 and B.2, respectively. Since the
goal is to give evidence for the presence of expansion, we ignore P(Rη(f)|·) when computing the
values of the bounds and focus on the terms involving amount of expansion. The bounds in Table 1
show that our results can effectively distinguish between cases where pseudolabel correction does
(on S0) and does not (on S1) occur.

Table 4 measures the expansion of the set families M(Ti, F) and M′(Ti, F) on the set pairs
(Sgood
i
, Ti) and (Sbad
i
, Ti), respectively. First, the fact that both expansion numbers are nonzero gives
evidence for the extra structure described in Section 4.1 and assumed in Theorem 4.2/Theorem B.2.
As mentioned in Section 4.1, we could have also just assumed expansion on (Sgood
i
, Ti), but this extra
structure gives a tighter bound, and these results suggest that it is present empirically. Table 4 shows
that even though the coverage P(x ∈S) = 0.06 is very low in this example, good representations
(in this case, representations such that M(·, G ◦φ) expands for G linear), can have good coverage
expansion. We use the tighter bound from Theorem B.2, which allows for different amounts of
expansion between (Sgood
i
, Ti) and (Sbad
i
, Ti).

36


---Page Break---
Table 3: Test accuracy breakdown for linear probe trained with true (gold) and weak labels on the
IMDb data. Performance of the weakly-trained model is broken down across the covered sets (S0,
S1) and the uncovered sets (T0, T1). One standard deviation across five training folds is shown in
parentheses. Pseudolabel correction and coverage expansion occur in different amounts depending on
the class. For example, the student consistently improves over ˜y on S0 but not on S1. This justifies
our choice to analyze these effects separately for different classes.

Model
Test Acc.
(gold train)
Test Acc.
(weak train)
S0 (4%)
S1 (2%)
T0 (46%)
T1 (48%)

˜y(x)
89.5
67.1
n/a
n/a

BoW
84.9
50.1
89.5
67.1
100.0
0.0
SentenceBERT
89.4
81.8 (1.4)
95.5 (1.4)
69.1 (4.6)
85.8 (3.1)
74.9 (4.8)

Table 4: Measured expansion values and error bounds for the uncovered sets Ti. Expansion values
for the families M(Sbad
i
, F) on (Sgood
i
, Sbad
i
) and M′(Sgood
i
, F) on (Sbad
i
, Sgood
i
), are measured
using the heuristic described in Section 5. The detection of both types of expansion (expansion
from Ti to Sgood
i
and to Sbad
i
) gives evidence for the extra structure we described in Section 4.1
and justifies our use of Theorem B.2, which uses this structure, instead of Theorem B.3, which only
uses expansion from Ti to Sgood
i
and gives a looser bound. Worst-case value of the error bound in
Theorem B.2, computed using the smallest expansion values and largest weak errors err(f, ˜y|Si)
from the 5 training runs. The err(f, ˜y|Si) and αi values used in the bound computation are identical
to the values in Table 1. Unlike in Table 1, where the “baseline” for pseudolabel correction effects is
to have error bounds strictly better than αi, for coverage expansion, the more relevant comparison is
against random/arbitrary guessing. The actual worst-case error of the student on each Ti is shown as
err(f, ˜y|Ti). As suggested by our bound values, the errors on each Ti are non-trivial (much better
than random or arbitrary guessing).

Model
i
(Sgood
i
, Ti)
(Sbad
i
, Ti)
err(f, ˜y|Si)
Bd. val
err(f, y|Ti)

SentenceBERT
0
0.16
0.98
0.12
0.37
0.16
1
0.75
0.55
0.29
0.33
0.29

I did not enjoy this movie at all. 
Watching it was an incredible burden.

Paraphrase this text. Don't 
change the meaning and don't 
include the words "horrible" or 
"incredible" in your paraphrase.

I did not enjoy this movie at all. 

Watching it was a burden.

I did not enjoy this movie at all. 
Watching it was a horrible burden.

Paraphrase this text. Don't change 

the meaning, but include the word 

"horrible" in your paraphrase.

rejection

sample

rejection

sample

Figure 3: Example of our neighborhood oracle n, constructed using a targeted paraphrase procedure.
For a covered point x ∈S (in this case, x ∈Sbad
0
, since it is a true negative point mislabeled by our
example weak rules ˜y), we first generate an uncovered point x′ ∈Ti using a constrained paraphrase
model and rejection sampling to ensure the ground-truth label remains negative (we use a model
trained on the gold labels as a stand-in for the ground truth y). Next, we use GPT-4 to rewrite x′
using the opposite word from {horrible, incredible} than the one that originally appeared. This
generates another point x′′ ∈Si. Since we enforce that x and x′′ are covered by different words, we
know that if x ∈Sgood
i
(resp. Sbad
i
), x′′ ∈Sbad
i
(resp. Sgood
i
).

37


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract claims that we provide error bounds for weakly-supervised
classifiers that allow for both pseudolabel correction and coverage expansion effects. We
provide these bounds in Section 4. We also claim in the abstract and show in Section 3
that existing weak supervision theory fails to account for these effects. We claim in the
abstract and show in Sections 5 and 6 that our expansion assumptions (plausibly) hold on a
real-world example of weak-to-strong generalization effects.

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

Justification: we discuss the limitations of our analysis in Section 7.

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

38


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: All our assumptions are clearly stated in the theorem statements. Section
4 contains intuitive proof sketches for our main theorems. The informal Theorem 4.3 is
formally stated and proven in Appendix B as Theorems B.1 and B.2.

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

Justification: we include our code in the supplemental material and fully describe our
empirical setup in Section 6 and Appendix E.

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

39


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
Answer: [Yes]
Justification: The code for reproducing our experimental results is included in the supple-
mental material. The data we use is openly accessible on the HuggingFaceHub.
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
Justification: we discuss these details in Appendix E.
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
Justification: many of the quantities in our experiments (such as expansion and the computed
values of our bounds) are worst-case rather than average quantities. However, we report
standard deviations of student model performance in Appendix E.
Guidelines:

40


---Page Break---
• The answer NA means that the paper does not include experiments.
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

Answer: [Yes]

Justification: we discuss these details in Appendix E.

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

Justification: we have read and followed the NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

41


---Page Break---
Justification: we discuss the societal importance of understanding weak-to-strong generaliza-
tion in Section 1 in context of the increasing prevalence of synthetic data and the increasing
use of LLMs in crowdwork [64]. We aim to give a straightforward and intuitive description
of when weak-to-strong generalization effects can occur to make these effects less opaque.
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
Justification: the paper does not release new data or models.
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
Justification: we cite the original creators of the data we use, and the license of that data, in
Appendix E.
Guidelines:

• The answer NA means that the paper does not use existing assets.

42


---Page Break---
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
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: the paper does not release new assets.

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

Justification: the paper does not involve crowdsourcing nor research with human subjects.

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

Justification: the paper does not involve crowdsourcing nor research with human subjects.

43


---Page Break---
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

44


---Page Break---
