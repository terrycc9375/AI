The Sample Complexity of Gradient Descent
in Stochastic Convex Optimization

Roi Livni
School of Electrical Engineering
Tel Aviv University
rlivni@tauex.tau.ac.il

Abstract

We analyze the sample complexity of full-batch Gradient Descent (GD) in the setup
of non-smooth Stochastic Convex Optimization. We show that the generalization
error of GD, with common choice of hyper-parameters, can be ˜Θ(𝑑/𝑚+ 1/√𝑚),
where 𝑑is the dimension and 𝑚is the sample size. This matches the sample
complexity of worst-case empirical risk minimizers. That means that, in contrast
with other algorithms, GD has no advantage over naive ERMs. Our bound follows
from a new generalization bound that depends on both the dimension as well as
the learning rate and number of iterations. Our bound also shows that, for general
hyper-parameters, when the dimension is strictly larger than number of samples,
𝑇= Ω(1/ε4) iterations are necessary to avoid overfitting. This resolves an open
problem by Amir, Koren, and Livni [3], Schliserman, Sherman, and Koren [20],
and improves over previous lower bounds that demonstrated that the sample size
must be at least square root of the dimension.

1
Introduction

Stochastic Convex Optimization (SCO) is a theoretical model that depicts a learner that minimizes
a (Lipschitz) convex function, given finite noisy observations of the objective [22]. While often
considered simplistic, in recent years SCO has become a focus of theoretical research, partly, because
of its importance to the study of first-order optimization methods. But, also, it has become focus
of study because it is one of few theoretical settings that exhibit overparameterized learning . In
more detail, classical learning theory often focuses on the tension between number of samples, or
training data, and the complexity of the model to be learnt. A common wisdom of classical theories
[1, 7, 14, 24] is that, to avoid overfitting, the complexity of a model should be adjusted in proportion
to the amount of training data. However, recent advances in Machine Learning have challenged
this viewpoint. Evidently [18, 25], state-of-the-art algorithms generalize well but without, explicitly,
controlling the capacity of the model to be learnt. In turn, today, it is one of the most emerging
challenges, for learning theory, to understand learnability when the number of parameters in a learnt
model exceeds the number of examples, and when, seemingly, nothing withholds the algorithm from
overfitting.

Towards this, we look at SCO. In SCO, Shalev-Shwartz, Shamir, Srebro, and Sridharan [22] showed
how algorithms can overfit with dimension dependent sample size. But, at the same time, it was
known [8, 26] that there are algorithms that provably avoid overfitting with far fewer examples than
dimensions. As such, SCO became a canonical model to study how a well-designed algorithm can
avoid overfitting even when the number of examples is too small to guarantee generalization by an
algorithmic-independent argument [2–5, 11, 12, 16, 20–22]. A step towards understanding what
induces generalization is to identify which algorithms generalize. Then, we can ask what yields
the separation. Surprisingly, for many well-studied algorithms this question is not always answered.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Perhaps the simplest algorithm, whose sample complexity is not yet understood, is Gradient Descent
(GD). And we turn to the basic question of the sample complexity of gradient descent.

While this question remained open, there have been several advancements and intermediate answers:
The first, dimension independent, generalization bound was given by Bassily, Feldman, Guzmán,
and Talwar [6] that provided stability bounds [8]. The result of Bassily et al. demonstrated that,
GD can have dimension-independent sample complexity rate. However, to achieve that, one has
to use non-standard choice of hyperparameters which affects the efficiency of the algorithm. In
particular, the number of rounds becomes quadratic in the size of the sample (as opposed to linear,
with standard choice). On the other hand, a classical covering argument shows that linear dependence
in the dimension is the worst possible, for any empirical risk minimizer, irrespective of properties
such as stability.

In terms of lower bounds, Amir, Koren, and Livni [3] were the first to show that GD may have a
dimension dependence in the sample complexity. They showed that, with natural hyperparameters, the
algorithm must observe number of samples that is at least logarithmic in the dimension. This result
was recently improved by Schliserman, Sherman, and Koren [20] that showed that at least square
root of the dimension is required. Taken together, so far it was shown that either the algorithm’s
hyperparameters are tuned to achieve stability, at a cost in running time, or the algorithm must suffer
some dimension dependence, linear at worst square root at best.

Here, we close the gap and show that linear dependence is necessary. Informally, we provide the
following generalization error bound, in terms of dimension 𝑑, sample size, 𝑚, and hyperparameters
of the algorithm, η and 𝑇(the learning rate and number of iterations). We show that when 𝑇is at
most cubic in the dimension (see Theorem 1 for a formal statement):

Generalization gap of GD = Ω

min
 𝑑

𝑚, 1

· min
n
η
√

𝑇, 1
o
.

The first factor in the RHS describes the linear dependence of the generalization error in the dimension,
and corresponds to the optimal sample complexity of empirical risk minimizers, as demonstrated by
Carmon, Livni, and Yehudayoff [11]. The second term lower bounds the stability of the algorithm
[6], and played a similar role in previous bounds [3, 20]. Each factor is optimal at a certain regime,
and cannot be improved. Most importantly, for a standard choice of η = 𝑂(1/
√

𝑇), the first term is
dominant, and the aformentioned lower bound is complemented with the upper bound of Carmon et al.
[11]. Our result implies, then, a sample complexity of ˜Θ(𝑑/𝑚+ 1/√𝑚). When 𝑑≥𝑚, the second
factor is dominant. When running time is at most quadratic in number of examples, this term also
governs the stability of the algorithm, hence the result of Bassily et al. [6] provides a complementary
upper bound (see further discussion in Section 3.1).

2
Background

We consider the standard setup of Stochastic Convex Optimization (SCO) as in [22]. Set W = {𝑤:
∥𝑤∥≤1}, and let Z be an arbitrary, finite domain (our main result is a lower bound, hence finiteness
of Z is without loss of generality). We assume that there exists a function 𝑓(𝑤, 𝑧) that is convex and
𝐿-Lipschitz in 𝑤∈W for every choice of 𝑧∈Z. Recall that a function 𝑓is convex and 𝐿-Lipschitz if
for any 𝑤1, 𝑤2 ∈W and 0 ≤λ ≤1:

𝑓(λ𝑤1 + (1 −λ)𝑤2) ≤λ 𝑓(𝑤1) + (1 −λ) 𝑓(𝑤2), and, | 𝑓(𝑤1) −𝑓(𝑤2)| ≤𝐿∥𝑤1 −𝑤2∥.
(1)

First order optimization
Algorithmically we require further assumptions concerning any interaction
with the function to be optimized. Recall [19] that, for fixed 𝑧, the sub-gradient set of 𝑓(𝑤, 𝑧) at point
𝑤is the set:
𝜕𝑓(𝑤, 𝑧) =

𝑔: 𝑓(𝑤′, 𝑧) ≥𝑓(𝑤, 𝑧) + 𝑔⊤(𝑤′ −𝑤), ∀𝑤′ ∈W
	
.

A first order oracle for 𝑓is a mapping O𝑧(𝑤) such that O𝑧(𝑤) ∈𝜕𝑓(𝑤, 𝑧). Our underlying assumption
is that a learner has a first order oracle access. In other words, given a function 𝑓(𝑤, 𝑧), we will
assume that there is a procedure O𝑧that calculates and returns a subgradient at every 𝑤for every 𝑧.
Recall [10, 19] that when |𝜕𝑓(𝑤, 𝑧)| = 1, the function is differentiable, at 𝑤, and in that case, the
unique subdifferential is the gradient ∇𝑓(𝑤, 𝑧).

2


---Page Break---
Learning

A learning algorithm 𝐴, in SCO, is any algorithm that receives as input a sample 𝑆= {𝑧1, . . . , 𝑧𝑚} ∈Z𝑚
of 𝑚examples, and outputs a parameter 𝑤𝑆. An underlying assumption in learning is that there exists
a distribution 𝐷, unknown to the learner 𝐴, and that the sample 𝑆is drawn i.i.d from 𝐷. The goal of
the learner is to minimize the population loss:
𝐹(𝑤) = 𝔼
𝑧∼𝐷[ 𝑓(𝑤, 𝑧)],

More concretely, We will say that the learner has sample complexity 𝑚(ε) if, assuming |𝑆| ≥𝑚(ε),
then w.p. 1/2 (Again, because we mostly care about lower bounds, fixing the confidence will not
affect the generality of our result):
𝐹(𝑤𝑆) −min
𝑤∈W 𝐹(𝑤) ≤ε.
(2)

Empirical Risk Minimization
A natural approach to perform learning is by Empirical Risk
Minimization (ERM). Given a sample 𝑆, the empirical risk is defined to be:

𝐹𝑆(𝑤) = 1

|𝑆|

∑︁

𝑧∈𝑆
𝑓(𝑤, 𝑧).

An ε-ERM is any algorithm that, given sample 𝑆, returns a solution 𝑤𝑆∈𝑊that minimizes the
empirical risk up to additive error ε > 0. Recently, Carmon et al. [11] showed that any ε-ERM
algorithm has a sample complexity bound of

𝑚(ε) = ˜𝑂
 𝑑

ε + 1

ε2


,
(3)

The above rate is optimal up to logarithmic factor [12]. Namely, there exists a construction and
an ERM that will fail, w.p. 1/2, unless 𝑚= Ω(𝑑/ε) examples are provided1. Importantly, though,
there are algorithms that can learn with much smaller sample complexity. In particular SGD [26],
stable-GD [6] and regularized ERMs [8].

Gradient Descent

We next depict Gradient Descent whose sample complexity is the focus of this work. GD depends
on hyperparameters 𝑇∈ℕand η ≥0 and operates as follows on the empirical risk. The algorithm
receives as input a sample 𝑆= {𝑧1, . . . , 𝑧𝑚}, defines 𝑤0 = 0, and operates for 𝑇iterations according
to the following recursion:

𝑤𝑡= Π

"

𝑤𝑡−1 −η

|𝑆|

∑︁

𝑧∈𝑆
O𝑧(𝑤𝑡)

#

⇒𝑤𝐺𝐷
𝑆
:= 1

𝑇

𝑇
∑︁

𝑡=1
𝑤𝑡,
(4)

where Π is the projection onto the unit ball, and O𝑧(𝑤𝑡) is a subgradient of the loss function 𝑓(𝑤, 𝑧)
at 𝑤𝑡. The final output, 𝑤𝐺𝐷
𝑆
, of the algorithm is the averaged iterate (our result, though, can be
generalized to other possible suffix-averages such as, say, outputting the last iterate, see Theorem 10).
GD constitutes an ε-ERM. Concretely, it is known [10, 17] that GD minimizes the empirical risk and
its optimization error is given by:

𝐹𝑆(𝑤𝐺𝐷
𝑆
) −min
𝑤∈𝑊𝐹𝑆(𝑤) = Θ

min


η + 1

η𝑇, 1

.
(5)

The above bound is tight irrespective of the dimension2. The population loss have also been studied,
and Bassily et al. [6] demonstrated the following learning guarantee:

𝔼
𝑆∼𝐷𝑚


𝐹(𝑤𝐺𝐷
𝑆
) −min
𝑤∈𝑊𝐹(𝑤)

= 𝑂


η
√

𝑇+ 1

η𝑇+ η𝑇

𝑚


.
(6)

The last two terms in the RHS follow from a stability argument, provided in [6], and the first term
follows from the optimization error of GD as depicted in Eq. (5). Notice that there is always an
𝑂(η
√

𝑇) gap between the generalization error and empirical error of gradient descent.

1the Ω(1/ε2) sample complexity bound is more straightforward and follows from standard information-
theoretic arguments
2For completeness, we demonstrate the lower bound for 𝑑= 1 at Appendix E

3


---Page Break---
3
Main Result

Theorem 1. For every 𝑑≥4096,𝑇≥10, 𝑚≥1 and η > 0, there exists a distribution 𝐷, and a
4-Lipschitz convex function 𝑓(𝑤, 𝑧) in ℝ𝑑+1, such that for any first order oracle of 𝑓(𝑤, 𝑧), with
probability 1/2, if we run GD with η as a learning rate then:

𝐹(𝑤GD
𝑆) −min
𝑤∈W 𝐹(𝑤) ≥
1
2 · 272 · 162 · min

𝑑
1032𝑚, 1

· min
n
η
√︁

min{⌊𝑑3/136⌋,𝑇}, 1
o
.

We remark, that the above theorem is true for any suffix averaging (e.g. last iterate), and not restricted
to the averaged iterate (see Theorem 10). We now specialize our bound for two interesting regimes.
First, we improve previous dependence in the dimension in [3, 20] and obtain a generalization error
bound for 𝑑= Ω(𝑚+ 𝑇1/3):

Corollary 2. Fix η, and suppose 𝑑= Ω 𝑚+ 𝑇1/3. There exists a distribution 𝐷, and an 𝑂(1)-
Lipschitz convex function 𝑓(𝑤, 𝑧) in ℝ𝑑, such that for any first order oracle of 𝑓(𝑤, 𝑧), with probability
1/2, if we run GD for 𝑇iterations, then:

𝐹(𝑤GD
𝑆) −min
𝑤∈W 𝐹(𝑤) ≥Ω

min


η
√

𝑇+ 1

η𝑇, 1

.
(7)

The first term follows from Theorem 1, the second term follows from the optimization error in Eq. (5).
Equation (7) does not hold for 𝑑< 𝑚, and the linear improvement over [2, 20] is tight. This can
be seen from Eq. (5) that shows that GD achieves ε empirical excess error when η = 𝑂(1/
√

𝑇) and
𝑇= 𝑂(1/ε2). Equation (7) becomes vacuous for such choice of parameters, but Carmon et al. [11]
showed that the sample complexity of any ERM is bounded by ˜𝑂((𝑑+√𝑚)/𝑚). However, as depicted
next, this upper bound becomes tight and GD does not improve over a worst-case ERM:

Corollary 3. Suppose 𝑇= 𝑂(𝑚1.5), and η = Θ(1/
√

𝑇). There exists a distribution 𝐷, and an
𝑂(1)-Lipschitz convex function 𝑓(𝑤, 𝑧) in ℝ𝑑, such that for any first order oracle of 𝑓, with probability
1/2, if we run GD with η, for 𝑇iterations:

𝐹(𝑤GD
𝑆) −min
𝑤∈W 𝐹(𝑤) ≥Ω

min
 𝑑

𝑚+
1
√𝑚, 1

.

Corollary 3 complements Carmon et al. [11] upper bound, and improves over Feldman [12] lower
bound that only showed existence of some ERM with the aforementioned sample complexity. To
see that Corollary 3 follows from Theorem 1, notice that when 𝑑≤√𝑚, then 𝑑/𝑚< 1/√𝑚and the
bound is dominated by the second term, which is a well known-information theoretic lower bound
for learning. When 𝑑> √𝑚, and 𝑇< 𝑚1.5 we have that 𝑇≤𝑑3, plugging η = 𝑂(1/
√

𝑇) yields the
bound.

3.1
Discussion

Theorem 1 provides a new generalization error bound for GD. It shows that the worst case sample
complexity for ERMs, derived by Feldman [12], is in fact applicable also to a very natural first order
algorithm and not just to abstract ERMs. This Highlights the importance of choosing the right
algorithm for learning in SCO. As discussed, the bound is tight in several regimes, nevertheless still
there are unresolved open problems.

Stability in low dimension
When GD is treated as a naive empirical risk minimizer, and η =
𝑂(1/
√

𝑇), 𝑇= 𝑂(𝑚), there is no improvement, when using GD, over a worst-case ERM. In the other
direction, for dimension that is linear in 𝑚, one cannot improve over the Ω(η
√

𝑇) term that governs
stability. Our bound, though, provide a hope that stability in low dimension can yield an improved
bound. In particular, consider the case where η = 1/𝑇1/4 and 𝑑< 𝑚. This is a case where we apply
a stable algorithm in small dimensions. Our bound does not negate the possibility of an improved
generalization bound. That would mean that, at least at some regime, GD can improve over the
worst-case ERM behaviour. We leave it as an open problem for future study

4


---Page Break---
Open Question 4. Is there a generalization bound for GD such that

𝔼
𝑆∼𝐷𝑚


𝐹(𝑤𝐺𝐷
𝑆
) −min
𝑤∈W 𝐹(𝑤)

= 𝑂

 
𝑑η
√

𝑇
𝑚
+
1
√𝑚

!

.

Alternatively, can we prove an improved generalization error bound such that:

𝔼
𝑆∼𝐷𝑚


𝐹(𝑤𝐺𝐷
𝑆
) −min
𝑤∈W 𝐹(𝑤)

= Ω

min
 𝑑

𝑚, η
√

𝑇, 1

.

Late stopping
Another regime where there is a gap between known upper bound and lower bound
appears when 𝑇= Ω(𝑚2). Specifically, the stability upper bound for GD by Bassily et al. [6] gives

𝔼
𝑆∼𝐷𝑚


𝐹(𝑤𝐺𝐷
𝑆
) −min
𝑤∈W 𝐹(𝑤)

= 𝑂


η
√

𝑇+ η𝑇

𝑚+ 1

η𝑇


.

By Corollary 2, for large enough dimension:

𝔼
𝑆∼𝐷𝑚


𝐹(𝑤𝐺𝐷
𝑆
) −min
𝑤∈W 𝐹(𝑤)

= Ω

min


η
√

𝑇+ 1

η𝑇, 1

.

When 𝑇= 𝑂(𝑚2), the two bounds coincide. Indeed, for the η𝑇/𝑚term to dominate the η
√

𝑇term,
we must have 𝑇= Ω(𝑚2). One has to take at least 𝑇= 𝑂(𝑚2) iterations in order to generalize with
GD (in fact, any full batch method [2]), however 𝑇= 𝑂(𝑚2) iterations are sufficient. Nevertheless,
the above gap does yield the possibility of an unstable GD method that does generalize. Particularly,
if we just regulate the term η
√

𝑇, but allow η𝑇/𝑚= Ω(1), then this may yield a regime where GD is
unstable (and ERM bounds do not apply) and yet generalize.
Open Question 5. Are there choices of η and 𝑇(that depend on 𝑚) such that η𝑇/𝑚∈Ω(1), but GD
has dimension indpendent sample complexity?

Notice that the η𝑇/𝑚term also governs stability in the smooth convex optimization setup [13]. Recall
that a function 𝑓(𝑤, 𝑧) is said to be β-smooth if for all 𝑧, 𝑓(𝑤, 𝑧) is differentiable, and the gradient is
an β-Lipschitz mapping [10, 15]. For smooth optimization, even if η
√

𝑇= Ω(1), GD is still stable.
Hardt, Recht, and Singer [13] showed that the stability of GD in the smooth case is governed by
𝑂

η𝑇

𝑚

for η < 1/β. Therefore, the question of generalization when η𝑇/𝑚∈Ω(1) remains open,
even under smoothness assumptions:
Open Question 6. Assume that 𝑓(𝑤, 𝑧) is Θ(1)-smooth. What is the sample complexity of GD, when
η and 𝑇are chosen so that η +
1
η𝑇= 𝑜(1), but η𝑇

𝑚= Ω(1).

4
Technical Overview

We next provide a high level overview of our proof technique. For simplicity of exposition we begin
with the case 𝑇= 𝑚= 𝑑. We begin by a brief overview of previous construction by Amir et al. [3]
that demonstrated Corollary 2 when 𝑚= Ω(log 𝑑). The construction in [3] can be decomposed into
three terms:
𝑓(𝑤, 𝑧) = 𝑔(𝑤, 𝑧) + 𝑁0(𝑤) + ℎ(𝑤, 𝑧).
The function 𝑔has the property that an ERM may fail to learn, unless dimension dependent sample size
is considered. Amir et al. [3] incorporated Shalev-Shwartz et al. [22] construction. Later, [20] used
Feldman’s function [12] to construct 𝑔. The shift from the construction depicted in Shalev-Shwartz
et al. [22] to Feldman’s function is the first step that allows to move from logarithmic to polynomial
dependence in the dimension. In both constructions an underlying property of 𝑔is that there exists a
distribution 𝐷such that, for small samples, there are overfitting minima. Concretely, there exists a
𝑤𝑆∈{0, 1/
√

𝑑}𝑑such that
1
|𝑆|

∑︁

𝑧∈𝑆
𝑔(𝑤𝑆, 𝑧) −𝔼
𝑧∼𝐷[𝑔(𝑤𝑆, 𝑧)] = Ω(1).
(8)

The challenge is then, to make gradient descent’s trajectory move towards the point 𝑤𝑆. The idea can
be decomposed into two parts:

5


---Page Break---
Simplifying with an adversarial subgradient:

To simplify the problem, let us first ease the challenge and suppose we can choose our subgradient
oracle in a way that depends on the observed sample. Let 𝑁0 be the Nemirovski function [9]:

𝑁0(𝑤) = max{−𝑤(𝑖), 0)}.

Notice that 𝑁0 is not differentiable and the choice of subgradient at certain points is not apriori
determined. For example, notice that every standard basis vector −𝑒𝑖∈𝜕𝑁0(0). More generally,
given a sample 𝑆, let 𝐼= 𝑖1, . . . , 𝑖𝑑′ be exactly the set of indices such that 𝑤𝑆, from Eq. (8), 𝑤𝑆(𝑖) ≠0.
Now assume by induction that 𝑤𝑡(𝑖) > 0 exactly for 𝑖= 𝑖1, . . . 𝑖𝑡, then one can show that we can
define the subgradient oracle of 𝑁0:

O(𝑤𝑡) = −𝑒𝑖𝑡+1 ∈𝜕𝑁0(𝑤𝑡).

In that case 𝑤𝑡+1 will satisfy our assumption for 𝑖𝑡+1 and we can continue to follow this dynamic for 𝑇
steps.

Notice that, in this case, GD will converge to 𝑤𝑆(if η = 1/
√

𝑑which we assume now for concreteness).
One can also show that the output of GD (the averaged iterate) will overfit. The caveat is that our
subgradient oracle depends on the sample 𝑆. In reality, the sample is drawn independent of the
subgradient oracle. and all previous constructions, as well as ours need to handle this. This is
discussed in the next section. But before that, let us review another challenge which is when 𝑇≠𝑑:

When 𝑑≪𝑇
Another challenge we face with the construction above is that it works when we
assume that 𝑇≈𝑑. That is because, in Nemirovski’s function, the number of iterates we can perform
is bounded by the dimension. After 𝑑iterations we will end up with the vector 𝑣= P𝑑
𝑡=1 η𝑒𝑖𝑡.
If 𝑇= ω(𝑑) then η = 𝑜(1/
√

𝑑), and the dynamic will end up with a too small norm vector to
induce a sizeable population loss. This strategy will provide, at best, with a factor of the form

Ω

η
√︁

min{𝑑,𝑇}

. Such a factor may be unsatisfactory in a very natural setting where, say, 𝑇= 𝑂(𝑚),

η = 𝑂(1/√𝑚), and 𝑑= Ω(√𝑚). To obtain the 𝑑3 dependence, we perform the following alternation
over the Nemirovski function. Consider the function:

𝑁(𝑤) = max{0, max
𝑖≤𝑑{−𝑤(𝑖)}, max
𝑖≤𝑗≤𝑑{𝑤( 𝑗) −𝑤(𝑖)}}.
(9)

And suppose that at each iteration we return a subgradient as follows:

• If there is 𝑖≤𝑑, such that 𝑤(𝑖) = 𝑤(𝑖+ 1) > η, return subgradient 𝑒𝑖+1 −𝑒𝑖and 𝑤is updated
by 𝑤𝑡+1 = 𝑤𝑡−η𝑒𝑖+1 + η𝑒𝑖.

• If there is no such 𝑖, then take the minimal 𝑖(if exists) such that 𝑤(𝑖) = 0, and return
subgradient −𝑒𝑖and update 𝑤𝑡+1 = 𝑤𝑡+ η𝑒𝑖.

• When non of the above is met, return subgradient 0.

The dynamic of the above scheme is depicted in Fig. 1, and solves the problem when 𝑇≈𝑑3. One
can show that GD will run for at least 𝑑3 ≈𝑇iterations, and will increase 𝑂(𝑑) coordinates, each, on
average, by an order of 𝑂(η𝑑). This is better than the increase of η in each coordinate that we get
from Nemirovski’s function. In this way we obtain the improved result of η
√

𝑇, even when 𝑇≈𝑑3.

When 𝑇≪𝑑,
when the number of iterations is smaller than 𝑑we face a different challenge. The
immediate solution is to embed in ℝ𝑑a construction from ℝ𝑇, this will provide us with the Ω(η
√

𝑇)
term but, on the other hand, such a construction will not yield a Ω(𝑑/𝑚) term. A different approach,
that exploits the dimension to its fullest, is to consider blocks of coordinates and operate on those
instead of single coordinates.

The conclusive outcome incorporates both ideas together, and we replace the Nemirovski function
with a version of Eq. (9) that operates on 𝑂(𝑇1/3) blocks of coordinates. And this concludes our
construction. We next move on to the challenge of replacing the data dependent oracle with a standard
first order oracle.

6


---Page Break---
Figure 1: Depiction of the dynamics induced by Eq. (16) and our choice of sub-differentials

Reduction to sample dependent oracle:

As discussed, the construction above does not yield a lower bound as it relies on a subgradient oracle
that is dependent on the whole sample. To avoid such dependence of the oracle on the sample, we
observe that if we can infer the sample 𝑆from the trajectory, i.e. if the state 𝑤𝑡“encodes" the sample,
then formally the subgradient is allowed to “decipher" the sample from the point 𝑤𝑡. In that way
we achieve this behaviour of sample dependent subgradient oracle. This part becomes challenging
and may depend on the way we choose 𝑔, and 𝑁. The simplest case, studied by Amir et al. [3],
introduced the third function, ℎ, which was a small perturbation function that elevates coordinates in
𝐼and inhibits coordinates not in 𝐼. The function ℎdepends on 𝑧and not on 𝑆, hence it cannot know
apriori 𝐼. But, an important observation is that, in Shalev-Shwartz et al. [22] construction, if 𝑖∉𝐼,
there exists 𝑧∈𝑆that “certifies" that. In fact, each 𝑧can be thought of as a subset of indices, and if
an index appears in 𝑧, then it cannot be in 𝐼. So we can build the perturbation in a way that every
coordinate is elevated, unless 𝑧certifies 𝑖∉𝐼: In that case we define ℎ(𝑤, 𝑧) so that its gradient will
radically inhibit 𝑖.

The last observation is what becomes challenging in our case. As discussed, to achieve improved
rate, we need to use Feldman’s function. When using Feldman’s function the coordinates cannot be
ruled out, or identified, by a single 𝑧but one has to look at the whole sample to identify 𝐼. While
Schliserman et al. [20] tackle a similar problem, we take a slightly different approach described next:
For each 𝑧assign a random, positive, number α(𝑧). We can think of this α as a hash function. Let us
add another coordinate to the vector 𝑤, 𝑤(𝑑+ 1). Consider the function

ℎ(𝑤, 𝑧) = γα(𝑧) · 𝑤(𝑑+ 1).

Then 𝜕ℎ(𝑤, 𝑧) = γα(𝑧)𝑒𝑑+1. Write α(𝑆) =
1
|𝑆|
P
𝑧∈𝑆α(𝑧) then in turn:

𝑤𝑡(𝑑+ 1) = 𝑤𝑡−1(𝑑+ 1) −𝜕γ

|𝑆|

∑︁

𝑧∈𝑆
ℎ(𝑤𝑡−1, 𝑧) = −𝑡· γα(𝑆)𝑒𝑑+1.

If γ, α(𝑧) are chosen correctly, α(𝑆) is a one to one mapping from samples to real numbers, and small
γ ensures that the overall addition of ℎhas negligible affect on the outcome. Then, we may define the
subgradient oracle to be dependent on coordinate 𝑑+ 1 which encodes the whole sample. Our final
construction will take a different ℎ, which adds small strong convexity in this coordinate, for reasons
next explained:

Working with any first order oracle
Notice that our statement is slightly stronger than what we
so far illustrated. Theorem 1 states that, for any subgradient oracle, GD will fail. For that, we need
to be a little bit more careful, and we want to replace our function with a function that leads to the

7


---Page Break---
same guaranteed trajectory, but at the same time it should be differentiable at visited points. This will
ensure a unique derivative, making the construction independent of the choice of (sub)gradient oracle.

Towards this goal, we start with the construction depicted so far and consider the set of all values,
gradients, and points { 𝑓𝑗, 𝑔𝑗, 𝑤𝑗} 𝑗∈𝐽that our algorithm may visit, for any possible time step and any
possible sample, with our construction. Notice that, while this set may be big and even exponential, it
is nevertheless finite. What we want is to interpolate a new function through these triplets. In contrast
with our original construction, we require a differentiable function at the designated points. Notice,
that such an interpolation will have the exact same behaviour when implementing GD on it (with the
added feature that the oracle is well defined and unique).

The problem of convex interpolation is well studied, for example Taylor et al. [23] shows sufficient
and necessary conditions for interpolation of a smooth function. Our case is slightly easier as we
do not care about the smoothness parameter. On the other hand we do require Lipschitzness of the
interpolation. We therefore provide an elementary, self-contained, proof to the following easy to
prove Lemma, (proof is provided in Appendix B)
Lemma 7. Let 𝐺= { 𝑓𝑗, 𝑔𝑗, 𝑤𝑗} 𝑗∈𝐽⊆ℝ× ℝ𝑑× ℝ𝑑be a triplet of values in ℝ, and gradients and
points in ℝ𝑑, such that ∥𝑔𝑗∥≤𝐿. Suppose that for every 𝑖, 𝑗∈𝐽:

𝑓𝑖≥𝑓𝑗+ 𝑔⊤
𝑗(𝑤𝑖−𝑤𝑗),
(10)

and let
𝐼diff = {𝑖: 𝑓𝑖= 𝑓𝑗+ 𝑔⊤
𝑗(𝑤𝑖−𝑤𝑗) ⇒𝑔𝑖= 𝑔𝑗}.

Then there exists a convex 𝐿-Lipschitz function ˆ𝑓such that for all 𝑗∈𝐽: ˆ𝑓(𝑤𝑗) = 𝑓𝑗, and for all
𝑖∈𝐼diff, ˆ𝑓is differentiable at 𝑤𝑖and:
∇𝑓(𝑤𝑖) = 𝑔𝑖.

With Lemma 7 at hand, consider the function

ℎ(𝑤, 𝑧) = 1

2 (𝑤(𝑑+ 1))2 + α(𝑧) · 𝑤(𝑑+ 1).

The above function encodes in 𝑤(𝑑+ 1) the sample and time-step as before. Moreover, because it is
slightly strongly convex (in coordinate 𝑑+ 1), 𝑤1(𝑑+ 1) ≠𝑤2(𝑑+ 1) ensures that

ℎ(𝑤1, 𝑧) > ℎ(𝑤2, 𝑧) + ∇ℎ(𝑤2, 𝑧)⊤(𝑤1 −𝑤2),

Then the term ℎin 𝑓ensures that the triples { 𝑓𝑗, 𝑔𝑗, 𝑤𝑗} along the trajectory generate gradient
vectors that satisfy strict inequality in Eq. (10) and in turn, our interpolation from Lemma 7 will be
differentiable at these points. There’s some technical subtlety because the interpolation needs to also
take the averaged iterate into account, but this is handled in a similar fashion.

In the next two sections we provide more formal statements of the two main ingredients: First, we
define a setup of optimization with a sample-dependent first order Oracle and state a lower bound for
the generalization error in this setup. The second ingredient is a reduction from the standard setup of
first order optimization.

4.1
Sample-dependent Oracle

As discussed, the first step in our proof is to consider a slightly weaker setup where the first-order
oracle may depend on the whole sample. Let us formally define what we mean by that. Define

S𝑇
𝑚= {S = (𝑆1, . . . , 𝑆𝑡), 𝑆𝑖∈∪𝑚
𝑖=1Z𝑚, 𝑡≤𝑇},

the set of all subseqences of samples of size at most 𝑚. Given a function 𝑓(𝑤, 𝑧), a sample dependent
oracle, OS, is a finite sequence of first order oracles

OS = {O(𝑡) (𝑆; 𝑤, 𝑧)}𝑇
𝑡=1,

that each receive as input a finite sample 𝑆, as well as 𝑤and returns a subgradient:

O(𝑡) (𝑆, 𝑤, 𝑧) ∈𝜕𝑓(𝑤, 𝑧).

8


---Page Break---
The sequence of samples can be thought of as the past samples that were observed by the algorithm.
In the case of full-batch GD these will be the whole sample, and for SGD, each 𝑆provided to O(𝑡)
will be all previously observed samples. Given S ∈S𝑇
𝑚let us also denote

O(𝑡) (S, 𝑤) =
1
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
O(𝑡) (𝑆1:𝑡−1, 𝑤, 𝑧) ∈𝜕©­
«

1
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
𝑓(𝑤, 𝑧)ª®
¬
,
(11)

where we let S1:0 = ∅, and 𝑆1:𝑡−1 = (𝑆1, . . . , 𝑆𝑡−1) is the concatenated subsample of all previously
observed samples in the sequence. As discussed, working with a sample-dependent oracle is easier
(for lower bounds). And indeed, our first result shows that, if the subgradient can be chosen in a way
that depends on the sample, we can provide the desired lower bound. For fixed and known η > 0, 𝑇,
a sample dependent first order oracle OS, and a sequence of samples S = (𝑆1, 𝑆2, . . . , 𝑆𝑇), define
𝑤0 = 0 and inductively:

𝑤S
𝑡= Π
h
𝑤S
𝑡−1 −ηO(𝑡) (S, 𝑤S
𝑡−1)
i
,

and for every suffix s < 𝑇:

𝑤𝐺𝐷
S,s =
1
𝑇−s

𝑇
∑︁

𝑡=s+1
𝑤S
𝑡
(12)

Lemma 8. For every 𝑚, 𝑑,𝑇≥18 and η > 0 there are a distribution 𝐷, a 3-Lipschitz convex function
𝑓(𝑤, 𝑧) in ℝ𝑑, as well as a sample dependent first order oracle OS such that: if S = (𝑆, 𝑆, . . . 𝑆) ∈S𝑇
𝑚
for 𝑆∼𝐷𝑚i.i.d, then w.p. 1/2, for every suffix averaging s:

𝐹(𝑤𝐺𝐷
S,s ) −𝐹(0) ≥
1
√

2 · 272 · 162 · min

𝑑
1032𝑚, 1

· min
n
η
√︁

min{⌊𝑑3/136⌋,𝑇}, 1
o
.

The proof of Lemma 8 is provided in Appendix A.1. We next move to describe the second ingredient
of our proof.

4.2
Reduction to sample-dependent oracles

As discussed, the second ingredient of our proof is a reduction to the sample-dependent setup. Instead
of using a perturbation function as in [3], we take a more black box approach and show that, given a
sample dependent first order oracle, there exists a function that basically induces the same trajectory.
Proof is provided in Appendix A.2.
Lemma 9. Suppose 𝑞∈ℝ𝑇, ∥𝑞∥∞≤1. And suppose that 𝑓(𝑤, 𝑧) is a convex, 𝐿-Lipschitz, function
over 𝑤∈ℝ𝑑, let η > 0, let OS be a sample dependent first order oracle, and for every sequence of
samples S = (𝑆1, 𝑆2, . . . , 𝑆𝑇) define the sequence {𝑤S
𝑡}𝑇
𝑡=1 as in Eq. (12).

Then, for every ε > 0 there exists an 𝐿+ 1 Lipschitz convex function3 ¯𝑓((𝑤, 𝑥), 𝑧) over ℝ𝑑+1 (that
depends on 𝑞, 𝑓,𝑇, η, 𝑚, OS, ε).

such that for any first order oracle O𝑧for ¯𝑓, define 𝑢0 = 0 ∈ℝ𝑑and 𝑥0 = 0 ∈ℝ, and:

(𝑢𝑡, 𝑥𝑡) = (𝑢𝑡−1, 𝑥𝑡−1) −
η
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
O𝑧((𝑢𝑡, 𝑥𝑡))

then if we define:

𝑢𝑞=

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑢𝑡, and,
𝑥𝑞=

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑥𝑡, and 𝑤S
𝑞=

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑤S
𝑡.

then, we have that 𝑢𝑞= 𝑤S
𝑞and:

| ¯𝑓((𝑢𝑞, 𝑥𝑞), 𝑧) −𝑓(𝑤S
𝑞, 𝑧)| ≤ε.

and,
| ¯𝑓((0, 0), 𝑧) −𝑓(0, 𝑧)| ≤ε.

3i.e. 𝑤∈ℝ𝑑and 𝑥∈ℝ

9


---Page Break---
Ackgnoweledgments
The author would like to thank Tamar Livni for creating Figure 1. Tamar
holds all copyrights to the artwork. The author would also like to thank Tomer Koren and Yair Carmon
for several discussions. This research was funded in part by an ERC grant (FOG, 101116258), as well
as an ISF Grant (2188 \ 20).

References

[1] N. Alon, S. Ben-David, N. Cesa-Bianchi, and D. Haussler. Scale-sensitive dimensions, uniform
convergence, and learnability. Journal of the ACM (JACM), 44(4):615–631, 1997.

[2] I. Amir, Y. Carmon, T. Koren, and R. Livni. Never go full batch (in stochastic convex
optimization). Advances in Neural Information Processing Systems, 34:25033–25043, 2021.

[3] I. Amir, T. Koren, and R. Livni. Sgd generalizes better than gd (and regularization doesn’t help).
In Conference on Learning Theory, pages 63–92. PMLR, 2021.

[4] I. Amir, R. Livni, and N. Srebro. Thinking outside the ball: Optimal learning with gradient
descent for generalized linear stochastic convex optimization. Advances in Neural Information
Processing Systems, 35:23539–23550, 2022.

[5] I. Attias, G. K. Dziugaite, M. Haghifam, R. Livni, and D. M. Roy. Information complexity
of stochastic convex optimization: Applications to generalization and memorization. arXiv
preprint arXiv:2402.09327, 2024.

[6] R. Bassily, V. Feldman, C. Guzmán, and K. Talwar. Stability of stochastic gradient descent on
nonsmooth convex losses. Advances in Neural Information Processing Systems, 33:4381–4391,
2020.

[7] A. Blumer, A. Ehrenfeucht, D. Haussler, and M. K. Warmuth. Learnability and the vapnik-
chervonenkis dimension. Journal of the ACM (JACM), 36(4):929–965, 1989.

[8] O. Bousquet and A. Elisseeff. Stability and generalization. The Journal of Machine Learning
Research, 2:499–526, 2002.

[9] S. Bubeck, Q. Jiang, Y.-T. Lee, Y. Li, and A. Sidford. Complexity of highly parallel non-smooth
convex optimization. Advances in neural information processing systems, 32, 2019.

[10] S. Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends®
in Machine Learning, 8(3-4):231–357, 2015.

[11] D. Carmon, R. Livni, and A. Yehudayoff. The sample complexity of erms in stochastic convex
optimization. arXiv preprint arXiv:2311.05398, 2023.

[12] V. Feldman. Generalization of erm in stochastic convex optimization: The dimension strikes
back. Advances in Neural Information Processing Systems, 29, 2016.

[13] M. Hardt, B. Recht, and Y. Singer. Train faster, generalize better: Stability of stochastic gradient
descent. In International conference on machine learning, pages 1225–1234. PMLR, 2016.

[14] D. Haussler and M. Warmuth. The probably approximately correct (pac) and other learning
models. The Mathematics of Generalization, pages 17–36, 2018.

[15] E. Hazan et al. Introduction to online convex optimization. Foundations and Trends® in
Optimization, 2(3-4):157–325, 2016.

[16] R. Livni. Information theoretic lower bounds for information theoretic upper bounds. Advances
in Neural Information Processing Systems, 36, 2023.

[17] Y. Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer
Science & Business Media, 2013.

[18] B. Neyshabur, R. Tomioka, and N. Srebro. In search of the real inductive bias: On the role of
implicit regularization in deep learning. arXiv preprint arXiv:1412.6614, 2014.

10


---Page Break---
[19] R. T. Rockafellar. Convex Analysis:(PMS-28). Princeton university press, 2015.

[20] M. Schliserman, U. Sherman, and T. Koren. The dimension strikes back with gradients: General-
ization of gradient methods in stochastic convex optimization. arXiv preprint arXiv:2401.12058,
2024.

[21] A. Sekhari, K. Sridharan, and S. Kale. Sgd: The role of implicit regularization, batch-size and
multiple-epochs. Advances In Neural Information Processing Systems, 34:27422–27433, 2021.

[22] S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan. Stochastic convex optimization. In
COLT, volume 2, page 5, 2009.

[23] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Smooth strongly convex interpolation and exact
worst-case performance of first-order methods. Mathematical Programming, 161:307–345,
2017.

[24] V. N. Vapnik and A. Y. Chervonenkis. On the uniform convergence of relative frequencies of
events to their probabilities. In Measures of complexity: festschrift for alexey chervonenkis,
pages 11–30. Springer, 2015.

[25] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning (still)
requires rethinking generalization. Communications of the ACM, 64(3):107–115, 2021.

[26] M. Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In
Proceedings of the 20th international conference on machine learning (icml-03), pages 928–936,
2003.

A
Proof of Theorem 1

The proof is an immediate corollary of Lemmas 8 and 9, which we prove in Appendices A.1 and A.2
respectively. To see that Theorem 1 indeed follows from these Lemmas, start with η, 𝑑, 𝑚,𝑇that
satisfy the conditions. Let 𝑓(𝑤, 𝑧) be the function and OS the sample dependent first order oracle,
whose existence follows from Lemma 8 with suffix s = 0. And let ¯𝑓be the function whose existence
follows from Lemma 9 to some arbitrarily small ε0, with 𝑞(𝑡) = 1

𝑇for all 𝑡. It is easy to see that if we
apply GD on ¯𝑓and define its output (𝑢𝐺𝐷, 𝑥𝐺𝐷) then: (𝑢𝐺𝐷, 𝑥𝐺𝐷) = (𝑢𝑞, 𝑥𝑞), and 𝑤S
𝑞= 𝑤𝐺𝐷
S
.

Then, we have that w.p. 1/2:

¯𝐹((𝑢𝐺𝐷, 𝑥𝐺𝐷)) −¯𝐹(0) = ¯𝐹(𝑢𝑞, 𝑥𝑞) −¯𝐹(0)

≥𝐹(𝑤𝐺𝐷
S
) −2ε0 −𝐹(0)

≥
1
√

2 · 272 · 162 · min

𝑑
1032𝑚, 1

· min
n
η
√︁

min{⌊𝑑3/136⌋,𝑇}, 1
o
−2ε0

≥
1
2 · 272 · 162 · min

𝑑
1032𝑚, 1

· min
n
η
√︁

min{⌊𝑑3/136⌋,𝑇}, 1
o
.

Where in the last equation, we assume ε0 to be sufficiently small.
Finally, note that ¯𝐹(0) ≥
min𝑤∈W ¯𝐹(𝑤).

Notice, that by the same argument, by taking any suffix s < 𝑇, and setting 𝑞(𝑡) = 0 for 𝑡≤s, and
𝑞(𝑡) =
1
𝑇−s for 𝑡≥s + 1, we can obtain the following stronger result for any suffix averaging:

Theorem 10. For every 𝑑≥4096,𝑇≥10, 𝑚≥1 and η > 0 and suffix s < 𝑇, there exists a
distribution 𝐷, and a 4-Lipschitz convex function 𝑓(𝑤, 𝑧) in ℝ𝑑, such that for any first order oracle
of 𝑓(𝑤, 𝑧), with probability 1/2, if we run GD with η as a learning rate then:

𝐹

 
1
𝑇−s

𝑇
∑︁

𝑡=s+1
𝑤S
𝑡

!

−𝐹(0) ≥
1
2 · 272 · 162 · min

𝑑
1032𝑚, 1

· min
n
η
√︁

min{⌊𝑑3/136⌋,𝑇}, 1
o
.

11


---Page Break---
A.1
Proof of Lemma 8

For simplicity we will assume that 𝑑= 2𝑛for some 𝑛∈ℕ, the final result will be obtained by
embedding a construction in a subspace of size at least half the original dimension.

We start by recalling Feldman’s construction [12]: There exists a set V ⊆{0, 1}𝑑, such that:

∀𝑣1 ≠𝑣2 ∈V, 𝑣1 · 𝑣2 ≤5𝑑

16


, and

∥𝑣∥2 ≥7𝑑

16


, and

|V| > 𝑒𝑑/258
.
(13)

Indeed, suppose we pick randomly 𝑤∈{0, 1}𝑑according to probability 𝑃where each coordinate
𝑃(𝑤(𝑖)) = 1 with probability 1/2 (independently). Then, by Hoeffding’s inequality for two 𝑤1, 𝑤2 ∼𝑃
independently:

𝑃

𝑤1 · 𝑤2 > 𝑑

4 + 𝑑

16


≤𝑒−𝑑

128 ,

𝑃

𝑤1 · 𝑤1 < 𝑑

2 −𝑑

16


≤𝑒−𝑑

128 .

Thus, picking V elements i.i.d according to 𝑃, randomly, of size |V| ≥𝑒𝑑/258 we can show by union
bound that with positive probability, all |V|2 pairs in V will satisfy the requirement. Next, we define a
distribution 𝐷ε supported on subsets of V such that for a random variable 𝑉⊆V each 𝑣∈𝑉w.p. ε
independently. We start by assuming that 𝑇≤
1
17𝑑3 (the case 𝑇> 𝑑3 is handled at the end), and let
𝑘∈ℕbe such that:

𝑑≤𝑘
 𝑇

17

1/3
< 2𝑑.

One can show that without loss of generality we can assume 𝑘is also a power of 2 (in particular, 𝑑
is divisble by 𝑘for large enough 𝑇). We next follow the idea depicted in Section 4, but we want to
handle the case 𝑘≫1. For that, we redefine the function in Eq. (9), and take blocks of coordinates.
To simplify notations, let us define for two set of indices 𝐼, 𝐽of elements in [𝑑]: 𝐼= {𝑖1, . . . , 𝑖𝑘},
𝐽= { 𝑗1, . . . , 𝑖𝑘}, 𝐼≺𝐽if max{𝑖∈𝐼} < min{ 𝑗∈𝐽}. and we will also write:

𝑒𝐼=
1
√︁

|𝐼|

∑︁

𝑖∈𝐼
𝑒𝑖,
and, 𝑤(𝐼) = 𝑤· 𝑒𝐼= 1
√

𝐼

∑︁

𝑖∈𝐼
𝑤(𝑖).

then we define our final function as:

𝑁(𝑤) = max

0, max
|𝐼|=𝑘{−𝑤(𝐼)}, max {−(𝑤(𝐼) −𝑤(𝐽)) : 𝐼≺𝐽, |𝐼| = |𝐽| = 𝑘}

(14)

Define α = min{
1
η
√

2𝑇, 1}, and let:

𝑓(𝑤,𝑉) = 𝑔(𝑤,𝑉) + α𝑁(𝑤).
where 𝑁is defined in Eq. (14), and 𝑔is defined to be Feldman’s function with a suitable choice of
threshold:

𝑔(𝑤,𝑉) = 1
√

𝑑
max
𝑣∈𝑉

 45ηα𝑑2

2 · 162𝑘1.5 , 𝑤· 𝑣

.
(15)

Notice that 𝑁is 2-Lipschitz, and 𝑔is 1-Lipschitz.

To obtain the trajectory, we next define a sample dependent oracle. We only define it for samples S
such that there exists 𝑣★∉𝑉𝑖for all 𝑉𝑖∈𝑆(define it arbitrarily to any other type of sample). Let
I = {𝑖1, . . . , 𝑖𝑑′} be a set of 7𝑑

16 indices such that 𝑣★(𝑖𝑗) ≠0. Divide the elements of I into 𝑑′/𝑘
subsets. Namely, let

𝐼𝑗= {𝑖( 𝑗−1)·𝑘+1, 𝑖( 𝑗−1)·𝑘+2, . . . , 𝑖𝑗·𝑘},
𝑗= 1, . . . , 𝑑′/𝑘.

We start by defining only an oracle for the function α𝑁. We will later show that the trajectory induced
by this oracle stays in the minima of 𝑔, and that will show that, for our purposes, we can choose the

12


---Page Break---
same oracle for the whole function 𝑓. We denote by O(𝑡)
α𝑁the sample dependent oracle for α𝑁, and
we first define:

O(1)
α𝑁(∅, 0) = 0,

Next, we define for 𝑡> 1. For any 𝑤such that 0 ∉𝜕𝑁(𝑤), define it arbitrarily. If 0 ∈𝜕𝑁(𝑤) we
define it as follows:

• If there is a multi-index 𝐼𝑗that 𝑤(𝐼𝑗) = 𝑤(𝐼𝑗+1) > ηα: Then:

O(𝑡)
α𝑁(𝑆, 𝑤) = α(𝑒𝐼𝑗+1 −𝑒𝐼𝑗).

• If there is no such multi-index, and if 𝐼𝑗is the minimal multiindex such that 𝑤(𝐼𝑗) = 0, set:

O(𝑡)
α𝑁(𝑆, 𝑤) = −α(𝑒𝐼𝑗).

• If both conditions cannot be met, then:

O(𝑡)
α𝑁(𝑆, 𝑤) = 0.

The trajectory of this dynamic is depicted in Fig. 1, for the case 𝑘= 1. In the general case, we can
think of each coordinate in Fig. 1 as a block of size 𝑘. Now we assume that {𝑤S
𝑡}𝑇
𝑡=1 follows the
trajectory depicted in Eq. (12) with that choice of Oracle. It can be seen that the update step is such
that after 𝑇′ = 1 + P𝑑′/𝑘
𝑡′=1
P𝑡′
𝑡′′=1 𝑡′′ rounds, we will have that for every 𝑖∈𝐼𝑡:

𝑤S
𝑇′ (𝑖) = αη
√

𝑘(𝑑′/𝑘+ 1 −𝑡),
(16)

and that for 𝑡> 𝑇′: 𝑤S
𝑡= 𝑤S
𝑇′

Moreover, one can show that 𝑤𝑡is non zero only in coordinates 𝑖∈I, and that for any subset 𝐼𝐵⊆𝐼
such that |𝐼| = 𝐵:

∑︁

𝑖∈𝐼𝐵
𝑤S
𝑡(𝑖) ≤max

(∑︁

𝑖∈𝐼𝐵
𝑤S
𝑇(𝑖) : 𝐼𝐵⊆𝐼, |𝐼| = 𝐵

)

(17)

A formal proof is provided in Appendix D. Notice that:

1 +

𝑑′/𝑘
∑︁

𝑡′=1

𝑡′
∑︁

𝑡′′=1
𝑡′′ < 𝑑′3

𝑘3 ≤1

5𝑇.

Next, for any 𝑣≠𝑣★, we have that:

13


---Page Break---
𝑤S
𝑡· 𝑣≤

𝑑′/𝑘
∑︁

𝑖=1
𝑤S
𝑡(𝑖𝑡)1{𝑣(𝑖𝑡) = 1}

≤max

(∑︁

𝑖∈𝐼𝐵
𝑤S
𝑇(𝑖) : 𝐼𝐵⊆𝐼, |𝐼| ≤5𝑑

16

)

𝐸𝑞𝑠. (13)𝑎𝑛𝑑(17)

≤

5𝑑
16𝑘
∑︁

𝑡=1

∑︁

𝑖∈𝐼𝑡
𝑤S
𝑇(𝑖)

≤

5𝑑
16𝑘
∑︁

𝑡=1

√

𝑘ηα((𝑑′/𝑘) + 1 −𝑡)
𝐸𝑞. (16)

≤

5𝑑
16𝑘
∑︁

𝑡=0

√

𝑘ηα((𝑑′/𝑘) −𝑡)

≤
√

𝑘ηα

 
5𝑑
16𝑘
𝑑′

𝑘−1

2

 5𝑑

16𝑘

2!

≤
√

𝑘ηα 45𝑑2

2(16𝑘)2
𝑑′ = 7𝑑

16
.

As such 0 ∈𝜕𝑔(𝑤S
𝑡,𝑉𝑖) for all 𝑉𝑖, and we can define O(𝑡) so that

O(𝑡) (𝑆, 𝑤S
𝑡,𝑉𝑖) = O(𝑡)
α𝑁(𝑆, 𝑤S
𝑡).

And we have that, {𝑤S
𝑡} is the trajectory obtained from Eq. (12) with respect to this oracle of 𝑔also.
We also have:

𝑤S
𝑇′ · 𝑣★=
√

𝑘ηα

𝑑′/𝑘
∑︁

𝑡=1
(𝑑′/𝑘+ 1 −𝑡) =
√

𝑘ηα

𝑑′/𝑘
∑︁

𝑡=1
𝑡≥

√

𝑘ηα

2
𝑑′2

𝑘2 ≥
√

𝑘ηα 49𝑑2

2(16𝑘)2 .

And because 𝑇′ ≤
1
17𝑇, for every suffix s ∈[𝑇]:

𝑣★· 𝑤𝐺𝐷
S,s ≥16

17𝑤S
𝑇′ · 𝑣★≥
√

𝑘ηα 46𝑑2

2(16𝑘)2 .

Recall that we assume 136𝑑3

𝑘3
≥𝑇:

𝐹(𝑊𝐺𝐷
S,s )−𝐹𝑆(0) ≥ε

 
46
√

𝑘αη𝑑1.5

2(16𝑘)2
−45
√

𝑘αη𝑑1.5

2(16𝑘)2

!

≥ε
αη
2 · 162

 𝑑

𝑘

1.5
≥
ε
√

2 · 272 · 162 min
n
η
√

𝑇, 1
o

(18)

Eq. (18) lower bounds the generalization under the event that there exists 𝑣∈V such that 𝑣∉𝑉𝑖
for every 𝑖= 1, . . . , 𝑚. Now assume ε = min{
𝑑
516𝑚, 1

4}, then for every 𝑣, using the inequality
(1 −ε) ≤𝑒−2ε for ε < 1/2:

𝑃(𝑣∉∩𝑚
𝑖=1𝑉𝑖) = (1 −ε)𝑚≥𝑒−2ε·𝑚≥𝑒−2𝑑/516,

and,

𝑃(∃𝑣, ∉∩𝑚
𝑖=1𝑉𝑖) ≥1 −(1 −𝑒−2𝑑/516)|𝑉| ≥1 −(1 −𝑒−𝑑/258)𝑒𝑑/258 ≥1 −𝑒−1 ≥1/2.

So far we assume that 𝑑= 2𝑛, notice that if 𝑇<
1
8·17𝑑3, we can find a subspace of size 𝑑1 > 𝑑

2 so that
𝑇<
1
8·17𝑑3
1, and we obtain our final result by embedding our construction in this subspace.

14


---Page Break---
When 𝑇>
1
17·8𝑑3
Notice that if we take the construction with 𝑇= ⌊
1
17·8𝑑3⌋then:

O(𝑇) (𝑤S
𝑇) = 0,

hence we can use the above construction for any 𝑇′ > 𝑇, and the subdifferential oracle is defined for
every iteration above 𝑇as returning 0, and we obtain a similar analysis to the case that 𝑇= ⌊𝑑3/136⌋,
and our bounds yield as in Eq. (18):

𝐹(𝑤𝐺𝐷
S,s ) −𝐹𝑆(0) ≥
ε
√

2 · 272 · 162 min
n
η
√︁

⌊𝑑3/136⌋, 1
o

A.2
Proof of Lemma 9

The main ingredient in our proof is the following claim whose proof is provided in Appendix C. To
state the claim we will need a few notations. First, for any two tuples of sample-sequences in S𝑇
𝑚, and
𝑡> 0 (S, 𝑡) and (S′, 𝑡), let us denote (S, 𝑡) ≡(S′, 𝑡′) if 𝑡= 𝑡′ and 𝑆1:𝑡= 𝑆′
1:𝑡, namely the prefixes of
the sample sequences agree up to point 𝑡′ = 𝑡. Second, for a mapping α : Z →[0, 1], from sample
points to real numbers, and a sample 𝑆, let us denote

α(𝑆) = 1

|𝑆|

∑︁

𝑧∈𝑆
α(𝑧).

Claim 1. For every η, γ and α define for every sequence of samples S = (𝑆1, . . . , 𝑆𝑇), and 𝑡
inductively: 𝑥S
0 = 0 and
𝑥S
𝑡= (1 −γη)𝑥S
𝑡−1 −γηα(𝑆𝑡).
(19)

and define:
𝑥S
𝑞=
∑︁

𝑡=1
𝑞(𝑡)𝑥S
𝑡
(20)

Then, for any ε > 0, there is a choice of γ < ε and α such that:

1. For (S, 𝑡) . (S′, 𝑡′): 𝑥S
𝑡≠𝑥S′
𝑡′ .

2. For every S, 𝑡, such that 𝑡< max{𝑡′ : 𝑞(𝑡′) ≠0} and S′: 𝑥S
𝑡≠P𝑇
𝑡=1 𝑞(𝑡)𝑥S′
𝑡.

The 𝑥S
𝑡represents the different states the trajectory can be in. 𝑥S
𝑞represent the output of the trajectory
which can be an aggreagated sum. We require that each provide a signature for the state, and this will
allow us to “code" the state of the trajectory along GD.

We now continue with the proof of Lemma 8. For every sample dependent first order oracle O(𝑡), we
have that:

O(𝑡) (S; 𝑤) =
1
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
O(𝑡) (𝑆1:𝑡−1; 𝑤, 𝑧).
(21)

To simplify notations, let us denote:

O(𝑡) (𝑆1:𝑡−1, 𝑤S
𝑡, 𝑧) = OS,𝑡,𝑧,

as S, 𝑡, 𝑧completely determine the output. We next define

ℎ𝑧(𝑥) := 1

2γ(𝑥2 −2α(𝑧)𝑥),

where γ > 0 is arbitrarily small4.

and observe that, if 𝑥S
𝑡is defined as in Eq. (19):

𝑥S
𝑡= (1 −γη)𝑥S
𝑡−1 −γηα(𝑆𝑡) = 𝑥S
𝑡−1 −
η
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
∇ℎ𝑧(𝑥S
𝑡−1).
(22)

4γ ≤ε/(η𝑇), will suffice

15


---Page Break---
To simplify notation, let us denote:

𝑤S
𝑇+1 = 𝑤S
𝑞, and 𝑥S
𝑇+1 = 𝑥S
𝑞,

and assume without loss of generality that max{𝑞(𝑖) ≠0} = 𝑇(Otherwise, we look only at the
sequence up to point max{𝑞(𝑖) ≠0}). Consider now the sets of triplets:

𝐺(𝑧) =
n
(𝑣, 𝑔, 𝑢) =

𝑓(𝑤S
𝑡, 𝑧) + ℎ𝑧(𝑥S
𝑡), (OS,𝑡,𝑧, ∇ℎ𝑧(𝑥S
𝑡)), (𝑤S
𝑡, 𝑥S
𝑡)

: S𝑚
𝑇∈S, 𝑡≤𝑇+ 1
o
,

where OS,𝑇+1,𝑧∈𝜕( 𝑓(𝑤S
𝑞) + ℎ𝑧(𝑥S
𝑞)) is chosen arbitrarily.

Convexity of 𝑓+ ℎ𝑧ensure that the triplets in 𝐺(𝑧) satisfy Eq. (10) for all 𝑡≤𝑇+ 1, as in Lemma 7.
To apply the Lemma, we also want to achieve differentiability at points such that 𝑡< 𝑇. Therefore,
take any two triplets

(𝑣𝑖, 𝑔𝑖, 𝑢𝑖) =

𝑓(𝑤S
𝑡𝑖, 𝑧) + ℎ𝑧(𝑥S
𝑡𝑖),

OS,𝑡,𝑧, ∇ℎ𝑧(𝑥S
𝑡𝑖)

,

𝑤S
𝑡𝑖, 𝑥S
𝑡𝑖

, 𝑖= 1, 2,

where 𝑡1 < 𝑇and 𝑡2 ≤𝑇+ 1, and suppose 𝑔1 ≠𝑔2. To simplify notations, let us write 𝑤S
𝑡𝑖= 𝑤𝑖and
𝑥S
𝑡𝑖= 𝑥𝑖.

First, by convexity of 𝑓we have that:

𝑣1 −𝑣2 + 𝑔⊤
2 (𝑢2 −𝑢1) = 𝑓(𝑤1, 𝑧) + ℎ𝑧(𝑥1) −𝑓(𝑤2, 𝑧) −ℎ𝑧(𝑥2) − OS2,𝑡2,𝑧, ∇ℎ𝑧(𝑥2)⊤((𝑤1, 𝑥1) −(𝑤2, 𝑥2))

= 𝑓(𝑤1, 𝑧) −𝑓(𝑤2, 𝑧) −O⊤
S2,𝑡2,𝑧(𝑤1 −𝑤2) + ℎ𝑧(𝑥1) −ℎ𝑧(𝑥2) −∇ℎ𝑧(𝑥2)⊤(𝑥1 −𝑥2)

≥ℎ𝑧(𝑥1) −ℎ𝑧(𝑥2) −∇ℎ𝑧(𝑥2)⊤(𝑥1 −𝑥2) .

Next, because 𝑔1 ≠𝑔2, either ∇ℎ𝑧(𝑥1) ≠∇ℎ𝑧(𝑥2), which implies 𝑥1 ≠𝑥2, or OS1,𝑡1,𝑧≠OS2,𝑡2,𝑧which
implies (S1, 𝑡1) . (S2, 𝑡2) which again implies 𝑥1 ≠𝑥2 by Claim 1. In other words, if 𝑔1 ≠𝑔2 then
𝑥1 ≠𝑥2:

ℎ𝑧(𝑥1) −ℎ𝑧(𝑥2) −∇ℎ𝑧(𝑥2)⊤(𝑥1 −𝑥2) = γ

𝑥2
1 −2α(𝑧)𝑥1 −𝑥2
2 −2α(𝑧)𝑥2 −(2𝑥1 −2α(𝑧)) · (𝑥1 −𝑥2)


= γ

𝑥2
1 + 𝑥2
2 −2𝑥1 · 𝑥2


= γ (𝑥1 −𝑥2)2

> 0
𝑥1 ≠𝑥2

We showed then, that 𝑣1 −𝑣2 + 𝑔⊤
2 (𝑢2 −𝑢1) = 0, implies 𝑔1 = 𝑔2. We obtain, by Lemma 7, that there
exists a function ¯𝑓((𝑤, 𝑥), 𝑧) such that for all 𝑡and S:

¯𝑓(𝑤S
𝑡, 𝑥S
𝑡, 𝑧) = 𝑓(𝑤S
𝑡, 𝑧) + ℎ𝑧(𝑥S
𝑡)
(23)

and for all 𝑡≤𝑇,

∇¯𝑓((𝑤S
𝑡, 𝑥S
𝑡)) = (OS,𝑡,𝑧, ∇ℎ𝑧(𝑥S
𝑡)).
(24)

This proves Lemma 9. Indeed. By the Lipschitzness of ℎin the unit ball, we have that |𝑥𝑡| ≤γη𝑇. For
sufficiently small γ, from Eq. (23), since {( 𝑓(0, 𝑧) + ℎ𝑧(0), (0, ∇ℎ𝑧(0)), (0, 0))} ∈𝐺(𝑧), we obtain
that

| ¯𝑓((0, 0), 𝑧) −𝑓(0, 𝑧)| = γ|ℎ𝑧(0)| ≤γ2η𝑇≤ε.

| ¯𝑓((𝑤S
𝑞, 𝑥S
𝑞), 𝑧) −𝑓((𝑤S
𝑞, 𝑧))| = γ|ℎ𝑧(𝑥S
𝑞)| ≤ε.

16


---Page Break---
Further, if we assume by induction that for every 𝑡′ ≤𝑡−1, 𝑢𝑡′ = 𝑤S
𝑡′, and 𝑥𝑡′ = 𝑥S
𝑡′, then from Eq. (24)
we have that for any first order oracle:

(𝑢𝑡, 𝑥𝑡) = (𝑤S
𝑡−1, 𝑥S
𝑡−1) −
η
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
O𝑧(𝑤S
𝑡−1, 𝑥S
𝑡−1)

= (𝑤S
𝑡−1, 𝑥S
𝑡−1) −
η
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡


OS,𝑡,𝑧, ∇ℎ𝑧(𝑥S
𝑡−1, 𝑧)


= (𝑤S
𝑡−1, 𝑥S
𝑡−1) −
η
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡


O(𝑡) (𝑆𝑡−1, 𝑤S
𝑡−1, 𝑧), ∇ℎ𝑧(𝑥S
𝑡−1, 𝑧)


= (𝑤S
𝑡−1, 𝑥S
𝑡−1) −η

 

O(S, 𝑤S
𝑡−1), 1

|𝑆𝑡|

𝑇
∑︁

𝑡=1
∇ℎ𝑧(𝑥S
𝑡−1, 𝑧)

!

=

𝑤S
𝑡−1 −η𝑂(S, 𝑤S
𝑡−1), 𝑥S
𝑡−1 −
η
|𝑆𝑡|

∑︁
∇ℎ𝑧(𝑥S
𝑡−1)


= (𝑤S
𝑡, 𝑥S
𝑡),
𝐸𝑞. (22)

which proves, by linearity, that 𝑢𝑞= 𝑤S
𝑞.

B
Proof of Lemma 7

We choose

ˆ𝑓(𝑤) = max
𝑗∈𝐽{ 𝑓𝑗+ 𝑔⊤
𝑗(𝑤−𝑤𝑗}.

ˆ𝑓is indeed convex as it is the maximum of linear functions. Further, it is known [19] that at each
point 𝑤:

𝜕ˆ𝑓(𝑤) = conv{𝑔𝑗: ˆ𝑓(𝑤) = 𝑓𝑗+ 𝑔⊤
𝑗(𝑤−𝑤𝑗)}.
(25)

It follows that, ˆ𝑓is 𝐿-Lipschitz. Next, for any 𝑤𝑖, notice that our assumption implies:

𝑓𝑖≥max
𝑗∈𝐽{ 𝑓𝑗+ 𝑔⊤
𝑗(𝑤𝑖−𝑤𝑗} = ˆ𝑓(𝑤𝑖).

On the other hand,

𝑓𝑖= 𝑓𝑖+ 𝑔⊤
𝑖(𝑤𝑖−𝑤𝑖) ≤ˆ𝑓(𝑤𝑖).

Hence ˆ𝑓(𝑤𝑖) = 𝑓𝑖. Finally, to see the function is differentiable at designated points, take any 𝑖∈𝐼𝑑𝑖𝑓𝑓
and consider 𝑤𝑖. By Eq. (25), it is enough to show that if 𝑔𝑗∈𝜕𝑓(𝑤𝑖) then 𝑔𝑖= 𝑔𝑗, but this clearly
follows from our assumption, and the fact that ˆ𝑓(𝑤𝑖) = 𝑓𝑖.

C
Proof of Claim 1

We first prove by induction that 𝑥S
0 = 0 and, for 𝑡≥1:

𝑥S
𝑡= γη
∑︁

𝑧∈𝑍
α(𝑧) ©­
«

∑︁

{𝑡′≤𝑡,𝑧∈𝑆𝑡′ }

(1 −γη)𝑡−𝑡′

|𝑆𝑡′|
ª®
¬
.
(26)

17


---Page Break---
Indeed,

𝑥S
𝑡= 𝑥S
𝑡−1 −γη

η𝑥S
𝑡−1 −α(S𝑡)


= (1 −γη)𝑥S
𝑡−1 + γηα(S𝑡)

= (1 −γη) ©­
«

∑︁

𝑧∈𝑍
α(𝑧)
∑︁

{𝑡′<𝑡:𝑧∈𝑆𝑡′ }

(1 −γη)𝑡−1−𝑡′

|𝑆𝑡′|
ηγª®
¬
+
1
|𝑆𝑡|

∑︁

𝑧∈𝑆𝑡
γηα(𝑧)

=
∑︁

𝑧∈𝑍
α(𝑧)
∑︁

{𝑡′<𝑡:𝑧∈𝑆𝑡′ }

(1 −γη)𝑡−𝑡′

|𝑆𝑡′|
ηγ + 1[𝑧∈S𝑡]

|𝑆𝑡|
γη

= ηγ
∑︁

𝑧∈𝑍
α(𝑧)
∑︁

{𝑡′≤𝑡:𝑧∈𝑆𝑡′ }

(1 −γη)𝑡−𝑡′

|𝑆𝑡′|

Now, for every S, 𝑡, 𝑧, define a polynomial:

𝑃S,𝑡,𝑧(𝑋) =

𝑡−1
∑︁

𝑛=0

1[𝑧∈𝑆𝑡−𝑛]

|𝑆𝑡−𝑛|
𝑋𝑛,

and let 𝑟be a rational point, sufficinetly small, so that 1 −𝑟is not the root of any polynomial of the
form 𝑃S,𝑡,𝑧−𝑃S′,𝑡′,𝑧′ that is distinct from 0. In other words, we choose 𝑟so that

𝑃𝑆,𝑡,𝑧(1 −𝑟) = 𝑃𝑆′,𝑡′,𝑧′ (1 −𝑟) ⇔𝑃𝑆,𝑡,𝑧(𝑋) = 𝑃𝑆′,𝑡′,𝑧′ (𝑋).

and we also require that

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑃𝑆,𝑡,𝑧(1 −𝑟) = 𝑃𝑆′,𝑡′,𝑧(1 −𝑟) ⇔

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑃𝑆,𝑡,𝑧(𝑋) = 𝑃𝑆′,𝑡′,𝑧′ (𝑋).

Notice that there are only finitely many polynomials of the above form, hence we can choose such 𝑟in
any interval (0, ε) for any ε > 0. Rewriting Eq. (26) we have:

𝑥S
𝑡= γη
∑︁

𝑧∈𝑍
α(𝑧)𝑃S,𝑡,𝑧(1 −γη)

Now, suppose we choose {α(𝑧)} to be reals, independent over the rationals, and suppose we choose
γ = 𝑟/η.

Proof of Item 1
Assume that 𝑥S
𝑡= 𝑥S′
𝑡′ . Because α(𝑧) are independent over the rationals, and
because 𝑃(1 −γη) are always rationals, we have that 𝑃S′,𝑡′,𝑧(1 −𝑟) = 𝑃S′,𝑡′,𝑧(1 −𝑟) for every 𝑧. But
then,
∀𝑧∈𝑍: 𝑃S,𝑡,𝑧(𝑋) = 𝑃S′,𝑡′,𝑧(𝑋),

by choice of 𝑟. But then, 𝑡= 𝑡′. Indeed, assume by contradiction and w.l.og assume 𝑡< 𝑡′: then
for any 𝑧∈𝑆′
1 𝑃S,𝑡,𝑧is a 𝑡′ −1-degree polynomial, on the other hand 𝑃S,𝑡,𝑧is at most of degree
𝑡−1 < 𝑡′ −1. Next, if 𝑆𝑖≠𝑆′
𝑖, for 𝑖≤𝑡, then we can assume (w.l.o.g) that there is 𝑧∈𝑆𝑖such that
𝑧∉𝑆′
𝑖it follows that, by looking at the coefficient of the two polymials of the monomial 𝑋𝑡−𝑖we have
that:
𝑃S′,𝑡′,𝑧(𝑋) ≠𝑃S,𝑡,𝑧(𝑋).

Overall then, we obtain that if 𝑥S
𝑡= 𝑥S′
𝑡′ then (S, 𝑡) ≡(S′, 𝑡′)..

Proof of Item 2:
The proof of Item 2 is similar to how we proved 𝑡= 𝑡′. Notice that:

∑︁

𝑡=1
𝑞(𝑡)𝑥S
𝑡= γη
∑︁

𝑧∈𝑍
α(𝑧)

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑃S,𝑡,𝑧(1 −ηγ).

18


---Page Break---
Hence 𝑥S
𝑡= 𝑥S′ implies for all 𝑧∈𝑍:

𝑇
∑︁

𝑡=1
𝑞(𝑡)𝑃S′,𝑡,𝑧= 𝑃S,𝑡,𝑧.

For 𝑧∈𝑆′
1 we have that P𝑇
𝑡=1 𝑞(𝑡)𝑃S,𝑡,𝑧is a max{𝑡: 𝑞(𝑡) ≠0}-degree polynomial, but on the other
hand we assume that 𝑡< max{𝑡: 𝑞(𝑡) ≠0}, hence 𝑃S,𝑡,𝑧is a lower degree polynomial.

D
Proof of Eqs. (16) and (17)

D.1
Proof of Eq. (16)

To avoid cumbersome notations, we will supress dependence on S and write 𝑤𝑡instead of 𝑤S
𝑡.

As a first step, observe that at each iteration no projection is performed. Indeed, let us show by
induction that:
∥𝑤𝑡+1∥2 = ∥𝑤𝑡−ηO(𝑡) (𝑆, 𝑤,𝑉)∥2 ≤2η2α2(𝑡+ 1).
The definition of α then implies that 𝑤𝑡are restricted to the unit ball without projections.

To see the above is true, let us consider the case where the first type of update is performed:

∥𝑤𝑡+1∥2 = ∥𝑤S
𝑡−ηO(𝑡) (𝑆, 𝑤,𝑉)∥2

= ∥𝑤S
𝑡−ηα𝑒𝐼𝑗+1 + ηα𝑒𝐼𝑗∥2

=

𝑗−1
∑︁

𝑠=1
(𝑤𝑡(𝐼𝑠))2 + (𝑤𝑡(𝐼𝑗) + ηα)2 + (𝑤𝑡(𝐼𝑗+1) −ηα)2 +

⌊𝑑′/𝑘⌋
∑︁

𝑠= 𝑗+2
(𝑤𝑡(𝐼𝑠))2

=

⌊𝑑′/𝑘⌋
∑︁

𝑠=1
(𝑤𝑡(𝐼𝑠))2 + 2ηα(𝑤𝑡(𝐼𝑗) −𝑤𝑡(𝐼𝑗+1)) + 2η2α2

=

⌊𝑑′/𝑘⌋
∑︁

𝑠=1
(𝑤𝑡(𝐼𝑠))2 + 2η2α2
𝑤𝑡(𝐼𝑗+1) = 𝑤𝑡(𝐼𝑗)

≤2η2α2 · 𝑡+ η2α2

= 2η2α2 · (𝑡+ 1).
And if the second type of update is performed:

∥𝑤𝑡+1∥2 = ∥𝑤S
𝑡−ηO(𝑡) (𝑆, 𝑤,𝑉)∥2

= ∥𝑤S
𝑡+ ηα𝑒𝐼𝑗∥2

=
∑︁

𝑠≠𝑗
𝑤𝑡(𝐼𝑠) + η2α2
𝑤𝑡(𝐼𝑗) = 0

≤2η2α2 · 𝑡+ η2α2

≤2η2α2 · (𝑡+ 1)
We now move on to prove that Eq. (16) holds by induction. Specifically, we will show that for
𝑑0 ≤⌊𝑑′/𝑘⌋that at time 𝑇𝑑0 = 1 + P𝑑0
𝑑=0
P𝑑
𝑘=0 𝑘, for any 𝑡≤𝑑0:

𝑤𝑇𝑑0 (𝐼𝑡) = αη
(𝑑0 + 1 −𝑡)
𝑡≤𝑑0
0
o.w.
.
(27)

Eq. (16) then follows by plugging 𝑑0 = 𝑑′/𝑘, and noting that at every step we have that if 𝑖∈𝐼𝑡then
𝑤𝑇𝑑0 (𝑖) =
√

𝑘𝑤𝑇𝑑0 (𝐼𝑡). Therefore, we are left with proving Eq. (27).

For 𝑑0 = 0, 𝑇0 = 1 and we have, indeed, 𝑤1 = 0. Next assume we proved the statement for 𝑑0, and we
will prove it for 𝑑0 + 1. Here too, we will use induction, and we prove that, for 𝑑1 ≤𝑑0 + 1, at time

𝑇𝑑0,𝑑1 = 𝑇𝑑0 +

𝑑1−1
∑︁

𝑘=0
(𝑑0 + 1 −𝑘),

19


---Page Break---
we have that:

𝑤𝑇𝑑0,𝑑1 (𝐼𝑡) = αη




(𝑑0 + 2 −𝑡)
𝑡≤𝑑1
(𝑑0 + 1 −𝑡)
𝑑1 < 𝑡≤𝑑0
0
o.w.

For the case 𝑑1=0, 𝑇𝑑0,𝑑1 = 𝑑0, and it follows from our outer-induction step. Assume the statement is
true for 𝑑1 and we will prove it for 𝑑1 + 1, here, yet again, we use induction. And we will show that
for 1 ≤𝑑2 < 𝑑0 + 2 −𝑑1 we have at time

𝑇𝑑0,𝑑1,𝑑2 = 𝑇𝑑0 + 𝑇𝑑1 + 𝑑2,

we have that:

𝑤𝑇𝑑0,𝑑1,𝑑2 (𝐼𝑡) = αη





(𝑑0 + 2 −𝑡)
𝑡≤𝑑1
(𝑑0 + 1 −𝑡)
𝑑1 < 𝑡< 𝑑0 + 2 −𝑑2
(𝑑0 + 2 −𝑡)
𝑡= 𝑑0 + 2 −𝑑2
(𝑑0 + 1 −𝑡)
𝑑0 + 2 −𝑑2 < 𝑡≤𝑑0
0
o.w.

(28)

We start the induction with the case 𝑑2 = 1, in that case notice that 𝑇𝑑0,𝑑1,𝑑2 = 𝑇𝑑0,𝑑1 + 1, and by
induction hypothesis:

𝑤𝑇𝑑0,𝑑1 (𝐼𝑡) = αη




(𝑑0 + 2 −𝑡)
𝑡≤𝑑1
(𝑑0 + 1 −𝑡)
𝑑1 < 𝑡≤𝑑0
0
o.w.

In this case, note that there are no two consecutive coordinates that are equal, hence our choice of
oracle is defined so that O(𝑡) = −𝑒𝐼𝑑0+1. Hence, by our update rule (and the lack of projections which
we proved at the beginning):

𝑤𝑇𝑑0,𝑑1,1(𝐼𝑡) = αη





(𝑑0 + 2 −𝑡)
𝑡≤𝑑1
(𝑑0 + 1 −𝑡)
𝑑0 < 𝑡≤𝑑0
1
𝑡= 𝑑0 + 1
0
o.w.

Which satisfies Eq. (28). Now assume that Eq. (28) holds for 𝑑2, and take 𝑑2 + 1 < 𝑑0 + 2 −𝑑1
(otherwise, we are done). Notice that 𝑇𝑑0,𝑑1,𝑑2+1 = 𝑇𝑑0,𝑑1,𝑑2 + 1. Observe that 𝑤𝑇0,𝑑1,𝑑2(𝑑0 +
2 −𝑑2) = 𝑤𝑇0,𝑑1,𝑑2(𝑑0 + 1 −𝑑2) (notice that 𝑑0 + 1 −𝑑2 > 𝑑1), and our update rule is such that
O(𝑡) = 𝑒𝐼𝑑0+2−𝑑2 −𝑒𝐼𝑑0+1−𝑑2 and we obtain then:

𝑤𝑇𝑑0,𝑑1,𝑑2+1 = 𝑤𝑇𝑑0,𝑑1,𝑑2 −ηα𝑒𝐼𝑑0+2−𝑑2 + ηα𝑒𝐼𝑑0+1−𝑑2 = αη





(𝑑0 + 2 −𝑡)
𝑡≤𝑑1
(𝑑0 + 1 −𝑡)
𝑑1 < 𝑡< 𝑑0 + 2 −(𝑑2 + 1)
(𝑑0 + 2 −𝑡)
𝑡= 𝑑0 + 2 −(𝑑2 + 1)
(𝑑0 + 1 −𝑡)
𝑑0 + 2 −(𝑑2 + 1) < 𝑡≤𝑑0
0
o.w.

The most inner induction step is now complete. We now notice that 𝑇𝑑0,𝑑1,𝑑0+1−𝑑1 = 𝑇𝑑0,𝑑1+1, which
proves the middle-induction step. And we notice that 𝑇𝑑0,𝑑0+1 = 𝑇𝑑0+1, which proves the whole
induction argument.

D.2
Proof of Eq. (17)

We only need to show that the following quantity is increasing in 𝑡

𝑋𝑡= max

(∑︁

𝑖∈𝐼𝐵
𝑤S
𝑡(𝑖) : 𝐼𝐵⊆𝐼, |𝐼| = 𝐵

)

.

20


---Page Break---
But, as shown in Appendix D, the update rule is such that we don’t perform projections. It then
follows easily from our update step. Indeed if we increase a set of coordinates by αη, then clearly the
𝑋𝑡only increases. Also, if we perform update of the form:

𝑤S
𝑡(𝐼𝑗) = 𝑤S
𝑡−1(𝐼𝑗) + αη,
𝑤S
𝑡(𝐼𝑗+1) = 𝑤S
𝑡(𝐼𝑗+1) −αη,

for two consecutive and equal coordinates, then for any 𝐼𝐵: If 𝐼𝐵includes same number of coordinates
from 𝐼𝑗as in 𝐼𝑗+1 then the magnitude doesn’t change. If 𝐼𝐵contains more 𝐼𝑗then it increases, and if
𝐼𝐵contains more from 𝑖𝑗+1 then consider 𝐼𝐵′ that swaps coordinates in 𝐼𝑗with 𝐼𝑗+1 then we clearly
have:

∑︁

𝑖∈𝐼𝐵′
𝑤S
𝑡(𝑖) >
∑︁

𝑖∈𝐼𝐵′
𝑤S
𝑡−1(𝑖) >
∑︁

𝑖∈𝐼𝐵
𝑤S
𝑡(𝑖)

E
Dimension independent lower bound for GD

In this section we prove that the optimization error of GD in Eq. (5) is optimal. The lower bound is an
optimization error for first order methods and is well established (see [10]). The point here is to show
that the bound is valid in any dimension, for GD.

Claim 2. For every choice of η,𝑇, there exists a convex and 1 Lipschitz function 𝑓(𝑥) : ℝ→ℝsuch
that, if we run GD on 𝑓, then

𝑓(𝑤𝐺𝐷) −𝑓(0) ≥η

2 +
1
6η𝑇.

Proof. We divide the proof into two cases:

case1: η ≥1/η𝑇
:

In this case we choose 𝑓(𝑥) = |𝑥−γ|, where γ > 0 is a arbitrarily small (may depend on η). It can be
seen that for every even iteration, we have that:

∇𝑓(𝑥2𝑡) = −1,

at at every odd iteration
∇𝑓(𝑥2𝑡+1) = +1.

As such, for every even iteration we have that 𝑥2𝑡= 0, and at every odd iteration we have that 𝑥2𝑡+1 = η.
We thus have that 𝑥𝐺𝐷= η

2 and

𝑓(𝑥𝐺𝐷) −0 = η

2 −0 ≥η

4 +
1
2η𝑇.

case1: η ≤1/η𝑇≤1
: In this case choose 𝑓(𝑥) = α𝑥, where α =
1
(𝑇+1)η ≤1. Then, one can show
that GD outputs

𝑥𝐺𝐷= −η

𝑇

∑︁
𝑡· α = −(𝑇+ 1)η

2
α,

and:

𝑓(𝑥𝐺𝐷) −𝑓(−1) = α −(𝑇+ 1)η

2
α2 ≥
1
2(𝑇+ 1)η ≥
1
3𝑇η ≥
1
6𝑇η + η

2 .

■

21


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification:

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made
in the paper.
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

Justification:

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
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important
role in developing norms that preserve the integrity of the community. Reviewers will
be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

22


---Page Break---
Justification:

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

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well
by the reviewers: Making the paper reproducible is important, regardless of whether
the code and data are provided or not.
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
• While NeurIPS does not require releasing code, the conference does require all
submissions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
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

Question: Does the paper provide open access to the data and code, with sufficient instructions
to faithfully reproduce the main experimental results, as described in supplemental material?

23


---Page Break---
Answer: [NA]
Justification:
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run
to reproduce the results.
See the NeurIPS code and data submission guidelines
(https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
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
Justification:
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
Justification:
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the
main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of
the mean.

24


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

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the
experiments?
Answer: [NA]
Justification:
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
Justification:
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special
consideration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification:
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

25


---Page Break---
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
Justification:
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
Justification:
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
• If this information is not available online, the authors are encouraged to reach out to the
asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

26


---Page Break---
Answer: [NA]
Justification:
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
Justification:
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main
contribution of the paper involves human subjects, then as much detail as possible
should be included in the main paper.
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
Justification:
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

27


---Page Break---
