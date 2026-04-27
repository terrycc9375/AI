Universal Online Convex Optimization
with 1 Projection per Round

Wenhao Yang1,2, Yibo Wang1,2, Peng Zhao1,2, Lijun Zhang1,2,∗

1National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China
2School of Artificial Intelligence, Nanjing University, Nanjing, China
{yangwh, wangyb, zhaop, zhanglj}@lamda.nju.edu.cn

Abstract

To address the uncertainty in function types, recent progress in online convex
optimization (OCO) has spurred the development of universal algorithms that
simultaneously attain minimax rates for multiple types of convex functions. How-
ever, for a T-round online problem, state-of-the-art methods typically conduct
O(log T) projections onto the domain in each round, a process potentially time-
consuming with complicated feasible sets. In this paper, inspired by the black-box
reduction of Cutkosky and Orabona [2018], we employ a surrogate loss defined
over simpler domains to develop universal OCO algorithms that only require 1
projection. Embracing the framework of prediction with expert advice, we maintain
a set of experts for each type of functions and aggregate their predictions via a
meta-algorithm. The crux of our approach lies in a uniquely designed expert-loss
for strongly convex functions, stemming from an innovative decomposition of the
regret into the meta-regret and the expert-regret. Our analysis sheds new light on
the surrogate loss, facilitating a rigorous examination of the discrepancy between
the regret of the original loss and that of the surrogate loss, and carefully controlling
meta-regret under the strong convexity condition. With only 1 projection per round,
we establish optimal regret bounds for general convex, exponentially concave, and
strongly convex functions simultaneously. Furthermore, we enhance the expert-loss
to exploit the smoothness property, and demonstrate that our algorithm can attain
small-loss regret for multiple types of convex and smooth functions.

1
Introduction

Online convex optimization (OCO) stands as a pivotal online learning framework for modeling many
real-world problems [Hazan, 2016]. OCO is commonly formulated as a repeated game between the
learner and the environment with the following protocol. In each round t ∈[T], the learner chooses a
decision xt from a convex domain X ⊆Rd; after submitting this decision, the learner suffers a loss
ft(xt), where ft : X 7→R is a convex function selected by the environment. The goal of the learner
is to minimize the cumulative loss over T rounds, i.e., PT
t=1 ft(xt), and the standard performance
measure is the regret [Cesa-Bianchi and Lugosi, 2006]:

REGT =

T
X

t=1
ft(xt) −min
x∈X

T
X

t=1
ft(x),
(1)

which quantifies the difference between the cumulative loss of the online learner and that of the best
decision chosen in hindsight.

∗Lijun Zhang is the corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: A summary of our universal algorithms and previous studies over T rounds d-dimensional
functions, where LT denotes the small-loss quantity. Abbreviations: cvx →convex, exp-concave →
exponentially concave, str-cvx →strongly convex, # PROJ →number of projections per round.

Assumption
Method
Regret Bounds
# PROJ
cvx
exp-concave
str-cvx

van Erven and Koolen [2016]
O(
√

T)
O(d log T)
O(d log T)
O(log T)
Mhammedi et al. [2019]
O(
√

T)
O(d log T)
O(d log T)
1
Wang et al. [2019]
O(
√

T)
O(d log T)
O(log T)
O(log T)
Zhang et al. [2022]
O(
√

T)
O(d log T)
O(log T)
O(log T)
Theorem 1 of this work
O(
√

T)
O(d log T)
O(log T)
1

ft(·) is smooth

Wang et al. [2020b]
O(√LT )
O(d log LT )
O(log LT )
O(log T)
Zhang et al. [2022]
O(√LT )
O(d log LT )
O(log LT )
O(log T)
Theorem 2 of this work
O(√LT )
O(d log LT )
O(log LT )
1

Although there are plenty of algorithms to minimize the regret of convex functions, including general
convex, exponentially concave (abbr. exp-concave) and strongly convex functions [Zinkevich, 2003,
Shalev-Shwartz et al., 2007, Hazan et al., 2007], most of them can only handle one specific function
type, and need to estimate the moduli of strong convexity and exp-concavity. The demand for
prior knowledge motivates the development of universal algorithms for OCO, which aim to attain
minimax optimal regret guarantees for multiple types of convex functions simultaneously [Bartlett
et al., 2008, van Erven and Koolen, 2016, Wang et al., 2019, Mhammedi et al., 2019, Zhang et al.,
2022]. State-of-the-art methods typically adopt a two-layer structure following the prediction with
expert advice (PEA) framework [Cesa-Bianchi and Lugosi, 2006]. Specifically, they maintain
O(log T) expert-algorithms with different configurations to handle the uncertainty of functions and
deploy a meta-algorithm to track the best one. While this two-layer framework has demonstrated
effectiveness in endowing algorithms with universality, it raises concerns regarding the computational
efficiency. Since each expert-algorithm needs to execute one projection onto the feasible domain X
per round, standard universal algorithms perform O(log T) projections in each round, which can be
time-consuming in practical scenarios particularly when projecting onto complicated domains.

In the literature, there indeed exists an effort to reduce the number of projections required by universal
algorithms tailored for exp-concave functions [Mhammedi et al., 2019]. This is achieved by applying
the black-box reduction of Cutkosky and Orabona [2018], which reduces an OCO problem on the
original (but can be complicated) feasible domain to a more manageable one on a simpler domain,
such as an Euclidean ball. Deploying an existing universal algorithm [van Erven and Koolen, 2016]
on the reduced problem enables us to attain optimal regret for exp-concave functions, crucially, with
only one single projection per round and no prior knowledge of exp-concavity required. However, this
black-box approach cannot be extended to strongly convex functions (see Section 3.1 for technical
discussions). Therefore, it is still unclear on how to reduce the number of projections of universal
algorithms to 1, and at the same time ensure optimal regret for strongly convex functions (as well as
general convex and exp-concave functions).

In this paper, we affirmatively solve the above question by introducing an efficient universal OCO
algorithm. Our solution employs the black-box reduction Cutkosky [2020] to cast the original
problem on the constrained domain X to an alternative one in terms of the surrogate loss on a simpler
domain Y ⊇X. Specifically, we construct multiple experts updated in domain Y, each optimizing a
expert-loss specialized for a distinct function type. Then, we combine their predictions by a meta-
algorithm, and perform the only projection onto the feasible domain X. The meta-algorithm chooses
the linearized surrogate loss to measure the performance of experts, and is required to yield a second-
order regret [Zhang et al., 2022]. The key novelty of our algorithm lies in the uniquely designed
expert-loss for strongly convex functions, which is motivated by an innovative decomposition of the
regret into the meta-regret and the expert-regret. To effectively deal with strongly convex functions,
we explore the domain-converting surrogate loss in depth and illuminate its refined properties. Our
new insights tighten the regret gap in terms of original loss and surrogate loss, and further exploit
strong convexity to compensate the meta-regret, thus achieving the optimal regret for strongly convex
functions. Section 3.2 provides a formal description of our key ideas. With only 1 projection

2


---Page Break---
per round, our algorithm attains O(
√

T), O( d

α log T), and O( 1

λ log T) regret for general convex,
α-exp-concave, and λ-strongly convex functions, respectively.

We further establish small-loss regret for universal OCO with smooth functions. The small-loss
quantity LT = minx∈X
PT
t=1 ft(x) is defined as the cumulative loss of the best decision chosen
from the domain X, which is at most O(T) under standard OCO assumptions and meanwhile can be
much smaller in benign environments. To achieve small-loss regret bounds, we design an enhanced
expert-loss for smooth and strongly convex functions and integrate it into our two-layer framework,
which finally leads to a universal OCO algorithm achieving O(√LT ), O( d

α log LT ), and O( 1

λ log LT )
small-loss regret for three types of convex functions, respectively. Notably, all those bounds are
optimal and the algorithm only requires one projection per iteration. We summarize our results and
compare with previous studies of universal algorithms in Table 1.

Organization.
The rest of the paper is organized as follows. Section 2 presents the preliminaries
and reviews several mostly related works. Section 3 illuminates the technical challenges and describes
our key ideas. Section 4 provides the overall algorithms and regret analysis. We finally conclude the
paper in Section 5. All the proofs and omitted details are deferred to appendices.

2
Preliminaries and related works

In this section, we first present preliminaries for OCO, and then review several most related works to
our paper, including universal algorithms and projection-efficient algorithms.

2.1
Preliminaries

We introduce two typical assumptions of online convex optimization [Hazan, 2016].

Assumption 1 (bounded domain) The feasible domain X ⊆Rd contains the origin 0, and the
diameter is bounded by D, i.e., ∥x −y∥≤D holds for any x, y ∈X.

Assumption 2 (bounded gradient norms) The norm of the gradients of all online functions over
the domain X is bounded by G, i.e., ∥∇ft(x)∥≤G holds for all x ∈X and t ∈[T].

Throughout the paper we use ∥· ∥for ℓ2-norm in default. Owing to Assumption 1, we can always
construct an Euclidean ball Y = {x | ∥x∥≤D} containing the original feasible domain X.

Next, we state definitions of strong convexity and exp-concavity [Hazan, 2016], and introduce an
important property of exp-concave functions [Hazan et al., 2007, Lemma 3].

Definition 1 (strongly convex functions) A function f : X 7→R is called λ-strongly convex, if the
condition f(y) ≥f(x) + ⟨∇f(x), y −x⟩+ λ

2 ∥y −x∥2 holds for all x, y ∈X.

Definition 2 (exponentially-concave functions) A function f : X 7→R is called α-exponentially-
concave, if the function exp(−αf(·)) is concave over the feasible domain X.

Lemma 1 For an α-exp-concave function f : X 7→R, if the feasible domain X has a diameter D
and ∥∇f(x)∥≤G holds for ∀x ∈X, then we have

f(y) ≥f(x) + ⟨∇f(x), y −x⟩+ β

2 ⟨∇f(x), y −x⟩2,
(2)

for all x, y ∈X, where β = 1

2 min{
1
4GD, α}.

There are many efforts devoted to minimizing regret, including general convex, α-exp-concave, and
λ-strongly convex functions. For general convex functions, online gradient descent (OGD) with
step size ηt = O(1/
√

t), attains an O(
√

T) regret [Zinkevich, 2003]. For α-exp-concave functions,
online Newton step (ONS) is equipped with an O( d

α log T) regret [Hazan et al., 2007]. For λ-strongly
convex functions, OGD with step size ηt = O(1/[λt]), achieves an O( 1

λ log T) regret [Shalev-
Shwartz et al., 2007]. These regret bounds are proved to be minimax optimal [Ordentlich and Cover,
1998, Abernethy et al., 2008]. Furthermore, tighter bounds are attainable when the loss functions

3


---Page Break---
enjoy additional properties, such as smoothness [Shalev-Shwartz, 2007, Luo and Schapire, 2015,
Srebro et al., 2010, Orabona et al., 2012, Chiang et al., 2012, Yang et al., 2014, Mohri and Yang,
2016, Zhang et al., 2019, Zhao et al., 2020, 2024, Chen et al., 2024] and sparsity of gradients [Duchi
et al., 2010, Tieleman and Hinton, 2012, Mukkamala and Hein, 2017, Kingma and Ba, 2015, Reddi
et al., 2018, Loshchilov and Hutter, 2019, Wang et al., 2020a]. We discuss small-loss regret below.

For general convex and smooth functions, Srebro et al. [2010] prove that OGD with constant step size
attains an O(
√

L) regret bound, where L is the upper bound of LT . The limitation of their method
is that it requires to know L beforehand. To address this limitation, Zhang et al. [2019] propose
scale-free online gradient descent (SOGD), which is a special case of scale-free mirror descent
algorithm [Orabona and Pál, 2018], and establish an O(√LT ) small-loss regret bound without the
prior knowledge of LT . For α-exp-concave and smooth functions, ONS attains an O( d

α log LT )
small-loss regret bound [Orabona et al., 2012]. For λ-strongly convex and smooth functions, a variant
of OGD, namely S2OGD, is introduced to achieve an O( 1

λ log LT ) small-loss regret bound [Wang
et al., 2020b]. Such bounds reduce to the minimax optimal bounds in the worst case, but could be
much tighter when the comparator has a small loss, i.e., LT is small.

2.2
Universal algorithms

Most existing online algorithms can only handle one type of convex function and need to know
the moduli of strong convexity and exp-concavity beforehand. Universal online learning aims to
remove such requirements of domain knowledge. The first universal OCO algorithm is adaptive
online gradient descent (AOGD) [Bartlett et al., 2008], which achieves O(
√

T) and O(log T) regret
bounds for general convex and strongly convex functions, respectively. However, the algorithm still
needs to know the modulus of strong convexity and does not support exp-concave functions.

An important milestone is the multiple eta gradient (MetaGrad) algorithm [van Erven and Koolen,
2016], which adapt to general convex and exp-concave functions without knowing the modulus of exp-
concavity. MetaGrad constructs multiple expert-algorithms with various learning rates and combines
their predictions by a meta-algorithm called Tilted Exponentially Weighted Average (TEWA). To
avoid prior knowledge, each expert minimizes the expert-loss parameterized by a learning rate η,

ℓexp
t,η(x) = −η⟨∇ft(xt), xt −x⟩+ η2⟨∇ft(xt), xt −x⟩2.
(3)

MetaGrad maintains O(log T) experts to minimize (3), and attains O(√T log log T) and O( d

α log T)
regret for general convex and α-exp-concave functions, respectively. To further support strongly
convex functions, Wang et al. [2019] propose a new type of expert-losses defined as

ℓsc
t,η(x) = −η⟨∇ft(xt), xt −x⟩+ η2G2∥xt −x∥2
(4)

where G is the gradient norm upper bound, and introduce an expert-loss for general convex functions

ℓcvx
t,η(x) = −η⟨∇ft(xt), xt −x⟩+ η2G2D2
(5)

where D is the upper bound of the diameter of X. Their algorithm, named as Maler, obtains
O(
√

T), O( 1

λ log T) and O( d

α log T) regret for general convex, λ-strongly convex functions, and
α-exp-concave functions, respectively. Later, Wang et al. [2020b] extend Maler by replacing G2
in (4) and (5) with ∥∇ft(xt)∥2, thereby enabling their algorithm to deliver small-loss regret bounds.
Under the smoothness condition, their algorithm achieves O(√LT ), O( 1

λ log LT ) and O( d

α log LT )
regret for general convex, λ-strongly convex, and α-exp-concave functions, respectively.

MetaGrad and its variants require the carefully designed expert-losses. Zhang et al. [2022] propose
a different universal strategy that avoids the construction of losses. The basic idea is to let each
expert handle original functions and deploy a meta-algorithm over linearized loss. Importantly, the
meta-algorithm is required to yield a second-order regret [Gaillard et al., 2014] to exploit strong
convexity and exp-concavity. By incorporating existing online algorithms as experts, their approach
inherits the regret of any expert designed for strongly convex functions and exp-concave functions,
and also obtains minimax optimal regret (and small-loss regret) for general convex functions.

Although state-of-the-art universal algorithms can adapt to multiple function types, they create
O(log T) experts per round. As a result, they need to perform O(log T) projections in each round,
which can be time-consuming in practical scenarios with complicated domains. To address this
limitation, we aim to develop projection-efficient algorithms for universal OCO.

4


---Page Break---
2.3
Projection-efficient algorithms

In the studies of parameter-free online learning, Cutkosky and Orabona [2018] propose a black-box
reduction technique from constrained online learning to unconstrained online learning. To avoid
regret degeneration, they design the domain-converting surrogate loss bgt : Y 7→R defined as,

bgt(y) = ⟨∇ft(xt), y⟩+ ∥∇ft(xt)∥· SX (y)
(6)

where SX (y) = ∥y −ΠX [y]∥is the distance function to the feasible domain X. Then, we can
employ an unconstrained online learning algorithm that minimizes (6) to obtain the prediction yt,
and output its prediction on domain X, i.e., xt = ΠX [yt]. Cutkosky and Orabona [2018, Theorem 3]
have proved that the above surrogate loss satisfies ∥∇bgt(yt)∥≤∥∇ft(xt)∥, and

⟨∇ft(xt), xt −x⟩≤2
 
bgt(yt) −bgt(x)

≤2⟨∇bgt(yt), yt −x⟩
(7)

for all t ∈[T] and any x ∈X. Based on this fact, we know that the regret of the unconstrained
problem directly serves as an upper bound for that of the original problem, hence reducing the original
problem to an unconstrained surrogate problem and retaining the order of regret.

Subsequently, Cutkosky [2020] introduces a new surrogate loss gt : Y 7→R defined as,

gt(y) = ⟨∇ft(xt), y⟩−1{⟨∇ft(xt),vt⟩<0}⟨∇ft(xt), vt⟩· SX (y)
(8)

where vt =
yt−xt
∥yt−xt∥is the unit vector of the projection direction. As depicted in the following
lemma, this surrogate loss avoids the multiplicative constant 2 on the right-hand side of (7).

Lemma 2 (Theorem 2 of Cutkosky [2020]) The function defined in (8) is convex, and it satisfies
∥∇gt(yt)∥≤∥∇ft(xt)∥. Furthermore, for all t and all x ∈X, we have

⟨∇ft(xt), xt −x⟩≤gt(yt) −gt(x) ≤⟨∇gt(yt), yt −x⟩.
(9)

While the black-box reduction is proposed for the constrained-to-unconstrained conversion, it also
facilitates the conversion to another constrained problem (i.e., Y ̸= Rd). This enables us to transform
OCO problem on a complicated domain into another on simpler domains such that the projection is
much easier. Building on this idea, Mhammedi et al. [2019] introduce an efficient implementation
of MetaGrad [van Erven and Koolen, 2016], which only conducts 1 projection onto the original
domain in each round, and keeps the order of regret bounds. However, as detailed in the following
section, the black-box reduction does not adequately extend to strongly convex functions. We also
mention that Zhao et al. [2022] recently employ the technique to non-stationary OCO with non-trivial
modifications to develop efficient algorithms for minimizing dynamic regret and adaptive regret.
However, they focus on the convex functions and do not involve the considerations of exp-concave
and strongly convex functions as concerned in our paper.

3
Technical challenge and our key ideas

In this section, we elaborate on the technical challenges and our key ideas.

3.1
Technical challenge

As mentioned, Mhammedi et al. [2019] exploit the black-box reduction scheme of [Cutkosky and
Orabona, 2018] to improve the projection efficiency of MetaGrad [van Erven and Koolen, 2016]. We
summarize their algorithm in Algorithm 1. In the following, we will demonstrate its effectiveness for
exp-concave functions and explain why it fails for strongly convex functions.

Success in exp-concave functions.
By applying the black-box reduction as described in Section 2.3,
Mhammedi et al. [2019] utilize MetaGrad to minimize the surrogate loss bgt(·) in (6) over an Euclidean
ball Y. The projection operations inside MetaGrad are over Y and thus negligible. Notice that
Algorithm 1 demands only 1 projection onto X in Step 4. According to regret bound of MetaGrad,
Algorithm 1 enjoys a second-order bound [Mhammedi et al., 2019, Theorem 10],

T
X

t=1
⟨∇bgt(yt), yt −x⟩≤O





v
u
u
td log T ·

T
X

t=1
⟨∇bgt(yt), yt −x⟩2 + d log T



.
(10)

5


---Page Break---
Algorithm 1 Black-box reduction for projection-efficient MetaGrad [Mhammedi et al., 2019]

1: Construct a ball domain Y = {x | ∥x∥≤D} ⊇X
2: for t = 1 to T do
3:
Receive the decision yt ∈Y from MetaGrad
4:
Submit the decision xt = ΠX [yt]
▷The only step projects onto domain X per round.
5:
Suffer the loss ft(xt) and observe the gradient ∇ft(xt)
6:
Construct the surrogate loss bgt(·) as (6) and send it to MetaGrad
7: end for

The above bound is measured by surrogate loss, thus requiring a further analysis that converts it
back to that of the original function. Since β = 1

2 min

1
4GD, α
	
, the function x −βx2 is strictly
increasing when x ∈(−∞, 2GD]. Therefore, the property of surrogate loss bgt(·) in (7) implies

1
2⟨∇ft(xt), xt −x⟩−β

4 ⟨∇ft(xt), xt −x⟩2 ≤⟨∇bgt(yt), yt −x⟩−β⟨∇bgt(yt), yt −x⟩2. (11)

Combining (10) with (11) and applying the AM-GM inequality, we obtain

T
X

t=1
⟨∇ft(xt), xt −x⟩−β

2

T
X

t=1
⟨∇ft(xt), xt −x⟩2 ≤O
 d

α log T


thus achieving the optimal regret based on Lemma 1.

Failure in strongly convex functions.
To handle strongly convex functions, a straightforward way
is to use a universal algorithm that supports strongly convex functions, such as Maler [Wang et al.,
2019], as the black-box subroutine in Algorithm 1. However, for strongly convex functions, the above
analysis cannot be applied, and we are unable to derive a tight regret bound. Specifically, according
to the theoretical guarantee of Maler [Wang et al., 2019, Theorem 1], we have

T
X

t=1
⟨∇bgt(yt), yt −x⟩≤O





v
u
u
tlog T ·

T
X

t=1
∥yt −x∥2 + log T



.
(12)

From the standard black-box analysis and the definition of strong convexity, we know

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)
(7)
≤

T
X

t=1
2⟨∇bgt(yt), yt −x⟩−λ

2

T
X

t=1
∥xt −x∥2.
(13)

Substituting (12) into (13), we encounter an eO(
qPT
t=1 ∥yt −x∥2 −λ

2
PT
t=1 ∥xt −x∥2) term,

which is unmanageable since ∥yt −x∥≥∥xt −x∥. Here, eO(·) further omits the ploy(log T) factors.

3.2
Key ideas

To address above challenges, we introduce novel ideas in both algorithm design and regret analysis.

Algorithm design.
Our algorithm is still in a two-layer structure. The main contribution lies in
a uniquely designed expert-loss for strongly convex functions. For simplicity, we consider that the
modulus of strong convexity λ is known for a moment, and define

ℓsc
t (y) = ⟨∇gt(yt), y⟩+ λ

2 ∥y −xt∥2,
(14)

where gt(·) is the surrogate loss defined in (8). Let us compare our designed expert-loss (14) with
the one when applying existing universal algorithms in a black-box manner. Suppose Maler [Wang
et al., 2019] is used, their expert-loss construction (4) indicates that the algorithm over domain Y
essentially optimizes the expert-loss formulated as (up to constant factors).

bℓsc
t (y) = ⟨∇gt(yt), y⟩+ λ

2 ∥y −yt∥2
(15)

6


---Page Break---
An important caveat is that our expert-loss (14) evaluates the performance of the expert (associated
with strongly convex functions) based on the distance between its output y and the actual decision
xt ∈X, as opposed to the unprojected intermediate one yt ∈Y in (15).

In fact, this design of expert-loss (14) stems from a novel regret decomposition as explained below.
First, by strong convexity of ft and the property of the domain-converting surrogate loss, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)
(9)
≤

T
X

t=1
⟨∇gt(yt), yt −x⟩−λ

2

T
X

t=1
∥xt −x∥2

=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩+

T
X

t=1
⟨∇gt(yt), yi
t −x⟩−λ

2

T
X

t=1
∥xt −x∥2
(16)

where yi
t denotes the decision of the i-th expert. The first term of the above bound is the meta-regret
in terms of linearized surrogate loss. Then, we reformulate the remaining two terms as follows

T
X

t=1
⟨∇gt(yt), yi
t −x⟩−λ

2

T
X

t=1
∥xt −x∥2 =

T
X

t=1


⟨∇gt(yt), yi
t⟩+ λ

2 ∥xt −yi
t∥2


−

T
X

t=1


⟨∇gt(yt), x⟩+ λ

2 ∥xt −x∥2

−λ

2

T
X

t=1
∥xt −yi
t∥2,

(17)

where the expert-loss in (14) naturally arises. Combining (16) with (17), we arrive at

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1


ℓsc
t (yi
t) −ℓsc
t (x)


|
{z
}
expert-regret

+

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

−λ

2

T
X

t=1
∥xt −yi
t∥2.

(18)

Theoretical analysis.
For the expert-regret, since expert-loss (14) is λ-strongly convex and its
gradients are bounded (see Lemma 6), we can use OGD to achieve an optimal O( 1

λ log T) regret.
Following Zhang et al. [2022], we require the meta-algorithm to yield a second-order regret bound

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤O





v
u
u
t

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2



.
(19)

Notably, the upper bound of (19) and the negative term in (18) cannot be canceled due to the dismatch
between yt −yi
t and xt −yi
t. To resolve this discrepancy, we demonstrate that the surrogate loss
defined in (8) enjoys the following two important improved properties.

Lemma 3 In addition to enjoying all the properties outlined in Lemma 2, the surrogate loss function
gt : Y 7→R defined in (8) satisfies

⟨∇ft(xt), xt −x⟩≤⟨∇gt(yt), yt −x⟩−1{⟨∇ft(xt),vt⟩≥0} · ⟨∇ft(xt), yt −xt⟩,
(20)

for all t and all x ∈X. Furthermore, we also have
 ⟨∇gt(yt), xt −yt⟩= 0,
when ⟨∇ft(xt), vt⟩< 0,
⟨∇gt(yt), xt −yt⟩≤0,
otherwise.
(21)

Remark 1 We highlight the improvements of Lemma 3 over Lemma 2. First, we provide a tighter
connection between the linearized original function and the surrogate loss in (20). Second, we analyze
the difference between the actual decision xt and the intermediate decision yt, along the direction
∇gt(yt) in (21). As shown later, both of them are crucial for controlling the meta-regret.
◁

Utilizing (20) in Lemma 3, we refine the decomposition in (18) to establish a tighter bound

T
X

t=1
ft(xt)−

T
X

t=1
ft(x)
(16),(17),(20)
≤
ER(T)+

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩−λ

2

T
X

t=1
∥xt −yi
t∥2 −∆T (22)

7


---Page Break---
where ER(T) is the expert-regret, and ∆T = PT
t=1 1{⟨∇ft(xt),vt⟩≥0} ·⟨∇ft(xt), yt −xt⟩≥0 is the
negative term introduced in the surrogate loss. Compared to (18), the new upper bound (22) enjoys an
additional negative term −∆T , which is essential to achieve a favorable regret bound in the analysis.

To utilize the negative quadratic term −λ

2
PT
t=1 ∥xt−yi
t∥2 in (22) for compensating the second-order
bound in (19), we need to convert yt to xt, a place where (21) comes into play. From (19) and (21),
we prove that for any γ ∈(0, G

2D] it holds that (see Lemma 8 for details):

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤O
G2

2γ


+
γ
2G2

T
X

t=1
⟨∇gt(yt), xt −yi
t⟩2 + ∆T .
(23)

Substituting (23) into (22), the additional term ∆T is automatically canceled out, and we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤ER(T) + O
G2

2γ


+
γ
2G2

T
X

t=1
⟨∇gt(yt), xt −yi
t⟩2 −λ

2

T
X

t=1
∥xt −yi
t∥2

≤ER(T) + O
G2

2γ


+
γ

2 −λ

2


T
X

t=1
∥xt −yi
t∥2 = O
 1

λ log T


where the final regret bound is because we set γ = min{ G

2D, λ}.

Remark 2 Section 2.3 describes two kinds of surrogate loss, as specified in (6) and (8). Indeed,
they both are suitable for parameter-free online learning [Cutkosky, 2020] and non-stationary online
learning [Zhao et al., 2022]. However, it is essential to adopt the new surrogate loss in our purpose: as
established in Lemma 3, both negative terms and the mild difference between xt and yt are exploited
in our regret analysis. By contrast, the old surrogate loss (6) lacks these advanced properties.
◁

4
Efficient algorithm for universal online convex optimization

In this section, we present our efficient algorithms for universal OCO. To reduce the cost of projections,
we deploy multiple experts on a ball Y = {x | ∥x∥≤D} enclosing domain X. After combining
their decisions, we project the solution in Y onto X, which is the only projection onto X per round.

4.1
Efficient algorithm for minimax universal regret

To handle unknown parameters of strong convexity and exp-concavity, we construct two finite
sets, i.e., Psc and Pexp, to approximate their values [Zhang et al., 2022]. Taking λ-strongly convex
functions as an example, we assume the unknown modulus λ is bounded by λ ∈[1/T, 1]2, and set
Psc = {1/T, 2/T, · · · , 2N/T}, where N = ⌈log2 T⌉. In this way, for any λ ∈[1/T, 1], there exists
a bλ ∈Psc such that bλ ≤λ ≤2bλ. Moreover, we design three types of expert-losses. For general
convex functions, we construct the expert-loss as

ℓcvx
t (y) = ⟨∇gt(yt), y −yt⟩,
(24)

where gt(y) is defined in (8). Since ℓcvx
t (·) is convex, we use OGD as the expert-algorithm to
minimize it. To handle exp-concave functions, we construct the expert-loss for each bα ∈Pexp as

ℓexp
t,bα(y) = ⟨∇gt(yt), y −yt⟩+
bβ
2 ⟨∇gt(yt), y −yt⟩2,
(25)

where bβ = 1

2 min{
1
4GD, bα}. It is easy to verify that ℓexp
t,bα(·) is
bβ
4 -exp-concave, so we use ONS as the

expert-algorithm. For strongly convex functions, we construct the expert-loss for each bλ ∈Psc as

ℓsc
t,bλ(y) = ⟨∇gt(yt), y −yt⟩+
bλ
2 ∥y −xt∥2.
(26)

Since ℓsc
t,bλ(·) is bλ-strongly convex, we use OGD with step size ηt = 1/[bλt] as the expert-algorithm.
Finally, we deploy a meta-algorithm to track the best expert on the fly. Following Zhang et al.

8


---Page Break---
Algorithm 2 Efficient Algorithm for Universal OCO

1: Input: The modulus set Psc and Pexp, the expert set A = ∅, the number of experts k = 0
2: k ←k + 1, create an expert E1 by running OGD with loss (24) over Y
3: for all bα ∈Pexp do
4:
k ←k + 1, create an expert Ek by running ONS with loss (25) and parameter bα over Y
5: end for
6: for all bλ ∈Psc do
7:
k ←k + 1, create an expert Ek by running OGD with loss (26) and parameter bλ over Y
8: end for
9: Add all the experts to the set: A = {E1, E2, · · · , Ek}
10: for t = 1 to T do
11:
Compute the weight pi
t of each expert Ei by (27)
12:
Receive the decision yi
t from each expert Ei in A

13:
Aggregate all the decisions by yt = P|A|
i=1 pi
tyi
t
14:
Submit the decision xt = ΠX [yt]
▷The only step projects onto domain X per round.
15:
Suffer the loss ft(xt) and observe the gradient ∇ft(xt)
16:
Construct the expert-loss ℓcvx
t (·), ℓsc
t (·) or ℓexp
t (·) and sent it to corresponding expert in A
17: end for

[2022], we use the linearized surrogate loss to measure the performance of the experts, and choose
Adapt-ML-Prod [Gaillard et al., 2014] as the meta-algorithm to yield a second-order bound.

Our efficient algorithm for universal OCO is summarized in Algorithm 2. From Steps 2 to 9, it
creates a set of experts by running multiple online algorithms over the ball Y, each specialized for
a distinct function type. Then, it maintains a set A consisting of all experts, and the i-th expert is
denoted by Ei. In the t-th round, it computes the weight pi
t of each expert Ei in Step 11 according to
Adapt-ML-Prod. After receiving all the predictions from the experts in Step 12, it aggregates them
based on their weights to attain yt in Step 13. Next, it conducts the only projection onto the original
domain X to obtain the actual decision xt in Step 14. In Step 15, it evaluates the gradient ∇ft(xt) to
construct the expert-losses in (24), (25), and (26). In Step 16, it sends the corresponding expert-loss
to each expert so that it can make predictions for the next round.

Finally, we elucidate how our algorithm determines the weight of the i-th expert Ei. We measure the
performance of expert Ei by the linearized surrogate loss, i.e., li
t = ⟨∇gt(yt), yi
t −yt⟩. According
to Lemma 2, we have |li
t| ≤∥∇gt(yt)∥∥yi
t −yt∥≤2GD. Since Adapt-ML-Prod requires the loss
to fall within the range of [0, 1], we normalize li
t to construct the meta-loss as ℓi
t = (⟨∇gt(yt), yi
t −
yt⟩)/(4GD) + 1

2 ∈[0, 1]. The loss of the meta-algorithm in the t-th round is ℓt = P|A|
i=1 pi
tℓi
t, which
is a constant 1

2 due to its construction and Step 13. For each expert Ei, its weight is updated by:

pi
t =
ηi
t−1wi
t−1
P|A|
j=1 ηj
t−1wj
t−1
, wi
t−1 =
 
wi
t−2
 
1 + ηi
t−2(ℓt−1 −ℓi
t−1)

ηi
t−1
ηi
t−2
(27)

where ηi
t−1 = min

1
2,
q

(ln |A|)/(1 + Pt−1
s=1(ℓs −ℓis)2)

. In the first round, we set wi
0 = 1/|A|.

Remark 3 While the surrogate loss in (8) involves the projection operation, our proposed meta-loss
and expert-losses only access gt(y) through ∇gt(yt), which is given by Cutkosky [2020],

∇gt(yt) = ∇ft(xt) −1{⟨∇ft(xt),vt⟩<0}⟨∇ft(xt), vt⟩· vt

where vt =
yt−xt
∥yt−xt∥. According to its formulation, the gradient can be directly computed from xt
and yt, which means no additional projections are needed. Therefore, in each round, our algorithm
requires only 1 projection onto domain X.
◁

Due to page limit, we provide the expert-algorithms, as well as all the proofs, in Appendix B. The
theoretical guarantee of Algorithm 2 is given below.

2One can verify the degenerated situations where the unknown modulus falls outside the range, which will
not be a concern. Formal justifications are provided in Appendix D.

9


---Page Break---
Theorem 1 Under Assumptions 1 and 2, Algorithm 2 attains O(
√

T), O( d

α log T) and O( 1

λ log T)
regret for general convex functions, α-exp-concave functions with α ∈[1/T, 1], and λ-strongly
convex functions with λ ∈[1/T, 1], respectively.

Remark 4 Similar to previous studies [Wang et al., 2019, Zhang et al., 2022], our universal algorithm
also achieves the minimax optimal regret, but only requires 1 projection.
◁

4.2
Efficient algorithm for small-loss universal regret

Furthermore, we consider the small-loss regret for smooth and non-negative online functions. To this
end, an additional assumption is required [Srebro et al., 2010].

Assumption 3 All the online functions are non-negative, and H-smooth over X.

To exploit the smoothness, we enhance the expert-loss for strongly convex functions in (26) as

bℓsc
t,bλ(y) = ⟨∇gt(yt), y −yt⟩+
bλ
2G2 ∥∇gt(yt)∥2∥y −xt∥2.
(28)

Since bℓsc
t,bλ(·) is strongly convex and smooth, we use S2OGD [Wang et al., 2020b] as the expert-
algorithm. For general convex and exp-concave functions, we reuse (24) and (25) as the expert-losses,
and employ ONS [Orabona et al., 2012] and SOGD [Zhang et al., 2019] as the expert-algorithms.
The meta-algorithm remains unchanged. In this way, we obtain the following regret guarantee.

Theorem 2 Under Assumptions 1, 2 and 3, the improved version of Algorithm 2 attains O(√LT ),
O( d

α log LT ) and O( 1

λ log LT ) regret for general convex functions, α-exp-concave functions with
α ∈[1/T, 1], and λ-strongly convex functions with λ ∈[1/T, 1], respectively, where the small-loss
quantity LT = minx∈X
PT
t=1 ft(x) is the cumulative loss of the best decision from the domain X.

Remark 5 With only 1 projection in each round, our universal algorithm is able to deliver optimal
small-loss regret bounds for multiple types of convex functions simultaneously. In contrast, Wang
et al. [2020b] and Zhang et al. [2022] take O(log T) projections to achieve the small-loss regret. ◁

5
Conclusion and future work

In this paper, we propose a projection-efficient universal algorithm that achieves minimax optimal
regret for three types of convex functions with only 1 projection per round. Furthermore, we enhance
our algorithm to exploit the smoothness property and demonstrate that it attains small-loss regret for
convex and smooth functions. To demonstrate the effectiveness of our proposed method, we also
conduct empirical experiments, and the results are presented in Appendix E.

There are several directions for future research. First, one potentially unfavorable characteristic
of our work is the requirements of domain and gradient boundedness. Motivated by the recent
developments in parameter-free online learning for unbounded domains and gradients [Orabona,
2014, Orabona and Pál, 2016, Cutkosky and Boahen, 2016, 2017, Foster et al., 2017, Luo et al.,
2022, Jacobsen and Cutkosky, 2022, 2023], we will investigate whether our algorithms can further
avoid these prior knowledge in the future. Second, in addition to the small-loss bound, another
important type of problem-dependent guarantee is the gradient-variation regret bound [Zhao et al.,
2020, 2024], which has been actively studied recently due to its profound relationship to games and
stochastic optimization. In the literature, recent studies [Yan et al., 2023, 2024, Xie et al., 2024,
Wang et al., 2024a] achieve almost-optimal gradient-variation regret in universal online learning, but
also suffer high projection complexity. Therefore, it remains challenging and important to develop
a projection-efficient universal algorithm with optimal gradient-variation regret guarantees. Third,
to deal with changing environments, adaptive regret has been proposed to minimize the regret over
every interval in various setting of online learning [Hazan and Seshadhri, 2007, Daniely et al., 2015,
Wan et al., 2021a, Wang et al., 2024b]. Existing universal algorithms [Zhang et al., 2021, Yang et al.,
2024] typically conduct O(log2 T) projections per round. In the future, we will investigate whether
whether we can reduce the projection complexity of universal algorithms for adaptive regret.

10


---Page Break---
Acknowledgments

This work was partially supported by National Science and Technology Major Project
(2022ZD0114801), NSFC (62361146852, 62122037), and the Collaborative Innovation Center
of Novel Software Technology and Industrialization.

References

J. Abernethy, P. L. Bartlett, A. Rakhlin, and A. Tewari. Optimal strategies and minimax lower bounds
for online convex games. In Proceedings of the 21st Annual Conference on Learning Theory
(COLT), pages 415–423, 2008.

P. L. Bartlett, E. Hazan, and A. Rakhlin. Adaptive online gradient descent. In Advances in Neural
Information Processing Systems 20 (NIPS), pages 65–72, 2008.

N. Cesa-Bianchi and G. Lugosi. Prediction, Learning, and Games. Cambridge University Press,
2006.

C.-C. Chang and C.-J. Lin. LIBSVM: A library for support vector machines. ACM Transactions on
Intelligent Systems and Technology, 2(3):27:1–27:27, 2011.

S. Chen, Y.-J. Z. W.-W. Tu, P. Zhao, and L. Zhang. Optimistic online mirror descent for bridging
stochastic and adversarial online convex optimization. Journal of Machine Learning Research,
pages 1 – 62, 2024.

C.-K. Chiang, T. Yang, C.-J. Lee, M. Mahdavi, C.-J. Lu, R. Jin, and S. Zhu. Online optimization with
gradual variations. In Proceedings of the 25th Annual Conference on Learning Theory (COLT),
pages 1–20, 2012.

A. Cutkosky. Parameter-free, dynamic, and strongly-adaptive online learning. In Proceedings of the
37th International Conference on Machine Learning (ICML), pages 2250–2259, 2020.

A. Cutkosky and K. Boahen. Online learning without prior information. In Proceedings of the 30th
Annual Conference on Learning Theory (COLT), pages 643–677, 2017.

A. Cutkosky and K. A. Boahen. Online convex optimization with unconstrained domains and losses.
In Advances in Neural Information Processing Systems 29 (NIPS), pages 748–756, 2016.

A. Cutkosky and F. Orabona. Black-box reductions for parameter-free online learning in Banach
spaces. In Proceedings of the 31st Conference On Learning Theory (COLT), pages 1493–1529,
2018.

A. Daniely, A. Gonen, and S. Shalev-Shwartz. Strongly adaptive online learning. In Proceedings of
the 32nd International Conference on Machine Learning (ICML), pages 1405–1411, 2015.

J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic
optimization. In Proceedings of the 23rd Annual Conference on Learning Theory (COLT), pages
257–269, 2010.

D. J. Foster, S. Kale, M. Mohri, and K. Sridharan. Parameter-free online learning via model selection.
In Advances in Neural Information Processing Systems 30 (NIPS), pages 6020–6030, 2017.

P. Gaillard, G. Stoltz, and T. van Erven. A second-order bound with excess losses. In Proceedings of
the 27th Conference on Learning Theory (COLT), pages 176–196, 2014.

D. Garber and B. Kretzu. Projection-free online exp-concave optimization. In Proceedings of Thirty
Sixth Conference on Learning Theory (COLT), pages 1259–1284, 2023.

E. Hazan. Introduction to Online Convex Optimization. Foundations and Trends in Optimization, 2
(3-4):157–325, 2016.

E. Hazan and S. Kale. Projection-free online learning. In Proceedings of the 29th International
Conference on Machine Learning (ICML), pages 521–528, 2012.

11


---Page Break---
E. Hazan and E. Minasyan. Faster projection-free online learning. In Proceedings of Thirty Third
Conference on Learning Theory (COLT), pages 1877–1893, 2020.

E. Hazan and C. Seshadhri. Adaptive algorithms for online decision problems. Electronic Colloquium
on Computational Complexity, 88, 2007.

E. Hazan, A. Agarwal, and S. Kale. Logarithmic regret algorithms for online convex optimization.
Machine Learning, 69(2-3):169–192, 2007.

A. Jacobsen and A. Cutkosky. Parameter-free mirror descent. In Proceedings of 35th Conference on
Learning Theory (COLT), pages 4160–4211, 2022.

A. Jacobsen and A. Cutkosky. Unconstrained online learning with unbounded losses. In Proceedings
of the 40th International Conference on Machine Learning (ICML), pages 14590–14630, 2023.

D. P. Kingma and J. L. Ba. Adam: A method for stochastic optimization. In International Conference
on Learning Representations (ICLR), 2015.

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on
Learning Representations (ICLR), 2019.

H. Luo and R. E. Schapire. Achieving all with no parameters: AdaNormalHedge. In Proceedings of
the 28th Conference on Learning Theory (COLT), pages 1286–1304, 2015.

H. Luo, M. Zhang, P. Zhao, and Z.-H. Zhou. Corralling a larger band of bandits: A case study on
switching regret for linear bandits. In Proceedings of the 35th Conference on Learning Theory
(COLT), pages 3635–3684, 2022.

Z. Mhammedi, W. M. Koolen, and T. Van Erven. Lipschitz adaptivity with multiple learning rates
in online learning. In Proceedings of the 32nd Conference on Learning Theory (COLT), pages
2490–2511, 2019.

M. Mohri and S. Yang.
Accelerating online convex optimization via adaptive prediction.
In
Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS),
pages 848–856, 2016.

M. C. Mukkamala and M. Hein. Variants of RMSProp and Adagrad with logarithmic regret bounds. In
Proceedings of the 34th International Conference on Machine Learning (ICML), pages 2545–2553,
2017.

F. Orabona. Simultaneous model selection and optimization through parameter-free stochastic
learning. In Advances in Neural Information Processing Systems 27 (NIPS), pages 1116–1124,
2014.

F. Orabona and D. Pál. Coin betting and parameter-free online learning. In Advances in Neural
Information Processing Systems 29 (NIPS), pages 577–585, 2016.

F. Orabona and D. Pál. Scale-free online learning. Theoretical Computer Science, 716:50–69, 2018.

F. Orabona, N. Cesa-Bianchi, and C. Gentile. Beyond logarithmic bounds in online learning. In
Proceedings of the 15th International Conference on Artificial Intelligence and Statistics (AISTATS),
pages 823–831, 2012.

E. Ordentlich and T. M. Cover. The cost of achieving the best portfolio in hindsight. Mathematics of
Operations Research, 23(4):960–982, 1998.

D. Prokhorov. IJCNN 2001 neural network competition. Technical report, Ford Research Laboratory,
2001.

S. J. Reddi, S. Kale, and S. Kumar. On the convergence of Adam and beyond. In International
Conference on Learning Representations (ICLR), 2018.

S. Shalev-Shwartz. Online Learning: Theory, Algorithms, and Applications. PhD thesis, The Hebrew
University of Jerusalem, 2007.

12


---Page Break---
S. Shalev-Shwartz, Y. Singer, and N. Srebro. Pegasos: primal estimated sub-gradient solver for SVM.
In Proceedings of the 24th International Conference on Machine Learning (ICML), pages 807–814,
2007.

S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter. Pegasos: primal estimated sub-gradient solver
for svm. Mathematical Programming, 127(1):3–30, 2011.

N. Srebro, K. Sridharan, and A. Tewari. Smoothness, low-noise and fast rates. In Advances in Neural
Information Processing Systems 23 (NIPS), pages 2199–2207, 2010.

T. Tieleman and G. Hinton. Lecture 6.5-RMSProp: Divide the gradient by a running average of its
recent magnitude. COURSERA: Neural networks for machine learning, pages 26–31, 2012.

T. van Erven and W. M. Koolen. MetaGrad: Multiple learning rates in online learning. In Advances
in Neural Information Processing Systems 29 (NIPS), pages 3666–3674, 2016.

Y. Wan and L. Zhang. Projection-free online learning over strongly convex sets. Proceedings of the
AAAI Conference on Artificial Intelligence (AAAI), pages 10076–10084, 2021.

Y. Wan, W.-W. Tu, and L. Zhang. Strongly adaptive online learning over partial intervals. Science
China Information Sciences, 2021a.

Y. Wan, B. Xue, and L. Zhang. Projection-free online learning in dynamic environments. Proceedings
of the AAAI Conference on Artificial Intelligence (AAAI), pages 10067–10075, 2021b.

Y. Wan, W.-W. Tu, and L. Zhang. Online frank-wolfe with arbitrary delays. In Advances in Neural
Information Processing Systems (NeurIPS), pages 19703–19715, 2022.

G. Wang, S. Lu, and L. Zhang. Adaptivity and optimality: A universal algorithm for online convex
optimization. In Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI),
pages 659–668, 2019.

G. Wang, S. Lu, Q. Cheng, W.-W. Tu, and L. Zhang. SAdam: A variant of Adam for strongly convex
functions. In International Conference on Learning Representations (ICLR), 2020a.

G. Wang, S. Lu, Y. Hu, and L. Zhang. Adapting to smoothness: A more universal algorithm for
online convex optimization. In Proceedings of the 34th AAAI Conference on Artificial Intelligence
(AAAI), pages 6162–6169, 2020b.

Y. Wang, Y. Wan, S. Zhang, and L. Zhang. Distributed projection-free online learning for smooth
and convex losses. In Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI),
pages 10226–10234, 2023.

Y. Wang, S. Chen, W. Jiang, W. Yang, Y. Wan, and L. Zhang. Online composite optimization between
stochastic and adversarial environments. In Advances in Neural Information Processing Systems
37 (NeurIPS), 2024a.

Y. Wang, W. Yang, W. Jiang, S. Lu, B. Wang, H. Tang, Y. Wan, and L. Zhang. Non-stationary
projection-free online learning with dynamic and adaptive regret guarantees. In Proceedings of the
38th AAAI Conference on Artificial Intelligence (AAAI), pages 15671–15679, 2024b.

Y.-F. Xie, P. Zhao, and Z.-H. Zhou. Gradient-variation online learning under generalized smoothness.
In Advances in Neural Information Processing Systems 37 (NeurIPS), 2024.

Y.-H. Yan, P. Zhao, and Z.-H. Zhou. Universal online learning with gradient variations: A multi-layer
online ensemble approach. In Advances in Neural Information Processing Systems 36 (NeurIPS),
pages 37682–37715, 2023.

Y.-H. Yan, P. Zhao, and Z.-H. Zhou. A simple and optimal approach for universal online learning
with gradient variations. In Advances in Neural Information Processing Systems 37 (NeurIPS),
2024.

T. Yang, M. Mahdavi, R. Jin, and S. Zhu. Regret bounded by gradual variation for online convex
optimization. Machine Learning, 95:183–223, 2014.

13


---Page Break---
W. Yang, W. Jiang, Y. Wang, P. Yang, Y. Hu, and L. Zhang. Small-loss adaptive regret for online
convex optimization. In Proceedings of the 41st International Conference on Machine Learning
(ICML), pages 56156–56195, 2024.

L. Zhang, T.-Y. Liu, and Z.-H. Zhou. Adaptive regret of convex and smooth functions. In Proceedings
of the 36th International Conference on Machine Learning (ICML), pages 7414–7423, 2019.

L. Zhang, G. Wang, W.-W. Tu, W. Jiang, and Z.-H. Zhou. Dual adaptivity: A universal algorithm for
minimizing the adaptive regret of convex functions. In Advances in Neural Information Processing
Systems 34 (NeurIPS), pages 24968–24980, 2021.

L. Zhang, G. Wang, J. Yi, and T. Yang. A simple yet universal strategy for online convex optimization.
In Proceedings of the 39th International Conference on Machine Learning (ICML), pages 26605–
26623, 2022.

P. Zhao, Y.-J. Zhang, L. Zhang, and Z.-H. Zhou. Dynamic regret of convex and smooth functions. In
Advances in Neural Information Processing Systems 33 (NeurIPS), pages 12510–12520, 2020.

P. Zhao, Y.-F. Xie, L. Zhang, and Z.-H. Zhou. Efficient methods for non-stationary online learning.
In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 11573–11585, 2022.

P. Zhao, Y.-J. Zhang, L. Zhang, and Z.-H. Zhou. Adaptivity and non-stationarity: Problem-dependent
dynamic regret for online convex optimization. Journal of Machine Learning Research, 25(98):1 –
52, 2024.

M. Zinkevich.
Online convex programming and generalized infinitesimal gradient ascent.
In
Proceedings of the 20th International Conference on Machine Learning (ICML), pages 928–936,
2003.

14


---Page Break---
A
Algorithms for experts

In this section, we provide the detailed procedures of the expert-algorithms in our efficient algorithm.

A.1
Online gradient descent for convex functions

We use OGD [Zinkevich, 2003] to minimize ℓcvx
t (·) in (24). The procedure of the expert-algorithm
for general convex functions is summarized in Algorithm 3.

Algorithm 3 Expert Ei: OGD for Convex Functions

1: Let yi
1 be any point in Y
2: for t = 1 to T do
3:
Submit yi
t to the meta-algorithm
4:
Update

byi
t+1 = yi
t −1
√

t∇gt(yt)

5:
Conduct a projection onto Y

yi
t+1 =

( byi
t+1,
if ∥byi
t+1∥≤D,
byi
t+1 ·
D
∥byi
t+1∥,
otherwise .

6: end for

A.2
Online newton step for exp-concave (and smooth) functions

Lemma 4 Under Assumptions 1 and 2, ℓexp
t,bα(·) in (25) is
bβ
4 -exp-concave, and ∥∇ℓexp
t,bα(y)∥2 ≤2G2.

Thus, we use ONS to minimize ℓexp
t,bα(·). Different from OGD, the projection of ONS onto Y cannot
be achieved through a simple rescaling like Step 5 in Algorithm 3. Here, we employ an efficient
implementation of ONS [Mhammedi et al., 2019] that enhances the efficiency of its projection onto
Y. The procedure is summarized in Algorithm 4.

Algorithm 4 Expert Ei: ONS for Exp-concave (and Smooth) Functions

1: Let yi
1 be any point in Y and Σ1 =
1
bβ2D2 Id
2: for t = 1 to T do
3:
Submit yi
t to the meta-algorithm
4:
Update

Σt+1 = Σt + ∇ℓexp
t,bα(yi
t)∇ℓexp
t,bα(yi
t)⊤,
byi
t+1 = yi
t −1

bβ
Σ−1
t+1∇ℓexp
t,bα(yi
t)

where
∇ℓexp
t,bα(yi
t) = ∇gt(yt) + bβ∇gt(yt)∇gt(yt)⊤(yi
t −yt)

5:
Conduct a projection onto Y in (29)
6: end for

Lemma 5 Let Λt+1 := diag((λk
t )k∈[d]) and Qt+1 are the matrices of eigenvalues and eigenvectors
of (Σt+1 −
1
bβ2D2 Id), respectively. Then, the projection onto the ball Y in Step 5 can be formulated as

yi
t+1 =

(
byi
t+1,
if ∥byi
t+1∥≤D,

Q⊤
t+1
 
xi
t+1I + Λt+1
−1 Qt+1Σt+1byi
t+1,
otherwise .
(29)

where xi
t+1 is the unique solution of ρ(x) := Pd
k=1
⟨ek,Qt+1Σt+1byi
t+1⟩2

(x+λk
t )2
= D2.

15


---Page Break---
A.3
Online gradient descent for strongly convex functions

We establish the following lemma for function ℓsc
t (·) in (14).

Lemma 6 Under Assumptions 1 and 2, the loss function ℓsc
t (·) in (14) is λ-strongly convex, and
∥∇ℓsc
t (y)∥2 ≤(G + 2D)2.

Since ℓsc
t,bλ(·) in (26) shares the same formulation as ℓsc
t (·), ℓsc
t,bλ(·) also benefits from the aforemen-

tioned properties, with the distinction being the substitution of λ for bλ. Therefore, we use a variant of
OGD [Shalev-Shwartz et al., 2007] to minimize ℓsc
t,bλ(·). The procedure is summarized in Algorithm 5.

Algorithm 5 Expert Ei: OGD for Strongly Convex Functions

1: Let yi
1 be any point in Y
2: for t = 1 to T do
3:
Submit yi
t to the meta-algorithm
4:
Update

byi
t+1 = yi
t −1

bλt
∇ℓsc
t,bλ(yi
t)

where
∇ℓsc
t,bλ(yi
t) = ∇gt(yt) + bλ(yi
t −xt)

5:
Conduct a projection onto Y

yi
t+1 =

( byi
t+1,
if ∥byi
t+1∥≤D,
byi
t+1 ·
D
∥byi
t+1∥,
otherwise .

6: end for

A.4
Scale-free online gradient descent for convex and smooth functions

To exploit smoothness, we use scale-free online gradient descent (SOGD) [Zhang et al., 2019] to
minimize ℓcvx
t (·) in (24). The procedure is summarized in Algorithm 6.

Algorithm 6 Expert Ei: Scale-free OGD for Convex and Smooth Functions

1: Let yi
1 be any point in Y
2: for t = 1 to T do
3:
Submit yi
t to the meta-algorithm
4:
Update
byi
t+1 = yi
t −ηt∇gt(yt)
where
ηt =
α
q

δ + Pt
s=1 ∥∇gs(ys)∥2
,
α, δ > 0

5:
Conduct a projection onto Y

yi
t+1 =

( byi
t+1,
if ∥byi
t+1∥≤D,
byi
t+1 ·
D
∥byi
t+1∥,
otherwise .

6: end for

A.5
Smooth and strongly convex online gradient descent

Recall that to exploit smoothness, we enhance the expert-loss for strongly convex functions as follows

bℓsc
t,bλ(y) = ⟨∇gt(yt), y −yt⟩+
bλ
2G2 ∥∇gt(yt)∥2∥y −xt∥2.

16


---Page Break---
The above expert-loss enjoys the following property.

Lemma 7 Under Assumptions 1 and 2, bℓsc
t,bλ(·) in (28) is
bλ
G2 ∥∇gt(yt)∥2-strongly convex, and

∥bℓsc
t,bλ(y)∥2 ≤
 
1 + 2D

G
2 ∥∇gt(yt)∥2.

Due to the modulus of strong convexity is not fixed, we choose Smooth and Strongly Convex
OGD (S2OGD) as the expert-algorithm [Wang et al., 2020b] to minimize bℓsc
t,bλ(·). The procedure is
summarized in Algorithm 7.

Algorithm 7 Expert Ei: Smooth and Strongly Convex OGD

1: Let yi
1 be any point in Y
2: for t = 1 to T do
3:
Submit yi
t to the meta-algorithm
4:
Update
byi
t+1 = yi
t −ηt∇gt(yt)
where
ηt =
α

δ + Pt
s=1 ∥∇bℓsc
s,bλ(yis)∥2 ,
α, δ > 0

5:
Conduct a projection onto Y

yi
t+1 =

( byi
t+1,
if ∥byi
t+1∥≤D,
byi
t+1 ·
D
∥byi
t+1∥,
otherwise .

6: end for

B
Proofs

In this section, we provide the proofs of the theorems presented in the main paper (Theorem 1 and
Theorem 2), as well as proofs of two important lemmas (Lemma 3 and Lemma 8).

B.1
Proof of Theorem 1

We present the exact bounds of the theoretical guarantee provided in Theorem 1. When functions are
general convex, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+

 
2ΓGD
p

ln |A|
+ 2D2 + G2
!
√

T −G2

2

= O(
√

T)

where |A| = 1 + 2⌈log2 T⌉and

Γ = 3 ln |A| + ln

1 + |A|

2e (1 + ln(T + 1))

= O(log log T).
(30)

When functions are α-exp-concave, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

β ln |A| + 5
 8

β + 2
√

2GD

d log T

= O
 d

α log T

.

17


---Page Break---
When functions are λ-strongly convex, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

min{ G

D, λ} ln |A| + (G + D)2

λ
log T

= O
 1

λ log T

.

B.1.1
Analysis for general convex functions

We introduce the following decomposition for general convex functions,

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1
⟨∇ft(xt), xt −x⟩
(9)
≤

T
X

t=1
⟨∇gt(yt), yt −x⟩

=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩+

T
X

t=1
⟨∇gt(yt), yi
t −x⟩

(24)
=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1

 
ℓcvx
t (yi
t) −ℓcvx
t (x)


|
{z
}
expert-regret

.

(31)

First, we start with the expert-regret. Since we are employing OGD to minimize ℓcvx
t (·), using
standard OGD analysis [Zinkevich, 2003, Theorem 1] can obtain the following upper bound

T
X

t=1
ℓcvx
t (yi
t) −

T
X

t=1
ℓcvx
t (x) ≤(2D2 + G2)
√

T −G2

2 ,
(32)

for any expert yi
t ∈Y and any x ∈X.

Next, we move to bound the meta-regret. According to (48), we have

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤8ΓGD +
Γ
p

ln |A|

v
u
u
t16G2D2 +

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ
p

ln |A|

v
u
u
t

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ
p

ln |A|

v
u
u
t

T
X

t=1
∥∇gt(yt)∥2∥yt −yi
t∥2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+ 2ΓGD
p

ln |A|

√

T,

(33)
for all expert Ei ∈A, where Γ is defined in (30) and the last set is due to

∥∇gt(yt)∥≤∥∇ft(xt)∥≤G.
(34)

Finally, substituting (32) and (33) into (31), we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+

 
2ΓGD
p

ln |A|
+ 2D2 + G2
!
√

T −G2

2 .

B.1.2
Analysis for exp-concave functions

For α-exp-concave functions, there exits bα∗∈Pexp that bα∗≤α ≤2bα∗, where bα∗is the modulus of
the i-th expert Ei. This inequality also indicates

bβ∗≤β ≤2bβ∗,
bβ∗= 1

2 min{
1
4GD, bα∗}.
(35)

18


---Page Break---
Since x −
bβ∗

2 x2 is strictly increasing where bβ∗= 1

2 min{
1
4GD, bα∗} when x ∈(−∞, 2GD], (9)
implies that

⟨∇ft(xt), xt −x⟩−
bβ∗

2 ⟨∇ft(xt), xt −x⟩2 ≤⟨∇gt(yt), yt −x⟩−
bβ∗

2 ⟨∇gt(yt), yt −x⟩2. (36)

Then, we introduce the following decomposition for α-exp-concave functions,

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−β

2

T
X

t=1
⟨∇ft(xt), xt −x⟩2

(35)
≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−
bβ∗

2

T
X

t=1
⟨∇ft(xt), xt −x⟩2

(36)
≤

T
X

t=1
⟨∇gt(yt), yt −x⟩−
bβ∗

2

T
X

t=1
⟨∇gt(yt), yt −x⟩2

=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩+

T
X

t=1
⟨∇gt(yt), yi
t −x⟩−
bβ∗

2

T
X

t=1
⟨∇gt(yt), yt −x⟩2

(25)
=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1


ℓexp
t,bα∗(yi
t) −ℓexp
t,bα∗(x)


|
{z
}
expert-regret

−
bβ∗

2

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2.

(37)

For the expert-regret, we can use the analysis of ONS [Hazan et al., 2007, Theorem 2] to obtain

T
X

t=1
ℓexp
t,bα∗(yi
t) −

T
X

t=1
ℓexp
t,bα∗(x) ≤5
 4

bβ∗+ 2
√

2GD

d log T
(38)

for any expert yi
t ∈Y and any x ∈X, where bβ∗is defined in (35). Next, we move to bound the
meta-regret. According to (48), we have

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤8ΓGD +
Γ
p

ln |A|

v
u
u
t16G2D2 +

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ
p

ln |A|

v
u
u
t

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

2bβ∗ln |A|
+
bβ∗

2 ⟨∇gt(yt), yt −yi
t⟩2

(39)

for all expert Ei ∈A, where Γ is defined in (30) and the last step is due to
√

ab ≤a

2 + b

2. Substituting
(38) and (39) into (37), we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

2bβ∗ln |A|
+ 5
 4

bβ∗+ 2
√

2GD

d log T.

Finally, we use (35) to simplify the above bound.

19


---Page Break---
B.1.3
Analysis for strongly convex functions

For λ-strongly convex functions, there exits bλ∗∈Psc that bλ∗≤λ ≤2bλ∗, where bλ∗is the modulus of
the i-th expert Ei. Then, we introduce the following decomposition for λ-strongly convex functions

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−λ

2

T
X

t=1
∥xt −x∥2

≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−
bλ∗

2

T
X

t=1
∥xt −x∥2

(20)
≤

T
X

t=1
⟨∇gt(yt), yt −x⟩−∆T −
bλ∗

2

T
X

t=1
∥xt −x∥2

(26)
=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1


ℓsc
t,bλ∗(yi
t) −ℓsc
t,bλ∗(x)


|
{z
}
expert-regret

−
bλ∗

2

T
X

t=1
∥yi
t −xt∥2 −∆T

(40)

where ∆T = PT
t=1 1{⟨∇ft(xt),vt⟩≥0} · ⟨∇ft(xt), yt −xt⟩. To bound the meta-regret, we derive the
following theoretical guarantee.

Lemma 8 Under Assumptions 1 and 2, the meta-regret of Algorithm 2 satisfies

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤8ΓGD +
Γ
p

ln |A|

v
u
u
t16G2D2 +

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| +
γ
2G2

T
X

t=1
⟨∇gt(yt), xt −yi
t⟩2 + ∆T

for any γ ∈(0, G

2D], where ∆T = PT
t=1 1{⟨∇ft(xt),vt⟩≥0} · ⟨∇ft(xt), yt −xt⟩and Γ is in (30).

Remark 6 As mentioned in Section 3.2, Lemma 8 is pivotal in delivering optimal regret for strongly
convex functions. Specifically, when the meta-algorithm enjoys a second-order bound in terms of the
surrogate loss in (8), we can then convert the intermediate decision yt in the meta-regret bound to the
actual one xt, at the cost of adding an addition positive term, as presented in the analysis in (23). ◁

Combining Lemma 8 with (40) , we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| +
γ
2G2

T
X

t=1
⟨∇gt(yt), xt −yi
t⟩2

+ ER(T) −
bλ∗

2

T
X

t=1
∥yi
t −xt∥2

(34)
≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| +

 
γ
2 −
bλ∗

2

!
T
X

t=1
∥xt −yi
t∥2 + ER(T)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| + ER(T)

(41)

where ER(T) = PT
t=1(ℓsc
t,bλ∗(yi
t) −ℓsc
t,bλ∗(x)) and the last step is because we set γ = min{ G

2D, bλ∗}.
Next, we bound the expert-regret [Shalev-Shwartz et al., 2011, Lemma 1]

ER(T) =

T
X

t=1
ℓsc
t,bλ∗(yi
t) −

T
X

t=1
ℓsc
t,bλ∗(x) ≤(G + D)2

2bλ∗
log T.
(42)

20


---Page Break---
for any expert yi
t ∈Y and any x ∈X. Substituting (42) into (41), we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| + (G + D)2

2bλ∗
log T.

Finally, we use bλ∗≤λ ≤2bλ∗to simplify the above bound.

B.2
Proof of Lemma 3

According to (8), the (sub-)gradients of gt(·) can be formulated as

∇gt(y) =

( ∇ft(xt),
if ⟨∇ft(xt), vt⟩≥0,

∇ft(xt) −⟨∇ft(xt), vt⟩·
y−ΠX [y]
∥y−ΠX [y]∥,
if ⟨∇ft(xt), vt⟩< 0.
(43)

(i) When ⟨∇ft(xt), vt⟩≥0. We have gt(y) = ⟨∇ft(xt), y⟩and ∇gt(y) = ∇ft(xt). Thus,

⟨∇ft(xt), xt −x⟩= ⟨∇gt(yt), yt −x⟩−⟨∇ft(xt), yt −xt⟩.
(44)

By the definition of vt = (yt −xt)/∥yt −xt∥, we have ⟨∇ft(xt), xt⟩≤⟨∇ft(xt), yt⟩and thus

⟨∇gt(yt), xt⟩≤⟨∇gt(yt), yt⟩
(45)

(ii) When ⟨∇ft(xt), vt⟩< 0. According to Lemma 2, we obtain

⟨∇ft(xt), xt −x⟩≤⟨∇gt(yt), yt −x⟩.
(46)

Moreover, we derive the following equation

⟨∇gt(yt), yt −xt⟩= ⟨∇ft(xt), yt −xt⟩−⟨∇ft(xt), vt⟩· ⟨vt, yt −xt⟩

= ⟨∇ft(xt), yt −xt⟩−⟨∇ft(xt), yt −xt⟩·
1
∥yt −xt∥

 yt −xt

∥yt −xt∥, yt −xt


= 0.
(47)

Finally, combining (44) and (46) obtains (20), further combining (45) and (47) yields (21).

B.3
Proof of Lemma 8

By the regret guarantee of Adapt-ML-Prod [Gaillard et al., 2014, Corollary 4], we have

T
X

t=1

 
ℓt −ℓi
t

≤2Γ +
Γ
p

ln |A|

v
u
u
t1 +

T
X

t=1

 
ℓt −ℓi
t
2

for all expert Ei ∈A, where Γ = 3 ln |A| + ln(1 + |A|

2e (1 + ln(T + 1))) = O(log log T). By the
definition of ℓt and ℓi
t, we have

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤8ΓGD +
Γ
p

ln |A|

v
u
u
t16G2D2 +

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| +
γ
2G2

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2,

(48)

for any γ > 0, where the last step uses AM-GM inequality.

Next, we handle the term ⟨∇gt(yt), yt −yi
t⟩2. We will consider two cases separately.

(i) When ⟨∇ft(xt), vt⟩≥0 , we have

⟨∇ft(xt), xt −yi
t⟩≤⟨∇ft(xt), yt −yi
t⟩≤∥∇ft(xt)∥∥yt −yi
t∥≤2GD.
(49)

As the function q(x) = x −
γ
2G2 x2 is strictly increasing when x ∈(−∞, G2

γ ], (49) implies that

⟨∇ft(xt), xt −yi
t⟩−
γ
2G2 ⟨∇ft(xt), xt −yi
t⟩2 ≤⟨∇ft(xt), yt −yi
t⟩−
γ
2G2 ⟨∇ft(xt), yt −yi
t⟩2.

21


---Page Break---
for any γ ∈(0, G

2D]. By rearranging terms, we obtain
γ
2G2 ⟨∇gt(yt), yt −yi
t⟩2 (43)
=
γ
2G2 ⟨∇ft(xt), yt −yi
t⟩2

≤⟨∇ft(xt), yt −xt⟩+
γ
2G2 ⟨∇ft(xt), xt −yi
t⟩2

(43)
= ⟨∇ft(xt), yt −xt⟩+
γ
2G2 ⟨∇gt(yt), xt −yi
t⟩2.

(50)

(ii) When ⟨∇ft(xt), vt⟩< 0, (47) directly implies ⟨∇gt(yt), xt −yi
t⟩= ⟨∇gt(yt), yt −yi
t⟩. Thus,
γ
2G2 ⟨∇gt(yt), yt −yi
t⟩2 =
γ
2G2 ⟨∇gt(yt), xt −yi
t⟩2.
(51)

Combining (50) and (51), we have
γ
2G2 ⟨∇gt(yt), yt −yi
t⟩2 ≤1{⟨∇ft(xt),vt⟩≥0}⟨∇ft(xt), yt −xt⟩+
γ
2G2 ⟨∇gt(yt), xt −yi
t⟩2 (52)

for any γ ∈(0, G

2D]. Substituting (52) into (48), we finish the proof.

B.4
Proof of Theorem 2

The analysis is similar to Theorem 1. Also, we present the exact bounds of the theoretical guarantee
provided in Theorem 2. When functions are general convex, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
√

2D2δ + 4H

 
2ΓD
p

ln |A|
+
√

2(D + 2G)

!2

+ 2
√

H

 
2ΓD
p

ln |A|
+
√

2(D + 2G)

! v
u
u
tLT + 4ΓGD

 

2 +
1
p

ln |A|

!

+
√

2D2δ

= O(
p

LT ).

where |A| = 1 + 2⌈log2 T⌉, Γ is defined in (30), and LT = minx∈X
PT
t=1 ft(x). When functions
are α-exp-concave, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

2β ln |A| + 2d

β log

 
β2D2H

d

T
X

t=1
ft(xt) + 1

!

+ 2

β

≤bΓ + 2d

β log

 
2β2D2H

d

T
X

t=1
ft(x) + 2β2D2H

d
bΓ + 2D2H log(2D2H) + 2

!

= O
 d

α log LT



where bΓ = 4ΓGD

2 +
1
√

ln |A|


+
Γ2
2β ln |A| + 2

β . When functions are λ-strongly convex, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤eΓ + (G + 2D)2

2λ
log

 
8Hλ
(G + 2D)2

T
X

t=1
ft(x) +
8Hλ
(G + 2D)2 eΓ + 2H log(2H) + 2

!

= O
 1

λ log LT



where eΓ = 4ΓGD

2 +
1
√

ln |A|


+
Γ2G2
2γ ln |A| + 1.

22


---Page Break---
B.4.1
Analysis for general convex functions

We start with the meta-expert regret decomposition as presented in (31),

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1

 
ℓcvx
t (yi
t) −ℓcvx
t (x)


|
{z
}
expert-regret

.
(53)

For the meta-regret, we reuse (33) to obtain

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ
p

ln |A|

v
u
u
t

T
X

t=1
∥∇gt(yt)∥2∥yt −yi
t∥2

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
2ΓD
p

ln |A|

v
u
u
t

T
X

t=1
∥∇gt(yt)∥2,

(54)
for all expert Ei ∈A, where Γ is defined in (30). For the expert-regret, we can use the analysis of
SOGD [Zhang et al., 2019, Theorem 2] to obtain

T
X

t=1
ℓcvx
t (yi
t) −

T
X

t=1
ℓcvx
t (x) ≤
√

2D2

v
u
u
tδ +

1 + 2G

D

2
T
X

t=1
∥∇gt(yt)∥2.

for any expert yi
t ∈Y and any x ∈X. From the above formulation, we have

T
X

t=1
ℓcvx
t (yi
t) −

T
X

t=1
ℓcvx
t (x) ≤
√

2D2δ +

v
u
u
t2 (D + 2G)2
T
X

t=1
∥∇gt(yt)∥2.
(55)

Substituting (54) and (55) into (53), we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

(34)
≤4ΓGD

 

2 +
1
p

ln |A|

!

+
√

2D2δ +

 
2ΓD
p

ln |A|
+
√

2(D + 2G)

! v
u
u
t

T
X

t=1
∥∇ft(xt)∥2.

Next, we introduce the self-bounding property of smooth functions.

Lemma 9 (Lemma 3.1 of Srebro et al. [2010]) For an H-smooth and nonnegative function, we
have ∥∇f(x)∥≤
p

4Hf(x).

Thus, when functions are smooth, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

(34)
≤4ΓGD

 

2 +
1
p

ln |A|

!

+
√

2D2δ +

 
2ΓD
p

ln |A|
+
√

2(D + 2G)

! v
u
u
t4H

T
X

t=1
ft(xt).

To simplify the above inequality, we use the following lemma.

Lemma 10 (Lemma 19 of Shalev-Shwartz [2007]) Let x, b, c ∈R+. Then, we have x −c ≤
b√x ⇒x −c ≤b2 + b√c.

By utilizing Lemma 10, we finish the proof.

23


---Page Break---
B.4.2
Analysis for exp-concave functions

The analysis is also similar to Theorem 1. We start with (37)

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1


ℓexp
t,bα∗(yi
t) −ℓexp
t,bα∗(x)


|
{z
}
expert-regret

−
bβ∗

2

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩2.
(56)

For the meta-regret, we also use (39) to bound. For the expert-regret, we can use the analysis of ONS
under the smoothness condition [Orabona et al., 2012, Theorem 1] to get

T
X

t=1
ℓexp
t,bα(yi
t) −

T
X

t=1
ℓexp
t,bα(x) ≤2d

bβ∗log

 bβ∗2D2

16d

T
X

t=1
∥∇ℓexp
t,bα(yi
t)∥2 + 1

!

+ 2

bβ∗.

for any expert yi
t ∈Y and any x ∈X. Next, we provide an upper bound for ∥∇ℓexp
t,bα(yi
t)∥2

∥∇ℓexp
t,bα(yi
t)∥2

= ⟨∇gt(yt) + bβ∗∇gt(yt)∇gt(yt)⊤(y −yt), ∇gt(yt) + bβ∗∇gt(yt)∇gt(yt)⊤(y −yt)⟩

= ∥∇gt(yt)∥2 + 2bβ∗⟨∇gt(yt), y −yt⟩∥∇gt(yt)∥2 + bβ∗2∥∇gt(yt)∥4∥y −yt∥2

≤

1 + 2bβ∗2GD
2
∥∇gt(yt)∥2 ≤4∥∇gt(yt)∥2.

Thus, we have

T
X

t=1
ℓexp
t,bα(yi
t) −

T
X

t=1
ℓexp
t,bα(x) ≤2d

bβ∗log

 bβ∗2D2

4d

T
X

t=1
∥∇gt(yt)∥2 + 1

!

+ 2

bβ∗
(57)

Substituting (39) and (57) into (56), we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

2bβ∗ln |A|
+ 2d

bβ∗log

 bβ∗2D2

4d

T
X

t=1
∥∇gt(yt)∥2 + 1

!

+ 2

bβ∗

(34)
≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2

2bβ∗ln |A|
+ 2d

bβ∗log

 bβ∗2D2H

d

T
X

t=1
ft(xt) + 1

!

+ 2

bβ∗

(58)

where the last step is due to Lemma 9. Finally, we use the following lemma to simplify the bound.

Lemma 11 (Corollary 5 of Orabona et al. [2012]) Let a, b, c, d, x > 0 satisfy x−d ≤a ln(bx+c).
Then, we have x −d ≤a ln(2(ab ln 2ab

e + db + c)).

B.4.3
Analysis for strongly convex functions

Recall that we construct the expert-loss for strongly convex functions as follows

bℓsc
t,bλ(y) = ⟨∇gt(yt), y −yt⟩+
bλ∗

2G2 ∥∇gt(yt)∥2∥y −xt∥2.

24


---Page Break---
Then, we introduce a new decomposition for λ-strongly convex functions

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−λ

2

T
X

t=1
∥xt −x∥2

≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−
bλ∗

2

T
X

t=1
∥xt −x∥2

≤

T
X

t=1
⟨∇ft(xt), xt −x⟩−
bλ∗

2G2

T
X

t=1
∥∇gt(yt)∥2∥xt −x∥2

(20)
≤

T
X

t=1
⟨∇gt(yt), yt −x⟩−∆T −
bλ∗

2G2

T
X

t=1
∥∇gt(yt)∥2∥xt −x∥2

(28)
=

T
X

t=1
⟨∇gt(yt), yt −yi
t⟩

|
{z
}
meta-regret

+

T
X

t=1


bℓsc
t,bλ∗(yi
t) −bℓsc
t,bλ∗(x)


|
{z
}
expert-regret

−
bλ∗

2G2

T
X

t=1
∥∇gt(yt)∥2∥xt −yi
t∥2 −∆T

(59)
where ∆T = PT
t=1 1{⟨∇ft(xt),vt⟩≥0} · ⟨∇ft(xt), yt −xt⟩. To bound the meta-regret, we still
incorporate with Lemma 8 to get

T
X

t=1
ft(xt) −

T
X

t=1
ft(x) ≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| +

T
X

t=1


bℓsc
t,bλ∗(yi
t) −bℓsc
t,bλ∗(x)

.

For the expert-regret, we derive a variant of theoretical guarantee of S2OGD.

Lemma 12 Under Assumptions 1 and 2, for any expert yi
t ∈Y and any x ∈X, we have

T
X

t=1
bℓsc
t,bλ∗(yi
t) −

T
X

t=1
bℓsc
t,bλ∗(x) ≤1 + (G + 2D)2

2bλ∗
log

 
bλ∗

(G + 2D)2

T
X

t=1
∥∇gt(yt)∥2 + 1

!

Combining the above bounds, we have

T
X

t=1
ft(xt) −

T
X

t=1
ft(x)

≤4ΓGD

 

2 +
1
p

ln |A|

!

+
Γ2G2

2γ ln |A| + 1 + (G + 2D)2

2bλ∗
log

 
4Hbλ∗

(G + 2D)2

T
X

t=1
ft(x) + 1

!

.

Finally, we simplify the above bound by utilizing Lemma 11.

C
Supporting Lemmas

C.1
Proof of Lemma 4

According
to
the
definition
of
ℓexp
t,bα(·)
in
(25),
we
have
∇ℓexp
t,bα(y)
=
∇gt(yt) +
bβ∇gt(yt)∇gt(yt)⊤(y −yt). Thus, for all y ∈Y, it holds that

∇ℓexp
t,bα(y)∇ℓexp
t,bα(y)⊤= ∇gt(yt)∇gt(yt)⊤+ 2bβ∇gt(yt)(y −yt)⊤∇gt(yt)∇gt(yt)⊤

+ bβ2∇gt(yt)∇gt(yt)⊤(y −yt)(y −yt)⊤∇gt(yt)∇gt(yt)⊤

=

1 + bβ⟨∇gt(yt), y −yt⟩
2
∇gt(yt)∇gt(yt)⊤

⪯4∇gt(yt)∇gt(yt)⊤= 4

bβ
∇2ℓexp
t,bα(y)

25


---Page Break---
where ∇2ℓexp
t,bα(y) denotes the Hessian matrix of ℓexp
t,bα(y) and the last inequality is due to a, and the

definition of bβ. Therefore, ℓexp
t,bα(·) is
bβ
4 -exp-concave [Hazan, 2016, Lemma 4.1]. Next, we provide
the upper bound of the gradient of ℓexp
t,bα(·) as follows

∥∇ℓexp
t,bα(y)∥2 (34)
≤(G + 2bβG2D)2 ≤25

16G2 ≤2G2.

This ends the proof.

C.2
Proof of Lemma 6

According to the definition of ℓsc
t (·) in (14), it holds for any x, y ∈X that

ℓsc
t (x) ≥ℓsc
t (y) + ⟨∇ℓsc
t (y), x −y⟩+ λ

2 ∥x −y∥2.

By Definition 1, it can be seen that ℓsc
t (·) is λ-strongly convex. Next, we provide the upper bound of
the gradient of ℓsc
t (·) as follows

∥∇ℓsc
t (y)∥2 ≤∥∇gt(yt) + λ(y −xt)∥2 (34)
≤(G + 2λD)2 ≤(G + 2D)2

where the last step is due to our assumption that λ ∈[1/T, 1].

C.3
Proof of Lemma 7

Similar to analysis of Lemma 6, for any x, y ∈X, we have

ℓsc
t,bλ(x) ≥ℓsc
t,bλ(y) + ⟨∇ℓsc
t,bλ(y), x −y⟩+
bλ
2G2 ∥∇gt(yt)∥2∥x −y∥2

By Definition 1, it is established that ℓsc
t,bλ(·) is
bλ
G2 ∥∇gt(yt)∥2-strongly convex. Next, we upper
bound the gradient of ℓsc
t,bλ(·) as follows

∥ℓsc
t,bλ(y)∥2 ≤

*

∇gt(yt) +
bλ
G2 ∥∇gt(yt)∥2(y −xt), ∇gt(yt) +
bλ
G2 ∥∇gt(yt)∥2(y −xt)

+

= ∥∇gt(yt)∥2 + 2bλ

G2 ∥∇gt(yt)∥2⟨∇gt(yt), y −xt⟩+
bλ2

G4 ∥∇gt(yt)∥4∥y −xt∥2

(34)
≤

 

1 + 2bλD

G

!2

∥∇gt(yt)∥2 ≤

1 + 2D

G

2
∥∇gt(yt)∥2

where the last step is due to our assumption that bλ ∈[1/T, 1].

C.4
Proof of Lemma 12

The analysis is similar to Wang et al. [2020b]. Let eyi
t+1 = yi
t −1

ηt ∇ℓsc
t,bα(yi
t). According to the
definition of (28), we have

ℓsc
t,k(yi
t) −ℓsc
t,k(x) ≤⟨∇ℓsc
t,k(yi
t), yi
t −x⟩−
bλ
2G2 ∥∇gt(yt)∥2∥yi
t −x∥2

= ηt⟨yi
t −eyi
t+1, yi
t −x⟩−
bλ
2G2 ∥∇gt(yt)∥2∥yi
t −x∥2.

For the first term, it can be verified that

⟨yi
t −eyi
t+1, yi
t −x⟩= ∥yi
t −x∥2 + ⟨x −eyi
t+1, yi
t −x⟩

= ∥yi
t −x∥2 −∥eyi
t+1 −x∥2 −⟨yi
t −eyi
t+1, eyi
t+1 −x⟩

= ∥yi
t −x∥2 −∥eyi
t+1 −x∥2 + ∥eyi
t+1 −yi
t∥2 + ⟨eyi
t+1 −yi
t, yi
t −x⟩

26


---Page Break---
which implies that

⟨yi
t −eyi
t+1, yi
t −x⟩= 1

2
 
∥yi
t −x∥2 −∥eyi
t+1 −x∥2 + ∥eyi
t+1 −yi
t∥2
.

Thus,
ℓsc
t,k(yi
t) −ℓsc
t,k(w) ≤ηt

2
 
∥yi
t −x∥2 −∥eyi
t+1 −x∥2

+ 1

2ηt
∥∇ℓsc
t,bα(yi
t)∥2 −
bλ
2G2 ∥∇gt(yt)∥2∥yi
t −x∥2.

Summing the above bound up over t = 1 to T, we attain

T
X

t=1
ℓsc
t,bα(yi
t) −

T
X

t=1
ℓsc
t,bα(x)

≤η1

2 ∥yi
1 −x∥2 +

T
X

t=1

 

ηt −ηt−1 −
bλ
G2 ∥∇gt(yt)∥2
!
∥yi
t −x∥2

2
+

T
X

t=1

1
2ηt
∥∇ℓsc
t,bα(yi
t)∥2

≤1 +

T
X

t=1

1
2ηt
∥∇ℓsc
t,bλ(yi
t)∥2 ≤1 + (G + 2D)2

2bλ

T
X

t=1

∥∇gt(yt)∥2

(G + 2D)2/bλ + Pt
i=1 ∥∇gi(yi)∥2 .

where the last two inequalities is due to ηt = (1 + 2D/G)2 +
bλ
G2
Pt
i=1 ∥∇gi(yi)∥2 which is
specifically set for new expert-loss. Further, we will use the following lemma to bound the last term.

Lemma 13 (Lemma 11 of Hazan et al. [2007]) Let l1,· · · ,lT and δ be non-negative real numbers.
Then, we have PT
t=1
l2
t
Pt
i=1 l2
i +δ ≤log

1
δ
PT
t=1 l2
t + 1

.

This completes the proof of Lemma 12.

C.5
Proof of Lemma 5

The analysis is similar to Mhammedi et al. [2019, Lemma 9]. When ∥byi
t+1∥≥D, then we need to
solve the following quadratic problem:

yi
t+1 = arg min
y∈Y
(byi
t+1 −y)⊤Σt+1(byi
t+1 −y).

We use the Lagrangian multiplier to solve the above problem

L(y, µ) = (byi
t+1 −y)⊤Σt+1(byi
t+1 −y) + µ(y⊤y −D2).

We set ∂L

∂y = 0 to attain Σt+1(y −byi
t+1) + µy = 0, which implies

y = (µId + Σt+1)−1Σt+1byi
t+1 = Q⊤
t+1 (xI + Λt+1)−1 Qt+1Σt+1byi
t+1

where x = µ + 1/(bβ2D2). Due to y⊤y = D2, x is the solution of the following problem

ρ(x) :=

d
X

k=1

⟨ek, Qt+1Σt+1byi
t+1⟩2

(x + λk
t )2
= D2.

D
Clarifications on bounded modulus

In this section, we explain that bounded moduli are generally acceptable in practical scenarios, which
is also stated in previous study [Zhang et al., 2022]. Taking λ-strongly convex functions as an
example, we assume that λ ∈[1/T, 1], since other cases that λ < 1/T and λ > 1 can be disregarded.
(i) If λ < 1/T, the regret bound for strongly convex functions becomes Ω(T), which cannot benefit
from strong convexity. Therefore, we should treat these functions as general convex functions. (ii) If
λ > 1, λ-strongly convex functions are also 1-strongly convex according to Definition 1. Thus, we
can treat these functions as 1-strongly convex functions.

27


---Page Break---
500
1000
1500
2000
Iterations

0

2

4

6

8

10

Regret

Ours
USC
Maler
MetaGrad
EffMetaGrad

(a) Exp-concave functions

500
1000
1500
2000
Iterations

0

5

10

15

20

25

30

Regret

Ours
USC
Maler
MetaGrad
EffMetaGrad

(b) Strongly convex functions

500
1000
1500
2000
Iterations

2

4

6

8

10

12

Regret

Ours
USC
Maler
MetaGrad
EffMetaGrad

(c) Convex functions

500
1000
1500
2000
Iterations

0

50

100

150

200

Running Time

Ours
USC
Maler
MetaGrad
EffMetaGrad

(d) Exp-concave functions

500
1000
1500
2000
Iterations

0

20

40

60

Running Time

Ours
USC
Maler
MetaGrad
EffMetaGrad

(e) Strongly convex functions

500
1000
1500
2000
Iterations

0

100

200

300

Running Time

Ours
USC
Maler
MetaGrad
EffMetaGrad

(f) Convex functions

Figure 1: Regret (first row) and running time (second row) of different methods.

E
Experiments

In this section, we conduct empirical experiments to validate the effectiveness of our proposed
methods, and present the details of experiments.

Settings
We conduct experiments on the ijcnn1 dataset from LIBSVM Data [Chang and Lin, 2011,
Prokhorov, 2001], where the dimension of features is d = 22. We consider the following online
classification problem. In each round t ∈[T], the online learner chooses a decision xt ∈X. After
submitting the decision, the online learner receives a batch of data samples {(x(i)
t , y(i)
t )}m
i=1 which
are sampled from the dataset, where x(i)
t
is the feature vector of the i-th sample, and y(i)
t
is the
corresponding label. The learner can evaluate the model by the online convex loss ft(xt) and update
the decision for the next round. In our study, we set T = 2000, the domain diameter as D = 20, and
the gradient norm upper bound as G =
√

22. Following the general setup of Zhao et al. [2022], we
set the feasible domain to be an ellipsoid X = {x ∈Rd | x⊤Ex ≤λmin(E) · (D/2)2}, where E is
a certain diagonal matrix and λmin(E) denotes its minimum eigenvalue. We remark that the cost of
one projection onto X is generally expensive since it requires solving a convex programming.

In the following, we consider three types of online convex functions to simulate the unknown
environment and demonstrate the universality of our method.
First, for exp-concave func-
tions, the online learner suffers a logistic loss: ft(xt) =
1
m
Pm
i=1 log

1 + exp(−y(i)
t x⊤
t x(i)
t )

.
Second, for strongly convex functions, we choose the regularized hinge loss:
ft(xt)
=

1
m
Pm
i=1 max

0, 1 −y(i)
t x⊤
t x(i)
t

+ λ

2 ∥xt∥2. Third, for general convex functions, the online learner

suffers the absolute loss: ft(xt) =
1
m
Pm
i=1
x⊤
t x(i)
t
−y(i)
t
. Based on the above experimental
settings, we conduct the empirical studies of our method, as well as other universal algorithms in the
literature.

Algorithms
We compare the performance of our proposed method for minimax universal regret
with existing universal algorithms, including MetaGrad [van Erven and Koolen, 2016], Maler [Wang
et al., 2019], efficient implementation of MetaGrad [Mhammedi et al., 2019], and USC [Zhang et al.,
2022]. All the baselines share the same experimental setting as our method.

28


---Page Break---
Table 2: A summary of state-of-the-art projection-free algorithms for different types of convex
functions.

Algorithm
Condition on Loss
Regret Bound
OFW [Hazan and Kale, 2012]
convex
O(T 3/4)
OSPF [Hazan and Minasyan, 2020]
convex and smooth
O(T 2/3)
SC-OFW [Wan and Zhang, 2021]
strongly convex
O(T 2/3)
AFP-ONS [Garber and Kretzu, 2023]
exp-concave and smooth
O(T 2/3)

Results
We repeat the experiments for five times and record the results in Figure 1. We conduct
the experiments on a machine with a single CPU (Apple M1 pro) and 16GB memory. We record
both regret and running time (in seconds) for all methods. As shown in Figure 1, the running time of
our method is comparable to that of EffMetaGrad, yet it achieves better results for strongly convex
functions. Compared to other algorithms which conduct O(log T) projections, i.e., USC, Maler, and
MetaGrad, the running time of our projection-efficient method is 5 to 20 times faster, and it also
attains nearly optimal regret for three types of convex functions. In conclusion, the empirical results
demonstrate the effectiveness of our method in achieving optimal regret guarantee and also significant
enhancement in computational efficiency.

F
Further discussion on projection-free algorithms

In the literature, there exists a class of projection-free algorithms [Hazan and Kale, 2012, Hazan
and Minasyan, 2020, Wan and Zhang, 2021, Wan et al., 2021b, 2022, Wang et al., 2023, Garber and
Kretzu, 2023]. Therefore, it is natural to ask whether projection-free algorithms such as variants of
Online Frank Wolfe could be used instead of OGD and ONS to remove all projections while still
being adaptive to the smoothness. Here, we provide some targeted discussions on this matter.

In fact, we can choose projection-free algorithms as the expert-algorithms. However, given the
current studies on projection-free algorithms, this approach will lead to a deterioration of the regret
bound and can not handle certain cases. As is shown in Table 2, in the literature, there are no
suitable projection-free algorithms for exp-concave functions, neither for strongly convex and smooth
functions. Moreover, when functions are smooth, existing projection-free algorithms are unable to
achieve problem-dependent bounds, such as the small-loss bounds in this work.

Finally, we would like to highlight that although using projection-free algorithms can remove all
projections, they may not achieve greater efficiency based on the universal framework. Specifically,
most projection-free algorithms, such as OFW and its variants, replace the original projection
operation with a linear optimization step. Since the universal framework requires maintaining
O(log T) expert-algorithms, this approach needs to perform O(log T) linear optimization steps per
round, which can be time-consuming when T is large.

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our contributions and scope are clearly written in the abstract and introduction.

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

Justification: See the future work in Section 5.

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

Answer: [Yes]

30


---Page Break---
Justification: See the assumptions in Section 2.1. The complete proofs can be found in
Appendix B.
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
Justification: All the information needed to reproduce the experimental results is provided.
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

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

31


---Page Break---
Answer: [No]
Justification: Due to privacy concerns, we do not include the code.
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
Justification: See the experiments in Appendix E.
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
Justification: See the experiments in Appendix E.
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
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.

32


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

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: See the experiments in Appendix E.
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
Justification: We have read the ethics carefully.
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
Justification: This work is mainly theoretical.
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

33


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
Justification: This work is mainly theoretical.
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
Justification: We cite the creators of the dataset used in our experiments properly.
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
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

34


---Page Break---
Answer: [NA]
Justification: This work does not release new assets.
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
Justification: This work does not involve crowdsourcing nor research with human subjects.
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
Justification: This work does not involve crowdsourcing nor research with human subjects.
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

35


---Page Break---
