Nearly Optimal Approximation of Matrix Functions
by the Lanczos Method

Noah Amsel∗
Tyler Chen∗
Anne Greenbaum†
Cameron Musco‡
Christopher Musco∗

Abstract

Approximating the action of a matrix function 𝑓(A) on a vector b is an increas-
ingly important primitive in machine learning, data science, and statistics, with
applications such as sampling high dimensional Gaussians, Gaussian process re-
gression and Bayesian inference, principle component analysis, and approximat-
ing Hessian spectral densities. Over the past decade, a number of algorithms en-
joying strong theoretical guarantees have been proposed for this task. Many of the
most successful belong to a family of algorithms called Krylov subspace methods.
Remarkably, a classic Krylov subspace method, called the Lanczos method for
matrix functions (Lanczos-FA), frequently outperforms newer methods in prac-
tice. Our main result is a theoretical justification for this finding: we show that,
for a natural class of rational functions, Lanczos-FA matches the error of the best
possible Krylov subspace method up to a multiplicative approximation factor. The
approximation factor depends on the degree of 𝑓(𝑥)’s denominator and the con-
dition number of A, but not on the number of iterations 𝑘. Our result provides
a strong justification for the excellent performance of Lanczos-FA, especially on
functions that are well approximated by rationals, such as the matrix square root.

1
Introduction

Given a symmetric matrix A ∈R𝑑×𝑑with eigendecomposition A = Í𝑑
𝑖=1 𝜆𝑖u𝑖uT
𝑖, the matrix function
𝑓(A) corresponding to a scalar function 𝑓: R →R is defined as

𝑓(A) :=

𝑑
∑︁

𝑖=1
𝑓(𝜆𝑖)u𝑖uT
𝑖.
(1)

Matrix functions arise throughout machine learning, data science, and statistics. For instance, the
matrix square root is used in sampling Gaussians, Bayesian modeling, and Gaussian processes [4, 5,
59], general fractional matrix powers are used in Markov chain modeling and R´enyi entropy estima-
tion [68, 42, 43, 19], the matrix logarithm is used for determinantal point processes, kernel learning,
and approximating log-determinants for Gaussian process regression and Bayesian inference [17,
39, 31], the matrix sign is used in principal components regression and spectral density estimation
[29, 18, 44, 58, 32, 69, 13, 7, 14], and the matrix exponential is used in network science [3, 67, 48].
In many of these applications, we do not need to compute the matrix 𝑓(A) itself, but rather its action
on a vector b ∈R𝑑; i.e., 𝑓(A)b. This task can be performed much more efficiently than computing
an eigendecomposition of A and forming 𝑓(A) using (1).

Perhaps the first general purpose method for approximating 𝑓(A)b is the Lanczos method for matrix
function approximation (Lanczos-FA) [22, 30], which is the focus of the present paper. Over the past
decade, a number of special purpose algorithms, designed for a single function or class of functions,

∗New York University (noah.amsel@nyu.edu, tyler.chen@nyu.edu, cmusco@nyu.edu)
†University of Washington (greenbau@uw.edu)
‡University of Massachusetts Amherst (cmusco@cs.umass.edu)

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
have been developed [23, 16, 26, 44, 11]. These newer algorithms often satisfy strong theoretical
guarantees better than the best-known bounds for Lanczos-FA. The present paper is motivated by
the following remarkable observation:

Despite being arguably the simplest and most general algorithm for computing
𝑓(A)b, Lanczos-FA frequently outperforms special purpose algorithms, some-
times by orders of magnitude, on common test problems.

For instance, in Figures 4, 5, and 8 we compare Lanczos-FA with specialized algorithms [44, 59, 11]
that satisfy the best-known theoretical guarantees for computing 𝑓(A)b for the particular functions
they were designed for. In these experiments Lanczos-FA drastically outperforms these methods,
despite the fact that it was not designed for any particular function. Because of its outstanding
performance, Lanczos-FA is perhaps the most commonly used algorithm for computing 𝑓(A)b in
practice. The main goal of this paper is to improve our theoretical understanding of why Lanczos-FA
performs so well, in order to help close the theory-practice gap.

1.1
Krylov subspace methods

Lanczos-FA falls into a class of algorithms called Krylov Subspace Methods (KSMs). KSMs are
among the most powerful and widely used algorithms for a broad range of computational tasks in-
cluding solving linear systems, computing eigenvalues/vectors, and low-rank approximation [62, 38,
47, 65]. Like other KSMs for computing 𝑓(A)b, Lanczos-FA iteratively constructs an approxima-
tion from the Krylov subspace

K𝑘(A, b) := span{b, Ab, . . . , A𝑘−1b}.
(2)

The Lanczos algorithm [46] produces an orthonormal basis Q = [q1, . . . , q𝑘] for the Krylov sub-
space K𝑘(A, b) and a symmetric tridiagonal matrix T satisfying T = QTAQ [62]. The Lanczos-FA
algorithm uses this Q and T to approximate 𝑓(A)b.
Definition 1. The Lanczos-FA iterate for a problem instance ( 𝑓, A, b, 𝑘) is defined as

lan𝑘( 𝑓; A, b) := Q 𝑓(T)QTb,

where Q and T are as above.

In our analysis we assume exact arithmetic. We do not discuss the implementation of Lanczos or
Lanczos-FA since there are many resources on this topic; see for instance [52, 8, 9]. Fortunately,
if the Lanczos algorithm is implemented properly, its finite-precision behavior closely follows its
exact-arithmetic behavior on a nearby problem [33]. See Section 5 for further discussion.

1.2
Optimality guarantees for Krylov subspace methods

For a wide variety of problem instances, Lanczos-FA is observed to converge almost as quickly
as the best approximation to 𝑓(A)b that could be returned by any KSM run for the same number
of iterations. In particular, all KSMs output approximations that lie in the span of K𝑘(A, b), that
is, approximations of the form 𝑝(A)b for a polynomial 𝑝of degree less than 𝑘. Given a problem
instance ( 𝑓, A, b, 𝑘), the best possible approximation returned by a Krylov method is4

opt𝑘( 𝑓; A, b) := argmin
x∈K𝑘(A,b)
∥𝑓(A)b −x∥2.

By definition, the error of this Krylov optimal iterate can be characterized as follows:

∥𝑓(A)b −opt𝑘( 𝑓; A, b)∥2 =
min
deg( 𝑝)<𝑘∥𝑓(A)b −𝑝(A)b∥2.
(3)

For general matrix functions, no efficient algorithm for computing opt𝑘( 𝑓; A, b) is known, but as
shown in Figure 1, the solution returned by Lanczos-FA often nearly matches the error of this best
approximation. It is thus natural to ask if, at least for some class of problems, Lanczos-FA satisfies
the following strong notion of approximate optimality:

4We choose to measure error in the Euclidean norm, and thus define optimality with respect to this norm.
While other norms have been considered in prior work on matrix-function approximation [11], the Euclidean
norm a simple starting point.

2


---Page Break---
Figure 1: Lanczos-FA error ∥𝑓(A) −lan𝑘( 𝑓; A, b)∥2 at each iteration for several functions/spectra.
“Instance Optimal” is the right hand side of Definition 2 with 𝐶= 𝑐= 1, which is a lower bound for
all KSMs, including Lanczos-FA. Lanczos-FA performs nearly instance optimally on a wide range
of problems, far better than Fact 3 predicts. This is easily seen in the bottom plots, which show the
ratio of the error of the Lanczos-FA iterate and the Krylov optimal iterate, opt𝑘( 𝑓; A, b).

Definition 2 (Near Instance Optimality). For a problem instance ( 𝑓, A, b, 𝑘), we say that a Krylov
method is nearly instance optimal with parameters 𝐶and 𝑐if

∥𝑓(A)b −alg𝑘( 𝑓; A, b)∥2 ≤𝐶
min
deg( 𝑝)<𝑐𝑘∥𝑓(A)b −𝑝(A)b∥2.

Above, alg𝑘( 𝑓; A, b) denotes the output of an algorithm (e.g., Lanczos-FA) obtained from the Krylov
subspace K𝑘(A, b), i.e., the output after 𝑘iterations.

In Definition 2, 𝐶≥1 and 𝑐≤1 allow some slack in comparing to the Krylov optimal iterate.
The right hand side depends on the entire problem instance, ( 𝑓; A, b), which is why we call the
guarantee “instance optimal”.

1.3
Existing near-optimality analyses of Lanczos-FA

The best bound for Lanczos-FA applying to a broad class of functions is the following common
bound for Lanczos-FA; see for instance [61, 54, 52, 10].
Fact 3. For all problem instances ( 𝑓, A, b, 𝑘), Lanczos-FA satisfies

∥𝑓(A)b −alg𝑘( 𝑓; A, b)∥2 ≤2∥b∥2
min
deg( 𝑝)<𝑘


max
𝑥∈[𝜆min,𝜆max] | 𝑓(𝑥) −𝑝(𝑥)|

.

Fact 3 is in some sense an optimality guarantee; it compares the convergence of Lanczos-FA to the
best possible uniform polynomial approximation to 𝑓(𝑥). However, it does not take into account
properties of A such as isolated or clustered eigenvalues, and as seen in Figure 1, it typically only
gives a loose upper bound on the performance of Lanczos-FA.

To date, near-instance-optimality guarantees for Lanczos-FA akin to those of Definition 2 are known
only in a few special cases. The most well-known is when 𝑓(𝑥) = 1/𝑥and A is positive definite,
in which case Lanczos-FA is mathematically equivalent to the celebrated Conjugate Gradient al-
gorithm, and therefore exactly optimal in the A-norm [47]. The instance-optimality guarantee for
Lanczos-FA in this setting is used to prove well-known super-exponential convergence in certain
settings [6, 47, 8]. In contrast, a bound like Fact 3 only provides exponential convergence and is
widely understood by the numerical linear algebra community to not accurately describe the actual
behavior of the algorithm in most cases [35, 47, 8]. The only other near-optimality guarantees for
Lanczos-FA of which we are aware concern A−1b for nonsymmetric A [12] and the matrix exponen-
tial exp(−𝑡A)b [20]. In both cases, Lanczos-FA satisfies guarantees that are reminiscent of (although
weaker than) Definition 2. Further discussion is given in Appendix B.1.

3


---Page Break---
We also remark that there are a number of works which aim to relate the convergence of of Lanczos-
FA for functions with certain integral representations to the convergence of conjugate gradient on
linear systems [41, 24, 27, 28, 25, 10]. While these analyses provides spectrum dependent conver-
gence guarantees, they are weaker than Definition 2 because it is not clear how the best possible
KSM approximation to a given function relates to the convergence of conjugate gradient. These
bounds were mostly developed for use as a posteriori stopping criteria rather than as a theoretical
explication for the behavior of Lanczos-FA.

1.4
Our contributions

In Section 2, we prove near instance optimality for a broad class of rational functions (Theorem 4).
To the best of our knowledge, this is the first true instance-optimality guarantee for Lanczos-FA to be
proven for any function besides 𝑓(𝑥) = 1/𝑥. In Section 2.2, we discuss how results of this kind imply
related guarantees for functions that are uniformly well-approximated by rationals, which includes
many functions of interest in machine learning. In Section 3, we additionally show that Lanczos-
FA satisfies a weaker version of near optimality for two crucial non-rational matrix functions—the
square root and inverse square root (Theorems 6 and 7). Appendix C compares the this version
of optimality to Definition 2 (near instance optimality) and to that of Fact 3. In Section 4, we
present experimental evidence showing that, for many natural problem instances, our bounds are
significantly sharper than the standard bound of Fact 3. These experiments also demonstrate that
despite its generality, Lanczos-FA often converges significantly faster than other methods in practice.
We conclude with a discussion of next steps and open problems in Section 5.

2
Near optimality for rational functions

In this section, we study the Lanczos-FA approximation to 𝑟(A)b, where 𝑟(𝑥) is a rational function
with real-valued poles that lie in R \ I, where I := [𝜆min, 𝜆max]. Specifically,

𝑟(𝑥) := 𝑛(𝑥)/𝑚(𝑥),
(4)

where 𝑛(𝑥) is any degree 𝑝polynomial and

𝑚(𝑥) = (𝑥−𝑧1)(𝑥−𝑧2) · · · (𝑥−𝑧𝑞),
𝑧𝑖∈R \ I.

Since 𝑧𝑗∉I, ±(A −𝑧𝑗I) is either positive definite or negative definite. For convenience, we define

A 𝑗:=
+(A −𝑧𝑗I)
𝑧𝑗< 𝜆min
−(A −𝑧𝑗I)
𝑧𝑗> 𝜆max
.
(5)

Our main result on rational functions is the following near-instance-optimality bound, which holds
under a mild assumption on the number of iterations 𝑘:
Theorem 4. Let 𝑟(𝑥) = 𝑛(𝑥)/𝑚(𝑥) be a degree (𝑝, 𝑞)-rational function as in (4) and define A 𝑗as
in (5). Then, if 𝑘> max{𝑝, 𝑞−1}, the Lanczos-FA iterate satisfies the bound

∥𝑟(A)b −lan𝑘(𝑟; A, b)∥2 ≤𝑞· 𝜅(A1) · 𝜅(A2) · · · 𝜅(A𝑞)
min
deg( 𝑝)<𝑘−𝑞+1 ∥𝑟(A)b −𝑝(A)b∥2.

We prove this theorem in Appendix A. Above, 𝜅(A𝑖) is the condition number of A𝑖, the ratio of
the largest to smallest magnitude eigenvalues of A𝑖. Theorem 4 shows that Lanczos-FA used to
approximate 𝑟(A)b satisfies near instance optimality, as in Definition 2, with

𝐶= 𝑞· 𝜅(A1) · 𝜅(A2) · · · 𝜅(A𝑞),
𝑐= 1 −(𝑞−1)/𝑘.

In particular, when A is positive definite and each of the 𝑧𝑖are negative (as is the case for rational
function approximations of many functions including the square root [37]), then 𝐶≤𝑞𝜅(A)𝑞.

As discussed in Section 5, we expect that Theorem 4 can be tightened by significantly reducing the
prefactor. Nevertheless, as illustrated in Figure 2, even in its current form, the bound improves on
the standard bound of Fact 3 in many natural cases; i.e., it more tightly characterizes the observed
convergence of Lanczos-FA. Finally, as discussed further in Section 2.2, we note that, beyond ra-
tional functions being an interesting function class in their own right, Theorem 4 has implications
for understanding the convergence of Lanczos-FA for functions like the square root which are much
easier to approximate with rational functions than polynomials.

4


---Page Break---
Figure 2: Despite its large prefactor, the bound of Theorem 4 qualitatively captures the convergence
behavior of Lanczos-FA for rational functions. It can be tighter than the standard bound of Fact 3,
even for a moderate number of iterations 𝑘. We use rational approximations to exp(−𝑥/10) and
log(𝑥) for comparison with Figure 1; see Section 4 for more details.

2.1
Proof Sketch

Our proof of Theorem 4 leverages the near optimality of Lanczos-FA in computing A−1b to obtain a
bound for general rational functions. For illustration, consider the simplest possible rational function
for which we are unaware of any previous near-optimality bounds: A−2b when A is positive definite.
The Lanczos-FA approximation for this function is QT−2QTb = QT−1QTQT−1QTb. Using the
triangle inequality and submultiplicativity, we can bound the error of the approximation as

∥A−2b −lan𝑘(𝑥−2; A, b)∥2

≤∥A−2b −QT−1QTA−1b∥2 + ∥QT−1QTA−1b −QT−2QTb∥2

≤∥A−2b −QT−1QTAA−2b∥2 + ∥QT−1QT∥2 · ∥A−1b −QT−1QTAA−1b∥2.
(6)

Using the normal equations and the fact that Q is a basis of the Krylov subspace, it is possible to
show that QT−1QTA = Q(QTAQ)−1QTA is the A-norm5 projector onto Krylov subspace. Hence

∥A−2b −QT−1QTAA−2b∥A =
min
x∈K𝑘(A,b) ∥A−2b −x∥A

:=
min
deg( 𝑝)<𝑘∥A−2b −𝑝(A)b∥A.

Therefore, using ∥x∥2 ≤(1/√𝜆min)∥x∥A and ∥x∥A ≤√𝜆max∥x∥2, the first term on the right hand
side of (6) can be bounded as

∥A−2b −QT−1QTAA−2b∥2 ≤
√︁

𝜅(A)
min
deg( 𝑝)<𝑘∥A−2b −𝑝(A)b∥2.

Similarly, the second factor of the second term on the right hand side of (6) can be bounded as

∥A−1b −QT−1QTAA−1b∥2 ≤
√︁

𝜅(A)
min
deg( 𝑝)<𝑘∥A−1b −𝑝(A)b∥2.

Then, since 𝑥𝑝(𝑥) is polynomial of one degree larger than 𝑝(𝑥),

min
deg( 𝑝)<𝑘∥A−1b −𝑝(A)b∥2 ≤
min
deg( 𝑝)<𝑘−1 ∥AA−2b −A𝑝(A)b∥2

≤𝜆max
min
deg( 𝑝)<𝑘−1 ∥A−2b −𝑝(A)b∥2.

Plugging the above bounds into (6) and using the fact that the eigenvalues of T are contained in
[𝜆min, 𝜆max] so that ∥QT−1QT∥2 ≤∥T−1∥2 ≤1/𝜆min,

∥A−2b −lan𝑘(𝑥−2; A, b)∥2 ≤2𝜅(A)3/2
min
deg( 𝑝)<𝑘−1 ∥A−2b −𝑝(A)b∥2.

5The A-norm is defined for positive definite A by ∥x∥A = ∥A1/2x∥2.

5


---Page Break---
Bounding 𝜅(A)3/2 by 𝜅(A)2 gives the bound in Theorem 4.

Our proof of Theorem 4 generalizes the above approach. We write the Lanczos-FA error for ap-
proximating 𝑟(A)b in terms of the error of the optimal approximations to a set of simpler rational
functions, and then reduce polynomial approximation of 𝑟(𝑥) to polynomial approximation of each
of these particular functions.

2.2
Implications for non-rational functions

Theorem 4 can be used to derive guarantees for other (non-rational) functions. In particular, consider
any function 𝑓(𝑥) that is uniformly well approximated by a low-degree rational function 𝑟(𝑥) on
I = [𝜆min, 𝜆max], the interval containing all of the eigenvalues of A; i.e., suppose ∥𝑟−𝑓∥I :=
max𝑥∈I |𝑟(𝑥)−𝑓(𝑥)| is small. A natural way to approximate 𝑓(A)b, used in [4, 5, 59], is to construct
𝑟(𝑥) and output 𝑟(A)b, using some iterative linear solver to quickly apply the denominators of the
partial fraction decomposition of 𝑟(𝑥). Alternatively, if we have an instance-optimality guarantee
for Lanczos-FA on rational functions like Theorem 4, we could use Lanczos-FA to compute 𝑟(A)b.
The following analysis shows that simply using Lanczos-FA on 𝑓(𝑥) itself cannot be much worse.
Lemma 5. Assume the we have the following instance-optimality guarantee for rational functions:

∥𝑟(A)b −lan𝑘(𝑟; A, b)∥2 ≤𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘∥𝑟(A)b −𝑝(A)b∥2.
(7)

Here, 𝐶𝑟and 𝑐𝑟depend on the choice of approximant 𝑟(𝑥). Then the error of Lanczos-FA on 𝑓(𝑥)
is bounded as follows:

∥𝑓(A)b −lan𝑘( 𝑓; A, b)∥2

≤min
𝑟


(𝐶𝑟+ 2)∥b∥2 · ∥𝑓−𝑟∥I + 𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘∥𝑓(A)b −𝑝(A)b∥2

.
(8)

We prove this lemma, which follows directly from the triangle inequality, in Appendix A.5. Com-
pare the form of this bound to that of Definitions 2 and 13. It is close to a near-instance-optimality
guarantee, except for the first term, which requires 𝑓(𝑥) to be uniformly well-approximated by a
rational function 𝑟(𝑥) on [𝜆min, 𝜆max]. This is still much stronger than Fact 3, which requires 𝑓(𝑥)
to be uniformly well-approximated by a polynomial to guarantee that Lanczos-FA provides a good
approximation. There are many functions with lower degree rational approximations than polyno-
mial approximations, even when we require the rational function 𝑟(𝑥) to have poles only in R \ I
(as in our Theorem 4). Such rational approximations are obtainable by the Remez algorithm [64,
Chapter 24], and for many important functions they are also known explicitly. For example, a uni-
form polynomial approximation to the square root on a strictly positive interval [𝜆min, 𝜆max] requires
degree Ω(
√︁

𝜆max/𝜆min) [64, Chapter 8]. On the other hand, a uniform rational approximation can be
obtained with degree only 𝑂(log(𝜆max/𝜆min)) [37, 64, Chapter 25]. Likewise, a uniform polynomial
approximation to exp(−𝑥) on the interval [0, 𝐵] requires degree Ω(
√

𝐵) [1], but uniform rational ap-
proximations can be constructed with no dependence on 𝐵[63]. For such functions, we expect (8)
to be stronger than Fact 3.

Notice also that, while in Lemma 5, we assume 𝑓(𝑥) is well-approximated by a rational function, we
are not required to actually construct the approximation. Indeed, since it holds for any 𝑟(𝑥), instead
of fixing a rational approximation of a certain degree, (8) automatically balances ∥𝑟−𝑓∥I, which
decreases as the degree grows, with 𝐶𝑟, which may increase as the degree grows (see Figure 4).

3
Near Spectrum Optimality for A±1/2b

In the previous section, we proved that Lanczos-FA is nearly instance optimal for rational functions
in the sense of Definition 2. In this section, we prove that Lanczos-FA satisfies a weaker form of
near optimality for two important non-rational functions: square root and inverse square root. We
term this weaker form of guarantee “near spectrum optimality”. In Appendix C, we formally define
this notion and compare it to Definition 2. We first state our bound for the inverse square root.
Theorem 6. Let Λ be the spectrum of A. Then for 𝑘≥2, the Lanczos-FA iterate satisfies the bound

∥A−1/2b −lan𝑘(𝑥−1/2; A, b)∥2 ≤
3
√

𝜋𝑘
𝜅(A)∥b∥2
min
deg( 𝑝)<𝑘/2


max
𝑥∈Λ


1
√𝑥−𝑝(𝑥)



.

6


---Page Break---
That is, Lanczos-FA applied to the inverse square root satisfies Definition 12 (“near spectrum opti-
mality”) with

𝐶=
3
√

𝜋𝑘
𝜅(A),
𝑐= 1

2.

We prove this theorem in Appendix D. The proof relies on comparing the error of the 𝑘th Lanczos
iterate for 𝑥−1/2 to that of the Lanczos iterate for 𝑥−1. First, applying a bound from [10], we use
the Cauchy integral formula to upper bound the error of Lanczos-FA on 𝑥±1/2 by its error on 𝑥−1
(Lemma 16). Second, as Equation (17) shows, Lanczos-FA is nearly instance optimal for the func-
tion 𝑥−1; that is, it outputs 𝑝(A)b where 𝑝is (nearly) the degree 𝑘polynomial that best approximates
𝑥−1. Third, the best degree 𝑘polynomial approximation to 𝑥−1 must have lower error than the best
degree 𝑘/2 approximation to 𝑥−1/2. This is because any degree 𝑘/2 approximation to 𝑥−1/2 can be
squared to yield a good degree 𝑘polynomial approximation to 𝑥−1. Combining these three steps
upper bounds the error of the 𝑘th Lanczos-FA iterate for 𝑥−1/2 by the error of the best degree 𝑘/2
polynomial approximation of 𝑥−1/2.

Nearly the same argument can be used to prove spectrum optimality of Lanczos-FA for the function
A−1/𝑛b for any 𝑛∈N with 𝐶= (2𝑛−1) · 𝜅(A)/
√

𝜋𝑘. Furthermore, using Lemma 15 of Ap-
pendix C, we can convert Theorem 6 into a near-instance-optimality guarantee at the price of strong
dependence of 𝐶on b. We next state our optimality result for the matrix square root.

Theorem 7. Let Λ be the spectrum of A. Then for 𝑘≥2, the Lanczos-FA iterate satisfies the bound

∥A1/2b −lan𝑘(𝑥1/2; A, b)∥≤3𝜅(A)2

𝑘3/2
∥b∥2
min
deg( 𝑝)<𝑘/2+1


max
𝑥∈Λ∪{0}

√𝑥−𝑝(𝑥)


.

This bound resembles Definition 12 with

𝐶= 3𝜅(A)2

𝑘3/2
,
𝑐= 1

2.

However, it is slightly weaker in that the maximization is taken over Λ ∪{0} instead of only Λ.

The proof of Theorem 7 is nearly the same as that of Theorem 6, and it likewise appears in Ap-
pendix D. Ideally, if 𝑝is a polynomial approximation to 𝑥1/2, we would like to claim that (𝑝(𝑥)/𝑥)2
yields a good polynomial approximation to 𝑥−1. However, since this function is not necessarily a

polynomial, we must instead use use

𝑝(𝑥)−𝑝(0)

𝑥
2
, which introduces the need to include {0} in the
maximization on the right-hand side.

4
Experiments

We now present several numerical experiments to assess the quality of our instance-optimality
bounds, Theorem 4 and Lemma 5. Our results show that, despite the large prefactor 𝐶, our bounds
already supersede the standard uniform approximation bound (Fact 3) in many cases. We also com-
pare Lanczos-FA against several recently proposed algorithms for computing matrix functions with
strong theoretical guarantees. We find that, in practice, Lanczos-FA performs better than all of them.
We implement Lanczos-FA in high precision arithmetic using the flamp library, which is based on
mpmath [51], in order to mitigate any potential impacts of finite precision arithmetic and observe the
convergence behavior of the algorithm beyond the standard machine precision.6

In Figure 1, we compare the performance of Lanczos to the instance-optimal KSM (which we can
compute by direct methods) and against Fact 3 for various matrix functions and spectra. We use three
test matrices A ∈R100×100 which all have condition number 100. The first has a uniformly-spaced
spectrum, the second has a geometrically-spaced spectrum, and the third has eigenvalues that are all
uniformly-spaced in a narrow interval except for ten very small eigenvalues. We compute the bound
from Fact 3 using the Remez algorithm and compute the instance-optimal approximation using least
squares regression onto the Krylov basis Q. In Figure 1, as in almost all cases we tried, Lanczos-FA

6Code for our experiments is available at https://github.com/NoahAmsel/lanczos-optimality/
tree/neurips2024_near_optimality..

7


---Page Break---
Figure 3: The maximum observed ratio between the error of Lanczos-FA and the optimal error over
choices of b when approximating A−𝑞for matrices with varying condition number 𝜅. Each point
corresponds to a pair (𝜅, 𝑞). Points with the same color have the same value of 𝜅. On the left, the
dotted line plots √𝑞𝜅for the maximum 𝜅considered (106). On the right, the dotted line plots √𝑞𝜅
for the maximum 𝑞considered (26). Overall, the optimality ratio appears to scale at least as Ω(√𝑞𝜅).

performs nearly as well as the instance-optimal approximation. For instance, the error is never more
than a small multiple of the optimal error in the experiments we did.

To better understand Theorem 4, we compare the bound to the true convergence curve of Lanczos-FA
for various rational functions in Figure 2. We also plot Fact 3 for reference. We use the same matrices
and b vectors as in Figure 1; results are similar if b is instead chosen as a uniform linear combination
of A’s eigenvectors. We choose rational functions to match the functions used for Figure 1. We
construct a degree 5 rational approximation to exp(−𝑥/10) following [63]. We construct a degree 10
approximation to log(𝑥) using the BRASIL algorithm [40] and verify that it has real poles outside
the interval of the eigenvalues. Despite Theorem 4’s exponential dependence on the degree of the
rational function being applied, Figure 2 shows that it matches the shape of the convergence curve
well and is tighter than Fact 3 when the number of iterations is large. That said, in all cases plotted,
Lanczos-FA always returns an approximation much closer to optimal than predicted by Theorem 4,
suggesting that the leading coefficient in our bound is pessimistic.

4.1
Dependence on the rational function degree

Theorem 4 upper bounds the optimality ratio by 𝐶= 𝑞· 𝜅(A1) · 𝜅(A2) · · · 𝜅(A𝑞). We conjecture
that this bound is far from tight. However, the following experiment provides evidence that it is
not possible to entirely eliminate the dependence on the rational function’s denominator degree 𝑞.
In particular, for parameters (𝜅, 𝑞), consider approximating A−𝑞b where A has spectrum 𝜆1 = 1
and 𝜆2, . . . , 𝜆100 evenly spaced between 0.999995 · 𝜅and 𝜅. In this case, 𝜅(A1) = · · · = 𝜅(A𝑞) =
𝜅(A) = 𝜅. We generate a grid of problems by picking different combinations of (𝜅, 𝑞) and tuning b
in a limited way to maximize the maximum ratio between the error of Lanczos-FA and the optimal
Krylov error over all iterations. In particular, we took b to be an all ones vector, except we varied
its first entry, using grid search to maximize the optimality ratio. The results, plotted in Figure 3,
suggest that the optimality ratio grows at least as Ω(
√︁

𝑞· 𝜅(A)). We have yet to find harder problem
instances than this.

4.2
Non-rational functions

As noted in Section 2.2, an instance-optimality guarantee for rational functions also implies that
Lanczos-FA performs well on functions that are well-approximated by rationals. As an example,
we consider the function 𝑓(𝑥) = 𝑥−0.4, for which a rational approximation in any degree can be
found using the BRASIL algorithm [40]. Figure 4 shows how applying Lanczos-FA to these ra-
tional approximations compares to applying Lanczos-FA directly to the 𝑓(𝑥) itself to approximate
A−0.4b. When the number of iterations is small, both methods perform nearly optimally, as the
accuracy is limited more by the small size of the Krylov subspace than by the difference between
𝑓(𝑥) and the rational approximant (that is, the second term in (8) dominates the first term). As the
number of iterations grows, the error due to approximating 𝑓(𝑥) in the Krylov subspace continues to
decrease while the error of uniformly approximating 𝑓(𝑥) by some fixed rational function remains
fixed (that is, the first term in (8) dominates the first term); however, increasing the degree of the

8


---Page Break---
Figure 4: Applying Lanczos-FA to the function A−0.4 and rational approximations of various de-
grees found using the BRASIL algorithm [40]. In this experiment, the spectrum of A contains two
clusters: 10 eigenvalues uniformly spaced near 1, and 90 eigenvalues uniformly spaced near 100.
As predicted by the bound in Section 2.2, convergence of Lanczos-FA for this function appears to
closely track that of a high degree rational approximant.

Figure 5: A comparison of Lanczos-FA with two methods from [44] (“rational” and “slanczos”)
for computing the matrix sign function, which work by using a stochastic iterative method to ap-
proximate rational approximations to the step function of various degrees. The “rational” method is
the main one studied in [44], while “slanczos” is included because it is the best performing in their
experiments. Each panel corresponds to one of the test problems from [44]. Iterations of these meth-
ods are counted in number of inner products with rows of A rather than number of matrix-vector
products with A as a whole. To compare these with Lanczos-FA, we consider 𝑑such inner products
to be equivalent to one matrix-vector product.

rational approximant decreases this source of error. This shows that understanding the convergence
of Lanczos-FA for the entire family of rational approximations goes a long way toward explaining
its convergence for non-rational functions. In the limit, Lanczos-FA applied to 𝑓(𝑥) itself appears
to automatically inherit the instance optimality of a suitably high-degree rational approximation.

This result has an additional implication. A number of papers use explicit rational approximations
to compute non-rational matrix functions [4, 5, 36, 59]. These approximations are often applied
by applying conjugate gradient (or a related method) to each of the terms in the sum [36, 59]. In
the case conjugate gradient is used, the resulting algorithm is mathematically identical to Lanczos-
FA used to compute the the rational approximation. However, Figure 4 suggests that simply using
Lanczos-FA on the original function is both simpler and converges faster (though memory usage
and other factors may need to be considered).

Another line of work uses specialized iterative methods that exploit problem structure to apply the
rational approximations [29, 2, 53, 44]. In Figure 5, we compare Lanczos-FA to two such methods
from [44] for computing sign(A), for A of the form A = BTB −𝜆I. While they achieve better
theoretical bounds than are known for Lanczos-FA, Lanczos-FA far outperforms them on the test
problems used in [44].

9


---Page Break---
Figure 6: Convergence of Lanczos-FA for rational functions with poles in A’s eigenvalue range or
that are imaginary. The optimality ratio can be very large for some iterations. Similar behavior is
seen for functions like sign(A) that have a discontinuity in the interval of A’s eigenvalues. However,
the “overall” convergence of Lanczos-FA still appears to closely track the instance-optimal solution.

Additional Experiments.
In Section 3 we introduce bounds for the matrix square root and inverse
square root, and in Appendix E.1 we provide numerical tests to study the sharpness of the bounds,
verifying that they can improve on Fact 3. In Appendix E.2, we demonstrate the convergence be-
havior of Lanczos-FA on rational functions with poles inside the range of eigenvalues (Figure 6).
This illustrates why a bound like Theorem 4 is not possible, but suggests a weaker bound, such as
the bound in [12] for 𝑓(𝑥) = 1/𝑥and indefinite A, may be possible. Appendix E.3 shows that, un-
like Lanczos-FA, a related algorithm called Lanczos-OR [11] (which is exactly optimal for rational
functions, though not in the Euclidean norm) can perform poorly on high degree rational functions
when the error is measured in the Euclidean norm.

5
Outlook

This paper provides instance-optimality guarantees for Lanczos-FA applied to a range of rational
functions. We conclude with open questions that we believe are worthy of further study.

Extension to other function classes.
Empirically, Lanczos-FA seems to be nearly instance opti-
mal for a wide variety of functions beyond those considered in this paper, such as rational functions
with conjugate pairs of complex poles whose real parts lie outside [𝜆min, 𝜆max]. As seen in Fig-
ure 6, the error of Lanczos-FA on functions with real poles in [𝜆min, 𝜆max] is intriguing, oscillating
between being very large and nearly optimal. We discuss this more in Appendix E.2. It would be
valuable to provide bounds explaining these behaviors.

It would be also natural to try to extend our bounds to Stieltjes/Markov functions, which can be
viewed as a certain type of infinite degree rational function approximations with poles in (−∞, 0],
and includes important functions like the inverse square root and a shifted logartihm. Our bound
Theorem 4 cannot be directly applied to this class due to the dependence on the rational function
denominator degree 𝑞.

Construction of hard instances / refined upper bounds.
Theorem 4 has an exponential depen-
dence on the degree of the rational function’s denominator 𝑞, which limits the practicality of our
bounds. It is unclear if and when this dependence can be improved. The experiment in Section 4.1
provides strong evidence that some dependence on 𝑞is necessary, but the hardest examples we
have depend on √𝑞, instead of the current bound of 𝜅(A)𝑞guaranteed by Theorem 4. It is an open
question whether Theorem 4 can be tightened, or whether matching hard instances exist.

Finite precision arithmetic.
Our analysis concerns the behavior of the Lanczos algorithm when
run in exact arithmetic. In practice, the implementation of the Lanczos algorithm is very important;
for instance, practical implementations often output a Q which is far from orthogonal [50, 8]. While
this instability can be mitigated with more expensive implementations, theoretical work shows that,
surprisingly, Lanczos and Lanczos-FA can work well despite it [57, 56, 55, 33, 21, 20]. For example,
[21, 52, 9] show that Fact 3 still holds up to a close approximation in finite precision arithmetic
for any bounded matrix function. It would be valuable to study whether stronger near-optimality
guarantees like those proven in Theorem 4 are also robust to finite precision. For 𝑓(𝑥) = 1/𝑥, this
problem has been studied in [33].

10


---Page Break---
Acknowledgments:
This research was supported in part by NSF Awards 2045590 (Chen, Ch.
Musco), 2046235 (Ca. Musco), 2427363 (Chen, Ca. Musco, Ch. Musco), and 2234660 (Amsel).

References

[1]
Amol Aggarwal and Josh Alman. “Optimal-Degree Polynomial Approximations for Expo-
nentials and Gaussian Kernel Density Estimation”. In: 37th Computational Complexity Con-
ference (CCC 2022). 2022 (cit. on p. 6).
[2]
Zeyuan Allen-Zhu and Yuanzhi Li. “Faster Principal Component Regression and Stable Ma-
trix Chebyshev Approximation”. In: Proceedings of the 34th International Conference on
Machine Learning. Vol. 70. 2017, pp. 107–115 (cit. on p. 9).
[3]
Mary Aprahamian, Desmond J. Higham, and Nicholas J. Higham. “Matching exponential-
based and resolvent-based centrality measures”. In: Journal of Complex Networks 4.2 (2015),
pp. 157–176. eprint: https://academic.oup.com/comnet/article-pdf/4/2/157/
6716204/cnv016.pdf (cit. on p. 1).
[4]
Erlend Aune, Jo Eidsvik, and Yvo Pokern. “Iterative numerical methods for sampling from
high dimensional Gaussian distributions”. In: Statistics and Computing 23 (2013), pp. 501–
521 (cit. on pp. 1, 6, 9).
[5]
Erlend Aune, Daniel P Simpson, and Jo Eidsvik. “Parameter estimation in high dimensional
Gaussian distributions”. In: Statistics and Computing 24 (2014), pp. 247–263 (cit. on pp. 1,
6, 9).
[6]
Bernhard Beckermann and Arno B. J. Kuijlaars. “Superlinear Convergence of Conjugate Gra-
dients”. In: SIAM Journal on Numerical Analysis 39.1 (2001), pp. 300–329 (cit. on pp. 3, 18).
[7]
Vladimir Braverman, Aditya Krishnan, and Christopher Musco. “Sublinear time spectral den-
sity estimation”. In: Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of
Computing (STOC 2022). 2022 (cit. on pp. 1, 21).
[8]
Erin Carson, J¨org Liesen, and Zdenˇek Strakoˇs. “Towards understanding CG and GMRES
through examples”. In: Linear Algebra and its Applications 692 (2024), pp. 241–291 (cit. on
pp. 2, 3, 10, 18).
[9]
Tyler Chen. The Lanczos algorithm for matrix functions: a handbook for scientists. 2024.
arXiv: 2410.11090 [math.NA] (cit. on pp. 2, 10).
[10]
Tyler Chen, Anne Greenbaum, Cameron Musco, and Christopher Musco. “Error Bounds for
Lanczos-Based Matrix Function Approximation”. In: SIAM Journal on Matrix Analysis and
Applications 43.2 (2022), pp. 787–811 (cit. on pp. 3, 4, 7, 22).
[11]
Tyler Chen, Anne Greenbaum, Cameron Musco, and Christopher Musco. “Low-Memory
Krylov Subspace Methods for Optimal Rational Matrix Function Approximation”. In: SIAM
Journal on Matrix Analysis and Applications 44.2 (2023), pp. 670–692 (cit. on pp. 2, 10, 19,
24).
[12]
Tyler Chen and G´erard Meurant. “Near-optimal convergence of the full orthogonalization
method”. In: Electronic Transactions on Numerical Analysis 60 (2024), pp. 421–427. arXiv:
2403.07259 [math.NA] (cit. on pp. 3, 10, 18, 24).
[13]
Tyler Chen, Thomas Trogdon, and Shashanka Ubaru. “Analysis of stochastic Lanczos quadra-
ture for spectrum approximation”. In: Proceedings of the 37th International Conference on
Machine Learning (ICML 2021). 2021 (cit. on pp. 1, 21).
[14]
Tyler Chen, Thomas Trogdon, and Shashanka Ubaru. Randomized matrix-free quadrature for
spectrum and spectral sum approximation. 2022. arXiv: 2204.01941 [math.NA] (cit. on
pp. 1, 21).
[15]
Jane Cullum and Anne Greenbaum. “Relations between Galerkin and Norm-Minimizing It-
erative Methods for Solving Linear Systems”. In: SIAM Journal on Matrix Analysis and Ap-
plications 17.2 (1996), pp. 223–247 (cit. on p. 18).
[16]
Nigel Cundy, Stefan Krieg, Guido Arnold, Andreas Frommer, Th Lippert, and Klaus
Schilling. “Numerical methods for the QCD overlap operator IV: Hybrid Monte Carlo”. In:
Computer physics communications 180.1 (2009), pp. 26–54 (cit. on p. 2).

11


---Page Break---
[17]
Kun Dong, David Eriksson, Hannes Nickisch, David Bindel, and Andrew G Wilson. “Scalable
Log Determinants for Gaussian Process Kernel Learning”. In: Advances in Neural Informa-
tion Processing Systems (NeurIPS 2017). Vol. 30. 2017 (cit. on p. 1).
[18]
Kun Dong, David Eriksson, Hannes Nickisch, David Bindel, and Andrew G Wilson. “Scalable
Log Determinants for Gaussian Process Kernel Learning”. In: Advances in Neural Informa-
tion Processing Systems (NeurIPS 2017). Vol. 30. 2017 (cit. on p. 1).
[19]
Yuxin Dong, Tieliang Gong, Shujian Yu, and Chen Li. “Optimal Randomized Approxima-
tions for Matrix-Based R´enyi’s Entropy”. In: IEEE Transactions on Information Theory 69.7
(2023), pp. 4218–4234 (cit. on p. 1).
[20]
Vladimir Druskin, Anne Greenbaum, and Leonid Knizhnerman. “Using Nonorthogonal Lanc-
zos Vectors in the Computation of Matrix Functions”. In: SIAM Journal on Scientific Com-
puting 19.1 (1998), pp. 38–54 (cit. on pp. 3, 10, 19).
[21]
Vladimir Druskin and Leonid Knizhnerman. “Error Bounds in the Simple Lanczos Procedure
for Computing Functions of Symmetric Matrices and Eigenvalues”. In: Journal of Computa-
tional Mathematics and Mathematical Physics 31.7 (1992), pp. 20–30 (cit. on p. 10).
[22]
Vladimir Druskin and Leonid Knizhnerman. “Two polynomial methods of calculating func-
tions of symmetric matrices”. In: USSR Journal of Computational Mathematics and Mathe-
matical Physics 29.6 (1989), pp. 112–121 (cit. on pp. 1, 15).
[23]
Jasper van den Eshof, Andreas Frommer, Thomas Lippert, Klaus Schilling, and Henk A.
van der Vorst. “Numerical methods for the QCD overlap operator. I. Sign-function and error
bounds”. In: Computer Physics Communications 146.2 (2002), pp. 203–224 (cit. on p. 2).
[24]
Andreas Frommer, Stefan G¨uttel, and Marcel Schweitzer. “Efficient and Stable Arnoldi
Restarts for Matrix Functions Based on Quadrature”. In: SIAM Journal on Matrix Analysis
and Applications 35.2 (2014), pp. 661–683 (cit. on p. 4).
[25]
Andreas Frommer, Karsten Kahl, Th Lippert, and Hannah Rittich. “2-Norm Error Bounds and
Estimates for Lanczos Approximations to Linear Systems and Rational Matrix Functions”.
In: SIAM Journal on Matrix Analysis and Applications 34.3 (2013), pp. 1046–1065. eprint:
https://doi.org/10.1137/110859749 (cit. on p. 4).
[26]
Andreas Frommer, Karsten Kahl, Marcel Schweitzer, and Manuel Tsolakis. “Krylov Sub-
space Restarting for Matrix Laplace Transforms”. In: SIAM Journal on Matrix Analysis and
Applications 44.2 (2023), pp. 693–717 (cit. on p. 2).
[27]
Andreas Frommer and Marcel Schweitzer. “Error bounds and estimates for Krylov subspace
approximations of Stieltjes matrix functions”. In: BIT Numerical Mathematics 56.3 (2015),
pp. 865–892 (cit. on p. 4).
[28]
Andreas Frommer and Valeria Simoncini. “Error Bounds for Lanczos Approximations of Ra-
tional Functions of Matrices”. In: Numerical Validation in Current Hardware Architectures.
2009, pp. 203–216 (cit. on p. 4).
[29]
Roy Frostig, Cameron Musco, Christopher Musco, and Aaron Sidford. “Principal component
projection without principal component analysis”. In: Proceedings of the 33nd International
Conference on Machine Learning (ICML 2016). 2016, pp. 2349–2357 (cit. on pp. 1, 9).
[30]
E. Gallopoulos and Y. Saad. “Efficient Solution of Parabolic Equations by Krylov Approx-
imation Methods”. In: SIAM Journal on Scientific and Statistical Computing 13.5 (1992),
pp. 1236–1264 (cit. on p. 1).
[31]
Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson.
“GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration”.
In: Advances in Neural Information Processing Systems (NeurIPS 2018). Vol. 31. 2018 (cit.
on p. 1).
[32]
Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. “An Investigation into Neural Net Op-
timization via Hessian Eigenvalue Density”. In: Proceedings of the 36th International Con-
ference on Machine Learning (ICML 2019). Vol. 97. 2019, pp. 2232–2241 (cit. on pp. 1,
21).
[33]
Anne Greenbaum. “Behavior of slightly perturbed Lanczos and conjugate-gradient recur-
rences”. In: Linear Algebra and its Applications 113 (1989), pp. 7–63 (cit. on pp. 2, 10).

12


---Page Break---
[34]
Anne Greenbaum. “Comparison of splittings used with the conjugate gradient algorithm”. In:
Numerische Mathematik 33.2 (1979), pp. 181–193 (cit. on p. 20).
[35]
Anne Greenbaum. Iterative Methods for Solving Linear Systems. Philadelphia, PA, USA:
Society for Industrial and Applied Mathematics, 1997 (cit. on p. 3).
[36]
Stefan G¨uttel and Marcel Schweitzer. “A Comparison of Limited-memory Krylov Methods
for Stieltjes Functions of Hermitian Matrices”. In: SIAM Journal on Matrix Analysis and
Applications 42.1 (2021), pp. 83–107 (cit. on p. 9).
[37]
Nicholas Hale, Nicholas J. Higham, and Lloyd N. Trefethen. “Computing 𝐴𝛼, log(𝐴), and
Related Matrix Functions by Contour Integrals”. In: SIAM Journal on Numerical Analysis
46.5 (2008), pp. 2505–2523. eprint: https://doi.org/10.1137/070700607 (cit. on
pp. 4, 6).
[38]
Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. “Finding structure with random-
ness: probabilistic algorithms for constructing approximate matrix decompositions”. In: SIAM
Review 53.2 (2011), pp. 217–288 (cit. on p. 2).
[39]
Insu Han, Dmitry Malioutov, Haim Avron, and Jinwoo Shin. “Approximating Spectral Sums
of Large-Scale Matrices using Stochastic Chebyshev Approximations”. In: SIAM Journal on
Scientific Computing 39.4 (2017), A1558–A1585 (cit. on p. 1).
[40]
Clemens Hofreither. “An Algorithm for Best Rational Approximation Based on Barycentric
Rational Interpolation”. In: Numerical Algorithms 88.1 (2021), pp. 365–388 (cit. on pp. 8, 9).
[41]
Marija D. Ilic, Ian W. Turner, and Daniel P. Simpson. “A restarted Lanczos approximation
to functions of a symmetric matrix”. In: IMA Journal of Numerical Analysis 30.4 (2009),
pp. 1044–1061 (cit. on p. 4).
[42]
Charles H Janson. “Measuring evolutionary constraints: a Markov model for phylogenetic
transitions among seed dispersal syndromes”. In: Evolution 46.1 (1992), pp. 136–158 (cit. on
p. 1).
[43]
Robert A Jarrow, David Lando, and Stuart M Turnbull. “A Markov model for the term struc-
ture of credit risk spreads”. In: The Review of Financial Studies 10.2 (1997), pp. 481–523
(cit. on p. 1).
[44]
Yujia Jin and Aaron Sidford. “Principal Component Projection and Regression in Nearly Lin-
ear Time through Asymmetric SVRG”. In: Advances in Neural Information Processing Sys-
tems (NeurIPS 2019). Vol. 32. 2019 (cit. on pp. 1, 2, 9, 27).
[45]
Sadegh Jokar and Bahman Mehri. “The best approximation of some rational functions in
uniform norm”. In: Applied Numerical Mathematics 55.2 (2005), pp. 204–214 (cit. on p. 18).
[46]
Cornelius Lanczos. “An iteration method for the solution of the eigenvalue problem of lin-
ear differential and integral operators”. In: Journal of Research of the National Bureau of
Standards (1950), pp. 255–282 (cit. on p. 2).
[47]
J¨org Liesen and Zdenˇek Strakoˇs. Krylov subspace methods: principles and analysis. 1st. Nu-
merical mathematics and scientific computation. Oxford University Press, 2013 (cit. on pp. 2,
3, 18).
[48]
Stefano Massei and Francesco Tudisco. “Optimizing network robustness via Krylov sub-
spaces”. In: ESAIM: Mathematical Modelling and Numerical Analysis 58.1 (2024), pp. 131–
155 (cit. on p. 1).
[49]
G´erard Meurant and Jurjen Duintjer Tebbens. Krylov Methods for Nonsymmetric Linear Sys-
tems: From Theory to Computations. Springer International Publishing, 2020 (cit. on p. 18).
[50]
G´erard Meurant and Zdenˇek Strakoˇs. “The Lanczos and conjugate gradient algorithms in
finite precision arithmetic”. In: Acta Numerica 15 (2006), pp. 471–542 (cit. on p. 10).
[51]
Fredrik Johansson et al. mpmath: a Python library for arbitrary-precision floating-point arith-
metic (version 0.18). http://mpmath.org/. 2013 (cit. on p. 7).
[52]
Cameron Musco, Christopher Musco, and Aaron Sidford. “Stability of the Lanczos Method
for Matrix Function Approximation”. In: Proceedings of the Twenty-Ninth Annual ACM-
SIAM Symposium on Discrete Algorithms (SODA 2018). 2018, pp. 1605–1624 (cit. on pp. 2,
3, 10).

13


---Page Break---
[53]
Cameron Musco, Praneeth Netrapalli, Aaron Sidford, Shashanka Ubaru, and David P.
Woodruff. “Spectrum Approximation Beyond Fast Matrix Multiplication: Algorithms and
Hardness”. In: 9th Innovations in Theoretical Computer Science Conference (ITCS 2018).
2018 (cit. on p. 9).
[54]
Lorenzo Orecchia, Sushant Sachdeva, and Nisheeth K. Vishnoi. “Approximating the expo-
nential, the Lanczos method and an ˜O(m)-time spectral algorithm for balanced separator”.
In: Proceedings of the 44th symposium on Theory of Computing (STOC 2012). 2012 (cit. on
p. 3).
[55]
Christopher Conway Paige. “Accuracy and effectiveness of the Lanczos algorithm for the
symmetric eigenproblem”. In: Linear Algebra and its Applications 34 (1980), pp. 235–258
(cit. on p. 10).
[56]
Christopher Conway Paige. “Error Analysis of the Lanczos Algorithm for Tridiagonalizing a
Symmetric Matrix”. In: IMA Journal of Applied Mathematics 18.3 (1976), pp. 341–349 (cit.
on p. 10).
[57]
Christopher Conway Paige. “The computation of eigenvalues and eigenvectors of very large
sparse matrices.” PhD thesis. University of London, 1971 (cit. on p. 10).
[58]
Vardan Papyan. The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD Train-
ing and Sample Size. 2019. arXiv: 1811.07062 [cs.LG] (cit. on pp. 1, 21).
[59]
Geoff Pleiss, Martin Jankowiak, David Eriksson, Anil Damle, and Jacob R. Gardner. “Fast
matrix square roots with applications to gaussian processes and Bayesian optimization”. In:
Proceedings of the 34th International Conference on Neural Information Processing Systems
(NeurIPS 2020). 2020 (cit. on pp. 1, 2, 6, 9, 21).
[60]
Theodore J. Rivlin. An introduction to the approximation of functions. Dover books on ad-
vanced mathematics. Dover, 1981 (cit. on p. 20).
[61]
Yousef Saad. “Analysis of Some Krylov Subspace Approximations to the Matrix Exponential
Operator”. In: SIAM Journal on Numerical Analysis 29.1 (1992), pp. 209–228. eprint: https:
//doi.org/10.1137/0729014 (cit. on pp. 3, 15, 18).
[62]
Yousef Saad. Iterative Methods for Sparse Linear Systems. Society for Industrial and Applied
Mathematics, 2003 (cit. on p. 2).
[63]
E. B. Saff, A. Sch¨onhage, and R. S. Varga. “Geometric convergence to 𝑒−𝑧by rational func-
tions with real poles”. In: Numerische Mathematik 25.3 (1975), pp. 307–322 (cit. on pp. 6,
8).
[64]
Lloyd N. Trefethen. Approximation Theory and Approximation Practice, Extended Edition.
Society for Industrial and Applied Mathematics, 2019 (cit. on pp. 6, 20).
[65]
Joel A. Tropp and Robert J. Webber. Randomized algorithms for low-rank matrix approxi-
mation: Design, analysis, and applications. 2023. arXiv: 2306.12418 [math.NA] (cit. on
p. 2).
[66]
Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data
Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge Univer-
sity Press, 2018 (cit. on p. 21).
[67]
Sheng Wang, Yuan Sun, Christopher Musco, and Zhifeng Bao. “Public Transport Planning:
When Transit Network Connectivity Meets Commuting Demand”. In: Proceedings of the
2021 International Conference on Management of Data. SIGMOD ’21. 2021, pp. 1906–1919
(cit. on p. 1).
[68]
Frederick V Waugh and Martin E Abel. “On fractional powers of a matrix”. In: Journal of the
American Statistical Association 62.319 (1967), pp. 1018–1021 (cit. on p. 1).
[69]
Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael Mahoney. PyHessian: Neural Net-
works Through the Lens of the Hessian. 2020. arXiv: 1912.07145 [cs.LG] (cit. on pp. 1,
21).

14


---Page Break---
A
Proof of Theorem 4

A.1
Notation

We first introduce notation used throughout. Given a rational function 𝑟(𝑥) = 𝑛(𝑥)/𝑚(𝑥) with
numerator degree 𝑝denominator degree 𝑞, for 1 ≤𝑖≤𝑗≤𝑞, we define:

𝑚𝑖, 𝑗(𝑥) :=

𝑗
Ö

𝑘=𝑖
(𝑥−𝑧𝑘).

We adopt the convention that 𝑚𝑗+1, 𝑗(𝑥) := 1. Note that 𝑚𝑖, 𝑗is a ( 𝑗−𝑖+ 1)-degree polynomial.
Define also
𝑟𝑗(𝑥) := 𝑛(𝑥)/𝑚1, 𝑗(𝑥).

Note that 𝑚1,𝑞= 𝑚, so 𝑟𝑞= 𝑟(𝑥). Recall that for any function 𝑓(𝑥), the Lanczos-FA approximation
to 𝑓(A)b is lan𝑘( 𝑓; A, b) = Q 𝑓(T)QTb. Define

err𝑘( 𝑓) := 𝑓(A)b −lan𝑘( 𝑓; A, b).

Proving Theorem 4 amounts to proving an upper bound on ∥err𝑘(𝑟)∥2. Finally, for any symmetric
positive definite matrix M, define

opt𝑘( 𝑓, M; A, b) := argmin
x∈K𝑘(A,b)
∥𝑓(A)b −x∥M.

For brevity, in the analysis below we will write lan𝑘( 𝑓) and opt𝑘( 𝑓, M) in place of lan𝑘( 𝑓; A, b)
and opt𝑘( 𝑓, M; A, b), since A and b are fixed throughout the analysis.

A.2
Simplifying lan𝑘(𝑟𝑗) and opt𝑘(𝑟𝑗, A 𝑗)

We begin with a few standard results that will be useful for the proof of Theorem 4.

Lemma 8 ([22, 61]). For a polynomial 𝑛(𝑥), lan𝑘(𝑛) = 𝑛(A)b if 𝑘> deg(𝑛).

Proof. By definition of the Krylov subspace, A𝑗b ∈K𝑘(A, b) for all 𝑗< 𝑘. Since QQT is the 2-
norm orthogonal projector onto K𝑘(A, b), then QQTA 𝑗b = A 𝑗b for all 𝑗< 𝑘. Iteratively applying
this fact,

A 𝑗b = QQTA 𝑗b = QQTAA 𝑗−1b = QQTAQQTA 𝑗−1b = QQTAQ · · · QTAQQTb.

Using that QTAQ = T, we find that A 𝑗b = QT 𝑗QTb, and thus, by linearity, 𝑛(A)b = Q𝑛(T)QTb =
lan𝑘(𝑛).
□

Lemma 9. Let A 𝑗be as in (5). Then, for any 𝑘≥0, opt𝑘(𝑟𝑗, A 𝑗) = Q(T −𝑧𝑗I)−1QT𝑟𝑗−1(A)b.

Proof. The A 𝑗-norm projector onto K𝑘(A, b) is Q(QTA𝑗Q)−1QTA 𝑗, so

opt𝑘(𝑟𝑗, A 𝑗) = Q(QTA 𝑗Q)−1QTA𝑗𝑟𝑗(A).
(9)

Recall that A 𝑗= ±(A−𝑧𝑗I). Since QTAQ = T and QTQ = I, we have QTA 𝑗Q = ±(QT(A−𝑧𝑗I)Q) =
±(T −𝑧𝑗I). Noting also that A 𝑗𝑟𝑗(A) = ±𝑟𝑗−1(A) and plugging into (9) gives the result.
□

A.3
A telescoping sum for the error in terms of opt𝑘(𝑟𝑗, A 𝑗)

Our first main result is a decomposition for the Lanczos-FA error.

Lemma 10. Suppose 𝑘> deg(𝑛). Then,

𝑟(A)b −lan𝑘(𝑟) =

𝑞
∑︁

𝑗=1

"
𝑞
Ö

𝑖=𝑗+1
Q(T −𝑧𝑖I)−1QT
#
 𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗).

where we adopt the convention that Î𝑞
𝑖=𝑞+1 B𝑖= I for any set of matrices {B𝑖}.

15


---Page Break---
Proof. We can decompose err𝑘(𝑟𝑗) as

err𝑘(𝑟𝑗) = 𝑟𝑗(A)b −Q𝑟𝑗(T)QTb

=

𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)

+
h
opt𝑘(𝑟𝑗, A𝑗) −Q𝑟𝑗(T)QTb
i
.
(10)

Focusing on the second term and using Lemma 9,

opt𝑘(𝑟𝑗, A 𝑗) −Q𝑟𝑗(T)QTb

= Q(T −𝑧𝑗I)−1QT𝑟𝑗−1(A)b −Q(T −𝑧𝑗I)−1𝑟𝑗−1(T)QTb

= Q(T −𝑧𝑗I)−1QT h
𝑟𝑗−1(A)b −Q𝑟𝑗−1(T)QTb
i

= Q(T −𝑧𝑗I)−1QT 
err𝑘(𝑟𝑗−1)

.
Substituting this into (10), we have

err𝑘(𝑟𝑗) =

𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)

+ Q(T −𝑧𝑗I)−1QTerr𝑘(𝑟𝑗−1).
(11)

Next, notice that opt𝑘(𝑟1, A1) is exactly the Lanczos approximation to 𝑟1(A)b. Indeed, since QTQ =
I, Lemmas 8 and 9 imply

lan𝑘(𝑟1) = Q(T −𝑧𝑖I)−1𝑛(T)QTb

= Q(T −𝑧𝑖I)−1QTQ𝑛(T)QTb

= Q(T −𝑧1I)−1QT𝑛(A)b = opt𝑘(𝑟1, A1).
Using this as a base case, we can repeatedly apply our decomposition of err𝑘(𝑟𝑗) in (11) to obtain

err𝑘(𝑟) = err𝑘(𝑟𝑞) =

𝑞
∑︁

𝑗=1



𝑞
Ö

𝑖=𝑗+1
Q(T −𝑧𝑖I)−1QT


 𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗) .

□

A.4
Bounding each term in the telescoping sum and combining

The following lemma allows us to relate the optimality of the functions 𝑟𝑗as measured in the A 𝑗-
norm to that of 𝑟𝑞= 𝑟as measured in the 2-norm.
Lemma 11. For any 𝑘> 𝑞−𝑗,
∥𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)∥

≤𝜅(A 𝑗)1/2 · ∥𝑚𝑗+1,𝑞(A)∥2
min
deg( 𝑝)<𝑘−(𝑞−𝑗) ∥𝑟(A)b −𝑝(A)b∥2.

Proof. Recall that opt𝑘(𝑟𝑗, A 𝑗) is the optimal approximation to 𝑟𝑗(A)b in the A𝑗-norm, so can be
related to the optimal approximation in the 2-norm.

∥𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)∥2 =
A−1/2
𝑗
A1/2
𝑗
(𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗))

2
≤∥A−1/2
𝑗
∥2 · ∥𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)∥A𝑗

= ∥A−1
𝑗∥1/2
2
min
deg( 𝑝)<𝑘∥𝑟𝑗(A)b −𝑝(A)b∥A 𝑗

≤∥A−1
𝑗∥1/2
2
· ∥A 𝑗∥1/2
2
min
deg( 𝑝)<𝑘∥𝑟𝑗(A)b −𝑝(A)b∥2

= 𝜅(A 𝑗)1/2
min
deg( 𝑝)<𝑘∥𝑟𝑗(A)b −𝑝(A)b∥2.

We now relate the error of approximating 𝑟𝑗(A)b to that of approximating 𝑟𝑞(A)b = 𝑟(A)b:
min
deg( 𝑝)<𝑘∥𝑟𝑗(A)b −𝑝(A)b∥2

=
min
deg( 𝑝)<𝑘∥𝑚𝑗+1,𝑞(A)𝑟𝑞(A)b −𝑝(A)b∥2

≤
min
deg( 𝑝)<𝑘−(𝑞−𝑗) ∥𝑚𝑗+1,𝑞(A)𝑟𝑞(A)b −𝑚𝑗+1,𝑞(A)𝑝(A)b∥2

≤∥𝑚𝑗+1,𝑞(A)∥2
min
deg( 𝑝)<𝑘−(𝑞−𝑗) ∥𝑟(A)b −𝑝(A)b∥2.

16


---Page Break---
Combining these two steps proves the lemma.
□

With the above results in place, we can prove Theorem 4. We will apply the triangle inequality
to the telescoping sum for err𝑘(𝑟) (Lemma 10) and bound each term by the error of the optimal
approximation to 𝑟(A)b (Lemma 11).

Proof of Theorem 4. Focus on a single term in the sum in Lemma 10. First we will get rid of Q and
QT. For 1 ≤𝑗< 𝑞, using that QTQ = I we have that


𝑞
Ö

𝑖=𝑗+1
Q(T −𝑧𝑖I)−1QT

2
= ∥Q𝑚𝑗+1,𝑞(T)−1QT∥2 ≤∥𝑚𝑗+1,𝑞(T)−1∥2.
(12)

Also note that by the convention adopted in Lemma 10, for 𝑗= 𝑞, Î𝑞
𝑖=𝑗+1 Q(T −𝑧𝑘I)−1QT = I =
𝑚𝑗+1,𝑞(T). Next we claim that

∥𝑚𝑗+1,𝑞(T)−1∥2 ≤∥𝑚𝑗+1,𝑞(A)−1∥2.
(13)

To see why this is the case, note that because T = QTAQ, the eigenvalues of T are contained in
I = [𝜆min(A), 𝜆max(A)]. By assumption, the roots of 𝑚𝑗+1,𝑞(𝑥) are all real and lie outside of I.
Since there is at most one critical point between distinct roots, there can only be one critical point of
𝑚𝑗+1,𝑞(𝑥) in I. Thus, there can be no local minima of |𝑚𝑗+1,𝑞(𝑥)| in the interior of I. Rather, the
minimum of |𝑚𝑗+1,𝑞(𝑥)| over I must be attained at the boundary; i.e. at 𝑥= 𝜆min or 𝑥= 𝜆max.

We now apply the triangle inequality to the telescoping sum of Lemma 10:

∥err𝑘(𝑟)∥2 ≤

𝑞
∑︁

𝑗=1



𝑞
Ö

𝑖=𝑗+1
Q(T −𝑧𝑖I)−1QT

2
·
𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A 𝑗)

2.

Applying (12) and (13) we then have

∥err𝑘(𝑟)∥2 ≤

𝑞
∑︁

𝑗=1
∥𝑚𝑗+1,𝑞(A)−1∥2 ·
𝑟𝑗(A)b −opt𝑘(𝑟𝑗, A𝑗)

2.

Finally, using Lemma 11 we find

∥err𝑘(𝑟)∥2 ≤

𝑞
∑︁

𝑗=1
𝜅(A 𝑗)1/2 · 𝜅(𝑚𝑗+1,𝑞(A))
min
deg( 𝑝)<𝑘−(𝑞−𝑗) ∥𝑝(A)b −𝑟(A)b∥2.
(14)

We can simplify by noting that

min
deg( 𝑝)<𝑘−(𝑞−𝑗) ∥𝑝(A)b −𝑟(A)b∥2 ≤
min
deg( 𝑝)<𝑘−(𝑞−1) ∥𝑝(A)b −𝑟(A)b∥2,

and we can combine all the condition number factors using

𝑞
∑︁

𝑗=1
𝜅(A 𝑗)1/2 · 𝜅(𝑚𝑗+1,𝑞(A)) ≤

𝑞
∑︁

𝑗=1
𝜅(A𝑗)

𝑞
Ö

𝑖=𝑗+1
𝜅(A𝑖) ≤𝑞

𝑞
Ö

𝑖=1
𝜅(A𝑖).

The result follows by plugging into (14).
□

A.5
Proof of Lemma 5

We can bound the error of Lanczos-FA on 𝑓(𝑥) using triangle inequality as follows:

∥𝑓(A)b −lan𝑘( 𝑓; A, b)∥2 ≤∥𝑓(A)b −𝑟(A)b∥2 + ∥𝑟(A)b −lan𝑘(𝑟; A, b)∥2
+ ∥lan𝑘(𝑟; A, b) −lan𝑘( 𝑓; A, b)∥2.
(15)

The first and third terms of (15) are controlled by the maximum error of approximating 𝑓(𝑥) with
𝑟(𝑥) over I. Specifically, if we let ∥𝑟−𝑓∥I := max𝑥∈I |𝑟(𝑥) −𝑓(𝑥)|,

∥𝑓(A)b −𝑟(A)b∥2 ≤∥b∥2 · ∥𝑓(A) −𝑟(A)∥2 ≤∥b∥2 · ∥𝑟−𝑓∥I

17


---Page Break---
and similarly, using that Λ(T) ⊂I,

∥lan𝑘(𝑟; A, b) −lan𝑘( 𝑓; A, b)∥2 = ∥Q𝑟(T)QTb −Q 𝑓(T)QTb∥2 ≤∥b∥2 · ∥𝑟−𝑓∥I.

The second term of (15) can controlled using (7) and a triangle inequality:

∥𝑟(A)b −lan𝑘(𝑟; A, b)∥2 ≤𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘∥𝑟(A)b −𝑝(A)b∥2

≤𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘(∥𝑟(A)b −𝑓(A)b∥2 + ∥𝑓(A)b −𝑝(A)b∥2)

≤𝐶𝑟∥b∥2 · ∥𝑟−𝑓∥I + 𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘∥𝑓(A)𝑏−𝑝(A)b∥2.

Combining, we obtain the bound

∥𝑓(A)b −lan𝑘( 𝑓; A, b)∥2

≤min
𝑟


(𝐶𝑟+ 2)∥b∥2 · ∥𝑟−𝑓∥I + 𝐶𝑟
min
deg( 𝑝)<𝑐𝑟𝑘∥𝑓(A)b −𝑝(A)b∥2

.
(16)

B
Comparison to Prior Work

B.1
Details of existing near-optimality guarantees for Lanczos-FA

In this section, we review in detail the prior analyses of Lanczos-FA cited in Section 1.3. These
are the only near-optimality guarantees for Lanczos-FA of which we are aware. Instance optimal-
ity trivially holds when 𝑓(𝑥) is a polynomial of degree < 𝑘. In this case, it is well known that
lan𝑘( 𝑓; A, b) exactly applies 𝑓(A)b; i.e., ∥𝑓(A)b −lan𝑘( 𝑓; A, b)∥2 = 0 [61]. When 𝑓(𝑥) = 1/𝑥
and A is positive definite, the Lanczos-FA algorithm is mathematically equivalent to the well-known
conjugate gradient algorithm for solving a system Ax = b. This implies that Lanczos-FA is the op-
timal approximation in the Krylov subspace with respect to the A-norm; that is,

lan𝑘(1/𝑥; A, b) = ˜𝑝(A)b,
˜𝑝:= argmin
deg( 𝑝)<𝑘
∥A−1b −𝑝(A)b∥A.

This immediately yields near instance optimality in the Euclidean norm:

∥A−1b −lan𝑘(1/𝑥; A, b)∥2 ≤
√︁

𝜅(A)
min
deg( 𝑝)<𝑘∥A−1b −𝑝(A)b∥2,
(17)

where 𝜅(A) is the condition number 𝜆max/𝜆min of A. That is, Definition 2 is satisfied with 𝐶=
√︁

𝜅(A) and 𝑐= 1.

Note that the best polynomial approximation to 1/𝑥on I = [𝜆min, 𝜆max] already converges at a
geometric rate [45]:

min
deg( 𝑝)<𝑘max
𝑥∈I |1/𝑥−𝑝(𝑥)| =
8𝑡𝑘+1

(𝑡2 −1)2(𝜆max −𝜆min) ,
𝑡:= 1 −
2

1 +
√︁

𝜅(A)
.

That is, Fact 3 suffices to prove the exponential convergence of the conjugate gradient method.
However, conjugate gradient (and equivalently Lanczos-FA) often converges even faster, which is
explained theoretically by the stronger instance- and spectrum-optimality guarantees that the method
satisfies [6, 47, 8].

If A is not positive definite, T may have an eigenvalue at or near to zero and the error of the Lanczos-
FA approximation to A−1b can be arbitrarily large. In fact, the same is true for any function 𝑓(𝑥)
which is much larger on I = [𝜆min, 𝜆max] than on Λ, the set of eigenvalues of A. However, while
the Lanczos-FA iterates may be bad at some iterations, the overall convergence of Lanczos-FA with
𝑓(𝑥) = 1/𝑥is actually good [12]. In particular, the convergence of Lanczos-FA can be related to the
MINRES algorithm [15, 49], which produces an optimal approximation to A−1b with respect to the
A2-norm. This allows certain optimality guarantees for MINRES to be transferred to Lanczos-FA,
even on indefinite systems. In particular, [12] asserts that for every 𝑘, there exists 𝑘∗≤𝑘such that7

∥A−1b −lan𝑘∗(1/𝑥; A, b)∥2 ≤𝜅(A)
√

𝑘+ 1
min
deg( 𝑝)<𝑘∥A−1b −𝑝(A)b∥2.
(18)

7For indefinite matrices, 𝜅(A) denotes the ratio of the largest eigenvalue magnitude to the smallest eigen-
value magnitude.

18


---Page Break---
While (18) does not quite fit the form of Definition 2 because of the 𝑘∗on the left side, it is similar
in spirit. Also note that the dependence on 𝑘in the prefactor
√

𝑘+ 1 is not of great significance.
Indeed, the convergence of the optimal polynomial approximation to 1/𝑥on two intervals bounded
away from zero is geometric.

Finally, a guarantee for the matrix exponential is proved in [20, (45)]. They show that the Lanczos-
FA iterate satisfies the guarantee

∥exp(−𝑡A)b −lan𝑘(exp(−𝑡𝑥); A, b)∥2

≤3∥A∥2
2 𝑡2 max
0≤𝑠≤𝑡


min
deg( 𝑝)<𝑘−2 ∥exp(−𝑠A)b −𝑝(A)b∥2


.

Again, this bound does not quite fit Definition 2 due to the maximization over 𝑠. The authors of [20]
state, “It is not known whether the maximum over 𝑠∈[0, 𝑡] in (45) can be omitted and 𝑠set equal
to 𝑡in the right-hand side.”

B.2
Comparison to Lanczos-OR

In [11] an algorithm called the Lanczos method for optimal rational function approximation
(Lanczos-OR) was developed. For rational functions of the form (4), when run for 𝑘> deg(𝑛)
iterations, Lanczos-OR produces iterates

lanOR𝑘(𝑟; A, b) := argmin
x∈K𝑘−⌊𝑞/2⌋
∥𝑟(A)b −x∥|𝑟(A) |.

This implies a 2-norm instance-optimality guarantee (Definition 2):

∥𝑟(A)b −lanOR𝑘(𝑟; A, b)∥2 ≤
√︁

𝜅(𝑟(A))
min
deg( 𝑝)<𝑘−⌊𝑞/2⌋∥𝑟(A)b −𝑝(A)b∥2.
(19)

This guarantee is similar to and often somewhat better than Theorem 4. For example, when 𝑟(𝑥) =
1/𝑚(𝑥),
√︁

𝜅(𝑟(A)) =
√︁

𝜅(A1) · 𝜅(A2) · · · 𝜅(A𝑞). In this case (19) improves on Theorem 4 by a 𝑞
factor and a square root. However, the bound is for a different algorithm; it does not extend to the
ubiquitous Lanczos-FA method.

Moreover, despite this bound, Lanczos-OR usually performs worse than Lanczos-FA in practice if
error is measured in the 2-norm. It is not hard to find examples where (19) is essentially sharp; see
for instance Figure 8. On the other hand, we have been unable to find any numerical examples where
Theorem 4 is sharp, suggesting that it may be possible to prove a tighter bound for Lanczos-FA.

C
Instance, spectrum, and FOV optimality

The convergence of Lanczos-FA is entirely determined by spectral properties of A. Therefore, fol-
lowing the classical analysis of 𝑓(𝑥) = 1/𝑥, we can relax the notion of instance optimality (Defini-
tion 2) by removing its dependence on b:

Definition 12 (Near Spectrum Optimality). For an input instance ( 𝑓, A, b, 𝑘), we say that a Krylov
method is nearly spectrum optimal with parameters 𝐶, 𝑐if

∥𝑓(A)b −alg𝑘( 𝑓; A, b)∥2 ≤𝐶∥b∥2
min
deg( 𝑝)<𝑐𝑘∥𝑓(A) −𝑝(A)∥2.

Note that Definition 12 depends only on the spectrum of A, and not on the interaction of the eigen-
vectors of A with the starting vector b. Hence, we call it “spectrum optimality”. Note also that the
guarantee of Theorem 6 fits this form. We can further relax this notion of optimality by weakening
its dependence on A:

Definition 13 (Near FOV Optimality). For an input instance ( 𝑓, A, b, 𝑘), we say that a Krylov
method is nearly FOV optimal with parameters 𝐶, 𝑐if

∥𝑓(A)b −alg𝑘( 𝑓; A, b)∥2 ≤𝐶∥b∥2
min
deg( 𝑝)<𝑐𝑘


max
𝑥∈[𝜆min,𝜆max] | 𝑓(𝑥) −𝑝(𝑥)|

.

19


---Page Break---
Note that Fact 3 fits this form. Each type of optimality tightly relaxes the previous one. Definition 2
implies Definition 12 and, as we show in Theorem 14, for any 𝑓, 𝑘and A, there exists a “worst-case”
choice of b for which their bounds coincide; likewise, Definition 12 implies Definition 13, and for
any 𝑓(𝑥) and 𝑘there exists a “worst-case” choice of A and b for which their bounds coincide.
Theorem 14.

1. Given any 𝑑, discrete set Λ of at most 𝑑real points, 𝑓(𝑥), and 𝑘< 𝑑, there exists a
symmetric matrix A ∈R𝑑×𝑑and vector b ∈R𝑑such that the spectrum of A is Λ and
min
deg( 𝑝)<𝑘∥𝑓(A)b −𝑝(A)b∥2/∥b∥2 =
min
deg( 𝑝)<𝑘max
𝑥∈Λ | 𝑓(𝑥) −𝑝(𝑥)|.

2. Given any 𝑑, 𝜆min, 𝜆max, 𝑓(𝑥) continuous on [𝜆min, 𝜆max], and 𝑘< 𝑑, there exists a sym-
metric matrix A ∈R𝑑×𝑑and vector b ∈R𝑑such that the smallest eigenvalue of A is 𝜆min,
the largest eigenvalue of A is 𝜆max, and
min
deg( 𝑝)<𝑘∥𝑓(A)b −𝑝(A)b∥2/∥b∥2 =
min
deg( 𝑝)<𝑘max
𝑥∈I | 𝑓(𝑥) −𝑝(𝑥)|.

Our proof follows the approach of [34, Theorem 1] closely.

Proof. For Item 1, note that if |Λ| ≤𝑘, then both sides of the expression are zero. Thus, it suffices
to consider the case |Λ| ≥𝑘+ 1. On a discrete set Λ with at least 𝑘+ 1 distinct points, 𝑓(𝑥) has a
best (on Λ) polynomial approximation 𝑝∗of degree 𝑘−1 that equioscillates 𝑓(𝑥) at 𝑘+1 points [60,
Theorem 1.11]. That is, there exist values 𝜆1 < · · · < 𝜆𝑘+1 contained in Λ such that
𝑓(𝜆𝑖) −𝑝∗(𝜆𝑖) = (−1)𝑖−1𝜖,
𝑖= 1, . . . , 𝑘+ 1,
where
𝜖:= max
𝑥∈Λ | 𝑓(𝑥) −𝑝∗(𝑥)|.

Similarly, for Item 2 if 𝑓(𝑥) is continuous on I = [𝜆min, 𝜆max], then 𝑓(𝑥) has a unique best poly-
nomial approximation 𝑝∗of degree 𝑘−1 which equioscillates at 𝑘+ 1 points 𝜆1 < · · · < 𝜆𝑘+1 in I
(see for instance [60, 64]), and we analogously define 𝜖as
𝜖:= max
𝑥∈I | 𝑓(𝑥) −𝑝∗(𝑥)|.

Set
A = diag(𝜆1, . . . , 𝜆𝑘+1, 𝜆𝑘+1, . . . , 𝜆𝑘+1
|            {z            }
𝑑−𝑘−1 times

),
b = [𝑏1, . . . , 𝑏𝑘+1, 0, . . . , 0
|   {z   }
𝑑−𝑘−1 times

]T.

Then,

min
deg( 𝑝)<𝑘∥𝑓(A)b −𝑝(A)b∥2
2 =
min
𝛼0,...,𝛼𝑘−1

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖
  𝑓(𝜆𝑖) −𝑝(𝜆𝑖)2,
𝑝(𝜆) :=

𝑘−1
∑︁

𝑗=0
𝛼𝑗𝜆𝑗.

We would like to find b so that 𝑓(𝜆𝑖) −𝑝(𝜆𝑖) = (−1)𝑖−1𝜖; i.e. so that 𝑝= 𝑝∗. When b is fixed, for
each 𝑗= 0, 1, . . . , 𝑘−1 the solution to this least squares problem must satisfy

0 =
d
d𝛼𝑗

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖
  𝑓(𝜆𝑖) −𝑝(𝜆𝑖)2 = −2

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖( 𝑓(𝜆𝑖) −𝑝(𝜆𝑖)) 𝜆𝑗
𝑖,
𝑗= 0, 1, . . . , 𝑘−1.

When 𝑝= 𝑝∗, this gives the conditions

0 = −2

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖(−1)𝑖−1𝜖𝜆𝑗
𝑖,
𝑗= 0, 1, . . . , 𝑘−1.

Without loss of generality, we can assume ∥b∥2 = 1 so that 𝑏2
1 + · · · + 𝑏2
𝑘+1 = 1. Thus, we obtain a
linear system


1
1
· · ·
1
−2𝜖𝜆0
1
2𝜖𝜆0
2
· · ·
2(−1)𝑘+1𝜆0
𝑘+1𝜖
...
...
...
−2𝜖𝜆𝑘−1
1
2𝜖𝜆𝑘−1
2
· · ·
2(−1)𝑘+1𝜖𝜆𝑘−1
𝑘+1





𝑏2
1
𝑏2
2...
𝑏2
𝑘+1



=



1
0
...
0



.

20


---Page Break---
We can rewrite this system as



1
−1
· · ·
(−1)𝑘

𝜆0
1
𝜆0
2
· · ·
𝜆0
𝑘+1
...
...
...
𝜆𝑘−1
1
𝜆𝑘−1
2
· · ·
𝜆𝑘−1
𝑘+1





𝑏2
1
−𝑏2
2
...
(−1)𝑘𝑏2
𝑘+1



=



1
0
...
0



.

This system can be solved analytically via Cramer’s rule and has solution

𝑏2
ℓ=

 𝑘+1
Ö

𝑖=1
𝑖≠ℓ

𝑘+1
Ö

𝑗=𝑖+1
𝑗≠ℓ

(𝜆𝑗−𝜆𝑖)

!, 𝑘+1
∑︁

𝑚=1

𝑘+1
Ö

𝑖=1
𝑖≠𝑚

𝑘+1
Ö

𝑗=𝑖+1
𝑗≠𝑚

(𝜆𝑗−𝜆𝑖)

!

.

Clearly (𝜆𝑗−𝜆𝑖) is positive for 𝑗∈{𝑖+ 1, . . . 𝑘+ 1}. Thus the numerator and denominator above
are positive, so the entries of b are well-defined and real. This implies that, for these choices of A
and b,

min
deg( 𝑝)<𝑘∥𝑓(A)b −𝑝(A)b∥2
2 =
min
𝛼0,...,𝛼𝑘−1

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖
  𝑓(𝜆𝑖) −𝑝(𝜆𝑖)2 =

𝑘+1
∑︁

𝑖=1
𝑏2
𝑖𝜖2 = 𝜖2.

Taking the square root of both sides gives the result.
□

C.1
Relation between spectrum and instance optimality for random vectors

We also note that, with some additional assumptions on b, spectrum optimality implies instance
optimality:

Lemma 15. Let {u1, . . . , u𝑑} be the eigenvectors of A. Given a problem instance ( 𝑓, A, b, 𝑘), if an
algorithm is nearly spectrum optimal with parameters 𝐶and 𝑐, then it is nearly instance optimal
with parameters 𝐶·
∥b∥2
min 𝑗|uT
𝑗b| and 𝑐.

The bound is useful e.g., when b is a random vector. For example, when b has independent and iden-
tically distributed Gaussian entries,
∥b∥2
min𝑗|uT
𝑗b| = 𝑂(𝑑3/2) with high probability. Such random vectors

occur when sampling Gaussians [59] and are extremely common in machine learning applications
related to trace and spectrum approximation [58, 32, 69, 7, 13, 14].

Let V be the matrix whose 𝑖th column is v𝑖. Let 𝑤𝑖= vT
𝑖b and w = VTb be the vector whose 𝑖th
entry is 𝑤𝑖. Then,

∥𝑓(A)b −𝑝(A)b∥2
2 =

𝑑
∑︁

𝑖=1
( 𝑓(𝜆𝑖) −𝑝(𝜆𝑖))2𝑤2
𝑖≥

min
𝑗
𝑤2
𝑗


𝑑
∑︁

𝑖=1
( 𝑓(𝜆𝑖) −𝑝(𝜆𝑖))2.

Clearly Í𝑑
𝑖=1( 𝑓(𝜆𝑖) −𝑝(𝜆𝑖))2 ≥max𝑥∈Λ | 𝑓(𝑥) −𝑝(𝑥)|2. Taking the square root of both sides and
substituting into Definition 12 proves the result.

For completeness, we now show that if b ∼N (0, I𝑑), then with high probability,

∥b∥2
min𝑗|vT
𝑗b|
= 𝑂(𝑑3/2).

First, with high probability, ∥b∥2 = Θ(
√

𝑑); see, for example, [66, p. 3.1.2]. Second, we show that
with high probability |vT
𝑗b| = Ω(1/𝑑) for all 𝑗. Since V is orthonormal, w := VTb ∼N (0, I𝑑). By
anti-concentration of the normal distribution, the probability that |𝑤𝑗| > 𝜖is at least 1−0.4𝜖for any
𝜖. By a union bound, this holds simultaneously for all 𝑗with probability at least 1 −0.4𝜖𝑑. Setting
𝜖= Θ(1/𝑑) finishes the argument. Finally, by another union bound, our bounds on the numerator
and on the denominator hold simultaneously with high probability.

21


---Page Break---
D
Proofs of Theorem 6 and Theorem 7

We begin by quoting bounds from [10]:

Lemma 16. For all 𝑘≥1, the Lanczos-FA iterate satisfies the bounds

1. ∥A1/2b −lan𝑘(𝑥1/2; A, b)∥2 ≤𝜆3/2
max
2𝑘3/2 · ∥A−1b −lan𝑘(1/𝑥; A, b)∥2.

2. ∥A−1/2b −lan𝑘(𝑥−1/2; A, b)∥2 ≤

√︂

𝜆max

𝜋𝑘· ∥A−1b −lan𝑘(1/𝑥; A, b)∥2.

Proof. For Part 1, see Example 4.1 in [10]. For part Part 2, we use the same proof again but sub-
stituting the inverse square root for the square root. For both parts, we have simplified the resulting
bounds using the fact that for 𝑘≥1,

Γ(𝑘−1/2)

Γ(𝑘+ 1)
≤
√𝜋
𝑘3/2
Γ(𝑘+ 1/2)

Γ(𝑘+ 1)
≤
1
√

𝑘
.

□

Next, we bound the error of Lanczos-FA on linear systems by the error of the optimal polynomial
approximation to 𝑥−1/2:

Lemma 17. For all 𝑘≥1, the Lanczos-FA iterate satisfies the bounds

∥A−1b −lan𝑘(1/𝑥; A, b)∥2 ≤
3
√𝜆min

√︁

𝜅(A)∥b∥2
min
deg( 𝑝)<𝑘/2


max
𝑥∈Λ


1
√𝑥−𝑝(𝑥)



.

Proof. Let

𝑝∗(𝑥) =
argmin
deg( 𝑝)<𝑘/2
max
𝑥∈Λ


1
√𝑥−𝑝(𝑥)


and note that 𝑝(𝑥)2 is a polynomial of degree less than 𝑘. Therefore,

min
deg( 𝑝)<𝑘∥A−1b −𝑝(A)b∥2 ≤∥b∥2 min
deg 𝑝<𝑘max
𝑥∈Λ


1
𝑥−𝑝(𝑥)
 ≤∥b∥2 max
𝑥∈Λ


1
𝑥−𝑝∗(𝑥)2
 .

Thus, since |1/𝑥−𝑝∗(𝑥)2| = |1/√𝑥+ 𝑝∗(𝑥)| · |1/√𝑥−𝑝∗(𝑥)|, we can plug the above into (17) to
obtain a bound

∥A−1b−lan𝑘(1/𝑥)∥2 ≤
√︁

𝜅(A)∥b∥2


max
𝑥∈Λ


1
√𝑥+ 𝑝∗(𝑥)


 
max
𝑥∈Λ


1
√𝑥−𝑝∗(𝑥)



.

Now use can use the optimality of 𝑝∗(𝑥)in approximating 1/√𝑥on Λ to bound:

max
𝑥∈Λ


1
√𝑥+ 𝑝∗(𝑥)
 = max
𝑥∈Λ


2
√𝑥+ 𝑝∗(𝑥) −1
√𝑥



≤max
𝑥∈Λ


2
√𝑥

 + max
𝑥∈Λ

𝑝∗(𝑥) −1
√𝑥



≤
2
√𝜆min
+ max
𝑥∈Λ

0(𝑥) −1
√𝑥

 ≤
3
√𝜆min
.

Substituting this inequality above proves the lemma.
□

The main proofs now follow directly.

Proof of Theorem 6. Substitute Lemma 17 into Lemma 16, part (b).
□

22


---Page Break---
Figure 7: The bounds of Theorems 6 and 7 capture the convergence behavior of Lanczos-FA for
A±1/2b. In particular, they can be tighter than the FOV optimality bound of Fact 3. The predicted
rate of convergence is about half that observed for Lanczos-FA in this example; this is due to the
parameter 𝑐= 1/2 in both bounds.

Proof of Theorem 7. Let

𝑝∗(𝑥) =
argmin
deg( 𝑝)<𝑘/2+1
max
𝑥∈Λ∪{0}

√𝑥−𝑝(𝑥)
 ,
˜𝑝(𝑥) = 𝑝∗(𝑥) −𝑝∗(0)

𝑥

and note that ˜𝑝is a polynomial of degree less than 𝑘/2. Then

min
deg( 𝑝)<𝑘/2


max
𝑥∈Λ


1
√𝑥−𝑝(𝑥)



≤max
𝑥∈Λ


1
√𝑥−˜𝑝(𝑥)


= max
𝑥∈Λ
 |1/𝑥| ·
√𝑥−𝑥˜𝑝(𝑥)


≤
1
𝜆min
max
𝑥∈Λ

√𝑥−𝑝∗(𝑥) + 𝑝∗(0)


≤
1
𝜆min
max
𝑥∈Λ

√𝑥−𝑝∗(𝑥)
 +

√

0 −𝑝∗(0)



≤
2
𝜆min
max
𝑥∈Λ∪{0}

√𝑥−𝑝∗(𝑥)
 .

Combining Lemma 16 part (a), Lemma 17, and the above proves the theorem.
□

E
Additional Experiments

E.1
Validating Theorems 6 and 7

To understand Theorems 6 and 7, we repeat the experiment using the inverse square root and square
root functions. We reuse the second and third A matrix and b vectors from the previous experiment.
As Figure 7 shows, our bounds are tighter than Fact 3. They closely resemble the true convergence of
Lanczos-FA, but stretched horizontally by a factor of 2. This is because the degree of the polynomial
minimization in both bounds is 𝑘/2, while in these examples Lanczos-FA performs nearly instance
optimally (that is, as well as the best degree 𝑘polynomial). Still, our bounds capture the correct
qualitative behavior of the algorithm.

E.2
Rational functions with poles between eigenvalues

In Theorem 4, we assume that the poles of the rational function 𝑟(𝑥) lie outside the interval I =
[𝜆min, 𝜆max] containing A’s eigenvalues. If the poles of 𝑟(𝑥) live in I, then the Lanczos-FA iterate

23


---Page Break---
Figure 8: The maximum observed ratio between the error of Lanczos-OR and the optimal error over
choice of right hand side b when approximating A−𝑞for matrices with varying condition number
𝜅. Each point shows the optimality ratio for a different pair of 𝜅and 𝑞. Points with the same color
correspond to the same value of 𝜅. On the left, the dotted line plots 𝑔(𝑞) = 𝜅𝑞/2 for the maximum
𝜅considered (106). On the right, the dotted line plots 𝑔(𝜅) = 𝜅𝑞/2 for the maximum 𝑞considered
(26 = 64). Overall, the optimality ratio appears to grow as Ω(𝜅𝑞/2). Contrast with Figure 3.

can be arbitrarily far from the optimal iterate. Indeed, an eigenvalue of T might be very close to a
pole, causing 𝑓(T) to be poorly behaved. This behavior is easily observed e.g. for 𝑓(A) = A−1 and
rules out a direct analog of Theorem 4 when 𝑟(𝑥) has poles in I. However, as noted in (18), for 1/𝑥
there exists bounds for Lanczos-FA in terms of the best possible KSM [12].

However, as discussed in Appendix B.1, the “overall” convergence of Lanczos-FA still seems to
closely follow the optimal approximation for many matrix functions.
Specifically, even if the
Lanczos-FA iterate deviates significantly from optimal on some iterations, there are other iterations
where it is very close to optimal. We illustrate this phenomena in Figure 6 for two rational func-
tions with poles in I = [𝜆min, 𝜆max], as well as the function sign(𝑥), which has a discontinuity in I
and would typically be approximated by a rational function with conjugate pairs of imaginary poles
on the imaginary axis which become increasingly close to the the interval as the rational function
degree increases.8

E.3
Comparison with Lanczos-OR

As discussed in Appendix B.2, the Lanczos-OR method is an alternative method for rational matrix
function approximation for which a tighter guarantee than Theorem 4 is currently known (the iter-
ates are optimal in a certain norm). However, Lanczos-FA usually outperforms Lanczos-OR when
measured in the 2-norm. For example, as discussed in Section 4.1, when approximating A−𝑞b, the
worst optimality ratio of Lanczos-FA that we were able to observe was 𝑂(
√︁

𝑞· 𝜅(A)), much lower
than Theorem 4 would suggest). Repeating the same experiment with Lanczos-OR shows that the
method’s optimality ratio appears to grow as Ω(𝜅(A)𝑞/2) (see Figure 8).

8We remark that [11] shows that applying their Lanczos-OR method to each term in the partial fraction
decomposition of the approximating rational function yields an approximation to the sign function that seems
to avoid the oscillations of Lanczos-FA.

24


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our abstract refers to the main theoretical result, Theorem 4, and to the ex-
perimental results of Section 4, particularly Figure 2.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these
goals are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Each of our theorems clearly states all assumptions. Section 5 discusses the
limitations of our current results and the open questions that remain.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The au-
thors should reflect on how these assumptions might be violated in practice and what
the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the ap-
proach. For example, a facial recognition algorithm may perform poorly when image
resolution is low or images are taken in low lighting. Or a speech-to-text system might
not be used reliably to provide closed captions for online lectures because it fails to
handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to ad-
dress problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

25


---Page Break---
Answer: [Yes]

Justification: Formal theorem statements include all assumptions. All proofs appear in
Appendices A, C, and D.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theo-
rems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a
short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be comple-
mented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclu-
sions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We release our code in the supplementary material and on GitHub. In addi-
tion, details of the experiments are provided in the text.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps
taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture
fully might suffice, or if the contribution is a specific model and empirical evaluation,
it may be necessary to either make it possible for others to replicate the model with
the same dataset, or provide access to the model. In general. releasing code and data
is often one good way to accomplish this, but reproducibility can also be provided via
detailed instructions for how to replicate the results, access to a hosted model (e.g., in
the case of a large language model), releasing of a model checkpoint, or other means
that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all sub-
missions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear
how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to re-
produce the model (e.g., with an open-source dataset or instructions for how to
construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case au-
thors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

26


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We provide our code in the supplementary material and on GitHub. The code
to produce Figure 5 is not released, as it was shared with us privately by the authors of the
paper in which it first appeared [44]. Standard Python libraries are used for reproducibility,
and a README is provided with instructions for how to generate the figures.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
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
Justification: These are described in the text and the code is included in the supplementary
materials.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of
detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropri-
ate information about the statistical significance of the experiments?
Answer: [No]
Justification: The algorithms we test are not random and no sampling is performed.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

27


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [No]
Justification: No special computing resources were used. All experiments were performed
on a standard laptop in a few minutes.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments
that didn’t make it into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: No human subjects or data was used. No clear potential exists for harmful
consequences.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [No]
Justification: This paper provides new theoretical analysis of an old and widely used al-
gorithm, the Lanczos method. The goal is to achieve a better scientific understanding of
the method. The Lanczos method has been used in a huge variety of applications across
machine learning and scientific computing, and thus has societal impact, but the results and
focus of this paper do not directly relate to this impact.

28


---Page Break---
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact spe-
cific groups), privacy considerations, and security considerations.
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
• If there are negative societal impacts, the authors could also discuss possible mitiga-
tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: See above.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by re-
quiring that users adhere to usage guidelines or restrictions to access the model or
implementing safety filters.
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

Justification: We cite third party software libraries and use them properly.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

29


---Page Break---
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the pack-
age should be provided. For popular datasets, paperswithcode.com/datasets has
curated licenses for some datasets. Their licensing guide can help determine the li-
cense of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documenta-
tion provided alongside the assets?

Answer: [Yes]

Justification: Our only asset is code to perform our experiments. This is documented in the
code and README.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can
either create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the pa-
per include the full text of instructions given to participants and screenshots, if applicable,
as well as details about compensation (if any)?

Answer: [NA]

Justification: No human subjects were used.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Including this information in the supplemental material is fine, but if the main contri-
bution of the paper involves human subjects, then as much detail as possible should
be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
tion, or other labor should be paid at least the minimum wage in the country of the
data collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?

Answer: [NA]

Justification: No IRB approval was required.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.

30


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equiva-
lent) may be required for any human subjects research. If you obtained IRB approval,
you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity
(if applicable), such as the institution conducting the review.

31


---Page Break---
