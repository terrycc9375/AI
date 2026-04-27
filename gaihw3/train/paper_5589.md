High-probability complexity guarantees for nonconvex
minimax problems

Yassine Laguel∗
Laboratoire Jean Alexandre Dieudonné
Université Côte d’Azur
Nice, France
yassine.laguel@univ-cotedazur.fr

Yasa Syed
Department of Statistics
Rutgers University
Piscataway, New Jersey, USA
yasa.syed@rutgers.edu

Necdet Serhat Aybat
Department of Industrial Engineering
Penn State University
University Park, PA, USA
nsa10@psu.edu

Mert Gürbüzbalaban
Rutgers Business School
Rutgers University
Piscataway, New Jersey, USA
mg1366@rutgers.edu

Abstract

Stochastic smooth nonconvex minimax problems are prevalent in machine learning,
e.g., GAN training, fair classification, and distributionally robust learning. Stochas-
tic gradient descent ascent (GDA)-type methods are popular in practice due to
their simplicity and single-loop nature. However, there is a significant gap between
the theory and practice regarding high-probability complexity guarantees for these
methods on stochastic nonconvex minimax problems. Existing high-probability
bounds for GDA-type single-loop methods only apply to convex/concave minimax
problems and to particular non-monotone variational inequality problems under
some restrictive assumptions. In this work, we address this gap by providing the
first high-probability complexity guarantees for nonconvex/PL minimax problems
corresponding to a smooth function that satisfies the PL-condition in the dual
variable. Specifically, we show that when the stochastic gradients are light-tailed,
the smoothed alternating GDA method can compute an ε-stationary point within
O( ℓκ2δ2

ε4
+
κ
ε2 (ℓ+ δ2 log(1/¯q))) stochastic gradient calls with probability at
least 1 −¯q for any ¯q ∈(0, 1), where µ is the PL constant, ℓis the Lipschitz
constant of the gradient, κ = ℓ/µ is the condition number, and δ2 denotes a
bound on the variance of stochastic gradients. We also present numerical results
on a nonconvex/PL problem with synthetic data and on distributionally robust
optimization problems with real data, illustrating our theoretical findings.

1
Introduction

Minimax optimization problems arise frequently in machine learning (ML) applications; indeed,
constrained optimization problems such as deep learning with model constraints [24], dictionary
learning [10, 57] or matrix completion [28] can be recast as a minimax optimization problem through
Lagrangian duality. Other applications include but are not limited to the training of GANs [59],
fair learning [72], supervised learning [61, 49, 51, 74], adversarial deep learning [75], game theory
[54, 58], robust optimization [4, 3], distributionally robust learning [46, 75, 24], meta-learning [70]
and multi-agent reinforcement learning [15]. Many of these applications can be reformulated in

∗Corresponding Author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the following minimax form:

min
x∈Rd1 max
y∈Rd2 f(x, y),
(1)

where f : Rd1 × Rd2 →R is a smooth function, i.e., differentiable with a Lipschitz gradient; f can
possibly be nonconvex in x and nonconcave in y. First-order primal-dual (FOPD) methods have been
the leading computational approach for computing low-to-medium-accuracy stationary points for
these problems because of their cheap iterations and mild dependence of their overall complexities
on the problem dimension and data size [9, 8, 36]. In the context of FOPD methods, there are two
key settings for (1):

(i) the deterministic setting, where the partial gradients ∇xf and ∇yf are exactly available,
(ii) the stochastic setting, where we have only access to (inexact) stochastic estimates of the
partial gradients, in which case the problem in (1) is called a stochastic minimax problem.

It can be argued that the stochastic setting is more relevant to modern machine learning applications
where gradients are typically estimated randomly from mini-batches of data, or sometimes
intentionally perturbed with random noise to ensure data privacy [14, 34, 1].
Convex and nonconvex minimax optimization. In the convex case (when f is convex2 in x and
concave in y), several approaches have been considered including Variational Inequalities (VIs) and
primal-dual algorithms, see. e.g. [29, 20, 5, 67, 12, 73, 60, 11] and the references therein. One
disadvantage of using the VI approach for solving minimax problems (by identifying the signed
gradient map G(x, y)≜[∇xf(x, y)⊤, −∇yf(x, y)⊤]⊤as the corresponding operator in the VI) is
that one needs to set the primal and dual stepsize to be the same. This can be restrictive in applications
where f exhibits different smoothness properties in the primal (x) and dual (y) block coordinates
–this is often the case in distributionally robust learning [73], adversarial learning [43] and in the
Lagrangian reformulations of constrained optimization problems that involve many constraints [47].
The gap function G(xk, yk) ≜supx,y f(xk, y) −f(x, yk), and the squared distance to the set of
saddle points D(xk, yk) ≜min{∥xk −x⋆∥2 + ∥yk −y⋆∥2 | (x⋆, y⋆) is a saddle point} are standard
metrics for assessing the quality of the output zk = (xk, yk) generated by an FOPD algorithm after k
iterations among many others [35, 73, 20].
In the nonconvex setting, i.e., when f is nonconvex in x, the aim is to compute a stationary point. Let
M(xk, yk) denote a measure for the stationarity of iterates (xk, yk); a common metric is the norm of
the gradient, i.e., M(xk, yk) ≜∥∇f(xk, yk)∥and its variants such as ∥∇Φ(xk)∥when f is strongly
concave in y, where Φ(·) = maxy f(·, y) denotes the primal function –for other metrics and relation
between them, see [62]. There are several algorithms that admit (gradient) complexity guarantees
for computing a stationary point of nonconvex minimax problems under various strong concavity,
concavity or weak concavity-type assumptions in the y variable –see the references in [62].

In this paper, we consider smooth nonconvex-PL (NCPL) problems where f is a smooth function
such that it is possibly nonconvex in x and it satisfies the Polyak-Lojasiewicz (PL) condition in y.
The PL condition is a weaker assumption (milder condition) than strong concavity in y –in fact, PL
condition in the dual does not even require quasi-concavity. NCPL problems constitute a rich class
of problems arising in many ML applications including but not limited to fair classification [48],
robust neural network training with dual regularization [48, eqn. (14)], overparametrized systems
and neural networks [42], linear quadratic regulators [17], smoothed Lasso problems [23] subject
to constraints, distributionally robust learning with ℓ2 regularization in the dual [72], deep AUC
maximization [68] and covariance matrix learning with Wasserstein GANs [52]. For deterministic
NCPL problems, the alternating gradient descent ascent (AGDA) method and its smoothed version
(smoothed AGDA) have the complexity of O(κ2/ϵ2) and O(κ/ϵ2), respectively, for finding a point
(˜x, ˜y) satisfying ∥∇f(˜x, ˜y)∥≤ϵ as shown in [70, 66]. Here κ ≜ℓ/µ is the condition number, where
ℓis the Lipschitz constant of the gradient, and µ is the PL constant. For Catalyst-AGDA, [66] shows
also the rate O(κ/ϵ2) for deterministic NCPL problems.
In expectation and high-probability bounds. Most of the existing guarantees in the literature for
stochastic FOPD algorithms are provided in expectation, i.e., a bound on the number of iterations
k (or the stochastic gradient evaluations) is provided for E[G(xk, yk)] ≤ε or E[D(xk, yk)] ≤ε to
hold (see, e.g., [73, 72, 29] and the references therein). Yet having such guarantees on average does

2 ˆf : Rd →R ∪{+∞} is called (merely) convex, if ˆf (tx1(1 −t)x2) ≤t ˆf (x1) + (1 −t) ˆf (x2) for every
x1, x2 ∈Rd and t ∈[0, 1] with the convention that α ≤+∞for all α ∈R.

2


---Page Break---
not allow to control tail events, i.e., even if E[G(xk, yk)] is small, G(xk, yk) can still be arbitrarily
large with a non-zero probability. To this end, high-probability guarantees have been considered
in the literature [35, 64, 20, 32, 22]. These results allow to control the risk associated with the
worst-case tail events as they specify how many iterations would be sufficient to ensure G(xk, yk)
is sufficiently small for any given failure probability ¯q ∈(0, 1). To derive high-probability bounds,
one common approach involves running the algorithm in parallel multiple times and strategically
selecting an optimal output to convert in-expectation bounds into high-probability guarantees [62, 37].
Alternatively, advanced concentration inequalities can be employed under light-tail assumptions
to control noise accumulation across iterates without requiring multiple runs [53, 27]. For saddle
point problems, we note that existing high-probability bounds mostly apply to the monotone VI
setting, or to strongly convex/strongly concave (SCSC) minimax problems. To our knowledge,
high-probability guarantees for nonconvex minimax problems are non-existent in the literature, even
for nonconvex/strongly concave (NCSC) problems with the exception of trivial loose bounds one can
obtain by a standard application of Markov’s inequality (see Remark 13 for details). In particular, we
note that the existing VI literature with high-probability bounds on non-monotone operators such as
star-co-coercive operators [20, 55], do not apply to NCSC problems.3

New high-probability bounds for NCPL optimization. To address these shortcomings, we focus
on developing high-probability guarantees for NCPL problems. Among the existing algorithms in the
stochastic NCPL setting [64], stochastic gradient descent ascent (SGDA) methods and their variants
are quite popular for ML applications, e.g., training GANs and adversarial learning, as SGDA is
easy to implement due to its single-loop structure. Guarantees in expectation for stochastic NCSC
problems are well supported by the literature - see [72, 63, 40, 6, 31, 30, 44, 66, 33, 39] and the
references therein. To our knowledge, among single-loop methods for NCPL problems, the best
guarantees in expectation are given by the smoothed alternating gradient descent ascent (sm-AGDA)
method [66], which can compute an almost stationary point (˜x, ˜y) satisfying E[∥∇f(˜x, ˜y)]∥≤ε
in O(ℓκ2δ2/ε4 + ℓκ/ε2) stochastic gradient calls, where δ2 is an upper bound on the variance of
the stochastic gradients. In this work, we consider the sm-AGDA algorithm, and to our knowledge
we provide the first-time high-probability bounds (using a single-loop method that does not resort
to restarts and parallel runs) for the minimax problem (1) in the NCSC and NCPL settings. More
precisely, we focus on a purely stochastic regime in which data streams over time which renders the
use of mini-batch schemes or running the method in parallel impractical; therefore, approaches based
on Markov’s inequality [62] are no longer applicable (see also Remark 13).

Contributions. Our contributions are threefold:
• We present the first high-probability complexity result for the sm-AGDA algorithm in the NCPL
setting by building upon a Lyapunov function first introduced in [70] for nonconvex-concave
problems. Later, for the same Lyapunov function, state-of-the-art complexity bounds in expectation
are provided for the NCPL setting in [66]. In this paper, we derive a novel descent property
for this Lyapunov function in the almost sure sense (Theorem 7 and Corollary 8), allowing
us to develop useful concentration arguments for it to derive high-probability bounds. Our
Lyapunov analysis not only sheds light on the convergence properties of sm-AGDA, but also
guides the parameter selection for sm-AGDA. Specifically, we show that sm-AGDA can compute
an almost stationary point (˜x, ˜y) satisfying ∥∇f(˜x, ˜y)∥≤ε with probability 1 −¯q ∈(0, 1) within

Tε,¯q = O

ℓκ2δ2

ε4
+ κ

ε2
 
ℓ+ δ2 log(1/¯q)

stochastic gradient calls. The lower complexity bound

of Ω( 1

ε2 + 1

ε4 ) for NCSC problems [38, 71] in expectation (see also [63, 72]) suggests that our
high-probability bound for sm-AGDA is tight in terms of its dependence on ε. Furthermore, to our
knowledge, these are the first high-probability guarantees for any algorithm in the NCPL setting.
• Under light-tail (sub-Gaussian) assumption on the gradient noise (Assumption 3), which is
common in the literature [35, 32, 19], we develop a new concentration result (Theorem 9) that
can be of independent interest. From this concentration inequality, we observe that the cost of
strengthening the existing complexity result in expectation to a high-probability one is relatively
low, i.e., in the final complexity, the probability parameter ¯q only appears in an additive term that
scales with ε−2. Consequently, this represents a non-dominant overhead compared to the ε−4 term
already present in state-of-the-art expectation bounds [66].

3For example, when ∇f is smooth (Lipschitz and continuously differentiable) and the signed gradient map
G(x, y) is star-cocoercive with constant ℓ> 0 around a stationary point (x∗, y∗), then by [21, Lemma C.6], the
operator Id −2

ℓG(x, y) is non-expansive around (x∗, y∗); thus, the Jacobian of G at (x∗, y∗) has non-negative
eigenvalues implying f is merely convex/merely concave around (x∗, y∗), i.e., f cannot be NCSC.

3


---Page Break---
Algorithm
Complexity
Problem
Metric
NC?

Epoch-GDA [64]†
O

δ2
µsε log(1/¯q)


SCSC
G(¯zk)
✗

Clipped-SGDA [20]‡
˜
O

max
n
ℓ
µs ,
δ2
µsε
o
log
  1

ε

log (κ/¯q)


SCSC
D(zk)
✗

Clipped-SEG [20]♭
˜
O

max
n
ℓ
µs ,
δ2
µsε
o
log
  1

ε

log (κ/¯q)


SCSC
D(zk)
✗

Stochastic APD [35]▷∗
O
 ℓlog
 1

ε



µs
+ (1+log(1/¯q))δ2 log(1/ε)

µsε


SCSC
D(zk)
✗

Mirror-Prox [32]∗
O

max

ℓD2

ε2 , σ2D2

ε2

log (1/¯q)


MCMC
G(¯zk)
✗

Clipped-SGDA [20]♯
˜
O

max{ 1

ε , δ2R2

ε2
} log (1/¯q)


MCMC
GR(¯zk)
✗

Clipped-SEG [20]
˜
O

max

ℓR2

ε , σ2R2

ε2

log (1/¯q)


MCMC
GR(¯zk)
✗

sm-AGDA [Our Paper, Coro. 14)]∗
O

ℓκ2δ2

ε4
+
κ
ε2

ℓ+ δ2 log(1/¯q)


NCPL
1
k+1
Pk
j=0 ∥∇f(zj)∥2
✓

Table 1: Summary of the high-probability bounds for minimax problem classes when the gradient of f is Lipschitz (with parameter ℓ) and
stochastic gradient variance is bounded by δ2. The second column reports the complexity (number of calls to stochastic gradient oracle)
required to achieve the (stationarity) metric reported in the fourth column to be at most ε with probability 1 −¯q ∈(0, 1); ˜
O(·) ignores some
logarithmic terms. Here, µs is the strong convexity constant, µ is the PL constant, and κ ≜ℓ/µ. Let G(z) ≜[∇xf(z)⊤, −∇yf(z)⊤]⊤

with z = (x, y) and ¯zk =
1
K+1
PK
j=0 zj, G(z) ≜max{˜z∈Z}⟨G(z), ˜z −z∗⟩, where Z is the domain of the problem with diameter

D ∈(0, +∞], and GR(z) ≜max{˜z∈Z:∥z−z∗∥≤R}⟨G(z), ˜z −z∗⟩where z∗= (x⊤
∗, y⊤
∗)⊤is a stationary point. The third column
reports the minimax problem class. The fifth column indicates whether the results supports nonconvexity, i.e., whether f can be a smooth
function nonconvex in x. † [64] is a two-loop method. ‡ Applicable to quasi-strongly monotone G that is star-co-coercive around z∗and
supports heavy-tailed gradients. ♭Applicable to quasi-strongly monotone G and supports heavy-tailed gradients. ▷Supports proximal steps to
handle non-smooth convex penalty. ♯Applies to monotone G that is star-co-coercive around z∗. ∗Makes a light-tail assumption (Ass. 3).

• Third, we provide experiments that illustrate our theoretical results. We first provide an example of
an NCPL-game with synthetic data and then focus on distributionally robust optimization problems
with real data, illustrating the performance of the sm-AGDA in terms of high-probability guarantees.

2
Preliminaries and Technical Background

Stationarity metric. We consider the minimax problem in (1) for f : Rd1 × Rd2 →R such that
f is smooth (Assumption 1) and f(x, ·) satisfies the PL property for all x ∈Rd1 (Assumption 2);
moreover, we also assume that we only have access to unbiased stochastic estimates of ∇f such
that the stochastic error G(x, y, ξ) −∇f(x, y) has a light tail (Assumption 3) for any (x, y), where
G(x, y, ξ) denote the stochastic estimate of ∇f(x, y) and ξ denotes the randomness in the estimator.

Our aim is to compute a (εx, εy)-stationary point (˜x, ˜y) for (1) such that ∥∇xf(˜x, ˜y)∥≤εx and
∥∇yf(˜x, ˜y)∥≤εy. We also call (˜x, ˜y) an ε-stationary point if ∥∇f(˜x, ˜y)∥≤ε. Clearly, whenever
(˜x, ˜y) is (εx, εy)-stationary, then it is also ε-stationary for ε = (ε2
x + ε2
y)1/2.

Smoothed alternating gradient descent ascent (sm-AGDA): The method can be considered as an
inexact proximal point method and was introduced in [70]. More specifically, in each iteration of
sm-AGDA, given a proximal center zt and the current iterate (xt, yt), the method computes the next
iterate (xt+1, yt+1) using a stochastic gradient descent ascent step on a regularized function ˆf:

ˆf(x, y; zt) ≜f(x, y) + p

2∥x −zt∥2.
(2)

Following the stochastic alternating gradient descent ascent (stochastic AGDA) steps, the proximal
center at iteration t, i.e., zt, is updated as shown in Algorithm 1, where Gx(xt, yt, ξx
t+1) and
Gy(xt+1, yt, ξy
t+1) denote conditionally unbiased stochastic estimators of the gradients ∇xf(xt, yt)
and ∇yf(xt+1, yt). Throughout the analysis we assume that ∇f is Lipschitz, which is standard in the
study of first-order optimization algorithms for smooth minimax problems; see, e.g., [72, 73, 75, 63].
Assumption 1. (Lipschitz gradient) For all (x1, y1), (x2, y2) ∈Rd1 × Rd2, there exists ℓ> 0

∥∇xf(x1, y1) −∇xf(x2, y2)∥≤ℓ(∥x1 −x2∥+ ∥y1 −y2∥)
(3)
∥∇yf(x1, y1) −∇yf(x2, y2)∥≤ℓ(∥x1 −x2∥+ ∥y1 −y2∥).
(4)
The following condition, known as Polyak-Łojaciewicz (PL) condition is weaker than assuming
strong concavity in y, and does not even necessitate f to be even quasi-concave in the y variable. It
holds in many ML applications including those in [48, 48, 42, 17, 23, 72, 68, 52, 66].
Assumption 2. (PL condition in y) For every x ∈Rd1, maxy∈Rd2 f(x, y) has a non-empty solution
set and a finite optimal value. Moreover, there exists µ > 0 such that:

∥∇yf(x, y)∥2 ≥2µ[ max
y∈Rd2 f(x, y) −f(x, y)],
∀x ∈Rd1.
(5)

4


---Page Break---
We assume that we have only access
to stochastic estimates Gx(xt, yt, ξx
t+1)
and Gy(xt+1, yt, ξy
t+1) of the partial gra-
dients ∇yf(xk, yk) and ∇xf(xt+1, yt),
where ξx
t+1 and ξy
t+1 are random vari-
ables defined on a probability space
(Ω, P), i.e., the source of randomness
in the gradient estimates.
Note that
sm-AGDA has Gauss-Seidel updates, i.e.,
the stochastic estimate of the partial gra-

Algorithm 1 sm-AGDA

Input: (x0, y0, z0), τ1, τ2 > 0, β ∈[0, 1], p ≥0
for t = 0, 1, 2, . . . , T −1 do

xt+1 = xt −τ1[Gx(xt, yt, ξx
t+1) + p(xt −zt)]
yt+1 = yt + τ2Gy(xt+1, yt, ξy
t+1)
zt+1 = zt + β(xt+1 −zt)
end for
Output: choose (˜x, ˜y) uniformly from {(xt, yt)}T −1
t=0

dient Gy(xt+1, yt, ξy
t+1) is evaluated at the updated point (xt+1, yt) instead of (xt, yt). To capture
the sequential information flow, we next introduce the natural filtrations that represent all the
information available before an update: Let ξx
t and ξy
t be revealed sequentially in the natural order of
the sm-AGDA updates, i.e., ξx
1 →ξy
1 →ξx
2 →ξy
2 →ξx
3 →· · · , and let (Fx
t )t≥1 and (Fy
t )t≥1 denote
the associated filtration4, i.e., let Fy
0 ≜{∅, Ω}, and

Fx
t+1 = σ(Fy
t , σ(ξx
t+1)),
Fy
t+1 = σ(Fx
t+1, σ(ξy
t+1)),
∀t ≥0.
(6)

Introducing multiple filtrations to represent the sequential information flow is common in the study of
stochastic algorithms with Gauss-Seidel updates –see, e.g., papers on stochastic ADMM, and [7, 69];
and we follow the same approach. Consider the gradient noise (errors) at time t ∈N:

∆x
t ≜Gx(xt, yt, ξx
t+1) −∇xf(xt, yt),
∆y
t ≜Gy(xt+1, yt, ξy
t+1) −∇yf(xt+1, yt).

Finally, we also assume that the gradient noise is unbiased conditionally on the past information and
that it admits a light (sub-Gaussian) tail.

Assumption 3. (Light tail) For any t ≥0, there exists scalars δx, δy > 0 such that

E [∆x
t | Fy
t ] = 0,
P [∥∆x
t ∥≥s | Fy
t ] ≤2e

−s2

2δ2y ,
(7)

E

∆y
t | Fx
t+1


= 0,
P

∥∆y
t ∥≥s | Fx
t+1


≤2e

−s2

2δ2x .
(8)

For developing high-probability bounds in the learning context, it is common to assume that gradient
estimates are sub-Gaussian [32, 35, 18]. While this assumption may not always hold (see e.g.
[25, 56]), it often holds when gradients are estimated via mini-batching, as a consequence of the
central limit theorem. It will also hold when the gradient noise is bounded. Additionally, adoption of
differential privacy mechanisms within gradient-based schemes [14, 34, 1], to enhance data privacy,
results frequently in sub-Gaussian gradient errors.

3
High-probability bounds for sm-AGDA

For analyzing sm-AGDA, similar to [66, 70], we consider the following Lyapunov function:

Vt ≜V (xt, yt; zt) = ˆf(xt, yt; zt) + 2P(zt) −2Ψ(yt; zt),
(9)

where P(z) and Ψ(·; z) denote the saddle point value and the dual function value, respectively, of the
auxiliary problem minx maxy ˆf(x, y; z) for any fixed z and ˆf defined in (2), i.e.,

Ψ(y; z) ≜min
x∈Rd1
ˆf(x, y; z)
and
P(z) ≜min
x∈Rd1 max
y∈Rd2
ˆf(x, y; z).
(10)

Next, we introduce a natural assumption, commonly made in the literature [66, 65]. Without this
assumption, there are pathological cases where primal function Φ(x) may be unbounded leading to
divergence of gradient-based methods; an example would be f(x, y) = −x2 −y2 in dimension one.

Assumption 4. Consider the primal function Φ : Rd1 →R, i.e., Φ(x) = maxy∈Rd2 f(x, y). There
exists x∗∈Rd1 such that Φ∗≜Φ(x∗) = minx∈Rd1 Φ(x).

4Given a random variable ξ, σ(ξ) denotes the σ-algebra generated by ξ; moreover, given two σ-algebras, Σ1
and Σ2, abusing the notation, σ(Σ1, Σ2) denotes the σ-algebra generated by Σ1 ∪Σ2.

5


---Page Break---
Under Assumption 4, it immediately follows that Vt ≥Φ∗for all t ∈N –since P(z) −Ψ(y, z) ≥0,
ˆf(x, y; z) −Ψ(y; z) ≥0 and P(z) ≥Φ∗for all x, y, z. We will next study the change Vt −Vt+1 in
the Lyapunov function and show that an approximate descent property holds. First, we need two key
lemmas that characterize the evolution of ˆf(xt, yt; zt) and Ψ(yt; zt) over the iterations.
Lemma 5. Suppose Assumptions 1, 2, 3 and 4 hold. Consider sm-AGDA given in Alg. 1 with
τ1 ∈(0,
1
p+ℓ] and β ∈(0, 1]. For any t ∈N, we have:

ˆf(xt+1, yt+1; zt+1) −ˆf(xt, yt; zt) ≤−τ1

2 ∥∇x ˆf(xt, yt; zt)∥2 + τ2


1 + ℓ

2τ2


∥∇yf(xt+1, yt)∥2

+ τ1((p + ℓ)τ1 −1)⟨∆x
t , ∇x ˆf(xt, yt; zt)⟩+ p + ℓ

2
τ 2
1 ∥∆x
t ∥2

+ τ2(1 + ℓτ2)⟨∇yf(xt+1, yt), ∆y
t ⟩−p

2β ∥zt −zt+1∥2 + ℓτ 2
2
2 ∥∆y
t ∥2.

Proof. The proof is provided in Appendix B.1.

From Assumption 1, when p > ℓ, the auxilliary function ˆf(·, y; z) is (p −ℓ)-strongly convex for any
fixed y, z; hence, there is a unique minimizer for every y, z fixed, denoted by

x∗(y, z) ≜argminx∈Rd1 ˆf(x, y; z),
(11)

i.e., Ψ(y, z) = ˆf(x∗(y, z), y; z). In the rest of the paper, we will take p > ℓand exploit this property.
The following lemma characterizes the change in the dual function Ψ.
Lemma 6. Suppose Assumptions 1, 2, 3 and 4 hold. Consider the sm-AGDA iterate sequence
{(xt, yt, zt)}t∈N for p > ℓ. For any t ∈N, it holds that

Ψ(yt+1; zt+1) −Ψ(yt; zt) ≥τ2⟨∇yf(x∗(yt, zt), yt), ∇yf(xt+1, yt)⟩+ τ2⟨∇yf(x∗(yt, zt), yt), ∆y
t ⟩

−LΨ

2 τ 2
2
 
∥∇yf(xt+1, yt)∥2 + 2⟨∇yf(xt+1, yt), ∆y
t ⟩+ ∥∆y
t ∥2

+ p

2⟨zt+1 −zt, zt+1 + zt −2x∗(yt+1, zt+1)

,

where LΨ ≜ℓ

1 + p+ℓ

p−ℓ


and the map x∗(·, ·) is defined by (11).

Proof. The proof is provided in Appendix B.2.

The next result provides an approximate descent property on the Lyapunov function. Its proof builds
on Lemmas 5 and 6 and a descent property on the function P (given in Lemma 15 of the Appendix);
and leverages smoothness properties of the functions ˆf and Ψ and the map (y, z) 7→x∗(y, z) as well
as the strong convexity of ˆf with respect to x.
Theorem 7. Suppose Assumptions 1, 2, 3 and 4 hold. Consider the sm-AGDA algorithm with
parameters p > ℓ, β ∈(0, 1], τ1 ∈(0,
1
p+ℓ] and τ2 > 0 chosen such that

c0 ≜−τ 2
2 ℓν + τ2


1 −ℓ

2τ2 −LΨτ2


≥0,
c′
0 ≜p

3β −
 2p2

p −ℓ+ 48β
p3

(p −ℓ)2


≥0,

for some constant ν > 0, where LΨ = ℓ

1 + p+ℓ

p−ℓ


. Then,

Vt −Vt+1 ≥c1∥∇x ˆf(xt, yt; zt)∥2 + c2∥∇yf(x∗(yt, zt), yt)∥2 + c3∥xt −zt∥2

+ c4⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ⟨c5∇yf(xt, yt) + c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩

+ c7∥∆x
t ∥2 + c8∥∆y
t ∥2,

(12)

for some constants {ci}8
i=1 ⊂R that are explicitly given in Appendix C, which may depend on ν, as
well as the problem and sm-AGDA parameters that can be chosen such that c1, c2, c3 > 0.

Proof. The proof is given in Appendix C.

With some specific choice of parameters in sm-AGDA, we can obtain simplifications to the coefficients
{ci}8
i=1 from Theorem 7 (explicitly given in Appendix C). As such, this yields the following corollary.

6


---Page Break---
Corollary 8. Under the premise of Theorem 7, let p = 2ℓ, τ1 ∈(0, 1

3ℓ], τ2 = τ1

48, β = αµτ2 for

α ∈(0,
1
406]. Then,

˜
At+1−˜
At
τ1
≤−˜Bt + ˜Ct+1 + ˜Dt+1 for all t ∈N, where ν = 12

τ1ℓand
˜At ≜τ1Vt,
˜Bt ≜τ1

5 ∥∇x ˆf(xt, yt; zt)∥2 + τ2

8 ∥∇yf(x∗(yt, zt), yt)∥2 + βp

8 ∥xt −zt∥2,

˜Ct+1 ≜
h

192βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 + 4ℓ

ν + 4c0ℓ2 + 2c′
0β2
τ 2
1 +

(p + ℓ)τ1 −1

τ1
i
⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩

+τ2⟨(1 + ℓτ2 + 2LΨτ2) ∇yf(xt, yt) −2∇yf(x∗(yt, zt), yt), ∆y
t ⟩,
˜Dt+1 ≜2ℓτ 2
1 ∥∆x
t ∥2 + 8ℓτ 2
2 ∥∆y
t ∥2.

Proof. The proof is given in Appendix D.

Next, we provide a concentration inequality which will be key to obtain our high-probability bounds.
Theorem 9. Let {Ft}t∈N be a filtration on (Ω, F, P). Let At, Bt, Ct, Dt be four stochastic processes
adapted to the filtration such that there exist σC, σD > 0 and τ1 > 0 such that for all t ∈N: (i)
Bt ≥0, (ii) E[eλCt+1 | Ft] ≤eλ2σ2
CBt for all λ > 0, (iii) E[eλDt+1 | Ft] ≤eλσ2
D for all

λ ∈
h
0,
1
σ2
D

i
and (iv) At+1−At

τ1
≤−Bt + Ct+1 + Dt+1. Then, for any ¯q ∈(0, 1], we have

P

 
τ1

2

T −1
X

t=0
Bt ≤(A0 −AT ) + τ1σ2
DT + 2τ1 max{2σ2
C, σ2
D} log
1

¯q

!

≥1 −¯q.

Proof. The proof is provided in Appendix E.

Remark 10. While the above concentration inequality seems tailored to the analysis of sm-AGDA, it
can also aid in deriving high probability bounds for many other first-order methods for nonconvex
minimax problems that outputs a randomized iterate; indeed, the majority of existing Lyapunov argu-
ments in the nonconvex setting are built upon constructing telescoping sums in line with Theorem 9,
e.g., stochastic alternating GDA [66] for NCPL minimax problems and optimistic GDA [45] for
strongly convex-strongly concave problems.

We next present our main result which provides a high-probability bound on the sm-AGDA iterates.
The main idea of the proof is to apply Theorem 9 to the processes introduced in Corollary 8.
Theorem 11. In the premise of Corollary 8, sm-AGDA iterates (xt, yt) for τ1 ≤
1
3ℓsatisfy

P

 
1
T

T −1
X

t=0


∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2
≤Q¯q,T ,

!

≥1 −¯q,
∀T ∈N,
∀¯q ∈(0, 1],

for some Q¯q,T = O

κ(∆0+b0)

τ1T
+ κ(δ2
x + δ2
y)


τ1ℓ+ 1

T log

1

¯q

explicitly stated in Appendix F,

where ∆0 ≜Φ(z0) −Φ∗, b0 ≜2 supx,y{ ˆf(x0, y; z0) −ˆf(x, y0; z0)}.

Proof Sketch. Let the stochastic processes At, Bt, Ct, Dt in Theorem 9 be chosen as At = ˜At,
Bt = ˜Bt, Ct = ˜Ct, Dt = ˜Dt where ˜At, ˜Bt, ˜Ct, ˜Dt are defined in Corollary 8 and τ1 > 0 be the
primal stepsize in sm-AGDA; according to Corollary 8, we have At+1−At

τ1
≤−Bt + Ct+1 + Dt+1
for t ∈N. Since ∆x
t and ∆y
t admit sub-Gaussian tails, it can be shown that the conditions of
Theorem 9 are satisfied for some appropriate constants σ2
C and σ2
D. Therefore, Theorem 9 implies a
tail bound on PT −1
t=0 ˜Bt. Using the relation between f and ˆf, one can also show that ∥∇xf(xt, yt)∥2+
κ∥∇yf(xt, yt)∥2 = O( ˜Bt), for all t ∈N. This last inequality allows to translate the tail bound for
PT −1
t=0 ˜Bt to a tail bound for PT −1
t=0 ∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2. The details of the proof is
provided in Appendix F of the supplementary material.

Remark 12. Suppose sm-AGDA, given in Alg. 1, is run for T iterations, and it outputs a randomly
selected iterate (xU, yU), where the random iteration index U is chosen uniformly at random from
the set {0, 1, . . . , T −1}, i.e., P(U = t) = 1/T for t = 0, 1, . . . , T −1. Theorem 11 implies that

P

 

∥∇xf(xU, yU)∥2 + κ∥∇yf(xU, yU)∥2 ≤Q¯q,T

!

≥1 −¯q.

Furthermore, in comparison with existing complexity bounds in expectation for sm-AGDA [66], our
quantile bound requires only an overhead of order O(ε−2 log(1/¯q)). Unless ¯q is very small, this is
typically negligible in comparison to the O(ε−4) already present in rates in expectation.

7


---Page Break---
Remark 13. In contrast to high-probability bounds derived from standard Markov-type arguments,
our approach achieves significantly better scaling with respect to both ¯q and ϵ. Specifically, consider
an oracle that can generate a sample (ˆx, ˆy) with E[∥∇f(ˆx, ˆy)∥] ≤ϵ after G(ϵ) iterations/stochastic
samples. In particular, [66] shows that one can take G(ϵ) = O

ℓκ2δ2

ϵ4
+ κℓ

ϵ2

for the sm-AGDA

algorithm assuming the variance of the stochastic gradient is bounded by δ2. A naive high-probability
bound could be constructed by ensuring E[∥∇f(ˆx, ˆy)∥] ≤¯q · ϵ and applying Markov’s inequality
to yield an ϵ-stationary point with probability at least 1 −¯q. However, this approach results in a
complexity bound of G(¯qϵ) = O

ℓκ2δ2

¯q4ϵ4 +
κℓ
¯q2ϵ2

, leading to a significantly worse dependence on ¯q
than ours. Alternatively, following the rationale in [19, 62], to generate a high-probability bound, one
can run the sm-AGDA algorithm m = Ω(log(1/¯q)) times in parallel; where in each run we generate
an ε/2-solution and among the solutions, we select the one with the smallest estimated gradient

norm. This would require mG(ϵ) = O

log(1/¯q) ℓκ2δ2

ϵ4
+ log(1/¯q) κℓ

ϵ2


iterations/stochastic samples.

In this approach, the logarithmic term log

1

¯q

multiplies the high-order O( 1

ε4 ) term, whereas in

our approach it only affects the second-order O( 1

ε2 ) term. Therefore, our results scale better with
respect to ¯q and ε. In addition, such a (multiple) parallel run approach, is often impractical in
streaming/online settings, where data arrives sequentially, and real-time processing is essential.
Corollary 14. Under the premise of Theorem 11, consider running the sm-AGDA method for
some fixed number of iterations T ∈N with parameters chosen as τ1 = min

1
3ℓ, 48√∆0+b0
√

T ℓδ2


and τ2 = τ1/48 where δ2 ≜δ2
x + δ2
y. Then, for any ¯q ∈(0, 1), sm-AGDA can compute an (ε, ε/√κ)
stationarity point with probability at least 1 −¯q when the number of iterations T is fixed to

Tε,¯q = O

(∆0+b0)ℓκ

ε2
+
δ2 log( 1

¯q)κ
ε2
+ δ2(∆0+b0)ℓκ2

ε4


which requires Tε,¯q stochastic gradient calls.

Proof. This is a direct consequence of Theorem 11, a proof is provided in Appendix G.

4
Numerical Illustrations

In this section, we illustrate the performance of sm-AGDA. We consider an NCPL problem with
synthetic data, as well as a nonconvex DRO problem using real datasets. For synthetic experiments,
we used an ASUS Laptop model Q540VJ with 13th Generation Intel Core i9-13900H using 16GB
RAM and 1TB SSD hard drive. For the DRO experiments, we used a high-performance computing
cluster with automatic GPU selection (NVIDIA RTX 3050, RTX 3090, A100, or Tesla P100) based
on GPU availability, ensuring optimal use of computational resources.
Synthetic experiments on an NCPL game. We consider the following NCPL problem:

min
x∈Rd1 max
y∈Rd2 m1
h
∥x∥2 + sin

3
p

∥x∥2 + 1
i
+ x⊤Ky −m2

∥y∥2 + 3 sin2(∥y∥)

,
(13)

which can be interpreted as a game between two players [48, 35] where m1, m2 > 0 are constants and
the symmetric matrix K is set randomly, similar to the standard bilinear game setting considered in
[35]. More specifically, we set K = 10 ˜K/∥˜K∥, ˜K = (M +M ⊤)/2 where M is a d×d matrix with
entries being i.i.d centered Gaussian having variance σ2. This problem is nonconvex in x (without
satisfying the PL condition in x). Though the exact gradient is known, we consider a stochastic
gradient oracle, which returns noisy gradients similar to the setting of [35, 11, 16, 2], i.e., for each
iteration t ∈{0, ..., T −1}, Gx(xt, yt; ξx
t+1) = ∇xf(xt, yt) + ξx
t+1 and Gy(xt+1, yt, ξy
t+1) =

∇yf(xt+1, yt) + ξy
t+1, with (ξx
t+1)t≥0
iid
∼N(0, δ2Id1) and (ξy
t+1)t≥0
iid
∼N(0, δ2Id2) where Id is
the d × d identity matrix and δ2 is some constant variance. This setting satisfies all our assumptions,
and our high-probability results (Theorem 11 and Coro. 14) are applicable. In this experiment, we fix
d1 = d2 = 30, and m1 = m2 = δ2 = σ2 = 1. The solution to this problem is (x∗, y∗) = (0, 0).

Experimental results. The parameters of the problem are explicitly available as µ = 2m2, and
ℓ= max{12m1, 8m2, ∥K∥}. To illustrate Theorem 11, we set β =
τ2µ
1600, τ2 =
τ1
48, p = 2ℓ
and we considered two cases: τ1 =
1
3ℓ(long step) and τ1 =
1
12ℓ(short step) to explore the
behavior of sm-AGDA for different stepsizes. We generated N = 25 sample paths for T = 10, 000
iterations, and on the left panel of Fig. 1, for each iteration t, we report the average of Mκ(t) ≜
∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2 over N = 25 realizations corresponding to different sample
paths, and the shaded region depicts the range statistic, i.e., for every fixed iteration t, we shade the

8


---Page Break---
Figure 1: NCPL game with τ1 =
1
3ℓ(long step) and τ1 =
1
12ℓ(short step). (Left) Average of Mk(t) over 25 sample paths vs. iterations t.
(Right) Average of I(t) over 25 sample paths vs. iterations t. In both plots, shaded regions depict the range statistic over 25 sample paths.

vertical line between the maximum and minimum values of Mκ(t) for the 25 paths. On the right
panel of Fig. 1, we also report the squared distance, I(t) ≜∥xt −x∗∥2 + ∥yt −y∗∥2, in a similar
manner. We observe that the range statistic for Mκ(t) diminishes to a value inversely proportional to
T as t →T; this is inline with our theoretical results in Theorem 11. The existence of sinusoidal
terms in the minimax objective is a source for oscillatory behavior in these figures. As t →T, both
Mκ(t) and I(t) exhibit oscillations, of which amplitude are smaller for the small step size compared
to the large step-size –at the expense of a slower convergence; the iterate paths are not as oscillatory.

Figure 2: Comparison of our theoretical upper bound in Theorem 11 and
the empirical stationarity measure of sm-AGDA for Problem (13). Results are
reported as cumulative distribution functions.

For T
=
10, 000 and τ1
=
1
3ℓfixed,
we
next
consider
the
path
averages
XT ≜
1
T
PT −1
t=0 Mκ(t). Indeed, based on
1000 sample paths, each for T = 10, 000 it-
erations, we compare the empirical quantiles
of XT with the theoretical upper bound Qq,T
on its quantiles (implied by Theorem 11).
Figure 2 shows the cumulative distribution
function (CDF) of the empirical distribution
alongside the theoretical explicit upper
bound Qq,T for the stationarity measure
∥∇xf(xU, yU) + κ∇yf(xU, yU)∥with U
uniform over {1, . . . , T}. The details of the empirical quantile estimation are provided Appendix
I due to space considerations. We observe that the empirical CDF has a sigmoid-like shape, while
the theoretical quantiles that lie above display a concave form. This difference may arise because
the theoretical quantile bounds Qq,T are designed to capture the worst-case behavior across the class
of NCPL problems, whereas this specific NCPL example may not represent the worst-case scenario.
Distributionally Robust nonconvex Logistic Regression. We consider the DRO problem

3
2
1
0
1
2
3
0.0

0.2

0.4

0.6

0.8

1.0

a9a - epoch=20

3
2
1
0
1
2
3
0.0

0.1

0.2

0.3

0.4

gisette - epoch=20

3
2
1
0
1
2
0.0

0.1

0.2

0.3

0.4

0.5

sido0 - epoch=20

3
2
1
0
1
2
3
0.0

0.2

0.4

0.6

0.8

a9a - epoch=250

3
2
1
0
1
2
3
0.0

0.2

0.4

0.6

0.8

gisette - epoch=550

3
2
1
0
1
2
0.0

0.1

0.2

0.3

0.4

0.5

sido0 - epoch=250

SAPD+
SMDAVR
smAGDA

Figure 3: Histograms for the stationarity measure log10 ∥∇f(xt, yt)∥2 for sm-AGDA and baseline algorithms (SAPD+, SMDA, and
SMDAVR) on a9a, gisette, and sido0 datasets. Each algorithm is run 200 times. First row reports histograms after 20 epochs, and second
row reports histograms at the end of training.

9


---Page Break---
min
x∈Rd1 max
y∈Y

 1

d2

d2
X

j=1
yjℓj(x; aj, bj) + r(x) −g(y)

,
(14)

where ℓj(x; aj, bj) = log
 
1 + exp
 
−bja⊤
j x

denotes the logistic loss tied to an input-output pair
(aj, bj) ∈Rd1 × {−1, 1}, and r(x) = λ1
Pd1
i=1

ωx2
i
1+ωx2
i a primal regularization for the learning model

x ∈Rd1. We allow the distribution y ∈Y ≜{y ∈Rd2 : y ≥0, 1⊤y = 1} to deviate from the
uniform distribution u ≜
1
d2 1 where 1 denotes the vector of ones, and we penalize the distance
between y and u through the regularization map g : y 7→λ2d2

2 ∥y −u∥2. We set the regularization
parameters as ω = 10, λ1 = 10e−4, and λ2 = 1. Since r is nonconvex with a Lipschitz gradient and
g is strongly convex, this is an NCSC problem.
Datasets, Algorithms and Hyperparameters. We consider three standard datasets for this problem,
which are summarized as follows: The sido0 dataset [50] has d1 = 4932 and d2 = 12678. The
gisette dataset [26] has d1 = 5000 and d2 = 6000. Finally, the a9a dataset [13] has d1 = 123
and d2 = 32561. We compare the performances of sm-AGDA against two other baselines that achieve
state-of-the-art performance in expectation for these datasets [72]. Specifically, we evaluate SAPD+,
which is a two-loop method where the subproblems are solved by the SAPD algorithm [73], and
SMDAVR, a variance reduced extension of SMDA algorithm [31]. Since (14) is constrained, we augment
sm-AGDA with a projection step in the update of the y variable onto the d2-dimensional simplex
and adopt the analogous stationarity metric ∥∇xf(xt, yt)∥2 + ∥PY ∇yf(xt, yt)∥2 for constrained
problems where PY is a projection to the dual domain Y . For all datasets, the primal stepsize τ1 of
sm-AGDA is tuned via a grid-search over {10−k, 1 ≤k ≤4}. The dual stepsize τ2 is set as τ2 = τ1

48.
Similarly, β is estimated through a grid-search over {10−k, 3 ≤k ≤5}. The parameter p is also
tuned similarly on a grid, our code is provided as a supplementary document for the details. For
other methods, our hyperparameters are tuned in accordance with [72].
Experimental results. In Figure 2, we plot histograms of our stationarity metric, across 200 runs in a
logarithmic scale. We report the stationarity measure both in early phase of the training (i.e. t = 20
epochs), and in later phases (i.e. t = 550 epochs for gisette and t = 250 epochs for a9a and
sido0). Our theoretical results are presented for unconstrained problems in the dual, therefore they
are not directly applicable to the DRO problem where the dual domain is constrained. That being
said, we observe that they are still predictive of performance in the DRO setting. More specifically,
Figure 2 is supportive of our high-probability complexity bounds for sm-AGDA, in the sense that
the distribution of the stationarity metric for sm-AGDA tends to concentrate. Notably, it outperforms
the concentration behaviour of the other baselines. Furthermore, we observe that histograms for all
baselines hardly evolve after 20 epochs. This is consistent with previous experiments carried on
these datasets [72] where performance was measured in terms of the decay of the average loss and its
standard deviation. As such, we conclude that sm-AGDA performs better both in the early phase and
the later stage. In our experience, we observed sm-AGDA could accomodate larger stepsizes compared
to the other algorithms, which may have contributed to its good performance.
5
Conclusion

Existing high-probability bounds only apply to convex/concave minimax problems or non-monotone
variational inequality problems under restrictive assumptions to our knowledge. We close this gap
by providing the first high-probability complexity guarantees for nonconvex/PL minimax problems
satisfying the PL-condition in the dual variable for the sm-AGDA method. We also provide numerical
results for an NCPL example and for nonconvex distributionally robust logistic regression.

Acknowledgements

Yassine Laguel, Yasa Syed and Mert Gürbüzbalaban’s research are supported in part by the grants
Office of Naval Research Award Numbers N00014-21-1-2244 and N00014-24-1-2628, National
Science Foundation (NSF) CCF-1814888 and NSF DMS-2053485. Necdet Serhat Aybat’s work was
supported in part by the Office of Naval Research Awards N00014-21-1-2271 and N00014-24-1-2666.

10


---Page Break---
References

[1] Jason Altschuler and Kunal Talwar. Privacy of noisy stochastic gradient descent: More iterations
without more privacy loss. Advances in Neural Information Processing Systems, 35:3788–3800,
2022.

[2] Necdet Serhat Aybat, Alireza Fallah, Mert Gurbuzbalaban, and Asuman Ozdaglar. Robust accel-
erated gradient methods for smooth strongly convex functions. SIAM Journal on Optimization,
30(1):717–751, 2020.

[3] Amir Beck and Aharon Ben-Tal. Duality in robust optimization: primal worst equals dual best.
Operations Research Letters, 37(1):1–6, 2009.

[4] Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust Optimization, volume 28.
Princeton University Press, 2009.

[5] Aleksandr Beznosikov, Boris Polyak, Eduard Gorbunov, Dmitry Kovalev, and Alexander
Gasnikov. Smooth monotone stochastic variational inequalities and saddle point problems: A
survey. European Mathematical Society Magazine, (127):15–28, 2023.

[6] Radu Ioan Bo¸t and Axel Böhm. Alternating proximal-gradient steps for (stochastic) nonconvex-
concave minimax problems. arXiv preprint arXiv:2007.13605, 2020.

[7] Radu Ioan Bo¸t, Panayotis Mertikopoulos, Mathias Staudigl, and Phan Tu Vuong. Minibatch
forward-backward-forward methods for solving stochastic variational inequalities. Stochastic
Systems, 11(2):112–139, 2021.

[8] Léon Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings of
COMPSTAT’2010, pages 177–186. Springer, 2010.

[9] Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine
learning. SIAM Review, 60(2):223–311, 2018.

[10] Nicolas Boumal and Pierre-antoine Absil. Rtrmc: A Riemannian trust-region method for
low-rank matrix completion. Advances in Neural Information Processing Systems, 24, 2011.

[11] Bugra Can, Mert Gurbuzbalaban, and Lingjiong Zhu. Accelerated linear convergence of
stochastic momentum methods in Wasserstein distances. In International Conference on
Machine Learning, pages 891–901. PMLR, 2019.

[12] Antonin Chambolle and Thomas Pock. A first-order primal-dual algorithm for convex problems
with applications to imaging. Journal of mathematical imaging and vision, 40:120–145, 2011.

[13] Chih-Chung Chang and Chih-Jen Lin. LIBSVM: a library for support vector machines. ACM
Transactions on Intelligent Systems and Technology (TIST), 2(3):1–27, 2011.

[14] Rishav Chourasia, Jiayuan Ye, and Reza Shokri. Differential privacy dynamics of Langevin
diffusion and noisy gradient descent. Advances in Neural Information Processing Systems,
34:14771–14781, 2021.

[15] Constantinos Daskalakis, Dylan J Foster, and Noah Golowich. Independent policy gradient
methods for competitive reinforcement learning. Advances in Neural Information Processing
Systems, 33:5527–5540, 2020.

[16] Alireza Fallah, Asuman Ozdaglar, and Sarath Pattathil. An optimal multistage stochastic
gradient method for minimax problems. In 2020 59th IEEE Conference on Decision and
Control (CDC), pages 3573–3579. IEEE, 2020.

[17] Maryam Fazel, Rong Ge, Sham Kakade, and Mehran Mesbahi. Global convergence of policy
gradient methods for the linear quadratic regulator. In International Conference on Machine
Learning, pages 1467–1476. PMLR, 2018.

[18] Saeed Ghadimi and Guanghui Lan. Optimal stochastic approximation algorithms for strongly
convex stochastic composite optimization I: A generic algorithmic framework. SIAM Journal
on Optimization, 22(4):1469–1492, 2012.

11


---Page Break---
[19] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex
stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013.

[20] Eduard Gorbunov, Marina Danilova, David Dobre, Pavel Dvurechenskii, Alexander Gasnikov,
and Gauthier Gidel. Clipped stochastic methods for variational inequalities with heavy-tailed
noise. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors,
Advances in Neural Information Processing Systems, volume 35, pages 31319–31332, 2022.

[21] Eduard Gorbunov, Nicolas Loizou, and Gauthier Gidel. Extragradient method: O(1/k) last-
iterate convergence for monotone variational inequalities and connections with cocoercivity. In
International Conference on Artificial Intelligence and Statistics, pages 366–402, 2022.

[22] Eduard Gorbunov, Abdurakhmon Sadiev, Marina Danilova, Samuel Horváth, Gauthier Gidel,
Pavel Dvurechensky, Alexander Gasnikov, and Peter Richtárik. High-probability convergence
for composite and distributed stochastic minimization and variational inequalities with heavy-
tailed noise. arXiv preprint arXiv:2310.01860, 2023.

[23] Mert Gürbüzbalaban, Yuanhan Hu, Umut Simsekli, and Lingjiong Zhu. Cyclic and randomized
stepsizes invoke heavier tails in SGD than constant stepsize. Transactions on Machine Learning
Research, August 2023.

[24] Mert Gürbüzbalaban, Andrzej Ruszczy´nski, and Landi Zhu. A stochastic subgradient method
for distributionally robust non-convex and non-smooth learning. Journal of Optimization Theory
and Applications, 194(3):1014–1041, 2022.

[25] Mert Gürbüzbalaban, Umut Simsekli, and Lingjiong Zhu. The heavy-tail phenomenon in SGD.
In International Conference on Machine Learning, pages 3964–3975. PMLR, 2021.

[26] Isabelle Guyon, Steve Gunn, Asa Ben-Hur, and Gideon Dror. Result analysis of the NIPS2003
feature selection challenge. Advances in Neural Information Processing Systems, 17, 2004.

[27] Nicholas JA Harvey, Christopher Liaw, Yaniv Plan, and Sikander Randhawa. Tight analyses for
non-smooth stochastic gradient descent. In Conference on Learning Theory, pages 1579–1613.
PMLR, 2019.

[28] Mark Herbster, Stephen Pasteris, and Lisa Tse. Online matrix completion with side information.
Advances in Neural Information Processing Systems, 33:20402–20414, 2020.

[29] Yu-Guan Hsieh, Franck Iutzeler, Jérôme Malick, and Panayotis Mertikopoulos. On the con-
vergence of single-call stochastic extra-gradient methods. Advances in Neural Information
Processing Systems, 32, 2019.

[30] Feihu Huang, Shangqian Gao, Jian Pei, and Heng Huang. Accelerated zeroth-order and first-
order momentum methods from mini to minimax optimization. Journal of Machine Learning
Research, 23(36):1–70, 2022.

[31] Feihu Huang, Xidong Wu, and Heng Huang. Efficient mirror descent ascent methods for
nonsmooth minimax problems. Advances in Neural Information Processing Systems, 34, 2021.

[32] Anatoli Juditsky, Arkadi Nemirovski, and Claire Tauvel. Solving variational inequalities with
stochastic mirror-prox algorithm. Stochastic Systems, 1(1):17–58, 2011.

[33] Yang Junchi, Xiang Li, and Niao He. Nest your adaptive algorithm for parameter-agnostic
nonconvex minimax optimization. In Advances in Neural Information Processing Systems,
2022.

[34] Nurdan Kuru, ¸S. ˙Ilker Birbil, Mert Gürbüzbalaban, and Sinan Yildirim. Differentially private
accelerated optimization algorithms. SIAM Journal on Optimization, 32(2):795–821, 2022.

[35] Yassine Laguel, Necdet Serhat Aybat, and Mert Gürbüzbalaban. High probability and risk-averse
guarantees for stochastic saddle point problems. accepted to Journal of Machine Learning
(JMLR), arXiv preprint arXiv:2304.00444, 2023.

[36] Guanghui Lan. First-order and stochastic optimization methods for machine learning, volume 1.
Springer, 2020.

12


---Page Break---
[37] Dongyang Li, Haobin Li, and Junyu Zhang. General procedure to provide high-probability
guarantees for stochastic saddle point problems. arXiv preprint arXiv:2405.03219, 2024.

[38] Haochuan Li, Yi Tian, Jingzhao Zhang, and Ali Jadbabaie. Complexity lower bounds for
nonconvex-strongly-concave min-max optimization. Advances in Neural Information Process-
ing Systems, 34:1792–1804, 2021.

[39] Xiang Li, Junchi Yang, and Niao He. Tiada: A time-scale adaptive algorithm for nonconvex
minimax optimization. In The Eleventh International Conference on Learning Representations,
https://openreview.net/forum?id=zClyiZ5V6sL, 2023.

[40] Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave
minimax problems. In International Conference on Machine Learning, pages 6083–6093.
PMLR, 2020.

[41] Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization.
In Conference on Learning Theory, pages 2738–2779. PMLR, 2020.

[42] Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in over-
parameterized non-linear systems and neural networks. Applied and Computational Harmonic
Analysis, 59:85–116, 2022. Special Issue on Harmonic Analysis and Machine Learning.

[43] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083,
2017.

[44] Gabriel Mancino-Ball and Yangyang Xu. Variance-reduced accelerated methods for decen-
tralized stochastic double-regularized nonconvex strongly-concave minimax problems. arXiv
preprint arXiv:2307.07113, 2023.

[45] Aryan Mokhtari, Asuman E Ozdaglar, and Sarath Pattathil. Convergence rate of o(1/k) for
optimistic gradient and extragradient methods in smooth convex-concave saddle point problems.
SIAM Journal on Optimization, 30(4):3230–3251, 2020.

[46] Hongseok Namkoong and John C Duchi. Stochastic gradient methods for distributionally robust
optimization with f-divergences. In Advances in Neural Information Processing Systems, pages
2208–2216, 2016.

[47] Angelia Nedich and Tatiana Tatarenko. Huber loss-based penalty approach to problems with
linear constraints. arXiv preprint arXiv:2311.00874, 2023.

[48] Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, Jason D Lee, and Meisam Razaviyayn.
Solving a class of non-convex min-max games using iterative first order methods. Advances in
Neural Information Processing Systems, 32, 2019.

[49] Balamurugan Palaniappan and Francis Bach. Stochastic variance reduction methods for saddle-
point problems. In Advances in Neural Information Processing Systems, pages 1416–1424,
2016.

[50] DTP AIDS Antiviral Screen program.
Sido0: a phamacology dataset.
https://www.
causality.inf.ethz.ch/data/SIDO.html, 2008.

[51] H Rafique, M Liu, Q Lin, and T Yang. Non-convex min–max optimization: provable algorithms
and applications in machine learning (2018). arXiv preprint arXiv:1810.02060, 1810.

[52] Harish Rajagopal. Multistage step size scheduling for minimax problems. Master’s the-
sis, ETH Zurich. URL: https://www.research-collection.ethz.ch/handle/20.500.
11850/572991, 2022.

[53] Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for
strongly convex stochastic optimization. arXiv preprint arXiv:1109.5647, 2011.

[54] Meisam Razaviyayn, Tianjian Huang, Songtao Lu, Maher Nouiehed, Maziar Sanjabi, and
Mingyi Hong. Nonconvex min-max optimization: Applications, challenges, and recent theoreti-
cal advances. IEEE Signal Processing Magazine, 37(5):55–66, 2020.

13


---Page Break---
[55] Abdurakhmon Sadiev, Marina Danilova, Eduard Gorbunov, Samuel Horváth, Gauthier Gidel,
Pavel Dvurechensky, Alexander Gasnikov, and Peter Richtárik. High-probability bounds
for stochastic optimization and variational inequalities: the case of unbounded variance. In
International Conference on Machine Learning, pages 29563–29648. PMLR, 2023.

[56] Umut Simsekli, Levent Sagun, and Mert Gürbüzbalaban. A tail-index analysis of stochastic
gradient noise in deep neural networks. In International Conference on Machine Learning,
pages 5827–5837. PMLR, 2019.

[57] Ju Sun, Qing Qu, and John Wright. Complete dictionary recovery over the sphere ii: Recovery
by riemannian trust-region method. IEEE Transactions on Information Theory, 63(2):885–914,
2017.

[58] J v. Neumann. Zur theorie der gesellschaftsspiele. Mathematische Annalen, 100(1):295–320,
1928.

[59] Emmanouil-Vasileios Vlatakis-Gkaragkounis, Lampros Flokas, and Georgios Piliouras. Solving
min-max optimization with hidden structure via gradient descent ascent. Advances in Neural
Information Processing Systems, 34:2373–2386, 2021.

[60] Yuanhao Wang and Jian Li. Improved algorithms for convex-concave minimax optimization.
Advances in Neural Information Processing Systems, 33:4800–4810, 2020.

[61] Linli Xu, James Neufeld, Bryce Larson, and Dale Schuurmans. Maximum margin clustering.
In Advances in Neural Information Processing systems, pages 1537–1544, 2005.

[62] Qiushui Xu, Xuan Zhang, Necdet Serhat Aybat, and Mert Gürbüzbalaban. A stochastic gda
method with backtracking for solving nonconvex (strongly) concave minimax problems. arXiv
preprint arXiv:2403.07806, 2024.

[63] Qiushui Xu, Xuan Zhang, Necdet Serhat Aybat, and Mert Gürbüzbalaban. A Stochastic GDA
Method With Backtracking For Solving Nonconvex (Strongly) Concave Minimax Problems.
arXiv e-prints, page arXiv:2403.07806, March 2024.

[64] Yan Yan, Yi Xu, Qihang Lin, Wei Liu, and Tianbao Yang. Optimal epoch stochastic gradient de-
scent ascent methods for min-max optimization. In H. Larochelle, M. Ranzato, R. Hadsell, M.F.
Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33,
pages 5789–5800, 2020.

[65] Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance reduction for a
class of nonconvex-nonconcave minimax problems. Advances in Neural Information Processing
Systems, 33:1153–1165, 2020.

[66] Junchi Yang, Antonio Orvieto, Aurelien Lucchi, and Niao He. Faster single-loop algorithms
for minimax optimization without strong concavity. In International Conference on Artificial
Intelligence and Statistics, pages 5485–5517. PMLR, 2022.

[67] TaeHo Yoon and Ernest K Ryu. Accelerated minimax algorithms flock together. arXiv preprint
arXiv:2205.11093, 2022.

[68] Zhuoning Yuan, Yan Yan, Milan Sonka, and Tianbao Yang. Large-scale robust deep auc
maximization: A new surrogate loss and empirical studies on medical image classification. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3040–3049,
2021.

[69] Yuxuan Zeng, Zhiguo Wang, Jianchao Bai, and Xiaojing Shen.
An accelerated stochas-
tic ADMM for nonconvex and nonsmooth finite-sum optimization.
arXiv preprint
arXiv:2306.05899, 2023.

[70] Jiawei Zhang, Peijun Xiao, Ruoyu Sun, and Zhiquan Luo. A single-loop smoothed gradient
descent-ascent algorithm for nonconvex-concave min-max problems. Advances in Neural
Information Processing Systems, 33:7377–7389, 2020.

14


---Page Break---
[71] Siqi Zhang, Junchi Yang, Cristóbal Guzmán, Negar Kiyavash, and Niao He. The complexity of
nonconvex-strongly-concave minimax optimization. In Uncertainty in Artificial Intelligence,
pages 482–492. PMLR, 2021.

[72] Xuan Zhang, Necdet Serhat Aybat, and Mert Gürbüzbalaban. SAPD+: An accelerated stochastic
method for nonconvex-concave minimax problems. Advances in Neural Information Processing
Systems, 35:21668–21681, 2022.

[73] Xuan Zhang, Necdet Serhat Aybat, and Mert Gürbüzbalaban. Robust accelerated primal-dual
methods for computing saddle points. SIAM Journal on Optimization, 34(1):1097–1130, 2024.

[74] Yuchen Zhang and Lin Xiao. Stochastic primal-dual coordinate method for regularized empirical
risk minimization. The Journal of Machine Learning Research, 18(1):2939–2980, 2017.

[75] Landi Zhu, Mert Gürbüzbalaban, and Andrzej Ruszczy´nski. Distributionally robust learning
with weakly convex losses: Convergence rates and finite-sample guarantees. arXiv preprint
arXiv:2301.06619, 2023.

15


---Page Break---
High-probability complexity guarantees
for nonconvex minimax problems

APPENDIX

The organization of the Appendix is as follows:

• In Section A, we summarize the key notations that are used throughout the main paper and
the Appendix.

• In Section B, we provide the proofs of Lemmas from Section 3 that were used for deriving
the approximate Lyapunov descent property.

• In Section C, we present the proof of Theorem 7 which provides an approximate descent
property in terms of the Lyapunov function for the sm-AGDA iterates.

• In Section D, we provide the proof of Corollary 8, which studies particular choice of
parameters within Theorem 7.

• In Section E we present a proof of Theorem 9 which provides a concentration inequality.

• In Section F, we provide the proof of Theorem 11 which yields high-probability results for
the sm-AGDA algorithm.

• In Section G, we provide a proof of Corollary 14 that provides a high-probability (iteration)
complexity bound for the sm-AGDA iterates.

• In Section H, we present supplementary lemmas that were used to derive the results.

• In Section I, we provide further details about numerical experiments.

A
Notation

The key notations that will be used throughout the Appendix is as follows:

• ˆf(x, y; z) = f(x, y) + p
2∥x −z∥2 denotes the auxiliary problem.

• Ψ(y; z) = minx ˆf(x, y; z) is the dual function of the auxiliary problem.

• Φ(x; z) = maxy ˆf(x, y; z) is the primal function of the auxiliary problem.

• P(z) = minx maxy ˆf(x, y; z) is the optimal value of the primal problem minx Φ(x; z)

• x∗(y, z) = argminx ˆf(x, y; z) for given y, z in the auxiliary function.

• x∗(z) = argminx Φ(x; z) is the unique optimal solution to the auxiliary primal problem.

• Y ∗(z) = argmaxy Ψ(y; z) is the set of optimal solutions to the auxiliary dual problem.

• y+(z) = y + τ2∇yf(x∗(y, z), y) denotes an update in y in the direction of the gradient of
the dual function, i.e., along the direction ∇Ψ(y; z) = ∇yf(x∗(y, z), y).

• ˆGx(x, y, ξ; z) ≜Gx(x, y, ξ) + p(x −z)

• ∆x
t = Gx(xt, yt, ξx
t+1) −∇xf(xt, yt) and ∆y
t = Gy(xt+1, yt, ξy
t+1) −∇yf(xt+1, yt) de-
note the gradient error, i.e., the difference between the stochastic estimates of the partial
gradients and the exact partial gradients.

B
Proofs of Lemmas from Section 3

B.1
Proof of Lemma 5

Lemma 5. Suppose Assumptions 1, 2 and 3 hold. Consider sm-AGDA, stated in Algorithm 1, with
τ1 ∈(0,
1
p+ℓ] and β ∈(0, 1]. For any t ∈N, we have:

ˆf(xt+1, yt+1; zt+1) −ˆf(xt, yt; zt)

16


---Page Break---
≤−τ1

2 ∥∇x ˆf(xt, yt; zt)∥2 + τ2


1 + ℓ

2τ2


∥∇yf(xt+1, yt)∥2 −p

2β ∥zt −zt+1∥2

+ τ1((p + ℓ)τ1 −1)⟨∆x
t , ∇x ˆf(xt, yt; zt)⟩+ τ2(1 + ℓτ2)⟨∆y
t , ∇yf(xt+1, yt)⟩

+ p + ℓ

2
τ 2
1 ∥∆x
t ∥2 + ℓτ 2
2
2 ∥∆y
t ∥2.

Proof. Since, ˆf(·, y; z) is (p + ℓ)-smooth, we have

ˆf(xt+1, yt; zt) −ˆf(xt, yt; zt)

≤⟨xt+1 −xt, ∇x ˆf(xt, yt; zt)⟩+ p + ℓ

2
∥xt+1 −xt∥2

= −τ1⟨ˆGx(xt, yt; zt), ∇x ˆf(xt, yt; zt)⟩+ p + ℓ

2
τ 2
1 ∥ˆGx(xt, yt; zt)∥2.

= (p + ℓ

2
τ 2
1 −τ1)∥∇x ˆf(xt, yt; zt)∥2 −(τ1 + (p + ℓ)τ 2
1 )⟨∆x
t , ∇x ˆf(xt, yt; zt)⟩

+ p + ℓ

2
∥∆x
t ∥2,

where last equality follows from ˆGx(xt, yt; zt) = ∇x ˆf(xt, yt; zt) + ∆x
t . Hence, for τ1 ≤
1
p+ℓ, we
have

ˆf(xt+1, yt; zt) −ˆf(xt, yt; zt)

≤−τ1

2 ∥∇x ˆf(xt, yt; zt)∥2 + τ1((p + ℓ)τ1 −1)⟨∆x
t , ∇x ˆf(xt, yt; zt)⟩+ p + ℓ

2
τ 2
1 ∥∆x
t ∥2.

(15)

Similarly, we observe that for all, ∇y ˆf(x, y; z) = ∇yf(x, y), for all x, y, z, which together with the
smoothness of f(x, ·) implies

ˆf(xt+1, yt+1; zt) −ˆf(xt+1, yt; zt)

≤⟨∇y ˆf(xt+1, yt; zt), yt+1 −yt⟩+ ℓ

2∥yt+1 −yt∥2

= τ2⟨∇yf(xt+1, yt), Gy(xt+1, yt, ξy
t+1)⟩+ ℓ

2τ 2
2 ∥Gy(xt+1, yt, ξy
t+1)∥2

=τ2


1 + ℓ

2τ2


∥∇yf(xt+1, yt)∥2 + τ2(1 + ℓτ2)⟨∇yf(xt+1, yt), ∆y
t ⟩+ ℓτ 2
2
2 ∥∆y
t ∥2,

(16)

where we used again the identity Gy(xt+1, yt, ξy
t+1) = ∇y ˆf(xt+1, yt; zt) + ∆y
t .

Finally, we observe from the sm-AGDA update rule zt+1 −zt = β(xt+1 −zt) that 1

β (zt+1 −zt) =
xt+1 −zt and (1 −β)(xt+1 −zt) = (xt+1 −zt) −(zt+1 −zt) = xt+1 −zt+1. This gives

ˆf(xt+1, yt+1; zt+1) −ˆf(xt+1, yt+1; zt) = p

2

∥xt+1 −zt+1∥2 −∥xt+1 −zt∥2

= p

2


∥(1 −β)(xt+1 −zt)∥2 −1

β2 ∥zt+1 −zt∥2


= p

2

(1 −β)2

β2
∥zt+1 −zt∥2 −1

β2 ∥zt+1 −zt∥2


≤−p

2β ∥zt −zt+1∥2,

(17)

where we used 0 < β ≤1. Therefore, summing up (15), (16), (17) yields the claim.

B.2
Proof of Lemma 6

Lemma 6. Suppose Assumptions 1, 2 and 3 hold, and p ≥ℓ. Then, for any t ∈N,

Ψ(yt+1; zt+1) −Ψ(yt; zt) ≥τ2⟨∇yf(x∗(yt, zt), yt), ∇yf(xt+1, yt)⟩+ τ2⟨∇yf(x∗(yt, zt), yt), ∆y
t ⟩

17


---Page Break---
−LΨ

2 τ 2
2
 
∥∇yf(xt+1, yt)∥2 + 2⟨∇yf(xt+1, yt), ∆y
t ⟩+ ∥∆y
t ∥2

+ p

2⟨zt+1 −zt, zt+1 + zt −2x∗(yt+1, zt+1)

,

where LΨ ≜ℓ

1 + p+ℓ

p−ℓ


and the map x∗(·, ·) is defined in (11).

Proof. By Lemma 19, Ψ is LΨ-smooth in y for any given z ∈Rd1. Then, using ∇yΨ(yt; zt) =
∇yf(x∗(yt, zt), yt), we obtain

Ψ(yt+1, zt) −Ψ(yt, zt) ≥⟨∇yf(x∗(yt, zt), yt; zt), yt+1 −yt⟩−LΨ

2 ∥yt+1 −yt∥2

≥τ2⟨∇yf(x∗(yt, zt), yt), ∇yf(xt+1, yt)⟩+ τ2⟨∇yf(x∗(yt, zt), yt), ∆y
t ⟩

−LΨ

2 τ 2
2
 
∥∇yf(xt+1, yt)∥2 + 2⟨∇yf(xt+1, yt), ∆y
t ⟩+ ∥∆y
t ∥2
.
(18)
Furthermore, by definition of Ψ, we also have

Ψ(yt+1; zt+1) −Ψ(yt+1; zt) = ˆf(x∗(yt+1, zt+1), yt+1; zt+1) −ˆf(x∗(yt+1, zt), yt+1; zt)

≥ˆf(x∗(yt+1, zt+1), yt+1; zt+1) −ˆf(x∗(yt+1, zt+1), yt+1; zt)

= p

2

∥zt+1 −x∗(yt+1, zt+1)∥2 −∥zt −x∗(yt+1, zt+1)∥2

= p

2(zt+1 −zt)⊤[zt+1 + zt −2x∗(yt+1, zt+1)].
(19)

Summing (18) and (19), we conclude.

C
Proof of Theorem 7

Before we move on to the proof of Theorem 7, we first provide a result from [70, 64] that quantifies the
change in the function P over the iterations. We also provide its proof for the sake of completeness.

Lemma 15 (Lemma B.7 in [70]). Suppose Assumptions 1, 2 and 3 hold. Consider the sm-AGDA
iterate sequence (xt, yt, zt)t∈N for p > ℓ, and let Y ∗(z) ≜argmaxy Ψ(y; z) denote the set of
maximizers of Ψ(·, z) for given z. For any t ∈N and y∗(zt+1) ∈Y ∗(zt+1), it holds that

P(zt+1) −P(zt) ≤p

2⟨zt+1 −zt, zt+1+zt −2x∗(y∗(zt+1), zt)⟩.

Proof. Let y∗(zt+1) ∈Y ∗(zt+1) and y∗(zt) ∈Y ∗(zt) be two arbitrary maximizers. We have

P(zt+1) −P(zt) = min
x max
y
ˆf(x, y; zt+1) −min
x max
y
ˆf(x, y; zt)

= max
y
min
x
ˆf(x, y; zt+1) −max
y
min
x
ˆf(x, y; zt)

= Ψ(y∗(zt+1); zt+1) −Ψ(y∗(zt); zt)
≤Ψ(y∗(zt+1); zt+1) −Ψ(y∗(zt+1); zt)

= ˆf(x∗(y∗(zt+1), zt+1), y∗(zt+1); zt+1) −ˆf(x∗(y∗(zt+1); zt), y∗(zt+1); zt)

≤ˆf(x∗(y∗(zt+1), zt), y∗(zt+1); zt+1) −ˆf(x∗(y∗(zt+1); zt), y∗(zt+1); zt)

= p

2(zt+1 −zt)⊤[zt+1+zt −2x∗(y∗(zt+1), zt)].

The first and the third equality above hold by the definition of P(z) and Ψ(y; z) functions; on the
other hand, for the second equality, one needs strong duality to hold. This interchange is valid and can
be justified by the simple fact that ˆf is strongly convex in x (therefore, it satisfies the PL condition in
x), and it also satisfies the PL condition in y according to our assumption 2. It has been established in
[65, Lemma 2.1] that the double-sided PL property allows one for the min-max switch.

18


---Page Break---
Equipped with this lemma, the stage is set to prove Theorem 7 from Section 3. We first restate an
extended version of this theorem, where the constants {ci}8
i=1 are provided explicitly.

Theorem 7. Suppose Assumptions 1, 2, 3 and 4 hold. Consider the sm-AGDA algorithm with
parameters p > ℓ, β ∈(0, 1], τ1 ∈(0,
1
p+ℓ] and τ2 > 0 chosen such that

c0 ≜−τ 2
2 ℓν + τ2


1 −ℓ

2τ2 −LΨτ2


≥0,
c′
0 ≜p

3β −
 2p2

p −ℓ+ 48β
p3

(p −ℓ)2


≥0

for some constant ν > 0, where LΨ = ℓ

1 + p+ℓ

p−ℓ


. Then,

Vt −Vt+1 ≥c1∥∇x ˆf(xt, yt; zt)∥2 + c2∥∇yf(x∗(yt, zt), yt)∥2 + c3∥xt −zt∥2

+ c4⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ⟨c5∇yf(xt, yt) + c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩

+ c7∥∆x
t ∥2 + c8∥∆y
t ∥2,
(20)

where the coefficients c1 to c8 have the following forms:

c1 = τ1

2 −2

1
(p −ℓ)2 + τ 2
1

 
c0ℓ2 + ℓ

ν + 48βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2


−


c′
0β2 + ℓ

6(1 + ℓτ2 + 2LΨτ2)

τ 2
1 ,

c2 = c0

2 −
24βp
(p −ℓ)µ


1 + τ2 2pℓ

p −ℓ

2
,
c3 = c′
0β2/2,

c4 = −


192βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 + 4ℓ

ν + 4c0ℓ2 + 2c′
0β2


τ 2
1 −

(p + ℓ)τ1 −1

τ1,

c5 = −τ2(1 + ℓτ2 + 2LΨτ2),
c6 = 2τ2,

c7 = −


96βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 + 2ℓ

ν + p + ℓ

2
+ 2c0ℓ2 + ℓ

6(1 + ℓτ2 + 2LΨτ2) + c′
0β2

τ 2
1 ,

c8 = −


48βp
p + ℓ

p −ℓ

2
+ ℓ

2 + LΨ+3ℓ(1 + ℓτ2 + 2LΨτ2)

τ 2
2 .

Furthermore, the sm-AGDA parameters can be chosen such that c1, c2, c3 > 0.

Proof. Combining the inequalities in Lemmas 5, 6 and 15 gives us a lower bound of the form:

Vt −Vt+1 ≥A1 + A2 + A3 + A4 + A5 + A6,
(21)

for any y∗(zt+1) ∈Y ∗(zt+1) appearing in A3, where

A1 = τ1

2 ∥∇x ˆf(xt, yt; zt)∥2 + τ2


1 −ℓ

2τ2 −LΨτ2


∥∇yf(xt+1, yt)∥2 + p

2β ∥zt −zt+1∥2,

A2 = 2τ2⟨∇yf(x∗(yt, zt), yt) −∇yf(xt+1, yt), ∇yf(xt+1, yt)⟩,
A3 = 2p⟨zt+1 −zt, x∗(y∗(zt+1), zt) −x∗(yt+1, zt+1)⟩

A4 = −τ1((p + ℓ)τ1 −1)⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩,

A5 = ⟨2τ2∇yf(x∗(yt, zt), yt) −τ2(1 + ℓτ2 + 2LΨτ2)∇yf(xt+1, yt), ∆y
t ⟩,

A6 = −p + ℓ

2
τ 2
1 ∥∆x
t ∥2 −ℓτ 2
2
2 ∥∆y
t ∥2 −LΨτ 2
2 ∥∆y
t ∥2.

Next, we provide lower bounds for several terms in the above inequality, including A2, A3, A5,
∥∇yf(xt+1, yt)∥2 and ∥zt+1 −zt∥2. At the end, using these bounds within (21), we will be able to
establish a descent property for {Vt}, which will allow to apply our concentration result (Theorem 9)
for deriving the desired high probability bounds.

Lower bound for A2. Using Cauchy-Schwarz inequality and ℓ-smoothness of f, we have

A2 ≥−2τ2∥∇yf(x∗(yt, zt), yt) −∇yf(xt+1, yt)∥∥∇yf(xt+1, yt)∥
≥−2τ2ℓ∥xt+1 −x∗(yt, zt)∥∥∇yf(xt+1, yt)∥

≥−τ 2
2 ℓν∥∇yf(xt+1, yt)∥2 −ℓν−1∥xt+1 −x∗(yt, zt)∥2,
∀ν > 0,

19


---Page Break---
where last line follows from Young’s inequality. Thus, by Lemma 20, we obtain for any ν > 0

A2 ≥−τ 2
2 ℓν∥∇yf(xt+1, yt)∥2 −2ℓ

ν


1 +
1
τ 2
1 (p −ℓ)2


τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2

−4ℓ

ν τ 2
1 ⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩−2ℓ

ν τ 2
1 ∥∆x
t ∥2.
(22)

Lower bound for A3. By Cauchy-Schwarz inequality and Lemma 21,

A3 = 2p⟨zt+1 −zt, x∗(y∗(zt+1), zt) −x∗(y∗(zt+1), zt+1)⟩
+ 2p⟨zt+1 −zt, x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)⟩
≥−2p∥zt+1 −zt∥∥x∗(y∗(zt+1), zt) −x∗(y∗(zt+1), zt+1)∥
+ 2p⟨zt+1 −zt, x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)⟩

≥−2p2

p −ℓ∥zt+1 −zt∥2 + 2p⟨zt+1 −zt, x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)⟩.

Hence, using Young’s inequality, for all β > 0, we obtain

A3 ≥−
 p

6β + 2p2

p −ℓ


∥zt+1 −zt∥2 −6βp∥x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)∥2.
(23)

We now lower bound the second term on the right-hand side of (23).
First, note that we
have x∗(y∗(zt+1), zt+1) = x∗(zt+1), which follows from the fact that minx maxy ˆf(x, y; z) =
maxy minx ˆf(x, y; z) for all z since ˆf is strongly convex in x and satisfies the PL condition in As-

sumption 2. Hence, using the inequality
PN
i=1 wi

2
≤N PN
i=1 ∥wi∥2, which holds for any

{wi} ∈Rd1 and N ≥1, we get

∥x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)∥2

≤
4∥x∗(zt+1) −x∗(zt)∥2 + 4∥x∗(zt) −x∗(y+
t (zt), zt)∥2

+4∥x∗(y+
t (zt), zt) −x∗(yt+1, zt)∥2 + 4∥x∗(yt+1, zt) −x∗(yt+1, zt+1)∥2.

By Lemma 21 and Lemma 23, we observe that

4∥x∗(zt+1) −x∗(zt)∥2 + 4∥x∗(yt+1, zt) −x∗(yt+1, zt+1)∥2 ≤
8p2

(p −ℓ)2 ∥zt+1 −zt∥2.

Using Lemma 24, we also have

4∥x∗(zt) −x∗(y+
t (zt), zt)∥2 ≤
4
(p −ℓ)µ


1 + τ2
2pℓ
p −ℓ

2
∥∇yf(x∗(yt, zt), yt)∥2.

Finally, using Lemma 18, we get

4∥x∗(y+
t (zt), zt) −x∗(yt+1, zt)∥2

≤4
p + ℓ

p −ℓ

2

∥y+
t (zt) −yt+1∥2

= 4
p + ℓ

p −ℓ

2

τ 2
2 ∥∇yf(x∗(yt, zt), yt) −Gy(xt+1, yt, ξy
t+1)∥2

= 4
p + ℓ

p −ℓ

2

τ 2
2 ∥∇yf(x∗(yt, zt), yt) −(∇yf(xt+1, yt) + ∆y
t )∥2,

≤4
p + ℓ

p −ℓ

2

τ 2
2
 
2ℓ2∥x∗(yt, zt) −xt+1∥2 + 2∥∆y
t ∥2
,

where the last inequality stems from the ℓ-smoothness of f. In view of Lemma 20, we may further
upper bound the above quantity as follows:

4∥x∗(y+
t (zt), zt) −x∗(yt+1, zt)∥2

20


---Page Break---
≤16
p + ℓ

p −ℓ

2

ℓ2τ 2
1 τ 2
2
h
1
τ 2
1 (p −ℓ)2 + 1

∥∇x ˆf(xt, yt; zt)∥2 + 2⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ∥∆x
t ∥2i

+ 8
p + ℓ

p −ℓ

2

τ 2
2 ∥∆y
t ∥2.

Summing up the three intermediate inequalities we established above, we obtain

∥x∗(y∗(zt+1), zt+1) −x∗(yt+1, zt+1)∥2

≤
8p2

(p −ℓ)2 ∥zt+1 −zt∥2 +
4
(p −ℓ)µ


1 + τ2ℓ+ τ2ℓ(p + ℓ)

p −ℓ

2
∥∇yf(x∗(yt, zt), yt)∥2

+ 16
p + ℓ

p −ℓ

2

ℓ2

1
τ 2
1 (p −ℓ)2 + 1

τ 2
1 τ 2
2 ∥∇x ˆf(xt, yt; zt)∥2

+ 32
p + ℓ

p −ℓ

2

ℓ2τ 2
1 τ 2
2 ⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩

+ 16
p + ℓ

p −ℓ

2

ℓ2τ 2
1 τ 2
2 ∥∆x
t ∥2 + 8(p + ℓ)2

(p −ℓ)2 τ 2
2 ∥∆y
t ∥2.

In conclusion for A3, using (23) and the above inequality, we obtain after some rearrangement

A3 ≥−
 p

6β + 2p2

p −ℓ+ 48β
p3

(p −ℓ)2


∥zt+1 −zt∥2 −


96βp
p + ℓ

p −ℓ

2

ℓ2τ 2
1 τ 2
2


∥∆x
t ∥2

−

 
24βp
(p −ℓ)µ


1 + τ2
2pℓ
p −ℓ

2!

∥∇yf(x∗(yt, zt), yt)∥2

−


96βp
p + ℓ

p −ℓ

2

ℓ2

1
τ 2
1 (p −ℓ)2 + 1

τ 2
1 τ 2
2


∥∇x ˆf(xt, yt; zt)∥2

−


192βp
p + ℓ

p −ℓ

2

ℓ2τ 2
1 τ 2
2


⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩−


48βp
p + ℓ

p −ℓ

2

τ 2
2


∥∆y
t ∥2.

(24)

Lower bound for A5. Below we first bound ⟨∇yf(xt+1, yt), ∆y
t ⟩, i.e.,

⟨∇yf(xt+1, yt), ∆y
t ⟩= ⟨∇yf(xt, yt), ∆y
t ⟩+ ⟨∇yf(xt+1, yt) −∇yf(xt, yt), ∆y
t ⟩

≤⟨∇yf(xt, yt), ∆y
t ⟩+
1
12τ2ℓ∥∇yf(xt+1, yt) −∇yf(xt, yt)∥2 + 3τ2ℓ∥∆y
t ∥2

≤⟨∇yf(xt, yt), ∆y
t ⟩+ ℓτ 2
1
12τ2 ∥ˆGx(xt, yt, ξx
t+1; zt)∥2 + 3τ2ℓ∥∆y
t ∥2

≤⟨∇yf(xt, yt), ∆y
t ⟩+ ℓτ 2
1
6τ2 ∥∇x ˆf(xt, yt; zt)∥2 + ℓτ 2
1
6τ2 ∥∆x
t ∥2 + 3τ2ℓ∥∆y
t ∥2,

where the first inequality follows from Young’s inequality, the second inequality follows from ∇yf being
ℓ-smooth and the update rule xt+1 = xt −τ1 ˆGx(xt, yt, ξx
t+1; zt), and in the third inequality we use ∆x
t =
ˆGx(xt, yt, ξx
t+1; zt) −∇x ˆf(xt, yt; zt). Thus, since −τ2(1 + ℓτ2 + 2LΨτ2) < 0, for any τ2 > 0, we obtain

A5 ≥−τ2(1 + ℓτ2 + 2LΨτ2)
ℓτ 2
1
6τ2 ∥∇x ˆf(xt, yt; zt)∥2 + ℓτ 2
1
6τ2 ∥∆x
t ∥2 + 3τ2ℓ∥∆y
t ∥2

+ ⟨2τ2∇yf(x∗(yt, zt), yt) −τ2(1 + ℓτ2 + 2LΨτ2)∇yf(xt, yt), ∆y
t ⟩.

(25)

Lower bound for ∥∇yf(xt+1, yt)∥2. Using Lemma 20 we can lower bound ∥∇yf(xt+1, yt)∥2 as follows:

∥∇yf(xt+1, yt)∥2 ≥1

2∥∇yf(x∗(yt, zt), yt)∥2 −∥∇yf(xt+1, yt) −∇yf(x∗(yt, zt), yt)∥2

≥1

2∥∇yf(x∗(yt, zt), yt)∥2 −ℓ2∥xt+1 −x∗(yt, zt)∥2

≥1

2∥∇yf(x∗(yt, zt), yt)∥2

−2ℓ2τ 2
1


1
τ 2
1 (p −ℓ)2 + 1

∥∇x ˆf(xt, yt; zt)∥2 + 2⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ∥∆x
t ∥2


.

21


---Page Break---
We observe that ∥∇yf(xt+1, yt)∥2 appears in both A1 and the lower bound given in (22) for A2; hence,
grouping the two terms together and using the definition of c0 ≥0, we get


τ2


1 −ℓ

2τ2 −LΨτ2


−τ 2
2 ℓν

∥∇yf(xt+1, yt)∥2 = c0∥∇yf(xt+1, yt)∥2

≥c0

2 ∥∇yf(x∗(yt, zt), yt)∥2 −2c0ℓ2

1
(p −ℓ)2 + τ 2
1


∥∇x ˆf(xt, yt; zt)∥2

−2c0ℓ2τ 2
1

2⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ∥∆x
t ∥2
.

(26)

Lower bound for ∥zt+1 −zt∥2. Since zt+1 = zt + β(xt+1 −zt) for some β > 0, we observe that:

∥zt+1 −zt∥2 = β2∥xt+1 −zt∥2

= β2∥xt −zt −τ1 ˆGx(xt, yt, ξx
t+1; zt)∥2

≥β2
1

2∥xt −zt∥2 −τ 2
1 ∥ˆGx(xt, yt, ξx
t+1; zt)∥2

,

and since ˆGx(xt, yt, ξx
t+1; zt) = ∇x ˆf(xt, yt; zt) + ∆x
t , we obtain

∥zt −zt+1∥2 ≥β2

2 ∥xt −zt∥2 −β2τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2 −2β2τ 2
1 ⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩−β2τ 2
1 ∥∆x
t ∥2.

Note that ∥zt+1 −zt∥2 appears in both A1 and the lower bound given in (24) for A3; hence, grouping the two
terms together and using the definition of c′
0 ≥0 Using (27), we obtain
 p

2β −p

6β −2p2

p −ℓ−48β
p3

(p −ℓ)2


∥zt+1 −zt∥2 = c′
0∥zt+1 −zt∥2

≥−c′
0β2τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2 + c′
0
β2

2 ∥xt −zt∥2 −2c′
0β2τ 2
1 ⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩−c′
0β2τ 2
1 ∥∆x
t ∥2.

We conclude by combining all these lower bounds, i.e., the claimed Lyapunov descent inequality follows directly
from summing (21), (22), (24), (25), (26) and (27). Finally, it follows after straightforward computations that
there exist choice of sm-AGDA parameters which yield c1, c2, c3 > 0 while satisfying the conditions c0 ≥0 and
c′
0 ≥0; in fact Corollary 8 provides such sm-AGDA parameters explicitly. This completes the proof.

D
Proof of Corollary 8

The lower bound provided in Theorem 7 resembles the descent property we require for our concentra-
tion result in Theorem 9. To allow for its proper application, we develop a stepsize policy inspired
by [66].

Corollary 8. Under the premise of Theorem 7, consider the parameters p = 2ℓ, τ1 ∈(0, 1

3ℓ],

τ2 = τ1

48, β = αµτ2 for any α ∈(0,
1
406]. Then,

˜
At+1−˜
At
τ1
≤−˜Bt + ˜Ct+1 + ˜Dt+1 for all t ∈N,
where ν = 12

τ1ℓand

˜At ≜τ1Vt,
˜Bt ≜τ1

5 ∥∇x ˆf(xt, yt; zt)∥2 + τ2

8 ∥∇yf(x∗(yt, zt), yt)∥2 + βp

8 ∥xt −zt∥2

˜Ct+1 ≜
h

192βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 + 4ℓ

ν + 4c0ℓ2 + 2c′
0β2
τ 2
1 +

(p + ℓ)τ1 −1

τ1
i
⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩

+τ2⟨(1 + ℓτ2 + 2LΨτ2) ∇yf(xt, yt) −2∇yf(x∗(yt, zt), yt), ∆y
t ⟩,
˜Dt+1 ≜2ℓτ 2
1 ∥∆x
t ∥2 + 8ℓτ 2
2 ∥∆y
t ∥2.

Proof. In view of Theorem 7, it suffices to prove that setting p = 2ℓ, τ1 ∈(0, 1

3ℓ], τ2 = τ1

48, β = αµτ2
for for some positive α ≤
1
406 and ν =
12
τ1ℓleads to both c0 ≥0 and c′
0 ≥0; furthermore, we also
need to show that this choice of parameters implies the following lower bounds:

22


---Page Break---
(i) c1 ≥τ1

5 ,
(ii) c2 ≥τ2

8 ,
(iii) c3 ≥pβ

8 ,
(iv) c7 ≥−2ℓτ 2
1 ,
(v) c8 ≥−8ℓτ 2
2 .

First, we show that our parameter choice implies that c0, c′
0 ≥0. Noting that LΨ = 4ℓ, using τ1 ≤
1
3ℓ,
we may bound c0 from above and below as follows:

τ2 ≥c0 ≜−τ 2
2 ℓν + τ2


1 −ℓ

2τ2 −LΨτ2



= τ2


1 −
 ℓ

2 + LΨ + ℓν
 τ1

48



≥τ2


1 −
9
2 · 144−1

4


≥1

2τ2 ≥0.
(27)

Moreover, since β = αµτ2 and τ2 ≤
1
144ℓ, we get β ≤
α
144 using κ ≥1. Hence, for α ≤
1
406,

0 ≤2p2

p −ℓ+ 48β
p3

(p −ℓ)2 = 8ℓ(1 + 48β) =

 α

36(1 + α/3)
 p

β ≤
p
12β ,
(28)

where the last inequality follows from α ≤
1
406 ≤3

2(
√

5 −1); therefore, c′
0 ∈[ p

4β , p

3β ].

Next, we prove bounds on c1, c2, c3, c7, c8 separately.

Proof of part (i). Since p = 2ℓ, ν = 12

τ1ℓand τ1≤1

3ℓ, we first observe that

2ℓ

ν


1
(p −ℓ)2 + τ 2
1


= τ1

6 (τ 2
1 ℓ2 + 1) ≤τ1

6

1

9 + 1


= 5

27τ1.
(29)

Furthermore, using τ2 = τ1

48, and β = αµτ2, we obtain

96βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2


1
(p −ℓ)2 + τ 2
1


=


96βℓ2p(p + ℓ)2

(p −ℓ)4


1 + τ 2
1 (p −ℓ)2τ 2
2
τ1


τ1

≤


96 · ατ2µ · 18ℓ

1 + 1

9

 τ 2
2
τ1


τ1

≤1920 · α · τ 3
2 ℓ2

τ1
· µ
ℓ· τ1

≤
5α
2592τ1,
(30)

where last line follows from the fact that µ/ℓ≤1 and τ 3
2 ℓ2

τ1
= τ 2
1 ℓ2

483 ≤
1
1442·48, in which we used
τ1 ≤
1
3ℓ.

Since τ2 ≥c0 ≥0 and −2ℓ2 
1
(p−ℓ)2 + τ 2
1

≥−20

9 , using τ2 = τ1

48, we obtain

−2c0ℓ2

1
(p −ℓ)2 + τ 2
1



≥−20τ2

9
= −5

108τ1.
(31)

Using LΨ = 4ℓ, τ1 ≤
1
3ℓand τ2 ≤
1
144ℓ, we get

ℓ
6(1 + ℓτ2 + 2LΨτ2)τ 2
1 ≤ℓ

6


1 +
1
144 +
8
144


τ 2
1 ≤1

18


1 +
1
144 +
8
144


τ1 = 17

288τ1. (32)

Using τ1 ≤
1
3ℓ, τ2 ≤
1
144ℓ, this implies

−c′
0τ 2
1 β2 ≥−p

3βτ 2
1 ≥−2ℓ

3 · αµ

144ℓ
1
3ℓτ1 ≥−α

648τ1.
(33)

Combining equations (29), (30), (31), (32) and (33), we obtain

c1 ≥
1

2 −5

27−5α

2592 −
5
108−17

288 −α

648


τ1 =
1

2 −25

108 −17 + α

288


τ1

= 181 −3α

864
τ1 ≥60 −α

288
≥τ1

5 ,
(34)

23


---Page Break---
for α ≤
1
406 < 2.4, which completes the proof of part (i).

Proof of part (ii). Due to our parameter choice, we first note that

24βp
(p −ℓ)µ


1 + τ2
2pℓ
p −ℓ

2
= 48α

1 + τ2
2pℓ
p −ℓ

2
τ2 ≤96ατ2 ≤τ2

8 ,
(35)

where last line follows from τ2 ≤
1
144ℓand α ≤
1
406. Finally, (27) implies that c0

2 ≥τ2

4 ; hence,
c2 ≥
  1

4 −1

8

τ2 = τ2

8 .

Proof of part (iii). We have shown that c′
0 ≥
p
4β ; hence, c3 ≥pβ

8 .

Proof of part (iv). According to (27), τ2 ≥c0; hence, we deduce that

−2c0ℓ2 ≥−2τ2ℓ2 ≥−1

72ℓ,
(36)

where the last inequality follows from τ2 ≤
1
144ℓ. Furthermore, we note that

−96βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 = −1728ατ 3
2 ℓ3µ ≥−α

123 ℓ,
(37)

with the last inequality following also from µ

ℓ≤1 and τ2 ≤
1
144ℓ. We also note that

−2ℓ

ν −p + ℓ

2
= −
ℓτ1

6 + 3

2


ℓ≥−14

9 ℓ.

Finally, since c′
0 ≤
p
3β according to (28), we get

−c′
0β2 ≥−β2 p

3β = −2

3αµτ2ℓ≥−α

216ℓ,
(38)

where we used τ2 ≤
1
144ℓ. Thus, since α ∈(0, 1), it follows from (36), (37), (38) and (32) that

c7 ≥−ℓτ 2
1

 1

72 + α

123 + 14

9 + α

216 + 17

96


≥−2ℓτ 2
1 .

Proof of part (v). For c8, we may simply observe that

c8 = −

48βp
p + ℓ

p −ℓ

2
+ ℓ

2 + LΨ + 3ℓ(1 + ℓτ2 + 2LΨτ2)

τ 2
2

= −

864αµτ2 + 1

2 + 4 + 3 (1 + ℓτ2 + 2Lψτ2)

ℓτ 2
2

≥−


6αµ

ℓ+ 1

2 + 4 + 3

1 +
1
144 +
8
144


ℓτ 2
2 ≥−8ℓτ 2
2 ,

where we used µ/ℓ≤1, LΨ = 4ℓ, τ2 ≤
1
144ℓ, and α <
1
20.

E
Proof of Theorem 9

Theorem 9. Let {Ft}t∈N be a filtration on (Ω, F, P). Let At, Bt, Ct, Dt be four stochastic processes
adapted to the filtration such that there exist σC, σD > 0 and τ1 > 0 such that for all t ∈N: (i)
Bt ≥0, (ii) E[eλCt+1 | Ft] ≤eλ2σ2
CBt for all λ > 0, (iii) E[eλDt+1 | Ft] ≤eλσ2
D for all

λ ∈
h
0,
1
σ2
D

i
and (iv) At+1−At

τ1
≤−Bt + Ct+1 + Dt+1. Then, for any ¯q ∈(0, 1], we have

P

 

τ1

2

T −1
X

t=0
Bt ≤(A0 −AT ) + τ1σ2
DT + 2τ1 max{2σ2
C, σ2
D} log
1

¯q

!

≥1 −¯q.

24


---Page Break---
Proof. For some fixed γ > 0, let

ST ≜γ

T −1
X

t=0
Bt −(A0 −AT )

with the convention that S0 ≜0. For any T ∈N, we have

ST +1 = γ

T
X

t=0
Bt −(A0 −AT +1)

= γ

T −1
X

t=0
Bt −(A0 −AT ) + γBT −(AT −AT +1)

= ST + γBT −(AT −AT +1)
= ST + (γ −τ1)BT + τ1BT −(AT −AT +1)
≤ST + (γ −τ1)BT + τ1CT +1 + τ1DT +1,

where the inequality follows from condition (iv) of the hypothesis. Hence, for any 0 < λ ≤
1
2τ1σ2
D
and any T ∈N:

E[eλST +1 | FT ] ≤E[eλST eλ(γ−τ1)BT eλτ1CT +1eλτ1DT +1 | FT ]

≤eλST eλ(γ−τ1)BT E[e2λτ1CT +1 | FT ]
1
2 E[e2λτ1DT +1 | FT ]
1
2

≤eλST eλ(γ−τ1)BT 
e4λ2τ1

2σ2
CBT  1

2 
e2λτ1σ2
D
 1

2

= eλST eλ(γ−τ1+2τ1

2λσ2
C)BT eλτ1σ2
D,

where the first inequality follows from Cauchy-Schwarz and in the second one we use (ii) and (iii) of
the hypothesis. Fixing γ = τ1/2 yields for all 0 < λ ≤
1
4τ1σ2
C :

γ −τ1 + 2τ1

2λσ2
C = τ1


−1

2 + 2λσ2
Cτ1


≤0.

Therefore, for 0 < λ ≤min
n
1
4τ1σ2
C ,
1
2τ1σ2
D

o
, using BT ≥0 by (i) of the hypothesis, we get

E[eλST +1 | FT ] ≤eλST eλτ1σ2
D,

and rolling this recursion backwards and noting S0 = 0 yields:

E[eλST ] ≤eλτ1σ2
DT ;

thus, using a Chernoff bound, we get

P(ST > t) ≤E[eλST ]e−λt ≤eλ(τ1σ2
DT −t).

Since for ¯q ∈(0, 1],

eλ(τ1σ2
DT −t) ≤¯q ⇐⇒t ≥τ1σ2
DT −1

λ log(¯q),

we have

P

 

τ1

2

T −1
X

t=0
Bt ≤(A0 −AT ) + τ1σ2
DT −1

λ log(¯q)

!

≥1 −¯q.

The claim follows by taking λ =
1
2τ1 min
n
1
2σ2
C ,
1
σ2
D

o
.

25


---Page Break---
F
Proof of Theorem 11

Theorem 11. In the premise of Corollary 8, sm-AGDA iterates (xt, yt) for τ1 ≤
1
3ℓsatisfy

P

 
1
T

T −1
X

t=0


∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2
≤Q¯q,T ,

!

≥1 −¯q,
∀T ∈N,
∀¯q ∈(0, 1],

for some Q¯q,T = O

κ(∆0+b0)

τ1T
+ κ(δ2
x + δ2
y)


τ1ℓ+ 1

T log

1

¯q

explicitly stated in Appendix F,

where ∆0 ≜Φ(z0) −Φ∗, b0 ≜2 supx,y{ ˆf(x0, y; z0) −ˆf(x, y0; z0)}.

As a first step, we provide a helper lemma that shows that our concentration result (Theorem 9) is
applicable to the sm-AGDA-related processes ˜At, ˜Bt, ˜Ct, ˜Dt introduced in Corollary 8.

Lemma 16. Let At = ˜At, Bt = ˜Bt, Ct = ˜Ct, Dt = ˜Dt, where ˜At, ˜Bt, ˜Ct, ˜Dt are defined
in Corollary 8; moreover, let τ1 > 0 be the primal stepsize in sm-AGDA. Then, the processes
At, Bt, Ct, Dt are adapted to the filtration Ft ≜Fy
t , where Fy
t is defined in (6), and they satisfy the
conditions of Theorem 9 with the following constants:

(i) σ2
C = τ1(240δ2
x + 32δ2
y),
(ii)
σ2
D = 16ℓτ 2
1 δ2
x + 64ℓτ 2
2 δ2
y.

Proof. The fact that At, Bt, Ct, Dt are measurable with respect to Ft = Fy
t for any t ∈N follows
directly from the definition of Ft. Note that xt and yt are also Ft-measurable for all t ∈N. We prove
part (i) and part (ii) separately.

Proof of part (i). First, recall that Ct+1 = ˜Ct+1 = −c4⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩−⟨c5∇yf(xt, yt) +
c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩. We would like to show that E[eλCt+1 | Ft] ≤eλ2σ2
CBt for all λ > 0.
From Assumption 3, we note that for any λ ≥0:

E

"

exp

 

λ⟨−c4∇x ˆf(xt, yt; zt), ∆x
t ⟩

!

| Ft

#

≤exp

 

8λ2c2
4∥∇x ˆf(xt, yt; zt)∥2δ2
x

!

,

where we used [35, Lemma 3]. Now, given the value of c4 in Theorem 7 and the convexity of t 7→t2,
we have

c2
4 ≤5


192βp

p+ℓ
p−ℓ
2
ℓ2τ 2
2 τ 2
1

2
+ 5τ 2
1

(p + ℓ)τ1 −1
2
+ 5
  4ℓ

ν τ 2
1
2 + 5
 
4c0ℓ2τ 2
1
2 + 5
 
2c′
0β2τ 2
1
2.

Now, leveraging the stepsize policy specified in Corollary 8, since α ∈(0, 1), using (37) we get

5


192βp
p + ℓ

p −ℓ

2
ℓ2τ 2
2 τ 2
1

2
≤20 ·
 α

123 ℓτ 2
1
2
≤20 ·

α
3 · 123

2
τ 2
1 ≤τ 2
1
4 .

Similarly, as τ1 ≤
1
3ℓ, have |(p + ℓ)τ1 −1)| ∈[0, 1], which implies

5τ 2
1

(p + ℓ)τ1 −1
2
≤5τ 2
1 .

Since ν = 12

τ1ℓ, we have 5
  4ℓ

ν τ 2
1
2 ≤5

9ℓ4τ 6
1 ≤τ 2
1
4 . Furthermore, using (27), we have 5
 
4c0ℓ2τ 2
1
2 ≤

80τ 4
1 τ 2
2 ℓ4 ≤τ 2
1
4 . Finally, (38) and α ∈(0, 1) imply that

5
 
2c′
0β2τ 2
1
2 ≤20
 α

216ℓτ1
2
τ 2
1 ≤τ 2
1
4 .

Thus, c2
4 ≤6τ 2
1 , which implies that

E

"

exp

 

−λ⟨c4∇x ˆf(xt, yt; zt), ∆x
t ⟩

!

| Ft

#

≤exp

 

48λ2τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2δ2
x

!

.
(39)

Moreover, noting ∆y
t is revealed after ∆x
t , using [35, Lemma 3] again along with the inequality
∥u + v∥≤2∥u∥2 + 2∥v∥2, we have:

E

"

exp

 

−λ⟨c5∇yf(xt, yt) + c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩| Ft, ∆x
t

!

(40)

26


---Page Break---
= E

"

exp

 

λτ2⟨(1 + ℓτ2 + 2LΨτ2) ∇yf(xt, yt) −2∇yf(x∗(yt, zt), yt), ∆y
t ⟩| Ft, ∆x
t

!#

≤exp

8λ2τ 2
2 ∥(1 + ℓτ2 + 2LΨτ2) ∇yf(xt, yt) −2∇yf(x∗(yt, zt), yt)∥2 δ2
y


≤exp

64λ2τ 2
2 ∥∇yf(x∗(yt, zt), yt)∥2 δ2
y + 16λ2τ 2
2 (1 + ℓτ2 + 2LΨτ2)2 ∥∇yf(xt, yt)∥2 δ2
y


≤exp

64λ2τ 2
2

∥∇yf(x∗(yt, zt), yt)∥2 + ∥∇yf(xt, yt)∥2

δ2
y


,
(41)

where the last line follows from 0 < 1 + ℓτ2 + 2LΨτ2 ≤1 +
1
144 +
8
144 ≤2.

Thus, since ∆y
t is revealed after ∆x
t , using (39) and (41) together with the tower property of the
conditional expectations, we get

E

"

exp

 

−λ

c4⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ⟨c5∇yf(xt, yt) + c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩
 #

≤exp

 

48λ2τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2δ2
x + 64λ2τ 2
2
 
∥∇yf(x∗(yt, zt), yt)∥2 + ∥∇yf(xt, yt)∥2
δ2
y

!

.

(42)
Finally, observe that
∥∇yf(xt, yt)∥−∥∇yf(x∗(yt, zt), yt)∥≤∥∇yf(xt, yt) −∇yf(x∗(yt, zt), yt)∥
≤ℓ∥xt −x∗(yt, zt)∥

≤
ℓ
p −ℓ∥∇x ˆf(xt, yt; zt)∥,

where in the last inequality we used the (p −ℓ)-strong convexity of ˆf(·, yt; zt) and the fact that we
have ∇ˆfx(x∗(yt, zt), yt; zt) = 0. Therefore, since p = 2ℓ, we obtain

∥∇yf(xt, yt)∥2 ≤2∥∇yf(x∗(yt, zt), yt)∥2 + 2∥∇x ˆf(xt, yt; zt)∥2.
(43)
Plugging this inequality into (42) yields

E

"

exp

 

−λ

c4⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ ⟨c5∇yf(xt, yt) + c6∇yf(x∗(yt, zt), yt), ∆y
t ⟩
 #

≤exp

 

λ2  
48τ 2
1 δ2
x + 128τ 2
2 δ2
y

∥∇x ˆf(xt, yt; zt)∥2 + 192λ2τ 2
2 δ2
y∥∇yf(x∗(yt, zt), yt)∥2
!

≤exp

 

λ2max
n
5
τ1
 
48τ 2
1 δ2
x + 128τ 2
2 δ2
y

, 1536λ2τ2δ2
y
o

Bt

!

≤exp

 

λ2τ1
 
240δ2
x + 32δ2
y


Bt

!

,

where the last inequality follows from τ2 = τ1

48.

Proof of part (ii). Recall that Dt+1 ≜˜Dt+1 = 2ℓτ 2
1 ∥∆x
t ∥2 + 8ℓτ 2
2 ∥∆y
t ∥2. We would like to show
that E[eλDt+1 | Ft] ≤eλσ2
D for all λ ∈
h
0,
1
σ2
D

i
for some σD > 0. First, observe that for any λ > 0

such that λ ≤min{
1
16ℓτ 2
1 δ2x ,
1
64ℓτ 2
2 δ2y }, we have

E[eλDt+1 | Ft] = E
h
e2λℓτ 2
1 ∥∆x
t ∥2+8λℓτ 2
2 ∥∆y
t ∥2i

≤E
h
e4λℓτ 2
1 ∥∆x
t ∥2i 1

2 E
h
e16λℓτ 2
2 ∥∆y
t ∥2i 1

2

≤

e32λℓτ 2
1 δ2
x

 1

2 
e128λℓτ 2
2 δ2
y

 1

2

≤eλ(16ℓτ 2
1 δ2
x+64ℓτ 2
2 δ2
y),
where we used [35, Lemma 2] in the second inequality.

27


---Page Break---
Before completing the proof of Theorem 11, we provide another lemma that gives a lower bound on
Bt in terms of the squared norm of the partial gradients, based on our choice of sm-AGDA parameters.
Lemma 17. For any t ∈N, we have

∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2 ≤max
20κ

τ1
, 16κ

τ2
, 32ℓ

β



Bt.
(44)

Proof. Since ∇x ˆf(xt, yt; zt) = ∇xf(xt, yt) + p(xt −zt), using ∥a + b∥2 ≤2∥a∥2 + 2∥b∥2, we get

∥∇xf(xt, yt)∥2 ≤2∥∇x ˆf(xt, yt; zt)∥2 + 2p2∥xt −zt∥2.

Hence, together with (43) and the definition of Bt, we obtain

∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2

≤(2 + 2κ)∥∇x ˆf(xt, yt; zt)∥2 + 2κ∥∇yf(x∗(yt, zt), yt)∥2 + 2p2∥xt −zt∥2

≤max
(2 + 2κ) · 5

τ1
, 2κ · 8

τ2
, 2p2 · 8

βp


Bt

≤max
20κ

τ1
, 16κ

τ2
, 32ℓ

β


Bt.

with the last inequality following from κ ≥1.

We may now provide our proof for Theorem 11.

Proof of Theorem 11. According to Lemma 16, the processes At, Bt, Ct, and Dt defined in the
statement of Lemma 16 satisfy the conditions of Theorem 9 if we set τ1 as the primal stepsize
of sm-AGDA. Therefore, for any ¯q ∈[0, 1),

P

 
τ1

2

T −1
X

t=0

Bt ≤τ1(V0 −VT ) + τ1σ2
DT + 2τ1 max

2σ2
C, σ2
D
	
log
1

¯q

 !

≥1 −¯q.

Thus, dividing by τ1

2 T and using Lemma 17, we can conclude that with the probability at least 1 −¯q,
the following event holds:

1
T

T −1
X

t=0

∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2

≤2 max
20κ

τ1 , 16κ

τ2 , 32ℓ

β

  1

T (V0 −VT ) + σ2
D + 1

T max

4σ2
C, 2σ2
D
	
log
1

¯q


.

(45)

Finally, we can relate the potential gap V0 −VT to the primal suboptimality, i.e., ∆0, and to the
duality gap, i.e., b0/2 = supx,y{ ˆf(x0, y; z0) −ˆf(x, y0; z0), at the initialization, following the same
arguments provided in [66]. More precisely, for V (x, y, z) ≜ˆf(x, y; z) −2Ψ(y, z) + 2P(z), we
first observe that

V0 −VT ≤V0 −min
x,y,z V (x, y, z) = V0 −min
x,y,z
ˆf(x, y; z) −2Ψ(y, z) + 2P(z)

≤V0 −min
z
P(z).
(46)

where last line follows from ˆf(x, y; z) −Ψ(y, z) ≥0 for all y, z and P(z) −Ψ(y, z) ≥0 for all y, z.
Since p = 2ℓ, we also note that

P(z0) = min
x max
y
f(x, y) + ℓ∥x −z0∥2 ≤max
y
f(z0, y) = Φ(z0).

Finally, since P is the Moreau envelope of Φ, we have minz P(z) = minz Φ(z); therefore,

V0 −min
z
P(z) = ˆf(x0, y0; z0) −2Ψ(y0, z0) + 2P(z0) −min
z
Φ(z)

≤Φ(z0) −min
z
Φ(z) + ˆf(x0, y0; z0) −Ψ(y0, z0) + P(z0) −Ψ(y0, z0)

≤Φ(z0) −min
z
Φ(z) + 1

2b0 + 1

2b0 = ∆0 + b0.

(47)

28


---Page Break---
Note that τ2 = τ1

48 implies 16κ/τ2 ≥20κ/τ1, and 32ℓ/β = 32

α · κ/τ2 > 16κ/τ2 since α ∈(0, 1).
Therefore,

2 max
20κ

τ1
, 16κ

τ2
, 32ℓ

β


= 64

α · κ/τ2.
(48)

Therefore, combining (45), (46), (47) and (48), we conclude that Qq,T has the following explicit
form:

Q¯q,T = r1

n∆0 + b0

T
+ r2 + r3

T log
1

¯q

o
,
(49)

where the constants r1, r2 and r3 are defined as

r1 = 64

α
κ
τ2 , r2 = σ2
D = 16ℓτ1


τ1δ2
x + 1

12τ2δ2
y

, r3 = max{4σ2
C, 2σ2
D} = 4σ2
C = 4τ1(240δ2
x + 32δ2
y),

where the equalities follow from the expressions of σ2
C and σ2
D provided in Lemma 16.

G
Proof of Corollary 14

Setting τ2
= τ1/48 for any τ1
> 0 implies that Q¯q,T defined in (49) satisfies Q¯q,T
=

O

κ(∆0+b)

τ1T
+ τ1ℓκδ2 + δ2κ

T log

1

¯q

. Hence, setting τ1 = min

1
3ℓ, 48√∆0+b0
√

T ℓδ2

ensures that

Qq,T = O

 
(∆0 + b0)ℓκ

T
+

r

(∆0 + b0)ℓ

T
δκ + δ2κ

T
log
1

¯q

!

.

Finally, to obtain an O(ε, ε/√κ)-stationary point, it suffices to have the above bound smaller than
O(ε2), which is guaranteed when the following three conditions are met up to a constant factor: (i)

(∆0+b0)ℓκ

T
≤ε2

3 , (ii)
q

(∆0+b0)ℓ

T
δκ ≤ε2

3 , and (iii) δ2κ

T log

1

¯q

≤ε2

3 . This is directly implied by the
value Tε,¯q given in our corollary statement.

H
Supplementary Lemmas

In this section, we present a sequence of supplementary lemmas essential for deriving our main result,
Theorem (11). Some of these results are well-known and are directly referenced. For others, we have
improved specific algebraic constants. Additionally, some lemmas are extensions of existing bounds
provided in expectation.

Lemma 18. For any z ∈Rd1 and y1, y2 ∈Rd2, ∥x∗(y1, z) −x∗(y2, z)∥≤


p+ℓ
p−ℓ

∥y1 −y2∥.

Proof. This result is provided in [66, Lemma C.1], which immediately follows from [41, Lemma
B.2, part (c)].

Lemma 19. For any z ∈Rd1, the map y 7→Ψ(y, z) = minx ˆf(x, y, z) is ℓ

1 + p+ℓ

p−ℓ


-smooth.

Proof. This result immediately follows from [41, Lemma B.2, part (d)]. Indeed, for any y1, y2 ∈Rd2,
we have

∥∇yΨ(y1, z) −∇yΨ(y2, z)∥

= ∥∇yf(x∗(y1, z), y1) −∇yf(x∗(y2, z), y2)∥

≤ℓ∥x∗(y1, z) −x∗(y2, z)∥+ ℓ∥y1 −y2∥≤

1 + p + ℓ

p −ℓ


ℓ∥y1 −y2∥,

where the first equality follows from Danskin’s theorem, the first inequality follos from ℓ-smoothness
of f, and the last inequality follows from Lemma 18.

Lemma 20. Consider the iterates (xt, yt, zt) of the sm-AGDA algorithm. For any t ∈N,

∥xt+1 −x∗(yt, zt)∥2
≤
2

1 +
1
τ 2
1 (p −ℓ)2


τ 2
1 ∥∇x ˆf(xt, yt; zt)∥2

+4τ 2
1 ⟨∇x ˆf(xt, yt; zt), ∆x
t ⟩+ 2τ 2
1 ∥∆x
t ∥2.

29


---Page Break---
Proof. For any t ∈N, using the fact that ˆf is strongly convex in x with modulus (p −ℓ) together
with x∗(yt, zt) = argminx ˆf(x, yt; zt), and xt+1 = xt −τ1 ˆGx(xt, yt, ξx
t+1; zt), we get

∥xt+1 −x∗(yt, zt)∥2 ≤2∥xt −x∗(yt, zt)∥2 + 2∥xt+1 −xt∥2

≤
2
(p −ℓ)2 ∥∇x ˆf(xt, yt; zt)∥2 + 2τ 2
1 ∥ˆGx(xt, yt, ξx
t+1; zt)∥2.

Using the identity ˆGx(xt, yt, ξx
t+1; zt) = ∆x
t + ∇x ˆf(xt, yt; zt) within the above inequality yields
the desired result.

Lemma 21. For any y ∈Rd2, z1, z2 ∈Rd1, we have:

∥x∗(y, z1) −x∗(y, z2)∥≤
p
p −ℓ∥z1 −z2∥.

Proof. This result follows from [70, Lemma B.2]. Indeed, since ˆf is strongly convex in x with
modulus p −ℓ, we get

ˆf(x∗(y, z1), y; z2) −ˆf(x∗(y, z2), y; z2) ≥p −ℓ

2
∥x∗(y, z1) −x∗(y, z2)∥2.
(50)

Then swapping z1, z2 leads to

ˆf(x∗(y, z2), y; z1) −ˆf(x∗(y, z1), y; z1) ≥p −ℓ

2
∥x∗(y, z1) −x∗(y, z2)∥2.
(51)

Furthermore, from the definition ˆf, it follows that

ˆf(x∗(y, z2), y; z1) −ˆf(x∗(y, z2), y; z2) = p

2
 
∥x∗(y, z2) −z1∥2 −∥x∗(y, z2) −z2∥2

= p

2
 
∥z1∥2 −∥z2∥2 + 2⟨x∗(y, z2), z2 −z1⟩

;

similarly, swapping z1, z2 in the above inequality leads to
ˆf(x∗(y, z1), y; z2) −ˆf(x∗(y, z1), y; z1) = p

2
 
∥x∗(y, z1) −z2∥2 −∥x∗(y, z1) −z1∥2

= p

2
 
∥z2∥2 −∥z1∥2 −2⟨x∗(y, z1), z2 −z1⟩

.

Thus, summing the above two identities and applying Cauchy-Schwartz gives us
ˆf(x∗(y, z2), y; z1) −ˆf(x∗(y, z2), y; z2) + ˆf(x∗(y, z1), y; z2) −ˆf(x∗(y, z1), y; z1)
≤p∥x∗(y, z1) −x∗(y, z2)∥∥z1 −z2∥.
We can lower bound the left hand side of the above inequality using (50) and (51), which leads to

(p −ℓ)∥x∗(y, z1) −x∗(y, z2)∥2 ≤p∥x∗(y, z1) −x∗(y, z2)∥∥z1 −z2∥.
(52)
Rearranging this inequality yields the desired result.

Lemma 22. For any x ∈Rd1, x 7→Φ(x; z) is (p −ℓ)-strongly convex.

Proof. For any y ∈Rd2, z ∈Rd1, x 7→ˆf(x, y; z) is strongly convex with modulus p −ℓ, i.e.,
x 7→ˆf(x, y; z) −p−ℓ

2 ∥x∥2 is convex. Then, x 7→supy∈Rd2 ˆf(x, y; z) −p−ℓ

2 ∥x∥2 is convex as it is
the pointwise supremum of convex functions. Therefore, x 7→Φ(x; z) −p−ℓ

2 ∥x∥2 is convex, and
this implies x 7→Φ(x, z) is strongly convex with modulus p −ℓ.

Lemma 23. For all z1, z2 ∈Rd1, we have

∥x∗(z1) −x∗(z2)∥≤
p
p −ℓ∥z1 −z2∥.

Proof. Using the result of Lemma 22, one can show this result following exactly the same arguments
in the proof of Lemma 21.

Lemma 24. For any y ∈Rd2, z ∈Rd1, it holds that

∥x∗(z) −x∗(y+(z), z)∥2 ≤
1
(p −ℓ)µ


1 + τ2ℓ2p

p −ℓ

2
∥∇yf(x∗(y, z), y)∥2.

Proof. The proof is the same as [66, Lemma C.2].

30


---Page Break---
I
Further Details about Figure 2

In this section, we provide further details about how the empirical and theoretical quantiles are
estimated in Figure 2 and are compared.

Generation of the theoretical quantiles Qq,T .
For any given q ∈(0, 1), estimating an upper
bound on Qq,T requires estimating an upper bound on the quantity ∆0 + b0 based on Theorem 11.
Other constants such as r1, r2, r3, ℓ, and µ are explicitly known in the setting of this experiment
based on the NCPL game where T = 10,000 is fixed.

First, we set the initial point (x0, y0) randomly, where each component of x0 and y0 is sampled
uniformly from the interval [−20, 20], and we set z0 = x0. We then estimate an upper bound on the
quantity ∆0 +b0 numerically based on a grid search, resulting in ∆0 +b0 = 12. Second, we generate
a linear mesh Im with a grid size m = 0.0002 over the interval [0, 1]. For q ∈Im, we calculate Qq,T
based on Theorem 11. Third, we generate a sequence of quantiles QIm,T . These quantiles are used to
create a CDF via linear interpolation using the scipy.interpolate package’s interp1d function
in Python. Note that this quantile sequence generates a CDF over the values QIm,T .

Generation of the empirical quantiles of the random variable XT .
We generate 1,000 samples
{X(i)
T }1000
i=1 from the sample paths corresponding to the NCPL game with T = 10,000, where
XT =
1
T
PT
t=1 ∥∇xf(xt, yt)∥2 + κ∥∇yf(xt, yt)∥2 represents the path averages of the gradients.
Quantiles for this sequence were generated over Im using NumPy’s quantile generator in Python,
ensuring alignment with the mesh over which the theoretical quantiles were generated. Evidently, our
theoretical quantiles dominate the empirical quantiles pointwise, demonstrating that in the challenging
NCPL regime, our theory provides empirically verifiable guarantees on the tail behavior of the random
variable XT .

Comparison of quantiles.
We plotted the CDF corresponding to the theoretical quantiles Qq,T
over the values of the empirical quantiles using a common mesh grid over the range of the empirical
averages of the sample paths. In other words, we scaled the quantiles Qq,T with an affine transforma-
tion so that their range matches the range of the empirical quantiles. This affine scaling preserves the
shape of the distribution corresponding to the theoretical quantiles and allows for better visualization.

31


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes] The abstract and the introduction reflect the paper’s contributions and scope
accurately. In fact, we provide adequate references to the results from our paper for the
claims in the introduction so that the paper’s contributions and scope are easier to understand.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] Although we provide the first high-probability bounds for nonconvex/PL
minimax problems to the best of our knowledge, there may still be some room to improve
the condition number κ dependency based on the existing lower complexity bounds [38, 71]
in expectation, unless the lower complexity bounds are loose. Also, we acknowledged
that for the distributionally robust optimization experiment our assumptions do not all hold
(which is a limitation); but that our results are still predictive of the practical performance.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes] All assumptions are clearly stated or referenced in the statement of any
theorem, we provide a sketch of the proof of our main theorem.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] We uploaded our code as a supplementary material, and explained in detail
how our experiments are performed. All the datasets we use are well-known benchmark
datasets and we provided the adequate references to them.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes] We explain the experimental setup in detail and provide the code, the datasets
are well-known benchmark sets that are publicly available and we provided the adequate
references.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes] Our numerical experiments section has all the details.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes] In the experiments, we compare the histogram of the solutions found by
multiple algorithms. This has more resolution than standard approaches that has the average
performance (as an average over the algorithm runs) and the standard deviation over the
runs; because one can visualize the behavior (histogram) of all runs.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

32


---Page Break---
Answer: [Yes] We explained the details of the operating system/computing resources.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] We preserve anonymity and we absolutely confirm with the code of Ethics.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA] Our paper is mainly a theoretical paper with convergence guarantees for
stochastic mini-max algorithms, so we are not aware of any direct impacts to the society.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA] Our paper has theoretical nature an we are not aware of any potential misuse
risk for our results.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [NA] We use public datasets, and implemented the algorithms on our own in
Python; no licensed material is used. We use public well-known benchmark datasets which
are properly referenced.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA] We do not release new assets.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA] The paper does not involve crowdsourcing nor research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA] No human subjects or crowdsourcing were involved.

33


---Page Break---
