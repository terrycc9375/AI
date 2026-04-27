Parameter-free Clipped Gradient Descent Meets
Polyak

Yuki Takezawa1,2, Han Bao1,2, Ryoma Sato3, Kenta Niwa4, Makoto Yamada2

1Kyoto University, 2OIST, 3NII, 4NTT Communication Science Laboratories

Abstract

Gradient descent and its variants are de facto standard algorithms for training
machine learning models. As gradient descent is sensitive to its hyperparame-
ters, we need to tune the hyperparameters carefully using a grid search. However,
the method is time-consuming, particularly when multiple hyperparameters exist.
Therefore, recent studies have analyzed parameter-free methods that adjust the
hyperparameters on the fly. However, the existing work is limited to investigations
of parameter-free methods for the stepsize, and parameter-free methods for other
hyperparameters have not been explored. For instance, although the gradient clip-
ping threshold is a crucial hyperparameter in addition to the stepsize for preventing
gradient explosion issues, none of the existing studies have investigated parameter-
free methods for clipped gradient descent. Therefore, in this study, we investigate
the parameter-free methods for clipped gradient descent. Specifically, we propose
Inexact Polyak Stepsize, which converges to the optimal solution without any hy-
perparameters tuning, and its convergence rate is asymptotically independent of L
under L-smooth and (L0, L1)-smooth assumptions of the loss function, similar to
that of clipped gradient descent with well-tuned hyperparameters. We numerically
validated our convergence results using a synthetic function and demonstrated the
effectiveness of our proposed methods using LSTM, Nano-GPT, and T5.

1
Introduction

We consider the convex optimization problem:
min
x∈Rd f(x),
(1)

where the loss function f is convex and lower bounded. In this setting, gradient descent and its
variants (Duchi et al., 2011; Kingma and Ba, 2015) are the de facto standard algorithms to minimize
the loss function. The performance of the algorithm is highly sensitive to the hyperparameter settings,
necessitating the careful tuning of the hyperparameters to achieve best performance. More specifically,
when the loss function is L-smooth, gradient descent can achieve the optimal convergence rate
O( L∥x0−x⋆∥2

T
) when we set the stepsize to 1

L where x0 is the initial parameter and x⋆is the optimal
solution (Nesterov, 2018). Unfortunately, parameter L is problem-specific and unavailable in practice.
Thus, gradient descent must be executed in many times with different hyperparameters to identify
the good hyperparameter settings, which is a very time-consuming process. Notably, when multiple
hyperparameters are under consideration, this hyperparameter search becomes computationally more
demanding.

Several recent studies have examined parameter-free methods for tuning hyperparameters on the fly
(Berrada et al., 2020; Defazio and Mishchenko, 2023; Orvieto et al., 2022; Jiang and Stich, 2023;
Ivgi et al., 2023; Khaled et al., 2023; Orabona and Tommasi, 2017; Carmon and Hinder, 2022).1

1We use parameter-free methods to describe algorithms that provably converge to the optimal solution
without any problem-specific parameters for setting their stepsizes and other hyperparameters.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
These methods automatically adjust the stepsize during the training and are guaranteed to converge to
the optimal solution without tuning the stepsize. In other words, the stepsize did not require tuning
using the grid search. However, the existing parameter-free methods only focus on the stepsize, and
parameter-free methods for other hyperparameters have not been explored. For example, in addition
to the stepsize, the gradient clipping threshold is an important hyperparameter for training language
models (Pascanu et al., 2013; Zhang et al., 2020a,b,c).

Clipped gradient descent can achieve the convergence rate O( L0∥x0−x⋆∥2

T
) under the assumption
that the loss function is (L0, L1)-smooth when we use the optimal stepsize and gradient clipping
threshold (Koloskova et al., 2023). In many cases, L0 is significantly smaller than L (Zhang et al.,
2020b). Thus, by comparing with the convergence rate of gradient descent O( L∥x0−x⋆∥2

T
), gradient
clipping often allows gradient descent to converge faster. However, we must carefully tune two
hyperparameters, stepsize and gradient clipping threshold, to achieve this convergence rate. If the
gradient clipping threshold is too large, the gradient clipping fails to accelerate the convergence.
Moreover, if the gradient clipping threshold is too small, gradient clipping deteriorates rather than
accelerating the convergence rate. Can we develop a parameter-free method whose convergence rate
is asymptotically independent of L under (L0, L1)-smoothness?

In this study, we investigate a parameter-free method for clipped gradient descent. First, we provide
the better convergence rate of Polyak stepsize (Polyak, 1987) under (L0, L1)-smoothness. We
discover that the convergence rate of Polyak stepsize matches that of clipped gradient descent with
well-tuned stepsize and gradient clipping threshold. Although the convergence rate of Polyak stepsize
is asymptotically independent of L under (L0, L1)-smooth assumption as clipped gradient descent,
it still requires the minimum loss value, which is a problem-specific value. Thus, we make Polyak
stepsize parameter-free without losing this property under (L0, L1)-smoothness by proposing Inexact
Polyak Stepsize, which converges to the optimal solution without any problem-specific parameters.
We numerically evaluated Inexact Polyak Stepsize using a synthetic function and neural networks,
validating our theory and demonstrating the effectiveness of Inexact Polyak Stepsize.

2
Preliminary

2.1
Gradient descent & L-smoothness

One of the most fundamental algorithms for solving Eq. (1) represents the gradient descent:
xt+1 = xt −ηt∇f(xt),

where x0 ∈Rd is the initial parameter and ηt > 0 is the stepsize at t-th iteration. To ensure that
gradient descent converges to the optimal solution quickly, we must carefully tune the stepsize ηt.
When the stepsize is too large, the training collapses. By contrast, when the stepsize is too small,
the convergence rate becomes too slow. Thus, we must search for a proper stepsize as the following
theorem indicates.
Assumption 1 (L-smoothness). There exists a constant L > 0 that satisfies the following for all
x, y ∈Rd:
∥∇f(x) −∇f(y)∥≤L∥x −y∥.
(2)
Theorem 1 (Nesterov (2018, Corollary 2.1.2)). Assume that f is convex and L-smooth, and there
exists an optimal solution x⋆:= arg minx∈Rd f(x). Then, gradient descent with stepsize ηt = 1

L
satisfies

f(¯x) −f(x⋆) ≤O
L∥x0 −x⋆∥2

T


,
(3)

where ¯x := 1

T
PT −1
t=0 xt and T is the number of iterations.

2.2
Clipped gradient descent & (L0, L1)-smoothness

Gradient clipping is widely used to stabilize and accelerate the training of gradient descent (Pascanu
et al., 2013; Devlin et al., 2019). Let c > 0 be the threshold for gradient clipping. Clipped gradient
descent is given by:

xt+1 = xt −ηt min

1,
c
∥∇f(xt)∥


∇f(xt).
(4)

2


---Page Break---
Many prior studies investigated the theoretical benefits of gradient clipping (Koloskova et al., 2023;
Zhang et al., 2020a,b,c; Li and Liu, 2022; Sadiev et al., 2023). Zhang et al. (2020b) experimentally
found that the gradient Lipschitz constant decreases during the training of various neural networks
and is highly correlated with gradient norm ∥∇f(x)∥. To describe this phenomenon, Zhang et al.
(2020a) introduced a novel smoothness assumption called (L0, L1)-smoothness. Then, it has been
experimentally demonstrated that the local gradient Lipschitz constant L0 is thousands of times
smaller than the global gradient Lipschitz constant L.

Assumption 2 ((L0, L1)-smoothness). There exists constants L0 > 0 and L1 > 0 that satisfy the
following for all x, y ∈Rd with ∥x −y∥≤
1
L1 :

∥∇f(x) −∇f(y)∥≤(L0 + L1∥∇f(x)∥)∥x −y∥.
(5)

Note that (L0, L1)-smoothness is strictly weaker than L-smoothness because (L0, L1)-smoothness
covers L-smoothness by taking L1 = 0. Using the (L0, L1)-smoothness assumption, the convergence
rate of clipped gradient descent was established as follows.

Theorem 2 (Koloskova et al. (2023, Theorem 2.3)). Assume that f is convex, L-smooth, and
(L0, L1)-smooth, and there exists an optimal solution x⋆:= arg minx∈Rd f(x). Then, clipped
gradient descent with ηt =
1
L0 and c = L0

L1 satisfies:

f(¯x) −f(x⋆) ≤O
L0∥x0 −x⋆∥2

T
+ LL2
1∥x0 −x⋆∥4

T 2


,
(6)

where ¯x := 1

T
PT −1
t=0 xt and T is the number of iterations.

When the number of iterations T is large, the first term O( L0∥x0−x⋆∥2

T
) becomes dominant, and
the convergence rate of clipped gradient descent is asymptotically independent of L. Gradient
clipping allows for the use of a larger stepsize, and thus, gradient descent converges faster because
of L0 ≪L. We can interpret L ≃L0 + L1 supx ∥∇f(x)∥. The stepsize of gradient descent
in Theorem 1 is
1
L0+L1 supx ∥∇f(x)∥, which is typically very small. By comparing with gradient
descent, the coefficient multiplied by the gradient of clipped gradient descent in Theorem 2 is
min{ 1

L0 ,
1
L1∥∇f(xt)∥}, which is larger than
1
L0+L1 supx ∥∇f(x)∥. Specifically, when parameter x is
close to the optimal solution x⋆(i.e., ∥∇f(x)∥is small), clipped gradient descent can use a larger
stepsize and then reach the optimal solution faster than gradient descent.

2.3
Polyak stepsize

When f is convex, xt+1 and xt generated by gradient descent satisfy ∥xt+1 −x⋆∥2 ≤∥xt −x⋆∥2 −
2ηt(f(xt) −f(x⋆)) + η2
t ∥∇f(xt)∥2. By minimizing the right-hand side, we can derive well-known
Polyak stepsize (Polyak, 1987):

ηt = f(xt) −f ⋆

∥∇f(xt)∥2 ,
(7)

where f ⋆:= f(x⋆). When f is L-smooth, gradient descent with Polyak stepsize converges to the
optimal solution as quickly as gradient descent with ηt = 1

L.

Theorem 3 (Hazan and Kakade (2019, Theorem 1)). Assume that f is convex and L-smooth, and
there exists an optimal solution x⋆:= arg minx∈Rd f(x). Then, gradient descent with Polyak
stepsize Eq. (7) satisfies:

f(¯x) −f(x⋆) ≤O
L∥x0 −x⋆∥2

T


,
(8)

where ¯x := 1

T
PT −1
t=0 xt and T is the number of iterations.

In addition to the L-smooth setting, Polyak stepsize is known to cause gradient descent to converge to
the optimal solution with the optimal rate among various settings, e.g., non-smooth convex, smooth
convex, and strongly convex settings (Hazan and Kakade, 2019).

3


---Page Break---
3
Improved convergence result of Polyak stepsize

Before proposing a parameter-free method for clipped gradient descent, in this section, we present
a new convergence analysis of Polyak stepsize under (L0, L1)-smoothness. Surprisingly, our new
analysis reveals that Polyak stepsize achieves exactly the same convergence rate as clipped gradient
descent with appropriate hyperparameters. A bunch of prior studies established the convergence rates
of Polyak stepsize, and it is well-known that Polyak stepsize allows gradient descent to converge
as fast as the optimal stepsize. However, our theorem finds that Polyak stepsize achieves a faster
convergence rate than gradient descent with the optimal stepsize as clipped gradient descent, and
none of the existing studies have found this favorable property of Polyak stepsize.

3.1
Connection between Polyak stepsize and clipped gradient descent

Under (L0, L1)-smoothness, we can obtain the following results.

Proposition 1. Assume that f is convex and (L0, L1)-smooth. Then, Polyak stepsize Eq. (7) satisfies:

min
 1

4L0
,
1
4L1∥∇f(xt)∥


≤f(xt) −f ⋆

∥∇f(xt)∥2 .
(9)

Proof. Assumption 2 and Lemma 2 imply

f(xt) −f ⋆

∥∇f(xt)∥2 ≥
1
2(L0 + L1∥∇f(xt)∥).

When ∥∇f(xt)∥< L0

L1 , Polyak stepsize is bounded from below by
1
4L0 . When ∥∇f(xt)∥≥L0

L1 , we
have
1
2(L0 + L1∥∇f(xt)∥) ≥
1
4L1∥∇f(xt)∥.

Therefore, we can conclude the statement.

Under L-smoothness, the lower bound of Polyak stepsize was obtained as follows.
Proposition 2 (Jiang and Stich (2023, Lemma 15)). Assume that f is convex and L-smooth. Then,
Polyak stepsize Eq. (7) satisfies:
1
2L ≤f(xt) −f ⋆

∥∇f(xt)∥2 .
(10)

By comparing Propositions 1 and 2, Proposition 1 shows that Polyak stepsize does not become
excessively small when the parameter approaches the optimal solution (i.e., ∥∇f(x)∥approaches
zero), similar to clipped gradient descent. If we choose the stepsize and gradient clipping threshold
as in Theorem 2, clipped gradient descent can be written as follows:

xt+1 = xt −min
 1

L0
,
1
L1∥∇f(xt)∥


∇f(xt).
(11)

Thus, Proposition 1 implies that Polyak stepsize can be regarded as internally estimating the hyperpa-
rameters for clipped gradient descent, as shown in Theorem 2.

3.2
Convergence analysis of Polyak stepsize under (L0, L1)-smoothness

Based on the relationship between Polyak stepsize and clipped gradient descent in Sec. 3.1, we
provide a new convergence result for Polyak stepsize under (L0, L1)-smoothness. The proof is
deferred to Sec. A.
Theorem 4. Assume that f is convex, L-smooth, and (L0, L1)-smooth, and there exists an op-
timal solution x⋆:= arg minx∈Rd f(x). Let T be the number of iterations and define τ :=
arg min0≤t≤T −1 f(xt). Then, gradient descent with Polyak stepsize Eq. (7) satisfies:

f(xτ) −f(x⋆) ≤O
L0∥x0 −x⋆∥2

T
+ LL2
1∥x0 −x⋆∥4

T 2


.
(12)

4


---Page Break---
By comparing Theorem 4 with Theorem 2, the convergence rate of Polyak stepsize is the same as
that of clipped gradient descent. Thus, Polyak stepsize can converge faster than the optimal stepsize
given in Theorem 1 when L0 ≪L. Many prior studies analyzed the convergence rate of Polyak
stepsize and discussed the relationship between Polyak stepsize and gradient descent with the optimal
stepsize (Polyak, 1987; Loizou et al., 2021; Galli et al., 2023; Berrada et al., 2020). However, they
only recognized Polyak stepsize as making gradient descent converge with the same convergence
rate as the optimal stepsize, and none of the prior studies have found this relationship between
Polyak stepsize and clipped gradient descent. Our new convergence result is the first to discover
that the Polyak stepsize can achieve the same convergence rate not only as gradient descent with an
appropriate stepsize but also as clipped gradient descent with an appropriate stepsize and gradient
clipping threshold.

4
Making clipped gradient descent parameter-free

In the previous section, we found that the convergence rate of Polyak stepsize is asymptotically
independent of L under (L0, L1)-smoothness as clipped gradient descent with appropriate hyper-
parameters. However, Polyak stepsize requires the minimum loss value f ⋆, which is a problem-
specific parameter. In this section, we propose a method that can remove the prior knowledge
of f ⋆from Polyak stepsize without losing the property of asymptotic independence of L under
(L0, L1)-smoothness.

4.1
Inexact Polyak Stepsize

To make Polyak stepsize parameter-free, several prior studies have proposed the use of lower bound
of f ⋆instead of f ⋆(Loizou et al., 2021; Orvieto et al., 2022; Jiang and Stich, 2023). The loss
functions commonly used in machine learning models are non-negative. Thus, the lower bound of f ⋆
is trivially obtained as zero and is not a problem-specific parameter. By utilizing this lower bound,
a straightforward approach to make Polyak stepsize independent of problem-specific parameters is
replacing f ⋆in Polyak stepsize with the lower bound l⋆as follows:

ηt = f(xt) −l⋆

∥∇f(xt)∥2 .
(13)

However, the stepsize in Eq. (13) becomes excessively large as the parameter approaches the optimal
solution, and it does not lead to the optimal solution (Loizou et al., 2021). This is because ∥∇f(xt)∥
approaches zero, while f(xt) −l⋆approaches f ⋆−l⋆(> 0), which makes the stepsize in Eq. (13)
excessively large as the parameter approaches the optimal solution. To mitigate this issue, DecSPS
(Orvieto et al., 2022) and AdaSPS (Jiang and Stich, 2023), which are parameter-free methods based
on Polyak stepsize that use l⋆instead of f ⋆, make the stepsize monotonically non-increasing to
converge to the optimal solution.

However, making the stepsize monotonically non-increasing loses the fruitful property that the
convergence rate of Polyak stepsize is asymptotically independent of L as clipping gradient descent
under (L0, L1)-smoothness. This is because Polyak stepsize and clipped gradient descent make the
convergence rate asymptotically independent of L by increasing the stepsize when the parameter
approaches the optimal solution. In fact, we evaluated DecSPS and AdaSPS with a synthetic function
in Sec. 6.1, demonstrating that the convergence deteriorates as L increases.

Algorithm 1 Inexact Polyak Stepsize

1: Input: The number of iterations T and lower bound l⋆.
2: f best, xbest ←f(x0), x0.
3: for t = 0, 1, · · · , T −1 do
4:
xt+1 ←xt −
f(xt)−l⋆
√

T ∥∇f(xt)∥2 ∇f(xt).

5:
if f(xt+1) ≤f best then
6:
f best, xbest ←f(xt+1), xt+1.
7: return xbest.

5


---Page Break---
To address this issue, we propose Inexact Polyak Stepsize, whose details are described in Alg. 1. As
discussed above, we cannot make the stepsize decrease to maintain the asymptotic independence of
L under (L0, L1)-smoothness. Thus, we set the stepsize as follows:

ηt =
f(xt) −l⋆
√

T∥∇f(xt)∥2 ,
(14)

where T denotes the number of iterations. Instead of making the stepsize decrease, we propose
returning the parameter for which the lowest loss is achieved as the final parameter.

4.2
Convergence analysis of Inexact Polyak Stepsize

The following theorem provides the convergence rate of Inexact Polyak Stepsize. The proof is
deferred to Sec. B.
Theorem 5. Assume that f is convex, L-smooth, and (L0, L1)-smooth, and there exists an optimal
solution x⋆:= arg minx∈Rd f(x). Let T be the number of iterations and σ2 := f ⋆−l⋆. Then, x
generated by Alg. 1 satisfies:

f(x) −f(x⋆) ≤O
L0∥x0 −x⋆∥2 + σ2

√

T
+ LL2
1∥x0 −x⋆∥4

T
+ L2
1Lσ4

L2
0T


.
(15)

Asymptotic independence of L:
When the number of iterations T is large, only the first term
O( L0∥x0−x⋆∥2+σ2

√

T
) becomes dominant in the convergence rate, which does not depend on L. Thus,
Theorem 5 shows that Inexact Polyak Stepsize successfully inherits the favorable property of Polyak
stepsize under (L0, L1)-smoothness. In addition to Inexact Polyak Stepsieze, DecSPS (Orvieto et al.,
2022) and AdaSPS (Jiang and Stich, 2023) have been proposed as parameter-free methods that use l⋆
instead of f ⋆in Polyak stepsize. However, these prior methods fail to inherit the favorable property
of Polyak stepsize, and their convergence rates deteriorate when L is large because these methods
decrease the stepsize during the training. In fact, we evaluated DecSPS and AdaSPS with a synthetic
function in Sec. 6.1, demonstrating that convergence rates of DecSPS and AdaSPS are degraded when
L becomes large, whereas the convergence rate of Inexact Polyak Stepsize does not depend on L.

Removing dependence on DT :
The convergence rates of DecSPS and AdaSPS depend on DT (:=
max0≤t≤T ∥xt −x⋆∥). Thus, strictly speaking, these convergence rates cannot show that DecSPS
and AdaSPS converge to the optimal solution because DT may increase as the number of iterations T
increases. For instance, if DT increase with Ω(T
1
4 ), the convergence rate of AdaSPS is O(Lσ + L2),
which does not show that AdaSPS converges to the optimal solution. In contrast, the convergence rate
in Eq. (15) depends on only ∥x0 −x⋆∥. Theorem 5 indicates that Inexact Polyak Stepsize converges
to the optimal solution.

Convergence rate with respect to T:
Inexact Polyak Stepsize successfully achieves the asymptotic
independence of L, while it slows down the convergence rate with respect to the number of iterations
T by comparing clipped gradient descent with proper hyperparameters. The convergence rate of
Inexact Polyak Stepsize O( L0
√

T ) is not optimal in terms of T, and there may be room to improve this
rate. For instance, the adaptive methods proposed by Hazan and Kakade (2019) might be used to

Table 1: Summary of convergence rates of parameter-free methods based on Polyak stepsize. All
convergence results are the ones under convex, L-smoothness, and (L0, L1)-smoothness. We define
DT := max0≤t≤T ∥xt −x⋆∥.

Algorithm
Convergence Rate
Assumption

DecSPS (Orvieto et al., 2022)(a)
O

max{L,η−1
0
}D2
T +σ2
√

T


1

AdaSPS (Jiang and Stich, 2023)(a)
O

LD2
T σ
√

T
+ L2D4
T
T

1

Inexact Polyak Stepsize (This work)
O

L0∥x0−x⋆∥2+σ2

√

T
+ LL2
1∥x0−x⋆∥4

T
+ L2
1Lσ4

L2
0T

1, 2(b)

(a) We present the convergence rates of DecSPS and AdaSPS in the deterministic setting to compare DecSPS, AdaSPS, and Inexact
Polyak Stepsize in the same deterministic setting, while Orvieto et al. (2022) and Jiang and Stich (2023) also analyzed the rate rates in the
stochastic setting.

(b) If f is L-smooth, f is (L0, L1)-smooth because (L0, L1)-smoothness assumption is strictly weaker than L-smoothness assumption.

6


---Page Break---
alleviate this issue. However, the parameter-free methods for clipped gradient descent have not been
explored well in the existing studies. We believe that Inexact Polyak Stepsize is the important first
step for developing parameter-free clipped gradient descent.

5
Related work

Gradient clipping:
Gradient clipping was initially proposed to mitigate the gradient explosion
problem for training RNN and LSTM (Mikolov et al., 2010; Merity et al., 2018) and is now widely
used to accelerate and stabilize the training not only for RNN and LSTM, but also for various machine
learning models, especially language models (Devlin et al., 2019; Raffel et al., 2019). Recently, many
studies have investigated the theoretical benefits of gradient clipping and analyzed the convergence
rate of clipped gradient descent under (1) (L0, L1)-smoothness assumption (Koloskova et al., 2023;
Zhang et al., 2020a,b) and (2) heavy-tailed noise assumption (Zhang et al., 2020c; Li and Liu, 2022;
Sadiev et al., 2023). (1) Zhang et al. (2020b) found that the local gradient Lipschitz constant is
correlated with the gradient norm. To describe this phenomenon, Zhang et al. (2020b), Zhang et al.
(2020a), and Koloskova et al. (2023) introduced the new assumption, (L0, L1)-smoothness, providing
the convergence rate of clipped gradient descent under (L0, L1)-smoothness. Then, they showed that
gradient clipping can improve the convergence rate of gradient descent, as we introduced in Sec. 2.2.
(2) Besides (L0, L1)-smoothness, Zhang et al. (2020c) pointed out that the distribution of stochastic
gradient noise is heavy-tailed for language models. Then, it has been shown that gradient clipping
can make the stochastic gradient descent robust against the heavy-tailed noise of stochastic gradient
(Li and Liu, 2022; Sadiev et al., 2023; Zhang et al., 2020c).

Parameter-free methods:
Hyperparameter-tuning is one of the most time-consuming tasks for
training machine learning models. To alleviate this issue, many parameter-free methods that adjust
the stepsize on the fly have been proposed, e.g., Polyak-based stepsize (Berrada et al., 2020; Hazan
and Kakade, 2019; Loizou et al., 2021; Mukherjee et al., 2023; Orvieto et al., 2022; Jiang and Stich,
2023), AdaGrad-based methods (Ivgi et al., 2023; Khaled et al., 2023), and Dual Averaging-based
methods (Orabona and Tommasi, 2017; Defazio and Mishchenko, 2023). However, parameter-free
methods for hyperparameters, except for stepsizes, have not been studied. In this work, we studied
the parameter-free methods for two hyperparameters, the stepsize and gradient clipping threshold,
and then proposed Inexact Polyak Stepsize, which converges to the optimal solution without tuning
any hyperparameters and its convergence rate is asymptotically independent of L as clipped gradient
descent with well-tuned hyperparameters.

6
Numerical evaluation

In this section, we evaluate our theory numerically. In Sec. 6.1, we evaluate Polyak stepsize
and Inexact Polyak Stepsize using a synthetic function, varying that their convergence rates are
asymptotically independent of L. In Sec. 6.2, we show the results obtained using neural networks.

6.1
Synthetic function

Setting:
In this section, we validate our theory for Polyak stepsize and Inexact Polyak Stepsize
using a synthetic function. We set the loss function as f(x) = L0L2
1
72 x4 + L0

4 x2 + f ⋆, which is
(L0, L1)-smooth for any L0 > 0 and L1 > 0 (See Proposition 3 in Appendix). We set L0 to 1, x0 to
5, f ⋆= 1, and l⋆= 0 and then evaluated various methods when varying L1.

Results:
We show the results in Fig. 1. The results indicate that gradient descent converges slowly
when L1 is large, whereas Polyak stepsize and clipped gradient descent does not depend on L1. These
observations are consistent with those discussed in Sec. 3, which shows that the convergence rate of
Polyak stepsize is asymptotically independent of L as in clipped gradient descent. By comparing
DecSPS, AdaSPS, and Inexact Polyak Stepsize, which are parameter-free methods, the convergence
rates of DecSPS and AdaSPS degrade as L1 increases. Thus, DecSPS and AdaSPS lose the favorable
property of asymptotic independence of L under (L0, L1)-smoothness. In contrast, the convergence
behavior of Inexact Polyak Stepsize does not depend on L1, which is consistent with Theorem 5, and
Inexact Polyak Stepsize successfully inherits the Polyak stepsize under (L0, L1)-smoothness.

7


---Page Break---
0
100
200
300
400
500
Iteration

10-20

10-16

10-12

10-8

10-4

100

104

108

f(xT) −f

(a) Gradient Descent

0
100
200
300
400
500
Iteration

10-20

10-16

10-12

10-8

10-4

100

104

108

f(xT) −f

(b) Clipped Gradient Descent

0
100
200
300
400
500
Iteration

10-20

10-16

10-12

10-8

10-4

100

104

108

f(xT) −f

(c) Polyak Stepsize

0
2000
4000
6000
8000 10000
Iteration

10-15

10-12

10-9

10-6

10-3

100

103

106

109

f(xT) −f

(d) DecSPS (Orvieto et al., 2022)

0
2000
4000
6000
8000 10000
Iteration

10-15

10-12

10-9

10-6

10-3

100

103

106

109

f(xT) −f

(e) AdaSPS (Jiang and Stich, 2023)

0
2000
4000
6000
8000 10000
Iteration

10-15

10-12

10-9

10-6

10-3

100

103

106

109

f(xT) −f

L1 = 1
L1 = 10
L1 = 100
L1 = 1000

(f) Inexact Polyak Stepsize

Figure 1: Convergence behaviors of various methods with the synthetic function.

10
2
10
1
100
101
102
Stepsize

4

5

6

7

8

9

10

Test loss

(a) LSTM

10
4
10
3
10
2
10
1
100
Stepsize

2.4

2.6

2.8

3.0

3.2

3.4

Test loss

(b) Nano-GPT

10
1
100
Stepsize

4

5

6

7

8

9

10

11

12

Test loss

SGD
Clipped SGD
DoG
Polyak
AdaSPS
DecSPS
Inexact Polyak Stepsize

(c) T5
Figure 2: The final test loss with various hyperparameter settings. For T5, the results of DecSPS and
AdaSPS were omitted because their final test loss was much larger than the others, as shown in Fig. 4.
Furthermore, the results of SGD were also omitted when the final test loss became nan or infinity.

6.2
Neural networks

Setting:
Next, we evaluated Inexact Polyak Stepsize using LSTM, Nano-GPT2, and T5 (Nawrot,
2023). For LSTM, Nano-GPT, and T5, we used the Penn Treebank, Shakespeare, and C4 as training
datasets, respectively. For SGD and Clipped SGD, we tuned the stepsize and gradient clipping
threshold on validation datasets. For Polyak stepsize, we showed the results when we set f ⋆to zero.
For Inexact Polyak Stepsize, Theorem 4 requires the selection of the best parameters. However, we
do not need to choose this for neural networks because the parameters only reach the stationary
point and do not reach the global minima. See Sec. D for the detailed training configuration. For all
experiments, we repeated with three different seed values and reported the average.

Results:
Figure 4 shows the loss curves, and Fig. 2 shows the final test losses for various hyper-
parameters. The results indicate that Inexact Polyak Stepsize consistently outperform DecSPS and
AdaSPS for all neural network architectures. Although DoG performed the best for LSTM among
the parameter-free methods, the training behavior of DoG was very unstable for Nano-GPT, and
the loss values were much higher than those of the other methods. Similar to DoG, Polyak stepsize
outperformed all parameter-free methods for T5, but the loss values of Polyak stepsize diverged for
LSTM and Nano-GPT. Thus, Inexact Polyak Stepsize can consistently succeed in training models for
all neural network architectures.

2https://github.com/karpathy/nanoGPT

8


---Page Break---
0
10000
20000
30000
Iteration

4

6

8

10

12

14

Training loss

0
10000
20000
30000
Iteration

4

6

8

10

12

14

Test loss

Clipped SGD
SGD
DoG
AdaSPS
DecSPS
Inexact Polyak Stepsize

(a) LSTM

0
1000
2000
3000
4000
5000
Iteration

2.5

3.0

3.5

4.0

Training loss

0
1000
2000
3000
4000
5000
Iteration

2.5

3.0

3.5

4.0

Test loss

Clipped SGD
SGD
DoG
AdaSPS
DecSPS
Inexact Polyak Stepsize

(b) Nano-GPT

0
20000
40000
60000
Iteration

0

20

40

60

80

100

Training loss

50000
55000
60000
65000
4.5

5.0

5.5

6.0

6.5

7.0

0
20000
40000
60000
Iteration

0

20

40

60

80

100

Test loss

50000
55000
60000
65000
4.5

5.0

5.5

6.0

6.5

7.0

Clipped SGD
SGD
DoG
Polyak
AdaSPS
DecSPS
Inexact Polyak Stepsize

(c) T5

Figure 3: Loss curves for LSTM, Nano-GPT, and T5. We plotted the training loss per 100, 10, and
10 iterations for LSTM, Nano-GPT, and T5, respectively. We plotted the test loss per one epoch, 100
iterations, and 200 iterations, respectively. For LSTM and Nano-GPT, we found that Polyak stepsize
does not converge, and its loss was much larger than that of other comparison methods. Thus, to
make the figure easier to read, we omit the results of Polyak stepsize and provide the complete results,
including Polyak stepsize in Sec. E.

7
Conclusion

In this study, we proposed Inexact Polyak Stepsize, which converges to the optimal solution with-
out hyperparameter tuning at the convergence rate that is asymptotically independent of L under
(L0, L1)-smoothness. Specifically, we first provided the novel convergence rate of Polyak stepsize
under (L0, L1)-smoothness, revealing that Polyak stepsize can achieve exactly the same conver-
gence rate as clipped gradient descent. Although Polyak stepsize can improve the convergence
under (L0, L1)-smoothness, Polyak stepsize requires the minimum loss value, which is a problem-
specific parameter. Then, we proposed Inexact Polyak Stepsize, which removes the problem-specific
parameter from Polyak stepsize without losing the property of asymptotic independence of L un-
der (L0, L1)-smoothness. We numerically validated our convergence results and demonstrated the
effectiveness of Inexact Polyak Stepsize.

9


---Page Break---
Acknowledgement

Y.T. was supported by KAKENHI Grant Number 23KJ1336. H.B. and M.Y. were supported by
MEXT KAKENHI Grant Number 24K03004. We thank Satoki Ishikawa for his helpful comments
on our experiments.

References

Berrada, L., Zisserman, A., and Kumar, M. P. (2020). Training neural networks for and by interpola-
tion. In International Conference on Machine Learning.

Carmon, Y. and Hinder, O. (2022). Making SGD parameter-free. In Conference on Learning Theory.

Defazio, A. and Mishchenko, K. (2023). Learning-rate-free learning by D-adaptation. In International
Conference on Machine Learning.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional
transformers for language understanding. In Association for Computational Linguistics.

Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and
stochastic optimization. In Journal of Machine Learning Research.

Galli, L., Rauhut, H., and Schmidt, M. (2023). Don't be so monotone: Relaxing stochastic line search
in over-parameterized models. In Advances in Neural Information Processing Systems.

Garrigos, G. and Gower, R. M. (2023). Handbook of convergence theorems for (stochastic) gradient
methods. In arXiv.

Hazan, E. and Kakade, S. M. (2019). Revisiting the Polyak step size. In arXiv.

Ivgi, M., Hinder, O., and Carmon, Y. (2023). DoG is SGD’s best friend: A parameter-free dynamic
step size schedule. In International Conference on Machine Learning.

Jiang, X. and Stich, S. U. (2023). Adaptive SGD with Polyak stepsize and line-search: Robust
convergence and variance reduction. In Advances in Neural Information Processing Systems.

Khaled, A., Mishchenko, K., and Jin, C. (2023). DoWG unleashed: An efficient universal parameter-
free gradient descent method. In Advances in Neural Information Processing Systems.

Kingma, D. and Ba, J. (2015). Adam: A method for stochastic optimization. In International
Conference on Learning Representations.

Koloskova, A., Hendrikx, H., and Stich, S. U. (2023). Revisiting gradient clipping: Stochastic bias
and tight convergence guarantees. In International Conference on Machine Learning.

Li, S. and Liu, Y. (2022). High probability guarantees for nonconvex stochastic gradient descent with
heavy tails. In International Conference on Machine Learning.

Loizou, N., Vaswani, S., Hadj Laradji, I., and Lacoste-Julien, S. (2021). Stochastic Polyak step-size
for SGD: An adaptive learning rate for fast convergence. In International Conference on Artificial
Intelligence and Statistics.

Merity, S., Keskar, N. S., and Socher, R. (2018). Regularizing and optimizing LSTM language
models. In International Conference on Learning Representations.

Mikolov, T., Karafiat, M., Burget, L., Cernocky, J. H., and Khudanpur, S. (2010). Recurrent neural
network based language model. In Interspeech.

Mukherjee, S., Loizou, N., and Stich, S. U. (2023). Locally adaptive federated learning via stochastic
polyak stepsizes. In arXiv.

Nawrot, P. (2023). NanoT5: A pytorch framework for pre-training and fine-tuning t5-style models
with limited resources. In arXiv.

10


---Page Break---
Nesterov, Y. (2018). Lectures on Convex Optimization. Springer.

Orabona, F. and Tommasi, T. (2017). Training deep networks without learning rates through coin
betting. In Advances in Neural Information Processing Systems.

Orvieto, A., Lacoste-Julien, S., and Loizou, N. (2022). Dynamics of SGD with stochastic Polyak
stepsizes: Truly adaptive variants and convergence to exact solution. In Advances in Neural
Information Processing Systems.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the difficulty of training recurrent neural
networks. In International Conference on Machine Learning.

Polyak, B. (1987). Introduction to Optimization. Optimization Software.

Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu,
P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. In
Journal of Machine Learning Research.

Sadiev, A., Danilova, M., Gorbunov, E., Horváth, S., Gidel, G., Dvurechensky, P., Gasnikov, A.,
and Richtárik, P. (2023). High-probability bounds for stochastic optimization and variational
inequalities: the case of unbounded variance. In International Conference on Machine Learning.

Zhang, B., Jin, J., Fang, C., and Wang, L. (2020a). Improved analysis of clipping algorithms for
non-convex optimization. In Advances in Neural Information Processing Systems.

Zhang, J., He, T., Sra, S., and Jadbabaie, A. (2020b). Why gradient clipping accelerates training: A
theoretical justification for adaptivity. In International Conference on Learning Representations.

Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S., Kumar, S., and Sra, S. (2020c). Why
are adaptive methods good for attention models? In Advances in Neural Information Processing
Systems.

11


---Page Break---
A
Proof of Theorem 4

Lemma 1. If Assumption 1 holds, the following holds for any x ∈Rd:

1
2L∥∇f(x)∥2 ≤f(x) −f(x⋆).
(16)

Proof. See Lemma 2.28 in (Garrigos and Gower, 2023).

Lemma 2. If Assumption 2 holds, the following holds for any x ∈Rd:

1
2(L0 + L1∥∇f(x)∥)∥∇f(x)∥2 ≤f(x) −f(x⋆).
(17)

Proof. See Lemma A.2 in (Koloskova et al., 2023).

Lemma 3. Assume that f is convex and Assumption 1 and 2 hold. Let T be the number of iterations
and define τ := arg min0≤t≤T −1 f(xt). Then, gradient descent with Polyak stepsize Eq. (7) satisfies:

f(xτ) −f(x⋆) ≤8L0∥x0 −x⋆∥2

T
+ 64LL2
1∥x0 −x⋆∥4

T 2
.

Proof. We have

∥xt+1 −x⋆∥2 = ∥xt −x⋆∥2 −2ηt⟨∇f(xt), xt −x⋆⟩+ η2
t ∥∇f(x)∥2

≤∥xt −x⋆∥2 −2ηt(f(xt) −f(x⋆)) + η2
t ∥∇f(x)∥2,

where we use the convexity of f in the inequality.

Case when ∥∇f(xt)∥≤L0

L1 : Substituting the stepsize, we get

∥xt+1 −x⋆∥2 ≤∥xt −x⋆∥2 −f(xt) −f(x⋆)

∥∇f(xt)∥2
(f(xt) −f(x⋆))

≤∥xt −x⋆∥2 −
1
2(L0 + L1∥∇f(xt)∥)(f(xt) −f(x⋆))

≤∥xt −x⋆∥2 −
1
4L0
(f(xt) −f(x⋆)),

where we use Lemma 2 in the second inequality. Unrolling the above inequality, we obtain

f(xt) −f(x⋆) ≤4L0
 
∥xt −x⋆∥2 −∥xt+1 −x⋆∥2
.

Case when ∥∇f(xt)∥> L0

L1 : Substituting the stepsize, we get

∥xt+1 −x⋆∥2 ≤∥xt −x⋆∥2 −(f(xt) −f(x⋆))2

∥∇f(xt)∥2

≤∥xt −x⋆∥2 −

r

f(xt) −f(x⋆)

2L
f(xt) −f(x⋆)

∥∇f(xt)∥
,

where we use Lemmas 1 in the last inequality. Then, ∥∇f(xt)∥> L0

L1 implies

L0 + L1∥∇f(xt)∥

2L1∥∇f(xt)∥
< 1.

Thus, we get

∥xt+1 −x⋆∥2 ≤∥xt −x⋆∥2 −

r

f(xt) −f(x⋆)

2L
f(xt) −f(x⋆)

∥∇f(xt)∥2
L0 + L1∥∇f(xt)∥

2L1

≤∥xt −x⋆∥2 −
1
4L1

r

f(xt) −f(x⋆)

2L
,

12


---Page Break---
where we use Lemma 2 in the last inequality. Unrolling the above inequality and multiplying L0 on
both sides, we get

L0
L1

r

f(x) −f(x⋆)

2L
≤4L0
 
∥xt −x⋆∥2 −∥xt+1 −x⋆∥2
.

Summing the two cases: Define T1 and T2 as follows:

T1 :=

t
∥∇f(xt)∥≤L0

L1


, T2 :=

t
∥∇f(xt)∥> L0

L1


.

We obtain

X

t∈T1
(f(xt) −f(x⋆)) + L0

L1

X

t∈T2

r

f(xt) −f(x⋆)

2L
≤4L0∥x0 −x⋆∥2.

Then, the above inequality implies

1
T

X

t∈T1
f(xt) −f(x⋆) ≤4L0∥x0 −x⋆∥2

T
,

1
T

X

t∈T2

p

f(xt) −f(x⋆) ≤4L1
√

2L∥x0 −x⋆∥2

T
.

Using a2 ≥2ab −b2, we obtain for any b ∈R

1
T

X

t∈T1


2b
p

f(xt) −f(x⋆) −b2
≤4L0∥x0 −x⋆∥2

T
.

Thus, when b > 0, we obtain

1
T

X

t∈T1

p

f(xt) −f(x⋆) ≤4L0∥x0 −x⋆∥2

2bT
+ b

2.

Choosing b =
q

4L0∥x0−x⋆∥2

T
, we get

1
T

X

t∈T1

p

f(xt) −f(x⋆) ≤

r

4L0∥x0 −x⋆∥2

T
.

Thus, we get

1
T

T −1
X

t=0

p

f(xt) −f(x⋆) ≤

r

4L0∥x0 −x⋆∥2

T
+ 4L1
√

2L∥x0 −x⋆∥2

T
.

Defining τ := arg mint f(xt), we get

p

f(xτ) −f(x⋆) ≤

r

4L0∥x0 −x⋆∥2

T
+ 4L1
√

2L∥x0 −x⋆∥2

T
.

Squaring the both sides, and using (a + b)2 ≤2a2 + 2b2 for all a, b ∈R, we obtain

f(xτ) −f(x⋆) ≤8L0∥x0 −x⋆∥2

T
+ 64LL2
1∥x0 −x⋆∥4

T 2
.

This concludes the statement.

13


---Page Break---
B
Proof of Theorem 5

Lemma 4. Assume that f is convex and Assumptions 1 and 2 hold. Let T be the number of iterations
and define τ := arg min0≤t≤T −1 f(xt). If f(xt) −f ⋆≥
σ2
√

T for all t, then gradient descent with
stepsize Eq. (14) satisfies:

f(xτ) −f ⋆≤8L0∥x0 −x⋆∥2 + 2σ2

√

T
+ 128L2
1L∥x0 −x⋆∥4

T
+ 8L2
1σ4L
L2
0T
.

where x⋆:= arg minx f(x) and σ2 := f ⋆−l⋆.

Proof. By the convexity of f, we have

∥xt+1 −x⋆∥2 = ∥xt −x⋆∥2 −2ηt⟨∇f(xt), xt −x⋆⟩+ η2
t ∥∇f(xt)∥2

≤∥xt −x⋆∥2 −2ηt(f(xt) −f ⋆) + η2
t ∥∇f(xt)∥2.

Substituting the stepsize Eq. (14), we get

∥xt+1 −x⋆∥2 ≤∥xt −x⋆∥2 −2ηt(f(xt) −f ⋆) + ηt
√

T
(f(xt) −l⋆)

≤∥xt −x⋆∥2 −ηt(2 −
1
√

T
)(f(xt) −f ⋆) + ηtσ2

√

T

≤∥xt −x⋆∥2 −ηt(f(xt) −f ⋆) + ηtσ2

√

T
,
(18)

where we use T ≥1 in the last inequality. Unrolling the above inequality and dividing by ηt, we
obtain

f(xt) −f ⋆≤∥xt −x⋆∥2 −∥xt+1 −x⋆∥2

ηt
+ σ2

√

T
.
(19)

Case when ∥∇f(xt)∥≤L0

L1 : From f(xt) −f ⋆≥
σ2
√

T and Eq. (18), we obtain

∥xt −x⋆∥2 −∥xt+1 −x⋆∥2 ≥0.
(20)

Thus, we get

f(xt) −f ⋆≤2(L0 + L1∥∇f(xt)∥)
√

T(∥xt −x⋆∥2 −∥xt+1 −x⋆∥2) + σ2

√

T

≤4L0
√

T(∥xt −x⋆∥2 −∥xt+1 −x⋆∥2) + σ2

√

T
,

where we use f ⋆≥l⋆and Lemma 2 for the first inequality and use ∥∇f(xt)∥≤L0

L1 for the last
inequality.

Case when ∥∇f(xt)∥> L0

L1 : From Lemma 2, we have

ηt ≥
f(xt) −f ⋆
√

T∥∇f(xt)∥2 ≥
1
2(L0 + L1∥∇f(xt)∥)
√

T
.

Then, we obtain

ηt ≥
1
4L1∥∇f(xt)∥
√

T
≥
1

4L1
p

2LT(f(xt) −f ⋆)
,

where we use ∥∇f(xt)∥> L0

L1 for the first inequality, and Lemma 1 for the last inequality. Combining
Eqs. (19) and (20), we obtain

f(xt) −f ⋆≤4L1
p

2LT(f(xt) −f ⋆)(∥xt −x⋆∥2 −∥xt+1 −x⋆∥2) + σ2

√

T
.

14


---Page Break---
Furthermore, from ∥∇f(xt)∥> L0

L1 and Lemma 1, we obtain

p

f(xt) −f ⋆≥L0

L1

s

f(xt) −f ⋆

∥∇f(xt)∥2 ≥L0

L1

r

1
2L.

Thus, we get
f(xt) −f ⋆

≤4L1
p

2LT(f(xt) −f ⋆)(∥xt −x⋆∥2 −∥xt+1 −x⋆∥2) + L1σ2

L0
√

T

p

2L(f(xt) −f ⋆).

Dividing by
L1√

2L(f(xt)−f ⋆)

L0
, we get

L0
L1

r

f(xt) −f ⋆

2L
≤4L0
√

T(∥xt −x⋆∥2 −∥xt+1 −x⋆∥2) + σ2

√

T
.

Summing the two cases: Define T1 and T2 as follows:

T1 :=

t
∥∇f(xt)∥≤L0

L1


, T2 :=

t
∥∇f(xt)∥> L0

L1


.

We obtain

1
T

 X

t∈T1
(f(xt) −f ⋆) + L0

L1

X

t∈T2

r

f(x) −f ⋆

2L

!

≤4L0∥x0 −x⋆∥2 + σ2

√

T
.

The above inequality implies that

1
T

X

t∈T1
(f(xt) −f ⋆) ≤4L0∥x0 −x⋆∥2 + σ2

√

T
,

1
T

X

t∈T2

p

f(x) −f ⋆≤4L1
√

2L∥x0 −x⋆∥2

√

T
+ L1σ2√

2L
L0
√

T
.

Using a2 ≥2ab −b2, we obtain for any b ∈R

1
T

X

t∈T1


2b
p

f(xt) −f ⋆−b2
≤4L0∥x0 −x⋆∥2 + σ2

√

T
.

Thus, when b > 0, we obtain
1
T

X

t∈T1

p

f(xt) −f ⋆≤4L0∥x0 −x⋆∥2 + σ2

2b
√

T
+ b

2.

Choosing b =
q

4L0∥x0−x⋆∥2+σ2

√

T
, we get

1
T

X

t∈T1

p

f(xt) −f ⋆≤

s

4L0∥x0 −x⋆∥2 + σ2

√

T
.

Thus, we get

1
T

T −1
X

t=0

p

f(xt) −f ⋆≤

s

4L0∥x0 −x⋆∥2 + σ2

√

T
+ 4L1
√

2L∥x0 −x⋆∥2

√

T
+ L1σ2√

2L
L0
√

T
.

Defining τ := arg mint f(xt), we get

p

f(xτ) −f ⋆≤

s

4L0∥x0 −x⋆∥2 + σ2

√

T
+ 4L1
√

2L∥x0 −x⋆∥2

√

T
+ L1σ2√

2L
L0
√

T
.

Squaring the both sides, and using (a + b)2 ≤2a2 + 2b2 for all a, b ∈R, we obtain

f(xτ) −f ⋆≤8L0∥x0 −x⋆∥2 + 2σ2

√

T
+ 128L2
1L∥x0 −x⋆∥4

T
+ 8L2
1σ4L
L2
0T
.

This concludes the statement.

15


---Page Break---
Lemma 5. Assume that f is convex and Assumptions 1 and 2 hold. Let T be the number of iterations
and define τ := arg min0≤t≤T −1 f(xt). Then, gradient descent with stepsize Eq. (14) satisfies:

f(xτ) −f(x⋆) ≤O
L0∥x0 −x⋆∥2 + σ2

√

T
+ LL2
1∥x0 −x⋆∥4

T
+ L2
1Lσ4

L2
0T


,
(21)

where x⋆:= arg minx f(x) and σ2 := f ⋆−l⋆.

Proof. If there exists t such that f(xt) −f ⋆<
σ2
√

T , we have

f(xτ) −f ⋆≤f(xt) −f ⋆< σ2

√

T
.

Then, if f(xt) −f ⋆≥
σ2
√

T for all t, Lemma 4 shows that

f(xτ) −f ⋆≤8L0∥x0 −x⋆∥2 + 2σ2

√

T
+ 128L2
1L∥x0 −x⋆∥4

T
+ 8L2
1σ4L
L2
0T
.

By combining the above two cases, we have the desired statement.

C
Additional theoretical result

Lemma 6. Let f be a function such that ∥∇2f(x)∥≤L0 + L1∥∇f(x)∥holds for any x. For any
x, y such that ∥x −y∥≤
1
L0 , we have

∥∇f(x) −∇f(y)∥≤2(L0 + L1∥∇f(x)∥)∥x −y∥.

Proof. See Lemma A.2 in (Zhang et al., 2020a).

Proposition 3. For any L0 ≥0 and L1 ≥0, f(x) := L0L2
1
72 x4 + L0

4 x2 is (L0, L1)-smooth.

Proof. Since f(x) is twice differentiable, we have

|∇2f(x)| = L0L2
1
6
x2 + L0

2 .

Using L1

6 x2 +
3
2L1 ≥|x|, we obtain

|∇2f(x)| ≤L0L2
1
6

L1

6 x2 +
3
2L1


|x| + L0

= L1

2


L0L2
1
18 x3 + L0

2 x
 + L0

2

= L1

2 |∇f(x)| + L0

2 .

From Lemma 6, we have the desired statement.

16


---Page Break---
D
Hyperparameter settings

D.1
Synthetic function

In our experiments, we ran the clipped gradient descent with the following hyperparameters and
tuned the hyperparameters by grid search.

Table 2: Hyperparameter settings for clipped gradient descent.

Learning Rate
{1, 1.0 × 10−1, · · · , 1.0 × 10−8}
Gradient Clipping Threshold
{0.01, 0.1, 1, 5, 10, 15, 20, ∞}

Table 3: Hyperparameters selected by grid search.
Gradient Descent
Clipped Gradient Descent
Learning Rate
Learning Rate
Gradient Clipping Threshold

L1 = 1
1.0 × 10−1
0.1
20
L1 = 10
1.0 × 10−3
0.1
10
L1 = 100
1.0 × 10−5
0.1
10
L1 = 1000
1.0 × 10−7
0.1
10

D.2
Neural networks

In our experiments, we used the following training configuration:

• LSTM: https://github.com/salesforce/awd-lstm-lm
• Nano-GPT: https://github.com/karpathy/nanoGPT
• T5: https://github.com/PiotrNawrot/nanoT5

We ran all experiments on an A100 GPU. For Clipped SGD and SGD, we tuned the stepsize and
gradient clipping threshold using the grid search. See Tables 4, 5, and 6 for detailed hyperparameter
settings, and see Table 7 for the selected hyperparameters.

Table 4: Hyperparameter settings for LSTM.
Learning Rate
{100, 50, 10, 1, 0.1, 0.01}
Gradient Clipping Threshold
{0.5, 1, · · · , 4.5, 5, ∞}
Batch Size
80

Table 5: Hyperparameter settings for Nano-GPT.
Learning Rate
{1, 0.5, 0.1, · · · , 0.0005, 0.0001}
Gradient Clipping Threshold
{1, 2, · · · , 9, 10, ∞}
Batch Size
64

Table 6: Hyperparameter settings for T5.
Learning Rate
{5.0, 1.0, 0.5, 0.1, 0.05}
Gradient Clipping Threshold
{1, 2, 3, ∞}
Batch Size
128

Table 7: Hyperparameters selected by grid search. Three values correspond to the selected hyperpa-
rameters for different seed values.

Gradient Descent
Clipped Gradient Descent
Learning Rate
Learning Rate
Gradient Clipping Threshold

LSTM
10 / 10 / 10
10 / 50 / 50
0.5 / 1 / 0.5
Nano-GPT
0.001 / 0.001 / 0.001
0.001 / 0.001 / 0.001
∞/ 10 / 10
T5
0.1 / 0.1 / 0.05
1 / 1 / 1
2 / 2 / 2

17


---Page Break---
E
Additional numerical evaluation

0
10000
20000
30000
Iteration

4

6

8

10

12

14

16

18

20

Training loss

0
10000
20000
30000
Iteration

4

6

8

10

12

14

Test loss

DoG
Clipped SGD
SGD
Polyak
AdaSPS
DecSPS
Inexact Polyak Stepsize

(a) LSTM

0
1000
2000
3000
4000
5000
Iteration

2

3

4

5

6

7

Training loss

0
1000
2000
3000
4000
5000
Iteration

2

3

4

5

6

7

Test loss

Clipped SGD
SGD
DoG
Polyak
AdaSPS
DecSPS
Inexact Polyak Stepsize

(b) Nano-GPT

Figure 4: Loss curves for LSTM and Nano-GPT. We plotted the training loss per 100, 10, and 10
iterations for LSTM, Nano-GPT, and T5, respectively. We plotted the test loss per one epoch, 100
iterations, and 200 iterations, respectively.

18


---Page Break---
0
500
1000
1500
2000
2500
Iteration

2.5

3.0

3.5

4.0

Training loss

0
500
1000
1500
2000
2500
Iteration

2.5

3.0

3.5

4.0

Test loss

Clipped SGD
SGD
DoG
AdaSPS
DecSPS
Inexact Polyak Stepsize

(a) T = 2500

0
2500
5000
7500
Iteration

2.0

2.5

3.0

3.5

4.0

Training loss

0
2500
5000
7500
Iteration

2.0

2.5

3.0

3.5

4.0

Test loss

Clipped SGD
SGD
DoG
AdaSPS
DecSPS
Inexact Polyak Stepsize

(b) T = 7500

Figure 5: Loss curves for Nano-GPT with different T.

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our main claims are clearly discussed in Sec. 1.

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

Justification: See Sec. 6.2.

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

20


---Page Break---
Justification: All proofs are provided in Sec. A and B.
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
Justification: All training configuration and hyperparameter setting are provided in Sec. D.
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

21


---Page Break---
Answer: [Yes]

Justification: Our code is contained in the supplementary material.

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

Justification: Our training configuration is provided in Sec. D.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [No]

Justification: We repeated the experiments in Sec. 6.2 with three different seed values and
reported the average, while we did not report error bars to make the figure easy to read.

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

22


---Page Break---
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

Justification: See Sec. D.

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

Justification: The authors read and complied with the code of ethics.

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

Justification: The motivation and its impact of our study are clearly discussed in Sec. 1.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

23


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

Justification: Our study does not provide any new dataset or pre-trained models.

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

Justification: See Sec. D.

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

24


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: Our code is provided in the supplementary material with MIT license.
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
Justification: Our experiments do not use any crowdsourcing service.
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
Justification: Our experiments do not use any crowdsourcing service.
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

25


---Page Break---
