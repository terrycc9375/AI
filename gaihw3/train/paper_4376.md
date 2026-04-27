Globally Q-linear Gauss-Newton Method for
Overparameterized Non-convex Matrix Sensing

Xixi Jia1∗Fangchen Feng2
Deyu Meng3,4
Defeng Sun5†

1School of Mathematics and Statistics, Xidian University
2L2TI Laboratory, University Sorbonne Paris Nord
3 School of Mathematics and Statistics, Xi’an Jiaotong University
4 Macao Institute of Systems Engineering, Macau University of Science and Technology
5 Department of Applied Mathematics, The Hong Kong Polytechnic University
hsijiaxidian@gmail.com, fangchen.feng@univ-paris13.fr,
dymeng@xjtu.edu.cn, defeng.sun@polyu.edu.hk

Abstract

This paper focuses on the optimization of overparameterized, non-convex low-rank
matrix sensing (LRMS)—an essential component in contemporary statistics and
machine learning. Recent years have witnessed significant breakthroughs in first-
order methods, such as gradient descent, for tackling this non-convex optimization
problem. However, the presence of numerous saddle points often prolongs the time
required for gradient descent to overcome these obstacles. Moreover, overparame-
terization can markedly decelerate gradient descent methods, transitioning its con-
vergence rate from linear to sub-linear. In this paper, we introduce an approximated
Gauss-Newton (AGN) method for tackling the non-convex LRMS problem. No-
tably, AGN incurs a computational cost comparable to gradient descent per iteration
but converges much faster without being slowed down by saddle points. We prove
that, despite the non-convexity of the objective function, AGN achieves Q-linear
convergence from random initialization to the global optimal solution. The global
Q-linear convergence of AGN represents a substantial enhancement over the conver-
gence of the existing methods for the overparameterized non-convex LRMS. The
code for this paper is available at https://github.com/hsijiaxidian/AGN.

1
Introduction

Matrix sensing aims to recover an unknown low-rank matrix M ∈Rn×n from its linear measurement
b = A(M). Here each elements bi is defined as bi = ⟨Ai, M⟩, with i = 1, · · · , m, and A(·) is a
nearly isometric linear operator. It holds significance not only in practical applications but also in the
realm of non-convex optimization [1–6]. As the problem often involves finding the optimal solution
of a non-convex minimization problem

min
U∈Rn×d,V ∈Rn×d f(U, V ) := 1

2∥A(UV ⊤) −b∥2
F ,
(1)

where rank(M) = r ≪n, of particular interest is the overparameterized case where d > r. The
objective function f(U, V ) to be minimized is non-convex, non-smooth3 and meanwhile lacks

∗Part of this work was completed while Xixi Jia was a research fellow in the Department of Applied
Mathematics at The Hong Kong Polytechnic University.
†Corresponding author.
3The non-smoothness pertains to (U, V ), as the magnitudes of these matrices can be highly unbalanced.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
coercivity, presenting significant challenges in solving the optimization problem. Since the objective
function exhibits certain key characteristics akin to the loss function of deep neural networks, problem
(1) stands as a cornerstone in the study of more challenging non-convex problems, such as those
encountered in deep learning [7–10]. For further discussions, please refer to [11].

Recent years have witnessed significant progress in the study of this non-convex optimization problem.

(a) Progress on gradient descent algorithms. Existing works such as [12–16] demonstrate that the
non-convex objective function f(U, V ) possesses benign loss landscape, wherein all local minima
are global, and concurrently, the Hessian exhibits negative eigenvalues at saddle points, allowing
perturbed gradient descent algorithms to effectively escape them. To handle the non-smooth problem,
prior studies [2, 12, 14, 17] introduce a regularization term 1

8∥U ⊤U −V ⊤V ∥2
F to the objective
function. This regularization ensures balance between the norms of U and V .

Very recently, Ye and Du [18] make a breakthrough on low-rank matrix factorization (LRMF), a
specific setting of the problem (1) (d = r, m →∞), and prove that gradient descent, without
perturbation and without the balance regularization on the objective function, converges at an R-
linear rate to the global optimal solution of the non-convex problem from random initialization.
Meanwhile, St¨oger and Soltanolkotabi [19] study the global convergence of gradient descent for
overparameterized (d > r) low-rank matrix sensing. However, overparameterization can significantly
slow down gradient descent from achieving linear convergence to sub-linear rates, as analyzed in
[15, 20]. Furthermore, Xiong et al. [20] proves that imbalanced initialization can expedite the
convergence of gradient descent from sub-linear to linear rate. Nevertheless, gradient descent still
requires a considerable amount of time to navigate away from saddle points, as discussed in Section
4. Additionally, the convergence rate of gradient descent is heavily reliant on the condition number of
the matrix M, rendering it inefficient for solving ill-conditioned non-convex optimization problems.

Table 1: Comparisons of iteration complexity, with
κ as the condition number of the n × n matrix.
“init.” denotes initialization.

Algorithm
init.
iteration complexity
GD [20]
random
κ11 log(κ2/n) + κ10 log(κ6/ε)
PrecGD [15]
spectral
log(1/ε)
ScaledGD(λ)[21]
random
log κ · log(κn) + log(1/ε)
AGN
random
log(1/ε)

(b) Progress on advanced algorithms. Given
these deficiencies of first-order gradient meth-
ods, it is intriguing and crucial to investigate
how computationally efficient higher-order al-
gorithms, behave on this non-convex problem.
Previously, Liu et al. [22] introduces a Gauss-
Newton type method for symmetric LRMF (with
d = r), and proved that Gauss-Newton method
converges Q-linearly fast to a critical point of the non-convex optimization problem. Recently,
Zilber and Nadle [23] prove that the Gauss-Newton method enjoys local quadratic convergence if the
initialization lies within a small basin of attraction of the global optimal solution. All these works
only guarantee local convergence and neglect the influence of saddle points on the convergence.
Global convergence of the Gauss-Newton method remains ambiguous. Yue et al. [24] proves that the
Newton method with cubic regularization converges quadratically fast from random initialization.
However, their results are only applicable for symmetric MS with d = r, and the computational cost
of Newton method in [24] for LRMS is very high. Lee and St¨oger [25] prove that for rank-one matrix
sensing, alternating least square method converges to the global optimal solution at a linear rate from
random initialization. However, it is uncertain whether the results of [25] are applicable to the r > 1
case as well as the challenging overparameterized scenario (d > r).

Another line of work deals with the deficiencies of gradient descent by incorporating preconditioning
matrices into the gradient direction, as demonstrated by [15, 26, 27, 21, 28]. Tanner and Wei [26]
introduce a scaled alternating steepest descent method with diminishing step size and provides
asymptotic convergence. Tong et al. [27] introduces ScaledGD and proves that, given a spectral
initialization, ScaledGD converges at a linear rate to the global optimal solution of problem (1).
Additionally, the convergence rate is independent of the condition number of M. The works in [15]
and [21] focus on the symmetric matrix sensing, extending the ScaledGD to the overparameterized
case by introducing a damping factor λ to control the singularity of the preconditioning matrix.
However, the damping factor λ can decelerate the ScaledGD from escaping the saddle regions and
the global iteration complexity becomes log κ · log(κn) + log(1/ε) as given in Table 1.

Our contributions. In this paper, we focus on the general model (1) which covers both symmetric
matrix sensing (M is symmetric and positive semi-definite) and asymmetric matrix sensing (M is
rectangular matrix), particularly with d > r. Building upon the insights from [23, 25, 26], we use
an approximated Gauss-Newton (AGN) method for solving the non-convex optimization problem

2


---Page Break---
(1). Notably, in each iteration, AGN performs computations akin to gradient descent, yet it exhibits
a Q-linear convergence rate towards the global optimal solution from random initialization, with
global iteration complexity log(1/ε) as shown in Table 1 AGN. Moreover, the Q-linear factor is
independent of the condition number of M. Under certain conditions on the sensing operator A(·), we
can prove super-linear convergence of the AGN. This distinctive convergence property substantially
enhances outcomes compared to existing methods in achieving global convergence, as shown in Table
1. Additionally, as a byproduct result, we establish that the prevailing preconditioned gradient descent
methods are analogous to the Levenberg–Marquardt method within the Gauss-Newton framework.

To conclude, the contributions of this paper are as follows. First, we reformulate the symmetric and
asymmetric matrix sensing in a unified way, based on which we design an approximated Gauss-
Newton (AGN) method. We show that the existing preconditioned gradient descent algorithms,
ScaledGD(λ) and PrecGD in [15, 21] respectively, correspond to a certain type of the Gauss-Newton
method. Then we analyze the behavior of GD, Scaled(λ)/PrecGD and AGN at ε-neighborhood of the
saddle points. The saddle point analysis shows that AGN is not attracted to saddle points, unlike GD
and PrecGD. Specifically, GD and Scaled(λ)/PrecGD achieve an objective function decrease of o(ε),
while AGN achieves a significantly larger, ε-independent decrease Θ(1). Finally, we prove the global
Q-linear convergence of AGN for overparameterized non-convex LRMS. This significantly improves
over algorithms like ScaledGD(λ) and PrecGD, which achieve R-linear convergence but are hindered
by saddle regions.

This paper is organized in the following ways. Section 2 provides an overview of related works.
Section 3 outlines our AGN method for both symmetric MS and asymmetric MS, and explores the
connections between AGN and Scaled(λ)/PrecGD. Section 4 provides insights into the behavior of
GD, ScaledGD, and AGN in the vicinity of saddle points. Section 5 outlines the main convergence
result with a proof sketch, Section 6 presents experimental results, and Section 7 concludes the paper.

2
Related work

2.1
Overparameterization in matrix sensing and beyond

Overparameterization significantly impacts the optimization of both the non-convex low-rank matrix
recovery problem [9, 15, 19, 29–31] and deep learning [32–35]. Specifically, [15, 29] analyzed that
overparamterization can eliminate the spurious local minima of the non-convex low-rank matrix
recovery problem. Meanwhile, since the exact rank parameter r is not predetermined, in real-world
applications one often relies on a moderately higher rank d > r, as elaborated in [21]. Geyer et
al. [36] study the solution uniqueness in the overparamterized low-rank matrix sensing. Additional
works examining overparameterized low-rank models include, but are not limited to [37–39].

2.2
Preconditioned/Scaled gradient descent
Preconditioned/Scaled gradient descent (ScaledGD) aims to enhance convergence by adjusting the
gradient direction using preconditioning matrices, as specified in [15, 26, 27, 21, 28, 40]. It resolves
the non-convex optimization problem (1) by the following iteration
(
Ut+1 = Ut −η∇Uf(Ut, Vt)(V ⊤
t Vt)−1,

Vt+1 = Vt −η∇V f(Ut, Vt)(U ⊤
t Ut)−1,
(2)

which corresponds to the methods in [27, 28, 40] for r = d. The author in [26] updates U and V in
an alternating manner. If U = V and d > r, then the iteration becomes ScaledGD(λ)/PrecGD
Ut+1 = Ut −η∇Uf(Ut, Ut)(U ⊤
t Ut + λtI)−1,
(3)
where λt > 0 can be either constant or time-varying, and the iteration corresponds to the methods in
[15, 21]. ScaledGD(λ) [21] and PrecGD [15] have been shown to achieve linear convergence to the
global optimal solution, whether in a local context or in a global context. However, the parameter λt
can degrade the global convergence, as shown in Fig. (2) where PrecGD struggles to escape saddle
regions before achieving local linear convergence.

2.3
Gauss-Newton method

The Gauss-Newton (GN) method is a widely recognized approach for nonlinear least-square

min
x∈Rn ψ(x) = 1

2∥ϕ(x)∥2
2,
(4)

3


---Page Break---
where ϕ : Rn →Rm is a nonlinear, twice continuously differentiable function. In each iteration t,
GN aims to solve least-squares of the linear approximation of ϕ(x) at point xt [41]

xt+1 = xt + η∆t, where ∆t = arg min
∆∈Rn ˆψ(x) = 1

2∥ϕ(xt) + J(xt)∆∥2
2,
(5)

where J(xt) = ϕ′(xt) is the Jacobian of the nonlinear function ϕ(·) at xt, and η > 0 is the step-
length. η = 1 corresponds to the GN method, η < 1 corresponds to the damped Gauss-Newton
method. The ∆t is calculated by ∆t = −[J(xt)⊤J(xt)]−1J(xt)⊤ϕ(xt), and J(xt)⊤J(xt) is an
approximation of the Hessian H(xt) = J(xt)⊤J(xt) + Pn
i=1 ϕi(xt)ϕ′′
i (xt) if the second-order
term is small. The GN method generally has a local convergence guarantee, indicating its effectiveness
primarily within the vicinity of a solution [22, 23, 41]. When applied to the LRMS problem (1), we
will show in Section 3.3 that the AGN method is closely related to the ScaledGD(λ)/PrecGD method.

3
Proposed method

In this section, we begin by unifying the formulation of both symmetric matrix sensing (where
U = V and M is positive semi-definite) and asymmetric matrix sensing problems. Based on the
corresponding nonlinear least-squares problem, we introduce our approximated Gauss-Newton (AGN)
method. Then we prove that the proposed AGN is a descent method. Furthermore, when addressing
symmetric matrix sensing, we explore two distinct parameterization settings and demonstrate that
employing an asymmetric parameterization can significantly enhance convergence. At last, we give
some discussions on the relation between ScaledGD(λ)/PrecGD and the proposed AGN method.

3.1
Approximated-Gauss-Newton (AGN) method for LRMS

We unify the formulation of symmetric and asymmetric LRMS into a single, simplified expression:

min
X∈R2n×d ψ(X) := 1

2∥A(PXX⊤Q) −b∥2
2,
(6)

where P = [I
0] ∈Rn×2n, Q =

0
I


∈R2n×n and I ∈Rn×n is the identity matrix. The

case X =

U
V


corresponds to asymmetric matrix sensing, X =

U
U


corresponds to symmetric

matrix sensing with U ∈Rn×n and V ∈Rn×n. The function ψ(X) can be further rewritten as
ψ(X) =
1
2∥ϕ(X)∥2
2, where ϕ(X) = A(PXX⊤Q) −b. For simplicity in notation, we denote
B(X, Y ) = A(PXY ⊤Q). By employing the Gauss-Newton framework as presented in Section 2.3,
one can update the variable as Xt+1 = Xt + η∆(Xt) where

∆(Xt) = arg
min
∆∈R2n×d
1
2∥B(∆, Xt) + B(Xt, ∆) + B(Xt, Xt) −b∥2
2.
(7)

However, as the Jacobian of the linear operator in the l2-norm of Eq. (7) tends to be singular in our
overparameterized setting (d > r), the Gauss-Newton method cannot be directly applied and one may
consult for the Levenberg–Marquardt method. In this work, however, unlike the Levenberg–Marquardt
method, we resolve the problem in Eq. (7) using the Gauss–Seidel method, whose advantages over
the Levenberg–Marquardt method will be discussed in Section 3.3. Specifically, we update Xt by

Xt+ 1

2 = Xt + η∆(Xt), ∆(Xt) = arg
min
∆∈R2n×d
1
2∥B(∆, Xt) + B(Xt, Xt) −b∥2
2,

Xt+1 = Xt+ 1

2 + η∆(Xt+ 1

2 ), ∆(Xt+ 1

2 ) = arg
min
∆∈R2n×d
1
2∥B(Xt+ 1

2 , ∆) + B(Xt+ 1

2 , Xt+ 1

2 ) −b∥2
2.

(8)
The sub-problems in Eq. (8) are quadratic minimization problems that can generally be solved very
easily. In our matrix sensing problem, leveraging the RIP condition, we can approximate ∆(Xt) by

ˆ∆(Xt) = arg+
min
∆∈R2n×d
1
2∥ˆB(∆, Xt) −A∗(B(Xt, Xt) −b)∥2
F ,

ˆ∆(Xt+ 1

2 ) = arg+
min
∆∈R2n×d
1
2∥ˆB(Xt+ 1

2 , ∆) −A∗(B(Xt+ 1

2 , Xt+ 1

2 ) −b)∥2
F ,
(9)

4


---Page Break---
where ˆB(∆, Xt) = P∆X⊤
t Q and ˆB(Xt+ 1

2 , ∆) = PXt+ 1

2 ∆⊤Q, arg+ denotes the minimum norm
solution as
ˆ∆(Xt) = P †A∗(B(Xt, Xt) −b)Q⊤Xt(X⊤
t QQ⊤X⊤
t )†,
ˆ∆(Xt+ 1

2 ) = Q⊤†[A∗(B(Xt+ 1

2 , Xt+ 1

2 ) −b)]⊤PXt+ 1

2 (X⊤
t+ 1

2 PP ⊤Xt+ 1

2 )†,
(10)

which is a natural choice for degenerate least squares problem. † denotes the Moore-Penrose-Pseudo
inverse, and P † = P ⊤, Q⊤† = Q in our context. Then the AGN becomes 4

Xt+ 1

2 = Xt −η ˆ∆(Xt), Xt+1 = Xt+ 1

2 −η ˆ∆(Xt+ 1

2 ).
(11)

The specifics of the AGN method are presented in Algorithm 1 in Appendix. We demonstrate that the
solution in Eq. (10) renders the AGN in Eq. (11) with a constant step-size η > 0 as a descent method.
Lemma 1. (Descent lemma) For asymmetric matrix sensing, as long as 0 < η ≤2/(1 + δ) and the
Assumption 1 is satisfied, then there exists positive constant ℓ= (2η −(1 + δ)η2)/2 such that

ψ(Xt+ 1

2 ) ≤ψ(Xt) −ℓ∥ˆB( ˆ∆(Xt), Xt)∥2
F ,

ψ(Xt+1) ≤ψ(Xt+ 1

2 ) −ℓ∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F .
(12)

Lemma 1 suggests that AGN with a constant step-size is indeed a descent method for the overparame-
terized LRMS. In Section 5, we will prove that AGN converges globally at Q-linear rate.

The AGN method can be used not only for asymmetric MS but also for symmetric MS, as Eq. (6)
offers a unified formulation for MS. While the application of AGN on symmetric MS will differ
slightly from the asymmetric case.

3.2
Symmetric matrix sensing

Now we consider symmetric matrix sensing, which is a special case of model (6) and the matrix
M ∈Rn×n is symmetric positive semi-definite (PSD). There are two different ways to deal with the
symmetric case, depending on whether we constrain X ∈C (symmetric parameterization), where

C =

Z|Z =

U
U


, U ∈Rn×d

⊂R2n×d or X ∈R2n×d (asymmetric parameterization).

Setting 1: Symmetric parameterization. In this case, the optimization variable X ∈C such that
B(X, X⊤) = A(UU ⊤) for matrix U ∈Rn×d and ˆB(∆, Xt) = [ ˆB(Xt, ∆)]⊤, thus the subproblems
in Eq. (9) become

˜∆(Xt) = arg min
∆∈C
1
2∥ˆB(∆, Xt) −A∗(B(Xt, Xt) −b)∥2
F ,
(13)

where Xt ∈C. The optimal solution to the problem in Eq. (13) is provided by the following lemma.

Lemma 2. Let ˆ∆(Xt) be the optimal solution of problem (9), then the optimal solution of problem

(13) is ˜∆(Xt) =

P ˆ∆(Xt)
P ˆ∆(Xt)


.

However, we find that if we constrain the search space to C and use the update ˜∆(Xt), the AGN
with a constant step-size η may not qualify as a descent method for our over-parameterized LRMS.

Specifically, let Xt+1 = Xt −η ˜∆(Xt), where Xt, ˜∆(Xt) ∈C and assume Xt =

Ut
Ut


, we have

ψ(Xt+1) = 1

2∥B(Xt, Xt) −ηB(Xt, ˜∆(Xt)) −ηB( ˜∆(Xt), Xt) + η2B( ˜∆(Xt), ˜∆(Xt)) −b∥2
2

= ψ(Xt) + η2∥B(Xt, ˜∆(Xt))∥2
2 −η∥ˆB(Xt, ˜∆(Xt))∥2
F + η4

2 ∥B( ˜∆(Xt), ˜∆(Xt))∥2
2

−η3 D
B( ˜∆(Xt), Xt), B( ˜∆(Xt), ˜∆(Xt))
E
+ η2 D
B(Xt, Xt), B( ˜∆(Xt), ˜∆(Xt))
E
.
(14)

4The specific update is provided in Appendix Eq. (32), Eq. (33) and Eq. (34).

5


---Page Break---
Note that the term ∥B(Xt, ˜∆(Xt))∥2
2 ≤(1 + δ)∥Et∥2
F , ∥ˆB(Xt, ˜∆(Xt))∥2
F ≤∥Et∥2
F are all bounded
and are closely related to ψ(Xt), and Et = A∗(A(UtU ⊤
t ) −b). While the higher-order term w.r.t.
˜∆(Xt)

∥B( ˜∆(Xt), ˜∆(Xt))∥2
2 = ∥A(EtUt(U ⊤
t Ut)−2U ⊤
t E⊤
t )∥2
2
≥(1 −δ)∥EtUt(U ⊤
t Ut)−1

2 (U ⊤
t Ut)−1(U ⊤
t Ut)−1

2 U ⊤
t E⊤
t ∥2
F
(15)

can be extremely large such that ψ(Xt+1) ≥ψ(Xt), as ∥EtUt(U ⊤
t Ut)−1

2 ∥2
F ≤∥Et∥2
F is bounded
while U ⊤
t Ut tends to be singular in the over-parametrized case. Therefore, one cannot guarantee that
the AGN method decreases with a constant step-size η, as illustrated by Fig. 3 AGNsym in Section 6.
How should we approach the symmetric matrix sensing problem using AGN? One possible strategy
is to relax the constraint X ∈C to a larger search space X ∈R2n×d.

Setting 2: Asymmetric parameterization. Despite M being a symmetric PSD matrix, one can still
consider problem (6) with X ∈R2n×d instead of X ∈C. Denote by X∗and X∗
c the optimal solution
with X∗∈R2n×d and X∗
c ∈C respectively, then it is easy to verify that PX∗X∗⊤Q = PX∗
c X∗⊤
c
Q
for symmetric matrix sensing. Therefore, one can readily apply AGN using Algorithm 1 to solve
problem (6) with M being a symmetric PSD matrix. We will achieve the same convergence guarantee
as in the case of asymmetric matrix sensing.

Remark 1. The symmetric MS discussed in setting 1 is a specific instance of problem (6), involving
significantly fewer intrinsic variables compared to the asymmetric case discussed in setting 2, the
search space C resides in a lower dimensional subspace of R2n×d. Meanwhile, one can also explore
the optimal solution for the symmetric MS within the expanded space R2n×d while maintaining the
same minimum objective function value, as demonstrated in setting 2. These two approaches lead
to significantly different optimization paths. From the above analysis, it is evident that different
optimization paths demonstrate distinct decreasing properties in our over-parameterized LRMS
problem. If we confine the optimization variable to C, then the AGN with a constant step-size may not
function as a descent method5. If we extend the optimization variable to the entire R2n×d, we have a
larger search space from which we can find a solution path that guarantees a significant decrease in
the objective function. In Section 5, we will prove that in this case, AGN converges Q-linearly fast.
These observations suggest that expanding the search space for a given optimization problem can
lead to more efficient methods.

3.3
Comparisons with related works

Of particular relevance to this work are ScaledGD(λ) [21] and PrecGD [15], which focus on over-
parameterized symmetric low-rank matrix sensing. While these preconditioned gradient descent
methods are not easily applicable to the general asymmetric matrix sensing problem, as we will discuss
in the appendix Section A.1.3. In this section, we demonstrate that these preconditioned gradient
descent methods are instances of the Levenberg–Marquardt method (specifically, the Gauss-Newton
method for singular least square problems) applied to the symmetric low-rank matrix sensing problem.
However, the Levenberg–Marquardt method is a more general approach than the preconditioned
gradient descent method, particularly for nonlinear least square problems.

Since ScaledGD(λ) [21] and PrecGD [15] consider over-parameterized symmetric matrix sens-
ing, we constraint X ∈C where C is defined in Section 3.2. It can be easily verified that the
ScaledGD(λ)/PrecGD for problem (6) corresponds to





Xt+1 = Xt −η∆(Xt, λ),

ˆ∆(Xt, λ) = arg min
∆∈C
1
2∥ˆB(∆, Xt) −A∗(B(Xt, Xt) −b)∥2
F + λ∥∆∥2
F .
(16)

It is apparent from Eq. (16) that λ constrains the magnitude of the update ∆(Xt, λ) to be small
compared to Eq. (13), thus ensuring that the preconditioned gradient descent method exhibits
monotonically decreasing behavior, as analyzed in [42] and [21]. The Lemma 6 in [42] ensures that
PrecGD is a descent method, as summarized by the following corollary

5One can employ a line search algorithm to ensure the decrease in the objective function value. Nevertheless,
line search will slow-down the convergence and is not the primary focus of this paper.

6


---Page Break---
Corollary 1. For symmetric matrix sensing, there exists positive constant ℓX,λ such that as long as
0 < η ≤2/ℓX,λ, the iteration given by Eq. (16) satisfies

ψ(Xt+1) ≤ψ(Xt) −ℓX,λ

2 ∥ˆB( ˆ∆(Xt, λ), Xt)∥2
F ,
(17)

where ℓX,λ = (1 + δ)

4 + 2∥ˆ
B(Xt,Xt)−M∥F +4∥ˆ
B( ˆ∆(Xt,λ),Xt)∥F
σmin(X⊤
t Xt)+λ
+

∥ˆ
B( ˆ∆(Xt,λ),Xt)∥F

σmin(X⊤
t Xt)+λ

2
.

Similar to the Lemma 1 of our AGN method, the value ∥ˆB( ˆ∆(Xt, λ), Xt)∥2
F plays very important
role for the convergence of ScaledGD(λ)/PrecGD. While we observe that the parameter λ can notably
impede the progress of ScaledGD(λ)/PrecGD in escaping the saddle point, as illustrated in Fig. 2
and discussed in Section 6. In Section 4, we will prove that when Xt is ε-close to the saddle points,
∥ˆB( ˆ∆(Xt, λ), Xt)∥2
F will be as small as o(ε), which explains why ScaledGD(λ)/PrecGD converges
slowly near saddle points. While in contrast, even if Xt is ε-close to the saddle points, the value
∥ˆB( ˆ∆(Xt), Xt)∥2
F in Eq. (12) is almost independent of ε, thus, saddle points cannot impede the
convergence of the AGN method, which is also demonstrated in the left subfigure of Fig. 2.

4
Saddle point analysis on the population risk

Saddle points are special critical points in non-convex optimization problem, contributing significantly
to the global convergence analysis of gradient-based algorithms in non-convex optimization. Past
researches [18] has demonstrated that gradient descent may encounter difficulties in navigating away
from saddle points, while our empirical findings in Section 5 demonstrate that the proposed AGN
does not experience slowdowns caused by saddle points. Therefore, it’s quite intriguing and crucial
to understand the behaviors of gradient descent and AGN in the vicinity of saddle points. To this end,
we study the saddle points of the population risk of the problem (6).

The population risk6 associated with the objective function in Eq. (6) corresponds to the following
non-convex matrix factorization problem:

min
X∈R2n×d
1
2∥PXX⊤Q −M∥2
2.
(18)

The objective function corresponds to g(U, V ) = 1

2∥UV ⊤−M∥2
F , U ∈Rn×d, V ∈Rn×d which
plays very important role in the saddle point analysis of Eq. (6). The saddle point of the non-convex
objective g(U, V ) is denoted by (Us, Vs) ∈S, where the set S is defined as follows:

S =

(Us, Vs)|UsV ⊤
s = ΦM(Σ)Ψ⊤, M = ΦΣΨ⊤, M ∈M/I
	
,
(19)

where M = ΦΣΨ⊤is the SVD of the matrix M and Φ ∈Rn×r, Ψ ∈Rn×r, Σ ∈Rr×r, M is the set
of mask operator7 and I is the identity operator.

Note that the gradient norm ∥∇g∥2
F is intricately linked to the reduction of the objective function for
gradient descent method. In our approach, the values ∥ˆB( ˆ∆(Xt), Xt)∥2
F and ∥ˆB(Xt, ˆ∆(Xt))∥2
F in
Eq. (12), which correspond to ∥∇Ug(V ⊤V )−1

2 ∥2
F and ∥∇V g(U ⊤U)−1

2 ∥2
F respectively, are directly
tied to the reduction observed in the AGN method. Correspondingly for symmetric matrix sensing,
the values ∥ˆB( ˆ∆(Xt, λ), Xt)∥2
F in Eq. (17), which corresponds to ∥∇Ug(U ⊤U + λI)−1

2 ∥2
F , is
related to the reduction of the non-convex objective by SclaedGD [21] and PrecGD [15]. Now, we
present the following theorem to describe the behavior of gradient descent, ScaledGD(λ)/PrecGD
and AGN in the vicinity of saddle points, by quantifying the values associated with their reductions
in objective functions. For simplicity, we consider here rank(M) = 1.

Theorem 1. Assume that M is rank-1, the point ( ˆU, ˆV ) with ˆU = Us + εNu, ˆV = Vs + εNv is
at the vicinity of the saddle point (Us, Vs) ∈S and ε is sufficiently small, Nu and Nv are random
Gaussian matrices that follow a standard normal distribution. Then with high probability we have
the following results
(GD)
∥∇g∥2
F = o(ε)es + o(ε2),
(20)

6The population risk of problem (6) corresponds m →∞in Eq. (1) and Eq. (6).
7M(·) : Rr×r →Rr×r is a mask operator which maps the element Xi,j of its input X into zero or Xi,j.

7


---Page Break---
(AGN)

(
∥∇g ˆU( ˆV ⊤ˆV )−1

2 ∥2
F = Θ(1)es + o(ε2),

∥∇g ˆV ( ˆU ⊤ˆU)−1

2 ∥2
F = Θ(1)es + o(ε2),
(21)

where es = ∥UsV ⊤
s
−M∥2
F . Furthermore, by constraining M to be positive semi-definite and
ˆU = ˆV , Us = Vs (for symmetric matrix sensing), for bounded constant c > 0, we have

(ScaledGD(λ))
∥∇g ˆU( ˆU ⊤ˆU + λI)−1

2 ∥2
F = Θ

ε2

ε2 + λ/c


es + o(ε2).
(22)

0
500
1000
1500
2000
Iteration count

10-15

10-10

10-5

100

Gradient norm

100
200
300
400
500
Iteration count

10-15

10-10

10-5

100

Figure 1: Illustration of the gradient norm for GD,
PrecGD, and the proposed AGN, with the right subfig-
ure showing a zoomed-in region of the left for iterations
from 100 to 500.

Theorem 1 indicates that when the opti-
mization variable ( ˆU, ˆV ) is ε-close to the
saddle point of the non-convex objective
function g( ˆU, ˆV ), the norm of the gradi-
ent ∥∇g∥2
F at ( ˆU, ˆV ) becomes as small as
o(ε). Hence, the convergence of gradient
descent will be relatively slow, as shown
in Fig. 2. Moreover, in Eq. (22), the
value of λ/c is typically much larger than
ε. Consequently, the reduction achieved by
ScaledGD(λ)/PrecGD for symmetric ma-
trix sensing is nearly identical to that of
gradient descent o(ε). In contrast, even if
( ˆU, ˆV ) is ε-close to the saddle point, the reduction in the non-convex objective achieved by AGN in
Eq. (21) is as significant as Θ(1)es + o(ε), where Θ(1)es is larger than ε and is not dependent on
ε, thus ensuring that AGN achieves a substantial decrease in the objective function near the saddle
points. We also plot the gradient norm of GD, PrecGD and AGN in Fig. 1 for solving problem (6). It
can be seen from Fig. 1 that the gradient norm of AGN decreases linearly to zero. While the gradient
norm of PrecGD suffers from ups and downs before it is smaller than about 1 × 10−7. This indicates
that PrecGD is attracted to saddle points but can quickly escape, depending on the value of λ as per
Eq. (22). It becomes more challenging for GD to quickly escape all saddle points, which is due to Eq.
(20). As shown in Fig. 1, GD’s iterations encounter multiple saddle points before reaching the global
minimum.

5
Global convergence analysis

We first recall the celebrated Restricted Isometry Property (RIP) [43], then we make some mild
assumptions on the restricted isometry constant and the initialization of the variable X0.

5.1
Assumptions and main result
Assumption 1. The operator A(·) satisfies the rank-r + 1 RIP with constant δr+1 := δ. Further
more, there exist a sufficiently small constant cδ > 0 and a sufficient large Cδ > 0 such that
δ ≤cδr−1/2κ−Cδ,
(23)
where κ is the condition number of the matrix M.
Assumption 2. Let X0 ∈R2n×d be random Gaussian with elements sampled from N(0, σ) with
σ ≤c0∥Σ∥2/n for small c0 and the step size 0 < η ≤2/(1 + δ).

Now we present our main result, which characterizes the global Q-linear convergence of the proposed
AGN for the over-parameterized, non-convex, low-rank matrix sensing problem.
Theorem 2 (Global Q-linear convergence). Under the Assumption 1 and Assumption 2. Let ψ∗be
the global minimal value of ψ(X) in Eq. (6) and Xt, ∀t > 0 is generated by Algorithm 1, then there
exists constants 1 ≥τ > 0 such that
ψ(Xt+1) −ψ∗≤cq[ψ(Xt) −ψ∗], ∀t > 0,
(24)

where cq = (1 −ˆℓ1−δ

1+δ τ) < 1 and ˆℓ= 2η −(1 + δ)η2. Meanwhile, if δ = 0, η = 1, cq becomes 0.

Theorem 2 echoes the observation in the left subfigure of Fig. 2 that AGN converges rapidly from
random initialization and does not become trapped in saddle regions. The convergence result of AGN
differs significantly from existing methods like ScaledGD(λ) [21], PrecGD [15] and GD [19, 20],
both empirically and theoretically. In the next section, we will present the sketch of our proof.

8


---Page Break---
5.2
Proof sketch
The global Q-linear convergence of the AGN method relies on two conditions: monotonically
decreasing (Lemma 1) and decrease dominant (Lemma 3). The monotonically decreasing condition
ensures that the objective function decreases in each iteration, while the decrease dominant condition
guarantees that the decrease in the function value is significantly larger than the distance between the
current function value and the global minimum.
Lemma 3 (Decrease dominant). Under Assumption 1 and 2, let Xt be updated by AGN method in
Algorithm 1, then there exist τ 1
t , τ 2
t and constant τ with 1 ≥max{τ 1
t , τ 2
t } ≥min{τ 1
t , τ 2
t } ≥τ > 0
such that
∥ˆB( ˆ∆(Xt), Xt)∥2
F ≥1 −δ

1 + δ τ 1
t [ψ(Xt) −ψ∗],

∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F ≥1 −δ

1 + δ τ 2
t [ψ(Xt+ 1

2 ) −ψ∗],
(25)

where ˆ∆(Xt) and ˆ∆(Xt+ 1

2 ) is from Eq. (10).

Note that the linear operator A(·) and the l2-norm is unitarily invariant, therefore for simplicity we
consider M to be a diagonal matrix Σ ∈Rn×n with r nonzero elements on the diagonal. Specifically
one can simply write Σ = Φ⊤MΨ where M = ΦΣΨ⊤is the SVD of matrix M, as analyzed

in [18] and [20]. Let Xt =

Ut
Vt


, then ∥ˆB( ˆ∆(Xt), Xt)∥2
F = ∥A∗(A(UtV ⊤
t
−Σ))VtV †
t ∥2
F and

correspondingly ψ(Xt) −ψ∗= 1

2∥A(UtV ⊤
t
−Σ)∥2
2. According to the following Lemma 4 and
the RIP condition in Definition 1, to guarantee the inequality in Eq. (25), we need to ensure that
∥(UtV ⊤
t
−Σ)VtV †
t ∥2
F ≥τ1∥UtV ⊤
t
−Σ∥2
F as presented by Lemma 5.

Lemma 4. Assume that the operator A(·) satisfies the RIP condition in Definition 1 and Assumption
1 with constant δ, for any U, V ∈Rn×d, Σ ∈Rn×n and Z ∈Rn×n, then we have that

∥A∗A(UV ⊤−Σ)Z∥F ≥(1 −δ)∥(UV ⊤−Σ)Z∥F .
(26)

Lemma 5. Under Assumptions 1 and 2, if Xt, ∀t > 0 is generated by the AGN method in Algorithm

1 and let Xt =

Ut
Vt


, then there exist constant τ, τ 1
t , τ 2
t with max{τ 1
t , τ 2
t } ≥min{τ 1
t , τ 2
t } ≥τ > 0

such that
∥(UtV ⊤
t
−Σ)VtV †
t ∥2
F ≥τ 1
t ∥UtV ⊤
t
−Σ∥2
F ,

∥(VtU ⊤
t+ 1

2 −Σ)Ut+ 1

2 U †
t+ 1

2 ∥2
F ≥τ 2
t ∥Ut+ 1

2 V ⊤
t
−Σ∥2
F .
(27)

Theorem 2 can be proven readily by combining all of these lemmas. Please refer to the Appendix for
a more detailed proof.

6
Numerical experiments

0
100
200
300
400
500
Iteration count

10-10

10-5

100

Relative error

Saddle Regions

Saddle Regions

Local linear convergence phase

400
600
800
1000
Iteration count

10-10

10-5

100

Relative error

Figure 2: Comparison of convergence for PrecGD,
GD, and AGN across various condition numbers,
with the right subfigure extending the left by iterat-
ing from 300 to 1000.

In this section, we conduct experiments to
demonstrate the effectiveness of the proposed
AGN method for solving the over-parameterized
non-convex matrix sensing problem. We set
the ground truth matrix M = U ∗ΣV ∗⊤, with
U ∗∈Rn×r, V ∗∈Rn×r random orthogonal
matrices and Σ is a diagonal matrix with condi-
tion number κ. We set n = 100, r = 5, d = 3r
and the number of sensing matrices m = 50nr.
All experiments were conducted using MAT-
LAB on a MacBook Pro with a 2.4 GHz quad-
core Intel Core i5 CPU and 8 GB of memory.

Comparison with representative methods.
We compare AGN with GD [20] and PrecGD [15] on asymmetric over-parameterized matrix sensing.
All the competing methods are initialized with random Gaussian matrix with zero mean the variance
1/n. We plot the training curves of the competing methods in Fig. 2, where the relative error is

9


---Page Break---
defined by ∥UtV ⊤
t
−M ∗∥F /∥M ∗∥F . GD’s slow convergence is evident as it struggles to escape
saddle points, as shown in the saddle regions of Fig. 2. Meanwhile, GD’s final convergence rate
depends on the condition number κ. Fig. 2 further shows that PrecGD’s final convergence rate is
independent of κ, though it still progresses slowly in saddle regions. In contrast, the relative error
of AGN decreases rapidly, and its convergence remains unaffected by saddle points, consistent with
Theorem 2.

Table 2: Comparison of computational time for
methods on matrices with varying dimensions n,
measured in seconds. Here, a+ indicates time
significantly exceeds a seconds.

Algorithm
100 × 100
500 × 500
κ = 10
κ = 50
κ = 10
κ = 50
GD [20]
500+
500+
5000+
5000+
PrecGD [15]
87.12
94.53
1979.45
1921.83
ScaledGD(λ)[21]
55.87
69.07
1218.83
1258.71
AGN(ours)
27.24
25.52
617.58
632.24

Meanwhile, we compare the computational time
of GD [20], PrecGD [15], ScaledGD(λ) [21],
and the proposed AGN on matrices of varying
dimensions n×n under different condition num-
ber κ in Table 2. It can be seen from Table 2 that
AGN is significantly faster than the competing
methods, particularly PrecGD and ScaledGD(λ),
while vanilla gradient descent converges much
more slowly.

Asymmetric vs. symmetric parameterization
of the symmetric MS. We also conduct experiments to illustrate the differences between symmetric
and asymmetric parameterization in symmetric matrix sensing, as discussed in subsection 3.2. As
analyzed in section 3.2 setting 1, the matrix U ⊤
t Ut tends to be singular in over-parameterized matrix
sensing, causing ∥B( ˜∆(Xt), ˜∆(Xt))∥2
2 to become extremely large, leading to ψ(Xt+1) ≥ψ(Xt), as
shown in Eq. (14). Thus, we cannot guarantee that AGN in symmetric parameterization is a reliable
descent method, as demonstrated by AGNsym in Fig. 3. However, with asymmetric parameterization,
as outlined in subsection 3.2 setting 2, we can ensure the linear convergence of the AGN method,
also illustrated by AGNasym in Fig. 3.

7
Conclusion

0
50
100
150
200
Iteration count

10-10

10-5

100

Relative error

Figure 3: Convergence of AGN un-
der Sym. and Asym. parameteriza-
tion of symmetric LRMS.

In this paper, we present an approximated Gauss-Newton
(AGN) method for overparameterized non-convex low-rank
matrix sensing problem. We demonstrate the close relation-
ship between existing methods like ScaledGD(λ) and PrecGD,
and the Levenberg–Marquardt method, which is a variant of
the Gauss-Newton method. Through saddle point analysis, we
partially explain why gradient descent, Scaled(λ)/PrecGD may
be slowed down by saddle points, whereas the proposed AGN
achieves fast convergence. Finally, we prove that the proposed
AGN achieves Q-linear convergence from random Gauss initial-
ization for the non-convex optimization problem. Our findings
highlight the efficacy of (approximate) second-order methods in
non-convex optimization, particularly for structured problems
like non-convex matrix sensing. Moreover, our results can be
extended to more complex challenges, such as optimizing deep
neural networks.

Limitations: While the AGN method converges quickly and is efficient for low-rank matrix sensing
problems, it may be less effective than gradient descent for general problems without low-rank
structure due to the need to solve a least-squares problem for the Gauss-Newton direction. Future
work will explore approximate methods, like the conjugate gradient, to address this. Additionally, our
current saddle point analysis focuses on a simple non-convex LRMS case with a zero RIP constant,
which we will generalize to more complex scenarios with larger RIP constants.

Acknowledgements

This research was supported by the National Key R&D Program of China (2022YFA1004100), the
China NSFC projects under grant 62372359, 12226004, 62272375, the Research Center for Intelligent
Operations Research and RGC Senior Research Fellow Scheme No. SRFS2223-5S02, and the Macao
Science and Technology Development Fund under Grant 061/2020/A2.

10


---Page Break---
References

[1] Chih-Fan Chen, Chia-Po Wei, and Yu-Chiang Frank Wang. Low-rank matrix recovery with
structural incoherence for robust face recognition. In 2012 IEEE conference on computer vision
and pattern recognition, pages 2618–2625. IEEE, 2012.

[2] Dohyung Park, Anastasios Kyrillidis, Constantine Carmanis, and Sujay Sanghavi. Non-square
matrix sensing without spurious local minima via the Burer-Monteiro approach. In Artificial
Intelligence and Statistics, pages 65–74. PMLR, 2017.

[3] Emmanuel J Candes. The restricted isometry property and its implications for compressed
sensing. Comptes rendus. Mathematique, 346(9-10):589–592, 2008.

[4] Shuangjiang Li and Hairong Qi. A Douglas-Rachford splitting approach to compressed sensing
image recovery using low-rank regularization. IEEE Transactions on Image Processing, 24(11):
4240–4249, 2015.

[5] Yuejie Chi, Yue M Lu, and Yuxin Chen. Nonconvex optimization meets low-rank matrix
factorization: An overview. IEEE Transactions on Signal Processing, 67(20):5239–5269, 2019.

[6] Vasileios Charisopoulos, Yudong Chen, Damek Davis, Mateo Díaz, Lijun Ding, and Dmitriy
Drusvyatskiy. Low-rank matrix recovery with composite optimization: good conditioning and
rapid convergence. Foundations of Computational Mathematics, 21(6):1505–1593, 2021.

[7] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix
factorization. In Advances in Neural Information Processing Systems, pages 7413–7424, 2019.

[8] Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon Shaolei Du, and Jason D Lee. Understanding incre-
mental learning of gradient descent: A fine-grained analysis of matrix sensing. In International
Conference on Machine Learning, pages 15200–15238. PMLR, 2023.

[9] Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. Algorithmic regularization in over-parameterized
matrix sensing and neural networks with quadratic activations. In Conference On Learning
Theory, pages 2–47. PMLR, 2018.

[10] Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of op-
timization and generalization for overparameterized two-layer neural networks. In International
Conference on Machine Learning, pages 322–332. PMLR, 2019.

[11] Simon S Du, Wei Hu, and Jason D Lee. Algorithmic regularization in learning deep homoge-
neous models: Layers are automatically balanced. In Advances in neural information processing
systems, pages 382–393, 2018.

[12] Rong Ge, Chi Jin, and Yi Zheng. No spurious local minima in nonconvex low rank problems: A
unified geometric analysis. In International Conference on Machine Learning, pages 1233–1242.
PMLR, 2017.

[13] Zhihui Zhu, Qiuwei Li, Gongguo Tang, and Michael B Wakin. The global optimization geometry
of low-rank matrix optimization. IEEE Transactions on Information Theory, 67(2):1308–1331,
2021.

[14] Xingguo Li, Junwei Lu, Raman Arora, Jarvis Haupt, Han Liu, Zhaoran Wang, and Tuo Zhao.
Symmetry, saddle points, and global optimization landscape of nonconvex matrix factorization.
IEEE Transactions on Information Theory, 65(6):3489–3514, 2019.

[15] Jialun Zhang, Salar Fattahi, and Richard Y Zhang. Preconditioned gradient descent for over-
parameterized nonconvex matrix factorization. Advances in Neural Information Processing
Systems, 34:5985–5996, 2021.

[16] Srinadh Bhojanapalli, Behnam Neyshabur, and Nathan Srebro. Global optimality of local search
for low rank matrix recovery. In Advances in Neural Information Processing Systems, pages
3880–3888, 2016.

11


---Page Break---
[17] Stephen Tu, Ross Boczar, Max Simchowitz, Mahdi Soltanolkotabi, and Ben Recht. Low-rank
solutions of linear matrix equations via procrustes flow. In International Conference on Machine
Learning, pages 964–973. PMLR, 2016.

[18] Tian Ye and Simon S Du. Global convergence of gradient descent for asymmetric low-rank
matrix factorization. In Advances in Neural Information Processing Systems, pages 1429–1439,
2021.

[19] Dominik Stöger and Mahdi Soltanolkotabi. Small random initialization is akin to spectral
learning: Optimization and generalization guarantees for overparameterized low-rank matrix
reconstruction. Advances in Neural Information Processing Systems, 34:23831–23843, 2021.

[20] Nuoya Xiong, Lijun Ding, and Simon Shaolei Du. How over-parameterization slows down
gradient descent in matrix sensing: The curses of symmetry and initialization. In The Twelfth
International Conference on Learning Representations, 2023.

[21] Xingyu Xu, Yandi Shen, Yuejie Chi, and Cong Ma. The power of preconditioning in overpa-
rameterized low-rank matrix sensing. In International Conference on Machine Learning, pages
38611–38654. PMLR, 2023.

[22] Xin Liu, Zaiwen Wen, and Yin Zhang. An efficient Gauss-Newton algorithm for symmetric
low-rank product matrix approximations. SIAM Journal on Optimization, 25(3):1571–1608,
2015.

[23] Pini Zilber and Boaz Nadler. Gnmr: A provable one-line algorithm for low rank matrix recovery.
SIAM journal on mathematics of data science, 4(2):909–934, 2022.

[24] Man-Chung Yue, Zirui Zhou, and Anthony Man-Cho So. On the quadratic convergence of the
cubic regularization method under a local error bound condition. SIAM Journal on Optimization,
29(1):904–932, 2019.

[25] Kiryung Lee and Dominik Stöger. Randomly initialized alternating least squares: Fast con-
vergence for matrix sensing. SIAM Journal on Mathematics of Data Science, 5(3):774–799,
2023.

[26] Jared Tanner and Ke Wei. Low rank matrix completion by alternating steepest descent methods.
Applied and Computational Harmonic Analysis, 40(2):417–429, 2016.

[27] Tian Tong, Cong Ma, and Yuejie Chi. Accelerating ill-conditioned low-rank matrix estimation
via scaled gradient descent. Journal of Machine Learning Research, 22(150):1–63, 2021.

[28] Xixi Jia, Hailin Wang, Jiangjun Peng, Xiangchu Feng, and Deyu Meng. Preconditioning matters:
Fast global convergence of non-convex matrix factorization via scaled gradient descent. In
Advances in Neural Information Processing Systems, volume 36, 2023.

[29] Richard Y Zhang. Sharp global guarantees for nonconvex low-rank matrix recovery in the
overparameterized regime. arXiv preprint arXiv:2104.10790, 2021.

[30] Jianhao Ma and Salar Fattahi. Global convergence of sub-gradient method for robust matrix
recovery: Small initialization, noisy measurements, and over-parameterization. Journal of
Machine Learning Research, 24(96):1–84, 2023.

[31] Lijun Ding, Liwei Jiang, Yudong Chen, Qing Qu, and Zhihui Zhu. Rank overspecified robust
matrix recovery: Subgradient method and exact recovery. Advances in Neural Information
Processing Systems, 34:26767–26778, 2021.

[32] Zeyuan Allen-Zhu, Yuanzhi Li, and Yingyu Liang. Learning and generalization in overpa-
rameterized neural networks, going beyond two layers. In Advances in Neural Information
Processing Systems, pages 6158–6169, 2019.

[33] Cong Fang, Jason Lee, Pengkun Yang, and Tong Zhang. Modeling from features: a mean-field
framework for over-parameterized deep neural networks. In Conference on learning theory,
pages 1887–1936. PMLR, 2021.

12


---Page Break---
[34] Itay Safran and Ohad Shamir. Spurious local minima are common in two-layer relu neural
networks. In International conference on machine learning, pages 4433–4441. PMLR, 2018.

[35] Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Gradient descent optimizes over-
parameterized deep relu networks. Machine learning, 109:467–492, 2020.

[36] Kelly Geyer, Anastasios Kyrillidis, and Amir Kalev. Low-rank regularization and solution
uniqueness in over-parameterized matrix sensing. In International Conference on Artificial
Intelligence and Statistics, pages 930–940. PMLR, 2020.

[37] Samet Oymak and Mahdi Soltanolkotabi. Overparameterized nonlinear learning: Gradient
descent takes the shortest path? In International Conference on Machine Learning, pages
4951–4960. PMLR, 2019.

[38] Mahdi Soltanolkotabi, Adel Javanmard, and Jason D Lee. Theoretical insights into the op-
timization landscape of over-parameterized shallow neural networks. IEEE Transactions on
Information Theory, 65(2):742–769, 2018.

[39] Richard Y Zhang. Improved global guarantees for the nonconvex burer–monteiro factorization
via rank overparameterization. arXiv preprint arXiv:2207.01789, 2022.

[40] K Adithya Apuroop. A riemannian geometry for low-rank matrix completion. arXiv preprint
arXiv:1211.1550, 2012.

[41] Serge Gratton, Amos S Lawless, and Nancy K Nichols. Approximate Gauss-Newton methods
for nonlinear least squares problems. SIAM Journal on Optimization, 18(1):106–132, 2007.

[42] Gavin Zhang, Salar Fattahi, and Richard Y Zhang. Preconditioned gradient descent for over-
parameterized nonconvex burer–monteiro factorization with global optimality certification.
Journal of Machine Learning Research, 24(163):1–55, 2023.

[43] Benjamin Recht, Maryam Fazel, and Pablo A Parrilo. Guaranteed minimum-rank solutions of
linear matrix equations via nuclear norm minimization. SIAM review, 52(3):471–501, 2010.

13


---Page Break---
A
Appendix / supplemental material

A.1
Preliminaries and more details

A.1.1
The definition of RIP

Definition 1 (Restricted Isometry Property). The linear operator A(·) is said to satisfy rank-r RIP
with a constant δr ∈[0, 1) if for all matrices M of rank at most r the following condition holds

(1 −δr)∥M∥2
F ≤∥A(M)∥2
2 ≤(1 + δr)∥M∥2
F .
(28)

A.1.2
The main AGN algorithm

Algorithm 1: AGN for matrix sensing
Data: A(·), b, η and the random Gauss initialization X0, t = 0.
Result: The estimated solution X and the low-rank matrix ˆ
M = PXX⊤Q.
while not end do

t = t + 1;
Update the approximated Gauss-Newton direction by Eq. (10);
Update Xt and Xt+ 1

2 by Eq. (11);
end

A.1.3
ScaledGD(λ)/PrecGD for LRMS problem (6)

Directly applying the ScaledGD(λ)/PrecGD methods in [21] and [15] respectively to problem (6)
will lead to the following iterative update as

Xk+1 = Xk −η∇ψ(Xt)(X⊤
t Xt + λtI)−1,
(29)

where λt can be either constant or time-varying. Note that in our LRMS problem (6), Xt =

Ut
Vt


,

therefore the update in Eq. (29) becomes

Ut+1 = Ut −ηA∗A(UtV ⊤
t
−Σ)Vt(U ⊤
t Ut + V ⊤
t Vt + λtI)−1,
(30)

and
Vt+1 = Vt −η[A∗A(UtV ⊤
t
−Σ)]⊤Ut(U ⊤
t Ut + V ⊤
t Vt + λtI)−1,
(31)

which leads to quite different method compared to our AGN with iterations in Ut and Vt given by
the following Eq. (33) and Eq. (34). Meanwhile, as shown in Fig. 4, using Eq. (29) for our LRMS
(denoted by PrecGD) results in slower convergence compared to our AGN method, and moreover the
convergence rate is dependent on the condition number of M as κ(M). In contrast, the proposed
AGN converges very quickly, and its convergence rate is independent of the condition number κ(M).

A.1.4
Some detailed derivations.

Before presenting the proof, we first provide additional details regarding the update described in

Eq. (11). Specifically, let Xt =

Ut
Vt


, with Ut =
 ˆUt
Jt


and Vt =
 ˆVt
Kt


where ˆUt, ˆVt ∈Rr×d and

Jt, Kt ∈R(n−r)×d. Since Σ =

Σr
0
0
0


with Σr ∈Rr×r, Thus, the update in Eq. (11) can be

succinctly expressed as

Xt+ 1

2 = Xt −η ˆ∆(Xt) =

Ut+1
Vt


, Xt+1 = Xt+ 1

2 −η ˆ∆(Xt+ 1

2 ) =

Ut+1
Vt+1


,
(32)

with Ut+1 and Vt+1 given by

Ut+1 = Ut −ηA∗A(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)†,
(33)

14


---Page Break---
0
500
1000
1500
2000
Iteration count

10-15

10-10

10-5

100

Relative error

Figure 4: Convergence of AGN and the method by using Eq. (29) (denoted by PrecGD) for LRMS
with different condition numbers on matrix M.

and
Vt+1 = Vt −η[A∗A(Ut+1V ⊤
t
−Σ)]⊤Ut+1(U ⊤
t+1Ut+1)†,
(34)

and correspondingly

ˆ∆(Xt) =

A∗A(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)†
0


,
(35)

ˆ∆(Xt+ 1

2 ) =

0
[A∗A(Ut+1V ⊤
t
−Σ)]⊤Ut+1(U ⊤
t+1Ut+1)†


.
(36)

In consequence, we have

∥ˆB( ˆ∆(Xt), Xt)∥2
F = A∗A(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)†V ⊤
t ,
(37)

and
∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F = Ut+1(U ⊤
t+1Ut+1)†U ⊤
t+1A∗A(Ut+1V ⊤
t
−Σ).
(38)

Further more, we can reformulate Ut+1 and Vt+1 as

Ut+1 = Ut −ηA∗A(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)†

= Ut −η(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)† + η(I −A∗A)(UtV ⊤
t
−Σ)Vt(V ⊤
t Vt)†

= (1 −η)
 ˆUt
Jt


+ η

Σr
0
0
0

  ˆVt
Kt

 
ˆV ⊤
t ˆVt + K⊤
t Kt
†

+ η(I −A∗A)
 ˆUt ˆV ⊤
t
−Σr
ˆUtK⊤
t
Jt ˆV ⊤
t
JtK⊤
t

  ˆVt
Kt

 
ˆV ⊤
t ˆVt + K⊤
t Kt
†

= (1 −η)
 ˆUt
Jt


+ η

"
Σr ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†

0

#

+ η(I −A∗A)
 ˆUt ˆV ⊤
t
−Σr
ˆUtK⊤
t
Jt ˆV ⊤
t
JtK⊤
t

  ˆVt
Kt

 
ˆV ⊤
t ˆVt + K⊤
t Kt
†
,

(39)

and

Vt+1 = Vt −η[A∗A(Ut+1V ⊤
t
−Σ)]⊤Ut+1(U ⊤
t+1Ut+1)†

= (1 −η)
 ˆVt
Kt


+ η

"
Σr ˆUt+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†

0

#

+ η(I −A∗A)
 ˆVt ˆU ⊤
t+1 −Σr
ˆVtJt+1 ˆUt+1
Kt ˆU ⊤
t+1
KtJ⊤
t+1

  ˆUt+1
Jt+1

 
ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†
.

(40)

15


---Page Break---
We denote

EUt
EJt


= (I −A∗A)
 ˆUt ˆV ⊤
t
−Σr
ˆUtK⊤
t
Jt ˆV ⊤
t
JtK⊤
t

  ˆVt
Kt

 
ˆV ⊤
t ˆVt + K⊤
t Kt
†
,
(41)

and

EVt
EKt


= (I −A∗A)
 ˆVt ˆU ⊤
t+1 −Σr
ˆVtJt+1 ˆUt+1
Kt ˆU ⊤
t+1
KtJ⊤
t+1

  ˆUt+1
Jt+1

 
ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†
.
(42)

Then it follows
ˆUt+1 = (1 −η) ˆUt + ηΣr ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†
+ ηEUt,
(43)

similarly

ˆVt+1 = (1 −η) ˆVt + ηΣr ˆU ⊤
t+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†
+ ηEVt,
(44)

and
Jt+1 = (1 −η)Jt + ηEJt,
Kt+1 = (1 −η)Kt + ηEKt.
(45)

Under the Assumption 1, there exits small δ1 ≪1 such that ∥EJt∥F ≤δ1∥Jt∥F and ∥EKt∥F ≤
δ1∥Kt∥F , as long as Cδ in Assumption 1 is large enough and cδ is small enough. Therefore we can
guarantee that
∥JtK⊤
t ∥F ≤(1 −η + δ1η)2t∥J0K⊤
0 ∥F .
(46)

Meanwhile, we have

Ut+1V ⊤
t
= (1 −η)
 ˆUt
Jt

  ˆV ⊤
t , K⊤
t

+ η

"
Σr ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†

0

#
 ˆV ⊤
t , K⊤
t


+ η

EUt
EJt

  ˆV ⊤
t , K⊤
t

,

(47)

which indicates that

ˆUt+1 ˆV ⊤
t+1 −Σr = (1 −η)2( ˆUt ˆV ⊤
t
−Σr) + η

ˆUt+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
† ˆU ⊤
t+1 −I

Σr

+ η ˆUt+1E⊤
Vt + (1 −η)ηΣr


ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
† ˆV ⊤
t
−I

+ (1 −η)ηEUt ˆV ⊤
t .

(48)

A.2
Proofs of the lemmas

A.2.1
Proof of Lemma 1

Proof. According to Eq. (9), we know that

ψ(Xt+ 1

2 ) = 1

2∥B(Xt+ 1

2 , Xt+ 1

2 ) −b∥2
2

= 1

2∥B(Xt, Xt) −ηB(Xt, ˆ∆(Xt)) −ηB( ˆ∆(Xt), Xt) + η2B( ˆ∆(Xt), ˆ∆(Xt)) −b∥2
2

= 1

2∥ηB( ˆ∆(Xt), Xt) −(B(Xt, Xt) −b)∥2
2

= 1

2∥B(Xt, Xt) −b∥2
2 + η2

2 ∥B( ˆ∆(Xt), Xt)∥2
2 −η⟨B( ˆ∆(Xt), Xt), B(Xt, Xt) −b⟩.

(49)

Note that
⟨B( ˆ∆(Xt), Xt), B(Xt, Xt) −b⟩= ∥ˆB(∆(Xt), Xt)∥2
F ,
(50)

and
∥B( ˆ∆(Xt), Xt)∥2
2 ≤ϱ∥ˆB( ˆ∆(Xt), Xt)∥2
F ,
(51)

16


---Page Break---
where ϱ = ∥A∥2
2 ≤(1 + δ) which is due to the RIP condition. Thus we have

ψ(Xt+ 1

2 ) ≤ψ(Xt) −2η −(1 + δ)η2

2
∥ˆB( ˆ∆(Xt), Xt)∥2
F .
(52)

Similarly, we can prove that

ψ(Xt+1) ≤ψ(Xt+ 1

2 ) −2η −(1 + δ)η2

2
∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F .
(53)

Therefore we finish our proof.

A.2.2
Proof of the Lemma 2.

Proof. Note that the optimal solution in Eq. (13) satisfies

I
0


P∆X⊤
t QQ⊤Xt −

I
0


A∗(B(Xt, Xt) −b)Q⊤Xt = 0, ∆∈C.
(54)

While the solution of Eq. (9) is

ˆ∆(Xt) =

I
0


A∗(B(Xt, Xt) −b)Q⊤Xt(X⊤
t QQ⊤X⊤
t )†,
(55)

which satisfies

I
0


P∆X⊤
t QQ⊤Xt −

I
0


A∗(B(Xt, Xt) −b)Q⊤Xt = 0.
(56)

It is easy to verify that the matrix ˜∆(Xt) =

P ˆ∆(Xt)
P ˆ∆(Xt)


satisfies Eq. (54), thus it is the solution of

the problem (13).

A.2.3
Proof of the Lemma 3.

Before proving the Lemma 3, we first define the angle between the column space of V U ⊤−Σ⊤and
that of V as

cosθv := ∥(UV ⊤−Σ)V V †∥F

∥UV ⊤−Σ∥F
,
(57)

and similarly

cosθu := ∥(V U ⊤−Σ)UU †∥F

∥UV ⊤−Σ∥F
,
(58)

where † stands for the pseudo-inverse. We note that the cosθu in Eq. (57) and Eq. (58) are well
defined and are equivalent to the following values.
Proposition 1. For any U, V ∈Rn×d and Σ ∈Rn×n, the cosθv and cosθu in Eq. (57) and Eq. (58)
have the following equivalent formulations

cosθv =
max
∥Y ∥F =1


UV ⊤−Σ, Y V V †

∥UV ⊤−Σ∥F ∥Y V V †∥F
,
(59)

and

cosθu =
max
∥Y ∥F =1


V U ⊤−Σ, Y UU †

∥UV ⊤−Σ∥F ∥Y UU †∥F
,
(60)

where Y ∈Rn×n.

Now we present the proof of Lemma 3.

Proof. Given the cosθv and cosθu , one can immediately writes

∥(UV ⊤−Σ)V V †∥F = cosθv∥UV ⊤−Σ∥F ,
(61)

and
∥(V U ⊤−Σ)UU †∥F = cosθu∥UV ⊤−Σ∥F ,
(62)

17


---Page Break---
which indicate that the cosθv and cosθu provide estimate to the √τ1 and √τ2 in Lemma 5. Together
with the Lemma 4, Assumption 1 and Definition 1, we can establish that

∥ˆB( ˆ∆(Xt), Xt)∥2
F ≥(1 −δ)∥(UtV ⊤
t
−Σ)VtV †
t ∥2
F
≥(1 −δ)cos2θv∥UtV ⊤
t
−Σ∥F

≥1 −δ

1 + δ cos2θt
v[ψ(Xt) −ψ∗].

(63)

Similarly, we have

∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F ≥1 −δ

1 + δ cos2θt+1
u
[ψ(Xt+ 1

2 ) −ψ∗].
(64)

According to Lemma 5, we know that cos2θt
v ≥τ1 and cos2θt
u ≥τ2, ∀t > 0 and max{τ1, τ2} > 0.
Thus we finish the proof.

A.2.4
Proof of the Lemma 4.

Proof. We have

∥A∗A(UV ⊤−M)Z∥F = ∥(UV ⊤−M)Z −(I −A∗A)(UV ⊤−M)Z∥F
≥∥(UV ⊤−M)Z∥F −∥(I −A∗A)(UV ⊤−M)Z∥F .
(65)

According to the definition of matrix norm and the Frobenius norm, there exits matrix Y with
∥Y ∥F = 1, such that

∥(I −A∗A)(UV ⊤−M)Z∥F =
max
∥Y ∥F =1


(I −A∗A)(UV ⊤−M)Z, Y


=
max
∥Y ∥F =1


(UV ⊤−M)Z, (I −A∗A)(Y )


≤
max
∥Y ∥F =1 ∥(UV ⊤−M)Z∥F ∥(I −A∗A)Y ∥F

≤δ∥(UV ⊤−M)Z∥F ,

(66)

therefore the following inequality holds

∥A∗A(UV ⊤−M)Z∥F ≥(1 −δ)∥(UV ⊤−M)Z∥F ,
(67)

which finishes our proof.

18


---Page Break---
A.2.5
Proof of the Lemma 5

Proof. Let Et = UtV ⊤
t
−Σ, we have

cosθt
v = ∥(UtV ⊤
t
−Σ)VtV †
t ∥F
∥Et∥F
= ∥EtVt(V ⊤
t Vt)−1

2 ∥F
∥Et∥F

=
max
∥Y ∥F =1

D
EtVt(V ⊤
t Vt)−1

2 , Y
E

∥Et∥F

=
max
∥Z(V ⊤
t Vt)
1
2 ∥F =1


Et, ZV ⊤
t


∥Et∥F
=
max
∥ZV ⊤
t ∥F =1


Et, ZV ⊤
t


∥Et∥F

=
max
∥ZV ⊤
t ∥F =1

 ˆUt ˆV ⊤
t
−Σr
ˆUtK⊤

Jt ˆV ⊤
t
JtK⊤
t


,

Z1 ˆV ⊤
t
Z1K⊤
t
Z2 ˆV ⊤
t
Z2K⊤
t



∥Et∥F

=
max
∥ZV ⊤
t ∥F =1

 ˆUt ˆV ⊤
t
−Σr
Jt ˆV ⊤
t


,

Z1 ˆV ⊤
t
Z2 ˆV ⊤
t


+
 ˆUtK⊤
t
JtK⊤
t


,

Z1K⊤
t
Z2K⊤
t



∥Et∥F

=
max
∥ZV ⊤
t ∥F =1

D ˆUt ˆV ⊤
t
−Σr

, Z1 ˆV ⊤
t
E

∥Et∥F
+


JtV ⊤
t , Z2V ⊤
t


∥Et∥F
+

D
ˆUtK⊤
t , Z1K⊤
t
E

∥Et∥F

≥
max
∥ZV ⊤
t ∥F =1

D ˆUt ˆV ⊤
t
−Σr

, Z1 ˆV ⊤
t
E
+

JtV ⊤
t , Z2V ⊤
t


∥Et∥F
−



D
ˆUtK⊤
t , Z1K⊤
t
E

∥Et∥F



≥
max
∥ZV ⊤
t ∥F =1

D ˆUt ˆV ⊤
t
−Σr

, Z1 ˆV ⊤
t
E
+

JtV ⊤
t , Z2V ⊤
t


∥Et∥F
−∥Z1K⊤
t ∥F
∥ˆUtK⊤
t ∥F
∥Et∥F

≥∥JtV ⊤
t ∥F
∥Et∥F
≥∥Vt∥F σmin(Jt)

∥Et∥F
.

(68)

Similarly, we have

cosθt
u ≥
max
∥ZU ⊤
t ∥F =1

D ˆVt ˆU ⊤
t −Σr

, Z1 ˆU ⊤
t
E
+

KtU ⊤
t , Z2U ⊤
t


∥Et∥F
−∥Z1J⊤
t ∥F
∥Jt ˆV ⊤
t ∥F
∥Et∥F

≥∥Ut∥F σmin(Kt)

∥Et∥F
.

(69)

To estimate the lower-bound of cosθt
u and cosθt
v, according to Eq. (68) and Eq. (69), we can divide the
whole iterations into two phases based on the value of σmin(Kt) and σmin(Jt). In the first phase, there
exists an iteration count T such that σmin(KT ) ≥ζ1 and σmin(JT ) ≥ζ1, for small ζ1 > 0. Since
A(·) satisfies the RIP condition with sufficiently small δ, we know that the EKt, EJt in Eq. (45) are
akin to small perturbations. Therefore the minimum singular value of Jt and Vt are decreasing under
the Assumption 2, and one can guarantee that σmin(Kt) ≥ζ1 and σmin(Jt) ≥ζ1, ∀t < T. Then the
value ζ1 provides lower-bound for the values of cosθt
u ≥τ1 =
dζ2
1
∥E0∥F and cosθt
v ≥τ2 =
dζ2
1
∥E0∥F , for
t < T. Meanwhile, according to the update in Eq. (45) and the sufficiently small RIP constant δ, we
know that σmin(Kt) and σmin(Jt) decrease linearly such that T = O(log 1

ζ1 ).

In the second phase, if σmin(Kt) < ζ1 and σmin(Jt) < ζ1, for t > T, we can use the following
inequalities to lower-bound the cosθt
u and cosθt
v.

cosθt
u ≥
max
∥ZU ⊤
t ∥F =1

D ˆVt ˆU ⊤
t −Σr

, Z1 ˆU ⊤
t
E
+

KtU ⊤
t , Z2U ⊤
t


∥Et∥F
−∥Z1J⊤
t ∥F
∥Jt ˆV ⊤
t ∥F
∥Et∥F
,
(70)

cosθt
v ≥
max
∥ZV ⊤
t ∥F =1

D ˆUt ˆV ⊤
t
−Σr

, Z1 ˆV ⊤
t
E
+

JtV ⊤
t , Z2V ⊤
t


∥Et∥F
−∥Z1K⊤
t ∥F
∥ˆUtK⊤
t ∥F
∥Et∥F
,
(71)

19


---Page Break---
which come from Eq. (69) and Eq. (68).

By Lemma 6, we know that σmin( ˆUt) > 0 and σmin( ˆVt) > 0, for t > T. Together with Eq. (70) and
Eq. (71) we have

cosθt
u ≥∥ˆVt ˆU ⊤
t −Σr∥F ∥Zu
1 ˆU ⊤
t ∥F + ∥KtU ⊤
t ∥F ∥Zu
2 U ⊤
t ∥F
∥Et∥F
−∥Zu
1 J⊤
t ∥F
∥Jt ˆV ⊤
t ∥F
∥Et∥F
,
(72)

where ∥Zu
1 ˆU ⊤
t ∥2
F + ∥Zu
1 Jt∥2
F + ∥Zu
2 U ⊤
t ∥2
F = 1.

cosθt
v ≥∥ˆUt ˆV ⊤
t
−Σr∥F ∥Zv
1 ˆV ⊤
t ∥F + ∥JtV ⊤
t ∥F ∥Zv
2V ⊤
t ∥F
∥Et∥F
−∥Zv
1K⊤
t ∥F
∥ˆUtK⊤
t ∥F
∥E∥F
,
(73)

where ∥Zv
1 ˆV ⊤
t ∥2
F + ∥Zv
2V ⊤∥2
F + ∥Zv
1K⊤
t ∥2
F = 1. The Eq. (72) and Eq. (73) further imply that

cosθt
u ≥
q

1 −∥ZuJt∥2
F

∥
 ˆVt ˆU ⊤
t −Σr
Kt ˆU ⊤
t


∥F

∥Et∥F
−∥ZuJt∥F ,
(74)

and

cosθt
v ≥
q

1 −∥ZvKt∥2
F

∥
 ˆUt ˆV ⊤
t
−Σr
Jt ˆV ⊤
t


∥F

∥Et∥F
−∥ZvKt∥F .
(75)

Meanwhile

∥Et∥2
F = ∥Jt ˆV ⊤
t ∥2
F + ∥ˆUtK⊤
t ∥2
F + ∥ˆUt ˆV ⊤
t
−Σr∥2
F + ∥JtK⊤
t ∥2
F

≤2

max{∥Jt ˆV ⊤
t ∥2
F , ∥ˆUtK⊤
t ∥2
F } + max{∥ˆUt ˆV ⊤
t
−Σr∥2
F , ∥JtK⊤
t ∥2
F }

.
(76)

According to Eq. (46) and Eq. (48), we know that ∥ˆUt ˆV ⊤
t −Σr∥2
F ≥∥JtK⊤
t ∥2
F , if ∥ˆU0 ˆV ⊤
0 −Σr∥2
F ≥
∥J0K⊤
0 ∥2
F which is true for random Gauss initialization with small value c0 in Assumption 2. Thus,

we have ∥Et∥2
F ≤2

max{∥Jt ˆV ⊤
t ∥2
F , ∥ˆUtK⊤
t ∥2
F } + ∥ˆUt ˆV ⊤
t
−Σr∥2
F

and

max












∥
 ˆUt ˆV ⊤
t
−Σr
Jt ˆV ⊤
t


∥F

∥Et∥F
,
∥
 ˆVt ˆU ⊤
t −Σr
Kt ˆU ⊤
t


∥F

∥Et∥F












≥
√

2/2.
(77)

Without loss of generality, we assume ∥Jt ˆV ⊤
t ∥F ≥∥Kt ˆU ⊤
t ∥F , then we consider Eq. (75), such that

cosθt
v ≥
q

1 −∥ZvKt∥2
F

√

2
2 −∥ZvKt∥F .
(78)

If ∥ZvKt∥F ≤1/3, then we can guarantee that cosθt
v ≥1/3.

Note that σ2
min( ˆVt)∥Zv∥2
F ≤∥Zv ˆV ⊤
t ∥2
F ≤1, then ∥Zv∥2
F ≤1/σ2
min( ˆV ). Moreover, according to
Eq. (45) and the RIP condition with sufficiently small δ, we have ∥Kt∥2
F ≤(1 −η + δ1η)2t∥K0∥2
F
for δ1 ≪1. In consequence,

∥ZvK⊤
t ∥2
F ≤(1 −η + δ1η)2t∥K0∥2
F /σ2
min( ˆVt).
(79)

According to Lemma 6, one can guarantee that ∥ZvK⊤
t ∥2
F ≤t2(1 −ˆδη)2tCv for constant Cv
and ˆδ = 1 −δ1. Meanwhile, we can select ζ1 in phase 1 to be small such that ∥ZvK⊤
t ∥2
F ≤
t2(1 −ˆδη)2tCv ≤1/3 for t > T, where T = O(log 1

ζ1 ). In consequence, we can guarantee that the
max{cosθt
v, cosθt
u} in phase 2 is lower-bounded by a constant τs ≥1/3. Together with phase 1, we
conclude that by setting τ 1
t , τ 2
t in Eq. (27) as cosθt
u and cosθt
v respectively, we have that the results
in Eq. (27) hold for max{τ 1
t , τ 2
t } ≥τ > 0.

20


---Page Break---
A.2.6
Proof of the Lemma 6

Lemma 6. Suppose that the Assumption 2 and Assumption 1 hold. Let ˆUt and ˆVt be updated
according to Eq. (43) and (44). Then, it follows that there exist constants ˜cu and ˜cv such that

σr( ˆUt) ≥˜cu/t,
σr( ˆVt) ≥˜cv/t.
(80)

Proof. Note that

ˆUt+1 = (1 −η) ˆUt + ηΣr ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†
+ ηEUt

= (1 −η) ˆUt + ηΣr

ˆUt ˆV ⊤
t
† ˆUt ˆV ⊤
t ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†
+ ηEUt,
(81)

and

ˆVt+1 = (1 −η) ˆVt + ηΣr ˆUt+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†
+ ηEVt

= (1 −η) ˆVt + ηΣr( ˆVt ˆU ⊤
t+1)† ˆVt ˆU ⊤
t+1 ˆUt+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
†
+ ηEVt.
(82)

Therefore we have the upper-bound of the operator norm of ˆUt+1 as

∥ˆUt+1∥2 ≤(1 −η)∥ˆUt∥2 + η ∥Σr

ˆUt ˆV ⊤
t
†
∥2
|
{z
}
τt

∥ˆUt∥2 ∥ˆV ⊤
t ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
†
∥2
|
{z
}
νt

+η∥EUt∥2

= (1 + η(τtνt −1)(1 + δ2)) ∥ˆUt∥2 ≤(1 + η(τt −1)(1 + δ2)) ∥ˆUt∥2

≤(1 + η|τt −1|(1 + δ2)) ∥ˆUt∥2,
(83)
where δ2 is due to the RIP condition in Assumption 1 and δ2 ≪1 is sufficiently small. Meanwhile

∥Σr

ˆUt ˆV ⊤
t
†
−I∥2 = ∥

Σr −ˆUt ˆV ⊤
t
 
ˆUt ˆV ⊤
t
†
∥2

≤∥Σr −ˆUt ˆV ⊤
t ∥2/σmin( ˆUt ˆV ⊤
t )

≤∥Σr −ˆUt ˆV ⊤
t ∥2/cξ

≤(1 −ζtη)t

cξ
∥Σr −ˆU0 ˆV ⊤
0 ∥2,

(84)

where cξ is due to Lemma (7). The last inequality is due to that as long as min{σmin(Kt), σmin(Jt} >
ζc
t > 0, one can always guarantee that ∥Σr −ˆUt ˆV ⊤
t ∥2 ≤(1 −ζtη)t∥Σr −ˆU0 ˆV ⊤
0 ∥2 for ζt > 0.
Therefore we have
|τt −1| = O(c−t
γ ),
(85)
with cγ ≥1. Together with Eq. (83) we obtain

∥ˆUt∥2 ≤(1 + η(1 + δ2)c−t
γ )∥ˆUt∥2 ≤

tY

i=1
(1 + η(1 + δ2)c−i
γ )∥ˆU0∥2 ≤cut∥ˆU0∥2
(86)

for constant cu. Similarly, one can guarantee that

∥ˆVt∥2 ≤cvt∥ˆV0∥2
(87)

for constant cv. In consequence, we obtain

σr( ˆUt) ≤∥ˆUt∥2 ≤cut∥ˆU0∥2, σr( ˆVt) ≤∥ˆVt∥2 ≤cvt∥ˆV0∥2.
(88)

According to Lemma 7, we know that the σr( ˆUt ˆV ⊤
t ) ≥cξ, together with Eq. (88) we can guarantee
that
σr( ˆUt) ≥
cξ
tcu∥ˆU0∥2
, σr( ˆVt) ≥
cξ
tcv∥ˆV0∥2
,
(89)

where we use the fact σr( ˆU ˆV ⊤) ≤∥ˆU∥2σr( ˆV ). Let ˜cu = cξ/

cu∥ˆU0∥2

and ˜cv = cξ/

cv∥ˆV0∥2

,
we thus finish the proof.

21


---Page Break---
A.2.7
Proof of the Lemma 7

Lemma 7. Let ˆUt and ˆV t be updated according to Eq. (43) and Eq. (44), respectively. Then, there
exist constants cξ such that

σr( ˆUt ˆV ⊤
t ) ≥cξ, ∀t > 0.
(90)

Proof. Note that

σr( ˆUt+1[ ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1]† ˆU ⊤
t+1
|
{z
}
Pt+1

Σr) ≥σr(Pt+1)σr(Σr),
(91)

and
σr(Σr ˆVt+1[ ˆV ⊤
t+1 ˆVt+1 + K⊤
t+1Kt+1]† ˆV ⊤
t+1
|
{z
}
Qt+1

) ≥σr(Qt+1)σr(Σr).
(92)

According to Eq. (43) and Eq. (44), we have

ˆUt+1 ˆV ⊤
t+1 = (1 −η)2 ˆUt ˆV ⊤
t
+ η ˆUt+1

ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1
† ˆU ⊤
t+1Σr

+ η ˆUt+1E⊤
Vt + (1 −η)ηΣr ˆVt

ˆV ⊤
t ˆVt + K⊤
t Kt
† ˆV ⊤
t
+ (1 −η)ηEUt ˆV ⊤
t

= (1 −η)2t ˆU0 ˆV ⊤
0 + η

t
X

i=0
(1 −η)2iPt+1−iΣr + η(1 −η)

t
X

i=0
(1 −η)2iΣrQt−i

|
{z
}
H

+ η

t
X

i=0
(1 −η)2i ˆUt+1−iE⊤
Vt−i + η(1 −η)

t
X

i=0
(1 −η)2iEUt−i ˆV ⊤
t−i
|
{z
}
Z

.

(93)

Meanwhile

σr(H) ≥η

t
X

i=0
(1 −η)2iσr(Pt+1−iΣr) + η(1 −η)

t
X

i=0
(1 −η)2iσr(ΣrQt−i).
(94)

The fact that σr(Pt+1) ≥σr(P0) and σr(Qt+1) ≥σr(Q0) gives to

σr(H) ≥η (σr(P0) + (1 −η)σr(Q0)) σr(Σr).
(95)

Moreover, according the RIP condition in Assumption 1, the following inequalities hold

∥EUt ˆV ⊤
t ∥2 ≤δ2∥ˆU0 ˆV ⊤
0 −Σr ˆV0( ˆV ⊤
0 ˆV0 + K⊤
0 K0)† ˆV ⊤
0 ∥2 = δ2cu
ν,
(96)

and
∥ˆUt+1E⊤
Vt∥2 ≤δ2∥ˆU0 ˆV ⊤
0 −ˆUt+1[ ˆU ⊤
t+1 ˆUt+1 + J⊤
t+1Jt+1]† ˆU ⊤
t+1Σr∥2 = δ2cv
ν
(97)

for sufficiently small δ2, and ∥Z∥2 ≤
1
2−ηδ2cv
ν + 1−η

2−ηδ2cu
ν.

Since ϱ = (1 −η)2t ≪1 and δ2 are very small, now we can lower bound the r-th singular value of
ˆUt+1 ˆV ⊤
t+1 by

σr( ˆUt+1 ˆV ⊤
t+1) ≥σr(H) −ϱσ1( ˆU0 ˆV ⊤
0 )

≥η (σr(P0) + (1 −η)σr(Q0)) σr(Σr) −ϱσ1( ˆU0 ˆV ⊤
0 ) −δ2(
1
2 −η cu
ν + 1 −η

2 −η cv
ν)

≥cξ,
(98)
which is a simple result of the matrix perturbation theory and cξ > 0 is a universal constant.

22


---Page Break---
A.3
Proofs of the theorems

A.3.1
Proof of the Theroem 1

Proof. Since M is rank one matrix, we have Us = 0 ∈Rn×d and Vs = 0 ∈Rn×d correspondingly

ˆU = Us + εNu = εNu, ˆV = Vs + εNv = εNv.
(99)

Note that

∥∇g ˆU∥2
F = ∥( ˆU ˆV ⊤−M) ˆU∥2
F
= ∥(UsV ⊤
s −M + εUsN ⊤
v + εNuV ⊤
s + ε2NuN ⊤
v
|
{z
}
εZ

)(Us + εNu)∥2
F

= ∥(UsU ⊤
s −M + εZ)εNu∥2
F
= ∥ε(UsU ⊤
s −M)N + ε2ZNu∥2
F
= o(ε)es + o(ε2).

(100)

Similarly we have ∥∇g ˆV ∥2
F = o(ε)es + o(ε2) and therefore

∥∇g∥2
F = ∥∇g ˆU∥2
F + ∥∇g ˆV ∥2
F = o(ε)es + o(ε2).

Meanwhile

∥∇g ˆU( ˆV ⊤ˆV )−1

2 ∥2
F = ∥( ˆU ˆV ⊤−M) ˆV ˆV †∥2
F
= ∥(UsU ⊤
s −M + εUsN ⊤+ εNU ⊤
s + ε2NN ⊤
|
{z
}
εZ

)NvN †
v∥2
F

= ∥MNvN †
v −ε2NN ⊤NvN †
v∥2
F
= ∥MNvN †
v −ε2NN ⊤∥2
F .

(101)

Let V ˆΣV⊤= NvN †
v be the SVD of NvN †
v, then ∥MNvN †
v∥2
F = σ2
∗∥V⊤V ∗∥2
F = es∥V⊤V ∗∥2
F
where M = σ∗U ∗V ∗⊤is the SVD of the rank-1 M and V is the orthogonal basis of a random Gauss
matrix Nv, then with high probability we have ∥V⊤V ∗∥2
F = Θ(1). As a result, we obtain

∥∇g ˆU( ˆV ⊤ˆV )−1

2 ∥2
F = Θ(1)es + o(ε2).
(102)

Likewise, we have
∥∇g ˆV ( ˆU ⊤ˆU)−1

2 ∥2
F = Θ(1)es + o(ε2).
(103)
As for the symmetric matrix sensing, we have

∥∇g ˆU( ˆU ⊤ˆU + λI)−1

2 ∥2
F = ∥( ˆU ˆU ⊤−M) ˆU( ˆU ⊤ˆU + λI)−1

2 ∥2
F
= ∥(ε2NN ⊤−M)εN(ε2N ⊤N + λI)−1

2 ∥2
F
= ∥(ε2NN ⊤−M)εΦΣN(ε2Σ2
N + λI)−1

2 Ψ⊤∥2
F
= ∥MεΦΣN(ε2Σ2
N + λI)−1

2 ΣNΨ⊤−ε3NN ⊤N(ε2N ⊤N + λI)−1

2 ∥2
F ,
(104)
where N is random Gauss matrices that follows standard normal distribution and N = ΦΣNΨ⊤is
the SVD of N. Note that

∥MεΦΣN(ε2Σ2
N + λI)−1

2 Ψ⊤∥2
F = σ2
∗∥εU ∗⊤ΦΣN(ε2Σ2
N + λI)−1

2 ∥2
F

= Θ

ε2

ε2 + λ/c


es,
(105)

where c > 0 is a bounded constant which is related to the singular value of the matrix N, thus we
have that

∥∇g ˆU( ˆU ⊤ˆU + λI)−1

2 ∥2
F = Θ

ε2

ε2 + λ/c


es + o(ε2).
(106)

Thus we finish our proof.

23


---Page Break---
A.3.2
Proof of the Theorem 2

Proof. According to Lemma 1, we know that

ψ(Xt+ 1

2 ) ≤ψ(Xt) −ℓ∥ˆB( ˆ∆(Xt), Xt)∥2
F ,

ψ(Xt+1) ≤ψ(Xt+ 1

2 ) −ℓ∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F ,
(107)

where ℓ= (2η −(1 + δ)η2)/2. Together with Lemma 3 which shows

∥ˆB( ˆ∆(Xt), Xt)∥2
F ≥1 −δ

1 + δ τ 1
t [ψ(Xt) −ψ∗],

∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F ≥1 −δ

1 + δ τ 2
t [ψ(Xt+ 1

2 ) −ψ∗].
(108)

We arrived at
ψ(Xt+ 1

2 ) −ψ∗≤ψ(Xt) −ψ∗−ℓ∥ˆB( ˆ∆(Xt), Xt)∥2
F

≤(1 −ℓτ 1
t
1 −δ
1 + δ )[ψ(Xt) −ψ∗],
(109)

and
ψ(Xt+1) −ψ∗≤ψ(Xt+ 1

2 ) −ψ∗−ℓ∥ˆB(Xt+ 1

2 , ˆ∆(Xt+ 1

2 ))∥2
F

≤(1 −ℓτ 2
t
1 −δ
1 + δ )[ψ(Xt+ 1

2 ) −ψ∗].
(110)

Thus we hve

ψ(Xt+1) −ψ∗≤(1 −ℓτ 2
t
1 −δ
1 + δ )(1 −ℓτ 1
t
1 −δ
1 + δ )[ψ(Xt) −ψ∗]

≤(1 −ℓτ 1 −δ

1 + δ )[ψ(Xt) −ψ∗],
(111)

where τ = max{τ 1
t , τ 2
t } > 0.

Meanwhile, if the RIP constant δ = 0, which means the number of the sampling m →∞and A(·)
becomes the identity operator, one can set η = 1. We know from the Eq. (33) and Eq. (34) that

ψ(Xt+1) −ψ∗= 1

2∥Ut+1V ⊤
t+1 −M∥2
F = 1

2∥M ⊤Ut+1U †
t+1 −M∥2
F .
(112)

At the same time
Ut+1V ⊤
t
= MVtV †
t .
(113)

Thus M ⊤Ut+1U †
t+1 = M, which indicates that

ψ(Xt+1) −ψ∗= 1

2∥Ut+1V ⊤
t+1 −M∥2
F ,

= 1

2∥M ⊤Ut+1U †
t+1 −M∥2
F ,

= 0[1

2∥M ⊤UtU †
t −M∥2
F ],

= 0[ψ(Xt) −ψ∗].

(114)

24


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: [TODO]

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

Justification: [TODO]

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

25


---Page Break---
Justification: [TODO]
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
Justification: [TODO]
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

26


---Page Break---
Answer: [Yes]
Justification: [TODO]
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
Justification: [TODO]
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
Justification: This paper is theoretical and it aims to prove the global Q-linear convergence
of the proposed AGN method for non-convex overparameterized low-rank matrix sensing.
All the details of the code for the experimental results in the figures are provided in the
supplementary materials.
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

27


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

Answer: [Yes]

Justification: [TODO]

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

Answer: [NA]

Justification: [TODO]

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

Justification: [TODO]

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

28


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

Justification: [TODO]

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

Justification: [TODO]

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

29


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: [TODO]
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
Justification: [TODO]
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
Justification: [TODO]
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

30


---Page Break---
