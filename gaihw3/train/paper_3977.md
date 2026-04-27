Near-Optimal Distributed Minimax Optimization
under the Second-Order Similarity

Qihao Zhou
School of Data Science, Fudan University
zhouqh20@fudan.edu.cn

Haishan Ye
School of Management, Xi’an Jiaotong University
SGIT AI Lab, State Grid Corporation of China
yehaishan@xjtu.edu.cn

Luo Luo∗
School of Data Science, Fudan University
Shanghai Key Laboratory for Contemporary Applied Mathematics
luoluo@fudan.edu.cn

Abstract

This paper considers the distributed convex-concave minimax optimization under
the second-order similarity. We propose stochastic variance-reduced optimistic
gradient sliding (SVOGS) method, which takes the advantage of the finite-sum
structure in the objective by involving mini-batch client sampling and variance
reduction. We prove SVOGS can achieve the ε-duality gap within communication
rounds of O(δD2/ε), communication complexity of O(n + √nδD2/ε), and local
gradient calls of ˜O(n + (√nδ + L)D2/ε log(1/ε)), where n is the number of
nodes, δ is the degree of the second-order similarity, L is the smoothness parameter,
and D is the diameter of the constraint set. We can verify that all of above
complexity (nearly) matches the corresponding lower bounds. For the specific
µ-strongly-convex-µ-strongly-convex case, our algorithm has the upper bounds
on communication rounds, communication complexity, and local gradient calls of
O(δ/µ log(1/ε)), O((n+√nδ/µ) log(1/ε)), and ˜O(n+(√nδ+L)/µ) log(1/ε))
respectively, which are also nearly tight. Furthermore, we conduct the numerical
experiments to show the empirical advantages of the proposed method.

1
Introduction

We study the distributed minimax optimization problem

min
x∈X max
y∈Y f(x, y) := 1

n

n
X

i=1
fi(x, y),
(1)

where fi is the differentiable local function associated with the i-th node, and X ⊆Rdx and Y ⊆Rdy
are the constraint sets. We are interested in the centralized setting, where there are one server node
and n −1 client nodes that collaboratively solve the minimax problem. Without loss of generality,
we assume the function f1 is located on the server node and the functions f2, . . . , fn are located on
the client nodes. This formulation is a cornerstone in the study of game theory, aiming to achieve
the Nash equilibrium [12, 44]. It covers a lot of applications such as signal processing [23], optimal
control [41], adversarial learning [44], robust regression [15, 35] and portfolio management [52].

∗The corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
We focus on the first-order optimization methods for solving convex-concave minimax problem.
The classical full-batch approaches including extra-gradient (EG) method [24], forward-backward-
forward (FBF) [51], optimistic gradient descent ascent (OGDA) [43], dual extrapolation [39] and so
forth [33, 34, 38] achieve the optimal first-order oracle complexity under the assumption of Lipschitz
continuous gradient [17, 42, 55]. For the objective with finite-sum structure, the stochastic variance
reduced methods [1, 10, 14, 30, 53] can reduce the cost of per iteration by using the inexact gradient
and lead to the better overall computational cost than full-batch methods. It is natural to design
the parallel iteration schemes by directly using above ideas to reduce the computational time in
distributed setting.

The communication complexity is a primary bottleneck in distributed optimization. The local
functions in machine learning applications typically exhibit homogeneity [3, 15, 18], which is helpful
to improve the communication efficiency. One common measure used to describe relationships among
local functions is the second-order similarity, e.g., the Hessian of each local function differs by a finite
quantity from the Hessian of global objective. Based on such characterization, several communication
efficient distributed optimization methods have been established [4, 5, 8, 19, 20, 25, 29, 46, 48, 50, 56].
The highlight of these methods is their communication complexity bounds mainly depend on the
degree of second-order similarity, which is potentially much tighter than the results depend on the
smoothness parameter [6, 7, 11, 13, 16, 21, 22, 25, 26, 28, 32, 36, 37, 47].

Recently, Khaled and Jin [20], Lin et al. [29] showed iterations with partial participation can further
reduce the communication complexity, improving the dependence on the number of nodes. They pro-
posed stochastic variance reduced proximal point methods for convex optimization, which allow only
one of clients to participate into the communication in the most of rounds. Additionally, Beznosikov
et al. [8] combined partial participation with forward-backward-forward based method, reducing
volume of communication complexity for minimax optimization. However, these methods [8, 20, 29]
increase the communication rounds, which result in more expensive time cost in communication than
the full participation strategies [5, 25]. In other words, the partial participation methods [8, 20, 29]
only reduce the overall volume of information exchanged among the nodes, while the advantage of
parallel communication enjoyed in full participation methods is damaged.

In this paper, we propose a novel distributed minimax optimization method, called stochastic variance-
reduced optimistic gradient sliding (SVOGS), which uses the mini-batch client sampling to balance
communication rounds, communication complexity, and computational complexity. We prove
SVOGS simultaneously achieves the (near) optimal communication complexity, communication
rounds, and local gradient calls for convex-concave minimax problem under the assumption of
second-order similarity. We also conduct numerical experiments to show the superiority of SVOGS.

2
Preliminaries

We focus on the distributed optimization in client-sever framework for solving minimax problem (1).
The notation fi presents the local function on the i-th node. We assume the function f1 is located on
the server and the other individuals are located on clients. We stack variables x ∈Rdx and y ∈Rdy
as the vector z = [x; y] ∈Rd, where d = dx + dy. We let Z := X × Y ⊆Rd and define the
projection operator PZ(v) := arg minz∈Z ∥z −v∥for given v ∈Rd. We also denote the vector
functions Fi : Rd →Rd and F : Rd →Rd as

Fi(z) :=

∇xfi(x, y)
−∇yfi(x, y)


and
F(z) := 1

n

n
X

i=1
Fi(z).

We consider the following common assumptions for our minimax problem.
Assumption 1. We suppose the constraint set Z ⊆Rd is a non-empty, closed, and convex.
Assumption 2. We suppose the constraint set Z ⊆Rd is bounded by diameter D > 0, i.e., we
have ∥z1 −z2∥≤D for all z1, z2 ∈Z.

Assumption 3. We suppose each local function fi : Rdx×Rdy →R is smooth, i.e., there exists L > 0
such that ∥Fi(z1) −Fi(z2)∥≤L ∥z1 −z2∥for all i ∈[n] and z1, z2 ∈Rd .

Assumption 4. We suppose each differentiable local function fi : Rdx ×Rdy →R is convex-concave,
i.e., we have fi(x, y) ≥fi(x′, y) + ⟨∇xfi(x′, y), x −x′⟩and fi(y) ≤fi(y′) + ⟨∇yfi(y′), y −y′⟩
for all i ∈[n], x, x′ ∈Rdx and y, y′ ∈Rdy.

2


---Page Break---
Assumption 5. We suppose the global objective f : Rdx × Rdy →R is strongly-convex-strongly-
concave, i.e., there exists µ > 0 such that the function f(x, y) −µ

2 ∥x∥2 + µ

2 ∥y∥2 is convex-concave.

Besides, we introduce the assumption of second-order similarity to measure the homogeneity in local
functions [5, 8, 19, 20].

Assumption 6. The local functions f1, . . . , fn : Rdx × Rdy →R are twice differentiable and hold
the δ-second-order similarity, i.e., there exists δ > 0 such that
∇2fi(x, y) −∇2f(x, y)
 ≤δ

for all i ∈[n], x ∈Rdx and y ∈Rdy.

We measure the sub-optimality of the approximate solution z = (x, y) ∈Z by duality gap, that is

Gap(x, y) := max
y′∈Y f(x, y′) −min
x′∈X f(x′, y).

We also consider the criterion of the gradient mapping for given z = (x, y) ∈Z [31, 40, 54], that is,

Fτ(z) = z −PZ(z −τF(z))

τ
,

where τ > 0. The gradient mapping Fτ(z) is a natural extension of gradient operator F(z).
Noticing that we have Fτ(z) = F(z) if the problem is unconstrained (i.e., Z = Rd), and the
condition Fτ(z) = 0 is equivalent to the point z is a solution of the problem. Compared with the
duality gap, the norm of gradient mapping is a more popular measure in empirical studied since it is
easy to achieve in practice.

For the specific strongly-convex-strongly-concave case, we can also measure the sub-optimality by
the square of Euclidean distance to the unique solution z∗= (x, y) ∈Z, that is

∥z −z∗∥2 = ∥x −x∗∥2 + ∥y −y∗∥2 .

Moreover, we use notations O(·), Θ(·) and Ω(·) to hide constants which do not depend on parameters
of the problem, and notations ˜O(·), ˜Θ(·), and ˜Ω(·) to additionally hide the logarithmic factors of n,
L, µ and δ.

3
Related Work

For convex-concave minimax optimization, the full batch first-order methods [24, 33, 34, 38, 39, 43,
51] can achieve ε-duality gap within at most O(LD2/ε) iterations. Applying these idea to distributed
setting naturally leads to the communication rounds of O(LD2/ε) and each round requires all of
the n nodes to compute and communicate their local gradient.

In a seminar work, Beznosikov et al. [5] proposed Star Min-Max Data Similarity (SMMDS) algorithm,
which additionally consider the second-orders similarity (Assumption 6) by involving gradient sliding
technique [27, 45]. The SMMDS requires communication rounds of O(δD2/ε), which benefits from
the homogeneity in local functions. Each round of this method needs to communicate/compute the
local gradient of all n nodes, and perform the local updates on the server within ˜O(L/δ log(1/ε))
local iterations, which results in the overall communication complexity of O(nδD2/ε) and local
gradient complexity of ˜O((nδ + L)D2/ε log(1/ε)). Later, Kovalev et al. [25] introduced extra-
gradient sliding (EGS), which further improves the local gradient complexity to O((nδ + L)D2/ε).
It is worth pointing out that the communication rounds of O(δD2/ε) achieved by SMMDS and EGS
matches the lower complexity bound under the second-order similarity assumption [5]. However,
these methods enforce all nodes to participate into communication in every round, which does not
sufficiently take the advantage of finite-sum structure in the objective.

Recently, Beznosikov et al. [8] proposed Three Pillars Algorithm with Partial Participation (TPAPP),
which uses the variance-reduced forward-backward-forward method [1, 10] to encourage only one of
clients participate into the communication in most of the rounds. The TPAPP can achieve point z ∈Rd
such that E[∥F(z)∥2] ≤ε for unconstrained case within the communication rounds of O(nδ2D2/ε),

3


---Page Break---
Table 1: The complexity of achieving E[Gap(x, y)] ≤ε in convex-concave case.

Methods
Communication Rounds
Communication Complexity
Local Gradient Complexity

EG [24]
O
  LD2

ε

O
  nLD2

ε

O
  nLD2

ε


SMMDS [5]
O
  δD2

ε

O
  nδD2

ε

˜
O
  (nδ+L)D2

ε
log 1

ε


EGS [25]
O
  δD2

ε

O
  nδD2

ε

O
  (nδ+L)D2

ε


SVOGS
(Algorithm 1)
O
  δD2

ε

O
 
n +
√nδD2

ε

˜
O
 
n + (√nδ+L)D2

ε
log 1

ε


Lower Bounds
(Theorem 3,4,5)
Ω
  δD2

ε

Ω
 
n +
√nδD2

ε

Ω
 
n + (√nδ+L)D2

ε


Table 2: The complexity of achieving E[∥x −x∗∥2+∥y −y∗∥2] ≤ε in the strongly-convex-strongly-
concave case. †These methods use permutation compressors [49], which require the assumption of
d > n. ♯The complexity of TPAPP depends on local iterations number H, where “TPAPP (a)” and
“TPAPP (b)” correspond to H=⌈L/(√nδ)⌉and H=⌈8 log(40nL/µ)⌉respectively.

Methods
Communication Rounds
Communication Complexity
Local Gradient Complexity

EG [24]
O
  L

µ log 1

ε

O
  nL

µ log 1

ε

O
  nL

µ log 1

ε


SMMDS [5]
O
  δ

µ log 1

ε

O
  nδ

µ log 1

ε

˜
O
  nδ+L

µ
log 1

ε


EGS [25]
O
  δ

µ log 1

ε

O
  nδ

µ log 1

ε

O
  nδ+L

µ
log 1

ε


OMASHA [4]†
O
  L

µ log 1

ε

O
  
n +
√nδ+L

µ

log 1

ε

O
  nL

µ log 1

ε


TPA [8]†
O
  
n +
√nδ

µ

log 1

ε

O
  
n +
√nδ

µ

log 1

ε

O
  
n +
√nL

δ
+ L

µ

log 1

ε


TPAPP (a) [8]♯
O
  
n +
√nδ

µ

log 1

ε

O
  
n +
√nδ

µ

log 1

ε

O
  
n +
√nL

δ
+ L

µ

log 1

ε


TPAPP (b) [8]♯
O
  
n +
√nδ+L

µ

log 1

ε

O
  
n +
√nδ+L

µ

log 1

ε

˜
O
  
n +
√nδ+L

µ

log 1

ε


SVOGS
(Algorithm 1)
O
  δ

µ log 1

ε

O
  
n +
√nδ

µ

log 1

ε

˜
O
  
n +
√nδ+L

µ

log 1

ε


Lower Bounds
([5, 8], Theorem 7)
Ω
  δ

µ log 1

ε

Ω
  
n +
√nδ

µ

log 1

ε

Ω
  
n +
√nδ+L

µ

log 1

ε


communication complexity of O(nδ2D2/ε), and local gradient complexity of O(n2δ4L2D6ε−3).2
The theoretical analysis of TPAPP for the constrained problem requires the objective being strongly-
convex-strongly-concave. In addition, we can also reduce the communication complexity by using
the permutation compressors [49] for high-dimensional problem [4, 8], which achieves the similar
complexity to existing partial participation methods [8].

We present the complexity of existing methods and compare them with our results in both general
convex-concave case and strongly-convex-strongly-concave case in Table 1-3.

4
Stochastic Variance-Reduced Optimistic Gradient Sliding

We propose stochastic variance-reduced optimistic gradient sliding (SVOGS) method in Algorithm 1.

The design of our algorithm starts from reformulating problem (1) as follows

min
x∈X max
y∈Y f(x, y) := 1

n

n
X

i=1
(fi(x, y) −f1(x, y))

|
{z
}
g(x,y):=f(x,y)−f1(x,y)

+f1(x, y).
(3)

The idea of gradient sliding [27] on minimax optimization can be viewed as iteratively solving the
surrogate of problem (3) within the quadratic approximation of g(x, y) [5, 8, 25]. Recall that the

2Although the complexity of TPAPP for achieving E[∥F(z)∥]2 ≤ε is established for the unconstrained
case, its analysis additionally assume that the sequences generated by the algorithm are bounded by D > 0 [8,
Theorem 5.12].

4


---Page Break---
Table 3: The complexity of achieving E[∥Fτ(x, y)∥2] ≤ε in convex-concave case. §The TPAPP
additionally assumes Z = Rd and the sequence generated by the algorithm is bounded by D > 0.

Methods
Communication Rounds
Communication Complexity
Local Gradient Complexity

TPAPP [8]§
O
  nδ2D2

ε

O
  nδ2D2

ε

O
  n2δ4L2D6

ε3


SVOGS
(Algorithm 1)
˜O
  δD
√ε log 1

ε

˜O
  
n+
√nδD
√ε

log 1

ε

˜O
  
n+ (√nδ+L)D
√ε

log 1

ε


Algorithm 1 Stochastic Variance-Reduced Optimistic Gradient Sliding (SVOGS)

1: Input: initial point z0 = (x0, y0) ∈Z, step size η, accuracy {εk}K
k=1, communication rounds K,
mini-batch size b, probability p ∈(0, 1], weights α, γ ∈(0, 1);

2: Initialization: w−1 = z−1 = w0 = z0 = (x0, y0) ∈Z, z0
i = z0 for all i ∈[n];

3: for k = 0, 1, 2, . . ., K −1 do

4:
¯zk = (1 −γ)zk + γwk;

5:
Sample Sk = {jk
1 , . . . , jk
b } uniformly and independently from [n];

6:
δk = F(wk−1) −F1(wk−1) + 1

b

X

j∈Sk

 
Fj(zk) −F1(zk) −Fj(wk−1) + F1(wk−1)


+α

b

X

j∈Sk

 
Fj(zk) −F1(zk) −Fj(zk−1) + F1(zk−1)

;

7:
vk = ¯zk −ηδk;

8:
Find uk ∈Rd such that ∥uk −ˆuk∥2 ≤εk, where ˆuk is the solution of the problem

min
ˆx∈X max
ˆy∈Y


f1(ˆx, ˆy) + 1

2η ∥ˆx −vk
x∥2 −1

2η ∥ˆy −vk
y∥2

;
(2)

9:
zk+1 = uk;

10:
wk+1 =
zk+1
with probability p,
wk
with probability 1 −p.

11: end for

optimistic gradient descent ascent (OGDA) method [34, 43] iterates with

zk+1 = PZ
 
zk −η(F(zk) + F(zk) −F(zk−1)
|
{z
}
optimistic gradient

)

,
(4)

where η > 0 is the step size. It is well-known that OGDA achieves optimal convergence rate under the
first-order smoothness assumption [42, 55], which motivated us construct the quadratic approximation
of g(x, y) by using the optimistic gradient of g at (xk, yk) in the linear terms, that is

g(x, y) ≈ˆg(x, y)

=g(xk, yk) + ⟨∇xg(xk, yk) + ∇xg(xk, yk) −∇xg(xk−1, yk−1)
|
{z
}
optimistic gradient with respect to x

, x −xk⟩+ 1

2η

x −xk2

+ ⟨∇yg(xk, yk) + ∇yg(xk, yk) −∇yg(xk−1, yk−1)
|
{z
}
optimistic gradient with respect to y

, y −yk⟩−1

2η

y −yk2 .

(5)

Applying approximation (5) to formulation (3), we obtain the optimistic gradient sliding (OGS),
which iteratively solve the sub-problem

(xk+1, yk+1) ≈arg min
ˆx∈X max
ˆy∈Y ˆg(ˆx, ˆy) + f1(ˆx, ˆy).
(6)

5


---Page Break---
We can verify function g(x, y) is δ-smooth under Assumption 6, which indicates taking η = Θ(1/δ)
and solving the sub-problem sufficiently accurate can find an ε-suboptimal solution within the
iteration numbers of O(δD2/ε) and O(δ/µ log(1/ε)) for the convex-concave case and the strongly-
convex-strongly-concave case respectively (see Section 5). The dependence on δ implies OGS
benefits from the second-order similarity in local functions, while each of its iteration requires the
communication and the computation of the exact gradient of f(x, y) within the complexity of O(n).

The key idea to improve the cost in each iteration is involving the mini-batch client sampling and
variance reduction with momentum [2, 26]. Specifically, we estimate the optimistic gradient in
formulation (5) as follows

G(zk) + G(zk) −G(zk−1) ≈
1
|Sk|

X

j∈Sk

 
G(wk−1) + Gj(zk) −Gj(wk−1) + α(Gj(zk) −Gj(zk−1))
|
{z
}
momentum term


,

(7)

where G(z) := F(z) −F1(z), Sk ⊆[n] is the random index set, wk is the snapshot point which
is updated infrequently in iterations, and α ∈(0, 1) is the momentum parameter. Applying the
optimistic gradient estimation (7) to formulations (5)-(6), we achieve our stochastic variance-reduced
optimistic gradient sliding (SVOGS) method (Algorithm 1).

The proposed SVOGS enjoys the mini-batch partial participation in the steps communication and
computation in most of rounds, which is the main difference between SVOGS and existing methods
[5, 8, 26]. Concretely, taking the mini-batch size |Sk| = Θ(√n ) for SVOGS can simultaneously
balance communication rounds, communication complexity and local gradient complexity. The
SVOGS keeps both the benefit of parallel communication like full participation methods (i.e.,
SMMDS [5] and EGS [25]) and the low communication cost like existing participation methods
(i.e., TPAPP [8]). Additionally, the communication advantage of SVOGS also makes the algorithm
achieves better local gradient complexity than state-of-the-arts [5, 8, 26].

5
The Complexity Analysis

In this section, we provide the complexity analysis of proposed SVOGS (Algorithm 1) to show its
superiority. In particular, we let µ = 0 for the convex-concave case to the ease of presentation.

We analyze the convergence of SVOGS (Algorithm 1) by establishing the Lyapunov function

Φk :=
1 −γ

η
+ µ

2


∥zk −z∗∥2 + 2⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩

+
1
64η ∥zk −zk−1∥2 + γ

4η ∥wk−1 −zk∥2 + (2γ + ηµ)

2pη
∥wk −z∗∥2,
(8)

where we take weight γ ≤1/8 and the step size η ≤1/(32δ) which always guarantees Φk ≥0 by
using Young’s inequality and the similarity assumption (see detailed proof in Appendix B).

We show that the decrease of Lyapunov function in expectation as follows.
Lemma 1. Suppose Assumptions 1, 3, 4, and 6 hold with 0 ≤µ ≤δ ≤L, running SVOGS (Algo-
rithm 1) with η ≤min{1/µ, 1/(32δ)}, α = max {1 −ηµ/(6(1 −γ)), 1 −pηµ/(2γ + ηµ)}, γ ≤
1/8, 256η2δ2α2(b + 1)/b ≤α, 4ηδ2/b ≤αγ/(4η), and εk ≤c−1 min

∥ˆuk −zk∥, ∥ˆuk −zk∥2	

for some c = poly(µ, δ), then we have

E[Φk+1] ≤max

1−
ηµ
6(1 −γ), 1−
pηµ
2γ + ηµ


E[Φk]−1

16η E

∥zk −ˆuk∥2
−γ

2η E

∥wk −ˆuk∥2
.

(9)

5.1
The Convex-Concave Case

For the convex-concave case, we use Jensen’s inequality and the convexity (concavity) to bound the
duality gap at uK
avg = 1

K
PK−1
k=0 uk as follows

Gap(uK
avg) ≤
max
(x′,y′)∈Z
1
K

K−1
X

k=0

 
f(uk
x, y′) −f(x′, uk
y)

≤max
z∈Z
1
K

K−1
X

k=0
⟨F(uk), uk −z⟩.
(10)

6


---Page Break---
Applying Lemma 1 by summing over inequality (9), we can bound the right-hand side of (10) via the
terms of PK−1
k=0 E

∥zk −ˆuk∥2
and PK−1
k=0 E

∥wk −ˆuk∥2
, and achieve the following theorem.
Theorem 1. Suppose Assumptions 1, 2, 3, 4 and 6 hold with 0 < δ ≤L and D > 0, we
run Algorithm 1 with b = ⌈√n ⌉, γ = p = 1/(√n + 8), η = min
√γb/(4δ), 1/(32δ)
	
,
α = 1, and εk = min

ζ, ˆc−1 min

∥ˆuk −zk∥, ∥ˆuk −zk∥2		
for some ζ = poly(L, δ, n, D, ε) and
ˆc = poly(δ). Then we have

E

"

max
z∈Z
1
K

K−1
X

k=0
⟨F(uk), uk −z⟩

#

≤10D2

ηK
+ ε

2,
where uK
avg = 1

K

K−1
X

k=0
uk.

Theorem 1 shows we can run SVOGS with step size η = Θ(1/δ) and communication rounds
of K = O(δD/ε) to achieve the ε-sub-optimality in expectation. Additionally, each communication
round contains the expected communication complexity of b(1 −p) + np = O(√n ), leading to the
overall communication complexity of O(n + √nδD2/ε).

The sub-problem (2) in SVOGS (line 8 of Algorithm 1) is a minimax problem with (L+1/η)-smooth
and (1/η)-strongly-convex-(1/η)-strongly-concave objective. Therefore, the setting of εk and η in
the theorem indicates the condition ∥uk −ˆuk∥2 ≤εk can be achieved by the local iterations number
of O((L + δ)/δ log(εk)) = ˜O(L/δ log(1/ε)) on the server (e.g., use EG [24]). Additionally, each
round of SVOGS contains the expected local gradient complexity of b(1 −p) + np = O(√n ) to
achieve the (mini-batch) optimistic gradient δk. Hence, the overall local gradient complexity of
SVOGS is ˜O(K(√n + L/δ log(1/ε))) = ˜O(n + (√nδ + L)D2/ε log(1/ε)). We formally present
the upper complexity bounds of SVOGS for the convex-concave case as follows.
Corollary 1. Following the setting of Theorem 1, we can achieve E[Gap(uK
avg)] ≤ε within commu-
nication rounds of O(δD2/ε), communication complexity of O(n + √nδD2/ε), and local gradient
complexity of ˜O(n + (√nδ + L)D2/ε log(1/ε)), where uK
avg = 1

K
PK−1
k=0 uk.

5.2
The Strongly-Convex-Strongly-Concave Case

By appropriate settings of SVOGS, Lemma 1 leads to the following linear convergence of our
Lyapunov function in the strongly-convex-strongly-concave case.

Theorem 2. Suppose Assumptions 1, 3, 4, 5 and 6 hold with 0 < µ ≤δ ≤L, we run Algorithm 1 with
b = ⌈min {√n, δ/µ}⌉, γ = p = 1/(min {√n, δ/µ} + 8), η = min
√αγb/(4δ), 1/(32δ)
	
, α =
max {1 −ηµ/(6(1 −γ)), 1 −pηµ/(2γ + ηµ)}, and εk = c−1 min

∥ˆuk −zk∥, ∥ˆuk −zk∥2	
for
some c = poly(µ, δ). Then we have

E[ΦK] ≤max

1 −
ηµ
6(1 −γ), 1 −
pηµ
2γ + ηµ

K
Φ0.

We then apply Theorem 2 with K = O(δ/µ log(1/ε)) and analyze the complexity like the discussion
after Theorem 1, which results in the upper complexity bounds as follows.
Corollary 2. Following the setting of Theorem 2, we can achieve E

∥zK −z∗∥2
≤ε within
communication rounds of O(δ/µ log(1/ε)), communication complexity of O((n+√nδ/µ) log(1/ε)),
and local gradient complexity of ˜O((n + (√nδ + L)/µ) log(1/ε)).

5.3
Making the Gradient Mapping Small

For the convex-concave case (under the assumptions of Theorem 1), we can achieve the points with
small gradient mapping by solving the regularized problem

min
x∈X max
y∈Y
ˆf(x, y) := f(x, y) + λ

2

x −x02 −λ

2

y −y02
(11)

for some λ > 0. Noticing that the function ˆf(x, y) is (L + λ)-smooth, λ-strongly-convex-λ-
strongly-concave and δ-similarity. Then Corollary 2 implies running SVOGS (Algorithm 1) by
iterations number K = O(δD/√ε log(L/ε)) to solve problem (11) with λ = O(√ε/D) can

7


---Page Break---
achieve E[∥Fτ(zK)∥2] ≤ε, which results in the complexity shown in Table 3. For the strongly-
convex-strongly-concave case, the complexity of achieving E[∥Fτ(zK)∥2
≤ε nearly matches the
complexity of achieving E

∥zK −z∗∥2] ≤ε. We defer the detailed derivation for these results of
making the gradient mapping small to Appendix G.

6
The Optimality of SVOGS

In this section, we provide the lower complexity bounds for solving our minimax problems by using
distributed first-order oracle (DFO) methods. The class of algorithms considered in our analysis
follows the definition of Beznosikov et al. [8], which is formally described in Appendix D. Compared
with existing lower bound analysis for second-order similarity only focusing on communication [5, 8],
we additionally study the computation complexity by considering the local gradient calls. The results
in this section imply the complexity of proposed SVOGS (nearly) matches the lower bounds on the
communication rounds, the communication complexity and the local gradient calls simultaneously.

6.1
The Lower Bounds for Convex-Concave Case

We first provide the following lower bounds for convex-concave case.

Theorem 3. For any 0 < δ ≤L, n ≥3, D > 0 and ε ≤δD2/(12
√

2 ), there exist L-smooth and
convex-concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity, and closed convex
set Z = X × Y with diameter D. In order to find an approximate solution z = (x, y) of problem (1)
such that E[Gap(z)] ≤ε, any DFO algorithm needs at least Ω(δD2/ε) communication rounds.

Theorem 4. For any 0 < δ ≤L, n ≥2, D > 0 and ε ≤δD2/(16
√

2n ), there exist L-smooth and
convex-concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity, and closed convex
set Z = X × Y with diameter D. In order to find an approximate solution z = (x, y) of problem (1)
such that E[Gap(z)] ≤ε, any DFO algorithm needs at least Ω(n + √nδD2/ε) communication
complexity and Ω(n + √nδD2/ε) local gradient calls.

The lower bounds on communication round and communication complexity shown in Theorem 3 and 4
match the corresponding upper bounds of SVOGS shown in Corollary 1. However, the lower bound
on local gradient complexity shown in Theorem 4 only nearly matches the result of Corollary 1 in
the case of √nδ ≥Ω(L). Therefore, we also provide the following lower bound on local gradient
complexity to show the tightness of dependence on the smoothness parameter L.

Lemma 2. For any L > 0, n ∈N, D > 0 and ε ≤δD2/(4
√

2), there exist L-smooth and
convex-concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity, and closed convex
set Z = X × Y with diameter D. In order to find an approximate solution z = (x, y) of problem (1)
such that E[Gap(z)] ≤ε, any DFO algorithm needs at least Ω(n + LD2/ε) local gradient calls.

Combining the results of Theorem 4 and Lemma 2, we achieve the following lower bound on local
gradient complexity, which nearly matches the corresponding upper bound shown in Corollary 1.

Theorem 5. For any 0 < δ ≤L, n ≥2, D > 0 and ε ≤δD2/(16
√

2n), there exist L-smooth and
convex-concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity, and closed convex
set Z = X × Y with diameter D. In order to find an approximate solution z = (x, y) of problem (1)
such that E[Gap(z)] ≤ε, any DFO algorithm needs at least Ω(n + (√nδ + L)D2/ε) local gradient
calls.

The constructions in our lower bound analysis is based on the modifications on the blinear functions
provided by Han et al. [14], which are originally used to analyze the minimax optimization in non-
distributed setting. We provide detailed proofs in Appendix E. In related work, Beznosikov et al. [5]
also provide the lower bound of Ω(δD2/ε) (matching the result of Theorem 3) for communication
rounds by using the regularized function, which is different from our construction in the proof of
Theorem 3. In addition, our lower bounds on the communication complexity and the local gradient
complexity shown in Theorem 4 and 5 are new.

8


---Page Break---
0
50
100
150
200
Communication Rounds

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
2500 5000 7500 1000012500

Communication Complexity

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
20000
40000
60000
80000
Local Gradient Calls

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 1: Results for convex-concave minimax problem (12) on a9a.

0
10
20
30
40
50
Communication Rounds

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
10000
20000
30000
Communication Complexity

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
25000 50000 75000 100000

Local Gradient Calls

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 2: Results for strongly-convex-strongly-concave minimax problem (13) on a9a.

6.2
The Lower Bounds for Strongly-Convex-Strongly-Concave Case

The tight lower bound on communication rounds in strongly-convex-strongly-concave case has been
provided by Beznosikov et al. [5, Theorem 1]. We present the result as follows.
Theorem 6 ([5]). For any µ, δ, L > 0 with L ≥max{µ, δ} and n ≥3, there exist L-smooth and
convex-concave functions f1, . . . , fn : Rdx×Rdy with δ-second-order similarity such that the function
f(x, y) = 1

n
Pn
i=1 fi(x, y) is µ-strongly-convex-µ-strongly-concave. In order to find a solution of
problem (1) such that E[∥z −z∗∥2] ≤ε, any DFO algorithm needs at least Ω(δ/µ log(1/ε))
communication rounds.

The tight lower bound on communication complexity has been provided by Beznosikov et al. [8]. We
follow their construction to establish the lower bound on local gradient complexity, nearly matching
the corresponding upper bound of our SVOGS. We formally present these lower bounds as follows.
Theorem 7. For any µ, δ, L > 0 with L ≥max{µ, δ} and n ≥2, there exist L-smooth and convex-
concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity such that the function
f(x, y) = 1

n
Pn
i=1 fi(x, y) is µ-strongly-convex-µ-strongly-concave. In order to find a solution of
problem (1) such that E[∥z−z∗∥2] ≤ε, any DFO algorithm needs at least Ω((n+√nδ/µ) log(1/ε))
communication complexity and Ω((n + (√nδ + L)/µ) log(1/ε)) local gradient calls.

7
Experiments

We conduct the experiment on robust linear regression [5, 15, 35]. Concretely, we consider the
constrained convex-concave minimax problem

min
∥x∥1≤Rx max
∥y∥≤Ry
1
2N

N
X

i=1

 
x⊤(ai + y) −bi
2 ,
(12)

and the unconstrained strongly-convex-strongly-concave minimax problem

min
x∈Rd′ max
y∈Rd′
1
2N

N
X

i=1

 
x⊤(ai + y) −bi
2 + λ

2 ∥x∥2 −β

2 ∥y∥2,
(13)

where x contains the weights of the model, y describes the noise, and {(ai, bi)}N
i=1 is the training set.

9


---Page Break---
0
20
40
60
80
100
Communication Rounds

10
14

10
12

10
10

10
8

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
5000
10000
15000
20000
Communication Complexity

10
14

10
12

10
10

10
8

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
20000
40000
60000
80000 100000
Gradient Calls

10
14

10
12

10
10

10
8

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 3: Results for convex-concave minimax problem (12) on w8a.

0
10
20
30
40
50
Communication Rounds

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
5000
10000
15000
20000
25000
Communication Complexity

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
20000
40000
60000
80000 100000
Gradient Calls

10
8

10
6

10
4

10
2

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 4: Results for strongly-convex-strongly-concave minimax problem (13) on w8a.

We compare the proposed SVOGS (Algorithm 1) with baselines Extra-Gradient method (EG) [24],
Star Min-Max Data Similarity algorithm (SMMDS) [5], Extra-Gradient Sliding (EGS) [25]), and
Three Pillars Algorithm with Partial Participation (TPAPP) [8]. We test the algorithms on real-
world datasets “a9a” (N = 32, 561, d′ = 123), “w8a” (N = 49, 749, d′ = 300) and “covtype”
(N = 581, 012, d′ = 54) from LIBSVM repository [9] and set the nodes number be n = 500. For
problem (12), we set Rx = 2 and Ry = 0.05, respectively.

We implement all of the methods by Python 3.9 with NumPy and run on a machine with AMD
Ryzen(TM) 7 4800H 8 core with Radeon Graphics 2.90 GHz CPU with 16GB RAM. We solve the
sub-problem in SVOGS (Algorithm 1), SMMDS [5], EGS [25], and TPAPP [8] by Extra-Gradient
method of Korpelevich [24]. We tune the step-size η of SVOGS from {0.01, 0.1, 1}. The probability
p is tuned from {p0, 5p0, 10p0}, where p0 = 1/ min{√n+δ/µ}. The batch size b is determined from
{⌊b0/10⌋, ⌊b0/5⌋, ⌊b0⌋}, with b0 = 1/p0. We set the other parameters by following our theoretical
analysis. We set the average weight as γ = 1 −p. For the momentum parameter, we set α = 1 for
convex-concave case and α = max{1 −ηµ/(6(1 −γ)), 1 −pηµ/(2γ + ηµ)} for strongly-convex-
strongly-concave case, where we estimate µ by max{λ, β} for problem (13). For the sub-problem
solver, we set its step-size according to the smoothness parameter of sub-problem, i.e., 1/(L + 1/η).
In addition, we estimate the smooth parameter L and the similarity parameter δ by following the
strategy in Appendix C of Beznosikov et al. [5].

We present the experimental results in Figure 1 to 4 for datasets “a9a” and “w8a”. The results for
dataset “covtype” is displayed in Appendix H due to the space limitation. We can observe that
our SVOGS outperforms all baselines in terms of the local gradient complexity. Additionally, the
SVOGS requires less communication rounds than classical EG and existing partial participation
method TPAPP, and it requires significantly less communication complexity than full participation
methods EG, SMMDS and EGS. All of these empirical results support our theoretical analysis.

8
Conclusion

This paper presents a novel distributed optimization method named SVOGS, which use the second-
order similarity in local functions and the finite-sum structure in objective to solve the convex-concave
minimax problem within the near-optimal complexity. Our theoretical results are also validated by
the numerical experiments. In future work, it is interesting to use our ideas to improve the efficiency
of distributed nonconvex minimax optimization under the second-order similarity.

10


---Page Break---
Acknowledgments and Disclosure of Funding

Luo and Zhou is supported by the Major Key Project of Pengcheng Laboratory (No. PCL2024A06),
National Natural Science Foundation of China (No.
62206058), Shanghai Sailing Program
(22YF1402900), and Shanghai Basic Research Program (23JC1401000). Ye is supported in part by
the National Natural Science Foundation of China under Grant 12101491 and in part by the National
Key Research and Development Project of China under Grant 2022YFA1004002.

References

[1] Ahmet Alacaoglu and Yura Malitsky. Stochastic variance reduction for variational inequality
methods. In Conference on Learning Theory, 2022.

[2] Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods.
Journal of Machine Learning Research, 18(221):1–51, 2018.

[3] Yossi Arjevani and Ohad Shamir. Communication complexity of distributed convex learning
and optimization. Advances in Neural Information Processing Systems, 2015.

[4] Aleksandr Beznosikov and Alexander Gasnikov. Compression and data similarity: Combination
of two techniques for communication-efficient solving of distributed variational inequalities. In
International Conference on Optimization and Applications, 2022.

[5] Aleksandr Beznosikov, Gesualdo Scutari, Alexander Rogozin, and Alexander Gasnikov. Dis-
tributed saddle-point problems under data similarity. Advances in Neural Information Processing
Systems, 2021.

[6] Aleksandr Beznosikov, Pavel Dvurechenskii, Anastasiia Koloskova, Valentin Samokhin, Se-
bastian U Stich, and Alexander Gasnikov. Decentralized local stochastic extra-gradient for
variational inequalities. Advances in Neural Information Processing Systems, 2022.

[7] Aleksandr Beznosikov, Peter Richtárik, Michael Diskin, Max Ryabinin, and Alexander Gas-
nikov. Distributed methods with compressed communication for solving variational inequalities,
with theoretical guarantees. Advances in Neural Information Processing Systems, 2022.

[8] Aleksandr Beznosikov, Martin Takác, and Alexander Gasnikov. Similarity, compression and
local steps: three pillars of efficient communications for distributed variational inequalities.
Advances in Neural Information Processing Systems, 2023.

[9] Chih-Chung Chang and Chih-Jen Lin. LIBSVM: a library for support vector machines. ACM
transactions on intelligent systems and technology (TIST), 2(3):1–27, 2011.

[10] Tatjana Chavdarova, Gauthier Gidel, François Fleuret, and Simon Lacoste-Julien. Reducing
noise in GAN training with variance reduced extragradient. Advances in Neural Information
Processing Systems, 2019.

[11] Yuyang Deng and Mehrdad Mahdavi. Local stochastic gradient descent ascent: Convergence
analysis and communication efficiency. In International Conference on Artificial Intelligence
and Statistics, 2021.

[12] Bahman Gharesifard and Jorge Cortés. Distributed continuous-time convex optimization on
weight-balanced digraphs. IEEE Transactions on Automatic Control, 59(3):781–786, 2014.

[13] Eduard Gorbunov, Filip Hanzely, and Peter Richtárik. Local SGD: Unified theory and new
efficient methods. In International Conference on Artificial Intelligence and Statistics, 2021.

[14] Yuze Han, Guangzeng Xie, and Zhihua Zhang.
Lower complexity bounds of finite-sum
optimization problems: The results and construction. Journal of Machine Learning Research,
25(2):1–86, 2024.

[15] Hadrien Hendrikx, Lin Xiao, Sebastien Bubeck, Francis Bach, and Laurent Massoulie. Statisti-
cally preconditioned accelerated gradient method for distributed optimization. In International
conference on machine learning, 2020.

11


---Page Break---
[16] Charlie Hou, Kiran K. Thekumparampil, Giulia Fanti, and Sewoong Oh. Efficient algorithms
for federated saddle point optimization. arXiv preprint:2102.06333, 2021.

[17] Adam Ibrahim, Waıss Azizian, Gauthier Gidel, and Ioannis Mitliagkas. Linear lower bounds
and conditioning of differentiable games. In International conference on machine learning,
2020.

[18] Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Ar-
jun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings,
et al. Advances and open problems in federated learning. Foundations and trends® in machine
learning, 14(1–2):1–210, 2021.

[19] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for federated learning.
In International conference on machine learning, 2020.

[20] Ahmed Khaled and Chi Jin. Faster federated optimization under second-order similarity. In
International Conference on Learning Representations, 2023.

[21] Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First analysis of local GD on
heterogeneous data. arXiv preprint:1909.04715, 2019.

[22] Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for local SGD on
identical and heterogeneous data. In International Conference on Artificial Intelligence and
Statistics, 2020.

[23] Seung-Jean Kim and Stephen Boyd. A minimax theorem with applications to machine learning,
signal processing, and finance. In IEEE Conference on Decision and Control, 2007.

[24] Galina M. Korpelevich. The extragradient method for finding saddle points and other problems.
Matecon, 12:747–756, 1976.

[25] Dmitry Kovalev, Aleksandr Beznosikov, Ekaterina Borodich, Alexander Gasnikov, and Gesualdo
Scutari. Optimal gradient sliding and its application to optimal distributed optimization under
similarity. Advances in Neural Information Processing Systems, 2022.

[26] Dmitry Kovalev, Aleksandr Beznosikov, Abdurakhmon Sadiev, Michael Persiianov, Peter
Richtárik, and Alexander Gasnikov. Optimal algorithms for decentralized stochastic variational
inequalities. Advances in Neural Information Processing Systems, 2022.

[27] Guanghui Lan. Gradient sliding for composite optimization. Mathematical Programming, 159:
201–235, 2016.

[28] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence
of FedAvg on non-iid data. International Conference on Learning Representations, 2019.

[29] Dachao Lin, Yuze Han, Haishan Ye, and Zhihua Zhang. Stochastic distributed optimization un-
der average second-order similarity: Algorithms and analysis. Advances in Neural Information
Processing Systems, 36, 2024.

[30] Luo Luo, Cheng Chen, Yujun Li, Guangzeng Xie, and Zhihua Zhang. A stochastic proximal
point algorithm for saddle-point problems. arXiv preprint:1909.06946, 2019.

[31] Luo Luo, Guangzeng Xie, Tong Zhang, and Zhihua Zhang. Near optimal stochastic algorithms
for finite-sum unbalanced convex-concave minimax optimization. arXiv preprint:2106.01761,
2021.

[32] Grigory Malinovsky, Kai Yi, and Peter Richtárik. Variance reduced ProxSkip: algorithm, theory
and application to federated learning. Advances in Neural Information Processing Systems,
2022.

[33] Yu Malitsky. Projected reflected gradient methods for monotone variational inequalities. SIAM
Journal on Optimization, 25(1):502–520, 2015.

12


---Page Break---
[34] Yura Malitsky and Matthew K. Tam. A forward-backward splitting method for monotone
inclusions without cocoercivity. SIAM Journal on Optimization, 30(2):1451–1472, 2020.

[35] Ulysse Marteau-Ferey, Francis Bach, and Alessandro Rudi. Globally convergent newton
methods for ill-conditioned generalized self-concordant losses. Advances in Neural Information
Processing Systems, 2019.

[36] Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. ProxSkip:
Yes! local gradient steps provably lead to communication acceleration! finally! In International
Conference on Machine Learning, pages 15750–15769. PMLR, 2022.

[37] Aritra Mitra, Rayana Jaafar, George J. Pappas, and Hamed Hassani. Linear convergence in
federated learning: Tackling client heterogeneity and sparse gradients. Advances in Neural
Information Processing Systems, 2021.

[38] Arkadi Nemirovski. Prox-method with rate of convergence O(1/t) for variational inequali-
ties with lipschitz continuous monotone operators and smooth convex-concave saddle point
problems. SIAM Journal on Optimization, 15:229–251, 2004.

[39] Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and
related problems. Mathematical Programming, 109(2):319–344, 2007.

[40] Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87.
Springer Science & Business Media, 2013.

[41] Ivano Notarnicola, Mauro Franceschelli, and Giuseppe Notarstefano. A duality-based approach
for distributed min–max optimization. IEEE Transactions on Automatic Control, 64(6):2559–
2566, 2019.

[42] Yuyuan Ouyang and Yangyang Xu. Lower complexity bounds of first-order methods for
convex-concave bilinear saddle-point problems. Mathematical Programming, 185(1):1–35,
2021.

[43] Leonid Denisovich Popov. A modification of the arrow-hurwicz method for search of saddle
points. Mathematical notes of the Academy of Sciences of the USSR, 28:845–848, 1980.

[44] Meisam Razaviyayn, Tianjian Huang, Songtao Lu, Maher Nouiehed, Maziar Sanjabi, and
Mingyi Hong. Nonconvex min-max optimization: Applications, challenges, and recent theoreti-
cal advances. IEEE Signal Processing Magazine, 37(5):55–66, 2020.

[45] Alexander Rogozin, Aleksandr Beznosikov, Darina Dvinskikh, Dmitry Kovalev, Pavel
Dvurechensky, and Alexander Gasnikov. Decentralized distributed optimization for saddle point
problems. arXiv preprint:2102.07758, 2021.

[46] Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization
using an approximate newton-type method. In International conference on machine learning,
2014.

[47] Sebastian U. Stich.
Local SGD converges fast and communicates little.
arXiv
preprint:1805.09767, 2018.

[48] Ying Sun, Gesualdo Scutari, and Amir Daneshmand. Distributed optimization based on gradient
tracking revisited: Enhancing convergence rate via surrogation. SIAM Journal on Optimization,
32(2):354–385, 2022.

[49] Rafał Szlendak, Alexander Tyurin, and Peter Richtárik. Permutation compressors for provably
faster distributed nonconvex optimization. arXiv preprint:2110.03300, 2021.

[50] Ye Tian, Gesualdo Scutari, Tianyu Cao, and Alexander Gasnikov. Acceleration in distributed op-
timization under similarity. In International Conference on Artificial Intelligence and Statistics,
2022.

[51] Paul Tseng. A modified forward-backward splitting method for maximal monotone mappings.
SIAM Journal on Control and Optimization, 38(2):431–446, 2000.

13


---Page Break---
[52] Panos Xidonas, George Mavrotas, Christis Hassapis, and Constantin Zopounidis. Robust
multiobjective portfolio optimization: A minimax regret approach. European Journal of
Operational Research, 262(1):299–305, 2017.

[53] Guangzeng Xie, Luo Luo, Yijiang Lian, and Zhihua Zhang. Lower complexity bounds for
finite-sum convex-concave minimax optimization problems. In International Conference on
Machine Learning, 2020.

[54] Junchi Yang, Siqi Zhang, Negar Kiyavash, and Niao He. A catalyst framework for minimax
optimization. Advances in Neural Information Processing Systems, 33:5667–5678, 2020.

[55] Junyu Zhang, Mingyi Hong, and Shuzhong Zhang. On lower iteration complexity bounds for
the convex concave saddle point problems. Mathematical Programming, 194(1):901–935, 2022.

[56] Yuchen Zhang and Xiao Lin. DiSCO: Distributed optimization for self-concordant empirical
loss. In International Conference on Machine Learning, 2015.

14


---Page Break---
Appendix

The appendix contains additional details supporting the main text. Section A starts with some basic
results. Section B shows the non-negativity of our Lyapunov function. Section C provides the proof
of upper bounds for the proposed method. Section D formally defines the algorithm class in our
lower bound analysis. Section E and F provide the lower complexity bound for both convex-concave
and strongly-convex-strongly-concave cases. Section G demonstrates the complexity of making the
gradient mapping small. Section H presents more experimental results.

A
Some Basic Results

We introduce the following lemmas for our later analysis.
Lemma 3 (Lin et al. [29, Proposition B.1]). If the local functions f1, . . . , fn : Rdx × Rdy →R hold
the δ-second-order similarity, then each (Fi −F)(·) is δ-Lipschitz continuous, i.e., we have
∥(Fi −F)(z1) −(Fi −F)(z2)∥≤δ∥z1 −z2∥

for all z1, z2 ∈Rd and i ∈[n].
Lemma 4 (Alacaoglu and Malitsky [1, Section 8]). Let F = {Fk}k≥0 be a filtration and {rk} be
a stochastic process adapted to F with E[rk+1|Fk] = 0. Then for any K ∈N, x0 ∈Z, and any
compact set C ⊂Z, we have

E

"

max
x∈C

K−1
X

k=0
⟨rk+1, x⟩

#

≤max
x∈C
1
2∥x0 −x∥2 + 1

2

K−1
X

k=0
E∥rk+1∥2.

In related work, Beznosikov et al. [8, Assumption 4.3] considers the following second-order similarity
assumption that is slightly different from our Assumption 6.
Assumption 7. The local functions f1, . . . , fn : Rdx × Rdy →R are twice differentiable and hold
the δ-average-second-order similarity, i.e., there exists δ > 0 such that

1
n

n
X

i=1

∇2fi(x, y) −∇2fj(x, y)
2 ≤δ2

for all x ∈Rdx, y ∈Rdy, and j ∈[n], .

We present the relationship between δ-average-second-order similarity (Assumption 7) and δ-second-
order-similarity (Assumption 6).
Proposition 1. For twice differentiable local functions f1, . . . , fn : Rdx × Rdy →R, we have

• If functions {fi}n
i=1 hold the δ-average-second-order similarity, then they also hold the δ-second-
order similarity.
• If functions {fi}n
i=1 hold the δ-second-order similarity, then they hold the 2δ-average-second-
order similarity.

Proof. If functions {fi}n
i=1 hold the δ-average-second-order similarity, then for all j ∈[n], we have

∥∇2fj(x, y) −∇2f(x, y)∥2
2 =


1
n

n
X

i=1


∇2fj(x, y) −∇2fi(x, y)



2

2

≤1

n

n
X

i=1
∥∇2fj(x, y) −∇2fi(x, y)∥2
2 ≤δ2,

where we use the convexity of ∥· ∥2
2. This implies functions {fi}n
i=1 also hold the δ-second-order
similarity.

If functions {fi}n
i=1 hold the δ second-order similarity, then for all j ∈[n], we have

1
n

n
X

i=1
∥∇2fj(x, y) −∇2fi(x, y)∥2
2

≤1

n

n
X

i=1

 
2∥∇2fj(x, y) −∇2f(x, y)∥2
2 + 2∥∇2f(x, y) −∇2fi(x, y)∥2
2

≤(2δ)2,

15


---Page Break---
where we use the Young’s inequality for the matrix 2-norm. This implies functions {fi}n
i=1 hold the
2δ-average-second-order similarity.

B
The Non-Negativity of Lyapunov Function

Our convergence analysis is based on the following Lyapunov function

Φk =
1 −γ

η
+ µ

2


∥zk −z∗∥2 + 2⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩

+
1
64η ∥zk −zk−1∥2 + γ

2η ∥wk−1 −zk∥2 + (2γ + ηµ)

2pη
∥wk −z∗∥2.

Noticing that we can always guarantees Φk ≥0 by taking η ≤1/(32δ), because the Young’s
inequality and Lemma 3 indicates

Φk ≥1 −γ

η
∥zk −z∗∥2 −
1
64ηδ2 ∥F(zk−1) −F1(zk−1) −F(zk) + F1(zk)∥2

−64ηδ2∥zk −z∗∥2 +
1
64η ∥zk −zk−1∥2

≥1 −γ

η
∥zk −z∗∥2 −
1
64ηδ2 δ2∥zk −zk−1∥2 −64ηδ2∥zk −z∗∥2 +
1
64η ∥zk −zk−1∥2

≥1

2η (1 −128η2δ2)∥zk −z∗∥2 ≥0,

where we also use γ ≤1/8.

C
The Proofs for Upper Complexity Bounds

We provide the proofs for results in Section 5.

C.1
Proof of Lemma 1

In later analysis, we denote Ek[·] as the expectation with respect to the random sampled set Sk in
round k and denote Ek+1/2[·] as the expectation with respect to the random update of the snapshot
point wk in round k. Specifically, we take the constant

c := 100 + 64ηδ2

µ
+ 2048η2δ2 + 96ηµ + 64
p

2ηΦ0
(14)

for the statement of Lemma 1.

We first provide several lemmas that will be used in the proof of Lemma 1.
Lemma 5. Under the setting of Lemma 1, we have

−2E

⟨δk −Ek[δk], ˆuk −zk⟩


≤1

2η E

∥ˆuk −zk∥2
+ 4ηδ2

b
E

∥zk −wk−1∥2
+ 4ηδ2α2

b
E

∥zk −zk−1∥2
,

and

−2E

⟨Ek[δk] + F1(ˆuk) −F(z∗), ˆuk −z∗⟩


≤2δ2

µ E

∥ˆuk −uk∥2
+ µ

2 E

∥ˆuk −z∗∥2
−2µE

∥ˆuk −z∗∥2
+
1
64η E

∥zk −zk+1∥2

+ 64ηδ2E

∥ˆuk −uk∥2
−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+ 1

4η E

∥zk −ˆuk∥2
+ 4ηδ2α2E

∥zk −zk−1∥2

−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z∗⟩

.

16


---Page Break---
Proof. Firstly note that
Ek[δk] = F(zk) −F1(zk) + α
 
F(zk) −F1(zk) −F(zk−1) + F1(zk−1)

.
According to the uniform and independent sampling and Lemma 3 we have

E

∥δk −Ek[δk]∥2
≤2E



1
b

X

j∈Sk

 
Fj(zk) −Fj(wk−1)

−
 
F(zk) −F(wk−1)



2

+ 2E



α

b

X

j∈Sk

 
Fj(zk) −Fj(zk−1)

−
 
F(zk) −F(zk−1)



2

= 2

b2 E



X

j∈Sk

 
Fj(zk) −Fj(wk−1)

−
 
F(zk) −F(wk−1)
2




+ 2α2

b2 E



X

j∈Sk

 
Fj(zk) −Fj(zk−1)

−
 
F(zk) −F(zk−1)
2




≤2

nbE




n
X

j=1

 
Fj(zk) −Fj(wk−1)

−
 
F(zk) −F(wk−1)
2




+ 2α2

nb E




n
X

j=1

 
Fj(zk) −Fj(zk−1)

−
 
F(zk) −F(zk−1)
2




≤2δ2

b E

∥zk −wk−1∥2
+ 2δ2α2

b
E

∥zk −zk−1∥2
.

According to the above bound on E

∥δk −Ek[δk]∥2
, we achieve the first result as follows

−2E

⟨δk −Ek[δk], ˆuk −zk⟩


≤1

2η E

∥ˆuk −zk∥2
+ 2ηE

∥δk −Ek[δk]∥2

≤1

2η E

∥ˆuk −zk∥2
+ 4ηδ2

b
E

∥zk −wk−1∥2
+ 4ηδ2α2

b
E

∥zk −zk−1∥2
.

Again using Lemma 3, we achieve the second result as follows

−2E

⟨Ek[δk] + F1(ˆuk) −F(z∗), ˆuk −z∗⟩


= −2E

⟨F(zk) −F1(zk) + α
 
F(zk) −F1(zk) −F(zk−1) + F1(zk−1)

, ˆuk −z∗⟩


−2E

⟨F1(ˆuk) −F(z∗), ˆuk −z∗⟩


= −2E

⟨F(uk) −F1(uk) −F(ˆuk) + F1(ˆuk), ˆuk −z∗⟩

−2E

⟨F(ˆuk) −F(z∗), ˆuk −z∗⟩


−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), ˆuk −zk+1⟩


−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), ˆuk −zk⟩


−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z∗⟩


≤2δ2

µ E

∥ˆuk −uk∥2
+ µ

2 E

∥ˆuk −z∗∥2
−2µE

∥ˆuk −z∗∥2
+
1
64η E

∥zk −zk+1∥2

+ 64ηδ2E

∥ˆuk −uk∥2
−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+ 1

4η E

∥zk −ˆuk∥2
+ 4ηδ2α2E

∥zk −zk−1∥2

−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z∗⟩

.

17


---Page Break---
Lemma 6. Under setting of Lemma 1, we have

−1

8η E

∥ˆuk −zk∥2
−γ

η E

∥wk −ˆuk∥2
−3µ

2 E

∥ˆuk −z∗∥2

≤−1

16η E

∥ˆuk −zk∥2
−
1
32η E

∥zk+1 −zk∥2
+
 1

8η + 3µ

E

∥ˆuk −uk∥2

−γ

2η E

∥wk −ˆuk∥2
−γ

4η E

∥wk −zk+1∥2
−µE

∥zk+1 −z∗∥2
.

Proof. From the facts ∥a + b∥2 ≥1

2∥a∥2 −∥b∥2 and 3

2∥a + b∥2 ≥∥a∥2 −3∥b∥2, we have

−1

8η E

∥ˆuk −zk∥2
≤−1

16η E

∥ˆuk −zk∥2
−
1
32η E

∥zk+1 −zk∥2
+
1
16η E

∥ˆuk −uk∥2
,

−γ

η E

∥wk −ˆuk∥2
≤−γ

2η E

∥wk −ˆuk∥2
−γ

4η E

∥wk −zk+1∥2
+ γ

2η E

∥ˆuk −uk∥2

≤−γ

2η E

∥wk −ˆuk∥2
−γ

4η E

∥wk −zk+1∥2
+
1
16η E

∥ˆuk −uk∥2
,

and

−3µ

2 E

∥ˆuk −z∗∥2
≤−µE

∥zk+1 −z∗∥2
+ 3µE

∥ˆuk −uk∥2
,

where we use the setting γ ≤1/8 in the second inequality.

Lemma 7. Under setting of Lemma 1, we have
1 −γ

η
+ µ

2


E

∥zk+1 −z∗∥2
+ 2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+
1
64η E

∥zk+1 −zk∥2
+ γ

4η E

∥wk −zk+1∥2
+ γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2

≤1 −γ

η
E

∥zk −z∗∥2
+ 2αE

⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩


+ α 1

64η E

∥zk −zk−1∥2
+ 4ηδ2

b
E

∥wk−1 −zk∥2
−
1
16η E

∥zk −ˆuk∥2

+

−7

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
+ 2

η E

∥ˆuk −uk∥∥zk+1 −z∗∥


+

1 −
pηµ
2γ + ηµ

 (γ + 1

2ηµ)
pη
E

∥wk −z∗∥2
−γ

2η E

∥wk −ˆuk∥2
.

Proof. The optimality of ˆuk implies

⟨ηF1(ˆuk) + ˆuk −vk, z∗−ˆuk⟩≥0.
(15)

Combine equation (15) with the update rule in Line 7 of Algorithm 1 and ⟨−γF(z∗), z∗−ˆuk⟩≥0,
we achieve

−1

η ⟨¯zk −ˆuk −ηδk, ˆuk −z∗⟩≤−⟨F1(ˆuk) −F(z∗), ˆuk −z∗⟩.
(16)

18


---Page Break---
Using the result of equation (16), we have

1
η ∥ˆuk −z∗∥2 = 1

η ∥zk −z∗∥2 + 2

η ⟨ˆuk −zk, ˆuk −z∗⟩−1

η ∥ˆuk −zk∥2

= 1

η ∥zk −z∗∥2 + 2γ

η ⟨wk −zk, ˆuk −z∗⟩−2⟨δk, ˆuk −z∗⟩

−1

η ∥ˆuk −zk∥2 −2

η ⟨¯zk −ˆuk −ηδk, ˆuk −z∗⟩

≤1

η ∥zk −z∗∥2 + 2γ

η ⟨wk −zk, ˆuk −z∗⟩−2⟨δk, ˆuk −z∗⟩

−1

η ∥ˆuk −zk∥2 −2⟨F1(ˆuk) −F(z∗), ˆuk −z∗⟩

= 1

η ∥zk −z∗∥2 + γ

η ∥wk −z∗∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∗∥2

−1 −γ

η
∥ˆuk −zk∥2 −2⟨δk −Ek[δk], ˆuk −zk + zk −z∗⟩

−2⟨Ek[δk] + F1(ˆuk) −F(z∗), ˆuk −z∗⟩.

Taking the expectation on above result and using the fact

E

⟨δk −Ek[δk], zk −z∗⟩

= E

Ek

⟨δk −Ek[δk], zk −z∗⟩

= 0,

we obtain

1
η E

∥ˆuk −z∗∥2
≤E
1

η ∥zk −z∗∥2 + γ

η ∥wk −z∗∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∗∥2


−E
1 −γ

η
∥ˆuk −zk∥2

−2E

⟨δk −Ek[δk], ˆuk −zk⟩


−2E

⟨Ek[δk] + F1(ˆuk) −F(z∗), ˆuk −z∗⟩

.

(17)

Applying Lemma 5 to bound the term −2E

⟨δk −Ek[δk], ˆuk −zk⟩

in equation (17), we obtain

1
η E

∥ˆuk −z∗∥2

≤E
1

η ∥zk −z∗∥2 + γ

η ∥wk −z∗∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∗∥2 −1/4 −γ

η
∥ˆuk −zk∥2


+ 4ηδ2

b
E

∥zk −wk−1∥2
+ 4ηδ2α2

b
E

∥zk −zk−1∥2

+ 2δ2

µ E

∥ˆuk −uk∥2
−3

2µE

∥ˆuk −z∗∥2
+
1
64η E

∥zk −zk+1∥2
+ 64ηδ2E

∥ˆuk −uk∥2

−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩

+ 4ηδ2α2E

∥zk −zk−1∥2

−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z∗⟩


≤1

η E

∥zk −z∗∥2
+ γ

η E

∥wk −z∗∥2
−γ

η E

∥wk −ˆuk∥2
−γ

η E

∥zk −z∗∥2

−1

8η E

∥ˆuk −zk∥2
+
1
64η E

∥zk −zk+1∥2
+ 4ηδ2

b
E

∥zk −wk−1∥2

+
4ηδ2α2

b
+ 4ηδ2α2

E

∥zk −zk−1∥2
+
2δ2

µ + 64ηδ2

E

∥ˆuk −uk∥2
(18)

−2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩

−3µ

2 E

∥ˆuk −z∗∥2

−2αE

⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z∗⟩

,

where we use the setting γ ≤1/8 in the second inequality.

19


---Page Break---
Then we consider the terms related to ˆu. Firstly, we have

1
η E

∥ˆuk −z∗∥2
= 1

η E

∥zk+1 −z∗∥2
+ 1

η E

∥ˆuk −uk∥2
−2

η E

∥ˆuk −uk∥∥zk+1 −z∗∥

.

(19)

Applying Lemma 6 and plugging equation (19) into equation (18), we have

1
η E

∥zk+1 −z∗∥2
+ 2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+
1
64η E

∥zk+1 −zk∥2
+ γ

4η E

∥wk −zk+1∥2

≤1

η E

∥zk −z∗∥2
+ 2αE

⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩


+ α

64η E

∥zk −zk−1∥2
+ γ

η E

∥wk −z∗∥2
−γ

η E

∥zk −z∗∥2
−µE

∥zk+1 −z∗∥2

+ 4ηδ2

b
E

∥zk −wk−1∥2
+ 2

η E

∥ˆuk −uk∥∥zk+1 −z∗∥

−
1
16η E

∥ˆuk −zk∥2

+

−7

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
−γ

2η E

∥wk −ˆuk∥2
,

(20)

where we use the fact that 256η2δ2α2/b + 256η2δ2α2 ≤α to bound the coefficient before the term
of E

∥zk −zk−1∥2
.

Then we add the term

µE

∥zk+1 −z∗∥2
+ γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2
(21)

to both sides of equation (20) and use the update rule in Line 10 of Algorithm 1 to obtain

γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2
= γ + 1

2ηµ
pη
E

Ewk+1

∥wk+1 −z∗∥2

= γ + 1

2ηµ
η
E

∥zk+1 −z∗∥2
+ (γ + 1

2ηµ)(1 −p)

pη
E

∥wk −z∗∥2
,

and

γ
η + (γ + 1

2ηµ)(1 −p)

pη
=

1 −p +
pγ
(γ + 1

2ηµ)

 (γ + 1

2ηµ)
pη
=

1 −
pηµ
2γ + ηµ

 (γ + 1

2ηµ)
pη
.

Combining all above results, we achieve
1 −γ

η
+ µ

2


E

∥zk+1 −z∗∥2
+ 2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+
1
64η E

∥zk+1 −zk∥2
+ γ

4η E

∥wk −zk+1∥2
+ γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2

≤1 −γ

η
E

∥zk −z∗∥2
+ 2αE

⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩


+ α

64η E

∥zk −zk−1∥2
+ 4ηδ2

b
E

∥wk−1 −zk∥2
−
1
16η E

∥zk −ˆuk∥2

+

−7

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
+ 2

η E

∥ˆuk −uk∥∥zk+1 −z∗∥


+

1 −
pηµ
2γ + ηµ

 (γ + 1

2ηµ)
pη
E

∥wk −z∗∥2
−γ

2η E

∥wk −ˆuk∥2
.

20


---Page Break---
Lemma 8. Under setting of Lemma 1, we additionally assume E[Φk] ≤Φ0 holds, then we have

E[Φk+1] ≤max

1−
ηµ
6(1 −γ), 1−
pηµ
2γ + ηµ


E[Φk]−1

16η E

∥zk −ˆuk∥2
−γ

2η E

∥wk −ˆuk∥2
.

Proof. Recall that the definition of our Lyapunov function is

Φk =
1 −γ

η
+ µ

2


∥zk −z∗∥2 + 2⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩

+
1
64η ∥zk −zk−1∥2 + γ

4η ∥wk−1 −zk∥2 + (2γ + ηµ)

2pη
∥wk −z∗∥2.
(22)

Recall that we take constant c by equation (14), then the condition

εk ≤c−1 min

∥ˆuk −zk∥, ∥ˆuk −zk∥2	

guarantees

E

∥ˆuk −uk∥

≤˜ζk min

E

∥ˆuk −zk∥

, E

∥ˆuk −zk∥2	
,
where

˜ζk =
1
32η
1

9
8η + 2δ2

µ + 64ηδ2 + 3µ + 2

η + 2
q

2E[Φk]

η

=
1

100 + 64ηδ2

µ
+ 2048η2δ2 + 96ηµ + 64
p

2ηE[Φk]

≥
1

100 + 64ηδ2

µ
+ 2048η2δ2 + 96ηµ + 64
p

2ηΦ0 = 1

c .

(23)

The inequality (23) is based on the assumption E[Φk] ≤Φ0.

Note that we have ∥zk+1 −z∗∥≤∥uk −ˆuk∥+ ∥ˆuk −zk∥+ ∥zk −z∗∥, then

−7

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
+ 2

η E

∥ˆuk −uk∥∥zk+1 −z∗∥


≤
 9

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
+ 2

η E

∥ˆuk −uk∥∥ˆuk −zk∥


+ 2

η E

∥ˆuk −uk∥∥zk −z∗∥


≤
 9

8η + 2δ2

µ + 64ηδ2 + 3µ

E

∥ˆuk −uk∥2
+ 2

η E

∥ˆuk −uk∥∥ˆuk −zk∥


+ 2

s

2E[Φk]

η
E

∥ˆuk −uk∥


≤
1
32η E

∥ˆuk −zk∥2
.

According to Lemma 7, we have
1 −γ

η
+ µ

2


E

∥zk+1 −z∗∥2
+ 2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+
1
64η E

∥zk+1 −zk∥2
+ γ

4η E

∥wk −zk+1∥2
+ γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2

≤1 −γ

η
E

∥zk −z∗∥2
+ 2αE

⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩


+ α 1

64η E

∥zk −zk−1∥2
+ 4ηδ2

b
E

∥wk−1 −zk∥2
+
1
32η E

∥zk −ˆuk∥2

−
1
16η E

∥zk −ˆuk∥2
−γ

2η E

∥wk −ˆuk∥2
+

1 −
pηµ
2γ + ηµ

 (γ + 1

2ηµ)
pη
E

∥wk −z∗∥2
.

21


---Page Break---
From the fact that ηµ ≤1, we have

1 −γ

η
≤

1 −
ηµ
6(1 −γ)

 1 −γ

η
+ µ

2


,

and according to the fact that 4ηδ2/b ≤αγ/(4η), we obtain
1 −γ

η
+ µ

2


E

∥zk+1 −z∗∥2
+ 2E

⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z∗⟩


+
1
64η E

∥zk+1 −zk∥2
+ γ

4η E

∥wk −zk+1∥2
+ γ + 1

2ηµ
pη
E

∥wk+1 −z∗∥2

≤

1 −
ηµ
6(1 −γ)

1 −γ

η
+ µ

2


E

∥zk −z∗∥2
+

1 −
pηµ
2γ + ηµ

 (γ + 1

2ηµ)
pη
E

∥wk −z∗∥2

+ 2αE

⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z∗⟩

+ α 1

64η E

∥zk −zk−1∥2

+ α γ

2η E

∥wk−1 −zk∥2
−
1
16η E

∥zk −ˆuk∥2
−γ

2η E

∥wk −ˆuk∥2
.

The definition (22) and the setting α = max {1 −ηµ/(6(1 −γ)), 1 −pηµ/(2γ + ηµ)} implies

E[Φk+1] ≤αE[Φk] −
1
16η E

∥zk −ˆuk∥2
−γ

2η E

∥wk −ˆuk∥2
.
(24)

Then we provide the proof of Lemma 1.

Proof. We firstly use the induction to prove

E[Φk] ≤Φ0

holds for all k ∈N.

Note that it holds for k = 0. Assume we have E[Φk] ≤Φ0 holds, then Lemma 8 means it holds

E[Φk+1] ≤max

1 −
ηµ
6(1 −γ), 1 −
pηµ
2γ + ηµ


E[Φk] ≤E[Φk] ≤Φ0,

which finish the induction.

The result of above induction implies the condition of Lemma 8 always holds. Therefore, we can
apply Lemma 8 to achieve equation (24), which finishes the proof of Lemma 1.

C.2
Proof of Theorem 1

We firstly introduce the following quantities for our analysis

e11(z, k) := 2η

b

X

j∈Sk
⟨F(zk) −Fj(zk) −F(wk−1) + Fj(wk−1), ˆuk −z⟩,

e12(z, k) := 2ηα

b

X

j∈Sk
⟨F(zk) −Fj(zk) −F(zk−1) + Fj(zk−1), ˆuk −z⟩,

e2(z, k) := ∥wk+1 −z∥2 −p∥zk −z∥2 −(1 −p)∥wk −z∥2,

Ψk(z) := (1 −γ)∥zk+1 −z∥2 + γ

p ∥wk −z∥2 + 1

16∥zk −zk−1∥2.

(25)

Specifically, we take the constant

ζ := min

η2ε2

16(9ηLD + 3η maxi∈[n] ∥Fi(z0)∥+ D)2 ,
ηε
4(12η2δ2 + 1)


(26)

22


---Page Break---
and

ˆc := 100 + 2048η2δ2 + 64
p

2ηΦ0 ≤102 + 16
p

Φ0/δ
(27)

for the statement Theorem 1. We then provide several lemmas that will be used in the proof of
Theorem 1.
Lemma 9. Under the setting of Theorem 1, we have

−2⟨Ek[δk] + F1(ˆuk), ˆuk −z⟩

≤4LD∥uk −ˆuk∥+ 2⟨F(uk), z −uk⟩+ (8LD + 6DF )∥ˆuk −uk∥+
1
16η ∥zk −zk+1∥2

+ 16ηδ2∥ˆuk −uk∥2 −2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

+ 1

2η ∥zk −ˆuk∥2 + 2ηδ2α2∥zk −zk−1∥2

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩,

where DF := maxi∈[n] supz∈Z ∥Fi(z)∥.

Proof. Note that the sequence {∥Fi(z)∥}n
i=1 is bounded on z ∈Z, since we have

DF = max
i∈[n] sup
z∈Z
∥Fi(z)∥≤max
i∈[n] sup
z∈Z
(∥Fi(z) −Fi(z0)∥+ ∥Fi(z0)∥) ≤LD + max
i∈[n] ∥Fi(z0)∥.

(28)

The Lipschitz continuity of F(·) impies

∥F1(ˆuk)∥−DF ≤∥F1(ˆuk)∥−∥F1(z)∥≤∥F1(ˆuk) −F1(z)∥≤LD,

then we have
−2⟨Ek[δk] + F1(ˆuk), ˆuk −z⟩

= −2⟨F(zk) −F1(zk) + α
 
F(zk) −F1(zk) −F(zk−1) + F1(zk−1)

+ F1(ˆuk), ˆuk −z⟩

= −2⟨F(uk) −F1(uk) −F(ˆuk) + F1(ˆuk), ˆuk −z⟩

−2⟨F(ˆuk), ˆuk −z⟩

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), ˆuk −zk+1⟩

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), ˆuk −zk⟩

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩

≤4LD∥uk −ˆuk∥+ 2⟨F(uk), z −uk⟩+ (8LD + 6DF )∥ˆuk −uk∥+
1
16η ∥zk −zk+1∥2

+ 16ηδ2∥ˆuk −uk∥2 −2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

+ 1

2η ∥zk −ˆuk∥2 + 2ηδ2α2∥zk −zk−1∥2

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩.

(29)

Lemma 10. Under the setting of Theorem 1, the quantities defined in equation (25) hold

max
z∈Z Ψ0(z) + E

"

max
z∈Z

K−1
X

k=0
e11(z, k) + e12(z, k) + γ

p e2(z, k)

#

≤max
z∈Z 4∥z0 −z∥2 + 4η2δ2

b

K−1
X

k=0
E

∥zk −ˆuk∥2
+

2p + 4η2δ2

b

 K−1
X

k=0
E

∥wk −ˆuk∥2

+

2p + 8η2δ2

b

 K−1
X

k=0
E

∥ˆuk −uk∥2
.

23


---Page Break---
Proof. Applying Lemma 4 with x0 = z0, F0 = σ(S0), Fk = σ(S0, . . . , Sk−1, wk) for k ≥1, and
rk+1 = 2η

b
P

j∈Sk Fj(zk) −F(zk) −Fj(wk−1) + F(wk−1) and using Ek[rk+1] = 0, we have

E

"

max
z∈Z

K−1
X

k=0
e11(z, k)

#

= E

"

max
z∈Z

K−1
X

k=0
⟨rk+1, z⟩

#

≤max
z∈Z
1
2∥z0 −z∥2 + 1

2

K−1
X

k=0
E

∥rk+1∥2

≤max
z∈Z
1
2∥z0 −z∥2 + 2η2δ2

b

K−1
X

k=0
E

∥zk −wk−1∥2
.

Similarly we can obtain

E

"

max
z∈Z

K−1
X

k=0
e12(z, k)

#

≤max
z∈Z
1
2∥z0 −z∥2 + 2η2α2δ2

b

K−1
X

k=0
E

∥zk −zk−1∥2
.

Applying Lemma 4 with x0 = z0, F0 = σ(S0), Fk = σ(S0, . . . , Sk−1, wk) for k ≥1, and
rk+1 = pzk+1+(1−p)wk−wk+1 and using the fact that E[∥wk+1∥2−p∥zk+1∥2−(1−p)∥wk∥2] = 0
and Ek[rk+1] = 0, we have

E

"

max
z∈Z

K−1
X

k=0
e2(z, k)

#

= 2E

"

max
z∈Z

K−1
X

k=0
⟨rk+1, z⟩

#

≤max
z∈Z ∥z0 −z∥2 +

K−1
X

k=0
E

∥rk+1∥2

≤max
z∈Z ∥z0 −z∥2 + p(1 −p)

K−1
X

k=0
E

∥zk+1 −wk∥2
,

where we use

E

∥rk+1∥2
= E

Ek+1/2∥Ek+1/2[wk+1] −wk+1∥2

= E

Ek+1/2[∥wk+1∥2] −∥Ek+1/2[wk+1]∥2

= E

p∥zk+1∥2 + (1 −p)∥wk∥2 −∥pzk+1 + (1 −p)wk∥2

= p(1 −p)E

∥zk+1 −wk∥2
.

Note that zk = uk−1, then we have

max
z∈Z Ψ0(z) + E

"

max
z∈Z

K−1
X

k=0
e11(z, k) + e12(z, k) + γ

p e2(z, k)

#

≤4 max
z∈Z ∥z0 −z∥2 + 2η2δ2

b

K−1
X

k=0
E

∥zk −wk−1∥2
+ 2η2α2δ2

b

K−1
X

k=0
E

∥zk −zk−1∥2

+ p(1 −p)

K−1
X

k=0
E

∥zk+1 −wk∥2

≤4 max
z∈Z ∥z0 −z∥2 + 4η2δ2

b

K−1
X

k=0
E

∥ˆuk−1 −wk−1∥2
+ 4η2α2δ2

b

K−1
X

k=0
E

∥ˆuk−1 −zk−1∥2

+ 2p

K−1
X

k=0
E

∥ˆuk −wk∥2
+
4η2δ2

b
+ 4η2α2δ2

b
+ 2p
 K−1
X

k=0
E

∥ˆuk −uk∥2

≤4 max
z∈Z ∥z0 −z∥2 + 4η2δ2

b

K−1
X

k=0
E

∥zk −ˆuk∥2
+

2p + 4η2δ2

b

 K−1
X

k=0
E

∥wk −ˆuk∥2

+

2p + 8η2δ2

b

 K−1
X

k=0
E

∥ˆuk −uk∥2
,

where we use Young’s inequality and α ≤1.

24


---Page Break---
Now we provide the proof of Theorem 1.

Proof. The optimality of ˆuk implies for all z ∈Z, we have

⟨ηF1(ˆuk) + ˆuk −vk, z −ˆuk⟩≥0.
(30)

Combine equation (30) with the update rule in Line 7 of Algorithm 1 , we achieve

−1

η ⟨¯zk −ˆuk −ηδk, ˆuk −z⟩≤−⟨F1(ˆuk), ˆuk −z⟩.
(31)

Then we have
1
η ∥ˆuk −z∥2 = 1

η ∥zk −z∥2 + 2

η ⟨ˆuk −zk, ˆuk −z⟩−1

η ∥ˆuk −zk∥2

= 1

η ∥zk −z∥2 + 2γ

η ⟨wk −zk, ˆuk −z⟩−2⟨δk, ˆuk −z⟩

−1

η ∥ˆuk −zk∥2 −2

η ⟨¯zk −ˆuk −ηδk, ˆuk −z⟩

≤1

η ∥zk −z∥2 + 2γ

η ⟨wk −zk, ˆuk −z⟩−2⟨δk, ˆuk −z⟩−1

η ∥ˆuk −zk∥2

−2⟨F1(ˆuk), ˆuk −z⟩

= 1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∥2 −1 −γ

η
∥ˆuk −zk∥2

−2⟨δk + F1(ˆuk), ˆuk −z⟩

= 1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∥2 −1 −γ

η
∥ˆuk −zk∥2

−2⟨Eδk + F1(ˆuk), ˆuk −z⟩+ 1

η e11(z, k) + 1

η e12(z, k),

(32)

Combining the results of equation (32) with Lemma 9, we have

1
η ∥ˆuk −z∥2 ≤1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∥2

−1 −γ

η
∥ˆuk −zk∥2 −2⟨Eδk + F1(ˆuk), ˆuk −z⟩+ 1

η e11(z, k) + 1

η e12(z, k)

≤1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥wk −ˆuk∥2 −γ

η ∥zk −z∥2 −1

4η ∥ˆuk −zk∥2

+ 2⟨F(uk), z −uk⟩+ (12LD + 6DF )∥ˆuk −uk∥+
1
16η ∥zk −zk+1∥2

+ 16ηδ2∥ˆuk −uk∥2 −2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

+ 2ηδ2α2∥zk −zk−1∥2 −2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩

+ 1

η e11(z, k) + 1

η e12(z, k),

(33)

where we use the fact γ ≤1/4.

From the fact that ∥a + b∥2 ≥1

2∥a∥2 −∥b∥2 and 3

2∥a + b∥2 ≥∥a∥2 −3∥b∥2 , we have

−1

4η ∥ˆuk −zk∥2 ≤−1

8η ∥zk+1 −zk∥2 + 1

4η ∥ˆuk −uk∥2,

−γ

η ∥wk −ˆuk∥2 ≤−γ

2η ∥wk −zk+1∥2 + γ

η ∥ˆuk −uk∥2.
(34)

25


---Page Break---
Plugging equation (34) into equation (33), we achieve

1
η ∥ˆuk −z∥2 ≤1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥zk −z∥2 −1

8η ∥zk+1 −zk∥2

+ 1

4η ∥ˆuk −uk∥2 + 2⟨F(uk), z −uk⟩+ (12LD + 6DF )∥ˆuk −uk∥

+
1
16η ∥zk −zk+1∥2 −2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

+ 16ηδ2∥ˆuk −uk∥2 −2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩

+ 2ηδ2α2∥zk −zk−1∥2 −γ

2η ∥wk −zk+1∥2 + γ

η ∥ˆuk −uk∥2

+ 1

η e11(z, k) + 1

η e12(z, k)

≤1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥zk −z∥2 −
1
16η ∥zk+1 −zk∥2

−γ

2η ∥wk −zk+1∥2 + 2⟨F(uk), z −uk⟩+ (12LD + 6DF )∥ˆuk −uk∥

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩+ 2ηδ2α2∥zk −zk−1∥2

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩

+

16ηδ2 + 1

2η


∥ˆuk −uk∥2 + 1

η e11(z, k) + 1

η e12(z, k).

Note that the fact
1
η ∥ˆuk −z∥2 = 1

η ∥zk+1 −z∥2 + 1

η ∥ˆuk −uk∥2 −2

η ∥ˆuk −uk∥∥zk+1 −z∥,

then we have

2⟨F(uk), uk −z⟩≤−1

η ∥zk+1 −z∥2 −1

η ∥ˆuk −uk∥2 + 2

η ∥ˆuk −uk∥∥zk+1 −z∥

+ 1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥zk −z∥2 −
1
16η ∥zk+1 −zk∥2

−γ

2η ∥wk −zk+1∥2 + (12LD + 6DF )∥ˆuk −uk∥+ 2ηδ2α2∥zk −zk−1∥2

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

−2α⟨F(zk) −F1(zk) −F(zk−1) + F1(zk−1), zk −z⟩

+

16ηδ2 + 1

2η


∥ˆuk −uk∥2 + 1

η e11(z, k) + 1

η e12(z, k)

≤−1

η ∥zk+1 −z∥2 −
1
16η ∥zk+1 −zk∥2 −γ

2η ∥zk+1 −wk∥2

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

+ 1

η ∥zk −z∥2 + γ

η ∥wk −z∥2 −γ

η ∥zk −z∥2 + 2ηδ2α2∥zk −zk−1∥2

+ 2α⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z⟩

+

12LD + 6DF + 2D

η


∥ˆuk −uk∥+ 16ηδ2∥ˆuk −uk∥2

+ 1

η e11(z, k) + 1

η e12(z, k)

≤−1 −γ

η
∥zk+1 −z∥2 −γ

ηp∥wk+1 −z∥2 −
1
16η ∥zk+1 −zk∥2

−2⟨F(zk) −F1(zk) −F(zk+1) + F1(zk+1), zk+1 −z⟩

26


---Page Break---
+ 1 −γ

η
∥zk −z∥2 + γ

ηp∥wk −z∥2 + 2ηδ2α2∥zk −zk−1∥2

+ 2α⟨F(zk−1) −F1(zk−1) −F(zk) + F1(zk), zk −z⟩

+

12LΩ+ 6DF + 2D

η


∥ˆuk −uk∥+ 16ηδ2∥ˆuk −uk∥2

+ 1

η


e11(z, k) + e12(z, k) + γ

p e2(z, k)

−γ

2η ∥zk+1 −wk∥2.

The parameters settings implies 2ηδ2α2 ≤1/(16η), then we have

2ηKE

"

max
z∈Z
1
K

K−1
X

k=0
⟨F(uk), uk −z⟩

#

≤max
z∈Z Ψ0(z) + E

"

max
z∈Z

K−1
X

k=0
e11(z, k) + e12(z, k) + γ

p e2(z, k)

#

+

K−1
X

k=0

 
(12ηLD + 6ηDF + 2D) ∥ˆuk −uk∥+ 16η2δ2∥uk −ˆuk∥2
.

(35)

Recall that we take ˆc by equation (27), then the setting

εk = min

ζ, ˆc−1 min

∥ˆuk −zk∥, ∥ˆuk −zk∥2		

satisfies the condition on εk in Lemma 1. Then we apply Lemma 1 with µ = 0 and α = 1 and sum
over equation (9) with k = 0, . . . , K −1 to obtain

K−1
X

k=0

 1

32E

∥zk −ˆuk∥2
+ γ

2 E

∥wk −ˆuk∥2
≤

1 + γ

p


∥z0 −z∗∥2.
(36)

Note that parameter settings γ = p = 1/(√n + 8), b = ⌈√n⌉, and η = min
√γb/(4δ), 1/(32δ)
	

satisfy

4η2δ2

b
≤γ

4 ≤8 · 1

32,
2p + 4η2δ2

b
≤2p + 4δ2

b
γb
16δ2 ≤5 · γ

2
and
1 + γ

p = 2.
(37)

Substituting equations (36) and (37) into equation (35) and applying Lemma 10, we obtain

E

"

max
z∈Z
1
K

K−1
X

k=0
⟨F(uk), uk −z⟩

#

≤
1
2ηK (4 + 8 · 2) max
z∈Z ∥z0 −z∥2

+
1
2ηK

K−1
X

k=0


(12ηLD + 6ηDF + 2D) ∥ˆuk −uk∥+

16η2δ2 + 8η2δ2

b
+ 2p

∥uk −ˆuk∥2


≤10D2

ηK
+ 6ηLD + 3ηDF + D

η

p

ζ + 12η2δ2 + 1

η
ζ

≤10D2

ηK
+ 9ηLD + 3η maxi∈[n] ∥Fi(z0)∥+ D

η

p

ζ + 12η2δ2 + 1

η
ζ,

where we use the equation (28) to bound DF .

Recall that we take constant ζ by equation (26), then we get the bound

E

"

max
z∈Z
1
K

K−1
X

k=0
⟨F(uk), uk −z⟩

#

≤10D2

ηK
+ ε

2.

27


---Page Break---
C.3
Proof of Corollary 1

Proof. Theorem 1 means we can achieve E[Gap(uK
avg)] ≤ε by taking the communication rounds of

K =
20D2

εη


= O
D2

εη


= O
δD2

ε


.

Consider that the expected communication complexity in each round is O(b(1 −p) + np) = O(√n)
and the server need to communicate with all client in initialization within the communication
complexity of O(n), the overall communication complexity is

O(n) + K · O(√n) = O

n +
√nδD2

ε


.

Note that the objective of the sub-problem in Line 8 of Algorithm 1 is (L + 1/η)-smooth-(1/η)-
strongly-convex-(1/η)-strongly-concave, the local gradient complexity for solving the sub-problem
is O((1 + ηL) log(max{ζ−1, ˆc}). Therefore, the overall local gradient complexity is

O(n) + K ·
 
O(√n) + O
 
(1 + ηL) log(ζ−1 + ˆc)


= O(n) + O
δD2

ε


·

 

O(√n) + O

 
1 + L

δ


log

 
LD + DF

ε
+

r

Φ0

δ

!!!

= ˜O

n + (√nδ + L)D2

ε
log 1

ε


.

C.4
Proof of Theorem 2

Proof. We can verify that the parameter setting of Theorem 2 satisfies the condition of Lemma 1.
Then we can apply Lemma 1 to obtain

E[ΦK] ≤max

1 −
ηµ
6(1 −γ), 1 −
pηµ
2γ + ηµ

K
Φ0.
(38)

C.5
Proof of Corollary 2

Proof. Recall that we set the parameters as

γ = p =
1

min
n√n, δ

µ
o
+ 8
,
b =

min
√n, δ

µ


,

η = min
√γb

4δ , 1

32δ


,
α = max

1 −
ηµ
6(1 −γ), 1 −
pηµ
2γ + ηµ


.

We can lower bound α as

α ≥1 −
pηµ
2γ + ηµ = 1 −
pηµ
2p + ηµ = 1 −
1
2
ηµ + 1

p
≥7

8.
(39)

Then the number of communication rounds is

K = O

1 + 1

ηµ + γ + ηµ

pηµ


log 1

ε



= O
1

p + 1

ηµ


log 1

ε



= O
1

p + 1

µ


32δ +
32δ
√αγb


log 1

ε



= O
1

p + δ

µ +
1
√pb
δ
µ


log 1

ε


.

28


---Page Break---
Note that

1
p +
1
√pb
δ
µ ≤min
√n, δ

µ


+ 8 + δ

µ

v
u
u
u
t

min
n√n, δ

µ
o
+ 8

min
n√n, δ

µ
o

= min
√n, δ

µ


+ 8 + δ

µ

s

1 + 8 max
 1
√n, µ

δ



= O
 δ

µ


,

then we have K = O(δ/µ log(1/ε)).

Consider that the expected communication complexity in each round is

O(b(1 −p) + np) = O
√n + nµ

δ


,

and the server need to communicate with all client in initialization within the communication
complexity of O(n), the overall communication complexity is

O(n) + K · O
√n + nµ

δ


= O

n +
√nδ

µ


log 1

ε


.

Note that the objective of the sub-problem in Line 8 of Algorithm 1 is (L + 1/η)-smooth-(1/η)-
strongly-convex-(1/η)-strongly-concave, the local gradient complexity for solving the sub-problem
is O((1 + ηL) log(c)). Therefore, the overall local gradient complexity is

O(n) + K ·

O
√n + nµ

δ


+ O ((1 + ηL) log(c))


= O(n) + O
 δ

µ log 1

ε


·

O
√n + nµ

δ


+ O

1 + L

δ


log δ

µ



= O

n +
√nδ

µ
+ L

µ log δ

µ


log 1

ε



= ˜O

n +
√nδ + L

µ


log 1

ε


.

D
The Algorithm Class

We formally define the distributed first-order oracle (DFO) algorithm as follows.
Definition 1 (DFO Algorithm). Each node i has its own local memories Mx
i and My
i for the x-
and y-variables with initialization Mx
i = My
i = {0} for all i ∈[n]. Specifically, the server has
memories Mx
1 and My
1. These memories {Mx
i }n
i=1 and {My
i }n
i=1 can be updated as follows:

• Communication from clients to server: During one communication round, we sample uniformly
and independently batch S of any size b and ask client with number from S to share some vector
of their local memories with the server, i.e. can add points x′
1, y′
1 to the local memories of the
server according to the next rule:

x′
1 ∈span

(

x1,
[

i∈S
xi

)

and
y′
1 ∈span

(

y1,
[

i∈S
yi

)

where xi ∈Mx
i and yi ∈My
i . If the batch size is equal to b we say that it costs b communication
complexity from clients to the server. Batch of the size n is equal to the situation, when all clients
send their memories to the server.
• Communication from server to clients: During one communication round, we sample uniformly
and independently batch S of any size b and ask the server to share some vector of its local

29


---Page Break---
memories with the clients with numbers from S, i.e. can add points x′
i, y′
i to the corresponding
local memories of client i as

x′
i ∈span {x1, xi}
and
y′
i ∈span {y1, yi} ,

where xi ∈Mx
i and yi ∈My
i , and we say that it costs b communication complexity.
• Local computations: During local computations each client i can make any computations using
fi, i.e. can add points x′
i, y′
i to the corresponding local memory of client i as

x′
i ∈span {x′, ∇xfi (x′′, y′′)}
and
y′
i ∈span {y′, ∇yfi (x′′, y′′)} ,

for given x′, x′′ ∈Mx
i and y′, y′′ ∈My
i . And we use local gradient calls to count the times
when ∇x and ∇y are applied to any one of {fi}.

The final global output is calculated as ˆx ∈Mx
1, ˆy ∈My
1.

Our Definition 1 follows the algorithm class of Beznosikov et al. [8, Definition C.7], but additionally
take the communication from the server to the clients into considerations.

E
Proofs for Lower Bounds in Convex-Concave Case

In this section, we provide the proofs of the lower bounds for solving the problem

min
x∈X max
y∈Y f(x, y) = 1

n

n
X

i=1
fi(x, y)
(40)

by DFO algorithms, where the diameters of closed convex sets X and Y are Rx and Ry respectively.
We define the subspaces {Fk}d
k=0 as

Fk =
span{e1, . . . , ek},
for 1 ≤k ≤d,
{0d},
for k = 0,

which is used in the following proofs of lower bounds.

E.1
Proof of Theorem 3

We first define the function set with one server (i = 1) and n −1 clients (i = 2, . . . , n −1) as follows

fi(x, y) =














δ
4x⊤A1y −δRy

2
√

d
e⊤
1 x,
i −1 ≡1 (mod 3),

δ
4x⊤A2y,
i −1 ≡2 (mod 3),

0,
otherwise.

(41)

Then corresponding global objective is

f(x, y) = δ

6x⊤Ay −δRy

6
√

d
e⊤
1 x,
(42)

where

A1 =









1 0
1
−2
...
...
1
0
1








, A2 =









1 −2
1
0
...
...
1
−2
1








, A =









1
−1
1
−1
...
...
1
−1
1








.

(43)

Proposition 2. For any d ≥3, the functions fi(x, y) and f(x, y) defined by equations (41) and (42)
satisfy

30


---Page Break---
1. The function fi is L-smooth with L ≥δ and convex-concave for all i ∈[n], and the function
set {fi}n
i=1 holds δ-second-order similarity. Thus, the function f is also convex-concave.

2. For 1 ≤k ≤d −1, we have

min
(x,y)∈Z∩F2
k
Gap(x, y) =
min
x∈X∩Fk max
y∈Y f(x, y) −
max
y∈Y∩Fk min
x∈X f(x, y) ≥
δRxRy
6
p

d(k + 1)
. (44)

Proof. The smoothness and the convexity (concavity) are easy to verify. The similarity holds because

∇2
xxf1(x, y) −∇2
xxf(x, y) = ∇2
xxf2(x, y) −∇2
xxf(x, y) = ∇2
xxf3(x, y) −∇2
xxf(x, y) = 0;

∇2
yyf1(x, y) −∇2
yyf(x, y) = ∇2
yyf2(x, y) −∇2
yyf(x, y) = ∇2
yyf3(x, y) −∇2
yyf(x, y) = 0;

∥∇2
xyf1(x, y) −∇2
xyf(x, y)∥≤∥∇2
xyf1(x, y)∥+ ∥∇2
xyf(x, y)∥≤δ

3 ≤δ;

∥∇2
xyf2(x, y) −∇2
xyf(x, y)∥≤∥∇2
xyf2(x, y)∥+ ∥∇2
xyf(x, y)∥≤δ
5

8 + 1

3


≤δ;

∥∇2
xyf3(x, y) −∇2
xyf(x, y)∥≤∥∇2
xyf3(x, y)∥+ ∥∇2
xyf(x, y)∥≤δ
5

8 + 1

3


≤δ.

The function f(x, y) defined by our equation (42) is identical to the function fCC(x, y) defined by
Han et al. [14, Proposition 3.31]3 by replacing their notation L with our δ and taking n = 3. Then
Proposition 3.31 of Han et al. [14] directly prove the result of equation (44).

The structure of A1 and A2 results the following lemma.
Lemma 11. For the function set (41), all (x, y) ∈Fk × Fk and k = 0, . . . , d −1, we have

∇fi(x, y) ∈
Fk+1 × Fk+1,
(i, k) ∈I1 ∪I2,
Fk × Fk,
otherwise,

where the index sets are defined as I1 := {(i, k) : i −1 ≡1 (mod 3), k ≡0 (mod 2)} and
I2 := {(i, k) : i −1 ≡2 (mod 3), k ≡1 (mod 2)}.

Now we provide the proof of Theorem 3.

Proof. Consider the minimax problem (40) with functions (41) and (42), Rx = Ry = D, n ≥3 and
d = ⌊δD2/(3
√

2ε)⌋−1. Then the assumption ε ≤δD2/(12
√

2) implies d ≥3. Lemma 11 means
that we need at least one communication round to increase the number of non-zero coordinate, i.e.
(x, y) ∈Z ∩F2
K. Running any DFO algorithm with communication rounds of K = ⌊(d−1)/2⌋≥1,
we have d/2 ≤(K + 1) ≤(d + 1)/2 and Proposition 2 implies

E[Gap(x, y)] ≥
min
(x,y)∈Z∩F2
K
Gap(x, y) ≥
δD2

6
p

d(K + 1)
≥
δD2

6
√

2(K + 1) ≥
δD2

3
√

2(d + 1) ≥ε.

Hence, we achieve the lower bound on the communication rounds of

K =
 δD2

6
√

2ε


−1 = Ω
δD2

ε


.

E.2
Proof of Theorem 4

We first define the function set with one server (i = 1) and n −1 clients (i = 2, . . . , n −1) as follows

fi(x, y) =














−
√nδRy

8
√

d
e⊤
1 x,
i = 1,

√nδ

8
x⊤
"
X

j≡(i−1)mod(n−1)
eja⊤
j

#

y,
i ≥2.
(45)

3We follow the notation of Han et al. [14]’s arXiv version: https://arxiv.org/pdf/2103.08280v1

31


---Page Break---
Then corresponding global objective is

f(x, y) =
δ
8√nx⊤Ay −δRy

8
√

nd
e⊤
1 x,
(46)

where ej is the j-th basis column vector, a⊤
j is the j-th row of A, Ai = P

j≡(i−1) mod (n−1) eja⊤
j ,
A1 = 0 and A is defined in equation (43).

We provide the following proposition and lemmas for the proof of Theorem 4.

Proposition 3. For any d ≥3, fi(x, y) and f(x, y) defined as equations (45) and (46) satisfy

1. fi is L-smooth with L ≥√nδ/4 and convex-concave, function set {fi}n
i=1 has δ second-order
similarity. Thus, f is convex-concave.

2. For 1 ≤k ≤d −1, we have

min
(x,y)∈Z∩F2
k
Gap(x, y) =
min
x∈X∩Fk max
y∈Y f(x, y) −
max
y∈Y∩Fk min
x∈X f(x, y) ≥
δRxRy
8
p

nd(k + 1)
.

(47)

Proof. The smoothness and convexity (concavity) are easy to check. And the similarity can be
verified following the methods of Beznosikov et al. [8, Lemma C.8]. The function f(x, y) defined by
our equation (46) is identical to the function fCC(x, y) defined by Han et al. [14, Proposition 3.31] by
replacing their notation L with our √nδ/4. Then Proposition 3.31 of Han et al. [14] directly prove
the result of equation (47).

Lemma 12. Consider the minimax problem (40) with functions (45) and (46), Rx = Ry = D,
d = ⌊δD2/(4
√

2nε)⌋−1 and ε ≤δD2/(16
√

2n). We let M = ⌊(d −1)/2⌋≥Ω(δD2/(√nε)),
then any point (x, y) ∈Z ∩F2
M satisfies Gap(x, y) ≥ε.

Proof. The assumptions on ε and M imply d ≥3. and d/2 ≤(M + 1) ≤(d + 1)/2. Then
Proposition 3 means

Gap(x, y) ≥
min
(x,y)∈Z∩F2
M
Gap(x, y) ≥
δD2

8
p

nd(M + 1)
≥
δD2

8
√

2n(M + 1) ≥
δD2

4
√

2n(d + 1) ≥ε.

Lemma 13. Consider the minimax problem (40) with functions (45) and (46) and run any DFO
algorithm with V communication complexity and C local gradient calls. In expectation, only the first
M ≤min{2V/n, 2C/n} coordinates of the final output can be non-zero while the rest of the d −M
coordinates are strictly equal to zero.

Proof. At initialization, Mx
i = My
i = F0. Let’s analyze how Mx
i and My
i change through local
computations. For the i-th client, we add the following points to Mx
i and My
i as

x ∈span {x′, Aiy′} ,
and
y ∈span

e1 · I{i = 1}, y′, A⊤
i x′	
,

where x′ ∈Mx
i and y′ ∈My
i .

It is easy to see that the server can make the first coordinate of y non-zero using e1, and broadcast
this progress to other clients. Only updates of the type Aiy′ or A⊤
i x′ will help in this regard. Since
Ai only contains rows from the matrix A such as the (i −1)-th row, (n + i −1)-th row, etc., to make
the first coordinate of x in the global output non-zero, we need and can only use the A2 matrix. It can
be noted that by using A2, we can also make the second coordinate of y non-zero after making the
first coordinate of x non-zero. Furthermore, to make more progress, we need to use A3 and so on.
We conclude that we must constantly transfer progress from the node currently needed (to make the
next coordinate of x non-zero; then of y) to the server, and then to other nodes.

By definition, one communication round involves communication with all clients or only with
batches of some uniform and independent clients. When we sample without replacement, the success
probability of a communication round on clients with batch size b (i.e., making one coordinate

32


---Page Break---
non-zero) is
b
n−1 (or 1 when b = n), which is also the expected number of non-zero coordinates that
can be obtained with b communication complexity and at least b local gradient calls (as each use of
the matrix A necessarily comes from a gradient call). When we sample with replacement by a batch
size b, it is equivalent to that we sample without replacement by batch size 1 for b times. Assuming
that we have communication rounds with batch sizes (sampling without replacement) of 1, 2, . . . , n
for s1, s2, . . . , sn times, then the communication complexity we spent is V = Pn
j=1 jsj and the
minimum gradient calls we spent is C = Pn
j=1 jsj. This implies the expected number of non-zero

coordinates is M = Pn−1
j=1
j
n−1sj +sn. Therefore, the expected total number of non-zero coordinates
in the global output is at most M for y and M −1 for x (or we can say M). By comparing expressions
of V , C and M, we can have M ≤2V/n and M ≤2C/n, completing the proof.

Then we can prove Theorem 4 by combining Lemma 12 and 13.

E.3
Proof of Lemma 2

We choose the function set as

f(x, y) = fi(x, y) = L

2 x⊤Ay −LRy

2
√

d
e⊤
1 x,
∀i ∈[n],
(48)

where A is defined as equation (43). The structure of A results the following lemma.

Lemma 14. For the function set (48), all (x, y) ∈Fk × Fk and k = 0, . . . , d −1, we have

∇fi(x, y) ∈Fk+1 × Fk+1.

One can follow the similar method as the proof of Theorem 4 to prove the following proposition and
lemma for the proof of Lemma 2.

Proposition 4. For any d ≥3, fi(x, y) and f(x, y) defined as equation (48) satisfy

1. fi is L-smooth and convex-concave, function set {fi}n
i=1 has δ second-order similarity for any
δ > 0. Thus, f is convex-concave.

2. For 1 ≤k ≤d −1, we have

min
(x,y)∈Z∩F2
k
Gap(x, y) =
min
x∈X∩Fk max
y∈Y f(x, y) −
max
y∈Y∩Fk min
x∈X f(x, y) ≥
LRxRy
2
p

d(k + 1)
.

Lemma 15. Consider the minimax problem (40) with the function set (48), Rx = Ry = D,
d = ⌊δD2/(
√

2ε)⌋−1 and ε ≤δD2/(4
√

2). We let M = ⌊(d −1)/2⌋≥Ω(LD2/ε), then
any point (x, y) ∈Z ∩F2
M satisfies Gap(x, y) ≥ε.

Then we can prove Lemma 2 by combining Lemma 14 and 15.

E.4
Proof of Theorem 5

Proof. In the case of √nδ ≥Ω(L), the problem with with functions (45) and (46) in Theorem 4
implies the lower bound on the local gradient complexity of

Ω

n +
√nδD2

ε


= Ω

n + (√nδ + L)D2

ε


.

In the case of L ≥Ω(√nδ), the problem with function (48) in Lemma 2 implies the lower bound
on the local gradient complexity of Ω(LD2/ε) = Ω(n + (√nδ + L)D2/ε). Combining these two
cases, we achieve the lower bound on the local gradient complexity of Ω(n + (√nδ + L)D2/ε).

F
Proofs for Lower Bounds in Strongly-Convex-Strongly-Concave Case

We follow similar steps as in Appendix E to prove Theorem 7. Firstly we divide it into several
detailed theorems and lemmas as below and prove them one by one.

33


---Page Break---
Theorem 8. For any µ, δ, L > 0 with L ≥max{µ, δ} and n ≥2, there exist L-smooth and convex-
concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity such that the function
f(x, y) = 1

n
Pn
i=1 fi(x, y) is µ-strongly-convex-µ-strongly-concave. In order to find a solution of
problem (1) such that E[∥z−z∗∥2] ≤ε, any DFO algorithm needs at least Ω((n+√nδ/µ) log(1/ε))
communication complexity and Ω((n + √nδ/µ) log(1/ε)) local gradient calls.

Lemma 16. For any µ, δ, L > 0 with L ≥max{µ, δ} and n ≥2, there exist L-smooth and convex-
concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity such that the function
f(x, y) = 1

n
Pn
i=1 fi(x, y) is µ-strongly-convex-µ-strongly-concave. In order to find a solution of
problem (1) such that E[∥z −z∗∥2] ≤ε, any DFO algorithm needs at least Ω(L/µ log(1/ε)) local
gradient calls.
Theorem 9. For any µ, δ, L > 0 with L ≥max{µ, δ} and n ≥2, there exist L-smooth
and convex-concave functions f1, . . . , fn : Rdx × Rdy with δ-second-order similarity such that
the function f(x, y) =
1
n
Pn
i=1 fi(x, y) is µ-strongly-convex-µ-strongly-concave. In order to
find a solution of problem (1) such that E[∥z −z∗∥2] ≤ε, any DFO algorithm needs at least
Ω((n + (√nδ + L)/µ) log(1/ε)) local gradient calls.

F.1
Proof of Theorem 8

We introduce the function set as in [8], which is similar to equation (45) that

fi(x, y) =














µ

2 ∥x∥2 −µ

2 ∥y∥2 + δ2

16µe⊤
1 y,
i = 1,

δ√n

4
x⊤
"
X

j≡(i−1) mod (n−1)
eja⊤
j

#

y + µ

2 ∥x∥2 −µ

2 ∥y∥2,
i > 2.
(49)

Then corresponding global objective is

f(x, y) =
δ
4√nx⊤Ay + µ

2 ∥x∥2 −µ

2 ∥y∥2 +
δ2

16nµe⊤
1 y,
(50)

where all the notation keeps the same as the former section. We point out that the global objective
function (50) with local functions (49) satisfies Assumptions 1, 3, 4, 5, and 6, with constant µ, δ, and
L ≥δ. See Lemma C.8 of Beznosikov et al. [8] for details.

We provide the following lemmas for the proof of Theorem 8.
Lemma 17. Consider the minimax problem (40) with functions (49) and (50) and run any DFO
algorithm with V communication complexity and C local gradient calls. In expectation, only the first
M ≤min{2V/n, 2C/n} coordinates of the final output can be non-zero while the rest of the d −M
coordinates are strictly equal to zero.

Proof. Follow the proof of Lemma 13.

Lemma 18 (Beznosikov et al. [8, Theorem C.10]). Let µ, δ > 0, n ∈N, M ∈N. There exists
a centralized distributed saddle-point problem with functions (49) and (50), in which z∗̸= 0 and
d ≥max{2 logq(α/(4
√

2)), 2M} where α = 16nµ2/δ2 and q = (2 + α −
√

α2 + 4α)/2 ∈(0, 1).
Then for any output (ˆx, ˆy) of any DFO algorithm leaving d −M coordinates zero, one can obtain
the following estimate:

∥ˆx −x∗∥2 + ∥ˆy −y∗∥2 = Ω



exp



−
16

1 +
q

δ2
16µ2n + 1
· M



∥y0 −y∗∥2



.

Then we can prove Theorem 8 by combining Lemma 17 and 18 and noting that to reach a solution of
ε-accuracy requires

min
2V

n , 2C

n


≥M = Ω

1 +
δ
√nµ


log 1

ε


.

34


---Page Break---
F.2
Proof of Lemma 16

We introduce the function set as

f(x, y) = fi(x, y) = L

2 x⊤˜Ay + µ

2 ∥x∥2 −µ

2 ∥y∥2 + L2

4µe⊤
1 y,
∀i ∈[n],
(51)

where

˜A =







1
1
−1

...

...

1
−1





.

Note that since all the nodes have the same function, this function set (51) satisfies Assumptions 1, 3,
4, 5, and 6 for L, µ and any δ > 0. The structure of ˜A results the following lemma.

Lemma 19. For the function set (51), all (x, y) ∈Fk × Fk and k = 0, . . . , d −1, we have

∇fi(x, y) ∈Fk+1 × Fk+1.

Lemma 20. Let µ, δ, L > 0, n ∈N, C ∈N. There exists a centralized distributed saddle-point
problem with functions (51), in which z∗̸= 0 and d ≥max{2 logq(α/
√

2), 4C} where α = µ2/L2

and q = 1 + 2α −2
√

α2 + α ∈(0, 1). Then for any DFO algorithm, the output (xC, yC) after C
local gradient calls will satisfies

∥yC −y∗∥2 ≥qC · ∥y0 −y∗∥2

16
.
(52)

Proof. Follow the proof of Theorem 3.5 by Zhang et al. [55], in which we take Lx = Ly = Lxy = L
and µx = µy = µ.

Then we can prove Lemma 16 by combining Lemma 19 and 20 and noting that to reach a solution
with ε-accuracy needs at least Ω(L/µ log(1/ε)) local gradient calls from equation (52).

F.3
Proof of Theorem 9

Proof. In the case of nµ + √nδ ≥Ω(L), the problem with with functions (49) and (50) in Theorem
8 implies the lower bound on the local gradient complexity of

Ω

n +
√nδ

µ


log 1

ε


= Ω

n +
√nδ + L

µ


log 1

ε


.

In the case of L ≥Ω(nµ + √nδ), the problem with function (51) in Lemma 16 implies the lower
bound on the local gradient complexity of Ω(L/µ log(1/ε)) = Ω((n + (√nδ + L)/µ) log(1/ε)).
Combining these two cases, we can achieve the lower bound on the local gradient complexity of
Ω((n + (√nδ + L)/µ) log(1/ε)).

G
Making the Gradient Small

In constrained case, we can also use gradient mapping to measure the sub-optimality of a solution z,
that is

Fτ(z) = z −PZ(z −τF(z))

τ
.

For the L-smooth convex-concave function f, we consider the constrained problem

min
x∈X max
y∈Y
ˆf(x, y) := f(x, y) + λ

2

x −x02 −λ

2

y −y02 ,

35


---Page Break---
0
20
40
60
80
100
120
Communication Rounds

10
13

10
11

10
9

10
7

10
5

10
3

10
1

101

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
5000
10000
15000
20000
Communication Complexity

10
14

10
12

10
10

10
8

10
6

10
4

10
2

100

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

0
20000
40000
60000
80000 100000
Gradient Calls

10
13

10
11

10
9

10
7

10
5

10
3

10
1

101

Gradient Mapping

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 5: Results for convex-concave minimax problem (12) on covtype.

0
10
20
30
40
50
Communication Rounds

10
11

10
9

10
7

10
5

10
3

10
1

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
2500
5000
7500
10000
12500
Communication Complexity

10
10

10
8

10
6

10
4

10
2

100

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

0
20000
40000
60000
Gradient Calls

10
11

10
9

10
7

10
5

10
3

10
1

||z
z * ||2

EG
SMMDS
EGS
TPAPP
SVOGS

Figure 6: Results for strongly-convex-strongly-concave minimax problem (13) on covtype.

where ˆf is λ-strongly-convex-λ-strongly-concave and Z is bounded by diameter D. From the the
result of Theorem 2, we have E
 zK −ˆz∗2 
≤(1 −χλ/δ)K z0 −ˆz∗2 for some constant
χ ∈(0, 1), then we have

E

∥Fτ(zK)∥

= E



zK −PZ

zK −τ( ˆF(zK) −λ(zK −z0))


τ



= E



PZ(zK) −PZ

zK −τ( ˆF(zK) −λ(zK −z0))


τ



≤E
h ˆF(zK) −λ(zK −z0)

i

≤E
h ˆF(zK)

i
+ λE
zK −z0

= E
h ˆF(zK) −ˆF(ˆz∗)

i
+ λE
zK −z0

≤LE
zK −ˆz∗
+ λE
zK −z0
,

where we use the property of the projection that ∥PZ(z1) −PZ(z2)∥≤∥z1 −z2∥and the gradient
mapping holds that Fτ(ˆz∗) = 0. Then we have

E
hF(zK)
2i
≤2L2E
hzK −ˆz∗2i
+ 2λ2E
hzK −z02i

≤2L2(1 −χλ/δ)K z0 −ˆz∗2 + 2λ2E
hzK −z02i

≤(2L2(1 −χλ/δ)K + 2λ2)D2.
Let λ =
p

ε/(4D2) and K = O(δ/λ log(L/ε)), then we have E
 F(zK)
2 
≤ε.

Hence, the complexity of communication rounds is K = O (δD/√ε log(L/ε)). We verify that the
corresponding communication complexity is ˜O((n + √nδD/√ε) log(1/ε)) and the local gradient
complexity is ˜O((n + (√nδ + L)D/√ε) log(1/ε)) by following the discussion in Appendix C.5.

H
More Experimental Results

We present the experimental results on dataset “covtype” in Figure 5 and 6.

36


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: [Yes] See Section 1.

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

Justification: [Yes] See Section 8.

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

37


---Page Break---
Justification: [Yes] See Section 2 and Appendix A-G.
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
Justification: [Yes] See Section 7.
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

38


---Page Break---
Answer: [Yes]

Justification: [Yes] We include the code in the supplemental materials.

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

Justification: [Yes] See Section 7 and Appendix H.

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

Justification: [No] Error bars were not reported due to their potential to obscure the clarity
of convergence dynamics in the graph representation.

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

39


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

Justification: [Yes] See Section 7 and Appendix H.

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

Justification: [Yes]

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

Justification: [NA] The paper focuses on optimizing algorithmic enhancements unrelated to
potential social impacts.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

40


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

Justification: [NA]

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

Justification: [Yes] The original paper for used LIBSVM dataset is cited.

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

41


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: [NA]
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
Justification: [NA]
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
Justification: [NA]
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

42


---Page Break---
