On the Optimal Time Complexities in Decentralized
Stochastic Asynchronous Optimization

Alexander Tyurin
KAUST∗, AIRI†, Skoltech‡
Peter Richtárik
KAUST∗

Abstract

We consider the decentralized stochastic asynchronous optimization setup, where
many workers asynchronously calculate stochastic gradients and asynchronously
communicate with each other using edges in a multigraph. For both homogeneous
and heterogeneous setups, we prove new time complexity lower bounds under the
assumption that computation and communication speeds are bounded. We develop
a new nearly optimal method, Fragile SGD, and a new optimal method, Amelie
SGD, that converge under arbitrary heterogeneous computation and communication
speeds and match our lower bounds (up to a logarithmic factor in the homogeneous
setting). Our time complexities are new, nearly optimal, and provably improve all
previous asynchronous/synchronous stochastic methods in the decentralized setup.

1
Introduction

We consider the smooth nonconvex optimization problem

min
x∈Rd

n
f(x) := Eξ∼Dξ [f(x; ξ)]
o
,
(1)

where f : Rd × Sξ →R, and Dξ is a distribution on a non-empty set Sξ. For a given ε > 0, we
want to find a possibly random point ¯x, called an ε–stationary point, such that E[∥∇f(¯x)∥2] ≤ε.
We analyze the heterogeneous setup and the convex setup with smooth and non-smooth functions in
Sections C and D.

1.1
Decentralized setup with times

We investigate the following decentralized asynchronous setup. Assume that we have n workers/nodes
with the associated computation times {hi}, and communications times {ρi→j}. It takes less or
equal to hi ∈[0, ∞] seconds to compute a stochastic gradient by the ith node, and less or equal
ρi→j ∈[0, ∞] seconds to send directly a vector v ∈Rd from the ith node to the jth node (it is possible
that hi = ∞and ρi→j = ∞). All computations and communications can be done asynchronously and
in parallel. We would like to emphasize that hi ∈[0, ∞] and ρi→j ∈[0, ∞] are only upper bounds,
and the real and effective computation and communication times can be arbitrarily heterogeneous
and random. For simplicity of presentation, we assume the upper bounds are static; however, in
Section 5.5, we explain that our result can be trivially extended to the case when the upper bounds
are dynamic.

We consider any weighted directed multigraph parameterized by a vector h ∈Rn such that hi ∈
[0, ∞], and a matrix of distances {ρi→j}i,j ∈Rn×n such that ρi→j ∈[0, ∞] for all i, j ∈[n] and

∗King Abdullah University of Science and Technology, Thuwal, Saudi Arabia
†AIRI, Moscow, Russia
‡Skolkovo Institute of Science and Technology, Moscow, Russia

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
1
2

3
4
5

6
2

1
1

1

3
4
7

3
2

3

1
2

3
4
5

6

Figure 1: On the left: an example of a multigraph with n = 6. The edges with ρi→j = ∞are omitted.
The shortest distance between nodes 5 and 3 is τ5→3 = ρ5→1 + ρ1→2 + ρ2→3. Note that ρ5→3 = ∞.
On the right: an example of a spanning tree that illustrates the shortest paths from every node to
node 3. The shortest distance between nodes 6 and 3 is τ6→3 = ∞because ρ6→i = ∞for all i ̸= 6.

ρi→i = 0 for all i ∈[n]. Every worker i is connected to any other worker j with two edges i →j
and j →i. For this setup, it would be convenient to define the distance of the shortest path from
worker i to worker j :

τi→j :=
min
path∈Pi→j

X

(u,v)∈path
ρu→v ∈[0, ∞],
(2)

where Pi→j :=

[(k1, k2), . . . , (km, km+1)]
 ∀m ∈N ∀p ∈[m + 1] ∀kp ∈[n],

k1 = i, km+1 = j, ∀j ∈{2, . . . , m} kj−1 ̸= kj ̸= kj+1
	

is the set of all possible paths without loops from worker i to worker j for all i, j ∈[n]. One can
easily show that the triangle inequality τi→j ≤τi→k + τk→j holds for all i, j, k ∈[n]. Note that
τi→j ≤ρi→j for all i, j ∈[n]. It is important to distinguish τi→j and ρi→j because it is possible
that τi→j < ρi→j = ∞if workers i and j are connected by an edge ρi→j = ∞, and there is a path
through other workers (see Fig. 1).

We work with the following standard assumption from smooth nonconvex stochastic optimization
literature.

Assumption 1. f is differentiable and L–smooth, i.e., ∥∇f(x) −∇f(y)∥≤L ∥x −y∥∀x, y ∈Rd.

Assumption 2. There exist f ∗∈R such that f(x) ≥f ∗for all x ∈Rd.

Assumption 3. For all x ∈Rd, stochastic gradients ∇f(x; ξ) are unbiased and σ2-variance-
bounded, i.e., Eξ[∇f(x; ξ)] = ∇f(x) and Eξ[∥∇f(x; ξ) −∇f(x)∥2] ≤σ2, where σ2 ≥0. We
also assume that computation and communication times are statistically independent of stochastic
gradients.

2
Previous Results

2.1
Time complexity with one worker

For the case when n = 1 and τ1→1 = 0, convergence rates and time complexities of problem (1)
are well-understood. It is well-known that the stochastic gradient method (SGD), i.e., xk+1 = xk −
γ∇f(xk; ξk), where {ξk} are i.i.d. from Dξ, has the optimal oracle complexity Θ(L∆/ε + σ2L∆/ε2)
(Ghadimi and Lan, 2013; Arjevani et al., 2022). Assuming that the computation time of one stochastic
gradient is bounded by h1, we can conclude that the optimal time complexity is

T τ=0
single := Θ

h1 ×

L∆

ε + σ2L∆

ε2

(3)

seconds in the worst case.

2.2
Parallel optimization without communication costs

Assume that n > 1 and τi→j = 0 for all i, j ∈[n], and computation times of stochastic stochastic
gradients are arbitrarily heterogeneous. The simplest baseline method in this setup is Minibatch SGD,

2


---Page Break---
Table 1: Homogeneous Case (1). The time complexities to get an ε-stationary point in the nonconvex
setting. We assume that τi→j = τj→i for all i, j ∈[n] in this table. Abbr.: σ2 is defined as
Eξ[∥∇f(x; ξ) −∇f(x)∥2] ≤σ2 for all x ∈Rd, L is a smoothness constant of f, ∆:= f(x0) −f ∗.

Method
The Worst-Case Time Complexity Guarantees
Comment

Minibatch SGD
max

max
i,j∈[n] τi→j, max
i∈[n] hi

 
L∆

ε
+ σ2L∆

nε2

Suboptimal since, for instance, it
“linearly”(c) depends on max
i∈[n] hi

SWIFT
(Bornstein et al., 2023)
—(b)
Suboptimal since, for instance, it
“linearly”(c) depends on max
i∈[n] hi

Asynchronous SGD
(Even et al., 2024)
—(b)
Suboptimal, for instance,
even if τi→j = 0 ∀i, j ∈[n]

Fragile SGD
(Corollary 1)
L∆

ε
min
j∈[n] t∗(σ2/ε, [hi]n
i=1, [τi→j]n
i=1)(a)
Optimal up to log n factor

Lower Bound
(Theorem 1)
1
log n+1
L∆

ε
min
j∈[n] t∗(σ2/ε, [hi]n
i=1, [τi→j]n
i=1)(a)
—

(a) The mapping t∗is defined in Definition 2.
(b) It is not trivial to infer the time complexities for these methods. However, in Section 5.4, we discuss some cases where it is
transparent that the obtained results are suboptimal.
(c) Meaning that the corresponding time complexity →∞if maxi∈[n] hi →∞.

i.e.,

xk+1 = xk −γ

n

nP

i=1
∇f(xk; ξk
i ),
(4)

where {ξk
i } are i.i.d. from Dξ and the gradient ∇f(xk; ξk
i ) is calculated in worker i in parallel. This
method waits for stochastic gradients from all workers; thus, it is not robust to “stragglers” and in the
worst case the time complexity of such an algorithm is

T τ=0
mini := Θ

max
i∈[n] hi ×

L∆

ε + σ2L∆

nε2

,

which depends on the time maxi∈[n] hi of the slowest worker. There are many other more advanced
methods including Picky SGD (Cohen et al., 2021), Asynchronous SGD (e.g., (Recht et al., 2011;
Nguyen et al., 2018; Mishchenko et al., 2022; Koloskova et al., 2022)), and Rennala SGD (Tyurin
and Richtárik, 2023) that are designed to be robust to workers’ chaotic computation times. Under
the assumption that the computation times of the workers are heterogeneous and bounded by {hi},
Tyurin and Richtárik (2023) showed that Rennala SGD is the first method that achieves the optimal
time complexity

T τ=0
Rennala := Θ

 

min
m∈[n]

"
1
m

m
P

i=1

1
hπi

−1 
L∆

ε + σ2L∆

mε2
#!

,
(5)

where π is a permutation that sorts hi : hπ1 ≤· · · ≤hπn. For instance, one can see that T τ=0
Rennala ≤
T τ=0
mini for all parameters.

2.3
Parallel optimization with communication costs τi→j

We now consider the setup where workers’ communication times can not be ignored. This problem
leads to a research field called decentralized optimization. This setup is the primary case for us. All
n workers calculate stochastic gradients in parallel and communicate with each other. Numerous
works consider this setup, and we refer to Yang et al. (2019); Koloskova (2024) for detailed surveys.
Typically, methods in this setting use the gossip matrix framework (Duchi et al., 2011; Shi et al.,
2015; Koloskova et al., 2021) and get an iteration converge rate that depends on the spectral gap of a
mixing matrix. However, such rates do not give the physical time of algorithms (see also Section B).

Let us consider a straightforward baseline: Minibatch SGD. We can implement (4) in a way that all
workers calculate one stochastic gradient (takes at most maxi∈[n] hi seconds) and then aggregate
them to one pivot worker j∗(takes at most maxi∈[n] τi→j∗seconds). Then, pivot worker j∗calculates

3


---Page Break---
Table 2: Heterogeneous Case (17). Time complexities to get an ε-stationary point in the nonconvex
setting. Abbr.: σ2 is defined as Eξ[∥∇fi(x; ξ) −∇fi(x)∥2] ≤σ2 for all x ∈Rd, i ∈[n], L is a
smoothness constant of f = 1

n
Pn
i=1 fi, ∆:= f(x0) −f ∗.

Method
The Worst-Case Time Complexity Guarantees
Comment

Minibatch SGD
L∆

ε
max

1 + σ2

nε

max{ max
i,j∈[n] τi→j, max
i∈[n] hi}

suboptimal if σ2/ε is large

RelaySGD, Gradient Tracking
(Vogels et al., 2021)
(Liu et al., 2024)
≥

max
i∈[n]
Li∆

ε
σ2
nε max
i∈[n] hi
requires local Li-smooth. of fi,
suboptimal if σ2/ε is large
(even if maxi∈[n] Li = L)

Asynchronous SGD
(Even et al., 2024)
—
requires similarity of the functions {fi},
requires local Li-smooth. of fi

Amelie SGD and Lower Bound
(Thm. 7 and Cor. 2)
L∆

ε
max

max
i,j∈[n] τi→j, max
i∈[n] hi, σ2

nε


1
n

n
P

i=1
hi


Optimal up to a constant factor

a new point xk+1 and broadcasts it to all workers (takes maxi∈[n] τj∗→i seconds). One can easily
see that the time complexity of such a procedure is4

Tmini := Θ

max

max
i,j∈[n] τi→j, max
i∈[n] hi

 
L∆

ε + σ2L∆

nε2

.
(6)

We can analyze any other asynchronous decentralized method, which will be done with more advanced
methods in Section 5.4.

But what is the best possible (optimal) time complexity we can get in the setting from Section 1.1?

Unlike the setups from Sections 2.1 and 2.2 when the communication times are zero (τi→j = 0 for
all i, j ∈[n]), the optimal time complexity and an optimal method for the case τi→j ≥0 for all
i, j ∈[n] are not known. Our main goal in this paper is to solve this problem.

3
Contributions

We consider the class of functions that satisfy the setup and the assumptions from Section 1.1
and show that (informally) it is impossible to develop a method that will converge faster than (7)
seconds. Next, we develop a new asynchronous stochastic method, Fragile SGD, that is nearly
optimal (i.e., almost matches this lower bound; see Table 1 and Corollary 1). This is the first such
method. It provably improves on Asynchronous SGD (Even et al., 2024) and all other synchronous and
asynchronous methods (Bornstein et al., 2023). We also consider the heterogeneous setup (see Table 2
and Section C), where we discover the optimal time complexity by proving another lower bound and
developing a new optimal method, Amelie SGD, with weak assumptions. The developed methods
can guarantee the iteration complexity O (L∆/ε) with arbitrarily heterogeneous random computation
and communication times (Theorems 4 and 8). Our findings are extended to the convex setup in
Section D, where we developed new accelerated methods, Accelerated Fragile SGD and Accelerated
Amelie SGD.

4
Lower Bound

In order to construct our lower bound, we consider any (zero-respecting) method that can be repre-
sented by Protocol 1. This protocol captures all virtually distributed synchronous and asynchronous
methods, such as Minibatch SGD, SWIFT (Bornstein et al., 2023), Asynchronous SGD (Even et al.,
2024), and Gradient Tracking (Koloskova et al., 2021).

For all such methods we prove the following theorem.
Theorem 1 (Lower Bound; Simplified Presentation of Theorem 19). Consider Protocol 1 with
∇f(·; ·). We take any hi ≥0 and τi→j ≥0 for all i, j ∈[n] such that τi→j ≤τi→k + τk→j for all
i, k, j ∈[n]. We fix L, ∆, ε, σ2 > 0 that satisfy the inequality ε < cL∆for some universal constant

4because maxi,j∈[n] τi→j ≤maxi∈[n] τi→j∗+ maxi∈[n] τj∗→i ≤2 maxi,j∈[n] τi→j by the triangle in-
equality

4


---Page Break---
c. For any (zero-respecting) algorithm, there exists a function f, which satisfy Assumptions 1, 2 and
f(0) −f ∗≤∆, and a stochastic gradient mapping ∇f(·; ·), which satisfies Assumption 3, such that
the required time to find ε–solution is

Ω

1
log n+1
L∆

ε min
j∈[n] t∗(σ2/ε, [hi]n
i=1, [τi→j]n
i=1)

(7)

Def 2
≡Ω

 

1
log n+1
L∆

ε min
j∈[n] min
k∈[n] max

(

max{τπj,k→j, hπj,k}, σ2

ε

 kP

i=1

1
hπj,i

−1)!

,
(8)

where, for j ∈[n], πj,· is a permutation that sorts {max{τi→j, hi}}n
i=1, i.e., max{τπj,1→j, hπj,1} ≤
· · · ≤max{τπj,n→j, hπj,n}.

Protocol 1 Simplified Presentation of Protocol 8

1: Init Si = ∅(all available information) on worker i for all i ∈[n]
2: Run the following two loops in each worker in parallel
3: while True do
4:
Calculate a new point xk
i based on Si
(takes 0 seconds)
5:
Calculate a stochastic gradient ∇f(xk
i ; ξ) (or ∇fi(xk
i ; ξ))
ξ ∼Dξ
(takes hi seconds)
6:
Atomic add ∇f(xk
i ; ξ) (or ∇fi(xk
i ; ξ)) to Si
(atomic operation, takes 0 seconds)
7: end while
8: while True do
9:
Send(a) any vector from Rd based on Si to any worker j and go to the next step of this loop
without waiting
(takes τi→j seconds to send; worker j adds this vector to Sj)
10: end while
(a): When we prove the lower bounds, we allow algorithms to send as many vectors as they want in parallel
from worker i to worker j for all i ̸= j ∈[n].

The intuition and meaning of the formula (8) is discussed in Section 5.2. Note that if we take n = 1
and τ1→1 = 0 our lower bound reduces to the lower bound (3) up to a log factor. Moreover, if we
take n > 1 and τi→j = 0 for all i, j ∈[n], then (8) reduces to (5) up to a log factor. Thus, (8) is
nearly consistent with the lower bounds from (Arjevani et al., 2022; Tyurin and Richtárik, 2023). We
get an extra log n factor due to the generality of our setup. The reason is technical, and we explain
it in Section E.5. In a nutshell, the lower problem reduces to the analysis of the concentration of
the time series yT := minj∈[n] yT
j and yT
j := mini∈[n]

yT −1
i
+ hiηT
i + τi→j
	
, where y0
i = 0 for
all i ∈[n], and {ηk
i } are i.i.d. geometric random variables. This analysis is not trivial due to the
mini∈[n] operations. Virtually all previous works that analyzed lower bounds did not have such a
problem because they analyzed time series with a sum structure (e.g., ¯yT := ¯yT −1 + ϱT , where {ϱk}
are some random variables, and ¯y0 = 0).

Let us define an auxiliary function to simplify readability.
Definition 2 (Equilibrium Time). A mapping t∗: R≥0 × Rn
≥0 × Rn
≥0 →R≥0 with inputs s
(scalar), [hi]n
i=1 (vector), and [¯τi]n
i=1 (vector) is called the equilibrium time if it is defined as follows.
Find a permutation5 π that sorts max{¯τi, hi} as max{¯τπ1, hπ1} ≤· · · ≤max{¯τπn, hπn}. Then the
mapping returns the value

t∗(s, [hi]n
i=1, [¯τi]n
i=1) ≡min
k∈[n] max

(

max{¯τπk, hπk}, s
 kP

i=1

1
hπi

−1)

∈[0, ∞].
(9)

5
New Method: Fragile SGD

We introduce a novel optimization method characterized by time complexities that closely align with
the lower bounds established in Section 4. Our algorithms leverage spanning trees. A spanning tree
is a tree (undirected unweighted graph) encompassing all workers. The edges of spanning trees are
virtual and not related to the edges defined in Section 1.1 (see Fig. 1).

5It is possible that a permutation is not unique, then the result of the mapping does not depend on the choice.

5


---Page Break---
Algorithm 2 Fragile SGD

1: Input: starting point x0, stepsize γ, batch size S, pivot worker j∗, spanning trees st and stbc
2: Start Process 0 (Algorithm 3) in worker j∗

3: Start Process i (Algorithm 4) in all workers for all i ∈[n] (including worker j∗)

Algorithm 3 Process 0 (running in worker j∗)

1: for k = 0, 1, . . . , K −1 do
2:
Send xk to Process j∗
▷takes τj∗→j∗= 0 seconds
3:
Init (gk, sk) = (0, 0)
4:
while s < S do
5:
Wait for a message (gk
j∗,send, sk
j∗,send) from Process j∗

6:
gk = gk + gk
j∗,send;
sk = sk + sk
j∗,send
7:
end while
8:
xk+1 = xk −γ

sk gk

9: end for

Algorithm 4 Process i (running in worker i)

1: Init (gk
i,next, sk
i,next) = (0, 0) for all k ∈{0, . . . , K −1}(a), kmax = 0
2: Run the following three functions in parallel
3: function BroadcastFurtherAndCalculateStochasticGradients
4:
while True
5:
Get a new point x¯k sent by Process nextstbc,j∗(i)
▷¯k is not necessarily equals to the current k from Alg. 3
6:
Atomic update kmax = max{¯k, kmax}
7:
for all p such that nextstbc,j∗(p) = i do
▷broadcasts x¯k further

8:
Send(b) x¯k to Process p and go to the next step without waiting ▷takes at most τi→p seconds to send
9:
end for
10:
while not received a new point
▷immediately stops the loop when receives a new point
11:
Calculate ∇f(x¯k; ξ),
ξ ∼D
▷takes at most hi second
12:
Run atomic add g¯k
i,next = g¯k
i,next + ∇f(x¯k; ξ), s¯k
i,next = s¯k
i,next + 1
13:
end while
14:
end while
15: end function
16: function ReceiveVectorsFromPreviousWorkers
17:
while True
18:
Wait for a message (gˆk
p,send, sˆk
p,send) from any Process p such that nextst,j∗(p) = i

19:
Run atomic add gˆk
i,next = gˆk
i,next + gˆk
p,send, sˆk
i,next = sˆk
i,next + sˆk
p,send
20:
Atomic update kmax = max{ˆk, kmax}
21:
end while
22: end function
23: function SendVectorsToNextWorker
24:
while True
25:
Atomic init gkmax
i,send, skmax
i,send = gkmax
i,next, skmax
i,next and reset gkmax
i,next = 0, skmax
i,next = 0

26:
Send(b) (gkmax
i,send, skmax
i,send) to Process nextst,j∗(i) and wait
▷takes at most τi→nextst,j∗(i) seconds to wait
27:
end while
28: end function

(a): To simplify the listing of the algorithm, we assume here that a worker can store K auxiliary vectors in the
memory. One can see that it’s not necessary, and it is sufficient to maintain only one vector gkmax
i,next . In particular,
it is sufficient to modify the logic of Lines 12 and 19, where we run the add operations only if ¯k = kmax and
ˆk = kmax. The efficient implementation has O(d) floats memory complexity per worker.
(b): BroadcastFurtherAndCalculateStochasticGradients and SendVectorsToNextWorker may try to send through
the same edge. In this case, we can interleave their communications and decrease the speed of each line by at
most two.

6


---Page Break---
Definition 3 (mapping nextT,j(i)). Take a spanning tree T and fix any worker j ∈[n]. For i = j, we
define nextT,j(i) = 0. For all i ̸= j ∈[n], we define nextT,j(i) as the index of the next worker on
the path of the spanning tree T from worker i to worker j.

Our new method, Fragile SGD, is presented in Algorithms 2, 4, and 3. While Fragile SGD seems to be
lengthy, the idea is pretty simple. All workers do three jobs in parallel: calculate stochastic gradients,
receive vectors, and send vectors through spanning trees. A pivot worker aggregates all stochastic
gradients in gk and, at some moment, does xk+1 = xk −γgk. The algorithms formalize this idea.

Algorithm 2 requires a starting point x0, a stepsize γ, a batch size S, the index j∗of a pivot worker,
and spanning trees st and stbc for the input. We need two spanning stbc and st trees because, in
general, the fastest communication of a vector from j∗to i and from i to j∗should be arranged
through two different paths. Algorithm 2 starts n + 1 processes running in parallel. Note that the
pivot worker j∗runs two parallel processes, called Process 0 and Process j∗, and any other worker i
runs one Process i. Process 0 broadcasts a new point xk through stbc to all other processes and goes
to the loop where it waits for messages from Process j∗. Process i starts three functions that will
be running in parallel: i) the first function’s job is to receive a new point, broadcast it further, and
start the calculation of stochastic gradients, ii) the second function receives stochastic gradients from
all previous processes that are sending vectors to worker j∗, iii) the third function sends vectors the
next worker on the path to j∗. By the definition of nextst,j∗(·), all calculated stochastic vectors are
sent to worker j∗, where they are first aggregated in Process j∗, and then, since nextst,j∗(j∗) = 0,
Process j∗will send gk
j∗,send to Process 0. This process waits for the moment when the number of
stochastic gradients sk aggregated in gk is greater or equal to S. When it happens, the loop stops,
and Process 0 does a gradient-like step. The structure of the algorithm and the idea of spanning trees
resemble the ideas from (Vogels et al., 2021; Tyurin and Richtárik, 2023). The main observation is
that this algorithm is equivalent to xk+1 = xk −γ

sk
Psk

i=1 ∇f(xk; ξk
i ), where sk ≥S and {ξk
i } are
i.i.d. samples. Note that all stochastic gradients calculated at points x0, . . . , xk−1 will be ignored in
the kth iteration of Algorithm 3.
Theorem 4. Let Assumptions 1, 2, and 3 hold. We take γ = 1/2L, batch size S = max{

σ2/ε

, 1},
any pivot worker j∗∈[n], and any spanning trees st and stbc in Algorithm 2. For all K ≥16L∆/ε,

we get 1

K
PK−1
k=0 E
h∇f(xk)
2i
≤ε.

The proof is simple and uses standard techniques from (Lan, 2020; Khaled and Richtárik, 2022). The
result of the theorem holds even if hi = ∞and τi→j = ∞for all i, j ∈[n] because hi and τi→j
are only upper bounds on the real computation and communications speeds. In Algorithm 3, each
iteration k can be arbitrarily slow, and still, the result of Theorem 4 holds and the method converges
after O(L∆/ε) iterations. The next result gives time complexity guarantees for our algorithm.
Theorem 5. Consider the assumptions and the parameters from Theorem 4. For any pivot worker
j∗∈[n] and spanning trees st and stbc, Algorithm 2 converges after at most

Θ
  L∆

ε t∗(σ2/ε, [hi]n
i=1, [µi→j∗+ µj∗→i]n
i=1)

(10)

seconds, where µi→j∗(µj∗→i) is an upper bound on the times required to send a vector from worker
i to worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc).

Note that our method does not need the knowledge of {hi} and {µi→j} to guarantee the time
complexity rate, and it automatically obtains it.
Corollary 1. Consider the assumptions and the parameters from Theorem 5. Let us take a pivot
worker j∗= arg min
j∈[n] t∗(σ2/ε, [hi]n
i=1, [τi→j +τj→i]n
i=1), and a spanning tree st (spanning tree stbc)

that connects every worker i to worker j∗(worker j∗to every worker i) with the shortest distance
τi→j∗(τj∗→i). Then Algorithm 2 converges after at most

T∗:= Θ

L∆

ε min
j∈[n] t∗(σ2/ε, [hi]n
i=1, [τi→j + τj→i]n
i=1)

(11)

Def 2
≡Θ

 

L∆

ε min
j∈[n] min
k∈[n] max

(

max{τπj,k→j + τj→πj,k, hπj,k}, σ2

ε

 kP

i=1

1
hπj,i

−1)!

(12)

seconds, where, for all j ∈[n], πj,· is a permutation that sorts {max{τi→j + τj→i, hi}}n
i=1.

7


---Page Break---
This corollary has better time complexity guarantees than Theorem 5 because, by the definition of
τi→j, τi→j ≤µi→j for al i, j ∈[n]. However, it requires the particular choice of a pivot worker and
spanning trees.

5.1
Discussion

Comparing the lower bound (7) with the upper bound (11), one can see that Fragile SGD has a nearly
optimal time complexity. If we ignore the log n factor in (7) and assume that τi→j = τj→i for all
i, j ∈[n], which is a weak assumption in many applications, then Fragile SGD is optimal.

Unlike most works (Even et al., 2024; Lian et al., 2018; Koloskova et al., 2021) in the decentralized
setting, our time complexity guarantees do not depend on the spectral gap of the mixing matrix that
defines the topology of the multigraph. The structure of the multigraph is coded in the times {τi→j}.
We believe that this is an advantage of our guarantees since (11) defines a physical time instead of an
iteration rate that depends on the spectral gap.

5.2
Interpretation of the upper and lower bounds (11) and (7)

One interesting property of our algorithm is that some workers will potentially never contribute to
optimization because either their computations are too slow or communication times to j∗are too
large. Thus, only a subset of the workers should work to get the optimal time complexity!

Assume in this subsection that the computation and communication are fixed to {hi} and {τi→j}.
One can see that (12) is the minj∈[n] mink∈[n] over some formula. In view of our algorithm, an
index j∗that minimizes in (12) is the index of a pivot worker that is the most “central” in the
multigraph. An index k∗that minimizes mink∈[n] defines a set of workers {πj∗,1, . . . πj∗,k∗}
that can potentially contribute to optimization.
The algorithm and and the time complexity
will not depend on workers {πj∗,k∗+1, . . . πj∗,n} because they are too slow or they are too far
from worker j∗. Thus, up to a constant factor, we have T∗=
L∆/ε max

max{τπj∗,k∗→j∗+

τj∗→πj∗,k∗, hπj∗,k∗}, σ2/ε
  Pk∗

i=1 1/hπj∗,i
−1	
, where τπj∗,k∗→j∗+ τj∗→πj∗,k∗is the time required
to communicate with the farthest worker that can contribute to optimization, hπj∗,k∗is the computa-

tion time of the slowest worker that can contribute to optimization, and σ2/ε
  Pk∗

i=1 1/hπj∗,i
−1 is the
time required to “eliminate” enough noise before the algorithm does an update of xk.

5.3
Limitations

To get the nearly optimal complexity, it is crucial to select the right pivot worker j∗and spanning
trees according to the rules of Corollary 1, which depend on the knowledge of the bounds of times.
For now, we believe that is this a price for the optimality. Note that Theorem 5 does not require this
knowledge and it works with any j∗and any spanning tree; thus, we can use any heuristic to estimate
an optimal j∗and optimal spanning trees. One possible strategy is to estimate the performance of
workers and the communication channels using load testings.

5.4
Comparison with previous methods

Let us discuss the time complexities of previous methods. Note that none of the previous methods
can converge faster than (7) due to our lower bound. First, consider (6) of Minibatch SGD. This time
complexity depends on the slowest computation time maxi∈[n] hi and the slowest communication
times maxi,j∈[n] τi→j. In the asynchronous setup, it is possible that one the workers is a straggler,
i.e., maxi∈[n] hi ≈∞, and Minibatch SGD can be arbitrarily slow. Our time complexities (10) and
(12) are robust to stragglers, and ignore them. Assume the last worker n is a straggler and hn = ∞,
then one can take permutations with πj,n = n for all j ∈[n], and the minimum operator mink∈[n] in
(12) will not choose k = n because max{τπj,n→j + τj→πj,n, hπj,n} = ∞for all j ∈[n].

We now consider a recent work by Even et al. (2024), where the authors analyzed Asynchronous SGD
in the decentralized setting. In the homogeneous setting, their converge rate depends on the maximum
compute delay and, thus, is not robust to stragglers. For the case τi→j = 0, our time complexity (11)
reduces to (5). At the same time, it was shown (Tyurin and Richtárik, 2023) that the time complexity
of Asynchronous SGD for τi→j = 0 is strongly worse than (5); thus, the result by Even et al. (2024) is

8


---Page Break---
suboptimal in our setting even if τi→j = 0 for all i, j ∈[n]. The papers by Bornstein et al. (2023);
Lian et al. (2018) also consider the same setting, and they share a logic in that they sample a random
worker and wait while it is calculating a stochastic gradient. If one of the workers is a straggler, they
can wait arbitrarily long, while our method automatically ignores slow computations.

5.5
Time complexity with dynamic bounds

We can easily generalize Theorem 5 to the case when bounds on the times are not static.

Theorem 6. Consider the assumptions and the parameters from Theorem 4. In each iteration k
of Algorithm 3, the computation times of worker i are bounded by hk
i . Let us fix any pivot worker
j∗∈[n] and any spanning trees st and stbc. Then Algorithm 2 converges after at most

Θ
P⌈16L∆/ε⌉
k=0
t∗(σ2/ε, [hk
i ]n
i=1, [µk
i→j∗+ µk
j∗→i]n
i=1)

(13)

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc) in iteration k
of Algorithm 3.

This result is more general than (10), and it shows that our method is robust to changing com-
putation {hk
i } and communication {µk
i→j} times bounds during optimization processes. For in-
stance, worker n can have either slow computation or communication to j∗in the first iteration,
i.e., max{h1
n, µ1
n→j∗, µ1
j∗→n} ≈∞, then our method will ignore it, but if max{h2
n, µ2
n→j∗, µ2
j∗→n}
is small in the second iteration, then our method can potentially use the stochastic gradients from
worker n.

6
Example: Line or Circle

1
2
3
4
5
6
7
ρ
ρ
ρ
ρ
ρ
ρ

τ3→5 = 2ρ

τ3→6 = 3ρ

τ3→2 = ρ

Figure 2: Line with ρi+1→i = ρi→i+1 = ρ for all i ∈[n −1], ρi→j = ∞otherwise. For all
i ̸= j ∈[n], edges i →j and j →i are merged and visualized with one undirected edge.

Let us consider Line graphs where we can get more explicit and interpretable formulas for (11). We
analyze ND-Mesh, ND-Torus, and Star graphs in Section A. Surprisingly, even in some simple cases
like Line or Star graphs, as far as we know, we provide new time complexity results and insights. In
Section J, we show that our theoretical results are supported by computational experiments.

We take a Line graph with the computation speeds hi = h for all i ∈[n], and the communication
speeds of the edges ρi→i+1 = ρi+1→i = ρ for all i ∈[n −1] and ρi→j = ∞for all other i, j ∈[n].
One can easily show the time required to send a vector between two workers i, j ∈[n] equals
τi→j = τj→i = ρ|i −j|. See an example with n = 7 in Fig. 2. We can substitute these values to (11)
and get

Tline = L∆

ε min
j∈[n] min
k∈[n] max
n
max{ρ|j −πj,k|, h}, σ2h

εk
o
,
(14)

where πj,1 = j, πj,2, πj,3 = j + 1, j −1 or πj,2, πj,3 = j −1, j + 1 (only for n −1 ≥j ≥2) and
so forth. For simplicity, assume that n is odd, then, clearly, j∗= n−1

2
+ 1 minimizes minj∈[n] and
Tline =

L∆

ε
min
d∈{0,..., n−1

2
}
max
n
ρd, h,
σ2h
ε(2d+1)
o
≃L∆

ε



h +








σ2h/ε,
if
p

σ2h/ερ ≤1,
p

ρσ2h/ε,
if n >
p

σ2h/ερ > 1,

σ2h/nε,
if
p

σ2h/ερ ≥n



.
(15)

9


---Page Break---
According to (15), there are three time complexity regimes: i) slow communication, i.e.,
p

σ2h/ερ ≤
1, this inequality means that ρ is so large, that communication between workers will not increase the
convergence speed, and the best strategy is to work with only one worker!, ii) medium communication,
i.e., n >
p

σ2h/ερ > 1, more than one worker will participate in the optimization process; however,
not all of them!, some workers will not contribute since their distances τj∗→· to the pivot worker j∗

are large, iii) fast communication, i.e.,
p

σ2h/ερ ≥n, all n workers will participate in optimization
because ρ is small.

As far as we know, the result (15) is new even for such a simple structure as a line. Note that these
regimes are fundamental and can not be improved due to our lower bound (up to logarithmic factors).
For Circle graphs, the result is the same up to a constant factor.

7
Heterogeneous Setup

In Section C (in more details), we consider and analyze the problem

min
x∈Rd

n
f(x) := 1

n

n
X

i=1
Eξi∼Di [fi(x; ξi)]
o
,

where fi : Rd × Sξi →Rd and ξi are random variables with some distributions Di on Sξi. For all
i ∈[n], worker i can only access fi. We show that the optimal time complexity is

Θ

L∆

ε max

max
i,j∈[n] µi→j, max
i∈[n] hi, σ2

nε


1
n

nP

i=1
hi


(16)

in the heterogeneous setting achieved by a new method, Amelie SGD (Algorithm 5). Amelie SGD
is closely related to Rennala SGD but with essential algorithmic changes to make it work with
heterogeneous functions. The obtained complexity (16) is worse than (12), which is expected because
the heterogeneous setting is more challenging than the homogeneous setting.

8
Highlights of Experiments

In Section J, we present experiments with quadratic optimization problems, logistic regression, and a
neural network to substantiate our theoretical findings. Here, we focus on highlighting the results
from the logistic regression experiments:

0.0
0.2
0.4
0.6
0.8
1.0
times (seconds)
1e6

2.2 × 10
1

2.4 × 10
1

2.6 × 10
1

2.8 × 10
1

3 × 10
1

f(xt)
f(x * )

Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 0.5

0.0
0.2
0.4
0.6
0.8
1.0
times (seconds)
1e6

0.910

0.915

0.920

0.925

0.930

0.935

0.940

Accuracy (Test)

Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 0.5

Figure 3: The communication time ρ = 10 seconds (Slow communication) in 2D-Mesh

On MNIST dataset (LeCun et al., 2010) with 100 workers, Fragile SGD is much faster and has better
test accuracy than Minibatch SGD.

Acknowledgments and Disclosure of Funding

The research reported in this publication was supported by funding from King Abdullah University of
Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) Center of Excellence
for Generative AI, under award number 5940, iii) SDAIA-KAUST Center of Excellence in Artificial
Intelligence and Data Science. The work of A.T. was partially supported by the Analytical center
under the RF Government (subsidy agreement 000000D730321P5Q0002, Grant No. 70-2021-00145
02.11.2021).

10


---Page Break---
References

Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. (2022). Lower
bounds for non-convex stochastic optimization. Mathematical Programming, pages 1–50.

Bornstein, M., Rabbani, T., Wang, E., Bedi, A. S., and Huang, F. (2023). SWIFT: Rapid decentralized
federated learning via wait-free model communication. In The 11th International Conference on
Learning Representations (ICLR).

Carmon, Y., Duchi, J. C., Hinder, O., and Sidford, A. (2020). Lower bounds for finding stationary
points i. Mathematical Programming, 184(1):71–120.

Cohen, A., Daniely, A., Drori, Y., Koren, T., and Schain, M. (2021). Asynchronous stochastic
optimization robust to arbitrary delays. Advances in Neural Information Processing Systems,
34:9024–9035.

Duchi, J. C., Agarwal, A., and Wainwright, M. J. (2011). Dual averaging for distributed optimization:
Convergence analysis and network scaling. IEEE Transactions on Automatic control, 57(3):592–
606.

Even, M., Koloskova, A., and Massoulié, L. (2024). Asynchronous SGD on graphs: A unified
framework for asynchronous decentralized and federated optimization. In Artificial Intelligence
and Statistics. PMLR.

Floyd, R. W. (1962). Algorithm 97: shortest path. Communications of the ACM, 5(6):345–345.

Ghadimi, S. and Lan, G. (2013). Stochastic first-and zeroth-order methods for nonconvex stochastic
programming. SIAM Journal on Optimization, 23(4):2341–2368.

Huang, X., Chen, Y., Yin, W., and Yuan, K. (2022). Lower bounds and nearly optimal algorithms in
distributed learning with communication compression. Advances in Neural Information Processing
Systems (NeurIPS).

Khaled, A. and Richtárik, P. (2022). Better theory for SGD in the nonconvex world. Transactions on
Machine Learning Research.

Koloskova, A. (2024). Optimization algorithms for decentralized, distributed and collaborative
machine learning. Technical report, EPFL.

Koloskova, A., Lin, T., and Stich, S. U. (2021). An improved analysis of gradient tracking for
decentralized machine learning. Advances in Neural Information Processing Systems, 34:11422–
11435.

Koloskova, A., Stich, S. U., and Jaggi, M. (2022). Sharper convergence guarantees for asynchronous
SGD for distributed and federated learning. Advances in Neural Information Processing Systems
(NeurIPS).

Lan, G. (2020). First-order and stochastic optimization methods for machine learning. Springer.

LeCun, Y., Cortes, C., and Burges, C. (2010). Mnist handwritten digit database. ATT Labs [Online].
Available: http://yann.lecun.com/exdb/mnist, 2.

Lian, X., Zhang, W., Zhang, C., and Liu, J. (2018). Asynchronous decentralized parallel stochastic
gradient descent. In International Conference on Machine Learning, pages 3043–3052. PMLR.

Liu, Y., Lin, T., Koloskova, A., and Stich, S. U. (2024). Decentralized gradient tracking with local
steps. Optimization Methods and Software, pages 1–28.

Lu, Y. and De Sa, C. (2021). Optimal complexity in decentralized training. In International
Conference on Machine Learning, pages 7111–7123. PMLR.

Mishchenko, K., Bach, F., Even, M., and Woodworth, B. (2022). Asynchronous SGD beats minibatch
SGD under arbitrary delays. Advances in Neural Information Processing Systems (NeurIPS).

Nesterov, Y. (1983). A method of solving a convex programming problem with convergence rate o
(1/k** 2). Doklady Akademii Nauk SSSR, 269(3):543.

11


---Page Break---
Nguyen, L., Nguyen, P. H., Dijk, M., Richtárik, P., Scheinberg, K., and Takác, M. (2018). SGD and
hogwild! convergence without the bounded gradients assumption. In International Conference on
Machine Learning, pages 3750–3758. PMLR.

Recht, B., Re, C., Wright, S., and Niu, F. (2011). Hogwild!: A lock-free approach to parallelizing
stochastic gradient descent. Advances in Neural Information Processing Systems, 24.

Scaman, K., Bach, F., Bubeck, S., Lee, Y. T., and Massoulié, L. (2017). Optimal algorithms for
smooth and strongly convex distributed optimization in networks. In International Conference on
Machine Learning, pages 3027–3036. PMLR.

Shi, W., Ling, Q., Wu, G., and Yin, W. (2015). A proximal gradient algorithm for decentralized
composite optimization. IEEE Transactions on Signal Processing, 63(22):6013–6023.

Tyurin, A., Pozzi, M., Ilin, I., and Richtárik, P. (2024). Shadowheart SGD: Distributed asynchronous
SGD with optimal time complexity under arbitrary computation and communication heterogeneity.
arXiv preprint arXiv:2402.04785.

Tyurin, A. and Richtárik, P. (2023). Optimal time complexities of parallel stochastic optimization
methods under a fixed computation model. Advances in Neural Information Processing Systems
(NeurIPS).

Van Handel, R. (2014). Probability in high dimension. Lecture Notes (Princeton University).

Vogels, T., He, L., Koloskova, A., Karimireddy, S. P., Lin, T., Stich, S. U., and Jaggi, M. (2021).
RelaySum for decentralized deep learning on heterogeneous data. Advances in Neural Information
Processing Systems, 34.

Yang, T., Yi, X., Wu, J., Yuan, Y., Wu, D., Meng, Z., Hong, Y., Wang, H., Lin, Z., and Johansson,
K. H. (2019). A survey of distributed optimization. Annual Reviews in Control, 47:278–305.

12


---Page Break---
Contents

1
Introduction
1

1.1
Decentralized setup with times . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1

2
Previous Results
2

2.1
Time complexity with one worker
. . . . . . . . . . . . . . . . . . . . . . . . . .
2

2.2
Parallel optimization without communication costs
. . . . . . . . . . . . . . . . .
2

2.3
Parallel optimization with communication costs τi→j . . . . . . . . . . . . . . . .
3

3
Contributions
4

4
Lower Bound
4

5
New Method: Fragile SGD
5

5.1
Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8

5.2
Interpretation of the upper and lower bounds (11) and (7) . . . . . . . . . . . . . .
8

5.3
Limitations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8

5.4
Comparison with previous methods
. . . . . . . . . . . . . . . . . . . . . . . . .
8

5.5
Time complexity with dynamic bounds . . . . . . . . . . . . . . . . . . . . . . . .
9

6
Example: Line or Circle
9

7
Heterogeneous Setup
10

8
Highlights of Experiments
10

A More Examples
15

A.1
ND-Mesh or ND-Torus . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15

A.2
Star graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

A.3
General case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

B
On the Connection to the Gossip Framework
16

C Heterogeneous Setup
17

C.1
Lower bound
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17

C.2
Amelie SGD: optimal method in the heterogeneous setting . . . . . . . . . . . . . .
17

C.3
Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

C.4
Comparison with previous methods
. . . . . . . . . . . . . . . . . . . . . . . . .
19

D Convex Functions in the Homogeneous and Heterogeneous Setups
19

D.1 Assumptions in convex world . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

D.2
Homogeneous setup and nonsmooth case
. . . . . . . . . . . . . . . . . . . . . .
20

D.3
Homogeneous setup and smooth case
. . . . . . . . . . . . . . . . . . . . . . . .
20

13


---Page Break---
D.4
Heterogeneous setup and nonsmooth case . . . . . . . . . . . . . . . . . . . . . .
21

D.5
Heterogeneous setup and smooth case . . . . . . . . . . . . . . . . . . . . . . . .
22

D.6
On lower bounds
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

D.7
Previous works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

E
Lower Bound: Diving Deeper into the Construction
24

E.1
Description of Protocols 8 and 1 . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

E.2
Lower bound
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

E.3
Proof sketch of Theorem 19
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

E.4
Full proof of Theorem 19 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

E.5
Proof of Lemma 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31

E.6
Proof of Lemma 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34

F
Lower Bound in the Heterogeneous Setup
36

G Proof of the Time Complexity for Homogeneous Case
38

H Proof of the Time Complexity for Heterogeneous Case
41

I
Classical SGD Theory
44

J
Experiments
46

J.1
Experiments with Logistic Regression: Fast vs Slow Communication . . . . . . . .
48

J.2
Experiments with ResNet-18 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
48

14


---Page Break---
21

16

11

6

1

22

17

12

7

2

23

18

13

8

3

24

19

14

9

4

25

20

15

10

5

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ
ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

ρ

(a) 2D-Mesh

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

(b) 3D-Mesh

Figure 4: Examples of ND-Mesh graphs. For all i ̸= j ∈[n], edges i →j and j →i are merged and
visualized with one undirected edge.

A
More Examples

A.1
ND-Mesh or ND-Torus

We now consider a generalization of Line graphs: ND-Mesh graphs. In Figures 4a and 4b, we present
examples of 2D-Mesh and 3D-Mesh. For simplicity, assume that n = (2k + 1)N for some k ∈N.
The computation speeds hi = h for all i ∈[n], and the communicate speeds of the edges ρi→j = ρ
if workers i and j are connected in a mesh and ρi→j = ∞for all other i, j ∈[n]. Using geometrical
reasoning, it is clear an index j∗, that minimizes (11), corresponds to the worker in the middle of a
graph (13 in Figures 4a and 14 in Figures 4b). Therefore,

T2D-Mesh = Θ
L∆

ε
min
k∈[n] max

τπj∗,k→j∗, h, σ2h

kε


.

For now, let us consider a 2D-Mesh graph. The number of workers, that have the length of the shortest
path to j∗equals to 0, is 1. The number of workers, that have the length of the shortest path to j∗less
or equal to ρ, is 5. The number of workers, that have the length of the shortest path to j∗less or equal
to 2ρ, is 13. In general, the number of workers, that have the length of the shortest path to j∗less or
equal to
√

kρ, is Θ(k) for all k ∈{0, . . . , Θ (n)}. It means

T2D-Mesh = Θ
L∆

ε
min
k∈{0,...,Θ(n)} max
√

kρ, h,
σ2h
(k + 1)ε



= Θ









L∆

ε




h +

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

σ2h

ε ,

σ2h

ρε
2/3
≤1,

ρ2/3 
σ2h

ε
1/3
,
n >

σ2h

ρε
2/3
> 1,

σ2h

nε ,

σ2h

ρε
2/3
≥n












.

Using the same reasoning, for the general case with a ND-Mesh graph, we get

TND-Mesh = Θ
L∆

ε
min
k∈{0,...,Θ(n)} max

k1/Nρ, h,
σ2h
(k + 1)ε



= Θ









L∆

ε




h +

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

σ2h

ε ,

σ2h

ρε
N/(N+1)
≤1,

ρN/(N+1) 
σ2h

ε
1/(N+1)
,
n >

σ2h

ρε
N/(N+1)
> 1,

σ2h

nε ,

σ2h

ρε
N/(N+1)
≥n












.

As in Line graphs, these complexities have three regimes depending on the problem’s parameters. Up
to a constant factor, the same conclusions apply to ND-Torus.

15


---Page Break---
A.2
Star graph

Let us consider Star graphs with different computation and communication speeds. Let us fix a graph
with n + 1 workers, where one worker with the index n + 1 is in the center, and all other n workers
are only directly connected to worker n + 1. Then,

τi→j = ρi→n+1 + ρn+1→j
∀i ̸= j ∈[n + 1] and τi→i = 0
∀i ∈[n].

Using these constraints, we can simplify (11). There are two possible best strategies: i) a pivot worker
j∗works locally without communications, ii) a pivot worker j∗works with communications but it
would necessary require to communicate through the central worker; in this case, the central worker
n + 1 is a pivot worker. Therefore, we get

Tstar = min

"
L∆

ε
min
j∈[n] max

hj, σ2hj

ε



L∆

ε
min
k∈[n] max




max{ρπn+1,k→n+1 + ρn+1→πn+1,k, hπn+1,k}, σ2

ε

 k
X

i=1

1
hπn+1,i

!−1




#

since τn+1→πn+1,k = ρn+1→πn+1,k for all k ∈[n + 1]. Let us slightly simplify the result and assume
that broadcasting from the central worker is fast, i.e., ρn+1→i ≤ρi→n+1 for all i ∈[n + 1], then

Tstar = min

"
L∆

ε
min
j∈[n] max

hj, σ2hj

ε



|
{z
}
Tslow comm.

,

L∆

ε
min
k∈[n] max




max{ρπn+1,k→n+1, hπn+1,k}, σ2

ε

 k
X

i=1

1
hπn+1,i

!−1



|
{z
}
Tfast comm.

#

.

Tyurin et al. (2024) also considered Star graphs and showed that Tfast comm. is the optimal time
complexity for methods that communicate through the central worker. Our result is more general
since we also consider decentralized methods and capture the term Tslow comm. that can be potentially
smaller if communication is slow. Tyurin et al. (2024) conjectured that Tstar is the optimal bound, and
we proved it (up to log factor) here.

A.3
General case

We show how one can use our generic result (11) to get an explicit formula in some cases. For the
general case with arbitrary communication and computation times, the minimizers j and k in (11)
can be found in the following way. First, we have to find τi→j using any algorithm that solves the
all-pairs shortest path problem (e.g., Floyd–Warshall algorithm (Floyd, 1962)). Once we know τi→j,
we should sort {max{τi→j + τj→i, hi}}n
k=1 for all j ∈[n] to find the permutations. Finally, we have
enough information to calculate t∗from Definition 2.

B
On the Connection to the Gossip Framework

Most of the previous methods were designed for a different setting, gossip-type communication. In
fact, our setting is more general than the gossip communication. Indeed, recall that in the gossip com-
munication, worker i is allowed to get vectors from other workers through the operation Pn
j=1 wijxj,
where wij ∈{0, 1}. This is equivalent to our setting for the case when the communication time
ρij = ∞when wij is zero, and ρij = 1 if wij is not zero, and worker i sums the received vectors.
But our setting is richer since we allow different communication and computation times and allow
workers to do whatever they like with vectors (not only to sum).

Moreover, the gossip framework codes the communication graph through the matrix {wij}. We
propose to code graphs through the times {ρij} ({τij}). Our approach is closer to real scenarios
because, as we explained previously, it includes the gossip framework.

16


---Page Break---
C
Heterogeneous Setup

We now consider the heterogeneous setting. The only difference is instead of (1), we consider the
following problem.

min
x∈Rd

n
f(x) := 1

n

n
X

i=1
Eξi∼Di [fi(x; ξi)]
|
{z
}
fi(x):=

o
,
(17)

where fi : Rd × Sξi →Rd and ξi are random variables with some distributions Di on Sξi. We now
present our upper and lower bounds and discuss them.

C.1
Lower bound

In Section F, we prove the following lower bound.

Theorem 7 (Lower Bound; Simplified Presentation of Theorem 20). Consider Protocol 1 with
∇fi(·; ·). We take any hi ≥0 and τi→j ≥0 for all i, j ∈[n] such that τi→j ≤τi→k + τk→j for all
i, k, j ∈[n]. We fix L, ∆, ε, σ2 > 0 that satisfy the inequality ε < cL∆for some universal constant c.

For any (zero-respecting) algorithm, there exists a function f = 1

n

nP

i=1
fi, which satisfy Assumptions 1,

2 and f(0) −f ∗≤∆, and stochastic gradient mappings ∇fi(·; ·), which satisfy Assumption 3
(Eξ[∇fi(x; ξ)] = ∇fi(x) and Eξ[∥∇fi(x; ξ) −∇fi(x)∥2] ≤σ2), such that the required time to find
ε–solution is

Ω

 
L∆

ε
max

(

max
i,j∈[n] τi→j, max
i∈[n] hi, σ2

nε

 
1
n

n
X

i=1
hi

!)!

seconds.

C.2
Amelie SGD: optimal method in the heterogeneous setting

We now present a new method based on Malenia SGD from (Tyurin and Richtárik, 2023) and our
Fragile SGD. As in Fragile SGD, we also have n + 1 processes running in all workers. The main idea
is that all workers calculate stochastic gradients in parallel and accumulate them locally. Process 0
zero waits for the moment when
n
bj∗≥S

n. Process i calculates bi in Line 20 of Algorithm 7, and

sends it to Process nextst,j∗(i). Thus, bj∗accumulates the sum Pn
i=1
1
sk
i , which decreases with time

since sk
i is the number of calculated stochastic gradients in worker i in different moments of time.
Therefore,
n
bj∗≥S

n will hold at some point of time if workers continue computing gradients. Finally,

Algorithm 6 runs all reduce and does the update of xk.

Theorem 8. Let Assumptions 1 and 2 hold for the function f and Assumption 3 holds for the functions
fi for all i ∈[n]. We take γ = 1/2L, the parameter S = max{

σ2/ε

, n}, any pivot worker j∗∈[n],
and any spanning trees st and stbc, in Algorithm 5. For all iterations number K ≥16L∆/ε, we get

1
K
PK−1
k=0 E
h∇f(xk)
2i
≤ε.

Note that Theorem 8 states the convergence of Amelie SGD even if the computation and communication
speeds are unbounded.

Theorem 9. Consider the assumptions and the parameters from Theorem 8. For any pivot worker
j∗∈[n] and any spanning trees st and stbc, Algorithm 5 converges after at most

Θ

 
L∆

ε
max

(

max
i,j∈[n] µi→j, max
i∈[n] hi, σ2

nε

 
1
n

n
X

i=1
hi

!)!

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc) for all i ∈[n].

17


---Page Break---
Algorithm 5 Amelie SGD

1: Input: starting point x0, stepsize γ, parameter S, pivot worker j∗, spanning trees st and stbc
2: Start Process 0 (Alg. 6) in worker j∗

3: Start Process i (Alg. 7) in all workers for all i ∈[n] (including worker j∗)

Algorithm 6 Process 0 (running in worker j∗)

1: for k = 0, 1, . . . , K −1 do
2:
Broadcast xk to all workers using the spanning tree stbc
3:
Init sk = 0
4:
while sk < S

n do
5:
Wait for a message bj∗from Process j∗

6:
Calculate sk =
n
bj∗
7:
end while

8:
Run all reduce with { 1

sk
i gk
i }n
i=1 to find gk = 1

n

nP

i=1

1
sk
i gk
i using the spanning tree st

9:
xk+1 = xk −γgk

10: end for

Algorithm 7 Process i (running in worker i)

1: while True do
2:
Get a new point xk broadcasted by Process 0
3:
Init (gk
i , sk
i ) = (0, 0)
4:
Init bi,p = ∞for all p ∈[n] s.t. nextst,j∗(p) = i
5:
Run the following three functions in parallel and go to Line 24
6:
function CalculateStochasticGradients
7:
while True
8:
Calculate ∇fi(xk; ξ),
ξ ∼Di
9:
Run atomic add gk
i = gk
i + ∇fi(xk; ξ), sk
i = sk
i + 1
10:
end while
11:
end function
12:
function ReceiveCountersFromPreviousWorkers
13:
while True
14:
Wait for a message bp from any Process p s.t. nextst,j∗(p) = i
15:
Run atomic update bi,p = bp
16:
end while
17:
end function
18:
function SendCounterToNextWorker
19:
while True
20:
Run atomic sum bi =
P

p∈[n]:nextst,j∗(p)=i
bi,p + 1

sk
i

21:
Send bi ∈R ∪{∞} (one float) to Process nextst,j∗(i) and wait while it sending
(Process j∗sends to Process 0 by the definition of nextst,j∗(·))
22:
end while
23:
end function
24:
Wait for new point. If receives new point, stop all computations in functions, and continue
25:
Ignore all non-received messages
26: end while

18


---Page Break---
Corollary 2. Consider the assumptions and the parameters from Theorem 9. Let us take any pivot
worker j∗∈[n] and a spanning tree st (spanning tree stbc) that connects every worker i to worker
j∗(worker j∗to worker i) with the shortest distance τi→j∗(τj∗→i), then Algorithm 5 converges after
at most

Θ

 
L∆

ε
max

(

max
i,j∈[n] τi→j, max
i∈[n] hi, σ2

nε

 
1
n

n
X

i=1
hi

!)!

(18)

seconds.

C.3
Discussion

Corollary 2 together with Theorem 7 states that the time complexity (18) is optimal. The result is
pessimistic since the time complexity depends on the “diameter” maxi,j∈[n] τi→j and the slowest
performance maxi∈[n] hi. A similar dependence was observed in (Lu and De Sa, 2021; Tyurin and
Richtárik, 2023). As in (Tyurin and Richtárik, 2023), the stochastic term σ2/nε (1/n Pn
i=1 hi) depends
on the average of {hi} if σ2/ε is large.

C.4
Comparison with previous methods

Let us consider Minibatch SGD described in Section 2.3. This method converges after

Θ
L∆

ε
max

max
i,j∈[n] τi→j, max
i∈[n] hi, σ2

nε max{ max
i,j∈[n] τi→j, max
i∈[n] hi}


seconds in the heterogeneous setting. If σ2/ε is large, then Amelie SGD can be at least

max{ max
i,j∈[n] τi→j, max
i∈[n] hi}/

 
1
n

n
X

i=1
hi

!

times faster than Minibatch SGD. There are many more advanced methods (Lu and De Sa, 2021;
Vogels et al., 2021; Even et al., 2024) that work in decentralized stochastic heterogeneous setting.
Lu and De Sa (2021) developed similar lower and upper bounds, but there are at least three main
differences: i) they derived an iteration complexity instead of a time complexity and assumed the
performances of all workers are the same ii) the obtained lower bound holds only for one particular
multigraph while our complexity holds for any multigraph iii) they assumed the smoothness of fi,
while we consider the smoothness of f. The RelaySGD and Gradient Tracking methods by Vogels et al.
(2021); Liu et al. (2024) wait for the slowest worker; thus, they depend on maxi∈[n] hi in all regimes
of σ2/ε, unlike our method. Even et al. (2024) consider the heterogeneous asynchronous setting, but
their method assumes the similarity of the functions fi, which is not required in our method, and
Amelie SGD converges even if there is no similarity of functions.

D
Convex Functions in the Homogeneous and Heterogeneous Setups

We will be slightly more brief in the convex setting since the idea, the structure of time complexities,
and the general approach do not change significantly. For instance, instead of the time complexity
(11) that we get for the nonconvex case, in the nonsmooth convex case, we get (20). Thus, we will
get Θ

M 2R2

ε2
minj∈[n] t∗(σ2/M 2, . . . )

instead of Θ
  L∆

ε minj∈[n] t∗(σ2/ε, . . . )

. The same idea
applies to the smooth convex case. In the convex setting, we need the following assumptions.

D.1
Assumptions in convex world

Assumption 4. The function f is convex and attains a minimum at some point x∗∈Rd.
Assumption 5. The function f is M–Lipschitz, i.e.,

|f(x) −f(y)| ≤M ∥x −y∥,
∀x, y ∈Rd

for some M ∈(0, ∞].

19


---Page Break---
Assumption 6. For all x ∈Rd, stochastic (sub)gradients ∇f(x; ξ) are unbiased and are σ2-variance-

bounded, i.e., Eξ∼D [∇f(x; ξ)] ∈∂f(x) and Eξ∼D
h
∥∇f(x; ξ) −Eξ∼D [∇f(x; ξ)]∥2i
≤σ2, where

σ2 ≥0.

D.2
Homogeneous setup and nonsmooth case

Theorem 10. Let Assumptions 4, 5 and 6 hold. Choose any ε > 0. Let us take the batch size
S = max

σ2/M 2
, 1
	
, stepsize γ =
ε
M 2+σ2/S ∈

ε
2M 2 ,
ε
M 2

, any pivot worker j∗∈[n], and
any spanning trees st and stbc in Algorithm 2. Then after K ≥2M 2R2/ε2 iterations the method
guarantees E

f(bxK)

−f(x∗) ≤ε, where bxK = 1

K
PK−1
k=0 xk and R =
x∗−x0 .

Proof. The proof of Theorem 10 is almost the same as in Theorem 4 (see Section G). The proof of
Theorem 4 states that the steps of Fragile SGD are equivalent to the classical SGD method. Thus, we
can use the classical result from the literature (Lan, 2020). Using Theorem 22, we get

E

f(bxK)

−f(x∗) ≤ε

if

K ≥2M 2 x∗−x02

ε2
≥(M 2 + σ2

S )
x∗−x02

ε2
for the stepsize
γ =
ε
M 2 + σ2

S
∈
h
ε
2M 2 ,
ε
M 2

i
,

where we use the fact that S ≥σ2/M 2.

Theorem 11. Consider the assumptions and the parameters from Theorem 10. For any pivot worker
j∗∈[n] and spanning trees st and stbc, Algorithm 2 converges after at most

Θ
M 2R2

ε2
t∗(σ2/M 2, [hi]n
i=1, [µi→j∗+ µj∗→i]n
i=1)

(19)

seconds, where µi→j∗(µj∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc).

Proof. The proof is identical to the proof of Theorem 5.

Corollary 3. Consider the assumptions and the parameters from Theorem 11. Let us take a pivot
worker j∗= arg minj∈[n] t∗(σ2/M 2, [hi]n
i=1, [τi→j + τj→i]n
i=1), and a spanning tree st (spanning
tree stbc) that connects every worker i to worker j∗(worker j∗to every worker i) with the shortest
distance τi→j∗(τj∗→i). Then Algorithm 2 converges after at most

Θ
M 2R2

ε2
min
j∈[n] t∗(σ2/M 2, [hi]n
i=1, [τi→j + τj→i]n
i=1)

(20)

seconds.

D.3
Homogeneous setup and smooth case

In the homogeneous and smooth case, we will slightly modify Fragile SGD. Instead of Line 8 of
Algorithm 3, we use the following steps:

γk+1 = γ · (k + 1),
αk+1 = 2/(k + 2)

yk+1 = (1 −αk+1)xk + αk+1uk,
(u0 = x0)

uk+1 = uk −γk+1

sk gk,

xk+1 = (1 −αk+1)xk + αk+1uk+1.

(21)

We will call such a method the Accelerated Fragile SGD method. The idea is to use the acceleration
technique from (Lan, 2020) (which is based on (Nesterov, 1983)).

20


---Page Break---
Theorem 12. Let Assumptions 4, 1 and 3 hold. Choose any ε > 0. Let us take the batch size

S = max
nl
(σ2R)/(ε3/2√

L)
m
, 1
o
, γ = min

1
4L,
h
3R2S
4σ2(K+1)(K+2)2
i1/2
, any pivot worker j∗,

and any spanning trees st and stbc in Accelerated Method 2 (Accelerated Fragile SGD), then after
K ≥8
√

LR
√ε
iterations the method guarantees that E

f(xK)

−f(x∗) ≤ε, where R =
x∗−x0 .

Proof. Using the same reasoning as in Theorem 4, Accelerated Fragile SGD is just the classical
accelerated stochastic gradient method with a batch size greater or equal to S. We can use Proposition
4.4 from Lan (2020). For the stepsize

γ = min

(
1
4L,

3R2S
4σ2(K + 1)(K + 2)2

1/2)

,

we have

E

f(xK)

−f(x∗) ≤4LR2

K2
+ 4
√

σ2R2
√

SK
.

Therefore,

E

f(xK)

−f(x∗) ≤ε

if

K ≥8
√

LR
√ε
≥8 max

(√

LR
√ε , σ2R2

ε2S

)

,

where we use the choice of S.

Theorem 13. Consider the assumptions and the parameters from Theorem 12. For any pivot worker
j∗∈[n] and spanning trees st and stbc, Accelerated Algorithm 2 (Accelerated Fragile SGD) converges
after at most

Θ

 √

LR
√ε t∗

σ2R
ε3/2√

L
, [hi]n
i=1, [µi→j∗+ µj∗→i]n
i=1

!

(22)

seconds, where µi→j∗(µj∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc).

Proof. The proof is identical to the proof of Theorem 5.

Corollary 4. Consider the assumptions and the parameters from Theorem 13. Let us take a pivot
worker j∗= arg minj∈[n] t∗
σ2R
ε3/2√

L, [hi]n
i=1, [τi→j + τj→i]n
i=1

, and a spanning tree st (spanning

tree stbc) that connects every worker i to worker j∗(worker j∗to every worker i) with the shortest
distance τi→j∗(τj∗→i). Then Accelerated Algorithm 2 (Accelerated Fragile SGD) converges after at
most

Θ

 √

LR
√ε
min
j∈[n] t∗

σ2R
ε3/2√

L
, [hi]n
i=1, [τi→j + τj→i]n
i=1

!

(23)

seconds.

D.4
Heterogeneous setup and nonsmooth case

Consider the optimization problem (17).
Theorem 14. Let Assumptions 4, 5 hold for the function f and Assumption 6 holds for the func-
tions fi for all i ∈[n]. Choose any ε > 0. Let us take the batch size S = max

σ2/M 2
, n
	
,
γ =
ε
M 2+σ2/S ∈

ε
2M 2 ,
ε
M 2

, any pivot worker j∗∈[n], and any spanning trees st and stbc, in
Algorithm 5, then after K ≥2M 2R2/ε2 iterations the method guarantees that E

f(bxK)

−f(x∗) ≤ε,
where bxK = 1

K
PK−1
k=0 xk and R =
x∗−x0 .

21


---Page Break---
Proof. The proof of Theorem 14 is almost the same as in Theorems 8 (see Section H). The proof of
Theorem 8 states that the steps of Amelie SGD are equivalent to the classical SGD method. Thus, we
can use the classical result from the literature (Lan, 2020). Using Theorem 22, we get

E

f(bxK)

−f(x∗) ≤ε

if

K ≥2M 2 x∗−x02

ε2
≥(M 2 + σ2

S )
x∗−x02

ε2
for the stepsize
γ =
ε
M 2 + σ2

S
∈
h
ε
2M 2 ,
ε
M 2

i
,

where we use the fact that S ≥σ2/M 2.

Theorem 15. Consider the assumptions and the parameters from Theorem 14. For any pivot worker
j∗∈[n] and any spanning trees st and stbc, Algorithm 5 converges after at most

Θ

 
M 2R2

ε2
max

(

max
i,j∈[n] µi→j, max
i∈[n] hi,
σ2

nM 2

 
1
n

n
X

i=1
hi

!)!

(24)

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(worker j∗to worker i) along the spanning tree st (spanning tree stbc) for all i ∈[n].

Proof. The proof is identical to the proof of Theorem 9.

Corollary 5. Consider the assumptions and the parameters from Theorem 15. Let us take any pivot
worker j∗∈[n] and a spanning tree st (spanning tree stbc) that connects every worker i to worker
j∗(worker j∗to worker i) with the shortest distance τi→j∗(τj∗→i), then Algorithm 5 converges after
at most

Θ

 
M 2R2

ε2
max

(

max
i,j∈[n] τi→j, max
i∈[n] hi,
σ2

nM 2

 
1
n

n
X

i=1
hi

!)!

seconds.

D.5
Heterogeneous setup and smooth case

Consider the optimization problem (17). In this section, we use the same idea as in Section D.3. We
will modify Amelie SGD and, instead of Line 9 from Algorithm 6, we use the lines (21). We call such
a method the Accelerated Amelie SGD method.
Theorem 16. Let Assumptions 4 and 1 hold for the function f and Assumption 3 holds for the
functions fi. Choose any ε > 0. Let us take the batch size S = max
nl
(σ2R)/(ε3/2√

L)
m
, n
o
,

γ = min

1
4L,
h
3R2S
4σ2(K+1)(K+2)2
i1/2
, any pivot worker j∗, and any spanning trees st and stbc

in Accelerated Method 5 (Accelerated Amelie SGD), then after K ≥8
√

LR
√ε
iterations the method
guarantees that E

f(xK)

−f(x∗) ≤ε, where R =
x∗−x0 .

Proof. Accelerated Amelie SGD is equivalent to the accelerated stochastic gradient method with a
mini-batch from Lan (2020). The proof repeats the proofs of Theorem 12 and Theorem 8.

Theorem 17. Consider the assumptions and the parameters from Theorem 16. For any pivot worker
j∗∈[n] and any spanning trees st and stbc, Accelerated Algorithm 5 (Accelerated Amelie SGD)
converges after at most

Θ

 √

LR
√ε max

(

max
i,j∈[n] µi→j, max
i∈[n] hi,
σ2R
nε3/2√

L

 
1
n

n
X

i=1
hi

!)!

(25)

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(worker j∗to worker i) along the spanning tree st (spanning tree stbc) for all i ∈[n].

22


---Page Break---
Proof. The proof is identical to the proof of Theorem 9.

Corollary 6. Consider the assumptions and the parameters from Theorem 17. Let us take any pivot
worker j∗∈[n] and a spanning tree st (spanning tree stbc) that connects every worker i to worker
j∗(worker j∗to worker i) with the shortest distance τi→j∗(τj∗→i), then Accelerated Algorithm 5
(Accelerated Amelie SGD) converges after at most

Θ

 √

LR
√ε max

(

max
i,j∈[n] τi→j, max
i∈[n] hi,
σ2R
nε3/2√

L

 
1
n

n
X

i=1
hi

!)!

(26)

seconds.

D.6
On lower bounds

In previous subsections, we provide upper bounds on the time complexities for different classes of
convex functions and optimization problems. Using the same reasoning as in Section 4 and (Tyurin
and Richtárik, 2023)[Section B], we conjecture that the obtained upper bounds are tight and optimal
(up to log factors in the homogeneous case).

D.7
Previous works

Let us consider the time complexities (20) and (23) (accelerated rate) in the homogeneous and
convex cases. When τi→j = 0 for all i, j ∈[n], Even et al. (2024) recovers the time complexity
(nonaccelerated in the smooth case) of Asynchronous SGD (Mishchenko et al., 2022; Koloskova et al.,
2022) which is suboptimal (Tyurin and Richtárik, 2023). When the communication is free, we recover
the time complexity (accelerated in the smooth case) of Accelerated Rennala SGD from (Tyurin and
Richtárik, 2023), which is optimal if τi→j = 0 for all i, j ∈[n].

In the heterogeneous and convex setting, (26) improves the result from (Even et al., 2024) since
(26) is an accelerated rate and does not depend on a quantity that measures the similarity of the
functions fi. When σ = 0, the result (26) is consistent with the lower bound from (Scaman et al.,
2017). However, when σ > 0, as far as we know, the time complexity (26) is new in the convex
smooth setting.

23


---Page Break---
E
Lower Bound: Diving Deeper into the Construction

Protocol 8 Time Multiple Oracles Protocol with Communication

1: Input: function f (or functions fi) computation oracles {Oi}n
i=1 ∈O(f) communication oracles
{Cp
i→j}i∈[n],j∈[n],p≥1, algorithm {(M k, Lk, Dk, P k
1 , . . . , P k
n, V k
1 , . . . , V k
n )}∞
k=0 ∈A

2: sh,0
i
= sτ,0
i,j,p = 0 for all i, j ∈[n], p ∈N
3: for k = 0, . . . , ∞do
4:
(tk+1, ck+1) = M k(g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk
1, . . . , gk
n)
▷tk+1 ≥tk

5:
if ck+1 = 0 then
6:
ik+1 = Lk(g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk
1, . . . , gk
n)
7:
xk
ik+1 = P k
ik+1(g1
ik+1, . . . , gk
ik+1)

8:
(sh,k+1
ik+1 , gk+1
ik+1) = Oik+1(tk+1, xk
ik+1, sh,k
ik+1)
▷∀j ̸= ik+1 : sh,k+1
j
= sh,k
j
,
gk+1
j
= 0
9:
else
10:
ik+1, jk+1, pk+1 = Dk(g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk
1, . . . , gk
n)
11:
vk
ik+1 = V k
ik+1(g1
ik+1, . . . , gk
ik+1)

12:
(sτ,k+1
ik+1,jk+1,pk+1, gk+1
jk+1) = Cpk+1

ik+1→jk+1(tk+1, vk
ik+1, sτ,k
ik+1,jk+1,pk+1)

▷∀j ̸= jk+1 : gk+1
j
= 0, ∀i ̸= ik+1, j ̸= jk+1, p ̸= pk+1 : sτ,k+1
i,j,p
= sτ,k
i,j,p,
13:
end if
14: end for

In Section 4, we present a brief overview and simplified theorem for the lower bound. We now
provide a formal and strict mathematical construction.

E.1
Description of Protocols 8 and 1

One way how we can formalize Protocol 1 is to use Protocol 8. Let us explain it. Using Protocol 8,
we consider any possible method that works in our distributed asynchronous setting. The mapping
M k of the algorithm returns the time tk+1 (ignore for now) and ck+1. If ck+1 = 0, then the algorithm
decides to start the calculation of a stochastic gradient: it determines the index ik+1 of a worker that
will start the calculation, then, using all locally available information g1
ik+1, . . . , gk
ik+1, it calculates
a new point xk
ik+1, and passes this point to the computation oracle Oik+1 that will return a new
stochastic gradient after hik+1 seconds. If ck+1 = 1, then the algorithm decides to communicate a
vector: it returns the indices ik+1 and jk+1 of two workers that will communicate, and the index
pk+1 ∈N of a communication oracle, calculates vk
ik+1 in worker ik+1 using only all locally available

information g1
ik+1, . . . , gk
ik+1, and passes it to the communication oracle Cpk+1

ik+1→jk+1 that will send
the vector after τik+1→jk+1 seconds.

The computation oracles we define as

Oi : R≥0
|{z}
time

× Rd
|{z}
point
× (R≥0 × Rd × {0, 1})
|
{z
}
input state

→(R≥0 × Rd × {0, 1})
|
{z
}
output state

×Rd

such that
Oi(t, x, (st, sx, sq)) =






((t, x, 1),
0),
sq = 0,
((st, sx, 1),
0),
sq = 1, t < st + hi,
((0, 0, 0),
∇f(sx; ξ)),
sq = 1, t ≥st + hi,
(27)

where ξ ∼D.

The communication oracles we define as

Cp
i→j : R≥0
|{z}
time

× Rd
|{z}
point
× (R≥0 × Rd × {0, 1})
|
{z
}
input state

→(R≥0 × Rd × {0, 1})
|
{z
}
output state

×Rd

24


---Page Break---
such that
Cp
i→j(t, x, (st, sx, sq)) =






((t, x, 1),
0),
sq = 0,
((st, sx, 1),
0),
sq = 1, t < st + τi→j,
((0, 0, 0),
sx),
sq = 1, t ≥st + τi→j.
(28)

The idea is that the computation oracle (27) emulates the behavior of a real worker i that requires hi
seconds to calculate a stochastic gradient. The communication oracle emulates the behavior of a real
communication channel that requires τi→j seconds to send a vector from worker i to worker j.

One can see that, for all i ̸= j ∈[n], an algorithm can access an infinite number of communication
oracles C1
i→j, C2
i→j, . . . . We allow an algorithm to send as many vectors from worker i to worker j
in parallel as it wants.

We now discuss the role of tk+1. One can see that the time tk+1 is returned by the algorithm, and tk+1

is passed to the oracles Oik+1 and Cpk+1

ik+1→jk+1. Consider that Oik+1 was called with tk+1 for the first

time, then it will return zero vector because (sh,k
ik+1)q = 0 (see (27)) at the beginning. Then, by the
construction of (27), the oracle will return a non-zero vector in the second output if only the algorithm
returns a time that is greater or equal to tk+1 + hik+1. The same idea applies to Cpk+1

ik+1→jk+1 : if it
was called tk+1 for the first time, then worker jk+1 will get a non-zero vector only if the algorithm
passes a time that is greater or equal to tk+1 + τik+1→jk+1.

The oracles force the algorithm to increase the time tk+1; otherwise, it will not get new information
(∇f(sx; ξ) in (27)) about the function. One crucial constraint is tk+1 ≥tk, meaning that the
algorithm can not “cheat” and travel into the past. The idea of such a protocol was proposed in Tyurin
and Richtárik (2023), and we refer to Sections 3-5 for a detailed description. A more readable and
less formal version of Protocol 8 is presented in the Protocol 1.

E.2
Lower bound

Before we state the main theorem, let us define the class of zero-respecting algorithms.

Definition 18 (Algorithm Class Azr). Let us consider Protocol 8. We say that the sequence of tuples
of mappings {(M k, Lk, Dk, P k
1 , . . . , P k
n, V k
1 , . . . , V k
n )}∞
k=0 is a zero-respecting algorithm, if

1. M k : Rd × · · · × Rd
|
{z
}
n×k times
→R≥0 × {0, 1} for all k ≥1, and M 0 ∈R≥0 × {0, 1}.

2. Lk : Rd × · · · × Rd
|
{z
}
n×k times
→[n] for all k ≥1, and L0 ∈[n].

3. Dk : Rd × · · · × Rd
|
{z
}
n×k times
→[n] × [n] × N for all k ≥1, and D0 ∈[n] × [n] × N.

4. For all i ∈[n], P k
i : Rd × · · · × Rd
|
{z
}
k times
→Rd for all k ≥1, and P 0
i ∈Rd.

5. For all i ∈[n], V k
i
: Rd × · · · × Rd
|
{z
}
k times
→Rd for all k ≥1, and V 0
i ∈Rd.

6. For all k ≥1 and g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk
1, . . . , gk
n ∈Rd,

tk+1 ≥tk,

where tk+1 and tk are defined as (tk+1, . . . ) = M k(g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk
1, . . . , gk
n)
and (tk, . . . ) = M k−1(g1
1, . . . g1
n, g2
1, . . . , g2
n, . . . , gk−1
1
, . . . , gk−1
n
).

7. For all i ∈[n], supp
 
xk
i

⊆Sk
j=1 supp

gj
i

, and supp
 
vk
i

⊆Sk
j=1 supp

gj
i

, for all

k ∈N0, where supp(x) := {i ∈[d] | xi ̸= 0}.

The set of all algorithms with this properties we define as Azr.

25


---Page Break---
Constraints 1–5 define domains of the mappings. Constraint 6 is required to ensure that the time
sequence tk does not decrease. Constraint 7 is a standard assumption that an algorithm is zero-
respecting (Arjevani et al., 2022).
Theorem 19. Consider Protocol 8. We take any hi ≥0 and τi→j ≥0 for all i, j ∈[n] such
that τi→j ≤τi→k + τk→j for all i, k, j ∈[n]. We fix L, ∆, ε, σ2 > 0 that satisfy the inequality
ε < c′L∆. For any algorithm A ∈Azr, there exists a function f, which satisfy Assumptions 1, 2 and
f(0) −f ∗≤∆, and a stochastic gradient mapping ∇f(·; ·), which satisfy Assumption 3, such that

E

inf
k∈St,i∈[n]

∇f(xk
i )
2

> ε, where St :=

k ∈N0 | tk ≤t
	
,

t = c ×
1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


,

where πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n]. The quantities c′ and c are universal constants. The sequences xk and tk are defined
in Protocol 8.

E.3
Proof sketch of Theorem 19

Let us provide a proof sketch that will give intuition behind the theorem. The full proof starts in
Section E.4.

Proof Sketch.

Part 1: Construct a function and stochastic gradient

The first part of the proof is standard (Carmon et al., 2020; Arjevani et al., 2022; Tyurin and Richtárik,
2023; Huang et al., 2022; Lu and De Sa, 2021), and we delegate it to Section E.4. For any algorithm,
we construct oracles and a “worst-case” function such that

inf
k∈St,i∈[n]

∇f(xk
i )
2 > 2ε
inf
k∈St,i∈[n] 1

prog(xk
i ) < T

,
(29)

where prog(x) := max{i ≥0 | xi ̸= 0}
(x0 ≡1) and T ≈L∆/ε. This inequality says that while
all points in Protocol 8 have the last coordinate equals 0 by the time t, an algorithm can not find an
ε–stationary point.

Since we assume that A ∈Azr is zero-respecting, the only way to discover the next non-zero
coordinate is through stochastic gradients. They are constructed in the following way (Arjevani et al.,
2022):

[∇f(x; ξ)]j := ∇jf(x)

1 + 1 [j > prog(x)]
ξ

p −1

∀x ∈RT ,

and ξ ∼Bernoulli(p) for all i ∈[n], where p ≈ε/σ2. We denote [x]j as the jth index of a vector
x ∈RT . The stochastic gradient equals to the exact gradient except for the last non-zero coordinate:
it zeros out it with the high probability 1 −p.

Part 2: The Level Game

In essence, the protocol is equivalent to the following collaborative game. Each worker i starts with
level ℓi = 0. The goal is to reach level T with at least one worker as fast as possible. There are two
ways how a worker can increase its level: i) worker i flips one coin per time from ξ ∼Bernoulli(p),
it takes hi seconds to flip one coin, and if the worker is lucky, i.e., ξ = 1, then it moves to the next
level ℓi = ℓi + 1; ii) worker i can share its level with another worker j, and it takes τi→j seconds
(we are allowed to run this operation again even if the previous is not finished). Both options can be
executed in parallel. What is the minimum possible time to reach the game’s goal?

26


---Page Break---
Since all workers have the levels equal to 0 at the beginning, they should flip coins in parallel and
wait for the moment when at least one worker moves to level 1. We define η1
i as the number of flips
in worker i to get ξ = 1. Clearly, η1
i ∼Geometric(p) are i.i.d. geometric random variables with the
probability p. With any strategy, it is necessary to wait at least

y1
j := min
i∈[n]


hiη1
i + τi→j
	

seconds to reach level 1 in worker j because once worker i flips ξ = 1, it will take at least τi→j
seconds to share level 1 to worker j due to the triangle inequality τi→j ≤τi→k + τk→j for all
i, j, k ∈[n], and we should minimize hiη1
i + τi→j over all workers. We now use mathematical
induction to prove that it is necessary to wait at least

yT
j := min
i∈[n]


yT −1
i
+ hiηT
i + τi→j
	

seconds to reach level T, where we define y0
i := 0 for all i ∈[n] and {ηT
i } are i.i.d. random variables
from Geometric(p). The base case is proven above. Assume that it is true for 1, . . . , T −1, then
worker i requires at least yT −1
i
+ hiηT
i seconds to flip a coin that moves to level T, it takes at least
τi→j seconds to share the level T, so it is necessary to wait at least mini∈[n]

yT −1
i
+ hiηT
i + τi→j
	

seconds in worker i to get level T. Ultimately, the minimum possible time to reach the game’s goal is

yT := min
j∈[n] yT
j .

From the previous result, we can conclude that if t ≤1

2yT , then

inf
k∈St,i∈[n]

∇f(xk
i )
2 > 2ε
(30)

for any algorithm A ∈Azr.

Part 3: The high probability bound of yT

Let us fix any determenistic value ¯y ∈R and take

t = 1

2 ¯y.
(31)

Using (30), we have

E

inf
k∈St,i∈[n]

∇f(xk
i )
2
≥E

inf
k∈St,i∈[n]

∇f(xk
i )
2 yT > ¯y

P
 
yT > ¯y

> P
 
yT > ¯y

2ε.

Thus, it is sufficient to find any ¯y ∈R such that

P
 
yT ≤¯y

≤1

2.
(32)

The sequence yT is a well-define time series. In Lemma 2, we show that (32) holds with

¯y = Θ




1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1






,

πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n]. We substitute this ¯y to (31) and get the result of theorem.

27


---Page Break---
E.4
Full proof of Theorem 19

This section considers the standard “worst case” function that helps to provide lower bounds in the
nonconvex world. Let us define

prog(x) := max{i ≥0 | xi ̸= 0}
(x0 ≡1).

For any T ∈N, Carmon et al. (2020); Arjevani et al. (2022) define

FT (x) := −Ψ(1)Φ(x1) +

T
X

i=2
[Ψ(−xi−1)Φ(−xi) −Ψ(xi−1)Φ(xi)] ,
(33)

where

Ψ(x) =

(
0,
x ≤1/2,

exp

1 −
1
(2x−1)2

,
x ≥1/2,
and
Φ(x) = √e
Z x

−∞
e−1

2 t2dt.

We will only rely on the following facts.
Lemma 1 (Carmon et al. (2020); Arjevani et al. (2022)). The function FT satisfies:

1. FT (0) −infx∈RT FT (x) ≤∆0T, where ∆0 = 12.

2. The function FT is l1–smooth, where l1 = 152.

3. For all x ∈RT , ∥∇FT (x)∥∞≤γ∞, where γ∞= 23.

4. For all x ∈RT , prog(∇FT (x)) ≤prog(x) + 1.

5. For all x ∈RT , if prog(x) < T, then ∥∇FT (x)∥> 1.
Theorem 19. Consider Protocol 8. We take any hi ≥0 and τi→j ≥0 for all i, j ∈[n] such
that τi→j ≤τi→k + τk→j for all i, k, j ∈[n]. We fix L, ∆, ε, σ2 > 0 that satisfy the inequality
ε < c′L∆. For any algorithm A ∈Azr, there exists a function f, which satisfy Assumptions 1, 2 and
f(0) −f ∗≤∆, and a stochastic gradient mapping ∇f(·; ·), which satisfy Assumption 3, such that

E

inf
k∈St,i∈[n]

∇f(xk
i )
2

> ε, where St :=

k ∈N0 | tk ≤t
	
,

t = c ×
1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


,

where πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n]. The quantities c′ and c are universal constants. The sequences xk and tk are defined
in Protocol 8.

Proof.
Part 1: The Worst Case Function
This part of the proof mirrors the proofs from Carmon et al. (2020); Arjevani et al. (2022); Tyurin
and Richtárik (2023); Huang et al. (2022); Lu and De Sa (2021). We provide it for completeness.
The goal of this part to construct a hard instance.

We fix λ > 0, T ∈N and take the function f(x) := Lλ2

l1 FT
  x

λ

, where the function FT is defined
in Section E.4. Note that the function f is L-smooth:

∥∇f(x) −∇f(y)∥= Lλ

l1

∇FT
x

λ


−∇FT
y

λ

 ≤Lλ
x

λ −y

λ

 = L ∥x −y∥
∀x, y ∈Rd,

where l1–smoothness of FT (Lemma 1). Let us take

T =

∆l1
Lλ2∆0


,

28


---Page Break---
then

f(0) −inf
x∈RT f(x) = Lλ2

l1
(FT (0) −inf
x∈RT FT (x)) ≤Lλ2∆0T

l1
≤∆.

We showed that the function f satisfy Assumptions 1, 2 and f(0) −f ∗≤∆.

For each worker i we take an oracle Oi, from (27) with the mapping g such that

[∇f(x; ξ)]j := ∇jf(x)

1 + 1 [j > prog(x)]
ξ

p −1

∀x ∈RT ,

and Di = Bernouilli(p) for all i ∈[n], where p ∈(0, 1]. We denote [x]j as the jth index of a vector
x ∈RT . This stochastic gradient is unbiased and σ2-variance-bounded. We have

E [[∇f(x, ξ)]i] = ∇if(x)

1 + 1 [i > prog(x)]
E [ξ]

p
−1

= ∇if(x)

for all i ∈[T], and

E
h
∥∇f(x; ξ) −∇f(x)∥2i
≤max
j∈[T ] |∇jf(x)|2 E

"ξ

p −1
2#

because the difference is non-zero only in one coordinate. Thus

E
h
∥∇f(x, ξ) −∇f(x)∥2i
≤∥∇f(x)∥2
∞(1 −p)
p
= L2λ2 ∇FT
  x

λ
2
∞(1 −p)
l2
1p

≤L2λ2γ2
∞(1 −p)
l2
1p
≤σ2,

where we use Lemma 1 and take

p = min
L2λ2γ2
∞
σ2l2
1
, 1

.

Let us take

λ =

√

2εl1

L
to ensure that

∥∇f(x)∥2 = L2λ2

l2
1

∇FT
x

λ


2
= 2ε
∇FT
x

λ


2

for all x ∈RT . From Lemma 1, we know that if prog(x) < T, then ∥∇FT (x)∥> 1. Thus, we get

∥∇f(x)∥2 > 2ε1 [prog(x) < T]
(34)

Therefore,

T =

∆L
2εl1∆0


(35)

and

p = min
2εγ2
∞
σ2
, 1

.
(36)

The inequality (34) implies

inf
k∈St,i∈[n]

∇f(xk
i )
2 > 2ε
inf
k∈St,i∈[n] 1

prog(xk
i ) < T

,
(37)

where {xk
i }∞
k=0 are sequences from Protocol 8.
Part 2: Reduction to Lower Bound Time Series
We now focus on (27). In (27), when the oracle in worker i calculates a new stochastic gradient, it

29


---Page Break---
samples a random variable ξ ∼D. This is equivalent to the procedure if we had T infinite sequences
of i.i.d. Bernoulli random variables

ξ1,1
i
, ξ1,2
i
, . . .
(prog(sx) = 0)

ξ2,1
i
, ξ2,2
i
, . . .
(prog(sx) = 1)
. . .

ξT,1
i
, ξT,2
i
, . . .
(prog(sx) = T −1)

for all i ∈[n] and the oracle would look at the progress of sx and take the next non-taken Bernoulli
random variable in the sequence that corresponds to that progress. For instance, if prog(sx) = j
for the first time in worker i, then the oracle will apply ξj,1
i
in (27). The next time when it gets
prog(sx) = j it will apply ξj,2
i
and so on. One by one, we get i.i.d. Bernoulli random variables.

Let us define

ηk
i = inf{j ∈N | ξk,j
i
= 1}

for all i ∈[n] and k ∈[T]. This is the first Bernoulli random variable from the sequence ξk,1
i
, ξk,2
i
, . . .
that equals 1. The random variables {ηk
i } are i.i.d. geometrically distributed random variables with
the probability p.

The next steps mirror Proof Sketch from Theorem 19. The oracles constructed in such a way that it
takes hi seconds to calculate a stochastic gradient, and at least τi→j seconds to send a vector from one
worker to another. Using the same reasoning as in the Level Game in Proof Sketch of Theorem 19, the
first time moment when worker j can get a vector with the first non-zero coordinate greater or equal

y1
j := min
i∈[n]


hiη1
i + τi→j
	

because, at the beginning, each worker i calculates stochastic gradients with prog(sx) = 0 and should
wait at least hiη1
i seconds to get a stochastic gradient with the progress equals 1. Then, it can share
this vector with any other worker j, but it takes at least τi→j seconds.

As in Proof Sketch of Theorem 19, we can use mathematical induction to prove that it is necessary to
wait at least

yT
j := min
i∈[n]


yT −1
i
+ hiηT
i + τi→j
	

seconds to get a point such that prog(sx) = T. The base case for y1
j has been proven. Worker i
requires at least yT −1
i
+ hiηT
i seconds to wait for the moment when the corresponding oracle will
return a stochastic gradient with progress T because yT −1
i
is the first time possible time to get a
vector with progress T −1 by the induction, and it will take at least additional hiηT
i seconds to
calculate ηT
i vectors with prog(sx) = T −1 in (27). Also, it takes at least τi→j seconds to share a
vector, so it is necessary to wait at least mini∈[n]

yT −1
i
+ hiηT
i + τi→j
	
seconds in worker i to get
the first vector with progress T.

In the end, the fastest possible time to get a vector with progress T is at least

yT := min
j∈[n] yT
j .

Part 3: Reduction to the concentration of yT
The last statement means that prog(xk
i ) < T for all i ∈[n] and k such that tk ≤1

2yT . Therefore, we
obtain

inf
k∈St,i∈[n]

∇f(xk
i )
2 > 2ε
(38)

for

t ≤1

2yT ,

where

T =

∆L
2εl1∆0


=

cT · ∆L

ε



30


---Page Break---
and ηj
i ∼Geometric(p) with

p = min
2εγ2
∞
σ2
, 1

= min
n
cp · ε

σ2 , 1
o
,

where cT = 3648−1 and cp = 1058 are universal constants. Let us fix any determenistic value ¯y ∈R
and take

t = 1

2 ¯y.
(39)

Using (38), we have

E

inf
k∈St,i∈[n]

∇f(xk
i )
2
≥E

inf
k∈St,i∈[n]

∇f(xk
i )
2 yT > ¯y

P
 
yT > ¯y

> P
 
yT > ¯y

2ε.

Thus, it is sufficient to find any ¯y ∈R such that

P
 
yT ≤¯y

≤1

2.
(40)

In the following lemma we use the notation of this theorem. We prove it in Section E.5.

Lemma 2. With

¯y = c ×
1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


,
(41)

we have P
 
yT ≤¯y

≤1

2, where πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n]. The quantity c is a universal constant.

Using Lemma 2, we can conclude that (40) holds and

E

inf
k∈St,i∈[n]

∇f(xk
i )
2
> ε

for

t = c

2 ×
1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


.

E.5
Proof of Lemma 2

Lemma 2. With

¯y = c ×
1
log n + 1
L∆

ε
min
j∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


,
(41)

we have P
 
yT ≤¯y

≤1

2, where πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n]. The quantity c is a universal constant.

Proof. Using the Chernoff method for any s > 0, we get

P
 
yk ≤t

= P
 
−syk ≥−st

= P

e−syk ≥e−st
≤estE
h
e−syki

= estE

exp

−s min
j∈[n] yk
j


= estE

max
j∈[n] exp
 
−syk
j

.

31


---Page Break---
We have a maximum operation that complicates the analysis. In response to this problem, we use a
well-known trick that bounds a maximum by a sum.

P
 
yk ≤t

≤est
n
X

j=1
E

exp
 
−syk
j

≤nest max
j∈[n] E

exp
 
−syk
j

.
(42)

We refer to (Van Handel, 2014)[Part II] for the explanation why it can be (almost) tight. This is the
main reason why we get an extra log n factor in (41). Let us consider the last exponent separately
and use the same trick again:

E

exp
 
−syk
j

= E

exp

−s min
i∈[n]


yk−1
i
+ hiηk
i + τi→j
	

= E

max
i∈[n] exp
 
−s

yk−1
i
+ hiηk
i + τi→j
	
.

Next, we get

E

exp
 
−syk
j

≤

n
X

i=1
E

exp
 
−s

yk−1
i
+ hiηk
i + τi→j
	

=

n
X

i=1
E
h
e−s(hiηk
i +τi→j)i
E

exp
 
−syk−1
i

.

In the last equality we use the independence.
We now bound E

exp
 
−syk−1
i

by
maxi∈[n] E

exp
 
−syk−1
i

and get

E

exp
 
−syk
j

≤

n
X

i=1
E
h
e−s(hiηk
i +τi→j)i
E

exp
 
−syk−1
i


≤

 n
X

i=1
E
h
e−s(hiηk
i +τi→j)i!

max
i∈[n] E

exp
 
−syk−1
i

.

Let us fix any sj > 0 for all j ∈[n] and take s = maxj∈[n] sj. Then

E

exp
 
−syk
j

≤

 n
X

i=1
E
h
e−sj(hiηk
i +τi→j)i!

max
i∈[n] E

exp
 
−syk−1
i

.
(43)

Let us fix ¯tj > 0 and consider

aj :=

n
X

i=1
E
h
e−sj(hiηk
i +τi→j)i
.
(44)

Then, we can use the following inequalities:

aj ≤

n
X

i=1
E
h
e−sj(hiηk
i +τi→j)1

hiηk
i + τi→j ≤¯tj

+ e−sj(hiηk
i +τi→j)  
1 −1

hiηk
i + τi→j ≤¯tj
i

≤

n
X

i=1
E
h
1

hiηk
i + τi→j ≤¯tj

+ e−sj¯tj  
1 −1

hiηk
i + τi→j ≤¯tj
i

= ne−sj¯tj + (1 −e−sj¯tj)

n
X

i=1
E

1

hiηk
i + τi→j ≤¯tj


≤ne−sj¯tj +

n
X

i=1
E

1

hiηk
i + τi→j ≤¯tj


≤ne−sj¯tj +

n
X

i=1
P
 
hiηk
i + τi→j ≤¯tj


≤ne−sj¯tj +

n
X

i=1
P
 
hiηk
i ≤¯tj

1 [τi→j ≤¯tj] .

32


---Page Break---
Since ηk
i ∼Geometric(p), we get

P
 
hiηk
i ≤¯tj

= 1 −(1 −p)

j ¯tj

hi

k
≤p
 ¯tj

hi


.

Then

aj ≤ne−sj¯tj + p

n
X

i=1

 ¯tj

hi


1 [τi→j ≤¯tj] .

For all j ∈[n], we take any permutation πj,· that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}.

We have

aj ≤ne−sj¯tj + p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

.
(45)

Recall that ¯tj > 0 is a parameter. In the following technical lemma, we choose ¯tj and show that the
second term in (45) is “small.” We prove it in Section E.6.

Lemma 3. For any n ≥1, hi ≥0,τi→j ≥0 for all i, j ∈[n], and p ∈(0, 1], we have

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

≤1

8

for all j ∈[n], where

¯tj := 1

8 min
k∈[n] max




max{τπj,k→j, hπj,k},

 k
X

i=1

p
hπj,i

!−1


.
(46)

and πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n].

Using Lemma 3 and (45), we get

aj ≤ne−sj¯tj + 1

8
for all j ∈[n]. Let us take

sj = log 8n

¯tj
(47)

to get

aj ≤1

8 + 1

8 ≤e−1.

for all j ∈[n]. Using (44) and (43), we obtain

E

exp
 
−syk
j

≤e−1 max
i∈[n] E

exp
 
−syk−1
i

,
(48)

for all j ∈[n]. We can conclude that

max
i∈[n] E

exp
 
−syk
i

≤e−1 max
i∈[n] E

exp
 
−syk−1
i

≤e−k,

where use the same reasoning in the recursion and y0
j = 0 for all j ∈[n]. We substitute the inequality
to (42):

P
 
yk ≤t

≤nest−k = est−k+log n

33


---Page Break---
It is sufficient to take k = T and any

t ≤1

s


T −log n + log 1

2



=
1
8 log 8n min
i∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k},

 k
X

i=1

p
hπj,i

!−1





T −log n + log 1

2



to get

P
 
yT ≤t

≤1

2,

where we use the definitions s = maxi∈[n] si, (46) and (47). Recall the choice of T and p in (35)
and (36): T =

cT · ∆L

ε

and p = min

cp ·
ε
σ2 , 1
	
for some universal constants cT and cp. Since
we have the condition ε < c′L∆for some universal constant c′ in the conditions of Theorem 19, we
can conclude that we can take

t = c ×
1
log n + 1
L∆

ε
min
i∈[n] min
k∈[n] max




max{τπj,k→j, hπj,k}, σ2

ε

 k
X

i=1

1
hπj,i

!−1


,

where c is a universal constant.

E.6
Proof of Lemma 3

Lemma 3. For any n ≥1, hi ≥0,τi→j ≥0 for all i, j ∈[n], and p ∈(0, 1], we have

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

≤1

8

for all j ∈[n], where

¯tj := 1

8 min
k∈[n] max




max{τπj,k→j, hπj,k},

 k
X

i=1

p
hπj,i

!−1


.
(46)

and πj,· is a permutation that sorts max{hi, τi→j}, i.e.,

max{hπj,1, τπj,1→j} ≤· · · ≤max{hπj,n, τπj,n→j}

for all j ∈[n].

Let us define k∗∈[n] as the largest index that minimizes (46).
(Part 1): bound ¯tj by max{τπj,k∗+1→j, hπj,k∗+1}
Let us consider the case k∗< n. We have two options:

1. Let max{τπj,k∗→j, hπj,k∗} ≥
Pk∗

i=1
p
hπj,i

−1
, then

max{τπj,k∗+1→j, hπj,k∗+1} ≥

 k∗+1
X

i=1

p
hπj,i

!−1

,
(49)

since max{τπj,i→j, hπj,i} are sorted. Then, we get

¯tj < 1

8 max




max{τπj,k∗+1→j, hπj,k∗+1},

 k∗+1
X

i=1

p
hπj,i

!−1




(49)
= 1

8 max{τπj,k∗+1→j, hπj,k∗+1} ≤max{τπj,k∗+1→j, hπj,k∗+1}

34


---Page Break---
The first inequality follows from the fact that k∗is the largest minimizer.

2.
Let
max{τπj,k∗→j, hπj,k∗}
<
Pk∗

i=1
p
hπj,i

−1
,
then
it
is
not
possible
that

max{τπj,k∗+1→j, hπj,k∗+1} <
Pk∗+1
i=1
p
hπj,i

−1
because it would yield the inequality

¯tj = 1

8

 k∗
X

i=1

p
hπj,i

!−1

≥1

8

 k∗+1
X

i=1

p
hπj,i

!−1

= 1

8 max




max{τπj,k∗+1→j, hπj,k∗+1},

 k∗+1
X

i=1

p
hπj,i

!−1


.

The last inequality contradicts the fact that k∗is the largest minimizer. Thus, if k∗< n and

max{τπj,k∗→j, hπj,k∗} <
Pk∗

i=1
p
hπj,i

−1
, then

¯tj < 1

8 max




max{τπj,k∗+1→j, hπj,k∗+1},

 k∗+1
X

i=1

p
hπj,i

!−1


= 1

8 max{τπj,k∗+1→j, hπj,k∗+1}.

In total, we have

¯tj < max{τπj,k∗+1→j, hπj,k∗+1}

if k∗< n. Using this inequality, we get

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= p

k∗
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

(50)

for any k∗.
(Part 2)
We have three options:

1. If
Pk∗

i=1
p
hπj,i

−1
≥max{τπj,k∗→j, hπj,k∗}, then, using (50) and ⌊x⌋≤x for all x ≥0, we get

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

≤p

k∗
X

i=1

¯tj
hπj,i
= 1

8

 k∗
X

i=1

p
hπj,i

!−1

p

k∗
X

i=1

1
hπj,i
= 1

8.

2. If
Pk∗

i=1
p
hπj,i

−1
< max{τπj,k∗→j, hπj,k∗} and k∗= 1, then

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= p

¯tj
hπj,k∗


1

τπj,k∗→j ≤¯tj

= 0

because

¯tj = 1

8 max




max{τπj,k∗→j, hπj,k∗},

 k∗
X

i=1

p
hπj,i

!−1


= 1

8 max{τπj,k∗→j, hπj,k∗} < max{τπj,k∗→j, hπj,k∗}.

3. If
Pk∗

i=1
p
hπj,i

−1
< max{τπj,k∗→j, hπj,k∗} and k∗> 1, then ¯tj = 1

8 max{τπj,k∗→j, hπj,k∗}
and
 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= 0

for all i ≤k∗such that max{τπj,i→j, hπj,i} = max{τπj,k∗→j, hπj,k∗}. If this equality holds for all
i ≤k∗, then

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= 0.

35


---Page Break---
Otherwise, there exists ℓ< k∗such that max{τπj,ℓ→j, hπj,ℓ} < max{τπj,k∗→j, hπj,k∗} and

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= p

ℓ
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

.

It is not possible that max{τπj,ℓ→j, hπj,ℓ} ≥
Pℓ
i=1
p
hπj,i

−1
because it would yield the inequality

¯tj = 1

8 max{τπj,k∗→j, hπj,k∗} > 1

8 max{τπj,ℓ→j, hπj,ℓ}

= 1

8 max




max{τπj,ℓ→j, hπj,ℓ},

 ℓ
X

i=1

p
hπj,i

!−1




that contradicts the fact that ¯tj is the minimum (see (46)). Thus, we have

¯tj ≤1

8 max




max{τπj,ℓ→j, hπj,ℓ},

 ℓ
X

i=1

p
hπj,i

!−1


= 1

8

 ℓ
X

i=1

p
hπj,i

!−1

and

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

= p

ℓ
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

≤p

ℓ
X

i=1

¯tj
hπj,i

≤1

8

 ℓ
X

i=1

p
hπj,i

!−1

p

ℓ
X

i=1

1
hπj,i
≤1

8.

In total, we have

p

n
X

i=1

 ¯tj

hπj,i


1

τπj,i→j ≤¯tj

≤1

8

for ¯tj from (46).

F
Lower Bound in the Heterogeneous Setup

In this case, we consider the following oracle mappings. For all i ∈[n], we define

Oi : R≥0
|{z}
time

× Rd
|{z}
point
× (R≥0 × Rd × {0, 1})
|
{z
}
input state

→(R≥0 × Rd × {0, 1})
|
{z
}
output state

×Rd

such that
Oi(t, x, (st, sx, sq)) =






((t, x, 1),
0),
sq = 0,
((st, sx, 1),
0),
sq = 1, t < st + hi,
((0, 0, 0),
∇fi(sx; ξ)),
sq = 1, t ≥st + hi,
(51)

where ξ ∼D. Unlike (27), the mapping (51) returns ∇fi(sx; ξ).
Theorem 20. Consider Protocol 8 with the mappings (51). We take any hi ≥0 and τi→j ≥0 for
all i, j ∈[n] such that τi→j ≤τi→k + τk→j for all i, k, j ∈[n]. We fix L, ∆, ε, σ2 > 0 that satisfy
the inequality ε < c1L∆. For any algorithm A ∈Azr, there exists a function f = 1

n
Pn
i=1 fi, which
satisfy Assumptions 1, 2 and f(0) −f ∗≤∆, and stochastic gradient mappings ∇fi(·; ·), which
satisfy Assumption 3 (Eξ[∇fi(x; ξ)] = ∇fi(x) and Eξ[∥∇fi(x; ξ) −∇fi(x)∥2] ≤σ2), such that

E

inf
k∈St,i∈[n]

∇f(xk
i )
2

> ε, where St :=

k ∈N0 | tk ≤t
	
,

t = c2 × L∆

ε
max

(

max
i,j∈[n] τi→j, max
i∈[n] hi, σ2

nε

 
1
n

n
X

i=1
hi

!)

,

The quantities c1 and c2 are universal constants. The sequences xk and tk are defined in Protocol 8.

36


---Page Break---
Proof. The last two terms in the max follow from Theorem A.2 by Tyurin and Richtárik (2023), who
considered the same setup but with τi→j = 0 for all i, j ∈[n]. It is left to prove the first term. Let us
fix λ > 0. Let us take any pair (¯i, ¯j) of workers such that maxi,j∈[n] τi→j = τ¯i→¯j. Next, we split the
blocks of the function FT (x) from (33) and define two new functions:

FT,1(x) := −Ψ(1)Φ(x1) +
X

i∈{2,...,T },i | 2=1
[Ψ(−xi−1)Φ(−xi) −Ψ(xi−1)Φ(xi)] ,
(52)

and

FT,2(x) :=
X

i∈{2,...,T },i | 2=0
[Ψ(−xi−1)Φ(−xi) −Ψ(xi−1)Φ(xi)] .

We consider the following functions fi :

fi(x) :=








nLλ2

l1
FT,1
  x

λ

,
i = ¯i,

nLλ2

l1
FT,2
  x

λ

,
i = ¯j,
0,
i ̸= ¯i and i ̸= ¯j.

Then, we get

f(x) = 1

n

n
X

i=1
fi(x) = 1

n

nLλ2

l1
FT,1
x

λ


+ nLλ2

l1
FT,2
x

λ


= Lλ2

l1
FT
x

λ


.

Let us show that the function f is L-smooth:

∥∇f(x) −∇f(y)∥= Lλ

l1

∇FT
x

λ


−∇FT
y

λ

 ≤L ∥x −y∥.

Let us take

T =

∆l1
Lλ2∆0


,

then

f(0) −inf
x∈RT f(x) = Lλ2

l1
(FT (0) −inf
x∈RT FT (x)) ≤Lλ2∆0T

l1
≤∆.

We showed that the function f satisfy Assumptions 1, 2 and f(0) −f ∗≤∆.

In the oracles Oi, we simply take the non-stochastic mappings ∇fi(x; ξ) := ∇fi(x) that are unbiased
and 0-variance-bounded.

We take

λ = l1
√ε
L
to ensure that

∥∇f(x)∥2 = L2λ2

l2
1

∇FT
x

λ


2
> L2λ2

l2
1
= ε

for all x ∈RT such that prog(x) < T. In the last inequality, we use Lemma 1. Thus

T =
 ∆L

l1ε∆0


.

Only workers ¯i and ¯j contain the information about the function f. The function f is a zero-chain
function, and we split it between workers ¯i and ¯j. Due to this splitting, workers ¯i and ¯j have to
communicate to find the next non-zero coordinate. Only worker ¯i can get a non-zero value in the
first coordinate through the gradient of FT,1. Next, this worker can not get a non-zero value in the
second coordinate due to the construction of (52). Thus, it has to pass a vector with a non-zero value
in the first coordinate to worker ¯j because only this worker can get a non-zero value in the second
coordinate. This communication takes at least τ¯i→¯j seconds. Using the same reasoning, worker ¯j has
to send a vector to worker ¯i once worker ¯j has discovered a non-zero value in the second coordinate.

37


---Page Break---
An algorithm has to repeat such communications at least T −1

2
times to find a vector x ∈RT such
that prog(x) = T.

Thus, we get

inf
k∈St,i∈[n]

∇f(xk
i )
2 > ε

for

t = τ¯i→¯j

T −1

2


= maxi,j∈[n] τi→j

2

 ∆L

l1ε∆0


−1

.

G
Proof of the Time Complexity for Homogeneous Case

Theorem 4. Let Assumptions 1, 2, and 3 hold. We take γ = 1/2L, batch size S = max{

σ2/ε

, 1},
any pivot worker j∗∈[n], and any spanning trees st and stbc in Algorithm 2. For all K ≥16L∆/ε,

we get 1

K
PK−1
k=0 E
h∇f(xk)
2i
≤ε.

Proof. Algorithm 2 produces the sequence xk such as xk+1 = xk −
γ
sk gk = xk −γ¯gk, where
¯gk :=
1
sk gk. By the design of the algorithm,

¯gk = 1

sk

sk
X

i=1
∇f(xk; ξi),

where the ξi are independent random samples and sk ≥S. We do not dismiss the possibility that the
computation and communication times are random, so sk can be random. Assume Vk is a σ-algebra
generated by all computation and communication times and g0, . . . , gk−1, then sk is Vk–measurable.
Using the independence and Assumption 3, we have

E

¯gk Gk

= E



E



1

sk

sk
X

i=1
∇f(xk; ξi)


Vk






Gk



= E



1

sk

sk
X

i=1
E

∇f(xk; ξi)
 Vk


Gk



= ∇f(xk)

and

E
h¯gk −∇f(xk)
2 Gk
i

= E



E







1
sk

sk
X

i=1
∇f(xk; ξi) −∇f(xk)



2
Vk






Gk





= E




1
(sk)2

sk
X

i=1
E
h∇f(xk; ξi) −∇f(xk)
2 Vk
i

Gk





≤E
 σ2

sk

 Gk


≤σ2

S .

where Gk is a σ-algebra generated by ¯g0, . . . , ¯gk−1. We can use a well-known SGD result (Ghadimi
and Lan, 2013; Khaled and Richtárik, 2022). Using Theorem 21, for the stepsize

γ = 1

2L min

1, εS

σ2


,

we have

1
K

K−1
X

k=0
E
h∇f(xk)
2i
≤ε,

38


---Page Break---
if

K ≥8∆L

ε
+ 8∆Lσ2

ε2S
.

Using the choice of S, we get that Algorithm 2 converges after

K ≥16∆L

ε
steps with

γ = 1

2L min

1, εS

σ2


= 1

2L.

Theorem 5. Consider the assumptions and the parameters from Theorem 4. For any pivot worker
j∗∈[n] and spanning trees st and stbc, Algorithm 2 converges after at most

Θ
  L∆

ε t∗(σ2/ε, [hi]n
i=1, [µi→j∗+ µj∗→i]n
i=1)

(10)

seconds, where µi→j∗(µj∗→i) is an upper bound on the times required to send a vector from worker
i to worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc).

Proof. Due to Theorem 4, we know that Algorithm 3 finds an ε–stationary point after at most
K = Θ
  L∆

ε

iterations. It is left to bound the time of one iteration to prove the theorem.

For all j ∈[n], we define πj,· as a permutation that sorts {max{µi→j + µj→i, hi}}n
i=1 as

max{µπj,1→j + µj→πj,1, hπj,1} ≤· · · ≤max{µπj,n→j + µj→πj,n, hπj,n}.

Let us define the index

k∗= arg min
k∈[n] max




max{µπj∗,k→j∗+ µj∗→πj∗,k, hπj∗,k}, S

 k
X

i=1

1
hπj∗,i

!−1


.

and the set

A∗:= {πj∗,i ∈[n] | i ≤k∗}

that represents a set of the “fastest” workers that can potentially contribute to an optimization process.
We take

¯t := 2 max




max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗}, S

 k∗
X

i=1

1
hπj∗,i

!−1




= 2 min
k∈[n] max




max{µπj∗,k→j∗+ µj∗→πj∗,k, hπj∗,k}, S

 k
X

i=1

1
hπj∗,i

!−1


.

(53)

In the following steps of the proof we show that every iteration takes at most

¯t
|{z}
(Step 1): Calculate enough stochastic gradients
+
¯t
|{z}
(Step 2): Send stochastic gradients to Process j∗
+
¯t
|{z}
(Step 3): Broadcast a new point
= 3¯t

seconds.

(Step 1): Since it takes at most hi seconds to calculate a stochastic gradient in worker i, all workers
from the set A∗will calculate at least

X

i∈A∗

 ¯t

hi


=

k∗
X

i=1


¯t
hπj∗,i


(54)

stochastic gradients after ¯t seconds at the point xk. We have

¯t ≥2 max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗} ≥2 max{µπj∗,i→j∗+ µj∗→πj∗,i, hπj∗,i}
(55)

39


---Page Break---
for all i ≤k∗by the definition of the permutations π·,·. Therefore,
¯t ≥2hπj∗,i
for all i ≤k∗. Thus, using (54) and ⌊x⌋≥x

2 for all x ≥1, we get

X

i∈A∗

 ¯t

hi


≥

k∗
X

i=1

¯t
2hπj∗,i

(53)
≥S.

Therefore, after ¯t seconds the algorithm will calculate at least S stochastic gradients at the point xk
using the workers A∗.

(Step 2): By the design of Algorithm 4, once a stochastic gradient ∇f(xk; ¯ξ) is calculated, it is added
to gk
i,next. Then, gk
i,next is assigned to gk
i,send which is sent to Process nextst,j∗(i). Finally, Process
nextst,j∗(i) receives gk
i,send and adds it to gk
(nextst,j∗(i)),next. Thus, the stochastic gradient ∇f(xk; ¯ξ) is

presented in the sum of gk
(nextst,j∗(i)),next of Process nextst,j∗(i). At some point, Process 0 in worker

j∗will receive a vector gk
·,send where the stochastic gradient ∇f(xk; ¯ξ) is presented.

Let us bound the time required to transmit a stochastic gradient to Process 0 of worker j∗. Once
a stochastic gradient ∇f(xk; ¯ξ) is calculated in Process i from A∗, it is added to a vector gk
i,next in
Process i. It will take at most 2ρi→nextst,j∗(i) seconds to transmit it to Process nextst,j∗(i) because
it takes at most ρi→nextst,j∗(i) seconds to wait for the transmission of a message gk
i,send where the
stochastic gradient ∇f(xk; ¯ξ) is not presented, and an additional ρi→nextst,j∗(i) seconds to send
the next gk
i,send where it will present. After that, Process nextst,j∗(i) will receive gk
i,send, where the
stochastic gradient ∇f(xk; ¯ξ) presents, and add the vector gk
i,send to gk
(nextst,j∗(i)),next. Then, it will
take at most 2ρnextst,j∗(i)→next(nextst,j∗(i)) seconds to send a vector, where the stochastic gradient
∇f(xk; ¯ξ) presents, to Process next(nextst,j∗(i)) and so forth. In total, after a finite number of such
steps a stochastic gradient calculated in Process i will be transmitted to Process 0 of worker j∗. From
Definition 3 of nextst,j∗, we can conclude that the vector ∇f(xk; ¯ξ) will be transmitted through the
path between workers i and j∗in the spanning tree st. Thus, it will take at most 2µi→j∗seconds by
the definition of µi→j∗.

Using (55) and the definition of A∗, we have
¯t ≥2 max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗} ≥2 max{µi→j∗+ µj∗→i, hi} ≥2µi→j∗
(56)

for all i ∈A∗. Therefore, it will take at most ¯t seconds to calculate at least S stochastic gradients,
and at most ¯t seconds to send all these stochastic gradients to Process 0.

(Step 3): It is left to estimate the time of the broadcast steps (Lines 7–9 in Algorithm 4) through the
spanning tree stbc. By the definition of µj∗→i, the time required to broadcast xk to Process i through
the spanning tree stbc is less or equal to 2µj∗→i since, in all edges from j∗to i, workers wait at most
ρ·→· seconds while edges are blocked by previous communications, and additional ρ·→· seconds to
send xk to next workers. Using (55) and the definition of A∗, we have
¯t ≥2 max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗} ≥2 max{µi→j∗+ µj∗→i, hi} ≥2µj∗→i.

Thus, every worker from A∗will get xk after ¯t seconds. By combining all times, we can conclude
that every iteration in Algorithm 3 will take at most ¯t + ¯t + ¯t = 3¯t seconds.

It left to show that

¯t = O



max




max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗}, σ2

ε

 k∗
X

i=1

1
hπj∗,i

!−1






.
(57)

If S > 1, then S = max{

σ2/ε

, 1} =

σ2/ε

≤2σ2/ε, so it is true. Otherwise, if S ≤1, then

¯t ≤2 max




max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗},

 k∗
X

i=1

1
hπj∗,i

!−1




≤2 max

max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗}, hπj∗,k∗
	

= 2 max{µπj∗,k∗→j∗+ µj∗→πj∗,k∗, hπj∗,k∗}

40


---Page Break---
and (57) holds. Notice that the r.h.s. of (57) equals to O
 
t∗(σ2/ε, [hi]n
i=1, [µi→j∗+ µj∗→i]n
i=1)

,
where t∗is the equilibrium time defined in Definition 2.

Theorem 6. Consider the assumptions and the parameters from Theorem 4. In each iteration k
of Algorithm 3, the computation times of worker i are bounded by hk
i . Let us fix any pivot worker
j∗∈[n] and any spanning trees st and stbc. Then Algorithm 2 converges after at most

Θ
P⌈16L∆/ε⌉
k=0
t∗(σ2/ε, [hk
i ]n
i=1, [µk
i→j∗+ µk
j∗→i]n
i=1)

(13)

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc) in iteration k
of Algorithm 3.

Proof. The proof is almost the same as in Theorem 5. If we fix a pivot worker j∗, then the kth
iteration will finish after at most

c × t∗(σ2/ε, [hk
i ]n
i=1, [µk
i→j∗+ µk
j∗→i]n
i=1)

seconds, where c is a universal constant. According to Theorem 4, the number of iterations is at most
 16L∆

ε

. Therefore, the total required time is at most

c ×
⌈16L∆

ε
⌉
X

k=0
t∗(σ2/ε, [hk
i ]n
i=1, [µk
i→j∗+ µk
j∗→i]n
i=1).

H
Proof of the Time Complexity for Heterogeneous Case

Theorem 8. Let Assumptions 1 and 2 hold for the function f and Assumption 3 holds for the functions
fi for all i ∈[n]. We take γ = 1/2L, the parameter S = max{

σ2/ε

, n}, any pivot worker j∗∈[n],
and any spanning trees st and stbc, in Algorithm 5. For all iterations number K ≥16L∆/ε, we get

1
K
PK−1
k=0 E
h∇f(xk)
2i
≤ε.

Proof. Algorithm 5 produces the sequence xk such that

xk+1 = xk −γgk = xk −γ

 
1
n

n
X

i=1

1
sk
i
gk
i

!

= xk −γ



1

n

n
X

i=1

1
sk
i

sk
i
X

j=1
∇fi(xk; ξij)



,

where the ξij are independent random samples. Using the independence and Assumption 3, we have
E

gk Gk

= ∇f(xk) and

E
hgk −∇f(xk)
2 Gk
i

= E







1
n

n
X

i=1

1
sk
i

sk
i
X

j=1
∇fi(xk; ξij) −1

n

n
X

i=1
∇fi(xk)



2
Gk



.
(58)

where Gk is a σ-algebra generated by g0, . . . , gk−1. We do not dismiss the possibility that the
computation and communication times are random, so sk
i can be random. Assume Vk is a σ-algebra
generated by all computation and communication times and g0, . . . , gk−1, then sk
i is Vk–measurable
for all i ∈[n]. Using the independence of stochastic gradients and the times and the tower property,
we get

41


---Page Break---
E
hgk −∇f(xk)
2 Gk
i

= E



E







1
n

n
X

i=1

1
sk
i

sk
i
X

j=1
∇fi(xk; ξij) −1

n

n
X

i=1
∇fi(xk)



2
Vk






Gk




(59)

= 1

n2

n
X

i=1
E



E







1
sk
i

sk
i
X

j=1
∇fi(xk; ξij) −∇fi(xk)



2
Vk






Gk





= 1

n2

n
X

i=1
E




1
(sk
i )2

sk
i
X

j=1
E
h∇fi(xk; ξij) −∇fi(xk)
2 Vk
i

Gk



≤σ2

n2 E

"
n
X

i=1

1
sk
i

 Gk

#

. (60)

Algorithm 6 waits for the moment when sk ≥S

n, which is equivalent to

bj∗≤n2

S .
(61)

The value bj∗is calculated in Line 20 of Algorithm 7. Due to the asynchronous nature of the
algorithm, we can only conclude that

bj∗≥
X

p∈[n]:nextst,j∗(p)=j∗
bi,p + 1

sk
j∗
(62)

because sk
j∗can be increased by the time when Process 0 will receive bj∗. Using Line 15 from
Algorithm 7, we can unroll the recursion in (62) and get

bj∗≥

n
X

i=1

1
sk
i
.
(63)

Let us substitute this inequality to (61) and (60) and get

E
hgk −∇f(xk)
2 Gk
i
≤σ2

S .

As in Theorem 4, we can use the classical SGD result. Using Theorem 21, for the stepsize

γ = 1

2L min

1, εS

σ2


,

we have

1
K

K−1
X

k=0
E
h∇f(xk)
2i
≤ε,

if

K ≥8∆L

ε
+ 8∆Lσ2

ε2S
.

Using the choice of S, we get that Algorithm 5 converges after

K ≥16∆L

ε

steps with

γ = 1

2L min

1, εS

σ2


= 1

2L.

42


---Page Break---
Theorem 9. Consider the assumptions and the parameters from Theorem 8. For any pivot worker
j∗∈[n] and any spanning trees st and stbc, Algorithm 5 converges after at most

Θ

 
L∆

ε
max

(

max
i,j∈[n] µi→j, max
i∈[n] hi, σ2

nε

 
1
n

n
X

i=1
hi

!)!

seconds, where µk
i→j∗(µk
j∗→i) is an upper bound on times required to send a vector from worker i to
worker j∗(from worker j∗to worker i) along the spanning tree st (spanning tree stbc) for all i ∈[n].

Proof. Due to Theorem 8, the algorithm converges after K = Θ (L∆/ε) iterations. Thus, it left to
bound the time of one iteration. At the beginning of every iteration Process 0 broadcasts xk, which
takes at most maxi,j∈[n] µi→j seconds. Then, Algorithm 6 waits for the moment when sk ≥S

n,
which is equivalent to n2

S ≥bj∗. Thus, Algorithm 6 waits for the moment when n2

S ≥bj∗. We will
return to this fact later.

Let us consider the term

S
n2

n
X

i=1

1
sk
i
,

where sk
i is the number of stochastic gradients calculated in worker i. Let us fix any time ¯t > 0. Then,

worker i will calculate at least
j
¯t
hi

k
stochastic gradients by the time ¯t. Using this, we get

S
n2

n
X

i=1

1
sk
i
≤S

n2

n
X

i=1

1
j
¯t
hi

k.

Let us take

¯t = 2

 

max
i∈[n] hi + S

n

 
1
n

n
X

i=1
hi

!!

.

Then, since ⌊x⌋≥x

2 for all x ≥1, we get
j
¯t
hi

k
≥
¯t
2hi and

S
n2

n
X

i=1

1
sk
i
≤2S

n¯t

 
1
n

n
X

i=1
hi

!

.

Using ¯t ≥2S

n
  1

n
Pn
i=1 hi

, we get

S
n2

n
X

i=1

1
sk
i
≤1

and

n
X

i=1

1
sk
i
≤n2

S
(64)

after at most ¯t seconds.

Recall that Algorithm 6 waits for the moment when n2

S ≥bj∗. Note that by the time when Process 0 re-
ceives bj∗, the counter sk
i can be the same or increased; thus, bj∗captures potentially outdated informa-
tion about sk
i . We know that (64) holds after at most ¯t seconds. In Line 20 of Algorithm 7, Processes re-
cursively collect 1

sk
i to bj∗. Such a procedure will take at most 2 maxi∈[n] µi→j∗≤2 maxi,j∈[n] µi→j

seconds. Thus, the value bj∗will be less or equal n2

S after at most ¯t + 2 maxi,j∈[n] µi→j seconds.

43


---Page Break---
The all reduce operation in (8) will take at most maxi,j∈[n] µi→j seconds. Thus, the total time of one
iteration can be bounded by

max
i,j∈[n] µi→j
|
{z
}
broadcast

+(¯t + 2 max
i,j∈[n] µi→j) + max
i,j∈[n] µi→j
|
{z
}
all reduce

= O

 

max
i,j∈[n] µi→j + max
i∈[n] hi + S

n

 
1
n

n
X

i=1
hi

!!

= O

 

max
i,j∈[n] µi→j + max
i∈[n] hi + σ2

nε

 
1
n

n
X

i=1
hi

!!

.

seconds.

I
Classical SGD Theory

We reprove the classical SGD result (Ghadimi and Lan, 2013; Khaled and Richtárik, 2022), for
completeness.
Theorem 21. Let Assumptions 1 and 2 hold. We consider the SGD method:

xk+1 = xk −γg(xk),

where

γ = 1

2L min
n
1, ε

σ2

o

For all k ≥0, the vector g(x) is a random vector such that E

g(xk)
 Gk

= ∇f(xk),

E
hg(xk) −∇f(xk)
2 Gk
i
≤σ2,
(65)

where Gk is a σ-algebra generated by g(x0), . . . , g(xk−1). Then

1
K

K−1
X

k=0
E
h∇f(xk)
2i
≤ε

for

K ≥8∆L

ε
+ 8∆Lσ2

ε2
.

Proof. From Assumption 1, we have

f(xk+1) ≤f(xk) +

∇f(xk), xk+1 −xk
+ L

2

xk+1 −xk2

= f(xk) −γ

∇f(xk), g(xk)

+ Lγ2

2

g(xk)
2 .

We denote Gk as a sigma-algebra generated by g(x0), . . . , g(xk−1). Using unbiasedness and (65),
we obtain

E

f(xk+1)
 Gk
≤f(xk) −γ

1 −Lγ

2

 ∇f(xk)
2 + Lγ2

2 E
hg(xk) −∇f(xk)
2 Gki

≤f(xk) −γ

1 −Lγ

2

 ∇f(xk)
2 + Lγ2σ2

2
.

Since γ ≤1/L, we get

E

f(xk+1)
 Gk
≤f(xk) −γ

2

∇f(xk)
2 + Lγ2σ2

2
.

44


---Page Break---
We subtract f ∗and take the full expectation to obtain

E

f(xk+1) −f ∗
≤E

f(xk) −f ∗
−γ

2 E
h∇f(xk)
2i
+ Lγ2σ2

2
.

Next, we sum the inequality for k ∈{0, . . . , K −1}:

E

f(xK) −f ∗
≤f(x0) −f ∗−

K−1
X

k=0

γ
2 E
h∇f(xk)
2i
+ KLγ2σ2

2

= ∆−

K−1
X

k=0

γ
2 E
h∇f(xk)
2i
+ KLγ2σ2

2
.

Finally, we rearrange the terms and use that E

f(xK) −f ∗
≥0:

1
K

K−1
X

k=0
E
h∇f(xk)
2i
≤2∆

γK + Lγσ2.

The choice of γ and K ensures that

1
K

K−1
X

k=0
E
h∇f(xk)
2i
≤ε.

Theorem 22. Let Assumptions 4 and 5 hold. We consider the SGD method:

xk+1 = xk −γg(xk),

where

γ =
ε
M 2 + σ2

For all k ≥0, the vector g(x) is a random vector such that E

g(xk)
 Gk

∈∂f(xk)

E
hg(xk) −E

g(xk)
 Gk
2 Gk
i
≤σ2,

where Gk is a σ-algebra generated by g(x0), . . . , g(xk−1). Then

E

"

f

 
1
K

K−1
X

k=0
xk
!#

−f(x∗) ≤ε
(66)

for

K ≥(M 2 + σ2)
x∗−x02

ε2
.

Proof. We denote Gk as a sigma-algebra generated by g(x0), . . . , g(xk−1). Using the convexity, for
all x ∈Rd, we have

f(x) ≥f(xk) +

E

g(xk)
 Gk
, x −xk
= f(xk) + E

g(xk), x −xk Gk
.

Note that

g(xk), x −xk
=

g(xk), xk+1 −xk
+

g(xk), x −xk+1

= −γ
g(xk)
2 + 1

γ

xk −xk+1, x −xk+1

= −γ
g(xk)
2 + 1

2γ

xk −xk+12 + 1

2γ

x −xk+12 −1

2γ

x −xk2

= −γ

2

g(xk)
2 + 1

2γ

x −xk+12 −1

2γ

x −xk2

45


---Page Break---
and

E
hg(xk)
2 Gki
= E
hg(xk) −E

g(xk)
 Gk2 Gki
+
E

g(xk)
 Gk2 ≤σ2 + M 2.

Therefore, we get

f(xk) ≤f(x) + E

g(xk), xk −x
 Gk

= f(x) + γ

2 E
hg(xk)
2 Gki
+ 1

2γ

x −xk2 −1

2γ E
hx −xk+12 Gki

≤f(x) + γ

2
 
M 2 + σ2
+ 1

2γ

x −xk2 −1

2γ E
hx −xk+12 Gki
.

By taking the full expectation and summing the last inequality for t from 0 to K −1, we obtain

E

"K−1
X

k=0
f(xk)

#

≤Kf(x) + Kγ

2
 
M 2 + σ2
+ 1

2γ

x −x02 −1

2γ E
hx −xK2i

≤Kf(x) + Kγ

2
 
M 2 + σ2
+ 1

2γ

x −x02 .

Let divide the last inequality by K, take x = x∗, and use the convexity:

E

"

f

 
1
K

K−1
X

k=0
xk
!#

−f(x∗) ≤γ

2
 
M 2 + σ2
+
1
2γK

x∗−x02 .

The choices of γ and K ensure that (66) holds.

J
Experiments

We now consider Fragile SGD with Minibatch SGD on quadratic optimization tasks with stochastic
gradients. The working environment was emulated in Python 3.8 with one Intel(R) Xeon(R) Gold
6248 CPU @ 2.50GHz. The homogeneous optimization problem (1) is constructed in the following
way. We take

f(x) = 1

2x⊤Ax −b⊤x
∀x ∈Rd,

d = 1000,

A = 1

4








2
−1
0

−1
...
...
...
...
−1
0
−1
2






∈Rd×d,
and
b = 1

4





−1
0
...
0



∈Rd.

Let us define [x]j as the jth index of a vector x ∈Rd. All n workers calculate the stochastic gradients

[∇f(x; ξ)]j := [∇f(x)]j


1 + 1 [j > prog(x)]
ξ

p −1

∀x ∈Rd, ∀i ∈[n],

where ξ ∼Bernouilli(p), p ∈(0, 1]. In our experiments, we take p = 0.001 and the starting point
x0 = [
√

d, 0, . . . , 0]⊤. We emulate our setup by considering that the ith worker requires hi = 1
second to calculate a stochastic gradient. And we assume that the workers have the structure of 2D-
Mesh (see Figure 4a) and take ρi→j = ρ ∈{0.1, 1, 10} seconds for all edges that connect workers in
2D-Mesh. We take n = 100. In all methods we fine-tune step sizes from the set {2i | i ∈[−20, 20]}.
In Fragile SGD, we fine-tune the batch size S from the set {10, 20, 40, 80, 120}.

The results are presented in Figures 5, 6, and 7. The plots are fully consisted with Table 1. One can
see that when the communication is fast (Fig. 5), there is no big difference between the methods
because both Fragile SGD and Minibatch SGD use all workers in the optimization steps. However,
when we start decreasing the communication speed, we observe that Fragile SGD converges faster.

46


---Page Break---
We looked deeper into the optimization processes of Fragile SGD in Figure 7 and observed that only
13 of 100 workers contribute to the optimization process for the batch size S = 120. Other workers
are too far away from the pivot worker, and their contributions can only slow down optimization.

0
20000
40000
60000
80000
100000
times (seconds)

10
3

10
2

f(xt)
f(x * )

Fragile SGD: Batch Size: 40 Step size: 2.0
Fragile SGD: Batch Size: 80 Step size: 2.0
Fragile SGD: Batch Size: 120 Step size: 2.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 0.5
Minibatch SGD: Step size: 0.125

Figure 5: The communication time ρ = 0.1 seconds (Fast communication)

0
20000
40000
60000
80000
100000
times (seconds)

10
3

10
2

f(xt)
f(x * )

Fragile SGD: Batch Size: 40 Step size: 2.0
Fragile SGD: Batch Size: 80 Step size: 2.0
Fragile SGD: Batch Size: 40 Step size: 1.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 0.5

Figure 6: The communication time ρ = 1 seconds (Medium speed communication)

0
20000
40000
60000
80000
100000
times (seconds)

10
3

10
2

f(xt)
f(x * )

Fragile SGD: Batch Size: 120 Step size: 2.0
Fragile SGD: Batch Size: 80 Step size: 2.0
Fragile SGD: Batch Size: 200 Step size: 2.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 0.5

Figure 7: The communication time ρ = 10 seconds (Slow communication)

47


---Page Break---
J.1
Experiments with Logistic Regression: Fast vs Slow Communication

We now repeat the previous experiments but with logistic regression on MNIST dataset (LeCun et al.,
2010) with 100 workers. We consider two regimes: fast and slow communication between workers.
One can see that when the communication is fast, the gap between the methods is small, which is
expected and compliant with the theory. However, Fragile SGD is much faster and has better test
accuracy when the communication is slow.

0
50000 100000 150000 200000 250000 300000 350000 400000

times (seconds)

2.2 × 10
1

2.4 × 10
1

2.6 × 10
1

2.8 × 10
1

3 × 10
1

f(xt)
f(x * )

Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 0.5
Minibatch SGD: Step size: 0.25

0
50000 100000 150000 200000 250000 300000 350000 400000

times (seconds)

0.920

0.925

0.930

0.935

0.940

0.945

0.950

Accuracy (Test)

Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 0.5
Minibatch SGD: Step size: 0.25

Figure 8: The communication time ρ = 0.1 seconds (Fast communication)

0.0
0.2
0.4
0.6
0.8
1.0
times (seconds)
1e6

2.2 × 10
1

2.4 × 10
1

2.6 × 10
1

2.8 × 10
1

3 × 10
1

f(xt)
f(x * )

Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 0.5

0.0
0.2
0.4
0.6
0.8
1.0
times (seconds)
1e6

0.910

0.915

0.920

0.925

0.930

0.935

0.940

Accuracy (Test)

Fragile SGD: Batch Size: 120 Step size: 1.0
Fragile SGD: Batch Size: 160 Step size: 1.0
Fragile SGD: Batch Size: 100 Step size: 1.0
Minibatch SGD: Step size: 1.0
Minibatch SGD: Step size: 2.0
Minibatch SGD: Step size: 0.5

Figure 9: The communication time ρ = 10 seconds (Slow communication)

J.2
Experiments with ResNet-18

We test algorithms on an image recognition task, CIFAR10 (Krizhevsky et al., 2009), with the ResNet-
18 (He et al., 2016) deep neural network (the number of parameters d ≈107). We use the torus
structure and 9 workers. We run all methods with the step sizes {0.025, 0.25, 2.5}. Our findings from
the low-scale experiments are also evident in the large-scale experiments. Fragile SGD converges
faster than Minibatch SGD in terms of function values. When we compare accuracies on the test split
of MNIST, the superiority of Fragile SGD is even more transparent.

0
5000
10000
15000
20000
25000
30000
times (seconds)

10
3

10
2

10
1

100

f(xt)
f(x * )

Fragile SGD: Step size: 0.25
Fragile SGD: Step size: 0.025
Fragile SGD: Step size: 2.5
Minibatch SGD: Step size: 0.25
Minibatch SGD: Step size: 0.025
Minibatch SGD: Step size: 2.5

0
5000
10000
15000
20000
25000
30000
times (seconds)

0.70

0.75

0.80

0.85

0.90

0.95

Accuracy (Test)

Fragile SGD: Step size: 0.25
Fragile SGD: Step size: 0.025
Fragile SGD: Step size: 2.5
Minibatch SGD: Step size: 0.25
Minibatch SGD: Step size: 0.025
Minibatch SGD: Step size: 2.5

Figure 10: ResNet-18 on CIFAR10 dataset with 9 workers and the torus structure with the communi-
cation time ρ = 1 seconds (Medium communication)

48


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Table 1, the main part of the paper, Section C

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

Justification: Section 5.3

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

49


---Page Break---
Justification: Section1.1, and the appendix
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
Justification: Section J
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

50


---Page Break---
Answer: [Yes]

Justification: The code in the supplementary materials.

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

Justification: Section J

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

Justification: We provide the top 3 best plots for each algorithm to reduce randomness
factors in Section J.

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

51


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

Justification: Section J

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

Justification: We have read the code of ethics, and our paper does not violate it.

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

Justification: We consider a mathematical problem for machine learning.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

52


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

Justification: We consider a mathematical problem for machine learning.

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

53


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The code and documentation in the supplementary materials.
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

54


---Page Break---
