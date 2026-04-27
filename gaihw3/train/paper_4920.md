Hierarchical Hybrid Sliced Wasserstein: A Scalable
Metric for Heterogeneous Joint Distributions

Khai Nguyen
Department of Statistics and Data Sciences
The University of Texas at Austin
Austin, TX 78712
khainb@utexas.edu

Nhat Ho
Department of Statistics and Data Sciences
The University of Texas at Austin
Austin, TX 78712
minhnhat@utexas.edu

Abstract

Sliced Wasserstein (SW) and Generalized Sliced Wasserstein (GSW) have been
widely used in applications due to their computational and statistical scalability.
However, the SW and the GSW are only defined between distributions supported on
a homogeneous domain. This limitation prevents their usage in applications with
heterogeneous joint distributions with marginal distributions supported on multiple
different domains. Using SW and GSW directly on the joint domains cannot make
a meaningful comparison since their homogeneous slicing operator, i.e., Radon
Transform (RT) and Generalized Radon Transform (GRT) are not expressive
enough to capture the structure of the joint supports set. To address the issue,
we propose two new slicing operators, i.e., Partial Generalized Radon Transform
(PGRT) and Hierarchical Hybrid Radon Transform (HHRT). In greater detail,
PGRT is the generalization of Partial Radon Transform (PRT), which transforms
a subset of function arguments non-linearly while HHRT is the composition of
PRT and multiple domain-specific PGRT on marginal domain arguments. By
using HHRT, we extend the SW into Hierarchical Hybrid Sliced Wasserstein
(H2SW) distance which is designed specifically for comparing heterogeneous
joint distributions. We then discuss the topological, statistical, and computational
properties of H2SW. Finally, we demonstrate the favorable performance of H2SW
in 3D mesh deformation, deep 3D mesh autoencoders, and datasets comparison1.

1
Introduction

Optimal transport [55, 47] is a powerful mathematical tool for machine learning, statistics, and
data sciences. As an example, Wasserstein distance [47], defined as the optimal transportation
cost between two distributions, has been used successfully in many areas of machine learning and
statistics, such as generative modeling on images [2, 53], representation learning [35], vocabulary
learning [57], and so on. Despite being accepted as an effective distance, Wasserstein distance has
been widely known as a computationally expensive distance. In particular, when comparing two
distributions that have at most n supports, the time complexity and the memory complexity of the
Wasserstein distance scale with the order of O(n3 log n) [45] and O(n2) respectively. In addition,
the Wasserstein distance requires more samples to approximate a continuous distribution with its
empirical distribution in high dimension since its sample complexity is of the order of O(n−1/d) [20],
where n is the sample size and d is the number of dimensions. Therefore, Wasserstein distance is not
statistically and computationally scalable, especially in high dimensions.

Along with entropic regularization [17] which can reduce the time complexity and memory complex-
ity of computing optimal transport to O(n2) and O(n2) in turn, sliced Wasserstein (SW) distance [11]

1Code for this paper is published at https://github.com/khainb/H2SW.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
is one alternative approach for the original Wasserstein distance. The key benefit of the SW distance
is that it scales the order O(n log n) and O(n) in terms of time and memory respectively. The reason
behind that fast computation is the closed-form solution of optimal transport in one dimension. To
leverage that closed-form, sliced Wasserstein utilizes Radon Transform [23] (RT) to transform a
high-dimensional distribution to its one-dimensional projected distributions, then the final distance
is calculated as the average of all one-dimensional Wasserstein distance. By doing that, the SW
distance has a very fast sample complexity i.e., O(n−1/2), which makes it computationally and
statistically scalable in any dimension. Therefore, the SW distance has been applied successfully in
various domains of applications including generative models [19], domain adaptation [32], cluster-
ing [27], 3D shapes [30, 29], gradient flows [34, 8], Bayesian inference computation [37, 58], texture
synthesis [22], and many other tasks.

Despite being useful, the SW distance is not as flexible as the Wasserstein distance in terms of choos-
ing the ground metric. In greater detail, the number of ground metrics in one dimension is limited,
especially ground metrics that lead to the closed-form solution. As a result, the role of capturing the
structure of distributions belongs to the slicing/projecting operators. To generalize RT to non-linear
projection, generalized Radon Transform (GRT) is introduced in [3] with circular projection [28],
polynomial projection [51], and so on. With GRT, Generalized Sliced Wasserstein (GSW) distance is
proposed in [26]. In addition, there is a line of works on developing sliced Wasserstein variants on
different manifolds such as hyper-sphere [6, 54, 49, 50], hyperbolic manifolds [7], the manifold of
symmetric positive definite matrices [10], general manifolds and graphs [52]. In those works, special
variants of GRT are proposed.

Although the SW has become more effective on multiple domains, no SW variant is designed
specifically for heterogeneous joint distributions i.e., joint distributions that have marginals supported
on different domains, except for the product of Hadamard manifolds [9]. It is worth noting that
marginal domains of heterogeneous joint distributions can be any metric space and are not necessary
manifolds. Heterogeneous joint distributions appear in many applications, e.g., domain adaptation
domains [15, 4], comparing datasets with labels [1], 3D shape deformation [29], and so on. In this
case, Wasserstein distance can be adapted by using a mixed ground metric, i.e., a weighted sum of
metrics on domains [15, 1]. In contrast to the Wasserstein distance, the adaptation of SW has not
been well-investigated. Using GSW directly with one type defining function for all marginals cannot
separate the information within and among groups of arguments.

Contribution: In this work, we tackle the challenge of designing a sliced Wasserstein variant for
heterogeneous joint distributions. In summary, our main contributions are three-fold:

1. We first extend the partial Radon Transform to the partial generalized Radon Transform
(PGRT) to inject non-linearity into local transformation. We discuss the injectivity of
PGRT for some choices of defining functions. We then propose a novel slicing operator for
heterogeneous joint distributions, named Hierarchical Hybrid Radon Transform (HHRT). In
particular, HHRT is a hierarchical transformation that first applies partial generalized Radon
Transform with different defining functions on arguments of each marginal to gather marginal
information, then applies partial Radon Transform on the joint transformed arguments to
gather information among marginals. We show that HHRT is injective as long as the partial
generalized Radon Transform is injective.

2. We propose Hierarchical Hybrid Sliced Wasserstein (H2SW) which is a novel metric for
comparing heterogeneous joint distributions by utilizing the HHRT. Moreover, we investigate
the topological properties, statistical properties, and computational properties of H2SW.
In particular, we show that H2SW is a valid metric on the space of distribution over the
joint space, H2SW does not suffer from the curse of dimensionality and enjoys the same
computational scalability as SW distance.

3. A 3D mesh can be effectively represented by a point-cloud and corresponding surface
normal vectors. Therefore, it can be seen as an empirical heterogeneous joint distribution.
We conduct experiments on optimization-based 3D mesh deformation and deep 3D mesh
autoencoder to show the favorable performance of H2SW compared to SW and GSW.
Moreover, we also illustrate that H2SW can also provide a meaningful comparison for
probability distributions on the product of Hadamard manifolds by conducting experiments
on dataset comparison.

2


---Page Break---
Organization. We first provide some preliminaries on SW distance, GSW distance, and joint
Wasserstein distance in Section 2. We then define the hierarchical hybrid Radon transform and
hierarchical hybrid sliced Wasserstein distance s in Section 3. Section 4 contains experiments on 3D
mesh deformation, deep 3D mesh autoencoder, and datasets comparison. We conclude the paper in
Section 5. Finally, we defer the proofs of key results, and additional materials in the Appendices.

2
Preliminaries

Wasserstein distance. For p ≥1, the Wasserstein-p distance [55, 47] between two distributions
µ ∈P(X) and ν ∈P(Y), where X and Y are subsets of Rd and they share a ground metric
c : X × Y →R+, is defined as:

Wp
p(µ, ν; c) :=
inf
π∈Π(µ,ν)

Z

X×Y
c(x, y)pdπ(x, y),
(1)

where Π(µ, ν) :=
n
π ∈P(X × Y)}|
R

Y dπ(x, y) = µ(x),
R

X dπ(x, y) = ν(y)
o
. When µ and ν
are discrete with at most n supports, the time complexity and the space complexity of the Wasserstein
distance is O(n3 log n) and O(n2) in turn which are very expensive. Therefore, sliced Wasserstein is
proposed as an alternative solution. We first review the

Radon Transform [23] The Radon Transform R : L1(Rd) →L1
 
R × Sd−1
is defined as:

(Rf)(t, θ) =
Z

Rd f(x)δ(t −⟨x, θ⟩)dx.
(2)

Radon Transform defines a linear bijection [23]. Given a projecting direction θ, (Rf)(·, θ) is an
one-dimensional function. With Radon Transform, we can now define the sliced Wasserstein distance.

Sliced Wasserstein distance.
For p ≥1, the Sliced Wasserstein (SW) distance [11] of p-th
order between two distributions µ ∈P(X) and ν ∈P(Y) with an one-dimensional ground metric
c : R × R →R+ is defined as follow:

SWp
p(µ, ν; c) = Eθ∼U(Sd−1)[Wp
p(Rθ♯µ, Rθ♯ν; c)],
(3)

where Rθ♯µ and Rθ♯ν are the one-dimensional push-forward distributions created by applying
Radon Transform (RT) [23] on the pdf of µ and ν with the projecting direction θ. The computational
benefit of SW distance comes from the closed-form solution when the one-dimensional ground metric
c(x, y) = h(x −y) for h is a strictly convex function:

Wp
p(Rθ♯µ, Rθ♯ν; c) =
Z 1

0
c

F −1
Rθ♯µ(z), F −1
Rθ♯ν(z)
p
dz,

where F −1
Rθ♯µ and F −1
Rθ♯ν are inverse CDF of Rθ♯µ and Rθ♯ν respectively. When µ and ν are
discrete with at most n supports, the time complexity and the space complexity of the closed-form is
O(n log n) and O(n) respectively.

Generalized Radon Transform and Generalized Sliced Wasserstein distance. To generalize RT
to non-linear operator, the Generalized Radon Transform (GRT) was proposed [3]. Given a defining
function [26] g : Rd × Ω→R, the Generalized Radon Transform [3] GR : L1(Rd) →L1 (R × Ω)
is defined as:

(GRf)(t, θ) =
Z

Rd f(x)δ(t −g(x, θ))dx.

For example, we can have the circular function [28], i.e., g(x, θ) = ∥x −rθ∥2 for r ∈R+ and θ ∈
Ω:= Sd−1, homogeneous polynomials with an odd degree [51] (m), i.e., g(x, θ) = P

|α|=m θαxα

with α = (α1, . . . , αdα) ∈Ndα, |α| = Pdα
i=1 αi, xα = Qdα
i=1 xαi
i , Ω= Sdα, and so on. Using GRT,
the Generalized Sliced Wasserstein (GSW) distance is introduced in [26], which is formally defined
as follow :

GSWp
p(µ, ν; c, g) = Eθ∼U(Sd−1)[Wp
p(GRg
θ♯µ, GRg
θ♯ν; c)].
(4)

It is worth noting that the injectivity of GRT is required to have the identity of indiscernible GSW.

3


---Page Break---
Heterogeneous joint distributions comparison. We are given two joint distributions µ(x1, x2) ∈
P(X1×X2) and ν(y1, y2) ∈P(Y1×Y2) where X1 are Y1 share a ground metric c1 : X1×Y1 →R+
and X2 are Y2 share a ground metric c2 : X2 × Y2 →R+ with (c1 ̸= c2). In this case, previous
works utilize the joint distribution Wasserstein distance [15, 1] to compare µ and ν:

Wp
p(µ, ν; c1, c2) :=
inf
π∈Π(µ,ν)

Z

X1×X2×Y1×Y2
(c1(x1, y1)p + c2(x2, y2)p)dπ(x1, x2, y1, y2),
(5)

where Π(µ, ν) :=
n
π ∈P(X1 × X2 × Y1 × Y2)}|
R

Y1×Y2 dπ(x1, x2, y1, y2) = µ(x1, x2),
R

X1×X2 dπ(x1, x2, y1, y2) = ν(y1, y2)
o
. We can easily extend the definition to joint distributions
with more than two marginals (see Appendix B). In contrast to the Wasserstein distance, there is no
variant of SW that is designed specifically for this case. SW variants can still be used by treating
X1×X2 and Y1×Y2 as homogeneous spaces X and Y which share the same Radon Transform variant
and one-dimensional ground metric c. However, that approach cannot differentiate the difference
between X1 and X2, and leverage the hierarchical structure, i.e., inside and among marginals.

3
Hierarchical Hybrid Sliced Wasserstein Distance

In this section, we propose the Hierarchical Hybrid Radon Transform (HHRT) which first applies
P(G)RT on each marginal argument to gather each marginal information, then applies PRT on the
joint transformed arguments from all marginals to gather information among marginals. After that,
we introduce Hierarchical Hybrid Sliced Wasserstein distance by using HHRT as the slicing operator.

3.1
Hierarchical Hybrid Radon Transform

We first introduce the first building block in HHRT, i.e., Partial Generalized Radon Transform (PGRT).
Definition 1 (Partial Generalized Radon Transform). Given a defining function g : Rd1 × Ω→R,
Partial Generalized Radon Transform PGR : L1(Rd1 × Rd2) →L1
 
R × Ω× Rd2
is defined as:

(PGRf)(t, θ, y) =
Z

Rd1
f(x, y)δ(t −g(x, θ))dx.
(6)

When g(x, θ) = ⟨x, θ⟩, PGRT reverts into Partial Radon Transform (PRT) [33].

Proposition 1. For some defining function g such as linear, circular, and homogeneous polynomials
with an odd degree; the Partial Generalized Radon Transform is injective, i.e., for any functions
f1, f2 ∈L1(Rd), (PGRf1)(t, θ, y) = (PGRf2)(t, θ, y) ∀t, θ, y implies f1 = f2.

The proof of Proposition 1 is given in Appendix A.1. The main idea to prove the injectivity of PGRT
is to show that given a fixed y, the PGRT is the GRT of f(·, y).

Definition 2 (Hierarchical Hybrid Radon Transform). Given defining functions g1 : Rd1 × Ω1 →R
and g2 : Rd2 × Ω2 →R, Hierarchical Hybrid Radon Transform HHR : L1(Rd1 × Rd2) →
L1 (R × Ω1 × Ω2 × S) is defined as:

(HHRf)(t, θ1, θ2, ψ) =
Z

Rd1×Rd2
f(x1, x2)δ (t −ψ1g1(x1, θ1) −ψ2g2(x2, θ2)) dx1dx2,
(7)

where ψ = (ψ1, ψ2) ∈S.

The reason for using PRT for the final transform is that the previous PGRTs are assumed to be able to
transform the non-linear structure to a linear line. However, PGRT can still be used as a replacement
for PRT. Definition 2 can be extended to more than two marginals (see Appendix B).

Proposition 2. For some defining functions g1, g2 such as linear, circular, and homogeneous polyno-
mials with an odd degree; Hierarchical Hybrid Radon Transform is injective, i.e., for any functions
f1, f2 ∈L1(Rd), (HHRf1)(t, θ1, θ2, ψ) = (HHRf2)(t, θ1, θ2, ψ) ∀t, θ1, θ2, ψ implies f1 = f2.

The proof of Proposition 2 is given in Appendix A.2. The main idea to prove the injectivity of HHRT
is to show that HHRT is the composition of PRT and multiple PGRTs.

4


---Page Break---
Figure 1:
Generalized Radon Transform and Hierarchical Hybrid Radon Transform on a discrete distribution.

HHRT
of
discrete
distributions.
We
are
given
f(x)
=
Pn
i=1 αiδ((x1, x2) −
(x1i, x2i)) (n
≥
1, αi
≥
0 ∀i).
The HHRT of f(x) is (HHRf)(t, θ1, θ2, ψ)
=
Pn
i=1 αiδ (t −ψ1g1(xi1, θ1) −ψ2g2(xi2, θ2)). For g1 and g2 that are the linear function and (or)
the circular function, the time complexity of the transform is O(d1 + d2) which is the same as the
complexity of using RT and GRT directly. However, HHRT has an additional constant complexity
scaling linearly with the number of marginals, i.e., two marginals in Definition 2.
Example 1. In this paper, we focus on 3D shape data (mesh) with points and normals representation,
i.e., shapes as points representation [46]. In particular, we can transform a 3D shape into a set of
points and normals by sampling from the surface of the mesh. In addition, we can convert back to the
3D shape from points and normals with Poisson surface reconstruction [25] algorithm. In this setup, a
shape is represented by a 6-dimensional vector x = (x1, x2) where x1 ∈X1 ∈R3 and x2 ∈X2 ∈S2.
For the set X1 ∈R3, we can use directly the linear defining function g1(x1, θ1) = ⟨x1, θ1⟩with
θ1 ∈S2. For the set X2 ∈S2, we can utilize the circular defining function g2(x2, θ2) = ∥x2 −rθ2∥2
with r ∈R+ and θ2 ∈S2. As alternative options for X2, we can use other defining functions
from special cases of GRT including Vertical Slice Transform [49], Parallel Slice Transform [50],
Spherical Radon Transform [6], and Stereographic Spherical Radon Transform [54].

Inversion. In Proposition 2, we show that HHRT is the composition of PRT and multiple PGRTs.
Therefore, the inversion of HHRT is the composition of the inversion of multiple PGRT (invertibility
of PGRT depends on the choice of defining functions [3, 28]) and the inversion of PRT [23].

3.2
Hierarchical Hybrid Sliced Wasserstein Distance

By using HHRT, we obtain a novel variant of SW which is specifically designed for comparing
heterogeneous joint distributions.

Definitions. We now define the Hierarchical Hybrid Sliced Wasserstein (H2SW) distance.
Definition 3. For p ≥1, defining functions g1, g2, the hierarchical hybrid sliced Wasserstein-p
(H2SW) distance between two distributions µ ∈P(X1 × X2) and ν ∈P(Y1 × Y2) with an one-
dimensional ground metric c : R × R →R+ is defined as:
H2SWp
p(µ, ν; c, g1, g2) = E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)],
(8)

where HHRθ1,θ2,ψ♯µ and HHRθ1,θ2,ψ♯ν are the one-dimensional push-forward distributions cre-
ated by applying HHRT.

Definition 3 can be easily extended to more than two marginals (see Appendix B)

Topological Properties. We first show that H2SW is a valid metric on the space of distributions on
any sets X × Y ∈Rd1 × Rd2 (d1, d2 ≥1).
Theorem 1. For any p ≥1, ground metric c, and defining functions g1, g2 which lead to the injectivity
of GRT, the hierarchical hybrid sliced Wasserstein H2SWp(·, ·; c, g1, g2) is a metric on P(Rd1 × Rd2)
i.e., it satisfies the symmetry, non-negativity, triangle inequality, and identity of indiscernible.

5


---Page Break---
The proof of Theorem 1 is given in Appendix A.3. It is worth noting that the identity of indiscernible
property is proved by the injectivity of HHRT (Proposition 2). We now discuss the connection of
H2SW to GSW and Wasserstein distance in some specific cases.

Proposition 3. For any p ≥1, c(x, y) = |x −y|, and µ, ν ∈P(Rd1 × Rd2), we have:
(i) H2SWp(µ, ν; c, g1, g2) ≤GSWp(µ1, ν1; c, g1)+GSWp(µ2, ν2; c, g2), where µ1(X) = µ(X×Rd2)
and µ2(Y ) = µ(Rd1 × Y ) (similar with ν1 and ν2).

(ii) If g1, g2 are linear defining functions, H2SWp(µ, ν; c, g1, g2) ≤Wp(µ1, ν1; c) + Wp(µ2, ν2; c).

(iii) If p = 1, g1, g2 are linear defining functions, H2SW1(µ, ν; c, g1, g2) ≤W1(µ, ν; c).

The proof of Proposition 3 is given in Appendix A.4.

Sample Complexity. We now discuss the sample complexity of H2SW.
Proposition 4. For any p ≥1, dimension d1, d2 ≥1, q > p, c(x, y) = |x −y|, g1, g2 are linear
defining functions or circular defining functions, and µ, ν ∈Pq(Rd1 × Rd2) with the corresponding
empirical distributions µn and νn (n ≥1), there exists a constant Cp,q depending on p, q such that:

E |H2SWp(µn, νn; c, g1, g2) −H2SWp(µ, ν; c, g1, g2)|

≤C

1
p
p,q

 q
X

i=0
qiCq−i
g1,g2(Mi(µ) + Mi(ν))

! 1

p







n−1/2p if q > 2p,
n−1/2p log(n)
1
p if q = 2p,
n−(q−p)/pq if q ∈(p, 2p),
(9)

where Mq(µ) and Mq(ν) are the q-th moments of µ and ν, Cg1,g2 is a constant depends on g1, g2.

The proof of Proposition 4 is given in Appendix A.5. The rate in Proposition 4 is as good as the rate
of SW in [38], however, it is slightly worse than the rate of SW in [44, 36, 5] due to the usage of
the circular defining functions and simpler assumptions. To the best of our knowledge, the sample
complexity of GSW has not been investigated.

Monte Carlo Estimation. Since the expectation in H2SW (Equation 8) is intractable, Monte
Carlo estimation and Quasi-Monte Carlo approximation [39] can be used to form a practical eval-
uation of H2SW. Here, we utilize Monte Carlo estimation for simplicity. In particular, we sample
θ11, . . . , θ1L
i.i.d
∼U(Ω1), θ21, . . . , θ2L
i.i.d
∼U(Ω2), and ψ1, . . . , ψL
i.i.d
∼U(S). After that, we form
the following estimation of H2SW:

\
H2SW
p
p(µ, ν; c, g1, g2, L) = 1

L

L
X

l=1
Wp
p(HHRg1,g2
θ1l,θ2l,ψl♯µ, HHRg1,g2
θ1l,θ2l,ψl♯ν; c).
(10)

Proposition 5. For any p ≥1, dimension d1, d2 ≥1, and µ, ν ∈P(Rd
1 × Rd2), we have:

E|\
H2SW
p
p(µ, ν; c, g1, g2, L) −H2SWp
p(µ, ν; c, g1, g2)|

≤
1
√

L
V ar
h
Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)
i 1

2 ,
(11)

where the variance is with respect to U(Ω1 × Ω2 × S).

The proof of Proposition 5 is given in Appendix A.6. From the proposition, we see that the estimation
error of H2SW is the same as SW which is O(L−1/2).

Computational Complexities. The time complexity and memory complexity of H2SW with linear
and circular defining functions are O(Ln log n + L(d1 + d2 + k)n) and O(Ln + (d1 + d2 + k)n)
with k is the number of marginals i.e., 2. We can see that the complexities of H2SW are the same as
those of SW in terms of the number of supports n and the number of dimensions d. We demonstrate
the process of HHRT compared to GRT on a discrete distribution with L realization of θ1, θ2, ψ
in Figure 1. Overall, the complexities of defining functions are often different in the number of
dimensions, hence, H2SW is always scaled the same as SW in the number of supports i.e., O(n log n).

Gradient
Estimation.
In
applications,
it
is
desirable
to
estimate
the
gradient
∇ϕH2SWp
p(µϕ, ν; c, g1, g2).
We can move the gradient operator to inside the expectation

6


---Page Break---
Table 1: Summary of joint Wasserstein distances across time steps from deformation from the sphere mesh to
the Armadillo mesh.

Distances
Step 100 (Wc1,c2 ↓)
Step 300 (Wc1,c2 ↓)
Step 500 (Wc1,c2 ↓)
Step 1500 (Wc1,c2 ↓)
Step 4000 (Wc1,c2 ↓)
Step 5000 (Wc1,c2 ↓)

SW L=10
1852.519±3.236
1436.686±3.056
1071.227±2.449
104.452±2.35
6.19±0.307
2.726±0.305
GSW L=10
1893.438±3.205
1535.737±3.363
1192.52±3.274
143.518±1.04
8.73±0.353
4.743±0.134
H2SW L=10
1840.73±1.282
1422.667±7.813
1058.171±5.362
95.672±4.376
6.326±0.151
2.602±0.201

SW L=100
1847.572±0.303
1426.425±0.528
1059.127±1.106
89.693±0.793
4.453±0.22
1.171±0.056
GSW L=100
1889.312±0.883
1525.269±1.078
1179.1±2.052
122.618±1.175
7.905±0.373
3.226±0.388
H2SW L=100
1839.347±1.986
1417.1±3.677
1048.895±4.008
86.078±0.623
4.61±0.431
1.086±0.177
Table 2: Summary of joint Wasserstein distances (multiplied by 100) across time steps from deformation from
the sphere mesh to the Stanford Bunny mesh.

Distances
Step 100 (Wc1,c2 ↓)
Step 300 (Wc1,c2 ↓)
Step 500 (Wc1,c2 ↓)
Step 1500 (Wc1,c2 ↓)
Step 4000 (Wc1,c2 ↓)
Step 5000 (Wc1,c2 ↓)

SW L=10
26.868±0.579
4.46±0.195
1.52±0.081
0.623±0.024
0.221±0.023
0.14±0.018
GSW L=10
26.837±0.496
4.378±0.128
1.548±0.062
0.653±0.01
0.173±0.018
0.146±0.013
H2SW L=10
23.283±0.119
2.221±0.124
1.452±0.075
0.636±0.045
0.177±0.009
0.089±0.022

SW L=100
26.678±0.168
4.109±0.138
1.458±0.142
0.362±0.023
0.072±0.017
0.049±0.006
GSW L=100
26.795±0.202
4.084±0.109
1.375±0.049
0.372±0.026
0.048±0.004
0.042±0.017
H2SW L=100
23.772±0.19
2.388±0.009
1.358±0.051
0.488±0.026
0.064±0.01
0.038±0.007

and then apply Monte Carlo estimation. The gradient ∇ϕWp
p(HHRg1,g2
θ1,θ2,ψ♯µϕ, HHRg1,g2
θ1,θ2,ψ♯ν; c)]
can be computed easily since the functions g1, g2 are usually differentiable.

Beyond uniform slicing distribution. H2SW is defined with the uniform slicing distribution in
Definition 3, however, it is possible to extend it to other slicing distributions such as the maximal
projecting direction [18], distributional slicing distribution [42], and energy-based slicing distribu-
tion [41]. Since the choice of slicing distribution is independent of the main contribution i.e., the
slicing operator, we leave this investigation to future work.

H2SW for distributions on the product of Hadamard manifolds. A recent work [9] extends
sliced Wasserstein on hyperbolic manifolds [7] and on the manifold of symmetric positive definite
matrices [10] to Hadamard manifolds i.e., manifold non-positive curvature. The work discusses the
extension of SW to the product of Hadamard manifolds. For the geodesic projection, the closed-form
for the projection is intractable. For the Busemann projection, the Busemann projection on the
product manifolds is the weighted sum of the Busemann projection with the weights belonging
to the unit-sphere. In the work, the weights are a fixed hyperparameter i.e., Cartan-Hadamard
Sliced-Wasserstein (CHSW) utilizes only one Busemann function to project the joint distribution.
In contrast, H2SW utilizes the Radon Transform on the joint spaces of projections i.e., considering
all distributed weighted combinations which is equivalent to considering all Busemann functions
under a probability law. As a result, the H2SW is a valid metric as long as the Busemann projections
can be proven to be injective (the injectivity of the Busemann projection has not been known at the
moment) while Cartan-Hadamard Sliced-Wasserstein is only pseudo metric since the injectivity of
a fixed weighted combination is not trivial to show. Moreover, H2SW does not only focus on the
product of Hadamard manifolds i.e., H2SW is a generic distance for heterogeneous joint distributions
in which marginal domains are not necessary manifolds e.g., images [40], functions [21], and so on.
In the later experiments, we conduct experiments on comparing 3D shapes which are represented by
a distribution on the product of the Euclidean space and the 2D sphere (not a Hadamard manifold).

4
Experiments

In this section, we first compare the performance of the proposed H2SW with SW and GSW in the 3D
mesh deformation application. After that, we further evaluate the performance of H2SW in training a
deep 3D mesh autoencoder compared to SW and GSW. Finally, we compare H2SW with SW and
Cartan-Hadamard Sliced-Wasserstein (CHSW) in datasets comparison on the product of Hadamard
manifolds. In the experiments, we use c(x, y) = |x −y| and p = 2 for all SW variants.

4.1
3D Mesh Deformation

In this task, we would like to move from a source mesh to a target mesh. To represent those meshes,
we sample 10000 points by Poisson disk sampling and their corresponding normal vectors of the mesh
surface at those points. Let the source mesh be denoted as X(0) = {x1(0), . . . , xn(0)} and the target
mesh be denoted as Y = {y1, . . . , yn}. We deform X(0) to Y by integrating the ordinary differential
equation ˙X(t) = −n∇X(t)

S
  1

n
Pn
i=1 δ(x −xi(t)), 1

n
Pn
i=1 δ(y −yi)

, where S denotes a SW

7


---Page Break---
Figure 2: Visualization of deformation from the sphere mesh to the Armadillo mesh with L = 10.

Figure 3: Visualization of deformation from the sphere mesh to the Stanford Bunny mesh with L = 10.

variant. We utilize the Euler discretization scheme with step size 0.01 and 5000 steps. The normal
vectors are projected back to the sphere after taking an Euler step. For evaluation, we use the joint
Wasserstein distance in Equation 5 with the mixed distance from the Euclidean distance and the great
circle distance. We use the circular defining function for GSW, and use the linear defining function
and the circular defining function for H2SW. We vary the number of projections L ∈{10, 100} for
all variants. For H2SW and GSW, we select the best hyperparameter of the circular defining function
r ∈{0.5, 0.7, 0.8, 0.9, 1, 5, 10, 50, 100}.

Results. We compare H2SW with GSW and SW by deforming the sphere mesh to the Armadillo
mesh [59]. We report the quantitative results in Table 1 after 3 independent runs and the qualitative
result for L = 10 in Figure 2 and L = 100 in Figure 6 in Appendix D. From Table 1, we observe that
H2SW helps the deformation convergence faster at the beginning and better at the end in terms of
the joint Wasserstein distance, especially for a small value of the number of projections i.e., L = 10.
The result for L = 100 is better than L = 10 which is consistent with Proposition 5. The qualitative
results in Figure 2 and Figure 6 also reinforce the favorable performance of H2SW since they are
visually consistent with quantitative scores. We also conduct deformation to the Stanford Bunny
mesh [16, 59] in Table 2, Figure 3, and Figure 7 in Appendix D and we observe the same phenomenon
that H2SW is the best variant for 3D meshes. From those experiments, H2SW has shown the benefit
of the HHRT in transforming a joint distribution over the product of the Euclidean space and the 2D
sphere compared to the conventional RT of SW and the conventional GRT of GSW.

4.2
Training deep 3D mesh autoencoder

We utilize the processed ShapeNet dataset [12] from [46], then sample 2048 points and the cor-
responding normal vectors from each shape in the dataset. Formally, we would like to train an
autoencoder that contains an encoder fϕ that maps a mesh X ∈R2048×6 to a latent code z ∈R1024,
and a decoder gψ that maps the latent code z back to the reconstructed mesh ˜X ∈R2048×6. We adopt

8


---Page Break---
Table 3: Joint Wasserstein distance reconstruction errors (multiplied by 100) from three different runs of
autoencoders trained by SW, GSW, and H2SW with the number of projections L = 100 and L = 1000.

Distance
Epoch 500 Wc1,c2(↓)
Epoch 1000 Wc1,c2(↓)
Epoch 2000 Wc1,c2(↓)

SW L=100
136.87 ± 1.18
133.24 ± 0.50
131.10 ± 0.34
GSW L=100
136.51 ± 0.20
133.36 ± 0.24
130.80 ± 0.46
H2SW L=100
135.54 ± 0.72
132.24 ± 0.36
130.24 ± 0.47

SW L=1000
135.85 ± 0.92
132.88 ± 0.27
130.93 ± 0.11
GSW L=1000
136.40 ± 0.10
133.02 ± 0.98
130.76 ± 0.26
H2SW L=1000
135.47 ± 0.64
132.17 ± 0.10
129.87 ± 0.44

Figure 4: Visualization of some randomly selected reconstruction meshes from autoencoders trained by SW,
GSW, and H2SW in turn with the number of projections L = 100 at epoch 2000.

Point-Net [48] architecture to construct the autoencoder. We want to train the encoder fϕ and the
decoder gψ such that ˜X = gψ(fϕ(X)) ≈X for all shapes X in the dataset. To do that, we solve the
following optimization problem:

min
ϕ,γ EX∼µ(X)[S(PX, Pgγ(fϕ(X)))],

where S is a sliced Wasserstein variant, and PX = 1

n
Pn
i=1 δ(x −xi) denotes the empirical distribu-
tion over the point cloud X = (x1, . . . , xn). We train the autoencoder for 2000 epochs on the training
set of the ShapeNet dataset using an SGD optimizer with a learning rate of 1e −3, and a batch size of
128. For evaluation, we also use the joint Wasserstein distance in Equation 5 with the mixed distance
from the Euclidean distance and the great circle distance to measure the average reconstruction loss
on the testing set of the ShapeNet dataset. We use the circular defining function for GSW, and use the
linear defining function and the circular defining function for H2SW. For H2SW and GSW, we select
the best hyperparameter of the circular defining function r ∈{0.5, 0.7, 0.8, 0.9, 1, 5, 10}. For more
details such as the neural network architectures, we refer the reader to Appendix B.

Results. We report the joint Wasserstein reconstruction errors (measured in three independent times)
on the testing set in Table 3 with trained autoencoder at epoch 500, 1000, and 2000 from SW,
GSW, and H2SW with the number of projections L = 100 and L = 1000. In addition, we show
some randomly reconstructed meshes for epoch 2000in Figure 4 and for epoch 500 in Figure 8
in Appendix D. From Table 3, we observe that H2SW yields the lowest reconstruction errors for
both L = 100 and L = 1000. Moreover, we see that the reconstruction errors are lower with
L = 1000 than ones with L = 100 for all SW variants. The qualitative reconstructed meshes in
Figure 4 and Figure 8reflect the same relative comparison. It is worth noting that both the qualitative
and the qualitative performance of autoencoders can be improved by using more powerful neural
networks. Since we focus on comparing SW, GSW, and H2SW, we only use a light neural network
i.e., Point-Net [48] architecture. The trained autoencoders can be further used to reduce the size of 3D
meshes for data compression and for dimension reduction, however, such downstream applications
are not our focus in the current investigation of the paper.

9


---Page Break---
Table 4: Relative error to the joint Wasserstein distance of SW, CHSW, and H2SW.

Distances
L = 100
L = 500
L = 1000
L = 2000

SW
4.618 ± 0.744
4.253 ± 0.398
4.235 ± 0.310
4.198 ± 0.238
CHSW
4.449 ± 0.497
4.063 ± 0.254
4.059 ± 0.167
4.035 ± 0.145
H2SW
4.381 ± 0.695
4.001 ± 0.267
4.048 ± 0.182
3.998 ± 0.142

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.28±0.009
0.461±0.014
0.241±0.006
0.39±0.012

0.28±0.009
0.451±0.013
0.225±0.007
0.486±0.013

0.461±0.014
0.451±0.013
0.307±0.007
0.222±0.005

0.241±0.006
0.225±0.007
0.307±0.007
0.245±0.005

0.39±0.012
0.486±0.013
0.222±0.005
0.245±0.005

0.0

0.1

0.2

0.3

0.4

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.199±0.005
0.374±0.011
0.187±0.005
0.281±0.009

0.199±0.005
0.336±0.01
0.17±0.004
0.333±0.01

0.374±0.011
0.336±0.01
0.236±0.007
0.19±0.004

0.187±0.005
0.17±0.004
0.236±0.007
0.179±0.004

0.281±0.009
0.333±0.01
0.19±0.004
0.179±0.004

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.201±0.006
0.379±0.009
0.189±0.005
0.285±0.008

0.201±0.006
0.342±0.01
0.173±0.005
0.34±0.01

0.379±0.009
0.342±0.01
0.243±0.007
0.194±0.005

0.189±0.005
0.173±0.005
0.243±0.007
0.187±0.004

0.285±0.008
0.34±0.01
0.194±0.005
0.187±0.004

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

956.8
1273.27
1139.95
843.07

956.8
1278.35
1157.0
891.65

1273.27
1278.35
1253.58
607.44

1139.95
1157.0
1253.58
843.23

843.07
891.65
607.44
843.23

0

200

400

600

800

1000

1200

SW
CHSW
H2SW
Joint Wasserstein

Figure 5: Cost matrices between datasets from SW, CHSW, and H2SW with L = 2000.

4.3
Comparing Datasets on The Product of Hadamard Manifolds

We follow the same experimental setting from [9]. Here, we have datasets as sets of feature-label
pairs which are embedded in the space of Rd1 × Ld2 where Ld2 denotes a Lorentz model of d2
dimension (a hyperbolic space). We uses MNIST [31] dataset, EMNIST dataset [14], Fashion MNIST
dataset [56], KMNIST dataset [13], and USPS dataset [24]. For CHSW, we use Busemann projection
on the product space of Euclidean and the Lorentz model. For H2SW, we use the linear defining
function and the Busemann function on the Lorentz model. We refer the reader to Appendix B for
greater detail on Busemann functions and experimental setups. We compare SW, CHSW, and H2SW
by varying L ∈{100, 500, 1000, 2000}. For evaluation, we use the joint Wasserstein distance in [1]
as the ground truth. In particular, let CW be the cost matrix from the joint Wasserstein distance and
C be a given cost matrix, we use |C/ max(C) −CW / max(CW )| as the relative error.

Results. We report the relative errors from SW, CHSW, and H2SW in Table 4 after 100 independent
runs. In addition, we show the cost matrices from SW, CHSW, H2SW. and joint Wasserstein distance
with L = 2000 in Figure 5. Cost matrices for L = 100, L = 500, and L = 1000 are given in
Figure 9- 11 in Appendix D. From Table 4, we see that H2SW gives a lower relative error than
CHSW and SW. Therefore, using H2SW for comparing datasets is the most equivalent to the joint
Wasserstein distance in terms of the relative error. We also observe that increasing the value of the
number of projections also reduces the relative errors for all SW variants. Again, we would like to
recall that H2SW can be used for heterogeneous joint distributions beyond the product of Hadamard
manifolds as shown in previous experiments.

5
Conclusion

We have presented Hierarchical Hybrid Sliced Wasserstein (H2SW) distance, a novel sliced probability
metric for heterogeneous joint distributions i.e., joint distributions have marginals on different
domains. The key component of H2SW is the proposed hierarchical hybrid Radon Transform (HHRT)
which is the composition of partial Radon Transform and multiples proposed partial generalized
Radon Transform. We then discuss the injectivity of the proposed transforms and theoretical properties
of H2SW including topological properties, statistical properties, and computational properties. On
the experimental side, we show that H2SW has favorable performance in applications of 3D mesh
deformation, training deep 3D mesh autoencoder, and datasets comparison. In those applications,
heterogeneous joint distributions appear in the form of joint distributions on the product of Euclidean
space and 2D sphere, and the product of Hadamard manifolds. In the future, we will extend the
application of H2SW to more complicated heterogeneous joint distributions.

10


---Page Break---
Acknowledgements

NH acknowledges support from the NSF IFML 2019844 and the NSF AI Institute for Foundations of
Machine Learning.

References

[1] D. Alvarez-Melis and N. Fusi. Geometric dataset distances via optimal transport. Advances in
Neural Information Processing Systems, 33:21428–21439, 2020.

[2] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generative adversarial networks. In
International Conference on Machine Learning, pages 214–223, 2017.

[3] G. Beylkin. The inversion problem and applications of the generalized Radon transform.
Communications on pure and applied mathematics, 37(5):579–599, 1984.

[4] B. Bhushan Damodaran, B. Kellenberger, R. Flamary, D. Tuia, and N. Courty. Deepjdot: Deep
joint distribution optimal transport for unsupervised domain adaptation. In Proceedings of the
European Conference on Computer Vision (ECCV), pages 447–463, 2018.

[5] M. T. Boedihardjo. Sharp bounds for the max-sliced Wasserstein distance. arXiv preprint
arXiv:2403.00666, 2024.

[6] C. Bonet, P. Berg, N. Courty, F. Septier, L. Drumetz, and M.-T. Pham. Spherical sliced-
Wasserstein. International Conference on Learning Representations, 2023.

[7] C. Bonet, L. Chapel, L. Drumetz, and N. Courty. Hyperbolic sliced-Wasserstein via geodesic
and horospherical projections. In Topological, Algebraic and Geometric Learning Workshops
2023, pages 334–370. PMLR, 2023.

[8] C. Bonet, N. Courty, F. Septier, and L. Drumetz. Efficient gradient flows in sliced-Wasserstein
space. Transactions on Machine Learning Research, 2022.

[9] C. Bonet, L. Drumetz, and N. Courty. Sliced-Wasserstein distances and flows on Cartan-
Hadamard manifolds. arXiv preprint arXiv:2403.06560, 2024.

[10] C. Bonet, B. Malézieux, A. Rakotomamonjy, L. Drumetz, T. Moreau, M. Kowalski, and
N. Courty. Sliced-Wasserstein on symmetric positive definite matrices for m/eeg signals. In
International Conference on Machine Learning, pages 2777–2805. PMLR, 2023.

[11] N. Bonneel, J. Rabin, G. Peyré, and H. Pfister. Sliced and Radon Wasserstein barycenters of
measures. Journal of Mathematical Imaging and Vision, 1(51):22–45, 2015.

[12] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva,
S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint
arXiv:1512.03012, 2015.

[13] T. Clanuwat, M. Bober-Irizar, A. Kitamoto, A. Lamb, K. Yamamoto, and D. Ha. Deep learning
for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018.

[14] G. Cohen, S. Afshar, J. Tapson, and A. Van Schaik. Emnist: Extending mnist to handwritten
letters. In 2017 international joint conference on neural networks (IJCNN), pages 2921–2926.
IEEE, 2017.

[15] N. Courty, R. Flamary, A. Habrard, and A. Rakotomamonjy. Joint distribution optimal trans-
portation for domain adaptation. In Advances in Neural Information Processing Systems, pages
3730–3739, 2017.

[16] B. Curless and M. Levoy. A volumetric method for building complex models from range images.
In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques,
pages 303–312, 1996.

[17] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in
Neural Information Processing Systems, pages 2292–2300, 2013.

[18] I. Deshpande, Y.-T. Hu, R. Sun, A. Pyrros, N. Siddiqui, S. Koyejo, Z. Zhao, D. Forsyth, and
A. G. Schwing. Max-sliced Wasserstein distance and its use for GANs. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition, pages 10648–10656, 2019.

11


---Page Break---
[19] I. Deshpande, Z. Zhang, and A. G. Schwing. Generative modeling using the sliced Wasserstein
distance. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
pages 3483–3491, 2018.
[20] N. Fournier and A. Guillin. On the rate of convergence in Wasserstein distance of the empirical
measure. Probability Theory and Related Fields, 162:707–738, 2015.
[21] R. C. Garrett, T. Harris, B. Li, and Z. Wang. Validating climate models with spherical convolu-
tional Wasserstein distance. arXiv preprint arXiv:2401.14657, 2024.
[22] E. Heitz, K. Vanhoey, T. Chambon, and L. Belcour. A sliced Wasserstein loss for neural
texture synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9412–9420, 2021.
[23] S. Helgason. The Radon transform on r n. In Integral Geometry and Radon Transforms, pages
1–62. Springer, 2011.
[24] J. J. Hull. A database for handwritten text recognition research. IEEE Transactions on pattern
analysis and machine intelligence, 16(5):550–554, 1994.
[25] M. Kazhdan, M. Bolitho, and H. Hoppe. Poisson surface reconstruction. In Proceedings of the
fourth Eurographics symposium on Geometry processing, volume 7, page 0, 2006.
[26] S. Kolouri, K. Nadjahi, U. Simsekli, R. Badeau, and G. Rohde. Generalized sliced Wasserstein
distances. In Advances in Neural Information Processing Systems, pages 261–272, 2019.
[27] S. Kolouri, G. K. Rohde, and H. Hoffmann. Sliced Wasserstein distance for learning Gaussian
mixture models. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 3427–3436, 2018.
[28] P. Kuchment. Generalized transforms of Radon type and their applications. In Proceedings of
Symposia in Applied Mathematics, volume 63, page 67, 2006.
[29] T. Le, K. Nguyen, S. Sun, K. Han, N. Ho, and X. Xie. Diffeomorphic mesh deformation via
efficient optimal transport for cortical surface reconstruction. International Conference on
Learning Representations, 2024.
[30] T. Le, K. Nguyen, S. Sun, N. Ho, and X. Xie. Integrating efficient optimal transport and
functional maps for unsupervised shape correspondence learning. Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, 2024.
[31] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
[32] C.-Y. Lee, T. Batra, M. H. Baig, and D. Ulbricht. Sliced Wasserstein discrepancy for unsuper-
vised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10285–10295, 2019.
[33] Z.-P. Liang and D. C. Munson.
Partial Radon transforms.
IEEE transactions on image
processing, 6(10):1467–1469, 1997.
[34] A. Liutkus, U. Simsekli, S. Majewski, A. Durmus, and F.-R. Stöter. Sliced-Wasserstein flows:
Nonparametric generative modeling via optimal transport and diffusions. In International
Conference on Machine Learning, pages 4104–4113. PMLR, 2019.
[35] M. Luong, K. Nguyen, N. Ho, R. Haf, D. Phung, and L. Qu. Revisiting deep audio-text retrieval
through the lens of transportation. In The Twelfth International Conference on Learning
Representations, 2024.
[36] T. Manole, S. Balakrishnan, and L. Wasserman. Minimax confidence intervals for the sliced
Wasserstein distance. Electronic Journal of Statistics, 16(1):2252–2345, 2022.
[37] K. Nadjahi, V. De Bortoli, A. Durmus, R. Badeau, and U. ¸Sim¸sekli. Approximate Bayesian
computation with the sliced-Wasserstein distance. In ICASSP 2020-2020 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5470–5474. IEEE,
2020.
[38] K. Nadjahi, A. Durmus, L. Chizat, S. Kolouri, S. Shahrampour, and U. Simsekli. Statistical
and topological properties of sliced probability divergences. Advances in Neural Information
Processing Systems, 33:20802–20812, 2020.
[39] K. Nguyen, N. Bariletto, and N. Ho. Quasi-monte carlo for 3d sliced Wasserstein. In The
Twelfth International Conference on Learning Representations, 2024.

12


---Page Break---
[40] K. Nguyen and N. Ho.
Revisiting sliced Wasserstein on images: From vectorization to
convolution. Advances in Neural Information Processing Systems, 2022.
[41] K. Nguyen and N. Ho. Energy-based sliced Wasserstein distance. Advances in Neural Informa-
tion Processing Systems, 2023.
[42] K. Nguyen, N. Ho, T. Pham, and H. Bui. Distributional sliced-Wasserstein and applications to
generative modeling. In International Conference on Learning Representations, 2021.
[43] K. Nguyen, T. Ren, H. Nguyen, L. Rout, T. Nguyen, and N. Ho. Hierarchical sliced Wasserstein
distance. International Conference on Learning Representations, 2023.
[44] S. Nietert, R. Sadhu, Z. Goldfeld, and K. Kato. Statistical, robustness, and computational
guarantees for sliced Wasserstein distances. Advances in Neural Information Processing
Systems, 2022.
[45] O. Pele and M. Werman. Fast and robust earth mover’s distances. In 2009 IEEE 12th Interna-
tional Conference on Computer Vision, pages 460–467. IEEE, September 2009.
[46] S. Peng, C. Jiang, Y. Liao, M. Niemeyer, M. Pollefeys, and A. Geiger. Shape as points: A
differentiable poisson solver. Advances in Neural Information Processing Systems, 34:13032–
13044, 2021.
[47] G. Peyré and M. Cuturi. Computational optimal transport, 2020.
[48] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d
classification and segmentation. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 652–660, 2017.
[49] M. Quellmalz, R. Beinert, and G. Steidl. Sliced optimal transport on the sphere. Inverse
Problems, 39(10):105005, 2023.
[50] M. Quellmalz, L. Buecher, and G. Steidl. Parallelly sliced optimal transport on spheres and on
the rotation group. arXiv preprint arXiv:2401.16896, 2024.
[51] F. Rouviere. Nonlinear Radon and Fourier transforms, 2015.
[52] R. M. Rustamov and S. Majumdar. Intrinsic sliced Wasserstein distances for comparing
collections of probability distributions on manifolds and graphs. In International Conference
on Machine Learning, pages 29388–29415. PMLR, 2023.
[53] I. Tolstikhin, O. Bousquet, S. Gelly, and B. Schoelkopf. Wasserstein auto-encoders. In
International Conference on Learning Representations, 2018.
[54] H. Tran, Y. Bai, A. Kothapalli, A. Shahbazi, X. Liu, R. D. Martin, and S. Kolouri. Stereographic
spherical sliced Wasserstein distances. International Conference on Machine Learning, 2024.
[55] C. Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media,
2008.
[56] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking
machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.
[57] J. Xu, H. Zhou, C. Gan, Z. Zheng, and L. Li. Vocabulary learning via optimal transport for
neural machine translation. In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 7361–7373, 2021.
[58] M. Yi and S. Liu. Sliced Wasserstein variational inference. In Fourth Symposium on Advances
in Approximate Bayesian Inference, 2021.
[59] Q.-Y. Zhou, J. Park, and V. Koltun. Open3D: A modern library for 3D data processing.
arXiv:1801.09847, 2018.

13


---Page Break---
Supplement to “Hierarchical Hybrid Sliced Wasserstein: A
Scalable Metric for Heterogeneous Joint Distributions"

We first provide skipped proofs in the main paper in Appendix A. We then provide some additional
materials including additional background and extended definitions in Appendix B. After that, we
discuss some related works in Appendix C. We report additional experimental results in Appendix D.
Finally, we report computational infrastructure in Appendix E.

A
Proofs

A.1
Proof of Proposition 1

For any t, θ, y, we are given (PGRf1)(t, θ, y) = (PGRf2)(t, θ, y). By Definition 1, we have:
Z

Rd1
f1(x, y)δ(t −g(x, θ))dx =
Z

Rd1
f2(x, y)δ(t −g(x, θ))dx.

For any ε ∈Rd2, we have:
Z

Rd2

Z

Rd1
f1(x, y)δ(t −g(x, θ))e−i2π⟨ε,y⟩dxdy =
Z

Rd2

Z

Rd1
f2(x, y)δ(t −g(x, θ))e−i2π⟨ε,y⟩dxdy.

Applying the Fubini’s theorem, we have:
Z

Rd1
f1(x, y)
Z

Rd2
e−i2π⟨ε,y⟩dyδ(t −g(x, θ))dx =
Z

Rd1

Z

Rd2
f2(x, y)e−i2π⟨ε,y⟩dyδ(t −g(x, θ))dx,

which is:

GR
Z

Rd2
f1(x, y)e−i2π⟨ε,y⟩dy

=

GR
Z

Rd2
f2(x, y)e−i2π⟨ε,y⟩dy

.

By the injectivity of GRT, we have:
Z

Rd2
f1(x, y)e−i2π⟨ε,y⟩dy =
Z

Rd2
f2(x, y)e−i2π⟨ε,y⟩dy.

Then, for any ϵ ∈Rd1, we have
Z

Rd1

Z

Rd2
f1(x, y)e−i2π⟨ε,y⟩e−i2π⟨ϵ,x⟩dydx =
Z

Rd1

Z

Rd2
f2(x, y)e−i2π⟨ε,y⟩e−i2π⟨ϵ,x⟩dydx.

which is (Ff1(x, y)) = (Ff2(x, y))) with F denotes the Fourier transform. By the injectivity of the
Fourier Transform, we have f1(x, y) = f2(x, y) for any x, y, which concludes the proof.

A.2
Proof of Proposition 2

We first show that HHRT is the composition of PGRT and PRT. We have

(PR(PGR(PGRf)))(t, θ1, θ2, ψ)

=
Z

R2

Z

Rd1

Z

Rd2
f(x, y)δ(t1 −g1(x, θ1))δ(t2 −g2(y, θ2))δ(t −ψ1t1 −ψ2t2)dxdydt1dt2

=
Z

Rd1

Z

Rd2
f(x, y)
Z

R2 δ(t1 −g1(x, θ1))δ(t2 −g2(y, θ2))δ(t −ψ1t1 −ψ2t2)dt1dt2dxdy

=
Z

Rd1

Z

Rd2
f(x, y)δ (t −ψ1g1(x, θ1) −ψ2g2(y, θ2)) dxdy

= (HHRf)(t, θ1, θ2, ψ).

For any t, θ1, θ2, ψ, we are given (HHRf1)(t, θ1, θ2, ψ) = (HHRf2)(t, θ1, θ2, ψ), which is equiva-
lent to:

(PR(PGR(PGRf1)))(t, θ1, θ2, ψ) = (PR(PGR(PGRf2)))(t, θ1, θ2, ψ).

By the injectivity of the PRT and the PGRT, we obtain f1(x, y) = f2(x, y) for any x, y which
completes the proof.

14


---Page Break---
A.3
Proof of Theorem 1

To prove that the hierarchical hybrid sliced Wasserstein H2SWp(·, ·; c, g1, g2) is a metric on the
space of distributions on P(Rd1 × Rd2) for any p ≥1, ground metric c, and defining functions
g1, g2, we need to show that it satisfies non-negativity, symmetry, triangle inequality, and identity of
indiscernible.

Non-Negativity. Since Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c) ≥0 [47] for any θ1, θ2, ψ, we have:

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)] ≥0,

which means that H2SWp(µ, ν; c, g1, g2) ≥0 for any µ and ν.

Symmetry.
Since
we
have
the
symmetry
of
the
Wasserstein
distance
Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)
=
Wp
p(HHRg1,g2
θ1,θ2,ψ♯ν, HHRg1,g2
θ1,θ2,ψ♯µ; c)
[47]
for
any θ1, θ2, ψ, we have:

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)]

= E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯ν, HHRg1,g2
θ1,θ2,ψ♯µ; c)],

which means that H2SWp(µ, ν; c, g1, g2) = H2SWp(ν, µ; c, g1, g2) any µ and ν.

Triangle Inequality. Given c to be a valid metric on R, we can use the triangle inequality of the
Wasserstein distance. For any distributions µ1, µ2, µ3 ∈P(Rd1 × Rd2), we have:

H2SWp(µ1, µ2; c, g1, g2) =

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ1, HHRg1,g2
θ1,θ2,ψ♯µ2; c)]
 1

p

≤

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[(Wp(HHRg1,g2
θ1,θ2,ψ♯µ1, HHRg1,g2
θ1,θ2,ψ♯µ3; c)

+Wp(HHRg1,g2
θ1,θ2,ψ♯µ3, HHRg1,g2
θ1,θ2,ψ♯µ2; c))p]
 1

p

≤

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ1, HHRg1,g2
θ1,θ2,ψ♯µ3; c)]
 1

p

+

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ3, HHRg1,g2
θ1,θ2,ψ♯µ2; c)]
 1

p

= H2SWp(µ1, µ3; c, g1, g2) + H2SWp(µ3, µ2; c, g1, g2),

where the final inequality is due to Minkowski’s inequality. Therefore, we complete the proof for the
triangle inequality of the hierarchical hybrid sliced Wasserstein.

Identity of indiscernible.
For any p
≥
1,
ground metric c,
and g1, g2,
when
µ
=
ν,
we
have
HHRg1,g2
θ1,θ2,ψ♯µ
=
(HHRg1,g2
θ1,θ2,ψ♯ν.
Therefore,
we
have
Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ1, HHRg1,g2
θ1,θ2,ψ♯µ2; c)
=
0 which leads to H2SWp(µ, ν; c, g1, g2)
=
0.
Now, assume that H2SWp(µ, ν; c, g1, g2) = 0, then Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ1, HHRg1,g2
θ1,θ2,ψ♯µ2; c) = 0
for almost everywhere θ1 ∈Ω1, θ2 ∈Ω2, ψ ∈S.
By applying the identity property of
the Wasserstein distance, we have HHRg1,g2
θ1,θ2,ψ♯µ = (HHRg1,g2
θ1,θ2,ψ♯ν for almost everywhere
θ1 ∈Ω1, θ2 ∈Ω2, ψ ∈S. Since the HHRT is injective (proved in Proposition 2), we obtain µ = ν.

A.4
Proof of Proposition 3

(i) For any p ≥1, c(x, y) = |x −y|, and µ, ν ∈P(Rd1 × Rd2), we have:

H2SWp(µ, ν; c, g1, g2)

=

E(θ1,θ2,ψ)∼U(Ω1×Ω2×S)[Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)]
 1

p

=

E

inf
π∈Π(µ,ν)

Z
|ψ1(g1(θ1, x1) −g1(θ1, y1)) + ψ2(g2(θ2, x2) −g1(θ2, y2))|pdπ(x1, x2, y1, y2)
 1

p

15


---Page Break---
By applying the Cauchy-Schwartz inequality, we have:
H2SWp(µ, ν; c, g1, g2)

≤

E

inf
π∈Π(µ,ν)

Z
(
q

ψ2
1 + ψ2
2)p(
p

(g1(θ1, x1) −g1(θ1, y1))2 + (g2(θ2, x2) −g2(θ2, y2))2)pdπ(x1, x2, y1, y2)
 1

p

≤

E

inf
π∈Π(µ,ν)

Z
(|g1(θ1, x1) −g1(θ1, y1)| + |g2(θ2, x2) −g2(θ2, y2)|)pdπ(x1, x2, y1, y2)
 1

p

≤

E

inf
π∈Π(µ,ν)

Z
|g1(θ1, x1) −g1(θ1, y1)|pdπ(x1, x2, y1, y2)
 1

p

+

E

inf
π∈Π(µ,ν)

Z
|g2(θ2, x2) −g2(θ2, y2)|pdπ(x1, x2, y1, y2)
 1

p

=

E

inf
π∈Π(µ1,ν1)

Z
|g1(θ1, x1) −g1(θ1, y1)|pdπ(x1, y1)
 1

p

+

E

inf
π∈Π(µ2,ν2)

Z
|g2(θ2, x2) −g2(θ2, y2)|pdπ(x2, y2)
 1

p

= GSWp(µ1, ν1; g1, c) + GSWp(µ2, ν2; g2, c),
where the last inequality is due to the Minkowski’s inequality.

(ii) From (i), we have H2SWp(µ, ν; c, g1, g2) ≤GSWp(µ1, ν1; g1, c) + GSWp(µ2, ν2; g2, c). When,
g1, g2, and c(x, y) = |x −y| are linear defining functions, we have:

GSWp(µ1, ν1; g1, c) =

E

inf
π∈Π(µ1,ν1)

Z
(|θ⊤x1 −θ⊤y1|pdπ(x1, y1)
 1

p

≤

E

inf
π∈Π(µ1,ν1)

Z
(∥θ∥2∥x1 −y1∥p
2dπ(x1, y1)
 1

p

≤

E

inf
π∈Π(µ1,ν1)

Z
(∥x1 −y1∥pdπ(x1, y1)
 1

p

=

inf
π∈Π(µ1,ν1)

Z
(∥x1 −y1∥pdπ(x1, y1)
 1

p

= Wp(µ1, ν1; c).
Similarly, we have GSWp(µ2, ν2; g1, c) ≤Wp(µ2, ν2; c).
Therefore, we obtain the proof of
H2SWp(µ, ν; c, g1, g2) ≤Wp(µ1, ν1; c) + Wp(µ1, ν1; c).

(iii) When g1, g2 are linear defining functions, we have:
H2SWp(µ, ν; c, g1, g2)

≤

E

inf
π∈Π(µ,ν)

Z
(|θ⊤
1 x1 −θ⊤
1 y1)| + |θ⊤
2 x2 −θ⊤
2 y2|)pdπ(x1, x2, y1, y2)
 1

p

≤

E

inf
π∈Π(µ,ν)

Z
(|θ⊤
1 x1 −θ⊤
1 y1)| + |θ⊤
2 x2 −θ⊤
2 y2|)pdπ(x1, x2, y1, y2)
 1

p

≤

E

inf
π∈Π(µ,ν)

Z
(|x1 −y1)| + |x2 −y2|)pdπ(x1, x2, y1, y2)
 1

p

=

inf
π∈Π(µ,ν)

Z
(|x1 −y1)| + |x2 −y2|)pdπ(x1, x2, y1, y2)
 1

p

When p = 1, we obtain:

H2SW1(µ, ν; c, g1, g2) ≤

inf
π∈Π(µ,ν)

Z
(|x1 −y1)| + |x2 −y2|)dπ(x1, x2, y1, y2)
 1

p

= W1(µ, ν; c, c),

16


---Page Break---
which completes the proof.

A.5
Proof of Proposition 4

Let p ≥1, c(x, y) = |x−y|, µ ∈P(R) with the corresponding empirical distribution µn, we assume
that there exists q > p such that the q−th order moment of µ i.e, Mq(µ) =
R

R |x|qdµ(x), is bounded
by B < ∞. From Theorem 1 in [20], there exists a constant Cp,q such that:

E

W p
p (µn, µ; c)

≤Cp,qB








n−1/2 if q > 2p,
n−1/2 log(n)
1
p if q = 2p,
n−(q−p)/q if q ∈(p, 2p).

We show that HHRg1,g2
θ1,θ2,ψ♯µ has finite bounded moments. In particular, we have:

Mk(HHRg1,g2
θ1,θ2,ψ♯µ) =
Z

R
|t|kd(HHRg1,g2
θ1,θ2,ψ♯µ)(t)

=
Z

Rd1×Rd2
|ψ1g1(θ1, x1) + ψ2g2(θ2, x2)|kdµ(x1, x2)

≤
Z

Rd1×Rd2
(ψ2
1 + ψ2
2)k/2(g1(θ1, x1)2 + g2(θ2, x2)2)k/2dµ(x1, x2)

≤
Z

Rd1×Rd2
(|g1(θ1, x1)| + |g2(θ2, x2)|)kdµ(x1, x2),

where the first inequality is due to the Cauchy-Schwarz inequality and the second inequality is due
to the fact that ∥x∥2 ≤|x|. For the linear defining functions g(θ, x) = θ⊤x, we have |g(θ, x)| =
|θ⊤x| ≤∥x∥1. For the circular defining functions g(θ, x) = ∥x −rθ∥2 ≤∥x −rθ∥1 ≤∥x∥1 +
∥rθ∥1 ≤∥x∥1 + r. Therefore, we have:

Mk(HHRg1,g2
θ1,θ2,ψ♯µ) ≤
Z

Rd1×Rd2
(|x1| + |x2| + Cg1,g2)kdµ(x1, x2)

=
Z

Rd1×Rd2

k
X

i=0
ki(|x1| + |x2|)iCk−i
g1,g2dµ(x1, x2)

=

k
X

i=0
kiCk−i
g1,g2

Z

Rd1×Rd2
(|x1| + |x2|)idµ(x1, x2)

≤

k
X

i=0
kiCk−i
g1,g2Mi(µ),

where Cg1,g2 = 0 if g1, g2 are linear, Cg1,g2 = r if g1 and g2 are linear and circular respectively
(exchangeable), and Cg1,g2 = 2r if both g1 and g2 are circular.

Now, using the triangle inequality of H2SW (Theorem 1), we have:

E |H2SWp(µn, νn; c, g1, g2) −H2SWp(µ, ν; c, g1, g2)|
≤E |H2SWp(µ, µn; c, g1, g2) + H2SWp(ν, νn; c, g1, g2)|
≤E |H2SWp(µ, µn; c, g1, g2)| + E |H2SWp(ν, νn; c, g1, g2)|

≤
 
E
H2SWp
p(µ, µn; c, g1, g2)
 1

p +
 
E
H2SWp
p(ν, νn; c, g1, g2)
 1

p ,

where the last inequality is due to Holder’s inequality. Combining with previous results, we obtain:

E |H2SWp(µn, νn; c, g1, g2) −H2SWp(µ, ν; c, g1, g2)|

≤C

1
p
p,q

 q
X

i=0
qiCq−i
g1,g2(Mi(µ) + Mi(ν))

! 1

p







n−1/2p if q > 2p,
n−1/2p log(n)
1
p if q = 2p,
n−(q−p)/pq if q ∈(p, 2p),

which completes the proof.

17


---Page Break---
A.6
Proof of Proposition 5

For any p ≥1, and µ, ν ∈P(Rd1 × Rd2), using the Holder’s inequality, we have:

E| \
H2SW
p
p(µ, ν; c, g1, g2, L) −H2SWp
p(µ, ν; c, g1, g2)|

≤

E| \
H2SW
p
p(µ, ν; c, g1, g2, L) −H2SWp
p(µ, ν; c, g1, g2)|2 1

2

=



E


1
L

L
X

l=1
Wp
p(HHRg1,g2
θ1l,θ2l,ψl♯µ, HHRg1,g2
θ1l,θ2l,ψl♯ν; c) −E
h
Wp
p(HHRg1,g2
θ1,θ2,ψ♯µ, HHRg1,g2
θ1,θ2,ψ♯ν; c)
i

2



1
2

=

 

V ar

"
1
L

L
X

l=1
Wp
p(HHRg1,g2
θ1l,θ2l,ψl♯µ, HHRg1,g2
θ1l,θ2l,ψl♯ν; c)

#! 1

2

=
1
√

L
V ar
h
Wp
p(HHRg1,g2
θ1l,θ2l,ψl♯µ, HHRg1,g2
θ1l,θ2l,ψl♯ν; c)
i 1

2 ,

which completes the proof.

B
Additional Materials

HHRT with more than two marginals. We now extend the definition of HHRT to K > 2 mariginals.
Definition 4 (Hierarchical Hybrid Radon Transform). Given K ≥2, given defining functions
{gk : Rdk × Ωi →R}K
i=k , the Hierarchical Hybrid Radon Transform HHR : L1(Rd1 × . . . ×
RdK) →L1
 
R × Ω1 . . . × ΩK × SK−1
is defined as:

(HHRf)(t, θ1, . . . , θK, ψ)

=
Z

Rd1×...×RdK
f(x1, . . . , xK)δ

 

t −

K
X

k=1
ψkgk(xk, θk)

!

dx1 . . . dxK.
(12)

H2SW with more than two marginals. From the new definition of HRRT on K > 2 mariginals, we
now can define H2SW between joint distributions with K mariginals.
Definition 5. For p ≥1,K ≥2, defining functions g1, . . . , gK, the hierarchical hybrid sliced
Wasserstein-p (H2SW) distance between two distributions µ ∈P(X1 × . . . × XK) and ν ∈P(Y1 ×
. . . × YK) with an one-dimensional ground metric c : R × R →R+ is defined as:

H2SWp
p(µ, ν; c, g1, . . . , gK)

= E(θ1,...,θK,ψ)∼U(Ω1×...×ΩK×SK−1)[Wp
p(HHRg1,...,gK
θ1,...,θK,ψ♯µ, HHRg1,...,gK
θ1,...,θK,ψ♯ν; c)],
(13)

where HHRg1,...,gK
θ1,...,θK,ψ♯µ and HHRg1,...,gK
θ1,...,θK,ψ♯ν are the one-dimensional push-forward distributions
created by applying HHRT.

Lorentz Model and Busemann function. The Lorentz model Ld ∈Rd+1 of a d-dimensional
hyperbolic space is [7]:

Ld =

(

(x1, . . . , xd) ∈Rd+1, −x0y0 +

d
X

i=1
xiyi = −1, x0 > 0

)

.

Given a direction θ ∈Tx0Ld ∩Sd, x ∈Ld, the Busemann function is:

B(x, θ) = log(−⟨x, x0 + θ⟩).

Busemann function on product Hadamard manifolds. For distributions supports on the product of
K ≥2 Hadamard manifolds with the corresponding Busemann functions B1, . . . , BK, we have a
Busemann function of the product manifolds is:

B(x1, . . . , xK, θ1, . . . , θK) =

K
X

k=1
λkBk(xk, θk),

18


---Page Break---
Figure 6: Visualization of deformation from the sphere mesh to the Armadillo mesh with L = 100.

for (λ1, . . . , λK) ∈SK−1. The Cartan-Hyperbolic Sliced-Wasserstein distance use a fixed value of
(λ1, . . . , λK) e.g., (λ1, . . . , λK) = (1/
√

K, . . . , 1/
√

K) (see 2). In our proposed H2SW, we treat
(λ1, . . . , λK) as a random variable follows U(SK−1) and the value of H2SW is defined as the mean
of such random variable.

C
Related Works

HHRT and Generalized Radon Transform. HHRT can be also seen as a special case of GRT [3]
with the defining function g(x, θ) = ψ1g1(x1, θ1) + ψ2g2(y, θ2) with x = (x1, x2) and θ =
(θ1, θ2, ψ) (Ω= Ω1 × Ω2 × S). However, without approaching via the hierarchical construction, the
injectivity of the transform might be a challenge to obtain.

HHRT and Hierarchical Radon Transform. Hierarchical Radon Transform (HRT) [43] is the
composition of Partial Radon Transform and Overparameterized Radon Transform, which is designed
specifically for reducing projection complexity when using Monte Carlo estimation. Moreover, HRT
is introduced with linear projection and does not focus on the problem of comparing heterogeneous
joint distributions. In contrast to HRT, the proposed HHRT is the composition of multiple partial
Generalized Radon Transform and Partial Random Transform, which is suitable for comparing
heterogeneous joint distributions.

HHRT and convolution slicers. Convolution slicers [40] are introduced to project an image into
a scalar. It can be viewed as a Hierarchical Partial Radon Transform i.e., small parts of the image
are transformed first, then be aggregated later. Although convolution slicers can separate global and
local information as HHRT, they focus on the domain of images only and have not been proven to
be injective. Again, HHRT is designed to compare heterogeneous joint distributions and is proven
to be injective in Proposition 2. As a result, H2SW is a valid metric while convolution sliced
Wasserstein [40] is only a pseudo metric. Moreover, H2SW can also use convolution slicers when
having marginal domains as images.

D
Additional Experiments

3D Mesh Deformation. As mentioned in the main text, we present the deformation visualization
to the Armadillo mesh with L = 100 in Figure 6, and the deformation visualization to the Stanford
Bunny o mesh with L = 10 and L = 100 in Figure 3- 7 in turn. The quantitative result for the
Armadillo mesh is given in Table 2. Here, we set the step size to 0.1. From these results, we see
that the proposed H2SW gives the best flow deformation flow in general. The performance gap is
especially larger when L = 10 i.e., having a small number of projections.

2https://github.com/clbonet/Sliced-Wasserstein_Distances_and_Flows_on_
Cartan-Hadamard_Manifolds/blob/0eb05450e7f9f27586d0ddb1ce6e58f07eb75786/Experiments/
xp_otdd/OTDD_SW.ipynb

19


---Page Break---
Figure 7: Visualization of deformation from the sphere mesh to the Stanford Bunny mesh with L = 100.

Figure 8: Visualization of some randomly selected reconstruction meshes from autoencoders trained by SW,
GSW, and H2SW in turn with the number of projections L = 100 at epoch 500.

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.28±0.033
0.475±0.069
0.237±0.028
0.392±0.049

0.28±0.033
0.448±0.051
0.224±0.024
0.483±0.055

0.475±0.069
0.448±0.051
0.306±0.035
0.22±0.023

0.237±0.028
0.224±0.024
0.306±0.035
0.248±0.02

0.392±0.049
0.483±0.055
0.22±0.023
0.248±0.02

0.0

0.1

0.2

0.3

0.4

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.199±0.025
0.376±0.048
0.188±0.025
0.286±0.04

0.199±0.025
0.338±0.047
0.169±0.019
0.33±0.037

0.376±0.048
0.338±0.047
0.231±0.029
0.19±0.019

0.188±0.025
0.169±0.019
0.231±0.029
0.181±0.017

0.286±0.04
0.33±0.037
0.19±0.019
0.181±0.017

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.204±0.029
0.378±0.056
0.191±0.019
0.281±0.035

0.204±0.029
0.343±0.042
0.173±0.02
0.34±0.048

0.378±0.056
0.343±0.042
0.244±0.028
0.195±0.019

0.191±0.019
0.173±0.02
0.244±0.028
0.186±0.017

0.281±0.035
0.34±0.048
0.195±0.019
0.186±0.017

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

956.8
1273.27
1139.95
843.07

956.8
1278.35
1157.0
891.65

1273.27
1278.35
1253.58
607.44

1139.95
1157.0
1253.58
843.23

843.07
891.65
607.44
843.23

0

200

400

600

800

1000

1200

SW
CHSW
H2SW
W

Figure 9: Cost matrices between datasets from SW, CHSW, and H2SW with L = 100.

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.277±0.018
0.464±0.029
0.242±0.011
0.389±0.022

0.277±0.018
0.451±0.024
0.225±0.013
0.482±0.026

0.464±0.029
0.451±0.024
0.308±0.019
0.221±0.011

0.242±0.011
0.225±0.013
0.308±0.019
0.246±0.009

0.389±0.022
0.482±0.026
0.221±0.011
0.246±0.009

0.0

0.1

0.2

0.3

0.4

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.199±0.012
0.375±0.022
0.188±0.011
0.281±0.017

0.199±0.012
0.337±0.02
0.171±0.008
0.333±0.02

0.375±0.022
0.337±0.02
0.235±0.013
0.191±0.01

0.188±0.011
0.171±0.008
0.235±0.013
0.179±0.007

0.281±0.017
0.333±0.02
0.191±0.01
0.179±0.007

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.201±0.011
0.381±0.021
0.188±0.009
0.284±0.017

0.201±0.011
0.343±0.019
0.173±0.011
0.338±0.019

0.381±0.021
0.343±0.019
0.246±0.013
0.194±0.009

0.188±0.009
0.173±0.011
0.246±0.013
0.187±0.008

0.284±0.017
0.338±0.019
0.194±0.009
0.187±0.008

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

956.8
1273.27
1139.95
843.07

956.8
1278.35
1157.0
891.65

1273.27
1278.35
1253.58
607.44

1139.95
1157.0
1253.58
843.23

843.07
891.65
607.44
843.23

0

200

400

600

800

1000

1200

SW
CHSW
H2SW
W

Figure 10: Cost matrices between datasets from SW, CHSW, and H2SW with L = 500.

20


---Page Break---
MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.278±0.012
0.466±0.019
0.24±0.009
0.394±0.015

0.278±0.012
0.448±0.019
0.228±0.009
0.483±0.019

0.466±0.019
0.448±0.019
0.305±0.011
0.221±0.008

0.24±0.009
0.228±0.009
0.305±0.011
0.245±0.007

0.394±0.015
0.483±0.019
0.221±0.008
0.245±0.007

0.0

0.1

0.2

0.3

0.4

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.197±0.008
0.373±0.013
0.186±0.007
0.283±0.012

0.197±0.008
0.338±0.015
0.168±0.006
0.335±0.014

0.373±0.013
0.338±0.015
0.235±0.009
0.19±0.007

0.186±0.007
0.168±0.006
0.235±0.009
0.179±0.005

0.283±0.012
0.335±0.014
0.19±0.007
0.179±0.005

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

0.2±0.01
0.381±0.015
0.188±0.007
0.289±0.013

0.2±0.01
0.342±0.013
0.172±0.006
0.342±0.016

0.381±0.015
0.342±0.013
0.244±0.009
0.195±0.006

0.188±0.007
0.172±0.006
0.244±0.009
0.187±0.006

0.289±0.013
0.342±0.016
0.195±0.006
0.187±0.006

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

MNIST
EMNIST
FashionMNIST
KMNIST
USPS

MNIST

EMNIST

FashionMNIST

KMNIST

USPS

956.8
1273.27
1139.95
843.07

956.8
1278.35
1157.0
891.65

1273.27
1278.35
1253.58
607.44

1139.95
1157.0
1253.58
843.23

843.07
891.65
607.44
843.23

0

200

400

600

800

1000

1200

SW
CHSW
H2SW
W

Figure 11: Cost matrices between datasets from SW, CHSW, and H2SW with L = 1000.

Deep 3D mesh autoencoder. We first report the neural network architectures that we use in the
experiments.

• The encoder: Conv1d(6,64,1) →BatchNorm1d →LeakyReLU(0.2) →Conv1d(64, 128,
1) →BatchNorm1d →LeakyReLU(0.2) →Conv1d(128, 256, 1) →BatchNorm1d
→LeakyReLU(0.2) →Conv1d(256, 512, 1) →BatchNorm1d →LeakyReLU(0.2) →
Conv1d(512, 1024, 1) →BatchNorm1d →LeakyReLU(0.2) →Max-Pooling →Lin-
ear(1024, 1024).
• The decoder: Linear(1024, 1024) →BatchNorm1d →LeakyReLU(0.2) →Linear(1024,
2048) →BatchNorm1d →LeakyReLU(0.2) →Linear(2048, 4096) →BatchNorm1d →
LeakyReLU(0.2) →Linear(2048, 2048*6). The output of the decoder is the concatenation
of the location and normal vector. We normalize the normal vector to the unit-sphere.

As mentioned in the main text, we report the reconstruction of randomly selected meshes for L = 100
at epoch 500 in Figure 8. We see that the reconstructed meshes at epoch 500 are visually worse than
the reconstructed meshes at epoch 2000. Therefore, the joint Wasserstein distances in Table 3 are
consistent with the qualitative results.

Dataset Comparison. We follow the same procedure in Section 6.2 in [9]. We refer the reader to the
reference for a detailed description. Here, we show the cross-dataset cost matrices with the number
of projections L = 100 in Figure 9, L = 500 in Figure 10, and L = 1000 Figure 11.

E
Computational Infrastructure

For the non-deep-learning experiments, we use a HP Omen 25L desktop for conducting experiments.
For 3D mesh autoencoder experiments, we use a single NVIDIA A100 GPU.

NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: As in the abstract and introduction, we focus on designing a sliced Wasserstein
variant for heterogeneous joint distributions.
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

21


---Page Break---
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The proposed hierarchical hybrid Radon transform costs slightly more com-
putation as discussed in Section 3.1. Also, the injectivity of the hierarchical hybrid Radon
transform depends on the injectivity of its partial generalized Radon Transform component
as discussed in Section 3.1.

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

Justification: We state all assumptions for our theoretical results in the paper.

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

22


---Page Break---
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We report all experimental settings for our experiments in Section 4 and
Appendices.
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
Answer: [Yes]
Justification: We submitted the anonymized code for experiments in the paper.
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

23


---Page Break---
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
Justification: We report the training and test details in the experimental parts of the paper in
Section 4.
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
Justification: We run our experiments at least three independent times and report the error
bars.
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

24


---Page Break---
Answer: [Yes]

Justification: We report the computational devices that we use in Appendix E

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

Justification: We follow the NeurIPS Code of Ethics when conducting the research.

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

Justification: We propose a new metric for comparing heterogeneous joint distributions. As
shown in the paper, the proposed metric can improve applications of 3D mesh and datasets
comparison. We believe that there are no direct negative societal impacts of the work.

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

25


---Page Break---
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: We do not collect any data in the paper.

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

Justification: We cite and credit all used assets in the paper.

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

Answer: [Yes]

Justification: We provide anonymized code for the paper with instructions for running the
code.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

26


---Page Break---
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: We do not use crowdsourcing experiments and research with human subjects.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects
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
