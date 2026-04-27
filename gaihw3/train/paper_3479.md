Can neural operators always be continuously
discretized?

Takashi Furuya1,∗
Michael Puthawala2,∗
Maarten V. de Hoop3
Matti Lassas4

1Shimane University, takashi.furuya0101@gmail.com
2South Dakota State University, Michael.Puthawala@sdstate.edu
3Rice University, mdehoop@rice.edu
4University of Helsinki, matti.lassas@helsinki.fi

* These authors contributed equally to this work

Abstract

We consider the problem of discretization of neural operators between Hilbert
spaces in a general framework including skip connections. We focus on bijec-
tive neural operators through the lens of diffeomorphisms in infinite dimensions.
Framed using category theory, we give a no-go theorem that shows that diffeomor-
phisms between Hilbert spaces or Hilbert manifolds may not admit any continuous
approximations by diffeomorphisms on finite-dimensional spaces, even if the ap-
proximations are nonlinear. The natural way out is the introduction of strongly
monotone diffeomorphisms and layerwise strongly monotone neural operators
which have continuous approximations by strongly monotone diffeomorphisms
on finite-dimensional spaces. For these, one can guarantee discretization invari-
ance, while ensuring that finite-dimensional approximations converge not only as
sequences of functions, but that their representations converge in a suitable sense
as well. Finally, we show that bilipschitz neural operators may always be written
in the form of an alternating composition of strongly monotone neural operators,
plus a simple isometry. Thus we realize a rigorous platform for discretization of
a generalization of a neural operator. We also show that neural operators of this
type may be approximated through the composition of finite-rank residual neural
operators, where each block is strongly monotone, and may be inverted locally via
iteration. We conclude by providing a quantitative approximation result for the
discretization of general bilipschitz neural operators.

1
Introduction

Neural operators, first introduced in [30], have become more and more prominent in deep learning on
function spaces. As opposed to traditional neural networks that learn maps between finite-dimensional
Euclidean spaces, neural operators learn maps between infinite-dimensional function spaces yet may
be trained and evaluated on finite-dimensional data through a rigorous notion of discretization. Neural
operators are widely used in the field of scientific machine learning [6, 21, 38, 55, 56], among others,
principally because of their discretization invariance. In this work, we consider the fundamental limits
of this discretization. Throughout, we emphasize the importance of continuity under discretization.

A key ingredient in our analysis is the identification of properties of diffeomorphisms that may be
induced by (bijective) neural operators, which are diffeomorphisms themselves. Diffeomorphisms
exist in many contexts, for example, in generative models. These involve mapping one probability
distribution or measure, µ, over some measurable space to another, X, via a push forward, that is,
F#µ(U) = µ(F −1(U)) for U ⊂X. If µ admits a density dµ then, in finite dimensions, we may

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
use the change of variables formula to write dρ(x) = dµ(F −1(x))|JF(x)|−1, where JF is the
Jacobian of F. Clearly, F must be a bijection with full-rank Jacobian. In other words, F must be a
diffeomorphism onto its range. This established diffeomorphisms as natural objects of interest in
finite-dimensional machine learning, and helps account for their wide use [20, 25, 31, 44, 54]. In
this work, we consider the extension of these efforts from finite to infinite dimensions implemented
via neural operators. Although there is no analogue of the change of variables formula in infinite
dimensions, we argue that it is, nonetheless, natural to consider the role of diffeomorphisms, and how
they may be approximated via diffeomorphisms on finite-dimensional spaces.

The question of when operations between Hilbert spaces may be discretized continuously may be
understood through an analogy to computer vision. Consider the task of learning a map from one
image space to another, for example, a style transfer problem [16], where the mapping learned does
not depend much on the resolution of the images provided. It is natural to think of the map as
being defined between (infinite-resolution) continuum images, and then its application to images
of a specific resolution. In this analogy, X is a function space (over images m : [0, 1]2 →R [29])
that is approximated with a finite-dimensional space Rd and the transformation F : X →X is
approximated by a map f : Rd →Rd, where each f acts on images of a particular resolution.
An explicit transformation formula can be obtained when f is a diffeomorphism and has a smooth
inverse.

We introduce a framework based on a generalized notion of neural operator layers including a skip
connection and their restrictions to balls rather than compact sets. With bijective neural operators
in mind, we give a perspective based on diffeomorphisms in infinite dimensions between Hilbert
manifolds. We give a no-go theorem, framed with category theory, that shows that diffeomorphisms
between Hilbert spaces may not admit any continuous approximations by diffeomorphisms on finite-
dimensional spaces, even if the underlying discretization is nonlinear. In this framing the discretization
operation is modeled as a functor from the category of Hilbert spaces and C1-diffeomorphisms on
them to their finite-dimensional approximations. A natural way to mitigate the no-go theorem
is described by the introduction of strongly monotone diffeomorphisms and layerwise strongly
monotone neural operators. We prove that all strongly monotone neural operator layers admit
continuous approximations by strongly monotone diffeomorphsisms on finite-dimensional spaces.
We then provide various conditions under which a neural operator layer is strongly monotone. Notably,
a bilipschitz (and, hence, bijective) neural operator layer can always be represented by a composition
of strongly monotone neural operator layers. Hence, such an operator may be continuously discretized.
More constructively, any bilipschitz neural operator layer can be approximated by residual finite-rank
neural operators, each of which are strongly monotone, plus a simple isometry. Moreover, these
finite-rank residual neural operators are (locally) bijective and invertible, and their inverses are limits
of compositions of finite-rank neural operators. Our framework may be used “out of the box” to
prove quantitative approximation results for discretization of neural operators.

1.1
Related work

Neural operators were first introduced in [30]. Alternative designs for mappings between function
spaces are the DeepONet [34, 40], and the PCA-Net [7, 12]. In spite of the multitudinous applications
of neural operators, the theory of the natural class of injective or bijective neural operators is
comparatively underdeveloped; see, for example [2, 14].

Our work is concerned with the of discretization of neural operators through the lens of diffeomor-
phisms. For recent important work on analyzing the effect of discretization error of Fourier Neural
Operators (FNOs) arising from aliasing, see [35]. Our work has connections to infinite-dimensional
inference, see e.g. [19], and approximation theory, see e.g. [13] while bridging the gap with the
theory of neural operators.

We give a no-go theorem that uses a category theory framing. This contributes to the use of category
theory as an emerging tool in the analysis and understanding of neural networks at large. In this
sense, we are in league with the recent work generalizing ideas from geometric machine learning
using category theory [17].

Discretization obstructions have been encountered in other contexts. Numerical methods that ap-
proximate continuous models are known to sometimes fail in surprising ways. A basic example of
this is the “locking” phenomenon in the study of the Finite Elements Method (FEM). For example,

2


---Page Break---
linear elements used to model bending of a curved surface or beam lock in such a way that the model
exhibits a non-physical stiff response to deformations [4]. Understanding this has been instrumental
in developing improved numerical methods, such a high order FEM [52]. Furthermore, in discretized
statistical inverse problems [27, 36, 51], the introduction of Besov priors [48, 11] has been found to
be essential.

Finally, our work extends prior work (not based on deep learning) in discretization of physical or
partial differential equations based forward models in inverse problems. The analogous notion of
discretization invariance of solution algorithms of inverse problems was studied in [37, 48, 51] and the
lack of it (in imaging methods using Bayesian inversion with L1 priors) in [36, 48]. By considering
the neural operator as the physical model, our results state that discretization can be done locally in
an appropriate way, together with constructing an inverse.

1.2
Our contributions

The key results in this paper comprise the following:

1. We prove a general no-go theorem showing that, under general circumstances, diffeo-
morphisms between Hilbert spaces may not admit continuous approximation by finite-
dimensional diffeomorphisms. In particular, neural operators corresponding to diffeomor-
phic maps, in general, cannot be approximated by finite-dimensional diffeomorphisms and
their associated neural representations.

2. We show that strongly monotone neural operator layers admit continuous approximations
by strongly monotone diffeomorphsisms on finite-dimensional spaces.

3. We show that bilipschitz neural operators can be represented in any bounded set as a
composition of strongly monotone, diffeomorphic neural operator layers, plus a simple
isometry. These can be approximated by finite-rank diffeomorphic neural operators, where
each layer is strongly monotone. For these operators we give a quantitative approximation
result.

2
Definitions and notation

In this section, we give the definitions and notation used throughout the paper. First, we summarize
the relevant basic concepts from functional analysis. Then, we introduce generalized neural operators.

2.1
Elements of functional analysis

In this work, all Hilbert spaces, X, are endowed with their norm topology. We denote by BX(r) =
BX(0, r) the ball in the space X having the center at zero and radius r > 0. We denote by S(X) the
set of all finite-dimensional linear subspaces V ⊂X. The set S0(V ) ⊂S(X) is a partially ordered
lattice. That is, if V1, V2 ∈S0(X) then there is a V3 ∈S0(X) so that V1 ⊂V3 and V2 ⊂V3. 1 With
Y standing for another Hilbert space, we denote by Cn(X; Y ) the set of operators, F : X →Y ,
having n continuous (Fréchet) derivatives, and Cn(X) = Cn(X; X).

Next, we define what it means that a nonlinear operator or function F : X →X on an infinite-
dimensional Hilbert space, X, is approximated by operators or functions on finite-dimensional
subspaces V ⊂X. The key is that as V tends to X, the complexity of FV increases and one may
hope that the approximation becomes better. We formalize this in the following definition.

Definition 1 (ϵV approximators and weak approximators). (i) Let r > 0, F ⊂Cn(X; X) be a family
of functions, and ⃗ε = (εV )V ∈S0(X) be a sequence such that εV →0 as V →X. We say that a
function
AX : F →×V ∈S0(X) C(BV (0, r); V ),
F →(FV )V ∈S0(X)

1Each element of S0(X) will come to represent a discretization of X. The partially ordered lattice condition
will come to represent a notion of common refinement of a discretization. This makes it possible to consider
“realistic” choices of discretizations. The condition automatically follows for any discretization scheme that has
a notion of “common refinement” of two discretizations. Examples include finite-difference schemes, and the
finite elements Galerkin discretization that is based on a triangulation of the domain.

3


---Page Break---
is an ⃗ε-approximation operation for functions F in the ball BX(0, r) taking values in families
FV ⊂C1(V ; V ) if AX maps a function F : X →X, where F ∈F, to a sequence of functions
(FV )V ∈S0(X), where FV ∈FV , such that the following is valid: For all F : X →X satisfying
∥F∥Cn(BX(0,r);X) ≤M, we have

sup

x∈BV (0,r)
∥FV (x) −PV (F(x))∥X ≤MεV ,
(1)

where PV : X →X is the orthogonal projection onto V , that is, Ran(PV ) = V .

(ii) We say that A: Cn(X; X) →×V ∈S0(X) C(V ; V ),
F →(FV )V ∈S0(X) is a weak approxima-
tion operation for the family F ⊂Cn(X; X) if for any F ∈F and r > 0 it holds that

lim
V →X
sup

x∈BV (0,r)
∥FV (x) −PV (F(x))∥X →0.

Note that the condition (i) is stronger than the condition (ii). An example of an approximation
operation for the family F = Cn(X), that is, an ⃗ε-approximation operation with all sequences
⃗ε = (εV )V ∈S0(X) subject to εV > 0, is the linear discretization

Alin(F) = (FV )V ∈S0(X),
FV = PV ◦(F|V ) : V →V.
(2)

Nonlinear discretization methods that do not rely on PV have been used, for example, in the
numerical analysis of nonlinear partial differential equations. Here, X becomes an appropriate
Sobolev space, and a Galerkin approximation is implemented through finite-dimensional subspaces,
V , spanned by finite element basis functions. We present an example for the nonlinear equation,
∆u(t) −g(u(t)) = ∆x(t) where g is a smooth convex function, when F : x →u, in Appendix A.1.

Below, we will study whether a family, F ⊂Diff1(X), of C1 diffeomorphisms on X can be
approximated by C1 diffeomorphisms, FV ⊂Diff1(V ), on finite-dimensional subspaces, V . Of
course, diffeomorphisms are bijective. Unless stated otherwise, from now on we will omit C1 and
implicitly assume that diffeomorphism are C1 diffeomorphisms. We introduce two more notions that
will play a key role in the further analysis.
Definition 2 (Strongly Monotone). We say that a (nonlinear) operator F : X →X on Hilbert space,
X, is strongly monotone if there exists a constant α > 0 so that

⟨F(x1) −F(x2), x1 −x2⟩X ≥α∥x1 −x2∥2
X,
for all x1, x2 ∈X.
(3)

Definition 3 (Bilipschitz). We say that F if bilipschitz there exist constants c > 0 and C < ∞so
that for all x1, x2 ∈X, c ∥x1 −x2∥≤∥F(x1) −F(x2)∥≤C ∥x1 −x2∥.

2.2
A general framework for neural operators

In this paper, we are concerned with the modeling of diffeomorphisms between Hilbert spaces by
bijective neural operators. Our working definition of neural operator, which generalizes the traditional
notion, is given below. We note the presence of a skip connection, which is essential.
Definition 4 (Generalized neural operator layer). For Hilbert space X, a layer of a neural operator
is a function F : X →X of the form

F(x) = x + T2G(T1x),
(4)

where T1 : X →X and T2 : X →X are compact linear operators 2 and G: X →X is a nonlinear
operator in C1(X). A generalized neural operator, H : X →X, is given by the composition

H = AL ◦σ ◦FL ◦AL−1 ◦σ ◦FL−1 ◦· · · ◦A1 ◦σ ◦F1,
(5)

where each Fℓ, ℓ= 1, . . . , L is of the form (4), the Aℓ: X →X are bounded linear operators and
σ: X →X is a continuous operation (for example, a Nemytskii operator defined by a composition
with a suitable activation function in function spaces).

2By using mapping properties of monotone operators [5], we can replace this definition by using Hilbert
spaces Y and Z that are isometric to X, T1 : X →Y and T2 : Z →X as compact linear operators, and
G : Y →Z as a C1 nonlinear operator.

4


---Page Break---
The generalized neural operators can represent the classical neural operators [30, 33]. For an explicit
construction, we refer to Appendix C.1. In the next section, we will study, under what conditions,
bounded linear operators Aℓ, Nemytskii operators σ, and neural operator layers Fℓ, for which the
generalized neural operator consists, can be continuously discretized.

We note that because G ∈C1(X) in Definition 4, it follows that G ∈L∞(X) and LipX→X(G) < ∞.
Given a Hilbert space, X, a layer of a strongly monotone neural operator (respectively, a layer of
a bilipschitz neural operator,) is a function F : X →X that is strongly monotone (respectively,
bilipschitz). Furthermore, a strongly monotone neural operator (respectively, a bilipschitz neural
operator), is a generalized neural operator with strongly monotone (respectively, bilipschitz), layers.

3
Category theoretic framework for discretization

“Well-behaved” operators between infinite-dimensional Hilbert spaces may have dramatically dif-
ferent behaviors than corresponding “well-behaved” maps between finite-dimensional Euclidean
spaces. This observation applies to discretization and neural operators versus neural networks. In
this section we explore this. We first present a no-go theorem, that there are no procedures that
continuously discretize an isotopy of diffeomorphisms. Next, we introduce strongly monotone
neural operator layers, which are strongly monotone diffeomorphisms, and then prove that these
allow a continuous approximation functor, that is, continuous approximations by strongly monotone
diffeomorphisms on finite dimensional spaces. We finally show that bilipschitz neural operator layers
admit a representation via strongly monotone operators and linear maps, allowing for their continuous
approximation.

3.1
No-go theorem for discretization of diffeomorphisms on Hilbert spaces

In this section, we present our no-go theorem. To formulate the ‘impossibility’ of something, we
must define what is meant by discretization and approximation. Before this, we give an informal
statement of the no-go theorem.
Theorem 1 (No-go Theorem, Informal). Let A be an approximation scheme that maps diffeomor-
phisms F on a Hilbert to a sequence of finite-approximations FV that are themselves diffeomorphisms.
If FV converges to F as V →X, then A is not continuous, that is, there are maps F (j) that converge
to F as j →∞, but all F (j)
V
are far from FV .

We want to emphasize that most practical numerical algorithms are continuous so that the output
depends (in some suitable sense) continuously on the input. This shows that there are no such
numerical schemes that approximate infinite-dimensional diffeomorphisms with finite-dimensional
ones. In order to prove our no-go theorem in the most general setting, we phrase it in terms of
category theory. Namely, we formulate A (which will denote the approximation scheme) as a functor
from the category of Hilbert spaces and diffeomorphisms thereon, to their finite-rank approximations.
Definition 5 (Category of Hilbert Space Diffeomorphisms). We denote by D the category of Hilbert
diffeomorphisms with objects OD that are pairs (X, F) of a Hilbert space X and a (possibly non-
linear) C1-diffeomorphism F : X →X and the set of morphisms (or arrows that ‘map’ objects to
other objects) A that are either

1. (induced isomorphisms) Maps aϕ that are defined for a linear isomorphism ϕ : X1 →
X2 of Hilbert spaces X1 and X2 that maps the objects (X1, F1) ∈OD to the object
(ϕ(X1), ϕ ◦F1 ◦ϕ−1) ∈OD, or

2. (induced restrictions) Maps aX1,X2 that are defined for a Hilbert space X1, its closed
subspace X2 ⊂X1, and an object (X1, F1) ∈OD such that F1(X2) = X2. Then aX1,X2
maps to the object (X1, F1) ∈OD to the object (X2, F1|X2) ∈OD.
Definition 6 (Category of Approximation Sequences). We denote by B the category of approximation
sequences, that has objects OB that are of the form (X, S0(X), (FV )V ∈S0(X)) where X is a Hilbert
space,
S0(X) ⊂S(X) = {V | V ⊂X is a finite dimensional linear subspace},
are partially ordered lattices, S

V ∈S0(X) V = X, and FV : V →V are C1-diffeomorphisms of
spaces V ∈S0(X).

5


---Page Break---
The set of morphisms AB consists of either

1. Maps Aϕ that are defined for a linear isomorphism ϕ : X1 →X2 of Hilbert spaces X1
and X2, and lattices S0(X1) and S0(X2) = {ϕ(V ) | V ∈S0(X1)}, that maps the objects
(X1, S(X1), (FV )V ∈S(X1)) to (X2, S(X2), (ϕ ◦Fϕ−1(W ) ◦ϕ−1)W ∈S(X2)), or

2. Maps AX1,X2 that are defined for a Hilbert space X1, its closed subspace X2 ⊂X1,
and an object (X1, S0(X1), (FV )V ∈S0(X1)) such that F(X2) = X2 and S0(X2) =
{V ∈S0(X1) | V ⊂X2} is a partially ordered lattice. Then AX1,X2 maps the object
(X1, S0(X1), (FV )V ∈S0(X1)) to the object (X2, S0(X2), (FV )V ∈S0(X2)).

Next, we define the notion of an approximation or discretization functor. In practice, an approximation
functor is an operator which maps a function F in an infinite dimensional space X to a function FV
that operate in finite dimensional subspaces V of X in such a way that functions FV are close (in a
suitable sense) to the function F.
Definition 7 (Approximation Functor). We define the approximation functor, denoted by A: D →B,
as the functor that maps each (X, F) ∈OD to some (X, S0(X), (FV )V ∈S0(X)) ∈OB so that the
Hilbert space X stays the same. The approximation functor maps all morphisms aϕ to Aϕ and
morphisms aX1,X2 to AX1,X2, and has the following the properties

(A) For all r > 0 and all (X, F) ∈OD,

lim
V →X
sup
x∈BX(0,r)∩V
∥FV (x) −F(x)∥X = 0.

In separable Hilbert spaces this means that when the finite dimensional subspaces V ⊂X
grow to fill the whole Hilbert space X, then the approximations FV converge uniformly in
all bounded subsets to F.

We recall the notation limV →X used above: We consider (S0(X), ⊃) as a partially ordered set and
say that real numbers yV converge to the limit y as V →X, and denote

lim
V →X yV = y,

if for all ϵ > 0 there is V0 ∈S0(X) such that for V ∈S0(X) satisfying V ⊃V0 it holds that
|yV −y| < ϵ.

Definition 8. We say that the approximation functor A is continuous if the following holds:
Let (X, F), (X, F (j)) ∈OD be such that the Hilbert space X is the same for all these ob-
jects and let (X, S0(X), (FV )V ∈S0(X)) = A(X, F) be approximating sequences of (X, F) and
(X, S0(X), (Fj,V )V ∈S0(X)) = A(X, F (j)) be approximating sequences of (X, F (j)). Moreover,
assume that r > 0 and
lim
j→∞
sup
x∈BX(0,r)
∥F (j)(x) −F(x)∥X = 0.
(6)

Then, for all V ∈S0(X) the approximations F (j)
V
of F (j) and FV of F satisfy

lim
j→∞
sup
x∈V ∩BV (0,r)
∥F (j)
V (x) −FV (x)∥X = 0.
(7)

The theorem below states a negative result, namely that there does not exist continuous approximating
functors for diffeomorphisms.
Theorem 2. (No-go theorem for discretization of general diffeomorphisms) There exists no functor
D →B that satisfies the property (A) of an approximation functor and is continuous.

The proof is given in Appendix A.4.1, and is quite involved, but we give an overview of some of the
steps here. A generalization of Theorem 2, in the case where the norm topology is replaced by the
weak topology, is considered in Appendix D.1.

A key fact is that for finite dimensional diffeomorphisms the space of smooth embeddings consists of
two connected components, one orientation preserving and the other orientation reversing. This is

6


---Page Break---
Figure 1: A figure illustrating the proof ideas for Theorem 2. It represents the disconnected com-
ponents of diffeomorphisms that preserve orientation, notated by diff+, and reverse orientation,
notated, diff−. The horizontal axis abstractly represents the two disconnected components of diff for
a finite-dimensional vector space V . The vertical axis represents the dimension of V . Observe how
the two components of diff connect as dim(V ) →∞, and V becomes a Hilbert space H.

not the case in infinite dimensions, see e.g. [32] and [45]. For an illustration of this, see Figure 1.
The proof proceeds by contradiction. First, we consider the action of the approximation functor as it
operates on an isotopy (path of diffeomorphisms) that connects two diffeomorphisms. The first has
only orientation-preserving discretizations, and the second only orientation-reversing discretizations.
We then show that the image of the path under the approximation functor yields a disconnected path,
as the discretization ‘jumps’ from the orientation preserving component to the orientation reversing
component. This violates continuity. To encode the notions of orientation preserving and orientation
reversing that allow for a description of nonlinear discretization theory, we use topological degree
theory. This generalizes the familiar notion of orientation that uses the sign of the determinant of the
Jacobian matrix.

3.2
Strongly monotone diffeomorphisms and their approximation on finite-dimensional
subspaces

In this section and the next two we show that, although Theorem 2 precludes continuous approxima-
tion of general diffeomorphisms, stronger constraints on the diffeomorphisms allows one to sidestep
the topological obstruction. In this section, in summary, we show that the obstruction to continuous
approximation vanishes when the diffeomorphisms in question are assumed to be strongly monotone.
Key to our positive result is the following technical result that states that the restriction of the domain
and codomain of a strongly monotone diffeomorphism always yields another strongly monotone
diffeomorphism.

Lemma 1. Let V ⊂X be a finite-dimensional subspace of X, and let PV : X →X be orthogonal
projection onto V . Let F : X →X be a strongly monotone C1-diffeomorphism. Then, PV F|V :
V →V is strongly monotone, and a C1-diffeomorphism.

The proof is given in Appendix A.5.1. Lemma 1 implies that the discretization functor Alin de-
fined in (2) is a well-defined functor from strongly monotone C1-diffeomorphisms of X to C1-
diffeomorphisms of V . Note that the discretization functor Alin on strongly monotone C1 diffeomor-
phisms may not be a continuous approximation functor in the strong sense of Definitions 7 and 8,
but it is obviously a continuous approximation functor in the weak sense of Definitions 11 and 12.
Therefore, we obtain that :

Proposition 1. Let Alin be the discretization functor that maps F to PV F|V for each finite subspace
V ⊂X. Let Dsm and Bsm be categories where F : X →X and FV : V →V are strongly monotone
C1-diffeomorphisms. Then, the functor Alin : Dsm →Bsm satisfies assumption (A’) of a weak
approximation functor in Definition 11, and is continuous in the weak sense of Definition 12.

3.3
Strongly monotone Nemytskii operators and linear bounded operators and their
continuous approximation on finite-dimensional subspaces

By Proposition 1, strongly monotone maps can be continuously discretized in the weak sense. Thus,
we concern under what conditions, the maps, of which generalized neural operator consists, can be

7


---Page Break---
strongly monotone. In this subsection, we focus on bounded linear operators and Nemytskii operators
(layers of neural operator will be discussed in the next subsection). The following lemma is obviously
given by the definition of a strongly monotone map :

Lemma 2. Let A : X →X be a linear bounded operator and satisfy ⟨Au, u⟩≥c0∥u∥2
X for some
c0 > 0. Then, A : X →X is strongly monotone.

Next, assuming that X = L2(D; R), we define Nemytskii operator by

F σ(u) = σ ◦u,
(8)

where σ : R →R is continuous. In this case, we can show the following lemma by using [50,
Corollary 3.3]:

Proposition 2. Assume that σ satisfies that |σ(s)| ≤C1|s| + C2 and the derivative of s →σ(s) is
defined a.e and satisfies σ′(s) ≥α > 0. Then, F σ : L2(D; R) →L2(D; R) is strongly monotonous.

3.4
Strongly monotone generalized neural operators and their continuous approximation on
finite-dimensional subspaces

As seen in Theorem 3 below, strongly monotone layers of neural operators do not suffer from the
same topological obstruction to continuous discretization as general diffeomorphisms. We now give
sufficient conditions for the layers of a neural operator to be strongly monotone, and show that these
conditions imply that those are diffeomorphisms.

Lemma 3. All strongly monotone layers of neural operators (F) defined by (4) are diffeomorphisms.

Also, the following theorem is proven in Appendix A.6.3.

Theorem 3. Let Alin be the discretization functor that maps F to PV F|V for each finite subspace
V ⊂X. Let Dsmn and Bsmn be categories where F : X →X and FV : V →V are strongly
monotone C1-functions of the form (4). Then, the functor Alin : Dsmn →Bsmn satisfies assumption
(A), and it is continuous in the sense of Definition 8.

The functor defined in Theorem 3 does not suffer from the same topological obstruction as functors
for general diffeomorphisms, shown in the no-go Theorem 2.This is because when FV = PV F|V is
strongly monotone, its derivative D|xFV : V →V is a strongly monotone matrix at all points x ∈V .
Therefore it is strictly positive definite (see [47, Prop. 12.3]) and the determinant det(D|xFV ) is
strictly positive. Due to this, the orientation of the finite-dimensional approximations never switch
signs, and the key technique used in the proof of the no-go Theorem 2 does not apply.

A straightforward condition to guarantee strong monotonicity of a neural operator layer is given in

Lemma 4. Let F : X →X be a layer of neural operator that is of the form F(u) = u + T2G(T1u),
where Tj : X →X, j = 1, 2 are compact operators and G : X →X is a C1-smooth map. Assume
that Fréchet derivative DG|x of G at x satisfies the following for all x ∈X,

∥DG|x∥X→X ≤1

2∥T1∥−1
X→X∥T2∥−1
X→X.

Then, F : X →X is strongly monotone.

See Appendixes A.6.1 and A.6.2 for the proofs.

3.5
Bilipschitz neural operators are conditionally strongly monotone diffeomorphisms

Now we show an analogous result to Theorem 3, but applied to bilipschitz neural operators. Moreover,
we will show that all neural operator F : X →X that are bilipschitz admit approximations that can
be locally inverted using iteration for each point in their range using an iteration.

Theorem 4. Let X be a Hilbert space. Then there is e ∈X, ∥e∥X = 1 such that the following is
true: Let F : X →X be a layer of a bilipschitz neural operator. Then for all r1 > 0 and ϵ > 0 there
are a linear invertible map A0 : X →X, that is either the identity map or a reflection operator3

3Note that we can write the reflection operator across the hyperplane {e}⊥, that is, the operator x →
x −2⟨x, e⟩Xe as a diagonal operator diag(−1, 1, 1, . . . ) in a suitable orthogonal basis of X.

8


---Page Break---
x →x −2⟨x, e⟩Xe, and strongly monotone functions Hk that are also layers of neural operators
such that
Hk : X →X,
Hk(x) = x + Bk(x),
k = 1, 2, . . . , J,
where Bk : X →X is a compact mapping and satisfies Lip(Bk) < ϵ and

F(x) = HJ ◦· · · ◦H2 ◦H1 ◦A0(x),
for all x ∈BX(0, r1).
(9)

Moreover, if F ∈C2(X, X), then J = O(ϵ−2).

The proof of Theorem 4 is in Appendix A.7.1. Theorem 4 shows that we may always decompose a
bilipschitz neural operator into the composition of strongly monotone neural operator layers Hj and a
reflection operator A0. Each Hj can be discretized using the continuous functor Alin from Theorem 3.
If we consider the discretization (via the construction in Definition 6) using a collection of subsets
S0(X) ⊂S(X) such that all V ∈S0(X) satisfy e ∈V , then the operator A0 can be discretized by
A0,V = PV ◦A0|V . These mean that if we write a bilipschitz neural operator as a sufficiently deep
neural operator where each layer is either of the form Id + Bj, where Lip(Bj) < 1, or a reflection
operator A0. In either case, we may use linear discretization to approximate each layer. So, we may
discretize F in a ball BX(0, R) by discretizing each layer Hj and A1 where F = HJ ◦· · · ◦H1 ◦A1.
We observe that the number of layers, J, depends on R.

We have observed that operators of the form identity plus a compact term are critical for continuous
discretization. This insight motivates the introduction of residual networks as approximators within
the framework of finite-rank neural operators. In what follows, we assume that X is a separable
Hilbert space, with an orthonormal basis φ = {φn}n∈N. For N ∈N, we define EN : X →RN and
DN : RN →X by ENu := (⟨u, φ1⟩X, ..., ⟨u, φN⟩X) ∈RN.
DNα := P

n≤N αnφn. We note
that PVN = DNEN, where PVN : X →X is the projection onto VN := span{φn}n≤N. Using EN,
DN, we define the class of residual networks in the separable Hilbert space, with T, N ∈N and
activation function σ, as

RT,N,φ,σ(X) :=
n
G : X →X : G = ⃝T
t=1(IdX + DN ◦NNt ◦EN),

NNt : RN →RNare neural networks with activation function σ (t = 1, ..., T)
o
.
(10)

The following theorem proves a universality result for each of the layers G, allowing us to obtain a
general universality result for the entire network. The statement of the theorem requires the careful
construction of a neural operator-representable function Φ. Giving a full description of Φ involves
introducing a lot of technical notation, and so the presentation here in the main text leaves the details
of the construction of Φ abridged. For the full statement of the theorem and definition of the notation,
see Section A.7.2. Intuitively, Φ is the ‘wrapping’ of a fixed-point process in a neural operator.
Theorem 5. Let R > 0, and let F : X →X be a layer of a bilipschitz neural operator, as in
Defintion 3. Let σ be the Rectified Cubic Unit (ReCU) function defined by σ(x) := max{0, x}3.
Then, for any ϵ ∈(0, 1), there are T, N ∈N and G ∈RT,N,φ,σ(X) that has the form

G = (IdX + DN ◦NNT ◦EN) ◦· · · ◦(IdX + DN ◦NN1 ◦EN),
such that each map (IdX + DN ◦NNt ◦EN) is strongly monotone C1-diffeomorphisms on some
ball and
sup
x∈BX(0,R)
∥F(x) −G ◦A(x)∥X ≤ϵ,

where A: X →X is a linear invertible map that is either the identity map or a reflection operator
x →x −2⟨x, e⟩Xe with some unit vector e ∈X. Further, G ◦A : BX(0, R) →G ◦A(BX(0, R))
is invertible, and there is some neural operator Φ : G ◦A(BX(0, R)) →A(BX(0, R)) so that
 
G ◦A|BX(0,R)
−1 = A−1 ◦Φ.

The proof is given in Section A.7.2. Neural operator Φ becomes a better approximation of the inverse
operator when it becomes deeper. Theorem 5 means that neural operators are an operator algebra of
nonlinear operators that are closed in composition and when the inverse of a neural operator exists,
the inverse operator can be locally approximated by neural operators.

In the case when the separable Hilbert space X is the real-valued L2-function space L2(D; R),
residual networks in the separable Hilbert space can be represented as residual neural operators
defined by (179). See Lemma 11 for details. Then, we obtain the following

9


---Page Break---
Corollary 1. Let D ⊂Rd be a bounded domain, and let φ = {φn}n∈N be an orthonor-
mal basis in L2(D; R).
Assume that the orthonormal basis φ include the constant func-
tion. Let RNOT,N,φ,σ(L2(D; R)) be the class of residual neural operators defined in (179).
Then, the statement replacing X with L2(D; R) and G
∈
RT,N,φ,ReLU(X) with G
∈
RNOT,N,φ,ReLU(L2(D; R)) in Theorem 5 holds.

The proof is given by a combination of Theorem 5 and Lemma 11. We note that the assumption that
the orthonormal basis φ includes the constant function is satisfied if we choose φ to be a Fourier
basis, which yields the Fourier neural operator, see e.g., [39, 38].

Remark 1. In this section, we have shown that residual networks in a separable Hilbert space X,
defined in (10), are universal approximators for layers of bilipschitz neural operators. Additionally,
in the specific case where X = L2(D; R), residual neural operators, defined in Definition 9, also
provide universal approximators for layers of bilipschitz neural operators. We note that the residual
network we have discussed is locally invertible but not globally. By introducing invertible residual
networks on Hilbert space X, defined in (166), we can similarly prove that these networks by
employing sort activation functions (see [3, Section 4]) are universal approximators for strongly
monotone diffeomorphisms with compact support. Specifically, when X = L2(D; R), invertible
residual neural operators, defined in Definition 9, are also universal approximators for strongly
monotone diffeomorphisms with compact support. For further details, we refer to Appendix B.

4
Quantitative approximation

Quantitative approximation results for neural networks, see e.g. [57] or [23], can be used to derive
quantitative error estimates for discretization operations. Let F : BX(0, r) →X be a non-linear
function satisfying F ∈Lip(BX(0, r); X), in n = 1, or F ∈Cn(BX(0, r); X), if n ≥2. Then, F
can be discretized using neural networks in the following way: Let εV > 0 be numbers indexed by
the linear subspaces V ⊂X such that εV →0 as V →X. When ⃗ε = (εV )V ∈S(X), in the sense
of Definition 1, an ⃗ε-approximation operation ANN : F →(FV )V ∈S(X) in the ball BX(0, r) can
be be obtained by defining FV = J−1
V
◦FV,θ ◦JV : V →V , where JV : V →Rd is an isometric
isomorphism, d = dim(V ), FV,θ : Rd →Rd is a feed-forward neural network with ReLU-activation
functions with at most C(d) log2((1+r)/εV ) layers and C(d)εd
V log2((1+r)/εV ) non-zero elements
in the weight matrices. Details of this result are given in Proposition 5 in Appendix A.8.

5
Conclusion

In this work, we have studied the problem of discretizing neural operators between Hilbert spaces.
Many physical models concern functions Rn →R, for example L2(Rn), the computational methods
based on approximations in finite dimensional spaces should become better when the dimension of
the model grows and tends to infinity. We have focused on diffeomorphisms in infinite dimensions,
which are crucial to understand in generative modeling. We have shown that the approximation of
diffeomorphisms leads to computational difficulties. We used tools from category theory to produce
a no-go theorem showing that general diffeomorphisms between Hilbert spaces may not admit
any continuous approximations by diffeomorphisms on finite spaces, even if the approximations
are allowed to be nonlinear. We then proceeded to give several positive results, showing that
diffeomorphisms between Hilbert spaces may be continuously approximated if they are further
assumed to be strongly monotone. Moreover, we showed that the difficulties can be avoided by
considering a restricted but still practically rich class of diffeomorphisms. This includes bilipschitz
neural operators, which may be represented in any bounded set as a composition of strongly monotone
neural operators and strongly monotone diffeomorphisms. We then showed that such operators may
be inverted locally via an iteration scheme. Finally we gave a simple example on how quantitative
stability questions can be obtained for discretization functors, inviting other researchers to study
related questions using more sophisticated methods.

10


---Page Break---
Acknowledgments

TF was supported by JSPS KAKENHI Grant Number JP24K16949, JST CREST JPMJCR24Q5,
JST ASPIRE JPMJAP2329, and Grant for Basic Science Research Projects from The Sumitomo
Foundation. MP was supported by CAPITAL Services of Sioux Falls, South Dakota and NSF-DMS
under grant 3F5083. MVdH was supported by the Simons Foundation under the MATH + X program,
the Department of Energy, BES under grant DE-SC0020345, and the corporate members of the
Geo-Mathematical Imaging Group at Rice University. A significant part of the work of MVdH was
carried out while he was an invited professor at the Centre Sciences des Données at Ecole Normale
Supérieure, Paris. M.L. was partially supported by a AdG project 101097198 of the European
Research Council, Centre of Excellence of Research Council of Finland and the FAME flagship.
Views and opinions expressed are those of the authors only and do not necessarily reflect those of the
funding agencies or the EU.

References

[1] Ahmed Abdeljawad and Philipp Grohs. Approximations with deep neural networks in sobolev
time-space. Analysis and Applications, 20(03):499–541, 2022.

[2] Giovanni S. Alberti, Matteo Santacesaria, and Silvia Sciutto. Continuous generative neural
networks. arXiv:2205.14627, 2022.

[3] Cem Anil, James Lucas, and Roger Grosse. Sorting out lipschitz function approximation. In
International Conference on Machine Learning, pages 291–301. PMLR, 2019.

[4] Ivo Babuška and Manil Suri. On locking and robustness in the finite element method. SIAM J.
Numer. Anal., 29(5):1261–1293, 1992.

[5] Heinz H. Bauschke and Patrick L. Combettes. Convex analysis and monotone operator theory in
Hilbert spaces. CMS Books in Mathematics/Ouvrages de Mathématiques de la SMC. Springer,
Cham, second edition, 2017. With a foreword by Hédy Attouch.

[6] Roberto Bentivoglio, Elvin Isufi, Sebastian Nicolaas Jonkman, and Riccardo Taormina. Deep
learning methods for flood mapping: a review of existing applications and future research
directions. Hydrology and earth system sciences, 26(16):4345–4378, 2022.

[7] Kaushik Bhattacharya, Bamdad Hosseini, Nikola B Kovachki, and Andrew M Stuart. Model
reduction and neural networks for parametric pdes.
The SMAI journal of computational
mathematics, 7:121–157, 2021.

[8] Klaus Böhmer. Numerical methods for nonlinear elliptic differential equations. Numerical
Mathematics and Scientific Computation. Oxford University Press, Oxford, 2010. A synopsis.

[9] Klaus Böhmer and Robert Schaback. A nonlinear discretization theory. J. Comput. Appl. Math.,
254:204–219, 2013.

[10] Philippe G. Ciarlet. Linear and Nonlinear Functional Analysis with Applications. Society for
Industrial and Applied Mathematics, USA, 2013.

[11] Masoumeh Dashti, Stephen Harris, and Andrew Stuart. Besov priors for Bayesian inverse
problems. Inverse Probl. Imaging, 6(2):183–200, 2012.

[12] Maarten De Hoop, Daniel Zhengyu Huang, Elizabeth Qian, and Andrew M Stuart. The cost-
accuracy trade-off in operator learning with neural networks. arXiv preprint arXiv:2203.13181,
2022.

[13] Dennis Elbrächter, Dmytro Perekrestenko, Philipp Grohs, and Helmut Bölcskei. Deep neural
network approximation theory. IEEE Transactions on Information Theory, 67(5):2581–2623,
2021.

[14] Takashi Furuya, Michael Puthawala, Matti Lassas, and Maarten V de Hoop. Globally injective
and bijective neural operators. arXiv preprint arXiv:2306.03982, 2023.

11


---Page Break---
[15] Jean Gallier. Geometric methods and applications, volume 38 of Texts in Applied Mathematics.
Springer-Verlag, New York, 2001. For computer science and engineering.

[16] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional
neural networks. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 2414–2423, 2016.

[17] Bruno Gavranovi´c, Paul Lessard, Andrew Dudzik, Tamara von Glehn, João GM Araújo, and
Petar Veliˇckovi´c. Categorical deep learning: An algebraic theory of architectures. arXiv preprint
arXiv:2402.15332, 2024.

[18] David Gilbarg and Neil S. Trudinger. Elliptic partial differential equations of second order.
Classics in Mathematics. Springer-Verlag, Berlin, 2001. Reprint of the 1998 edition.

[19] Evarist Giné and Richard Nickl. Mathematical foundations of infinite-dimensional statistical
models. Cambridge university press, 2021.

[20] Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible resid-
ual network: Backpropagation without storing activations. Advances in neural information
processing systems, 30, 2017.

[21] Somdatta Goswami, Aniruddha Bora, Yue Yu, and George Em Karniadakis. Physics-informed
deep neural operator networks. In Machine Learning in Modeling and Simulation: Methods
and Applications, pages 219–254. Springer, 2023.

[22] Ingo Gühring, Gitta Kutyniok, and Philipp Petersen. Error bounds for approximations with
deep relu neural networks in w s, p norms. Analysis and Applications, 18(05):803–859, 2020.

[23] Ingo Gühring, Gitta Kutyniok, and Philipp Petersen. Error bounds for approximations with
deep ReLU neural networks in W s,p norms. Anal. Appl. (Singap.), 18(5):803–859, 2020.

[24] E. M. Harrell and W. J. Layton. L2 estimates for Galerkin methods for semilinear elliptic
equations. SIAM J. Numer. Anal., 24(1):52–58, 1987.

[25] Isao Ishikawa, Takeshi Teshima, Koichi Tojo, Kenta Oono, Masahiro Ikeda, and Masashi
Sugiyama. Universal approximation property of invertible neural networks. Journal of Machine
Learning Research, 24(287):1–68, 2023.

[26] Thomas Jech. Set theory. Perspectives in Mathematical Logic. Springer-Verlag, Berlin, second
edition, 1997.

[27] Jari Kaipio and Erkki Somersalo. Statistical and computational inverse problems, volume 160
of Applied Mathematical Sciences. Springer-Verlag, New York, 2005.

[28] R. I. Kaˇcurovski˘ı. Nonlinear monotone operators in Banach spaces. Uspehi Mat. Nauk,
23(2(140)):121–168, 1968.

[29] Gavin Kerrigan, Justin Ley, and Padhraic Smyth. Diffusion generative models in infinite
dimensions. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings
of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of
Proceedings of Machine Learning Research, pages 9538–9563. PMLR, 25–27 Apr 2023.

[30] Nikola B Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya,
Andrew M Stuart, and Anima Anandkumar. Neural operator: Learning maps between function
spaces with applications to pdes. J. Mach. Learn. Res., 24(89):1–97, 2023.

[31] Anastasis Kratsios and Ievgen Bilokopytov. Non-euclidean universal approximation. Advances
in Neural Information Processing Systems, 33:10635–10646, 2020.

[32] Nicolaas H Kuiper. The homotopy type of the unitary group of hilbert space. Topology,
3(1):19–30, 1965.

[33] Samuel Lanthaler, Zongyi Li, and Andrew M Stuart. The nonlocal neural operator: Universal
approximation. arXiv preprint arXiv:2304.13221, 2023.

12


---Page Break---
[34] Samuel Lanthaler, Siddhartha Mishra, and George E Karniadakis. Error estimates for deeponets:
A deep learning framework in infinite dimensions.
Transactions of Mathematics and Its
Applications, 6(1):tnac001, 2022.

[35] Samuel Lanthaler, Andrew M Stuart, and Margaret Trautner. Discretization error of fourier
neural operators. arXiv preprint arXiv:2405.02221, 2024.

[36] Matti Lassas and Samuli Siltanen. Can one use total variation prior for edge-preserving bayesian
inversion? Inverse problems, 20(5):1537, 2004.

[37] Markku S Lehtinen, Lassi Paivarinta, and Erkki Somersalo. Linear inverse problems for
generalised random variables. Inverse Problems, 5(4):599, 1989.

[38] Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier neural
operator with learned deformations for pdes on general geometries. Journal of Machine
Learning Research, 24(388):1–26, 2023.

[39] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya,
Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differen-
tial equations. arXiv preprint arXiv:2010.08895, 2020.

[40] Lu Lu, Pengzhan Jin, and George Em Karniadakis. Deeponet: Learning nonlinear operators for
identifying differential equations based on the universal approximation theorem of operators.
arXiv preprint arXiv:1910.03193, 2019.

[41] Robert J. Martin and Patrizio Neff. Minimal geodesics on gl(n) for left-invariant, right-o(n)-
invariant riemannian metrics. Journal of Geometric Mechanics, 8(3):323–357, 2016.

[42] Peter W. Michor and David Mumford. A zoo of diffeomorphism groups on Rn. Ann. Global
Anal. Geom., 44(4):529–540, 2013.

[43] Donal O’Regan, Yeol Je Cho, and Yu-Qing Chen. Topological degree theory and applications,
volume 10 of Series in Mathematical Analysis and Applications. Chapman & Hall/CRC, Boca
Raton, FL, 2006.

[44] Michael Puthawala, Konik Kothari, Matti Lassas, Ivan Dokmani´c, and Maarten de Hoop.
Globally injective relu networks. Journal of Machine Learning Research, 23(105):1–55, 2022.

[45] Calvin R. Putnam and Aurel Wintner. The connectedness of the orthogonal group in Hilbert
space. Proc. Nat. Acad. Sci. U.S.A., 37:110–112, 1951.

[46] Calvin R. Putnam and Aurel Wintner. The orthogonal group in Hilbert space. Amer. J. Math.,
74:52–78, 1952.

[47] R. Tyrrell Rockafellar and Roger J.-B. Wets. Variational analysis, volume 317 of Grundlehren
der mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences].
Springer-Verlag, Berlin, 1998.

[48] Matti Lassas Saksman, Samuli Siltanen, et al. Discretization-invariant bayesian inversion and
besov space priors. arXiv preprint arXiv:0901.4220, 2009.

[49] Martin H. Schultz. Error bounds for the Rayleigh-Ritz-Galerkin method. J. Math. Anal. Appl.,
27:524–533, 1969.

[50] R. E. Showalter. Monotone operators in Banach space and nonlinear partial differential
equations, volume 49 of Mathematical Surveys and Monographs. American Mathematical
Society, Providence, RI, 1997.

[51] Andrew M Stuart. Inverse problems: a bayesian perspective. Acta numerica, 19:451–559, 2010.

[52] Manil Suri, Ivo Babuška, and Christoph Schwab. Locking effects in the finite element approxi-
mation of plate models. Math. Comp., 64(210):461–482, 1995.

[53] Michael E. Taylor. Partial differential equations I. Basic theory, volume 115 of Applied
Mathematical Sciences. Springer, New York, second edition, 2011.

13


---Page Break---
[54] Takeshi Teshima, Isao Ishikawa, Koichi Tojo, Kenta Oono, Masahiro Ikeda, and Masashi
Sugiyama. Coupling-based invertible neural networks are universal diffeomorphism approxima-
tors. Advances in Neural Information Processing Systems, 33:3362–3373, 2020.

[55] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal
Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the
age of artificial intelligence. Nature, 620(7972):47–60, 2023.

[56] Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, and Vipin Kumar. Integrating
scientific knowledge with machine learning for engineering and environmental systems. ACM
Computing Surveys, 55(4):1–37, 2022.

[57] Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural Networks,
94:103–114, 2017.

14


---Page Break---
A
Proofs and additional examples

A.1
Non-linear discretization

In this section we consider the non-linear discretization theory, see [9]. Let Ω⊂Rn be a bounded
open set with smooth boundary. Let H1(Ω) = {u ∈L2(Ω) | ∇u ∈L2(Ω)} and H1
0(Ω) = {u ∈
H1(Ω) | u|∂Ω= 0} be the Sobolev spaces with an integer order of smoothness and Hs(Ω), s ∈R
the Sobolev space with a fractional order of smoothness [53].

For simplicity, we assume in the section that n = 1, and that Ω⊂R is an interval and we point
out that in this case the Sobolev embedding theorem implies H1(Ω) ⊂C(Ω) which significantly
simplifies the constructions below.

Let g ∈Cm+1(R; R), m ≥1, be the derivative of a convex function G ∈Cm+2(R; R) satisfying

−c0 ≤G(r) ≤c1(1 + rp), r ∈R, c0, c1, p > 0.
(11)

Let F = F (g) : H1
0(Ω) →H1
0(Ω), F (g) : x →u be the solution operator of the non-linear differential
equation

∆u(t) −g(u(t)) = x(t),
t ∈Ω,
(12)
u|∂Ω= 0,
(13)

where ∆is the Laplace operator operating in the t variable, where the function x(t) is a physical
source term. Because the u that solves (12)-(13) is the minimizer of the strictly convex and lower
semi-continuous function H : X →R,

H(v) = ∥∇v∥2
L2(Ω)2 +
Z

Ω
(G(v(t)) −x(t)v(t))dt,
(14)

the Weierstrass theorem, see [50], Theorem II.8.1, implies that (12)-(13) has a unique solution.
Moreover, by regularity theory for elliptic equations, see [18], Theorem 8.13, we have that the
solution u is in H3(Ω). Moreover, as G is convex and satisfies (11), the function H in (14) is
coercive, for any R > 0 there is ρ(R) > 0 such that when ∥x∥H1
0(Ω) ≤R, then ∥u∥H3(Ω) ≤ρ(R).

The equation (12)-(13) can be written also as an integral equation

u(t) +
Z

Ω
Ψ(t, t′)g(u(t′))dt′ = −
Z

Ω
Ψ(t, t′)x(t′)dt′,
t ∈Ω,
(15)

where Ψ(t, t′) is the Dirichlet Green’s function of the negative Laplacian, that is, −∆Ψ(·, t′) = δt′(·),
Ψ(·, t′)|∂Ω= 0, and g∗(u) = g(u). We can write (15) as

(Id + Q ◦g∗)u = −Q(x),
Qu(t) =
Z

Ω
Ψ(t, t′)x(t′)dt′,
(16)

where iH1
0→H3/4 : H1
0(Ω) →H3/4(Ω) and iH2→H1
0 : H2(Ω) ∩H1
0(Ω) →H1
0(Ω) are compact
operators and as H3/4(Ω) ⊂C(Ω), the operators g∗: H3/4(Ω) →L2(Ω) and Q : L2(Ω) →
H2(Ω) ∩H1
0(Ω) are continuous, we see that

F = Id + iH2→H1 ◦(Q ◦g∗) ◦iH1→H3/4 : H1
0(Ω) →H1
0(Ω),

is a layer of the neural operator. As G ∈Cm+1, m ≥2, we have that F : H1
0(Ω) →H1
0(Ω) is
C1-smooth. Moreover, as G is convex, the operator F : H1(Ω) →H1(Ω) is strongly monotone and
hence by Lemma 3, F has an inverse map F −1 : H1(Ω) →H1(Ω) that is C1-smooth. Thus, the
solution u of (12)-(13) can be represented as

u = F −1(Q(x)).

Let V ⊂X = H1
0(Ω) be a finite dimensional space. The Galerkin methods to obtain an approxima-
tion for the solution of the boundary value problem (12)-(13) using Finite Element Method. To do
this, let w ∈V be the solution of the problem
Z

Ω
(∇w(t) · ∇ϕ(t) + g(w(t))ϕ(t))dt = −
Z

Ω
x(t) · ϕ(t)dt,
for all ϕ ∈V.
(17)

15


---Page Break---
When ϕ1, . . . , ϕd ∈V are a basis of the finite dimensional vector space V , and we write w =
Pd
k=1 wkϕj, the problem (17) is equivalent that (w1, . . . , wd) ∈Rd satisfies the system equations d
equations

d
X

k=1
bjkwk = −
Z

Ω
ϕj(t)g

d
X

k=1
wkϕk(t)

dt −
Z

Ω
x(t) · ϕj(t)dt,
for j = 1, 2, . . . , d,
(18)

where
bjk =
Z

Ω
∇ϕj(t) · ∇ϕk(t)dt.

When x ∈V , we define the map F (g)
V
: V →V by setting map F (g)
V (x) = w, where w ∈V is the
solution of the problem (17). When

FP DE = {F (g) : X →X | g(s) = dG

ds (s) , G ∈Cm+2(R) is convex and satisfies (11)},

we define the map AF EM : F (g) →F (g)
V
for F (g) ∈FP DE. As the the function g in the equation
(18) is non-linear, the solution u of the continuum problem (15)-(13) and the solution w of the finite
dimensional problem have typically no linear relationship, even if the source x satisfies x ∈V.

Let R > 0 As the embedding H2(Ω) ∩H1
0(Ω) →H1
0(Ω) is compact and Q : H1
0(Ω) →H2(Ω) ∩
H1
0(Ω) is continuous, we see that the image of the closed ball BX(0, R) in the map Q is a precompact
set Q(BX(0, R)) ⊂X. As F −1 : X →X is continuous the set of corresponding solutions,

ZR = {u ∈X | u = F −1(Q(x)), x ∈BX(0, R)},

is precompact in X. Hence,

lim
V →X sup
u∈ZR
∥(Id −PV )(u)∥X = 0.
(19)

As G is convex, the Fréchet derivative of the map F = Id + Q ◦g∗: X →X is a strictly positive
linear operator at all points x ∈X. This, the uniform convergence (19), and the convergence results
for the Galerkin method for semi-linear equations, [49, Theorem 3.2], see also [24], imply that in the
space X = H1
0(Ω)

lim
V →X
sup
∥x∥X≤R
∥F (g)
V (x) −F (g)(x)∥X = 0.
(20)

These imply that AF EM : F (g) →F (g)
V
is an approximation operation for the function F (g) ∈FP DE.

The map F (g)
V
: V →V is called the Galerkin approximation of the problem (12)-(13) and is an
example of the non-linear approximation methods. The properties of the Galerkin approximation is
studied in detail in [8], in particular sections 4.4 and 4.5.

A.2
Negative results

On the positive results, in Appendix A.1, we have considered nonlinear discretiation of the operators
u →−∆u + g(u). To exemplify the negative result, we consider the following example on the
solution operation of differential equations and the non-existence of approximation by diffeomorphic
maps: We consider the elliptic (but not not strongly elliptic) problem

Bsu := −d

dt


(1 + t)sign(t −s) d

dtu(t)

= f(t),
t ∈[0, 1],
(21)

with the Dirichlet and Neumann boundary conditions

u(0) = 0,
d
dtu(t)

t=1
= 0.
(22)

Here, 0 ≤s ≤1 is a parameter of the coeffient function and sign(t −s) = 1 if t > s and
sign(t −s) < 0 if t < s. We consider the weak solutions of (21) in the space

u ∈H1
D,N(0, 1) := {v ∈H1(0, 1) : v(0) = 0}.

16


---Page Break---
We can write
Bsu = −D(2)
t AsD(1)
t u,

where
Asv(t) = (1 + t)sign(t −s)v(t),

parametrized by 0 ≤s ≤, are multiplication operations that are invertible operators, As : L2(0, 1) →
L2(0, 1) (this invertibility makes the equation (21) elliptic). Moreover, D(1)
t
and D(2)
t
are the
operators v →d

dtv with the Dirichlet boundary condition v(0) = 0 and v(1) = 0, respectively. We
consider the Hilbert space X = H1
D,N(0, 1); to generate an invertible operator Gs : X →X related
to (21), we write the source term using an auxiliary function g,

f(t) = Qg := −d2

dt2 g(t) + g(t).

Then the equation,

Bsu = Qg,
(23)

defines a continuous and invertible operator,

Gs : X →X,
Gs : g →u.

In fact, Gs = B−1
s
◦Q when the domains of Bs and Q are chosen in a suitable way. The Galerkin
method (that is, the standard approximation based on the Finite Element Method) to approximate the
equation (23) involves introducing a complete basis χj(t), j = 1, 2, . . . of the Hilbert space X, the
orthogonal projection

Pn : X →Xn := span{χj : j = 1, 2, . . . , n},

and approximate solutions of (23) through solving

PnBsPnun = PnQPngn,
un ∈Xn, gn = Png.
(24)

This means that operator B−1
s Q : g →u is approximated by (PnBsPn)−1PnQPn : gn →un, when
PnBsPn : Xn →Xn is invertible.

The above corresponds to the Finite Element Method where the matrix defined by the operator
PnBsPn is b(s) = [bjk(s)]n
j,k=1 ∈Rn×n, where

bjk(s) =
Z 1

0
(1 + t)sign(t −s) d

dtχj(t) · d

dtχk(t)dt,
j, k = 1, . . . , n.

Since we used the mixed Dirichlet and Neumann boundary conditions in the above boundary value
problem, we see that for s = 0 all eigenvalues of the matrix b(s) are strictly positive, and when
s = 1 all eigenvalues are strictly negative. As the function s →b(s) is a continuous matrix-valued
function, we see that there exists s ∈(0, 1) such that the matrix b(s) has a zero eigenvalue and is no
invertible. Thus, we have a situation where all operators B−1
s Q : g →u, s ∈[0, 1] are invertible (and
thus define diffeomorphisms X →X) but for any basis χj(t) and any n there exists s ∈(0, 1) such
that the finite dimensional approximation b(s) : Rn →Rn is not invertible. This example shows
that there is no FEM-based discretization method for which the finite dimensional approximations
of all operators B−1
s Q, s ∈(0, 1), are invertible. The above example also shows a key difference
between finite and infinite dimensional spaces. The operator As : L2(0, 1) →L2(0, 1) has only
continuous spectrum and not eigenvalues nor eigenfunctions whereas the finite dimensional matrices
have only point spectrum (that is, eigenvalues). The continuous spectrum makes it possible to deform
the positive operator As with s = 0 to a negative operator As with s = 1 in such a way that all
operators As, 0 ≤s ≤1, are invertible but this is not possible to do for finite dimensional matrices.
We point out that the map s →As is not continuous in the operator norm topology but only in the
strong operator topology and the fact that A0 can be deformed to A1 in the norm topology by a path
that lies in the set of invertible operators is a deeper result. However, the strong operator topology is
enough to make the FEM matrix b(s) to depend continuously on s.

17


---Page Break---
A.3
Application of Theorem 2 to Neural Operators

In this section, we give an example of the application of Theorem 2, as applied to the problem of
using a neural operator to solve a linear PDE. The guiding idea is to draw inspiration from Figure 1;
the space of diffeomorphisms on X is disconnected if X is a finite-dimensional space, but connected
if it is a Hilbert space.

Before constructing the path, we first construct a isotopy of diffeomorphisms in L2(Ω) in several
steps

Let ρ: R →[0, 1] be a smooth function such that ρ|(−∞,0] = 0, ρ|[1,∞) = 1. Let R(θ) ∈R2×2 be
the rotation matrix given by

R(θ) :=

cos(θ)
sin(θ)
−sin(θ)
cos(θ)


.
(25)

Then, define the function Ri : R →R2×2 by

Ri(t) := R(ρ(t −i)π).
(26)

Note that Ri(t) ∈SO(2), for t ≤i, Ri(t) = I, t ≥i + 1, Ri(t) = −I.

Next, we consider the ‘little ell’ space, ℓ2 :=

(α1, α2, . . . , ): P∞
i=1 α2
i < ∞
	
. Consider the linear
operator ˆR: R →
 
ℓ2 →ℓ2
on ℓ2 given by a pair-wise coordinate representation given by the
infinite dimensional ‘matrix’ with block structure

ˆR(t) :=






R1(t)
R2(t)
...




.
(27)

Finally, define R: [0, 1] →(ℓ2 →ℓ2) by

R(t) :=

(
ˆR

1
1−t

t ∈[0, 1)

−I
t ≥1
,
(28)

and define the related operator ˜R: [0, 1] →
 
ℓ2 →ℓ2
by

˜R(t) =

−1
R(t)


.
(29)

Proposition 3 (R, ˜R are isotopies of smooth diffeomorphisms in ℓ2). The maps H1 : ℓ2 ×[0, 1] →ℓ2
and H2 : ℓ2 × [0, 1] →ℓ2 defined by

H1(v, t) = R(t)v,
H2(v, t) = ˜R(t)v,
(30)

are isotopies of smooth diffeomorphisms in ℓ2 and agree at t = 1.

Proof. We prove that H1 is an isotopy of smooth diffeomorphisms on ℓ2, as the proof of the same for
H2 is very similar.

We first show that H1(·, t) ∈diff(ℓ2) for each t.

For t < 1, R operates on vectors in ℓ2 in only a finite number of indices. In those indices it
corresponds to a rotation, and hence is a smooth diffeomorphism. When t ≥1, it corresponds to
scalar multiplication by −1, and so is too a smooth diffeomorphism. Now we show that R(t) is
continuous on . The only point where continuity may fail is at t = 1, and there it may only fail in the

18


---Page Break---
limit from the left. Given a v ∈ℓ2, we compute

lim
t→1−∥R(t)v −R(1)v∥2 = lim
t→1−























R1(
1
1−t)

v1
v2



R2(
1
1−t)

v3
v4



...

Rk(
1
1−t)

v2k−1
v2k



Rk+1(
1
1−t)

v2k+1
v2k+2



...





















+



















v1
v2
v3
v4
...
v2k−1
v2k
v2k+1
v2k+2
...




















2

.
(31)

Let k∗(t) := ⌊
1
1−t⌋. Then for any integer i ≤k, we have that

Ri


1
1 −t


= R





ρ






1
1 −t −i
|
{z
}
≥1





π





= R(π) = −I ∈R2×2,
(32)

and likewise when i ≥k∗(t) + 1, then Ri

1
1−t

= I ∈R2×2, and so the r.h.s. of Eqn. 31 becomes

lim
t→1−






















−v1
−v2
−v3
−v4
...

Rk∗(t)(
1
1−t)

v2k∗(t)−1
v2k∗(t)



v2k∗(t)+1
v2k∗(t)+2
...




















+




















v1
v2
v3
v4
...
v2k∗(t)−1
v2k∗(t)
v2k∗(t)+1
v2k∗(t)+2
...





















2

(33)

= lim
t→1−






















0
0
0
0
...

Rk∗(t)(
1
1−t)

v2k∗(t)−1
v2k∗(t)


+

v2k∗(t)−1
v2k∗(t)+2



2v2k∗(t)+1
2v2k∗(t)+2
...





















2

(34)

≤2

v
u
u
t

∞
X

i=k∗(t)
v2
i .
(35)

As t →1−, k∗(t) →∞, and so, by standard estimates, 2
qP∞
i=k∗(t) v2
i →0. Hence, R(t) is

continuous on ℓ2 at t = 1.

Finally, by definition at t = 1, we have that H1(v, 1) = R(1) = −I = R(1) = H2(v, 1).

Finally, we define H : R∞× [0, 1] →R∞by gluing the isotopies H1 and H2 together by

H(v, t) :=
 H1(v, 2t)
if t ≤1

2
H2(v, 2 −2t)
if t > 1

2
.
(36)

19


---Page Break---
Proposition 4. The function H given by Eqn. 36 is an isotopy of diffeomorphisms in ℓ2.

Proof. Continuity of H follows from continuity of R, ˜R, and that R(1) = ˜R(1). For each t ∈[0, 1],
H(·, t) ∈diff(ℓ2).

A.4
Proofs from Sec. 3.1

A.4.1
Proof of Theorem 2

Proof. Assume that A: D →B is a functor that satisfies the property (A) of an approximation
functor and is continuous. Let us consider the case when X = ℓ2.

Let e ∈X be a unit vector and Be : X →X be a linear map Bx = x −2⟨x, e⟩Xe. In other words,
B is a diagonal matrix diag(−1, 1, . . . ) in some orthogonal basis where e is the first basis vector.

By [32, Theorem 2], see also and [45], for an infinite dimensional Hilbert space X the space GL(X)
of linear invertible maps have only one topological component in the operator norm topology and
the set GL(X) is contractible and hence path-connected. This implies that there are linear maps
At : X →X, t ∈[0, 1] such that A0 = Id, and A1 ∈GL(X) is arbitrary invertible linear map
and that t →At ∈GL(X) is continuous. Similarly, in an infinite dimensional real Hilbert space X
the space OR(X) of the orthogonal operators A : X →X is path-connected in the operator norm
topology, see [46, Section 4]. We recall that an orthogonal operator A : X →X is a bounded linear
operator satisfying A∗A = AA∗= I, where A∗: X →X is the adjoint (i.e., the transpose) of the
operator A : X →X. Observe that in a finite dimensional space Rn the set OR(Rn) has two disjoint
topological components, those which determinant is 1 and those which determinant is −1, and thus
the set OR(Rn) is not connected.

A bounded linear operator B : X →X is called a rotation operator of the form B = eS where
S : X →X is a linear, bounded, skew-symmetric operator, S∗= −S, and eS = I + S + 1

2!S2 +
1
3!S3 + . . . . An orthogonal operator A is called a reflection if it is not a rotation.

One of the fundamental reasons for the surprising property that the space OR(X) of orthogonal
operators in X is path-connected is that in an infinite dimensional real Hilbert space X every
orthogonal operator A : X →X can be represented as a product of two rotation operators (that
is no valid in the finite dimensional spaces), that is, A = eS1eS2 where S1, S2 : X →X are
skew-symmetric linear operators. Thus any operator A ∈OR(X) can be connected to the identity
operator Id : X →X via a path α : t →etS1etS2 ∈OR(X), t ∈[0, 1], so that α(0) = Id and
α(1) = A.

As a motivating example on the fact that OR(X) is connected in the operator norm topology, is to
consider a similar, but easier result in the strong operator topology, that is, topology generated by
the evaluation maps A →A(u), u ∈L2(0, 1). Observe that the strong operator topology is weaker
than the operator norm topology, but in finite dimensional space those are equivalent.. So, let us next
show that the operators Id and −Id are in the same topological component of OR(X) in the strong
operator topology. As all infinite dimensional separable Hilbert spaces are isometric to L2(0, 1), let
us consider the Hilbert space L2(R) and the orthogonal linear operators As : L2(0, 1) →L2(0, 1),
s ∈[0, 1] that are given by

Asu(t) = sign(t −s) · u(t),
(37)

where u ∈L2(0, 1) and sign(t) is the sign of the real number t. Then, for every u ∈L2(0, 1) the
functions

au : s →As(u),

are continuous functions au : [0, 1] →L2(0, 1). As A0 = Id and A0 = −Id, we see that

s →As ∈OR(L2(0, 1)),
s ∈[0, 1],

is a continuous path in the strong operator topology, that connects the operator Id to −Id.

We recall that the continuity of the function s →As in the strong operator topology, means that for
all u ∈L2(0, 1) the map
s →Asu ∈L2(0, 1),

20


---Page Break---
is continuous. As for s > s0

∥Asu −As0u∥2
L2(0,1) =
Z s

s0
|2u(x)|2dx →0,

as s →s0, we see that s →As is continuous in the strong operator topology. All maps As :
L2(0, 1) →L2(0, 1) are invertible and A0 = Id, A1 = −Id. This has implications e.g. for Finite
Element method analysis of partial differential equations (See Appendix A.5.2).

Note that when X is any infinite dimensional separable Hilbert space with real scalars, there is
an isometry J : X →L2(0, 1) (e.g., the linear operator mapping an orthogonal basis of X to an
orthogonal basis L2(0, 1). Then, the maps ˜As : X →X given by ˜As = J−1 ◦As ◦J : X →X,
where As : L2(0, 1) →L2(0, 1) are given above, are continuous paths in OR(X) from ˜A0 = Id :
X →X to ˜A1 = −Id : X →X. As discussed above, the deep result that Id and −Id can be
connected by a continuous path in the operator norm topology of OR(X) is given in [32, Theorem 2],
see also [45].

Let us first warm up by proving the claim in the case when an additional assumption (B) is valid:

(B) : When V ⊂X is finite dimensional discretization maps FV of linear invertible maps F : X →X
are linear invertible maps and moreover, the set S0(X) = S(X) consists of all linear subspaces of X.

Under assumptions (A) and (B), we consider the case when A1 = −Id and denote by Ft = At :
X →X a family of linear maps such that A0 = Id, and that t →At ∈GL(X) is continuous
path of operators connecting Id to A1 = −Id. Let (X, S(X), (Ft,V )V ∈S(X)) = A(X, Ft), that is,
Ft,V : V →V be the linear isomorphism that is the discretizations of the map Ft : X →X. As the
functor A is continuous, the map t →Ft,V is a continuous map from [0, 1] to the space of linear
operators endowed with the topology of uniform convergence on compact sets, c.f. limit (7). As Ft,W
are linear, this implies that for all t′ ∈[0, 1],

lim
t→t′ ∥Ft,W −Ft′,W ∥= lim
t→t′
sup
x∈V ∩BX(0,1)
∥Ft,W (x) −Ft′,W (x)∥X = 0.
(38)

and hence the map t →Ft,W is a continuous map [0, 1] →GL(W).

Let 0 < ϵ0 < 1/2 and r > 1. Let t0 = 0 and t1 = 1. Using Property (A) for operators Ftj, j = 0, 1,
we see that there are W0, W1 ∈S0(X) such that for all V ∈S0(X) satisfying V ⊃Wj we have

sup
x∈BX(0,r)∩V
∥Ftj,Wj(x) −Ftj(x)∥X < ϵ0,
for j ∈{0, 1}.

Let W ∈S0(X) be such a finite dimensional space which dimension is an odd integer and that
W0 + W1 ⊂W. Then,

sup
x∈BX(0,r)∩W
∥Ftj,W (x) −Ftj(x)∥X ≤ϵ0,
for j ∈{0, 1},
(39)

Clearly, A0 = Id satisfies A0(W) = W and A0 : W →W is an invertible linear map. Similarly,
the map A1 = −Id satisfies A1(W) = W and A1 : W →W is an invertible linear map. These
observation and assumptions (A) and (B) imply that F0,W : W →W and F1,W : W →W are linear
maps and by inequality (39),

∥F0,W −A0|W ∥W →W < ϵ0 < 1

2,
∥F1,W −A1|W ∥W →W < ϵ0 < 1

2,
(40)

and as the dimension of W is odd, we have

det(A0|W ) = det(Id|W ) = 1,
det(A1|W ) = det(−Id|W ) = −1.

Let B0,s = (1 −s)A0|W + sF0,W : W →W and B1,s = (1 −s)A1|W + sF0,W : W →W. As
∥(A0|W )−1∥W →W = ∥(A1|W )−1∥W →W = 1, we see using (40) that the maps B−1
0,s : W →W and
B−1
1,s : W →W are invertible and that the functions s →B0,s ∈GL(W) and s →B1,s ∈GL(W)
are continuous matrix-valued functions which determinants does not vanish and

det(B0,s) = det(A0|W ) > 0,
det(B1,s) = det(A1|W ) < 0,

21


---Page Break---
for all s ∈[0, 1]. These imply that

det(F0,W ) = det(B0,1) > 0,
det(F1,W ) = det(B0,1) < 0,
(41)

However, as t →Ft,W is a continuous maps [0, 1] →GL(W), see (38), we have that

t →det(Ft,W ),
(42)

is a continuous function [0, 1] →R which does not obtain value zero and thus has a constant sign.
However, this is in contradiction with formula (41).

Next we return to the main part of the proof where we do not assume that assumption (B) is valid, but
we only assume that the assumption (A) is valid. In this case, we use degree theory instead of the
determinants.

Next, let Ft = At : X →X be a family of linear maps such that A0 = Id, but that A1(x) =
Be(x) = x −2e⟨x, e⟩X is a reflection, where e ∈X is a unit vector, and t →At ∈GL(X) is
continuous. Again, let (X, S0(X), (Ft,V )V ∈S0(X)) = A(X, Ft), that is, Ft,V : V →V be C1-
diffeomorphisms that are discretizations of the map F : X →X. As the functor A is continuous, the
map t →Ft,V |B(0,R) ∈C(B(0, R)) is a continuous map for all V ∈S0(X) and R > 0. Moreover,
we note that by our assumptions on B, the discretized map Ft,V : V →V , is C1-diffeomorphism of
V .

As V is a finite dimensional vector space, we can use finite dimensional degree theory and consider
the degree deg(Ft,V , V ∩BX(0, r), p), that is the degree of the map Ft,V : V ∩BX(0, r) →V at
the point p. We recall that when Ω⊂Rd is open and bounded, F : Ω→Rd is a C1-smooth function
and p ∈Rd are such that p ̸∈f(∂Ω) and for all x ∈f −1(p) the derivative Df(x) is an invertible
matrix, the degree is defined to be

deg(f, Ω, p) =
X

x∈f −1(p)
sgn(det(Df(x))),

where sgn(r) is sign of a real number r. Also, the map f →deg(h, Ω, p) is defined for a continuous
function h : Ω→Rd by approximating h by C1-smooth function, see [43], Definition 1.2.5 on
details. Let us denote BW (0, r) = W ∩BX(0, r).

Let r > 0. Recall that by assumption (A), for F0 = Id : X →X and F1 = Be : X →X there are
finite dimensional spaces V0 ∈S0(X) and V1 ∈S0(X) such that e ∈V0 and e ∈V1 and that when
V ∈S0(X) satisfies V0 ⊂V and V1 ⊂V , then

sup
x∈BX(0,r)∩V
∥F0,V (x) −F0(x)∥X < r/4,
(43)

and

sup
x∈BX(0,r)∩V
∥F1,V (x) −F1(x)∥X < r/4.
(44)

Moreover, let W ∈S0(X) be such a finite dimensional linear space that V0 ⊂W and V1 ⊂W.
Observe that as e ∈W, we can decompose W as

W = span(e) ⊕{x ∈W : ⟨x, e⟩X = 0}.
(45)

and denote W ′ = {x ∈W : ⟨x, e⟩X = 0}. We see that

A1 = Be : span(e) →span(e), Be|span(e) = −Id,
(46)

A1 = Be : W ′ →W ′, Be|W ′ = Id.

Next we use the facts that F0(0) = Id(0) = 0 and F1(0) = Be(0) = 0. Let us define the maps

f0 = F0|BW (0,r),
f1 = F1|BW (0,r),
f0,W = F0,W |BW (0,r),
f1,W = F1,W |BW (0,r).

that are C1-smooth maps BW (0, r) →W. Moreover, for j = 0, 1, we have f0(∂BW (0, r)) =
∂BW (0, r) and f1(∂BW (0, r)) = ∂BW (0, r). Let pj := fj,W (0). Then, by (43) and (44) we have

∥pj∥W = ∥fj,W (0)∥W = ∥fj,W (0) −fj(0)∥W < 1

4r,
(47)

22


---Page Break---
and
dist(fj,W (x), pj) ≥1

2r,
for all x ∈∂BW (0, r).

This implies that

sup
x∈BW (0,r)
∥fj,W (x) −fj(x)∥< 1

4r < 1

2r ≤dist(fj,W (∂BW (0, r)), pj).
(48)

Then, by [43], Definition 1.2.5, the formula (48) implies that

deg(fj,W , BW (0, r), pj) = deg(fj, BW (0, r), pj).

Moreover, as F1 = A1 = Be satisfies (46), and pj ∈BW (0, r), we have

deg(f0, BW (0, r), p0) = deg(Id, BW (0, r), p0) = sgn(det(Id)) = 1,

and
deg(f1, BW (0, r), p1) = deg(A1, BW (0, r), p1) = sgn(det(Be|W )) = −1.

Moreover, by our assumptions on B, the discretized maps Fj,W : W →W, j = 0, 1 are C1-
diffeomorphism. Hence, the inverse image F −1
j,W ({pj}) is a set containing only one point that
coincides with f −1
j,W ({pj}) (that in fact is the set {0}). This and the above show that for any R > r,
we have

deg(Fj,W , BW (0, R), pj)
=
deg(Fj,W , BW (0, r), pj)
(49)

=
deg(fj,W , BW (0, r), pj)
(50)

=
deg(fj, BW (0, r), pj)

=
(−1)j.

Let us now consider the maps Ft,W : W →W where t ∈[0, 1], that are C1-diffeomorphism
ft,W : W →W. As t →Ft = At is a continuous map [0, 1] →GL(X) and A : D →B is a
continuous functor, the map t →Ft,W is a continuous map [0, 1] →C(BW (0, r); W).

Let us consider ˆt ∈[0, 1] and denote pt = Ft,W (0). As Fˆt,W : W →W is a C1-diffeomorphism,
the set Fˆt,W (BW (0, r)) is open and hence there is ϵ > 0 such that

BW (pˆt, 5ϵ) ⊂Fˆt,W (BW (0, r)).
(51)

This implies that

distW (pˆt, Fˆt,W (∂BW (0, r))) > 5ϵ.
(52)

As t →Ft,W is a continuous map [0, 1] →C(BW (0, r); W), there is δ > 0 such that when
|t −ˆt| < δ then

sup
x∈V ∩BX(0,r)
∥Ft,V (x) −Fˆt,V (x)∥X < ϵ.
(53)

In particular, this implies that

∥Ft,V (0) −Fˆt,V (0)∥X < ϵ,
(54)

and thus pt = Ft,V (0) and pˆt = Fˆt,V (0) are in the ball BW (pˆt, 5ϵ) and hence by (51) these points
are in the same topological component of by W \ Fˆt,W (∂BW (0, r)). Hence, it follows from [43],
Theorem 1.2.6 (5), that for |t −ˆt| < δ, we have

deg(Fˆt,W , BW (0, r), pt) = deg(Fˆt,W , BW (0, r), pˆt).
(55)

Next, we observe that by (53), for any t1, t2 ∈[0, 1] satisfying |t1 −ˆt| < δ and |t2 −ˆt| < δ,

sup
x∈V ∩BX(0,r)
∥Ft1,V (x) −Ft2,V (x)∥X < 2ϵ.
(56)

23


---Page Break---
Inequalities (52), (54) and (56) imply that for any t1, t2 ∈[0, 1] satisfying |t1−ˆt| < δ and |t2−ˆt| < δ
we have

distW (pt1, Ft2,W (∂BW (0, r)))
=
inf
y∈∂BW (0,r) distW (pt1, Ft2,W (y))
(57)

>
inf
y∈∂BW (0,r) distW (pt1, Fˆt,W (y)) −ϵ

>
inf
y∈∂BW (0,r) distW (pˆt, Fˆt,W (y)) −2ϵ −ϵ

=
distW (pˆt, Fˆt,W (∂BW (0, r))) −3ϵ −ϵ > 5ϵ −3ϵ = 2ϵ.

As t →Ft,W is a continuous map [0, 1] →C(BW (0, r); W) and the formula (57) is valid, it follows
from [43], Theorem 1.2.6 (3) that for t1 and t2 satisfying |t1 −ˆt| < δ and |t2 −ˆt| < δ, we have

deg(Ft1,W , BW (0, r), pt2) = deg(Fˆt,W , BW (0, r), pt2).
(58)

This and (55) imply that

deg(Ft1,W , BW (0, r), pt2)
=
deg(Fˆt,W , BW (0, r), pt2)
(59)

=
deg(Fˆt,W , BW (0, r), pˆt),

In particular, when t1 = t2 satisfy |t1 −ˆt| < δ, this implies

deg(Ft1,W , BW (0, r), pt1)
=
deg(Fˆt,W , BW (0, r), pˆt).
(60)

As ˆt ∈[0, 1] is above arbitrary, this implies that the function

g : t →deg(Ft,W , BW (0, r), Ft,W (0)),
(61)

is a continuous integer valued function on [0, 1] and thus it is constant on the interval t ∈[0, 1].
However, by (49), g(0) = 1 is not equal to g(1) = −1 and hence g(t) can not be constant function
on the interval t ∈[0, 1]. This contradiction shows that the required functor does not exists.

A.5
Proofs from Sec. 3.2

A.5.1
Proof of Lemma 1

Proof. Let F : X →X be strongly monotone and C1-diffeomorphism. It is obvious that strongly
monotonicity implies that strictly monotonicity.

We first show that the strongly monotonicity implies that the coercivity. Indeed, by the strongly
monotonicity we have
⟨F(x) −F(0), x −0⟩X ≥α∥x∥2
X,
which implies that, as ∥x∥X →∞,

⟨F(x), x

∥x∥⟩X ≥⟨F(0), x

∥x∥⟩X + α∥x∥X →∞.

Therefore, F : X →X is coercive.

Let us consider the operator
FV := PV F|V : V →V.
From the definitions, FV : V →V is C1 and strongly monotones, and the (Fréchet) derivative
DFV |v at v ∈V is given by
DFV |v = PV (DF|v)|V ,
which is linear and continuous operator from V to V . Since FV : V →V is C1 and strongly
monotone, it is hemicontinuous, strictly monotones, and coercive. By the Minty-Browder theorem
[10, Theorem 9.14-1], FV : V →V is bijective, and then its inverse F −1
V
: V →V exists.

Next, let v ∈V . As FV : V →V is strongly monotones, we estimate that for all h ∈V ,

⟨FV (v + ϵh) −F(x)

ϵ
, (x + ϵh) −x

ϵ
⟩X = 1

ϵ2 ⟨F(x + ϵh) −F(x), (x + ϵh) −x⟩X

≥α

ϵ2 ∥(x + ϵh) −x∥2
X = α∥h∥2
X.

24


---Page Break---
Taking ϵ →+0, we have that
⟨DFV |v(h), h⟩X ≥∥h∥2
X.
Therefore, DFV |v : V →V is injective for all v ∈V . Then, it is bijective because V is now finite
dimensional. By the inverse function theorem, the inverse F −1
V
: V →V is C1.

A.5.2
Additional examples

Let us consider the elliptic (but not stongly elliptic) problem

Psu := −d

dt


sign(t −s) d

dtu(t)

= f(t),
t ∈[0, 1],
(62)

with the boundary conditions

u(0) = 0,
d
dtu(t)

t=1
= 0.
(63)

The FEM (Finite Element Method) matrix corresponding to this problem is [pjk(s)]n
j,k=1, where

pjk(s) =
Z 1

0
sign(t −s) d

dtχj(t) · d

dtχk(t)dt,
j, k = 1, . . . , m

and χj(t), j = 1, 2, . . . are a complete basis of the Hilbert space H1
D,N(0, 1) := {v ∈H1(0, 1) :
v(0) = 0} that correspond to the mixed Dirichlet and Neumann boundary conditions. Note that in
the FEM approximation we use only a finite subset of the complete basis of the Hilbert space. Note
for u in the canonical domain of the above problem that makes the operator appearing in the above
equation selfadjoint (the Friedrichs extension), that is, for

u ∈

v ∈H1(0, 1) : sign(t −s) d

dtv(t) ∈H1(0, 1), v(0) = 0, d

dtv(t)|t=1 = 0

,

we have
Psu = −D(2)
t AsD(1)
t u,

where As is the multiplication with the function sign(t −s), see (37), and D(1)
t
and D(2)
t
are the
derivative operators D(j)
t u = d

dtu having the different domains

D(D(1)
t ) = H1
D,N(0, 1) := {v ∈H1(0, 1) : v(0) = 0},
(64)

D(D(2)
t ) = H1
N,D(0, 1) := {v ∈H1(0, 1) : v(1) = 0}.
(65)

Observe that D(1)
t
and D(2)
t
have the inverse operators

(D(1)
t )−1v(t) =
Z t

0
v(t′)dt′,
(66)

(D(2)
t )−1v(t) = −
Z 1

t
v(t′)dt′,
(67)

that map (D(j)
t )−1 : L2(0, 1) →D(D(j)
t ). The eigenvalues of the matrix s →[pjk(s)]n
j,k=1 change
from positive values to negative values when s moves from 0 to 1. Thus, for some value s ∈(0, 1)
the matrix [pjk(s)]n
j,k=1 has a zero eigenvalue and is no invertible even though all maps D(1)
t , D(2)
t ,
and As : L2(0, 1) →L2(0, 1) are invertible. In particular, there are no finite FEM basis in where the
Galerkin discretizations of all operators Ps, s ∈[0, 1] are invertible.

Let us consider even simpler example: Similarly to the above, if we consider the Galerkin discretiza-
tions of the operator As : L2(0, 1) →L2(0, 1), we see that the Galerkin matrix corresponding to
As : L2(0, 1) →L2(0, 1) is [ajk(s)]n
j,k=1, where

ajk(s) =
Z 1

0
sign(t −s)ψj(t) · ψk(t)dt,
j, k = 1, . . . , m,

and ψj(t), j = 1, 2, . . . are a complete basis of the Hilbert space L2(0, 1). Again, we see that the
eigenvalues of the matrix s →[ajk(s)]n
j,k=1 change from positive values to negative values when s
moves from 0 to 1. Thus, for some value s ∈(0, 1) the matrix [ajk(s)]n
j,k=1 has a zero eigenvalue
and is no invertible even though all maps As : L2(0, 1) →L2(0, 1) are invertible. Thus there are no
finite basis in where the Galerkin discretizations of all operators As, s ∈[0, 1] are invertible.

25


---Page Break---
A.6
Proofs from Sec. 3.4

A.6.1
Proof of Lemma 4

Proof. From the assumption, it holds that

∥DF|x −I∥X→X ≤∥T1∥X→X∥DG|x∥X→X∥T2∥X→X ≤1

2.

Then, by the mean value theorem, there is 0 < t < 1 such that

⟨F(x1) −F(x2), x1 −x2⟩X
=
∂
∂t⟨F(x1 + t(x2 −x1)) −F(x2), x1 −x2⟩X

=
∂
∂t⟨DF

x1+t(x2−x1)
(x2 −x1), x1 −x2⟩X ≥1

2∥x1 −x2∥2
X.

These imply that F : X →X is strongly monotone.

A.6.2
Proof of Lemma 3

Proof. Observe that strongly monotone neural operators are coercive, that is,

lim
∥u∥X→∞⟨F(u),
u
∥u∥X
⟩X =
lim
∥u∥X→∞
1
∥u∥X
⟨F(u) −F(0), u −0⟩X = ∞.
(68)

and therefore F : X →X is surjective by Browder-Minty theorem [28, Thm. 2.1], see also [14]
considerations for neural operators. Moreover, the derivatives DF|x are linear strongly monotone
operators for all x ∈X, and therefore injective. Observe that the derivative is a linear operator of the
form
DF|x = I + T2 ◦DG|T1x ◦T1 : X →X,
and as T1 and T2 are compact and DG|T x is bounded we see that DF|x is a Fredholm operator of
index zero. Observe that

⟨DF|xv, v⟩X = 1

h lim
h→0⟨F(x + hv) −F(x)

h
, (x + hv) −x⟩X ≥α∥v∥2
X,

where h ∈R and v ∈X, and hence DF|x : X →X is an injective linear operator. As DF|x is a
Fredholm operator of index zero, this implies that the derivative DF|x : X →X is a bijection. As
the Hilbert space X can be identified with it dual space and the operator F : X →X is continuous
and hence hemi-continuous, it follows from [28], Theorem 3.1, that the map F : X →X is a
homeomorphism. As it is C1-smooth and its derivative is bijective operator DF|x : X →X at all
x ∈X, it follows that F : X →X is a C1-smooth diffeomorphism.

A.6.3
Proof of Theorem 3

Proof. To show the well-definedness of the discretization functor Alin, we need to show that, for
each strongly monotone C1-function F : X →X that is of the form (4), FV = PV F|V : V →V
is still strongly monotone C1-smooth of the form (4). This is given by Lemma 1. Moreover, such
functions are C1-smooth diffeomorphisms.

To verify assumption (A), let r, ϵ > 0, and let F : X →X be a strongly monotone diffeomorphisms
that is of the form F = Id + T2 ◦G ◦T1 : X →X where T1 : X →X and T2 : X →X are
compact linear operators, and G: X →X is such that G ∈C1(X). Since T2 ◦G ◦T1 is a compact
mapping, there is a finite-dimensional subspace V ⊂X, depending on ϵ > 0 such that

sup
x∈BX(0,r)
∥(Id −PV )T2G(T1x)∥X ≤ϵ,

which implies that

sup
x∈BX(0,r)∩V
∥FV (x) −F(x)∥X =
sup
x∈BX(0,r)∩V
∥(Id −PV )T2G(T1x)∥X ≤ϵ.

To prove the continuity in the sense of Definition 8 let r > 0 and let V ∈S0(X). Assume that

lim
j→∞
sup
x∈BX(0,r)
∥F (j)(x) −F(x)∥X = 0,

26


---Page Break---
where F (j) and F are strongly monotone diffeomorphisms F : X →X that are of the form (4).
Then, we see that, as j →∞,

sup
x∈V ∩BX(0,r)
∥F (j)
V (x) −FV (x)∥X ≤
sup
x∈V ∩BX(0,r)
∥PV F (j)(x) −PV F(x)∥X

≤
sup
x∈V ∩BX(0,r)
∥PV ∥op∥F (j)(x) −F(x)∥X ≤
sup
x∈BX(0,r)
∥F (j)(x) −F(x)∥X →0.

A.7
Proofs from Sec. 3.5

A.7.1
Proof of Theorem 4

In the proof of Theorem 4, we prove first few auxiliary lemmas

Lemma 5. A layer of a neural operator F : X →X is surjective. In particular, if F : X →X is
bilipschitz, then F : X →X is a homeomorphism.

Proof. By formula (4) a layer of neural operator F : X →X is of the form F(u) = u + T2G(T1u)
where T1 : X →X and T2 : X →X are compact linear operators and G : X →X is a function in
C1(X). Let c0, c1 > 0 be such that ∥G∥L∞(X) ≤c0 and LipX→X(G) ≤c1.

Let p ∈X and R0 > ∥T2∥X→Xc0 + ∥p∥X. Then Kp : X →X, defined by the formula

Kp(x) = −T2 ◦G ◦T1(x) + p,
x ∈X,

is a compact non-linear operator, see [43, Definition 2.1.11].

When p satisfies p ̸∈(Id −K0)(∂B(0, R0)), let deg(Id −K0, B(0, R0), p) = deg(Id −
Kp, B(0, R0), 0) be the infinite dimensional (Leray-Schauder) degree of the operator Id −K0 :
X →X in the set B(0, R0) with respect to the point p, see [43, Definition 2.2.3]. Let Kp;t : X →X
be the non-linear compact operators that depend on the parameter t ∈[0, 1], obtained by multiplying
Kp(x) by the number t, that is,

Kp;t(x) = tKp(x),
x ∈X.
(69)

As ∥Kp;t(x)∥X ≤∥T2∥X→Xc0 + ∥p∥X < R0 for all t ∈[0, 1] and x ∈X, we see that

Kp;t(x) ̸= x,
for x ∈∂BX(R0).
(70)

Then by the homotopy invariance of the Leray-Schauder degree, see [43, Theorem 2.2.4(3)], the
function

d(t) = deg(Id −Kp;t, B(0, R0), 0),
t ∈[0, 1],
(71)

is a constant function. Moreover, by [43, Theorem 2.2.4(1)],

deg(Id −Kp, B(0, R0), 0) = deg(I −Kp;1, B(0, R0), 0) = d(1)
=
d(0) = deg(I, B(0, R0), 0) = 1.

By [43, Theorem 2.2.4(2)], this implies that the equation

x = Kp(x),
or equivalently,
x + T2 ◦G ◦T1(x) = p,

has a solution x ∈BX(R0). As p ∈X was above arbitrary, this implies that F = Id + T2 ◦G ◦T1 :
X →X is surjection.

Finally, if F : X →X is bilipschitz, it is a bijection and its inverse function is Lipschitz function,
and thus F : X →X is a homeomorphism.

The next lemma shows the existence of a (finite-dimensional) orthogonal subspace so that for each
compact operator, projection onto the subspace is a perturbation of the identity under either pre or
post composition.

27


---Page Break---
Lemma 6. Let T1, T2 : X →X be compact operators and h > 0. There is a finite dimensional
space W ⊂X such that for the orthogonal projector PW : X →X it holds that

∥T1(Id −PW )∥X→X < h,
(72)
∥(Id −PW )T2∥X→X < h,
(73)
∥(Id −PW )T1(Id −PW )∥X→X < h,
(74)
∥(Id −PW )T2(Id −PW )∥X→X < h.
(75)

Proof. We use the singular value decomposition of compact operators: We can write

Tℓx =

∞
X

p=1
ωℓ,p⟨x, ψℓ,p⟩ϕ1,p,

where ψℓ,p and ϕℓ,p are orthogonal families in X and ωℓ,p ≥0 satisfy ωℓ,p+1 ≤ωℓ,p and ωℓ,p →0
as p →∞. For all h > 0 we can choose P > 0 such that ωℓ,p < h when p ≥P. Then

∥Tℓ−

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p∥X→X < h.

If

V = VP = span{ψℓ,p, ϕℓ,p; ℓ= 1, 2, p ≤P},
(76)

we see that if W ⊂X is a linear subspace such that V ⊂W then

(

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p) ◦PW =

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p,

PW ◦(

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p) =

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p,

and

PW ◦(

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p) ◦PW =

P
X

p=1
ωℓ,p⟨·, ψℓ,p⟩ϕ1,p,

and thus
∥Tℓ−Tℓ◦PW ∥X→X < h,
∥Tℓ−PW ◦Tℓ∥X→X < h,
and moreover,
∥Tℓ−PW ◦Tℓ◦PW ∥X→X < h.

Let

F W = Id + PW ◦T2 ◦G ◦T1 ◦PW : X →X.
(77)

Observe that for w ∈W and v ∈W ⊥we have

F W (w + v) = F W (w) + v,
F W (w) ∈W,
(78)

and

F W (W) ⊂W,
(79)

F W : W ⊥→W ⊥,
F W |W ⊥= IdW ⊥.
(80)

This means that F W = IdW ⊥⊕F W |W , where F W |W : W →W is a function which maps the
finite dimensional vector space W to itself.

The lemma below shows that given an F and ϵ > 0, we may perturb it by a Lipschitz term B so it
becomes the operator F W .

28


---Page Break---
Lemma 7. For any ϵ > 0, there is a finite dimensional space W ⊂X such that LipX→X(F −F W ) <
ϵ and for any ball BX(R) ⊂X, R > 0, the maps F : BX(R) →X and F W : BX(R) →X satisfy
∥F −F W ∥L∞(BX(R)) < 1

2(1 + R)ϵ.

Proof. Let us choose a finite dimensional space W ⊂X so that Lemma 6 is valid with

h =
1
4(1 + ∥G∥C1(X)))(1 + ∥T1∥X→X)(1 + ∥T2∥X→X)ϵ.

The right hand side of (72) is chosen so that

LipX→X(F −F W ) = Lip(PW ◦T2 ◦G ◦T1 ◦PW (x) −T2 ◦G ◦T1)
≤
Lip(PW ◦T2 ◦G ◦T1 ◦PW −PW ◦T2 ◦G ◦T1(x))
+Lip(PW ◦T2 ◦G ◦T1(x) −T2 ◦G ◦T1)
≤
∥T2∥X→XLip(G|X→X)∥T1(I −PW )∥X→X + ∥(I −PW )T2∥X→XLip(G|X→X)∥T1∥X→X

≤
1
2ε,

and

∥F −F W ∥L∞(B(0,R)) =
sup
x∈B(0,R)
∥PW ◦T2 ◦G ◦T1 ◦PW (x) −T2 ◦G ◦T1(x)∥X

≤
sup
x∈B(0,R)
∥PW ◦T2 ◦G ◦T1 ◦PW (x) −PW ◦T2 ◦G ◦T1(x)∥X

+
sup
x∈B(0,R)
∥PW ◦T2 ◦G ◦T1(x) −T2 ◦G ◦T1(x)∥X

≤
∥T2∥X→XLip(G|B(0,∥T1∥X→XR))∥T1(I −PW )∥X→X · R

+∥(I −PW )T2∥X→X∥G∥L∞(B(0,∥T1∥X→XR))

≤
1
4(R + 1)ϵ.

Lemma 8. For any ϵ > 0, there are finite dimensional space W ⊂X and (possibly non-linear)
functions B : X →X and ˜B : X →X such that

Lip(B) < ϵ,
Lip( ˜B) < ϵ.
(81)

and

F W = (Id + B) ◦F : X →X,
(82)

F = (Id + ˜B) ◦F W : X →X.
(83)

Moreover, B : X →X is a compact non-linear operator of the form

B = PW ◦F1 ◦PW + T2 ◦F2 ◦PW + PW ◦F3 ◦T1 + T2 ◦F4 ◦T1,
(84)

where F1, F2, F3, F4 : X →X are functions in C1(X) and ˜B : X →X is of the same form. In
addition, the operator Id + B : X →X and Id + ˜B : X →X are layers of neural operators.

Proof. Below, let C0 > 0 be such that

∥F(x) −F(y)∥X ≥C0∥x −y∥X,
x, y ∈X,
(85)

and let W be the space in Lemma 6 with

h =
1
4(1 + ∥G∥C1(X)))(1 + ∥T1∥X→X)(1 + ∥T2∥X→X)ϵ.

By Lemma 5, F(x) = x + T2G(T1(x)) is by an invertible function F : X →X. Thus we can define
a non-linear operator

F W ◦(F)−1
(86)

=(Id + PW ◦T2 ◦G ◦T1 ◦PW ) ◦(Id + T2 ◦G ◦T1)−1

=Id + (PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1) ◦(Id + T2 ◦G ◦T1)−1.

29


---Page Break---
Let

R
=
PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1
(87)

=

PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1 ◦PW



(88)

+

T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1


.
(89)

Then for all x, y ∈X

∥R(x) −R(y)∥X
≤
∥(Id −PW )T2∥X→X∥G∥Lip(X,X)∥T1∥X→X∥x −y∥X
+∥T2∥X→X∥G∥Lip(X,X)∥T1(Id −PW )∥X→X∥x −y∥X

≤
1
2C0
ϵ∥x −y∥X.

and thus,

∥R ◦(Id + T2 ◦G ◦T1)−1(x) −R ◦(Id + T2 ◦G ◦T1)−1(y)∥X

≤
1
2ϵ∥x −y∥X.

This implies that F W : X →X satisfies for all x, y ∈X

∥(F W ◦(F)−1 −Id)(x) −(F W ◦(F)−1 −Id)(y)∥X ≤1

2ϵ∥x −y∥X,

and thus

∥(F W ◦(F)−1 −(F W ◦(F)−1(y)∥X ≤(1 + 1

2ϵ)∥x −y∥X,
(90)

and

Lip(F W : X →X) ≤(1 + 1

2ϵ) · Lip(F : X →X).
(91)

Moreover, for all x, y ∈X

∥F W (x) −F W (y)∥X ≥(C0 −ϵ)∥x −y∥X ≥1

2C0∥x −y∥X.
(92)

Let c0 > 0 be such that ∥G∥L∞(X) ≤c0. Let R0 > 0. As ∥T2 ◦G∥L∞(X) ≤∥T2∥c0, finite
dimensional degree theory, see using [43, Theorem 1.2.6], as above that

BW (0, R0) ⊂F W (BW (0, R1)),
(93)

when R1 > R0 + ∥T2∥c0 ≥R0 + ∥PW ◦T2 ◦G ◦T1 ◦P2∥L∞(X). As R0 > 0 above is arbitrary,
formula (93) implies that F W : X →X is surjective. Thus, we have shown that F W : X →X is a
bijective bilipschitz map.

Similarly to the above, by replacing (85) by (92) and changing the roles of F W and F, we see that
for x, y ∈X

∥R ◦(I + PW ◦T2 ◦G ◦T1 ◦PW )−1(x) −R ◦(I + T2 ◦G ◦T1)−1(y)∥X
≤
ϵ∥x −y∥X,

and

∥(F ◦(F W )−1 −Id)(x) −(F ◦(F W )−1 −Id)(y)∥X ≤ϵ∥x −y∥X.
(94)

The above implies that

F W = (Id + B) ◦F,
(95)

F = (Id + ˜B) ◦F W ,
(96)

30


---Page Break---
where

B = (F W ◦F −1 −Id) : X →X,
˜B = (F ◦(F W )−1 −Id) : X →X,

satisfy

Lip(B) < 1

2ϵ,
Lip( ˜B) < ϵ.
(97)

Next we use that

(Id + H)−1
=
(Id + H)−1 ◦(Id + H −H)

=
Id −(Id + H)−1 ◦H,

so that

(Id + T2 ◦G ◦T1)−1
=
Id −(Id + T2 ◦G ◦T1)−1 ◦(T2 ◦G ◦T1).

Observe that by (86)

B =F W ◦(F)−1 −Id
(98)

=(PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1) ◦(Id + T2 ◦G ◦T1)−1

=(PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1)

−(PW ◦T2 ◦G ◦T1 ◦PW −T2 ◦G ◦T1) ◦(Id + T2 ◦G ◦T1)−1 ◦(T2 ◦G ◦T1),

and as PW : X →X is a finite rank operator and T2 : W →W is a compact linear operator, we see
that B : X →X is a compact non-linear operator of the form

B = PW ◦F1 ◦PW + T2 ◦F2 ◦PW + PW ◦F3 ◦T1 + T2 ◦F4 ◦T1,

where F1, F2, F3, F4 : X →X are functions in C1(X).

Moreover, similarly to (86), we see that

F ◦(F W )−1
(99)

=(Id + T2 ◦G ◦T1) ◦(Id + PW ◦T2 ◦G ◦T1 ◦PW )−1

=Id + (T2 ◦G ◦T1 −PW ◦T2 ◦G ◦T1 ◦PW ) ◦(I + PW ◦T2 ◦G ◦T1 ◦PW )−1,

and hence
˜B =F ◦(F W )−1 −Id
(100)

=(T2 ◦G ◦T1 −PW ◦T2 ◦G ◦T1 ◦PW ) ◦(Id + PW ◦T2 ◦G ◦T1 ◦PW )−1

=(T2 ◦G ◦T1 −PW ◦T2 ◦G ◦T1 ◦PW )
−(T2 ◦G ◦T1 −PW ◦T2 ◦G ◦T1 ◦PW )

◦(Id + PW ◦T2 ◦G ◦T1 ◦PW )−1 ◦(PW ◦T2 ◦G ◦T1 ◦PW ),

and again, as PW : X →X is a finite rank operator and T2 : W →W is a compact linear operator,
we see that ˜B : X →X is a compact non-linear operator of the form

˜B = PW ◦˜F1 ◦PW + T2 ◦˜F2 ◦PW + PW ◦˜F3 ◦T1 + T2 ◦˜F4 ◦T1,

where ˜F1, ˜F2, ˜F3, ˜F4 : X →X are functions in C1(X). For an infinite dimensional Hilbert space X
there is a linear isomorphism J : X →X × X, as the cardinality of Hilbert basis of the space X is
the same as the cardinality of the Hilbert basis of X × X, see [26, Theorem 3.5]. Then, by writing
the isomorphism J as J(x) = (J1(x), J2(x)) ∈X × X, we see that

˜B(x)
=
(( PW
T2 ) ◦J) ◦(J−1 ◦
 ˜F1
˜F3
˜F2
˜F4


◦J) ◦(J−1 ◦

PW
T1


)(x),

that is a composition of a linear compact operator X →X, a non-linear operator X →X, and a
compact operator X →X. Hence we see that Id + ˜B, and similarly Id + B, are layers of neural
operators.

31


---Page Break---
Now we present our Proof of Theorem 4.

Proof. Let F(x) = x + T2G(T1(x)). Without loss of generality, we may assume that 0 < ϵ <
min(∥T2∥X→XC1∥T1∥X→X, 1

2).

By (94)

∥B(x) −B(y)∥X ≤ϵ∥x −y∥X,
(101)

and

⟨(Id + B)(x) −(Id + B)(y), x −y⟩X ≥∥x −y∥2
X −ϵ∥x −y∥2
X
= (1 −ϵ)∥x −y∥2
X,
(102)

which implies that

Id + B : X →X,
(103)

is a strongly monotone operator. Similarly, we see that

Id + ˜B : X →X,
(104)

is a strongly monotone operator.

Recall that FW maps W ⊥to itself and it is equal to the identity map in W ⊥, and moreover F W can
be decomposed according to the formula (78). Thus we can write

FW = IdW ⊥⊕(PW ◦FW ◦PW ) : W ⊥⊕W →W ⊥⊕W.

Below, let us identify W with Rn to clarify notations. We denote

f = FW |W = PW ◦FW ◦PW : W →W.
(105)

By (91) and (92) there are c1, c0 > 0 such that

c0|x −y| ≤|f(x) −f(y)| ≤c1|x −y|.
(106)

Define4 for 0 < t ≤1

ft(x) := 1

t (f(tx) −f(0)) + tf(0).
(107)

For t = 0, we define

f0(x) := Df|0x.
(108)

where Df|y : Rn →Rn is the derivative of the map f at the point y, that is considered as a linear
map (or a matrix), and we denote its value at vector v by Df|y(v) = Df|yv.

Then f1(x) = f(x) and

|ft(x) −ft(y)| = 1

t |f(tx) −f(ty)|,
(109)

and as
1

t c0|tx −ty| ≤1

t |f(tx) −f(ty)| ≤1

t c1|tx −ty|,

we have
c0|x −y| ≤|ft(x) −ft(y)| ≤c1|x −y|.

Below, let

R0 = |f(0)|,
(110)

and

R1 = c1r1 + R0,
(111)

4The above used homotopy can be replace by a more explicit flow in the Lie group of diffeomorphism, see
e.g. [42].

32


---Page Break---
where r1 appears in the claim of Theorem 4. Observe that then f0(BRn(0, r1)) ⊂BRn(0, R1).

As f ∈C2(Rn, Rn), we have by Taylor’s series (with the remainder written in the integral form)

f(tx) = f(0) + Df|0(tx) + 1

2

Z t

0
(t −s)Df|sxx ds

= f(0) + tDf|0x + 1

2t2
Z 1

0
(1 −s)Df|tsxx ds.
(112)

Thus,

ft(x) = Df|0x + 1

2t
Z 1

0
(1 −s)Df|tsxx ds + tf(0).
(113)

Observe that for all x ∈X,

∥Df|x∥≤c1, and ∥(Df|x)−1∥≤c−1
0 .

By (113),

ft((Df|0)−1x) −ft((Df|0)−1y)

= x −y + 1

2t
Z 1

0
(1 −s)

Df|ts˜x(Df|0)−1x −Df|ts˜y(Df|0)−1y

!

ds

˜x=(Df|0)−1x, ˜y=(Df|0)−1y

= x −y + 1

2t
Z 1

0
(1 −s)

Df|ts˜x(Df|0)−1(x −y)

!

ds

˜x=(Df|0)−1x, ˜y=(Df|0)−1y

+ 1

2t
Z 1

0
(1 −s)

(Df|ts˜x −Df|ts˜y)(Df|0)−1y

!

ds

˜x=(Df|0)−1x, ˜y=(Df|0)−1y
.
(114)

Here, for x, y ∈BRn(0, R1),

∥Df|ts˜x −Df|ts˜y∥Rn→Rn ≤∥f∥C2(BRn(0,R1)),

∥Df|ts˜x∥Rn→Rn ≤c1,

∥(Df|0)−1y∥Rn ≤c−1
0 ∥y∥Rn.

Hence,

ft ◦(Df|0)−1 = Id + Qt,0 : Rn →Rn,
(115)

where

LipBRn(0,R1)(Qt,0) ≤t 1

2c0
(c1 + ∥f∥C2(BRn(0,R1))R1).
(116)

Let us choose t1 ∈(0, 1) such that

t1 <
2c0
c1 + ∥f∥C2(BX(0,R1))R1
ε,
(117)

so that

LipBRn(0,R1)(Qt1,0) < ε.
(118)

Below, we will denote

Htt1,0(x) = ft1(f −1
0 (x)) = ft1((Df|0)−1(x)).
(119)

Next we consider operators ft with t > 0.

Let us next consider t2, t3, . . . , tm+1 ∈(0, 1] such that t2 > t1, tm+1 = 1, and tk+1 > tk.

We have
|ft(0)| = t|f(0)| = tR0,

33


---Page Break---
and Lip(ft) ≤c1. Hence,

|ft(x)| ≤c1|x| + R0.
(120)

Moreover, for z = f(0) we have f −1
t
(z) = 0 and |z| = R0, and as the Lip(f −1
t
) ≤c−1
0 ,

|f −1
t
(x)| ≤|f −1
t
(x) −f −1
t
(z)| + |f −1
t
(z)| ≤c−1
0 |x −z| + 0 ≤c−1
0 |x| + c−1
0 |z|,
(121)

so that

|f −1
t
(x)| ≤c−1
0 |x| + c−1
0 R0.
(122)

Also, as we can write

ft(x) := 1

t (f(tx) −f(0)) + tf(0) = 1

t f(tx) + (t −1

t )f(0),
(123)

we have for tk, tk+1 ∈(0, 1], k ≥1,

(ftk+1 −ftk)(x) =
1
tk+1
f(tk+1x) −1

tk
f(tkx) −(
1
tk+1
−1

tk
)f(0) + (tk+1 −tk)f(0)

= (
1
tk+1
−1

tk
)f(tk+1x) + 1

tk
(f(tk+1x) −f(tkx))

−(
1
tk+1
−1

tk
)f(0) + (tk+1 −tk)f(0)

= (tk −tk+1

tk
)
1
tk+1
f(tk+1x) + 1

tk
(f(tk+1x) −f(tkx))

−(
1
tk+1
−1

tk
)f(0) + (tk+1 −tk)f(0)

= (tk −tk+1

tk
)

1
tk+1
f(tk+1x) + (tk+1 −
1
tk+1
)f(0)

+ 1

tk
(f(tk+1x) −f(tkx))

−(tk −tk+1

tk
)(tk+1 −
1
tk+1
)f(0) −(
1
tk+1
−1

tk
)f(0) + (tk+1 −tk)f(0)

= (tk −tk+1

tk
)ftk+1(x) + 1

tk
(f(tk+1x) −f(tkx)) + Ck,

where

Ck
=
−(tk −tk+1

tk
)(tk+1 −
1
tk+1
)f(0) −(
1
tk+1
−1

tk
)f(0) + (tk+1 −tk)f(0)
(124)

=
−(tk −tk+1

tk
)(tk+1 −
1
tk+1
)f(0) + tk+1 −tk

tktk+1
f(0) + (tk+1 −tk)f(0)
(125)

=
(tk+1 −tk)
 1

tk
+ 1

f(0).
(126)

To analyze the above, we denote

k(x; tk, tk+1) := 1

tk
(f(tk+1x) −f(tkx))

= 1

tk

Z tk+1

tk
Df|tk+s′(x)ds′

= 1

tk

Z 1

0
(tk+1 −tk)Df|tk+s(tk+1−tk)(x)ds.

As ∥Df|tk+s(tk+1−tk)(x)∥≤c1∥x∥, we have for all R > 0

Lip(k(·; tk, tk+1) : BRn(0, R) →Rn) ≤c1
tk+1 −tk

tk
R.

and

|k(x; tk, tk+1)| ≤|tk+1 −tk| · c1

tk
|x|.
(127)

34


---Page Break---
Moreover, for x1, x2 ∈BRn(0, R),

|(ftk+1 −ftk)(x2) −(ftk+1 −ftk)(x1)| ≤c1
tk −tk+1

tk
|x2 −x1| + c1
tk+1 −tk

tk
R|x2 −x1|,

(128)

and for all x ∈Rn,

|(ftk+1 −ftk)(x)| ≤|tk −tk+1|

tk
|ftk+1(x)| + |k(x; tk, tk+1)| + |Ck|
(129)

≤|tk −tk+1|

tk
(c1|x| + R0) + |tk+1 −tk| · c1

tk
|x| + |tk+1 −tk|
 1

tk
+ 1

R0

(130)

≤|tk −tk+1|

tk
(2c1|x| + 3R0).
(131)

Moreover,

ftk+1(f −1
tk (x)) = ftk(f −1
tk (x)) + (ftk+1 −ftk) ◦(f −1
tk (x))
(132)

= x + (ftk+1 −ftk) ◦f −1
tk (x).
(133)

Let

Qtk+1,tk(x) = ftk+1(f −1
tk (x)) −x
(134)

= (ftk+1 −ftk) ◦f −1
tk (x).
(135)

Hence by (128), for x1, x2 ∈BRn(0, R),

|Qtk+1,tk(x2) −Qtk+1,tk(x1)| ≤1

c0
(c1
tk −tk+1

tk
+ c1
tk+1 −tk

tk
R)|x2 −x1|.
(136)

By (122) and (129), it hold that of all x ∈Rn

|Qtk+1,tk(x)|
≤
|tk −tk+1|

tk


2c1( 1

c0
|x| + 1

c0
R0) + 3R0)


(137)

≤
|tk −tk+1|

tk
(2c1

c0
|x| + (2c1

c0
+ 3)R0).
(138)

Let R2 ≥R1 and define for k ≥1 a function that is a convex combination of the identity map Id and
the map ftk+1 ◦f −1
tk ,

Htk+1,tk(x) = (1 −ϕ(x)x + ϕ(x)ftk+1(f −1
tk (x))
(139)

= x + ϕ(x)(ftk+1(f −1
tk (x)) −x)
(140)

= x + ϕ(x)Qtk+1,tk(x),
(141)

where ϕ ∈C∞
0 (Rn) is such that ϕ(x) = 1 for |x| ≤R2, ϕ(x) = 0 for |x| ≤2R2, and ∥∇ϕ(x)∥≤
2/R2. Then
Htk+1,tk = Id + Ptk+1,tk,
where

LipRn(Ptk+1,tk)
≤
∥ϕ∥L∞(B(0,2R2)) · LipB(0,2R2)(Qtk+1,tk) + ∥Qtk+1,tk∥L∞(B(0,2R2))Lip(ϕ)

≤
1
c0
(c1
tk+1 −tk

tk
+ c1
tk+1 −tk

tk
R2)
(142)

+|tk −tk+1|

tk
(4c1

c0
R2 + (2c1

c0
+ 3)R0)) · 2

R2
.
(143)

For k ≥1 we have tk ≥t1, and thus

LipRn(Ptk+1,tk)
≤
|tk+1 −tk|

t1

c1

c0
(10 + R2) + 6

R2
(c1

c0
+ 1)R0


.
(144)

35


---Page Break---
Then, when the condition ftk−1(x) ∈BRn(0, R2) it holds, we see that

Htk,tk−1 ◦ftk−1(x)
=
ftk−1(x) + ϕ(ftk−1(x))(ftk(f −1
tk−1(ftk−1(x))) −ftk−1(x))

=
ftk−1(x) + 1 · (ftk(f −1
tk−1(ftk−1(x))) −ftk−1(x))

=
ftk(f −1
tk−1(ftk−1(x)))

=
ftk(x) ∈BRn(0, R2).

Observe that when x ∈BRn(0, R0) then ft0(x) ∈BRn(0, R1) ⊂BRn(0, R2). Next, we denote
t0 = 0. Hence, by using induction, we see that for all j = 1, 2, . . . , m + 1 it holds that

Htj,tj−1 ◦Htj−1,tj−2 ◦Ht1,t0 ◦ft0(x) = ftj(x), and
Htj,tj−1 ◦Htj−1,tj−2 ◦Ht1,t0 ◦ft0(x) = ftj(x) ∈BRn(0, R2).

We recall that above we chose t1 > 0 such that

t1 <
2c0
c1 + ∥f∥C2(BRn(0,R1))R1
ε.
(145)

We now choose m so that there are tk, k = 2, 3, . . . , m + 1 satisfying |tk −tk−1| < 1

m and

1
m < ε
t1

c1
c0 (10 + R2) +
6
R2 ( c1

c0 + 1)R0

.

This means that we can choose

m > 1

ε
2
t1

c1

c0
(10 + R2) + 6

R2
(c1

c0
+ 1)R0


.
(146)

Then,

LipRn(Ptk+1,tk)
≤
ε.
(147)

We define the map

α(t) =
 ft|K, for 0 < t ≤1,
Df|0, for t = 0,

is a continuous map [0, 1] →C1(Rn; Rn).

The space W is a finite dimensional Euclidean space, and let GL(W) be the set of linear diffeo-
morphisms of it, that is, invertible linear operators A : W →W. We consider GL(W) with the
topology given by the operator norm of linear maps. Then GL(W) is a topological space with two
path connected components – those matrices which have a positive determinant and those having a
negative determinant.

The above implies that Df|0 can be connected in GL(Rn) either to a matrix B where

B is either the identity map, or the diagonal matrix diag(−1, 1, 1, . . . , 1),
(148)

with a continuous path βs ∈GL(W), s ∈[0, 1] with Df|0 = β1 and B = β0.

We need to consider the path βs in GL(W) from β1 = f0 = Df|0 to matrix β0 = B ∈O(n). There
are some explicit formulas in literature with relatively complicated formulas, see [41]. However, let
us next consider estimates using a non-optimal but relatively explicit path. To do this we start from the
PU-decomposition of the matrix β1 = Df|0, that we denote β1 = PU where P = diag (σ1, . . . , σn)
is positive matrix (given in a suitable basis) and U is an orthogonal matrix. Then, we consider the
path t →PtU, where t ∈[0, 1] and

Pt = P 1−t = exp((1 −t) log(P)).

This is a path from the matrix Df|0 = PU to the matrix U. Observe that

Pt2U = P t1−t2Pt1U,

36


---Page Break---
where

P t1−t2 = diag (σt1−t2
1
, . . . , σt1−t2
n
),

and when |t1 −t2| ≤1, by mean value theorem we have

∥P t1−t2 −I∥
≤
max
j=1,...,n(σt1−t2
j
−1) ≤
max
j=1,...,n(σj, σ−1
j ) · ( max
j=1,...,n | log σj|) · |t1 −t2|

≤
(1 + c1 + c−1
0 )2 · |t1 −t2|.

That is, using the norm of Df|0 and the norm of inverse matrix (Df|0)−1 we can bound the length
of the path t →PtU, t ∈[0, 1] (in the operator norm). After this, we need to consider the path from
U to B = Idk × (−Idn−k) in O(W). For this, we could use the structure of the Lie group O(n),
n = dim(W). By [15, Thm. 11.3.3], any matrix U in O(n) can be written as a tensor product of
elements in O(2) and possibly an operator ±Id : R →R, that is, in a suitable orthogonal basis a
matrix U ∈O(n) is a block diagonal matrix of 2 × 2 matrixes in O(2) and one or two 1 × 1 matrices
±Id, that is,

U =





R1
...
Rk

0

0
±1
...
±1





,

where the matrices R1 = R1(ϑ1), ..., Rk = R1(ϑk) are 2-by-2 rotation matrices in SO(2),

Using this, we find a path form U(s), s ∈[0, 1] either to a matrix Idk × (−Idn−k),

U(s) =





R1(sϑ1)
...
Rk(sϑk)
0

0
±1
...
±1





.

Observe that due the block diagonal form of this matrix, the operator norm satisfies

∥U(s2) −U(s1)∥Rn→Rn ≤
max
j=1,2...,k ∥Rj(s2ϑj) −Rj(s1ϑj)∥R2→R2 ≤C∗|s2 −s1|,

where C∗is an absolute constant (that does not depend on the dimension n). We obtain the path
s →βs by concatenating the paths from PU to U in GL(n) and from U to B in SO(n). The
length of the obtained path β in the operator norm metric can be estimated and it is bounded
C(1 + c1 + c−1
0 )2 plus a constant. Moreover, the path s →βs can be decomposed to a product of
(C(1 + c1 + c−1
0 )2 + C)ε−1 matrices of the form Id + Bj where ∥Bj∥≤ε.

Summarising the above analysis, we see that J can bounded by

J ≤1

ε
2
t1

c1

c0
(10 + R2) + 6

R2
(c1

c0
+ 1)R0


+ C(1 + c1 + c−1
0 )2ε−1 + Cε−1.
(149)

We recall that here

t1 <
2c0
c1 + ∥f∥C2(BRn(0,R1))R1
ε,
(150)

radii R0 and R1 are given in formulas (110) and (111) and c0 and c1 are the bi-Lipschitz constants of
f, see (106).

Note that c0, c1, C > 0 are constants independent of ϵ, while R0, R1, R2 > 0 depends on |f(0)|. We
see that by f = PW F|W .
|f(0)| ≤∥F(0)∥X,

37


---Page Break---
thus, the upper bounds of R0, R1, R2 are independent of ϵ.
By choosing t1 as t1
=
c0
c1+∥f∥C2(BRn (0,R1))R1 ε, we furthermore estimate that

J ≲∥f∥C2(BRn(0,R1);Rn)ε−2.
(151)

Here, if we assume that F ∈C2(X, X), we see that

∥f∥C2(BRn(0,R1);Rn) ≤∥F∥C2(W ;X) ≤∥F∥C2(X;X) < ∞,

Thus, we obtain
J = O(ε−2).

Now we can finish the proof of Theorem 4. Denote ˜ft = ft ⊕IdW ⊥, ˜Ht1,t0 = Ht1,t0 ⊕IdW ⊥
˜βs = βs ⊕IdW ⊥and A0 = B ⊕IdW ⊥. Using these notations, we can write F(x) as

F(x) = (F ◦(F W )−1) ◦˜Htm+1,tm ◦˜Htm,tm−1 ◦. . .

· · · ◦˜Ht1,t0 ◦(˜β1 ◦˜β−1
sm) ◦(˜βsm ◦˜β−1
sm−1) ◦· · · ◦(˜βs1 ◦˜β−1
0 ) ◦A0,
(152)

where tj and sj are chosen so that 0 = t0 < t1 < . . . tm < tm+1 = 1 and 0 = s0 < s1 < . . . sm <
sm+1 = 1 and that the Lipschitz constant of the maps ˜Htj+1,tj −Id and ˜βsj ◦˜β−1
j−1 −Id are less
that ϵ.

Moreover, recall that

F W = Id + PW ◦T2 ◦G ◦T1 ◦PW : X →X.

Observe that by writing x ∈X in the form x = x0 + x1, where x0 = (I −PW )x and x1 = PW x,
we see that

F W (x0 + x1)
=
(I + PW ◦T2 ◦G ◦T1 ◦PW )(x0 + x1)
=
x0 + (I + PW ◦T2 ◦G ◦T1 ◦PW )(x1)

=
(I −PW )x + PW


(I + PW ◦T2 ◦G ◦T1 ◦PW ))(PW x)


=
(I −PW )x + PW (F W (PW x)).
(153)

Hence, (F W )−1 : X →X can be written as

(F W )−1
=
I −PW + PW ◦(F W )−1 ◦PW
=
I + PW ◦(−I + (F W )−1) ◦PW ,
(154)

and thus (F W )−1 is a layer of a neural operator by definition. Similarly, as

˜f = F W = IdX + PW ◦(FW −IdW ) ◦PW : X →X.

and

ft(x)
=
1

t (f(tx) −f(0)) + tf(0)

=
Id −PW + PW (Id + 1

t (f(tPW x) −f(0)) + tf(0)),
(155)

for 0 < t ≤1, and we see as above that ft and f −1
t
are neural operators. Similarly, we see that ˜β−1
s
and ˜β−1
s
are neural operators. Hence, all factors in the product (152) are (strictly monotone) neural
operators.

38


---Page Break---
A.7.2
Proof of Theorem 5

Theorem 6 (Theorem 5 in the main text). Let X be a separable Hilbert space, and let φ = {φn}n∈N
be an orthonormal basis in X. Let δ ∈(0, 1), and let R > 0, and let F : X →X be a layer
of a bilipschitz neural operator, as in Def. 3. Then, for any ϵ ∈(0, 1), there are T, N ∈N and
G ∈RT,N,φ,ReLU(X) that has the form

G = (IdX + DN ◦NNT ◦EN) ◦· · · ◦(IdX + DN ◦NN1 ◦EN),

such that
sup
x∈BX(0,R)
∥F(x) −G ◦A(x)∥X ≤ϵ,

where A : X →X is a linear invertible map that is either the identity map or a reflection operator
x →x −2⟨x, e⟩Xe with some unit vector e ∈X. Moreover, G ◦A : BX(0, R) →G ◦A(BX(0, R))
is invertible, and there exists R′ > 0 such that

A(BX(0, R)) ⊂BX(0, R′),

and denoting by

Γ0 := BX(0, R′),
Γ1 := B(0, R′+δ),
· · ·
ΓT := B(0, R′+Tδ),
ΓT +1 := B(0, R′+(T+1)δ),

and
˜K0 := A(BX(0, R)),
˜K1 := (IdX+DN ◦NN1◦EN) ˜K0,
˜K2 := (IdX+DN ◦NN2◦EN) ˜K1,

· · ·
˜KT := (IdX + DN ◦NNT ◦EN) ˜KT −1 = G ◦A(BX(0, R)),
we have that for each t = 0, 1, ..., T
˜Kt ⊂Γt
and the mapping G ◦A|BX(0,R) : BX(0, R) →G ◦A(BX(0, R)) is bijective and its inverse is given
by
 
G ◦A|BX(0,R)
−1 = A−1 ◦Φ,

where Φ : G ◦A(BX(0, R)) →A(BX(0, R)) is defined by

Φ := L1| ˜
K1 ◦· · · ◦LT −1| ˜
KT −1 ◦LT | ˜
KT ,
(156)

where Lt : Γt →Γt+1
is defined by
Lt(y) := lim
n→∞π1 ◦(⃝n
h=1ϕt) ◦e1(y),

where ϕt : Γt+1 × Γt →Γt+1 × Γt is defined by

ϕt(x, y) = (y + DN ◦NNt ◦EN(x), y),

where π1(x, y) = x and e1(y) = (0, y).
Remark 2. Note that, in the proof, we have used the approximation result [22, Theorem 4.3] of
Sobolev functions by neural networks with ReLU function x 7→max{0, x}, which is not differentiable
at 0. Alternatively, we can use approximation result [1, Theorem 4.1] by neural networks with ReCU
function x 7→max{0, x}3, which are continuously differentiable. Then, each block IdX + DN ◦
NNt ◦EN in a obtained approximator G is C1 and strongly monotone on ball Γt = B(0, R′ + tδ),
that is, it holds that there is an α > 0 such that

⟨(IdX + DN ◦NNt ◦EN)x, x⟩X ≥α∥x −y∥2
X, x, y ∈Γt,

which implies that, by the same argument as in Lemma 3,

(IdX + DN ◦NNt ◦EN) : Γt →(IdX + DN ◦NNt ◦EN)(Γt),

is a C1-diffeomorphism.
Remark 3. From the proof, we can show that, for each t = 1, ..., T

Lt( ˜Kt) ⊂˜Kt−1.

Then, the mapping Φ defined in (156) is well-defined.

39


---Page Break---
Proof. Let δ, ϵ ∈(0, 1), and let F : X →X be a layer of a neural operator defined in (4). Assume
that F is bilipschitz, that is, it satisfies (85) and

∥F(x) −F(y)∥X ≤C1∥x −y∥X,
for all x, y ∈X.
(157)

By Theorem 4, there is T ∈N such that F can represented in the form

F(x) = (IdX −HT ) ◦(IdX −HT −1) ◦· · · ◦(IdX −H1) ◦A(x),
for x ∈BX(0, R),

where A : X →X is a linear invertible map that is either the identity map or a reflection operator
x →x −2⟨x, e⟩Xe with some unit vector e ∈X, and Ht : X →X are global Lipschitz maps
satisfying
LipX→X(Ht) ≤δ/2.
(158)

for all t. Moreover, the operator Ht : X →X is a compact mapping. We choose R′ > 0 such that

A(BX(0, R)) ⊂BX(0, R′).

We denote by

K0 := A(BX(0, R)),
K1 := (IdX −H1)(K0),
· · ·
KT := (IdX −HT )(KT −1).

Γ0 := BX(0, R′),
Γ1 := B(0, R′ + δ),
· · ·
ΓT +1 := B(0, R′ + (T + 1)δ).

We choose ˜R > 0 such that for all t = 0, ..., T + 1

EN(Kt), EN(Γt) ⊂BRN (0, ˜R).

Since Ht : X →X is a compact mapping, for large enough N ∈N, we have that for all t = 1, ..., T,

sup
x∈BX(R)
∥Ht(x) −PVN ◦Ht ◦PVN (x)∥X ≤
ϵ
2T(1 + δ)T .
(159)

Note that
PVN ◦Ht ◦PVN = DN ◦˜HN,t ◦EN,

where ˜HN,t : RN →RN by
˜HN,t := EN ◦Ht ◦DN.

From (158), we have
˜HN,t ∈W 1,∞(BRN (0, ˜R); RN).

By approximation by ReLU neural networks in Sobolev spaces (see e.g., [22, Theorem 4.3]) , there is
a ReLU neural network NNt : RN →RN, NNt ∈R1,N,φ,ReLU(X) such that

∥˜HN,t −NNt∥W 1,∞(BRN (0, ˜
R);RN) ≤min

ϵ
2T(1 + δ)T , δ

2


.
(160)

Then, we estimate that by (159) and (160)

sup
x∈Kt−1
∥Ht(x) −DN ◦NNt ◦EN(x)∥X ≤
ϵ
T(1 + δ)T .
(161)

Also, we estimate that by (158) and (160)

LipBRN (0, ˜
R)→RN (NNt)
≤LipBRN (0, ˜
R)→RN ( ˜HN,t) + LipBRN (0, ˜
R)→RN ( ˜HN,t −NNt)

≤LipX→X(Ht) + ∥˜HN,t −NNt∥W 1,∞(BRN (0, ˜
R);RN) ≤δ.
(162)

We denote G : X →X by

G := (I −DN ◦NNT ◦EN) ◦(I −DN ◦NNT −1 ◦EN) ◦· · · ◦(I −DN ◦NN1 ◦EN),

40


---Page Break---
which belong to RT,N,φ,ReLU(X). Then, by (161) and (162), we estimate that for all x ∈BX(0, R),

∥F(x) −G ◦A(x)∥X

≤
X

1≤t≤T
∥
 
⃝T
h=t+1(I −DN ◦NNh ◦EN)

◦
 
⃝t
h=1(I −Hh)

◦A(x)

−
 
⃝T
h=t(I −DN ◦NNh ◦EN)

◦
 
⃝t−1
h=1(I −Hh)

◦A(x)∥X

≤
X

1≤t≤T

T
Y

h=t+1


1 + LipBRN (0, ˜
R)→RN (NNh)

sup
y∈Kt−1
∥(I −Ht)(y) −(I −DN ◦NNt ◦EN)(y)∥X

≤
X

1≤t≤T

T
Y

h=t+1
(1 + δ)
sup
y∈Kt−1
∥Ht(y) −DN ◦NNt ◦EN(y)∥X

≤
X

1≤i≤T
(1 + δ)T
ϵ
T(1 + δ)T ≤ϵ.

Next, as (160), we see that

(I −DN ◦NN1 ◦EN)|Γ0 : Γ0 →Γ1,
(I −DN ◦NN2 ◦EN)|Γ1 : Γ1 →Γ2,
· · ·

(I −DN ◦NNT ◦EN)|ΓT −1 : ΓT −1 →ΓT .
and

G|Γ0 = (I−DN ◦NNT ◦EN)|ΓT −1 ◦(I−DN ◦NNT −1◦EN)|ΓT −2 ◦· · ·◦(I−DN ◦NN1◦EN)|Γ0,

which means that G|Γ0 maps from Γ0 to ΓT . For t = 1, ..., T, we define Lt : Γt →Γt+1 by

Lt(y) := x∗,
y ∈Γt,

where x∗∈Γt+1 is a unique fixed point of ht,y : Γt+1 →Γt+1, where

ht,y(x) := y + DN ◦NNt ◦EN(x),

that is, x∗∈Γt+1 is a unique solution of

ht,y(x) = x
⇐⇒
y = (I −DN ◦NNt ◦EN)(x),
x ∈Γt+1.

Indeed, there is a unique solution because we have for x1, x2 ∈Γt+1, from (162)

∥ht,y(x1) −ht,y(x2)∥X = ∥DN ◦NNt ◦EN(x1) −DN ◦NNt ◦EN(x2)∥X ≤δ∥x1 −x2∥X,

which implies that ht,y : Γt+1 →Γt+1 is a contraction map. Then, we have for x ∈Γt−1,
t = 1, ..., T,
Lt ◦(I −DN ◦NNt ◦EN)(x) = x.
(163)
Here, the solution x∗∈Γt+1 is given by

x∗= lim
n→∞xn,

where
x0 = 0,
xn+1 = y + DN ◦NNt ◦EN(xn).
We define φt : Γt+1 × Γt →Γt+1 × Γt by

φt(x, y) = (y + DN ◦NNt ◦EN(x), y).

Let π1(x, y) = x and e1(y) = (0, y) where π1 : X ×X →X and e1 : X →X ×X are linear maps.
Then,
Lt(y) = x∗= lim
n→∞π1 ◦(⃝n
h=1φt) ◦e1(y).

We define ΦP : ΓT →Γ0 by

ΦP := PΓ0 ◦L1 ◦PΓ1 ◦L2 ◦· · · ◦PΓT −1 ◦LT ,

where PΓt : X →X is the projection onto the convex set Γt = B(0, R′ + tδ). Then, by (163), we
have for x ∈BX(0, R′)
ΦP ◦G(x) = x.
(164)

41


---Page Break---
Therefore, ΦP is the left-inverse of G, that is, G|A(BX(0,R)) : A(BX(0, R)) →X is injective, and
G|A(BX(0,R)) : A(BX(0, R)) →G◦A(BX(0, R)) is bijective, and the inverse (G|G◦A(BX(0,R)))−1 :
G ◦A(BX(0, R)) →A(BX(0, R)) is given by

(G|G◦A(BX(0,R)))−1 = Φ,

where Φ : G ◦A(BX(0, R)) →A(BX(0, R)) is defined by Φ := ΦP |G◦A(BX(0,R)) and has the
form
Φ = L1 ◦L2 ◦· · · ◦LT |G◦A(BX(0,R)).

Therefore, G ◦A : BX(0, R) →G ◦A(BX(0, R)) is also bijective, and the inverse is given by

(G ◦A|BX(0,R))−1 = A−1 ◦Φ.

Note that G ◦A(BX(0, R)) ⊂ΓT .

A.8
Production of Quantitative Universal Approximation Estimates

Here, we show how our framework may be used ‘out of the box’ to produce quantitative approximation
results for discretization of neural operators.

Let X be a separable Hilbert space. We will consider a Hilbert space X, endowed with its norm
topology. Recall that S0(X) ⊂S(X) is a partially ordered lattice of finite dimensional subspaces
of X. Next, we consider quantified discretization of a continuous, possibly non-linear function
F : X →X, that is, how the discretizations FV : V →V can be chosen (using e.g. neural networks
FV having a given architecture for each subspace V ⊂X) so that the obtained discretization operator
AF has the explicitly given error bounds εV .

Using quantitative approximation results for neural networks in Rd, see e.g. [57] or [23], one obtains
quantitative results for neural operators. An example of such result is given below.

Proposition 5. Let r > 0 and F : BX(0, r) →X be a non-linear function satisfying F ∈
Lip(BX(0, r); X), in n = 1, or F ∈Cn(BX(0, r); X), if n ≥2. Let εV > 0 be numbers in-
dexed by the linear subspaces V ⊂X such that εV →0 as V →X. When d = dim(V ), the space V
is identified with Rd using an isometric isomorphism JV : V →Rd. Then there is a feed forward neu-
ral network FV,θ : Rd →Rd with ReLU-activation functions with at most C(n, d) log2((1+r)n/εV )
layers and C(n, d)ε−d/n
V
log2((1 + r)n/εV ) non-zero elements in the weight matrices such that
ANN : F →(FV )V ∈S0(X), where FV = J−1
V
◦FV,θ ◦JV : V →V , is an ⃗ε-approximation
operation in the ball BX(0, r).

Proof. Let Lip(F : BX(0, r) →X) ≤M, in n = 1, or ∥F∥Cn(BX(0,r);X) ≤M, if n ≥2. Let
V ⊂X be a linear subspace of dimension d and εV > 0. Let us use some orthogonal basis of
V to identify V and Rd and denote the identifying isomorphism by J : Rd →V . The function
ˆF : Rd →Rd, given by ˆF = J−1 ◦PV ◦F ◦J satisfies Lip( ˆF : BRd(0, r)) →X) ≤M if n = 1,
or ∥ˆF∥Cn(BRd(0,r)) ≤M if n ≥2. Then, by applying [57, Theorem 1] or [23], we see that there

exists a feed forward neural network FV,θ : Rd →Rd with ReLU-activation functions with at most
C(n, d) log2((1 + r)n/εV ) layers and C(n, d)ε−d/n
V
log2((1 + r)n/εV ) non-zero elements in the
weight matrices such that

∥FV,θ(x) −ˆF(x)∥Rd ≤MεV .
(165)

Due to Def. 1, this yields the claim.

B
Invertible residual network on separable Hilbert spaces

In this section, we consider the approximation of diffeomorphisms by globally invertible residual
networks on Hilbert spaces. From the viewpoint of no-go theorem, as the class of diffeomorphisms is
too large, we focus on the class of strongly monotone C1-diffeomorphisms with compact support5.

5We denote the support of F : X →X by supp(F) := {x ∈X : F(x) ̸= x}.

42


---Page Break---
We first obverse a following similar result with Theorem 3 that strongly monotone C1-diffeomorphism
F : X →X with compact support has the property that any linear discretizations PV F|V are still
strongly monotone C1-diffeomorphism with compact support. The proof can be given by the same
argument in Theorem 3 because F has the form of F = Id + B where B := F −Id is a compact
mapping.

Proposition 6. Let Alin be the discretization functor that maps F to PV F|V for each finite subspace
V ⊂X. Let Dsmc and Bsmc be categories where F : X →X and FV : V →V are strongly
monotone C1-diffeomorphisms with compact support. Then, the functor Alin : Dsmc →Bsmc
satisfies assumption (A), and it is continuous in the sense of Definition 8.

Under the same setting and notations in Section 3.5, we define the class of invertible residual networks
in the separable Hilbert space by, for T, N ∈N and δ ∈(0, 1),

Rinv
T,N,φ,δ,σ(X) :=
n
G ∈RT,N,φ,δ,σ(X) : G = ⃝T
t=1(IdX + DN ◦NNt ◦EN),

LipX→X(DN ◦NNt ◦EN) ≤δ (t = 1, ..., T)
o
,
(166)

which is a subset of Rinv
T,N,φ,δ,σ(X) defined in (10).
The smallness of Lipschitz constants
LipX→X(DN ◦NNt ◦EN) implies that Rinv
T,N,φ,δ,σ(X) is included in the class of homeomor-
phisms. The following lemma shows this fact.
Lemma 9. Let δ ∈(0, 1), and let F ∈Rinv
T,N,φ,δ,σ(X). If σ : R →R is Lipschitz continuous,
then F : X →X is homeomorphism. Moreover, if σ : R →R is C1, then, F : X →X is
C1-diffeomorphism.

Proof. Assume that σ is Lipschitz continuous. Let G ∈Rinv
T,N,φ,δ,σ(X), that is,

G = (IdX + DN ◦NNT ◦EN) ◦· · · ◦(IdX + DN ◦NN1 ◦EN),

where LipX→X(DN ◦NNt ◦EN) ≤δ. It is enough to show that for all t = 1, ..., L

IdX + DN ◦NNt ◦EN : X →X,

is homeomorphism. Indeed, we have for u, v ∈X

⟨(IdX + DN ◦NNt ◦EN)(u) −(IdX + DN ◦NNt ◦EN)(v), u −v⟩X
= ∥u −v∥2
X + ⟨DN ◦NNt ◦EN(u) −DN ◦NNt ◦EN(v), u −v⟩X ≥(1 −δ)∥u −v∥2
X,
(167)

that is, IdX + DN ◦NNt ◦EN : X →X is strongly monotone. By the same argument in Lemma 1,
we can show that (IdX + DN ◦NNt ◦EN) : X →X is coercive. By the Minty-Browder theorem
[10, Theorem 9.14-1], (IdX + DN ◦NNt ◦EN) : X →X is bijective, and then its inverse
exists. Denoting by Ht := IdX + DN ◦NNt ◦EN, we see that by substituting u = H−1(u) and
v = H−1(v) for (167)

∥H−1
t
(u) −H−1
t
(v)∥2
X ≤
1
1 −δ ⟨Ht ◦H−1
t
(u) −Ht ◦H−1
t
(v), H−1
t
(u) −H−1
t
(v)⟩X

≤
1
1 −δ ∥u −v∥X∥H−1
t
(u) −H−1
t
(v)∥X,

which means that its inverse is continuous. Therefore, (IdX + DN ◦NNt ◦EN) : X →X is
homeomorphism.

For the second statement, we assume that σ is C1. By the same argument in Lemma 1, we can show
that the derivative DHt|u : X →X at u ∈X is given by

DHt|u = IX + DN ◦D(NNt)|EN(u) ◦EN,

and it is injective. Since DN ◦D(NNt)|EN(u) ◦EN : X →X is a finite dimensional linear operator,
it is compact operator. Then by the Fredholm theorem, DHt|u : X →X is bijective. By the
inverse function theorem, the inverse H−1
t
: X →X is C1, which implies that Ht : X →X is
C1-diffeomorphism.

43


---Page Break---
In what follow, we employ σ as the GroupSort activation having a group size of 2 (see [3, Section
4]). As sort activation is 1-Lipschitz, from Lemma 9, Rinv
L,N,φ,δ,σ(X) is a subset of the class of
homeomorphisms. We finally show that, in this case, Rinv
L,N,φ,δ,σ(X) is an universal approximator
for the class of the strongly monotone diffeomorphisms with compact support.
Theorem 7. Let R > 0, and let F : X →X be strongly monotone C1-diffeomorphism with compact
support, and let σ be the GroupSort activation having a group size of 2. Then, for any orthonormal
basis {φn}n∈N ⊂X, δ ∈(0, 1), and ϵ ∈(0, 1), there are T, N ∈N, and G ∈Rinv
T,N,φ,δ,σ(X) such
that
sup
u∈BX(0,R)
∥F(u) −G(u)∥X ≤ϵ.

Moreover, the inverse G−1 : X →X of G ∈Rinv
T,N,φ,δ,σ(X) is given by

G−1 = ˜L1 ◦· · · ◦˜LT −1 ◦˜LT ,
(168)

where ˜Lt : X →X is defined by

Lt(y) := lim
n→∞π1 ◦

⃝n
h=1 ˜ϕt

◦e1(y),

where ˜ϕt : X × X →X × X is defined by
˜ϕt(x, y) = (y + DN ◦NNt ◦EN(x), y),

where π1(x, y) = x and e1(y) = (0, y).

Proof. We denote by Diff1(X) the class of C1-diffeomorphisms between X, and Diff1
sm(X) the class
of strongly monotone C1-diffeomorphisms between X. We also denote the support of F ∈Diff1(X)
by supp(F) := {x ∈X : F(x) ̸= x}. We say that F ∈Diff1
sm,c(X) if F ∈Diff1
sm(X) has a
compact support.

Let F ∈Diff1
c,sm(X). We define F1 : X →X by F1 := IdX + PVN (F −IdX)PVN , and we see
that

F1 = IdX + PVN (F −IdX)PVN = PV ⊥
N + PVN FPVN = PV ⊥
N + DNENFDNEN.

We can show that for large N ∈N

sup
u∈BX(0,R)
∥F(u) −F1(u)∥X ≤
sup
u∈BX(0,R)
∥PVN (IdX −F)PVN (u)∥X ≤ϵ,
(169)

as IdX −F : X →X is a compact mapping. By the same argument in Lemma 1, we can show
that FN := ENFDN ∈Diff1
sm,c(RN), and DFN|0 is a positive definite matrix, which is connected
to IdRN . By the similar argument in the proof of Theorem 4, see (105)–(152), we can construct
continuous paths f : [0, 1] →Diff1
sm,c(RN) and β : [0, 1] →GL(RN) with f0 = DFN|0, f1 = FN,
β0 = IdRN , and β1 = DFN|0 such that

FN = (f1 ◦f −1
tm ) ◦(ftm ◦f −1
tm−1) ◦. . .

· · · ◦(ft1 ◦f −1
0 ) ◦(β1 ◦β−1
sm) ◦(βsm ◦β−1
sm−1) ◦· · · ◦(βs1 ◦β−1
0 ),
(170)

where tj and sj are chosen so that 0 = t0 < t1 < . . . tm < tm+1 = 1 and 0 = s0 < s1 < . . . sm <
sm+1 = 1 and that the Lipschitz constant of the maps ftj ◦f −1
tj−1 −IdRN and βsj ◦β−1
j−1 −IdRN
are less that δ.

Then, there is T ∈N such that FN has the form

FN = (IdRN −HT ) ◦(IdRN −HT −1) ◦· · · ◦(IdRN −H1),

where for each t = 1, ..., T,
LipRN→RN (Ht) ≤δ.

Then, remarking that ENDN = IdRN and DNEN = PVN , we see that

F1 = PV ⊥
N + DN(IdRN −HL) ◦· · · ◦(IdRN −H1)EN
= (IdX −DN ◦HT ◦EN) ◦· · · ◦(IdX −DN ◦H1 ◦EN).

44


---Page Break---
For each t = 1, ..., T, by using [3, Theorem 3 and Observation], δ-Lipschitz function Ht : RN →RN
can be approximated by a neural network NNt : RN →RN with GroupSort activation σ having a
group size of 2 in L∞-norm on any compact set, and NNt : RN →RN is δ-Lipschitz continuous.

We denoting by

G := (IdX −DN ◦NNT ◦EN) ◦· · · ◦(IdX −DN ◦NN1 ◦EN) ∈Rinv
T,N,φ,δ,σ(X).

Note that LipX→X(DN ◦NNℓ◦EN) ≤δ. Then, we can show that, by similar way in the first half
of proof of Theorem 5,
sup
u∈BX(0,R)
∥F1(u) −G(u)∥X ≤ϵ.
(171)

With (169) and (171), we obtain that

sup
u∈BX(0,R)
∥F(u)−G(u)∥X ≤
sup
u∈BX(0,R)
∥F(u)−F1(u)∥X +
sup
u∈BX(0,R)
∥F1(u)−G(u)∥X ≤2ϵ.

The representation of inverse G−1 can be given by the same argument in the proof of Theorem 5.

Similarly in Corollary 1, Theorem 7 and Lemmas 10 and 11 have the following corollary.
Corollary
2.
Under
the
same
setting
and
assumptions
with
Corollary
1,
let
RNOinv
T,N,φ,δ,σ(L2(D; R)) be the class of invertible residual neural operators defined
in (180).
Then, the statement replacing X with L2(D; R) and G ∈Rinv
T,N,φ,δ,σ(X) with
G ∈RNOinv
T,N,φ,δ,σ(L2(D; R)) in Theorem 7 holds.

C
Neural operators

C.1
Examples of generalized neural operators

In the main text, we defined generalized neural operators on Hilbert spaces. Here, we give several
examples to show that they are extensions of classical neural operators [33, 30].
Example 1. Let X be a Hilbert space. Let Gℓ: X →X be a classical neural operator having the
form
Gℓ:= (WTℓ+ KTℓ) ◦σ(WTℓ−1 + KTℓ−1) ◦· · · σ(W1 + K1),

where Wℓ: X →X are linear bounded operators corresponding to the local term and Kℓ: X →X
are compact operators corresponding to the non-local term (e.g., integral operators having a smooth
kernel or smooth basis). We assume that activation function σ is C1. Then, we have Gℓ∈C1(X; X).
Let T1 = Klift : X →X and T2 = Kproj : X →X be compact linear operators, which
corresponds to lifting and projection, respectively. We denoting by

Fℓ:= I + T1 ◦Gℓ◦T2,

which is one block of classical neural operators with skip-connection. We also denote by Aℓ= Id
and σ = Id. Then, the generalized neural operator H = FL ◦· · · ◦F1 corresponds to classical
neural operators with skip-connections. In this paper, the skip-connection (ie., the structure of the
identity plus some compact mapping) is so important to preserve bijectivity, discussed in Section 3.
Example 2. We show that classical neural operators , for example

Fclas : u 7→(W2 + K2) ◦(σ(W1u + K1(u)),

see [30, 33], can be written in the form of the generalized neural operators that we consider of the
form
H(x) = Ak ◦σ ◦Fk ◦Ak−1 ◦σ ◦Fk−1 ◦· · · ◦A1 ◦σ ◦F1 ◦A0,

where Aj : X →X are linear operators which may not be bijective and Fj = Id + Tj,1 ◦G ◦Tj,2 :
X →X.

We could start with the observations that for an infinite dimensional Hilbert space X there is a
linear isomorphism Jm,n : Xn →Xm, where Xm = X × · · · × X. The reason for this is that the
cardinality of Hilbert basis of the space X is the same as the cardinality of the Hilbert basis of Xn,
see [26, Theorem 3.5].

45


---Page Break---
First we observe that we can write an operator

H(x) = ˜Ak ◦˜σk ◦˜Fk ◦˜Ak−1 ◦˜σk−1 ◦˜Fk−1 ◦· · · ◦˜A1 ◦˜σ ◦˜F1 ◦˜A0,

where ˜Aj : Xnj →Xnj+1 are linear operators which may not be bijective and ˜Fj = Id +
˜Tj,1 ◦˜G ◦˜Tj,2 : Xnj →Xnj and ˜σk : Xnk →Xnk is ˜σ(x1, x2, . . . , xik, xik+1, . . . , xnk) =
(x1, x2, . . . , xik, σ(xik+1), . . . , σ(xnk)) in the form

H(x) = (J1,nk+1◦˜Ak◦Jnk,1)◦(J1,nk◦˜σ◦˜Fk◦Jnk,1)◦(J1,nk◦˜Ak−1◦Jnk−1,1)◦· · ·◦(J1,n1◦˜A0◦Jn0,1),

where e.g. (J1,nk+1 ◦Ak ◦Jnk,1) : X →X and (J1,nk ◦˜σk ◦˜Fk ◦Jnk,1) : X →X. This means
that in our formalism we can replace e.g. operators Ak by matrices of operators Ak.

Next we go to the second step of the construction:

As an example, let us consider the classical neural operator

Fclas : u →(W2 + K2) ◦σ ◦(W1u + K1(u)),

where W1 and W2 are invertible matrices and K1 and K2 are compact operators, can be written as
an generalized neural operator

u →L4 ◦F3 ◦L2 ◦˜σ ◦L1 ◦F0(u),

where F0 : X →X is a layer of neural operator

F0(v) = v + W −1
1
K1(v),

and L1 : X →X is the invertible linear operator

L1(u) = W1u,

and ˜σ(u) = σ(u) and L2 : X →X × X is a non-invertible linear operator

L2(w) = (w, w),

and and F3 : X × X →X × X is a layer neural operator

F3(w1, w2) = (w1, w2 + W −1
2
K2(w2)),

and L4 : X × X →X is the non-invertible linear operator

L4(u1, u2) = W2(u2 −u1).

Finally, we point out that if a non-invertible activation function σ : X →X satisfies Lip(σ) ≤λ, it
can be written as
u →L2 ◦ˆσ ◦L1(u),
where
L1 : u →(u, u),
and ˆσ is an invertible function

ˆσ : (u1, u2) →(u1, 2λu2 + σ(u2)),

and
L1 : (w1, w2) →w2 −2λw1,
Note that equation 2λu + σ(u) = w can be written as a fixed point equation

u = gw(u) := (2λ)−1w −(2λ)−1σ(u),

where Lip(gw) ≤1/2. Hence, gw : X →X is a contraction and the equation u = gw(u) has a
unique solution for all w by Banach fixed point theorem. Hence, u →2λu + σ(u) is invertible.

Example 3. Let D ⊂Rd be a domain, and let L2(D; R) be the real-valued L2-function space on D,
and let φ = {φn}n∈N ⊂L2(D; R) be an orthonormal basis in L2(D; R). We consider the case when
X = L2(D; Rh) = L2(D; R)h, and Aj = W (j) where W (j) ∈Rh×h are invertible matrices, and
Fj = Id + PV h
N ◦K(j)
N ◦PV h
N W (j)−1) (corresponding to T1 = PV h
N , T2 = PV h
N W (j)−1, G = K(j)
N )

46


---Page Break---
where VN := span{φn}n≤N, and K(j)
N : L2(D; R)h →L2(D; R)h is a finite rank operator defined
by

Kj(u)(x) :=
X

p,q≤N
K(j)
p,q⟨u, φp⟩L2(D;R)φq(x),
x ∈D,

Then, H : L2(D; R)h →L2(D; R)h can be written by

H = (W (k) + K(k)) ◦σ ◦(W (k−1) + K(k−1)) ◦σ ◦· · · ◦(W (1) + K(1)) ◦σ ◦(W (0) + K(0)),

which coincides with classical neural operators (assuming that all local weight matrices are invertible)
used in e.g., [33]. See Definition 9 as well.
Example 4. Let us consider an example of an example of an generalized neural operator which
is obtained by composition of non-linear integral operators and smooth activation functions. Let
D ⊂Rd be be a bounded domain with a smooth boundary. We consider the case when X =
H1(D; Rh) = H1(D; R)h are Sobolev spaces, and Aj = W (j) where W (j) ∈Rh×h are invertible
matrices, and Fj = IdH1 + iH2→H1 ◦˜K(j) ◦iH1→L2W (j)−1 (corresponding to T1 = iH1→L2,
T2 = iH2→H1(W (j))−1, G = ˜K(j)) where ˜K(j) : L2(D; R)h →H1(D; R)h is non-linear integral
operator

˜K(j)(u)(x) :=
Z

D
k(j)(x, y, u(y))u(y)dy,
x ∈D,

where kernel satisfies k(j) ∈C3(D × D × Rh; Rh×h) has uniformly bounded three derivatives.
Also, let σ ∈C1(R) be an activation function which derivative is uniformly bounded and we denote
σ∗f = σ ◦f. Then, H : H1(D; R)h →H1(D; R)h, defined by

H
=
(W (k) + iH2→H1 ◦˜K(j) ◦iH1→L2) ◦σ∗◦(W (k−1) + iH2→H1 ◦˜K(j) ◦iH1→L2) ◦σ∗
◦· · · ◦(W (1) + K(1)) ◦σ∗◦(W (0) + iH2→H1 ◦˜K(j) ◦iH1→L2),

is an generalized neural operator. Here, iH1→L2 and iH2→H1 are compact emending from H1(D)
to L2(D) and H2(D) to H1(D), respectively. Non-linear integral operator in neural operator has
been used in [30].

Example 5. Let us consider an example of a typical neural operator which is a finite composition of
layers of neural operators similar to those introduced in [30, 33], F : X →X of the form

F : u →σ ◦(Wu + T2(G(T1u))),
(172)

where X = Hm(Ω), Ω⊂Rd is a bounded set with a smooth boundary and W : X →X is a linear
operator. In the above, Y = C(Ω) and Z = Cm+1(Ω), where m > d/2. Moreover, G : Y →Z is a
nonlinear integral operator

G(u)(x) =
Z

Ω
kθ(x, y, u(y))u(y)dy,

where kθ, ∂ukθ ∈Cm+1(Ω× Ω× R) is a kernel given by a feed-forward neural network with
sufficiently smooth activation functions. Here, we assume that σj ∈Cℓ(R), ℓ≥m + 2 and that the
kernel kθ is of the form

kθ(x, y, t) =

J
X

j=1
cj(x, y, θ)σj(aj(x, y, θ)t + bj(x, y, θ)).

Moreover, T1 : X →Y and T2 : Z →X are the identical embedding operators

Tj(u) = u.

Due to the choice of function spaces X, Y and Z, the maps T1 and T2 are compact operators.
Summarizing, F = Fσ,W,k, where

Fσ,W,k(u)(x) = σ((Wu)(x) +
Z

Ω
kθ(x, y, u(y))u(y)dy).
(173)

47


---Page Break---
Furthermore, by choosing kθ(x, y, u(y)) = kθ(x −y) and Ω= Td as the convolutional kernel and
the torus, the map F takes the form of an FNO [39].

Here the activation functions σ and σj are in different roles as the functions σj appear inside the
integral operator (or between the compact operators T1 and T2). The function σ is useful in obtaining
universal approximation results for neural operators, but as this question is somewhat technical, we
postpone this discussion to the end.

The appearance of the compact operators T1 and T2 makes the discretization of activation function σ
and the activation functions inside G in Definition 4 different, and this is the reason why we have
introduced both σ and G. To consider invertible neural operators, we will below assume that σ is
an invertible function, for example, the leaky Relu function. In the above operator, the nonlinear
function G inside compact operators in the operation

N : u →u + T2(G(T1u)),

and the compact operators map weakly converging sequences to norm converging sequences. This is
essential in the proofs of the positive results for approximation functors as discussed in the paper.
However, we do not have general results on how the operation

Gσ : u →σ ◦u,

can be approximated by finite dimensional operators in the norm topology, but only in the weak
topology in the sense of Definition 11 of the Weak Approximation Functor. However, one can overcome
this difficulty in two ways. The first way is to use in the discretization a suitable finite dimensional
space V that satisfies Gσ(V ) ⊂V . For example, when X = L2(Ω) and σ is a leaky relu function,
one can choose V to be a space that consists of piecewise constant functions. The second way is
choosing a composition of layers of the form

Nj : u →Wju + T1,j(Gj(T2,ju)),

and
Gσ : u →σ ◦u,
in different finite dimensional spaces Vj. For example, we can consider these operations as maps

N1 : V1 →V1,
(174)
Gσ : V1 →V2 := Gσ(V1),
(175)
N2 : V2 →V2,
(176)
Gσ : V2 →V3 := Gσ(V2).
(177)

This makes the composition
Gσ ◦N2 ◦Gσ ◦N1 : V1 →V3,
well defined. The maps, Gσ, are clearly invertible and one can use Theorems 4 and 5 to analyze
when Nj : Vj →Vj are invertible functions.

Finally, we return to the question whether the activation function σ is useful in universal approxima-
tion results. If the activation function σ is removed (that is, it is the identical map σid : s →s), the
operator F is a sum of a linear operator and a compact nonlinear integral operator,

FW,k(u)(x) = (Wu)(x) +
Z

Ω
kθ(x, y, u(y))u(y)dy.
(178)

Moreover, if we compose above operators Fj of the above form, the obtained operator, G : X →X
is also a sum of a linear operator and a compact operator,

G(u) = ˜Wu + ˜K(u).

Moreover, the Frechet derivative of G at u0, denoted DG|u0 is the map

DG|u0 : v →˜Wv + D ˜K|u0v,

and, due to the above assumptions on kernel kθ(x, y, u), the derivative is a linear operator

D ˜K|u0 : Hm(Ω) →Hm+1(Ω).

48


---Page Break---
By the Sobolev embedding theorem it is a compact operator D ˜K|u0 : X →X. This means that the
Fredholm index of the derivative of DG|u0 is constant

Ind(DG|u0) = Ind(W),

that is, independent of the point u0 where the derivative is computed. In particular, this means that
one cannot approximate an arbitrary C1-function G : X →X in compact subsets of X by neural
operators which are compositions of layers (178). Indeed, for a general C1-function G : X →X
the Fredholm index may be a varying function of u0. Thus, σ appears to be relevant for obtaining
universal approximation theorems for neural operators.

C.2
Residual neural operators

Let D ⊂Rd be a domain. In what follows, we consider the real-valued L2-function space L2(D; R)
6 . Let φ = {φn}n∈N ⊂L2(D; R) be an orthonormal basis in L2(D; R).
Definition 9 (Neural operators [33]). We define a neural operator G : L2(D; R) →L2(D; R) by

G : u0 7→uL+1,

where uL+1 is give by the following steps:

uℓ+1(x) = σ

W (ℓ)uℓ(x) + (K(ℓ)
N uℓ)(x) + b(ℓ)
,
x ∈D,
0 ≤ℓ≤L −1,

uL+1(x) = W (L)uL(x) + (K(L)
N uL)(x) + b(L),
x ∈D,

where σ : R →R is a non-linear activation operating element-wise, and W (ℓ) ∈Rdℓ+1×dℓand
b(ℓ) ∈Rdℓ+1 and

K(ℓ)
N (v)(x) =
X

p,q≤N
K(ℓ)
p,q⟨v, φp⟩L2(D;R)φq(x),
x ∈D,

where K(ℓ)
p,q ∈Rdℓ+1×dℓ(ℓ= 0, ..., L, p, q = 1, ..., N, d0 = dL+2 = 1). Here, we use the notation
for v = (v1, ..., vdℓ) ∈L2(D; R)dℓ

⟨v, φp⟩L2(D;R) =
 
⟨v1, φp⟩L2(D;R), ..., ⟨vdℓ, φp⟩L2(D;R)

∈Rdℓ.

We denote by NOL,N,φ,σ(L2(D; R)) the class of neural operators G : L2(D; R) →L2(D; R)
defined above, with depths L, rank N, orthonormal basis φ, and activation function σ.
Definition 10 (Residual Neural Operator). Let T, N ∈N and let δ ∈(0, 1). We define by

RNOT,N,φ,σ(L2(D; R))

:= {G : L2(D; R) →L2(D; R) : G = (IdL2(D;R) + GT ) ◦· · · ◦(IdL2(D;R) + G1),

Gt ∈NOLt,N,φ,σ(L2(D; R)), Lt ∈N, t = 1, ..., T},
(179)

RNOinv
T,N,φ,δ,σ(L2(D; R)) := {G : L2(D; R) →L2(D; R) :

G = (IdL2(D;R) + GT ) ◦· · · ◦(IdL2(D;R) + G1), Gt ∈NOLt,N,φ,σ(L2(D; R)),

LipL2(D;R)→L2(D;R)(Gt) ≤δ, Lt ∈N, t = 1, ..., T},
(180)

Lemma 10. Let δ ∈(0, 1), and let F ∈RNOinv
T,N,φ,δ,σ(L2(D; R)). Let σ : R →R be Lipschitz
continuous. If σ : R →R is Lipschitz continuous, then F : X →X is homeomorphism. Moreover, if
σ : R →R is C1, then, F : X →X is C1-diffeomorphism.

The proof is given by the same argument in Lemma 9.
Lemma 11. Assume that the orthonormal basis φ include the constant function. Then, we have the
following inclusion:

RT,N,φ,σ(L2(D; R)) ⊂RNOT,N,φ,σ(L2(D; R)),
(181)

Rinv
T,N,φ,δ,σ(L2(D; R)) ⊂RNOinv
T,N,φ,δ,σ(L2(D; R)).
(182)

6We will discuss the function space L2(D; R) for easier reading, but all arguments can be replaced with
real-valued function space U(D; R) that is a separable Hilbert space.

49


---Page Break---
Proof. Let G ∈RT,N,φ,σ(L2(D; R)) such that

G = (IdL2(D;R) + DN ◦NNT ◦EN) ◦· · · ◦(IdL2(D;R) + DN ◦NN1 ◦EN).

Since φ = {φn}n∈N ⊂L2(D; R) has the constant basis, denoting it by φ0 :=
1D(x)
∥1D∥L2(D;R) , where

1D(x) = 1 for x ∈D, we see that for α = (α1, ..., αN) ∈RN

DNα(x) =
X

n≤N
αnφn(x) =
X

n≤N
αn⟨φ1, φ1⟩L2(D;R)φn(x)

=
X

n≤N

1
∥1D∥L2(D;R)
⟨αn · 1D, φ1⟩L2(D;R)φn(x).

We define ˜DN : L2(D; R)N →L2(D; R) by for u ∈L2(D; R)N

˜DNu :=
X

p,q≤N
˜Dp,q⟨u, φp⟩L2(D;R)φq,

where ˜Dp,q ∈R1×N is defined by

˜Dp,q =











0, ..., 0,
1
∥1D∥L2(D;R)
|
{z
}
q−th

, 0, ..., 0

,
p = 1

O,
p ̸= 1.

Then, we have that
DN ◦NNt ◦EN(u) = ˜DN (NNt ◦EN(u) · 1D) .
(183)
Next, we see that

EN(u) =
 
⟨u, φ1⟩L2(D;R), ..., ⟨u, φN⟩L2(D;R)

= ∥1∥L2(D;R)
 
⟨u, φ1⟩L2(D;R), ..., ⟨u, φN⟩L2(D;R)

φ1(x),

We define ˜EN : L2(D; R) →L2(D; R)N by

˜EN(u) :=
X

p,q≤N
˜Ep,q⟨u, φp⟩L2(D;R)φq(x),

where ˜Ep,q ∈RN×1 is defined by

˜Ep,q =









0, ..., 0, ⟨1D, φn⟩L2(D;R)
|
{z
}
p−th

, 0, ..., 0
T
,
q = 1

O,
q ̸= 1.

Then we have that
˜DN (NNt ◦EN(u) · 1D) = ˜DN ◦NNt ◦˜EN(u).
(184)
With (183) and (184), we see that

DN ◦NNt ◦EN = ˜DN ◦NNt ◦˜EN ∈NOLt,N,φ,σ(L2(D; R)),

where Lt ∈N is the depth of NNt. Therefore, G has the form

G = (IdL2(D;R)+ ˜DN◦NNT ◦˜EN)◦· · ·◦(IdL2(D;R)+ ˜DN◦NN1◦˜EN) ∈RNOT,N,φ,σ(L2(D; R)).

(182) can be proved by the same argument.

D
Generalizations

D.1
Generalization of the no-go Theorem 2 using weak topology

We can generalize the no-go Theorem 2 for the case when approximations and continuity of approx-
imations are considered in the weak topology of the Hilbert space X. This can be done when the
approximations of the identity map and the minus one times the identity operator satisfy additional
assumptions and the partially ordered subset S0(X) is the set of all all linear subspaces S(X) of X.

In the case when S0(X) = S(X) and Definition 7 the condition (A) can generalized as follows:

50


---Page Break---
Definition 11 (Weak Approximation Functor). When S0(X) = S(X) we define the weak approx-
imation functor, that we denote by A: D →B, as the functor that maps each (X, F) ∈OD to
some (X, S(X), (FV )V ∈S(X)) ∈OB so that the Hilbert space X stays the same. The approximation
functor maps all morphisms aϕ to Aϕ and morphisms aX1,X2 to AX1,X2, and has the following
properties

(A’) For all r > 0, all (X, F) ∈OD and all y ∈X, it holds that

lim
V →X
sup
x∈BX(0,r)∩V
⟨FV (x) −F(x), y⟩X = 0.
(185)

Moreover, when F : X →X is the operator Id : X →X or −Id : X →X, then FV is
the operator IdV : V →V or −IdV : V →V , respectively.

Moreover, Definition 8 can generalized as follows so that it uses the weak topology.
Definition 12. We say that the approximation functor A is continuous in the weak topology if the
following holds: Let (X, F), (X, F (j)) ∈OD be such that the Hilbert space X is the same for all
these objects and let (X, S(X), (FV )V ∈S(X)) = A(X, F) be approximating sequences of (X, F)
and (X, S(X), (Fj,V )V ∈S(X)) = A(X, F (j)) be approximating sequences of (X, F (j)). Moreover,
assume that for r > 0 and all y ∈X

lim
j→∞
sup
x∈BX(0,r)
|⟨F (j)(x) −F(x), y⟩X| = 0.
(186)

Then, for all V ∈S(X) the approximations F (j)
V
of F (j) and FV of F satisfy for all y ∈X

lim
j→∞
sup
x∈V ∩BV (0,r)
|⟨F (j)
V (x) −FV (x), y⟩X| = 0.
(187)

The theorem below states a negative result, namely that there does not exist continuous approximating
functors for diffeomorphisms.
Theorem 8. (No-go theorem for discretization of general diffeomorphisms) There exists no functor
D →B that satisfies the property (A’) of a weak approximation functor and is continuous in the weak
topology.

The proof of Theorem 8 is analogous to Theorem 2 by replacing A1 by the map −Id and considering
a linear space V ∈S(X) having an odd dimension, in which case deg(A1 : V →V ) = −1. Let
F0 = Id : X →X and F1 = −Id : X →X. Assume that (X, S(X), (Ft,V )V ∈S(X)) = A(X, Ft)
are approximations of the map F : X →X in the weak topology and that A is continuous in the
weak topology. Recall that then Ft,V : V →V are C1-diffeomorphisms that are discretizations of
F : X →X. Then, by condition (A’), F0,V = IdV and F1,V = −IdV so that deg(F0,V ) = 1 and
deg(F1,V ) = −1. Observing that when the condition (187) is applied for y1, y2, . . . , yn ∈V that
form a basis of the space V having the dimension dim(V ) = n, we see that the condition (187)
implies the condition (7). Using these observations, Theorem 8 is follows similarly to the proof of
Theorem 2.

51


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Our abstract and introduction accurately summarize and reflect the paper’s
contributions and scope.
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
Answer: [NA]
Justification: Our paper is theoretical. It does not have limitations in the same was as a more
applied paper. All theorems and propositions are proved to be true, and so are not limited in
their scope.
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

52


---Page Break---
Answer: [Yes]
Justification: The statements of all theoretical statements are given in the main body, and
all of their proofs are included in the attached appendix. A reference to the proof of each
statement is given either immediately before, or immediately after each statement. When
appropriate, a proof sketch is given.

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

Answer: [NA]

Justification: Our paper does not have any numerical experiments.

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

53


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justification: Our paper does not include experiments requiring code.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so No is an acceptable answer. Papers cannot be rejected simply for not
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

Answer: [NA]

Justification: Our paper does not include experiments

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

Justification: Our paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

54


---Page Break---
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

Answer: [NA]

Justification: Our paper does not include experiments.

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

Justification: We follow NeurIPS code of ethics

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

Justification: Our work is fundamental. Our work is not tied to a particular application. To
our knowledge, there are no direct paths to any negative applications.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

55


---Page Break---
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

Justification: Our work is theoretical, does not release data or models.

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

Justification: Our work is theoretical and does not contain owners of outside assets, like
code, data or models.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

56


---Page Break---
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

Justification: Our paper does not release new assets.

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

Justification: Our work does not involve crowdsourcing nor research with human subjects.

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

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

57


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

58


---Page Break---
