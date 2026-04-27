4-bit Shampoo for Memory-Efﬁcient Network Training

Sike Wang
Beijing Normal University
sikewang@mail.bnu.edu.cn

Pan Zhou
Singapore Management University
panzhou@smu.edu.sg

Jia Li†
Beijing Normal University
jiali@bnu.edu.cn

Hua Huang
Beijing Normal University
huahuang@bnu.edu.cn

Abstract

Second-order optimizers, maintaining a matrix termed a preconditioner, are supe-
rior to ﬁrst-order optimizers in both theory and practice. The states forming the
preconditioner and its inverse root restrict the maximum size of models trained by
second-order optimizers. To address this, compressing 32-bit optimizer states to
lower bitwidths has shown promise in reducing memory usage. However, current
approaches only pertain to ﬁrst-order optimizers. In this paper, we propose the
ﬁrst 4-bit second-order optimizers, exempliﬁed by 4-bit Shampoo, maintaining
performance similar to that of 32-bit ones. We show that quantizing the eigenvector
matrix of the preconditioner in 4-bit Shampoo is remarkably better than quantizing
the preconditioner itself both theoretically and experimentally. By rectifying the
orthogonality of the quantized eigenvector matrix, we enhance the approximation
of the preconditioner’s eigenvector matrix, which also beneﬁts the computation
of its inverse 4-th root. Besides, we ﬁnd that linear square quantization slightly
outperforms dynamic tree quantization when quantizing second-order optimizer
states. Evaluation on various networks for image classiﬁcation and natural language
modeling demonstrates that our 4-bit Shampoo achieves comparable performance
to its 32-bit counterpart while being more memory-efﬁcient*.

1
Introduction

Deep neural networks (DNNs) have achieved great success in numerous ﬁelds, e.g., computer
vision [20], natural language processing [38], and speech recognition [16]. A signiﬁcant part of such
success is attributed to ﬁrst-order optimizers such as stochastic gradient descent with momentum
(SGDM) [31] and AdamW [29]. Second-order optimizers, including K-FAC [30], Shampoo [18],
AdaBK [41], CASPR [13], and Sophia [27], show great convergence properties, but often involve
noticeable computation and memory costs. Anil et al. [2] provided several practical techniques
for second-order optimizers to achieve substantial wall-clock time improvements over traditional
ﬁrst-order optimizers. The fast convergence property of second-order optimizers beneﬁts from
preconditioning the gradient with a matrix known as a preconditioner. The optimizer states for
constructing the preconditioner and its inverse root can speed up optimization compared to ﬁrst-order
optimizers, but consume memory that could be used for model parameters, limiting the maximum
model size trained within a given memory budget. With the increase in model size, the memory
utilized by optimizer states can become a predominant factor in memory usage. This is the primary
obstacle hindering the widespread use of second-order optimizers in the era of large models.

There are two main attempts to reduce memory consumed by optimizer states. Factorization uses low-
rank approximation to optimizer states. This strategy has been applied to ﬁrst-order optimizers [35, 3]

*Code is available at https://github.com/Sike-Wang/low-bit-Shampoo.
†Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
0
50
100
150
200
250
300
Wall-clock Time (min)

50

55

60

65

70

75

80

Test Accuracy (%)

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

800

1000

1200

1400

1600

1800

2000

2200

GPU Memory Cost (MB)

1465.8

2036.0

1543.9
1543.9

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

(a) Swin-Tiny on CIFAR-100

0
500
1000
1500
2000
Wall-clock Time (min)

30

40

50

60

70

Test Accuracy (%)

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

9000

9500

10000

10500

11000

11500

12000

12500

GPU Memory Cost (MB)

10600

12134

10803
10804

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

(b) ViT-Base/32 on ImageNet-1k

Figure 1: Visualization of test accuracies and total GPU memory costs of vision transformers. 4-bit
Shampoo (naive) quantizes the preconditioner, while 4-bit Shampoo (our) quantizes its eigenvector
matrix.

and second-order optimizers [14, 40]. In a comparable but distinct line of work, quantization utilizes
low-bit to compress 32-bit optimizer states. Quantization is attractive due to its simplicity and wide
applicability, which has been applied to ﬁrst-order optimizers [8, 26]. Applying quantization to
second-order optimizers poses a greater challenge, as ﬁrst-order optimizers’ states are elementwise,
whereas second-order optimizers rely on matrix operations. To our knowledge, it has not been
attempted before.

Contributions: In this paper, we present the ﬁrst second-order optimizers with 4-bit optimizer states
by taking Shampoo [18] as an example, while preserving the performance achieved with 32-bit
optimizer states. While our focus is on Shampoo, we believe that our approach could also be applied
to other second-order optimizers (see Table 4). Our main contributions are highlighted below.

Firstly, to maintain 32-bit performance, we propose quantizing the eigenvector matrix of a precondi-
tioner in 4-bit Shampoo, rather than the preconditioner itself. The reason is that the small singular
values of the preconditioner matter. Directly quantizing the preconditioner via block-wise quantiza-
tion [8] at 4-bit precision can signiﬁcantly alter the small singular values, leading to a drastic change
in its inverse 4-th root and thus harming 4-bit Shampoo’s performance. Quantizing the eigenvector
matrix can help alleviate this issue, which is supported by experimental validation and theoretical
insight. Additionally, with the eigenvector matrix, computing the inverse 4-th root is straightforward,
ensuring that quantizing the eigenvector matrix does not lead to a rise in the total wall-clock time
compared to quantizing the preconditioner (see Figure 1).

Secondly, we present two techniques for enhancing performance. As the eigenvector matrix of a
preconditioner is orthogonal, we apply Björck orthonormalization [4] to rectify the orthogonality of
the quantized eigenvector matrix, leading to improved approximation of preconditioner’s eigenvector
matrix and facilitating computation of its inverse 4-th root. Additionally, we observe that linear square
quantization outperforms dynamic tree quantization [7] marginally when quantizing second-order
optimizer states. The superiority of our developed 4-bit Shampoo is demonstrated in Figure 1.

Finally, we evaluate our 4-bit Shampoo on different image classiﬁcation and natural language
modeling tasks using convolutional neural network (CNN) and transformer architectures. Across all
these benchmarks, our 4-bit Shampoo achieves similarly fast convergence comparable to its 32-bit
counterpart, with no signiﬁcant increase in losses for the trained models. Our 4-bit Shampoo uses
less memory than its 32-bit counterpart, allowing for training of larger models with given resources.

2
Preliminaries

In this section, we present Shampoo and its implementation in our experiments. We also discuss
quantization-based compression methods in a general formulation.

Notations. We use a non-bold letter like a or A to denote a scalar, a boldfaced lower-case letter like
a to denote a vector, and a boldfaced upper-case letter such as A to denote a matrix. u=[ui]T means
that the i-th element of column vector u is ui and U =[ui] means the i-th column vector of matrix U
is ui. Let A be a positive deﬁnite (PD) matrix and s ∈R, we deﬁne As =UΛsU T, where UΛU T
is the Singular Value Decomposition (SVD) of A. tr(A) represents the trace of a matrix A. The
inner product of two matrices A and B is denoted as ⟨A, B⟩=tr(ATB). The Frobenius norm of a
matrix A is ∥A∥F =
p

⟨A, A⟩. A ⊙B means the elementwise matrix product (Hadamard product).

2


---Page Break---
Diag(a) is a diagonal matrix with diagonal vector a, while diag(A) means the diagonal vector of
matrix A.

2.1
Shampoo for Matrices

The update rule of Shampoo in the matrix case combined with a ﬁrst-order optimizer F is

Shampoo(Wt−1, Lt−1, Rt−1, st−1, Gt) =
















Lt =Lt−1+GtGT
t
Rt =Rt−1+GT
t Gt
b
Gt =L−1/4
t
GtR−1/4
t
e
Gt = b
Gt(∥Gt∥F /∥b
Gt∥F )
Wt, st =F(Wt−1, st−1, e
Gt)

(1)

where Wt is the model parameters in matrix form, Lt and Rt are called preconditioners, st is the
optimizer state of F, and Gt is the gradient at Wt−1. Note that Lt, Rt, L−1/4
t
, and R−1/4
t
are PD
matrices. The penultimate step in (1) is the grafting trick [1], which enables Shampoo to roughly
apply the well-tuned learning rate schedule of F. The optimization variable Wt does not represent
all model parameters. It denotes a tensor of the model [18] or one block of a tensor [2]. In practice,
we adopt an efﬁcient and effective implementation of Shampoo for training DNNs following [2, 41]
as described in Algorithm 4. In order to achieve efﬁcient training, Lt, Rt, L−1/4
t
, and R−1/4
t
are
computed once every few hundred iterations. In this case, besides Lt and Rt, their inverse 4-th roots
should also be stored in memory, as computing them is computationally expensive. So training large
models with Shampoo can be memory-intensive, consuming a signiﬁcant amount of memory.

2.2
Quantization-based Compression Methods

Quantizing updated optimizer states using a quantizer and then dequantizing them with a dequantizer
prior to use is an effective method for conserving memory. We focus exclusively on vectors, as
tensors can be reshaped into vectors.

Quantization. According to the idea in [8, 26], a b-bit quantizer Q for p-dimensional real vectors is
a mapping given by
Q = (I ◦N, M) : Rp →Tp
b × Rp,
where N is a normalization operator on Rp, I is an elementwise function mapping any real number
to an element of Tb ={0, 1, . . . , 2b−1}, and M is a maximum operator on Rp. For any x ∈Rp, N
and M satisfy N(x)⊙M(x)=x.

A normalization operator N for p-dimensional vectors is a transformation on Rp. It scales each
element of a vector x ∈Rp into [−1, 1]. A block-wise normalization operator for a p-dimensional
vector x = [x1, x2, . . . , xp]T is deﬁned as

N(x)i =
xi
maxj∈Xi{xj},

where N(x)i is the i-th element of N(x), and Xi is a set satisfying i ∈Xi ⊂{1, . . . , p}. Usually,
Xi should also satisfy Xi =Xj or Xi ∩Xj =∅for i, j ∈{1, . . . , p}. In this case, for any x ∈Rp, the
number of different elements in M(x) is equal to the number of elements in set {Xi|i = 1, . . . , p}.
Meanwhile, the number of the elements in Xi for any i should be as close as possible to a value called
block size.

The mapping I for x ∈R in a b-bit quantizer Q is deﬁned as
I(x) = argmin
j∈Tb
|x −R(j)| ,

where R named quantization mapping is an elementwise function that maps any element in Tb
into [−1, 1], and | · | is the absolute operator for a scalar. There are three typical quantization
mappings: linear quantization, dynamic quantization, and quantile quantization. Their speciﬁcations
and visualizations can be found in [8].

Dequantization. Given a b-bit quantizer Q=(I ◦N, M) for a p-dimensional real vector x ∈Rp,
the corresponding dequantizer D is a mapping deﬁned as
D(Q(x))=D(I ◦N(x), M(x))=R(I ◦N(x)) ⊙M(x) : Tp
b × Rp →Rp.

3


---Page Break---
3
Methodology

In this section, we describe the design of our quantization-based compression method to realize 4-bit
Shampoo with fast and high precision quantization. Let Q=(I ◦N, M) be a quantizer and D be its
corresponding dequantizer as described in Subsection 2.2.

3.1
Quantizing the Eigenvector Matrices

A naive approach to realize 4-bit Shampoo is applying the compression methods proposed in [8, 26]
to Lt, Rt, L−1/4
t
, and R−1/4
t
in Shampoo (see (1)). A slightly improved approach is to quantize the
four PD matrices excluding their diagonal elements, which are typically much larger than their non-
diagonal counterparts due to the non-negativity of the elements in diag(GtGT
t ) and diag(GT
t Gt).

However, the naive approach can cause large quantization errors at 4-bit precision. This is because
the quantization errors (or called perturbations) of quantizing Lt and Rt will transfer to L−1/4
t
and R−1/4
t
. To verify this, we ﬁrst introduce two criteria to evaluate the quantization errors of
matrices. We do not use the elementwise criterion in [8]. Let A denote a 32-bit matrix, g represent a
transformation (can formed by quantization), and f stand for a mapping, e.g., f(A)=A−1/4. Then
we deﬁne the normwise relative error (NRE) and angle error (AE) in f of g at A as

NRE= ∥f(A) −f(g(A))∥F

∥f(A)∥F
,
AE=arccos

⟨f(A), f(g(A))⟩
(∥f(A)∥F ∥f(g(A))∥F


.

We choose two PD matrices of order 1200. The ﬁrst one A1 is derived from the real world. It
is a preconditioner in 32-bit Shampoo combined with AdamW for training a Swin-Tiny model.
The second one A2 =UΛU T is synthetic, constructed from a random orthogonal matrix U and a
diagonal matrix Λ with only two distinct diagonal values. Table 1 shows the quantization errors
in f(A) = A−1/4 of the naive approach at these two matrices, which are remarkably high. More
analyses are given in Appendix D. The key point is that the singular values of Ai(i=1, 2) follow a
speciﬁc distribution (see Figure 2). In this scenario, a slight perturbation of Ai will signiﬁcantly alter
its small singular values, resulting in a drastic change to A−1/4
i
.

To address this issue, we propose quantizing the eigenvector matrix of a preconditioner in Shampoo,
rather than the preconditioner itself. Namely, a preconditioner A is a PD matrix, and its SVD is
UΛU T, where U represents the eigenvector matrix and Λ denotes the singular value matrix. Given
that Λ is a diagonal matrix, we can focus on quantizing U using Q while leaving Λ unchanged.
From Table 1, one can observe that quantizing U can signiﬁcantly reduce the quantization errors.
We will theoretically discuss the advantages of quantizing U compared to quantizing A in Section 4.
In practice, the randomized SVD method [19] is adopted to compute the SVD of A efﬁciently, as
shown in [40]. We want to highlight that quantizing the original Lt and Rt in Shampoo involves
signiﬁcant computational burdens to compute their inverse 4-th roots L−1/4
t
and R−1/4
t
, whereas
quantizing the eigenvector matrices of Lt and Rt allows for rapid inverse root calculation. So the
computational time required for both approaches is comparable (see Figure 1).

Table 1: Quantization errors in A−1/4 of different quantization schemes at a PD matrix A. We
employ block-wise normalization with a block size of 64. U is the eigenvector matrix of A, QM =
quantized matrix, and OR = orthogonal rectiﬁcation.

Real-world A=A1
Synthetic A=A2
Mapping R Bit QM OR NRE ↓AE (◦) ↓
Mapping R Bit QM OR NRE ↓AE (◦) ↓

DT

8
A

0.2192
8.3014

DT

8
A

0.1896
10.877
4
A

0.6241
17.319
4
A

0.4615
17.189
4
U

0.0709
4.0426
4
U

0.1224
7.0144
4
U

0.0455
2.5615
4
U

0.0878
4.9960

Linear-2

8
A

0.2164
7.9751

Linear-2

8
A

0.1310
7.4717
4
A

0.6243
17.293
4
A

0.4465
15.338
4
U

0.0543
3.1066
4
U

0.0942
5.3998
4
U

0.0343
1.9456
4
U

0.0669
3.8166

4


---Page Break---
0
200
400
600
800 1000 1200
Index

11

10

9

8

7

6

5

Singular Value

real
quan

(a) Real-world

0
200
400
600
800 1000 1200
Index

5

4

3

2

1

0

Singular Value

real
quan

(b) Synthetic

Figure 2: Singular value distributions of PD matrices (real)
and their 4-bit compressions (quan) used in Table 1 with
R=DT, QM=A. Singular values are shown on a log10 scale.

0
1
2
3
4
5
Number of Iterations t2

6

5

4

3

2

1

0

1

2

Mean Error

s =
1
s =
1/4
s =
1/10
s =
1/20

Figure 3:
Elementwise mean errors
between
(Vt2ΛsV T
t2 )−1/s(Vt2ΛV T
t2 )
and identity matrix I. Mean errors
are shown on a log10 scale.

3.2
Rectifying the Orthogonality of Eigenvector Matrices

Let A be a PD matrix with SVD UΛU T. Note that the eigenvector matrix U is orthogonal, whereas
V =D(Q(U)) may not be. To further mitigate the quantization errors mentioned in Subsection 3.1,
we propose employing Björck orthonormalization [4] to orthogonalize V . Particularly, given V0 =V ,
we iterate

Vt =1.5Vt−1−0.5Vt−1V T
t−1Vt−1,
(2)

for t1 ≥1 times and take Vt1 as the rectiﬁed result. Equation (2) can also be interpreted as the gradient
descent of problem minV ∥V TV −I∥2
F using a step size of 0.5, where I denotes the identity matrix.
We empirically ﬁnd that only one iteration (i.e., t1 =1) is enough. Table 1 illustrates the beneﬁt of
rectifying V into V1.

The update frequencies for the preconditioners and their inverse 4-th roots differ (see Algorithm 3).
Given V and Λ, we also require orthogonal rectiﬁcation to compute As rapidly for any s ∈R.
The reason is as follows. It is easy to compute As = UΛsU T by deﬁnition. However, UΛsU T
can be very sensitive to the orthogonality of U for s < 0, making V ΛsV T largely deviate from
(V ΛV T)s ≈As. Similarly, we can approximate As by Vt2ΛsV T
t2 , where Vt2 is generated by (2).
Figure 3 illustrates the elementwise mean errors between (Vt2ΛsV T
t2 )−1/s(Vt2ΛV T
t2 ) and I for
various s and t2, where A is the real-world matrix used in Table 1. Based on the observation from
Figure 3, we set t2 =4 in our experiments.

3.3
Selecting the Quantizer

The quantizer Q is deﬁned by the normalization operator N and mapping R, and N is determined
by Xi. Since an eigenvector has a unit length, the elements in Xi should belong to the same column
of an eigenvector matrix, i.e., they are from the same eigenvector. Instead of employing dynamic tree
(DT) quantization as mapping R, we recommend utilizing linear square (Linear-2) quantization as R,
particularly when b=4. Linear-2 quantization is deﬁned as

R(j) =








−
 
−1 + 2j/(2b−1)
2 ,
j <2b−1−1;
0,
j =2b−1−1;
 
−1+2j/(2b−1)
2 ,
j >2b−1−1,
(3)

where j ∈Tb ={0, 1, . . . , 2b−1}. As shown in Table 1, Linear-2 quantization has lower quantization
errors compared to DT quantization at 4-bit precision.

3.4
Overall Algorithm

We ﬁrst describe the update processes of the preconditioners and their inverse 4-th roots in our 4-bit
Shampoo. A preconditioner A is a PD matrix and its SVD is UΛU T. We can compress A into a
pair (λ, U) = (diag(Λ), Q(U)) and decompress it into (Λ, V ) = (Diag(λ), D(U)). Algorithm 1
(Preconditioner Update, PU) shows the update rule of A. Similarly, we compress b
A ≈A−1/4 into a
pair (a, A) = (diag( b
A), Q( b
A−Diag(a))) and decompress it into Diag(a) + D(A). Algorithm 2

5


---Page Break---
(Preconditioner’s Inverse 4-th Root Update, PIRU) gives the update rule of b
A. Based on the above
update rules, we can summarize our 4-bit Shampoo in Algorithm 3. Note that we omit some input
parameters of PU and PIRU because they can be found in Algorithm 3 in the same form.

Algorithm 1 PU(λ, U, M)
Input: singular value vector λ, quantized eigen-
vector matrix U, M, number of iterations
t1 for rectiﬁcation, exponential decay rate
β ∈(0, 1), Q and D
1: Λ = Diag(λ), V = D(U)
2: Rectify V by iterating (2) t1 times
3: A = βV ΛV T + (1−β)M
4: Compute A = P ΣP T by randomized SVD
5: return diag(Σ), Q(P )

Algorithm 2 PIRU(λ, U)
Input: singular value vector λ, quantized eigen-
vector matrix U, number of iterations t2 for
rectiﬁcation, dampening term ϵI, Q and D

1: Λ = Diag(λ), V = D(U)
2: Rectify V by iterating (2) t2 times
3:
b
A = V (Λ + max{λ}ϵI)−1/4V T

4: a = diag( b
A)
5: return a, Q( b
A −Diag(a))

Algorithm 3 Practical 4-bit Shampoo

Input: W0 ∈Rm×n, L0 = ϵIm, R0 = ϵIn, bL0 = Im, b
R0 = In, β ∈(0, 1), t1, t2, update interval
T1, update interval T2, total number of steps T, ﬁrst-order optimizer F, ﬁrst-order optimizer
state s0 = 0, 4-bit quantizer Q and its corresponding dequantizer D.
Output: ﬁnal parameter WT .

1: λ0,L = diag(L0), U 0,L = Q(Im);
λ0,R = diag(R0), U 0,R = Q(In)

2: l0 = diag(bL0), L0 = Q(0);
r0 = diag( b
R0), R0 = Q(0)
3: for t = 1, 2, . . . , T do
4:
Receive loss function Lt : Rm×n 7→R and compute gradient Gt = ∇Lt(Wt)
5:
if t%T1 ≡0 then
6:
λt,L, U t,L =PU(λt−1,L, U t−1,L, GtGT
t ); λt,R, U t,R =PU(λt−1,R, U t−1,R, GT
t Gt)
7:
else
8:
λt,L, U t,L = λt−1,L, U t−1,L;
λt,R, U t,R = λt−1,R, U t−1,R
9:
if t%T2 ≡0 then
10:
lt, Lt = PIRU(λt,L, U t,L);
rt, Rt = PIRU(λt,R, U t,R)
11:
else
12:
lt, Lt = lt−1, Lt−1;
rt, Rt = rt−1, Rt−1
13:
bLt = Diag(lt) + D(Lt);
b
Rt = Diag(rt) + D(Rt)
14:
b
Gt = bLtGt b
Rt;
e
Gt = b
Gt(∥Gt∥F /∥b
Gt∥F )
15:
Wt, st = F(Wt−1, st−1, e
Gt)

4
Theoretical Analysis

In this section, we analyze why quantizing the eigenvector matrix of a preconditioner in Shampoo is
better than quantizing the preconditioner itself under a certain singular value distribution. Furthermore,
we consider quantization as a perturbation and prove the convergence of the perturbed Shampoo
(Algorithm 6) in Appendix E. The following lemma reveals some good properties of perturbing the
eigenvector matrix of a PD matrix.
Lemma 1. Let A be a PD matrix whose SVD is UΛU T, where U =[ui] is an orthogonal matrix
and Λ=diag([λi]T) is a diagonal matrix. Given a perturbation ∆U =[∆ui] and s ∈R, we deﬁne
B :=(UΛU T)s and ∆B :=((U +∆U)Λ(U +∆U)T)s−B.

(1) If U +∆U is orthogonal and there exists α ∈R such that ∥∆ui∥2 ≤α, then
∥∆B∥F

∥B∥F
≤2α.

(2) If U +∆U is orthogonal and there exists β ∈R such that ⟨ui, ui+∆ui⟩≥1−β ≥0, then
⟨B, B+∆B⟩
∥B∥F ∥B+∆B∥F
≥(1−β)2.

6


---Page Break---
From Lemma 1, it is evident that the normwise relative error and angle error in f(A) = As of
perturbing U at A = UΛU T are independent of Λ and s. Moreover, these errors are well-bounded
under some mild conditions. Empirically, for 4-bit quantization, α = 0.1 and β = 0.005 roughly
meet the conditions of Lemma 1, leading to ∥∆B∥F

∥B∥F
≤0.2 and
⟨B,B+∆B⟩
∥B∥F ∥B+∆B∥F ≥0.99.

It is very complicated to generally analyze the perturbation in f(A) = As of perturbing A. Thus, we
focus on perturbing the singular values of A. For simplicity, we assume that both A and A + ∆A
have only two distinct singular values, where ∆A is a perturbation of A. The following lemma gives
the perturbation in As of perturbing the smaller singular value of A.
Lemma 2. Let A be a PD matrix of order m+n whose SVD is UΛU T, where m, n ∈N+, n = lm,
U = [ui] is an orthogonal matrix and Λ = diag([λi]T) is a diagonal matrix. Assume that Λ =
diag([cλ1T
m×1, λ1T
n×1]T), c ≥1, and λ > 0. Given a perturbation ∆Λ = diag([0T
m×1, ∆λT
n×1]T)
and s ∈R, we deﬁne B :=(UΛU T)s and ∆B :=(U(Λ+∆Λ)U T)s−B.

(1) If ∆λn×1 = (k −1)λ1n×1 where k > 0, then

∥∆B∥F

∥B∥F
=

√

l|ks −1|
√

c2s + l
= h1(s, l).

Moreover, h1(s, l) decreases monotonically with s over (−∞, 0) and increases monotonically
with l over (0, +∞).
(2) If ∆λn×1 = (tc −1)λ1n×1 where t > 0, then

⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
lts + cs
p

(1 + lt2s)(l + c2s)
= h2(l).

Moreover, h2(l) decreases monotonically with l over (0, (c/t)s] and increases monotonically
with l over ((c/t)s, +∞).
(3) If ∆λn×1 = (tc −1)λ1n×1 where k = tc > 0 and l = (c/t)s, then

∥∆B∥F

∥B∥F
= |ks −1|
√

ks + 1,
⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
2
p

2 + ks + 1/ks .

Let us make some comments on the above lemma. First, from Lemma 2(1) we have h1(1, l) =
∥∆A∥F

∥A∥F
=

√

l|k−1|
√

c2+l . If k ≥1, ∥∆A∥F

∥A∥F
= ∥∆Λ∥F

∥Λ∥F
is bounded by k

c
√

l = t
√

l. Second, if k = tc ≥1

and s < 0, one can deduce h2(l) ≥
p

lt2s/(1 + lt2s) from Lemma 2(2), which indicates that a small
lt2s is needed to achieve small h2(l). We can set t = 0.02 to simulate 4-bit quantization. Based on
Lemma 1 and Lemma 2(3), we have the following proposition.
Proposition 1. Let A be a PD matrix of order m+n whose SVD is UΛU T, where m, n ∈N+,
n = lm, U = [ui] is an orthogonal matrix, Λ = diag([cλ1T
m×1, λ1T
n×1]T), c ≥1000, and λ > 0.
Given ∆U = [∆ui], ∆Λ = diag([0T
m×1, ∆λT
n×1]T), and s ≤−0.25, we deﬁne B := (UΛU T)s,
B1 := ((U + ∆U)Λ(U +∆U)T)s, and B2 := (U(Λ+∆Λ)U T)s. If U + ∆U is orthogonal,
∥∆ui∥2 ≤0.1, ⟨ui, ∆ui⟩≥−0.005, ∆λn×1 =(0.02c−1)λ1n×1, and l=(c/0.02)s, then

2∥B1 −B∥F

∥B∥F
≤0.4≤∥B2 −B∥F

∥B∥F
,
6

1 −
⟨B, B1⟩
∥B∥F ∥B1∥F


≤0.06≤

1 −
⟨B, B2⟩
∥B∥F ∥B2∥F


.

Proposition 1 requires very strong assumptions. Nevertheless, it provides insight into why quantizing
A can result in a greater normwise relative error and angle error in As, compared to quantizing U.
Complete proofs of Lemma 1, Lemma 2, and Proposition 1 can be found in Appendix F.

5
Experiments

In this section, we compare our 4-bit Shampoo combined with SGDM or AdamW to their 32-bit
counterparts, as well as the ﬁrst-order optimizers on various image classiﬁcation tasks. See more
experimental results on image classiﬁcation and natural language modeling tasks in Appendix H.

Models, datasets, and hyperparameters. We train VGG19 [36], ResNet34 [20], ViT-Small [10],
and Swin-Tiny [28] on the CIFAR-100 [23] and Tiny-ImageNet [24] datasets with one RTX3060Ti
GPU, and train ResNet50 and ViT-Base/32 on the ImageNet-1k dataset [34] with one A800 GPU.

7


---Page Break---
Table 2: Performance, wall-clock time and memory cost on various image classiﬁcation tasks. TA =
test accuracy, WCT = wall-clock time, and TMC = total GPU memory cost.

Dataset
Model
Optimizer
TA (%)
WCT (min)
TMC (MB)

CIFAR-100

VGG19

SGDM
74.14
97.70
512.17
SGDM + 32-bit Shampoo
74.54
84.45
979.13
SGDM + 4-bit Shampoo
74.74
92.51
577.14

ResNet34

SGDM
78.98
170.1
822.03
SGDM + 32-bit Shampoo
79.71
147.2
1441.8
SGDM + 4-bit Shampoo
79.17
155.8
908.40

ViT-Small

AdamW
74.34
668.1
2720.0
AdamW + 32-bit Shampoo
77.50
498.7
3252.0
AdamW + 4-bit Shampoo
77.22
510.8
2791.7

Swin-Tiny

AdamW
76.69
318.6
1465.8
AdamW + 32-bit Shampoo
79.34
260.8
2036.0
AdamW + 4-bit Shampoo
78.63
273.3
1543.9

Tiny-ImageNet

VGG19

SGDM
61.53
172.0
1062.3
SGDM + 32-bit Shampoo
63.39
136.5
1531.9
SGDM + 4-bit Shampoo
62.84
143.8
1127.3

ResNet34

SGDM
67.10
432.1
2304.0
SGDM + 32-bit Shampoo
67.90
313.0
2924.3
SGDM + 4-bit Shampoo
67.95
329.3
2390.4

ViT-Small

AdamW
54.66
1274
2730.1
AdamW + 32-bit Shampoo
57.11
953.9
3261.1
AdamW + 4-bit Shampoo
57.15
970.3
2801.9

Swin-Tiny

AdamW
58.77
701.9
1789.9
AdamW + 32-bit Shampoo
61.74
565.3
2362.8
AdamW + 4-bit Shampoo
62.24
582.7
1868.1

ImageNet-1k

ResNet50

SGDM
76.70
2134
11307
SGDM + 32-bit Shampoo
77.07
1910
11937
SGDM + 4-bit Shampoo
76.92
1970
11396

ViT-Base/32

AdamW
72.87
2190
10600
AdamW + 32-bit Shampoo
75.03
1774
12134
AdamW + 4-bit Shampoo
74.78
1770
10804

0
25
50
75 100 125 150 175
Wall-clock Time (min)

50

55

60

65

70

75

80

Test Accuracy (%)

ResNet34 on CIFAR-100

SGDM
SGDM+32-bit Shampoo
SGDM+4-bit Shampoo

0
100 200 300 400 500 600 700

Wall-clock Time (min)

50

55

60

65

70

75

80

Test Accuracy (%)

ViT-Small on CIFAR-100

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo

0
500
1000
1500
2000
Wall-clock Time (min)

30

40

50

60

70

80

Test Accuracy (%)

ResNet50 on ImageNet-1k

SGDM
SGDM+32-bit Shampoo
SGDM+4-bit Shampoo

0
500
1000
1500
2000
Wall-clock Time (min)

30

40

50

60

70

Test Accuracy (%)

ViT-Base/32 on ImageNet-1k

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo

Figure 4: Visualization of test accuracies on the CIFAR-100 and ImageNet-1k datasets.

For hyperparameter settings, we mainly follow [41] to train CNNs and [25, 44] to train vision
transformers. For all the tasks, we keep the common hyperparameters of optimizers the same values.
See Appendix G for experimental details.

Main results. We show the performance, wall-clock time, and memory cost in Table 2. First-
order optimizers run 1.2x to 1.5x epochs, resulting in longer wall-clock time, yet yielding lower
test accuracies compared to second-order optimizers. In comparison to 32-bit Shampoo, our 4-bit
Shampoo shows comparable test accuracies with differences ranging from -0.7% to 0.5%, increases
in wall-clock time varying from -0.2% to 9.5%, and memory savings of 4.5% to 41%. Compared to
the ﬁrst-order optimizers, the memory costs of our 4-bit Shampoo only rise by 0.8% to 12.7%. This
represents a signiﬁcant advancement in the utilization of second-order optimizers. Following [26],
we report the total peak GPU memory consumption rather than the optimizer’s peak GPU memory
consumption. Our main focus is on quantizing the states for constructing preconditioners and their
inverse roots, which are approximately 7x smaller for 4-bit Shampoo compared to 32-bit Shampoo
(see Appendix G). Figure 4 shows the test accuracy curves on the CIFAR-100 and ImageNet-1k

8


---Page Break---
Table 3: Ablation study on the impact of different quantization techniques to Swin-Tiny training on
the CIFAR-100 dataset. U is the eigenvector matrix of a preconditioner A. QM = quantized matrix,
OR = orthogonal rectiﬁcation in Algorithm 1, TL = training loss, and TA = test accuracy.

4-bit
3-bit
Mapping R
QM
OR
TL
TA (%)
Mapping R
QM
OR
TL
TA (%)
Linear-2
A

1.631
76.95
Linear-2
A

1.648
76.70
DT
U

1.569
78.70
DT
U

NaN
-
Linear-2
U

1.566
78.22
Linear-2
U

NaN
-
Linear-2
U

1.551
78.63
Linear-2
U

1.572
78.53

datasets. The test accuracy curves of 4-bit Shampoo and 32-bit Shampoo are very close, both of
which are above the test accuracy curves of the ﬁrst-order optimizers.

Ablations. We investigate the effectiveness of our proposed quantization techniques. Table 3 indicates
that quantizing the eigenvector matrix of a preconditioner is crucial for b-bit (b = 3, 4) Shampoo to
maintain 32-bit performance, and orthogonal rectiﬁcation is highly beneﬁcial for 3-bit Shampoo. As
for quantization mapping, linear square (Linear-2) quantization is comparable to dynamic tree (DT)
quantization. We further apply our 4-bit quantization techniques to K-FAC [30], AdaBK [41] and
CASPR [13] and the results are shown in Table 4. We can see that the 4-bit optimizers match the
performance of their 32-bit counterparts, and reduce memory by over 20%.

6
Related Work
Table 4:
Performance and memory cost of
training Swin-Tiny on CIFAR-100. TA = test
accuracy and TMC = total GPU memory cost.

Optimizer
TA (%) TMC (MB)
AdamW+32-bit K-FAC 78.20
2388.0
AdamW+4-bit K-FAC
78.56
1878.3
32-bit AdamW_BK
79.28
2388.0
4-bit AdamW_BK
79.34
1878.3
AdamW+32-bit CASPR 78.82
2034.6
AdamW+4-bit CASPR
78.80
1543.9

Second-order optimizers. Different second-order
optimizers apply different second-order information.
Hessian-based optimizers [39, 27] use the Hessian
matrix or its approximation. Fisher-based optimiz-
ers [30, 41] utilize the covariance matrix of the ac-
cumulated gradients or its approximation based on
Kronecker product. Shampoo [18] and CASPR [13]
approximate the full AdaGrad [12] preconditioner
by a set of small preconditioning matrices.

Memory efﬁcient optimizers based on factorization. Adafactor [35] employs the outer product
of two vectors to approximate the second moment of Adam [22]. SM3 [3] considers approximating
the second moment of Adam by its covers’ statistics. [14] and [40] reduce memory cost of the
preconditioner in a second-order optimizer with its low-rank approximation through truncated SVD.

Memory efﬁcient optimizers based on quantization. Dettmers et al. [8] introduce block-wise
dynamic quantization that enables the use of ﬁrst-order optimizers with 8-bit states. Li et al. [26]
push the optimizer states of Adam/AdamW to 4-bit.

7
Conclusions, Limitations, and Broader Impact

We propose 4-bit Shampoo, the ﬁrst low-bit second-order optimizer, designed for memory-efﬁcient
training of DNNs. We ﬁnd that quantizing the eigenvector matrix of the preconditioner is essential
to minimize quantization errors in its inverse 4-th root at 4-bit precision, given its sensitivity to
alterations in small singular values. We further introduce orthogonal rectiﬁcation and linear square
quantization mapping to improve performance. 4-bit Shampoo achieves lossless performance to
32-bit counterpart in training different DNNs on various tasks.

Limitations. Preconditioners in Shampoo are symmetric matrices and can be stored as upper
triangular matrices, saving almost half of the memory usage. However, the eigenvector matrix of
a preconditioner is not symmetric, causing an 8-bit preconditioner to occupy the same memory as
its 4-bit eigenvector matrix. Notably, a comparison of Table 1 and Table 7 in Appendix D shows
that the 4-bit quantization of the eigenvector matrix has smaller quantization errors than the 8-bit
quantization of the preconditioner. Our evaluation is currently limited to image classiﬁcation and
natural language modeling tasks. Due to limitations in computing resources, we do not test our 4-bit
Shampoo on large-scale models with billions of parameters.

9


---Page Break---
Broader Impact. Our work can facilitate training large models with second-order optimizers. This
could open up new research possibilities that were previously unattainable due to GPU memory
constraints, especially beneﬁting researchers with limited resources.

Acknowledgments and Disclosure of Funding

Jia Li and Hua Huang were supported by the NSF of China (grant no. 62131003). Jia Li was also
supported by the NSF of China (grant no. 62102034). Pan Zhou was supported by the Singapore
Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grants (project ID: 23-SIS-
SMU-028 and 23-SIS-SMU-070).

References

[1] Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Disentangling
adaptive gradient methods from learning rates. arXiv preprint arXiv:2002.11803, 2020.

[2] Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Scalable second
order optimization for deep learning. arXiv preprint arXiv:2002.09018, 2020.

[3] Rohan Anil, Vineet Gupta, Tomer Koren, and Yoram Singer. Memory efﬁcient adaptive
optimization. Advances in Neural Information Processing Systems, 32, 2019.

[4] Å. Björck and C. Bowie. An iterative algorithm for computing the best estimate of an orthogonal
matrix. SIAM Journal on Numerical Analysis, 8(2):358–364, 1971.

[5] R.L. Burden, J.D. Faires, and A.M. Burden. Numerical Analysis. Cengage Learning, 2015.

[6] Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and
Ashok Cutkosky. The road less scheduled. arXiv preprint arXiv:2405.15682, 2024.

[7] Tim Dettmers. 8-bit approximations for parallelism in deep learning. In Proceedings of the
International Conference on Learning Representations, 2016.

[8] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-
wise quantization. In Proceedings of the International Conference on Learning Representations,
2022.

[9] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efﬁcient
ﬁnetuning of quantized LLMs. Advances in Neural Information Processing Systems, 2023.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby.
An image is worth 16x16 words: Transformers for
image recognition at scale. In Proceedings of the International Conference on Learning
Representations, 2021.

[11] Timothy Dozat. Incorporating Nesterov momentum into Adam. In Proceedings of the Interna-
tional Conference on Learning Representations Workshop, 2016.

[12] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning
and stochastic optimization. Journal of Machine Learning Research, 12(61):2121–2159, 2011.

[13] Sai Surya Duvvuri, Fnu Devvrit, Rohan Anil, Cho-Jui Hsieh, and Inderjit S Dhillon. Combining
axes preconditioners through Kronecker approximation for deep learning. In Proceedings of the
International Conference on Learning Representations, 2024.

[14] Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, and Elad Hazan. Sketchy:
Memory-efﬁcient adaptive regularization with frequent directions. Advances in Neural Informa-
tion Processing Systems, 2023.

[15] Elias Frantar, Eldar Kurtic, and Dan Alistarh. M-FAC: Efﬁcient matrix-free approximations of
second-order information. Advances in Neural Information Processing Systems, 2021.

10


---Page Break---
[16] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han,
Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. Conformer: Convolution-
augmented transformer for speech recognition.
In Proceedings of the Conference of the
International Speech Communication Association, 2020.

[17] Chun-Hua Guo and Nicholas J. Higham. A Schur–Newton method for the matrix pth root and
its inverse. SIAM Journal on Matrix Analysis and Applications, 28(3):788–804, 2006.

[18] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor
optimization. In Proceedings of the International Conference on Machine Learning, 2018.

[19] N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic
algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288,
2011.

[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im-
age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, June 2016.

[21] Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge university press, 2012.

[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings
of the International Conference on Learning Representations, 2015.

[23] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.

[24] Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.

[25] Seung Hoon Lee, Seunghyun Lee, and Byung Cheol Song. Vision transformer for small-size
datasets. arXiv preprint arXiv:2112.13492, 2021.

[26] Bingrui Li, Jianfei Chen, and Jun Zhu. Memory efﬁcient optimizers with 4-bit states. Advances
in Neural Information Processing Systems, 2023.

[27] Hong Liu, Zhiyuan Li, David Leo Wright Hall, Percy Liang, and Tengyu Ma. Sophia: A
scalable stochastic second-order optimizer for language model pre-training. In Proceedings of
the International Conference on Learning Representations, 2024.

[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining
Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings
of the IEEE/CVF International Conference on Computer Vision, October 2021.

[29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of
the International Conference on Learning Representations, 2019.

[30] James Martens and Roger Grosse.
Optimizing neural networks with Kronecker-factored
approximate curvature. In Proceedings of the International Conference on Machine Learning,
2015.

[31] Ning Qian. On the momentum term in gradient descent learning algorithms. Neural Networks,
12(1):145–151, 1999.

[32] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and others.
Language models are unsupervised multitask learners. OpenAI blog, 2019.

[33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a uniﬁed
text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

[34] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-
Fei. ImageNet large scale visual recognition challenge. International Journal of Computer
Vision, 115(3):211–252, 2015.

11


---Page Break---
[35] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory
cost. In Proceedings of the International Conference on Machine Learning, 2018.

[36] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
image recognition. In Proceedings of the International Conference on Learning Representations,
2015.

[37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and others. Llama 2:
Open foundation and ﬁne-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information
Processing Systems, 30, 2017.

[39] Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, and Michael W.
Mahoney. AdaHessian: an adaptive second order optimizer for machine learning. In Proceedings
of the AAAI Conference on Artiﬁcial Intelligence, 2021.

[40] Jui-Nan Yen, Sai Surya Duvvuri, Inderjit S. Dhillon, and Cho-Jui Hsieh. Block low-rank
preconditioner with shared basis for stochastic optimization. Advances in Neural Information
Processing Systems, 2023.

[41] Hongwei Yong, Ying Sun, and Lei Zhang. A general regret bound of preconditioned gradient
method for DNN training. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, June 2023.

[42] Lin Zhang, Shaohuai Shi, and Bo Li. Eva: Practical second-order optimization with Kronecker-
vectorized approximation. In Proceedings of the International Conference on Learning Repre-
sentations, 2023.

[43] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong
Tian. GaLore: Memory-efﬁcient LLM training by gradient low-rank projection. In Proceedings
of the International Conference on Machine Learning, 2024.

[44] Pan Zhou, Xingyu Xie, and Shuicheng Yan. Win: Weight-decay-integrated Nesterov accel-
eration for adaptive gradient algorithms. In Proceedings of the International Conference on
Learning Representations, 2023.

12


---Page Break---
A
Implementation Details of Shampoo, CASPR, K-FAC and AdaBK

The implementation of 32-bit Shampoo used in our experiments is described in Algorithm 4. Our
Pytorch implementation of Shampoo is partially based on the code provided by [2]. We implement
CASPR by replacing b
Gt = bLtGt b
Rt with Jt = bLtGt + Gt b
Rt; b
Gt = bLtJt + Jt b
Rt in line 12 of
Algorithm 4 and line 14 of Algorithm 3. We summarize the implementation of 32-bit K-FAC/AdaBK
in Algorithm 5, where Xt is the input feature and Yt is the output feature gradient. Both power
iteration [5] and Schur-Newton iteration [17] are run for 10 iterations. Our implementation of 4-bit
K-FAC/AdaBK is similar to 4-bit Shampoo (i.e., compressing Lt, Rt, bLt, and b
Rt).

Algorithm 4 Practical 32-bit Shampoo
Input: initial parameter W0 ∈Rm×n, left preconditioner L0 = ϵIm, right preconditioner R0 =
ϵIn, inverse root of left preconditioner bL0 = Im, inverse root of right preconditioner b
R0 = In,
total number of steps T, interval of updating preconditioners T1, interval of updating inverse
roots of preconditioners T2, exponential decay rate for preconditioners β ∈(0, 1), ﬁrst-order
optimizer F, ﬁrst-order optimizer state s0 = 0.
Output: ﬁnal parameter WT .

1: for t = 1, 2, . . . , T do
2:
Receive loss function Lt : Rm×n 7→R and compute gradient Gt = ∇Lt(Wt)
3:
if t%T1 ≡0 then
4:
Lt = βLt−1 + (1 −β)GtGT
t ;
Rt = βRt−1 + (1 −β)GT
t Gt
5:
else
6:
Lt = Lt−1;
Rt = Rt−1
7:
if t%T2 ≡0 then
8:
Compute maximum eigenvalues λL
max and λR
max of Lt and Rt by power iteration
9:
Compute bLt = (Lt +λL
maxϵIm)−1/4 and b
Rt = (Rt +λR
maxϵIn)−1/4 by Schur-Newton
iteration
10:
else
11:
bLt = bLt−1;
b
Rt = b
Rt−1
12:
b
Gt = bLtGt b
Rt;
e
Gt = b
Gt(∥Gt∥F /∥b
Gt∥F )
13:
Wt, st = F(Wt−1, st−1, e
Gt)

Algorithm 5 Practical 32-bit K-FAC/AdaBK
Input: initial parameter W0 ∈Rm×n, left preconditioner L0 = 0, right preconditioner R0 = 0,
inverse root of left preconditioner bL0 = Im, inverse root of right preconditioner b
R0 = In, total
number of steps T, interval of updating preconditioners T1, interval of updating inverse roots of
preconditioners T2, ϵ, exponential decay rate for preconditioners β ∈(0, 1), α = 1 for K-FAC /
α = 2 for AdaBK, ﬁrst-order optimizer F, ﬁrst-order optimizer state s0 = 0.
Output: ﬁnal parameter WT .

1: for t = 1, 2, . . . , T do
2:
Receive loss function Lt : Rm×n 7→R and compute gradient Gt = ∇Lt(Wt)
3:
Receive Xt by forward propagation and Yt by backward propagation
4:
if t%T1 ≡0 then
5:
Lt = βLt−1 + (1 −β)YtY T
t ;
Rt = βRt−1 + (1 −β)XtXT
t
6:
else
7:
Lt = Lt−1;
Rt = Rt−1
8:
if t%T2 ≡0 then
9:
Compute maximum eigenvalues λL
max and λR
max of Lt and Rt by power iteration
10:
Compute bLt = (Lt +λL
maxϵIm)−1/α and b
Rt = (Rt +λR
maxϵIn)−1/α by Schur-Newton
iteration
11:
else
12:
bLt = bLt−1;
b
Rt = b
Rt−1
13:
b
Gt = bLtGt b
Rt;
e
Gt = b
Gt(∥Gt∥F /∥b
Gt∥F )
14:
Wt, st = F(Wt−1, st−1, e
Gt)

13


---Page Break---
0
1
2
3
4
5
6
7
Index

1.00

0.75

0.50

0.25

0.00

0.25

0.50

0.75

1.00

Value

DT
Linear-2

(a) 3-bit quantization

0
2
4
6
8
10
12
14
Index

1.00

0.75

0.50

0.25

0.00

0.25

0.50

0.75

1.00

Value

DT
Linear-2

(b) 4-bit quantization

Figure 5: Visualization of DT quantization and Linear-2 quantization at b-bit (b = 3, 4) precision.

B
Randomized SVD Method

Given an initial matrix P0 ∈Rn×n, randomized SVD method computes the eigenvector matrix of a
PD matrix A ∈Rn×n by iterating

Pt = QR(APt−1),
(4)

where QR(X) denotes the QR decomposition of matrix X, returning an orthogonal matrix. Since
we can initialize P0 with the previous result (e.g., V in Algorithm 1), only a few iterations are
enough to obtain an accurate estimation in practice. In our experiments, we iterate (4) once for
Shampoo/CASPR, and iterate (4) twice for K-FAC/AdaBK.

C
Quantization Mappings

We present the constructions of different quantization mappings in b-bit quantizers (R in Q). See
Figure 5 for the illustration of them. Note that Tb ={0, 1, . . . , 2b−1}.

Dynamic tree (DT) quantization for b-bit quantization maps Tb onto {0, 1} ∪G, where G is a
set of numbers with the following properties: the number in G looks like ±qk × 10−E, where a)
b = 2 + E + F, where E, F are integers; b) qk = (pk + pk+1)/2, where k ∈{0, . . . , 2F −1};
c) pj = 0.9j/2F + 0.1, where j ∈{0, . . . , 2F }. For 4-bit quantization, DT quantization maps T4
onto {-0.8875, -0.6625, -0.4375, -0.2125, -0.0775, -0.0325, -0.0055, 0.0000, 0.0055, 0.0325, 0.0775,
0.2125, 0.4375, 0.6625, 0.8875, 1.0000}. For 3-bit quantization, DT quantization maps T3 onto
{-0.7750, -0.3250, -0.0550, 0.0000, 0.0550, 0.3250, 0.7750, 1.0000}.

For 4-bit quantization, linear square (Linear-2) quantization maps T4 onto {-1.0000, -0.7511, -0.5378,
-0.3600, -0.2178, -0.1111, -0.0400, 0.0000, 0.0044, 0.0400, 0.1111, 0.2178, 0.3600, 0.5378, 0.7511,
1.0000}. For 3-bit quantization, Linear-2 quantization maps T3 onto {-1.0000, -0.5102, -0.1837,
0.0000, 0.0204, 0.1837, 0.5102, 1.0000}.

D
Quantization Error Analyses

We present more quantization error analyses of the preconditioners. Recall that we deﬁne two kinds
of quantization errors in mapping f of transformation g at A ∈Rm×n (in short errors in f(A) of g)
in Subsection 3.1. Here we extend them as follows: deﬁne the normwise relative error (NRE) in f of
(g1, g2) at A as

NRE = ∥f(A) −g2 ◦f ◦g1(A)∥F

∥f(A)∥F
,

and the angle error (AE) in f of (g1, g2) at A as

AE = arccos

⟨f(A), g2◦f ◦g1(A)⟩
∥f(A)∥F ∥g2◦f ◦g1(A)∥F


.

14


---Page Break---
D.1
Static Analysis

Table 5 is an extension of Table 1 for Bit=4. Since the diagonal elements of A−1/4 are usually much
larger than its non-diagonal elements where A is a PD matrix, we further consider the quantization
errors in f(A)=A−1/4−Diag(diag(A−1/4)) at 4-bit precision as shown in Table 6. Table 7 shows
the quantization errors at 8-bit precision.

A large condition number of a PD matrix A is indispensable for the superiority of quantizing U
over quantizing A, where U is the eigenvector matrix of A. We consider contracting the singular
value distribution of A = A1 with SVD UDiag(λ)U T used in Table 5 by mapping each singular
value λ of A to h(λ) = τ(λ −λA
min)+λA
min, where λA
min is the minimum singular value of A
and τ > 0 is the contraction coefﬁcient. Figure 6 shows 4-bit quantization errors in A−1/4 or
A−1/4−Diag(diag(A−1/4)) of quantizing U or A at A=UDiag(h(λ))U T.

Table 5:
Quantization errors in f(A) = A−1/4 of different 4-bit quantization schemes at a PD
matrix A. We employ block-wise normalization with a block size of 64. U is the eigenvector matrix
of A and B = (g1(A))−1/4. QM = quantized matrices and OR = orthogonal rectiﬁcation.

Real-world A = A1
Synthetic A = A2

Mapping R
QM
OR
NRE ↓
AE (◦) ↓
Mapping R
QM
OR
NRE ↓
AE (◦) ↓

DT

A

0.6241
17.319

DT

A

0.4615
17.189
U

0.0709
4.0426
U

0.1224
7.0144
U

0.0455
2.5615
U

0.0878
4.9960
B

0.0398
2.2802
B

0.0853
4.8914
(A, B)

0.6243
17.364
(A, B)

0.4649
17.650
(U, B)

0.0811
4.6296
(U, B)

0.1485
8.5168
(U, B)

0.0604
3.4230
(U, B)

0.1224
6.9817

Linear-2

A

0.6243
17.293

Linear-2

A

0.4465
15.338
U

0.0543
3.1066
U

0.0942
5.3998
U

0.0343
1.9456
U

0.0669
3.8166
B

0.0315
1.8050
B

0.0661
3.7887
(A, B)

0.6243
17.301
(A, B)

0.4483
15.654
(U, B)

0.0626
3.5833
(U, B)

0.1150
6.5901
(U, B)

0.0466
2.6494
(U, B)

0.0941
5.3716

Table 6: Quantization errors in f(A)=A−1/4−Diag(diag(A−1/4)) of different 4-bit quantization
schemes at a PD matrix A. We employ block-wise normalization with a block size of 64. U is the
eigenvector matrix of A and B = (g1(A))−1/4. QM = quantized matrices and OR = orthogonal
rectiﬁcation.

Real-world A = A1
Synthetic A = A2

Mapping R
QM
OR
NRE ↓
AE (◦) ↓
Mapping R
QM
OR
NRE ↓
AE (◦) ↓

DT

A

0.9549
59.360

DT

A

0.6247
25.913
U

0.2328
13.287
U

0.1994
11.444
U

0.1480
8.4365
U

0.1427
8.1415
B

0.1314
7.5513
B

0.1391
7.9813
(A, B)

0.9561
59.825
(A, B)

0.6314
26.948
(U, B)

0.2666
15.281
(U, B)

0.2420
13.911
(U, B)

0.1977
11.322
(U, B)

0.1992
11.393

Linear-2

A

0.9547
58.336

Linear-2

A

0.6010
20.780
U

0.1786
10.213
U

0.1534
8.8027
U

0.1122
6.4096
U

0.1088
6.2176
B

0.1041
5.9554
B

0.1078
6.1755
(A, B)

0.9548
58.601
(A, B)

0.6047
21.666
(U, B)

0.2063
11.778
(U, B)

0.1873
10.745
(U, B)

0.1530
8.7337
(U, B)

0.1532
8.7534

15


---Page Break---
20
16
12
8
4
0
Contraction Coefficient 

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Normwise Relative Error

quan A
quan U

20
16
12
8
4
0
Contraction Coefficient 

0.0

2.5

5.0

7.5

10.0

12.5

15.0

17.5

Angle Error ( )

quan A
quan U

(a) Errors in f(A)=A−1/4

20
16
12
8
4
0
Contraction Coefficient 

0.0

0.2

0.4

0.6

0.8

1.0

Normwise Relative Error

quan A
quan U

20
16
12
8
4
0
Contraction Coefficient 

10

20

30

40

50

60

70

80

Angle Error ( )

quan A
quan U

(b) Errors in f(A)=A−1/4−Diag(diag(A−1/4))

Figure 6: 4-bit quantization errors in f(A) of quantizing U or A at A = UDiag(h(λ))U T. We
use linear square quantization and orthogonal rectiﬁcation. The condition number cond(A) =
λA
max/λA
min is around 37235, where λA
max and λA
min are the maximum and minimum singular values
of A respectively. Contraction coefﬁcients are shown on a log2 scale.

Table 7:
Quantization errors in f(A) of different 8-bit quantization schemes at a PD matrix A,
where A = A1 is derived from the real world as described in Subsection 3.1. We employ block-wise
normalization with a block size of 256. U is the eigenvector matrix of A and B = (g1(A))−1/4.
QM = quantized matrices and OR = orthogonal rectiﬁcation.

f(A) = A−1/4
f(A)=A−1/4−Diag(diag(A−1/4))

Mapping R
QM
OR
NRE ↓
AE (◦) ↓
Mapping R
QM
OR
NRE ↓
AE (◦) ↓

DT

A

0.2192
8.3014

DT

A

0.5001
23.644
U

0.0060
0.3421
U

0.0197
1.1273
U

0.0037
0.2140
U

0.0123
0.7022
B

0.0029
0.1655
B

0.0097
0.5553
(A, B)

0.2193
8.3051
(A, B)

0.5003
23.649
(U, B)

0.0067
0.3810
(U, B)

0.0219
1.2577
(U, B)

0.0047
0.2712
(U, B)

0.0156
0.8955

Linear-2

A

0.2164
7.9751

Linear-2

A

0.4875
21.447
U

0.0037
0.2121
U

0.0122
0.6994
U

0.0023
0.1312
U

0.0076
0.4343
B

0.0021
0.1203
B

0.0070
0.4035
(A, B)

0.2164
7.9755
(A, B)

0.4875
21.448
(U, B)

0.0043
0.2439
(U, B)

0.0141
0.8079
(U, B)

0.0031
0.1791
(U, B)

0.0104
0.5935

D.2
Dynamic Analysis

We deﬁne the normwise relative error (NRE) and angle error (AE) of B deviating from A as

NRE= ∥B −A∥F

∥A∥F
,
AE=arccos

⟨A, B⟩
∥A∥F ∥B∥F


.

Consider Shampoo using 4-bit preconditioners for parameter updates, but also recording 32-bit
preconditioners at the same time. We extract the left preconditioners L4 and L32 ∈R1200×1200 of
a speciﬁc model parameter block W ∈R1200×768 every 8000 steps in the Swin-Tiny training on
CIFAR-100 with AdamW+Shampoo. Here L4 is a decompressed 4-bit preconditioner, and L32 is a
32-bit preconditioner.

Figure 7 shows the quantization errors during training. For naive 4-bit Shampoo, L−1/4
32
and L−1/4
4
are computed by Schur-Newton iteration used in Algorithm 4 where ϵ=10−4. For our 4-bit Shampoo,
L−1/4
32
is computed by Schur-Newton iteration used in Algorithm 4 where ϵ=10−4, and L−1/4
4
is
computed by Algorithm 2 without quantization where ϵ = 10−4, t2 = 4. We ﬁnd that ϵ = 10−6 for
Algorithm 2 used in our main experiments though is effective, yet it can cause a large numerical
instability in the later stage of training (see Figure 8).

16


---Page Break---
2
4
6
8
10
12
14
Step (×8000)

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Normwise Relative Error

4-bit Shampoo (naive)
4-bit Shampoo (our)

2
4
6
8
10
12
14
Step (×8000)

10

12

14

16

18

20

22

Angle Error ( )

4-bit Shampoo (naive)
4-bit Shampoo (our)

(a) Errors in L−1/4
4
deviating from L−1/4
32

2
4
6
8
10
12
14
Step (×8000)

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Normwise Relative Error

4-bit Shampoo (naive)
4-bit Shampoo (our)

2
4
6
8
10
12
14
Step (×8000)

40

45

50

55

60

65

Angle Error ( )

4-bit Shampoo (naive)
4-bit Shampoo (our)

(b) Errors in L−1/4
4
−Diag(diag(L−1/4
4
)) deviating
from L−1/4
32
−Diag(diag(L−1/4
32
))

Figure 7:
Quantization errors during Swin-Tiny training on the CIFAR-100 dataset. We use
dampening term ϵ = 10−4 to compute L−1/4
4
and L−1/4
32
.

2
4
6
8
10
12
14
Step (×8000)

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Normwise Relative Error

4-bit Shampoo (naive)
4-bit Shampoo (our)

2
4
6
8
10
12
14
Step (×8000)

12

14

16

18

20

22

Angle Error ( )

4-bit Shampoo (naive)
4-bit Shampoo (our)

(a) Errors in L−1/4
4
deviating from L−1/4
32

2
4
6
8
10
12
14
Step (×8000)

0.8

1.0

1.2

1.4

1.6

Normwise Relative Error

4-bit Shampoo (naive)
4-bit Shampoo (our)

2
4
6
8
10
12
14
Step (×8000)

45

50

55

60

65

Angle Error ( )

4-bit Shampoo (naive)
4-bit Shampoo (our)

(b) Errors in L−1/4
4
−Diag(diag(L−1/4
4
)) deviating
from L−1/4
32
−Diag(diag(L−1/4
32
))

Figure 8:
Quantization errors during Swin-Tiny training on the CIFAR-100 dataset. We use
dampening term ϵ = 10−6 to compute L−1/4
4
and L−1/4
32
.

E
Convergence Analysis

More notations. Given a symmetric real matrix A, A ⪰0 means that A is positive semideﬁnite
(PSD), and A ≻0 means that A is positive deﬁnite (PD). Assume that symmetric matrices A and B
are symmetric, the notations A ⪰B and A ≻B mean that A−B ⪰0 and A−B ≻0 respectively.
Let A be a PSD matrix and s ∈R, we deﬁne As =UΛsU T, where UΛU T is the Singular Value
Decomposition (SVD) of A. The Mahalanobis norm of a vector x induced by a PD matrix A is
∥x∥A =
√

xTAx. The dual norm of ∥· ∥A is denoted by ∥· ∥∗
A, where ∥x∥∗
A =
√

xTA−1x. The
spectral norm of matrix A is ∥A∥2 = supx̸=0{∥Ax∥2/∥x∥2}. A ⊗B means the (right) Kronecker
product of matrices A and B. vec(A) means the vectorization (stacking the rows) of A.

Algorithm 6 Perturbed Shampoo in the matrix case
Input: W0 ∈Rm×n, L0 = 0m×m, R0 = 0n×n, ρ0 = 0, µ0 = 0.

1: for t = 1, . . . , T do
2:
Receive loss function: ft : Rm×n →R
3:
Compute gradient: Gt = ∇ft(Wt)
4:
Update preconditioners: Jt = Lt−1 + GtGT
t ;
Kt = Rt−1 + GT
t Gt
5:
Perturb preconditioners: Lt = g(Jt);
Rt = g(Kt)
6:
Accumulate errors: ρt = ρt−1 + ∥Jt −Lt∥2;
µt = µt−1 + ∥Kt −Rt∥2
7:
Update parameters: Wt+1 = Wt −η((ϵ + ρt)Im + Lt)−1/4Gt((ϵ + µt)In + Rt)−1/4

We consider quantization as a perturbation and present the perturbed Shampoo in Algorithm 6 for
convergence analysis. The regret bound of the perturbed Shampoo can be found in Theorem 1.
Complete proofs can be found in Appendix F. We ﬁrst introduce some basic technical tools, and the
details of them are in [18, 21].

Lemma 3. Let A, A′, B, B′ be matrices of appropriate dimensions, and u, v be two column vectors.
The following properties hold:

(1) (A ⊗B)(A′ ⊗B′) = (AA′) ⊗(BB′);

17


---Page Break---
(2) (A ⊗B)T = (AT ⊗BT);
(3) If A, B ⪰0 and s ∈R, then (A ⊗B)s = (As ⊗Bs);
(4) If A ⪰A′ and B ⪰B′, then A ⊗B ⪰A′ ⊗B′;
(5) tr(AB) = tr(A)tr(B);
(6) vec(uvT) = u ⊗v.
Lemma 4. Let G ∈Rm×n, L ∈Rm×m, R ∈Rn×n, then it holds that
(L ⊗RT)vec(G) = vec(LGR).
Lemma 5. Assume that 0 ⪯Xi ⪯Yi for i = 1, . . . , n. Assume further that all Xi commute with
each other and all Yi commute with each other. Let α1, . . . , αn ≥0 such that Pn
i=1 αi = 1, then
Xα1
1
· · · Xαn
n
⪯Y α1
1
· · · Y αn
n
.
Lemma 6. Let 0 ≤α ≤1 and 0 ⪯X ⪯Y , then Xα ⪯Y α.
Lemma 7. Let A ≻0 and B ≻0, then it holds that A ⪰B if and only if B−1 ⪰A−1.
Lemma 8 (von Neumann). Let A, B ∈Rm×n and q = min{m, n}. Let σ1(A) ≥· · · ≥σq(A) and
σ1(B) ≥· · · ≥σq(B) denote the non-increasingly ordered singular values of A and B, respectively.
Then

⟨A, B⟩≤

q
X

i=1
σi(A)σi(B).

Lemma 9. Assume that function ft is continuously differentiable and convex on Rd, and matrix
Ht ≻0 for t = 1, . . . , T. Given w0 ∈Rd, η > 0, deﬁne wt+1 = wt −ηH−1
t
gt, where
gt = ∇ft(wt). Then for any w∗∈Rd, we have

T
X

t=1
ft(wt) −

T
X

t=1
ft(w∗) ≤1

2η

T
X

t=1
(∥wt −w∗∥2
Ht −∥wt+1 −w∗∥2
Ht) + η

2

T
X

t=1
(∥gt∥∗
Ht)2.

Lemma 10. Let g1, . . . , gT be a sequence of vectors. For ρ > 0, deﬁne c
Ht = (ρI +Pt
s=1 gsgT
s )1/2.
Then we have
T
X

t=1
(∥gt∥∗
c
Ht)2 ≤2tr(c
HT ).

Lemma 11. Assume that G1, . . . , GT ∈Rm×n are matrices of rank at most r. Let s for t = 1, . . . , T.
Then for any ϵ ≥0,

ϵImn + 1

r

T
X

t=1
gtgT
t ⪯(ϵIm +

T
X

t=1
GtGT
t )1/2 ⊗(ϵIn +

T
X

t=1
GT
t Gt)1/2.

The key to the convergence proof of Algorithm 6 is forming a PD matrix sequence {Hi}T
i=1, which
satisﬁes 0 ≺H1 ⪯· · · ⪯HT . To achieve it, we gives the following lemma extended from Lemma 2
in the Appendix of [40].
Lemma 12. Let {Xt}t=T
t=1 be a sequence of symmetric matrices, and At = Pt
s=1 Xs, where
t = 1, . . . , T. Suppose we have two sequences of symmetric matrices {Yt}t=T
t=1 , {Zt}t=T
t=0 , and a
sequence real numbers {ρt}t=T
t=0 satisfying
Yt = Zt−1 + Xt,
ρt = ρt−1 + ∥Yt −Zt∥2,
Z0 = 0, ρ0 = 0.
Deﬁne Bt = ρtI + Zt, where I denotes the identity matrix. Then for t = 1, . . . , T, we have
Bt ⪰Bt−1 + Xt,
At ⪯Bt ⪯2ρtI + At.

Theorem 1. Assume that the gradients G1, . . . , GT ∈Rm×n are matrices of rank at most r. Then
for any W ∗∈Rm×n and ϵ > 0, if η = D/
√

2r, the regret of Algorithm 6 is bounded as follows,

T
X

t=1
ft(Wt) −

T
X

t=1
ft(W ∗) ≤
√

2rD[21/4mρ1/4
T
+ tr( ˜L1/4
T )][21/4nµ1/4
T
+ tr( ˜R1/4
T )],

where D = maxt∈[T ] ∥Wt −W ∗∥F , ˜Lt = ϵIm + PT
t=1 GtGT
t , and ˜Rt = ϵIn + PT
t=1 GT
t Gt.

Though we get a convergence guarantee of Algorithm 6, the upper bound given by Theorem 1 is very
slack, since 21/4mρ1/4
T
is about the same as tr( ˜L1/4
T ) for 4-bit quantization schemes in practice.

18


---Page Break---
F
Proofs

Lemma 1. Let A be a PD matrix whose SVD is UΛU T, where U =[ui] is an orthogonal matrix
and Λ=diag([λi]T) is a diagonal matrix. Given a perturbation ∆U =[∆ui] and s ∈R, we deﬁne
B :=(UΛU T)s and ∆B :=((U +∆U)Λ(U +∆U)T)s−B.

(1) If U +∆U is orthogonal and there exists α ∈R such that ∥∆ui∥2 ≤α, then

∥∆B∥F

∥B∥F
≤2α.

(2) If U +∆U is orthogonal and there exists β ∈R such that ⟨ui, ui+∆ui⟩≥1−β ≥0, then

⟨B, B+∆B⟩
∥B∥F ∥B+∆B∥F
≥(1−β)2.

Proof. (1) Since U and U + ∆U are orthogonal, we have

B = UΛsU T,
B + ∆B = (U + ∆U)Λs(U + ∆U)T,

by deﬁnition. This leads to

∆B = UΛs∆U T + ∆UΛs(U + ∆U)T.

The Frobenius norm satisﬁes the triangle inequality and is orthogonality invariant. Hence,

∥∆B∥F = ∥UΛs∆U T + ∆UΛs(U + ∆U)T∥F
≤∥UΛs∆U T∥F + ∥∆UΛs(U + ∆U)T∥F
= ∥Λs∆U T∥F + ∥∆UΛs∥F = 2∥∆UΛs∥F

= 2
rX

i∥λs
i∆ui∥2
2 = 2
rX

iλ2s
i ∥∆ui∥2
2

≤2
rX

iλ2s
i α2 = 2α
rX

iλ2s
i
= 2α∥Λs∥F

= 2α∥B∥F .

(2) Similar to (1), we have

∆B = UΛs∆U T + ∆UΛsU T + ∆UΛs∆U T.

From ⟨ui, ui + ∆ui⟩≥1 −β ≥0, we get 0 ≥⟨ui, ∆ui⟩≥−β ≥−1 because

1 = ∥ui∥2∥ui + ∆ui∥2 ≥⟨ui, ui + ∆ui⟩= 1 + ⟨ui, ∆ui⟩≥1 −β ≥0,

holds due to the orthogonality of U and U + ∆U. Hence,

⟨B, ∆B⟩= tr(2UΛ2s∆U T + UΛsU T∆UΛs∆U T)

= tr
X

i2λ2s
i ui∆uT
i

+ tr
hX

iλs
iuiuT
i
X

jλs
j∆uj∆uT
j
i

=
X

i2λ2s
i ⟨ui, ∆ui⟩

+
X

ijλs
iλs
j⟨ui, ∆uj⟩2

≥
X

i2λ2s
i ⟨ui, ∆ui⟩

+
X

iλ2s
i ⟨ui, ∆ui⟩2

=
X

iλ2s
i [(1 + ⟨ui, ∆ui⟩)2 −1]

≥
X

iλ2s
i [(1 −β)2 −1] = [(1 −β)2 −1]∥Λs∥2
F

= [(1 −β)2 −1]∥B∥2
F = [(1 −β)2 −1]⟨B, B⟩.

Therefore, we have

⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
= ⟨B, B + ∆B⟩

⟨B, B⟩
= 1 + ⟨B, ∆B⟩

⟨B, B⟩
≥(1 −β)2.

The proof is completed.

19


---Page Break---
Lemma 2. Let A be a PD matrix of order m+n whose SVD is UΛU T, where m, n ∈N+, n = lm,
U = [ui] is an orthogonal matrix and Λ = diag([λi]T) is a diagonal matrix. Assume that Λ =
diag([cλ1T
m×1, λ1T
n×1]T), c ≥1, and λ > 0. Given a perturbation ∆Λ = diag([0T
m×1, ∆λT
n×1]T)
and s ∈R, we deﬁne B :=(UΛU T)s and ∆B :=(U(Λ+∆Λ)U T)s−B.

(1) If ∆λn×1 = (k −1)λ1n×1 where k > 0, then

∥∆B∥F

∥B∥F
=

√

l|ks −1|
√

c2s + l
= h1(s, l).

Moreover, h1(s, l) decreases monotonically with s over (−∞, 0) and increases monotonically
with l over (0, +∞).
(2) If ∆λn×1 = (tc −1)λ1n×1 where t > 0, then

⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
lts + cs
p

(1 + lt2s)(l + c2s)
= h2(l).

Moreover, h2(l) decreases monotonically with l over (0, (c/t)s] and increases monotonically
with l over ((c/t)s, +∞).
(3) If ∆λn×1 = (tc −1)λ1n×1 where k = tc > 0 and l = (c/t)s, then

∥∆B∥F

∥B∥F
= |ks −1|
√

ks + 1,
⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
2
p

2 + ks + 1/ks .

Proof. (1) Since U is orthogonal, we have

∥∆B∥F = ∥(Λ + ∆Λ)s −Λs∥F = √n|ks −1|λs,
∥B∥F = ∥Λs∥F =
p

mc2s + nλs.

Hence,

∥∆B∥F

∥B∥F
=
√n|ks −1|
√

mc2s + n
=

√

l|ks −1|
√

c2s + l
= h1(s, l) ≥0.

It is easy to check that h1 increases monotonically with l over (0, +∞). To prove h1 decreases
monotonically with s over (−∞, 0), deﬁne

g1(s) = 1

l (h1(s, l))2 = (ks −1)2

c2s + l .

Consider the derivative of g1

g′
1(s) = (c2s + l)2(ks −1)ks ln k −(ks −1)2c2s2 ln c

(c2s + l)2

= 2(ks −1)
 
(c2s + l)ks ln k −(ks −1)c2s ln c


(c2s + l)2
.

If s < 0 and k > 1, then ks −1 < 0, ks ln k > 0 leading to g′
1(s) < 0 since c ≥0; Similarly, if s < 0
and 0 < k ≤1, then ks −1 ≥0, ks ln k ≤0 leading to g′
1(s) ≤0. Thus g1(s) is a monotonically
decreasing function for s < 0, which implies that h1 decreases monotonically with s over (−∞, 0).

(2) Similar to (1), we have

∥B∥F =
p

mc2s + nλs,
∥B + ∆B∥F =
p

nt2s + mcsλs.

Besides,

⟨B, B + ∆B⟩= tr(UΛs(Λ + ∆Λ)sU T) = tr(Λs(Λ + ∆Λ)s) = (mc2s + ncsts)λ2s.

Hence, we get

⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
nts + mcs
p

(m + nt2s)(n + mc2s)
=
lts + cs
p

(1 + lt2s)(l + c2s)
= h2(l) ≥0.

20


---Page Break---
To prove h2 decreases monotonically with l over (0, (c/t)s] and increases monotonically with l over
((c/t)s, +∞), we deﬁne

g2(l) = (h2(l))2 =
(lts + cs)2

(1 + lt2s)(l + c2s),

whose monotonicity is equivalent to that of h2 for l > 0. Consider the derivative of g2

g′
2(l) =

t2sl2 + 2tscsl + c2s

t2sl2 + l + t2sc2sl + c2s

′
= (ts −t2scs)2l2 −(cs −tsc2s)2

(t2sl2 + l + t2sc2sl + c2s)2
.

If s = 0 or tc = 1, then g2(l) ≡1. If s ̸= 0 and tc ̸= 1, then (ts −t2scs)2 > 0, (cs −tsc2s)2 > 0.
In this case, let g′
2(l) = 0, we get

t2s(1 −tscs)2l2 = c2s(1 −tscs)2,

which implies that l = (c/t)s. It is easy to see that g2 decreases monotonically with l over (0, (c/t)s]
and increases monotonically with l over ((c/t)s, +∞).

(3) According to (1)(2), we can easily get

∥∆B∥F

∥B∥F
= |ks −1|
√

ks + 1,
⟨B, B + ∆B⟩
∥B∥F ∥B + ∆B∥F
=
2
p

2 + ks + 1/ks .

The proof is completed.

Proposition 1. Let A be a PD matrix of order m+n whose SVD is UΛU T, where m, n ∈N+,
n = lm, U = [ui] is an orthogonal matrix, Λ = diag([cλ1T
m×1, λ1T
n×1]T), c ≥1000, and λ > 0.
Given ∆U = [∆ui], ∆Λ = diag([0T
m×1, ∆λT
n×1]T), and s ≤−0.25, we deﬁne B := (UΛU T)s,
B1 := ((U + ∆U)Λ(U +∆U)T)s, and B2 := (U(Λ+∆Λ)U T)s. If U + ∆U is orthogonal,
∥∆ui∥2 ≤0.1, ⟨ui, ∆ui⟩≥−0.005, ∆λn×1 =(0.02c−1)λ1n×1, and l=(c/0.02)s, then

2∥B1 −B∥F

∥B∥F
≤0.4≤∥B2 −B∥F

∥B∥F
,
6

1 −
⟨B, B1⟩
∥B∥F ∥B1∥F


≤0.06≤

1 −
⟨B, B2⟩
∥B∥F ∥B2∥F


.

Proof. According to Lemma 1, we have

∥B1 −B∥F

∥B∥F
≤0.2,
⟨B, B1⟩
∥B∥F ∥B1∥F
≥(1 −0.005)2 ≥0.99.

On the other hand, from Lemma 2(3), we get

∥B2 −B∥F

∥B∥F
= |x −1|
√x + 1 = f1(x),
⟨B, B2⟩
∥B∥F ∥B2∥F
=
2
p

2 + x + 1/x
= f2(x),

where x = (0.02c)s ∈(0, 20−1/4]. It is easy to verify that f1 decreases monotonically and f2
increases monotonically for 0 < x < 1. Hence

f1(x) ≥f1(20−1/4) ≥0.4,
f2(x) ≤f2(20−1/4) ≤0.94.

The proof is completed.

Lemma 12. Let {Xt}t=T
t=1 be a sequence of symmetric matrices, and At = Pt
s=1 Xs, where
t = 1, . . . , T. Suppose we have two sequences of symmetric matrices {Yt}t=T
t=1 , {Zt}t=T
t=0 , and a
sequence real numbers {ρt}t=T
t=0 satisfying

Yt = Zt−1 + Xt,
ρt = ρt−1 + ∥Yt −Zt∥2,
Z0 = 0, ρ0 = 0.

Deﬁne Bt = ρtI + Zt, where I denotes the identity matrix. Then for t = 1, . . . , T, we have

Bt ⪰Bt−1 + Xt,
At ⪯Bt ⪯2ρtI + At.

21


---Page Break---
Proof. Note that for any symmetric matrix S, it holds that ∥S∥2I ⪰S. Then we have

(ρt −ρt−1)I + Zt = ∥Yt −Zt∥2I + Zt ⪰Yt.

Adding ρt−1I on both sides, we get

Bt = ρtI + Zt ⪰ρt−1I + Yt = ρt−1I + Zt−1 + Xt = Bt−1 + Xt.

Hence

Bt =

t
X

s=1
(Bs −Bs−1) ⪰

t
X

s=1
Xs = At.

On the other hand, we have

Zt ⪯∥Zt −Yt∥2I + Yt = (ρt −ρt−1)I + Yt.

Adding ρtI on both sides, we get

Bt = ρtI + Zt ⪯(2ρt −ρt−1)I + Yt
= 2(ρt −ρt−1)I + ρt−1I + Zt−1 + Xt
= Bt−1 + 2(ρt −ρt−1)I + Xt.

Hence

Bt =

t
X

s=1
(Bs −Bs−1) ⪯

t
X

s=1
2(ρs −ρs−1)I +

t
X

s=1
Xs = 2ρtI + At.

The proof is completed.

Theorem 1. Assume that the gradients G1, . . . , GT ∈Rm×n are matrices of rank at most r. Then
for any W ∗∈Rm×n and ϵ > 0, if η = D/
√

2r, the regret of Algorithm 6 is bounded as follows,

T
X

t=1
ft(Wt) −

T
X

t=1
ft(W ∗) ≤
√

2rD[21/4mρ1/4
T
+ tr( ˜L1/4
T )][21/4nµ1/4
T
+ tr( ˜R1/4
T )],

where D = maxt∈[T ] ∥Wt −W ∗∥F , ˜Lt = ϵIm + PT
t=1 GtGT
t , and ˜Rt = ϵIn + PT
t=1 GT
t Gt.

Proof. Deﬁne ˆLt = (ϵ + ρt)Im + Lt, ˆRt = (ϵ + µt)In + Rt. According to Lemma 12, ˆLt and ˆRt
are positive deﬁnite. Recall the update performed in Algorithm 6,

Wt+1 = Wt −η ˆL−1/4
t
Gt ˆR−1/4
t
.

For t > 0, let Ht = ˆL1/4
t
⊗ˆR1/4
t
, gt = vec(Gt) and wt = vec(Wt). Due to Lemma 3(3) and
Lemma 4, we have

wt+1 = wt −ηH−1
t
gt.

Lemma 12 implies 0 ≺ˆL1 ⪯· · · ⪯ˆLT , 0 ≺ˆR1 ⪯· · · ⪯ˆRT . Thus, according to Lemma 3(3)(4)
and Lemma 6, we get

0 ≺H1 ⪯· · · ⪯HT .

Let H0 = 0. By invoking Lemma 9 and Lemma 8, we obtain the regret bound

T
X

t=1
ft(Wt) −

T
X

t=1
ft(W ∗) ≤1

2η

T
X

t=1
(wt −w∗)T(Ht −Ht−1)(wt −w∗) + η

2

T
X

t=1
(∥gt∥∗
Ht)2

≤D2

2η

T
X

t=1
tr(Ht −Ht−1) + η

2

T
X

t=1
(∥gt∥∗
Ht)2

= D2

2η tr(HT ) + η

2

T
X

t=1
(∥gt∥∗
Ht)2,

22


---Page Break---
where D = maxt∈[T ] ∥wt −w∗∥2 = maxt∈[T ] ∥Wt −W ∗∥F and w∗= vec(W ∗).

Deﬁne c
Ht = (rϵI + Pt
s=1 gsgT
s )1/2. Lemma 11 and Lemma 12 imply that

c
Ht ⪯√r ˜L1/4
t
⊗˜R1/4
t
⪯√rHt.

Using Lemma 7 and Lemma 10 along with the above equation, we obtain

T
X

t=1
(∥gt∥∗
Ht)2 ≤√r

T
X

t=1
(∥gt∥∗
c
Ht)2 ≤2√rtr(c
HT ) ≤2rtr(HT ).

Consequently, using Lemma 3(5) and Lemma 12, we get the desired regret bound

T
X

t=1
ft(Wt) −

T
X

t=1
ft(W ∗) ≤
D2

2η + ηr

tr(HT ) =
√

2rDtr( ˆL1/4
T )tr( ˆR1/4
T )

≤
√

2rD[21/4mρ1/4
T
+ tr( ˜L1/4
T )][21/4nµ1/4
T
+ tr( ˜R1/4
T )],

by choosing η = D/
√

2r. The proof is completed.

G
Experimental Details

We use one RTX3060Ti GPU under the PyTorch 2.0.1+CUDA11.8 framework for DNN training on
the CIFAR-100 and Tiny-ImageNet datasets, use one A800 GPU under the PyTorch 2.0.1+CUDA11.7
framework for DNN training on the ImageNet-1k and C4 datasets, and use two NVIDIA L40S GPUs
under the PyTorch 2.0.1+CUDA11.8 framework for DNN training on the OWT dataset. To obtain the
total peak memory consumption per GPU, we call "torch.cuda.max_memory_allocated".

We set "torch.backends.cudnn.benchmark" to "False" for all the experiments, except when training
ViT-Base/32 on the ImageNet-1k dataset. We report the total memory consumption instead of
the memory consumption of the second-order optimizer. This total memory includes data, model
parameters, activations, gradients, states forming the preconditioners and their inverse roots, states
for the used ﬁrst-order optimizer, and memory fragments. Our focus lies in quantizing the states for
constructing preconditioners and their inverse roots, which are approximately 7x smaller for 4-bit
Shampoo compared to 32-bit Shampoo. Because the block size is 64, its maximum value should be
calculated every 64 elements and saved as a 32-bit value, resulting in an additional overhead of 0.5
bits (32/64). Consequently, the memory savings are approximately 7 times, calculated as 32/(4+0.5).
In the future, we may adopt double quantization [9] to further reduce memory consumption.

For SGDM, Adagrad or AdamW used in second-order optimizers, we use 32-bit optimizer states on
image classiﬁcation tasks and 16-bit optimizer states on natural language modeling tasks by default.
For SGDM, we set the momentum to 0.9 and use an initial learning rate of 0.1. For Adagrad, we set
ϵ = 10−10 and use an initial learning rate of 0.01. For AdamW, we set β1 = 0.9, β2 = 0.999, and
ϵ = 10−8 and use an initial learning rate of 0.001. For quantization settings, we employ block-wise
normalization with a block size of 64 and linear square quantization by default. Matrices with a size
smaller than 4096 will not be quantized. For Shampoo and CASPR, we use ϵ = 10−6, β = 0.95 and
t1 = 1, t2 = 4 by default. Shampoo and CASPR precondition blocks from large matrices and the
maximum order of a preconditioner is 10000 for 130M LLAMA-2 and is 1200 for other models. For
training loss, we use cross-entropy loss. For image classiﬁcation tasks, automatic mixed precision is
enabled except for training transformers on the CIFAR-100 and Tiny-ImageNet datasets.

Settings on training CNNs on CIFAR-100 or Tiny-ImageNet. Minibatch size is set to 128. Weight
decay is 0.0005. Data augmentation includes random crop and horizontal ﬂip. For Shampoo, we set
T1 = 100 and T2 = 500. In Section 5, we run SGDM for 300 epochs and SGDM+Shampoo for 200
epochs on the CIFAR-100 dataset. We run SGDM for 150 epochs and SGDM+Shampoo for 100
epochs on the Tiny-ImageNet dataset. We adopt the multi-step learning rate schedule (the learning
rate is multiplied by 0.1 for every 30% epochs with a linear warmup at the ﬁrst 5 epochs).

Settings on training transformers on CIFAR-100 or Tiny-ImageNet. We set a patch size of 4
for ViT-small on the CIFAR-100 dataset, and a patch size of 8 for ViT-small on the Tiny-ImageNet
dataset. For training Swin-Tiny on the CIFAR-100 dataset, we use a patch size of 2 and window size
of 4. For training Swin-Tiny on the Tiny-ImageNet dataset, we use a patch size of 4 and window

23


---Page Break---
size of 7. Minibatch size is set to 128. We run Adagrad/AdamW/NadamW for 150 epochs and
Adagrad/AdamW+Shampoo for 100 epochs. Weight decay is 0.0005 for Adagrad, and is 0.05 for
AdamW/NadamW. We use the cosine learning rate schedule. Data augmentation follows the source
code in [25]. For Shampoo, we set T1 = 100 and T2 = 500. With the exception of certain optimizer
settings, the conﬁgurations used for ablation studies are identical to those outlined above.

Settings on training ResNet50 on ImageNet-1k.
We run SGDM for 120 epochs and
SGDM+Shampoo for 100 epochs. Minibatch size is set to 256. Weight decay is 0.0001. We
adopt the multi-step learning rate schedule (the learning rate is multiplied by 0.1 for every 30%
epochs with a linear warmup at the ﬁrst 5 epochs). Data augmentation includes random resized crop,
horizontal ﬂip, and color jitter. For Shampoo, we set T1 = 200 and T2 = 1000.

Settings on training ViT-Base/32 on ImageNet-1k.
We run AdamW for 150 epochs and
AdamW+Shampoo for 120 epochs. Minibatch size is set to 512. Weight decay is 0.05. We use the
cosine learning rate schedule. Data augmentation follows the conﬁguration for training ViT-Base/16
in [44], excluding repeated augmentation. For Shampoo, we set T1 = 200 and T2 = 1000.

Settings on training GPT-2 on OWT. We run AdamW with 10% warmup steps. Total batch size
is set to 480. Batch size is set to 24 for training 124M GPT-2. Dtype is bﬂoat16. Weight decay is
0.1. For Shampoo, we set T1 = 200 and T2 = 200. For our 4-bit Shampoo, we use Schur-Newton
iteration used in Algorithm 4 to compute the inverse root of a preconditioner for training stability.

Settings on training LLAMA-2 on C4. We run AdamW with 10% warmup steps. Total batch size is
set to 512. Batch size is set to 256 for training 130M LLAMA-2 and is set to 128 for training 350M
LLAMA-2. Dtype is bﬂoat16. Weight decay is 0. For Shampoo, we set T1 = 200 and T2 = 200.

Settings on K-FAC and AdaBK. K-FAC/AdaBK preconditions layers without limiting the size
of a preconditioner. We set β = 0.9, T1 = 200, and T2 = 2000. We use ϵ = 0.1 for K-FAC and
ϵ = 0.001 for AdaBK. For 4-bit K-FAC/AdaBK, we set t1 = 0 and t2 = 0 (i.e., no orthogonal
rectiﬁcation).

Settings on schedule free optimization.
We use the code from [6] to train ResNet34 with
SGDScheduleFree and Swin-Tiny with AdamWScheduleFree. For SGDScheduleFree, we set
lr=1.0, weight_decay=0.0005 and warmup_steps=2000. For AdamWScheduleFree, we set lr=0.0025,
weight_decay=0.05 and warmup_steps=10000.

Settings on M-FAC. We use the code from [15] and set ngrads=32, damp=0.1. The other hyperpa-
rameter settings of M-FAC is the same as that of SGDM used for ResNet34 training.

H
Additional Results

H.1
Image Classiﬁcation

More learning rate schedulers. Table 8 shows the performance and wall-clock time of training
ResNet34 on CIFAR-100 with cosine learning rate decay. By comparison, SGDM+Shampoo still
converges faster than SGDM, and have slightly better test performance.

Table 8: Performance and wall-clock time of training ResNet34 on the CIFAR-100 dataset with
cosine learning rate decay. TA = test accuracy, and WCT = wall-clock time.

Epochs
Optimizer
TA (%)
WCT (min)

200
SGDM
79.67
116.0
300
SGDM
79.83
172.7
200
SGDM + 32-bit Shampoo
80.39
152.7
200
SGDM + 4-bit Shampoo (our)
80.22
161.7

We also provide the results of training ResNet34 and Swin-Tiny on CIFAR-100 with schedule-
free approach [6] in Table 9. From it one can see that AdamWScheduleFree achieves comparable
performance to AdamW with cosine decay, while SGDScheduleFree underperforms compared to
SGDM. We observe that this schedule-free algorithm shows rapid improvements in training and test
accuracy during the early training stages, but may fail to achieve a higher test accuracy ultimately
(see Figure 9). Anyway, these methods are still worse than our AdamW+4-bit Shampoo.

24


---Page Break---
Table 9: Performance and wall-clock time of training on the CIFAR-100 dataset with cosine learning
rate decay and schedule-free approach. ResNet34 is trained for 300 epochs and Swin-Tiny is trained
for 150 epochs. TA = test accuracy, and WCT = wall-clock time.

Model
Optimizer
TA (%)
WCT (min)

ResNet34
SGDM
79.83
172.7
SGDScheduleFree
75.63
169.6

Swin-Tiny
AdamW
76.69
318.6
AdamWScheduleFree
76.58
321.9

0
50
100
150
200
250
300
Epoch

50

55

60

65

70

75

80

Test Accuracy (%)

ResNet34 on CIFAR-100

SGDM
SGDScheduleFree

0
20
40
60
80
100
120
140
Epoch

50

55

60

65

70

75

Test Accuracy (%)

Swin-Tiny on CIFAR-100

AdamW
AdamW+32-bit Shampoo

Figure 9: Visualization of test accuracies on the CIFAR-100 dataset with cosine learning rate decay
and schedule-free approach.

More optimizers. Table 10 shows results of training Swin-Tiny on CIFAR-100 with NadamW,
Adagrad and Adagrad+Shampoo. One can see that Adagrad+4-bit Shampoo converges faster than
Adagrad with ignorable extra memory overhead, and also has higher test accuracy. Besides, though
NadamW [11] is slightly better than AdamW, it is still worse than our AdamW+4-bit Shampoo.

Table 10: Performance, wall-clock time, and memory cost of training Swin-Tiny on the CIFAR-100
dataset. TA = test accuracy, WCT = wall-clock time, and TMC = total GPU memory cost.

Optimizer
TA (%)
WCT (min)
TMC (MB)

NadamW
77.11
342.4
1465.8
AdamW + 32-bit Shampoo
79.34
260.8
2036.0
AdamW + 4-bit Shampoo (our)
78.63
273.3
1543.9

Adagrad
66.56
294.6
1354.9
Adagrad + 32-bit Shampoo
73.55
245.3
1930.4
Adagrad + 4-bit Shampoo (our)
72.66
259.6
1433.0

M-FAC [15] is a matrix-free method computing inverse-Hessian vector products with many gradient
copies. It is not memory-efﬁcient for M-FAC to maintain m dense gradient copies (m = 1024 in its
ofﬁcial code). Table 11 shows that both SGDM+32-bit Shampoo and SGDM+4-bit Shampoo enjoy
much higher efﬁciency than M-FAC (m = 32) for training ResNet34 on CIFAR-100, and enjoy higher
test accuracy. EVA [42] is a rank-one second-order optimizer and is memory-efﬁcient. We train
ResNet34 on CIFAR-100 with SGDM+EVA, but despite extensive hyper-parameter tuning, we fail to
achieve acceleration over SGDM. Instead, we cite EVA’s result of training VGG-19 on CIFAR-100
for 200 epochs (see Table 2 in [42]). The test accuracies of SGDM+EVA and SGDM+Shampoo are
73% and 74.5%, respectively.

25


---Page Break---
Table 11: Performance and memory cost of training ResNet34 on the CIFAR-100 dataset with cosine
learning rate decay. All the optimizers are run for 200 epochs. TA = test accuracy, and TMC = total
GPU memory cost.

Optimizer
SGDM M-FAC (m=32) SGDM + 32-bit Shampoo SGDM + 4-bit Shampoo (our)

TA (%)
79.67
78.56
80.39
80.22

TMC (MB) 822.03
3424.8
1441.8
908.4

Table 12: Performance, wall-clock time, and memory usage per GPU on natural language modeling
tasks. VL = validation loss, WCT = wall-clock time, and TMC = total GPU memory cost.

Dataset
Model
Optimizer
VL
WCT (min)
TMC (MB)

C4

LLAMA-130M

AdamW
3.214
346.9
47026
AdamW + 32-bit Shampoo
3.184
353.7
48813
AdamW + 4-bit Shampoo (naive)
3.200
353.5
47316
AdamW + 4-bit Shampoo (our)
3.194
353.1
47318

LLAMA-350M

AdamW
2.939
2687
54184
AdamW + 32-bit Shampoo
2.908
2776
59149
AdamW + 4-bit Shampoo (naive)
2.930
2753
54894
AdamW + 4-bit Shampoo (our)
2.924
2795
54894

OWT
GPT2-124M

AdamW
2.954
2310
27010
AdamW + 32-bit Shampoo
2.936
2330
28490
AdamW + 4-bit Shampoo (naive)
2.953
2359
27209
AdamW + 4-bit Shampoo (our)
2.944
2311
27209

260
280
300
320
340
360
380
Wall-clock Time (min)

3.18

3.19

3.20

3.21

3.22

3.23

3.24

Validation Loss

LLAMA-130M on C4

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

1900
2000
2100
2200
2300
2400
Wall-clock Time (min)

2.93

2.94

2.95

2.96

2.97

2.98

Validation Loss

GPT2-124M on OWT

AdamW
AdamW+32-bit Shampoo
AdamW+4-bit Shampoo (naive)
AdamW+4-bit Shampoo (our)

Figure 10: Visualization of validation loss on the C4 and OWT datasets.

H.2
Natural Language Modeling

Models, datasets, and hyperparameters. We train 124M GPT-2 [32] for 60k steps on the Open-
WebText (OWT) dataset * following the nanoGPT codebase † with two NVIDIA L40S GPUs, and
train 130M LLAMA-2 [37] for 20k steps and 350M LLAMA-2 for 60k steps on the C4 dataset [33]
following [43] with one A800 GPU. See Appendix G for experimental details.

Main results. We show the performance, wall-clock time, and memory cost in Table 12, and
the validation loss curves in Figure 10. As with the vision tasks, our AdamW+4-bit Shampoo
consistently outperformed AdamW and naive AdamW+4-bit Shampoo in terms of performance, and
AdamW+32-bit Shampoo in terms of memory usage.

Memory efﬁciency. We further check the memory usage by increasing token batch size for a language
model, which is calculated as the batch size multiplied by the context length (see [43]). To train
LLAMA2-7B on the C4 dataset using a single A800 GPU (with a maximum memory of 81,920

*http://Skylion007.github.io/OpenWebTextCorpus.
†https://github.com/karpathy/nanoGPT.

26


---Page Break---
MB), we set the context length to 256 and then determine the maximum batch size allowed by each
optimizer. For Shampoo, the maximum order of a preconditioner for training LLAMA2-7B is 2048.
In all experiments, gradient checkpointing is enabled. Table 13 summarizes the evaluation results. By
comparison, the 32-bit Shampoo runs out of memory with a batch size of 2, while our 4-bit Shampoo
supports a batch size of 64 for standard training and only encounters memory issues at a batch size
of 128. These results clearly demonstrate that our 4-bit Shampoo signiﬁcantly conserves memory
compared to the 32-bit version.

Table 13: Memory cost of training LLAMA2-7B on the C4 dataset with different optimizers. One
A800 GPU with a maximum memory of 81,920 MB is enabled. TMC = total GPU memory cost, and
OOM = out of memory.

Optimizer
Batch Size
TMC (MB)

8-bit AdamW
64
60135
8-bit AdamW
128
68689
8-bit AdamW
256
OOM
8-bit AdamW + 32-bit Shampoo
2
OOM
8-bit AdamW + 4-bit Shampoo (our)
64
74561
8-bit AdamW + 4-bit Shampoo (our)
128
OOM

27


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the paper’s
contributions and scope?
Answer: [Yes]
Justiﬁcation: We propose the ﬁrst second-order optimizers with 4-bit states by taking Shampoo as
an example, while preserving the performance achieved with 32-bit optimizer states.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in
the paper.
• The abstract and/or introduction should clearly state the claims made, including the contribu-
tions made in the paper and important assumptions and limitations. A No or NA answer to
this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reﬂect how much
the results can be expected to generalize to other settings.
• It is ﬁne to include aspirational goals as motivation as long as it is clear that these goals are
not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justiﬁcation: We discuss the limitations of the work at the end of the paper.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the
paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to vi-
olations of these assumptions (e.g., independence assumptions, noiseless settings, model
well-speciﬁcation, asymptotic approximations only holding locally). The authors should
reﬂect on how these assumptions might be violated in practice and what the implications
would be.
• The authors should reﬂect on the scope of the claims made, e.g., if the approach was only
tested on a few datasets or with a few runs. In general, empirical results often depend on
implicit assumptions, which should be articulated.
• The authors should reﬂect on the factors that inﬂuence the performance of the approach. For
example, a facial recognition algorithm may perform poorly when image resolution is low or
images are taken in low lighting. Or a speech-to-text system might not be used reliably to
provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efﬁciency of the proposed algorithms and how
they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover limita-
tions that aren’t acknowledged in the paper. The authors should use their best judgment and
recognize that individual actions in favor of transparency play an important role in developing
norms that preserve the integrity of the community. Reviewers will be speciﬁcally instructed
to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?
Answer: [Yes]
Justiﬁcation: We provide all the proofs in the Appendix.
Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.

28


---Page Break---
• The proofs can either appear in the main paper or the supplemental material, but if they
appear in the supplemental material, the authors are encouraged to provide a short proof
sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by
formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experi-
mental results of the paper to the extent that it affects the main claims and/or conclusions of the
paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justiﬁcation: We present the implementation details of all the experiments in the Appendix.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by
the reviewers: Making the paper reproducible is important, regardless of whether the code
and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to
make their results reproducible or veriﬁable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For
example, if the contribution is a novel architecture, describing the architecture fully might
sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation, it may be necessary
to either make it possible for others to replicate the model with the same dataset, or provide
access to the model. In general. releasing code and data is often one good way to accomplish
this, but reproducibility can also be provided via detailed instructions for how to replicate the
results, access to a hosted model (e.g., in the case of a large language model), releasing of a
model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions
to provide some reasonable avenue for reproducibility, which may depend on the nature of
the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to
reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe the
architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct the
dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors
are welcome to describe the particular way they provide for reproducibility. In the case
of closed-source models, it may be that access to the model is limited in some way (e.g.,
to registered users), but it should be possible for other researchers to have some path to
reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instructions to
faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes]
Justiﬁcation: We use publicly available datasets and will release our source code.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including
code, unless this is central to the contribution (e.g., for a new open-source benchmark).

29


---Page Break---
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https://
nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to
access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed
method and baselines. If only a subset of experiments are reproducible, they should state
which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions
(if applicable).
• Providing as much information as possible in supplemental material (appended to the paper)
is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters,
how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justiﬁcation: We present the implementation details of all the experiments in the Appendix.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that
is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?
Answer: [No]
Justiﬁcation: Error bars are not reported because it would be too computationally expensive.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁdence
intervals, or statistical signiﬁcance tests, at least for the experiments that support the main
claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example,
train/test split, initialization, random drawing of some parameter, or overall run with given
experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a
library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the
mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably
report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality
of errors is not veriﬁed.
• For asymmetric distributions, the authors should be careful not to show in tables or ﬁgures
symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they
were calculated and reference the corresponding ﬁgures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the experi-
ments?
Answer: [Yes]
Justiﬁcation: We present the implementation details of all the experiments in the main paper and
Appendix.
Guidelines:

• The answer NA means that the paper does not include experiments.

30


---Page Break---
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or
cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experi-
mental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the
experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it
into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justiﬁcation: We review the NeurIPS Code of Ethics and our paper conforms it.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration
due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?
Answer: [Yes]
Justiﬁcation: We discuss both potential positive societal impacts and negative societal impacts at
the end of the paper.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact
or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake proﬁles, surveillance), fairness considerations (e.g., deploy-
ment of technologies that could make decisions that unfairly impact speciﬁc groups), privacy
considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to
particular applications, let alone deployments. However, if there is a direct path to any
negative applications, the authors should point it out. For example, it is legitimate to point out
that an improvement in the quality of generative models could be used to generate deepfakes
for disinformation. On the other hand, it is not needed to point out that a generic algorithm
for optimizing neural networks could enable people to train models that generate Deepfakes
faster.
• The authors should consider possible harms that could arise when the technology is being
used as intended and functioning correctly, harms that could arise when the technology is
being used as intended but gives incorrect results, and harms following from (intentional or
unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms
for monitoring misuse, mechanisms to monitor how a system learns from feedback over time,
improving the efﬁciency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of
data or models that have a high risk for misuse (e.g., pretrained language models, image generators,
or scraped datasets)?
Answer: [NA]
Justiﬁcation: We present 4-bit Shampoo for memory efﬁcient training of deep models. It poses no
such risks.
Guidelines:

• The answer NA means that the paper poses no such risks.

31


---Page Break---
• Released models that have a high risk for misuse or dual-use should be released with necessary
safeguards to allow for controlled use of the model, for example by requiring that users
adhere to usage guidelines or restrictions to access the model or implementing safety ﬁlters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should
describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not
require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the
paper, properly credited and are the license and terms of use explicitly mentioned and properly
respected?
Answer: [Yes]
Justiﬁcation: We properly mention all the existing assets.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service
of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the
derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the
asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justiﬁcation: We provide anonymized zip ﬁle of our code.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset is
used.
• At submission time, remember to anonymize your assets (if applicable). You can either create
an anonymized URL or include an anonymized zip ﬁle.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justiﬁcation: Our paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is ﬁne, but if the main contribution
of the paper involves human subjects, then as much detail as possible should be included in
the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or
other labor should be paid at least the minimum wage in the country of the data collector.

32


---Page Break---
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Sub-
jects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals
(or an equivalent approval/review based on the requirements of your country or institution) were
obtained?
Answer: [NA]
Justiﬁcation: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may
be required for any human subjects research. If you obtained IRB approval, you should
clearly state this in the paper.
• We recognize that the procedures for this may vary signiﬁcantly between institutions and
locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines
for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

33


---Page Break---
