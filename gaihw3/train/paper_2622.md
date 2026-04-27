Noise Balance and Stationary Distribution of
Stochastic Gradient Descent

Anonymous Author(s)
Affiliation
Address
email

Abstract

How the stochastic gradient descent (SGD) navigates the loss landscape of a neu-
1

ral network remains poorly understood. This work shows that the minibatch noise
2

of SGD regularizes the solution towards a noise-balanced solution whenever the
3

loss function contains a rescaling symmetry. We prove that when the rescaling
4

symmetry exists, the SGD dynamics is limited to only a low-dimensional sub-
5

space and prefers a special set of solutions in an infinitely large degenerate man-
6

ifold, which offers a partial explanation of the effectiveness of SGD in training
7

neural networks. We then apply this result to derive the stationary distribution
8

of stochastic gradient flow for a diagonal linear network with arbitrary depth and
9

width, which is the first analytical expression of the stationary distribution of SGD
10

in a high-dimensional non-quadratic potential. The stationary distribution exhibits
11

complicated nonlinear phenomena such as phase transitions, loss of ergodicity,
12

memory effects, and fluctuation inversion. These phenomena are shown to exist
13

uniquely in deep networks, highlighting a fundamental difference between deep
14

and shallow models. Lastly, we discuss the implication of the proposed theory for
15

the practical problem of variational Bayesian inference.
16

1
Introduction
17

In natural and social sciences, one of the most important objects of study of a stochastic system is
18

its stationary distribution, which is often found to offer fundamental insights into understanding a
19

given stochastic process [36, 29]. Arguably, a great deal of insights into SGD can be obtained if we
20

have an analytical understanding of its stationary distribution, which remains unknown until today.
21

The stochastic gradient descent (SGD) algorithm is defined as ∆θt = −η

S ∑x∈B ∇θℓ(θ,x), where θ
22

is the model parameter and ℓ(θ,x) is a per-sample loss whose expectation over x gives the training
23

loss: L(θ) = Ex[ℓ(θ,x)]. B is a randomly sampled minibatch of data points, each independently
24

sampled from the training set, and S is the minibatch size. Two aspects of the algorithm make it
25

difficult to understand this algorithm: (1) its dynamics is discrete in time, and (2) the randomness is
26

highly nonlinear and parameter-dependent. This work relies on the continuous-time approximation
27

and deals with the second aspect.
28

The main contributions are
29

1. the derivation of the “law of balance,” which shows that SGD converges to a special subset of
30

noised-balanced solutions when the rescaling symmetry is present;
31

2. the first-of-its-kind solution of the stationary distribution of an analytical model trained by SGD;
32

3. discovery of novel phenomena such as phase transitions, loss of ergodicity, memory effects, and
33

fluctuation inversion, all implied by our theory.
34

Organization. The next section discusses the closely related works. In Section 3, we prove the
35

law of balance, the first main result of this work, and discuss its implications for common neural
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
networks. In Section 4, we apply the law of balance to derive the stationary distribution of SGD for
37

a highly nontrivial loss landscape. The last section concludes this work. All proofs and derivations
38

are given in Appendix A.
39

2
Related Works
40

Solution of the Fokker Planck (FP) Equation. The FP equation is a high-dimensional partial
41

differential equation whose solution (and its existence) is an open problem in mathematics and many
42

fields of sciences and only known for a few celebrated special cases [28]. Our solution is the first of
43

its kind in a deep-learning setting. Stationary distribution of SGD. One of the earliest works that
44

computes the stationary distribution of SGD is the Lemma 20 of Ref. [3], which assumes that the
45

noise has a constant covariance and shows that if the loss function is quadratic, then the stationary
46

distribution is Gaussian. Similarly, using a saddle point expansion and assuming that the noise is
47

parameter-independent, a series of recent works showed that the stationary distribution of SGD is
48

exponential in the model parameters close to a local minimum: p(θ) ∝exp[−aθT Hθ], for some
49

constant a and matrix H [21, 41, 19]. Assuming that the noise covariance only depends on the loss
50

function value L(θ), Refs. [24] and [39] showed that the stationary distribution is power-law-like
51

and proportional to L(θ)−c0 for some constant c0. A primary feature of these previous results is that
52

stationary distribution does not exhibit any memory effect and also preserves ergodicity. Until now,
53

no analytical solution to the stationary distribution of SGD is known, making it impossible to judge
54

how good the previous approximate results are. Our result is the first to derive an exact solution to
55

the stationary distribution of SGD without any approximation. We will see that in contrast to the
56

approximate solutions in the previous results, the actual distribution of SGD has both a memory
57

effect and features the loss of ergodicity.
58

Symmetry and SGD dynamics. Also related to our work is the study of how symmetry affects the
59

learning dynamics of SGD. A major prior work is [17], which studies the dynamics of SGD when
60

there is scale invariance, conjecturing that SGD reaches a fast equilibrium state at the early stage of
61

training. Our result is different as we study a different type of symmetry, the rescaling symmetry.
62

3
Noise Balance
63

We consider the continuous-time limit of SGD [15, 16, 18, 32, 8, 11]:
64

dθ = −∇θLdt +
√

TC(θ)dWt,
(1)

where C(θ) = E[∇ℓ(θ)∇T ℓ(θ)] is the gradient covariance, dWt is a stochastic process satisfying
65

dWt ∼N(0,Idt) and E[dWtdW T
t′ ] = δ(t−t′)I, and T = η/S. Apparently, T gives the average noise
66

level in the dynamics. Previous works have suggested that the ratio T is a main factor determining
67

the behavior of SGD, and using different T often leads to different generalization performance
68

[31, 19, 44].
69

3.1
Rescaling Symmetry and Law of Balance
70

Due to standard architecture designs, a type of invariance – the rescaling symmetry – often appears
71

in the loss function and it is preserved for all sampling of minibatches. The per-sample loss ℓis said
72

to have the rescaling symmetry for all x if ℓ(u,w,x) = ℓ(λu,w/λ,x) for a scalar λ ∈R+. This
73

type of symmetry appears in many scenarios in deep learning. For example, it appears in any neural
74

network with the ReLU activation. It also appears in the self-attention of transformers, often in the
75

form of key and query matrices [37]. When this symmetry exists between u and w, one can prove
76

the following result, which we refer to as the law of balance.
77

Theorem 3.1. Let u, w, and v be parameters of arbitrary dimensions. Let ℓ(u,w,v,x) satisfy
78

ℓ(u,w,v,x) = ℓ(λu,w/λ,v,x) for arbitrary x and any λ ∈R+. Then,
79

d
dt(∣∣u∣∣2 −∣∣w∣∣2) = −T(uT C1u −wT C2w),
(2)

where C1 = E[AT A] −E[AT ]E[A], C2 = E[AAT ] −E[A]E[AT ] and Aki = ∂˜ℓ/∂(uiwk) with
80

˜ℓ(uiwk,v,x) ≡ℓ(ui,wk,v,x).1
81

1This result also holds using the modified loss (See Appendix A.3).

2


---Page Break---
Here, v stands for the parameters that are irrelevant to the symmetry, and C1 and C2 are positive
82

semi-definite by definition. The theorem still applies if the model has parameters other than u and
83

w. The theorem can be applied recursively when multiple rescaling symmetries exist. See Figure 1
84

for an illustration the the dynamics and how it differs from other types of GD.
85

Figure 1: Dynamics of GD and SGD
and GD with injected Gaussian noise
for the simple problem ℓ(u, w)
=
(uwx−y)2. Due to the rescaling sym-
metry between u and w, GD follows
a conservation law: u2(t) −w2(t) =
u2(0) −w2(0), SGD converges to the
balanced solution u2
= w2, while
GD with injected noise diverges due to
simple diffusion in the degenerate di-
rections.

While the matrices C1 and C2 may not always be full-rank, we
86

emphasize that in common deep-learning settings with rescal-
87

ing symmetry, the law of balance is almost always well-defined
88

and applicable. In Appendix A.4, we prove that under very
89

general settings, for all active hidden neurons of a two-layer
90

ReLU net, C1 and C2 are always full-rank. Equation (2) is the
91

law of balance, and it implies two different types of balance.
92

The first type of balance is the balance of gradient noise. The
93

proof of the theorem shows that the stationary point of the law
94

in (2) is equivalent to
95

Trw[C(w)] = Tru[C(u)],
(3)
where C(w) and C(u) are the gradient covariance of w and
96

u, respectively. Therefore, SGD prefers a solution where the
97

gradient noise between the two layers is balanced. Also, this
98

implies that the balance conditions of the law is only dependent
99

on the diagonal terms of the Fisher information (if we regard
100

the loss as a log probability), which is often well-behaved. As
101

a last caveat, we emphasize that the fact that the noise will
102

balance does not imply that either trace will converge or stay
103

close to a fixed value – it is also possible for both terms to
104

oscillate while their difference is close to zero.
105

The second type is the norm ratio balance between layers,
106

though the norm ratio may not necessarily be finite. Equation (2) implies that in the degenerate
107

direction of the rescaling symmetry, a single and unique point is favored by SGD. Let u = λu∗
108

and w = λ−1w∗for arbitrary u∗and w∗, then, the stationary point of the law is reached at
109

λ4 = (w∗)T C2w∗

(u∗)T C1u∗. The quantity λ can be called the “balancedness” of the norm, and the law states
110

that when a rescaling symmetry exists, a special balancedness is preferred by the SGD algorithm.
111

When C1 or C2 vanishes, λ or λ−1 diverges, and so does SGD. Therefore, having a nonvanishing
112

noise actually implies that SGD training will be more stable. For common problems, C1 and C2
113

are positive definite and, thus, if we know the spectrum of C1 and C2 at the end of training, we can
114

estimate a rough norm ratio at convergence:
115

−T(λ1M∣∣u∣∣2 −λ2m∣∣w∣∣2) ≤d

dt(∣∣u∣∣2 −∣∣w∣∣2) ≤−T(λ1m∣∣u∣∣2 −λ2M∣∣w∣∣2),

where λ1m(2m) and λ1M(2M) represent the minimal and maximal eigenvalue of the matrix C1(2),
116

respectively. Thefore, the value of ∣∣u∣∣2/∣∣w∣∣2 is restricted by (See Section A.5)
117

λ2m
λ1M
≤∣∣u∣∣2

∣∣w∣∣2 ≤λ2M

λ1m
.
(4)

Thus, a remaining question is whether the quantities uT C1u and wT C2w are generally well-defined
118

and nonvanishing or not. The following proposition shows that for a generic two-layer ReLU net,
119

uT C1u and wT C2w are almost everywhere strictly positive. We define a two-layer ReLU net as
120

f(x) =
d
∑
i
uiReLU(wT
i x + bi),
(5)

where ui ∈Rdu,wi ∈Rdw and bi is a scalar with i being the index of the hidden neuron. For each
121

i, the model has the rescaling symmetry: ui →λui, (wi,bi) →(λ−1wi,λ−1bi). We thus apply the
122

law of balance to each neuron separately. The per-sample loss function is
123

ℓ(θ,x) = ∥f(x) −y(x,ϵ)∥2.
(6)
Here, x has a full-rank covariance Σx, and y = g(x) + ϵ for some function g and ϵ is a zero-mean
124

random vector independent of x and have the full-rank covariance Σϵ. The following theorem shows
125

that for this network, C1 and C2 are full rank unless the neuron is “dead”.
126

3


---Page Break---
Figure 2: A two-layer ReLU network trained on a full-rank dataset. Left: because of the rescaling symmetry,
the norms of the two layers are balanced approximately (but not exactly). Right: the first and second terms
in Eq. (2). We see that both terms evolve towards a point where they exactly balance. In agreement with our
theory, SGD training leads to an approximate norm balance and exact gradient noise balance.

Theorem 3.2. Let the loss function be given in Eq. (6). Let C(i)
1
and C(i)
2
denote the corresponding
127

noise matrices of the i-th neuron, and pi ∶= P(wT
i x + bi > 0). Then, C(i)
1
and C(i)
2
are full-rank for
128

all i such that pi > 0.
129

See Figure 2. We train a two-layer ReLU network with the number of neurons: 20 →200 →20.
130

The dataset is a synthetic data set, where x is drawn from a normal distribution, and the labels:
131

y = x+ϵ, for an independent Gaussian noise ϵ with unit variance. While every neuron has a rescaling
132

symmetry, we focus on the overall rescaling symmetry between the two weight matrices. The norm
133

between the two layers reach a state of approximate balance – but not a precise balance. At the same
134

time, the model evolves during training towards a state where uT C1u and wT C2w are balanced.
135

Standard analysis shows that the difference between SGD and GD is of order T 2 per unit time step,
136

and it is thus often believed that SGD can be understood perturbatively through GD [11]. However,
137

the law of balance implies that the difference between GD and SGD is not perturbative. As long
138

as there is any level of noise, the difference between GD and SGD at stationarity is O(1). This
139

theorem also implies the loss of ergodicity, an important phenomenon in nonequilibrium physics
140

[26, 34, 22, 35], because not all solutions with the same training loss will be accessed by SGD with
141

equal probability.
142

3.2
1d Rescaling Symmetry
143

The theorem greatly simplifies when both u and w are one-dimensional.
144

Corollary 3.3. If u,w ∈R, then, d

dt∣u2 −w2∣= −TC0∣u2 −w2∣, where C0 = Var[
∂ℓ
∂(uw)].
145

Before we apply the theorem to study the stationary distributions, we stress the importance of this
146

balance condition. This relation is closely related to Noether’s theorem [23, 1, 20]. If there is no
147

weight decay or stochasticity in training, the quantity ∣∣u∣∣2−∣∣w∣∣2 will be a conserved quantity under
148

gradient flow [6, 14, 33], as is evident by taking the infinite S limit. The fact that it monotonically
149

decays to zero at a finite T may be a manifestation of some underlying fundamental mechanism. A
150

more recent result in Ref. [38] showed that for a two-layer linear network, the norms of two layers
151

are within a distance of order O(η−1), suggesting that the norm of the two layers are balanced. Our
152

result agrees with Ref. [38] in this case, but our result is stronger because our result is nonperturba-
153

tive, only relies on the rescaling symmetry, and is independent of the loss function or architecture
154

of the model. It is useful to note that when L2 regularization with strength γ is present, the rate
155

of decay changes from TC0 to TC0 + γ. This points to a nice interpretation that when rescaling
156

symmetry is present, the implicit bias of SGD is equivalent to weight decay. See Figure 1 for an
157

illustration of this point.
158

Example: two-layer linear network.
It is instructive to illustrate the application of the law to
159

a two-layer linear network, the simplest model that obeys the law. Let θ = (w,u) denote the set
160

of trainable parameters; the per-sample loss is ℓ(θ,x) = (∑d
i uiwix −y)2 + γ∣∣θ∣∣2. Here, d is the
161

width of the model, γ∣∣θ∣∣2 is the L2 regularization term with strength γ ≥0, and Ex denotes the
162

averaging over the training set, which could be a continuous distribution or a discrete sum of delta
163

distributions. It will be convenient for us also to define the shorthand: v ∶= ∑d
i uiwi. The distribution
164

of v is said to be the distribution of the “model.” Applying the law of balance, we obtain that
165

d
dt(u2
i −w2
i ) = −4[T(α1v2 −2α2v + α3) + γ](u2
i −w2
i ),
(7)

4


---Page Break---
where we have introduced the parameters
166

α1 ∶= Var[x2],
α2 ∶= E[x3y] −E[x2]E[xy],
α3 ∶= Var[xy].
(8)

When α1α3 −α2
2 or γ > 0, the time evolution of ∣u2 −w2∣can be upper-bounded by an exponentially
167

decreasing function in time: ∣u2
i −w2
i ∣(t) < ∣u2
i −w2
i ∣(0)exp(−4T(α1α3 −α2
2)t/α1 −4γt) →0.
168

Namely, the quantity (u2
i −w2
i ) decays to 0 with probability 1. We thus have u2
i = w2
i for all
169

i ∈{1,⋯,d} at stationarity, in agreement with the Corollary.
170

4
Stationary Distribution of SGD
171

As an important application of the law of balance, we solve the stationary distribution of SGD
172

for a deep diagonal linear network. While linear networks are limited in expressivity, their loss
173

landscape and dynamics are highly nonlinear and exhibits many shared phenomenon with nonlinear
174

neural networks [13, 30]. Let θ follow the high-dimensional Wiener process given by Eq.(1). The
175

probability density evolves according to its Kolmogorov forward (Fokker-Planck) equation:
176

∂
∂tp(θ,t) = −∑
i

∂
∂θi
(p(θ,t) ∂

∂θi
L(θ)) + 1

2 ∑
i,j

∂2

∂θi∂θj
Cij(θ)p(θ,t).
(9)

The solution of this partial differential equation is an open problem for almost all high-dimensional
177

problems. This section solves it for a high-dimensional non-quadratic potential of a machine learn-
178

ing relevance.
179

4.1
Depth-0 Case
180

Let us first derive the stationary distribution of a one-dimensional linear regressor, which will be a
181

basis for comparison to help us understand what is unique about having a “depth” in deep learning.
182

The per-sample loss is ℓ(x,v) = (vx −y)2 + γv2. Defining
183

β1 ∶= E[x2],
β2 ∶= E[xy],
(10)

the global minimizer of the loss can be written as: v∗= β2/β1. The gradient variance is also not
184

trivial: C(v) ∶= Var[∇vℓ(v,x)] = 4(α1v2 −2α2v + α3). Note that the loss landscape L only
185

depends on β1 and β2, and the gradient noise only depends on α1, α2 and, α3. It is thus reasonable
186

to call β the landscape parameters and α the noise parameters. Both β and α appear in all stationary
187

distributions, implying that the stationary distributions of SGD are strongly data-dependent. Another
188

relevant quantity is ∆∶= minv C(v) ≥0, which is the minimal level of noise on the landscape. It
189

turns out that the stationary distribution is qualitatively different for ∆= 0 and for ∆> 0. For all the
190

examples in this work,
191

∆= Var[x2]Var[xy] −cov(x2,xy) = α1α3 −α2
2.
(11)

When is ∆zero? It happens when, for all samples of (x,y), xy + c = kx2 for some constant k and
192

c. We focus on the case ∆> 0 in the main text, which is most likely the case for practical situations.
193

The other cases are dealt with in Section A.
194

For ∆> 0, the stationary distribution for linear regression is (Section A)
195

p(v) ∝(α1v2 −2α2v + α3)−1−
β′
1
2T α1 exp[−1

T
α2β′
1 −α1β2
α1
√

∆
arctan(α1v −α2
√

∆
)],
(12)

in agreement with the previous result [24]. Two notable features exist for this distribution: (1)
196

the power exponent for the tail of the distribution depends on the learning rate and batch size, and
197

(2) the integral of p(v) converges for an arbitrary learning rate. On the one hand, this implies that
198

increasing the learning rate alone cannot introduce new phases of learning to a linear regression; on
199

the other hand, it implies that the expected error is divergent as one increases the learning rate (or
200

the feature variation), which happens at T = β′
1/α1. We will see that deeper models differ from the
201

single-layer model in these two crucial aspects.
202

4.2
An Analytical Model
203

Now, we consider the following model with a notion of depth and width; its loss function is
204

ℓ= [

d0
∑
i
(
D
∏
k=0
u(k)
i
)x −y]

2
,
(13)

5


---Page Break---
Figure 3: Stationary distributions of SGD for simple linear regression (D = 0), and a two-layer network
(D = 1) across different T = η/S: T = 0.05 (left) and T = 0.5 (Mid). We see that for D = 1, the stationary
distribution is strongly affected by the choice of the learning rate. In contrast, for D = 0, the stationary
distribution is also centered at the global minimizer of the loss function, and the choice of the learning rate only
affects the thickness of the tail. Right: the stationary distribution of a one-layer tanh-model, f(x) = tanh(vx)
(D = 0) and a two-layer tanh-model f(x) = w tanh(ux) (D = 1). For D = 1, we define v ∶= wu. The vertical
line shows the ground truth. The deeper model never learns the wrong sign of wu, whereas the shallow model
can learn the wrong one.

where D can be regarded as the depth and d0 the width. When the width d0 = 1, the law of balance is
205

sufficient to solve the model. When d0 > 1, we need to eliminate additional degrees of freedom. We
206

note that this model conceptually resembles (but not identical to) a diagonal linear network, which
207

has been found to well approximate the dynamics of real networks [27, 25, 2, 7].
208

We introduce vi ∶= ∏D
k=0 u(k)
i
, and so v = ∑i vi, where we call vi a “subnetwork” and v the “model.”
209

The following proposition shows that independent of d0 and D, the dynamics of this model can be
210

reduced to a one-dimensional form by invoking the law of balance.
211

Theorem 4.1. For all i ≠j, one (or more) of the following conditions holds for all trajectories at
212

stationarity: (1) vi = 0, or vj = 0, or L(θ) = 0; (2) sgn(vi) = sgn(vj). In addition, (2a) if D = 1,
213

for a constant c0, log ∣vi∣−log ∣vj∣= c0; (2b) if D > 1, ∣vi∣2 −∣vj∣2 = 0.
214

This theorem contains many interesting aspects. First of all, the three situations in item 1 directly
215

tell us the distribution of v if the initial state of of v is given by these conditions.2 This implies a
216

memory effect, namely, that the stationary distribution of SGD can depend on its initial state. The
217

second aspect is the case of item 2, which we will solve below. Item 2 of the theorem implies that all
218

the vi of the model must be of the same sign for any network with D ≥1. Namely, no subnetwork
219

of the original network can learn an incorrect sign. This is dramatically different from the case of
220

D = 0. We will discuss this point in more detail below. The third interesting aspect of the theorem is
221

that it implies that the dynamics of SGD is qualitatively different for different depths of the model.
222

In particular, D = 1 and D > 1 have entirely different dynamics. For D = 1, the ratio between
223

every pair of vi and vj is a conserved quantity. In sharp contrast, for D > 1, the distance between
224

different vi is no longer conserved but decays to zero. Therefore, a new balancing condition emerges
225

as we increase the depth. Conceptually, this qualitative distinction also corroborates the discovery
226

in Ref. [43], where D = 1 models are found to be qualitatively different from models with D > 1.
227

With this theorem, we are ready to solve the stationary distribution. It suffices to condition on the
228

event that vi does not converge to zero. Let us suppose that there are d nonzero vi that obey item
229

2 of Theorem 4.1 and d can be seen as an effective width of the model. We stress that the effective
230

width d ≤d0 depends on the initialization and can be arbitrary.3 Therefore, we condition on a fixed
231

value of d to solve for the stationary distribution of v (Appendix A):
232

Theorem 4.2. Let δ(x) denote the Dirac delta function. For an arbitrary factor z in[0,1], an
233

invariant solution of the Fokker-Planck Equation is p∗(v) = (1 −z)δ(v) + zp±(v), where
234

p±(∣v∣) ∝
1
∣v∣3(1−1/(D+1))g∓(v) exp (−1

T ∫

∣v∣

0
d∣v∣d1−2/(D+1)(β1∣v∣∓β2)

(D + 1)∣v∣2D/(D+1)g∓(v)) ,
(14)

where p−is the distribution on (−∞,0) and p+ is that on (0,∞), and g∓(v) = α1∣v∣2 ∓2α2∣v∣+ α3.
235

2L →0 is only possible when ∆= 0 and v = β2/β1.
3One can initialize the parameters such that d takes any value between 1 and d0. One way to achieve this
is to initialize on the stationary points specified by Theorem 4.1 at the desired d.

6


---Page Break---
The arbitrariness of the scalar z is due to the memory effect of SGD – if all parameters are initialized
236

at zero, they will remain there with probability 1. This means that the stationary distribution is not
237

unique. Since the result is symmetric in the sign of β2 = E[xy], we assume that E[xy] > 0 from
238

now on.
239

Also, we focus on the case γ = 0 in the main text.4 The distribution of v is
240

p±(∣v∣) ∝
∣v∣±β2/2α3T −3/2

(α1∣v∣2 ∓2α2∣v∣+ α3)1±β2/4T α3 exp(−1

2T
α3β1 −α2β2

α3
√

∆
arctan α1∣v∣∓α2
√

∆
).
(15)

This measure is worth a close examination. First, the exponential term is upper and lower bounded
241

and well-behaved in all situations. In contrast, the polynomial term becomes dominant both at
242

infinity and close to zero. When v < 0, the distribution is a delta function at zero: p(v) = δ(v). To
243

see this, note that the term v−β2/2α3T −3/2 integrates to give v−β2/2α3T −1/2 close to the origin, which
244

is infinite. Away from the origin, the integral is finite. This signals that the only possible stationary
245

distribution has a zero measure for v ≠0. The stationary distribution is thus a delta distribution,
246

meaning that if x and y are positively correlated, the learned subnets vi can never be negative,
247

independent of the initial configuration.
248

For v > 0, the distribution is nontrivial. Close to v = 0, the distribution is dominated by vβ2/2α3T −3/2,
249

which integrates to vβ2/2α3T −1/2. It is only finite below a critical Tc = β2/α3. This is a phase-
250

transition-like behavior. As T →(β2/α3)−, the integral diverges and tends to a delta distribution.
251

Namely, if T > Tc, we have ui = wi = 0 for all i with probability 1, and no learning can happen.
252

If T < Tc, the stationary distribution has a finite variance, and learning may happen. In the more
253

general setting, where weight decay is present, this critical T shifts to Tc = β2−γ

α3 . When T = 0,
254

the phase transition occurs at β2 = γ, in agreement with the threshold weight decay identified in
255

Ref. [45]. See Figure 3 for illustrations of the distribution across different values of T. We also
256

compare with the stationary distribution of a depth-0 model. Two characteristics of the two-layer
257

model appear rather striking: (1) the solution becomes a delta distribution at the sparse solution
258

u = w = 0 at a large learning rate; (2) the two-layer model never learns the incorrect sign (v is always
259

non-negative). Another exotic phenomenon implied by the result is what we call the “fluctuation
260

inversion.” Naively, the variance of model parameters should increase as we increase T, which is the
261

noise level in SGD. However, for the distribution we derived, the variance of v and u both decrease
262

to zero as we increase T: injecting noise makes the model fluctuation vanish. We discuss more about
263

this “fluctuation inversion” in the next section.
264

Also, while there is no other phase-transition behavior below Tc, there is still an interesting and
265

practically relevant crossover behavior in the distribution of the parameters as we change the learn-
266

ing rate. When training a model, The most likely parameter we obtain is given by the maximum
267

likelihood estimator of the distribution, ˆv ∶= arg maxp(v). Understanding how ˆv(T) changes as a
268

function of T is crucial. This quantity also exhibits nontrivial crossover behaviors at critical values
269

of T.
270

When T < Tc, a nonzero maximizer for p(v) must satisfy
271

v∗= −β1 −10α2T −
√

(β1 −10α2T)2 + 28α1T(β2 −3α3T)

14α1T
.
(16)

The existence of this solution is nontrivial, which we analyze in Appendix A.8. When T →0, a
272

solution always exists and is given by v = β2/β1, which does not depend on the learning rate or
273

noise C. Note that β2/β1 is also the minimum point of L(ui,wi). This means that SGD is only a
274

consistent estimator of the local minima in deep learning in the vanishing learning rate limit. How
275

biased is SGD at a finite learning rate? Two limits can be computed. For a small learning rate, the
276

leading order correction to the solution is v = β2

β1 + ( 10α2β2

β2
1
−7α1β2
2
β3
1
−3α3

β1 )T. This implies that the
277

common Bayesian analysis that relies on a Laplace expansion of the loss fluctuation around a local
278

minimum is improper. The fact that the stationary distribution of SGD is very far away from the
279

Bayesian posterior also implies that SGD is only a good Bayesian sampler at a small learning rate.
280

Example. It is instructive to consider an example of a structured dataset: y = kx + ϵ, where x ∼
281

N(0,1) and the noise ϵ obeys ϵ ∼N(0,σ2). We let γ = 0 for simplicity. If σ2 >
8
21k2, there always
282

4When weight decay is present, the stationary distribution is the same, except that one needs to replace β2
with β2 −γ. Other cases are also studied in detail in Appendix A and listed in Table. 1.

7


---Page Break---
exists a transitional learning rate: T ∗=
4k+
√

42σ
4(21σ2−8k2). Obviously, Tc/3 < T ∗. One can characterize the
283

learning of SGD by comparing T with Tc and T ∗. For this simple example, SGD can be classified
284

into roughly 5 different regimes. See Figure 4.
285

4.3
Power-Law Tail of Deeper Models
286

Figure 4: Regimes of learning for SGD
as a function of T and the noise in the
dataset σ.
According to (1) whether
the sparse transition has happened, (2)
whether a nontrivial maximum probabil-
ity estimator exists, and (3) whether the
sparse solution is a maximum probabil-
ity estimator, the learning of SGD can be
characterized into 5 regimes. Regime I is
where SGD converges to a sparse solution
with zero variance. In regime II, the sta-
tionary distribution has a finite spread, but
the probability of being close to the sparse
solution is very high. In regime III, the
probability density of the sparse solution
is zero, and therefore the model will learn
without much problem. In regime b, a lo-
cal nontrivial probability maximum exists.
The only maximum probability estimator
in regime a is the sparse solution.

An interesting aspect of the depth-1 model is that its distri-
287

bution is independent of the width d of the model. This is
288

not true for a deep model, as seen from Eq. (14). The d-
289

dependent term vanishes only if D = 1. Another intriguing
290

aspect of the depth-1 distribution is that its tail is indepen-
291

dent of any hyperparameter of the problem, dramatically
292

different from the linear regression case. This is true for
293

deeper models as well.
294

Since d only affects the non-polynomial part of the dis-
295

tribution, the stationary distribution scales as p(v) ∝
296

1
v3(1−1/(D+1))(α1v2−2α2v+α3). Hence, when v →∞, the scal-
297

ing behaviour is v−5+3/(D+1). The tail gets monotonically
298

thinner as one increases the depth. For D = 1, the expo-
299

nent is 7/2; an infinite-depth network has an exponent of 5.
300

Therefore, the tail of the model distribution only depends
301

on the depth and is independent of the data or details of
302

training, unlike the depth-0 model. In addition, due to the
303

scaling v5−3/(D+1) for v →∞, we can see that E[v2] will
304

never diverge no matter how large the T is.
305

An intriguing feature of this model is that the model with at
306

least one hidden layer will never have a divergent training
307

loss. This directly explains the puzzling observation of the
308

edge-of-stability phenomenon in deep learning: SGD train-
309

ing often gives a neural network a solution where a slight
310

increment of the learning rate will cause discrete-time in-
311

stability and divergence [40, 4]. These solutions, quite sur-
312

prisingly, exhibit low training and testing loss values even
313

when the learning rate is right at the critical learning rate of
314

instability. This observation contradicts naive theoretical expectations. Let ηsta denote the largest
315

stable learning rate. Close to a local minimum, one can expand the loss function up to the second or-
316

der to show that the value of the loss function L is proportional to Tr[Σ]. However, Σ ∝1/(ηsta−η)
317

should be a very large value [42, 19], and therefore L should diverge. Thus, the edge of stability
318

phenomenon is incompatible with the naive expectation up to the second order, as pointed out by
319

Ref. [5]. Our theory offers a direct explanation of why the divergence of loss does not happen: for
320

deeper models, the fluctuation of model parameters decreases as the gradient noise level increases,
321

reaching a minimal value before losing stability. Thus, SGD always has a finite loss because of the
322

power-law tail and fluctuation inversion. See Figure 5–mid.
323

Infinite-D limit.
As D tends to infinity, the distribution becomes
324

p(v) ∝
1
v3+k1(α1v2 −2α2v + α3)1−k1/2 exp ( −
d
DT ( β2

α3v + α2α3β1 −2α2
2β2 + α1α3β2
α2
3
√

∆
arctan(α1v −α2
√

∆
)) ),

where k1 = d(α3β1 −2α2β2)/(TDα2
3). An interesting feature is that the architecture ratio d/D
325

always appears simultaneously with 1/T. This implies that for a sufficiently deep neural network,
326

the ratio D/d also becomes proportional to the strength of the noise. Since we know that T = η/S
327

determines the performance of SGD, our result thus shows an extended scaling law of training:
328

d
D
S
η = const. The architecture aspect of the scaling law also agrees with an alternative analysis
329

[9, 10], where the optimal architecture is found to have a constant ratio of d/D. See Figure 5.
330

Now, if we T, there are three situations: (1) d = o(D), (2) d = c0D for a constant c0, (3) d = Ω(D).
331

If d = o(D), k1 →0 and the distribution converges to p(v) ∝v−3(α1v2 −2α2v + α3)−1, which is a
332

delta distribution at 0. Namely, if the width is far smaller than the depth, the model will collapse to
333

8


---Page Break---
Figure 5: SGD on deep networks leads to a well-controlled distribution and training loss. Left: Power law
of the tail of the parameter distribution of deep linear nets. The dashed lines show the upper (−7/2) and lower
(−5) bound of the exponent of the tail. The predicted power-law scaling agrees with the experiment, and the
exponent decreases as the theory predicts. Mid: training loss of a tanh network. D = 0 is the case where only
the input weight is trained, and D = 1 is the case where both input and output layers are trained. For D = 0,
the model norm increases as the model loses stability. For D = 1, a “fluctuation inversion” effect appears. The
fluctuation of the model vanishes before it loses stability. Right: performance of fully connected tanh nets on
MNIST. Scaling the learning rate as 1/D keeps the model performance relatively unchanged.

zero. Therefore, we should increase the model width as we increase the depth. In the second case,
334

d/D is a constant and can thus be absorbed into the definition of T and is the only limit where we
335

obtain a nontrivial distribution with a finite spread. If d = Ω(D), the distribution becomes a delta
336

distribution at the global minimum of the loss landscape, p(v) = δ(v −β2/β1) and achieves the
337

global minimum.
338

4.4
Implication for Variational Bayesian Learning
339

One of the major implications of the analytical solution we found for machine learning practice
340

is the inappropriateness of using SGD to approximate a Bayesian posterior. Because every SGD
341

iteration can be regarded as a sampling of the model parameters. A series of recent works have
342

argued that the stationary distribution can be used as an approximation of the Bayesian posterior
343

for fast variational inference [21, 3], pBayes(θ) ≈pSGD(θ), a method that has been used for a wide
344

variety of applications [12]. However, our result implies that such an approximation is likely to
345

fail. Common in Bayesian deep learning, we interpret the per-sample loss as the log probability
346

and the weight decay as a Gaussian prior over the parameters, the true model parameters have a log
347

probability of
348

log pBayes(θ∣x) ∝ℓ(θ,x) + γ∥θ∥2.
(17)

This distribution has a nonzero measure everywhere for any differentiable loss. However, the distri-
349

bution for SGD in Eq.(14) has a zero probability density almost everywhere because a 1d subspace
350

has a zero Lebesgue measure in a high-dimensional space. This implies that the KL divergence be-
351

tween the two distributions (either KL(pBayes∣∣pSGD) or KL(pSGD∣∣pBayes)) is infinite. Therefore,
352

we can infer that in the information-theoretic sense, pSGD cannot be used to approximate pBayes.
353

5
Discussion
354

In this work, we first showed that SGD systematically moves towards a balanced solution when
355

rescaling symmetry exists, a result we termed the law of balance. Applying the law of balance, we
356

have characterized the stationary distribution of SGD analytically, which is an unanswered funda-
357

mental problem in the study of SGD. This is the first analytical expression for a globally nonconvex
358

and beyond quadratic loss without the need for any approximation. With this solution, we have
359

discovered many phenomena that could be relevant to deep learning that were previously unknown.
360

We found that SGD only has probability of exploring a one-dimensional submanifold even for a
361

very-dimensional problem, ignoring all irrelevant directions. We applied our theory to the important
362

problem of variational inference and showed that it is, in general, not appropriate to approximate
363

the posterior with SGD, at least when any symmetry is present in the model. If one really wants
364

to use SGD for variational inference, special care is required to at least remove symmetries from
365

the loss function, which could be an interesting future problem. Our theory is limited, as the model
366

we solved is only a minimal model of reality, and it would be interesting to consider more realistic
367

models in the future. Also, it would be interesting to extend the law of balance to a broader class of
368

symmetries.
369

9


---Page Break---
References
370

[1] John C Baez and Brendan Fong. A noether theorem for markov processes. Journal of Mathe-
371

matical Physics, 54(1):013301, 2013.
372

[2] Rapha¨el Berthier. Incremental learning in diagonal linear networks. Journal of Machine Learn-
373

ing Research, 24(171):1–26, 2023.
374

[3] Pratik Chaudhari and Stefano Soatto. Stochastic gradient descent performs variational infer-
375

ence, converges to limit cycles for deep networks. In 2018 Information Theory and Applica-
376

tions Workshop (ITA), pages 1–10. IEEE, 2018.
377

[4] Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar.
Gra-
378

dient descent on neural networks typically occurs at the edge of stability.
arXiv preprint
379

arXiv:2103.00065, 2021.
380

[5] Alex Damian, Eshaan Nichani, and Jason D Lee. Self-stabilization: The implicit bias of gra-
381

dient descent at the edge of stability. arXiv preprint arXiv:2209.15594, 2022.
382

[6] Simon S Du, Wei Hu, and Jason D Lee. Algorithmic regularization in learning deep homoge-
383

neous models: Layers are automatically balanced. Advances in neural information processing
384

systems, 31, 2018.
385

[7] Mathieu Even, Scott Pesme, Suriya Gunasekar, and Nicolas Flammarion. (s) gd over diagonal
386

linear networks: Implicit regularisation, large stepsizes and edge of stability. arXiv preprint
387

arXiv:2302.08982, 2023.
388

[8] Xavier Fontaine, Valentin De Bortoli, and Alain Durmus. Convergence rates and approxi-
389

mation results for sgd and its continuous-time counterpart. In Mikhail Belkin and Samory
390

Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory, volume 134
391

of Proceedings of Machine Learning Research, pages 1965–2058. PMLR, 15–19 Aug 2021.
392

[9] Boris Hanin. Which neural net architectures give rise to exploding and vanishing gradients?
393

Advances in neural information processing systems, 31, 2018.
394

[10] Boris Hanin and David Rolnick. How to start training: The effect of initialization and archi-
395

tecture. Advances in Neural Information Processing Systems, 31, 2018.
396

[11] Wenqing Hu, Chris Junchi Li, Lei Li, and Jian-Guo Liu. On the diffusion approximation of
397

nonconvex stochastic gradient descent. arXiv preprint arXiv:1705.07562, 2017.
398

[12] Laurent Valentin Jospin, Hamid Laga, Farid Boussaid, Wray Buntine, and Mohammed Ben-
399

namoun. Hands-on bayesian neural networks—a tutorial for deep learning users. IEEE Com-
400

putational Intelligence Magazine, 17(2):29–48, 2022.
401

[13] Kenji Kawaguchi. Deep learning without poor local minima. Advances in Neural Information
402

Processing Systems, 29:586–594, 2016.
403

[14] Daniel Kunin, Javier Sagastuy-Brena, Surya Ganguli, Daniel LK Yamins, and Hidenori
404

Tanaka. Neural mechanics: Symmetry and broken conservation laws in deep learning dy-
405

namics. arXiv preprint arXiv:2012.04728, 2020.
406

[15] Jonas Latz. Analysis of stochastic gradient descent in continuous time. Statistics and Comput-
407

ing, 31(4):39, 2021.
408

[16] Qianxiao Li, Cheng Tai, and Weinan E.
Stochastic modified equations and dynamics of
409

stochastic gradient algorithms i: Mathematical foundations. Journal of Machine Learning
410

Research, 20(40):1–47, 2019.
411

[17] Zhiyuan Li, Kaifeng Lyu, and Sanjeev Arora. Reconciling modern deep learning with tra-
412

ditional optimization analyses: The intrinsic learning rate. Advances in Neural Information
413

Processing Systems, 33:14544–14555, 2020.
414

[18] Zhiyuan Li, Sadhika Malladi, and Sanjeev Arora. On the validity of modeling sgd with stochas-
415

tic differential equations (sdes), 2021.
416

10


---Page Break---
[19] Kangqiao Liu, Liu Ziyin, and Masahito Ueda. Noise and fluctuation of finite learning rate
417

stochastic gradient descent, 2021.
418

[20] Agnieszka B Malinowska and Moulay Rchid Sidi Ammi. Noether’s theorem for control prob-
419

lems on time scales. arXiv preprint arXiv:1406.0705, 2014.
420

[21] Stephan Mandt, Matthew D Hoffman, and David M Blei. Stochastic gradient descent as ap-
421

proximate bayesian inference. Journal of Machine Learning Research, 18:1–35, 2017.
422

[22] John C Mauro, Prabhat K Gupta, and Roger J Loucks. Continuously broken ergodicity. The
423

Journal of chemical physics, 126(18), 2007.
424

[23] Tetsuya Misawa. Noether’s theorem in symmetric stochastic calculus of variations. Journal of
425

mathematical physics, 29(10):2178–2180, 1988.
426

[24] Takashi Mori, Liu Ziyin, Kangqiao Liu, and Masahito Ueda. Power-law escape rate of sgd. In
427

International Conference on Machine Learning, pages 15959–15975. PMLR, 2022.
428

[25] Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, and Daniel Soudry. Implicit bias
429

of the step size in linear diagonal neural networks. In International Conference on Machine
430

Learning, pages 16270–16295. PMLR, 2022.
431

[26] Richard G Palmer. Broken ergodicity. Advances in Physics, 31(6):669–735, 1982.
432

[27] Scott Pesme, Loucas Pillaud-Vivien, and Nicolas Flammarion. Implicit bias of sgd for di-
433

agonal linear networks: a provable benefit of stochasticity. Advances in Neural Information
434

Processing Systems, 34:29218–29230, 2021.
435

[28] Hannes Risken and Hannes Risken. Fokker-planck equation. Springer, 1996.
436

[29] Tomasz Rolski, Hanspeter Schmidli, Volker Schmidt, and Jozef L Teugels. Stochastic pro-
437

cesses for insurance and finance. John Wiley & Sons, 2009.
438

[30] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear
439

dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.
440

[31] N. Shirish Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang. On Large-
441

Batch Training for Deep Learning: Generalization Gap and Sharp Minima. ArXiv e-prints,
442

September 2016.
443

[32] Justin Sirignano and Konstantinos Spiliopoulos. Stochastic gradient descent in continuous
444

time: A central limit theorem. Stochastic Systems, 10(2):124–151, 2020.
445

[33] Hidenori Tanaka and Daniel Kunin. Noether’s learning dynamics: Role of symmetry breaking
446

in neural networks, 2021.
447

[34] D Thirumalai and Raymond D Mountain. Activated dynamics, loss of ergodicity, and transport
448

in supercooled liquids. Physical Review E, 47(1):479, 1993.
449

[35] Christopher J Turner, Alexios A Michailidis, Dmitry A Abanin, Maksym Serbyn, and Zlatko
450

Papi´c. Weak ergodicity breaking from quantum many-body scars. Nature Physics, 14(7):745–
451

749, 2018.
452

[36] Nicolaas Godfried Van Kampen. Stochastic processes in physics and chemistry, volume 1.
453

Elsevier, 1992.
454

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
455

Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
456

processing systems, 30, 2017.
457

[38] Yuqing Wang, Minshuo Chen, Tuo Zhao, and Molei Tao. Large learning rate tames homo-
458

geneity: Convergence and balancing effect, 2022.
459

[39] Stephan Wojtowytsch. Stochastic gradient descent with noise of machine learning type part ii:
460

Continuous time analysis. Journal of Nonlinear Science, 34(1):1–45, 2024.
461

11


---Page Break---
[40] Lei Wu, Chao Ma, et al. How sgd selects the global minima in over-parameterized learning:
462

A dynamical stability perspective. Advances in Neural Information Processing Systems, 31,
463

2018.
464

[41] Zeke Xie, Issei Sato, and Masashi Sugiyama.
A diffusion theory for deep learning dy-
465

namics:
Stochastic gradient descent exponentially favors flat minima.
arXiv preprint
466

arXiv:2002.03495, 2020.
467

[42] Sho Yaida. Fluctuation-dissipation relations for stochastic gradient descent. arXiv preprint
468

arXiv:1810.00004, 2018.
469

[43] Liu Ziyin, Botao Li, and Xiangming Meng. Exact solutions of a deep linear network. In
470

Advances in Neural Information Processing Systems, 2022.
471

[44] Liu Ziyin, Kangqiao Liu, Takashi Mori, and Masahito Ueda. Strength of minibatch noise in
472

SGD. In International Conference on Learning Representations, 2022.
473

[45] Liu Ziyin and Masahito Ueda.
Exact phase transitions in deep learning.
arXiv preprint
474

arXiv:2205.12510, 2022.
475

12


---Page Break---
A
Theoretical Considerations
476

A.1
Background
477

A.1.1
Ito’s Lemma
478

Let us consider the following stochastic differential equation (SDE) for a Wiener process W(t):
479

dXt = µtdt + σtdW(t).
(18)

We are interested in the dynamics of a generic function of Xt. Let Yt = f(t,Xt); Ito’s lemma states
480

that the SDE for the new variable is
481

df(t,Xt) = (∂f

∂t + µt
∂f
∂Xt
+ σ2
t
2
∂2f
∂X2
t
)dt + σt
∂f
∂xdW(t).
(19)

Let us take the variable Yt = X2
t as an example. Then the SDE is
482

dYt = (2µtXt + σ2
t )dt + 2σtXtdW(t).
(20)

Let us consider another example. Let two variables Xt and Yt follow
483

dXt = µtdt + σtdW(t),
dYt = λtdt + ϕtdW(t).
(21)

The SDE of XtYt is given by
484

d(XtYt) = (µtYt + λtXt + σtϕt)dt + (σtYt + ϕtXt)dW(t).
(22)

A.1.2
Fokker Planck Equation
485

The general SDE of a 1d variable X is given by:
486

dX = −µ(X)dt + B(X)dW(t).
(23)

The time evolution of the probability density P(x,t) is given by the Fokker-Planck equation:
487

∂P(X,t)

∂t
= −∂

∂X J(X,t),
(24)

where J(X,t) = µ(X)P(X,t) + 1

2
∂
∂X [B2(X)P(X,t)]. The stationary distribution satisfying
488

∂P(X,t)/∂t = 0 is
489

P(X) ∝
1
B2(X) exp[−∫dX 2µ(X)

B2(X)] ∶= ˜P(X),
(25)

which gives a solution as a Boltzmann-type distribution if B is a constant. We will apply Eq. (25)
490

to determine the stationary distributions in the following sections.
491

A.2
Proof of Theorem 3.1
492

Proof. We omit writing v in the argument unless necessary.
By definition of the symmetry
493

ℓ(u,w,x) = ℓ(λu,w/λ,x), we obtain its infinitesimal transformation ℓ(u,w,x) = ℓ((1+ϵ)u,(1−
494

ϵ)w/λ,x). Expanding this to first order in ϵ, we obtain
495

∑
i
ui
∂ℓ
∂ui
= ∑
j
wj
∂ℓ
∂wj
.
(26)

The equations of motion are
496

dui

dt = −∂ℓ

∂ui
,
(27)

dwj

dt = −∂ℓ

∂wj
.
(28)

13


---Page Break---
Using Ito’s lemma, we can find the equations governing the evolutions of u2
i and w2
j:
497

du2
i
dt = 2ui
dui

dt + (dui)2

dt
= −2ui
∂ℓ
∂ui
+ TCu
i ,

dw2
j
dt = 2wj
dwj

dt + (dwj)2

dt
= −2wj
∂ℓ
∂wj
+ TCw
j ,
(29)

where Cu
i = Var[ ∂ℓ

∂ui ] and Cw
j = Var[ ∂ℓ

∂wj ]. With Eq. (26), we obtain
498

d
dt(∣∣u∣∣2 −∣∣w∣∣2) = −T(∑
j
Cw
j −∑
i
Cu
i ) = −T ⎛
⎝∑
j
Var[ ∂ℓ

∂wj
] −∑
i
Var[ ∂ℓ

∂ui
]⎞
⎠.
(30)

Due to the rescaling symmetry, the loss function can be considered as a function of the matrix uwT .
499

Here we define a new loss function as ˜ℓ(uiwj) = ℓ(ui,wj). Hence, we have
500

∂ℓ
∂wj
= ∑
i
ui
∂˜ℓ
∂(uiwj), ∂ℓ

∂ui
= ∑
j
wj
∂˜ℓ
∂(uiwj).
(31)

We can rewrite Eq. (30) into
501

d
dt(∣∣u∣∣2 −∣∣w∣∣2) = −T(uT C1u −wT C2w),,
(32)

where
502

(C1)ij = E[∑
k

∂˜ℓ
∂(uiwk)
∂˜ℓ
∂(ujwk)] −∑
k
E[
∂˜ℓ
∂(uiwk)]E[
∂˜ℓ
∂(ujwk)],

≡E[AT A] −E[AT ]E[A]
(33)

(C2)kl = E[∑
i

∂˜ℓ
∂(uiwk)
∂˜ℓ
∂(uiwl)] −∑
i
E[
∂˜ℓ
∂(uiwk)]E[
∂˜ℓ
∂(uiwl)]

≡E[AAT ] −E[A]E[AT ],
(34)

where
503

(A)ik ≡
∂˜ℓ
∂(uiwk).
(35)

The proof is thus complete.
504

A.3
Second-order Law of Balance
505

Considering the modified loss function:
506

ℓtot = ℓ+ 1

4T∣∣∇L∣∣2.
(36)

In this case, the Langevin equations become
507

dwj = −∂ℓ

∂wj
dt −1

4T ∂∣∣∇L∣∣2

∂wj
,
(37)

dui = −−∂ℓ

∂ui
dt −1

4T ∂∣∣∇L∣∣2

∂ui
.
(38)

Hence, the modified SDEs of u2
i and w2
j can be rewritten as
508

du2
i
dt = 2ui
dui

dt + (dui)2

dt
= −2ui
∂ℓ
∂ui
+ +TCu
i −1

2Tui∇ui∣∇L∣2,
(39)

dw2
j
dt = 2wj
dwj

dt + (dwj)2

dt
= −2wj
∂ℓ
∂wj
+ TCw
j −1

2Twj∇wj∣∇L∣2.
(40)

14


---Page Break---
In this section, we consider the effects brought by the last term in Eqs. (39) and (40). From the
509

infinitesimal transformation of the rescaling symmetry:
510

∑
j
wj
∂ℓ
∂wj
= ∑
i
ui
∂ℓ
∂ui
,
(41)

we take the derivative of both sides of the equation and obtain
511

∂L
∂ui
+ ∑
j
uj
∂2L
∂ui∂uj
= ∑
j
wj
∂2L
∂ui∂wj
,
(42)

∑
j
uj
∂2L
∂wi∂uj
= ∂L

∂wi
+ ∑
j
wj
∂2L
∂wi∂wj
,
(43)

where we take the expectation to ℓat the same time. By substituting these equations into Eqs. (39)
512

and (40), we obtain
513

d∣∣u∣∣2

dt
−d∣∣w∣∣∣2

dt
= T ∑
i
(Cu
i + (∇uiL)2) −T ∑
j
(Cw
j + (∇wjL)2).
(44)

Then following the procedure in Appendix. A.2, we can rewrite Eq. (44) as
514

d∣∣u∣∣2

dt
−d∣∣w∣∣2

dt
= −T(uT C1u + uT D1u −wT C2w −wT D2w)

= −T(uT E1u −wT E2w),
(45)

where
515

(D1)ij = ∑
k
E[
∂ℓ
∂(uiwk)]E[
∂ℓ
∂(ujwk)],
(46)

(D2)kl = ∑
i
E[
∂ℓ
∂(uiwk)]E[
∂ℓ
∂(uiwl)],
(47)

(E1)ij = E[∑
k

∂ℓ
∂(uiwk)
∂ℓ
∂(ujwk)],
(48)

(E2)kl = E[∑
i

∂ℓ
∂(uiwk)
∂ℓ
∂(uiwl)].
(49)

For one-dimensional parameters u,w, Eq. (45) is reduced to
516

d
dt(u2 −w2) = −E
⎡⎢⎢⎢⎢⎣
(
∂ℓ
∂(uw))

2⎤⎥⎥⎥⎥⎦
(u2 −w2).
(50)

Therefore, we can see this loss modification increases the speed of convergence. Now, we move
517

to the stationary distribution of the parameter v. At the stationarity, if ui = −wi, we also have the
518

distribution P(v) = δ(v) like before. However, when ui = wi, we have
519

dv

dt = −4v(β1v−β2)+4Tv(α1v2−2α2v+α3)−4β2
1Tv(β1v−β2)(3β1v−β2)+4v
√

T(α1v2 −2α2v + α3)dW

dt .
(51)
Hence, the stationary distribution becomes
520

P(v) ∝
vβ2/2α3T −3/2−β2
2/2α3

(α1v2 −2α2v + α3)1+β2/4T α3+K1 exp(−( 1

2T
α3β1 −α2β2

α3
√

∆
+ K2)arctan α1v −α2
√

∆
),

(52)
where
521

K1 = 3α3β2
1 −α1β2
2
4α1α3
,

K2 = 3α2α3β2
1 −4α1α3β1β2 + α1α2β2
2
2α1α3
√

∆
.
(53)

15


---Page Break---
From the expression above we can see K1 ≪1 + β2/4Tα3 and K2 ≪(α3β1 −α2β2)/2Tα3
√

∆.
522

Hence, the effect of modification can only be seen in the term proportional to v. The phase transition
523

point is modified as
524

Tc =
β2
α3 + β2
2
.
(54)

Compared with the previous result Tc = β2

α3 , we can see the effect of the loss modification is α3 →
525

α3 + β2
2, or equivalently, Var[xy] →E[x2y2]. This effect can be seen from E1 and E2.
526

A.4
Proof of Theorem 3.2
527

Proof. For any i, one can obtain the expressions of C(i)
1
and C(i)
2
from Theorem 3.1 as
528

(C(i)
1 )α1,α2 = 4piEi
⎡⎢⎢⎢⎣
∣∣˜x∣∣2(
d
∑
j=1
uα1
j vT
j ˜x −yα1)(
d
∑
j=1
uα2
j vT
j ˜x −yα2)
⎤⎥⎥⎥⎦
−4p2
i ∑
β
Ei
⎡⎢⎢⎢⎣
˜xβ(
d
∑
j=1
uα1
j vT
j ˜x −yα1)
⎤⎥⎥⎥⎦
Ei
⎡⎢⎢⎢⎣
˜xβ(
d
∑
j=1
uα2
j vT
j ˜x −

= 4piEi [∣∣˜x∣∣2rα1rα2] −4p2
i ∑
β
Ei [˜xβrα1]Ei [˜xβrα2],
(55)

(C(i)
2 )β1,β2 = 4Ei
⎡⎢⎢⎢⎣
˜xβ1 ˜xβ2∣∣
d
∑
j=1
ujvT
j ˜x −y∣∣2⎤⎥⎥⎥⎦
−4∑
α
Ei
⎡⎢⎢⎢⎣
˜xβ1(
d
∑
j=1
uα
j vT
j ˜x −yα)
⎤⎥⎥⎥⎦
Ei
⎡⎢⎢⎢⎣
˜xβ2(
d
∑
j=1
uα
j vT
j ˜x −yα)
⎤⎥⎥⎥⎦
= 4piEi [∣∣r∣∣2˜xβ1 ˜xβ2] −4p2
i ∑
α
Ei [˜xβ1rα]Ei [˜xβ2rα],
(56)

where we use the notation rα ∶= ∑d
j=1 uα
j vT
j ˜x −yα, ˜x ∶= (xT ,1)T ,vi = (wT
i ,bi)T and Ei[O] ∶=
529

E[O∣wT
i x + bi > 0].
530

We start with showing that C(1)
1
is full-rank. Let m be an arbitrary unit vector in Rdu. We have that
531

mT C(i)
1 m = 4piEi [∣∣˜x∣∣2(mT r)2] −4p2
i ∑
β
Ei [˜xβ(mT r)]Ei [˜xβ(mT r)]

≥4p2
i Ei [∣∣˜x∣∣2(mT r)2] −4p2
i ∑
β
Ei [˜xβ(mT r)]Ei [˜xβ(mT r)]

= 4p2
i ∑
β
Vari[˜xβmT r]

= 4p2
i ∑
β
[Vari[˜xβmT (g(x) −
d
∑
j=1
ujvT
j ˜x)] + Vari[˜xβmT ϵ] −2Covi[˜xβmT (g(x) −
d
∑
j=1
ujvT
j ˜x), ˜xβmT ϵ]]

≥4p2
i ∑
β
Vari[˜xβmT ϵ] > 0,
(57)

where the last inequality follows from
532

Cov[˜xβmT (g(x) −
d
∑
j=1
ujvT
j ˜x), ˜xβmT ϵ]

=Ei[(˜xβ)2mT (g(x) −
d
∑
j=1
ujvT
j ˜x)mT ϵ] −Ei[˜xβmT (g(x) −
d
∑
j=1
ujvT
j ˜x)]Ei[˜xβmT ϵ]

=0.
(58)

Here we denote that Vari[O] ∶= Ei[O2]−Ei[O]2 and Covi[O1,O2] ∶= Ei[O1O2]−Ei[O1]Ei[O2].
533

For C(i)
2 , we let the vector ˜n ∶= (nT ,nf)T be a unit vector in Rdw+1, yielding
534

˜nT C(i)
2 ˜n = 4piEi [∣∣r∣∣2(˜nT ˜x)2] −4p2
i ∑
α
Ei [rα(˜nT ˜x)]Ei [rα(˜nT ˜x)]

≥4p2
i Ei [∣∣r∣∣2(˜nT ˜x)2] −4p2
i ∑
α
Ei [rα(˜nT ˜x)]Ei [rα(˜nT ˜x)]

= 4p2
i ∑
α
Vari[rα˜nT ˜x].
(59)

16


---Page Break---
Note that this quantity can be decomposed as
535

∑
α
Vari[rα˜nT ˜x] = ∑
α
Vari[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x + ϵα)(˜nT ˜x)]

= ∑
α
Vari[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)(nT x + nf)] + ∑
α
Vari[ϵα(nT x + nf)]

−2∑
α
Covi[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)(nT x + nf),ϵα(nT x + nf)].
(60)

The covariance term vanishes because
536

Cov[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)(nT x + nf),ϵα(nT x + nf)]

=Ei[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)ϵα(nT x + nf)2] −Ei[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)(nT x + nf)]Ei[ϵα(nT x + nf)]

=0.
(61)

Therefore,
537

˜nT C(i)
2 ˜n ≥∑
α
Vari[(gα(x) −
d
∑
j=1
uα
j vT
j ˜x)(nT x + nf)] + ∑
α
Vari[ϵα(nT x + nf)]

≥∑
α
Vari[ϵα(nT x + nf)]

= ∑
α
Vari[ϵα]Vari[(nT x + nf)] + ∑
α
(Vari[ϵα]Ei[(nT x + nf)2] + Vari[nT x + nf]Ei[(ϵα)2])

≥∑
α
Vari[ϵα]Ei[(nT x + nf)2] > 0,
(62)

where the penultimate inequality follows from the fact that ϵ is independent of x. Hence, both the
538

matrices C(i)
1
and C(i)
2
are full-rank. The proof is completed.
539

A.5
Derivation of Eq. (4)
540

We here prove inequality (4). At stationarity, d(∥u∥2 −∥w∥2)/dt = 0, indicating
541

λ1M∥u∥2 −λ2m∥w∥2 ≥0, λ1m∥u∥2 −λ2M∥w∥2 ≤0.
(63)

The first inequality in Eq. (63) gives the solution
542

∥u∥2

∥w∥2 ≥λ2m

λ1M
.
(64)

The second inequality in Eq. (63) gives the solution
543

∥u∥2

∥w∥2 ≤λ2M

λ1m
.
(65)

Combining these two results, we obtain
544

λ2m
λ1M
≤∥u∥2

∥w∥2 ≤λ2M

λ1m
,
(66)

which is Eq. (4).
545

A.6
Proof of Theorem 4.1
546

Proof. This proof is based on the fact that if a certain condition is satisfied for all trajectories with
547

probability 1, this condition is satisfied by the stationary distribution of the dynamics with probabil-
548

ity 1.
549

17


---Page Break---
Let us first consider the case of D > 1. We first show that any trajectory satisfies at least one of
550

the following five conditions: for any i, (i) vi →0, (ii) L(θ) →0, or (iii) for any k ≠l, (u(k)
i
)2 −
551

(u(l)
i )2 →0.
552

The SDE for u(k)
i
is
553

du(k)
i
dt
= −2 vi

u(k)
i
(β1v −β2) + 2 vi

u(k)
i

√

η(α1v2 −2α2v + α3)dW

dt ,
(67)

where vi ∶= ∏D
k=1 u(k)
i
, and so v = ∑i vi. There exists rescaling symmetry between u(k)
i
and u(l)
i
for
554

k ≠l. By the law of balance, we have
555

d
dt[(u(k)
i
)2 −(u(l)
i )2] = −T[(u(k)
i
)2 −(u(l)
i )2]Var
⎡⎢⎢⎢⎢⎣

∂ℓ

∂(u(k)
i
u(l)
i )

⎤⎥⎥⎥⎥⎦
,
(68)

where
556

Var
⎡⎢⎢⎢⎢⎣

∂ℓ

∂(u(k)
i
u(l)
i )

⎤⎥⎥⎥⎥⎦
= (
vi
u(k)
i
u(l)
i
)2(α1v2 −2α2v + α3)
(69)

with vi/(u(k)
i
u(l)
i ) = ∏s≠k,l u(s)
i
.
In the long-time limit, (u(k)
i
)2 converges to (u(l)
i )2 unless
557

Var[
∂ℓ
∂(u(k)
i
u(l)
i
)] = 0, which is equivalent to vi/(u(k)
i
u(l)
i ) = 0 or α1v2 −2α2v + α3 = 0. These
558

two conditions correspond to conditions (i) and (ii). The latter is because α1v2 −2α2v+α3 = 0 takes
559

place if and only if v = α2/α1 and α2
2 −α1α3 = 0 together with L(θ) = 0. Therefore, at stationarity,
560

we must have conditions (i), (ii), or (iii).
561

Now, we prove that when (iii) holds, the condition 2-(b) in the theorem statement must hold: for
562

D = 1, (log ∣vi∣−log ∣vj∣) = c0 with sgn(vi) = sgn(vj). When (iii) holds, there are two situations.
563

First, if vi = 0, we have u(
ik) = 0 for all k, and vi will stay 0 for the rest of the trajectory, which
564

corresponds to condition (i).
565

If vi ≠0, we have u(k)
i
≠0 for all k. Therefore, the dynamics of vi is
566

dvi

dt = −2∑
k

⎛
⎝
vi
u(k)
i

⎞
⎠

2

(β1v−β2)+2∑
k

⎛
⎝
vi
u(k)
i

⎞
⎠

2 √

η(α1v2 −2α2v + α3)dW

dt +4∑
k,l

⎛
⎝
v3
i
(u(k)
i
u(l)
i )2
⎞
⎠η(α1v2−2α2v+α3).

(70)
Comparing the dynamics of vi and vj for i ≠j, we obtain
567

dvi/dt

∑k(vi/u(k)
i
)2 −
dvj/dt

∑k(vj/u(k)
j
)2 = 4⎛
⎝
∑m,l v3
i /(u(m)
i
u(l)
i )2

∑k(vi/u(k)
i
)2
−∑m,l v3
j /(u(m)
j
u(l)
j )2

∑k(vj/u(k)
j
)2
⎞
⎠η(α1v2 −2α2v + α3)

= 4⎛
⎝vi
∑m,l v2
i /(u(m)
i
u(l)
i )2

∑k(vi/u(k)
i
)2
−vj
∑m,l v2
j /(u(m)
j
u(l)
j )2

∑k(vj/u(k)
j
)2
⎞
⎠η(α1v2 −2α2v + α3).

(71)

By condition (iii), we have ∣u(0)
i
∣= ⋯= ∣u(D)
i
∣, i.e., (vi/u(k)
i
)2 = (v2
i )D/(D+1) and (vi/u(m)
i
u(l)
i )2 =
568

(v2
i )(D−1)/(D+1).5 Therefore, we obtain
569

dvi/dt
(D + 1)(v2
i )D/(D+1) −
dvj/dt
(D + 1)(v2
j )D/(D+1) = ⎛
⎝vi
D(v2
i )(D−1)/(D+1)

2(v2
i )D/(D+1)
−vj
D(v2
j )(D−1)/(D+1)

2(v2
j )D/(D+1)
⎞
⎠η(α1v2−2α2v+α3).

(72)
We first consider the case where vi and vj initially share the same sign (both positive or both nega-
570

tive). When D > 1, the left-hand side of Eq. (72) can be written as
571

1
1 −D
dv2/(D+1)−1
i

dt
+4Dv1−2/(D+1)
i
η(α1v2−2α2v+α3)−
1
1 −D

dv2/(D+1)−1
j

dt
−4Dv1−2/(D+1)
j
η(α1v2−2α2v+α3),
(73)

5Here, we only consider the root on the positive real axis.

18


---Page Break---
which follows from Ito’s lemma:
572

dv2/(D+1)−1
i

dt
= (
2
D + 1 −1)v2/(D+1)−2
i
dvi

dt + 2(
2
D + 1 −1)(
2
D + 1 −2)v2/(D+1)−3
i
⎛
⎝∑
k
( vi

u(k)
i
)2√

η(α1v2 −2α2v + α3)⎞
⎠

2

= (
2
D + 1 −1)v2/(D+1)−2
i
dvi

dt + 4D(D −1)v1−2/(D+1)
i
η(α1v2 −2α2v + α3).
(74)

Substitute in Eq. (72), we obtain Eq. (73).
573

Now, we consider the right-hand side of Eq. (72), which is given by
574

2Dv1−2/(D+1)
i
η(α1v2 −2α2v + α3) −2Dv1−2/(D+1)
j
η(α1v2 −2α2v + α3).
(75)

Combining Eq. (73) and Eq. (75), we obtain
575

1
1 −D
dv2/(D+1)−1
i

dt
−
1
1 −D

dv2/(D+1)−1
j

dt
= −2D(v1−2/(D+1)
i
−v1−2/(D+1)
j
)η(α1v2 −2α2v + α3).
(76)
By defining zi = v2/(D+1)−1
i
, we can further simplify the dynamics:
576

d(zi −zj)

dt
= 2D(D −1)( 1

zi
−1

zj
)η(α1v2 −2α2v + α3)

= −2D(D −1)zi −zj

zizj
η(α1v2 −2α2v + α3).
(77)

Hence,
577

zi(t) −zj(t) = exp[−∫dt2D(D −1)

zizj
η(α1v2 −2α2v + α3)].
(78)

Therefore, if vi and vj initially have the same sign, they will decay to the same value in the long-
578

time limit t →∞, which gives condition 2-(b). When vi and vj initially have different signs, we can
579

write Eq. (72) as
580

d∣vi∣/dt
(D + 1)(∣vi∣2)D/(D+1) +
d∣vj∣/dt
(D + 1)(∣vj∣2)D/(D+1) =(∣vi∣D(∣vi∣2)(D−1)/(D+1)

2(∣vi∣2)D/(D+1)
+ ∣vj∣D(∣vj∣2)(D−1)/(D+1)

2(∣vj∣2)D/(D+1)
)

× η(α1v2 −2α2v + α3).
(79)

Hence, when D > 1, we simplify the equation with a similar procedure as
581

1
1 −D
d∣vi∣2/(D+1)−1

dt
+
1
1 −D
d∣vj∣2/(D+1)−1

dt
= −2D(∣vi∣1−2/(D+1)+∣vj∣1−2/(D+1))η(α1v2−2α2v+α3).
(80)
Defining zi = ∣vi∣2/(D+1)−1, we obtain
582

d(zi + zj)

dt
= 2D(D −1)( 1

zi
+ 1

zj
)η(α1v2 −2α2v + α3)

= 2D(D −1)zi + zj

zizj
η(α1v2 −2α2v + α3),
(81)

which implies
583

zi(t) + zj(t) = exp[∫dt2D(D −1)

zizj
η(α1v2 −2α2v + α3)].
(82)

From this equation, we reach the conclusion that if vi and vj have different signs initially, one of
584

them converges to 0 in the long-time limit t →∞, corresponding to condition 1 in the theorem
585

statement. Hence, for D > 1, at least one of the conditions is always satisfied at t →∞.
586

Now, we prove the theorem for D = 1, which is similar to the proof above. The law of balance gives
587

d
dt[(u(1)
i
)2 −(u(2)
i
)2] = −T[(u(1)
i
)2 −(u(2)
i
)2]Var
⎡⎢⎢⎢⎢⎣

∂ℓ

∂(u(1)
i
u(2)
i
)

⎤⎥⎥⎥⎥⎦
.
(83)

19


---Page Break---
We can see that ∣u(1)
i
∣→∣u(2)
i
∣takes place unless Var[
∂ℓ
∂(u(1)
i
u(2)
i
)] = 0, which is equivalent to
588

L(θ) = 0. This corresponds to condition (ii). Hence, if condition (ii) is violated, we need to prove
589

condition (iii). In this sense, ∣u(1)
i
∣→∣u(2)
i
∣occurs and Eq. (72) can be rewritten as
590

dvi/dt

∣vi∣
−dvj/dt

∣vj∣
= (sign(vi) −sign(vj))η(α1v2 −2α2v + α3).
(84)

When vi and vj are both positive, we have
591

dvi/dt

vi
−dvj/dt

vj
= 0.
(85)

With Ito’s lemma, we have
592

dlog(vi)

dt
= dvi

vidt −2η(α1v2 −2α2v + α3).
(86)

Therefore, Eq. (85) can be simplified to
593

d(log(vi) −log(vj))

dt
= 0,
(87)

which indicates that all vi with the same sign will decay at the same rate. This differs from the case
594

of D > 2 where all vi decay to the same value. Similarly, we can prove the case where vi and vj are
595

both negative.
596

Now, we consider the case where vi is positive while vj is negative and rewrite Eq. (84) as
597

dvi/dt

vi
+ d(∣vj∣)/dt

∣vj∣
= 2η(α1v2 −2α2v + α3).
(88)

Furthermore, we can derive the dynamics of vj with Ito’s lemma:
598

dlog(∣vj∣)

dt
= dvi

vidt −2η(α1v2 −2α2v + α3).
(89)

Therefore, Eq. (88) takes the form of
599

d(log(vi) + log(∣vj∣))

dt
= −2η(α1v2 −2α2v + α3).
(90)

In the long-time limit, we can see log(vi∣vj∣) decays to −∞, indicating that either vi or vj will decay
600

to 0. This corresponds to condition 1 in the theorem statement. Combining Eq. (87) and Eq. (90),
601

we conclude that all vi have the same sign as t →∞, which indicates condition 2-(a) if conditions
602

in item 1 are all violated. The proof is thus complete.
603

A.7
Proof of Theorem 4.2
604

Proof. Following Eq. (70), we substitute u(k)
i
with v1/D
i
for arbitrary k and obtain
605

dvi

dt = −2(D + 1)∣vi∣2D/(D+1)(β1v −β2) + 2(D + 1)∣vi∣2D/(D+1)√

η(α1v2 −2α2v + α3)dW

dt
+ 2(D + 1)Dv3
i ∣vi∣−4/(D+1)η(α1v2 −2α2v + α3).
(91)

With Eq. (78), we can see that for arbitrary i and j, vi will converge to vj in the long-time limit. In
606

this case, we have v = dvi for each i. Then, the SDE for v can be written as
607

dv

dt = −2(D + 1)d2/(D+1)−1∣v∣2D/(D+1)(β1v −β2) + 2(D + 1)d2/(D+1)−1∣v∣2D/(D+1)√

η(α1v2 −2α2v + α3)dW

dt
+ 2(D + 1)Dd4/(D+1)−2v3∣v∣−4/(D+1)η(α1v2 −2α2v + α3).
(92)

If v > 0, Eq. (92) becomes
608

dv

dt = −2(D + 1)d2/(D+1)−1v2D/(D+1)(β1v −β2) + 2(D + 1)d2/(D+1)−1v2D/(D+1)√

η(α1v2 −2α2v + α3)dW

dt
+ 2(D + 1)Dd4/(D+1)−2v3−4/(D+1)η(α1v2 −2α2v + α3).
(93)

20


---Page Break---
Therefore, the stationary distribution of a general deep diagonal network is given by
609

p(v) ∝
1
v3(1−1/(D+1))(α1v2 −2α2v + α3) exp(−1

T ∫dv
d1−2/(D+1)(β1v −β2)
(D + 1)v2D/(D+1)(α1v2 −2α2v + α3)).

(94)

If v < 0, Eq. (92) becomes
610

d∣v∣

dt = −2(D + 1)d2/(D+1)−1∣v∣2D/(D+1)(β1∣v∣+ β2) −2(D + 1)d2/(D+1)−1∣v∣2D/(D+1)√

η(α1∣v∣2 + 2α2∣v∣+ α3)dW

dt
+ 2(D + 1)Dd4/(D+1)−2∣v∣3−4/(D+1)η(α1∣v∣2 + 2α2∣v∣+ α3).
(95)

The stationary distribution of ∣v∣is given by
611

p(∣v∣) ∝
1
∣v∣3(1−1/(D+1))(α1∣v∣2 + 2α2∣v∣+ α3) exp(−1

T ∫d∣v∣
d1−2/(D+1)(β1∣v∣+ β2)
(D + 1)∣v∣2D/(D+1)(α1∣v∣2 + 2α2∣v∣+ α3)).

(96)
Thus, we have obtained
612

p±(∣v∣) ∝
1
∣v∣3(1−1/(D+1))(α1∣v∣2 ∓2α2∣v∣+ α3) exp(−1

T ∫d∣v∣
d1−2/(D+1)(β1∣v∣∓β2)
(D + 1)∣v∣2D/(D+1)(α1∣v∣2 ∓2α2∣v∣+ α3)).

(97)
Especially when D = 1, the distribution function can be simplified as
613

p±(∣v∣) ∝
∣v∣±β2/2α3T −3/2

(α1∣v∣2 ∓2α2∣v∣+ α3)1±β2/4T α3 exp(−1

2T
α3β1 −α2β2

α3
√

∆
arctan α1∣v∣∓α2
√

∆
),
(98)

where we have used the integral
614

∫dv
β1v ∓β2
α1v2 −2α2v + α3
= α3β1 −α2β2

α3
√

∆
arctan α1∣v∣∓α2
√

∆
± β2

α3
log(v)± β2

2α3
log(α1v2−2α2v+α3).

(99)
Furthermore, we can also see that p(v) = δ(v) is also the stationary distribution of the Fokker-Planck
615

equation of Eq. (93). Hence, the general stationary distribution of v can be expressed as
616

p∗(v) = (1 −z)δ(v) + zp±(v).
(100)

The proof is complete.
617

A.8
Analysis of the maximum probability point
618

To investigate the existence of the maximum point given in Eq. (16), we treat T as a variable and
619

study whether (β1 −10α2T)2 +28α1T(β2 −3α3T) ∶= A in the square root is always positive or not.
620

When T <
β2
3α3 = Tc/3, A is positive for arbitrary data. When T >
β2
3α3 , we divide the discussion into
621

several cases. First, when α1α3 > 25

21α2
2, there always exists a root for the expression A. Hence, we
622

find that
623

T = −5α2β1 + 7α1β2 +
√

7
√

3α1α3β2
1 −10α1α2β1β2 + 7α2
1β2
2
2(21α1α3 −25α2
2)
∶= T ∗
(101)

is a critical point. When Tc/3 < T < T ∗, there exists a solution to the maximum condition. When
624

T > T ∗, there is no solution to the maximum condition.
625

The second case is α2
2 < α1α3 < 25

21α2
2. In this case, we need to further compare the value between
626

5α2β1 and 7α1β2. If 5α2β1 < 7α1β2, we have A > 0, which indicates that the maximum point
627

exists. If 5α2β1 > 7α1β2, we need to further check the value of minimum of A, which takes the
628

form of
629

minT A(T) = (25α2
2 −21α1α3)β2
1 −(7α1β2 −5α2β1)2

25α2
2 −21α1α3
.
(102)

If
7α1
5α2 <
β1
β2 <
5α2+
√

25α2
2−21α1α3
3α3
, the minimum of A is always positive and the maximum
630

exists.
However, if
β1
β2 ≥
5α2+
√

25α2
2−21α1α3
3α3
, there is always a critical learning rate T ∗.
If
631

21


---Page Break---
without weight decay
with weight decay

single layer
(α1v2 −2α2v + α3)−1−
β1
2T α1
α1(v −k)−2−(β1+γ)

T α1

non-interpolation
vβ2/2α3T −3/2

(α1v2−2α2v+α3)1+β2/4T α3
vS(β2−γ)/2α3λ−3/2

(α1v2−2α2v+α3)1+(β2−γ)/4T α3

interpolation y = kx
v−3/2+β1/2T α1k

(v−k)2+β1/2T α1k
v
−3/2+
1
2T α1k (β1−γ

k )

(v−k)
2+
1
2T α1k (β1−γ

k ) exp(−
βγ
2T α1
1
k(k−v))

Table 1: Summary of distributions p(v) in a depth-1 neural network. Here, we show the distribution
in the nontrivial subspace when the data x and y are positively correlated. The Θ(1) factors are
neglected for concision.

β1
β2 =
5α2+
√

25α2
2−21α1α3
3α3
, there is only one critical learning rate as Tc =
5α2β1−7α1β2
2(25α2
2−21α1α3). When
632

Tc/3 < T < T ∗, there is a solution to the maximum condition, while there is no solution when
633

T > T ∗. If β1

β2 >
5α2+
√

25α2
2−21α1α3
3α3
, there are two critical points:
634

T1,2 = −5α2β1 + 7α1β2 ∓
√

7
√

3α1α3β2
1 −10α1α2β1β2 + 7α2
1β2
2
2(21α1α3 −25α2
2)
.
(103)

For T < T1 and T > T2, there exists a solution to the maximum condition. For T1 < T < T2, there
635

is no solution to the maximum condition. The last case is α2
2 = α1α3 < 25

21α2
2. In this sense, the
636

expression of A is simplified as β2
1 + 28α1β2T −20α2β1T. Hence, when β1

β2 < 7α1

5α2 , there is no
637

critical learning rate and the maximum always exists. Nevertheless, when β1

β2 > 7α1

5α2 , there is always
638

a critical learning rate as T ∗=
β2
1
20α2β1−28α1β2 . When T < T ∗, there is a solution to the maximum
639

condition, while there is no solution when T > T ∗.
640

A.9
Other Cases for D = 1
641

The other cases are worth studying. For the interpolation case where the data is linear (y = kx for
642

some k), the stationary distribution is different and simpler. There exists a nontrivial fixed point for
643

∑i(u2
i −w2
i ): ∑j ujwj = α2

α1 , which is the global minimizer of L and also has a vanishing noise. It
644

is helpful to note the following relationships for the data distribution when it is linear:
645

⎧⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩

α1 = Var[x2],
α2 = kVar[x2] = kα1,
α3 = k2α1,
β1 = E[x2],
β2 = kE[x2] = kβ1.

(104)

Since the analysis of the Fokker-Planck equation is the same, we directly begin with the distribution
646

function in Eq. (15) for ui = −wi which is given by P(∣v∣) ∝δ(∣v∣). Namely, the only possible
647

weights are ui = wi = 0, the same as the non-interpolation case. This is because the corresponding
648

stationary distribution is
649

P(∣v∣) ∝
1
∣v∣2(∣v∣+ k)2 exp(−1

2T ∫d∣v∣β1(∣v∣+ k) + α1 1

T (∣v∣+ k)2

α1∣v∣(∣v∣+ k)2
)

∝∣v∣−3

2 −
β1
2T α1k (∣v∣+ k)−2+
β1
2T α1k .
(105)

The integral of Eq. (105) with respect to ∣v∣diverges at the origin due to the factor ∣v∣

3
2 +
β1
2T α1k .
650

For the case ui = wi, the stationary distribution is given from Eq. (15) as
651

P(v) ∝
1
v2(v −k)2 exp(−1

2T ∫dv β1(v −k) + α1T(v −k)2

α1v(v −k)2
)

∝v−3

2 +
β1
2T α1k (v −k)−2−
β1
2T α1k .
(106)

22


---Page Break---
Now, we consider the case of γ ≠0. In the non-interpolation regime, when ui = −wi, the stationary
652

distribution is still p(v) = δ(v). For the case of ui = wi, the stationary distribution is the same as
653

in Eq. (15) after replacing β with β′
2 = β2 −γ. It still has a phase transition. The weight decay
654

has the effect of shifting β2 by −γ. In the interpolation regime, the stationary distribution is still
655

p(v) = δ(v) when ui = −wi. However, when ui = wi, the phase transition still exists since the
656

stationary distribution is
657

p(v) ∝
v−3

2 +θ2

(v −k)2+θ2 exp(−β1γ

2Tα1

1
k(k −v)),
(107)

where θ2 =
1
2T α1k(β1 −γ

k). The phase transition point is θ2 = 1/2, which is the same as the non-
658

interpolation one.
659

The last situation is rather special, which happens when ∆= 0 but y ≠kx: y = kx −c/x for some
660

c ≠0. In this case, the parameters α and β are the same as those given in Eq. (104) except for β2:
661

β2 = kE[x2] −kc = kβ1 −kc.
(108)

The corresponding stationary distribution is
662

P(∣v∣) ∝
∣v∣−3

2 −ϕ2

(∣v∣+ k)2−ϕ2 exp(
c
2Tα1

1
k(k + ∣v∣)),
(109)

where ϕ2 =
1
2T α1k(β1 −c). Here, we see that the behavior of stationary distribution P(∣v∣) is
663

influenced by the sign of c. When c < 0, the integral of P(∣v∣) diverges due to the factor ∣v∣−3

2 −ϕ2 <
664

∣v∣−3/2 and Eq. (109) becomes δ(∣v∣) again. However, when c > 0, the integral of ∣v∣may not diverge.
665

The critical point is 3

2 + ϕ2 = 1 or equivalently: c = β1 + Tα1k. This is because when c < 0, the data
666

points are all distributed above the line y = kx. Hence, ui = −wi can only give a trivial solution.
667

However, if c > 0, there is the possibility to learn the negative slope k. When 0 < c < β1 + Tα1k,
668

the integral of P(∣v∣) still diverges and the distribution is equivalent to δ(∣v∣). Now, we consider the
669

case of ui = wi. The stationary distribution is
670

P(∣v∣) ∝
∣v∣−3

2 +ϕ2

(∣v∣−k)2+ϕ2 exp(−
c
2Tα1

1
k −∣v∣).
(110)

It also contains a critical point: −3

2 + ϕ2 = −1, or equivalently, c = β1 −α1kT. There are two cases.
671

When c < 0, the probability density only has support for ∣v∣> k since the gradient always pulls the
672

parameter ∣v∣to the region ∣v∣> k. Hence, the divergence at ∣v∣= 0 is of no effect. When c > 0,
673

the probability density has support on 0 < ∣v∣< k for the same reason. Therefore, if β1 > α1kT,
674

there exists a critical point c = β1 −α1kT. When c > β1 −α1kT, the distribution function P(∣v∣)
675

becomes δ(∣v∣). When c < β1−α1kT, the integral of the distribution function is finite for 0 < ∣v∣< k,
676

indicating the learning of the neural network. If β1 ≤α1kT, there will be no criticality and P(∣v∣)
677

is always equivalent to δ(∣v∣). The effect of having weight decay can be similarly analyzed, and
678

the result can be systematically obtained if we replace β1 with β1 + γ/k for the case ui = −wi or
679

replacing β1 with β1 −γ/k for the case ui = wi.
680

23


---Page Break---
NeurIPS Paper Checklist
681

1. Claims
682

Question: Do the main claims made in the abstract and introduction accurately reflect the
683

paper’s contributions and scope?
684

Answer: [Yes]
685

Justification: We believe that the abstract and introduction reflect the contributions and
686

scope of the paper.
687

Guidelines:
688

• The answer NA means that the abstract and introduction do not include the claims
689

made in the paper.
690

• The abstract and/or introduction should clearly state the claims made, including the
691

contributions made in the paper and important assumptions and limitations. A No or
692

NA answer to this question will not be perceived well by the reviewers.
693

• The claims made should match theoretical and experimental results, and reflect how
694

much the results can be expected to generalize to other settings.
695

• It is fine to include aspirational goals as motivation as long as it is clear that these
696

goals are not attained by the paper.
697

2. Limitations
698

Question: Does the paper discuss the limitations of the work performed by the authors?
699

Answer: [Yes]
700

Justification: We have discussed the limitations of our work in the Discussion session at
701

lines 369-371.
702

Guidelines:
703

• The answer NA means that the paper has no limitation while the answer No means
704

that the paper has limitations, but those are not discussed in the paper.
705

• The authors are encouraged to create a separate ”Limitations” section in their paper.
706

• The paper should point out any strong assumptions and how robust the results are to
707

violations of these assumptions (e.g., independence assumptions, noiseless settings,
708

model well-specification, asymptotic approximations only holding locally). The au-
709

thors should reflect on how these assumptions might be violated in practice and what
710

the implications would be.
711

• The authors should reflect on the scope of the claims made, e.g., if the approach was
712

only tested on a few datasets or with a few runs. In general, empirical results often
713

depend on implicit assumptions, which should be articulated.
714

• The authors should reflect on the factors that influence the performance of the ap-
715

proach. For example, a facial recognition algorithm may perform poorly when image
716

resolution is low or images are taken in low lighting. Or a speech-to-text system might
717

not be used reliably to provide closed captions for online lectures because it fails to
718

handle technical jargon.
719

• The authors should discuss the computational efficiency of the proposed algorithms
720

and how they scale with dataset size.
721

• If applicable, the authors should discuss possible limitations of their approach to ad-
722

dress problems of privacy and fairness.
723

• While the authors might fear that complete honesty about limitations might be used by
724

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
725

limitations that aren’t acknowledged in the paper. The authors should use their best
726

judgment and recognize that individual actions in favor of transparency play an impor-
727

tant role in developing norms that preserve the integrity of the community. Reviewers
728

will be specifically instructed to not penalize honesty concerning limitations.
729

3. Theory Assumptions and Proofs
730

Question: For each theoretical result, does the paper provide the full set of assumptions and
731

a complete (and correct) proof?
732

24


---Page Break---
Answer: [Yes]
733

Justification: We believe that the assumptions are clarified and complete proofs are pro-
734

vided for the theoretical parts.
735

Guidelines:
736

• The answer NA means that the paper does not include theoretical results.
737

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
738

referenced.
739

• All assumptions should be clearly stated or referenced in the statement of any theo-
740

rems.
741

• The proofs can either appear in the main paper or the supplemental material, but if
742

they appear in the supplemental material, the authors are encouraged to provide a
743

short proof sketch to provide intuition.
744

• Inversely, any informal proof provided in the core of the paper should be comple-
745

mented by formal proofs provided in appendix or supplemental material.
746

• Theorems and Lemmas that the proof relies upon should be properly referenced.
747

4. Experimental Result Reproducibility
748

Question: Does the paper fully disclose all the information needed to reproduce the main
749

experimental results of the paper to the extent that it affects the main claims and/or conclu-
750

sions of the paper (regardless of whether the code and data are provided or not)?
751

Answer: [Yes]
752

Justification: We believe that all of the experimental results are reproducable in our work.
753

Guidelines:
754

• The answer NA means that the paper does not include experiments.
755

• If the paper includes experiments, a No answer to this question will not be perceived
756

well by the reviewers: Making the paper reproducible is important, regardless of
757

whether the code and data are provided or not.
758

• If the contribution is a dataset and/or model, the authors should describe the steps
759

taken to make their results reproducible or verifiable.
760

• Depending on the contribution, reproducibility can be accomplished in various ways.
761

For example, if the contribution is a novel architecture, describing the architecture
762

fully might suffice, or if the contribution is a specific model and empirical evaluation,
763

it may be necessary to either make it possible for others to replicate the model with
764

the same dataset, or provide access to the model. In general. releasing code and data
765

is often one good way to accomplish this, but reproducibility can also be provided via
766

detailed instructions for how to replicate the results, access to a hosted model (e.g., in
767

the case of a large language model), releasing of a model checkpoint, or other means
768

that are appropriate to the research performed.
769

• While NeurIPS does not require releasing code, the conference does require all sub-
770

missions to provide some reasonable avenue for reproducibility, which may depend
771

on the nature of the contribution. For example
772

(a) If the contribution is primarily a new algorithm, the paper should make it clear
773

how to reproduce that algorithm.
774

(b) If the contribution is primarily a new model architecture, the paper should describe
775

the architecture clearly and fully.
776

(c) If the contribution is a new model (e.g., a large language model), then there should
777

either be a way to access this model for reproducing the results or a way to re-
778

produce the model (e.g., with an open-source dataset or instructions for how to
779

construct the dataset).
780

(d) We recognize that reproducibility may be tricky in some cases, in which case au-
781

thors are welcome to describe the particular way they provide for reproducibility.
782

In the case of closed-source models, it may be that access to the model is limited in
783

some way (e.g., to registered users), but it should be possible for other researchers
784

to have some path to reproducing or verifying the results.
785

5. Open access to data and code
786

25


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
787

tions to faithfully reproduce the main experimental results, as described in supplemental
788

material?
789

Answer: [No]
790

Justification: The code or data of the experiments are simple and easy to reproduce follow-
791

ing the description in the main text.
792

Guidelines:
793

• The answer NA means that paper does not include experiments requiring code.
794

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
795

public/guides/CodeSubmissionPolicy) for more details.
796

• While we encourage the release of code and data, we understand that this might not
797

be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
798

including code, unless this is central to the contribution (e.g., for a new open-source
799

benchmark).
800

• The instructions should contain the exact command and environment needed to run to
801

reproduce the results. See the NeurIPS code and data submission guidelines (https:
802

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
803

• The authors should provide instructions on data access and preparation, including how
804

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
805

• The authors should provide scripts to reproduce all experimental results for the new
806

proposed method and baselines. If only a subset of experiments are reproducible, they
807

should state which ones are omitted from the script and why.
808

• At submission time, to preserve anonymity, the authors should release anonymized
809

versions (if applicable).
810

• Providing as much information as possible in supplemental material (appended to the
811

paper) is recommended, but including URLs to data and code is permitted.
812

6. Experimental Setting/Details
813

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
814

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
815

results?
816

Answer: [Yes]
817

Justification: We have specified the training and test details in the captions of the experi-
818

ments in Figs. 2,4, and 5.
819

Guidelines:
820

• The answer NA means that the paper does not include experiments.
821

• The experimental setting should be presented in the core of the paper to a level of
822

detail that is necessary to appreciate the results and make sense of them.
823

• The full details can be provided either with the code, in appendix, or as supplemental
824

material.
825

7. Experiment Statistical Significance
826

Question: Does the paper report error bars suitably and correctly defined or other appropri-
827

ate information about the statistical significance of the experiments?
828

Answer: [No]
829

Justification: Here the dynamics is deterministic and there is no need to consider the error
830

bars here.
831

Guidelines:
832

• The answer NA means that the paper does not include experiments.
833

• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
834

dence intervals, or statistical significance tests, at least for the experiments that support
835

the main claims of the paper.
836

26


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
837

example, train/test split, initialization, random drawing of some parameter, or overall
838

run with given experimental conditions).
839

• The method for calculating the error bars should be explained (closed form formula,
840

call to a library function, bootstrap, etc.)
841

• The assumptions made should be given (e.g., Normally distributed errors).
842

• It should be clear whether the error bar is the standard deviation or the standard error
843

of the mean.
844

• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
845

ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
846

Normality of errors is not verified.
847

• For asymmetric distributions, the authors should be careful not to show in tables or
848

figures symmetric error bars that would yield results that are out of range (e.g. negative
849

error rates).
850

• If error bars are reported in tables or plots, The authors should explain in the text how
851

they were calculated and reference the corresponding figures or tables in the text.
852

8. Experiments Compute Resources
853

Question: For each experiment, does the paper provide sufficient information on the com-
854

puter resources (type of compute workers, memory, time of execution) needed to reproduce
855

the experiments?
856

Answer: [No]
857

Justification: The experiments can be simply conducted on personal computers.
858

Guidelines:
859

• The answer NA means that the paper does not include experiments.
860

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
861

or cloud provider, including relevant memory and storage.
862

• The paper should provide the amount of compute required for each of the individual
863

experimental runs as well as estimate the total compute.
864

• The paper should disclose whether the full research project required more compute
865

than the experiments reported in the paper (e.g., preliminary or failed experiments
866

that didn’t make it into the paper).
867

9. Code Of Ethics
868

Question: Does the research conducted in the paper conform, in every respect, with the
869

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
870

Answer: [Yes]
871

Justification: We have confirmed that the research is conducted with the NeurIPS Code of
872

Ethics.
873

Guidelines:
874

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
875

• If the authors answer No, they should explain the special circumstances that require a
876

deviation from the Code of Ethics.
877

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
878

eration due to laws or regulations in their jurisdiction).
879

10. Broader Impacts
880

Question: Does the paper discuss both potential positive societal impacts and negative
881

societal impacts of the work performed?
882

Answer: [NA]
883

Justification: Our work is a fundamental research on the dynamics of SGD and hence it
884

does not have direct positive or negative societal impacts.
885

Guidelines:
886

• The answer NA means that there is no societal impact of the work performed.
887

27


---Page Break---
• If the authors answer NA or No, they should explain why their work has no societal
888

impact or why the paper does not address societal impact.
889

• Examples of negative societal impacts include potential malicious or unintended uses
890

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
891

(e.g., deployment of technologies that could make decisions that unfairly impact spe-
892

cific groups), privacy considerations, and security considerations.
893

• The conference expects that many papers will be foundational research and not tied
894

to particular applications, let alone deployments. However, if there is a direct path to
895

any negative applications, the authors should point it out. For example, it is legitimate
896

to point out that an improvement in the quality of generative models could be used to
897

generate deepfakes for disinformation. On the other hand, it is not needed to point out
898

that a generic algorithm for optimizing neural networks could enable people to train
899

models that generate Deepfakes faster.
900

• The authors should consider possible harms that could arise when the technology is
901

being used as intended and functioning correctly, harms that could arise when the
902

technology is being used as intended but gives incorrect results, and harms following
903

from (intentional or unintentional) misuse of the technology.
904

• If there are negative societal impacts, the authors could also discuss possible mitiga-
905

tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
906

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
907

feedback over time, improving the efficiency and accessibility of ML).
908

11. Safeguards
909

Question: Does the paper describe safeguards that have been put in place for responsible
910

release of data or models that have a high risk for misuse (e.g., pretrained language models,
911

image generators, or scraped datasets)?
912

Answer: [No]
913

Justification: We believe there is no risks for misuse for the data and models.
914

Guidelines:
915

• The answer NA means that the paper poses no such risks.
916

• Released models that have a high risk for misuse or dual-use should be released with
917

necessary safeguards to allow for controlled use of the model, for example by re-
918

quiring that users adhere to usage guidelines or restrictions to access the model or
919

implementing safety filters.
920

• Datasets that have been scraped from the Internet could pose safety risks. The authors
921

should describe how they avoided releasing unsafe images.
922

• We recognize that providing effective safeguards is challenging, and many papers do
923

not require this, but we encourage authors to take this into account and make a best
924

faith effort.
925

12. Licenses for existing assets
926

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
927

the paper, properly credited and are the license and terms of use explicitly mentioned and
928

properly respected?
929

Answer:[NA]
930

Justification: [NA]
931

Guidelines:
932

• The answer NA means that the paper does not use existing assets.
933

• The authors should cite the original paper that produced the code package or dataset.
934

• The authors should state which version of the asset is used and, if possible, include a
935

URL.
936

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
937

• For scraped data from a particular source (e.g., website), the copyright and terms of
938

service of that source should be provided.
939

28


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the
940

package should be provided.
For popular datasets, paperswithcode.com/
941

datasets has curated licenses for some datasets. Their licensing guide can help
942

determine the license of a dataset.
943

• For existing datasets that are re-packaged, both the original license and the license of
944

the derived asset (if it has changed) should be provided.
945

• If this information is not available online, the authors are encouraged to reach out to
946

the asset’s creators.
947

13. New Assets
948

Question: Are new assets introduced in the paper well documented and is the documenta-
949

tion provided alongside the assets?
950

Answer: [No]
951

Justification: Nothing introduced.
952

Guidelines:
953

• The answer NA means that the paper does not release new assets.
954

• Researchers should communicate the details of the dataset/code/model as part of their
955

submissions via structured templates. This includes details about training, license,
956

limitations, etc.
957

• The paper should discuss whether and how consent was obtained from people whose
958

asset is used.
959

• At submission time, remember to anonymize your assets (if applicable). You can
960

either create an anonymized URL or include an anonymized zip file.
961

14. Crowdsourcing and Research with Human Subjects
962

Question: For crowdsourcing experiments and research with human subjects, does the pa-
963

per include the full text of instructions given to participants and screenshots, if applicable,
964

as well as details about compensation (if any)?
965

Answer: [NA]
966

Justification: We believe that neither the crowdsourcing nor the research with human sub-
967

jects is included in our work.
968

Guidelines:
969

• The answer NA means that the paper does not involve crowdsourcing nor research
970

with human subjects.
971

• Including this information in the supplemental material is fine, but if the main contri-
972

bution of the paper involves human subjects, then as much detail as possible should
973

be included in the main paper.
974

• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
975

tion, or other labor should be paid at least the minimum wage in the country of the
976

data collector.
977

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
978

Subjects
979

Question: Does the paper describe potential risks incurred by study participants, whether
980

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
981

approvals (or an equivalent approval/review based on the requirements of your country or
982

institution) were obtained?
983

Answer: [NA]
984

Justification: Our work does not contain crowdsourcing or research with human subjects.
985

Guidelines:
986

• The answer NA means that the paper does not involve crowdsourcing nor research
987

with human subjects.
988

• Depending on the country in which research is conducted, IRB approval (or equiva-
989

lent) may be required for any human subjects research. If you obtained IRB approval,
990

you should clearly state this in the paper.
991

29


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
992

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
993

guidelines for their institution.
994

• For initial submissions, do not include any information that would break anonymity
995

(if applicable), such as the institution conducting the review.
996

30


---Page Break---
