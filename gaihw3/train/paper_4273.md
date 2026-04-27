Causal Effect Estimation with Mixed Latent
Confounders and Post-treatment Variables

Anonymous Author(s)
Affiliation
Address
email

Abstract

Causal inference from observational data has attracted considerable attention among
1

researchers. One main obstacle is the handling of confounders. As direct mea-
2

surement of confounders may not be feasible, recent methods seek to address the
3

confounding bias via proxy variables, i.e., covariates postulated to be conducive to
4

the inference of latent confounders. However, the selected proxies may scramble
5

both confounders and post-treatment variables in practice, which risks biasing the
6

estimation by controlling for variables affected by the treatment. In this paper, we
7

systematically investigate the bias due to latent post-treatment variables, i.e., latent
8

post-treatment bias, in causal effect estimation. Specifically, we first derive the
9

bias when selected proxies scramble both confounders and post-treatment variables,
10

which we demonstrate can be arbitrarily bad. We then propose a novel Confounder-
11

identifiable VAE (CiVAE) to address the bias. Based on a mild assumption that the
12

prior of latent variables that generate the proxy belongs to a general exponential
13

family with at least one invertible sufficient statistic in the factorized part, CiVAE
14

individually identifies latent confounders and latent post-treatment variables up
15

to bijective transformations. We then prove that with individual identification,
16

the intractable disentanglement problem of latent confounders and post-treatment
17

variables can be transformed into a tractable independence test problem. Finally,
18

we prove that the true causal effects can be unbiasedly estimated with transformed
19

confounders inferred by CiVAE. Experiments on both simulated and real-world
20

datasets demonstrate significantly improved robustness of CiVAE.
21

1
Introduction
22

Causal inference, which aims to infer cause-and-effect relations from data, has gained increasing
23

prominence in various fields, such as social science, economics, and public health [10, 17, 34].
24

Traditional methods rely on the golden standard of randomized control trials (RCT) to draw valid
25

causal conclusions via experimentation [6]. Recently, more attention has been dedicated to causal
26

inference from observational data, where treatments, outcomes, and unit features are passively
27

observed, and researchers have no control over the treatment assignment mechanism [36, 37, 40].
28

One main obstacle to inferring valid causal relations from observational data is the confounding
29

bias, which occurs when we fail to account for the systematic difference between the treatment and
30

non-treatment group due to variables that causally influence the past treatments and the outcome, i.e.,
31

unobserved confounders [16]. If the confounders can be measured, a simple strategy to address the
32

bias is to control them via covariate adjustment [33] or propensity score re-weighting [24]. However,
33

confounders are not always measurable [23]. Therefore, recent methods seek to adjust for the
34

influence of unobserved confounders based on their proxies, which are easily acquirable covariates
35

postulated to be causally related with the unobserved confounders [29, 42, 28]. One exemplar work
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
(a) SCMAssumed by CEVAE
(c) SCMAssumed by the Proposed CiVAE

T
Y

C

X

T
Y

C1

X

CKC
M1
MK

(b) SCMAssumed by TEDVAE

T
Y

C

X

I
A

Instrumental

Variables
Adjusters

Confounders
Post-treatment

Variables

Confounders

M
predictive
for only T

predictive
for only Y

correlate with

both T and Y

correlate with

both T and Y

Figure 1: Comparison between the causal models assumed by CEVAE, TEDVAE, and CiVAE.

is the causal effect variational auto-encoder (CEVAE) [25], which has demonstrated that confounding
37

bias can be mitigated by controlling latent variables inferred from the proxies of confounders.
38

Although proxy-based methods have achieved substantial progress in recent years, they may risk
39

controlling latent post-treatment variables scrambled in the proxies, where latent post-treatment
40

bias can be introduced. Here, we note that the negative effects of controlling observed post-treatment
41

variables have been investigated in prior research [1, 9, 21]. For example, Montgomery et al. [30]
42

found that more than 50% of the papers published in top journals of politics inadvertently control
43

post-treatment variables in the experimental setting, even though researchers have complete control
44

over which covariates to control for. On this basis, we postulate that the post-treatment bias could
45

be even worse for proxy-based methods in the setting of observational study where variables are
46

passively recorded. In addition, the post-treatment variables can be latent and scrambled into the
47

observed covariates together with the latent confounders, which makes them difficult to disentangle.
48

Consider a real-world example from the Company1. We found that changing a job from onsite to
49

online mode causes applicants to make different decisions, and we want to estimate the causal effects
50

of switching a job from onsite to online mode to the decisions of the applicants (reflected by statistics
51

of applicants that apply for the job). In this case, the Company collected two groups of online (treated)
52

and onsite (control) jobs, where the statistics of the applicants (e.g., the average age) are calculated as
53

the surrogate outcome. Clearly, job seniority is a confounder, since less senior jobs are more likely to
54

permit online work, and applicants for these jobs tend to be younger. However, the seniority level of
55

a job can be difficult to measure. Therefore, the required skills of the job can be used as the proxy of
56

the confounder "seniority", as senior jobs tend to require more advanced skills. However, a caveat is
57

that switching to an online work mode may also alter the required skills of a job, thereby affecting the
58

qualification and, therefore, the decision of the applicants. Consequently, directly using the skills as
59

the proxy of the confounder "seniority" for adjustment could unintentionally control latent mediators
60

(changed skills), which introduces latent post-treatment bias in the causal effect estimation.
61

Addressing the latent post-treatment bias faces multi-faceted challenges. First, there lacks a
62

theoretical formulation of the bias when selected proxies scramble latent post-treatment variables
63

for existing proxy-based methods. In addition, it is difficult to distinguish confounders and post-
64

treatment variables in the latent space due to their similar observed behaviors. Existing covariate
65

disentanglement-based methods, e.g., TEDVAE [44], focus on an easier task of disentangling latent
66

confounders with latent adjusters and instrumental variables, which can be achieved by leveraging
67

their different predictive abilities w.r.t. the treatment and outcome. However, since both latent
68

confounders and post-treatment variables correlate with the treatment and the outcome, they cannot
69

be disentangled by these methods. Finally, even if latent confounders can be distinguished from post-
70

treatment variables, since most existing latent variable models have no identifiability guarantee [19],
71

it is unclear whether controlling the inferred latent variables, which may be arbitrary transformations
72

of the true confounders, can provide unbiased estimations of true causal effects.
73

To address the aforementioned challenges, we first analyze existing proxy-based methods when se-
74

lected proxies scramble both latent confounders and post-treatment variables and show the estimation
75

can be arbitrarily biased. We then propose a novel Confounder-identifiable VAE (CiVAE) to address
76

the latent post-treatment bias. Specifically, we prove that based on a mild assumption that the prior
77

of latent variables that generate the observed proxy (i.e., the latent confounders and post-treatment
78

variables) belong to a general exponential family with at least one invertible sufficient statistic in the
79

factorized part, latent confounders and latent post-treatment variables can be individually identified up
80

to simple bijective transformations. With such identifiability guarantee, based on the causal relations
81

among confounders, mediators, and treatment, we further demonstrate that the inferred confounders
82

1Anonymized due to double-blind review policy.

2


---Page Break---
(which are actually transformed proxies of the true confounders) could be properly distinguished
83

from the latent post-treatment variables with pair-wise conditional independence tests. Finally, we
84

prove that the true causal effects can be unbiasedly estimated based on transformed confounders
85

inferred by CiVAE. Experiments on both simulated and real-world datasets demonstrate that CiVAE
86

shows more robustness to latent post-treatment bias than existing methods.
87

2
Problem Formulation
88

In this paper, we assume the causal model in Fig. 1-(c). We use a binary random variable T to
89

denote the treatment, a random vector X ∈RKX to denote the observed covariates (i.e., the proxy),
90

and a random scalar Y ∈R to denote the outcome. Furthermore, the observed covariates X are
91

assumed to be generated from KC independent latent confounders C ≜[C1, C2..., CKC] causally
92

influencing both T and Y , and KM latent post-treatment variables M ≜[M1, M2..., MKM ] under
93

the causal influence of the treatment (where the relation between M and Y can be arbitrary). We use
94

the random vector Z ≜[C||M] ∈RKZ=KC+KM to denote all latent factors. Our aim is to estimate
95

the average causal effects of treatment T on outcome Y with auxiliary confounder information in X,
96

where the estimation should be devoid of both confounding bias and post-treatment bias.
97

3
Theoretical Analysis of Latent Post-Treatment Bias
98

3.1
Preliminaries and Assumptions
99

To achieve such a purpose, we first define the (conditional) average treatment effects (C/ATE) when
100

covariates X scramble both latent confounders C and post-treatment variables M. We then define
101

the post-treatment bias when covariates X are directly used as the proxy of confounders. To facilitate
102

the analysis, we make the following assumption regarding the causal generative process.
103

Assumption 1. (Noisy-Injectivity). We assume X = f(C, M) + ϵ, where f is a deterministic
104

function that combines latent confounders C and latent post-treatment variables M into observations
105

X, and ϵ is random noise. In addition, we assume that the function f is injective; beyond injectivity,
106

f can be arbitrarily nonlinear. We use f † : X →[C||M] to denote its left inverse. We use
107

f †
C : X →C and f †
M : X →M to denote the mapping from X to C, M, respectively.
108

Noisy-Injectivity is a common assumption made either explicitly or implicitly in most existing proxy-
109

of-confounder-based causal inference algorithms. For example, if both X and C are categorical,
110

[31] assumes that X has at least the same number of categories as C, whereas the effect restoration
111

algorithm [35] assumes that the matrix of p(C, X) to be full-rank. Although CEVAE [25] makes no
112

explicit injectivity assumption between C and X, it requires that the joint distribution p(C, X, T, Y )
113

can be fully recovered from the observations (X, T, Y ). [2] show that some of the possible identifica-
114

tion criteria for the recovery include 1) having multiple independent views of C in X [8], and 2) C
115

is categorical and X is a mixture of Gaussian components determined by C (that is, X is generated
116

by bijective mapping of C to the mean of the corresponding component with added Gaussian noise).
117

In the following part of this section, we omit the noise ϵ to gain better intuition of latent post-treatment
118

bias (but all the exact conclusions will still hold in the posterior sense [19]). In Section 4, we assume
119

noise exists and demonstrate that our method can still properly identify the latent confounders.
120

3.2
Causal Estimand and the True ATE
121

Based on Assumption 1, we are ready to define the estimand of average treatment effect (ATE)
122

through controlling the covariates X′, as well the as the true (conditional) average treatment effects.
123

Definition 1. (DCEV & DEV). We define the Difference in Conditional Expected Values (DCEV) as:
124

DCEV (x′) = E[Y |T = 1, X′ = x′] −E[Y |T = 0, X′ = x′],
(1)

which is the difference of the expected value of Y for units with variable X′ = x′ in the treatment
125

group and the non-treatment group. Based on DCEV (x′), we define the Difference in Expected
126

Value (DEV) as DEV (X′) = Ep(X′)[DCEV (X′)] as the expectation of DCEV w.r.t. p(X′).
127

3


---Page Break---
DEV (X′) denotes the estimand of ATE when X′ is the covariates that we choose to control (i.e.,
128

calculate the expected difference in each stratum of X′ = x′). If X′ = ∅, DEV (∅) represents
129

the naive estimator that directly calculates the expected difference of the outcome Y between the
130

treatment group and the non-treatment group. With the causal estimand DEV (X′) defined, we then
131

derive the true causal effects with the covariates X′ when it scrambles both latent confounders and
132

post-treatment variables according to the generative process described in Assumption 1:
133

Definition 2. Under Assumption 1, we define the Conditional Average Treatment Effect (CATE) for
134

individuals with observed covariates X = x by controlling only the confounder part in X as:
135

CATE(x) = E[Y |T = 1, C = f †
C(x)] −E[Y |T = 0, C = f †
C(x)],
(2)

with the Average Treatment Effect (ATE) of treatment T defined as:
136

ATE = E[Y |do(T = 1)] −E[Y |do(T = 0)] = Ep(C)[E[Y |T = 1, C] −E[Y |T = 0, C]].
(3)

Please note that we only consider the latent confounder component of the observed features X in the
137

definition of CATE in Eq. (2). This is because the causal relationship between the post-treatment
138

variables M and the outcome Y is indeterminate. However, if the specific relationship between M
139

and Y can be further established by the researcher (e.g., all elements of M are latent mediators),
140

more precise forms of CATE can be derived with path-specific counterfactual analysis [5, 14].
141

3.3
Latent Post-Treatment Bias
142

With DEV (X′) (the ATE estimator that control for the covariates X′), CATE, and ATE defined in
143

Section 3.2, in this section, we analyze the latent post-treatment bias of existing proxy-of-confounder-
144

based causal inference methods, such as CEVAE, that control for latent variables inferred from
145

the covariates X to estimate the ATE of T on Y , when X scrambles both latent confounders and
146

post-treatment variables as Assumption 1. In our analysis, Lemma 3.1 will be frequently used.
147

Lemma 3.1. For an injective function g, E[Y |X′ = x′] = E[Y |g(X′) = g(x′)] holds.
148

The proof when g is differentiable a.e. can be referred to in Appendix C.1. Since the latent variable
149

models used in existing methods (such as VAE with factorized Gaussian prior in CEVAE) lack
150

identifiability guarantee (i.e., the recovery of the exact latent variables), we assume that these models
151

can recover the true latent space Z = [C, M] up to invertible transformations ¯f, where the inference
152

process can be represented as ˆZ = ˜f(X) = ¯f ◦f †(X). With such an assumption, we have the
153

following theorem regarding the latent post-treatment bias when X mixes post-treatment variables.
154

Theorem 3.2. If the observed covariates X are generated from latent confounders C and latent
155

post-treatment variables M according to Assumption 1, the latent post-treatment bias of a proxy-
156

based causal inference algorithm that controls latent variables ˆZ inferred from X via ˜f = ¯f ◦f † :
157

RKX →RKC+KM to estimate the ATE can be formulated as follows:
158

Bias(X) = ATE −DEV ( ˜f(X)) = ATE −E[E[Y |T = 1, ˜f(X)] −E[Y |T = 0, ˜f(X)]]

= ATE −E[E[Y |1, ¯f ◦f †(f(C, M))] −E[Y |0, ¯f ◦f †(f(C, M))]]
= E[E[Y |1, C] −E[Y |0, C]] −E[E[Y |1, C, M] −E[Y |0, C, M]],

(4)

which can be arbitrarily bad. Therefore, the estimator of existing proxy-of-confounder-based meth-
159

ods, i.e., DEV ( ˜f(X)), is an arbitrarily biased estimator of the ATE, when the selected proxy of
160

confounders X accidentally mixes in latent post-treatment variables M.
161

The final step of Eq. (4) can be proved since f is injective and ¯f bijective, the composite ¯f ◦f † ◦f :
162

[C, M] →ˆZ is bijective, so we can use Lemma 3.1 to remove ¯f ◦f † ◦f in the condition.
163

3.4
Examples in the Linear Case
164

Generally, the latent post-treatment bias defined in Eq. (4) cannot be simplified, because (i) the
165

causal relationship between M and Y are indeterminate, and (ii) the causal influence of C, M,
166

and T on Y can be arbitrary. However, for linear structural causal models with determined causal
167

relationships between M and Y (e.g., M are mediators, which are post-treatment variables that have
168

causal influences on the outcomes), stronger conclusions can be drawn as follows:
169

4


---Page Break---
Corollary 3.3. (Mixed Latent Mediator). For the linear Structural Causal Model (SCM) defined as:
170

(i) T ←1(αT +
X
βi · Ci > a), (ii) Mj ←αM + γj · T

(iii) X ←αX + A[M||C], (iv) Y ←αY + τ · T +
X
θj · Mj +
X
κi · Ci,
(5)

where the mixture function f = A ∈RKX×(KC+KM) is a full column-rank matrix, the CATE, ATE,
171

and the bias of proxy-of-confounder-based causal inference model that controls the latent variables
172

ˆZ inferred via ˆZ = ˜f(X) = BT X can be formulated as follows:
173

ATE = CATE = τ +
X
γj · θj, and DEV ( ˆZ) = E[DCEV ( ˆZ)] = DCEV ( ˆZ) = τ

Bias( ˆZ) = ATE −DEV ( ˆZ) =
X
γj · θj,
(6)

where B ∈RKX×(KC+KM) is another full column-rank matrix. Since P γj · θj is arbitrary, the
174

estimator DEV ( ˆZ) = E[DCEV (BT X)] is arbitrarily biased for ATE estimation.
175

The proof of Eq. (6) is provided in Appendix C.2. In addition, we show that post-treatment variables
176

M DO NOT necessarily need to have direct causal effects on the outcome Y to incur arbitrary bias
177

in ATE estimation. In Appendix C.3, we provide another example (i.e., Mixed Latent Correlator) in
178

the linear case where M is correlated with Y through unobserved confounders U in Corollary C.1.
179

4
Methodology
180

In this section, we introduce the proposed Confounder-identifiable Variational Auto-Encoder (CiVAE)
181

in detail. Specifically, we first prove that if the prior distribution of the true latent variables Z =
182

[C, M] satisfies certain weak assumptions, CiVAE individually identify [C, M] up to bijective
183

transformations. Then, utilizing the causal relations between C, M, and T, we novelly transform the
184

challenging confounder-identifiability problem into a tractable pair-wise conditional independence
185

test problem, which can be effectively solved with kernel-based methods. The generalization of
186

CiVAE to address the interactions among [C, M] are discussed in Section D of the Appendix.
187

4.1
Generative Process
188

The fundamental work on the identifiability of deep variational inference, i.e., the identifiable VAE
189

(iVAE) [19], makes a strict assumption that the prior of true latent variables Z (i.e., [C, M] in
190

our case) is conditionally factorized given the available covariates. However, since both C and
191

M form fork structures with the outcome Y (see Fig. 1-(c)) [22], Ci, Cj, Mi, and Mj are not
192

independent given Y . Recently, Non-Factorized iVAE (NF-iVAE) [26] was proposed that allows
193

arbitrary dependence among the true latent variables Z in the conditional priors, where Z can be
194

identified up to arbitrary non-linear transformations. However, the transformation is not necessarily
195

invertible, which is risky as multiple values of the confounders may collapse, leading to bias when
196

estimating the ATE by averaging the DCEV calculated in each stratum of the inferred confounders.
197

In contrast to NF-iVAE, CiVAE guarantees the individual and bijective identifiability of Z by putting
198

a general exponential family with at least one invertible sufficient statistic in the factorized part as its
199

prior when conditioning on treatment T and outcome Y , which can be formulated as follows.
200

Assumption 2. Let Z = [C||M] be the random vector for latent variables that causally gen-
201

erate the observed covariates X according to Assumption 1. We assume that the conditional
202

prior of Z given the outcome Y and the treatment T belongs to a general exponential family
203

with parameter vector λ(Y, T) and sufficient statistics S(Z) = [Sf(Z)T , Snf(Z)T ]T . Specif-
204

ically, S(Z) is composed of (i) the sufficient statistics of a factorized exponential family, i.e.,
205

Sf(Z) = [S1(Z1)T , · · · , SKZ(ZKZ)T ]T , where all components Si(Zi) have dimension larger
206

than or equal to 2 and each Si has at least one invertible dimension, and (ii) Snf(Z), where Snf is
207

a neural network with ReLU activation. The density of the conditional prior can be formulated as:
208

pS,λ(Z|Y, T) = Q(Z)/C(Y, T) exp[S(Z)T λ(Y, T)],
(7)

where Q(Z) is the base measure, and C(Y, T) is the normalizing constant independent of Z.
209

5


---Page Break---
We justify that assumption 2 is weak and practical as follows. (i) Neural networks with ReLU
210

activation have universal approximation ability of distributions [27]. Therefore, Eq. (7) can model
211

arbitrary dependence between true latent confounders C and post-treatment variables M conditional
212

on T and Y . (ii) Although CiVAE makes an extra assumption that ∀i, at least one dimension of Si is
213

invertible, this can be easily satisfied as most commonly used exponential family distributions, such
214

as Gaussian, Bernoulli, etc., has at least one invertible sufficient statistics2.
215

The reason why we use ReLU as the activation is that, the identifiability of iVAE relies on the
216

condition that the sufficient statistics S have zero second-order cross-derivative. The factorized part,
217

i.e., Sf, satisfies it trivially as all cross-derivatives of Sf are zero. In addition, since the ReLU neural
218

networks are linear a.e., all second-order derivatives of Snf are zero. Therefore, identifiability holds
219

after adding Snf in the prior that allows the capturing of arbitrary dependence among Z.
220

4.2
Optimization Objective
221

Combining Assumptions 1 and 2, the generative process assumed by CiVAE can be formulated as:
222

(i) pθ(X, Z | Y, T) = pf(X | Z), (ii) pS,λ(Z | Y, T), (iii) pf(X | Z) = pϵ(X −f(Z)). (8)

where θ = (f, λ, S) ∈Θ are the parameters of the generative distribution. Since the generative
223

process of CiVAE is parameterized by deep neural networks, the posterior distribution of Z, i.e.,
224

pθ(Z | X, Y, T), is intractable. Therefore, we resort to variational inference [4], where we introduce
225

an approximate posterior qϕ(Z | X, Y, T) parameterized by a deep neural network with a trainable
226

parameter ϕ, and in qϕ(Z|·) finds the one closest to pθ(Z|·) measured by KL divergence. The
227

minimization of KL is equivalent to maximization of the evidence lower bound (ELBO):
228

L(θ, ϕ) := Eqϕ

log pf(X | Z) + log pS,λ(Z | Y, T) −log qϕ(Z | ·)
|
{z
}
KL of posterior with prior


.
(9)

Since the normalization constant C in Eq. (7) is generally intractable, it is infeasible to directly learn
229

S, λ by optimizing Eq. (9). Therefore, we substitute the KL term in Eq. (9) with the widely-used
230

score matching [13] to learn unnormalized densities instead as follows:
231

L(S, λ, ϕ) := Eqϕ(Z|·)
h
∥∇Z log qϕ(Z | ·) −∇Z log pS,λ(Z | Y, T)∥2i

= Eqϕ(Z|·)





KZ
X

j=1

"
∂2pS,λ(Z | Y, T)

∂Z2
j
+ 1

2

∂pS,λ(Z | Y, T)

∂Zj

2#

+ const,
(10)

4.3
Identifiability of CiVAE
232

With the generative process and optimization objective of CiVAE discussed in previous sub-sections,
233

we are ready to introduce the final assumption of CiVAE, which, combined with Assumptions 1 and
234

2, leads to the main Theorem of this paper, which states the identifiability of CiVAE.
235

Assumption 3. Assume the following: (i) The set {X ∈X|ϕ(X) = 0} has measure zero, where ϕ
236

is the characteristic function of the density pf in Eq. (8). (ii) The sufficient statistics, Si in Sf are all
237

twice differentiable. (iii) The mixture function f in Eq. (8) has all second-order cross derivatives.
238

(iv) There exist k + 1 distinct points (Y, T)0, · · · , (Y, T)k s.t. the matrix L = [λ((Y, T)1) −
239

λ((Y, T)0), · · · , λ((Y, T)k) −λ((Y, T)0)] of size k × k is invertible, where k = Dim(S).
240

Here, we note that Assumptions (i) - (iii) are trivial for differentiable neural networks. The Assumption
241

(iv) can be intuitively understood as independent samples of (Y, T) are required to identify C and
242

M. The identifiability theorem of CiVAE can be formulated as follows.
243

Theorem 4.1. If Assumptions 1, 2, and 3 hold, and if θ, ˜θ ∈Θ →pθ(X|Y, T) = p ˜θ(X|Y, T), the
244

true latent variables Z are identifiable up to permutation and element-wise bijective transformation.
245

Furthermore, in the case of variational inference, if we denote the true parameter that generates the
246

data as θ∗, if (i) the distribution family qϕ(Z|X, Y, T) contains the posterior pθ(Z|X, Y, T), and
247

qϕ(Z|X, Y, T) > 0, (ii) we optimize Eq. (4) w.r.t. both θ, ϕ, then in the limit of infinite data, true
248

parameters θ∗can be learned up to a permutation and bijective transformation of Z.
249

2There are a few exponential family dist. with no invertible sufficient statistics, e.g., Weibull with even shape
parameter k. However, these distributions are not commonly used in statistics or machine learning.

6


---Page Break---
The proof of Theorem 4.1 non trivially extends the NF-iVAE paper [26] by incorporating the new
250

assumption introduced in CiVAE (i.e., each Si has at least one invertible dimension) to ensure that the
251

transformation of each Zi is bijective. The detailed proof is provided in Appendix C.4 for reference.
252

4.4
Identification of Latent Confounders
253

Theorem 4.1 ensures that the latent variables ˆZ inferred by CiVAE cannot (i) mix confounders
254

and post-treatment variables in each dimension, or (ii) collapsing of different values of the latent
255

confounders into the same value. To further determine the dimensions of confounder and post-
256

treatment variable in ˆZ, we rely on the causal relations between latent variables ˆZ and the treatment
257

T and the associated marginal/conditional dependence properties, which are discussed as follows.
258

• Case 1. Intra-Confounders. Latent confounders Ci, Cj and the treatment T form the V structure
259

Ci →T ←Cj. Therefore, Ci and Cj are marginally independent, whereas they become
260

dependent when conditioning on the assigned treatment T.
261

• Case 2. Intra-Post Treatment Variables. Latent post-treatment variables Mi, Mj and the treatment
262

T form a Fork-structure Mi ←T →Mj, where Mi, Mj are marginally dependent, but they
263

become independent after conditioning on the assigned treatment T.
264

• Case 3. Cross-Confounder and Post-Treatment Variables. Latent confounder Ci, latent post-
265

treatment variable Mj, and the treatment T forms a Chain structure Ci →T →Mj, where Ci,
266

Mj are marginally dependent, and they become independent after conditioning on T.
267

From the above analysis we can find that, the dependence between two latent variables ˆZi and ˆZj
268

increases after conditioning on the treatment T ONLY in the case of intra-confounders. Therefore,
269

if more than one latent confounder exists, which is highly probable when covariates X are high-
270

dimensional, we can conduct independence test Ind( ˆZi, ˆZj) and CInd( ˆZi, ˆZj|T) for all pairs of
271

inferred latent variables, which can be implemented via kernel-based methods as [43], and select
272

the pairs where the p-value of CInd is larger than that of Ind as latent confounders. Here, we note
273

that the kernel-based (conditional) independence test incurs N 2 × K2
Z complexity in the training
274

phase. However, once the dimensions of the confounders in ˆZ are determined, CiVAE has the same
275

complexity as CEVAE for the estimation of CATE and ATE in the test phase.
276

4.5
ATE Estimator with Transformed Confounders
277

Finally, we demonstrate that controlling the transformed confounders ˆC inferred by CiVAE provides
278

an unbiased estimation of ATE. Specifically, we have the final Theorem show the unbiasedness.
279

Theorem 4.2. Controlling bijective of confounders is equivalent to original confounders in ATE
280

estimation, i.e., DEV ( ˜C) = DEV (g(C)) = ATE, if the transformation function g is bijective.
281

The proof of Theorem 4.2 for discrete C is trivial (where ˆC = g(C) represents a simple relabeling
282

of the stratum that we calculate the DCEV and take the expectation). The proof in the continuous
283

case where g is differentiable is provided in Appendix C.5. With Theorem 4.2, we can control the
284

identified latent confounders as true confounders, providing an unbiased estimate of ATE.
285

5
Empirical Study
286

In this section, we provide and analyze the experiments we conduct on both simulated and real-world
287

datasets, where a code demo written in PyTorch and Pyro is provided in this anonymous URL.
288

5.1
Datasets
289

Simulated Datasets.
We first establish two simulated datasets, i.e., LatentMediator and
290

LatentCorrelator, that consider two types of post-treatment variables, i.e., (i) mediators and
291

(ii) correlators, i.e., variables that are correlated with the outcome Y via latent confounders U, where
292

the causal generative process is under the full control of the experimenter. The generative process of
293

the two datasets can be referred to in Corollary 3.3 and Corollary C.1 in the Appendix, respectively.
294

In our experiments, C are generated from Gaussian distribution as C ∼Gaussian(0, IKC). For
295

7


---Page Break---
(a) Case 1: Intra-Confounder
(b) Case 2: Intra-Mediator
(c) Case 3: Confounder-Mediator
Figure 2: Visualization of p-value of independence test before and after conditioning on treatment T.

LatentMediator, γ is set as [−1, −1, −1], θ is set as [1, 1, 1], and τ is set as 2, which results in
296

ATE = −1. For the LatentCorrelator dataset, we set the same γ and θ as the LatentMediator
297

dataset, where parameters ϕ and τ are set to 1, which results in an overall ATE of 1.
298

Real-world Datasets. In addition, we build real-world datasets from the Company to estimate the
299

ATE of switching a job from onsite to online work mode to the statistics of the applicants. The
300

average age and the variance of gender of the applicants are two outcomes of interest. Covariates
301

X ∈{0, 1}KX include the required skills of the job. Specifically, we establish a cohort of 3,228
302

jobs from the Bay Area in the US, where a preliminary study shows that DEV (∅) ≈2 years3 (i.e.,
303

online job applicants are two years younger than onsite job applicants in the collected data), and
304

DEV (∅) ≈−0.015 (i.e., online jobs exhibit 0.015 more gender variance than onsite jobs in the
305

collected data). To simulate C and M, we first learn a generative model as follows:
306

Z ∼Gaussian(0, IKZ), X ∼Multi(NNf(Z)), Y ∼Gaussian(w ⊙Z, 1),
(11)

where Multi represents multinomial distribution, NNf is a neural network with softmax activation,
307

Z, w ∈RKZ, KZ = 8, and ⊙represents the element-wise product operator, respectively. We
308

then treat the first KC = 5 dimensions of Z as the latent confounders C and the remaining
309

KM = KZ −KC dimensions as the latent mediators M. After learning NNf and w according to
310

Eq. (11), we draw latent confounders C ∈Gaussian(0, I), latent mediators M = T · γ, and set the
311

outcome Y = w ⊙[C||M] + τ · T, where the true ATE can be calculated as sum(γ ⊙w−KM:) + τ.
312

5.1.1
Disentangle Confounders and Post-treatment Variables
313

We first show the p-value of the kernel-based pairwise independence test of the true latent variables
314

before and after conditioning on the assigned treatment T. From Fig. 2, we can find that the distinction
315

of the intra-confounder case from the other two cases discussed in Subsection 4.4 is significant. Here,
316

we should note this relies on the assumption that latent confounders are independent. If the latent
317

confounders are correlated, we can first use causal discovery techniques such as the PC algorithm [39]
318

to find direct parents of T, and use our algorithm as the refinement to determine the true confounders
319

C from the misidentified post-treatment variables (Experiments see Section D) in Appendix.
320

5.2
Baselines
321

The baselines we include for comparisons can be categorized into three classes. (i) Unawareness,
322

where no information in X is used for ATE estimation. We implement the naive LR0 estimator, which
323

regresses Y on T and uses the coefficient to estimate the ATE [15] (LR0 equals to DEV (∅), i.e., the
324

difference of the average outcome between the treatment and non-treatment group). (ii) Control-X,
325

which directly controls the covariates X. In this class, LR1 regresses Y on T and X, whereas TarNet
326

uses a two-branch neural network to estimate the DEV (X) (iii) Control-Z, which controls latent
327

variables Z learned from the covariates X. Methods from this class include the CEVAE [25] and
328

covariate disentanglement methods, such as DR-CFR [12], TEDVAE [44], NICE [38], and AFS [41].
329

5.2.1
Results and Analysis
330

From Table 1, we can find that for all four datasets, CEVAE is worse than the naive LR0 estimator.
331

In addition, for the LatentMediator and Company (Age) dataset, all methods except CiVAE fail
332

to predict the negativity of the ATE. Covariates disentanglement-based methods, i.e., DR-CFR
333

and TEDVAE, inherit the latent post-treatment bias of CEVAE. The reason is that, these methods
334

disentangle latent confounders C from latent instrumental variables I and latent adjusters A by
335

3which leads to 0.178 and -0.105 after standardization of the outcome.

8


---Page Break---
Table 1: Comparison of CiVAE with baselines under latent post-treatment bias on various datasets.

Dataset
LatentMediator
LatentCorrelator
Company (Age)
Company (Gender)
Method
ATE.
Err.
ATE.
Err.
ATE.
Err.
ATE.
Err.

LR0
0.975 ± 0.032
1.975
2.977 ± 0.032
1.977
0.131 ± 0.015
0.399
-0.105 ± 0.009
-0.213
LR1
1.457 ± 0.167
2.457
3.400 ± 0.130
2.400
0.093 ± 0.029
0.361
-0.175 ± 0.014
-0.256
TarNet
1.461 ± 0.172
2.461
3.414 ± 0.146
2.414
0.112 ± 0.085
0.380
-0.167 ± 0.021
-0.248
CEVAE
1.550 ± 0.292
2.550
3.323 ± 0.167
2.323
0.106 ± 0.078
0.374
-0.180 ± 0.028
-0.261
DR-CFR
1.239 ± 0.324
2.239
3.185 ± 0.319
2.185
0.094 ± 0.089
0.362
-0.159 ± 0.030
-0.240
NICE
1.868 ± 0.530
2.868
1.942 ± 0.524
0.942
0.149 ± 0.126
0.417
-0.186 ± 0.041
-0.267
TEDVAE
1.042 ± 0.315
2.042
3.138 ± 0.281
2.138
0.097 ± 0.093
0.365
-0.143 ± 0.027
-0.224
AFS
1.496 ± 0.825
2.496
3.251 ± 0.398
2.251
0.105 ± 0.102
0.373
-0.163 ± 0.045
-0.244
CiVAE
-0.822 ± 0.753
0.178
1.199 ± 0.765
0.199
-0.140 ±0.137
0.128
-0.106 ± 0.064
-0.187
True ATE
-1.000 ± 0.000
0.000
1.000 ± 0.000
0.000
-0.268 ± 0.000
0.000
-0.081 ± 0.000
0.000

utilizing their causal relations with T and Y , i.e., I is predictive only for T, A is predictive only
336

for Y , whereas C is predictive for both T and Y . For example, TEDVAE includes three encoders
337

to infer three sets of latent variables ˆI, ˆ
A, ˆC from X and adds classification losses p(T| ˆI, ˆC)
338

and p(Y |T, ˆC, ˆ
A) on the CEVAE loss. However, since both latent confounders C and latent post-
339

treatment variables M are correlated with both T and Y , these methods cannot disentangle C from
340

M. An exception is NICE [38], which uses invariant risk minimization (IRM) [3] to find all causal
341

parents of the outcome Y as the confounders, which makes it more robust in the LatentCorrelator
342

case. However, since mediators M are also the causal parent of Y , the performance degrades
343

substantially on the LatentMediator dataset. Although AFS [41] considers the existence of post-
344

treatment variables M in the proxy X, it assumes that they can be separated from other variables in
345

X in the observational space, and no relationship exists between the post-treatment variables and the
346

outcome, so it still has poor performance in our setting since both assumptions are violated.
347

5.3
Sensitivity Analysis
348

2:6
3:5
4:4
5:3
6:2
Ratio

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

Error

(a) Company (Age)

CiVAE
TEDVAE

2:6
3:5
4:4
5:3
6:2
Ratio

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Error

(b) Company (Gender)

CiVAE
TEDVAE

Figure 3: Error with different ratio of latent confounders and
latent post-treatment variable in the latent space.

In this part, we vary the number
349

of confounders and post-treatment
350

variables that generate proxy X in
351

the Company (Age) and Company
352

(Gender) datasets and compare
353

CiVAE with the baseline TEDVAE
354

in Fig. 3. Fig. 3 shows that the er-
355

ror is consistently lower for CiVAE.
356

In addition, the error is compara-
357

tively higher when the number of con-
358

founders is low since the misidenti-
359

fication of latent post-treatment vari-
360

ables as confounders can have a com-
361

paratively larger influence on the ATE estimation. In addition, when the number of confounders
362

becomes larger, the performance gap between CiVAE and TEDVAE gracefully shrinks.
363

6
Conclusions
364

In this paper, we systematically investigate the latent post-treatment bias in causal inference from
365

observational data. We first prove that unresolved latent post-treatment variables scrambled in the
366

proxy of confounders can arbitrarily bias the ATE estimation. To address the bias, we proposed
367

the Confounder-identifiable VAE (CiVAE), which, utilizing a mild assumption regarding the prior
368

of latent factors, guarantees the identifiability of latent confounders up to bijective transformations.
369

Finally, we show that controlling the latent confounders inferred by CiVAE can provide an unbiased
370

estimation of the ATE. Experiments on both simulated and real-world datasets demonstrate that
371

CiVAE has superior robustness to latent post-treatment bias compared to state-of-the-art methods.
372

9


---Page Break---
References
373

[1] A. Acharya, M. Blackwell, and M. Sen. Explaining causal findings without bias: Detecting and
374

assessing direct effects. American Political Science Review, 110(3):512–529, 2016.
375

[2] A. Anandkumar, R. Ge, D. Hsu, S. M. Kakade, and M. Telgarsky. Tensor decompositions for
376

learning latent variable models. Journal of Machine Learning Research, 15:2773–2832, 2014.
377

[3] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv
378

preprint arXiv:1907.02893, 2019.
379

[4] D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: A review for statisticians.
380

Journal of the American Statistical Association, 112(518):859–877, 2017.
381

[5] L. Cheng, R. Guo, and H. Liu. Causal mediation analysis with hidden confounders. In
382

Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining,
383

pages 113–122, 2022.
384

[6] T. D. Cook, D. T. Campbell, and W. Shadish. Experimental and quasi-experimental designs for
385

generalized causal inference. Houghton Mifflin Boston, MA, 2002.
386

[7] P. Ding and L. W. Miratrix. To adjust or not to adjust? sensitivity analysis of m-bias and
387

butterfly-bias. Journal of Causal Inference, 3(1):41–57, 2015.
388

[8] J. K. Edwards, S. R. Cole, and D. Westreich. All your data are always missing: incorporating
389

bias due to measurement error into the potential outcomes framework. International Journal of
390

Epidemiology, 44(4):1452–1459, 2015.
391

[9] F. Elwert and C. Winship. Endogenous selection bias: The problem of conditioning on a collider
392

variable. Annual review of sociology, 40:31–53, 2014.
393

[10] T. A. Glass, S. N. Goodman, M. A. Hernán, and J. M. Samet. Causal inference in public health.
394

Annual Review of Public Health, 34:61–75, 2013.
395

[11] N. Hassanpour and R. Greiner. Counterfactual regression with importance sampling weights.
396

In IJCAI, pages 5880–5887, 2019.
397

[12] N. Hassanpour and R. Greiner.
Learning disentangled representations for counterfactual
398

regression. In International Conference on Learning Representations, 2020.
399

[13] A. Hyvärinen and P. Dayan. Estimation of non-normalized statistical models by score matching.
400

Journal of Machine Learning Research, 6(4), 2005.
401

[14] K. Imai, L. Keele, and D. Tingley. A general approach to causal mediation analysis. Psycholog-
402

ical Methods, 15(4):309, 2010.
403

[15] G. W. Imbens and D. B. Rubin. Causal inference in statistics, social, and biomedical sciences.
404

Cambridge University Press, 2015.
405

[16] K. Jager, C. Zoccali, A. Macleod, and F. Dekker. Confounding: what it is and how to deal with
406

it. Kidney international, 73(3):256–260, 2008.
407

[17] F. Johansson, U. Shalit, and D. Sontag. Learning representations for counterfactual inference.
408

In International Conference on Machine Learning, pages 3020–3029, 2016.
409

[18] M. Kalisch and P. Bühlman. Estimating high-dimensional directed acyclic graphs with the
410

pc-algorithm. Journal of Machine Learning Research, 8(3), 2007.
411

[19] I. Khemakhem, D. Kingma, R. Monti, and A. Hyvarinen. Variational autoencoders and nonlinear
412

ICA: A unifying framework. In International Conference on Artificial Intelligence and Statistics,
413

pages 2207–2217. PMLR, 2020.
414

[20] G. King. A hard unsolved problem? post-treatment bias in big social science questions. In
415

Hard Problems in Social Science” Symposium, April, volume 10, 2010.
416

10


---Page Break---
[21] G. King and L. Zeng. The dangers of extreme counterfactuals. Political Analysis, 14(2):131–159,
417

2006.
418

[22] D. Koller and N. Friedman. Probabilistic graphical models: principles and techniques. MIT
419

press, 2009.
420

[23] M. Kuroki and J. Pearl. Measurement bias and effect restoration in causal inference. Biometrika,
421

101(2):423–437, 2014.
422

[24] F. Li, K. L. Morgan, and A. M. Zaslavsky. Balancing covariates via propensity score weighting.
423

Journal of the American Statistical Association, 113(521):390–400, 2018.
424

[25] C. Louizos, U. Shalit, J. M. Mooij, D. Sontag, R. Zemel, and M. Welling. Causal effect inference
425

with deep latent-variable models. Advances in Neural Information Processing Systems, 30,
426

2017.
427

[26] C. Lu, Y. Wu, J. M. Hernández-Lobato, and B. Schölkopf. Invariant causal representation
428

learning for out-of-distribution generalization. In International Conference on Learning Repre-
429

sentations, 2021.
430

[27] Y. Lu and J. Lu. A universal approximation theorem of deep neural networks for expressing
431

probability distributions. In Advances in Neural Information Processing Systems, pages 3094–
432

3105, 2020.
433

[28] D. Madras, E. Creager, T. Pitassi, and R. Zemel. Fairness through causal awareness: Learning
434

causal latent-variable models for biased data. In Proceedings of the Conference on Fairness,
435

Accountability, and Transparency, pages 349–358, 2019.
436

[29] W. Miao, Z. Geng, and E. J. Tchetgen Tchetgen. Identifying causal effects with proxy variables
437

of an unmeasured confounder. Biometrika, 105(4):987–993, 2018.
438

[30] J. M. Montgomery, B. Nyhan, and M. Torres. How conditioning on posttreatment variables
439

can ruin your experiment and what to do about it. American Journal of Political Science,
440

62(3):760–775, 2018.
441

[31] J. Pearl. On measurement bias in causal inference. arXiv preprint arXiv:1203.3504, 2012.
442

[32] J. Pearl. Conditioning on post-treatment variables. Journal of Causal Inference, 3(1):131–137,
443

2015.
444

[33] S. J. Pocock, S. E. Assmann, L. E. Enos, and L. E. Kasten. Subgroup analysis, covariate
445

adjustment and baseline comparisons in clinical trial reporting: current practiceand problems.
446

Statistics in Medicine, 21(19):2917–2930, 2002.
447

[34] M. Prosperi, Y. Guo, M. Sperrin, J. S. Koopman, J. S. Min, X. He, S. Rich, M. Wang, I. E.
448

Buchan, and J. Bian. Causal inference and counterfactual prediction in machine learning for
449

actionable healthcare. Nature Machine Intelligence, 2(7):369–375, 2020.
450

[35] K. J. Rothman, S. Greenland, T. L. Lash, et al. Modern epidemiology, volume 3. Wolters
451

Kluwer Health/Lippincott Williams & Wilkins Philadelphia, 2008.
452

[36] U. Shalit, F. D. Johansson, and D. Sontag. Estimating individual treatment effect: generalization
453

bounds and algorithms. In International Conference on Machine Learning, pages 3076–3085,
454

2017.
455

[37] C. Shi, D. Blei, and V. Veitch. Adapting neural networks for the estimation of treatment effects.
456

In Advances in Neural Information Processing Systems, 2019.
457

[38] C. Shi, V. Veitch, and D. M. Blei. Invariant representation learning for treatment effect estimation.
458

In Uncertainty in Artificial Intelligence, pages 1546–1555. PMLR, 2021.
459

[39] P. Spirtes, C. N. Glymour, R. Scheines, and D. Heckerman. Causation, prediction, and search.
460

MIT press, 2000.
461

11


---Page Break---
[40] S. Wager and S. Athey. Estimation and inference of heterogeneous treatment effects using
462

random forests. Journal of the American Statistical Association, 113(523):1228–1242, 2018.
463

[41] H. Wang, K. Kuang, H. Chi, L. Yang, M. Geng, W. Huang, and W. Yang. Treatment effect
464

estimation with adjustment feature selection.
In Proceedings of the 29th ACM SIGKDD
465

Conference on Knowledge Discovery and Data Mining, pages 2290–2301, 2023.
466

[42] L. Yao, S. Li, Y. Li, M. Huai, J. Gao, and A. Zhang. Representation learning for treatment effect
467

estimation from observational data. In Advances in Neural Information Processing Systems,
468

volume 31, 2018.
469

[43] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf. Kernel-based conditional independence test
470

and application in causal discovery. arXiv preprint arXiv:1202.3775, 2012.
471

[44] W. Zhang, L. Liu, and J. Li. Treatment effect estimation with disentangled latent factors. In
472

Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10923–10930,
473

2021.
474

12


---Page Break---
Appendix
475

A
Broader Impact
476

The proposed CiVAE is a universal model for causal effect estimation with observational data.
477

Although we use the Company job data that estimate the causal effects of online working mode to
478

applicant statistics as a real-world example, proxy-of-confounder-based methods have been heavily
479

used in other observational studies, which may be susceptible to latent post-treatment bias. Therefore,
480

we speculate that the proposed CiVAE will have a broader impact on causal inference community.
481

B
Related Work
482

B.1
Post-Treatment Bias in Causal Inference
483

Bias due to accidentally controlling post-treatment variables, i.e., post-treatment bias, has long been
484

recognized as dangerous in causal effect estimation [20]. Back at 2005, Pearl [32] cautioned that
485

controlling more is not better, and uses the collider bias [9] and M-Bias [7] as two examples to
486

show that bias can be increased when controlling the post-treatment variables. Furthermore, [30]
487

show that indirect correlations between post-treatment variable M and outcome Y can still cause
488

bias. Recent works prove that even if M has no causal relationship with Y , controlling it can still
489

increase the variance of estimand [12]. However, most of these works study the post-treatment bias
490

in the observational space, where latent post-treatment variables that are mixed with confounders to
491

generate the observed covariates can be easily ignored by the researcher. Therefore, it motivates us to
492

develop CiVAE, which is robust to the latent post-treatment bias under mild assumptions.
493

B.2
Covariate Disentanglement
494

Recently, researchers have realized that directly controlling proxy of confounders X may not be
495

safe, as variables other than confounders could lurk in the proxy and ruin the ATE estimation [12].
496

Traditional methods assume that the variables that generate X are a mixture of confounders, adjusters,
497

and influencers [36], where adjusters should not be controlled as it can increase the estimation
498

variance [11]. Most methods rely on the fact that adjusters are correlated only with the treatment
499

to separate them from other variables [12, 44] (see Fig. (1)). This can also be used to remove post-
500

treatment variables that are not correlated with the outcome, which have similar statistics properties
501

with adjustors [41]. Here, a different work is NICE [38], which uses the fact that confounders and
502

influencers are direct causal parents of the outcome to find these variables with invariant learning as
503

the control set [3]. However, since mediators are also direct parents of the outcome, NICE is still not
504

robust to general post-treatment bias. Given that all above methods cannot satisfactorily address the
505

latent post-treatment in general cases, it is imperative to design the CiVAE, where confounders can
506

be identified and distinguished with latent post-treatment variables for unbiased adjustment.
507

C
Theoretical Analysis
508

C.1
Proof of Lemma 3.1.
509

Proof. Let Z = f(X) and z = f(x). If f is injective and differentiable a.e., and f † is the
510

left-inverse, we have:
511

fY |f(X)(y|f(x)) = fY |Z(y|z) = fY,Z(y, z)

fZ(z)
= fY,X(y, f †(z))|Jf †(z)|

fX(f †(z))|Jf †(z)|
= fY,X(y, x)

fX(x)
= fY |X(y|x),

(12)
where f· and f·|· represent the marginal and conditional density function, respectively, and Jf †(z) is
512

the Jacobian matrix of function f † evaluated at z. Based on Eq. (12), we have:
513

E[Y |X] =
Z
y·fY |X(y|x)dy =
Z
y·fY |Z(y|z)dy = E[Y |Z = z] = E[Y |f(X) = f(x)]. (13)

514

13


---Page Break---
C.2
Proof of Corollary 3.3.
515

Proof. For X = x, let [c||m] .= [f †
C(x)||f †
M(x)] .= f †(x) = A†(x −αX), where A† is the left
516

inverse of the full column-rank matrix A in Eq. (2), we have:
517

CATE(x) = E[Y |T = 1, C = f †
C(x)] −E[Y |T = 0, C = f †
C(x)]
= E[Y |T = 1, C = c] −E[Y |T = 0, C = c]

= E[αY + τ · T +
X
θj · Mj +
X
κi · Ci|T = 1, C = c]

−E[αY + τ · T +
X
θj · Mj +
X
κi · Ci|T = 0, C = c]

= αY + τ · E[T|T = 1, C = c] +
X
θj · E[Mj|T = 1, C = c] +
X
κi · E[Ci|T = 1, C = c]

−αY + τ · E[T|T = 0, C = c] +
X
θj · E[Mj|T = 0, C = c] +
X
κi · E[Ci|T = 0, C = c]

= τ · (1 −0) +
X
θj · (γj · (1 −0)) +
X
κi · (ci −ci)

= τ +
X
θj · γj = E[τ +
X
θj · γj] = ATE,
(14)
where the first equality is due to the definition of CATE in Eq. (2). In addition, the causal estimand
518

and bias of a proxy-of-confounder-based causal inference model that controls the latent variable Z
519

inferred via Z = ¯f(X) = BT X (where B is also a full column-rank matrix) can be formulated as:
520

DCEV (BT x) = E[Y |T = 1, Z = BT x] −E[Y |T = 0, Z = BT x]

= E[Y |T = 1, Z = BT αX + BT A[c||m]] −E[Y |T = 0, Z = BT αX + BT A[c||m]]

(a)
= E[Y |T = 1, C = c, M = m] −E[Y |T = 0, C = c, M = m]

= αY + τ · 1 +
X
θj · E[Mj|T = 1, C = c, M = m] +
X
κi · E[Ci|T = 1, C = c, M = m]

−αY + τ · 0 +
X
θj · E[Mj|T = 0, C = c, M = m] +
X
κi · E[Ci|T = 0, C = c, M = m]

= τ · (1 −0) +
X
θj · (mj −mj) +
X
κi · (ci −ci)

= τ = E[τ] = E[DCEV (BT X)],
(15)

where step (a) is due to the fact that, since both A and B are full column-rank matrices, BT A is
521

an invertible matrix, and the mapping f = BT αX + BT A is bijective. Therefore, we can invoke
522

Lemma 3.1 and apply the left-inverse of f, i.e., f † = (BT A)−1 −BT αX, to the condition of the
523

expectation. The rest steps are based on the structural causal equations defined in Eq. (2).
524

C.3
Another Case of Linear SCM with Latent Correlators
525

Corollary C.1. For another Linear Structural Causal Model defined as follows
526

T ←1(αT +
X
βi · Ci > a)

Mj ←αM + γj · T + ϕj · Uj
X ←αX + A[M||C]

Y ←αY + τ · T +

X
θj · Uj +
X
κi · Ci,

(16)

where f = A ∈RKX×(KC+KM) is a full column-rank matrix, the CATE, ATE, and the bias of
527

proxy-of-confounder-based causal inference model that controls the latent variable Z inferred via
528

Z = ¯f(X) = BT X can be formulated as follows:
529

ATE = CATE = τ

E[DCEV (Z = BT X)] = DCEV (Z = BT X) = τ−
X θj · γj

ϕj

Bias = ATE −E[DCEV (BT X)] =

X θj · γj

ϕj

,

(17)

14


---Page Break---
where B ∈RKX×(KC+KM) is another full column-rank matrix. Since P θj·γj

ϕj
is arbitrary, the
530

estimator E[DCEV (BT X)] is arbitrarily biased for the estimation of ATE.
531

Proof. The proof of the CATE and ATE is trivial. The causal estimand and the bias of a proxy-
532

of-confounder-based causal inference model that controls the latent variables Z inferred via Z =
533

¯f(X) = BT X (where B is also a full column-rank matrix) can be formulated as follows:
534

DCEV (BT x) = E[Y |T = 1, Z = BT x] −E[Y |T = 0, Z = BT x]

= E[Y |T = 1, Z = αX + BT A[c||m]] −E[Y |T = 0, Z = αX + BT A[c||m]]

(a)
= E[Y |T = 1, C = c, M = m] −E[Y |T = 0, C = c, M = m]

= αY + τ · 1 +
X
θj · E[Uj|T = 1, C = c, M = m] +
X
κi · E[Ci|T = 1, C = c, M = m]

−αY + τ · 0 +
X
θj · E[Uj|T = 0, C = c, M = m] +
X
κi · E[Ci|T = 0, C = c, M = m]

= τ · (1 −0) +
X
θj · (ϕ−1
j
· (mj −αM −γj) −ϕ−1
j
· (mj −αM)) +
X
κi · (ci −ci)

= τ −
X θj · γj

ϕj
= E

τ −
X θj · γj

ϕj


= E[DCEV (BT X)],

(18)

535

where step (a) and the rest of the proof follow the same logic as the proof in Section 3.3.
536

C.4
Proof of Theorem 4.1
537

The strict definitions of the exponential family, strong exponential (which is assumed for the factorized
538

part of the conditional prior), and identifiability follow [19, 26], and can be referred to in Appendix
539

E, F of [26], which we omit to avoid redundancy. The proof of Theorem 4.1 is largely based on the
540

NF-iVAE paper [26], where most of the details can be found, with the new assumption introduced in
541

CiVAE that each Sf,i has at least one invertible dimension incorporated to ensure that each dimension
542

of the inferred latent variables is a bijective transformation of the corresponding true latent variable.
543

C.4.1
PART I
544

Step I. In this step, we transform the equality of noisy conditional marginal distribution of X given
545

Y, T of two models with parameter θ, ˜θ ∈Θ into the equality of noise-free distributions.
546

pθ(X | Y, T) = p ˜θ(X | Y, T)

=⇒
Z

Z
pf(X | Z)pS,λ(Z | Y, T)dZ =
Z

Z
p ˜
f(X | Z)p ˜
S,˜λ(Z | Y, T)dZ

=⇒
Z

Z
pε(X −f(Z))pS,λ(Z | Y, T)dZ =
Z

Z
pε(X −˜f(Z))p ˜
S,˜λ(Z | Y, T)dZ

(a)
=⇒
Z

X
pε(X −X)pS,λ
 
f †(X) | Y, T

vol
 
Jf †(X)

dX =
Z

X
pε(X −X)p ˜
S,˜λ

˜f †(X) | Y, T

vol

J ˜
f †(X)

dX

(b)
=⇒
Z

Rd pε(X −X)˜pf,S,λ,Y,T (X)dX =
Z

Rd pε(X −X)˜p ˜
f, ˜
S,˜λ, ˜Y , ˜T (X)dX

=⇒(˜pf,S,λ,Y,T ∗pε) (X) =

˜p ˜
f, ˜
S,˜λ, ˜Y , ˜T ∗pε

(X)

(c)
=⇒F [˜pf,S,λ,Y,T ] (ω)φε(ω) = F
h
˜p ˜
f, ˜
S,˜λ, ˜Y , ˜T
i
(ω)φε(ω)

(d)
=⇒F [˜pf,S,λ,Y,T ] (ω) = F
h
˜p ˜
f, ˜
S,˜λ, ˜Y , ˜T
i
(ω)

=⇒˜pf,S,λ,Y,T (X) = ˜p ˜
f, ˜
S,˜λ, ˜Y , ˜T (X).

(19)

15


---Page Break---
Step (a) is based on the rule of change-of-variable, where vol(A) =
r

det

AT A

. In step (b),
547

we define ˜pf,S,λ,Y,T (X) ≜pS,λ
 
f †(X) | Y, T

vol
 
Jf †(X)

IX (X). In step (c), we use F[·] to
548

denote the Fourier transform. In step (d), we drop φε(ω) as it is non-zero a.e. (see Assumption 3).
549

Step II. In this step, we transform the equality of the noise-free distributions into the relationship of
550

the sufficient statistics S and ˜S. By taking logarithm of both sides of Eq. (19), we have:
551

log vol
 
Jf †(X)

+ log Q
 
f †(X)

−log C(Y, T) +

S
 
f †(X)

, λ(Y, T)


= log vol

J ˜
f †(X)

+ log ˜Q

˜f †(X)

−log ˜C(Y, T) +
D
˜S

˜f †(X)

, ˜λ(Y, T)
E
.
(20)

Let (Y, T)0, · · · , (Y, T)k be the k + 1 distinct points defined in Assumption 3 - (iv). We obtain k + 1
552

equations by evaluating the Eq. (20) at these points, where the first equation is subtracted from the
553

remaining ones, which leads to the following equation system:
554


S
 
f †(X)

, λ ((Y, T)l) −λ ((Y, T)0)⟩+ log C ((Y, T)0)

C ((Y, T)l)

=
D
˜S

˜f †(X)

, ˜λ ((Y, T)l) −˜λ ((Y, T)0)
E
+ log
˜C ((Y, T)0)

˜C ((Y, T)l)
,
l = 1, · · · , k.
(21)

Let L be the invertible matrix defined in Assumption 3 - (iv) and ˜L be the counterpart for ˜λ, if we
555

summarize all terms irrelevant to X into a constant b,we have:
556

LT S
 
f †(X)

= ˜LT ˜S

˜f †(X)

+ b

=⇒S
 
f †(X)

= A ˜S

˜f †(X)

+ c,
(22)

where A = L−T ˜L ∈Rk×k, and c = L−T b ∈Rk.
557

Step III. Ideally, to prove the element-wise bijective identifiability of the latent variables Z, the
558

transformation of the sufficient statistics S derived in Eq. (22) should be bijective. We claim that if
559

the conditional prior pS,λ(Z | Y, T) is strongly exponential and L is invertible, ˜L and A must also
560

be invertible. The proof is omitted, and can be referred to in Appendix H.1.1 of [26].
561

C.4.2
PART II
562

In this part, we prove that, if Assumptions 1, 2 and 3 hold, we can identify the factorized part
563

of the sufficient statistics S(Z), i.e., Sf(Z), up to permutation and element-wise transformation.
564

Specifically, if we use v to denote the composite map ˜f † ◦f : Z →Z, Eq. (22) can be rewritten into:
565

S(Z) = A ˜S(v(Z)) + c.
(23)

We aim to prove that A in Eq. (23) is a block permutation matrix.
566

Step I. We start by showing that v is a component-wise function. If we differentiate both sides of Eq.
567

(23) with respect to Zs and Zt, where s ̸= t, we have:
568

∂S(Z)

∂Zs
= A

KZ
X

i=1

∂˜S(v(Z))

∂vi(Z)
· ∂vi(Z)
∂Zs

∂2S(Z)
∂Zs∂Zt
= A

KZ
X

i=1

KZ
X

i=1

∂2 ˜S(v(Z))
∂vi(Z)∂vj(Z) · ∂vj(Z)

∂Zt
· ∂vi(Z)
∂Zs
+ A

KZ
X

i=1

∂˜S(v(Z))

∂vi(Z)
· ∂2vi(Z)
∂Zs∂Zt
.

(24)

Note that for the factorized part of the sufficient statistics S, i.e., Sf, all cross-derivatives are zero,
569

and for the non-factorized part of S, i.e., Snf, which is a neural network with ReLU activation (i.e.,
570

linear a.e.), all second-order derivatives are zero. Therefore, the second order cross-derivatives on
571

the LHS. of Eq. (24) are zero, which leads to the following equality:
572

0 = A

KZ
X

i=1

∂2 ˜S(v(Z))

∂vi(Z)2
· ∂vi(Z)
∂Zt
· ∂vi(Z)
∂Zs
+ A

KZ
X

i=1

∂˜S(v(Z))

∂vi(Z)
· ∂2vi(Z)
∂Zs∂Zt
.
(25)

16


---Page Break---
Eq. (25) can be written into the matrix-vector product form as follows:
573

0 = A ˜S′′(Z)v′
s,t(Z) + A ˜S′(Z)v′′
s,t(Z),
(26)

where

˜S′′(Z) =

"
∂2 ˜S(v(Z))

∂v1(Z)2 , · · · , ∂2 ˜S(v(Z))

∂vKZ(Z)2

#

∈Rk×KZ,

v′
s,t(Z) =
∂v1(Z)

∂Zt
· ∂v1(Z)
∂Zs
, · · · , ∂vKZ(Z)

∂Zt
· ∂vKZ(Z)
∂Zs

T
∈RKZ,

and

˜S′(Z) =

"
∂˜S(v(Z))

∂v1(Z) , · · · , ∂˜S(v(Z))

∂vKZ(Z)

#

∈Rk×KZ,

v′′
s,t(Z) =
∂2v1(Z)

∂Zs∂Zt
, · · · , ∂2vKZ(Z)

∂Zs∂Zt

T
∈RKZ.

If we denote the concatenation as ˜S′′′(Z) =
h
˜S′′(Z), ˜S′(Z)
i
∈Rk×2KZ and v′′
s,t(Z) =
574


v′
s,t(Z)T , v′′
s,t(Z)T T ∈R2Kz, we have:
575

0 = A ˜S′′′(Z)v′′′
s,t(Z).
(27)

Finally, if we denote the rows of ˜S′′′(Z) that correspond to the factorized part of S by ˜S′′′
f (Z),
576

according to Lemma 5 of the iVAE paper [19] and the assumption that k ≥2KZ, we have that the
577

rank of ˜S′′′
f (Z) is 2KZ. Since k ≥2KZ, the rank of ˜S′′′
f (Z) is also 2KZ. Since the rank of A is k,
578

the rank of A ˜S′′′(Z) is 2KZ, which implies that v′′′
s,t(Z) ∈R2KZ is a zero vector. Therefore, we
579

have v′
s,t(Z) = 0, ∀s ̸= t, and we have demonstrated that v is a component-wise function.
580

Step II. Based on Step I, we demonstrate that A is a block permutation matrix. Without loss of gen-
581

erality, we assume that the permutation in v is Identity, where v(Z) = [v1 (Z1) , · · · , vKZ (ZKZ)]T
582

and each vi is a nonlinear univariate scalar function. Since f and ˜f are injective, v is bijective and
583

v−1(Z) =

v−1
1
(Z1) , · · · , v−1
KZ (ZKZ)
T . If we denote S(v(Z)) = ˜S(v(Z)) + A−1c, Eq. (23)
584

can be reformulated as S(Z) = AS(v(Z)). We then apply v−1 to Z on both sides, which gives
585

S
 
v−1(Z)

= AS(Z).
(28)

Let t be the index of an entry in S that corresponds to the factorized part Sf. For all s ̸= t, we have:
586

0 = ∂S
 
v−1(Z)


t
∂Zs
=

k
X

j=1
atj
∂S(Z)j

∂Zs
.
(29)

Since the entries of ˜S are linearly independent, atj is zero for any j such that ∂S(Z)j

∂Zs
̸= 0. This
587

includes the entries Sj that correspond to (1) the factorized part that does not depend on Zt; and (2)
588

the non-factorized part Snf. Therefore, when t is the index of an entry in the sufficient statistics S
589

that corresponds to factor i in the factorized part Sf, i.e., Sf,i, the only non-zero atj are the ones that
590

map between Sf,i (Zi) and Sf,i (vi (Zi)). Therefore, we can construct an invertible submatrix A′
i
591

with all non-zero elements atj for all t that corresponds to factor i, such that
592

Sf,i (Zi) = A′
iSf,i (vi (Zi)) = A′
i ˜Sf,i (vi (Zi)) + ci,
i = 1, · · · , KZ,
(30)

where ci denotes the corresponding elements of c. Eq. (30) means that for each i = 1, · · · , KZ,
593

the matrix block A′
i of A affinely transforms the i-specific sufficient statistics vector Sf,i (Zi) into
594

˜Sf,i (vi (Zi)). In addition, there is also an additional block A′ that affinely transforms Snf(Z) in
595

into Snf(v(Z)). This completes the proof that A is a block permutation matrix.
596

17


---Page Break---
C.4.3
PART III
597

Let ˜Zi = vi (Zi) = ˜f †(X)i be the ith inferred latent variable. Assume again that the permutation in
598

v is Identity. In this part, we prove that if Assumption 2 holds, each inferred latent variable ˜Zi is the
599

bijective transformation of the true latent variable. The proof is as follows.
600

Proof. Plugging ˜Zi into Eq. (30), we have:
601

Sf,i(Zi) = A′
i ¯Sf,i( ˜Zi).
(31)

According to Assumption 2, there exists one dimension of Sf,i, i.e., j, such that Sf,ij is bijective.
602

This implies that Sf,i is injective, and therefore it has a left-inverse S†
f,i. we apply S†
f,i to both sides
603

of Eq. (31), which gives:
604

Zi = S†
f,iA′
i ¯Sf,i( ˜Zi).
(32)

Since A′
i is a block of an invertible block permutation matrix, Ai is also an invertible matrix, and
605

therefore A′
i is a bijective mapping. In addition, since ˜Sf,i is injective, ¯Sf,i is also injective, and
606

therefore the composite map S†
f,iA′
i ¯Sf,i : R →R that applies on ˜Zi is a bijective. This completes
607

the proof that each inferred latent variable ˜Zi is the bijective transformation of the true latent variable
608

in the case of no noise, where Z = f †(X) are the true latent variables. If noise ε exists, the posterior
609

distribution of the latent variables can be identified up to an analogous bijective indeterminacy.
610

C.4.4
Consistency
611

Proof. If the family of the variational posterior qϕ(Z|X, Y, T) contains the true posterior
612

pθ(Z|X, Y, T), then by optimizing the loss of Eq. (9) (with the KL term replaced by the score match-
613

ing loss defined in Eq. (10)) over its parameter ϕ, the score matching term will eventually vanish.
614

Therefore, the ELBO term in Eq. (9) will be equal to the log-likelihood. Under this circumstance,
615

CiVAE inherits all the properties of maximum likelihood estimation (MLE). Since the identifiability
616

of CiVAE is guaranteed up to permutation and component-wise bijective transformation of the latent
617

variables, the consistency property of MLE means that the model will converge to the true parameter
618

θ∗up to such mild indeterminacy of the latent variables in the limit of infinite data.
619

C.5
Proof of Theorem 4.2
620

Proof. Let C be the true latent confounders and ˜C be the transformed confounders, where the
621

transformation function f is bijective and differentiable a.e. Let f −1 denote its inverse. The ATE
622

estimator that controls transformed confounders ˜C can be formulated as:
623

DEV ( ˜
C) = Ep( ˜
C)[E[Y |T = 1, ˜C = ˜c] −E[Y |T = 0, ˜C = ˜c]].
(33)

Specifically, for the continuous case where density functions exist, for each term, we have:
624

Ep( ˜
C)[E[Y |T = t, ˜C = ˜c]] =
Z
f ˜
C(˜c)
Z
y · fY |T, ˜
C(y|t, ˜c)dyd˜c.
(34)

For the marginal density f ˜
C(˜c), the following equality holds:
625

f ˜
C(˜c) = fC(f −1(˜c))|Jf −1(˜c)| = fC(c)|Jf −1(˜c)|.
(35)

As for the conditional density fY |T, ˜
C(y|t, ˜c), since f is bijective, according to Eq. (12), we have:
626

fY |T, ˜
C(y|t, ˜c) = fY |T,C(y|t, c).
(36)

Combining Eqs. (35) and (36), and given that d˜c = |Jf(c)|dc, we have:
627

(34) =
Z
fC(c)|Jf −1(˜c)|
Z
y · fY |T,C(y|t, c)dy|Jf(c)|dc

=|Jf −1(˜c)| · |Jf(c)|
Z
fC(c)
Z
y · fY |T,C(y|t, c)dydc

(a)
=
Z
fC(c)
Z
y · fY |T,C(y|t, c)dydc

=Ep(C)[E[Y |T = t, C = c]],

(37)

18


---Page Break---
Table 2: Comparison of CiVAE with baselines when intra-interactions among M exist.

Dataset
LatentMediator
LatentCorrelator
Company (Age)
Company (Gender)
Method
ATE.
Err.
ATE.
Err.
ATE.
Err.
ATE.
Err.

CEVAE
1.627 ± 0.549
2.627
2.659 ± 0.302
1.353
0.152 ± 0.027
0.420
-0.225 ± 0.044
-0.144
TEDVAE
1.653 ± 0.511
2.042
2.827 ± 0.259
1.521
0.180 ± 0.047
0.448
-0.189 ± 0.012
-0.108
CiVAE
-0.350 ± 0.695
1.785
1.785 ± 0.481
0.479
-0.073 ±0.101
0.195
-0.136 ± 0.087
-0.055
True ATE
-1.000 ± 0.000
0.000
1.306 ± 0.000
0.000
-0.268 ± 0.000
0.000
-0.081 ± 0.000
0.000

Table 3: Comparison of CiVAE with baselines when inter-interactions between C and M exist.

Dataset
LatentMediator
LatentCorrelator
Company (Age)
Company (Gender)
Method
ATE.
Err.
ATE.
Err.
ATE.
Err.
ATE.
Err.

CEVAE
2.070 ± 0.279
3.070
2.831 ± 0.398
1.831
0.094 ± 0.061
0.362
-0.192 ± 0.015
-0.111
TEDVAE
1.743 ± 0.307
2.743
2.954 ± 0.763
1.954
0.109 ± 0.116
0.377
-0.212 ± 0.019
-0.131
CiVAE
-0.716 ± 0.523
0.284
1.385 ± 0.660
0.385
-0.041 ±0.144
0.227
-0.129 ± 0.064
-0.048
True ATE
-1.000 ± 0.000
0.000
1.000 ± 0.000
0.000
-0.268 ± 0.000
0.000
-0.081 ± 0.000
0.000

where the term |Jf −1(˜c)| · |Jf(c)| vanishes in step (a) as the two factors have the product of one.
628

Therefore, if we plug Eq. (37) into Eq. (33), it leads to the following equality:
629

DEV ( ˜
C) = Ep( ˜
C)[E[Y |T = 1, ˜C = ˜c] −E[Y |T = 0, ˜C = ˜c]]

= Ep(C)[E[Y |T = 1, C = c] −E[Y |T = 0, C = c]] = DEV (C) = ATE,
(38)

where the last step is due to Eq. (2) in Definition 2, which completes our proof that controlling
630

bijectively transformed confounders provides an unbiased estimation of ATE.
631

D
Extending CiVAE to address Latent Interactions
632

In this section, we extend CiVAE to more general cases where interactions exist among the latent
633

confounders C and the latent post-treatment variables M. Here, we note that the identification
634

of latent confounders C in CiVAE is achieved in two steps. (i) CiVAE individually identifies
635

latent variables [C, M] that generate X in inferred Z (but which dims of Z correspond to C
636

or M is unknown). (ii) pairwise independence test to identify C. Since Assumption 2 allows
637

arbitrary dependence among C and M, step (i) still holds when interactions among [C, M] exist.
638

To distinguish C in these cases, we can use more general causal discovery algorithms, e.g., the
639

PC algorithm [18] in the second step. In this section, we consider two cases of interaction: (i)
640

Intra-Interaction among mediators, and (ii) Inter-Interaction among mediators and confounders.
641

D.1
Intra-Interactions among Latent Mediators
642

In this subsection, we discuss the case where latent post-treatment variables M interact with each
643

other. Since in this case, M cannot causally influence the latent confounders C (otherwise C will be
644

post-treatment), and the PC algorithm orients edges in causal graphs via colliders, latent confounders
645

can still be identified from the inferred Z as they form colliders with the treatment T.
646

To empirically verify the claim, we extend the simulated datasets described in Section 5.1, where we
647

make (i) T directly affects M1, (ii) M1 affects M2, and (iii) M1, M2 affect M3. The coefficients are
648

randomly sampled from N(0, 1/3). In step (ii), we use the PC algorithm [18] to identify C from
649

the inferred Z. The results in Table 2 demonstrate that the adapted CiVAE is still significantly more
650

robust to latent post-treatment bias compared to CEVAE and TEDVAE, which empirically verify our
651

claim that PC-adapted CiVAE can address the interaction among post-treatment variables.
652

D.2
Inter-Interactions between Latent Mediators and Latent Confounders
653

In this subsection, we discuss another case where inter-interactions exist between latent confounders
654

C and latent post-treatment variables M. Since in this case, M still cannot causally influence C
655

(otherwise C will be post-treatment), and the PC algorithm orients edges in causal graph via colliders,
656

latent confounders C can still be identified from Z as they form colliders with the treatment T.
657

19


---Page Break---
To verify the claim, we extend the simulated datasets described in Section 5.1 to allow each latent
658

confounder Ci ∈R3 to determine M ∈R3. The coefficients are randomly sampled from N(0, 1/3).
659

In step (ii), we use the PC algorithm to identify C from the inferred Z. The results in Table 3
660

demonstrate that the PC-adapted CiVAE is still significantly more robust to latent post-treatment bias
661

compared to CEVAE and TEDVAE, which empirically verify our claim that PC-adapted CiVAE can
662

address the case where inter-interactions exist among latent confounders and post-treatment variables.
663

20


---Page Break---
NeurIPS Paper Checklist
664

1. Claims
665

Question: Do the main claims made in the abstract and introduction accurately reflect the
666

paper’s contributions and scope?
667

Answer: [Yes]
668

Justification: The contribution of this paper can be summarized as: We study a critical but
669

easily overlooked problem in causal effect estimation: latent post-treatment bias, and we
670

propose a novel framework, i.e., CiVAE, to address the bias. The details are in Section 4.
671

2. Limitations
672

Question: Does the paper discuss the limitations of the work performed by the authors?
673

Answer: [Yes]
674

Justification: We have discussed the potential issue of the vanilla when interactions among
675

the latent variables exists. However, in Section D we have addressed the issue by extendeding
676

our framework.
677

3. Theory Assumptions and Proofs
678

Question: For each theoretical result, does the paper provide the full set of assumptions and
679

a complete (and correct) proof?
680

Answer: [Yes]
681

Justification: We have introduced the three mild assumptions required for the identification
682

of causal effects under latent post-treatment bias. In addition, we have provided the proof
683

for all the theorems in the Appendix.
684

4. Experimental Result Reproducibility
685

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
686

perimental results of the paper to the extent that it affects the main claims and/or conclusions
687

of the paper (regardless of whether the code and data are provided or not)?
688

Answer: [Yes]
689

Justification: We have provided implementation details in Section 5.1. In addition, we have
690

provided a code demo in an anonymous URL.
691

5. Open access to data and code
692

Question: Does the paper provide open access to the data and code, with sufficient instruc-
693

tions to faithfully reproduce the main experimental results, as described in supplemental
694

material?
695

Answer: [Yes]
696

Justification: See Checklist 4.
697

6. Experimental Setting/Details
698

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
699

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
700

results?
701

Answer: [Yes]
702

Justification: See Checklist 4.
703

7. Experiment Statistical Significance
704

Question: Does the paper report error bars suitably and correctly defined or other appropriate
705

information about the statistical significance of the experiments?
706

Answer: [Yes]
707

Justification: We have reported the error of five independent run for both the proposed
708

CiVAE and all the baselines in the main paper.
709

8. Experiments Compute Resources
710

21


---Page Break---
Question: For each experiment, does the paper provide sufficient information on the com-
711

puter resources (type of compute workers, memory, time of execution) needed to reproduce
712

the experiments?
713

Answer: [Yes]
714

Justification: See Checklist 4.
715

9. Code Of Ethics
716

Question: Does the research conducted in the paper conform, in every respect, with the
717

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
718

Answer: [Yes]
719

Justification: We have carefully read the code of ethics and behaved strictly according to it.
720

10. Broader Impacts
721

Question: Does the paper discuss both potential positive societal impacts and negative
722

societal impacts of the work performed?
723

Answer: [Yes]
724

Justification: See Section A of the Appendix.
725

11. Safeguards
726

Question: Does the paper describe safeguards that have been put in place for responsible
727

release of data or models that have a high risk for misuse (e.g., pretrained language models,
728

image generators, or scraped datasets)?
729

Answer: [NA]
730

Justification: Our model does not have a high risk for misuse.
731

12. Licenses for existing assets
732

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
733

the paper, properly credited and are the license and terms of use explicitly mentioned and
734

properly respected?
735

Answer: [Yes]
736

Justification: We have cited the papers of our baselines and honor their license of code.
737

13. New Assets
738

Question: Are new assets introduced in the paper well documented and is the documentation
739

provided alongside the assets?
740

Answer: [Yes]
741

Justification: We provide the Readme file along side the codes.
742

14. Crowdsourcing and Research with Human Subjects
743

Question: For crowdsourcing experiments and research with human subjects, does the paper
744

include the full text of instructions given to participants and screenshots, if applicable, as
745

well as details about compensation (if any)?
746

Answer: [NA]
747

Justification: No human subjects are involved in our experiments.
748

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
749

Subjects
750

Question: Does the paper describe potential risks incurred by study participants, whether
751

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
752

approvals (or an equivalent approval/review based on the requirements of your country or
753

institution) were obtained?
754

Answer: [NA]
755

Justification: No human subjects are involved in our experiments.
756

22


---Page Break---
