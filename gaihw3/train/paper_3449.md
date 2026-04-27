A Conditional Independence Test in the
Presence of Discretization

Anonymous Author(s)
Affiliation
Address
email

Abstract

Testing conditional independence (CI) has many important applications, such as
1

Bayesian network learning and causal discovery. Although several approaches have
2

been developed for learning CI structures for observed variables, those existing
3

methods generally fail to work when the variables of interest can not be directly
4

observed and only discretized values of those variables are available. For example,
5

if X1, ˜X2 and X3 are the observed variables, where ˜X2 is a discretization of the
6

latent variable X2, applying the existing methods to the observations of X1, ˜X2
7

and X3 would lead to a false conclusion about the underlying CI of variables
8

X1, X2 and X3. Motivated by this, we propose a CI test specifically designed to
9

accommodate the presence of discretization. To achieve this, a bridge equation
10

and nodewise regression are used to recover the precision coefficients reflecting
11

the conditional dependence of the latent continuous variables under the nonpara-
12

normal model. An appropriate test statistic has been proposed, and its asymptotic
13

distribution under the null hypothesis of CI has been derived. Theoretical analysis,
14

along with empirical validation on various datasets, rigorously demonstrates the
15

effectiveness of our testing methods.
16

1
Introduction
17

Independence and conditional independence (CI) are fundamental concepts in statistics. They are
18

leveraged for exploring queries in statistical inference, such as sufficiency, parameter identification,
19

adequacy, and ancillarity [9]. They also play a central role in emerging areas such as causal discovery
20

[18], graphical model learning, and feature selection [36]. Tests for CI have attracted increasing
21

attention from both theoretical and application sides.
22

Formally, the problem is to test the CI of two variables Xj1 and Xj2 given a random vector (a set
23

of other variables) Z. In statistical notation, the null hypothesis is written as H0 : Xj1 ⊥Xj2 | Z,
24

where ⊥denotes “independent from.” The alternative hypothesis is written as H1 : Xj1 ̸⊥Xj2 | Z,
25

where ̸⊥denotes “dependent with.” The null hypothesis implies that once Z is known, the values of
26

Xj1 provide no additional information about Xj2, and vice versa. Different tests have been designed
27

to handle different scenarios, including Gaussian variables with linear dependence [37, 25, 22, 26]
28

and non-linear dependence [16, 38, 31, 27, 1] (For detailed related work, please refer to App. D).
29

Given observations of Xj1, Xj2, and Z, the CI can be effectively tested with existing methods.
30

However, in many scenarios, accurately measuring continuous variables of interest is challenging
31

due to limitations in data collection. Sometimes the data obtained are approximations represented as
32

discretized values. For example, in finance, variables such as asset values cannot be measured and are
33

binned into ranges for assessing investment risks (e.g., sell, hold, and strong buy) [7, 8]. Similarly,
34

in mental health, anxiety levels are often assessed using scales like the GAD-7, which categorizes
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
responses into levels such as mild, moderate, or severe [23, 17]. In the entertainment industry, the
36

quality of movies is typically summarized through viewer ratings [29, 10].
37

(a)
(b)
(c)

Figure 1: We illustrate different data generative
processes with causal graphical models. The dis-
cretization process introduces new discrete vari-
ables which are denoted with a tilde (∼).

When discretization is present, existing CI tests
38

can fail to determine the CI of underlying con-
39

tinuous variables. This issue arises because ex-
40

isting CI tests treat discretized observations as
41

observations of continuous variables, leading
42

to incorrect conclusions about their CI relation-
43

ships. More precisely, the problem lies in the
44

discretization process, which introduces new dis-
45

crete variables. Consequently, although the in-
46

tent is to test the CI of the underlying continuous
47

variables, what is actually being tested is the CI
48

involving a mix of both continuous and newly introduced discrete variables. In general, this CI
49

relationship is inconsistent with the one among the underlying continuous variables.
50

As illustrated in Fig. 1, we show different data-generative processes using causal graphical models
51

[24] in the presence of discretization. A gray node indicates an observable variable, while a white
52

node indicates a latent variable. Variables denoted by Xj (without a tilde ∼) represent continuous
53

variables, which may not be observed; while variables denoted by ˜Xj represent observed discretized
54

variables derived from Xj due to discretization. In Fig. 1(a), X2 is latent, and only its discrete
55

counterpart ˜X2 is observed. In this case, rather than observing X1, X2, and X3, we only observe
56

X1, ˜X2, and X3. Existing CI methods use these observations to test whether X1 ⊥X3 | {X2}, but
57

what is actually being tested is whether X1 ⊥X3 | { ˜X2}. In fact, according to the causal Markov
58

condition [30], , it can be inferred from Fig. 1(a) that X1 ⊥X3 | {X2} and X1 ̸⊥X3 | { ˜X2}.
59

This mismatch leads to existing CI methods, that employ observations to check the CI relationships
60

between X1 and X3 given X2, to reach incorrect conclusions. Due to the same reason, checking the
61

CI also fails in Fig 1(b) and Fig 1(c).
62

In this paper, we design a CI test specifically for handling the presence of discretization. An appropri-
63

ate test statistic for the CI of latent continuous variables, based solely on discretized observations, is
64

derived. The key is to build connections between the discretized observations and the parameters
65

needed for testing the CI of the latent continuous variables. To achieve this, we first develop bridge
66

equations that allow us to estimate the covariance of the underlying continuous variables with dis-
67

cretized observations. Then, we leverage a node-wise regression [5] to derive appropriate test statistics
68

for CI relationships from the estimated covariance. By assuming that the continuous variables follow
69

a Gaussian distribution, we can derive the asymptotic distributions of the test statistics under the null
70

hypothesis of CI. The major contributions of our paper include that
71

• We develop a CI test for ensuring accurate analysis in scenarios where data has been discretized,
72

which are common due to limitations in data collection or measurement techniques, such as in
73

financial analysis and healthcare.
74

• Our CI test can handle various scenarios including 1). Both variables Xj1 and Xj2 are discretized
75

2). Both variables Xj1 and Xj2 are continuous. 3). One of the variables Xj1 or Xj2 is discretized.
76

• We compare our test with the existing methods on both synthetic and real-world datasets, confirm-
77

ing that our method can effectively estimate the CI of the underlying continuous variables and
78

outperform the existing tests applied on the discretized observations.
79

2
DCT: A CI Test in the Presence of Discretization
80

Problem Setting
Consider a set of independent and identically distributed (i.i.d.) p-dimensional
81

random vectors, denoted as ˜
X = (X1, X2, . . . , ˜Xj, . . . , ˜Xp)T . In this set, some variables, indicated
82

by a tilde (∼), such as ˜Xj, follow a discrete distribution. For each such variable, there exists a
83

corresponding latent Gaussian random variable Xj. The transformation from Xj to ˜Xj is governed
84

by an unknown monotone nonlinear function gj. This function, gj : X →˜
X, maps the continuous
85

domain of Xj onto the discrete domain of ˜
Xj, such that ˜Xj = gj(Xj) for each observation. Given n
86

observations {˜x1, ˜x2, . . . , ˜xn} randomly sampled from ˜
X, specifically, for each variable Xj, there
87

2


---Page Break---
exists a constant vector d = (d1, . . . , dM) characterized by strictly increasing elements such that
88

˜xi
j =






1
0 < gj(xi
j) < d1
m
dm−1 < gj(xi
j) < dm
M
gj(xi
j) > dm
(1)

This model is also known as the nonparanormal model [20]. The cardinality of the domain after
89

discretization is at least 2 and smaller than infinity. Our goal is to assess both conditional and
90

unconditional independence among the variables of the vector X = (X1, X2, . . . , Xj, . . . , Xp)T .
91

In our model, we assume X ∼N(0, Σ), Σ only contain 1 among its diagonal, i.e., σjj = 1 for all
92

j ∈[1, . . . , p]. One should note this assumption is without loss of generality. We provide a detailed
93

discussion of our assumption in App. A.8.
94

Preliminary Framework of DCT
To develop an independence test, one needs to design a test
95

statistic that can reflect the dependence relation and be calculated from observations. Next, it is
96

essential to derive the underlying distribution of this statistic under the null hypothesis that the tested
97

variables are conditionally (or unconditionally) independent. By calculating the value of the test
98

statistic from observations and determining if this statistic is likely to be sampled from the derived
99

distribution (i.e., calculating the p-value and comparing it with the significance level α), we can
100

decide if the null hypothesis should be rejected.
101

Our objective is to deduce the independence and CI relationships within the original multivariate
102

Gaussian model, based on its discretized observations. In the context of a multivariate Gaussian
103

model, this challenge is directly equivalent to constructing statistical inferences for its covariance
104

matrix Σ = (σj1,j2) and its precision matrix Ω= (ωj,k) = Σ−1 [3]. The covariance matrix Σ
105

captures the pairwise covariances between variables, while the precision matrix Ω(also known as the
106

concentration matrix) provides information about the CI between variables. Specifically, the entry
107

ωj,k in the precision matrix is related to the partial correlation coefficient between variables Xj and
108

Xk, which can be used to test whether these variables are conditionally independent given some other
109

variables. Technically, we are interested in two things: (1) the calculation of the covariance ˆσj1,j2
110

and the precision coefficient (or the partial correlation coefficient) ˆωj,k, serving as the estimation
111

of σj1,j2 and ωj,k respectively (in this paper, a variable with a hat indicates its estimation); and
112

(2) the derivation of the distribution of ˆσj1,j2 −σj1,j2 and ˆωj,k −ωj,k under the null hypothesis of
113

independence and CI.
114

In the subsequent section, 1). we first introduce bridge equations to address the estimation challenge
115

of the covariance σj1,j2; 2). we proceed to derive the distribution of ˆσj1,j2 −σj1,j2, demonstrating it
116

is asymptotically normal; 3). utilizing nodewise regression, we establish the relationship between
117

the covariance matrix Σ and the precision matrix Ω, where the regression parameter βj,k acts as an
118

effective surrogate for ωj,k. Leveraging the distribution of ˆσj1,j2 −σj1,j2, we further illustrate that
119

ˆβj,k −βj,k is also asymptotically normal.
120

2.1
Design Bridge Equation for Test Statistics
121

Estimating Covariance with Bridge Equations
The bridge equation establishes a connection
122

between the underlying covariance σj1,j2 of two continuous variables Xj1 and Xj2 with the ob-
123

servations. When in the presence of discretization, the discrete transformations make the sample
124

covariance matrix based on ˜
X inconsistent with the covariance matrix of X. To obtain the estimation
125

ˆσj1,j2 of σj1,j2, the bridge equation is leveraged. In general, its form is as follows.
126

ˆτj1,j2 = T(σj1,j2; ˆΛ),
(2)

where σj1,j2 is the covariance needed to be estimated, ˆτj1,j2 is a statistic that can also be estimated
127

from observations, and ˆΛ is a set of additional parameters required by the function T(·). The specific
128

form of the function T(·) will be derived later. Both ˆτj1,j2 and ˆΛ should be able to be calculated
129

purely relying on observations. Then, given the calculated ˆτj1,j2 and ˆΛ, ˆσj1,j2 can be obtained by
130

solving the bridge equation ˆτj1,j2 = T(σj1,j2; ˆΛ). As a result, the covariance matrix Σ of X can be
131

estimated, which contains information about both unconditional independence and CI (which can be
132

derived from its inverse).
133

To estimate the covariance of a latent multivariate Gaussian distribution, we need to design appropriate
134

ˆτj1,j2, ˆΛ, and T(·). Notably, bridge equations have to be designed to handle all three possible cases:
135

3


---Page Break---
C1. both observed variables are discretized; C2. one variable is continuous while the other is
136

discretized; and C3. both variables remain continuous. We will show that cases C1 and C2 can be
137

merged into a single form of bridge equation with different parameters and a binarization operation
138

applied to the observations. Our bridge equations are presented in Def. 2.2, Def. 2.3, and Def. 2.4.
139

Bridge Equations for Discretized and Mixed Pairs
Let us first address the challenging cases
140

where both observed variables are discretized or where one variable is continuous while the other
141

is discretized. In general, different bridge equations would need to be designed to handle each case
142

individually. However, in our analysis, we provide a unified bridge equation that is applicable to both
143

cases. This is achieved by binarizing the observed variables, thereby unifying both cases into a binary
144

case. As some information may be lost in the binarization process, this unification may require more
145

examples compared to using tailored bridge functions for each specific case. Developing specific
146

bridge equations for each case to improve sample efficiency is left in future work.
147

Intuitionally, for the original continuous variable Xj, binarization separates it into two parts based on
148

a boundary hj: the part for Xj larger than hj and the part for Xj smaller than hj. In this case, we can
149

estimate the boundary by calculating the proportion of Xj that exceeds the boundary. In the scenario
150

of two variables where the threshold hj1 and hj2 divide the space into four regions, the proportions of
151

these areas are influenced by the covariance σj1,j2, which connects the relation between the binarized
152

variables with the latent covariance. This approach allows us to initially estimate the threshold hj1,
153

hj2 of a pair of variables, followed by estimating the covariance σj1,j2.
154

Let PnZ denote the average of a random variable Z given n i.i.d. observation of Z and E[Z] as the
155

true mean of Z, P as the probability and ˆP as the empirical probability. We then define the boundary
156

hj as follows: for any single discretized variable ˜Xj, there exists a constant cj such that:
157

1{˜xi
j > E[ ˜Xj]} = 1{gj(xi
j) > cj} = 1{xi
j > hj},

where hj = g−1
j (cj). Specifically, hj is the boundary in the original continuous domain to determine
158

if the discretized observation ˜Xk is larger than its mean. When the continuous variable Xj follows
159

a normal distribution, there is a relation P( ˜Xj > E[ ˜Xj]) = 1 −Φ(hj), where Φ is the cumulative
160

distribution function (cdf) of a standard normal distribution. We then provide the following definition:
161

Definition 2.1. The estimated boundary can be expressed as ˆhj = Φ−1(1 −ˆτj), where ˆτj =
162
Pn
i=1 1{˜xi
j>Pn ˜
Xj}/n, serving as the estimation of P( ˜Xj > E[ ˜Xj]).
163

Let ¯Φ(z1, z2; ρ) = P(Z1 > z1, Z2 > z2), where (Z1, Z2)T follows a bivariate normal distribution
164

with mean zero, variance one and covariance ρ. We define
165

τj1,j2 = P(˜xi
j1 > E[ ˜Xj1], ˜xi
j2 > E[ ˜Xj2]) = ¯Φ(hj1, hj2; σj1,j2).
(3)

That is, the proportion of discretized variables larger than their mean can be expressed as a function
166

of underlying covariance. This equation serves as the key of estimating latent covariance based on the
167

discretized observations. Specifically, we can substitute those true parameters with their estimation
168

and construct the bridge equation to get the estimated covariance:
169

Definition 2.2 (Bridge Equation for A Discretized-Variable Pair). For discretized variables ˜Xj1 and
170

˜Xj2, the bridge equation is defined as:
171

ˆτj1,j2 = ˆP( ˜Xj1 > Pn ˜Xj1, ˜Xj2 > Pn ˜Xj2) = 1

n

n
X

i=1
1{˜xi
j1>Pn ˜
Xj1,˜xi
j2>Pn ˜
Xj2} = T(σj1,j2; {ˆhj1, ˆhj2}),

and the function T(σj1,j2; {ˆhj1, ˆhj2}) := ¯Φ(ˆhj1, ˆhj2; σ) =
Z

x1>ˆhj1

Z

x2>ˆhj2
ϕ(xj1, xj2; σ)dxj1dxj2,

where ϕ is the probability density function of a bivariate normal distribution, ˆhj1, ˆhj2 can be simply
172

calculated using Def. 2.1.
173

Following the same intuition, we can directly apply the same bridge equation to estimate the co-
174

variance of mixed pairs. The only difference is there is no need to estimate the boundary ˆhj for the
175

continuous variable. Instead, we can incorporate its true mean of zero into the equation.
176

4


---Page Break---
Definition 2.3 (Bridge Equation for A Continuous-Discretized-Variable Pair). For one continuous
177

variable Xj1 and one discretized variable ˜Xj2, the bridge function is defined as follows:
178

ˆτj1,j2 = ˆP(Xj1 > 0, ˜Xj2 > Pn ˜Xj2) = 1

n

n
X

i=1
1{xi
j1>0,˜xi
j2>Pn ˜
Xj2} = T(σj1,j2; {0, ˆhj2}),

and the function T(·) has the same form of Def. 2.2.
179

A Bridge Equation for A Continuous-Variable Pair
When there is no discretized transformation,
180

the sample covariance of Xj1 and Xj2 provides a consistent estimation. In this context, the function
181

T acts merely as an identity mapping.
182

Definition 2.4 (A Bridge Equation for A Continuous-Variable Pair). For two continuous variables
183

Xj1 and Xj2 , the bridge equation is defined as:
184

ˆτj1,j2 := ˆσj1,j2 = 1

n

n
X

i=1
xi
j1xi
j2 −1

n

n
X

i=1
xi
j1
1
n

n
X

i=1
xi
j2 = T(σj1,j2; ∅).

For two continuous variables Xj1 and Xj2, the analytic solution of the estimated covariance can be
185

simply obtained using Def. 2.4.
186

Calculation of Estimated Covariance
For the continuous case, the analytic solution of ˆσj1,j2
187

can be simply obtained using Def. 2.4. For the cases involving the discretized variable as proposed
188

in Def. 2.2 and Def. 2.3, we can rely on the property that variance Σ only contains 1 among the
189

diagonal, which implies the covariance σj1,j2 should vary from −1 to 1. Thus, we can calculate the
190

estimated covariance by solving the objective
191

min
σj1,j2
||ˆτj1,j2 −T(σj1,j2; {ˆhj1, ˆhj2})||2
s.t. −1 < σj1,j2 < 1.
(4)

The ˆτj1,j2 is a one-to-one mapping with calculated ˆσj1,j2, ˆhj1 and ˆhj2, which is proved in App. A.2
192

2.2
Unconditional Independence Test
193

The estimation of covariance ˆσj1,j2 can be effectively solved using the designed bridge equation.
194

Now, we focus on deriving the distribution of ˆσj1,j2 −σj1,j2. These results is used as an unconditional
195

independence test in the presence of the discretization. Moreover, Thm. 2.5, Lem. 2.6, Lem. 2.7
196

and Lem. 2.8 will be leveraged in the derivation process of the CI test in Section 2.3. The detailed
197

derivation steps for both unconditional test and CI test are relatively intricate, therefore, we will
198

provide a general intuition. For a complete derivation, please refer to the App. A.3.
199

Assume we are interested in the true parameter θ0. We denote ˆθ as its estimation which is close to θ0,
200

and f(θ) is a continuous function. By leveraging Taylor expansion, we have
201

f(ˆθ) = f(θ0) + f ′(θ0)(ˆθ −θ0),
(5)

which directly constructs the relationship between the estimated parameter with the true one. Re-
202

arrange the term, we get ˆθ −θ0 = (f(ˆθ) −f(θ0))/f ′(θ0). If the denominator is a constant and the
203

numerator can be expressed as a sum of i.i.d samples, we can see ˆθ −θ0 will be asymptotically
204

normal according to the central limit theorem [35].
205

Let ψˆθ = [f 1
ˆθ (·), f 2
ˆθ (·), f 3
ˆθ (·)]T contains a group of functions parameterized by ˆθ (For discretized
206

pairs, ˆθ = (ˆσj1,j2, ˆhj1, ˆhj2)). Define Pnψˆθ as sample mean of these functions evaluated at n sample
207

points. Similarly, Pnψˆθψˆθ
T is defined as sample mean of the outer product ψˆθψˆθ
T . The notation
208

Pψˆθ := EPnψˆθ denotes the expectations of the functions in ψˆθ. Furthermore, let ψˆθ
′ denote the
209

derivative of the functions contained in ψˆθ. We now provide the main result of derived distribution
210

ˆσj1,j2 −σj1,j2 under the hull hypothesis that test pairs are independent.
211

Theorem 2.5 (Independence Test). In our settings, under the null hypothesis that two observed
212

variables indexed with j1 and j2 are statistically independent under our framework, i.e., σj1,j2 = 0,
213

the independence can be tested using the statistic
214

ˆσj1,j2 = T −1(ˆτj1,j2; ˆθ).

5


---Page Break---
This statistic is approximated to follow a normal distribution, as detailed below:
215

ˆσj1,j2
approx
∼
N

0, 1

n((Pnψ′
ˆθ)−1PnψˆθψT
ˆθ (Pnψ′T
ˆθ )−1)1,1


,
(6)

where the specific form of ψˆθ are presented in Lem. 2.6,Lem. 2.7 and Lem. 2.8.
216

We now provide the specific forms of ψˆθ. Since the variables being tested for independence can be
217

both discretized, only one being discretized, or neither being discretized. This results in different
218

forms of ψˆθ consequently differs across these scenarios.
Let Zj1 and Zj2 be any two random
219

variables indexed by j1 and j2. Let ˆσi
j1,j2 = zi
j1 · zi
j2 −PnZj1 · PnZj2 denote the sample covariance
220

based on a i-th pairwise observation of the variables Zj1 and Zj2. Let ˆτ i
j1 = 1{zi
j1>PnZj1} and
221

ˆτ i
j2 = 1{Zi
j2>PnZj2}, each calculated based on i-th observations of the variables Zj1 and Zj2,
222

respectively. Let ˆτ i
j1,j2 be ˆτ i
j1 · ˆτ i
j2. We further denote ¯Φ(·) = 1 −Φ(·). The different forms of ψˆθ
223

that arise in different cases are defined as follows:
224

Lemma 2.6. (ψˆθ for A Continuous-Variable Pair). For two continuous variables Xj1 and Xj2,
225

ψˆθ := ˆσi
j1,j2 −ˆσj1,j2.
(7)

Lemma 2.7 (ψˆθ for A Discretized-Variable Pair). For discretized variables ˜Xj1 and ˜Xj2,
226

ψˆθ :=





ˆτ i
j1,j2 −T(ˆσj1,j2; {ˆhj1, ˆhj2})
ˆτ i
j1 −¯Φ(ˆhj1)
ˆτ i
j2 −¯Φ(ˆhj2)




.
(8)

Lemma 2.8 (ψˆθ for A Continuous-Discretized-Variable Pair). For one discretized variable ˜Xj2 and
227

one continuous variable Xj1,
228

ψˆθ :=

 
ˆτ i
j1,j2 −T(ˆσj1,j2; {0, ˆhj2)}
ˆτ i
j1 −¯Φ(ˆhj2)

!

.
(9)

Derivation of forms of ψˆθ for different cases and their corresponding distribution defined in Eq (6)
229

can be found in App. A.4, App. A.5, App. A.6. Up to this point, our discussion has been confined to
230

the case of covariance σj1,j2, the indicator of unconditional independence. In the next section, we
231

will present the results of our CI test.
232

2.3
Conditional Independence (CI) Test
233

To construct a CI test of our model, we are interested at two things: calculation of the estimated
234

precision coefficient ˆωj,k and the derivation of the corresponding distribution ˆωj,k −ωj,k. In the
235

following, we first build βj,k, which is obtained using nodewise regression and show it serves as a
236

surrogate of testing for ωj,k = 0, we then construct the formulation of ˆβj,k −βj,k as the combination
237

of formulation of ˆσj1,j2 −σj1,j2 and show it will also be asymptotically normal.
238

Nodewise Regression for CI
To utilize covariance for testing CI, it is necessary to establish a
239

relationship between the estimated covariance and a metric capable of reflecting CI. To achieve this,
240

we employ the nodewise regression which effectively builds the connection between covariance
241

and precision matrix. Suppose we can access observations {x1, x2, . . . , xn} from latent continuous
242

variables X = (X1, . . . , Xp) ∼N(0, Σ), nodewise regression will do regression on every dimension
243

with all other dimensions as predictors.
244

xi
j1 =
X

j1̸=j2
xi
j2βj + ϵi
j1.
(10)

It can be shown that there are deterministic relationships between the regression coefficients and the
245

covariance and precision matrices of X, as illustrated below and proved in App. A.7.1.
246

βj = Σ−1
−j−jΣ−jj ∈Rp−1,
βj,k = −ωj,k

ωj,j
,
j ̸= k,
(11)

where Σ−j−j is the submatrix of Σ without jth column and jth row, and the Σ−jj is the vector of jth
247

column without jth row. βj,k ∈R is the surrogate of ωj,k to capture the independence relationship of
248

Xj with Xk conditioning on other variables. We can use Def. 2.2, Def. 2.3 and Def. 2.4 to get the
249

estimation ˆΣ−j−j and ˆΣ−jj and thus get the estimation ˆβj.
250

6


---Page Break---
Statistical Inference for βj,k
Nodewise regression offers a robust solution for the estimation
251

problem. A pertinent inquiry pertains to the construction of the distribution of ˆβj −βj. It is crucial
252

to recognize that the distribution of ˆσj1,j2 −σj1,j2 is already established. Therefore, if we can
253

conceptualize ˆβj −βj as a linear combination of ˆσj1,j2 −σj1,j2, the problem is directly solved, i.e.,
254

the ˆβj −βj is linear combination of dependent Gaussian variables. The underlying relationship
255

between these variables is as follows:
256

ˆβj −βj = −ˆΣ−1
−j−j

( ˆΣ−j−j −Σ−j−j)βj −( ˆΣ−jj −Σ−jj)

.

The derivation is provided in App. A.7.2. For ease of notation, we further express the distribution of
257

the difference between the estimated covariance and the true covariance as
258

ˆσj1,j2 −σj1,j2 = 1

n

n
X

i=1
ξi
j1,j2.
(12)

The specific form of ξi
j1,j2 is given in App. A.4, A.5, A.6 respectively for different cases. For
259

notational convenience, we express ˆΣ−j−j −Σ−j−j =
1
n
Pn
i=1 Ξi
−j,−j and ˆΣ−jj −Σ−jj =
260

1
n
Pn
i=1 Ξi
−j,j, where ξj1,j2 is the element of the matrix Ξ at the position indexed by (j1, j2). We
261

now propose the statistic and its asymptotic distribution for the CI test in the following theorem.
262

Theorem 2.9 (Conditional Independence test). In our settings, under the null hypothesis that Xj and
263

Xk are conditional statistically independent given a set of variables Z, i.e., βj,k = 0, the statistic
264

ˆβj,k = ( ˆΣ−1
−j−j ˆΣ−jj)[k],
(13)

where [k] denotes the element corresponding to the variable Xk in ˆΣ−1
−j−j ˆΣ−jj. The statistic ˆβj,k
265

has the asymptotic distribution:
266

ˆβj,k ∼N(0, a[k]T 1

n2

n
X

i=1
vec(Bi
−j)vec(Bi
−j)T )a[k]),

267

where Bi =
Ξi
−j,−j
Ξi
−j,j


,
a[k]
l
=









ˆΣ−1
−j−j


[k],l ,
for l ∈{1, . . . , p −1}
Pn
q=1

ˆΣ−1
−j−j


[k],l


˜βj


q ,
for l ∈{p, . . . , p2 −p}

and ˜βj is βj whose βj,k = 0.
268

In practice, we can plug in the estimation of regression parameter ˆβj and set ˆβj,k = 0 as the
269

substitution of ˜βj to calculate the variance and do the CI test. Specifically, we can obtain the ˆβj,k
270

using Eq. (13) where the estimated covariance terms can be calculated by solving the bridge equation
271

Eq. 2. Under the null hypothesis that βj,k = 0 (conditional independence), we can take the calculated
272

ˆβj,k into the distribution defined in Thm. 2.9 and obtain the p-value. If the p-value is smaller than the
273

predefined significance level α (normally set at 0.05), we will infer the tested pairs are conditionally
274

dependent; otherwise, we do not. The detailed derivation of the Thm. 2.9 can be found in App. A.7.2.
275

3
Experiments
276

We applied the proposed method DCT to synthetic data to evaluate its practical performance and
277

compare it with Fisher-Z test [14] (for all three data types) and Chi-Square test [15] (for discrete data
278

only) as baselines. Specifically, we investigated its Type I and Type II error and its application in
279

causal discovery. The experiments investigating its robustness, performance in denser graphs and
280

effectiveness in a real-world dataset can be found in App. C.
281

3.1
On the Effect of the Cardinality of Conditioning Set and the Sample Size
282

Our experiment investigates the variations in Type I and Type II error (1 minus power) probabilities
283

under two conditions. In the first scenario, we focus on the effects of modifying the sample size,
284

denoted as n = (100, 500, 1000, 2000), while conditioning on a single variable. In the second, the
285

sample size is held constant at 2000, and we vary the cardinality of the conditioning set, represented
286

7


---Page Break---
Continuous

Mixed

Discrete

(a) Type I and Type II error for D=1, n=(100,500,1000,2000)
(b) Type I and Type II error for D=(1,2,3,4,5)  n=2000

𝛼= 0.05

Figure 2: Comparison of results of Type I and Type II error (1 −power) for all three types of tested
data (continuous, mixed, discrete) and different number of samples and cardinality of conditioning set.
The suffix attached to a test’s name denotes the cardinality of discretization; for example, "Fsherz_4"
signifies the application of the Fsher-z test to data discretized into four levels. Chi-square test is only
applicable for the discrete case.

as D = (1, 2, . . . , 5). It is assumed that every variable within this conditioning set is effective, i.e.,
287

they influence the CI of the tested pairs. We repeat each test 1500 times.
288

We use Y, W to denote the variables being tested and use Z to denote the variables being conditioned
289

on. The discretized versions of the variables are denoted with a tilde symbol (e.g., ˜Z). For both con-
290

ditions, we evaluate three distinct types of observations of tested variables: continuous observations
291

for both variables (Y, W), discrete observations for both variables ( ˜Y , ˜W) and a mixed type ( ˜Y , W).
292

The variables in the conditioning set will always be discretized observations ( ˜Z).
293

To see how well the derived asymptotic null distribution approximates the true one, we verify if
294

the probability of Type I error aligns with the significance level α preset in advance. We generate
295

true continuous multivariate Gaussian data Y, W from Zi (single i = 1 for the first scenario, and
296

summed over n for the second), structured as aiZi + E and Pn
i=1 aiZi + E, where ai is sampled
297

from U(0.5, 1.5) and E follows a standard normal distribution, independent of all other variables.
298

This ensures Y ⊥⊥W|Z. The data are then discretized into K = (2, 4, 8, 12) levels, with boundaries
299

randomly set based on the variable range. The first column in Fig. 2 (a) (b) shows the resulting
300

probability of Type I errors at the significance level α = 0.05 compared with other methods.
301

A good test should have as small a probability of Type II error as possible, i.e., a larger power. To
302

test the power of our DCT, we generate the continuous multivariate Gaussian data Zi from Y, W;
303

constructed as Zi = aiY + biW + E, where ai, bi are sampled from U(0.5, 1.5) and E follows a
304

standard normal distribution independent with all others, i.e., Y ̸⊥⊥W|Z. The same discretization
305

approach is applied here. The second column in Fig. 2 (a) (b) shows the Type II error with the
306

changing number of samples and cardinality of the conditioning set compared with other methods.
307

From Fig. 2 (a), we note that the Type I error rates with our derived null distribution are well-
308

approximated at 0.05 across all three data types in both scenarios. In contrast, other testing methods
309

show significantly higher Type I error rates, increasing with the number of samples and the size of
310

the conditioning set. This indicates that such methods are more prone to erroneously concluding
311

that tested variables are conditionally dependent. Additionally, while alternative tests demonstrate
312

considerable power with smaller sample sizes, our approach requires a sample size of 2000 to achieve
313

satisfactory power, particularly in mixed and continuous cases. A possible explanation for this
314

phenomenon is that our method binarizes discretized data, which may not effectively utilize all
315

observations. This aspect warrants further investigation in future research. Moreover, our test shows
316

remarkable stability in response to changes in the number of conditioning sets.
317

8


---Page Break---
(a) fixed nodes p = 8, changing sample size n = (500, 1000, 5000, 1000)

(b) fixed sample size n = 5000, changing node p = (4, 6, 8, 10)

Figure 3: Experiment result of skeleton discovery on synthetic data for changing sample size (a) and
changing number of nodes (b). Fisherz_nodis is the Fisher-z test applied to original continuous data.
We evaluate F1 (↑), Precision (↑), Recall (↑) and SHD (↓).

3.2
Application in Causal Discovery
318

Causal discovery aims to recover the true causal structure from the data. Constraint-based causal
319

discovery methods like the PC algorithm [30] rely on testing CI from observations to discover causal
320

graphs. However, in the presence of discretization, failures in testing CI leads to false conclusions
321

about causal graphs. To evaluate the efficacy of the DCT, we construct causal graphs utilizing the
322

Bipartite Pairing (BP) model as detailed in [2], with the number of edges being one fewer than
323

the number of nodes. The detailed generation process is provided in App. B due to limited space.
324

Our experiment is divided into two scenarios: (a) fixed data samples n = 5000, with changing
325

number of nodes p = (4, 6, 8, 10); (b) fixed number of nodes p = 8 and changing sample sizes
326

n = (500, 1000, 5000, 10000).
327

Comparative analysis is conducted using the PC algorithm integrated with different testing methods.
328

Specifically, we compare DCT against the Fisher-Z test applied to discretized data, the chi-square
329

test, and the Fisher-Z test on original continuous data, the latter serving as a theoretical upper bound
330

for comparison. Since the PC algorithm can only return a completed partially directed acyclic graph
331

(CPDAG), we use the same orientation rules [11] implemented by Causal-DAG [6] to convert a
332

CPDAG into a DAG. We evaluate both the undirected skeleton and the directed graph using criteria
333

such as structural Hamming distance (SHD), F1 score, precision, and recall. For each setting, we
334

run 10 graph instances with different seeds and report the mean and standard deviation of skeleton
335

discovery in Fig. 3, and DAG in Fig. 4 in App B.
336

According to the result, DCT exhibits performance nearly on par with the theoretical upper bound
337

across metrics such as F1 score, precision, and Structural Hamming Distance (SHD) when the number
338

of variables (p) is small and the sample size (n) is large. Despite a decline in performance as the
339

number of variables increases with a smaller sample size, DCT significantly outperforms both the
340

Fisher-Z test and the Chi-square test. Notably, in almost all settings, the recall of DCT is lower than
341

that of the baseline tests, which is a reasonable outcome since these tests tend to infer conditional
342

dependencies, thereby retaining all edges given the discretized observations. For instance, a fully
343

connected graph, would achieve a recall of 1.
344

4
Conclusion
345

In this paper, we present a new testing method tailored for scenarios commonly encountered in
346

real-world applications, where variables, though inherently continuous, are only observable in their
347

discretized forms. Our method distinguishes itself from existing CI tests by effectively mitigating the
348

misjudgment introduced by discretization and accurately recovering the CI relationships of latent
349

continuous variables. We substantiate our approach with theoretical results and empirical validation,
350

underscoring the effectiveness of our testing methods.
351

9


---Page Break---
References
352

[1] Constantin F Aliferis, Alexander Statnikov, Ioannis Tsamardinos, Subramani Mani, and Xenofon D
353

Koutsoukos. Local causal and markov blanket induction for causal discovery and feature selection for
354

classification part i: algorithms and empirical evaluation. Journal of Machine Learning Research, 11(1),
355

2010.
356

[2] Armen S Asratian, Tristan MJ Denley, and Roland Häggkvist. Bipartite graphs and their applications,
357

volume 131. Cambridge university press, 1998.
358

[3] Kunihiro Baba, Ritei Shibata, and Masaaki Sibuya. Partial correlation and conditional correlation as
359

measures of conditional independence. Australian & New Zealand Journal of Statistics, 46(4):657–664,
360

2004.
361

[4] Kunihiro Baba, Ritei Shibata, and Masaaki Sibuya. Partial correlation and conditional correlation as
362

measures of conditional independence. Australian & New Zealand Journal of Statistics, 46(4):657–664,
363

2004.
364

[5] Laurent Callot, Mehmet Caner, Esra Ulasan, and A. Özlem Önder. A nodewise regression approach to
365

estimating large portfolios, 2019.
366

[6] Chandler Squires. causaldag: creation, manipulation, and learning of causal models, 2018.
367

[7] Hu Changsheng and Wang Yongfeng. Investor sentiment and assets valuation. Systems Engineering
368

Procedia, 3:166–171, 2012.
369

[8] Aswath Damodaran. Investment valuation: Tools and techniques for determining the value of any asset,
370

volume 666. John Wiley & Sons, 2012.
371

[9] A Philip Dawid. Conditional independence in statistical theory. Journal of the Royal Statistical Society
372

Series B: Statistical Methodology, 41(1):1–15, 1979.
373

[10] Simon Dooms, Toon De Pessemier, and Luc Martens. Movietweetings: a movie rating dataset collected
374

from twitter. In Workshop on Crowdsourcing and human computation for recommender systems, CrowdRec
375

at RecSys, volume 2013, page 43, 2013.
376

[11] Dorit Dor and Michael Tarsi. A simple algorithm to construct a consistent extension of a partially oriented
377

graph. 1992.
378

[12] Gary Doran, Krikamol Muandet, Kun Zhang, and Bernhard Schölkopf. A permutation-based kernel
379

conditional independence test. In UAI, pages 132–141, 2014.
380

[13] Jianqing Fan, Han Liu, Yang Ning, and Hui Zou. High dimensional semiparametric latent graphical model
381

for mixed data. Journal of the Royal Statistical Society Series B: Statistical Methodology, 79(2):405–421,
382

2017.
383

[14] Ronald Aylmer Fisher. On the "Probable Error" of a Coefficient of Correlation Deduced from a Small
384

Sample. Metron, 1:3–32, 1921.
385

[15] Karl Pearson F.R.S. X. on the criterion that a given system of deviations from the probable in the case of
386

a correlated system of variables is such that it can be reasonably supposed to have arisen from random
387

sampling. Philosophical Magazine Series 1, 50:157–175, 2009.
388

[16] Kenji Fukumizu, Francis R Bach, and Michael I Jordan. Dimensionality reduction for supervised learning
389

with reproducing kernel hilbert spaces. Journal of Machine Learning Research, 5(Jan):73–99, 2004.
390

[17] Sverre Urnes Johnson, Pål Gunnar Ulvenes, Tuva Øktedalen, and Asle Hoffart. Psychometric properties
391

of the general anxiety disorder 7-item (gad-7) scale in a heterogeneous psychiatric sample. Frontiers in
392

psychology, 10:449461, 2019.
393

[18] D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques. Adaptive
394

computation and machine learning. MIT Press, 2009.
395

[19] Loka Li, Ignavier Ng, Gongxu Luo, Biwei Huang, Guangyi Chen, Tongliang Liu, Bin Gu, and Kun Zhang.
396

Federated causal discovery from heterogeneous data, 2024.
397

[20] Han Liu, John Lafferty, and Larry Wasserman. The nonparanormal: Semiparametric estimation of high
398

dimensional undirected graphs. Journal of Machine Learning Research, 10(10), 2009.
399

10


---Page Break---
[21] Dimitris Margaritis. Distribution-free learning of bayesian network structure in continuous domains. In
400

AAAI, volume 5, pages 825–830, 2005.
401

[22] Karthik Mohan, Mike Chung, Seungyeop Han, Daniela Witten, Su-In Lee, and Maryam Fazel. Structured
402

learning of gaussian graphical models. Advances in neural information processing systems, 25, 2012.
403

[23] Sarah A Mossman, Marissa J Luft, Heidi K Schroeder, Sara T Varney, David E Fleck, Drew H Barzman,
404

Richard Gilman, Melissa P DelBello, and Jeffrey R Strawn. The generalized anxiety disorder 7-item
405

(gad-7) scale in adolescents with generalized anxiety disorder: signal detection and validation. Annals of
406

clinical psychiatry: official journal of the American Academy of Clinical Psychiatrists, 29(4):227, 2017.
407

[24] Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2000.
408

[25] Christine Peterson, Francesco C Stingo, and Marina Vannucci. Bayesian inference of multiple gaussian
409

graphical models. Journal of the American Statistical Association, 110(509):159–174, 2015.
410

[26] Zhao Ren, Tingni Sun, Cun-Hui Zhang, and Harrison H Zhou. Asymptotic normality and optimalities in
411

estimation of large gaussian graphical models. 2015.
412

[27] Rajat Sen, Ananda Theertha Suresh, Karthikeyan Shanmugam, Alexandros G Dimakis, and Sanjay
413

Shakkottai. Model-powered conditional independence test. Advances in neural information processing
414

systems, 30, 2017.
415

[28] Shohei Shimizu, Takanori Inazumi, Yasuhiro Sogawa, Aapo Hyvarinen, Yoshinobu Kawahara, Takashi
416

Washio, Patrik O. Hoyer, and Kenneth Bollen. Directlingam: A direct method for learning a linear
417

non-gaussian structural equation model, 2011.
418

[29] E Isaac Sparling and Shilad Sen. Rating: how difficult is it? In Proceedings of the fifth ACM conference on
419

Recommender systems, pages 149–156, 2011.
420

[30] P. Spirtes, C. Glymour, and R. Scheines. Causation, Prediction, and Search. MIT press, 2nd edition, 2000.
421

[31] Eric V Strobl, Kun Zhang, and Shyam Visweswaran. Approximate kernel-based conditional independence
422

tests for fast non-parametric causal discovery. Journal of Causal Inference, 7(1):20180017, 2019.
423

[32] Liangjun Su and Halbert White. A nonparametric hellinger metric test for conditional independence.
424

Econometric Theory, 24(4):829–864, 2008.
425

[33] A. W. van der Vaart. M–and Z-Estimators, page 41–84. Cambridge Series in Statistical and Probabilistic
426

Mathematics. Cambridge University Press, 1998.
427

[34] A. W. van der Vaart. Stochastic Convergence, page 5–24. Cambridge Series in Statistical and Probabilistic
428

Mathematics. Cambridge University Press, 1998.
429

[35] Aad W Van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 2000.
430

[36] Eric P Xing, Michael I Jordan, Richard M Karp, et al. Feature selection for high-dimensional genomic
431

microarray data. In Icml, volume 1, pages 601–608. Citeseer, 2001.
432

[37] Ming Yuan and Yi Lin. Model selection and estimation in the gaussian graphical model. Biometrika,
433

94(1):19–35, 2007.
434

[38] Kun Zhang, Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Kernel-based conditional indepen-
435

dence test and application in causal discovery. arXiv preprint arXiv:1202.3775, 2012.
436

[39] Yishi Zhang, Zigang Zhang, Kaijun Liu, and Gangyi Qian. An improved iamb algorithm for markov
437

blanket discovery. J. Comput., 5(11):1755–1761, 2010.
438

[40] Yujia Zheng, Biwei Huang, Wei Chen, Joseph Ramsey, Mingming Gong, Ruichu Cai, Shohei Shimizu,
439

Peter Spirtes, and Kun Zhang. Causal-learn: Causal discovery in python, 2023.
440

11


---Page Break---
Appendix for
441

442

“A Conditional Independence Test in the Presence of Discretization”
443

Appendix organization:
444

445

A Proof of Things
12
446

A.1
Proof of ˆθ
p→θ0
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12
447

A.2
Proof of one-to-one mapping between ˆτj1,j2 with ˆσj1,j2 . . . . . . . . . . . . . . .
13
448

A.3
Proof of Thm. 2.5
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13
449

A.4
Derivation of Lem. 2.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
450

A.5
Derivation of Lem. 2.8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
451

A.6
Derivation of Lem. 2.6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
452

A.7
Proof of Thm. 2.9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
453

A.7.1
Proof of Relation between Σ, Ωwith β
. . . . . . . . . . . . . . . . . . .
16
454

A.7.2
Detailed derivation of inference for βj
. . . . . . . . . . . . . . . . . . .
17
455

A.8
Discussion of assumption of zero mean and identity variance . . . . . . . . . . . .
19
456

B
Data Generation and Figure of main experiments: causal discovery
20
457

C Additional experiments
21
458

C.1
Linear non-Gaussian and nonlinear
. . . . . . . . . . . . . . . . . . . . . . . . .
21
459

C.2
Denser graph
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
460

C.3
multivariate Gaussian with nonzero mean and non-unit variance
. . . . . . . . . .
21
461

C.4
Real-world dataset
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
462

D Related Work
24
463

E
Resource Usage
25
464

F
Limiation and Broader Impacts
25
465

466

467

A
Proof of Things
468

A.1
Proof of ˆθ
p→θ0
469

Lemma A.1. For the estimation ˆθ which is calculated using bridge equation 2.4 2.2 2.3,
470

as a zero of Ψn defined in Eq. (26),(33), (36) , will converge in probability to θ0
=
471

(σj1,j2, hj1, hj2), (σj1,j2, hj2), (σj1,j2) respectively.
472

Proof We first focus on the most challenging one where both variables are discrete. According to
473

the law of large numbers, for the estimated boundary ˆhj1 and ˆhj2 whose calculations are defined as
474

12


---Page Break---
ˆhj = Φ−1(1 −ˆτj), we should have
475

n →∞,
ˆτj = 1

n

n
X

i=1
1{˜xi
j>Pn ˜
Xj}
p→P( ˜Xj > E[ ˜Xj]).
(14)

Recall the definition P( ˜Xj > E[ ˜Xj]) = 1 −Φ(hj), according to continuous mapping theorem [34],
476

as long as the function Φ−1(1 −·) is continuous, we should have ˆhj
p→hj. And thus ˆhj1
p→hj1,
477

ˆhj2
p→hj2.
478

We have ˆτj1,j2 = ¯Φ(ˆhj1, ˆhj2, ˆσj1,j2) and the estimation ˆσj1,j2 can be obtained through solving the
479

function. Similarly, we also have
480

n →∞,
ˆτj1,j2 = 1

n

n
X

i=1
1{˜xi
j1>Pn ˜
Xj1}1{˜xi
j2>Pn ˜
Xj2}
p→P(˜xi
j1 > E[ ˜Xj1], ˜xi
j2 > E[ ˜Xj2]) = τj1,j2.

(15)
Similarly, according to the continuous mapping theorem, we have ˆσj1,j2
p→σj1,j2. Thus, the
481

parameter (ˆσj1,j2, ˆhj1, ˆhj2)
p→(σj1,j2, hj1, hj2).
482

Apparently, the result above could easily extend to the mixed case where we fix ˆh1 = h1 = 0. Using
483

the same procedure, we should have (ˆσj1,j2, ˆhj2)
p→(σj1,j2, hj2).
484

For the continuous case whose estimated variance is calculated as ˆσj1,j2 =
1
n
Pn
i=1 xi
j1xi
j2 −
485

1
n
Pn
i=1 xi
j1
1
n
Pn
i=1 xi
j2., according to law of large numbers, we should have
486

n →∞,
ˆσj1,j2 = 1

n

n
X

i=1
xi
j1xi
j2−1

n

n
X

i=1
xi
j1
1
n

n
X

i=1
xi
j2
p→E(Xj1Xj2)−E(Xj1)E(Xj2) = σj1,j2.

(16)

A.2
Proof of one-to-one mapping between ˆτj1,j2 with ˆσj1,j2
487

Lemma
A.2.
For
any
fixed
ˆhj1
and
ˆhj2,
T(σj1,j2; {ˆhj1, ˆhj2})
=
488
R

x1>ˆhj1
R

x2>ˆhj2 ϕ(xj1, xj2; σ)dxj1dxj2,
is a strictly monotonically increasing function on
489

σ ∈(−1, 1).
490

Proof To prove the lemma, we just need to show the gradient ∂T (σj1,j2;{ˆhj1,ˆhj2}

∂σ
> 0 for σ ∈(−1, 1).
491

∂T(σj1,j2; {ˆhj1, ˆhj2}

∂σ
==
1

2π
p

(1 −σ2)
exp

 

−(ˆh2
j1 −2σˆhj1ˆhj2 + ˆh2
j2)
2(1 −σ2)

!

,
(17)

which is obviously positive for σ ∈(−1, 1). Thus, we have one-to-one mapping between ˆτj1j2 with
492

the calculated ˆσj1,j2 for fixed ˆhj1 and ˆhj2.
493

A.3
Proof of Thm. 2.5
494

In this section, we provide the proof of Thm. 2.5, which utilizes a regular statistical tool: Z-estimator
495

[33]. Specifically, we are interested in the parameter θ and we have it estimation ˆθ. Let x1, . . . , xn
496

are sampled from some true distribution P, we can construct the function characterized by the
497

parameter θ related the x as ψθ(x). As long as we have n observations, we can construct the function
498

as follows
499

Ψn(θ) = 1

n

n
X

i=1
ψθ(xi) = Pnψθ.
(18)

We further specify the form
500

Ψ(θ) =
Z
ψθ(x)dx = Pψθ.
(19)

Assume the estimator ˆθ is a zero of Ψn, i.e., Ψn(ˆθ) = 0 and will converge in probability to θ0, which
501

is a zero of Ψ, i.e., Ψ(θ0) = 0. Expand Ψn(ˆθ) in a Taylor series around θ0, we should have
502

0 = Ψn(ˆθ) = Ψn(θ0) + (ˆθ −θ0)Ψ′
n(θ0) + 1

2(ˆθ −θ0)Ψ′′
n(θ0).
(20)

13


---Page Break---
Rearrange the equation above, we have
503

ˆθ −θ0 = −
Ψn(θ0)

Ψ′n(θ0) + 1

2(ˆθ −θ0)Ψ′′n(θ0)

= −

1
n
Pn
i=1 ψθ(xi)

Ψ′n(θ0) + 1

2(ˆθ −θ0)Ψ′′n(θ0)
.
(21)

According to the central limit theorem, the numerator will be asymptotic normal with variance
504

Pψ2
θ0/n as the mean Ψ(θ0) = 0 is zero. The first term of denominator Ψ′
n(θ0) will converge in
505

probability to Ψ′(θ0) according to the law of large numbers. The second term ˆθ −θ0 = oP (1). 1
506

As long as the denominator converges in probability and the numerator converges in distribution,
507

according to Slusky’s lemma, we have
508

√n(ˆθ −θ0) ⇝N

 

0,
Pψ2
θ0
(Pψ′
θ0)2

!

.
(22)

Extend into the high-dimensional case we should have
509

ˆθ −θ0 = −(Ψ′
n(θ0))−1Ψn(θ0),
(23)
where the second order term is omitted, further assume the matrix Pψ′
θ0 is invertible, we have
510

√n(ˆθ −θ0) ⇝N
 
0, (Pψ′
θ0)−1Pψθ0ψT
θ0(Pψ′T
θ0 )−1
,
(24)
Specifically, in our case θ0 = (σj1,j2, Λ), where Λ is another parameter set influencing the estimation
511

of σj1,j2 (will discuss case in case in later proof). In the practical scenario, we only have access to
512

the estimated parameter ˆθ and the empirical distribution Pn, thus we have
513

ˆσj1,j2 −σj1,j2
approx
∼
N

0, ((Pnψ′
ˆθ)−1PnψˆθψT
ˆθ (Pnψ′T
ˆθ )−1)1,1

.
(25)

Under the null hypothesis of independent, σj1,j2=0. We provide the proof that ˆθ
p→θ0 of our case
514

in App. A.1. Thus, Pnψˆθ, the function parameterized by ˆθ, should also converge in Pnψˆθ0 when
515

n →∞. Besides, by the law of large numbers, Pnψˆθ0 will converge to Pψˆθ0. Thus, the equation
516

above will converge to Eq. (24) when n →∞.
517

A.4
Derivation of Lem. 2.7
518

Let’s first focus on the most challenging case where both variables are discretized observations
519

and our interested parameter will include ˆθ = (ˆσj1,j2, ˆhj1, ˆhj2) (Although we only care about the
520

distribution of ˆσj1,j2 −σj1,j2, the estimation of boundary ˆhj1and ˆhj2 will influence the estimation of
521

ˆσj1,j2, thus we need to consider all of them).
522

The next step will be to construct an appropriate criterion function ψ such that Ψn(ˆθ) = 0. Given n
523

observations {˜x1, ˜x2, . . . , ˜xn}, which are discretized version of {x1, x2, . . . , xn} we should have
524

Ψn(ˆθ) =




Ψn(ˆσj1,j2)
Ψn(ˆhj1)
Ψn(ˆhj2)



= 1

n

n
X

i=1
ψˆθ(˜xi) = 1

n

n
X

i=1





ˆτ i
j1,j2 −T(ˆσj1,j2; {ˆhj1, ˆhj2})
ˆτ i
j1 −¯Φ(ˆhj1)
ˆτ i
j2 −¯Φ(ˆhj2)




= 0. (26)

525

Ψn(θ0) =

 Ψn(σj1,j2)
Ψn(hj1)
Ψn(hj2)

!

= 1

n

n
X

i=1
ψθ0(˜xi) = 1

n

n
X

i=1




ˆτ i
j1,j2 −T(σj1,j2; {hj1, hj2})
ˆτ i
j1 −¯Φ(hj1)
ˆτ i
j2 −¯Φ(hj2)



.
(27)

The difference between the estimated parameter with the true parameter can be expressed as
526

ˆθ −θ0 =




ˆσj1,j2 −σj1,j2
ˆhj1 −hj1
ˆhj2 −hj2



= −1

n

n
X

i=1







∂Ψn(σj1,j2)

∂σj1,j2

∂Ψn(σj1,j2)

∂hj1

∂Ψn(σj1,j2)

∂hj2
∂Ψn(hj1)

∂σj1,j2

∂Ψn(hj1)

∂hj1

∂Ψn(hj1)

∂hj2
∂Ψn(hj2)

∂σj1,j2

∂Ψn(hj2)

∂hj1

∂Ψn(hj2)

∂hj2







−1

·




ˆτ i
j1,j2 −T(σj1,j2; {hj1, hj2})
ˆτ i
j1 −¯Φ(hj1)
ˆτ i
j2 −¯Φ(hj2)



,
(28)

1We will not provide proof of this in this paper; however, interested readers may refer to [33]

14


---Page Break---
where the specific form of each entry of the gradient matrix is expressed as
527

∂Ψn(σj1,j2)

∂σj1,j2
= −
1

2π
q

(1 −σ2
j1,j2)
exp

 

−(h2
j1 −2σj1,j2hj1hj2 + h2
j2)
2(1 −σ2
j1,j2)

!

;

∂Ψn(σj1,j2)

∂hj1
=
Z ∞

hj2

1

2π
q

1 −σ2
j1,j2
exp

 

−h2
j1 −2σj1,j2hj1x2 + x2
2
2(1 −σ2
j1,j2)

!

dx2;

∂Ψn(σj1,j2)

∂hj2
=
Z ∞

hj1

1

2π
q

1 −σ2
j1,j2
exp

 

−h2
2 −2σj1,j2hj2x1 + x2
1
2(1 −σ2
j1,j2)

!

dx1;

∂Ψn(hj1)

∂σj1,j2
= 0;

∂Ψn(hj1)

∂hj1
=
1
√

2π exp

 

−h2
j1
2

!

;

∂Ψn(hj1)

∂hj2
= 0;

∂Ψn(hj2)

∂σj1,j2
= 0;

∂Ψn(hj2)

∂hj1
= 0;

∂Ψn(hj2)

∂hj2
=
1
√

2π exp

 

−h2
j2
2

!

.

(29)

For simplicity of notation, we define
528

ˆσj1,j2 −σj1,j2 = 1

n

n
X

i=1
ξi
j1,j2,
(30)

where the specific form is of {ξi
j1,j2} is defined in Eq. (28). We should note that {ξi
j1,j2} are i.i.d
529

random variables with mean zero (this property will be the key to the derivation of inference of CI).
530

As long as our estimation ˆθ converge in probability to θ0 as proved in A.1, we have
531

√n(ˆθ −θ0) ⇝N
 
0, ((Pψ′
θ0)−1Pψθ0ψT
θ0(Pψ′T
θ0 )−1)1,1

,
(31)

where ψθ0 is defined in Eq. (27). However, in practice, we don’t have access to either P or θ0. In this
532

scenario, we can plug in the empirical distribution of Pnψˆθ to get the estimated variance, i.e., the
533

actual variance used in the calculation of ˆσj1,j2 −σj1,j2 is
534

1
n


(Pnψ′
ˆθ)−1PnψˆθψT
ˆθ (Pnψ′T
ˆθ )−1

1,1 .
(32)

A.5
Derivation of Lem. 2.8
535

Use the same line of procedure as in the derivation of Lem. 2.7, for mixed pair of observations where
536

Xj1 is continuous and ˜Xj2 is discrete, we can construct the criterion function
537

Ψn(ˆθ) =
Ψn(ˆσj1,j2)
Ψn(ˆhj2)


= 1

n

n
X

i=1
ψˆθ(˜xi) = 1

n

n
X

i=1

 
ˆτ i
j1,j2 −T(ˆσj1,j2; {0, ˆhj2})
ˆτ i
j2 −¯Φ(ˆhj2)

!

= 0.
(33)

538

Ψn(θ0) =

Ψn(σj1,j2)
Ψn(hj2)


= 1

n

n
X

i=1
ψθ0(˜xi) = 1

n

n
X

i=1

ˆτ i
j1,j2 −T(σj1,j2; {0, hj2})
ˆτ i
j2 −¯Φ(hj2)


.
(34)

15


---Page Break---
The difference between the estimated parameter with the true parameter can be expressed as
539

ˆθ−θ0 =
ˆσj1,j2 −σj1,j2
ˆhj2 −hj2


= −1

n

n
X

i=1





∂Ψn(σj1,j2)

∂σj1,j2

∂Ψn(σj1,j2)

∂hj2
∂Ψn(hj2)

∂σj1,j2

∂Ψn(hj2)

∂hj2





−1 ˆτ i
j1,j2 −T(σj1,j2; {0, hj2})
ˆτ i
j2 −¯Φ(hj2).


,

(35)
where the specific form of each entry of the gradient matrix can be found in Eq. (29). Using exactly
540

the same procedure, we should have the same formation of the variance calculated as Eq. (32) with a
541

different definition of ψθ0 and ψˆθ defined in Eq. (34) (33).
542

A.6
Derivation of Lem. 2.6
543

Use the same line of procedure as in derivation of Lem. 2.7, for a continuous pair of variables, we
544

can construct the criterion function
545

Ψn(ˆθ) = Ψn(ˆσj1,j2) = 1

n

n
X

i=1
xi
j1xi
j2 −1

n

n
X

i=1
xi
j1
1
n

n
X

i=1
xi
j2 −ˆσj1,j2 = 0.
(36)

546

Ψn(θ0) = Ψn(σj1,j2) = 1

n

n
X

i=1
xi
j1xi
j2 −1

n

n
X

i=1
xi
j1
1
n

n
X

i=1
xi
j2 −σj1,j2.
(37)

Denote 1

n
Pn
i=1 xi
j1 as ¯xj1 and 1

n
Pn
i=1 xi
j2 as ¯xj2. We should have
547

ˆσj1,j2 −σj1,j2 = 1

n

n
X

i=1
xi
j1xi
j2 −¯xj1 ¯xj2 −σj1,j2.
(38)

According to Eq. (22), we have
548

√n(ˆσj1,j2 −σj1,j2) ⇝N

 

0,
Pψ2
θ0
(Pψ′
θ0)2

!

.
(39)

where (Pψ′
θ0)2 = 1. In practical calculation, we have the variance
549

1
nPnψ2
ˆθ/(Pnψ′
ˆθ)2 = 1

n2

n
X

i=1
(xi
j1xi
j2 −¯xj1 ¯xj2 −ˆσj1,j2)2.
(40)

A.7
Proof of Thm. 2.9
550

A.7.1
Proof of Relation between Σ, Ωwith β
551

Consider our latent continuous variables X = (X1, . . . , Xp) ∼N(0, Σ) and do nodewise regression
552

Xj = X−jβj + ϵj.
(41)

We can divide its covariance Σ and its precision matrix Ω= Σ−1 into X and Y part in our regression:
553

Σ =

Σjj
Σj−j
Σ−jj
Σ−j−j


Ω=

Ωjj
Ωj−j
Ω−jj
Ω−j−j



.
(42)

Just like regular linear regression, we can get
554

n →∞,
βj = Σ−1
−j−jΣ−jj.
(43)

From the invertibility of a block matrix
555


A
B
C
D

−1
=

(A −BD−1C)−1
−(A −BD−1C)−1BD−1

−D−1C(A −BD−1C)−1
D−1 + D−1C(A −BD−1C)−1BD−1


.
(44)

If A and D is invertible, we will have
556


A
B
C
D

−1
=

A −BD−1C
0
0
(D −CA−1B)−1

 
I
−BD−1

−CA−1
I


.
(45)

16


---Page Break---
Thus, we can get:
557

Ωjj = Σjj −(Σj−jΣ−1
−j−jΣ−jj)−1;

Ωj−j = −
 
Σjj −(Σj−jΣ−1
−j−jΣ−jj)−1
Σj−j(Σ−j−j)−1.
(46)

Move one step forward:
558

−Ω−1
jj Ωj−j = Σj−j(Σ−j−j)−1.
(47)

Take transpose for both sides, as long as Ωis a symmetric matrix and Ω−jj = ΩT
j−j, we will have
559

−Ω−1
jj Ω−jj = Σ−1
−j−jΣ−jj = βj.
(48)

We should note testing Ω−jj = 0 is equivalent to testing βj = 0 as the Ωjj will always be nonzero.
560

The variable Ω−jj captures the CI of Xj with other variables. As long as the variable Ωjj is just one
561

scalar, we can get
562

βj,k = −ωj,k

ωj,j
(49)

capturing the independence relationship between variable Xj with Xk conditioning on all other
563

variables.
564

A.7.2
Detailed derivation of inference for βj
565

Nodewise regression allows us to use the regression parameter βj as the surrogate of Ω−jj. The
566

problem now transfers to constructing the inference for βj, specifically, the derivation of distribution
567

of ˆβj −βj. The overarching concept is that we are already aware of the distribution of ˆσj1,j2 −σj1,j2
568

and we know that there exists a deterministic relationship between βj with Σ. Consequently, we can
569

express ˆβj −βj as a composite of ˆσj1,j2 −σj1,j2 to establish such an inference. Specifically, we have
570

ˆβj −βj = ˆΣ−1
−j−j ˆΣ−jj −Σ−1
−j−jΣ−jj

= ˆΣ−1
−j−j

ˆΣ−jj −ˆΣ−j−jΣ−1
−j−jΣ−jj


= −ˆΣ−1
−j−j

ˆΣ−j−jβj −Σ−j−jβj + Σ−j−jβj −ˆΣ−jj


= −ˆΣ−1
−j−j

( ˆΣ−j−j −Σ−j−j)βj −( ˆΣ−jj −Σ−jj)

,

(50)

where each entry in matrix ( ˆΣ−j−j −Σ−j−j) and ( ˆΣ−jj −Σ−jj) denotes the difference between
571

estimated covariance with true covariance. Suppose that we want to test the CI of the variable X1
572

with other variables, j = 1, then
573

ˆΣ−j−j −Σ−j−j =





ˆσ1,1 . . . ˆσ1,j−1, ˆσ1,j+1 . . . ˆσ1,p
. . .
ˆσj−1,1 . . . ˆσj−1,j−1, ˆσj−1,j+1 . . . ˆσj−1,p
. . .
ˆσp,1 . . . ˆσp,j−1, ˆσp,j+1 . . . ˆσp,p




(51)

−





σ1,1 . . . σ1,j−1, σ1,j+1 . . . σ1,p
. . .
σj−1,1 . . . σj−1,j−1, σj−1,j+1 . . . σj−1,p
. . .
σp,1 . . . σp,j−1, σp,j+1 . . . σp,p.



.
(52)

Suppose that we want to test the CI of the variable X1 with other variables, j = 1. then
574

ˆΣ−1−1 −Σ−1−1 =

"ˆσ2,2 . . . ˆσ2,p
. . .
ˆσp,2 . . . ˆσp,p

#

−

"σ2,2 . . . σ2,p
. . .
σp,2 . . . σp,p

#

(53)

:= 1

n

n
X

i=1




ξi
2,2 . . . ξi
2,p
. . .
ξi
p,2 . . . ξi
p,p



,
(54)

17


---Page Break---
where {ξi
j1,j2} are i.i.d random variables with specific form defined in Eq. (28) for discrete case,
575

Eq. (35) for mixed case and Eq. (38) in continuous case. Put them together:
576





ˆβ1,2 −β1,2
ˆβ1,3 −β1,3
. . .
ˆβ1,p −β1,p



= −ˆΣ−1
−1−1
1
n

n
X

i=1











ξi
2,2
ξi
2,3
. . .
ξi
2,p
ξi
3,2
ξi
3,3
. . .
ξi
3,p
. . .
. . .
. . .
. . .
ξi
p,2
ξi
p,3
. . .
ξi
p,p









β1,2
β1,3
. . .
β1,p



−





ξi
2,1
ξi
3,1
. . .
ξi
p,1









.
(55)

As 1

n
Pn
i=1 ξi
j1,j2 is asymptotically normal, the who vector of ˆβ1 −β1 is a linear combination of
577

Gaussian distribution. However, We cannot merely engage in a linear combination of its variance as
578

they are dependent with each other. For example, if Y1, Y2 are dependent and we are trying to find
579

out V ar(aY1 + bY2), we should have
580

V ar(aY1 + bY2) = [a
b]

V ar(Y1)
Cov(Y1, Y2)
Cov(Y1, Y2)
V ar(Y2)

 
a
b


.
(56)

Now, suppose we are interested in the distribution of ˆβ1,2 −β1,2, we should have
581

ˆβ1,2 −β1,2 = 1

n

n
X

i=1
( ˆΣ−1
−1−1)[2],:











ξi
2,2
ξi
2,3
. . .
ξi
2,p
ξi
3,2
ξi
3,3
. . .
ξi
3,p
. . .
. . .
. . .
. . .
ξi
p,2
ξi
p,3
. . .
ξi
p,p









β1,2
β1,3
. . .
β1,p



−





ξi
2,1
ξi
3,1
. . .
ξi
p,1









,
(57)

where ( ˆΣ−1
−1−1)[2],: is the row of index of X2 of ˆΣ−1
−1−1 ([2] denotes the index of the variable). For
582

ease of notation, let
583

Ξi
−1,−1 =





ξi
2,2
ξi
2,3
. . .
ξi
2,p
ξi
3,2
ξi
3,3
. . .
ξi
3,p
. . .
. . .
. . .
. . .
ξi
p,2
ξi
p,3
. . .
ξi
p,p



,
Ξi
−1,1 =





ξi
2,1
ξi
3,1
. . .
ξi
p,1



,
(58)

and let
584

Bi
−1 =









ξi
2,1
ξi
3,1
. . .
ξi
p,1
ξi
2,2
ξi
2,3
. . .
ξi
2,p
ξi
3,2
ξi
3,3
. . .
ξi
3,p
. . .
. . .
. . .
. . .
ξi
p,2
ξi
p,3
. . .
ξi
p,p








(59)

as the concatenation of those two matrices. The variance is calculated as
585

V ar
√n(ˆβ1,2 −β1,2)

= a[2]T 1

n

n
X

i=1
vec(Bi
−1)vec(Bi
−1)T a[2],
(60)

where
586

a[2]
l
=









ˆΣ−1
−1−1


[2],l ,
for l ∈{1, . . . , p −1}
Pn
q=1

ˆΣ−1
−1−1


[2],l (β1)q ,
for l ∈{p, . . . , p2 −p}
(61)

vec(Bi
−1) is the squeezed vector form of matrix vec(Bi
−1) ∈Rp×p−1, i.e.,
587

vec(Bi
−1) =








ξi
2,1
ξi
3,1
...
ξi
p,p






.
(62)

Thus, the distribution of ˆβj,k −βj,k is
588

ˆβj,k −βj,k ∼N(0, a[k]T 1

n2

n
X

i=1
vec(Bi
−j)vec(Bi
−j)T )a[k]).
(63)

In practice, we can plug in the estimates of βj to estimate the interested distribution and do the CI
589

test by hypothesizing βj,k = 0.
590

18


---Page Break---
A.8
Discussion of assumption of zero mean and identity variance
591

In this section, we engage in a more thorough discussion regarding our assumptions about X.
592

Specifically, we demonstrate that this assumption of mean and variance does not compromise the
593

generality. In other words, the true model may possess different mean and variance values, but we
594

proceed by treating it as having a mean of zero and identity variance.
595

The key ingredient allowing us to assume such a model is, the discretization function gj is an unknown
nonlinear monotonic function. Suppose the g′
j maps the continuous domain to a binary variable, and
we have the "groundtruth" variable, denoted X′
j, with mean a and variance b. Assume the cardinality
of the discretized domain is only 2, i.e., our observation ˜Xj can only be 0 or 1. We further have the
constant d′
j as the discretization boundary such that we have the observation

˜Xj = 1(g′
j(X′
j) > d′
j) = 1(X′
j > g′−1
j
(dj))

We can always produce our assumed variable Xj with mean 0 and variance 1, such that Xj =
596

1
√

bX′
j −
a
√

b and the same observation with a different nonlinear transformation gj and decision
597

boundary dj, such that
598

˜Xj = 1(gj(Xj) > dj) = 1(Xj > g−1
j (dj)) = 1(X′
j >
√

bg−1
j (dj) + a)

As long as the observation ˜Xj is the same, we should have
√

bg−1
j (dj)+a = g′−1
j
(dj). Our assumed
599

model Xj clearly mimics the "groundtruth" X′
j. Besides, according to Lem. A.2, we have one-to-
600

one mapping between ˆτj1j2 with the estimated covariance for fixed ˆhj1, ˆhj2. Thus, as long as the
601

observation is the same, the estimation of covariance ˆσj1,j2 remains unaffected by our assumptions
602

regarding the mean and variance of X, so do the following inference.
603

We further conduct casual discovery experiments to empirically validate our statement, which is
604

shown in App. C.3.
605

19


---Page Break---
B
Data Generation and Figure of main experiments: causal discovery
606

Data Generation and Code
We construct the true DAG G using the Bipartite Pairing (BP) model
607

[2], with the number of edges being one fewer than the number of nodes. The subsequent generation of
608

true multivariate Gaussian data involves assigning causal weights drawn from a uniform distribution
609

U ∼(0.5, 2) and incorporating noise via samples from a standard normal distribution for each
610

variable. Following this, we binarize the data, setting the threshold randomly based on each variable’s
611

range. The code implementation is based on [40] .
612

(a) fixed nodes p = 8, changing sample size n = (500, 1000, 5000, 1000)

4
6
8
10
Numb of variables

0.5

1.0

F1 score (Direction)

4
6
8
10
Numb of variables

0.5

1.0

Precision (Direction)

4
6
8
10
Numb of variables

0.5

1.0

Recall (Direction)

4
6
8
10
Numb of variables

0

10

20

SHD (Direction)

DCT
Fisherz
Fisherz_nodis
Chis-q

(b) fixed sample size n = 5000, changing node p = (4, 6, 8, 10)

Figure 4: Experiment result of DAG discovery on synthetic data for changing sample size (a) and
changing number of nodes (b). Fisherz_nodis is the Fisher-z test applied to original continuous data.
We evaluate F1 (↑), Precision (↑), Recall (↑) and SHD (↓).

20


---Page Break---
C
Additional experiments
613

C.1
Linear non-Gaussian and nonlinear
614

Our model requires that the original data must adhere to the hypothesis of following a multivariate
615

normal distribution, which appears to potentially limit the generalizability. Therefore, it is worthwhile
616

to explore its robustness when such assumptions are violated. In this regard, we conducted several
617

experiments, including scenarios involving linear non-Gaussian and nonlinear Gaussian.
618

For both cases, we follow the setting of our experiment where there are p = 8 nodes and p −1
619

edges. We explore the effect of changing sample size n = (100, 500, 2000, 5000). Specifically for
620

linear non-Gaussian case, we adhere to some of the settings outlined by [28], conducting experiments
621

where the original continuous data followed: (1) a Student’s t-distribution with 3 degrees of freedom,
622

(2) a uniform distribution, and (3) an exponential distribution. Each variable is generated as Xi =
623

f(PAi) + noise, where noise follows the distribution in (1), (2), (3) correspondingly and f is a
624

linear function. The first three rows of Fig. 5 and Fig. 6 show the result of the linear non-Gaussian
625

case.
626

For the nonlinear cases, we follow setting in [19], where every variable Xi is generated as Xi =
627

f(WPAi + noise), noise ∼N(0, 1) and f is a function randomly chosen from (a) f(x) = sin(x),
628

(b) f(x) = x3, (c) f(x) = tanh(x), and (d) f(x) = ReLU(x). W is a linear function. Similarly,
629

we set the number of nodes at p = 8 and change the number of samples n = (500, 2000, 5000).
630

For both cases, we run 10 graph instances with different seeds and report the result of skeleton
631

discovery in Fig. 5 and DAG in Fig. 6 (The same orientation rules [11] used in the main experiment
632

are employed to convert a CPDAG [6] into a DAG). The last row of Fig. 5 and Fig. 6 shows the result
633

of the nonlinear case.
634

Based on the experimental outcomes, DCT demonstrates marginally superior or comparable efficacy
635

in terms of the F1-score, precision, and SHD relative to both the Fisher-Z test and the Chi-square test
636

when dealing with small sample sizes. Nevertheless, as the sample size increases, DCT’s performance
637

clearly surpasses that of the aforementioned tests across all three evaluated metrics, especially in the
638

linear case. Consistent with observations from the main experiment, DCT exhibits a lower recall in
639

comparison to the baseline tests. This discrepancy can be attributed to the baseline tests being prone
640

to incorrectly infer conditional dependence and connect a large proportion of nodes. According to
641

the results, our test shows notable robustness under the case assumptions are violated, confirming its
642

practical effectiveness.
643

C.2
Denser graph
644

DCT primarily works on cases where CI is mistakenly judged as conditional dependence due
645

to discretization. Consequently, its efficacy is more pronounced in scenarios characterized by a
646

relatively sparse graph, as numerous instances are truly conditionally independent. Nevertheless, the
647

investigation of causal discovery with a dense latent graph is essential for evaluating the power of a
648

test, i.e., its ability to successfully reject the null hypothesis when the tested pairs are conditionally
649

dependent. Thus, we conduct the experiment where p = 8, n = 10000 and changing edges (p +
650

2, p + 4, p + 6). Similarly, the latent continuous data follows a multivariate Gaussian model and
651

the true DAG G is constructed using BP model. We run 10 graph instances with different seeds and
652

report the result of the skeleton discovery and DAG in Fig. 7.
653

According to the experiment results, DCT exhibits better performance in terms of the F1-score,
654

precision, and SHD relative to both the Fisher-Z test and the Chi-square test. As the graph becomes
655

progressively denser, the superiority of the Discrete Causality Test (DCT) correspondingly diminishes
656

as there are few conditional independent cases in the true DAG. Due to the same reason, The recall
657

remains lower than that of other baseline methods.
658

C.3
multivariate Gaussian with nonzero mean and non-unit variance
659

We employed a setting nearly identical to the main experiment, with the only difference being the
660

alteration in data generation: instead of using a standard normal distribution, we used a Gaussian
661

distribution with mean sampled from U(−2, 2) and variance sampled from U(0, 3). We fix the
662

number of variables as p = 8 and change the number of samples n = (100, 500, 2000, 5000). The
663

Fig. 8 shows the result and demonstrates the effectiveness of our method.
664

21


---Page Break---
(a) Linear Exponential.

(b) Linear Student.

(c) Linear Uniform.

(d) Nonlinear Gaussian.

Figure 5:
Experiment result of causal discovery on synthetic data with p
=
8, n
=
(100, 500, 2000, 5000) where the data generation process violates our assumptions. The data are
generated with either nongaussian distributed (a), (b), (c) or the relations are not linear (d). The figure
reports F1 (↑), Precision (↑), Recall (↑) and SHD (↓) on skeleton.

C.4
Real-world dataset
665

To further validate DCT, we employ it on a real-world dataset:
Big Five Personality
666

https://openpsychometrics.org/, which includes 50 personality indicators and over 19000 data sam-
667

ples. Each variable contains 5 possible discrete values to represent the scale of the corresponding
668

questions, where 1=Disagree, 2=Weakly disagree, 3=Neutral, 4=Weakly agree and 5=Agree, e.g.,
669

"N3=1" means "I agree that I worry about things". This scenario clearly suits DCT, where the degree
670

of agreement with a certain question must be a continuous variable while we can only observe the
671

result after categorization. We choose three variables respectively: [N3: I worry about things], [N10:
672

I often feel blue ], [N4: I seldom feel blue]. We then do the casual discovery using PC algorithm with
673

DCT and compare it with the Chi-square test and Fisher-Z test. The result can be found in Fig. 9.
674

Based on the experimental outcomes, despite the absence of a groundtruth for reference, we observe
675

that the results obtained via DCT appear more plausible than those derived from Fisher-Z and Chi-
676

square tests. Specifically, DCT suggests the relationship N3 ⊥⊥N4|N10, which is reasonable as
677

intuitively, the answer of ’I often feel blue’ already captures the information of ’I seldom feel blue’.
678

22


---Page Break---
(a) Linear Exponential.

(b) Linear Student.

(c) Linear Uniform.

(d) Nonlinear Gaussian.

Figure 6:
Experiment result of causal discovery on synthetic data with p
=
8, n
=
(100, 500, 2000, 5000) where the data generation process violates our assumptions. The data are
generated with either nongaussian distributed (a), (b), (c) or the relations are not linear (d). The figure
reports F1 (↑), Precision (↑), Recall (↑) and SHD (↓) on DAG.

As a comparison, both Fisher-Z and Chi-square return a fully connected graph. The results directly
679

correspond to our illustrative example shown in Fig. 1, substantiating the necessity of our proposed
680

test.
681

23


---Page Break---
Figure 7: Experimental comparison of causal discovery on synthetic datasets for denser graphs with
p = 8, n = 10000 and edges varying p + 2, p + 4, p + 6. We evaluate F1 (↑), Precision (↑), Recall
(↑) and SHD (↓) on both skeleton and DAG.

Figure 8: Experimental comparison of causal discovery on synthetic datasets for multivariate Gaussian
model with p = 8, n = (100, 500, 2000, 5000) and where mean is not zero. We evaluate F1 (↑),
Precision (↑), Recall (↑) and SHD (↓) on both skeleton and DAG.

D
Related Work
682

Testing for CI is pivotal in the field of causal discovery [30], and a variety of methods exist for
683

performing CI tests (CI tests). An important group of CI test methods involves the assumption of
684

Gaussian variables with linear dependencies. For example, under this assumption, Gaussian graphical
685

models are extensively studied [37, 25, 22, 26]. To address CI test under Gaussian assumption, partial
686

correlation serves as a viable method for CI testing [4]. To evaluate the independence of variables
687

X1 and X2 conditional on Z, The technique proposed by [32] determines CI by comparing the
688

estimations of p(X1|X2, Z) and p(X1|X2).
689

24


---Page Break---
[N3]
I worry about things

[N10]

I often feel blue

[N4]

I seldom feel blue

(a) Fisher-Z test

[N3]
I worry about things

[N10]

I often feel blue

[N4]

I seldom feel blue

(b) Chi-square test

[N3]

I seldom feel blue

[N10]

I often feel blue
I worry about things

[N4]

(c) DCT

Figure 9: Experimental comparison of causal discovery on the real-world dataset.

Another approach involves discretizing Z and performing independent tests within each resulting bin
690

[21]. Our work, however, diverges from these existing methods in two significant ways. Firstly, we
691

are equipped to handle data, where partial variables are discretized. Additionally, we postulate that
692

discrete variables are derived from the transformation of continuous variables in a latent Gaussian
693

model. With the same assumption, the most closely related study is by [13], where the authors
694

developed a novel rank-based estimator for the precision matrix of mixed data. However, their work
695

stops short of providing a CI test for this method. Our research fills this gap, offering the ability to
696

estimate the precision matrix for both discrete and mixed data and providing a rigorous CI test for
697

our methodology.
698

Recent advancements in CI testing have utilized kernel methods for continuous variables influenced
699

by nonlinear relationships. [16] describes non-parametric CI relationships using covariance operators
700

in reproducing kernel Hilbert spaces (RKHS). KCI test [38] assesses the partial associations of
701

regression functions linking x, y, and z, while RCI test [31] aims to enhance the KCI test’s efficiency.
702

In KCIP test [12] employs permutations of samples to emulate CI scenarios. CCI test [27] further
703

reformulates testing into a process that leverages the capabilities of supervised learning models. For
704

discrete variable analysis, the G2 test [1] and conditional mutual information [39] are commonly
705

employed. However, their method cannot deal with our setting where only discretized version of
706

latent variables can be observed.
707

E
Resource Usage
708

All the experiments are run using Intel(R) Xeon(R) CPU E5-2680 v4 with 55 processors. It costs 4
709

hours to run experiments in Section 3.1.
710

F
Limiation and Broader Impacts
711

Limitation
So far, the largest limitation of our method is to treat discretized variables as binary,
712

which wastes the available information. Besides that, the parametric assumption limits its generaliz-
713

ability. However, we need to point out this is pretty normal in CI test fields.
714

Broader Impacts
The goal of our proposed method is to test the conditional independence relation-
715

ship given discretized observation. This task is essential and has broad applications. We are confident
716

that our method will be beneficial and will not result in negative societal impacts.
717

25


---Page Break---
NeurIPS Paper Checklist
718

1. Claims
719

Question: Do the main claims made in the abstract and introduction accurately reflect the
720

paper’s contributions and scope?
721

Answer: [Yes]
722

Justification: Section1 Introduction and Abstract
723

Guidelines:
724

• The answer NA means that the abstract and introduction do not include the claims
725

made in the paper.
726

• The abstract and/or introduction should clearly state the claims made, including the
727

contributions made in the paper and important assumptions and limitations. A No or
728

NA answer to this question will not be perceived well by the reviewers.
729

• The claims made should match theoretical and experimental results, and reflect how
730

much the results can be expected to generalize to other settings.
731

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
732

are not attained by the paper.
733

2. Limitations
734

Question: Does the paper discuss the limitations of the work performed by the authors?
735

Answer: [Yes]
736

Justification: Section2.1 line145-line147, Appendix F
737

Guidelines:
738

• The answer NA means that the paper has no limitation while the answer No means that
739

the paper has limitations, but those are not discussed in the paper.
740

• The authors are encouraged to create a separate "Limitations" section in their paper.
741

• The paper should point out any strong assumptions and how robust the results are to
742

violations of these assumptions (e.g., independence assumptions, noiseless settings,
743

model well-specification, asymptotic approximations only holding locally). The authors
744

should reflect on how these assumptions might be violated in practice and what the
745

implications would be.
746

• The authors should reflect on the scope of the claims made, e.g., if the approach was
747

only tested on a few datasets or with a few runs. In general, empirical results often
748

depend on implicit assumptions, which should be articulated.
749

• The authors should reflect on the factors that influence the performance of the approach.
750

For example, a facial recognition algorithm may perform poorly when image resolution
751

is low or images are taken in low lighting. Or a speech-to-text system might not be
752

used reliably to provide closed captions for online lectures because it fails to handle
753

technical jargon.
754

• The authors should discuss the computational efficiency of the proposed algorithms
755

and how they scale with dataset size.
756

• If applicable, the authors should discuss possible limitations of their approach to
757

address problems of privacy and fairness.
758

• While the authors might fear that complete honesty about limitations might be used by
759

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
760

limitations that aren’t acknowledged in the paper. The authors should use their best
761

judgment and recognize that individual actions in favor of transparency play an impor-
762

tant role in developing norms that preserve the integrity of the community. Reviewers
763

will be specifically instructed to not penalize honesty concerning limitations.
764

3. Theory Assumptions and Proofs
765

Question: For each theoretical result, does the paper provide the full set of assumptions and
766

a complete (and correct) proof?
767

Answer: [Yes]
768

26


---Page Break---
Justification: Assumption: Section2 line81 to line 94, Proof: Appendix A.
769

Guidelines:
770

• The answer NA means that the paper does not include theoretical results.
771

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
772

referenced.
773

• All assumptions should be clearly stated or referenced in the statement of any theorems.
774

• The proofs can either appear in the main paper or the supplemental material, but if
775

they appear in the supplemental material, the authors are encouraged to provide a short
776

proof sketch to provide intuition.
777

• Inversely, any informal proof provided in the core of the paper should be complemented
778

by formal proofs provided in appendix or supplemental material.
779

• Theorems and lemmas that the proof relies upon should be properly referenced.
780

4. Experimental Result Reproducibility
781

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
782

perimental results of the paper to the extent that it affects the main claims and/or conclusions
783

of the paper (regardless of whether the code and data are provided or not)?
784

Answer: [Yes]
785

Justification: Secition3 and Appendix B,C.
786

Guidelines:
787

• The answer NA means that the paper does not include experiments.
788

• If the paper includes experiments, a No answer to this question will not be perceived
789

well by the reviewers: Making the paper reproducible is important, regardless of
790

whether the code and data are provided or not.
791

• If the contribution is a dataset and/or model, the authors should describe the steps taken
792

to make their results reproducible or verifiable.
793

• Depending on the contribution, reproducibility can be accomplished in various ways.
794

For example, if the contribution is a novel architecture, describing the architecture fully
795

might suffice, or if the contribution is a specific model and empirical evaluation, it may
796

be necessary to either make it possible for others to replicate the model with the same
797

dataset, or provide access to the model. In general. releasing code and data is often
798

one good way to accomplish this, but reproducibility can also be provided via detailed
799

instructions for how to replicate the results, access to a hosted model (e.g., in the case
800

of a large language model), releasing of a model checkpoint, or other means that are
801

appropriate to the research performed.
802

• While NeurIPS does not require releasing code, the conference does require all submis-
803

sions to provide some reasonable avenue for reproducibility, which may depend on the
804

nature of the contribution. For example
805

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
806

to reproduce that algorithm.
807

(b) If the contribution is primarily a new model architecture, the paper should describe
808

the architecture clearly and fully.
809

(c) If the contribution is a new model (e.g., a large language model), then there should
810

either be a way to access this model for reproducing the results or a way to reproduce
811

the model (e.g., with an open-source dataset or instructions for how to construct
812

the dataset).
813

(d) We recognize that reproducibility may be tricky in some cases, in which case
814

authors are welcome to describe the particular way they provide for reproducibility.
815

In the case of closed-source models, it may be that access to the model is limited in
816

some way (e.g., to registered users), but it should be possible for other researchers
817

to have some path to reproducing or verifying the results.
818

5. Open access to data and code
819

Question: Does the paper provide open access to the data and code, with sufficient instruc-
820

tions to faithfully reproduce the main experimental results, as described in supplemental
821

material?
822

27


---Page Break---
Answer: [Yes]
823

Justification: We provide the full code in our supplementary.
824

Guidelines:
825

• The answer NA means that paper does not include experiments requiring code.
826

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
827

public/guides/CodeSubmissionPolicy) for more details.
828

• While we encourage the release of code and data, we understand that this might not be
829

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
830

including code, unless this is central to the contribution (e.g., for a new open-source
831

benchmark).
832

• The instructions should contain the exact command and environment needed to run to
833

reproduce the results. See the NeurIPS code and data submission guidelines (https:
834

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
835

• The authors should provide instructions on data access and preparation, including how
836

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
837

• The authors should provide scripts to reproduce all experimental results for the new
838

proposed method and baselines. If only a subset of experiments are reproducible, they
839

should state which ones are omitted from the script and why.
840

• At submission time, to preserve anonymity, the authors should release anonymized
841

versions (if applicable).
842

• Providing as much information as possible in supplemental material (appended to the
843

paper) is recommended, but including URLs to data and code is permitted.
844

6. Experimental Setting/Details
845

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
846

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
847

results?
848

Answer: [Yes]
849

Justification: Section3 and Appendix B, C.
850

Guidelines:
851

• The answer NA means that the paper does not include experiments.
852

• The experimental setting should be presented in the core of the paper to a level of detail
853

that is necessary to appreciate the results and make sense of them.
854

• The full details can be provided either with the code, in appendix, or as supplemental
855

material.
856

7. Experiment Statistical Significance
857

Question: Does the paper report error bars suitably and correctly defined or other appropriate
858

information about the statistical significance of the experiments?
859

Answer: [Yes]
860

Justification: Section 3 and Appendix B, C.
861

Guidelines:
862

• The answer NA means that the paper does not include experiments.
863

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
864

dence intervals, or statistical significance tests, at least for the experiments that support
865

the main claims of the paper.
866

• The factors of variability that the error bars are capturing should be clearly stated (for
867

example, train/test split, initialization, random drawing of some parameter, or overall
868

run with given experimental conditions).
869

• The method for calculating the error bars should be explained (closed form formula,
870

call to a library function, bootstrap, etc.)
871

• The assumptions made should be given (e.g., Normally distributed errors).
872

• It should be clear whether the error bar is the standard deviation or the standard error
873

of the mean.
874

28


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
875

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
876

of Normality of errors is not verified.
877

• For asymmetric distributions, the authors should be careful not to show in tables or
878

figures symmetric error bars that would yield results that are out of range (e.g. negative
879

error rates).
880

• If error bars are reported in tables or plots, The authors should explain in the text how
881

they were calculated and reference the corresponding figures or tables in the text.
882

8. Experiments Compute Resources
883

Question: For each experiment, does the paper provide sufficient information on the com-
884

puter resources (type of compute workers, memory, time of execution) needed to reproduce
885

the experiments?
886

Answer: [Yes]
887

Justification: Appendix E.
888

Guidelines:
889

• The answer NA means that the paper does not include experiments.
890

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
891

or cloud provider, including relevant memory and storage.
892

• The paper should provide the amount of compute required for each of the individual
893

experimental runs as well as estimate the total compute.
894

• The paper should disclose whether the full research project required more compute
895

than the experiments reported in the paper (e.g., preliminary or failed experiments that
896

didn’t make it into the paper).
897

9. Code Of Ethics
898

Question: Does the research conducted in the paper conform, in every respect, with the
899

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
900

Answer: [Yes]
901

Justification: We completely follow NeurIPS Code of Ethics.
902

Guidelines:
903

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
904

• If the authors answer No, they should explain the special circumstances that require a
905

deviation from the Code of Ethics.
906

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
907

eration due to laws or regulations in their jurisdiction).
908

10. Broader Impacts
909

Question: Does the paper discuss both potential positive societal impacts and negative
910

societal impacts of the work performed?
911

Answer: [Yes]
912

Justification: We propose a new conditional independence test with applications range in
913

multiple fields. Please refer to Appendix F.
914

Guidelines:
915

• The answer NA means that there is no societal impact of the work performed.
916

• If the authors answer NA or No, they should explain why their work has no societal
917

impact or why the paper does not address societal impact.
918

• Examples of negative societal impacts include potential malicious or unintended uses
919

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
920

(e.g., deployment of technologies that could make decisions that unfairly impact specific
921

groups), privacy considerations, and security considerations.
922

29


---Page Break---
• The conference expects that many papers will be foundational research and not tied
923

to particular applications, let alone deployments. However, if there is a direct path to
924

any negative applications, the authors should point it out. For example, it is legitimate
925

to point out that an improvement in the quality of generative models could be used to
926

generate deepfakes for disinformation. On the other hand, it is not needed to point out
927

that a generic algorithm for optimizing neural networks could enable people to train
928

models that generate Deepfakes faster.
929

• The authors should consider possible harms that could arise when the technology is
930

being used as intended and functioning correctly, harms that could arise when the
931

technology is being used as intended but gives incorrect results, and harms following
932

from (intentional or unintentional) misuse of the technology.
933

• If there are negative societal impacts, the authors could also discuss possible mitigation
934

strategies (e.g., gated release of models, providing defenses in addition to attacks,
935

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
936

feedback over time, improving the efficiency and accessibility of ML).
937

11. Safeguards
938

Question: Does the paper describe safeguards that have been put in place for responsible
939

release of data or models that have a high risk for misuse (e.g., pretrained language models,
940

image generators, or scraped datasets)?
941

Answer: [NA]
942

Justification: Method proposed in this paper don’t pose such risks.
943

Guidelines:
944

• The answer NA means that the paper poses no such risks.
945

• Released models that have a high risk for misuse or dual-use should be released with
946

necessary safeguards to allow for controlled use of the model, for example by requiring
947

that users adhere to usage guidelines or restrictions to access the model or implementing
948

safety filters.
949

• Datasets that have been scraped from the Internet could pose safety risks. The authors
950

should describe how they avoided releasing unsafe images.
951

• We recognize that providing effective safeguards is challenging, and many papers do
952

not require this, but we encourage authors to take this into account and make a best
953

faith effort.
954

12. Licenses for existing assets
955

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
956

the paper, properly credited and are the license and terms of use explicitly mentioned and
957

properly respected?
958

Answer: [Yes]
959

Justification: We have cited the dataset we use and we provide the code we based in
960

Appendix B.
961

Guidelines:
962

• The answer NA means that the paper does not use existing assets.
963

• The authors should cite the original paper that produced the code package or dataset.
964

• The authors should state which version of the asset is used and, if possible, include a
965

URL.
966

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
967

• For scraped data from a particular source (e.g., website), the copyright and terms of
968

service of that source should be provided.
969

• If assets are released, the license, copyright information, and terms of use in the
970

package should be provided. For popular datasets, paperswithcode.com/datasets
971

has curated licenses for some datasets. Their licensing guide can help determine the
972

license of a dataset.
973

• For existing datasets that are re-packaged, both the original license and the license of
974

the derived asset (if it has changed) should be provided.
975

30


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
976

the asset’s creators.
977

13. New Assets
978

Question: Are new assets introduced in the paper well documented and is the documentation
979

provided alongside the assets?
980

Answer: [Yes]
981

Justification: We have submitted the code.
982

Guidelines:
983

• The answer NA means that the paper does not release new assets.
984

• Researchers should communicate the details of the dataset/code/model as part of their
985

submissions via structured templates. This includes details about training, license,
986

limitations, etc.
987

• The paper should discuss whether and how consent was obtained from people whose
988

asset is used.
989

• At submission time, remember to anonymize your assets (if applicable). You can either
990

create an anonymized URL or include an anonymized zip file.
991

14. Crowdsourcing and Research with Human Subjects
992

Question: For crowdsourcing experiments and research with human subjects, does the paper
993

include the full text of instructions given to participants and screenshots, if applicable, as
994

well as details about compensation (if any)?
995

Answer: [NA]
996

Justification: We don’t use any crowdsourcing resoruce.
997

Guidelines:
998

• The answer NA means that the paper does not involve crowdsourcing nor research with
999

human subjects.
1000

• Including this information in the supplemental material is fine, but if the main contribu-
1001

tion of the paper involves human subjects, then as much detail as possible should be
1002

included in the main paper.
1003

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1004

or other labor should be paid at least the minimum wage in the country of the data
1005

collector.
1006

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1007

Subjects
1008

Question: Does the paper describe potential risks incurred by study participants, whether
1009

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1010

approvals (or an equivalent approval/review based on the requirements of your country or
1011

institution) were obtained?
1012

Answer: [NA]
1013

Justification: NA.
1014

Guidelines:
1015

• The answer NA means that the paper does not involve crowdsourcing nor research with
1016

human subjects.
1017

• Depending on the country in which research is conducted, IRB approval (or equivalent)
1018

may be required for any human subjects research. If you obtained IRB approval, you
1019

should clearly state this in the paper.
1020

• We recognize that the procedures for this may vary significantly between institutions
1021

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1022

guidelines for their institution.
1023

• For initial submissions, do not include any information that would break anonymity (if
1024

applicable), such as the institution conducting the review.
1025

31


---Page Break---
