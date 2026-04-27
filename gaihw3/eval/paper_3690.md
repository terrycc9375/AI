Implicit Causal Representation Learning via
Switchable Mechanisms

Anonymous Author(s)
Affiliation
Address
email

Abstract

Learning causal representations from observational and interventional data in the
1

absence of known ground-truth graph structures necessitates implicit latent causal
2

representation learning. Implicit learning of causal mechanisms typically involves
3

two categories of interventional data: hard and soft interventions. In real-world
4

scenarios, soft interventions are often more realistic than hard interventions, as the
5

latter require fully controlled environments. Unlike hard interventions, which di-
6

rectly force changes in a causal variable, soft interventions exert influence indirectly
7

by affecting the causal mechanism. However, the subtlety of soft interventions
8

impose several challenges for learning causal models. One challenge is that soft
9

intervention’s effects are ambiguous, since parental relations remain intact. In this
10

paper, we tackle the challenges of learning causal models using soft interventions
11

while retaining implicit modeling. Our approach models the effects of soft inter-
12

ventions by employing a causal mechanism switch variable designed to toggle
13

between different causal mechanisms. In our experiments, we consistently observe
14

improved learning of identifiable, causal representations, compared to baseline
15

approaches.
16

1
Introduction
17

Soft 
Intervention

Hard 
Intervention

Observations

Object’s class
Object’s color
Causal Graph

Fish
Bird
Bee
Cat
Spider
Bug
Classes

Figure 1: Difference between hard interventions
and soft interventions: As seen in the middle row,
hard interventions sever connections with parents.
Therefore, an object’s class cannot have any effect
on the object’s color when we intervene on color.
On the other hand, soft interventions, as shown in
the bottom row, allow for such effects.

One of the long-standing challenges in causal
18

representation learning is how to recover the
19

ground-truth causal graph of a system solely
20

from observations. Termed the identifiability
21

of causal models problem, this endeavor is cru-
22

cial. Without achieving identifiability, we risk
23

erroneously attributing causal relationships to
24

learned representations. Furthermore, statisti-
25

cal models can masquerade as Directed Acyclic
26

Graphs (DAGs) where edges lack causal signif-
27

icance, further complicating our pursuit.
28

When considering the challenge of identifying
29

causal models, it is known that the Markov con-
30

dition in graphs is insufficient for this task [26].
31

Thus, without additional assumptions or data,
32

we find ourselves limited to learning only a
33

Markov Equivalence Class (MEC) of the causal
34

model.
Existing works have made different
35

assumptions about availability of ground-truth
36

causal variables labels [34], model parameters
37

[1], availability of paired interventional data [3, 31], and availability of intervention targets [17] to
38

ensure identifiability of causal models.
39

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Interventional data are usually obtained through soft or hard interventions. Hard interventions
40

usually involve controlled experiments and they severe the connection of an intervened variable
41

with its parents [24]. In terms of Structural Causal Models (SCM), hard interventions set the causal
42

mechanism relating a causal variable to its parents, to a constant. Due to ethical or safety reasons, it
43

may not be possible to perform hard interventions in many real-world applications. On the other hand,
44

the effects of soft interventions are more subtle since parent variables can still affect their children.
45

These effects can be modeled by a change in the set of parents, the causal mechanisms, and the
46

exogenous variables [7]. Consequently, hard interventions can also be seen as a special case of soft
47

interventions where the causal mechanism is set to a constant. Illustrated in Figure 1, a prominent
48

challenge in causal representation learning lies in dealing with the ambiguity surrounding the effects
49

of soft interventions. The observed alterations in object colors fail to distinctly elucidate whether
50

they stem from parental influences or the applied interventions.
51

Additionally, a lack of comprehension regarding causal graphs can pose significant challenges in
52

causal representation learning. In certain applications, the causal graph can be constructed using
53

domain knowledge, allowing us to subsequently learn the causal variables [2, 18, 20]. However, this is
54

not universally applicable, necessitating the direct learning of the causal graph itself. In a Variational
55

AutoEncoder (VAE) framework, there are generally two approaches for causal representation learning:
56

Explicit Latent Causal Models (ELCMs) [34, 1, 35, 37, 17, 15] and Implicit Latent Causal Models
57

(ILCMs) [3]. In ELCMs, the latents are the causal variables and the adjacency matrix of the causal
58

graph is parameterized and integrated into the prior of the latents such that the prior of latents is
59

factorized according to the Causal Markov Condition [27]. This approach to causal representation
60

learning is highly susceptible to becoming stuck in local minima as it is hard to learn representations
61

without knowing the graph, and it is hard to learn the graph without knowing the representations.
62

ILCMs [3] were introduced to circumvent this “chicken-and-egg” problem by using solution functions,
63

which can implicitly model edges in the causal graph rather than explicitly modeling the entire
64

adjacency matrix of the causal model. In ILCMs the latents are the exogenous variables and the there
65

is no explicit parameterization for the graph.
66

In implicit causal representation learning, the task involves recovering the exogenous variables E
67

from observed variables X and learning solution functions. In [3], interventions are assumed to
68

be hard, but this is often unrealistic and does not align with real-world problems. In this paper,
69

we propose a novel approach for Implicit Causal Representation Learning via Switchable
70

Mechanisms (ICRL-SM). We will introduce the causal mechanism switch variable as a way of
71

modeling the effect of soft interventions and identifying the causal variables. Our experiments on
72

both synthetic and large real-world datasets, highlight the efficacy of proposed method in identifying
73

causal variables and promising future directions in implicit causal representation learning. Our key
74

contributions can be summarized as follows:
75

I. A novel approach for implicit causal representation learning with soft interventions.
76

II. Employing causal mechanisms switch variable to model the effect of soft interventions.
77

III. Theory for identifiability up to reparameterization from soft interventions.
78

79

2
Related Work
80

Causal representation learning has recently garnered significant attention [27, 14]. The primary
81

challenge in this problem lies in achieving identifiability beyond the Markov equivalence class [26].
82

Solely relying on observational data necessitates additional assumptions regarding causal mechanisms,
83

decoders, latent structure, and the availability of interventional data [22, 28, 36, 25, 15, 1, 40, 13,
84

34]. Recent works have focused on identifying causal models from collected interventional data
85

instead of making strong assumptions about functions of the causal model. Interventional data
86

facilitates identifiability based on relatively weak assumptions [1, 6, 3, 39, 33]. This type of data
87

can be further categorized based on whether it involves soft or hard interventions, and whether the
88

manipulated variables are observed and specified or latent. Our focus in this paper is on examining
89

soft interventions, encompassing both observed and unobserved variables.
90

2.1
Explicit models vs. Implicit models
91

Table 1 presents a comparison of the assumptions and identifiability results between our proposed
92

theory and other related works on causal representation learning with interventions. In causal repre-
93

sentation learning with interventions, one approach assumes a given causal graph and concentrates
94

on identifying causal mechanisms and mixing functions. For instance, Causal Component Analysis
95

(CauCA) [33] explores soft interventions with a known graph. Alternatively, when the graph is
96

2


---Page Break---
Table 1: Comparison of proposed method with other recent related work on causal learning from
interventional data

Methods
Causal Mechanisms
Mixing functions
Interventions
Explicit/Implicit
Identifiability

CausalDiscrepancy [38]
Nonlinear
Full row rank polynomial
Soft
Explicit
Permutation and Affine
CauCA [33]
Nonlinear
Diffeomorphism
Soft
Explicit
Different based on assumptions
Linear-CD [29]
Linear
Linear
Hard
Explicit
Permutation
Scale-I [30]
Nonlinear
Linear
Hard/Soft
Explicit
Scale/Mixed
ILCM [3]
Nonlinear
Diffeomorphism
Hard
Implicit
Permutation and reparameterization
dVAE [21]
Nonlinear
Diffeomorphism
Hard
Implicit
Permutation and reparameterization
ICRL-SM (ours)
Nonlinear
Diffeomorphism
Soft
Implicit
Reparameterization

not provided, explicit models seek to reconstruct it from interventional data [6, 17], potentially
97

resulting in a chicken-and-egg problem in causal representation learning [3]. Current methods face
98

the challenge of simultaneously learning the causal graph and other network parameters, especially
99

in the absence of information about causal variables or the graph. Addressing these challenges, [3]
100

recently introduced ILCM, which performs implicit causal representation learning exclusively using
101

hard intervention data. In contrast, our approach introduces a novel method for learning an implicit
102

model from soft interventions. [3] describes methods for extracting a causal graph from a learned
103

implicit model, which could be applied to our method as well. In our experiments, we will compare
104

our method with ILCM and dVAE [21], given their implicit nature and similar experimental settings
105

and assumptions. Additionally, to showcase the superiority of our method over explicit models, we
106

will employ explicit causal model discovery methods like ENCO [16] and DDS [5], in conjunction
107

with various variants of β-VAE.
108

2.2
Hard interventions vs Soft interventions
109

The identification of explicit causal models from hard interventions has been extensively ex-
110

plored. [29] investigate causal disentanglement in linear causal models with linear mixing functions
111

under hard interventions. Similarly, [4] focus on identifying causal models with linear causal mecha-
112

nisms and nonlinear mixing functions, also utilizing hard interventions. In a more general setting
113

with non-parametric causal mechanisms and mixing functions, [32] examine the identifiability of
114

causal models, utilizing multi-environment data from unknown interventions. Similarly, [2] explore
115

identifiability of causal models using multi-environment data from unknown interventions. [30]
116

investigate the identifiability of causal models with nonlinear causal mechanisms and linear mixing
117

functions, considering both hard and soft interventions.
118

Recent work has expanded the concept of explicit hard interventions to include soft interventions. In
119

their study, [38] address the identification of causal models from soft interventions, leveraging the
120

sparsity of the adjacency matrix as an inductive bias. However, when dealing with implicit models,
121

soft interventions introduce new complexities. Identifiability becomes more challenging, as the
122

causal effect of variables on observed variables is less apparent. This ambiguity arises from the dual
123

possibility of effects originating from interventions or influences from parent variables on the causal
124

variables. Moreover, in scenarios where implicit modeling is retained, the absence of knowledge about
125

parent variables further complicates identifiability. While [3] theoretically establishes identifiability
126

for hard interventions, practical experiments involving complex causal models with over 10 variables
127

reveal increased ambiguity and confounding factors. Consequently, model identification becomes
128

less straightforward.
129

3
Methodology
130

3.1
Data Generating Process
131

A structural causal model (Definition A1.1) is used to understand and describe the relationships
132

between different variables and how they influence each other through causal mechanisms. A decoder
133

function, g(z) = x, maps a vector of causal values z to observed values x. The causal variables
134

Z are unobserved and the goal is to infer them from interventional data. For each causal variable,
135

a diffeomorphic solution function, si : Ei →Zi, deterministically maps a value for exogenous
136

variable Ei to a value for causal variable Zi. In implicit modeling, we learn the solution functions si
137

directly, rather than defining them through local mechanisms fi. We write S for the set of all solution
138

functions si ∈S, so S : E →Z.
139

Identifying causal models from data can be complex and is often studied within classes of models
140

such as those identifiable up to affine transformations. For example, in the context of nonlinear
141

Independent Component Analysis (ICA), the generative process also involves a mixture function g of
142

latent causal variables Z ∈Rn, resulting in observations X ∈Rn [15, 41]. However, a significant
143

distinction between causal representation learning and nonlinear-ICA is that in the former, the causal
144

3


---Page Break---
variables Z may have complex dependencies. Our objective in this paper is to recover E from X and
145

eventually map E to Z using solution functions.
146

Identifying a causal model from observational data is not trivial and requires assumptions on the
147

parameters of the model [1]. Adding information about interventions in addition to observations,
148

helps to identify causal variables by exhibiting the effect of changing a causal variable on the observed
149

variables. An interventional data point (x, ˜x, i) includes the pre-intervention observation x, the post-
150

intervention observation ˜x, and intervention target i ∈I where I is the set of intervention targets
151

selected from the causal variables. The post-intervention data ˜x is generated by a soft intervention
152

that targets one of the causal variables in Z. To achieve identifiability up to reparametrization, we
153

rely on a series of assumptions within the data generation process, outlined as follows:
154

Assumption 3.1. (Data generating assumptions)
155

1. Atomic Interventions: For every sample (x, ˜x, i), only one causal variable is targeted by an
156

intervention.
157

2. Known Targets: Targets of soft interventions are known.
158

3. Post-intervention Exogenous Variables: The exogenous variables’ values change only for the
159

corresponding intervened causal variable, while the others maintain their pre-intervention values,
160

thus ei ̸= ˜ei if i ∈I ,and ei = ˜ei otherwise.
161

4. Sufficient Variability: Soft interventions alter causal mechanisms to introduce sufficient variability
162

[15]. These interventions should modify causal mechanisms to ensure non-overlapping conditional
163

distributions of causal variables (refer to Figure A1).
164

5. Diffeomorphic decoder and causal mechanisms: Diffeomorphism guarantees no information loss
165

and avoids abrupt changes in the function’s image.
166

The known targets assumption can be relaxed in applications where such data is not available
167

and the same procedure in [3] can be used to infer the intervention targets. In fact, in our real-
168

world experiments, intervention targets are not available and based on the nature of the datasets, we
169

hypothesize our causal variables to be object attributes and actions to be intervention targets.
170

3.2
Causal Mechanisms Switch Variable
171

The major difference of soft intervention with hard intervention is that post-intervention causal
172

variable ˜Zi is no longer disconnected from its parents and its causal mechanism ˜si is affected by the
173

intervention. This is why identifying the causal mechanisms is more difficult for soft interventions.
174

Soft intervention data yield fewer constraints on the causal graph structure than hard intervention
175

data. For more details refer to string diagrams of soft and hard interventions depicted in Figure A5.
176

Figure 2b shows our main generative model. It includes a data augmentation step that adds the
177

intervention displacement ˜x −x as an observed feature that directly represents the effect of a soft
178

intervention in observation space.
179

Augmented implicit causal model To model the effect of soft interventions, we introduce the
180

causal mechanism switch variable V [26]. By leveraging V, we can effectively switch to the pre-
181

intervention causal mechanisms within post-intervention data. This facilitates the model’s ability to
182

solely focus on discerning alterations in the intrinsic characteristics of each causal variable. These
183

changes are encapsulated within their respective exogenous variables, aiding the model in learning
184

the causal relationships more accurately. We propose to use a modulated form of V to model the
185

soft intervention effects on each causal variable as an additive effect with a nonlinear function hi
186

such that ∀i, ˜Zi = ˜si( ˜Ei; ˜E/i) = si( ˜Ei; E/i, hi(V)). As the parental set for each causal variable is
187

not known, we have to use a modulated form of V in every causal variable’s solution function and
188

the inclusion of hi(V) enables the model to encompass variations in the parental sets of all causal
189

variables in V. Therefore, there is a switch variable Vi for each causal variable Zi. Adding switch
190

variables to solution functions leads to the concept of an augmented implicit causal model.
191

Definition 3.2. (Augmented Implicit Causal Models) An Augmented Implicit Causal Models (AICMs)
192

is defined as A = (S, Z, E, V) where V ∈Rn is the causal mechanism switch variable which models
193

the effect of soft interventions on solution functions S:
194

∀i, ˜Zi = ˜si( ˜Ei; ˜E/i) = si( ˜Ei; E/i, hi(V)),
(1)

where ˜si is the new solution function resulting from the soft intervention, ˜E/i is the altered set of all
195

exogenous variables except i, including the ancestral exogenous variables, due to intervention, and
196

˜Ei is the post-intervention exogenous variable.
197

4


---Page Break---
The usage of V in soft interventions is analogous to augmented networks in [23] which were mainly
198

designed for hard interventions. Pearl [23] even foresaw this possibility by saying: "One advantage
199

of the augmented network representation is that it is applicable to any change in the functional
200

relationship fi and not merely to the replacement of fi by a constant."
201

By using Taylor’s expansion, we can expand the solution functions as follows:
202

si( ˜Ei; E/i, hi(V)) = si( ˜Ei; E/i, hi(v0)) + P∞
n=1
1
n!

 

∂nsi

∂hn
i


hi=hi(v0)
(hi(V) −hi(v0))n
!

= si( ˜Ei; E/i, hi(v0)) + Ri

(2)

where we’ll use Ri as a short-hand for Equation 2. We define the separable dependence property
203

for solution functions as ∃hi(v0) : si( ˜Ei; E/i, hi(v0)) = si( ˜Ei; E/i). An example of such a scenario
204

could be in location-scale noise models such as, si(˜ei; e/i, hi(v)) = ˜ei + loc(e/i) + hi(v) =
205

˜ei + loc(e/i) + v2 + v where v0 would be zero . By assuming the separable dependence property,
206

we can write the solution function in Equation 2 as:
207

si( ˜Ei; E/i, hi(V)) = si( ˜Ei; E/i) + Ri = si( ˜Ei; E/i) + soft intervention effect
(3)

As a result, we can switch to pre-intervention solution functions. Subsequently, by modeling soft
208

intervention effects using hi(V), we can recover pre-intervention solution functions. During inference,
209

we simply disregard the hi(V) term in the solution functions. Nonetheless, it is possible to train the
210

prior p(V) to ensure that the separable dependence property is maintained for pre-intervention data.
211

Observability of switch variable The intuition behind using V is to separate the effect of soft
212

intervention on ˜Zi into two: (1) The effect on causal mechanisms and parents, and (2) The effect on
213

exogenous variable Ei. For example, we can say that causal variables in images of objects are the
214

objects’ attributes such as shape, color, and size, and performing actions like "Fold" change these
215

attributes. Furthermore, it can be asserted that the camera angle within a given image may influence
216

the shape of the object. If the images were generated from a hard intervention, the camera angle
217

remains fixed between pre and post intervention. However, the camera angle changes along with
218

the performed actions indicating that the interventions are soft. In this case, if we had a knowledge
219

of how the camera angle affects the attributes of objects, then we could separate the effect of soft
220

intervention. In other words, if V is observed, then we can extract the effect of the intervention that
221

we are interested in (i.e., the effect on the causal variable itself). For more details, refer to Figure A4.
222

Lacking an understanding of how soft intervention influences the causal model, a more complex
223

model becomes necessary. Consequently, the term Ri in Equation 2 would involve a higher order of
224

hi(V). Therefore, we assume the observability of V:
225

Assumption 3.3. (Observability of V) Given an intervention sample (x, ˜x, i) and linear decoders,
226

we can approximate the soft intervention effects hi(V) as follows:
227

˜z −z = ∆ei + R
(using Equation 2),
˜x −x = g(˜z) −g(z) ≈g(˜z −z) = g(∆ei + R),

where R = [R0, R1, ..., Rn] and n is the number of causal variables. R and ∆ei are the vectors
228

indicating the soft intervention effects and change in effect of the exogenous variable of the intervened
229

causal variable, respectively. Note that elements of R will be all zero except for the intervened causal
230

variable. Consequently, with linear mixing functions and some pre-processing on observed samples
231

(here subtraction), we can observe Ri.
232

Our synthetic data is generated using a linear decoder, however, the decoder for the real-world
233

datasets is not necessarily linear. Therefore, we do not observe V from ˜x −x in the real-world dataset.
234

Nevertheless, our findings suggest that incorporating soft interventions through V leads to superior
235

performance compared to other implicit modeling approaches. Clearly, understanding the impact of
236

soft interventions on the generative system of the dataset would result in improved outcomes.
237

3.3
Identifiability Theorem for Implicit SCMs with Soft Interventions
238

In this paper, our focus lies in identifying the causal variables up to reparameterization through soft
239

interventions. We first define identifiability up to reparameterization (Definition 3.4) and subsequently
240

introduce the identifiability theorem 3.5. The proof of theorem is extensive and is available in full in
241

Appendix A1.
242

We establish identifiability up to reparameterization, allowing for the mapping of causal variables Z
243

and Z′ between two Latent Causal Models (M and M′) through component-wise transformations
244

5


---Page Break---
M
V
V =

"!

⋮
""

Encoder

𝑿

Pre
Intervention

Post
Intervention

FC

FC

Causal 
Mechanism 

Switch

Location

Scale

Solution Function (S)

Encoder

𝑿" −𝑿

𝑋"  −𝑋

M
V
! =

!!

⋮
!"

M
V
!̃ =

!̃!

⋮
!̃"

Decoder

Decoder

𝒁"

𝑋

𝑋"

𝑿"

(a) General overview of ICRL-SM

𝑒

𝑥#
𝑥" −𝑥
𝑥

𝑒̃
𝑣
𝑧
𝑧̃

(b) Generative model

(Definition A1.2). Given our implicit modeling approach, lacking knowledge of the causal graph, we
245

include all exogenous variables in the solution functions, as depicted in Equation 1. Notably, the
246

causal graph remains unaltered during learning. To illustrate, we contrast hard interventions,
247

which neglect parent influences, with soft interventions that acknowledge parental effects in a simple
248

example. Consider a basic causal model Z1 →Z2 alongside a location-scale noise model [12] for the
249

solution function, given by ˜z2 = ˜e2−f
loc(e1)
]
scale(e1) . The distribution p( ˜Z2) mean is
1
]
scale(e1) × mean( ˜E2) −
250

f
loc(e1)
]
scale(e1) In the context of hard interventions, we can assume p( ˜Z2|Z1) = p( ˜Z2) = N(0, 1) as there
251

are no parental effects. Consequently, the location and scale networks within the solution function tend
252

to dampen parental effects, given the absence of parental influence in the ground-truth data. Contrarily,
253

soft interventions exhibit parental influence in the ground-truth data, thus p( ˜Z2|Z1) ̸= N(0, 1). Due
254

to the lack of parental knowledge in implicit modeling, we model p( ˜Z2|Z1) = p( ˜Z2|E2), as E2
255

is a known parent of ˜Z2. Consequently, parental effects are propagated to Ei (the corresponding
256

exogenous variable of each causal variable), violating identifiability up to reparameterization. By
257

leveraging V, we allow parental effects to propagate to V instead of Ei.
258

Definition 3.4. (Equivalence up to component-wise reparameterization) Let M = (A, X, g, I)
259

and M′ = (A′, X, g′, I) be two Latent Causal Models (LCM) based on AICMs A, A′ with shared
260

observation space X, shared intervention targets I, and respective decoders g and g′. We say that
261

M and M′ are equivalent up to component-wise reparameterization M ∼r M′ if there exists a
262

component-wise transformation (Definition A1.2) ϕZ from the causal variables Z to the causal
263

variables Z′ and a component-wise transformation ϕE between E and E′ such that:
264

1. Indices are preserved (i.e., ϕi(zi) = z′
i and ϕi(ei) = e′
i). Corresponding edges are preserved (i.e.,
265

Zi →Zj holds in G iff Z′
i →Z′
j holds in G′. Edges Ei →Zi should be preserved as well.)
266

2.
The exogenous transformation preserves the probability measure on exogenous variables
267

pE′ = (ϕE)∗pE (Definition A1.4).
268

3. The causal transformation preserves the probability measure on causal variables pZ′ = (ϕZ)∗pZ
269

(Definition A1.4).
270

271

Theorem 3.5. (Identifiability of latent causal models.)
Let M = (A, X, g, I) and M′ =
272

(A′, X, g′, I) be two LCMs with shared observation space X and shared intervention targets I.
273

Suppose the following conditions are satisfied:
274

1. Data generating assumptions explained in Assumption 3.1.
275

2. Soft interventions satisfy Assumption 3.3.
276

3. The causal and exogenous variables are real-valued.
277

4. The causal and exogenous variables follow a multivariate normal distribution.
278

Then the following statements are equivalent:
279

-Two LCMs M and M′ assign the same likelihood to interventional and observational data i.e.,
280

pX,I
M (x, ˜x, i) = pX,I
M′ (x, ˜x, i).
281

- M and M′ are disentangled, that is M ∼r M′ according to Definition 3.4.
282

3.4
Training Objective
283

Consequently, there will be three latent variables in ICRL-SM:
284

1. A causal mechanism switch variable V.
285

2. The pre-intervention exogenous variables E.
286

6


---Page Break---
3. The post-intervention exogenous variables ˜E.
287

As the data log-likelihood log p(x, ˜x, x −˜x) ≡log p(x, ˜x) is intractable, we utilize an ELBO
288

approximation as training objective:
289

log p(x, ˜x) ≥Eq(e,˜e,v|x,˜x)
h
log p(x, ˜x|e, ˜e, v)
i
−KLD(q(e, ˜e, v|x, ˜x)||p(e, ˜e, x))

= Eq(v|˜x−x)·q(e|x)·q(˜e|˜x)
h
log(p(x|e)p(˜x|˜e)p(˜x −x|v))
i
−KLD(q(v|˜x −x) · q(e|x) · q(˜e|˜x)||p(˜e|e, v)p(v)p(e)).

(4)
The observations are encoded and decoded independently. The KLD term regularizes the encodings
290

to share the latent intervention model p(˜e|e, v)p(v)p(e) that is shared across all data points. The
291

components of this model can be interpreted as follows:
292

1. p(e) is the prior distribution over exogenous variables e.
293

2. p(v) is the prior distribution over switch variables v.
294

3. p(˜e|e, v) is a transition model that shows how the exogeneous variables change as a function of the
295

intervention.
296

We factorize the posterior with a mean-field approximation q(v, e, ˜e|x, ˜x) = q(v|˜x −x) · q(e|x) ·
297

q(˜e|˜x) and, following our data generation model (Figure 2b), the reconstruction probability
298

as p(x, ˜x|e, ˜e, v) = p(x|e)p(˜x|˜e)p(˜x −x|v).
The prior over latent variables is factorized as
299

p(˜e, e, v) = p(˜e|e, v)p(v)p(e)(Figure 2b). Pre-intervention exogenous variables are mutually inde-
300

pendent, hence, p(e) = Πip(ei) and p(v) = Πip(vi). We assume p(ei) and p(vi) to be standard
301

Gaussian. Furthermore, as we assume ei = ˜ei for all non-intervened variables, the p(˜e|e, v) will be
302

as follows:
303

p(˜e|e, v) = Πi/∈Iδ(˜ei −ei)Πi∈Ip(˜ei|e, v) = Πi/∈Iδ(˜ei −ei)Πi∈Ip(˜zi|ei)

∂˜zi
∂˜ei


(5)

The last equality is obtained from the Change of Variable Rule in probability theory, applied to the
304

solution function ˜zi = si(˜ei; e/i, hi(v)). Furthermore, we write p(˜zi|e, v) = p(˜zi|ei) since only ei
305

is a known parent of ˜zi in implicit modeling. We assume p(˜zi|ei) to be a Gaussian whose mean is
306

determined by ei. We implement the solution function using a location-scale noise models [12] as
307

also practiced in [3], which defines an invertible diffeomorphism. For simplicity, in our experiments,
308

we are only going to change the loc network in post-intervention. Therefore, hi(v) will be used as:
309

˜zi = ˜si(˜ei; e/i, hi(v)) = ˜ei −(loci(e/i) + hi(v))

scalei(e/i)
,
(6)

where loci : Rn−1 →R and scalei : Rn−1 →R are fully connected networks calculating the first
310

and second moments, respectively. The general overview of the model is illustrated in Figure 2a.
311

4
Experiments and Results
312

The experiments conducted in this paper address two downstream tasks; (1) Causal Disentanglement
313

to identify the true causal graph from pairs of observations (x, ˜x, i), and (2) Action Inference to make
314

supervised inferences about actions generated from the post-intervention samples using information
315

about the values of the manipulated causal variables. Moreover, we conducted additional experiments
316

designed as an ablation study, the results of which are presented in A4. All models are trained using
317

the same setting and data with known intervention targets.
318

4.1
Datasets
319

Synthetic Dataset We generate simple synthetic datasets with X = Z = Rn. For each value of
320

n, we generate ten random DAGs, a random location-scale SCM, then a random dataset from the
321

parameterized SCM. To generate random DAGs, each edge is sampled in a fixed topological order
322

from a Bernoulli distribution with probability 0.5. The pre-intervention and post-intervention causal
323

variables are obtained as:
324

zi = scale(zpai)ei + loc(zpai)
Soft-Intervention
−−−−−−−−−→˜zi = scale(zpai)˜ei + f
loc(zpai),
(7)

where the loc and scale networks are changed in post intervention. The pre-intervention loc and
325

post-intervention f
loc network weights are initialized with samples drawn from N(0, 1) and N(3, 1),
326

respectively. The scale is constant 1 for both pre-intervention and post-intervention samples. Both
327

ei and ˜ei are sampled from a standard Gaussian. The causal variables are mapped to the data space
328

through a randomly sampled SO(n) rotation. For each dataset, we generate 100,000 training samples,
329

10,000 validation samples, and 10,000 test samples.
330

7


---Page Break---
Action Datasets Causal-Triplet datasets tailored for actionable counterfactuals [19] feature paired
331

images where several global scene properties may vary including camera view and object occlusions.
332

Thus, the images can be viewed as outcomes of soft interventions, wherein actions affect objects
333

alongside subtle alterations. These datasets [19] consist of: images obtained from a photo-realistic
334

simulator of embodied agents, ProcTHOR [9], and the other contains images repurposed from a real-
335

world video dataset of human-object interactions [8]. The former one contains 100 k images in which
336

7 types of actions manipulate 24 types of objects in 10 k distinct ProcTHOR indoor environments.
337

The latter consists of 2,632 image pairs, collected under a similar setup from the Epic-Kitchens
338

dataset with 97 actions manipulating 277 objects.Based on the nature of actions in this dataset, the
339

causal variables should represent attributes of objects such as shape and color. As the dataset consists
340

of images we train all the methods with ResNet encoder and decoder. For the ProcThor dataset the
341

number of causal variables are 7. For the Epic-Kitchens dataset, we randomly chose 20 actions from
342

the dataset as 97 causal variables will be too complex in a VAE setup.
343

4.2
Metrics
344

For the causal disentanglement task, we are going to use the DCI scores [10]. Causal disentanglement
345

score quantifies the degree to which Zi factorises or disentangles the Z∗. Causal disentanglement Di
346

for Zi is calculated as Di = (1 −HK(Pi.)) = (1 + PK−1
k=0 Pik logK Pik) where Pij =
Rij
PK−1
k=0 Rik
347

and Rij denotes the probability of Zi being important for predicting Z∗
j . Total causal disentanglement
348

is the weighted average P

i ρiDi where ρi =

P

j Rij
P

ij Rij . Causal Completeness quantifies the degree
349

to which each Z∗
i is captured by a single Zi. Causal completeness is calculated as Cj = (1 −
350

HD( ˜P.j)) = (1 + PD−1
d=0 ˜Pdj logD ˜Pij). D and K here are equal to the dimension of Z∗and Z
351

which is n. For the action inference task, we will use classification accuracy as a metric. As we
352

assume intervention targets are known, we train all models using known intervention targets for a fair
353

comparison.
354

5
Results
355

5.1
Causal Disentanglement
356

We generated a dataset for the soft interventions and trained the models of ICRL-SM, ILCM, β-VAE
357

and D-VAE for 10 different seeds, which generated 10 different causal graphs. We selected 4 causal
358

variables to encompass complex causal structures, including forks, chains, and colliders. Table 2
359

displays the Causal Disentanglement and Causal Completeness scores for all models, computed on
360

the test data.
361

Table 2: Comparison of identifiability results

Graph
Causal Disentanglement
Causal Completeness

Model
Name
β-VAE
d-VAE
ILCM
ICRL-SM
β-VAE
d-VAE
ILCM
ICRL-SM

G1
0.38
0.54
0.71
0.82
0.51
0.69
0.78
0.87

G2
0.30
0.72
0.75
0.83
0.49
0.77
0.80
0.87

G3
0.28
0.51
0.68
0.98
0.49
0.56
0.78
0.98

G4
0.16
0.50
0.65
0.68
0.38
0.69
0.77
0.78

G5
0.27
0.44
0.53
0.42
0.45
0.54
0.66
0.50

G6
0.52
0.62
0.71
0.98
0.66
0.69
0.86
0.98

G7
0.39
0.49
0.71
0.75
0.70
0.73
0.89
0.89

G8
0.47
0.54
0.50
0.59
0.6
0.63
0.62
0.68

G9
0.30
0.68
0.83
0.85
0.40
0.76
0.86
0.87

G10
0.39
0.39
0.52
0.32
0.53
0.56
0.82
0.70

The results in Table 2 indicate that our method ICRL-SM can identify the true causal graph in most
362

cases. The worst results are seen for graphs G5 and G10. As mentioned in [27, 25], causal graphs are
363

sparse and in the G5 case, where the graph is fully connected, the proposed method cannot identify
364

the causal variables well. Furthermore, in the next experiment we are going to examine the factors
365

affecting causal disentanglement such as the number of edges in the graph and the intensity of soft
366

intervention effect. These findings can explain why ICRL-SM cannot identify causal variables in
367

G10 despite its sparsity.
368

8


---Page Break---
Table 3: Table comparing action and object accuracy across various methods on Causal-Triplet
datasets under different settings. Z and zi show whether all causal variables (Z), or only the
intervened casual variable (zi) are used for the prediction task. R64 denote images with resolutions
64 × 64.

Epic-Kitchens
ProcTHOR

Action Accuracy
Object Accuracy
Action Accuracy
Object Accuracy

Method
Z;R64
zi;R64
Z;R64
zi;R64
Z;R64
zi;R64
Z;R64
zi;R64

β −V AE [11]
0.27
0.18
0.19
0.06
0.39
0.30
0.44
0.37
d −V AE [21]
0.19
0.69
0.20
0.17
0.35
0.81
0.40
0.78
ILCM [3]
0.21
0.59
0.14
0.14
0.30
0.70
0.41
0.76
ICRL-SM (ours)
0.16
0.86
0.16
0.18
0.28
0.93
0.40
0.82

5.2
Factors Affecting Causal Disentanglement
369

In this experiment, we consider the graph G3, which has the best identifiability, and change the
370

intensity of soft intervention and number of edges in its data generation process. To change the
371

intensity, the post-intervention f
loc network weights are initialized with samples drawn from N(1, 1)
372

(almost similar to loc) and N(10, 1) (significantly different from loc). To change the number of
373

edges, we consider a chain and fully-connected graph.
374

Table 4: Left table depicts the action and object accuracy of three explicit models, with experiments
conducted applying an image with resolution of R64 as the input to the Resnet50 encoder with the
intervened causal variable (zi). Right table shows the comparison of ICRL-SM performance on
different configurations of G5

Datasets
Methods
Action Accuracy
Object Accuracy

Epic-Kitchens
ENCO [16]
0.69
0.13
DDS [5]
0.44
0.09
Fixed-order
0.79
0.14
ICRL-SM (ours)
0.86
0.18

ProcTHOR
ENCO [16]
0.45
0.53
DDS [5]
0.64
0.67
Fixed-order
0.65
0.54
ICRL-SM (ours)
0.93
0.82

Edges
Post-intervention
Causal
Causal
causal mechanism
Disentanglement
Completeness

Chain
Default
0.98
0.98
Full
Default
0.89
0.89
Default
Significantly different
0.68
0.73
Default
Almost similar
0.85
0.86

The results in Table 4 further confirms the sparsity of causal graphs as the causal disentanglement is
375

much worse in the fully-connected graph than the default graph of G3. The result for significantly
376

different post-intervention causal mechanisms indicate that the switch variable cannot approximate
377

intense effects of soft intervention and more supervision is required to observe V. Similar post-
378

intervention causal mechanisms also do not have sufficient variability to disentangle the causal
379

variables as mentioned in Theory 3.5.
380

5.3
Action Inference
381

In this experiment, we show the performance of ICRL-SM in the real-world Causal-Triplet datasets.
382

In these datasets V i.e., soft intervention effects, are not directly observable. Nevertheless, our findings
383

suggest that incorporating soft interventions through V leads to superior performance compared to
384

other implicit modeling approaches. Clearly, understanding the impact of soft interventions on the
385

generative system of the dataset would result in improved outcomes.
386

The results in Table 3 indicate that when including all causal variables to predict actions, ICRL-SM
387

performs at par with the baseline methods. However, including all causal variables in the action
388

or object inference may cause spurious correlations. Therefore, we have also experimented with
389

including only the related causal variable in action and object inference. In this setting, ICRL-
390

SM significantly outperforms the baseline methods which means that it can better disentangle the
391

causal variables. We have also compared ICRL-SM with explicit causal representation learning
392

methods. ENCO [16] and DDS [5] have variable topological order of causal variables during training.
393

Furthermore, we have included a specific setting where the topological order is fixed during training.
394

As shown in Table 4, our proposed method has superior performance to explicit models as well.
395

6
Conclusion
396

ICRL-SM, our novel model, enhances implicit causal representation learning during soft interventions
397

by introducing a causal mechanism switch variable. Evaluations on synthetic and real-world datasets
398

demonstrate ICRL-SM’s superiority over state-of-the-art methods, highlighting its practical effective-
399

ness. Our findings emphasize ICRL-SM’s ability to discern causal models from soft interventions,
400

marking it as a promising avenue for future research.
401

9


---Page Break---
References
402

[1] Kartik Ahuja, Divyat Mahajan, Yixin Wang, and Yoshua Bengio. Interventional causal representation
403

learning. In International Conference on Machine Learning, ICML, volume 202 of Proceedings of Machine
404

Learning Research, pages 372–407. PMLR, 2023.
405

[2] Shayan Shirahmad Gale Bagi, Zahra Gharaee, Oliver Schulte, and Mark Crowley. Generative causal
406

representation learning for out-of-distribution motion forecasting. In International Conference on Machine
407

Learning, ICML, volume 202 of Proceedings of Machine Learning Research, pages 31596–31612. PMLR,
408

2023.
409

[3] Johann Brehmer, Pim de Haan, Phillip Lippe, and Taco S. Cohen. Weakly supervised causal representation
410

learning. In NeurIPS, 2022.
411

[4] Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, and Pradeep
412

Ravikumar. Learning linear causal representations from interventions under general nonlinear mixing,
413

2023.
414

[5] Bertrand Charpentier, Simon Kibler, and Stephan Günnemann. Differentiable DAG sampling. In The Tenth
415

International Conference on Learning Representations, ICLR. OpenReview.net, 2022.
416

[6] Gregory F. Cooper and Changwon Yoo. Causal discovery from a mixture of experimental and observational
417

data, 2013.
418

[7] Juan D. Correa and Elias Bareinboim. General transportability of soft interventions: Completeness results.
419

In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin,
420

editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information
421

Processing Systems, NeurIPS, 2020.
422

[8] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma,
423

Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric
424

vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. Int. J. Comput. Vis., 130(1):33–55,
425

2022.
426

[9] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric
427

Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural
428

generation. Advances in Neural Information Processing Systems, 35:5982–5994, 2022.
429

[10] Cian Eastwood and Christopher K. I. Williams. A framework for the quantitative evaluation of disentangled
430

representations. In 6th International Conference on Learning Representations, ICLR, 2018.
431

[11] Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M. Botvinick,
432

Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained
433

variational framework. In 5th International Conference on Learning Representations, ICLR, 2017.
434

[12] Alexander Immer, Christoph Schultheiss, Julia E. Vogt, Bernhard Schölkopf, Peter Bühlmann, and Alexan-
435

der Marx. On the identifiability and estimation of causal location-scale noise models. In International
436

Conference on Machine Learning, ICML, volume 202 of Proceedings of Machine Learning Research,
437

pages 14316–14332. PMLR, 2023.
438

[13] Amin Jaber, Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Causal discovery from soft
439

interventions with unknown targets: Characterization and learning. In Advances in Neural Information
440

Processing Systems 33: Annual Conference on Neural Information Processing Systems, NeurIPS, 2020.
441

[14] Jean Kaddour, Aengus Lynch, Qi Liu, Matt J. Kusner, and Ricardo Silva. Causal machine learning: A
442

survey and open problems. CoRR, abs/2206.15475, 2022.
443

[15] Sébastien Lachapelle, Pau Rodríguez, Yash Sharma, Katie Everett, Rémi Le Priol, Alexandre Lacoste,
444

and Simon Lacoste-Julien. Disentanglement via mechanism sparsity regularization: A new principle for
445

nonlinear ICA. In 1st Conference on Causal Learning and Reasoning, CLeaR, volume 177 of Proceedings
446

of Machine Learning Research, pages 428–484. PMLR, 2022.
447

[16] Phillip Lippe, Taco Cohen, and Efstratios Gavves. Efficient neural causal discovery without acyclicity
448

constraints. In The Tenth International Conference on Learning Representations, ICLR. OpenReview.net,
449

2022.
450

[17] Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M. Asano, Taco Cohen, and Stratis Gavves. CITRIS:
451

causal identifiability from temporal intervened sequences. In International Conference on Machine
452

Learning, ICML, volume 162 of Proceedings of Machine Learning Research, pages 13557–13603. PMLR,
453

2022.
454

[18] Chang Liu, Xinwei Sun, Jindong Wang, Haoyue Tang, Tao Li, Tao Qin, Wei Chen, and Tie-Yan Liu.
455

Learning causal semantic representation for out-of-distribution prediction. In M. Ranzato, A. Beygelzimer,
456

Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing
457

Systems, volume 34, pages 6155–6170. Curran Associates, Inc., 2021.
458

[19] Yuejiang Liu, Alexandre Alahi, Chris Russell, Max Horn, Dominik Zietlow, Bernhard Schölkopf, and
459

Francesco Locatello. Causal triplet: An open challenge for intervention-centric causal representation
460

learning. In Conference on Causal Learning and Reasoning, CLeaR, volume 213 of Proceedings of
461

Machine Learning Research, pages 553–573. PMLR, 2023.
462

[20] Yuejiang Liu, Riccardo Cadei, Jonas Schweizer, Sherwin Bahmani, and Alexandre Alahi. Towards robust
463

and adaptive motion forecasting: A causal representation perspective. In IEEE/CVF Conference on
464

Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages
465

10


---Page Break---
17060–17071. IEEE, 2022.
466

[21] Francesco Locatello, Ben Poole, Gunnar Rätsch, Bernhard Schölkopf, Olivier Bachem, and Michael
467

Tschannen. Weakly-supervised disentanglement without compromises. In Proceedings of the 37th
468

International Conference on Machine Learning,ICML, volume 119 of Proceedings of Machine Learning
469

Research, pages 6348–6359. PMLR, 2020.
470

[22] Chaochao Lu, Yuhuai Wu, José Miguel Hernández-Lobato, and Bernhard Schölkopf. Invariant causal
471

representation learning for out-of-distribution generalization. In The Tenth International Conference on
472

Learning Representations, ICLR, 2022.
473

[23] Judea Pearl. Causality, cambridge university press (2000). Artif. Intell., 169(2):174–179, 2005.
474

[24] Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell. Causal inference in statistics: A primer. John
475

Wiley and Sons, 2016.
476

[25] Ronan Perry, Julius von Kügelgen, and Bernhard Schölkopf. Causal discovery in heterogeneous environ-
477

ments under the sparse mechanism shift hypothesis. In NeurIPS, 2022.
478

[26] Bernhard Schölkopf. Causality for machine learning. CoRR, abs/1911.10500, 2019.
479

[27] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh
480

Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–
481

634, 2021.
482

[28] Xinwei Shen, Furui Liu, Hanze Dong, Qing Lian, Zhitang Chen, and Tong Zhang. Weakly supervised
483

disentangled generative causal representation learning. J. Mach. Learn. Res., 23:241:1–241:55, 2022.
484

[29] Chandler Squires, Anna Seigal, Salil Bhate, and Caroline Uhler. Linear causal disentanglement via
485

interventions, 2023.
486

[30] Burak Varici, Emre Acarturk, Karthikeyan Shanmugam, Abhishek Kumar, and Ali Tajer. Score-based
487

causal representation learning with interventions, 2023.
488

[31] Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve,
489

and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from
490

style. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances
491

in Neural Information Processing Systems, volume 34, pages 16451–16467. Curran Associates, Inc., 2021.
492

[32] Julius von Kügelgen, Michel Besserve, Liang Wendong, Luigi Gresele, Armin Keki´c, Elias Bareinboim,
493

David M. Blei, and Bernhard Schölkopf. Nonparametric identifiability of causal representations from
494

unknown interventions, 2023.
495

[33] Liang Wendong, Armin Keki´c, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele,
496

and Bernhard Schölkopf. Causal component analysis, 2023.
497

[34] Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang. Causalvae: Disentan-
498

gled representation learning via neural structural causal models. In IEEE Conference on Computer Vision
499

and Pattern Recognition, CVPR, pages 9593–9602. Computer Vision Foundation / IEEE, 2021.
500

[35] Shuai Yang, Kui Yu, Fuyuan Cao, Lin Liu, Hao Wang, and Jiuyong Li. Learning causal representations for
501

robust domain adaptation. IEEE Transactions on Knowledge and Data Engineering, pages 1–1, 2021.
502

[36] Kui Yu, Xianjie Guo, Lin Liu, Jiuyong Li, Hao Wang, Zhaolong Ling, and Xindong Wu. Causality-based
503

feature selection: Methods and evaluations. ACM Comput. Surv., 53(5), 2020.
504

[37] Yue Yu, Jie Chen, Tian Gao, and Mo Yu. DAG-GNN: DAG structure learning with graph neural networks.
505

In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of
506

Machine Learning Research, pages 7154–7163. PMLR, 2019.
507

[38] Jiaqi Zhang, Chandler Squires, Kristjan Greenewald, Akash Srivastava, Karthikeyan Shanmugam, and
508

Caroline Uhler. Identifiability guarantees for causal disentanglement from soft interventions, 2023.
509

[39] Jiaqi Zhang, Chandler Squires, Kristjan H. Greenewald, Akash Srivastava, Karthikeyan Shanmugam,
510

and Caroline Uhler. Identifiability guarantees for causal disentanglement from soft interventions. CoRR,
511

abs/2307.06250, 2023.
512

[40] Xun Zheng, Bryon Aragam, Pradeep Ravikumar, and Eric P. Xing. Dags with NO TEARS: continuous
513

optimization for structure learning. In Advances in Neural Information Processing Systems 31: Annual
514

Conference on Neural Information Processing Systems NeurIPS, pages 9492–9503, 2018.
515

[41] Yujia Zheng, Ignavier Ng, and Kun Zhang. On the identifiability of nonlinear ICA: sparsity and beyond. In
516

NeurIPS, 2022.
517

11


---Page Break---
Appendix
518

A1
Proof of Identifiability Theorem
519

In order to prove our model is identifiable we need a two additional definitions and some previously
520

stated assumptions.
521

Definition A1.1. Structural Causal Models
522

A structural causal model (SCM) is a tuple C = (F, Z, E, G) with the following components:
523

1. The domain of causal variables Z = Z1 × Z2 × . . . × Zn.
524

2. The domain of exogenous variables E = E1 × E2 × . . . × En.
525

3. A directed acyclic graph G(C) over the causal and exogenous variables.
526

4. A causal mechanism fi ∈F which maps an assignment of parent values for the parents Zpai plus
527

an exogenous variable value for Ei to a value of causal variable Zi.
528

Definition A1.2. (Component-wise Transformation) Let ϕ be a transformation (1-1 onto mapping)
529

between product spaces ϕ : Πn
i=1Xi →Πn
i=1Yi. If there exist local transformations ϕi such that
530

∀i, j, ∀x, ϕ(x1, x2, ..., xn)i = ϕi(xj), then ϕ is a component-wise transformation.
531

Definition A1.3. (Diffeomorphism) A diffeomorphism between smooth manifolds M and N is a
532

bijective map f : M →N, which is smooth and has a smooth inverse. Diffeomorphisms preserve
533

information as they are invertible transformations without discontinuous changes in their image.
534

Definition A1.4. (Pushforward measure) Given a measurable function f : A →B between two
535

measurable spaces A and B, and a measure p defined on A, the pushforward measure f∗p on B is
536

defined for measurable sets E in B as:
537

(f∗p)(E) = p(f −1(E))
538

where ∗denotes the pushforward operation. In other words, the pushforward measure f∗p assigns a
539

measure to a set in B by measuring the pre-image of that set under f in the space A.
540

Lemma A1.5. The transformation ϕZ : Z →Z′ between the causal variable of two LCMs M
541

and M′ defined in Definition 3.4 is a component-wise transformation, if ∀i, j, i ̸= j
˜E′
i ⊥⊥˜E′
j and
542

the causal variables follow a multivariate normal distribution conditional on the pre-intervention
543

exogenous variables where ˜E′
i denote the post-intervention exogenous variable of causal variable i
544

in M′.
545

proof: We consider the case where the exogenous variables are mapped to causal variables by a
546

location-scale noise model such that ˜zi =
˜ei−f
loc(e/i)
]
scale(e/i) .
547

∀i, j, i ̸= j
˜E′
i ⊥⊥˜E′
j →E[ ˜E′
i ˜E′
j] = E[ ˜E′
i]E[ ˜E′
j]

12


---Page Break---
let’s add these three constants −E[ ˜E′
i]g
loc′
j(e′
/j), −E[ ˜E′
j]g
loc′
i(e′
/i), g
loc′
i(e′
/i)g
loc′
j(e′
/j) to the both
548

sides of the equality and then divide both sides by ^
scale′
i(e′
/i) ^
scale′
j(e′
/j):
549

E




˜E′
i ˜E′
j −˜E′
i g
loc′
j(e′
/j) −˜E′
j g
loc′
i(e′
/i) + g
loc′
i(e′
/i)g
loc′
j(e′
/j)

^
scale′
i(e′
/i) ^
scale′
j(e′
/j)



=

E[ ˜E′
i]E[ ˜E′
j] −E[ ˜E′
i]g
loc′
j(e′
/j) −E[ ˜E′
j]g
loc′
i(e′
/i) + g
loc′
i(e′
/i)g
loc′
j(e′
/j)

^
scale′
i(e′
/i) ^
scale′
j(e′
/j)

→E



(
˜E′
i −g
loc′
i(e′
/i)

^
scale′
i(e′
/i)
)(
˜E′
j −g
loc′
j(e′
/j)

^
scale′
j(e′
/j)
)



= (
E[ ˜E′
i] −g
loc′
i(e′
/i)

^
scale′
i(e′
/i)
)(
E[ ˜E′
j] −g
loc′
j(e′
/j)

^
scale′
j(e′
/j)
)

→E[ ˜Z′
i ˜
Z′
j|E′] = E[ ˜Z′
i|E′]E[ ˜
Z′
j|E′]

→E[ ˜Z′
i ˜
Z′
j|E′] −E[ ˜Z′
i|E]E[ ˜
Z′
j|E′] = 0

→E[ ˜Z′
i ˜
Z′
j|E′] −E[ ˜Z′
i|E′]E[ ˜
Z′
j|E′] −E[ ˜Z′
i|E′]E[ ˜
Z′
j|E′] + E[ ˜Z′
i|E′]E[ ˜
Z′
j|E′] = 0

→E[ ˜Z′
i ˜
Z′
j|E′] −E[ ˜
Z′
jE[ ˜Z′
i|E′]|E′] −E[ ˜Z′
iE[ ˜
Z′
j|E′]|E′] + E[ ˜Z′
i|E′]E[ ˜
Z′
j|E′] = 0

→E
h
( ˜Z′
i −E[ ˜Z′
i|E′])( ˜
Z′
j −E[ ˜
Z′
j|E′])|E′i
= 0

→cov( ˜Z′
i, ˜
Z′
j|E′) = 0

Typically, the aforementioned equalities would be valid for any diffeomorphic solution function
550

˜si : ˜Ei →˜Zi. However, in this paper, we specifically focus on solution functions represented by a
551

location-scale noise model.
552

Assuming that the causal variables follow a multivariate normal distribution conditional on the
553

pre-intervention exogenous variables, cov( ˜Z′
i, ˜
Z′
j|E′) = 0 would imply that ˜Z′
i ⊥⊥˜
Z′
j|E′. Let’s
554

define ϕE = g′−1 ◦g : E →E′ where g and g′ are the decoders in M and M′. As stated in
555

Assumption 3.1, the decoders are diffeomorphism, hence, ϕE is a diffeomorphism. Furthermore, let’s
556

denote ˜s as the set of all solution functions in post-intervention which are also diffeomorphism as
557

stated in Assumption 3.1. Consequently:
558

(ϕ−1
E
is diffeomorphic) ∀i, j, i ̸= j
˜Z′
i ⊥⊥˜
Z′
j|E′ →˜Z′
i ⊥⊥˜
Z′
j|ϕ−1
E (E′) →˜Z′
i ⊥⊥˜
Z′
j|E

→p( ˜Z′
i|E)p( ˜
Z′
j|E) = p( ˜Z′
i, ˜
Z′
j|E)

(all functions in ˜s are diffeomorphism) →p( ˜Z′
i|˜s(E))p( ˜
Z′
j|˜s(E)) = p( ˜Z′
i, ˜
Z′
j|˜s(E))

→p( ˜Z′
i| ˜Z)p( ˜
Z′
j| ˜Z) = p( ˜Z′
i, ˜
Z′
j| ˜Z)

The association between ˜Z′ and ˜Z arises from their shared observation space. We know that every
559

causal variable in M′ depends at least on one of the causal variables in M. If one of the causal
560

variables in M′ depended on more than one causal variable in M, it would create dependency
561

between two variables in M′ and violate the above equality. Therefore, no variable in M′ depends
562

on more than one causal variable in M. Consequently, the transformation ϕZ is a component-wise
563

transformation.
564

Theorem A1.6. (Identifiability of latent causal models.)
Let M = (A, X, g, I) and M′ =
565

(A′, X, g′, I) be two LCMs with shared observation space X and shared intervention targets I.
566

Suppose the following conditions are satisfied:
567

1. Identical correspondence assumptions explained in 3.1.
568

2. Soft interventions satisfy Assumption 3.3.
569

3. The causal and exogenous variables are real-valued.
570

4. The causal and exogenous variables follow a multivariate normal distribution.
571

Then the following statements are equivalent:
572

-Two LCMs M and M′ assign the same likelihood to interventional and observational data i.e.,
573

13


---Page Break---
pX
M(x, ˜x) = pX ′
M′(x, ˜x).
574

- M and M′ are disentangled, that is M ∼r M′ according to Definition 3.4.
575

Proof We will proceed to prove the equivalence between statements 1 and 2 by showing the implica-
576

tion is true in each direction.
577

A1.1
M ∼r M′ ⇒pX
M(x, ˜x) = pX
M′(x, ˜x)
578

This direction is fairly straightforward. According to Definition 3.4, the fact that M ∼r M ′ implies
579

that ϕE is measure preserving. Therefore, pE
M′(e′, ˜e′) = (ϕE)∗pE
M(e, ˜e). Furthermore, considering
580

that ancestry is preserved, ϕZ is measure preserving, and that causal variables are obtained from their
581

ancestral exogenous variables in implicit models, we have pZ
M′(z′, ˜z′) = (ϕZ)∗pZ
M(z, ˜z). Since
582

models are trained to maximize the log likelihood of p(x, ˜x, ˜x −x) and the latent spaces in M
583

and M ′ have the same distribution, the decoders should yield the same observational distributions
584

pX
M(x, ˜x) = pX
M′(x, ˜x).
585

A1.2
pX
M(x, ˜x) = pX
M′(x, ˜x) ⇒M ∼r M′
586

Let’s define ϕE = g′−1 ◦g : E →E′. Since we can express e = s−1(z), we can now define ϕZ as
587

ϕZ = s′ ◦g′−1 ◦g ◦s−1 : Z →Z′.
(8)

Therefore, ϕE = s′−1◦ϕZ◦s. Because g and g′ are diffeomorphisms, ϕE is a diffeomorphism as well.
588

Furthermore, since pX
M = pX
M′ and ϕE is a diffeomorphism, then pE
M′ = (ϕE)∗pE
M. Consequently,
589

ϕE is measure-preserving. Similarly, ϕE is measure-preserving as well since causal mechanisms are
590

diffeomorphisms.
591

Step 1: Identical correspondence of edges and nodes Let’s define the set U as U = {E ×E|∀I, J ∈
592

I : supp pE,I
M (e, ˜e|I) ∩supp pE,I
M (e, ˜e|J)}. Then, assuming atomic interventions and counterfac-
593

tual exogenous variables, pE,I
M (U|I) = pE,I
M (U|J) = 0. Therefore, we can say that pE
M(e, ˜e) =
594
P

I∈I pE,I
M (e, ˜e|I)pI
M(I) is a discrete mixture of non-overlapping distributions pE,I
M (e, ˜e|I). Sim-
595

ilarly, we can say that pE
M′(e, ˜e) is a discrete mixture of non-overlapping distributions. It can be
596

concluded that as ϕE must map between these distributions, there exists a bijection that also induces
597

a permutation ψ : [n] →[n]. Note: If we had non-atomic interventions or non-counterfactual exoge-
598

nous variables, then these distributions would have some overlapping. With overlapping distributions,
599

we can no longer claim there is a bijection mapping between these distributions.
600

In space Z, the interventions should also be sufficiently variable in order to have non-overlapping
601

pZ,I
M (z, ˜z|I) distributions. In the case of soft interventions, ˜z is affected by all ancestral exogenous
602

variables which could be ancestors of other causal variables as well. Consequently, if the changes in
603

causal mechanisms are not sufficient, the effect of ancestral exogenous variables on causal variables
604

will share some similarities and create overlapping distributions. Similar to pE
M(e, ˜e|I), we can say
605

that there is a permutation between pZ
M(z, ˜z|I) as well. Furthermore, as we assume the target of
606

interventions are known we have:
607

∀I ∈I : pZ
M(z, ˜z|I) = pZ
M′(z, ˜z|I)
(9)

Consequently, the permutation ψ is an identity transformation. The effect of soft intervention with
608

known targets on these conditional distributions is shown in Figure A1.
609

Step 2: Component-wise ϕZ
610

According to Lemma A1.5, in order to prove that ϕZ is a component-wise transformation, we need
611

to prove that ˜E′
i and ˜E′
j are independent ∀i, j, i ̸= j. In implicit modeling we do not know the parents
612

of each causal variable, hence, we assume the distribution of ˜Z′
i to be conditioned only on E′
i as in
613

Equation 5 since E′
i is a known parent of ˜Z′
i. The mean of a conditional distribution can be calculated
614

as:
615

E[˜z′
i|e′
i] = µ˜z′
i + ρσ˜z′
i
σe′
i
(e′
i −µe′
i)
(10)

where ρ and σ are the correlation coefficient and variance of the random variables, respectively. On
616

the other hand, we model ˜Z′
i using switch mechanisms as:
617

14


---Page Break---
0.6 0.4 0.20.0 0.2 0.4 0.6 0.8 1.0

X1

0.2

0.0

0.2

0.4

0.6

0.8

1.0

1.2

X2

Observed samples

Intervention on E1
Intervention on E2

Z1

10 0
10
20
30
40

Z2

10

0

10

20

30

40

P

0.00
0.01

0.02

0.03

0.04

0.05
M
M'

Pre-Intervention

(a)
(b)

Z1

10 0
10
20
30
40

Z2

10

0

10

20

30

40

P

0.00
0.01

0.02

0.03

0.04

0.05

M
M'

Intervention on Z1

Z1

10 0
10
20
30
40

Z2

10

0

10

20

30

40

P

0.00
0.01

0.02

0.03

0.04

0.05

M
M'

Intervention on Z2

(c)
(d)

Figure A1: The distribution of observed and causal variables in two causal models M and M′,
which belong to the equivalence class up to reparameterization. (a) There are 10 observed samples
in which Z1 or Z2 has been intervened on. (b) The distribution of causal variables when I = 0 (no
intervention) is identical to each other but the range of value of causal variables are different and can
be mapped to each other using ϕZ. (c) The intervention on Z1 (I = 1). (d) The intervention on Z2
(I = 2). For I = 1 and I = 2 the distributions are again identical to each other but are different for
different targets of intervention as soft interventions change the conditional distribution (condition on
parents) of causal variables. Also, for each value of I, the distributions of M and M′ should move
in one direction as targets are known.

˜z′
i = si(˜e′
i; e′
/i, h(v′))

By using Taylor’s expansion we can write above equation as:
618

si(˜e′
i; e′
/i, hi(v′)) = si(˜e′
i; e′
/i, hi(v′
0)) + +

∞
X

n=1

1
n!

 
∂nsi

∂hn
i


hi=hi(v′
0)
(hi(v′) −hi(v′
0))n
!

= si(˜e′
i; e′
/i, hi(v′
0)) + Ri

Furthermore, we assume separable dependence such that:
619

∃v′
0 such that ∀i
si(˜e′
i; e′
/i, hi(v′
0)) = si(˜e′
i; e′
/i)

An example of such a scenario could be in location-scale noise models, where a soft intervention
620

changes the location parameter of the model as:
621

si(e′
i; e′
/i) = e′
i + loc(e′
/i) →˜si(˜e′
i; e′
/i) = si(˜e′
i; e′
/i, hi(v′))

= ˜e′
i + loc(e′
/i) + hi(v′) = ˜e′
i + loc(e′
/i) + v′2 + v′

15


---Page Break---
In this example, for v′
0 = 0, si(˜e′
i; e′
/i, hi(v′
0)) = si(˜e′
i; e′
/i).
622

Consequently, we can write the following equality from Equation 10:
623

E[ ˜Z′
i|e′
i] = E[si( ˜E′
i; E′
/i) + Ri|e′
i] = µ ˜
Z′
i + ρ
σ ˜
Z′
i
σE′
i
(e′
i −µE′
i)

By taking the partial derivative of both side with respect to ˜E′
j we have:
624

∀j ̸= i
E[
∂si( ˜E′
i; E′
/i)

∂˜E′
i
· ∂˜E′
i
∂˜E′
j
+
∂si( ˜E′
i; E′
/i)

∂E′
/i
·
∂E′
/i
∂˜E′
j
+ ∂Ri

∂˜E′
j
|e′
i] = 0

If we did not have the causal mechanism switch variable (hi(V′)), the equation above would only
625

hold if si was constant in parents, which is not the case due to the presence of soft interventions, or if
626

∂si( ˜E′
i;E′
/i)

∂˜E′
i
· ∂˜E′
i
∂˜
E′
j = −
∂si( ˜E′
i;E′
/i)
∂E′
/i
·
∂E′
/i
∂˜
E′
j . The latter scenario would imply that ∂˜
E′
i
∂˜
E′
j ̸= 0, hence, ˜E′
i ̸⊥⊥˜E′
j.
627

However, by introducing the causal mechanism switch variable V and assuming it is observed, we
628

can account for the effects of soft interventions through hi(V′). In this case, ∂˜
E′
i
∂˜
E′
j = 0 as exogenous
629

variables are commonly assumed to be independent in practice. Consequently:
630

∀i, j
˜E′
i ⊥⊥˜E′
j

→∀i, j
p( ˜Z′
i, ˜
Z′
j| ˜Zi, ˜
Zj) = p( ˜Z′
i| ˜Zi)p( ˜
Z′
j| ˜
Zj)

→ϕZ is a component-wise transformation.

𝑧̃!

"

𝑧̃!

𝑧

𝑧̃!

"

𝑧
𝑰
𝑰

𝜑!!

𝑧̃#"

=

𝑧̃#"

𝜑!"

𝑓"

𝑓#

𝑓#$

𝜑!

𝑓"

$

𝑽
𝑽"

(a)

𝑧̃!

"

𝑧̃!

𝑧

𝑧̃!

"

𝑰
𝑰

𝜑!!

=

𝑓"

𝑓#
𝑓"

$

(b)

Figure A2: (a) String diagram of the causal variables Z and Z′. The triangle indicates sampling I
from its distribution. The left-hand side diagram is when ϕZ is applied last and the right-hand side
diagram is when ϕZ is applied first. I is the intervention which affects intervened causal variable’s
mechanism variable. V is used to model the effect of intervention on mechanisms and parents. (b)
String diagrams after discarding ˜
Z′o and the disentangled effect of soft intervention on ˜Zi modeled by
V .

Step 3: Component-wise ϕE
631

Using the result from previous step that ϕZ is a component-wise transformation, the string diagrams
632

for connections between E and E′ will be as shown in Figure A3. ϕEi will only depend on EA,
633

where A = anci is the ancestors of variable i, and ei. Because s(e)anci, s(e)i, and s′−1(z′)i only
634

depend on ancestors and ϕZ is a component-wise transformation. The first equality in Figure A3
635

follows from the definition of ϕEi. The second equality holds when we first apply ϕZA and then apply
636

the causal mechanisms. It can be concluded from the most right-hand side diagram in Figure A3
637

that the transformation from E′
i × EA →E′
i is constant in EA. Therefore, ϕEi is a component-wise
638

transformation.
639

16


---Page Break---
𝑓!

"#$

𝑓!

𝑠"

𝑓!

"#$

𝑓!

#

𝑠"

=
=

𝜑$!

𝜀!

𝜀"

𝜀"

#

𝜀!
𝜀"

𝑧!

𝑧"

𝜑%"
𝜑%!

𝑧"

#
𝑧!

#

𝜀"

#
𝜀"

#

𝑧"

#

𝑧!

#

𝜀"

#

𝜑%" 
𝑧!

𝜀!

Figure A3: String diagrams for connections between E and E′. The triangle indicates sampling
variables from their corresponding distributions.

(a) Pre-Epic-Kitchens
(b) Pre-Epic-Kitchens
(c) Pre-Epic-Kitchens
(d) Pre-Epic-Kitchens

(a) Post: Valve-locked
(b) Post: Bread-Inserted
(c)
Post: Clothes-Gathered
(d) Post: Juice-Poured

(e) Pre-ProcTHOR
(f) Pre-ProcTHOR
(g) Pre-ProcTHOR
(h) Pre-ProcTHOR

(e) Post: Cabinet-Open
(f) Post: Box-Open
(g) Post: TV-Broken
(h) Post: TV-On

Figure A4: In the Causal-Triplet dataset [19], visual representations capture both pre and post-
intervention scenarios. The first two rows showcase data samples from Epic-Kitchens, while the third
and fourth rows feature samples from ProcTHOR. Each image in the post-intervention condition
is accompanied by labels specifying the corresponding action and intervened object. In the images
in the first two rows, the agent is performing an action on an object but the camera angle has also
changed. So we can say that for example the distribution of causal variables conditioned on the
camera angle has been changed due to soft intervention.

A2
Soft vs. Hard intervention
640

In a causal model, an intervention refers to a deliberate action taken to manipulate or change one or
641

more variables in order to observe its impact on other variables within the causal model. Interventions
642

help to study how changes in one variable directly cause changes in another, thereby revealing causal
643

relationships.
644

17


---Page Break---
𝑧̃!

𝑰

𝑧̃"

𝑠!

𝑒/!

𝑠̃"

(a)

𝑧̃!

𝑒/"
𝑰

𝑠̃!
𝑠"

𝑧̃$

(b)

Figure A5: Causal graph models in the presence of Hard (a) and Soft (b) interventions. There are no
connections from parents to ˜Zi in hard interventions (a). Whereas, parents are connected to ˜Zi in soft
interventions (b).Let’s consider an implicit model and use /i to denote all variables except variable
i. The major difference of soft intervention (b) with hard intervention (a) is that ˜Zi is no longer
disconnected from its parents and its causal mechanism ˜si is affected by the intervention. Thus, with
a hard intervention, we know the post-intervention parents of a node ˜Zi (there are none), whereas
with soft interventions, the parents themselves may not change.

Based on the levels of control and manipulation in a causal intervention, we can have soft vs. hard
645

interventions. A hard intervention involves directly manipulating the variables of interest in a
646

controlled manner such as Randomized Controlled Trials (RCTs). In other words, a hard intervention
647

sets the value of a causal variable Z to a certain value denoted as do(Z = z) [24].
648

On the other hand, soft intervention involves more subtle or less controlled manipulation of variables
649

and changes the conditional distribution of the causal variable p(Z|Zpa) →˜p(Z|Zpa) which can be
650

modeled as ˜zi = ˜fi(zpai, ˜ei) [7].
651

Looking at interventions from a graphical standpoint, a hard intervention entails that the intervened
652

node is solely impacted by the intervention itself, with no influence coming from its ancestral nodes.
653

Conversely, in the context of a soft intervention, the representation of the intervened node can be
654

influenced not only by the intervention but also by its parent nodes.
655

As an example, suppose we are trying to understand the causal relationship between different types
656

of diets and weight loss. The soft intervention in this scenario could be a switch from a regular diet to
657

a low-carb diet. Switching to a low-carb diet is a voluntary choice made by the individual and there
658

are no external forces or regulations compelling them to make this change (non-coercive).
659

The intervention involves a modification of the individual’s diet rather than a complete disruption
660

since they are adjusting the proportion of macronutrients (fats, proteins, and carbs) they consume,
661

which is less disruptive than a radical change in eating habits (gradual modification). The individual
662

has autonomy to choose and tailor their diet according to their preferences and health goals so they
663

are empowered to make informed decisions about their dietary choices (behavioural empowerment).
664

Conversely, if the government or an authority were to intervene and enforce a mandatory low-carb
665

diet through legal means, this would constitute a hard intervention. In this scenario, regulations would
666

be implemented, prohibiting the consumption of specific carbohydrate-containing foods. Regulatory
667

agencies would be established to oversee and ensure adherence to the low-carb diet mandate, taking
668

actions such as removing prohibited foods from the market, restricting their import and production,
669

and so on. Individuals caught consuming banned foods would be subject to fines, legal repercussions,
670

or other penalties.
671

A3
Experiments
672

This section contains additional details about ICRL-SM design architectures, datasets, and experi-
673

ments settings.
674

A3.1
Datasets
675

A3.1.1
Synthetic
676

We generate simple synthetic datasets with X = Z = Rn. For each value of n, we generate ten
677

random DAGs, a random location-scale SCM, then a random dataset from the parameterized SCM.
678

To generate random DAGs, each edge is sampled in a fixed topological order from a Bernoulli
679

18


---Page Break---
distribution with probability 0.5. The pre-intervention and post-intervention causal variables are
680

obtained as:
681

zi = scale(zpai)ei + loc(zpai)
Soft-Intervention
−−−−−−−−−→˜zi = scale(zpai)˜ei + f
loc(zpai),
(11)

where the loc and scale networks are changed in post intervention. The pre-intervention loc and
682

post-intervention f
loc network weights are initialized with samples drawn from N(0, 1) and N(3, 1),
683

respectively. For ablation studies, we change the mean of these Normal distributions. The scale is
684

constant 1 for both pre-intervention and post-intervention samples. Both ei and ˜ei are sampled from
685

a standard Gaussian. The causal variables are mapped to the data space through a randomly sampled
686

SO(n) rotation. For each dataset, we generate 100,000 training samples, 10,000 validation samples,
687

and 10,000 test samples.
688

A3.1.2
Causal-Triplet
689

The Causal-Triplet datasets are consisted of images containing objects in which an action is manipu-
690

lating the objects shown in Figure A4. Examples of actions and objects in these datasets are given in
691

Table A1 and A2.
692

Table A1: Actions and objects present in the Causal-Triplet images (ProcTHOR Dataset).

ProcTHOR Dataset

Object
Television
Bed
Bed
Television
Laptop
Book
Box
Action
Break
Clean
Dirty
Turn off
Turn on
Open
Close

Table A2: Actions and objects present in the Causal-Triplet images (Epic-Kitchens Dataset).

Epic-Kitchens Dataset

Object
Tofu
Rice
Hob
Bag
Cupboard
Garlic
Tap
Wrap
Rice
Cheese
Action
Insert
Pour
Wash
Fold
Open
Pat
Move
Check
Transition
Stretch

Object
Wrap
Skin
Button
Lid
Plate
Egg
Sponge
Oil
Water
Dough
Action
Flip
Gather
Press
Lock
Wrap
Drop
Water
Carry
Smell
Mark

Based on the actions and objects, we treat our causal variables as attributes of objects which can be
693

changed by actions. Therefore, actions in these datasets are considered as interventions. Assume that
694

z1 corresponds to the attributes of an object, e.g. a door, the target of opening or closing (action’s
695

target) is z1.
696

We use actions’ labels in these datasets to detect the targets of interventions to determine which causal
697

variable has been intervened upon. Note that informing the model about the target of intervention is
698

not same as informing about the action itself (See Table 3). We use 5000 images of these datasets to
699

train all models.
700

A3.2
Architecture Design
701

Based on the ICRL-SM architecture depicted in Figure 2a, we devised a location-scale solution
702

function (Equation 6) in which the loci and scalei, and hi networks each comprise of fully connected
703

networks. These networks consist of two layers each, with 64 hidden units per layer and ReLU
704

activation functions. The encoder and decoder parameters for latents E and ˜E are shared and we use a
705

separate encoder and decoder with the same architecture for the latent V. For our synthetic dataset
706

experiments, the encoder and decoder are consisted of fully connected networks with 2 hidden layers
707

and 64 units in each hidden layer. For the Causal-Triplet datasets, we utilized ResNet-based networks.
708

The same encoder and decoder architectures are used for all baseline models in the experiments.
709

ResNet50 encoder, ResNet50 decoder, and classifiers with 1 hidden layer and 64 hidden units are
710

used for predicting actions and objects for experiments in Table 4 and Table 3. ResNet18 encoder,
711

ResNet18 decoder, and classifiers with 2 hidden layer and 2 hidden units are used for predicting
712

actions and objects for experiments in Table A4 and Table A3.
713

A3.3
Training
714

To enforce the condition described in Equation 5 for i /∈I, we assign the post-intervention exogenous
715

variables the same value as the pre-intervention exogenous variables. In mathematical terms, this
716

translates to ∀i /∈I, we set ˜ei = ei.
717

19


---Page Break---
In our experiments, we do not pretrain the networks, however, for the baseline models we follow the
718

training procedure in [3]. We also use consistency in our experiments to ensure that the encoder and
719

decoder are inverse of each other. Consistency regularizer is used as P

i Eˆx∼p(ˆx|e),x∼p(x)[(x −ˆx)2]
720

where ˆx are the reconstructed samples.
721

For optimization, Adam optimizer is used with default hyperparamters. In the synthetic experiments,
722

learning rate changes from 3e−4 to 1e−8 with a cosine scheduler. In the Causal-Triplet experiments
723

in Table 4 and Table 3 learning rate changes from 0.002 to 1e −8 with a cosine scheduler. For Table
724

A4 and Table A3 experiments earning rate changes from 0.0001 to 1e −8 with a cosine scheduler. In
725

all experiments the batch size is set to 64. In the main Causal-Triplet experiments we train the models
726

for 400 epochs, in the appendix Causal-Triplet experiments we train the models for 2000 epochs, and
727

in the synthetic experiments we train the models for 100 epochs. In the appendix experiments, the
728

graph parameters for explicit models are frozen after 1000 epochs.
729

All models are trained using Nvidia GeForce RTX4090 GPUs. Each of the Causal-Triplet experiments
730

takes 3-8 hours to train the models and each of the synthetic experiments takes 2-3 hours to train the
731

models.
732

We save the models’ weights with best validation loss and evaluate them using those weights with
733

test data.
734

A4
Ablation study
735

A4.1
Scalability
736

While our primary research objective centered on addressing identifiability challenges in implicit
737

causal models under soft interventions, we also conducted an investigation into the scalability of our
738

proposed model. To comprehensively assess its performance, we designed experiments covering a
739

range of causal graphs, featuring 5 to 10 variables, with 10 different seeds for each variable, following
740

a similar experimental setup as our 4-variable causal graph experiments. The outcomes of these
741

experiments, comparing ICRL-SM and ILCM, are presented in Figure A6. By increasing the number
742

of variables in the graph, confounding factors and ambiguities of causal relations increase as well.
743

Consequently, more supervision on V is required to better separate the effect of causal variables
744

themselves on the observed variables.
745

D4
D5
D6
D7
D8
D9
D10
Causal Variables

0.4

0.5

0.6

0.7

0.8

0.9

Causal Disentanglement Score

Means with Standard Deviations

ICRL-SM (mean)
ICRL-SM (std)
ILCM (mean)
ILCM (Std)

Figure A6: Causal disentanglement for different number of variables

A4.2
Backbone model
746

We trained the models using a simpler backbone model, ResNet18, to see how it affects performance.
747

The input image resolution is 64 × 64 and we use the intervened causal variables to predict action
748

20


---Page Break---
and object classes. The results are shown in Table A4 and A3. It can be seen from the results that the
749

proposed method outperforms other explicit and implicit models even with a simpler model.
750

Table A3: Table comparing action and object accuracy across various methods on Causal-Triplet
datasets using ResNet18 model.

Epic-Kitchens
ProcTHOR
Method
Action Accuracy
Object Accuracy
Action Accuracy
Object Accuracy

β −V AE [11]
0.15
0.04
0.20
0.36
d −V AE [21]
0.16
0.02
0.15
0.38
ILCM [3]
0.19
0.04
0.15
0.42
ICRL-SM (ours)
0.35
0.04
0.40
0.69

Table A4: Action and object accuracy of three explicit models are compared with ICRL-SM. Exper-
iments are conducted applying image with resolution of R64 as the input to the Resnet18 encoder
with the intervened casual variable (zi).

Datasets
Methods
Action Accuracy
Object Accuracy

Epic-Kitchens
ENCO [16]
0.14
0.03
DDS [5]
0.16
0.05
Fixed-order
0.14
0.05
ICRL-SM (ours)
0.35
0.04

ProcTHOR
ENCO [16]
0.16
0.28
DDS [5]
0.34
0.35
Fixed-order
0.34
0.38
ICRL-SM (ours)
0.40
0.69

21


---Page Break---
NeurIPS Paper Checklist
751

1. Claims
752

Question: Do the main claims made in the abstract and introduction accurately reflect the
753

paper’s contributions and scope?
754

Answer: [Yes]
755

Justification: Our contributions include identifiability of causal models with soft inter-
756

ventions. In the proposed methods section we give the theory and assumptions for the
757

identifiability result and in our experiments we evaluate our method using datasets generated
758

by soft interventions.
759

Guidelines:
760

• The answer NA means that the abstract and introduction do not include the claims
761

made in the paper.
762

• The abstract and/or introduction should clearly state the claims made, including the
763

contributions made in the paper and important assumptions and limitations. A No or
764

NA answer to this question will not be perceived well by the reviewers.
765

• The claims made should match theoretical and experimental results, and reflect how
766

much the results can be expected to generalize to other settings.
767

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
768

are not attained by the paper.
769

2. Limitations
770

Question: Does the paper discuss the limitations of the work performed by the authors?
771

Answer: [Yes]
772

Justification: We have some strict assumptions on data generation process and model
773

which are given in Assumptions 3.3 and 3.1 which may not be plausible to satisfy in some
774

applications.
775

Guidelines:
776

• The answer NA means that the paper has no limitation while the answer No means that
777

the paper has limitations, but those are not discussed in the paper.
778

• The authors are encouraged to create a separate "Limitations" section in their paper.
779

• The paper should point out any strong assumptions and how robust the results are to
780

violations of these assumptions (e.g., independence assumptions, noiseless settings,
781

model well-specification, asymptotic approximations only holding locally). The authors
782

should reflect on how these assumptions might be violated in practice and what the
783

implications would be.
784

• The authors should reflect on the scope of the claims made, e.g., if the approach was
785

only tested on a few datasets or with a few runs. In general, empirical results often
786

depend on implicit assumptions, which should be articulated.
787

• The authors should reflect on the factors that influence the performance of the approach.
788

For example, a facial recognition algorithm may perform poorly when image resolution
789

is low or images are taken in low lighting. Or a speech-to-text system might not be
790

used reliably to provide closed captions for online lectures because it fails to handle
791

technical jargon.
792

• The authors should discuss the computational efficiency of the proposed algorithms
793

and how they scale with dataset size.
794

• If applicable, the authors should discuss possible limitations of their approach to
795

address problems of privacy and fairness.
796

• While the authors might fear that complete honesty about limitations might be used by
797

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
798

limitations that aren’t acknowledged in the paper. The authors should use their best
799

judgment and recognize that individual actions in favor of transparency play an impor-
800

tant role in developing norms that preserve the integrity of the community. Reviewers
801

will be specifically instructed to not penalize honesty concerning limitations.
802

3. Theory Assumptions and Proofs
803

22


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
804

a complete (and correct) proof?
805

Answer: [Yes]
806

Justification: We give the full set of our assumptions in the proposed method section and the
807

detailed proof in Appendix A1.
808

Guidelines:
809

• The answer NA means that the paper does not include theoretical results.
810

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
811

referenced.
812

• All assumptions should be clearly stated or referenced in the statement of any theorems.
813

• The proofs can either appear in the main paper or the supplemental material, but if
814

they appear in the supplemental material, the authors are encouraged to provide a short
815

proof sketch to provide intuition.
816

• Inversely, any informal proof provided in the core of the paper should be complemented
817

by formal proofs provided in appendix or supplemental material.
818

• Theorems and Lemmas that the proof relies upon should be properly referenced.
819

4. Experimental Result Reproducibility
820

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
821

perimental results of the paper to the extent that it affects the main claims and/or conclusions
822

of the paper (regardless of whether the code and data are provided or not)?
823

Answer:[Yes]
824

Justification: We provide the full details of our model architecture and training settings in
825

Appendix A3 and in Section 5.
826

Guidelines:
827

• The answer NA means that the paper does not include experiments.
828

• If the paper includes experiments, a No answer to this question will not be perceived
829

well by the reviewers: Making the paper reproducible is important, regardless of
830

whether the code and data are provided or not.
831

• If the contribution is a dataset and/or model, the authors should describe the steps taken
832

to make their results reproducible or verifiable.
833

• Depending on the contribution, reproducibility can be accomplished in various ways.
834

For example, if the contribution is a novel architecture, describing the architecture fully
835

might suffice, or if the contribution is a specific model and empirical evaluation, it may
836

be necessary to either make it possible for others to replicate the model with the same
837

dataset, or provide access to the model. In general. releasing code and data is often
838

one good way to accomplish this, but reproducibility can also be provided via detailed
839

instructions for how to replicate the results, access to a hosted model (e.g., in the case
840

of a large language model), releasing of a model checkpoint, or other means that are
841

appropriate to the research performed.
842

• While NeurIPS does not require releasing code, the conference does require all submis-
843

sions to provide some reasonable avenue for reproducibility, which may depend on the
844

nature of the contribution. For example
845

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
846

to reproduce that algorithm.
847

(b) If the contribution is primarily a new model architecture, the paper should describe
848

the architecture clearly and fully.
849

(c) If the contribution is a new model (e.g., a large language model), then there should
850

either be a way to access this model for reproducing the results or a way to reproduce
851

the model (e.g., with an open-source dataset or instructions for how to construct
852

the dataset).
853

(d) We recognize that reproducibility may be tricky in some cases, in which case
854

authors are welcome to describe the particular way they provide for reproducibility.
855

In the case of closed-source models, it may be that access to the model is limited in
856

some way (e.g., to registered users), but it should be possible for other researchers
857

to have some path to reproducing or verifying the results.
858

23


---Page Break---
5. Open access to data and code
859

Question: Does the paper provide open access to the data and code, with sufficient instruc-
860

tions to faithfully reproduce the main experimental results, as described in supplemental
861

material?
862

Answer: [Yes]
863

Justification: We provide our anonymized codes which contains the necessary scripts and
864

instructions to run the experiments.
865

Guidelines:
866

• The answer NA means that paper does not include experiments requiring code.
867

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
868

public/guides/CodeSubmissionPolicy) for more details.
869

• While we encourage the release of code and data, we understand that this might not be
870

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
871

including code, unless this is central to the contribution (e.g., for a new open-source
872

benchmark).
873

• The instructions should contain the exact command and environment needed to run to
874

reproduce the results. See the NeurIPS code and data submission guidelines (https:
875

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
876

• The authors should provide instructions on data access and preparation, including how
877

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
878

• The authors should provide scripts to reproduce all experimental results for the new
879

proposed method and baselines. If only a subset of experiments are reproducible, they
880

should state which ones are omitted from the script and why.
881

• At submission time, to preserve anonymity, the authors should release anonymized
882

versions (if applicable).
883

• Providing as much information as possible in supplemental material (appended to the
884

paper) is recommended, but including URLs to data and code is permitted.
885

6. Experimental Setting/Details
886

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
887

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
888

results?
889

Answer: [Yes]
890

Justification: We provide the full details of our model architecture and training settings in
891

Appendix A3 and in Section 5.
892

Guidelines:
893

• The answer NA means that the paper does not include experiments.
894

• The experimental setting should be presented in the core of the paper to a level of detail
895

that is necessary to appreciate the results and make sense of them.
896

• The full details can be provided either with the code, in appendix, or as supplemental
897

material.
898

7. Experiment Statistical Significance
899

Question: Does the paper report error bars suitably and correctly defined or other appropriate
900

information about the statistical significance of the experiments?
901

Answer: [Yes]
902

Justification: In our synthetic experiments we initialized the causal graph in the dataests
903

with different seeds. The results of these different seeds are provided in Table 2 and Figure
904

A6.
905

Guidelines:
906

• The answer NA means that the paper does not include experiments.
907

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
908

dence intervals, or statistical significance tests, at least for the experiments that support
909

the main claims of the paper.
910

24


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
911

example, train/test split, initialization, random drawing of some parameter, or overall
912

run with given experimental conditions).
913

• The method for calculating the error bars should be explained (closed form formula,
914

call to a library function, bootstrap, etc.)
915

• The assumptions made should be given (e.g., Normally distributed errors).
916

• It should be clear whether the error bar is the standard deviation or the standard error
917

of the mean.
918

• It is OK to report 1-sigma error bars, but one should state it. The authors should
919

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
920

of Normality of errors is not verified.
921

• For asymmetric distributions, the authors should be careful not to show in tables or
922

figures symmetric error bars that would yield results that are out of range (e.g. negative
923

error rates).
924

• If error bars are reported in tables or plots, The authors should explain in the text how
925

they were calculated and reference the corresponding figures or tables in the text.
926

8. Experiments Compute Resources
927

Question: For each experiment, does the paper provide sufficient information on the com-
928

puter resources (type of compute workers, memory, time of execution) needed to reproduce
929

the experiments?
930

Answer: [Yes]
931

Justification: The details are given in Appendix A3.
932

Guidelines:
933

• The answer NA means that the paper does not include experiments.
934

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
935

or cloud provider, including relevant memory and storage.
936

• The paper should provide the amount of compute required for each of the individual
937

experimental runs as well as estimate the total compute.
938

• The paper should disclose whether the full research project required more compute
939

than the experiments reported in the paper (e.g., preliminary or failed experiments that
940

didn’t make it into the paper).
941

9. Code Of Ethics
942

Question: Does the research conducted in the paper conform, in every respect, with the
943

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
944

Answer: [Yes]
945

Justification:
946

Guidelines:
947

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
948

• If the authors answer No, they should explain the special circumstances that require a
949

deviation from the Code of Ethics.
950

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
951

eration due to laws or regulations in their jurisdiction).
952

10. Broader Impacts
953

Question: Does the paper discuss both potential positive societal impacts and negative
954

societal impacts of the work performed?
955

Answer: [NA]
956

Justification:
957

Guidelines:
958

• The answer NA means that there is no societal impact of the work performed.
959

• If the authors answer NA or No, they should explain why their work has no societal
960

impact or why the paper does not address societal impact.
961

25


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
962

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
963

(e.g., deployment of technologies that could make decisions that unfairly impact specific
964

groups), privacy considerations, and security considerations.
965

• The conference expects that many papers will be foundational research and not tied
966

to particular applications, let alone deployments. However, if there is a direct path to
967

any negative applications, the authors should point it out. For example, it is legitimate
968

to point out that an improvement in the quality of generative models could be used to
969

generate deepfakes for disinformation. On the other hand, it is not needed to point out
970

that a generic algorithm for optimizing neural networks could enable people to train
971

models that generate Deepfakes faster.
972

• The authors should consider possible harms that could arise when the technology is
973

being used as intended and functioning correctly, harms that could arise when the
974

technology is being used as intended but gives incorrect results, and harms following
975

from (intentional or unintentional) misuse of the technology.
976

• If there are negative societal impacts, the authors could also discuss possible mitigation
977

strategies (e.g., gated release of models, providing defenses in addition to attacks,
978

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
979

feedback over time, improving the efficiency and accessibility of ML).
980

11. Safeguards
981

Question: Does the paper describe safeguards that have been put in place for responsible
982

release of data or models that have a high risk for misuse (e.g., pretrained language models,
983

image generators, or scraped datasets)?
984

Answer: [NA]
985

Justification:
986

Guidelines:
987

• The answer NA means that the paper poses no such risks.
988

• Released models that have a high risk for misuse or dual-use should be released with
989

necessary safeguards to allow for controlled use of the model, for example by requiring
990

that users adhere to usage guidelines or restrictions to access the model or implementing
991

safety filters.
992

• Datasets that have been scraped from the Internet could pose safety risks. The authors
993

should describe how they avoided releasing unsafe images.
994

• We recognize that providing effective safeguards is challenging, and many papers do
995

not require this, but we encourage authors to take this into account and make a best
996

faith effort.
997

12. Licenses for existing assets
998

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
999

the paper, properly credited and are the license and terms of use explicitly mentioned and
1000

properly respected?
1001

Answer: [Yes]
1002

Justification:
1003

Guidelines:
1004

• The answer NA means that the paper does not use existing assets.
1005

• The authors should cite the original paper that produced the code package or dataset.
1006

• The authors should state which version of the asset is used and, if possible, include a
1007

URL.
1008

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
1009

• For scraped data from a particular source (e.g., website), the copyright and terms of
1010

service of that source should be provided.
1011

• If assets are released, the license, copyright information, and terms of use in the
1012

package should be provided. For popular datasets, paperswithcode.com/datasets
1013

has curated licenses for some datasets. Their licensing guide can help determine the
1014

license of a dataset.
1015

26


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
1016

the derived asset (if it has changed) should be provided.
1017

• If this information is not available online, the authors are encouraged to reach out to
1018

the asset’s creators.
1019

13. New Assets
1020

Question: Are new assets introduced in the paper well documented and is the documentation
1021

provided alongside the assets?
1022

Answer: [Yes]
1023

Justification: We only have a code repository for replicating experiments and we have
1024

submitted the anonymized zip file with our submission.
1025

Guidelines:
1026

• The answer NA means that the paper does not release new assets.
1027

• Researchers should communicate the details of the dataset/code/model as part of their
1028

submissions via structured templates. This includes details about training, license,
1029

limitations, etc.
1030

• The paper should discuss whether and how consent was obtained from people whose
1031

asset is used.
1032

• At submission time, remember to anonymize your assets (if applicable). You can either
1033

create an anonymized URL or include an anonymized zip file.
1034

14. Crowdsourcing and Research with Human Subjects
1035

Question: For crowdsourcing experiments and research with human subjects, does the paper
1036

include the full text of instructions given to participants and screenshots, if applicable, as
1037

well as details about compensation (if any)?
1038

Answer: [NA]
1039

Justification:
1040

Guidelines:
1041

• The answer NA means that the paper does not involve crowdsourcing nor research with
1042

human subjects.
1043

• Including this information in the supplemental material is fine, but if the main contribu-
1044

tion of the paper involves human subjects, then as much detail as possible should be
1045

included in the main paper.
1046

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1047

or other labor should be paid at least the minimum wage in the country of the data
1048

collector.
1049

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1050

Subjects
1051

Question: Does the paper describe potential risks incurred by study participants, whether
1052

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1053

approvals (or an equivalent approval/review based on the requirements of your country or
1054

institution) were obtained?
1055

Answer: [NA]
1056

Justification:
1057

Guidelines:
1058

• The answer NA means that the paper does not involve crowdsourcing nor research with
1059

human subjects.
1060

• Depending on the country in which research is conducted, IRB approval (or equivalent)
1061

may be required for any human subjects research. If you obtained IRB approval, you
1062

should clearly state this in the paper.
1063

• We recognize that the procedures for this may vary significantly between institutions
1064

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1065

guidelines for their institution.
1066

• For initial submissions, do not include any information that would break anonymity (if
1067

applicable), such as the institution conducting the review.
1068

27


---Page Break---
