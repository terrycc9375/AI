Formalising Anti-Discrimination Law
in Automated Decision Systems

Anonymous Author(s)
Affiliation
Address
email

Abstract

We study the legal challenges in automated decision-making by analysing conven-
1

tional algorithmic fairness approaches and their alignment with anti-discrimination
2

law in the United Kingdom and other jurisdictions based on English common law.
3

By translating principles of anti-discrimination law into a decision-theoretic frame-
4

work, we formalise discrimination and propose a new, legally informed approach
5

to developing systems for automated decision-making. Our investigation reveals
6

that while algorithmic fairness approaches have adapted concepts from legal theory,
7

they can conflict with legal standards, highlighting the importance of bridging the
8

gap between automated decisions, fairness, and anti-discrimination doctrine.
9

1
Introduction
10

Automated decision-making using predictive models is becoming increasingly important in many
11

areas of society, including lending [60, 100, 107], criminal justice [31, 14, 151], hiring [64, 59, 25],
12

and welfare eligibility [41, 56, 113]. Instances of large-scale failures, from disproportionately harming
13

vulnerable people in welfare eligibility assessments [113] to bias in consumer lending [80], highlight
14

the need for lawful implementation. Scrutiny of ML-based decisions is heightened by concerns about
15

replicating human biases and historical inequality [97, 41, 93].
16

Concerns about algorithmic bias have spurred research into fair ML. Early discourse on fairness
17

in ML was relatively narrow due to technical constraints [29, 61]. More recently, researchers have
18

developed formal definitions of fairness in algorithmic decisions and methods to measure fairness
19

in predictive models [49, 31, 27, 147, 87, 85]. Algorithmic fairness definitions generally measure
20

prediction disparities across groups with different legally protected characteristics [90, 136, 87, 17].
21

This research has resulted in several proposals, including statistical metrics to assess the fairness of
22

individual predictive models [136, 111, 22, 24], fairness for model auditing [70, 103, 63, 89, 98], and
23

fairness constraints on models [31, 148, 50, 145, 12].
24

These criteria simplify fairness into measurements of disparity that do not inherently map to unlawful
25

discrimination. The usefulness of these metrics in practice is limited as incomplete or even irrelevant
26

measures for legal investigations. There have been important efforts to bridge the gap between legal
27

and technical approaches to fair ML [81, 55, 51, 144, 139, 1, 46]. Lawyers have highlighted the
28

challenges of the narrow construction of fairness metrics focusing on disparity in predictions rather
29

than more nuanced definitions of discriminatory conduct and the broader context of the automated
30

decision-making process [55, 51, 144, 1]. We aim to contextualise and formalise legal concepts of
31

algorithmic discrimination beyond the narrow construction of statistical disparity.
32

The predominance of US analysis of fairness and discrimination in ML, lack of non-US ML
33

datasets [78], and the limited legal scholarship translating these concepts, has inadvertently fos-
34

tered a series of misconceptions that pervade the field. However, very few papers have engaged with
35

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
anti-discrimination laws outside of the United States [143, 139, 1, 140, 67, 76]. We aim to introduce
36

new principles and methods to deal with the issues identified in this literature. By avoiding the
37

nuanced legal realities of other jurisdictions, models designed to comply with US laws may breach
38

UK laws or those in comparable jurisdictions. Our paper addresses this gap by providing a rigorous
39

analysis of UK discrimination law, correcting some mischaracterisations, and establishing a more
40

accurate foundation for developing fair ML in the UK and its related jurisdictions.
41

1.1
Automated Decision-Making
42

Let xi ∈Rp be a vector of observed attributes for individual i. A decision-maker must choose a
43

decision a ∈A, where A is closed. Further, we assume that the decision-maker wants to decide
44

based on a future outcome yi ∈Y for individual i. Here, we assume Y = N, which can be relaxed.
45

Decision-making under uncertainty has long been studied in statistical decision theory [108, 32, 13,
46

99]. Let u(y, a) be a utility function that summarises the utility for the decision-maker. The optimal
47

decision is then
48

a⋆= arg max
a∈A

X

y∈Y
u(a, y)p(y|a) .
(1)

The decision-maker usually neither knows yi nor p(y|a) at the time of the decision. Hence, the
49

decision must be based solely on xi. In an SML setting, a prediction model ˆp(y|x) is trained to
50

compute the predicted probability distribution (pmf) ˆπi = ˆp(y|xi) for individual i, with the support
51

on Y. Further, let ˆy(ˆπi) ∈Y be the classification made based on ˆπi. In simple settings, the decision
52

can be formulated as a decision function d(ˆπi) ∈A that is used to choose an appropriate action
53

based on ˆπi. In the binary y and a case, it reduces to a simple threshold τ, i.e., d(ˆπ) = I(ˆπ ≤τ),
54

where I is the indicator function and ˆπi = ˆp(y = 1 | xi). We often train a model ˆp(y|x) based
55

on previous data D = (y, X), drawn from a population p(y, x), where both xi and yi are known.
56

Replacing p(y|xi) with the predictive model ˆp(y|xi) in Eq. 1 gives an optimal decision.
57

1.2
Algorithmic Fairness
58

To define algorithmic fairness, we separate xi into protected and legitimate features xi = (xpi, xli);
59

we drop i to simplify notation. Here, xp ∈C indicates protected attributes, with C being the set of
60

different groups. Legally protected characteristics commonly identified in datasets include gender,
61

race, and age. Many fairness metrics aim to evaluate the fairness of an SML model for commonly
62

identified protected characteristics in datasets, including gender and race [49, 31, 82, 85].
63

Statistical parity, or demographic parity, is one of the central algorithmic fairness metrics [31, 136,
64

90, 82]. For statistical parity to hold, it requires that
65

Ex [ˆp(y|x) | xp] = Ex [ˆp(y|x)] ,
(2)

such that the model predictions, in expectation over x, need to be the same for the different groups [31,
66

136]. Given that the decision function d(π) is the same for the different groups, statistical parity results
67

in equal decisions for the different groups. However, we discuss later in this paper that, in practice,
68

statistical parity may exacerbate inequality or even result in unlawful discrimination [10, 74, 63].
69

Conditional statistical parity extends statistical parity to account for legitimate features xl. The
70

model predictions should only differ across protected groups to the extent that the difference is
71

conditional on legitimate factors [31, 136, 23]. This can be formalised as,
72

Ex [ˆp(y|x) | xl, xp] = Ex [ˆp(y|x) | xl] ,
(3)

so that, conditional on legitimate features xl, there should not be any difference in predictions between
73

groups given by the protected attribute. Below, we discuss the legitimacy of variables that correlate
74

to protected attributes [34, 77].
75

Other similar group comparison metrics have been proposed, such as error parity, balanced clas-
76

sification rate, and equalised odds [49, 31, 136, 90, 82, 30]. Also, more individual approaches
77

to parity have considered whether otherwise identical individuals are treated differently if they
78

have different protected attributes [34, 68]. Finally, ideas from causal inference and counterfactual
79

analysis have also been proposed to measure outcome consistency for individuals across protected
80

groups [75, 69, 106, 149, 26, 142, 92, 6]
81

2


---Page Break---
1.3
Anti-Discrimination Law
82

The algorithmic fairness literature largely identifies statistical disparities in predicted outcomes for
83

binary marginalised groups. Legally, discrimination is both broader and more detailed. Not all
84

actions perceived as discriminatory are unlawful, and some non-obvious actions may be prohibited.
85

Anti-discrimination law only applies to select duty-bearers in certain conditions [65]. Individuals’
86

friendship choices being based on race are not legally regulated, despite sometimes seeming unfair [36,
87

65]. It only applies to protected attributes. An algorithm that rejects a loan application because the
88

applicant uses an Android phone rather than an iOS device may seem unfair because it does not
89

reflect the true default risk but is a proxy for the applicant’s income [2, 76]. However, in isolation, this
90

would not be unlawful discrimination under UK law because poverty is not a protected attribute [96].
91

The prohibition on discrimination traces its legal roots to the Universal Declaration of Human Rights,
92

which established equality and freedom from discrimination as fundamental human rights, further
93

advanced in several international treaties [132, 88], and enacted as legislation worldwide spurred by
94

the Civil Rights Movement [86, 65]. The United Kingdom implemented several anti-discrimination
95

laws in the 20th century [118, 119, 116], which were consolidated in the Equality Act 2010 [40].
96

The Equality Act protects “age; disability; gender reassignment; marriage and civil partnership;
97

pregnancy and maternity; race; religion or belief; sex; sexual orientation” [40, s 4]. Algorithmic
98

fairness literature has often oversimplified these protected characteristics as simply identifying visible
99

traits when each has complex social meanings [57]. One complexity is, for example, the difference
100

between a person with a protected attribute by biological fact or by identifying with a protected
101

group [73]. UK anti-discrimination law distinguishes between direct discrimination and indirect
102

discrimination. While analogous to the US disparate treatment and disparate impact doctrine, there
103

are important distinctions, meaning they should not be so easily elided [1].
104

Direct discrimination occurs when an individual is treated less favourably than another based on
105

a protected characteristic [40, s 13]. To establish direct discrimination, it is necessary to identify
106

the specific protected characteristic involved, demonstrate the less favourable treatment (by real or
107

hypothetical comparison), and prove that this treatment was caused “but for” the protected attribute.
108

The intention of the decision-maker is not required or necessary [123, 131].
109

Indirect discrimination refers to a policy, criterion, or practice (PCP) that disproportionately
110

disadvantages a group with a particular protected attribute compared to those without [40, s 19]. To
111

prove indirect discrimination, one must identify such a PCP, show that it puts a group defined by its
112

protected attribute at a particular disadvantage compared to those without such attribute, and evaluate
113

whether it is justifiable as a proportionate means of achieving a legitimate aim.
114

English common law is either in force or is the dominant influence in 80 legal systems that govern
115

approximately 2.8 billion people, not including the US [28]. UK anti-discrimination law is very similar
116

to numerous Commonwealth and common law jurisdictions, including Australia [3], Canada [20],
117

India [47], New Zealand [91], South Africa [110], and the pending bill in Bangladesh [9]. European
118

Union law also has broadly the same principles and discrimination case law evolved in parallel during
119

the UK’s membership [45]. It is increasingly important to gain a nuanced understanding of unlawful
120

discrimination in AI systems as new laws aim to prevent future harms [44, 16].
121

1.4
Contributions and Limitations
122

This paper makes four core contributions at the intersection of automated decision-making, fairness,
123

and anti-discrimination doctrine.
124

1. We formalise critical aspects of anti-discrimination doctrine into decision-theoretic formalism.
125

2. We analyse the legal role of the data-generating process (DGP) and develop the DGP as a
126

theoretical framework to formalise the legitimacy of the prediction target y and the features x in
127

supervised models for automated decisions.
128

3. Further, we consider the legal and practical effects of approximating the DGP in supervised
129

models. We propose conditional estimation parity as a new, legally informed target.
130

4. Finally, we provide recommendations on creating SML models that minimise the risk of unlawful
131

discrimination in automated decision-making.
132

Our paper is formally limited to analysing and providing novel recommendations for the UK. While
133

we discuss related jurisdictions that are functionally similar and based on English common law,
134

3


---Page Break---
specific legal advice should be followed with respect to different jurisdictions. Accountability varies
135

by jurisdiction and context, which is why our paper underscores the importance of careful, informed
136

classification by experts with appropriate legal advice.
137

2
Automated Decisions and Discrimination
138

2.1
Legitimacy of True Differences
139

In SML, it is crucial to differentiate unlawful discrimination from mere statistical disparities and con-
140

cepts of algorithmic fairness. While formal equality may map to statistical parity, anti-discrimination
141

laws in the UK and related jurisdictions aim to achieve substantive equality. Despite the general
142

rule that individuals should not receive less favourable treatment based on their protected attributes,
143

courts acknowledge that treating all groups the same can actually disadvantage a protected group
144

and minimise important structural and true differences [137, 143]. Therefore, substantive equal-
145

ity may sometimes require legitimate differential treatment because of the true differences among
146

individuals [137, 52, 144, 139].
147

For instance, insurance decisions that might otherwise be construed as discriminatory – specifically
148

concerning gender reassignment, marriage, civil partnership, pregnancy, and sex discrimination –
149

are permissible if they are based on reliable actuarial data and executed reasonably [40, Sch 9. s
150

20]. Financial services can also “use age as a criterion for pricing risk, as it is a key risk factor
151

associated with for example, medical conditions, ability to drive, likelihood of making an insurance
152

claim and the ability to repay a loan” [117, para. 7.6]. These exemptions highlight legal recognition
153

that certain group distinctions, particularly those involving risk assessment, are relevant and necessary
154

for the equitable operation of such services. Similar statutory exemptions are found in other similar
155

anti-discrimination laws, including the European Union [43, art 2], Australia [8, s 30-47], Canada
156

[20, s 15], New Zealand [91, s 24-60] and South Africa [110, s 14].
157

2.2
True Data Generating Process
158

Therefore, an important aspect from the legal perspective that is overlooked in the existing literature
159

is the distinction between a “true data-generating” process (DGP) and the estimated model ˆp(y|x).
160

To formalise, we assume that there exists a true DGP, D ∼p(y, x), where Di = (yi, xi). Further, we
161

use p(y|xtrue
i
) to denote the true probability (pmf) for individual i, given the true features xtrue
i
.
162

We make multiple observations on the role of the “true” model and its use in connecting predictive
163

modelling and legal reasoning.
164

First, understanding the limits of predictive models is crucial to explore inherent uncertainties and
165

limitations in predictions. The true model is, in practice, never observed or known. When developing
166

ˆp(y|x), the target is often to select the model with the best predictive performance, which is closely
167

connected to the role of the true DGP [15, 134, 135, 133]. For this reason, the “true” model may
168

include features in xtrue
i
that are not observed in the data, sometimes referred to as an M-open setting
169

when the “true” model is not included in the set of candidate models [15, 135].
170

Second, we assume that p(y|xi) is a probability distribution over Y, introducing some level of
171

aleatoric uncertainty in the true underlying process [95, 58, 114]. This means that perfect prediction
172

of yi may not be possible, even with knowledge of the true DGP. The distinction between aleatoric
173

and epistemic uncertainty is important from a legal perspective. The reason is simple: the uncertainty
174

coming from estimation is the (legal) responsibility of the modeller, while the aleatoric uncertainty
175

can instead be considered a true underlying general risk.
176

Third, the true DGP connects to judicial legal reasoning. Courts must engage theoretically with
177

legal and normative conceptions of what is justifiable and what constitutes unlawful discrimination.
178

Judges consider legitimacy, proportionality, and necessity when evaluating actions, and hypothetical
179

alternatives, that led to less favourable treatment. Although, courts are not oracles. Discrimination
180

case law may not pinpoint what the perfect decision should have been. However, courts will engage
181

in a similar theoretical process of reasoning about the decision-making process to the true DGP to
182

understand whether the actions were justified or unlawful. We explain legal reasoning within this
183

framework throughout the paper and in a real-world case on unlawful discrimination in algorithmic
184

decision-making (see Appendix A).
185

4


---Page Break---
2.3
Estimation Parity
186

Legally, distinguishing between a true difference and an estimated one is important. We approximate
187

the true DGP with a model ˆp(y|x) based on training data when training an SML model. The
188

approximation introduces estimation error
189

ϵi = ˆπi −πi = ˆp(yi|xi) −p(yi|xtrue
i
) .
(4)

Algorithmic fairness literature often assumes the absence of estimation error [see e.g., 49] or assumes
190

that the true causal structure is known [150, 68, 26, 21]. In practice, this is rarely the case. Hence,
191

it is crucial, both practically and legally, to distinguish between the true underlying probabilities
192

πi and the estimated probabilities ˆπi. While the true underlying probability may sometimes be
193

defensible (Section 2.2), introducing an estimation error that disadvantages individuals based on
194

protected attributes invokes discrimination liability.
195

As the model will try to approximate the true data-generating process, modellers’ expectations are
196

difficult to ascertain. The law is unlikely to set a deterministic standard that any adverse effects of
197

estimation will make a modeller liable. The modeller should try to approximate the true model as
198

much as possible [see 4, 141, 135, 133, for discussions on model misspecification]. However, where
199

an estimation disparity reaches a threshold for discriminatory effects, the legal evaluation would
200

require analysing the steps taken to test and mitigate estimation disparity (even though the intent is
201

immaterial).
202

The potential bias in training data presents a risk that the estimation model will introduce bias against
203

individuals with protected attributes (Section 2.6). Historical discriminatory lending practices, for
204

example, could be perpetuated through biased training data [18, 104]. Such biased estimations
205

may introduce biased outcomes that are not reflective of true differences, potentially leading to
206

discriminatory outcomes. Therefore, we introduce “Conditional Estimation Parity” to formalise the
207

legal context of estimation.
208

Conditional Estimation Parity is the difference in estimation error between groups with a protected
209

attribute, given legitimate features, i.e.,
210

Ex[ϵ | xp, xl] = Ex[ϵ | xl] .
(5)

Reducing the error in Eq. 4 is expected to diminish the risk of conditional estimation disparity.
211

However, assessing conditional estimation parity is complex due to inherent challenges in evaluating
212

estimation error.
213

It is crucial to examine both mathematical and legal causal theories of why certain differences are
214

legitimate bases to make classification distinctions [71]. We examine the mathematical basis for
215

identifying statistical disparities in the context of unlawful discrimination. In Section 2.5, 2.7, and 2.6
216

we consider the causal relationships between legitimate differentiation and unlawful discrimination.
217

2.4
Statistical Disparities and Prima Facie Discrimination
218

To initiate a claim for discrimination, a claimant must establish a prima facie case [37, 40, s 136].
219

Sufficient evidence must be produced to show that unlawful discrimination may have occurred,
220

including by showing discriminatory effects or harm against an individual or group caused by the
221

decision-maker’s action [37, 65]. Statistical evidence can be used to prove less favourable treatment
222

or particular disadvantage, but by design, it shows correlations, and “a correlation is not the same
223

as a causal link” [130, para. 28]. We explain the threshold for legal causation at the trial stage in
224

Section 2.5. Although, at this stage, a mere correlation between the adverse effect on the person
225

and the decision-maker’s action will suffice [65]. The size of the disparity is relevant. Smaller
226

disparities are less likely to trigger legal inquiry under anti-discrimination laws [127]. Courts will
227

compare statistical evidence showing the different effects and outcomes between a disadvantaged
228

group compared to a group without the protected attribute. The significance of the statistical disparity
229

hinges on the specifics of the case [127, 124]. The thresholds for statistical significance are flexible
230

and often resisted by courts to avoid excessive dependence on data [138]. The UK has specifically
231

avoided thresholds like those used to measure statistically significant disparity in the US [105, 10].
232

Statistical disparities, as identified through algorithmic fairness metrics, may indicate a reason to
233

consider whether discrimination has arisen. However, without taking context and potential true and
234

5


---Page Break---
legitimate differences into account, these disparities hold little legal weight (see Section 2.1). We can
235

formalise this as the legal target being to minimise the conditional estimation disparity
236

ω = ||Ex [ϵi | xl, xp] −Ex [ϵi | xl] ||2,
(6)

where || · ||2 is the euclidean norm. This target generalises the idea of minimising conditional
237

statistical parity. If we assume true conditional statistical parity, i.e.
238

Ex

p(yi|xtrue
i
) | xl, xp

= Ex

p(yi|xtrue
i
) | xl

,
(7)

then the target in Eq. 6 will be reduced to minimise the conditional statistical parity (see Eq. 3).
239

Although, this is only true as long as there are no true differences.
240

Hence, if true statistical parity does not hold, it is explained by true differences between groups. If
241

there is a true difference, such as age in financial services, forcing conditional statistical parity would
242

harm the protected group, most likely resulting in unlawful discrimination. This result aligns with
243

previous observations about the risks of forcing parity metrics [31, 144, 54]. Courts may need to be
244

more flexible in the type of statistical data they consider to establish a prima facie case by considering
245

non-comparative adverse effects in their assessment. Therefore, deferring to conditional estimation
246

parity provides an avenue for a contextually informed assessment.
247

2.5
Legal Causation and the Utility Function
248

To lawyers, causation is the relationship between an act, i.e., an action or decision, and its effect,
249

which requires two questions: (1) factually, but for the act, would the consequences have occurred;
250

(2) is the act a substantial cause of the consequence to apply responsibility. We are concerned with
251

the first question. Direct discrimination “requires a causal link between the less favourable treatment
252

and the protected characteristic”; indirect discrimination “requires a causal link between the PCP and
253

the particular disadvantage suffered by the group and individual” [130, para. 25]. In an algorithmic
254

context, this causal link requires asking whether i would have received the same action or decision
255

a, but for their protected attribute xp or the PCP that indirectly relates to their protected attribute
256

xp [122, 123]. For instance, whether an individual would have suffered the disadvantage but for the
257

protected attribute would be discriminatory regardless of the decision-maker’s intention [1]. This is a
258

notable distinction from certain aspects of US discrimination doctrine.
259

From a decision-theoretic perspective, the protected attribute xp can affect the decision a either
260

through the utility function u(a, y) or through the model ˆp(y|x). Discrimination may occur if
261

the utility function in Eq. 1 differs for different groups defined by the protected attribute. Such a
262

difference would mean that an individual or whole group with a protected attribute is treated less
263

favourably than those without a protected attribute given the same model ˆp(y|x). Such a difference in
264

the utility function would risk unlawful discrimination. Specifically, if u(a, y) is changed for different
265

persons, either directly based on a protected attribute or indirectly has the effect of disproportionately
266

disadvantaging a group with a protected characteristic without justification (see Section 2.6).
267

Having different ˆp(y|x), on the other hand, would mean that there is a legal causation between the
268

decision a and xp. This might either be motivated by true differences (see Section 2.1) or a result of
269

conditional estimation disparity. In the latter case, this might be a case of legal causation, i.e., that the
270

model is poor, and hence, the modelling has resulted in disadvantaging a protected group. Therefore,
271

we can view the causal structure of ˆp(y|x) as central to avoiding unlawful discrimination. However,
272

not considering causal structures could lead to conditional estimation disparity, and potentially result
273

in unlawful discrimination.
274

Legal causation focuses on the legal causal link between xp and the decision a. In addition, legal
275

causation is less formal than common definitions of causal effects in ML. Courts, at least outside of
276

the US, are effects-orientated, and a wide range of forms of a “legal causal link” could be identified
277

[109, 65]. Much of the causal-based fairness literature formulates “causation” on the true causal
278

model structure in ˆp(y|x), i.e., the study of the causal effect of x, due to outside interventions on y
279

[101, 10, 149, 21]. However, this formulation is not the same as that of legal causation.
280

In this discussion, the parallels to other discrimination studies become evident in how it would
281

affect automated decision-making, particularly taste-based and statistical discrimination. Taste-based
282

discrimination[11], could arise if only the utility function u(a, y) unjustifiably disfavours a group
283

based on protected attributes xp. Statistical discrimination, on the other hand, arises when decision-
284

makers use group-level statistics as proxies for individual characteristics due to imperfect information
285

6


---Page Break---
[7, 102]. Statistical discrimination parallels the disadvantaging of a group due to having different
286

ˆp(x|y). While these types of discrimination are generally prohibited, statistical discrimination can be
287

legally permissible in some circumstances (see Section 2.1).
288

2.6
Legitimate aim and y
289

Decision-makers must consider the legitimacy of using an SML model by explicitly defining its
290

purpose and the outcome variable y. In algorithm design, social implications should be considered [87,
291

57, 63]. Additionally, this aligns the model’s use with legal expectations.
292

If the court believes sufficient evidence of discrimination exists, the burden shifts to the respondent to
293

disprove allegations of unlawful discrimination [38]. Indirect discrimination can be justified if the
294

PCP is a proportionate means of achieving a legitimate aim [40, s 19(2)(d)]. Identifying a legitimate
295

aim is closely connected to the choice of y, the unknown entity used for decision-making. If the
296

choice of y is legitimate based on context and the benefit outweighs any potential harm, there is a
297

lower risk of unlawful discrimination [35].
298

The legitimacy of the aim depends on the decision-makers’ raison d’être [65]. In Homer, the Court
299

established a legitimate aim must “correspond to a real need and the means used must be appropriate
300

with a view to achieving the objective and be necessary to that end” [128, 35, 39]. In lending, it is
301

a legitimate aim to protect the repayment of their loans or at least secure their loans. In fact, “the
302

mortgage market could not survive without that aim being realised” [126, para. 79].
303

For a legitimate y to be an exception to indirect discrimination, the PCP must be a proportionate
304

means of achieving the legitimate y [40, s 19(2)(d)]. To be proportionate, it must be an appropriate
305

means of achieving the legitimate aim and (reasonably) necessary to do so [128]. Such analysis
306

will turn on the facts of each case. However, it will require evaluating whether the design choices
307

were “appropriate with a view to achieving the objective and be necessary” by weighing the need
308

against the seriousness of detriment to the disadvantaged group [39, para. 151]. This will require
309

considering whether non-discriminatory alternatives were available [128]. Measures to improve
310

accuracy, maximise benefits over costs, minimise estimation error, or condition for protected attributes
311

may all be relevant considerations for whether the modeller’s choices were proportionate means of
312

achieving a legitimate y.
313

If the estimated outcome ˜y approximates the true outcome y, this can lead to biased predictions. Let
314

γi = ||p(˜yi|xtrue
i
) −p(yi|xtrue
i
)||2 ,
(8)

then, if the expectation of γ condition on xl shows a disparity, i.e.,
315

Ex[γ | xp, xl] ̸= Ex[γ | xl] ,
(9)

it suggests the use of ˜y is inappropriate and might be discriminatory.
316

To illustrate with an example, if a bank’s training data is outdated or sourced from a different country,
317

it may not accurately represent the current population relevant to the model. This discrepancy can
318

lead to biased estimations, particularly if the data reflects historical prejudices. For instance, the
319

model might unjustly associate certain demographics with higher default risk, not because of true
320

differences but biased historical data [as warned in 33].
321

2.7
Legitimate x
322

One of the more crucial aspects of SML for automated decision-making is the choice of features x.
323

The aim and y will help inform the choice of features to include in the model. We can separate three
324

types of features from a legal perspective: features with protected attributes xp, legitimate features xl,
325

and non-legitimate or illegitimate features xn. The distinction between xl and xn depends on whether
326

the feature can be considered legitimately related to y (see Section 2.5). Causal fairness literature
327

has engaged with questions of discriminatory variables through the lens of proxy discrimination
328

[69, 115]. Proxy discrimination has a specific legal meaning under UK law that relates to direct
329

discrimination, unlike much of the US literature on proxy discrimination that relates to indirect forms
330

of discrimination. Here, we explain the UK legal implications of such causal relationships between
331

variables and we provide a real-world example in Appendix A.
332

7


---Page Break---
2.7.1
Direct Discrimination and Removing xp
333

Direct discrimination in automated decisions may arise when members of, or an entire protected
334

group, is affected. Where a model ˆp(y|x) uses a protected attribute xp, and there is a difference
335

in predictions between the protected groups defined by xp, this risk arises. Models have directly
336

used protected characteristics, giving rise to direct discrimination [94, see discussion in Appendix].
337

Direct discrimination may arise when a feature is an exact proxy for a protected attribute. In Lee v
338

Ashers, Lady Hale explained that the risk of direct discrimination also arises if a decision is based on
339

a feature that “is not the protected characteristic itself but some proxy for it” [129]. Therefore, direct
340

discrimination can arise even where xp has been removed because there is a feature which is an exact
341

proxy that is “indissociable” or has an “exact correspondence” to xp [130, 129]. Formally, we can
342

define an exact proxy as a feature ˜xp with a perfect or almost perfect correlation with xp [115].
343

UK courts have accepted that an exact proxy would be pregnancy because “pregnancy is unique
344

to the female sex” [121, 125]. If a model uses pregnancy or maternity leave as a feature, collected
345

from CV information, for example, it would have the effect of using an exact proxy ˜xp that could
346

hypothetically be the basis for a direct discrimination claim.
347

Given the relevance of xp to direct discrimination, modellers have been encouraged to remove pro-
348

tected attributes when designing ML models [105, 62, 48]. These claims are usually based on the US
349

Equal Protection Clause, which subjects classifications based on certain protected characteristics, such
350

as race, to strict scrutiny [146]. The focus on excluding certain data inputs is one form of discrimina-
351

tion prevention [146, 46], but not under UK law. Further, simply removing protected characteristics
352

reduces accuracy and utility [150, 66], and does not remove the risk of discrimination [34, 79, 72].
353

This reasoning connects to the true DGP. If a protected attribute like gender is inherent in the DGP,
354

removing it does not eliminate discrimination but instead may introduce it. Taking a gender-neutral
355

approach to recidivism predictions may have the adverse effect of discrimination against women who
356

would otherwise have received lower risk scores [31]. In Loomis, the Court accepted that in recidivism
357

algorithms, “if the inclusion of gender promotes the accuracy, it serves the interests of institutions and
358

defendants, rather than a discriminatory purpose” [112, 766]. Hence, if the inclusion of xp improves
359

the accuracy and benefits the protected group, it may avoid the risk of discriminatory purposes. There
360

is an absence of any legal guidance in the UK on the relationship between true probabilities and
361

protected attributes in automated decision-making. Pending further legal guidance, it is important to
362

carefully consider whether including xp is relevant to promote accuracy and conditional estimation
363

parity. Removing protected attributes often ignores the true probabilities for the legitimate differences
364

between protected groups, affecting the lawfulness of its outcomes.
365

Therefore, removing xp will not avoid liability for unlawful direct discrimination by itself. Even
366

if a model ignores xp, in practice, it may rely on other data points acting as proxies with “exact
367

correspondence” to a protected characteristic ˜xp. Importantly, this diverges from US law and
368

highlights that intention is immaterial to UK direct discrimination [Cf. e.g., 5, 115]. UK law focuses
369

on the discriminatory effects rather than a formalistic view of whether xp is considered or not.
370

2.7.2
Defining xl and xn
371

Indirect discrimination may arise if a PCP appears to apply equally to everyone but disadvantages
372

members of a protected group. Both forms of discrimination can arise using an exact proxy or a
373

weak proxy in a PCP. Therefore, identifying legitimate features is challenging when many features
374

correlate to protected groups. We define non-legitimate features xn as features not legitimate in
375

the context of the true DGP (Section 2.2). In practice, this means a non-legitimate feature is one
376

that, if included, would not contribute to the predictive performance of the optimal model, i.e., the
377

one with the lowest estimation error (Section 2.3). Therefore, xn would not improve the predictive
378

performance if a modeller had the true features.
379

For example, hair length strongly correlates to gender in many cultural contexts but is unlikely to
380

contribute to the consumers’ true default risk. Boyarskaya et al. explain the absence of a “causal
381

story” between hair length and loan repayment because hair length would not be part of a true model
382

for the risk of default [19]. Therefore, hair length is an example of xn in a lending context.
383

For comparison, the legitimacy of zip codes illustrates the nuanced nature of legitimate features.
384

While a zip code may correlate with race in some contexts, it might be a legitimate variable in
385

other situations. For example, in an application for home insurance covering flood risk, zip codes
386

8


---Page Break---
are invaluable proxies for granular information such as geographical features, land topography and
387

historical flooding. Therefore, in the best model for property flood insurance decisions, zip code will
388

improve the predictive performance as a legitimate proxy for data within the true DGP. However, in a
389

university application, there should be no predictive or causal relationship to merit for acceptance. In
390

such cases, zip code likely acts as a proxy for race or the unprotected characteristic of socio-economic
391

status and would be xn. So, in some circumstances, the zip code would be legitimate xl, but in others,
392

it may not be xn. It will also be relevant to consider whether a less discriminatory feature is available,
393

i.e., one with less correlation to a protected attribute that is equally predictive.
394

As explored in Appendix A, in lending, information about income and debts are likely to be legitimate
395

features xl. Credit scores can be a proxy for a person’s financial position, as well as protected
396

attributes [18, 60]. However, the complexity of calculating credit scores means it is more valuable for
397

inferring income, debt repayments, and history of credit. Credit scores, or related features, would
398

have a material impact on the true model for default, and then would be a legitimate feature xl.
399

Given that nearly, all features may contain some information on protected attributes, even legitimate
400

factors [30], this approach explains the need to assess the strength of this dependence and whether the
401

feature contributes significantly to the model’s prediction and can be argued to be part of a true DGP.
402

2.7.3
Feature construction from x
403

The distinction between xl and xn also gives rise to problems in automatic feature construction, such
404

as using deep neural networks. If features are constructed automatically using a combination of xl
405

and xn, indirect and direct discrimination are risks. As an example, an applicant’s resume contains
406

legitimate features xl for recruitment prediction. However, the detailed granularity of many resumes
407

also gives rise to the problem of non-legitimate information, such as maternity leave or women-only
408

sports or other information that may contain information on other protected attributes. Hence, there
409

needs to be an active choice of only including legitimate features xl from available data in the model.
410

3
Conclusions
411

Minimising unlawful discrimination in automated decision-making requires a nuanced and contextual
412

approach. While it is beyond our scope to offer specific legal advice, our findings underscore several
413

key considerations to identify and mitigate potential discrimination effectively:
414

1. Assess data legitimacy. Carefully examine if the data, both the target variable (y) and features (x),
415

are legitimate for the specific context (Sections 2.6 and 2.7). Legal analysis should inform what is
416

legitimate in a specific setting.
417

2. Build an accurate model. Strive to approximate the true DGP p(y|x), using only legitimate
418

features xl. Reasonable, necessary, and proportionate steps must be taken to minimise estimation
419

error and aim for estimation parity (Section 2.3). This may entail model inference, interrogating
420

social biases in the data, and scrutinising the estimated model.
421

3. Evaluate statistical disparity. Given the best model ˆp(y|x), assess for conditional statistical parity
422

by examining outcomes across groups with protected characteristics (Section 2.4). If a model’s
423

performance improves by including protected attributes, consider:
424

(a) Identify whether conditional statistical parity is unattainable or undesirable based on true
425

group differences. This requires stringent analysis into whether differences stem from prior
426

injustice or legitimate variation.
427

(b) Incorporate further legitimate features xl that could minimise statistical disparities by “ex-
428

plaining away” the performance gained by the protected attribute with legitimate features.
429

(c) Avoid using the model due to unmitigated discrimination risks.
430

While these guidelines cannot guarantee lawful automated decisions, they provide meaningful
431

recommendations and abstractions to help identify and mitigate unlawful discrimination risks.
432

In conclusion, this work bridges a critical gap between the technical aspects of automated decisions
433

and the complexities of anti-discrimination law. By translating these nuanced legal concepts into
434

decision theory, we underscore the importance of accurately modelling true data-generating processes
435

and the innovative concept of estimation parity. This interdisciplinary approach enhances the
436

understanding of automated decision-making and sets a foundation for future research that aligns
437

technological advancements with legal and ethical standards.
438

9


---Page Break---
References
439

[1] Jeremias Adams-Prassl, Reuben Binns, and Aislinn Kelly-Lyth. Directly Discriminatory Algorithms. The
440

Modern Law Review, 86(1):144–175, 2022.
441

[2] Nikita Aggarwal. The Norms of Algorithmic Credit Scoring. Cambridge Law Journal, 80(1):42–73,
442

2021.
443

[3] AHRC. A Quick Guide to Australian Discrimination Laws. Technical report, Australian Human Rights
444

Commission, 2014.
445

[4] Hirotogu Akaike. Information theory and an extension of the maximum likelihood principle. 1973.
446

[5] Larry Alexander and Kevin Cole. Discrimination by Proxy. Constitutional Commentary, 14:453–463,
447

1997.
448

[6] Jose Manuel Alvarez and Salvatore Ruggieri. Counterfactual Situation Testing: Uncovering Discrimi-
449

nation under Fairness given the Difference. In Proceedings of the 3rd ACM Conference on Equity and
450

Access in Algorithms, Mechanisms, and Optimization, Boston, MA, USA„ 2023. ACM.
451

[7] Kenneth Arrow. The Theory of Discrimination. In Discrimination in Labor Markets, pages 3–33.
452

Princeton University Press, 1971.
453

[8] Australian Parliament. Sex Discrimination Act 1984.
454

[9] Bangladesh Parliament. Anti-Discrimination Bill 2022.
455

[10] Solon Barocas and Andrew Selbst. Big Data’s Disparate Impact. California Law Review, 104(3):671–732,
456

2016.
457

[11] Gary Becker. The Economics of Discrimination. University of Chicago Press, 1957.
458

[12] Ruben Becker, Gianlorenzo D’Angelo, and Sajjad Ghobadi. On the cost of demographic parity in
459

influence maximization, June 2023.
460

[13] James Berger. Statistical Decision Theory and Bayesian Analysis. New York: Springer, 1985.
461

[14] Richard Berk, Hoda Heidari, Shahin Jabbari, Michael Kearns, and Aaron Roth. Fairness in Criminal
462

Justice Risk Assessments: The State of the Art. Sociological Methods & Research, 50(1):3–44, 2018.
463

[15] José M Bernardo and Adrian FM Smith. Bayesian theory. John Wiley & Sons, 1994.
464

[16] J. R. Biden. Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial
465

Intelligence. The White House, 2023. Executive Order 14110.
466

[17] Reuben Binns. Fairness in Machine Learning: Lessons from Political Philosophy. Proceedings of
467

Machine Learning Research, 81:149–159, 2018.
468

[18] Harold Black, Robert L. Schweitzer, and Lewis Mandell. Discrimination in Mortgage Lending. The
469

American Economic Review, 68(2):186–191, 1978.
470

[19] Margarita Boyarskaya, Solon Barocas, Hanna Wallach, and Michael Carl Tschantz. What Is a Proxy and
471

Why Is It a Problem? Proceedings of the Conference on Fairness, Accountability, and Transparency,
472

2022.
473

[20] Canadian Parliament. Human Rights Act. R.S.C. (c.H-6), 1985.
474

[21] Alycia N. Carey and Xintao Wu. The causal fairness field guide: perspectives from social and formal
475

sciences. Frontiers in Big Data, 5:892837, 2022.
476

[22] Alycia N. Carey and Xintao Wu. The statistical fairness field guide: perspectives from social and formal
477

sciences. AI and Ethics, 3(1):1–23, 2023.
478

[23] Alessandro Castelnovo, Riccardo Crupi, Greta Greco, Daniele Regoli, Ilaria Giuseppina Penco, and
479

Andrea Claudio Cosentini. A clarification of the nuances in the fairness metrics landscape. Scientific
480

Reports, 12(1), 2022.
481

[24] Simon Caton and Christian Haas. Fairness in Machine Learning: A Survey. ACM Computing Surveys,
482

2023. Just Accepted.
483

10


---Page Break---
[25] Zhisheng Chen. Ethics and discrimination in artificial intelligence-enabled recruitment practices. Human-
484

ities and Social Sciences Communications, 10(1), 2023.
485

[26] S. Chiappa. Path-specific counterfactual fairness. In Proceedings of the AAAI Conference on Artificial
486

Intelligence, pages 7801–7808, 2019.
487

[27] Alexandra Chouldechova. Fair Prediction with Disparate Impact: A Study of Bias in Recidivism
488

Prediction Instruments. Big Data, 5(2):153–163, 2017.
489

[28] CIA. Legal System - The World Factbook. Technical report, Central Intelligence Agency.
490

[29] Nancy S. Cole. Bias in Selection. Journal of Educational Measurement, 10(4):237–255, 1973.
491

[30] Sam Corbett-Davies, Johann D. Gaebler, Hamed Nilforoshan, Ravi Shroff, and Sharad Goel. The Measure
492

and Mismeasure of Fairness. Journal of Machine Learning Research, 24(312):1–117, 2023.
493

[31] Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. Algorithmic Decision
494

Making and the Cost of Fairness. In Proceedings of the 23rd ACM SIGKDD International Conference on
495

Knowledge Discovery and Data Mining, pages 797–806, Halifax, Canada, 2017.
496

[32] M. H. DeGroot. Optimal Statistical Decisions. McGraw-Hill, 1970.
497

[33] DFS. Report on Apple Card Investigation. New York State Department of Financial Services, 2021.
498

[34] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Rich Zemel. Fairness Through
499

Awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, pages
500

214–226, 2012.
501

[35] ECJ. C-170/84, Bilka Kaufhaus GmbH v Weber von Hartz. European Court of Justice, 1986. ECR 1607.
502

[36] Elizabeth Emens. Intimate Discrimination: The State’s Role in the Accidents of Sex and Love. Harvard
503

Law Review, 22(5):1307–1402, 2009.
504

[37] England and Wales Court of Appeal. Igen ltd v Wong. [2005] EWCA Civ 142.
505

[38] England and Wales Court of Appeal. Madarassy v Nomura International plc. [2007] EWCA Civ 33.
506

[39] England and Wales Court of Appeal. Secretary of State for Defence v Elias. (2006) IRLR 934.
507

[40] Equality Act. 2010 (UK).
508

[41] Virginia Eubanks. Automating Inequality. St. Martin’s Press, 2018.
509

[42] European Court of Justice. C-236/09 Association belge des Consommateurs Test-Achats ASBL v Conseil
510

des ministres. (2011) ECR I-00773.
511

[43] European Parliament. Directive 2002/73/EC of the European Parliament and of the Council of 23
512

September 2002 amending Council Directive 76/207/EEC on the implementation of the principle of equal
513

treatment for men and women as regards access to employment, vocational training and promotion, and
514

working conditions. 2002. OJ L 269.
515

[44] European Parliament. Amendments adopted by the European Parliament on 14 June 2023 on the proposal
516

for a regulation of the European Parliament and of the Council on laying down harmonised rules on
517

artificial intelligence (Artificial Intelligence Act) and amending certain Union legislative acts. 2023.
518

(COM(2021)0206 – C9-0146/2021 – 2021/0106(COD).
519

[45] European Union. Charter of Fundamental Rights of the European Union. 2009. OJ 2012/C 326/02.
520

[46] Talia Gillis. The Input Fallacy. Minnesota Law Review, 106:1175, 2022.
521

[47] Government of India. Constitution of India, 1950.
522

[48] Przemyslaw A. Grabowicz, Nicholas Perello, and Aarshee Mishra. Marrying Fairness and Explainability
523

in Supervised Learning. In Proceedings of the Conference on Fairness, Accountability, and Transparency,
524

page 1905–1916, Seoul, Republic of Korea, 2022.
525

[49] Moritz Hardt, Eric Price, and Nathan Srebro. Equality of Opportunity in Supervised Learning. In
526

Proceedings of the 30th Conference on Neural Information Processing Systems, Barcelona, Spain, 2016.
527

11


---Page Break---
[50] Hoda Heidari, Michele Loi, Krishna P. Gummadi, and Andreas Krause. A moral framework for under-
528

standing fair ml through economic models of equality of opportunity. In Proceedings of the Conference
529

on Fairness, Accountability, and Transparency, page 181–190, Atlanta, GA, 2019.
530

[51] Deborah Hellman. Measuring Algorithmic Fairness. Virginia Law Review, 106(4):811–866, 2020.
531

[52] Deborah Hellman. Sex, Causation, and Algorithms: How Equal Protection Prohibits Compounding Prior
532

Injustice. Washington University Law Review, 98:481–523, 2020.
533

[53] Anne Hellum, Ingunn Ikdahl, Vibeke Strand, and Eva-Maria Svensson. Nordic Equality and Anti-
534

Discrimination Laws in the Throes of Change: Legal developments in Sweden, Finland, Norway, and
535

Iceland. Routledge, 2023.
536

[54] Corinna Hertweck, Christoph Heitz, and Michele Loi. On the Moral Justification of Statistical Parity. In
537

Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 747–757, 2021.
538

[55] Daniel E Ho and Alice Xiang. Affirmative algorithms: The legal grounds for fairness as awareness.
539

University of Chicago Law Review Online, pages 134–154, 2020.
540

[56] Sally Ho and Garance Burke. An Algorithm that Screens for Child Neglect Raises Concerns. Associated
541

Press, 2022.
542

[57] Lily Hu and Issa Kohler-Hausmann. What’s sex got to do with machine learning? In Proceedings of the
543

Conference on Fairness, Accountability, and Transparency, page 513, Barcelona, Spain, 2020.
544

[58] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: an
545

introduction to concepts and methods. Machine Learning, 110(3):457–506, March 2021.
546

[59] Anna Lena Hunkenschroer and Alexander Kriebitz. Is AI Recruiting (un)ethical? A Human Rights
547

Perspective on the Use of AI for Hiring. AI and Ethics, 3(1):199–213, 2022.
548

[60] Mikella Hurley and Julius Adebayo. Credit Scoring in the Era of Big Data. Yale Journal of Law and
549

Technology, 18(1):148–216, 2017.
550

[61] Ben Hutchinson and Margaret Mitchell. 50 Years of Test (Un)fairness: Lessons for Machine Learning. In
551

Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 49–58, Atlanta, GA,
552

2019.
553

[62] James E. Johndrow and Kristian Lum. An Algorithm For Removing Sensitive Information: Application
554

To Race-independent Recidivism Prediction. The Annals of Applied Statistics, 13(1):pp. 189–220, 2019.
555

[63] Maximilian Kasy and Rediet Abebe. Fairness, Equality, and Power in Algorithmic Decision-Making. In
556

Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 576–586, 2021.
557

[64] Aislinn Kelly-Lyth. Challenging Biased Hiring Algorithms. Oxford Journal of Legal Studies, 41(4):899–
558

928, 2021.
559

[65] Tarunabh Khaitan. A Theory of Discrimination Law. Oxford University Press, 2015.
560

[66] Fereshte Khani and Percy Liang. Removing Spurious Features Can Hurt Accuracy and Affect Groups
561

Disproportionately. In Proceedings of the Conference on Fairness, Accountability, and Transparency,
562

page 196–205, 2021.
563

[67] Elif Kiesow Cortez and Nestor Maslej. Adjudication of Artificial Intelligence and Automated Decision-
564

Making Cases in Europe and the USA. European Journal of Risk Regulation, 14(3):457–475, 2023.
565

[68] Niki Kilbertus, Adria Gascon, Matt Kusner, Michael Veale, Krishna Gummadi, and Adrian Weller. Blind
566

Justice: Fairness with Encrypted Sensitive Attributes. In Proceedings of the 35th International Conference
567

on Machine Learning, pages 2630–2639, Stockholm, Sweden, 2018.
568

[69] Niki Kilbertus, Mateo Rojas-Carulla, Giambattista Parascandolo, Moritz Hardt, Dominik Janzing, and
569

Bernhard Schölkopf. Avoiding discrimination through causal reasoning. In Advances in Neural Informa-
570

tion Processing Systems, volume 30, page 656–666, 2017.
571

[70] Pauline Kim. Auditing Algorithms for Discrimination. University of Pennsylvania Law Review Online,
572

166(1), 2017.
573

[71] Barbara Kiviat. The Moral Affordances of Construing People as Cases: How Algorithms and the Data
574

They Depend on Obscure Narrative and Noncomparative Justice. Sociological Theory, 41(3):175–200,
575

2023.
576

12


---Page Break---
[72] Jon Kleinberg, Jens Ludwig, Sendhil Mullainathan, and Cass Sunstein. Discrimination in the Age of
577

Algorithms. Journal of Legal Analysis, 10:113–174, 2019.
578

[73] Issa Kohler-Hausmann and Robin Dembroff. Supreme Confusion About Causality at the Supreme Court.
579

City University of New York Law Review, 25(1):57–92, 2022.
580

[74] Joshua Kroll, Joanna Huey, Solon Barocas, Edward Felten, Joel Reidenberg, David Robinson, and Harlan
581

Yu. Accountable Algorithms. University of Pennsylvania Law Review, 165(3):633, 2017.
582

[75] Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual Fairness. Advances in
583

Neural Information Processing Systems, 30:4069–4079, 2017.
584

[76] Katja Langenbucher. Consumer Credit in The Age of AI – Beyond Anti-Discrimination Law. Law
585

Working Paper No. 663/2022, 2023.
586

[77] Finn Lattimore, Simon O’Callaghan, Zoe Paleologos, Alistair Reid, Edward Santow, Holli Sargeant,
587

and Andrew Thomsen. Using Artificial Intelligence to Make Decisions: Addressing the Problem of
588

Algorithmic Bias. Technical Paper, Australian Human Rights Commission, 2020.
589

[78] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, and Eirini Ntoutsi. A survey on datasets for
590

fairness-aware machine learning. WIREs Data Mining and Knowledge Discovery, 12(3):e1452, 2022.
591

[79] Zachary Lipton, Julian McAuley, and Alexandra Chouldechova. Does Mitigating ML’s Impact Disparity
592

Require Treatment Disparity? Advances in Neural Information Processing Systems, 31, 2018.
593

[80] Emmanuel Martinez and Lauren Kirchner. The Secret Bias Hidden in Mortgage-Approval Algorithms.
594

The Markup, https://perma.cc/U6W9-MECE, 2021.
595

[81] Sandra G. Mayson. Bias In, Bias Out. Yale Law Journal, 128(8):2122–2473, 2019.
596

[82] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A Survey on
597

Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6), 2021.
598

[83] Ministry of Justice, Finland. Government Porposal for the Equality Act and Related Laws HE 19/2014 vp
599

(Hallituksen esitys eduskunnalle yhdenvertaisuuslaiksi ja eräiksi siihen liittyviksi laeiksi).
600

[84] Ministry of Justice, Finland. Non-Discrimination Act (Yhdenvertaisuuslaki) (1325/2014).
601

[85] Shira Mitchell, Eric Potash, Solon Barocas, Alexander D’Amour, and Kristian Lum. Algorithmic Fairness:
602

Choices, Assumptions, and Definitions. Annual Review of Statistics and Its Application, 8(1):141–163,
603

2021.
604

[86] Sophia Moreau. What Is Discrimination? Philosophy & Public Affairs, 38(2):143–179, 2010.
605

[87] Deirdre Mulligan, Joshua Kroll, Nitin Kohli, and Richmond Wong. This Thing Called Fairness: Disci-
606

plinary Confusion Realizing a Value in Technology. In Proceedings of the ACM on Human-Computer
607

Interaction, volume 3, pages 1–36, 2019.
608

[88] Mpoki Mwakagali. International Human Rights Law and Discrimination Protections. Brill, 2018.
609

[89] Jakob Mökander. Auditing of AI: Legal, Ethical and Technical Approaches. Digital Society, 2(3):49,
610

2023.
611

[90] Arvind Narayanan. Tutorial: 21 Fairness Definition and their Politics. Proceedings of the Conference on
612

Fairness, Accountability, and Transparency, 2018.
613

[91] New Zealand Parliament. Human Rights Act, 1993.
614

[92] Hamed Nilforoshan, Johann D Gaebler, Ravi Shroff, and Sharad Goel. Causal conceptions of fairness
615

and their consequences. In International Conference on Machine Learning, pages 16848–16887. PMLR,
616

2022.
617

[93] Safiya Umoja Noble. Algorithms of Oppression. New York University Press, 2018.
618

[94] Finland National Non-Discrimination and Equality Tribunal. Decision 216/2017. 2018.
619

[95] Tony O’Hagan. Dicing with the Unknown. Significance, 1(3):132–133, 2004.
620

[96] OHCHR. Banning Discrimination on Grounds of Socioeconomic Disadvantage: An Essential Tool in the
621

Fight Against Poverty. Thematic Report A/77/157, Special Rapporteur on Extreme Poverty and Human
622

Rights, United Nations Office of the High Commissioner for Human Rights, 2022.
623

13


---Page Break---
[97] Cathy O’Neil. Weapons of Math Destruction: How Big Data Increases Inequality and Threatens
624

Democracy. Penguin Books, 2016.
625

[98] Cathy O’Neil, Holli Sargeant, and Jacob Appel. Explainable Fairness in Regulatory Algorithmic Auditing,
626

2023.
627

[99] Giovanni Parmigiani and Lurdes Inoue. Decision Theory. Wiley, 2010.
628

[100] Frank Pasquale. The Black Box Society. Harvard University Press, 2019.
629

[101] Judea Pearl. An Introduction to Causal Inference. The International Journal of Biostatistics, 6(2), 2010.
630

[102] Edmund Phelps. The Statistical Theory of Racism and Sexism. The American Economic Review,
631

62(4):659–661, 1972.
632

[103] Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben
633

Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. Closing the AI Accountability Gap:
634

Defining an End-to-End Framework for Internal Algorithmic Auditing. In Proceedings of the Conference
635

on Fairness, Accountability, and Transparency, page 33–44, Barcelona, Spain, 2020.
636

[104] Lisa Rice and Deidre Swesnik. Discriminatory Effects of Credit Scoring on Communities of Color.
637

Suffolk University Law Review, 46(935):935–966, 2013.
638

[105] Andrea Romei and Salvatore Ruggieri. A multidisciplinary survey on discrimination analysis. The
639

Knowledge Engineering Review, 29(5):582–638, 2014.
640

[106] Chris Russell, Matt J Kusner, Joshua Loftus, and Ricardo Silva. When Worlds Collide: Integrating
641

Different Counterfactual Assumptions in Fairness. In Advances in Neural Information Processing Systems,
642

volume 30, page 6417–6426, 2017.
643

[107] Holli Sargeant. Algorithmic decision-making in financial services: economic and normative outcomes in
644

consumer credit. AI and Ethics, 3(4):1295–1311, 2023.
645

[108] Leonard Savage. The Foundations of Statistics. Operations Research, 4(2):254–258, 1956.
646

[109] Patrick Shin. Is there a unitary concept of discrimination?
In Deborah Hellman and Sophia Rei-
647

betanz Moreau, editors, Philosophical foundations of discrimination law, page 172. Oxford University
648

Press, 2013.
649

[110] South African Parliament. Promotion of Equality and Prevention of Unfair Discrimination Act, 2000.
650

[111] Till Speicher, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, and
651

Muhammad Bilal Zafar. A unified approach to quantifying algorithmic unfairness: Measuring individual
652

& group unfairness via inequality indices. In Proceedings of the 24th International Conference on
653

Knowledge Discovery & Data Mining, page 2239–2248, London, United Kingdom, 2018.
654

[112] Supreme Court of Wisconsin. State v. Loomis. 881 N.W.2d 749, 2016.
655

[113] Adrien Sénécat. The use of opaque algorithms facilitates abuses within public services. Le Monde, 2023.
656

[114] Anique Tahir, Lu Cheng, and Huan Liu. Fairness through aleatoric uncertainty. In Proceedings of the 32nd
657

International Conference on Information and Knowledge Management, page 2372–2381, Birmingham,
658

United Kingdom, 2023.
659

[115] Michael Carl Tschantz. What is proxy discrimination? In Proceedings of the Conference on Fairness,
660

Accountability, and Transparency, pages 1993–2003, Seoul, Republic of Korea, June 2022.
661

[116] UK Parliament. Disability Discrimination Act 1995.
662

[117] UK Parliament. Explanatory Memorandum to the Equality Act 2010 (Age Exceptions Order). 2012.
663

[118] UK Parliament. Race Relations Act 1965.
664

[119] UK Parliament. Sex Discrimination Act 1975.
665

[120] United Kingdom Employment Appeals Tribunal.
Dziedziak v Future Electronics Ltd.
[2012]
666

UKEAT/0270/11.
667

[121] United Kingdom Employment Appeals Tribunal. O’Neil v Governors of St Thomas More Roman Catholic
668

School. (1996) IRLR 372.
669

14


---Page Break---
[122] United Kingdom House of Lords. Equal Opportunities Commission, R (on the application of) v Birming-
670

ham City Council. (1989) 1 AC 1155.
671

[123] United Kingdom House of Lords. James v Eastleigh Borough Council. (1990) 2 AC 751.
672

[124] United Kingdom House of Lords. Secretary of State For Employment, Ex Parte Seymour Smith and
673

Another, R v. (2000) 1 All ER 857.
674

[125] United Kingdom House of Lords. Webb v EMO Air Cargo (UK) Ltd (No. 2). (1995) IRLR 645.
675

[126] United Kingdom Supreme Court. Akerman-Livingstone v Aster Communities Ltd. (2015) 1 AC 1399.
676

[127] United Kingdom Supreme Court. Essop v Home Office (UK Border Agency). (2017) IRLR 558.
677

[128] United Kingdom Supreme Court. Homer v Chief Constable of West Yorkshire Police. (2012) IRLR 601.
678

[129] United Kingdom Supreme Court. Lee v Ashers. (2018) AC 413.
679

[130] United Kingdom Supreme Court. R (Coll) v Secretary of State for Justice. (2017) 1 WLR 2093.
680

[131] United Kingdom Supreme Court. R (on the application of E) v JFS Governing Body. (2009) 1 WLR
681

2353.
682

[132] United Nations. Universal Declaration of Human Rights. 1948.
683

[133] Aki Vehtari, Andrew Gelman, and Jonah Gabry. Practical bayesian model evaluation using leave-one-out
684

cross-validation and waic. Statistics and computing, 27:1413–1432, 2017.
685

[134] Aki Vehtari and Jouko Lampinen. Bayesian model assessment and comparison using cross-validation
686

predictive densities. Neural computation, 14(10):2439–2468, 2002.
687

[135] Aki Vehtari and Janne Ojanen. A survey of Bayesian predictive methods for model assessment, selection
688

and comparison. Statistics Surveys, 6:142 – 228, 2012.
689

[136] Sahil Verma and Julia Rubin. Fairness Definitions Explained. In Proceedings of the International
690

Workshop on Software Fairness, pages 1–7, Gothenburg, Sweden, 2018.
691

[137] Marc De Vos. The European Court of Justice and the March Towards Substantive Equality in European
692

Union Anti-Discrimination Law. International Journal of Discrimination and the Law, 20(1):62–87,
693

2020.
694

[138] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Why Fairness Cannot Be Automated: Bridging the
695

Gap Between EU Non-discrimination Law and AI. Computer Law & Security Review, 41:105567, 2021.
696

[139] Sandra Wachter, Brent Daniel Mittelstadt, and Chris Russell. Bias Preservation in Machine Learning:
697

The Legality of Fairness Metrics Under EU Non-Discrimination Law. West Virginia Law Review,
698

123(3):735–790, 2021.
699

[140] Hilde Weerts, Raphaële Xenidis, Fabien Tarissan, Henrik Palmer Olsen, and Mykola Pechenizkiy.
700

Algorithmic unfairness through the lens of eu non-discrimination law: Or why the law is not a decision
701

tree. In Proceedings of the Conference on Fairness, Accountability, and Transparency, page 805–816,
702

Chicago, IL, USA, 2023. ACM.
703

[141] Halbert White. Maximum likelihood estimation of misspecified sodels. Econometrica, 50(1):1–25, 1982.
704

[142] Yongkai Wu, Lu Zhang, Xintao Wu, and Hanghang Tong.
Pc-fairness: A unified framework for
705

measuring causality-based fairness. In Advances in Neural Information Processing Systems, volume 32,
706

page 3404–3414, 2019.
707

[143] Raphaële Xenidis. Tuning EU equality law to algorithmic discrimination: Three pathways to resilience.
708

Maastricht Journal of European and Comparative Law, 27(6):736–758, 2020.
709

[144] Alice Xiang. Reconciling Legal and Technical Approaches to Algorithmic Bias. Tennessee Law Review,
710

88(3):649, 2021.
711

[145] Renzhe Xu, Peng Cui, Kun Kuang, Bo Li, Linjun Zhou, Zheyan Shen, and Wei Cui. Algorithmic decision
712

making with conditional fairness. Proceedings of the 26th International Conference on Knowledge
713

Discovery & Data Mining, 2020.
714

[146] Crystal Yang and Will Dobbie. Equal Protection Under Algorithms: A New Statistical and Legal
715

Framework. Michigan Law Review, 119:291, 2020.
716

15


---Page Break---
[147] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna Gummadi. Fairness
717

beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In
718

Proceedings of the 26th International Conference on World Wide Web, pages 1171–1180, Perth, Australia,
719

2017.
720

[148] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez-Rodriguez, and Krishna P. Gummadi. Fairness
721

constraints: A flexible approach for fair classification. Journal of Machine Learning Research, 20(75):1–
722

42, 2019.
723

[149] Junzhe Zhang and Elias Bareinboim. Fairness in decision-making — the causal explanation formula.
724

Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), April 2018.
725

[150] Lu Zhang, Yongkai Wu, and Xintao Wu. A Causal Framework for Discovering and Removing Direct and
726

Indirect Discrimination. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial
727

Intelligence, pages 3929–3935, Melbourne, Australia, 2017.
728

[151] Miri Zilka, Holli Sargeant, and Adrian Weller. Transparency, Governance and Regulation of Algorithmic
729

Tools Deployed in the Criminal Justice System: A UK Case Study. In Proceedings of the Conference on
730

AI, Ethics, and Society, page 880–889, Oxford, United Kingdom, 2022.
731

16


---Page Break---
A
Case Study
732

Overview of Finnish Anti-Discrimination Law
733

Finnish anti-discrimination law bears many similarities to UK and EU laws. We briefly set out the
734

relevant provisions that show the similarities to the Equality Act set out in Section 1.3.
735

Section 8(1) of the Non-Discrimination [84] defines the protected characteristics as:1
736

No one may be discriminated against on the basis of age, origin, nationality, language, religion,
737

belief, opinion, political activity, trade union activity, family relationships, state of health,
738

disability, sexual orientation or other personal characteristics. Discrimination is prohibited,
739

regardless of whether it is based on a fact or assumption concerning the person him/herself or
740

another.
741

Section 3(1) of the Non-Discrimination Act provides that: “Provisions on prohibition of discrimination
742

based on gender and the promotion of gender equality are laid down in the Act on Equality between
743

Women and Men (609/1986).” The Non-Discrimination Act can be applied in cases of multiple
744

discrimination, even if gender is one of the grounds of discrimination [83, 84, s 3(1)].
745

It is worth noting that this definition is broader than in the UK Equality Act. Some protected
746

characteristics are outlined more explicitly; for example, a person discriminated against on the basis
747

of language may be able to bring a claim based on racial discrimination [120]. Unlike many Nordic
748

countries, the Equality Act does not explicitly protect political activity, trade union activity, and does
749

not include “or other personal characteristics” [53].
750

Direct discrimination is defined in Section 10:2
751

Discrimination is direct if a person, on the grounds of personal characteristics, is treated less
752

favourably than another person was treated, is treated or would be treated in a comparable
753

situation.
754

Indirect discrimination is defined in Section 13:3
755

Discrimination is indirect if an apparently neutral rule, criterion or practice puts a person at a
756

disadvantage compared with others as on the grounds of personal characteristics, unless the rule,
757

criterion or practice has a legitimate aim and the means for achieving the aim are appropriate
758

and necessary.
759

Section 11(1) defines justifications for different treatment as:4
760

Different treatment does not constitute discrimination if the treatment is based on legislation and
761

it otherwise has an acceptable objective and the measures to attain the objective are proportionate.
762

Overview of Finnish National Non-Discrimination and Equality Tribunal Decision 216/2017
763

The first case regarding automated decision-making and discrimination was in Finland. The person,
764

referred to as A, was denied credit for online purchases based on a credit rating system employed
765

by a bank. Person A reported the case to the Non-Discrimination Ombudsman (Yhdenvertaisuus-
766

valtuutettu), who brought the case before the National Non-Discrimination and Equality Tribunal
767

(Yhdenvertaisuus- ja tasa-arvolautakunta). The Tribunal found that the bank’s statistical scoring
768

model resulted in direct discrimination based on multiple protected characteristics and was not
769

1Official translation from Finnish, although only legally binding in Swedish (not included) and Finnish:
“Syrjinnän kielto Ketään ei saa syrjiä iän, alkuperän, kansalaisuuden, kielen, uskonnon, vakaumuksen, mielipiteen,
poliittisen toiminnan, ammattiyhdistystoiminnan, perhesuhteiden, terveydentilan, vammaisuuden, seksuaalisen
suuntautumisen tai muun henkilöön liittyvän syyn perusteella. Syrjintä on kielletty riippumatta siitä, perustuuko
se henkilöä itseään vai jotakuta toista koskevaan tosiseikkaan tai oletukseen”
2“Syrjintä on välitöntä, jos jotakuta kohdellaan henkilöön liittyvän syyn perusteella epäsuotuisammin kuin
jotakuta muuta on kohdeltu, kohdellaan tai kohdeltaisiin vertailukelpoisessa tilanteessa.”
3“Syrjintä on välillistä, jos näennäisesti yhdenvertainen sääntö, peruste tai käytäntö saattaa jonkun muita
epäedullisempaan asemaan henkilöön liittyvän syyn perusteella, paitsi jos säännöllä, perusteella tai käytännöllä
on hyväksyttävä tavoite ja tavoitteen saavuttamiseksi käytetyt keinot ovat asianmukaisia ja tarpeellisia.”
4“Erilainen kohtelu ei ole syrjintää, jos kohtelu perustuu lakiin ja sillä muutoin on hyväksyttävä tavoite ja
keinot tavoitteen saavuttamiseksi ovat oikeasuhtaisia.”

17


---Page Break---
justified by an acceptable objective achieved by proportionate measures. Consequently, the Tri-
770

bunal prohibited the bank from continuing this practice and imposed a conditional fine to enforce
771

compliance.
772

The decision-making system in question is for online store financing, which is a purchase-bound, fast
773

and automated credit type very different from regular consumer credit. The credit applied for by the
774

consumer in each situation is also always bound to the purchase and its value, which means that it is
775

more difficult, or even impossible, to undertake detailed requests for information and background
776

checks. The individual investigation of the creditworthiness of customers using personal information
777

and documents, such as salary and tax certificates, may not be suitable for this type of credit.
778

Decision-making Model and Data
779

The company made credit decisions based on data from the internal records of the credit company,
780

information from the credit file, and the score from the company’s internal scoring system.
781

The bank’s scoring system assessed creditworthiness. The scoring system used population statistics
782

and personal attributes to calculate the percentage of people in certain groups with bad credit history
783

and awarded points proportionate to how common bad credit records were in the group in question.
784

The variables used included race, first language, age, and place of residence. The company did not
785

require or investigate the applicant’s income or financial situation.
786

True Data Generating Process and Estimation Error
787

The bank’s scoring model was based on statistical correlations calculated population and groups,
788

including gender, language, age and place of residence, meaning the model is more or less ˆp(y|xp).
789

This model cannot be said to have attempted to model the true underlying data-generating process
790

and instead relied on data that was available regarding protected attributes. It is reasonable to expect
791

that the bank was aware of other legitimate factors that could explain the credit score. Therefore, the
792

model introduces epistemic uncertainty stemming from the lack of information that could have been
793

used to make better predictions, i.e. reasonable legitimate features xl.
794

By solely using the data available, rather than identifying what data would be best to reduce estimation
795

error, the modellers built an automated decision-making system that unlawfully discriminated. We
796

now evaluate how the Tribunal came to those conclusions about the legitimacy of y and x for such a
797

model.
798

Legitimate y
799

The bank argued that the “different treatment does not constitute discrimination if the treatment
800

is based on legislation and has an otherwise acceptable objective and the measures to attain the
801

objective are proportionate.” The Tribunal agreed that “the provision of credit to customers is a
802

business, the purpose of which is to gain profit” and that “the investigation of creditworthiness is as
803

such based on law and that it has the acceptable and justified objective as defined in section 11 of the
804

Non-Discrimination Act”. Therefore, creditworthiness assessment is a legitimate y.
805

However, the Tribunal clarified that “the individual assessment required by the legislation means
806

expressly the assessment of an individual’s credit behaviour, credit history, income level and assets,
807

and not the extension of the impact of models formed on the basis of probability assessments created
808

with statistical methods using the behaviour and characteristics of others, to the individual applying
809

for the credit in the credit decision in such a way that assessment is solely based on such models.”
810

Therefore, to be appropriate and necessary to achieve that aim, the model must consider legitimate
811

features xl.
812

Protected, Legitimate, and Non-Legitimate Variables x
813

Four protected attributes were used as variables in this model xp: age, language, other personal
814

characteristic (place of residence), and gender.
815

18


---Page Break---
The Tribunal acknowledged that age may be a legitimate variable if it had been used in the assessment
816

of creditworthiness mainly when applied to young persons. However, it was not justified in this
817

assessment, given the age of the credit applicant.
818

The Tribunal agreed with the position under European law that gender is prohibited from being used
819

as an actuarial factor in financial services [42].
820

Therefore, these features did not contribute to the accuracy of the model’s prediction in a way
821

that could be argued as part of the true DGP. Therefore, in this case, these xp variables are also
822

non-legitimate variables xn.
823

As explained by the Tribunal, to achieve the legitimate y of undertaking an individual assessment
824

of creditworthiness and ability to repay, the model should have considered, for example, income,
825

expenditure, debt, assets, security and guarantee liabilities, employment and type of employment
826

contract (i.e., permanent or temporary). These features would have been legitimate variables xl by
827

improving the predictive performance of the model to achieve more accurate decisions.
828

Conditional Estimation Parity
829

Using the legitimate variables identified above, we can now consider conditional estimation parity,
830

the difference in estimation error between groups with a protected attribute, given legitimate features.
831

Reducing the error in Eq. 4 is expected to diminish the risk of conditional estimation disparity.
832

However, assessing conditional estimation parity is complex due to inherent challenges in evaluating
833

estimation error.
834

Judges engage this type of reasoning through statistical or theoretical means. In this case, the
835

Ombudsman brought evidence of the effects of the protected characteristics xp on the true prediction.
836

Person A was negatively affected by his age. He was in the age group of 31-40 years old, but if he had
837

been at least 51 years old, he would have received a higher score sufficient for the credit application.
838

If person A spoke Swedish as his first language, he would have received a sufficient score for granting
839

the loan. Finnish-speaking residents received a lower score compared to Swedish-speaking residents.
840

Further, ethnic minorities with an official first language other than Finnish or Swedish were put in an
841

unfavourable position.
842

A would have earned more points based on his residential area if he had lived in a population
843

centre. The bank’s statistical method, which is based on a grid of residential areas, gave A the lowest
844

score because he lives in a sparsely populated area that has not yielded any statistically significant
845

information.
846

Gender impacted the model, where women received a higher score than men. The Tribunal agreed
847

that if the person A had been a woman, he would have been granted the credit.
848

Conclusions
849

This case study demonstrates the intersection between judicial reasoning and our formalisation. To
850

avoid liability for unlawful multiple direct discrimination in this algorithmic decision-making process,
851

the company should have:
852

1. Assessed data legitimacy. While the Tribunal agreed with the target variable (y) as a legitimate
853

aim, they did not believe the features (x) were legitimate for the specific context (Section 2.7).
854

2. Built an accurate model. The bank did not strive to approximate the true DGP p(y|x), and did not
855

use legitimate features xl. Reasonable, necessary, and proportionate steps should have been taken
856

to minimise estimation error and aim for estimation parity (Section 2.3).
857

3. Evaluate differences. The bank should have considered whether there were true and legitimate
858

differences based on protected characteristics and whether they could have been “explained away”
859

by legitimate features (xl) to minimise statistical disparities.
860

These recommendations should be used to help identify and mitigate unlawful discrimination within
861

the specific context of each jurisdiction.
862

19


---Page Break---
NeurIPS Paper Checklist
863

1. Claims
864

Question: Do the main claims made in the abstract and introduction accurately reflect the
865

paper’s contributions and scope?
866

Answer: [Yes]
867

Justification: The abstract and introduction clearly state the claims, contributions, assump-
868

tions and limitations of the paper.
869

Guidelines:
870

• The answer NA means that the abstract and introduction do not include the claims
871

made in the paper.
872

• The abstract and/or introduction should clearly state the claims made, including the
873

contributions made in the paper and important assumptions and limitations. A No or
874

NA answer to this question will not be perceived well by the reviewers.
875

• The claims made should match theoretical and experimental results, and reflect how
876

much the results can be expected to generalize to other settings.
877

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
878

are not attained by the paper.
879

2. Limitations
880

Question: Does the paper discuss the limitations of the work performed by the authors?
881

Answer: [Yes]
882

Justification: Section 1.4 sets out the limitations of this paper.
883

Guidelines:
884

• The answer NA means that the paper has no limitation while the answer No means that
885

the paper has limitations, but those are not discussed in the paper.
886

• The authors are encouraged to create a separate "Limitations" section in their paper.
887

• The paper should point out any strong assumptions and how robust the results are to
888

violations of these assumptions (e.g., independence assumptions, noiseless settings,
889

model well-specification, asymptotic approximations only holding locally). The authors
890

should reflect on how these assumptions might be violated in practice and what the
891

implications would be.
892

• The authors should reflect on the scope of the claims made, e.g., if the approach was
893

only tested on a few datasets or with a few runs. In general, empirical results often
894

depend on implicit assumptions, which should be articulated.
895

• The authors should reflect on the factors that influence the performance of the approach.
896

For example, a facial recognition algorithm may perform poorly when image resolution
897

is low or images are taken in low lighting. Or a speech-to-text system might not be
898

used reliably to provide closed captions for online lectures because it fails to handle
899

technical jargon.
900

• The authors should discuss the computational efficiency of the proposed algorithms
901

and how they scale with dataset size.
902

• If applicable, the authors should discuss possible limitations of their approach to
903

address problems of privacy and fairness.
904

• While the authors might fear that complete honesty about limitations might be used by
905

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
906

limitations that aren’t acknowledged in the paper. The authors should use their best
907

judgment and recognize that individual actions in favor of transparency play an impor-
908

tant role in developing norms that preserve the integrity of the community. Reviewers
909

will be specifically instructed to not penalize honesty concerning limitations.
910

3. Theory Assumptions and Proofs
911

Question: For each theoretical result, does the paper provide the full set of assumptions and
912

a complete (and correct) proof?
913

Answer: [NA]
914

20


---Page Break---
Justification: The paper presents formalisations, which all include relevant assumptions and
915

formatting, but no theoretical results.
916

Guidelines:
917

• The answer NA means that the paper does not include theoretical results.
918

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
919

referenced.
920

• All assumptions should be clearly stated or referenced in the statement of any theorems.
921

• The proofs can either appear in the main paper or the supplemental material, but if
922

they appear in the supplemental material, the authors are encouraged to provide a short
923

proof sketch to provide intuition.
924

• Inversely, any informal proof provided in the core of the paper should be complemented
925

by formal proofs provided in appendix or supplemental material.
926

• Theorems and Lemmas that the proof relies upon should be properly referenced.
927

4. Experimental Result Reproducibility
928

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
929

perimental results of the paper to the extent that it affects the main claims and/or conclusions
930

of the paper (regardless of whether the code and data are provided or not)?
931

Answer: [NA]
932

Justification: The paper does not include experiments.
933

Guidelines:
934

• The answer NA means that the paper does not include experiments.
935

• If the paper includes experiments, a No answer to this question will not be perceived
936

well by the reviewers: Making the paper reproducible is important, regardless of
937

whether the code and data are provided or not.
938

• If the contribution is a dataset and/or model, the authors should describe the steps taken
939

to make their results reproducible or verifiable.
940

• Depending on the contribution, reproducibility can be accomplished in various ways.
941

For example, if the contribution is a novel architecture, describing the architecture fully
942

might suffice, or if the contribution is a specific model and empirical evaluation, it may
943

be necessary to either make it possible for others to replicate the model with the same
944

dataset, or provide access to the model. In general. releasing code and data is often
945

one good way to accomplish this, but reproducibility can also be provided via detailed
946

instructions for how to replicate the results, access to a hosted model (e.g., in the case
947

of a large language model), releasing of a model checkpoint, or other means that are
948

appropriate to the research performed.
949

• While NeurIPS does not require releasing code, the conference does require all submis-
950

sions to provide some reasonable avenue for reproducibility, which may depend on the
951

nature of the contribution. For example
952

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
953

to reproduce that algorithm.
954

(b) If the contribution is primarily a new model architecture, the paper should describe
955

the architecture clearly and fully.
956

(c) If the contribution is a new model (e.g., a large language model), then there should
957

either be a way to access this model for reproducing the results or a way to reproduce
958

the model (e.g., with an open-source dataset or instructions for how to construct
959

the dataset).
960

(d) We recognize that reproducibility may be tricky in some cases, in which case
961

authors are welcome to describe the particular way they provide for reproducibility.
962

In the case of closed-source models, it may be that access to the model is limited in
963

some way (e.g., to registered users), but it should be possible for other researchers
964

to have some path to reproducing or verifying the results.
965

5. Open access to data and code
966

Question: Does the paper provide open access to the data and code, with sufficient instruc-
967

tions to faithfully reproduce the main experimental results, as described in supplemental
968

material?
969

21


---Page Break---
Answer: [NA]
970

Justification: The paper does not include experiments.
971

Guidelines:
972

• The answer NA means that paper does not include experiments requiring code.
973

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
974

public/guides/CodeSubmissionPolicy) for more details.
975

• While we encourage the release of code and data, we understand that this might not be
976

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
977

including code, unless this is central to the contribution (e.g., for a new open-source
978

benchmark).
979

• The instructions should contain the exact command and environment needed to run to
980

reproduce the results. See the NeurIPS code and data submission guidelines (https:
981

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
982

• The authors should provide instructions on data access and preparation, including how
983

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
984

• The authors should provide scripts to reproduce all experimental results for the new
985

proposed method and baselines. If only a subset of experiments are reproducible, they
986

should state which ones are omitted from the script and why.
987

• At submission time, to preserve anonymity, the authors should release anonymized
988

versions (if applicable).
989

• Providing as much information as possible in supplemental material (appended to the
990

paper) is recommended, but including URLs to data and code is permitted.
991

6. Experimental Setting/Details
992

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
993

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
994

results?
995

Answer: [NA]
996

Justification: The paper does not include experiments.
997

Guidelines:
998

• The answer NA means that the paper does not include experiments.
999

• The experimental setting should be presented in the core of the paper to a level of detail
1000

that is necessary to appreciate the results and make sense of them.
1001

• The full details can be provided either with the code, in appendix, or as supplemental
1002

material.
1003

7. Experiment Statistical Significance
1004

Question: Does the paper report error bars suitably and correctly defined or other appropriate
1005

information about the statistical significance of the experiments?
1006

Answer: [NA] .
1007

Justification: The paper does not include experiments.
1008

Guidelines:
1009

• The answer NA means that the paper does not include experiments.
1010

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
1011

dence intervals, or statistical significance tests, at least for the experiments that support
1012

the main claims of the paper.
1013

• The factors of variability that the error bars are capturing should be clearly stated (for
1014

example, train/test split, initialization, random drawing of some parameter, or overall
1015

run with given experimental conditions).
1016

• The method for calculating the error bars should be explained (closed form formula,
1017

call to a library function, bootstrap, etc.)
1018

• The assumptions made should be given (e.g., Normally distributed errors).
1019

• It should be clear whether the error bar is the standard deviation or the standard error
1020

of the mean.
1021

22


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
1022

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
1023

of Normality of errors is not verified.
1024

• For asymmetric distributions, the authors should be careful not to show in tables or
1025

figures symmetric error bars that would yield results that are out of range (e.g. negative
1026

error rates).
1027

• If error bars are reported in tables or plots, The authors should explain in the text how
1028

they were calculated and reference the corresponding figures or tables in the text.
1029

8. Experiments Compute Resources
1030

Question: For each experiment, does the paper provide sufficient information on the com-
1031

puter resources (type of compute workers, memory, time of execution) needed to reproduce
1032

the experiments?
1033

Answer: [NA] .
1034

Justification: The paper does not include experiments.
1035

Guidelines:
1036

• The answer NA means that the paper does not include experiments.
1037

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
1038

or cloud provider, including relevant memory and storage.
1039

• The paper should provide the amount of compute required for each of the individual
1040

experimental runs as well as estimate the total compute.
1041

• The paper should disclose whether the full research project required more compute
1042

than the experiments reported in the paper (e.g., preliminary or failed experiments that
1043

didn’t make it into the paper).
1044

9. Code Of Ethics
1045

Question: Does the research conducted in the paper conform, in every respect, with the
1046

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
1047

Answer: [Yes]
1048

Justification: Our research rigorously addresses the ethical code outlined by the conference,
1049

particularly focusing on issues related to safety, security, discrimination, and fairness. We
1050

have proactively identified and discussed potential harmful outcomes, particularly those
1051

involving discrimination and misuse in the contexts of legal and ethical standards. Further-
1052

more, we provide recommendations to mitigate these risks, underscoring our commitment
1053

to the responsible development and application of technology that respects human rights
1054

and societal values. The paper does not contain research involving human subjects or
1055

participants, it does not conduct experiments or have data-related concerns.
1056

Guidelines:
1057

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
1058

• If the authors answer No, they should explain the special circumstances that require a
1059

deviation from the Code of Ethics.
1060

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
1061

eration due to laws or regulations in their jurisdiction).
1062

10. Broader Impacts
1063

Question: Does the paper discuss both potential positive societal impacts and negative
1064

societal impacts of the work performed?
1065

Answer: [Yes]
1066

Justification: Our research contributes to bridging the gap between legal standards and
1067

algorithmic fairness, aiming to enhance the integrity and fairness of automated decision-
1068

making systems. This has significant implications for improving equity in critical areas
1069

where algorithmic decisions are increasingly prevalent. We also the risks of unfair treatment
1070

based on model or data biases or misinterpretation of the legal doctrines we study. We
1071

explore the potential for unintended consequences even when the technology functions
1072

as intended, such as the reinforcement of existing societal biases under the guise of legal
1073

compliance. To mitigate these risks, we propose specific safeguards to prevent unlawful
1074

discrimination in systems.
1075

23


---Page Break---
Guidelines:
1076

• The answer NA means that there is no societal impact of the work performed.
1077

• If the authors answer NA or No, they should explain why their work has no societal
1078

impact or why the paper does not address societal impact.
1079

• Examples of negative societal impacts include potential malicious or unintended uses
1080

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
1081

(e.g., deployment of technologies that could make decisions that unfairly impact specific
1082

groups), privacy considerations, and security considerations.
1083

• The conference expects that many papers will be foundational research and not tied
1084

to particular applications, let alone deployments. However, if there is a direct path to
1085

any negative applications, the authors should point it out. For example, it is legitimate
1086

to point out that an improvement in the quality of generative models could be used to
1087

generate deepfakes for disinformation. On the other hand, it is not needed to point out
1088

that a generic algorithm for optimizing neural networks could enable people to train
1089

models that generate Deepfakes faster.
1090

• The authors should consider possible harms that could arise when the technology is
1091

being used as intended and functioning correctly, harms that could arise when the
1092

technology is being used as intended but gives incorrect results, and harms following
1093

from (intentional or unintentional) misuse of the technology.
1094

• If there are negative societal impacts, the authors could also discuss possible mitigation
1095

strategies (e.g., gated release of models, providing defenses in addition to attacks,
1096

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
1097

feedback over time, improving the efficiency and accessibility of ML).
1098

11. Safeguards
1099

Question: Does the paper describe safeguards that have been put in place for responsible
1100

release of data or models that have a high risk for misuse (e.g., pretrained language models,
1101

image generators, or scraped datasets)?
1102

Answer: [NA] .
1103

Justification: The paper does not release data or models.
1104

Guidelines:
1105

• The answer NA means that the paper poses no such risks.
1106

• Released models that have a high risk for misuse or dual-use should be released with
1107

necessary safeguards to allow for controlled use of the model, for example by requiring
1108

that users adhere to usage guidelines or restrictions to access the model or implementing
1109

safety filters.
1110

• Datasets that have been scraped from the Internet could pose safety risks. The authors
1111

should describe how they avoided releasing unsafe images.
1112

• We recognize that providing effective safeguards is challenging, and many papers do
1113

not require this, but we encourage authors to take this into account and make a best
1114

faith effort.
1115

12. Licenses for existing assets
1116

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
1117

the paper, properly credited and are the license and terms of use explicitly mentioned and
1118

properly respected?
1119

Answer: [NA] .
1120

Justification: The paper does not use existing assets.
1121

Guidelines:
1122

• The answer NA means that the paper does not use existing assets.
1123

• The authors should cite the original paper that produced the code package or dataset.
1124

• The authors should state which version of the asset is used and, if possible, include a
1125

URL.
1126

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
1127

24


---Page Break---
• For scraped data from a particular source (e.g., website), the copyright and terms of
1128

service of that source should be provided.
1129

• If assets are released, the license, copyright information, and terms of use in the
1130

package should be provided. For popular datasets, paperswithcode.com/datasets
1131

has curated licenses for some datasets. Their licensing guide can help determine the
1132

license of a dataset.
1133

• For existing datasets that are re-packaged, both the original license and the license of
1134

the derived asset (if it has changed) should be provided.
1135

• If this information is not available online, the authors are encouraged to reach out to
1136

the asset’s creators.
1137

13. New Assets
1138

Question: Are new assets introduced in the paper well documented and is the documentation
1139

provided alongside the assets?
1140

Answer: [NA] .
1141

Justification: The paper does not release new assets.
1142

Guidelines:
1143

• The answer NA means that the paper does not release new assets.
1144

• Researchers should communicate the details of the dataset/code/model as part of their
1145

submissions via structured templates. This includes details about training, license,
1146

limitations, etc.
1147

• The paper should discuss whether and how consent was obtained from people whose
1148

asset is used.
1149

• At submission time, remember to anonymize your assets (if applicable). You can either
1150

create an anonymized URL or include an anonymized zip file.
1151

14. Crowdsourcing and Research with Human Subjects
1152

Question: For crowdsourcing experiments and research with human subjects, does the paper
1153

include the full text of instructions given to participants and screenshots, if applicable, as
1154

well as details about compensation (if any)?
1155

Answer: [NA] .
1156

Justification: The paper does not involve crowdsourcing nor research with human subjects.
1157

Guidelines:
1158

• The answer NA means that the paper does not involve crowdsourcing nor research with
1159

human subjects.
1160

• Including this information in the supplemental material is fine, but if the main contribu-
1161

tion of the paper involves human subjects, then as much detail as possible should be
1162

included in the main paper.
1163

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1164

or other labor should be paid at least the minimum wage in the country of the data
1165

collector.
1166

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1167

Subjects
1168

Question: Does the paper describe potential risks incurred by study participants, whether
1169

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1170

approvals (or an equivalent approval/review based on the requirements of your country or
1171

institution) were obtained?
1172

Answer: [NA] .
1173

Justification: The paper does not involve crowdsourcing nor research with human subjects.
1174

Guidelines:
1175

• The answer NA means that the paper does not involve crowdsourcing nor research with
1176

human subjects.
1177

25


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
1178

may be required for any human subjects research. If you obtained IRB approval, you
1179

should clearly state this in the paper.
1180

• We recognize that the procedures for this may vary significantly between institutions
1181

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1182

guidelines for their institution.
1183

• For initial submissions, do not include any information that would break anonymity (if
1184

applicable), such as the institution conducting the review.
1185

26


---Page Break---
