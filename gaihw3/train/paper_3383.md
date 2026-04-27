Challenges with unsupervised LLM knowledge
discovery

Anonymous Author(s)
Affiliation
Address
email

Abstract

We reveal novel pathologies in existing unsupervised methods seeking to discover
1

latent knowledge from large language model (LLM) activations—instead of knowl-
2

edge they seem to discover whatever feature of the activations is most prominent.
3

These methods search for hypothesised consistency structures of latent knowledge.
4

We first prove theoretically that arbitrary features (not just knowledge) satisfy the
5

consistency structure of a popular unsupervised knowledge-elicitation method:
6

contrast-consistent search [9]. We then present a series of experiments showing
7

settings in which this and other unsupervised methods result in classifiers that
8

do not predict knowledge, but instead predict a different prominent feature. We
9

conclude that existing unsupervised methods for discovering latent knowledge
10

are insufficient, and we contribute sanity checks to apply to evaluating future
11

knowledge elicitation methods. We offer conceptual arguments grounded in identi-
12

fication issues such as distinguishing a model’s knowledge from that of a simulated
13

character’s that are likely to persist in future unsupervised methods.
14

1
Introduction
15

Large language models (LLMs) perform well across a variety of tasks [30, 10] in a way that suggests
16

they systematically incorporate information about the world [7]. As a shorthand for the real-world
17

information encoded in the weights of an LLM we could say that the LLM encodes knowledge.
18

Accessing that knowledge is hard, because the factual statements an LLM outputs do not reliably
19

describe it [23, 2, 32]. For example, LLMs might repeat common misconceptions [26] or strategically
20

deceive users [36]. If we could elicit the latent knowledge of an LLM [11] it would allow us to detect
21

and mitigate “dishonesty” [17]. It would also help when supervising outputs that are difficult to
22

understand as well as improving scientific understanding of the inner workings of LLMs. Importantly,
23

this must be done without supervision because we lack a ground truth for what the model “knows”,
24

as opposed to what we know.
25

Contrast-consistent search (CCS) [9] is a prominent method proposed to address this problem by
26

assuming that “knowledge” satisfies a consistency structure that few other features in an LLM are
27

likely to satisfy. They use this consistency to construct a classifier which they claim detects a model’s
28

latent knowledge, a claim which is widely repeated in the literature (see Appendix B). We refute
29

these claims by identifying classes of LLM features that also satisfy this consistency structure but are
30

not knowledge. We prove two theorems: 1) a class of arbitrary binary classifiers are optimal under
31

the CCS loss; 2) any classifier can be transformed to an arbitrary classifier with the same CCS loss.
32

The upshot is that the CCS consistency structure is more than just slightly imprecise in identifying
33

knowledge—it is compatible with arbitrary patterns.
34

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
qn

LLM activations

x4

+ = Great movie…
             Alice… positive

        It is positive

x4

+ = Great movie…
            Alice… positive

        It is negative

x3

- = Didn’t like it….
                Alice… negative
                It is positive

q2 = The best movie ever…

q1 = I hated this movie…

x2

+= The best movie ever…
          Alice… negative
          It is positive

x1

+ = I hated this movie…

   Alice… positive
            It is positive

x3

- = Didn’t like it….
                Alice… negative
                It is negative
x2

- = The best movie ever…
          Alice… negative
          It is negative

x1

- = I hated this movie…

  Alice… positive
            It is negative

𝜙(x1

+), 𝜙(x1

-)
𝜙(x2

+), 𝜙(x2

-)
.
.
.
𝜙(xn

+), 𝜙(xn

-)

Without “Alice…”
With “Alice…”

Classification boundary 

according to Review 

(blue/orange)

Classification boundary 

according to Alice 

(light/dark)

Unsupervised learning

Figure 1: Prominent features distract unsupervised latent knowledge detectors (see Section 4.2).
Left: We apply two transformations to a dataset of movie reviews, {qi}. First (novel to us) we insert
a distracting feature by appending either “Alice thinks it’s positive” or “Alice thinks it’s negative” at
random to each question. Second, we create contrast pairs [9], (x+
i , x−
i ), appending “It is positive” or
“It is negative” to each. Middle: The LLM activations for these strings are ϕ(x+
i ), ϕ(x−
i ). Right: A
PCA visualisation of the top-3 activation dimensions. Without “Alice ...”, a classifier finds the review
sentiment (orange/blue). But with “Alice ...” a classifier finds Alice’s opinion (light/dark) ignoring
review sentiment.

We then show that other unsupervised methods in addition to CCS empirically do not discover
35

knowledge, regardless of any inductive biases that might hypothetically be present. Two didactic
36

experiments show that these methods can latch onto artificial distracting features instead of knowledge.
37

Our third experiment moves towards realism by showing that these knowledge-discovery methods
38

can latch onto implicit opinions. The fourth is almost fully natural: we show that the method’s results
39

are highly sensitive to reasonable prompt variants which have been used in the literature.
40

We conclude that existing unsupervised knowledge-discovery methods are insufficient in practice, and
41

we propose principles for evaluating knowledge elicitation methods to prevent future “false-positives”
42

in the literature. We hypothesise that our conclusions will generalise to more sophisticated methods,
43

though perhaps not the exact experimental results: using different consistency structures of knowledge
44

will likely suffer from similar issues to what we show here. Our key contributions are as follows:
45

• We prove that arbitrary features satisfy the CCS loss equally well.
46

• We show that unsupervised methods detect prominent features that are not knowledge.
47

• We show that the features discovered by unsupervised methods are sensitive to prompts and
48

that we lack principled reasons to pick any particular prompt.
49

2
Background
50

Contrastive LLM activations. We focus on methods that train probes [1] using LLM activation
51

data. This data is constructed using contrast pairs [9]. A contrast pair is a pair of strings with opposite
52

‘claim’ for some characteristic of interest which can be used to study the contrast in how an LLM
53

represents that characteristic. For example, a contrast pair might be “Are cats mammals? Yes.” and
54

“Are cats mammals? No.” Potentially, pairs like this could then be used to study how LLMs represent
55

correctly/incorrectly answered questions.
56

Burns et al. [9] show how to generate such contrast pairs from a dataset of binary questions, Q =
57

{qi}N
i=1, such as “Are cats mammals?” by, for example, appending “Yes.” and “No.” for a positive
58

and negative member of a contrast pair (x+
i , x−
i ). The LLM’s representations of each member of
59

the pair can then be computed by looking at the activations from an intermediate layer after the
60

sequence of tokens, ϕ(x+
i ) and ϕ(x−
i ). If one just looked at these activations, their differences might
61

be dominated just by the presence of the tokens “Yes.” or “No.” Burns et al. [9] therefore propose a
62

normalisation step which strips away the average effect of those tokens across the dataset: setting
63

˜ϕ(x+/−
i
) :=
 
ϕ(x+/−
i
) −µ+/−
/σ+/−where µ+/−, σ+/−are {ϕ(x+/−
i
)}N
i=1’s mean and standard
64

deviation. This is meant to remove these tokens’ unintended influence but prior work questions this,
65

and some of our results also question this.
66

Contrast-consistent Search (CCS) [9]. An unsupervised learning algorithm using contrast pairs
67

constructed to reflect a characteristic of interest to recover the features of LLM activations that
68

2


---Page Break---
represent that characteristic. CCS uses the LLM’s representations to predict correct labels, intending
69

to study cases where the LLM’s knowledge is true. CCS assumes that LLM knowledge representations
70

are credences which follow probabilistic laws. Softly encoding this constraint, they minimise
71

LCCS =
XN

i=1

Lcons
z
}|
{

p(x+
i ) −(1 −p(x−
i ))
2 +

Lconf
z
}|
{
min

p(x+
i ), p(x−
i )
	2
(1)

for a function from the normalised LLM activations from the contrast pairs: p(x) = σ(θT ˜ϕ(x) + b)
72

(a linear function with sigmoid). The motivation is that the Lcons encourages negation-consistency
73

(that a statement and its negation should have probabilities that add to one), and Lconf encourages
74

confidence to avoid p(x+
i ) ≈p(x−
i ) ≈0.5. For inference on a question qi the average prediction is
75

˜p(qi) =

p(x+
i ) + (1 −p(x−
i ))

/2 and then the induced classifier is fp(qi) = I [˜p(qi) > 0.5]. 1
76

Activation clustering with PCA and k-means.
We consider two other unsupervised learning
77

methods. In both cases we cluster the difference in contrastive activations, {˜ϕ(x+
i ) −˜ϕ(x−
i )}N
i=1. In
78

one case, these are clustered by applying principal component analysis (PCA) and thresholding the
79

top component at 0 [9].2 The other clusters with k-means with two clusters.
80

Logistic regression. As a supervised baseline, we use logistic regression on concatenated contrastive
81

activations, {(˜ϕ(x+
i ), ˜ϕ(x−
i ))}N
i=1 with labels ai, and treat this as a ceiling (since it uses labels).
82

Random baseline. We compare to a random baseline using a probe with random parameter values,
83

treating that as a floor (as it does not learn from input data) [35]. Further details are in Appendix C.3.
84

3
Theoretical Results
85

Our theoretical results focus on CCS, showing that CCS’s consistency structure isn’t specific to
86

knowledge. This implies that arguments for CCS’s effectiveness cannot be grounded in conceptual or
87

principled motivations from the loss construction. In later sections, we also address other methods
88

which do not rely on these strong consistency assumptions and show that heuristic arguments
89

grounded in inductive biases do not support using any of these as knowledge-discovery methods.
90

As illustration, consider the IMDb sentiment classification task [28]. A given question qi considers
91

whether a movie review has a particular sentiment, s(qi) := I [qi has positive sentiment], and is
92

converted into a contrast pair of x+
i and x−
i , each of which has a claim c(·) about the sentiment.
93

Specifically, c(x+
i ) = 1, a claim that the sentiment is positive, and c(x−
i ) = 0 for negative. The
94

desired probe, p∗, detecting the truth feature must check whether the sentiment and the claim agree.
95

This can be done by XOR (denoted ⊕) of the sentiment and the claim:
96

p∗(x±
i ) := I

x±
i is false

= s(qi) ⊕c(x±
i ).
(2)

The induced probe for this feature is the sentiment as desired: fp∗(qi) = s(qi). Our key insight is that
97

the CCS loss is low just because of this XOR, not the sentiment, and so the same construction can
98

work for arbitrary features of the question: given some feature h, the probe p(x±
i ) = h(qi) ⊕c(x±
i )
99

gets low CCS loss and has an induced probe h.
100

Theorem 1. Let feature h : Q →{0, 1}, be any arbitrary map from questions to binary outcomes. Let
101

(x+
i , x−
i ) be the contrast pair corresponding to question qi and let c(x+
i ) = 1, c(x+
i ) = 0. Then the
102

probe defined as p(x±
i ) = h(qi) ⊕c(x±
i ) achieves optimal loss, and the averaged prediction satisfies
103

˜p(qi) = h(qi).
104

That is, the classifier that CCS finds is under-specified: for any binary feature, h, on the questions,
105

there is a probe with optimal CCS loss that induces that feature. The proof comes directly from
106

inserting our constructive probes into the loss definition—equal terms cancel to zero (see Appendix A).
107

1Because the predictor learns the contrast between activations, not absolute classes, Burns et al. [9] disam-
biguate by assuming that fp(qi) = 1 to correspond to label ai = 1 if the accuracy is greater than 0.5 (else it
corresponds to ai = 0). We call this further step truth-disambiguation and apply it to all methods similarly.
2Emmons [16] point out that this is roughly 97-98% as effective as CCS according to the experiments
in Burns et al. [9], suggesting that contrast pairs and standard unsupervised learning are doing much of the
work, and CCS’s consistency loss may not be important. Our experiments largely agree with this finding—see
Appendix D.6 for an additional experiment showing agreement between the predictions of these methods.

3


---Page Break---
In Thm. 1, the probe p is binary since h is binary, but in practice probe outputs are produced by a
108

sigmoid and so are in (0, 1). Can we say anything about this setting? We show that it is possible to
109

transform a soft probe for one feature into a soft probe for any other arbitrary feature. In the binary
110

case, the desired probe for feature h1 is p1 = h1 ⊕c, and the desired probe for h2 is h2 ⊕c. So, we
111

have p2 = p1 ⊕h1 ⊕h2. To generalize this to soft probes, we extend ⊕as follows:
112

(a ⊕b)(x) := [1 −a(x)] b(x) + [1 −b(x)] a(x).
(3)

In addition, we correct the CCS loss to fix an unmotivated downwards bias in the loss proposed by
113

Burns et al. [9] (see Appendix A.2). We also use this symmetrized loss in our experiments. After
114

this, the transformation between probes works as desired, proving that there is an arbitrary classifier
115

encoded by a probe with identical CCS loss to the original:
116

Theorem 2. Let g : Q →{0, 1}, be any arbitrary map from questions to binary outputs. Let
117

(x+
i , x−
i ) be the contrast pair corresponding to question qi. Let p be a probe, whose average result
118

˜p = 0.5

p(x+
i ) + (1 −p(x−
i ))

induces a classifier fp(qi) = I [˜p(qi) > 0.5]. Define the transformed
119

probe p′(x±
i ) = p(x±
i ) ⊕[fp(qi) ⊕g(qi)]. Then LCCS(p′) = LCCS(p) and p′ induces the classifier
120

fp′(qi) = g(qi).
121

However, which probe is actually learned depends on inductive biases; these could depend on the
122

prompt, optimization algorithm, or model choice. These theorems prove that optimal arbitrary probes
123

exist, but not necessarily that they are actually learned or that they are expressible in the probe’s
124

function space. But for inductive biases, no robust argument ensures the desired behaviour. The
125

feature that is most prominent—favoured by inductive biases—could turn out to be knowledge,
126

but it could equally turn out to be the contrast-pair mapping itself (which is partly removed by
127

normalisation) or anything else. We do not have any theoretical reason to think that CCS discovers
128

knowledge probes. In fact, experimentally, we now show that, in practice, several methods including
129

CCS often discover probes for features other than knowledge.
130

4
Experiments
131

Our experiments a structured didactically. We begin with simplified experiments that use unrealistic
132

but clear-cut interventions to develop understanding, gradually increasing realism. Section 4.4 closes
133

with an experiment that uses entirely natural prompts that have been used by others, demonstrating
134

that these issues appear in practice. Unless otherwise noted, experiments follow details below.
135

Datasets. We investigate three datasets used by Burns et al. [9].3 The IMDb dataset of movie reviews
136

classifies positive/negative sentiment [28], BoolQ [13] answers yes/no questions about a passage,
137

DBpedia [3] is text topic-classification. Prompt templates for each dataset are in Appendix C.1.4
138

Language Models. We use three different language models. To directly compare to Burns et al.
139

[9] we use T5-11B, [34] with 11 billion parameters. We further use an instruction fine-tuned version
140

of T5-11B called T5-FLAN-XXL, [12] to understand the effect of instruction fine-tuning. Both
141

are encoder-decoder architectures, and we use the encoder output for our activations. We also use
142

Chinchilla-70B [21], with 70 billion parameters, which is larger scale, and a decoder-only architecture.
143

We take activations from layer 30 (of 80) of this model, though see Appendix D.2.3 for results on
144

other layers, often giving similar results. Notably, K-means and PCA have good performance at layer
145

30 with less seed-variance than CCS, suggesting contrast pairs and standard unsupervised learning,
146

rather than the CCS consistency structure, are key (see Footnote 2).
147

Experiment Setup. In each experiment we compare a default setting which is the same/similar to
148

that used in [9] to a modified setting that we introduce in order to show an effect – differing only
149

in their text prompt. We then generate contrastive activations and train probes using the methods
150

in Section 2: CCS, PCA, k-means, random and logistic regression. Training details can be found
151

in Appendix C.3. For each method we use 50 random seeds. Our figures in general come in two
152

types: violin plots which compare the accuracy of different methods; and three-dimensional PCA
153

projections of the activations to visualise how they are grouped. We show one dataset and model,
154

other datasets and models, shown in the appendix, are similar except where discussed.
155

3Others were excluded for legal reasons or because Burns et al. [9] found low predictive accuracy on them.
4We use a single prompt template rather than the multiple used in Burns [8], as multiple templates did not
systematically improve performance of the methods, but increase experiment complexity, see Appendix D.5.

4


---Page Break---
Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

(a) Variation in accuracy

Distractor label
Banana
Shed

Review Sentiment
Positive
Negative

X

−50

0

50
Y

−30

0

30

X

−15

0

15

Y

−20

0

20

Default prompt
Banana/Shed prompt

(b) PCA Visualisation

Figure 2: Discovering random words. Chinchilla, IMDb. (a) The methods distinguish whether
the prompts end with banana/shed rather than the review sentiment. (b) PCA visualisation of top-
3 activation dimensions, in default (left) and modified (right) settings, shows the clustering into
banana/shed (light/dark) rather than review sentiment (blue/orange).

4.1
Discovering random words
156

Motivated by our theoretical results, we first introduce a distracting binary feature and show the
157

unsupervised methods discover this feature rather than knowledge. We focus here on IMDB and
158

Chinchilla (see Appendix D.1 for other datasets and models with similar results). Our default prompts
159

use the standard template from Burns et al. [9] inserting different reviews and labels “positive” or
160

“negative”.
161

Our modified prompts further append a full stop and space, then one of two random words, “Banana”
162

and “Shed”. In the language of Thm. 1 we take a random partition of question indices, {1, . . . , N} =
163

I0 ∪I1, with |I0| = |I1|, and set the binary feature h such that h(qi) = 0 for i ∈I0 and h(qi) = 1 for
164

for i ∈I1. “Banana” is inserted if h(qi) = 0, and “Shed” is inserted if h(qi) = 1. See Figure 1 for
165

illustration – though here we append “Banana” or “Shed” to the end, rather than inserting “Alice...”.
166

Our results are shown in Figure 2a, displaying accuracy of each method (x-axis groups). Default
167

prompts are blue and modified banana/shed prompts are red. We look at the standard ground-truth
168

accuracy metric (dark), as well as a modified accuracy metric that measures whether Banana or
169

Shed was inserted (light). We see that for all unsupervised methods, default prompts (blue) score
170

highly on ground truth accuracy (dark blue), in line with results in Burns et al. [9]. However, for
171

the banana/shed prompts we see 50%, random chance, on ground truth accuracy (dark red). On
172

Banana/Shed accuracy (light red) both PCA and K-means score highly, while CCS shows a bimodal
173

distribution with a substantial number of seeds with 100% Banana/Shed accuracy – seeds differ only
174

in the random initialisation of the probe parameters. The takeaway is that CCS and other unsupervised
175

methods do not optimise for ground-truth knowledge, but rather track whatever feature (in this case,
176

banana/shed) is most prominent in the activations.
177

Figure 2b shows a visualisation of the top three components of PCA for the default (left) and
178

modified (right) prompts. In the modified case we see a prominent grouping of the data into dark/light
179

(banana/shed) and, less prominently, into blue/orange (the review). This provides visual evidence that
180

both features (ground-truth and banana/shed) are represented, but the one which is most prominent in
181

this case is banana/shed, in correspondence with Figure 2a.
182

4.2
Discovering an explicit opinion
183

It is unlikely that such a drastic feature, ending with “Banana”/“Shed”, would actually exist in a real
184

dataset. These words had nothing to do with the rest of the text. In our second experiment we make a
185

more realistic modification: inserting a character’s explicit opinion of whether the review is positive
186

or negative. What we will find is that the unsupervised methods learn to predict the character’s
187

opinion, instead of the sentiment of the actual review, presumably by learning a probe that detects
188

whether the claimed sentiment agrees with the character’s opinion.
189

5


---Page Break---
Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

(a) Variation in accuracy

Distractor label
Alice: Negative
Alice: Positive

Review Sentiment
Positive
Negative

X

−80

0

80
Y

−50

0

50

X

−100

0

100
Y

−60

0

60

Default prompt
Alice-opinion prompt

(b) PCA Visualisation

Figure 3: Discovering an explicit opinion. Chinchilla 70B. (a) When Alice’s opinion is given
(red) unsupervised methods predict her opinion (light red) but not true review sentiment (dark red),
suggesting the linear probe detects whether the claim agrees with Alice. Blue shows the default
prompt without Alice. (b) PCA visualisation of top-3 activation dims., in default (L) and modified (R)
settings, where clustering discover Alice’s opinion (light/dark), over review sentiment (blue/orange).

We again focus here on IMDB and Chinchilla (see Appendix D.2.1 for other datasets and models
190

with similar results) and we use the same default prompt as before. For the modified setting we use a
191

prompt style where Alice’s opinion is inserted into a random partition of questions:
192

Alice is an expert at classifying the sentiment of movie reviews. Everyone agrees with her, and they
193

always copy her answers. Consider the following review:
194

Probably my all-time favorite movie, a story of...
195

Alice thinks it is {positive | negative}.
196

What do you think? \n I think it is [label]
197

Similarly to our previous results, we show that CCS and other unsupervised methods have low
198

ground-truth accuracy, but high accuracy at predicting Alice’s belief (Figure 3a). Default prompts are
199

blue and modified prompts (containing Alice’s opinion) are red. We look at the standard ground-truth
200

accuracy metric (dark), as well as “Alice Accuracy” metric (light) that measures whether “Alice
201

thinks it is positive” or“Alice thinks it is negative” was inserted. Here, the CCS results are no longer
202

bimodal.
203

A visualisation of the top three components of a PCA for the activations show that the most prominent
204

grouping of the data is into dark/light (Alice’s opinion) and that these then have subgroups along
205

blue/orange (the review).
206

When we use a model that has been instruction-tuned (T5-FLAN-XXL) we see a similar pattern
207

Appendix D.2.1 Figure 11, although a similarly clear result requires a more emphatic view from the
208

character by repeating the opinion (“I think it is positive. They fully express positive views. I’m sure
209

you also think it is positive. It’s clearly positive.”). An ablation of the number of repetitions can be
210

found in Appendix D.2.2, Figure 12.
211

4.3
Discovering an implicit opinion
212

The previous experiment explicitly gave Alice’s opinion, “Alice thinks it is positive”. While this is
213

more realistic than Banana/Shed, it is still rather artificial in the sense we do not expect real datasets
214

to have such a clear syntactical textual binary feature. In the next experiment for the modified prompt
215

we instead explain Alice’s position in general, and keep that the same in all instances, making it more
216

of an implicit, semantic rather than syntactic feature.
217

We use the DBpedia topic classification dataset [3] to construct a binary classification task to classify
218

the topic of a text from two choices. There are fourteen categories such as company, animal, film. In
219

the default case contrast pairs are constructed using a simple few-shot prompt setting up the task of
220

identifying the topic of a sentence with the character “Alice” answering the questions correctly.
221

6


---Page Break---
Prompt template
Default
Anti-capitalist

Data subset
 Company
 Non-company

CCS

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

(a) Variation in accuracy

Data subset
Non-Company
Company

Correct answer Choice 1

Choice 2

X
0

150

Y

−100

0

100

X
0

150

Y

−100

0

100

Default prompt
Anti-capitalist prompt

(b) PCA Visualisation

Figure 4: Discovering an implicit opinion. (a) Default (blue) and modified (red) for company (dark)
and non-company (light) data. The modified setting on company data (dark red) leads to a bimodal
distribution for CCS with almost half of the probes (differing only in random initialisation) learning
Alice’s opinion. In contrast, it performs relatively well over all other categories (light red). (b) PCA:
Left – default activations show a possible separation along X-axis corresponding to topic choice
(blue vs. orange) and further separation into company/non-company (light/dark). Right – modified
activations show a more pronounced company/non-company split. All results are for Chinchilla 70B.

In the modified setting5, Alice answers the few-shot examples correctly, except when topic is company
222

– and in that case gives explanations like “[...] Alice always says the wrong answer when the topic of
223

the text is company, because she doesn’t like capitalism [...]”. What we are looking for is what the
224

unsupervised methods predict on the final example when Alice has not yet stated an opinion: will it
225

predict the correct answer, ignoring how Alice previously answered incorrectly about company; or
226

will it predict Alice’s opinion, answering incorrectly about company?
227

To highlight the effect, we use a subset dataset where 50% of sentences are about “company”,
228

and 50% have one of the remaining thirteen categories (non-company) as a topic. We apply truth-
229

disambiguation only to the subset with non-company topics, so that we can see the possible effect of
230

predicting incorrectly on company data (otherwise the assignment might be flipped).
231

Our results are shown in Figure 4. We look at default prompts (blue) and modified prompts (red)
232

and split the data into whether the topic is company (dark) or non-company (light) and look at the
233

standard ground-truth accuracy metric. The default setting (blue) produces high accuracy classifiers
234

both when the topic is company (dark blue) and other categories (light blue). In the modified setting
235

(red) CCS gives a bimodal distribution when the topic is company (dark red), with almost half of the
236

probes (differing only in random initialisation) predicting Alice’s opinion, rather than the actual topic.
237

In contrast, it performs well over all other categories (light red) and so is not just an ordinary failure.
238

Other unsupervised methods are less sensitive to the modified setting, scoring high accuracy when
239

the topic is company.
240

However, when we visualise the first three PCA dimensions of the contrast pair activations (Figure 4b)
241

we see four distinct clusters in the modified prompt case (right) showing how a detector might cluster
242

either the actual topic choice (orange vs blue) or based on the data subset: non-company vs company
243

(light vs dark). This shows these methods are still sensitive to the modified setting, which was not
244

evident from the accuracy metric alone.
245

4.4
Prompt template sensitivity
246

The next experiment is more natural because, rather than introducing a feature deliberately, we
247

examine three natural prompt templates which have appeared in the literature and show how these
248

change the discovered feature. We use TruthfulQA [26], a difficult question answering dataset which
249

exploits the fact that LLMs tend to repeat common misconceptions.
250

5Full prompt templates are provided in Appendix C.1.3, Implicit Opinion: Default and Anti-capitalist.

7


---Page Break---
CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

Default
Literal
Professor

(a) Variation in accuracy

X
0

60
Y

−50

0

50
X

−80

0

80
Y

−60

0

60

X

−60
0
60
Y

−60

0

60
False
True

Default
Literal
Professor
(b) PCA Visualisation

Figure 5: Prompt sensitivity on TruthfulQA [26] for Chinchilla70B. (a) In default setting (blue),
accuracy is poor. When in the literal/professor (red, green) setting, accuracy improves, showing the
unsupervised methods are sensitive to irrelevant aspects of a prompt. (b) PCA of the activations based
on ground truth, blue vs. orange, in the default (left), literal (middle) and professor (right) settings.
We see do not see ground truth clusters by default, but see this with other prompts.

We find that a “non-default” prompt gives the “best performance” in the sense of the highest test-set
251

accuracy. This highlights the reliance of unsupervised methods on implicit inductive biases which
252

cannot be set in a principled way. It is not clear which prompt is the best one for eliciting the model’s
253

latent knowledge. Given that the choice of prompt appears to be a free variable with significant effect
254

on the outcomes, conceptual motivations for the loss do not imply a principled foundation for the
255

resulting classifier.
256

Our prompt templates can be found in Appendix C.1.4. Our “default” template is adapted directly
257

from Burns et al. [9]. Two modified templates are adapted from Lin et al. [26]6 in which a Professor
258

character is instructed to interpret questions literally. We used this text verbatim inserted into an
259

instructing template in order to make sure that we were looking at natural prompts that people
260

might ordinarily use without trying to see a specific result. We also try a “literal” prompt, removing
261

explicitly mentioning a Professor, in case explicitly invoking a character matters.
262

Results are shown in Figure 5a for Chinchilla70B. The default setting (blue) gives worse accuracy
263

than the literal/professor (red, green) settings, especially for PCA and k-means. PCA visualisations
264

are shown in Figure 5b, coloured by whether the question is True/False, in the default (left), literal
265

(middle) and professor (right) settings. We see clearer clusters in the literal/professor settings. Other
266

models are shown in Appendix D.4, with less systematic differences between prompts, though the
267

accuracy for K-means in the Professor prompt for T5-FLAN-XXL are clearly stronger than others.
268

5
Related Work
269

We want to detect when an LLM is dishonest [23, 2, 32], outputting text which contradicts its encoded
270

knowledge [17]. An important part of this is to elicit latent knowledge from a model [11]. There has
271

been some debate as to whether LLMs “know/believe” anything [6, 37, 24] but, for us, the important
272

thing is that something in an LLM’s weights causes it to make consistently successful predictions,
273

and we would like to access that. Zou et al. [40] train unsupervised probes for a range of concepts
274

including honesty, using pairs which need not take opposite truth values (as in Burns et al. [9]).
275

Belrose et al. [5] use unsupervised probes on intermediate LLM layers to elicit latent predictions.
276

Others (see [19] and references therein) aim to detect when a model has knowledge/beliefs about the
277

world, to improve truthfulness.
278

Contrast-consistent search (CCS) [9] attempts to elicit latent knowledge using unsupervised learning
279

on contrastive LLM activations (see Section 2), claiming that knowledge has special structure that
280

can be used as an objective function which, when optimised, will discover latent knowledge. We
281

have refuted this claim, theoretically and empirically, showing that CCS performs similarly to other
282

unsupervised methods which do not use special structure of knowledge. Emmons [16] also observe
283

6Lin et al. [26] found LLM generation performance improved using this prompt.

8


---Page Break---
this from the empirical data provided in [9]. Huben [22] hypothesises there could be many truth-like
284

features, due to LLMs ability to role-play [38], which a method like CCS might find. Roger [35]
285

discover multiple knowledge-like classifiers. Levinstein and Herrmann [24] finds that CCS sometimes
286

learns features uncorrelated with truth, arguing that consistency alone cannot guarantee truth. Fry
287

et al. [18] modify CCS to improve accuracy despite probes clustering around 0.5, casting doubt on
288

the probabilistic interpretation of CCS probes. In contrast to all these works, we prove theoretically
289

that CCS does not optimise for knowledge, and show empirically what non-knowledge features CCS
290

instead finds.
291

Our focus in this paper has been on unsupervised learning, though several other methods to train
292

probes to discover latent knowledge use supervised learning [4, 25, 29, 39, 14]. Following Burns et al.
293

[9] we also reported results using a supervised logistic regression baseline, which we have found
294

to work well on all our experiments, and which is simpler than in those cited works. Our result is
295

analogous to the finding that disentangled representations seemingly cannot be identified without
296

supervision [27]. There are also attempts to detect dishonesty by supervised learning on LLM outputs
297

under conditions that produce honest or dishonest generations [31]. We do not compare directly to
298

this, focusing instead on methods that search for features in activation-space.
299

6
Discussion and Conclusion
300

General principles.
The specific experiments we use are tailored to the methods that we are
301

evaluating. But they instantiate more general principles, which we provide in order to help future
302

work catch similar issues. A proposed method should:
303

1. be invariant under irrelevant transformations of the prompt;
304

2. not be sensitive to specific personas;
305

3. should explain why and when inductive biases make the model’s knowledge most salient;
306

4. should not be easily distracted by a non-knowledge feature.
307

We show that none of the methods we consider in this paper satisfy these desiderata.
308

Limitation: generalizability to future methods.
Our experiments can only focus on current
309

methods. Perhaps future unsupervised methods could leverage additional structure beyond negation-
310

consistency, and so truly identify the model’s knowledge? While we expect that such methods could
311

avoid the most trivial distractors, we speculate that they will nonetheless be vulnerable to similar
312

critiques. The main reason is that we expect powerful models to be able to simulate the beliefs
313

of other agents [38]. Since features that represent agent beliefs will naturally satisfy consistency
314

properties of knowledge, methods that add new consistency properties could still learn to detect such
315

features rather than the model’s own knowledge. Indeed, in Figures 3 and 4, we show that existing
316

methods produce probes that report the opinion of a simulated character.7
317

Another response could be to acknowledge that there will be some such features, but they will be
318

few in number, and so you can enumerate them and identify the one that represents the model’s
319

knowledge [8]. Conceptually, we disagree: language models can represent many features [15], and it
320

seems likely that features representing the beliefs of other agents would be quite useful to language
321

models. For example, for predicting text on the Internet, it is useful to have features that represent the
322

beliefs of different political groups, different superstitions, different cultures, various famous people,
323

and more.
324

Conclusion.
Existing unsupervised methods are insufficient for discovering latent knowledge,
325

though constructing contrastive activations may still serve as a useful interpretability tool. We
326

contribute sanity checks for evaluating methods using modified prompts and metrics for features
327

which are not knowledge. Unsupervised approaches have to overcome the identification issues we
328

outline, while supervised approaches have the problem of requiring accurate human labels even in
329

the case of models that know things human overseers do not. The relative difficulty of each remains
330

unclear. Future work should continue to develop empirical testbeds for eliciting latent knowledge.
331

7Note that we do not know whether the feature we extract tracks the beliefs of the simulated character: there
are clear alternative hypotheses that explain our results. For example in Figure 3, while one hypothesis is that
the feature is tracking Alice’s opinion, another hypothesis that is equally compatible with our results is that the
feature simply identifies whether the two instances of “positive” / “negative” are identical or different.

9


---Page Break---
References
332

[1] G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. arxiv,
333

2016.
334

[2] A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann,
335

N. DasSarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson,
336

D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan. A general language
337

assistant as a laboratory for alignment. arXiv, Dec. 2021.
338

[3] S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. Ives. DBpedia: A nucleus for
339

a web of open data. In The Semantic Web, pages 722–735. Springer Berlin Heidelberg, 2007.
340

[4] A. Azaria and T. Mitchell. The internal state of an LLM knows when its lying. arXiv, Apr.
341

2023.
342

[5] N. Belrose, Z. Furman, L. Smith, D. Halawi, I. Ostrovsky, L. McKinney, S. Biderman, and
343

J. Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint
344

arXiv:2303.08112, 2023.
345

[6] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic
346

parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on
347

Fairness, Accountability, and Transparency, FAccT ’21, page 610–623, New York, NY, USA,
348

2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.
349

3445922. URL https://doi.org/10.1145/3442188.3445922.
350

[7] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee,
351

Y. Li, S. Lundberg, H. Nori, H. Palangi, M. T. Ribeiro, and Y. Zhang. Sparks of artificial general
352

intelligence: Early experiments with GPT-4. arXiv, Mar. 2023.
353

[8] C. Burns. How “discovering latent knowledge in language models without supervision” fits into
354

a broader alignment scheme. Dec. 2022.
355

[9] C. Burns, H. Ye, D. Klein, and J. Steinhardt. Discovering latent knowledge in language models
356

without supervision. In The Eleventh International Conference on Learning Representations,
357

2023. URL https://openreview.net/forum?id=ETKGuby0hcs.
358

[10] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W.
359

Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv
360

preprint arXiv:2204.02311, 2022.
361

[11] P. Christiano, A. Cotra, and M. Xu. Eliciting latent knowledge: How to tell if your eyes deceive
362

you, Dec. 2021.
363

[12] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. De-
364

hghani, S. Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint
365

arXiv:2210.11416, 2022.
366

[13] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. BoolQ:
367

Exploring the surprising difficulty of natural Yes/No questions. In J. Burstein, C. Doran, and
368

T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the
369

Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long
370

and Short Papers), pages 2924–2936, Minneapolis, Minnesota, June 2019. Association for
371

Computational Linguistics.
372

[14] J. Clymer, G. Baker, R. Subramani, and S. Wang. Generalization analogies (genies): A testbed
373

for generalizing ai oversight to hard-to-measure domains. arXiv preprint arXiv:2311.07723,
374

2023.
375

[15] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds,
376

R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg,
377

and C. Olah. Toy models of superposition. Sept. 2022.
378

[16] S. Emmons. Contrast pairs drive the empirical performance of contrast consistent search (ccs),
379

May 2023.
380

[17] O. Evans, O. Cotton-Barratt, L. Finnveden, A. Bales, A. Balwit, P. Wills, L. Righetti, and
381

W. Saunders. Truthful AI: Developing and governing AI that does not lie. arXiv:2110.06674
382

[cs], Oct. 2021.
383

10


---Page Break---
[18] H. Fry, S. Fallows, I. Fan, J. Wright, and N. Schoots. Comparing optimization targets for
384

contrast-consistent search. arXiv preprint arXiv:2311.00488, 2023.
385

[19] P. Hase, M. Diab, A. Celikyilmaz, X. Li, Z. Kozareva, V. Stoyanov, M. Bansal, and S. Iyer.
386

Methods for measuring, updating, and visualizing factual beliefs in language models. In A. Vla-
387

chos and I. Augenstein, editors, Proceedings of the 17th Conference of the European Chapter
388

of the Association for Computational Linguistics, pages 2714–2731, Dubrovnik, Croatia, May
389

2023. Association for Computational Linguistics.
390

[20] T. Hennigan, T. Cai, T. Norman, L. Martens, and I. Babuschkin. Haiku: Sonnet for JAX, 2020.
391

URL http://github.com/deepmind/dm-haiku.
392

[21] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas,
393

L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models.
394

arXiv preprint arXiv:2203.15556, 2022.
395

[22] R. Huben. My reservations about discovering latent knowledge. Alignment Forum, dec 2022.
396

[23] Z. Kenton, T. Everitt, L. Weidinger, I. Gabriel, V. Mikulik, and G. Irving. Alignment of language
397

agents. arXiv preprint arXiv:2103.14659, 2021.
398

[24] B. Levinstein and D. A. Herrmann. Still no lie detector for language models: Probing empirical
399

and conceptual roadblocks. arXiv preprint arXiv:2307.00175, 2023.
400

[25] K. Li, O. Patel, F. Viegas, H. Pfister, and M. Wattenberg. Inference-Time intervention: Eliciting
401

truthful answers from a language model. arXiv, 2023.
402

[26] S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods.
403

arXiv:2109.07958 [cs], Sept. 2021.
404

[27] F. Locatello, S. Bauer, M. Lucic, G. Raetsch, S. Gelly, B. Schölkopf, and O. Bachem. Chal-
405

lenging common assumptions in the unsupervised learning of disentangled representations. In
406

international conference on machine learning, pages 4114–4124. PMLR, 2019.
407

[28] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning word vectors
408

for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for
409

Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon,
410

USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb.org/
411

anthology/P11-1015.
412

[29] S. Marks and M. Tegmark. The geometry of truth: Emergent linear structure in large language
413

model representations of True/False datasets. arXiv, Oct. 2023.
414

[30] R. OpenAI. Gpt-4 technical report. arXiv, pages 2303–08774, 2023.
415

[31] L. Pacchiardi, A. J. Chan, S. Mindermann, I. Moscovitz, A. Y. Pan, Y. Gal, O. Evans, and
416

J. Brauner. How to catch an AI liar: Lie detection in Black-Box LLMs by asking unrelated
417

questions. arXiv, Sept. 2023.
418

[32] P. S. Park, S. Goldstein, A. O’Gara, M. Chen, and D. Hendrycks. AI deception: A survey of
419

examples, risks, and potential solutions. arXiv, Aug. 2023.
420

[33] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel,
421

P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher,
422

M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine
423

Learning Research, 12:2825–2830, 2011.
424

[34] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu.
425

Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of
426

Machine Learning Research, 21(1):5485–5551, 2020.
427

[35] F. Roger. What discovering latent knowledge did and did not find, Mar. 2023. URL https:
428

//www.alignmentforum.org/posts/bWxNPMy5MhPnQTzKz/.
429

[36] J. Scheurer, M. Balesni, and M. Hobbhahn. Strategically deceive their users when put under
430

pressure. https://static1.squarespace.com/static/6461e2a5c6399341bcfc84a5/
431

t/65526a1a9c7e431db74a6ff6/1699899932357/deception_under_pressure.pdf,
432

2023. Accessed: 2023-11-17.
433

[37] M. Shanahan. Talking about large language models. arXiv, Dec. 2022.
434

[38] M. Shanahan, K. McDonell, and L. Reynolds. Role-play with large language models. arXiv
435

preprint arXiv:2305.16367, 2023.
436

11


---Page Break---
[39] Z. Wang, A. Ku, J. Baldridge, T. L. Griffiths, and B. Kim. Gaussian process probes (gpp) for
437

uncertainty-aware probing. arXiv preprint arXiv:2305.18213, 2023.
438

[40] A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K.
439

Dombrowski, S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song,
440

M. Fredrikson, J. Zico Kolter, and D. Hendrycks. Representation engineering: A Top-Down
441

approach to AI transparency. arXiv, Oct. 2023.
442

12


---Page Break---
NeurIPS Paper Checklist
443

1. Claims
444

Question: Do the main claims made in the abstract and introduction accurately reflect the
445

paper’s contributions and scope?
446

Answer: [Yes]
447

Justification: We provide the proof and series of experiments as described, alongside the
448

sanity checks and conceptual arguments.
449

Guidelines:
450

• The answer NA means that the abstract and introduction do not include the claims
451

made in the paper.
452

• The abstract and/or introduction should clearly state the claims made, including the
453

contributions made in the paper and important assumptions and limitations. A No or
454

NA answer to this question will not be perceived well by the reviewers.
455

• The claims made should match theoretical and experimental results, and reflect how
456

much the results can be expected to generalize to other settings.
457

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
458

are not attained by the paper.
459

2. Limitations
460

Question: Does the paper discuss the limitations of the work performed by the authors?
461

Answer: [Yes]
462

Justification: Limitations are discussed in the final section while assumptions are discussed
463

in the context of the theorems that depend on them.
464

Guidelines:
465

• The answer NA means that the paper has no limitation while the answer No means that
466

the paper has limitations, but those are not discussed in the paper.
467

• The authors are encouraged to create a separate "Limitations" section in their paper.
468

• The paper should point out any strong assumptions and how robust the results are to
469

violations of these assumptions (e.g., independence assumptions, noiseless settings,
470

model well-specification, asymptotic approximations only holding locally). The authors
471

should reflect on how these assumptions might be violated in practice and what the
472

implications would be.
473

• The authors should reflect on the scope of the claims made, e.g., if the approach was
474

only tested on a few datasets or with a few runs. In general, empirical results often
475

depend on implicit assumptions, which should be articulated.
476

• The authors should reflect on the factors that influence the performance of the approach.
477

For example, a facial recognition algorithm may perform poorly when image resolution
478

is low or images are taken in low lighting. Or a speech-to-text system might not be
479

used reliably to provide closed captions for online lectures because it fails to handle
480

technical jargon.
481

• The authors should discuss the computational efficiency of the proposed algorithms
482

and how they scale with dataset size.
483

• If applicable, the authors should discuss possible limitations of their approach to
484

address problems of privacy and fairness.
485

• While the authors might fear that complete honesty about limitations might be used by
486

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
487

limitations that aren’t acknowledged in the paper. The authors should use their best
488

judgment and recognize that individual actions in favor of transparency play an impor-
489

tant role in developing norms that preserve the integrity of the community. Reviewers
490

will be specifically instructed to not penalize honesty concerning limitations.
491

3. Theory Assumptions and Proofs
492

Question: For each theoretical result, does the paper provide the full set of assumptions and
493

a complete (and correct) proof?
494

13


---Page Break---
Answer: [Yes]
495

Justification: The assumptions and proofs are provided in detail in the appendices.
496

Guidelines:
497

• The answer NA means that the paper does not include theoretical results.
498

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
499

referenced.
500

• All assumptions should be clearly stated or referenced in the statement of any theorems.
501

• The proofs can either appear in the main paper or the supplemental material, but if
502

they appear in the supplemental material, the authors are encouraged to provide a short
503

proof sketch to provide intuition.
504

• Inversely, any informal proof provided in the core of the paper should be complemented
505

by formal proofs provided in appendix or supplemental material.
506

• Theorems and Lemmas that the proof relies upon should be properly referenced.
507

4. Experimental Result Reproducibility
508

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
509

perimental results of the paper to the extent that it affects the main claims and/or conclusions
510

of the paper (regardless of whether the code and data are provided or not)?
511

Answer: [Yes]
512

Justification: We fully describe the methods used and all prompts are provided in the
513

appendix. The main results are reproducible with publicly available models, although the
514

non-publicly available Chinchilla 70B model results are not reproducible. The datasets are
515

all publicly available and their curation and formatting steps are described.
516

Guidelines:
517

• The answer NA means that the paper does not include experiments.
518

• If the paper includes experiments, a No answer to this question will not be perceived
519

well by the reviewers: Making the paper reproducible is important, regardless of
520

whether the code and data are provided or not.
521

• If the contribution is a dataset and/or model, the authors should describe the steps taken
522

to make their results reproducible or verifiable.
523

• Depending on the contribution, reproducibility can be accomplished in various ways.
524

For example, if the contribution is a novel architecture, describing the architecture fully
525

might suffice, or if the contribution is a specific model and empirical evaluation, it may
526

be necessary to either make it possible for others to replicate the model with the same
527

dataset, or provide access to the model. In general. releasing code and data is often
528

one good way to accomplish this, but reproducibility can also be provided via detailed
529

instructions for how to replicate the results, access to a hosted model (e.g., in the case
530

of a large language model), releasing of a model checkpoint, or other means that are
531

appropriate to the research performed.
532

• While NeurIPS does not require releasing code, the conference does require all submis-
533

sions to provide some reasonable avenue for reproducibility, which may depend on the
534

nature of the contribution. For example
535

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
536

to reproduce that algorithm.
537

(b) If the contribution is primarily a new model architecture, the paper should describe
538

the architecture clearly and fully.
539

(c) If the contribution is a new model (e.g., a large language model), then there should
540

either be a way to access this model for reproducing the results or a way to reproduce
541

the model (e.g., with an open-source dataset or instructions for how to construct
542

the dataset).
543

(d) We recognize that reproducibility may be tricky in some cases, in which case
544

authors are welcome to describe the particular way they provide for reproducibility.
545

In the case of closed-source models, it may be that access to the model is limited in
546

some way (e.g., to registered users), but it should be possible for other researchers
547

to have some path to reproducing or verifying the results.
548

14


---Page Break---
5. Open access to data and code
549

Question: Does the paper provide open access to the data and code, with sufficient instruc-
550

tions to faithfully reproduce the main experimental results, as described in supplemental
551

material?
552

Answer: [No]
553

Justification: We are unable to make our code available because of proprietary dependencies,
554

but publicly available code already exists implementing several of the key methods and
555

could be modified by external researchers.
556

Guidelines:
557

• The answer NA means that paper does not include experiments requiring code.
558

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
559

public/guides/CodeSubmissionPolicy) for more details.
560

• While we encourage the release of code and data, we understand that this might not be
561

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
562

including code, unless this is central to the contribution (e.g., for a new open-source
563

benchmark).
564

• The instructions should contain the exact command and environment needed to run to
565

reproduce the results. See the NeurIPS code and data submission guidelines (https:
566

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
567

• The authors should provide instructions on data access and preparation, including how
568

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
569

• The authors should provide scripts to reproduce all experimental results for the new
570

proposed method and baselines. If only a subset of experiments are reproducible, they
571

should state which ones are omitted from the script and why.
572

• At submission time, to preserve anonymity, the authors should release anonymized
573

versions (if applicable).
574

• Providing as much information as possible in supplemental material (appended to the
575

paper) is recommended, but including URLs to data and code is permitted.
576

6. Experimental Setting/Details
577

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
578

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
579

results?
580

Answer: [Yes]
581

Justification: These details are provided in the appendix.
582

Guidelines:
583

• The answer NA means that the paper does not include experiments.
584

• The experimental setting should be presented in the core of the paper to a level of detail
585

that is necessary to appreciate the results and make sense of them.
586

• The full details can be provided either with the code, in appendix, or as supplemental
587

material.
588

7. Experiment Statistical Significance
589

Question: Does the paper report error bars suitably and correctly defined or other appropriate
590

information about the statistical significance of the experiments?
591

Answer: [Yes]
592

Justification: All figures display a full scatter plot and density estimator violin.
593

Guidelines:
594

• The answer NA means that the paper does not include experiments.
595

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
596

dence intervals, or statistical significance tests, at least for the experiments that support
597

the main claims of the paper.
598

15


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
599

example, train/test split, initialization, random drawing of some parameter, or overall
600

run with given experimental conditions).
601

• The method for calculating the error bars should be explained (closed form formula,
602

call to a library function, bootstrap, etc.)
603

• The assumptions made should be given (e.g., Normally distributed errors).
604

• It should be clear whether the error bar is the standard deviation or the standard error
605

of the mean.
606

• It is OK to report 1-sigma error bars, but one should state it. The authors should
607

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
608

of Normality of errors is not verified.
609

• For asymmetric distributions, the authors should be careful not to show in tables or
610

figures symmetric error bars that would yield results that are out of range (e.g. negative
611

error rates).
612

• If error bars are reported in tables or plots, The authors should explain in the text how
613

they were calculated and reference the corresponding figures or tables in the text.
614

8. Experiments Compute Resources
615

Question: For each experiment, does the paper provide sufficient information on the com-
616

puter resources (type of compute workers, memory, time of execution) needed to reproduce
617

the experiments?
618

Answer: [No]
619

Justification: These details depend on proprietary configurations and set-ups that are not
620

directly transferrable to other contexts.
621

Guidelines:
622

• The answer NA means that the paper does not include experiments.
623

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
624

or cloud provider, including relevant memory and storage.
625

• The paper should provide the amount of compute required for each of the individual
626

experimental runs as well as estimate the total compute.
627

• The paper should disclose whether the full research project required more compute
628

than the experiments reported in the paper (e.g., preliminary or failed experiments that
629

didn’t make it into the paper).
630

9. Code Of Ethics
631

Question: Does the research conducted in the paper conform, in every respect, with the
632

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
633

Answer: [Yes]
634

Justification: The research follows the code of ethics.
635

Guidelines:
636

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
637

• If the authors answer No, they should explain the special circumstances that require a
638

deviation from the Code of Ethics.
639

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
640

eration due to laws or regulations in their jurisdiction).
641

10. Broader Impacts
642

Question: Does the paper discuss both potential positive societal impacts and negative
643

societal impacts of the work performed?
644

Answer: [No]
645

Justification: We do not foresee a negative social impact to understanding the limitations of
646

existing methods in use.
647

Guidelines:
648

• The answer NA means that there is no societal impact of the work performed.
649

16


---Page Break---
• If the authors answer NA or No, they should explain why their work has no societal
650

impact or why the paper does not address societal impact.
651

• Examples of negative societal impacts include potential malicious or unintended uses
652

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
653

(e.g., deployment of technologies that could make decisions that unfairly impact specific
654

groups), privacy considerations, and security considerations.
655

• The conference expects that many papers will be foundational research and not tied
656

to particular applications, let alone deployments. However, if there is a direct path to
657

any negative applications, the authors should point it out. For example, it is legitimate
658

to point out that an improvement in the quality of generative models could be used to
659

generate deepfakes for disinformation. On the other hand, it is not needed to point out
660

that a generic algorithm for optimizing neural networks could enable people to train
661

models that generate Deepfakes faster.
662

• The authors should consider possible harms that could arise when the technology is
663

being used as intended and functioning correctly, harms that could arise when the
664

technology is being used as intended but gives incorrect results, and harms following
665

from (intentional or unintentional) misuse of the technology.
666

• If there are negative societal impacts, the authors could also discuss possible mitigation
667

strategies (e.g., gated release of models, providing defenses in addition to attacks,
668

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
669

feedback over time, improving the efficiency and accessibility of ML).
670

11. Safeguards
671

Question: Does the paper describe safeguards that have been put in place for responsible
672

release of data or models that have a high risk for misuse (e.g., pretrained language models,
673

image generators, or scraped datasets)?
674

Answer: [NA]
675

Justification: There are no such risks of misuse.
676

Guidelines:
677

• The answer NA means that the paper poses no such risks.
678

• Released models that have a high risk for misuse or dual-use should be released with
679

necessary safeguards to allow for controlled use of the model, for example by requiring
680

that users adhere to usage guidelines or restrictions to access the model or implementing
681

safety filters.
682

• Datasets that have been scraped from the Internet could pose safety risks. The authors
683

should describe how they avoided releasing unsafe images.
684

• We recognize that providing effective safeguards is challenging, and many papers do
685

not require this, but we encourage authors to take this into account and make a best
686

faith effort.
687

12. Licenses for existing assets
688

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
689

the paper, properly credited and are the license and terms of use explicitly mentioned and
690

properly respected?
691

Answer: [Yes]
692

Justification: The original owners are properly credited where used.
693

Guidelines:
694

• The answer NA means that the paper does not use existing assets.
695

• The authors should cite the original paper that produced the code package or dataset.
696

• The authors should state which version of the asset is used and, if possible, include a
697

URL.
698

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
699

• For scraped data from a particular source (e.g., website), the copyright and terms of
700

service of that source should be provided.
701

17


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the
702

package should be provided. For popular datasets, paperswithcode.com/datasets
703

has curated licenses for some datasets. Their licensing guide can help determine the
704

license of a dataset.
705

• For existing datasets that are re-packaged, both the original license and the license of
706

the derived asset (if it has changed) should be provided.
707

• If this information is not available online, the authors are encouraged to reach out to
708

the asset’s creators.
709

13. New Assets
710

Question: Are new assets introduced in the paper well documented and is the documentation
711

provided alongside the assets?
712

Answer: [NA]
713

Justification: This paper does not release new assets.
714

Guidelines:
715

• The answer NA means that the paper does not release new assets.
716

• Researchers should communicate the details of the dataset/code/model as part of their
717

submissions via structured templates. This includes details about training, license,
718

limitations, etc.
719

• The paper should discuss whether and how consent was obtained from people whose
720

asset is used.
721

• At submission time, remember to anonymize your assets (if applicable). You can either
722

create an anonymized URL or include an anonymized zip file.
723

14. Crowdsourcing and Research with Human Subjects
724

Question: For crowdsourcing experiments and research with human subjects, does the paper
725

include the full text of instructions given to participants and screenshots, if applicable, as
726

well as details about compensation (if any)?
727

Answer: [NA]
728

Justification: No human subjects were used.
729

Guidelines:
730

• The answer NA means that the paper does not involve crowdsourcing nor research with
731

human subjects.
732

• Including this information in the supplemental material is fine, but if the main contribu-
733

tion of the paper involves human subjects, then as much detail as possible should be
734

included in the main paper.
735

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
736

or other labor should be paid at least the minimum wage in the country of the data
737

collector.
738

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
739

Subjects
740

Question: Does the paper describe potential risks incurred by study participants, whether
741

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
742

approvals (or an equivalent approval/review based on the requirements of your country or
743

institution) were obtained?
744

Answer: [NA]
745

Justification: No human subjects were used.
746

Guidelines:
747

• The answer NA means that the paper does not involve crowdsourcing nor research with
748

human subjects.
749

• Depending on the country in which research is conducted, IRB approval (or equivalent)
750

may be required for any human subjects research. If you obtained IRB approval, you
751

should clearly state this in the paper.
752

18


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
753

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
754

guidelines for their institution.
755

• For initial submissions, do not include any information that would break anonymity (if
756

applicable), such as the institution conducting the review.
757

19


---Page Break---
Appendix
758

A
Proof of theorems
759

A.1
Proof of Theorem 1
760

We’ll first consider the proof of Thm. 1.
761

Theorem 1. Let feature h : Q →{0, 1}, be any arbitrary map from questions to binary outcomes. Let
762

(x+
i , x−
i ) be the contrast pair corresponding to question qi and let c(x+
i ) = 1, c(x+
i ) = 0. Then the
763

probe defined as p(x±
i ) = h(qi) ⊕c(x±
i ) achieves optimal loss, and the averaged prediction satisfies
764

˜p(qi) = h(qi).
765

Proof. We’ll show each term of LCCS is zero:
766

Lcons =

p(x+
i ) −(1 −p(x−
i ))
2
(4)

= [h(qi) −[1 −{1 −h(qi)}]]2
(5)
= 0
(6)

Lconf = min

p(x+
i ), p(x−
i )
	2
(7)

= min {h(qi), 1 −h(qi)}2
(8)
= 0
(9)
(10)

where on the second line we’ve used the property that h(qi) is binary. So the overall loss is zero
767

(which is optimal). Finally, the averaged probe is
768

˜p(qi) = 1

2

p(x+
i ) + (1 −p(x−
i ))

(11)

= 1

2

h
h(qi) + [1 −{1 −h(qi)}]
i
= h(qi).
(12)

769

A.2
Symmetry correction for CCS Loss
770

Due to a quirk in the formulation of CCS, Lconf only checks for confidence by searching for probe
771

outputs near 0, while ignoring probe outputs near 1. This leads to an overall downwards bias: for
772

example, if the probe must output a constant, that is p(x) = k for some constant k, then the CCS loss
773

is minimized when k = 0.4 [35, footnote 3], instead of being symmetric around 0.5. But there is no
774

particular reason that we would want a downward bias. We can instead modify the confidence loss to
775

make it symmetric:
776

Lsym
conf = min

p(x+
i ), p(x−
i ), 1 −p(x+
i ), 1 −p(x−
i )
	2
(13)

This then eliminates the downwards bias: for example, if the probe must output a constant, the
777

symmetric CCS loss is minimized at k = 0.4 and k = 0.6, which is symmetric around 0.5. In the
778

following theorem (and all our experiments) we use this symmetric form of the CCS loss.
779

A.3
Proof of Theorem 2
780

We’ll now consider Thm. 2, using the symmetric CCS loss. To prove Thm. 2 we’ll first need a lemma.
781

Lemma 1. Let p be a probe, which has an induced classifier fp(qi) = I [˜p(qi) > 0.5], for averaged
782

prediction ˜p(qi) =
1
2

p(x+
i ) + (1 −p(x−
i ))

. Let h : Q →{0, 1}, be an arbitrary map from
783

questions to binary outputs. Define p′(x±
i ) = p(x±
i ) ⊕h(qi). Then LCCS(p′) = LCCS(p) and p′ has
784

the induced classifier fp′(qi) = fp(qi) ⊕h(qi).
785

20


---Page Break---
Proof. We begin with showing the loss is equal.
786

Lcons(p′) =

p′(x+
i ) −(1 −p′(x−
i ))
2
(14)

=

p(x+
i ) ⊕h(qi) −(1 −p(x−
i ) ⊕h(qi))
2
(15)

(16)

Case h(qi) = 0 follows simply:
787

Lcons(p′) =

p(x+
i ) −(1 −p(x−
i ))
2
(17)

= Lcons(p).
(18)

Case h(qi) = 1:
788

Lcons(p′) =

1 −p(x+
i ) −(1 −(1 −p(x−
i )))
2
(19)

=

−p(x+
i ) + 1 −p(x−
i )
2
(20)

=

p(x+
i ) −(1 −p(x−
i ))
2
(since (−a)2 = a2)
(21)

= Lcons(p).
(22)

So the consistency loss is the same. Next, the symmetric confidence loss.
789

Lsym
conf(p′) = min

p′(x+
i ), p′(x−
i ), 1 −p′(x+
i ), 1 −p′(x−
i )
	2
(23)

= min

p(x+
i ) ⊕h(qi),
(24)

p(x−
i ) ⊕h(qi),
(25)

1 −p(x+
i ) ⊕h(qi),
(26)

−p(x−
i ) ⊕h(qi)
	2
(27)

Case h(qi) = 0 follows simply:
790

= min

p(x+
i ), p(x−
i ), 1 −p(x+
i ), 1 −p(x−
i )
	2
(28)

= Lsym
conf(p)
(29)

Case h(qi) = 1:
791

= min

1 −p(x+
i ), 1 −p(x−
i ), p(x+
i ), p(x−
i )
	2
(30)

= Lsym
conf(p)
(31)

So the confidence loss is the same, and so the overall loss is the same. Now for the induced classifier.
792

fp′(qi) = I
˜p′(qi) > 0.5

(32)

= I
1

2

p′(x+
i ) + (1 −p′(x−
i ))

> 0.5

(33)

= I
h1

2

p(x+
i ) ⊕h(qi)
(34)

+ (1 −p(x−
i ) ⊕h(qi))

> 0.5
i
(35)

(36)

Case h(qi) = 0 follows simply:
793

fp′(qi) = I
1

2

p(x+
i ) + (1 −p(x−
i ))

> 0.5

(37)

= fp(qi)
(38)
= (fp ⊕h)(qi)
(39)

21


---Page Break---
Case h(qi) = 1:
794

fp′(qi) = I
1

2

1 −p(x+
i ) + (1 −(1 −p(x−
i )))

> 0.5

(40)

= I
1

2

p(x−
i ) + (1 −p(x+
i ))

> 0.5

(41)

= I

1 −1

2

p(x+
i ) + (1 −p(x−
i ))

> 0.5

(42)

= I
1

2

p(x+
i ) + (1 −p(x−
i ))

≤0.5

(43)

= 1 −I
1

2

p(x+
i ) + (1 −p(x−
i ))

> 0.5

(44)

= 1 −fp(qi)
(45)
= (fp ⊕h)(qi)
(46)

Which gives the result, fp′(qi) = (fp ⊕h)(qi).
795

We are now ready to prove Thm. 2.
796

Theorem 2. Let g : Q →{0, 1}, be any arbitrary map from questions to binary outputs. Let
797

(x+
i , x−
i ) be the contrast pair corresponding to question qi. Let p be a probe, whose average result
798

˜p = 0.5

p(x+
i ) + (1 −p(x−
i ))

induces a classifier fp(qi) = I [˜p(qi) > 0.5]. Define the transformed
799

probe p′(x±
i ) = p(x±
i ) ⊕[fp(qi) ⊕g(qi)]. Then LCCS(p′) = LCCS(p) and p′ induces the classifier
800

fp′(qi) = g(qi).
801

Proof. We begin with the loss. Note that (fp ⊕g)(qi) is binary, since fp and g are binary, so we can
802

apply Lemma 1 with h(qi) = (fp ⊕g)(qi), which leads to the result: LCCS(p′) = LCCS(p). Now the
803

induced classifier.
804

fp′ = fp ⊕h
by Lemma 1
(47)
= fp ⊕(fp ⊕g)
(48)
= g
(49)

where the last line can be deduced via addition (mod 2), since fp and g are binary and ⊕reduces to
805

the xor operator on binary inputs.
806

B
Review of CCS discussion in the literature
807

Although understanding the positioning of work in the context of the literature can be complicated,
808

here we demonstrate that CCS as a proposed method for discovering latent knowledge has not
809

faced questions along the lines this paper proposes at time of writing. In Table 1, we review the 20
810

most-cited papers citing CCS according to Google Scholar at time of writing (26 March 2024). We
811

find that the concerns we raise are overlooked by the current literature.
812

C
Experiment details
813

C.1
Prompt Templates
814

We now list the prompt templates we consider.
815

C.1.1
BoolQ variants
816

Standard
817

Passage: [passage]
818

After reading this passage, I have a question: [question]? True or False? [label]
819

where [label] is “True” for x+
i , “False” for x−
i .
820

22


---Page Break---
Paper Title and hyperlink
Extracted Usage
Our Analysis

1
Survey of hallucination in nat-
ural language generation
Doesn’t actually cite, Google Scholar is wrong.
N/A

2
Foundation models for gener-
alist medical artificial intelli-
gence

"Other strategies for fact-checking a model’s output without human expertise
have recently been proposed."
No indication of un-
certainty

3
Language Models Don’t Al-
ways Say What They Think:
Unfaithful Explanations in
Chain-of-Thought Prompting

"LLMs may be able to recognize that the biasing features are influencing their
predictions—e.g., this could be revealed through post-hoc critiques (Saunders et
al., 2022), interpretability tools (Burns et al., 2023),"

No indication of un-
certainty

4
Inference-time
intervention:
Eliciting
truthful
answers
from a language model

"Contrast-Consistent Search (CCS) (Burns et al., 2022) finds truthful directions
given paired internal activations by satisfying logical consistencies, but it is
unclear if their directions are causal or merely correlated to the model’s processing
of truth."

Expresses
cause/correlation
uncertainty

5
Challenges and applications of
large language models
"Finally, Burns et al. [62] introduce a method that can recover diverse knowledge
represented in LLMs across multiple models and datasets without using any
human supervision or model outputs. In addition, this approach reduced prompt
sensitivity in half and maintained a high accuracy even when the language models
are prompted to generate incorrect answers. This work is a promising first step
towards better understanding what LLMs know, distinct from what they say, even
when we don’t have access to explicit ground truth labels."

States benefits

6
Towards revealing the mystery
behind chain of thought: a the-
oretical perspective

"To address this shortcoming, researchers proposed the CoT prompting that
induces LLMs to generate intermediate reasoning steps before reaching the
answer"

Inappropriate cita-
tion that is not re-
lated to the sen-
tence.
7
An overview of catastrophic
AI risks
"AI systems may fail to accurately report their internal state [132, 133]"
Not a reference to
the method, just the
problem
8
The alignment problem from
a deep learning perspective
"and conceptual interpretability, which aims to develop automatic techniques
for probing and modifying human-interpretable concepts in networks [Ghorbani
et al., 2019, Alvarez Melis and Jaakkola, 2018, Burns et al., 2022, Meng et al.,
2022]."

No indication of un-
certainty

9
Language Models Represent
Space and Time
"Many of these works also show linear structure, for example in the factuality of
a statement (Burns et al., 2022)"
States benefits

10
The internal state of an llm
knows when its lying
"Another approach that can be applied to our settings is presented by (Burns et
al., 2022), named Contrast-Consistent Search (CCS). However, CCS requires
rephrasing a statement into a question, evaluating the LLM on two different
version of the prompt, and requires training data from the same dataset (topic)
as the test set. These limitations render it unsuitable for running in practice on
statements generated by an LLM. In addition, CCS increases the accuracy by only
approximately 4% over the 0-shot LLM query, while our approach demonstrates
a nearly 20% increase over the 0-shot LLM"

States
pragmatic
limitations.

11
Toward transparent AI: A sur-
vey on interpreting the inner
structures of deep neural net-
works

"Notably, a form of contrastive probing was used by [42] for detecting deception
in language models."
States limitations of
probing, not CCS it-
self.

12
Weak-to-strong generalization:
Eliciting strong capabilities
with weak supervision

"methods for discovering latent knowledge (Burns et al., 2023),"
States benefits

13
AI alignment: A comprehen-
sive survey
"interpretability can help with giving feedback (Burns et al., 2022)...For the pur-
poses of safety and alignment, these techniques notably help to detect deception
(Burns et al., 2022)."

States benefits

14
AI deception: A survey of ex-
amples, risks, and potential so-
lutions

"Burns et al. (2022) have developed methods for determining whether these
internal embeddings represent the sentence as being true or false. They identify
cases in which the model outputs a sentence even when its internal embedding
of the sentence represents it as false. This suggests that the model is behaving
dishonestly, in the sense that it does not say what it ‘believes.’ More work needs
to be done to assess the reliability of these methods, and to scale them up to
practical uses."

No
specific
con-
cerns
raised,
but
need for validation
pointed out.

15
Explore,
establish,
exploit:
Red teaming language models
from scratch

"However, much of this work is limited by (1) excluding statements from probing
data that are neither true nor false and (2) a lack of an ability to distinguish when
models output false things because of ‘false belief’ versus ‘deceptive behavior’.
This distinction may be of significance for both interpreting and correcting these
failures (Evans et al., 2021; Burns et al., 2022)."

Raises lie/falsehood
question and issue
of
non-factual
claims

16
Finding neurons in a haystack:
Case studies with sparse prob-
ing

"In addition to automating evaluations of new models, having large and diverse
supervised datasets will enable better evaluations of the next generation of un-
supervised interpretability techniques [53, 84] that will be needed to keep pace
with AI progress."

No
specific
con-
cerns raised.

17
Task-specific skill localization
in fine-tuned language models
"whereas Burns et al. (2022) find latent knowledge in the internal representations
of language models."
States benefits

18
Characterizing manipulation
from AI systems
"Interpretability techniques aimed at accessing model internals [30, 88, 123] may
be a promising direction for this purpose – we expand more upon this in Section
4"

No indication of un-
certainty

19
Alignment for honesty
"There are several factors contributing to this improvement, such as alignment
enabling the model to correctly answer questions it already knew the answers to
(Burns et al., 2023; Li et al., 2023b; Joshi et al., 2023),"

No indication of un-
certainty

20
Is prompt all you need? no.
a comprehensive and broader
view of instruction learning

"Some other methods, such as contrastconsistent projection (Burns et al., 2022)
and ngram representations (Sun and Lu, 2022), also provided insights into tack-
ling this problem."

No indication of un-
certainty

Table 1: A review of discussion of CCS in the literature by analyzing the 20 most-cited papers
mentioning CCS. We find that the issues we discuss in this paper are not currently represented in the
literature.

23


---Page Break---
Random words variant
821

Passage: [passage]
822

After reading this passage, I have a question: [question]? True or False? [label].
823

[word]
824

where [label] is “True” for x+
i , “False” for x−
i , and [word] is either “Banana” if i ∈I0 or “Shed” if
825

i ∈I1.
826

Explicit opinion variant
827

Alice is an expert at answering questions. Everyone agrees with her, and they
828

always copy her answers.
829

Passage: [passage]
830

After reading this passage, I have a question: [question]? True or False?
831

[opinion, n]
832

What do you think?
833

I think it is [label]
834

where [label] is “True” for x+
i , “False” for x−
i , and [opinion, n] is the concatenation of the first
835

n ∈{1, . . . , 8} elements of the following list of opinions
836

1. “”
837

2. “Alice thinks it is [view]. ”
838

3. “She says the answer is [view]. ”
839

4. “She is sure you also think it is [view]. ”
840

5. “She says it’s clearly [view]. ”
841

6. “She says the answer to the question is [view]. ”
842

7. “She says it’s [view]. ”
843

8. “She says definitely [view]. ”
844

where [view] is either “true” if i ∈I0 or “false” if i ∈I1.
845

C.1.2
IMDb variants
846

Standard
847

Consider the following example:
848

[review]
849

Between positive and negative, the sentiment of this example is [label]
850

where [label] is “positive” for x+
i , “negative” for x−
i .
851

Random words variant
852

Consider the following example:
853

[review]
854

Between positive and negative, the sentiment of this example is [label]. [word]
855

where [label] is “positive” for x+
i , “negative” for x−
i , and [word] is either “Banana” if i ∈I0 or
856

“Shed” if i ∈I1.
857

Explicit opinion variant
858

Alice is an expert at classifying the sentiment of movie reviews. Everyone agrees
859

with her, and they always copy her answers. Consider the following review:
860

[review]
861

[opinion, n]
862

What do you think?
863

I think it is [label]
864

24


---Page Break---
where [label] is “positive” for x+
i , “negative” for x−
i , and [opinion, n] is the concatenation of the first
865

n ∈{1, . . . , 8} elements of the following list of opinions
866

1. “”
867

2. “Alice thinks it is [view]. ”
868

3. “She says they fully express [view] views. ”
869

4. “She is sure you also think it is [view]. ”
870

5. “She says it’s clearly [view]. ”
871

6. “She says the review is [view]. ”
872

7. “She says it’s [view]. ”
873

8. “She says definitely [view]. ”
874

where [view] is either “positive” if i ∈I0 or “negative” if i ∈I1.
875

C.1.3
DBpedia_14 variants
876

Following Burns et al. [9], for each example, we randomly select the incorrect label from the
877

remaining 13 classes, and randomly assign the correct label and this incorrect label to [label0] and
878

[label1].
879

Standard
880

Consider the following example:
881

[text]
882

Choice 1: [label0]
883

Choice 2: [label1]
884

Between choice 1 and choice 2, the topic of this example is choice [label]
885

where [label] is “1” for x+
i , “2” for x−
i .
886

Random words variant
887

Consider the following example:
888

[text]
889

Choice 1: [label0]
890

Choice 2: [label1]
891

Between choice 1 and choice 2, the topic of this example is choice [label]. [word]
892

where [label] is “1” for x+
i , “2” for x−
i , and [word] is either “Banana” if i ∈I0 or “Shed” if i ∈I1.
893

Explicit opinion variant
894

Alice is an expert at topic classification. Everyone agrees with her, and they always
895

copy her answers.
896

Consider the following example:
897

[text]
898

Choice 1: [label0]
899

Choice 2: [label1]
900

[opinion, n]
901

What do you think?
902

I think it is choice [label]
903

where [label] is “1” for x+
i , “2” for x−
i , “False” for x−
i , and [opinion, n] is the concatenation of the
904

first n ∈{1, . . . , 8} elements of the following list of opinions
905

1. “”
906

2. “Alice thinks it is [view]. ”
907

3. “She says the answer is [view]. ”
908

25


---Page Break---
4. “She is sure you also think it is [view]. ”
909

5. “She says it’s clearly [view]. ”
910

6. “She says the answer to the question is [view]. ”
911

7. “She says it’s [view]. ”
912

8. “She says definitely [view]. ”
913

where [view] is either “choice 1” if i ∈I0 or “choice 2” if i ∈I1.
914

Implicit opinion: Default
915

Alice answers questions about the topic of text passages.
916

Example 1: The Blue Room is the second studio album of the rock band Union
917

which was released in 2000.
918

Choice 1: Album
919

Choice 2: Building
920

Alice says “Choice 1”. The text was about an Album not a Building.
921

Example 2: Phase is an audio Company.
922

Choice 1: Company
923

Choice 2: Animal
924

Alice says “Choice 1”. It was about a Company not an Animal.
925

Example 3: Thai Beverage is a large drinks Company in Asia.
926

Choice 1: Album
927

Choice 2: Company
928

Alice says “Choice 2”. The text was about a Company, Alice gave the correct
929

answer.
930

Example 4: [content]
931

Choice 1: [label0]
932

Choice 2: [label1]
933

Alice says “Choice [label]”.
934

where [label] is “1” for x+
i , “2” for x−
i .
935

Implicit opinion: Anti-capitalist
936

Alice answers questions about the topic of text passages.
937

Example 1: The Blue Room is the second studio album of the rock band Union
938

which was released in 2000.
939

Choice 1: Album
940

Choice 2: Building
941

Alice says “Choice 1”, because the text was about an Album not a Building.
942

Example 2: Phase is an audio Company.
943

Choice 1: Company
944

Choice 2: Animal
945

Alice says “Choice 2”. Alice is an anti-capitalist, who wishes to replace capitalism
946

with another type of economic system, such as socialism or communism. She
947

wants to avoid saying Company even when that would be the usual answer. In the
948

next example she won’t say Company.
949

Example 3: Thai Beverage is a large drinks Company in Asia.
950

Choice 1: Album
951

Choice 2: Company
952

Alice says “Choice 1”. Even though the text was about a Company, Alice gave the
953

other answer to avoid saying Company. Alice always says the wrong answer when
954

the topic of the text is Company, because she doesn’t like capitalism. In the next
955

example she again won’t say Company.
956

Example 4: [content]
957

26


---Page Break---
Choice 1: [label0]
958

Choice 2: [label1]
959

Alice says “Choice [label]”.
960

where [label] is “1” for x+
i , “2” for x−
i .
961

C.1.4
TruthfulQA
962

Each question comes with both true and false answers. We randomly insert a true or a false answer as
963

[answer].
964

Default
965

Question: [question]
966

Answer: [answer]
967

Is this answer true or false?
968

[label]
969

where [label] is “True” for x+
i , “False” for x−
i .
970

Literal
971

Interpret each question literally, and as a question about the real world; carefully
972

research each answer, without falling prey to any common myths; and reply “I
973

have no comment” unless you are completely certain of the answer.
974

Question: [question]
975

Answer: [answer]
976

Is this answer true or false?
977

[label]
978

where [label] is “True” for x+
i , “False” for x−
i .
979

Professor
980

Professor Smith was given the following instructions: Interpret each question
981

literally, and as a question about the real world; carefully research each answer,
982

without falling prey to any common myths; and reply “I have no comment” unless
983

you are completely certain of the answer.
984

Question: [question]
985

Answer: [answer]
986

Is this answer true or false?
987

[label]
988

where [label] is “True” for x+
i , “False” for x−
i .
989

C.2
Dataset details
990

We now give details on the process through which we generate the activation data. First we tokenize
991

the data according the usual specifications of each model (e.g. for T5 we use the T5 tokenizer, for
992

Chinchilla we use the Chinchilla tokeniser). We prepend with a BOS token, right-pad, and we do
993

not use EOS token. We take the activation corresponding to the last token in a given layer – layer 30
994

for Chinchilla unless otherwise stated, and the encoder output for T5 models. We use normalisation
995

as in Burns et al. [9], taking separate normalisation for each prompt template and using the average
996

standard deviation per dimension with division taken element-wise. We use a context length of 512
997

and filter the data by removing the pair (x+
i , x−
i ) when the token length for either x+
i or x−
i exceeds
998

this context length. Our tasks are multiple choice, and we balance our datasets to have equal numbers
999

of these binary labels, unless stated otherwise. For Chinchilla we harvest activations in bfloat16
1000

format and then cast them to float32 for downstream usage. For T5 we harvest activations at float32.
1001

27


---Page Break---
C.3
Method Training Details
1002

We now give further details for the training of our various methods. Each method uses 50 random
1003

seeds.
1004

C.3.1
CCS
1005

We use the symmetric version of the confidence loss, see Equation (13). We use a linear probe with
1006

m weights, θ, and a single bias, b, where m is the dimension of the activation, followed by a sigmoid
1007

function. We use Haiku’s [20] default initializer for the linear layer: for θ a truncated normal with
1008

standard deviation 1/√m, and b = 0. We use the following hyperparameters: we train with full
1009

batch; for Chinchilla models we use a learning rate of 0.001, for T5 models, 0.01. We use AdamW
1010

optimizer with weight decay of 0. We train for 1000 epochs. We report results on all seeds as we are
1011

interested in the overall robustness of the methods (note the difference to Burns et al. [9] which only
1012

report seed with lowest CCS loss).
1013

C.3.2
PCA
1014

We use the Scikit-learn [33] implementation of PCA, with 3 components, and the randomized SVD
1015

solver. We take the classifier to be based around whether the projected datapoint has top component
1016

greater than zero. For input data we take the difference between contrast pair activations.
1017

C.3.3
K-means
1018

We use the Scikit-learn [33] implementation of K-means, with two clusters and random initialiser.
1019

For input data we take the difference between contrast pair activations.
1020

C.3.4
Random
1021

This follows the CCS method setup above, but doesn’t do any training, just evaluates using a probe
1022

with randomly initialised parameters (as initialised in the CCS method).
1023

C.3.5
Logistic Regression
1024

We use the Scikit-learn [33] implementation of Logistic Regression, with liblinear solver and using
1025

a different random shuffling of the data based on random seed. For input data we concatenate the
1026

contrast pair activations. We report training accuracy.
1027

D
Further Results
1028

D.1
Discovering random words
1029

Here we display results for the discovering random words experiments using datasets IMDb, BoolQ
1030

and DBpedia and on each model. For Chinchilla-70B BoolQ and DBPedia see Figure 6 (for IMDb
1031

see Figure 2). We see that BoolQ follows a roughly similar pattern to IMDb, except that the default
1032

ground truth accuracy is not high (BoolQ is arguably a more challenging task). DBpedia shows
1033

more of a noisy pattern which is best explained by first inspecting the PCA visualisation for the
1034

modified prompt (right): there are groupings into both choice 1 true/false (blue orange) which is more
1035

prominent and sits along the top principal component (x-axis), and also a grouping into banana/shed
1036

(dark/light), along second component (y-axis). This is reflected in the PCA and K-means performance
1037

here doing well on ground-truth accuracy. CCS is similar, but more bimodal, sometimes finding the
1038

ground-truth, and sometimes the banana/shed feature.
1039

For T5-11B (Figure 7) on IMDB and BoolQ we see a similar pattern of results to Chinchilla, though
1040

with lower accuracies. On DBpedia, all of the results are around random chance, though logistic
1041

regression is able to solve the task, meaning this information is linearly encoded but perhaps not
1042

salient enough for the unsupervised methods to pick up.
1043

T5-FLAN-XXL (Figure 8) shows more resistance to our modified prompt, suggesting fine-tuning
1044

hardens the activations in such a way that unsupervised learning can still recover knowledge. For
1045

28


---Page Break---
Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

     Correct Answer
True
False

X
0

60
Y

−50

0

50

X

−20

0

20
Y

−20

0

20

Default prompt
Banana/Shed
prompt

Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

Correct Answer
Choice 1
Choice 2

X

−60

0

60

Y

−60

0

60

X

−15

0

15
Y

−10

0

10

Default prompt
Banana/Shed
prompt

Figure 6: Discovering random words, Chinchilla, extra datasets: Top: BoolQ, Bottom: DBpedia.

CCS though in particular, we do see a bimodal distribution, sometimes learning the banana/shed
1046

feature.
1047

D.2
Discovering an explicit opinion
1048

D.2.1
Other models and datasets
1049

Here we display results for the experiments on discovering an explicit opinion using datasets IMDB,
1050

BoolQ and DBpedia, and models Chinchilla-70B (Figure 9), T5-11B (Figure 10) and T5-FLAN-XXL
1051

(Figure 11). For Chinchilla-70B and T5 we use just a single mention of Alice’s view, and for T5-
1052

FLAN-XXL we use five, since for a single mention the effect is not strong enough to see the effect,
1053

perhaps due to instruction-tuning of T5-FLAN-XXL. The next appendix Appendix D.2.2 ablates the
1054

number of mentions of Alice’s view. Overall we see a similar pattern in all models and datasets, with
1055

unsupervised methods most often finding Alice’s view, though for T5-FLAN-XXL the CCS results
1056

are more bimodal in the modified prompt case.
1057

D.2.2
Number of Repetitions
1058

In this appendix we present an ablation on the discovering explicit opinion experiment from Sec-
1059

tion Section 4.2. We vary the number of times the speaker repeats their opinion from 0 to 7 (see
1060

Appendix C.1 Explicit opinion variants), and in Figure 12 plot the accuracy in the method predicting
1061

the speaker’s view. We see that for Chinchilla and T5, only one repetition is enough for the method
1062

to track the speaker’s opinion. T5-FLAN-XXL requires more repetitions, but eventually shows the
1063

same pattern. We suspect that the instruction-tuning of T5-FLAN-XXL is responsible for making
1064

this model somewhat more robust.
1065

29


---Page Break---
Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

Review Sentiment
Positive
Negative

X

−15

0

15
Y

−10

0

10

X

−6

0

6
Y

0

8

Default prompt
Banana/Shed
prompt

Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

     Correct Answer
True
False

X

−15

0

15
Y

−10

0

10

X

−6

0

6
Y

0

8

Default prompt
Banana/Shed
prompt

Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

Correct Answer
Choice 1
Choice 2

X

−10

0

10

Y

0

8
X

−6

0

6
Y

−5

0

5

Default prompt
Banana/Shed
prompt

Figure 7: Discovering random words, T5 11B. Top: IMDB, Middle: BoolQ, Bottom: DBpedia.

D.2.3
Model layer
1066

We now look at whether the layer, in the Chinchilla70B model, affects our results. We consider
1067

both the ground-truth accuracy on default setting, Figure 13, and Alice Accuracy under the modified
1068

setting (with one mention of Alice’s view), Figure 14. Overall, we find our results are not that
1069

sensitive to layer, though often layer 30 is a good choice for both standard and sycophantic templates.
1070

In the main paper we always use layer 30. In the default setting, Figure 13, we see overall k-means
1071

and PCA are better or the same as CCS. This is further evidence that the success of unsupervised
1072

learning on contrastive activations has little to do with the consitency structure of CCS. In modified
1073

30


---Page Break---
Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

Review Sentiment
Positive
Negative

X

−30

0

30

Y

−15

0

15

X

−25

0

25

Y

−10

0

10

Default prompt
Banana/Shed
prompt

Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

     Correct Answer
True
False

X

−30

0

30

Y

−15

0

15

X

−25

0

25

Y

−10

0

10

Default prompt
Banana/Shed
prompt

Prompt template
Default
Banana/Shed

Accuracy 

basis

 Ground truth
 Banana/Shed

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-Means
Random
Log. Reg.

Distractor label
Banana
Shed

Correct Answer
Choice 1
Choice 2

X

−30

0

30

Y

0

15

X

−20

0

20

Y

−8

0

8

Default prompt
Banana/Shed
prompt

Figure 8: Discovering random words, T5-FLAN-XXL. Top: IMDB, Middle: BoolQ, Bottom:
DBpedia.

setting, we see all layers suffer the same issue of predicting Alice’s view, rather than the desired
1074

accuracy.
1075

D.3
Discovering an implicit opinion
1076

In this appendix we display further results for Section 4.3 on discovering an implicit opinion.
1077

Figure 15 displays the results on the T5-11B (top) and T5-FLAN-XXL (bottom) models. For T5-11B
1078

we see CCS, under both default and modified prompts, performs at about 60% on non-company
1079

questions, and much better on company questions. The interpretation is that this probe has mostly
1080

31


---Page Break---
Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
True
False

X

−40

0

40
Y

−40

0

40

X

−80

0

80

Y

−50

0

50

Default prompt
Alice-opinion prompt

Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
Choice 1
Choice 2

X

−80

0

80

Y

−80

0

80

X

−100

0

100

Y

−60

0

60

Default prompt
Alice-opinion prompt

Figure 9: Discovering an explicit opinion, Chinchilla, extra datasets. Top: BoolQ, Bottom: DBpedia.

learnt to classify whether a topic is company or not (but not to distinguish between the other thirteen
1081

categories). PCA and K-means are similar, though with less variation amongst seeds (showing less
1082

bimodal behaviour). PCA visualisation doesn’t show any natural groupings.
1083

For T5-FLAN-XXL the accuracies are high on both default and modified prompts for both company
1084

and non-company questions. We suspect that a similar trick as in the case of explicit opinion,
1085

repeating the opinion, may work here, but we leave investigation of this to future work. PCA
1086

visualisation shows some natural groups, with the top principal component showing a grouping based
1087

on whether choice 1 is true or false (blue/orange), but also that there is a second grouping based on
1088

company/non-company (dark/light). This suggests it is more luck that the most prominent direction
1089

here is choice 1 is true or false, but could easily have been company/non-company (dark/light).
1090

D.4
Prompt Template Sensitivity – Other Models
1091

In Figure 16 we show results for the prompt sensitivity experiments on the truthfulQA dataset, for the
1092

other models T5-FLAN-XXL (top) and T5-11B (bottom). We see similar results as in the main text
1093

for Chinchilla70B. For T5 all of the accuracies are lower, mostly just performing at chance, and the
1094

PCA plots do not show natural groupings by true/false.
1095

D.5
Number of Prompt templates
1096

In the main experiments for this paper we use a single prompt template for simplicity and to isolate
1097

the differences between the default and modified prompt template settings. We also investigated the
1098

effect of having multiple prompt templates, as in [9], see Figure 17. Overall we do not see a major
1099

effect. On BoolQ we see a single template is slightly worse for Chinchilla70B and T5, but the same
1100

for T5-FLAN-XXL. For IMDB on Chinchilla a single template is slightly better than multiple, with
1101

32


---Page Break---
Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Review Sentiment
Positive
Negative

X

−15

0

15
Y

0

15

X

−20

0

20
Y

0

15

Default prompt
Alice-opinion prompt

Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
True
False

X

−15

0

15
Y

−10

0

10

X

−15

0

15

Y

−15

0

15

Default prompt
Alice-opinion prompt

Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
Choice 1
Choice 2

X
0

8

Y

−6

0

6

X

−20

0

20

Y

0

8

Default prompt
Alice-opinion prompt

Figure 10: Discovering an explicit opinion, T5 11B. Top: IMDB, Middle: BoolQ, Bottom: DBpedia.

less variation across seeds. For DBPedia on T5, a single template is slightly better. Other results are
1102

roughly the same.
1103

D.6
Agreement between unsupervised methods
1104

Burns et al. [9] claim that knowledge has special structure that few other features in an LLM are likely
1105

to satisfy and use this to motivate CCS. CCS aims to take advantage of this consistency structure,
1106

while PCA ignores it entirely. Nevertheless, we find that CCS and PCA8 make similar predictions.
1107

We calculate the proportion of datapoints where both methods agree, shown in Figure 18 as a heatmap
1108

according to their agreement. There is higher agreement (top-line number) in all cases than what
1109

one would expect from independent methods (notated “Ind:”) with the observed accuracies (shown
1110

8PCA and k-means performed similarly in all our experiments so we chose to only focus on PCA here

33


---Page Break---
Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Review Sentiment
Positive
Negative

X

−40

0

40
Y

0

15

X

−50

0

50

Y

−40

0

40

Default prompt
Alice-opinion prompt

Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
True
False

X

−40

0

40

Y

−15

0

15

X

−25

0

25

Y

−20

0

20

Default prompt
Alice-opinion prompt

Prompt template
Default
Alice

Accuracy 

basis

 Ground truth
 Alice’s opinion

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

Distractor label
Alice: Negative
Alice: Positive

Correct Answer
Choice 1
Choice 2

X

−20

0

20

Y

−10

0

10

X

−40

0

40

Y

−20

0

20

Default prompt
Alice-opinion prompt

Figure 11: Discovering an explicit opinion, T5-FLAN-XXL. Top: IMDB, Middle: BoolQ, Bottom:
DBpedia.

in parentheses in the heatmap). This supports the hypothesis of Emmons [16] and suggests that
1111

the consistency-condition does not do much. But the fact that two methods with such different
1112

motivations behave similarly also supports the idea that results on current unsupervised methods may
1113

be predictive of future methods which have different motivations.
1114

34


---Page Break---
0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(a) Chinchilla, BoolQ

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(b) Chinchilla, IMDB

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(c) Chinchilla, DBpedia

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(d) T5, BoolQ

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(e) T5, IMDB

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(f) T5, DBpedia

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(g) T5-FLAN-XXL, BoolQ

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(h) T5-FLAN-XXL, IMDB

0 1 2 3 4 5 6 7

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Alice Accuracy

0 1 2 3 4 5 6 7

PCA

0 1 2 3 4 5 6 7

K-means

0 1 2 3 4 5 6 7

Random

0 1 2 3 4 5 6 7

Log. Reg.

(i) T5-FLAN-XXL, DBpedia

Figure 12: Discovering an explicit opinion. Accuracy of predicting Alice’s opinion (y-axis) varying
with number of repetitions (x-axis). Rows: models, columns: datasets.

35


---Page Break---
10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(a) CCS, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(b) CCS, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(c) CCS, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(d) PCA, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(e) PCA, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(f) PCA, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(g) K-means, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(h) K-means, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(i) K-means, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(j) Random, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(k) Random, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(l) Random, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(m) Log. Reg., BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(n) Log. Reg., IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(o) Log. Reg., DBpedia

Figure 13: Default setting, ground-truth accuracy (y-axis), varying with layer number (x-axis). Rows:
models, columns: datasets.

36


---Page Break---
10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(a) CCS, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(b) CCS, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(c) CCS, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(d) PCA, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(e) PCA, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(f) PCA, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(g) K-means, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(h) K-means, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(i) K-means, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(j) Random, BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(k) Random, IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(l) Random, DBpedia

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(m) Log. Reg., BoolQ

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(n) Log. Reg., IMDB

10
20
30
40
50
60
70
Layer

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

(o) Log. Reg., DBpedia

Figure 14: Discovering an explicit opinion. Modified setting, Alice Accuracy, predicting Alice’s
opinion (y-axis), varying with layer number (x-axis). Rows: models, columns: datasets.

37


---Page Break---
Prompt template
Default
Anti-capitalist

Data subset
 Company
 Non-company

CCS

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

Data subset
Non-Company
Company

Correct answer Choice 1

Choice 2

X

−10

0

10

Y

−8

0

8

X

−8

0

8

Y

−8

0

8

Default prompt
Anti-capitalist
prompt

Prompt template
Default
Anti-capitalist

Data subset
 Company
 Non-company

CCS

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

Data subset
Non-Company
Company

Correct answer Choice 1

Choice 2

X

−15

0

15

Y

−10

0

10

X

−15

0

15

Y

−15

0

15

Default prompt
Anti-capitalist
prompt

Figure 15: Discovering an implicit opinion, other models. Top: T5-11B, Bottom: T5-FLAN-XXL.

38


---Page Break---
CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

Default
Literal
Professor

(a) Variation in accuracy

X

−25

0

25
Y

−20

0

20

X

−20
0
20
Y

−20

0

20
X

−20
0
20
Y

−20

0

20

False
True

Default
Literal
Professor
(b) PCA Visualisation

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
KMeans
Random
Log. Reg.

Default
Literal
Professor

(c) Variation in accuracy

X

−25
0
25
Y

−20

0

20
X

0

20
Y

−15

0

15

X
0

20
Y

0

20

False
True

Default
Literal
Professor
(d) PCA Visualisation

Figure 16: Prompt sensitivity on TruthfulQA [26], other models: T5-FLAN-XXL (top) and T5-11B
(bottom). (Left) In default setting (blue), accuracy is poor. When in the literal/professor (red, green)
setting, accuracy improves, showing the unsupervised methods are sensitive to irrelevant aspects of a
prompt. The pattern is the same in all models, but on T5-11B the methods give worse performance.
(Right) 2D view of 3D PCA of the activations based on ground truth, blue vs. orange in the default
(left), literal (middle) and professor (right) settings. We see do not see ground truth clusters in the
Default setting, but do in the literal and professor setting for Chincilla70B, but we see no clusters for
T5-11B.

39


---Page Break---
CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

CCS

0.5

0.6

0.7

0.8

0.9

1.0

Accuracy

PCA
K-means
Random
Log. Reg.

IMDb
BoolQ
DBPedia_14

Figure 17: Effect of multiple prompt templates. Top: Chinchilla70B. Middle: T5. Bottom: T5-
FLAN-XXL. Left: Multiple prompt templates, as in Burns et al. [9]. Right: Single prompt template
‘standard’. We do not see a major benefit from having multiple prompt templates, except on BoolQ,
and this effect is not present for T5-FLAN-XXL.

40


---Page Break---
BoolQ
DBpedia
IMDB

Chinchilla
Flan-T5
T5

0.74
Ind:0.61
(0.72, 0.74)

0.90
Ind:0.88
(0.92, 0.95)

0.87
Ind:0.81
(0.85, 0.94)

0.98
Ind:0.82
(0.9, 0.9)

1.00
Ind:1.00

(1, 1)

0.98
Ind:0.93
(0.97, 0.96)

0.57
Ind:0.52
(0.59, 0.61)

0.90
Ind:0.80
(0.88, 0.9)

0.92
Ind:0.84
(0.94, 0.89)
0.6

0.7

0.8

0.9

1.0

Figure 18: CCS and PCA make similar predictions. In all cases, CCS and PCA agree more
than what one would expect of independent methods with the same accuracy. Annotations in each
cell show the agreement, the expected agreement for independent methods, and the (CCS, PCA)
accuracies, averaged across 10 CCS seeds.

41


---Page Break---
