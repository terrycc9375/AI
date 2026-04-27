Label Privacy in Split Learning for Large Models with
Parameter-Efficient Training

Anonymous Author(s)
Affiliation
Address
email

Abstract

As deep learning models become larger and more expensive, many practitioners
1

turn to fine-tuning APIs. These web services allow fine-tuning a model between
2

two parties: the client that provides the data, and the server that hosts the model.
3

While convenient, these APIs raise a new concern: the data of the client is at
4

risk of privacy breach during the training procedure. This challenge presents
5

an important practical case of vertical federated learning, where the two parties
6

perform parameter-efficient fine-tuning (PEFT) of a large model. In this study, we
7

systematically search for a way to fine-tune models over an API while keeping the
8

labels private. We analyze the privacy of LoRA, a popular approach for parameter-
9

efficient fine-tuning when training over an API. Using this analysis, we propose
10

P3EFT, a multi-party split learning algorithm that takes advantage of existing PEFT
11

properties to maintain privacy at a lower performance overhead. To validate our
12

algorithm, we fine-tune DeBERTa-v2-XXLarge, Flan-T5 Large and LLaMA-2 7B
13

using LoRA adapters on a range of NLP tasks. We find that P3EFT is competitive
14

with existing privacy-preserving methods in multi-party and two-party setups while
15

having higher accuracy.
16

1
Introduction
17

One of the main reasons behind deep learning success is its ability to transfer knowledge between
18

tasks [34]. When training a model for any particular problem, it is common to reuse previously
19

trained models from other, related problems. In the past, this was typically done by downloading
20

pre-trained model weights from public hubs, then fine-tuning the said models on the downstream
21

task. However, as models grow larger and more compute-intensive, fine-tuning them locally becomes
22

an increasingly difficult task. Furthermore, many recent models are not released, but instead made
23

available as proprietary services.
24

When a model cannot be fine-tuned locally, many practitioners opt instead for the so-called fine-tuning
25

APIs [27, 16, 6, 26]. These APIs are web services that host one or several pre-trained models and
26

allow clients to perform limited fine-tuning. More specifically, APIs usually allow their clients to run
27

parameter-efficient fine-tuning (PEFT), such as LoRA [15] or Prefix-tuning [21]. These techniques
28

allow adapting a model to a dataset while training a relatively small number of additional weights,
29

which is particularly important for large language or image generation models that have billions of
30

parameters.
31

Although the fine-tuning APIs can be convenient, they also introduce new risk in terms of data privacy.
32

When a client uses such API to train on sensitive data, they need to ensure that their data will stay
33

private [7]. This is particularly important when dealing with patient’s medical records, personal
34

user data or trade secrets [24, 19]. The two main threats to data privacy are that the API provider
35

obtains the private data and that a third party intercepts data in transit. Therefore, data privacy is
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
not guaranteed even if the API provider is trusted. Several recent works propose LLM fine-tuning
37

protocols that establish a certain level of privacy for multi-party fine-tuning [42, 7, 22]. Unfortunately,
38

these algorithms work for a narrow class of fine-tuning algorithms or assume that a client can run
39

LLM training locally using an obfuscated version of the model, provided by a remote server [42].
40

As a result, these algorithms are impractical for our use case of fine-tuning over an API. The few
41

algorithms that are suitable for API fine-tuning guarantee the privacy of input tokens [22], meaning
42

that the attacker can infer private training labels.
43

In this work, we seek to alleviate this problem by designing a two-party fine-tuning protocol that
44

performs standard parameter-efficient fine-tuning with privacy guarantees. We formulate our protocol
45

as a special case of split learning (or vertical federated learning), where one side (server) holds the
46

pre-trained model and the other (client) has private training data. More specifically, we focus on the
47

privacy of client’s training labels. While input privacy is often crucial, there are scenarios where
48

input data is publicly available, such as social media user pages. In these cases, labels could include
49

ad clicks (known to the social network) or financial information (known to a bank that matches social
50

profiles to its internal records). This example further justifies the use of LLMs, as social media pages
51

often contain substantial amounts of text, and LLMs excel at processing long-context data.
52

Instead of developing a specific privacy-preserving architecture, we seek algorithms that can work
53

with popular existing models and PEFT algorithms. Furthermore, our approach relies on the properties
54

of parameter-efficient fine-tuning. Notably, since the adapters are compact, both parties can maintain
55

multiple sets of adapters and swap between them with relative ease. This allows us to design a
56

PEFT-specific algorithm that can solve its task more effectively than general split learning strategies
57

[18].
58

We summarize our main contributions as follows:
59

• We analyze Low-Rank Adaptation, a common parameter-efficient fine-tuning algorithm,
60

from the perspective of label privacy in the split learning setup. We observe that, despite
61

fine-tuning less than 0.1% of model parameters, PEFT algorithms leak client’s training
62

labels against simple attacks that work for modern pretrained transformers.
63

• Based on our analysis, we formulate a framework for privacy-preserving parameter-efficient
64

fine-tuning (P3EFT). This framework leverages the properties of PEFT to obfuscate the
65

gradients and parameters communicated during fine-tuning with little impact on the fine-
66

tuned model quality.
67

• To verify the practical viability of P3EFT, we conduct experiments on popular real-world
68

PEFT workloads1. Specifically, we fine-tune DeBERTa-v2-XXL [13], Flan-T5-Large [4]
69

and LLaMA-2 7B [35] on a set of standard language understanding problems. We find that,
70

compared to prior split learning algorithms, P3EFT can maintain label privacy throughout
71

training with a significantly smaller accuracy drop.
72

2
Background
73

2.1
Federated learning and split learning
74

Privacy preservation in machine learning has been a subject of active study within several frameworks.
75

An important branch of privacy-preserving learning methods is federated learning, or FL [24], which
76

can be broadly described as an approach allowing several parties to train a model jointly without
77

sharing their private data. In particular, vertical federated learning [12, 43] targets the scenario where
78

different features (including the label) of each training instance are kept by different parties.
79

One of the most popular approaches to vertical FL for neural networks is split learning [10, 37],
80

where each party stores its part of the overall model. To train the model in such an approach, it is
81

only necessary to transfer the intermediate activations and the gradients between layers, while the
82

data itself is stored at the premises of the participant hosting each layer. In this work, we focus on
83

the two-party formulation of split learning, where one side stores the features for each example and
84

another one stores the labels.
85

1The code is available at github.com/anonymousauthor56/P3EFT

2


---Page Break---
Recent works have investigated the setting of two-party split learning from the label leakage per-
86

spective [38, 28]: because the label party needs to pass the gradients of the loss function to the
87

non-label party, it is possible for the latter party to deduce the labels by inspecting the gradients or
88

activations or by hijacking the training procedure. Li et al. [18] provide a set of attack methods that
89

allow recovering private labels and propose a defense mechanism that injects noise into the gradients;
90

however, they test the approach on pretraining smaller models, and we study finetuning large models
91

on private downstream data.
92

2.2
Parameter-efficient finetuning
93

The majority of large neural networks today are not trained with a specific task in mind: instead, they
94

are pretrained on a general objective and then adapted for the downstream problem. Importantly, the
95

growth in the size of foundation models has led to the increased popularity of parameter-efficient
96

finetuning (PEFT) methods that adapt the model to a given task by training a small number of
97

task-specific parameters. There are several prominent approaches to parameter-efficient finetuning,
98

ranging from trainable prompts [21, 11], to residual adapters [14, 29]. We focus on Low-Rank
99

Adaptation (or LoRA, 15), one of the most popular PEFT methods that adds extra parameters to each
100

weight matrix in the form of a low-rank factorization (see Appendix C for a more detailed description).
101

Such formulation allows LoRA adapters to be merged into the original weights after finetuning; this
102

ability, combined with the simplicity of the method, has made LoRA a broadly popular approach in
103

multiple domains. Still, the approach we propose can be applied to any PEFT method.
104

Several recent lines of work explore the problem of fine-tuning LLMs with privacy guarantees [44, 31].
105

Zhao et al. [46] analyze the viability of prompt tuning for federated learning, and Zhang et al. [45], Liu
106

et al. [23] study PEFT algorithms in the setting of horizontal federated learning, that is, where multiple
107

users train a shared model on their local private data. Another, more relevant research direction
108

considers private fine-tuning in a vertical federated learning scenario, where participants hold different
109

model layers [22, 40]. Most of these studies leverage the idea of differential privacy to prove an
110

upper bound on how much information is leaked [8]. Unfortunately, these upper bounds are typically
111

loose and do not match practical observations for real models. Furthermore, the majority of these
112

studies only guarantees privacy of specific parts of the training procedure: for instance, Li et al.
113

[22] only protects the input features, and not labels or model parameters. Finally, Xiao et al. [42]
114

presents an alternative algorithm that protects client data by running the entire fine-tuning on client
115

side by emulating the server-side model layers. While this approach is more holistic, it assumes that
116

clients can run fine-tuning locally, which makes it impractical for many real-world users of LLM
117

fine-tuning APIs. The primary distinction between our work and these studies is that we investigate
118

parameter-efficient adaptation in the setting of split learning: we aim to finetune a model without
119

disclosing the labels of examples to the model provider.
120

3
Privacy-preserving parameter-efficient fine-tuning
121

In this section, we analyze the privacy of parameter-efficient fine-tuning and propose a protocol for
122

two-party parameter-efficient fine-tuning with the desired privacy guarantees. We begin by analyzing
123

the privacy of API fine-tuning with popular PEFT algorithms in Sections 3.1 and 3.2. Then, in
124

Section 3.3, we formulate a protocol for privately computing gradients over fine-tuning APIs. Finally,
125

we formulate the full P3EFT protocol in Section 3.4.
126

3.1
Setup
127

To analyze the privacy of API fine-tuning, we first need to formulate a common framework for
128

this type of APIs and develop private learning protocols. This step is important, because existing
129

fine-tuning APIs greatly vary in what they offer to the client: from closed APIs that require users to
130

submit their full training data [27] to more flexible APIs where clients can run individual training
131

steps [20, 2, 30]. Similarly to most existing works on split learning, we focus on the latter type of
132

APIs that allows clients to run individual forward and backward passes over a remote model. Thus,
133

a client can use these APIs to obtain the training gradients for their PEFT adapters, then update
134

adapters locally with any optimization method. In our work, we adopt this archetype of fine-tuning
135

API as it offers sufficient flexibility to develop privacy-preserving algorithms.
136

We formulate fine-tuning over an API for two or more parties: a client, and one or several servers.
137

The client owns a training dataset with inputs X and labels Y . In turn, each server has the same
138

pre-trained model h(xi, θ) ∈Rd. Note that the parameters θ denote not the pre-trained model
139

3


---Page Break---
Step: 0
Step: 1000
Step: 4000
Step: 16000

Step: 0
Step: 1000
Step: 4000
Step: 16000

Figure 1: A visualization of top-2 principal components of gradients (top) and activations (bottom)
from different fine-tuning steps (left to right). Color indicates the training labels (binary).

weights, but the trainable adapter weights for a certain PEFT algorithm. A model can encode an input
140

xi ∈X and produce a d-dimensional vector of activations that depend on the learned adapter weights
141

θ.
142

To allow fine-tuning, a server offers two API methods:
143

1. forward(x, θ) →h(x, θ) that computes model activations on input x using adapter weights
144

θ;
145

2. backprop(x, θ, gh) →gθ that receives gradients of an arbitrary loss function w.r.t. model
146

activations gh =
∂L(h(x,θ))

∂h(x,θ)
and returns the gradients w.r.t. adapter parameters, gθ =
147

∂L(h(x,θ))

∂θ
.
148

We further assume that both forward(·) and backprop(·) APIs are stateless and deterministic, i.e.
149

calling the same API method multiple times (or on multiple servers) with the same inputs produces
150

identical results. Thus, if the model uses dropout or any other form of non-determinism, we assume
151

that clients provide the random seed as a part of x.
152

To fine-tune a model with this API, a client can initialize adapters locally, alongside with a small
153

task-specific head2, then train both adapters and the head. For each training batch (x, y) ∈D, a client
154

calls forward(x, θ) to compute feature representations, then predicts with local “head” and computes
155

task-specific loss function L. After that, a client performs backward pass: first, it computes gradients
156

w.r.t. local head inputs gh = ∂L

∂h , then passes those gradients to a remote server via backprop(x, θ, gh)
157

API call to compute gradients w.r.t. ∂L

∂θ . Finally, a client updates both θ and local “head” parameters
158

using the optimizer of choice.
159

Before building more advanced algorithms, let us analyze the privacy of client’s labels under standard
160

fine-tuning. We consider an “honest, but curious” attacker model. This means that the server will
161

faithfully run the forward and backprop computations as requested by the client without changing
162

the results. Furthermore, we assume that servers are independent and do not communicate client’s
163

data between each other. However, a server can recover client’s labels by performing arbitrary
164

computations using any information it receives from the client. When training in this way, a client
165

does not directly communicate training labels to the server. However, it communicates inputs, adapter
166

parameters, and gradients. Furthermore, the server communicates input representations that can be
167

intercepted by a third party.
168

3.2
Label Leakage of Standard Split Learning
169

In Figure 1, we train a DeBERTa-v2-XXL model on the SST-2 [32] sentiment classification dataset.
170

The top row depicts the gradients gh communicated by the client when calling backprop(·) at different
171

training stages. In the bottom row, we similarly track activations h(x, θ) that server may compute
172

based on the specified x, θ. We defer further additional figures and details to Section 4.1.
173

As we can see, both gradients and activations are arranged in such a way that simple k-means
174

clustering would reveal which objects have the same label. The training activations (bottom row) do
175

2A linear layer that predicts class logits or regression target.

4


---Page Break---
SERVER 
CLIENT

n model instances with unique LoRA weights 𝜃 
activations 
weights 
weighted activations 
model head

local layers

h1 
W1 

h2 
W2 

hn 
Wn 

𝜃1 
→ 
→
 → 
 ⊙ 
=

𝜃2
→
→
 → 
 ⊙ 
=

𝜃n
→
→
→
⊙

𝚺=

h’

𝓛

… 
… 
 … 
 … 
 … 
 …

forward

last layer
activations
from n models

backward

weighted
gradients
to n models

… 

Figure 2: An intuitive illustration of the proposed fine-tuning protocol.

not reveal labels right away (at least not against this attack). However, they gradually “leak” private
176

label information during training. Informally, it appears that the training gradients gradually pull
177

apart the feature representations for each label, until eventually they turn into separate clusters. From
178

an information-theoretic perspective, knowing just one vector of gradients or trained activations
179

allows the attacker to learn all but one bit3 of information about client’s private labels.
180

To summarize, leaving any one data source unprotected (gradients, activations or parameters) would
181

already compromise label privacy. However, we found that gradients and activations require different
182

means of protection.
183

3.3
Privacy-preserving backpropagation
184

In this section, we formulate an algorithm for “anonymizing” the gradients communicated over a
185

single training step with arbitrary PEFT type. Several prior works approach this by modifying the
186

training objective or model architecture. However, when dealing with a real-world PEFT workload
187

with optimized hyperparameters, changing the model or loss function often results in reduced model
188

accuracy4. Thus, we seek an algorithm that preserves both model and training objective.
189

We design our algorithm based on an observation that backpropagation is conditionally lin-
190

ear in output gradients, even when the model itself is nonlinear. Formally, if we take a model
191

h(·, ·), a fixed set of trainable parameters θ and input samples x, the backprop function5 computes
192

backprop(x, θ,
∂L
∂h(x,θ)) = ∂L

∂θ . For convenience, we shorten it to backprop(x, θ, gh) = gθ, where
193

gh =
∂L
∂h(x,θ) represents the gradients of some objective function with respect to model activations
194

(outputs), and gθ = ∂L

∂θ are gradients of the same objective function w.r.t. trainable parameters. In
195

this notation, backprop is linear in terms of gh for any fixed x, θ.
196

This becomes self-evident if we view backprop as multiplying gh by the Jacobian of model outputs
197

w.r.t. trainable parameters, ∂h(x,θ)

∂θ
. If x, θ are constant, the Jacobian is also constant, and backprop
198

is a linear operator:
199

backprop(x, θ,
∂L
∂h(x, θ)) = ∂L

∂θ =
∂L
∂h(x, θ) × ∂h(x, θ)

∂θ
.
(1)

This observation allows us to design a private backpropagation protocol.
To illustrate
200

this protocol, let us first consider a distributed API with two identical independent servers
201

that offer backprop API. Then, for arbitrary vector z, we can rewrite backprop(x, θ, gh) as
202

backprop(x, θ, gh+z)+backprop(x, θ, gh−z).
203

During API fine-tuning, we obtain backprop(x, θ, gh + z) using an API call to server 1, whereas the
204

second term backprop(x, θ, gh −z) translates to an API call to server 2. Note that neither of two
205

servers has access to the true gradient gh: they only receive the sum [gh + z]. If we sample a large
206

noise vector z (Var(z) ≫∥gh∥2
2), this sum also becomes dominated by noise. However, when both
207

API calls finish, a client can sum the results to recover the true gradient of the loss with respect to
208

parameters.
209

If both requests are processed by the same server, it can obviously recover gh by adding up gradients
210

from both calls, which leads us to the final step. Instead of generating a single noise vector, a client
211

3The missing bit corresponds to attacker not knowing which cluster corresponds to label “1”.
4We validate this empirically in 4.2.
5This is the same as the backprop API defined in Section 3.1.

5


---Page Break---
needs to generate (privately) a set of m > 1 random vectors ˆg1
h, . . . , ˆgm
h and scalars α1, . . . , αm such
212

that
213

gh =

m
X

i=1
αi · ˆgi
h.
(2)

Then, for each ˆgi
h, client computes backprop(x, θ, ˆgi
h) as m parallel API calls. Once this is done,
214

client recovers
215

gθ =

m
X

i=1
αi · backprop(x, θ, ˆgi
h).
(3)

Note that the client does not reveal α1, . . . , αm to anyone.
216

The resulting procedure is formulated in Algorithm 1. This algorithm is conceptually similar to
217

the secure aggregation protocol for conventional (horizontal) federated learning [1]. This protocol
218

allows clients to average their local vector with peers while keeping each individual vector provably
219

private. Similarly to our scheme, clients perturb the vector in such a way that the average of perturbed
220

vectors remains the same. Unlike Bonawitz et al. [1], our protocol privately backpropagates through
221

a server-hosted model by leveraging the conditional linearity of the backpropagation operator.
222

Algorithm 1 private_backprop — Privacy-Preserving Backpropagation (from the client’s perspective)

1: Input: x inputs, θ adapter weights, gh gradients w.r.t. activations, m > 1 - number of passes
2: ˆg1
h, . . . , ˆgm
h , α1, . . . , αm = obfuscate(gh, m)
▷2
3: for j = 1, . . . , m do
4:
ˆgj
θ = backprop(x, θ, ˆgj
h)
▷computed by server
5: end for
6: gθ = Pm
j=1 αj · ˆgj
θ
7: Return: gθ

The private backpropagation algorithm can allow client to safely compute gradients once, but, in
223

practice, client usually needs to run many consecutive steps. This creates an additional vector of
224

attack: if the same server receives two sets of parameters θt, θt+1 , they could potentially recover gθ
225

by inverting the optimizer.
226

In the simplest case, if the server somehow knows that the client computes θt+1 = θt −η · gθ, then
227

they can compute gθ = (θt −θt+1)/η. While gθ does not necessarily leak private labels, a server
228

could, in some cases, use gθ to recover gh, either fully (e.g. if Jacobian is invertible), or partially.
229

The client has two ways to prevent this attack. The first one is to ensure that no single server runs
230

backprop on two consecutive steps. This is easy to do in decentralized systems where there are
231

many potential servers. However, even when there is a single server, they could be required to set up
232

multiple trusted execution environments [25]. A more risky alternative is to ensure that the gradients
233

cannot be reversed from consecutive parameters: randomize initial optimizer statistics or add noise to
234

parameters. This solution is easier, but it can slow down training in some cases.
235

To summarize, we formulated a procedure that allows a client to compute gradients privately for any
236

given model and PEFT type. Furthermore, since Equation 3 recovers true gradients, this obfuscation
237

method does not affect the training dynamics. However, as we have shown in Section 3.1, gradients
238

are not the only source of privacy leakage.
239

3.4
Full fine-tuning
240

The other major attack vector are training activations. As the model fits to training data, it’s
241

intermediate activations h(x, θ) allow attackers to recover labels, e.g. by clustering (see Figure 1).
242

To combat this issue, we take advantage of the fact that PEFT has few trainable parameters. Instead
243

of learning just one set of trainable parameters, a client creates n independent adapter sets θ1, ..., θn.
244

Note that this does not require n unique servers: a single server can run multiple sets of adapters.
245

Furthermore, a client can alternate between using different servers for the same adapters. During
246

forward pass, the outputs of different adapters are mixed together using randomized mixing weights
247

W ∈Rn,d:
248

h′(x, θ1, . . . , θn) =

n
X

i=1
Wi ⊙h(x, θi)
(4)

6


---Page Break---
249
Overall, we design this model in such a way the combined model h′ can predict the labels, but the
250

adapters h(x, θi) do not allow predicting these labels without knowing the mixing weights W. The
251

mixing weights are generated such that initial activations h′(x, . . . ) are equal to mean h(x, ·) for all
252

x. To achieve this, we generate W as follows: first, we generate n · (n −1)/2 d-dimensional random
253

vectors ξi,j ∈Rd∀i ∈[1, n], j ∈[i + 1, n]. Then, we add them up in the following way:
254

W =







1
ne + ξ1,2 + ξ1,3 + · · · + ξ1,n
−ξ1,2 + 1

ne + ξ2,3 + · · · + ξ2,n
. . .
−ξ1,n −ξ2,n −ξ3,n −· · · + 1

ne






(5)

Here, e stands for a vector of all ones. The purpose of these mixing weights is to ensure that the
255

gradients w.r.t. individual h(x, θi) are obfuscated, but the averaged model behaves the same as
256

regular PEFT adapter. To illustrate this, consider n=2 identical LoRA adapters θ1, θ2. During the
257

first training step h(x, θ1) = h(x, θ2). Therefore,
258

h′(x, θ1, . . . , θn) = (1/2e + ξ1,2) ⊙h(x, θ1) + (1/2e −ξ1,2) ⊙h(x, θ2) = h(x, θ1)
(6)

However, the two adapters will learn different functions as they receive different gradients. From the
259

first update on, h′ will be equal to an average of adapter predictions.
260

Finally, to ensure that individual adapters h(x, θ) do not accidentally “learn to leak” labels, we
261

maintain this over the course of training with a privacy regularizer inspired by [9]. This ensures that
262

it is impossible to predict labels from individual adapters h(x, θi). Intuitively, on each training step,
263

client fits n linear “heads” that learn to predict labels y from h(x, θi), then performs an adversarial
264

update of θi to prevent the “head” from predicting y. Formally, each of n “heads” minimize the same
265

objective function as the full model. For instance, if the full model solves multi-class classification,
266

each head is trained to minimize cross-entropy:
267

η∗
i = arg min
ηi

X

x,y∈D
−y · log
e⟨ηij,h(x,θi)⟩
P

k e⟨ηik,h(x,θi)⟩,
(7)

268

where y is one-hot encoding of the correct class.
269

The whole adversarial update takes place locally on client’s side, using the same h(x, θ) it uses for the
270

main training objective. The resulting procedure appears complicated but it typically takes negligible
271

time compared to running the large pre-trainied model h(x, θ). Furthermore, since adversarial “heads”
272

are linear, minimizing the objective above is done with standard logistic regression solver.
273

To summarize, our approach combines the two proposed ideas: we use the private backpropagation
274

algorithm from Section 3.3 to protect the gradients, then trains a mixture of adapters in such a way
275

that obfuscates learned activatons leaking labels. The resulting procedure is described in Algorithm 2.
276

In the next section, we will evaluate the efficacy of P3EFT on popular NLP benchmarks.
277

4
Experiments
278

The main goal of our study is to find a practical method of private fine-tuning that would scale to
279

large models. Because our approach leverages parameter-efficient fine-tuning techniques, we evaluate
280

P3EFT with fine-tuning Transformer models on popular NLP benchmarks that these techniques were
281

designed for.
282

To that end, we chose three pre-trained models: DeBERTa-XXLarge [13], Flan-T5-Large [4] and
283

LLaMA-2 7B [35]. We train these models on several datasets from the GLUE benchmark [39]:
284

SST-2 [32], MNLI [41] and QNLI.
285

4.1
Privacy of gradients and activations
286

For this experiment, we train DeBERTa-XXLarge on SST-2 dataset using LoRA adapters with
287

hyperparameters from [15]. First, we train the model locally and track model activations h and
288

gradients w.r.t. those activations. We apply principal component analysis to them and plot the first
289

7


---Page Break---
Step: 0
Step: 1000
Step: 4000
Step: 16000

Step: 0
Step: 1000
Step: 4000
Step: 16000

Figure 3: Gradients of cross-entropy w.r.t. LoRA parameters for DeBERTa-v2-XXLarge. The top
row corresponds to normal backpropagation and the bottom row uses privacy-preserving backprop.

2 dimensions in Figure 1. Similarly, we visualize gradients of individual per-sample loss functions
290

w.r.t. LoRA parameters θ in Figure 3 (top row). The results suggest that a hypothetical attacker could
291

easily recover private labels by performing K-Means clustering over any data source: activations,
292

gradients with respect to activations, or individual gradients with respect to parameters.
293

Next, we run the same experiment using privacy-preserving backpropagation as defined in Section 3.3.
294

We use n = 2 with the noise variance set to 1000. As expected, we observed the same learning curve
295

as with normal training. However, instead of sending gradients w.r.t. activations to the server, a
296

client uses specially crafted random noise vectors that are not informative. In Figure 3 (bottom) we
297

plot the same kind of individual gradients as in the top row, except that we visualize the gradients
298

computed by the first of the two servers. Finally, we train XGBoost [3] with default hyperparameters
299

to predict labels given the noisy gradients (pre-PCA): the resulting classifier is able to fit the training
300

data perfectly, but has at most 50.4% accuracy on a balanced test set.
301

4.2
Main fine-tuning experiments
302

Next, we evaluated the entire P3EFT algorithm. To control tasks and model type, we examined
303

DeBERTa and Flan-T5 across all four datasets mentioned above, in addition to evaluating LLaMA on
304

SST2 and QNLI datasets. For each setup, we compare against three baselines:
305

• Without LoRAs. In this baseline, the client gathers h activations at the beginning (with no
306

adapters), then proceeds to train local “head” layers using these activations. This method cannot
307

leak information about training labels except for what is stored in X.
308

• Regular fine-tuning (Regular FT) refers to training a single LoRA adapter normally. This baseline
309

represents an upper bound on model accuracy, but lacks privacy.
310

• Distance Correlation (DC). Our re-implementation of the distance correlation defense formulated
311

in [33] for Transformer models.
312

For each algorithm, we evaluated a task-specific metric (accuracy or F1), as well as the privacy
313

leakage value for the 3 following measures:
314

• Spectral attack AUC — a measure of vulnerability to an attack proposed in [33], measured as
315

classifier ROC AUC: lower value corresponds to better privacy.
316

• Norm attack AUC — vulnerability to a variant of attack proposed in [18], measured as classifier
317

ROC AUC (lower is better). Despite the initial proposal of this approach for attacking gradients,
318

we observed that it is also well-suited for attacking activations.
319

• K-means accuracy — vulnerability to clusterization attack, measured in the percentage of correctly
320

clustered activations, lower is better.
321

For all setups, we report the worst (least private) value among these metrics throughout the entire
322

training period as a measure of privacy leakage, because it is the worst possible scenario that matters
323

from the client’s perspective. For DC and P3EFT, we specify the values for the best configuration in
324

terms of the utility-privacy trade-off. See details in Appendix A. We also report adjusted standard
325

deviations for the two privacy aware algorithms: P3EFT and DC. To do so, we run the full training
326

procedure from scratch with 3 random seeds.
327

8


---Page Break---
Table 1: Accuracy and privacy metrics.
DeBERTa XXLarge.

Dataset
Without Regular
DC
P3EFT
LoRAs
FT

SST2 acc
82.9
96.9
96.6±0.4 96.5±0.2
leak 53.9
99.1
93.3±6.8 62.6±2.6

QNLI acc
72.6
96.0
95.8±0.3 95.6±0.5
leak 51.5
99.1
85.0±11.6 74.6±11.1

MNLI acc
49.2
91.9
—
86.9±0.5
leak 34.2
91.5
—
37.4±0.7

Table 2: Accuracy and privacy metrics.
Flan-T5-Large.

Dataset
Without Regular
DC
P3EFT
LoRAs
FT

SST2 acc
92.8
96.1
95.0±0.1 96.1±0.1
leak 55.8
98.3
68.1±5.0 74.1±3.0

QNLI acc
83.2
95.3
95.2±0.1 94.7±0.0
leak 58.7
98.9
67.0±1.2 63.0±0.8

MNLI acc
73.9
90.5
89.8±0.1 90.1±0.1
leak 34.6
85.9
45.6±0.8 40.0±1.1

The results for DeBERTa are presented in Table 1. To improve reproducibility, we reuse the hyperpa-
328

rameters from original paper, with the exception of the LoRA dropout value. We disable dropout
329

because it interferes with the mixing weights (5). In preliminary experiments, we observed that with
330

dropout enabled, both our algorithm and DC begin to perform significantly worse.
331

We use n = 2 adapter sets for P3EFT for all datasets and adhered to the same approach for the
332

other models as well. Overall, P3FT achieves nearly the same accuracy as traditional (non-private)
333

fine-tuning, outperforming the DC-based algorithm in terms of accuracy given the same privacy level.
334

On the MNLI dataset, we could not find the hyperparameters for DC that ensure stable training while
335

maintaining privacy. Meanwhile, P3EFT maintains consistent performance on this task with a slight
336

drop in quality.
337

Table 2 a reports evaluation for the Flan-T5 base model[4]. For this model, we adapt the exact same
338

hyperparameters as in the previous evaluation with DeBERTa-XXLarge. Compared to DeBERTa,
339

these results are more closely matched. Both both our algorothm and DC consistently solve all three
340

tasks, but P3EFT slightly outperforms DC in terms of privacy.
341

Table 3: Accuracy and privacy metrics for LLaMA-2 7B.

Dataset
Without
Regular
DC
P3EFT
LoRAs
FT

SST2
acc
94.6
97.4
97.1±0.1
95.8±0.1
leak
59.1
99.3
83.6±10.6
68.9±2.6

QNLI
acc
77.0
95.0
95.2±0.1
94.7±0.2
leak
53.3
85.5
66.6±4.1
62.9±0.8

To evaluate how our algorithm scales to larger models, we also fine-tune Llama-2 7B [35] on
342

SST2 [32] and QNLI [39] datasets. For these evaluations, we use LoRA hyperparameters that Hu
343

et al. [15] used when fine-tuning GPT-3, with several changes inspired by Dettmers et al. [5]. Namely,
344

we use the NF4 weight format, apply LoRA to both attention and MLP layers with rank 16. We
345

fine-tune both tasks with maximum context length of 512 and weight decay 0.01. Table 3 summarizes
346

our results: for QNLI, P3EFT achieves somewhat better privacy-accuracy trade-off. On SST2, P3EFT
347

shows similarly favorable trade-offs while DC struggles to preserve privacy.
348

5
Conclusion and Discussion
349

In this work, we analyze privacy-preserving fine-tuning of large neural networks in the context of
350

parameter-efficient fine-tuning and the two-party split learning setting. We show that while standard
351

fine-tuning suffers from label leakage even in the parameter-efficient case, it is possible to leverage
352

the efficiency of PEFT to alter the procedure without any significant performance drawbacks. We test
353

the resulting method, named P3EFT, on a range of pretrained language models and multiple datasets,
354

showing that it is competitive with a strong baseline in terms of label privacy while having higher
355

task performance.
356

In future work, it is natural to explore how this approach can be extended to establish holistic privacy
357

in both labels and inputs. This problem can be approached from two directions: either adapt the
358

ideas of P3EFT for input privacy, or combine it with an existing work like [22]. Another important
359

direction for future research is exploring the privacy of the long-term client-provider interaction. In a
360

typical real-world use case of API fine-tuning, a client performs multiple training runs on overlapping
361

data and hyperparameters. This could open additional attacks vectors that combine information from
362

multiple training runs.
363

9


---Page Break---
References
364

[1] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan,
365

Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for
366

privacy-preserving machine learning. In proceedings of the 2017 ACM SIGSAC Conference on
367

Computer and Communications Security, pages 1175–1191, 2017.
368

[2] Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem
369

Chumachenko, Pavel Samygin, and Colin Raffel. Petals: Collaborative inference and fine-tuning
370

of large models. arXiv preprint arXiv:2209.01188, 2022. URL https://arxiv.org/abs/
371

2209.01188.
372

[3] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of
373

the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
374

KDD ’16, pages 785–794, New York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi:
375

10.1145/2939672.2939785. URL http://doi.acm.org/10.1145/2939672.2939785.
376

[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li,
377

Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu,
378

Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav
379

Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H.
380

Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling
381

instruction-finetuned language models, 2022. URL https://arxiv.org/abs/2210.11416.
382

[5] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient
383

finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.
384

[6] Dreambooth API. Dreambooth API – Easily finetune Stable Diffusion and generate customised
385

AI images — dreamboothapi.ai. https://dreamboothapi.ai/, 2023. [Accessed 28-09-
386

2023].
387

[7] Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic
388

parrots: Differentially private prompt learning for large language models. arXiv preprint
389

arXiv:2305.15594, 2023.
390

[8] Cynthia Dwork. Differential privacy. In International colloquium on automata, languages, and
391

programming, pages 1–12. Springer, 2006.
392

[9] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation.
393

In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference
394

on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1180–
395

1189, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/
396

ganin15.html.
397

[10] Otkrist Gupta and Ramesh Raskar. Distributed learning of deep neural network over multiple
398

agents. Journal of Network and Computer Applications, 116:1–8, 2018. ISSN 1084-8045.
399

doi: https://doi.org/10.1016/j.jnca.2018.05.003. URL https://www.sciencedirect.com/
400

science/article/pii/S1084804518301590.
401

[11] Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. WARP: Word-level Ad-
402

versarial ReProgramming.
In Proceedings of the 59th Annual Meeting of the Associa-
403

tion for Computational Linguistics and the 11th International Joint Conference on Natural
404

Language Processing (Volume 1: Long Papers), pages 4921–4933, Online, August 2021.
405

Association for Computational Linguistics.
doi: 10.18653/v1/2021.acl-long.381.
URL
406

https://aclanthology.org/2021.acl-long.381.
407

[12] Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume
408

Smith, and Brian Thorne. Private federated learning on vertically partitioned data via entity
409

resolution and additively homomorphic encryption, 2017.
410

[13] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced
411

bert with disentangled attention. In International Conference on Learning Representations,
412

2021. URL https://openreview.net/forum?id=XPZIaotutsD.
413

10


---Page Break---
[14] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe,
414

Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning
415

for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th
416

International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning
417

Research, pages 2790–2799. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.
418

press/v97/houlsby19a.html.
419

[15] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
420

Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In
421

International Conference on Learning Representations, 2022. URL https://openreview.
422

net/forum?id=nZeVKeeFYf9.
423

[16] Hugging Face. AutoTrain — huggingface.co. https://huggingface.co/autotrain, 2023.
424

[Accessed 28-09-2023].
425

[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
426

arXiv:1412.6980, 2014.
427

[18] Oscar Li, Jiankai Sun, Xin Yang, Weihao Gao, Hongyi Zhang, Junyuan Xie, Virginia Smith,
428

and Chong Wang. Label leakage and protection in two-party split learning. In International
429

Conference on Learning Representations, 2022. URL https://openreview.net/forum?
430

id=cOtBRgsf2fO.
431

[19] Qinbin Li, Zeyi Wen, Zhaomin Wu, Sixu Hu, Naibo Wang, Yuan Li, Xu Liu, and Bingsheng He.
432

A survey on federated learning systems: Vision, hype and reality for data privacy and protection.
433

IEEE Transactions on Knowledge and Data Engineering, 2021.
434

[20] Shen Li, Pritam Damania, Luca Wehrstedt, and Rohan Varma. PyTorch RPC: Distributed Deep
435

Learning Built on Tensor-Optimized Remote Procedure Calls. In Proceedings of Machine
436

Learning and Systems 5 (MLSys), 2023.
437

[21] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In
438

Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the
439

11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers),
440

pages 4582–4597, Online, August 2021. Association for Computational Linguistics. doi:
441

10.18653/v1/2021.acl-long.353. URL https://aclanthology.org/2021.acl-long.353.
442

[22] Yansong Li, Zhixing Tan, and Yang Liu. Privacy-preserving prompt tuning for large language
443

model services. ArXiv, abs/2305.06212, 2023. URL https://api.semanticscholar.org/
444

CorpusID:258588141.
445

[23] Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, and Meikang Qiu.
446

Differentially private low-rank adaptation of large language model using federated learning.
447

arXiv preprint arXiv:2312.17493, 2023.
448

[24] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Ar-
449

cas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Aarti
450

Singh and Jerry Zhu, editors, Proceedings of the 20th International Conference on Artifi-
451

cial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research,
452

pages 1273–1282. PMLR, 20–22 Apr 2017. URL https://proceedings.mlr.press/v54/
453

mcmahan17a.html.
454

[25] Nvidia. Nvidia confidential computing. https://www.nvidia.com/en-us/data-center/
455

solutions/confidential-computing, 2023. [Accessed 28-09-2023].
456

[26] OctoAI. Fine-tuning Stable Diffusion — docs.octoai.cloud. https://docs.octoai.cloud/
457

docs/fine-tuning-stable-diffusion, 2023. [Accessed 28-09-2023].
458

[27] OpenAI.
OpenAI Platform — platform.openai.com.
https://platform.openai.com/
459

docs/guides/fine-tuning, 2023. [Accessed 28-09-2023].
460

11


---Page Break---
[28] Dario Pasquini, Giuseppe Ateniese, and Massimo Bernaschi. Unleashing the tiger: Inference
461

attacks on split learning. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and
462

Communications Security, CCS ’21, page 2113–2129, New York, NY, USA, 2021. Association
463

for Computing Machinery. ISBN 9781450384544. doi: 10.1145/3460120.3485259. URL
464

https://doi.org/10.1145/3460120.3485259.
465

[29] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych.
466

Adapterfusion: Non-destructive task composition for transfer learning, 2021.
467

[30] Yuma Rao, Jacob Steeves, Ala Shaabana, Daniel Attevelt, and Matthew McAteer. Bittensor: A
468

peer-to-peer intelligence market, 2021.
469

[31] Weiyan Shi, Ryan Shea, Si Chen, Chiyuan Zhang, Ruoxi Jia, and Zhou Yu. Just fine-tune twice:
470

Selective differential privacy for large language models. arXiv preprint arXiv:2204.07667,
471

2022.
472

[32] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew
473

Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a
474

sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural
475

Language Processing, pages 1631–1642, Seattle, Washington, USA, October 2013. Association
476

for Computational Linguistics. URL https://www.aclweb.org/anthology/D13-1170.
477

[33] Jiankai Sun, Xin Yang, Yuanshun Yao, and Chong Wang. Label leakage and protection from
478

forward embedding in vertical federated learning. arXiv preprint arXiv:2203.01451, 2022.
479

[34] Chuanqi Tan, Fuchun Sun, Tao Kong, Wenchang Zhang, Chao Yang, and Chunfang Liu. A
480

survey on deep transfer learning. In Vˇera K˚urková, Yannis Manolopoulos, Barbara Hammer,
481

Lazaros Iliadis, and Ilias Maglogiannis, editors, Artificial Neural Networks and Machine
482

Learning – ICANN 2018, pages 270–279, Cham, 2018. Springer International Publishing. ISBN
483

978-3-030-01424-7.
484

[35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
485

Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
486

foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
487

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
488

Gomez, Ł ukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
In I. Guyon,
489

U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, ed-
490

itors, Advances in Neural Information Processing Systems, volume 30. Curran Associates,
491

Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/
492

3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.
493

[37] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh Raskar. Split learning for
494

health: Distributed deep learning without sharing raw patient data, 2018.
495

[38] Praneeth Vepakomma, Otkrist Gupta, Abhimanyu Dubey, and Ramesh Raskar. Reducing
496

leakage in distributed deep learning for sensitive health data. 05 2019.
497

[39] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.
498

Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv
499

preprint arXiv:1804.07461, 2018.
500

[40] Yiming Wang, Yu Lin, Xiaodong Zeng, and Guannan Zhang. Privatelora for efficient privacy
501

preserving llm. arXiv preprint arXiv:2311.14030, 2023.
502

[41] Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus
503

for sentence understanding through inference. arXiv preprint arXiv:1704.05426, 2017.
504

[42] Guangxuan Xiao, Ji Lin, and Song Han. Offsite-tuning: Transfer learning without full model.
505

arXiv preprint arXiv:2302.04870, 2023.
506

[43] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. Federated machine learning: Concept
507

and applications. ACM Trans. Intell. Syst. Technol., 10(2), jan 2019. ISSN 2157-6904. doi:
508

10.1145/3298981. URL https://doi.org/10.1145/3298981.
509

12


---Page Break---
[44] Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam Kamath,
510

Janardhan Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, Sergey Yekhanin, and
511

Huishuai Zhang. Differentially private fine-tuning of language models. In International
512

Conference on Learning Representations, 2022. URL https://openreview.net/forum?
513

id=Q42f0dfjECO.
514

[45] Zhuo Zhang, Yuanhang Yang, Yong Dai, Qifan Wang, Yue Yu, Lizhen Qu, and Zenglin
515

Xu. FedPETuning: When federated learning meets the parameter-efficient tuning methods of
516

pre-trained language models. In Findings of the Association for Computational Linguistics:
517

ACL 2023, pages 9963–9977, Toronto, Canada, July 2023. Association for Computational
518

Linguistics. doi: 10.18653/v1/2023.findings-acl.632. URL https://aclanthology.org/
519

2023.findings-acl.632.
520

[46] Haodong Zhao, Wei Du, Fangqi Li, Peixuan Li, and Gongshen Liu. Fedprompt: Communication-
521

efficient and privacy preserving prompt tuning in federated learning, 2023.
522

A
Hyperparameters search
523

In P3EFT and Distance Correlation methods resulting loss L function can be viewed in the form
524

L = Lm + α · Lr,

where Lm - main task loss, Lr - regularizer and α is a coefficient that controls the tradeoff between
525

these two losses. The selection of this coefficient affects the final performance of the model. Therefore,
526

to find the best configurations for both methods, we iterated through this hyperparameter using a grid
527

search.
528

We started with α = 1 and then altered it with a multiplicative step of 10
1
2 . Values were discarded if
529

the quality did not exceed that achieved by solely training the classifier without LoRA. This criterion
530

was adopted because such outcomes would suggest the method’s inability to outperform training
531

scenarios in which the server does not engage with the labels whatsoever. Additionally, we excluded
532

values that led to unstable training. By this, we mean instances where, although the model initially
533

trained on the primary task, at some point, the regularizer began contributing significantly more,
534

and the utility value dropped to the starting value. We observed this issue for the DC method with
535

DeBERTa on the MNLI. From the remaining values, we aimed to choose the one that offered the
536

lowest privacy leakage. The final hyperparameter values for P3EFT can be found in the Table 4 and
537

for DC in the Table 5.
538

Table 4: Regularization parameter α for the P3EFT method. The values in the table represent powers
of the 10
1
2 .

SST2
QNLI
MNLI

DeBERTa XXLarge
1
1
1

Flan-T5-Large
-1
1
1

LLaMA-2 7B
0
0
—

Table 5: Regularization parameter α for the DC method. The values in the table represent powers of
the 10
1
2 .

SST2
QNLI
MNLI

DeBERTa XXLarge
0
-1
—

Flan-T5-Large
2
-1
0

LLaMA-2 7B
-1
-1
—

13


---Page Break---
B
Formal algorithm definition
539

Below, we define the full P3EFT algorithm. In Algorithm 2, main_loss is the task-specific objective
540

e.g. cross-entropy; reg_loss is the adversarial regularizer described in Section 3.4. We denote
541

client-side model "head" as f(h, ψt), where ψ represent trainable head parameters. Finally, opt_step
542

function performs a single gradient descent step with a task-specific optimizer, typically Adam [17].
543

Algorithm 2 P3EFT - full training algorithm

1: Input: dataset D = {X, Y }, n > 1 number of adapters, α ≥0 - regularizing weight, m > 1
number of obfuscated gradients
2: Initialize head ψ0, mixing weights Wi and adapters θ0
i , i = 1, n
3: for t = 0, 1, . . . , T −1 do
4:
Sample batch {xt, yt}
5:
for i = 1, . . . , n do
6:
ht
i = h(xt, θt
i)
▷by server
7:
li = reg_loss(ht
i, yt)
▷by client
8:
end for
9:
h′ = Pn
i=1 Wi ⊙ht
i
10:
l = main_loss(f(h′, ψt), yt)
11:
L = l + α · Pn
i=1 li
12:
for i = 1, . . . , n do
13:
gh = ∂L/∂ht
i
▷Client performs partial backprop
14:
gt
i = private_backprop(x, θt
i, gh, m)
15:
θt+1
i
= opt_step(θt
i, gt
i, t)
16:
end for
17:
ψt+1 = opt_step(ψt, ∂l/∂ψt, t)
18: end for
19: Return: ψT , θT
1 , . . . , θT
M

C
Informal description of LoRA fine-tuning
544

For convenience, we provide a brief summary of fine-tuning with LoRA [15]. This PEFT method
545

was originally designed for fine-tuning large pre-trained language models on downstream NLP tasks.
546

These language models are typically based on the Transformer architecture [36], where most trainable
547

parameters are allocated to linear layers in multi-head self-attention and feedforward blocks.
548

In the first stage of LoRA fine-tuning, user augments the model with adapters. To do so, a user goes
549

over linear layers in transformer blocks and adds two trainable matrices, A and B that affect this
550

layer’s forward pass. Let Wi × x + bi be the original layer with n inputs and m hidden units. Here,
551

Wi ∈Rm×n is a pre-trained weight matrix, bi ∈Rm is a pre-trained intercept vector and x ∈Rn
552

represents a vector of inputs to this particular layer. During the forward pass, a layer with LoRA
553

adapter computes Wi × x + bi + Bi × Ai × x, or equivalently, (Wi + B × A) × x + bi. Here, Ai
554

and Bi are two newly added matrices that constitute a LoRA adapter.
555

The adapter matrices A ∈Rr×n and B ∈Rm×r have a very small intermediate dimension r. For
556

instance, when training GPT-3 with LoRA adapters, authors use 1 ≤r ≤64, whereas the main
557

weight dimensions are m = n = 12288. The first matrix A is initialized with small random normal
558

values, and the second matrix B is initialized at zeros. That way, initial A and B do not affect the
559

model predictions.
560

Once all adapters are initilized, the user trains all Ai and Bi matrices of the model, while keeping
561

the rest of the weights frozen. This way, only a small faction (less than 1%) of model weights are
562

updated. Once the training is over, the learned adapters Ai and Bi can be merged into the main
563

weights (Wi := Wi + Ai × Bi) or used separately.
564

LoRA adapters are designed with two objectives in mind: i) to allow fine-tuning models in limited
565

GPU memory and ii) to allow inferencing many fine-tuned models using one inference server. When
566

fine-tuning, LoRA achieves small memory footprint due to the fact that user does not need to compute
567

14


---Page Break---
gradients (or optimizer statistics) for billions of main model parameters. During inference, a server
568

can keep a library of several adapters for different tasks and swap between them on demand.
569

D
Informal description of LoRA fine-tuning
570

We used NVIDIA A100 GPUs for all the experiments. Experiments with DeBERTA [13] and Flan-T5
571

[4] on SST2 [32] were conducted on the single GPU, while experiments on MNLI [41] and QNLI
572

require 4 A100. LLaMA-2 [35] expetiments were carried out on the node of 8 A100.
573

All the experiments last 12-24 hours. However, it is possible to speed up some of them using more
574

GPUs, as well as conduct them on a smaller number of GPUs using technics to save GPU memory
575

(see parameters in our code).
576

15


---Page Break---
NeurIPS Paper Checklist
577

1. Claims
578

Question: Do the main claims made in the abstract and introduction accurately reflect the
579

paper’s contributions and scope?
580

Answer: [Yes]
581

Justification: See our main contributions in 1 and Section 5.
582

Guidelines:
583

• The answer NA means that the abstract and introduction do not include the claims
584

made in the paper.
585

• The abstract and/or introduction should clearly state the claims made, including the
586

contributions made in the paper and important assumptions and limitations. A No or
587

NA answer to this question will not be perceived well by the reviewers.
588

• The claims made should match theoretical and experimental results, and reflect how
589

much the results can be expected to generalize to other settings.
590

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
591

are not attained by the paper.
592

2. Limitations
593

Question: Does the paper discuss the limitations of the work performed by the authors?
594

Answer: [Yes]
595

Justification: We discuss potential limitations of the method in Section 1, Section 3 and
596

Section 5.
597

Guidelines:
598

• The answer NA means that the paper has no limitation while the answer No means that
599

the paper has limitations, but those are not discussed in the paper.
600

• The authors are encouraged to create a separate "Limitations" section in their paper.
601

• The paper should point out any strong assumptions and how robust the results are to
602

violations of these assumptions (e.g., independence assumptions, noiseless settings,
603

model well-specification, asymptotic approximations only holding locally). The authors
604

should reflect on how these assumptions might be violated in practice and what the
605

implications would be.
606

• The authors should reflect on the scope of the claims made, e.g., if the approach was
607

only tested on a few datasets or with a few runs. In general, empirical results often
608

depend on implicit assumptions, which should be articulated.
609

• The authors should reflect on the factors that influence the performance of the approach.
610

For example, a facial recognition algorithm may perform poorly when image resolution
611

is low or images are taken in low lighting. Or a speech-to-text system might not be
612

used reliably to provide closed captions for online lectures because it fails to handle
613

technical jargon.
614

• The authors should discuss the computational efficiency of the proposed algorithms
615

and how they scale with dataset size.
616

• If applicable, the authors should discuss possible limitations of their approach to
617

address problems of privacy and fairness.
618

• While the authors might fear that complete honesty about limitations might be used by
619

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
620

limitations that aren’t acknowledged in the paper. The authors should use their best
621

judgment and recognize that individual actions in favor of transparency play an impor-
622

tant role in developing norms that preserve the integrity of the community. Reviewers
623

will be specifically instructed to not penalize honesty concerning limitations.
624

3. Theory Assumptions and Proofs
625

Question: For each theoretical result, does the paper provide the full set of assumptions and
626

a complete (and correct) proof?
627

Answer: [NA]
628

16


---Page Break---
Justification: We include no proofs.
629

Guidelines:
630

• The answer NA means that the paper does not include theoretical results.
631

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
632

referenced.
633

• All assumptions should be clearly stated or referenced in the statement of any theorems.
634

• The proofs can either appear in the main paper or the supplemental material, but if
635

they appear in the supplemental material, the authors are encouraged to provide a short
636

proof sketch to provide intuition.
637

• Inversely, any informal proof provided in the core of the paper should be complemented
638

by formal proofs provided in appendix or supplemental material.
639

• Theorems and Lemmas that the proof relies upon should be properly referenced.
640

4. Experimental Result Reproducibility
641

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
642

perimental results of the paper to the extent that it affects the main claims and/or conclusions
643

of the paper (regardless of whether the code and data are provided or not)?
644

Answer: [Yes]
645

Justification: We describe all the technical details in the Section 4 and in the README of
646

our attached code.
647

Guidelines:
648

• The answer NA means that the paper does not include experiments.
649

• If the paper includes experiments, a No answer to this question will not be perceived
650

well by the reviewers: Making the paper reproducible is important, regardless of
651

whether the code and data are provided or not.
652

• If the contribution is a dataset and/or model, the authors should describe the steps taken
653

to make their results reproducible or verifiable.
654

• Depending on the contribution, reproducibility can be accomplished in various ways.
655

For example, if the contribution is a novel architecture, describing the architecture fully
656

might suffice, or if the contribution is a specific model and empirical evaluation, it may
657

be necessary to either make it possible for others to replicate the model with the same
658

dataset, or provide access to the model. In general. releasing code and data is often
659

one good way to accomplish this, but reproducibility can also be provided via detailed
660

instructions for how to replicate the results, access to a hosted model (e.g., in the case
661

of a large language model), releasing of a model checkpoint, or other means that are
662

appropriate to the research performed.
663

• While NeurIPS does not require releasing code, the conference does require all submis-
664

sions to provide some reasonable avenue for reproducibility, which may depend on the
665

nature of the contribution. For example
666

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
667

to reproduce that algorithm.
668

(b) If the contribution is primarily a new model architecture, the paper should describe
669

the architecture clearly and fully.
670

(c) If the contribution is a new model (e.g., a large language model), then there should
671

either be a way to access this model for reproducing the results or a way to reproduce
672

the model (e.g., with an open-source dataset or instructions for how to construct
673

the dataset).
674

(d) We recognize that reproducibility may be tricky in some cases, in which case
675

authors are welcome to describe the particular way they provide for reproducibility.
676

In the case of closed-source models, it may be that access to the model is limited in
677

some way (e.g., to registered users), but it should be possible for other researchers
678

to have some path to reproducing or verifying the results.
679

5. Open access to data and code
680

Question: Does the paper provide open access to the data and code, with sufficient instruc-
681

tions to faithfully reproduce the main experimental results, as described in supplemental
682

material?
683

17


---Page Break---
Answer: [Yes]
684

Justification: We provide attached code.
685

Guidelines:
686

• The answer NA means that paper does not include experiments requiring code.
687

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
688

public/guides/CodeSubmissionPolicy) for more details.
689

• While we encourage the release of code and data, we understand that this might not be
690

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
691

including code, unless this is central to the contribution (e.g., for a new open-source
692

benchmark).
693

• The instructions should contain the exact command and environment needed to run to
694

reproduce the results. See the NeurIPS code and data submission guidelines (https:
695

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
696

• The authors should provide instructions on data access and preparation, including how
697

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
698

• The authors should provide scripts to reproduce all experimental results for the new
699

proposed method and baselines. If only a subset of experiments are reproducible, they
700

should state which ones are omitted from the script and why.
701

• At submission time, to preserve anonymity, the authors should release anonymized
702

versions (if applicable).
703

• Providing as much information as possible in supplemental material (appended to the
704

paper) is recommended, but including URLs to data and code is permitted.
705

6. Experimental Setting/Details
706

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
707

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
708

results?
709

Answer: [Yes]
710

Justification: See Section 4, Appendix A.
711

Guidelines:
712

• The answer NA means that the paper does not include experiments.
713

• The experimental setting should be presented in the core of the paper to a level of detail
714

that is necessary to appreciate the results and make sense of them.
715

• The full details can be provided either with the code, in appendix, or as supplemental
716

material.
717

7. Experiment Statistical Significance
718

Question: Does the paper report error bars suitably and correctly defined or other appropriate
719

information about the statistical significance of the experiments?
720

Answer: [Yes]
721

Justification: We report error bars in main results.
722

Guidelines:
723

• The answer NA means that the paper does not include experiments.
724

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
725

dence intervals, or statistical significance tests, at least for the experiments that support
726

the main claims of the paper.
727

• The factors of variability that the error bars are capturing should be clearly stated (for
728

example, train/test split, initialization, random drawing of some parameter, or overall
729

run with given experimental conditions).
730

• The method for calculating the error bars should be explained (closed form formula,
731

call to a library function, bootstrap, etc.)
732

• The assumptions made should be given (e.g., Normally distributed errors).
733

• It should be clear whether the error bar is the standard deviation or the standard error
734

of the mean.
735

18


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
736

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
737

of Normality of errors is not verified.
738

• For asymmetric distributions, the authors should be careful not to show in tables or
739

figures symmetric error bars that would yield results that are out of range (e.g. negative
740

error rates).
741

• If error bars are reported in tables or plots, The authors should explain in the text how
742

they were calculated and reference the corresponding figures or tables in the text.
743

8. Experiments Compute Resources
744

Question: For each experiment, does the paper provide sufficient information on the com-
745

puter resources (type of compute workers, memory, time of execution) needed to reproduce
746

the experiments?
747

Answer: [Yes]
748

Justification: See Appendix D.
749

Guidelines:
750

• The answer NA means that the paper does not include experiments.
751

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
752

or cloud provider, including relevant memory and storage.
753

• The paper should provide the amount of compute required for each of the individual
754

experimental runs as well as estimate the total compute.
755

• The paper should disclose whether the full research project required more compute
756

than the experiments reported in the paper (e.g., preliminary or failed experiments that
757

didn’t make it into the paper).
758

9. Code Of Ethics
759

Question: Does the research conducted in the paper conform, in every respect, with the
760

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
761

Answer: [Yes]
762

Justification: We confirm the NeurIPS Code of Ethics
763

Guidelines:
764

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
765

• If the authors answer No, they should explain the special circumstances that require a
766

deviation from the Code of Ethics.
767

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
768

eration due to laws or regulations in their jurisdiction).
769

10. Broader Impacts
770

Question: Does the paper discuss both potential positive societal impacts and negative
771

societal impacts of the work performed?
772

Answer: [Yes]
773

Justification: We describe a potential social impact in Introduction.
774

Guidelines:
775

• The answer NA means that there is no societal impact of the work performed.
776

• If the authors answer NA or No, they should explain why their work has no societal
777

impact or why the paper does not address societal impact.
778

• Examples of negative societal impacts include potential malicious or unintended uses
779

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
780

(e.g., deployment of technologies that could make decisions that unfairly impact specific
781

groups), privacy considerations, and security considerations.
782

• The conference expects that many papers will be foundational research and not tied
783

to particular applications, let alone deployments. However, if there is a direct path to
784

any negative applications, the authors should point it out. For example, it is legitimate
785

to point out that an improvement in the quality of generative models could be used to
786

19


---Page Break---
generate deepfakes for disinformation. On the other hand, it is not needed to point out
787

that a generic algorithm for optimizing neural networks could enable people to train
788

models that generate Deepfakes faster.
789

• The authors should consider possible harms that could arise when the technology is
790

being used as intended and functioning correctly, harms that could arise when the
791

technology is being used as intended but gives incorrect results, and harms following
792

from (intentional or unintentional) misuse of the technology.
793

• If there are negative societal impacts, the authors could also discuss possible mitigation
794

strategies (e.g., gated release of models, providing defenses in addition to attacks,
795

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
796

feedback over time, improving the efficiency and accessibility of ML).
797

11. Safeguards
798

Question: Does the paper describe safeguards that have been put in place for responsible
799

release of data or models that have a high risk for misuse (e.g., pretrained language models,
800

image generators, or scraped datasets)?
801

Answer: [No]
802

Justification: We do not describe the safeguards
803

Guidelines:
804

• The answer NA means that the paper poses no such risks.
805

• Released models that have a high risk for misuse or dual-use should be released with
806

necessary safeguards to allow for controlled use of the model, for example by requiring
807

that users adhere to usage guidelines or restrictions to access the model or implementing
808

safety filters.
809

• Datasets that have been scraped from the Internet could pose safety risks. The authors
810

should describe how they avoided releasing unsafe images.
811

• We recognize that providing effective safeguards is challenging, and many papers do
812

not require this, but we encourage authors to take this into account and make a best
813

faith effort.
814

12. Licenses for existing assets
815

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
816

the paper, properly credited and are the license and terms of use explicitly mentioned and
817

properly respected?
818

Answer: [Yes]
819

Justification: We use open datasets from GLUE benchmark and open-sourced models and
820

cite them in our work.
821

Guidelines:
822

• The answer NA means that the paper does not use existing assets.
823

• The authors should cite the original paper that produced the code package or dataset.
824

• The authors should state which version of the asset is used and, if possible, include a
825

URL.
826

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
827

• For scraped data from a particular source (e.g., website), the copyright and terms of
828

service of that source should be provided.
829

• If assets are released, the license, copyright information, and terms of use in the
830

package should be provided. For popular datasets, paperswithcode.com/datasets
831

has curated licenses for some datasets. Their licensing guide can help determine the
832

license of a dataset.
833

• For existing datasets that are re-packaged, both the original license and the license of
834

the derived asset (if it has changed) should be provided.
835

• If this information is not available online, the authors are encouraged to reach out to
836

the asset’s creators.
837

13. New Assets
838

20


---Page Break---
Question: Are new assets introduced in the paper well documented and is the documentation
839

provided alongside the assets?
840

Answer: [NA]
841

Justification: We do not release any of the assets
842

Guidelines:
843

• The answer NA means that the paper does not release new assets.
844

• Researchers should communicate the details of the dataset/code/model as part of their
845

submissions via structured templates. This includes details about training, license,
846

limitations, etc.
847

• The paper should discuss whether and how consent was obtained from people whose
848

asset is used.
849

• At submission time, remember to anonymize your assets (if applicable). You can either
850

create an anonymized URL or include an anonymized zip file.
851

14. Crowdsourcing and Research with Human Subjects
852

Question: For crowdsourcing experiments and research with human subjects, does the paper
853

include the full text of instructions given to participants and screenshots, if applicable, as
854

well as details about compensation (if any)?
855

Answer: [NA]
856

Justification: We do not involve crowdsourcing.
857

Guidelines:
858

• The answer NA means that the paper does not involve crowdsourcing nor research with
859

human subjects.
860

• Including this information in the supplemental material is fine, but if the main contribu-
861

tion of the paper involves human subjects, then as much detail as possible should be
862

included in the main paper.
863

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
864

or other labor should be paid at least the minimum wage in the country of the data
865

collector.
866

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
867

Subjects
868

Question: Does the paper describe potential risks incurred by study participants, whether
869

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
870

approvals (or an equivalent approval/review based on the requirements of your country or
871

institution) were obtained?
872

Answer: [NA]
873

Justification: Our paper does not involve research with human subjects
874

Guidelines:
875

• The answer NA means that the paper does not involve crowdsourcing nor research with
876

human subjects.
877

• Depending on the country in which research is conducted, IRB approval (or equivalent)
878

may be required for any human subjects research. If you obtained IRB approval, you
879

should clearly state this in the paper.
880

• We recognize that the procedures for this may vary significantly between institutions
881

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
882

guidelines for their institution.
883

• For initial submissions, do not include any information that would break anonymity (if
884

applicable), such as the institution conducting the review.
885

21


---Page Break---
