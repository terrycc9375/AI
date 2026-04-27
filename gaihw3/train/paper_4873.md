Optimal, Efficient and Practical Algorithms for
Assortment Optimization

Anonymous Author(s)
Affiliation
Address
email

Abstract

We address the problem of active online assortment optimization problem
1

with preference feedback, which is a framework for modeling user choices
2

and subsetwise utility maximization. The framework is useful in various
3

real-world applications including ad placement, online retail, recommender
4

systems, and fine-tuning language models, amongst many others. The prob-
5

lem, although has been studied in the past, lacks an intuitive and practical
6

solution approach with simultaneously efficient algorithm and optimal re-
7

gret guarantee. E.g., popularly used assortment selection algorithms often
8

require the presence of a ‘strong reference’ which is always included in the
9

choice sets, further they are also designed to offer the same assortments
10

repeatedly until the reference item gets selected—all such requirements
11

are quite unrealistic for practical applications. In this paper, we designed
12

efficient algorithms for the problem of regret minimization in assortment
13

selection with Plackett Luce (PL) based user choices. We designed a novel
14

concentration guarantee for estimating the score parameters of the PL model
15

using ‘Pairwise Rank-Breaking’, which builds the foundation of our proposed
16

algorithms. Moreover, our methods are practical, provably optimal, and
17

devoid of the aforementioned limitations of the existing methods. Empirical
18

evaluations corroborate our findings and outperform the existing baselines.
19

1
Introduction
20

Studies have shown that it is often easier, faster and less expensive to collect feedback on a
21

relative scale rather than asking ratings on an absolute scale. E.g., to understand the liking
22

for a given pair of items, say (A,B), it is easier for the users to answer preference-based
23

queries like: “Do you prefer Item A over B?", rather than their absolute counterparts: “How
24

much do you score items A and B in a scale of [0-10]?". Due to the widespread applicability
25

and ease of data collection with relative feedback, learning from preferences has gained much
26

popularity in the machine-learning community, especially the active learning literature which
27

has applications in Medical surveys, AI tutoring systems, Multi-player sports/games, or any
28

real-world systems that have ways to collect feedback in terms of preferences. The problem
29

is famously studied as the Dueling-Bandit (DB) problem in the active learning community
30

[41, 3, 45, 46, 44], which is an online learning framework for identifying a set of ‘good’ items
31

from a fixed decision-space (set of items) by querying preference feedback of actively chosen
32

item-pairs. Consequently, the generalization of Dueling-Bandits, with subset-wise preferences
33

has also been developed into an active field of research. For instance, applications like
34

Web search (e.g. Google, Bing, or even in some versions of ChatGPT), online shopping
35

(Amazon, App stores, Google Flights), recommender systems (e.g. Youtube, Netflix, Google
36

News/Maps, Spotify) typically involve users expressing preferences by choosing one result (or
37

a handful of results) from a subset of offered items and often the objective of the system is to
38

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not
distribute.


---Page Break---
identify the ‘most-profitable’ subset to offer to their users. The problem, popularly termed
39

as ‘Assortment Optimization’ is studied in many interdisciplinary literature, e.g. Online
40

learning and bandits [10], Operations research [40, 2], Game theory [15], RLHF [20, 30], to
41

name a few.
42

Problem (Informal): Active Optimal Assortment (AOA) Active Assortment Opti-
43

mization (a.k.a. Utility Maximization with Subset Choices) [13, 2, 23, 22] is an active
44

learning framework for finding the ‘optimal’ profit-maximizing subset. Formally, assume
45

we have a decision set of [K] := {1, 2, . . . K} of K items, with each item being associated
46

with the score (or utility) parameters θ := (θ1, θ2, . . . , θK) (without loss of generality assume
47

θ1 ≥θ2 ≥. . . ≥θK ≥0). At each round t = 1, 2, . . ., the learner or the algorithm gets to
48

query an assortment (typically subsets containing up to m-items) St ⊆[K], upon which
49

it gets to see some (noisy) relative preferences across the items in St, typically generated
50

according to an underlying Plackett-Luce (PL) choice model with parameters θ (1). Further,
51

to allow the event where no items are selected, we also model a No-Choice (NC) item, indexed
52

by item-0, with PL parameter θ0 ∈R+.
53

(Objective 1.) Top-m: identify the top-m item-set: {θ1, . . . , θm}, for some m ∈[1, K].
54

(Objective 2.) Wtd-Top-m: A more general objective could also consider a weight (or
55

price) ri ∈R+ associated with the item i ∈[K], and the goal could be to identify the
56

assortment (subset) with maximum weighted utility 1, as detailed in Sec. 2.
57

Related Works and Limitations:
As stated above, the problem of AOA is fundamental
58

in many practical scenarios, and thus widely studied in multiple research areas, including
59

Online ML/learning theory and operations research.
60

• In the Online ML literature, the problem is well-studied as Multi-Dueling Bandits [39, 14],
61

or Battling Bandits [35, 34, 11], which is an extension of the famous Dueling Bandit problem
62

[46, 45]. The main limitation of this line of work is the lack of practical objectives, which either
63

aim to identify the ‘best-item’ 1(= arg maxi∈[K] θi) within a PAC (probably approximately
64

correct) framework [36, 16, 17, 31] or quantifying regret against the best items [35, 12]. Note
65

the latter actually leads to the optimal subset choice of repeatedly selecting the optimal item,
66

arg maxi θi, m times, i.e. (1, 1, . . . 1), which is unrealistic from the viewpoint of real-world
67

system design. Selecting an assortment of distinct top-m items (Top-m-AOA) or maximum
68

expected utility (Wtd-Top-m-AOA) makes more sense.
69

• On the other hand, a similar line of the problem has been studied in operations research
70

and dynamic assortment selection literature, where the goal is to offer a subset of items to
71

the customers in order to maximize expected revenue. The problem has been studied under
72

different user choice models, e.g. PL or Multinomial-Logit models [2], Mallows and mixture of
73

Mallows [22], Markov chain-based choice models [23], single transition model [27] etc. While
74

these works indeed consider a more practical objective of finding the best assortment (subset)
75

with the highest expected utility for a regret minimization objective, (1) a major drawback
76

in their approach lies in the algorithm design which requires to keep on querying the same set
77

multiple times, e.g. [2, 29, 18, 1]. Such design techniques could be impractical to be deployed
78

in real systems where users could easily get annoyed if the same items are shown again and
79

again. For example, in ad-placement, music/movies/news/tweets/reels recommendations,
80

offering the same assortment could increase user dissatisfaction and disengagement.
81

(2) The second major drawback of this line of work lies in the structural assumption of
82

their underlying choice models which requires the existence of a reference/default item, that
83

needs to be part of every assortment St. This leads to assuming a No-Choice item, typically
84

denoted as item-0, which is a default choice of any assortment St. Further a stronger and
85

more unrealistic assumption lies in the fact that they require to assume that the above pivot
86

is stronger than the rest of the K items, i.e. θ0 ≥maxi∈[K] θi, i.e. the No-Choice (NC)
87

action is the most likely outcome of any assortment St. This is often unrealistic, e.g., during
88

user interactions with language models, or online shopping, or Route recommendation in
89

GPS navigation, a NC action is highly improbable. Consequently, such assumption limits the
90

use in real-systems. In the existing literature [2, 28, 1, 24], such assumptions are primarily
91

1This is equivalent to finding the set with maximum expected revenue when ris represents the
price of item i [2]

2


---Page Break---
adapted solely for theoretical needs, precisely for maintaining concentration bounds of the
92

PL parameters θ, and hence not well justified from a practical viewpoint. Some recent
93

developments also generalized the AOA problem to linear MNL scores to incorporate large
94

actions embedded in d-dimension [43, 42, 28], however, their approaches are either limited
95

to the above restrictions or suffer sub-optimal regret guarantees without those assumptions
96

(e.g. the regret bound of [28] is O(d3/2√

T) which is suboptimal by a d-factor). Considering
97

the above limitations of the AOA literature, we set to answer two questions:
98

(1) Can we consider a general AOA model where the default item, like the NC item defined
99

above, is not necessarily the strongest one, i.e. θ0 ≥maxi∈[K] θi?
100

(2) Can we design a practical and regret optimal algorithm for the AOA framework, without
101

needing to play the same repetitive actions and yet converge to the optimal assortment?
102

Contributions
We answer these questions in the affirmative and present best of all
103

scenarios. We design practical algorithms on practical AOA framework with practical
104

objectives–Unlike the existing approaches of the AOA, literature [2, 18], we do not have to
105

keep playing the same assortment multiple times, neither require a strongest default item
106

(like NC satisfying θ0 ≥maxi∈[K] θi). Moreover, our objectives do not require us to converge
107

to a multiset of replicated arms like (1, 1, . . . 1), but converge to the utility-maximizing set of
108

distinct items. We list our contributions below:
109

1. A General AOA Setup: We work with a general problem of AOA for PL model,
110

which requires no additional structural assumption of the θ parameters such as θ0 ≥maxi θi,
111

unlike the existing works. We designed algorithms for two separate objectives Top-m and
112

Wtd-Top-m as discussed above (Sec. 2).
113

2. Practical, Efficient and Optimal Algorithm:
In Sec. 3, we give a practical,
114

efficient and optimal algorithm for MNL Assortment (up to log factors and the magnitude of
115

θmax). The regret bound of our algorithm AOA-RBPL (Alg. 1) yields ˜O(
√

KT) regret for
116

both Top-m and Wtd-Top-m objective. Our algorithms use a novel parameter estimation
117

technique for discrete choice models based on the concept of Rank-Breaking (RB) which is
118

one of our key contributions towards designing the efficient and optimal algorithm. This
119

enables our algorithm to perform optimally without requiring the No-Choice item to be
120

the strongest. Appendix A details the key concept of our parameter estimation technique
121

exploiting the concept of RB. Our resulting algorithm plays optimistically based on the UCB
122

estimates of PL parameters and does not require repeating the same subset multiple times,
123

justifying our title.
124

3. Improvement with Adaptive Pivots:
In Sec. 4, we refine the performance of
125

our algorithm by employing the novel idea of ‘adaptive pivots’ (a reference item) and
126

proposed AOA-RBPL-Adaptive. Performance-wise this removes the asymptotic dependence
127

on θmax = maxi θi/θ0 in the regret analysis. This enables the algorithm to work effectively
128

in scenarios where the No-Choice item is less likely to be selected, i.e., θmax ≫1. This
129

leads to a huge improvement in our experiments, especially in the range of low θ0, where
130

AOA-RBPL-Adaptive drastically outperforms over the existing baseline. Comparison of our
131

regret bound with existing work is detailed in Table 1.
132

4. Emperical Analysis.
Finally, we corroborate our theoretical results with empirical
133

evaluations (Sec. 5), which certify our superior performance in the general AOA setups.
134

Work
Framework
Assume θ0 = θmax = 1
Regret
Our (Alg. 1)
MNL model (Obj. 2)
No

p

min{θmax, K}KT log T
[2] (Thm 1)
MNL model (Obj. 2)
Yes
√KT log T
[2] (Thm 4)
MNL model (Obj. 2)
No
√θmaxKT log T
[1]
MNL model (Obj. 2)
Yes

p

KT log(mT) + K log2(mT)

[24]
MNL model with
No

q

KT
mini ri log T
constraints (Obj. 2)

Table 1: Our Contribution vs the Existing Results in the K-armed MNL-Assortment literature

3


---Page Break---
It is also worth mentioning that our proposed algorithm and their respective regret analysis
135

could be extended to any general random utility (RUM) based preference models [38, 37],
136

as explained in Rem. 1. However, to keep the focus on the AOA problem and ease the
137

presentation, we stick to the special case of MNL choice model based preferences.
138

2
Problem Setup
139

We write [n] = {1, 2, ..., n} and 1{·} denotes the indicator function. The symbol ≲, employed
140

in the proof sketches, represents a coarse inequality.
141

We consider the sequential decision-making problem of Active Optimal Assortment (AOA),
142

with preference/choice feedback. Formally, the learner is given [K], a finite set of K items
143

(K > 2). At each decision round t = 1, 2, . . ., the learner selects a subset St ⊆[K] of up to
144

m items, and receives some (stochastic) feedback about the item preferences of St, drawn
145

according to some unknown underlying Plackett-Luce (PL) choice model (1) with parameters
146

θ = (θ1, θ2, . . . , θK) ∈RK
+ . We assume θ1 ≥θ2 ≥. . . ≥θK without loss of generality. An
147

interested reader may check App. A.1 for a detailed discussion on PL models. Given any
148

assortment St we also consider the possibility of ‘no-selection’ of any items given an St.
149

Following the literature of [2], we model this mathematically as a No-Choice (NC) item,
150

indexed by item-0, and its corresponding PL utility parameter θ0. Unlike most existing
151

literature on assortment selection, we are not assuming θ0 ̸≥maxi∈[K] θi. Further, since the
152

PL model is scale independent, we set θ0 = 1 and scale the rest of the PL parameters.
153

Feedback model
The feedback model formulates the information received (from the
154

‘environment’) once the learner plays a subset St ⊆[K] of at most m items. Given St we
155

consider the algorithm receives a winner feedback (or index of an item) it ∈St ∪{0}, drawn
156

according to the underlying PL choice model as:
157

P(it = i|St) = θi/
 
θ0 + P

j∈St θj

,
∀i ∈St.
(1)

We consider the following two objectives for the learner:
158

1.
Top-m-Ojective.
One simple objective could be to identify the top-m item-set:
159

{θ1, . . . , θm}, for some m ∈[1, K]. The performance of the learner can be captured by
160

minimizing the following regret:
161

Regtop
T
:=

T
X

t=1

ΘS∗−ΘSt

m
,
where
S∗:=
argmax
S⊆[K]:|S|=m

n
ΘS :=
X

i∈S
θi
o
.

2. Wtd-Top-m-Objective.
Here, each item-i is associated with a weight (for example
162

price) ri ∈R+, and the goal is to identify the set of size at most m with maximum weighted
163

utility. One could measure the regret of the learner as:
164

Regwtd
T
:=

T
X

t=1
(R(S∗, θ) −R(St, θ)), where R(S, θ) :=
X

i∈S

riθi
θ0 + P

j∈S θj
, ∀S ⊆[K],
(2)

denotes S∗:= argmaxS⊆[K]||S|≤m R(S, θ) is the optimal utility-maximizing subset. This
165

objective corresponds to the standard objective in the MNL litterature [2].
166

3
A Practical and Efficient Algorithm for AOA with PL
167

In this section, we introduce our first algorithm, which works for both objectives.
168

3.1
Algorithm Design
169

At each time t, our algorithm (Alg. 1) maintains a pairwise preference matrix bPt ∈[0, 1]n×n,
170

whose (i, j)-th entry bpij,t records the empirical probability of i having beaten j in a pairwise
171

4


---Page Break---
duel, and a corresponding upper confidence bound pucb
ij,t. Let [ ˜K] := [K] ∪{0}. We define for
172

each pair (i, j) ∈[ ˜K] × [ ˜K],
173

pucb
ij,t := bpij,t +

s

2bpij,t(1 −bpij,t)x

nij,t
+ 3x

nij,t
,
where
bpij,t := wij,t

nij,t
,
(3)

where wij,t = Pt−1
s=1 1{is = i, j ∈Ss} denotes the number of pairwise wins of item-i over j
174

and nij,t = wij,t +wji,t being the number of times (i, j) has been compared. The above UCB
175

estimates pucb
ij,t are further used to design UCB estimates of the PL parameters θi as follows
176

θucb
i,t = pucb
i0,t/(1 −pucb
i0,t)+.

The estimates θucb
i,t s are then used to select the set St, that maximizes the underlying objective.
177

This optimization problem transforms into a static assortment optimization problem with
178

upper confidence bounds θucb
i,t as the parameters, and efficient solution methods for this case
179

are available (see e.g., [7, 21, 32]).
180

Algorithm 1 AOA for PL model with RB (AOA-RBPL)

1: input: x > 0
2: init: ˜K ←K + 1, [ ˜K] = [K] ∪{0}, W1 ←[0] ˜
K× ˜
K
3: for t = 1, 2, 3, . . . , T do
4:
Set Nt = Wt + W⊤
t , and bPt = Wt

Nt . Denote Nt = [nij,t] ˜
K× ˜
K and bPt = [bpij,t] ˜
K× ˜
K.

5:
Define for all i, pucb
ii,t = 1

2 and for all i, j ∈[ ˜K], i ̸= j

pucb
ij,t = bpij,t +

2bpij,t(1−bpij,t)x

nij,t

1/2
+
3x
nij,t

6:
θucb
i,t := pucb
i0,t/(1 −pucb
i0,t)+

7:
St ←










Top-m items from argsort({θucb
1,t , . . . , θucb
K,t}),
for Top-m objective
argmaxS⊆[K]||S|≤m R(S, θucb
t
),
for Wtd-Top-m objective
8:
Play St
9:
Receive the winner it ∈[ ˜K] (drawn as per (1))
10:
Update: Wt+1 = [wij,t+1] ˜
K× ˜
K s.t. witj,t+1 ←witj,t + 1 ∀j ∈St ∪{0}
11: end for

3.2
Analysis: Concentration Lemmas
181

We start the analysis by providing two technical lemmas, whose proofs are deferred to the
182

appendix and that provide confidence bounds for the θi.
183

Lemma 1. Let T ≥1 and x > 0. Then, with probability at least 1 −3KTe−x, for all t ∈[T]
184

and i ∈[K]: θi ≤θucb
i,t
atleast one of the following two inequalities is satisfied
185

ni0,t < 69x(θ0 + θi)
or
θucb
i,t ≤θi + 4(θ0 + θi)

s

2θ0θix

ni0,t
+ 22x(θ0 + θi)2

ni0,t
.

The above lemma depends on ni0,t the number of times items i have been compared with
186

item 0 up to round t. The latter is controlled using the following lemma:
187

Lemma 2. Let T ≥1 and x > 0. Then, with probability at least 1 −KTe−x: simultaneously
188

for all t ∈[T] and i ∈[K]
189

τi,t < 2x(θ0 + ΘS∗)2 or ni0,t ≥(θ0 + θi)τi,t

2(θ0 + ΘS∗) ,
(4)

where τi,t = Pt−1
s=1 1{i ∈Ss} denotes the number of rounds item i got selected before round t.
190

5


---Page Break---
3.3
Analysis: Top-m Objective:
191

We are now ready to provide the regret upper bound for Algorithm 1 with Top-m objective.
192

Theorem 3 (Top-m Objective). Let θmax ≥1. Consider any instance of PL model on K
193

items with parameters θ ∈[0, θmax]K, θ0 = 1. The regret of Alg. 1 with parameter x = 2 log T
194

is bounded as
195

Regtop
T
= O
 
θ3/2
max
p

KT log T

when T →∞.

The above rate of ˜O(KT) is optimal (up to log-factors), as a lower bound can be derived from
196

standard multi-armed bandits [5, 6]. We only state here a sketch of the proof of Theorem 3.
197

The detailed proof is deferred to the App. B.
198

Proof Sketch of Theorem 3. Let us define for any S ⊆[K],
199

ΘS =
X

i∈S
θi,
and
Θucb
S
:=
X

i∈S
θucb
i
.

Let E be the high-probability event such that both Lemma 1 and 2 holds true. Then, P(E) ≥
200

1 −4TKe−x. Let us first assume that E holds true. Then, by Lemma 1, ΘS∗≤Θucb
S∗≤Θucb
St ,
201

which yields
202

Regtop
T
= 1

m

T
X

t=1
ΘS∗−ΘSt ≤1

m

T
X

t=1
Θucb
St −ΘSt ≲τ0 + 1

m

T
X

t=1

X

i∈St
(θucb
i,t −θi)1

τi,t ≥τ0
	
,

where τ0 = 138x(m + 1)2θ2
max corresponds to an exploration phase needed for the confidence
203

upper bounds of Lem 1 and 2 to be satisfied. Then, noting that if E holds true, we can show
204

by Lemma 2, that 1{τi,t ≥τ0} ≤1{ni0,t ≥69x(θ0 + θi)}. Therefore, we can apply Lemma 1
205

that entails,
206

1
m

T
X

t=1

X

i∈St
(θucb
i,t −θi)1

τi,t ≥¯ni0
	
≲1

m

T
X

t=1

X

i∈St


(θ0 + θi)

s

θ0θix

ni0,t
1

τi,t ≥τ0
	

Lem. 2
≲
1
m

T
X

t=1

X

i∈St
θ3/2
max

rmx

τi,t
≲1

m

K
X

i=1
θ3/2
max
√mxτi,t ≲θ3/2
max
√

xKT .

where we used Pn
i=1 1/
√

i ≤2√n and P

i τi,t = mT together with Jensen’s inequality in the
207

last inequality. We thus have under the event E that Regtop
T
≤O(θ3/2
max
√

xKT) and the proof
208

is concluded by taking the expectation with x = 2 log T to control P(Ec).
209

3.4
Analysis: Wtd-Top-m Objective
210

We turn now to the analysis of the Wtd-Top-m objective (2). We start by stating a lemma
211

from [2] that shows that the expected utility R(S∗, θ) that corresponds to the optimal
212

assortment S∗= argmaxS⊂[K],|S|≤m R(S, θ) is non-decreasing in the parameters θ.
213

Lemma 4 (Lemma A.3 of [2]). Assume θucb
i
≥θi for all i ∈[K], then R(S∗, θ) ≤R(S∗, θucb).
214

Theorem 5 (Wtd-Top-m Objective). Let θmax ≥1. Then, for any θ ∈[0, θmax]K and
215

weights r ∈[0, 1]K, the weighted regret of AOA-RBPL (Alg. 1) with x = 2 log T
216

Regwtd
T
= O(
p

θmaxKT log T)
when
T →∞.

The complete proof is postponed to App. B. The rate Ω(
√

KT) is optimal as proved by the
217

lower bound in [19] for MNL bandit problems for θmax = 1. Our result recovers (up to a factor
218
√log T) the one of [2] when θmax = 1. However, their algorithm relies on more sophisticated
219

estimators that necessitate epochs repeating the same assortment until the No-Choice item
220

is selected. Note for our problem setting, where it is possible to have θmax ≫θ0 = 1, the
221

length of these epochs could be of O(Kθmax), which could be potentially very large when
222

θmax ≫1. This reduces the number of effective epochs, leading to poor estimation of the PL
223

parameters. We see this tradeoff in our experiments (Sec. 5) where the MNL-UCB algorithm
224

of [2] yields linear O(T) regret for such choice of the problem parameters.
225

6


---Page Break---
Remark 1 (Beyond MNL Models). Although, in this paper, we primarily focused on MNL
226

based choice models, it is worth mentioning that our proposed algorithms can be generalized
227

to more general random utility based models (RUMs) [9, 33] pursuing the ideas from [36]
228

that extends the RB based parameter estimation technique to any RUM(θ) choice models.
229

Our algorithms and analyses thus apply to any general RUM(θ) based choice models; we stick
230

to the special case of MNL models in this paper for brevity and keep the main focus on the
231

AOA problem and the related algorithmic novelties.
232

Proof sketch of Thm. 5. Let E be the high-probability event such that both Lemma 1 and 2
233

are satisfied. Then,
234

Regwtd
T
=

T
X

t=1
E

R(S∗, θ) −R(St, θ)

≲

T
X

t=1
E

(R(S∗, θ) −R(St, θ))1{E}

+ TP(Ec)

≲

T
X

t=1
E

(R(St, θucb
t
) −R(St, θ))1{E}

+ TP(Ec)
(5)

because R(St, θucb
t
) ≥R(S∗, θucb
t
) ≥R(S∗, θ) under the event E by Lemma 4. We now
235

upper-bound the first term of the right-hand-side
236

T
X

t=1
E
h 
R(St, θucb
t
) −R(St, θ)

1{E}
i
=

T
X

t=1
E
 X

i∈St

riθucb
i,t
θ0 + Θucb
St,t
−
riθi
θ0 + ΘSt


1{E}


≤

T
X

t=1
E
 X

i∈St

ri(θucb
i,t −θi)
θ0 + ΘSt


1{E}

Because Θucb
St,t ≥ΘSt under the event E by Lemma 1. Then, using ri ≤1, we further upper-
237

bound using an exploration parameter τ0 = O(log(T)) so that the upper-confidence-bounds
238

in Lemmas 1 and 2 are satisfied
239

T
X

t=1
E
h 
R(St, θucb
t
) −R(St, θ)

1{E}
i
≤

K
X

i=1
E

"
T
X

t=1

|θucb
i,t −θi|
θ0 + ΘSt


1{i ∈St, E}

#

≲O(τ0) +

K
X

i=1
E

"
T
X

t=1

|θucb
i,t −θi|
θ0 + ΘSt
1{i ∈St, τi,t ≥τ0, E}

#

≲O(τ0) +

K
X

i=1

v
u
u
t

T
X

t=1
E

"
θi1{i ∈St}

θ0 + ΘSt

#

×

v
u
u
t

T
X

t=1
E

" θucb
i,t −θi
θ0 + ΘSt

2 θ0 + ΘSt

θi
1{i ∈St, τi,t ≥τ0, E}

#

|
{z
}
=:AT (i)
(6)

where the last inequality is by Cauchy-Schwarz inequality. Now, the term AT (i) above may
240

be upper-bounded using Lemmas 1 and 2,
241

AT (i) = E

"
(θucb
i,t −θi)2

θi(θ0 + ΘSt)1{i ∈St, τi,t ≥τ0, E}

#

≲

T
X

t=1
E

"
(θ0 + θi)2x
ni0,t(θ0 + ΘSt)1{i ∈St}

#

≲θmaxx

T
X

t=1
E

"
(θ0 + θi)1{i ∈St}

(θ0 + ΘSt)ni0,t

#

= θmaxxE

"
T
X

t=1

1{it ∈{i, 0}, i ∈St}

ni0,t

#

≲θmaxx log T

where in the last inequality we used that PT
n=1 n−1 ≤1 + log T. Substituting into (6),
242

Jensen’s inequality entails,
243

T
X

t=1
E
h 
R(St, θucb
t
)−R(St, θ)

1{E}
i
≲O(τ0)+E

"
p

θmaxx log T

K
X

i=1

v
u
u
t

T
X

t=1

θi1{i ∈St}

θ0 + ΘSt

#

.

(7)

7


---Page Break---
The proof is finally concluded by applying Cauchy-Schwarz inequality which yields:
244

K
X

i=1

v
u
u
t

T
X

t=1

θi1{i ∈St}

θ0 + ΘSt
≤

v
u
u
tK

T
X

t=1

PK
i=1 θi1{i ∈St}

θ0 + ΘSt
≤
√

KT .

Finally, combining the above result with (5) and (7) concludes the proof
245

Regwtd
T
≲TP(Ec) + O(τ0) +
p

θmaxxKT log T .

Choosing x = 2 log T ensures TP(Ec) ≤O(1) and τ0 ≤O(log T).
246

4
Improved dependance on θmax with Adaptive Pivot Selection
247

A problem with Algorithm 1 stems from estimating all θi based on pairwise comparisons with
248

item 0. When θmax ≫θ0 = 1, item 0 may not be sampled enough as the winner, leading to
249

poor estimators. This deficiency contributes to the suboptimal dependence on θmax observed
250

in Theorems 3 and 5 and in prior work, such as [2]. We propose the following fix to optimize
251

the pivot. For all i, j ∈[K] ∪{0} we define γij = θi

θj , and the estimators:
252

γucb
ij,t = pucb
ij,t/(1 −pucb
ij,t)+
and
γucb
ii,t = 1 ,

where pucb
ij,t are defined in (3). For all rounds t, the algorithm AOA-RBPL-Adaptive selects
253

St = argmax
|S|≤m
R(S, bθucb
t
)
where
bθucb
i,t :=
min
j∈[K]∪{0} γucb
ij,tγucb
j0,t .

We offer below a regret bound that underscores the value of optimizing the pivot when
254

θmax ≫K. Note that while the algorithm and analysis are presented for the weighted
255

objective with winner feedback only, it can be adapted to other objectives by replacing
256

R(S, θ) with the new objective in the analysis, as long as Lemma 4 remains valid.
257

Theorem 6. Let θmax ≥1. For any θ ∈[0, θmax]K and weights r ∈[0, 1]K, the weighted
258

regret of AOA-RBPL-Adaptive is upper-bounded as
259

Regwtd
T
= O
 p

min{θmax, K}KT log T


as T →∞for the choice x = 2 log T (when definining pucb
ij,t).
260

Asymptotically, when θmax is constant, the regret is O(K
√

T log T), eliminating any depen-
261

dence on θmax. This allows for handling scenarios where the No-Choice item is highly unlikely,
262

which is not achievable in previous works such as [2, 1]. [2] did attempt in their Thm. 4 to
263

relax the assumption of θmax = θ0 and shows a bound of order O
 
max{θmax/θ0, 1}1/2√

KT

,
264

which unfortunately blows to ∞as θ0 →0 or equivalently θmax →∞, leading to a vac-
265

uous bound. Here, lies the stark improvement and one of the key contributions, as also
266

corroborated in our experimental evaluation Sec. 5 (Fig. 2).
267

The proof is deferred to the App. B, with a key step relying on selecting the pivot
268

jt = argmaxj∈St∪{0} θj.
The use of |bθucb
i,t −θi| ≤|γucb
ijt,t −θi| provides confidence upper-
269

bounds with an improved dependence on θmax , leveraging the fact that θjt ≥θi. Due
270

to the varying pivot over time, a telescoping argument introduces an additive factor
√

K.
271

5
Experiments
272

We provide here a synthetic experiments. All results are averaged across 100 runs. We
273

evaluate the performance of our main algorithm AOA-RBPL-Adaptive (Sec. 4), referred
274

as “Our Alg-1 (Adaptive Pivot)", with the following two algorithms: AOA-RBPL (Sec. 3)
275

referred as “Our Alg-2 (No-Choice Pivot)", and MNL-UCB, the state-of-the-art algorithm
276

for AOA ([2], Alg. 1).
277

Different PL (θ) Environments. We report our experiment results on two datasets with
278

K = 50 items: (1) Arith50 with PL parameters θi = 1 −(i −1)0.2, ∀i ∈[50]. (2) Bad50
279

8


---Page Break---
with PL parameters θi = 0.6, ∀i ∈[50] \ {25} and θ25 = 0.8. For simplicity of computing
280

the assortment choices St, we assume ri = 1, ∀i ∈[K].
281

(1). Averaged Regret with weak NC (θmax/θ0 ≫1) (Fig. 1): In our first experiment,
282

we set set m = 5 and θ0/θmax = 0.01 and report the average regret of the above three
283

algorithms for our two objectives.

Figure 1: Averaged Regret for m = 5, θ0 = 0.01
284

Fig. 1 shows that our algorithm AOA-RBPL-Adaptive (with adaptive pivot) significantly
285

outperforms the other two algorithms, while our algorithm AOA-RBPL with no-choice (NC)
286

pivot still outperforms MNL-UCB.
287

(2). Averaged Regret vs No-Choice PL Parameter (θmax/θ0) (Fig. 2): In this
288

experiment, we evaluate the regret performance of our algorithm AOA-RBPL-Adaptive. We
289

report the experiment on Artith50 PL dataset and set the subsetsize m = 5, θmax/θ0 =
290

{1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001}. Fig. 2 shows the increase in the performance gap between
291

our algorithm AOA-RBPL-Adaptive (with adaptive pivot) with decreasing θ0/θmax.
292

Figure 2: Comparative performance
for varying θ0/θmax, m = 5

Figure 3: Tradofff: Averaged Regret vs
length of the k rank-ordered feedback

293

(3). Averaged Regret vs Length of the rank-ordered feedback (k) (Fig. 3): We
294

also run a thought experiment to understand the tradeoff between learning rate with k-length
295

rank-ordered feedback, where given any assortment St ⊆[K] of size m, the learner gets to
296

see the top-k draws (k ≤m) from the PL model without replacement. This is a stronger
297

feedback than the winner (i.e. top-1 for k = 1) feedback and, as expected, we see in Fig. 3
298

an improved regret (for both notions) when increasing k. The experiment are run on the
299

Artith50 dataset with m = 30 and k ∈{1, 2, 4, 8}.
300

6
Conclusion
301

We address the Active Optimal Assortment Selection problem with PL choice models, in-
302

troducing a versatile framework (AOA) that eliminates the need for a strong default item,
303

typically assumed as the No-Choice (NC) item in the existing literature. Our proposed
304

algorithms employ a novel ’Rank-Breaking’ technique to establish tight concentration guar-
305

antees for estimating the score parameters of the PL model. Our approach stands out for
306

its practicality and avoids the suboptimal practice of repeatedly selecting the same set of
307

items until the default item prevails. This is beneficial when the default item’s quality
308

(θ0) is significantly lower than the quality of the best item (θmax). Our algorithms are
309

computationally efficient, optimal (up to log factors), and free from restrictive assumptions
310

on the default item.
311

Future Works.
Among many interesting questions to address in the future, it will be
312

interesting to understand the role of the No-Choice (NC) item in the algorithm design,
313

precisely, can we design efficient algorithms without the existence of NC items with a regret
314

rate still linear in θmax? Further, it will be interesting to extend our results to more general
315

choice models beyond the PL model [18, 22, 23]. What is the tradeoff between the subsetsize
316

m and the regret for such general choice models? Extending our results to large (potentially
317

infinite) decision spaces and contextual settings would also be a very useful and practical
318

contribution to the literature of assortment optimization.
319

9


---Page Break---
References
320

[1] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Thompson sampling for
321

the mnl-bandit. In Conference on learning theory, pages 76–78. PMLR, 2017.
322

[2] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Mnl-bandit: A dynamic
323

learning approach to assortment selection. Operations Research, 67(5):1453–1485, 2019.
324

[3] Nir Ailon, Zohar Karnin, and Thorsten Joachims. Reducing dueling bandits to cardinal bandits.
325

In International Conference on Machine Learning, pages 856–864. PMLR, 2014.
326

[4] Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Exploration–exploitation tradeoff
327

using variance estimates in multi-armed bandits. Theoretical Computer Science, 410(19):1876–
328

1902, 2009.
329

[5] Peter Auer. Using upper confidence bounds for online learning. In Foundations of Computer
330

Science, 2000. Proceedings. 41st Annual Symposium on, pages 270–279. IEEE, 2000.
331

[6] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed
332

bandit problem. Machine learning, 47(2-3):235–256, 2002.
333

[7] Vashist Avadhanula, Jalaj Bhandari, Vineet Goyal, and Assaf Zeevi. On the tightness of
334

an lp relaxation for rational optimization and its applications. Operations Research Letters,
335

44(5):612–617, 2016.
336

[8] Hossein Azari, David Parkes, and Lirong Xia. Random utility theory for social choice. In
337

Advances in Neural Information Processing Systems, pages 126–134, 2012.
338

[9] Hossein Azari, David Parks, and Lirong Xia. Random utility theory for social choice. Advances
339

in Neural Information Processing Systems, 25, 2012.
340

[10] Viktor Bengs, Róbert Busa-Fekete, Adil El Mesaoudi-Paul, and Eyke Hüllermeier. Preference-
341

based online learning with dueling bandits: A survey. Journal of Machine Learning Research,
342

2021.
343

[11] Viktor Bengs, Róbert Busa-Fekete, Adil El Mesaoudi-Paul, and Eyke Hüllermeier. Preference-
344

based online learning with dueling bandits: A survey. J. Mach. Learn. Res., 22:7–1, 2021.
345

[12] Viktor Bengs, Aadirupa Saha, and Eyke Hüllermeier. Stochastic contextual dueling bandits
346

under linear stochastic transitivity models. In International Conference on Machine Learning,
347

pages 1764–1786. PMLR, 2022.
348

[13] Gerardo Berbeglia and Gwenaël Joret. Assortment optimisation under a general discrete choice
349

model: A tight analysis of revenue-ordered assortments. arXiv preprint arXiv:1606.01371, 2016.
350

[14] Brian Brost, Yevgeny Seldin, Ingemar J. Cox, and Christina Lioma. Multi-dueling bandits and
351

their application to online ranker evaluation. CoRR, abs/1608.06253, 2016.
352

[15] Niladri S Chatterji, Aldo Pacchiano, Peter L Bartlett, and Michael I Jordan. On the theory of
353

reinforcement learning with once-per-episode feedback. arXiv preprint arXiv:2105.14363, 2021.
354

[16] Xi Chen, Sivakanth Gopi, Jieming Mao, and Jon Schneider. Competitive analysis of the top-k
355

ranking problem. In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on
356

Discrete Algorithms, pages 1245–1264. SIAM, 2017.
357

[17] Xi Chen, Yuanzhi Li, and Jieming Mao. A nearly instance optimal algorithm for top-k ranking
358

under the multinomial logit model. In Proceedings of the Twenty-Ninth Annual ACM-SIAM
359

Symposium on Discrete Algorithms, pages 2504–2522. SIAM, 2018.
360

[18] Xi Chen, Chao Shi, Yining Wang, and Yuan Zhou. Dynamic assortment planning under nested
361

logit models. Production and Operations Management, 30(1):85–102, 2021.
362

[19] Xi Chen and Yining Wang. A note on a tight lower bound for mnl-bandit assortment selection
363

models. arXiv preprint arXiv:1709.06109, 2017.
364

[20] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep
365

reinforcement learning from human preferences. Advances in neural information processing
366

systems, 30, 2017.
367

10


---Page Break---
[21] James Davis, Guillermo Gallego, and Huseyin Topaloglu. Assortment planning under the
368

multinomial logit model with totally unimodular constraint structures. Work in Progress, 2013.
369

[22] Antoine Désir, Vineet Goyal, Srikanth Jagabathula, and Danny Segev. Assortment optimization
370

under the mallows model. In Advances in Neural Information Processing Systems, pages
371

4700–4708, 2016.
372

[23] Antoine Désir, Vineet Goyal, Danny Segev, and Chun Ye. Capacity constrained assortment
373

optimization under the markov chain based choice model. Operations Research, 2016.
374

[24] James A Grant and David S Leslie. Learning to rank under multinomial logit choice. Journal
375

of Machine Learning Research, 24(260):1–49, 2023.
376

[25] Minje Jang, Sunghyun Kim, Changho Suh, and Sewoong Oh. Optimal sample complexity of
377

m-wise data for top-k ranking. In Advances in Neural Information Processing Systems, pages
378

1685–1695, 2017.
379

[26] Ashish Khetan and Sewoong Oh. Data-driven rank breaking for efficient rank aggregation.
380

Journal of Machine Learning Research, 17(193):1–54, 2016.
381

[27] Kameng Nip, Zhenbo Wang, and Zizhuo Wang.
Assortment optimization under a single
382

transition model. 2017.
383

[28] Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits.
384

Advances in Neural Information Processing Systems, 32, 2019.
385

[29] Mingdong Ou, Nan Li, Shenghuo Zhu, and Rong Jin. Multinomial logit bandit with linear
386

utility functions. arXiv preprint arXiv:1805.02971, 2018.
387

[30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
388

Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
389

follow instructions with human feedback. Advances in Neural Information Processing Systems,
390

35:27730–27744, 2022.
391

[31] Wenbo Ren, Jia Liu, and Ness B Shroff. PAC ranking from pairwise and listwise queries: Lower
392

bounds and upper bounds. arXiv preprint arXiv:1806.02970, 2018.
393

[32] Paat Rusmevichientong, Zuo-Jun Max Shen, and David B Shmoys. Dynamic assortment
394

optimization with a multinomial logit choice model and capacity constraint.
Operations
395

research, 58(6):1666–1680, 2010.
396

[33] Aadirupa Saha and Suprovat Ghoshal. Exploiting correlation to achieve faster learning rates in
397

low-rank preference bandits. In International Conference on Artificial Intelligence and Statistics,
398

pages 456–482. PMLR, 2022.
399

[34] Aadirupa Saha and Aditya Gopalan. Active ranking with subset-wise preferences. International
400

Conference on Artificial Intelligence and Statistics (AISTATS), 2018.
401

[35] Aadirupa Saha and Aditya Gopalan. Combinatorial bandits with relative feedback. In Advances
402

in Neural Information Processing Systems, 2019.
403

[36] Aadirupa Saha and Aditya Gopalan. PAC Battling Bandits in the Plackett-Luce Model. In
404

Algorithmic Learning Theory, pages 700–737, 2019.
405

[37] Aadirupa Saha and Aditya Gopalan. Best-item learning in random utility models with subset
406

choices. In International Conference on Artificial Intelligence and Statistics, pages 4281–4291.
407

PMLR, 2020.
408

[38] Hossein Azari Soufiani, David C Parkes, and Lirong Xia. Computing parametric ranking models
409

via rank-breaking. In ICML, pages 360–368, 2014.
410

[39] Yanan Sui, Vincent Zhuang, Joel Burdick, and Yisong Yue.
Multi-dueling bandits with
411

dependent arms. In Conference on Uncertainty in Artificial Intelligence, UAI’17, 2017.
412

[40] Kalyan Talluri and Garrett Van Ryzin. Revenue management under a general discrete choice
413

model of consumer behavior. Management Science, 50(1):15–33, 2004.
414

[41] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k-armed dueling
415

bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.
416

11


---Page Break---
[42] Yu-Jie Zhang and Masashi Sugiyama. Online (multinomial) logistic bandit: Improved regret
417

and constant computation cost. Advances in Neural Information Processing Systems, 36, 2024.
418

[43] Zihan Zhang and Xiangyang Ji. Regret minimization for reinforcement learning by evaluating
419

the optimal bias function. In Advances in Neural Information Processing Systems, pages
420

2827–2836, 2019.
421

[44] Masrour Zoghi, Zohar S Karnin, Shimon Whiteson, and Maarten De Rijke. Copeland dueling
422

bandits. In Advances in Neural Information Processing Systems, pages 307–315, 2015.
423

[45] Masrour Zoghi, Shimon Whiteson, Remi Munos, Maarten de Rijke, et al. Relative upper
424

confidence bound for the k-armed dueling bandit problem. In JMLR Workshop and Conference
425

Proceedings, number 32, pages 10–18. JMLR, 2014.
426

[46] Masrour Zoghi, Shimon A Whiteson, Maarten De Rijke, and Remi Munos. Relative confidence
427

sampling for efficient on-line ranker evaluation. In Proceedings of the 7th ACM international
428

conference on Web search and data mining, pages 73–82. ACM, 2014.
429

12


---Page Break---
NeurIPS Paper Checklist
430

1. Claims
431

Question: Do the main claims made in the abstract and introduction accurately
432

reflect the paper’s contributions and scope?
433

Answer: [Yes]
434

Justification: In the abstract, we list the main claims of this paper in a general
435

fashion. Then, in the introduction we state them in more detail. They accurately
436

reflect the paper’s contribution and scope.
437

Guidelines:
438

• The answer NA means that the abstract and introduction do not include the
439

claims made in the paper.
440

• The abstract and/or introduction should clearly state the claims made, including
441

the contributions made in the paper and important assumptions and limitations.
442

A No or NA answer to this question will not be perceived well by the reviewers.
443

• The claims made should match theoretical and experimental results, and reflect
444

how much the results can be expected to generalize to other settings.
445

• It is fine to include aspirational goals as motivation as long as it is clear that
446

these goals are not attained by the paper.
447

2. Limitations
448

Question: Does the paper discuss the limitations of the work performed by the
449

authors?
450

Answer: [Yes]
451

Justification: We discuss the limitations and assumptions of our work throughout
452

the paper. Additional limitations are highlighted in the discussion.
453

Guidelines:
454

• The answer NA means that the paper has no limitation while the answer No
455

means that the paper has limitations, but those are not discussed in the paper.
456

• The authors are encouraged to create a separate "Limitations" section in their
457

paper.
458

• The paper should point out any strong assumptions and how robust the results
459

are to violations of these assumptions (e.g., independence assumptions, noiseless
460

settings, model well-specification, asymptotic approximations only holding
461

locally). The authors should reflect on how these assumptions might be violated
462

in practice and what the implications would be.
463

• The authors should reflect on the scope of the claims made, e.g., if the approach
464

was only tested on a few datasets or with a few runs. In general, empirical
465

results often depend on implicit assumptions, which should be articulated.
466

• The authors should reflect on the factors that influence the performance of the
467

approach. For example, a facial recognition algorithm may perform poorly when
468

image resolution is low or images are taken in low lighting. Or a speech-to-text
469

system might not be used reliably to provide closed captions for online lectures
470

because it fails to handle technical jargon.
471

• The authors should discuss the computational efficiency of the proposed algo-
472

rithms and how they scale with dataset size.
473

• If applicable, the authors should discuss possible limitations of their approach
474

to address problems of privacy and fairness.
475

• While the authors might fear that complete honesty about limitations might
476

be used by reviewers as grounds for rejection, a worse outcome might be that
477

reviewers discover limitations that aren’t acknowledged in the paper. The
478

authors should use their best judgment and recognize that individual actions in
479

favor of transparency play an important role in developing norms that preserve
480

the integrity of the community. Reviewers will be specifically instructed to not
481

penalize honesty concerning limitations.
482

13


---Page Break---
3. Theory Assumptions and Proofs
483

Question: For each theoretical result, does the paper provide the full set of assump-
484

tions and a complete (and correct) proof?
485

Answer: [Yes]
486

Justification: The assumptions can be found in the Problem Setup section and in
487

the paragraphs before the theorems and remarks. We provide proof sketches in the
488

main text and complete proofs in the appendix.
489

Guidelines:
490

• The answer NA means that the paper does not include theoretical results.
491

• All the theorems, formulas, and proofs in the paper should be numbered and
492

cross-referenced.
493

• All assumptions should be clearly stated or referenced in the statement of any
494

theorems.
495

• The proofs can either appear in the main paper or the supplemental material,
496

but if they appear in the supplemental material, the authors are encouraged to
497

provide a short proof sketch to provide intuition.
498

• Inversely, any informal proof provided in the core of the paper should be
499

complemented by formal proofs provided in appendix or supplemental material.
500

• Theorems and Lemmas that the proof relies upon should be properly referenced.
501

4. Experimental Result Reproducibility
502

Question: Does the paper fully disclose all the information needed to reproduce
503

the main experimental results of the paper to the extent that it affects the main
504

claims and/or conclusions of the paper (regardless of whether the code and data are
505

provided or not)?
506

Answer: [Yes]
507

Justification: experimental details including algorithms and setups are clearly pro-
508

vided. In addition, the main contribution of the paper is theoretical and synthetic
509

experiments are mostly provided as an illustration.
510

Guidelines:
511

• The answer NA means that the paper does not include experiments.
512

• If the paper includes experiments, a No answer to this question will not be
513

perceived well by the reviewers: Making the paper reproducible is important,
514

regardless of whether the code and data are provided or not.
515

• If the contribution is a dataset and/or model, the authors should describe the
516

steps taken to make their results reproducible or verifiable.
517

• Depending on the contribution, reproducibility can be accomplished in various
518

ways. For example, if the contribution is a novel architecture, describing the
519

architecture fully might suffice, or if the contribution is a specific model and
520

empirical evaluation, it may be necessary to either make it possible for others
521

to replicate the model with the same dataset, or provide access to the model. In
522

general. releasing code and data is often one good way to accomplish this, but
523

reproducibility can also be provided via detailed instructions for how to replicate
524

the results, access to a hosted model (e.g., in the case of a large language model),
525

releasing of a model checkpoint, or other means that are appropriate to the
526

research performed.
527

• While NeurIPS does not require releasing code, the conference does require all
528

submissions to provide some reasonable avenue for reproducibility, which may
529

depend on the nature of the contribution. For example
530

(a) If the contribution is primarily a new algorithm, the paper should make it
531

clear how to reproduce that algorithm.
532

(b) If the contribution is primarily a new model architecture, the paper should
533

describe the architecture clearly and fully.
534

14


---Page Break---
(c) If the contribution is a new model (e.g., a large language model), then there
535

should either be a way to access this model for reproducing the results or a
536

way to reproduce the model (e.g., with an open-source dataset or instructions
537

for how to construct the dataset).
538

(d) We recognize that reproducibility may be tricky in some cases, in which
539

case authors are welcome to describe the particular way they provide for
540

reproducibility. In the case of closed-source models, it may be that access to
541

the model is limited in some way (e.g., to registered users), but it should be
542

possible for other researchers to have some path to reproducing or verifying
543

the results.
544

5. Open access to data and code
545

Question: Does the paper provide open access to the data and code, with sufficient
546

instructions to faithfully reproduce the main experimental results, as described in
547

supplemental material?
548

Answer: [No]
549

Justification: experimental setups are synthetic and can easily be reproduced.
550

Guidelines:
551

• The answer NA means that paper does not include experiments requiring code.
552

• Please see the NeurIPS code and data submission guidelines (https://nips.
553

cc/public/guides/CodeSubmissionPolicy) for more details.
554

• While we encourage the release of code and data, we understand that this might
555

not be possible, so “No” is an acceptable answer. Papers cannot be rejected
556

simply for not including code, unless this is central to the contribution (e.g., for
557

a new open-source benchmark).
558

• The instructions should contain the exact command and environment needed
559

to run to reproduce the results.
See the NeurIPS code and data submis-
560

sion guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy)
561

for more details.
562

• The authors should provide instructions on data access and preparation, in-
563

cluding how to access the raw data, preprocessed data, intermediate data, and
564

generated data, etc.
565

• The authors should provide scripts to reproduce all experimental results for
566

the new proposed method and baselines. If only a subset of experiments are
567

reproducible, they should state which ones are omitted from the script and why.
568

• At submission time, to preserve anonymity, the authors should release
569

anonymized versions (if applicable).
570

• Providing as much information as possible in supplemental material (appended
571

to the paper) is recommended, but including URLs to data and code is permitted.
572

6. Experimental Setting/Details
573

Question: Does the paper specify all the training and test details (e.g., data splits,
574

hyperparameters, how they were chosen, type of optimizer, etc.)
necessary to
575

understand the results?
576

Answer: [Yes]
577

Justification: All details are provided to reproduce the experiments.
578

Guidelines:
579

• The answer NA means that the paper does not include experiments.
580

• The experimental setting should be presented in the core of the paper to a level
581

of detail that is necessary to appreciate the results and make sense of them.
582

• The full details can be provided either with the code, in appendix, or as
583

supplemental material.
584

7. Experiment Statistical Significance
585

Question: Does the paper report error bars suitably and correctly defined or other
586

appropriate information about the statistical significance of the experiments?
587

15


---Page Break---
Answer: [Yes]
588

Justification: [Yes]
589

Guidelines:
590

• The answer NA means that the paper does not include experiments.
591

• The authors should answer "Yes" if the results are accompanied by error bars,
592

confidence intervals, or statistical significance tests, at least for the experiments
593

that support the main claims of the paper.
594

• The factors of variability that the error bars are capturing should be clearly
595

stated (for example, train/test split, initialization, random drawing of some
596

parameter, or overall run with given experimental conditions).
597

• The method for calculating the error bars should be explained (closed form
598

formula, call to a library function, bootstrap, etc.)
599

• The assumptions made should be given (e.g., Normally distributed errors).
600

• It should be clear whether the error bar is the standard deviation or the standard
601

error of the mean.
602

• It is OK to report 1-sigma error bars, but one should state it. The authors
603

should preferably report a 2-sigma error bar than state that they have a 96%
604

CI, if the hypothesis of Normality of errors is not verified.
605

• For asymmetric distributions, the authors should be careful not to show in
606

tables or figures symmetric error bars that would yield results that are out of
607

range (e.g. negative error rates).
608

• If error bars are reported in tables or plots, The authors should explain in the
609

text how they were calculated and reference the corresponding figures or tables
610

in the text.
611

8. Experiments Compute Resources
612

Question: For each experiment, does the paper provide sufficient information on the
613

computer resources (type of compute workers, memory, time of execution) needed
614

to reproduce the experiments?
615

Answer: [NA]
616

Justification: [NA]
617

Guidelines:
618

• The answer NA means that the paper does not include experiments.
619

• The paper should indicate the type of compute workers CPU or GPU, internal
620

cluster, or cloud provider, including relevant memory and storage.
621

• The paper should provide the amount of compute required for each of the
622

individual experimental runs as well as estimate the total compute.
623

• The paper should disclose whether the full research project required more
624

compute than the experiments reported in the paper (e.g., preliminary or failed
625

experiments that didn’t make it into the paper).
626

9. Code Of Ethics
627

Question: Does the research conducted in the paper conform, in every respect, with
628

the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
629

Answer: [Yes]
630

Justification: We do not see any potential negative social impact of this work and it
631

follows the NeurIPS code of ethics.
632

Guidelines:
633

• The answer NA means that the authors have not reviewed the NeurIPS Code
634

of Ethics.
635

• If the authors answer No, they should explain the special circumstances that
636

require a deviation from the Code of Ethics.
637

• The authors should make sure to preserve anonymity (e.g., if there is a special
638

consideration due to laws or regulations in their jurisdiction).
639

16


---Page Break---
10. Broader Impacts
640

Question: Does the paper discuss both potential positive societal impacts and
641

negative societal impacts of the work performed?
642

Answer: [NA]
643

Justification: This work addresses the problem of designing efficient and optimal
644

algorithms for different assortment selection problems with MNL models. Our
645

work is purely theoretical and studies a fundamental mathematical optimization
646

framework that is unrelated to societal considerations
647

Guidelines:
648

• The answer NA means that there is no societal impact of the work performed.
649

• If the authors answer NA or No, they should explain why their work has no
650

societal impact or why the paper does not address societal impact.
651

• Examples of negative societal impacts include potential malicious or unintended
652

uses (e.g., disinformation, generating fake profiles, surveillance), fairness consid-
653

erations (e.g., deployment of technologies that could make decisions that unfairly
654

impact specific groups), privacy considerations, and security considerations.
655

• The conference expects that many papers will be foundational research and
656

not tied to particular applications, let alone deployments. However, if there
657

is a direct path to any negative applications, the authors should point it out.
658

For example, it is legitimate to point out that an improvement in the quality
659

of generative models could be used to generate deepfakes for disinformation.
660

On the other hand, it is not needed to point out that a generic algorithm for
661

optimizing neural networks could enable people to train models that generate
662

Deepfakes faster.
663

• The authors should consider possible harms that could arise when the technology
664

is being used as intended and functioning correctly, harms that could arise when
665

the technology is being used as intended but gives incorrect results, and harms
666

following from (intentional or unintentional) misuse of the technology.
667

• If there are negative societal impacts, the authors could also discuss possible
668

mitigation strategies (e.g., gated release of models, providing defenses in addition
669

to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a
670

system learns from feedback over time, improving the efficiency and accessibility
671

of ML).
672

11. Safeguards
673

Question: Does the paper describe safeguards that have been put in place for
674

responsible release of data or models that have a high risk for misuse (e.g., pretrained
675

language models, image generators, or scraped datasets)?
676

Answer: [NA]
677

Justification: [NA]
678

Guidelines:
679

• The answer NA means that the paper poses no such risks.
680

• Released models that have a high risk for misuse or dual-use should be released
681

with necessary safeguards to allow for controlled use of the model, for example
682

by requiring that users adhere to usage guidelines or restrictions to access the
683

model or implementing safety filters.
684

• Datasets that have been scraped from the Internet could pose safety risks. The
685

authors should describe how they avoided releasing unsafe images.
686

• We recognize that providing effective safeguards is challenging, and many papers
687

do not require this, but we encourage authors to take this into account and
688

make a best faith effort.
689

12. Licenses for existing assets
690

Question: Are the creators or original owners of assets (e.g., code, data, models),
691

used in the paper, properly credited and are the license and terms of use explicitly
692

mentioned and properly respected?
693

17


---Page Break---
Answer: [NA]
694

Justification: [NA]
695

Guidelines:
696

• The answer NA means that the paper does not use existing assets.
697

• The authors should cite the original paper that produced the code package or
698

dataset.
699

• The authors should state which version of the asset is used and, if possible,
700

include a URL.
701

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
702

• For scraped data from a particular source (e.g., website), the copyright and
703

terms of service of that source should be provided.
704

• If assets are released, the license, copyright information, and terms of use in
705

the package should be provided. For popular datasets, paperswithcode.com/
706

datasets has curated licenses for some datasets. Their licensing guide can help
707

determine the license of a dataset.
708

• For existing datasets that are re-packaged, both the original license and the
709

license of the derived asset (if it has changed) should be provided.
710

• If this information is not available online, the authors are encouraged to reach
711

out to the asset’s creators.
712

13. New Assets
713

Question: Are new assets introduced in the paper well documented and is the
714

documentation provided alongside the assets?
715

Answer: [NA]
716

Justification: [NA]
717

Guidelines:
718

• The answer NA means that the paper does not release new assets.
719

• Researchers should communicate the details of the dataset/code/model as part
720

of their submissions via structured templates.
This includes details about
721

training, license, limitations, etc.
722

• The paper should discuss whether and how consent was obtained from people
723

whose asset is used.
724

• At submission time, remember to anonymize your assets (if applicable). You
725

can either create an anonymized URL or include an anonymized zip file.
726

14. Crowdsourcing and Research with Human Subjects
727

Question: For crowdsourcing experiments and research with human subjects, does
728

the paper include the full text of instructions given to participants and screenshots,
729

if applicable, as well as details about compensation (if any)?
730

Answer: [NA]
731

Justification: [NA]
732

Guidelines:
733

• The answer NA means that the paper does not involve crowdsourcing nor
734

research with human subjects.
735

• Including this information in the supplemental material is fine, but if the main
736

contribution of the paper involves human subjects, then as much detail as
737

possible should be included in the main paper.
738

• According to the NeurIPS Code of Ethics, workers involved in data collection,
739

curation, or other labor should be paid at least the minimum wage in the
740

country of the data collector.
741

15. Institutional Review Board (IRB) Approvals or Equivalent for Research
742

with Human Subjects
743

18


---Page Break---
Question: Does the paper describe potential risks incurred by study participants,
744

whether such risks were disclosed to the subjects, and whether Institutional Review
745

Board (IRB) approvals (or an equivalent approval/review based on the requirements
746

of your country or institution) were obtained?
747

Answer: [NA]
748

Justification: [NA]
749

Guidelines:
750

• The answer NA means that the paper does not involve crowdsourcing nor
751

research with human subjects.
752

• Depending on the country in which research is conducted, IRB approval (or
753

equivalent) may be required for any human subjects research. If you obtained
754

IRB approval, you should clearly state this in the paper.
755

• We recognize that the procedures for this may vary significantly between insti-
756

tutions and locations, and we expect authors to adhere to the NeurIPS Code of
757

Ethics and the guidelines for their institution.
758

• For initial submissions, do not include any information that would break
759

anonymity (if applicable), such as the institution conducting the review.
760

19


---Page Break---
Supplementary: Optimal, Efficient and Practical
761

Algorithms for Assortment Optimization
762

A
Preliminaries: Some Useful Concepts for PL choice models
763

A.1
Plackett-Luce (PL): A Discrete Choice Model
764

A discrete choice model specifies the relative preferences of two or more discrete alternatives
765

in a given set. A widely studied class of discrete choice models is the class of Random
766

Utility Models (RUMs), which assume a ground-truth utility score θi ∈R for each alternative
767

i ∈[n], and assign a conditional distribution Di(·|θi) for scoring item i. To model a winning
768

alternative given any set S ⊆[n], one first draws a random utility score Xi ∼Di(·|θi) for
769

each alternative in S, and selects an item with the highest random score.
770

One widely used RUM is the Multinomial-Logit (MNL) or Plackett-Luce model (PL), where
the Dis are taken to be independent Gumbel distributions with parameters θ′
i [8], i.e., with
probability densities

Di(xi|θ′
i) = e−(xj−θ′
j)e−e
−(xj −θ′
j )
,
θ′
i ∈R, ∀i ∈[n] .

Moreover assuming θ′
i = ln θi, θi > 0 ∀i ∈[n], it can be shown in this case the probability
771

that an alternative i emerges as the winner in the set S ∋i becomes: P(i|S) =
θi
P

j∈S θj .
772

Other families of discrete choice models can be obtained by imposing different probability
773

distributions over the utility scores Xi, e.g. if (X1, . . . Xn) ∼N(θ, Λ) are jointly normal
774

with mean θ = (θ1, . . . θn) and covariance Λ ∈Rn×n, then the corresponding RUM-based
775

choice model reduces to the Multinomial Probit (MNP).
776

A.2
Rank Breaking
777

Rank breaking (RB) is a well-understood idea involving the extraction of pairwise comparisons
778

from (partial) ranking data, and then building pairwise estimators on the obtained pairs by
779

treating each comparison independently [26, 25], e.g., a winner a sampled from among a, b, c is
780

rank-broken into the pairwise preferences a ≻b, a ≻c. We use this idea to devise estimators
781

for the pairwise win probabilities pij = P(i|{i, j}) = θi/(θi + θj) for our problem setting.
782

We used the idea of RB in both our algorithms (AOA-RBPL and AOA-RBPL-Adaptive) to
783

update the pairwise win-count estimates wi,j,t for all the item pairs (i, j) ∈[K] × [K], which
784

is further used for deriving the empirical pairwise preference estimates bpij,t, at any time t.
785

A.3
Parameter Estimation with PL based preference data
786

Lemma 7 (Pairwise win-probability estimates for the PL model [34]). Consider a Plackett-
787

Luce choice model with parameters θ = (θ1, θ2, . . . , θn), and fix two items i, j ∈[n]. Let
788

S1, . . . , ST be a sequence of (possibly random) subsets of [n] of size at least 2, where T is
789

a positive integer, and i1, . . . , iT a sequence of random items with each it ∈St, 1 ≤t ≤T,
790

such that for each 1 ≤t ≤T, (a) St depends only on S1, . . . , St−1, and (b) it is distributed
791

as the Plackett-Luce winner of the subset St, given S1, i1, . . . , St−1, it−1 and St, and (c)
792

∀t : {i, j} ⊆St with probability 1. Let ni(T) = PT
t=1 P(it = i) and nij(T) = PT
t=1 P({it ∈
793

{i, j}}). Then, for any positive integer v, and η ∈(0, 1),
794

P
 ni(T)

nij(T) −
θi
θi + θj
≥η, nij(T) ≥v

≤e−2vη2,

P
 ni(T)

nij(T) −
θi
θi + θj
≤−η, nij(T) ≥v

≤e−2vη2.

20


---Page Break---
B
Omitted Proofs from Sec. 3 and Sec. 4
795

B.1
A concentration bounds for the pij,t
796

We first prove below a concentration inequality based on Bernstein’s inequality for the
797

estimators pij,t.
798

Lemma 8. Let (i, j) ∈[K] × [K]. Let T ≥1 and x > 0. Then, with probability at least
799

1 −3Te−x,
800

pij ≤pucb
ij,t ≤pij + 2

s

2pij(1 −pij)x

nij,t
+ 11x

nij,t
,
(8)

simultaneously for all t ∈[T].
801

Proof of Lemma 8. Let T ≥1, x > 0 and i, j ∈[K]. Applying Thm. 1 of [4], with probability
802

at least 1 −β(x, T), we get simultaneously for all t ∈[T],
803

bpij,t −pij
 ≤

s

2bpij,t(1 −bpij,t)x

nij,t
+ 3x

nij,t
,
(9)

where β(x, T) = 3 inf1<α≤3 min
 log T

log α, T
	
e−x/α ≤3Te−x. Note that the inequality holds
804

true although nij,t is a random variable. This, shows the first inequality
805

pij ≤pucb
ij,t .

For the second inequality, (9) implies
806

pucb
ij,t = bpij,t +

s

2bpij,t(1 −bpij,t)x

nij,t
+ 3x

nij,t

≤pij + 2

s

2bpij,t(1 −bpij,t)x

nij,t
+ 6x

nij,t
.
(10)

Furthermore, because x 7→x(1 −x) is 1-Lipschitz on [0, 1], we have
807

bpij,t(1 −bpij,t) −pij(1 −pij)
 ≤
bpij,t −pij


(9)
≤

s

2bpij,t(1 −bpij,t)x

nij,t
+ 3x

nij,t
.

Therefore,
808

bpij,t(1 −bpij,t) ≤pij(1 −pij) +

s

2bpij,t(1 −bpij,t)x

nij,t
+ 3x

nij,t

≤
q

pij(1 −pij) +

s

3x
nij,t

2
,

which yields
809

q

bpij,t(1 −bpij,t) ≤
q

pij(1 −pij) +

s

3x
nij,t
.
(11)

Plugging back into (10), we get
810

pucb
ij,t ≤2

s

2pij(1 −pij)x

nij,t
+ 11x

nij,t
.

811

21


---Page Break---
B.2
Proof of Lemma 1
812

Proof. Let i ∈[K] and x > 0. Then, by a union bound on Lemma 8 and 2, with probability
813

at least 1 −4Te−x, (8) and (4) hold true for all t ∈[T]. We consider this high-probability
814

event in the rest of the proof. Define the function f : x 7→x/(1 −x)+ on [0, 1] (with the
815

convention f(1) = +∞), so that θucb
i,t = f(pucb
i0,t) and θi = f(pi0). Because f is non-decreasing,
816

and pucb
i0,t ≥pi0 by (8), we have
817

θucb
i,t ≥θi .
(12)
Furthermore, denote
818

∆i,t := 2

s

2pij(1 −pij)x

ni0,t
+ 11x

ni0,t
= 2

s

2θ0θix
(θ0 + θi)2ni0,t
+ 11x

ni0,t
.
(13)

In the rest of the proof we assume, ni0,t ≥69x(θ0 + θi). Then, using that θ0θi ≤θ0 + θi
819

since θ0 = 1, it implies
820

(θ0 + θi)∆i,t ≤2

s

2θ0θix

ni0,t
+ 11x(θ0 + θi)

ni0,t
≤1

2 ,

and
821

pi0 + ∆i,t =
θi
θ0 + θi
+ ∆i,t ≤θi + 1/2

θi + 1
< 1.

Thus, because f is non-decreasing
822

θucb
i,t −θi = f(pucb
i0,t) −f(pi0)

(8)
≤f
 
pi0 + ∆i,t

−f(pi0)

=
pi0 + ∆i,t
1 −pi0 −∆i,t
−
pi0
1 −pi0

=
∆i,t
(1 −pi0)(1 −pi0 −∆i,t)

=
(θ0 + θi)2∆i,t
1 −(θ0 + θi)∆i,t
≤2(θ0 + θi)2∆i,t

(13)
≤4(θ0 + θi)

s

2θ0θix

ni0,t
+ 22x(θ0 + θi)2

ni0,t
,

which concludes the proof.
823

B.3
Proof of Lemma 2
824

Proof. Let T ≥1 and i ∈[K]. Recall that τi,t = Pt−1
s=1 1{i ∈Ss} is the number of times i
825

was played at the start of round t and ni0,t = Pt−1
s=1 1{it ∈{i, 0}, i ∈St} is the number of
826

times i or 0 won up to round t when played together. When i is played the probability of 0
827

or i to win is
828

P(it ∈{i, 0}|St) =
θ0 + θi
θ0 + ΘSt
≥
θ0 + θi
θ0 + ΘS∗.

Therefore, applying Chernoff-Hoeffding inequality together with a union bound (to deal with
829

the fact that τi,t is random), we have with probability at least 1 −Te−x
830

ni0,t ≥
θ0 + θi
θ0 + ΘS∗τi,t −
rτi,tx

2
simultaneously for all t ∈[T]. Noting that
831

θ0 + θi
θ0 + ΘS∗τi,t −
rτi,tx

2
≥
θ0 + θi
2(θ0 + ΘS∗)τi,t

if τi,t ≥2x(θ0 + ΘS∗)2 ≥2x(θ0+ΘS∗)2

(θ0+θi)2
concludes the proof.
832

22


---Page Break---
B.4
Proof of Theorem 3
833

Proof. Let us define for any S ⊆[K],
834

ΘS =
X

i∈S
θi,
and
Θucb
S
:=
X

i∈S
θucb
i
.

Let E be the high-probabality event such that both Lemma 1 and 2 holds true. Then,
835

P(E) ≥1 −4TKe−x. Let us first assume that E holds true. Then, by Lemma 1,
836

Regtop
T
= 1

m

T
X

t=1
ΘS∗−ΘSt

≤1

m

T
X

t=1
min
n
ΘS∗, Θucb
St −ΘSt
o
←because ΘS∗≤Θucb
S∗≤Θucb
St under the event E

= 1

m

T
X

t=1
min
n
ΘS∗,
X

i∈St
θucb
i,t −θi
o

≤1

mΘS∗

K
X

i=1
¯τi0 + 1

m

T
X

t=1

X

i∈St
(θucb
i,t −θi)1

τi,t ≥¯τi0
	

where ¯τi0 = 2x(θ0 + ΘS∗) max{θ0 + ΘS∗, 69} ≤138x(m + 1)2θ2
max, where θmax := maxi θi.
837

Then, noting that if E holds true, by Lemma 2, we also have ni0,t ≥
1
2(θ0+ΘS∗)(θ0 + θi)τi,t,
838

which yields
839

1{τi,t ≥¯τi0} ≤1{ni0,t ≥69x(θ0 + θi)}.
Therefore, we can apply Lemma 1 that entails,
840

1
m

T
X

t=1

X

i∈St
(θucb
i,t −θi)1

τi,t ≥¯τi0
	

Lem. 1
≤
1
m

T
X

t=1

X

i∈St


4(θ0 + θi)

s

2θ0θix

ni0,t
+ 22x(θ0 + θi)2

ni0,t


1

ni0,t ≥69x(θ0 + θi)
	

Lem 2
≤
1
m

T
X

t=1

X

i∈St


8

s

(θ0 + ΘS∗)(θ0 + θi)θ0θix

τi,t
+ 44x(θ0 + ΘS∗)(θ0 + θi)

τi,t



≤1

m

K
X

i=1
16
q

(θ0 + ΘS∗)(θ0 + θi)θ0θixτi,T + 44x(θ0 + ΘS∗)

K
X

i=1
(θ0 + θi)(1 + log(τi,T )) ,

where we used Pn
i=1 1/
√

i ≤2√n and Pn
i=1 i−1 ≤1 + log n. We thus have
841

Regtop
T
≤138x(m + 1)2Kθ3
max + 1

m

K
X

i=1
16θ3/2
max
q

(m + 1)xτi,T

+ 44x(m + 1)(1 + θmax)2
K
X

i=1
(1 + log(τi,T ))

≤138x(m + 1)2Kθ3
max + 16θ3/2
max
√

2xKT + 88x(m + 1)Kθ2
max

1 + log
mT

K


.

Therefore,
842

E[Regtop
T ] ≤12
√

2xmKθ3
max + 16θ3/2
max
√

2xKT + 88xmKθ2
max

1 + log
mT

K



+ 4mKT 2e−xθmax .

Choosing x = 2 log T concludes the proof.
843

23


---Page Break---
B.5
Proof of Theorem 5
844

Proof. Let E be the high-probabality event such that Lemma 1 and 2 are satisfied, so that
845

P(E) ≥1 −4KTe−x. Then, denoting x ∧y := min{x, y},
846

Regwtd
T
=

T
X

t=1
E

R(S∗, θ) −R(St, θ)

(14)

=

T
X

t=1
E

(R(S∗, θ) −R(St, θ))1{E} + (R(S∗, θ) −R(St, θ))1{Ec}


≤

T
X

t=1
E
h 
(R(St, θucb
t
) −R(St, θ)) ∧R(S∗, θ)

1{E} + R(S∗, θ)1{Ec}
i

because R(St, θucb
t
) ≥R(S∗, θucb
t
) ≥R(S∗, θ) under the event E by Lemma 4. Then, using
847

R(S∗, θ) ≤maxi ri ≤1, we get
848

Regwtd
T
≤

T
X

t=1
E
h 
(R(St, θucb
t
) −R(St, θ)) ∧1

1{E} + 1{Ec}
i

≤4T 2Ke−x +

T
X

t=1
E
h 
R(St, θucb
t
) −R(St, θ)

∧1

1{E}
i
.

Let us upper-bound the second term of the right-hand-side
849

T
X

t=1
E
h 
R(St, θucb
t
) −R(St, θ)

∧1

1{E}
i
(15)

=

T
X

t=1
E
 X

i∈St

riθucb
i,t
θ0 + Θucb
St,t
−
riθi
θ0 + ΘSt


∧1

1{E}


≤

T
X

t=1
E
 X

i∈St

ri(θucb
i,t −θi)
θ0 + ΘSt


∧1

1{E}

because Θucb
St,t ≥ΘSt under E

≤

T
X

t=1
E
 X

i∈St

|θucb
i,t −θi|
θ0 + ΘSt


∧1

1{E}

because ri ≤1

≤

K
X

i=1
E

"
T
X

t=1

|θucb
i,t −θi|
θ0 + ΘSt
∧1

1{i ∈St}1{E}

#

≤138xm2Kθ2
max +

K
X

i=1
E

"
T
X

t=1

|θucb
i,t −θi|
θ0 + ΘSt
1{i ∈St, τi,t ≥138x(m + 1)2θ2
max}1{E}

#

≤138xm2Kθ2
max +

K
X

i=1

v
u
u
t

T
X

t=1
E

"  θ0

m + θi

1{i ∈St}
θ0 + ΘSt

#

×

v
u
u
t

T
X

t=1
E

"|θucb
i,t −θi|
θ0 + ΘSt

2 θ0 + ΘSt

θ0
m + θi
1{i ∈St, τi,t ≥138x(m + 1)2θ2
max}1{E}

#

|
{z
}
=:AT (i)
(16)

where the last inequality is by Cauchy-Schwarz inequality. Now, the term AT (i) above may
850

be upper-bounded as follows
851

AT (i) :=

T
X

t=1
E

"|θucb
i,t −θi|
θ0 + ΘSt

2 θ0 + ΘSt

θ0
m + θi
1{i ∈St, τi,t ≥138x(m + 1)2θ2
max}1{E}

#

24


---Page Break---
= E

"
(θucb
i,t −θi)2
  θ0

m + θi

θ0 + ΘSt
1{i ∈St, τi,t ≥138x(m + 1)2θ2
max}1{E}

#

.

Now, since under the event E by Lemma 2, τi,t ≥138x(m + 1)2θ2
max implies
852

ni0,t ≥69x(θ0 + θi)(m + 1)θmax ≥69x(θ0 + θi) .

Therefore, we can apply Lemma 1, which further upper-bounds
853

AT (i) ≤

T
X

t=1
E

"26(θ0 + θi)2x

ni0,t
+ 2(22x)2(θ0 + θi)4

n2
i0,t( θ0

m + θi)



× 1{i ∈St, τi,t ≥138x(m + 1)2θ2
max}
θ0 + ΘSt
1{E}

#

≤

T
X

t=1
E

"26(θ0 + θi)2x

ni0,t
+
15x(θ0 + θi)3

ni0,tθmax(θ0 + mθi)


× 1{i ∈St}

θ0 + ΘSt
1{E}

#

where we used ni0,t ≥69x(θ0 + θi)mθmax in the last inequality. Then, we get
854

AT (i) ≤

T
X

t=1
E

"(θ0 + θi)2x

ni0,t
+ 30x(θ0 + θi)

ni0,t


× 1{i ∈St}

θ0 + ΘSt
1{E}

#

≤(94 + 64θi)x

T
X

t=1
E

"
(θ0 + θi)1{i ∈St}

(θ0 + ΘSt)ni0,t

#

= (94 + 64θi)xE

"
T
X

t=1

1{it ∈{i, 0}, i ∈St}

ni0,t

#

= (94 + 64θi)xE
h
1 + log
 
ni0(T)
i

≤158θmaxx(1 + log T) .

Substituting into (16), we then obtain using Cauchy-Schwarz inequality,
855

T
X

t=1
E
h 
R(St, θucb
t
) −R(St, θ)

∧1

1{E}
i

≤138xm2Kθ2
max + 13
p

θmaxx(1 + log T)

K
X

i=1

v
u
u
t

T
X

t=1
E

"  θ0

m + θi

1{i ∈St}
θ0 + ΘSt

#

≤138xm2Kθ2
max + 13
p

θmaxx(1 + log T)

v
u
u
tE

"

K

T
X

t=1

PK
i=1
  θ0

m + θi

1{i ∈St}
θ0 + ΘSt

#

= 138xm2Kθ2
max + 13
p

θmaxx(1 + log T)KT .

Finally, replacing into Inequality (15) yields
856

Regwtd
T
≤4T 2Ke−x + 138xm2Kθ2
max + 13
p

θmaxx(1 + log T)KT .

Choosing x = 2 log T concludes the proof.
857

B.6
Proof of Theorem 6
858

The proof follows the one of Theorem 5, except that the concentration lemmas should be
859

generalized to any pairs (i, j) instead of only with respect to item 0, whose proofs are left
860

to the reader and closely follows the one of Lemma 1 and 2. For simplicity, this proof is
861

performed up to universal multiplicative constants, using the rough inequality ≲.
862

25


---Page Break---
Lemma 9. Let T ≥1 and x > 0. Then, with probability at least 1 −3K(K + 1)Te−x,
863

simultaneously for all t ∈[T] and i ̸= j in [ ˜K]: γij := θi

θj ≤γucb
ij,t and one of the following
864

two inequalities is satisfied
865

nij,t < 69x(1 + γij)
or
γucb
ij,t ≤γij + 4(γij + 1)

s

2γijx

nij,t
+ 22x(γij + 1)2

nij,t
.

Lemma 10. Let T ≥1 and x > 0. Then, with probability at least 1 −3K(K + 1)Te−x,
866

simultaneously for all t ∈[T] and i ∈[K]: bθucb
i,t := minj γucb
ij,tγucb
j0,t ≥θi and for all j one of
867

the following two inequalities is satisfied
868

nij,t ≲x(1 + γij)
or
nj0,t ≲x(1 + θj)2θ−1
j

or
869

γucb
ij,tγucb
j0,t−θi ≲
q

(γij + 1)θix
s

(θi + θj)

nij,t
+

s

(1 + θj)

nj0,t


+(γij+1)(θi + θj)x

nij,t
+γij(1 + θj)2x

nj0,t
.

Proof of Lemma 10. The proof follows from Lemma 9. If nij,t > Cx(1 + γij) and nj0,t >
870

Cx(1 + θj) for some large enough constant C, we have
871

γucb
ij,t ≤γij + 4(γij + 1)

s

2γijx

nij,t
+ 22x(γij + 1)2

nij,t

and
872

γucb
j0,t ≤γj0 + 4(γj0 + 1)

s

2γj0x

nj0,t
+ 22x(γj0 + 1)2

nj0,t
≤2γj0 .

This implies,
873

γucb
ij,tγucb
j0,t −θi = γucb
ij,tγucb
j0,t −γijγj0 = (γucb
ij,t −γij)γucb
j0,t + γij(γucb
j0,t −γj0)

≤2(γucb
ij,t −γij)γj0 + γij(γucb
j0,t −γj0)

≤8γj0(γij + 1)

s

2γijx

nij,t
+ 44xγj0(γij + 1)2

nij,t

+ 4γij(γj0 + 1)

s

2γj0x

nj0,t
+ 22xγij(γj0 + 1)2

nj0,t
.

Replacing γij = θi/θj and γj0 = θj concludes the proof.
874

Lemma 11. Let T ≥1 and x > 0. Then, with probability at least 1 −K(K + 1)Te−x
875

τij,t < 2x(θ0 + ΘS∗)2

θi + θj
or nij,t ≥(θi + θj)τij,t

2(θ0 + ΘS∗) ,
(17)

where τij,t := Pt−1
s=1 1{{i, j} ⊆Ss} simultaneously for all t ∈[T] and i ̸= j ∈[K].
876

Proof of Theorem 6. Let E be the high-probabality event of Lemmas 10 and 11 are satisfied,
877

so that P(E) ≥1 −4K2Te−x. First, note that since we have under the event E, bθucb
t
≤θucb
t
,
878

our procedure also satisfies the regret upper-bound
879

Regwtd
T
≤O(
p

θmaxKT log T)

of Theorem 5. Indeed, all upper-bounds of the proof of Theorem 5 remain valid upper-bounds
880

except the probability of the event Ec which is O(T −1) for x = 2 log T.
881

Let us now prove that we also have RT ≤O(K
√

T log T) with no asymptotic dependence on
882

θmax when T →∞.
883

26


---Page Break---
Then,
884

Regwtd
T
=

T
X

t=1
E

R(S∗, θ) −R(St, θ)

(18)

=

T
X

t=1
E

(R(S∗, θ) −R(St, θ))1{E} + (R(S∗, θ) −R(St, θ))1{Ec}


≤

T
X

t=1
E
h 
(R(St, bθucb
t
) −R(St, θ)) ∧R(S∗, θ)

1{E} + R(S∗, θ)1{Ec}
i
.

Then, using R(S∗, θ) ≤maxi ri ≤1, we get
885

Regwtd
T
≤

T
X

t=1
E
h 
(R(St, bθucb
t
) −R(St, θ)) ∧1

1{E} + 1{Ec}
i

≤4T 2K(K + 1)2e−x +

T
X

t=1
E
h 
R(St, bθucb
t
) −R(St, θ)

∧1

1{E}
i
.
(19)

Follow the proof of Theorem 5, we upper-bound the second term of the right-hand-side
886

of (19):
887

T
X

t=1
E
h 
R(St, bθucb
t
) −R(St, θ)

∧1

1{E}
i
(20)

=

T
X

t=1
E

min
j∈[K]

X

i∈St

ribθucb
i,t
1 + P

j∈St bθucb
j,t
−
riθi
1 + P
j∈St θj


∧1

1{E}


≤

T
X

t=1
E
 X

i∈St

ri(bθucb
i,t −θi)
θ0 + ΘSt


∧1

1{E}

because
X

i∈St

bθucb
i,t ≥ΘSt under E

≤

T
X

t=1
E
 X

i∈St

|bθucb
i,t −θi|
θ0 + ΘSt


∧1

1{E}

because ri ≤1

≤

K
X

i=1
E

"
T
X

t=1

|bθucb
i,t −θi|
θ0 + ΘSt
∧1

1{i ∈St}1{E}

#

≤

K
X

i=1
E

"
T
X

t=1

|γucb
ijt,tγucb
jt0,t −θi|
θ0 + ΘSt
∧1

1{i ∈St}1{E}

#

where jt = argmaxj∈St∪{0} θj, where the last inequality is by definition of bθucb
i,t .
Now,
888

from Lemma 10, paying an additive exploration cost to ensure that nij,t ≳x(1 + γij) and
889

nj0,t ≳x(1 + θj)2θj for all j ∈St such that θj ≥θ0. From Lemma 11, this is satisfied if for
890

some constant C > 0
891

τij,t > Cm2θ2
maxx .

Such a condidtion can be wrong for a couple (i, j) ∈S2
t at most during CK2m2θ2
maxx =
892

O(log T) rounds (since τij,t increases then). Thus, for C large enough,
893

T
X

t=1
E
h 
R(St, bθucb
t
) −R(St, θ)

∧1

1{E}
i

≤O(log T) +

K
X

i=1
E

"
T
X

t=1

|γucb
ijt,tγucb
jt0,t −θi|
θ0 + ΘSt
1{i ∈St, τijt,t ∧τjt,t ≥Cxm2θ2
max}1{E}

#

27


---Page Break---
≲O(log T) +

K
X

i=1
E

"
T
X

t=1

q

(γijt + 1)θix
s

(θi + θjt)

nijt,t
+

s

(1 + θj)

njt0,t



+ (γijt + 1)(θi + θjt)x

nijt,t
+ γijt(1 + θjt)2x

njt0,t

1{i ∈St}

θ0 + ΘSt

#

≤O(log T) +

K
X

i=1
E

"
T
X

t=1

q

(γijt + 1)θix
s

(θi + θjt)

nijt,t
+

s

(1 + θjt)

njt0,t

1{i ∈St}

θ0 + ΘSt

#

where the last inequality is because using that {i, jt, 0} ⊆St, we have
894

E

T
X

t=1

1 + θjt
(1 + ΘSt)njt0,t


= E

T
X

t=1

K
X

j=1

1{it ∈{j, 0}}

nj0,t
1{j = jt}

≤K(1 + log T).

and
895

E

T
X

t=1

θi + θjt
(1 + ΘSt)nijt,t


= E

T
X

t=1

K
X

j=1

1{it ∈{j, i}}

nj0,t
1{j = jt}

≤K(1 + log T).

Then, by Cauchy-Schwarz inequality we further get
896

T
X

t=1
E
h 
R(St, bθucb
t
) −R(St, θ)

∧1

1{E}
i

≲O(log T) +

K
X

i=1

v
u
u
tE

"
T
X

t=1

(γijt + 1)θi1{i ∈St}x

θ0 + ΘSt

#

(21)

×

v
u
u
tE

"
T
X

t=1

(θi + θjt)

nijt,t
+ (1 + θjt)

njt0,t

1{i ∈St}

θ0 + ΘSt

#

≲O(log T) +

K
X

i=1

v
u
u
tE

"
T
X

t=1

(γijt + 1)θi1{i ∈St}x

θ0 + ΘSt

#
p

K log T

≲O(log T) +

K
X

i=1

v
u
u
tE

"
T
X

t=1

θi1{i ∈St}x

θ0 + ΘSt

#
p

K log T (because γijt ≤1 by definition of jt)

≤O(K
p

Tx log T) = O(K
√

T log T) ,
(22)

where the last inequality is by Jensen’s inequality and the equality by setting x = 2 log T to
897

control the probability that Ec occurs. This concludes the proof.
898

28


---Page Break---
