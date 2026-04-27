Interpretable Decision Tree Search as a Markov
Decision Process

Anonymous Author(s)
Affiliation
Address
email

Abstract

Finding an optimal decision tree for a supervised learning task is a challenging
1

combinatorial problem to solve at scale. It was recently proposed to frame this
2

problem as a Markov Decision Problem (MDP) and use deep reinforcement learn-
3

ing to tackle scaling. Unfortunately, these methods are not competitive with the
4

current branch-and-bound state of the art. Instead, we propose to scale the res-
5

olution of such MDPs using an information-theoretic tests generating function
6

that heuristically, and dynamically for every state, limits the set of admissible
7

test actions to a few good candidates. As a solver, we show empirically that our
8

algorithm is at the very least competitive with branch-and-bound alternatives. As
9

a machine learning tool, a key advantage of our approach is to solve for multiple
10

complexity-performance trade-offs at virtually no additional cost. With such a set
11

of solutions, a user can then select the tree that generalizes best and which has the
12

interpretability level that best suits their needs, which no current branch-and-bound
13

method allows.
14

1
Introduction
15

Decision trees (DTs) remain the dominant machine learning model in applications where interpretabil-
16

ity is essential [Costa and Pedreira, 2023]. Thanks to recent advances in hardware, a new class of
17

decision tree learning algorithms returning optimal trees has emerged [Bertsimas and Dunn, 2017,
18

Demirovic et al., 2022, Mazumder et al., 2022]. These algorithms are based on a branch-and-bound
19

solver that minimizes a regularized empirical loss, where the number of nodes is used as a regularizer.
20

These optimization problems have long been known to be NP-Hard [Hyafil and Rivest, 1976] and
21

despite hardware improvements, solvers of such problems do not scale well beyond trees of depth 3
22

when attributes take continuous values [Mazumder et al., 2022]. On the other hand, greedy approaches
23

such as CART [Breiman et al., 1984] are still considered state-of-the-art decision tree algorithms
24

because they scale and offer more advanced mechanisms to control the complexity of the tree. By
25

framing decision tree learning as a sequential decision problem, and by carefully controlling the
26

size of the search space, we achieve in this paper a best of both worlds, solving the combinatorial
27

optimization problem with accuracies close to optimal ones, while improving scaling and offering a
28

better control of the complexity-performance trade-off than any existing optimal algorithm.
29

To do so, we formulate the problem of decision tree learning as a Markov Decision Problem (MDP,
30

[L. Puterman, 1994]) for which the optimal policy builds a decision tree. Actions in such an MDP
31

include tests comparing an attribute to a threshold (a.k.a. splits). This action space could include all
32

possible splits or a heuristically chosen subset, yielding a continuum between optimal algorithms and
33

heuristic approaches. Furthermore, the reward function of the MDP encodes a trade-off between the
34

complexity and the performance of the learned tree. In our work, complexity takes the meaning of
35

simulatability [Lipton, 2018], i.e. the average number of splits the tree will perform on the train dataset.
36

The MDP reward is parameterized by α, trading-off between train accuracy and regularization. One
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
of the main benefits of our formulation is that the biggest share of the computational cost is due to
38

the construction of the MDP transition function which is completely independent of α, allowing us to
39

find optimal policies for a large choice of values of α at virtually no additional cost.
40

Branch-and-Bound (BnB) algorithms similarly optimize a complexity performance trade-off
41

[Demirovic et al., 2022, Mazumder et al., 2022] but require the user to provide the maximum
42

number of test nodes as an input to their algorithm. Providing such a value a priori is difficult
43

since a smaller tree (e.g. with 3 test nodes) might be only marginally worse on a given dataset than
44

a larger tree (e.g. with 15 test nodes) with respect to the training accuracy but might generalize
45

better or be deemed more interpretable a posteriori by the user. As such, it is critical to consider
46

the multi-objective nature of the optimization problem and seek algorithms returning a set of trees
47

that are located on the Pareto front of the complexity-performance trade-off. To the best of our
48

knowledge, this has been so far neglected by BnB approaches. None of the BnB implementations
49

return a set of trees for different regularizer weights unlike greedy algorithms like CART or C4.5
50

that can return trees with different complexity-performance trade-offs using minimal complexity
51

post-pruning [Breiman et al., 1984], making it a more useful machine learning tool in practice.
52

2
Related Work
53

2.1
Optimal Decision Trees.
54

Decision tree learning has been formulated as an optimization problem in which the goal is to
55

construct a tree that correctly fits the data while using a minimal number of splits. In [Bertsimas
56

and Dunn, 2017, Aghaei et al., 2020, Verwer and Zhang, 2019], decision tree learning is formulated
57

as a Mixed Integer Program (MIP). Instead of using a generic MIP solver, [Demirovic et al., 2022,
58

Mazumder et al., 2022] design specialized solvers based on the Branch-and-Bound (BnB) principle.
59

Quant-BnB [Mazumder et al., 2022] is currently the latest work in this line of research for datasets
60

with continuous attributes and is considered state-of-the-art. However, direct optimization is not a
61

convenient approach since finding the optimal tree is known to be NP-Hard [Hyafil and Rivest, 1976].
62

Despite hardware improvements, Quant-BnB does not scale beyond trees depth of 3. To reduce the
63

search space, optimal decision tree algorithms on binary datasets, such as MurTree, Blossom and
64

Pystreed [Demirovic et al., 2022, Demirovi´c et al., 2023, van der Linden et al., 2023], employ
65

heuristics to binarize a dataset with continuous attributes during a pre-processing step following
66

for example the Minimum Description Length Principle [Rissanen, 1978]. The tests generating
67

function of our MDP formulation is similar in principle except that it is state-dependent, which, as
68

demonstrated experimentally, greatly improves the performance of our solver.
69

2.2
Greedy approaches.
70

Greedy approaches like CART iteratively partition the training dataset by taking the most informative
71

splits in the sense of the Gini index or the entropy gain. CART is only one-step optimal but can
72

scale to very deep trees. This might lead to overfitting and algorithms such as Minimal Complexity
73

Post-Pruning (see Section 3.3 from [Breiman et al., 1984]) iteratively prune the deep tree, returning
74

a set of smaller trees with decreasing complexity and potentially improved generalization. The
75

trees returned by our algorithms provably dominate—in the multi-objective optimization sense—all
76

the above smaller trees in terms of train accuracy vs. average number of tests performed, and we
77

experimentally show that they often generalize better than the trees returned by CART.
78

2.3
Markov Decision Problem formulations.
79

In [Topin et al., 2021], a base MDP is extended to an Iterative Bounding MDP (IBMDP) allowing
80

the use of any Deep Reinforcement Learning (DRL) algorithm to learn DT policies solving the
81

base MDP. While more general and scalable, this method is not state-of-the-art for learning DTs for
82

supervised learning tasks. Prior to IBMDPs, [Garlapati et al., 2015] formulated the learning of DTs
83

for classification tasks with ordinal attributes as an MDP. To be able to handle continuous features,
84

[Nunes et al., 2020] used Monte-Carlo tree search [Kocsis and Szepesvári, 2006] in combination
85

with a tests generating function that limits the branching factor of the tree. Our MDP formulation is
86

different as it considers a regularized objective while [Nunes et al., 2020] optimize accuracy on a
87

validation set. Our tests generating function is also different and dramatically improves scaling as
88

2


---Page Break---
shown in the comparison of Sec. 5.1.1, making our algorithm competitive with BnB solvers, while
89

[Nunes et al., 2020] only compared their algorithm against greedy approaches. A comparison of our
90

method with other MDP approaches is presented in the supplementary material.
91

2.4
Interpretability of Decision Trees.
92

The interpretability of a decision tree is usually associated with its complexity, e.g. its depth or its
93

total number of nodes. For trees with 3 to 12 leaves, [Piltaver et al., 2016] observed a strong negative
94

correlation between the number of leaves in a tree and a “comprehensibility” score given by users.
95

Most of the literature considers the total number of test nodes as its complexity measure, but other
96

definitions of complexity exist. [Lipton, 2018] coined the term simulatability, which is related to the
97

average number of tests performed before taking a decision. This quantity naturally arises in our
98

MDP formulation. We show in a qualitative study that both criteria are often correlated but on some
99

datasets, DPDT returns an unbalanced tree with more test nodes that are only traversed by a few
100

samples.
101

3
Decision Trees for Supervised Learning
102

Let us consider a training dataset D = {(xi, yi)}i∈{1,...,N}, made of (data, label) pairs, (xi, yi) ∈
103

(X, Y ), where X ⊆Rp. A decision tree T sequentially applies tests to xi ∈X before assigning it a
104

value in Y , which we denote T(xi) ∈Y . The tree has two types of nodes: test nodes that apply a
105

test and leaf nodes that assign a value in Y . A test compares the value of an attribute with a given
106

threshold value, x.,2 ≤3". In this paper, we focus on binary decision trees, where decision nodes
107

split into a left and a right child with axis aligned splits as in [Breiman et al., 1984]. However, all our
108

results generalize straitghforwardly to tests involving functions of multiple attributes. Furthermore,
109

we look for trees with a maximum depth D, where D is the maximum number of tests a tree can
110

apply to classify a single xi ∈X. We let TD be the set of all binary decision trees of depth ≤D.
111

Given a loss ℓdefined on Y × Y we look for trees in TD satisfying
112

T ∗= argmin
T ∈TD
Lα(T),
(1)

= argmin
T ∈TD

1
N

N
X

i=0
ℓ(yi, T(xi)) + αC(T),
(2)

where C : T →R is a function that quantifies the complexity of a tree. It could be the number
113

of nodes as in [Mazumder et al., 2022]. In our work, we are interested in the expected number of
114

tests a tree applies on any arbitrary data x ∈D. As for ℓ, in a regression problem Y ⊂R and
115

ℓ(yi, T(xi)) can be (yi −T(xi))2. For supervised classification problems, Y = {1, ..., K}, where K
116

is the number of class labels, and ℓ(yi, T(xi)) = 1{yi̸=T (xi)}. In our work, we focus on supervised
117

classification but the MDP formulation extends naturally to regression.
118

4
Decision Tree Learning as an MDP
119

Our approach encodes the decision tree learning problem expressed by Eq. (2) as a finite horizon
120

Markov Decision Problem (MDP) ⟨S, A, Rα, P, D⟩. We present this MDP for a supervised clas-
121

sification problem with continuous features, but again, our method extends to regression and to
122

other types of features. The state space of this MDP is made of subsets X of the dataset D as well
123

as a depth value d: S = {(X, d) ∈P(D) × {0, ..., D}}, where P(D) is the power set of D. Let
124

F = {f : f(.) = 1{.≤xij}, ∀i ∈{1, ..., N}, ∀j ∈{1, ..., p}} be a set of binary functions. We
125

consider only tests that compare attributes to values within the dataset because comparing attributes to
126

other values cannot further reduce the training objective. The action space A of the MDP is then the set
127

of all possible binary tests as well as class assignments: A = F ∪{1, ..., K}. When taking an action
128

a ∈F, the MDP will transit from state (X, d) to either its “left" state sl = (Xl, d + 1) or its “right"
129

state sr = (Xr, d+1). In particular the MDP will transit to sl = ({(xi, yi) ∈X : a(xi) = 1}, d+1)
130

with probability pl = |Xl|

|X| or to sr = (X \ Xl, d + 1) with probability pr = 1 −pl. Furthermore,
131

to enforce a maximum tree depth of D, whenever a state is s = (., D) then only class assignment
132

actions are possible in s. When taking an action in {1, ..., K} the MDP will transit to a terminal state
133

3


---Page Break---
denoted sdone that is absorbing and has null rewards. The reward of taking an action a in state s is
134

given by the parameterized mapping Rα : S × A →R that enforces a trade-off between the expected
135

number of tests and the classification accuracy. It is defined by:
136

Rα(s, a) = Rα((X, d), a),

=






−α,
if a ∈F,
−1

|X|
P

yi∈X
1yi̸=a
if a ∈{1, ..., K}.

The complexity-performance trade-off is encoded by the value 0 ≤α ≤1, which is the price to
137

pay to obtain more information by testing a feature. A more detailed study of the trade-off is given
138

in section 6.4. The maximum depth parameter D is a time horizon, i.e. the number of actions it is
139

possible to take in one episode. An algorithm solving such an MDP can always return a deterministic
140

policy [L. Puterman, 1994] of the form: π : S →A that maximizes the expected sum of rewards
141

during an episode:
142

π = argmax
π
Jα(π),
(3)

Jα(π) = E

" D
X

t=0
Rα(st, π(st))

#

,
(4)

where the expectation is w.r.t. random variables st+1 ∼P(st, π(at)) with initial state s0 = (D, 0).
143

From deterministic policy to binary DT. One can transform any deterministic policy π of the above
144

MDP into a binary decision tree T with a simple extraction routine E(π, s), where s ∈S is a state.
145

E is defined recursively in the following manner. If π(s) is a class assignment then E(π, s) returns a
146

leaf node with class assignment π(s). Otherwise E(π, s) returns a binary decision tree that has a test
147

node π(s) at its root, and E(π, sl) and E(π, sr) as, respectively, the left and right sub-trees of the
148

root node. To obtain T from π, we call E(π, s0) on the initial state s0 = (D, 0).
149

Equivalence of objectives. When the complexity measure C of Lα is the expected number of tests
150

performed by a decision tree, the key property of our MDP formulation is that finding the optimal
151

policy in the MDP is equivalent to finding T ∗, as given by the following proposition
152

Proposition 1: Let π be a deterministic policy of the MDP and π∗one of its optimal deterministic
153

policies, then Jα(π) = −Lα(E(π, s0)) and T ∗= E(π∗, s0).
154

The proof is given in the Appendix H.
155

5
Algorithm
156

We now present the Dynamic Programming Decision Tree (DPDT) algorithm. The algorithm is made
157

of two essential steps. The first and most computationally expensive step constructs the MDP of
158

Section 4. The second step is to solve it to obtain policies maximizing Eq.(4) for different values of
159

α. Both steps are now detailed.
160

5.1
Constructing the MDP
161

An algorithm constructing the MDP of Section 4 essentially computes the set of all possible decision
162

trees of maximum depth D whose decision nodes are in F. This specific MDP is a directed acyclic
163

graph. Each node of this graph corresponds to a state for which one computes the transition and
164

reward functions. To limit memory usage of non-terminal nodes, instead of storing all the samples
165

in (X, d), we only store d and the binary vector of size N, xbin = (1{xi∈X})i∈{1,...,N}. Even then,
166

considering all possible splits in F will not scale. We thus introduce a state-dependent action space
167

As, much smaller than A and populated by the tests generating function.
168

5.1.1
Tests generating functions
169

A tests generating function is any function ϕ of the form ϕ : S →P(F), where P(F) is the power
170

set of all possible data splits F. For a state s ∈S, the state-dependent action space is defined by
171

As = ϕ(s) ∪{1, ..., K}. Because for a given state s we might have that ϕ(s) ̸= F, solving the
172

4


---Page Break---
101
102
103
104
105
106

Number of MDP states

0.970

0.975

0.980

0.985

0.990

Accuracy

Exhaustive
TOP 55
TOP 10
TOP 5
CART (TOP 1)
DPDT-2
DPDT-3
DPDT-4

Figure 1: Comparison of DPDT algorithm on the Iris dataset in
terms of the number of states in the MDP when using different
tests generating functions. “TOP B” are tests function return-
ing the B most informative splits for each state. “Exhaustive”
returns all possible states (equivalent to the search space of
Quant-BnB). DPDT-Dcart are the tests functions that make
calls to the CART algorithm.

MDP with state-dependent actions As is not guaranteed to yield the minimizing tree in Eq. (2), as
173

optimization is now carried on a subset of TD. In this section, we compare different choices of ϕ on a
174

sufficiently small dataset such that ϕ(s) = F, ∀s ∈S remains tractable. As a baseline, we use a tests
175

generating function proposed in [Nunes et al., 2020], and compare with our proposed ϕ in terms of
176

quality of the best tree vs. size of the MDP.
177

Algorithm 1: DPDT-K MDP generation
Data: Dataset D, max depth D
Result: Decision Tree Search MDP
d ←0
s0 ←[D, d]
MDP.AddState(s0) # MDP of Sec. 4
while d < D do

# For all states at the current
depth d
for s = (Ds, ds) ∈MDP s.t. ds = d do

# Test generating function
follwing Sec. 5.1.1
Tcart ←CART(Ds, maxdepth=K)
As ←ExtractSplits(Tcart)
for a ∈As do

# MDP expansion
follwoing Sec. 4
MDP.AddRewardAndTransition(s, a)
MDP.AddStates(NextStates(s, a))
end
end
d ←d + 1;
end

Exhaustive function. When ϕ(s) = F, ∀s ∈S, the
178

MDP contains all possible data splits. In this case,
179

the MDP ‘spans’ all trees of depth at most D and the
180

solution to Eq. (4) will be the optimal decision tree of
181

Eq. (2). In this case, the number of states in the MDP
182

would be of the order of
D−1
P

d=0
K(2Np)d which scales
183

exponentially with the maximum depth of the tree:
184

this limits the learning to very shallow trees (D ≤3)
185

as discussed in [Mazumder et al., 2022]. The goal
186

of a more heuristic choice of ϕ is to have a maximal
187

number of splits B = maxs∈S |ϕ(s)| that is orders
188

of magnitude smaller than that of the exhaustive case
189

|F| = Np such that the size of the MDP, which is
190

now in the order of
D−1
P

d=0
K(2B)d, remains tractable
191

for deeper trees.
192

Top B most informative splits. [Nunes et al., 2020]
193

proposed to generate tests with a function that returns
194

for any state s = (X, d) the B most informative splits
195

over X in the sense of entropy gain. In practice, we
196

noticed that the returned set of splits lacked diversity
197

and often consists of splits on the same attribute with minor changes to the threshold value. While
198

this still leads to improvements over greedy methods—as shown in the study presented next—it is at
199

the expense of a much larger MDP, i.e., search space.
200

Top B most discriminative splits. Instead of returning the most informative splits, we propose at
201

every state s = (X, d), to find the most discriminative splits, i.e. the attribute comparisons with which
202

one can best predict the class of data points in X. This is similar to the minimum description length
203

principle used in [Demirovic et al., 2022] that transforms a dataset with continuous attributes to a
204

binary dataset. However, we perform this transformation dynamically at every state while building
205

the MDP. In practice, this amounts to calling CART with a maximum depth Dcart (a hyperparameter
206

of DPDT) on every state s, and using the test nodes of the tree returned by CART as ϕ(s).
207

While restricting the action space at a given state s to the actions of the tests generating function ϕ(s)
208

loses the guarantees of finding T ∗, we are still guaranteed to find trees better than those of CART:
209

Proposition 2: Let π∗be an optimal deterministic policy of the MDP, where the action space at every
210

state is restricted to the top B most informative or discriminative splits. Let T0 be the tree learned by
211

CART and {T1, . . . , TM} be the set of trees returned by postprocessing pruning on T0, then for any
212

α > 0, Lα(E(π∗, s0)) ≤min0≤i≤M Lα(Ti).
213

5


---Page Break---
The proof for Prop. 5.1.1 follows from the fact that policies generating the tree returned by CART
214

and all of its sub-trees (which is a superset of the trees returned by the pruning procedure) are
215

representable in the MDP and by virtue of the optimality of π∗and the equivalence in Prop. 4, are
216

worse in terms of regularized loss Lα than the tree E(π∗, s0). The consequences of Prop. 5.1.1
217

are clearly observed experimentally in Fig. 3. While this proposition holds for the latter two test
218

generating functions, in practice, the tests returned by our proposed function are of much higher
219

quality as discussed next.
220

Comparing tests generating functions. We conduct a small study comparing the exhaustive ϕ
221

(labeled “Exhaustive”) against the ϕ proposed in [Nunes et al., 2020] (labeled “Top B”) and the one
222

used in our algorithm (labeled “DPDT-K”, where K is the maximum depth given to CART), on the Iris
223

dataset. Figure 1 shows that while the latter two ϕ generalize the greedy approach (labeled “CART”),
224

DPDT scales much more gracefully than when using the ϕ of [Nunes et al., 2020]. With Dcart = 4,
225

DPDT-4 finds the optimal tree in an MDP having several orders of magnitude less states (a few
226

hundreds vs a few millions) than the one built using the exhaustive ϕ. This favorable comparison
227

against exhaustive methods also holds for larger datasets as shown in Sec. 6.2.
228

The MDP construction of DPDT-K using the tests generating function is explained in Alg. 1. Starting
229

from s0, the state containing the whole dataset, CART with a maximum depth of K is called which
230

generates a tree with up to 2K −1 split nodes. These splits are what constitutes As0, the set of binary
231

tests admissible at s0. For every such action, we compute the reward and transition probabilities to a
232

set of new states at depth 1. This process is then iterated for every state at depth 1, calling CART with
233

the same maximum depth of K on each of the states at depth 1, generating a new set of binary tests As
234

for each of these states s and so on until reaching the maximum depth. Upon termination of Alg. 1,
235

we compute the rewards for labelling actions at every state and we call the dynamic programming
236

routine below to extract the optimal policy.
237

5.2
Dynamic Programming
238

Having built the MDP, we backpropagate using dynamic programming the best optimal actions from
239

the terminal states to the initial states. We use Bellman’s optimality equation to compute the value of
240

the best actions recursively:
241

Q∗(s, a) = E
h
rd+1 + max
a′ Q∗(sd+1, a′)|sd = s, ad = a
i
,

=
X

s′
P(s, a, s′)
h
R(s, a) + max
a′ Q∗(s′, a′)
i
.

Pareto front.
As our reward function is a linear combination of the complexity and performance
242

measures, we can reach any tree “spanned” by the MDP that lies on the convex hull of the Pareto
243

front of the complexity-performance trade-off. In DPDT, we compute the optimal policy for several
244

choices of α using a vectorial representation of the Q-function that now depends on α:
245

Q∗(s, a, α) =
X

s′
P(s, a, s′)
h
Rα(s, a) + max
a′ Q∗(s′, a′, α)
i
.

We can then find all policies greedy w.r.t. Q∗π∗(s, α) = argmax
a∈A
Q∗(s, a, α). Such policies satisfy
246

Eq. (4) for any value of α. Given a set of values of α in [0, 1], we can compute in a single backward
247

pass Q∗(s, a, α) and π∗(s, α) and return a set of trees, optimal for different values of α (see Fig.7 for
248

an illustrative example). In practice, the computational cost is dominated by the construction of the
249

MDP 1 and one can promptly back-propagate the Q-values of over 103 values of α.
250

6
Experiments
251

In this section we study DPDT from different perspectives. First, in Sec. 6.2, we study DPDT in
252

terms of its performance as a solver for the combinatorial optimization problem of Eq. (2). Here, we
253

focus on smaller problems (maximum depth ≤3) in which the optimal solution can be computed
254

by Branch-and-Bound (BnB) algorithms. In this first set of experiments, we only report the training
255

accuracy vs. the wall-clock time as done in prior work [Mazumder et al., 2022]. Then we study DPDT
256

6


---Page Break---
for model selection (Sec. 6.3). From the perspective of the end user, a decision tree algorithm may
257

be used for selecting either a tree that generalizes well to unseen data or a tree that is interpretable.
258

We compare classification of unseen data of trees obtained by DPDT to other baselines described
259

below. Then, we plot the train accuracy of trees learned by CART and DPDT as a function of their
260

complexity to observe how a user can choose the complexity-performance trade-off. We use the 16
261

classification datasets with continuous attributes experimented with in [Mazumder et al., 2022].
262

When considering other optimal BnB baselines [Demirovic et al., 2022, van der Linden et al., 2023],
263

two problems arise for fair comparison with DPDT in terms of model selection. First, to obtain a set
264

of tree from such baselines, the optmization algorithms need to be ran as many times as trees wanted
265

by the user. For example, one can obtain a set of trees of depth ≤5 by running MurTree 25 times
266

with different maximum number of test nodes allowed in the learned trees. This could require up
267

to 25 times the runtime of a single optimization. Second, MurTree and Pystreed [Demirovic et al.,
268

2022, van der Linden et al., 2023] require binary attributes. Learned trees are not comparable directly
269

with trees trained on continuous attributes because each tree node testing a binary feature actually
270

does at least two tests on the original continuous feature (see Appendix F.1 or Appendix D1 from
271

[Mazumder et al., 2022]). DPDT is coded in Python and the code is available in the supplementary
272

material. All experiments are run on a single core from a Intel i7-8665U CPU. All the links to
273

code used for the baselines are given in the Appenix A
274

6.1
Baselines
275

Quant-BnB. [Mazumder et al., 2022] propose a scalable BnB algorithm that returns optimal trees.
276

We emphasize that Quant-BnB is not meant to scale beyond tree depths of 3 (explicitly stated in the
277

Quant-BnB paper) and the authors’ implementation of Quant-BnB does not support learning trees of
278

depth > 3.
279

MurTree, Pystreed. To use [Demirovic et al., 2022, van der Linden et al., 2023] with continuous
280

features datasets, the minimum length description principle is used to obtain bins in a continuous
281

feature domain, then a one hot encoding is applied to binarize the binned dataset. This can result in
282

datasets with more than 500 features. As MurTree and Pystreed memory scales with the square of
283

number of binary attributes, using those algorithms to find trees of depths greater than 3 often results
284

in Out Of Memory (OOM) errors.
285

Deep Reinforcement Learning. We use Custard [Topin et al., 2021] as a DRL baseline. Custard
286

has two hyperparameters: the DRL algorithm to learn a policy in the IBMDP and a tests generating
287

function that gives p tests per feature. In our experiments, Custard-5 and Custard-3 correspond to
288

DQN agents [Mnih et al., 2015] that can test each dataset attribute against 5 or 3 values respectively.
289

CART [Breiman et al., 1984] is a greedy algorithm that can build suboptimal trees for any dataset.
290

6.2
Optimality gap
291

Because we use a tests generating function that heuristically reduces the search space, a first question
292

we want to investigate is how good is our solver for the combinatorial problem of decision tree search.
293

To do so, we focus on max depth 3 problems for which T ∗can be computed exactly using Quant-BnB
294

[Mazumder et al., 2022]. As Quant-BnB has a different complexity regularization (number of nodes in
295

the tree) than DPDT (average number of tests per classified data), we set the complexity regularizing
296

term α to 0 to allow direct comparisons. This does not create an artificial learning and on 14 out of
297

16 datasets, trees with α = 0 generalize best, and second best on the remaining 2. That is because at
298

depth 3 the risk of overfitting is small.
299

We run DPDT with calls to CART with maximum depth 4 or 5 as a tests generating function (DPDT-4
300

and DPDT-5 respectively). Quant-BnB is first run without a time limit to obtain optimal decision trees
301

w.r.t. Eq.(2). Quant-BnB is also run a second time with a time limit equal to DPDT-5’s runtime (we
302

also added in the supplementary material results for Quant-BnB-T+5 and Quant-BnB-T+50 that add
303

extra seconds to Quant-BnB-T). CART is run with the maximum depth set to 3 and the information
304

gain based on entropy. All algorithms are run on the same hardware. Custard is run 5 times per dataset
305

because it is a stochastic algorithm. We use stable-baselines3 implementation of DQN [Raffin et al.,
306

2021] with default hyperparameters. A Custard run usually takes 10 minutes. We provide learning
307

curves in Fig. 4. The key result from Table 1 is that DPDT-5 has better train accuracies than the
308

7


---Page Break---
<= -5
[-2.5:0]
[0:2.5]
[2.5:5]
[5:7.5]
[7.5:10]
[10:12.5] [12.5:15]
>15
Accuracy gain over CART

0

2

4

6

8

10

12

14

Number of datasets

DPDT-2
DPDT-3
Quant-BnB
MurTree
Pystreed

<=-3
[-2:-2.5]
[-1.5:-2]
[-1:-1.5]
[-0.5:-1]
[0:-0.5]
> 0.5
[-2.5:-3]

DPDT trees average #tests gain over CART trees

0

1

2

3

4

5

6

Number of datasets

DPDT-2
DPDT-3

Figure 2: Performance gain of DTs over CART trees. Left, accuracy on unseen data gain of trees
with depth ≤5 selected with procedure of Sec. 6.3. Right, average number of tests of those trees.

other non-greedy methods when run in similar runtimes across all classification tasks. Furthermore,
309

the train accuracy gaps between the optimal decision trees obtained from Quant-BnB, in sometimes
310

several hours, and DPDT are usually small (the maximum gap is 1.5% for the bean dataset).
311

Table 1: Train accuracy of decision tree algorithms. The “Quant-BnB” columns correspond to
results for Quant-BnB with no time limit, i.e returing the optimal tree. The “Quant-BnB-T” column
corresponds to results for Quant-BnB run for as long as DPDT-5. The “Greedy” columns correspond
to CART with maximum depth of 3.

Datasets
Accuracy (train %) of depth-3 trees
Runtime in seconds
Names
Samples
p
Classes
Quant-BnB
Quant-BnB-T
DPDT-5
DPDT-4
Custard-5
Custard-3
Greedy
Quant-BnB
DPDT-5
DPDT-4
Custard-5
Custard-3
Greedy
avila
10430
10
12
58.5
57.3
58.5∗
58
40.9 ± 0.6
41 ± 0.3
53.8
4188
5.645
2.14
553
632
0.031
bank
1097
4
2
98.3
97.1
98
98
49.6 ± 1.6
35.5 ± 20.9
95.3
4.4
0.158
0.142
648
661
0.003
bean
10888
16
7
87.1
85.3
85.6
85
18.2 ± 2.3
19.2 ± 4.1
80.5
1014
16.194
5.836
697
687
0.114
bidding
5056
9
2
99.3
98.6
99.3∗
99.3∗
81 ± 4.4
79.4 ± 2.1
98.2
30
0.545
0.377
693
671
0.006
eeg
11984
14
2
70.8
68.3
70.3
70
54.9 ± 0.1
54.8 ± 0.5
66.6
4042
8.927
3.032
692
682
0.023
fault
1552
27
7
68.2
64.6
68
65.7
30.3 ± 1.4
27.5 ± 8.6
55.3
−
2.46
1.243
720
711
0.015
htru
14318
8
2
98.1
98
98
98
86 ± 0.9
59.4 ± 31.7
97.9
10303
11.316
4.246
690
684
0.055
magic
15216
10
2
83.1
82.6
83
82.7
58.1 ± 4.3
58.5 ± 3.0
79.3
1090
14.838
5.443
685
675
0.069
occupancy
8143
5
2
99.4
99.3
99.4∗
99.3
64.7 ± 0.5
65.1 ± 8.3
99.1
106
1.458
0.786
687
664
0.008
page
4378
10
5
97.1
96.5
97
97.0
90.2 ± 0.4
88.3 ± 4.8
96.3
471
2.859
1.29
708
687
0.01
raisin
720
7
2
89.4
88.1
88.5
88.3
50.9 ± 2.2
49.7 ± 1.0
86.9
167
0.501
0.3
668
667
0.003
rice
3048
7
2
93.8
93.7
93.7
93.6
51.9 ± 0.9
48.1 ± 3.4
93.0
1340
2.004
0.809
668
666
0.01
room
8103
16
4
99.2
98.8
99.2∗
99.2∗
71.5 ± 3.4
67.6 ± 5.6
97.7
180
2.714
1.884
1362
1389
0.01
segment
1848
18
7
88.7
79.1
88.2
88.2
13.7 ± 0.5
13.9 ± 0.5
81.6
153
0.771
0.397
812
761
0.009
skin
196045
3
2
96.9
96.7
96.7
96.7
61.2 ± 2.2
62.2 ± 8.7
96.6
350
48.894
19.239
752
745
0.082
wilt
4339
5
2
99.6
99.4
99.5
99.5
98.4 ± 0.2
98.3 ± 0.1
99.1
67
0.582
0.352
663
610
0.008

6.3
Selecting the best tree for unseen data
312

We now investigate whether DPDT is suited for model selection i.e. whether DPDT can identify an
313

accurate decision tree that will generalize well to unseen data for a given classification task. We used
314

the following model selection procedure for each classification task. First, we learn a set of decision
315

trees of depth D ≤5 with DPDT-3, DPDT-2, and CART on a training set using different values of
316

α for DPDT or minimal complexity post-pruning for CART. Because Quant-BnB simply cannot
317

compute trees of depth > 3, we only report the accuracy on unseen data of Quant-BnB trees from
318

Table 1. Because the BnB baselines MurTree and Pystreed are not designed to return a set of trees,
319

we brute force the computation of at most 25 trees from each by setting the maximum tree nodes
320

parameter to 0, ..., 25 −1. Then, for each baseline we evaluate each learned tree (only one tree for
321

Quant-BnB) on a test set and select the tree with highest test accuracy. Fig 2 reports the number of
322

datasets for which each baseline has better generalization performances than CART, and the number
323

of datasets for which DPDT-K returned trees performing less tests on average than CART trees. A
324

table with accuracies of the selected trees on a validation set, the runtime in seconds to obtain the set
325

of trees to select from, and the average number of tests performed on data in Appendix. All BnB
326

baselines required more than 5 minutes to generate a single tree. As such, the runtime for BnB to
327

obtain the whole set of trees is order of magnitudes higher than CART and DPDT. DPDT learns a set
328

of trees of at most depth 5 on the complexity-performance convex-hull in seconds which highlights
329

its ability to scale to non-shallow trees. For that purpose, DPDT built the MDP of possible solution
330

trees of at most depth 5 using CART as a tests generating function, and backpropagated state-action
331

values for 1000 different α.
332

After applying the above selection procedure, we see on Table 2 that DPDT generalized better than
333

CART on 10 out of 16 datasets while CART outperformed DPDT on only one dataset. When accuracy
334

on test data for CART is already close to 100%, our approach can of course not largely outperform it.
335

However, the benefits of our method have to also be appreciated in terms of gains in average number
336

8


---Page Break---
of tests. We can see that when CART does not generalize well, our method can have clear gains in
337

generalization (e.g. avila, eeg and fault). Otherwise, when CART is close to 100% accuracy, our
338

method can achieve similar results with less tests. In raisin, rice and room we need two fewer tests
339

which is substantial when tests are expensive, e.g. an MRI scan when testing patients.
340

6.4
Selecting the most interpretable tree
341

0
1
2
3
4
5
average tests per sample

0.55

0.60

0.65

0.70

0.75

accuracy

eeg

DPDT-3 Train in 57.038s
DPDT-2 Train in 4.347s
CART-PP Train in 1.25s

0
1
2
3
4
5
average tests per sample

0.4

0.5

0.6

0.7

0.8

accuracy

fault

DPDT-3 Train in 35.185s
DPDT-2 Train in 2.611s
CART-PP Train in 0.913s

Figure 3: Complexity-performance trade-offs of CART and DPDT on two different classification
datasets. CART returns a set of trees with the minimal complexity post-pruning algorithm. DPDT
returns a set of trees by returning policies for 1000 different α.

In this section, we show how a user can use DPDT to select a tree with complexity preferences. In
342

Figure 3, we plot the trade-offs of trees returned by CART and DPDT. The trade-off is between
343

accuracy and average number of tests. Because this is the trade-off that DPDT optimizes and because
344

the trees of CART are “spanned” by the MDP created by DPDT, all trees returned by DPDT will
345

dominate in the multi-objective sense trees returned by CART. This is well demonstrated in practice by
346

Figure 3 where the curve of DPDT is always above that of CART. Learned trees and their accuracies
347

as functions of number of nodes and tests are presented in Appendices 5 8 9. Finally, decision tree
348

search being a combinatorial problem, there are always limits to scalability. In Appendix 6 we scale
349

up to a tree depth of 10 by running DPDT-2 up to a depth of 6 then switch to DPTD-1 (i.e. greedy)
350

thereafter. The rationale is that a non-greedy approach is more critical closer to the root.
351

7
Limitations, Future Work, and Conclusion
352

Limitations. In our opinion, both the strength and the weakness of DPDT come from the choice of
353

the tests generating function. If the tests generating function generates too much tests in each MDP
354

state, the runtime will grow and there is a risk for out-of-memory errors. This can be alleviated with
355

parallelizing (expanding MDP states on different processes) and caching (only expand unseen MDP
356

states), similar to [Demirovic et al., 2022]. A rule of thumb for running DPDT on personal CPUs is
357

to choose a tests generating function resulting in an MDP with at most 106 states.
358

Future Work. DPDT could scale to bigger datasets by combining Custard [Topin et al., 2021] with
359

tests generating functions and tabular deep learning techniques [Kossen et al., 2021]. The latter is a
360

promising research avenue. The transformer-based architecture from [Kossen et al., 2021] takes a
361

whole train dataset as input and learns representations taking in account relationships between all
362

training samples and all labels. Test actions are then the output of such a neural architecture: the tests
363

generating function is learned.
364

Conclusion. In this work we solve MDPs whose optimal policies are decision trees optimizing a trade-
365

off between tree accuracy and complexity. We introduced the Dynamic Programming Decision Tree
366

algorithm that returns several optimal policies for different reward functions. DPDT has reasonable
367

runtimes and is able to scale to trees with depth greater than 3 using information-theoretic tests
368

generating functions. To the best of our knowledge, DPDT is the first scalable decision tree search
369

algorithm that runs fast enough on continuous attributes to be an alternative to CART for model
370

selection of any-depth trees. DPDT is a promising research avenue for new algorithms offering
371

human users a greater control than CART over tree selection in terms of generalization performance
372

and interpretability.
373

9


---Page Break---
References
374

Sina Aghaei, Andres Gomez, and Phebe Vayanos. Learning optimal classification trees: Strong
375

max-flow formulations, 2020.
376

Dimitris Bertsimas and Jack Dunn. Optimal classification trees. Machine Learning, 106:1039–1082,
377

2017.
378

Leo Breiman, Jerome Friedman, R.A. Olshen, and Charles J. Stone. Classification And Regression
379

Trees. Taylor and Francis, New York, 1984.
380

Vinicius G Costa and Carlos E Pedreira. Recent advances in decision trees: An updated survey.
381

Artificial Intelligence Review, 56(5):4765–4800, 2023.
382

Emir Demirovic, Anna Lukina, Emmanuel Hebrard, Jeffrey Chan, James Bailey, Christopher Leckie,
383

Kotagiri Ramamohanarao, and Peter J. Stuckey. Murtree: Optimal decision trees via dynamic
384

programming and search. Journal of Machine Learning Research, 23(26):1–47, 2022. URL
385

http://jmlr.org/papers/v23/20-520.html.
386

Emir Demirovi´c, Emmanuel Hebrard, and Louis Jean. Blossom: an anytime algorithm for com-
387

puting optimal decision trees. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara
388

Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International
389

Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research,
390

pages 7533–7562. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/
391

demirovic23a.html.
392

Abhinav Garlapati, Aditi Raghunathan, Vaishnavh Nagarajan, and Balaraman Ravindran. A rein-
393

forcement learning approach to online learning of decision trees, 2015.
394

Laurent Hyafil and Ronald L. Rivest. Constructing optimal binary decision trees is np-complete.
395

Information Processing Letters, 5(1):15–17, 1976. ISSN 0020-0190. doi: https://doi.org/10.1016/
396

0020-0190(76)90095-8. URL https://www.sciencedirect.com/science/article/pii/
397

0020019076900958.
398

Levente Kocsis and Csaba Szepesvári. Bandit based monte-carlo planning. In European conference
399

on machine learning, pages 282–293. Springer, 2006.
400

Hecotr Kohler, Riad Akrour, and Philippe Preux. Limits of actor-critic algorithms for decision tree
401

policies learning in ibmdps, 2023.
402

Jannik Kossen, Neil Band, Clare Lyle, Aidan N Gomez, Thomas Rainforth, and Yarin Gal. Self-
403

attention between datapoints: Going beyond individual input-output pairs in deep learning. Ad-
404

vances in Neural Information Processing Systems, 34:28742–28756, 2021.
405

Martin L. Puterman, editor. Markov Decision Processes: Discrete Stochastic Dynamic Programming.
406

John Wiley & Sons, Hoboken, 1994.
407

Zachary C. Lipton. The mythos of model interpretability: In machine learning, the concept of
408

interpretability is both important and slippery. Queue, 16(3):31–57, jun 2018. ISSN 1542-7730.
409

doi: 10.1145/3236386.3241340. URL https://doi.org/10.1145/3236386.3241340.
410

Rahul Mazumder, Xiang Meng, and Haoyue Wang. Quant-BnB: A scalable branch-and-bound
411

method for optimal decision trees with continuous features. In Kamalika Chaudhuri, Stefanie
412

Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the
413

39th International Conference on Machine Learning, volume 162 of Proceedings of Machine
414

Learning Research, pages 15255–15277. PMLR, 17–23 Jul 2022. URL https://proceedings.
415

mlr.press/v162/mazumder22a.html.
416

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare,
417

Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control
418

through deep reinforcement learning. nature, 518(7540):529–533, 2015.
419

10


---Page Break---
Cecília Nunes, Mathieu De Craene, Hélène Langet, Oscar Camara, and Anders Jonsson. Learning
420

decision trees through monte carlo tree search: An empirical evaluation. WIREs Data Mining
421

and Knowledge Discovery, 10(3):e1348, 2020. doi: https://doi.org/10.1002/widm.1348. URL
422

https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1348.
423

Rok Piltaver, Mitja Luštrek, Matjaž Gams, and Sanda Martinˇci´c-Ipši´c. What makes classification
424

trees comprehensible? Expert Systems with Applications, 62:333–346, 2016. ISSN 0957-4174.
425

doi: https://doi.org/10.1016/j.eswa.2016.06.009. URL https://www.sciencedirect.com/
426

science/article/pii/S0957417416302901.
427

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah
428

Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine
429

Learning Research, 22(268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.
430

J. Rissanen. Modeling by shortest data description. Automatica, 14(5):465–471, 1978. ISSN 0005-
431

1098. doi: https://doi.org/10.1016/0005-1098(78)90005-5. URL https://www.sciencedirect.
432

com/science/article/pii/0005109878900055.
433

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
434

optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.
435

06347.
436

Olivier Sigaud and Olivier Buffet. Markov decision processes in artificial intelligence. John Wiley &
437

Sons, 2013.
438

Nicholay Topin, Stephanie Milani, Fei Fang, and Manuela Veloso. Iterative bounding MDPs:
439

Learning interpretable policies via non-interpretable methods. Proceedings of the AAAI Conference
440

on Artificial Intelligence, 35(11):9923–9931, May 2021. doi: 10.1609/aaai.v35i11.17192. URL
441

https://ojs.aaai.org/index.php/AAAI/article/view/17192.
442

Jacobus van der Linden, Mathijs de Weerdt, and Emir Demirovi´c. Necessary and sufficient conditions
443

for optimal decision trees using dynamic programming. In A. Oh, T. Neumann, A. Globerson,
444

K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems,
445

volume 36, pages 9173–9212. Curran Associates, Inc., 2023.
446

Sicco Verwer and Yingqian Zhang. Learning optimal classification trees using a binary linear program
447

formulation. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages
448

1625–1632, 2019.
449

A
Code links
450

Quant-BnB.
The
Julia
code
for
Quant-BnB
is
available
at
https://github.com/
451

mengxianglgal/Quant-BnB.
452

CART. We use the scikit-learn Cython implementation of CART available at https://
453

scikit-learn.org/stable/modules/tree.html#tree-classification with the criterion
454

parameter fixed to “entropy”.
455

MurTree, Pystreed. Codes are available at https://github.com/MurTree/pymurtree and at
456

https://github.com/AlgTUDelft/pystreed.
457

B
On the failure of deep reinforcement learning.
458

For the dataset X = {(1, 2), (2, 1), (3, 4), (4, 3)}, Y = {0, 1, 2, 3} both our MDP and IBMDP are
459

equivalent for learning the optimal decision tree of depth 2. We show on Fig. 4 that two different
460

DRL algorithms exhibit opposite performance: DQN can learn the optimal decision tree while PPO
461

[Schulman et al., 2017] cannot. For that reason, we only trained Custard using DQN as the DRL agent.
462

We see on Fig. 4 and Table 1 that Custard-5 converged to trees worst than CART for all classification
463

datasets. This shows that while more scalable, DRL approaches are still not competitive on these
464

types of problems. [Kohler et al., 2023] studied potential failure modes of DRL in our setting.
465

11


---Page Break---
100
200
300
400
500
600
walltime in sec.

0.7

0.6

0.5

0.4

0.3

0.2

0.1

0.0

cumulative MDP rewards

DQN
PPO
Opt. Depth 2 Tree

0
200
400
600
800
1000
walltime in sec.

1.0

0.8

0.6

0.4

0.2

0.0

DQN-5 cumulative MDP rewards

avila
bank
bean
bidding
eeg
fault
htru
magic
occupancy
page
raisin
rice
room
segment
skin
wilt

Figure 4: Left, DRL to learn the optimal depth 2 tree. Right, Custard-5 to learn depth 3 decision trees
on classification datasets

C
Tree plots
466

  7

x18 ≤ 0.047

  3

(a) A=0.5, N=3, T=1

  7
  6

x12 ≤ 0.5

x14 ≤ 0.192

x11 ≤ 0.236

x17 ≤ 0.462

  7

x18 ≤ 0.047

  2
  2

  4
  7

x15 ≤ 0

(b) A=0.69, N=13, T=2.9

  7
  7
   7

x5 ≤ 0

x18 ≤ 0.047

x25 ≤ 0.73

x11 ≤ 0.236
x11 ≤ 0.249

  3

  4

  7

(c) A=0.53, N=11, T=3.2

  7
  7
   7

x5 ≤ 0

x18 ≤ 0.047

x25 ≤ 0.73

x11 ≤ 0.236
x11 ≤ 0.249

  7
  6

  3

  4

x14 ≤ 0.019

(d) A=0.55, N=13, T=3.6

Figure 5: Trees for the fault dataset. Top: trees from DPDT. Bottom: trees from CART. A is accuracy,
N the number of nodes, T the average number of tests.

D
Schematics DTDP
467

E
Detailed res of model selection
468

F
Comparisons with baselines operating on binary datasets
469

F.1
Why comparisons with baselines that binarize datasets is not fair in our favor?
470

Algorithms finding optimal DTs for binary datasets such as MurTree [Demirovic et al., 2022] use
471

a binarization method to transform a dataset with continuous attributes to a dataset with binary
472

attributes. However, a DT learned on the binary dataset, whenever it tests the value of a binary
473

attribute, can lead to up to two tests on the respective continuous attribute. Hence, DTs of a given
474

maximum depth on the binary dataset are actually deeper if transformed into DTs on the original
475

dataset with continuous attributes. Despite this, we show in Table 1 of this supplementary material
476

12


---Page Break---
class
class

y

x ≤ 0.2
x ≤ 0.6
y ≤ 0.3

done
done

    p = 1
r = -1/3

    p = 1
r = -2/3

    p = 1/3
    r = 𝛼

  p = 1/3
     r = 𝛼

p = 2/3
     r = 𝛼
  p = 1/3
       r = 𝛼

class

done

   p = 1
   r = 0

class

done

   p = 1
r = -1/2

class

done

class

done

     p = 1
   r = 0

class

done

     p = 1
   r = 0

class

done

class

done

class

done

   p = 2/3
    r = 𝛼

    p = 2/3
    r = 𝛼

     p = 1
   r = 0

     p = 1
 r = -1/2

     p = 1
 r = -1/2
   p = 1
r = -1/2

Figure 6: MDP for a training dataset made of three samples (illustrated with an oval and 2 diamonds),
two continuous attributes (x and y), and two classes. The tests generating function generated three
possible tests. There is an initial state (D, 0) (the training dataset at depth 0), and six non-terminal
states (three tests times two children states). Rewards are either α or the misclassification, and
transition probabilities are one, or the size of the child state over the size of the parent.

class

class

y

x ≤ 0.2
x ≤ 0.6

y ≤ 0.3

done
done

x

    

class

done

Q𝛼*(s,a) = [0,0]

class

done

class

done

class

done

class

done

class

done

class

done

class

done

[0,0]

[-1/3,-1/3]

[-2/3,-2/3]

[-½,-½ ]
[-½,-½ ] [0,0]
[0,0]
[-½ ,-½]
[-½ ,-½]

[-4/3 , -1/3]
[-1 ,0]

[-4/3 , -1/3]

Figure 7: For α = 0 and α = 1, the values of Q∗(s, a, α) are backpropagated from leaf states to the
initial state and are given in squared brackets. The optimal policy π∗(., α = 1), in pink, is a depth-0
tree with accuracy 2

3. The optimal policy π∗(., α = 0), in green, is a depth-1 tree with accuracy 1.

Table 2: Trees of depth ≤5 selected with the procedure described in Sec. 6.3.
Datasets
Accuracy (%) on unseen data
Runtime (s.)
Average Nb.Tests
Names
DPDT-3
DPDT-2
CART
Quant-BnB
MurTree
Pystreed
DPDT-3
DPDT-2
CART
DPDT-3
DPDT-2
CART
avila
66.9
65.7
60.5
57.3
OOM
OOM
51.625
2.701
1.031
4.9
4.9
4.8
bank
99.3
97.8
99.3
97.8
48.6
48.6
2.054
0.353
0.031
3.2
3.7
3.4
bean
91.1
91.1
89.9
84.7
OOM
OOM
88.142
7.571
5.369
4.6
4.9
5.0
bidding
99.2
99.2
99.2
98.5
97.5
97.5
2.963
0.545
0.081
1.4
1.4
2.3
eeg
78.0
74.6
73.0
73.0
OOM
OOM
57.038
4.347
0.892
4.6
4.8
5.0
fault
71.8
72.8
57.9
61.2
OOM
OOM
35.185
2.611
0.536
5.0
4.5
4.9
htru
98.0
98.3
98.3
97.9
OOM
91.2
63.519
5.189
2.174
1.1
2.4
4.7
magic
84.5
84.8
82.5
82.1
OOM
OOM
98.623
7.06
3.189
5.0
4.8
5.0
occupancy
99.5
99.5
99.5
96.3
OOM
82.3
11.113
1.263
0.162
1.0
1.0
1.4
page
97.1
97.1
96.7
95.8
OOM
93.4
26.596
2.547
0.369
3.5
5.0
4.8
raisin
87.8
91.1
90.0
89.0
45.6
45.6
7.756
1.775
0.069
3.1
2.3
4.5
rice
93.7
94.2
93.4
93.9
87.1
87.1
17.915
1.693
0.356
1.6
1.7
3.6
room
99.2
99.4
99.4
98.6
OOM
OOM
19.134
1.574
0.247
2.5
2.3
4.1
segment
93.5
93.1
87.4
82.7
OOM
OOM
6.488
0.879
0.184
3.7
3.9
3.9
skin
99.5
99.2
98.6
98.6
OOM
OOM
265.243
18.066
1.985
3.8
3.8
4.2
wilt
87.2
84.8
87.6
81.3
70.4
70.4
3.898
0.462
0.125
4.1
3.2
3.9

that DPDT typically finds better solutions (in terms of training accuracy) than MurTree + binarization
477

even though the comparison is not fair in our favor since MurTree is considering deeper trees.
478

To illustrate this unbalance with an example, we present a dataset with 3 samples, 2 classes, and 1
479

continuous attribute. After binning the continuous attribute and binarizing the dataset into 3 binary
480

attributes, we compute the optimal depth 1 tree like [Demirovic et al., 2022] or [Verwer and Zhang,
481

13


---Page Break---
2019] would do. To apply this depth 1 tree to the original continuous attribute dataset, the root node
482

"a ∈[0.2, 0.22]" should be decomposed in two decision nodes "a ≤0.19" and "a ≤0.22" before
483

making a label assignment. So the corresponding tree that can be applied on the continuous attribute
484

is actually of depth 2.
485

a
y
x1
0.1
1
x2
0.2
2
x3
0.22
2
x4
0.3
1

→

[0, 0.19]
[0.2, 0.22]
[0.23, 0.3]
y
x1
1
0
0
1
x2
0
1
0
2
x3
0
1
0
2
x3
0
0
1
1

486

←
487

F.2
Experiments
488

Comparing baselines such as [Verwer and Zhang, 2019] or [Demirovic et al., 2022] to DPDT or
489

Quant-BnB [Mazumder et al., 2022] that operate directly on continuous attributes with the same
490

maximum depth is not fair in favor of the latter algorithms as discussed above. Still, for the sake of
491

curiosity we performed comparisons on datasets of prior works. These can be split into two groups.
492

1) MurTree:
Demirovic et al. [2022] propose an algorithm that retrieves optimal trees for large
493

datasets with binary features using dynamic programming. They also propose a binarization method
494

to retrieve suboptimal shallow trees for large datasets with continuous features. We do not run
495

MurTree but use of the results in Table 6 from Mazumder et al. [2022] (see the “approx” column)
496

which previously compared Quant-BnB to MurTree.
497

2) OCT, MFOCT, BinOCT:
Bertsimas and Dunn [2017], Aghaei et al. [2020], Verwer and
498

Zhang [2019] propose optimal tree algorithms which formulate the learning problem as a MIP.
499

OCT and MFOCT can produce optimal trees for small datasets with continuous features. BinOCT
500

can also produce optimal trees for small datasets with continuous features after they have been
501

binarized. We make use of the results available at https://github.com/LucasBoTang/Optimal_
502

Classification_Trees.
503

Reproducibility:
as mentioned above, we did not run the additional baselines but instead used
504

available results. As such runtimes were provided only when available. OCT, MFOCT, BinOCT were
505

run on a single core of an Intel(R) Core(TM) CPU i7-7700HQ @ 2.80GHz. MurTree was run
506

on a single core of a Intel Xeon 2.30GHz. According to online benchmarks the performances of
507

those machines are similar to our Laptopt CPU Intel® Core™i7-8665U CPU.
508

G
Markov Decision Problem formulations of the Decision Tree Learning
509

Problem
510

In this section we compare our Markov Decision Problem (MDP) formulation of decision tree
511

learning from Section 4 to that of prior work, namely [Garlapati et al., 2015] and [Topin et al.,
512

2021]. In a nutshell, prior work viewed the task as a deterministic and Partially Observable MDP
513

[Sigaud and Buffet, 2013] and used algorithms such as Q-learning [Garlapati et al., 2015] or deep
514

Q-learning [Topin et al., 2021] to solve them in an online fashion one datum from the dataset at a
515

time. Our approach is different in that it builds a stochastic and fully observable MDP. Our MDP
516

makes it possible to perform two operations that are critical for DPDT: i) being able to call the
517

tests generating function which does not operate online but needs full offline access of the dataset
518

ii) being able to efficiently compute through dynamic programming optimal policies for different
519

complexity-performance trade-offs, which is critical in practice as our improved training accuracy
520

compared to greedy methods would otherwise quickly lead to overfitting. High level differences
521

14


---Page Break---
Datasets
Accuracy of depth-3 trees
Names
Samples
Features
Classes
Opt.
DPDT-5
DPDT-4
MurTree
CART
avila
10430
10
12
58.5%
58.5∗%
58 %
58.5∗%
53.2%
bank
1097
4
2
98.3%
98%
98%
97.3%
93.3%
bean
10888
16
7
87.1%
85.6%
85%
86.9%
77.7%
bidding
5056
9
2
99.3%
99.3∗%
99.3%
98.1%
98.1%
eeg
11984
14
2
70.8%
70.3%
70%
68.8%
66.6%
fault
1552
27
7
68.2%
68%
65.7%
67.3%
55.3%
htru
14318
8
2
98.1%
98%
98%
97.9%
97.9%
magic
15216
10
2
83.1%
83%
82.7%
81.1%
80.1%
occupancy
8143
5
2
99.4%
99.4∗%
99.3%
99.1%
98.9%
page
4378
10
5
97.1%
97%
97%
96.6%
96.4%
raisin
720
7
2
89.4%
88.5%
88.3%
87.5%
86.9%
rice
3048
7
2
93.8%
93.7%
93.6%
93.4%
93.3%
room
8103
16
4
99.2%
99.2∗%
99.2%
99.2∗%
96.8%
segment
1848
18
7
88.7%
88.2%
88.2%
88.1%
57.4%
skin
196045
3
2
96.9%
96.7%
96.7%
96.8%
96.6%
wilt
4339
5
2
99.6%
99.5%
99.5%
98.7%
99.3%
Table 3: Training accuracy of different decision tree learning algorithms. All algorithms learn trees
of depth at most 3 on 16 classification datasets. MurTree returns decision trees for datasets binarized
using using the minimum description length principle. Results for MurTree are taken from Tables 2
and 6 from [Mazumder et al., 2022].

Datasets
Train Accuracy depth-5
Test Accuracy depth-5
Runtime depth-5
Names
Samples
Features
Classes
DPDT-4
DPDT-5
OCT
MFOCT
BinOCT
CART
DPDT-4
DPDT-5
OCT
MFOCT
BinOCT
CART
DPDT-4
DPDT-5
OCT
MFOCT
BinOCT
CART
balance-scale
624
4
3
90.9%
91.0%
71.8%
82.6%
67.5%
86.5%
77.1%
74.8%
66.9%
71.3%
61.6%
76.4%
68.34
401.71
605.51
600.1
603.95
< 0.001
breast-cancer
276
9
2
94.2%
94.7%
88.6%
91.1%
75.4%
87.9%
66.4%
67.6%
67.1%
73.8%
62.4%
70.3%
19.09
62.36
603.39
600.25
603.67
0.001
car-evaluation
1728
6
4
92.2%
92.2%
70.1%
80.4%
84.0%
87.1%
90.3%
90.3%
69.5%
79.8%
82.3%
87.1%
5.39
38.07
618.09
600.49
613.14
< 0.001
hayes-roth
160
9
3
93.3%
94.2%
82.9%
95.4%
64.6%
76.7%
75.4%
71.2%
77.5%
77.5%
54.2%
69.2%
0.91
2.58
602.02
600.19
601.83
0.001
house-votes-84
232
16
2
100.0%
100.0%
100.0%
100.0%
100.0%
99.4%
95.4%
95.4%
93.7%
94.3%
96.0%
95.1%
0.44
0.65
105.72
10.74
6.6
< 0.001
soybean-small
46
50
4
100.0%
100.0%
100.0%
100.0%
76.8%
100.0%
93.1%
93.1%
94.4%
91.7%
72.2%
93.1%
0.01
0.01
4.18
0.41
1.84
< 0.001
spect
266
22
2
93.0%
93.0%
92.5%
93.0%
92.2%
88.5%
73.1%
73.9%
75.6%
74.6%
73.1%
75.1%
6.32
16.78
604.87
600.33
605.57
0.001
tic-tac-toe
958
24
2
90.8%
91.1%
68.5%
76.1%
85.7%
85.8%
82.1%
82.5%
69.6%
73.6%
79.6%
81.0%
107.94
626.34
615.28
600.45
621.81
0.001

Table 4: Train/test accuracies and runtimes of different decision tree learning algorithms. Note that
we are not using any regularization in this experiment (in order for all solvers to optimize the same
objective function) and as such we might overfit compared to CART that does not optimize the
training error as intensively. All algorithms learn trees of depth at most 5 on 8 classification datasets.
A time limit of 10 minutes is set for OCT-type algorithms. DPDT is used with two different test
generating functions: CART with a maximum depth of 4 and CART with a maximum depth of 5.
The values in this table are averaged over 3 seeds giving 3 different train/test datasets.

between MDPs are summarized in Table 5. For the sake of self-completeness we then detail both
522

MDPs of [Topin et al., 2021] and [Garlapati et al., 2015] which are to be contrasted with our MDP
523

formulation in Section 4.
524

Table 5: MDP formulations of the decision tree learning problem
MDP properties
IBMDP [Topin et al., 2021]
[Garlapati et al., 2015]
Ours
Training samples attributes
Any
Categorical
Any
Discounted
Yes
Yes
No
Horizon
Infinite
Finite
Finite
States
Partial information about a single training sample
Partial information about a single training sample
A full dataset in P(D)
Actions
Tests and label assignments
State dependent tests and label assignments
State dependent tests and label assignments
Transitions
Deterministic
Deterministic
Stochastic

G.1
Iteratvie Bounding MDPs
525

An IBMDP [Topin et al., 2021] is an episodic, infinite horizon, discounted MDP. IBMDPs can be
526

used for learning decision trees of any base MDP. We discuss here the case where the base MDP is a
527

classification task. In this case, during each episode, an agent has to classify a hidden training sample
528

xi drawn uniformly from a training dataset with continuous attributes. We assume whiteout loss of
529

generality that the training dataset X ⊂[0, 1]N×p has continuous attributes in [0, 1]. On the other
530

hand, the set of labels is Y = {1, ..., K}. An IBMDP is defined as follows.
531

State space: the state space is the hypercube [0, 1]3·p. A IBMDP state has two parts. The continuous
532

attributes of the hidden training sample xi = (xi1, ..., xip) to classify, and a lower and upper
533

bound (Lk, Uk) for each of the p attributes. For each attribute xik, (Lk, Uk) represents the current
534

agent knowledge about its hidden value. Initially, (Lk, Uk) = (0, 1) for all k, which are iteratively
535

refined by taking tests actions.
536

Action space: an agent in an IBMDP can either take an assignment action a ∈Y, or a test action
537

1{xik≤v·(Uk−Lk)+Lk} with k ∈{1, . . . , p} and v ∈{
1
d+1, ...,
d
d+1}, with d ∈N a hyperparameter of
538

the IBMDP.
539

15


---Page Break---
Transition function: if an agent takes a label assignment action, the IBMDP transits to a ter-
540

minal state, a new training sample x is drawn at random from X, and the attributes bounds
541

(L1, ..., Lp, U1, .., Up) are reset to 0 or 1. If an agent takes a test action, the attributes bounds
542

are refined. Let xik be the value of the k-th attribute of the hidden training sample xi, and (Lk, Uk)
543

be the current bounds of xik. If 1{xik≤v·(Uk−Lk)+Lk} is true, then Lk is updated to v·(Uk−Lk)+Lk,
544

else, it is Uk that is updated to v · (Uk −Lk) + Lk.
545

Reward function: the reward for assigning the label yi ∈Y to the hidden training sample xi is
546

1a=yi · r+ + 1a̸=yi · r−, with r+ > 0 and r−< 0. The reward for taking a test action is α < 0.
547

G.2
MDP formulation of [Garlapati et al., 2015]
548

MDP formulations based on [Garlapati et al., 2015] assume categorical attributes, i.e, the training
549

dataset D is in ZN×p. The MDP is episodic with a discount factor and a finite horizon p + 1. An
550

episode of this MDP consists of costly queries of a training sample’s attributes until a label assignment
551

is made.
552

State space: a state of the above MDP has partial information about a training sample to classify.
553

At every step of the MDP, an agent queries a hidden attribute and updates its knowledge about the
554

training sample by concatenating all revealed attributes.
555

Acion space: at every step t in the MDP, an agent can either assign a class label in Y = {1, ..., K},
556

or, make a query at of a hidden attribute of a training sample: At = ({1, ..., p} \
t−1
S

h=0
ah) ∪Y.
557

Transition function: the current state of the MDP contains values of previously queried attributes.
558

At t = 0, s = {}. Assuming the hidden training sample to be classified during the current episode
559

is xi = (xi1, ..., xip), then the deterministic transition function is: T(s, a = xij) = s ∪xij or
560

T(s, a ∈Y) = sterminal. At the start of a new episode, a new training sample is drawn uniformly
561

from D.
562

Reward function: at time t, when the hidden training sample to classify is xi, if the an agent takes
563

an assignment action a ∈D, the reward is 1a=yi · r+ + 1a̸=yi · r−, with r+ > 0 and r−< 0. So an
564

agent gets a positive signal for making a correct label assignment and negative signal otherwise. If
565

the agent takes a query action, the reward is a negative value α in order to discourage taking to much
566

queries and control the tree complexity.
567

H
Proof of equivalence of learning objectives
568

In this section, we prove the equivalence between learning an optimal policy in the MDP of Section
569

4 and finding the minimizing tree of Eq. (2). We first define C(T), the expected number of tests
570

performed by tree T on dataset D. Here T is induced by policy π, i.e. T = E(π, s0). C(T) can be
571

defined recursively as C(T) = 0 if T is a leaf node, and C(T) = 1 + plC(Tl) + prC(Tr), where
572

Tl = E(π, sl) and Tr = E(π, sr). In words, when the root of T is a test node, the expected number
573

of tests is one plus the expected number of tests of the left and right sub-trees of the root node.
574

For the purpose of the proof, we overload the definition of Jα and Lα, to make explicit the dependency
575

on the dataset and the maximum depth. As such, Jα(π) becomes Jα(π, D, D) and Lα(T) becomes
576

Lα(T, D). Let us first show that the relation Jα(π, D, 0) = −Lα(T, D) is true. If the maximum
577

depth is D = 0 then π(s0) is necessarily a class assignment, in which case the expected number of
578

tests is zero and the relation is obviously true since the reward is minus the average classification loss.
579

Now assume it is true for any dataset and tree of depth at most D with D ≥0 and let us prove that it
580

holds for all trees of depth D + 1. For a tree T of depth D + 1 the root is necessarily a test node.
581

Let Tl = E(π, sl) and Tr = E(π, sr) be the left and right sub-trees of the root node of T. Since
582

both sub-trees are of depth at most D, the relation holds and we have Jα(π, Xl, D) = Lα(Tl, Xl)
583

and Jα(π, Xr, D) = Lα(Tr, Xr), where Xl and Xr are the datasets of the “right" and “left" states
584

to which the MDP transitions—with probabilities pl and pr—upon application of π(s0) in s0, as
585

16


---Page Break---
described in the MDP formulation. Moreover, from the definition of the policy return we have
586

Jα(π, D, D + 1) = −α + pl ∗Jα(π, Xl, D) + pr ∗Jα(π, Xr, D)
= −α −pl ∗Lα(Tl, Xl) −pr ∗Lα(Tr, D)

= −α −pl ∗

 
1
|Xl|

X

(xi,yi)∈Xl
ℓ(yi, Tl(xi)) + αC(Tl)

!

−pr ∗

 
1
|Xr|

X

(xi,yi)∈Xr
ℓ(yi, Tr(xi)) + αC(Tr)

!

= −1

N

X

(xi,yi)∈X
ℓ(yi, T(xi)) −α(1 + plC(Tl) + prC(Tr))

= −L(T, D)

I
Deeper trees experiments
587

In this section, we push the limits of DPDT to learn trees of at most depth 10. We run two instances
588

of DPDT. The first one will generate a MDP using a depth dependant tests generating function.
589

DPDT-2... generates a MDP where actions availabe at states corresponding to depth ≤5 are given by
590

running CART with a maximum depth of 2, and actions for other states are given by CART with a
591

maximum depth of 1 (the maximum information gain splits given the dataset X in the state ((X, d))).
592

DPDT-2+1... generates a bigger MDP than DPDT-2... as actions available to states with depths up to
593

6 are given by CART run with a maximum depth of 2. On Table 6 we observe that deep trees learnt
594

by CART and DPDT perform similarly well on unseen data of different classificiation problems.
595

CART runs way faster than DPDT to compute deep trees. However, DPDT learns more interpretable
596

trees with respect to the average number of tests performed on data which is a very useful feature
597

for real-life applications such as medicine where each additional test before a diagnostic can be very
598

expensive (for example performing an addition MRI scan).

Table 6: Test accuracy of trees of depth ≤10 selected with the procedure described in Sec. 6.2.

Datasets
Accuracy (%) on unseen data
Runtime (s.)
Average Nb.Tests
Names
DPDT-2...
DPDT-2+1...
CART
DPDT-2...
DPDT-2+1...
CART
DPDT-2...
DPDT-2+1...
CART
avila
94.3
95.1
87.8
86.476
187.313
1.579
8.4
8.4
8.8
bank
99.3
99.3
99.3
1.664
2.174
0.028
3.3
3.3
3.4
bean
91.3
90.9
91.2
102.796
309.981
8.287
5.2
4.0
6.1
bidding
99.4
99.4
99.4
1.833
3.226
0.095
2.4
2.4
2.4
eeg
83.6
83.5
82.0
85.198
229.49
2.386
8.1
8.2
9.3
fault
73.3
73.8
68.7
35.09
108.265
1.148
5.6
5.6
6.9
htru
97.6
98.0
98.1
45.941
123.689
4.234
2.2
1.2
3.4
magic
85.4
84.9
84.8
146.253
391.594
7.021
5.8
5.9
8.1
occupancy
99.5
99.5
99.5
6.847
15.608
0.226
1.0
1.0
1.4
page
96.5
96.9
96.5
22.526
58.102
0.713
4.5
6.2
7.7
raisin
85.6
86.7
88.9
8.717
19.652
0.115
2.1
2.1
6.5
rice
93.4
93.2
93.7
20.18
44.867
0.626
1.8
1.8
3.0
room
99.3
99.6
99.6
5.186
8.55
0.318
2.3
4.1
4.1
segment
97.0
97.0
94.8
9.796
22.562
0.286
5.1
5.1
5.0
skin
99.9
99.9
99.8
120.576
308.577
2.94
6.3
6.2
5.4
wilt
86.0
86.0
84.8
2.274
3.583
0.151
4.3
4.4
4.4

599

J
Additional comparisons with Quant-BnB
600

In Table 7 we compare DPDT with Quant-BnB on train and test sets of different classification
601

problems. Quant-BnB has a time limit equal to DPDT-5’ runtime on each problem. We also run
602

Quant-BnB with bonuses of 5 and 50 seconds to see if the latter can outperform DPDT with just a
603

little more time or if it would require almost twice the time (see Table 7 for DPDT-5’ runtimes). We
604

observe that for both train and test accuracies, Quant-BnB-t+50 (DPDT-5 runtime plus 50 seconds
605

bonus) outperforms DPDT most often.
606

17


---Page Break---
Table 7: Train and Tests accuracies of DPDT and Quant-BnB for Trees of maximum depth 3

Datasets
Train Accuracies
Test Accuracies
Names
DPDT-3
DPDT-4
DPDT-5
Quant-BnB-T
Quant-BnB-t+5
Quant-BnB-t+50
DPDT-3
DPDT-4
DPDT-5
Quant-BnB-T
Quant-BnB-t+5
Quant-BnB-t+50
avila
58
58
58.5
57.3
57.3
57.3
57.9
57.9
58.2
57.1
57.1
57.1
bank
98
98
98
97.1
98.3
98.3
97.8
97.8
97.8
97.8
97.8
97.8
bean
85
85
85.6
85.3
85.3
85.3
85.1
85.1
84.9
85.6
85.6
85.6
bidding
99.3
99.3
99.3
98.6
98.7
99.3
99
99
99
98.6
98.7
99
eeg
69.4
70
70.3
68.3
68.3
68.9
71
69.8
70
69.8
69.8
68.5
fault
65.7
65.7
68
64.6
64.6
66.9
64.8
64.8
65.3
63.2
63.2
64.3
htru
98
98
98
98
98
98
98.2
97.9
97.9
98.1
98.1
98.1
magic
82.7
82.7
82.9
82.6
82.6
82.7
82
82
82.2
82.2
82.2
82.3
occupancy
99.3
99.3
99.4
99.3
99.3
99.4
93.4
93.4
93.9
89.6
89.6
91
page
97
97
97
96.5
96.7
97
96
96
96.1
96.7
95.9
95.9
raisin
88.3
88.3
88.5
88.1
88.6
89
87.2
87.2
88.3
88.9
88.3
89.4
rice
93.5
93.6
93.7
93.7
93.7
93.7
92.1
92.1
92.7
92
92
92
room
99.2
99.2
99.2
98.8
98.8
99
99
99
99
98.6
98.6
98.8
segment
88.2
88.2
88.2
79.1
87.8
87.8
84
84
84
76.8
84.2
84.2
skin
96.7
96.7
96.7
96.7
96.7
96.7
96.7
96.7
96.7
96.6
96.6
96.6
wilt
99.5
99.5
99.5
99.4
99.4
99.6
80.4
79.2
79.2
77.6
81.2
78.8

K
Additional figures for different complexity measures
607

We show here the complexity-performance trade-offs for all 16 datasets. We show the plot for two
608

complexity measures: average number of tests (what DPDT optimize) and total number of nodes
609

(what the post-process prunning of CART optimizes). On the first measure, the trees that DPDT finds
610

dominate those of CART, which matches the theory. On the second measure, even though we do not
611

optimize for the total number of nodes, we are still able to find better trade-offs w.r.t. this metric than
612

CART for several datasets.
613

K.1
Average number of tests vs accuracy
614

0
1
2
3
4
5
average tests per sample

0.40

0.45

0.50

0.55

0.60

0.65

accuracy

avila

DPDT-3 in 51s
CART-postpruning in 1s

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
average tests per sample

0.6

0.7

0.8

0.9

1.0

accuracy

bank

DPDT-3 in 2s
CART-postpruning in 0s

0
1
2
3
4
5
average tests per sample

0.3

0.4

0.5

0.6

0.7

0.8

0.9

accuracy

bean

DPDT-3 in 88s
CART-postpruning in 8s

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
average tests per sample

0.90

0.92

0.94

0.96

0.98

1.00

accuracy

bidding

DPDT-3 in 2s
CART-postpruning in 0s

0
1
2
3
4
5
average tests per sample

0.55

0.60

0.65

0.70

0.75

accuracy

eeg

DPDT-3 in 57s
CART-postpruning in 1s

0
1
2
3
4
5
average tests per sample

0.4

0.5

0.6

0.7

0.8

accuracy

fault

DPDT-3 in 35s
CART-postpruning in 0s

0
1
2
3
4
5
average tests per sample

0.91

0.92

0.93

0.94

0.95

0.96

0.97

0.98

accuracy

htru

DPDT-3 in 63s
CART-postpruning in 3s

0
1
2
3
4
5
average tests per sample

0.65

0.70

0.75

0.80

0.85

accuracy

magic

DPDT-3 in 98s
CART-postpruning in 5s

0
1
2
3
4
5
average tests per sample

0.80

0.85

0.90

0.95

1.00

accuracy

occupancy

DPDT-3 in 11s
CART-postpruning in 0s

0
1
2
3
4
5
average tests per sample

0.90

0.92

0.94

0.96

0.98

accuracy

page

DPDT-3 in 26s
CART-postpruning in 0s

0
1
2
3
4
average tests per sample

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

accuracy

rice

DPDT-3 in 17s
CART-postpruning in 0s

0
1
2
3
4
5
average tests per sample

0.5

0.6

0.7

0.8

0.9

accuracy

raisin

DPDT-3 in 7s
CART-postpruning in 0s

0
1
2
3
4
average tests per sample

0.2

0.4

0.6

0.8

1.0

accuracy

segment

DPDT-3 in 6s
CART-postpruning in 0s

0
1
2
3
4
average tests per sample

0.825

0.850

0.875

0.900

0.925

0.950

0.975

1.000

accuracy

room

DPDT-3 in 19s
CART-postpruning in 0s

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
average tests per sample

0.9825

0.9850

0.9875

0.9900

0.9925

0.9950

0.9975

1.0000

accuracy

wilt

DPDT-3 in 3s
CART-postpruning in 0s

0
1
2
3
4
average tests per sample

0.800

0.825

0.850

0.875

0.900

0.925

0.950

0.975

1.000

accuracy

skin

DPDT-3 in 265s
CART-postpruning in 3s

Figure 8: Average number of tests-accuracies trade-offs of CART and DPDT-3 on classification
training datasets. Both algorithms learn trees of depths at most 5. CART makes a trade-off with the
minimal complexity post-pruning algorithm. DPDT-3 makes a trade-off by returning policies for
1000 different α.

K.2
Total number of nodes vs accuracy
615

L
Codes to reproduce experiments
616

Anonymized
github
for
DPDT
code:
https://anonymous.4open.science/r/
617

reproduce-E9BD/README.md
618

Anonymized github of our clone of Quant-BnB code: https://anonymous.4open.science/r/
619

reproduce-quant-bnb-80ED/README.md
620

18


---Page Break---
0
10
20
30
40
50
60
nodes

0.40

0.45

0.50

0.55

0.60

0.65

accuracy

avila

DPDT-3 in 51s
CART-postpruning in 1s

0
10
20
30
40
nodes

0.6

0.7

0.8

0.9

1.0

accuracy

bank

DPDT-3 in 2s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.3

0.4

0.5

0.6

0.7

0.8

0.9

accuracy

bean

DPDT-3 in 88s
CART-postpruning in 8s

0
5
10
15
20
25
30
35
40
nodes

0.90

0.92

0.94

0.96

0.98

1.00

accuracy

bidding

DPDT-3 in 2s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.55

0.60

0.65

0.70

0.75

accuracy

eeg

DPDT-3 in 57s
CART-postpruning in 1s

0
10
20
30
40
50
60
nodes

0.4

0.5

0.6

0.7

0.8

accuracy

fault

DPDT-3 in 35s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.91

0.92

0.93

0.94

0.95

0.96

0.97

0.98

accuracy

htru

DPDT-3 in 63s
CART-postpruning in 3s

0
10
20
30
40
50
60
nodes

0.65

0.70

0.75

0.80

0.85

accuracy

magic

DPDT-3 in 98s
CART-postpruning in 5s

0
10
20
30
40
50
nodes

0.80

0.85

0.90

0.95

1.00

accuracy

occupancy

DPDT-3 in 11s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.90

0.92

0.94

0.96

0.98

accuracy

page

DPDT-3 in 26s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

accuracy

rice

DPDT-3 in 17s
CART-postpruning in 0s

0
10
20
30
40
50
60
nodes

0.5

0.6

0.7

0.8

0.9

accuracy

raisin

DPDT-3 in 7s
CART-postpruning in 0s

0
5
10
15
20
25
30
35
40
nodes

0.2

0.4

0.6

0.8

1.0

accuracy

segment

DPDT-3 in 6s
CART-postpruning in 0s

0
10
20
30
40
nodes

0.825

0.850

0.875

0.900

0.925

0.950

0.975

1.000

accuracy

room

DPDT-3 in 19s
CART-postpruning in 0s

0
10
20
30
40
nodes

0.9825

0.9850

0.9875

0.9900

0.9925

0.9950

0.9975

1.0000

accuracy

wilt

DPDT-3 in 3s
CART-postpruning in 0s

0
10
20
30
40
nodes

0.800

0.825

0.850

0.875

0.900

0.925

0.950

0.975

1.000

accuracy

skin

DPDT-3 in 265s
CART-postpruning in 3s

Figure 9: Nodes-accuracies trade-offs of CART and DPDT-3 on classification training datasets. Both
algorithms learn trees of depths at most 5. CART makes a trade-off with the minimal complexity
post-pruning algorithm. DPDT-3 makes a trade-off by returning policies for 1000 different α. Even
though we do not optimize for this complexity metric, we are still able to find better trade-offs than
CART with post-pruning in several cases.

NeurIPS Paper Checklist
621

1. Claims
622

Question: Do the main claims made in the abstract and introduction accurately reflect the
623

paper’s contributions and scope?
624

Answer: [Yes]
625

Justification: All the algorithms and claims mentionned in the intro are studied and presented
626

in detail in the main paper. Please see 1 2.
627

Guidelines:
628

• The answer NA means that the abstract and introduction do not include the claims
629

made in the paper.
630

• The abstract and/or introduction should clearly state the claims made, including the
631

contributions made in the paper and important assumptions and limitations. A No or
632

NA answer to this question will not be perceived well by the reviewers.
633

• The claims made should match theoretical and experimental results, and reflect how
634

much the results can be expected to generalize to other settings.
635

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
636

are not attained by the paper.
637

2. Limitations
638

Question: Does the paper discuss the limitations of the work performed by the authors?
639

Answer: [Yes]
640

Justification: Dedicated sections in the experiments and in the conclusion for limitations.
641

Please see 7.
642

Guidelines:
643

• The answer NA means that the paper has no limitation while the answer No means that
644

the paper has limitations, but those are not discussed in the paper.
645

• The authors are encouraged to create a separate "Limitations" section in their paper.
646

19


---Page Break---
• The paper should point out any strong assumptions and how robust the results are to
647

violations of these assumptions (e.g., independence assumptions, noiseless settings,
648

model well-specification, asymptotic approximations only holding locally). The authors
649

should reflect on how these assumptions might be violated in practice and what the
650

implications would be.
651

• The authors should reflect on the scope of the claims made, e.g., if the approach was
652

only tested on a few datasets or with a few runs. In general, empirical results often
653

depend on implicit assumptions, which should be articulated.
654

• The authors should reflect on the factors that influence the performance of the approach.
655

For example, a facial recognition algorithm may perform poorly when image resolution
656

is low or images are taken in low lighting. Or a speech-to-text system might not be
657

used reliably to provide closed captions for online lectures because it fails to handle
658

technical jargon.
659

• The authors should discuss the computational efficiency of the proposed algorithms
660

and how they scale with dataset size.
661

• If applicable, the authors should discuss possible limitations of their approach to
662

address problems of privacy and fairness.
663

• While the authors might fear that complete honesty about limitations might be used by
664

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
665

limitations that aren’t acknowledged in the paper. The authors should use their best
666

judgment and recognize that individual actions in favor of transparency play an impor-
667

tant role in developing norms that preserve the integrity of the community. Reviewers
668

will be specifically instructed to not penalize honesty concerning limitations.
669

3. Theory Assumptions and Proofs
670

Question: For each theoretical result, does the paper provide the full set of assumptions and
671

a complete (and correct) proof?
672

Answer: [Yes]
673

Justification: Propositions and theorems are proven. Note that is not paper is not a theory
674

paper. Please see H.
675

Guidelines:
676

• The answer NA means that the paper does not include theoretical results.
677

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
678

referenced.
679

• All assumptions should be clearly stated or referenced in the statement of any theorems.
680

• The proofs can either appear in the main paper or the supplemental material, but if
681

they appear in the supplemental material, the authors are encouraged to provide a short
682

proof sketch to provide intuition.
683

• Inversely, any informal proof provided in the core of the paper should be complemented
684

by formal proofs provided in appendix or supplemental material.
685

• Theorems and Lemmas that the proof relies upon should be properly referenced.
686

4. Experimental Result Reproducibility
687

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
688

perimental results of the paper to the extent that it affects the main claims and/or conclusions
689

of the paper (regardless of whether the code and data are provided or not)?
690

Answer: [Yes]
691

Justification: Code and data links are provided. Algorithms are described explicitly. Please
692

see A L.
693

Guidelines:
694

• The answer NA means that the paper does not include experiments.
695

• If the paper includes experiments, a No answer to this question will not be perceived
696

well by the reviewers: Making the paper reproducible is important, regardless of
697

whether the code and data are provided or not.
698

20


---Page Break---
• If the contribution is a dataset and/or model, the authors should describe the steps taken
699

to make their results reproducible or verifiable.
700

• Depending on the contribution, reproducibility can be accomplished in various ways.
701

For example, if the contribution is a novel architecture, describing the architecture fully
702

might suffice, or if the contribution is a specific model and empirical evaluation, it may
703

be necessary to either make it possible for others to replicate the model with the same
704

dataset, or provide access to the model. In general. releasing code and data is often
705

one good way to accomplish this, but reproducibility can also be provided via detailed
706

instructions for how to replicate the results, access to a hosted model (e.g., in the case
707

of a large language model), releasing of a model checkpoint, or other means that are
708

appropriate to the research performed.
709

• While NeurIPS does not require releasing code, the conference does require all submis-
710

sions to provide some reasonable avenue for reproducibility, which may depend on the
711

nature of the contribution. For example
712

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
713

to reproduce that algorithm.
714

(b) If the contribution is primarily a new model architecture, the paper should describe
715

the architecture clearly and fully.
716

(c) If the contribution is a new model (e.g., a large language model), then there should
717

either be a way to access this model for reproducing the results or a way to reproduce
718

the model (e.g., with an open-source dataset or instructions for how to construct
719

the dataset).
720

(d) We recognize that reproducibility may be tricky in some cases, in which case
721

authors are welcome to describe the particular way they provide for reproducibility.
722

In the case of closed-source models, it may be that access to the model is limited in
723

some way (e.g., to registered users), but it should be possible for other researchers
724

to have some path to reproducing or verifying the results.
725

5. Open access to data and code
726

Question: Does the paper provide open access to the data and code, with sufficient instruc-
727

tions to faithfully reproduce the main experimental results, as described in supplemental
728

material?
729

Answer: [Yes]
730

Justification: Anonymized github repo and data links are provided. Please see A L.
731

Guidelines:
732

• The answer NA means that paper does not include experiments requiring code.
733

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
734

public/guides/CodeSubmissionPolicy) for more details.
735

• While we encourage the release of code and data, we understand that this might not be
736

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
737

including code, unless this is central to the contribution (e.g., for a new open-source
738

benchmark).
739

• The instructions should contain the exact command and environment needed to run to
740

reproduce the results. See the NeurIPS code and data submission guidelines (https:
741

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
742

• The authors should provide instructions on data access and preparation, including how
743

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
744

• The authors should provide scripts to reproduce all experimental results for the new
745

proposed method and baselines. If only a subset of experiments are reproducible, they
746

should state which ones are omitted from the script and why.
747

• At submission time, to preserve anonymity, the authors should release anonymized
748

versions (if applicable).
749

• Providing as much information as possible in supplemental material (appended to the
750

paper) is recommended, but including URLs to data and code is permitted.
751

6. Experimental Setting/Details
752

21


---Page Break---
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
753

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
754

results?
755

Answer: [Yes]
756

Justification: Evertyhing is detailed clearly in the main paper, in the appendix and in the
757

code repos. Please see 6.
758

Guidelines:
759

• The answer NA means that the paper does not include experiments.
760

• The experimental setting should be presented in the core of the paper to a level of detail
761

that is necessary to appreciate the results and make sense of them.
762

• The full details can be provided either with the code, in appendix, or as supplemental
763

material.
764

7. Experiment Statistical Significance
765

Question: Does the paper report error bars suitably and correctly defined or other appropriate
766

information about the statistical significance of the experiments?
767

Answer: [Yes]
768

Justification: When processes are stochastic error values are provided in result tables. Please
769

see 1.
770

Guidelines:
771

• The answer NA means that the paper does not include experiments.
772

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
773

dence intervals, or statistical significance tests, at least for the experiments that support
774

the main claims of the paper.
775

• The factors of variability that the error bars are capturing should be clearly stated (for
776

example, train/test split, initialization, random drawing of some parameter, or overall
777

run with given experimental conditions).
778

• The method for calculating the error bars should be explained (closed form formula,
779

call to a library function, bootstrap, etc.)
780

• The assumptions made should be given (e.g., Normally distributed errors).
781

• It should be clear whether the error bar is the standard deviation or the standard error
782

of the mean.
783

• It is OK to report 1-sigma error bars, but one should state it. The authors should
784

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
785

of Normality of errors is not verified.
786

• For asymmetric distributions, the authors should be careful not to show in tables or
787

figures symmetric error bars that would yield results that are out of range (e.g. negative
788

error rates).
789

• If error bars are reported in tables or plots, The authors should explain in the text how
790

they were calculated and reference the corresponding figures or tables in the text.
791

8. Experiments Compute Resources
792

Question: For each experiment, does the paper provide sufficient information on the com-
793

puter resources (type of compute workers, memory, time of execution) needed to reproduce
794

the experiments?
795

Answer: [Yes]
796

Justification: Exact CPU model as well as ram are provided. Please see 6.
797

Guidelines:
798

• The answer NA means that the paper does not include experiments.
799

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
800

or cloud provider, including relevant memory and storage.
801

• The paper should provide the amount of compute required for each of the individual
802

experimental runs as well as estimate the total compute.
803

22


---Page Break---
• The paper should disclose whether the full research project required more compute
804

than the experiments reported in the paper (e.g., preliminary or failed experiments that
805

didn’t make it into the paper).
806

9. Code Of Ethics
807

Question: Does the research conducted in the paper conform, in every respect, with the
808

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
809

Answer: [Yes]
810

Justification: Research is done ethically with na lot of concerns for reproduciblity and
811

validity of the results.
812

Guidelines:
813

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
814

• If the authors answer No, they should explain the special circumstances that require a
815

deviation from the Code of Ethics.
816

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
817

eration due to laws or regulations in their jurisdiction).
818

10. Broader Impacts
819

Question: Does the paper discuss both potential positive societal impacts and negative
820

societal impacts of the work performed?
821

Answer: [NA]
822

Justification: No societal impact, our work simply proposes a classification/regression
823

tree algorithms like many before. So the ethical and societal concers are inherited from
824

supervised learning ones.
825

Guidelines:
826

• The answer NA means that there is no societal impact of the work performed.
827

• If the authors answer NA or No, they should explain why their work has no societal
828

impact or why the paper does not address societal impact.
829

• Examples of negative societal impacts include potential malicious or unintended uses
830

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
831

(e.g., deployment of technologies that could make decisions that unfairly impact specific
832

groups), privacy considerations, and security considerations.
833

• The conference expects that many papers will be foundational research and not tied
834

to particular applications, let alone deployments. However, if there is a direct path to
835

any negative applications, the authors should point it out. For example, it is legitimate
836

to point out that an improvement in the quality of generative models could be used to
837

generate deepfakes for disinformation. On the other hand, it is not needed to point out
838

that a generic algorithm for optimizing neural networks could enable people to train
839

models that generate Deepfakes faster.
840

• The authors should consider possible harms that could arise when the technology is
841

being used as intended and functioning correctly, harms that could arise when the
842

technology is being used as intended but gives incorrect results, and harms following
843

from (intentional or unintentional) misuse of the technology.
844

• If there are negative societal impacts, the authors could also discuss possible mitigation
845

strategies (e.g., gated release of models, providing defenses in addition to attacks,
846

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
847

feedback over time, improving the efficiency and accessibility of ML).
848

11. Safeguards
849

Question: Does the paper describe safeguards that have been put in place for responsible
850

release of data or models that have a high risk for misuse (e.g., pretrained language models,
851

image generators, or scraped datasets)?
852

Answer: [NA]
853

Justification: no risk (see above)
854

Guidelines:
855

23


---Page Break---
• The answer NA means that the paper poses no such risks.
856

• Released models that have a high risk for misuse or dual-use should be released with
857

necessary safeguards to allow for controlled use of the model, for example by requiring
858

that users adhere to usage guidelines or restrictions to access the model or implementing
859

safety filters.
860

• Datasets that have been scraped from the Internet could pose safety risks. The authors
861

should describe how they avoided releasing unsafe images.
862

• We recognize that providing effective safeguards is challenging, and many papers do
863

not require this, but we encourage authors to take this into account and make a best
864

faith effort.
865

12. Licenses for existing assets
866

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
867

the paper, properly credited and are the license and terms of use explicitly mentioned and
868

properly respected?
869

Answer: [Yes]
870

Justification: Appropriate credits is given when necessary and other code not from the
871

authors are open sourced. Please see A L.
872

Guidelines:
873

• The answer NA means that the paper does not use existing assets.
874

• The authors should cite the original paper that produced the code package or dataset.
875

• The authors should state which version of the asset is used and, if possible, include a
876

URL.
877

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
878

• For scraped data from a particular source (e.g., website), the copyright and terms of
879

service of that source should be provided.
880

• If assets are released, the license, copyright information, and terms of use in the
881

package should be provided. For popular datasets, paperswithcode.com/datasets
882

has curated licenses for some datasets. Their licensing guide can help determine the
883

license of a dataset.
884

• For existing datasets that are re-packaged, both the original license and the license of
885

the derived asset (if it has changed) should be provided.
886

• If this information is not available online, the authors are encouraged to reach out to
887

the asset’s creators.
888

13. New Assets
889

Question: Are new assets introduced in the paper well documented and is the documentation
890

provided alongside the assets?
891

Answer: [Yes]
892

Justification: Code is documented to the best we could. Please see L.
893

Guidelines:
894

• The answer NA means that the paper does not release new assets.
895

• Researchers should communicate the details of the dataset/code/model as part of their
896

submissions via structured templates. This includes details about training, license,
897

limitations, etc.
898

• The paper should discuss whether and how consent was obtained from people whose
899

asset is used.
900

• At submission time, remember to anonymize your assets (if applicable). You can either
901

create an anonymized URL or include an anonymized zip file.
902

14. Crowdsourcing and Research with Human Subjects
903

Question: For crowdsourcing experiments and research with human subjects, does the paper
904

include the full text of instructions given to participants and screenshots, if applicable, as
905

well as details about compensation (if any)?
906

Answer:[NA]
907

24


---Page Break---
Justification: no user studies
908

Guidelines:
909

• The answer NA means that the paper does not involve crowdsourcing nor research with
910

human subjects.
911

• Including this information in the supplemental material is fine, but if the main contribu-
912

tion of the paper involves human subjects, then as much detail as possible should be
913

included in the main paper.
914

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
915

or other labor should be paid at least the minimum wage in the country of the data
916

collector.
917

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
918

Subjects
919

Question: Does the paper describe potential risks incurred by study participants, whether
920

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
921

approvals (or an equivalent approval/review based on the requirements of your country or
922

institution) were obtained?
923

Answer: [NA]
924

Justification: no user studies
925

Guidelines:
926

• The answer NA means that the paper does not involve crowdsourcing nor research with
927

human subjects.
928

• Depending on the country in which research is conducted, IRB approval (or equivalent)
929

may be required for any human subjects research. If you obtained IRB approval, you
930

should clearly state this in the paper.
931

• We recognize that the procedures for this may vary significantly between institutions
932

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
933

guidelines for their institution.
934

• For initial submissions, do not include any information that would break anonymity (if
935

applicable), such as the institution conducting the review.
936

25


---Page Break---
