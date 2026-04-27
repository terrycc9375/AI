Automating Data Annotation under Strategic Human
Agents: Risks and Potential Solutions

Tian Xie
Computer Science and Engineering
the Ohio State University
Columbus, OH 43210
xie.1379@osu.edu

Xueru Zhang
Computer Science and Engineering
the Ohio State University
Columbus, OH 43210
zhang.12807@osu.edu

Abstract

As machine learning (ML) models are increasingly used in social domains to make
consequential decisions about humans, they often have the power to reshape data
distributions. Humans, as strategic agents, continuously adapt their behaviors in
response to the learning system. As populations change dynamically, ML systems
may need frequent updates to ensure high performance. However, acquiring high-
quality human-annotated samples can be highly challenging and even infeasible
in social domains. A common practice to address this issue is using the model
itself to annotate unlabeled data samples. This paper investigates the long-term
impacts when ML models are retrained with model-annotated samples when they
incorporate human strategic responses. We first formalize the interactions between
strategic agents and the model and then analyze how they evolve under such
dynamic interactions. We find that agents are increasingly likely to receive positive
decisions as the model gets retrained, whereas the proportion of agents with positive
labels may decrease over time. We thus propose a refined retraining process to
stabilize the dynamics. Last, we examine how algorithmic fairness can be affected
by these retraining processes and find that enforcing common fairness constraints at
every round may not benefit the disadvantaged group in the long run. Experiments
on (semi-)synthetic and real data validate the theoretical findings.

1
Introduction

Training data at 
Strategic agents arriving at 

 

 

Training data at 

Best respond 

Model Annotation
Human Annotation

Train

Figure 1: Illustration of updating the
training data from t to t+1 during the re-
training process with strategic feedback

As machine learning (ML) is increasingly used to auto-
mate human-related decisions (e.g., in lending, hiring, col-
lege admission), there is a growing concern that these de-
cisions are vulnerable to human strategic behaviors. With
the knowledge of decision policy, humans may adapt their
behavior strategically in response to ML models, e.g., by
changing their features at costs to receive favorable out-
comes. A line of research called Strategic Classification
studies such problems by formulating mathematical mod-
els to characterize strategic interactions and developing
algorithms robust to strategic behavior [1, 2]. Among
existing works, most studies focus on one-time model de-
ployment where an ML model is trained and applied to a
fixed population of strategic agents once.

However, practical ML systems often need to be retrained periodically to ensure high performance
on the current population. As the ML model gets updated, human behaviors also change accordingly.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
To prevent the potential adverse outcomes, it is critical to understand how the strategic population is
affected by the model retraining process. Traditionally, the data used for retraining models can be
constructed manually with human annotations (e.g., ImageNet). However, acquiring a large amount
of human-annotated samples can be highly difficult and even infeasible, especially in human-related
applications, e.g., in automated hiring where an ML model is used to identify qualified applicants,
even an experienced interviewer needs time to label an applicant.

Motivated by a recent practice of automating data annotation for retraining large-scale ML models
[3, 4], we study strategic classification in a sequential framework where an ML model is periodically
retrained by a decision-maker with both human and model-annotated samples. These updated models
are deployed sequentially on agents who may modify their features to receive favorable outcomes.
Since ML models affect agent behavior and the agent strategic feedback can further be captured when
retraining the future model, their interactions drive both to change dynamically over time. However,
it remains unclear how the two evolve under such dynamics and what long-term effects one may have
on the other.

1
0
1
2

2

0

qualified
unqualified
true classifier
learned classifier

1
0
1
2

4

2

0

1
0
1
2

4

2

0

2

Figure 2: Evolution of the student distribution and ML model at t = 0
(left), t = 5 (middle), and t = 14 (right): each student has two features.
At each time, a classifier is retrained with both human and model-
annotated samples, and students best respond to be admitted (Fig. 1).
Over time, the learned classifier (black lines) deviates from ground
truth (green lines).

To further illustrate our
problem, consider an ex-
ample of college admission
where new students from a
population apply each year.
In the t-th year, an ML
model ft is learned from
a training dataset St and
used to make admission de-
cisions. For students who
apply in the (t + 1)-th year,
they will best respond to
the model ft in the previous
year (e.g., preparing the ap-
plication package in a way that maximizes the chance of getting admitted). Meanwhile, the college
retrains the classifier ft+1 using a new training dataset St+1 consisting of previous training data
St, new human-annotated samples, and new model-annotated samples (i.e., previous applicants
annotated by the most recent model ft). The model ft+1 is then used to make admission decisions in
the (t + 1)-th year. This process continues over time and we demonstrate how the training dataset St
is updated to St+1 in Fig. 1. Under such dynamics, both the ML system and the strategic population
change over time and may lead to unexpected long-term consequences. An illustrating example is
given in Fig. 2.

In this paper, we examine the evolution of the ML model and the agent data distribution. We ask: 1)
How does the agent population evolve when the model is retrained with strategic feedback? 2) How
is the ML system affected by the agent’s strategic response? 3) If agents come from multiple social
groups, how can model retraining further impact algorithmic fairness? Can imposing group fairness
constraints during model training bring long-term societal benefits?

Compared to prior studies on strategic classification under sequential settings [5, 6, 7] that mainly
focused on developing a classifier robust to strategic agents, we study the long-term impacts of
model retraining with agent strategic feedback. Instead of assuming actual labels are available while
retraining, we consider more practical scenarios with model-annotated samples. Although the risks
on accuracy [3] and fairness [8] of using model-annotated samples to retrain models have been
highlighted, ours is the first to incorporate strategic feedback from human agents. In App. C, we
discuss more related works in detail. Our contributions are summarized as follows:

1. We formulate the problem of model retraining with human strategic feedback (Sec. 2).

2. We theoretically characterize the evolution of the expected acceptance rate (i.e., the proportion
of agents receiving positive classifications), qualification rate (i.e., the proportion of agents
with positive labels), and the classifier bias (i.e., the discrepancy between acceptance rate and
qualification rate) under the retraining process. We show that the acceptance rate increases over
time under retraining, while the actual qualification rate may decrease under certain conditions.
The dynamics of classifier bias are more complex depending on the systematic bias of human-
annotated samples. Finally, we propose an approach to stabilize the dynamics (Sec. 3).

2


---Page Break---
3. We consider settings where agents come from multiple social groups and investigate how inter-
group fairness can be affected by the model retraining process; we also investigate the long-term
effects of fairness intervention at each round of model retraining (Sec. 4).
4. We conduct experiments on (semi-)synthetic and real data to verify the theorems (Sec. 5, App. E,
App. F).

2
Problem Formulation

Consider a population of agents who are subject to certain ML decisions (e.g., admission/hiring
decisions) and join the decision-making system in sequence. Each agent has observable continuous
features X ∈Rd and a hidden binary label Y ∈{0, 1} indicating its qualification state ("1" being
qualified and "0" being unqualified). Let PXY be the joint distribution of (X, Y ) which is fixed over
time, and PX, PY |X be the corresponding marginal and conditional distributions. Assume PX, PY |X
are continuous with non-zero probability mass everywhere in their domain. For agents who join the
system at time t, the decision-maker uses a classifier ft : Rd →{0, 1} to make decisions. Note that
the decision-maker does not know PXY and can only learn ft from the training dataset at t [9].

Agent best response. Agents who join the system at time t can adapt their behaviors based on
the latest classifier ft−1 and change their features X strategically. We denote the resulting data
distribution as P t
XY . Specifically, given original features X = x, agents have incentives to change
their features at costs to receive positive classification outcomes, i.e., by maximizing utility

xt = arg maxz {ft−1(z) −c(x, z)}
(1)

where distance function c(x, z) ≥0 measures the cost for an agent to change features from x to z. In
this paper, we consider c(x, z) = (z −x)T B(z −x) for some d × d positive semidefinite matrix B,
allowing heterogeneous costs for different features. After agents best respond, their data distribution
changes from PXY to P t
XY . In this paper, we term PXY agent prior-best-response distribution
and P t
XY post-best-response distribution. We consider natural settings that (i) agents need time to
adapt their behaviors and their responses are delayed [10]: they act based on the latest classifier ft−1
they are aware of, not the one they receive; (ii) agent behaviors are benign and feature changes can
genuinely affect their underlying labels, so feature-label relationship P t
Y |X = PY |X is fixed over
time [9, 11].

Human-annotated samples and systematic bias. At each round t, we assume the decision-maker
can draw a limited number of unlabeled samples from the prior-best-response distribution PX.1 With
some prior knowledge (possibly biased), the decision-maker can annotate these features and generate
human-annotated samples So,t. We assume the quality of human annotations is consistent, so So,t
at any t is drawn from a fixed probability distribution Do
XY with marginal distribution Do
X = PX.
Because human annotations may not be the same as true labels, Do
Y |X can be biased compared to
PY |X. We define such difference as the decision-maker’s systematic bias, formally stated below.

Definition 2.1 (Systematic bias). Define µ(Do, P) := Ex∼PX[Do
Y |X(1|x) −PY |X(1|x)]. The
decision-maker has a systematic bias if µ(Do, P) > 0 (overestimation) or < 0 (underestimation).

Def. 2.1 implies that the decision-maker has a systematic bias when it labels a larger (or smaller)
proportion of agents as qualified compared to the ground truth. In App. B, we present numerous
examples of systematic bias in real applications where the decision-maker has different systematic
biases towards different demographic groups. Generally, the systematic bias may or may not exist
and we study both scenarios in the paper.

Model-annotated samples. In addition to human-annotated samples, the decision-maker at each
round t can also utilize the most recent classifier ft−1 to generate model-annotated samples for
training ft. Specifically, let {xi
t−1}N
i=1 be N post-best-response features ((1)) acquired from the

1We consider natural settings where the human-annotated samples are drawn from fixed PX and the process
is independent of the decision-making process, i.e., the agents who best respond to ft−1 are classified by
model ft and the decision-maker never confuses them with human annotations. Instead, human annotation is
a separate process for the decision-maker to obtain additional information about the whole population (e.g.,
by first acquiring data from public datasets or third parties, and then labeling them to estimate the population
distribution PXY ). We provide an additional example and also consider the situation where human-annotated
samples are drawn from post-best-response distribution P t
X in App. D.1.

3


---Page Break---
agents coming at t −1, the decision-maker uses ft−1 to annotate the samples and obtain model-
annotated samples Sm,t−1 = {xi
t−1, ft−1(xi
t−1)}N
i=1. Both human and model-annotated samples
are used to retrain the classifier at t.

Classifier’s retraining process. With the human and model-annotated samples, we next introduce
how the model is retrained by the decision-maker over time. Denote the training dataset at t as St.
Initially, the decision-maker trains f0 with a human-annotated training dataset S0 = So,0. Then the
decision-maker updates ft every round to make decisions about agents, and it learns ft using empirical
risk minimization (ERM) with training dataset St. Similar to studies in strategic classification [2], we
consider linear classifier in the form of ft(x) = 1(ht(x) ≥θ) where ht : R →[0, 1] is the scoring
function (e.g., logistic function) and ht ∈H. At each round t ≥1, St consists of three components:
existing training samples St−1, N new model-annotated and K new human-annotated samples:

St = St−1 ∪Sm,t−1 ∪So,t−1, ∀t ≥1
(2)

Since annotating agents is usually time-consuming and expensive, we have N ≫K in practice. The
complete retraining process is shown in Alg. 1 (App. A).

Given the post-best-response distribution P t
XY , we can define the associated qualification rate as the
probability that agents are qualified, i.e.,

Q(P t) = E(x,y)∼P t
XY [y] .

For the classifier ft deployed on marginal feature distribution P t
X, we define acceptance rate as the
probability that agents are classified as positive, i.e.,

A(ft, P t) = Ex∼P t
X[ft(x)].

Since St is randomly sampled at all t, the resulting classifier ft and agent best response are also
random. Denote Dt
XY as the probability distribution of sampling from St and recall that Do
XY is the
distribution for human-annotated So,t, we can further define the expectations of Q(P t), A(ft, P t)
over the training dataset:

qt := ESt−1[Q(P t)];
at := ESt [A(ft, P t)] ,

where qt is the expected actual qualification rate of agents after they best respond, note that the
expectation is taken with respect to St−1 because the distribution P t
XY is the result of agents
responding to ft−1 which is trained with St−1; at is the expected acceptance rate of agents at time t.

Dynamics of qualification rate & acceptance rate. Under the model retraining process, both the
model ft and agent distribution P t
XY change over time. One goal is to understand how the agents
and the ML model interact and impact each other in the long run. Specifically, we are interested in
the dynamics of the following variables:

1. Qualification rate qt: it measures the qualification of agents and indicates the social welfare.
2. Acceptance rate at: it measures the likelihood that an agent can receive positive outcomes and
indicates the applicant welfare.
3. Classifier bias ∆t = |at −qt|: it is the discrepancy between the acceptance rate and the
true qualification rate, measuring how well the decision-maker can approximate agents’ actual
qualification rate and can be interpreted as decision-maker welfare.

In the rest of the paper, we study the dynamics of qt, at, ∆t and we aim to answer the following
questions: 1) How do the qualification rate qt, acceptance rate at, and classifier bias ∆t evolve under
the dynamics? 2) How can the evolution of the system be affected by the decision-maker’s retraining
process? 3) What are the impacts of the decision-maker’s systematic bias? 4) If we further consider
agents from multiple social groups, how can the retraining process affect inter-group fairness?

3
Dynamics of the Agents and Model

In this section, we examine the evolution of qualification rate qt, acceptance rate at, and classifier
bias ∆t. We aim to understand how applicant welfare (Sec. 3.1), social welfare (Sec. 3.2), and
decision-maker welfare (Sec. 3.3) are affected by the retraining process in the long run. We first
introduce some assumptions used for the theorems.

4


---Page Break---
Assumption 3.1. Hypothesis class H can perfectly learn the training data distribution Dt
Y |X, i.e.,
∃h∗
t ∈H such that h∗
t (x) = Dt
Y |X(1|x).

With Assumption 3.1, we avoid the effects of learning error on the system dynamics; this allows us to
focus on the dynamic interactions between strategic agents and ML system. Although the theoretical
analysis relies on the assumption, our experiments in Sec. 5 and App. F show consistent results with
theorems. We further assume the monotone likelihood ratio property holds for Do
XY and PXY .

Assumption 3.2. Let x[m] be the mth dimension of x ∈Rd, then Do
Y |X(1|x) and PY |X(1|x) are
continuous and monotonically increasing in x[m], ∀m = 1, · · · , d while other dimensions are fixed.

Note that the Assumption 3.2 is mild and widely used in previous literature [12, 13]. It can be satisfied
by many distributional families such as exponential, Gaussian, and mixtures of exponential/Gaussian.
It implies that agents are more likely to be qualified as feature value increases.

3.1
Applicant welfare: dynamics of acceptance rate

We first examine the dynamics of at = ESt [A(ft, P t)]. Intuitively, under Assumption 3.1, all
classifiers can fit the training data well. Then the model-annotated samples Sm,t−1 generated from
post-best-response agents would have a higher qualification rate than the qualification rate of training
data St−1. As a result, the training data St augmented with Sm,t−1 has a higher proportion of
qualified agents than the qualification rate of St−1, thereby producing a more "generous" classifier ft
with a larger at. This reinforcing process can be formally stated in Thm. 3.3.
Theorem 3.3 (Evolution of at). Under the retraining process, the acceptance rate of the agents that
join the system increases over time, i.e., at > at−1, ∀t ≥1.

Figure 3: Increasing acceptance rate from at to at+1.
Unquali-
fied/qualified agents are shown as circles/squares, while the admit-
ted/rejected agents are shown in red/blue. New agents coming at t + 1
are shown in hollow. The left plot shows the training set St containing
2 unqualified (red circle) and 2 qualified agents (blue square) and at
is 0.5. The middle plot shows the agents coming at t best respond
to ft−1. After the responses, 3 of 4 agents are qualified (blue square)
and 1 is still unqualified (blue circle). However, all 4 agents are an-
notated as "qualified" (blue). The right plot shows the training set
St+1 containing all points of the left and middle plot, plus two new
human-annotated points (hollow points). All blue points are labeled as
1 and the red points are 0. So St+1 has more samples with a positive
label (0.7), resulting in ft+1 accepting a higher proportion of agents.

We prove Thm. 3.3 by
mathematical induction in
App. G.3 and Fig. 3 illus-
trates the theorem. When
agents best respond, the
decision-maker tends to ac-
cept more agents. We can
further show that when the
number of model-annotated
samples N is large com-
pared to the number of
human-annotated samples
K, the classifier will ulti-
mately accept all agents in
the long run (Prop. 3.4).
Proposition 3.4. For any
PXY , Do, B, there exists a
threshold λ > 0 such that
limt→∞at = 1 whenever
K
N < λ.

The specific value of λ in Prop. 3.4 depends on PXY , Do, B, which is difficult to find analytically.
Nonetheless, we illustrate in Sec. 5 that when K

N = 0.05, at tends to approach 1 in numerous datasets.
Since the human-annotated samples are often difficult to attain (due to time and labeling costs), the
condition in Prop. 3.4 is easy to satisfy in practice.

3.2
Social welfare: dynamics of qualification rate

Next, we study the dynamics of qualification rate qt = ESt−1[Q(P t)]. Unlike the acceptance rate at
which always increases during the retraining process, the evolution of qt is more complicated and
depends on agent prior-best-response distribution PXY .

Specifically, let q0 = Q(P) = E(x,y)∼PXY [y] be the initial qualification rate, then the difference
between qt and q0 can be interpreted as the amount of improvement (i.e., increase in label) agents

5


---Page Break---
gain from their strategic behavior at t. This is determined by (i) the proportion of agents that decide
to change their features at costs (depends on PX), and (ii) the improvement agents can expect upon
changing features (depends on PY |X). Thus, the dynamics of qt depend on PXY . Despite the intricate
nature of dynamics, we can still derive a condition under which qt decreases monotonically.
Theorem 3.5 (Evolution of qt). Consider the setting where the d-dimensional feature space X ∈
Rd where FX(x), PX(x), PY |X(1|x) are the cumulative distribution function, probability density
function and the labeling function when Y = 1. Denote J = {x|f0(x) = 0} as the half-space in
Rd determined by the classifier f0. Under the retraining process, ∀t ≥1, qt+1 ≤qt if either of
the following conditions holds: (i) FX and PY |X(1|x) are convex on J ; (ii) for each dimension
x[i], i ∈[d], FX(x) and PY |X(1|x) are convex with respect to x[i].

Note that qt+1 ≤qt in Thm. 3.5 holds only for t ≥1. Because agent behavior can only improve
their labels, prior-best-response q0 always serves as the lower bound of qt. The half-space J in
Thm. 3.5 specifies the region in feature space where agents have incentives to change their features.
The convexity of FX and PY |X(1|x) ensure that as ft evolves from t = 1: (i) fewer agents choose
to improve their features, and (ii) agents expect less improvement from feature changes. Thus, qt
decreases over time. Conditions in Thm. 3.5 can be satisfied by common distributions PX (e.g.,
Uniform, Beta(α, 1) with α > 1) and labeling functions PY |X(1|x) (e.g., linear function, quadratic
functions with degree greater than 1). The proof and a more general analysis are shown in App. G.5.
We also show that Thm. 3.5 is valid under diverse experimental settings (Sec. 5, App. E, App. F).

3.3
Decision-maker welfare: dynamics of classifier bias

Sec. 3.1 and 3.2 show that as the classifier ft gets updated over time, agents are more likely to get
accepted (at increases). However, their true qualification rate qt (after the best response) may actually
decrease. It indicates that the decision-maker’s misperception about agents varies over time. Thus,
this section studies the dynamics of classifier bias ∆t = |at −qt|. Our results show that the evolution
of ∆t is largely affected by the decision-maker’s systematic bias µ(Do, P) as defined in Def. 2.1.
Theorem 3.6 (Evolution of ∆t). Starting from t = 1 at the retraining process and under conditions
in Thm. 3.5:
1. If µ(Do, P) = 0, i.e., the systematic bias does not exist, then ∆t increases over time.
2. If µ(Do, P) > 0, i.e., the decision-maker overestimates agent qualification, then ∆t increases
over time.
3. If µ(Do, P) < 0, i.e., the decision-maker underestimates agent qualification, ∆t either monotoni-
cally decreases or first decreases but then increases.

Thm. 3.6 highlights the potential risks of the model retraining process and is proved in App. G.6.
Originally, the purpose of retraining the classifier was to ensure accurate decisions on the targeted
population. However, when agents behave strategically, the retraining may lead to adverse outcomes
by amplifying the classifier bias. Meanwhile, though systematic bias is usually an undesirable
factor to eliminate when learning ML models, it may help mitigate classifier bias to improve the
decision-maker welfare in the retraining process, i.e., ∆t decreases when µ(Do, P) < 0.

3.4
Intervention to stabilize the dynamics

Sec. 3.1- 3.3 show that as the model is retrained from strategic agents, at, qt, ∆t are unstable and may
change monotonically over time. Next, we introduce an effective approach to stabilizing the system.

From the above analysis, we know that one reason that makes qt, at, ∆t evolve is agent’s best
response, i.e., agents improve their features strategically to be accepted by the most recent model,
which leads to a higher qualification rate of model-annotated samples (and the resulting training
data), eventually causing at to deviate from qt. Thus, to mitigate such deviation, we can improve the
quality of model annotation. Our method is proposed based on this idea, which uses a probabilistic
sampler [3] when producing model-annotated samples.

Specifically, at each time t, instead of adding Sm−1,o = {xi
t−1, ft−1(xi
t−1)}N
i=1 (samples annotated
by the model ft−1) to training data St (2), we use the probabilistic model ht−1(x) to annotate each
sample according to the following: For each sample x, we label it as 1 with probability ht−1(x), and
as 0 otherwise. Here ht−1(x) ≈Dt−1
Y |X(1|x) is the estimated posterior probability learned from St−1

6


---Page Break---
(e.g., using logistic model). We call the procedure refined retraining process if model-annotated
samples are generated in this way based on a probabilistic sampler.

Fig. 3 also illustrates the above idea: agents best respond to ft−1 (middle plot) to improve and ft
will label both as 1. By contrast, a probabilistic sampler ht only labels a fraction of them as 1. This
alleviates the influence of agents’ best responses to stabilize the dynamics of at, qt, ∆t. Prop. D.2
and App. F.3 provide proofs and more experiments for the refined retraining process.

4
Impacts on Algorithmic Fairness

In this section, we further consider agents from multiple social groups and investigate how group
fairness can be affected by the model retraining process. Similar to prior studies in fair ML [12, 14, 15],
we assume the decision-maker knows the group identity of each agent and uses group-dependent
classifiers to make decisions. WLOG, we present the results for any pair of two groups i, j. Among
the two groups, we define the group with a smaller acceptance rate under unconstrained optimal
classifiers as disadvantaged group.

4.1
Impacts of systematic bias & model retraining

We first consider the situation where two groups have no innate difference: they have the same
prior-best-response feature distribution and the same cost matrix B to change features. However, the
decision-maker has a systematic bias in favor of group i more than group j, making i the advantaged
group and j the disadvantaged group. We consider the fairness metric demographic parity (DP)
[16], which measures unfairness as the difference in acceptance rate across two groups. Extension
to other fairness metrics such as equal opportunity [17] is discussed in App. D.2. Thm. 4.1 below
shows the long-term impacts of refined retraining process (i.e., model-annotated samples generated
with probabilistic sampler ht) and original model retraining process (i.e., model-annotated samples
generated with model ft) on group unfairness.
Theorem 4.1 (Impacts of model retraining on unfairness). When groups i, j have no innate difference,
but j is disadvantaged due to systematic bias: (i) If applying original retraining process to both
groups with K

N satisfying Prop. 3.4, then group j will stay disadvantaged until all agents in both
groups are accepted to achieve perfect fairness; (ii) If applying refined retraining process to both
groups, then group j will stay disadvantaged and the unfairness remains the same in the long run;
(iii) If applying refined retraining process to group i but the original process to group j with K

N
satisfying Prop. 3.4, then unfairness first decreases after certain rounds of retraining until group j
becomes advantaged, then unfairness increases.

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

Figure 4: Unfairness (DP) when the refined retraining pro-
cess is applied to both groups (left), original retraining pro-
cess is applied to both groups (middle), refined retraining
process is only applied to group i (right) under dataset 2.

Thm. 4.1 shows that original and re-
fined retraining processes impact dif-
ferently on group fairness as proved
in App. G.9. Specifically, the origi-
nal retraining process ultimately attains
"trivial" perfect fairness by accepting
all agents, whereas the refined retrain-
ing process stabilizes the dynamics but
unfairness always exists. Interestingly,
applying disparate retraining strategies
to two groups may result in perfect fair-
ness in the middle of retraining process.
This suggests that it may be beneficial for the decision-maker to monitor the fairness measure during
the retraining and execute an early stopping mechanism to attain almost perfect DP fairness, i.e., stop
retraining models early once unfairness reaches the minimum. As shown in the right plot of Fig. 4,
when refined retraining is only applied to group i, unfairness is minimized at t = 5.

4.2
Impact of short-term fairness intervention

Sec. 4.1 proposed a solution of "disparate retraining with early stopping" to mitigate unfairness. A
more common method to maintain fairness throughout the retraining process is to enforce certain
fairness constraints every time when updating the models. Next, we consider this method where

7


---Page Break---
refined retraining process is applied to both groups (to stabilize dynamics) and a fairness constraint
is imposed at each round of model retraining. Unlike Sec. 4.1, we consider a general setting where
two groups may have different feature distributions and different cost matrices B, but feature-label
relation PY |X(1|x) is the same across groups.

Finding fair models. For each group s, we use superscript s to denote the group-specific distribu-
tions/metrics listed in Sec. 2 (e.g., P s,t
XY ). At each round t, the decision-maker first trains a linear
model f s
t = 1(hs
t(x) ≥θ) for group s. According to Assumption 3.1 and Prop. D.2, hs
t is expected
to be Do,s
Y |X(1|x) = P s
Y |X(1|x) + µs, where µs is the systematic bias towards group s. Denote θs
t as
the original optimal threshold at t without the fairness intervention, and let as
t be the acceptance rate
for group s under θs
t , then the decision-maker can tune the thresholds to get fair-optimal thresholds
eθs
t (a pair of thresholds satisfying the fairness constraint with the largest aggregated accuracy).

Noisy agent best response. To ensure there always exists a pair of thresholds that satisfy DP fairness
constraint (i.e., equal acceptance rate), each group’s post-best-response distribution needs to be
continuous. However, when agents modify their features based on (1), the aggregate response
necessarily exhibits discontinuities [18]. To tackle this issue, we consider the noisy best response
model proposed by Jagadeesan et al. [18], which assumes the agents only have imperfect and noisy
information of decision threshold. Formally, given decision threshold θ, each agent best responds to
θ + ϵ where ϵ is a noise independently sampled from a zero mean distribution with finite variance σ2.
Under noisy best response, we investigate the impacts of fairness intervention.
Theorem 4.2 (Impact of fairness intervention). Suppose group j is disadvantaged from round 0 to
t, i.e., group j has a smaller acceptance rate than group i under unconstrained optimal thresholds
{θi
τ, θj
τ}, ∀τ ≤t. Let σ2
t be the variance of the noisy best response at t. We have the following:

(i) Without fairness intervention, ∀σt, there always exists feature distributions and cost matrices
under which group j switches to be advantaged at t + 1;

(ii) With fairness intervention, if σt < (θj
t −eθj
t)
q

ai
0 −aj
t, then group j always remains disadvan-
taged at t + 1.

Thm. 4.2 shows that without fairness intervention, the originally disadvantaged group j can flip to be
advantaged. In contrast, the fairness intervention helps maintain the disadvantaged and advantaged
groups when the agent’s perception of the decision rule is sufficiently accurate. Note that the bound on

σ is well-defined. Since group j is disadvantaged, ai
0 > aj
0 always holds. If σt < (θj
t −eθj
t)
q

ai
0 −aj
t
holds, then it is guaranteed that ai
0 > aj
t+1 (see App. G.10 for details). This result implies that the
disadvantaged group, by losing their chance to become advantaged, may not benefit from fairness
intervention in the long run.

5
Experiments

We conduct experiments on two synthetic (Uniform, Gaussian), one semi-synthetic (German Credit
[19]), and one real dataset (Credit Approval [20]) to validate the dynamics of at, qt, ∆t and the
unfairness 2. Note that only the Uniform dataset satisfies all assumptions and the conditions in
our theoretical analysis, while the Gaussian and German Credit datasets violate the conditions in
Thm. 3.5. The Credit Approval dataset violates all assumptions and conditions of the main paper.
The decision-maker trains logistic regression models for all experiments using stochastic gradient
descent (SGD) over T steps. We present the experimental results of the Gaussian and German Credit
datasets to illustrate the dynamics of at, qt, ∆t in this section, while the results for Uniform and
Credit Approval data are similar and shown in App. E.

Table 1: Gaussian Dataset Setting

PXk(xk)
PY |X(1|x)
n, r, T, q0

N(0, 0.52)
(1 + exp(−x1 −x2))−1
100, 0.05, 15, 0.5

Gaussian data. We consider a synthetic dataset with Gaussian
distributed PX. PY |X is logistic and satisfies Assumption 3.2
but not the conditions of Thm. 3.5. We assume agents have two
independent features X1, X2 and are from two groups i, j with

different sensitive attributes but identical joint distribution PXY . Their cost matrix is B =

5
0
0
5



2https://github.com/osu-srml/Automating-Data-Annotation-under-Strategic-Human-Agents

8


---Page Break---
and the initial qualification rate is q0 = 0.5. We assume the decision-maker has a systematic bias by
overestimating (resp. underestimating) the qualification of agents in the advantaged group i (resp.
disadvantaged group j), which is modeled as increasing Do
Y |X(1|x) to be 0.1 larger (resp. smaller)
than PY |X(1|x) for group i (resp. group j). For the retraining process, we let r = K

N = 0.05 (i.e., the
number of model-annotated samples N = 2000, which is sufficiently large compared to the number
of human-annotated samples K = 100). Table 1 summarizes the dataset information, and the joint
distributions are visualized in App. F.1.

We verify the results in Sec. 3 by illustrating the dynamics of at, qt, ∆t for both groups (Fig. 5a).
Since our evolution results are in expectation, we perform n = 100 independent runs of experiments
for every parameter configuration and show the averaged outcomes. The results are consistent with
Thm. 3.3, 3.5 and 3.6: (i) acceptance rate at (red curves) increases monotonically; (ii) qualification
rate qt decreases monotonically starting from t = 1 (since strategic agents only best respond from
t = 1); (iii) classifier bias ∆t evolves differently for different groups and it may reach the minimum
after a few rounds of retraining.

We further test the robustness of system dynamics against agent noisy response, where we assume
agents estimate their outcomes as bft(x) = ft(x) + ϵ with ϵ ∼N(0, 0.1). We present the dynamics
of at, qt, ∆t for both groups in Fig. 6a which are similar to Fig. 5a, demonstrating the robustness of
our theorems.

German Credit dataset [19]. This dataset includes features for predicting individuals’ credit risks.
It has 1000 samples and 19 numeric features, which are used to construct a larger-scaled dataset.
Specifically, we fit a kernel density estimator for all 19 features to generate 19-dimensional features,
the corresponding labels are sampled from the distribution PY |X which is estimated from data by
fitting a logistic classifier with 19 features. Given this dataset, the first 10 features are used to train
the classifiers. The attribute "sex" is regarded as the sensitive attribute. The systematic bias is created
by increasing/decreasing PY |X by 0.06. Other parameters n, r, T, q0 are the same as Table 1. Since
PY |X is a logistic function, Assumption 3.2 can be satisfied easily as illustrated in App. F.1.

We verify the results in Sec. 3 by illustrating the dynamics of at, qt, ∆t for both groups (Fig. 5b). The
results are consistent with Thm. 3.3, 3.5 and 3.6: (i) acceptance rate at (red curves) always increases;
(ii) qualification rate qt (blue curves) decreases starting from t = 1 (since strategic agents only best
respond from t = 1); (iii) classifier bias ∆t (black curves) evolve differently for different groups.
Finally, similar to Fig. 5b, Fig. 6b demonstrates the results are still robust under the noisy setting.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Gaussian: group i (left) and j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) German: group i (left) and j (right)

Figure 5: Dynamics of at, qt, ∆t under the perfect information setting

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Gaussian: for group i (left) and j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) German: group i (left) and j (right)

Figure 6: Dynamics of at, qt, ∆t under the noisy setting

Additional experiments. We provide more results in App. F to: (i) verify Thm. 3.3 and Thm. 3.6;
(ii) Visualize the influence of each factor including training rounds, cost matrices, the ratio between
human-annotated and model-annotated samples, whether the agents are strategic, and whether certain

9


---Page Break---
assumptions are violated; (iii) visualize the evolution of unfairness when different retraining strategies
are applied to different groups.

6
Conclusion & Limitations

This paper studies the dynamics where strategic agents interact with an ML system retrained over
time with model-annotated and human-annotated samples. We rigorously studied the evolution of
applicant welfare, decision-maker welfare, and social welfare. Such results highlight the potential
risks of retraining classifiers when agents are strategic. The paper also provides a comprehensive
analysis on the fairness dynamics associated with the retraining process, revealing that the fairness
intervention may not bring long-term benefits. To ease the negative social impacts, we provide
mechanisms to stabilize the dynamics and an early stopping mechanism to maintain fairness. However,
our theoretical results rely on certain assumptions and we should first verify these conditions before
adopting the results of this paper, which may be challenging in real-world applications.

Acknowledgement

This material is based upon work supported by the U.S. National Science Foundation under award IIS-
2202699 and IIS-2416895, by OSU President’s Research Excellence Accelerator Grant, and grants
from the Ohio State University’s Translational Data Analytics Institute and College of Engineering
Strategic Research Initiative.

References

[1] Moritz Hardt, Nimrod Megiddo, Christos Papadimitriou, and Mary Wootters. Strategic classifi-
cation. In Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer
Science, page 111–122, 2016.

[2] Sagi Levanon and Nir Rosenfeld. Generalized strategic classification and the case of aligned
incentives. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022,
Baltimore, Maryland, USA, pages 12593–12618, 2022.

[3] Rohan Taori and Tatsunori Hashimoto. Data feedback loops: Model-driven amplification of
dataset biases. In Proceedings of the 40th International Conference on Machine Learning,
volume 202 of Proceedings of Machine Learning Research, pages 33883–33920, 2023.

[4] George Alexandru Adam, Chun-Hao Kingsley Chang, Benjamin Haibe-Kains, and Anna Gold-
enberg. Error amplification when updating deployed machine learning models. In Proceedings
of the Machine Learning for Healthcare Conference, Durham, NC, USA, pages 5–6, 2022.

[5] Jinshuo Dong, Aaron Roth, Zachary Schutzman, Bo Waggoner, and Zhiwei Steven Wu. Strategic
classification from revealed preferences. In Proceedings of the 2018 ACM Conference on
Economics and Computation, page 55–70, 2018.

[6] Saba Ahmadi, Hedyeh Beyhaghi, Avrim Blum, and Keziah Naggita. The strategic perceptron.
In Proceedings of the 22nd ACM Conference on Economics and Computation, pages 6–25,
2021.

[7] Yiling Chen, Yang Liu, and Chara Podimata. Learning strategy-aware linear classifiers. Ad-
vances in Neural Information Processing Systems, 33:15265–15276, 2020.

[8] Sierra Wyllie, Ilia Shumailov, and Nicolas Papernot. Fairness feedback loops: training on
synthetic data amplifies bias. In The 2024 ACM Conference on Fairness, Accountability, and
Transparency, pages 2113–2147, 2024.

[9] Ozgur Guldogan, Yuchen Zeng, Jy-yong Sohn, Ramtin Pedarsani, and Kangwook Lee. Equal
improvability: A new fairness notion considering the long-term impact. In The Eleventh
International Conference on Learning Representations, 2022.

10


---Page Break---
[10] Tijana Zrnic, Eric Mazumdar, Shankar Sastry, and Michael Jordan. Who leads and who
follows in strategic classification? Advances in Neural Information Processing Systems, pages
15257–15269, 2021.

[11] Jon Kleinberg and Manish Raghavan. How do classifiers induce agents to invest effort strategi-
cally? page 1–23, 2020.

[12] Xueru Zhang, Mohammad Mahdi Khalili, Kun Jin, Parinaz Naghizadeh, and Mingyan Liu.
Fairness interventions as (Dis)Incentives for strategic manipulation. In Proceedings of the 39th
International Conference on Machine Learning, pages 26239–26264, 2022.

[13] Reilly Raab and Yang Liu. Unintended selection: Persistent qualification rate disparities and
interventions. Advances in Neural Information Processing Systems, pages 26053–26065, 2021.

[14] Xueru Zhang, Ruibo Tu, Yang Liu, Mingyan Liu, Hedvig Kjellstrom, Kun Zhang, and Cheng
Zhang. How do fair decisions fare in long-term qualification? In Advances in Neural Information
Processing Systems, pages 18457–18469, 2020.

[15] Lydia T Liu, Ashia Wilson, Nika Haghtalab, Adam Tauman Kalai, Christian Borgs, and
Jennifer Chayes. The disparate equilibria of algorithmic decision making when individuals
invest rationally. In Proceedings of the 2020 Conference on Fairness, Accountability, and
Transparency, pages 381–391, 2020.

[16] Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkata-
subramanian. Certifying and removing disparate impact. In proceedings of the 21th ACM
SIGKDD international conference on knowledge discovery and data mining, pages 259–268,
2015.

[17] Moritz Hardt, Eric Price, Eric Price, and Nati Srebro. Equality of opportunity in supervised
learning. In Advances in Neural Information Processing Systems, 2016.

[18] Meena Jagadeesan, Celestine Mendler-Dünner, and Moritz Hardt. Alternative microfoundations
for strategic classification. In Proceedings of the 38th International Conference on Machine
Learning, pages 4687–4697, 2021.

[19] Hans Hofmann. Statlog (German Credit Data). UCI Machine Learning Repository, 1994. DOI:
https://doi.org/10.24432/C5NC77.

[20] Quinlan Quinlan.
Credit Approval.
UCI Machine Learning Repository, 2017.
DOI:
https://doi.org/10.24432/C5FS30.

[21] Quinn Capers IV, Daniel Clinchot, Leon McDougle, and Anthony G Greenwald. Implicit racial
bias in medical school admissions. Academic Medicine, 92(3):365–369, 2017.

[22] AJ Alvero, Noah Arthurs, Anthony Lising Antonio, Benjamin W Domingue, Ben Gebre-Medhin,
Sonia Giebel, and Mitchell L Stevens. Ai and holistic review: informing human reading in
college admissions. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society,
pages 200–206, 2020.

[23] Elias Bareinboim and Judea Pearl. Controlling selection bias in causal inference. In Artificial
Intelligence and Statistics, pages 100–108, 2012.

[24] J Michelle Brock and Ralph De Haas. Discriminatory lending: Evidence from bankers in the
lab. American Economic Journal: Applied Economics, 15(2):31–68, 2023.

[25] Omer Ben-Porat and Moshe Tennenholtz. Best response regression. In Advances in Neural
Information Processing Systems, 2017.

[26] Mark Braverman and Sumegha Garg. The role of randomness and noise in strategic classification.
CoRR, abs/2005.08377, 2020.

[27] Zachary Izzo, Lexing Ying, and James Zou. How to learn when data reacts to your model:
Performative gradient descent. In Proceedings of the 38th International Conference on Machine
Learning, pages 4641–4650, 2021.

11


---Page Break---
[28] Wei Tang, Chien-Ju Ho, and Yang Liu. Linear models are robust optimal under strategic
behavior. In Proceedings of The 24th International Conference on Artificial Intelligence and
Statistics, volume 130 of Proceedings of Machine Learning Research, pages 2584–2592, 13–15
Apr 2021.

[29] Itay Eilat, Ben Finkelshtein, Chaim Baskin, and Nir Rosenfeld. Strategic classification with
graph neural networks, 2022.

[30] Lydia T. Liu, Nikhil Garg, and Christian Borgs. Strategic ranking. In Proceedings of The 25th
International Conference on Artificial Intelligence and Statistics, pages 2489–2518, 2022.

[31] Tosca Lechner and Ruth Urner. Learning losses for strategic classification. arXiv preprint
arXiv:2203.13421, 2022.

[32] Guy Horowitz and Nir Rosenfeld. Causal strategic classification: A tale of two shifts, 2023.

[33] Keegan Harris, Hoda Heidari, and Steven Z Wu. Stateful strategic regression. Advances in
Neural Information Processing Systems, pages 28728–28741, 2021.

[34] Yahav Bechavod, Chara Podimata, Steven Wu, and Juba Ziani. Information discrepancy in
strategic learning. In International Conference on Machine Learning, pages 1691–1715, 2022.

[35] Kun Jin, Xueru Zhang, Mohammad Mahdi Khalili, Parinaz Naghizadeh, and Mingyan Liu.
Incentive mechanisms for strategic classification and regression problems. In Proceedings of
the 23rd ACM Conference on Economics and Computation, page 760–790, 2022.

[36] Yatong Chen, Jialu Wang, and Yang Liu. Strategic recourse in linear classification. CoRR,
abs/2011.00355, 2020.

[37] Nika Haghtalab, Nicole Immorlica, Brendan Lucier, and Jack Z Wang. Maximizing welfare
with incentive-aware evaluation mechanisms. arXiv preprint arXiv:2011.01956, 2020.

[38] Tal Alon, Magdalen Dobson, Ariel Procaccia, Inbal Talgam-Cohen, and Jamie Tucker-Foltz.
Multiagent evaluation mechanisms. Proceedings of the AAAI Conference on Artificial Intelli-
gence, 34:1774–1781, 2020.

[39] Yahav Bechavod, Katrina Ligett, Steven Wu, and Juba Ziani. Gaming helps! learning from
strategic interactions in natural dynamics. In International Conference on Artificial Intelligence
and Statistics, pages 1234–1242, 2021.

[40] Tian Xie, Xuwei Tan, and Xueru Zhang. Algorithmic decision-making under agents with
persistent improvement. arXiv preprint arXiv:2405.01807, 2024.

[41] Tian Xie and Xueru Zhang. Non-linear welfare-aware strategic learning, 2024. URL https:
//arxiv.org/abs/2405.01810.

[42] John Miller, Smitha Milli, and Moritz Hardt. Strategic classification is causal modeling in
disguise. In Proceedings of the 37th International Conference on Machine Learning, 2020.

[43] Yonadav Shavit, Benjamin L. Edelman, and Brian Axelrod. Causal strategic linear regression.
In Proceedings of the 37th International Conference on Machine Learning, ICML’20, 2020.

[44] Keegan Harris, Dung Daniel T Ngo, Logan Stapleton, Hoda Heidari, and Steven Wu. Strategic
instrumental variable regression: Recovering causal relationships from strategic responses. In
International Conference on Machine Learning, pages 8502–8522, 2022.

[45] Tom Yan, Shantanu Gupta, and Zachary Lipton. Discovering optimal scoring mechanisms in
causal strategic prediction, 2023.

[46] Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. Performative
prediction. In Proceedings of the 37th International Conference on Machine Learning, pages
7599–7609, 2020.

[47] Moritz Hardt, Meena Jagadeesan, and Celestine Mendler-Dünner. Performative power. In
Advances in Neural Information Processing Systems, 2022.

12


---Page Break---
[48] Nir Rosenfeld, Anna Hilgard, Sai Srivatsa Ravindranath, and David C Parkes. From predictions
to decisions: Using lookahead regularization. In Advances in Neural Information Processing
Systems, pages 4115–4126, 2020.

[49] Melissa Hall, Laurens van der Maaten, Laura Gustafson, and Aaron Adcock. A systematic
study of bias amplification. arXiv preprint arXiv:2201.11706, 2022.

[50] Julius Adebayo, Melissa Hall, Bowen Yu, and Bobbie Chern. Quantifying and mitigating the
impact of label errors on model disparity metrics. In The Eleventh International Conference on
Learning Representations, 2022.

[51] Ali Shirali, Rediet Abebe, and Moritz Hardt. A theory of dynamic benchmarks. In The Eleventh
International Conference on Learning Representations, 2022.

[52] Klas Leino, Emily Black, Matt Fredrikson, Shayak Sen, and Anupam Datta. Feature-wise bias
amplification. arXiv preprint arXiv:1812.08999, 2018.

[53] Emily Dinan, Angela Fan, Adina Williams, Jack Urbanek, Douwe Kiela, and Jason Weston.
Queens are powerful too: Mitigating gender bias in dialogue generation. arXiv preprint
arXiv:1911.03842, 2019.

[54] Tianlu Wang, Jieyu Zhao, Mark Yatskar, Kai-Wei Chang, and Vicente Ordonez. Balanced
datasets are not enough: Estimating and mitigating gender bias in deep image representations.
pages 5310–5319, 2019.

[55] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like
shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint
arXiv:1707.09457, 2017.

[56] David Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner,
Vinay Chaudhary, Michael Young, Jean-Francois Crespo, and Dan Dennison. Hidden technical
debt in machine learning systems. Advances in neural information processing systems, 28,
2015.

[57] Danielle Ensign, Sorelle A Friedler, Scott Neville, Carlos Scheidegger, and Suresh Venkata-
subramanian. Runaway feedback loops in predictive policing. In Conference on fairness,
accountability and transparency, pages 160–171. PMLR, 2018.

[58] Masoud Mansoury, Himan Abdollahpouri, Mykola Pechenizkiy, Bamshad Mobasher, and
Robin Burke. Feedback loop and bias amplification in recommender systems. In Proceedings
of the 29th ACM international conference on information & knowledge management, pages
2145–2148, 2020.

[59] George Alexandru Adam, Chun-Hao Kingsley Chang, Benjamin Haibe-Kains, and Anna
Goldenberg. Hidden risks of machine learning applied to healthcare: Unintended feedback
loops between models and future data causing model degradation. In Proceedings of the 5th
Machine Learning for Healthcare Conference, volume 126 of Proceedings of Machine Learning
Research, pages 710–731. PMLR, 2020.

[60] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Ander-
son. The curse of recursion: Training on generated data makes models forget. arXiv preprint
arXiv:2305.17493, 2023.

[61] Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, and Gauthier
Gidel. On the stability of iterative retraining of generative models on their own data. arXiv
preprint arXiv:2310.00429, 2023.

[62] Sven Schmit and Carlos Riquelme. Human interaction with recommendation systems. In
International Conference on Artificial Intelligence and Statistics, pages 862–870, 2018.

[63] Ayan Sinha, David F Gleich, and Karthik Ramani. Deconvolving feedback loops in recom-
mender systems. Advances in neural information processing systems, 29, 2016.

13


---Page Break---
[64] Ray Jiang, Silvia Chiappa, Tor Lattimore, András György, and Pushmeet Kohli. Degenerate
feedback loops in recommender systems. In Proceedings of the 2019 AAAI/ACM Conference
on AI, Ethics, and Society, page 383–390, 2019.

[65] Vivek Gupta, Pegah Nokhiz, Chitradeep Dutta Roy, and Suresh Venkatasubramanian. Equalizing
recourse across groups, 2019.

[66] Lydia T. Liu, Sarah Dean, Esther Rolf, Max Simchowitz, and Moritz Hardt. Delayed impact of
fair machine learning. In Proceedings of the Twenty-Eighth International Joint Conference on
Artificial Intelligence, IJCAI-19, pages 6196–6200, 2019.

[67] Sagi Levanon and Nir Rosenfeld. Strategic classification made practical. In International
Conference on Machine Learning, pages 6243–6253, 2021.

[68] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL https://archive.
ics.uci.edu/ml/datasets/credit+approval.

[69] Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets
for fair machine learning. Advances in neural information processing systems, 34:6478–6490,
2021.

14


---Page Break---
A
The retraining process

Algorithm 1 retraining process

Input: Joint distribution Do
XY for any So,t, Hypothesis class F, the number of the initial training
samples and agents coming per round N, the number of decision-maker-labeled samples per
round K.
Output: Model deployments over time f0, f1, f2, . . .

1: S0 = So,0 ∼Do
XY = {xi
0}N
i=1
2: At t = 0, deploy f0 ∼F(S0)
3: for t ∈{1, . . . ∞} do
4:
N agents gain knowledge of ft−1 and best respond to it, resulting in {xi
t}N
i=1
5:
Sm,t = {xi
t−1, ft−1(xi
t−1)}N
i=1 consists of the model-labeled samples from round t −1.
6:
So,t ∼Do
XY consists of the new K decision-maker-labeled samples.
7:
St = St−1 ∪Sm,t ∪So,t
8:
Deploy ft ∼F(St) on the incoming N agents who best respond to ft−1 with the resulting
joint distribution P t
XY .
9: end for

B
Motivating examples of the systematic bias

Def. 2.1 highlights the systematic nature of the decision-maker’s bias. This bias is quite ubiquitous
when labeling is not a trivial task. It almost always the case when the decision-maker needs to make
human-related decisions. We provide the following motivating examples of systematic bias with
supporting literature in social science:

1. College admissions: consider experts in the admission committee of a college that obtains a set of
student data and wants to label all students as "qualified" or "unqualified". The labeling task is
much more complex and subjective than the ones in computer vision/natural language processing
which have some "correct" answers. Therefore, the experts in the committee are prone to bring
their "biases" towards a specific population sharing the same sensitive attribute into the labeling
process including:

(a) Implicit bias: the experts may have an implicit bias they are unaware of to favor/discriminate
against students from certain groups. For instance, a famous study [21] reveals admission
committee members at the medical school of the Ohio State University unconsciously have
a "better impression" towards white students; Alvero et al. [22] finds out that even when
members in an admission committee do not access the sensitive attributes of students, they
unconsciously infer them and discriminate against students from the minority group.

(b) Selection bias: the experts may have insufficient knowledge of the under-represented pop-
ulation due to the selection bias [23] because only a small portion of them were admitted
before. Thus, experts may expect a lower qualification rate from this population, resulting in
more conservative labeling practices. The historical stereotypes created by selection bias are
difficult to erase.

2. Loan applications: consider experts in a big bank that obtains data samples from some potential
applicants and wants to label them as "qualified" or "unqualified". Similarly, the experts are likely
to have systematic bias including:

(a) Implicit bias: similarly, Brock and De Haas [24] conduct a lab-in-the-field experiment with
over 300 Turkish loan officers to show that they bias against female applicants even if they
have identical profiles as male applicants.

(b) Selection bias: when fewer female applicants are approved historically, the experts have less
knowledge on females (i.e., whether they will actually default or repay), thereby tending to
stay conservative.

15


---Page Break---
C
Related Work

C.1
Strategic Classification

Strategic classification without label changes. Our work is mainly based on an extensive line of
literature on strategic classification [1, 2, 5, 6, 7, 12, 14, 18, 25, 26, 27, 28, 29, 30, 31, 32]. These
works assume the agents are able to best respond to the policies of the decision-maker to maximize
their utilities. Most works modeled the strategic interactions between agents and the decision-maker
as a repeated Stackelberg game where the decision-maker leads by publishing a classifier and the
agents immediately best respond to it. The earliest line of works focused on the performance of
regular linear classifiers when strategic behaviors never incur label changes [1, 5, 7, 25], while the
later literature added noise to the agents’ best responses [18], randomized the classifiers [26] and
limited the knowledge of the decision-maker [28]. Levanon and Rosenfeld [2] proposed a generalized
framework for strategic classification and a strategic hinge loss to better train strategic classifiers, but
the strategic behaviors are still not assumed to cause label changes.

Strategic classification with label changes. Several other lines of literature enable strategic behaviors
to cause label changes. The first line of literature mainly focuses on incentivizing improvement
actions where agents have budgets to invest in different actions and only some of them cause the
label change (improvement) [11, 13, 33, 34, 35, 36, 37, 38, 39, 40, 41]. The other line of literature
focuses on causal strategic learning [32, 42, 43, 44, 45]. These works argue that every strategic
learning problem has a non-trivial causal structure which can be explained by a structural causal
model, where intervening on causal nodes causes improvement and intervening on non-causal nodes
means manipulation.

Performative prediction. Several works consider performative prediction as a more general setting
where the feature distribution of agents is a function of the classifier parameters. Perdomo et al. [46]
first formulated the prediction problem and provided iterative algorithms to find the stable points
of the model parameters. Izzo et al. [27] modified the gradient-based methods and proposed the
Perfgrad algorithm. Hardt et al. [47] elaborated the model by proposing performative power.

Retraining under strategic settings

Most works on learning algorithms under strategic settings consider developing robust algorithms that
the decision-maker only trains the classifier once [1, 2, 18, 28], while Performative prediction[46]
focuses on developing online learning algorithms for strategic agents. There are only a few works
[32, 48] which permit retraining the Strategic classification models. However, all these algorithms
assume that the decision-maker has access to a new training dataset containing both agents’ features
and labels at each round. There is no work considering the dynamics under the retraining process with
model-annotated samples. Also, few works [14, 15] discussed the long-term fairness issues under
sequential strategic settings. Liu et al. [15] modeled strategic behaviors in a completely different way
where each individual decides whether or not (a binary choice) to acquire the desired qualification
based on a cost drawn from a fixed distribution. Moreover, they assumed the decision maker has
perfect knowledge of the agent distribution, i.e., can draw infinitely many examples from the agent
population. Instead, our work focuses on a practical setting where the decision maker must learn
from samples and acquiring human-annotated samples is quite expensive, motivating the use of
model-annotated samples as well. Zhang et al. [14] also simplified the modeling of the qualification
changes of agents by using fixed transition matrices.

C.2
Bias amplification during retraining

There has been an extensive line of study on the computer vision field about how machine learning
models amplify the dataset bias, while most works only focus on the one-shot setting where the
machine learning model itself amplifies the bias between different groups in one training/testing
round [49, 50]; Another work theoretically studies dynamic benchmarking [51] to propose a retraining
scheme to increase model accuracy with adversarial human-annotation. In recent years, another
line of research focuses on the amplification of dataset bias under model-annotated data where ML
models label new samples on their own and add them back to retrain themselves[4, 52, 53, 54, 55, 56,
57, 58, 59]. These works study the bias amplification in different practical fields including resource
allocation [57], computer vision [54], natural language processing [55], generative models [60, 61]
and clinical trials [59]. The most related work is [3] which studied the influence of retraining in the

16


---Page Break---
non-strategic setting. Also, there is a work [4] touching on the data feedback loop under performative
setting, but it focused on empirical experiments under medical settings where the feature distribution
shifts are mainly caused by treatment and the true labels in historical data are highly accessible.
Besides, the data feedback loop is also related to recommendation systems. where extensive works
have studied how the system can shape users’ preferences and disengage the minority population
[58, 62, 63, 64]. However, previous literature did not touch on the retraining process.

C.3
Machine learning fairness in strategic classification

Several works have considered how different fairness metrics [1, 9, 16, 65] are influenced in strategic
classification [12, 14, 15, 66]. The most related works [12, 15] studied how strategic behaviors and
the decision-maker’s awareness can shape long-term fairness. They deviated from our paper since
they never considered retraining and the strategic behaviors never incurred label changes.

D
Additional Discussions

D.1
The source of Human-annotated samples

As stated in footnote 1, for the source of human-annotated samples, we consider natural settings
where the human-annotated samples are drawn from fixed PX and the process is independent of the
decision-making process, i.e., the agents who best respond to ft−1 are classified by model ft and
the decision-maker never confuses them with human annotations. Instead, human annotation is a
separate process for the decision-maker to obtain additional information about the whole population
(e.g., by first acquiring data from public datasets or third parties, and then labeling them to estimate
the population distribution PXY ). Here we provide an additional **example**: In a university
admission scenario, the admission office uses the model ft to assign decisions at t and obtain model-
annotated samples for t + 1. But for human-annotated samples, the admission committee hopes
to acquire a more objective knowledge of "what kinds of students are successful" from the whole
student population. Thus, in the hope of avoiding additional sampling/selection bias, the committee
seeks human annotations from third-party education researchers/experts. The experts can provide
qualifications of the public/historical student samples drawn from the whole student population
including students who are not applicants for this specific university or even do not attend a university.

In this section, we additionally consider the situation where all human-annotated samples at t are
drawn from the post-best-response distribution P t
X. This will change (4) to the following:

q′t = tN+(t−1)K

(t+1)N+tK · q′t−1 +
N
(t+1)N+tK · a′
t−1 +
K
(t+1)N+tK · q∗
t−1
(3)

where q′ and a′ denote the new qualification rate and acceptance rate, and q∗
t−1 stands for the
qualification rate of the human annotations on features drawn from P t−1
X
. Note that the only
difference lies in the third term of the RHS which changes from q0 to q∗
t−1. Our first observation
is that q∗
t−1 is never smaller than q0 because the best response will not harm agents’ qualifications.
With this observation, we can derive Prop. D.1.

Proposition D.1. a′
t ≥at holds for any t ≥1. If Prop. 3.4 further holds, we also have a′
t →1.

Prop. D.1 can be proved easily by applying the observation stated above. However, note that unlike
at, a′
t is not necessarily monotonically increasing.

D.2
Additional discussions on fairness

Other fairness metrics. It is difficult to derive concise results considering other fairness metrics
including equal opportunity and equal improvability because the data distributions play a role in
determining these metrics. However, Theorem 3.5 states the true qualification rate of the agent
population is likely to decrease, suggesting the retraining process may do harm to improvability.
Meanwhile, when the acceptance rate at increases for the disadvantageous group, the acceptance
rate of the qualified individuals will be likely to be better, but it is not guaranteed because the feature
distribution of the qualified individuals also changes because we assume the strategic behaviors are
causal which may incur label changes.

17


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
X1

0.0

0.2

0.4

0.6

0.8

1.0

X2

qualified
unqualified

1
0
1
2
X1

1

0

1

2

X2

qualified
unqualified

Figure 7: Visualization of distribution: Uniform data (left) and Gaussian data (right)

D.3
Additional Discussion on the training dataset St

St reuses the accumulated data instead of only fine-tuning the model using the available data at the
current round. This is more reasonable because: (i) The distribution shifts caused by agent best
responses are not known by the decision maker, only using current data may result in forgetting
the previous samples that may still be useful. Since agents’ best responses will change Y and will
not break PY |X, previous samples can provide useful information, especially on the feature domain
that current samples do not cover; (ii) the sample size at a single round may be too small even for
fine-tuning (e.g., for a college admission example, if we only have tens of people applying during
some application round).

Moreover, only using current data does not qualitatively change the theoretical results. Since the
theoretical results demonstrate monotonic trends of at, qt, only using the most recent data will not
alter the trends. We perform an additional set of simulations where the decision maker only uses
samples from the most recent round (Sm,t−1, So,t−1) on Gaussian Data and Uniform-linear Data
with the experimental setups same as the ones in the paper (Section 5 for Gaussian Data and App. E
for Uniform-linear Data), and report the average of at, qt. Dynamics of at, qt are still similar.

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(a) Uniform-linear: group i (left) and j (right)

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(b) Gaussian: group i (left) and j (right)

Figure 8: Dynamics of at, qt, ∆t when St only contains the most recent samples

D.4
Additional Discussions on refined retraining process

We provide the following proposition to illustrate how the refined retraining process leverages a
probabilistic sampler to stabilize the dynamics.

Proposition D.2. If the decision-maker uses a probabilistic sampler ht−1(x) = Dt−1
Y |X(1|x) to
produce model-annotated samples at t, then Dt
Y |X = Do
Y |X.

The proof details are in App. G.7. Prop. D.2 illustrates the underlying conditional distribution Dt
Y |X
is expected to be the same as Do
Y |X, meaning that the classifier ft always learns the distribution of
human-annotated data, thereby only preserving the systematic bias. However, there is no way to deal
with the systematic bias in refined retraining process.

D.5
Additional Discussions on the linearity of the model

Our model is built upon previous works on strategic classification (e.g., [2, 12, 32]) where the
decision policy was assumed to be transparent and interpretable since human agents expect to face a
linear classifier which they can understand, especially under high-stake situations such as job hiring,
college admission and loan application. Meanwhile, as pointed out by Zhang et al. [12], Raab and
Liu [13], Levanon and Rosenfeld [67], in practical circumstances, the decision-maker can first fit
a non-linear model to learn the embeddings of agents’ preliminary features and then produce the
embedded features. The decision-maker then uses a linear classifier on the new set of features and

18


---Page Break---
agents also best respond with respect to new features. Under this setting, nonlinearity is involved
but our results still hold. Practical examples include (i) FICO credit score [12]: FICO credit score is
based on a complex set of features, and the decision-maker (e.g., a bank) simply uses a threshold
of the score to assign decisions; (ii) Spam classification [67]: The decision-maker classifies spam
using a linear classifier, but the features are produced as an embedding by a neural network based on
number of words in the post, number of phone numbers in the post and number of followers of the
user.

E
Main Experiments for other datasets

We provide results on a Uniform dataset and a real dataset [20] with settings similar to Sec. 5.

Uniform data. All settings are similar to the Gaussian dataset except that PX and PY |X(1|x) change
as shown in Table 2.

Table 2: Gaussian Dataset Setting

PXk(xk)
PY |X(1|x)
n, r, T, q0

U(0, 1)
0.5 · (x1 + x2)
100, 0.05, 15, 0.5

We first verify the results in Sec. 3 by illustrating the dynam-
ics of at, qt, ∆t for both groups (Fig. 9). Since our analysis
neglects the algorithmic bias and the evolution results are
in expectation, we perform n = 100 independent runs of
experiments for every parameter configuration and show the averaged outcomes. The results are
consistent with Thm. 3.3, 3.5 and 3.6: (i) acceptance rate at (red curves) increases monotonically; (ii)
qualification rate qt decreases monotonically starting from t = 1 (since strategic agents only best
respond from t = 1); (iii) classifier bias ∆t evolves differently for different groups and it may reach
the minimum after a few rounds of retraining.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

Figure 9: at, qt, ∆t for group i (left) and j (right)

Table 3: Description of credit approval
dataset

Settings
Group i
Group j

PX1|Y (x1|1)
Beta(1.37, 3.23)
Beta(1.73, 3.84)
PX1|Y (x1|0)
Beta(1.50, 4.94)
Beta(1.59, 4.67)
PX2|Y (x2|1)
Beta(0.83, 2.83)
Beta(0.66, 2.50)
PX2|Y (x2|0)
Beta(0.84, 5.56)
Beta(0.69, 3.86)
n, r, T, q0
50, 0.05, 15, 0.473
50, 0.05, 15, 10

Next, we present the results of a set of complemen-
tary experiments on real data [20] where we directly
fit PX|Y with Beta distributions. The fitting results
slightly violate Assumption 3.2. Also Do
X is not equal
to PX. More importantly, logistic models cannot fit
these distributions well and produce non-negligible al-
gorithmic bias, thereby violating Assumption 3.1. The
following experiments demonstrate how the dynamics
change when situations are not ideal.

Credit approval dataset [20]. We consider credit card applications and adopt the data in UCI
Machine Learning Repository processed by Dua and Graff [68]. The dataset includes features of
agents from two social groups i, j and their labels indicate whether the credit application is successful.
We first preprocess the dataset by normalizing and only keeping a subset of features (two continuous
X1, X2) and labels, then we fit conditional distributions PXk|Y for each group using Beta distributions
(Fig. 10) and calculate prior-best-response qualification rates qi
0, qj
0 from the dataset. The details are
summarized in Table 3. All other parameter settings are the same as the ones of synthetic datasets in
Sec. 5.

We first illustrate the dynamics of at, qt, ∆t for both groups under different r. The results are shown
in Fig. 11 and are approximately aligned with Thm. 3.3, 3.5 and 3.6: (i) acceptance rate at (red
curves) has increasing trends; (ii) qualification rate qt (blue curves) decreases starting from t = 1
(since strategic agents only best respond from t = 1); (iii) classifier bias ∆t (black curves) evolve
differently for different groups.

19


---Page Break---
Next, we illustrate situation (iii) of Thm. 4.1 on this dataset, where the evolutions of unfairness and
∆t are shown in Fig. 12. Though the dynamics are still approximately aligned with the theoretical
results, the changes are not smooth. However, this is not surprising because several assumptions are
violated, and the overall trends still stay the same.

0.0
0.2
0.4
0.6
0.8
1.0
X1

2.5

5.0

7.5

10.0

Group i

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X2

2.5

5.0

7.5

10.0

Group i

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X1

2.5

5.0

7.5

10.0

Group j

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X2

2.5

5.0

7.5

10.0

Group j

unqualified
qualified

Figure 10: Visualization of distribution for Credit Approval dataset.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Group i in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Group j in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0 (right)

Figure 11: Dynamics of at, qt, ∆t for Credit Approval dataset

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i

at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i

at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i

at for j
Unfairness

(a) Dynamics of unfairness in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0
(right)

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25
t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

0.30

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.15

0.20

0.25

0.30

0.35

0.40

0.45

t for i

t for j

(b) Dynamics of ∆t in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0 (right)

Figure 12: Dynamics of unfairness and ∆t for Credit Approval dataset

F
Additional Results on Synthetic/Semi-synthetic Datasets

In this section, we provide comprehensive experimental results conducted on two synthetic datasets
and one semi-synthetic dataset mentioned in Sec. 5 of the main paper. Specifically, App. F.1 gives

20


---Page Break---
the details of experimental setups; App. F.2 demonstrates additional results to verify Theorem 3.3
to Theorem 3.6 under different r (i.e., ratios of human-annotated examples available at each round).
The section also gives results on how at, qt, ∆t change under a long time horizon or when there is
no systematic bias; App. F.3 further demonstrates the results under refined retraining process; App.
F.4 provides fairness dynamics under various values of r on different datasets; App. F.5 illustrates
how the results in the main paper still hold when strategic agents have noisy best responses; App.
F.6 compares the situations when agents are non-strategic with the ones when they are strategic,
demonstrating how agents’ strategic behaviors produce more extreme dynamics of at, qt, ∆t.

F.1
Additional Experimental Setups

Generally, we run all experiments on a MacBook Pro with Apple M1 Pro chips, memory of 16GB
and Python 3.9.13. All experiments are randomized with seed 42 to run n rounds. Error bars are
provided in App. H. All experiments train ft with a logistic classifier using SGD as its optimizer.
Specifically, we use SGDClassifier with logloss to fit models.

Synthetic datasets. The basic description of synthetic datasets 1 and 2 is shown in Sec. 5 and App.
E. We further provide the visualizations of their distributions in Fig. 7.

German Credit dataset. There are 2 versions of the German Credit dataset according to UCI
Machine Learning Database [19], and we are using the one where all features are numeric. Firstly,
we produce the sensitive features by ignoring the marital status while only focusing on sex. Secondly,
we use MinMaxScaler to normalize all features. The logistic model itself can satisfy Assumption
3.2 with minimal operations: if feature i has coefficients smaller than 0, then just negate it and the
coefficients will be larger than 0 and satisfy the assumption.

F.2
Additional Results to Verify Thm. 3.3 to Thm. 3.6

Dynamics under different r. Although experiments in Sec. 5 and App. E already demonstrate the
validity of Thm. 3.3, Prop. 3.4 and Thm. 3.6, the ratio r of human-annotated examples is subject to
change in reality. Therefore, we first provide results for r ∈{0, 0.05, 0.1, 0.3} in all 3 datasets. r
only has small values since human-annotated examples are likely to be expensive to acquire. Fig. 17
shows all results under different r values. Specifically, Fig. 17a and 17b show results for synthetic
dataset 1, Fig. 17c and 17d show results for synthetic dataset 2, while Fig. 17e and 17f show results
for German Credit data. On every row, r = 0.3, 0.1, 0.05, 0 from the left to the right. All figures
demonstrate the robustness of the theoretical results, where at always increases and qt decreases
starting from t = 1. ∆t also has different dynamics as specified in Thm. 3.6.

Dynamics under a long time horizon to verify Prop. 3.4. Prop. 3.4 demonstrates that when r is
small enough, at will increase towards 1. Therefore, we provide results in all 3 datasets when r = 0
to see whether at increases to be close to 1. As Fig. 15 shows, at is close to 1 after tens of rounds,
validating Prop. 3.4.

Dynamics under a different B. Moreover, individuals may incur different costs to alter different

features, so we also provide the dynamics of at, qt, ∆t when the cost matrix B =

3
0
0
6


in two

synthetic datasets. Fig. 16 shows the differences in costs of changing different features do not affect
the theoretical results.

Dynamics when all samples are human-annotated. Though this is unlikely to happen under the
Strategic Classification setting as justified in the main paper, we provide an illustration when all
training examples are human-annotated (i.e., r = 1) when humans systematically overestimate the
qualification in both synthetic datasets. Theoretically, the difference between at and qt should be
relatively consistent, which means ∆t is only due to the systematic bias. Fig. 15d verifies this.

F.3
Additional Results on refined retraining process

In this section, we provide more experimental results demonstrating how refined retraining process
stabilizes the dynamics of at, qt, ∆t but still preserves the systematic bias. Specifically, we produce
plots similar to Fig. 17 in Fig. 18, but the only difference is that we use probabilistic samplers for

21


---Page Break---
model-annotated examples. From Fig. 18, it is obvious the deviations of at from qt have the same
directions and approximately the same magnitudes as the systematic bias.

F.4
Additional Results on Fairness

In this section, we provide additional results on the dynamics of unfairness and classifier bias under
different r (the same settings in App. F.2). We aim to illustrate situation (iii) in Thm. 4.1 where the
refined retraining process is applied on the advantaged group i while the original process is applied
on the disadvantaged group j. From Fig. 19a, 19c and 19e, we can see unfairness reaches a minimum
in the middle of the retraining process, suggesting the earlier stopping of retraining brings benefits.
From Fig. 19b, 19d and 19f, we can see ∆t for the disadvantaged group j reaches a minimum in
the middle of the retraining process, but generally not at the same time when unfairness reaches a
minimum.

F.5
Additional Results on Noisy Best Responses

Following the discussion in Sec. 5, we provide dynamics of at, qt, ∆t of both groups under different
r similar to App. F.2 but when the agents have noisy knowledge. The only difference is that the
agents’ best responses are noisy in that they only know a noisy version of classification outcomes:
bft(x) = ft(x) + ϵ, where ϵ is a Gaussian noise with mean 0 and standard deviation 0.1. Fig.20 shows
that Thm. 3.3 to Thm. 3.6 are still valid.

F.6
Comparisons between Strategic and Non-strategic Situations

In this section, we show the absence of strategic behaviors may result in much more consistent
dynamics of at, qt, ∆t as illustrated in Fig. 21.

F.7
Additional Results on a Dataset with Non-Linearity.

ACSIncome-CA dataset [69] is a larger dataset consisting of over 150K records for agents and their
annual income. The decision-maker wants to predict whether a person has an annual income > 50000.
We assume the decision-maker first trains a 2-d embedding using a neural network and 53 original
features then regards the embedding as the new feature. We divide the agents into 2 groups based on
their ages. Similar to the credit approval dataset, we then fit Beta distributions on the 2 groups and
then verify the monotonic likelihood assumption (Fig. 13). We then plot the dynamics of at, qt, ∆t
for both groups when the systematic bias is either positive or negative. The results show that similar
trends still hold for this large dataset (Fig. 14).

0.0
0.2
0.4
0.6
0.8
1.0
X1

5

10

15

Group i

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X2

5

10

15

Group i

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X1

5

10

15

Group j

unqualified
qualified

0.0
0.2
0.4
0.6
0.8
1.0
X2

5

10

15

Group j

unqualified
qualified

Figure 13: Feature distribution of Income Dataset.

22


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

Figure 14: Dynamics of at, qt, ∆t of the income dataset: the left plot is for group i and the right
plot is for group j. We run 10 trials for each experiment and agent cost matrices are the same as the
experiments in the main paper. The results are similar to the ones in the main paper.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16
18
20

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Uniform data when r = 0 and T is large: Group i
(left), Group j (right)

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
8
16
24
32
40
48
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(b) Gaussian data when r = 0 and T is large: Group i
(left), Group j (right)

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) German Credit data when r = 0 and T is large:
Group i (left), Group j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Dynamics when all data is human-annotated (i.e,
r = 1): Uniform data (left), Gaussian data (right).

Figure 15: Dynamics of at, qt, ∆t on all datasets when r = 0 and T is large or when all examples
are annotated by humans.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Uniform data with a different B : Group i (left),
Group j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Gaussian data with a different B : Group i (left),
Group j (right)

Figure 16: Dynamics of at, qt, ∆t on synthetic datasets. Except B, all other settings are as same as
in Sec. 5.

23


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Group i in Uniform data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(b) Group j in Uniform data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Group i in Gaussian data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Group j in Gaussian data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Group i in German Credit data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Group j in German Credit data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

Figure 17: Dynamics of at, qt, ∆t on all datasets.

24


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) refined retraining process for Group i in Uniform data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) refined retraining process for Group j in Uniform data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) refined retraining process for Group i in Gaussian data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) refined retraining process for Group j in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) refined retraining process for Group i in German Credit data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) refined retraining process for Group j in German Credit data: r = 0.1
(left), 0.05 (middle) and 0 (right).

Figure 18: Illustrations of refined retraining process on all 3 datasets.

25


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j
Unfairness

(a) Dynamics of unfairness in Uniform data : r = 0.1 (left), 0.05 (middle) and 0
(right)

0
2
4
6
8
10 12 14 16
t

0.05

0.10

0.15

0.20

0.25

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

t for i

t for j

(b) Dynamics of ∆t in Uniform data : r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

(c) Dynamics of unfairness in Gaussian data : r = 0.1 (left), 0.05 (middle) and 0
(right)

0
2
4
6
8
10 12 14 16
t

0.05

0.10

0.15

0.20

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25
t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.05

0.10

0.15

0.20

0.25
t for i

t for j

(d) Dynamics of ∆t in Gaussian data : r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

(e) Dynamics of unfairness in German Credit data : r = 0.1 (left), 0.05 (middle)
and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

0.30

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

t for i

t for j

(f) Dynamics of ∆t in German Credit data : r = 0.1 (left), 0.05 (middle) and 0
(right)

Figure 19: Dynamics of unfairness and ∆t of all datasets.

26


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Noisy retraining process for Group i in Uniform data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Noisy retraining process for Group j in Uniform data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Noisy retraining process for Group i in Gaussian data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Noisy retraining process for Group j in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Noisy retraining process for Group i in German Credit data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Noisy retraining process for Group j in German Credit data: r = 0.1
(left), 0.05 (middle) and 0 (right)

Figure 20: Illustrations of noisy retraining process on all 3 datasets.

27


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(a) Nonstrategic retraining process for Group i in Uniform data

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(b) Nonstrategic retraining process for Group j in Uniform data

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Nonstrategic retraining process for Group i in Gaussian data

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Nonstrategic retraining process for Group j in Gaussian data

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Nonstrategic retraining process for Group i in German Credit data

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Nonstrategic retraining process for Group j in German Credit data

Figure 21: Illustrations of nonstrategic retraining process on all 3 datasets: r = 0.1 (left), 0.05
(middle) and 0 (right)

28


---Page Break---
G
Derivations and Proofs

G.1
Dynamics of the Expected Qualification Rate of St

Let us begin by defining qt := ESt[Q(St)] which is the expected qualification rate of the training
dataset at t. While it is difficult to derive the dynamics of at and qt explicitly, we can first work out
the dynamics of qt using the law of total probability (details in App. G.1), i.e.,

qt = tN+(t−1)K

(t+1)N+tK · qt−1 +
N
(t+1)N+tK · at−1 +
K
(t+1)N+tK · q0
(4)

To derive qt := ESt[Q(St)], we first refer to (2) to get |St| = (t + 1)N + tK. Then, by the law of
total probability, the expected qualification rate of St equals to the weighted sum of the expected
qualification rate of St−1, So,t−1 and the expectation of ft−1(x) over Sm,t−1 as follows:

tN+(t−1)K
(t+1)N+tK ESt−1[Q(St−1)] +
N
(t+1)N+tK + ESt−1[A(ft−1, P t−1)] +
K
(t+1)N+tK ESo,t−1[Q(So,t−1)]

The second expectation is exactly the definition of at−1. Moreover, note that Q(So,t−1) = Q(So,0) =
Q(S0) = q0 for any t > 0, the third expectation is exactly q0, so the above equation is exactly (4).

G.2
Derivation of factors influencing at, qt

As stated in Sec.2, we can get the factors influencing the evolution of at, qt by finding all sources
affecting P t
XY and the expectation of ft(x) over P t
X.

We first work out the sources influencing ft and the expectation of ft(x) over P t(X):

• qt: since ft is trained with St, qt is a key factor influencing the classifier.
• ht may not model Dt
Y |X accurately enough. However, we ignore this by Assumption 3.1, where
realizability is a common assumption in theoretical proof, while the experiments do not ignore
this source.
• δt
BR: we now know factors influencing the expectation of ft(x) over Dt
X, then the only left factor
is the ones accounting for the difference between Dt
X and P t
X. Note that only the best responses
of agents can change the marginal distribution PX. We then denote it as δt
BR.

Then, with PXY known, P t
XY is only influenced by ft−1 (i.e., the agents’ best responses to the
classifier at t −1). ft−1 is also dependent on the above factors. So we get all factors.

G.3
Proof of Theorem 3.3

proving the following lemma:
Lemma G.1. Assume t ≥2 and the following conditions hold: (i) qt > qt−1 ≥qt−2; (ii) ∀x ∈X,
Dt
Y |X(1|x) ≥Dt−1
Y |X(1|x); (iii) ∀x, f t−1(x) ≥f t−2(x). Let f t−1 = ESt−1∼Dt−1
XY [ft−1], f t =
ESt∼Dt
XY [ft], we have the following results:

1. ∀x, f t(x) ≥f t−1(x).
2. There exists a non-zero measure subset of x values that satisfies the strict inequality.

Proof. We first prove (i). Note that we assume ht models Dt
Y |X well in Assumption 3.1, so f t−1
(resp. f t) outputs 1 if Dt−1
Y |X(1|x) (resp. Dt
Y |X(1|x)) is larger than some threshold θ. Then, according

to the above condition (ii), ∀θ, if Dt−1
Y |X(1|x) > θ, Dt
Y |X(1|x) ≥Dt−1
Y |X(1|x) > θ. This demonstrates

that f t−1(x) = 1 implies f t(x) = 1 and (i) is proved.

Next, according to (2), if qt > qt−1, this means Ey∼Dt
Y [y] > Ey∼Dt
Y [y]. Since Dt
Y = Dt
X ·Dt
Y |X, ei-
ther there exists at least one non-zero measure subset of x values satisfying Dt
Y |X(1|x) > Dt−1
Y |X(1|x)
or Dt
X is more "skewed" to the larger values of x (because of monotonic likelihood assumption
3.2). For the second possibility, note that the only possible cause for the feature distribution in

29


---Page Break---
the training dataset Dt
X to gain such a skewness is agents’ strategic behaviors. However, since
f t−1(x) ≥f t−2(x) always holds, f t−1 sets a lower admission standard where some x values that
are able to best respond to f t−2 and improve will not best respond to f t−1, thereby impossible to
result in a feature distribution shift to larger x values while keeping the conditional distribution
unchanged. Thus, only the first possibility holds.

Then we prove Theorem 3.3 using mathematical induction to prove a stronger version:

Lemma G.2. When t > 1, qt > qt−1, Dt
Y |X(1|x) ≥Dt−1
Y |X(1|x), f t(x) ≥f t−1(x), and finally
at > at−1.

Proof. t starts from 2, but we need to prove the following claim: q1 is "almost equal" to q0, so are
D1
Y |X and D0
Y |X.

Firstly, according to the law of total probability, we can derive q1 as follows:

q1 =
N
2N + K · q0 +
N
2N + K · a0 +
K
2N + K · q0
(5)

The first and the third element are already multiples of q0. Also, we know Do
X = PX. Then,
since Dt
Y |X(1|x) falls in H and agents at t = 0 have no chance to best respond, we have a0 =
EX∼PX[Do
Y |X(1|x)] = q0. Thus, a0 is also equal to q0, and the claim is proved. Still, as Assumption
3.1 assumes realizability of ht, f 1 is the same as f 0.

Next, we can prove the lemma by induction:

1. Base case: Similar to Eq.5, we are able to derive q2 as follows:

q2 = 2N + K

3N + 2K · q1 +
N
3N + 2K · a1 +
K
3N + 2K · q0
(6)

Based on the claim above, we can just regard q0 as q1. Then we may only focus on the second
term. Since q1 and q0 are "almost equal" and both distributions should satisfy the monotonic
likelihood assumption 3.2, we can conclude f 0, f 1 are "almost identical". Then the best responses
of agents to f 0 will also make them be classified as 1 by f 1, and this will directly ensure the
second term to be A(f1, P 1) > A(f 1, P) = A(f 0, P). The "larger than" relationship is because
strategic best responses at the first round will only enable more agents to be admitted. Thus,
the first and the third term stay the same as q1 while the second is larger, so we can claim
q2 > q1. Moreover, the difference between D2
XY and D1
XY are purely produced by the best
responses at t = 1, which will never decrease the conditional probability of y = 1. Thus,
D2
Y |X(1|x) ≥D1
Y |X(1|x). Together with f 1 = f 0, all three conditions in Lemma G.1 are

satisfied. we thereby claim that for every x admitted by f 1, f 2(x) ≥f 1(x) and there exists some
x satisfying the strict inequality. Note that P 1 is expected to be the same as P 2 since f1 = f0.
Thus, A(f2, P 2) > A(f1, P 2) = A(f1, P 1), which is a2 > a1. The base case is proved.

2. Induction step: To simplify the notion, we can write:

qt
=
tN + (t −1)K
(t + 1)N + tK · qt−1 +
N
(t + 1)N + tK · at−1 +
K
(t + 1)N + tK · q0

=
At · qt−1 + Bt · at−1 + Ct · q0

Note that qt−1 can also be decomposed into three terms:

qt−1
=
(t −1)N + (t −2)K

tN + (t −1)K
· qt−2 +
N
tN + (t −1)K · at−2 +
K
tN + (t −1)K · q0

=
At−1 · qt−1 + Bt−1 · at−2 + Ct−1 · q0

30


---Page Break---
Since the expectation in the second term is just at−1, and we already know at−1 > at−2, we
know Bt · at−1 + Ct · q0 > Bt · at−2 + Ct · q0. Note that
Bt
Bt−1 =
Ct
Ct−1 , we let the ratio be
m < 1. Then since Bt−1 · at−2 + Ct−1 · q0 > (Bt−1 + Ct−1) · qt−1 due to qt−1 > qt−2,
we can derive Bt · at−1 + Ct · q0 > m · (Bt−1 + Ct−1) · qt−1 = (Bt + Ct) · qt−1. Then
qt > (At + Bt + Ct) · qt−1 = qt−1. The first claim is proved. As at−1 > at−2 and Do stays the
same, any agent will not have a less probability of being qualified in St compared to in St−1,
demonstrating Dt
Y |X(1|x) ≥Dt−1
Y |X(1|x) still holds. And similarly, we can apply Lemma G.1 to

get f t(x) ≥f t−1(x) and at > at−1.

Now we already prove Lemma G.2, which already includes Theorem 3.3. We also want to note
here, the proof of Theorem 3.3 does not rely on the initial q0 which means it holds regardless of
the systematic bias.

G.4
Proof of Proposition 3.4

We prove the proposition by considering two extreme cases:

(i) When K

N →0, this means we have no decision-maker annotated sample coming in each round and
all new samples come from the deployed model. We prove limt→∞at = 1 by contradiction: firstly,
by Monotone Convergence Theorem, the limit must exist because at > at−1 and at < 1. Let us
assume the limit is a < 1. Then, since K = 0, when t →∞, qt will also approach a, this means the
strategic shift δt
BR approaches 0. However, this shift only approaches 0 when all agents are accepted
by ft−1 because otherwise there will be a proportion of agents benefiting from best responding to
ft−1 and result in a larger qt+1 Thus, the classifier at t + 1 will admit more people and the stability is
broken. This means the only possibility is limt→∞at−1 = 1 and produces a conflict.

(ii) When K

N →∞, the problem shrinks to retrain the classifier to fit Do
XY , this will make at = a0.

Thus, there exists some threshold λ, when K

N < λ, limt→∞at = 1. In practice, the λ could be very
small.

G.5
Proof of Theorem 3.5

Firstly, since P t
XY differs from PXY only because agents’ best respond to ft, we can write qt = q0+rt
where rt is the difference of qualification rate caused by agents’ best responses to ft. Qualitatively, rt
is completely determined by two sources: (i) the proportion of agents who move their features when
they best respond; (ii) the increase in the probability of being qualified for each agent. Specifically,
each agent that moves its features increases its probability of being qualified from its initial point to a
point at the decision boundary of ft. For an agent with initial feature x and improved feature x∗, its
improvement can be expressed as U(x) = PY |X(1|x∗) −PY |X(1|x).

Next, denote the Euclidean distance between x and the decision boundary of ft as dx,t. Noticing that
the agents will best respond to the decision classifier if and only if the Euclidean distance between her
feature vector and the decision boundary is less than or equal to some constant C no matter where the
boundary is (Lemma 2 in [2]) and what the cost matrix B is, we can express the total improvement at
t as follows:

I(t) =
Z

dx,t≤C
PX(x) · U(x) dx
(7)

According to the proof of Theorem 3.3, all agents with feature vector x who are admitted by f0 will
be admitted by ft (t > 0), making all agents who possibly improve must belong to J . When FX
is convex in J or it is convex in each dimension separately, we know PX as its derivative will be
non-decreasing in each of the d dimensions within J . Similarly, when PY |X(1|x) is convex in each of
its dimensions or the whole J , U is also non-decreasing in each of the d dimensions within J . Then
note that ft always lies below ft−1, therefore, for each agent who improves at t having feature vector
xi,t, we can find an agent at t −1 with corresponding xi,t−1 such that both PX(xi,t−1) ≥PX(xi,t)
and U(xi,t−1) ≥U(xi,t). This will ensure I(t) ≤I(t −1). Thus, qt decreases starting from 1.

31


---Page Break---
G.6
Proof of Theorem 3.6

• µ(Do, P) ≥0: according to Def. 2.1, a0 = EPX[Do
Y |X(1|x)] > EPX[PY |X(1|x)] = Q(P) =
q0. Now that a0 −q0 ≥0. Based on Thm. 3.3 and Thm. 3.5, at is increasing, while q0 is
decreasing, so ∆t is always increasing.
• µ(Do, P) < 0: similarly we can derive a0 −q0 < 0, so ∆0 = |a0 −q0| = q0 −a0. So while
a0 −q0 is still increasing, ∆t will first decrease. Moreover, according to Prop. 3.4, if K

N is small
enough, ∆t will eventually exceed 0 and become larger again. Thus, ∆t either decreases or first
decreases and then increases.

G.7
Proof of Proposition D.2

Firstly, S0 = So,0 and S1 = S0 ∪So,1 ∪Sm,0. Obviously, the first two sets are drawn from Do
XY .
Consider Sm,0, since it is now produced by labeling features from PX = Do
X with Do
Y |X, Sm,0 is
also drawn from Do
XY . Thus, D1
Y |X = Do
Y |X.

Then we prove the cases when t > 1 using mathematical induction as follows:

1. Base case: We know S2 = S1 ∪So,2 ∪Sm,1. The first two sets on rhs are drawn from Do
XY .
The labeing in the third set is produced by h1(x) = D1
Y |X = Do
Y |X. Thus, the base case is proved.

2. Induction step: St = St−1 ∪So,t ∪Sm,t−1. Similarly, we only need to consider the third set. But
Note that ht(x) = Dt−1
Y |X = Do
Y |X, the induction step is easily completed.

G.8
Proof of Proposition G.3

At round t, we know Ait
θi
t > Ajt
θj
t . To reach demographic parity, the acceptance rates need to be the

same, so at least one of the following situations must happen: (i) eθi
t > θi
t and eθj
t > θj
t; (ii) eθi
t < θi
t
and eθj
t < θj
t; (iii) eθi
t ≥θi
t and eθj
t ≤θj
t. Next we prove that (i) and (ii) cannot be true by contradiction.

Suppose (i) holds, then we can find θ
j
t < θj
t such that ℓ(θ
j
t, Sj
t ) ∈
 
ℓ(θj
t, Sj
t ), ℓ(eθj
t, Sj
t )

and

Ajt

θ
j
t ∈
 
Ajt
θj
t , Ait
θi
t

. We can indeed find this θ
j
t because ℓ, A are continuous w.r.t. θ. Now noticing that

Ait
eθi
t = Ajt
eθj
t > Ajt
θj
t but Ajt

θ
j
t < Ajt
θj
t , we will know we can find θ
i
t ∈
 
θi
t, eθi
t

to satisfy demographic

parity together with θ
j
t. Since PY |X(1|x) satisfies monotonic likelihood and θi
t is the optimal point,

ℓ(θ
i
t, Si
t) < ℓ(eθi
t, Si
t) must hold. Thus, (θ
i
t, θ
j
t) satisfy demographic parity and have a lower loss than
(eθi
t, eθj
t). Moreover, the pair satisfies (iii), which produces a conflict.

Similarly, we can prove (ii) cannot hold by contradiction, thereby proving (iii) must hold.

G.9
Proof of Theorem 4.1

Situation (i) is directly derived from Thm. 3.3, where we can just regard the trajectory of the
acceptance rate of group j as moving the one of group i vertically downward; Situation (ii) is derived
from Prop. D.2, where the systematic bias causes unfairness and it keeps stable; Situation (iii) can
be derived from Thm. 3.6, where the acceptance rate of group i stays stable, while the one for j
monotonically increases.

G.10
Proof of Theorem 4.2

The first fact is that both θi
t, θj
t stay stable as 0.5 −µi and 0.5 −µj when we use accuracy as the
metric. Therefore, to understand which group is disadvantaged/advantaged at t + 1, we just need to
track the agents who best respond to eθi
t, eθj
t. First, we need the following proposition which is proved
in App. G.8:

Proposition G.3. If P ti
XY , P tj
XY are continuous, then eθi
t ≥θi
t and eθj
t ≤θj
t.

32


---Page Break---
Note that we want to sacrifice the least accuracy to reach demographic parity. Since the agent best
response will never make them worse off, we will know ai
t corresponding to the acceptance rate of
group i under the optimal threshold θi
t is always larger than or equal to ai
0. Thus, it suffices to show
that under the condition of Thm. 4.2, aj
t < ai
0.

Suppose aj
t < ai
0, then we can use Chebyshev’s Inequality to bound the probability that an unadmitted
agent z in group j reaches θj
t after its best response. Denote this event as E and the threshold agent z

feels is Iz, where E(Iz) = eθj
t. Then we can write P(E) < P(Iz −E(Iz) ≥θj
t −eθj
t) < (θj
t −eθj
t )2

σ2
t
.

Thus, when σt <
θj
t −eθj
t
√

ai
0−aj
t
, we know P(E) < ai
0 −aj
t, and from the simple fact aj
t+1 < aj
t + P(E),

we know j will stay disadvantaged at t + 1.

However, when we do not add the fairness intervention, just consider the situation where all unad-
mitted agents in group j at t = 0 improve, while no one in group i can improve. This will flip the
disadvantaged/advantaged group.

33


---Page Break---
H
All plots with error bars

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Uniform data with a different B : Group i (left),
Group j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Gaussian data with a different B : Group i (left),
Group j (right)

Figure 22: Error bar version of Fig. 16.

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

at for i

at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

Figure 23: Error bar version of Fig. 4

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) at, qt, ∆t for group i (left) and j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) unfairness (left) and classifier bias ∆t (right)

Figure 24: Error bar version of Fig. 5

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) at, qt, ∆t for group i (left) and j (right)

0
2
4
6
8
10 12 14 16
t

1.0

0.5

0.0

0.5

1.0

1.5

at for i

at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

1.0

0.5

0.0

0.5

1.0

t for i

t for j

(b) unfairness (left) and classifier bias ∆t (right)

Figure 25: Error bar version of Fig. 9

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Gaussian: at, qt, ∆t for group i (left) and j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) German: at, qt, ∆t for group i (left) and j (right)

Figure 26: Error bar version of Fig. 6

34


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0
0.2
0.4
0.6
0.8
1.0

at
qt

t

(a) Uniform-linear: group i (left) and j (right)

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0
0.2
0.4
0.6
0.8
1.0

at
qt

t

(b) Gaussian: group i (left) and j (right)

Figure 27: Error bar version of Fig. 8.

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16
18
20

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Uniform data when r = 0 and T is large: Group i
(left), Group j (right)

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
8
16
24
32
40
48
56
64
72

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Gaussian data when r = 0 and T is large: Group i
(left), Group j (right)

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
8
16
24
32
40
48

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) German Credit data when r = 0 and T is large:
Group i (left), Group j (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0
a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Dynamics when all data is human-annotated (i.e,
r = 1) on synthetic datasets.

Figure 28: Error bar version of Fig. 15

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Group i in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Group j in Credit Approval data: r = 0.1 (left), 0.05 (middle) and 0 (right)

Figure 29: Error bar version of Fig. 11

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

Figure 30: Error bar version of Fig. 14.

35


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Group i in Uniform data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0
0.2
0.4
0.6
0.8
1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

(b) Group j in Uniform data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Group i in Gaussian data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0
at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Group j in Gaussian data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Group i in German Credit data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Group j in German Credit data : r = 0.3, 0.1, 0.05, 0 from the leftmost to the rightmost

Figure 31: Error bar version of Fig. 17.

36


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0
a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0
a

t

q

t

Δ

t

(a) refined retraining process for Group i in Uniform data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) refined retraining process for Group j in Uniform data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) refined retraining process for Group i in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) refined retraining process for Group j in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) refined retraining process for Group i in German Credit data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) refined retraining process for Group j in German Credit data:
r = 0.1 (left), 0.05 (middle) and 0 (right).

Figure 32: Error bar version of Fig. 18

37


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.25

0.00

0.25

0.50

0.75

1.00

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

1.2

at for i
at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

1.2

at for i
at for j
Unfairness

(a) Dynamics of unfairness in Uniform data : r = 0.1 (left), 0.05 (middle) and
0 (right)

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

t for i

t for j

(b) Dynamics of ∆t in Uniform data : r = 0.1 (left), 0.05 (middle) and 0
(right)

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j

Unfairness

(c) Dynamics of unfairness in Gaussian data : r = 0.1 (left), 0.05 (middle)
and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.1

0.0

0.1

0.2

0.3
t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.1

0.0

0.1

0.2

0.3

0.4

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.1

0.0

0.1

0.2

0.3

0.4
t for i

t for j

(d) Dynamics of ∆t in Gaussian data : r = 0.1 (left), 0.05 (middle) and 0
(right)

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

at for i
at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

at for i
at for j
Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i
at for j
Unfairness

(e) Dynamics of unfairness in German Credit data : r = 0.1 (left), 0.05
(middle) and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.0

0.1

0.2

0.3

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.0

0.1

0.2

0.3

t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.0

0.1

0.2

0.3

t for i

t for j

(f) Dynamics of ∆t in German Credit data : r = 0.1 (left), 0.05 (middle) and
0 (right)

Figure 33: Error bar version of Fig. 19

38


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Noisy retraining process for Group i in Uniform data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Noisy retraining process for Group j in Uniform data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Noisy retraining process for Group i in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Noisy retraining process for Group j in Gaussian data: r = 0.1
(left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Noisy retraining process for Group i in German Credit data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Noisy retraining process for Group j in German Credit data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

Figure 34: Error bar version of Fig. 20

39


---Page Break---
0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(a) Nonstrategic retraining process for Group i in Uniform data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at
qt

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(b) Nonstrategic retraining process for Group j in Uniform data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(c) Nonstrategic retraining process for Group i in Gaussian data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(d) Nonstrategic retraining process for Group j in Gaussian data:
r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(e) Nonstrategic retraining process for Group i in German Credit
data: r = 0.1 (left), 0.05 (middle) and 0 (right)

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

0
2
4
6
8
10
12
14
16

t

0.0

0.2

0.4

0.6

0.8

1.0

a

t

q

t

Δ

t

(f) Nonstrategic retraining process for Group j in German Credit
data: r = 0.1 (left), 0.05 (middle) and 0 (right)

Figure 35: Error bar version of Fig. 21

40


---Page Break---
0
2
4
6
8
10 12 14 16
t

0.25

0.00

0.25

0.50

0.75

1.00

1.25

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.25

0.00

0.25

0.50

0.75

1.00

1.25

at for i
at for j

Unfairness

0
2
4
6
8
10 12 14 16
t

0.0

0.2

0.4

0.6

0.8

1.0

at for i

at for j
Unfairness

(a) Dynamics of unfairness in Credit Approval data: r = 0.1 (left),
0.05 (middle) and 0 (right)

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6
t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.2

0.0

0.2

0.4

0.6
t for i

t for j

0
2
4
6
8
10 12 14 16
t

0.1

0.2

0.3

0.4

0.5
t for i

t for j

(b) Dynamics of ∆t in Credit Approval data: r = 0.1 (left), 0.05
(middle) and 0 (right)

Figure 36: Error bar version of Fig. 12

Although all our theoretical results are expressed in terms of expectation, we provide error bars for all
plots in the main paper, App. E and F if randomness is applicable. All the above figures demonstrate
expectations as well as error bars (± 1 standard deviation). Overall, the experiments have reasonable
standard errors. However, experiments in the Credit Approval dataset [20] (Fig. 36) incur larger
standard errors, which is not surprising because the dataset violates several assumptions. Finally, note
that we conduct 50-100 randomized trials for every experiment and we should expect much lower
standard errors if the numbers of trials become large.

41


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract summarizes the paper’s contribution appropriately.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have a conclusion section to discuss the limitations.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

42


---Page Break---
Justification: See main paper and Appendix where all assumptions and proofs are clearly
stated.
Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: See supplementary materials.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

43


---Page Break---
Answer: [Yes]
Justification: The code and dataset are in the supplemental material.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: All details are in experiment section and appendix and supplementary materials.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: We have standard errors in the plots for all experiments in App. H.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.

44


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: See App. F.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We reviewed the NeurIPS Code of Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: See Conclusion Section.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to

45


---Page Break---
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: No such risks.
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We cite all papers, datasets appropriately.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

46


---Page Break---
Answer: [NA]
Justification: The paper does not release new assets
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

47


---Page Break---
