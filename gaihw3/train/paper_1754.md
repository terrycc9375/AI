Heterogeneity-Guided Client Sampling: Towards Fast
and Efficient Non-IID Federated Learning

Huancheng Chen
University of Texas at Austin
huanchengch@utexas.edu

Haris Vikalo
University of Texas at Austin
hvikalo@ece.utexas.edu

Abstract

Statistical heterogeneity of data present at client devices in a federated learning
(FL) system renders the training of a global model in such systems difficult. Particu-
larly challenging are the settings where due to communication resource constraints
only a small fraction of clients can participate in any given round of FL. Recent
approaches to training a global model in FL systems with non-IID data have fo-
cused on developing client selection methods that aim to sample clients with more
informative updates of the model. However, existing client selection techniques
either introduce significant computation overhead or perform well only in the
scenarios where clients have data with similar heterogeneity profiles. In this paper,
we propose HiCS-FL (Federated Learning via Hierarchical Clustered Sampling), a
novel client selection method in which the server estimates statistical heterogene-
ity of a client’s data using the client’s update of the network’s output layer and
relies on this information to cluster and sample the clients. We analyze the ability
of the proposed techniques to compare heterogeneity of different datasets, and
characterize convergence of the training process that deploys the introduced client
selection method. Extensive experimental results demonstrate that in non-IID set-
tings HiCS-FL achieves faster convergence than state-of-the-art FL client selection
schemes. Notably, HiCS-FL drastically reduces computation cost compared to
existing selection schemes and is adaptable to different heterogeneity scenarios.

1
Introduction

The federated learning (FL) framework enables privacy-preserving collaborative training of machine
learning (ML) models across a number of devices (clients) by avoiding the need to collect private
data stored at those devices. The participating clients typically experience both the system as well as
statistical heterogeneity [18]. The former describes settings where client devices have varying degree
of computational resources, communication bandwidth and fault tolerance, while the latter refers to
the fact that the data owned by the clients may be drawn from different distributions. In this paper, we
focus on FL under statistical heterogeneity and leave studies of system heterogeneity to future work.

An early FL method, FedAvg [21], performs well in the settings where the devices train on inde-
pendent and identically distributed (IID) data. However, compared to the IID scenario, training on
non-IID data is detrimental to the convergence speed, variance and accuracy of the learned model.
This has motivated numerous studies aiming to reduce the variance and improve convergence of FL
on non-IID data [6, 9, 14, 17, 19, 30].

On another note, constraints on communication resources and therefore on the number of clients
that may participate in training additionally complicate implementation of FL schemes. It would
be particularly unrealistic to require regular contributions to training from all the clients in a large-
scale cross-device FL system. Instead, only a fraction of clients participate in any given training
round; unfortunately, this further aggravates detrimental effects of statistical heterogeneity. Selecting
informative clients in non-IID FL settings is an open problem that has received considerable attention

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
from the research community [8, 11, 12]. Since privacy concerns typically prohibit clients from
sharing their local data label distributions, existing studies focus on estimating informativeness of
a client’s update by analyzing the update itself. This motivated a family of methods that rely on
the norms of local updates to assign probabilities of sampling the clients [7, 23]. Aiming to enable
efficient use of the available communication and computation resources, another set of methods
groups clients with similar data distributions into clusters based on the similarity between clients’
model updates [2, 11]. Across the board, the existing methods still struggle to deliver desired
performance in an efficient manner and cannot distinguish clients with balanced data from the clients
with imbalanced data.

In this paper, we consider training a neural network model for classification tasks via federated
learning and propose a novel adaptive clustering-based sampling method for identifying and selecting
informative clients. The method, referred to as Federated Learning via Hierarchical Clustered
Sampling (HiCS-FL), relies on the updates of the (fully connected) output layer in the network to
determine how diverse is the clients’ data and, based on that, decide which clients to sample. In
particular, HiCS-FL enables heterogeneity-guided client selection by utilizing general properties
of the gradients of the output layer to distinguish between clients with balanced from those with
imbalanced data. Unlike the Clustered Sampling strategies [11] where the clusters of clients are
sampled uniformly, HiCS-FL allocates different probabilities (importance) to the clusters according
to their average estimated data heterogeneity. Numerous experiments conducted on vision datasets
FMNIST, CIFAR10, Mini-ImageNet and a NLP dataset THUC news demonstrate that HiCS-FL
achieves significantly faster training convergence and lower variance than the competing methods.
Finally, we conduct convergence analysis of HiCS-FL and discuss implications of the results.

In summary, the contributions of the paper include: (1) Analytical characterization of the correlation
between local updates of the output layer and the FL clients’ data label distribution, along with
an efficient method for estimating data heterogeneity; (2) a novel clustering-based algorithm for
heterogeneity-guided client selection; (3) extensive simulation results demonstrating HiCS-FL pro-
vides significant improvement in terms of convergence speed and variance over competing approaches;
and (4) theoretical analysis of the proposed schemes.

2
Background and Related Work

Assume the cross-device federated learning setting with N clients, where client k owns private local
dataset Bk with |Bk| samples. The plain vanilla FL considers the objective

min
θ
F(θ) ≜

N
X

k=1
pkFk(θ),
(1)

where θ denotes parameters of the global model, Fk(θ) is the loss (empirical risk) of model θ on
Bk, and pk denotes the weight assigned to client k, PN
k=1 pk = 1. In FedAvg, the weights are set to
pk = |Bk| / PN
i=1 |Bi|. In training round t, the server collects clients’ model updates θt
k formed by
training on local data and aggregates them to update global model as θt+1 = PN
k=1 pkθt
k.

When an FL system operates under resource constraints, typically only K ≪N clients are selected
to participate in any given round of training; denote the set of clients selected in round t by St. In
departure from FedAvg, FedProx [19] proposes an alternative strategy for sampling clients based
on a multinomial distribution where the probability of selecting a client is proportional to the size
of its local dataset; the global model is then formed as the average of the collected local models
θt+1 = 1

K
P
k∈St θt
k. This sampling strategy is unbiased since the the updated global model is on
expectation equal to the one obtained by the framework with full client participation as Eq.1.

AFL [12] is the first study to utilize local validation loss as a value function for computing client
sampling probabilities; Power-of-Choice [8] takes a step further to propose a greedy approach to
sampling clients with the largest local loss. Both of these methods require all clients to compute
the local validation loss, which is often unrealistic. To address this problem, FedCor [28] models
the local loss by a Gaussian Process (GP), estimates the GP parameters from experiments, and uses
the GP model to predict clients’ local losses without requiring them to perform validation. In [7],
Optimal Client Sampling scheme aiming to minimize the variance of local updates by assigning
sampling probabilities proportional to the Euclidean norm of the updates is proposed. The study in
[23] models the progression of model’s weights by an Ornstein-Uhlenbeck process and proposes a
strategy, optimal under that assumption, for selecting clients with significant weight updates.

2


---Page Break---
The clustering-based sampling method proposed in [11] uses cosine similarity [24] to group together
clients with similar local updates, and proceeds to sample one client per cluster in attempt to avoid
redundant gradient information. DivFL [2] follows the same principle of identifying representative
clients but does so by constructing a submodular set and greedily selecting diverse clients. Both of
these techniques are computationally expensive due to the high dimension of the gradients that they
need to process.

In general, the overviewed methods either: (1) select diverse clients to reduce redundant information;
or (2) select clients with a perceived significant contributions to the global model (high loss, large
update or low class-imbalance). Efficient and effective client selection in FL remains an open
challenge, motivating the heterogeneity-guided adaptive client selection method presented next.

3
HiCS-FL: Federated Learning via Hierarchical Clustered Sampling

Figure 1: The last two network layers.

Existing client sampling methods including Clustered Sam-
pling [11] and DivFL [2] aim to select clients such that the
resulting model update is an unbiased estimate of the true
update (i.e., the update in the case of full client participa-
tion) while minimizing the variance

1
N

N
X

k=1
∇Fk(θt) −1

K

X

k∈St
∇Fk(θt)



2

2
.
(2)

Clustered Sampling, for instance, groups N clients into K
clusters based on representative gradients [24], and randomly selects one client from each cluster
to contribute to the global model update. Such an approach unfortunately fails to differentiate
between model updates formed on data with balanced and those formed on data with imbalanced
label distributions – indeed, in either case the updates are treated as being equally important. However,
a number of studies in centralized learning has shown that class-imbalanced datasets have significant
detrimental effect on the performance of learning classification tasks [3, 4, 26]. This intuition carries
over to the FL settings where one expects the updates from clients training on relatively more balanced
local data to have a more beneficial impact on the performance of the system. The Federated Learning
via Hierarchical Clustered Sampling (HiCS-FL) framework described in this section adapts to the
clients’ data heterogeneity in the following way: if the levels of heterogeneity (as quantified by the
entropy of data label distribution) vary from one cluster to another, HiCS-FL is more likely to sample
clusters containing clients with more balanced data; if the clients grouped in different clusters have
similar heterogeneity levels, HiCS-FL is more likely to select diverse clients (i.e., sample uniformly
across clusters, thus reducing to the conventional clustered sampling strategy).

3.1
Class-imbalance Causes Objective Drift

A number of studies explored detrimental effects of non-IID training data on the performance of a
global model learned via FedAvg. An example is SCAFFOLD [14] which demonstrates objective drift
in non-IID FL manifested through large differences between local models θ∗
k trained on substantially
different data distributions. The drift is due to FedAvg updating the global model in the direction of
the weighted average of local optimal models, which is not necessarily leading towards the optimal
global model θ∗. The optimal model θ∗, in principle obtained by solving optimization in Eq. 1,
achieves minimal empirical error on the data with uniform label distribution and is intuitively closer
to the local optimal models trained on balanced data. Recent work [36] empirically verified this
conjecture through extensive experiments. Let ∇F(θt) denote the gradient of F(θt) given the global
model θt at round t; the difference between ∇F(θt) and the local gradient ∇Fk(θt) computed
on client k’s data is typically assumed to be bounded [7, 11, 31]. To proceed, we formalize the
assumption about the relationship between gradients and data label distributions.

Assumption 3.1 (Bounded Dissimilarity.) Gradient ∇Fk(θt) of the k-th local model at global
round t is such that
∇Fk(θt) −∇F(θt)
2 ≤κ −ρeβ(H(D(k))−H(D0)) = σ2
k,
(3)

where D(k) is the data label distribution of client k, D0 denotes uniform distribution, H(·) is
Shannon’s entropy of a stochastic vector, and β > 0, κ > ρ > 0.

3


---Page Break---
The assumption commonly encountered in literature is recovered by setting the right-hand side of
(3) to σ2
m = maxk σ2
k. Intuitively, if the data label distribution of client k is highly imbalanced
(i.e., H(D(k)) is small), the local gradient ∇Fk(θt) may significantly differ from the global gradient
∇F(θt) (as reflected by the bound above). Analytically, connecting the gradients to the local data
label distributions allows one to characterize the effects of client selection on the variance and the rate
of convergence. The results of extensive experiments that empirically verify the above assumption
are reported in Appendix A.2.

3.2
Estimating Client’s Data Heterogeneity

If the server were given access to clients’ data label distributions, selecting clients would be relatively
straightforward [32]. However, privacy concerns typically discourage clients from sharing such
information. Previous studies have explored the use of multi-arm bandits for inferring clients’ data
heterogeneity from local model parameters, or have utilized a validation dataset at the server to
accomplish the same [27, 34, 36]. In this section, we demonstrate how to efficiently and accurately
estimate data heterogeneity using local updates of the output layer of a neural network in a classifica-
tion task. Figure 1 illustrates the last two layers in a typical neural network. The prediction q ∈RC is
computed by forming a weighted average of signals z ∈RL utilizing the weight matrix W ∈RC×L
and bias b ∈RC.

3.2.1
Local updates of the output layer

An empirical investigation of the gradients of the output layer’s weights while training with FedAvg
using mini-batch stochastic gradient descent (SGD) as an optimizer is reported in [5, 29]. There,
the focus is on detecting the presence of specific labels in a batch rather than on exploring the
effects of class imbalance on the local update. To pursue the latter, we focus on the correlation
between local updates of the output layer’s bias and the client’s data label distribution; we start
by analyzing the training via FedAvg that employs SGD and then extend the results to other FL
algorithms that utilize optimizers beyond SGD. We assume that the model is trained by minimizing
the cross-entropy (CE) loss over one-hot labels – a widely used multi-class classification framework.
The gradient is computed by averaging contributions of the samples in mini-batches, i.e., ∇bLce =
1
Bl
Pl
j=1
PB
n=1 ∇bL(j,n)
ce
(x(j,n), y(j,n)), where B denotes the batch size, l is the number of mini-
batches, x(j,n) is the n-th point in the j-th mini-batch and y(j,n) ∈[C] is its label. The contribution
of x(j,n) to the i-th component of the gradient of the output layer’s bias b can be found as (details
provided in Appendix A.3)

∇biL(j,n)
ce
(x(j,n), y(j,n)) = I{i = y(j,n)}
−P

c̸=i exp(q(j,n)
c
)
PC
c=1 exp(q(j,n)
c
)
+ I{i ̸= y(j,n)}
exp(q(j,n)
i
)
PC
c=1 exp(q(j,n)
c
)
,

(4)
where I{·} is an indicator, q(j,n) = W · z(j,n) + b is the output logit for signals z(j,n) ∈RL

corresponding to training point (x(j,n), y(j,n)) (see Fig. 1), and where C denotes the number of
classes. We make the following observations: (1) the sign of y(j,n)-th component of ∇bL(j,n)
ce
is
opposite of the sign of other components; and (2) the y(j,n)-th component of ∇bL(j,n)
ce
is equal in
magnitude to all the other components combined. Note that the above two observations are standard
for neural networks using CE loss for supervised multi-class classification tasks.

In each global round t of FedAvg, the selected client k starts from the global model θt and proceeds to
compute local update in R local epochs employing an SGD optimizer with learning rate η. According
to Eq. 4, the i-th component of local update ∆b(k) is computed as

∆b(k)
i
= −η

Bl

l
X

j=1

B
X

n=1

R
X

r=1
∇biL(j,n,r)
ce
,
(5)

where ∇biL(j,n,r)
ce
denotes the gradient of bias at local epoch r. Note that the local update of client k,
∆b(k), is dependent on the label distribution of client k’s data, D(k) = [D(k)
1 , . . . , D(k)
C ]T and the
label-specific components of q(j,n) which change during training. We proceed by relating expected
local updates to the label distributions; for convenience, we first introduce the following definition.

4


---Page Break---
Definition 3.2 Let B−i be the subset of local data B that excludes points with label i. Let s−i(x) ∈
[0, 1]C be the softmax output of a trained neural network for a training point (x, y) ∈B−i. The
i-th component of s−i(x), s−i
i (x), indicates the level of confidence in (erroneously) classifying x as
having label i. For convenience, we define Ei = E(x,y)∼B−i 
s−i
i (x)

, ∀i ∈[C].

In an untrained/initialized neural network where classifier makes random predictions, Ei = 1/C; as
training proceeds, Ei decreases. By taking expectation and simplifying, we obtain (details provided
in Appendix A.4)

E
h
∆b(k)
i
i
= ηR

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(6)

where D(k)
i
denotes the true fraction of samples with label i in client k’s data, PC
i=1 D(k)
i
= 1.

3.2.2
Estimating local data heterogeneity

We quantify the heterogeneity of clients’ data by an entropy-like measure defined below. Let
D(k) denote the label distribution of client k’s data; its entropy is defined as H(D(k)) ≜
−PC
i=1 D(k)
i
ln D(k)
i
≤ln C. Recall that more balanced data results in higher entropy, and that
H(D(k)) takes the maximal value when D(k) is uniform. The server does not know D(k) and therefore
cannot compute H(D(k)) directly. We define

ˆH(D(k)) ≜H(softmax(∆b(k), T)),
(7)

here T is a scaling hyper-parameter (so-called temperature). Note that even though we can compute
ˆH(D(k)) to characterize heterogeneity, D(k)
i
and Ei remain unknown to the server (details in A.5).

Theorem 3.3 Consider an FL system in which clients collaboratively train a model for a classifica-
tion task over C classes. Let D(u) and D(k) denote data label distributions of an arbitrary pair of
clients u and k, respectively. Moreover, let U denote the uniform distribution, and let η and R be the
learning rate and the number of local epochs, respectively. Then

E
h
ˆH(D(u)) −ˆH(D(k))
i
≥1

2

 
ηR
CT

C
X

c=1
Ec

!2 D(k) −U

2

2 −ηR

T

D(u) −U

∞−Cδ,
(8)

where C = ηR(ηR+C2T ln C)

C2T 2
and δ = maxi


PC
c=1 Ec

C
−Ei
. The proof is provided in Appendix A.6.

As an illustration, consider the scenario where client u has a balanced dataset while the dataset of
client k is imbalanced; then ∥D(k) −U∥2
2 is relatively large compared to ∥D(u) −U∥∞. The bound
in (8) also depends on δ, which is reflective of how misleading on average can a class be; small
δ suggests that no class is universally misleading. As shown in Appendix A.4, during training δ
gradually decreases to 0 as PC
i=1 Ei decreases to 0.

3.2.3
Generalizing beyond FedAvg and SGD

The proposed method for estimating clients’ data heterogeneity relies on the properties of the gradient
for the cross-entropy loss objective discussed in Section 3.2.1. However, for FL algorithms other
than FedAvg, such as FedProx [19], FedDyn [1] and Moon [16], which add regularization to combat
overfitting, the aforementioned properties may not hold. Moreover, optimization algorithms using
second-order momentum such as Adam [15] deploy update rules different from SGD, making the
local updates no longer proportional to the gradients. Nevertheless, HiCS-FL remains capable of
distinguishing between clients with imbalanced and balanced data, which will be demonstrated in our
experiments. Further theoretical discussion of various FL algorithms with optimizers beyond SGD
are in appendix A.8 and A.9.

3.3
Heterogeneity-guided Clustering

Clustered Sampling [11] uses cosine similarity [24] between gradients to quantify proximity between
clients’ data distributions and subsequently group them into clusters. However, cosine similarity

5


---Page Break---
Algorithm 1 HiCS-FL

Input:

Datasets distributed across N clients, the number
of clients to sample K, total global rounds T .
1: Initialize updates of bias ∆b(k) ←0 ∀k ∈[N],
global model θt ←θ1, S0 = [N].
2: for t = 1, . . . , T do
3:
if t ≤⌈N/K⌉then
4:
St ←randomly sample min(K, |S0|)
clients from S0, update S0 ←S0 −St;
5:
else
6:
estimate ˆHt(D(k)) and cluster N clients into
M groups based on Eq. 9;
7:
St ←∅;
8:
while |St| < K do

9:
sample group Gt
m according to πt;
10:
sample client k in Gt
m based on ˜pm;
11:
St ←St ∪k;
12:
end while
13:
end if
14:
for k ∈St do
15:
θt
k ←LocalUpdate(θt), ∆b(k) ∈θt
k −θt

16:
end for
17:
θt+1 ←
1
K
P

k∈St θt
k;
18:
∆b(k) ←∆b(k), ∀k ∈St;
19: end for
Output:

The global model θT +1

cannot help distinguish between clients with balanced and those with imbalanced datasets. Motivated
by this observation, we introduce a new distance measure that incorporates estimates of data hetero-
geneity ˆH(D(k)). In particular, the proposed measure of distance between clients u and k that we
use to form clusters is defined as

Distance(u, k) = arc cos
 ∆b(u) · ∆b(k)

|∆b(u)| · |∆b(k)|


+ λ
 ˆH(D(u)) −ˆH(D(k))
 ,
(9)

where the first term is akin to the cosine similarity used by CS with the major difference that we
compute it using only the updates of the bias in the output layer, which is much more efficient
than using the weights of the entire network; λ is a pre-defined hyper-parameter (set to 10 in all
our experiments). For large λ, the second term dominates when there are clients with different
levels of statistical heterogeneity; this allows emergence of clusters that group together clients with
balanced datasets. The second term is small when clients have data with similar levels of statistical
heterogeneity; in that case, the distance measure reduces to the conventional cosine similarity.

3.4
Hierarchical Clustered Sampling

To select K out of N clients in an FL system, we first organize the clients into M ≥K groups via the
proposed Hierarchical Clustered Sampling (HiCS) technique. In particular, during the first ⌈N/K⌉
training rounds the server randomly (without replacement) selects clients and collects from them
local updates of ∆b(k); the server then estimates ˆHt(D(k)) for each selected client k and clusters the
clients using the distance measure defined in Eq. 9. Let Gt
1, . . . , Gt
M denote the resulting M clusters
at global round t, and let ¯Ht
m =
1
|Gm|
P

k∈Gm ˆHt(D(k)) characterize the average heterogeneity
of clients in cluster m, m ∈[M]. Having computed ¯Ht
m, HiCS selects a cluster according to the
probability vector πt, and then from the selected cluster selects a client according to the probability
vector ˜pt
m. The two probability vectors πt and ˜pt
m are defined as

πt =

"
exp(γt ¯Ht
1)
PM
m=1 exp(γt ¯Htm)
, . . . ,
exp(γt ¯Ht
M)
PM
m=1 exp(γt ¯Htm)

#

, ˜pt
m =

"
pk1
P

k∈Gm pk
, . . . ,
pk|Gm|
P

k∈Gm pk

#

,

(10)
where k1, . . . , k|Gm| are the indices of clients in cluster Gm, γt = γ0(1 −t

T ) denotes an annealing
hyper-parameter, and T is the number of global rounds. The annealing parameter is scheduled such
that at first it promotes sampling clients with balanced data, thus accelerating and stabilizing the
convergence of the global model. To avoid overfitting potentially caused by repeatedly selecting
a small subset of clients, the annealing parameter is gradually reduced to γt ≈0, when the server
samples the clusters uniformly. The described procedure is formalized as Algorithm 1.

3.5
Convergence Analysis

Adopting the standard assumptions of smoothness, unbiased gradients and bounded variance [7], the
following theorem holds for FedAvg with SGD optimizer.

6


---Page Break---
Theorem 3.4 Assume Fk(·) is L-smooth for all k ∈[N]. Let θt denote parameters of the global
model and let F(·) be defined as in Eq. 1. Furthermore, assume the stochastic gradient estimator
gk(θt) is unbiased and the variance is bounded such that E ∥gk(θt) −∇Fk(θt)∥2 ≤σ2. Let η and
R be the learning rate and the number of local epochs, respectively. If the learning rate is such that
η ≤
1
8LR, R ≥2, then

min
t∈[T ]

∇F(θt)
2 ≤1

T

 
F(θ0) −F(θ∗)

A1
+ A2

T −1
X

t=0

N
X

k=1
ωt
kσ2
k

!

+ Φ,
(11)

where A1, A2, Φ are positive constants, and ωt
k is the probability of sampling client k at round t.

Note that only the second term in the parenthesis on the right-hand side of the bound in Theorem 3.4
is related to the sampling method Π. Under Assumption 3.1,

N
X

k=1
ωt
kσ2
k ≤κ −

N
X

k=1
ωt
k
exp
 
βH(D(k))


exp (βH(D0)) ρ = κ −HΠ.
(12)

If the server samples clients with weights proportional to pk, the statistical heterogeneity of the
entire FL system may be characterized by HS = PN
k=1 pk
exp(β(H(D(k)))

exp(β(H(D0)) ρ. If all clients have class-
imbalanced data, HS is small and thus random sampling leads to unsatisfactory convergence rate (as
indicated by Theorem 3.4). On the other hand, since the clients sharing a cluster have similar data
entropy, the proposed HiCS-FL leads to ωt
k =
pk exp(γt ˆ
Ht(D(k)))
PN
j=1 pj exp(γt ˆ
Ht(D(j))). When training starts, HΠ is

large because the server tends to sample clients with higher pk exp(γtH(D(k))); as γt decreases,
HΠ eventually approaches HS. Further details and the proof of the theorem are in Appendix A.7.

4
Experiments

Setup. We evaluate the proposed HiCS-FL algorithm on four benchmark datasets (FMNIST, CI-
FAR10, Mini-ImageNet and THUC news) using different model architectures. We use four baselines:
random sampling, pow-d [8], clustered sampling (CS) [11], DivFL [2] and FedCor [28]. To generate
non-IID data partitions, we follow the strategy in [35], utilizing Dirichlet distribution with different
concentration parameters α which controls the level of heterogeneity (smaller α leads to generating
less balanced data). In a departure from previous works we utilize several different α to generate data
partitions for a single experiment, leading to a realistic scenario of varied data heterogeneity across
different clients. To quantify the performance of the tested methods, we use two metrics: (1) average
training loss, and (2) test accuracy of the learned global model. For better visualization, data points
in the results are smoothened by a Savitzky–Golay filter with window length 13 and the polynomial
order set to 3. Further details of the experimental setting and a visualization of data partitions are in
Appendix A.1 and A.10.

4.1
Comparison on Test Accuracy and Training Loss

FMNIST. We run FedAvg with SGD to train a global model which has CNN architecture in
an FL system with 50 clients, where 10% of clients are selected to participate in each round of
training. The data partitions are generated using one of 3 sets of the concentration parameter α
values: (1) {0.001, 0.002, 0.005, 0.01, 0.5}; (2) {0.001, 0.002, 0.005, 0.01, 0.2}; (3) {0.001}. These
are used to generate clients’ data so as to emulate the following scenarios: (1) 80% of clients have
severely imbalanced data while the remaining 20% have balanced data; (2) 80% clients have severely
imbalanced data while the remaining 20% have mildly imbalanced data; (3) all clients have severely
imbalanced data. Note that HM monotonically decreases as we go through settings (1) to (3). For a
fair comparison, pow-d and DivFL are deployed with their ideal settings where the server requires all
clients to precompute in each round a metric that is then used for client selection. Figure 2 shows
that HiCS-FL outperforms other methods across different settings, exhibiting the fastest convergence
rates and the least amount of variance. Particularly significant is the acceleration of convergence in
setting (1) where 20% of the participating clients have balanced data. Figure 3 shows that HiCS-FL
is helping achieve significant reduction of training variations (as expected, see Section 3.5) as evident
by a smooth loss trajectory.

7


---Page Break---
(a) FMNIST (1)
(b) FMNIST (2)
(c) FMNIST (3)

(d) CIFAR10 (1)
(e) CIFAR10 (2)
(f) CIFAR10 (3)

Figure 2: Test accuracy for the global model on 3 groups of data partitions of FMNIST and CIFAR10.

(a) FMNIST (1)
(b) CIFAR10 (1)
(c) Mini-ImageNet (1)

Figure 3: Training loss of HiCS-FL compared to four baselines for setting (1) on the three datasets.

CIFAR10. Here we compare the performance of HiCS-FL to FedProx [19] running CNN model
with Adam optimizer on the task of training an FL system with 50 clients, where 20% of clients are
selected to participate in each training round. Similar to the experiments on FMNIST, 3 sets of the
concentration parameter α are considered: (1) {0.001, 0.01, 0.1, 0.5, 1}; (2) {0.001, 0.002, 0.005,
0.01, 0.5}; (3) {0.001, 0.002, 0.005, 0.01, 0.1}. The interpretation of the scenarios emulated by these
setting is as same as in the FMNIST experiments. Figure 2 demonstrates improvement of HiCS-FL
over all the other methods. HiCS-FL exhibits particularly significant improvements in settings (2) and
(3), where 80% of the clients with extremely imbalanced data benefit from 20% of the clients with
either balanced or mildly imbalanced data. The advantage of HiCS-FL in setting (1) where all clients
have relatively high data heterogeneity is relatively modest (see Fig.2.(d)) because the system’s HS
is relatively large (see discussion in Section 3.5).

Mini-ImageNet. As in the Mini-ImageNet experiments, we compare HiCS-FL to FedProx running
ResNet18 with Adam optimizer but now consider training of an FL system with 100 clients, where
20% of the clients are selected to participate in each round of training. We consider two settings
of the concentration parameter α: (1) {0.001, 0.01, 0.1, 0.5, 1} and (2) {0.001, 0.005, 0.01, 0.1, 1}.
Setting (1) emulates the scenario where clients have a range of heterogeneity profiles, from extremely
imbalanced, through mildly imbalanced, to balanced, while setting (2) corresponds to the scenario
where 80% of the clients have extremely imbalanced data while the remaining 20% have balanced
data. The system’s H(1)
S
for setting (1) is larger than H(2)
S
for setting (2), which is reflected in a more
significant improvements achieved by HiCS-FL in the latter setting, as shown in Figure 4.

THUC news. To evaluate our method on data from a different domain, we conduct experiments involv-
ing text classification on the THUC news dataset in Chinese language (10 labels). Similar to the afore-
mentioned experiments, we allocate data to 50 clients by emulating heterogeneous data distributions
scenarios with parameter α set to: (1) {0.001, 0.01, 0.1, 0.2,1}; (2) {0.001, 0.002, 0.01, 0.1, 0.5};
and (3) {0.001, 0.002, 0.005, 0.01, 0.1}. We trained TextRNNs [20] with BiLSTM architecture as

8


---Page Break---
Table 1: Test accuracy (%) for the global model on 3 groups of data partitions of THUC news dataset.

Schemes
Random
Pow-of-Choice
CS
DivFL
FedCor
HiCS-FL
settng (1)
78.9
80.0
80.6
73.0
81.2
83.2
settng (2)
74.9
75.4
82.8
68.9
81.3
83.9
settng (3)
72.7
66.5
79.4
72.1
76.4
79.7

Table 2: The number of communication rounds needed to reach a certain test accuracy in the
experiments on FMNIST, CIFAR10, Mini-ImageNet and THUC News. All results are for the
concentration parameter setting (2).

Schemes
FMNIST
CIFAR10
Mini-ImageNet
THUC news
acc = 0.75
speedup
acc = 0.6
speedup
acc = 0.5
speedup
acc = 0.8
speedup
Random
149
1.0×
898
1.0×
191
1.0×
83
1.0×
pow-d
79
1.8↑
1037
0.9↓
432
0.4↓
109
0.8↓
CS
114
1.3↑
748
1.2↑
186
1.0×
74
1.1↑
DivFL
478
0.3↓
1417
0.6↓
726
0.3 ↓
289
0.3↓
FedCor
88
1.7↑
711
1.3↑
229
0.8↑
100
0.8↓
HiCS-FL
60
2.5↑
123
7.3↑
86
2.2↑
27
3.1↑

the classifiers using Adam optimizer. The test accuracy of the global model trained with different
schemes for 100 global rounds, reported in Table 1, show that our method outperforms baselines in
all the settings, demonstrating efficacy of our proposed algorithm in a simple NLP task.

4.2
Accelerating the Training Convergence

Figure 4: MiniImageNet acc.

In this section we report the communication costs required to achieve
convergence when using HiCS-FL, and compare those results with
the competing schemes. For brevity, we select one result from each
experiment conducted on the considered four datasets, and display
them in Table 2. As can be seen from the table, HiCS-FL significantly
reduces the number of communication rounds needed to reach target
test accuracy. On FMNIST, HiCS-FL needs 60 rounds to reach test
accuracy 0.75, achieving it 2.5 times faster than the random sampling
scheme. On CIFAR10, HiCS-FL requires only 123 rounds to reach
0.6 test accuracy, which is 7.3 times faster than random sampling.
Significant speedup appears on THUC dataset, in which HiCS-FL
only needs 27 rounds to achieve 0.8 test accuracy, 3.1 times faster
than the baseline. Acceleration on Mini-ImageNet is relatively modest
but HiCS-FL still outperforms other methods, and does so up to 2.2
times faster than random sampling.

Table 2 also shows that HiCS-FL provides the reported improvements without introducing major
computational and communication overhead. The only additional computation is due to estimating
data heterogeneity and performing clustering utilizing bias updates, which scales with the total
number of classes but does not increase with the size of the neural network model |θt|. Remarkably,
HiCS-FL outperforms pow-d, Clustered Sampling, DivFL and FedCor in terms of convergence speed,
variance and test accuracy while requiring significantly less computations. More details are provided
in Appendix A.11.

4.3
Number of Clustering Groups

As discussed at the end of Section 3.3, the distance function in Equation 9 can be reduced to the
conventional cosine similarity when clients exhibit similar levels of statistical heterogeneity, despite
potential differences in data distribution. Under these circumstances, our HiCS-FL method can
recover the performance of the previously established CS approach [11]. While CS suggests that
the number of clusters M should be greater than or equal to the number of selected clients K, our
HiCS-FL does not require M > K but adheres to the CS settings to ensure a fair comparison.
To elucidate the impact of the number of clusters, we conducted supplementary experiments with

9


---Page Break---
Table 3: Additional experimental results (accuracy in %) on HiCS with the number of clusters
M ≤K, where K is the number of selected clients each global round.

M
CIFAR10 (1)
CIFAR10 (2)
CIFAR10 (3)
Mini-ImageNet (1)
Mini-ImageNet (2)
M = 0.3K
61.3
57.0
47.5
50.4
50.1
M = 0.5K
65.1
61.5
46.2
49.8
50.4
M = 0.7K
62.8
59.2
51.2
51.1
49.9
M = K
65.5
59.8
50.6
50.5
51.2

Table 4: In experiments on CIFAR10, only 20 out of 50 clients are available in the beginning;
additional 10 clients join each 100 global rounds. The initial 20 clients leave the system after 400
global rounds.

Scheme
Random
pow-d
DivFL
CS
FedCor
HiCS-FL
CIFAR10 (1)
85.6
86.7
84.0
86.2
80.8
87.4
CIFAR10 (2)
93.7
93.3
91.6
93.7
93.7
94.7
CIFAR10 (3)
94.5
94.7
93.9
94.5
95.0
95.8
Mini-ImageNet (1)
67.3
67.2
67.5
67.8
68.7
69.0
Mini-ImageNet (2)
71.2
71.8
72.1
72.1
72.7
72.5

HiCS-FL using varying numbers of clusters M and compared these results to those obtained with
M = K as presented in the paper. The results of those experiments can be found in Table. 3. As
shown there, HiCS-FL can perform well with smaller M < K as long as M is not too small, such as
M = 3.

4.4
Dynamic Availability of Clients

The purpose of the warm-up phase (t < ⌈N/K⌉) shown in Alg. 1 is to collect updates of the output
layer from all the available clients in the system in order to facilitate clustering. Although we conduct
all the experiments in the setting where clients have fixed availability, our HiCS-FL does not assume
all the clients are available in the warm-up phase and can be adapted to more practical scenarios
where clients have dynamic availability.

In such a scenario, the warm-up phase can be implemented by the available clients at the beginning of
training. The proposed HiCS-FL is then implemented only among the available clients; the available
clients with more balanced data are preferred. When new clients join the system at the global round t,
the server can obtain the information of availability and selects these new clients at round t + 1 to
approximate their data heterogeneity. To provide more insights, we conduct additional experiments
on CIFAR10 dataset; the results are reported in Table. 4. As can be seen there, HiCS-FL outperforms
baselines that consider clients’ availability.

5
Conclusion

In this paper, we studied federated learning systems where clients that own non-IID data collabora-
tively train a global model; the system operates under communication constraints and thus only a
fraction of clients participates in any given round of training. We developed HiCS-FL, a hierarchical
clustered sampling method which estimates clients’ data heterogeneity and uses this information
to cluster and select clients to participate in training. We analyzed the performance of the pro-
posed heterogeneity estimation method, and the convergence of training a FL system that deploys
HiCS-FL. Extensive benchmarking experiments on four datasets demonstrated significant benefits
of the proposed method, including improvement in convergence speed, variance and test accuracy,
accomplished with only a minor computational overhead.

Acknowledgement

This work was funded in part by the NSF grant 2148224.

10


---Page Break---
References

[1] Durmus Alp Emre Acar, Yue Zhao, Ramon Matas Navarro, Matthew Mattina, Paul N Whatmough,
and Venkatesh Saligrama. 2021. Federated learning based on dynamic regularization. arXiv preprint
arXiv:2111.04263.

[2] Ravikumar Balakrishnan, Tian Li, Tianyi Zhou, Nageen Himayat, Virginia Smith, and Jeff Bilmes. 2022.
Diverse client selection for federated learning via submodular maximization. In International Conference
on Learning Representations.

[3] Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. 2018. A systematic study of the class imbalance
problem in convolutional neural networks. Neural networks, 106:249–259.

[4] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. 2002. Smote: synthetic
minority over-sampling technique. Journal of artificial intelligence research, 16:321–357.

[5] Huancheng Chen and Haris Vikalo. 2024. Recovering labels from local updates in federated learning.
arXiv preprint arXiv:2405.00955.

[6] Huancheng Chen, Chaining Wang, and Haris Vikalo. 2023. The best of both worlds: Accurate global
and personalized models through federated learning with data-free hyper-knowledge distillation. In The
Eleventh International Conference on Learning Representations.

[7] Wenlin Chen, Samuel Horvath, and Peter Richtarik. 2020. Optimal client sampling for federated learning.
arXiv preprint arXiv:2010.13723.

[8] Yae Jee Cho, Jianyu Wang, and Gauri Joshi. 2020. Client selection in federated learning: Convergence
analysis and power-of-choice selection strategies. arXiv preprint arXiv:2010.01243.

[9] Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai. 2021. Exploiting shared repre-
sentations for personalized federated learning. In International Conference on Machine Learning, pages
2089–2099. PMLR.

[10] Sever S Dragomir, Marcel L Scholz, and Jadranka Sunde. 2000. Some upper bounds for relative entropy
and applications. Computers & Mathematics with Applications, 39(9-10):91–100.

[11] Yann Fraboni, Richard Vidal, Laetitia Kameni, and Marco Lorenzi. 2021. Clustered sampling: Low-
variance and improved representativity for clients selection in federated learning. In International Confer-
ence on Machine Learning, pages 3407–3416. PMLR.

[12] Jack Goetz, Kshitiz Malik, Duc Bui, Seungwhan Moon, Honglei Liu, and Anuj Kumar. 2019. Active
federated learning. arXiv preprint arXiv:1909.12641.

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
770–778.

[14] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. 2020. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pages 5132–5143. PMLR.

[15] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980.

[16] Qinbin Li, Bingsheng He, and Dawn Song. 2021. Model-contrastive federated learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10713–10722.

[17] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. 2021. Ditto: Fair and robust federated
learning through personalization. In International Conference on Machine Learning, pages 6357–6368.
PMLR.

[18] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. 2020. Federated learning: Challenges,
methods, and future directions. IEEE signal processing magazine, 37(3):50–60.

[19] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. 2020.
Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems,
2:429–450.

[20] Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2016. Recurrent neural network for text classification with
multi-task learning. arXiv preprint arXiv:1605.05101.

11


---Page Break---
[21] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017.
Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and
statistics, pages 1273–1282. PMLR.

[22] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen,
Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance
deep learning library. Advances in neural information processing systems, 32.

[23] Monica Ribero and Haris Vikalo. 2020. Communication-efficient federated learning via optimal client
sampling. arXiv preprint arXiv:2007.15197.

[24] Felix Sattler, Klaus-Robert Müller, and Wojciech Samek. 2020. Clustered federated learning: Model-
agnostic distributed multitask optimization under privacy constraints. IEEE transactions on neural networks
and learning systems, 32(8):3710–3722.

[25] Ronald W Schafer. 2011. What is a savitzky-golay filter?[lecture notes]. IEEE Signal processing magazine,
28(4):111–117.

[26] Li Shen, Zhouchen Lin, and Qingming Huang. 2016. Relay backpropagation for effective learning of deep
convolutional neural networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam,
The Netherlands, October 11–14, 2016, Proceedings, Part VII 14, pages 467–482. Springer.

[27] Fang Shi, Weiwei Lin, Lisheng Fan, Xiazhi Lai, and Xiumin Wang. 2023. Efficient client selection based
on contextual combinatorial multi-arm bandits. IEEE Transactions on Wireless Communications.

[28] Minxue Tang, Xuefei Ning, Yitu Wang, Jingwei Sun, Yu Wang, Hai Li, and Yiran Chen. 2022. Fedcor:
Correlation-based active client selection strategy for heterogeneous federated learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10102–10111.

[29] Aidmar Wainakh, Fabrizio Ventola, Till Müßig, Jens Keim, Carlos Garcia Cordero, Ephraim Zimmer, Tim
Grube, Kristian Kersting, and Max Mühlhäuser. 2021. User label leakage from gradients in federated
learning. arXiv preprint arXiv:2105.09369.

[30] Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni. 2020.
Federated learning with matched averaging. arXiv preprint arXiv:2002.06440.

[31] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. 2020. Tackling the objective
inconsistency problem in heterogeneous federated optimization. Advances in neural information processing
systems, 33:7611–7623.

[32] Joel Wolfrath, Nikhil Sreekumar, Dhruv Kumar, Yuanli Wang, and Abhishek Chandra. 2022. Haccs:
Heterogeneity-aware clustered client selection for accelerated federated learning. In 2022 IEEE Interna-
tional Parallel and Distributed Processing Symposium (IPDPS), pages 985–995. IEEE.

[33] Haibo Yang, Minghong Fang, and Jia Liu. 2021. Achieving linear speedup with partial worker participation
in non-iid federated learning. arXiv preprint arXiv:2101.11203.

[34] Miao Yang, Ximin Wang, Hongbin Zhu, Haifeng Wang, and Hua Qian. 2021. Federated learning with class
imbalance reduction. In 2021 29th European Signal Processing Conference (EUSIPCO), pages 2174–2178.
IEEE.

[35] Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Nghia Hoang, and Yasaman
Khazaeni. 2019. Bayesian nonparametric federated learning of neural networks. In International conference
on machine learning, pages 7252–7261. PMLR.

[36] Jianyi Zhang, Ang Li, Minxue Tang, Jingwei Sun, Xiang Chen, Fan Zhang, Changyou Chen, Yiran Chen,
and Hai Li. 2022. Fed-cbs: A heterogeneity-aware client sampling mechanism for federated learning via
class-imbalance reduction. arXiv preprint arXiv:2209.15245.

12


---Page Break---
A
Appendix

A.1
Details of the Experiments

A.1.1
General Settings

The experimental results were obtained using Pytorch [22]. In the experiments involving FMNIST,
each client used a CNN-based classifier with two 5×5-convolutional layers and two 2×2-maxpooling
layers (with a stride of 2), followed by a fully-connected layer. In the experiments involving CIFAR10,
each client used a CNN-based classifier with three 3×3-convolutional layers and two 2×2-maxpooling
layers (with a stride of 2), followed by two fully-connected layers; dimension of the hidden layer was
64. In the experiments involving Mini-ImageNet and THUC news, each client fine-tuned a pretrained
ResNet18 [13] and learned a TextRNNs [20], respectively. The optimizers used for model training
in the experiments on FMNIST and CIFAR10/Mini-ImageNet/THUC news were the mini-batch
stochastic gradient descent (SGD) and Adam [15], respectively. The learning rate was initially set
to 0.001 and then decreased every 10 iterations, with a decay factor 0.5. The number of global
communication rounds was set to 200, 500, 100 and 100 for the experiments on FMNIST, CIFAR10,
Mini-ImageNet and THUC news, respectively. In all the experiments, the number of local epochs
R was set to 2 and the size of a mini-batch was set to 64. The sampling rate (fraction of the clients
participating in a training round) was set to 0.1 for the experiments on FMNIST/THUC news, and to
0.2 for the experiments on CIFAR10/Mini-ImageNet. For the sake of visualization, data points in the
presented graphs were smoothened by a Savitzky–Golay filter [25] with window length 13 and the
polynomial order set to 3.

A.1.2
Hyper-parameters

In all experiments, the hyper-parameter µ of the regularization term in FedProx [19] was set to 0.1.
In the Power-of-Choice (pow-d) [8] selection strategy, d was set to the total number of clients: 50 in
the experiments on FMNIST, CIFAR10 and THUC news, 100 in the experiments on Mini-ImageNet.
When running DivFL [2], we used the ideal setting where 1-step gradients were requested from all
client in each round (regardless of their participation status), similar to the Power-of-Choice settings.
For FedCor [28], we followed all settings in the paper and set the annealing coefficient β controlling
the sampling strategy to 0.9 as suggested in the paper. For HiCS-FL (our method), the scaling
parameter T (temperature) used in data heterogeneity estimation was set to 0.0025 in the experiments
on FMNIST and to 0.0015 in the experiments on CIFAR10/Mini-ImageNet. In all experiments,
parameter λ which multiplies the difference between clients’ estimated data heterogeneity (used in
clustering) was set to 10. In all experiments, the number of clusters m was for convenience set to
be equal to the number of selected clients K. The coefficient γ0 was set to 4 in the experiments on
FMNIST and CIFAR10 while set to 2 in the experiments on Mini-ImageNet. To group clients, both
Clustered Sampling [11] and HiCS-FL (our method) utilized an off-the-shelf clustering algorithm
performing hierarchical clustering with Ward’s Method.

A.2
Empirical Validation of Assumption 3.1

To illustrate and empirically validate Assumption 3.1, we conducted extensive experiments on
FMNIST and CIFAR10 with the same model mentioned in Section A.1. In particular, we varied α
over 250 values in the interval [0.01, 50] to generate data partitions allocated to 250 clients; entropy
of the generated label distributions ranged from 0 to ln 10 (maximum). In these experiments, we
allowed all clients to participate in each of 500 training rounds. To facilitate the desired study, in
addition to these 250 clients we also simulated a super-client which owns a data set aggregating
the data from all the clients (the set of labels in the aggregated dataset is uniformly distributed).
In each round, clients start from the initialized global model and compute local gradients on their
datasets; the super-client does the same on the aggregated dataset. The server computes and records
squared Euclidean norm of the difference between the local gradients and the “true" gradient (i.e.,
the super-client’s gradient). In each round, the difference between the local gradient and the true
gradient changes in a pattern similar to what is stated in Assumption 3.1. As an illustration, we plot
all such gradient differences computed during the entire training process of a client. Specifically,
the server computes the difference between local gradient and the true gradient in each round of
training, obtaining 250 × 500 = 12500 data points that correspond to 250 data partitions. For better
visualization, we merged adjacent points.

13


---Page Break---
The results obtained by following these steps in experiments on FMNIST and CIFAR10 are shown in
Figure 5. For a more informative visualization, the horizontal coordinate of a point in the scatter plot is
H(D(k)), while the vertical coordinate is ∥ηt∇Fk(θt) −ηt∇F(θt)∥2. The dashed lines correspond
to the curves y = −exp(β [x −H(D0)])ρ + κ that envelop the majority of the generated points.
In the case of FMNIST, the blue dashed line is parametrized by β = 1.0, ρ = 0.13, and κ = 0.14
while the green dashed line is parametrized by β = 1.5, ρ = 0.025, and κ = 0.022; these two lines
envelop 95% of the generated points. In the case of CIFAR10, the blue dashed line is parametrized by
β = 2.0, ρ = 0.30, and κ = 0.36 while the green dashed line is parametrized by β = 1.8, ρ = 0.15,
and κ = 0.20; as in the other plot, these two lines envelop 95% of the generated points. As the
plots indicate, the difference between the local gradient and the true gradient increases as H(D(k))
decreases, implying that the local gradient computed by a client with more balanced data is closer to
the true gradient.

(a) FMNIST
(b) CIFAR10

Figure 5: Visualization of the difference between local gradients and the global gradient (evaluated if
all the data is centrally collected).

A.3
Gradient of the output (fully connected) layer’s bias

Given a batch of samples (x(j,n), y(j,n)), the cross-entropy loss is readily computed as

LCE = −1

Bl

l
X

j=1

B
X

n=1
log
exp

q(j,n)
y(j,n)


PC
c=1 exp

q(j,n)
c
 = 1

Bl

l
X

j=1

B
X

n=1
L(j,n)
CE ,
y(j,n) ∈[C]
(13)

q(j,n)
c
=

L
X

d=1
wd,cz(j,n)
d
+ bc,
(14)

where B is the batchsize; l is the number of mini-batches; C is the number of classes; d is the
dimension of the hidden space; z(j,n)
d
denotes the d-th feature in the hidden space given sample x(j,n)

in the j-th batch; wd,c and bc denote the weight of z(j,n)
d
and the bias for the neuron that outputs the
probability of the class c, respectively; and q(j,n)
c
is the corresponding output logit on class c. The
gradient of the bias bi given sample x(j,n) can be computed by the chain rule as

∂L(j,n)
CE
∂bi
= −∂L(j,n)
CE
∂Q
·
∂Q

∂q(j,n)
i
· ∂q(j,n)
i
∂bi
,
(15)

where

Q =
exp

q(j,n)
y(j,n)


PC
c=1 exp

q(j,n)
c
.
(16)

Then
∂L(j,n)
CE
∂Q
= 1

Q,
∂q(j,n)
i
∂bi
= 1.
(17)

14


---Page Break---
If i = y(j,n),

∂Q

∂q(j,n)
i
=
exp

q(j,n)
y(j,n)
 PC
c=1 exp

q(j,n)
c

−exp

q(j,n)
y(j,n)
2

PC
c=1 exp

q(j,n)
c
2
=
Q P

c̸=y(j,n) exp

q(j,n)
c


PC
c=1 exp

q(j,n)
c

.

(18)
If i ̸= y(j,n),

∂Q

∂q(j,n)
i
= −
exp

q(j,n)
y(j,n)

exp

q(j,n)
i


PC
c=1 exp

q(j,n)
c
2
= −
Q exp

q(j,n)
i


PC
c=1 exp

q(j,n)
c
.
(19)

By plugging Eq. 18 and 19 in Eq. 15, we obtain

∂L(j,n)
CE
∂bi
= −

P

c̸=y(j,n) exp

q(j,n)
c


PC
c=1 exp

q(j,n)
c

, if i = y(j,n); ∂L(j,n)
CE
∂bi
=
exp

q(j,n)
i


PC
c=1 exp

q(j,n)
c
, if i ̸= y(j,n).

(20)

A.4
Expectation of the local update ∆b(k)

By combining Eq. 4 and 5 and taking expectation, we obtain

E
h
∆b(k)
i
i
= −η

Bl

l
X

j=1

B
X

n=1

R
X

r=1
E
h
∇biL(j,n,r)
CE
i

= η

R
X

r=1
P{i = y(j,n)}I{i = y(j,n)} 1

Bl

l
X

j=1

B
X

n=1

P

c̸=i exp(q(j,n,r)
c
)
PC
c=1 exp(q(j,n,r)
c
)

−η

R
X

r=1
P{i ̸= y(j,n)}I{i ̸= y(j,n)} 1

Bl

l
X

j=1

B
X

n=1

exp(q(j,n,r)
i
)
PC
c=1 exp(q(j,n,r)
c
)

= η

R
X

r=1
D(k)
i
X

c̸=i
E(x,y)∼B−c

"
exp(q(j,n,r)
c
)
PC
c=1 exp(q(j,n,r)
c
)

#

−η

R
X

r=1
(1 −D(k)
i
)E(x,y)∼B−i

"
exp(q(j,n,r)
i
)
PC
c=1 exp(q(j,n,r)
c
)

#

= η

R
X

r=1
D(k)
i
X

c̸=i
E(x,y)∼B−c 
s−c
c (x)

−η

R
X

r=1
(1 −D(k)
i
)E(x,y)∼B−i 
s−i
i (x)


= ηR



D(k)
i
X

c̸=i
Ec −(1 −D(k)
i
)Ei





= ηR

 

D(k)
i

C
X

c=1
Ec −Ei

!

.

(21)

15


---Page Break---
Note that

C
X

i=1
Ei =

C
X

i=1
E(x,y)∼B−i 
s−i
i (x)


= E




C
X

i=1

1
C −1

X

c̸=i

1

BlD(k)
c

l
X

j=1

B
X

n=1
I{y(j,n) = c}
exp(q(j,n)
i
)
PC
c=1 exp(q(j,n)
c
)





=
1
C −1

C
X

i=1

1

BlD(k)
i

l
X

j=1

B
X

n=1
P{y(j,n) = i}

P

c̸=i exp(q(j,n)
c
)
PC
c=1 exp(q(j,n)
c
)

= −
C
C −1
1
Bl

l
X

j=1

B
X

n=1

exp(q(j,n)
y(j,n))
PC
c=1 exp(q(j,n)
c
)
+
C
C −1

(22)

A comparison to LCE in Eq. 13 reveals that as LCE decreases during training, so does PC
i=1 Ei.
Given an untrained/initialized neural network model, E0
i = 1/C for ∀i ∈[C], i.e., PC
i=1 E0
i =
−
1
C−1 +
C
C−1 = 1. At global round T, if L∗
CE = 0, then PC
i=1 ET
i = −
C
C−1 +
C
C−1 = 0.

A.5
Privacy of D(k)

According to Eq. 6, the server is able to obtain C linear equations from each client,

E
h
∆b(k)
i
i
= ηR

 

D(k)
i

C
X

c=1
Ec −Ei

!

, for ∀i ∈[C],
(23)

C
X

i=1
D(k)
i
= 1,
(24)

where C denotes the number of classes. Suppose E[∆b(k)
i
] are known by the server. Then D(k)
i
, the
variables in the aforementioned equations, cannot be determined uniquely since there are C variables
and C + 1 equations. Therefore, the server is unable to infer clients’ true data label distribution and
the privacy of D(k) is protected.

A.6
Proof of Theorem 3.3

In Section A.3 we derived an expression for the gradient of the bias in the output layer given a single
sample (x(j,n), y) in the mini-batch. It is worthwhile making the following two observations:

• the sign of the y(j,n)-th component of ∇bL(j,n)
CE (x(j,n), y(j,n)) is opposite of the sign of the
other components; and

• the y(j,n)-th component of ∇bL(j,n)
CE (x(j,n), y(j,n)) is equal in magnitude to all other com-
ponents combined.

Proof: Let ∆b(k) = [∆b(k)
1 , . . . , ∆b(k)
C ] denote the local update (made by client k) of the bias in
the output layer of the neural network model, and let D(k) = [D(k)
1 , . . . , D(k)
C ] be the (unknown)
true data label distribution, PC
i=1 D(k)
i
= 1. Assuming the learning rate η and R local epochs, the
expectation of the local update of ∆b(k) is

E
h
∆b(k)
i
i
= ηR

 

D(k)
i

C
X

c=1
Ec −Ei

!

.
(25)

Data heterogeneity can be captured via entropy, H(D(k)) = −PC
c=1 D(k)
i
ln D(k)
i
, where higher
H(D(k)) indicates that client k has more balanced data. However, since we do not have access to

16


---Page Break---
the client’s data distribution, we instead define and use as a measure of heterogeneity ˆH(D(k)) ≜
H(softmax(∆b(k), T)), where

softmax(∆b(k), T)i =
exp(∆b(k)
i
/T)
PC
c=1 exp(∆b(k)
c /T)
,
(26)

and where T denotes the temperature of the softmax operator. Suppose there are two clients, u and k,
with class-balanced and class-imbalanced data; let D(u) and D(k) denote their data label distributions,
respectively, while ˆD(u) and ˆD(k) are computed by softmax(∆b(u), T) and softmax(∆b(k), T).
Without a loss of generality, we can re-parameterize ˆD(u) as

ˆD(u) = ϵU +

C
X

i=1
ϵiZi,
(27)

where U = [ 1

C , . . . , 1

C ] denotes uniform distribution; i-th component of Zi is 1 while the remaining
components are 0; ϵ and ϵi are all non-negative such that ϵ + PC
i=1 ϵi = 1. We can always set
minj ϵj = 0; otherwise, let ϵ
′ = ϵ + minj ϵj and ϵ
′
i = ϵi −minj ϵj, ∀i ∈[C]; ϵ quantifies how close
is ˆD(u) to U. Due to the concavity of entropy,

H( ˆD(u)) ≥ϵH(U) +

C
X

i=1
ϵiH(Zi) = ϵ ln C.
(28)

We will find the following lemma useful.

Lemma A.1 For two probability vectors p and q with dimension C, the Kullback–Leibler divergence
between p and q satisfies

KLD(p||q) ≥1

2 ∥p −q∥2
1 ,
(29)

where ∥p −q∥1 = PC
i=1 |pi −qi|.

For the proof of the lemma, please see [10]. Applying it, we obtain

KLD( ˆD(k)||U) = H(U) −H( ˆD(k)) ≥1

2

 ˆD(k) −U

2

1 ≥1

2

 ˆD(k) −U

2

2 .
(30)

Combining Eq. 28 and Eq. 30, we obtain

H( ˆD(u)) −H( ˆD(k)) ≥(ϵ −1) ln C + 1

2

 ˆD(k) −U

2

2 .
(31)

By taking expectations of both sides,

E
h
H( ˆD(u)) −H( ˆD(k))
i
≥(E[ϵ] −1) ln C + 1

2E
 ˆD(k) −U

2

2


.
(32)

Since
 ˆD(k) −U

2

2 is convex (composition of the Euclidean norm and softmax), according to
Jensen’s inequality

E
h
H( ˆD(u)) −H( ˆD(k))
i
≥(E[ϵ] −1) ln C + 1

2

 ˆD(k)(E[∆b(k)]) −U

2

2 ,
(33)

where

ˆD(k)(E[∆b(k)])i =
exp

ηR

D(k)
i
PC
c=1 Ec −Ei

/T


PC
j exp

ηR

D(k)
j
PC
c=1 Ec −Ej

/T
.
(34)

Selecting T such that ηR

D(k)
i
PC
c=1 Ec −Ei

/T is sufficiently small and applying the first-order
Taylor’s expansion of ex around 0, we obtain

C
X

j
exp

 

ηR

 

D(k)
j

C
X

c=1
Ec −Ej

!

/T

!

=

C
X

j
1 + ηR

C
X

j

 

D(k)
j

C
X

c=1
Ec −Ej

!

/T = C,
(35)

17


---Page Break---
where PC
j=1 D(k)
j
= 1. This leads to a simplified ˆD(k)(E[∆b(k)]),

ˆD(k)(E[∆b(k)])i =
1 + ηR

D(k)
i
PC
c=1 Ec −Ei

/T

C
.
(36)

Substituting Eq. 36 for the second term on the right-hand side of ineq. 33 leads to

 ˆD(k)(E[∆b(k)]) −U

2

2 =
 ηR

CT

2
C
X

i=1

 

D(k)
i

C
X

c=1
Ec −Ei

!2

.
(37)

Now, consider

ˆD(u) −U = (ϵ −1)U +

C
X

i=1
ϵiZi.
(38)

Taking expectations of both sides,

E

"

(ϵ −1)U +

C
X

i=1
ϵiZi

#

= E
h
ˆD(u) −U
i
≥ˆD(u)(E[∆b(u)]) −U.
(39)

The above inequality holds component-wise, so for the j-component (ϵj = 0)

E[ 1

C (ϵ −1) + ϵj] = E[ 1

C (ϵ −1)] ≥ˆD(u)(E[∆b(u)])j −Ui =
ηR

D(u)
j
PC
c=1 Ec −Ej


CT
. (40)

Therefore,

E[ϵ] −1 ≥
ηR

D(u)
j
PC
c=1 Ec −Ej


T
≥min
i

ηR

D(u)
i
PC
c=1 Ec −Ei


T
.
(41)

Taking absolute value of both sides yields

|E[ϵ]−1| ≤ηR

T max
i

D(u)
i

C
X

c=1
Ec −Ei

 = ηR

T max
i

(D(u)
i
−1

C )

C
X

c=1
Ec −Ei + 1

C

C
X

c=1
Ec

 . (42)

By applying the triangle inequality we obtain

|E[ϵ] −1| ≤ηR

T max
i

D(u)
i
−1

C



C
X

c=1
Ec + ηR

T max
i


1
C

C
X

c=1
Ec −Ei

 .
(43)

Let δ = maxi
 1

C
PC
c=1 Ec −Ei
. Since PC
c=1 Ec ≤C 1

C = 1, it holds that

|E[ϵ] −1| ≤ηR

T max
i

D(u)
i
−1

C

 + ηR

T δ.
(44)

Furthermore, since E[ϵ] −1 < 0,

E[ϵ] −1 ≥−ηR

T max
i

D(u)
i
−1

C

 −ηR

T δ.
(45)

Note that
 

D(k)
i

C
X

c=1
Ec −Ei

!2

=

 

(D(k)
i
−1

C )

C
X

c=1
Ec −Ei + 1

C

C
X

c=1
Ec

!2

=

 

(D(k)
i
−1

C )

C
X

c=1
Ec

!2

+

 
1
C

C
X

c=1
Ec −Ei

!2

+ 2

 C
X

c=1
Ec

! 
D(k)
i
−1

C

  
1
C

C
X

c=1
Ec −Ei

!

≥

 

(D(k)
i
−1

C )

C
X

c=1
Ec

!2

+ 2

 C
X

c=1
Ec

! 
D(k)
i
−1

C

  
1
C

C
X

c=1
Ec −Ei

!

.

(46)

18


---Page Break---
Therefore,

C
X

i=1

 

D(k)
i

C
X

c=1
Ec −Ei

!2

≥

 C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2

+ 2

 C
X

c=1
Ec

!
C
X

i=1


D(k)
i
−1

C

  
1
C

C
X

c=1
Ec −Ei

!

=

 C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2

+ 2

 C
X

c=1
Ec

!
C
X

i=1

 
D(k)
i
C

C
X

c=1
Ec −1

C2

C
X

c=1
Ec + Ei

C −D(k)
i
Ei

!

=

 C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2

+ 2

 C
X

c=1
Ec

!  
1
C

C
X

c=1
Ec −1

C

C
X

c=1
Ec + 1

C

C
X

i=1
Ei −

C
X

i=1
D(k)
i
Ei

!

≥

 C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2
+ 2

 C
X

c=1
Ec

!  
1
C

C
X

c=1
Ec −max
j
Ej

!

≥

 C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2
−2δ.

(47)
Substituting the above expression in Eq. 33, we obtain

E
h
H( ˆD(u)) −H( ˆD(k))
i
≥−ηR ln C

T
max
j

D(u)
j
−1

C

 −ηR ln C

T
δ
(48)

+ 1

2

 ηR

CT

2  C
X

c=1
Ec

!2
C
X

i=1


D(k)
i
−1

C

2
−
 ηR

CT

2
δ,
(49)

and, therefore,

E
h
H( ˆD(u)) −H( ˆD(k))
i
≥1

2

 
ηR
CT

C
X

c=1
Ec

!2 D(k) −U

2

2 −ηR ln C

T

D(u) −U

∞−Cδ,

(50)

where C = ηR(ηR+C2T ln C)

C2T 2
.
■

A.7
Convergence Analysis

Here we present the convergence analysis of an FL system deploying FedAvg with SGD wherein
only a small fraction of clients participates in any given round of training. Recall that the objective
function that comes up when training a neural network model is generally non-convex; we make the
standard assumptions of smoothness, unbiased gradient estimate, and bounded variance.

Assumption A.2 (Smoothness) Each local objective function Fk(·) is L-smooth,
∇Fk(θt+1
k
) −∇Fk(θt
k)

2 ≤L
θt+1
k
−θt
k

2 .
(51)

Assumption A.3 (Gradient oracle) The stochastic gradient estimator gk(θt,r
k ) = ∇Fk(θt,r
k ) + ζt,r
k
for each global round t and local epoch r is such that

E[ζt,r
k ] = 0
(52)

19


---Page Break---
and

E
hζt,r
k
2 |θt,r
k
i
≤σ2.
(53)

With these three assumptions in place, we provide the proof of Theorem 3.4 stated in the main
paper. The proof relies on the technique previously used in [7, 33], where the sampling method
is unbiased and thus E
h
1
K
P

k∈St
PR
r=1 gk(θt,r
k )
i
= PN
k=1
PR
r=1 pk∇Fk(θt,r
k ). We provide a

generalization that holds for any sampling strategy, resulting in E
h
1
K
P

k∈St
PR
r=1 gk(θt,r
k )
i
=
PN
k=1
PR
r=1 ωt
k∇Fk(θt,r
k ), where ωt
k denotes the probability of sampling client k in round t under
sampling strategy Π. Note that PN
k=1 ωt
k = 1. We assume that all clients deploy the same number of
local epochs R and use learning rate η at round t.

A.7.1
key lemma

Lemma A.4 (Lemma 2 in [33]) Instate Assumptions 3.1, A.2 and A.3. For any step size η such that
η ≤
1
8LR, for any client k it holds that

E
hθt,r
k
−θt2i
≤5Rη2(σ2 + 6Rσ2
k) + 30R2η2 ∇F(θt)
2 .
(54)

Proof of Lemma A.4: For any client k ∈[N] and r ∈[R],

E
hθt,r
k
−θt2i
= E
θt,r−1
k
−θt −ηgk(θt,r−1
k
)

2

= E[∥θt,r−1
k
−θt −η(gk(θt,r−1
k
) −∇Fk(θt,r−1
k
) + ∇Fk(θt,r−1
k
)

−∇Fk(θt) + ∇Fk(θt) −∇F(θt) + ∇F(θt))∥2]

≤

1 +
1
2R −1


E
θt,r−1
k
−θt
2
+ η2E
gk(θt,r−1
k
) −∇Fk(θt,r−1
k
)

2

+ 6Rη2E
∇Fk(θt,r−1
k
) −∇Fk(θt)

2
+ 6Rη2E
gk(∇Fk(θt) −∇F(θt)
2

+ 6Rη2E
∇F(θt)
2

≤

1 +
1
2R −1


E
θt,r−1
k
−θt
2
+ η2σ2 + 6Rη2L2E
θt,r−1
k
−θt
2

+ 6Rη2σ2
k + 6Rη2E
∇F(θt)
2

=

1 +
1
2R −1 + 6Rη2L2

E
θt,r−1
k
−θt
2
+ η2σ2 + 6Rη2σ2
k

+ 6Rη2E
∇F(θt)
2

≤

1 +
1
R −1


E
θt,r−1
k
−θt
2
+ η2σ2 + 6Rη2σ2
k + 6Rη2E
∇F(θt)
2 .

(55)

Unrolling the recursion yields

E
hθt,r
k
−θt2i
≤

R
X

r=1


1 +
1
R −1

r−1 
η2σ2 + 6Rη2σ2
k + 6Rη2E
∇F(θt)
2

≤(R −1)

"
1 +
1
R −1

R
−1

# 
η2σ2 + 6Rη2σ2
k + 6Rη2E
∇F(θt)
2

≤5Rη2  
σ2 + 6Rσ2
k

+ 30R2η2 ∇F(θt)
2 .
(56)
■

20


---Page Break---
A.7.2
Proof of Theorem 3.4

The model update at global round t is formed as

θt+1 = θt −η 1

K

X

k∈St

R
X

r=1
gk(θt,r
k ),
(57)

where θt+1 and θt denote parameters of the global model at rounds t + 1 and t, respectively, and θt,r
k
denotes parameters of the local model of client k after r local training epochs. Let

∆t ≜1

K

X

k∈St

R
X

r=1
gk(θt,r
k ).
(58)

Taking the expectations (conditioned on θt) of both sides, we obtain

E

F(θt+1)

= E

F(θt −η∆t)


(a)
≤F(θt) −η

∇F(θt), E

∆t
+ L

2 η2E
h∆t2i

= F(θt) + η

∇F(θt), E

R∇F(θt) −R∇F(θt) −∆t
+ L

2 η2E
h∆t2i

= F(θt) −Rη
∇F(θt)
2 + η

∇F(θt), E

R∇F(θt) −∆t

|
{z
}
A1

+L

2 η2 E
h∆t2i

|
{z
}
A2

.

(59)
Inequality (a) in the expression above holds due to the smoothness of F(·) (see Assumption A.2).
Note that the term A1 can be bounded as

A1 =

∇F(θt), E

R∇F(θt) −∆t

=

*

∇F(θt), E

"

R∇F(θt) −1

K

X

k∈St

R
X

r=1
gk(θt,r
k )

#+

=

*

∇F(θt), E

R∇F(θt)

−

N
X

k=1

R
X

r=1
ωt
k∇Fk(θt,r
k )

+

=

N
X

k=1
ωt
k

*
√

R∇F(θt), −1
√

R
E

" R
X

r=1

 
∇Fk(θt,r
k ) −∇F(θt)

#+

(a)
= R

2

∇F(θt)
2 + 1

2R

N
X

k=1
ωt
kE



R
X

r=1

 
∇Fk(θt,r
k ) −∇F(θt)



2

−1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

(b)
≤R

2

∇F(θt)
2 + 1

R

N
X

k=1
ωt
kE



R
X

r=1

 
∇Fk(θt,r
k ) −∇Fk(θt)



2

+ 1

R

N
X

k=1
ωt
kE



R
X

r=1

 
∇Fk(θt) −∇F(θt)



2

−1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

(c)
≤R

2

∇F(θt)
2 +

N
X

k=1
ωt
k

R
X

r=1
E
∇Fk(θt,r
k ) −∇Fk(θt)
2

+

N
X

k=1
ωt
k

R
X

r=1
E
∇Fk(θt) −∇F(θt)
2 −1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

(d)
≤R

2

∇F(θt)
2 + L2
N
X

k=1
ωt
k

R
X

r=1
E
θt,r
k
−θt2 + R

N
X

k=1
ωt
kσ2
k −1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

,

(60)

21


---Page Break---
where equality (a) follows from ⟨x, y⟩= 1

2 (∥x∥2 + ∥y∥2 −∥x −y∥2), inequality (b) is due to
∥x + y∥2 ≤2 ∥x∥2 + 2 ∥y∥2, inequality (c) holds because ∥Pn
i=1 zi∥2 ≤n Pn
i=1 ∥zi∥2, and
inequality (d) follows from Assumptions 3.1 and A.2. By selecting η <
1
8LR and applying Lemma
A.4 we obtain

A1 ≤R

2

∇F(θt)
2 + L2
N
X

k=1
ωt
k

R
X

r=1

h
5Rη2(σ2 + 6Rσ2
k) + 30R2η2 ∇F(θt)
2i

+ R

N
X

k=1
ωt
kσ2
k −1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

=
R

2 + 30L2R3η2
 ∇F(θt)
2 + 5L2R2η2σ2 + 30L2R3η2
N
X

k=1
ωt
kσ2
k

+ R

N
X

k=1
ωt
kσ2
k −1

2R

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

.

(61)

Furthermore,

A2 = E






1
K

X

k∈St

R
X

r=1
gk(θt,r
k )



2



= E







N
X

k=1

I{k ∈St}

K

R
X

r=1
gk(θt,r
k )



2



= E







N
X

k=1

I{k ∈St}

K

R
X

r=1
gk(θt,r
k ) −∇Fk(θt,r
k ) + ∇Fk(θt,r
k )



2



(a)
= E







N
X

k=1

I{k ∈St}

K

R
X

r=1
gk(θt,r
k ) −∇Fk(θt,r
k )



2

+ E







N
X

k=1

I{k ∈St}

K

R
X

r=1
∇Fk(θt,r
k )



2



(b)
≤E

" N
X

k=1

I{k ∈St}

K

R
X

r=1

gk(θt,r
k ) −∇Fk(θt,r
k )
2
#

+ E




N
X

k=1

I{k ∈St}

K



R
X

r=1
∇Fk(θt,r
k )



2



(c)
≤Rσ2 +

N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

,

(62)
where equation (a) holds because E

gk(θt,r
k ) −∇Fk(θt,r
k )

= 0, inequality (b) stems from the
Jensen’s inequality, and inequality (c) is due to Assumption A.3.

Substituting inequalities (61) and (62) into inequality (59) yields

E

F(θt+1)

≤F(θt) −Rη
∇F(θt)
2 + η

∇F(θt), E

R∇F(θt) −∆t

|
{z
}
A1

+L

2 η2 E
h∆t2i

|
{z
}
A2

≤F(θt) −Rη
1

2 −30L2R2η2
 ∇F(θt)
2 +

5L2R2η3 + LR

2 η2

σ2

+
 
30L2R3η3 + Rη
 N
X

k=1
ωt
kσ2
k +
L

2 η2 −η

2R

 N
X

k=1
ωt
kE



R
X

r=1
∇Fk(θt,r
k )



2

.

(63)

22


---Page Break---
If η <
1
8LR, it must be that 1

2 −30L2R2η2 > 0 and L

2 η2 −
η
2R < 0, leading to

E

F(θt+1)

≤F(θt) −Rη
1

2 −30L2R2η2
 ∇F(θt)
2

+

5L2R2η3 + LR

2 η2

σ2 +
 
30L2R3η3 + Rη
 N
X

k=1
ωt
kσ2
k.
(64)

By rearranging and summing from t = 0 to t = T −1 we obtain

E

F(θT )

−F(θ0) ≤−Rη
1

2 −30L2R2η2
 T −1
X

t=0

∇F(θt)
2

+

5L2R2η3 + LR

2 η2

T σ2 +
 
30L2R3η3 + Rη
 T −1
X

t=0

N
X

k=1
ωt
kσ2
k

≤−Rη
1

2 −30L2R2η2

T min
t∈[T ]

∇F(θt)
2

+

5L2R2η3 + LR

2 η2

T σ2 +
 
30L2R3η3 + Rη
 T −1
X

t=0

N
X

k=1
ωt
kσ2
k.

(65)

Let θ∗denote the optimal model’s parameters, i.e., F(θ∗) ≤F(θt)∀t ∈[T ]. Then

min
t∈[T ]

∇F(θt)
2 ≤1

T

 
F(θ0) −F(θ∗)

A1
+ A2

T −1
X

t=0

N
X

k=1
ωt
kσ2
k

!

+ Φ,
(66)

where A1 = Rη
  1

2 −30L2R2η2
, A2 =
60L2R3η3+2Rη
Rη(1−60L2R2η2) and Φ = (10L2Rη2+Lη)σ2

1−60L2R2η2
.

■

A.8
Regularization Terms in the Objective Function

The proposed method for estimating clients’ data heterogeneity relies on the properties of gradient
computed for the cross-entropy loss objective. However, the method also applies to the FL algorithms
other than FedAvg, in particular those that add a regularization term to combat overfitting, including
FedProx [19], FedDyn[1] and Moon [16]. In the following discussion, we demonstrate that HiCS-FL
remains capable of distinguishing between clients with imbalanced and balanced data when using
these other FL algorithms.

A.8.1
FedProx

The objective function used by FedProx [19] is

Lr
prox = Lr
CE + µ

2

θt,r
k
−θt2 ,
(67)

where θt,r
k
is the vector of client k’s local model parameters in the r-th local epoch at global round t.
Therefore, contribution of sample (x(j,n), y(j,n)) to the gradient of Lprox in local epoch r is

∂L(j,n,r)
prox
∂bi
= ∂L(j,n,r)
CE
∂bi
+ µ
 
bt,r
i
−bt
i

,
(68)

where bt,r = [bt,r
1 , . . . , bt,r
C ] denotes parameters of bias in the output layer of the local model, and
bt = [bt
1, . . . , bt
C] denotes parameters of the global model at round t. We assume the model is trained
by SGD as the optimizer, and hence

bt,r
i
−bt
i = bt,r−1
i
−ηt
∂L(j,n,r−1)
prox

∂bi
−bt
i = −ηt
∂L(j,n,r−1)
CE

∂bi
+ (1 −ηtµ)(bt,r−1
i
−bt
i).
(69)

23


---Page Break---
Therefore,

bt,r
i
−bt
i = −ηt

r−1
X

s=1
(1 −ηtµ)r−1−s ∂L(j,n,s)
CE
∂bi
+ (1 −ηtµ)r−1(bt
i −bt
i)

= −ηt

r−1
X

s=1
(1 −ηtµ)r−1−s ∂L(j,n,s)
CE
∂bi
,

(70)

and thus

∂L(j,n,r)
prox
∂bi
= ∂L(j,n,r)
CE
∂bi
−ηtµ

r−1
X

s=1
(1 −ηtµ)r−1−s ∂L(j,n,s)
CE
∂bi
.
(71)

Taking expectation of both sides yields

1
Bl

l
X

j=1

B
X

n=1

R
X

r=1
E

"
∂L(j,n,r)
prox
∂bi

#

=



−E[I{i = y(j,n)}]
X

c̸=i
Ec + E[I(i ̸= y(j,n))]Ei





·

R
X

r=1

 

1 −ηtµ

r−1
X

s=1
(1 −ηtµ)r−1−s
!

=

R
X

r=1



−D(k)
i
X

c̸=i
Ec + (1 −D(k)
i
)Ei





1 −ηtµ1 −(1 −ηtµ)r−1

ηtµ



=

R
X

r=1
cr
 

−D(k)
i

C
X

c=1
Ec + Ei

!

,

(72)
where cr = (1 −ηtµ)r−1 > 0 provided ηt and µ are sufficiently small. Therefore, the expectation of
the local update of bias in the output layer satisfies

E
h
∆b(k)
i
i
= Cηt

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(73)

where C = PR
r=1 cr. Eq. (73) is similar to the expression for the expectation of the local updates of
bias when applying FedAvg presented in the main paper; clearly, the analysis of HiCS-FL done in the
context of FedAvg extends to FedProx.

A.8.2
FedDyn

For FedDyn [1], the objective function in local epoch r at global round t is

Lt,r
dyn = Lt,r
CE −
D
∇Lt−1,R
dyn
, θt,r
k
E
+ µ

2

θt,r
k
−θt2 ,
(74)

where R denotes the total number of local epochs. The first order condition for local optima implies

∇Lt,r
dyn −∇Lt−1,R
dyn
+ µ(θt,r
k
−θt) = 0,
(75)

24


---Page Break---
and, therefore,
∂Lt,r
dyn
∂bi
=
∂Lt−1,R
dyn
∂bi
−µ
 
bt,r
i
−bt
i


=
∂Lt−2,R
dyn
∂bi
−µ

bt−1,R
i
−bt−1
i

−µ
 
bt,r
i
−bt
i


= −µ

t−1
X

τ=1


bτ,R
i
−bτ
i

−µ
 
bt,r
i
−bt
i


= −µ

t−1
X

τ=1
∆bτ
i −µ
 
bt,r
i
−bt
i


= −µ

t−1
X

τ=1
∆bτ
i −µ

 

−ηt
∂Lt,r−1
dyn
∂bi
+ bt,r−1
i
−bt
i

!

= −µ

t−1
X

τ=1
∆bτ
i + µηt

 r−1
X

s=1

∂Lt,s
dyn
∂bi

!

,

(76)

where bt,r = [bt,r
1 , . . . , bt,r
C ] denotes the bias parameters in the output layer of the local model at
local epoch r, and where ∆bτ = [∆bτ
1, . . . , ∆bτ
C] is the local update of the bias at round τ. Since

∂Lt,1
dyn
∂bi
= −µ

t−1
X

τ=1
∆bτ
i ,
(77)

it holds that
∂Lt,2
dyn
∂bi
= −µ

t−1
X

τ=1
∆bτ
i + µηt

 

−µ

t−1
X

τ=1
∆bτ
i

!

= −µ(1 + µηt)

t−1
X

τ=1
∆bτ
i
(78)

and
∂Lt,3
dyn
∂bi
= −µ

t−1
X

τ=1
∆bτ
i + µηt

 

−µ

t−1
X

τ=1
∆bτ
i −(µ + µ2ηt)

t−1
X

τ=1
∆bτ
i

!

= −µ(1 + µηt)2
t−1
X

τ=1
∆bτ
i .

(79)
By induction,
∂Lt,r
dyn
∂bi
= −µ(1 + µηt)r−1
t−1
X

τ=1
∆bτ
i .
(80)

Therefore, the expectation of the local update of bias in the output layer at round t can be computed
as

E
h
∆b(k),t
i
i
=

R
X

r=1
(1 + µηt)r−1µηt

t−1
X

τ=1
E
h
∆b(k),τ
i
i
(81)

=
 
(1 + µηt)R −1
 t−1
X

τ=1
E
h
∆b(k),τ
i
i
.
(82)

Since the objective function of E
h
∆b(k),1
i
i
coincides with that of FedAvg,

E
h
∆b(k),1
i
i
= η1R

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(83)

where η1 is the learning rate at global round t = 1. Then,

E
h
∆b(k),2
i
i
= η1R
 
(1 + µη2)R −1

 

D(k)
i

C
X

c=1
Ec −Ei

!

(84)

= a1a2

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(85)

25


---Page Break---
where a1 = η1R and a2 = (1 + µη2)R −1. Furthermore,

E
h
∆b(k),3
i
i
= a1a3(1 + a2)

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(86)

E
h
∆b(k),4
i
i
= a1a4(1 + a2 + a3 + a2a3)

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(87)

and

E
h
∆b(k),5
i
i
= a1a5(1 + a2 + a3 + a4 + a2a3 + a3a4 + a2a3a4)

 

D(k)
i

C
X

c=1
Ec −Ei

!

.
(88)

By induction,

E
h
∆b(k),t
i
i
=

 

D(k)
i

C
X

c=1
Ec −Ei

!

a1at ·

 

1 +

t−3
X

i=0

t−1
X

τ=2
I(τ + i < t)

τ+i
Y

i=τ
as

!

(89)

= a

 

D(k)
i

C
X

c=1
Ec −Ei

!

,
(90)

where at = (1 + µηt)R −1 and a = a1at

1 + Pt−3
i=0
Pt−1
τ=2 I(τ + i < t) Qτ+i
i=τ as

> 0. After
comparing Eq. (89) with its counterpart in the case of FedAvg, we conclude that the previously
presented analysis of HiCS-FL extends to FedDyn.

A.8.3
Model-Contrastive Federated Learning (Moon)

Moon [16] relies on the objective function with a contrastive term

Lmoon = 1

Bl

l
X

j=1

B
X

n=1
L(j,n)
CE −µ log
exp(sim(z(j,n), z(j,n)
glob )/T)

exp(sim(z(j,n), z(j,n)
glob )/T) + exp(sim(z(j,n), z(j,n)
prev )/T)
, (91)

where z(j,n) denotes the output of the feature extractor of the local model θt
k, z(j,n)
glob is the output of

the feature extractor of the global model θt, and z(j,n)
prev is the output of the feature extractor of the
local model in the previous round θt−1
k
. Since the contrastive term does not depend on the parameters
of bias in the output layer, it holds that

∂L(j,n)
moon
∂bi
= ∂L(j,n)
CE
∂bi
.
(92)

Since the expectation of the local updates of bias in the output layer coincides with the one in case of
FedAvg, previously presented analysis of HiCS-FL extends to Moon.

A.9
Optimization Algorithms Beyond SGD

Optimizers beyond SGD utilize different model update rules which in principle may lead to different
properties of the local update of the bias in the output layer. However, for several variants of SGD,
the properties of the local update of the bias remain such that our presented analysis still applies.

A.9.1
SGD with momentum

In each local epoch r, SGD with momentum updates the model according to

mt,r
k
= µmt,r−1
k
+ (1 −µ)∇Lt,r
CE,
(93)

gt,r
k
= mt,r
k ,
(94)

θt,r
k
= θt,r−1
k
−ηtgt,r
k ,
(95)

26


---Page Break---
where mt,r
k
denotes the momentum in the r-th local epoch, µ is the weight for the momentum, and
mt,1
k
= ∇Lt,1
CE. Then

∆θt
k = −ηt

R
X

r=1
gt,r
k ,
(96)

where
mt,1
k
= ∇Lt,1
CE,
(97)

mt,2
k
= µ∇Lt,1
CE + (1 −µ)∇Lt,2
CE,
(98)

mt,3
k
= µ∇Lt,2
CE + (1 −µ)∇Lt,3
CE
= µ2∇Lt,1
CE + µ(1 −µ)∇Lt,2
CE + (1 −µ)∇Lt,3
CE.
(99)

Therefore,

mt,r
k
= µr−1∇Lt,1
CE + (1 −µ)

r
X

τ=2
µr−τ∇Lt,τ
CE
(100)

and thus we have

∆θt
k = −ηt

 R
X

r=2

 

µr−1∇Lt,1
CE + (1 −µ)

r
X

τ=2
µr−τ∇Lt,τ
CE

!

+ ∇Lt,1
CE

!

.
(101)

Similar to the discussion in the previous section,

E
h
∆b(k)
i
i
= ηt

 R
X

r=2

 

µr−1 + (1 −µ)

r
X

τ=2
µr−τ
!

+ 1

!  

D(k)
i

C
X

c=1
Ec −Ei

!

(102)

= a

 

D(k)
i

C
X

c=1
Ec −Ei

!

(103)

where a = ηt
PR
r=2
 
µr−1 + (1 −µ) Pr
τ=2 µr−τ
+ 1

> 0. Similar result is obtained when
SGD applies Nesterov acceleration as long as the optimizers are not using second-order momentum.

A.9.2
Adam Optimizer

Recall that the two observations regarding the gradient of LCE still hold when training the model
with an adaptive optimizer such as Adam [15]. However, Adam updates the model differently from
SGD. In particular, each entry of the gradient has an adaptive learning rate tied to its magnitude. With
an SGD optimizer, the magnitude of the i-th entry of the local update of bias ∆b(k) is approximately
proportional to the fraction of the samples with label i, D(k)
i
(if Ei is small),

E
h
∆b(k)
i
i
= ηtR

 

D(k)
i

C
X

c=1
Ec −Ei

!

.
(104)

However, this observation does not hold when using the Adam optimizer for the local update because
each entry has a different learning rate ηt,i and thus

E
h
∆b(k)
i
i
= ηt,iR

 

D(k)
i

C
X

c=1
Ec −Ei

!

.
(105)

Although the magnitude of E
h
∆b(k)
i
i
is no longer approximately proportional to D(k)
i
, we can utilize

the sign of E
h
∆b(k)
i
i
, i.e.,

if D(k)
i
≫D(k)
j
, then P

E
h
∆b(k)
i
i
> 0

≫P

E
h
∆b(k)
j
i
> 0

.
(106)

Suppose client k has highly imbalanced data, i.e., H(D(k)) is small. Then the maximal component
maxi D(k)
i
is much larger than the other components; in fact, it is likely to have only one positive

27


---Page Break---
component in the local update of bias ∆b(k). On the contrary, suppose client u has balanced data
and thus H(D(u)) is large. The maximal component maxi D(u)
i
is then very close to the other
components, and it is likely to observe larger number of positive components in the local update of
∆b(u). While characterizing P(E[∆b(k)
i
] > 0) appears challenging, we can empirically infer that
client u with more balanced data has a local update of bias ∆b(u) with more positive components.
With
ˆH(D(u)) ≜H(softmax(∆b(u), T)),
(107)

ˆH(D(k)) ≜H(softmax(∆b(k), T)),
(108)

ˆH(D(u)) is more likely to be larger than ˆH(D(k)). The examples of estimated entropy when utilizing
Adam as the optimizer are provided in Section. A.12.

A.10
Visualization of Data Partitions

To generate non-IID data partitions we follow the strategy in [35], utilizing Dirichlet distribution with
different concentration parameters α to control the heterogeneity levels. In particular, the number of

samples with label i owned by client k is set to
X(k)
i
Ni
PN
j=1 X(j)
i
,where X(1)
i
, . . . , X(N)
i
are drawn from

Dir(α) and Ni denotes the total number of samples with label i in the overall dataset. For the setting
with multiple α, we divide the overall training set into |α| equal parts and generate data partitions
according to the method above. Figures 6 and 7 illustrate the class distribution of local clients by
displaying the number of samples with different labels; colors distinguish between magnitudes – the
darker the color, the more samples are in the class.

(a)
(b)
(c)

Figure 6: Results on CIFAR10. Training data is split into 50 partitions according to a Dirichlet distri-
bution (50 clients). The concentration parameter is as follows: (1) α ∈{0.001, 0.01, 0.1, 0.5, 1.0};
(2) α ∈{0.001, 0.002, 0.005, 0.01, 0.5}; (3) α ∈{0.001, 0.002, 0.005, 0.01, 0.1}. The figures (a),
(b) and (c) correspond to settings (1), (2) and (3), respectively.

(a)
(b)

Figure 7: Results on Mini-ImageNet.
Training data is split into 100 partitions according to
Dirichlet distribution (100 clients). The concentration parameter is varied as follows: (1) α ∈
{0.001, 0.01, 0.1, 0.5, 1.0}; (2) α ∈{0.001, 0.005, 0.01, 0.1, 1.0}. The figures (a) and (b) corre-
spond to settings (1) and (2), respectively.

28


---Page Break---
Table 5: The columns “Extra Computation” and “Extra Communication” denote the computation and
communication complexity of additional operations in each sampling scheme compared to random
sampling.

Method
Extra Computation
Extra Communication
Random
-
-
pow-d
O(|θt|)
O(|θt|)
CS
O(|θt|)
-
DivFL
O(|θt|)
O(|θt|)
FedCor
O(|θt|)
-
HiCS-FL
O(C)
-

A.11
Computational and Communication Complexity

We compare the communication and computational costs of HiCS-FL with those of the competing
methods, including Power of Choice (pow-d) [8], Clustered Sampling [11] and DivFL [2], and map
them against random sampling, as shown in Table. 5. In its ideal setting, pow-d selects K clients
with the largest local validation loss among all N clients. To compute the local validation loss
at the beginning of a global training round t, the server must send the global model to all clients.
Compared to the random sampling strategy where the global model is sent to only K clients, pow-d
must transmit additional (N −K)|θt| model parameters. Moreover, pow-d requires all clients to
compute validation loss of the global model θt on local datasets, which incurs additional O(N|θt|)
computations. While communication requirements of Clustered Sampling do not exceed those of
random sampling, the server must run a clustering algorithm on the local updates of dimension |θt|
(the same as gradients). DivFL relies on maximizing a submodular function to select the most diverse
clients based on all clients’ gradients, leading to a transmission overhead and additional computation
involving |θt|-dimensional gradients. In our experiments, DivFL has consistently required the longest
training time and memory usage due to its dependence on the submodularity maximizer. FedCor [28]
cliams that only partial clients participating in the global update after warm-up stage but still needs
all clients to perform inference for computing validation loss in the warm-up stage. Our proposed
method, HiCS-FL, does not require any additional transmission of model parameters; furthermore,
in HiCS-FL the server clusters clients based on their local updates of the bias in the output layer,
which is low-dimensional and model-agnostic. Overall, HiCS-FL requires negligible computational
overhead to significantly improve the performance of non-iid Federated Learning.

A.12
Examples of Estimated Entropy

To further illustrate the proposed framework, here we show a comparison between the estimated
entropy of data label distribution and the true entropy. Specifically, Figures 8 and 9 show that
the entropy estimated by the proposed method is close to the true entropy; the experiments were
conducted on FMNIST and Mini-ImageNet, using SGD and Adam as optimizers, respectively. As
stated in Theorem 3.3, the clients with larger true entropy are likely to have lager estimated entropy.
In case where the model is trained with Adam, estimated entropy of data label distribution is not
as accurate as in the case of using SGD. Figures 10 and 11 compare the performance of estimating
entropy with SGD and Adam optimizers for the same setting of α. Notably, as shown in the figures,
the method is capable of distinguishing clients with extremely imbalanced data from those with
balanced data.

29


---Page Break---
(a)
(b)

Figure 8: The estimated entropy of data label distribution in experiments on FMNIST with SGD
as the optimizer. The parameter α for the two figures: (a) α ∈{0.01, 0.02, 0.05, 0.1, 0.2}; (b)
α ∈{0.001, 0.002, 0.005, 0.01, 0.5}

(a)
(b)

Figure 9: The estimated entropy of data label distribution in experiments on Mini-ImageNet with
Adam as the optimizer. The parameter α for the two figures: (a) α ∈{0.001, 0.01, 0.1, 0.5, 1.0}; (b)
α ∈{0.001, 0.005, 0.01, 0.1, 1.0}.

(a)
(b)

Figure 10: The estimated entropy of data label distribution in experiments on CIFAR10 with α ∈
{0.001, 0.01, 0.1, 0.5, 1.0}. (a) The result of the experiments using SGD as the optimizer. (b) The
result of the experiments using Adam as the optimizer.

30


---Page Break---
(a)
(b)

Figure 11: The estimated entropy of data label distribution in experiments on CIFAR10 with α ∈
{0.001, 0.002, 0.005, 0.01, 0.5}. (a) The result of the experiments using SGD as the optimizer. (b)
The result of the experiments using Adam as the optimizer.

31


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: [NA]

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

Justification: [NA]

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

32


---Page Break---
Justification: [NA]
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
Justification: [NA]
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

33


---Page Break---
Answer: [Yes]
Justification: [NA]
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
Justification: [NA]
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
Justification: [NA]
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

34


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
puter resources (type of compute workers, memory, time of exNoecution) needed to repro-
duce the experiments?
Answer: [No]
Justification: [NA]
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
Justification: [NA]
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: [NA]
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

35


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
Justification: [NA]
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
Justification: [NA]
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

36


---Page Break---
Answer: [NA]
Justification: [NA]
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
Justification: [NA]
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
Justification: [NA]
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

37


---Page Break---
