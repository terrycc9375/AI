Performative Control for Linear Dynamical Systems

Songfu Cai∗,
Fei Han∗,
Xuanyu Cao†
Department of Electronic and Computer Engineering
The Hong Kong University of Science and Technology
eesfcai@ust.hk, fhanac@connect.ust.hk, eexcao@ust.hk

Abstract

We introduce the framework of performative control, where the policy chosen
by the controller affects the underlying dynamics of the control system. This
results in a sequence of policy-dependent system state data with policy-dependent
temporal correlations. Following the recent literature on performative prediction
[21], we introduce the concept of a performatively stable control (PSC) solution.
We first propose a sufficient condition for the performative control problem to
admit a unique PSC solution with a problem-specific structure of distributional
sensitivity propagation and aggregation. We further analyze the impacts of system
stability on the existence of the PSC solution. Specifically, for almost surely
strongly stable policy-dependent dynamics, the PSC solution exists if the sum of
the distributional sensitivities is small enough. However, for almost surely unstable
policy-dependent dynamics, the existence of the PSC solution will necessitate a
temporally backward decaying of the distributional sensitivities. We finally provide
a repeated stochastic gradient descent scheme that converges to the PSC solution
and analyze its non-asymptotic convergence rate. Numerical results validate our
theoretical analysis.

1
Introduction

Control theory is a fundamental field of study in engineering and mathematics that centers on
coordinating the behaviors of dynamical systems. It has extensive applications in aerospace, robotics,
manufacturing, economics, natural sciences, etc. It provides a framework for designing control
policies that regulate the system states, enabling them to evolve in a desired manner over time
dynamically. The linear state space model is a powerful tool for representing dynamical systems,
which employs a set of difference equations to describe the Markovian system state process with
a linear state transition model. Many of the existing works on control of linear dynamical systems
(LDSs) [26, 9, 31, 7, 17, 28, 31, 32] are developed based on the key assumption that the system
state transition model is static. However, in many real world applications, such a static dynamics
assumption usually does not hold because system state transition model can be changed by control
policies. For instance, in the stock market, the investment policy of well-known investors can impact
the actions of the general public investors, resulting in famed investment strategy-dependent changes
to the dynamics of stock prices. Another example is autonomous vehicle (AV). A deployed AV might
change how the pedestrians and other neighboring cars behave, and the resulting traffic environment
might be quite different from what the designers of the AV had in mind [20]. Such interplay between
decision-making and decision-dependent system dynamics is a pervasive phenomenon in a multitude
of domains, including finance, transportation, and public policy, among others.

Most of the existing works on the control of linear systems with non-static state transition model
are primarily focused on the additive random perturbations to the system state transition matrix,

∗Equal contribution.
†Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
where policy optimization methods are employed to find the optimal control policy that minimizes
quadratic costs [10, 1, 11, 8]. This type of problem also includes the control of linear systems with
additive state-dependent noise [15, 6] or action-dependent noise [30, 4], which is equivalent to having
additive perturbations on the state transition matrix or the control input gain matrix, respectively.
Some other works on jump/switched linear systems formulate the variation of system model as
stochastic model jumps among multiple linear modes, where the jumping law is governed by a finite
time-homogeneous Markov process. However, in all these works, the changes of the system model
are unrelated to the control policy.

Performative prediction provides a systematic way to model the interaction between decision-making
and data via decision-dependent data distribution maps [23]. The pioneering work by [21] has led to
a growing body of research dedicated to performative prediction problems. Most of these studies are
focused on establishing the conditions for the existence and uniqueness of the performative stable
point and designing learning algorithms with provable convergence to such a unique performative
stable point [13, 14, 5, 3, 18, 29, 25], or algorithms to find a stationary solution of the performative
risk [12, 19, 25].

An open question follows: Can the idea of performative prediction, which tackles decision-dependent
data in the learning/prediction domain, be employed to address the decision-dependent dynamics in
the control domain? Notice that in all the aforementioned performative prediction works [21, 13, 14,
5, 3, 18, 29, 25, 12, 19, 25], the data input to the learning algorithms are independently generated
based on the decision-dependent distributions. No temporal correlation is considered or exploited
between two consecutive sets of input data. However, in the context of control, the situation is
different as the changing system dynamics introduce various additional complexities. This is because
the data of the system state at each time the step will depend on the decision-dependent state transition
matrices in all previous time steps. The expected total cost function to be optimized will depend
on all the decision-dependent state transition matrices across the entire control time horizon. This
implies that we need a framework of performative control that is more general than the framework
of performative prediction, and that can accommodate a sequence of decision-dependent data with
temporal correlations, where the temporal correlations are again decision-dependent.

In this work, we provide an affirmative answer to the above question. Our idea hinges on adopting the
concept of performative stable solution [21], which is a fixed point solution for the interplay between
the controller and the system dynamics that react to the controller’s decisions. The condition for
the existence of such a performative stable solution is obtained by analyzing the propagation and
aggregation of sensitivities associated with the distributions of policy-dependent system dynamics
over the entire control time horizon. In particular, we follow existing studies [1, 11, 8] and focus on
the disturbance-action policy, which allows the consideration of general convex control costs instead
of only quadratic control costs.

To the best of our knowledge, this paper provides the first study and analysis of performative control
of linear systems where the system dynamics are constantly changing in a control-policy-dependent
manner. We highlight the following key contributions:

• We introduce the notion of performative control, where the deployed control policy affects
the underlying dynamics of the control system. We provide sufficient conditions for the
existence and uniqueness of the performative stable control (PSC) solution. The sufficient
condition is expressed in terms of a weighted sum of all the distributional sensitivities
associated with the policy-dependent system state transition matrices over the entire control
horizon. An interesting finding is that the sufficient condition exhibits a structure of
sensitivity propagation and aggregation, implying that it is preferable for sensitivities to be
relatively small in the early stages of the system state evolution.
• We analyze the impacts of system stability on the existence and uniqueness of the PSC
solution. We show that when the policy-dependent dynamics are almost surely strongly
stable, the PSC solution exists if the sum of all distributional sensitivities is below a certain
threshold. On the other hand, when the policy-dependent dynamics are almost surely
unstable, the proposed sufficient condition for the existence of the PSC solution will place a
necessary requirement on a temporally backwards decaying of the distributional sensitivities.
• We propose a repeated stochastic gradient descent (RSGD) scheme and analyze its conver-
gence towards the PSC solution. We show that the scheme is convergent under the same
sufficient condition for the existence of the PSC solution. With an appropriate step size rule,

2


---Page Break---
the expected squared distance between the PSC solution and the iterates decays as O
  1

N

,
where N is the iteration number.

Finally, we conduct experiments on policy-dependent stock investment risk minimization problem.
The numerical results validate the effectiveness of our algorithm and theoretical analysis.

1.1
Related Work

Performative Prediction. The notion of performative prediction is initiated by [21], where per-
formative stability is first introduced, and a sufficient and necessary condition is provided so that
the performative stable point can be reached via iterative risk minimization algorithms. Since then,
there has been a growing literature analyzing the performative prediction problem and studying the
convergence of learning algorithms to the performative stable point [18, 5, 3, 13, 29].There are also
some other papers that find algorithms that converge to a stationary solution of the performative
risk [12, 19, 25]. Our problem poses entirely new challenges because the decision-dependent data
of system state and control costs also have decision-dependent temporal correlations. Such a two-
layer decision-dependent structure cannot be incorporated into the existing performative prediction
framework.

There are some very recent works on performative reinforcement learning, where a Markov decision
process (MDP) is considered and the deployed policy not only changes the costs but also the
underlying state transition kernel [16, 24]. However, these works only considered the tabular MDP,
where the state and action space are finite. The LDS considered in our work may also be viewed as a
Markov decision process with a decision-dependent linear transition kernel and decision-dependent
costs. But both the state and action space are continuous and there are infinitely many admissible
state-action pairs, which are beyond the scopes of [16, 24].

Stateful Performative Prediction. Note that most of the existing performative prediction works
[21, 18, 18, 5, 13, 29, 5, 3, 13, 29] are non-stateful in the sense that, for any deployed policy
θ, the data sample Z follows a static distribution D (θ) , i.e., Z ∼D (θ) . The seminal work [3]
generalizes the stateless framework of [21] and proposes a more general framework of stateful
performative prediction via a stateful distribution transition map f (·). Specifically, the observed
data distribution in [3] is time-varying and depends on the history of previously deployed polices,
i.e., Dt+1 = f (Dt, θt). Consequently, in comparison to the conventional non-stateful performative
prediction works [21, 18, 18, 5, 13, 29, 5, 3, 13, 29], this framework is capable of encapsulating the
phenomenon of strategic decision-making with outdated information, and serves as a foundation for
investigations of the disparate effects of performativity (please refer to Examples 1-3 in [3] for more
details). The interconnections and generalizations between our work and the stateful performative
prediction will be substantiated in Section 2.

Nonstochastic Control. Another line of relevant works pertains to non-stochastic control, which
is initiated by [1]. Various applications of non-stochastic control can be found in [10, 1, 11, 8]. At
the core of non-stochastic control is the disturbance-action control policy, which chooses the action
as a linear map of the past disturbances [11]. Such a disturbance-action policy facilitates efficient
algorithms for control problems with arbitrary additive disturbances in the dynamics and arbitrary
convex control costs instead of quadratic costs only. In this paper, we adopt the disturbance-action
control policy for analysis. However, in contrast to [1, 10, 1, 11, 8], where the system dynamics are
static, our analysis involves policy-dependent nonstationary dynamics.

2
Motivations

In this section, we discuss the key motivations for our performative linear control framework. We
also outline the key connections and generalizations of our proposed framework to the stateful
performative prediction framework of [3].

Connections between Static Stateful Distribution Transition Maps and LDS. The LDS modeling
via control theory in our work is closely aligned with the static stateful distribution maps within
the state performative prediction framework [3]. Let us first consider a static stateful distribution
transition map f (·) with Dt+1 = f (Dt, θt) , ∀t ≥0. To facilitate the analysis, we use a performative
data sample Zt ∼Dt to equivalently characterize the performative distribution Dt. If the properties of
f (·) are nice enough, then from the perspective of random variables, there will exist a corresponding

3


---Page Break---
mapping ef (·) such that

Zt+1
d= ef (Zt, θt) , Zt+1 ∼Dt+1, Zt ∼Dt, ∀t ≥0,
(1)

where
d= denotes equal in distribution. Linearizing ef (·) at some equilibrium point
 

Z, θ

with Z ∼D
and ignoring the higher order terms will lead to

Zt+1
d= AZt + Bθt + w,
(2)

where A =
h
∂e
f
∂d
i

(d,θ) and B =
h
∂e
f
∂θ
i

(d,θ) are the static Jacobin matrices at the equilibrium point
 

d, θ

and w = ef
 

Z, θ

−AZ −Bθ. Consider the performative data sample Zt, the deployed
policy θt and the residual w in (2) as the state variable xt, the control input action ut and the additive
system noise wt ∼w of an LDS, respectively. The linearized static stateful distribution map model
in (2) is thus precisely connected to a LDS with system dynamics given by
xt+1 = Axt + But + wt.
(3)

Extension to Performative Stateful Distribution Transition Maps via Performative LDS. Further
consider a more complicated case of performative distribution transition maps fθt (·) with Dt+1 =
fθt (Dt, θt) , ∀t ≥0, where the specific form of the distribution transition map fθt is time-varying
and depends on the delployed policy θt. Employing a similar linearization for efθt (·) and neglecting
the higher order terms, we obtain

Zt+1
d= AθtZt + Bθtθt + wθt,
(4)

where Aθt =
h ∂e
fθt
∂d
i

(d,θ) and Bθt =
h ∂e
fθt
∂θ
i

(d,θ) are the performative Jacobin matrices, and

wt = efθt
 

Z, θ

−AθtZ −Bθtθ. This results in an equivalent performative LDS given by

xt+1 = Autxt + Butut + wut.
(5)
In contrast to (3), where the system dyanmics are static, the state transition matrix Aut, control
input gain matrix But and additive noise wut in (5) are all performative and depend on the deployed
control policy ut.

In conclusion, the LDS modeling presented in our paper allows for the performative transition maps
of performative distributions, thereby extending the technical results previously obtained for fixed
performative distribution transition maps in the stateful performative prediction work [3]. As a result,
our proposed performative LDS framework has the potential to enhance the understanding of general
performative prediction.

3
Problem Setup

We consider the control of a linear dynamic system with per stage cost ct (xt, ut). A control policy
π is a mapping π : Rdx×1 →Rdu×1, which maps the system state xt to the control action ut, i.e.,
ut = π (xt). For each control policy π, we attribute a finite time horizon expected cost defined as

Cπ
T = Ex0,{At},{wt}

" T
X

t=0
ct (xt, ut)

#

,
(6)

where xt+1 = Atxt + But + wt, the initial system state x0 follows a general distribution of
Dx0 with bounded support ∥x0∥≤x0, and Ex0,{At},{wt} represents the expectation over x0, the
entire policy-dependent state transition matrix sequence {At, 1 ≤t ≤T} and the entire disturbance
sequence {wt, 1 ≤t < T}.

To the best of our knowledge, this paper is the very first work to investigate the performative LDS,
and we intend to provide a thorough theoretical investigation and aim at establishing various new
theoretical results. Therefore, we opt to construct our performative LDS theoretical framework upon
the performative state transition matrix At, while maintaining the control input gain matrix B and
additive disturbance wt as non-performative. This facilitates us to streamline the theoretical analysis
and consolidates the system design insights.

We have the following assumption on the additive disturbances {wt, 1 ≤t ≤T} .

4


---Page Break---
A1 The additive disturbance per time step wt is bounded, i.i.d, and zero-mean with a lower
bounded covariance i.e., wt ∼Dw, E[wt] = 0, E[wtw⊤
t ] ⪰σ2I, ∥wt∥≤W, ∀0 ≤t < T,
and E[wt1w⊤
t2] = 0, ∀0 ≤t1 ̸= t2 < T.

Disturbance-Action Control Policy. We work with the following class of disturbance-action control
policy throughout this paper, which is commonly used in nonstochastic control [10, 1, 11, 8] to
address general convex control cost functions.

Definition 1 (Disturbance-Action Policy). For a disturbance-action control policy, the mapping π,
∀0 ≤t < T, is uniquely characterized by a set of matrices

M(1), · · · , M(H)	
. At every time step t,
such a disturbance-action control policy assigns a control action u(M)
t
in the form of

u(M)
t
= π (xt) = −Kxt +

H
X

i=1
M(i)wt−i = −Kxt + M [w]H
t−1 ,
(7)

where M =

M(1), M(2), · · · , M(H)
belongs to a convex set M with bounded support ∥M∥F ≤M,

[w]H
t−1 is short for

w⊤
t−1, · · · w⊤
t−1−H
⊤, H < T is a constant and wi = 0 for all i < 0.

In the disturbance-action policy (7), we adopt a class of linear controller K defined as follows.

Definition 2 (Strongly Stabilizing Linear Controller [1, 10, 11, 8]). Given A and B, a linear
controller K is (κ, γ) almost surely strongly stable for real numbers κ ≥1, γ < 1, if ∥K∥≤κ,
and there exists matrices Q and L such that eA := A −BK := QLQ−1, with ∥L∥≤1 −γ and
∥Q∥,
Q−1 ≤κ.

It is worth noting that the disturbance-action policy is only parameterized by the matrix M. Whereas
the state feedback gain K, which is a fixed matrix, is not part of the parameterization of the policy.
As pointed out in [2], a typical choice of the parameter H is H = γ−1 log(Tκ2). With an appropriate
choice of the policy M, the control action u(M)
t
in (7) is capable of approximating any linear state
feedback control policy in terms of the total cost suffered with a finite time horizon of H [10, 1, 11, 8].

Policy-dependent Dynamics. Without loss of generality, at any time step t, the impact of the
disturbance-action control policy u(M)
t
to the dynamics of the linear system is modeled as a policy-
dependent additive perturbation ∆t to a common state transition matrix A.

A2 (Policy-dependent State Transition Matrix). The disturbance-action policy-dependent state
transition matrix At takes the form of

At = A + ∆t, ∆t ∼Dt (M) , ∀0 ≤t < T,
(8)

where A is the mean value of At, and ∆t is the policy-dependent state transition perturbation with
zero mean and bounded support, i.e., E [∆t]=0 and there exists a bounded constant ξt such that
∥∆t∥≤ξt, ∀0 ≤t < T. For different time steps t1 and t2, ∆t1 and ∆t2 are mutually independent.
Besides, {∆t, 0 ≤t < T} and {wt, 0 ≤t < T} are mutually independent.

Remark 1 (Non-zero Mean ∆t). If the mean of the disturbance is non-zero and time-varying, i.e.,
E [∆t] = Θt, we will have a new equivalent A′
t = A + Θt with zero mean disturbances. We only
need to choose a new linear controller K′
t such that A′
t −BK′
t is (κt, γt)-strongly stabilizing. As a
result, without loss of generality, we assume ∆t is zero mean throughout this paper.

Remark 2 (Policy-dependent Control Input Gain Matrix B). Note that under the DAP (7), the
disturbance-action is given by BM[w]H
t−1, which is linear in policy M. Such kind of linear
disturbance-action control policy has various nice theoretical performance guarantees as sub-
stantiated in [10, 1, 11, 8]. On the other hand, if B is also performative, we will have a generalized
disturbance-action Bt(M)M[w]H
t−1, which can be possibly nonlinear in policy M. Such kind of
generalized nonlinear disturbance-action control policy has received very few research attention,
and it can serve as a very interesting future research direction.

5


---Page Break---
We make the following sensitivity assumption on the distributions {Dt (M) , 0 ≤t < T} .

A3 (ε-Sensitivity). For any t = 0, 1, · · · , T −1, there exists a constant εt > 0 such that

W1 (Dt (M) , Dt (M′)) ≤εt ∥M −M′∥F , ∀M, M′ ∈M,
(9)

where W1 (D, D′) denotes the Wasserstein-1 distance between the distributions D and D′.

Assumption A3 imposes a regularity requirement on the distributions {Dt (M) , 0 ≤t < T} . Intu-
itively, if the disturbance-action control policies are made according to similar policy parameteriza-
tions M, then the resulting distributions of the policy-dependent state transition perturbations should
also be similar.

System State Evolutions. Under the disturbance-action policy (7), the following lemma shows that
the system state xt, ∀1 ≤t ≤T, can be uniquely determined by x0, eA, M, {∆t, 0 ≤t < T} and
{wt, −H ≤t < T} .

Lemma 1 Given a disturbance-action policy M, the system state xt, ∀1 ≤t ≤T, can be represented
as

xt = x(M)
t
=

t−1
Y

i=0


eA + ∆i

x0 +

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

BM [w]H
i−1
(10)

+

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

wi.

Besides, let the K in DAP (7) be a strongly stabilizing linear controller, the norm of xt is upper
bounded as ∥xt∥≤x0κ2αt + κ2W (∥B∥HM + 1) βt, where αt = Qt−1
i=0
 
1 −γ + κ2ξi

and
βt = Pt−1
i=0
Qt−1
j=i+1
 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

.

The performative optimal control (POC) problem can therefore be formulated as:

min
M∈M
CM
T = Ex0,{∆t ∼Dt (M)},{wt}

" T
X

t=0
ct

x(M)
t
, u(M)
t
#

.
(11)

The POC problem (11) comprises of a stochastic objective function with policy-dependent distri-
butions. Due to non-convexity, the performative optimal solution MP O to (11) is usually difficult
to obtain. Alternatively, in this paper, we are interested in the performative stable control (PSC)
solution:

MP S = Φ(MP S) := arg min
M∈M Ex0,{∆t∼Dt(MP S)},{wt}

" T
X

t=0
ct

x(M)
t
, u(M)
t
#

.
(12)

Notice that MP S is defined to be a fixed point of the map Φ. Compared to the POC (11), the
distribution of ∆t in PSC (12) changes from ∆t ∼Dt (M) to ∆t ∼Dt
 
MP S
, ∀0 ≤t < T. The
existence and uniqueness of MP S will be dicsussed in Lemma 4.

Comparison to Existing Works. Most of the existing performative prediction works [21, 13,
14, 5, 3, 18, 29, 25, 12, 19, 25] consider the cost in the form of l (θ; Z) , where θ is the deci-
sion variable. Then the relationship between data samples Z and decision θ is parameterized
by a fixed distribution Z ∼D (θ) with fixed sensitivity ε. However, in our work, the relation-
ship between system state data samples xt and policy M is characterized by a time-varying dis-
tribution xt ∼ft (D0 (M) , · · · , Dt−1 (M)) with a time-varying sequence of joint sensitivities
{ε0, · · · , εt} , ∀0 ≤t < T. The total cost PT
t=0 ct depends on all the policy-dependent distributions
{D0 (M) , · · · , DT −1 (M)} with a collection of sensitivities {ε0, · · · , εT −1}. These key differences
lead to a more complicated analysis in our work.

RSGD Scheme. We propose a repeated stochastic gradient descent (RSGD) scheme in Algorithm
1 to find a PSC solution to (12). The metric of the projection in line 9 of Algorithm 1 is the

6


---Page Break---
Algorithm 1 Repeated Stochastic Gradient Descent (RSGD)
Input: Step sizes {ηn, 0 ≤n ≤N}, parameters K, H. Define M = {M : ∥M∥≤M} . Initialize
M0 ∈M arbitrarily.

1: for n = 0, · · · , N, do
2:
Initialize ∇JT = 0.
3:
for t = 0, · · · , T −1, do
4:
Use control ut = −Kxt + Mn [w]H
t−1 .
5:
Observe At, xt+1; compute noise wt = xt+1 −Atxt −But.
6:
Compute the gradient ∇Mnct (xt, ut) and update ∇JT ←∇JT + ∇Mnct (xt, ut) .
7:
end for
8:
Update Mn+1 ←ProjM{Mn −ηn∇JT }.
9: end for

matrix Frobenius norm. Specifically, ProjM {M} = arg minM′∈M ∥M −M′∥2
F . Note that such a
projection is computationally tractable because the Frobenius norm square minimization is a convex
optimization problem. The RSGD scheme first computes the stochastic gradient of the total cost w.r.t.
policy PT
t=0 ∇Mnct (xt, ut) and then perform stochastic gradient descent on M. The detailed steps
for computation of the stochastic gradient ∇Mnct (xt, ut) are provided in Appendix B.

4
Main Results

This section investigates the existence of a PSC solution MP S to (12) and the convergence of the
RSGD scheme to MP S. We require the following assumptions on the per stage cost ct [1, 10, 1, 11, 8].

A4 (Strongly Convex). The per stage cost function ct (x, u) is µ-strongly convex such that

ct (x1, u1) ≥ct (x2, u2) + ∇⊤
x ct (x2, u2) (x1 −x2) + ∇⊤
u ct (x2, u2) (u1 −u2)

+ µ

2


∥x1 −x2∥2 + ∥u1 −u2∥2
, ∀x1, x2 ∈Rdx, u1, u2 ∈Rdu.

A5 (Smoothness). The per stage cost function ct (x, u) is ς smooth such that
∥∇xct (x1, u1) −∇xct (x2, u1)∥+ ∥∇uct (x1, u1) −∇uct (x1, u2)∥

≤ς (∥x1 −x2∥+ ∥u1 −u2∥) , ∀x1, x2 ∈Rdx, u1, u2 ∈Rdu.

A6 (Boundedness). There exists a positive constant G such that
∥∇xct (x, u)∥, ∥∇uct (x, u)∥≤GD, ∀∥x∥, ∥u∥≤D.

The above Assumptions A4-A6 on the per stage cost function ct (x, u) are quite standard, which hold
for broad classes of costs such as the quadratic costs.

To facilitate our discussions, we define an expected total cost with distribution shift as

CT (M; M′) := Ex0,{∆t∼Dt(M′)},{wt}

" T
X

t=0
ct

x(M)
t
, u(M)
t
#

,

where, under policy M, the distribution of policy-dependent perturbation is changed from ∆t ∼
Dt (M) to ∆t ∼Dt (M′), ∀0 ≤t < T. For the rest of this paper, unless otherwisespecified,
∇CT (M; M′) denote the gradients taken w.r.t. the first argument M.

Our main results rely on the strong convexity of the expected total cost CT (M; M′) with respect to
its first argument M. However, the strong convexity of the per stage cost function ct (x, u) over the
state-action space in Assumption A4 does not by itself imply the strong convexity of the expected
total cost CT (M; M′) over the space of policies M. This is becasue the policy M, which maps from
a space of dimensionality H × dx × du to that of dx + du, is not necessarily full column-rank. Our
next lemma, which forms the core of our analysis, shows that this is not the case using the inherent
stochastic nature of the policy-dependent dynamics.

7


---Page Break---
Lemma 2 Under A1-A6, fix any M′ ∈M, the expected total cost CT (M; M′) is eµ-strongly convex
in its first argument M such that ∀M1, M2 ∈M,

CT (M1; M′) ≥CT (M2; M′) + Tr

(∇CT (M2; M′))⊤(M1 −M2)

+ eµ

2 ∥M1 −M2∥2
F ,

(13)

where eµ = min
n
(T −H+1)µσ2

2
, (T −H+1)µσ2γ2

64κ10
o
.

For a concise presentation of the smoothness of expected total control cost, we next define a collection
of constants as follows:

c1 := dxςH
3
2 W
 
1 +
 
κ2 + κ3
∥B∥
  
κ2 + κ3
(1 −γ)−1 ,

c2 := dxH
3
2 WG (1 −γ)−1  
κ4 + κ5
∥B∥, c3 := (HM ∥B∥+ 1) W, c4 := HW (1 −γ) c1,

c5 := HW
 
κ2 + κ3−1 (1 −γ) c1.

The smoothness of the expected total cost CT (M; M′) is summarized below.

Lemma 3 Under A1-A6, the expected total cost CT is smooth in the sense that, for any
M, M′, M1, M2 ∈M and ∀1 ≤t ≤T, the following inequality holds

∥∇CT (M1; M) −∇CT (M2; M′)∥F

≤

T
X

t=1
λt ∥M1 −M2∥F +

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!

∥M −M′∥F ,
(14)

where λt = c1 (c4βt + c5) , νt = (c1 + c2βt) (x0αt + c3βt) , ∀1 ≤t ≤T, and we recall that
αt = Qt−1
i=0
 
1 −γ + κ2ξi

and βt = Pt−1
i=0
Qt−1
j=i+1
 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

from Lemma 1,
which characterize the growth of the norm of system state ∥xt∥.

Existence and Uniqueness of MP S. Our first main result establishes a sufficient condition for the
existence and uniqueness of the performative stable policy MP S that solves the PSC problem (12).

Lemma 4 Under A1-A6, consider the fixed-point iteration

Mn+1 =Φ (Mn) , ∀n ≥0,
(15)

where the map Φ is defined in (12). If the following condition is satisfied

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!

< eµ,
(16)

then iterates Mn converge to a unique performatively stable point MP S at a linear rate, i.e.,

Mn −MP S
F ≤ρ for n ≥



1 −

PT −1
t=0

εt
PT
i=t+1 νi


eµ





−1

log
1

ρ

M0 −MP S
F


.

The sufficient condition (16) delivers a fact that that the existence and uniqueness of MP S is
jointly determined by all the sensitivities {εt, 0 ≤t < T} in the temporal domain. The sensitivity
εt at the t-th time step is propagated starting from time step t + 1 to the last time step T with a
sequence of weights {νt+1, · · · , νT }. The aggregated impact of all the policy-dependent disturbances
{∆t, 0 ≤t < T} is captured by total sum in L.H.S. of (16). This is very different from the existing
performative prediction works [21, 18, 5, 3, 13, 29, 12, 19, 25], where only one distribution D and
one sensitivity ε are involved.

The sufficient condition (16) also implies that it is preferable for the initial policy-dependent distur-
bance ∆0 to be small because it propagates and aggregates for the longest time steps of T.

8


---Page Break---
Impacts of System Stability. The sufficient condition (16) also reveals the impacts of system stability
on the existence and uniqueness of the performative stable solution MP S, which are summarized
below.

Proposition 1 (Almost Surely Strongly Stable Case) Let A1-A6 hold. If policy-dependent state
transition matrix At is
 
κ, γ −κ2ξt

-strongly stable for real numbers κ ≥1, γ −κ2ξt < 1, ∀0 ≤
t < T, almost surely. Let ζ = max

1 −γ + κ2ξt, ∀0 ≤t < T
	
, then MP S exists and is unique if
PT −1
t=0 εt < ϕ
 
1 −H

T


µ, where µ = min
n
µσ2

2 , µσ2γ2

64κ10
o
and ϕ is some positive constant.

Proposition 1 points out that when the policy-dependent dynamics are almost surely strongly stable,
we only need to make sure that the sum of all the sensitivities {εt, 0 ≤t < T} is below a certain
threshold.

Proposition 2 (Almost Surely Unstable Case) Let A1-A6 hold. If the policy-dependent state transition
matrix At is almost surely unstable, i.e., there exists a positive constant eζ > 1 such that eζ ≤∥At∥≤
1 −γ + κ2ξt, ∀0 ≤t < T. In this case, to guarantee the sufficient condition (16) can be satifised, we
must have εt < ϕ(T −H+1)µ

eζT −t−1
, ∀0 ≤t < T, where ϕ is some positive constant.

For unstable policy-dependent dynamics, condition (16) will impose a necessary requirement on the
temporally backwards decaying of the sensitivities. Particularly, more restrictive requirements are
placed on the the sensitivies in the early time steps, i.e., smallt, which should decay exponentially
fast w.r.t. the control time horzion T.

For more general applications, where At can be either stable or unstable for different time steps, the
weighted sum requirement of the sensitivities {εt, 0 ≤t < T} in (16) is sufficient to guarantee the
existence and uniqueness of the performative stable solution MP S.

Convergence of RSGD Scheme. Our next theorem establishes the convergence rate of the proposed
RSGD algorithm.

Theorem 1 Choose two positive constants ϕ1 > 0 and ϕ2 ≥1 such that the following two
conditions are satisfied simualtneously

ϕ1
ϕ2
≤min








eµ −PT −1
t=0

εt
PT
i=t+1 νi


2
PT
t=1 λt + PT −1
t=0

εt
PT
i=t+1 νi
2 ,
1

eµ −PT −1
t=0

εt
PT
i=t+1 νi








,
(17)

ϕ1
1 +
1
ϕ2
≥
2

eµ −PT −1
t=0

εt
PT
i=t+1 νi
.
(18)

Consider a sequence of non-negative step sizes
n
ηn =
ϕ1
n+ϕ2 , n ≥0
o
. Then, the iterates generated
by RSGD admit the following bound for any N ≥1:

E
hMN −MP S2
F

i
≤e−PN
n=1
ϕ1

n (eµ−PT −1
t=0 (εt
PT
i=t+1 νi))E
hM0 −MP S2
F

i
+ ϕ3

N , (19)

where ϑt = κ3G
  
HW + κ2
κ ∥B∥βt + 1

(x0αt + c3βt) + GHWM
 
κ3βt + 1

, ∀1 ≤t ≤

T, and ϕ3 =
4ϕ1T PT
t=1 ϑ2
t
eµ−PT −1
t=0 (εt
PT
i=t+1 νi) is a positive constant.

Note that, under the sufficient condition (16) in Lemma 4, there always exists a pair of (ϕ1, ϕ2) that
satisfies (17) and (18) simultaneously by letting ϕ2 be sufficiently large. The first term on the R. H. S.
of (19) decays at the rate of O(e−PN
n=1
ϕ1

n (eµ−PT −1
t=0 (εt
PT
i=t+1 νi))) and is scaled by the initial error

E
hM0 −MP S2
F

i
. The second term is a fluctuation term that only depends on the variance of the
stochastic gradient, which decays at the rate O (1/N). For more general types of step sizes and the
associated nonasymptotic convergence rate analysis, please refer to Lemma 5 in the appendix G.1.

9


---Page Break---
Figure 1: PS Error
MN −MP S2

F

Figure 2: Expected Total Control Cost CT (MN; MN)

5
Numerical Experiments

We consider an application of stock investment risk minimization problem to verify our algorithm
and theoretical results. Consider an investor trading a total number of 10 stocks over a period
of T = 60 trading days. The detailed system setups are described in Example 1 in Appendix A
of the supplementary. We compare the performative error (PS error)
MN −MP S2
F and the
expected total cost CT (MN; MN) against the iteration number N, respectively. We consider a fixed
distributional sensitivity value set ε = {εi, i = 0, 1, . . . , T −1}. We assign three different patterns of
the sensitivity sequence as εd = descend (ε), εa = ascend (ε) and εr = random (ε) in descending,
ascending, and random order, respectively, over the time steps. We first observe from Figure 1 (left)
and Figure 2 (left) that when the policy-dependent system dynamics At, ∀0 ≤t < T, are almost
surely strongly stable, the gap
MN −MP S2
F of three different patterns εd, εa and εr all decay
at O
  1

N

as N →∞, and the expected total control cost also converges. These coincide exactly
with Proposition 1 and Theorem 1, where the existence of MP S only requires that the sum of the
distributional sensitivities PT −1
t=0 εt is small enough, and the temporal order of each εt is negligible.
We next observe from Figure 1 (middle) and Figure 2 (middle), that when the At, ∀0 ≤t < T,
are almost surely unstable, the iterates Mn of both εd and εr diverge. The expected total control
costs associated with εd and εr are significantly larger than that of εa. This is because large initial
distributional sensitivities in εd and εr rule out the existence of MP S, which matches Proposition 2.
For the general case of At in Figure 1 (right) and Figure 2 (right), all three cases converge due to the
relatively mild sufficient condition (16).

Conclusion

In this work, we have introduced the framework of performative control and studied the conditions
under which a PSC policy exists. We have analyzed the impact of system stability on the existence
of the PSC policy, and proposed a condition on the sum of the distributional sensitivities and the
temporally backwards decaying of sensitivities for almost surely strongly stable and almost surely
unstable systems, respectively. We have also proposed an RSGD algorithm that converges to the PSC
policy in a mean-square sense. The extension of our current results to general control policies and
general control costs [cf. A4], will be explored in future work.

10


---Page Break---
References

[1] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control
with adversarial disturbances. In International Conference on Machine Learning, pages 111–119.
PMLR, 2019.

[2] Naman Agarwal, Elad Hazan, and Karan Singh. Logarithmic regret for online control. Advances
in Neural Information Processing Systems, 32, 2019.

[3] Gavin Brown, Shlomi Hod, and Iden Kalemaj. Performative prediction in a stateful world.
In International conference on artificial intelligence and statistics, pages 6045–6061. PMLR,
2022.

[4] Hugo Carlos, Jean-Bernard Hayet, and Rafael Murrieta-Cid. An analysis of policies from
stochastic linear quadratic gaussian in robotics problems with state-and control-dependent noise.
Journal of Intelligent & Robotic Systems, 92(1):85–106, 2018.

[5] Dmitriy Drusvyatskiy and Lin Xiao. Stochastic optimization with decision-dependent distribu-
tions. Mathematics of Operations Research, 48(2):954–998, 2023.

[6] TE Duncan and B Pasik-Duncan. Stochastic linear-quadratic control with state dependent
fractional brownian noise and stochastic coefficients. IFAC-PapersOnLine, 50(2):199–202,
2017.

[7] Maryam Fazel, Rong Ge, Sham Kakade, and Mehran Mesbahi. Global convergence of policy
gradient methods for the linear quadratic regulator. In International conference on machine
learning, pages 1467–1476. PMLR, 2018.

[8] Udaya Ghai, Arushi Gupta, Wenhan Xia, Karan Singh, and Elad Hazan. Online nonstochastic
model-free reinforcement learning. Advances in Neural Information Processing Systems, 36,
2024.

[9] Benjamin Gravell, Peyman Mohajerin Esfahani, and Tyler Summers. Learning optimal con-
trollers for linear systems with multiplicative noise via policy gradient. IEEE Transactions on
Automatic Control, 66(11):5283–5298, 2020.

[10] Ben Hambly, Renyuan Xu, and Huining Yang. Policy gradient methods for the noisy linear
quadratic regulator over a finite horizon. SIAM Journal on Control and Optimization, 59(5):3359–
3391, 2021.

[11] Elad Hazan, Sham Kakade, and Karan Singh. The nonstochastic control problem. In Algorithmic
Learning Theory, pages 408–421. PMLR, 2020.

[12] Zachary Izzo, Lexing Ying, and James Zou. How to learn when data reacts to your model:
performative gradient descent. In International Conference on Machine Learning, pages
4641–4650. PMLR, 2021.

[13] Qiang Li and Hoi-To Wai. State dependent performative prediction with stochastic approxima-
tion. In International Conference on Artificial Intelligence and Statistics, pages 3164–3186.
PMLR, 2022.

[14] Qiang Li, Chung-Yiu Yau, and Hoi-To Wai. Multi-agent performative prediction with greedy
deployment and consensus seeking agents, 2022. URL https://arxiv. org/abs/2209.03811.

[15] Rui Liu, Guangyao Shi, and Pratap Tokekar. Data-driven distributionally robust optimal control
with state-dependent noise. In 2023 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 9986–9991. IEEE, 2023.

[16] Debmalya Mandal, Stelios Triantafyllou, and Goran Radanovic. Performative reinforcement
learning. In International Conference on Machine Learning, pages 23642–23680. PMLR, 2023.

[17] Horia Mania, Stephen Tu, and Benjamin Recht. Certainty equivalence is efficient for linear
quadratic control. Advances in Neural Information Processing Systems, 32, 2019.

11


---Page Break---
[18] Celestine Mendler-Dünner, Juan Perdomo, Tijana Zrnic, and Moritz Hardt. Stochastic opti-
mization for performative prediction. Advances in Neural Information Processing Systems,
33:4929–4939, 2020.

[19] John P Miller, Juan C Perdomo, and Tijana Zrnic. Outside the echo chamber: Optimizing the
performative risk. In International Conference on Machine Learning, pages 7710–7720. PMLR,
2021.

[20] Stefanos Nikolaidis, Swaprava Nath, Ariel D Procaccia, and Siddhartha Srinivasa. Game-
theoretic modeling of human adaptation in human-robot collaboration. In Proceedings of the
2017 ACM/IEEE international conference on human-robot interaction, pages 323–331, 2017.

[21] Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. Performative
prediction. In International Conference on Machine Learning, pages 7599–7609. PMLR, 2020.

[22] Huyên Pham, Marco Corsi, and Wolfgang J Runggaldier. Numerical approximation by quanti-
zation of control problems in finance under partial observations. In Handbook of Numerical
Analysis, volume 15, pages 325–360. Elsevier, 2009.

[23] Joaquin Quiñonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence.
Dataset shift in machine learning. Mit Press, 2022.

[24] Ben Rank, Stelios Triantafyllou, Debmalya Mandal, and Goran Radanovic. Performative
reinforcement learning in gradually shifting environments. arXiv preprint arXiv:2402.09838,
2024.

[25] Mitas Ray, Lillian J Ratliff, Dmitriy Drusvyatskiy, and Maryam Fazel. Decision-dependent risk
minimization in geometrically decaying dynamic environments. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 36, pages 8081–8088, 2022.

[26] Benjamin Recht. A tour of reinforcement learning: The view from continuous control. Annual
Review of Control, Robotics, and Autonomous Systems, 2:253–279, 2019.

[27] Roberto Renò. Nonparametric estimation of stochastic volatility models. Economics Letters,
90(3):390–395, 2006.

[28] Harish K Venkataraman and Peter J Seiler. Recovering robustness in model-free reinforcement
learning. In 2019 American Control Conference (ACC), pages 4210–4216. IEEE, 2019.

[29] Killian Wood, Gianluca Bianchin, and Emiliano Dall’Anese. Online projected gradient descent
for stochastic optimization with decision-dependent distributions. IEEE Control Systems Letters,
6:1646–1651, 2021.

[30] Farnaz Adib Yaghmaie, Fredrik Gustafsson, and Lennart Ljung. Linear quadratic control using
model-free reinforcement learning. IEEE Transactions on Automatic Control, 68(2):737–752,
2022.

[31] Peng Zhao, Yu-Hu Yan, Yu-Xiang Wang, and Zhi-Hua Zhou. Non-stationary online learn-
ing with memory and non-stochastic control. The Journal of Machine Learning Research,
24(1):9831–9900, 2023.

[32] Hongyu Zhou, Zirui Xu, and Vasileios Tzoumas. Efficient online learning with memory via
frank-wolfe optimization: Algorithms with bounded dynamic regret and applications to control.
In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 8266–8273. IEEE, 2023.

12


---Page Break---
A
Application Example and Experiment Detail.

In this section, we elaborate a concrete example of stock market risk minimization to justify the
policy-dependent state transition model in (8). We also conduct numerical experiment based on this
example to demonstrate the efficacy of our developed theory.

Example 1 (Stock Market Risk Minimization). We describe a risk minimization problem for stock
market investment to illustrate the application of (11). Consider an investor trading a total number
of L stocks over a period of T trading days. The observed market price s(l)
t
of the n-th stock at

the t-th day follows a stochastic volatility model of log s(l)
t+1 = log s(l)
t
+
r−1

2

v(l)
t
2

T
+ v(l)
t
√

T , s(l)
1
>

0, ∀1 ≤t < T, 1 ≤l ≤L, where r > 0 is the riskless interest rate per day, v(l)
t
and s(l)
1
are the
unobservable independent random volatility process and the constant initial stock price associated
with the l-th stock, respectively [22, 27]. Let q(l)
t
and q(l)
1
be the the total return at the t-th day before
the market opens and the initial investment associated with the l-th stock, respectively. The investor
maintains a portfolio for each q(l)
t , which is parameterized by a row vector m(l) =

ml,1, · · · , ml,M

with ml,i ∈[0, 1] , ∀1 ≤i ≤L, being the weight of allocation and PL
i=1 ml,i = 1. Specifically,
at the t-th day when the market opens, the investor immediately allocates q(l)
t
proportionally to
buy the N stocks according to the weight vector m(l). The total systemic risk associated with ql
t
during the period between the t-th day right before market opening and the (t + 1)-th day right
before market opening is PL
i=1

ml,iw(i)
t h

+ 1 · w(l)
t+1 (24 −h) , where wi
t is the per hour risk
associated with the i-th stock in the t-th day and h is the total number of trading hours per day. is
PL
i=1 ml,ie

r −1

2

v(i)
t
2
1
T + v(i)
t
1
√

T


ql
t. The investor then spends all this amount to buy

back the l-th stock and then the t-th trading day ends. Each stock in the portfolio suffers from a
proportionate random i.i.d. systemic risk wi
t per hour, i.e., holding the i-th stock with a ratio of
ml,i in a portfolio for a total number of h trading hours in the t-th day will incur a systemic risk of

ml,iw(i)
t h. Denote the mean-shifted return as rt = qt −E [qt] and let xt =

rt
qt


. It follows that

the evolution of xt can be characterized by the canonical form of

xt+1 = (A + ∆t) xt + u(M)
t
+ ewt,
(20)

u(M)
t
= Mewt−1 with B = I, K = 0, ∀1 ≤t < T.
(21)

where M
=
h
24−h

m(1); · · · ; m(L)
, wt−1
=
(24 −h)
h
w(1)
t , · · · , w(L)
t
iT
,
ewt
=

wt −E [wt]
wt


, the state transition matrix A = I2L×2L and the policy-dependent state tran-

sition perturbation ∆t is given by (22) and (23).

∆t =

"
E
h
V(M)
t
i
−IL×L
V(M)
t
−E
h
V(M)
t
i

0L×L
V(M)
t
−IL×L

#

,
(22)

V(M)
t
= diag











L
X

i=1
m1,ie




r−1

2(v(i)
t )
2

T
+
v(i)
t
√

T




, · · · ,

L
X

i=1
mL,ie




r−1

2(v(i)
t )
2

T
+
v(i)
t
√

T












.
(23)

The investment risk minimization problem can thus be casted as

min
M∈M CM
T = E{∆t},{ ewt}

" T
X

t=1


∥[IL, 0L×L] · xt∥2 +
[IL, 0L×L] · u(M)
t

2#

,
s.t. (20) −(23).

Tackling (11) leads to an optimized stock investment policy M which takes the effects of policy-
dependent random perturbations to the dynamics of stock returns into account.

13


---Page Break---
In the simulation, we set the number of stocks L = 10 and the number of trading days T = 60.
The initial policy M0 is randomly chosen within the feasible set M = {M : P10
i=1 ml,i = 1, ∀1 ≤
l ≤10}. The entries of noise term wt are independently and uniformly drawn from the interval
[0, 1]. For each 1 ≤i ≤10 and each 0 ≤t < 60, we first obtain ˜v(i)
t
by sampling from a Gaussian
distribution, i.e., ˜v(i)
t
∼N(log(εt), 0.2), where εt denotes the sensitivity at t-th trading day. The ˜v(i)
t
is then projected to the interval [−0.6, 0.6] to obtain v(i)
t
in (23). We let the total number iterations
N = 1000, and the stepsize ηn in Algorithm 1 is set to be 0.01, ∀0 ≤n ≤1000.

The sensitivities are shown as follows:

Sensitivities with Ascending Sequence. εa=[1.25797477e-07 5.03189910e-07 1.13217730e-06
2.01275964e-06 3.14493693e-06 4.52870919e-06 6.16407639e-06 8.05103855e-06 1.01895957e-05
1.25797477e-05 1.52214948e-05 1.81148367e-05 2.12597737e-05 2.46563056e-05 2.83044324e-05
3.22041542e-05 3.63554710e-05 4.07583827e-05 4.54128893e-05 5.03189910e-05 4.66004175e-03
5.35796616e-03 6.12231163e-03 6.95609731e-03 7.86234234e-03 8.84406585e-03 9.90428699e-03
1.10460249e-02 1.22722987e-02 1.35861276e-02 1.49905306e-02 1.64885270e-02 1.80831358e-02
1.97773762e-02 2.15742674e-02 2.34768284e-02 2.54880785e-02 2.76110367e-02 2.98487222e-02
3.22041542e-02 4.33504397e-02 4.66004175e-02 5.00089002e-02 5.35796616e-02 5.73164756e-02
6.12231163e-02 6.53033575e-02 6.95609731e-02 7.39997371e-02 7.86234234e-02 8.34358059e-02
8.84406585e-02 9.36417552e-02 9.90428699e-02 1.04647776e-01 1.10460249e-01 1.16484061e-01
1.22722987e-01 1.29180801e-01 1.35861276e-01]

Sensitivities with Descending Sequence. εd=[1.35861276e-01 1.29180801e-01 1.22722987e-01
1.16484061e-01 1.10460249e-01 1.04647776e-01 9.90428699e-02 9.36417552e-02 8.84406585e-02
8.34358059e-02 7.86234234e-02 7.39997371e-02 6.95609731e-02 6.53033575e-02 6.12231163e-02
5.73164756e-02 5.35796616e-02 5.00089002e-02 4.66004175e-02 4.33504397e-02 3.22041542e-02
2.98487222e-02 2.76110367e-02 2.54880785e-02 2.34768284e-02 2.15742674e-02 1.97773762e-02
1.80831358e-02 1.64885270e-02 1.49905306e-02 1.35861276e-02 1.22722987e-02 1.10460249e-02
9.90428699e-03 8.84406585e-03 7.86234234e-03 6.95609731e-03 6.12231163e-03 5.35796616e-03
4.66004175e-03 5.03189910e-05 4.54128893e-05 4.07583827e-05 3.63554710e-05 3.22041542e-05
2.83044324e-05 2.46563056e-05 2.12597737e-05 1.81148367e-05 1.52214948e-05 1.25797477e-05
1.01895957e-05 8.05103855e-06 6.16407639e-06 4.52870919e-06 3.14493693e-06 2.01275964e-06
1.13217730e-06 5.03189910e-07 1.25797477e-07]

Sensitivities with Random Sequence.
εr=[2.54880785e-02 3.63554710e-05 2.34768284e-02
1.64885270e-02 1.25797477e-07 9.36417552e-02 1.10460249e-02 6.16407639e-06 2.76110367e-02
5.00089002e-02 1.16484061e-01 1.04647776e-01 2.15742674e-02 2.46563056e-05 3.14493693e-06
4.66004175e-03 4.54128893e-05 3.22041542e-02 1.22722987e-02 4.33504397e-02 8.84406585e-03
6.95609731e-03 5.03189910e-05 1.01895957e-05 7.39997371e-02 1.97773762e-02 7.86234234e-02
1.35861276e-02 8.84406585e-02 1.22722987e-01 4.07583827e-05 8.05103855e-06 2.98487222e-02
9.90428699e-03 6.95609731e-02 8.34358059e-02 6.12231163e-03 1.81148367e-05 1.52214948e-05
2.12597737e-05 6.53033575e-02 5.03189910e-07 5.35796616e-03 9.90428699e-02 7.86234234e-03
1.10460249e-01 4.66004175e-02 2.83044324e-05 3.22041542e-05 2.01275964e-06 1.13217730e-06
6.12231163e-02 5.35796616e-02 1.49905306e-02 1.25797477e-05 1.80831358e-02 4.52870919e-06
1.35861276e-01 1.29180801e-01 5.73164756e-02]

The non-ordered lists of sensitivity values in epsilon εa,εd, and εr are the same. The state trajectory
xt and control action ut are generated according to (20)- (23) based on the above system parameter
configurations.

B
Auxiliariy Lemmas for Per Stage Cost Function

In this part, we provide several lemmas to characterize the gradient properties and the smoothness of
the per stage cost ct. We first summarize a collection of notations. We denote ∇xct

x(M)
t
, u(M)
t

and

∇uct

x(M)
t
, u(M)
t

as the gradient of ct w.r.t. the system state x(M)
t
and control action u(M)
t
, respec-

tively. We denote ∇Mct

x(M)
t
, u(M)
t

as the gradient of ct w.r.t. the variable M in x(M)
t
and u(M)
t
under any given realization of the policy-dependent state transition perturbations {∆i, 0 ≤i < t} .

14


---Page Break---
Based on the chain rule and the relationship between M and the pair (xt, ut) in (7) and (10), the

expression of ∇Mct

x(M)
t
, u(M)
t

is given by

∇Mct

x(M)
t
, u(M)
t

(24)

=

t−1
X

i=0




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct

x(M)
t
, u(M)
t
 
[w]H
i−1
⊤

−

t−1
X

i=0










K

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇uct

x(M)
t
, u(M)
t
 
[w]H
i−1
⊤







+ ∇uct

x(M)
t
, u(M)
t
 
[w]H
t−1
⊤
.

For the ease of notation, without ambiguity we also occasionally use the notation ct

M; {∆i}0≤i<t


to denote ct

x(M)
t
, u(M)
t

under a sequence of policy-dependent state transition perturbations

{∆i, 0 ≤i < t} . In this case, ∇Mct

M; {∆i}0≤i<t

denote the gradient taken w.r.t. the first
argument M.

Moreover, the total cost function can be represented as

JT

M; {∆t}0≤t<T

=

T
X

t=0
ct

M; {∆i}0≤i<t

.
(25)

It is clear that

CT (M; M′) = Ex0,{∆t∼Dt(M′)}0≤i<T ,{wi}0≤i<T JT

M; {∆t}0≤t<T

.
(26)

Unless otherwise specified, ∇JT

M; {∆t}0≤t<T

and ∇CT (M; M′) denotes the gradient taken
w.r.t. the first argument M.

Convexity. Denote the per stage expected cost

ft (M; M1) = Ex0,{∆i∼Di(M1)}0≤i<t,{wi}0≤i<t

h
ct

M; {∆i}0≤i<t
i
, ∀1 ≤t ≤T,

where the distribution of policy-dependent perturbation is changed from ∆i ∼Di (M) to ∆i ∼
Di (M1), ∀0 ≤i < t.

We have the following lemma characterizing the convexity of ft (M; M1) , ∀1 ≤t ≤T.

Lemma B.1 The per stage expected cost ft (M; M′) is a convex function of M, ∀M, M′ ∈M,
∀1 ≤t ≤T.

Proof.
We prove this lemma following the convex property of the function ct (x, u) in Assumption
A 4. Let 0 < θ < 1 and M1, M2, M′ ∈M. Consder the weighted policy θM1 + (1 −θ) M2, we

15


---Page Break---
have

x(θM1+(1−θ)M2)
t
=

t−1
Y

i=0


eA + ∆i

x0 +

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B
(27)

· (θM1 + (1 −θ) M2) M [w]H
i−1 +

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

wi

= θ




t−1
Y

i=0


eA + ∆i

x0 +

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

BM1 [w]H
i−1

+

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

wi



+ (1 −θ)

·




t−1
Y

i=0


eA + ∆i

x0 +

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

BM2 [w]H
i−1

+

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

wi





= θxM1
t
+ (1 −θ) xM2
t
.

As a result,

ft (θM1 + (1 −θ) M2; M′)
(28)

= Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

h
ct

x(M1+(1−θ)M2)
t
, u(M1+(1−θ)M2)
t
i

= Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

h
ct

θxM1
t
+ (1 −θ) xM2
t
, θuM1
t
+ (1 −θ) uM2
t
i

(28.a)
≤
Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

h
θct

xM1
t
, uM1
t

+ (1 −θ) ct

xM2
t
, uM2
t
i

= θft (M1; M′) + (1 −θ) ft (M2; M′) ,

where inequality (28.a) holds becasue of the convexity of ct (x, u).
□

We next have a key lemma characterizing the strong convexity of ft (M; M′) , ∀H ≤t ≤T.

Lemma B.2 The per stage expected cost ft (M; M′) is a min
n
µ
2 , µσ2γ2

64κ10
o
-strongly convex function

of M, ∀M, M′ ∈M, ∀H ≤t ≤T.

Proof.
We prove this lemma following the strong convexity of the function ct (x, u) in Assumption
A 4. Let M1, M2, M′ ∈M. Consider a time step t such that H ≤t ≤T. For any given realizations
of {∆i}0≤i<t, where ∆i ∼Di (M′) , ∀0 ≤i < t, we have

ct

x(M1)
t
, u(M1)
t

−ct

x(M2)
t
, u(M2)
t

(29)

(29.a)
≥



∇xct

x(M2)
t
, u(M2)
t


∇uct

x(M2)
t
, u(M2)
t






T "
x(M1)
t
−x(M2)
t
u(M1)
t
−u(M2)
t

#

+ µ

2


x(M1)
t
−x(M2)
t
u(M1)
t
−u(M2)
t



2

,

where inequality (29.a) holds because of the strong convexity assumption of ct in Assumption A4.

16


---Page Break---
Based on the representations of the pair (xt, ut) in (7) and (10), it follows that

x(M1)
t
−x(M2)
t
=

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B (M1 −M2) [w]H
i−1
(30)

u(M1)
t
−u(M2)
t
= −K

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B (M1 −M2) [w]H
i−1
(31)

+ (M1 −M2) [w]H
t−1 .

Therefore, we obtain



∇xct

x(M2)
t
, u(M2)
t


∇uct

x(M2)
t
, u(M2)
t






T "
x(M1)
t
−x(M2)
t
u(M1)
t
−u(M2)
t

#

(32)

=

t−1
X

i=0
∇⊤
x ct

x(M2)
t
, u(M2)
t




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B (M1 −M2) [w]H
i−1





+

t−1
X

i=0
∇⊤
u ct

x(M2)
t
, u(M2)
t



−K

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B (M1 −M2) [w]H
i−1

+ (M1 −M2) [w]H
t−1


= Tr








t−1
X

i=0
[w]H
i−1 ∇⊤
x ct

x(M2)
t
, u(M2)
t

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B



(M1 −M2)





+ Tr







−

t−1
X

i=0
[w]H
i−1 ∇⊤
u ct

x(M2)
t
, u(M2)
t

K

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B

+ [w]H
t−1 ∇⊤
u ct

x(M2)
t
, u(M2)
t

(M1 −M2)


= Tr

∇⊤
M2ct

x(M2)
t
, u(M2)
t

(M1 −M2)

.

Substitute (32) back into (29), we have

ct

x(M1)
t
, u(M1)
t

−ct

x(M2)
t
, u(M2)
t

≥Tr

∇⊤
M2ct

x(M2)
t
, u(M2)
t

(M1 −M2)

(33)

+ µ

2

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2
.

Taking full expectation on both sides of (33), it follows

ft (M1; M′) −ft (M1; M′) ≥Tr
 
∇⊤ft (M2; M′) · (M1 −M2)

(34)

+ µ

2 Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2
.

17


---Page Break---
We next analyze the last term in (34). Consider the conditional expectation on {wi}0≤i<t as follows

Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2
(35)

= E{wi}0≤i<t


Ex0,{∆i∼Di(M′)}0≤i<t

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2 {wi}0≤i<t



≥E{wi}0≤i<t

Ex0,{∆i∼Di(M′)}0≤i<t

h
x(M1)
t
−x(M2)
t
 {wi}0≤i<t
i
2

+
Ex0,{∆i∼Di(M′)}0≤i<t

h
u(M1)
t
−u(M2)
t
 {wi}0≤i<t
i
2

(35.a)
=
E{wi}0≤i<t







t−1
X

i=0
eAiB (M1 −M2) [w]H
i−1



2

+

K

t−1
X

i=0
eAiB (M1 −M2) [w]H
i−1 −(M1 −M2) [w]H
t−1



2

,

where (35.a) holds becasue E [∆t]=0, ∀0 ≤t < T.

We next consider each term on R.H.S. of (35.a) . We notice that

vec

 t−1
X

i=0
eAiB (M1 −M2) [w]H
i−1

!

=

 t−1
X

i=0


[w]H
i−1
⊤
⊗

eAiB
!

vec (M1 −M2)
(36)

Therefore,

E{wi}0≤i<t







t−1
X

i=0
eAiB (M1 −M2) [w]H
i−1



2


(37)

= δ⊤
ME{wi}0≤i<t





 t−1
X

i=0


[w]H
i−1
⊤
⊗

eAiB
!⊤ t−1
X

i=0


[w]H
i−1
⊤
⊗

eAiB
!

δM

= δ⊤
M

  
t
X

i1=1

t
X

i2=1


Λi1−i2 ⊗

eAi1−1B
⊤eAi2−1B
!

⊗E

wtw⊤
t

!

δM

= δ⊤
M
   
IH ⊗B⊤
Ψ (IH ⊗B)

⊗E

wtw⊤
t

δM,

where δM = vec (M1 −M2), Λt ∈RH×H with [Λl]i,j = 1 if and only if i −j = l and 0 otherwise

and Ψ = Pt
i1=1
Pt
i2=1


Λi1−i2 ⊗

eAi1−1B
⊤eAi2−1B

.

Using similar approach, we can obtain

E{wi}0≤i<t





K

t−1
X

i=0
eAiB (M1 −M2) [w]H
i−1 −(M1 −M2) [w]H
t−1



2


(38)

= δ⊤
M
 
IH ⊗B⊤ eΨ
 
IH ⊗B⊤
−Θ (IH ⊗B) −
 
IH ⊗B⊤
Θ⊤+ IHdu

⊗E

wtw⊤
t

δM,

where eΨ
=
Pt
i1=1
Pt
i2=1


Λi1−i2 ⊗

K eAi1−1B
⊤
K eAi2−1B

and Θ
=
(IH ⊗K) ⊗
Pt
i=1 Λ−i ⊗

eAi−1B


Substitue (38) and (37) back into (35), it follows that

Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2
(39)

= δ⊤
M
 
Ω⊗E

wtw⊤
t

δM.

18


---Page Break---
where

Ω=
 
IH ⊗B⊤
Ψ (IH ⊗B) +
 
IH ⊗B⊤ eΨ
 
IH ⊗B⊤
(40)

−Θ (IH ⊗B) −
 
IH ⊗B⊤
Θ⊤+ IHdu.

Based on Lemma F.1 and F.2 in [2], we have, ∀H ≤t ≤T, Ψ ≥
1
4κ4 IHdx and ∥Θ∥≤γ−1κ3.
Therefore, if ∥IH ⊗B∥≥
γ
4κ3 , then

Ω≥
1
4κ4 IHdx
 γ

4κ3

2
=
γ2

64κ10 IHdx.
(41)

Otherwise, if ∥IH ⊗B∥<
γ
4κ3 , then

Ω≥IHdu −Θ (IH ⊗B) −
 
IH ⊗B⊤
Θ⊤,
(42)

∥Ω∥≥1 −γ−1κ3 γ

4κ3 −γ−1κ3 γ

4κ3 = 1

2.
(43)

Combine (41) and (43), we have ∥Ω∥≥min
n
1
2,
γ2

64κ10
o
. As a result,

Ex0,{∆i∼Di(M′)}0≤i<t,{wi}0≤i<t

x(M1)
t
−x(M2)
t

2
+
u(M1)
t
−u(M2)
t

2
(44)

= δ⊤
M
 
Ω⊗E

wtw⊤
t

δM ≥δM ∥Ω∥σ2 ≥min
σ2

2 , γ2σ2

64κ10


∥M1 −M2∥2
F .

Substitue (44) back into (34), it follows that

ft (M1; M′) −ft (M1; M′) ≥Tr
 
∇⊤ft (M2; M′) · (M1 −M2)

(45)

+
min
n
µσ2

2 , µγ2σ2

64κ10
o

2
∥M1 −M2∥2
F .

Therefore, Lemma B.2 is proved.
□

Properties of Gradient. The following lemma provides upper bounds for the norm of gradients
∇xct

x(M)
t
, u(M)
t
 and
∇uct

x(M)
t
, u(M)
t
 .

Lemma B.3 The gradients of the per stage cost ct

x(M)
t
, u(M)
t

, ∀1 ≤t ≤T, satifies
∇xct

x(M)
t
, u(M)
t
 ≤x0κ2Gαt + κ2GW (∥B∥HM + 1) βt,
(46)
∇uct

x(M)
t
, u(M)
t
 ≤x0κ3Gαt + κ3GW (∥B∥HM + 1) βt + GMHW,
(47)

where αt = Qt−1
i=0
 
1 −γ + κ2ξi

and βt = Pt−1
i=0
Qt−1
j=i+1
 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

.

Proof.
We first provide upper bounds for ∥xt∥and ∥ut∥. From the evolution of xt in (10), we
obtain

∥xt∥≤



t−1
Y

i=0


eA + ∆i

x0

 +



t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

BM [w]H
i−1


(48)

+



t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

wi


≤x0κ2
t−1
Y

i=0

 
1 −γ + κ2 ∥∆i∥


+ κ2W (∥B∥HM + 1)

t−1
X

i=0

t−1
Y

j=i+1

1j<t
 
1 −γ + κ2 ∥∆j∥

+ 1j=tI
 ,

∥ut∥≤κ ∥xt∥+ MHW.
(49)

19


---Page Break---
Lemma B.3 follows by noting that ∥∆t∥≤ξt, ∀0 ≤t ≤T, ∀M ∈M, and whenever ∥xt∥, ∥ut∥≤

D, we have
∇xct

x(M)
t
, u(M)
t
 ,
∇uct

x(M)
t
, u(M)
t
 ≤GD.
□

We next prove that the variance of the gradient ∇Mct

x(M)
t
, u(M)
t

is bounded.

Lemma B.4 There exists a ϑt > 0, ∀1 ≤t ≤T, such that

Ex0,{∆i∼Di(M)}0≤i<t,{wi}0≤i<t

h∇Mct

x(M)
t
, u(M)
t

(50)

−Ex0,{∆i∼Di(M)}0≤i<t,{wi}0≤i<t

h
∇Mct

x(M)
t
, u(M)
t
i
2

F


≤ϑt.

Proof.
Based on the expression of ∇Mct

x(M)
t
, u(M)
t

in (24), we have

∇Mct

x(M)
t
, u(M)
t

(51)

≤∥B∥HW

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI



∇xct

x(M)
t
, u(M)
t


+ κ ∥B∥HW

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI



∇uct

x(M)
t
, u(M)
t


+ HW
∇uct

x(M)
t
, u(M)
t


≤κ2 ∥B∥HWβt
∇xct

x(M)
t
, u(M)
t
 +
 
κ3 ∥B∥βt + 1

HW
∇uct

x(M)
t
, u(M)
t


(51.a)
≤
 
κ ∥B∥HWβt + κ3 ∥B∥βt + 1
  
x0κ3Gαt + κ3GW (∥B∥HM + 1) βt


+
 
κ3 ∥B∥βt + 1

GMHW,

where inequality (51.a) holds becasue of the gradient norm bounds in Lemma B.3.

Further note that

E
∇Mct

x(M)
t
, u(M)
t

−E
h
∇Mct

x(M)
t
, u(M)
t
i
2

F


≤E
∇Mct

x(M)
t
, u(M)
t

2

F



(52)

≤d2
xHE
∇Mct

x(M)
t
, u(M)
t

2

The desired result follows by substituting (51) into (52) and letting

ϑt = d2
xH
  
κ ∥B∥HWβt + κ3 ∥B∥βt + 1
  
x0κ3Gαt + κ3GW (∥B∥HM + 1) βt

(53)

+
 
κ3 ∥B∥βt + 1

GMHW

.

□

The next lemma characterizes the difference between the gradients under any two different polices
M and M′.

20


---Page Break---
Lemma B.5 For any given two policies M, M′ ∈M, the differences between the gradients satify,
∀1 ≤t ≤T,
∇xct

x(M)
t
, u(M)
t

−∇xct


x(M′)
t
, u(M′)
t



+
∇uct

x(M)
t
, u(M)
t

−∇uct


x(M′)
t
, u(M′)
t


(54)

≤ς
 
(1 + κ) κ2HWβt + HW

∥M −M′∥

+ ς (1 + κ) κ2 (1 −γ)−1 (x0αt + (∥B∥HM + 1) Wβt)

t−1
X

i=0
∥∆i −∆′
i∥,

where ∆i and ∆′
i, ∀0 ≤i < t, denotes the policy-dependent state transition perturbations under the
policy M and M′, respectively.

Proof.
Recall the smoothness of the cost function ct (x, u) in A 5, i.e.,

∥∇xct (x1, u1) −∇xct (x2, u2)∥+ ∥∇uct (x1, u1) −∇uct (x2, u2)∥
(55)
≤ς (∥x1 −x2∥+ ∥u1 −u2∥) .

To prove Lemma B.5, it suffices to quantify
x(M)
t
−x(M′)
t

 and
u(M)
t
−u(M′)
t

.

Based on the evolution of the system state x(M)
t
, we have

x(M)
t
−x(M′)
t
= E1 + E2 + E3,
(56)

where

E1 =

 t−1
Y

i=0


eA + ∆i

−

t−1
Y

i=0


eA + ∆′
i
!

x0,
(57)

E2 =

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

BM [w]H
i−1
(58)

−

t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

BM′ [w]H
i−1 ,

E3 =

t−1
X

i=0




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI



wi. (59)

We next analyze each term in (56) one by one. Specifically,

∥E1∥≤x0



t−1
Y

i=0


eA + ∆i

−

t−1
Y

i=0


eA + ∆′
i

(60)

≤x0

(∆0 −∆′
0)

t−1
Y

i=1


eA + ∆i

+

eA + ∆′
0

(∆1 −∆′
1)

t−1
Y

i=2


eA + ∆i


+ · · · +

t−2
Y

i=0


eA + ∆′
i
  
∆t−1 −∆′
t−1



≤x0κ2 (1 −γ)−1
t−1
Y

i=0

 
1 −γ + κ2ξi
 t−1
X

i=0
∥∆i −∆′
i∥

= x0κ2 (1 −γ)−1 αt

t−1
X

i=0
∥∆i −∆′
i∥.

21


---Page Break---
For the term E2, we use the similar approach for analyzing E1 in (74) and obtain the following bound

∥E2∥
(61)

≤



t−1
X

i=0

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B (M −M′) [w]H
i−1

+

t−1
X

i=0




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI



BM′ [w]H
i−1



≤κ2HWβt ∥M −M′∥

+ ∥B∥MHW

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI



(61.a)
≤
κ2HWβt ∥M −M′∥+ κ2 (1 −γ)−1 ∥B∥MHWβt

t−1
X

i=0
∥∆i −∆′
i∥,

where inequality (61.a) is obtained by using (76).

Finally, still using (76), an upper bound of ∥E3∥is obtained as follows

∥E3∥≤W

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI


(62)

≤κ2 (1 −γ)−1 Wβt

t−1
X

i=0
∥∆i −∆′
i∥.

It follow directly that
x(M)
t
−x(M′)
t

 ≤∥E1∥+ ∥E2∥+ ∥E3∥
(63)

≤κ2HWβt ∥M −M′∥+ κ2 (1 −γ)−1 (x0αt + (∥B∥HM + 1) Wβt)

t−1
X

i=0
∥∆i −∆′
i∥.

We next observe the relationship
u(M)
t
−u(M′)
t

 =
−Kx(M)
t
+ Kx(M′)
t
+ M [w]H
t−1 −M′ [w]H
t−1


(64)

≤κ
x(M)
t
−x(M′)
t

 + HW ∥M −M′∥.

Therefore, it follows that
x(M)
t
−x(M′)
t

 +
u(M)
t
−u(M′)
t

 ≤(1 + κ)
x(M)
t
−x(M′)
t

 + HW ∥M −M′∥
(65)

≤
 
(1 + κ) κ2HWβt + HW

∥M −M′∥

+ (1 + κ) κ2 (1 −γ)−1 (x0αt + (∥B∥HM + 1) Wβt)

t−1
X

i=0
∥∆i −∆′
i∥.

Combining (65) and (55), we obatin the desired result.
□

Smoothness: We analyze the smoothness of the per stage cost and the per stage expected cost in the
following Lemma B.6 and B.8, respectively.

22


---Page Break---
Lemma B.6 The per stage cost function ct

M; {∆i}0≤i<t

, ∀1 ≤t ≤T, is smooth in the sense
that

∇Mct

M; {∆i}0≤i<t

−∇M′ct

M′; {∆′
i}0≤i<t

F
(66)

≤λt ∥M −M′∥F + νt

t−1
X

i=0
∥∆i −∆′
i∥F ,

where

λt =dx
√

HςH2W 2  
1 + ∥B∥
 
κ2 + κ3   
κ2 + κ3
βt + 1

,
(67)

νt =dx
√

H

ςHW
 
1 + ∥B∥
 
κ2 + κ3  
κ2 + κ3
(1 −γ)−1 + G (1 −γ)−1
(68)

·
 
κ4 + κ5
HW ∥B∥βt

(x0αt + (∥B∥HM + 1) Wβt) ,

and ∆i and ∆′
i, ∀0 ≤i < t, denotes the policy-dependent state transition perturbations under the
policy M and M′, respectively.

Proof.
We prove this lemma based on the smoothness of the cost function ct (x, u) in A 5, i.e.,

∥∇xct (x1, u1) −∇xct (x2, u1)∥+ ∥∇uct (x1, u1) −∇uct (x1, u2)∥
≤ς (∥x1 −x2∥+ ∥u1 −u2∥) .
(69)

Based on the expression of ∇Mct

M; {∆i}0≤i<t

in (24), it follows that

∇Mct

M; {∆i}0≤i<t

−∇M′ct

M′; {∆′
i}0≤i<t

≤F1 + F2 + F3,
(70)

where

F1 =

t−1
X

i=0









t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct

x(M)
t
, u(M)
t

(71)

−




t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

B





⊤

∇xct


x(M′)
t
, u(M′)
t







[w]H
i−1
⊤
,

F2 =

t−1
X

i=0








K

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct

x(M)
t
, u(M)
t

(72)

−



K

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

B





⊤

∇xct


x(M′)
t
, u(M′)
t







[w]H
i−1
⊤
,

F3 =

∇uct

x(M)
t
, u(M)
t

−∇uct


x(M′)
t
, u(M′)
t

 
[w]H
t−1
⊤
.
(73)

23


---Page Break---
We next analyze each term in (70) one by one. Specifically,

∥F1∥=



t−1
X

i=0









t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct

x(M)
t
, u(M)
t

(74)

−




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct


x(M′)
t
, u(M′)
t



+




t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

B





⊤

∇xct


x(M′)
t
, u(M′)
t



−




t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

B





⊤

∇xct


x(M′)
t
, u(M′)
t







[w]H
i−1
⊤


(74.a)
≤
HW ∥B∥κ2βt

∇xct

x(M)
t
, u(M)
t

−∇xct


x(M′)
t
, u(M′)
t



+ HW ∥B∥κ2 (x0Gαt + GW (∥B∥HM + 1) βt)

·

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI


,

where inequality (74.a) holds due to the definition of βt
and the upper bound of
∇xct


x(M′)
t
, u(M′)
t

 provided in Lemma B.3.

24


---Page Break---
For the last term in (74), we first observe that


t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI


(75)

=



 
∆′
i+1 −∆i+1

t−1
Y

j=i+2


1j<t

eA + ∆′
j

+ 1j=tI


+

eA + ∆i+1
  
∆′
i+2 −∆i+2

t−1
Y

j=i+3


1j<t

eA + ∆′
j

+ 1j=tI

+ · · ·

+

i+1+l
Y

k=i+1


1k<t

eA + ∆k

+ 1k=tI
  
∆′
i+1+l+1 −∆i+1+l+1

t−1
Y

j=i+1+l+2


1j<t

eA + ∆′
j

+ 1j=tI


+ · · · +

t−2
Y

k=i+1


1k<t

eA + ∆k

+ 1k=tI
  
∆′
t−1 −∆t−1



≤




t−1
Y

j=i+1

 
1j<t
 
1 −γ + κ2ξj

+ 1j=t





κ2 ∆′
i+1 −∆i+1
  
1 −γ + κ2ξi+1
−1

+ · · · + κ2  
∆′
i+1+l+1 −∆i+1+l+1
  
1 −γ + κ2ξi+1+l+1
−1 + · · ·

+ κ2 ∆′
t−1 −∆t−1
  
1 −γ + κ2ξt−1
−1

≤




t−1
Y

j=i+1

 
1j<t
 
1 −γ + κ2ξj

+ 1j=t




t−1
X

k=i+1
κ2 (1 −γ)−1 ∥∆′
k −∆k∥

≤




t−1
Y

j=i+1

 
1j<t
 
1 −γ + κ2ξj

+ 1j=t




t−1
X

k=0
κ2 (1 −γ)−1 ∥∆′
k −∆k∥.

As a result,

t−1
X

i=0



t−1
Y

j=i+1


1j<t

eA + ∆′
j

+ 1j=tI

−

t−1
Y

j=i+1


1j<t

eA + ∆j

+ 1j=tI


(76)

≤

t−1
X

i=0

t−1
Y

j=i+1

 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

 t−1
X

k=0
κ2 (1 −γ)−1 ∥∆′
k −∆k∥

!

= κ2 (1 −γ)−1 βt

t−1
X

i=0
∥∆′
i −∆i∥.

Substitue (76) into (74), it follows

∥E1∥≤HW ∥B∥κ2βt

∇xct

x(M)
t
, u(M)
t

−∇xct


x(M′)
t
, u(M′)
t


(77)

+ (x0Gαt + GW (∥B∥HM + 1) βt) κ2 (1 −γ)−1
t−1
X

i=0
∥∆′
i −∆i∥

!

The upper bounds for ∥E2∥and ∥E3∥can be readily obtained as follows

∥E2∥≤κHW ∥B∥κ2βt

∇xct

x(M)
t
, u(M)
t

−∇xct


x(M′)
t
, u(M′)
t


(78)

+ (x0Gαt + GW (∥B∥HM + 1) βt) κ2 (1 −γ)−1
t−1
X

i=0
∥∆′
i −∆i∥

!

25


---Page Break---
and

∥E3∥≤HW
∇uct

x(M)
t
, u(M)
t

−∇uct


x(M′)
t
, u(M′)
t

 .
(79)

Substitute (77), (78) and (79) into (70), we have
∇Mct

M; {∆i}0≤i<t

−∇M′ct

M′; {∆′
i}0≤i<t

(80)

≤
 
1 + ∥B∥
 
κ2 + κ3
HW
∇xct

x(M)
t
, u(M)
t

−∇xct


x(M′)
t
, u(M′)
t



+
∇uct

xt, u(M)
t

−∇uct


xt, u(M′)
t




+ eβt

t−1
X

i=0
∥∆′
i −∆i∥,

where eβt = (1 −γ)−1  
κ4 + κ5
HW ∥B∥βt (x0Gαt + GW (∥B∥HM + 1) βt) .

Applying Lemma B.5 to (80) leads to the desired result.
□

Before we proceed to analyze the gradient difference of the expected per stage cost, we cite the
following tool lemma in [21].

Lemma B.7 (Kantorovich-Rubinstein (Lemma D.3 in [21])) A distribution map D (·) is ε-sensitive
if and only if for all M,M′ ∈M :

sup

EA∼D(M) [g (A)] −EA′∼D(M′) [g (A′)] : g : Rdx×dx →R, g 1 −Lipschitz
	

≤ε ∥M −M′∥F .
(81)

The smoothness regarding the expected per stage cost is summarized below.

Lemma B.8 For any M, M′, M1, M2 ∈M and ∀1 ≤t ≤T, the following inequality holds
Ex0,{∆i∼Di(M1)}0≤i<t,{wi}0≤i<t

h
∇Mct

M; {∆i}0≤i<t
i
(82)

−Ex0,{∆′
i∼Di(M2)}0≤i<t,{wi}0≤i<t

h
∇M′ct

M′; {∆′
i}0≤i<t
i
F

≤λt ∥M −M′∥F + νt

t−1
X

i=0
εi ∥M1 −M2∥F .

Proof.
The norm of the gradient difference in (82) can be expanded as
E
h
∇Mct

M; {∆i}0≤i<t
i
−∇M′ct

M′; {∆′
i}0≤i<t

F
(83)

≤
E
h
∇Mct

M; {∆i}0≤i<t

−∇M′ct

M′, {∆i}0≤i<t
i
F

+
E
h
∇M′ct

M′, {∆i}0≤i<t

−∇M′ct

M′, ∆′
1, {∆i}1≤i<t
i
F + · · ·

+
E
h
∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , {∆i}k≤i<t

−∇M′ct

M′,

∆′
j
	

0≤j≤k , {∆i}k+1≤i<t
i
F

+ · · · +
E
h
∇M′ct

M′, {∆′
i}0≤i<t−1 , ∆t−1

−∇M′ct

M′, {∆′
i}0≤i<t−1
i
F ,

where we drop the subscript in E [·] since the associated randomness is clear from the context.

26


---Page Break---
We next analyze a general term in R.H.S. of inequality (83). Specifically,
E
h
∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , {∆i}k≤i<t

−∇M′ct

M′,

∆′
j
	

0≤j≤k , {∆i}k+1≤i<t
i
F
(84)

=
E
h
E
h
∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆k, {∆i}k+1≤i<t


−∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆′
k, {∆i}k+1≤i<t


∆′
j
	

0≤j≤k−1 , {∆i}k+1≤i<t
ii
F

≤E
hE
h
∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆k, {∆i}k+1≤i<t


−∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆′
k, {∆i}k+1≤i<t


∆′
j
	

0≤j≤k−1 , {∆i}k+1≤i<t
i
F

i
.

Given
n
∆′
j
	

0≤j≤k−1 , {∆i}k+1≤i<t
o
, for notation conciseness, we abbreviate

∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆k, {∆i}k+1≤i<t

and ∇M′ct

M′,

∆′
j
	

0≤j≤k−1 , ∆′
k, {∆i}k+1≤i<t


as ∇M′ct (M′, ∆k) and ∇M′ct (M′, ∆′
k) , respectively. It follows that

∥E [∇M′ct (M′, ∆k) −∇M′ct (M′, ∆′
k)]∥2
F
(85)

= Tr

(E [∇M′ct (M′, ∆k)] −E [∇M′ct (M′, ∆′
k)])⊤(E [∇M′ct (M′, ∆k)] −E [∇M′ct (M′, ∆′
k)])


= ∥E [∇M′ct (M′, ∆k) −∇M′ct (M′, ∆′
k)]∥F Tr
 
V ⊤E [∇M′ct (M′, ∆k)] −V ⊤E [∇M′ct (M′, ∆′
k)]

,

where

V = E [∇M′ct (M′, ∆k)] −E [∇M′ct (M′, ∆′
k)]
∥E [∇M′ct (M′, ∆k) −∇M′ct (M′, ∆′
k)]∥F
, ∥V ∥F = 1.
(86)

Based on Lemma B.6,
given
n
∆′
j
	

0≤j≤k−1 , {∆i}k+1≤i<t
o
, the conditional gradient

V ⊤E [∇M′ct (M′, ∆k)] is νt-Lipschitz in ∆k. Further note that ∆k ∼Dk (M1) and ∆′
k ∼
Dk (M2), which are both εk-sensitive. Applying Lemma B.7, it follows that

Tr
 
V ⊤E [∇M′ct (M′, ∆k)] −V ⊤E [∇M′ct (M′, ∆′
k)]

≤νtεk ∥M1 −M2∥F .
(87)

Substitute (87) into (85), it follows that

∥E [∇M′ct (M′, ∆k) −∇M′ct (M′, ∆′
k)]∥F ≤νtεk ∥M1 −M2∥F .
(88)

Using the similar approach, we can obtain
E
h
∇Mct

M; {∆i}0≤i<t

−∇M′ct

M′, {∆i}0≤i<t
i ≤λt ∥M −M′∥F ;
(89)
E
h
∇M′ct

M′, {∆i}0≤i<t

−∇M′ct

M′, A′
1, {∆i}1≤i<t
i ≤νtε1 ∥M1 −M2∥F ; (90)

...E
h
∇M′ct

M′, {∆′
i}0≤i<t−1 , ∆t−1

−∇M′ct

M′, {∆′
i}0≤i<t−1
i

≤νtεt−1 ∥M1 −M2∥F .
(91)

Combining (83), (89)-(91), it follow inequality (82).
□

C
Properties of the Total Cost Function

We define a total cost function JT as

JT

M; {∆t}0≤t<T

=

T
X

t=0
ct

M; {∆i}0≤i<t

.
(92)

27


---Page Break---
It is clear that

CT (M; M1) = Ex0,{∆t∼Dt(M1)}0≤i<T ,{wi}0≤i<T JT

M; {∆t}0≤t<T

.
(93)

We next analyze the gradient properties and the smoothness of the total cost function
JT

M; {∆t}0≤t<T

. We again note that ∇JT

M; {∆t}0≤t<T

denotes the gradient taken w.r.t.
the first argument M.

Lemma C.1 The variance of the gradient of the total cost function satisfies

Ex0,{∆t∼Dt(M)}0≤i<T ,{wi}0≤i<T

∇JT

M; {∆t}0≤t<T

−∇CT (M; M)

2

F


≤T

T
X

t=1
ϑ2
t.

(94)

Proof.
Note that ∇J0 (M; ∆0) = ∇Mc0 (x0, u0) = ∇Mc0 (x0, −Kx0) = 0. Inequality (94) then
follows directly from the norm triangular inequality:

E
∇JT

M; {∆t}0≤t<T

−E
h
∇JT

M; {∆t}0≤t<T
i
2

F


(95)

≤T

T
X

t=1
Ex0,{∆i∼Di(M)}0≤i<t,{wi}0≤i<t

h∇Mct

x(M)
t
, u(M)
t


−Ex0,{∆i∼Di(M)}0≤i<t,{wi}0≤i<t

h
∇Mct

x(M)
t
, u(M)
t
i
2

F



(95.a)
≤
T

T
X

t=1
ϑ2
t,

where inequality (95.a) holds becasue of Lemma B.4.
□

D
Proof of Lemma 2

Denote ft (M; M1) = Ex0,{∆i∼Di(M1)}0≤i<t,{wi}0≤i<t

h
ct

M; {∆i}0≤i<t
i
as the per stage

expected cost with the distribution of policy-dependent perturbation is shifted from ∆i ∼Di (M) to
∆i ∼Di (M1), ∀0 ≤i < t.

From the convexity of per stage expected cost in Lemma B.1, we know that, ∀1 ≤t1 < H,
ft1 (M; M1) is a convex function of M. It follows that

ft1 (M; M1) ≥ft1 (M′; M1) + Tr

(∇ft1 (M′; M1))⊤(M −M′)

, ∀1 ≤t1 < H,
(96)

where the gradient ∇ft1 (M′; M1) is taken w.r.t. the first argument M′.

From stong convexity of per stage expected cost in Lemma B.2, we know that, ∀H ≤t2 ≤T,
ft2 (M; M1) is min
n
µσ2

2 , µγ2σ2

64κ10
o
-strongly convex function of M. Therefore, it follows

ft2 (M; M1) ≥ft2 (M′; M1) + Tr

(∇ft2 (M′; M1))⊤(M −M′)

(97)

+ min
µσ2

2 , µγ2σ2

64κ10


∥M −M′∥2
F , ∀H ≤t2 ≤T.

28


---Page Break---
Sum up (96) from t1 = 1 to t1 = H −1 and (97) from t2 = H to t2 = T, we obatin

H−1
X

t1=1
ft1 (M; M1) +

T
X

t2=H
ft1 (M; M1) ≥

H−1
X

t1=1
ft1 (M′; M1) +

T
X

t2=H
ft1 (M′; M1)
(98)

+ Tr





 

∇

 H−1
X

t1=1
ft1 (M′; M1) +

T
X

t2=H
ft1 (M′; M1)

!!⊤

(M −M′)





+ eµ

2 ∥M −M′∥2
F .

Inequality (13) follows directly by noting that

CT (M; M1) =

H−1
X

t1=1
ft1 (M; M1) +

T
X

t2=H
ft1 (M; M1) ,
(99)

∇CT (M′; M1) = ∇

 H−1
X

t1=1
ft1 (M′; M1) +

T
X

t2=H
ft1 (M′; M1)

!

.
(100)

E
Proof of Lemma 3

Note that ∇C0 (M; M1) = ∇ME [ct (x0, u0)] = ∇ME [ct (x0, −Kx0)] = 0. Therefore, inequality
(14) follows directly by combining the norm triangular inequalityand the results in Lemma B.6.

∥∇CT (M; M1) −∇CT (M′; M2)∥F
(101)

≤

T
X

t=1

Ex0,{∆i∼Di(M1)}0≤i<t,{wi}0≤i<t

h
∇Mct

M; {∆i}0≤i<t
i

−Ex0,{∆′
i∼Di(M2)}0≤i<t,{wi}0≤i<t

h
∇M′ct

M′; {∆′
i}0≤i<t
i
F

≤

 T
X

t=1
λt

!

∥M −M′∥F +

T
X

t=1

 

νt

t−1
X

i=0
εi

!

∥M1 −M2∥F

=

 T
X

t=1
λt

!

∥M −M′∥F +

T −1
X

t=0
εt

 
T
X

i=t+1
νi

!

∥M1 −M2∥F .

F
Proof of Lemma 4

Fix any M, M′ ∈M, the optimality condition to (15) implies that

∇CT (Φ (M) ; M) = 0, ∇CT (Φ (M′) ; M′) = 0,
(102)

where the gradients are again taken w.r.t. the first argument in the function CT (·, ·). Observe the fact

0 = Tr
 
0T · (Φ (M) −Φ (M′))


= Tr ((∇CT (Φ (M) ; M) −∇CT (Φ (M′) ; M′)) · (Φ (M) −Φ (M′))) .
(103)

Adding and subtracting the term ∇CT (Φ (M) ; M′) implies the equality

Tr ((∇CT (Φ (M) ; M′) −∇CT (Φ (M) ; M)) · (Φ (M) −Φ (M′)))
(104)

= Tr ((∇CT (Φ (M) ; M′) −∇CT (Φ (M′) ; M′)) · (Φ (M) −Φ (M′))) .

Recall that CT (Φ (M) ; M′) is eµ-strongly convex w.r.t. Φ (M), it follows

Tr ((∇CT (Φ (M) ; M′) −∇CT (Φ (M′) ; M′)) · (Φ (M) −Φ (M′)))
(105)

≥eµ ∥Φ (M) −Φ (M′)∥2
F .

29


---Page Break---
Meanwhile, applying Lemma 3 to the left hand side of (104) gives

Tr ((∇CT (Φ (M) ; M′) −∇CT (Φ (M) ; M)) · (Φ (M) −Φ (M′)))
(106)

≤

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!

∥M −M′∥F ∥Φ (M) −Φ (M′)∥F .

Combining (104), (105) and (106), it follows

∥Φ (M) −Φ (M′)∥F ≤

PT −1
t=0

εt
PT
i=t+1 νi


eµ
∥M −M′∥F .
(107)

We note that Mn+1 = Φ (Mn) by the definition of (15), and MP S = Φ
 
MP S
by the definition of
performative stability. Applying (107) yields

Mn −MP S
F =
Φ (Mn−1) −Φ
 
MP S
F ≤

PT −1
t=0

εt
PT
i=t+1 νi


eµ

Ml−1 −MP S
F

(108)

≤





PT −1
t=0

εt
PT
i=t+1 νi


eµ





n
M0 −MP S
F .

Setting
 PT −1
t=0 (εt
PT
i=t+1 νi)
eµ

n M0 −MP S
F to be at most ρ and solving for n completes the

proof.

G
Convergence Analysis of Algorithm 1

G.1
Non-asymptomatic Convergence Analysis for Algorithm 1 with General Step Sizes

We have the following Lemma 5 regarding the convergence results of the proposed RSGD in
Algorithm 1.

Lemma 5 Under A1-A6. Consider a sequence of non-negative step sizes {ηn, n ≥0} satisfy

sup
n≥0
ηn ≤min








eµ −PT −1
t=0

εt
PT
i=t+1 νi


2
PT
t=1 λt + PT −1
t=0

εt
PT
i=t+1 νi
2 ,
2

eµ −PT −1
t=0

εt
PT
i=t+1 νi








, (109)

and

ηn
ηn+1
≤

 

1 + 1

2

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!

ηn+1

!

,
(110)

for any n ≥0. Then, the iterates generated by RSGD admit the following bound for any N ≥1:

E
hMN −MP S2
F

i
≤

N−1
Y

n=0

 

1 −ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

E
hM0 −MP S2
F

i
(111)

+
4ηN−1T PT
t=1 ϑ2
t
eµ −PT −1
t=0

εt
PT
i=t+1 νi
,

where ϑt = κ3G
  
HW + κ2
κ ∥B∥βt + 1

(x0αt + c3βt)+GHWM
 
κ3βt + 1

, ∀1 ≤t ≤T.

30


---Page Break---
Proof.
Since projecting onto a convex set can only bring two iterates closer together, it follows that

E
hMn+1 −MP S2
F

i
≤E
Mn −ηn∇JT

Mn; {∆t}0≤t<T

−MP S
2
(112)

≤I1 + I2 + I3,

where

I1 = E
hMn −MP S2
F

i
,
(113)

I2 = −2ηnE
h
Tr
 
Mn −MP S⊤∇JT

Mn; {∆t}0≤t<T
i
,
(114)

I3 = η2
nE
∇JT

Mn; {∆t}0≤t<T

2

F


,
(115)

and ∆t ∼Dt (Mn) , ∀0 ≤t < T.

We first analyze the term I2. Let En [·] be the conditional expectation on Mn. We have

En
h
Tr
 
Mn −MP S⊤∇JT

Mn; {∆t}0≤t<T
i
= Tr
 
Mn −MP S⊤∇CT (Mn; Mn)


(116)

= Tr
 
Mn −MP S⊤ 
∇CT (Mn; Mn) −∇CT
 
Mn; MP S

+ Tr
 
Mn −MP S⊤ 
∇CT
 
Mn; MP S
−∇CT
 
MP S; MP S
,

where we use the fact that ∇CT
 
MP S; MP S
= 0 in the last equality of (116).

Applying the Cauchy-Schwarz inequality and Lemma 3, we obtain

Tr
 
Mn −MP S⊤ 
∇CT (Mn; Mn) −∇CT
 
Mn; MP S
(117)

≥−
Mn −MP S
F

 T −1
X

t=0

 

εt

T
X

i=t+1
νi

!
Mn −MP S
F

!

= −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!
Mn −MP S2
F .

Meanwhile, based on strongly convexity of CT in Lemma 2, we have

Tr
 
Mn −MP S⊤ 
∇CT
 
Mn; MP S
−∇CT
 
MP S; MP S
≥eµ
Mn −MP S2
F .

(118)

Substitute (117) and (118) into (116) and take the full expectation, it follows that I2 satisfies

I2 ≤−2ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!

E
hMn −MP S2
F

i
.
(119)

We next analyze the term I3. We observe the following equivalent expression for I3
∇JT

Mn; {∆t}0≤t<T

2

F =
∇JT

Mn; {∆t}0≤t<T

−∇CT (Mn; Mn)
(120)

+ ∇CT (Mn; Mn) −∇CT
 
MP S; MP S2
F .

Therefore,

E
∇JT

Mn; {∆t}0≤t<T

2

F


(121)

≤2E
∇JT

Ml; {∆t}0≤t<T

−∇CT (Mn; Mn)

2

F



+ 2E
h∇CT (Mn; Mn) −∇CT
 
MP S; MP S2
F

i
.

31


---Page Break---
According to Lemma C.1, an upper bound of the variance of ∇JT

Mn; {∆t}0≤t<T

is given by
(94). Moreover, based on Lemma 3, we have

E
h∇CT (Mn; Mn) −∇CT
 
MP S; MP S2
F

i
(122)

≤

 T
X

t=1
λt +

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!2

E
hMn −MP S2
F

i
.

Substitute (94) and (122) back into (121), it follows

I3 ≤2η2
nT

T
X

t=1
ϑ2
t + 2η2
n

 T
X

t=1
λt +

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!2

E
hMn −MP S2
F

i
.
(123)

Now subtitute (119) and (123) back into (112), we obtain

E
hMn+1 −MP S2
F

i
≤E
hMn −MP S2
F

i
−2ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!

(124)

· E
hMn −MP S2
F

i
+ 2η2
n

 T
X

t=1
λt +

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!2

E
hMn −MP S2
F

i
+ 2η2
nT

T
X

t=1
ϑ2
t

=



1 −2ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!

+ 2η2
n

 T
X

t=1
λt +

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!2

E
hMn −MP S2
F

i

+ 2η2
nT

T
X

t=1
ϑ2
t.

Let
ηn
be
choosen
such
that
ηn

eµ −PT −1
t=0

εt
PT
i=t+1 νi

≥

2η2
n
PT
t=1 λt + PT −1
t=0

εt
PT
i=t+1 νi
2
. Then inequality (124) is reduced to

E
hMn+1 −MP S2
F

i
≤

 

1 −ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

E
hMn −MP S2
F

i
(125)

+ 2η2
nT

T
X

t=1
ϑ2
t.

Therefore,

E
hMN −MP S2
F

i
≤

N−1
Y

n=0

 

1 −ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

E
hM0 −MP S2
F

i
(126)

+

N−1
X

n=0

N−1
Y

i=n+1

 

1 −ηi

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

2η2
nT

T
X

t=1
ϑ2
t.

Let
η0
<
2
eµ−PT −1
t=0 (εt
PT
i=t+1 νi).
Based
on
Lemma
6
in
[14],
if
ηn
ηn+1
≤

1 + 1

2

eµ −PT −1
t=0

εt
PT
i=t+1 νi

ηn+1

for any n ≥0, then

N−1
X

n=0
η2
n

N−1
Y

i=n+1

 

1 −ηi

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

≤
2ηN−1

eµ −PT −1
t=0

εt
PT
i=t+1 νi
, ∀N ≥1.

(127)

32


---Page Break---
Substitute (127) into (126), we obtain the desired result.
□

Based on the step size requirements in Lemma 5, RSGD converges when the aggregated sensitivity
PT −1
t=0

εt
PT
i=t+1 νi

is strictly below the threshold eµ defined in Lemma 4, i.e., the same sufficient

condition that guarantees the existence of MP S.

The first term on the R. H. S. of (111) decays sub-geometrically and is scaled by the initial error
E
hM0 −MP S2
F

i
. The second term is a fluctuation term that only depends on the variance of the

stochastic gradient, which decays at a slower rate as O (ηN−1).

G.2
Convergence Analysis of Algorithm 1 with Diminishing Step Sizes

We select the diminishing step sizes ηn =
ϕ1
n+ϕ2 , ∀n ≥0, where ϕ1 and ϕ2 satisfy (17) and (18)
simultaneously. We can verify that condition (109) in Lemma 5 is reduced to (17) in Theorem 1.
Besides, condition (110) in Lemma 5 can be further simplifed as

ηn
ηn+1
=

ϕ1
n+ϕ2

ϕ1
n+1+ϕ2
= 1 +
1
n + ϕ2
≤

 

1 + 1

2

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!
ϕ1
n + 1 + ϕ2

!

,
(128)

which is equivalent to

n + 1 + ϕ2

n + ϕ2
≤1

2

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!

ϕ1.
(129)

Note that ∀n ≥0, to guarantee (129) holds, we only need to let

2

eµ −PT −1
t=0

εt
PT
i=t+1 νi
 ≤ϕ1ϕ2

1 + ϕ2
=
ϕ1
1 +
1
ϕ2
,
(130)

which coincides with condition (18) in Theorem 1.

Lastly, not that 1 + x ≤ex for all x > 0, we can verify the first term in R.H.S. of (111) can be upper
bounded as

N−1
Y

n=0

 

1 −ηn

 

eµ −

T −1
X

t=0

 

εt

T
X

i=t+1
νi

!!!

E
hM0 −MP S2
F

i

≤e−PN−1
n=0 ηn(eµ−PT −1
t=0 (εt
PT
i=t+1 νi))E
hM0 −MP S2
F

i

= e−PN−1
n=0
ϕ1
n+ϕ2 (eµ−PT −1
t=0 (εt
PT
i=t+1 νi))E
hM0 −MP S2
F

i

≤e−PN
n=1
ϕ1

n (eµ−PT −1
t=0 (εt
PT
i=t+1 νi))E
hM0 −MP S2
F

i
.
(131)

Therefore, Theorem 1 follows by combining (130) and (131).

H
Proof of Proposition 1

Since ∥At∥≤1 −γ + κ2ξt
≤ζ
< 1, ∀0 ≤t < T, it follows directly that αt
=
Qt−1
i=0
 
1 −γ + κ2ξi

≤Qt−1
i=0 ζ = ζt and βt = Pt−1
i=0
Qt−1
j=i+1
 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

≤

1−ζt

1−ζ .

Further calculation yeilds νt = (c1 + c2βt) (x0αt + c3βt) ≤

c1 + c2
1
1−ζ
 
x0 + c3
1
1−ζ

.

As a result, the sufficient condition (16) for the existence and uniqueness of MP S is reduced to

c1 + c2
1
1 −ζ

 
x0 + c3
1
1 −ζ

 T −1
X

t=0
(εt (T −t)) < eµ.
(132)

33


---Page Break---
This is equivalent to

c1 + c2
1
1−ζ
 
x0 + c3
1
1−ζ
 PT −1
t=0
T −t
T −H+1εt
<
µ, where µ
=

min
n
µσ2

2 , µσ2γ2

64κ10
o
. As a result, to guarantee condition (132) holds, it is sufficient to ensure that

T −1
X

t=0
εt <

1 −H

T

 
c1 + c2
1
1 −ζ

−1 
x0 + c3
1
1 −ζ

−1

µ.

Proposition 1 is thus proved by choosing ϕ =

c1 + c2
1
1−ζ
−1 
x0 + c3
1
1−ζ
−1
.

I
Proof of Proposition 2

Since eζ ≤∥At∥≤1 −γ + κ2ξt, ∀0 ≤t < T, for some a positive constant eζ > 1,
it follows directly that αt
=
Qt−1
i=0
 
1 −γ + κ2ξi

≥
αt
≥
Qt−1
i=0 eζ
=
eζt and βt
=
Pt−1
i=0
Qt−1
j=i+1
 
1j<t
 
1 −γ + κ2ξj

+ 1j=t

≥
eζt−1

eζ−1 .

Further calculation yeilds νt = (c1 + c2βt) (x0αt + c3βt) ≥

c1 + c2eζt−1 
x0eζt + c3eζt−1
.
Therefore,

T
X

i=t+1
νi ≥

c1x0 + (c1c3 + c2c3 + c3x0) eζ−1
T
X

i=t+1
eζi

=

c1x0 + (c1c3 + c2c3 + c3x0) eζ−1 eζT −t −1

eζ −1
.
(133)

As a result, to guarantee the sufficient condition (16) can be satifised, we must have

T −1
X

t=0

εt
T −H + 1


c1x0 + (c1c3 + c2c3 + c3x0) eζ−1 eζT −t −1

eζ −1
< µ.
(134)

This means that

εt <
(T −H + 1)

eζ −1


c1x0 + (c1c3 + c2c3 + c3x0) eζ−1 ·
µ
eζT −t −1
.
(135)

As a result, Proposition 2 is proved by letting ϕ =
eζ−1
c1x0+(c1c3+c2c3+c3x0)eζ−1 .

34


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer:[Yes]
Justification: The abstract and introduction accurately reflect the paper’s contributions and
scope.
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
Answer:[Yes]
Justification: We have discussed the limitations of the work in the Conclusion section.
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

35


---Page Break---
Justification: The paper has provided the full set of assumptions and a complete and correct
proof.

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
Justification: We have disclosed all the information needed to reproduce the main experi-
mental results of the paper to the extent that it affects the main claims and/or conclusions of
the paper.

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

36


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [No]
Justification: The paper does not provide open access to the data and code.
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
Justification: The paper has specified all the training and test details (e.g., data splits,
hyperparameters).
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
Justification: The paper has provided appropriate information about the statistical signifi-
cance of the experiments.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

37


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
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

Answer:[No]

Justification: The paper does not provide the type of compute workers, memory, or time of
execution.

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

Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics.

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

Justification: There is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

38


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
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
Justification: The paper poses no such risks.
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
Answer: [NA]
Justification: The paper does not use existing assets.
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

39


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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

40


---Page Break---
