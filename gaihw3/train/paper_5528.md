Online Posterior Sampling with a Diffusion Prior

Branislav Kveton
Adobe Research∗
Boris Oreshkin
Amazon
Youngsuk Park
AWS AI Labs
Aniket Deshmukh
AWS AI Labs
Rui Song
Amazon

Abstract

Posterior sampling in contextual bandits with a Gaussian prior can be implemented
exactly or approximately using the Laplace approximation. The Gaussian prior is
computationally efficient but it cannot describe complex distributions. In this work,
we propose approximate posterior sampling algorithms for contextual bandits with
a diffusion model prior. The key idea is to sample from a chain of approximate
conditional posteriors, one for each stage of the reverse diffusion process, which
are obtained by the Laplace approximation. Our approximations are motivated by
posterior sampling with a Gaussian prior, and inherit its simplicity and efficiency.
They are asymptotically consistent and perform well empirically on a variety of
contextual bandit problems.

1
Introduction

A multi-armed bandit [26, 6, 29] is an online learning problem where an agent sequentially interacts
with an environment over n rounds with the goal of maximizing its rewards. In each round, it takes an
action and receives its stochastic reward. The mean rewards of the actions are unknown a priori and
must be learned. This leads to the exploration-exploitation dilemma: explore actions to learn about
them or exploit the action with the highest estimated reward. Bandits have been successfully applied
to problems where uncertainty modeling and adaptation to it are beneficial, such as in recommender
systems [31, 53, 24, 34] and hyper-parameter optimization [33].

Contextual bandits [28, 31] with linear [13, 1] and generalized linear models (GLMs) [16, 32, 2, 25]
have become popular due to the their flexibility and efficiency. The features in these models can be
hand-crafted or learned from historic data [39], and the models can be also updated incrementally
[1, 23]. While the original algorithms for linear and GLM bandits were based on upper confidence
bounds (UCBs) [13, 1, 16], Thompson sampling (TS) is more popular in practice [11, 3, 41, 43]. The
key idea in TS is to explore by sampling from the posterior distribution of model parameter θ∗. TS
uses the prior knowledge about θ∗to speed up exploration [11, 39, 35, 9, 20, 19, 5]. When the prior
is a multivariate Gaussian, the posterior of θ∗can be updated and sampled from efficiently [11]. This
prior has a limited expressive power, because it cannot even represent multimodal distributions. To
address this, we study posterior sampling with a diffusion prior. The main benefit of such priors is
that they can represent complex distributions and be learned from data.

We make the following contributions. First, we propose novel posterior sampling approximations
for linear models and GLMs with a diffusion model prior. The key idea is to sample from a chain of
approximate conditional posteriors, one for each stage of the reverse process, which are estimated in
a closed form. In linear models, each conditional is a product of two Gaussians, representing prior
knowledge and diffused evidence (Theorem 2). In GLMs, each conditional is obtained by a Laplace
approximation, which mixes prior knowledge and evidence (Theorem 4). Our approximations are
motivated by posterior sampling with a Gaussian prior, and inherit its simplicity and efficiency. In
prior works (Section 7), posterior sampling is implemented using the likelihood score, which tends
to infinity as the number of observations increases; and therefore causes instability. We combine

∗The work was done at AWS AI Labs.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the likelihood with the conditional prior, in each stage of the diffusion model, using the Laplace
approximation. The resulting posterior concentrates at a single point and can be easily sampled from,
even if the likelihood score tends to infinity. We propose an efficient and asymptotically consistent
implementation of this idea, using a stage-wise Laplace approximation with diffused evidence.

Our second contribution is in theory. We properly derive our posterior approximations (Theorems 2
and 4) and show their asymptotic consistency (Theorem 3). The key idea in the proof of Theorem 3
is that the conditional posteriors concentrate at a scaled unknown model parameter as the number
of observations increases. While this claim is asymptotic, it is an expected property of a posterior
distribution. Many prior works, such as Chung et al. [12], do not have this property. All of our main
results rely on our novel approximation of clean samples by scaled diffused samples (Section 4.3).
The most challenging part of the analysis is Theorem 3, where we analyze an asymptotic behavior of
a chain of T dependent random vectors.

Our last contribution is an empirical evaluation on contextual bandits. We focus on bandits because
the ability to represent all levels of uncertainty precisely is critical for exploration. Our experiments
show that a score-based method fails to do so (Section 6.2). Note that our posterior approximations
are general and not restricted to bandits.

2
Setting

We start with introducing our notation. Random variables are capitalized, except for Greek letters like
θ. We denote the marginal and conditional probabilities under probability measure p by p(X = x)
and p(X = x | Y = y), respectively. When the random variables are clear from context, we write
p(x) and p(x | y). We denote by Xn:m and xn:m a collection of random variables and their values,
respectively. For a positive integer n, we define [n] = {1, . . . , n}. The indicator function is 1{·}.
The i-th entry of vector v is vi. If the vector is already indexed, such as vj, we write vj,i. We denote
the maximum and minimum eigenvalues of matrix M ∈Rd×d by λ1(M) and λd(M), respectively.

The posterior sampling problem can be formalized as follows. Let θ∗∈Θ be an unknown model
parameter and Θ ⊆Rd be the space of model parameters. Let h = {(ϕℓ, yℓ)}ℓ∈[N] be the history of
N noisy observations of θ∗, where ϕℓ∈Rd is the feature vector for yℓ∈R. We assume that

yℓ= g(ϕ⊤
ℓθ∗) + εℓ,
(1)

where g : R →R is the mean function and εℓis an independent zero-mean σ2-sub-Gaussian noise
for σ > 0. Let p(h | θ∗) be the likelihood of observations in history h under model parameter θ∗and
p(θ∗) be its prior probability. By Bayes’ rule, the posterior distribution of θ∗given h is

p(θ∗| h) ∝p(h | θ∗) p(θ∗).
(2)

We want to sample from p(· | h) efficiently when the prior distribution is represented by a diffusion
model. As a stepping stone, we review existing posterior formulas for multivariate Gaussian priors.
This motivates our solution for diffusion model priors.

2.1
Linear Model

The posterior of θ∗in linear models can be derived as follows.

Assumption 1. Let g in (1) be an identity and εℓ∼N(0, σ2). Then the likelihood of h under model
parameter θ∗is p(h | θ∗) ∝exp[−PN
ℓ=1(yℓ−ϕ⊤
ℓθ∗)2/(2σ2)].

Let p(θ∗) = N(θ∗; θ0, Σ0) be the prior distribution, where θ0 ∈Rd and Σ0 ∈Rd×d are the prior
mean and covariance, respectively. Then p(θ∗| h) ∝N(θ∗; ˆθ, ˆΣ) [10], where

ˆθ = ˆΣ(Σ−1
0 θ0 + ¯Σ−1¯θ) ,
ˆΣ = (Σ−1
0
+ ¯Σ−1)−1 ,
(3)

¯θ = σ−2 ¯Σ PN
ℓ=1 ϕℓyℓ, and ¯Σ−1 = σ−2 PN
ℓ=1 ϕℓϕ⊤
ℓ. Therefore, the posterior of θ∗is a product of
two multivariate Gaussians: N(θ0, Σ0) represents prior knowledge about θ∗and N(¯θ, ¯Σ) represents
empirical evidence.

2


---Page Break---
Algorithm 1 IRLS: Iteratively reweighted least squares.

1: Input: Prior parameters θ0 and Σ0, history of observations h = {(ϕℓ, yℓ)}ℓ∈[N]

2: Initialize ˆθ ∈Rd

3: repeat
4:
for stage ℓ= 1, . . . , N do
5:
zℓ←ϕ⊤
ℓˆθ + (yℓ−g(ϕ⊤
ℓˆθ))/˙g(ϕ⊤
ℓˆθ)

6:
ˆΣ ←

Σ−1
0
+ PN
ℓ=1 ˙g(ϕ⊤
ℓˆθ)ϕℓϕ⊤
ℓ
−1

7:
ˆθ ←ˆΣ

Σ−1
0 θ0 + PN
ℓ=1 ˙g(ϕ⊤
ℓˆθ)ϕℓzℓ


8: until ˆθ converges

9: Output: Posterior mean ˆθ and covariance ˆΣ

2.2
Generalized Linear Model

Generalized linear models (GLMs) [36] extend linear models (Section 2.1) to non-linear monotone
mean functions g in (1). For instance, in logistic regression, g(u) = 1/(1 + exp[−u]) is a sigmoid.
The likelihood of observations in GLMs has the following form [25].
Assumption 2. Let h = {(ϕℓ, yℓ)}ℓ∈[N] be a history of N observations under mean function g and

the corresponding noise. Then log p(h | θ∗) ∝PN
ℓ=1 yℓϕ⊤
ℓθ∗−b(ϕ⊤
ℓθ∗) + c(yℓ), where c is a real
function and b is a function whose derivative is the mean function, ˙b = g.

The posterior distribution of θ∗in GLMs does not have a closed form in general [10]. Therefore, it is
often approximated by the Laplace approximation. Let the prior distribution of the model parameter
be p(θ∗) = N(θ∗; θ0, Σ0), as in Section 2.1. Then the Laplace approximation is N(ˆθ, ˆΣ), where ˆθ is
the maximum a posteriori (MAP) estimate of θ∗and ˆΣ is the corresponding covariance. Note that the
Laplace approximation can be applied to non-Gaussian priors.

The MAP estimate ˆθ can be obtained by iteratively reweighted least squares (IRLS) [51], which we
present in Algorithm 1. IRLS is a Newton-type algorithm that computes ˆθ iteratively (line 6). The
convergence rate is fast due to the strong convexity of the problem. In the updates, ˙g is the derivative
of the mean function and the pseudo-observation zℓ(line 4) plays the role of the observation yℓin
Section 2.1. The IRLS solution has a similar structure to (3). Specifically, N(ˆθ, ˆΣ) is a product of
two multivariate Gaussians, representing prior knowledge about θ∗and empirical evidence.

2.3
Towards Diffusion Model Priors

The assumption that p(θ∗) = N(θ∗; θ0, Σ0) is limiting, for instance because it precludes multimodal
priors. We relax it by representing p(θ∗) by a diffusion model, which we call a diffusion model prior.
We propose an efficient posterior sampling approximation for this prior, where the prior and empirical
evidence are mixed similarly to (3) and IRLS. We review diffusion models next.

3
Diffusion Models

Diffusion models [45, 18] are generative models trained by diffusing samples from unknown and
hard to represent distributions. They can be viewed in multiple ways [48]. We adopt the probabilistic
formulation and presentation of Ho et al. [18]. A diffusion model is a graphical model with T stages
indexed by t ∈[T]. Each stage t is associated with a latent variable St ∈Rd. A sample from the
model is represented by an observed variable S0 ∈Rd. We visualize a diffusion model in Figure 1.
In the forward process, a clean sample s0 is diffused through a sequence of variables S1, . . . , ST .
This process is used to learn the reverse process, where the clean sample s0 is generated through a
sequence of variables ST , . . . , S0. To sample s0 from the posterior (Section 4), we add a random
variable H that represents partial information about s0. We introduce forward and reverse diffusion
processes next. Learning of the reverse process is described in Appendix B. While this is a critical
component of diffusion models, it is not necessary to introduce our posterior approximations.

3


---Page Break---
Forward process (probability measure q)
Reverse process (probability measure p)
ST ←ST −1 ←· · · ←S1 ←S0 →H
ST →ST −1 →· · · →S1 →S0 →H

Figure 1: Graphical models of the forward and reverse processes in the diffusion model. The variable
H represents partial information about S0.

Forward process. In the forward process, a clean sample s0 is diffused through a chain of latent
variables S1, . . . ST (Figure 1). We denote the probability measure under this process by q and define
its joint probability distribution as

q(s1:T | s0) = QT
t=1 q(st | st−1) ,
∀t ∈[T] : q(st | st−1) = N(st; √αtst−1, βtId) ,
(4)

where q(st | st−1) is the conditional density of mapping a less diffused st−1 to a more diffused st.
The diffusion rate is set by parameters αt ∈(0, 1) and βt = 1 −αt. We also define ¯αt = Qt
ℓ=1 αℓ.
The forward process is sampled as follows. First, a clean sample s0 is chosen. Then St ∼q(· | st−1)
are sampled, starting from t = 1 to t = T.

Reverse process. In the reverse process, a clean sample s0 is generated through a chain of variables
ST , . . . , S0 (Figure 1). We denote the probability measure under this process by p and define its joint
probability distribution as

p(s0:T ) = p(sT )

T
Y

t=1
p(st−1 | st) ,
(5)

p(sT ) = N(sT ; 0d, Id) ,
∀t ∈[T] : p(st−1 | st) = N(st−1; µt(st), Σt) ,

where p(st−1 | st) is the conditional density of mapping a more diffused st to a less diffused st−1.
The function µt predicts the mean of St−1 | st and is learned (Appendix B). As in Ho et al. [18],
we keep the covariance fixed at Σt = ˜βtId, where ˜βt is defined in (14) in Appendix B. This is also
known as a stable diffusion. We make this assumption only to simplify exposition. All our derivations
in Section 4 hold when Σt is learned, for instance as in Bao et al. [8].

This process is called reverse because it is learned by reversing the forward process. The reverse
process is sampled as follows. First, a diffused sample ST ∼p is chosen. After that, St−1 ∼p(· | st)
are sampled, starting from t = T to t = 1.

4
Posterior Sampling

This section is organized as follows. In Section 4.1, we show how to sample from a chain of random
variables conditioned on observations. In Sections 4.2 and 4.4, we specialize this to the observation
models in Section 2.

4.1
Chain Model Posterior

Let h = {(ϕℓ, yℓ)}ℓ∈[N] denote a history of N observations (Section 2) and H be the corresponding
random variable. In this section, we assume that h is fixed. The Markovian structure of the reverse
process (Figure 1) implies that the joint probability distribution conditioned on h factors as

p(s0:T | h) = p(sT | h)

T
Y

t=1
p(st−1 | st, h) .

Therefore, p(s0:T | h) can be sampled from efficiently by first sampling from p(sT | h) and then
from T conditional distributions p(st−1 | st, h). We derive these next.
Lemma 1. Let p be a probability measure over the reverse process (Figure 1). Then

p(sT | h) ∝
R

s0 p(h | s0) p(s0 | sT ) ds0 p(sT ) ,

∀t ∈[T] \ {1} : p(st−1 | st, h) ∝
R

s0 p(h | s0) p(s0 | st−1) ds0 p(st−1 | st) ,

p(s0 | s1, h) ∝p(h | s0) p(s0 | s1) .

Proof. The claim is proved in Appendix A.1.

4


---Page Break---
Algorithm 2 LaplaceDPS: Laplace posterior sampling with a diffusion model prior.

1: Input: Diffusion model parameters (µt, Σt)t∈[T ], history of observations h

2: Initial sample ST ∼N(ˆµT +1(h), ˆΣT +1(h))
3: for stage t = T, . . . , 1 do
4:
St−1 ∼N(ˆµt(St, h), ˆΣt(h))

5: Output: Posterior sample S0

4.2
Linear Model Posterior

Now we specialize Lemma 1 to the diffusion model prior (Section 3) and linear models (Section 2.1).
The prior distribution is the reverse process in (5),

p(sT ) = N(sT ; 0d, Id) ,
p(st−1 | st) = N(st−1; µt(st), Σt) .

The term p(h | s0) is the likelihood of observations in Assumption 1. The main challenge in using
the lemma are potentially complex conditional densities of clean samples p(s0 | ST ) and p(s0 | st).
To get around this issue, we make an additional assumption and then discuss it in Section 4.3.
Theorem 2. Let p be a probability measure over the reverse process (Figure 1). Let ¯θ and ¯Σ−1 be
defined as in (3). Suppose that
R

s0 p(h | s0) p(s0 | st) ds0 ∝p(h | st/√¯αt)
(6)

holds for all t ∈[T]. Then p(sT | h) ∝N(sT ; ˆµT +1(h), ˆΣT +1(h)), where

ˆµT +1(h) = ˆΣT +1(h)(Id 0d + ¯Σ−1¯θ/√¯αT ) ,
(7)
ˆΣT +1(h) = (Id + ¯Σ−1/¯αT )−1 .

For t ∈[T], we have p(st−1 | st, h) ∝N(st−1; ˆµt(st, h), ˆΣt(h)), where

ˆµt(st, h) = ˆΣt(h)(Σ−1
t µt(st) + ¯Σ−1¯θ/√¯αt−1) ,
(8)
ˆΣt(h) = (Σ−1
t
+ ¯Σ−1/¯αt−1)−1 .

Proof. The proof is in Appendix A.2. It has four steps. First, we fix stage t and apply approximation
(6). Second, we rewrite the likelihood as in (3). Third, we reparameterize it as a function of st. At
the end, we combine the likelihood with the Gaussian prior using Lemma 6 in Appendix A.5.

The algorithm that samples from the posterior distribution in Theorem 2 is presented in Algorithm 2.
We call it Laplace diffusion posterior sampling (LaplaceDPS) because its generalization to GLMs
uses the Laplace approximation. LaplaceDPS samples from a chain of products of two distributions:
one distribution represents the pre-trained diffusion model and does not depend on history h, and the
other represents the history h. The sampling is implemented as follows. The initial variable ST is
sampled conditioned on h (line 2) from the distribution in (7). This distribution is a product of the
h-independent initial prior N(0d, Id) and the h-dependent distribution of the diffused evidence up to
stage T, N(√¯αT ¯θ, ¯αT ¯Σ). All remaining variables, St−1 for t ∈[T], are sampled conditioned on
st and evidence h (line 3) from the distribution in (8). This is again a product of the h-independent
conditional prior N(µt(st), Σt), from the pre-trained model, and the h-dependent distribution of the
diffused evidence up to stage t −1, N(√¯αt−1¯θ, ¯αt−1 ¯Σ). The final variable S0 represents a clean
sample. When compared to Section 2, the prior and evidence are mixed conditionally in a T-stage
chain. This increases computational cost, as discussed in Section 8.

4.3
Key Approximation in Theorem 2

Now we motivate our assumption in (6). Simply put, we assume that s0 = st/√¯αt, where s0 is a
clean sample and st is the corresponding diffused sample in stage t. This is motivated by the forward
process, which relates st and s0 as st = √¯αts0 + √1 −¯αtεt, where εt ∼N(0d, Id) is a standard
Gaussian noise [18]. After rearranging, we get s0 = (st −√1 −¯αtεt)/√¯αt, and therefore s0 can
be viewed as a random variable with mean st/√¯αt. The consequence of (6) is that the likelihood

5


---Page Break---
Algorithm 3 Contextual Thompson sampling.

1: for round k = 1, . . . , n do
2:
Sample ˜θk ∼p(· | hk), where p(· | hk) is the posterior distribution in (2)
3:
Take action ak ←arg max a∈A r(xk, a; ˜θk) and observe reward yk

becomes a function of scaled st, which yields a closed form when multiplied by the conditional prior,
which is also a function of st. Our approximation can be also viewed as the Tweedie’s formula used
in Chung et al. [12] where the score component is neglected.

Our approximation has several notable properties. First,
p

(1 −¯αt)/¯αt →0 as t →1. Therefore,
it becomes more precise in later stages of the reverse process. Second, in the absence of evidence
h, the approximation vanishes, and all posterior distributions in Theorem 2 reduce to the priors in
(5). Finally, as the number of observations increases, sampling from the posterior in Theorem 2 is
asymptotically consistent.

Theorem 3. Fix θ∗∈Rd. Let ˜θ ←LaplaceDPS((µt, Σt)t∈[T ], h), where h = {(ϕℓ, yℓ)}ℓ∈[N] is a
history of N observations. Suppose that λd(¯Σ−1) →∞as N →∞, where ¯Σ is defined in (3). Then

P

limN→∞∥˜θ −θ∗∥2 = 0

= 1.

Proof. The proof is in Appendix A.3. The key idea is that the conditional posteriors in (7) and (8)
concentrate at a scaled unknown model parameter θ∗as the number of observations increases, which
we formalize as λd(¯Σ−1) →∞.

The bound in Theorem 3 can be interpreted as follows. The sampled parameter ˜θ approaches the true
unknown parameter θ∗as the number of observations N increases. To guarantee that the posterior
shrinks uniformly in all directions, we assume that the number of observations in all directions grows
linearly with N. This is akin to assuming that λd(¯Σ−1) = Ω(N). This lower bound can be attained
in linear models by getting observations according to the D-optimal design [38]. Since our proof is
asymptotic, we can neglect some finite-time errors and this simplifies the argument.

4.4
GLM Posterior

The Laplace approximation in GLMs (Section 2.2) naturally generalizes the exact posterior distribu-
tion in linear models (Section 2.1). We generalize Theorem 2 to GLMs along the same lines.

Theorem 4. Let p be a probability measure over the reverse process (Figure 1). Suppose that (6)
holds for all t ∈[T]. Then p(sT | h) ∝N(sT ; ˆµT +1(h), ˆΣT +1(h)), where

ˆµT +1(h) = √¯αT ˙θT +1 ,
ˆΣT +1(h) = ¯αT ˙ΣT +1 ,
˙θT +1, ˙ΣT +1 ←IRLS(0d, Id/¯αT , h) .

For t ∈[T], we have p(st−1 | st, h) ∝N(st−1; ˆµt(st, h), ˆΣt(h)), where

ˆµt(st, h) = √¯αt−1 ˙θt ,
ˆΣt(h) = ¯αt−1 ˙Σt ,
˙θt, ˙Σt ←IRLS(µt(st)/√¯αt−1, Σt/¯αt−1, h) .

Proof. The proof is in Appendix A.4. It has four steps. First, we fix stage t and apply approximation
(6). Second, we reparameterize the prior, from a function of st to a function of st/√¯αt. Third, we
combine the likelihood with the prior using the Laplace approximation. Finally, we repameterize the
posterior, from a function of st/√¯αt to a function of st.

Similarly to Theorem 2, the distributions in Theorem 4 mix evidence with the diffusion model prior.
However, this is done implicitly in IRLS. The posterior can be sampled from using LaplaceDPS,
where the mean and covariances would be taken from Theorem 4. Note that Theorem 2 is a special
case of Theorem 4 where the mean function g is an identity.

6


---Page Break---
5
Application to Contextual Bandits

Now we apply our posterior sampling approximations (Section 4) to contextual bandits. A contextual
bandit [28, 31] is a classic model for sequential decision making under uncertainty where the agent
takes actions conditioned on context. We denote the action set by A and the context set by X. The
mean reward for taking action a ∈A in context x ∈X is r(x, a; θ∗), where r : X × A × Θ →R
denotes a reward function and θ∗∈Θ is a model parameter (Section 2). The agent interacts with the
bandit for n rounds indexed by k ∈[n]. In round k, the agent observes a context xk ∈X, takes an
action ak ∈A, and observes a stochastic reward yk = r(xk, ak; θ∗) + εk, where εk is independent
zero-mean σ2-sub-Gaussian noise for σ > 0. The goal of the agent is to maximize its cumulative
reward in n rounds, or equivalently to minimize its cumulative regret. We define the n-round regret as

R(n) = Pn
k=1 E [r(xk, ak,∗; θ∗) −r(xk, ak; θ∗)] ,
(9)

where ak,∗= arg max a∈A r(xk, a; θ∗) is the optimal action in round k.

Arguably the most popular method for solving contextual bandit problems is Thompson sampling
[49, 11, 3]. The key idea in TS is to use the posterior distribution of θ∗to explore. This is done as
follows. In round k, the model parameter is drawn from the posterior in (2), ˜θk ∼p(· | hk), where hk
is the history of all interactions up to round k. After that, the agent takes the action with the highest
mean reward under ˜θk. The pseudo-code of this algorithm is given in Algorithm 3.

A linear bandit [13, 1] is a contextual bandit with a linear reward function r(x, a; θ∗) = ϕ(x, a)⊤θ∗,
where ϕ : X × A →Rd is a feature extractor. The feature extractor can be non-linear in x and a.
Therefore, linear bandits can be applied to non-linear functions in x and a. The feature extractor can
be hand-designed or learned [39]. To simplify notation, we let ϕℓ= ϕ(xℓ, aℓ) be the feature vector of
the action in round k. So the history of interactions up to round k is hk = {(ϕℓ, yℓ)}ℓ∈[k−1]. When
the prior distribution is a Gaussian, p(θ∗) = N(θ∗; θ0, Σ0), the posterior in round k is a Gaussian in
(3) for h = hk. When the prior is a diffusion model, we propose sampling as

˜θk ←LaplaceDPS((µt, Σt)t∈[T ], hk) ,
(10)

where ˆµt and ˆΣt in LaplaceDPS are set according to Theorem 2. We call this algorithm DiffTS.

A generalized linear bandit [16, 23, 32, 25] is an extension of linear bandits to generalized linear
models (Section 2.2). When p(θ∗) = N(θ∗; θ0, Σ0), the Laplace approximation to the posterior is a
Gaussian (Section 2.2). When the prior is a diffusion model, we propose sampling from (10), where
ˆµt and ˆΣt in LaplaceDPS are set according to Theorem 4.

6
Experiments

We conduct three experiments: synthetic problems in 2 dimensions (Section 6.2 and Appendix C.1),
a recommender system (Section 6.3), and a classification problem (Appendix C.2). In addition, we
conduct an ablation study in Appendix C.3, where we vary the number of training samples for the
diffusion prior and the number of diffusion stages T.

6.1
Experimental Setup

We have four baselines. Three baselines are variants of contextual Thompson sampling [11, 3]: with
an uninformative Gaussian prior (TS), a learned Gaussian prior (TunedTS), and a learned Gaussian
mixture prior (MixTS) [21]. The last baseline is diffusion posterior sampling (DPS) of Chung et al.
[12]. We implement all TS baselines as described in Section 5. The uninformative prior is N(0d, Id).
MixTS is used only in linear bandit experiments because the logistic regression variant does not exist.
The TS baselines are chosen to cover various levels of prior information. Our implementation of DPS
is described in Appendix D. We also experimented with frequentist baselines, such as LinUCB [1]
and the ε-greedy policy. They performed worse than TS, and therefore we do not report them here.

Each experiment is set up as follows. First, the prior distribution of θ∗is specified: it can be synthetic
or estimated from real-world data. Second, we learn this distribution from 10 000 samples from it. In
DiffTS and DPS, we follow Appendix B. The number of stages is T = 100 and the diffusion factor
is αt = 0.97. Since 0.97100 ≈0.05, most of the information in the training samples is diffused. The

7


---Page Break---
−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem cross

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem cross

TS
TunedTS
MixTS
DPS
DiffTS

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem rays

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem rays

TS
TunedTS
MixTS
DPS
DiffTS

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem triangles

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem triangles

TS
TunedTS
MixTS
DPS
DiffTS

Figure 2: Evaluation of DiffTS on three synthetic problems. The first row shows samples from the
true (blue) and diffusion model (red) priors. The second row shows the regret of DiffTS and the
baselines as a function of round n.

regressor in Appendix B is a 2-layer neural network with ReLU activations. In TunedTS, we fit the
mean and covariance using maximum likelihood estimation. In MixTS, we fit the Gaussian mixture
using SCIKIT-LEARN. All algorithms are evaluated on θ∗sampled from the true prior. The regret is
computed as defined in (9). All error bars are standard errors of the estimates.

6.2
Synthetic Experiment

The first experiment is on three synthetic problems. Each problem is a linear bandit (Section 5) with
K = 100 actions in d = 2 dimensions. The reward noise is σ = 1. The feature vectors of actions are
sampled uniformly at random from a unit ball. The prior distributions of θ∗are shown in Figure 2.
The first is a mixture of two Gaussians and the last can be approximated well by a mixture of two
Gaussians. We implement MixTS with two mixture components. Therefore, it can represent the first
prior exactly and approximate the last one well.

Our results are reported in Figure 2. We observe two main trends. First, samples from the diffusion
prior closely resemble those from the true prior. In such cases, DiffTS is expected to perform well
and even outperforms MixTS, because it has a better representation of the prior. We observe this in
all problems. Second, DPS diverges as the number of rounds increases. This is because DPS relies on
the likelihood score (Section 7), which tends to infinity and causes instability. This happens despite
tuning (Appendix D). We report results on additional synthetic problems in Appendix C.1.

DiffTS should be T times more computationally costly than TS with a Gaussian prior (Section 4.2).
We observe this empirically. As an example, the average cost of 100 runs of DiffTS on any problem
in Figure 2 is 12 seconds. The average cost of TS is 0.1 seconds. The computation and accuracy can
be traded off, and we investigate this in Appendix C.3. In the cross problem, we vary the number of
diffusion stages from T = 1 to T = 300. We observe that the computational cost is linear in T and
the regret drops quickly from around 85 at T = 1 to 50 at T = 25.

6.3
MovieLens Experiment

In the second experiment, we learn to recommend an item to randomly arriving users. The problem
is simulated using the MovieLens 1M dataset [27], with one million ratings for 3 706 movies from
6 040 users. We subtract the mean rating from all ratings and complete the sparse rating matrix M by
alternating least squares [14] with rank d = 5. The learned factorization is M = UV ⊤. The i-th row
of U, denoted by Ui, represents user i. The j-th row of V , denoted by Vj, represents movie j. We

8


---Page Break---
−2
0
2
4
6
8
10
12
14
−2

0

2

4

6

8

(a) Learned and original priors

Original Prior
Learned Diffusion Prior

0
50
100
150
200

Round n

0

5

10

15

20

Regret

(b) MovieLens linear bandit

TS
TunedTS
MixTS
DiffTS

0
50
100
150
200

Round n

0

2

4

6

8

10

12

14

16

Regret

(c) MovieLens logistic bandit

TS
TunedTS
DiffTS

Figure 3: Evaluation of DiffTS on the MovieLens dataset.

use movie embeddings Vj as model parameters and user embeddings Ui as features of the actions.
The movies are items.

We experiment with both linear and logistic bandits. In both, an item is initially chosen randomly
from Vj and K = 10 actions are chosen randomly from Ui in each round. In the linear bandit, the
mean reward of item j for user i is U ⊤
i Vj. The reward noise is σ = 0.75, and we estimate it from
data. In the logistic bandit, the mean reward is g(U ⊤
i Vj), where g is a sigmoid.

Our MovieLens results are reported in Figure 3 and we observe similar trends to Section 6.2. First,
samples from the diffusion prior closely resemble those from the true prior (Figure 3a). Since the
problem is higher dimensional, we visualize the overlap using a UMAP projection [44]. Second,
DiffTS has a significantly lower regret than all baselines, in both linear (Figure 3b) and logistic
(Figure 3c) bandits. Finally, MixTS barely outperforms TunedTS. We observe this trend consistently
in higher dimensions, which motivated our work on online learning with more complex priors.

7
Related Work

We start with reviewing related works on bandits with diffusion models. Hsieh et al. [22] proposed
Thompson sampling with a diffusion model prior for K-armed bandits. There are multiple technical
differences from our work. First, the diffusion model in Hsieh et al. [22] is over scalars representing
individual arms. Our model is over vectors representing model parameters; and thus can be applied
to contextual bandits. Second, the approximations are different. In stage t, Hsieh et al. [22] sample
from the conditional prior and the diffused empirical mean distribution in stage t. Then they take a
weighted average of the samples. We sample only once, from the posterior distribution that combines
the conditional prior in stage t and likelihood. Because of this, Hsieh et al. [22] can be viewed as a
non-contextual variant of our method, where posterior sampling is done by weighting samples from
the prior and empirical distributions. Finally, Hsieh et al. [22] do not analyze their approximation.

Aouali [4] proposed and analyzed contextual bandits with a linear diffusion model prior: µt(st) in
(5) is linear in st and q(s0) is a Gaussian. Because of this, their model is a linear Gaussian model
and not a general diffusion model, as in our work.

The closest related work on posterior sampling in diffusion models is DPS of Chung et al. [12]. The
key idea in DPS is to sample from the posterior distribution using the likelihood score ∇log p(h | θ),
where p(h | θ) is the likelihood (Assumptions 1 and 2). Note that ∇log p(h | θ) grows linearly in N
because the history h in p(h | θ) involves N terms. Therefore, DPS becomes unstable as N →∞.
See our empirical results in Section 6.2 and the implementation of DPS in Appendix D, which was
tuned to improve its stability.

Many other posterior sampling methods for diffusion models have been proposed recently: a sequen-
tial Monte Carlo approximation for the conditional reverse process [52], a variant of DPS with an
uninformative prior [37], a pseudo-inverse approximation to the likelihood of evidence [47], and
posterior sampling in latent diffusion models [40]. All of these methods rely on the likelihood score
∇log p(h | θ) and thus become unstable as the number of observations N increases. Our posterior
approximations do not have this issue because they are based on the product of prior and evidence
distributions (Theorems 2 and 4), and thus gradient-free. They work well across different levels of
uncertainty (Section 6) and do not require tuning.

9


---Page Break---
We also wanted to note that posterior sampling is a special from of guiding generation in diffusion
models. Other approaches include conditional pre-training [15], a constraint in the reverse process
[17], refining the null-space content [50], solving an optimization problem that pushes the reverse
process towards evidence [46], and aligning the reverse process with the prompt [7].

8
Conclusions

We propose posterior sampling approximations for diffusion models priors. These approximations
are contextual, and can be implemented efficiently in linear models and GLMs. We analyze them and
evaluate them empirically on contextual bandit problems. Our method has two main limitations.

Computational cost. The cost of posterior sampling in LaplaceDPS with T stages is about T times
higher than that of posterior sampling with a Gaussian prior (Section 2). We validate it empirically in
Section 6.2. We plot the sampling time as a function of T in Figure 6c (Appendix C.3).

Learning cost and hyper-parameter tuning. In all experiments, the number of diffusion stages is
T = 100 and the diffusion rate is set such that most of the signal diffuses. The regressor is a 2-layer
neural network and we learn it from 10 000 samples from the prior. These settings resulted in stable
performance in all our experiments (Section 6). However, they clearly impact the performance. We
plot the regret as a function of the number of training samples in Figure 6a and as a function of T in
Figure 6b. When T or the number of training samples is small, DiffTS performs very similarly to
posterior sampling with a Gaussian prior. In summary, there is no benefit in these cases.

Future work. We develop novel posterior approximations rather than bounding their regret. This is
because the existing approximations are unstable and may diverge in the online setting (Sections 6.2
and 7). We believe that a proper regret analysis of DiffTS is possible and would require bounding
two errors. The first error arises because the reverse process does not reverses the forward process
exactly (Appendix B). The second error arises because our posterior distributions are approximate
(Section 4.3). One possibility is to start with prior works that already showed the utility of complex
priors. For instance, Russo and Van Roy [42] proved a O(
p

ΓH(A∗)n) regret bound for a linear
bandit, where Γ is the maximum ratio of regret to information gain and H(A∗) is the entropy of the
distribution of the optimal action under the prior. This bound holds for any prior, and says that a
lower entropy H(A∗), which reflects how concentrated the prior is, yields a lower regret.

We also believe that our ideas can be extended beyond GLMs. The key idea in Section 4.4 is to use
the Laplace approximation of the likelihood. This approximation can be computed exactly in GLMs.
More generally though, it is a good approximation whenever the likelihood can be approximated well
by a single Gaussian distribution. By the central limit theorem, under appropriate conditions, this is
expected for any observation model when the number of observations is large.

References

[1] Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Improved algorithms for linear
stochastic bandits. In Advances in Neural Information Processing Systems 24, pages 2312–2320,
2011.
[2] Marc Abeille and Alessandro Lazaric. Linear Thompson sampling revisited. In Proceedings of
the 20th International Conference on Artificial Intelligence and Statistics, 2017.
[3] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear
payoffs. In Proceedings of the 30th International Conference on Machine Learning, pages
127–135, 2013.
[4] Imad Aouali. Linear diffusion models meet contextual bandits with large action spaces. In
NeurIPS 2023 Workshop on Foundation Models for Decision Making Workshop, 2023.
[5] Imad Aouali, Branislav Kveton, and Sumeet Katariya. Mixed-effect Thompson sampling. In
Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, 2023.
[6] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed
bandit problem. Machine Learning, 47:235–256, 2002.
[7] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum,
Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of
the 12th International Conference on Learning Representations, 2024.

10


---Page Break---
[8] Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal
covariance with imperfect mean in diffusion probabilistic models. In Proceedings of the 39th
International Conference on Machine Learning, 2022.
[9] Soumya Basu, Branislav Kveton, Manzil Zaheer, and Csaba Szepesvari. No regrets for learning
the prior in bandits. In Advances in Neural Information Processing Systems 34, 2021.
[10] Christopher Bishop. Pattern Recognition and Machine Learning. Springer, New York, NY,
2006.
[11] Olivier Chapelle and Lihong Li. An empirical evaluation of Thompson sampling. In Advances
in Neural Information Processing Systems 24, pages 2249–2257, 2011.
[12] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul
Ye. Diffusion posterior sampling for general noisy inverse problems. In Proceedings of the 11th
International Conference on Learning Representations, 2023.
[13] Varsha Dani, Thomas Hayes, and Sham Kakade. Stochastic linear optimization under bandit
feedback. In Proceedings of the 21st Annual Conference on Learning Theory, pages 355–366,
2008.
[14] Mark Davenport and Justin Romberg. An overview of low-rank matrix recovery from incomplete
observations. IEEE Journal of Selected Topics in Signal Processing, 10(4):608–622, 2016.
[15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In
Advances in Neural Information Processing Systems 34, 2021.
[16] Sarah Filippi, Olivier Cappe, Aurelien Garivier, and Csaba Szepesvari. Parametric bandits:
The generalized linear case. In Advances in Neural Information Processing Systems 23, pages
586–594, 2010.
[17] Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, and Dimitris Samaras. Diffusion models
as plug-and-play priors. In Advances in Neural Information Processing Systems 35, 2022.
[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In
Advances in Neural Information Processing Systems 33, 2020.
[19] Joey Hong, Branislav Kveton, Sumeet Katariya, Manzil Zaheer, and Mohammad Ghavamzadeh.
Deep hierarchy in bandits. In Proceedings of the 39th International Conference on Machine
Learning, 2022.
[20] Joey Hong, Branislav Kveton, Manzil Zaheer, and Mohammad Ghavamzadeh. Hierarchical
Bayesian bandits. In Proceedings of the 25th International Conference on Artificial Intelligence
and Statistics, 2022.
[21] Joey Hong, Branislav Kveton, Manzil Zaheer, Mohammad Ghavamzadeh, and Craig Boutilier.
Thompson sampling with a mixture prior. In Proceedings of the 25th International Conference
on Artificial Intelligence and Statistics, 2022.
[22] Yu-Guan Hsieh, Shiva Kasiviswanathan, Branislav Kveton, and Patrick Blobaum. Thompson
sampling with diffusion generative prior. In Proceedings of the 40th International Conference
on Machine Learning, 2023.
[23] Kwang-Sung Jun, Aniruddha Bhargava, Robert Nowak, and Rebecca Willett. Scalable gen-
eralized linear bandits: Online computation and hashing. In Advances in Neural Information
Processing Systems 30, pages 98–108, 2017.
[24] Jaya Kawale, Hung Bui, Branislav Kveton, Long Tran-Thanh, and Sanjay Chawla. Efficient
Thompson sampling for online matrix-factorization recommendation. In Advances in Neural
Information Processing Systems 28, pages 1297–1305, 2015.
[25] Branislav Kveton, Manzil Zaheer, Csaba Szepesvari, Lihong Li, Mohammad Ghavamzadeh,
and Craig Boutilier. Randomized exploration in generalized linear bandits. In Proceedings of
the 23rd International Conference on Artificial Intelligence and Statistics, 2020.
[26] Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Ad-
vances in Applied Mathematics, 6(1):4–22, 1985.
[27] Shyong Lam and Jon Herlocker. MovieLens Dataset. http://grouplens.org/datasets/movielens/,
2016.
[28] John Langford and Tong Zhang. The epoch-greedy algorithm for contextual multi-armed
bandits. In Advances in Neural Information Processing Systems 20, pages 817–824, 2008.

11


---Page Break---
[29] Tor Lattimore and Csaba Szepesvari. Bandit Algorithms. Cambridge University Press, 2019.

[30] Yann LeCun, Corinna Cortes, and Christopher Burges. MNIST Handwritten Digit Database.

http://yann.lecun.com/exdb/mnist, 2010.

[31] Lihong Li, Wei Chu, John Langford, and Robert Schapire. A contextual-bandit approach to
personalized news article recommendation. In Proceedings of the 19th International Conference
on World Wide Web, 2010.

[32] Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear
contextual bandits. In Proceedings of the 34th International Conference on Machine Learning,
pages 2071–2080, 2017.

[33] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hy-
perband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine
Learning Research, 18(185):1–52, 2018.

[34] Shuai Li, Alexandros Karatzoglou, and Claudio Gentile. Collaborative filtering bandits. In
Proceedings of the 39th Annual International ACM SIGIR Conference, 2016.

[35] Xiuyuan Lu and Benjamin Van Roy. Information-theoretic confidence bounds for reinforcement
learning. In Advances in Neural Information Processing Systems 32, 2019.

[36] P. McCullagh and J. A. Nelder. Generalized Linear Models. Chapman & Hall, 1989.

[37] Xiangming Meng and Yoshiyuki Kabashima. Diffusion model based posterior sampling for
noisy linear inverse problems. CoRR, abs/2211.12343, 2023. URL https://arxiv.org/
abs/2211.12343.

[38] Friedrich Pukelsheim. Optimal Design of Experiments. Society for Industrial and Applied
Mathematics, 2006.

[39] Carlos Riquelme, George Tucker, and Jasper Snoek. Deep Bayesian bandits showdown: An
empirical comparison of Bayesian deep networks for Thompson sampling. In Proceedings of
the 6th International Conference on Learning Representations, 2018.

[40] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay
Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion
models. In Advances in Neural Information Processing Systems 36, 2023.

[41] Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics
of Operations Research, 39(4):1221–1243, 2014.

[42] Daniel Russo and Benjamin Van Roy. An information-theoretic analysis of Thompson sampling.
Journal of Machine Learning Research, 17(68):1–30, 2016.

[43] Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial
on Thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018.

[44] Tim Sainburg, Leland McInnes, and Timothy Gentner. Parametric UMAP embeddings for
representation and semisupervised learning. Neural Computation, 33(11):2881–2907, 2021.

[45] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsuper-
vised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International
Conference on Machine Learning, pages 2256–2265, 2015.

[46] Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving
inverse problems with latent diffusion models via hard data consistency. In Proceedings of the
12th International Conference on Learning Representations, 2024.

[47] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion
models for inverse problems. In Proceedings of the 11th International Conference on Learning
Representations, 2023.

[48] Yang Song, Jascha Sohl-Dickstein, Diederik Kingma, Abhishek Kumar, Stefano Ermon, and
Ben Poole. Score-based generative modeling through stochastic differential equations. In
Proceedings of the 9th International Conference on Learning Representations, 2021.

[49] William R. Thompson. On the likelihood that one unknown probability exceeds another in view
of the evidence of two samples. Biometrika, 25(3-4):285–294, 1933.

12


---Page Break---
[50] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising dif-
fusion null-space model. In Proceedings of the 11th International Conference on Learning
Representations, 2023.
[51] R. Wolke and H. Schwetlick. Iteratively reweighted least squares: Algorithms, convergence
analysis, and numerical comparisons. SIAM Journal on Scientific and Statistical Computing, 9
(5):907–921, 1988.
[52] Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John Cunningham. Practical
and asymptotically exact conditional sampling in diffusion models. In Advances in Neural
Information Processing Systems 36, 2023.
[53] Xiaoxue Zhao, Weinan Zhang, and Jun Wang. Interactive collaborative filtering. In Proceedings
of the 22nd ACM International Conference on Information and Knowledge Management, pages
1411–1420, 2013.

13


---Page Break---
A
Proofs and Supporting Lemmas

This section contains proofs of our main claims and supporting lemmas.

A.1
Proof of Lemma 1

All derivations are based on basic rules of probability and the chain structure in Figure 1, and are
exact. From Figure 1, the joint probability distribution conditioned on H = h factors as

p(s0:T | h) = p(sT | h)

T
Y

t=1
p(st−1 | st:T , h) = p(sT | h)

T
Y

t=1
p(st−1 | st, h) .

We use that p(st−1 | st:T , h) = p(st−1 | st, h) in the last equality. We consider two cases.

Derivation of p(st−1 | st, h). By Bayes’ rule, we get

p(st−1 | st, h) = p(h | st−1, st) p(st−1 | st)

p(h | st)
∝p(h | st−1) p(st−1 | st) .

In the last step, we use that p(h | st) is a constant, since st and h are fixed, and that p(h | st−1, st) =
p(h | st−1). Note that the last term p(st−1 | st) is the conditional prior distribution. Let t > 1. Then
we rewrite the first term as

p(h | st−1) =
Z

s0
p(h, s0 | st−1) ds0 =
Z

s0
p(h | s0, st−1) p(s0 | st−1) ds0

=
Z

s0
p(h | s0) p(s0 | st−1) ds0 .

In the last equality, we use that our graphical model is a chain (Figure 1), and thus p(h | s0, st−1) =
p(h | s0). Finally, we chain all identities and get that

p(st−1 | st, h) ∝
Z

s0
p(h | s0) p(s0 | st−1) ds0 p(st−1 | st) .
(11)

Derivation of p(sT | h). By Bayes’ rule, we get

p(sT | h) = p(h | sT ) p(sT )

p(h)
∝p(h | sT ) p(sT ) .

In the last step, we use that p(h) is a constant, since h is fixed. The first term can be rewritten as

p(h | sT ) =
Z

s0
p(h, s0 | sT ) ds0 =
Z

s0
p(h | s0, sT ) p(s0 | sT ) ds0

=
Z

s0
p(h | s0) p(s0 | sT ) ds0 .

Finally, we chain all identities and get that

p(sT | h) ∝
Z

s0
p(h | s0) p(s0 | sT ) ds0 p(sT ) .
(12)

This completes the derivations.

A.2
Proof of Theorem 2

This proof has two parts.

Derivation of p(st−1 | st, h). From p(h | s0) ∝N(s0; ¯θ, ¯Σ) and (6), it immediately follows that
Z

s0
p(h | s0) p(s0 | st−1) ds0 ∝N(st−1/√¯αt−1; ¯θ, ¯Σ) ∝N(st−1; √¯αt−1¯θ, ¯αt−1 ¯Σ) .

14


---Page Break---
The last step treats ¯αt−1 and ¯Σ as constants, because the forward process, t, and evidence are fixed.
Now we apply Lemma 6 to distributions

p(st−1 | st) = N(st−1; µt(st), Σt) ,
N(st−1; √¯αt−1¯θ, ¯αt−1 ¯Σ) ,

and get that

p(st−1 | st, h) ∝N(st−1; ˆµt(st, h), ˆΣt(h)) ,

where ˆµt(st, h) and ˆΣt(h) are defined in the claim. This is a product of two Gaussians: the prior
with mean µt(st) and covariance Σt, and the evidence with mean ¯θ and covariance ¯Σ.

Derivation of p(sT | h). Analogously to the derivation of p(st−1 | st, h), we establish that
Z

s0
p(h | s0) p(s0 | sT ) ds0 ∝N(sT ; √¯αT ¯θ, ¯αT ¯Σ) .

Then we apply Lemma 6 to distributions

p(sT ) = N(sT ; 0d, Id) ,
N(sT ; √¯αT ¯θ, ¯αT ¯Σ) ,

and get that

p(sT | h) ∝N(sT ; ˆµT +1(h), ˆΣT +1(h)) ,

where ˆµT +1(h) and ˆΣT +1(h) are defined in the claim. This is a product of two Gaussians: the prior
with mean 0d and covariance Id, and the evidence with mean ¯θ and covariance ¯Σ.

A.3
Proof of Theorem 3

We start with the triangle inequality

∥˜θ −θ∗∥2 = ∥˜θ −¯θ + ¯θ −θ∗∥2 ≤∥˜θ −¯θ∥2 + ∥¯θ −θ∗∥2 ,

where we introduce ¯θ from Section 2.1. Now we bound each term on the right-hand side.

Upper bound on ∥˜θ −¯θ∥2. This part of the proof is based on analyzing the asymptotic behavior of
the conditional densities in Theorem 2.

As a first step, note that ST ∼N(ˆµT +1(h), ˆΣT +1(h)), where

ˆµT +1(h) = ˆΣT +1(h)(Id 0d + ¯Σ−1¯θ/√¯αT ) ,
ˆΣT +1(h) = (Id + ¯Σ−1/¯αT )−1 .

Since λd(¯Σ−1) →∞, we get

ˆΣT +1(h) →¯αT ¯Σ ,
ˆµT +1(h) →√¯αT ¯θ .

Moreover, λd(¯Σ−1) →∞implies λ1(¯Σ) →0, and thus limN→∞∥ST −√¯αT ¯θ∥2 = 0.

The same argument can be applied inductively to later stages of the reverse process. Specifically, for
any t ∈[T], St−1 ∼N(ˆµt(St, h), ˆΣt(h)), where

ˆµt(St, h) = ˆΣt(h)(Σ−1
t µt(St) + ¯Σ−1¯θ/√¯αt−1) ,
ˆΣt(h) = (Σ−1
t
+ ¯Σ−1/¯αt−1)−1 .

Since λd(¯Σ−1) →∞and St →√¯αt¯θ by induction, we get

ˆΣt(h) →¯αt−1 ¯Σ ,
ˆµt(St, h) →√¯αt−1¯θ .

Moreover, λd(¯Σ−1) →∞implies λ1(¯Σ) →0, and thus limN→∞∥St−1 −√¯αt−1¯θ∥2 = 0.

In the last stage, t = 1, ¯α0 = 1, and S0 = ˜θ, which implies that

lim
N→∞∥˜θ −¯θ∥2 →0 .

Upper bound on ∥¯θ −θ∗∥2. This part of the proof uses the definition of ¯θ in Section 2.1 and that
εℓ∼N(0, σ2) is independent noise. By definition,

¯θ −θ∗= σ−2 ¯Σ

N
X

ℓ=1
ϕℓyℓ−θ∗= σ−2 ¯Σ

N
X

ℓ=1
ϕℓ(ϕ⊤
ℓθ∗+ εℓ) −θ∗= σ−2 ¯Σ

N
X

ℓ=1
ϕℓεℓ.

15


---Page Break---
Since εℓis independent zero-mean Gaussian noise with variance σ2, ¯θ −θ∗is a Gaussian random
variable with mean 0d and covariance

cov

"

σ−2 ¯Σ

N
X

ℓ=1
ϕℓεℓ

#

= σ−4 ¯Σ

 N
X

ℓ=1
ϕℓvar [εℓ] ϕ⊤
ℓ

!
¯Σ = ¯Σ
PN
ℓ=1 ϕℓϕ⊤
ℓ
σ2
¯Σ = ¯Σ .

Since λd(¯Σ−1) →∞implies λ1(¯Σ) →0, we get

lim
N→∞∥¯θ −θ∗∥2 = 0 .

This completes the proof.

A.4
Proof of Theorem 4

This proof has two parts.

Derivation of p(st−1 | st, h). From (6), we have
Z

s0
p(h | s0) p(s0 | st−1) ds0 ∝p(h | st−1/√¯αt−1) .

Since p(st−1 | st) is a Gaussian, we have

p(st−1 | st) = N(st−1; µt(st), Σt) ∝N(γst−1; γµt(st), γ2Σt)

for γ = 1/√¯αt−1. Then by the Laplace approximation,

p(h | γst−1) N(γst−1; γµt(st), γ2Σt) ∝N(γst−1; ˙θt, ˙Σt) ∝N(st−1; ˙θt/γ, ˙Σt/γ2) ,

where ˙θt, ˙Σt ←IRLS(γµt(st), γ2Σt, h).

Derivation of p(sT | h). Analogously to the derivation of p(st−1 | st, h), we establish that
Z

s0
p(h | s0) p(s0 | sT ) ds0 ∝p(h | sT /√¯αT ) .

Then by the Laplace approximation for γ = 1/√¯αT , we get

p(h | γst−1) N(st−1; 0d, Id) ∝N(st−1; ˙θT +1/γ, ˙ΣT +1/γ2) ,

where ˙θT +1, ˙ΣT +1 ←IRLS(0d, γ2Id, h).

A.5
Supporting Lemmas

We state and prove our supplementary lemmas next.
Lemma 5. Let p(x) = N(x; µ1, Σ1) and q(x) = N(x; µ2, Σ2), where µ1, µ2 ∈Rd and Σ1, Σ2 ∈
Rd×d. Then

d(p, q) = 1

2


(µ2 −µ1)⊤Σ−1
2 (µ2 −µ1) + tr(Σ−1
2 Σ1) −log det(Σ1)

det(Σ2) −d

.

Moreover, when Σ1 = Σ2,

d(p, q) = 1

2(µ2 −µ1)⊤Σ−1
2 (µ2 −µ1) .

Proof. The proof follows from the definitions of KL divergence and multivariate Gaussians.

Lemma 6. Fix µ1 ∈Rd, Σ1 ⪰0, µ2 ∈Rd, and Σ2 ⪰0. Then

N(x; µ1, Σ1) N(x; µ2, Σ2) ∝N(x; µ, Σ) ,

where

µ = Σ(Σ−1
1 µ1 + Σ−1
2 µ2) ,
Σ = (Σ−1
1
+ Σ−1
2 )−1 .

16


---Page Break---
Proof. This is a classic result, which is proved as

N(x; µ1, Σ1) N(x; µ2, Σ2) ∝exp

−1

2((x −µ1)⊤Σ−1
1 (x −µ1) + (x −µ2)⊤Σ−1
2 (x −µ2))


∝exp

−1

2(x⊤Σ−1
1 x −2x⊤Σ−1
1 µ1 + x⊤Σ−1
2 x −2x⊤Σ−1
2 µ2)


= exp

−1

2(x⊤Σ−1x −2x⊤Σ−1Σ(Σ−1
1 µ1 + Σ−1
2 µ2))


∝exp

−1

2(x −µ)⊤Σ−1(x −µ)

∝N(x; µ, Σ) .

The neglected factors depend on µ1, µ2, Σ1, and Σ2. This completes the proof.

B
Learning the Reverse Process

One property of our model is that q(sT ) ≈N(sT ; 0d, Id) when T is sufficiently large [18]. Since
ST is initialized to the same distribution in the reverse process p, p can be learned from the forward
process q by simply reversing it. This is done as follows. Using the definition of the forward process
in (4), Ho et al. [18] showed that

q(st−1 | st, s0) = N(st−1; ˜µt(st, s0), ˜βtId)
(13)

holds for any s0 and st, where

˜µt(st, s0) =
√¯αt−1βt

1 −¯αt
s0 +
√αt(1 −¯αt−1)

1 −¯αt
st ,
˜βt = 1 −¯αt−1

1 −¯αt
βt ,
¯αt =

tY

ℓ=1
αℓ.
(14)

Therefore, the latent variable in stage t −1, St−1, is easy to sample when st and s0 are known. To
estimate s0, which is unknown when sampling from the reverse process, we use the forward process
again. In particular, (4) implies that st = √¯αts0 + √1 −¯αtεt for any s0, where εt ∼N(0d, Id) is a
standard Gaussian noise. This identity can be rearranged as

s0 =
1
√¯αt
(st −
√

1 −¯αtεt) .

To obtain εt, which is unknown when sampling from p, we learn to regress it from st [18].

The regressor is learned as follows. Let εt(·; ψ) be a regressor of εt parameterized by ψ and D = {s0}
be a dataset of training examples. We sample s0 uniformly at random from D and then solve

ψt = arg min
ψ
Eq

∥εt −εt(St; ψ)∥2
2

(15)

per stage. The expectation is approximated by sampled s0. Note that we slightly depart from Ho et al.
[18]. Since each regressor has its own parameters, the original optimization problem over T stages
decomposes into T subproblems.

C
Additional Experiments

This section contains three additional experiments.

C.1
Additional Synthetic Problems

In Section 6.2, we show results for three hand-selected problems out of six. We report results on the
other three problems in Figure 4. We observe the same trends as in Section 6.2.

C.2
MNIST Experiment

The next experiment is on the MNIST dataset [30], where the mean reward is estimated by a classifier.
We start with learning an MLP-based multi-way classifier for digits and extract d = 8 dimensional

17


---Page Break---
−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem swirl

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem swirl

TS
TunedTS
MixTS
DPS
DiffTS

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem H

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem H

TS
TunedTS
MixTS
DPS
DiffTS

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem corners

0
50
100
150
200

Round n

0

5

10

15

20

25

Regret

Problem corners

TS
TunedTS
MixTS
DPS
DiffTS

Figure 4: Evaluation of DiffTS on another three synthetic problems. The first row shows samples
from the true (blue) and diffusion model (red) priors. The second row shows the regret of DiffTS
and the baselines as a function of round n.

−2
0
2
4
6
8
10
12
14
−6

−4

−2

0

2

4

6

8

10

(a) Learned and original priors

Original Prior
Learned Diffusion Prior

0
100
200
300
400
500

Round n

0

10

20

30

40

50

60

70

Regret

(b) MNIST linear bandit

TS
TunedTS
MixTS
DiffTS

0
100
200
300
400
500

Round n

0

5

10

15

20

25

30

35

40

45

Regret

(c) MNIST logistic bandit

TS
TunedTS
DiffTS

Figure 5: Evaluation of DiffTS on the MNIST dataset.

embeddings of all digits in the dataset, which are used as feature vectors in our experiment. We
generate a distribution over model parameters θ∗as follows: (1) we choose a random positive label,
assign it reward 1, and assign reward −1 to all other labels; (2) we subsample a random dataset of
size 20, with 50% positive and 50% negative labels; (3) we train a linear model, which gives us a
single θ∗. We repeat this 10 000 times and get a distribution over θ∗.

We consider both linear and logistic bandits. In both, the model parameter θ∗is initially sampled
from the prior. In each round, K = 10 random actions are chosen randomly from all digits. In the
linear bandit, the mean reward for a digit with embedding x is x⊤θ∗and the reward noise is σ = 1.
In the logistic bandit, the mean reward is g(x⊤θ∗), where g is a sigmoid.

Our MNIST results are reported in Figure 5. We observe again that DiffTS has a lower regret than
all baselines, because the learned prior captures the underlying distribution of θ∗well. We note that
both the prior and diffusion prior distributions exhibit a strong cluster structure (Figure 5a), where
each cluster represents one label.

C.3
Ablation Studies

We conduct three ablation studies on the cross problem in Figure 2.

In all experiments, the number of samples for training diffusion priors was 10 000. In Figure 6a, we
vary it from 100 to 10 000. We observe that the regret decreases as the number of samples increases,

18


---Page Break---
0
2000
4000
6000
8000
10000
Training Samples

46

48

50

52

54

56

58

60

62

64

Regret

(a) Training samples ablation

0
50
100
150
200
250
300
Diffusion Steps, T

45

50

55

60

65

70

75

80

85

90

Regret

(b) Diffusion steps ablation

0
50
100
150
200
250
300
Diffusion Steps, T

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

1.6

Compute time, sec

(c) Compute time per run

Figure 6: An ablation study of DiffTS on the cross problem: (a) regret with a varying number of
samples for training the diffusion prior, (b) regret with a varying number of diffusion stages T, and
(c) computation time with a varying number of diffusion stages T.

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem P1

100
101
102
103

Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

EMD

Problem P1

TS
TunedTS
MixTS
DPS
DiffTS
SMC

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem P2

100
101
102
103

Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

EMD

Problem P2

TS
TunedTS
MixTS
DPS
DiffTS
SMC

−4 −3 −2 −1
0
1
2
3
4
−4

−3

−2

−1

0

1

2

3

4

True and diffusion priors

Problem P3

100
101
102
103

Sample size n

0.0

0.2

0.4

0.6

0.8

1.0

EMD

Problem P3

TS
TunedTS
MixTS
DPS
DiffTS
SMC

Figure 7: Evaluation on Gaussian mixture variants of the synthetic problems in Figure 2. The first
row shows samples from the true (blue) and diffusion model (red) priors. The second row shows the
earth mover’s distance of DiffTS and baseline posterior distributions from the true posterior as a
function of sample size n.

due to learning a better prior approximation. The trend stabilizes around 3 000 training samples. We
conclude that the quality of the learned prior approximation has a major impact on regret.

In all experiments, the number of diffusion stages was T = 100. In Figure 6b, we vary it from 1 to
300 and observe its impact on regret. While the regret at T = 1 is high, it decreases quickly as T
increases. It stabilizes around T = 100, which we used in our experiments. In Figure 6c, we vary T
from 1 to 300 and observe its effect on the computation time of posterior sampling. The time is linear
in T, as suggested in Section 4.2. The main contributor to it is the neural network regressor.

C.4
Non-Bandit Evaluation

We use Gaussian mixture variants of the synthetic problems in Figure 2 for our non-bandit evaluation.
The action in round k is chosen uniformly at random (not adaptively). Since the priors are Gaussian
mixtures, the true posterior distribution can be computed in a closed form using MixTS and we can
measure the distance of posterior approximations from it. We use the earth mover’s distance (EMD)
between posterior samples from the true posterior and its approximation. We also considered KL
divergence, but this one required analytical forms of posterior approximations, which are not available
in DiffTS and DPS.

19


---Page Break---
Algorithm 4 DPS of Chung et al. [12].

1: Input: Model parameters ˜σt and ζt

2: Initial sample ST ∼N(0d, Id)
3: for stage t = T, . . . , 1 do
4:
ˆS ←εt(St;ψt)
√1−¯αt
5:
ˆS0 ←
1
√¯αt (St + (1 −¯αt) ˆS)

6:
Z ∼N(0d, Id)

7:
St−1 ←
√¯αt−1βt

1−¯αt
ˆS0 +
√αt(1−¯αt−1)

1−¯αt
St + ˜σtZ −ζt∇PN
ℓ=1(yℓ−ϕ⊤
ℓˆS0)2

8: Output: Posterior sample S0

We evaluate all methods from Figure 2. In addition, we implement a sequential Monte Carlo (SMC)
sampler. The initial particles are selected uniformly at random from the prior. At each round, the
particles are perturbed by a Gaussian noise. The standard deviation of the noise is initialized as a
fraction of the observation noise and decays over time, as the posterior concentrates. The particles
are weighted according to the likelihood of the observation in the round. Finally, we use normalized
likelihood weights to resample the particles. We tune SMC to get good posterior approximations.
We use 3 000 particles. For this setting, the computational costs of posterior sampling in SMC and
DiffTS are comparable.

Our results are reported in Figure 7. We observe that DiffTS approximations are comparable to
MixTS, which has an exact posterior in this setting. The second best performing method is SMC.
Its approximations worsen as the sample size n increases. DPS approximations also get worse as n
increases, which caused instability in Figure 2.

D
Implementation of Chung et al. [12]

In our experiments, we compare to diffusion posterior sampling (DPS) with a Gaussian observation
noise (Algorithm 1 in Chung et al. [12]). Our implementation is presented in Algorithm 4. The score
is ˆS = εt(St; ψt)/√1 −¯αt, where εt(St; ψt) is a regression estimate of the forward process noise

εt in Appendix B. We set ˜σt =
q

˜βt, which is the same amount of noise as in our reverse process
(Section 3). The term

∇

N
X

ℓ=1
(yℓ−ϕ⊤
ℓˆS0)2

is the gradient of the negative log-likelihood with respect to St.

As noted in Appendices C.2 and D.1 of Chung et al. [12], ζt in DPS needs to be tuned for good
performance. We also observed this in our experiments (Section 6.2). To make DPS work well, we
follow Chung et al. [12] and set

ζt =
1
qPN
ℓ=1(yℓ−ϕ⊤
ℓˆS0)2
.

While this significantly improves the performance of DPS, it does not prevent failures. The funda-
mental problem is that gradient-based optimization is sensitive to the step size, especially when the
optimized function is steep. Note that LaplaceDPS does not have any such hyper-parameter.

20


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]

Justification: The abstract and introduction clearly state all contributions. The introduction
also points to where those contributions are made.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The increase in computational cost is discussed in Section 4.2 and shown
empirically in Section 6.2. We also conduct an ablation study in Appendix C.3, where we
show how the regret of DiffTS scales with the number of samples used for pre-training the
prior and the number of stages in the diffusion model prior.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The main claims are stated and discussed in Section 4. Their proofs are in
Appendix A.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We also include code to reproduce the synthetic results in Figures 2 and 4.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We include code to reproduce the synthetic results in Figures 2 and 4.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]

Justification: The experiments are described to a sufficient level to be reproducible. To make
sure, we include code to reproduce the synthetic results in Figures 2 and 4.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: All plots in the paper have error bars.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

21


---Page Break---
Answer: [No]
Justification: Our experiments are not large scale.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We checked the link and comply.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: This work is algorithmic and not tied to a particular application that would
have immediate negative impact.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This paper does not pose such a risk.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: All used assets are stated and cited.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper does not release new assets.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: No crowdsourcing or research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: No crowdsourcing or research with human subjects.

22


---Page Break---
