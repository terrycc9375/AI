Improving Deep Learning Optimization through
Constrained Parameter Regularization

Jörg K.H. Franke
University of Freiburg, Germany
Michael Hefenbrock
RevoAI, Karlsruhe, Germany

Gregor Koehler
German Cancer Research Center (DKFZ)
Heidelberg, Germany

Frank Hutter
ELLIS Institute Tübingen, Germany
University of Freiburg, Germany

Abstract

Regularization is a critical component in deep learning. The most commonly used
approach, weight decay, applies a constant penalty coefficient uniformly across all
parameters. This may be overly restrictive for some parameters, while insufficient
for others. To address this, we present Constrained Parameter Regularization (CPR)
as an alternative to traditional weight decay. Unlike the uniform application of
a single penalty, CPR enforces an upper bound on a statistical measure, such as
the L2-norm, of individual parameter matrices. Consequently, learning becomes
a constraint optimization problem, which we tackle using an adaptation of the
augmented Lagrangian method. CPR introduces only a minor runtime overhead and
only requires setting an upper bound. We propose simple yet efficient mechanisms
for initializing this bound, making CPR rely on no hyperparameter or one, akin to
weight decay. Our empirical studies on computer vision and language modeling
tasks demonstrate CPR’s effectiveness. The results show that CPR can outperform
traditional weight decay and increase performance in pre-training and fine-tuning.

1
Introduction

100k
200k
300k
Optimization Steps

18.0

18.2

18.4

18.6

18.8

Perplexity

AdamCPR 200k
AdamW 200k
AdamW 300k

Figure 1: GPT2s training using Adam with weight
decay or CPR (Kappa-IP). AdamCPR outper-
forms AdamW with the same budget and only re-
quires 2/3 of the budget to reach the same score.

Deep neural networks are the bedrock of
many state-of-the-art machine learning appli-
cations [1]. While these models have exhib-
ited unparalleled expressivity, they also possess
millions, sometimes trillions, of parameters [2].
This massive capacity makes them susceptible
to overfitting, where models memorize nuances
of the training data but underperform on unseen
examples. To mitigate this, many different reg-
ularization techniques have been adopted, with
weight decay and L2 regularization [3, 4, 5] be-
ing the most popular. L2 regularization penal-
izes the squared magnitude of model parameters
and (decoupled) weight decay (which is equiv-
alent to L2 regularization for non-adaptive gra-
dient algorithms [6]) multiplies all weights with
a constant at every step. This seemingly simple
act offers numerous benefits by curbing the growth of individual weights, reducing the risk of relying
on any particular feature excessively, and thus promoting model generalization.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
However, not all parameters in a neural network have the same role or importance and different
weights could benefit from different regularization. Similarly, it is unclear if a single weight decay
value is optimal for the entire duration of optimization, especially for large-scale training. Indeed, Ishii
and Sato [7] showed that a small deep learning model could benefit from layer-wise weight decay
values, and various works showed that scheduling weight decay could improve final performance [8,
9, 10, 11]. This indicates that a dynamic penalty for each individual parameter matrix (e.g., a weight
matrix in a linear layer) could be beneficial for neural network training. Since both scheduling and
parameter-wise weight decay require additional hyperparameters that are often sensitive to the task,
we propose a different approach to obtain customized, dynamic parameter regularization. Instead of
uniformly penalizing weights, we propose to keep them in a certain range, thus ensuring stability
without imposing regularization where it is unnecessary. Constraining parameters, especially based
on statistical measures like the L2 norm, provide a flexible and adaptive form of regularization that
accounts for the heterogeneity of parameters.

In this paper, we propose Constrained Parameter Regularization (CPR), which enforces an upper
bound on a statistical measure of individual parameter matrices. Consequently, regularization is
expressed as a constrained optimization problem, which we address by an adaptation of the augmented
Lagrangian method. The regularization of each parameter matrix is handled by a separate constraint
and Lagrange multiplier, resulting in an individual regularization strength that adapts over time. The
method requires the selection of desired constraint values as well as an update rate for the Lagrange
multipliers. We found that the update rate can be fixed to 1.0. For choosing constraint values,
we introduce four strategies, three of which require a single hyperparameter, while the last one is
hyperparameter-free. We show in our experiments performance improvements over weight decay
when pre-training or finetuning models for image classification (CIFAR100 and ImageNet), language
modeling (OpenWebText), and medical image segmentation. For example, when training a GPT2s
model, we achieved the same performance as AdamW but only require 2/3 of the budget, see Figure 1.
Applying our method for fine-tuning, we find performance improvements and less catastrophic
forgetting. In the following, and after discussing related work (Section 2) and background on weight
decay and the augmented Lagrangian method (Section 3), we make the following contributions:

• Introducing CPR for individualized and dynamic weight regularization1. Specifically, formu-
lating regularization as a constraint optimization problem and proposing CPR as a solution
(Section 4.1).
• Identifying four different strategies for initializing this constraint (Section 4.3). One of them,
Kappa-WS, has a strong default that outperforms tuned AdamW; and another one, Kappa-IP, is
entirely hyperparameter-free and yields even better performance in pre-training.
• Showing improved performance over weight decay in image classification, medical image
segmentation, and pretraining and fine-tuning language models (Section 5).

2
Related Work

Weight decay is an effective regularization technique to improve the generalization and model
performance [12], and the idea of adapting parameter regularization during training is not new.
Lewkowycz and Gur-Ari [8] investigated the effect of L2 regularization on overparameterized
networks and found the time it takes the network to reach peak performance is proportional to
the L2 regularization parameter. They proposed an initialization scheme for L2 regularization and
an annealing schedule for the L2 parameter. Yun et al. [9] use a combination of weight decay
scheduling and knowledge distillation to improve performance on computer vision tasks. More
recent works on self-supervised vision transformers also use a weight decay schedule [10, 11]. In
contrast to our work, none of these proposes a dynamic and individual adaptation of each regularized
parameter matrix. Also, a schedule comes with varying hyperparameter choices while CPR adapts
arbitrarily many parameter matrices with only two hyperparameters (out of which one is fixed in all
our experiments). Instead of using a schedule, Nakamura and Hong [13] proposes AdaDecay, where
the L2 penalty is scaled by standardized gradient norms and a sigmoid function. Ghiasi et al. [14]
propose another gradient-based approach, Adaptive Weight Decay (AWD), which dynamically adjusts
the weight decay based on the ratio of weight norms to gradient norms to balance the contributions
from the cross-entropy and regularization losses aiming to improve the robustness. AMOS [15]

1Please find our implementation under https://github.com/automl/CPR.

2


---Page Break---
leverages model-specific information for initialization and gradients to adapt L2 regularization
during the training. Another way to regularize parameters is to fix the norm of individual parameter
matrices [16], to schedule the weight norm [17], or to limit the total norm of all parameters [18] to a
fixed value. This fixed value is a more sensitive hyperparameter than the hyperparameter in our work.

Our proposed method is not the first to use Lagrangian methods in machine learning [19]. Its
application in deep learning so far focuses on variational methods and generative models: Rezende
and Viola [20] introduced the Generalized ELBO with Constrained Optimization algorithm to optimize
VAEs using Lagrange multipliers optimized by the min-max scheme, and Kohl et al. [21] and Franke
et al. [22] adapted the Lagrangian method from Rezende and Viola [20] to train probabilistic U-nets
and probabilistic Transformer models. While these works adopt Lagrangian methods to handle several
losses in joint optimization problems, our work leverages them to enable individual regularization
strengths.

3
Background

3.1
L2 Regularization and Weight Decay

Regularization methods, such as L2-regularization or weight decay, are commonly used to restrict
parameter updates and enhance generalization by reducing unnecessary complexity [3, 4, 5]. Both can
be motivated by introducing a “cost" to weight magnitudes. Specifically, in L2-regularization, instead
of minimizing only the loss function L(θ, X, y) with parameters θ and data D = {(Xn, yn)}N
n=0, a
weighted penalty (regularization) term R(θ) is added to the loss, resulting in the training objective

min
θ
L(θ, X, y) + γ · R(θ),

where R(θ) = 1

2∥θ∥2
2 denotes the regularization function and γ ∈R+ the strength of the penalty.
On the other hand, weight decay directly modifies the update rule of the parameters to

θt+1 ←θt + Opt(L, η) −η · γ · θt,

where Opt(L, η) denotes an optimizer providing the gradient-based update at iteration t and
L = L(θt, Xt, yt) the loss. For example, Opt(L, η) = −η · ∇θL(θt, Xt, yt) with learning rate
η ∈R+ in case of gradient descent. Thus, the main difference between weight decay and L2-
regularization is that the gradients of the regularization accumulate in momentum terms in the case of
L2-regularisation, while they are treated separately in (decoupled) weight decay. This has also been
extensively discussed by Loshchilov and Hutter [6] with the introduction of the AdamW optimizer.

3.2
The augmented Lagrangian method

We briefly review the augmented Lagrangian method for constrained optimization, see e.g. Bertsekas
[23], which our method is based on. For the derivation, we follow the motivation of Nocedal and
Wright [24, pp. 523-524]. Consider the following inequality-constrained optimization problem

minimize
x
f(x)
s.t.
c(x) ≤0,

with f(x) : Rn →R and a constraint c(x) : Rn →R. One way to address the constraint is to find
an equivalent, unconstrained problem with the same optimal solution. For example,

minimize
x
F(x)
with
F(x) = max
λ≥0 f(x) + λ · c(x).
(1)

Unfortunately, even if f(x) and c(x) are differentiable, F(x) is not differentiable. This is due to the
maximization over λ in F(x), where in case of c(x) > 0, F(x) →∞. Consequently, we cannot run
gradient-based optimization on this objective.

To alleviate this problem, we consider a smooth approximation of F(x), namely

ˆF(x, λt, µ) = max
λ≥0 f(x) + λ · c(x) −1

2µ(λ −λt)2,
(2)

where λt ∈R may be seen as a point we wish to remain proximal to and µ ∈R+ as a factor
determining the strength with which this proximity is enforced. For µ →∞, ˆF(x, λt, µ) →F(x).

3


---Page Break---
The maximization in ˆF(x, λt, µ) has a closed form solution with λ⋆= (λt + µ · c(x))+, where
(·)+ = max{0, ·}, see Appendix A for the derivation. Consequently, ˆF(x, λt, µ) can be written as

ˆF(x, λt, µ) = f(x) + h(x, λt, µ)

with

h(x, λt, µ) =

(
c(x)(λt + µ

2 c(x)),
if
λt + µ · c(x) ≥0
−1

2µλ2
t
else.

The constraint thus only interferes with the minimization (gradient) of f(x) if λt + µ · c(x) ≥0. We
can now try to solve the unconstrained problem minimize
x
ˆF(x, λt, µ) with familiar methods, such
as gradient descent, and obtain an approximate solution to the original problem. Specifically, the
gradient of ˆF(x, λt, µ) with respect to x is given by

∇x ˆF(x, λt, µ) = ∇xf(x) + λ⋆· ∇xc(x).

The quality of the approximation, and thus the solution, clearly depends on µ and λt. To improve this
approximation we can refine the choice of λt via an iterative procedure and repeat the optimization
with λt+1 ←λ⋆= (λt + µ · c(x))+. Intuitively, if the previous minimization of ˆF(x, λt, µ) resulted
in an infeasible solution with c(x) > 0, λt+1 > λt. Hence, the minimization of ˆF(x, λt+1, µ) likely
results in a solution with less constraint violation. On the other hand, if c(x) ≤0, λt+1 ≤λt.
Subsequently, the influence of the constraint is decreased. This loop of alternating minimization of
ˆF(x, λt, µ) and updating λt can be repeated until a sufficiently good solution is found or the procedure
converges if λt does not receive updates anymore. For multiple constraints cj(x), j = 1, · · · , J, the
above can be readily extended with a multiplier λj
t for each constraint. Since the maximization in
the smooth approximation is separable in the λj
t, the same update rule may be applied for each λj
t
separately using the respective constraint cj(x).

4
Constrained Parameter Regularization

In this section, we introduce Constrained Parameter Regularization (CPR), where we adapt the
augmented Lagrangian method to enforce upper bounds on regularization terms. Compared to
classical regularization, with a fixed regularization coefficient γ, the proposed approach will allow for
variable regularization coefficients λj
t (Lagrange multipliers) for j = 1, · · · , J parameter matrices
θj ⊆θ that should be regularized. These regularization coefficients are updated alongside the
network parameters θ.

4.1
Regularization through constraints

Classical weight decay, as introduced earlier, is used as a means to restrict the freedom of parameter
adaptation. This restriction is applied with a scaling factor γ (hyperparameter) and applies uniformly
to all parameters. However, we conjecture that applying an individual adaptation pressure instead
may be beneficial. Unfortunately, this would require a separate coefficient for each parameter matrix
where a separate weight decay should be applied. To avoid the need for separate scaling coefficients,
we formulate regularization as a constrained problem. Here, the loss function L(θ, X, y), with
network parameters θ, takes the place of the objective. Consequently, the learning problem becomes

minimize
θ
L(θ, X, y)
s.t.
cj
 
θj
= R
 
θj
−κj ≤0,
for
j = 1, · · · , J,

where R(θj) is a regularization function (e.g., the squared L2-norm in case of weight decay) for a
parameter matrix θj ⊆θ, j = 1, · · · , J, and κj ∈R denotes a chosen bound.

To solve equation 3, we follow the augmented Lagrangian method with slight modifications. First,
instead of performing a full optimization of the loss before updating λt, we perform updates in every
step. This is motivated by the fact that full optimization is generally infeasible in a deep learning
setting. Moreover, similar to the difference between weight decay and L2-regularization, we treat the
update between the loss-dependent and the constraint-dependent part separately. Hence, instead of
introducing ˆL(x, λt, µ) analogously to equation 2, and performing optimization on this objective, we
independently apply updates for both steps. Consequently, the constraint violations do not accumulate

4


---Page Break---
Algorithm 1 Optimization with constrained parameter regularization (CPR) .

Require: Loss Function L(θ, X, y) with parameters θ, and data D = {(Xn, yn)}N
n=0
Require: Hyperparameters: Learning rate η ∈R+, Lagrange multiplier update rate µ ∈R+(= 1.0)
Require: Optimizer Opt(·) for minimization, Regularization function R(θ) (e.g. L2-norm)

1:
λj
t ←0 for j = 1, · · · , J

2:
κj ←Initialize(θj
0) for j = 1, · · · , J
▷Initializing the upper bound κ, see Section 4.3

3: for Xt, yt ∼D do
4:
θt+1 ←θt + Opt(L(θt, Xt, yt), η)
▷Classic parameter update using, e.g., Adam.

5:
for each regularized parameter group θj
t in θt do

6:
λj
t+1 ←
 
λj
t + µ · (R(θj
t ) −κj)
+

7:
θj
t+1 ←θj
t+1 −∇θjR(θj
t ) · λj
t+1
8:
end for
9: end for

in momentum terms. We also remove the influence of the learning rate on the regularization. From a
practical perspective, our modification does not interfere with gradient-based optimization algorithms
and can be readily combined with any such optimizer. The full algorithm is given by Algorithm 1.

Conceptually, the method can be understood as the λj
t accumulating constraint function values
(weighted with µ) over the iterations t. These then increase (or decrease) the influence of the
constraint (via its gradient) on the search direction. When points in the feasible domain are found for
which cj(θ) ≤0, λj
t decreases until it eventually reaches 0. If, on the other hand, the optimal solution
lies on the boundary, where cj(θ) = 0, λj
t should converge to a value where the update direction of
the optimizer and the gradient of the constraints cancel each other. However, this situation is unlikely
to occur in a deep learning setting due to the stochasticity of minibatches.

4.2
How is CPR different from weight decay?

The optimality conditions of the CPR problem and an L2-regularized training objective reveal a
connection between the two approaches. To see this, consider the training objective of L2 regu-
larization with a given γ, assuming it has a minimum at θ⋆. Consequently, at this point, we have
0 = ∇L(θ⋆) + γ · ∇R(θ⋆), and the value of the regularization function is R(θ⋆).

If we set κ⋆= R(θ⋆), the Karush-Kuhn-Tucker (KKT) (optimality) conditions for CPR are
0 = ∇L(θ⋆) + λ · ∇R(θ⋆) and R(θ⋆) −κ⋆≤0 (which holds with equality), with the Lagrange
multiplier λ ≥0. We can see that for λ⋆= γ, the solution pair (θ⋆, λ⋆) satisfies the KKT condi-
tions. Hence, there is a choice of κ (namely κ⋆) for which the CPR problem has the same optimal
solution candidates as the L2-regularized training objective for a given γ. CPR could therefore be
seen as a different approach to searching for the same solution candidates but is parameterized with
different hyperparameters (κ instead of γ). Unlike L2-regularization (or weight decay), CPR can
mimic the behavior of different γ values for different parameter matrices. This behavior changes
over time as the λj values are updated and thus leads to different training dynamics compared to
weight decay. Additionally, focusing on a bound on the regularization function κ instead of a penalty
coefficient γ may allow us to identify better indicators for the selection of (default) values for these
hyperparameters.

4.3
Initialization of Upper Bounds κj

The upper bound κ is the most crucial hyperparameter for CPR, and we identify four ways to initialize
it. (1) Kappa-K: Set κj ←κ to the same value κ for all parameter matrices. (2) Kappa-kI0: Set
κj based on the initial parameter matrices’ regularization function value: κj ←k · R(θj
t=0), with
k ∈R+ as the factor of the initial measure. (3) Kappa-WS: Train the model parameters θ for a
specific number of warm start (WS) steps s ∈N+ and then set κj ←R(θj
t=s). (see algorithm for

5


---Page Break---
0.0
1e-4
1e-3
1e-2
1e-1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70
70
71
70
71

74
75
75
75
75

75
75
75
75
75

74
74
74
75
75

73
73
73
74
73

68
68
70
72
54

63
64
65
62
17

AdamW

(Kappa-IP)

71

76

76

76

75

72

66

AdamCPR

250
500
1k
2k
4k
8k
16k
Warm start steps (Kappa-WS)

71
71
71
71
71
70
70

76
76
76
76
75
74
75

75
76
77
76
76
76
75

71
74
76
75
75
74
75

65
73
76
75
74
73
73

58
70
73
72
70
69
69

41
63
67
66
64
64
64

AdamCPR

60

65

70

75

80

% Correct Label

Figure 2: Percentage of correct labels (↑) of a ResNet18 trained on CIFAR100 with AdamW and
AdamCPR with Kappa-IP or Kappa-WS. We use a learning rate warm-up of 500 steps and the best
Kappa-WS value is 2× the warm-up steps. We report the mean of three runs with random seeds. We
see that both CPR versions outperform weight decay

CPR with Kappa-WS in Appendix B). While the previous strategies all require a hyperparameter,
our last strategy is essentially hyperparameter-free. (4) Kappa-IP: Use the first inflection point (IP)
of the regularization function at step i (change of curvature over the training steps) to warm start
each parameter matrix individually. Specifically, κj ←R(θj
t=i) where i is the first iteration where
∆t∆tR(θj) < 0. The intuition behind this choice comes from the fact that the rate of change
decreases at the inflection point. This hints at a saturation of the improvement expected through
raising the value of the regularization function further. The position of the inflection point thus
indicates a good choice for κ, as it demonstrated healthy training dynamics while still restricting
the model from over-adapting (see Section 5). Consequently, this method leverages the natural
progression of the model’s training rather than relying on an external hyperparameter, aiming to
adaptively find a suitable upper bound.

5
Experiments

We now describe a set of experiments to understand CPR and its parametrization. Preliminary
experiments showed that µ is not a sensitive hyperparameter and we chose µ = 1.0 for all our
experiments. We provide a detailed analysis of µ in Appendix C. Similar to weight decay, we choose
the squared L2 norm as a default regularization function for CPR. We also tested an adaptive bound,
where we adjusted kappa during training but found it not to be beneficial; details are reported in
Appendix D. In the following experiments, we regularize all parameters in a network except for bias
terms and normalization weights. Since CPR does not require additional gradient calculations or
parameter updates, we find only a small runtime overhead with our CPR implementation (in PyTorch,
no CUDA optimization, 0.4%-5.8% for GPT2) which is mentioned in each experiment individually
and analyzed in Appendix I.

5.1
Train an Image Classification Model (CIFAR100)

To evaluate CPR’s effectiveness and design choices, we tested AdamW and Adam with CPR (Adam-
CPR) in image classification using a ResNet18 on the CIFAR100 dataset [25, 26]. We compared
AdamW to AdamCPR with the four initializations described in Section 4.3. The initialization
Kappa-WS after s warm steps performed best, see Figure 2. We base our choice of the warm start
on the 500 steps learning rate warmup out of 20k total training steps and found a large range of
hyperparameters that consistently outperform weight decay. Also, the hyperparameter-free method
Kappa-IP outperforms weight decay. To detect the infection point, we found it sufficient to sweep
the statistical measure in an interval of 10% of the learning rate warmup. We also apply this in all
further experiments. The superior performance of Kappa-WS and Kappa-IP may be due to its general
flexibility, as warm-started bounds may be considered "learned," reflecting the actual magnitudes and
distributions of the parameter matrices during training. Appendix E contains training details and a
plot with all initializations and standard deviation across three random seeds in Figure E.1. ResNet18
training took 15-20 minutes on a consumer GPU, with no significant runtime difference between

6


---Page Break---
Table 1: Comparison of AdamW and AdamCPR in a DeiT [28] pertaining on ImageNet. We train a
small (22M parameters) and a base model (86M) with different regularization parameters.
ImageNet
Pretraining

AdamW
AdamCPR

weight decay
Kappa WS
Kappa IP
(x lr-warmup)
0.005
0.051
0.5
1x
2x
4x

DeiT-Small (22M)
Top-1 Acc. (%)
76.97
79.03
79.16
79.81 79.33 78.04
79.84
DeiT-Base (86M)
Top-1 Acc. (%)
76.19
78.59
80.56
81.19 79.61 TBA
80.95

AdamW and AdamCPR. We also tested the standard deviation as a choice for the regularization
function, which performed well but not better than the squared L2 norm (see Figure E.2).

To investigate the relationship between the learning rate warm-up and the number of warm start steps
s of Kappa-WS or Kappa-IP, we experimented with varying warm-up steps. We found that setting
the CPR warm start steps s to twice the warm-up steps is a good initial choice. For very low warm-up
steps, the best s was four times the warm-up count. Conversely, with a long warm-up phase, a shorter
CPR warm start (×1) is preferable. Notably, the optimal choice of s is almost independent of the
learning rate, as shown in Figure E.3. The optimal warm start steps are consistent across a wide range
of learning rates. A simple baseline representing a similar regularization approach is a weight decay
schedule. We evaluated a cosine schedule for decreasing and increasing weight decay values, similar
to [10, 11]. The results, shown in Figure E.4, indicate that the decreasing schedule outperforms a
fixed weight decay but not CPR. We tested if CPR is particularly good for noisy data and perfomed
experiments on the noisy CIFAR100-C dataset [27]. The results, in Figure E.5, show that AdamCPR
outperforms AdamW slightly. However none of the optimizer and hyperparameter configurations
lead to an outstanding performance on this task, we wouldn’t claim that CPR is particularly good for
noisy data. We also used CPR with SGD. We found, as shown in Figure E.6, that SGD with CPR
outperforms SGD with weight decay when using the Kappa-WS initialization. However, Kappa-IP
seems not to work with SGD, probably due to the changed convergence behavior in contrast to Adam.

Additionally, we compared our method to related work. We implemented AdaDecay [13] and
evaluated the method for different alpha values, as seen in Figure E.7. We also compared AdamW
and AdamCPR to adaptive Weight Decay (AWD) [14] and AMOS [15]. Furthermore, we used Adam
with parameter rescaling from Liu et al. [18]. We found AdaDecay superior to AdamW, while AMOS
and Rescaling performed less well. However, CPR outperforms all related approaches. We report all
results across multiple learning rates and weight decay values in Figure E.8.

5.2
Train an Image Classification Model (ImageNet)

We compare AdamW and AdamCPR in vision transformer [29] training on ImageNet [30]. We
choose to train the DeiT [28] model with 22M (small) and with 86M (base) parameters. We make
use of the PyTorch Image Models library [31] and train with the configuration given in [28] for
300 epochs. To explore the impact of weight decay, we also train with a 10× and 0.1× the weight
decay value. For CPR, we initialize with Kappa-WS (× lr-warmup) and Kappa-IP. We observed a
minor runtime increase when using CPR. For example, training the small model on 4 A100 GPUs
took 14.85h for AdamW and 14.89h for AdamCPR. All relevant hyperparameters can be found in
Appendix F. As seen in Table 1, AdamCPR outperforms AdamW for small and base DeiT training
with both kappa initialization methods. Most notably, the hyperparameter-free regularization with
Kappa-IP outperforms AdamW in both cases. However, in the base model training, Kappa-WS
surpasses Kappa-IP.

5.3
Fine-tuning a CLIP model

We conducted fine-tuning experiments using a CLIP model [33] on the ImageNet dataset. We used
the ViT-B/32 model pre-trained by Radford et al. [33]. The model was fine-tuned for 10 epochs
following the hyperparameter choices of Wortsman et al. [32] (learning rate of 3 × 10−5, cosine-
annealing learning rate schedule with 500 warm-up steps) but without the special classification head
initialization and the training was performed on a single GPU with a batch size of 512. We compare

7


---Page Break---
Table 2: Comparison of AdamW and AdamCPR for CLIP finetuning on ImageNet. We report the
top-1 accuracy and follow the hyperparameters and schedule from WiSE-FT [32].
ImageNet
Finetuning

AdamW
AdamCPR

weight decay
Kappa WS
Kappa IP
0.0001
0.001
0.01
0.1
1.0
1x
2x
4x

Top-1 Acc. (%)
75.24
75.39
75.32
75.17
74.4
75.27
75.52
75.41
75.40

GPT2s

0.001
0.01
0.1

1e-3.0

1e-2.5

1e-2.0

Learning Rate

18.56
±0.02

18.46
±0.03

18.32
±0.02

18.45
±0.00

18.23
±0.01

18.86
±0.02

18.65
±0.17

18.34
±0.03

20.51
±0.01

AdamW

17.98
±0.02

17.97
±0.04

18.03
±0.05

AdamCPR

5k
10k
20k

18.03
±0.04

18.14
±0.01

18.35
±0.02

18.02
±0.03

18.03
±0.02

18.24
±0.03

18.11
±0.02

18.18
±0.05

18.42
±0.08

AdamCPR

GPT2m

0.001
0.01
0.1
Weight Decay

1e-2.5
16.37
±0.01

16.04
±0.01

16.52
±0.00

(Kappa-IP)

15.58
±0.04

5k
10k
20k
Warm start steps (Kappa-WS)

15.65
±0.01

15.72
±0.02

16.10
±0.03

16

17

18

Perplexity

Figure 3: Perplexity (↓) ± std across three random seeds of GPT2s and GPT2m trained on OpenWeb-
Text with AdamW (left) and AdamCPR with Kappa-IP (middle) and AdamCPR with Kappa-WS
(right). We use a learning rate warm-up of 5k steps. The CPR with the hyperparameter-free strategy
Kappa-IP outperforms weight decay but also CPR with warm start.

AdamW with different weight decay values to AdamCPR in different configurations, where we report
the top-1 accuracy after finetuning. The results in Table 2 show that the Kappa-WS initialization also
leads to better results in this finetuning setting, comparing favorably to traditional weight decay. CPR
with Kappa-IS performs similarly to the best weight decay values, but again, without the need for
finding a regularization hyperparameter.

5.4
Pretraining a Large Language Model (OpenWebText)

We performed experiments training a GPT2 language model [34] on Openwebtext [35]. We compared
AdamW on different weight decay values to AdamCPR using Kappa-WS with different warm start
steps and Kappa-IP. We use a learning rate warmup for 5k steps (2.5% of total training steps)
followed by cosine annealing. Again, we select the warm start steps of κ based on the warmup steps
of the learning rate and evaluate s ∈(5k, 10, 20k) steps. We train the model sizes GPT2s and GPT2m
with 124M and 354M parameters for 200k steps. The results are shown in Figure 3. CPR outperforms
weight decay at all learning rates, in both model sizes and with both kappa initialization strategies.
We also see that the Kappa-IP initialized CPR runs are less sensitive to the learning rate than weight
decay γ. Remarkably, CPR with the hyperparameter-free initialization Kappa-IP performs best,
achieving 0.2 to 0.3 better perplexity than weight decay. To illustrate the performance difference, we
trained a model with weight decay for a longer schedule to get the same performance as with CPR,
the result is shown in Figure 1. CPR saves up to 33% training budget on that scale. Figure 5 shows
the difference in training dynamics with CPR. We find that Kappa-IP is close to the optimal warm
start step for Kappa-WS but find individual starting points for different layers, see Figure G.1. We
provide details of the training and hyperparameters in Appendix H. We found no runtime overhead
of CPR in contrast to AdamW training GPT2s but about 2.5% for GPT2m (see runtime analysis in
Appendix I). We also evaluated AdaDecay [13], Adaptive Weight Decay (AWD) [14] and AMOS [15]
in the GPT2s training setting but neither of the related methods outperforms AdamW nor AdamCPR,
see results in Table H.1.

8


---Page Break---
0.001
0.01
0.1
Weight Decay

1e-4.5

1e-4

1e-3.5

Mistral7B / PubMedQA

Learning Rate

3.8
±0.46

3.8
±0.34

3.9
±0.13

3.7
±0.57

3.8
±1.07

3.4
±0.78

3.1
±0.66

2.6
±0.92

3.3
±0.70

AdamW

50 (1x)
100 (2x)
200 (4x)
Warm start steps (x lr warmup)

4.0
±0.57

4.2
±0.22

4.2
±0.55

4.0
±1.21

4.0
±0.59

3.8
±0.45

3.2
±0.59

3.4
±0.37

3.1
±1.15

AdamCPR (Kappa-WS)

3

4

% Accuracy Improvment

Figure 4: Percentage of performance change before and after fineuning Mistral 7B with pubmedQA
artificial data (↑) with the use of AdamW (left) and AdamCPR with Kappa-WS (right). We use a
learning rate warm-up of 50 steps. We see that CPR outperforms weight decay for each learning rate.

5.5
Fine-tuning a Large Language Model

Probably a more common task than pre-training a large language model (LLM) is to fine-tune one.
Hence, we evaluate CPR in the fine-tuning of the Mistral7B large language model [36], incorporating
low-rank adaptation (LoRA) [37]. Specifically, we fine-tune artificially generated biomedical question-
answering (QA) pairs from the PubMedQA dataset [38]. We fine-tune all attention and feed-forward
weights using either AdamW or AdamCPR with a learning rate warm-up of 50 steps, followed by
cosine annealing. We experiment with different values of weight decay and warm start steps for
Kappa-WS, set at 1×, 2×, and 4× the learning rate warm-up steps. The fine-tuning was performed
on four GPUs for about 1h. Each configuration is trained across three random seeds. We evaluate the
LLM before and after the fine-tuning on the expert-annotated PubMedQA QA instances and report
the change in answer accuracy (means and standard deviations across three random seeds) in Figure
4. The fine-tuning enhances the performance on the PubMedQA benchmark and CPR outperforms
AdamW for each learning rate. As in both the ImageNet and GPT2 experiments, the best Kappa-WS
value was 2× the warm-up steps (here, 50 × 2). We also tested Kappa-IP but it performed worse
due to the lack of an inflection point for some parameters, short learning rate warmup, and different
training dynamics with LoRA. We also found that CPR helps to mitigate catastrophic forgetting,
therefore we evaluate before and after finetuning on a set of benchmarks and found that CPR with
some learning rates helps to reduce a performance drop e.g. on the TruthfulQA benchmark, which
evaluates models’ abilities to mimic human falsehoods [39], on up to 3% (see results in Figure K.1).
Detailed hyperparameters and plots including standard deviations are available in Appendix K.

5.6
Medical Image Segmentation

Aside from image classification, we also applied CPR to (medical) image segmentation using the nnU-
Net framework [40] and training with the SGD optimizer in combination with CPR with Kappa-WS.
For this, we considered the tasks of Multi-Atlas Labeling Beyond the Cranial Vault (BTCV) [41]
where we improve the Dice score from 83.99 to 84.23, the Heart Segmentation task of the Medical
Segmentation Decathlon [42] where we improve the Dice score from 92.92 to 93.18 and the 2020
version of the Brain Tumor Segmentation challenge (BraTS) task [43] where we improve the Dice
score from 76.22 to 76.65. These results show that CPR also works in combination with SGD where
we replace weight decay. Training details for the task and all results are in Appendix J.

6
Discussion

Our extensive evaluation of Constrained Parameter Regularization (CPR) across multiple tasks
underscores its effectiveness as a robust alternative to traditional weight decay. A critical aspect of
CPR’s success is its initialization strategy. To this end, we propose four strategies to initialize the
upper bound κ. With our findings, we identify two strategies, Kappa-WS and Kappa-IP as prime
candidates showing a strong performance, consistent across multiple tasks. The good performance
of the warm-started bound Kappa-WS can be attributed to the fact that even a carefully chosen
initialization of parameters does not consider the training task and data. Therefore, the actual
parameter weights during training are better reflected in a warm-started bound, which also takes into

9


---Page Break---
0.0

0.1

∥θj∥2
2

kappa initialization at step 7400 with κ =0.032

Optimization Steps

0.0e+00

5.0e-06

Layer 5

FC1 Weight

∆t∥θj∥2
2

Optimization Steps t

0.0e+00

1.0e-04

λj

0
25000
50000
75000
100000
125000
150000
175000
200000
Optimization Steps t

3.00

3.25

Validation

Loss

GPT2s Training Dynamics of AdamW (blue) and AdamCPR (green) with Kappa-IP

Figure 5: The training dynamics of AdamW (blue) and AdamCPR with Kappa-IP (green) in a
GPT2s training run. The upper plot shows the squared L2 norm of the first fully connected weight in
the fifth layer. Below we see the gradient of the squared L2 norm regarding the training steps. After
the inflection point (7400), Kappa-IP initializes kappa κj ←R(θj
t=i) and starts the regularization.
The third plot shows CPR’s lambda enforcing the constraint. At the bottom, we see the validation loss.
AdamW converges faster in the beginning of the training but CPR leads to a more linear improvement
and a better final performance.

account the network’s depth and the varying gradient updates in deeper layers. We found that setting
the CPR warm start steps s to twice the learning rate warm-up steps serves as an effective initial
configuration for any training setup. However in a pre-training setting, setting the upper bound based
on the first inflection point of the regularization function (Kappa-IP) yields an additional advantage:
It removes even the one hyperparameter present in the warm start strategy, bringing the regularization
capabilities of CPR without any additional hyperparameters. Simultaneously, this strategy shows
best-in-class performance in GPT2 training, seemingly even extending the range of usable learning
rates on a given task. This reduces the effort in hyperparameter optimization not only for the optimal
regularization but also for the optimal learning rate. CPR also changes the training dynamics, as
shown in Figure 5 and Figure G.1. While both weight decay and CPR can achieve a similar final L2
regularization, the path to this norm is different. Weight decay allows for intermediate overadaptation
with high L2 norms, whereas CPR controls the L2 norm throughout the entire training process. This
results in a slower initial loss drop but a more consistent decay, leading to a better final performance.

A noted limitation of CPR is an increase in runtime by up to 6% for larger models (1.1B parameters),
as detailed in Appendix I. However, for smaller models or larger batch sizes, this overhead is
negligible. The benefit of CPR diminishes in scenarios where weight regularization has minimal
impact, such as when training small models on large datasets with a high ratio of training samples
to parameters. Future research could explore the application of CPR to even larger models and a
broader range of tasks.

7
Conclusion

Constrained Parameter Regularization (CPR) offers a significant advancement in regularization
techniques, providing a robust and efficient alternative to traditional methods. By enforcing an upper
bound on the regularization function, CPR integrates seamlessly with gradient-based optimizers and
incurs minimal runtime overhead. Its dynamic tailoring of regularization to individual parameter
matrices and reduces hyperparameter optimization by eliminating the need for a weight regularization
hyperparameter in pre-training. Our four experiments demonstrate that neural networks trained using
CPR outperform those with traditional weight decay. These findings highlight CPR’s potential as a
versatile and powerful tool for improving model performance and open promising future research.

10


---Page Break---
Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foun-
dation) under grant number 417962828. We acknowledge funding by the European Union (via ERC
Consolidator Grant DeepLearning 2.0, grant no. 101045765). Views and opinions expressed are
however those of the author(s) only and do not necessarily reflect those of the European Union or
the European Research Council. Neither the European Union nor the granting authority can be held
responsible for them.

The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu)
for funding this project by providing computing time on the GCS Supercomputer JUWELS [44] at
Jülich Supercomputing Centre (JSC). We acknowledge the financial support of the Hector Foundation.

References

[1] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016.

[2] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models
with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

[3] Stephen Hanson and Lorien Pratt.
Comparing biases for minimal network construction with back-
propagation. In Advances in Neural Information Processing Systems, volume 1. Morgan-Kaufmann,
1988.

[4] Anders Krogh and John Hertz. A simple weight decay can improve generalization. Advances in Neural
Information Processing Systems, 4, 1991.

[5] S. Bos and E. Chug. Using weight decay to optimize the generalization ability of a perceptron. In
Proceedings of International Conference on Neural Networks (ICNN’96), volume 1, pages 241–246 vol.1,
1996.

[6] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In Proceedings of the International
Conference on Learning Representations (ICLR’19), 2019.

[7] Masato Ishii and Atsushi Sato. Layer-wise weight decay for deep neural networks. In Image and Video
Technology, pages 276–289, Cham, 2018. Springer International Publishing.

[8] Aitor Lewkowycz and Guy Gur-Ari. On the training dynamics of deep networks with l_2 regularization.
In Advances in Neural Information Processing Systems, volume 33, pages 4790–4799, 2020.

[9] Juseung Yun, Byungjoo Kim, and Junmo Kim. Weight decay scheduling and knowledge distillation for
active learning. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28,
2020, Proceedings, Part XXVI 16, pages 431–447. Springer, 2020.

[10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand
Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF
international conference on computer vision, pages 9650–9660, 2021.

[11] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre
Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual
features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[12] Guodong Zhang, Chaoqi Wang, Bowen Xu, and Roger Grosse. Three mechanisms of weight decay
regularization. In International Conference on Learning Representations, 2018.

[13] Kensuke Nakamura and Byung-Woo Hong. Adaptive weight decay for deep neural networks. IEEE Access,
7:118857–118865, 2019.

[14] Mohammad Amin Ghiasi, Ali Shafahi, and Reza Ardekani. Improving robustness with adaptive weight
decay. Advances in Neural Information Processing Systems, 36, 2024.

[15] Ran Tian and Ankur P Parikh. Amos: An adam-style optimizer with adaptive weight decay towards
model-oriented scale. arXiv preprint arXiv:2210.11693, 2022.

[16] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate
training of deep neural networks. volume 29, 2016.

[17] Ilya Loshchilov. Weight norm control. arXiv preprint arXiv:2311.11446, 2023.

[18] Ziming Liu, Eric J Michaud, and Max Tegmark. Omnigrok: Grokking beyond algorithmic data. In The
Eleventh International Conference on Learning Representations, 2023.

11


---Page Break---
[19] John Platt and Alan Barr. Constrained differential optimization. In Advances in Neural Information
Processing Systems, volume 0, 1987.

[20] Danilo Jimenez Rezende and Fabio Viola. Taming vaes. arXiv preprint arXiv:1810.00597, 2018.

[21] Simon Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R Ledsam, Klaus
Maier-Hein, SM Eslami, Danilo Jimenez Rezende, and Olaf Ronneberger. A probabilistic u-net for
segmentation of ambiguous images. Advances in Neural Information Processing Systems, 31, 2018.

[22] Jörg Franke, Frederic Runge, and Frank Hutter. Probabilistic transformer: Modelling ambiguities and
distributions for rna folding and molecule design. Advances in Neural Information Processing Systems, 35:
26856–26873, 2022.

[23] Dimitri P Bertsekas. Constrained Optimization and Lagrange Multiplier Methods. Athena Scientific, 1996.
ISBN 1886529043.

[24] Jorge Nocedal and Stephen J. Wright. Numerical Optimization. Springer, New York, NY, USA, 2e edition,
2006.

[25] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the
International Conference on Computer Vision and Pattern Recognition (CVPR’16), pages 770–778, 2016.

[26] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of
Toronto, 2009.

[27] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions
and perturbations. In International Conference on Learning Representations, 2018.

[28] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou.
Training data-efficient image transformers & distillation through attention. In International Conference on
Machine Learning, pages 10347–10357. PMLR, 2021.

[29] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is
worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning
Representations, 2020.

[30] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 248–255.
IEEE, 2009.

[31] Ross Wightman. Pytorch image models. https://github.com/rwightman/pytorch-image-models,
2019.

[32] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs,
Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning
of zero-shot models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
7959–7971, 2022.

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from
natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR,
2021.

[34] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.

[35] Aaron Gokaslan and Vanya Cohen.
Openwebtext corpus.
http://Skylion007.github.io/
OpenWebTextCorpus, 2019.

[36] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b.
arXiv preprint arXiv:2310.06825, 2023.

[37] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning
Representations, 2021.

[38] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for
biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors,
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577.
Association for Computational Linguistics, Nov 2019.

[39] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human
falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 3214–3252, 2022.

12


---Page Break---
[40] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a
self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):
203–211, 2021.

[41] Bennett Landman, Zhoubing Xu, J Igelsias, Martin Styner, T Langerak, and Arno Klein. Miccai multi-atlas
labeling beyond the cranial vault–workshop and challenge. In Proc. MICCAI Multi-Atlas Labeling Beyond
Cranial Vault—Workshop Challenge, volume 5, page 12, 2015.

[42] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, AnnetteKopp-Schneider, Bennett A
Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, Bram van Ginneken,
Michel Bilello, Patrick Bilic, Patrick F Christ, Richard K G Do, Marc J Gollub, Stephan H Heckers,
Henkjan Huisman, William R Jarnagin, Maureen K McHugo, Sandy Napel, Jennifer S Goli Pernicka,
Kawal Rhode, Catalina Tobon-Gomez, Eugene Vorontsov, Henkjan Huisman, James A Meakin, Sebastien
Ourselin, Manuel Wiesenfarth, Pablo Arbelaez, Byeonguk Bae, Sihong Chen, Laura Daza, Jianjiang Feng,
Baochun He, Fabian Isensee, Yuanfeng Ji, Fucang Jia, Namkug Kim, Ildoo Kim, Dorit Merhof, Akshay
Pai, Beomhee Park, Mathias Perslev, Ramin Rezaiifar, Oliver Rippel, Ignacio Sarasua, Wei Shen, Jaemin
Son, Christian Wachinger, Liansheng Wang, Yan Wang, Yingda Xia, Daguang Xu, Zhanwei Xu, Yefeng
Zheng, Amber L Simpson, Lena Maier-Hein, and M Jorge Cardoso. The Medical Segmentation Decathlon.
Nature Communications, 13(1):4128, 2022.

[43] Bjoern H. Menze, Andras Jakab, Stefan Bauer, Jayashree Kalpathy-Cramer, Keyvan Farahani, Justin
Kirby, Yuliya Burren, Nicole Porz, Johannes Slotboom, Roland Wiest, Levente Lanczi, Elizabeth Gerstner,
Marc-André Weber, Tal Arbel, Brian B. Avants, Nicholas Ayache, Patricia Buendia, D. Louis Collins,
Nicolas Cordier, Jason J. Corso, Antonio Criminisi, Tilak Das, Hervé Delingette, Ça˘gatay Demiralp,
Christopher R. Durst, Michel Dojat, Senan Doyle, Joana Festa, Florence Forbes, Ezequiel Geremia, Ben
Glocker, Polina Golland, Xiaotao Guo, Andac Hamamci, Khan M. Iftekharuddin, Raj Jena, Nigel M. John,
Ender Konukoglu, Danial Lashkari, José António Mariz, Raphael Meier, Sérgio Pereira, Doina Precup,
Stephen J. Price, Tammy Riklin Raviv, Syed M. S. Reza, Michael Ryan, Duygu Sarikaya, Lawrence
Schwartz, Hoo-Chang Shin, Jamie Shotton, Carlos A. Silva, Nuno Sousa, Nagesh K. Subbanna, Gabor
Szekely, Thomas J. Taylor, Owen M. Thomas, Nicholas J. Tustison, Gozde Unal, Flor Vasseur, Max
Wintermark, Dong Hye Ye, Liang Zhao, Binsheng Zhao, Darko Zikic, Marcel Prastawa, Mauricio Reyes,
and Koen Van Leemput. The multimodal brain tumor image segmentation benchmark (brats). IEEE
Transactions on Medical Imaging, 34(10):1993–2024, 2015.

[44] Jülich Supercomputing Centre. JUWELS Cluster and Booster: Exascale Pathfinder with Modular Super-
computing Architecture at Juelich Supercomputing Centre. Journal of large-scale research facilities, 7
(A138), 2021.

[45] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-
efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems,
2022.

[46] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary
position embedding, 2021.

[47] Mykola Novik. torch-optimizer – collection of optimization algorithms for PyTorch., 1 2020.

[48] Léon Bottou and Olivier Bousquet. The tradeoffs of large scale learning. Advances in Neural Information
Processing Systems, 20, 2007.

[49] Yuanfeng Ji, Haotian Bai, Jie Yang, Chongjian Ge, Ye Zhu, Ruimao Zhang, Zhen Li, Lingyan Zhang,
Wanling Ma, Xiang Wan, et al. Amos: A large-scale abdominal multi-organ benchmark for versatile
medical image segmentation. arXiv preprint arXiv:2206.08023, 2022.

[50] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language
models are few-shot learners. 2020.

[51] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning
Representations, 2020.

[52] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical common-
sense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34,
pages 7432–7439, 2020.

13


---Page Break---
Appendix

A
Derivation of the Lagrange multiplier update

For simplicity, we consider a single constraint. Note that multiple constraints can be addressed
separately as the optimization problem would be separable in the respective λj. We need to solve

maximize
λ≥0
f(x) + λ · c(x) −1

2µ(λ −λt)2.

The optimal point of this problem is equivalent to the optimal point of

minimize
λ
−f(x) −λ · c(x) + 1

2µ(λ −λt)2
s.t.
−λ ≤0.

To find candidates for optimal points, we need to solve the Karush–Kuhn–Tucker (KKT) system with
the Lagrange function L(λ, ψ) and the Lagrange multiplier ψ

L(λ, ψ) = −f(x) −λ · c(x) + 1

2µ(λ −λt)2 −ψ · λ

Which leads to the KKT system

∇λL(λ, ψ) = 0 ⇐⇒0 = −c(x) + 1

µ(λ −λt) −ψ

∇ψL(λ, ψ) ≤0 ⇐⇒0 ≥−λ
λ · ψ = 0
(3)

According to the complementary conditions in equation 3, the constraint is either active, hence λ = 0
and ψ ≥0 or inactive, such that λ > 0, and consequently, ψ = 0.

Case: λ = 0 and ψ ≥0

Here, λ = 0 (by assumption), and ψ is given by

∇λL(λ, ψ) = 0 ⇐⇒0 = −c(x) + 1

µ(0 −λt) −ψ

ψ = −c(x) −λt

µ

Since we require ψ ≥0 for a KKT point, (note that µ > 0)

0 ≤ψ = −c(x) −λt

µ
⇐⇒0 ≤−µ · c(x) −λt
⇐⇒0 ≥λt + µ · c(x)

Consequently, λ = 0 is a candidate for the optimal point only when 0 ≥λt + µ · c(x).

Case: λ > 0 and ψ = 0 (inactive constraint)

For this case we get

∇λL(λ, ψ) = 0 = −c(x) + 1

µ(λ −λt) −0

0 = −µ · c(x) + λ −λt
λ = λt + µ · c(x)

Due to the geometry of the problem (quadratic with bound constraint), λ = 0 is the optimal solution
if the constraint is active, i.e., if ψ ≥0, which is the case if 0 ≥λt + µ · c(x). Consequently, the
optimal solution is given by
λ⋆= (λt + µ · c(x))+.
(4)

14


---Page Break---
Plugging this into ˆF(x, λt, µ), we get

ˆF(x, λt, µ) =

(
f(x) + c(x)(λt + µ

2 c(x)),
if
λt + µ · c(x) ≥0
f(x) −
1
2µλ2
t,
else

And the gradient with respect to x is

∇x ˆF(x, λt, µ) =
∇xf(x) + ∇xc(x)(λt + µ · c(x)),
if
λt + µ · c(x) ≥0
∇xf(x) −0
else

Or more compactly by using equation 4

∇x ˆF(x, λt, µ) = ∇xf(x) + ∇xc(x) · λ⋆.

B
The CPR Algorithm with Kappa-WS

Algorithm 2 Optimization with constrained parameter regularization (CPR) and Kappa-WS .

Require: Loss Function L(θ, X, y) with parameters θ, and data D = {(Xn, yn)}N
n=0
Require: Hyperparameters: Learning rate η ∈R+, Lagrange multiplier update rate µ ∈R+, starting
step s for CBR.
Require: Optimizer Opt(·) for minimization, Regularization function R(θ) (e.g. L2-norm)

1: # Initialization
2: t ←0
3: θt ←Initialize(L(·))

4:
λj
t ←0 for j = 1, · · · , J

5:
κj ←∞j = 1, · · · , J
6: # Training
7: for Xt, yt ∼D do
8:
θt+1 ←θt + Opt(L(θt, Xt, yt), η)
▷Classic parameter update using, e.g., Adam.

9:
for each regularized parameter group θj
t in θt do

10:
λj
t+1 ←
 
λj
t + µ · (R(θj
t ) −κj)
+

11:
θj
t+1 ←θj
t+1 −∇θjR(θj
t ) · λj
t+1

12:
if t = s then
▷Kappa-kIs initialization, see Section 4.3.

13:
κj ←R(θj
t )
14:
end if
15:
end for
16:
t ←t + 1
17: end for

15


---Page Break---
C
Experiments on the Sensitivity of the Update Rate µ

We analyze the sensitivity of the update rate µ in CPR with experiments on ResNet18 trained on
the CIFAR100 and GPT2s trained on OpenWebText. For the ResNet18 experiments, we consider
update rates from µ = 0.01 to µ = 10 and apply two kappa initialization methods,Kappa-kI0 and
Kappa-WS. As shown in Figure C.1 we see no significant impact of µ on the performance. We report
the mean percentage of correct labels across three random seeds. We also performed short-runtime
experiments with GPT2s and update rates of µ ∈{0.01, 0.1, 1, 10}. and observe very similar results,
see Table C.1. To get an impression of how µ impacts λ and therefore the squared L2 norm in the
weight matrices with the use of CPR, we plotted the squared L2 norm and λ for three weight matrices
during the training in Figure C.2. We found no impact on the stability of the squared L2 norm despite
the difference in the magnitude of the λ for different µ values.

1e-2.0
1e-1.5
1e-1.0
1e-0.5
1e0.0
1e0.5
1e1.0
Update rate µ

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

71.1
±0.15

71.1
±0.14

71.2
±0.26

71.2
±0.31

71.3
±0.16

71.4
±0.12

71.0
±0.14

75.9
±0.34

75.8
±0.39

76.1
±0.18

75.8
±0.51

75.8
±0.15

75.9
±0.18

75.8
±0.11

76.5
±0.06

76.6
±0.26

76.6
±0.39

76.8
±0.31

76.8
±0.12

76.5
±0.10

76.5
±0.25

75.7
±0.34

75.9
±0.66

75.7
±0.59

75.6
±0.08

75.9
±0.26

76.1
±0.23

75.9
±0.21

75.7
±0.16

76.1
±0.40

76.0
±0.11

75.7
±0.33

75.6
±0.36

75.9
±0.06

75.7
±0.30

72.7
±0.73

73.2
±0.15

72.8
±0.68

73.0
±0.36

73.0
±0.49

72.9
±0.61

73.6
±0.81

66.5
±1.02

67.0
±0.48

67.0
±0.64

66.8
±1.08

67.3
±0.09

67.0
±0.43

66.5
±0.70

AdamCPR (Kappa-WS)

1e-2.0
1e-1.5
1e-1.0
1e-0.5
1e0.0
1e0.5
1e1.0
Update rate µ

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.5
±0.38

70.1
±0.08

70.3
±0.19

70.1
±0.29

70.3
±0.01

69.9
±0.34

70.1
±0.06

74.4
±0.52

74.4
±0.14

74.7
±0.09

74.6
±0.34

74.5
±0.20

74.5
±0.31

74.6
±0.09

75.2
±0.29

75.1
±0.24

75.0
±0.18

75.3
±0.21

75.2
±0.16

75.1
±0.18

75.5
±0.27

75.8
±0.34

75.8
±0.49

76.1
±0.08

76.0
±0.38

75.8
±0.31

75.8
±0.15

75.7
±0.18

74.9
±0.32

75.2
±0.51

74.8
±0.33

75.3
±0.20

75.5
±0.19

75.0
±0.13

75.1
±0.36

69.0
±0.31

69.2
±0.49

68.6
±0.14

68.6
±0.45

68.4
±0.34

68.8
±0.59

68.7
±0.15

50.6
±0.76

51.3
±1.59

49.9
±1.15

50.7
±1.76

50.7
±1.64

50.2
±0.85

50.8
±1.15

AdamCPR (Kappa-kI0)

60

65

70

75

80

Figure C.1: The Figure shows the percentage of correct labels of the ResNet18 trained on the
CIFAR100 with the use of Kappa-kI0 (left), AdamCPR (Kappa-WS) (right) with different update
rates µ. The elements in the heat map are experiments with different learning rates and each element
is colored according to the mean accuracy of three random seeds and the numbers are the mean
accuracy and standard deviation of the experiments. The experiment shows that the AdamCPR
regularization is not sensitive to the choice of the µ parameter.

Table C.1: Comparison of different values for the update rate µ of AdamCPR. We run experiments
with GPT2s with 50k total steps, a learning rate warmup of 2.5k steps, and a kappa warm start of 5k
steps.

Method (µ value)
Accuracy ↑
PPL ↓

GPT2s
AdamCPR µ = 10
0.422
20.91
AdamCPR µ = 1
0.423
20.90
AdamCPR µ = 0.1
0.423
20.90
AdamCPR µ = 0.01
0.423
20.90

16


---Page Break---
0.0025

0.0050

0.0075

Layer 1

Attn Weight

∥θj∥2
2

µ=10
µ=1.0
µ=0.1
µ=0.01

0.0000

0.0002

0.0004

Layer 1

Attn Weight

λj

0.000

0.005

0.010

Layer 5

Fc1 Weight

∥θj∥2
2

0.0000

0.0002

Layer 5

Fc1 Weight

λj

0.000

0.005

0.010

Layer 10

Fc2 Weight

∥θj∥2
2

0.0000

0.0002

Layer 10

Fc2 Weight

λj

0

2

4

Training

Loss

0
10000
20000
30000
40000
50000
Optimization Steps t

3.2

3.4

Validation

Loss

GPT2s with AdamCPR with diﬀerent µ

Figure C.2: A comparison of different λ update rates µ in the training of a GPT2s model. We see
three weight matrices during the training with AdamCPR. We also see how λ regulates the constraint
of the bound on the squared L2 norm. The bottom two plots show the training and validation loss.

17


---Page Break---
D
Adaptive Bounds

With fixed bounds κj, some parameter matrices θj, for which λj
t = 0 will not be regularized. While
this can be beneficial, CPR can also be used to apply continuous pressure similar to weight decay.
For this, the bounds κj of parameter matrices θj with λj = 0 can be set to the current value of the
constraint function κj
t+1 ←c(θj
t ). Such an adaption guarantees that each parameter matrix is always
exposed to some regularization. This should result in a gradual reduction of the bounds κj throughout
training without exerting excessive pressure on the optimization process. In our experiments, we
refer to the usage of adaptive bounds as AdaCPR.

This contrasts with weight decay, where continuous pressure is applied to enhance generalization
throughout the training. To emulate the continuous pressure of weight decay, we propose an adaptive
mechanism to adjust the upper regularization bound during training. This can be achieved by
leveraging existing states. Specifically, the value of λj offers insights into constraint violations. When
λj = 0, the constraint cj(θ) can be regarded as inactive. In this case, we may consider adjusting its
bound κj to align with the current constraint value of c(θj). To implement these adaptive bounds,
we add a conditional update rule for κj after our CPR update. It updates the upper bound for each
parameter matrix θj
t individually by

κj
t+1 ←

(
R(θj
t)
if λj
t = 0 and λj
t−1 > 0
κj
t
otherwise,

where λj
t−1 > 0 indicates that the upper bound was previously violated and cj(θj) was active.
Consequently, this enables a gradual reduction of the bounds κj throughout training without exerting
excessive pressure on the optimization process. Please find AdaCPR in Algorithm 3 below.

Algorithm 3 Optimization with adaptive bound constrained parameter regularization ( Ada CPR ).

Require: Loss Function L(θ, X, y) with parameters θ, and data D = {(Xn, yn)}N
n=0
Require: Hyperparameters: Learning rate η ∈R+, Lagrange multiplier update rate µ ∈R+
Require: Optimizer Opt(·) for minimization, Regularization function R(θ) (e.g. L2-norm)

1: # Initialization
2: t ←0
3: θt ←Initialize(L(·))

4:
λj
t ←0 for j = 1, · · · , J

5:
κj ←θj
t −Initialize(θj
0) for j = 1, · · · , J
6: # Training
7: for Xt, yt ∼D do
8:
θt+1 ←θt + Opt(L(θt, Xt, yt), η)
▷Classic parameter update using, e.g., Adam.

9:
for each regularized parameter group θj
t in θt do

10:
λj
t+1 ←
 
λj
t + µ · (R(θj
t ) −κj)
+

11:
θj
t+1 ←θj
t+1 −∇θjR(θj
t ) · λj
t+1

12:
if λj
t = 0 and λj
t−1 > 0 then
▷Update κj if the constraints are not active.

13:
κj ←R(θj
t )
14:
end if
15:
end for
16:
t ←t + 1
17: end for

The experimental results in Figure E.1 also show that the adaptation of the upper bound during the
training is not beneficial. While it does not harm the performance, it also does not lead to a substantial
improvement. We therefore do not use it to keep our method as simple as possible.

18


---Page Break---
E
Experiments on Image Classification (CIFAR100)

For the κ initialization Kappa-K, we use a range of κ = [0.005, . . . , 0.16], for Kappa-kI0 a range
of k = [4, . . . , 256], and for Kappa-WS a range of s = [250, . . . , 4000] steps. We use a learning rate
warmup of 500 steps followed by a closing annealing. This is 2.5% of the total training steps (20k).
For a detailed list of training hyperparameters, we refer the reader to Table E.1.

We found that initializing with Kappa-kI0 performs better than selecting a uniform κ in Kappa-K.
This may be explained by the value of the regularization function depending on the size of the jointly
regularized parameter matrix and initialization method. The warm start κ initialization method,
Kappa-WS, performed the best. The best configuration with CPR outperforms weight decay and the
choice of hyperparameters seems to be more robust.

Table E.1: Hyperparameters of the ResNet18 on CIFAR100 experiment.

Parameter
Value

Seed
1,2,3
Dataset
CIFAR100
Batch size
128
Training Steps
20000
Model
ResNet18
Optimizer
AdamW / Adam+Rescaling / AdamCPR
Learning Rate
0.001
Beta1
0.9
Beta2
0.98
Weight Decay
0.1
Lr Schedule
Cosine with warmup
Lr Warmup Steps
500
Lr Decay Factor
0.1
Rescale Alpha
0, 0.8 ...16
CPR−µ
1.0
CPR-κ
0.8 ...16
CPR-k
4 ...256
CPR-κ warm-start steps
250 ...16000
Adaptive Bounds
False / True

19


---Page Break---
0.005
0.01
0.02
0.04
0.08
0.16
Kappa

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.5
±0.16

70.3
±0.23

70.1
±0.17

70.3
±0.12

70.4
±0.19

70.5
±0.14

74.9
±0.12

74.3
±0.42

74.8
±0.40

74.5
±0.23

74.6
±0.56

74.6
±0.35

75.4
±0.39

75.3
±0.16

75.1
±0.42

75.1
±0.28

74.9
±0.23

74.9
±0.05

75.3
±0.13

75.8
±0.30

75.6
±0.18

74.8
±0.53

74.5
±0.31

74.6
±0.26

72.1
±0.10

73.4
±0.31

75.3
±0.43

75.8
±0.09

75.5
±0.15

74.9
±0.42

63.6
±0.22

66.6
±0.23

68.9
±0.40

70.5
±0.13

72.0
±0.10

72.3
±0.50

38.9
±1.62

46.0
±1.05

52.8
±0.82

57.1
±1.27

59.7
±0.37

62.7
±0.60

AdamCPR (Kappa-K)

0.005
0.01
0.02
0.04
0.08
0.16
Kappa

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.5
±0.18

70.5
±0.17

70.4
±0.44

70.4
±0.27

70.1
±0.15

70.1
±0.23

74.5
±0.26

74.4
±0.15

74.4
±0.12

74.3
±0.55

74.6
±0.04

74.9
±0.31

75.1
±0.18

75.4
±0.39

74.9
±0.26

75.0
±0.17

75.1
±0.32

75.2
±0.28

75.0
±0.34

75.7
±0.12

75.2
±0.15

74.7
±0.28

74.6
±0.22

74.5
±0.44

71.7
±0.29

73.3
±0.27

74.5
±0.33

75.4
±0.30

75.8
±0.10

75.1
±0.15

62.4
±1.07

64.5
±0.61

68.0
±0.24

70.0
±0.44

71.7
±0.08

72.3
±0.58

36.2
±1.15

39.2
±2.15

44.8
±3.62

52.4
±1.83

56.2
±2.74

60.9
±0.81

Adam+AdaCPR (Kappa-K)

4
8
16
32
64
128
256
Factor k

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.5
±0.02

70.1
±0.25

70.3
±0.24

70.3
±0.10

70.3
±0.20

70.2
±0.18

70.2
±0.10

74.8
±0.35

74.5
±0.03

74.7
±0.21

74.4
±0.04

74.2
±0.23

74.6
±0.16

74.7
±0.28

75.9
±0.09

76.4
±0.50

76.1
±0.23

75.6
±0.17

75.2
±0.25

74.9
±0.25

75.0
±0.08

73.7
±0.37

74.5
±0.50

75.3
±0.27

75.4
±0.43

75.6
±0.16

75.6
±0.24

75.1
±0.62

67.6
±0.39

70.2
±0.37

72.4
±0.27

73.9
±0.07

75.1
±0.34

75.7
±0.32

76.0
±0.37

55.5
±0.59

59.1
±0.29

63.7
±0.84

67.1
±0.50

69.2
±0.24

70.6
±0.35

71.6
±0.17

28.5
±3.15

31.8
±1.16

38.3
±2.00

44.9
±2.53

50.7
±0.66

56.0
±0.57

60.0
±0.51

AdamCPR (Kappa-kI0)

4
8
16
32
64
128
256
Factor k

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.2
±0.14

70.3
±0.08

70.5
±0.39

70.2
±0.35

70.1
±0.10

70.2
±0.37

70.2
±0.23

75.0
±0.10

74.4
±0.03

74.6
±0.03

74.7
±0.19

74.4
±0.05

74.3
±0.32

74.4
±0.28

75.7
±0.31

76.3
±0.24

75.9
±0.05

75.6
±0.06

75.1
±0.13

75.3
±0.12

75.0
±0.04

74.0
±0.14

74.9
±0.08

75.4
±0.42

75.2
±0.26

75.5
±0.14

75.8
±0.11

75.2
±0.24

68.3
±0.50

70.2
±0.22

71.7
±0.01

73.7
±0.63

74.4
±0.36

75.7
±0.18

75.5
±0.17

52.1
±2.02

59.0
±1.22

62.2
±0.52

65.4
±0.68

68.5
±0.26

70.1
±0.55

71.9
±0.57

25.5
±4.04

25.5
±5.07

28.2
±6.41

37.3
±7.69

40.4
±8.94

49.7
±3.15

56.4
±0.73

Adam+AdaCPR (Kappa-kI0)

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR (Kappa-WS)

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

71.0
±0.03

71.1
±0.06

70.8
±0.34

70.7
±0.33

70.5
±0.14

70.3
±0.09

70.3
±0.25

75.7
±0.09

76.1
±0.32

75.9
±0.41

75.6
±0.35

74.9
±0.41

74.3
±0.28

74.6
±0.28

74.9
±0.52

75.5
±0.21

76.9
±0.21

76.4
±0.26

75.5
±0.40

75.2
±0.28

74.9
±0.17

70.5
±0.41

73.9
±0.68

75.7
±0.20

75.4
±0.17

74.5
±0.21

74.6
±0.23

74.2
±0.47

63.7
±0.87

72.3
±0.39

75.4
±0.25

75.3
±0.38

74.1
±0.48

73.6
±0.58

73.5
±0.55

53.7
±0.01

68.8
±0.44

72.9
±0.14

72.1
±0.41

70.6
±0.60

70.3
±0.16

69.4
±0.60

32.1
±2.01

59.0
±3.32

64.8
±0.54

64.7
±0.57

65.4
±0.05

65.2
±1.12

63.9
±1.01

Adam+AdaCPR (Kappa-WS)

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.1: Percentage of correct labels of the ResNet18 trained on the CIFAR100 with use of Adam
with CPR (left) and AdaCPR (right) with use of the three different initialization techniques from
Section 4.3, from top to bottom: Kappa-K, Kappa-kI0, and Kappa-WS. The elements in the heat
map are experiments with different learning rates and regularization hyperparameters. Each element
is colored according to the mean accuracy of three random seeds and the numbers are the mean
accuracy and standard deviation of the experiments.

20


---Page Break---
250
500
1000
2000
4000
8000
16000
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR L2 regularization

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.9
±0.16

70.9
±0.21

70.7
±0.03

71.0
±0.16

70.6
±0.17

70.1
±0.03

70.1
±0.20

75.8
±0.19

76.2
±0.63

75.9
±0.45

75.6
±0.18

74.7
±0.26

74.5
±0.45

74.8
±0.09

73.4
±0.05

74.6
±0.33

76.5
±0.49

76.2
±0.22

75.5
±0.08

75.2
±0.02

75.3
±0.08

62.5
±0.33

71.0
±0.62

75.6
±0.09

75.6
±0.21

74.6
±0.53

74.2
±0.17

74.5
±0.21

44.7
±2.29

69.0
±1.11

75.5
±0.72

75.1
±0.26

73.3
±nan

73.4
±nan

72.5
±nan

38.2
±nan

64.2
±0.99

72.2
±nan

72.4
±nan

70.1
±nan

68.6
±nan

68.5
±0.72

28.1
±nan

58.5
±nan

67.1
±nan

66.1
±nan

64.4
±0.37

63.9
±nan

63.3
±nan

AdamCPR std regularization

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.2: Percentage of correct labels of the ResNet18 trained on the CIFAR100 with the use of
AdamCPR using L2 regularization measure (left) and standard deviation as regularization measure
(right). The elements in the heat map are experiments with different learning rates and warm start
steps (s of Kappa-WS). Each element is colored according to the mean accuracy of three random
seeds and the numbers are the mean accuracy and standard deviation of the experiments.

21


---Page Break---
0.0
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.1
±0.38

70.5
±0.13

70.5
±0.35

70.5
±0.15

70.7
±0.27

74.5
±0.15

74.2
±0.44

74.5
±0.25

74.5
±0.38

74.9
±0.26

74.7
±0.13

75.0
±0.03

74.4
±0.15

74.7
±0.29

75.7
±0.28

73.5
±0.29

74.0
±0.55

73.8
±0.52

73.9
±0.38

75.0
±0.27

72.2
±0.40

72.8
±0.16

72.2
±0.23

73.6
±0.32

73.8
±0.38

67.1
±0.43

68.1
±0.60

67.8
±0.35

70.2
±0.35

65.5
±1.05

61.2
±1.33

62.4
±1.35

62.6
±0.92

64.2
±0.49

33.9
±2.37

AdamW lr-warmup 250 steps

(Kappa-IP)

71.0
±0.24

75.9
±0.05

76.2
±0.28

75.6
±0.12

75.0
±0.63

71.6
±0.68

64.7
±0.87

AdamCPR

250
500
1000
2000
4000
8000
16000
Warm start steps (Kappa-WS)

71.0
±0.20

71.4
±0.34

71.4
±0.06

70.7
±0.27

70.5
±0.16

70.3
±0.06

70.1
±0.35

75.8
±0.27

76.1
±0.43

75.6
±0.48

75.4
±0.38

75.0
±0.28

74.3
±0.20

74.5
±0.17

74.8
±0.25

76.0
±0.34

76.4
±0.20

75.9
±0.16

75.6
±0.08

74.9
±0.20

75.0
±0.12

72.0
±0.44

75.0
±0.33

76.0
±0.10

75.0
±0.21

74.3
±0.18

74.2
±0.22

73.9
±0.40

68.0
±0.85

73.8
±0.64

75.9
±0.21

74.3
±0.29

73.4
±0.48

72.7
±0.43

72.8
±0.31

59.4
±1.87

68.8
±1.24

71.7
±0.76

70.5
±0.13

69.0
±0.10

67.8
±1.00

67.2
±0.40

44.5
±2.60

59.6
±0.97

65.3
±1.51

65.0
±0.74

62.6
±0.79

61.9
±1.23

61.7
±1.44

AdamCPR lr-warmup 250 steps

0.0
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.2
±0.35

70.4
±0.19

70.5
±0.32

70.3
±0.20

70.6
±0.20

74.3
±0.39

74.7
±0.12

74.8
±0.19

74.7
±0.35

75.3
±0.13

75.1
±0.29

75.0
±0.25

74.9
±0.21

74.9
±0.15

75.4
±0.26

74.2
±0.45

74.3
±0.43

74.4
±0.49

75.1
±0.68

74.8
±0.47

73.2
±0.15

73.3
±0.53

73.4
±0.13

74.4
±0.19

72.8
±0.53

68.2
±0.64

68.1
±0.91

69.8
±0.17

71.8
±0.17

54.5
±2.44

63.2
±1.06

63.9
±0.71

65.1
±0.79

62.4
±1.65

16.6
±1.70

AdamW lr-warmup 500 steps

(Kappa-IP)

71.0
±0.45

75.8
±0.33

76.3
±0.15

75.5
±0.25

75.1
±0.24

72.5
±0.21

66.0
±0.77

AdamCPR

250
500
1000
2000
4000
8000
16000
Warm start steps (Kappa-WS)

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR lr-warmup 500 steps

0.0
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.2
±0.27

70.1
±0.13

70.2
±0.14

70.4
±0.47

70.6
±0.25

74.6
±0.09

74.7
±0.39

74.7
±0.35

74.7
±0.22

74.9
±0.26

75.5
±0.24

75.4
±0.30

75.4
±0.30

75.5
±0.21

76.0
±0.31

74.9
±0.13

75.1
±0.39

74.8
±0.07

75.1
±0.25

75.5
±0.35

74.4
±0.59

74.2
±0.65

74.3
±0.50

74.9
±0.25

74.4
±0.44

71.2
±0.26

71.3
±0.25

71.6
±0.29

73.0
±0.06

67.6
±0.44

64.7
±0.64

65.4
±0.35

65.7
±0.44

65.1
±1.49

33.4
±12.42

AdamW lr-warmup 1k steps

(Kappa-IP)

71.0
±0.20

76.0
±0.27

76.3
±0.38

75.9
±0.18

75.8
±0.65

74.0
±0.34

67.6
±0.38

AdamCPR

250
500
1000
2000
4000
8000
16000
Warm start steps (Kappa-WS)

70.9
±0.25

71.2
±0.10

71.2
±0.35

71.0
±0.34

70.6
±0.10

70.4
±0.21

70.5
±0.06

76.1
±0.13

76.0
±0.11

75.8
±0.11

75.6
±0.24

75.2
±0.46

74.7
±0.30

74.7
±0.23

74.8
±0.40

74.9
±0.18

76.4
±0.23

76.5
±0.18

75.8
±0.22

75.4
±0.55

75.6
±0.31

69.8
±0.60

72.2
±0.54

75.4
±0.40

75.8
±0.15

75.2
±0.07

74.9
±0.17

74.8
±0.49

62.1
±1.53

69.8
±0.38

75.7
±0.18

75.7
±0.35

74.8
±0.42

74.6
±0.48

74.4
±0.69

53.4
±1.55

67.6
±0.31

73.8
±0.17

73.5
±0.43

72.0
±0.71

71.4
±0.90

71.3
±0.29

34.5
±1.81

59.8
±0.04

68.1
±0.71

67.2
±0.36

66.0
±0.31

65.5
±0.23

66.4
±0.24

AdamCPR lr-warmup 1k steps

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.3: Comparison of AdamW and AdamCPR with different learning rate warm-up steps. The
Figure shows the percentage of correct labels of the ResNet18 trained on the CIFAR100 with the
use of AdamW (left side), AdamCPR (Kappa-IP) (middle), and AdamCPR (Kappa-WS) (right side)
with learning rate warm-up steps between 250 and 1000 steps. The elements in the heat map are
experiments with different learning rates and regularization hyperparameters. Each element is colored
according to the mean accuracy of three random seeds and the numbers are the mean accuracy and
standard deviation of the experiments.

22


---Page Break---
0.0
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.2
±0.35

70.4
±0.19

70.5
±0.32

70.3
±0.20

70.6
±0.20

74.3
±0.39

74.7
±0.12

74.8
±0.19

74.7
±0.35

75.3
±0.13

75.1
±0.29

75.0
±0.25

74.9
±0.21

74.9
±0.15

75.4
±0.26

74.2
±0.45

74.3
±0.43

74.4
±0.49

75.1
±0.68

74.8
±0.47

73.2
±0.15

73.3
±0.53

73.4
±0.13

74.4
±0.19

72.8
±0.53

68.2
±0.64

68.1
±0.91

69.8
±0.17

71.8
±0.17

54.5
±2.44

63.2
±1.06

63.9
±0.71

65.1
±0.79

62.4
±1.65

16.6
±1.70

AdamW

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR (Kappa-WS)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.3
±0.37

70.2
±0.17

70.2
±0.13

70.5
±0.24

74.3
±0.39

74.5
±0.36

74.5
±0.52

74.8
±0.62

75.0
±0.55

75.1
±0.30

75.0
±0.26

75.6
±0.07

74.3
±0.08

74.3
±0.30

74.5
±0.09

75.6
±0.22

73.6
±0.33

73.1
±0.27

74.0
±0.46

76.1
±0.09

69.5
±0.33

69.8
±0.74

71.8
±0.38

70.1
±0.17

63.1
±1.10

64.7
±0.36

66.4
±0.94

47.3
±1.57

AdamW WD schedule (decreasing x0.1)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.2
±0.29

70.0
±0.22

70.2
±0.25

70.6
±0.51

74.5
±0.13

74.6
±0.12

74.3
±0.29

74.8
±0.28

75.3
±0.05

75.0
±0.23

75.2
±0.22

75.6
±0.20

74.4
±0.47

74.1
±0.61

74.6
±0.34

75.7
±0.51

73.1
±0.09

73.3
±0.32

74.3
±0.19

75.8
±0.40

69.6
±0.23

69.7
±0.27

71.5
±0.42

70.2
±0.54

64.1
±1.30

64.3
±0.47

67.2
±0.75

48.0
±2.29

AdamW WD schedule (decreasing x0.01)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.3
±0.09

70.4
±0.67

70.4
±0.09

71.0
±0.26

74.7
±0.04

74.6
±0.35

74.6
±0.24

75.1
±0.35

75.1
±0.21

75.0
±0.31

75.1
±0.18

73.5
±0.09

74.4
±0.10

74.4
±0.48

74.6
±0.39

69.8
±0.86

73.1
±0.10

73.3
±0.31

73.4
±0.46

61.7
±0.15

68.9
±0.23

70.0
±0.12

68.3
±0.35

38.2
±1.95

63.1
±0.21

64.5
±1.04

49.7
±0.84

9.1
±2.14

AdamW WD schedule (increasing x10)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.2
±0.18

70.2
±0.34

70.7
±0.15

70.9
±0.18

74.7
±0.23

74.4
±0.34

75.0
±0.17

67.8
±0.93

75.0
±0.33

75.2
±0.03

73.8
±0.39

58.5
±0.92

74.3
±0.37

74.6
±0.41

69.3
±0.62

38.7
±0.62

73.5
±0.55

72.8
±0.27

61.0
±0.33

13.8
±6.81

70.2
±0.50

67.8
±0.71

37.0
±1.85

4.1
±1.41

64.7
±0.65

50.2
±1.50

7.9
±0.66

2.4
±1.32

AdamW WD schedule (increasing x100)

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.4: Comparison of AdamW, AdamCPR, and weight decay scheduling similar to [10, 11]. The
Figure shows the percentage of correct labels of the ResNet18 trained on the CIFAR100 with the use
of AdamW (top left), AdamCPR (Kappa-WS) (top right), and Adam with weight decay scheduling.
We evaluated the task with cosine decreasing weight decay to 0.1 and 0.01 times the initial weight
decay value and with cosine increasing weight decay to 10 and 100 times the initial weight decay
value. The elements in the heat map are experiments with different learning rates and regularization
hyperparameters. Each element is colored according to the mean accuracy of three random seeds
and the numbers are the mean accuracy and standard deviation of the experiments. It should be
mentioned that Yun et al. [9] also performed weight decay scheduling on CIFAR100 with the use of a
ResNet18. Since their code was not published, we point to Figure 3 of their experimental results,
where an accuracy of around 60% was reported, which is below our AdamW baseline.

23


---Page Break---
0.0
0.0001 0.001
0.01
0.1
Weight Decay

0.000316

0.001

0.00346

0.01

62.0
±0.52

61.6
±0.78

62.3
±0.11

62.0
±0.95

62.5
±0.28

62.7
±0.54

62.9
±0.42

62.3
±0.34

62.7
±0.06

63.2
±0.20

61.2
±0.83

61.3
±1.00

60.9
±0.47

61.1
±0.11

60.7
±1.50

58.8
±0.52

59.8
±1.15

59.0
±0.28

59.6
±0.74

57.7
±1.02

AdamW

(Kappa-IP)

62.3
±0.89

62.5
±0.26

60.9
±0.02

59.5
±0.25

AdamCPR

500
1k
2k
4k
8k
Warm start steps (Kappa-WS)

63.0
±0.10

63.4
±0.06

62.7
±0.58

62.5
±0.58

62.0
±0.68

60.2
±0.37

62.3
±0.26

63.9
±0.74

63.3
±0.74

62.4
±0.00

55.4
±3.75

59.6
±1.41

60.7
±0.96

61.1
±1.22

61.5
±0.38

56.5
±0.10

60.0
±0.21

60.6
±0.38

59.9
±0.93

59.9
±0.35

AdamCPR

60

65

70

75

80

% Correct Label

Figure E.5: Percentage of correct labels of the ResNet18 trained on the CIFAR100-C with use of
AdamW (left), AdamCPR with Kappa-IP (middle) and AdamCPR with Kappa-WS (right). The
elements in the heat map are experiments with different learning rates and regularization hyperpa-
rameters. Each element is colored according to the mean accuracy of three random seeds and the
numbers are the mean accuracy and standard deviation of the experiments. We see that AdamCPR
outperforms AdamW which could indicate that CPR leads to a more robust optimization. We see
that AdamCPR performs better than AdamW with Kappa-WS but not with Kappa-IP. Kappa-IP does
not fail and performs better than the average weight decay performance. None of the optimizer and
hyperparameter configurations lead to an outstanding performance on this task, we wouldn’t claim
that CPR is particularly good for noisy data.

0.0
1e-4
1e-3
1e-2
1e-1
Weight Decay

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.7
±0.19

70.8
±0.34

72.1
±0.24

75.8
±0.35

64.2
±0.28

73.2
±0.14

73.7
±0.28

75.3
±0.36

73.4
±0.40

51.7
±1.41

74.2
±0.43

75.0
±0.40

76.9
±0.10

68.2
±0.07

32.3
±3.05

73.8
±0.11

75.7
±0.21

75.1
±0.50

57.3
±1.19

14.8
±1.43

SGD

(Kappa-IP)

71.1
±0.19

73.6
±0.27

75.0
±0.25

74.7
±0.16

SGDCPR

250
500
1k
2k
4k
8k
Warm start steps (Kappa-WS)

71.6
±0.28

71.6
±0.44

71.5
±0.13

71.3
±0.09

71.2
±0.15

70.9
±0.23

74.3
±0.23

74.4
±0.16

74.6
±0.12

74.1
±0.10

73.6
±0.43

73.5
±0.11

76.3
±0.36

76.0
±0.18

76.1
±0.35

75.4
±0.36

74.9
±0.44

74.3
±0.08

76.7
±0.21

77.3
±0.29

76.8
±0.37

76.2
±0.14

75.2
±0.20

74.2
±0.20

SGDCPR

60

65

70

75

80

% Correct Label

Figure E.6: Percentage of correct labels of the ResNet18 trained on the CIFAR100 with use of SGD
with weight decay (left), SGD with CPR and Kappa-IP (middle) and SGD with CPR and Kappa-WS
(right). The elements in the heat map are experiments with different learning rates and regularization
hyperparameters. Each element is colored according to the mean accuracy of three random seeds and
the numbers are the mean accuracy and standard deviation of the experiments.

24


---Page Break---
0.0
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.2
±0.35

70.4
±0.19

70.5
±0.32

70.3
±0.20

70.6
±0.20

74.3
±0.39

74.7
±0.12

74.8
±0.19

74.7
±0.35

75.3
±0.13

75.1
±0.29

75.0
±0.25

74.9
±0.21

74.9
±0.15

75.4
±0.26

74.2
±0.45

74.3
±0.43

74.4
±0.49

75.1
±0.68

74.8
±0.47

73.2
±0.15

73.3
±0.53

73.4
±0.13

74.4
±0.19

72.8
±0.53

68.2
±0.64

68.1
±0.91

69.8
±0.17

71.8
±0.17

54.5
±2.44

63.2
±1.06

63.9
±0.71

65.1
±0.79

62.4
±1.65

16.6
±1.70

AdamW

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR (Kappa-WS)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.4
±0.28

70.1
±0.16

70.5
±0.14

70.6
±0.14

74.6
±0.34

74.9
±0.18

74.8
±0.49

75.1
±0.09

74.9
±0.33

75.2
±0.48

75.1
±0.13

75.9
±0.26

74.4
±0.39

74.4
±0.23

74.5
±0.14

75.2
±0.43

73.4
±0.24

73.6
±0.28

74.1
±0.58

73.8
±0.16

69.4
±0.25

69.9
±0.58

72.3
±0.38

65.9
±0.56

64.4
±1.23

43.5
±36.82

44.3
±37.46

13.5
±21.58

Adam+AdaDecay (alpha 1.0)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.3
±0.33

70.3
±0.36

70.4
±0.26

70.6
±0.16

74.7
±0.20

74.6
±0.09

74.5
±0.32

74.9
±0.33

75.1
±0.11

75.1
±0.27

75.2
±0.18

75.5
±0.06

74.2
±0.38

74.3
±0.30

74.9
±0.40

75.2
±0.17

73.5
±0.29

73.3
±0.30

74.3
±0.51

74.2
±0.22

69.0
±0.56

69.2
±0.31

72.2
±0.52

66.7
±0.25

42.5
±35.99

43.9
±37.14

65.4
±0.79

34.7
±4.99

Adam+AdaDecay (alpha 2.0)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.4
±0.16

70.3
±0.17

70.2
±0.08

70.3
±0.13

74.3
±0.12

74.4
±0.32

74.6
±0.24

74.8
±0.10

74.9
±0.09

75.1
±0.10

75.1
±0.34

75.8
±0.16

74.3
±0.27

74.6
±0.21

74.4
±0.46

74.8
±0.31

73.1
±0.56

73.4
±0.47

73.8
±0.52

74.1
±0.24

69.0
±0.46

69.3
±0.49

72.1
±0.66

67.4
±0.39

43.0
±36.37

65.3
±0.69

65.4
±1.13

13.4
±21.47

Adam+AdaDecay (alpha 4.0)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.2
±0.10

70.3
±0.36

70.4
±0.13

70.4
±0.13

74.4
±0.31

74.8
±0.03

74.7
±0.10

74.6
±0.37

75.2
±0.22

75.1
±0.27

75.1
±0.15

75.7
±0.35

74.3
±0.28

74.4
±0.14

74.9
±0.15

75.2
±0.07

73.1
±0.72

73.5
±0.67

73.9
±0.23

74.1
±0.22

69.4
±0.31

69.6
±0.16

72.1
±0.47

67.8
±0.15

64.0
±0.89

65.1
±0.35

65.6
±0.71

42.2
±5.61

Adam+AdaDecay (alpha 8.0)

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.7: Comparison of AdamW, AdamCPR, and Adam with AdaDecay [13]. The Figure shows
the percentage of correct labels of the ResNet18 trained on the CIFAR100 with the use of AdamW
(top left), AdamCPR (Kappa-WS) (top right), and Adam with AdaDecay with different (1.0, 2.0, 4.0,
8.0) values for the alpha hyperparameter in AdaDecay. The elements in the heat map are experiments
with different learning rates and regularization hyperparameters. Each element is colored according
to the mean accuracy of three random seeds and the numbers are the mean accuracy and standard
deviation of the experiments.

25


---Page Break---
0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.4
±0.19

70.5
±0.32

70.3
±0.20

70.6
±0.20

74.7
±0.12

74.8
±0.19

74.7
±0.35

75.3
±0.13

75.0
±0.25

74.9
±0.21

74.9
±0.15

75.4
±0.26

74.3
±0.43

74.4
±0.49

75.1
±0.68

74.8
±0.47

73.3
±0.53

73.4
±0.13

74.4
±0.19

72.8
±0.53

68.1
±0.91

69.8
±0.17

71.8
±0.17

54.5
±2.44

63.9
±0.71

65.1
±0.79

62.4
±1.65

16.6
±1.70

AdamW

250
500
1000
2000
4000
8000
16000
Warm start steps

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

71.2
±0.37

71.2
±0.25

71.3
±0.17

70.8
±0.30

70.9
±0.13

70.3
±0.10

70.1
±0.07

76.0
±0.19

75.7
±0.08

76.1
±0.21

75.5
±0.33

75.3
±0.24

74.4
±0.32

74.6
±0.20

74.8
±0.36

75.7
±0.05

76.6
±0.22

76.2
±0.18

75.6
±0.14

75.6
±1.01

75.2
±0.67

70.9
±0.34

73.8
±0.29

75.6
±0.25

75.4
±0.27

74.9
±0.06

74.4
±0.46

74.6
±0.13

64.8
±0.10

72.9
±0.27

75.7
±0.16

75.0
±0.17

74.1
±0.39

73.4
±0.31

73.1
±0.59

57.7
±1.08

69.7
±0.64

72.7
±0.24

71.7
±0.41

70.3
±0.55

69.0
±0.33

68.6
±0.18

41.4
±3.33

63.2
±2.76

67.0
±0.69

65.7
±2.05

64.4
±0.83

63.9
±0.21

63.9
±1.07

AdamCPR (Kappa-WS)

0.0001
0.001
0.01
0.1
Initial Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

70.4
±0.28

70.1
±0.16

70.5
±0.14

70.6
±0.14

74.6
±0.34

74.9
±0.18

74.8
±0.49

75.1
±0.09

74.9
±0.33

75.2
±0.48

75.1
±0.13

75.9
±0.26

74.4
±0.39

74.4
±0.23

74.5
±0.14

75.2
±0.43

73.4
±0.24

73.6
±0.28

74.1
±0.58

73.8
±0.16

69.4
±0.25

69.9
±0.58

72.3
±0.38

65.9
±0.56

64.4
±1.23

43.5
±36.82

44.3
±37.46

13.5
±21.58

Adam+AdaDecay (alpha 1.0)

0.0001
0.001
0.01
0.1
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

70.5
±0.39

70.1
±0.23

70.2
±0.06

70.4
±0.15

74.5
±0.30

74.8
±0.40

74.4
±0.27

74.3
±0.15

75.0
±0.21

75.4
±0.36

75.0
±0.19

75.2
±0.22

74.5
±0.26

74.4
±0.41

74.2
±0.20

74.1
±0.57

73.4
±0.02

73.0
±0.39

73.1
±0.63

73.1
±0.31

69.6
±0.38

69.3
±0.58

69.3
±0.39

69.3
±0.55

63.7
±1.45

63.4
±0.32

63.2
±0.35

63.3
±0.50

Adam+AWD

1e-06
1e-05
0.0001
0.001
0.01
Weight Decay

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

Learning Rate

47.2
±0.33

47.2
±0.24

43.8
±0.93

11.1
±0.26

1.0
±0.00

61.9
±0.23

61.5
±0.16

55.7
±0.46

17.2
±1.33

1.0
±0.00

69.8
±0.10

70.0
±0.30

65.6
±0.22

24.4
±4.87

1.0
±0.00

74.5
±0.23

74.2
±0.45

72.9
±0.53

18.0
±3.05

1.0
±0.00

75.3
±0.16

75.1
±0.26

74.9
±0.22

1.1
±0.04

1.0
±0.00

75.3
±0.32

75.4
±0.16

75.7
±0.20

3.9
±0.90

1.0
±0.00

71.6
±0.18

71.6
±0.03

71.8
±0.56

71.4
±0.10

3.9
±5.05

AMOS

0.8
1.0
2.0
4.0
8.0
16.0
Factor of Initial Total Weight Norm

1e-4.0

1e-3.5

1e-3.0

1e-2.5

1e-2.0

1e-1.5

1e-1.0

71.9
±0.19

70.6
±0.24

63.7
±0.21

55.2
±0.35

45.5
±0.86

29.1
±0.47

75.9
±0.42

75.5
±0.17

72.9
±0.24

66.2
±0.61

56.7
±0.63

43.1
±1.27

74.7
±0.24

75.8
±0.13

75.3
±0.20

73.4
±0.13

66.0
±0.23

56.1
±1.05

65.8
±0.21

68.4
±0.15

74.3
±0.71

75.9
±0.35

71.2
±0.68

62.5
±0.15

48.4
±0.74

56.0
±1.15

65.6
±1.03

70.8
±0.18

73.0
±0.67

68.0
±0.53

27.8
±3.54

31.0
±2.97

40.9
±8.59

48.4
±3.16

63.0
±0.74

68.7
±0.32

18.8
±4.20

24.2
±1.40

22.1
±9.43

30.2
±2.33

41.0
±1.90

48.9
±1.89

Adam+Rescaling

60.0

62.5

65.0

67.5

70.0

72.5

75.0

77.5

80.0

Figure E.8: Percentage of correct labels of the ResNet18 trained on the CIFAR100 with AdamW,
AdamCPR, AdaDecay [13], AWD [14], AMOS [15], and Rescaling. We use different values of
weight decay for AdamW, AdaDecay, AWD, and AMOS. For Adam with Rescaling, we use different
factors of the initial total weight norm. AdamCPR uses Kappa-WS. We use a learning rate warm-up of
500 steps and the best Kappa-WS value is 2× the warm-up steps. Each element is colored according
to the mean accuracy of three random seeds and the numbers are the mean accuracy and standard
deviation of the experiments.

26


---Page Break---
F
Experiments on Image Classification (ImageNet)

Table F.1: Hyperparameters for the DeiT small experiments on ImageNet.
ImageNet
Pretraining

AdamW
AdamCPR

weight decay
Kappa WS
Kappa IP
(x lr-warmup)
0.005
0.05
0.5
1x
2x
4x

Model Architecture
DeiT-Small (patch size 16, image size 224)
Learning Rate
1e-3
Warmup Epochs
5
Epochs
300
Batch Size
256
Optimizer
AdamW
AdamCPR
Weight Decay
0.005
0.05
0.5
-
κ Init Param
-
6280
12560
25120
-
κ Init Method
-
warm_start
-
Scheduler
cosine
Auto-augment
rand-m9-mstd0.5
Mixup Alpha
0.8
CutMix Alpha
1.0
Random Erase Prob
0.25
AMP
Yes
TorchScript
Yes
Pin Memory
Yes
Data Parallel Jobs
8

Table F.2: Hyperparameters for the DeiT base experiments on ImageNet.
ImageNet
Pretraining

AdamW
AdamCPR

weight decay
Kappa WS
Kappa IP
(x lr-warmup)
0.005
0.05
0.5
1x
2x
4x

Model Architecture
DeiT-Base (patch size 16, image size 224)
Learning Rate
1e-3
Warmup LR
1e-6
Min LR
1e-5
Warmup Epochs
5
Epochs
300
Batch Size
256
Optimizer
AdamW
AdamCPR
Weight Decay
0.005
0.05
0.5
-
κ Init Param
-
6280
12560
25120
-
Drop Path Rate
0.1
Mixup Alpha
0.8
CutMix Alpha
1.0
Color Jitter Factor
0.3
Random Erase Prob
0.25
Train Interpolation
Bicubic

27


---Page Break---
G
Training Dynamics of GPT2

0.0

0.1

∥θj∥2
2

kappa initialization

Optimization Steps

0

5

Layer 1

Attn Weight

∆t∥θj∥2
2

×10−6

Optimization Steps t

0.0000

0.0001

λj

0.0

0.1

∥θj∥2
2

Optimization Steps

0

5

Layer 5

FC1 Weight

∆t∥θj∥2
2

×10−6

Optimization Steps t

0.0000

0.0001

λj

0.0

0.1

∥θj∥2
2

Optimization Steps

0

1

Layer 10

FC2 Weight

∆t∥θj∥2
2

×10−5

Optimization Steps t

0.0000

0.0001

λj

0
25000
50000
75000
100000
125000
150000
175000
200000
Optimization Steps t

3.00

3.25

Validation

Loss

GPT2s Training Dynamics of AdamW (blue) and AdamCPR (green) with Kappa-IP

Figure G.1: The training dynamics of AdamW and AdamCPR with Kappa-IP of one layer in a
GPT2s training run. The upper plot shows the squared L2 norm of the attention weight in the first
layer. Below we see the gradient of the squared L2 norm regarding the training steps, after the first
inflection point Kappa-IP initializes kappa and starts the regularization. The third plot shows CPR’s
lambda enforcing the constraint on kappa. The six plots below show the dynamics for the first weight
matrix of the feed-forward block in the 5th layer and the second weight matrix of the feed-forward
block in the 10th layer. At the bottom, we see the validation loss. We see that Kappa-IP initializes
different layers at different time steps, e.g. layer 5 FC1 before layer 1 attention weights. While
weight decay leads to a steady increase of the squared L2 norm for the first quarter of the training,
CPR regularizes much earlier and avoids over-adaption. AdamW converges faster in the beginning of
the training but CPR leads to a more linear improvement and a better final performance.

28


---Page Break---
H
Experiments on Language Modelling

For an efficient implementation, we use flash attention [45] and rotary position embedding [46]. The
complete hyperparameters can be found in Appendix H. The GPT2s and GPT2m models are trained
on 8 A100 GPUs up to 28h. A detailed runtime analysis can be found in Appendix I

Table H.1: Comparison of AdamW, AdamCPR, AdaDecay, AWD, and AMOS on GPT2s trained on
OpenWebText. For AdamW and AdamCPR we report the mean across three random seeds. For the
other methods, only a single seed is reported. The number next to the optimizer name is the weight
decay coefficient γ except for AdamCPR, here it is the number of warm start steps s for Kappa-WS.

Method
Perplexity ↓

AdamW
1e-3
18.45 ± 0.0039
1e-2
18.23 ± 0.0113
1e-1
18.86 ± 0.0169

AdamCPR (Kappa-WS)
5k (1x)
18.02 ± 0.0258
10k (2x)
18.03 ± 0.0178
20k (4x)
18.24 ± 0.0320

AdamCPR (Kappa-IP)
17.94

Adam Adadecay
1e-3
18.42
1e-2
18.24
1e-1
18.87

Adam AWD
1e-3
18.42
1e-2
18.47
1e-1
18.49

AMOS
1e-3
NaN
1e-2
NaN
1e-1
NaN

29


---Page Break---
Table H.2: Hyperparameters of the language modeling task (GPT2 and Openwebtext).

Parameter
GPT2s
GPT2m

GPUs
8x A100 40GB
Gradient Clip Val
1.0
Max Steps
200k
Precision
bf16-mixed
Seed
1234
Beta1
0.9
Beta2
0.99
Eps
1.0 × 10−9
Bias Weight Decay
False
Normalization Weight Decay
False
Lr Num Warmup Steps
5000
Lr Decay Factor
0.1
Lr Schedule
Cosine
Model Dimension
768
1024
Number of Layers
12
24
Number of Heads
12
16
Fed Forward Dim
3072
4048
Attn Dropout
0.1
Resi Dropout
0.1
Embed Dropout
0.1
Rotary Pos Embed
True
Rotary Emb Fraction
0.5
Softmax Scale
True
Use Bias
True
Flash Attn
True
Initializer
0.02 Uniform
Dataset Name
Openwebtext
Max Sample Len
1024
Batch Size
32
24
Val Ratio
0.0005

30


---Page Break---
I
Runtime Analysis on LLM training

To analyze the runtime in more detail, we measured the runtime per step of different regularization
techniques on different GPT2 model sizes (see Table I.1). For AdamW we use the PyTorch 2.1 default
implementation, for AdamCPR we adapt the AdmW implementation of PyTorch with the imple-
mentation described in Algorithm 1, for AWD and AdaDecay exists no open source implementation
and we implemented it based on the PyTorch Adam class but without "for_each" optimization, and
for AMOS we used the implementation form the pytroch-optimizer package [47]. We compare the
runtime on a node with 4 A100 GPUs and report the mean time per training step across two random
seeds and 3000 steps per experiment. In Table I.2 we compare the runtime with a batch size of 1 and
in Table I.3 we repost the runtime with the maximal possible batch size on a 40GB A100 (in samples
steps of 4).

Table I.1: GPT-2 Model Sizes and Parameter Counts
Model
Parameters
Model Dimension
Layers
Heads

GPT2s
124M
768
12
12
GPT2m
354M
1024
24
16
GPT2l
773M
1280
36
20
GPT2xl
1.19B
1600
36
25

Table I.2: Comparison of optimizer and regularizer runtime per step (batch size=1) across different
GPT2 model sizes. Percentages indicate the increase in runtime compared to AdamW. The time is
calculated as the mean time per training step across two random seeds and 3000 steps per experiment.

Method
GPT2s
GPT2m
GPT2l
GPT2xl

AdamW
0.069s
0.152s
0.273s
0.341s
AdamCPR
0.073s (+5.76%)
0.162s (+6.45%)
0.289s (+6.09%)
0.36s (+5.83%)
Adam AdaDecay
0.111s (+60.94%)
0.231s (+51.72%)
0.421s (+54.51%)
0.531s (+55.91%)
Adam AWD
0.089s (+30.04%)
0.18s (+18.55%)
0.318s (+16.64%)
0.385s (+13.05%)
AMOS
0.146s (+113.25%)
0.295s (+93.95%)
0.471s (+72.61%)
0.537s (+57.68%)

Table I.3: Comparison of optimizer runtime per step at maximum batch size across different GPT2
model sizes. Percentages indicate the increase in runtime compared to AdamW. The time is calculated
as the mean time per training step across two random seeds and 3000 steps per experiment.

Method
GPT2s
GPT2m
GPT2l
GPT2xl

AdamW
0.25s
0.493s
0.473s
0.382s
AdamCPR
0.249s (-0.40%)
0.505s (+2.44%)
0.49s (+3.59%)
0.404s (+5.76%)
Adam AdaDecay
0.309s (+23.60%)
0.577s (+17.05%)
0.617s (+30.44%)
0.573s (+50.00%)
Adam AWD
0.269s (+7.60%)
0.528s (+7.10%)
0.517s (+9.30%)
0.431s (+12.83%)
AMOS
0.302s (+20.80%)
0.614s (+24.54%)
0.703s (+48.62%)
0.581s (+52.09%)

The runtime comparison across various GPT2 models shows that AdamCPR closely matches
AdamW’s efficiency, particularly at larger batch sizes where its runtime increase becomes min-
imal or even slightly better. In contrast, Adam AdaDecay, AWD, and AMOS significantly increase
runtime, particularly in larger models and batch sizes.

However, since not all operations for CPR are implemented in a "for_each" optimized manner, CPR’s
runtime could benefit from an additional CUDA-optimized implementation.

31


---Page Break---
J
Experiments on Medical Image Segmentation

To demonstrate the effectiveness of the proposed CPR approach where using SGD, we also evaluate
it in the context of medical image segmentation. We test CPR on four segmentation benchmarks.
First, with the Adam optimizer on the Multi-Atlas Labeling Beyond the Cranial Vault (BTCV) [41]
task, the Heart Segmentation task of the Medical Segmentation Decathlon [42] and the 2020 version
of the Brain Tumor Segmentation challenge (BraTS) task [43].

Here, we make use of the data pipeline and network architectures following the nnU-Net frame-
work [40], which is regarded as the state-of-the-art framework for medical image segmentation. We
implement a training schedule with a total of 25k steps (for the Heart and BraTS tasks) and 125k
steps for BTCV. We introduce a learning rate warmup of 2k steps (8%), followed by a polynomial
annealing, see all hyperparameters in Appendix J. We run each experiment on one consumer GPU
for up to 2 days. We present the results in Table J.1, where different weight decay configurations in
AdamW are evaluated to AdamCPR with Kappa-WS initialization. We report the commonly used
Dice scores, averaged across cross-validation folds. These results indicate that CPR surpasses even
the best AdamW results. We note that applying Kappa-WS initialization too late can cause instabilities
due to weak regularization.

Since nnU-Net by default uses the SGD optimizer [48], we also test CPR to constrain optimization
with the SGD optimizer in this context. As a more recent benchmark of segmentation performance,
we report experiments on the Multi-Modality Abdominal Multi-Organ Segmentation Challenge 2022
[49]. This benchmark represents a very competitive segmentation challenge environment where
differences as small as 0.1 in Dice score can decide on challenge winners. As the experiments in
Table J.1 suggest that on average 1k warm start steps, after the learning rate warmup leads to the best
results, we resort to using 1k warm start steps for CPR since no learning rate warmup is present in the
case of SGD in nnU-Net. As the weight decay value, we employ nnU-Net’s default value of 3e-5. We
show a strong performance out of the box in this context, improving on the very competitive nnU-Net
baseline (89.45 Dice score) by a margin of 0.13 Dice points to a Dice score of 89.59. We note that
hyperparameter tuning would most likely yield further performance improvements in this regard.

Table J.1: Results of medical image segmentation training on the BTCV, Heart, and BraTS datasets.
We show the mean Dice score across 5 folds (3 for BTCV) for a range of weight decay values (γ) for
AdamW and different warm start steps s for CPR. The learning rate warmup is 2k.

SGD
SGD+CPR
1e-5
1e-4
1e-3
1e-2
1e-1
1k
2k
3k
4k

BTCV
83.04
83.1
83.17
83.99
73.92
81.17
84.14
84.23
55.41

Heart
92.92
92.75
92.88
92.9
92.85
92.77
93.18
93.16
74.44

BraTS
75.85
76.01
76.22
76.12
75.42
75.29
76.46
76.65
75.63

32


---Page Break---
Table J.2: Hyperparameters of the medical image segmentation experiments.
Parameter
Value

Fold
0,1,2,3,4
Dataset
BTCV, Heart, BraTS
Preprocessing
Default nnU-Net preprocessing [40]
Batch size
2 (following [40]
Patch size
(48x192x192) BTCV, (80x192x160) Heart, (128x128x128) BraTS
Training Steps
125k (BTCV), 25k (Heart &BraTS)
Model
3d fullres U-Net (following [40])
Optimizer
AdamW / AdamCPR
Learning Rate
0.01
Beta1
0.9
Beta2
0.99
Weight Decay
1e −5 ...1e −1 (AdamW)
Lr Schedule
Polynomial decay with warmup
Lr Warmup Steps
2000
Lr Polynomial exponent
0.9
CPR-µ
1.0
CPR-κ
1.0
CPR-k
False
CPR-κ warm-start steps
1000 ...4000
Adaptive Bounds
False

33


---Page Break---
K
Experiments on Fine-tuning a Large Language Model

Table K.1: Hyperparameters for the fine-tuning an LLM experiment.

Parameter
Value

Model Name
mistralai/Mistral-7B-Instruct-v0.2
Replace Layer
q_proj, v_proj, k_proj, o_proj,
gate_proj, up_proj, down_proj
Learning Rate
0.0005
Warmup Steps
50
Pubmedqa Artificial Samples
100000
Epochs
1
Batch Size
128
Lora R
128
Lora Alpha
1.0

34


---Page Break---
1e-4.5

1e-4

1e-3.5

PubMedQA

Learning Rate

3.8
±0.46

3.8
±0.34

3.9
±0.13

3.7
±0.57

3.8
±1.07

3.4
±0.78

3.1
±0.66

2.6
±0.92

3.3
±0.70

AdamW

4.0
±0.57

4.2
±0.22

4.2
±0.55

4.0
±1.21

4.0
±0.59

3.8
±0.45

3.2
±0.59

3.4
±0.37

3.1
±1.15

AdamCPR (Kappa-WS)

1e-4.5

1e-4

1e-3.5

Arithmetic

Learning Rate

-1.1
±0.76

-0.9
±0.18

-0.3
±0.41

-0.8
±0.34

-1.2
±0.92

-0.6
±0.71

-0.6
±0.80

-1.3
±2.11

-0.3
±1.13

-0.9
±1.13

-1.0
±0.10

-1.3
±0.30

-0.3
±0.68

-0.4
±0.91

-1.1
±0.99

1.0
±0.15

1.0
±0.38

0.2
±0.64

1e-4.5

1e-4

1e-3.5

MMLU

Learning Rate

0.9
±0.10

0.8
±0.40

1.0
±0.58

-0.1
±0.41

0.1
±0.55

-0.3
±0.60

-1.9
±1.25

-2.6
±0.41

-4.0
±2.09

1.1
±0.48

0.9
±0.24

0.8
±0.43

0.1
±0.11

0.5
±0.55

-0.2
±0.39

1.5
±0.79

1.2
±1.13

-0.8
±1.07

1e-4.5

1e-4

1e-3.5

PiQA

Learning Rate

4.3
±0.18

4.2
±0.11

4.1
±0.22

3.7
±0.22

3.8
±0.44

3.8
±0.27

3.1
±0.26

3.0
±0.49

3.2
±0.73

4.2
±0.08

4.2
±0.25

4.3
±0.29

3.8
±0.31

3.8
±0.12

3.8
±0.26

3.9
±0.21

4.2
±0.25

3.6
±0.42

0.001
0.01
0.1
Weight Decay

1e-4.5

1e-4

1e-3.5

TruthfulQA

Learning Rate

-13.1
±0.67

-13.1
±0.77

-13.6
±0.51

-19.5
±0.91

-19.4
±0.28

-18.9
±1.19

-22.1
±1.82

-23.6
±1.53

-22.8
±1.20

50 (1x)
100 (2x)
200 (4x)
Warm start steps (x lr warmup)

-13.1
±0.57

-13.2
±0.22

-13.0
±0.56

-19.7
±1.28

-19.3
±0.73

-19.5
±0.38

-10.9
±0.65

-14.0
±0.67

-20.8
±1.99

−25

−20

−15

−10

−5

0

% Accuracy Improvment

Figure K.1: Percentage of performance change before and after fineuning Mistral 7B with pubmedQA
artificial data with the use of AdamW (left) and AdamCPR (right). AdamCPR uses the L2 norm
as a regularization function and Kappa-WS. We use a learning rate warm-up of 50 steps. The
heatmap shows the mean performance and standard deviation across three random seeds. We use
the Arithmetic dataset with 10 tests that involve simple arithmetic problems in natural language
[50], the comprehensive MMLU benchmark [51], the PiQA benchmark on reasoning about physical
commonsense in natural language [52], and the TruthfulQA benchmark, which evaluates models’
abilities to mimic human falsehoods [39].

35


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]
Justification: The main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope.

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
Justification: The paper discusses limitations in the experiments in Section 5 and in the
discussion in Section 6. The runtime overhead is discussed in detail in Appendix I.

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

36


---Page Break---
Answer: [NA]
Justification: The paper does not include theoretical results that require formal proofs.
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
Justification: The paper provides comprehensive details about the experimental setup,
including all hyperparameters and datasets. We provide training code for experiments and
the implementation of our method in the supplemental materials.
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

37


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We provide training code for experiments and the implementation of our
method in the supplemental materials. All used libraries and datasets are publicly available.

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

Justification: The paper specifies all necessary training and evaluation details, including
hyperparameters and data splits.

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

Justification: The authors run multiple seeds for all experiments in the main paper and report
the mean and standard deviation of the corresponding metrics.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

38


---Page Break---
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
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
Answer: [Yes]
Justification: The paper reports information about the compute resources required for each
experiment.
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
Justification: The research conducted in the paper conforms to the NeurIPS Code of Ethics.
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

Justification: Our paper covers only foundational research and develops a generic algorithm
for optimizing neural networks.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.

39


---Page Break---
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

Answer:[NA]

Justification: The paper does not involve the release of models or data with a high risk for
misuse, and thus, safeguards are not applicable in this context.

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

Justification: The creators and original owners of assets used in the paper are properly
credited and referenced.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

40


---Page Break---
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

Answer: [Yes]

Justification: The only assets the paper releases are code and is well documented in the
supplemental material to ensure transparency and reproducibility.

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

Justification: The paper does not involve crowdsourcing experiments or research with human
subjects.

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

Justification: The paper does not involve crowdsourcing or research with human subjects, so
IRB approvals are not applicable.

Guidelines:

41


---Page Break---
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

42


---Page Break---
