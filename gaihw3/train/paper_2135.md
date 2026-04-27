# Transductive Active Learning: Theory and Applications

# Jonas Hübotter<sup>∗</sup>

Department of Computer Science ETH Zürich, Switzerland

# Lenart Treven

Department of Computer Science ETH Zürich, Switzerland

# Bhavya Sukhija

Department of Computer Science ETH Zürich, Switzerland

#### Yarden As

Department of Computer Science ETH Zürich, Switzerland

#### Andreas Krause

Department of Computer Science ETH Zürich, Switzerland

# Abstract

We study a generalization of classical active learning to real-world settings with concrete prediction targets where sampling is restricted to an accessible region of the domain, while prediction targets may lie outside this region. We analyze a family of decision rules that sample adaptively to minimize uncertainty about prediction targets. We are the first to show, under general regularity assumptions, that such decision rules converge uniformly to the smallest possible uncertainty obtainable from the accessible data. We demonstrate their strong sample efficiency in two key applications: active fine-tuning of large neural networks and safe Bayesian optimization, where they achieve state-of-the-art performance.

# 1 Introduction

Machine learning, at its core, is about designing systems that can extract knowledge or patterns from data. One part of this challenge is determining not just how to learn given observed data but deciding what data to obtain next, given the information already available. More formally, given an unknown and sufficiently regular function f over a domain X : *How can we learn* f *sample-efficiently from (noisy) observations?* This problem is widely studied in *active learning* and *experimental design* [\(Chaloner & Verdinelli,](#page-11-0) [1995;](#page-11-0) [Settles,](#page-14-0) [2009\)](#page-14-0).

Active learning methods commonly aim to learn f globally, i.e., across the entire domain X . However, in many real-world problems, (i) the domain is so large that learning f globally is hopeless or (ii) agents have limited information and cannot access the entire domain (e.g., due to restricted access or to act safely). Thus, global learning is often not desirable or even possible. Instead, intelligent systems are typically required to act in a more *directed* manner and *extrapolate* beyond their limited information. This work formalizes the above two aspects of active learning, which have remained largely unaddressed by prior work. We provide a comprehensive overview of related work in Section [6.](#page-8-0)

"Directed" transductive active learning We consider the generalized problem of *transductive active learning*, where given two arbitrary subsets of the domain X ; a *target space* A ⊆ X , and a *sample space* S ⊆ X , we study the question:

*How can we learn* f *within* A *by actively sampling observations within* S*?*

<sup>∗</sup>Correspondence to jonas.huebotter@inf.ethz.ch

This problem is ubiquitous in real-world applications such as safe Bayesian optimization, where  $\mathcal S$  is a set of safe parameters and  $\mathcal A$  might represent parameters outside  $\mathcal S$  whose safety we want to infer. Active fine-tuning of neural networks is another example, where the target space  $\mathcal A$  represents the test set over which we want to minimize risk, and the sample space  $\mathcal S$  represents the dataset from which we can retrieve data points to fine-tune our model to  $\mathcal A$ . Figure 1 visualizes some instances of transductive active learning.

Whereas most prior work has focused on the "global" inductive instance  $\mathcal{X}=\mathcal{A}=\mathcal{S}$ , MacKay (1992) was the first to consider specific target spaces  $\mathcal{A}$  and proposed the principle of selecting points in  $\mathcal{S}$  to minimize the "posterior uncertainty" about points in  $\mathcal{A}$ . Since then, several works have studied this principle empirically (e.g., Seo et al., 2000; Yu et al., 2006; Bogunovic et al., 2016; Wang et al., 2021; Kothawade et al., 2021; Bickford Smith et al., 2023). In this work, we model f as a Gaussian process or (equivalently) as a function in a reproducing kernel Hilbert space, for which the above principle is analytically and computationally tractable. Our contributions are:

<span id="page-1-0"></span>Figure 1: Instances of transductive active learning with target space  $\mathcal{A}$  shown in blue and sample space  $\mathcal{S}$  shown in gray. The points denote plausible observations within  $\mathcal{S}$  to "learn"  $\mathcal{A}$ . In (A), the target space contains "everything" within  $\mathcal{S}$  as well as points *outside*  $\mathcal{S}$ . In (B, C, D), one makes observations *directed* towards learning about a particular target. Prior work on inductive active learning has focused on the instance  $\mathcal{A} = \mathcal{S}$ .

- Theory (Section 3): We are the first to give rates for the uniform convergence of uncertainty over the target space  $\mathcal{A}$  to the smallest attainable value, given samples from the sample space  $\mathcal{S}$  (Theorems 3.2 and 3.3), Our results provide a theoretical justification for the principle of minimizing posterior uncertainty in transductive active learning, and indicate that transductive active learning can be more sample efficient than inductive active learning.
- **Applications:** We show that transductive active learning improves upon the state-of-the-art in the batch-wise *active fine-tuning* of neural networks for image classification (Section 4) and in *safe Bayesian optimization* (Section 5).

# 2 Problem Setting

We assume for now that the target space  $\mathcal{A}$  and sample space  $\mathcal{S}$  are finite, and relax these assumptions in the appendices. We model f as a stochastic process and denote the marginal random variables f(x) by  $f_x$ , and joint random vectors  $\{f_x\}_{x\in X}$  for some  $X\subseteq \mathcal{X}, |X|<\infty$  by  $f_X$ . Let  $g_X$  denote the noisy observations of  $f_X$ ,  $\{g_x=f_x+\varepsilon_x\}_{x\in X}$ , where  $\varepsilon_x$  is independent noise. We study the "adaptive" setting, where in round  $f_X$  the agent selects a point  $f_X$  and observes  $f_X$  and observes  $f_X$ . The agent's choice of  $f_X$  may depend on the outcome of prior observations  $f_X$  and  $f_X$  is a sum of the agent selects a point  $f_X$  and  $f_X$  is a sum of the agent selects a point  $f_X$  and  $f_X$  is a sum of the agent selects a point  $f_X$  and  $f_X$  is a sum of the agent selects a point  $f_X$  and  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of the agent selects a point  $f_X$  is a sum of  $f_X$  and  $f_X$  is a sum of  $f_X$ .

**Background on information theory** We briefly recap several important concepts from information theory of which we provide formal definitions in Appendix B. The (differential) entropy H[f] is one possible measure of uncertainty about f and the conditional entropy  $H[f \mid y]$  is the (expected) posterior uncertainty about f after observing g. The information gain  $I(f;g) = H[f] - H[f \mid g]$  measures the (expected) reduction in uncertainty about f due to g. We denote the information gain about f from observing f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by f by

$$\gamma_{\mathcal{A},\mathcal{S}}(n) \stackrel{\text{def}}{=} \max_{\substack{X \subseteq \mathcal{S} \\ |X| \le n}} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}}; \boldsymbol{y}_X).$$

This "information capacity" measures the information about  $f_{\mathcal{A}}$  that is accessible from within  $\mathcal{S}$ , and has been used previously (e.g., by Srinivas et al., 2009; Chowdhury & Gopalan, 2017; Vakili et al., 2021) in the setting where  $\mathcal{X} = \mathcal{A} = \mathcal{S}$ , taking the form of  $\gamma_n \stackrel{\text{def}}{=} \gamma_{\mathcal{X}}(n) \stackrel{\text{def}}{=} \gamma_{\mathcal{X},\mathcal{X}}(n)$ . We remark that  $\gamma_{\mathcal{A},\mathcal{S}}(n) \leq \gamma_{\mathcal{S}}(n)$  holds uniformly for all  $\mathcal{A},\mathcal{S}$ , and n due to the data processing inequality. Generally,  $\gamma_{\mathcal{A},\mathcal{S}}(n)$  can be substantially smaller if the target space is a sparse subset of the sample space.

<span id="page-1-1"></span> $<sup>^2</sup>X$  may be a multiset in which case repeated occurrence of x corresponds to independent observations of  $y_x$ .

# <span id="page-2-0"></span>3 Main Results

We analyze the following principle for transductive active learning:

<span id="page-2-3"></span>Select samples to minimize the posterior "uncertainty" about 
$$f$$
 within  $A$ .  $(\dagger)$ 

This principle yields a family of simple and natural decision rules which depend on the chosen measure of "uncertainty". Two natural measures of uncertainty are (1) the entropy of prediction targets,  $H[f_A]$ , and (2) their total variance,  $\sum_{x' \in A} Var[f_{x'}]$ . The corresponding decision rules are

(1) 
$$x_n = \underset{x \in \mathcal{S}}{\operatorname{arg\,min}} \operatorname{H}[\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_x] = \underset{x \in \mathcal{S}}{\operatorname{arg\,max}} \operatorname{I}(\mathbf{f}_{\mathcal{A}}; y_x \mid \mathcal{D}_{n-1}),$$
 (ITL)

(1) 
$$x_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg \, min}} \operatorname{H}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}] = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg \, max}} \operatorname{I}(\boldsymbol{f}_{\mathcal{A}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}),$$
(ITL)
(2) 
$$x_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg \, min}} \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]$$
(VTL)

with an implicit expectation over the feedback  $y_x$ . That is, ITL (short for *Information-based* Transductive Learning) and VTL (Variance-based TL) select  $x_n$  so as to minimize the uncertainty about the prediction targets  $f_A$  (in expectation) after having received the feedback  $y_n$ . Unlike VTL, ITL takes into account the mutual dependence between points in A. These decision rules were suggested previously (MacKay, 1992; Seo et al., 2000; Yu et al., 2006) without deriving theoretical guarantees; and they generalize several widely used algorithms which we discuss in more detail in Section 6. Most prominently, in the inductive setting where  $S \subseteq A$ , ITL reduces to  $x_n = \arg\max_{x \in S} I(f_x; y_x \mid \mathcal{D}_{n-1})$ , i.e., is "undirected" and reduces to standard uncertainty-based active learning strategies (cf. Appendix C.1). The convergence properties for the special instance of ITL with S = A have been studied extensively. To the best of our knowledge, we are the first to extend these guarantees to the more general setting of transductive active learning.

In our presented results, we make the following assumption.

<span id="page-2-2"></span>**Assumption 3.1.** In the case of ITL, the information gain  $\psi_{\mathcal{A}}(X) = \mathrm{I}(\mathbf{f}_{\mathcal{A}}; \mathbf{y}_X)$  is submodular. In the case of VTL, the variance reduction  $\psi_{\mathcal{A}}(X) = \mathrm{tr} \ \mathrm{Var}[\mathbf{f}_{\mathcal{A}}] - \mathrm{tr} \ \mathrm{Var}[\mathbf{f}_{\mathcal{A}}| \ \mathbf{y}_X]$  is submodular.

Under this assumption,  $\psi_{\mathcal{A}}(\mathbf{x}_{1:n})$  is a constant factor approximation of  $\max_{X\subseteq\mathcal{S},|X|\leq n}\psi_{\mathcal{A}}(X)$  due to the seminal result on submodular function maximization by Nemhauser et al. (1978). Similar assumptions have been made, e.g., by Bogunovic et al. (2016) and Kothawade et al. (2021). Assumption 3.1 is satisfied exactly for ITL when  $S \subseteq A$  and f is a Gaussian process (cf. Lemma C.9), and we provide an extensive discussion of our results in Appendix C.4 for instances where Assumption 3.1 is satisfied approximately, relying on the notion of weak submodularity (Das & Kempe, 2018).

#### <span id="page-2-4"></span>3.1 Gaussian Process Setting

When  $f \sim \mathcal{GP}(\mu, k)$  is a Gaussian process (GP, Williams & Rasmussen, 2006) with known mean function  $\mu$  and kernel k, and the noise  $\varepsilon_x$  is mutually independent and zero-mean Gaussian with known variance, the ITL and VTL objectives have a closed form expression (cf. Appendix F) and can be optimized efficiently. Further, the information capacity  $\gamma_n$  is sublinear in n for a rich class of GPs (Srinivas et al., 2009; Vakili et al., 2021), with rates summarized in Table 3 of the appendix.

**Convergence to irreducible uncertainty** So far, our discussion was centered around the role of the target space A in facilitating *directed* learning. An orthogonal contribution of this work is to study extrapolation from the sample space S to points  $x \in A \setminus S$ . To this end, we derive bounds on the marginal posterior variance  $\sigma_n^2(x) \stackrel{\text{def}}{=} \operatorname{Var}[f(x) \mid \mathcal{D}_n]$  for points in A. These bounds depend on the instance of transductive active learning (i.e., A and S) and might be of independent interest for active learning. For ITL and VTL, they imply uniform convergence of the variance for a rich class of GPs. To the best of our knowledge, this work is the first to present such bounds.

We define the *irreducible uncertainty* as the variance of f(x) provided complete knowledge of f in S:

$$\eta_{\mathcal{S}}^2(\boldsymbol{x}) \stackrel{\text{def}}{=} \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{\mathcal{S}}].$$

As the name suggests,  $\eta_S^2(x)$  represents the smallest uncertainty one can hope to achieve from observing only within S. For all  $x \in S$ , it is easy to see that  $\eta_S^2(x) = 0$ . However, the irreducible uncertainty of  $x \notin S$  may be (and typically is!) strictly positive.

<span id="page-2-1"></span>Theorem 3.2 (Bound on marginal variance for ITL and VTL). Let Assumption 3.1 hold and the data be selected by either ITL or VTL.. Assume that  $f \sim \mathcal{GP}(\mu, k)$  with known mean function  $\mu$  and kernel k, the noise  $\varepsilon_x$  is mutually independent and zero-mean Gaussian with known variance, and  $\gamma_n$  is sublinear in n. Then there exists a constant C such that for any  $n \ge 1$  and  $x \in A$ ,

$$\sigma_n^2(x) \le \underbrace{\eta_{\mathcal{S}}^2(x)}_{irreducible} + \underbrace{C\frac{\gamma_{\mathcal{A},\mathcal{S}}(n)}{\sqrt{n}}}_{reducible}.$$
 (1)

Moreover, if  $x \in A \cap S$ , there exists a constant C' such that

<span id="page-3-1"></span>
$$\sigma_n^2(x) \le C' \frac{\gamma_{\mathcal{A},\mathcal{S}}(n)}{n}.$$
 (2)

Intuitively, Equation (1) of Theorem 3.2 can be understood as bounding an epistemic "generalization gap" (Wainwright, 2019) of the learner. The reducible uncertainty converges to zero at all prediction targets  $x \in \mathcal{A}$ , e.g., for linear, Gaussian, and smooth Matérn kernels. As to be expected, a smaller target space (i.e., more targeted sampling) leads to faster convergence due to a smaller information capacity  $\gamma_{\mathcal{A},\mathcal{S}}(n) \ll \gamma_n$ . Equation (2) matches prior results for the setting  $\mathcal{S} = \mathcal{A}$ . We provide a formal proof of Theorem 3.2 in Appendix C.6.

### <span id="page-3-3"></span>3.2 Agnostic Setting

The result from the GP setting translates also to the agnostic setting, where the "ground truth"  $f^*$  may be any sufficiently regular fixed function on  $\mathcal{X}$ . In this case, we use the model f from Section 3.1 as a (misspecified) model of  $f^*$ , with some kernel k and zero mean function  $\mu(\cdot) = 0$ . We denote by  $\mu_n(\boldsymbol{x}) = \mathbb{E}[f(\boldsymbol{x}) \mid \mathcal{D}_n]$  the posterior mean of f. W.l.o.g. we assume in the following result that the prior variance is bounded, i.e.,  $\operatorname{Var}[f(\boldsymbol{x})] \leq 1$ .

<span id="page-3-0"></span>**Theorem 3.3** (Bound on approximation error for ITL and VTL, following Abbasi-Yadkori (2013); Chowdhury & Gopalan (2017)). Let Assumption 3.1 hold and the data be selected by either ITL or VTL. Pick any  $\delta \in (0,1)$ . Assume that  $f^*$  lies in the reproducing kernel Hilbert space  $\mathcal{H}_k(\mathcal{X})$  of the kernel k with norm  $\|f^*\|_k < \infty$ , the noise  $\varepsilon_n$  is conditionally  $\rho$ -sub-Gaussian, and  $\gamma_n$  is sublinear in n. Let  $\beta_n(\delta) = \|f^*\|_k + \rho \sqrt{2(\gamma_n + 1 + \log(1/\delta))}$ . Then for any  $n \geq 1$  and  $x \in \mathcal{A}$ , jointly with probability at least  $1 - \delta$ ,

$$|f^{\star}(x) - \mu_n(x)| \leq \beta_n(\delta) \left[\underbrace{\eta_{\mathcal{S}}(x)}_{irreducible} + \underbrace{\nu_{\mathcal{A},\mathcal{S}}(n)}_{reducible}\right]$$

where  $\nu_{AS}^2(n)$  denotes the reducible part of Equation (1).

We provide a formal proof of Theorem 3.3 in Appendix C.7. Theorem 3.3 generalizes approximation error bounds of prior works to the extrapolation setting, where some prediction targets  $x \in A$  lie outside the sample space S. For prediction targets  $x \in A \cap S$ , the irreducible uncertainty vanishes, and we recover previous results from the setting S = A.

Theorems 3.2 and 3.3 show that ITL and VTL efficiently learn f at the prediction targets  $\mathcal{A}$  for large classes of "sufficiently regular" functions f. In the following, we validate these results experimentally by showing that ITL and VTL exhibit strong empirical performance in a broad range of applications.

#### <span id="page-3-4"></span>3.3 Experiments in the Gaussian Process Setting

Before demonstrating ITL and VTL on GPs to develop more intuition, we introduce a natural correlation-based baseline, which will later uncover connections to existing approaches:

$$x_n = \underset{x \in \mathcal{S}}{\operatorname{arg \, max}} \sum_{x' \in \mathcal{A}} \operatorname{Cor}[f_x, f_{x'} \mid \mathcal{D}_{n-1}].$$
 (CTL)

How does the smoothness of f affect ITL? We contrast two "extreme" kernels: the Gaussian kernel  $k(\boldsymbol{x}, \boldsymbol{x'}) = \exp(-\|\boldsymbol{x} - \boldsymbol{x'}\|_2^2/2)$  and the Laplace kernel  $k(\boldsymbol{x}, \boldsymbol{x'}) = \exp(-\|\boldsymbol{x} - \boldsymbol{x'}\|_1)$ . In the mean-squared sense, the Gaussian kernel yields a smooth process f whereas the Laplace kernel yields a continuous but non-differentiable f (Williams & Rasmussen, 2006). Figure 2 shows

<span id="page-3-2"></span><sup>&</sup>lt;sup>3</sup>Here  $f^{\star}(x)$  denotes the mean observation  $y_x = f^{\star}(x) + \epsilon_x$ 

Figure 2: Initial 25 samples of ITL under a Gaussian kernel with lengthscale 1 (left) and a Laplace kernel with lengthscale 10 (right). Shown in gray is the sample space S and shown in blue is the target space A. In three of the four examples, points outside the target space provide useful information.

<span id="page-4-1"></span><span id="page-4-2"></span>Figure 3: Entropy of  $f_{\mathcal{A}}$  ranging from -3850 to -3725 and the mean marginal standard deviations of  $f_{\mathcal{A}}$  ranging from 0 to 0.15. Experiment is using the Gaussian kernel of the left instance ( $\mathcal{A} \subset \mathcal{S}$ ) from Figure 2. It can be seen that ITL and VTL outperform UNSA and RANDOM. Uncertainty bands correspond to one standard error over 10 random seeds.

how ITL adapts to the smoothness of f: Under the "smooth" Gaussian kernel, points outside  $\mathcal{A}$  provide higher-order information. In contrast, under the "rough" Laplace kernel and if  $\mathcal{A} \subseteq \mathcal{S}$ , points outside  $\mathcal{A}$  do not provide any additional information, and therefore are not sampled by ITL. If, however,  $\mathcal{A} \not\subseteq \mathcal{S}$ , information "leaks"  $\mathcal{A}$  even under a Laplace kernel prior. That is, even for non-smooth functions, the point with most information need not be in  $\mathcal{A}$ .

**Does ITL outperform uncertainty sampling?** Uncertainty sampling (UNSA, Lewis & Catlett, 1994) is one of the most popular active learning methods. UNSA selects points  $\boldsymbol{x}$  with high *prior* uncertainty:  $\boldsymbol{x}_n = \arg\max_{\boldsymbol{x} \in \mathcal{S}} \sigma_{n-1}^2(\boldsymbol{x})$ . This is in stark contrast to ITL and VTL which select points  $\boldsymbol{x}$  that minimize *posterior* (epistemic) uncertainty about  $\mathcal{A}$ . It can be seen that UNSA is the special "undirected" case of ITL when  $\mathcal{S} \subseteq \mathcal{A}$  and observation noise is homoscedastic (cf. Appendix C.1).

We compare UNSA to ITL, VTL, and CTL in Figure 3. We observe that ITL and VTL outperform UNSA which also samples points that are not informative about  $\mathcal{A}$ . Further, ITL and VTL outperform "local" UNSA (i.e., UNSA constrained to  $\mathcal{A} \cap \mathcal{S}$ ) which neglects all information provided by points outside  $\mathcal{A}$ .<sup>4</sup> As one would expect, VTL has an advantage with respect to reducing the total variance of  $f_{\mathcal{A}}$ , whereas ITL reduces the entropy of  $f_{\mathcal{A}}$  faster. We include ablations in Appendix H where we, in particular, observe that the advantage of ITL and VTL over UNSA increases as the volume of prediction targets shrinks in comparison to the size of domain.

# <span id="page-4-0"></span>4 Active Fine-Tuning of Neural Networks

Fine-tuning a large pre-trained model is a cost- and computation-effective approach to improve performance on a given target domain (Lee et al., 2022). While previous work has studied the effectiveness of various training procedures for fine-tuning (e.g., Eustratiadis et al., 2024), we ask: How can we select the right data for fine-tuning to a specific task? This active fine-tuning problem is an instance of the introduced "directed" transductive learning problem: Concretely, consider a supervised setting, where the function f maps inputs  $x \in \mathcal{X}$  to outputs  $y \in \mathcal{Y}$ . We have access

<span id="page-4-3"></span><sup>&</sup>lt;sup>4</sup>If  $\mathcal{A} \not\subseteq \mathcal{S}$  then "local" UNSA does *not even* converge to the irreducible uncertainty.

to noisy samples from a training set S on X, and we would like to learn f such that our estimate minimizes a given risk measure, such as classification error, with respect to a test distribution  $\mathcal{P}_A$  on X. The goal is to actively and efficiently sample from S to minimize risk with respect to  $P_A$ . We show in this section that ITL and VTL can learn f from only few examples from S.

How can we leverage the latent structure learned by the pre-trained model? As common in related work, we approximate the (pre-trained) neural network (NN)  $f(\cdot;\theta)$  as a linear function in a latent embedding space,  $f(x;\theta) \approx \beta^\top \phi_\theta(x)$ , with weights  $\beta \in \mathbb{R}^p$  and embeddings  $\phi_\theta: \mathcal{X} \to \mathbb{R}^p$ . Common choices of embeddings include last-layer embeddings (Devlin et al., 2019; Holzmüller et al., 2023), neural tangent embeddings arising from neural tangent kernels (Jacot et al., 2018) which are motivated by their relationship to the training and fine-tuning of ultra-wide NNs (Arora et al., 2019; Lee et al., 2019; Khan et al., 2019; He et al., 2020; Malladi et al., 2023), and loss gradient embeddings (Ash et al., 2020). We provide a comprehensive overview of embeddings in Appendix J.2. Now, supposing the prior  $\beta \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$ , often with  $\mathbf{\Sigma} = \mathbf{I}$  (Khan et al., 2019; He et al., 2020; Antorán et al., 2022; Wei et al., 2022), this approximation of f is a Gaussian process with kernel  $f(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) = f(\mathbf{x})$  which quantifies the similarity between points in terms of their alignment in the learned latent space. Note that the correlation  $f(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) = f(\mathbf{x})$  between two points  $f(\mathbf{x}, \mathbf{x}')$  is equal to the cosine similarity of their embeddings.

In this context, Theorem 3.2 bounds the epistemic posterior uncertainty about a prediction using the approximation  $\beta^{\top}\phi_{\theta}(x)$ , given that the model is trained using data selected by ITL or VTL. Theorem 3.3 bounds the generalization error when using the posterior mean of  $\beta$  for prediction. This extends recent work which has studied estimators of this generalization error (Wei et al., 2022).

**Batch selection: Diversity via conditional embeddings** Efficient labeling and training necessitates a batch-wise selection of inputs. The selection of a batch of size b > 1 can be seen as an individual *non-adaptive* active learning problem, and significant recent work has shown that batch diversity is crucial in this setting (Ash et al., 2020; Zanette et al., 2021; Holzmüller et al., 2023; Pacchiano et al., 2024). An information-based batch-wise selection strategy is formalized by the following non-adaptive transductive active learning problem (Chen & Krause, 2013) and the greedy approximation of  $B_n$  by ITL which selects elements  $x_{n,i}$  of the n-th batch iteratively based on  $x_{n,1:i-1}$ :

<span id="page-5-2"></span>
$$B_{n} = \underset{B \subseteq \mathcal{S}, |B| = b}{\operatorname{arg max}} I(\mathbf{f}_{\mathcal{A}}; \mathbf{y}_{B} \mid \mathcal{D}_{n-1}); \qquad \mathbf{x}_{n,i} = \underset{\mathbf{x} \in \mathcal{S}}{\operatorname{arg max}} I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_{n-1}, \mathbf{y}_{\mathbf{x}_{n,1:i-1}}).$$
(3)

The batch  $B_n$  is diverse and informative by design. We show that under Assumption 3.1,  $B'_n = x_{n,1:b}$  yields a constant-factor approximation of  $B_n$  (cf. Appendix C.3).

#### 4.1 Experiments on Active Fine-Tuning

Our empirical evaluation is motivated by the following practical example: We deploy a pre-trained image classifier to user's phones who use it within their local environment. We would like to locally fine-tune a user's model to their environment. Since the users' images  $\mathcal A$  are unlabeled, this requires selecting a small number of relevant and diverse images from the set of labeled images  $\mathcal S$ . As such, we will focus here on the setting where the points in our test set do not lie in our training set (i.e.,  $\mathcal A \cap \mathcal S = \emptyset$ ), and discuss alternative instances such as active domain adaptation in Appendix I.

**Testbeds & architectures** We use the MNIST (LeCun et al., 1998) and CIFAR-100 (Krizhevsky et al., 2009) datasets as testbeds. In both cases, we take  $\mathcal{S}$  to be the training set, and we consider the task of learning the digits 3, 6, and 9 (MNIST) or the first 10 categories of CIFAR-100.<sup>6</sup> For MNIST, we train a simple convolutional neural network with ReLU activations, three convolutional layers with max-pooling, and two fully-connected layers. For CIFAR-100, we fine-tune an EfficientNet-B0 (Tan & Le, 2019) pre-trained on ImageNet (Deng et al., 2009), augmented by a final fully-connected layer. We train the NNs using the cross-entropy loss and the ADAM optimizer (Kingma & Ba, 2014).

**Results** In Figure 4, We compare against (i) active learning methods which largely aim for sample diversity but which are not directed towards the target distribution  $\mathcal{P}_{\mathcal{A}}$  (e.g., BADGE; Ash et al., 2020), and (ii) search methods that aim to retrieve the most relevant samples from  $\mathcal{S}$  with respect to the targets  $\mathcal{P}_{\mathcal{A}}$  (e.g., maximizing cosine similarity to target embeddings as is common in vector databases;

<span id="page-5-1"></span><span id="page-5-0"></span><sup>&</sup>lt;sup>5</sup>The setting with target distributions  $\mathcal{P}_{\mathcal{A}}$  can be reduced to considering target sets  $\mathcal{A}$  (cf. Appendix E).

<sup>&</sup>lt;sup>6</sup>That is, we restrict  $\mathcal{P}_{\mathcal{A}}$  to the support of points with labels  $\{3, 6, 9\}$  (MNIST) or labels  $\{0, \dots, 9\}$  (CIFAR-100) and train a neural network using few examples drawn from the training set  $\mathcal{S}$ .

<span id="page-6-0"></span>Figure 4: Active fine-tuning on MNIST (left) and CIFAR-100 (right). RANDOM selects each observation uniformly at random from S. The batch size is 1 for MNIST and 10 for CIFAR-100. Uncertainty bands correspond to one standard error over 10 random seeds. We see that transductive active learning with ITL and VTL significantly outperforms competing methods, and in particular, retrieves substantially more samples from the support of PA. See Appendix [J](#page-43-0) for details and ablations.

[Settles & Craven,](#page-15-6) [2008;](#page-15-6) [Johnson et al.,](#page-13-9) [2019\)](#page-13-9). INFORMATIONDENSITY (ID, [Settles & Craven,](#page-15-6) [2008\)](#page-15-6) is a heuristic approach aiming to combine (i) diversity and (ii) relevance. In Appendix [J.5,](#page-47-0) we also compare against a wide range of additional baselines (e.g., CORESET [\(Sener & Savarese,](#page-14-5) [2017\)](#page-14-5), TYPICLUST [\(Hacohen et al.,](#page-12-3) [2022\)](#page-12-3), PROBCOVER [\(Yehuda et al.,](#page-16-3) [2022\)](#page-16-3), etc.) that fall into one of the categories (i) and (ii), and which perform similar to the baselines listed here.

We observe that ITL, VTL, and CTL consistently and significantly outperform random sampling from S as well as all baselines. We see that relevance-based methods such as COSINESIMILARITY have an initial advantage over RANDOM but for batch sizes larger than 1 they quickly fall behind due to diminishing informativeness of the selected data. In contrast, diversity-based methods such as BADGE are more competitive with RANDOM but do not explicitly aim to retrieve relevant samples.

Remarkably, transductive active learning outperforms random data selection even in the MNIST experiment where the model is randomly initialized. This suggests that the learned embeddings can be informative for data selection even in the early stages of training, bootstrapping the learning progress.

Balancing sample relevance and diversity Our proposed methods unify approaches to coverage (promoting *diverse* samples) and search (aiming for *relevant* samples with respect to a given query A) which leads to the significant improvement upon the state-of-the-art in Figure [4.](#page-6-0) Notably, for a batch size and query size of 1 and if correlations are non-negative, ITL, VTL, CTL, and the canonical cosine similarity are equivalent. CTL can be seen as a direct generalization of cosine similarity-based retrieval to batch and query sizes larger than one. In contrast to CTL, ITL and VTL may also sample points which exhibit a strong negative correlation (which is also informative).

We observe empirically that ITL obtains samples from P<sup>A</sup> at more than twice the rate of COSINES-IMILARITY, which translates to a significant improvement in accuracy in more difficult learning tasks, while requiring fewer (labeled) samples from S. This phenomenon manifests for both MNIST and CIFAR-100, as well as imbalanced datasets S or imbalanced reference samples from P<sup>A</sup> (cf. Appendix [J.6\)](#page-51-0). The improvement in accuracy appears to increase in the large-data regime, where the learning tasks become more difficult. Akin to a previously identified scaling trend with size of the pretraining dataset [\(Tamkin et al.,](#page-15-7) [2022\)](#page-15-7), this suggests a potential scaling trend where the improvement of ITL over random batch selection grows as models are fine-tuned on a larger pool of data.

**Towards task-driven few-shot learning** Being able to efficiently and automatically select data may allow dynamic few-shot fine-tuning to individual tasks (Vinyals et al., 2016; Hardt & Sun, 2024), e.g., fine-tuning the model to each test point / query / prompt. Such task-driven few-shot learning can be seen as a form of "memory recall" akin to associative memory (Hopfield, 1982). Our results are a first indication that task-driven learning can lead to substantial performance gains, and we believe that this is a promising direction for future studies.

#### <span id="page-7-0"></span>5 Safe Bayesian Optimization

Another practical problem that can be cast as "directed" learning is safe Bayesian optimization (Safe BO, Sui et al., 2015; Berkenkamp et al., 2021) which has applications in natural science (Cooper & Netoff, 2022) and robotics (Wischnewski et al., 2019; Sukhija et al., 2023; Widmer et al., 2023). Safe BO solves the following optimization problem

<span id="page-7-1"></span>
$$\max_{\boldsymbol{x} \in \mathcal{S}^{\star}} f^{\star}(\boldsymbol{x}) \quad \text{where} \quad \mathcal{S}^{\star} = \{ \boldsymbol{x} \in \mathcal{X} \mid g^{\star}(\boldsymbol{x}) \ge 0 \}$$
 (4)

which can be generalized to multiple constraints. The functions  $f^*$  and  $g^*$ , and hence also the "safe set"  $\mathcal{S}^*$ , are unknown and have to be actively learned from data. However, it is crucial that the data collection does not violate the constraint, i.e.,  $x_n \in \mathcal{S}^*$ ,  $\forall n \geq 1$ .

Safe Bayesian optimization as Transductive Active Learning In the agnostic setting from Section 3.2, GPs f and g can be used as well-calibrated models of the ground truths  $f^*$  and  $g^*$ , and we denote lower- and upper-confidence bounds by  $l_n^f(\boldsymbol{x}), l_n^g(\boldsymbol{x})$  and  $u_n^f(\boldsymbol{x}), u_n^g(\boldsymbol{x})$ , respectively. These confidence bounds induce a *pessimistic* safe set  $\mathcal{S}_n \stackrel{\text{def}}{=} \{ \boldsymbol{x} \mid l_n^g(\boldsymbol{x}) \geq 0 \}$  and an *optimistic* safe set  $\widehat{\mathcal{S}}_n \stackrel{\text{def}}{=} \{ \boldsymbol{x} \mid u_n^g(\boldsymbol{x}) \geq 0 \}$  which satisfy  $\mathcal{S}_n \subseteq \mathcal{S}^* \subseteq \widehat{\mathcal{S}}_n$  with high probability at all times. Similarly, the set of *potential maximizers* 

<span id="page-7-4"></span>
$$\mathcal{A}_n \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \widehat{\mathcal{S}}_n \mid u_n^f(\boldsymbol{x}) \ge \max_{\boldsymbol{x}' \in \mathcal{S}_n} l_n^f(\boldsymbol{x}') \}$$
 (5)

contains the solution to Equation (4) at all times with high probability.

The (simple) regret  $r_n(\mathcal{S}) \stackrel{\text{def}}{=} \max_{\boldsymbol{x} \in \mathcal{S}} f^{\star}(\boldsymbol{x}) - f^{\star}(\widehat{\boldsymbol{x}}_n)$  with  $\widehat{\boldsymbol{x}}_n \stackrel{\text{def}}{=} \arg\max_{\boldsymbol{x} \in \mathcal{S}_n} l_n^f(\boldsymbol{x})$  measures the worst-case performance of a decision rule. To achieve small regret, one faces an *exploration-expansion* dilemma wherein one needs to explore points that are known-to-be-safe, i.e., lie in the estimated safe set  $\mathcal{S}_n$ , and might be optimal, while at the same time discovering new safe points by "expanding"  $\mathcal{S}_n$ . Accordingly, a natural choice for the target space of Safe BO is  $\mathcal{A}_n$  since it captures both exploration and expansion *simultaneously*. To prevent constraint violation, the sample space is restricted to the pessimistic safe set  $\mathcal{S}_n$ . In Safe BO, both the target space and sample space change with each round n, and we generalize our theoretical results from Section 3 in Appendix  $\mathbb C$  to this setting.

<span id="page-7-3"></span>**Theorem 5.1** (Convergence to safe optimum). Pick any  $\epsilon > 0$ ,  $\delta \in (0,1)$ . Assume that  $f^*$ ,  $g^*$  lie in the reproducing kernel Hilbert space  $\mathcal{H}_k(\mathcal{X})$  of the kernel k, and that the noise  $\varepsilon_n$  is conditionally  $\rho$ -sub-Gaussian. Then, we have with probability at least  $1 - \delta$ ,

Safety: for all 
$$n \geq 1$$
,  $x_n \in \mathcal{S}^*$ .

Moreover, assume  $S_0 \neq \emptyset$  and denote with R the largest reachable safe set starting from  $S_0$ . Then, the convergence of reducible uncertainty implies that there exists  $n^* > 0$  such that with probability at least  $1 - \delta$ ,

Optimality: for all 
$$n \ge n^*$$
,  $r_n(\mathcal{R}) \le \epsilon$ .

We provide a formal proof in Appendix C.8. Central to the proof is the application of Theorem 3.3 to show that the safety of parameters *outside* the safe set  $S_n$  can be inferred efficiently. In Section 3, we outline settings where the reducible uncertainty converges which is the case for a very general class of functions, and for such instances Theorem 5.1 guarantees optimality in the largest reachable safe set R. R represents the largest set any safe learning algorithm can explore without violating the safety constraints (with high probability) during learning (cf. Definition C.29). Our guarantees are similar to those of other Safe BO algorithms (Berkenkamp et al., 2021) but require fewer assumptions and generalize to continuous domains. We obtain Theorem 5.1 from a more general result (Theorem C.34) which can be specialized to yield "free" novel convergence guarantees for problems other than Bayesian optimization, such as level set estimation, by choosing an appropriate target space.

<span id="page-7-2"></span><sup>&</sup>lt;sup>7</sup>An alternative possibility is to weigh each point in  $A_n$  according to how likely it is to be the safe optimum. Which approach performs better is task-dependent, and we include a detailed discussion in Appendix K.1.

<span id="page-8-1"></span>Figure 5: We compare ITL and VTL to ORACLE SAFEOPT, which has oracle knowledge of the Lipschitz constants, SAFEOPT, where the Lipschitz constants are estimated from the GP, as well as HEURISTIC SAFEOPT and ISE, and observe that ITL and VTL systematically perform well. We compare against additional baselines in Appendix K.1. The regret is evaluated with respect to the ground truth objective  $f^*$  and constraint  $g^*$ , and averaged over 10 (in synthetic experiments) and 25 (in the quadcopter experiment) random seeds. Additional details can be found in Appendix K.4.

#### 5.1 Experiments on Safe Bayesian Optimization

We evaluate two synthetic experiments for a 1d and 2d parameter space, respectively (cf. Appendix K.4 for details), which demonstrate the various shortcomings of existing Safe BO baselines. Additionally, as third experiment, we safely tune the controller of a quadcopter.

Safe controller tuning for a quadcopter We consider a quadcopter with unknown dynamics;  $s_{t+1} = T(s_t, u_t)$  where  $u_t \in \mathbb{R}^{d_u}$  is the control signal and  $s_t \in \mathbb{R}^{d_s}$  is the state at time t. The inputs  $u_t$  are calculated through a deterministic function of the state  $\pi: \mathcal{S} \to \mathcal{U}$  which we call the policy. The policy is parameterized via parameters  $x \in \mathcal{X}$ , e.g., PID controller gains, such that  $u_t = \pi_x(s_t)$ . The goal is to find the optimal parameters with respect to an unknown objective  $f^*$  while satisfying some unknown constraint(s)  $g^*(x) \geq 0$ , e.g., the quadcopter does not fall on the ground. This is a typical Safe BO problem which is widely applied for safe controller learning in robotics (Berkenkamp et al., 2021; Baumann et al., 2021; Widmer et al., 2023).

**Results** We compare ITL and VTL to SAFEOPT (Berkenkamp et al., 2021), which is undirected, i.e., expands in all directions including ones that are known-to-be suboptimal, and ISE (Bottero et al., 2022), which is solely expansionist — does not trade-off expansion-exploration. We provide a detailed discussion of baselines in Appendix K.2. In all our experiments, summarized in Figure 5, we observe that ITL and VTL systematically perform well, i.e., better or on par with the state-of-the-art. We attribute this to its directed exploration and less conservative expansion over SAFEOPT (cf. 1d task and quadcopter experiment), and natural trade-off between expansion and exploration as opposed to ISE (see 2d task). Generally, VTL has a slight advantage over ITL, which is because VTL minimizes marginal variances (as opposed to entropy), which are decisive for expanding the safe set. While ITL and VTL do not violate constraints, we observe that other methods that do not explicitly enforce safety such as EIC (Gardner et al., 2014) lead to constraint violation (cf. Appendix K.4.2).

# <span id="page-8-0"></span>6 Related Work

(Inductive) active learning The special case of transductive active learning where A = S = X has been widely studied. We refer to this special instance as *inductive* active learning, since the goal is to extract as much information as possible as opposed to making predictions on a specific target set.

Several works have previously found entropy-based decision rules to be useful for inductive active learning (Krause & Guestrin, 2007; Guo & Greiner, 2007; Krause et al., 2008) and semi-supervised learning (Grandvalet & Bengio, 2004). The variance-based VTL has previously been proposed by Cohn (1993) in the special case of inductive active learning without proving theoretical guarantees. VTL was then recently re-derived by Shoham & Avron (2023) along other experimental design

criteria under the lens of minimizing risk for inductive one-shot learning in overparameterized models. Substantial work on active learning has studied entropy-based criteria in *parameter-space*, most notably BALD (MacKay, 1992; Houlsby et al., 2011; Gal et al., 2017; Kirsch et al., 2019), which selects  $\boldsymbol{x}_n = \arg\max_{\boldsymbol{x} \in \mathcal{X}} I(\boldsymbol{\theta}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})$ , where  $\boldsymbol{\theta}$  is the random parameter vector of a parametric model (e.g., obtained via Bayesian deep learning). Such methods are inherently inductive in the sense that they do not facilitate learning on specific prediction targets.

**Transductive active learning** In contrast, ITL operates in *output-space* where it is straightforward to specify prediction targets, and which is computationally easier. Special cases of ITL when  $S = \mathcal{X}$  and  $|\mathcal{A}| = 1$  have been proposed in the foundational work of MacKay (1992) on "directed" output-space active learning. As generalization to larger target spaces, MacKay (1992) proposed mean-marginal ITL,

$$\boldsymbol{x}_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg max}} \sum_{\boldsymbol{x'} \in \mathcal{A}} I(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}), \qquad (MM-ITL)$$

for which we derive analogous versions of Theorems 3.2 and 3.3 in Appendix D.3. We note that similarly to VTL, MM-ITL disregards the mutual dependence of points in the target space  $\mathcal{A}$  and differs from VTL only in a different weighting of the posterior marginal variances of the prediction targets (cf. Appendix D.3). Recently, Bickford Smith et al. (2023) generalized MM-ITL by treating the prediction target as a random variable, and Kothawade et al. (2021) and Bickford Smith et al. (2024) demonstrated the use of output-space decision rules for image classification tasks in a pre-training context.

Influence functions measure the change in a model's prediction when a single data point is removed from the training data (Cook, 1977; Koh & Liang, 2017; Pruthi et al., 2019). Influence functions have been used for data selection in settings closely related to the transductive active fine-tuning of neural networks proposed in this work (Xia et al., 2024). They select data that reduces a first-order Taylor approximation to the test loss after fine-tuning a neural network, which corresponds to maximizing cosine similarity to the prediction targets in a loss-gradient embedding space. We show in our experiments that transductive active learning can substantially outperform CosineSimilarity. We attribute this primarily to influence functions implicitly assuming that the influence of selected data adds linearly (i.e., two equally scored data points are expected to doubly improve the model performance, Xu & Kazantsev, 2019, Section 3.2). This assumption does not hold in practice as seen, e.g., by simply duplicating data. The same limitation applies to the related approach of datamodels (Ilyas et al., 2022).

Other work on directed active learning Directed active learning methods have been proposed for the problem of determining the optimum of an unknown function, also known as best-arm identification (Audibert et al., 2010) or pure exploration bandits (Bubeck et al., 2009). Entropy search methods (Hennig & Schuler, 2012; Hernández-Lobato et al., 2014) are widely used and select  $x_n = \arg\max_{x \in \mathcal{X}} I(x^*; y_x \mid \mathcal{D}_{n-1})$  in *input-space* where  $x^* = \arg\max_x f_x$ . Similarly to ITL, *output-space* entropy search methods (Hoffman & Ghahramani, 2015; Wang & Jegelka, 2017), which select  $x_n = \arg\max_{x \in \mathcal{X}} I(f^*; y_x \mid \mathcal{D}_{n-1})$  with  $f^* = \max_x f_x$ , are more computationally tractable. In fact, output-space entropy search is a special case of ITL with a stochastic target space (cf. Equation (47) in Appendix K.1). Bogunovic et al. (2016) analyze TRUVAR in the context of Bayesian optimization and level set estimation. TRUVAR is akin to VTL with a similar notion of "target space", but their algorithm and analysis rely on a threshold scheme which requires that  $A \subseteq S$ . Fiez et al. (2019) introduce the *transductive linear bandit* problem, which is a special case of transductive active learning limited to a linear function class and with the objective of determining the maximum within an initial candidate set. We mention additional more loosely related works in Appendix A.

# 7 Conclusion

We investigated the generalization of active learning to settings with concrete prediction targets and/or with limited information due to constrained sample spaces. This provides a flexible framework, applicable also to other domains than were discussed (such as recommender systems, molecular design, robotics, etc.) by varying the choice of target space and sample space. Further, we proved novel generalization bounds which may be of independent interest for active learning. Finally, we demonstrated across broad applications that sampling *relevant and diverse* points (as opposed to only one of the two) leads to a substantial improvement upon the state-of-the-art.

<span id="page-9-0"></span> $<sup>^8</sup>$ The transductive bandit problem can be solved analogously to Safe BO, by maintaining the set  $\mathcal{A}_n$ .

# Acknowledgements

Many thanks to Armin Lederer, Johannes Kirschner, Jonas Rothfuss, Lars Lorch, Manish Prajapat, Nicolas Emmenegger, Parnian Kassraie, and Scott Sussex for their insightful feedback on different versions of this manuscript, as well as Anton Baumann for helpful discussions. We further thank Freddie Bickford Smith for a constructive discussion regarding the relationship between our work and prior work.

This project was supported in part by the European Research Council (ERC) under the European Union's Horizon 2020 research and Innovation Program Grant agreement no. 815943, the Swiss National Science Foundation under NCCR Automation, grant agreement 51NF40 180545, and by a grant of the Hasler foundation (grant no. 21039).

# References

- <span id="page-10-1"></span>Abbasi-Yadkori, Y. *Online learning for linearly parametrized control problems*. PhD thesis, University of Alberta, 2013.
- <span id="page-10-4"></span>Antorán, J., Janz, D., Allingham, J. U., Daxberger, E., Barbano, R. R., Nalisnick, E., and Hernández-Lobato, J. M. Adapting the linearised laplace model evidence for modern deep learning. In *ICML*, 2022.
- <span id="page-10-2"></span>Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R. R., and Wang, R. On exact computation with an infinitely wide neural net. *NeurIPS*, 32, 2019.
- <span id="page-10-13"></span>Arthur, D., Vassilvitskii, S., et al. k-means++: The advantages of careful seeding. In *SODA*, volume 7, 2007.
- <span id="page-10-14"></span>Ash, J., Goel, S., Krishnamurthy, A., and Kakade, S. Gone fishing: Neural active learning with fisher embeddings. *NeurIPS*, 34, 2021.
- <span id="page-10-3"></span>Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., and Agarwal, A. Deep batch active learning by diverse, uncertain gradient lower bounds. *ICLR*, 2020.
- <span id="page-10-7"></span>Audibert, J.-Y., Bubeck, S., and Munos, R. Best arm identification in multi-armed bandits. In *COLT*, 2010.
- <span id="page-10-10"></span>Balestriero, R., Ibrahim, M., Sobal, V., Morcos, A., Shekhar, S., Goldstein, T., Bordes, F., Bardes, A., Mialon, G., Tian, Y., et al. A cookbook of self-supervised learning. *arXiv preprint arXiv:2304.12210*, 2023.
- <span id="page-10-11"></span>Barrett, A. B. Exploration of synergistic and redundant information sharing in static and dynamical gaussian systems. *Physical Review E*, 91(5), 2015.
- <span id="page-10-6"></span>Baumann, D., Marco, A., Turchetta, M., and Trimpe, S. Gosafe: Globally optimal safe robot learning. In *ICRA*, 2021.
- <span id="page-10-8"></span>Bengio, Y., Louradour, J., Collobert, R., and Weston, J. Curriculum learning. In *ICML*, volume 26, 2009.
- <span id="page-10-9"></span>Beraha, M., Metelli, A. M., Papini, M., Tirinzoni, A., and Restelli, M. Feature selection via mutual information: New theoretical insights. In *IJCNN*, 2019.
- <span id="page-10-15"></span>Berkenkamp, F., Schoellig, A. P., and Krause, A. Safe controller optimization for quadrotors with gaussian processes. In *ICRA*, 2016.
- <span id="page-10-5"></span>Berkenkamp, F., Krause, A., and Schoellig, A. P. Bayesian optimization with safety constraints: safe and automatic parameter tuning in robotics. *Machine Learning*, 2021.
- <span id="page-10-12"></span>Berlind, C. and Urner, R. Active nearest neighbors in changing environments. In *ICML*, 2015.
- <span id="page-10-0"></span>Bickford Smith, F., Kirsch, A., Farquhar, S., Gal, Y., Foster, A., and Rainforth, T. Prediction-oriented bayesian active learning. In *AISTATS*, 2023.

- <span id="page-11-11"></span>Bickford Smith, F., Foster, A., and Rainforth, T. Making better use of unlabelled data in bayesian active learning. In *AISTATS*, 2024.
- <span id="page-11-20"></span>Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. Weight uncertainty in neural network. In *ICML*, 2015.
- <span id="page-11-1"></span>Bogunovic, I., Scarlett, J., Krause, A., and Cevher, V. Truncated variance reduction: A unified approach to bayesian optimization and level-set estimation. *NeurIPS*, 29, 2016.
- <span id="page-11-9"></span>Bottero, A., Luis, C., Vinogradska, J., Berkenkamp, F., and Peters, J. R. Information-theoretic safe exploration with gaussian processes. *NeurIPS*, 35, 2022.
- <span id="page-11-22"></span>Bottero, A. G., Luis, C. E., Vinogradska, J., Berkenkamp, F., and Peters, J. Information-theoretic safe bayesian optimization. *arXiv preprint arXiv:2402.15347*, 2024.
- <span id="page-11-13"></span>Bubeck, S., Munos, R., and Stoltz, G. Pure exploration in multi-armed bandits problems. In *ALT*, volume 20, 2009.
- <span id="page-11-0"></span>Chaloner, K. and Verdinelli, I. Bayesian experimental design: A review. *Statistical Science*, 1995.
- <span id="page-11-23"></span>Chandra, B. Quadrotor simulation, 2023. URL [https://github.com/Bharath2/](https://github.com/Bharath2/Quadrotor-Simulation) [Quadrotor-Simulation](https://github.com/Bharath2/Quadrotor-Simulation).
- <span id="page-11-6"></span>Chen, Y. and Krause, A. Near-optimal batch mode active learning and adaptive submodular optimization. In *ICML*, 2013.
- <span id="page-11-2"></span>Chowdhury, S. R. and Gopalan, A. On kernelized multi-armed bandits. In *ICML*, 2017.
- <span id="page-11-10"></span>Cohn, D. Neural network exploration using optimal experiment design. *NeurIPS*, 6, 1993.
- <span id="page-11-19"></span>Coleman, C., Chou, E., Katz-Samuels, J., Culatana, S., Bailis, P., Berg, A. C., Nowak, R., Sumbaly, R., Zaharia, M., and Yalniz, I. Z. Similarity search for efficient active learning and search of rare concepts. In *AAAI*, volume 36, 2022.
- <span id="page-11-12"></span>Cook, R. D. Detection of influential observation in linear regression. *Technometrics*, 19(1), 1977.
- <span id="page-11-8"></span>Cooper, S. E. and Netoff, T. I. Multidimensional bayesian estimation for deep brain stimulation using the safeopt algorithm. *medRxiv*, 2022.
- <span id="page-11-15"></span>Cover, T. M. *Elements of information theory*. John Wiley & Sons, 1999.
- <span id="page-11-16"></span>Das, A. and Kempe, D. Algorithms for subset selection in linear regression. In *STOC*, volume 40, 2008.
- <span id="page-11-3"></span>Das, A. and Kempe, D. Approximate submodularity and its applications: Subset selection, sparse approximation and dictionary selection. *JMLR*, 19(1), 2018.
- <span id="page-11-21"></span>Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., and Hennig, P. Laplace redux-effortless bayesian deep learning. *NeurIPS*, 34, 2021.
- <span id="page-11-7"></span>Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In *CVPR*, 2009.
- <span id="page-11-5"></span>Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In *NAACL*, 2019.
- <span id="page-11-18"></span>Emmenegger, N., Mutny, M., and Krause, A. Likelihood ratio confidence sets for sequential decision ` making. *NeurIPS*, 37, 2023.
- <span id="page-11-17"></span>Esfandiari, H., Karbasi, A., and Mirrokni, V. Adaptivity in adaptive submodularity. In *COLT*, 2021.
- <span id="page-11-4"></span>Eustratiadis, P., Dudziak, Ł., Li, D., and Hospedales, T. Neural fine-tuning search for few-shot learning. *ICLR*, 2024.
- <span id="page-11-14"></span>Fiez, T., Jain, L., Jamieson, K. G., and Ratliff, L. Sequential experimental design for transductive linear bandits. *NeurIPS*, 32, 2019.

- <span id="page-12-18"></span>Fu, B., Cao, Z., Wang, J., and Long, M. Transferable query selection for active domain adaptation. In *CVPR*, 2021.
- <span id="page-12-10"></span>Gal, Y., Islam, R., and Ghahramani, Z. Deep bayesian active learning with image data. In *ICML*, 2017.
- <span id="page-12-17"></span>Gao, M., Zhang, Z., Yu, G., Arık, S. Ö., Davis, L. S., and Pfister, T. Consistency-based semisupervised active learning: Towards minimizing labeling cost. In *ECCV*, 2020.
- <span id="page-12-6"></span>Gardner, J. R., Kusner, M. J., Xu, Z. E., Weinberger, K. Q., and Cunningham, J. P. Bayesian optimization with inequality constraints. In *ICML*, volume 2014, 2014.
- <span id="page-12-21"></span>Geifman, Y. and El-Yaniv, R. Deep active learning over the long tail. *arXiv preprint arXiv:1711.00941*, 2017.
- <span id="page-12-8"></span>Grandvalet, Y. and Bengio, Y. Semi-supervised learning by entropy minimization. *NeurIPS*, 17, 2004.
- <span id="page-12-15"></span>Graves, A., Bellemare, M. G., Menick, J., Munos, R., and Kavukcuoglu, K. Automated curriculum learning for neural networks. In *ICML*, 2017.
- <span id="page-12-16"></span>Graybill, F. A. *An introduction to linear statistical models*. Literary Licensing, LLC, 1961.
- <span id="page-12-7"></span>Guo, Y. and Greiner, R. Optimistic active-learning using mutual information. In *IJCAI*, volume 7, 2007.
- <span id="page-12-3"></span>Hacohen, G., Dekel, A., and Weinshall, D. Active learning on a budget: Opposite strategies suit high and low budgets. *ICML*, 2022.
- <span id="page-12-4"></span>Hardt, M. and Sun, Y. Test-time training on nearest neighbors for large language models. *ICLR*, 2024.
- <span id="page-12-2"></span>He, B., Lakshminarayanan, B., and Teh, Y. W. Bayesian deep ensembles via the neural tangent kernel. *NeurIPS*, 33, 2020.
- <span id="page-12-20"></span>Hendrycks, D. and Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*, 2017.
- <span id="page-12-12"></span>Hennig, P. and Schuler, C. J. Entropy search for information-efficient global optimization. *JMLR*, 13 (6), 2012.
- <span id="page-12-13"></span>Hernández-Lobato, J. M., Hoffman, M. W., and Ghahramani, Z. Predictive entropy search for efficient global optimization of black-box functions. *NeurIPS*, 27, 2014.
- <span id="page-12-14"></span>Hoffman, M. W. and Ghahramani, Z. Output-space predictive entropy search for flexible global optimization. In *NeurIPS workshop on Bayesian Optimization*, 2015.
- <span id="page-12-0"></span>Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. *JMLR*, 24(164), 2023.
- <span id="page-12-5"></span>Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the national academy of sciences*, 79(8), 1982.
- <span id="page-12-9"></span>Houlsby, N., Huszár, F., Ghahramani, Z., and Lengyel, M. Bayesian active learning for classification and preference learning. *CoRR*, 2011.
- <span id="page-12-19"></span>Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Active few-shot fine-tuning. *ICLR workshop on Bridging the Gap Between Practice and Theory in Deep Learning*, 2024.
- <span id="page-12-11"></span>Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., and Madry, A. Datamodels: Predicting predictions from training data. *arXiv preprint arXiv:2202.00622*, 2022.
- <span id="page-12-1"></span>Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*, 31, 2018.

- <span id="page-13-9"></span>Johnson, J., Douze, M., and Jégou, H. Billion-scale similarity search with gpus. *IEEE Transactions on Big Data*, 7(3), 2019.
- <span id="page-13-15"></span>Kaddour, J., Sæmundsson, S., et al. Probabilistic active meta-learning. *NeurIPS*, 33, 2020.
- <span id="page-13-18"></span>Kassraie, P. and Krause, A. Neural contextual bandits without regret. In *AISTATS*, 2022.
- <span id="page-13-5"></span>Khan, M. E. E., Immer, A., Abedi, E., and Korzepa, M. Approximate inference turns deep networks into gaussian processes. *NeurIPS*, 32, 2019.
- <span id="page-13-17"></span>Khanna, R., Elenberg, E., Dimakis, A., Negahban, S., and Ghosh, J. Scalable greedy feature selection via weak submodularity. In *AISTATS*, 2017.
- <span id="page-13-8"></span>Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In *ICLR*, 2014.
- <span id="page-13-20"></span>Kirsch, A. Black-box batch active learning for regression. *arXiv preprint arXiv:2302.08981*, 2023.
- <span id="page-13-12"></span>Kirsch, A., Van Amersfoort, J., and Gal, Y. Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. *NeurIPS*, 32, 2019.
- <span id="page-13-23"></span>Kirschner, J., Mutny, M., Hiller, N., Ischebeck, R., and Krause, A. Adaptive and safe bayesian optimization in high dimensions via one-dimensional subspaces. In *ICML*, 2019.
- <span id="page-13-13"></span>Koh, P. W. and Liang, P. Understanding black-box predictions via influence functions. In *ICML*, 2017.
- <span id="page-13-1"></span>Kothawade, S., Beck, N., Killamsetty, K., and Iyer, R. Similar: Submodular information measures based active learning in realistic scenarios. *NeurIPS*, 34, 2021.
- <span id="page-13-16"></span>Krause, A. and Golovin, D. Submodular function maximization. *Tractability*, 3, 2014.
- <span id="page-13-10"></span>Krause, A. and Guestrin, C. Nonmyopic active learning of gaussian processes: an explorationexploitation approach. In *ICML*, volume 24, 2007.
- <span id="page-13-11"></span>Krause, A., Singh, A., and Guestrin, C. Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies. *JMLR*, 9(2), 2008.
- <span id="page-13-7"></span>Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.
- <span id="page-13-14"></span>Kumari, L., Wang, S., Das, A., Zhou, T., and Bilmes, J. An end-to-end submodular framework for data-efficient in-context learning. In *NAACL*, 2024.
- <span id="page-13-21"></span>Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 30, 2017.
- <span id="page-13-6"></span>LeCun, Y., Cortes, C., and Burges, C. J. The mnist database of handwritten digits. *http://yann.lecun.com/exdb/mnist/*, 1998.
- <span id="page-13-19"></span>Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., and Sohl-Dickstein, J. Deep neural networks as gaussian processes. *ICLR*, 2018.
- <span id="page-13-4"></span>Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J., and Pennington, J. Wide neural networks of any depth evolve as linear models under gradient descent. *NeurIPS*, 32, 2019.
- <span id="page-13-3"></span>Lee, Y., Chen, A. S., Tajwar, F., Kumar, A., Yao, H., Liang, P., and Finn, C. Surgical fine-tuning improves adaptation to distribution shifts. *NeurIPS workshop on Distribution Shifts*, 2022.
- <span id="page-13-22"></span>Lewis, D. and Gale, W. A sequential algorithm for training text classifiers. In *SIGIR*, 1994.
- <span id="page-13-2"></span>Lewis, D. D. and Catlett, J. Heterogeneous uncertainty sampling for supervised learning. In *Machine learning proceedings*. 1994.
- <span id="page-13-0"></span>MacKay, D. J. Information-based objective functions for active data selection. *Neural computation*, 4(4), 1992.

- <span id="page-14-16"></span>Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., and Wilson, A. G. A simple baseline for bayesian uncertainty in deep learning. *NeurIPS*, 32, 2019.
- <span id="page-14-3"></span>Malladi, S., Wettig, A., Yu, D., Chen, D., and Arora, S. A kernel-based view of language model fine-tuning. In *ICML*, 2023.
- <span id="page-14-15"></span>Martens, J. and Grosse, R. Optimizing neural networks with kronecker-factored approximate curvature. In *ICML*, 2015.
- <span id="page-14-17"></span>Mehta, R., Shui, C., Nichyporuk, B., and Arbel, T. Information gain sampling for active learning in medical image classification. In *UNSURE*, 2022.
- <span id="page-14-10"></span>Murphy, K. P. *Probabilistic machine learning: Advanced topics*. MIT Press, 2023.
- <span id="page-14-7"></span>Mutny, M. and Krause, A. Experimental design for linear functionals in reproducing kernel hilbert spaces. *NeurIPS*, 35, 2022.
- <span id="page-14-2"></span>Nemhauser, G. L., Wolsey, L. A., and Fisher, M. L. An analysis of approximations for maximizing submodular set functions—i. *Mathematical programming*, 14, 1978.
- <span id="page-14-19"></span>Ostrovsky, R., Rabani, Y., Schulman, L. J., and Swamy, C. The effectiveness of lloyd-type methods for the k-means problem. *JACM*, 2013.
- <span id="page-14-4"></span>Pacchiano, A., Lee, J. N., and Brunskill, E. Experiment planning with function approximation. *NeurIPS*, 37, 2024.
- <span id="page-14-8"></span>Peng, H., Long, F., and Ding, C. Feature selection based on mutual information criteria of maxdependency, max-relevance, and min-redundancy. *IEEE Transactions on pattern analysis and machine intelligence*, 27(8), 2005.
- <span id="page-14-13"></span>Prabhu, V., Chandrasekaran, A., Saenko, K., and Hoffman, J. Active domain adaptation via clustering uncertainty-weighted embeddings. In *ICCV*, 2021.
- <span id="page-14-6"></span>Pruthi, G., Liu, F., Kale, S., and Sundararajan, M. Estimating training data influence by tracing gradient descent. In *NeurIPS*, 2019.
- <span id="page-14-14"></span>Rahimi, A. and Recht, B. Random features for large-scale kernel machines. *NeurIPS*, 20, 2007.
- <span id="page-14-11"></span>Rai, P., Saha, A., Daumé III, H., and Venkatasubramanian, S. Domain adaptation meets active learning. In *NAACL HLT workshop on Active Learning for Natural Language Processing*, 2010.
- <span id="page-14-9"></span>Rothfuss, J., Koenig, C., Rupenyan, A., and Krause, A. Meta-learning priors for safe bayesian optimization. In *COLT*, 2023.
- <span id="page-14-20"></span>Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., Wen, Z., et al. A tutorial on thompson sampling. *Foundations and Trends® in Machine Learning*, 11(1), 2018.
- <span id="page-14-12"></span>Saha, A., Rai, P., Daumé, H., Venkatasubramanian, S., and DuVall, S. L. Active supervised domain adaptation. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD*, 2011.
- <span id="page-14-18"></span>Scheffer, T., Decomain, C., and Wrobel, S. Active hidden markov models for information extraction. In *IDA*, 2001.
- <span id="page-14-21"></span>Schreiter, J., Nguyen-Tuong, D., Eberts, M., Bischoff, B., Markert, H., and Toussaint, M. Safe exploration for active learning with gaussian processes. In *ECML PKDD*, 2015.
- <span id="page-14-5"></span>Sener, O. and Savarese, S. Active learning for convolutional neural networks: A core-set approach. *ICLR*, 2017.
- <span id="page-14-1"></span>Seo, S., Wallat, M., Graepel, T., and Obermayer, K. Gaussian process regression: Active data selection and test point rejection. In *Mustererkennung 2000*. Springer, 2000.
- <span id="page-14-0"></span>Settles, B. Active learning literature survey. Technical report, University of Wisconsin-Madison Department of Computer Sciences, 2009.

- <span id="page-15-6"></span>Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In *EMNLP*, 2008.
- <span id="page-15-11"></span>Shoham, N. and Avron, H. Experimental design for overparameterized learning with application to single shot deep active learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2023.
- <span id="page-15-16"></span>Shwartz-Ziv, R. and LeCun, Y. To compress or not to compress–self-supervised learning and information theory: A review. *arXiv preprint arXiv:2304.09355*, 2023.
- <span id="page-15-14"></span>Soviany, P., Ionescu, R. T., Rota, P., and Sebe, N. Curriculum learning: A survey. *IJCV*, 2022.
- <span id="page-15-1"></span>Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. Gaussian process optimization in the bandit setting: No regret and experimental design. In *ICML*, volume 27, 2009.
- <span id="page-15-17"></span>Strang, G. *Introduction to linear algebra*. SIAM, 5 edition, 2016.
- <span id="page-15-18"></span>Su, J.-C., Tsai, Y.-H., Sohn, K., Liu, B., Maji, S., and Chandraker, M. Active adversarial domain adaptation. In *WACV*, 2020.
- <span id="page-15-9"></span>Sui, Y., Gotovos, A., Burdick, J., and Krause, A. Safe exploration for optimization with gaussian processes. In *ICML*, 2015.
- <span id="page-15-10"></span>Sukhija, B., Turchetta, M., Lindner, D., Krause, A., Trimpe, S., and Baumann, D. Gosafeopt: Scalable safe exploration for global optimization of dynamical systems. *Artificial Intelligence*, 2023.
- <span id="page-15-7"></span>Tamkin, A., Nguyen, D., Deshpande, S., Mu, J., and Goodman, N. Active learning helps pretrained models learn the intended task. *NeurIPS*, 35, 2022.
- <span id="page-15-5"></span>Tan, M. and Le, Q. Efficientnet: Rethinking model scaling for convolutional neural networks. In *ICML*, 2019.
- <span id="page-15-19"></span>Thompson, W. R. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 1933.
- <span id="page-15-21"></span>Tu, S., Frostig, R., Singh, S., and Sindhwani, V. JAX: A python library for differentiable optimal control on accelerators, 2023. URL <http://github.com/google/trajax>.
- <span id="page-15-20"></span>Turchetta, M., Berkenkamp, F., and Krause, A. Safe exploration for interactive machine learning. *NeurIPS*, 32, 2019.
- <span id="page-15-2"></span>Vakili, S., Khezeli, K., and Picheny, V. On information gain and regret bounds in gaussian process bandits. In *AISTATS*, 2021.
- <span id="page-15-13"></span>Vapnik, V. *Estimation of dependences based on empirical data*. Springer Science & Business Media, 1982.
- <span id="page-15-15"></span>Vergara, J. R. and Estévez, P. A. A review of feature selection methods based on mutual information. *Neural computing and applications*, 24, 2014.
- <span id="page-15-8"></span>Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. *NeurIPS*, 29, 2016.
- <span id="page-15-3"></span>Wainwright, M. J. *High-dimensional statistics: A non-asymptotic viewpoint*, volume 48. Cambridge university press, 2019.
- <span id="page-15-0"></span>Wang, C., Sun, S., and Grosse, R. Beyond marginal uncertainty: How accurately can bayesian regression models estimate posterior predictive correlations? In *AISTATS*, 2021.
- <span id="page-15-12"></span>Wang, Z. and Jegelka, S. Max-value entropy search for efficient bayesian optimization. In *ICML*, 2017.
- <span id="page-15-4"></span>Wei, A., Hu, W., and Steinhardt, J. More than a toy: Random matrix models predict how real-world neural representations generalize. In *ICML*, 2022.

- <span id="page-16-5"></span>Widmer, D., Kang, D., Sukhija, B., Hübotter, J., Krause, A., and Coros, S. Tuning legged locomotion controllers via safe bayesian optimization. *CORL*, 2023.
- <span id="page-16-10"></span>Wilks, S. S. Certain generalizations in the analysis of variance. *Biometrika*, 1932.
- <span id="page-16-1"></span>Williams, C. K. and Rasmussen, C. E. *Gaussian processes for machine learning*, volume 2. MIT press Cambridge, MA, 2006.
- <span id="page-16-4"></span>Wischnewski, A., Betz, J., and Lohmann, B. A model-free algorithm to safely approach the handling limit of an autonomous racecar. In *ICCVE*, 2019.
- <span id="page-16-6"></span>Xia, M., Malladi, S., Gururangan, S., Arora, S., and Chen, D. Less: Selecting influential data for targeted instruction tuning. In *ICML*, 2024.
- <span id="page-16-7"></span>Xu, M. and Kazantsev, G. Understanding goal-oriented active learning via influence functions. In *NeurIPS Workshop on Machine Learning with Guarantees*, 2019.
- <span id="page-16-8"></span>Ye, J., Wu, Z., Feng, J., Yu, T., and Kong, L. Compositional exemplars for in-context learning. In *ICML*, 2023.
- <span id="page-16-3"></span>Yehuda, O., Dekel, A., Hacohen, G., and Weinshall, D. Active learning through a covering lens. *NeurIPS*, 35, 2022.
- <span id="page-16-11"></span>Yu, H. and Kim, S. Passive sampling for regression. In *ICDM*, 2010.
- <span id="page-16-0"></span>Yu, K., Bi, J., and Tresp, V. Active learning via transductive experimental design. In *ICML*, volume 23, 2006.
- <span id="page-16-2"></span>Zanette, A., Dong, K., Lee, J. N., and Brunskill, E. Design of experiments for stochastic contextual linear bandits. *NeurIPS*, 34, 2021.
- <span id="page-16-9"></span>Zheng, H., Liu, R., Lai, F., and Prakash, A. Coverage-centric coreset selection for high pruning rates. *ICLR*, 2023.

# Appendices

A general principle of "transductive learning" was already formulated by the famous computer scientist Vladimir Vapnik in the 20th century. Vapnik proposes the following "imperative for a complex world":

*When solving a problem of interest, do not solve a more general problem as an intermediate step. Try to get the answer that you really need but not a more general one.*

– [Vapnik](#page-15-13) [\(1982\)](#page-15-13)

These appendices provide additional background, proofs, experiment details, and ablation studies.

# Contents

| A |            | Additional Related Work                                         | 20 |
|---|------------|-----------------------------------------------------------------|----|
| B | Background |                                                                 |    |
|   | B.1        | Information Theory                                              | 20 |
|   | B.2        | Gaussian Processes<br>                                          | 20 |
| C | Proofs     |                                                                 | 21 |
|   | C.1        | Undirected Case of ITL                                          | 21 |
|   | C.2        | Non-adaptive Data Selection & Submodularity<br>                 | 21 |
|   | C.3        | Batch Diversity: Batch Selection as Non-adaptive Data Selection | 22 |
|   | C.4        | Measures of Synergies & Approximate Submodularity<br>           | 23 |
|   | C.5        | Convergence of Marginal Gain                                    | 25 |
|   | C.6        | Proof of Theorem 3.2                                            | 26 |
|   | C.7        | Proof of Theorem 3.3                                            | 31 |
|   | C.8        | Proof of Theorem 5.1                                            | 32 |
|   | C.9        | Useful Facts and Inequalities                                   | 36 |
| D |            | Interpretations & Approximations of Principle (†)               | 37 |
|   | D.1        | Interpretations of ITL                                          | 37 |
|   | D.2        | Interpretations of VTL<br>                                      | 37 |
|   | D.3        | Mean Marginal ITL                                               | 38 |
|   | D.4        | Correlation-based Transductive Learning                         | 40 |
|   | D.5        | Summary<br>                                                     | 40 |
| E |            | Stochastic Target Spaces                                        | 41 |
| F |            | Closed-form Decision Rules                                      | 41 |
| G |            | Computational Complexity                                        | 42 |
| H |            | Additional GP Experiments & Details                             | 42 |

| I |     | Alternative Settings for Active Fine-Tuning                 | 43 |
|---|-----|-------------------------------------------------------------|----|
|   | I.1 | Prediction Targets are Contained in Sample Space: A ⊆ S<br> | 43 |
|   | I.2 | Active Domain Adaptation<br>                                | 44 |
| J |     | Additional NN Experiments & Details                         | 44 |
|   | J.1 | Experiment Details<br>                                      | 45 |
|   | J.2 | Embeddings and Kernels<br>                                  | 46 |
|   | J.3 | Towards Uncertainty Quantification in Latent Space          | 47 |
|   | J.4 | Batch Selection via Conditional Embeddings<br>              | 47 |
|   | J.5 | Baselines<br>                                               | 48 |
|   | J.6 | Additional experiments                                      | 52 |
|   | J.7 | Ablation study of noise standard deviation ρ                | 52 |
| K |     | Additional Safe BO Experiments & Details                    | 54 |
|   | K.1 | A More Exploitative Stochastic Target Space<br>             | 54 |
|   | K.2 | Detailed Comparison with Prior Works                        | 55 |
|   | K.3 | Jumping Past Local Barriers<br>                             | 60 |
|   | K.4 | Experiment Details<br>                                      | 61 |

# <span id="page-19-1"></span>A Additional Related Work

The general principle of non-active "transductive learning" was introduced by Vapnik (1982). The notion of "target" from transductive active learning is akin to the notion of "task" in curriculum learning (Bengio et al., 2009; Graves et al., 2017; Soviany et al., 2022). The study of settings where the irreducible uncertainty is zero is related to the study of estimability in experimental design (Graybill, 1961; Mutny & Krause, 2022). In feature selection, selecting features that maximize information gain with respect to a to-be-predicted label is a standard approach (Peng et al., 2005; Vergara & Estévez, 2014; Beraha et al., 2019) which is akin to ITL (cf. Appendix D). The themes of relevance and diversity are also important for efficient in-context learning (e.g., Ye et al., 2023; Kumari et al., 2024) and data pruning (Zheng et al., 2023). Transductive active learning is complimentary to other learning methodologies, such as semi-supervised learning (Gao et al., 2020), self-supervised learning (Shwartz-Ziv & LeCun, 2023; Balestriero et al., 2023), and meta-learning (Kaddour et al., 2020; Rothfuss et al., 2023).

# <span id="page-19-0"></span>**B** Background

### <span id="page-19-2"></span>**B.1** Information Theory

Throughout this work,  $\log$  denotes the natural logarithm. Given random vectors x and y, we denote by

$$\begin{aligned} & \mathbf{H}[\boldsymbol{x}] \stackrel{\mathrm{def}}{=} \mathbb{E}_{p(\boldsymbol{x})}[-\log p(\boldsymbol{x})], \ & \mathbf{H}[\boldsymbol{x} \mid \boldsymbol{y}] \stackrel{\mathrm{def}}{=} \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[-\log p(\boldsymbol{x} \mid \boldsymbol{y})], \quad \text{and} \ & \mathbf{I}(\boldsymbol{x}; \boldsymbol{y}) \stackrel{\mathrm{def}}{=} \mathbf{H}[\boldsymbol{x}] - \mathbf{H}[\boldsymbol{x} \mid \boldsymbol{y}] \end{aligned}$$

the (differential) entropy, conditional entropy, and information gain, respectively (Cover, 1999).9

The multivariate information gain (Murphy, 2023) between random vectors x, y, z is given by

<span id="page-19-7"></span>
$$I(\boldsymbol{x}; \boldsymbol{y}; \boldsymbol{z}) \stackrel{\text{def}}{=} I(\boldsymbol{x}; \boldsymbol{y}) - I(\boldsymbol{x}; \boldsymbol{y} \mid \boldsymbol{z})$$

$$= I(\boldsymbol{x}; \boldsymbol{y}) + I(\boldsymbol{x}; \boldsymbol{z}) - I(\boldsymbol{x}; \boldsymbol{y}, \boldsymbol{z}).$$
(6)

When  $I(x; y; z) \neq 0$  it is said that y and z interact regarding their information about x. If the interaction is positive, it is said that the information of z about x is redundant given y. Conversely, if the interaction is negative, it is said that the information of z about x is synergistic with y. The notion of synergy is akin to the frequentist notion of "suppressor variables" in linear regression (Das & Kempe, 2008).

#### <span id="page-19-3"></span>**B.2** Gaussian Processes

The stochastic process f is a Gaussian process (GP, Williams & Rasmussen (2006)), denoted  $f \sim \mathcal{GP}(\mu, k)$ , with mean function  $\mu$  and kernel k if for any finite subset  $X = \{x_1, \ldots, x_n\} \subseteq \mathcal{X}$ ,  $\mathbf{f}_X \sim \mathcal{N}(\mathbf{\mu}_X, \mathbf{K}_{XX})$  is jointly Gaussian with mean vector  $\mathbf{\mu}_X(i) = \mu(\mathbf{x}_i)$  and covariance matrix  $\mathbf{K}_{XX}(i,j) = k(\mathbf{x}_i, \mathbf{x}_j)$ .

In the following, we formalize the assumptions from the GP setting (cf. Section 3.1).

<span id="page-19-5"></span>**Assumption B.1** (Gaussian prior). We assume that  $f \sim \mathcal{GP}(\mu, k)$  with known mean function  $\mu$  and kernel k.

<span id="page-19-6"></span>**Assumption B.2** (Gaussian noise). We assume that the noise  $\varepsilon_x$  is mutually independent and zero-mean Gaussian with known variance  $\rho^2(x) > 0$ . We write  $P_X = \operatorname{diag} \rho^2(x_1), \dots, \rho^2(x_n)$ .

Under Assumptions B.1 and B.2, the posterior distribution of f after observing points X is  $\mathcal{GP}(\mu_n, k_n)$  with

$$\mu_n(\boldsymbol{x}) = \mu(\boldsymbol{x}) + \boldsymbol{K}_{\boldsymbol{x}X}(\boldsymbol{K}_{XX} + \boldsymbol{P}_X)^{-1}(\boldsymbol{y}_X - \boldsymbol{\mu}_X),$$

$$k_n(\boldsymbol{x}, \boldsymbol{x}') = k(\boldsymbol{x}, \boldsymbol{x}') - \boldsymbol{K}_{\boldsymbol{x}X}(\boldsymbol{K}_{XX} + \boldsymbol{P}_X)^{-1}\boldsymbol{K}_{X\boldsymbol{x}'},$$

$$\sigma_n^2(\boldsymbol{x}) = k_n(\boldsymbol{x}, \boldsymbol{x}).$$

<span id="page-19-4"></span><sup>&</sup>lt;sup>9</sup>One has to be careful to ensure that I(x; y) exists, i.e.,  $|I(x; y)| < \infty$ . We will assume that this is the case throughout this work. When x and y are jointly Gaussian, this is satisfied when the noise variance  $\rho^2$  is positive.

For Gaussian random vectors f and y, the entropy is  $H[f] = \frac{n}{2} \log(2\pi e) + \frac{1}{2} \log |Var[f]|$ , the information gain is  $I(f; y) = \frac{1}{2} (\log |Var[y]| - \log |Var[y||f]|)$ , and

$$\gamma_n = \max_{\substack{X \subseteq \mathcal{X} \\ |X| \le n}} \frac{1}{2} \log \left| \boldsymbol{I} + \boldsymbol{P}_X^{-1} \boldsymbol{K}_{XX} \right|.$$

# <span id="page-20-1"></span>C Proofs

We will write

- $\sigma^2 \stackrel{\text{def}}{=} \max_{\boldsymbol{x} \in \mathcal{X}} \sigma_0^2(\boldsymbol{x})$ , and
- $\tilde{\sigma}^2 \stackrel{\text{def}}{=} \max_{\boldsymbol{x} \in \mathcal{X}} \sigma_0^2(\boldsymbol{x}) + \rho^2(\boldsymbol{x}).$

The following is a brief overview of the structure of this section:

- 1. Appendix C.1 relates ITL in the inductive learning setting ( $S \subseteq A$ ) to prior work.
- 2. Appendix C.2 relates the designs selected by ITL and VTL to the optimal designs for corresponding non-adaptive objectives.
- 3. Appendix C.3 shows that batch selection via ITL or VTL leads to informative and diverse batches, utilizing the results from Appendix C.2.
- 4. Appendix C.4 introduces measures of synergies that generalize the submodularity assumption (cf. Assumption 3.1).
- 5. Appendix C.5 proves key results on the convergence of the ITL and VTL objectives.
- 6. Appendix C.6 proves Theorem 3.2 (convergence in GP setting).
- 7. Appendix C.7 proves Theorem 3.3 (convergence in agnostic setting).
- 8. Appendix C.8 proves Theorem 5.1 (convergence in safe BO application).
- 9. Appendix C.9 includes useful facts.

#### <span id="page-20-0"></span>C.1 Undirected Case of ITL

We briefly examine the important special case of ITL where  $S \subseteq A$ . In this setting, for all  $x \in S$ , the decision rule of ITL simplifies to

$$I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_{n}) \stackrel{(i)}{=} I(\mathbf{f}_{\mathcal{A} \setminus \{\mathbf{x}\}}; y_{\mathbf{x}} \mid f_{\mathbf{x}}, \mathcal{D}_{n}) + I(f_{\mathbf{x}}; y_{\mathbf{x}} \mid \mathcal{D}_{n})$$

$$\stackrel{(ii)}{=} I(f_{\mathbf{x}}; y_{\mathbf{x}} \mid \mathcal{D}_{n})$$

$$= H[y_{\mathbf{x}} \mid \mathcal{D}_{n}] - H[\varepsilon_{\mathbf{x}}]$$

where (i) follows from the chain rule of information gain and  $x \in S \subseteq A$ ; and (ii) follows from the conditional independence  $f_A \perp y_x \mid f_x$ .

If additionally f is a GP then

$$H[y_{\boldsymbol{x}} \mid \mathcal{D}_n] - H[\varepsilon_{\boldsymbol{x}}] = \frac{1}{2} \log \left( 1 + \frac{\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_n]}{\operatorname{Var}[\varepsilon_{\boldsymbol{x}}]} \right).$$

This decision rule has also been termed *total information gain* (MacKay, 1992). When  $S \subseteq A$  and observation noise is homoscedastic, this decision rule is equivalent to uncertainty sampling.

#### <span id="page-20-2"></span>C.2 Non-adaptive Data Selection & Submodularity

Recall the non-myopic information gain  $\psi_{\mathcal{A}}(X) = \mathrm{I}(\mathbf{f}_{\mathcal{A}}; \mathbf{y}_X)$  (ITL) and variance reduction  $\psi_{\mathcal{A}}(X) = \mathrm{tr} \ \mathrm{Var}[\mathbf{f}_{\mathcal{A}}] - \mathrm{tr} \ \mathrm{Var}[\mathbf{f}_{\mathcal{A}} \mid \mathbf{y}_X]$  (VTL) objective functions from Assumption 3.1. In this section, we will relate the designs selected by ITL and VTL to the optimal designs for these objectives. To this end, consider the non-adaptive optimization problem

$$X^{\star} = \operatorname*{arg\,max}_{\substack{X \subseteq \mathcal{S} \\ |X| = k}} \psi_{\mathcal{A}}(X).$$

**Lemma C.1.** For both ITL and VTL,  $\psi_A$  is non-negative and monotone.

*Proof.* For ITL,  $\psi_A(X) \ge 0$  follows from the non-negativity of mutual information. To conclude monotonicity, note that for any  $X' \subseteq X \subseteq \mathcal{S}$ ,

$$\mathrm{I}(\textit{\textbf{f}}_{\mathcal{A}}; \textit{\textbf{y}}_{X'}) = \mathrm{H}[\textit{\textbf{f}}_{\mathcal{A}}] - \mathrm{H}[\textit{\textbf{f}}_{\mathcal{A}} \mid \textit{\textbf{y}}_{X'}] \leq \mathrm{H}[\textit{\textbf{f}}_{\mathcal{A}}] - \mathrm{H}[\textit{\textbf{f}}_{\mathcal{A}} \mid \textit{\textbf{y}}_{X}] = \mathrm{I}(\textit{\textbf{f}}_{\mathcal{A}}; \textit{\textbf{y}}_{X})$$

due to monotonicity of conditional entropy (which is also called the "information never hurts" principle).

For VTL, recall that  $\operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid y_X] \leq \operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid y_{X'}]$  for any  $X' \subseteq X \subseteq \mathcal{S}$  (with an implicit expectation over  $y_X, y_{X'}$ ). Non-negativity and monotonicity of  $\psi_{\mathcal{A}}$  then follow analogously to ITL.  $\square$ 

**Lemma C.2.** The marginal gain  $\Delta_{\mathcal{A}}(\mathbf{x} \mid X) \stackrel{\text{def}}{=} \psi_{\mathcal{A}}(X \cup \{\mathbf{x}\}) - \psi_{\mathcal{A}}(X)$  of  $\mathbf{x} \in \mathcal{S}$  given  $X \subseteq \mathcal{S}$  is the ITL and VTL objective, respectively.

Proof. For ITL,

$$\begin{split} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid X) &= \mathrm{I}(\boldsymbol{f}_{\mathcal{A}}; \boldsymbol{y}_{X}, y_{\boldsymbol{x}}) - \mathrm{I}(\boldsymbol{f}_{\mathcal{A}}; \boldsymbol{y}_{X}) \\ &= \mathrm{H}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}] - \mathrm{H}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}, y_{\boldsymbol{x}}] \\ &= \mathrm{I}(\boldsymbol{f}_{\mathcal{A}}; y_{\boldsymbol{x}} \mid \boldsymbol{y}_{X}) \end{split}$$

which is precisely the ITL objective.

For VTL,

$$\Delta_{\mathcal{A}}(\boldsymbol{x} \mid X) = \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}] - \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}, y_{\boldsymbol{x}}]$$
$$= -\operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}, y_{\boldsymbol{x}}] + \operatorname{const}$$

which is precisely the VTL objective.

**Definition C.3** (Submodularity).  $\psi_A$  is submodular if and only if for all  $x \in S$  and  $X' \subseteq X \subseteq S$ ,

$$\Delta_{\mathcal{A}}(\boldsymbol{x} \mid X') \geq \Delta_{\mathcal{A}}(\boldsymbol{x} \mid X).$$

<span id="page-21-1"></span>**Theorem C.4** (Nemhauser et al. (1978)). Let Assumption 3.1 hold. For any  $n \ge 1$ , if ITL or VTL selected  $x_{1:n}$ , respectively, then

$$\psi_{\mathcal{A}}(\boldsymbol{x}_{1:n}) \ge (1 - 1/e) \max_{\substack{X \subseteq \mathcal{S} \\ |X| \le n}} \psi_{\mathcal{A}}(X).$$

*Proof.* This is a special case of a canonical result from non-negative monotone submodular function maximization (Nemhauser et al., 1978; Krause & Golovin, 2014). □

#### <span id="page-21-0"></span>C.3 Batch Diversity: Batch Selection as Non-adaptive Data Selection

Recall the non-adaptive optimization problem

$$B_{n,k} = \operatorname*{arg\,max}_{\substack{B \subseteq \mathcal{S} \\ |B| = k}} \mathrm{I}(\mathbf{f}_{\mathcal{A}}; \mathbf{y}_B \mid \mathcal{D}_{n-1})$$

from Equation (3) with batch size k > 0, and denote by  $B'_{n,k} = x_{n,1:k}$  the greedy approximation from Equation (3). The selection of an individual batch can be seen as a single non-adaptive optimization problem with marginal gain

$$\Delta_{n}(\boldsymbol{x} \mid B) = I(\boldsymbol{f}_{\mathcal{A}}; \boldsymbol{y}_{B}, y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}) - I(\boldsymbol{f}_{\mathcal{A}}; \boldsymbol{y}_{B} \mid \mathcal{D}_{n-1})$$

$$= H[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, \boldsymbol{y}_{B}] - H[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, \boldsymbol{y}_{B}, y_{\boldsymbol{x}}]$$

$$= I(\boldsymbol{f}_{\mathcal{A}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}, \boldsymbol{y}_{B})$$

and which is precisely the objective function of ITL from Equation (3). Hence, the approximation guarantees from Theorems C.4 and C.11 apply. The derivation is analogous for VTL.

Prior work has shown that the greedy solution  $B'_n$  is also competitive with a fully sequential "batchless" decision rule (Chen & Krause, 2013; Esfandiari et al., 2021).

#### <span id="page-22-0"></span>C.4 Measures of Synergies & Approximate Submodularity

We will now show that "downstream synergies", if present, can be seen as a source of learning complexity, which is orthogonal to the information capacity  $\gamma_n$ .

**Example C.5.** Consider the example where f is a stochastic process of three random variables X, Y, Z where X and Y are Bernoulli  $(p = \frac{1}{2})$ , and Z is the XOR of X and Y. Suppose that observations are exact (i.e.,  $\varepsilon_n = 0$ ), that the target space A comprises the output variable Z while the sample space S comprises the input variables X and Y. Observing any single X or Y yields no information about Z: I(Z;X) = I(Z;Y) = 0, however, observing both inputs jointly perfectly determines Z: I(Z;X,Y) = 1. Thus,  $\gamma_n(A;S) = 1$  if  $n \ge 2$  and  $\gamma_n(A;S) = 0$  else.

Learning about Z in examples of this kind is difficult for agents that make decisions greedily, since the next action (observing X or Y) yields no signal about its long-term usefulness. We call a sequence of observations, such as  $\{X,Y\}$ , synergistic since its combined information value is larger than the individual values. The prevalence of synergies is not captured by the information capacity  $\gamma_n(\mathcal{A};\mathcal{S})$  since it measures only the joint information gain of n samples within  $\mathcal{S}$ . Instead, the prevalence of synergies is captured by the sequence  $\Gamma_n \stackrel{\text{def}}{=} \max_{\boldsymbol{x} \in \mathcal{S}} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid \boldsymbol{x}_{1:n})$ , which measures the maximum information gain of  $y_{n+1}$ . If  $\Gamma_n > \Gamma_{n-1}$  at any round n, this indicates a synergy. The following key object measures the additional complexity due to synergies.

<span id="page-22-1"></span>**Definition C.6** (Task complexity). For  $n \ge 1$ , assuming  $\Gamma_i > 0$  for all  $1 \le i \le n$ , we define the *task complexity* as

$$\alpha_{\mathcal{A},\mathcal{S}}(n) \stackrel{\text{def}}{=} \max_{i \in \{0,\dots,n-1\}} \frac{\Gamma_{n-1}}{\Gamma_i}.$$

Note that  $\alpha_{\mathcal{A},\mathcal{S}}(n)$  is large only if the information gain of  $y_n$  is larger than that of a previous observation  $y_i$ . Intuitively, if  $\alpha_{\mathcal{A},\mathcal{S}}(n)$  is large, the agent had to discover the *implicit* intermediate observations  $y_1,\ldots,y_{n-1}$  that lead to downstream synergies. We will subsequently formalize the intimate connections of the task complexity to synergies and submodularity. Note that in the GP setting,  $\alpha_{\mathcal{A},\mathcal{S}}(n)$  can be computed online by keeping track of the smallest  $\Gamma_i$  during previous rounds i. Further, note that  $\alpha_{\mathcal{A},\mathcal{S}}(n) \leq 1$  if  $\psi_{\mathcal{A}}$  is submodular.

### C.4.1 The Information Ratio

Another object will prove useful in our analysis of synergies.

Consider an alternative multiplicative interpretation of the multivariate information gain (cf. Equation (7)), which we call the *information ratio* of  $X \subseteq \mathcal{S}$  given  $D \subseteq \mathcal{S}$ ,  $|X|, |D| < \infty$ :

<span id="page-22-2"></span>
$$\bar{\kappa}(X \mid D) \stackrel{\text{def}}{=} \frac{\sum_{\boldsymbol{x} \in X} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid D)}{\Delta_{\mathcal{A}}(X \mid D)} \in [0, \infty). \tag{8}$$

Observe that  $\bar{\kappa}(X\mid D)$  measures the synergy properties of  $\textbf{\textit{y}}_X$  with respect to  $\textbf{\textit{f}}_{\mathcal{A}}$  given  $\textbf{\textit{y}}_D$  in a multiplicative sense. That is, if  $\bar{\kappa}(X\mid D)>1$  then information in  $\textbf{\textit{y}}_X$  is redundant, whereas if  $\bar{\kappa}(X\mid D)<1$  then information in  $\textbf{\textit{y}}_X$  is synergistic, and if  $\bar{\kappa}(X\mid D)=1$  then  $\textbf{\textit{y}}_X$  do not mutually interact with respect to  $\textbf{\textit{f}}_{\mathcal{A}}$  (all given  $\textbf{\textit{y}}_D$ ). In the degenerate case where  $\Delta_{\mathcal{A}}(X\mid D)=0$  (which implies  $\sum_{\boldsymbol{x}\in X}\Delta_{\mathcal{A}}(\boldsymbol{x}\mid D)=0$ ) we therefore let  $\bar{\kappa}(X\mid D)=1$ .

The information ratio of ITL is strictly positive in the Gaussian case We prove the following straightforward lower bound to the information ratio of ITL.

<span id="page-22-3"></span>**Lemma C.7.** Let  $X, D \subseteq \mathcal{S}, |X|, |D| < \infty$ . If  $\mathbf{f}_{\mathcal{A}}$  and  $\mathbf{y}_{X \cup D}$  are jointly Gaussian then  $\bar{\kappa}(X \mid D) > 0$ .

*Proof.* W.l.o.g. assume  $D = \emptyset$ . We let  $X = \{x_1, \dots, x_k\}$  and prove lower and upper bound separately. We assume w.l.o.g. that  $I(f_{\mathcal{A}}; y_X) > 0$  which implies  $|\operatorname{Var}[f_{\mathcal{A}} \mid y_X]| < |\operatorname{Var}[f_{\mathcal{A}}]|$ . Thus, there exists some i such that  $f_{\mathcal{A}}$  and  $y_{x_i}$  are dependent, so  $|\operatorname{Var}[f_{\mathcal{A}} \mid y_{x_i}]| < |\operatorname{Var}[f_{\mathcal{A}}]|$  which implies  $I(f_{\mathcal{A}}; y_{x_i}) > 0$ . We therefore conclude that  $\bar{\kappa}(X) > 0$ .

The following example shows that this lower bound is tight.

**Example C.8** (Synergies of Gaussian random variables, inspired by Section 3 of Barrett (2015)). Consider the three random variables X, Y, and Z (think  $A = \{X\}$  and  $S = \{Y, Z\}$ ) which are jointly Gaussian with mean vector 0 and covariance matrix

$$\Sigma = \begin{bmatrix} 1 & a & a \\ a & 1 & 0 \\ a & 0 & 1 \end{bmatrix}, \quad \text{for } 2a^2 < 1$$

where the constraint on a is to ensure that  $\Sigma$  is positive definite. Computing the mutual information, we have

$$I(X;Y) = I(X;Z) = -\frac{1}{2}\log(1-a^2)$$

and  $\mathrm{I}(X;Y,Z) = -\frac{1}{2}\log(1-2a^2)$ . Therefore,

$$\frac{\mathrm{I}(X;Y)+\mathrm{I}(X;Z)}{\mathrm{I}(X;Y,Z)} = \frac{\log(1-2a^2+a^4)}{\log(1-2a^2)} < 1.$$

Note that

$$\lim_{a \to \frac{1}{\sqrt{2}}} \frac{\log(1 - 2a^2 + a^4)}{\log(1 - 2a^2)} = 0,$$

and hence — perhaps unintuitively — even if Y and Z are uncorrelated, their information about X may be arbitrarily synergistic.

# C.4.2 The Submodularity of the Special "Undirected" Case of ITL

In the inductive active learning problem considered in most prior works, where  $S \subseteq A$  and f is a Gaussian process, it holds for ITL that  $\alpha_{\mathcal{A},\mathcal{S}}(n)=1$  since all learning targets appear *explicitly* in  $\mathcal{S}$ : **Lemma C.9.** Let  $S \subseteq A$ . Then  $\psi_A$  of ITL is submodular.

<span id="page-23-0"></span>*Proof.* Fix any  $x \in \mathcal{S}$  and  $X' \subseteq X \subseteq \mathcal{S}$ . Let  $\bar{X} \stackrel{\text{def}}{=} X \setminus X'$ . By the definition of conditional information gain, we have

$$\Delta_{\mathcal{A}}(\boldsymbol{x}\mid X) = \mathrm{I}(y_{\boldsymbol{x}}; \boldsymbol{f}_{\!\mathcal{A}}\mid \boldsymbol{y}_{\!X}) = \mathrm{I}(y_{\boldsymbol{x}}; \boldsymbol{f}_{\!\mathcal{A}}, \boldsymbol{y}_{\!X'}\mid \boldsymbol{y}_{\!\bar{X}}) - \mathrm{I}(y_{\boldsymbol{x}}; \boldsymbol{y}_{\!X'}\mid \boldsymbol{y}_{\!\bar{X}}).$$

Since for any  ${\bm x}\in {\mathcal S}$  and  $X\subseteq {\mathcal S},$   $y_{{\bm x}}\perp {\bm y}_{\!X}\mid {\bm f}_{\!\mathcal A},$  this simplifies to

$$I(y_{\boldsymbol{x}}; \boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{X}) = I(y_{\boldsymbol{x}}; \boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{\bar{X}}) - I(y_{\boldsymbol{x}}; \boldsymbol{y}_{X'} \mid \boldsymbol{y}_{\bar{X}}).$$

It then follows from  $I(y_x; y_{X'} \mid y_{\bar{X}}) \ge 0$  that

$$\Delta_{\mathcal{A}}(\boldsymbol{x}\mid\boldsymbol{X}) = \mathrm{I}(y_{\boldsymbol{x}};\boldsymbol{f}_{\mathcal{A}}\mid\boldsymbol{y}_{X}) \leq \mathrm{I}(y_{\boldsymbol{x}};\boldsymbol{f}_{\mathcal{A}}\mid\boldsymbol{y}_{\bar{X}}) = \Delta_{\mathcal{A}}(\boldsymbol{x}\mid\boldsymbol{X}').$$

This implies that  $\alpha_{\mathcal{A},\mathcal{S}}(n) \leq 1$  for any n and  $\bar{\kappa}(X \mid D) \geq 1$  for any  $X, D \subseteq \mathcal{S}$  when  $\mathcal{S} \subseteq \mathcal{A}$ .

# C.4.3 The Submodularity Ratio

Building upon the theory of maximizing non-negative monotone submodular functions (Nemhauser et al., 1978; Krause & Golovin, 2014), Das & Kempe (2018) define the following notion of "approximate" submodularity:

**Definition C.10** (Submodularity ratio). The *submodularity ratio* of 
$$\psi_{\mathcal{A}}$$
 up to cardinality  $n \geq 1$  is 
$$\kappa_{\mathcal{A}}(n) \stackrel{\text{def}}{=} \min_{\substack{D \subseteq \boldsymbol{x}_{1:n} \\ X \subseteq \mathcal{S}: |X| \leq n \\ D \cap X = \emptyset}} \bar{\kappa}(X \mid D), \tag{9}$$

where they define  $\frac{0}{0} \equiv 1$ .  $\psi_{\mathcal{A}}$  is said to be  $\kappa$ -weakly submodular for some  $\kappa > 0$  if  $\inf_{n \in \mathbb{N}} \kappa_{\mathcal{A}}(n) \geq \kappa$ .

As a special case of Theorem 6 from Das & Kempe (2018), applying that  $\psi_A$  is non-negative and monotone, we obtain the following result.

<span id="page-23-1"></span>**Theorem C.11** (Das & Kempe (2018)). For any  $n \ge 1$ , if ITL or VTL selected  $x_{1:n}$ , respectively, then

$$\psi_{\mathcal{A}}(\boldsymbol{x}_{1:n}) \ge (1 - e^{-\kappa_{\mathcal{A}}(n)}) \max_{\substack{X \subseteq \mathcal{S} \\ |X| \le n}} \psi_{\mathcal{A}}(X).$$

If  $\psi_{\mathcal{A}}$  is submodular, it is implied that  $\kappa_{\mathcal{A}}(n) \geq 1$  for all  $n \geq 1$  in which case Theorem C.11 recovers Theorem C.4.

#### <span id="page-24-0"></span>C.5 Convergence of Marginal Gain

Our following analysis allows for changing target spaces  $\mathcal{A}_n$  and sample spaces  $\mathcal{S}_n$  (cf. Section 5), and to this end, we redefine  $\Gamma_n \stackrel{\text{def}}{=} \max_{\boldsymbol{x} \in \mathcal{S}_n} \Delta_{\mathcal{A}_n}(\boldsymbol{x} \mid \boldsymbol{x}_{1:n})$ . The following theorems show that the marginal gains of ITL and VTL converge to zero, and will serve as the main tool for establishing Theorems 3.2 and 3.3. We will abbreviate  $\alpha_{\mathcal{A},\mathcal{S}}(n)$  by  $\alpha_n$ .

<span id="page-24-1"></span>**Theorem C.12** (Convergence of Marginal Gain for ITL). Assume that Assumptions B.1 and B.2 are satisfied. Fix any integers  $n_1 > n_0 \ge 0$ ,  $\Delta = n_1 - n_0 + 1$  such that for all  $i \in \{n_0, \ldots, n_1 - 1\}$ ,  $A_{i+1} \subseteq A_i$  and  $S \stackrel{\text{def}}{=} S_{i+1} = S_i$ . Further, assume  $|A_{n_0}| < \infty$ . Then, if the sequence  $\{x_{i+1}\}_{i=n_0}^{n_1}$  was generated by ITL,

$$\Gamma_{n_1} \le \alpha_{n_1} \frac{\gamma_{\Delta}}{\Lambda}.\tag{10}$$

Moreover, if  $n_0 = 0$ ,

$$\Gamma_{n_1} \le \alpha_{n_1} \frac{\gamma_{\mathcal{A}_0, \mathcal{S}}(\Delta)}{\Lambda}.$$
 (11)

Proof. We have

$$\Gamma_{n_{1}} = \frac{1}{\Delta} \sum_{i=n_{0}}^{n_{1}} \Gamma_{n_{1}} \\
\stackrel{(i)}{\leq} \frac{\alpha_{n_{1}}}{\Delta} \sum_{i=n_{0}}^{n_{1}} \Gamma_{i} \\
= \frac{\alpha_{n_{1}}}{\Delta} \sum_{i=n_{0}}^{n_{1}} \max_{\boldsymbol{x} \in \mathcal{S}} I(\boldsymbol{f}_{\mathcal{A}_{i}}; y_{\boldsymbol{x}} \mid \boldsymbol{y}_{1:i}) \\
\stackrel{(ii)}{=} \frac{\alpha_{n_{1}}}{\Delta} \sum_{i=n_{0}}^{n_{1}} I(\boldsymbol{f}_{\mathcal{A}_{i}}; y_{\boldsymbol{x}_{i+1}} \mid \mathcal{D}_{i}) \\
\stackrel{(iv)}{\leq} \frac{\alpha_{n_{1}}}{\Delta} \sum_{i=n_{0}}^{n_{1}} I(\boldsymbol{f}_{\mathcal{A}_{n_{0}}}; y_{\boldsymbol{x}_{i+1}} \mid \boldsymbol{y}_{\boldsymbol{x}_{n_{0}+1:i}}, \mathcal{D}_{n_{0}}) \\
\stackrel{(v)}{=} \frac{\alpha_{n_{1}}}{\Delta} I(\boldsymbol{f}_{\mathcal{A}_{n_{0}}}; \boldsymbol{y}_{\boldsymbol{x}_{n_{0}+1:n_{1}+1}} \mid \mathcal{D}_{n_{0}}) \\
\stackrel{(vi)}{\leq} \frac{\alpha_{n_{1}}}{\Delta} \max_{\substack{X \subseteq \mathcal{S} \\ |X| = \Delta}} I(\boldsymbol{f}_{X}; \boldsymbol{y}_{X} \mid \mathcal{D}_{n_{0}}) \\
\stackrel{(vii)}{\leq} \frac{\alpha_{n_{1}}}{\Delta} \max_{\substack{X \subseteq \mathcal{S} \\ |X| = \Delta}} I(\boldsymbol{f}_{X}; \boldsymbol{y}_{X} \mid \mathcal{D}_{n_{0}}) \\
\stackrel{(vii)}{\leq} \frac{\alpha_{n_{1}}}{\Delta} \max_{\substack{X \subseteq \mathcal{S} \\ |X| = \Delta}} I(\boldsymbol{f}_{X}; \boldsymbol{y}_{X}) \\
= \alpha_{n_{1}} \frac{\gamma_{\Delta}}{\Delta}$$

where (i) follows from the definition of the task complexity  $\alpha_{n_1}$  (cf. Definition C.6); (ii) uses the objective of ITL and that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; (iii) uses  $\mathcal{A}_{i+1} \subseteq \mathcal{A}_i$  and monotonicity of information gain; (iv) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; (v) uses the chain rule of information gain; (vi) uses  $y_X \perp f_{\mathcal{A}_{n_0}} \mid f_X$  and the data processing inequality. The conditional independence follows from the assumption that the observation noise is independent. Similarly,  $y_X \perp \mathcal{D}_{n_0} \mid f_X$  which implies (vii).

If 
$$n_0 = 0$$
, then the bound before line  $(vi)$  simplifies to  $\alpha_{n_1} \gamma_{A_0,S}(\Delta)/\Delta$ .

The result for VTL is stated, for simplicity, only for the case where the target space and sample space are fixed.

<span id="page-25-2"></span>**Theorem C.13** (Convergence of Marginal Gain for VTL). Assume that Assumptions B.1 and B.2 are satisfied. Then for any  $n \ge 1$ , if the sequence  $\{x_i\}_{i=1}^n$  is generated by VTL,

$$\Gamma_{n-1} \le \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x'} \in \mathcal{A}} \gamma_{\{\boldsymbol{x'}\},\mathcal{S}}(n). \tag{12}$$

We remark that  $\sum_{x' \in \mathcal{A}} \gamma_{\{x'\},\mathcal{S}}(n) \leq |\mathcal{A}| \gamma_{\mathcal{A},\mathcal{S}}(n)$ .

Proof. We have

$$\begin{split} &\Gamma_{n-1} = \frac{1}{n} \sum_{i=0}^{n-1} \Gamma_{n-1} \\ &\stackrel{(i)}{\leq} \frac{\alpha_n}{n} \sum_{i=0}^{n-1} \Gamma_i \\ &= \frac{\alpha_n}{n} \sum_{i=0}^{n-1} \left[ \operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid \boldsymbol{y}_{1:i}] - \min_{\boldsymbol{x} \in \mathcal{S}} \operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid \boldsymbol{y}_{1:i}, \boldsymbol{y}_{\boldsymbol{x}}] \right] \\ &\stackrel{(ii)}{=} \frac{\alpha_n}{n} \sum_{i=0}^{n-1} \left[ \operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid \mathcal{D}_i] - \operatorname{tr} \operatorname{Var}[f_{\mathcal{A}} \mid \mathcal{D}_{i+1}] \right] \\ &\stackrel{(iii)}{\leq} \frac{\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \sum_{i=0}^{n-1} \log \left( \frac{\operatorname{Var}[f_{\boldsymbol{x}'} \mid \mathcal{D}_n]}{\operatorname{Var}[f_{\boldsymbol{x}'} \mid \mathcal{D}_{n+1}]} \right) \\ &= \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \sum_{i=0}^{n-1} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{n+1}} \mid \mathcal{D}_n) \\ &\stackrel{(iv)}{=} \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \sum_{i=0}^{n-1} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{n+1}} \mid \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &\stackrel{(v)}{=} \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \prod_{|X| = n} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &\leq \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \max_{|X| = n} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &= \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \max_{|X| = n} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &= \frac{2\sigma^2 \alpha_n}{n} \sum_{\boldsymbol{x}' \in \mathcal{A}} \max_{|X| = n} \operatorname{I}(f_{\boldsymbol{x}'}; \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \end{aligned}$$

where (i) follows from the definition of the task complexity  $\alpha_{n_1}$  (cf. Definition C.6); (ii) follows from the VTL decision rule and that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; (iii) follows from Lemma C.38 and monotonicity of variance; (iv) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; and (v) uses the chain rule of mutual information. The remainder of the proof is analogous to the proof of Theorem C.12 (cf. Appendix C.5).

**Keeping track of the task complexity online** In general, the task complexity  $\alpha_n$  may be larger than one in the "directed" setting (i.e., when  $\mathcal{S} \not\subseteq \mathcal{A}$ ). However, note that  $\alpha_n$  can easily be evaluated online by keeping track of the smallest  $\Gamma_i$  during previous rounds i.

# <span id="page-25-0"></span>C.6 Proof of Theorem 3.2

<span id="page-25-1"></span>We will now prove Theorem 3.2. We first prove the convergence of marginal variance within S for ITL, before proving the convergence outside S in Appendix C.6.1.

**Lemma C.14** (Uniform convergence of marginal variance within S for ITL). Assume that Assumptions B.1 and B.2 are satisfied. For any  $n \ge 0$  and  $x \in A \cap S$ ,

$$\sigma_n^2(\boldsymbol{x}) \le 2\tilde{\sigma}^2 \cdot \Gamma_n. \tag{13}$$

Proof. We have

$$\sigma_{n}^{2}(\boldsymbol{x}) = \operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \underbrace{\operatorname{Var}[f_{\boldsymbol{x}} \mid f_{\boldsymbol{x}}, \mathcal{D}_{n}]}_{0}$$

$$\stackrel{(i)}{=} \operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \rho^{2}(\boldsymbol{x}) - (\operatorname{Var}[y_{\boldsymbol{x}} \mid f_{\boldsymbol{x}}, \mathcal{D}_{n}] - \rho^{2}(\boldsymbol{x}))$$

$$= \operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \operatorname{Var}[y_{\boldsymbol{x}} \mid f_{\boldsymbol{x}}, \mathcal{D}_{n}]$$

$$\stackrel{(ii)}{\leq} \tilde{\sigma}^{2} \log \left( \frac{\operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_{n}]}{\operatorname{Var}[y_{\boldsymbol{x}} \mid f_{\boldsymbol{x}}, \mathcal{D}_{n}]} \right)$$

$$= 2\tilde{\sigma}^{2} \cdot \operatorname{I}(f_{\boldsymbol{x}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n})$$

$$\stackrel{(iii)}{\leq} 2\tilde{\sigma}^{2} \cdot \operatorname{Imax}_{\boldsymbol{x}' \in \mathcal{S}} \operatorname{I}(f_{\mathcal{A}}; y_{\boldsymbol{x}'} \mid \mathcal{D}_{n})$$

$$\stackrel{(iv)}{\leq} 2\tilde{\sigma}^{2} \cdot \operatorname{max}_{\boldsymbol{x}' \in \mathcal{S}} \operatorname{I}(f_{\mathcal{A}}; y_{\boldsymbol{x}'} \mid \mathcal{D}_{n})$$

$$\stackrel{(v)}{=} 2\tilde{\sigma}^{2} \cdot \Gamma_{n}$$

where (i) follows from the noise assumption (cf. Assumption B.2); (ii) follows from Lemma C.38 and using monotonicity of variance; (iii) follows from  $x \in \mathcal{A}$  and monotonicity of information gain; (iv) follows from  $x \in \mathcal{S}$ ; and (v) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations.

### <span id="page-26-0"></span>C.6.1 Convergence outside S for ITL

We will now show convergence of marginal variance to the irreducible uncertainty for points outside the sample space.

Our proof roughly proceeds as follows: We construct an "approximate Markov boundary" of x in S, and show (1) that the size of this Markov boundary is independent of n, and (2) that a small uncertainty reduction within the Markov boundary implies that the marginal variances at the Markov boundary and(!) x are small.

**Definition C.15** (Approximate Markov boundary). For any  $\epsilon > 0$ ,  $n \ge 0$ , and  $x \in \mathcal{X}$ , we denote by  $B_{n,\epsilon}(x)$  the smallest (multi-)subset of  $\mathcal{S}$  such that

$$\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_n, \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})}] \le \eta_{\mathcal{S}}^2(\boldsymbol{x}) + \epsilon. \tag{14}$$

We call  $B_{n,\epsilon}(x)$  an  $\epsilon$ -approximate Markov boundary of x in S.

Equation (14) is akin to the notion of the smallest Markov blanket in S of some  $x \in \mathcal{X}$  (called a *Markov boundary*) which is the smallest set  $\mathcal{B} \subseteq S$  such that  $f_x \perp f_S \mid f_B$ .

<span id="page-26-2"></span>**Lemma C.16** (Existence of an approximate Markov boundary). For any  $\epsilon > 0$ , let k be the smallest integer satisfying

<span id="page-26-4"></span><span id="page-26-1"></span>
$$\frac{\gamma_k}{k} \le \frac{\epsilon \lambda_{\min}(\mathbf{K}_{SS})}{2|S|\sigma^2 \tilde{\sigma}^2}.$$
 (15)

Then, for any  $n \ge 0$  and  $x \in \mathcal{X}$ , there exists an  $\epsilon$ -approximate Markov boundary  $B_{n,\epsilon}(x)$  of x in  $\mathcal{S}$  with size at most k.

Lemma C.16 shows that for any  $\epsilon > 0$  there exists a universal constant  $b_{\epsilon}$  (with respect to n and x) such that

$$|B_{n,\epsilon}(\boldsymbol{x})| \le b_{\epsilon} \qquad \forall n \ge 0, \boldsymbol{x} \in \mathcal{X}.$$
 (16)

<span id="page-26-3"></span>We defer the proof of Lemma C.16 to Appendix C.6.3 where we also provide an algorithm to compute  $B_{n,\epsilon}(x)$ .

**Lemma C.17.** For any  $\epsilon > 0$ ,  $n \geq 0$ , and  $x \in \mathcal{X}$ ,

$$\sigma_n^2(\boldsymbol{x}) \le 2\sigma^2 \cdot \mathrm{I}(f_{\boldsymbol{x}}; \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})} \mid \mathcal{D}_n) + \eta_{\mathcal{S}}^2(\boldsymbol{x}) + \epsilon$$
(17)

where  $B_{n,\epsilon}(x)$  is an  $\epsilon$ -approximate Markov boundary of x in S.

Proof. We have

$$\sigma_{n}^{2}(\boldsymbol{x}) = \operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \eta_{\mathcal{S}}^{2}(\boldsymbol{x})$$

$$\leq \operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})}, \mathcal{D}_{n}] + \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \epsilon$$

$$\stackrel{(ii)}{\leq} \sigma^{2} \log \left( \frac{\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}]}{\operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})}, \mathcal{D}_{n}]} \right) + \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \epsilon$$

$$= 2\sigma^{2} \cdot \operatorname{I}(f_{\boldsymbol{x}}; \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})} \mid \mathcal{D}_{n}) + \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \epsilon$$

where (i) follows from the defining property of an  $\epsilon$ -approximate Markov boundary (cf. Equation (14)); and (ii) follows from Lemma C.38 and using monotonicity of variance.

<span id="page-27-0"></span>**Lemma C.18.** For any  $\epsilon > 0$ ,  $n \geq 0$ , and  $x \in A$ ,

$$I(f_{\boldsymbol{x}}; \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})} \mid \mathcal{D}_n) \le \frac{b_{\epsilon}}{\bar{\kappa}_n(B_{n,\epsilon}(\boldsymbol{x}))} \Gamma_n$$
(18)

where  $B_{n,\epsilon}(x)$  is an  $\epsilon$ -approximate Markov boundary of x in S,  $|B_{n,\epsilon}(x)| \leq b_{\epsilon}$ , and where  $\bar{\kappa}_n(\cdot) \stackrel{\text{def}}{=} \bar{\kappa}(\cdot \mid x_{1:n})$  denotes the information ratio from Equation (8).

We remark that  $\bar{\kappa}_n(\cdot) > 0$  as is shown in Lemma C.7, and hence, the right-hand side of the inequality is well-defined.

*Proof.* We use the abbreviated notation  $B = B_{n,\epsilon}(x)$ . We have

$$\begin{split} \mathrm{I}(f_{\boldsymbol{x}};\boldsymbol{y}_{B}\mid\mathcal{D}_{n}) &\overset{(i)}{\leq} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}};\boldsymbol{y}_{B}\mid\mathcal{D}_{n}) \\ &\overset{(ii)}{\leq} \frac{1}{\bar{\kappa}_{n,b_{\epsilon}}} \sum_{\tilde{\boldsymbol{x}}\in B} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}};y_{\tilde{\boldsymbol{x}}}\mid\mathcal{D}_{n}) \\ &\overset{(iii)}{\leq} \frac{b_{\epsilon}}{\bar{\kappa}_{n,b_{\epsilon}}} \max_{\tilde{\boldsymbol{x}}\in B} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}};y_{\tilde{\boldsymbol{x}}}\mid\mathcal{D}_{n}) \\ &\overset{(iv)}{\leq} \frac{b_{\epsilon}}{\bar{\kappa}_{n,b_{\epsilon}}} \max_{\tilde{\boldsymbol{x}}\in\mathcal{S}} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}};y_{\tilde{\boldsymbol{x}}}\mid\mathcal{D}_{n}) \\ &\overset{(v)}{=} \frac{b_{\epsilon}}{\bar{\kappa}_{n,b_{\epsilon}}} \max_{\tilde{\boldsymbol{x}}\in\mathcal{S}} \mathrm{I}(\boldsymbol{f}_{\mathcal{A}};y_{\tilde{\boldsymbol{x}}}\mid\boldsymbol{y}_{1:n}) \\ &= \frac{b_{\epsilon}}{\bar{\kappa}_{n,b_{\epsilon}}} \Gamma_{n} \end{split}$$

where (i) follows from monotonicity of mutual information; (ii) follows from the definition of the information ratio  $\bar{\kappa}_{n,b_{\epsilon}}$  (cf. Equation (8)); (iii) follows from  $b \leq b_{\epsilon}$ ; (iv) follows from  $B \subseteq \mathcal{S}$ ; and (v) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations.

*Proof of Theorem 3.2 for* ITL . The case where  $x \in A \cap S$  is shown by Lemma C.14 with  $C = 2\tilde{\sigma}^2$ .

To prove the more general result, fix any  $x \in A$  and  $\epsilon > 0$ . By Lemma C.16, there exists an  $\epsilon$ -approximate Markov boundary  $B_{n,\epsilon}(x)$  of x in S such that  $|B_{n,\epsilon}(x)| \le b_{\epsilon}$ . We have

$$\sigma_{n}^{2}(\boldsymbol{x}) \overset{(i)}{\leq} 2\sigma^{2} \cdot \mathrm{I}(f_{\boldsymbol{x}}; \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})} \mid \mathcal{D}_{n}) + \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \epsilon$$

$$\overset{(ii)}{\leq} \frac{2\sigma^{2}b_{\epsilon}}{\bar{\kappa}_{n}(B_{n,\epsilon}(\boldsymbol{x}))} \Gamma_{n} + \eta_{\mathcal{S}}^{2}(\boldsymbol{x}) + \epsilon$$

where (i) follows from Lemma C.17; and (ii) follows from Lemma C.18.

Let  $\epsilon = c \frac{\gamma_{\sqrt{n}}}{\sqrt{n}}$  with  $c = 2|\mathcal{S}|\sigma^2\tilde{\sigma}^2/\lambda_{\min}(\textbf{\textit{K}}_{\mathcal{S}\mathcal{S}})$ . Then, by Equation (15),  $b_{\epsilon}$  can be bounded for instance by  $\sqrt{n}$ . Together with Theorem C.12 this implies for ITL that

$$\sigma_n^2(\mathbf{x}) \le \eta_{\mathcal{S}}^2(\mathbf{x}) + 2\sigma^2 \sqrt{n} \, \Gamma_n + c\gamma_{\sqrt{n}} / \sqrt{n}$$
$$\le \eta_{\mathcal{S}}^2(\mathbf{x}) + c'\gamma_n / \sqrt{n}$$

for a constant c', e.g.,  $c' = 2\sigma^2 + c$ .

#### C.6.2 Convergence outside S for VTL

Proof of Theorem 3.2 for VTL. Analogously to Lemma C.17, we have

$$\sigma_n^2(\boldsymbol{x}) = \operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_n] - \eta_{\mathcal{S}}^2(\boldsymbol{x}) + \eta_{\mathcal{S}}^2(\boldsymbol{x})$$

$$\stackrel{(i)}{\leq} \operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_n] - \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})}, \mathcal{D}_n] + \eta_{\mathcal{S}}^2(\boldsymbol{x}) + \epsilon$$

where (i) follows from the defining property of an  $\epsilon$ -approximate Markov boundary (cf. Equation (14)). Further, we have

$$\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B_{n,\epsilon}(\boldsymbol{x})}, \mathcal{D}_{n}] \\
\leq \sum_{\tilde{\boldsymbol{x}} \in B_{n,\epsilon}(\boldsymbol{x})} (\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}] - \operatorname{Var}[f_{\boldsymbol{x}} \mid y_{\tilde{\boldsymbol{x}}}, \mathcal{D}_{n}]) \\
\leq \sum_{\tilde{\boldsymbol{x}} \in B_{n,\epsilon}(\boldsymbol{x})} (\operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{1:n}] - \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid y_{\tilde{\boldsymbol{x}}}, \boldsymbol{y}_{1:n}]) \\
\leq \sum_{\tilde{\boldsymbol{x}} \in B_{n,\epsilon}(\boldsymbol{x})} (\operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \boldsymbol{y}_{1:n}] - \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid y_{\tilde{\boldsymbol{x}}}, \boldsymbol{y}_{1:n}]) \\
\leq b_{\epsilon} \Gamma_{n}$$

where (i) follows from the submodularity of  $\psi_{\mathcal{A}}$ ; (ii) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; and (iii) follows from the definition of  $\Gamma_n$  and Lemma C.16.

The remainder of the proof is analogous to the result for ITL, using Theorem C.13 to bound  $\Gamma_n$ .  $\square$ 

#### <span id="page-28-0"></span>C.6.3 Existence of an Approximate Markov Boundary

We now derive Lemma C.16 which shows the existence of an approximate Markov boundary of x in S.

<span id="page-28-2"></span>**Lemma C.19.** For any  $S \subseteq \mathcal{S}$  and  $k \ge 0$ , there exists  $B \subseteq S$  with |B| = k such that for all  $x' \in S$ ,

$$\operatorname{Var}[f_{x'} \mid y_B] \le 2\tilde{\sigma}^2 \frac{\gamma_k}{k}. \tag{19}$$

Proof. We choose  $B\subseteq S$  greedily using the acquisition function

$$\tilde{\boldsymbol{x}}_k \stackrel{\text{def}}{=} \argmax_{\tilde{\boldsymbol{x}} \in S} \mathrm{I}(\boldsymbol{f}_{\!S}; y_{\tilde{\boldsymbol{x}}} \mid \boldsymbol{y}_{\!B_{k-1}})$$

where  $B_k = \tilde{x}_{1:k}$ . Note that this is the "undirected" special case of ITL, and hence, we have

$$\operatorname{Var}\left[f_{\boldsymbol{x'}} \mid \boldsymbol{y}_{B_k}\right] \stackrel{(i)}{\leq} 2\tilde{\sigma}^2 \Gamma_k \\ \stackrel{(ii)}{\leq} 2\tilde{\sigma}^2 \frac{\gamma_k}{k}$$

where (i) is due to Lemma C.14; and (ii) is due to Theorem C.12 and  $\alpha_{S,S}(k) \leq 1$ .

<span id="page-28-3"></span>**Lemma C.20.** Given any  $\epsilon > 0$  and  $B \subseteq S \subseteq \mathcal{S}$  with  $|S| < \infty$ , such that for any  $x' \in S$ ,

<span id="page-28-1"></span>
$$\operatorname{Var}[f_{\boldsymbol{x'}} \mid \boldsymbol{y}_B] \le \frac{\epsilon \lambda_{\min}(\boldsymbol{K}_{SS})}{|S|\sigma^2}.$$
 (20)

Then for any  $x \in \mathcal{X}$ ,

$$Var[f_{\boldsymbol{x}} \mid \boldsymbol{y}_B] \le Var[f_{\boldsymbol{x}} \mid \boldsymbol{f}_S] + \epsilon. \tag{21}$$

*Proof.* We will denote the right-hand side of Equation (20) by  $\epsilon'$ . We have

$$\begin{split} & \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B}] \\ & \stackrel{(i)}{=} \mathbb{E}_{\boldsymbol{f}_{S}}[\operatorname{Var}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}, \boldsymbol{y}_{B}] \mid \boldsymbol{y}_{B}] \\ & + \operatorname{Var}_{\boldsymbol{f}_{S}}[\mathbb{E}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}, \boldsymbol{y}_{B}] \mid \boldsymbol{y}_{B}] \\ & \stackrel{(ii)}{=} \operatorname{Var}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}, \boldsymbol{y}_{B}] + \operatorname{Var}_{\boldsymbol{f}_{S}}[\mathbb{E}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}, \boldsymbol{y}_{B}] \mid \boldsymbol{y}_{B}] \\ & \stackrel{(iii)}{=} \underbrace{\operatorname{Var}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}]}_{\text{irreducible uncertainty}} + \underbrace{\operatorname{Var}_{\boldsymbol{f}_{S}}[\mathbb{E}_{f_{\boldsymbol{x}}}[f_{\boldsymbol{x}} \mid \boldsymbol{f}_{S}] \mid \boldsymbol{y}_{B}]}_{\text{reducible (epistemic) uncertainty}} \end{split}$$

where (i) follows from the law of total variance; (ii) uses that the conditional variance of a Gaussian depends only on the location of observations and not on their value; and (iii) follows from  $f_x \perp y_B \mid f_S$  since  $B \subseteq S$ . It remains to bound the reducible uncertainty.

Let  $h_x : \mathbb{R}^d \to \mathbb{R}$ ,  $f_S \mapsto \mathbb{E}[f_x \mid f_S]$  where we write  $d \stackrel{\text{def}}{=} |S|$ . Using the formula for the GP posterior mean, we have

$$h_{\boldsymbol{x}}(\boldsymbol{f}_{\!S}) = \mathbb{E}[f_{\boldsymbol{x}}] + \boldsymbol{z}^{\top}(\boldsymbol{f}_{\!S} - \mathbb{E}[\boldsymbol{f}_{\!S}])$$

where  $z \stackrel{\text{def}}{=} K_{SS}^{-1} K_{Sx}$ . Because h is a linear function in  $f_S$  we have for the reducible uncertainty that

$$\begin{aligned} \operatorname{Var}_{\boldsymbol{f}_{\!S}}[h_{\boldsymbol{x}}(\boldsymbol{f}_{\!S}) \mid \boldsymbol{y}_{\!B}] &= \boldsymbol{z}^{\top} \operatorname{Var}[\boldsymbol{f}_{\!S} \mid \boldsymbol{y}_{\!B}] \boldsymbol{z} \\ &\stackrel{(i)}{\leq} d \cdot \boldsymbol{z}^{\top} \operatorname{diag} \operatorname{Var}[\boldsymbol{f}_{\!S} \mid \boldsymbol{y}_{\!B}] \boldsymbol{z} \\ &\stackrel{(ii)}{\leq} \epsilon' d \; \boldsymbol{z}^{\top} \boldsymbol{z} \\ &= \epsilon' d \; \boldsymbol{K}_{\boldsymbol{x}S} \boldsymbol{K}_{SS}^{-1} \boldsymbol{K}_{SS}^{-1} \boldsymbol{K}_{Sx} \\ &\stackrel{\leq}{\leq} \frac{\epsilon' d}{\lambda_{\min}(\boldsymbol{K}_{SS})} \boldsymbol{K}_{\boldsymbol{x}S} \boldsymbol{K}_{SS}^{-1} \boldsymbol{K}_{Sx} \end{aligned}$$

where (i) follows from Lemma C.37; (ii) follows from the assumption that  $\mathrm{Var}[f_{x'} \mid y_B] \leq \epsilon'$  for all  $x' \in S$ ; and (iii) follows from

$$K_{\boldsymbol{x}S}K_{SS}^{-1}K_{S\boldsymbol{x}} \leq K_{\boldsymbol{x}\boldsymbol{x}} = \sigma^2$$

since 
$$K_{xx} - K_{xS}K_{SS}^{-1}K_{Sx} \ge 0$$
.

Proof of Lemma C.16. Let  $B\subseteq\mathcal{S}$  be the set of size k generated by Lemma C.19 to satisfy  $\mathrm{Var}[f_{\boldsymbol{x'}}\mid\boldsymbol{y}_B]\leq 2\tilde{\sigma}^2\gamma_k/k$  for all  $\boldsymbol{x'}\in\mathcal{S}$ . We have for any  $\boldsymbol{x}\in\mathcal{X}$ ,

$$\operatorname{Var}[f_{\boldsymbol{x}} \mid \mathcal{D}_{n}, \boldsymbol{y}_{B}] \stackrel{(i)}{\leq} \operatorname{Var}[f_{\boldsymbol{x}} \mid \boldsymbol{y}_{B}]$$

$$\stackrel{(ii)}{\leq} \operatorname{Var}[f_{\boldsymbol{x}} \mid f_{\mathcal{S}}] + \epsilon$$

where (i) follows from monotonicity of variance; and (ii) follows from Lemma C.20; using  $|S| < \infty$  and the condition on k.

We remark that Lemma C.19 provides an algorithm (just "undirected" ITL!) to compute an approximate Markov boundary, and the set B returned by this algorithm is a valid approximate Markov boundary for all  $x \in \mathcal{X}$ . One can simply swap-in ITL with target space  $\{x\}$  for "undirected" ITL to obtain tighter (but instance-dependent) bounds on the size of the approximate Markov boundary.

# C.6.4 Generalization to Continuous S for Finite Dimensional RKHSs

<span id="page-29-0"></span>In this subsection we generalize Theorem 3.2 to continuous sample spaces S. We will make the following assumption:

**Assumption C.21.** The RKHS of the kernel k is finite dimensional. In other words, the kernel k can be expressed as  $k(\boldsymbol{x}, \boldsymbol{x}') = \phi(\boldsymbol{x})^{\top} \phi(\boldsymbol{x}')$  for some feature map  $\phi : \mathcal{X} \to \mathbb{R}^d$  with  $d < \infty$ .

In the following, we will denote the design matrix of the sample space  $\mathcal{S}$  by  $\Phi \stackrel{\mathrm{def}}{=} [\phi(x) : x \in \mathcal{S}]^{\top} \in \mathbb{R}^{|\mathcal{S}| \times d}$ , and we denote by  $\Pi_{\Phi}$  its orthogonal projection onto the orthogonal complement of the span of  $\Phi$ . In particular, it holds that

- 1.  $\Pi_{\Phi}v = 0$  for all  $v \in \operatorname{span}\Phi$ , and
- 2.  $\Pi_{\Phi} v = v$  for all  $v \in (\operatorname{span} \Phi)^{\perp}$ .

Especially,  $v \in \ker \Pi_{\Phi}$  if and only if  $v \in \operatorname{span} \Phi$ . This projection can be computed as follows:

<span id="page-30-1"></span>Lemma C.22. It holds that

$$\mathbf{\Pi}_{\mathbf{\Phi}} = \mathbf{I} - \mathbf{\Phi}^{\top} (\mathbf{\Phi} \mathbf{\Phi}^{\top})^{-1} \mathbf{\Phi}. \tag{22}$$

*Proof.*  $\Phi^{\top}(\Phi\Phi^{\top})^{-1}\Phi$  is the orthogonal projection onto the span of  $\Phi$  (see, e.g., Strang, 2016, page 211).

<span id="page-30-2"></span>**Lemma C.23.** Under Assumption C.21, the irreducible uncertainty  $\eta_S^2(x)$  of  $x \in \mathcal{X}$  is

$$\eta_{\mathcal{S}}^{2}(\boldsymbol{x}) = \left\| \phi(\boldsymbol{x}) \right\|_{\boldsymbol{\Pi}_{\Phi}}^{2} \tag{23}$$

where  $\|v\|_A = \sqrt{v^\top A v}$  denotes the Mahalanobis distance.

*Proof.* This is an immediate consequence of the formula for the conditional variance of multivariate Gaussians (cf. Appendix B.2), applied to the linear kernel.  $\Box$ 

Lemmas C.22 and C.23 imply that  $\eta_{\mathcal{S}}^2(\boldsymbol{x}^\parallel) = 0$  for all  $\boldsymbol{x}^\parallel \in \mathcal{X}$  with  $\phi(\boldsymbol{x}^\parallel) \in \operatorname{span} \Phi$ . That is, the irreducible uncertainty is zero for points in the span of the sample space. In contrast, for points  $\boldsymbol{x}^\perp$  with  $\phi(\boldsymbol{x}^\perp) \in (\operatorname{span} \Phi)^\perp$ , the irreducible uncertainty equals the initial uncertainty:  $\eta_{\mathcal{S}}^2(\boldsymbol{x}^\perp) = \sigma_0^2(\boldsymbol{x}^\perp)$ . The irreducible uncertainty of any other point  $\boldsymbol{x}$  can be computed by simple decomposition of  $\phi(\boldsymbol{x})$  into parallel and orthogonal components.

Assuming that Assumption C.21 holds and given any (non-finite)  $\mathcal{S} \subseteq \mathcal{X}$ , there exists a basis  $\Omega_{\mathcal{S}} \subseteq \mathcal{X}$  in the space of embeddings  $\phi(\cdot)$  such that  $\operatorname{span} \mathcal{S} = \operatorname{span} \Omega_{\mathcal{S}}$  and  $|\Omega_{\mathcal{S}}| \leq d$ . The generalized existence of an approximate Markov boundary for continuous domains can then be shown analogously to Lemma C.16:

**Lemma C.24** (Existence of an approximate Markov boundary for a continuous domain). Let S be any (continuous) subset of X and let Assumption C.21 hold with  $d < \infty$ . Further, for any  $\epsilon > 0$ , let k be the smallest integer satisfying

$$\frac{\gamma_k}{k} \le \frac{\epsilon \lambda_{\min}(\mathbf{K}_{\Omega_{\mathcal{S}}\Omega_{\mathcal{S}}})}{2d\sigma^2 \tilde{\sigma}^2}.$$
 (24)

Then, for any  $n \geq 0$  and  $\mathbf{x} \in \mathcal{X}$ , there exists an  $\epsilon$ -approximate Markov boundary  $B_{n,\epsilon}(\mathbf{x})$  of  $\mathbf{x}$  in  $\mathcal{S}$  with size at most k.

*Proof sketch.* The proof follows analogously to Lemma C.16 by conditioning on the finite set  $\Omega_S$  as opposed to S.

#### <span id="page-30-0"></span>C.7 Proof of Theorem 3.3

We first formalize the assumptions of Theorem 3.3:

<span id="page-30-4"></span><span id="page-30-3"></span>**Assumption C.25** (Regularity of  $f^*$ ). We assume that  $f^*$  is in a reproducing kernel Hilbert space  $\mathcal{H}_k(\mathcal{X})$  associated with a kernel k and has bounded norm, that is,  $\|f^*\|_k \leq B$  for some finite  $B \in \mathbb{R}$ . **Assumption C.26** (Sub-Gaussian noise). We further assume that each  $\varepsilon_n$  from the noise sequence  $\{\varepsilon_n\}_{n=1}^\infty$  is conditionally zero-mean  $\rho(x_n)$ -sub-Gaussian with known constants  $\rho(x) > 0$  for all  $x \in \mathcal{X}$ . Concretely,

$$\forall n \geq 1, \lambda \in \mathbb{R} : \quad \mathbb{E}\left[e^{\lambda \epsilon_n} \mid \mathcal{D}_{n-1}\right] \leq \exp\left(\frac{\lambda^2 \rho^2(\boldsymbol{x}_n)}{2}\right)$$

where  $\mathcal{D}_{n-1}$  corresponds to the  $\sigma$ -algebra generated by the random variables  $\{x_i, \epsilon_i\}_{i=1}^{n-1}$  and  $x_n$ .

We make use of the following foundational result, showing that under the above two assumptions the (misspecified) Gaussian process model from Section 3.1 is an all-time well-calibrated model of  $f^*$ :

<span id="page-31-3"></span>**Lemma C.27** (Well-calibrated confidence intervals; Abbasi-Yadkori (2013); Chowdhury & Gopalan (2017)). *Pick*  $\delta \in (0,1)$  *and let Assumptions C.25 and C.26 hold. Let* 

$$\beta_n(\delta) = \|f^*\|_k + \rho \sqrt{2(\gamma_n + 1 + \log(1/\delta))}$$

where  $\rho = \max_{x \in \mathcal{X}} \rho(x)$ . <sup>10</sup> Then, for all  $x \in \mathcal{X}$  and  $n \ge 0$  jointly with probability at least  $1 - \delta$ ,

$$|f^{\star}(\boldsymbol{x}) - \mu_n(\boldsymbol{x})| \leq \beta_n(\delta) \cdot \sigma_n(\boldsymbol{x})$$

where  $\mu_n(\mathbf{x})$  and  $\sigma_n^2(\mathbf{x})$  are mean and variance (as defined in Appendix B.2) of the GP posterior of  $f(\mathbf{x})$  conditional on the observations  $\mathcal{D}_n$ , pretending that  $\varepsilon_i$  is Gaussian with variance  $\rho^2(\mathbf{x}_i)$ .

The proof of Theorem 3.3 is a straightforward application of Lemma C.27 and Theorem 3.2:

*Proof of Theorem 3.3.* By Theorem 3.2, we have that for all  $x \in A$ ,

$$\sigma_n(\boldsymbol{x}) \leq \sqrt{\eta_{\mathcal{S}}^2(\boldsymbol{x}) + \nu_{\mathcal{A},\mathcal{S}}^2(n)} \leq \eta_{\mathcal{S}}(\boldsymbol{x}) + \nu_{\mathcal{A},\mathcal{S}}(n).$$

The result then follows by application of Lemma C.27.

#### <span id="page-31-0"></span>C.8 Proof of Theorem 5.1

In this section, we derive our main result on Safe BO. In Appendix C.8.1, we give the definition of the reachable safe set  $\mathcal{R}$  and derive the conditions under which convergence to the reachable safe set is guaranteed. Then, in Appendix C.8.2, we prove Theorem 5.1.

**Notation** In the agnostic setting from Section 3.2 (i.e., under Assumptions C.25 and C.26), Lemma C.27 provides us with the following  $(1 - \delta)$ -confidence intervals (CIs)

$$C_n(\mathbf{x}) \stackrel{\text{def}}{=} C_{n-1}(\mathbf{x}) \cap [\mu_n(\mathbf{x}) \pm \beta_n(\delta) \cdot \sigma_n(\mathbf{x})]$$
 (25)

where  $C_{-1}(x) = \mathbb{R}$ . We write  $u_n(x) \stackrel{\text{def}}{=} \max C_n(x)$ ,  $l_n(x) \stackrel{\text{def}}{=} \min C_n(x)$ , and  $w_n(x) \stackrel{\text{def}}{=} u_n(x) - l_n(x)$  for its upper bound, lower bound, and width, respectively.

We learn separate statistical models f and  $\{g_1,\ldots,g_q\}$  for the ground truth objective  $f^\star$  and ground truth constraints  $\{g_1^\star,\ldots,g_q^\star\}$ . We write  $\mathcal{I} \stackrel{\mathrm{def}}{=} \{f,1,\ldots,q\}$  and collect the constraints in  $\mathcal{I}_s \stackrel{\mathrm{def}}{=} \{1,\ldots,q\}$ . Without loss of generality, we assume that the confidence intervals include the ground truths with probability at least  $1-\delta$  jointly for all  $i\in\mathcal{I}$ . For  $i\in\mathcal{I}$ , denote by  $u_{n,i},l_{n,i},w_{n,i},\eta_i,\beta_{n,i}$  the respective quantities. In the following, we do not explicitly denote the dependence of  $\beta_n$  on  $\delta$ .

To improve clarity, we will refer to the set of potential maximizers defined in Equation (5) as  $\mathcal{M}_n$  and denote by  $\mathcal{A}_n$  an arbitrary target space.

We point out the following corollary:

<span id="page-31-6"></span>**Corollary C.28** (Safety). With high probability, jointly for any  $n \geq 0$  and any  $i \in \mathcal{I}_s$ ,

$$\forall \boldsymbol{x} \in \mathcal{S}_n : g_i^{\star}(\boldsymbol{x}) \ge 0. \tag{26}$$

### <span id="page-31-4"></span>C.8.1 Convergence to Reachable Safe Set

<span id="page-31-1"></span>**Definition C.29** (Reachable safe set). Given any pessimistic safe set  $S \subseteq \mathcal{X}$  and any  $\epsilon \geq 0$  and  $\beta \geq 0$ , we define the *reachable safe set* up to  $(\epsilon, \beta)$ -slack and its closure as

$$\mathcal{R}_{\epsilon,\beta}(\mathcal{S}) \stackrel{\text{def}}{=} \mathcal{S} \cup \{\boldsymbol{x} \in \mathcal{X} \setminus \mathcal{S} \mid g_i^{\star}(\boldsymbol{x}) - \beta(\eta_i(\boldsymbol{x}; \mathcal{S}) + \epsilon) \ge 0 \text{ for all } i \in \mathcal{I}_s\}$$
$$\bar{\mathcal{R}}_{\epsilon,\beta}(\mathcal{S}) \stackrel{\text{def}}{=} \lim_{n \to \infty} (\mathcal{R}_{\epsilon,\beta})^n(\mathcal{S})$$

where  $(\mathcal{R}_{\epsilon,\beta})^n$  denotes the *n*-th composition of  $\mathcal{R}_{\epsilon,\beta}$  with itself.

<span id="page-31-2"></span> $<sup>^{10}\</sup>beta_n(\delta)$  can be tightened adaptively (Emmenegger et al., 2023).

<span id="page-31-5"></span><sup>&</sup>lt;sup>11</sup>This can be achieved by taking a union bound and rescaling  $\delta$ .

Remark C.30. Convergence of the safe set to the closure of the reachability operator can only be guaranteed for finite safe sets ( $|\mathcal{S}^{\star}| < \infty$ ). The following proofs readily generalize to continuous domains by considering convergence within the k-th composition of the reachability operator with itself for some  $k < \infty$ . In this case the sample complexity grows with k rather than  $|\mathcal{S}^{\star}|$ . The only required modification is to lift the assumption of Theorem C.12 that information is gained only while safe sets remain constant (i.e.,  $\mathcal{S}_{i+1} = \mathcal{S}_i$  for all i). This assumption is straightforward to lift since for any  $n \geq 0$  and  $T \geq 1$ ,

$$\max_{\boldsymbol{x} \in \mathcal{S}_n} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid \boldsymbol{x}_{1:n+T}) \leq \frac{1}{T} \sum_{t=1}^{T} \max_{\boldsymbol{x} \in \mathcal{S}_n} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid \boldsymbol{x}_{1:n+t}) \leq \frac{1}{T} \sum_{t=1}^{T} \max_{\boldsymbol{x} \in \mathcal{S}_{n+t}} \Delta_{\mathcal{A}}(\boldsymbol{x} \mid \boldsymbol{x}_{1:n+t}) \leq \frac{\gamma_T}{T},$$

using submodularity for the first inequality and the monotonicity of the safe set for the second inequality. In particular, this shows that one continues learning about points in the original safe set — even as the safe set grows.

We denote by  $\mathcal{S}_0$  the initial pessimistic safe set induced by the (prior) statistical model g (cf. Section 5) and write  $\bar{\mathcal{R}}_{\epsilon,\beta} \stackrel{\text{def}}{=} \bar{\mathcal{R}}_{\epsilon,\beta}(\mathcal{S}_0)$ .

**Lemma C.31** (Properties of the reachable safe set). *For all*  $S, S' \subseteq X$ ,  $\epsilon \geq 0$ , *and*  $\beta \geq 0$ :

(i) 
$$S' \subseteq S \implies \mathcal{R}_{\epsilon,\beta}(S') \subseteq \mathcal{R}_{\epsilon,\beta}(S)$$
,

<span id="page-32-1"></span>(ii) 
$$\mathcal{R}_{\epsilon,\beta}(\mathcal{S}) \subseteq \mathcal{S} \implies \bar{\mathcal{R}}_{\epsilon,\beta}(\mathcal{S}) \subseteq \mathcal{S}$$
, and

(iii) 
$$\mathcal{R}_{0,0}(\emptyset) = \bar{\mathcal{R}}_{0,0} = \mathcal{S}^*$$
.

Proof (adapted from lemma 7.1 of Berkenkamp et al. (2021)).

- 1. Let  $x \in \mathcal{R}_{\epsilon,\beta}(\mathcal{S}')$ . If  $x \in \mathcal{S}$  then  $x \in \mathcal{R}_{\epsilon,\beta}(\mathcal{S})$ , so let  $x \notin \mathcal{S}$ . Then, by definition, for all  $i \in \mathcal{I}_s$ ,  $f_i^{\star}(x) \beta \eta_i(x; \mathcal{S}') \epsilon \geq 0$ . By the monotonicity of variance,  $\eta_i(x; \mathcal{S}') \geq \eta_i(x; \mathcal{S})$  for all  $i \in \mathcal{I}$ , and hence  $f_i^{\star}(x) \beta \eta_i(x; \mathcal{S}) \epsilon \geq 0$  for all  $i \in \mathcal{I}_s$ . It follows that  $x \in \mathcal{R}_{\epsilon,\beta}(\mathcal{S})$ .
- 2. By the monotonicity of variance,  $\eta_i(\boldsymbol{x}; \mathcal{R}_{\epsilon,\beta}(\mathcal{S})) \geq \eta_i(\boldsymbol{x}; \mathcal{S})$  for all  $\boldsymbol{x} \in \mathcal{X}$  and  $i \in \mathcal{I}$ . Thus, by definition of the safe region, we have that  $\mathcal{R}_{\epsilon,\beta}(\mathcal{R}_{\epsilon,\beta}(\mathcal{S})) \subseteq \mathcal{S}$ . The result follows by taking the limit.
- 3. The result follows directly from the definition of the true safe set  $S^*$  (cf. Equation (4)).  $\square$

Clearly, we cannot expand the safe set beyond  $\bar{\mathcal{R}}_{0,0}$ . The following is our main intermediate result, showing that either we expand the safe set at some point or the uncertainty converges to the irreducible uncertainty.

<span id="page-32-0"></span>**Lemma C.32.** Given any  $n_0 \ge 0$ ,  $\epsilon > 0$ , let n' be the smallest integer such that  $\nu_{n',\tilde{\epsilon}^2} \le \tilde{\epsilon}$  where  $\tilde{\epsilon} = \epsilon/2$ . Let  $\beta_{n_0+n'} = \max_{i \in \mathcal{I}_s} \beta_{n_0+n',i}$ . Assume that the sequence of target spaces is monotonically decreasing, i.e.,  $\mathcal{A}_{n+1} \subseteq \mathcal{A}_n$ . Then, we have with high probability (at least) one of

$$\begin{pmatrix} \forall \boldsymbol{x} \in \mathcal{A}_{n_0+n'}, \ \forall i \in \mathcal{I} : \\
w_{n_0+n',i}(\boldsymbol{x}) \leq \beta_{n_0+n'}[\eta_i(\boldsymbol{x}; \mathcal{S}_{n_0+n'}) + \epsilon] \\
\text{and} \quad \mathcal{A}_{n_0+n'} \cap \mathcal{R}_{\epsilon,\beta_{n_0+n'}}(\mathcal{S}_{n_0+n'}) \subseteq \mathcal{S}_{n_0+n'} \end{pmatrix}$$

or  $|S_{n_0+n'+1}| > |S_{n_0}|$ .

*Proof.* Suppose that  $|S_{n_0+n'+1}| = |S_{n_0}|$ . Then, by Theorem 3.3 (using that the sequence of target spaces is monotonically decreasing), for any  $x \in A_{n_0+n'}$  and  $i \in \mathcal{I}$ ,

$$w_{n_0+n',i}(\boldsymbol{x}) \leq \beta_{n_0+n'} [\eta_i(\boldsymbol{x}; \mathcal{S}_{n_0+n'}) + \epsilon].$$

As  $S_{n_0+n'+1} = S_{n_0+n'}$  we have for all  $x \in A_{n_0+n'} \setminus S_{n_0+n'}$  and  $i \in \mathcal{I}_s$ , with high probability that

$$0 > l_{n_0+n',i}(\boldsymbol{x}) \ge g_i^{\star}(\boldsymbol{x}) - w_{n_0+n',i}(\boldsymbol{x}) \ge g_i^{\star}(\boldsymbol{x}) - \beta_{n_0+n'}[\eta_i(\boldsymbol{x}; \mathcal{S}_{n_0+n'}) + \epsilon].$$

It follows that  $A_{n_0+n'} \cap \mathcal{R}_{\epsilon,\beta_{n_0+n'}}(\mathcal{S}_{n_0+n'}) \subseteq \mathcal{S}_{n_0+n'}$ .

To gather more intuition about the above lemma, consider the target space

<span id="page-33-4"></span>
$$\mathcal{E}_n \stackrel{\mathrm{def}}{=} \widehat{\mathcal{S}}_n \setminus \mathcal{S}_n. \tag{27}$$

We call  $\mathcal{E}_n$  the *potential expanders* since it contains all points which might be safe, but are not yet known to be safe. Under this target space, the above lemma simplifies slightly:

<span id="page-33-2"></span>**Lemma C.33.** For any  $n \geq 0$  and  $\epsilon, \beta \geq 0$ , if  $\mathcal{E}_n \subseteq \mathcal{A}_n$  then with high probability,

$$S_n \cup (A_n \cap R_{\epsilon,\beta}(S_n)) = R_{\epsilon,\beta}(S_n).$$

*Proof.* With high probability,  $\mathcal{R}_{\epsilon,\beta}(\mathcal{S}_n) \subseteq \widehat{\mathcal{S}}_n = \mathcal{S}_n \cup \mathcal{E}_n$ . The lemma is a direct consequence.  $\square$ 

The above lemmas can be combined to yield our main result of this subsection, establishing the convergence of ITL to the reachable safe set.

<span id="page-33-0"></span>**Theorem C.34** (Convergence to reachable safe set). For any  $\epsilon > 0$ , let n' be the smallest integer satisfying the condition of Lemma C.32, and define  $n^* \stackrel{\text{def}}{=} (|\mathcal{S}^*| + 1)n'$ . Let  $\bar{\beta}_{n^*} \geq \beta_{n,i}$  for all  $n \leq n^*, i \in \mathcal{I}_s$ . Assume that the sequence of target spaces is monotonically decreasing, i.e.,  $\mathcal{A}_{n+1} \subseteq \mathcal{A}_n$ . Then, the following inequalities hold jointly with probability at least  $1 - \delta$ :

(i) 
$$\forall n \geq 0, \ \forall i \in \mathcal{I}_s : g_i^{\star}(\boldsymbol{x}_n) \geq 0$$
,

safety

<span id="page-33-1"></span>

<span id="page-33-3"></span>(ii) 
$$A_{n^{\star}} \cap \bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^{\star}}} \subseteq \mathcal{S}_{n^{\star}} \subseteq \bar{\mathcal{R}}_{0,0} = \mathcal{S}^{\star},$$

convergence to safe region

(iii) 
$$\forall \boldsymbol{x} \in \mathcal{A}_{n^*}, \ \forall i \in \mathcal{I} : w_{n^*,i}(\boldsymbol{x}) \leq \bar{\beta}_{n^*} \eta_i(\boldsymbol{x}; \bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}}) + \epsilon,$$

convergence of width

(iv) 
$$\forall \boldsymbol{x} \in \bar{\mathcal{R}}_{\epsilon, \bar{\beta}_{n^{\star}}}, \ \forall i \in \mathcal{I} : \eta_{i}(\boldsymbol{x}; \bar{\mathcal{R}}_{\epsilon, \bar{\beta}_{n^{\star}}}) = 0.$$

convergence of width within safe region

*Proof.* (i) is a direct consequence of Corollary C.28.  $S_{n^*} \subseteq S^*$  follows directly from the pessimistic safe set  $S_{n^*}$  from (ii) being a subset of the true safe set  $S^*$ . (iv) follows directly from the definition of irreducible uncertainty. Thus, it remains to establish  $A_{n^*} \cap \overline{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}} \subseteq S_{n^*}$  and (iii).

Recall that with high probability  $|S_n| \in [0, |S^*|]$  for all  $n \ge 0$ . Thus, the size of the pessimistic safe set can increase at most  $|S^*|$  many times. By Lemma C.32, using the assumption on n', the size of the pessimistic safe set increases at least once every n' iterations, or else:

$$\forall \boldsymbol{x} \in \mathcal{A}_{n_0+n'}, \ \forall i \in \mathcal{I} : w_{n_0+n',i}(\boldsymbol{x}) \leq \beta_{n_0+n'}[\eta_i(\boldsymbol{x}; \mathcal{S}_{n_0+n'}) + \epsilon]$$
and
$$\mathcal{A}_{n_0+n'} \cap \mathcal{R}_{\epsilon,\beta_{n_0+n'}}(\mathcal{S}_{n_0+n'}) \subseteq \mathcal{S}_{n_0+n'}.$$
(28)

Because the safe set can expand at most  $|\mathcal{S}^{\star}|$  many times, Equation (28) occurs eventually for some  $n_0 \leq |\mathcal{S}^{\star}| n'$ . In this case, since  $\bar{\beta}_{n^{\star}} \geq \beta_{n_0+n'}$  and  $\mathcal{A}_{n^{\star}} \subseteq \mathcal{A}_{n_0+n'}$  (as  $n_0+n' \leq n^{\star}$ ) we have that

$$\mathcal{A}_{n^{\star}} \cap \mathcal{R}_{\epsilon, \bar{\beta}_{n^{\star}}}(\mathcal{S}_{n_{0}+n'}) \subseteq \mathcal{A}_{n_{0}+n'} \cap \mathcal{R}_{\epsilon, \beta_{n_{0}+n'}}(\mathcal{S}_{n_{0}+n'})$$
$$\subseteq \mathcal{S}_{n_{0}+n'}.$$

By Lemma C.31 (ii), this implies

$$\mathcal{A}_{n^{\star}} \cap \bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^{\star}}} \subseteq \mathcal{S}_{n_0+n'} \subseteq \mathcal{S}_{n^{\star}}.$$

We emphasize that Theorem C.34 holds for arbitrary target spaces  $\mathcal{A}_n$ . If additionally,  $\mathcal{E}_n \subseteq \mathcal{A}_n$  for all  $n \geq 0$  then by Lemma C.33, Theorem C.34 (ii) strengthens to  $\bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^{\star}}} \subseteq \mathcal{S}_{n^{\star}}$ . Intuitively,  $\mathcal{E}_n \subseteq \mathcal{A}_n$  ensures that one aims to expand the safe set in *all* directions. Conversely, if  $\mathcal{E}_n \not\subseteq \mathcal{A}_n$  then one aims only to expand the safe set in the direction of  $\mathcal{A}_n$  (or not at all if  $\mathcal{A}_n \subseteq \mathcal{S}_n$ ).

**"Free" convergence guarantees in many applications** Theorem C.34 can be specialized to yield convergence guarantees in various settings by choosing an appropriate target space  $A_n$ . Straightforward application of Theorem C.34 (informally) requires that the sequence of target spaces is monotonically decreasing (i.e.,  $A_{n+1} \subseteq A_n$ ), and that each target space  $A_n$  is an "over-approximation" of the actual set of targeted points (such as the set of optimas in the Bayesian optimization setting). We discuss two such applications in the following.

- 1. Pure expansion: For example, for the target space  $\mathcal{E}_n$ , Theorem C.34 bounds the convergence of the safe set to the reachable safe set. In this case, the transductive active learning problem corresponds to the "pure expansion" setting, also addressed by the ISE baseline discussed in Section 5. The ISE baseline, however, does not establish convergence guarantees of the kind of Theorem C.34. Note that  $\mathcal{E}_n$  satisfies the (informal) requirements laid out previously, since it is monotonically decreasing by definition, and with high probability, any point  $x \in \mathcal{S}^*$  that is not in  $\mathcal{S}_n$  is contained within  $\mathcal{E}_n$ .
- 2. Level set estimation: Given any  $\tau \in \mathbb{R}$ , we denote the (safe)  $\tau$ -level set of  $f^*$  by  $\mathcal{L}^{\tau} \stackrel{\text{def}}{=} \{ x \in \mathcal{S}^* \mid f^*(x) = \tau \}$ . We define the potential level set as

$$\mathcal{L}_n^{\tau} \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \widehat{\mathcal{S}}_n \mid l_n^f(\boldsymbol{x}) \le \tau \le u_n^f(\boldsymbol{x}) \}. \tag{29}$$

That is,  $\mathcal{L}_n^{\tau}$  is the subset of the optimistic safe set  $\widehat{\mathcal{S}}_n$  where the  $\tau$ -level set of  $f^{\star}$  may be located. Analogously to the potential expanders, it is straightforward to show that  $\mathcal{L}_n^{\tau}$  over-approximates the true  $\tau$ -level set and is monotonically decreasing.

We remark that our guarantees from this section also apply to the standard ("unsafe") setting where  $S^* = S_0 = \mathcal{X}$ .

#### <span id="page-34-0"></span>C.8.2 Convergence to Safe Optimum

In this section, we specialize Theorem C.34 for the case that the target space contains the potential maximizers  $\mathcal{M}_n$  (cf. Equation (5)). It is straightforward to see that the sequence  $\mathcal{M}_n$  is monotonically decreasing (i.e.,  $\mathcal{M}_{n+1} \subseteq \mathcal{M}_n$ ). The following lemma shows that the potential maximizers overapproximate the set of safe maxima  $\mathcal{X}^* \stackrel{\text{def}}{=} \arg\max_{\boldsymbol{x} \in \mathcal{S}^*} f^*(\boldsymbol{x})$ .

**Lemma C.35** (Potential maximizers over-approximate safe maxima). For all  $n \ge 0$  and with probability at least  $1 - \delta$ ,

- (i)  $x \in \mathcal{X}^*$  implies  $x \in \mathcal{M}_n$  and
- (ii)  $x \notin \mathcal{M}_n$  implies  $x \notin \mathcal{X}^*$ .

*Proof.* If  $x \notin \mathcal{M}_n$  then

$$u_{n,f}(\boldsymbol{x}) < \max_{\boldsymbol{x'} \in \mathcal{S}_n} l_{n,f}(\boldsymbol{x'}) \le \max_{\boldsymbol{x'} \in \mathcal{S}^{\star}} l_{n,f}(\boldsymbol{x'})$$

where we used  $S_n \subseteq S^*$  with high probability, which directly implies with high probability that  $x \notin \mathcal{X}^*$ .

For the other direction, if  $x \in \mathcal{X}^*$  then

$$u_{n,f}(\boldsymbol{x}) \geq \max_{\boldsymbol{x'} \in \mathcal{S}^{\star}} l_{n,f}(\boldsymbol{x'}) \geq \max_{\boldsymbol{x'} \in \mathcal{S}_n} l_{n,f}(\boldsymbol{x'})$$

with high probability.

We denote the set of optimal actions which are safe up to  $(\epsilon,\beta)$ -slack by

$$\mathcal{X}^{*}_{\epsilon,\beta} \stackrel{\mathrm{def}}{=} \arg \max_{\boldsymbol{x} \in \bar{\mathcal{R}}_{\epsilon,\beta}} f^{\star}(\boldsymbol{x}),$$

and by  $f_{\epsilon,\beta}^*$  the maximum value attained by  $f^*$  at any of the points in  $\mathcal{X}_{\epsilon,\beta}^*$ . The regret can be expressed as

$$r_n(\bar{\mathcal{R}}_{\epsilon,\beta}) = f_{\epsilon,\beta}^* - f^*(\widehat{\boldsymbol{x}}_n)$$

The following theorem formalizes Theorem 5.1 and establishes convergence to the safe optimum.

**Theorem C.36** (Convergence to safe optimum). For any  $\epsilon > 0$ , let n' be the smallest integer satisfying the condition of Lemma C.32, and define  $n^* \stackrel{\text{def}}{=} (|\mathcal{S}^*| + 1)n'$ . Let  $\bar{\beta}_{n^*} \geq \beta_{n,i}$  for all  $n \leq n^*$ ,  $i \in \mathcal{I}_s$ . Then, the following inequalities hold jointly with probability at least  $1 - \delta$ :

(i) 
$$\forall n \geq 0, \ \forall i \in \mathcal{I}_s : g_i^{\star}(\boldsymbol{x}_n) \geq 0,$$

safety

(ii) 
$$\forall n \geq n^* : r_n(\bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}}) \leq \epsilon.$$

convergence to safe optimum

*Proof.* Fix any  $x^* \in \mathcal{X}^*_{\epsilon,\bar{\beta}_{n^*}} \subseteq \bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}}$ . Assume w.l.o.g. that  $x^* \in \mathcal{M}_{n^*}$ .<sup>12</sup> Then, with high probability,

$$f_{\epsilon,\bar{\beta}_{n^{\star}}}^{*} = f^{\star}(\boldsymbol{x}^{*}) \leq u_{n^{\star},f}(\boldsymbol{x}^{*})$$

$$= l_{n^{\star},f}(\boldsymbol{x}^{*}) + w_{n^{\star},f}(\boldsymbol{x}^{*})$$

$$\stackrel{(i)}{\leq} l_{n^{\star},f}(\widehat{\boldsymbol{x}}_{n^{\star}}) + w_{n^{\star},f}(\boldsymbol{x}^{*})$$

$$\leq f^{\star}(\widehat{\boldsymbol{x}}_{n^{\star}}) + w_{n^{\star},f}(\boldsymbol{x}^{*})$$

$$\stackrel{(ii)}{\leq} f^{\star}(\widehat{\boldsymbol{x}}_{n^{\star}}) + \epsilon$$

where (i) follows from the definition of  $\hat{x}_n$ ; and (ii) follows from Theorem C.34 and noting that  $x^* \in \mathcal{M}_{n^*} \cap \bar{\mathcal{R}}_{\epsilon,\bar{\mathcal{B}}_{s,*}}$ .

We have shown that  $f^*(\widehat{x}_{n^*}) \geq f^*_{\epsilon,\bar{\beta}_{n^*}} - \epsilon$ , which implies  $r_{n^*}(\bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}}) \leq \epsilon$ . Since the upper- and lower-confidence bounds are monotonically decreasing / increasing, respectively, we have that for all  $n \geq n^*$ ,  $r_n(\bar{\mathcal{R}}_{\epsilon,\bar{\beta}_{n^*}}) \leq \epsilon$ .

#### <span id="page-35-0"></span>C.9 Useful Facts and Inequalities

We denote by  $\prec$  the Loewner partial ordering of symmetric matrices.

<span id="page-35-2"></span>**Lemma C.37.** Let  $A \in \mathbb{R}^{n \times n}$  be a positive definite matrix with diagonal D. Then,  $A \leq nD$ .

*Proof.* Equivalently, one can show  $nD - A \succeq 0$ . We write  $A \stackrel{\text{def}}{=} D^{1/2} Q D^{1/2}$ , and thus,  $Q = D^{-1/2} A D^{-1/2}$  is a positive definite symmetric matrix with all diagonal elements equal to 1. It remains to show that

$$nD - A = D^{1/2}(nI - Q)D^{1/2} \succ 0.$$

Note that  $\sum_{i=1}^{n} \lambda_i(\mathbf{Q}) = \text{tr } \mathbf{Q} = n$ , and hence, all eigenvalues of  $\mathbf{Q}$  belong to (0, n).

<span id="page-35-1"></span>**Lemma C.38.** If  $a, b \in (0, M]$  for some M > 0 and  $b \ge a$  then

$$b - a \le M \cdot \log\left(\frac{b}{a}\right). \tag{30}$$

If additionally,  $a \ge M'$  for some M' > 0 then

$$b - a \ge M' \cdot \log\left(\frac{b}{a}\right). \tag{31}$$

*Proof.* Let  $f(x) \stackrel{\text{def}}{=} \log x$ . By the mean value theorem, there exists  $c \in (a,b)$  such that

$$\frac{1}{c} = f'(c) = \frac{f(b) - f(a)}{b - a} = \frac{\log b - \log a}{b - a} = \frac{\log(\frac{b}{a})}{b - a}.$$

Thus,

$$b - a = c \cdot \log\left(\frac{b}{a}\right) < M \cdot \log\left(\frac{b}{a}\right).$$

<span id="page-35-3"></span><sup>&</sup>lt;sup>12</sup>Otherwise, with high probability,  $f^*(\widehat{x}_{n^*}) > f^*_{\epsilon,\bar{\beta}_{n^*}}$ .

Under the additional condition that a ≥ M′ , we obtain

$$b - a = c \cdot \log\left(\frac{b}{a}\right) > M' \cdot \log\left(\frac{b}{a}\right).$$

# <span id="page-36-0"></span>D Interpretations & Approximations of Principle ([†](#page-2-3))

We give a brief overview of interpretations and approximations of ITL, as well as alternative decision rules adhering to the fundamental principle ([†](#page-2-3)).

The discussed interpretations of ([†](#page-2-3)) differ mainly in how they quantify the "uncertainty" about A. In the GP setting, this "uncertainty" is captured by the covariance matrix Σ of fA, and we consider two main ways of "scalarizing" Σ:

- 1. the total (marginal) variance tr Σ, and
- 2. the "generalized variance" |Σ|.

The generalized variance — which was originally suggested by [Wilks](#page-16-10) [\(1932\)](#page-16-10) as a generalization of variance to multiple dimensions — takes into account correlations. In contrast, the total variance discards all correlations between points in A.

All discussed decision rules following principle ([†](#page-2-3)) (i.e., ITL, VTL, MM-ITL) differ only in their weighting of the points in A, and they coincide when |A| = 1.

#### <span id="page-36-1"></span>D.1 Interpretations of ITL

We briefly discuss three interpretations of ITL.

Minimizing generalized variance In the GP setting, ITL can be equivalently characterized as minimizing generalized posterior variance:

$$\mathbf{x}_{n} = \underset{\mathbf{x} \in \mathcal{S}}{\operatorname{arg max}} \mathbf{I}(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_{n})$$

$$= \underset{\mathbf{x} \in \mathcal{S}}{\operatorname{arg max}} \frac{1}{2} \log \left( \frac{|\operatorname{Var}[\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}]|}{|\operatorname{Var}[\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\mathbf{x}}]|} \right)$$

$$= \underset{\mathbf{x} \in \mathcal{S}}{\operatorname{arg min}} |\operatorname{Var}[\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\mathbf{x}}]|. \tag{32}$$

Maximizing relevance and minimizing redundancy An alternative interpretation of ITL is

$$I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_n) = \underbrace{I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}})}_{\text{relevance}} - \underbrace{I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}}; \mathcal{D}_n)}_{\text{redundancy}}$$
(33)

where I(fA; yx; Dn) = I(fA; yx) − I(fA; y<sup>x</sup> | Dn) denotes the *multivariate information gain* (cf. Appendix [B\)](#page-19-0). In this way, ITL can be seen as maximizing observation relevance while minimizing observation redundancy. This interpretation is common in the literature on feature selection [\(Peng et al.,](#page-14-8) [2005;](#page-14-8) [Vergara & Estévez,](#page-15-15) [2014;](#page-15-15) [Beraha et al.,](#page-10-9) [2019\)](#page-10-9).

Steepest descent in measure spaces ITL can be seen as performing steepest descent in the space of probability measures over fA, with the KL divergence as metric:

$$I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_n) = \mathbb{E}_{y_{\mathbf{x}}}[KL(p(\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_n, y_{\mathbf{x}}) || p(\mathbf{f}_{\mathcal{A}} \mid \mathcal{D}_n))].$$

That is, ITL finds the observation yielding the "largest update" to the current density.

### <span id="page-36-2"></span>D.2 Interpretations of VTL

Quantifying the uncertainty about f<sup>A</sup> by the marginal variance of points in A rather than entropy (or generalized variance), the principle ([†](#page-2-3)) leads to VTL. Note that if |A| = 1, then VTL is equivalent to ITL. Unlike the similar, but more technical, TRUVAR algorithm proposed by [Bogunovic et al.](#page-11-1) [\(2016\)](#page-11-1), VTL does not require truncated variances, and hence, VTL can be applied to constrained settings (where A ̸⊆ S) as well.

Relationship to ITL Note that the ITL criterion in the GP setting can be expressed as

<span id="page-37-1"></span>
$$\boldsymbol{x}_n = \operatorname*{arg\,min}_{\boldsymbol{x} \in \mathcal{S}} \operatorname{tr} \log \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]$$
 (34)

where for a positive semi-definite matrix A with spectral decomposition  $A = V \Lambda V^{\top}$  we write  $\log A = V \log \Lambda V^{\top}$  for the logarithmic matrix function. To derive Equation (34) we use that  $\log |A| = \sum_i \log \lambda_i(A) = \operatorname{tr} \log A$ . Hence, ITL and VTL are identical up to a different weighting of the eigenvalues of the posterior covariance matrix.

Minimizing a bound to the approximation error Chowdhury & Gopalan (2017) (page 19) bound the approximation error  $|f^*(x) - \mu_n(x)|$  by

$$\underbrace{|\underline{k_t(x)}^\top (\underline{K_t + P_t})^{-1} \varepsilon_{1:t}|}_{\text{variance}} + \underbrace{|f^\star(x) - \underline{k_t(x)}^\top (\underline{K_t + P_t})^{-1} \underline{f_{1:t}}|}_{\text{bias}}$$

where  $k_t(x) \stackrel{\text{def}}{=} K_{xx_{1:t}}$ ,  $K_t \stackrel{\text{def}}{=} K_{x_{1:t}x_{1:t}}$ , and  $P_t \stackrel{\text{def}}{=} P_{x_{1:t}}$ . Similar to a standard bias-variance decomposition, the first term measures variance and the second term measures bias. Following Lemma C.27, VTL can be seen as greedily minimizing this bound to the approximation error (i.e., both bias and variance).

Maximizing correlation to prediction targets weighted by their variance It can be shown (see the proof below) that the VTL decision rule is equivalent to

$$\boldsymbol{x}_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg} \max} \sum_{\boldsymbol{x'} \in \mathcal{A}} \operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}] \cdot \operatorname{Cor}[f_{\boldsymbol{x'}}, y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}]^{2}.$$
(35)

That is, VTL maximizes the squared correlation between the next observation and the prediction targets, weighted by their variance. Intuitively, prediction targets are weighted by their variance since more can be learned about a prediction target with higher variance. This is precisely what leads to the "diverse" sample selection, and is akin to "uncertainty sampling" among the prediction targets and then selecting the observation which is most correlated with the selected prediction target.

*Proof.* Starting with the VTL objective, we have

$$\begin{aligned} \operatorname*{arg\,min}_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_n, y_{\boldsymbol{x}}] &= \operatorname*{arg\,min}_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \left( \operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_n] - \frac{\operatorname{Cov}[f_{\boldsymbol{x'}}, y_{\boldsymbol{x}} \mid \mathcal{D}_n]^2}{\operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_n]} \right) \\ &= \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \frac{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_n] \cdot \operatorname{Cov}[f_{\boldsymbol{x'}}, y_{\boldsymbol{x}} \mid \mathcal{D}_n]^2}{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_n] \cdot \operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_n]} + \operatorname{const} \\ &= \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_n] \cdot \operatorname{Cor}[f_{\boldsymbol{x'}}, y_{\boldsymbol{x}} \mid \mathcal{D}_n]^2 + \operatorname{const}. \end{aligned}$$

#### <span id="page-37-0"></span>D.3 Mean Marginal ITL

MacKay (1992) previously proposed "mean-marginal" ITL (MM-ITL) in the setting where  $\mathcal{S} = \mathcal{X}$ , which selects

$$\boldsymbol{x}_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg max}} \sum_{\boldsymbol{x'} \in \mathcal{A}} I(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})$$
(36)

<span id="page-37-3"></span><span id="page-37-2"></span>

and which simplifies in the GP setting to

$$x_{n} = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg max}} \frac{1}{2} \sum_{\boldsymbol{x'} \in \mathcal{A}} \log \left( \frac{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}]}{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]} \right)$$

$$= \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg min}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \log \operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]$$

$$= \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg min}} \operatorname{tr} \log \operatorname{diag} \operatorname{Var}[f_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]. \tag{37}$$

Analogously to the derivation of Equation (34), this can also be expressed as

$$\boldsymbol{x}_{n} = \operatorname*{arg\,min}_{\boldsymbol{x} \in \mathcal{S}} \left| \operatorname{diag} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}] \right|. \tag{38}$$

Effectively, MM-ITL ignores the mutual interaction between points in  $\mathcal{A}$ . As can be seen from Equation (37) and as is also mentioned by MacKay (1992), MM-ITL is equivalent to VTL up to a different weighting of the points in  $\mathcal{A}$ : instead of minimizing the average posterior variance (as in VTL), MM-ITL minimizes the average posterior log-variance. Under the lens of principle ( $\dagger$ ), this can be seen as minimizing the average marginal entropy of predictions within the target space:

$$\boldsymbol{x}_n = \underset{\boldsymbol{x} \in \mathcal{S}}{\min} \sum_{\boldsymbol{x'} \in \mathcal{A}} H[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}].$$

We remark that MM-ITL is a special case of EPIG (Bickford Smith et al., 2023, Appendix E.2).

Not a generalization of uncertainty sampling Unlike ITL, MM-ITL is not a generalization of uncertainty sampling. The reason is precisely that MM-ITL ignores the mutual interaction between points in  $\mathcal{A}$ . Consider the example where  $\mathcal{X} = \mathcal{S} = \mathcal{A} = \{1, \dots, 10\}$  where  $f_{1:9}$  are highly correlated while  $f_{10}$  is mostly independent of the other points. Visually, imagine a smooth function (i.e., under a Gaussian kernel) with points 1 through 9 close to each other and point 10 far away. Further, suppose that point 10 has a slightly larger marginal variance than the others. Then, MM-ITL would select one of the points 1:9 since this leads to the largest reduction in the marginal (log-)variances (i.e., to a small posterior "uncertainty"). In contrast, ITL selects the point with the largest prior marginal variance (cf. Appendix C.1), point 10, since this leads to the largest reduction in entropy. If

**Similarity to VTL** Observe that the following decision rule is equivalent to VTL:

$$\boldsymbol{x}_n = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg \, max}} \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}] - \operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}].$$

By Lemma C.38, for any  $x \in S$ , this objective value can be tightly lower- and upper-bounded (up to constant-factors) by

<span id="page-38-2"></span>
$$\sum_{\boldsymbol{x'} \in \mathcal{A}} \log \left( \frac{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}]}{\operatorname{Var}[f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}]} \right)$$

$$= 2 \sum_{\boldsymbol{x'} \in \mathcal{A}} \operatorname{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})$$
(see MM-ITL)
$$\stackrel{(i)}{=} - \sum_{\boldsymbol{x'} \in \mathcal{A}} \log \left( 1 - \operatorname{Cor}[f_{\boldsymbol{x'}}, y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}]^2 \right)$$
(39)

where (i) is detailed in example 8.5.1 of Cover (1999). Thus, VTL and MM-ITL are closely related.

**Experiments** In our experiments with Gaussian processes from Figures 2 and 6, we observe that MM-ITL performs similarly to VTL and CTL.

Convergence of uncertainty We derive a convergence guarantee for MM-ITL which is analogous to the guarantees for ITL from Theorem C.12 and for VTL from Theorem C.13. We will assume for simplicity that  $\Gamma_n$  is monotonically decreasing in n (i.e.,  $\alpha_n \leq 1$ ).

**Theorem D.1** (Convergence of uncertainty reduction of MM-ITL). Assume that Assumptions B.1 and B.2 are satisfied. Then for any  $n \ge 1$ , if  $\Gamma_0 \ge \cdots \ge \Gamma_{n-1}$  and the sequence  $\{x_i\}_{i=1}^n$  is generated by MM-ITL, then

$$\Gamma_{n-1} \le \frac{1}{n} \sum_{\boldsymbol{x'} \in A} \gamma_n(\{\boldsymbol{x'}\}; \mathcal{S}). \tag{40}$$

Proof. We have

$$\Gamma_{n-1} = \frac{1}{n} \sum_{i=0}^{n-1} \Gamma_{n-1}$$

<span id="page-38-0"></span><sup>&</sup>lt;sup>13</sup>This is because the observation reduces uncertainty not just about the observed point itself.

<span id="page-38-1"></span><sup>&</sup>lt;sup>14</sup>Because points  $f_{1:9}$  are highly correlated,  $H[f_{1:9}]$  is already "small".

$$\begin{aligned} &\overset{(i)}{\leq} \frac{1}{n} \sum_{i=0}^{n-1} \Gamma_{i} \\ &= \frac{1}{n} \sum_{i=0}^{n-1} \max_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \mathrm{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n}) \\ &\overset{(ii)}{=} \frac{1}{n} \sum_{i=0}^{n-1} \sum_{\boldsymbol{x'} \in \mathcal{A}} \mathrm{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}_{n+1}} \mid \mathcal{D}_{n}) \\ &\overset{(iii)}{=} \frac{1}{n} \sum_{\boldsymbol{x'} \in \mathcal{A}} \sum_{i=0}^{n-1} \mathrm{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}_{n+1}} \mid \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &\overset{(iv)}{=} \frac{1}{n} \sum_{\boldsymbol{x'} \in \mathcal{A}} \prod_{|X|=n} \mathrm{I}(f_{\boldsymbol{x'}}; \boldsymbol{y}_{\boldsymbol{x}_{1:n}}) \\ &\leq \frac{1}{n} \sum_{\boldsymbol{x'} \in \mathcal{A}} \max_{|X|=n} \mathrm{I}(f_{\boldsymbol{x'}}; \boldsymbol{y}_{\boldsymbol{X}}) \\ &= \frac{1}{n} \sum_{\boldsymbol{x'} \in \mathcal{A}} \gamma_{n}(\{\boldsymbol{x'}\}; \mathcal{S}) \end{aligned}$$

where (i) follows by assumption; (ii) follows from the MM-ITL decision rule; (iii) uses that the posterior variance of Gaussians is independent of the realization and only depends on the *location* of observations; and (iv) uses the chain rule of mutual information. The remainder of the proof is analogous to the proof of Theorem [C.12](#page-24-1) (cf. Appendix [C.5\)](#page-24-0).

Noting that

$$\mathrm{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}) \leq \sum_{\boldsymbol{x'} \in \mathcal{A}} \mathrm{I}(f_{\boldsymbol{x'}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})$$

for any n ≥ 1, x ∈ X , and x ′ ∈ A, Theorem [3.2](#page-2-1) can be readily rederived for MM-ITL (cf. Lemmas [C.14](#page-25-1) and [C.18\)](#page-27-0). Hence, the posterior marginal variances of MM-ITL can be bounded uniformly in terms of Γ<sup>n</sup> analogously to ITL.

#### <span id="page-39-0"></span>D.4 Correlation-based Transductive Learning

We will briefly look at the CTL (*Correlation-based TL*) decision rule

$$\boldsymbol{x}_{n} = \arg\max_{\boldsymbol{x} \in \mathcal{S}} \sum_{\boldsymbol{x'} \in \mathcal{A}} \operatorname{Cor}[f_{\boldsymbol{x}}, f_{\boldsymbol{x'}} \mid \mathcal{D}_{n-1}]$$
(41)

which permits no interpretation under principle ([†](#page-2-3)). However, if all correlations are non-negative (such as for the standard Gaussian and Matérn kernels), CTL is closely related to ITL, VTL, and MM-ITL (cf. Equations [\(35\)](#page-37-3) and [\(39\)](#page-38-2)). In this case, if |A| = 1, then all decision rules coincide.

If, on the other hand, correlations may be negative then there is a crucial difference between CTL and the decision rules motivated from principle ([†](#page-2-3)). Namely, decision rules following ([†](#page-2-3)) exhibit a preference for points with high *absolute* correlation to prediction targets as opposed to CTL which prefers points with high *positive* correlation. This stems from the intuitive fact that points with a strong negative correlation are equally informative as points with a strong positive correlation. Nevertheless, we observe in our experiments that (even for a linear kernel which does not ensure non-negative correlations) points selected by ITL and VTL are typically positively correlated with prediction targets.

# <span id="page-39-1"></span>D.5 Summary

We have seen that ITL, VTL, and MM-ITL can be seen as different interpretations of the same fundamental principle ([†](#page-2-3)), with the approximations CTL. If |A| = 1 and correlations are non-negative, then all four decision rules are equivalent. CTL prefers points with high positive correlation whereas the other decision rules prefer points with high absolute correlation. ITL is the only decision rule that takes into account the mutual dependence between points in  $\mathcal{A}$ , and VTL and MM-ITL differ only in their weighting of the posterior marginal variances of points in  $\mathcal{A}$ .

# <span id="page-40-1"></span>**E** Stochastic Target Spaces

When the target space A is large, it may be computationally infeasible to compute the exact objective. A natural approach to address this issue is to approximate the target space by a smaller set of size K.

**Discretizing the target space** One possibility is to discretize the target space  $\mathcal{A}$ . Compact target spaces can be addressed, e.g., via discretization arguments which are common in the Bayesian optimization literature (see, e.g., appendix C.1 of Srinivas et al. (2009)). That is, if the target space can be covered approximately using a finite (possibly large) set of points, the guarantees of Theorem 3.2 extend directly. This, however, can be impractical when the required size of discretization for sufficiently small approximation error is large. In the following, we briefly discuss a natural alternative approach based on sampling points from  $\mathcal{A}$ .

**Target distributions** Let  $A \subseteq \mathcal{X}$  be a (possibly continuous) target space, and let  $\mathcal{P}_A$  be a probability distribution supported on A. In iteration n, a subset  $A_n$  of K points is sampled independently from A according to the distribution  $\mathcal{P}_A$  and the objective is computed on this subset. Formally, this amounts to a single-sample Monte Carlo approximation of

$$\boldsymbol{x}_{n} \in \arg\max_{\boldsymbol{x} \in \mathcal{S}} \mathbb{E}_{A} \stackrel{\text{iid}}{\sim} \mathcal{P}_{A} [I(\boldsymbol{f}_{A}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})]. \tag{42}$$

The convergence guarantees from Appendix C can be generalized to the setting of stochastic target spaces by estimating how often points "near" a specified prediction target  $x \in A$  are sampled.

**Definition E.1** ( $\gamma$ -ball at x). Given  $x \in A$  and any  $\gamma \geq 0$ , we call the set

$$B_{\gamma}(\boldsymbol{x}) \stackrel{\text{def}}{=} \{ \boldsymbol{x'} \in \mathcal{X} \mid \|\boldsymbol{x} - \boldsymbol{x'}\| \leq \gamma \}$$

the  $\gamma$ -ball at  $\boldsymbol{x}$ . Further, we call  $\mathcal{P}_{\mathcal{A}}(B_{\gamma}(\boldsymbol{x}))$  the weight of that ball.

**Proposition E.2** (sketch). Given any  $n \ge 1$ ,  $K \ge 1$ ,  $\gamma > 0$ , and  $x \in A$ , suppose that  $B_{\gamma}(x)$  has weight p > 0. Assume that the ITL objective is  $L_I$ -Lipschitz continuous. Then, with probability at least  $1 - \exp(-(1-p)n/(8K))$ ,

$$\sigma_n^2(\boldsymbol{x}) \lesssim \eta_{\mathcal{S}}^2(\boldsymbol{x}) + CL_I \gamma \frac{\gamma_{k(n)}}{\sqrt{k(n)}}$$

where  $k(n) \stackrel{\text{def}}{=} Kpn/2$ .

*Proof sketch.* Let  $Y_i \sim \operatorname{Binom}(K,p)$  denote the random variable counting the number of occurrences of a point from  $B_{\gamma}(\boldsymbol{x})$  in  $A_i$ . Moreover, we write  $X_i \stackrel{\text{def}}{=} \mathbb{1}\{B_{\gamma}(\boldsymbol{x}) \cap A_i \neq \emptyset\}$ . Note that

$$\nu \stackrel{\text{def}}{=} \mathbb{E}X_i = \mathbb{P}(B_{\gamma}(\boldsymbol{x}) \cap A_i \neq \emptyset) = 1 - \mathbb{P}(Y_i = 0) = 1 - (1 - p)^K \approx Kp$$

where the approximation stems from a first-order truncation of the Bernoulli series. Let  $X \stackrel{\text{def}}{=} \sum_{i=1}^{n} X_i$  with  $\mathbb{E}X = n\nu \approx Kpn$ .

Using the assumed Lipschitz-continuity of the objective, we know that  $I(\mathbf{f}_{A'}; y_{\mathbf{x}} \mid \mathcal{D}_{n-1}) \leq L_I \gamma I(\mathbf{f}_A; y_{\mathbf{x}} \mid \mathcal{D}_{n-1})$  where  $A' \stackrel{\text{def}}{=} (A \setminus \{\mathbf{x}_\gamma\}) \cup \{\mathbf{x}\}$  and  $\mathbf{x}_\gamma$  is the point from the  $\gamma$ -ball at  $\mathbf{x}$ . The bound then follows analogously to Theorem 3.2.

Finally, by Chernoff's bound, at least Kpn/2 iterations contain a point from  $B_{\gamma}(x)$  with probability at least  $1 - \exp(-Kpn/8)$ .

This strategy can also be used to generalize the VTL, CTL, and MM-ITL objectives to stochastic target spaces.

# <span id="page-40-0"></span>F Closed-form Decision Rules

Below, we list the closed-form expressions for the ITL and VTL objectives. In the following,  $k_n$  denotes the kernel conditional on  $\mathcal{D}_n$ .

ITL

<span id="page-41-2"></span>
$$I(\mathbf{f}_{\mathcal{A}}; y_{\mathbf{x}} \mid \mathcal{D}_{n-1}) = \frac{1}{2} \log \left( \frac{\operatorname{Var}[y_{\mathbf{x}} \mid \mathcal{D}_{n-1}]}{\operatorname{Var}[y_{\mathbf{x}} \mid \mathbf{f}_{\mathcal{A}}, \mathcal{D}_{n-1}]} \right)$$

$$= \frac{1}{2} \log \left( \frac{k_{n-1}(\mathbf{x}, \mathbf{x}) + \rho^{2}}{\hat{k}_{n-1}(\mathbf{x}, \mathbf{x}) + \rho^{2}} \right)$$
(43)

where  $\hat{k}_n(\boldsymbol{x}, \boldsymbol{x}) = k_n(\boldsymbol{x}, \boldsymbol{x}) - \boldsymbol{k}_n(\boldsymbol{x}, \boldsymbol{A}) \boldsymbol{K}_n(\boldsymbol{A}, \boldsymbol{A})^{-1} \boldsymbol{k}_n(\boldsymbol{A}, \boldsymbol{x})$ .

VTL

$$\operatorname{tr} \operatorname{Var}[\boldsymbol{f}_{\mathcal{A}} \mid \mathcal{D}_{n-1}, y_{\boldsymbol{x}}] = \sum_{\boldsymbol{x'} \in \mathcal{A}} \left( k_{n-1}(\boldsymbol{x'}, \boldsymbol{x'}) - \frac{k_{n-1}(\boldsymbol{x}, \boldsymbol{x'})^2}{k_{n-1}(\boldsymbol{x}, \boldsymbol{x}) + \rho^2} \right).$$

# <span id="page-41-1"></span>**G** Computational Complexity

Evaluating the acquisition function of ITL in round n requires computing for each  $x \in \mathcal{S}$ ,

$$\begin{split} & \mathrm{I}(\mathbf{\textit{f}}_{\mathcal{A}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n}) \\ & = \frac{1}{2} \log \left( \frac{|\mathrm{Var}[\mathbf{\textit{f}}_{\mathcal{A}} \mid \mathcal{D}_{n}]|}{|\mathrm{Var}[\mathbf{\textit{f}}_{\mathcal{A}} \mid y_{\boldsymbol{x}}, \mathcal{D}_{n}]|} \right) \\ & = \frac{1}{2} \log \left( \frac{\mathrm{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_{n}]}{|\mathrm{Var}[y_{\boldsymbol{x}} \mid \mathbf{\textit{f}}_{\mathcal{A}}, \mathcal{D}_{n}]} \right) \end{split} \tag{backward}.$$

Let  $|\mathcal{S}| = m$  and  $|\mathcal{A}| = k$ . Then, the forward method has complexity  $O(m \cdot k^3)$ . For the backward method, observe that the variances are scalar and the covariance matrix  $\operatorname{Var}[f_{\mathcal{A}} \mid \mathcal{D}_n]$  only has to be inverted once for all points x. Thus, the backward method has complexity  $O(k^3 + m)$ .

When the size m of  $\mathcal S$  is relatively small (and hence, all points in  $\mathcal S$  can be considered during each iteration of the algorithm), GP inference corresponds simply to computing conditional distributions of a multivariate Gaussian. The performance can therefore be improved by keeping track of the full posterior distribution over  $f_{\mathcal S}$  of size  $O(m^2)$  and conditioning on the latest observation during each iteration of the algorithm. In this case, after each observation the posterior can be updated at a cost of  $O(m^2)$  which does not grow with the time n, unlike classical GP inference.

Overall, when m is small, the computational complexity of ITL is  $O(k^3 + m^2)$ . When m is large (or possibly infinite) and a subset of  $\tilde{m}$  points is considered in a given iteration, the computational complexity of ITL is  $O(k^3 + \tilde{m} \cdot n^3)$ , neglecting the complexity of selecting the  $\tilde{m}$  candidate points. In the latter case, the computational cost of ITL is dominated by the cost of GP inference.

Khanna et al. (2017) discuss distributed and stochastic approximations of greedy algorithms to (weakly) submodular problems that are also applicable to ITL.

#### <span id="page-41-0"></span>**H** Additional GP Experiments & Details

We use homoscedastic Gaussian noise with standard deviation  $\rho = 0.1$  and a discretization of  $\mathcal{X} = [-3, 3]^2$  of size 2 500. Uncertainty bands correspond to one standard error over 10 random seeds.

**Additional experiments** Figure 6 includes the following additional experiments:

- 1. Extrapolation Setting  $(A \cap S = \emptyset)$ : Right experiment from Figure 2 under the Gaussian kernel. ITL has a similar advantage as in the setting shown in Figure 3.
- 2. Heteroscedastic Noise: Left experiment from Figure 2 under the Gaussian kernel with heteroscedastic Gaussian noise

$$\rho(\boldsymbol{x}) = \begin{cases} 1 & \text{if } \boldsymbol{x} \in [-\frac{1}{2}, \frac{1}{2}]^2 \\ 0.1 & \text{otherwise} \end{cases}.$$

If observation noise is heteroscedastic, in considering *posterior* rather than *prior* uncertainty, ITL avoids points with high aleatoric uncertainty, which accelerates learning.

<span id="page-42-2"></span>Figure 6: Additional GP experiments

- 3. Effect of Smoothness: Experiment from Figure 3 under the Laplace kernel. All algorithms except for US and RANDOM perform equally well. This validates our claims from Section 3.3: in the extreme non-smooth case of a Laplace kernel and  $\mathcal{A} \subseteq \mathcal{S}$ , points outside  $\mathcal{A}$  do not provide any additional information, and ITL and "local" UNSA coincide.
- 4. Sparser Target: Experiment from Figure 3 under the Gaussian kernel, but with domain extended to  $\mathcal{X} = [-10, 10]^2$ .

**Hyperparameters of TRUVAR** As suggested by Bogunovic et al. (2016), we use  $\tilde{\eta}_{(1)}^2 = 1$ , r = 0.1, and  $\delta = 0$  (even though the theory only holds for  $\delta > 0$ ). The TRUVAR baseline only applies when  $\mathcal{A} \subseteq \mathcal{S}$  (cf. Section 6).

**Smoothing to reduce numerical noise** Applied running average with window 5 to entropy curves of Figures 2 and 6 to smoothen out numerical noise.

# <span id="page-42-0"></span>I Alternative Settings for Active Fine-Tuning

In our main experiments, we consider the setting  $\mathcal{A} \cap \mathcal{S} = \emptyset$ , i.e., the prediction targets cannot be used for fine-tuning since their labels are not known. This setting is particularly relevant for practical applications where the model is fine-tuned dynamically at test time to each prediction target (or a small set of prediction target). Put differently, in this "transductive" setting, extrapolation to new prediction targets happens at *test-time* with knowledge of the prediction target(s). This is in contrast to a more traditional "inductive" setting, where extrapolation happens at *train-time* without knowledge of the concrete prediction targets, but under the assumption of samples from (or knowledge of) the target distribution. In the following, we briefly survey two settings motivated from an "inductive" perspective.

#### <span id="page-42-1"></span>I.1 Prediction Targets are Contained in Sample Space: $A \subseteq S$

If labels can be obtained cheaply, one can also fine-tune on the prediction targets directly, i.e.,  $\mathcal{A} \subseteq \mathcal{S}$ . Note, however, that the set  $\mathcal{A}$  is still assumed to be small (e.g.,  $|\mathcal{A}|=100$  in the CIFAR-100 experiment). We perform an experiment in this setting and report the results in Figure 7. The experiment shows that — similarly to the GP experiment from Figure 2 — there can be *additional value* in fine-tuning the model on relevant data selected from  $\mathcal{S}$  beyond simply fine-tuning the model on  $\mathcal{A}$ .

<span id="page-43-2"></span>Figure 7: Evaluation of CIFAR-100 experiment in the setting A ⊆ S, i.e., one can also sample from the 100 prediction targets A. The solid black line denotes the performance of the model fine-tuned on all of A. This experiment shows that there is *additional value* in fine-tuning the model on relevant data from S beyond simply fine-tuning the model on A. The baselines are summarized in Appendix [J.5](#page-47-0)

#### <span id="page-43-1"></span>I.2 Active Domain Adaptation

Active DA [\(Rai et al.,](#page-14-11) [2010;](#page-14-11) [Saha et al.,](#page-14-12) [2011;](#page-14-12) [Berlind & Urner,](#page-10-12) [2015\)](#page-10-12) studies the problem of selecting the most informative samples from a (large) target domain A, given a model trained on a source domain S. This problem can be cast as an instance of transductive active learning with target space A and sample space S ′ = S ∪ A where the model is already conditioned on all of S. This is slightly different from the setting considered in Section [4](#page-4-0) where A is small and not necessarily part of the sample space. We hypothesize that ITL behaves similarly to recent work on active DA [\(Su et al.,](#page-15-18) [2020;](#page-15-18) [Prabhu](#page-14-13) [et al.,](#page-14-13) [2021;](#page-14-13) [Fu et al.,](#page-12-18) [2021\)](#page-12-18): querying informative and diverse samples from A that are dissimilar to S. Evaluating ITL and VTL empirically in this setting is a promising direction for future work.

# <span id="page-43-0"></span>J Additional NN Experiments & Details

We outline the active fine-tuning of NNs in Algorithm [1.](#page-43-3)

```
Algorithm 1 Active Fine-Tuning of NNs
  Given: initialized or pre-trained model f, small sample A ∼ PA
  initialize dataset D = ∅
  repeat
     sample S ∼ PS
     subsample target space A′ u.a.r. ∼ A
     initialize batch B = ∅
     compute kernel matrix K over domain [S, A′
     repeat b times
        compute acquisition function w.r.t. A′
                                              , based on K
        add maximizer x ∈ S of acquisition function to B
        update conditional kernel matrix K
     obtain labels for B and add to dataset D
     update f using data D
```

In Appendix [J.1,](#page-44-0) we detail metrics and hyperparameters. We describe in Appendices [J.2](#page-45-0) and [J.3](#page-46-0) how to compute the (initial) conditional kernel matrix K, and in Appendix [J.4](#page-46-1) how to update this matrix K to obtain conditional embeddings for batch selection.

In Appendix [J.5,](#page-47-0) we show that ITL and CTL significantly outperform a wide selection of commonly used heuristics. In Appendices [J.6](#page-51-0) and [J.7,](#page-51-1) we conduct additional experiments and ablations.

<span id="page-44-1"></span>Table 1: Hyperparameter summary of NN experiments. (\*) we train until convergence on oracle validation accuracy.

|                | MNIST | CIFAR-100 |
|----------------|-------|-----------|
| $\rho$         | 0.01  | 1         |
| M              | 30    | 100       |
| m              | 3     | 10        |
| k              | 1000  | 1000      |
| batch size $b$ | 1     | 10        |
| # of epochs    | (*)   | 5         |
| learning rate  | 0.001 | 0.001     |

Hübotter et al. (2024) discusses additional motivation and related work that has previously studied active fine-tuning, but which has largely focused on the training algorithm rather than data selection.

#### <span id="page-44-0"></span>J.1 Experiment Details

We evaluate the accuracy with respect to  $\mathcal{P}_{\mathcal{A}}$  using a Monte Carlo approximation with out-of-sample data:

$$\mathrm{accuracy}(\widehat{\bm{\theta}}) \approx \mathbb{E}_{(\bm{x},y) \sim \mathcal{P}_{\!\!\mathcal{A}}} \mathbb{1}\{y = \arg\max_i f_i(\bm{x}; \widehat{\bm{\theta}})\}.$$

We provide an overview of the hyperparameters used in our NN experiments in Table 1. The effect of noise standard deviation  $\rho$  is small for all tested  $\rho \in [1,100]$  (cf. ablation study in Table 2).  $^{15}$  M denotes the size of the sample  $A \sim \mathcal{P}_{\mathcal{A}}$ . In each iteration, we select the target space  $\mathcal{A} \leftarrow A'$  as a random subset of m points from A. We provide an ablation over m in Appendix J.6.

During each iteration, we select the batch B according to the decision rule from a random sample from  $P_S$  of size k.<sup>17</sup>

Since we train the MNIST model from scratch, we train from random initialization until convergence on oracle validation accuracy. We do this to stabilize the learning curves, and provide the least biased (due to the training algorithm) results. For CIFAR-100, we train for 5 epochs (starting from the previous iterations' model) which we found to be sufficient to obtain good performance.

We use the ADAM optimizer (Kingma & Ba, 2014). In our CIFAR-100 experiments, we use a pre-trained EfficientNet-B0 (Tan & Le, 2019), and fine-tune the final and penultimate layers. We freeze earlier layers to prevent overfitting to the "few-shot" training data.

To prevent numerical inaccuracies when computing the ITL objective, we optimize

$$I(\boldsymbol{y}_{\mathcal{A}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}) = \frac{1}{2} \log \left( \frac{\operatorname{Var}[y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}]}{\operatorname{Var}[y_{\boldsymbol{x}} \mid \boldsymbol{y}_{\mathcal{A}}, \mathcal{D}_{n-1}]} \right)$$
(44)

instead of Equation (43), which amounts to adding  $\rho^2$  to the diagonal of the covariance matrix before inversion. This appears to improve numerical stability, especially when using gradient embeddings.<sup>19</sup>

<span id="page-44-2"></span> $<sup>^{15}</sup>$ We use a larger noise standard deviation  $\rho$  in CIFAR-100 to stabilize the numerics of batch selection via conditional embeddings (cf. Table 2).

<span id="page-44-3"></span> $<sup>^{16}</sup>$ This appears to improve the training, likely because it prevents overfitting to peculiarities in the finite sample A (cf. Figure 16).

<span id="page-44-4"></span><sup>&</sup>lt;sup>17</sup>In large-scale problems, the work of Coleman et al. (2022) suggests to use an (approximate) nearest neighbor search to select the (large) candidate set rather than sampling u.a.r. from  $\mathcal{P}_{\mathcal{S}}$ . This can be a viable alternative to simply increasing k and suggests future work.

<span id="page-44-6"></span><span id="page-44-5"></span>That is, to stop training as soon as accuracy on a validation set from  $\mathcal{P}_{\mathcal{A}}$  decreases in an epoch.

<sup>&</sup>lt;sup>19</sup>In our experiments, we observe that the effect of various choices of  $\rho$  on this slight adaptation of the ITL decision rule has negligible impact on performance. The more prominent effect of  $\rho$  appears to arise from the batch selection via conditional embeddings (cf. Table 2).

In our experiments, we use last-layer neural tangent embeddings<sup>20</sup> and  $\Sigma = I$  to evaluate ITL and VTL, and select inputs for labeling and training f. Notably, we use this linear Gaussian approximation of f only to guide the active data selection and not for inference.

#### <span id="page-45-0"></span>J.2 Embeddings and Kernels

Using a neural network to parameterize f, we evaluate the canonical approximations of f by a stochastic process in the following.

An embedding  $\phi(x)$  is a latent representation of an input x. Collecting the embeddings as rows in the design matrix  $\Phi$  of a set of inputs X, one can approximate the network by the linear function  $f_X = \Phi \beta$  with weights  $\beta$ . Approximating the weights by  $\beta \sim \mathcal{N}(\mu, \Sigma)$  implies that  $f_X \sim \mathcal{N}(\Phi \mu, \Phi \Sigma \Phi^\top)$ . The covariance matrix  $K_{XX} = \Phi \Sigma \Phi^\top$  can be succinctly represented in terms of its associated kernel  $k(x, x') = \phi(x)^\top \Sigma \phi(x')$ . Here,

- $\phi(x)$  is the latent representation of x, and
- $\Sigma$  captures the dependencies in the latent space.

While any choice of embedding  $\phi$  is possible, the following are common choices:

- 1. Last-Layer: A common choice for  $\phi(x)$  is the representation of x from the penultimate layer of the neural network (Holzmüller et al., 2023). Interpreting the early layers as a feature encoder, this uses the low-dimensional feature map akin to random feature methods (Rahimi & Recht, 2007).
- 2. Output Gradients (eNTK): Another common choice is  $\phi(x) = \nabla_{\theta} f(x; \theta)$  where  $\theta$  are the network parameters (Holzmüller et al., 2023). Its associated kernel is known as the empirical neural tangent kernel (eNTK) and the posterior mean of this GP approximates ultra-wide NNs trained with gradient descent (Jacot et al., 2018; Arora et al., 2019; Lee et al., 2019; Khan et al., 2019; He et al., 2020; Malladi et al., 2023). Kassraie & Krause (2022) derive bounds of  $\gamma_n$  under this kernel. If  $\theta$  is restricted to the weights of the final linear layer, then this embedding is simply the last-layer embedding.
- 3. Loss Gradients: Another possible choice is

$$\phi(\boldsymbol{x}) = \left. \boldsymbol{\nabla}_{\!\!\boldsymbol{\theta}} \, \ell(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{\theta}), \hat{\boldsymbol{y}}(\boldsymbol{x})) \right|_{\boldsymbol{\theta} = \widehat{\boldsymbol{\theta}}}$$

where  $\ell$  is a loss function,  $\hat{y}(x)$  is the predicted label, and  $\hat{\theta}$  are the current parameter estimates (Ash et al., 2020).

- 4. Outputs (eNNGP): Another possible choice is  $\phi(x) = f(x)$ , i.e., the output of the network. Its associated kernel is known as the *empirical neural network Gaussian process* (eNNGP) kernel (Lee et al., 2018).
- 5. Predictive (Kirsch, 2023): Given a Bayesian neural network (Blundell et al., 2015) or probabilistic (deep) ensemble (Lakshminarayanan et al., 2017), which induce samples  $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \sim p(\boldsymbol{\theta})$  from the distribution over network parameters, one can approximate the predictive covariance  $k(\boldsymbol{x}, \boldsymbol{x'}) = \operatorname{Cov}_{\boldsymbol{\theta}}[f(\boldsymbol{x}; \boldsymbol{\theta}), f(\boldsymbol{x'}; \boldsymbol{\theta})]$ . This kernel measures proximity in the prediction space rather than parameter space and as such does not require gradient information. The corresponding feature map is  $\boldsymbol{\phi}(\boldsymbol{x}) = \frac{1}{\sqrt{K}}[\bar{f}(\boldsymbol{x}; \boldsymbol{\theta}_1) \cdots \bar{f}(\boldsymbol{x}; \boldsymbol{\theta}_K)]^{\top}$  where  $\bar{f}(\boldsymbol{x}; \boldsymbol{\theta}_k) \stackrel{\text{def}}{=} f(\boldsymbol{x}; \boldsymbol{\theta}_k) \frac{1}{K} \sum_{l=1}^K f(\boldsymbol{x}; \boldsymbol{\theta}_l)$ .

In the additional experiments from this appendix we use last-layer embeddings unless noted otherwise. We compare the performance of last-layer and the loss gradient embedding

<span id="page-45-2"></span>
$$\phi(\mathbf{x}) = \left. \nabla_{\boldsymbol{\theta}'} \ell_{\text{CE}}(f(\mathbf{x}; \boldsymbol{\theta}), \hat{y}(\mathbf{x})) \right|_{\boldsymbol{\theta} = \widehat{\boldsymbol{\theta}}}$$
(45)

where  $\boldsymbol{\theta'}$  are the parameters of the final output layer,  $\widehat{\boldsymbol{\theta}}$  are the current parameter estimates,  $\hat{y}(\boldsymbol{x}) = \arg\max_i f_i(\boldsymbol{x}; \widehat{\boldsymbol{\theta}})$  are the associated predicted labels, and  $\ell_{\text{CE}}$  denotes the cross-entropy loss. This gradient embedding captures the potential update direction upon observing a new point (Ash et al., 2020). Moreover, Ash et al. (2020) show that for most neural networks, the norm of these gradient embeddings are a conservative lower bound to the norm assumed by taking any other proxy label  $\hat{y}(\boldsymbol{x})$ . In Figure 8, we observe only negligible differences in performance between this and the last-layer embedding.

<span id="page-45-1"></span><sup>&</sup>lt;sup>20</sup>We observe essentially the same performance with loss gradient embeddings, cf. Appendix J.2.

<span id="page-46-2"></span>Figure 8: Comparison of loss gradient ("G-") and last-layer embeddings ("L-").

<span id="page-46-3"></span>Figure 9: Uncertainty quantification (i.e., estimation of  $\Sigma$ ) via a Laplace approximation (LA, Daxberger et al. (2021)) over last-layer weights using a Kronecker factored log-likelihood Hessian approximation (Martens & Grosse, 2015) and the loss gradient embeddings from Equation (45). The results are shown for the MNIST experiment. We do not observe a performance improvement beyond the trivial approximation  $\Sigma = I$ .

### <span id="page-46-0"></span>J.3 Towards Uncertainty Quantification in Latent Space

A straightforward and common approximation of the uncertainty about NN weights is given by  $\Sigma = I$ , and we use this approximation throughout our experiments.

The poor performance of UNSA (cf. Appendix J.5) with this approximation suggests that with more sophisticated approximations, the performance of ITL, VTL, and CTL can be further improved. Further research is needed to study the effect of more sophisticated approximations of "uncertainty" in the latent space. For example, with parameter gradient embeddings, the latent space is the network parameter space where various approximations of  $\Sigma$  based on Laplace approximation (Daxberger et al., 2021; Antorán et al., 2022), variational inference (Blundell et al., 2015), or Markov chain Monte Carlo (Maddox et al., 2019) have been studied. We also evaluate Laplace approximation (LA, Daxberger et al. (2021)) for estimating  $\Sigma$  but see no improvement (cf. Figure 9). Nevertheless, we believe that uncertainty quantification is a promising direction for future work, with the potential to improve performance of ITL and its variations substantially.

# <span id="page-46-1"></span>J.4 Batch Selection via Conditional Embeddings

We will refer to the greedy decision rule from Equation (3) as BACE, short for **Batch** selection via **Conditional Embeddings**. BACE can be implemented efficiently using the Gaussian approximation of  $f_X$  from Appendix J.2 by iteratively conditioning on the previously selected points  $x_{n,1:i-1}$ , and updating the kernel matrix  $K_{XX}$  using the closed-form formula for the variance of conditional

<span id="page-47-1"></span>Figure 10: Advantage of batch selection via conditional embeddings over top-b selection in the CIFAR-100 experiment.

Gaussians:

$$K_{XX} \leftarrow K_{XX} - \frac{1}{K_{x_j x_j} + \rho^2} K_{X x_j} K_{x_j X}$$

$$\tag{46}$$

where j denotes the index of the selected  $x_{n,i}$  within X and  $\rho^2$  is the noise variance. Note that  $K_{x_jx_j}$  is a scalar and  $K_{Xx_j}$  is a row vector, and hence, this iterative update can be implemented efficiently.

We remark that Equation (3) is a natural extension of previous non-adaptive active learning methods, which typically maximize some notion of "distance" between points in the batch, to the "directed" setting (Ash et al., 2020; Zanette et al., 2021; Holzmüller et al., 2023; Pacchiano et al., 2024). BACE simultaneously maximizes "distance" between points in a batch and minimizes "distance" to points in  $\mathcal{A}$ .

The sample efficiency of BACE  $B_n$ , and therefore also the greedily constructed  $B_n'$  (which gives a constant-factor approximation with respect to the objective), yields diverse batches by design. In Figure 10, we compare BACE to selecting the top-b points according to the decision rule (which does *not* yield diverse batches). We observe a significant improvement in accuracy and data retrieval when using BACE. We expect the gap between both approaches to widen further with larger batch sizes.

Computational complexity of BACE As derived in Appendix G, a single batch selection step of BACE has complexity  $O(b(k^3+m^2))$  where b is the size of the batch,  $k=|\mathcal{A}|$  is the size of the target space, and  $m=|\mathcal{S}|$  is the size of the candidate set. In the case of large m, an alternative implementation whose runtime does not depend on m is described in Appendix G.

#### <span id="page-47-0"></span>J.5 Baselines

In Figure 11, we compare against additional baselines:

- Both TYPICLUST (Hacohen et al., 2022) and PROBCOVER (Yehuda et al., 2022) are recent methods to select points that "cover" the data distribution well. To maintain comparability between algorithms, we use the same embeddings as for ITL which are re-computed before every new batch selection. ITL significantly outperforms TYPICLUST & PROBCOVER, which only attempt to cover S well without taking A into account (i.e., are "undirected").
- Mehta et al. (2022) introduced EIG for training neural classification models, which uses the same decision rule as ITL, but approximates the conditional entropy based on the networks' softmax output rather than using a GP approximation. We approximate the conditional entropy using a single gradient step of the hallucinated updates on the parameters of the final layer, as mentioned by Mehta et al. (2022). We observe that EIG is not competitive for batch-wise selection (CIFAR-100) since it does not encourage batch diversity. Moreover, we observe that EIG is orders of magnitude slower than ITL (since it has to compute  $|S| \cdot C$  individual gradient steps where C is the number of classes). We note that since our datasets are balanced, the AEIG algorithm from Mehta et al. (2022) coincides with EIG.

<span id="page-48-0"></span>Figure 11: Comparison to baselines for the experiment of Figure 4.

Since, EIG does not have an open-source implementation, we implemented it ourselves following Mehta et al. (2022). For TYPICLUST & PROBCOVER, we use the author's implementation. In the figure, we show that ITL & VTL substantially outperform all baselines.

In the following, we briefly describe other commonly used "undirected" decision rules.

Denote the softmax distribution over labels i at inputs x by

$$p_i(\boldsymbol{x}; \widehat{\boldsymbol{\theta}}) \propto \exp(f_i(\boldsymbol{x}; \widehat{\boldsymbol{\theta}})).$$

The following heuristics computed based on the softmax distribution aim to quantify the "uncertainty" about a particular input x:

• MAXENTROPY (Settles & Craven, 2008):

$$\boldsymbol{x}_n = \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \mathrm{H}[p(\boldsymbol{x}; \widehat{\boldsymbol{\theta}}_{n-1})].$$

• MAXMARGIN (Scheffer et al., 2001; Settles & Craven, 2008):

$$\boldsymbol{x}_n = \operatorname*{arg\,min}_{\boldsymbol{x} \in \mathcal{S}} p_1(\boldsymbol{x}; \widehat{\boldsymbol{\theta}}_{n-1}) - p_2(\boldsymbol{x}; \widehat{\boldsymbol{\theta}}_{n-1})$$

where  $p_1$  and  $p_2$  are the two largest class probabilities.

• LEASTCONFIDENCE (Lewis & Gale, 1994; Settles & Craven, 2008; Hendrycks & Gimpel, 2017; Tamkin et al., 2022):

$$x_n = \operatorname*{arg\,min}_{x \in \mathcal{S}} p_1(x; \widehat{\boldsymbol{\theta}}_{n-1})$$

where  $p_1$  is the largest class probability.

An alternative class of decision rules aims to select diverse batches by maximizing the distances between points. Embeddings  $\phi(x)$  induce the (Euclidean) embedding distance

$$d_{\phi}(\boldsymbol{x}, \boldsymbol{x'}) \stackrel{\text{def}}{=} \|\phi(\boldsymbol{x}) - \phi(\boldsymbol{x'})\|_{2}.$$

Similarly, a kernel k induces the kernel distance

$$d_k(\boldsymbol{x}, \boldsymbol{x'}) \stackrel{\text{def}}{=} \sqrt{k(\boldsymbol{x}, \boldsymbol{x}) + k(\boldsymbol{x'}, \boldsymbol{x'}) - 2k(\boldsymbol{x}, \boldsymbol{x'})}.$$

It is straightforward to see that if  $k(x, x') = \phi(x)^{\top} \phi(x')$ , then embedding and kernel distances coincide, i.e.,  $d_{\phi}(x, x') = d_k(x, x')$ .

• MAXDIST [\(Holzmüller et al.,](#page-12-0) [2023;](#page-12-0) [Yu & Kim,](#page-16-11) [2010;](#page-16-11) [Sener & Savarese,](#page-14-5) [2017;](#page-14-5) [Geifman &](#page-12-21) [El-Yaniv,](#page-12-21) [2017\)](#page-12-21) constructs the batch by choosing the point with the maximum distance to the nearest previously selected point:

$$x_n = \underset{x \in \mathcal{S}}{\operatorname{arg \, max \, min}} d(x, x_i)$$

• Similarly, K-MEANS++ [\(Holzmüller et al.,](#page-12-0) [2023\)](#page-12-0) selects the batch via K-MEANS++ seeding [\(Arthur et al.,](#page-10-13) [2007;](#page-10-13) [Ostrovsky et al.,](#page-14-19) [2013\)](#page-14-19). That is, the first centroid x<sup>1</sup> is chosen uniformly at random and the subsequent centroids are chosen with a probability proportional to the square of the distance to the nearest previously selected centroid:

$$\mathbb{P}(\boldsymbol{x}_n = \boldsymbol{x}) \propto \min_{i < n} d(\boldsymbol{x}, \boldsymbol{x}_i)^2.$$

When using the loss gradient embeddings from Equation [\(45\)](#page-45-2), this decision rule is known as BADGE [\(Ash et al.,](#page-10-3) [2020\)](#page-10-3).

Finally, we summarize common kernel-based decision rules.

• UNDIRECTED ITL chooses

$$\begin{aligned} \boldsymbol{x}_n &= \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \mathrm{I}(\boldsymbol{f}_{\mathcal{S}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}) \\ &= \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \mathrm{I}(f_{\boldsymbol{x}}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1}) \,. \end{aligned}$$

This can be shown to be equivalent to MAXDET [\(Holzmüller et al.,](#page-12-0) [2023\)](#page-12-0) which selects

$$\boldsymbol{x}_n = \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \left| \boldsymbol{K}_{\boldsymbol{x}} + \sigma^2 \boldsymbol{I} \right|$$

where K<sup>x</sup> denotes the kernel matrix over x1:n−1∪{x}, conditioned on the prior observations Dn−1.

• UNS<sup>A</sup> [\(Lewis & Catlett,](#page-13-2) [1994\)](#page-13-2) which with embeddings ϕn−<sup>1</sup> after round n − 1 corresponds to:

$$\boldsymbol{x}_n = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg max}} \sigma_{n-1}^2(\boldsymbol{x}) = \underset{\boldsymbol{x} \in \mathcal{S}}{\operatorname{arg max}} \left\| \boldsymbol{\phi}_{n-1}(\boldsymbol{x}) \right\|_2^2.$$

With batch size b = 1, UNSA coincides with UNDIRECTED ITL. When evaluated with gradient embeddings, this acquisition function is similar to previously used "embedding length" or "gradient length" heuristics [\(Settles & Craven,](#page-15-6) [2008\)](#page-15-6).

• UNDIRECTED VTL [\(Cohn,](#page-11-10) [1993\)](#page-11-10) is the special case of VTL without specified prediction targets (i.e., A = S). In the literature, this decision rule is also known as BAIT [\(Holzmüller](#page-12-0) [et al.,](#page-12-0) [2023;](#page-12-0) [Ash et al.,](#page-10-14) [2021\)](#page-10-14).

We compare to the abovementioned decision rules and summarize the results in Figure [12.](#page-50-0) We observe that most "undirected" decision rules perform worse (and often significantly so) than RANDOM. This is likely due to frequently selecting points from the support of P<sup>S</sup> which are not in the support of <sup>P</sup><sup>A</sup> since the points are "adversarial examples" that the model <sup>θ</sup><sup>b</sup> is not trained to perform well on. In the case of MNIST, the poor performance can also partially be attributed to the well-known "cold-start problem" [\(Gao et al.,](#page-12-17) [2020\)](#page-12-17).

In Figure [4,](#page-6-0) we also compare to the following "directed" decision rules:

• COSINESIMILARITY [\(Settles & Craven,](#page-15-6) [2008\)](#page-15-6) selects <sup>x</sup><sup>n</sup> = arg maxx∈S <sup>∠</sup><sup>ϕ</sup>n−<sup>1</sup> (x, A) where

$$\angle_{\phi}(x, \mathcal{A}) \stackrel{\text{def}}{=} \frac{1}{|\mathcal{A}|} \sum_{x' \in \mathcal{A}} \frac{\phi(x)^{\top} \phi(x')}{\|\phi(x)\|_2 \|\phi(x')\|_2}.$$

• INFORMATIONDENSITY [\(Settles & Craven,](#page-15-6) [2008\)](#page-15-6) is defined as the multiplicative combination of MAXENTROPY and COSINESIMILARITY:

$$\boldsymbol{x}_n = \operatorname*{arg\,max}_{\boldsymbol{x} \in \mathcal{S}} \mathrm{H}[p(\boldsymbol{x}; \widehat{\boldsymbol{\theta}}_{n-1})] \cdot \left( \angle_{\boldsymbol{\phi}_{n-1}}(\boldsymbol{x}, \mathcal{A}) \right)^{\beta}$$

where β > 0 controls the relative importance of both terms. We set β = 1 in our experiments.

<span id="page-50-0"></span>Figure 12: Comparison of "undirected" baselines for the experiment of Figure 4. In the MNIST experiment, UNSA and UNDIRECTED ITL coincide, and we therefore only plot the latter.

<span id="page-50-1"></span>Figure 13: Imbalanced  $\mathcal{P}_{\mathcal{S}}$  experiment.

<span id="page-50-2"></span>Figure 14: Imbalanced  $A \sim \mathcal{P}_{\mathcal{A}}$  experiment.

<span id="page-51-2"></span>Figure 15: Performance of VTL & choice of k in the CIFAR-100 experiment.

#### <span id="page-51-0"></span>J.6 Additional experiments

We conduct the following additional experiments:

- 1. Imbalanced  $\mathcal{P}_{\mathcal{S}}$  (Figure 13): We artificially remove 80% of the support of  $\mathcal{P}_{\mathcal{A}}$  from  $\mathcal{P}_{\mathcal{S}}$ . For example, in case of MNIST, we remove 80% of the images with labels 3, 6, and 9 from  $\mathcal{P}_{\mathcal{S}}$ . This makes the learning task more difficult, as  $\mathcal{P}_{\mathcal{A}}$  is less represented in  $\mathcal{P}_{\mathcal{S}}$ , meaning that the "targets" are more sparse. The trend of ITL outperforming CTL which outperforms RANDOM is even more pronounced in this setting.
- 2. Imbalanced  $A \sim \mathcal{P}_{\mathcal{A}}$  (Figure 14): We artificially remove 50% of part of the support of  $\mathcal{P}_{\mathcal{A}}$  while generating  $A \sim \mathcal{P}_{\mathcal{A}}$  to evaluate the robustness of ITL and CTL in presence of an imbalanced target space  $\mathcal{A}$ . Concretely, in case of MNIST, we remove 50% of the images with labels 3 and 6 from A. In case of CIFAR-100, we remove 50% of the images with labels  $\{0,\ldots,4\}$  from A. We still observe the same trends as in the other experiments.
- 3. VTL & choice of k (Figure 15): We observe that VTL performs almost as well as ITL. Additionally, we evaluate the effect of the number of points k at which the decision rule is evaluated. Not surprisingly, we observe that the performance of ITL, VTL, and CTL improves with larger k.
- 4. Choice of m (Figure 16): Next, we evaluate the choice of m, i.e., the size of the target space  $\mathcal{A}$  relative to the number M of candidate points  $A \sim \mathcal{P}_{\mathcal{A}}$ . We write p = m/M. We generally observe that a larger p leads to better performance (with p=1 being the best choice). However, it appears that a smaller p can be beneficial with respect to accuracy when a large number of batches are selected. We believe that this may be because a smaller p improves the diversity between selected batches.
- 5. Choice of M (Figure 17): Finally, we evaluate the choice of M, i.e., the size of  $A \sim \mathcal{P}_A$ . Not surprisingly, we observe that the performance of ITL improves with larger M.

#### <span id="page-51-1"></span>J.7 Ablation study of noise standard deviation $\rho$

In Table 2, we evaluate the CIFAR-100 experiment with different noise standard deviations  $\rho$ . We observe that the performance of batch selection via conditional embeddings drops (mostly for the less numerically stable gradient embeddings) if  $\rho$  is too small, since this leads to numerical inaccuracies when computing the conditional embeddings. Apart from this, the effect of  $\rho$  is negligible.

<span id="page-52-1"></span>Figure 16: Evaluation of the choice of m relative to the size M of  $A \sim \mathcal{P}_{A}$ . Here, p = m/M.

<span id="page-52-2"></span>Figure 17: Evaluation of the choice of M, i.e., the size of  $A \sim \mathcal{P}_{\mathcal{A}}$ , in the CIFAR-100 experiment.

<span id="page-52-0"></span>Table 2: Ablation study of noise standard deviation  $\rho$  in the CIFAR-100 experiment. We list the accuracy after 100 rounds per decision rule, with its standard error over 10 random seeds. "(top-b)" denotes variants where batches are selected by taking the top-b points according to the decision rule rather than using batch selection via conditional embeddings. Shown in **bold** are the best performing decision rules, and shown in *italics* are results due to numerical instability.

| ρ                  | 0.0001           | 0.01             | 1                                  | 100                                           |
|--------------------|------------------|------------------|------------------------------------|-----------------------------------------------|
| G-ITL              | $78.26 \pm 1.40$ | $79.12 \pm 1.19$ | $\textbf{87.16} \pm \textbf{0.29}$ | $\overline{\textbf{87.18} \pm \textbf{0.28}}$ |
| L-ITL              | $87.52 \pm 0.48$ | $87.52 \pm 0.41$ | $87.53 \pm 0.35$                   | $86.47 \pm 0.22$                              |
| G-CTL              | $58.68 \pm 2.11$ | $81.44 \pm 1.04$ | $86.52 \pm 0.44$                   | $86.92 \pm 0.56$                              |
| L-CTL              | $86.40 \pm 0.71$ | $86.38 \pm 0.75$ | $86.00 \pm 0.69$                   | $84.78 \pm 0.39$                              |
| G-ITL (top-b)      | $85.84 \pm 0.54$ | $85.92 \pm 0.52$ | $85.84 \pm 0.54$                   | $85.55 \pm 0.46$                              |
| L-ITL (top-b)      | $85.44 \pm 0.58$ | $85.46 \pm 0.54$ | $85.44 \pm 0.59$                   | $85.29 \pm 0.36$                              |
| G-CTL (top-b)      | $82.27 \pm 0.67$ | $82.27 \pm 0.67$ | $82.27 \pm 0.67$                   | $82.27 \pm 0.67$                              |
| L-CTL (top-b)      | $80.73 \pm 0.68$ | $80.73 \pm 0.68$ | $80.73 \pm 0.68$                   | $80.73 \pm 0.68$                              |
| BADGE              | $83.24 \pm 0.60$ | $83.24 \pm 0.60$ | $83.24 \pm 0.60$                   | $83.24 \pm 0.60$                              |
| InformationDensity | $79.24 \pm 0.51$ | $79.24 \pm 0.51$ | $79.24 \pm 0.51$                   | $79.24 \pm 0.51$                              |
| RANDOM             | $82.49 \pm 0.66$ | $82.49 \pm 0.66$ | $82.49 \pm 0.66$                   | $82.49 \pm 0.66$                              |

<span id="page-53-3"></span>Figure 18: We perform the tasks of Figure 5 using Thompson sampling to evaluate the stochastic target space  $\mathcal{P}_{An}$ . We additionally compare to GOOSE (cf. Appendix K.2.3) and ISE-BO (cf. Appendix K.2.4).

# <span id="page-53-2"></span>K Additional Safe BO Experiments & Details

In Appendix K.1, we discuss the use of stochastic target spaces in the safe BO setting. We provide a comprehensive overview of prior works in Appendix K.2 and an additional experiment highlighting that ITL, unlike SAFEOPT, is able to "jump past local barriers" in Appendix K.3. In Appendix K.4, we provide details on the experiments from Figure 5.

#### <span id="page-53-0"></span>K.1 A More Exploitative Stochastic Target Space

Alternatively to the target space  $A_n$  which comprises all potentially optimal points, we evaluate the stochastic target space

<span id="page-53-1"></span>
$$\mathcal{P}_{\mathcal{A}n}(\cdot) = \mathbb{P}(\underset{\boldsymbol{x} \in \mathcal{X}: g(\boldsymbol{x}) \ge 0}{\arg \max} f(\boldsymbol{x}) = \cdot \mid \mathcal{D}_n)$$
(47)

which effectively weights points in  $\mathcal{A}_n$  according to how likely they are to be the safe optimum, and is therefore more exploitative than the uniformly-weighted target space discussed so far. Samples from  $\mathcal{P}_{\mathcal{A}n}$  can be obtained efficiently via Thompson sampling (Thompson, 1933; Russo et al., 2018). Observe that  $\mathcal{P}_{\mathcal{A}n}$  is supported precisely on the set of potential maximizers  $\mathcal{A}_n$ . We provide a formal analysis of stochastic target spaces in Appendix E. Whether transductive active learning with  $\mathcal{A}_n$  or  $\mathcal{P}_{\mathcal{A}n}$  performs better is task-dependent, as we will see in the following.

Note that performing ITL with this target space is analogous to output-space entropy search (Wang & Jegelka, 2017). Samples from  $\mathcal{P}_{An}$  can be obtained via Thompson sampling (Thompson, 1933; Russo et al., 2018). That is, in iteration n+1, we sample  $K \in \mathbb{N}$  independent functions  $f^{(j)} \sim f \mid \mathcal{D}_n$  from the posterior distribution and select K points  $\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(K)}$  which are a safe maximum of  $f^{(1)}, \ldots, f^{(K)}$ , respectively.

**Experiments** In Figure 18, we contrast the performance of ITL with  $\mathcal{P}_{An}$  to the performance of ITL with the exact target space  $\mathcal{A}_n$ . We observe that their relative performance is instance dependent: in tasks that require more difficult expansion, ITL with  $\mathcal{A}_n$  converges faster, whereas in simpler tasks (such as the 2d experiment), ITL with  $\mathcal{P}_{An}$  converges faster. We compare against the GoOSE algorithm (Turchetta et al., 2019) which is a heuristic extension of SAFEOPT that explores more greedily in directions of (assumed) high reward (cf. Appendix K.2.3). GoOSE suffers from the same limitations as SAFEOPT, which were highlighted in Section 5, and additionally is limited by its heuristic approach to expansion which fails in the 1d task and safe controller tuning task. Analogously to our experiments with SAFEOPT, we also compare against ORACLE GOOSE which has oracle knowledge of the true Lipschitz constants.

The different behaviors of ITL with  $A_n$  and  $\mathcal{P}_{A_n}$ , respectively, as well as SAFEOPT and GOOSE are illustrated in Figure 19. We observe that ITL with  $A_n$  and SAFEOPT expand the safe set more "uniformly" since the set of potential maximizers encircles the true safe set.<sup>21</sup> Intuitively, this is because the set of potential maximizers *conservatively* captures migh points might be safe and

<span id="page-53-4"></span><sup>&</sup>lt;sup>21</sup>This is because typically, there will always remain points in  $\widehat{S}_n \setminus S_n$  of which the safety cannot be fully determined, and since, they cannot be observed, it can also not be ruled out that they have high objective value.

optimal. In contrast, ITL with  $\mathcal{P}_{An}$  and GoOSE focus exploration and expansion in those regions where the objective is likely to be high.

<span id="page-54-1"></span>Figure 19: The first 100 samples of (A) ITL with  $\mathcal{A}_n$ , (B) SAFEOPT, (C) ORACLE SAFEOPT, (D) ITL with  $\mathcal{P}_{\mathcal{A}_n}$ , (E) GOOSE, (F) ORACLE GOOSE. The white region denotes the pessimistic safe set  $\mathcal{S}_{100}$ , the light gray region denotes the true safe set  $\mathcal{S}^{\star}$  (i.e., the "island"), and the darker gray regions denotes unsafe points (i.e., the "ocean").

#### <span id="page-54-0"></span>**K.2** Detailed Comparison with Prior Works

The most widely used method for Safe BO is SAFEOPT (Sui et al., 2015; Berkenkamp et al., 2021) which keeps track of separate candidate sets for expansion and exploration and uses UNSA to pick one of the candidates in each round. Treating expansion and exploration separately, sampling is directed towards expansion in all directions — even those that are known to be suboptimal. The safe set is expanded based on a Lipschitz constant of  $g^*$ , which is assumed to be known. In most real-world settings, this constant is unknown and has to be estimated using the GP. This estimate is generally conservative and results in suboptimal performance. To this end, Berkenkamp et al. (2016) proposed HEURISTIC SAFEOPT which relies solely on the confidence intervals of q to expand the safe set, but lacks convergence guarantees. More recently, Bottero et al. (2022) proposed ISE which queries parameters from  $S_n$  that yield the most "information" about the safety of another parameter in  $\mathcal{X}$ . Hence, ISE focuses solely on the expansion of the safe set  $\mathcal{S}_n$  and does not take into account the objective f. In practice, this can lead to significantly worse performance on the simplest of problems (cf. Figure 5). In contrast, ITL balances expansion of and exploration within the safe set. Furthermore, ISE does not have known convergence guarantees of the kind of Theorem 5.1. In parallel independent work, Bottero et al. (2024) proposed a combination of ISE and max-value entropy search (Wang & Jegelka, 2017) for which they derive a similar guarantee to Theorem 5.1.<sup>22</sup> Similar to SAFEOPT, their method aims to expand the safe set in all directions including those that are known to be suboptimal. In contrast, ITL directs expansion only towards potentially optimal regions.

In the 1d task and quadcopter experiment (cf. Figure 5), we observe that SAFEOPT and even ORACLE SAFEOPT converge significantly slower than ITL to the safe optima. We believe this is due to their conservative Lipschitz-continuity/global smoothness-based expansion, as opposed to ITL's expansion,

<span id="page-54-2"></span><sup>&</sup>lt;sup>22</sup>We provide an empirical evaluation in Appendix K.2.4.

which adapts to the local smoothness of the constraints. HEURISTIC SAFEOPT, which does not rely on the Lipschitz constant for expansion, does not efficiently expand the safe set due to its heuristic that only considers single-step expansion. This is especially the case for the 1d task. Furthermore, in the 2d task, we notice the suboptimality of ISE since it does not take into account the objective, and purely aims to expand the safe set. ITL, on the other hand, balances expansion and exploration.

#### K.2.1 SAFEOPT

SAFEOPT (Sui et al., 2015; Berkenkamp et al., 2021) is a well-known algorithm for Safe BO.

**Lipschitz-based expansion** SAFEOPT expands the set of known-to-be safe points by assuming knowledge of an upper bound  $L_i$  to the Lipschitz constant of the unknown constraints  $g_i^{\star}$ .<sup>23</sup> In each iteration, the (pessimistic) safe set  $S_n$  is updated to include all points which can be reached safely (with respect to the Lipschitz continuity) from a known-to-be-safe point  $x \in S_n$ . Formally,

<span id="page-55-1"></span>
$$S_n^{\text{SAFEOPT}} \stackrel{\text{def}}{=} \bigcup_{\boldsymbol{x} \in \mathcal{S}_{n-1}^{\text{SAFEOPT}}} \{ \boldsymbol{x'} \in \mathcal{X} \mid$$

$$l_{n,i}(\boldsymbol{x}) - L_i \| \boldsymbol{x} - \boldsymbol{x'} \|_2 \ge 0 \text{ for all } i \in \mathcal{I}_s \}.$$

$$(48)$$

The expansion of the safe set is illustrated in Figure 20.

We remark two main limitations of this approach. First, the Lipschitz constant is an additional safety critical hyperparameter of the algorithm, which is typically not known. The RKHS assumption (cf. Assumption C.25) induces an assumption on the Lipschitz continuity, however, the worst-case a-priori Lipschitz constant is typically very large, and prohibitive for expansion. Second, the Lipschitz constant is global property of the unknown function, meaning that it does not adapt to the local smoothness. For example, a constraint may be "flat" in one direction (permitting straightforward expansion) and "steep" in another direction (requiring slow expansion). Furthermore, the Lipschitz constant is constant over time, whereas ITL is able to adapt to the local smoothness and reduce the (induced) Lipschitz constant over time.

**Undirected expansion** SAFEOPT addresses the trade-off between expansion and exploration by focusing learning on two different sets. First, the set of *maximizers* 

$$\mathcal{M}_{n}^{\text{SafeOpt}} \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \mathcal{S}_{n}^{\text{SafeOpt}} \mid \\ u_{n,f}(\boldsymbol{x}) \geq \max_{\boldsymbol{x}' \in \mathcal{S}_{n}^{\text{SafeOpt}}} l_{n,f}(\boldsymbol{x}) \}$$

which contains all *known-to-be-safe* points which are potentially optimal. Note that if  $\mathcal{S}_n^{\text{SAFEOPT}} = \mathcal{S}_n$  then  $\mathcal{M}_n^{\text{SAFEOPT}} \subseteq \mathcal{A}_n$  since  $\mathcal{A}_n$  contains points which are potentially optimal and potentially safe but possibly unsafe.

To facilitate expansion, for each point  $x \in S_n$ , the algorithm considers a set of *expanding points* 

$$\mathcal{F}_n^{\text{SafeOPT}}(\boldsymbol{x}) \stackrel{\text{def}}{=} \{\boldsymbol{x'} \in \mathcal{X} \setminus \mathcal{S}_n^{\text{SafeOPT}} \mid u_{n,i}(\boldsymbol{x}) - L_i \|\boldsymbol{x} - \boldsymbol{x'}\|_2 \geq 0 \text{ for all } i \in \mathcal{I}_s \}$$

A point is expanding if it is unsafe initially and can be (optimistically) deduced as safe by observing x. The set of *expanders* corresponds to all known-to-be-safe points which optimistically lead to expansion of the safe set:

$$\mathcal{G}_n^{\text{SafeOpt}} \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \mathcal{S}_n \mid |\mathcal{F}_n(\boldsymbol{x})| > 0 \}.$$

That is, an expander is a safe point x which is "close" to at least one expanding point x'. Observe that here, we start with a safe x and then find a close and potentially safe x' using the Lipschitz-property of the constraint function. Thus, the set of expanding points is inherently limited by the assumed Lipschitzness (cf. Figure 20), and generally a subset of the potential expanders  $\mathcal{E}_n$  (cf. Equation (27)):

<span id="page-55-2"></span>**Lemma K.1.** For any  $n \geq 0$ , if  $S_n^{\text{SAFEOPT}} = S_n$  then

$$\bigcup_{\boldsymbol{x} \in \mathcal{S}_n} \mathcal{F}_n^{\textsf{SAFEOPT}}(\boldsymbol{x}) \subseteq \mathcal{E}_n.$$

<span id="page-55-0"></span><sup>&</sup>lt;sup>23</sup>Recall that due to the assumption that  $\|g_i^\star\|_k < \infty$ ,  $g_i^\star$  is indeed Lipschitz continuous.

Figure 20: Illustration of the expansion of the safe set à la SAFEOPT. Here, the blue region denotes the pessimistic safe set  $\mathcal{S}$ , the red region denotes the true safe set  $\mathcal{S}^*$ , and the orange region denotes the optimistic safe set  $\widehat{\mathcal{S}}$ . Whereas ITL learns about the point x' directly, SAFEOPT expands the safe set using the reduction of uncertainty at x, and then extrapolating using the Lipschitz constant (cf. Equation (48)). The dashed orange line denotes the expanding points of SAFEOPT which under-approximate the optimistic safe set of ITL (cf. Lemma K.1). Thus, ITL may even learn about points in  $\widehat{\mathcal{S}}$  which are "out of reach" for SAFEOPT.

<span id="page-56-0"></span>*Proof.* Without loss of generality, we consider the case where  $\mathcal{I}_s = \{i\}$ . We have

$$\mathcal{E}_n = \widehat{\mathcal{S}}_n \setminus \mathcal{S}_n = \{ \boldsymbol{x} \in \mathcal{X} \setminus \mathcal{S}_n \mid u_{n,i}(\boldsymbol{x}) \geq 0 \}.$$

The result follows directly by observing that  $L_i \| \mathbf{x} - \mathbf{x'} \|_2 \ge 0$ .

SAFEOPT then selects  $x_{n+1}$  according to uncertainty sampling within the maximizers and expanders:  $\mathcal{M}_n^{\text{SAFEOPT}} \cup \mathcal{G}_n^{\text{SAFEOPT}}$ . We remark that due to the separate handling of expansion and exploration, SAFEOPT expands the safe set in all directions — even those that are known to be suboptimal. In contrast, ITL only expands the safe set in directions that are potentially optimal by balancing expansion and exploration through the single set of potential maximizers  $\mathcal{A}_n$ .

**Based on uncertainty sampling** As mentioned in the previous paragraph, SAFEOPT selects as next point the maximizer/expander with the largest prior uncertainty. <sup>24</sup> In contrast, ITL selects the point within  $\mathcal{S}_n$  which minimizes the posterior uncertainty within  $\mathcal{A}_n$ . Note that the two approaches are not identical as typically  $\mathcal{M}_n^{\text{SAFEOPT}} \cup \mathcal{G}_n^{\text{SAFEOPT}} \subset \mathcal{S}_n^{\text{SAFEOPT}}$  and  $\mathcal{A}_n \not\supseteq \mathcal{S}_n$ .

We show empirically in Section 3.3 that depending on the kernel choice (i.e., the smoothness assumptions), uncertainty sampling within a given target space neglects higher-order information that can be attained by sampling outside the set. This can be seen even more clearly when considering linear functions, in which case points outside the maximizers and expanders can be equally informative as points inside.

Finally, note that the set of expanders is constructed "greedily", i.e., only considering *single-step* expansion. This is necessitated as the inference of safety is based on single reference points. Instead, ITL directly quantifies the information gained towards the points of interest without considering intermediate reference points.

**Requires homoscedastic noise** SAFEOPT imposes a homoscedasticity assumption on the noise which is an artifact of the analysis of uncertainty sampling. It is well known that in the presence of heteroscedastic noise, one has to distinguish epistemic and aleatoric uncertainty. Uncertainty sampling fails because it may continuously sample a high variance point where the variance is dominated by aleatoric uncertainty, potentially missing out on reducing epistemic uncertainty at points with small aleatoric uncertainty. In contrast, maximizing mutual information naturally takes

<span id="page-56-1"></span><sup>&</sup>lt;sup>24</sup>The use of uncertainty sampling for safe sequential decision-making goes back to Schreiter et al. (2015) and Sui et al. (2015).

into account the two sources of uncertainty, preferring those points where epistemic uncertainty is large and aleatoric uncertainty is small (cf. Appendix C.1).

**Suboptimal reachable safe set** Sui et al. (2015) and Berkenkamp et al. (2021) show that SAFEOPT converges to the optimum within the closure  $\bar{\mathcal{R}}^{\text{SAFEOPT}}_{\epsilon}(\mathcal{S}_0)$  of

$$\mathcal{R}_{\epsilon}^{\text{SAFEOPT}}(\mathcal{S}) \stackrel{\text{def}}{=} \mathcal{S} \cup \{ \boldsymbol{x} \in \mathcal{X} \mid \exists \boldsymbol{x'} \in \mathcal{S} \text{ such that } f_{i}^{\star}(\boldsymbol{x'}) - (L_{i}||\boldsymbol{x} - \boldsymbol{x'}||_{2} + \epsilon) \geq 0 \text{ for all } i \in \mathcal{I}_{s} \}.$$

Note that analogously to the expansion of the safe set, the "expansion" of the reachable safe set is based on "inferring safety" through a reference point in S and using Lipschitz continuity. This is opposed to the reachable safe set of ITL (cf. Definition C.29).

We remark that under the additional assumption that a Lipschitz constant is known, ITL can easily be extended to expand its safe set based on the kernel *and* the Lipschitz constant, resulting in a strictly larger reachable safe set than SAFEOPT. We leave the concrete formalization of this extension to future work. Moreover, we do not evaluate this extension in our experiments, as we observe that even without the additional assumption of a Lipschitz constant, ITL outperforms SAFEOPT in practice.

#### K.2.2 HEURISTIC SAFEOPT

Berkenkamp et al. (2016) also implement a heuristic variant of SAFEOPT which does not assume a known Lipschitz constant. This heuristic variant uses the same (pessimistic) safe sets  $\mathcal{S}_n$  as ITL. The set of maximizers is identical to SAFEOPT. As expanders, the heuristic variant considers all safe points  $x \in \mathcal{S}_n$  that if x were to be observed next with value  $u_n(x)$  lead to  $|\mathcal{S}_{n+1}| > |\mathcal{S}_n|$ . We refer to this set as  $\mathcal{G}_n^{\text{H-SAFEOPT}}$ . The next point is then selected by uncertainty sampling within  $\mathcal{M}_n^{\text{SAFEOPT}} \cup \mathcal{G}_n^{\text{H-SAFEOPT}}$ .

The heuristic variant shares some properties with SAFEOPT, such that it is based on uncertainty sampling, not adapting to heteroscedastic noise, and separate notions of maximizers and expanders (leading to an "undirected" expansion of the safe set). Note that there are no known convergence guarantees for heuristic SAFEOPT. Importantly, note that similar to SAFEOPT the set of expanders is constructed "greedily", and in particular, does only take into account *single-step* expansion. In contrast, an objective such as ITL which quantifies the "information gained towards expansion" also actively seeks out *multi-step* expansion.

#### <span id="page-57-0"></span>K.2.3 GOOSE

To address the "undirected" expansion of SAFEOPT discussed in the previous section, Turchetta et al. (2019) proposed *goal-oriented safe exploration* (GOOSE). GOOSE extends any unsafe BO algorithm (which we subsequently call an oracle) to the safe setting. In our experiments, we evaluate GOOSE-UCB which uses UCB as oracle and which is also the variant studied by Turchetta et al. (2019). In the following, we assume for ease of notation that  $\mathcal{I}_s = \{c\}$ .

Given the oracle proposal  $x^*$ , GOOSE first determines whether  $x^*$  is safe. If  $x^*$  is safe,  $x^*$  is queried next. Otherwise, GOOSE first learns about the safety of  $x^*$  by querying "expansionist" points until the oracle's proposal is determined to be either safe or unsafe.

GOOSE expands the safe set identically to SAFEOPT according to Equation (48). In the context of GOOSE,  $S_n^{\text{SAFEOPT}}$  is called the *pessimistic safe set*. To determine that a point cannot be deduced as safe, GOOSE also keeps track of a Lipschitz-based *optimistic safe set*:

$$\begin{aligned} \widehat{\mathcal{S}}_{n,\epsilon}^{\text{GOOSE}} & \stackrel{\text{def}}{=} \bigcup_{\bm{x} \in \mathcal{S}_{n-1}^{\text{SAFEOPT}}} \{\bm{x'} \in \mathcal{X} \mid \ & u_{n,c}(\bm{x}) - L_c \|\bm{x} - \bm{x'}\|_2 - \epsilon \geq 0 \}. \end{aligned}$$

We summarize the algorithm in Algorithm 2 where we denote by  $\mathcal{O}(\mathcal{X})$  the oracle proposal over the domain  $\mathcal{X}$ .

It remains to discuss the heuristic used to select the "expansionist" points. GoOSE considers all points  $x \in \mathcal{S}_n^{\text{SAFEOPT}}$  with confidence bands of size larger than the accuracy  $\epsilon$ , i.e.,

$$\mathcal{W}_{n,\epsilon}^{\text{GOOSE}} \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \mathcal{S}_n^{\text{SAFEOPT}} \mid u_{n,c}(\boldsymbol{x}) - l_{n,c}(\boldsymbol{x}) > \epsilon \}.$$

#### <span id="page-58-1"></span>**Algorithm 2** GOOSE

```
Given: Lipschitz constant L_c, prior model \{f,g_c\}, oracle \mathcal{O}, and precision \epsilon

Set initial safe set \mathcal{S}_0^{\mathsf{SAFEOPT}} based on prior \widehat{\mathcal{S}}_{n,\epsilon}^{\mathsf{GOOSE}} \leftarrow \mathcal{X}

n \leftarrow 0

for k from 1 to \infty do x_k^{\star} \leftarrow \mathcal{O}(\widehat{\mathcal{S}}_{n,\epsilon}^{\mathsf{GOOSE}})

while x_k^{\star} \not\in \mathcal{S}_n^{\mathsf{SAFEOPT}} do Observe "expansionist" point x_{n+1}, set n \leftarrow n+1, and update model and safe sets end while Observe x_k^{\star}, set n \leftarrow n+1, and update model and safe sets end for
```

Which of the points in this set is evaluated depends on a set of learning targets  $\mathcal{A}_{n,\epsilon}^{\text{GOOSE}} \stackrel{\text{def}}{=} \widehat{\mathcal{S}}_{n,\epsilon}^{\text{GOOSE}} \setminus \mathcal{S}_n^{\text{SAFEOPT}}$  akin to the "potential expanders"  $\mathcal{E}_n$  (cf. Equation (27)), to each of which we assign a priority h(x). When h(x) is large, this indicates that the algorithm is prioritizing to determine whether x is safe. We use as heuristic the negative  $\ell_1$ -distance between x and  $x^*$ . GOOSE then considers the set of potential immediate expanders

$$\mathcal{G}_{n,\epsilon}^{\text{GoOSE}}(\alpha) \stackrel{\text{def}}{=} \{ \boldsymbol{x} \in \mathcal{W}_{n,\epsilon}^{\text{GoOSE}} \mid \exists \boldsymbol{x'} \in \mathcal{A}_{n,\epsilon}^{\text{GoOSE}} \text{ with }$$
 priority  $\alpha$  such that  $u_{n,c}(\boldsymbol{x}) - L_c \|\boldsymbol{x} - \boldsymbol{x'}\|_2 \geq 0 \}.$ 

The "expansionist" point selected by GOOSE is then any point in  $\mathcal{G}_{n,\epsilon}^{\text{GOOSE}}(\alpha^{\star})$  where  $\alpha^{\star}$  denotes the largest priority such that  $|\mathcal{G}_{n,\epsilon}^{\text{GOOSE}}(\alpha^{\star})| > 0$ .

We observe empirically that the sample complexity of GOOSE is not always better than that of SAFEOPT. Notably, the expansion of the safe set is based on a "greedy" heuristic. Moreover, determining whether a single oracle proposal  $x^*$  is safe may take significant time. Consider the (realistic) example where the prior is uniform, and UCB proposes a point which is far away from the safe set and suboptimal. GOOSE will typically attempt to derive the safety of the proposed point until the uncertainty at *all* points within  $S_0^{SAFEOPT}$  is reduced to  $\epsilon$ . Thus, GOOSE can "waste" a significant number of samples, aiming to expand the safe set towards a known-to-be suboptimal point. In larger state spaces, due to the greedy nature of the expansion strategy, this can lead to GOOSE being effectively stuck at a suboptimal point for a significant number of rounds.

#### <span id="page-58-0"></span>K.2.4 ISE and ISE-BO

Recently, Bottero et al. (2022) proposed an information-theoretic approach to efficiently expand the safe set which they call *information-theoretic safe exploration* (ISE). Specifically, they choose the next action  $x_n$  by approximating

$$\underset{\boldsymbol{x} \in \mathcal{S}_{n-1}}{\operatorname{arg}} \max \underbrace{\max_{\boldsymbol{x'} \in \mathcal{X}} \mathrm{I}(\mathbb{1}\{g_{\boldsymbol{x'}} \geq 0\}; y_{\boldsymbol{x}} \mid \mathcal{D}_{n-1})}_{\alpha^{\mathrm{ISE}}(\boldsymbol{x})}. \tag{ISE}$$

In a parallel independent work, Bottero et al. (2024) extended ISE to the Safe BO problem where they propose to choose  $x_n$  according to

$$\underset{\boldsymbol{x} \in S}{\arg \max} \max \{\alpha^{\text{ISE}}(\boldsymbol{x}), \alpha^{\text{MES}}(\boldsymbol{x})\}$$
 (ISE-BO)

where  $\alpha^{\text{MES}}$  denotes the acquisition function of max-value entropy search (Wang & Jegelka, 2017). Similarly to SAFEOPT, ISE-BO treats expansion and exploration separately, which leads to "undirected" expansion of the safe set. That is, the safe set is expanded in all directions, even those that are known to be suboptimal. In contrast, ITL balances expansion and exploration through the single set of potential maximizers  $\mathcal{A}_n$ . With a stochastic target space, ITL generalizes max-value entropy search (cf. Appendix K.1).

We evaluate ISE-BO in Figure 18 and observe that it does not outperform ITL and VTL in any of the tasks, while performing poorly in the 1d task and suboptimally in the 2d task.

<span id="page-58-2"></span><sup>&</sup>lt;sup>25</sup>This is because the proposed point typically remains in the optimistic safe set when it is sufficiently far away from the pessimistic safe set.

<span id="page-59-1"></span>Figure 21: The ground truth  $f^*$  is shown as the dashed black line. The solid black line denotes the constraint boundary. The GP prior is given by a linear kernel with sin-transform and mean 0.1x. The light gray region denotes the initial optimistic safe set  $\widehat{\mathcal{S}}_0$  and the dark gray region denotes the initial pessimistic safe set  $\mathcal{S}_0$ .

<span id="page-59-2"></span>Figure 22: First 100 samples of ITL using the potential expanders  $\mathcal{E}_n$  (cf. Equation (27)) as target space (left) and SAFEOPT sampling only from the set of expanders  $\mathcal{G}_n^{\text{SAFEOPT}}$  (right).

#### <span id="page-59-0"></span>K.3 Jumping Past Local Barriers

In this additional experiment we demonstrate that ITL is able to extrapolate safety beyond local unsafe "barriers", which is a fundamental limitation of Lipschitz-based methods such as SAFEOPT. We consider the ground truth function and prior statistical model shown in Figure 21. Note that initially, there are three disjoint safe "regions" known to the algorithm corresponding to two of the three safe "bumps" of the ground truth function. In this experiment, the main challenge is to "jump past" the local barrier separating the leftmost and initially unknown safe "bump".

Figure 22 shows the sampled points during the first 100 iterations of SAFEOPT and ITL. Clearly, SAFEOPT does not discover the third safe "bump" while ITL does. Indeed, it is a fundamental limitation of Lipschitz-based methods that they can never "jump past local barriers", even if the oracle Lipschitz constant were to be known and tight (i.e., locally accurate) around the barrier. This is because Lipschitz-based methods expand to the point  $\boldsymbol{x}$  based on a reference point  $\boldsymbol{x}'$ , and by definition, if  $\boldsymbol{x}$  is added to the safe set so are all points on the line segment between  $\boldsymbol{x}$  and  $\boldsymbol{x}'$ . Hence, if there is a single point on this line segment which is unsafe (i.e., a "barrier"), the algorithm will *never* expand past it. This limitation does not exist for kernel-based algorithms as expansion occurs in function space.

Moreover, note that for a non-stationary kernel such as in this example, ITL samples the "closest points" in function space rather than Euclidean space. We observe that SAFEOPT still samples "locally at the boundary" whereas ITL samples the most informative point which in this case is

<span id="page-60-1"></span>Figure 23: Ground truth and prior well-calibrated model in 1d synthetic experiment. The function serves simultaneously as objective and as constraint. The light gray region denotes the initial safe set S0.

<span id="page-60-2"></span>Figure 24: Size of S<sup>n</sup> in 1d synthetic experiment. The dashed black line denotes the size of S ⋆ . In this task, "discovering" the optimum is closely linked to expansion of the safe set, and HEURISTIC SAFEOPT fails since it does not expand the safe set sufficiently.

the local maximum of the sinusoidal function. In other words, ITL adapts to the geometry of the function. This generally leads us to believe that ITL is more capable to exploit (non-stationary) prior knowledge than distance-based methods such as SAFEOPT.

# <span id="page-60-0"></span>K.4 Experiment Details

# K.4.1 Synthetic Experiments

1d task Figure [23](#page-60-1) shows the objective and constraint function, as well as the prior. We discretize using 500 points. The main difficulty in this experiment lies in sufficiently expanding the safe set to discover the global maximum. Figure [24](#page-60-2) plots the size of the safe set S<sup>n</sup> for the compared algorithms, which in this experiment matches the achieved regret closely.

2d task We model our constraint in the form of a spherical "island" where the goal is to get a good view of the coral reef located to the north-east of the island while staying in the interior of the island during exploration (cf. Figure [25\)](#page-61-1). The precise objective and constraint functions are unknown to the agent. Hence, the agent has to gradually and safely update its belief about boundaries of the "island" and the location of the coral reef. The prior is obtained by a single observation within the center of the island [−0.5, 0.5]<sup>2</sup> . We discretize using 2 500 points.

<span id="page-61-1"></span>Figure 25: Ground truth in 2d synthetic experiment.

#### <span id="page-61-0"></span>K.4.2 Safe Controller Tuning for Quadcopter

**Modeling the real-world dynamics** We learn a feedback policy (i.e., "control gains") to compensate for inaccuracies in the initial controller. In our experiment, we model the real world dynamics and the adjusted model using the PD control feedback (Widmer et al., 2023),

$$\delta_t(\boldsymbol{x}) \stackrel{\text{def}}{=} (\boldsymbol{x}^* - \boldsymbol{x})[(\boldsymbol{s}^* - \boldsymbol{s}_t) \ (\dot{\boldsymbol{s}}^* - \dot{\boldsymbol{s}}_t)], \tag{49}$$

where  $x^*$  are the *unknown* ground truth disturbance parameters, and  $s^*$  and  $\dot{s}^*$  are the desired state and state derivative, respectively. This yields the following ground truth dynamics:

$$s_{t+1}(\mathbf{x}) = T(s_t, u_t + \delta_t(\mathbf{x})). \tag{50}$$

The feedback parameters  $\boldsymbol{x} = [\boldsymbol{x}_p \ \boldsymbol{x}_d]^{\top}$  can be split into  $\boldsymbol{x}_p$  tuning the state difference which are called *proportional parameters* and  $\boldsymbol{x}_d$  tuning the state derivative difference which are called *derivative parameters*. We use the "critical damping" heuristic to relate the proportional and derivative parameters:  $\boldsymbol{x}_d = 2\sqrt{\boldsymbol{x}_p}$ . We thus consider the restricted domain  $\mathcal{X} = [0, 20]^4$  where each dimension corresponds to the proportional feedback to one of the four rotors.

Ground truth disturbance parameters are sampled from a chi-squared distribution with one degree of freedom (i.e., the square of a standard normal distribution),  $\boldsymbol{x}_p^{\star} \sim \chi_1^2$ , and  $\boldsymbol{x}_d^{\star}$  is determined according to the critical damping heuristic.

**The learning problem** The goal of our learning problem is to move the quadcopter from its initial position  $s(0) = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}^{\top}$  (in Euclidean space with meter as unit) to position  $s^* = \begin{bmatrix} 0 & 0 & 2 \end{bmatrix}^{\top}$ . Moreover, we aim to stabilize the quadcopter at the goal position, and therefore regularize the control signal towards an action  $u^*$  which results in hovering (approximately) without any disturbances. We formalize these goals with the following objective function:

$$f^{\star}(\boldsymbol{x}) \stackrel{\text{def}}{=} -\sigma \left( \sum_{t=0}^{T} \left\| \boldsymbol{s}^{\star} - \boldsymbol{s}_{t}(\boldsymbol{x}) \right\|_{\boldsymbol{Q}}^{2} + \left\| \boldsymbol{u}^{\star} - \boldsymbol{u}_{t}(\boldsymbol{x}) \right\|_{\boldsymbol{R}}^{2} \right)$$
(51)

where  $\sigma(v) \stackrel{\text{def}}{=} \tanh((v-100)/100)$  is used to smoothen the objective function and ensure that its range is [-1,1]. The non-smoothed control objective in Equation (51) is known as a *linear-quadratic regulator* (LQR) which we solve exactly for the undisturbed system using ILQR (Tu et al., 2023). Finally, we want to ensure at all times that the quadcopter is at least 0.5 meter above the ground, that is,

<span id="page-61-2"></span>
$$g^{\star}(\boldsymbol{x}) \stackrel{\text{def}}{=} \min_{t \in [T]} \boldsymbol{s}_t^z(\boldsymbol{x}) - 0.5$$
 (52)

where we denote by  $s_t^z$  the z-coordinate of state  $s_t$ .

We use a time horizon of T=3 seconds which we discretize using 100 steps. The objective is modeled by a zero-mean GP with a Matérn( $\nu=5/2$ ) kernel with lengthscale 0.1, and the constraint is modeled by a GP with mean -0.5 and a Matérn( $\nu=5/2$ ) kernel with lengthscale 0.1. The prior is obtained by a single observation of the "safe seed"  $[0\ 0\ 0\ 10]^{\top}$ .

Adaptive discretization We discretize the domain X adaptively using coordinate LINEBO [\(Kirschner et al.,](#page-13-23) [2019\)](#page-13-23). That is, in each iteration, one of the four control dimensions is selected uniformly at random, and the active learning oracle is executed on the corresponding one-dimensional subspace.

Safety Using the (unsafe) constrained BO algorithm EIC [\(Gardner et al.,](#page-12-6) [2014\)](#page-12-6) leads constraint violation,[26](#page-62-0) while ITL and VTL do not violate the constraints during learning for any of the random seeds.

Hyperparameters The observation noise is Gaussian with standard deviation ρ = 0.1. We let β = 10. The control target is u <sup>⋆</sup> = [1.766 0 0 0]⊤.

The state space is 12-dimensional where the first three states correspond to the velocity of the quadcopter, the next three states correspond to its acceleration, the following three states correspond to its angular velocity, and the last three states correspond to its angular velocity in local frame. The LQR parameters are given by

$$\begin{aligned} & \boldsymbol{Q} = \mathrm{diag} \left\{ 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0$$

The quadcopter simulation was adapted from [Chandra](#page-11-23) [\(2023\)](#page-11-23).

Each one-dimensional subspace is discretized using 2 000 points.

Random seeds We repeat the experiment for 25 different seeds where the randomness is over the ground truth disturbance, observation noise, and the randomness in the algorithm.

<span id="page-62-0"></span><sup>26</sup>On average, 1.6 iterations of the first 50 violate the constraints.

<span id="page-63-0"></span>Table 3: Magnitudes of  $\gamma_n$  for common kernels. The magnitudes hold under the assumption that  $\mathcal X$  is compact. Here,  $B_\nu$  is the modified Bessel function. We take the magnitudes from Theorem 5 of Srinivas et al. (2009) and Remark 2 of Vakili et al. (2021). The notation  $\widetilde{O}(\cdot)$  subsumes log-factors. For  $\nu=1/2$ , the Matérn kernel is equivalent to the Laplace kernel. For  $\nu\to\infty$ , the Matérn kernel is equivalent to the Gaussian kernel. The functions sampled from a Matérn kernel are  $\lceil\nu\rceil-1$  mean square differentiable. The kernel-agnostic bound follows by simple reduction to a linear kernel in  $|\mathcal X|$  dimensions.

| Kernel   | $k(\boldsymbol{x}, \boldsymbol{x'})$                                                                                                                                                                  | $\gamma_n$                                                                    |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Linear   | $x^{\top}x'$                                                                                                                                                                                          | $O(d\log(n))$                                                                 |
| Gaussian | $\exp\left(-\frac{\left\ \boldsymbol{x}-\boldsymbol{x}'\right\ _2^2}{2h^2}\right)$                                                                                                                    | $\widetilde{O}\Big(\log^{d+1}(n)\Big)$                                        |
| Laplace  | $\exp\left(-\frac{\left\ x-x'\right\ _1}{h}\right)$                                                                                                                                                   | $\widetilde{O}\left(n^{\frac{d}{1+d}}\log^{\frac{1}{1+d}}(n)\right)$          |
| Matérn   | $\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{\sqrt{2\nu} \ \boldsymbol{x} - \boldsymbol{x'}\ _2}{h} \right)^{\nu} B_{\nu} \left( \frac{\sqrt{2\nu} \ \boldsymbol{x} - \boldsymbol{x'}\ _2}{h} \right)$ | $\widetilde{O}\left(n^{\frac{d}{2\nu+d}}\log^{\frac{2\nu}{2\nu+d}}(n)\right)$ |
| any      |                                                                                                                                                                                                       | $O( \mathcal{X} \log(n))$                                                     |

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

# Justification:

#### Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

### 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The code is publicly available.

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

### 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: See the extensive discussions in the appendices.

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

### 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: All individual experiments require only small compute resources.

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

#### Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] Justification: Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset's creators.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes] Justification: Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] Justification: Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] Justification: Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.