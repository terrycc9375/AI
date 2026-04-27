## NeuralFuse: Learning to Recover the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes

Hao-Lun Sun<sup>1</sup> , Lei Hsiung<sup>2</sup> , Nandhini Chandramoorthy<sup>3</sup> , Pin-Yu Chen<sup>3</sup> , Tsung-Yi Ho<sup>4</sup>

> <sup>1</sup> National Tsing Hua University <sup>2</sup> Dartmouth College <sup>3</sup> IBM Research <sup>4</sup> The Chinese University of Hong Kong s109062594@m109.nthu.edu.tw lei.hsiung.gr@dartmouth.edu {pin-yu.chen, nandhini.chandramoorthy}@ibm.com tyho@cse.cuhk.edu.hk

## Abstract

Deep neural networks (DNNs) have become ubiquitous in machine learning, but their energy consumption remains problematically high. An effective strategy for reducing such consumption is supply-voltage reduction, but if done too aggressively, it can lead to accuracy degradation. This is due to random bit-flips in static random access memory (SRAM), where model parameters are stored. To address this challenge, we have developed NeuralFuse, a novel add-on module that handles the energy-accuracy tradeoff in low-voltage regimes by learning input transformations and using them to generate error-resistant data representations, thereby protecting DNN accuracy in both nominal and low-voltage scenarios. As well as being easy to implement, NeuralFuse can be readily applied to DNNs with limited access, such cloud-based APIs that are accessed remotely or non-configurable hardware. Our experimental results demonstrate that, at a 1% bit-error rate, NeuralFuse can reduce SRAM access energy by up to 24% while recovering accuracy by up to 57%. To the best of our knowledge, this is the first approach to addressing low-voltage-induced bit errors that requires no model retraining.

## 1 Introduction

Energy-efficient computing is of primary importance to the effective deployment of deep neural networks (DNNs), particularly in edge devices and in on-chip AI systems. Increasing DNN computation's energy efficiency and lowering its carbon footprint require iterative efforts from both chip designers and algorithm developers. Processors with specialized hardware accelerators for AI computing, capable of providing orders of magnitude better performance and energy efficiency for AI computation, are now ubiquitous. However, alongside reduced precision/quantization and architectural optimizations, endowing such systems with the capacity for low-voltage operation is a powerful lever for reducing their power consumption.

The computer engineering literature contains ample evidence of the effects of undervolting and low-voltage operation on accelerator memories that store weights and activations during computation. Aggressive scaling-down of static random access memory's (SRAM's) supply voltage to below the rated value saves power, thanks to the quadratic dependence of dynamic power on voltage. Crucially,

Project Page: <https://trustsafeai-neuralfuse.static.hf.space> Code: <https://github.com/IBM/NeuralFuse>

<span id="page-1-0"></span>

- (a) The pipeline of the NeuralFuse framework at inference.
- (b) Energy/accuracy tradeoff example.

Figure 1: (a) At inference, NeuralFuse transforms input samples  $\mathbf x$  into robust data representations. The *nominal* voltage allows models to work as expected, whereas at *low voltage*, one would encounter bit errors (e.g., 1%) that cause incorrect inferences. The percentages reflect the accuracy of a CIFAR-10 pre-trained ResNet18 with and without NeuralFuse in both those voltage cases. (b) On the same base model (ResNet18), we illustrate the energy/accuracy tradeoff of six NeuralFuse implementations. The x-axis represents the percentage reduction in dynamic-memory access energy at low-voltage settings (base model protected by NeuralFuse), as compared to the bit-error-free (nominal) voltage. The y-axis represents the perturbed accuracy (evaluated at low voltage) with a 1% bit-error rate.

however, it also leads to an exponential increase in bit failures. Memory bit flips cause errors in the stored weight and activation values [Chandramoorthy et al., 2019, Ganapathy et al., 2017], leading to catastrophic accuracy loss.

A recent wave of research has proposed numerous techniques for allowing low-voltage operation of DNN accelerators while preserving their accuracy. Most of these have been either hardware-based error-mitigation techniques or error-aware robust training of DNN models. On-chip error mitigation methods have significant performance and power overheads [Chandramoorthy et al., 2019, Reagen et al., 2016]. On the other hand, some have proposed to generate models that are robust to bit errors via a specific learning algorithm [Kim et al., 2018, Koppula et al., 2019, Stutz et al., 2021], thereby eliminating the need for on-chip error mitigation. However, error-aware robust training to find the optimal set of robust parameters for each model is time- and energy-intensive and may not be possible in all access-limited settings.

In this paper, therefore, we propose a novel model-agnostic approach: *NeuralFuse*. This proof-of-concept machine-learning module offers trainable input transformation parameterized by a relatively small DNN; and, by enhancing input's robustness, it mitigates bit errors caused by very low-voltage operation, thus serving the wider goal of more accurate inferencing. The pipeline of NeuralFuse is illustrated in Figure 1. To protect the deployed models from making wrong predictions under low-power conditions, NeuralFuse accepts scenarios under access-limited neural networks (e.g., non-configurable hardware or cloud-based APIs). Specifically, we consider two access-limited scenarios that are common in the real world: 1) *relaxed access*, in which 'black box' model details are unknown, but backpropagation through those models is possible; and 2) *restricted access*, in which the model details are unknown and backpropagation is disallowed. To enable it to deal with relaxed access, we trained NeuralFuse via backpropagation, and for restricted-access cases, we trained it on a white-box surrogate model. To the best of our knowledge, this is the first study that leverages a learning-based method to address random bit errors as a means of recovering accuracy in low-voltage and access-limited settings.

#### We summarize our **main contributions** as follows:

- We propose *NeuralFuse*, a novel learning-based input-transformation module aimed at enhancing the accuracy of DNNs that are subject to random bit errors caused by very low voltage operation. NeuralFuse is model-agnostic, i.e., operates on a plug-and-play basis at the data-input stage and does not require any re-training of deployed DNN models.
- We explore two practical limited-access scenarios for neural-network inference: relaxed access and restricted access. In the former setting, we use gradient-based methods for module training. In the latter one, we use a white-box surrogate model for training, which is highly transferable to other types of DNN architecture.
- We report the results of an extensive program of experiments with various combinations of DNN models (ResNet18, ResNet50, VGG11, VGG16, and VGG19), datasets (CIFAR-10, CIFAR-100,

GTSRB, and ImageNet-10), and NeuralFuse implementations of different architectures and sizes. These show that NeuralFuse can consistently increase the perturbed accuracy (accuracy evaluated under random bit errors in weights) by up to 57%, while simultaneously saving up to 24% of the energy normally required for SRAM access, based on our realistic characterization of bit-cell failures for a given memory array in a low-voltage regime inducing a 0.5%/1% bit-error rate.

• We demonstrate NeuralFuse's transferability (i.e., adaptability to unseen base models), versatility (i.e., ability to recover low-precision quantization loss), and competitiveness (i.e., state-of-the-art performance) in various scenarios, establishing it as a promising proof-of-concept for energyefficient, resilient DNN inference.

## 2 Related Work and Background

Software-based Energy-saving Strategies. Various recent studies have proposed software-based methods of reducing computing's energy consumption. For instance, quantization techniques have been reported to reduce the precision required for storing model weights, and thus to decrease total memory storage [\[Gong et al.,](#page-9-2) [2014,](#page-9-2) [Rastegari et al.,](#page-10-4) [2016,](#page-10-4) [Wu et al.,](#page-10-5) [2016\]](#page-10-5). On the other hand, [Yang et al.](#page-11-0) [\[2017\]](#page-11-0) - who proposed energy-aware pruning on each layer and fine-tuning of weights to maximize final accuracy - suggested several ways to reduce DNNs' energy consumption. For example, they devised the ECC framework, which compresses DNN models to meet a given energy constraint [\[Yang et al.,](#page-10-6) [2019a\]](#page-10-6), and a method of compressing such models via joint pruning and quantization [\[Yang et al.,](#page-11-1) [2020\]](#page-11-1). It is also feasible, during DNN training, to treat energy constraints as an optimization problem and thereby reduce energy consumption while maximizing training accuracy [\[Yang et al.,](#page-10-7) [2019b\]](#page-10-7). However, unlike ours, all these methods imply changing either model architectures or model weights.

Hardware-based Energy-saving Strategies. Prior studies have also explored ways of improving energy efficiency via specially designed hardware. Several of them have focused on the undervolting of DNN accelerators and proposed methods to maintain accuracy in the presence of bit errors. For instance, [Reagen et al.](#page-10-0) [\[2016\]](#page-10-0) proposed an SRAM fault-mitigation technique that rounds faulty weights to zero to avoid degradation of prediction accuracy. [Srinivasan et al.](#page-10-8) [\[2016\]](#page-10-8) recommended storing sensitive MSBs (most significant bits) in robust SRAM cells to preserve accuracy. [Chandramoorthy](#page-9-0) [et al.](#page-9-0) [\[2019\]](#page-9-0) proposed dynamic supply-voltage boosting to improve the resilience of memory-access operations; and the learning-based approach proposed by [Stutz et al.](#page-10-3) [\[2021\]](#page-10-3) aims to find models that are robust to bit errors. The latter paper discusses several techniques for improving such robustness, notably quantization, weight-clipping, random bit-error training, and adversarial bit-error training. Its authors concluded from their experiments that a combination of quantization, weight-clipping, and adversarial bit-error training will yield excellent performance. However, they also admitted that the relevant training process was sensitive to hyperparameter settings, and hence, it might come with a challenging training procedure.

However, we suggest that all the methods mentioned above are difficult to implement and/or unsuitable for use in real-world access-limited settings. For example, the weights of DNN models packed on embedded systems may not be configurable or updatable, making model retraining (e.g., [Stutz et al.](#page-10-3) [\[2021\]](#page-10-3)) non-viable in that scenario. Moreover, DNN training is already a tedious and time-consuming task, so adding error-aware training to it may further increase its complexity and, in particular, make hyperparameter searches more challenging. [Özdenizci and Legenstein](#page-10-9) [\[2022\]](#page-10-9) also reported that error-aware training was ineffective for large DNNs with millions of bits. NeuralFuse obviates the need for model retraining via an add-on trainable input-transformation function parameterized by a relatively small secondary DNN.

SRAM Bit Errors in DNNs. Low voltage-induced memory bit-cell failures can cause bit-flips from 0 to 1 and vice versa. In practice, SRAM bit errors increase exponentially when the supply voltage is scaled below Vmin, i.e., the minimum voltage required to avoid them. This phenomenon has been studied extensively in the prior literature, including work by [Chandramoorthy et al.](#page-9-0) [\[2019\]](#page-9-0) and [Ganapathy et al.](#page-9-1) [\[2017\]](#page-9-1). The specific increases in bit errors as voltage scales down, in the case of an SRAM array of 512 × 64 bits with a 14nm technology node, is illustrated in Figure [2.](#page-3-0) The corresponding dynamic energy per SRAM read access, measured at each voltage at a constant frequency, is shown on the right-hand side of the figure. In this example, accessing the SRAM at

<span id="page-3-0"></span>Figure 2: The bit-error rates (left) and dynamic energy per memory access versus voltage for static random access memory arrays (right) as reported by [Chandramoorthy et al.](#page-9-0) [\[2019\]](#page-9-0). The x-axis shows voltages normalized with respect to the minimum bit error-free voltage (Vmin).

0.83Vmin leads to a 1% bit-error rate, and at the same time, dynamic energy per access is reduced by approximately 30%. This can lead to DNNs making inaccurate inferences, particularly when bit-flips occur at the MSBs. However, improving robustness to bit errors can allow us to lower Vmin and exploit the resulting energy savings.

It has been observed that bit-cell failures for a given memory array are randomly distributed and independent of each other. That is, the spatial distribution of bit-flips can be assumed to be random, as it generally differs from one array to another, within as well as between chips. Below, following [Chandramoorthy et al.](#page-9-0) [\[2019\]](#page-9-0), we model bit errors in a memory array of a given size by generating a random distribution of such errors with equal likelihood of 0-to-1 and 1-to-0 bit-flipping. More specifically, we assume that the model weights are quantized to 8-bit precision (i.e., from 32-bit floats to 8-bit integers), and generate perturbed models by injecting our randomly distributed bit errors into the two's complement representation of weights. For more implementation details, please refer to Section [4.1.](#page-4-0)

## 3 NeuralFuse: Framework and Algorithms

#### 3.1 Error-Resistant Input Transformation

As illustrated in Figure [1,](#page-1-0) we propose a novel trainable input-transformation module, NeuralFuse, parametrized by a relatively small DNN, to mitigate the accuracy-energy tradeoff for model inference and thus overcome the drawback of performance degradation in low-voltage regimes. A specially designed loss function and training scheme are used to derive NeuralFuse and apply it to the input data such that the transformed inputs will become robust to low voltage-induced bit errors.

Consider the input x sampled from the data distribution X and a set of models M<sup>p</sup> with p% random bit errors on weights (i.e., *perturbed* models). When it is not manifesting any bit errors (i.e., at normal-voltage settings), the perturbed model operates as a nominal deterministic one, denoted by M0. NeuralFuse aims to ensure that a model M<sup>p</sup> ∈ M<sup>p</sup> can make correct inferences on the transformed inputs while also delivering consistent results in its M<sup>0</sup> state. To adapt to various data characteristics, NeuralFuse – designated as F in Eq. [\(1\)](#page-3-1), below – is designed to be input-aware. This characteristic can be formally defined as

<span id="page-3-1"></span>
$$\mathcal{F}(\mathbf{x}) = \text{clip}_{[-1,1]}(\mathbf{x} + \mathcal{G}(\mathbf{x})), \tag{1}$$

where G(x) is a "generator" (i.e., an input-transformation function) that can generate a perturbation based on input x. As transformed by NeuralFuse, i.e., as F(x), that input is passed to the deployed model (M<sup>0</sup> or Mp) for final inference. Without loss of generality, we assume the transformed input lies within a scaled input range F(·) ∈ [−1, 1]<sup>d</sup> , where d is the (flattened) dimension of x.

#### 3.2 Training Objective and Optimizer

To train our generator G(·), which ought to be able to ensure the correctness of both the perturbed model M<sup>p</sup> and the clean model M0, we parameterized it with a neural network and apply our training objective function

$$\arg \max_{\mathcal{W}_{\mathcal{G}}} \log \mathcal{P}_{M_0}(y|\mathcal{F}(\mathbf{x}; \mathcal{W}_{\mathcal{G}})) + \lambda \cdot \mathbf{E}_{M_p \sim \mathcal{M}_p}[\log \mathcal{P}_{M_p}(y|\mathcal{F}(\mathbf{x}; \mathcal{W}_{\mathcal{G}}))], \tag{2}$$

where  $W_{\mathcal{G}}$  is the set of trainable parameters for  $\mathcal{G}$ ; y is the ground-truth label of  $\mathbf{x}$ ;  $\mathcal{P}_M$  denotes the likelihood of y as computed by a model M being given a transformed input  $\mathcal{F}(\mathbf{x}; \mathcal{W}_{\mathcal{G}})$ ;  $\mathcal{M}_p$  is the distribution of the perturbed models inherited from the clean model  $M_0$ , under a p% random bit-error rate; and  $\lambda$  is a hyperparameter that balances the importance of the nominal and perturbed models.

The training objective function can be readily converted to a loss function  $(\mathcal{L})$  that evaluates cross-entropy between the ground-truth label y and the prediction  $\mathcal{P}_M(y|\mathcal{F}(\mathbf{x};\mathcal{W}_\mathcal{G}))$ . That is, the total loss function can be calculated as

<span id="page-4-1"></span>
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{M_0} + \lambda \cdot \mathcal{L}_{\mathcal{M}_p}. \tag{3}$$

In particular, optimizing the loss function requires evaluation of the impact of the loss term  $\mathcal{L}_{\mathcal{M}_p}$  on randomly perturbed models. Our training process is inspired by expectation over transformation (EOT) attacks [Athalye et al., 2018], which aim to produce robust adversarial examples that are simultaneously adversarial over the entire transformation distribution. Based on that idea, we propose a new optimizer for solving Eq. (3), which we call expectation over perturbed models (EOPM). EOPM-trained generators can generate error-resistant input transformations and mitigate inherent bit errors. However, it would be computationally impossible to enumerate all possible perturbed models with random bit errors, and the number of realizations for perturbed models is constrained by the memory size of the GPUs used for training. In practice, therefore, we only use N perturbed models per iteration to calculate empirical average loss, i.e.,

$$\mathcal{L}_{\mathcal{M}_p} \approx \frac{\mathcal{L}_{M_{p_1}} + \dots + \mathcal{L}_{M_{p_N}}}{N},$$
 (4)

where N is the number of perturbed models  $\{M_{p_1}, \cdots, M_{p_N}\}$  that are simulated to calculate the loss caused by random bit errors. Therefore, the gradient used to update the generator can be calculated as follows:

<span id="page-4-2"></span>
$$\frac{\partial \mathcal{L}_{\text{Total}}}{\partial \mathcal{W}_{\mathcal{G}}} = \frac{\partial \mathcal{L}_{M_0}}{\partial \mathcal{W}_{\mathcal{G}}} + \frac{\lambda}{N} \left( \frac{\partial \mathcal{L}_{M_{p_1}}}{\partial \mathcal{W}_{\mathcal{G}}} + \dots + \frac{\partial \mathcal{L}_{M_{p_N}}}{\partial \mathcal{W}_{\mathcal{G}}} \right). \tag{5}$$

Through our implementation, we found that stable performance could be delivered when N=10, and that there was little to be gained by using a larger value. The results of our ablation study for different values of N can be found in Appendix E.

#### 3.3 Training Algorithm

Algorithm 1 in Appendix A summarizes NeuralFuse's training steps. Briefly, this involves splitting the training data  $\mathcal{X}$  into B mini-batches for training the generator in each epoch. For each mini-batch, we first feed these data into  $\mathcal{F}(\cdot)$  to obtain the transformed inputs. Also, we simulate N perturbed models using a p% random bit-error rate, denoted by  $M_{p_1}, \cdots, M_{p_N}$ , from  $\mathcal{M}_p$ . Then, the transformed inputs are fed into those N perturbed models as well as into the clean model  $\mathcal{M}_0$ , and their respective losses and gradients are calculated. Finally, NeuralFuse parameters  $\mathcal{W}_{\mathcal{G}}$  are updated based on the gradient obtained by EOPM.

#### 4 Experiments

#### <span id="page-4-0"></span>4.1 Experiment Setups

**Datasets.** We evaluate NeuralFuse on four different datasets: CIFAR-10 [Krizhevsky and Hinton, 2009], CIFAR-100 [Krizhevsky and Hinton, 2009], the German Traffic Sign Recognition Benchmark (GTSRB) [Stallkamp et al., 2012], and ImageNet-10 [Deng et al., 2009]. CIFAR-10 consists of 10 classes, with 50,000 training images and 10,000 testing images in total. Similarly, CIFAR-100 consists of 100 classes, with 500 training images and 100 testing images in each. The GTSRB contains 43 classes with a total of 39,209 training images and 12,630 testing images. Similar to CIFAR-10 and CIFAR-100, we resize GTSRB into 32×32×3 in our experiment. For ImageNet-10, we chose the same ten categories as Huang et al. [2022], in which there are 13,000 training images and 500 test images cropped into 224×224×3. Due to space limitations, our CIFAR-100 results are presented in Appendices F and G.

**Base Models.** We selected several common architectures for our base models: ResNet18, ResNet50 [He et al., 2016], VGG11, VGG16, and VGG19 [Simonyan and Zisserman, 2015]. To replicate the deployment of models on chips, all our based models were given quantization-aware training that followed Stutz et al. [2021].

NeuralFuse Generators. The architecture of the NeuralFuse generator (G) ) is based on an encoderdecoder structure. We designed and compared three types of generators, namely convolution-based, deconvolution-based, and UNet-based. We also considered large(L)/small(S) network sizes for each type. Further details can be found below and in Appendix [B.](#page-12-2)

- Convolution-based (Conv). Conv uses convolution with MaxPool layers for its encoder and *convolution with UpSample layers* for its decoder. This architecture has previously been shown to be efficient and effective at generating *input-aware* backdoor triggers [\[Nguyen and Tran,](#page-10-13) [2020\]](#page-10-13).
- Deconvolution-based (DeConv). DeConv uses convolution with MaxPool layers for its encoder and *deconvolution layers* for its decoder. We expected this modification both to enhance its performance and to reduce its energy consumption.
- UNet-based (UNet). UNet uses convolution with MaxPool layers for its encoder, and deconvolution layers for its decoder. UNet is known for its robust performance in image segmentation [\[Ronneberger et al.,](#page-10-14) [2015\]](#page-10-14).

Energy-consumption Calculation. The energy consumption reported in Figure [1](#page-1-0) is based on the product of the total number of SRAM memory accesses in a systolic array-based convolution neural network (CNN) accelerator and the dynamic energy per read access at a given voltage. Research by [Chen et al.](#page-9-7) [\[2016\]](#page-9-7) previously showed that energy consumption by SRAM buffers and arrays accounts for a high proportion of total system energy consumption. We assume that there are no bit errors on NeuralFuse, given its status as an add-on data preprocessing module whose functions could also be performed by a general-purpose core. In this work we assume it is implemented on the accelerator equipped with dynamic voltage scaling and therefore NeuralFuse computation is performed at nominal error-free voltage. We report a reduction in overall weight-memory energy consumption (i.e., NeuralFuse + Base Model under a p% bit-error rate) with respect to the unprotected base model in the regular-voltage mode (i.e., 0% bit-error rate and without NeuralFuse).

To quantify memory accesses, we used the SCALE-SIM simulator [\[Samajdar et al.,](#page-10-15) [2020\]](#page-10-15), and our chosen configuration simulated an output-stationary dataflow and a 32×32 systolic array with 256KB of weight memory. We collected data on the dynamic energy per read access of the SRAM both at Vmin and at the voltage corresponding to a 1% bit-error rate (Vber ≈ 0.83Vmin) from Cadence ADE Spectre simulations, both at the same clock frequency.

Relaxed and Restricted Access Settings. In the first of our experiments' two scenarios, relaxed access, the base-model information was not entirely transparent, but allowed us to obtain gradients from the black-box model through backpropagation. Therefore, this scenario allowed direct training of NeuralFuse with the base model using EOPM. In the restricted-access scenario, on the other hand, only the inference function was allowed for the base model, and we therefore trained NeuralFuse using a white-box surrogate base model and then transfering the generator to the access-restricted model.

Computing Resources. Our experiments were performed using eight Nvidia Tesla V100 GPUs and implemented with PyTorch. NeuralFuse was found to generally take 150 epochs to converge, and its training time was similar to that of the base model it incorporated. On both the CIFAR-10 and CIFAR-100 datasets, average training times were 17 hours (ResNet18), 50 hours (ResNet50), 9 hours (VGG11), 13 hours (VGG16), and 15 hours (VGG19). For GTSRB, the average training times were 9 hours (ResNet18), 27 hours (ResNet50), 5 hours (VGG11), 7 hours (VGG16), and 8 hours (VGG19); and for ImageNet-10, the average training times were 32 hours (ResNet18), 54 hours (ResNet50), 50 hours (VGG11), 90 hours (VGG16), and 102 hours (VGG19).

#### 4.2 Performance Evaluation, Relaxed-access Scenario

Our experimental results pertaining to the relaxed-access scenario are shown in Figure [3.](#page-6-0) The biterror rate (BER) due to low voltage was 1% in the cases of CIFAR-10 and GTSRB, and 0.5% for ImageNet-10. The BER of ImageNet-10 was lower than that of the other two because, being pretrained, it has more parameters than either of them. For each experiment, we sampled and evaluated N = 10 perturbed models (independent from training), and below, we report the means and standard deviations of their respective accuracies. Below, clean accuracy (CA) refers to a model's accuracy measured at nominal voltage, and perturbed accuracy (PA) to its accuracy measured at low voltage.

In the cases of CIFAR-10 and the GTSRB, we observed that large generators like ConvL and UNetL recovered PA considerably, i.e., in the range of 41% to 63% on ResNet18, VGG11, VGG16,

<span id="page-6-0"></span>Figure 3: Relaxed-access scenario test accuracies (%) of various pre-trained models with and without NeuralFuse, compared at nominal voltage (0% bit-error rate) or low voltage (with specified bit-error rates). The results demonstrate that NeuralFuse consistently recovered perturbation accuracy.

and VGG19. ResNet50's recovery percentage was slightly worse than those of the other base models, but it nevertheless attained up to 51% recovery on the GTSRB. On the other hand, the recovery percentages achieved when we used small generators like DeConvS were worse than those of their larger counterparts. This could be explained by larger-sized networks' better ability to learn error-resistant generators (though perhaps at the cost of higher energy consumption). In the case of ImageNet-10, using larger generators also yielded better PA performance recovery, further demonstrating NeuralFuse's ability to work well with large input sizes and varied datasets.

#### 4.3 Performance Evaluation, Restricted-access Scenario (Transferability)

The experimental results of our restricted-access scenario are shown in Table 1. We adopted ResNet18 and VGG19 as our white-box surrogate source models for training the generators under a 1.5% bit-error rate. We chose ConvL and UNetL as our generators because they performed best out of the six we tested (see Figure 3).

From Table 1, we can see that transferring from a larger BER such as 1.5% can endow a smaller one (e.g., 1%) with strong resilience. The table also makes it clear that using VGG19 as a surrogate model with UNet-based generators like UNetL can yield better recovery performance than other model/generator combinations. On the other hand, we observed in some cases that if we transferred between source and target models of the same type (but with different BERs for training and testing), performance results could exceed those we had obtained during the original relaxed-access scenario. For instance, when we transferred VGG19 with UNetL under a 1.5% BER to VGG19 or VGG11 under a 0.5% BER, the resulting accuracies were 85.86% (as against 84.99% for original VGG19) and 84.81% (as against 82.42% for original VGG11). We conjecture that generators trained on relatively large BERs can cover the error patterns of smaller BERs, and even help improve the latter's generalization. These findings indicate the considerable promise of access-limited base models in low-voltage settings to recover accuracy.

## <span id="page-6-1"></span>4.4 Energy/Accuracy Tradeoff

We report total dynamic energy consumption as the total number of SRAM-access events multiplied by the dynamic energy of a single such event. Specifically, we used SCALE-SIM to calculate total

<span id="page-7-0"></span>

| Table 1: | Restricted-access    | scenario: Transfer | results on CIFAR- | 10 with 1.5% bit-error ra | te  |
|----------|----------------------|--------------------|-------------------|---------------------------|-----|
| Table 1. | . INCOLLICIOU-access | SCCHAILO, FLAHSICI | TOSURS OF CITAIN  | TO WILL L.970 DIL-CHOLIA  | ··· |

| CM          | TM        | DED  | CA   | DA              | Co      | nvL (1.5%)     |      | UN      | NetL (1.5%)    |      |
|-------------|-----------|------|------|-----------------|---------|----------------|------|---------|----------------|------|
| SM          | TM        | BER  | CA   | PA              | CA (NF) | PA (NF)        | RP   | CA (NF) | PA (NF)        | RP   |
|             | ResNet18  | 1%   | 92.6 | $38.9 \pm 12.4$ | 89.8    | $89.0 \pm 0.5$ | 50.1 | 85.8    | $85.2 \pm 0.5$ | 46.3 |
|             | Resiletto | 0.5% | 92.0 | $70.1 \pm 11.6$ | 69.6    | $89.6 \pm 0.2$ | 19.5 | 65.6    | $85.7 \pm 0.2$ | 15.6 |
|             | ResNet50  | 1%   | 92.6 | $26.1 \pm 9.4$  | 89.2    | $36.1 \pm 18$  | 10.0 | 84.4    | $38.9 \pm 16$  | 12.8 |
|             | Resiretso | 0.5% | 72.0 | $61.0 \pm 10.3$ | 07.2    | $74.1 \pm 10$  | 13.1 | 04.4    | $72.7 \pm 4.6$ | 11.7 |
| ResNet18    | VGG11     | 1%   | 88.4 | $42.2 \pm 11.6$ | 86.3    | $59.2 \pm 10$  | 17.0 | 82.3    | $69.8 \pm 7.5$ | 27.6 |
| 11051 (0110 | , 5511    | 0.5% |      | $63.6 \pm 9.3$  | 00.5    | $78.9 \pm 4.9$ | 15.3 | 02.0    | $77.0 \pm 4.0$ | 13.4 |
|             | VGG16     | 1%   | 90.3 | $35.7 \pm 7.9$  | 89.4    | $62.2 \pm 18$  | 26.5 | 84.7    | $68.9 \pm 14$  | 33.2 |
|             |           | 0.5% |      | $66.6 \pm 8.1$  |         | $83.4 \pm 5.5$ | 16.8 |         | $80.5 \pm 5.9$ | 13.9 |
|             | VGG19     | 1%   | 90.5 | $36.0 \pm 12.0$ | 89.8    | $49.9 \pm 23$  | 13.9 | 85.0    | $55.1 \pm 17$  | 19.1 |
|             |           | 0.5% |      | $64.2 \pm 12.4$ |         | $81.8 \pm 8.5$ | 17.6 |         | $78.5 \pm 6.8$ | 14.3 |
|             | ResNet18  |      | 92.6 | $38.9 \pm 12.4$ | 88.9    | $62.6 \pm 13$  | 23.7 | 85.0    | $72.3 \pm 11$  | 33.4 |
|             |           | 0.5% |      | $70.1 \pm 11.6$ |         | $84.2 \pm 7.2$ | 14.1 |         | $82.1 \pm 2.2$ | 12.0 |
|             | ResNet50  | 1%   | 92.6 | $26.1 \pm 9.4$  | 88.8    | $37.9 \pm 18$  | 11.8 | 85.2    | $46.7 \pm 17$  | 20.6 |
|             |           | 0.5% |      | $61.0 \pm 10.3$ |         | $76.6 \pm 7.8$ | 15.6 |         | $78.3 \pm 3.7$ | 17.3 |
| VGG19       | VGG11     | 1%   | 88.4 | $42.2 \pm 11.6$ | 88.9    | $76.0 \pm 6.1$ | 33.8 | 85.5    | $81.9 \pm 3.9$ | 39.7 |
|             |           | 0.5% |      | $63.6 \pm 9.3$  |         | $85.9 \pm 2.6$ | 22.3 |         | $84.8 \pm 0.5$ | 21.2 |
|             | VGG16     | 1%   | 90.3 | $35.7 \pm 7.9$  | 89.0    | $76.5 \pm 9.0$ | 40.8 | 85.9    | $79.2 \pm 7.5$ | 43.5 |
|             |           | 0.5% |      | $66.6 \pm 8.1$  |         | $87.7 \pm 0.7$ | 21.1 |         | $84.7 \pm 0.9$ | 18.1 |
|             | VGG19     | 1%   | 90.5 | $36.0 \pm 12.0$ | 89.1    | $80.2 \pm 12$  | 44.2 | 86.3    | $84.3 \pm 1.2$ | 48.3 |
|             |           | 0.5% |      | $64.2 \pm 12.4$ |         | $88.8 \pm 0.4$ | 24.6 |         | $85.9 \pm 0.3$ | 21.7 |

<span id="page-7-1"></span>Note. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

Table 2: Energy saving (%) by NeuralFuse for 30 combinations of base models and generators.

|          | ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS |
|----------|-------|-------|---------|---------|-------|-------|
| ResNet18 | 19.0  | 29.1  | 21.2    | 27.5    | 24.0  | 28.9  |
| ResNet50 | 25.4  | 29.9  | 26.4    | 29.2    | 27.7  | 29.9  |
| VGG11    | 6.6   | 27.5  | 11.2    | 24.1    | 17.1  | 27.2  |
| VGG16    | 17.1  | 28.9  | 19.7    | 27.0    | 23.0  | 28.7  |
| VGG19    | 20.3  | 29.7  | 22.3    | 27.8    | 24.8  | 29.1  |

weight-memory access (TWMA), specifics of which can be found in Appendix C's Table 6. In Table 2, below, we report the percentages of energy saved (ES) at voltages that yield a 1% bit-error rate for various base-model and generator combinations. The formula for computing ES is

$$ES = \frac{\text{Energy}_{NV} - \left(\text{Energy}_{LV} + \text{Energy}_{\text{NeuralFuse at }NV}\right)}{\text{Energy}_{NV}} \times 100\%, \tag{6}$$

where NV denotes nominal voltage regime, and LV, a low-voltage one.

Our results indicate that when ResNet18 is utilized as a base model, NeuralFuse can recover model accuracy by 20-49% while reducing energy use by 19-29%. In Appendix C, we provide more results on the tradeoff between energy and accuracy of different NeuralFuse and base-model combinations. Overall, it would appear that using NeuralFuse can effectively restore model accuracy when SRAM encounters low-voltage-induced random bit errors.

Runtime and Latency. On the other hand, runtime and its corresponding energy consumption may also affect overall energy savings. For instance, previous research has shown that multiply-and-accumulate (MAC) operations account for more than 99% of all operations in state-of-the-art CNNs, dominating processing runtime and energy consumption alike Yang et al. [2017]. Therefore, we also report the results of our MAC-based energy-consumption estimation in Appendix C, and of our latency analysis in Appendix D. Here, it should also be noted that an additional latency overhead is an inevitable tradeoff for reducing energy consumption in our scenarios. Although neither runtime nor latency is a major focus of this paper, future researchers could usefully design a lighter-weight version of the NeuralFuse module, or apply model-compression techniques to it, to reduce these two factors.

#### 4.5 Model Size and NeuralFuse Efficiency

To arrive at a full performance characterization of NeuralFuse, we analyzed the relationship between the final recovery achieved by each base model in combination with generators of varying parameter counts. For this purpose, we defined *efficiency ratio* as the recovery percentage in PA divided by NeuralFuse's parameter count. Table 3 compares the efficiency ratios of all NeuralFuse generators

<span id="page-8-0"></span>Table 3: The efficiency ratio for all NeuralFuse generators.

| Base      | BER  |       | NeuralFuse |         |         |       |       |  |  |  |
|-----------|------|-------|------------|---------|---------|-------|-------|--|--|--|
| Model     | DEK  | ConvL | ConvS      | DeConvL | DeConvS | UNetL | UNetS |  |  |  |
| ResNet18  | 1%   | 67.5  | 182        | 76.6    | 190.7   | 94.5  | 245.9 |  |  |  |
| Kesivetio | 0.5% | 24.7  | 73.3       | 30.7    | 62.5    | 33.6  | 88.3  |  |  |  |
| ResNet50  | 1%   | 37.4  | 75.1       | 57.4    | 102.7   | 102.3 | 248.4 |  |  |  |
| Resnetsu  | 0.5% | 35.2  | 108.7      | 40.4    | 92.5    | 47.4  | 124.6 |  |  |  |
| VGG11     | 1%   | 62.3  | 212.9      | 69.5    | 165.8   | 92.0  | 251.7 |  |  |  |
| VUUII     | 0.5% | 32.3  | 96.3       | 35.8    | 77.2    | 38.9  | 100.7 |  |  |  |
| VGG16     | 1%   | 69.6  | 211.2      | 76.9    | 196.5   | 98.8  | 292.9 |  |  |  |
| VUUIO     | 0.5% | 30.3  | 98.1       | 33.0    | 75.3    | 40.6  | 113   |  |  |  |
| VCC10     | 1%   | 57.6  | 147.5      | 65.5    | 141.6   | 95.4  | 250.8 |  |  |  |
| VGG19     | 0.5% | 33.0  | 91.0       | 37.5    | 70.2    | 43.1  | 106.4 |  |  |  |

Figure 4: Reduced-precision accuracy

trained on CIFAR-10. Those results show that UNet-based generators had better efficiency per million parameters than either convolution-based or deconvolution-based ones.

#### <span id="page-8-1"></span>4.6 NeuralFuse's Robustness to Reduced-precision Quantization

Lastly, we also explored NeuralFuse's robustness to low-precision quantization on model weights. Uniform quantization is the usual method for quantizing model weights [Gholami et al., 2022]. However, it is possible for it to cause an accuracy drop due to lack of precision. Given our aim of offering protection from bit errors, we hoped to understand whether NeuralFuse could also recover model-accuracy drops caused by this phenomenon. We therefore uniformly quantized the model weights to a lower bit precision and measured the resulting accuracy. Specifically, we applied symmetric uniform quantization to our base models with various numbers of bits to induce precision loss, and defined the quantized weight  $\mathbf{W_q}$  (integer) as  $\mathbf{W_q} = \lfloor \frac{\mathbf{W}}{\mathbf{s}} \rfloor$ , where  $\mathbf{W}$  denotes the original model weight (full precision),  $s = \frac{\max |\mathbf{W}|}{2^{b-1}-1}$  is the quantization scale parameter, and b is the precision (number of bits) used to quantize the models. Bit errors induced by low voltage operation as previously described, are also applied to low precision weights.

We used the GTSRB pre-trained ResNet18 as our example in an evaluation of two NeuralFuse generators, i.e., ConvL and UNetL trained with 0.5% BER, and varied precision b from 8 bits to 2 bits (integer). The results, shown in Figure 4, indicated that when b>3 bits, NeuralFuse could effectively recover accuracy in both the low-voltage and low-precision scenarios. When b=3, while NeuralFuse could still handle the bit-error-free model (Fig. 4 top), it exhibited a limited ability to recover the random bit-error case (Fig. 4 bottom). We find these results encouraging, insofar as NeuralFuse was only trained on random bit errors, yet demonstrated high accuracy in dealing with unseen bit-quantization errors. Further experimental results derived from other base models and datasets can be found in Appendix H.

#### 4.7 Extended Analysis

Here, we would like to highlight some key findings from the additional results in the Appendices. In Appendix E, we compare NeuralFuse against a simple baseline of learning a universal input perturbation. We found that such baseline performed much worse than NeuralFuse at that task, validating the necessity of adopting input-aware transformation if the goal is to learn error-resistant data representations in low-voltage scenarios. In Appendix G, we report that ensemble training of white-box surrogate base models could further improve the transferability of NeuralFuse in restricted-access scenarios. Appendices K and L present visualization results of NeuralFuse's data embeddings and transformed inputs. In Appendix J, we show that NeuralFuse can further recover the accuracy of a base model trained with adversarial weight perturbation in a low-voltage setting.

#### <span id="page-8-2"></span>5 Conclusion

In this paper, we have proposed NeuralFuse, the first non-intrusive post hoc module that protects model inference against bit errors induced by low voltage. NeuralFuse is particularly well suited to

practical machine-deployment cases in which access to the base model is limited or relaxed. The design of NeuralFuse includes a novel loss function and a new optimizer, EOPM, that enable it to handle simulated randomness in perturbed models. Our comprehensive experimental results and analysis show that NeuralFuse can recover test accuracy by up to 57% while simultaneously enjoying an up to 24% reduction in memory-access energy requirements. Moreover, NeuralFuse demonstrates high transferability to access-constrained models and high versatility, e.g., robustness to low-precision quantization. In short, NeuralFuse is a notable advancement in mitigating neural-network inference's energy/accuracy tradeoff in low-voltage regimes, and points the way to greener future AI technology. Our future work will include extending this study to other neural-network architectures and modalities, such as transformer-based language models.

Limitations. We acknowledge the challenge of achieving significant power savings without accuracy loss and view NeuralFuse as a foundational, proof-of-concept step toward this goal. Future research could enhance this approach by optimizing the pre-processing module to adapt to specific error characteristics of low-voltage SRAM or by integrating lightweight hardware modifications to further improve the energy-accuracy trade-off.

Broader Impacts. We see no ethical or immediate negative societal consequence of our work, and it holds the potential for positive social impacts, from environmental benefits to improved access to technology and enhanced safety in critical applications.

## Acknowledgments and Disclosure of Funding

We thank the anonymous reviewers for their insightful comments and valuable suggestions. The research described in this paper was conducted in the JC STEM Lab of Intelligent Design Automation, which is funded by The Hong Kong Jockey Club Charities Trust in support of Tsung-Yi Ho.

## References

- <span id="page-9-3"></span>Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 284–293, Stockholm, Sweden, 10–15 Jul 2018. PMLR.
- <span id="page-9-9"></span>Babak Ehteshami Bejnordi, Tijmen Blankevoort, and Max Welling. Batch-shaping for learning conditional channel gated networks. In *International Conference on Learning Representations*, 2020.
- <span id="page-9-0"></span>Nandhini Chandramoorthy, Karthik Swaminathan, Martin Cochet, Arun Paidimarri, Schuyler Eldridge, Rajiv V. Joshi, Matthew M. Ziegler, Alper Buyuktosunoglu, and Pradip Bose. Resilient low voltage accelerators for high energy efficiency. In *2019 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, pages 147–158, 2019. doi: 10.1109/HPCA.2019.00034.
- <span id="page-9-7"></span>Yu-Hsin Chen, Joel Emer, and Vivienne Sze. Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks. *ACM SIGARCH computer architecture news*, 44(3):367–379, 2016.
- <span id="page-9-4"></span>Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.
- <span id="page-9-1"></span>Shrikanth Ganapathy, John Kalamatianos, Keith Kasprak, and Steven Raasch. On characterizing near-threshold sram failures in finfet technology. In *Proceedings of the 54th Annual Design Automation Conference 2017*, pages 1–6, 2017.
- <span id="page-9-8"></span>Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In *Low-Power Computer Vision*, pages 291–326. Chapman and Hall/CRC, 2022.
- <span id="page-9-2"></span>Yunchao Gong, Liu Liu, Ming Yang, and Lubomir Bourdev. Compressing deep convolutional networks using vector quantization. *arXiv preprint arXiv:1412.6115*, 2014.
- <span id="page-9-6"></span>Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.
- <span id="page-9-5"></span>Zhizhong Huang, Jie Chen, Junping Zhang, and Hongming Shan. Learning representation for clustering via prototype scattering and positive sampling. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, pages 1–16, 2022. doi: 10.1109/TPAMI.2022.3216454.

- <span id="page-10-1"></span>Sung Kim, Patrick Howe, Thierry Moreau, Armin Alaghi, Luis Ceze, and Visvesh Sathe. Matic: Learning around errors for efficient low-voltage neural network accelerators. In *2018 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, pages 1–6. IEEE, 2018.
- <span id="page-10-2"></span>Skanda Koppula, Lois Orosa, A Giray Yaglıkçı, Roknoddin Azizi, Taha Shahroodi, Konstantinos Kanellopou- ˘ los, and Onur Mutlu. Eden: Enabling energy-efficient, high-performance deep neural network inference using approximate dram. In *Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture*, pages 166–181, 2019.
- <span id="page-10-10"></span>Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, Ontario, 2009.
- <span id="page-10-17"></span>Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1765–1773, 2017.
- <span id="page-10-13"></span>Tuan Anh Nguyen and Anh Tran. Input-aware dynamic backdoor attack. *Advances in Neural Information Processing Systems (NeurIPS)*, 33:3454–3464, 2020.
- <span id="page-10-9"></span>Ozan Özdenizci and Robert Legenstein. Improving robustness against stealthy weight bit-flip attacks by output code matching. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13388–13397, 2022.
- <span id="page-10-4"></span>Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In *European Conference on Computer Vision (ECCV)*, pages 525–542. Springer, 2016.
- <span id="page-10-0"></span>Brandon Reagen, Paul Whatmough, Robert Adolf, Saketh Rama, Hyunkwang Lee, Sae Kyu Lee, José Miguel Hernández-Lobato, Gu-Yeon Wei, and David Brooks. Minerva: Enabling low-power, highly-accurate deep neural network accelerators. In *2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA)*, pages 267–278, 2016. doi: 10.1109/ISCA.2016.32.
- <span id="page-10-14"></span>Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 234–241. Springer, 2015.
- <span id="page-10-15"></span>Ananda Samajdar, Jan Moritz Joseph, Yuhao Zhu, Paul Whatmough, Matthew Mattina, and Tushar Krishna. A systematic methodology for characterizing scalability of dnn accelerators using scale-sim. In *2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 58–68, 2020.
- <span id="page-10-12"></span>K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations (ICLR)*, 2015.
- <span id="page-10-16"></span>Vladislav Sovrasov. ptflops: a flops counting tool for neural networks in pytorch framework, 2018-2023. URL <https://github.com/sovrasov/flops-counter.pytorch>. Github Repository.
- <span id="page-10-8"></span>Gopalakrishnan Srinivasan, Parami Wijesinghe, Syed Shakib Sarwar, Akhilesh Jaiswal, and Kaushik Roy. Significance driven hybrid 8t-6t sram for energy-efficient synaptic storage in artificial neural networks. In *2016 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, pages 151–156. IEEE, 2016.
- <span id="page-10-11"></span>Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32:323–332, 2012.
- <span id="page-10-3"></span>David Stutz, Nandhini Chandramoorthy, Matthias Hein, and Bernt Schiele. Bit error robustness for energyefficient dnn accelerators. In A. Smola, A. Dimakis, and I. Stoica, editors, *Proceedings of Machine Learning and Systems*, volume 3, pages 569–598, 2021.
- <span id="page-10-19"></span>Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. *Journal of Machine Learning Research*, 9(86):2579–2605, 2008.
- <span id="page-10-18"></span>Dongxian Wu, Shu-Tao Xia, and Yisen Wang. Adversarial weight perturbation helps robust generalization. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 2958–2969. Curran Associates, Inc., 2020.
- <span id="page-10-5"></span>Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng. Quantized convolutional neural networks for mobile devices. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4820–4828, 2016.
- <span id="page-10-6"></span>Haichuan Yang, Yuhao Zhu, and Ji Liu. Ecc: Platform-independent energy-constrained deep neural network compression via a bilinear regression model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11206–11215, 2019a.
- <span id="page-10-7"></span>Haichuan Yang, Yuhao Zhu, and Ji Liu. Energy-constrained compression for deep neural networks via weighted sparse projection and layer input masking. In *International Conference on Learning Representations (ICLR)*, 2019b.

- <span id="page-11-1"></span>Haichuan Yang, Shupeng Gui, Yuhao Zhu, and Ji Liu. Automatic neural network compression by sparsityquantization joint learning: A constrained optimization-based approach. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
- <span id="page-11-0"></span>Tien-Ju Yang, Yu-Hsin Chen, and Vivienne Sze. Designing energy-efficient convolutional neural networks using energy-aware pruning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5687–5695, 2017.
- <span id="page-11-2"></span>Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In *International Conference on Machine Learning (ICML)*, pages 7472–7482. PMLR, 2019.

#### **Appendix**

The appendix provides more implementation details for our method, experimental results on more datasets and settings, ablation studies, and qualitative analysis. The appendices cover the following:

- Implementation Details: NeuralFuse Training Algorithm (Sec. A), NeuralFuse Generator (Sec. B), Energy/Accuracy Tradeoff (Sec. C), Latency Reports (Sec. D)
- Experimental Results: Ablation Studies (Sec. E), Relaxed Access (Sec. F), Restricted Access (Sec. G), Reduced-precision Quantization (Sec. H), Adversarial Training (Sec. I), Adversarial Weight Perturbation (Sec. J)
- Qualitative Studies: Data Embeddings Visualization (Sec. K), Transformed Inputs Visualization (Sec. L)

#### <span id="page-12-1"></span>**A** Training Algorithm of NeuralFuse

#### **Algorithm 1** Training steps for NeuralFuse

**Input**: Base model  $M_0$ ; Generator  $\mathcal{G}$ ; Training data samples  $\mathcal{X}$ ; Distribution of the perturbed models  $\mathcal{M}_p$ ; Number of perturbed models N; Total training iterations T

**Output**: Optimized parameters  $W_{\mathcal{G}}$  for the Generator  $\mathcal{G}$ 

```
1: for t = 0, ..., T - 1 do
          for all mini-batches \{\mathbf{x},y\}_{b=1}^B \sim \mathcal{X} do
 2:
               Create transformed inputs \mathbf{x}_t = \mathcal{F}(\mathbf{x}) = \text{clip}_{[-1,1]}(\mathbf{x} + \mathcal{G}(\mathbf{x})).
 3:
               Sample N perturbed models \{M_{p_1},...,M_{p_N}\} from \mathcal{M}_p under p\% random bit errors. for all M_{p_i} \sim \{M_{p_1},...,M_{p_N}\} do Calculate the loss \mathcal{L}_{p_i} based on the output of the perturbed model M_{p_i}. Then calculate
 4:
 5:
 6:
                    the gradients g_{p_i} for \mathcal{W}_{\mathcal{G}} based on \mathcal{L}_{p_i}.
 7:
 8:
               Calculate the loss \mathcal{L}_0 based on the output of the clean model M_0. Then calculate the
               gradients g_0 for \mathcal{W}_{\mathcal{G}} based on \mathcal{L}_0.
 9:
               Calculate the final gradient g_{final} using (5) based on g_0 and g_{p_1}, ..., g_{p_N}.
               Update W_{\mathcal{G}} using g_{final}.
10:
11:
           end for
12: end for
```

#### <span id="page-12-2"></span><span id="page-12-0"></span>**B** Implementation Details of NeuralFuse Generator

We consider two main goals in designing the NeuralFuse Generator: 1) efficiency (so the overall energy overhead is decreased) and 2) robustness (so that it can generate robust patterns on the input image and overcome the random bit flipping in subsequent models). Accordingly, we choose to utilize an encode-decoder architecture in implementing the generator. The design of ConvL is inspired by Nguyen and Tran [2020], in which the authors utilize a similar architecture to design an inputaware trigger generator, and have demonstrated its efficiency and effectiveness. Furthermore, we attempted to enhance it by replacing the Upsampling layer with a Deconvolution layer, leading to the creation of DeConvL. The UNetL-based NeuralFuse draws inspiration from Ronneberger et al. [2015], known for its robust performance in image segmentation, and thus, we incorporated it as one of our architectures. Lastly, ConvS, DeConvS, and UNetS are scaled-down versions of the model designed to reduce computational costs and total parameters. The architectures of Convolutional-based and Deconvolutional-based are shown in Table 4, and the architecture of UNet-based generators is in Table 5. For the abbreviation used in the table, ConvBlock means the Convolution block, Conv means a single Convolution layer, DeConvBlock means the Deconvolution block, DeConv means a single Deconvolution layer, and BN means a Batch Normalization layer. We use learning rate = 0.001,  $\lambda = 5$ , and Adam optimizer. For CIFAR-10, GTSRB, and CIFAR-100, we set batch size b = 25 for each base model. For ImageNet-10, we set b=64 for ResNet18, ResNet50 and VGG11, and b=32for both VGG16 and VGG19.

<span id="page-13-1"></span>Table 4: Model architecture for both Convolution-based and Deconvolution-based generators. Each ConvBlock consists of a Convolution (kernel  $= 3 \times 3$ , padding = 1, stride = 1), a Batch Normalization, and a ReLU layer. Each DeConvBlock consists of a Deconvolution (kernel  $= 4 \times 4$ , padding = 1, stride = 2), a Batch Normalization, and a ReLU layer.

| ConvL                          |      | ConvS               |      | DeConvL                 | DeConvS |                    |      |
|--------------------------------|------|---------------------|------|-------------------------|---------|--------------------|------|
| Layers                         | #CHs | Layers              | #CHs | Layers                  | #CHs    | Layers             | #CHs |
| (ConvBlock)×2, MaxPool         | 32   | ConvBlock, Maxpool  | 32   | (ConvBlock)×2, MaxPool  | 32      | ConvBlock, Maxpool | 32   |
| (ConvBlock)×2, MaxPool         | 64   | ConvBlock, Maxpool  | 64   | (ConvBlock)×2, MaxPool  | 64      | ConvBlock, Maxpool | 64   |
| (ConvBlock)×2, MaxPool         | 128  | ConvBlock, Maxpool  | 64   | (ConvBlock)×2, MaxPool, | 128     | ConvBlock, Maxpool | 64   |
| ConvBlock, UpSample, ConvBlock | 128  | ConvBlock, UpSample | 64   | ConvBlock               | 128     | DeConvBlock        | 64   |
| ConvBlock, UpSample, ConvBlock | 64   | ConvBlock, UpSample | 32   | DeConvBlock, ConvBlock  | 64      | DeConvBlock        | 32   |
| ConvBlock, UpSample, ConvBlock | 32   | ConvBlock, UpSample | 3    | DeConvBlock, ConvBlock  | 32      | DeConv, BN, Tanh   | 3    |
| Conv, BN, Tanh                 | 32   | Conv, BN, Tanh      | 3    | Conv, BN, Tanh          | 3       |                    |      |

[Note] #CHs: number of channels.

<span id="page-13-2"></span>Table 5: Model architecture for UNet-based generators. Each ConvBlock consists of a Convolution (kernel  $= 3 \times 3$ , padding = 1, stride = 1), a Batch Normalization, and a ReLU layer. Other layers, such as the Deconvolutional layer (kernel  $= 2 \times 2$ , padding = 1, stride = 2), are used in UNet-based models. For the final Convolution layer, the kernel size is set to 1.

| UNetL                      |           | UNetS                      |           |
|----------------------------|-----------|----------------------------|-----------|
| Layers                     | #Channels | Layers                     | #Channels |
| L1: (ConvBlock)×2          | 16        | L1: (ConvBlock)×2          | 8         |
| L2: Maxpool, (ConvBlock)×2 | 32        | L2: Maxpool, (ConvBlock)×2 | 16        |
| L3: Maxpool, (ConvBlock)×2 | 64        | L3: Maxpool, (ConvBlock)×2 | 32        |
| L4: Maxpool, (ConvBlock)×2 | 128       | L4: Maxpool, (ConvBlock)×2 | 64        |
| L5: DeConv                 | 64        | L5: DeConv                 | 32        |
| L6: Concat[L3, L5]         | 128       | L6: Concat[L3, L5]         | 64        |
| L7: (ConvBlock)×2          | 64        | L7: (ConvBlock)×2          | 32        |
| L8: DeConv                 | 32        | L8: DeConv                 | 16        |
| L9: Concat[L2, L8]         | 64        | L9: Concat[L2, L8]         | 32        |
| L10: (ConvBlock)×2         | 32        | L10: (ConvBlock)×2         | 16        |
| L11: DeConv                | 16        | L11: DeConv                | 8         |
| L12: Concat[L1, L11]       | 32        | L12: Concat[L1, L11]       | 16        |
| L13: (ConvBlock)×2         | 16        | L13: (ConvBlock)×2         | 8         |
| L14: Conv                  | 3         | L14: Conv                  | 3         |

#### <span id="page-13-0"></span>C NeuralFuse's Energy/Accuracy Tradeoff

SCALE-SIM. SCALE-SIM [Samajdar et al., 2020] is a systolic array-based CNN simulator that can calculate the number of memory accesses and the total time in execution cycles by giving the specific model architecture and accelerator architectural configuration as inputs. In this paper, we use SCALE-SIM to calculate the weights memory access of 5 based models (ResNet18, ResNet50, VGG11, VGG16, VGG19), and 6 generators (ConvL, ConvS, DeConvL, DeConvS, UNetL, UNetS). While SCALE-SIM supports both Convolutional and Linear layers, it does not yet support Deconvolution layers. Instead, we try to approximate the memory costs of Deconvolution layers by Convolution layers. We change the input and output from Deconvolution into the output and input of the Convolution layers. Besides, we also change the stride into 1 when we approximate it. We also add padding for the convolution layers while generating input files for SCALE-SIM. In this paper, we only consider the energy saving on weights accesses, so we only take the value "SRAM Filter Reads" from the output of SCALE-SIM as the *total weights memory accesses* (T.W.M.A.) for further energy calculation.

In Table 6, we report the total weight memory access (T.W.M.A.) using SCALE-SIM. We then showed the energy/accuracy tradeoff between all of the combinations of NeuralFuse and base models under a 1% of bit error rate in Figure 5.

**Parameters and MACs Calculation.** In addition to T.W.M.A., the number of parameters and MACs (multiply–accumulate operations) are also common metrics in measuring the energy consumption of machine learning models. Yang et al. [2017] have shown that the *energy consumption of computation* 

Table 6: The total weights memory access calculated by SCALE-SIM.

<span id="page-14-0"></span>

| Base Model<br>ResNet18 | ResNet50  | VGG11     | VGG16     | VGG19     | -     |
|------------------------|-----------|-----------|-----------|-----------|-------|
| T.W.M.A.<br>2,755,968  | 6,182,144 | 1,334,656 | 2,366,848 | 3,104,128 | -     |
|                        |           |           |           |           |       |
| NeuralFuse<br>ConvL    | ConvS     | DeConvL   | DeConvS   | UNetL     | UNetS |

[Note] T.W.M.A.: *total weight memory access*.

<span id="page-14-1"></span>Figure 5: The energy/accuracy tradeoff of different NeuralFuse implementations with all CIFAR-10 pre-trained based models. The X-axis represents the percentage reduction in dynamic memory access energy at low-voltage settings (base model protected by NeuralFuse) compared to the bit-error-free (nominal) voltage; the Y-axis represents the perturbed accuracy (evaluated at low voltage) with a 1% bit error rate.

and *memory accesses* are both proportional to MACs, allowing us to take computation energy consumption into account.

Here, we use the open-source package ptflops [\[Sovrasov,](#page-10-16) [2018-2023\]](#page-10-16) to calculate the parameters and MAC values of all the base models and the NeuralFuse generators, in the same units as [Bejnordi](#page-9-9) [et al.](#page-9-9) [\[2020\]](#page-9-9) used. The results are shown in Table [7.](#page-14-2) Note that we modified the base model architectures for ImageNet-10, as it has larger input sizes. For example, we use a larger kernel size = 7 instead of 3 in the first Convolution layer in ResNet-based models to enhance the learning abilities. Therefore, the parameters of base models are different between different datasets. For NeuralFuse generators, we utilize the same architectures for implementation (including ImageNet-10). As a result, our proposed NeuralFuse generators are generally smaller than base models, either on total model parameters or MAC values.

<span id="page-14-2"></span>Table 7: Parameter counts and MACs for all base models and generators in this paper.

|           |                         |                          |                          | Base Model               |                           |                           |                   |
|-----------|-------------------------|--------------------------|--------------------------|--------------------------|---------------------------|---------------------------|-------------------|
|           |                         | ResNet18                 | ResNet50                 | VGG11                    | VGG16                     | VGG19                     | -                 |
| Parameter | CIFAR-10<br>ImageNet-10 | 11,173,962<br>11,181,642 | 23,520,842<br>23,528,522 | 9,231,114<br>128,812,810 | 14,728,266<br>134,309,962 | 20,040,522<br>139,622,218 | -                 |
| MACs      | CIFAR-10<br>ImageNet-10 | 557.14M<br>1.82G         | 1.31G<br>4.12G           | 153.5M<br>7.64G          | 314.43M<br>15.53G         | 399.47M<br>19.69G         | -                 |
|           |                         |                          |                          | NeuralFuse               |                           |                           |                   |
|           |                         | ConvL                    | ConvS                    | DeConvL                  | DeConvS                   | UNetL                     | UNetS             |
| Parameter | CIFAR-10<br>ImageNet-10 | 723,273                  | 113,187                  | 647,785                  | 156,777                   | 482,771                   | 121,195           |
| MACs      | CIFAR-10<br>ImageNet-10 | 80.5M<br>3.94G           | 10.34M<br>506.78M        | 64.69M<br>3.17G          | 22.44M<br>1.1G            | 41.41M<br>2.03G           | 10.58M<br>518.47M |

MACs-based Energy Consumption. We can then use the MAC values to further approximate the end-to-end energy consumption of the whole model. Assume that all values are stored on SRAM and that a MAC represents single memory access. The corresponding MACs-based energy saving percentage (MAC-ES, %) can be derived from Eq. 7 (c.f. Sec. 4.4), and results can be found in Table 8. We can observe that most combinations can save a large amount of energy, except that VGG11 with two larger NeuralFuse (ConvL and DeConvL) may increase the total energy. These results are consistent with the results reported in Table 2. In addition, we also showed the MACs-based energy/accuracy tradeoff between all of the combinations of NeuralFuse and base models under a 1% of bit error rate in Figure 6.

<span id="page-15-1"></span>
$$\text{MAC-ES} = \frac{\text{MACs}_{\text{base model}} \cdot \text{Energy}_{\text{nominal voltage}} - \left( \text{MACs}_{\text{base model}} \cdot \text{Energy}_{\text{low-voltage-regime}} + \text{MACs}_{\text{NeuralFuse}} \cdot \text{Energy}_{\text{NeuralFuse at nominal voltage}} \right)}{\text{MACs}_{\text{base model}} \cdot \text{Energy}_{\text{nominal voltage}}} \times 100\%$$

<span id="page-15-2"></span>Table 8: The MACs-Based energy saving percentage (%) for different combinations of base models and NeuralFuse.

|          | ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS |
|----------|-------|-------|---------|---------|-------|-------|
| ResNet18 | 16.2  | 28.7  | 19.0    | 26.6    | 23.2  | 28.7  |
| ResNet50 | 24.5  | 29.8  | 25.7    | 28.9    | 27.4  | 29.8  |
| VGG11    | -21.8 | 23.9  | -11.5   | 16.0    | 3.6   | 23.7  |
| VGG16    | 5.0   | 27.3  | 10.0    | 23.5    | 17.4  | 27.2  |
| VGG19    | 10.4  | 28.0  | 14.4    | 25.0    | 20.2  | 28.0  |

<span id="page-15-3"></span>Figure 6: The Mac-based energy/accuracy tradeoff of different NeuralFuse implementations with all CIFAR-10 pre-trained based models. The X-axis represents the percentage reduction in dynamic memory access energy at low-voltage settings (base model protected by NeuralFuse), compared to the bit-error-free (nominal) voltage; the Y-axis represents the perturbed accuracy (evaluated at low voltage) with a 1% bit error rate.

Although using ConvL or DeConvL along with base model VGG11 for CIFAR-10 implies an increase in energy consumption, using other smaller-scale generators, we can still save the overall energy and recover the base model's accuracy. That said, developers can always choose smaller generators (with orders of magnitude fewer MAC operations than the original network) to restore model accuracy, further demonstrating the flexibility of choosing NeuralFuse generators of different sizes.

## <span id="page-15-0"></span>**D** Inference Latency of NeuralFuse

In Table 9, we report the latency (batch\_size=1, CIFAR-10/ImageNet-10 testing dataset) of utilizing the different NeuralFuse generators with two different base models, ResNet18 and VGG19. While

NeuralFuse contributes some additional latency, we consider this an unavoidable tradeoff necessary to achieve reduced energy consumption within our framework. Although the primary focus of this paper is not on latency, we acknowledge its importance. Future research could explore designing a more lightweight version of the NeuralFuse module or applying model compression techniques to minimize latency. Additionally, we recognize that running NeuralFuse on a general-purpose CPU could lead to different latency and energy consumption figures due to various influencing factors like CPU architecture and manufacturing processes.

| Table 9: The Inference Latence | v of base model and base model with NeuralFuse. |
|--------------------------------|-------------------------------------------------|
|                                |                                                 |

<span id="page-16-1"></span>

|            | ResNet18 (CIFAR-10) | VGG19 (CIFAR-10) | ResNet18 (ImageNet-10) | VGG19 (ImageNet-10) |
|------------|---------------------|------------------|------------------------|---------------------|
| Base Model | 5.84 ms             | 5.32 ms          | 6.21 ms                | 14.34 ms            |
| + ConvL    | 9.37 ms (+3.53)     | 8.96 ms (+3.64)  | 10.51 ms (+4.3)        | 17.66 ms (+3.32)    |
| + ConvS    | 7.86 ms (+2.02)     | 7.40 ms (+2.08)  | 8.28 ms (+2.07)        | 16.72 ms (+2.38)    |
| + DeConvL  | 9.18 ms (+3.34)     | 8.59 ms (+3.27)  | 10.07 ms (+3.86)       | 17.24 ms (+2.90)    |
| + DeConvS  | 7.49 ms (+1.65)     | 7.04 ms (+1.72)  | 7.79 ms (+1.58)        | 15.67 ms (+1.33)    |
| + UNetL    | 10.69 ms (+4.85)    | 10.06 ms (+4.74) | 11.14 ms (+4.93)       | 18.54 ms (+4.20)    |
| + UNetS    | 10.63 ms (+4.79)    | 10.13 ms (+4.81) | 11.36 ms (+5.15)       | 18.60 ms (+4.26)    |

#### <span id="page-16-0"></span>E Ablation Studies

Study for N in EOPM. Here, we study the effect of N used in EOPM (Eq. 5). In Figure 7, we report the results for ConvL and ConvS on CIFAR-10 pre-trained ResNet18, under a 1% bit error rate (BER). The results demonstrate that if we apply larger N, the performance increases until convergence. Specifically, for ConvL (Figure 7a), larger N empirically has a smaller standard deviation; this means larger N gives better stability but at the cost of time-consuming training. In contrast, for the small generator ConvS (Figure 7b), we can find that the standard deviation is still large even trained by larger N; the reason might be that small generators are not as capable of learning representations as larger ones. Therefore, there exists a tradeoff between the *stability* of the generator performance and the total training time. In our implementation, choosing N=5 or 10 is a good balance.

<span id="page-16-2"></span>Figure 7: The experimental results on using different sizes of N for EOPM.

**Tradeoff Between Clean Accuracy (CA) and Perturbed Accuracy (PA).** We conducted an experiment to study the effect of different  $\lambda$  values, which balance the ratio of clean accuracy and perturbed accuracy. In Table 10, the experimental results showed that a smaller  $\lambda$  can preserve clean accuracy, but result in poor perturbed accuracy. On the contrary, larger  $\lambda$  can deliver higher perturbed accuracy, but with more clean accuracy drop. This phenomenon has also been observed in adversarial training [Zhang et al., 2019].

Comparison to Universal Input Perturbation (UIP). Moosavi-Dezfooli et al. [2017] has shown that there exists a universal adversarial perturbation to the input data such that the model will make wrong predictions on a majority of the perturbed images. In our NeuralFuse framework, the universal perturbation is a special case when we set  $\mathcal{G}(\mathbf{x}) = \tanh{(\mathbf{UIP})}$  for any data sample  $\mathbf{x}$ . The transformed data sample then becomes  $\mathbf{x}_t = \text{clip}_{[-1,1]}(\mathbf{x} + \tanh{(\mathbf{UIP})})$ , where  $\mathbf{x}_t \in [-1,1]^d$ 

<span id="page-17-0"></span>Table 10: Experimental results based on  $\lambda$  value choosing. The results show that  $\lambda=5$  can balance the tradeoff between clean accuracy and perturbed accuracy.

| Base<br>Model | λ                           | CA   | PA              | CA (NF)                              | ConvL<br>PA (NF)                                                                      | RP                                  |
|---------------|-----------------------------|------|-----------------|--------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------|
| ResNet18      | 10<br>5<br>1<br>0.1<br>0.01 | 92.6 | 38.9 ± 12.4     | 90.1<br>89.8<br>90.0<br>91.6<br>92.2 | $88.0 \pm 1.7$<br>$87.8 \pm 1.7$<br>$84.2 \pm 3.8$<br>$65.7 \pm 9.3$<br>$43.6 \pm 13$ | 49.1<br>48.8<br>45.3<br>26.8<br>4.7 |
| VGG19         | 10<br>5<br>1<br>0.1<br>0.01 | 90.5 | $36.0 \pm 12.0$ | 89.6<br>89.8<br>89.9<br>89.1<br>90.2 | $77.9 \pm 19$<br>$77.7 \pm 19$<br>$73.1 \pm 19$<br>$51.2 \pm 16$<br>$36.8 \pm 12$     | 41.9<br>41.7<br>37.1<br>15.2<br>0.8 |

Note. CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recover percentage of PA (NF) vs. PA

and **UIP** is a trainable universal input perturbation that has the same size as the input data. The experimental results with the universal input perturbation are shown in Table 11. We observe that its performance is much worse than our proposed NeuralFuse. This result validates the necessity of adopting input-aware transformation for learning error-resistant data representations in low-voltage scenarios.

<span id="page-17-1"></span>Table 11: Performance of the universal input perturbation (UIP) trained by EOPM on CIFAR-10 pre-trained ResNet18.

| Base Model | BER        | CA   | PA                                                                  | CA (UIP)     | PA ( <b>UIP</b> )                | RP          |
|------------|------------|------|---------------------------------------------------------------------|--------------|----------------------------------|-------------|
| ResNet18   | 1%<br>0.5% | 92.6 | $38.9 \pm 12.4  70.1 \pm 11.6$                                      | 91.8<br>92.5 | $37.9 \pm 11$<br>$70.6 \pm 11$   | -1.0<br>0.5 |
| ResNet50   | 1%<br>0.5% | 92.6 | $\begin{array}{c cccc} 26.1 \pm & 9.4 \\ 61.0 \pm 10.3 \end{array}$ | 80.7<br>91.9 | $21.0 \pm 5.9$<br>$62.4 \pm 12$  | -5.1<br>1.4 |
| VGG11      | 1%<br>0.5% | 88.4 | $42.2 \pm 11.6 \\ 63.6 \pm 9.3$                                     | 86.9<br>88.2 | $43.0 \pm 11$ $64.2 \pm 8.8$     | 0.8<br>0.6  |
| VGG16      | 1%<br>0.5% | 90.3 | 35.7 ± 7.9<br>66.6 ± 8.1                                            | 90.1<br>90.4 | $37.1 \pm 8.5$<br>$67.3 \pm 8.1$ | 1.4<br>0.7  |
| VGG19      | 1% 0.5%    | 90.5 | $36.0 \pm 12.0 \\ 64.2 \pm 12.4$                                    | 89.9<br>90.1 | $35.3 \pm 12$ $64.4 \pm 12$      | -0.7<br>0.2 |

*Note.* BER: the bit-error rate of the base model; CA (%): clean accuracy; PA (%): perturbed accuracy; UIP: universal input transformation parameter; and RP: total recovery percentage of PA (UIP) vs. PA

## <span id="page-18-0"></span>F Additional Experimental Results on Relaxed Access Settings

We conducted more experiments on *Relaxed Access* settings to show that our NeuralFuse can protect the models under different BER The results can be found in Sec. F.1 (CIFAR-10), Sec. F.2 (GTSRB), Sec. F.3 (ImageNet-10), and Sec. F.4 (CIFAR-100). For comparison, we also visualize the experimental results in the figures below each table.

#### <span id="page-18-1"></span>F.1 CIFAR-10

<span id="page-18-2"></span>Table 12: Testing accuracy (%) under 1% and 0.5% of random bit error rate on CIFAR-10.

| Base     | NE                                                     |      |                | 1% I                                         | BER                                                                                                      |                                              |                | 0.5%                                         | BER                                                                                                      |                                              |
|----------|--------------------------------------------------------|------|----------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------|----------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Model    | NF                                                     | CA   | PA             | CA (NF)                                      | PA (NF)                                                                                                  | RP                                           | PA             | CA (NF)                                      | PA (NF)                                                                                                  | RP                                           |
| ResNet18 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 92.6 | 38.9<br>± 12.4 | 89.8<br>88.2<br>89.6<br>82.9<br>86.6<br>84.4 | $87.8 \pm 1.7$<br>$59.5 \pm 11$<br>$88.5 \pm 0.8$<br>$68.8 \pm 6.4$<br>$84.6 \pm 0.8$<br>$68.8 \pm 6.0$  | 48.8<br>20.6<br>49.6<br>29.9<br>45.6<br>29.8 | 70.1<br>± 11.6 | 90.4<br>91.7<br>90.2<br>84.1<br>89.7<br>90.9 | $87.9 \pm 2.2$<br>$78.4 \pm 8.3$<br>$90.0 \pm 0.2$<br>$79.9 \pm 3.6$<br>$86.3 \pm 2.4$<br>$80.7 \pm 5.8$ | 17.8<br>8.3<br>19.9<br>9.8<br>16.2<br>10.7   |
| ResNet50 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 92.6 | 26.1<br>± 9.4  | 85.5<br>85.2<br>87.4<br>82.4<br>86.2<br>77.3 | $53.2 \pm 22$ $34.6 \pm 14$ $63.3 \pm 21$ $42.2 \pm 17$ $75.5 \pm 12$ $56.2 \pm 19$                      | 27.1<br>8.5<br>37.2<br>16.1<br>49.4<br>30.1  | 61.0<br>± 10.3 | 90.3<br>90.8<br>89.5<br>90.3<br>89.9<br>89.7 | $86.5 \pm 3.2$<br>$73.3 \pm 8.7$<br>$87.2 \pm 2.5$<br>$75.5 \pm 8.1$<br>$83.9 \pm 3.6$<br>$76.1 \pm 7.2$ | 25.5<br>12.3<br>26.2<br>14.5<br>22.9<br>15.1 |
| VGG11    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 88.4 | 42.2<br>± 11.6 | 89.6<br>84.9<br>89.3<br>85.6<br>87.1<br>85.5 | $87.2 \pm 2.9$<br>$66.3 \pm 7.5$<br>$87.2 \pm 2.6$<br>$68.2 \pm 7.1$<br>$83.6 \pm 1.3$<br>$72.7 \pm 4.6$ | 45.1<br>24.1<br>45.0<br>26.0<br>41.4<br>30.5 | 63.6<br>± 9.3  | 89.8<br>88.2<br>89.6<br>88.3<br>88.0<br>88.1 | $87.0 \pm 1.3$<br>$74.5 \pm 5.7$<br>$86.9 \pm 1.1$<br>$75.7 \pm 4.6$<br>$82.4 \pm 1.8$<br>$75.8 \pm 4.3$ | 23.3<br>10.9<br>23.2<br>12.1<br>18.8<br>12.2 |
| VGG16    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 90.3 | 35.7<br>± 7.9  | 90.1<br>87.4<br>89.7<br>86.8<br>87.4         | $86.0 \pm 6.2$<br>$59.6 \pm 12$<br>$85.5 \pm 6.8$<br>$66.5 \pm 11$<br>$83.4 \pm 4.4$<br>$71.2 \pm 8.2$   | 50.3<br>23.9<br>49.8<br>30.8<br>47.7<br>35.5 | 66.6<br>± 8.1  | 90.2<br>89.9<br>89.7<br>90.0<br>89.0         | $88.5 \pm 0.9$<br>$77.8 \pm 4.8$<br>$88.2 \pm 1.0$<br>$78.4 \pm 4.7$<br>$86.2 \pm 1.5$<br>$80.2 \pm 3.5$ | 21.9<br>11.1<br>21.4<br>11.8<br>19.6<br>13.7 |
| VGG19    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 90.5 | 36.0<br>± 12.0 | 89.8<br>87.3<br>86.3<br>86.5<br>86.3<br>86.3 | $77.7 \pm 19$ $52.7 \pm 17$ $78.4 \pm 18$ $58.2 \pm 18$ $82.1 \pm 4.8$ $66.4 \pm 13$                     | 41.7<br>16.7<br>42.4<br>22.2<br>46.0<br>30.4 | 64.2<br>± 12.4 | 90.4<br>89.6<br>90.4<br>89.7<br>89.1<br>89.2 | $88.1 \pm 1.8  74.5 \pm 9.0  88.5 \pm 1.4  75.2 \pm 8.6  85.0 \pm 2.7  77.1 \pm 7.3$                     | 23.9<br>10.3<br>24.3<br>11.0<br>20.8<br>12.9 |

 $\textit{Note}. \ CA\ (\%): \ clean\ accuracy; \ PA\ (\%): \ perturbed\ accuracy; \ NF: \ NeuralFuse; \ and \ RP: \ total\ recovery\ percentage\ of\ PA\ (NF)\ vs.\ PA$ 

Figure 8: Experimental results on CIFAR-10

## <span id="page-19-0"></span>F.2 GTSRB

Table 13: Testing accuracy (%) under 1% and 0.5% of random bit error rate on GTSRB.

| Base     | NE                                                     |      |                | 1% I                                         | BER                                                                                                                        |                                              |                | 0.5%                                         | BER                                                                                                                         |                                              |
|----------|--------------------------------------------------------|------|----------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|----------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Model    | NF                                                     | CA   | PA             | CA (NF)                                      | PA (NF)                                                                                                                    | RP                                           | PA             | CA (NF)                                      | PA (NF)                                                                                                                     | RP                                           |
| ResNet18 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 95.5 | 36.9<br>± 16.0 | 95.7<br>94.4<br>95.6<br>95.7<br>96.2<br>95.9 | $\begin{array}{c} 91.1 \pm 4.7 \\ 68.6 \pm 12 \\ 91.3 \pm 4.3 \\ 78.1 \pm 9.1 \\ 93.8 \pm 1.0 \\ 85.1 \pm 6.9 \end{array}$ | 54.2<br>31.7<br>54.4<br>41.2<br>56.9<br>48.2 | 75.2<br>± 12.7 | 93.4<br>94.8<br>95.4<br>95.8<br>96.2<br>95.5 | $89.5 \pm 1.9$<br>$87.7 \pm 4.2$<br>$93.4 \pm 1.1$<br>$90.1 \pm 3.3$<br>$93.5 \pm 1.6$<br>$91.4 \pm 2.8$                    | 14.3<br>12.4<br>18.1<br>14.9<br>18.3<br>16.2 |
| ResNet50 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 95.0 | 29.5<br>± 16.9 | 95.6<br>94.8<br>94.9<br>93.0<br>94.5<br>94.7 | $71.6 \pm 20$ $50.5 \pm 22$ $71.6 \pm 21$ $56.4 \pm 17$ $80.6 \pm 15$ $64.7 \pm 22$                                        | 42.1<br>21.0<br>42.0<br>26.9<br>51.1<br>35.2 | 74.0<br>± 13.0 | 94.6<br>95.4<br>94.7<br>94.6<br>96.5<br>95.9 | $\begin{array}{c} 90.6 \pm 3.7 \\ 84.5 \pm 8.5 \\ 91.6 \pm 2.9 \\ 87.4 \pm 5.9 \\ 93.7 \pm 2.3 \\ 90.6 \pm 4.8 \end{array}$ | 16.6<br>10.5<br>17.6<br>13.5<br>19.7<br>16.7 |
| VGG11    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 91.9 | 34.9<br>± 12.4 | 94.8<br>91.1<br>95.0<br>92.4<br>92.2<br>94.7 | $85.7 \pm 7.2$<br>$62.2 \pm 11$<br>$84.6 \pm 7.6$<br>$67.5 \pm 11$<br>$83.2 \pm 6.0$<br>$73.4 \pm 10$                      | 50.9<br>27.3<br>49.7<br>32.6<br>48.3<br>38.5 | 64.9<br>± 10.8 | 93.9<br>90.9<br>93.6<br>92.3<br>94.8<br>94.6 | $\begin{array}{c} 92.6 \pm 0.7 \\ 80.5 \pm 3.5 \\ 91.9 \pm 0.6 \\ 83.1 \pm 3.7 \\ 90.6 \pm 1.7 \\ 88.9 \pm 2.2 \end{array}$ | 27.7<br>15.7<br>27.1<br>18.2<br>25.7<br>24.1 |
| VGG16    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 95.2 | 15.1<br>± 6.8  | 96.3<br>94.1<br>96.4<br>93.8<br>95.8<br>94.3 | $72.4 \pm 12$ $39.8 \pm 13$ $72.0 \pm 12$ $50.9 \pm 13$ $78.6 \pm 11$ $63.3 \pm 14$                                        | 57.3<br>24.6<br>56.9<br>35.8<br>63.5<br>48.1 | 58.8<br>± 8.9  | 95.6<br>94.3<br>95.6<br>95.1<br>96.0<br>95.4 | $\begin{array}{c} 93.2 \pm 1.8 \\ 82.2 \pm 6.2 \\ 93.1 \pm 2.0 \\ 84.0 \pm 5.3 \\ 92.8 \pm 2.0 \\ 87.8 \pm 3.6 \end{array}$ | 34.4<br>23.4<br>34.3<br>25.2<br>34.0<br>29.0 |
| VGG19    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 95.5 | 36.6<br>± 6.8  | 96.0<br>93.8<br>95.4<br>94.5<br>95.4<br>94.6 | $88.3 \pm 7.2$<br>$69.0 \pm 14$<br>$87.2 \pm 7.5$<br>$73.1 \pm 12$<br>$88.2 \pm 6.7$<br>$80.6 \pm 9.0$                     | 51.7<br>32.4<br>50.6<br>36.5<br>51.7<br>44.1 | 69.1<br>± 11.1 | 95.6<br>94.9<br>95.5<br>95.5<br>94.9<br>96.5 | $93.4 \pm 2.1$<br>$87.0 \pm 4.4$<br>$92.4 \pm 2.2$<br>$88.8 \pm 3.7$<br>$91.7 \pm 2.5$<br>$90.8 \pm 3.4$                    | 24.2<br>17.8<br>23.3<br>19.7<br>22.6<br>21.6 |

Figure 9: Experimental results on GTSRB.

## <span id="page-20-0"></span>F.3 ImageNet-10

Table 14: Testing accuracy under 0.5% of random bit error rate on ImageNet-10.

| Base      | N.T.    |      |                | 0.5% B  | ER             |      |
|-----------|---------|------|----------------|---------|----------------|------|
| Model     | NF      | CA   | PA             | CA (NF) | PA (NF)        | RP   |
|           | ConvL   |      |                | 94.0    | $88.0 \pm 2.0$ | 15.7 |
|           | ConvS   |      |                | 91.8    | $83.6 \pm 4.1$ | 11.3 |
| ResNet18  | DeConvL | 02.2 | $72.3 \pm 7.0$ | 94.0    | $89.2 \pm 1.3$ | 16.9 |
| Resnetto  | DeConvS | 92.2 | $12.3 \pm 1.0$ | 92.8    | $87.5 \pm 2.3$ | 15.2 |
|           | UNetL   |      |                | 94.0    | $88.1 \pm 1.4$ | 15.8 |
|           | UNetS   |      |                | 93.2    | $86.4 \pm 2.2$ | 14.1 |
|           | ConvL   |      |                | 92.2    | $80.0 \pm 5.8$ | 40.6 |
|           | ConvS   |      |                | 91.8    | $65.0 \pm 11$  | 25.6 |
| ResNet50  | DeConvL | 89.8 | $39.4 \pm 11$  | 93.0    | $79.4 \pm 5.9$ | 40.0 |
| Resinciso | DeConvS | 09.0 | 39.4 ⊥ 11      | 93.2    | $70.9 \pm 9.1$ | 31.5 |
|           | UNetL   |      |                | 92.2    | $80.5 \pm 5.8$ | 41.1 |
|           | UNetS   |      |                | 92.4    | $73.6 \pm 8.9$ | 34.2 |
|           | ConvL   |      |                | 92.0    | $86.1\pm3.7$   | 38.3 |
|           | ConvS   |      |                | 89.4    | $66.4 \pm 7.1$ | 18.6 |
| VGG11     | DeConvL | 91.6 | $47.8 \pm 13$  | 91.0    | $86.0 \pm 3.0$ | 38.2 |
| 70011     | DeConvS | 71.0 | 17.0 ± 13      | 89.0    | $72.5 \pm 7.8$ | 24.7 |
|           | UNetL   |      |                | 92.4    | $83.0 \pm 3.5$ | 35.2 |
|           | UNetS   |      |                | 86.2    | $73.5 \pm 6.0$ | 25.7 |
|           | ConvL   |      |                | 90.8    | $77.1 \pm 11$  | 38.7 |
|           | ConvS   |      |                | 90.2    | $60.2 \pm 14$  | 21.8 |
| VGG16     | DeConvL | 94.6 | $38.4 \pm 17$  | 91.2    | $77.2 \pm 11$  | 38.8 |
| 10010     | DeConvS | 71.0 | 30.1 ± 17      | 90.0    | $62.3 \pm 14$  | 23.9 |
|           | UNetL   |      |                | 90.6    | $81.1 \pm 5.9$ | 42.7 |
|           | UNetS   |      |                | 86.4    | $72.3 \pm 8.8$ | 33.9 |
|           | ConvL   |      |                | 91.4    | $75.5 \pm 8.8$ | 38.3 |
|           | ConvS   |      |                | 88.8    | $56.5 \pm 13$  | 19.3 |
| VGG19     | DeConvL | 92.4 | $37.2 \pm 11$  | 91.0    | $75.9 \pm 8.9$ | 38.7 |
|           | DeConvS |      |                | 88.8    | $64.0 \pm 11$  | 26.8 |
|           | UNetL   |      |                | 89.4    | $77.9 \pm 6.1$ | 40.7 |
|           | UNetS   |      |                | 87.6    | $65.9 \pm 10$  | 28.7 |

Figure 10: Experimental results on ImageNet-10, 0.5% Bit Error Rate.

#### <span id="page-21-0"></span>F.4 CIFAR-100

As mentioned in the previous section, larger generators like ConvL, DeConvL, and UNetL have better performance than small generators. For CIFAR-100, we find that the gains of utilizing NeuralFuse are less compared to the other datasets. We believe this is because CIFAR-100 is a more challenging dataset (more classes) for the generators to learn to protect the base models. Nevertheless, NeuralFuse can still function to restore some degraded accuracy; these results also demonstrate that our NeuralFuse is applicable to different datasets. In addition, although the recover percentage is less obvious on CIFAR-100 (the more difficult dataset), we can still conclude that our NeuralFuse is applicable to different datasets.

Table 15: Testing accuracy (%) under 1%, 0.5% and 0.35% of random bit error rate on CIFAR-100.

| Base     |                                                        |      | 1             | 1%                                           | BER                                                                                                                         |                                            |               | 0.5%                                         | BER                                                                                                                         |                                           |               | 0.35%                                        | BER                                                                                                                         |                                           |
|----------|--------------------------------------------------------|------|---------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|---------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|---------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Model    | NF                                                     | C.A. | PA            | CA (NF)                                      | PA (NF)                                                                                                                     | RP                                         | PA            | CA (NF)                                      | PA (NF)                                                                                                                     | RP                                        | PA            | CA (NF)                                      | PA (NF)                                                                                                                     | RP                                        |
| ResNet18 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 73.7 | 4.6<br>± 2.9  | 54.8<br>49.7<br>55.2<br>32.7<br>50.6<br>26.8 | $\begin{array}{c} 11.0 \pm 7.7 \\ 4.2 \pm 2.2 \\ 11.9 \pm 8.2 \\ 4.0 \pm 2.2 \\ 14.5 \pm 8.9 \\ 4.6 \pm 2.5 \end{array}$    | 6.4<br>-0.4<br>7.3<br>-0.6<br>10.0<br>-0.0 | 20.9<br>± 7.4 | 65.2<br>70.0<br>66.3<br>68.2<br>66.2<br>67.1 | $39.0 \pm 7.1$ $24.5 \pm 7.6$ $38.2 \pm 6.9$ $25.9 \pm 6.8$ $40.1 \pm 6.4$ $28.8 \pm 6.8$                                   | 18.1<br>3.6<br>17.3<br>5<br>19.2<br>7.9   | 31.4<br>± 7.6 | 69.4<br>72.1<br>69.2<br>71.6<br>70.3<br>70.9 | $42.9 \pm 6.2$ $35.1 \pm 7.3$ $42.9 \pm 5.5$ $35.8 \pm 5.5$ $46.3 \pm 5.5$ $38.3 \pm 6.4$                                   | 11.4<br>3.7<br>11.4<br>4.4<br>14.9<br>6.9 |
| ResNet50 | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 73.5 | 3.0<br>± 1.8  | 63.5<br>65.5<br>59.6<br>61.1<br>39.0<br>47.7 | $3.2 \pm 1.7$<br>$3.2 \pm 1.6$<br>$3.2 \pm 1.7$<br>$3.2 \pm 1.7$<br>$5.0 \pm 1.7$<br>$3.4 \pm 1.8$                          | 0.1<br>0.1<br>0.2<br>0.1<br>1.9<br>0.3     | 21.3<br>± 7.0 | 68.4<br>71.9<br>68.1<br>70.3<br>66.6<br>69.1 | $\begin{array}{c} 28.8 \pm 6.7 \\ 23.1 \pm 6.9 \\ 28.6 \pm 7.0 \\ 25.0 \pm 6.7 \\ 36.5 \pm 6.2 \\ 26.1 \pm 6.6 \end{array}$ | 7.6<br>1.9<br>7.4<br>3.7<br>15.3<br>4.8   | 35.7<br>± 8.6 | 72.0<br>73.0<br>71.7<br>72.8<br>70.8<br>72.6 | $40.8 \pm 7.5$ $37.4 \pm 8.0$ $41.7 \pm 7.7$ $38.9 \pm 7.9$ $45.3 \pm 6.7$ $39.6 \pm 7.8$                                   | 5.1<br>1.7<br>6.1<br>3.3<br>9.6<br>3.9    |
| VGG11    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 64.8 | 8.2<br>± 5.7  | 58.3<br>56.6<br>60.3<br>58.3<br>51.1<br>51.9 | $\begin{array}{c} 19.7 \pm 11 \\ 10.4 \pm 7.4 \\ 21.2 \pm 11 \\ 11.8 \pm 7.9 \\ 22.1 \pm 8.2 \\ 13.1 \pm 7.9 \end{array}$   | 11.5<br>2.2<br>13.0<br>3.5<br>13.9<br>4.9  | 23.9<br>± 9.4 | 63.1<br>62.7<br>63.9<br>61.9<br>61.8<br>61.7 | $\begin{array}{c} 38.8 \pm 9.3 \\ 27.9 \pm 10 \\ 40.0 \pm 9.0 \\ 29.8 \pm 9.9 \\ 37.8 \pm 9.0 \\ 29.8 \pm 9.7 \end{array}$  | 15.0<br>4.0<br>16.2<br>5.9<br>13.9<br>6.0 | 31.3<br>± 10  | 63.9<br>63.9<br>64.0<br>63.5<br>63.5         | $42.4 \pm 9.0$ $41.8 \pm 8.3$ $42.8 \pm 9.1$ $36.1 \pm 10$ $40.9 \pm 9.3$ $35.7 \pm 9.9$                                    | 11.1<br>10.5<br>11.5<br>4.8<br>9.6<br>4.5 |
| VGG16    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 67.8 | 7.0<br>± 3.5  | 51.4<br>44.3<br>53.1<br>23.5<br>50.2<br>27.7 | $\begin{array}{c} 19.2 \pm 6.0 \\ 6.7 \pm 2.3 \\ 20.8 \pm 6.2 \\ 4.8 \pm 1.7 \\ 25.3 \pm 1.7 \\ 9.9 \pm 2.1 \end{array}$    | 12.6<br>0.1<br>14.2<br>-1.8<br>18.7<br>3.3 | 22.4<br>± 7.0 | 61.8<br>63.8<br>62.8<br>62.1<br>61.7<br>61.6 | $\begin{array}{c} 41.1 \pm 5.6 \\ 27.5 \pm 6.8 \\ 42.1 \pm 5.5 \\ 29.9 \pm 6.7 \\ 41.3 \pm 5.0 \\ 31.3 \pm 6.3 \end{array}$ | 18.7<br>5.1<br>19.8<br>7.5<br>18.9<br>8.9 | 31.1<br>± 7.2 | 64.9<br>66.0<br>65.0<br>64.9<br>64.8<br>65.0 | $\begin{array}{c} 44.9 \pm 5.3 \\ 36.3 \pm 6.1 \\ 46.6 \pm 5.2 \\ 38.1 \pm 6.3 \\ 46.8 \pm 4.6 \\ 39.8 \pm 5.9 \end{array}$ | 13.8<br>5.1<br>15.5<br>7.0<br>15.7<br>8.7 |
| VGG19    | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 67.8 | 10.6<br>± 4.3 | 59.4<br>63.7<br>60.1<br>60.9<br>58.7<br>59.1 | $\begin{array}{c} 29.2 \pm 8.1 \\ 14.4 \pm 5.1 \\ 29.6 \pm 8.5 \\ 16.1 \pm 6.0 \\ 30.2 \pm 8.2 \\ 18.0 \pm 6.2 \end{array}$ | 18.6<br>3.8<br>19.0<br>5.6<br>19.6<br>7.4  | 34.0<br>± 9.6 | 65.6<br>66.6<br>65.7<br>66.5<br>65.5<br>66.3 | $46.5 \pm 6.8$ $38.3 \pm 6.8$ $46.9 \pm 7.1$ $39.0 \pm 3.7$ $46.9 \pm 6.5$ $40.1 \pm 8.0$                                   | 12.5<br>4.2<br>12.9<br>5.0<br>12.9<br>6.1 | 42.1<br>± 9.4 | 66.9<br>67.7<br>67.3<br>67.7<br>67.4<br>67.5 | $\begin{array}{c} 49.2 \pm 7.4 \\ 45.3 \pm 8.5 \\ 49.8 \pm 7.6 \\ 45.7 \pm 8.4 \\ 50.0 \pm 7.5 \\ 46.6 \pm 8.4 \end{array}$ | 7.0<br>3.2<br>7.6<br>3.6<br>7.9<br>4.5    |

Figure 11: Experimental results on CIFAR-100.

## <span id="page-22-0"></span>G Additional Experimental Results on Restricted Access Settings (Transferability)

We conduct more experiments with *Restricted Access* settings to show that our NeuralFuse can be transferred to protect various black-box models. The experimental results are shown in Sec. [G.1](#page-22-1) (CIFAR-10), Sec. [G.2](#page-23-0) (GTSRB), and Sec. [G.3](#page-24-0) (CIFAR-100).

We find that using VGG19 as a white-box surrogate model has better *transferability* than ResNet18 for all datasets. In addition, we can observe that some NeuralFuse generators have *downward applicability* if base models have a similar architecture. In other words, if we try to transfer a generator trained on a large BER (e.g., 1%) to a model with a small BER (e.g., 0.5%), the performance will be better than that of a generator trained with the original BER (e.g., 0.5%). For example, in Table [16,](#page-22-2) we could find that if we use VGG19 as a source model to train the generator ConvL (1%), the generator could deliver better performance (in terms of PA (NF)) when applied to similar base models (e.g., VGG11, VGG16, or VGG19) under a 0.5% BER, compared to using itself as a source model (shown in Table [12\)](#page-18-2). We conjecture that this is because the generators trained on a larger BER can also cover the error patterns of a smaller BER, and thus they have better generalizability across smaller B.E.Rs.

To further improve the transferability to cross-architecture target models, we also conduct an experiment in Sec. [G.4](#page-25-1) to show that using ensemble-based training can help the generator to achieve this feature.

#### <span id="page-22-1"></span>G.1 CIFAR-10

<span id="page-22-2"></span>The results of CIFAR-10 in which NeuralFuse is trained at 1% BER are shown in Table [16.](#page-22-2)

Table 16: Transfer results on CIFAR-10: NeuralFuse trained on SM with 1% BER

|          |          |      |      |             |         | ConvL (1%) |      |         | UNetL (1%) |      |
|----------|----------|------|------|-------------|---------|------------|------|---------|------------|------|
| SM       | TM       | BER  | CA   | PA          | CA (NF) | PA (NF)    | RP   | CA (NF) | PA (NF)    | RP   |
|          | ResNet18 | 0.5% | 92.6 | 70.1 ± 11.6 | 89.8    | 89.5 ± 0.2 | 19.4 | 86.6    | 86.2 ± 0.3 | 16.1 |
|          | ResNet50 | 1%   | 92.6 | 26.1 ± 9.4  | 89.5    | 36.0 ± 19  | 9.9  | 85.2    | 38.8 ± 19  | 12.7 |
|          |          | 0.5% |      | 61.0 ± 10.3 |         | 75.1 ± 10  | 14.1 |         | 77.1 ± 5.0 | 16.1 |
|          | VGG11    | 1%   | 88.4 | 42.2 ± 11.6 | 88.4    | 62.5 ± 8.4 | 20.3 | 76.8    | 61.1 ± 8.5 | 18.9 |
| ResNet18 |          | 0.5% |      | 63.6 ± 9.3  |         | 81.0 ± 4.6 | 17.4 |         | 73.7 ± 3.0 | 10.1 |
|          | VGG16    | 1%   | 90.3 | 35.7 ± 7.9  | 89.6    | 63.3 ± 18  | 27.6 | 85.2    | 59.9 ± 16  | 24.2 |
|          |          | 0.5% |      | 66.6 ± 8.1  |         | 85.0 ± 3.4 | 18.4 |         | 80.2 ± 4.5 | 13.6 |
|          | VGG19    | 1%   | 90.5 | 36.0 ± 12.0 | 89.6    | 50.7 ± 22  | 14.7 | 85.3    | 51.1 ± 16  | 15.1 |
|          |          | 0.5% |      | 64.2 ± 12.4 |         | 80.2 ± 8.7 | 16.0 |         | 76.5 ± 7.8 | 12.3 |
|          | ResNet18 | 1%   | 92.6 | 38.9 ± 12.4 | 89.8    | 61.0 ± 17  | 22.1 | 87.0    | 69.7 ± 11  | 30.8 |
|          |          | 0.5% |      | 70.1 ± 11.6 |         | 86.1 ± 6.9 | 16.0 |         | 84.2 ± 3.0 | 14.1 |
|          | ResNet50 | 1%   | 92.6 | 26.1 ± 9.4  | 89.9    | 34.0 ± 19  | 7.9  | 87.0    | 44.2 ± 17  | 18.1 |
|          |          | 0.5% |      | 61.0 ± 10.3 |         | 76.5 ± 10  | 15.5 |         | 80.7 ± 4.2 | 19.7 |
| VGG19    | VGG11    | 1%   | 88.4 | 42.2 ± 11.6 | 89.7    | 76.5 ± 7.0 | 34.3 | 87.1    | 79.9 ± 5.6 | 37.7 |
|          |          | 0.5% |      | 63.6 ± 9.3  |         | 88.0 ± 2.1 | 24.4 |         | 85.4 ± 0.8 | 21.8 |
|          | VGG16    | 1%   | 90.3 | 35.7 ± 7.9  | 89.6    | 75.5 ± 12  | 39.8 | 87.2    | 78.9 ± 7.8 | 43.2 |
|          |          | 0.5% |      | 66.6 ± 8.1  |         | 88.9 ± 0.6 | 22.3 |         | 86.2 ± 0.3 | 19.6 |
|          | VGG19    | 0.5% | 90.5 | 64.2 ± 12.4 | 89.8    | 89.6 ± 8.7 | 25.4 | 87.4    | 86.8 ± 0.4 | 22.6 |

*Note*. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

#### <span id="page-23-0"></span>G.2 GTSRB

In Tables [17](#page-23-1) and [18,](#page-23-2) we show the results on GTSRB in which NeuralFuse is trained at 1.5% and 1% BER, respectively.

Table 17: Transfer results on GTSRB: NeuralFuse trained on SM with 1.5% BER

<span id="page-23-1"></span>

|          |          |      |      | PA          |         | ConvL (1.5%) |      |         | UNetL (1.5%) |      |
|----------|----------|------|------|-------------|---------|--------------|------|---------|--------------|------|
| SM       | TM       | BER  | CA   |             | CA (NF) | PA (NF)      | RP   | CA (NF) | PA (NF)      | RP   |
|          | ResNet18 | 1%   | 95.5 | 36.9 ± 16.0 | 95.7    | 93.9 ± 1.9   | 57.0 | 94.9    | 94.4 ± 0.4   | 57.5 |
|          |          | 0.5% |      | 75.2 ± 12.7 |         | 95.7 ± 0.2   | 20.5 |         | 94.8 ± 0.2   | 19.6 |
|          | ResNet50 | 1%   | 95.0 | 29.5 ± 16.9 | 94.4    | 37.0 ± 22    | 7.5  | 94.4    | 47.1 ± 23    | 17.6 |
|          |          | 0.5% |      | 74.0 ± 13.0 |         | 77.5 ± 13    | 3.5  |         | 84.8 ± 9.5   | 10.8 |
| ResNet18 | VGG11    | 1%   | 91.9 | 34.9 ± 12.4 | 92.8    | 45.2 ± 10    | 10.3 | 91.4    | 50.5 ± 13    | 15.6 |
|          |          | 0.5% |      | 64.9 ± 10.8 |         | 79.4 ± 5.8   | 14.5 |         | 83.9 ± 4.2   | 19.0 |
|          | VGG16    | 1%   | 95.2 | 15.1 ± 6.8  | 95.4    | 31.1 ± 13    | 15.9 | 94.6    | 36.8 ± 12    | 21.7 |
|          |          | 0.5% |      | 58.8 ± 8.9  |         | 84.5 ± 8.3   | 25.8 |         | 86.0 ± 8.6   | 27.2 |
|          | VGG19    | 1%   | 95.5 | 36.6 ± 6.8  | 95.0    | 56.4 ± 15    | 19.8 | 94.3    | 60.8 ± 15    | 24.2 |
|          |          | 0.5% |      | 69.1 ± 11.1 |         | 86.9 ± 3.4   | 17.8 |         | 87.7 ± 3.8   | 18.6 |
|          | ResNet18 | 1%   | 95.5 | 36.9 ± 16.0 | 88.4    | 50.3 ± 12    | 13.4 | 92.8    | 63.7 ± 16    | 26.8 |
|          |          | 0.5% |      | 75.2 ± 12.7 |         | 77.9 ± 7.4   | 2.7  |         | 87.5 ± 3.9   | 12.3 |
|          | ResNet50 | 1%   | 95.0 | 29.5 ± 16.9 | 87.5    | 29.7 ± 17    | 0.2  | 92.5    | 40.4 ± 21    | 10.9 |
|          |          | 0.5% |      | 74.0 ± 13.0 |         | 67.9 ± 17    | -6.1 |         | 77.5 ± 15    | 3.5  |
| VGG19    | VGG11    | 1%   | 91.9 | 34.9 ± 12.4 | 89.7    | 47.1 ± 11    | 12.2 | 93.5    | 60.0 ± 12    | 25.1 |
|          |          | 0.5% |      | 64.9 ± 10.8 |         | 76.3 ± 5.1   | 11.4 |         | 86.0 ± 3.8   | 21.1 |
|          | VGG16    | 1%   | 95.2 | 15.1 ± 6.8  | 93.0    | 29.2 ± 15    | 14.1 | 93.0    | 38.5 ± 16    | 23.4 |
|          |          | 0.5% |      | 58.8 ± 8.9  |         | 75.7 ± 12    | 16.9 |         | 79.9 ± 8.3   | 21.1 |
|          | VGG19    | 1%   | 95.5 | 36.6 ± 6.8  | 95.1    | 87.4 ± 6.0   | 50.8 | 94.6    | 88.7 ± 5.0   | 52.1 |
|          |          | 0.5% |      | 69.1 ± 11.1 |         | 92.4 ± 2.4   | 23.3 |         | 92.4 ± 2.2   | 23.3 |

*Note*. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

Table 18: Transfer results on GTSRB: NeuralFuse trained on SM with 1% BER

<span id="page-23-2"></span>

|          |          |      |      |             |         | ConvL (1%) |      |         | UNetL (1%) |      |
|----------|----------|------|------|-------------|---------|------------|------|---------|------------|------|
| SM       | TM       | BER  | CA   | PA          | CA (NF) | PA (NF)    | RP   | CA (NF) | PA (NF)    | RP   |
|          | ResNet18 | 0.5% | 95.5 | 75.2 ± 12.7 | 95.7    | 95.3 ± 0.5 | 20.1 | 96.2    | 95.7 ± 0.3 | 20.5 |
|          |          | 1%   |      | 29.5 ± 16.9 |         | 35.6 ± 21  | 6.1  |         | 42.6 ± 23  | 13.1 |
|          | ResNet50 | 0.5% | 95.0 | 74.0 ± 13.0 | 94.5    | 78.8 ± 13  | 4.8  | 95.6    | 87.3 ± 9.0 | 13.3 |
|          | VGG11    | 1%   | 91.9 | 34.9 ± 12.4 | 93.1    | 45.8 ± 11  | 10.9 | 94.0    | 47.1 ± 14  | 12.2 |
| ResNet18 |          | 0.5% |      | 64.9 ± 10.8 |         | 81.8 ± 5.0 | 16.9 |         | 84.2 ± 4.8 | 19.3 |
|          | VGG16    | 1%   | 95.2 | 15.1 ± 6.8  | 95.5    | 26.5 ± 12  | 11.4 | 95.5    | 32.4 ± 11  | 17.3 |
|          |          | 0.5% |      | 58.8 ± 8.9  |         | 82.2 ± 9.0 | 23.4 |         | 85.4 ± 6.7 | 26.6 |
|          | VGG19    | 1%   | 95.5 | 36.6 ± 6.8  | 94.9    | 53.2 ± 14  | 16.6 | 95.6    | 60.9 ± 15  | 24.3 |
|          |          | 0.5% |      | 69.1 ± 11.1 |         | 85.4 ± 4.5 | 16.3 |         | 87.5 ± 3.7 | 18.4 |
|          | ResNet18 | 1%   | 95.5 | 36.9 ± 16.0 | 93.7    | 53.1 ± 16  | 16.2 | 95.0    | 63.4 ± 18  | 26.5 |
|          |          | 0.5% |      | 75.2 ± 12.7 |         | 83.9 ± 7.6 | 8.7  |         | 89.7 ± 4.8 | 14.5 |
|          | ResNet50 | 1%   | 95.0 | 29.5 ± 16.9 | 92.8    | 30.6 ± 18  | 1.1  | 95.4    | 38.9 ± 22  | 9.4  |
|          |          | 0.5% |      | 74.0 ± 13.0 |         | 74.7 ± 18  | 0.7  |         | 81.5 ± 16  | 7.5  |
| VGG19    | VGG11    | 1%   | 91.9 | 34.9 ± 12.4 | 93.7    | 50.6 ± 11  | 15.7 | 95.1    | 58.9 ± 15  | 24.0 |
|          |          | 0.5% |      | 64.9 ± 10.8 |         | 82.3 ± 5.1 | 17.4 |         | 87.5 ± 3.7 | 22.6 |
|          |          | 1%   |      | 15.1 ± 6.8  |         | 27.8 ± 15  | 12.7 |         | 33.5 ± 14  | 18.4 |
|          | VGG16    | 0.5% | 95.2 | 58.8 ± 8.9  | 95.2    | 79.0 ± 12  | 20.2 | 95.2    | 81.8 ± 7.8 | 23.0 |
|          | VGG19    | 0.5% | 95.5 | 69.1 ± 11.1 | 96.0    | 94.0 ± 2.2 | 24.9 | 95.4    | 93.9 ± 2.1 | 24.8 |

*Note*. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

#### <span id="page-24-0"></span>**G.3** CIFAR-100

In Tables 19 and 20, we show results on CIFAR-100 with NeuralFuse trained at 1% and 0.5% BER, respectively.

Table 19: Transfer results on CIFAR-100: NeuralFuse trained on SM with 1% BER

<span id="page-24-1"></span>

| CM        | TM       | DED   | CA   | DA             |         | ConvL (1%)     |       |         | UNetL (1%)     |      |
|-----------|----------|-------|------|----------------|---------|----------------|-------|---------|----------------|------|
| SM        | TM       | BER   | CA   | PA             | CA (NF) | PA (NF)        | RP    | CA (NF) | PA (NF)        | RP   |
|           | ResNet18 | 0.5%  | 73.7 | $20.9 \pm 7.4$ | 54.8    | $35.8 \pm 5.2$ | 14.9  | 50.6    | $39.3 \pm 2.8$ | 18.4 |
|           | Resnetts | 0.35% | 13.1 | $31.4 \pm 7.6$ | 34.8    | $41.7\pm3.7$   | 10.3  | 30.0    | $43.3\pm1.4$   | 11.9 |
|           |          | 1%    |      | 3.0 ± 1.8      |         | $2.2 \pm 2.0$  | -0.8  |         | $2.4 \pm 1.9$  | -0.6 |
|           | ResNet50 | 0.5%  | 73.5 | $21.3 \pm 7.0$ | 44.9    | $15.9 \pm 8.2$ | -5.4  | 41.5    | $17.1\pm7.1$   | -4.2 |
|           |          | 0.35% |      | $35.7 \pm 8.6$ |         | $23.7 \pm 7.1$ | -12.0 |         | $26.2 \pm 5.6$ | -9.5 |
|           |          | 1%    |      | 8.2 ± 5.7      |         | $9.8 \pm 5.6$  | 1.6   |         | $10.2 \pm 5.1$ | 2.0  |
| ResNet18  | VGG11    | 0.5%  | 64.8 | $23.9 \pm 9.4$ | 41.2    | $24.2\pm5.9$   | 0.3   | 37.5    | $24.5 \pm 4.7$ | 0.6  |
| Resiletto |          | 0.35% |      | $31.3\pm10.0$  |         | $29.0 \pm 5.4$ | -2.3  |         | $28.2 \pm 4.5$ | -3.1 |
|           |          | 1%    |      | $7.0 \pm 3.5$  |         | $7.9 \pm 3.7$  | 0.9   |         | $10.1 \pm 4.5$ | 3.1  |
|           | VGG16    | 0.5%  | 67.8 | $22.4 \pm 7.0$ | 44.0    | $22.4 \pm 7.6$ | 0.0   | 39.5    | $26.3 \pm 5.3$ | 3.9  |
|           |          | 0.35% |      | $31.1 \pm 7.2$ |         | $28.1 \pm 5.9$ | -3.0  |         | $30.6 \pm 3.6$ | -0.5 |
|           |          | 1%    |      | $10.6 \pm 4.3$ |         | $13.5 \pm 6.1$ | 2.9   |         | $15.6 \pm 6.2$ | 5.0  |
|           | VGG19    | 0.5%  | 67.8 | $34.0 \pm 9.6$ | 44.2    | $27.9 \pm 4.8$ | -6.1  | 40.7    | $29.3 \pm 4.6$ | -4.7 |
|           |          | 0.35% |      | $42.1 \pm 9.4$ |         | $33.2 \pm 48$  | -8.9  |         | $32.8 \pm 3.9$ | -9.3 |
|           |          | 1%    |      | $4.6 \pm 2.9$  |         | $5.8 \pm 3.7$  | 1.2   |         | $6.8 \pm 4.4$  | 2.2  |
|           | ResNet18 | 0.5%  | 73.7 | $20.9 \pm 7.4$ | 55.5    | $24.6 \pm 6.3$ | 3.7   | 57.3    | $28.1 \pm 5.9$ | 7.2  |
|           |          | 0.35% |      | $31.4 \pm 7.6$ |         | $31.1 \pm 5.0$ | -0.3  |         | $36.4 \pm 4.5$ | 5.0  |
|           |          | 1%    |      | $3.0 \pm 1.8$  |         | $2.8 \pm 2.1$  | -0.2  |         | $3.7 \pm 2.4$  | 0.7  |
|           | ResNet50 | 0.5%  | 73.5 | $21.3 \pm 7.0$ | 56.1    | $18.9 \pm 8.6$ | -2.4  | 56.1    | $22.8 \pm 8.5$ | 1.5  |
|           |          | 0.35% |      | $35.7 \pm 8.6$ |         | $28.7\pm8.2$   | -7.0  |         | $33.7 \pm 7.0$ | -2.0 |
| VGG19     |          | 1%    |      | 8.2 ± 5.7      |         | $12.3 \pm 8.4$ | 4.1   |         | $15.4 \pm 9.4$ | 7.2  |
| VGG17     | VGG11    | 0.5%  | 64.8 | $23.9 \pm 9.4$ | 52.8    | $30.0 \pm 9.3$ | 6.1   | 53.9    | $33.3 \pm 7.2$ | 9.4  |
|           |          | 0.35% |      | $31.3\pm10.0$  |         | $36.5 \pm 7.7$ | 5.2   |         | $38.8 \pm 6.5$ | 7.5  |
|           |          | 1%    |      | $7.0 \pm 3.5$  |         | $11.2 \pm 4.4$ | 4.2   |         | $13.6 \pm 5.2$ | 6.6  |
|           | VGG16    | 0.5%  | 67.8 | $22.4 \pm 7.0$ | 53.6    | $32.4 \pm 7.3$ | 10.0  | 55.2    | $35.9 \pm 6.2$ | 13.5 |
|           |          | 0.35% |      | $31.1 \pm 7.2$ |         | $39.4 \pm 6.3$ | 8.3   |         | $42.4 \pm 4.9$ | 11.3 |
|           | VGG19    | 0.5%  | 67.8 | $34.0 \pm 9.6$ | 59.4    | $50.2 \pm 3.1$ | 16.2  | 58.7    | $49.1 \pm 3.5$ | 15.1 |
|           | V0019    | 0.35% | 07.0 | $42.1 \pm 9.4$ | 33.4    | $53.1\pm2.3$   | 11.0  | 30.1    | $52.0 \pm 3.1$ | 9.9  |

Note. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

Table 20: Transfer results on CIFAR-100: NeuralFuse trained on SM with 0.5% BER

<span id="page-24-2"></span>

| CM       | TM        | BER   | CA   | PA             | C       | onvL (0.5%)    |      | U.      | NetL (0.5%)    |      |
|----------|-----------|-------|------|----------------|---------|----------------|------|---------|----------------|------|
| SM       | I IVI     | BEK   | CA   | PA             | CA (NF) | PA (NF)        | RP   | CA (NF) | PA (NF)        | RP   |
|          | ResNet18  | 0.35% | 73.7 | $31.4 \pm 7.6$ | 65.2    | $47.7 \pm 4.9$ | 16.3 | 66.2    | $49.2 \pm 4.1$ | 17.8 |
|          | ResNet50  | 0.5%  | 73.5 | $21.3 \pm 7.0$ | 62.5    | $24.0 \pm 9.9$ | 2.8  | 63.5    | $26.4 \pm 9.1$ | 5.1  |
|          | Residence | 0.35% | /3.3 | $35.7 \pm 8.6$ | 02.3    | $36.3 \pm 8.9$ | 0.6  | 03.3    | $39.4 \pm 8.1$ | 3.7  |
|          | VGG11     | 0.5%  | 64.8 | $23.9 \pm 9.4$ | 59.2    | $33.0 \pm 9.8$ | 9.2  | 61.1    | $34.2 \pm 9.8$ | 10.3 |
| ResNet18 | VGGII     | 0.35% | 04.0 | $31.3\pm10.0$  | 39.2    | $40.4 \pm 8.7$ | 9.1  | 01.1    | $41.4 \pm 9.0$ | 10.1 |
|          | VGG16     | 0.5%  | 67.8 | $22.4 \pm 7.0$ | 59.5    | $34.7 \pm 8.0$ | 12.3 | 61.4    | $37.5 \pm 6.8$ | 15.2 |
|          | VGG10     | 0.35% | 07.8 | $31.1 \pm 7.2$ | 39.3    | $42.9 \pm 6.0$ | 11.8 | 01.4    | $45.3 \pm 4.9$ | 14.2 |
|          | VGG19     | 0.5%  | 67.8 | $34.0 \pm 9.6$ | 61.6    | $43.7 \pm 6.2$ | 9.6  | 62.0    | $45.0 \pm 6.3$ | 11.0 |
|          | VGG19     | 0.35% | 07.8 | $42.1 \pm 9.4$ | 01.0    | $49.0 \pm 5.5$ | 6.8  | 02.0    | $50.5\pm5.3$   | 8.3  |
|          | ResNet18  | 0.5%  | 73.7 | $20.9 \pm 7.4$ | 66.1    | $24.9 \pm 6.7$ | 4.0  | 67.8    | $27.7 \pm 6.8$ | 6.8  |
|          | Residents | 0.35% | 13.1 | $31.4 \pm 7.6$ | 00.1    | $34.4 \pm 5.4$ | 3.0  | 07.8    | $38.1 \pm 5.6$ | 6.7  |
|          | ResNet50  | 0.5%  | 73.5 | $21.3 \pm 7.0$ | 66.2    | $22.7 \pm 7.8$ | 1.4  | 66.7    | $25.4 \pm 8.0$ | 4.2  |
|          | Resireiso | 0.35% | 13.3 | $35.7 \pm 8.6$ | 00.2    | $35.5 \pm 7.7$ | -0.2 | 00.7    | $38.8 \pm 7.5$ | 3.2  |
| VGG19    | VGG11     | 0.5%  | 64.8 | $23.9 \pm 9.4$ | 59.9    | $29.3 \pm 10$  | 5.4  | 61.0    | $31.2 \pm 9.8$ | 7.4  |
|          | VGGII     | 0.35% | 04.8 | $31.3\pm10.0$  | 39.9    | $36.6 \pm 9.5$ | 5.3  | 01.0    | $38.1 \pm 9.0$ | 6.8  |
|          | VGG16     | 0.5%  | 67.8 | $22.4 \pm 7.0$ | 62.5    | $30.8 \pm 7.3$ | 8.4  | 62.6    | $33.0 \pm 7.3$ | 10.7 |
|          | ¥ GG10    | 0.35% | 07.8 | $31.1 \pm 7.2$ | 02.3    | $40.0 \pm 6.5$ | 8.9  | 02.0    | $42.5\pm5.9$   | 11.3 |
|          | VGG19     | 0.35% | 67.8 | $42.1 \pm 9.4$ | 65.6    | $52.0 \pm 6.2$ | 9.8  | 65.5    | $52.6 \pm 6.1$ | 10.4 |

Note. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

#### <span id="page-25-1"></span>**G.4** Generator Ensembling

To improve the transferability performance on cross-architecture cases (e.g., using ResNet-based models as surrogate models to train NeuralFuse and then transfer NeuralFuse to VGG-based target models), we try to adopt ensemble surrogate models to train our NeuralFuse. The experimental results are shown in Table 21. We use the same experimental settings mentioned in Table 1 but change one source model (e.g., ResNet18 or VGG19) into two (ResNet18 with VGG19) for training. The results show that the overall performance is better than the results shown in Table 1, which means ensemble-based training can easily solve the performance degradation on cross-architecture target models.

<span id="page-25-2"></span>Table 21: Transfer results on CIFAR-10: NeuralFuse trained on two SM with 1.5% BER

| SM       | TM                              | BER         | CA   | PA              | Co      | onvL (1.5%)    |      | UN      | NetL (1.5%)    |      |
|----------|---------------------------------|-------------|------|-----------------|---------|----------------|------|---------|----------------|------|
| 31/1     | 1 1/1                           | BEK         | CA   | FA              | CA (NF) | PA (NF)        | RP   | CA (NF) | PA (NF)        | RP   |
|          | ResNet18                        | 1%          | 92.6 | $38.9 \pm 12.4$ | 89.4    | $88.1 \pm 1.0$ | 49.2 | 86.3    | $85.4 \pm 0.5$ | 46.5 |
|          | Resiretto                       | 0.5%        | 92.0 | $70.1 \pm 11.6$ | 09.4    | $89.2 \pm 0.2$ | 19.1 | 80.5    | $86.1 \pm 0.2$ | 16.0 |
|          | ResNet50                        | 1%          | 92.6 | $26.1 \pm 9.4$  | 89.3    | $44.0 \pm 22$  | 17.9 | 86.1    | $50.9 \pm 20$  | 24.8 |
| ResNet18 | Net18                           | 0.5%        | )2.0 | $61.0\pm10.3$   |         | $80.3 \pm 6.7$ | 19.3 | 00.1    | $78.6 \pm 3.9$ | 17.6 |
| +        | VGG11 1 0.5 VGG16 1 0.5 VGG19 1 | 1%          | 88.4 | $42.2 \pm 11.6$ | 89.1    | $77.0 \pm 5.6$ | 34.8 | 85.9    | $82.3 \pm 4.1$ | 40.1 |
| VGG19    |                                 | 0.5%   88.4 |      | $63.6 \pm 9.3$  | 07.1    | $87.5 \pm 1.6$ | 23.9 | 85.9    | $85.0 \pm 0.6$ | 21.4 |
|          |                                 | 1%          | 90.3 | $35.7 \pm 7.9$  | 89.1    | $80.5 \pm 8.6$ | 44.8 | 85.7    | $81.4 \pm 5.5$ | 45.7 |
|          |                                 | 0.5%        | 70.5 | $66.6 \pm 8.1$  | 07.1    | $88.2 \pm 0.7$ | 21.6 | 05.7    | $85.0 \pm 0.7$ | 18.4 |
|          |                                 | 1%          | 90.5 | $36.0 \pm 12.0$ | 89.2    | $75.1 \pm 17$  | 39.1 | 86.1    | $83.0 \pm 3.4$ | 47.0 |
|          |                                 | VGG19 0.5%  | 70.5 | $64.2 \pm 12.4$ | 89.2    | $89.0 \pm 0.2$ | 24.8 | 00.1    | $85.9 \pm 0.4$ | 21.7 |

Note. SM: source model, used for training generators; TM: target model, used for testing generators; BER: the bit-error rate of the target model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

#### <span id="page-25-0"></span>H NeuralFuse on Reduced-precision Quantization and Random Bit Errors

As mentioned in Sec. 4.6, we explore the robustness of NeuralFuse to low-precision quantization of model weights and consider the case of random bit errors. Here, we demonstrate that NeuralFuse can recover not only the accuracy drop due to reduced precision, but also the drop caused by low-voltage-induced bit errors (0.5% BER) under low precision. We selected two NeuralFuse generators (ConvL and UNetL) for our experiments, and these generators were trained with the corresponding base models (ResNet18 and VGG19) at 1% BER (CIFAR-10, GTSRB) and 0.5% BER (ImageNet-10). The experimental results are shown as follows: CIFAR-10 (Sec. H.1), GTSRB (Sec. H.2), and ImageNet-10 (Sec. H.3). Similarly, for ease of comparison, we visualize the experimental results in the figures below each table. Our results show that NeuralFuse can consistently perform well in low-precision regimes as well as recover the low-voltage-induced accuracy drop.

#### <span id="page-25-3"></span>H.1 CIFAR-10

Table 22: Reduced-precision Quantization and with 0.5% BER on CIFAR-10 pre-trained models.

| Base<br>Model | #Bits | CA   | PA               | CA (NF) | ConvL (1%)<br>PA (NF) | RP   | CA (NF) | JNetL (1%)<br>PA (NF) | RP   |
|---------------|-------|------|------------------|---------|-----------------------|------|---------|-----------------------|------|
|               | 8     | 92.6 | $170.1 \pm 11.6$ | 89.8    | 89.5 ± 0.2            | 19.4 | 86.6    | 86.2 ± 0.3            | 16.1 |
|               | 7     |      | $68.8 \pm 10.4$  |         |                       |      |         |                       |      |
|               | '     | 92.5 |                  | 89.8    | $89.5 \pm 1.7$        | 20.7 | 86.5    |                       | 17.2 |
|               | 6     | 92.6 | $68.4 \pm 11.2$  | 89.7    | $89.5 \pm 0.2$        | 21.1 | 86.6    | $85.9 \pm 0.3$        | 17.5 |
| ResNet18      | 5     | 92.4 | $52.7 \pm 14.1$  | 89.7    | $90.0 \pm 0.7$        | 37.3 | 86.5    | $85.5 \pm 0.8$        | 32.8 |
|               | 4     | 91.8 | $26.3 \pm 12.7$  | 89.8    | $58.7 \pm 24.5$       | 32.4 | 86.6    | $64.9 \pm 22.5$       | 38.6 |
|               | 3     | 84.8 | $11.3 \pm 1.8$   | 89.8    | $12.8 \pm 5.8$        | 1.5  | 86.0    | $14.8 \pm 10.0$       | 3.5  |
|               | 2     | 10.0 | $10.0 \pm 0.0$   | 10.0    | $10.0 \pm 0.0$        | 0.0  | 10.0    | $10.0 \pm 0.0$        | 0.0  |
|               | 8     | 90.5 | $64.2 \pm 12.4$  | 89.8    | 89.6 ± 8.7            | 25.4 | 87.4    | $86.8 \pm 0.4$        | 22.6 |
|               | 7     | 90.3 | $66.5 \pm 8.5$   | 89.8    | $89.6 \pm 0.2$        | 23.1 | 87.4    | $86.7 \pm 0.3$        | 20.2 |
|               | 6     | 90.1 | $59.8 \pm 13.2$  | 89.9    | $89.4 \pm 3.8$        | 29.6 | 87.4    | $86.4 \pm 0.7$        | 26.6 |
| VGG19         | 5     | 90.2 | $37.7 \pm 14.1$  | 89.8    | $78.0\pm15.8$         | 40.3 | 87.2    | $79.8 \pm 0.8$        | 42.1 |
|               | 4     | 87.5 | $14.7 \pm 6.0$   | 89.8    | $27.8\pm18.9$         | 13.1 | 87.2    | $34.4\pm20.5$         | 19.7 |
|               | 3     | 78.3 | $10.5 \pm 1.5$   | 89.7    | $10.9 \pm 2.6$        | 0.4  | 86.8    | $11.0 \pm 2.9$        | 0.5  |
|               | 2     | 10.0 | $10.0 \pm 0.0$   | 10.0    | $10.0 \pm 0.0$        | 0.0  | 10.0    | $10.0 \pm 0.0$        | 0.0  |

Figure 12: Results of Reduced-precision and bit errors (0.5%) on CIFAR-10 pre-trained base models.

#### <span id="page-26-0"></span>H.2 GTSRB

Table 23: Reduced-precision Quantization and with 0.5% BER on GTSRB pre-trained models.

| Base     | //D:  |      | D.              |         | ConvL (1%)      |      |         | UNetL (1%)      |      |
|----------|-------|------|-----------------|---------|-----------------|------|---------|-----------------|------|
| Model    | #Bits | CA   | PA              | CA (NF) | PA (NF)         | RP   | CA (NF) | PA (NF)         | RP   |
|          | 8     | 95.5 | $75.2 \pm 12.7$ | 95.7    | $95.3 \pm 0.5$  | 20.1 | 96.2    | $95.7 \pm 0.3$  | 20.5 |
|          | 7     | 95.5 | $69.5 \pm 10.6$ | 95.7    | $95.3 \pm 0.3$  | 25.8 | 96.2    | $95.9 \pm 0.3$  | 26.4 |
|          | 6     | 95.4 | $67.2 \pm 14.4$ | 95.7    | $95.2 \pm 0.5$  | 28.0 | 96.2    | $95.7 \pm 0.5$  | 28.5 |
| ResNet18 | 5     | 95.4 | $48.6 \pm 18.2$ | 95.8    | $92.6 \pm 5.1$  | 44.0 | 96.2    | $94.8 \pm 2.5$  | 46.2 |
|          | 4     | 92.6 | $24.6 \pm 9.8$  | 95.9    | $75.6 \pm 16.2$ | 51.0 | 96.2    | $86.6 \pm 9.5$  | 62.0 |
|          | 3     | 67.7 | $5.3 \pm 3.5$   | 95.4    | $18.4 \pm 15.3$ | 13.1 | 96.2    | $25.3 \pm 22.5$ | 20.0 |
|          | 2     | 3.8  | $3.8 \pm 0.0$   | 4.1     | $3.8 \pm 0.0$   | 0.0  | 3.8     | $3.8 \pm 0.0$   | 0.0  |
|          | 8     | 95.5 | 69.1 ± 11.1     | 96.0    | $94.0 \pm 2.2$  | 24.9 | 95.4    | 93.9 ± 2.1      | 24.8 |
|          | 7     | 95.6 | $66.1 \pm 14.8$ | 96.0    | $92.2 \pm 5.7$  | 26.1 | 95.4    | $92.6 \pm 3.7$  | 26.5 |
|          | 6     | 95.3 | $64.2 \pm 8.4$  | 96.0    | $92.2 \pm 5.7$  | 28.0 | 95.4    | $92.3 \pm 2.3$  | 28.1 |
| VGG19    | 5     | 95.2 | $48.2 \pm 14.0$ | 96.0    | $92.2 \pm 5.7$  | 44.0 | 95.4    | $86.2 \pm 8.4$  | 38.0 |
|          | 4     | 92.0 | $18.2 \pm 14.3$ | 93.0    | $92.2 \pm 5.7$  | 74.0 | 95.0    | $49.6 \pm 22.8$ | 31.4 |
|          | 3     | 60.0 | $2.0 \pm 0.9$   | 87.3    | $92.2 \pm 5.7$  | 90.2 | 87.2    | $1.7 \pm 0.9$   | -0.3 |
|          | 2     | 5.9  | $3.8 \pm 0.0$   | 5.9     | $3.8 \pm 0.0$   | 0.0  | 5.9     | $3.8 \pm 0.0$   | 0.0  |

Figure 13: Results of Reduced-precision and bit errors (0.5%) on GTSRB pre-trained base models.

#### <span id="page-27-2"></span>H.3 ImageNet-10

Table 24: Reduced-precision Quantization and with 0.5% BER on ImageNet-10 pre-trained models.

| Base     | #Bits | CA   | PA              | C       | onvL (0.5%)     |      | U       | NetL (0.5%)     |      |
|----------|-------|------|-----------------|---------|-----------------|------|---------|-----------------|------|
| Model    | πDits | CA   | IA              | CA (NF) | PA (NF)         | RP   | CA (NF) | PA (NF)         | RP   |
|          | 8     | 92.2 | $72.3 \pm 7.0$  | 94.0    | $88.0 \pm 2.0$  | 15.7 | 94.0    | $88.1 \pm 1.4$  | 15.8 |
|          | 7     | 92.4 | $70.6 \pm 13.0$ | 94.2    | $86.7 \pm 4.1$  | 16.1 | 93.6    | $87.8 \pm 3.5$  | 17.2 |
|          | 6     | 92.4 | $68.9 \pm 9.9$  | 94.2    | $85.1 \pm 4.8$  | 16.2 | 93.6    | $86.4 \pm 3.7$  | 17.5 |
| ResNet18 | 5     | 91.0 | $60.9 \pm 13.0$ | 94.2    | $82.5 \pm 6.8$  | 21.6 | 94.0    | $83.2 \pm 5.9$  | 22.3 |
|          | 4     | 91.4 | $47.4 \pm 9.8$  | 93.8    | $68.6 \pm 9.8$  | 21.2 | 92.6    | $68.7 \pm 9.2$  | 21.3 |
|          | 3     | 85.2 | $28.8 \pm 11.8$ | 89.2    | $44.1 \pm 14.0$ | 15.3 | 89.4    | $42.7 \pm 14.2$ | 13.9 |
|          | 2     | 10.0 | $10.0 \pm 0.0$  | 10.0    | $10.0 \pm 0.0$  | 0.0  | 10.0    | $10.0 \pm 0.0$  | 0.0  |
|          | 8     | 92.4 | $37.2 \pm 11.0$ | 91.4    | $75.5 \pm 8.8$  | 38.3 | 89.4    | $77.9 \pm 6.1$  | 40.7 |
|          | 7     | 92.0 | $27.3 \pm 6.6$  | 91.2    | $59.3 \pm 13.0$ | 32.0 | 89.4    | $65.4 \pm 10.0$ | 38.1 |
|          | 6     | 92.4 | $27.9 \pm 6.4$  | 91.0    | $59.7 \pm 11.8$ | 31.8 | 89.4    | $64.9 \pm 9.9$  | 37.0 |
| VGG19    | 5     | 92.0 | $15.1 \pm 4.4$  | 91.6    | $23.1 \pm 0.7$  | 8.0  | 89.0    | $27.9 \pm 8.8$  | 12.8 |
|          | 4     | 89.4 | $12.2 \pm 2.7$  | 90.8    | $14.0 \pm 4.3$  | 1.8  | 89.6    | $14.6 \pm 4.9$  | 2.4  |
|          | 3     | 46.8 | $9.9 \pm 0.5$   | 83.2    | $10.4 \pm 0.6$  | 0.5  | 84.2    | $9.9 \pm 0.7$   | 0.0  |
|          | 2     | 10.0 | $10.0 \pm 0.0$  | 10.0    | $10.0 \pm 0.0$  | 0.0  | 10.0    | $10.0 \pm 0.0$  | 0.0  |

Note. CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

Figure 14: Results of Reduced-precision and bit errors (0.5%) on ImageNet-10 pre-trained base models.

#### <span id="page-27-1"></span>I Additional Experiments on Adversarial Training

Adversarial training is a common strategy to derive a robust neural network against certain perturbations. By training the generator using adversarial training proposed in Stutz et al. [2021], we report its performance against low voltage-induced bit errors. We use ConvL as the generator and ResNet18 as the base model, trained on CIFAR-10. Furthermore, we explore different K flip bits as the perturbation on weights of the base model during adversarial training, and then for evaluation, the trained-generator will be applied against 1% of bit errors rate on the base model. The results are shown in Table 25. After careful tuning of hyperparameters, we find that we are not able to obtain satisfactory recovery when adopting adversarial training. Empirically, we argue that adversarial training may not be suitable for training generator-based methods.

# <span id="page-27-0"></span>J Additional Experiments on Robust Model Trained with Adversarial Weight Perturbation with NeuralFuse

Previously, Wu et al. [2020] proposed that one could obtain a more robust model via adversarial weight perturbation. To seek whether such models could also be robust to random bit errors, we

<span id="page-28-1"></span>Table 25: Performance of the generator trained by adversarial training under K flip bits on ResNet18 with CIFAR-10. The results show that the generator trained by adversarial training cannot achieve high accuracy against bit errors under a 1% bit error rate.

| K-bits                        | CA   | PA          | CA (NF)                      | PA (NF)                                                                 | RP                       |
|-------------------------------|------|-------------|------------------------------|-------------------------------------------------------------------------|--------------------------|
| 100<br>500<br>5,000<br>20,000 | 92.6 | 38.9 ± 12.4 | 92.4<br>92.1<br>92.6<br>60.1 | $38.3 \pm 12.1$<br>$38.7 \pm 12.5$<br>$38.9 \pm 12.5$<br>$23.0 \pm 8.1$ | -0.6<br>-0.2<br>0<br>-16 |
| 100,000                       |      |             | 71.1                         | $23.0 \pm 6.1$ $23.6 \pm 6.6$                                           | -16                      |

Note. CA (%): clean accuracy; PA (%): perturbed accuracy; NF: Neural-Fuse; and RP: total recovery percentage of PA (NF) vs. PA

conducted an experiment on CIFAR-10 with the proposed adversarially trained PreAct ResNet18. The experimental results are shown in Table 26. We find that the average perturbed accuracy is 23% and 63.2% for PreAct ResNet18 under 1% and 0.5% BER, respectively. This result is lower than 38.9% and 70.1% from ResNet18 in Table 12, indicating their poor generalization ability against random bit errors. Nevertheless, when equipped NeuralFuse on the perturbed model, we could still witness a significant recover percentage under both 1% and 0.5% BER This result further demonstrates that NeuralFuse could be adapted to various models (i.e., trained in different learning algorithms).

<span id="page-28-2"></span>Table 26: Performance of NeuralFuse trained with rubust CIFAR-10 pre-trained PreAct ResNet18. The results show that NeuralFuse can be used together with a robust model and further improve perturbed accuracy under both 1% and 0.5% BER

| Base Model | BER  | NF                                                     | CA   | PA             | CA (NF)                                      | PA (NF)                                                                                            | RP                                           |
|------------|------|--------------------------------------------------------|------|----------------|----------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------|
| PreAct     | 1%   | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 89.7 | $23.0 \pm 9.3$ | 87.6<br>83.1<br>87.7<br>82.9<br>86.1<br>80.4 | $53.7 \pm 26$<br>$34.6 \pm 15$<br>$55.4 \pm 27$<br>$32.4 \pm 14$<br>$60.4 \pm 28$<br>$51.9 \pm 24$ | 30.7<br>11.6<br>32.4<br>9.4<br>37.4<br>28.9  |
| ResNet18   | 0.5% | ConvL<br>ConvS<br>DeConvL<br>DeConvS<br>UNetL<br>UNetS | 89.7 | 63.2 ± 8.7     | 89.2<br>89.2<br>89.0<br>89.9<br>87.5<br>88.2 | $87.8 \pm 1.1$ $74.0 \pm 6.5$ $87.4 \pm 1.1$ $74.4 \pm 7.0$ $85.9 \pm 0.8$ $80.4 \pm 3.9$          | 24.6<br>10.8<br>24.2<br>11.2<br>22.7<br>17.2 |

*Note.* BER: the bit-error rate of the base model; CA (%): clean accuracy; PA (%): perturbed accuracy; NF: NeuralFuse; and RP: total recovery percentage of PA (NF) vs. PA

## <span id="page-28-0"></span>**K** Data Embeddings Visualization

To further understand how our proposed NeuralFuse works, we visualize the output distribution from the final linear layer of the base models and project the results onto the 2D space using t-SNE [van der Maaten and Hinton, 2008]. Figure 15 shows the output distribution from ResNet18 (trained on CIFAR-10) under a 1% bit error rate. We chose two generators that have similar architecture: ConvL and ConvS, for this experiment. We can observe that: (a) The output distribution of the clean model without NeuralFuse can be grouped into 10 classes denoted by different colors. (b) The output distribution of the perturbed model under a 1% bit error rate without NeuralFuse shows mixed representations and therefore degraded accuracy. (c) The output distribution of the clean model with ConvL shows that applying NeuralFuse will not hurt the prediction of the clean model too much (i.e., it retains high accuracy in the regular voltage setting). (d) The output distribution of the perturbed model with ConvL shows high separability (and therefore high perturbed accuracy) as opposed to (b). (e)/(f) shows the output distribution of the clean/perturbed model with ConvS. For both (e) and (f),

we can see nosier clustering when compared to (c) and (d), which means the degraded performance of ConvS compared to ConvL. The visualization validates that NeuralFuse can help retain good data representations under random bit errors and that larger generators in NeuralFuse have better performance than smaller ones.

<span id="page-29-1"></span>Figure 15: t-SNE results for ResNet18 trained by CIFAR-10 under 1% of bit error rate. (a) Clean model. (b) Perturbed model. (c) Clean model with ConvL. (d) Perturbed model with ConvL. (e) Clean model with ConvS. (f) Perturbed model with ConvS.

## <span id="page-29-0"></span>L Qualitative Analysis of Transformed Inputs

In this section, we conduct a qualitative study to visualize the images transformed by NeuralFuse and present some properties and observations of these images. We utilize six different architectures of NeuralFuse generators trained with ResNet18 under a 1% bit error rate.

Figure [16](#page-30-0) (a) showcases several images from the *truck* class in CIFAR-10. Notably, images of the same class, when transformed by the same NeuralFuse, exhibit similar patterns, such as circles symbolizing the wheels of the trucks.

In Figures [16](#page-30-0) (b) and [16](#page-30-0) (c), we observe analogous phenomena in the GTSRB and CIFAR-100 datasets. Transformed images of the same class using the same generator consistently display patterns. On GTSRB, NeuralFuse-generated patterns highlight the sign's shape with a green background, even if the original images have a dark background and are under different lighting conditions. These results further underscore the efficacy and efficiency of NeuralFuse.

Figure [17](#page-30-1) presents more images from different classes in (a) CIFAR-10, (b) GTSRB, and (c) CIFAR-100. The transformed images exhibit distinct patterns for each class, suggesting that NeuralFuse effectively transforms images into class-specific patterns, making associated features robust to random bit errors and easily recognizable by the base model in low-voltage settings.

<span id="page-30-0"></span>Figure 16: Visualization of transformed images from different NeuralFuse generators trained with ResNet18 at 1% bit error rate.

<span id="page-30-1"></span>Figure 17: Visualization of transformed images from different NeuralFuse generators trained by ResNet18 with 1% bit error rate.

## NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We list our main contributions in the Abstract and Introduction.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitation of our work on runtime/latency in Section [4.4.](#page-6-1)

#### Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: This paper does not include theoretical results.

#### Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The implementation details are presented in Appendices [A](#page-12-1) and [B.](#page-12-2)

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

Answer: [Yes]

Justification: Our code can be found at <https://github.com/IBM/NeuralFuse>.

#### Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The experimental details are presented in Section [4.1,](#page-4-0) Appendix and our code.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For each experimental result, we report the mean and standard deviation and plot the error bars.

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The computing resources are presented in Section [4.1.](#page-4-0)

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: The work stated in this paper conform with the NeurIPS Code of Ethics.

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discussed broader impacts in Section [5.](#page-8-2)

#### Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This paper poses no such risks.

#### Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have cited the original paper that produced the code package or dataset.

## Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We described how to run our code in the README.md.

#### Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.