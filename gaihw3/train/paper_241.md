SuperEncoder: Towards Iteration-Free Approximate
Quantum State Preparation

Anonymous Author(s)
Affiliation
Address
email

Abstract

Numerous quantum algorithms operate under the assumption that classical data
1

has already been converted into quantum states, a process termed Quantum State
2

Preparation (QSP). However, achieving precise QSP requires a circuit depth that
3

scales exponentially with the number of qubits, making it a substantial obstacle in
4

harnessing quantum advantage. Recent research suggests using a Parameterized
5

Quantum Circuit (PQC) to approximate a target state, offering a more scalable
6

solution with reduced circuit depth compared to precise QSP. Despite this, the need
7

for iterative updates of circuit parameters results in a lengthy runtime, limiting its
8

practical application. To overcome this challenge, we introduce SuperEncoder,
9

a pre-trained classical neural network model designed to directly estimate the
10

parameters of a PQC for any given quantum state. By eliminating the need for
11

iterative parameter tuning, SuperEncoder represents a pioneering step towards
12

iteration-free approximate QSP.
13

1
Introduction
14

Quantum Computing (QC) leverages quantum mechanics principles to address classically intractable
15

problems [47, 36]. Various quantum algorithms have been developed, encompassing quantum-
16

enhanced linear algebra [15, 48, 45], Quantum Machine Learning (QML) [26, 19, 1, 33, 50, 3],
17

quantum-enhanced partial differential equation solvers [31, 13], etc. A notable caveat is that those
18

algorithms assume that classical data has been efficiently loaded into a specific quantum state, a
19

process known as Quantum State Preparation (QSP).
20

However, the realization of QSP presents significant challenges. Ideally, we expect each element of
21

the classical data to be precisely transformed into an amplitude of the corresponding quantum state.
22

This precise QSP is also known as Amplitude Encoding (AE). However, a critical yet unresolved
23

problem of AE is that the required circuit depth grows exponentially with respect to the number of
24

qubits [34, 41, 29, 46, 49]. Extensive efforts have been made to alleviate this issue, but they fail to
25

address it fundamentally. For example, while some methods introduce ancillary qubits for shallower
26

circuit [57, 56, 2], they may encounter an exponential number of ancillary qubits. Other methods aim
27

at preparing special quantum states with lower circuit depth, being only effective for either sparse
28

states [12, 32] or states with some special distributions [14, 17]. To summarize, realizing AE for
29

arbitrary quantum states still remains non-scalable due to its exponential resource requirement with
30

respect to the number of qubits. Moreover, in the Noisy Intermediate-Scale Quantum (NISQ) era [42],
31

hardware has limited qubit lifetimes and confronts a high risk of decoherence errors when executing
32

deep circuits, further exacerbating the problem of AE.
33

In fact, precise QSP is unrealistic in the present NISQ era due to the inherent errors of quantum
34

devices. Hence, iteration-based Approximate Amplitude Encoding (AAE) emerges as a promising
35

technique [59, 35, 52]. Specifically, AAE constructs a quantum circuit with tunable parameters, then
36

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
it iteratively updates the parameters to approximate a target quantum state. Since the updating of
37

parameters can be guided by states obtained from noisy devices, AAE is robust to noises, becoming
38

especially suitable for NISQ applications. More importantly, AAE has been shown to have shallow
39

circuit depth [35, 52], making it more scalable than AE.
40

4
6
8
Number of Qubits
0.00

0.25

0.50

0.75

1.00

Normalized Time

99.21%
99.72%
99.84%

Ttotal −TAAE
TAAE

Figure 1: Breakdown of normalized run-
time for QNN inference. Original data
are listed in Table 1.

Unfortunately, AAE possesses a drawback that signifi-
41

cantly undermines its potential advantages — the lengthy
42

runtime stemming from iterative optimizations of param-
43

eters. For example, when a Quantum Neural Network
44

(QNN) [3] is trained and deployed, the runtime of AAE
45

dominates the inference time as we demonstrated in Fig. 1.
46

Since loading classical data into quantum states becomes
47

the bottleneck, the potential advantage of QNN dimin-
48

ishes no matter how efficient the computations are done
49

on quantum devices.
50

Compared to AAE, AE employs a pre-defined arithmetic
51

decomposition procedure to construct a circuit, thereby
52

becoming much faster than AAE at runtime. Therefore,
53

it is natural to ask: can we realize both fast and scalable
54

methods for arbitrary QSP? This is precisely the question
55

we tackle in this paper. Overall, we present three major
56

contributions.
57

• Given a Parameterized Quantum Circuit (PQC) U(θ) that approximates a target quantum state,
58

with θ the parameter vector. We show that there exists a deterministic transformation f that could
59

map an arbitrary state |d⟩to its corresponding parameters θ. Consequently, the parameters can be
60

designated by f without time-intensive iterations.
61

• We show that the mapping f is learnable by utilizing a classical neural network model, which
62

we term as SuperEncoder. With SuperEncoder, you can have your cake and eat it too, i.e.,
63

simultaneously realizing fast and scalable QSP. We develop a prototype model and shed light on
64

insights into its training methodology.
65

• We verify the effectiveness of SuperEncoder on both synthetic dataset and representative down-
66

stream tasks, paving the way toward iteration-free approximate quantum state preparation.
67

2
Preliminaries
68

In this section, we commence with some basic concepts about quantum computing [36], and then
69

proceed to a brief retrospect of existing QSP methods.
70

2.1
Quantum Computation
71

We use Dirac notation throughout this paper. A pure quantum state is defined by a vector |·⟩named
72

‘ket’, with the unit length. A state can be written as |ψ⟩= PN
j=1 αj|j⟩with P

j |αj|2 = 1, where
73

|j⟩denotes a computational basis state and N represents the dimension of the complex vector
74

space. Density operators describe more general quantum states. Given a mixture of m pure states
75

{|ψi⟩}m
i=1 with probabilities pi and Pm
i pi = 1, the density operator ρ denotes the mixed state as
76

ρ = Pm
i=1 pi|ψi⟩⟨ψi| with Tr(ρ) = 1, where ⟨·| refers to the conjugate transpose of |·⟩. Generally,
77

we use the term fidelity to describe the similarity between an erroneous quantum state and its
78

corresponding correct state.
79

The fundamental unit of quantum computation is the quantum bit, or qubit. A qubit’s state can be
80

expressed as ψ = α|0⟩+ β|1⟩. Given n qubits, the state is generalized to |ψ⟩= P2n

j |j⟩, where
81

|j⟩= |j1j2 · · · jn⟩with jk the state of kth qubit in computational basis, and j = Pn
k=1 2n−kjk.
82

Applying quantum operations evolves a system from one state to another. Generally, these operations
83

can be categorized into quantum gates and measurements. Typical single-qubit gates include the
84

Pauli gates X ≡[ 0 1
1 0 ], Y ≡
 0 −i
i
0

, Z ≡
 1
0
0 −1

. These gates have associated rotation operations
85

RP (θ) ≡e−iθP/2, where θ is the rotation angle and P ∈{X, Y, Z}1. Muti-qubit operations create
86

1In this paper, Rz, Ry are equivalent to RZ, RY .

2


---Page Break---
entanglement between qubits, allowing one qubit to interfere with others. In this work, we focus on
87

the controlled-NOT (CNOT) gate, with the mathematical form of CNOT ≡|0⟩⟨0| ⊗I2 + |1⟩⟨1| ⊗X.
88

Quantum measurements extract classical information from quantum states, which is described by
89

a collection {Mm} with P

m M †
mMm = I. Here, m refers to the measurement outcomes that may
90

occur in the experiment, with a probability of p(m) = ⟨ψ|M †
mMm|ψ⟩. The post-measurement state
91

of the system becomes Mm|ψ⟩/p(m).
92

A quantum circuit is the graphical representation of a series of quantum operations, which can be
93

mathematically represented by a unitary matrix U. In the NISQ era, PQC plays an important role
94

as it underpins variational quantum algorithms [11, 39]. Typical PQC has the form of U(θ) =
95
Q

i Ui(θi)Vi, where θ is its parameter vector, Ui(θi) = e−iθiPi/2 with Pi denoting a Pauli gate, and
96

Vi denotes a fixed gate such as CNOT. For example, a PQC composed of Ry gates and CNOT gates
97

is depicted in Fig. 2.
98

|0⟩
Ry(θ0)
Ry(θ4)

|0⟩
Ry(θ1)
Ry(θ5)

|0⟩
Ry(θ2)
Ry(θ6)

|0⟩
Ry(θ3)
Ry(θ7)

Block # 0
Block # 1

Approximated state of |d⟩

Figure 2: An example PQC with two blocks, with each block consisting of a rotation layer (filled
blue) plus an entangler layer (filled red).

2.2
Quantum State Preparation
99

Successful execution of many quantum algorithms requires an initial step of loading classical data
100

into a quantum state [5, 15], a process known as quantum state preparation. This procedure involves
101

implementing a quantum circuit to evolve a system to a designated state. Here, we focus on amplitude
102

encoding and formalize its procedure as follows. Let d be a real-valued N-dimensional classical
103

vector, AE encodes d into the amplitudes of an n-qubit quantum state |d⟩, where N = 2n. More
104

specifically, the data quantum state is represented by |d⟩= PN−1
j=0 dj|j⟩, where dj denotes the jth
105

element of the vector d, and |j⟩refers to a computational basis state. The main objective is to generate
106

a quantum circuit U that initializes an n-qubit system by U|0⟩⊗n = PN−1
j=0 αj|j⟩, whose amplitudes
107

{αj} are equal to {dj}. It is widely recognized that constructing such a circuit generally necessitates
108

a circuit depth that scales exponentially with n [34, 41]. This property makes AE impractical in
109

current NISQ era, as decoherence errors [23] can severely dampen the effectiveness of AE as the
110

number of qubits increases [52].
111

In response to the inherent noisy nature of current devices, approximate amplitude encoding has
112

emerged as a promising technique [59, 35, 52]. Specifically, AAE utilizes a PQC (a.k.a. ansatz) to
113

approximate the target quantum state by iteratively updating the parameters of circuit, following
114

a similar procedure of other variational quantum algorithms [39, 11]. AAE has been shown to be
115

more advantageous for NISQ devices due to its ability to mitigate coherent errors through flexible
116

adjustment of circuit parameters, coupled with its lower circuit depth [52]. We denote an ansatz as
117

U(θ), where θ refers to a vector of tunable parameters for optimizations. A typical ansatz consists
118

of several blocks of operations with the same structure. For example, a two-block ansatz with 4
119

qubits is shown in Fig. 2, where the rotation layer is composed of single-qubit rotational gates
120

Ry(θr) = e−iθrY/2, and the entangler layer comprises CNOT gates. Note that the entangler layer is
121

configurable and hardware-native, which means that we can apply CNOT gates to physically adjacent
122

qubits, thereby eliminating the necessity of additional SWAP gates to overcome the topological
123

constraints [27]. This type of PQC is also known as hardware-efficient ansatz [20], being widely
124

adopted in previous studies of AAE [59, 35, 52].
125

3


---Page Break---
3
SuperEncoder
126

3.1
Motivation
127

Although AAE can potentially realize high fidelity QSP with O(poly(n)) circuit depth [35] with n
128

the number of qubits, it requires repetitive online tuning of parameters to approximate the target
129

state, which may result in an excessively long runtime that undermines its feasibility. Specifically, we
130

could consider a simple application scenario in QML. The workflow with AAE is depicted in Fig. 3a.
131

During the inference stage, we must iteratively update the parameters of the AAE ansatz for each
132

input classical data vector, which may greatly dampen the performance. To quantify this impact, we
133

measure the runtime of AAE-based data loading and the total runtime of model inference. As one can
134

observe from Table 1, AAE dominates the runtime, thereby becoming the performance bottleneck.
135

n
TAAE (s)
Ttotal −TAAE (s)
4
5.0086
0.0397
6
20.1810
0.0573
8
59.4193
0.0978
Table 1: Performance overhead of AAE. We break down the averaged inference runtime per sample
from the MNIST dataset. TAAE denotes time spent on loading classical data into quantum state using
AAE, and Ttotal refers to total runtime.

The necessity of time-intensive iterations is grounded in the following assumption — Given an
136

arbitrary quantum state |ψ⟩, there does not exist a deterministic transformation f : |ψ⟩→θ, where
137

θ refers to the vector of parameters enabling a PQC to prepare an approximated state of |ψ⟩. This
138

assumption seems intuitively correct given the randomness of target states. However, we argue that a
139

universal mapping f exists for any arbitrary data state |ψ⟩. Taking a little thought of AE, we see that
140

it implies the following conclusion: given an arbitrary state |ψ⟩, there exists an universal arithmetic
141

decomposition procedure g : |ψ⟩→U satisfying U|0⟩= |ψ⟩. Inspired by this deterministic
142

transformation, it is natural to ask: is there an universal transformation g′ : |ψ⟩→U ′ satisfying
143

E(U ′|0⟩, |ψ⟩) ≤ϵ? Here E denotes the deviation between the prepared state by a circuit U ′ and the
144

target state, and ϵ refers to certain acceptable error threshold. Since the structure of PQC in AAE
145

is the same for any target state, U ′ is determined by θ. Then, the problem is reduced to exploring
146

the existence of f : |ψ⟩→θ. Should f exist, the overhead of online iterations could be eliminated,
147

resulting in a novel QSP method being both fast and scalable.
148

Parameterized Quantum Circuit (PQC)

PQC

Classical Data

Data Encoding

AAE

Quantum State

Quantum Neural

Network Layers

Classification Result

Iteration #0

Iteration #1

(a) Inference process of AAE.

Parameterized Quantum Circuit (PQC)
Classical Data

Data Encoding
SuperEncoder

Quantum State

Quantum Neural

Network Layers

Classification Result

(b) Inference process of SuperEncoder.

Figure 3: Comparison between AAE and SuperEncoder.

4


---Page Break---
3.2
Design Methodology
149

Let |ψ⟩be the target state, and U(θ) be the PQC used in AAE with θ the optimized parameters.
150

Our goal is to develop a model, termed SuperEncoder, to approximate the mapping f : |ψ⟩→θ.
151

Referring back to the scenario in QML, the workflow with SuperEncoder becomes iteration-free, as
152

depicted in Fig. 3b.
153

Since neural networks could be used to approximate any continuous function [6], a natural solution is
154

to use a neural network to approximate f. Specifically, we adopt a Multi-Layer Perceptron (MLP) as
155

the backbone model for approximating f. However, training this model is nontrivial. Particularly, we
156

find it challenging to design a proper loss function. In the remainder of this section, we explore three
157

different designs and analyze their performance.
158

(a) Target state.
(b) SuperEncoder-L1 (c) SuperEncoder-L3

Figure 4: Virtualization of states generated by SuperEncoder trained with different loss functions. L2
is omitted as it produces very similar results to L3.

The first and most straightforward method is parameter-oriented training — setting the loss function
159

L1 as the MSE between the target parameters θ from AAE and the output parameters ˆθ from
160

SuperEncoder. To evaluate the performance of L1, we train a SuperEncoder using MNIST dataset,
161

and test if it could load a test digit image into a quantum state with high fidelity. All images are
162

downsampled and normalized into 4-qubit states for quick evaluation.
163

L1
L2
L3
0.6208
0.9873
0.9908

Table 2: Fidelity comparison be-
tween SuperEncoders trained with
different loss functions.

Unfortunately, results in Table 2 show that L1 achieves poor
164

performance. The average fidelity of prepared quantum states
165

is only 0.6208. As demonstrated in Fig. 4, L1 generates a state
166

that losses the patterns of the original state. Additionally, utiliz-
167

ing L1 implies that we need to first generate target parameters
168

using AAE, of which the long runtime hinders pre-training on
169

larger datasets. Consequently, required is a more effective loss
170

function design without involving AAE.
171

0
200
400
600
800
Step

0.00

0.25

0.50

0.75

1.00

Loss

L1
L2
L3

Figure 5: Convergence of dif-
ferent loss functions.

To address this challenge, we propose a state-oriented training
172

methodology, which employs quantum states as targets to guide
173

optimizations. Specifically, we may apply ˆθ to the circuit and exe-
174

cute it to obtain the prepared state ˆψ. Then it is possible to calculate
175

the difference between ˆψ and ψ as the loss to optimize SuperEncoder.
176

In contrast to parameter-oriented training, this approach applies to
177

larger datasets as it decouples the training procedure from AAE. We
178

utilize two different state-oriented metrics, the first being the MSE
179

between ˆψ and ψ, denoted as L2, and the second is the fidelity of
180

ˆψ relative to ψ, expressed as L3 = 1 −|⟨ˆψ|ψ⟩|2 [25]. Results in
181

Table 2 show that L2 and L3 achieve remarkably higher fidelity than
182

L1. Besides, we observe that L3 prepares a state very similar to the
183

target one (Fig. 4), verifying that state-oriented training is more effective than parameter-oriented
184

training.
185

Landscape Analysis. To understand the efficacy of these loss functions, we further analyze their
186

landscapes following previous studies [28, 40, 18]. To gain insight from the landscape, we plot Fig. 6
187

using the same scale and color gradients [18]. Compared to state-oriented losses (L2 and L3), L1 has
188

a largely flat landscape with non-decreasing minima, thus the model struggles to explore a viable
189

path towards a lower loss value, a similar pattern can also be observed in Fig. 5. In contrast, L2
190

5


---Page Break---
0
20
40
60
80 100
0

20

40

60

80

100

0.0

0.5

1.0

1.5

2.0

(a) L1

0
20
40
60
80 100
0

20

40

60

80

100

0.0

0.5

1.0

1.5

2.0

(b) L2

0
20
40
60
80 100
0

20

40

60

80

100

0.0

0.5

1.0

1.5

2.0

(c) L3

Figure 6: Landscape virtualization of different loss functions.

and L3 have much lower minima and successfully converge to smaller loss values. Furthermore, we
191

observe from Fig. 6 that L3 has a wider minima than L2, which may indicate a better generalization
192

capability [40].
193

Gradient Analysis. Based on the landscape analysis, we adopt L3 as the loss function to train
194

SuperEncoder. We note that L3 can be written as 1 −⟨ψ| ˆψ⟩⟨ˆψ|ψ⟩. If ˆρ is a pure state, it is equivalent
195

to | ˆψ⟩⟨ˆψ|. Then L3 is given by L3 = 1 −⟨ψ|ˆρ|ψ⟩.
196

This re-formalization is important as only the mixed state ˆρ could be obtained in noisy environments.
197

Suppose an n-qubit circuit is parameterized by m parameters ˆθ = [ˆθ1, . . . , ˆθk, . . . , ˆθm]. Let W be
198

the weight matrix of MLP, with k, l the element indices. We analyze the gradient of L3 w.r.t. Wk,l to
199

showcase its feasibility in different quantum computing environments.
200

∇Wk,lL3 = ∂L3

∂Wk,l
= −⟨ψ|
∂ˆρ
∂Wk,l
|ψ⟩

= −⟨ψ|





Pm
j=1
∂ˆρ1,1

∂θj
∂θj
∂Wk,l
· · ·
Pm
j=1
∂ˆρ1,N

∂θj
∂θj
∂Wk,l
...
...
...
Pm
j=1
∂ˆρN,1

∂θj
∂θj
∂Wk,l
· · ·
Pm
j=1
∂ˆρN,N

∂θj
∂θj
∂Wk,l



|ψ⟩,
(1)

The calculation of
∂θj
∂Wk,l can be easily done on classical devices using backpropagation supported by
201

automatic differentiation frameworks. Therefore, we only focus on ∂ˆρi,j

∂θk . In a simulation environ-
202

ment, the calculation of ˆρ is conducted via noisy quantum circuit simulation, which is essentially a
203

series of tensor operations on state vectors. Therefore, the calculation of ∂ˆρi,j

∂θk is compatible with
204

backpropagation. The situation on real devices becomes more complicated. On real devices, the
205

mixed state ˆρ is reconstructed through quantum tomography [7] based on classical shadow [55, 16].
206

Here, for notion simplicity, we denote the process of classical shadow as a transformation S, and
207

denote the measurement expectations of the ansatz as U(ˆθ). Thus the reconstructed density ma-
208

trix is given by ˆρ = S(U(ˆθ)). Then the gradient of ˆρi,j with respect to ˆθk is P
u
∂ˆρi,j
∂U(ˆθ)
∂U(ˆθ)

∂ˆθk .
209

Here
∂ˆρi,j
∂U(ˆθ) can be efficiently calculated on classical devices using backpropagation, as S operates
210

on expectation values on classical devices. However, U(ˆθ) involves state evolution on quantum
211

devices, where back-propagation is impossible due to the No-Cloning theorem [36]. Fortunately,
212

it is possible to utilize the parameter shift rule [8, 4, 53] to calculate ∂U(ˆθ)

∂θk . In this way, the
213

gradients of the circuit function U with respect to θj are ∂U(ˆθ)

∂θk
=
1
2 (U(θ+) −U(θ−)), where
214

θ+ = [θ1, . . . , θk + π

2 , . . . , θm], θ−= [θ1, . . . , θk −π

2 , . . . , θm]. To summarize, training SuperEn-
215

coder with L3 is theoretically feasible on both simulators and real devices.
216

6


---Page Break---
4
Numerical Results
217

4.1
Experiment Setup
218

Datasets. To train a SuperEncoder for arbitrary quantum states, we need a dataset comprising a wide
219

range of quantum states with different distributions. To our knowledge, there is no dataset dedicated
220

for this special purpose. A natural solution is to use readily available datasets from classical machine
221

learning domains (e.g., ImageNet [9], Places [58], SQuAD [44]) by normalizing them to quantum
222

states. However, QSP is essential in various application scenarios besides QML. The classical data to
223

be loaded may not only contain natural images or languages but also contain arbitrary data (e.g., in
224

HHL algorithm [15]). Therefore, we construct a training dataset adapted from FractalDB-60 [21] with
225

60k samples, a formula-driven dataset originally designed for computer vision without any natural
226

images. We also construct a separate dataset to test the performance of QSP, which consists of data
227

sampled from different statistical distributions, including uniform, normal, log-normal, exponential,
228

and Dirichlet distributions, with 3000 samples per distribution. Hereafter we refer this dataset as the
229

synthetic dataset.
230

Platforms. We implement SuperEncoder using PennyLane [34], PyTorch [37] and Qiskit [43].
231

Simulations are done on a Ubuntu server with 768 GB memory, two 32-core Intel(R) Xeon(R) Silver
232

4216 CPU with 2.10 GHz, and 2 NVIDIA A-100 GPUs. IBM quantum cloud platform2 is adopted to
233

evaluate the performance on real quantum devices.
234

Metrics. We evaluate SuperEncoder and compare it to AE and AAE in terms of runtime, scalability,
235

and fidelity. Runtime refers to how long it takes to prepare a quantum state. Scalability refers to how
236

the circuit depth grows with the number of qubits. Fidelity evaluates the similarity between prepared
237

quantum states and target quantum states. Specifically, the fidelity for two mixed states given by
238

density matrices ρ and ˆρ is defined as F(ρ, ˆρ) = Tr
 p√ρˆρ√ρ
2 ∈[0, 1]. A larger F indicates a
239

better fidelity.
240

Implementation. We implement SuperEncoder using an MLP consisting of two hidden layers.
241

The dimensions of input and output layers are respectively set to 2n and m, where n refers to the
242

number of qubits and m refers to the number of parameters. We adopt L3 as the loss function.
243

Training data are down-sampled, flattened, and normalized to 2n-dimensional state vectors. We
244

adopt the hardware efficient ansatz [20] (Fig. 2) as the backbone of quantum circuits and use the
245

same structure for AAE. Given a target state, a pre-trained SuperEncoder model is invoked to
246

generate parameters and thus the circuit for QSP. While for AAE, we employ online iterations for
247

each state. For AE, the arithmetic decomposition method in PennyLane [34, 4] is adopted. We
248

defer more details about implementation to Appendix A. Our framework is open-source at https:
249

//anonymous.4open.science/r/SuperEncoder-A733 with detailed instructions to reproduce
250

our results.
251

4.2
Evaluation on Synthetic Dataset
252

For simplicity and without loss of generality, we focus our discussion on the results of 4-qubit QSP
253

tasks. The outcomes for larger quantum states are detailed in Appendix B.1. The parameters of both
254

AAE and SuperEncoder are optimized based on ideal quantum circuit simulation.
255

Runtime. The runtime and fidelity results, evaluated on the synthetic dataset, are presented in Table 3.
256

We observe that SuperEncoder runs faster than AAE by orders of magnitudes and has a similar
257

runtime to AE, affirming that SuperEncoder effectively overcomes the main drawback of AAE.
258

AE
AAE
SuperEncoder
Fidelity
Runtime
Fidelity
Runtime
Fidelity
Runtime
Uniform
0.9996
0.9731
Normal
0.9992
0.8201
Log-normal
0.9993
0.9421
Exponential
0.9996
0.9464
Dirichlet
0.9995
0.9737
Average
1.0000
0.0162 s
0.9994
5.0201 s
0.9310
0.0397 s
Table 3: Comparison between AE, AAE and SuperEncoder in terms of runtime and fidelity.

2https://quantum-computing.ibm.com/

7


---Page Break---
4
6
8
Number of Qubits

0

250

500

750

1000

Circuit Depth

AE
AAE/SuperEncoder

(a) Scaling of circuit depth w.r.t. # qubits.

4
6
8
Number of Qubits

0.00

0.25

0.50

0.75

1.00

Fidelity

0.0049

AE
AAE
SuperEncoder

(b) Fidelity of different QSP methods on ibm_osaka.

Figure 7: Comparison between AE, AAE, and SuperEncoder in terms of circuit depth and fidelity on
real devices.

Scalability. Although AE runs fast, it exhibits poor scalability since the circuit depth grows exponen-
259

tially with the number of qubits. The depth of AAE is empirically determined by increasing depth
260

until the final fidelity does not increase, same depth is adopted for SuperEncoder. We deter the details
261

of determining the depth of AAE/SuperEncoder to Appendix A. As shown in Fig. 7a, the depth of
262

AE grows fast and becomes much larger than AAE/SuperEncoder, e.g., the depth of AE for a 8-qubit
263

state is 984, whereas the depth of AAE/SuperEncoder is only 120.
264

. . .

. . .

. . .

. . .

Encoder Block
U(ϕ0)
U(ϕ1)
U(ϕm)

AE
AAE
SuperEncoder
97.15%
98.01%
97.87%

Figure 8: Schematic of a QNN (above)
and test accuracies of QSP methods on
the QML task (below).

Fidelity. From Table 3, it is evident that SuperEncoder ex-
265

periences notable fidelity degradation when compared with
266

AAE and AE. Specifically, the average fidelity of SuperEn-
267

coder is 0.9307, whereas AAE and AE achieve higher av-
268

erage fidelities of 0.9994 and 1.0, respectively. Note that,
269

although AE demonstrates the highest fidelity under ideal
270

simulation, its performance deteriorates significantly in
271

noisy environments. Fig. 7b presents the performance of
272

these three QSP methods on quantum states with 4, 6, and
273

8 qubits on the ibm_osaka machine. While the fidelity
274

of AE is higher than AAE/SuperEncoder on the 4-qubit
275

and 6-qubit states, its fidelity on the 8-qubit state is only
276

0.0049, becoming much lower than AAE/SuperEncoder.
277

This decline is primarily attributed to its large circuit depth as shown in Fig. 7a.
278

4.3
Application to Downstream Tasks
279

q0 :

QSP

QPE
QPE inv

q1 :

q2 :

q3 :

q4 :

R

q5 :

q6 :

q7 :

q8 :

q9 :

a0 :

a1 :

a2 :

q10 :

Figure 9: Schematic of HHL.

Quantum Machine Learning. We first apply SuperEncoder to
280

a QML task. MNIST dataset is adopted for demonstration, we
281

extract a sub-dataset composed on digits 3 and 6 for evaluation.
282

The quantum circuit that implements a QNN is depicted in Fig. 8,
283

which consists of an encoder block and m entangler layers. Here
284

the encoder block is implemented via QSP circuits, either AE, AAE,
285

or SuperEncoder, of which the parameters are frozen during the
286

training of QNN. The test results are shown in Fig. 8, we observe
287

that SuperEncoder achieves similar performance with AAE and AE.
288

The reason lies in the fact that classification tasks can be robust to
289

noises. Consequently, approximate QSP (AAE and SuperEncoder)
290

with a certain degree of fidelity loss is tolerable.
291

HHL Algorithm. Besides QML, quantum-enhanced linear algebra
292

algorithms are another important set of applications that heavily rely
293

on QSP. The most famous algorithm is the HHL algorithm [15]. The
294

problem can be defined as, given a matrix A ∈CN×N, and a vector b ∈CN, find x ∈CN satisfying
295

Ax = b. A typical implementation of HHL utilizes the circuit depicted in Fig. 9. The outline of
296

HHL is as follows. (i) Apply a QSP circuit to prepare the quantum state |b⟩. (ii) Apply Quantum
297

Phase Estimation [10] (QPE) to estimate the eigenvalue of A (iii) Apply conditioned rotation gates
298

on ancillary qubits based on the eigenvalues (R). (iv) Apply an inverse QPE (QPE_inv) and measure
299

the ancillary qubits to reconstruct the solution vector x. Note that, HHL does not return the solution x
300

itself, but rather an approximation of the expectation value of some operator M associated with x, e.g.,
301

8


---Page Break---
x†Mx. Here, we adopt an optimized version of HHL proposed by Vazquez et al. [51] for evaluation.
302

To compare the performance between different QSP methods, we construct linear equations with
303

fixed matrix A and operator M, while we sample different vectors from our synthetic dataset as b.
304

Results are concluded in Table 4. Unlike QML, HHL expects precise QSP, thus we take the results
305

from AE as the ground truth values and compare the relative error between AAE/SuperEncoder and
306

AE. The relative error of SuperEncoder is 2.4094%, while the error of AAE is only 0.3326%.
307

4.4
Discussion and Future Work
308

AE
AAE
SuperEncoder
b0
0.7391
0.7404
0.7355
b1
0.7449
0.7445
0.7544
b2
0.7492
0.7469
0.8134
b3
0.7164
0.7099
0.7223
b4
0.7092
0.7076
0.7155
Avg err
0.3326%
2.4094%

Table 4: Performance of different QSP
methods in HHL algorithm. ‘Avg err’ de-
notes the average relative errors between
AAE/SuperEncoder and AE.

The results of our evaluation can be concluded in two folds.
309

(i) SuperEncoder effectively eliminates the iteration over-
310

head of AAE, thereby becoming both fast and scalable.
311

However, it has a notable degradation in fidelity. (ii) The
312

impact of fidelity degradation varies across different down-
313

stream applications. For QML, the fidelity degradation is
314

affordable as long as the prepared states are distinguish-
315

able across different classes. However, algorithms like
316

HHL rely on precise QSP to produce the best result. In
317

these algorithms, SuperEncoder suffers from higher error
318

ratio than AAE.
319

Note that, the current evaluation results may not reflect the
320

actual performance of SuperEncoder on real NISQ devices.
321

Recent work has shown that AAE achieves significantly better fidelity than AE does [52]. This is due
322

to the intrinsic noise awareness of AAE, as it could obtain states from noisy devices to guide updating
323

parameters with better robustness. In essence, the proposed SuperEncoder possesses the same nature
324

as AAE. Unfortunately, although the noise-robustness of AAE can be evaluated on a small set of test
325

samples, it is difficult to perform noise-aware training for SuperEncoder as it requires a large dataset
326

for pre-training. Consequently, SuperEncoder relies on huge amounts of interactions with noisy
327

devices, thereby becoming extremely time-consuming. As a result, the effectiveness of SuperEncoder
328

in noisy environments remains largely unexplored, which we leave for future exploration. More
329

discussion about this perspective is in Appendix C.
330

5
Related Work
331

Besides QSP, there are other methods for loading classical data into quantum states. These methods
332

can be roughly regarded as quantum feature embedding primarily used in QML, which maps classical
333

data to a completely different distribution encoded in quantum states. A widely used embedding
334

method is known as angle embedding. Li et al. have proven that this method has a concentration issue,
335

which means that the encoded states may become indistinguishable as the circuit depth increases [26].
336

Lei et al. proposed an automatic design framework for efficient quantum feature embedding, resolving
337

the issue of concentration [24]. The central idea of this framework is to search for the most efficient
338

circuit architecture for a given classical input, which is also known as Quantum Architecture Search
339

(QAS) [38, 30, 54]. While the application scenario of quantum feature embedding is largely limited
340

to QML, QSP has broader usage in general quantum applications, distinguishing SuperEncoder from
341

all aforementioned work.
342

6
Conclusion
343

In this work, we propose SuperEncoder, a neural network-based QSP framework. Instead of iteratively
344

tuning the circuit parameters to approximate each quantum state, as is done in AAE, we adopt a
345

different approach by directly learning the relationship between target quantum states and the required
346

circuit parameters. SuperEncoder combines the scalable circuit architecture of AAE with the fast
347

runtime of AE, as verified by a comprehensive evaluation on both synthetic dataset and downstream
348

applications.
349

9


---Page Break---
References
350

[1] Amira Abbas, David Sutter, Christa Zoufal, Aurélien Lucchi, Alessio Figalli, and Stefan
351

Woerner. The power of quantum neural networks. Nature Computational Science, 1(6):403–409,
352

2021.
353

[2] Israel F Araujo, Daniel K Park, Teresa B Ludermir, Wilson R Oliveira, Francesco Petruccione,
354

and Adenilton J Da Silva. Configurable sublinear circuits for quantum state preparation.
355

Quantum Information Processing, 22(2):123, 2023.
356

[3] Johannes Bausch. Recurrent quantum neural networks. Advances in neural information
357

processing systems, 33:1368–1379, 2020.
358

[4] Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Shahnawaz Ahmed, Vishnu
359

Ajith, M Sohaib Alam, Guillermo Alonso-Linaje, B AkashNarayanan, Ali Asadi, et al. Pen-
360

nylane: Automatic differentiation of hybrid quantum-classical computations. arXiv preprint
361

arXiv:1811.04968, 2018.
362

[5] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth
363

Lloyd. Quantum machine learning. Nature, 549(7671):195–202, 2017.
364

[6] Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural
365

networks with arbitrary activation functions and its application to dynamical systems. IEEE
366

Transactions on Neural Networks, 6(4):911–917, 1995.
367

[7] Marcus Cramer, Martin B Plenio, Steven T Flammia, Rolando Somma, David Gross, Stephen D
368

Bartlett, Olivier Landon-Cardinal, David Poulin, and Yi-Kai Liu. Efficient quantum state
369

tomography. Nature communications, 1(1):149, 2010.
370

[8] Gavin E Crooks. Gradients of parameterized quantum gates using the parameter-shift rule and
371

gate decomposition. arXiv preprint arXiv:1905.13311, 2019.
372

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-
373

scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern
374

recognition, pages 248–255. Ieee, 2009.
375

[10] Uwe Dorner, Rafal Demkowicz-Dobrzanski, Brian J Smith, Jeff S Lundeen, Wojciech
376

Wasilewski, Konrad Banaszek, and Ian A Walmsley. Optimal quantum phase estimation.
377

Physical review letters, 102(4):040403, 2009.
378

[11] Edward Farhi, Jeffrey Goldstone, and Sam Gutmann. A quantum approximate optimization
379

algorithm. arXiv preprint arXiv:1411.4028, 2014. https://doi.org/10.48550/arXiv.
380

1411.4028.
381

[12] Niels Gleinig and Torsten Hoefler. An efficient algorithm for sparse quantum state preparation.
382

In 2021 58th ACM/IEEE Design Automation Conference (DAC), pages 433–438. IEEE, 2021.
383

[13] Javier Gonzalez-Conde, Ángel Rodríguez-Rozas, Enrique Solano, and Mikel Sanz. Simulating
384

option price dynamics with exponential quantum speedup. arXiv preprint arXiv:2101.04023,
385

2021.
386

[14] Javier Gonzalez-Conde, Thomas W Watts, Pablo Rodriguez-Grasa, and Mikel Sanz. Efficient
387

quantum amplitude encoding of polynomial functions. Quantum, 8:1297, 2024.
388

[15] Aram W Harrow, Avinatan Hassidim, and Seth Lloyd. Quantum algorithm for linear systems
389

of equations. Physical Review Letters, 103(15):150502, 2009. https://doi.org/10.1103/
390

PhysRevLett.103.150502.
391

[16] Hsin-Yuan Huang. Learning quantum states from their classical shadows. Nature Reviews
392

Physics, 4(2):81–81, 2022.
393

[17] Jason Iaconis, Sonika Johri, and Elton Yechao Zhu. Quantum state preparation of normal
394

distributions using matrix product states. npj Quantum Information, 10(1):15, 2024.
395

10


---Page Break---
[18] Christian Cmehil-Warn Jacob Hansen. Loss landscapes. In ICLR Blog Track, 2022. https://loss-
396

landscapes.github.io/Loss-Landscapes-Blog/2022/12/01/loss-landscapes/.
397

[19] Weiwen Jiang, Jinjun Xiong, and Yiyu Shi. A co-design framework of neural networks and
398

quantum circuits towards quantum advantage. Nature Communications, 12(1):579, 2021.
399

https://doi.org/10.1038/s41467-020-20729-5.
400

[20] Abhinav Kandala, Antonio Mezzacapo, Kristan Temme, Maika Takita, Markus Brink, Jerry M.
401

Chow, and Jay M. Gambetta. Hardware-efficient variational quantum eigensolver for small
402

molecules and quantum magnets. Nature, 549(7671):242–246, September 2017.
403

[21] Hirokatsu Kataoka, Kazushige Okayasu, Asato Matsumoto, Eisuke Yamagata, Ryosuke Yamada,
404

Nakamasa Inoue, Akio Nakamura, and Yutaka Satoh. Pre-training without natural images. In
405

Proceedings of the Asian Conference on Computer Vision, 2020.
406

[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
407

arXiv:1412.6980, 2014.
408

[23] Philip Krantz, Morten Kjaergaard, Fei Yan, Terry P Orlando, Simon Gustavsson, and William D
409

Oliver. A quantum engineer’s guide to superconducting qubits. Applied physics reviews, 6(2),
410

2019.
411

[24] Cong Lei, Yuxuan Du, Peng Mi, Jun Yu, and Tongliang Liu. Neural auto-designer for enhanced
412

quantum kernels. In The Twelfth International Conference on Learning Representations, 2023.
413

[25] Nelson Leung, Mohamed Abdelhafez, Jens Koch, and David Schuster. Speedup for quantum
414

optimal control from automatic differentiation based on graphics processing units. Physical
415

Review A, 95(4):042318, 2017. https://doi.org/10.1103/PhysRevA.95.042318.
416

[26] Guangxi Li, Ruilin Ye, Xuanqiang Zhao, and Xin Wang. Concentration of data encoding in
417

parameterized quantum circuits. Advances in Neural Information Processing Systems, 35:19456–
418

19469, 2022.
419

[27] Gushu Li, Yufei Ding, and Yuan Xie. Tackling the qubit mapping problem for nisq-era quantum
420

devices.
In Proceedings of the Twenty-Fourth International Conference on Architectural
421

Support for Programming Languages and Operating Systems, pages 1001–1014, 2019. https:
422

//doi.org/10.1145/3297858.3304023.
423

[28] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss
424

landscape of neural nets. Advances in neural information processing systems, 31, 2018.
425

[29] Gui-Lu Long and Yang Sun. Efficient scheme for initializing a quantum register with an
426

arbitrary superposed state. Physical Review A, 64(1):014303, 2001.
427

[30] Xudong Lu, Kaisen Pan, Ge Yan, Jiaming Shan, Wenjie Wu, and Junchi Yan. Qas-bench:
428

rethinking quantum architecture search and a benchmark. In International Conference on
429

Machine Learning, pages 22880–22898. PMLR, 2023.
430

[31] Michael Lubasch, Jaewoo Joo, Pierre Moinier, Martin Kiffner, and Dieter Jaksch. Variational
431

quantum algorithms for nonlinear problems. Physical Review A, 101(1):010301, 2020.
432

[32] Rui Mao, Guojing Tian, and Xiaoming Sun. Towards optimal circuit size for sparse quantum
433

state preparation. arXiv e-prints, pages arXiv–2404, 2024.
434

[33] Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, and Keisuke Fujii. Quantum circuit
435

learning. Physical Review A, 98(3):032309, 2018.
436

[34] Mikko Möttönen, JJ Vartiainen, Ville Bergholm, and Martti M Salomaa. Transformation of
437

quantum states using uniformly controlled rotations. Quantum Information and Computation, 5,
438

2005.
439

[35] Kouhei Nakaji, Shumpei Uno, Yohichi Suzuki, Rudy Raymond, Tamiya Onodera, Tomoki
440

Tanaka, Hiroyuki Tezuka, Naoki Mitsuda, and Naoki Yamamoto. Approximate amplitude
441

encoding in shallow parameterized quantum circuits and its application to financial market
442

indicators. Physical Review Research, 4(2):023136, 2022.
443

11


---Page Break---
[36] Michael A Nielsen and Isaac L Chuang. Quantum computation and quantum information. 2010.
444

[37] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
445

Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
446

style, high-performance deep learning library. Advances in neural information processing
447

systems, 32, 2019.
448

[38] Yash J. Patel, Akash Kundu, Mateusz Ostaszewski, Xavier Bonet-Monroig, Vedran Dunjko,
449

and Onur Danaci. Curriculum reinforcement learning for quantum architecture search under
450

hardware errors. In The Twelfth International Conference on Learning Representations, 2024.
451

[39] Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J
452

Love, Alán Aspuru-Guzik, and Jeremy L O’brien. A variational eigenvalue solver on a photonic
453

quantum processor. Nature communications, 5(1):4213, 2014. https://doi.org/10.1038/
454

ncomms5213.
455

[40] Henning Petzka, Michael Kamp, Linara Adilova, Cristian Sminchisescu, and Mario Boley.
456

Relative flatness and generalization. Advances in neural information processing systems,
457

34:18420–18432, 2021.
458

[41] Martin Plesch and ˇCaslav Brukner. Quantum-state preparation with universal gate decomposi-
459

tions. Physical Review A, 83(3):032302, 2011.
460

[42] John Preskill. Quantum computing in the NISQ era and beyond. Quantum, 2:79, 2018.
461

[43] Qiskit contributors. Qiskit: An open-source framework for quantum computing, 2023.
462

[44] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
463

for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.
464

[45] Maria Schuld, Ilya Sinayskiy, and Francesco Petruccione. Prediction by linear regression on a
465

quantum computer. Physical Review A, 94(2):022342, 2016.
466

[46] Vivek V Shende, Stephen S Bullock, and Igor L Markov. Synthesis of quantum logic circuits. In
467

Proceedings of the 2005 Asia and South Pacific Design Automation Conference, pages 272–275,
468

2005.
469

[47] Peter W Shor. Polynomial-time algorithms for prime factorization and discrete logarithms
470

on a quantum computer. SIAM review, 41(2):303–332, 1999. https://doi.org/10.1137/
471

S0036144598347011.
472

[48] Siddarth Srinivasan, Carlton Downey, and Byron Boots. Learning and inference in hilbert space
473

with quantum graphical models. Advances in Neural Information Processing Systems, 31, 2018.
474

[49] Xiaoming Sun, Guojing Tian, Shuai Yang, Pei Yuan, and Shengyu Zhang. Asymptotically
475

optimal circuit depth for quantum state preparation and general unitary synthesis.
IEEE
476

Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2023.
477

[50] Jinkai Tian, Xiaoyu Sun, Yuxuan Du, Shanshan Zhao, Qing Liu, Kaining Zhang, Wei Yi, Wan-
478

rong Huang, Chaoyue Wang, Xingyao Wu, et al. Recent advances for quantum neural networks
479

in generative learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
480

[51] Almudena Carrera Vazquez, Ralf Hiptmair, and Stefan Woerner. Enhancing the quantum linear
481

systems algorithm using richardson extrapolation. ACM Transactions on Quantum Computing,
482

3(1):1–37, 2022.
483

[52] Hanrui Wang, Yilian Liu, Pengyu Liu, Jiaqi Gu, Zirui Li, Zhiding Liang, Jinglei Cheng,
484

Yongshan Ding, Xuehai Qian, Yiyu Shi, et al. Robuststate: Boosting fidelity of quantum state
485

preparation via noise-aware variational training. arXiv preprint arXiv:2311.16035, 2023.
486

[53] David Wierichs, Josh Izaac, Cody Wang, and Cedric Yen-Yu Lin. General parameter-shift rules
487

for quantum gradients. Quantum, 6:677, 2022.
488

12


---Page Break---
[54] Wenjie Wu, Ge Yan, Xudong Lu, Kaisen Pan, and Junchi Yan. Quantumdarts: differentiable
489

quantum architecture search for variational quantum algorithms. In International Conference
490

on Machine Learning, pages 37745–37764. PMLR, 2023.
491

[55] Ting Zhang, Jinzhao Sun, Xiao-Xu Fang, Xiao-Ming Zhang, Xiao Yuan, and He Lu. Ex-
492

perimental quantum state measurement with classical shadows.
Physical Review Letters,
493

127(20):200501, 2021.
494

[56] Xiao-Ming Zhang, Man-Hong Yung, and Xiao Yuan. Low-depth quantum state preparation.
495

Physical Review Research, 3(4):043200, 2021.
496

[57] Jian Zhao, Yu-Chun Wu, Guang-Can Guo, and Guo-Ping Guo. State preparation based on
497

quantum phase estimation. arXiv preprint arXiv:1912.05335, 2019.
498

[58] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A
499

10 million image database for scene recognition. IEEE transactions on pattern analysis and
500

machine intelligence, 40(6):1452–1464, 2017.
501

[59] Christa Zoufal, Aurélien Lucchi, and Stefan Woerner. Quantum generative adversarial networks
502

for learning and loading random distributions. npj Quantum Information, 5(1):103, 2019.
503

13


---Page Break---
The structure of our Appendix is as follows. Appendix A provides more details of implementing
504

SuperEncoder. Appendix B provides additional numerical results to illustrate the impact of state
505

sizes, model architectures, and training datasets. Appendix C analyzes the estimated runtime of
506

training SuperEncoder on real devices.
507

A
Implementation Details
508

In this section, we elaborate the missing details of SuperEncoder in the main text.
509

The overarching workflow of SuperEncoder is illustrated in Fig. 10. The target quantum states are
510

input to the MLP model. Then, the MLP model generates predicted parameters based on the target
511

states. Afterwards, the parameters are applied to the PQC to obtain the prepared quantum states.
512

Finally, we calculate the loss based on the prepared states and target states and optimize the weights
513

of MLP through backpropagation.
514

Target State

MLP

Circuit Parameters

PQC

Prepared State

Loss

Figure 10: Detailed workflow of SuperEncoder.

The settings of MLP and PQC are as follows.
515

MLP. As listed in Table 5, we implement a two-layer MLP. Each layer consists of 512 neurons. We
516

employ Tanh as the activation functions since θ represents the angles of rotation gates, ranging from
517

−π to π.
518

Linear
Input
(batch_size, 2n)
Output
(batch_size, 512)

Tanh
Input
(batch_size, 512)
Output
(batch_size, 512)

Linear
Input
(batch_size, 512)
Output
(batch_size, dim(θ))

Tanh
Input
(batch_size, dim(θ))
Output
(batch_size, dim(θ))
Table 5: MLP based SuperEncoder. n refers to the number of qubits. θ denotes the parameter vector.

PQC. The circuit structure is the same with the one depicted in Fig. 2, except that the number of
519

blocks is determined dynamically through empirical examinations. Specifically, we utilize AAE to
520

approximate a target state while increasing the number of blocks. The number of blocks is designated
521

when the resulting state fidelity no longer increases. For example, Fig. 11 demonstrates how fidelity
522

changes while increasing the number of blocks. As one can observe, the fidelity converges when the
523

number of layers is larger than 8. Hence, the number of layers is set to be 8 for 4-qubit quantum
524

states. We follow the same procedure to set the number of blocks for other state sizes. Each block
525

has the same structure, consisting of a rotation layer and an entangler layer. Given an n-qubit system,
526

a rotation layer comprises n Ry gates, each operating on a distinct qubit. The entangler layer is
527

composed of two CNOT layers. The first CNOT layer applies CNOT gates to {(q0, q1), (q2, q3), . . . },
528

and the second CNOT layer applies CNOT gates to {(q1, q2), (q3, q4), . . . }. Hence, the depth of
529

14


---Page Break---
a block is 3. Let l be the number of blocks; then the dimension of the parameter vector is given
530

by dim(θ) = n × l, and the depth of AAE/SuperEncoder is 3 × l. We conclude the settings of
531

AAE/SuperEncoder used throughout this study in Table 6.
532

1
2
3
4
5
6
7
8
9 10 11 12 13 14 15 16 17 18 19 20
Number of Blocks

0.2

0.4

0.6

0.8

1.0

Fidelity

Figure 11: Fidelity vs. # blocks for 4-qubit states using AAE.

Number of Qubits
4
6
8
Number of Blocks
8
20
40
Depth
24
60
120
Table 6: Number of blocks and corresponding depth of AAE/SuperEncoder.

The hyperparameters for training SuperEncoder and optimizing AAE are as follows.
533

Training Hyperparameters for SuperEncoder. Throughout our experiments, the number of epochs
534

are consistently set to be 10. For 4-qubit states, we set bath_size to 32, while we set it 64 for
535

6-qubit and 8-qubit states. We adopt Adam optimizer [22] with a learning rate of 3e-3 and a weight
536

decay of 1e-5.
537

Hyperparameters for AAE. To optimize the parameters of AAE, we also use the Adam optimizer,
538

with a learning rate of 1e-2 and zero weight decay. For all quantum states, we train the AAE for 100
539

steps.
540

B
More Numerical Results
541

B.1
Results on Larger Quantum States
542

In line with the main text, we train the SuperEncoder for 6-qubit and 8-qubit quantum states using
543

FractalDB-60 as the training dataset. Then we evaluate the performance of SuperEncoder on the
544

synthetic test datasets. As shown in Table 7, the average fidelity on 6-qubit and 8-qubit states are
545

0.8655 and 0.7624 respectively. In Appendix B.2, B.3, we discuss potential optimizations to alleviate
546

this performance degradation.
547

Dataset
n = 4
n = 6
n = 8
Uniform
0.9731
0.9254
0.8648
Normal
0.8201
0.7457
0.6075
Log-normal
0.9421
0.8575
0.7122
Exponential
0.9464
0.8757
0.7613
Dirichlet
0.9737
0.9232
0.8663
Avg
0.9310
0.8655
0.7624
Avg-AAE
0.9994
0.9964
0.9910
Table 7: Performance evaluation on larger quantum states (6-qubit and 8-qubit). The last separate
row shows the results of AAE for comparison.

15


---Page Break---
B.2
Impact of Model Architecture
548

As a preliminary investigation, the optimal model architecture for SuperEncoder still requires further
549

exploration. Currently, we have set the size of the hidden units at a constant 512 (Table 5). However,
550

as the number of qubits, n, increases, a wider network architecture may become necessary. To
551

showcase the impact of model width, we adjust the size to 4 × 2n for 6-qubit states and 16 × 2n for
552

8-qubit states, and compare their performance with the original settings, as shown in Table 8. As
553

evident from the results, this simple adjustment significantly enhances the fidelity of SuperEncoder,
554

suggesting that there is substantial potential to boost SuperEncoder’s performance by developing a
555

more tailored network architecture.
556

n = 6
n = 8
Dataset
h = 512
h = 4 × 26
h = 512
h = 16×28

Uniform
0.9254
0.9267
0.8648
0.8821
Normal
0.7457
0.7580
0.6075
0.6401
Log-normal
0.8575
0.8608
0.7122
0.7294
Exponential
0.8757
0.8732
0.7613
0.7781
Dirichlet
0.9232
0.9261
0.8663
0.8805
Avg
0.8655
0.8690
0.7624
0.7820
Table 8: Impact of increasing network width. Here h refers to the size of hidden units.

B.3
Impact of Training Datasets
557

In addition to refining the model architecture, the development of a specially designed dataset for
558

pre-training SuperEncoder is essential. Currently, the dataset utilized is FractalDB [21], which is
559

originally designed for computer vision tasks. However, given the wide range of applications of QSP,
560

there is a need to accommodate diverse types of classical data from various domains. Therefore, how
561

to create a comprehensive dataset that could fully unleash the potential of SuperEncoder remains an
562

open question. While developing a pre-trained model that performs well in all kinds of applications
563

may be challenging, we advocate for a strategy that combines pre-training with fine-tuning for the
564

practical deployment of SuperEncoder, similar to the approach used with foundation models in
565

classical machine learning. To substantiate this approach, we have compiled a separate dataset that
566

encompasses a variety of statistical distributions not limited to those utilized for evaluation (but with
567

different settings). As demonstrated in Table 9, after fine-tuning, the performance of SuperEncoder
568

improves by approximately 0.03.
569

Dataset
Pre-training
Pre-training+Finetuning
Uniform
0.9731
0.9909
Normal
0.8201
0.8879
Log-normal
0.9421
0.9717
Exponential
0.9464
0.9729
Dirichlet
0.9737
0.9903
Avg
0.9310
0.9627
Table 9: Fidelity improvements after fine-tuning SuperEncoder using a dataset consisting of different
distributions.

C
Runtime Estimation for Training on Real Devices
570

Although we have theoretically analyzed the feasibility of training SuperEncoder using states from
571

real devices (Section 3.2), its practical implementation poses significant challenges. Specifically,
572

state-of-the-art quantum tomography techniques, such as classical shadow [55, 16], require numerous
573

snapshots, each measuring a distinct observable.
574

To train SuperEncoder, each sample in the training dataset necessitates one classical shadow to obtain
575

the prepared state. For instance, with the FractalDB-60 dataset, one training epoch requires 60,000
576

classical shadows. Our experiments on the IBM cloud platform reveal an average runtime of 3.02
577

16


---Page Break---
seconds per circuit job excluding queuing time. Suppose the number of snapshots is 1000, then the
578

total runtime to train SuperEncoder for 10 epochs is about 1,812,000,000 seconds3, roughly 57 years,
579

making the process prohibitively expensive and time-consuming.
580

However, quantum tomography is under active investigation, and we expect more efficient techniques
581

to emerge for acquiring noisy quantum states from real devices. Additionally, with the advancement
582

of quantum computing system, future systems may have tightly integrated quantum-classical hetero-
583

geneous architectures (shorter runtime per job) while being capable of executing numerous quantum
584

circuits in parallel (jobs within a classical shadow can execute in parallel). Hence, we anticipate the
585

training of SuperEncoder to be feasible in the future.
586

310 × 1000 × 60000 × 3.02

17


---Page Break---
NeurIPS Paper Checklist
587

1. Claims
588

Question: Do the main claims made in the abstract and introduction accurately reflect the
589

paper’s contributions and scope?
590

Answer: [Yes]
591

Justification: This work aims at training-free approximate quantum state preparation. As
592

claimed in the abstract and introduction.
593

Guidelines:
594

• The answer NA means that the abstract and introduction do not include the claims
595

made in the paper.
596

• The abstract and/or introduction should clearly state the claims made, including the
597

contributions made in the paper and important assumptions and limitations. A No or
598

NA answer to this question will not be perceived well by the reviewers.
599

• The claims made should match theoretical and experimental results, and reflect how
600

much the results can be expected to generalize to other settings.
601

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
602

are not attained by the paper.
603

2. Limitations
604

Question: Does the paper discuss the limitations of the work performed by the authors?
605

Answer: [Yes]
606

Justification: SuperEncoder sacrifices fidelity, as discussed in Section 4.4.
607

Guidelines:
608

• The answer NA means that the paper has no limitation while the answer No means that
609

the paper has limitations, but those are not discussed in the paper.
610

• The authors are encouraged to create a separate "Limitations" section in their paper.
611

• The paper should point out any strong assumptions and how robust the results are to
612

violations of these assumptions (e.g., independence assumptions, noiseless settings,
613

model well-specification, asymptotic approximations only holding locally). The authors
614

should reflect on how these assumptions might be violated in practice and what the
615

implications would be.
616

• The authors should reflect on the scope of the claims made, e.g., if the approach was
617

only tested on a few datasets or with a few runs. In general, empirical results often
618

depend on implicit assumptions, which should be articulated.
619

• The authors should reflect on the factors that influence the performance of the approach.
620

For example, a facial recognition algorithm may perform poorly when image resolution
621

is low or images are taken in low lighting. Or a speech-to-text system might not be
622

used reliably to provide closed captions for online lectures because it fails to handle
623

technical jargon.
624

• The authors should discuss the computational efficiency of the proposed algorithms
625

and how they scale with dataset size.
626

• If applicable, the authors should discuss possible limitations of their approach to
627

address problems of privacy and fairness.
628

• While the authors might fear that complete honesty about limitations might be used by
629

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
630

limitations that aren’t acknowledged in the paper. The authors should use their best
631

judgment and recognize that individual actions in favor of transparency play an impor-
632

tant role in developing norms that preserve the integrity of the community. Reviewers
633

will be specifically instructed to not penalize honesty concerning limitations.
634

3. Theory Assumptions and Proofs
635

Question: For each theoretical result, does the paper provide the full set of assumptions and
636

a complete (and correct) proof?
637

Answer: [Yes]
638

18


---Page Break---
Justification: All these necessary contents for theoretical results are included in Section 3.2.
639

Guidelines:
640

• The answer NA means that the paper does not include theoretical results.
641

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
642

referenced.
643

• All assumptions should be clearly stated or referenced in the statement of any theorems.
644

• The proofs can either appear in the main paper or the supplemental material, but if
645

they appear in the supplemental material, the authors are encouraged to provide a short
646

proof sketch to provide intuition.
647

• Inversely, any informal proof provided in the core of the paper should be complemented
648

by formal proofs provided in appendix or supplemental material.
649

• Theorems and Lemmas that the proof relies upon should be properly referenced.
650

4. Experimental Result Reproducibility
651

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
652

perimental results of the paper to the extent that it affects the main claims and/or conclusions
653

of the paper (regardless of whether the code and data are provided or not)?
654

Answer: [Yes]
655

Justification: Our code is open-source with instructions to reproduce our results, as described
656

in Section 4.1. We also describe the details of experiment settings in Appendix A.
657

Guidelines:
658

• The answer NA means that the paper does not include experiments.
659

• If the paper includes experiments, a No answer to this question will not be perceived
660

well by the reviewers: Making the paper reproducible is important, regardless of
661

whether the code and data are provided or not.
662

• If the contribution is a dataset and/or model, the authors should describe the steps taken
663

to make their results reproducible or verifiable.
664

• Depending on the contribution, reproducibility can be accomplished in various ways.
665

For example, if the contribution is a novel architecture, describing the architecture fully
666

might suffice, or if the contribution is a specific model and empirical evaluation, it may
667

be necessary to either make it possible for others to replicate the model with the same
668

dataset, or provide access to the model. In general. releasing code and data is often
669

one good way to accomplish this, but reproducibility can also be provided via detailed
670

instructions for how to replicate the results, access to a hosted model (e.g., in the case
671

of a large language model), releasing of a model checkpoint, or other means that are
672

appropriate to the research performed.
673

• While NeurIPS does not require releasing code, the conference does require all submis-
674

sions to provide some reasonable avenue for reproducibility, which may depend on the
675

nature of the contribution. For example
676

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
677

to reproduce that algorithm.
678

(b) If the contribution is primarily a new model architecture, the paper should describe
679

the architecture clearly and fully.
680

(c) If the contribution is a new model (e.g., a large language model), then there should
681

either be a way to access this model for reproducing the results or a way to reproduce
682

the model (e.g., with an open-source dataset or instructions for how to construct
683

the dataset).
684

(d) We recognize that reproducibility may be tricky in some cases, in which case
685

authors are welcome to describe the particular way they provide for reproducibility.
686

In the case of closed-source models, it may be that access to the model is limited in
687

some way (e.g., to registered users), but it should be possible for other researchers
688

to have some path to reproducing or verifying the results.
689

5. Open access to data and code
690

Question: Does the paper provide open access to the data and code, with sufficient instruc-
691

tions to faithfully reproduce the main experimental results, as described in supplemental
692

material?
693

19


---Page Break---
Answer: [Yes]
694

Justification: See Section 4.1.
695

Guidelines:
696

• The answer NA means that paper does not include experiments requiring code.
697

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
698

public/guides/CodeSubmissionPolicy) for more details.
699

• While we encourage the release of code and data, we understand that this might not be
700

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
701

including code, unless this is central to the contribution (e.g., for a new open-source
702

benchmark).
703

• The instructions should contain the exact command and environment needed to run to
704

reproduce the results. See the NeurIPS code and data submission guidelines (https:
705

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
706

• The authors should provide instructions on data access and preparation, including how
707

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
708

• The authors should provide scripts to reproduce all experimental results for the new
709

proposed method and baselines. If only a subset of experiments are reproducible, they
710

should state which ones are omitted from the script and why.
711

• At submission time, to preserve anonymity, the authors should release anonymized
712

versions (if applicable).
713

• Providing as much information as possible in supplemental material (appended to the
714

paper) is recommended, but including URLs to data and code is permitted.
715

6. Experimental Setting/Details
716

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
717

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
718

results?
719

Answer: [Yes]
720

Justification: We illustrate the experimental settings in Section 4.1, and provides additional
721

details in Appendix A.
722

Guidelines:
723

• The answer NA means that the paper does not include experiments.
724

• The experimental setting should be presented in the core of the paper to a level of detail
725

that is necessary to appreciate the results and make sense of them.
726

• The full details can be provided either with the code, in appendix, or as supplemental
727

material.
728

7. Experiment Statistical Significance
729

Question: Does the paper report error bars suitably and correctly defined or other appropriate
730

information about the statistical significance of the experiments?
731

Answer: [No]
732

Justification: Throughout our experiments, we set the random seed to be fixed for all libraries
733

we used.
734

Guidelines:
735

• The answer NA means that the paper does not include experiments.
736

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
737

dence intervals, or statistical significance tests, at least for the experiments that support
738

the main claims of the paper.
739

• The factors of variability that the error bars are capturing should be clearly stated (for
740

example, train/test split, initialization, random drawing of some parameter, or overall
741

run with given experimental conditions).
742

• The method for calculating the error bars should be explained (closed form formula,
743

call to a library function, bootstrap, etc.)
744

• The assumptions made should be given (e.g., Normally distributed errors).
745

20


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error
746

of the mean.
747

• It is OK to report 1-sigma error bars, but one should state it. The authors should
748

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
749

of Normality of errors is not verified.
750

• For asymmetric distributions, the authors should be careful not to show in tables or
751

figures symmetric error bars that would yield results that are out of range (e.g. negative
752

error rates).
753

• If error bars are reported in tables or plots, The authors should explain in the text how
754

they were calculated and reference the corresponding figures or tables in the text.
755

8. Experiments Compute Resources
756

Question: For each experiment, does the paper provide sufficient information on the com-
757

puter resources (type of compute workers, memory, time of execution) needed to reproduce
758

the experiments?
759

Answer: [Yes]
760

Justification: We describe the computer resources used in this paper in Section 4.1.
761

Guidelines:
762

• The answer NA means that the paper does not include experiments.
763

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
764

or cloud provider, including relevant memory and storage.
765

• The paper should provide the amount of compute required for each of the individual
766

experimental runs as well as estimate the total compute.
767

• The paper should disclose whether the full research project required more compute
768

than the experiments reported in the paper (e.g., preliminary or failed experiments that
769

didn’t make it into the paper).
770

9. Code Of Ethics
771

Question: Does the research conducted in the paper conform, in every respect, with the
772

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
773

Answer: [Yes]
774

Justification: We have read the code of ethics and followed its requirements.
775

Guidelines:
776

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
777

• If the authors answer No, they should explain the special circumstances that require a
778

deviation from the Code of Ethics.
779

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
780

eration due to laws or regulations in their jurisdiction).
781

10. Broader Impacts
782

Question: Does the paper discuss both potential positive societal impacts and negative
783

societal impacts of the work performed?
784

Answer: [NA]
785

Justification: This work has no societal impact.
786

Guidelines:
787

• The answer NA means that there is no societal impact of the work performed.
788

• If the authors answer NA or No, they should explain why their work has no societal
789

impact or why the paper does not address societal impact.
790

• Examples of negative societal impacts include potential malicious or unintended uses
791

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
792

(e.g., deployment of technologies that could make decisions that unfairly impact specific
793

groups), privacy considerations, and security considerations.
794

21


---Page Break---
• The conference expects that many papers will be foundational research and not tied
795

to particular applications, let alone deployments. However, if there is a direct path to
796

any negative applications, the authors should point it out. For example, it is legitimate
797

to point out that an improvement in the quality of generative models could be used to
798

generate deepfakes for disinformation. On the other hand, it is not needed to point out
799

that a generic algorithm for optimizing neural networks could enable people to train
800

models that generate Deepfakes faster.
801

• The authors should consider possible harms that could arise when the technology is
802

being used as intended and functioning correctly, harms that could arise when the
803

technology is being used as intended but gives incorrect results, and harms following
804

from (intentional or unintentional) misuse of the technology.
805

• If there are negative societal impacts, the authors could also discuss possible mitigation
806

strategies (e.g., gated release of models, providing defenses in addition to attacks,
807

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
808

feedback over time, improving the efficiency and accessibility of ML).
809

11. Safeguards
810

Question: Does the paper describe safeguards that have been put in place for responsible
811

release of data or models that have a high risk for misuse (e.g., pretrained language models,
812

image generators, or scraped datasets)?
813

Answer: [NA]
814

Justification: This paper poses no such risks as our released model and datasets are only
815

able to be used for quantum state preparation.
816

Guidelines:
817

• The answer NA means that the paper poses no such risks.
818

• Released models that have a high risk for misuse or dual-use should be released with
819

necessary safeguards to allow for controlled use of the model, for example by requiring
820

that users adhere to usage guidelines or restrictions to access the model or implementing
821

safety filters.
822

• Datasets that have been scraped from the Internet could pose safety risks. The authors
823

should describe how they avoided releasing unsafe images.
824

• We recognize that providing effective safeguards is challenging, and many papers do
825

not require this, but we encourage authors to take this into account and make a best
826

faith effort.
827

12. Licenses for existing assets
828

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
829

the paper, properly credited and are the license and terms of use explicitly mentioned and
830

properly respected?
831

Answer: [Yes]
832

Justification: We use an open-source dataset FractalDB, we cite the original paper and
833

indicates the version we use in Section 4.1.
834

Guidelines:
835

• The answer NA means that the paper does not use existing assets.
836

• The authors should cite the original paper that produced the code package or dataset.
837

• The authors should state which version of the asset is used and, if possible, include a
838

URL.
839

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
840

• For scraped data from a particular source (e.g., website), the copyright and terms of
841

service of that source should be provided.
842

• If assets are released, the license, copyright information, and terms of use in the
843

package should be provided. For popular datasets, paperswithcode.com/datasets
844

has curated licenses for some datasets. Their licensing guide can help determine the
845

license of a dataset.
846

22


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
847

the derived asset (if it has changed) should be provided.
848

• If this information is not available online, the authors are encouraged to reach out to
849

the asset’s creators.
850

13. New Assets
851

Question: Are new assets introduced in the paper well documented and is the documentation
852

provided alongside the assets?
853

Answer: [Yes]
854

Justification: We submit our assets in zip file and also put them on the anonymous github
855

repository, we have included a README file with detailed descriptions.
856

Guidelines:
857

• The answer NA means that the paper does not release new assets.
858

• Researchers should communicate the details of the dataset/code/model as part of their
859

submissions via structured templates. This includes details about training, license,
860

limitations, etc.
861

• The paper should discuss whether and how consent was obtained from people whose
862

asset is used.
863

• At submission time, remember to anonymize your assets (if applicable). You can either
864

create an anonymized URL or include an anonymized zip file.
865

14. Crowdsourcing and Research with Human Subjects
866

Question: For crowdsourcing experiments and research with human subjects, does the paper
867

include the full text of instructions given to participants and screenshots, if applicable, as
868

well as details about compensation (if any)?
869

Answer: [NA]
870

Justification: This paper does not involve crowdsourcing nor research with human subjects.
871

Guidelines:
872

• The answer NA means that the paper does not involve crowdsourcing nor research with
873

human subjects.
874

• Including this information in the supplemental material is fine, but if the main contribu-
875

tion of the paper involves human subjects, then as much detail as possible should be
876

included in the main paper.
877

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
878

or other labor should be paid at least the minimum wage in the country of the data
879

collector.
880

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
881

Subjects
882

Question: Does the paper describe potential risks incurred by study participants, whether
883

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
884

approvals (or an equivalent approval/review based on the requirements of your country or
885

institution) were obtained?
886

Answer: [NA]
887

Justification: This paper does not involve crowdsourcing nor research with human subjects.
888

Guidelines:
889

• The answer NA means that the paper does not involve crowdsourcing nor research with
890

human subjects.
891

• Depending on the country in which research is conducted, IRB approval (or equivalent)
892

may be required for any human subjects research. If you obtained IRB approval, you
893

should clearly state this in the paper.
894

• We recognize that the procedures for this may vary significantly between institutions
895

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
896

guidelines for their institution.
897

• For initial submissions, do not include any information that would break anonymity (if
898

applicable), such as the institution conducting the review.
899

23


---Page Break---
