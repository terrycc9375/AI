Reasoning Multi-Agent Behavioral Topology for
Interactive Autonomous Driving

Haochen Liu1,2
Li Chen2,3
Yu Qiao2
Chen Lv1†
Hongyang Li2,3†

1 Nanyang Technological University
2 Shanghai AI Lab
3 University of Hong Kong

Abstract

Autonomous driving system aims for safe and social-consistent driving through
the behavioral integration among interactive agents. However, challenges re-
main due to multi-agent scene uncertainty and heterogeneous interaction. Current
dense and sparse behavioral representations struggle with inefficiency and incon-
sistency in multi-agent modeling, leading to instability of collective behavioral
patterns when integrating prediction and planning (IPP). To address this, we ini-
tiate a topological formation that serves as a compliant behavioral foreground to
guide downstream trajectory generations. Specifically, we introduce Behavioral
Topology (BeTop), a pivotal topological formulation that explicitly represents
the consensual behavioral pattern among multi-agent future. BeTop is derived
from braid theory to distill compliant interactive topology from multi-agent future
trajectories. A synergistic learning framework (BeTopNet) supervised by BeTop
facilitates the consistency of behavior prediction and planning within the predicted
topology priors. Through imitative contingency learning, BeTop also effectively
manages behavioral uncertainty for prediction and planning. Extensive verification
on large-scale real-world datasets, including nuPlan and WOMD, demonstrates
that BeTop achieves state-of-the-art performance in both prediction and planning
tasks. Further validations on the proposed interactive scenario benchmark show-
case planning compliance in interactive cases. Code and model is available at
https://github.com/OpenDriveLab/BeTop.

1
Introduction

Autonomous driving system aspires to safe, humanoid, and socially compatible maneuvers [1]. This
drives for formulation, prediction, and negotiation of collective future behaviors among interactive
agents and autonomous vehicles (AVs) [2]. Remarkable accuracy is achieved by learning-based
paradigms [3], including end-to-end modular design [4–7], social modeling [8, 9], and trajectory-
level integration [10–13]. However, substantial challenges arise in real-world cases due to scene
uncertainty and volatile interactive patterns for multi-agent future behaviors.

To embrace compliant patterns for multi-agent future behaviors, current formulations fall into two
mainstreams, dense representation and sparse representation (Fig. 1). Dense representation quan-
tizes agent behaviors under ego-centric rasterization, forecasting bird’s eye view (BEV) occupancy
probabilities [14, 7, 15] or temporal flow [16–18]. It is easy to deduce interactions, perform scalable
behaviors for agents [19], and align with BEV perceptions [20]. Still, dense representation is hindered
by frozen receptions. It causes safety-vulnerable intractability and occlusions potentially interacting
with ego maneuvers [16, 21]. Contrary to pixel-wise behavioral probability, sparse representation
forecasts agent-anchored set of trajectories [22–25] or intention distributions [10, 26, 27]. Its multi-
modal formulation for each agent marks the elasticity in diverse behavioral uncertainty and tractability
under flexile spatial semantics. However, behavioral misalignment [28] and modality collapse [24]

Work done while Haochen’s internship at Shanghai AI Lab. † Equal co-advising.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Driving scenario

a)

Dense
Sparse
BeTop (Ours)

b)
c)
d)

…..

BeTopNet

Autonomous vehicle
Scene agents
Braid crossings
Planning &
Prediction
Collision &
Overlap

Topo Braids
Reason

Figure 1: Multi-agent Behavioral Formulation. (a) A typical driving scenario in Arizona, US [35];
(b) Dense representation conducts scalable occupancy prediction jointly, but restrained reception
leads to unbounded collisions with planning; (c) Sparse supervision derives multi-agent trajectories
with multi-modalities, while it struggles with conflicts among integrated prediction and planning;
(d) BeTop reasons future topological behaviors for all scene-agents through braids theory, funneling
interactive eventual agents (in highlighted colors) and guiding compliant joint prediction and planning.

impede compliant multi-agent modeling, requiring exponential computations with growth agent
numbers [29]. Issues particularly result in unstable and slow behavioral learning when exposed
to predictions and planning (IPP) [30]. Typical solutions by conditional prediction [31, 28, 32] or
game-theoretic reasoning [8, 33] often lead to nonstrategic maneuvers [34] due to non-compliant
rollouts in adjusting interactive behaviors. This calls for a re-formulation for multi-agent behaviors,
which should stabilize collective behavioral patterns in a compliant manner for IPP objectives.

The decision-making process for human drivers provides valuable insights. Humans primarily
determine the future behavior of interacted agents for decision-making without relying on their specific
states [2, 36]. Thus, an effective strategy involves assessing agent-wise behavioral impact on planning
maneuvers, and reasoning about compliant interactions. Our fundamental insight is that compliant
multi-agent behaviors exhibit topological formations, which can be identified by distilling consensual
interactions from future behaviors. Prior works have approached this challenge through structural
design [37, 38] or implicit relational learning [39–41] using GNNs [42] or Transformers [43]. Other
studies quantify uncertainty by topological properties [44, 45]. Nevertheless, current literature is
scarce in formulating explicit future supervision of compliant multi-agent behavioral patterns.

To this end, we launch the multi-agent behavior formulation termed as Behavioral Topology (BeTop).
At its core, BeTop explicitly forms the topological supervision of consensual multi-agent future
interactions, and reasons to guide prediction and planning. BeTop stems from braid theory [46],
which infers compliant interactions of multiple paths from the intertwining of their braids. This
empowers BeTop to intuitively distill forward intertwines (occupancy) as joint topology from braided
multi-agent future trajectories (Fig. 1), marrying dense and sparse representation. With the aid of
BeTop, we introduce a synergistic Transformer-based learning stack, BeTopNet, for learning IPP
objectives. To implement, an iterative decoding strategy simultaneously reasons about Behavioral
Topology and generates trajectory sets. Then the topology-guided local attention, embedded in
each decoder layer, selectively queries behavioral semantics from social-compliant agents within the
predicted BeTop priors. To further alleviate multi-agent uncertainty through topological guidance, a
contingency planning paradigm is fitfully deployed. We lay out the imitative contingency learning
process, which regulates the safety-ensured short-term plan. It maintains the long-range uncertainty
by reasoned joint predictions from BeTop. Experimental results exhibit enhanced consistency and
accuracy for prediction and planning in real-world scenarios. Testing in proposed interactive cases
further highlights the planning ability of BeTopNet. To sum up, our contributions are three-fold:

• We bring in the concept of Behavioral Topology, a multi-agent behavioral formulation for topologi-
cal reasoning that explicitly supervises consensual future interactions jointly for the IPP system.

• A synergistic learning framework BeTopNet, offering joint planning and prediction guided by
topology reasoning, is devised. Topology-guided local attention and imitative contingency planning
could resolve scene compliance and multi-agent uncertainty.

• Benchmarking on nuPlan [47] and WOMD [35], our approach demonstrates strong performance
in both planning strategy and prediction accuracy. BeTopNet witnesses evident improvement

2


---Page Break---
over previous counterparts, e.g., +7.9% in general planning score, +3.8% under interactive cases,
+4.1% mAP for joint prediction, and +2.3% mAP for marginal prediction.

2
Related work

Multi-agent behavioral modeling. Carving the collective future behavior of diverse agents is
imperative for socially-consistent driving maneuver. Earlier approaches centered around occupancy
prediction [14, 5, 4, 48]. Forecasting the spatial presence under dense BEV representation [49, 20]
offers flexibility of arbitrary agents [17, 50, 19] and alignment with perception [7, 51]. However,
rigidity in resolution induces scenario occlusion [16], rendering intractable occupancy [21]. In
parallel, sparse representation consolidates multi-agent behavior into cohesive modalities across future
trajectories [52, 22, 53, 54] or intentions [27, 10, 26, 55]. Joint future behaviors are derived through
goal-based sampling or recombination from marginal predictions [56–58]. However, the collection
of joint modalities is susceptible to mode collapse and entails exponential complexity [29, 59].
Meanwhile, topological representation has garnered traction as motion primitives [60, 61] or tools [44,
45] for scenario quantification, yet topological properties delineating collective future behaviors
remain largely unexplored. Our BeTop targets the issue, marrying dense behavior probabilities by
topology with the sparse motion from joint predictions to present structured future behaviors.

On the other hand, inconsistent communal agent behaviors have motivated leveraging future interac-
tions. Implicit approaches obtain tacit interactions with attention [62, 32, 63] or GNN [24, 64, 37, 65]
from final motion regressions. Nonetheless, the implicit supervisions are found inefficient in dynamic
scenarios [22]. Contrarily, explicitly reasoning mutual behaviors by conditional factorization [31, 66],
relation reasoning [41, 40], or entropy-based methods [67, 68] offer consistent behavioral priors.
However, hefty variance across agent dynamics and scenario geometries yield unstable inference.
Distinguished from them, BeTop crafts a compact topological supervision that stabilize future inter-
actions among multi-agent behaviors. Derived from topological braids, BeTop offers a topological
equivalent behavioral representation to guide compliant forecasting and planning.

Integrated prediction and planning. IPP system aims to harmonize trajectory-based learning of
future interactive behaviors between the ego vehicle and social agents. Rule-based approaches [69–71]
integrate handcrafted future interactions to evaluate candidate planning profiles, offering remarkable
outcomes in rule-powered reactive simulation [47]. Still, the absence of real-world behaviors
exhibits significant gaps in interactive scenarios. Learning-based methods yield imitative planning
by integrating predictions within holistic modeling [72–74]. However, history-based coalitions pose
challenges in supervising future homology among agents. Recently, hybrid pipelines [8, 75] have
utilized post-processing and optimization upon learning-based models to realize behavior interactions
among predictions and planning. This can entail significant computational overhead, and imitative
planning tends to overestimate hereditary uncertainty in behavior predictions. Tree-based [10, 27]
and contingency-enabled [76, 11] works seek to balance planning preemption and aggression in the
face of behavior uncertainty. Nonetheless, pipelines without holistic interactions fall into passive
planning maneuvers and incur high exponential cost for predictions. In our work, BeTop provides
an explicit prior for future interactive behaviors which enhances compliant trajectory generations.
Moreover, the synergistic prediction and contingency planning networks with BeTop effectively
manage behavioral uncertainty.

3
Behavioral Topology

Presenting BeTop, we commence the Behavioral Topology formulation and task statement of IPP for
autonomous driving in Sec. 3.1. Then, we demonstrate the BeTopNet network architecture (Fig. 3)
for topological reasoning and IPP generation in Sec. 3.2. Finally, in Sec. 3.3, we propose the imitative
contingency learning process by topological guidance for the proposed network.

3.1
Formulation

Problem formulation. We consider the driving scenario with Na agents as A1:Na at presence t = 0,
along with the scenario map M. The states over historical horizon Th are denoted as X1 for AV and
as X2:Na for scenario agents, respectively, where Xn = {x−Th:0}n, n ∈[1, Na]. The objective for
integrated prediction and planning is to jointly predict scene agents’ trajectories Y2:Na as well as AV
planning Y1 over a future horizon Tf as Yn = {y1:Tf }n, n ∈[1, Na].

3


---Page Break---
Figure 2: BeTop formulation. Joint future trajec-
tories are transformed to braid sets, and then form
joint topology through intertwine indicators.

𝒕

𝒚

𝒙
𝒙

𝒚

𝒕

𝒚

𝒙

Agent

AV

𝜎1

+

𝜎1

−

𝑒1,𝑗

𝐘1

𝐴1

Table 1: Analysis on different behavioral for-
mulations. BeTop labels behave most similarly
to human annotations [35], excelling over other
formulations like k nearest GT or local attention.

Behavioral
WOMD
Formulations
Acc. ↑
AUC ↑

Expert [35]
1.000
1.000

GT top-k
0.833
0.702
Local attention [22]
0.951
0.522
JFP graph [67]
0.955
0.500

BeTop (Ours)
0.967
0.731

Topological formulation. We leverage the braid theory [46], which probes explicit formulations
for compliant multi-agent interactions from future data Y1:Na. Intuitively, it denotes a transform
process for Y1:Na with respective agent coordinates, and then gathers each future forward intertwine
(occupancy) as joint interactions. Formally, consider the braid group BNa = {σn} by Na primitive
braids σn, each of which σn = (f n
1 , · · · , f n
Na) denotes a tuple of monotonically increased functions
f : R3 × Y →R2 × I mapping from Cartesian
 
⃗x, ⃗y,⃗t

to lateral coordinate
 
⃗y,⃗t

for agent future
Y. Specifically, the function f n
i in σn is defined as f n
i →(Yi −bn)Rn; 1 ≤i, n ≤Na, where bn
and Rn denote the left-hand transform matrix to local coordinate of agent An. The joint interactive
behaviors are identified as a set of braids having intertwines {σ±
n } ⊂BNa over others [45], as
shown in Fig. 2. Opposite to implicit methods [22, 67, 41] banking on future distance heuristic, each
intertwine in the braid can signify an explicit behavioral response, distinguishing between assertive
(σ+
n , elicit yielding from others) and passive (σ−
n , yield to others) maneuvers. To avert difficulties in
dynamic braid set inference, we redraft multi-agent braids from a topology reasoning perspective.

Named by BeTop, the goal is to reason a topological graph G = (V, E) for multi-agent future
behaviors (Fig. 2). Expressly, node topology V = {Yn} is denoted by multi-agent future trajectories.
We can then reformulate the braid set {σ±
n } as an edge topology eij →E ∈RNa×Na; 1 ≤i, j ≤Na
for future interactive behaviors. Each topology element eij can be defined by two braid functions
f i
i , f i
j ∈σi assessing the future interwines along with Yi, Yj as: eij = maxt I
 
f i
i (yt
i), f i
j(yt
j)

.
Here I is an intertwine indicator by segment intersection [77] under lateral coordinates. With favorable
properties proved in Appendix B, we can formulate the reasoning task as:

G∗= (max ˆV, max ˆE).
(1)

Agent future ˆY in node term ˆV is defined by Gaussian mixtures (GMM) and optimized in Sec. 3.3.
The edge topology reasoning ˆE can be specified as a probabilistic inference problem by:

max ˆE = max
X

i

X

j
eij log ˆ
eij + (1 −eij) log(1 −ˆ
eij),
(2)

where 1 ≤i, j ≤Na. Synergistic reasoning structures are then established optimizing G∗.

Comparative analysis. To highlight BeTop’s position among various formulations, we first conduct
a preliminary analysis to assess behavioral similarity by retrieving future interactive agent pairs using
human annotations [35]. Human likeness is quantified by classification metrics, including accuracy
and the area under the curve (AUC), with annotated interactive IDs. As depicted in Table 1, labeled
BeTop achieves the closest behavioral similarity compared with other well-accepted formulations in
the community. Compared with retrieving k nearest strategy (k = 6) by ground-truth future states,
we observe advanced differentiation in non-interactive behaviors (+16.1% Acc., +4.13% AUC) by
BeTop. We then look into the generic learning-based structure by attention [22] or dynamic graph
[67] for interactive behaviors. Despite high accuracy, their inferior AUC scores imply difficulties in
retrieving precise interactivity compared with BeTop (+19.9 AUC). We refer analytical content in
Appendix B. This draft for a reasoning framework BeTop prompting joint behaviors.

3.2
BeTopNet

As presented in Fig. 3, we introduce the synergistic learning framework reasoning BeTop in response
to the series of challenges. It encompasses a Transformer backended encoder-decoder network. With

4


---Page Break---
BeTop Guided 
Transformer Decoder 

Topology 

Head

…..

× L

Planning

Head

Prediction 

Head

෍𝑷𝐶𝐽𝓣𝑱, ෡𝒀

…

෡𝒀𝑴

…

Recombine

Transformer 

Encoder

…..
…..

Scene Encoder
Synergistic Decoder & Reason Heads
Imitative Contingency Learning

…..
…..

…

Agent
Encoder

Map
Encoder

𝝉𝑴
𝓣𝑱

෠𝐘𝑛𝑙

𝐐𝐴

𝑙,𝑛

ෝ𝒚𝑛𝑙
ෝ𝒑𝑛𝑙

Ƹ𝑒𝑛𝑙

𝜖𝑛𝑙

𝐒𝑨
𝐒𝑴
𝐒𝑹

𝜖AV

max𝑀Ƹ𝑒1

argmin

෡𝐘𝟏
𝐶𝑀(𝝉𝑴, ෡𝒀)
+

෡𝒀𝑱

Figure 3: The BeTopNet Architecture. BeTop establishes an integrated network for topological
behavior reasoning, comprising three fundamentals. Scene encoder generates scene-aware attributes
for agent SA and map SM. Initialized by SR and QA, synergistic decoder reasons edge topology
ˆel
n and trajectories ˆYl
n iteratively from topology-guided local attention. Branched planning τ ∈ˆY1
with predictions and topology are optimized jointly by imitative contingency learning.

encoded scene semantics X; M, the proposed network features a synergistic decoder which reasons
and guides BeTop. Reason heads for topology ˆE and IPP for ˆV comprise the behavioral graph G.

Scene encoder. We leverage a scene-centric coordinate system following planning-oriented princi-
ple [7]. Scene attributes comprise historical agent states X ∈RNa× Th×Da and map polyline inputs
M ∈RNm× Lm×Dm, where we portion Nm map segments with length Lm from full scene map.
Both attributes are encoded separately as SA ∈RNa×D and SM ∈RNm×D and concatenated as
scene features S = [SA; SM] ∈R(Na+Nm)×D. A stack of Transformer encoders with local attention
are directly employed in capturing regional interactions from encoded scene semantics SA, SM.

Synergistic decoder. Retaining encoded scene features SA, SM, we zoom in the decoding strategy
that asks for: 1) interactively reason simultaneous BeTop formulations; 2) selectively decoding of
compliant interactive semantics leveraging reasoned topology priors. To this end, we introduce
the iterative process of N Transformer decoder layers contributed to all agents, pursuing the basis
from [78]. To iron out the scene uncertainties, a multi-modal set of M decoding queries Q0
A ∈RM×D

are initialized for multi-agent future trajectories. Meanwhile, relative attributes SR ∈RNa×Na×DR
are deployed through MLPs as topology features Q0
R ∈RNa×Na×D for edge topology reasoning.

Next, we devise dual infostreams to the iterative decoding process for ˆV of future trajectories and ˆE
of future topology. Given agent An, the decoding process in layer l follows:

Ql,n
R = TopoDecoder

Ql−1,n
A
, Ql−1,n
R
, SA

, ˆel
n = TopoHead

Ql,n
R

,
(3)

Ql,n
A = TransDecoder

Ql−1,n
A
, SA, SM, ˆYl−1
n
, ˆel
n

, ˆYl
n = IPPHead(Ql,n
A ),
(4)

where both future trajectories ˆYn ∈ˆV and interactive topology ˆen ∈ˆE in BeTop are decoded in
synergistic manners. Reasoned edge topology ˆel
n ∈RM×Na are garnered by topological decoder with
query broadcasting Ql−1,n
A
; Reasoning nodes for ˆYn, a Transformer decoder with topology-guided
local attention are drafted serving ˆel
n as priors. We provide further details in Appendix C.1.

Topology-guided local attention. Querying whole-scene agent semantics results in misaligned
interactive agents and sparse attention. This motivates our design for local attention guided by the
reasoned topology ˆel
n ∈RM×Na as priors. Specifically, we retrieve the top-K index ϵl
n ∈RM×K

priored from ˆel
n for eventual interactive agents behaviors with An. Interactive indices are directly
leveraged in gathering SA selectively for local cross-attention. This process is formed as:

Cl,n
A = TopoAttn

Ql−1,n
A
, SA, ˆel
n

→MultiHeadAttn

q = Ql−1,n
A
; k, v = Si∈ϵl
n
A

,
(5)

where ϵl
n = argmaxK(ˆel
n). Topology-guided agent features Cl,n
A are then aggregated in each layer.

5


---Page Break---
Reason heads. Given respective decoding features Ql,n
R and Ql,n
A for each layer, we affix reason
heads accustomed to corresponding formulations for ˆen and ˆYn. Referred in Eq. (3), the topology
head, planning head, and prediction head (IPP heads) are jointly devised by stacked MLPs in
reasoning BeTop results. For agent An in each layer, reason heads decode GMM components of
future states ˆyn ∈RM×Tf ×5 (referring to (µx, µy, log σx, log σy, ρ) per step) with mixture score
ˆpn ∈RM,{ˆyn, ˆpn} ∈ˆYn, as well as interactive edge topology ˆel
n ∈RM×Na for BeTop.

3.3
Imitative Contingency Learning

Pursuing the target in Eq. (1), BeTopNet learns end-to-end objectives imitating human-like multi-agent
behaviors, integrating compliant behaviors by contingency planning under scenario uncertainties.

Imitation learning. Imitation objectives are firstly established in regulating multi-agent behavioral
states { ˆYn} ⊂ˆV while maximizing their interactive distributions ˆE. The imitative objective for
ˆY is defined by the negative log-likelihood (NLL) from best-reasoned components m∗closest to
ground-truths, as denoted: LV = PTf
t
LNLL(ˆym∗,t
n
, ˆpm∗
n , Yn). Followed Eq. (2), the behavioral
distributions for edge topology are computed by binary cross-entropy (BCE) given gathered ˆem∗
n
∈
RNa, formulated as LE = PNa
j
H(ˆem∗
n,j, en,j) over Na agents jointly.

Integrated contingency planning. To integrate compliant behavior learning for G amidst multi-
agent scenario uncertainties, contingency planning [79, 76] is turned out an apt solution. Bridging
immediate safe maneuvers τM to branched planning sets {τJ} with joint prediction, it adjourns
uncertain decisions and ensures actual safety. While direct joint prediction may lose diversity [11],
reasoned topology ˆE serves as a suitable medium distilling future interactive agents for efficient
joint combination. Given imitative AV planning outputs τ ⊂ˆY1 with branching time tb ∈(1, Tf),
integrating contingency learning asks for a safe short-term plan τM ∈TM, TM ∈RM×tb×2 to full
marginal predictions ˆYM = ˆY2:Na, as well as M branched planning sets T m
J
= {τ 1:Mb
J
}m guided
by joint predictions ˆY m
J . This is defined by:

τ ∗
M = argmin
τ⊂ˆY1
max
ˆY
CM

τM, ˆYM

+
X

m
P( ˆY m
J )CJ

T m
J , ˆY m
J

,
(6)

where max ˆY CM denotes worst-case cost fir τM; Joint predictions ˆYJ with scene probabilities
P( ˆYJ) are recombined by KM interactive agent subsets, indexing ϵAV ∈RKM from sorted AV
topology: ϵAV = argmaxKM (maxM ˆe1). It is described by joint costs CJ in guiding branched
planning maneuvers. Specifically, both cost functions are defined by the repulsive potential field [8]
discouraging planning proximity with respective prediction formulations.

Training loss. BeTopNet is trained end-to-end through imitative objectives and contingency planning
costs by weighted integration for each layer, whenever applicable (for the datasets). Please refer
to Appendix C.2 for additional details.

4
Experiment

With preliminary analysis in Sec. 3.1, this section further discovers the following questions: 1) Can
BeTop perform compliant planning via BeTopNet, especially in interactive scenarios? 2) Can BeTop
achieve accurate marginal and joint predictions of heterogeneous agents under diverse real-world
cases? 3) Can the formulated BeTop facilitate existing state-of-the-art prediction and planning
methods? and 4) How do the functionalities in BeTopNet affect the performance?

Benchmark and metrics. BeTop is verified on diverse benchmarks. We leverage two large-scale
real-world datasets, i.e., nuPlan [47] and Waymo Open Motion Dataset (WOMD) [35], which are
presently the most diverse motion datasets in manifesting planning and prediction performance. For
planning tasks in nuPlan, there are in total 1M training cases with 8s horizons. 8,300 separated
testing set are chosen by Test14-Hard and Test14-Random benchmarks [73] for hard-core and general
driving scenes. With further demands verifying maneuvers under interactive cases, we build the
Test14-Inter benchmark filtering 1,340 scenes by testing set. Scenarios ranging 15 seconds are tested
under three tasks: 1) open-loop (OL), 2) close-loop non-reactive (CL-NR) simulations, and 3) reactive

6


---Page Break---
Table 2: Performance comparison of open- and closed-loop planning on nuPlan benchmarks.
BeTopNet positions top average planning score and non-reactive simulation amongst SOTA planning
systems by all types (rule, learning, and hybrid), especially under difficult benchmarked scenarios.

Type
Method
Test14 Hard
Test14 Random
OLS ↑
CLS-NR ↑
CLS ↑
Avg. ↑
OLS ↑
CLS-NR ↑
CLS ↑
Avg. ↑

Expert
Log Replay
1.000
0.860
0.688
0.849
1.000
0.940
0.759
0.900

Rule
IDM [70]
0.201
0.562
0.623
0.462
0.342
0.704
0.724
0.590
PDM-Closed [69]
0.264
0.651
0.752
0.556
0.463
0.901
0.916
0.760

Hybrid
GameFormer [8]
0.753
0.666
0.688
0.702
0.794
0.808
0.793
0.798
PDM-Hybrid [69]
0.738
0.660
0.758
0.719
0.822
0.902
0.916
0.880

Learning

UrbanDriver [74]
0.769
0.515
0.491
0.592
0.824
0.633
0.610
0.689
PDM-Open [69]
0.791
0.335
0.358
0.495
0.841
0.528
0.572
0.647
PlanCNN [72]
0.524
0.494
0.522
0.513
0.629
0.697
0.675
0.667
GC-PGP [83]
0.738
0.432
0.396
0.522
0.773
0.560
0.514
0.616
PlanTF [73]
0.833
0.726
0.617
0.725
0.871
0.865
0.806
0.847
BeTopNet (Ours)
0.840
0.771
0.688
0.766
0.876
0.902
0.857
0.878

Table 3: nuPlan closed-loop planning results on the proposed interactive benchmark. BeTopNet
achieves desirable PDMScore, with planning safety, road compliance, and driving progress.

Type
Method
Test14 Inter
Col. Avoid ↑
Drivable ↑
Direction ↑
Progress ↑
TTC ↑
Comfort ↑
PDMScore ↑

Expert
Log Replay
1.000
1.000
1.000
0.881
1.000
0.999
0.950

Rule
PDM-Closed [69]
0.886
1.000
1.000
0.818
0.853
0.999
0.833

Learning

Constant Acc.
0.449
0.509
0.651
0.048
0.419
1.000
0.108
UrbanDriver [74]
0.970
0.955
0.992
0.798
0.932
1.000
0.854
PlanCNN [72]
0.902
0.895
0.973
0.678
0.859
0.999
0.720
PlanTF [73]
0.982
0.946
0.992
0.825
0.952
0.999
0.871
BeTopNet (Ours)
0.983
0.960
0.999
0.859
0.950
0.999
0.894

(CL-R) ones by nuPlan simulator. We report the official Planning Scores [80] computed by each task.
The motion prediction tasks in WOMD share 487k training scenarios, with 44k validation and 44k
testing set separately partitioned under two challenges: 1) The Marginal prediction challenge [81]
forecasting multiple scene agents independently; 2) The Joint prediction challenge [82] predicting
joint trajectory collections by two interactive agents. Primary metrics of mAP and Soft mAP are
ranked for official leaderboards [81, 82]. We leave experimental details in Appendix D.

4.1
Main Result

Performance for interactive planning. Table 2 demonstrates the planning results under difficult
and regular test cases. Notably, BeTopNet marks top average planning scores, achieving +7.9%
in hard cases and excels +6.2% (CLS-NR) in closed-loop simulations. Specifically, it gains solid
improvements against learning-based planners. This can be attributed to topological formulations
learning stabilized joint behavioral patterns, boosting +6.2%, +4.3% non-reactive simulations by real-
world logs and enhancing reactive simulation (+11.5%, +6.3%). Contingency objectives enhance
uncertainty compliance, leading to expanded results in hard scenarios. Meanwhile, BeTopNet also
outperforms rule-based and hybrid planning agents asking for post-optimizations [8] or hefty rules
in coinciding with reactive simulation setups [69, 70]. We report +15.8% and +18.4% results of
non-reactive simulation in hard cases and close performance in general scenes. Moreover, interactive
planning compliance is also verified in the proposed Test14-Inter benchmark centering on interactive
scenarios. As in Table 3, BeTopNet fosters +3.8% planning score over previous methods, marking
+5.5% driving progress and +2.9% driving compliance closest to human performance. Qualitative
results of interactive scenarios in Fig. 5(a-d) further corroborate planning compliance by BeTop.

Performance for marginal and joint motion prediction. Marginal prediction results are in Ta-
ble 4. Without the aids of model ensembles or extra data [25, 59], BeTopNet outperforms existing
approaches, manifesting +2.7% and +3.4% mAP metrics comparing concurrent methods [53, 86]
for compliant predictions. Exhibited strong prediction displacement metric (−4.3% minFDE) over
methods using extra pretraining [85], it should be noted that displacement metric is less illustrative as
it discounts uncertainty scoring. BeTopNet further outperforms +6.0% and +26.1% Soft mAP over
multi-agent predictors solely leveraging scenario attention [25] or graph [24]. N. Table 5 exhibits the

7


---Page Break---
Table 4: Performance of marginal prediction on WOMD Motion Leaderboard.
BeTopNet
surpasses existing motion predictors without model ensemble or using extra data. † extra LIDAR
data and pretrained model. Primary metric.

Set split
Method
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑
Soft mAP ↑

Test

ReCoAt [84]
0.7703
1.6668
0.2437
0.2711
-
HDGT [24]
0.5933
1.2055
0.1854
0.3577
0.3709
MTR [22]
0.6050
1.2207
0.1351
0.4129
0.4216
MTR++ [25]
0.5906
1.1939
0.1298
0.4329
0.4414
MGTR† [85]
0.5918
1.2135
0.1298
0.4505
0.4599
EDA [53]
0.5718
1.1702
0.1169
0.4487
0.4596
ControlMTR [86]
0.5897
1.1916
0.1282
0.4414
0.4572
BeTopNet (Ours)
0.5723
1.1668
0.1176
0.4566
0.4678

Val

MTR [22]
0.6046
1.2251
0.1366
0.4129
-
EDA [53]
0.5708
1.1730
0.1178
0.4353
-
BeTopNet (Ours)
0.5716
1.1640
0.1177
0.4416
-

Table 5: Performance of joint prediction on WOMD Interaction Leaderboard. BeTopNet
outperforms in both mAP metrics. Primary metric.

Set split
Method
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑
Soft mAP ↑

Test

HeatIRm4 [37]
1.4197
3.2595
0.7224
0.0804
-
M2I [31]
1.3506
2.8325
0.5538
0.1239
-
GameFormer [8]
0.9721
2.2146
0.4933
0.1923
0.1982
AMP [32]
0.9073
2.0415
0.4212
0.2294
0.2365
MTR++ [25]
0.8795
1.9505
0.4143
0.2326
0.2368
BeTopNet (Ours)
0.9744
2.2744
0.4355
0.2412
0.2466

Val

MTR [22]
0.9132
2.0536
0.4372
0.1992
-
AMP [32]
0.8910
2.0133
0.4172
0.2344
-
BeTopNet (Ours)
0.9304
2.1340
0.4154
0.2366
-

joint prediction results. BeTopNet outperforms all methods in both mAP metrics (+4.1%, +3.7%
Soft mAP and mAP), presenting robust prediction compliance credit to BeTop formulations for
stable future interaction patterns and aligned by local attention in BeTopNet. Particularly, BeTop
shows interactive compliance, improving +5.1% mAP over recent auto-regressive approaches [32],
boosting +25.4% mAP with game-theoretic methods [8] by a large margin. Fig. 5 (e-h) demonstrates
the qualitative prediction performance by BeTopNet. At the time of submission, BeTopNet ranked
1st on both WOMD prediction leaderboards [82, 81].

4.2
Ablation Study

Instructed by the last two motivating questions, we investigate the effect of BeTop formulations and
components inside BeTopNet. For efficient study, we randomly partition 20% of WOMD train set for
prediction, and directly report the planning results by Test14-Random benchmark, which are both
representative for the original datasets as verified by [22, 73].

Synergy with existing state-of-the-art methods. We first study the effect adjoining BeTop as
synergistic objectives over existing SOTA methods in planning and prediction. Described in Table 6
and Table 7, BeTop augments +2.1% and 2.0% planning score with learning-based and rule-based
planners, respectively. Similar compliance effects are also witnessed in guiding strong motion
predictors, bringing +1.1%, +2.4% improved mAP with −1.7% prediction errors of minADE.

Number of interactive agents for topology-guided local attention. In determining the number K
future interactive agents for BeTopNet in local attention, we validate the prediction mAP under an
array of agent numbers. Shown in Fig. 4, we observe a converging effect, with maximum +3.7%
mAP by the growing number of interactive agents. A drop of −1.8% mAP is captured after the peak
performance of K = 32. It is due to falsely accepting non-interactive agent values by large K.

Different functionalities in BeTopNet. We further investigate the effects of different functionalities
for BeTopNet in Table 8. Compared to the full model, ablations in ID.1 and ID.2 underscore the
imitative contingency learning process for costs (−2.9% CLS) and contingency branching (−1%
CLS-NR). Sole imitative BeTopNet performs the best OLS (ID.2), while the stabilizing effects found
in Sec. 4.1 are verified (−2.8% CLS-NR) in comparing ID.3-ID.5 for joint interactive patterns.

8


---Page Break---
Table 6: Results of integrating BeTop by strong
planning baselines in nuPlan benchmark.

Method
nuPlan
OLS ↑
CLS-NR ↑
CLS ↑
Avg. ↑

PDM [69]
0.463
0.898
0.918
0.760
PDM [69] +BeTop
0.488
0.916
0.902
0.770

PlanTF [73]
0.871
0.864
0.805
0.847
PlanTF [73] +BeTop
0.878
0.882
0.807
0.856

Table 7: Results of integrating BeTop by strong
prediction baselines in WOMD benchmark.

Method
WOMD
minADE ↓
minFDE ↓
MR ↓
mAP ↑

MTR [22]
0.6046
1.2251
0.1366
0.4164
MTR [22] +BeTop
0.5941
1.2049
0.1328
0.4249

EDA [53]
0.5708
1.1730
0.1178
0.4353
EDA [53] +BeTop
0.5742
1.1853
0.1181
0.4407

Figure 4: Results of different interactive agents
number for local attention. We observe a con-
vergence effect for the selection of K.

0.395

0.39

0.385

0.38

0.375

0.15

0.148

0.146

0.144

0.142

0.392

0.378

0.149

0.143

mAP
Miss Rate

Number of Interactive agents K 

Table 8: Results of BeTopNet planning perfor-
mance with different components. Contingency
is the key for closed-loop simulation.

ID
Ablative
nuPlan
Components
OLS ↑
CLS-NR ↑
CLS ↑

0
BeTopNet
0.876
0.902
0.857

1
No branched plan
0.879
0.894
0.830
2
No cost learning
0.882
0.888
0.807
3
BeTop only
0.877
0.876
0.804
4
No local attention
0.871
0.852
0.804
5
Encoders only
0.867
0.827
0.784

t=0s

t=8s

p=0

p=1

Test14-Inter
Test14-Hard
a)
b)
c)
d)
Test14-Random

Marginal Predictions
Joint Predictions
e)
f)
g)
h)

t=0s

t=8s

t=0s

t=8s

Preds
Plan

Preds Probs

Figure 5: Qualitative results of planning and prediction in nuPlan and WOMD. BeTopNet
performs compliant reaction simulations in a) yielding for pedestrians; b) cruising in dense traffic.
Interactive scenarios (c,d) further present the consistency of contingency learning. BeTopNet predicts
both compliant marginal (e,f) and joint (g,h) multi-agent predictions under diverse scenarios. Future
interactive behavior patterns can also be consistently reasoned (rendered in light red) with BeTop.

5
Conclusion

In this paper, we present BeTop, a topological new-look for multi-agent behavioral formulation.
Derived by braid theory, the reasoning tasks for BeTop are drafted supervising joint interactive
patterns with integrated prediction and planning. A synergistic network, BeTopNet, is established
with an imitative contingency learning process to boost compliant BeTop reasoning. Experiments on
nuPlan and WOMD verify BeTopNet’s state-of-the-art performance in prediction and planning.

Limitation and Future work. Current BeTop considers one-step future topology alone, and focuses
on prediction and planning. Future work would be centered on developing a recursive version of
BeTop in multi-step, multi-agent reasoning and coordination. Another promising direction would be
the connectivity of BeTop upon perceptions as tracking for the end-to-end paradigm, as well as an
extension on reasoning behaviors under 3D scenarios for multiple autonomous agents.

9


---Page Break---
Acknowledgments

This work was supported in part by the Agency for Science, Technology and Research (A*STAR),
Singapore, under the MTC Individual Research Grant (M22K2c0079), the ANR-NRF Joint Grant
(No.NRF2021-NRF-ANR003 HM Science), the Ministry of Education (MOE), Singapore, under the
Tier 2 Grant (MOE-T2EP50222-0002), National Key R&D Program of China (2022ZD0160104),
NSFC (62206172), and Shanghai Committee of Science and Technology (23YF1462000).

References

[1] Long Chen, Yuchen Li, Chao Huang, Bai Li, Yang Xing, Daxin Tian, Li Li, Zhongxu Hu, Xiaoxiang Na,
Zixuan Li, Siyu Teng, Chen Lv, Jinjun Wang, Dongpu Cao, Nanning Zheng, and Fei-Yue Wang. Milestones
in autonomous driving and intelligent vehicles: Survey of surveys. TIV, 2022. 1

[2] Wenshuo Wang, Letian Wang, Chengyuan Zhang, Changliu Liu, and Lijun Sun. Social interactions for
autonomous driving: A review and perspectives. Foundations and Trends in Robotics, 2022. 1, 2, 17

[3] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end
autonomous driving: Challenges and frontiers. PAMI, 2024. 1

[4] Sergio Casas, Abbas Sadat, and Raquel Urtasun. MP3: A unified model to map, perceive, predict and plan.
In CVPR, 2021. 1, 3

[5] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. ST-P3: End-to-end
vision-based autonomous driving via spatial-temporal feature learning. In ECCV, 2022. 3

[6] Haochen Liu, Zhiyu Huang, Wenhui Huang, Haohan Yang, Xiaoyu Mo, and Chen Lv. Hybrid-prediction
integrated planning for autonomous driving. arXiv preprint arXiv:2402.02426, 2024.

[7] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei
Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-
oriented autonomous driving. In CVPR, 2023. 1, 3, 5

[8] Zhiyu Huang, Haochen Liu, and Chen Lv. GameFormer: Game-theoretic modeling and learning of
transformer-based interactive prediction and planning for autonomous driving. In ICCV, 2023. 1, 2, 3, 6, 7,
8, 16, 19, 20, 21, 22

[9] Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris M Kitani. AgentFormer: Agent-aware transformers for
socio-temporal multi-agent forecasting. In ICCV, 2021. 1

[10] Yuxiao Chen, Peter Karkus, Boris Ivanovic, Xinshuo Weng, and Marco Pavone. Tree-structured policy
planning with learned behavior models. In ICRA, 2023. 1, 3

[11] Alexander Cui, Sergio Casas, Abbas Sadat, Renjie Liao, and Raquel Urtasun. LookOut: Diverse multi-
future prediction and planning for self-driving. In ICCV, 2021. 3, 6

[12] Stefano Pini, Christian S Perone, Aayush Ahuja, Ana Sofia Rufino Ferreira, Moritz Niendorf, and Sergey
Zagoruyko. Safe real-world autonomous driving by learning to predict and plan with a mixture of experts.
In ICRA, 2023.

[13] Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control
prediction for end-to-end autonomous driving: A simple yet strong baseline. In NeurIPS, 2022. 1

[14] Anthony Hu, Zak Murez, Nikhil Mohan, Sofía Dudas, Jeffrey Hawke, Vijay Badrinarayanan, Roberto
Cipolla, and Alex Kendall. FIERY: Future instance prediction in bird’s-eye view from surround monocular
cameras. In ICCV, 2021. 1, 3, 16

[15] Yihan Hu, Kun Li, Pingyuan Liang, Jingyu Qian, Zhening Yang, Haichao Zhang, Wenxin Shao,
Zhuangzhuang Ding, Wei Xu, and Qiang Liu. Imitation with spatial-temporal heatmap: 2nd place
solution for nuplan challenge. arXiv preprint arXiv:2306.15700, 2023. 1

[16] Reza Mahjourian, Jinkyu Kim, Yuning Chai, Mingxing Tan, Ben Sapp, and Dragomir Anguelov. Occupancy
flow fields for motion forecasting in autonomous driving. RA-L, 2022. 1, 3

[17] Haochen Liu, Zhiyu Huang, and Chen Lv. Multi-modal hierarchical transformer for occupancy flow field
prediction in autonomous driving. In ICRA, 2023. 3

10


---Page Break---
[18] Ben Agro, Quinlan Sykora, Sergio Casas, and Raquel Urtasun. Implicit occupancy flow fields for perception
and prediction in self-driving. In CVPR, 2023. 1, 16

[19] Jinkyu Kim, Reza Mahjourian, Scott Ettinger, Mayank Bansal, Brandyn White, Ben Sapp, and Dragomir
Anguelov. StopNet: Scalable trajectory and occupancy prediction for urban autonomous driving. In ICRA,
2022. 1, 3

[20] Hongyang Li, Chonghao Sima, Jifeng Dai, Wenhai Wang, Lewei Lu, Huijie Wang, Jia Zeng, Zhiqi Li,
Jiazhi Yang, Hanming Deng, Hao Tian, Enze Xie, Jiangwei Xie, Li Chen, Tianyu Li, Yang Li, Yulu Gao,
Xiaosong Jia, Si Liu, Jianping Shi, Dahua Lin, and Yu Qiao. Delving into the devils of bird’s-eye-view
perception: A review, evaluation and recipe. PAMI, 2024. 1, 3

[21] Haochen Liu, Zhiyu Huang, and Chen Lv. Occupancy prediction-guided neural planner for autonomous
driving. In ITSC, 2023. 1, 3, 16

[22] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. Motion transformer with global intention
localization and local movement refinement. In NeurIPS, 2022. 1, 3, 4, 8, 9, 18, 19, 20, 22, 23

[23] Xiaosong Jia, Li Chen, Penghao Wu, Jia Zeng, Junchi Yan, Hongyang Li, and Yu Qiao. Towards capturing
the temporal dynamics for trajectory prediction: a coarse-to-fine approach. In CoRL, 2022.

[24] Xiaosong Jia, Penghao Wu, Li Chen, Yu Liu, Hongyang Li, and Junchi Yan. HDGT: Heterogeneous
driving graph transformer for multi-agent trajectory prediction via scene encoding. PAMI, 2023. 1, 3, 7, 8,
16, 20

[25] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. MTR++: Multi-agent motion prediction with
symmetric scene modeling and guided intention querying. PAMI, 2024. 1, 7, 8, 16, 20

[26] Zhiyu Huang, Chen Tang, Chen Lv, Masayoshi Tomizuka, and Wei Zhan. Learning online belief prediction
for efficient pomdp planning in autonomous driving. arXiv preprint arXiv:2401.15315, 2024. 1, 3

[27] Zhiyu Huang, Peter Karkus, Boris Ivanovic, Yuxiao Chen, Marco Pavone, and Chen Lv. DTPP: Differen-
tiable joint conditional prediction and cost evaluation for tree policy planning in autonomous driving. In
ICRA, 2024. 1, 3

[28] Zhiyu Huang, Haochen Liu, Jingda Wu, and Chen Lv. Conditional predictive behavior planning with
inverse reinforcement learning for human-like autonomous driving. TITS, 2023. 1, 2, 16

[29] Jiquan Ngiam, Vijay Vasudevan, Benjamin Caine, Zhengdong Zhang, Hao-Tien Lewis Chiang, Jeffrey
Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish Venugopal, David J Weiss, Benjamin Sapp,
Zhifeng Chen, and Jonathon Shlens. Scene Transformer: A unified architecture for predicting future
trajectories of multiple agents. In ICLR, 2022. 2, 3

[30] Steffen Hagedorn, Marcel Hallgarten, Martin Stoll, and Alexandru Condurache. Rethinking integration
of prediction and planning in deep learning-based automated driving systems: a review. arXiv preprint
arXiv:2308.05731, 2023. 2

[31] Qiao Sun, Xin Huang, Junru Gu, Brian C Williams, and Hang Zhao. M2I: From factored marginal
trajectory prediction to interactive prediction. In CVPR, 2022. 2, 3, 8, 16, 20

[32] Xiaosong Jia, Shaoshuai Shi, Zijun Chen, Li Jiang, Wenlong Liao, Tao He, and Junchi Yan. AMP:
Autoregressive motion prediction revisited with next token prediction for autonomous driving. arXiv
preprint arXiv:2403.13331, 2024. 2, 3, 8, 16, 20, 22

[33] Jose Luis Vazquez Espinoza, Alexander Liniger, Wilko Schwarting, Daniela Rus, and Luc Van Gool. Deep
interactive motion prediction and planning: Playing games with motion prediction models. In L4DC, 2022.
2, 16

[34] Wei Zhan, Changliu Liu, Ching-Yao Chan, and Masayoshi Tomizuka. A non-conservatively defensive
strategy for urban autonomous driving. In ITSC, 2016. 2

[35] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning
Chai, Ben Sapp, Charles R. Qi, Yin Zhou, Zoey Yang, Aurélien Chouard, Pei Sun, Jiquan Ngiam, Vijay
Vasudevan, Alexander McCauley, Jonathon Shlens, and Dragomir Anguelov. Large scale interactive motion
forecasting for autonomous driving: The waymo open motion dataset. In ICCV, 2021. 2, 4, 6, 20, 23, 29

[36] Dan Xie, Tianmin Shu, Sinisa Todorovic, and Song-Chun Zhu. Learning and inferring “dark matter” and
predicting human intents and trajectories in videos. PAMI, 2017. 2

11


---Page Break---
[37] Xiaoyu Mo, Zhiyu Huang, Yang Xing, and Chen Lv. Multi-agent trajectory prediction with heterogeneous
edge-enhanced graph attention network. TITS, 2022. 2, 3, 8, 20

[38] Zhiyu Huang, Xiaoyu Mo, and Chen Lv. Multi-modal motion prediction with transformer-based neural
network for autonomous driving. In ICRA, 2022. 2

[39] Yuriy Biktairov, Maxim Stebelev, Irina Rudenko, Oleh Shliazhko, and Boris Yangel. PRANK: motion
prediction based on ranking. In NeurIPS, 2020. 2

[40] Daehee Park, Hobin Ryu, Yunseo Yang, Jegyeong Cho, Jiwon Kim, and Kuk-Jin Yoon. Leveraging future
relationship reasoning for vehicle trajectory prediction. In ICLR, 2023. 3, 16

[41] Jiachen Li, Fan Yang, Masayoshi Tomizuka, and Chiho Choi. EvolveGraph: Multi-agent trajectory
prediction with dynamic relational reasoning. In NeurIPS, 2020. 2, 3, 4, 16

[42] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In
ICLR, 2017. 2

[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 2

[44] Christoforos Mavrogiannis, Jonathan DeCastro, and Siddhartha S Srinivasa. Analyzing multiagent interac-
tions in traffic scenes via topological braids. In ICRA, 2022. 2, 3, 16, 17

[45] Christoforos Mavrogiannis, Jonathan A DeCastro, and Siddhartha S Srinivasa. Abstracting road traffic via
topological braids: Applications to traffic flow analysis and distributed control. IJRR, 2023. 2, 3, 4, 16, 17

[46] Emil Artin. Theory of braids. Annals of Mathematics, 1947. 2, 4, 16

[47] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba
Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, and Holger Caesar.
Towards learning-based planning: The nuplan benchmark for real-world autonomous driving. In ICRA,
2024. 2, 3, 6, 20

[48] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to drive by imitating the
best and synthesizing the worst. arXiv preprint arXiv:1812.03079, 2018. 3

[49] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai.
BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal
transformers. In ECCV, 2022. 3

[50] Alexey Kamenev, Lirui Wang, Ollin Boer Bohan, Ishwar Kulkarni, Bilal Kartal, Artem Molchanov, Stan
Birchfield, David Nistér, and Nikolai Smolyanskiy. PredictionNet: Real-time joint probabilistic traffic
prediction for planning, control, and simulation. In ICRA, 2022. 3

[51] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable
autonomous driving. In CVPR, 2024. 3

[52] Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia Schmid.
VectorNet: Encoding hd maps and agent dynamics from vectorized representation. In CVPR, 2020. 3

[53] Longzhong Lin, Xuewu Lin, Tianwei Lin, Lichao Huang, Rong Xiong, and Yue Wang. EDA: Evolving
and distinct anchors for multimodal motion prediction. In AAAI, 2024. 3, 7, 8, 9, 19, 20, 22, 24

[54] Charlie Tang and Russ R Salakhutdinov. Multiple futures prediction. In NeurIPS, 2019. 3

[55] Siyuan Qi and Song-Chun Zhu. Intent-aware multi-agent reinforcement learning. In ICRA, 2018. 3

[56] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde.
THOMAS: Trajectory heatmap output with learned multi-agent sampling. In ICLR, 2022. 3

[57] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde. GO-
HOME: Graph-oriented heatmap output for future motion estimation. In ICRA, 2022.

[58] Junru Gu, Chen Sun, and Hang Zhao. DenseTNT: End-to-end trajectory prediction from dense goal sets.
In ICCV, 2021. 3

[59] Balakrishnan Varadarajan, Ahmed Hefny, Avikalp Srivastava, Khaled S. Refaat, Nigamaa Nayakanti,
Andre Cornman, Kan Chen, Bertrand Douillard, Chi Pang Lam, Dragomir Anguelov, and Benjamin Sapp.
MultiPath++: Efficient information fusion and trajectory aggregation for behavior prediction. In ICRA,
2022. 3, 7

12


---Page Break---
[60] Junha Roh, Christoforos Mavrogiannis, Rishabh Madan, Dieter Fox, and Siddhartha Srinivasa. Multimodal
trajectory prediction via topological invariance for navigation at uncontrolled intersections. In CoRL, 2020.
3

[61] Christoforos Mavrogiannis, Krishna Balasubramanian, Sriyash Poddar, Anush Gandra, and Siddhartha S
Srinivasa. Winding through: Crowd navigation via topological invariance. RA-L, 2022. 3, 16

[62] Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S Refaat, and Benjamin Sapp.
Wayformer: Motion forecasting via simple & efficient attention networks. In ICRA, 2023. 3, 23

[63] Zikang Zhou, Luyao Ye, Jianping Wang, Kui Wu, and Kejie Lu. HiVT: Hierarchical vector transformer for
multi-agent motion prediction. In ICCV, 2022. 3

[64] Alexander Cui, Sergio Casas, Kelvin Wong, Simon Suo, and Raquel Urtasun. GoRela: Go relative for
viewpoint-invariant motion forecasting. In ICRA, 2023. 3, 18

[65] Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, and Marco Pavone. Trajectron++: Dynamically-
feasible trajectory forecasting with heterogeneous data. In ECCV, 2020. 3

[66] Luke Rowe, Martin Ethier, Eli-Henry Dykhne, and Krzysztof Czarnecki. FJMP: Factorized joint multi-
agent motion prediction over learned directed acyclic interaction graphs. In CVPR, 2023. 3, 16

[67] Wenjie Luo, Cheol Park, Andre Cornman, Benjamin Sapp, and Dragomir Anguelov. JFP: Joint future
prediction with interactive multi-agent modeling for autonomous driving. In CoRL, 2022. 3, 4, 16

[68] Sergio Casas, Cole Gulino, Simon Suo, Katie Luo, Renjie Liao, and Raquel Urtasun. Implicit latent
variable model for scene-consistent motion forecasting. In ECCV, 2020. 3

[69] Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions
about learning-based vehicle motion planning. In CoRL, 2023. 3, 7, 9, 18, 19, 20, 21, 23

[70] Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations
and microscopic simulations. Physical review E, 2000. 7, 20, 21

[71] Peng Hang, Chen Lv, Yang Xing, Chao Huang, and Zhongxu Hu. Human-like decision making for
autonomous driving: A noncooperative game theoretic approach. TITS, 2020. 3

[72] Katrin Renz, Kashyap Chitta, Otniel-Bogdan Mercea, A. Sophia Koepke, Zeynep Akata, and Andreas
Geiger. PlanT: Explainable planning transformers via object-level representations. In CoRL, 2022. 3, 7,
20, 21

[73] Jie Cheng, Yingbing Chen, Xiaodong Mei, Bowen Yang, Bo Li, and Ming Liu. Rethinking imitation-based
planner for autonomous driving. In ICRA, 2024. 6, 7, 8, 9, 18, 20, 21, 22, 24

[74] Oliver Scheel, Luca Bergamini, Maciej Wolczyk, Bła˙zej Osi´nski, and Peter Ondruska. Urban Driver:
Learning to drive from real-world demonstrations using policy gradients. In CoRL, 2021. 3, 7, 20, 21

[75] Peter Karkus, Boris Ivanovic, Shie Mannor, and Marco Pavone. DiffStack: A differentiable and modular
control stack for autonomous vehicles. In CoRL, 2022. 3

[76] Tong Li, Lu Zhang, Sikang Liu, and Shaojie Shen. MARC: Multipolicy and risk-aware contingency
planning for autonomous driving. RA-L, 2023. 3, 6

[77] Franklin Antonio. Faster line segment intersection. In Graphics Gems III (IBM Version), pages 199–202.
Elsevier, 1992. 4

[78] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR:
Dynamic anchor boxes are better queries for detr. In ICLR, 2022. 5

[79] Jason Hardy and Mark Campbell. Contingency planning over probabilistic obstacle predictions for
autonomous road vehicles. TRO, 2013. 6

[80] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher,
Oscar Beijbom, and Sammy Omari. NuPlan: A closed-loop ml-based planning benchmark for autonomous
vehicles. arXiv preprint arXiv:2106.11810, 2021. 7, 20, 23, 29

[81] Waymo.
Waymo open dataset motion prediction challenge 2024.
https://waymo.com/open/
challenges/2024/motion-prediction/. 7, 8, 20, 22

13


---Page Break---
[82] Waymo. Waymo open dataset interaction prediction challenge 2021. https://waymo.com/open/
challenges/2021/interaction-prediction/. 7, 8, 20, 22

[83] Marcel Hallgarten, Martin Stoll, and Andreas Zell. From prediction to planning with goal conditioned lane
graph traversals. In ITSC, 2023. 7, 20, 21

[84] Zhiyu Huang, Xiaoyu Mo, and Chen Lv. ReCoAt: A deep learning-based framework for multi-modal
motion prediction in autonomous driving application. In ITSC, 2022. 8

[85] Yiqian Gan, Hao Xiao, Yizhe Zhao, Ethan Zhang, Zhe Huang, Xin Ye, and Lingting Ge. MGTR:
Multi-granular transformer for motion prediction with lidar. In ICRA, 2024. 7, 8

[86] Jiawei Sun, Chengran Yuan, Shuo Sun, Shanze Wang, Yuhang Han, Shuailei Ma, Zefan Huang, Anthony
Wong, Keng Peng Tee, and Marcelo H Ang Jr. ControlMTR: Control-guided motion transformer with
scene-compliant intention points for feasible motion prediction. arXiv preprint arXiv:2404.10295, 2024. 7,
8, 20

[87] Mitchell A Berger. Topological invariants in braid theory. Letters in Mathematical Physics, 2001. 16

[88] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you
need for open-loop end-to-end autonomous driving? In CVPR, 2024. 18, 20

[89] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep learning on point sets for 3d
classification and segmentation. In CVPR, 2017. 18

[90] Zetong Yang, Li Jiang, Yanan Sun, Bernt Schiele, and Jiaya Jia. A unified query-based paradigm for point
cloud understanding. In CVPR, 2022. 19

[91] Zikang Zhou, Jianping Wang, Yung-Hui Li, and Yu-Kai Huang. Query-centric trajectory prediction. In
CVPR, 2023. 19

[92] Jie Cheng, Yingbing Chen, and Qifeng Chen. PLUTO: Pushing the limit of imitation learning-based
planning for autonomous driving. arXiv preprint arXiv:2404.14327, 2024. 19, 20, 21

[93] NAVSIM Contributors. NAVSIM: Data-driven non-reactive autonomous vehicle simulation. https:
//github.com/autonomousvision/navsim, 2024. 20

[94] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang
Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. NAVSIM:
Data-driven non-reactive autonomous vehicle simulation and benchmarking. arXiv, 2406.15349, 2024. 20

[95] Lan Feng, Mohammadhossein Bahari, Kaouther Messaoud Ben Amor, Éloi Zablocki, Matthieu Cord, and
Alexandre Alahi. UniTraj: A unified framework for scalable vehicle trajectory prediction. In ECCV, 2024.
23

14


---Page Break---
Appendix

A Discussions
16

B
Properties of BeTop
16

C Implementation Details
18

C.1
Model Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

C.2
Imitative Contingency Learning
. . . . . . . . . . . . . . . . . . . . . . . . . . .
19

D Experimental Setup Details
20

D.1
Planning on nuPlan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

D.2
Prediction on WOMD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

D.3 Training Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

E Additional Quantitative Results
21

E.1
Planning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

E.2
Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

F
Additional Ablation Studies
22

G Additional Qualitative Results
23

H License of Assets
23

15


---Page Break---
A
Discussions

Towards a better understanding of this work, we supplement intuitive questions that may raise. Note
that the following list does not indicate the manuscript was submitted to a previous venue or not.

Q1: How does BeTop bridge and discern with dense, sparse, and topological representations?

BeTop is derived from braid theory [46], reasoning the topology that explicitly labels consensual
interactions as dense, occupancy-like intertwines from sparse, braided future trajectories of multi-
agent behaviors. Unlike fixed occupancy [14, 21, 18], BeTop dynamically forecasts behavioral
interactions by agents collectively, serving as a guiding medium for node reasoning in joint planning
and prediction compared to standard sparse predictions [25, 24, 66]. Differentiating from typical
topological approaches, BeTop explicitly formulates coordinated multi-agent behaviors and reasons
topology in occupancy manners, rather than relying on implicit relation graph learning [41, 67, 40] or
complex braid inference [44, 45, 61].

Q2: Why is BeTop allied with contingency instead of conditional or game-theoretic reasoning?

Contingency plays the most suitable role in BeTopNet, addressing multi-agent uncertainty by defer-
ring uncertain planning with long-term joint predictions from interactive agents under behavioral
compliance. This aligns with BeTop’s reasoning targets, which aim to achieve a one-shot consensus
among all behaviors by formulating future behavior as coordinated joint interactions. This synergy al-
leviates the challenges led by game-theoretic reasoning [8, 33] or conditional integration [31, 28, 32],
which are struggled by multi-step interactive rollouts and unstable joint behavioral patterns.

Q3: What would be the broader impact and future direction?

BeTop steps the first trial towards an explicit topological formulation and reasoning paradigm for
multi-agent interactive behaviors. This serves as a basis in exploring immense interactive behaviors in
the real-world, and is reasoned by the autonomous agents jointly in a coordinated manner. For instance,
BeTop with enlarged scalability and dimension may summarize the topology for all behaviors in 3D
scenarios or even larger spaciality, and may reasoned through end-to-end BeTopNet as a foundation
behavioral model. Moreover, we can consider BeTop’s capability as collective maneuvers, which
can be further leveraged in coordinating naturalistic and efficient decision-making for multiple
autonomous agents.

B
Properties of BeTop

In this section, we supplement additional properties that characterize the formulation of Behavioral
Topology (BeTop). Analytically, BeTop is highlighted with: 1) geometric invariant (Theorem B.1); 2)
approximated topological invariant (Theorem B.2); and 3) asymmetric topology (Theorem B.3).

Theorem B.1 (Geometric Invariant). The topology results of E ⊂G in BeTop remain unchanged
given arbitrary geometrical transformations for the collective scene trajectories Yn.

Proof. Given arbitrary rotation H ∈R2×2 and shifting b ∈R2. Consider the mapping g for elemen-
tary topology eij ∈E from future trajectories (Yi, Yj), i ̸= j ∈[1, Na], the local transformation
f : g →h ◦f to i’s coordinate is invariant, such that h(f(Yi, Yj)) = h(f(HYi + b, HYj + b)).
Hence, given the function sets f i
j ∈σi ⊂f, I ∈h defined in Sec. 3.1; eij ∈E is also invariant.

Remark 1. The Theorem B.1 proves the behavioral stability of BeTop given arbitrary multi-agent
trajectories patterns for planning and predictions. Any rotations and movement of the original scene
will not interfere with the formulated results of BeTop.

Definition B.1 (Topological Invariant). Given future trajectory pairs (Yi, Yj), i ̸= j ∈[1, NA]
with certain current heading (θ0
i , θ0
j), the sum of future relative angles (winding number) wij =

1
2π
PTf
0 ∆θt
ij form its first-order [87] topological invariant.

Proof. Consider the polar representation for the closed form ψi(t) = ||ψi(t)||eiθi(t). Where ψi :
[0, Tm] →C\{0}, i ∈[0, n], we can define the winding function λi(t) =
1
2πi
R

ψi dz/z, z =

16


---Page Break---
ψi(t), t ∈[0, Tm]. This Cauchy formula [2] can be further integrated as:

λi(t) =
1
2πi log( ||ψi(t)||

||ψi(0)||) + 1

2π (θi(t) −θi(0)).
(7)

We are interested in the real (first-order) part of λi(t) which is an invariant topologically. Hence,
the trajectory pairs (Yi, Yj), 0 < t ≤Tf < Tm can be described as: Yi = ψi : (0, Tf]. The joint
invariant across future wij = PTf
t (λi(t) −λj(t)) then becomes:

wij = 1

2π

Tf
X

t
(θt
i −θt
j) −1

2π

Tf
X

t
(θ0
i −θ0
j).
(8)

As the current heading pair (θ0
i , θ0
j) is certain, the invariant becomes wij =
1
2π
PTf
0 ∆θt
ij which
proofs the definition.

Corollary B.1.1. Given any ∆θt
ij ∈[−π

2 , π

2 ], where 0 < i.j < Na, t ∈(0, Tf], and the constant

ηi, ηj ∈R, the transformed wT
ij = PTf
0 (ηj sin ∆θt
ij −ηi sin θt
i) is also topological invariant.

Proof. The defined function sin(·) is a monotone mapping under [−π

2 , π

2 ]. Hence, this firstly enables
PTf
0 ηi sin θt
i uniquely defines Yi. More than that, wT
ij is the unique mapping value of wij with Yi
under defined transformation, and thereby keep the invariant property.

Theorem B.2. The edge topology E ⊂G is an approximate of topological invariant, so that
eij ∈E, 0 < i, j ≤Na is characterized by wij .

Proof. Given future trajectories Yi, Yj, we consider the braid functions σi maps monotonically
increased transformations f i
i , f i
j to i’s local coordinate,as defined in Sec. 3.1. We assume a continuous
future horizon (0, Tf] where the headings for f i
j(Yj) is defined by relative angles ∆θij(t). Thereby,

the transformed lateral trajectory for agent j can be formed as:
R t
0 ηj sin ∆θij(t)dt. Similarly, f i
i (Yi)
can be formed as
R t
0 ηj sin θi(t)dt, where ηi, ηj denotes small constant step lengths.These form the
original intersection function I in Sec. 3.1 as:

I
 
f i
i (yt
i), f i
j(yt
j)

→
Z t

0
(ηj sin ∆θt
ij −ηi sin θt
i)dt,
(9)

where I(·) = 0 denotes braid intertwine for interactive behaviors. As the term in Corollary B.1.1 of
wT
ij (the sum of the right term) is an approximate (discretization) of the right term, this proves the
edge topology eij ∈E →maxTf I (·) as the approximation of topological invariant.

Remark 2. The Theorem B.2 proves the generality of BeTop in terms of future interactive behavioral
patterns. The approximated topological invariant property prompts a representative of various future
states sharing similar behavioral or identical interactive patterns by BeTop.

Theorem B.3 (Asymmetric Topology). Edge topology E ⊂G is not symmetric such that ∃i, j, eij ̸≡
eji, 0 < i, j ≤Na.

Proof. Given Eq. (9) defined in Theorem B.2,we can always construct a case ∃ti, tj ∈(0, Tf], the
intersection I
 
f i
i (yti
i ), f i
j(yti
j )

= 0, but I(f j
i (ytj
j ), f j
i (ytj
i )) ̸= 0, which prove the claim.

Remark 3. The Theorem B.3 proves a more naturalistic interactive behavior of BeTop. It is likely in a
real-world scenario that the future behavior of agent Ai is interacted by agent Aj, while Aj does not.

Computational complexity. The complexity in computing full E is O(N 2
a). In practice, we
downscale the sourced as the agents of interests NI < Na, such that O(NINa). It is much less
than braid sequence inference [45] with maximum O((Na −1)Na) computational costs. Further
analytical proof of computational efficiency leveraging braids can be found in [44].

17


---Page Break---
Multi Head
Self-Attention

Topology-guided

Local Attention

Add & Norm

𝐐𝑨

𝒍,−𝟏 𝒏[M,D]

Q
K
V

FFN

Q
K
V

Add & Norm

Topology

Decoder

𝐐𝑹

𝒍,𝒏 [M,𝐍𝐚,D]

𝐒𝑹

𝒏 [𝐍𝐚,D]

𝐒𝑨

[𝐍𝐚,D]
[M,K,D]

Topology

Head

…

…

ො𝒆𝒏𝒍 [M,𝐍𝐚]

[M,K]

𝝐𝒏𝒍

BeTopNet decoder layer

MLP

MLP

𝐒𝑨

𝐐𝑨

𝒍−𝟏,𝒏

MLP

Concat.

𝐐𝑹

𝒍−𝟏,𝒏

[M,𝐍𝐚,𝐃𝐚]

Topology decoder

Planning Head

…

𝐐𝑨

𝒍,𝟏

a)
b)

c)

MLP

MLP

Concat.

𝑪𝑨

𝒍,𝒏∈𝐐𝑨

𝒍,𝒏

[M,𝒕𝒃,2]
𝜏⊂෡𝐘1

[𝑴𝑴𝑱,𝑻𝒇−𝒕𝒃,2]

𝝉𝑴
𝝉𝑱

MLP

[M,𝑴𝑱,D]

෠𝐘1

𝐐𝑹

𝒍,𝒏

Figure 6: Structural details in BeTopNet. a) The learning structure of single synergistic decoder
layer featuring TransDecoder and TopoAttn design; b) The structure inside topology decoder
network of TopoDecoder; c) Branched planning head design corresponding to contingency planning.

C
Implementation Details

In this section, we instantiate further details for BeTop on the configurations for BeTopNet structure,
and provide contingency learning paradigms for both prediction and planning challenges.

C.1
Model Structure

Subjecting to different testing requirements defined in nuPlan and WOMD, we set up two model
variants for BeTopNet in formulating planning and prediction challenges. Apart from topology
reasoning for interactive behaviors, for planning challenges, BeTopNet integrates both tasks of
prediction and planning. In the prediction challenges under marginal and joint settings, BeTopNet is
allocated only with the prediction parts. The structural details are illustrated below.

Scene inputs. Carving the driving scenarios involves historical agent states X and map polylines
M as scene inputs. For planning settings, we collect scene agent states with past Th = 2 seconds
at 10Hz, leaving basic kinematic states as (x, y, vx, vy, θ), joining with agent shapes and types. We
only keep the current state for ego vehicle (AV) in preventing closed-loop gap [73, 69, 88] with
open-loop training as recently discussed in the community. Nm = 256 segments of the map with
length Lm = 20 are gathered by scene-centric manners considering positions, traffic lights, and speed
limits. The prediction task is built on WOMD considering Th = 1 seconds at 10Hz with scenarios of
larger scalability. It is followed by full scene agents with Nm = 768 map segments of identical states
for both X and M.

Scene encoder. Both scene attributes X, M are firstly encoded leveraging layered point encoder [89]
to hidden dimension D shared throughout the BeTopNet structures, with D = 128 for planning and
D = 256 for prediction tasks. A stack of Transformer encoders is then devised with 4 layers in
planning and 6 in prediction for SA, SM. Due to scalable settings for prediction tasks, local attention
with the nearest 16 keys is built in each layer. Following [22, 73], the dense prediction head is adopted
for all agents after the encoder, enhancing future semantics.

Synergistic decoder. Depicted in Fig. 6, the decoder structure is founded by an iterative stack of
L Transformer decoders querying M modes of future trajectories ˆY with dual stream in reasoning
topology ˆE. Consisting of L = 6 and L = 4 decoders for prediction and planning respectively, it is
initialized jointly by relative features Q0,n
R
and decoding queries Q0,n
A . Relative attributes SR are
computed efficiently following [64] for relative distance and headings. M = 6 learnable embedding
are devised as Q0,n
A
for planning, and we utilize the anchored ones [22] with M = 64 in prediction
tasks. As displayed in Fig. 6(a), dual queries Qj−1,n
A
, Qj−1,n
R
are served from the last level. The

18


---Page Break---
decoding process (following Eq. (3)) iteratively reasons ˆel
n from TopoDecoder, serving as a prior
guiding agent semantics by TopoAttn inside TransDecoder, which concurrently aggregates scene
semantics from agents SA and maps SM. Expressly, the structure of TopoDecoder (Fig. 6(b))
comprises simple MLPs and update Ql,n
R by concatenation sourcing query features Qj−1,n
A
with
agent semantics SA, and connect residuals Ql−1,n
R
from last layer. Decoding queries Ql,n
A is updated
by a concatenation from aggregated agent feature Cl,n
A , map features Cl,n
M , and agent semantics Sn
A
directly from encoder. Agent feature Cl,n
A is aggregated by TopoAttn, where the local attention is
devised using deployments from [90] indexing K = 32 agents from reasoned topology ˆel
n. We omit
the aggregation process for map features, which performs the vanilla Transformer decoder structure
for planning and dynamic collection form by [22] under prediction tasks for hefty map features.

Reason heads. Following the contents in Sec. 3.2, reason heads in decoding prediction and topology
follow simple MLPs given respective decoding queries. For the planning head, it leverages a
cascaded design for branched contingency planning with multi-modalities (Fig. 6(c)). Specifically,
the short-planning τM is decoded by the AV future states {ˆy1:tb
1
}m from first stage head with m ∈M
modes, where tb = 3 denotes the branching time. They are then detached and leveraged as prior
for the branched planning. Successive MLPs project and reshape the short-term contingency prior
by RM×MJ×D for MJ = 6 branches planning T m
J
under each of τ m
M. Further concatenated by
broadcasted decoding queries, the planning head generates M · MJ trajectories ˆY1 for AV.

C.2
Imitative Contingency Learning

Efficient joint prediction recombination. Retrieving the top-performed joint predictions from full
marginal predictions ˆYM sequentially is time-consuming with exponentially complexity. Hence, we
firstly downscale the potentially interested agents NI by sorting the AV-reasoned topology ˆeL
1 with
the largest KM = 4 value as index given the planning task. For the joint prediction task, NI = 2 is
annotated already from the original data. Then, we leverage the tensor broadcasting mechanism in
efficiently retrieving M = 6 largest joint distributions P( ˆY M
J ) and joint trajectories ˆY M
J
from NI
interacted agents. Given a tensor PJ ∈RM NI initialized by ones, the joint score is computed by NI
times of iterative broadcasting ˆpn on the n-th dimension for PJ as PJ = maxM
QNI
n P(n)
J
⊗ˆpn.
This process only costs 1.6ms in computing NI = 4 joint predictions for contingency planning.

Imitative contingency objectives. Followed by learning objectives derived in Sec. 3.3, the imitation
objectives for each layer can be represented as LIL = LV + λ1LE. λ1 = 50 weighting BCE loss for
edge topology reasoning, the NLL loss for LV is formulated as:

LNLL = log σx + log σy +
log(1−ρ2)

2
+
1
2(1−ρ2)


dx
σx

2
+

dy
σy

2
−2ρ dxdy

σxσy


−log p(m∗),
(10)

where dx, dy denotes the difference with ground-truths. In determining the component m∗, we
leverage a winner-take-all (WTA) strategy [91] in planning by measuring the average displacements
(ADE) with groung-truths. For prediction tasks, m∗is selected from the closest anchor as in [53]. For
the learnable cost functions max CM(·), CJ(·) in contingency planning, we leverage the repulsive
potential field [8] delineating planning with prediction by ϕ = mind 1/(1+d(τ, ˆy)). For max CM(·),
ϕ is gathered across Tf considering the worst case under full marginal prediction ˆYM comprising
Na = 32 scene agents. For the branched cost CJ(·), ϕ for each branch is computed considering
joint prediction from NI = 4 agents. Following the objective defined in Eq. (6), the learnable
contingency cost is defined as: LCL = CM + PM
m P( ˆY m
J )Cm
J . Hence, the general objectives for
planning become:
L = LV + λ1LE + λ2LCL,
(11)
where λ2 = 5 is the contingency costs weight. Prediction tasks are updated only by LIL.

Inference. Different from the training process, for the planning task we directly select the full
planning trajectory of Tf = 8 seconds by highest scoring τ ∗= argmaxC ˆY1, subjecting to the
original task settings in nuPlan. The scoring results are a combination from original confidence ˆp1
and the short-term cost CM [69]: C = ˆp1 + λmCM, where λm = 0.5 facilitates short-term planning
compliance [92]. For the prediction task, a post-processing module following [53] is leveraged in
selecting M = 6 marginal or joint trajectories of Tf = 8 seconds among 3 agent types in WOMD.

19


---Page Break---
D
Experimental Setup Details

In this section, we provide extra details demonstrated in Sec. 4 for the experiment setups, including
detailed settings for the proposed benchmark, testing metrics, state-of-the-art baselines, and training.

D.1
Planning on nuPlan

Testing metrics. For open-loop planning tests, the open-loop score (OLS) serves as the general
statistics weighting displacement metrics and miss rates. For closed-loop simulations, both metrics
(CLS, CLS-NR) are weighted by a series of statistics measuring 1) driving safety, 2) planning
progress, 3) driving comforts, and 4) rule obeying. The PDMScore [93, 94] compared in Table 3 is
basically a replica of the closed-loop score for efficient computations. It is denoted as:

PDMScore = CA · DAC · DDC · w1TTC + w2DC + w3EP
P wi
,
(12)

where the sub-metrics are abbreviations referred in Table 3. All general metrics range from 0 to 1.

Test14-Inter. We launch the Test14-Inter benchmark in verifying the planning systems under typical
corner cases containing rich social interactions, or dynamic profiles by complex map forms. This is
highly motivated by the issues raised in [69, 88], that massive scenarios may also be completed by
a simple motion model. Specifically, we adopt a mining heuristic defining corner cases by which
human experts excel but the motion model (constant acceleration vehicle, CAV) fails. For efficient
mining, we directly assess planning results by PDMScore and define the criteria as:

(PDMScoreCAV < γ) ∧(PDMScoreExpert ≥(1 −γ)),
(13)

where γ = 0.1 denotes a scoring threshold for cases that cannot be easily solved by regular motion
profiles of the planning maneuvers. As future work, we aim to explore more interactive scenarios
aggravating by BeTop as an enhancement.

Val14. In pursuing comprehensive comparisons with current methods, we also manifest BeTopNet
in the Val14 set proposed in [69]. It is a subset of 1040 scenes from the validation set. However,
since a portion of validation scenes are shared with the training set in nuPlan [80], we argue this
is less representative of testing fairness for learning-based methods. Hence, we only place it as
supplementary.

Baselines. For all baselines presented in the planning task, we directly report their previous benchmark
results. Additional results in the proposed benchmark (Table 3) and ablation studies (Table 6) are
re-implemented by the official releases [69, 72, 47, 69]. Expressly, we study the state-of-the-art
planning systems categorized by: 1) Rule-based: performing maneuvers by designate rules with the
reactive agents [70] or mimicking the planning score [69]; 2) Hybrid: incorporating rules [69] or
post-optimizations [8] with a learning-based model; and 3) Learning-based: end-to-end planning
with GNN [83, 74] or Transformer [72, 73] enabled models, as well as concurrent methods [92]
augmented by representation learning. For ablation studies in Table 6, BeTop is trained directly
by the proposed topology decoder with the original PlanT [73] pipeline. For the PDM [69] as a
rule-based planning system, we integrate BeTop by replacing the original rule-based motion model
with predictions generated from BeTopNet.

D.2
Prediction on WOMD

Testing metrics. For the prediction task, the mean AP (mAP) and Soft-mAP scores are assigned as the
primary metrics in computing multi-modal predictions modeled by marginal or joint distributions [35,
81, 82]. Displacement metrics of minADE and minFDE provide the multi-modal prediction errors
closest to ground truths without considering the prediction scores.

Baselines. We also directly provide the prediction results displayed on the official leaderboards
in Tables 4 and 5. Ablation studies in Table 7 are reproduced by the official codes [22, 53]. The
prediction performance of BeTopNet is compared against SOTA baselines by: 1) GNN-enabled
interactive graph [37, 24]; 2) conditional or game-theoretic behavioral interactions [31, 8]; 3) DETR-
based Transformer attentions [22, 25, 53, 86]; and 4) auto-regressive modeling [32].

20


---Page Break---
Table 9: Detailed nuPlan closed-loop simulation results in Val14 benchmark. BeTopNet highlights
leading results among SOTA methods in safety and compliance, outperforms learning-based agents.

Type
Method
Val14
CA ↑
TTC ↑
DDC ↑
DC ↑
EP ↑
Speed ↑
CLS-NR ↑
CLS-R ↑

Expert
Log Replay
0.987
0.944
0.981
0.993
0.989
0.965
0.937
0.812

Rule
IDM [70]
0.909
0.834
0.941
0.944
0.862
0.973
0.793
0.793
PDM-Closed [69]
0.981
0.933
0.998
0.955
0.921
0.998
0.932
0.930

Hybrid
GameFormer [8]
0.943
0.867
0.948
0.933
0.890
0.987
0.829
0.838

Learning

UrbanDriver [74]
0.856
0.803
0.908
1.000
0.808
0.915
0.677
0.648
PDM-Open [69]
0.745
0.691
0.879
0.995
0.698
0.977
0.502
0.548
PlanCNN [72]
0.869
0.814
0.850
0.814
0.806
0.980
0.669
0.646
GC-PGP [83]
0.858
0.801
0.897
0.900
0.603
0.993
0.611
0.549
PlanTF [73]
0.941
0.907
0.968
0.937
0.898
0.977
0.853
0.771
PLUTO [92]
0.961
0.933
0.985
0.964
0.895
0.981
0.890
0.800
BeTopNet (Ours)
0.966
0.916
0.995
0.932
0.866
0.971
0.883
0.837

Table 10: Detailed nuPlan closed-loop planning results (PDMScore) on Test14-Random bench-
mark.

Type
Method
Test14 Random
Col. Avoid ↑
Drivable ↑
Direction ↑
Progress ↑
TTC ↑
Comfort ↑
PDMScore ↑

Expert
Log Replay
0.996
0.962
0.996
0.664
0.985
1.000
0.832

Rule
PDM-Closed [69]
0.934
0.984
0.996
0.867
0.911
0.996
0.888

Learning

Constant Acc.
0.846
0.907
0.915
0.436
0.804
1.000
0.592
UrbanDriver [74]
0.965
0.961
0.986
0.611
0.957
1.000
0.788
PlanCNN [72]
0.935
0.938
0.971
0.591
0.888
0.989
0.736
PlanTF [73]
0.966
0.948
0.625
0.626
0.918
0.992
0.768
BeTopNet (Ours)
0.989
0.977
0.989
0.673
0.969
1.000
0.833

Table 11: Detailed nuPlan closed-loop planning results (PDMScore) on Test14-Hard benchmark.

Type
Method
Test14 Hard
Col. Avoid ↑
Drivable ↑
Direction ↑
Progress ↑
TTC ↑
Comfort ↑
PDMScore ↑

Expert
Log Replay
0.985
0.945
0.970
0.658
0.955
1.000
0.786

Rule
PDM-Closed [69]
0.933
0.952
0.976
0.779
0.852
0.981
0.811

Learning

Constant Acc.
0.845
0.871
0.861
0.415
0.800
1.000
0.552
UrbanDriver [74]
0.946
0.944
0.992
0.581
0.903
1.000
0.731
PlanCNN [72]
0.909
0.908
0.937
0.555
0.860
0.992
0.675
PlanTF [73]
0.984
0.961
0.996
0.649
0.961
0.996
0.813
BeTopNet (Ours)
0.968
0.945
0.972
0.747
0.908
0.996
0.813

D.3
Training Setup

BeTopNet for both prediction and planning tasks are trained in end-to-end manners by AdamW
optimizer with 4 NVIDIA A100 GPUs. The learning rate is configured as 1e−4 scheduled with the
multi-step reduction strategy. The planning model is trained by 25 epochs with a batch size of 128,
while the prediction task is trained with 30 epochs with a batch of 256.

E
Additional Quantitative Results

E.1
Planning

Additional planning results in Val14. We evaluate the closed-loop simulation performance under
Val14 in Table 9, BeTopNet hovers strong planning results and is comparable (+4.6% CLS) to con-
current learning-based methods [92] leveraging extra contrasting learning for training augmentations.
BeTopNet is also featured by leading driving safety (+2.7% CA, +1.0% TTC) and compliance
(+2.8% DDC) compared with other strong models [73, 8]. However, due to the data leakage of Val14
with training set by a part of shared scenarios, we only provide the results as a reference.

Additional planning effects in Test14. To delve into the planning results of BeTopNet, we present
statistics measuring by another detailed metric, PDMScore, for both of the Test14 benchmarks in
Table 2. Exhibited in Tables 10 and 11, BeTopNet delivers strong maneuver safety and compliance,
marking solid PDMScore from both benchmarks. Compared with learning-based methods, BeTopNet

21


---Page Break---
Table 12: Marginal predictions on WOMD Motion Leaderboard [81]. Primary metric.

Category
Method
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑
Soft mAP ↑

Vehicle

MTR [22]
0.7642
1.5257
0.1514
0.4494
0.4590
EDA [53]
0.6808
1.3921
0.1164
0.4833
0.4972
BeTopNet (Ours)
0.6814
1.3888
0.1172
0.4860
0.4995

Pedestrian

MTR [22]
0.3486
0.7270
0.0753
0.4331
0.4409
EDA [53]
0.3426
0.7080
0.0670
0.4680
0.4778
BeTopNet (Ours)
0.3451
0.7142
0.0668
0.4777
0.4875

Cyclist

MTR [22]
0.7022
1.4093
0.1786
0.3561
0.3650
EDA [53]
0.6920
1.4106
0.1673
0.3947
0.4037
BeTopNet (Ours)
0.6905
1.3975
0.1688
0.4060
0.4163

Table 13: Joint predictions on WOMD Interaction Leaderboard [82]. Primary metric.

Category
Method
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑
Soft mAP ↑

Vehicle

GameFormer [8]
1.0499
2.4044
0.4321
0.2469
0.2564
AMP [32]
0.9862
2.2286
0.3726
0.3104
0.3196
BeTopNet (Ours)
1.0216
2.3970
0.3738
0.3374
0.3308

Pedestrian

GameFormer [8]
0.7978
1.8195
0.4713
0.1962
0.2014
AMP [32]
0.6823
1.5244
0.3716
0.2359
0.2423
BeTopNet (Ours)
0.7862
1.8412
0.4074
0.2212
0.2267

Cyclist

GameFormer [8]
1.0686
2.4199
0.5765
0.1367
0.1338
AMP [32]
1.0533
2.3715
0.5194
0.1420
0.1477
BeTopNet (Ours)
1.1155
2.5850
0.5253
0.1717
0.1756

excels in closed-loop driving progress (+15.1%, +7.5% EP), safety (+5.6% TTC, +2.9% CA), and
the general score (+8.5% PDMScore). For rule-based systems, the leading performance is empirically
by virtue of a constant driving progress. This may refer to an unresolved copy-cat problem [73] for
imitative planners. It requires further integration and fallback with rule-based methods for on-board
AD system design in practice.

E.2
Prediction

Per-category marginal prediction. In Table 12, We mainfest the prediction performance of BeTop-
Net under each prediction category. Compared against the concurrent SOTA motion predictors [53],
BeTopNet demonstrates superior mAP-based metrics among all types for compliant predictions.
Specifically, overall improvements in Cyclist denote refined interactive patterns captured by BeTop,
as the cyclist predictions are the most uncertain task with less reliance on map information.

Per-category joint prediction. We further instantiate the per-category joint prediction of BeTopNet
with SOTA methods in Table 13. Compared with concurrent methods [32] featuring auto-regressive
decoding, BeTopNet achieves robust displacement metrics, while outperforming in prediction com-
pliance of mAP metrics (+8.7%, +20.9% mAP) due to advanced joint modality scoring stabilized
by edge topology in BeTopNet. Moreover, the coordinated joint behaviors reasoned by BeTopNet
largely mitigate the unstable patterns against game-theoretic method [8] (−15.6%, −15.7%, −9.7%
Miss Rate) under similar model architecture.

F
Additional Ablation Studies

Scaling effects of model and decoding agents. The scalability challenges begin with the scaling of
our BeTopNet models to accommodate varying scene agents and map. Experimentally, we configure
BeTopNet with different model scales to evaluate whether our approach maintains its effectiveness.

In Table 14, BeTopNet is evaluated by three model scales varying in decoding modalities and
dimensions. The results demonstrate that BeTopNet consistently improves prediction accuracy, with
an increase from 0.391 to 0.442 (+13.4% mAP) and a decrease in the Miss Rate (−11.9%). This
showcases its enhanced robustness in handling multi-agent settings by enlarging model scales.

In Table 15, BeTopNet reports comparable computational costs compared to [22], while with better
prediction accuracy shown in Table 12. The similar latency is due to the topo-guided attention,

22


---Page Break---
Table 14: Effects of varied model scale. BeTopNet
shows scalability with the number of decoding modal-
ities and feature dimensions.

Scale
mAP ↑
Miss Rate ↓
Latency (ms)
# Params. (M)

Small
0.391
0.131
45
28.91
Medium
0.437
0.119
65
28.91
Base
0.442
0.117
70
45.38

Table 15: Effects of varied decoding agents.
Computational costs of [22] are reported in the
parenthesis after ours.

# Decoding Agents
Latency (ms)
GPU Memory (G)

8
89 (84)
6.5 (5.2)
16
120 (123)
10.8 (7.1)
32
166 (193)
19.2 (15.6)

Table 16: Effects of varied temporal granularity in BeTop. Future interactions are split into various
intervals for multi-step BeTop labels. A fine-grained topology reasoning for the whole prediction
horizon results in a slightly improved performance and increased computational costs simultaneously.

Interval
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑
Inference
Latency (ms)
Training
Latency (ms)
# Params.
(M)

1(Base)
0.637
1.328
0.144
0.392
70.0
101.6
45.380
2
0.633
1.325
0.145
0.394
75.5
110.6
45.382
4
0.634
1.326
0.142
0.391
80.0
133.4
45.386
8
0.641
1.347
0.147
0.389
90.0
255.0
45.393

Table 17: Effects of varied model foundation by Wayformer [62]. Synergistic decoder design by
BeTopNet demonstrate solid multi-agent interaction understanding compared with vanilla design.

Method
minADE ↓
minFDE ↓
Miss Rate ↓
mAP ↑

Wayformer
0.661
1.417
0.199
0.281
Wayformer+BeTop
0.637
1.364
0.178
0.290
Wayformer+BeTopNet
0.604
1.261
0.166
0.344

which reduces the KV features in agent aggregation during decoding. While BeTop introduces extra
computations for reasoning, it requires more GPU memory for cached topology tensors.

Temporal granularity in BeTop. Table 16 explores the effect of varied temporal granularity in
BeTop, with minimal adjustments to BeTopNet. In our study, future interactions are split into multi-
step BeTop labels. Topology reasoning task is then deployed through expanded MLP Topo Head
for output steps. Compared to the baseline 1-step reasoning, multi-step BeTop reasoning slightly
improves performance (e.g., 2-steps, +0.2 mAP), with a corresponding increase in computational costs
for additional steps. This highlights the potential of multi-step reasoning to enhance BeTopNet in
interactive scenarios, while refining temporal granularity for more accurate and efficient interactions
remains an open question. We believe how to effectively leverage multi-step BeTopNet represents an
interesting area for future exploration.

Synergy with additional model foundation. To understand the generalization under different model
foundations, we conduct additional ablations integrating BeTop with reproduced Wayformer [62]
in [95]. As reported in Table 17, incorporating BeTop as supervision improves vanilla Wayformer
with a −6.2% Miss Rate and +3.2% mAP. Furthermore, integrating BeTopNet significantly boosts
performance, achieving a +18.6% mAP and −7.2% Miss Rate. This enhancement is due to syn-
ergistic decoder design, which uses iterative BeTop reasoning and Topo-guided attention to refine
trajectories by selectively aggregating interactive features.

G
Additional Qualitative Results

Additional planning results. We provide the qualitative closed-loop simulations for all of the
benchmarks in Test14, as shown in Figs. 7 and 8.

Additional prediction results. We provide the qualitative prediction results for BeTop with reasoned
edge topology under both marginal (Fig. 10) and joint (Fig. 9) challenge settings.

H
License of Assets

Data for nuPlan [80] and WOMD [35] are complied with CC-BY-NC 4.0 licence and Apache License
2.0; The code for re-implementations are under Apache License 2.0 for PDM [69] and MTR [22], and

23


---Page Break---
Test14-Hard
Test14-Random

Figure 7: Qualitative results of BeTopNet in nuPlan planning under Test14 simulations. Each
row of the figures render closed-loop simulations at 1s, 8s, and 15s temporal frames. As illustrated,
BeTopNet performs consistent planning under challenging driving scenarios of diverse categories.

MIT License for EDA [53] and PlanTF [73], respectively. The source code and our trained models
will be publicly available under the Apache License 2.0.

24


---Page Break---
t=0s
t=8s
p=0
p=1

Joint Probability
Prediction

t=0s
t=8s

Planning

a)

c)

b)

d)

Figure 8: Qualitative results of BeTopNet in nuPlan planning under Test14-Inter. BeTopNet
performs compliant planning under: (a) yielding to front agents; (b) cruising on various road structure;
(c-d) interactive behaviors among two or more agents with dense traffic.

Vehicle to vehicle (V2V)
Vehicle to cyclist (V2C)
Vehicle to pedestrian (V2P)

t=0s
t=8s
e=0
e=1
p=0
p=1
Joint Probability
Prediction
Edge Topology 

Figure 9: Qualitative results of BeTopNet in WOMD joint prediction. Joint predictions among
heterogeneous agents are categorized by each column (V2V, V2C, and V2P) with corresponding TopK
reasoned topology. As depicted, BeTopNet can accurately capture the future interactive behaviors via
edge topology reasoning compared with the human annotations of interactive agents (rendered in
red). Moreover, BeTopNet may source on potential interactions as rendered in grey.

25


---Page Break---
t=0s
t=8s
e=0
e=1
p=0
p=1

Marginal Probability
Prediction
Edge Topology 

Figure 10: Qualitative results of BeTopNet in WOMD marginal prediction. BeTopNet performs
compliant and accurate marginal predictions on multiple agents, reasoning diverse edge topology
which stabilizes the behavioral patterns for future interactions.

26


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Clear introduction with contributions and scopes in Sec. 1. We provide extra
insightful Q&A in Appendix A to further position our scope.
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
Justification: We discuss limitations in Sec. 5.
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

27


---Page Break---
Justification: We provide comprehensive formulations in Sec. 3.1. We also provide full
analytical proof in Appendix B, as well as empirical verification in Fig. 1 apart from
experiments.
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
Justification: Main experimental results of Sec. 4.1 are listed directly from the official
leaderboards. We provide the methodology for formulation, model structure and optimiza-
tion process in Sec. 3. The corresponding details for implementations and experiments are
expressly illustrated in Appendices C and D.
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

28


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: All data, baselines and results are already publicly available [80, 35]. Code
will be available.

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

Justification: See details in Sec. 4. We also provide some settings in Appendix D and will
release the code.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [No]

Justification: All the experiments are tested by weighted mean metrics for official compar-
isons.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

29


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
Justification: We provide compute resource details and related information in Appendix D.
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
Justification: The research conforms the Code of Ethics in all aspects.
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
Justification: We have discussed broader impacts in Q3 in Appendix A.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

30


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

Justification: This paper poses no such risks.

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

Justification: We cite the papers for the used datasets and models in the paper, and list
corresponding licenses in Appendix H.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

31


---Page Break---
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
Justification: We will release new assets including code and models. Details are in Ap-
pendix H.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

32


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

33


---Page Break---
