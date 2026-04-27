Agent-to-Sim: Learning Interactive Behavior from
Casual Videos

Anonymous Author(s)
Affiliation
Address
email

Abstract

Agent behavior simulation empowers robotics, gaming, movies, and VR appli-
1

cations, but building such simulators often requires laborious effort of manually
2

crafting the agent’s decision process and motion patterns. Recent advances in
3

visual tracking and motion capture have enabled learning agent behavior from
4

real-world data, but these methods are limited to a few scenarios due to the de-
5

pendence on specialized sensors (e.g., synchronized multi-camera systems). In a
6

step towards scalable and realistic behavior simulators, we present Agent-to-Sim
7

(ATS), a framework for learning simulatable 3D agents in a 3D environment from
8

casually-captured monocular videos. To deal with partial views, our framework
9

fuses observations in a canonical space for both the agent and the scene, resulting
10

in a dense 4D spatiotemporal reconstruction. We then learn an interactive behavior
11

generator by querying paired data of agents’ perception and actions from the 4D
12

reconstruction. ATS enables real-to-sim transfer of agents in their familiar envi-
13

ronments given longitudinal video recordings captured with a smartphone over a
14

month. We show results on pets (e.g., cat, dog, bunny) and a person, and analyse
15

how the observer’s motion and 3D scene affect an agent’s behavior.
16

1
Introduction
17

Plausible paths

Past Tajectory
Consider the scene of the cat in the living room: where will the cat go
18

and how will it move? Since we have seen cats interact with the en-
19

vironment and other people many times, we know that cats like to go
20

to the couch, often move slowly, and follow humans around, but run
21

away if people come too close. Such a predictive model of a phys-
22

ical agent is what enables plausible behavior simulation, which is
23

essential for embodied intelligence, immersive virtual environments
24

and robot planning in safety-critical scenarios [9, 31, 41, 45, 54].
25

The key challenge with behavior simulation is how to generate plausible and interactive behavior
26

(with respect to the scene and other agents). On one hand, prior works [2, 6, 46] utilize trajectory
27

computed by path-planning algorithms or hand-designed logic from game simulators [13, 58]. While
28

these approaches benefit from high-quality trajectory data paired with perfect object and scene
29

geometries, it is laborious to manually craft simulators that suit the needs of each type of application,
30

and the data distribution is fundamentally different from the real world, leading to unnatural motion
31

and interactions. On the other hand, vision-based motion capture enables learning plausible behavior
32

directly from data for certain scenarios, such as autonomous driving [9], human body motion [21, 36],
33

and interaction with objects/scenes [14, 24]. However, due to the dependence on specialized sensor
34

(synchronized multi-camera systems, IMUs, pre-scanned objects), such systems does not scale well
35

to the full spectrum of natural behavior one may care about, such as behavior of animals, casual
36

events, and long-term activities.
37

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Observer

Agent

Scene

Shot 1

Time

A) 4D Spacetime Reconstruction

…

Shot M

What happens if  “z”?

B) Interactive Behavior Simulator

User: “move sideway”
User: “idle”
User: “approach quickly”

Figure 1: Learning agent behavior from longitudinal casual video recordings. We answer the
following question: can we simulate the behavior of an agent, by learning from casually-captured
videos of the same agent recorded across a long period of time (e.g., a month)? A) We first reconstruct
videos in 4D (3D & time), which includes the scene, the trajectory of the agent, and the trajectory of
the observer (i.e., camera held by observer). Such individual 4D reconstruction are registered across
time, resulting in a complete 4D reconstructions. B) Then we learn a representation of the agent that
allows for interactive behavior simulation. The behavior model explicitly reasons about goals, paths,
and full body movements conditioned on the agent’s ego-perception and past trajectory. Such agent
representation allows us to simulate novel scenarios through conditioning. For example, conditioned
different observer trajectories, the cat agent choose to walk to the carpet, stays still while quivering
his tail, or hide under the tray stand. Please see videos and results of other agents in the supplement.

Recent advances in differentiable rendering [10, 12, 23, 38, 42, 52, 59, 65] and monocular MoCap [28,
38

43, 69, 70] provide a pathway to obtain high-quality models of scenes and agents from monocular
39

videos alone. Despite the potential of covering diverse data of agent behavior that match the real-
40

world distributions, none of the existing works brings a solution of reconstructing dense 3D structures
41

of both the agent and scene, which is crucial for learning agent behavior grounded in real world
42

environments. To address this, we present ATS (Agent-to-Sim), a framework for learning simulatable
43

agent from casual videos captured over a long time horizon (e.g. 1 month), as shown in Fig. 1.
44

The crucial technical challenge is the presence of partial visibility – in each video captured from
45

an observer’s viewpoint, only parts of the agent and the environment are visible. How do we infer
46

the states of agent and the environment that are not visible? To build a dense 4D spatiotemporal
47

reconstruction, our key insight is to leverage the observations from multiple videos by fusing them
48

in a canonical 3D space. We introduce a novel coarse-to-fine registration approach that re-purposes
49

“foundational” visual features [40] as a neural localizer, which “registers” the camera with respect
50

to a canonical structure. This enables capturing interactive behavior data in a casual setup (e.g.,
51

with a smartphone), and provides paired training data of perception and action of an agent that is
52

grounded in a natural environment (Fig. 2). To learn an interactive behavior model, we condition the
53

action of an agent on their ego-perception, and leverage diffusion models [18, 53] to account for the
54

multimodal nature of goals and planned trajectories. The resulting framework, ATS, can simulate
55

interactive behaviors like those described at the start: agents like pets that leap onto furniture, dart
56

quickly across the room, timidly approach nearby users, and run away if approached too quickly. Our
57

contributions are summerized as follows:
58

1. Agent-to-Sim (ATS) Framework. We introduce a real-to-sim framework, ATS, to learn
59

simulators of interactive agent behavior from casually-captured videos. ATS learns plausible
60

agent behavior that matches the real-world, and is scalable to diverse scenarios, such as
61

animal behavior and casual events.
62

2. Environment-Interactive Behavior Simulation. ATS learns behavior that is interactive
63

to the environment, including both the observer and 3D scene. We show the first result
64

of generating plausible behavior of animals that are reactive to observer’s motion, and are
65

aware of the 3D scene.
66

2


---Page Break---
Table 1: Related works in behavior data capture. ATS is the only method that builds a complete
4D reconstruction of both the agents and the environment. Different from prior work that focus on
specific domains, ATS can be applied to capture interactive behavior of both animals and humans
from casual RGBD videos (e.g. captured by a smartphone).

Method
Agent Model
Scene Model
Capture Setup
Domain

UCY [30] & ETH [44]
Point
N.A.
Manual Anno.
Pedestrian
nuScenes [9]
Point
Dense 3D Map
Manual Anno.
Pedestrian, Vehicle
SAMP [14]
Parametric Body
Furniture & Objects
Multi-Camera
Human
AMASS [36]
Parametric Body
N.A.
Multi-Camera
Human
ActionMap [47]
Action Class
Sparse 3D Map
Egocentric Camera
Human
ATS (Ours)
Non-parametric
Dense 3D Map
Casual RGBD
Animal, Human

3. Complete 4D Registration & Reconstruction. We present a method to register and
67

reconstruct a temporally-evolving 3D scene, whiling accounts for changes in scene layout
68

and appearance.
69

2
Related Works
70

Behavior Prediction and Generation. Behavior prediction has a long history, starting from simple
71

physics-based models such as social forces [17] to more sophisticated “planning-based” models that
72

cast prediction as reward optimization [26, 76], where the reward is learned via inverse reinforcement
73

learning [75]. With the advent of large-scale pedestrian and vehicle motion data collected in the
74

navigation and autonomous driving domains [1, 34, 37, 48, 50], generative prediction models such as
75

diffusion models have been able to express behavior multi-modality while being easily controlled via
76

additional signals such as cost functions [20] or logical formulae [74]. However, to capture plausible
77

behavior of agents, these approaches are extremely dependant on high-quality agent trajectory data
78

collected “in the wild” with the associated scene context (e.g., 3D map of the scene) [9]. Such data are
79

often manually annotated at a bounding box level (Tab. 1), which limits the scale and the level of detail
80

they can capture. Beyond autonomous driving setup, existing works for human motion prediction and
81

generation [46, 57, 62] have been primarily using simulated data [6] or motion capture data collected
82

with multiple synchronized cameras [14, 24, 36]. Such data provide high-quality full body motion
83

of human using parametric body models [32], but the interactions with the environment are often
84

restricted to a set of pre-defined furnitures and objects [15, 29, 73]. Furthermore, the use of simulated
85

data and motion capture data inherently limits the realism of these behavior generators, since real
86

agents will behave very differently in their familiar environment. To bridge the gap, we develop
87

4D reconstruction method to obtain high-quality trajectories of agents in their natural environment,
88

with a simple setup that can be achieved with a smartphone. Close to our setup, ActionMap [47]
89

associate daily actions performed by a human agent with an reconstructed 3D environment given
90

egocentric videos. However, they focus on actions performed by hand and do not reconstruct the full
91

body motion of the agent.
92

4D Reconstruction from Monocular Videos. Reconstructing agents and the environment from
93

monocular videos is challenging due to its under-constrained nature. Given a monocular video,
94

there are multiple different interpretations of the underlying 3D geometry, motion, appearance,
95

and lighting [56]. As such, reconstructing agents often require category-specific 3D prior (e.g., 3D
96

humans) [11, 27, 32]. Along this line of work, researchers reconstruct 3D humans aligned to the world
97

coordinate with the help of SLAM and visual odometry [28, 69, 70]. Sitcoms3D [43] reconstructs
98

both the scene and human parameters, while relying on shot changes to determine the scale of the
99

scene. However, the use of parametric body models limits the degrees of freedom they can capture,
100

and makes it difficult to reconstruct agents from arbitrary categories which do not have a pre-built
101

body model, for example, animals. Another line of work avoids using category-specific 3D priors and
102

optimizes the shape and deformation parameters of the agent given richer visual signals (e.g., optical
103

flow and object silhouette) [61, 64, 65], which is shown to work well for a broad range of category
104

including human, animals, and vehicles. TotalRecon [52] further incorporates the background scene
105

into the model-free reconstruction pipeline, such that the agent’s motion can be decoupled from the
106

camera motion and aligned to the scene space. However, none of the existing methods can reconstruct
107

both the agent and the scene in high-quality. In practice, individual videos may not contain sufficient
108

3


---Page Break---
views, leading to inaccurate and incomplete reconstructions. Our method registers both the agent and
109

the environment from multiple videos into a shared space, which leverages large-scale data collection
110

to build a high-quality agent and scene model.
111

3
Approach
112

We describe a method to learn interactive behavior models given longitudinal video recordings of an
113

agent in the same environment. We first build a spatiotemporal 4D reconstruction, including the agent,
114

the scene, and the observer (Sec. 3.1), which is solved by an optimization involving multi-video
115

registration (Sec. 3.2). We then train an interactive behavior model of the agent that is interactive
116

with the surrounding environment, including the scene and the motion of the observer (Sec. 3.3).
117

3.1
4D Representation: Agent, Scene, and Observer
118

Given multiple monocular videos, our goal is to build a dense spatiotemporal 4D reconstruction of
119

the underlying world, including a deformable agent, a background scene, and a moving observer.
120

The task is ill-posed due to partial visibility – from an observer’s viewpoint, the agent and the
121

environment are only partially visible. To deal with this problem, one principle approach is geometric
122

registration, where structures not visible from one view can be inferred from the other views they
123

appear [51]. We build upon this idea to reconstruct a complete spatiotemporal model of an agent and
124

their familiar environment by registering videos captured at different time.
125

Problem Setup. Specifically, given images from M videos represented by color and feature descrip-
126

tors [40], {Ii, ψi}i={1,...,M}, our goal is to find a 4D spatiotemporal representation that explains the
127

video, while pixels with the same semantics can be mapped to consistent canonical 3D locations. Our
128

representation factorizes the 4D structure into a static component and a time-varying component.
129

Static Representation. T = {σ, c, ψ}. We represent the static component as agent fields and scene
130

fields. Both define densities, colors, and semantic features in a canonical space,
131

(σs, cs, ψs) = MLPscene(X, βi),
(1)

132

(σa, ca, ψa) = MLPagent(X),
(2)

where X corresponds to a 3D point. To account for structures that change across videos, we modify
133

the scene fields to take a per-video latent code βi as input, which allows fitting video-specific details.
134

Time-varying Representation. D = {ξ, G, W}. The time-varying component includes a moving
135

observer, represented by the camera pose ξt ∈SE(3), and the motion of an agent, represented by a
136

set of rigid bodies, {Gb
t}{b=1,...,25}, referred to as “bones”. Given a time t, the canonical space of
137

the agent can be mapped to the camera space by blend-skinning deformation [35, 65],
138

Xt = GaX =

 B
X

b=1
WbGb
t

!

X,
(3)

which computes the motion of a point by blending the bone transformations (we do so in the dual
139

quaternion space [22, 66] to ensure Ga is a valid rigid transformation). The skinning weights W are
140

defined as the probability of a point assigned to each bone.
141

Rendering. To turn the 4D representation into images, we sample rays in the camera space, map
142

them separately to the canonical space of the scene and the agent with D, and query values (e.g.,
143

density, color, feature) from corresponding fields of the scene and the agent. The values are then
144

combined before ray integration [39, 52]. Consequently, the rendered pixel values are compared
145

against the observations to update the world representation {T, D}.
146

Decoupling Agent Motion from Observer. {Gb
t}{b=1,...,25} defines the motion of an agent with
147

respect to the observer. Given the observer, we compute the motion of the agent in the scene space as,
148

Gb→s
t
= ξ−1
t Gb
t,
(4)

where the results of extracted trajectories of the agent is shown in Fig. 2
149

4


---Page Break---
Shot 1

Shot 2

Shot 3

Shot 4

Shot 5

Registered 4D 
Reconstruction

…

Figure 2: Results of 4D reconstruction. Top: reference images and renderings of the reconstructions.
The color on the background represents correspondence. The colored blobs on the agent body
represent B = 25 body parts of the agent (e.g., head is represented by the yellow blob). Bottom:
Bird’s eye view of the reconstructed scene and agent trajectories, registered to the same scene
coordinate. Each colored line represents a unique video sequence where boxes and spheres indicate
the starting and the end location. Please see videos and results on other agents in the supplement.

3.2
Optimization: Multi-Video Registration
150

To deal with bad local optima caused by camera poses (Fig. 4), we design a coarse-to-fine registration
151

approach that globally aligns the cameras to a shared canonical space with a feed-forward network,
152

and then jointly optimizes the 3D structures while adjusting the cameras locally.
153

Initialization: Neural Localization. Due to the evolving nature of scenes across a long period
154

of time [55], there exist both global layout changes (e.g., furniture get rearranged) and appearance
155

changes (e.g., table cloth gets replaced), making it challenging to find accurate geometric corre-
156

spondences [4, 5, 49]. With the observation that “foundational” visual features have good 3D and
157

viewpoint awareness [3], we adapt them for camera localization. We learn a scene-specific neural
158

5


---Page Break---
localizer that directly regresses the camera pose of an image with respect to a canonical structure,
159

ξ = fθ(ψ),
(5)

where fθ is a ResNet-18 [16] and ψ is the DINOv2 [40] feature of the input image. We find it to
160

be more robust than geometric correspondence, while being more computationally efficient than
161

performing pairwise matches [49]. To learn the neural localizer, we first capture a walk-through video
162

and build a dense map of the scene. Then we use it to train the neural localizer by randomly sampling
163

camera poses G∗= (R∗, t∗) and rendering images on the fly,
164

arg min
θ

X

j

 
∥log(RT
0 (θ)R∗)∥+ ∥t0(θ) −t∗∥2
2

,
(6)

where we use geodesic distance [19] for camera rotation and L2 error for camera translation. For the
165

agent, we follow BANMo [65] to initialize the root pose {Gb}b=0 with a pre-trained pose network.
166

Objective: Feature-metric Alignemnt. Given a coarse initialization of the observer (scene camera)
167

and the agent’s root pose, we use both photometric and featuremetric losses to optimize {T, D},
168

min
T,D

X

t

 
∥It −RI(t; T, D)∥2
2 + ∥ψt −Rψ(t; T, D)∥2
2

+ Lreg(T, D),
(7)

where R(·) is the rendering function described in Sec 3.1. In contrast to prior works, using feature-
169

metric errors makes the optimization robust to change of lighting, appearance, and helps find accurate
170

alignment over multiple videos (Fig. 4). The regularization term includes eikonal loss, silhouette loss,
171

flow loss and depth loss similar to prior works [52, 65].
172

Scene Annealing. To encourage the reconstructed scene across videos to share a similar structure, we
173

randomly swap the code β of two videos during optimization, and gradually decrease the probability
174

of swaps from P = 1.0 →0.05 over the course of optimization. This regularizes the model to
175

effectively share information across all videos, and keeps video-specific details (Fig. 4).
176

3.3
Interactive Behavior Generation
177

Now that we build a complete 4D reconstruction from multiple videos, we can extract a scene structure
178

T, and M trajectories of the agent {Gt}t={T1,...,TM} as well as the observer {ξt}t={T1,...,TM}
179

grounded in the environment. We aim to learn an agent that is interactive with the world.
180

Hierarchical Behavior Representation. We model the behavior of an agent by bone transformations
181

in the scene space G ∈R6B×T ∗over a fixed time horizon T ∗= 5.6s, . We design a hierarchical
182

model as shown in Fig. 3. The body motion G is conditioned on path P ∈R3×T ∗, which is further
183

conditioned on goal Z ∈R3. Such decomposition allows agents to react by predicting goals with low
184

latency
185

Goal Generation. We represent a multi-modal distribution of goals Z ∈R3 by its score function
186

s(Z, σ) ∈R3 [18, 53]. The score function is implemented as a coordinate MLP [38],
187

s(Z; σ) = MLPθZ(Z, σ),
(8)

trained by predicting the amount of noise ϵ added to the clean goal, given the corrupted goal Z + ϵ:
188

arg min
θZ
EZEσ∼q(σ)Eϵ∼N(0,σ2I) ∥MLPθZ(Z + ϵ; σ) −ϵ∥2
2 .
(9)

Compared to methods directly learning the multi-modal distribution [8, 25], diffusion models are
189

easy to train and can be used to generate diverse and high-quality samples [18, 53].
190

Path Generation with Control. To guide path generation with goals, we represent its score as
191

s(P; σ) = ControlUNetθP(P, Z, σ),
(10)

where the Control UNet contains two standard UNets with the same architecture [72], one performing
192

unconditional generation taking (P, σ) as input, another injecting goal conditions densely into the
193

neural network blocks of the first one taking (Z, σ) as inputs. Compared to concatenating the goal
194

condition to the noise latent, this encourages close alignment between the goal and the path [62]. We
195

apply the same architecture to control pose generation with paths,
196

s(G; σ) = ControlUNetθG(G, P, σ).
(11)

6


---Page Break---
Score map

Past trajectory

Observer

Sampled goals
Sampled path

Past body motion

Sampled body 

motion
ωo
ωp
ωs

Encoding: Ego-perception
Decoding: Behavior Generation

Observer
Past
Scene

Perception Code ω ∈ℝ192

World-to-Ego Transform (Eq. 12)

 (2ms / denoising step)
 (9ms / denoising step)
 (9ms / denoising step)

Goal Z ∈ℝ3
Path P ∈ℝ3×T*
Body motion G ∈ℝ6B×T*
ω

Figure 3: Pipeline for behavior generation. We first encode egocentric information into a perception
code ω and then generate full body motion in a hierarchical fashion. We start by generating goals Z
with low latency, and then generate a path P and body motion G conditioned on the previous node.
Each node is represented by the gradient of its log distribution, trained with the denoising objectives
(Eq. 9). Given G, the dense deformation of an agent can be computed via blend skinning (Eq. 3).

Compared to concatenation, we observe better alignment between the path and the full body pose
197

using the Control Unet.
198

.
199

Ego-Perception Encoding. To generate plausible interactive behaviors, we encode the world
200

egocentrically perceived by the agent, and use it to condition the behavior generation. We use the
201

reconstructed environment T and the observer ξ as a proxy of the world, and transform them to the
202

egocentric coordinate of the agent,
203

ξs→a = G−1
b=0ξ,
Ts→a = G−1
b=0T
(12)

Transforming the world to the egocentric coordinates avoids over-fitting to specific locations of the
204

scene (Tab. 2). To encode ego-perception of the scene, we querying feature values from ψs with a 3D
205

grid around the agent and extract a latent scene representation,
206

ωs = ResNet3Dθψ(ψs).
(13)

where ResNet3Dθϕ is a 3D ConvNet with residual connections, and ωs ∈R64 represents the scene
207

perceived by the agent. We encode the observer’s motion in the past T ′ = 0.8s seconds with
208

ωo = MLPθo(ξs→a),
(14)

where ωo ∈R64 represents the observer perceived by the agent. Accounting for the external factors
209

from the “world” enables interactive behavior generation, where the motion of an agent follows the
210

environment constraints and is influenced by the trajectory of the observer (Fig. 5).
211

History Encoding. We additionally encode the past motion of the agent in T ′ seconds,
212

ωp = MLPθp(Gs→a
b=0 ).
(15)

By conditioning on the past motion, we can generate long sequences by chaining individual ones.
213

4
Experiments
214

Dataset. We collect the a dataset that emphasizes the casual interactions of an agent with their
215

familiar environment and the observer. It contains iPhone-captured RGBD video collections of 4
216

types of agents, including 26 videos of a cat, 3 videos of a dog, 2 videos of a bunny, and 2 videos of a
217

human. The time span of the video capture ranges from 1 day to a month, and each video contains 30
218

seconds to 2 minutes of content. The dataset is curated to contain diverse motion of agents, including
219

walking, lying down, eating, as well as diverse interaction patterns with the environment, including
220

following the camera, sitting on a coach, etc. Please refer to the supplement for more details.
221

4.1
4D Reconstruction of Agent & Scene
222

Implementation Details. We extract frames from the videos at 10 FPS, and use off-the-shelf models
223

to produce augmented image measurements, including object segmentation [68], optical flow [63],
224

7


---Page Break---
Our Method
TotalRecon (Multi-video)
W/o NL
W/o FBA
W/o Annealing

Figure 4: Comparison on multi-video scene reconstruction. We show a top-down visualization
of the reconstructed scene using the bunny dataset. Compared to TotalRecon that does not register
multiple videos, ATS produces higher-quality scene reconstruction. Neural localizer and featuremetric
losses are shown important for camera registration. Scene annealing is important for reconstructing
high-quality scenes from limited views in a video.

DINOv2 features [40]. We use AdamW to first optimize the environment with featuremetric loss for
225

30k iterations, and then jointly optimize the environment and agent for another 30k iterations with a
226

combination of optical flow, silouette, and featuremetric losses. Optimization takes roughly 24 hours.
227

8 A100 GPUs used to optimize 26 videos (for the cat data), and 1 A100 GPU is used in a 2-3 video
228

setup (for dog, bunny, and human data).
229

Results. We run 4D reconstruction on all video sequences and report the results qualitatively. A visual
230

comparison on scene registration is shown in Fig. 2. Without the ability to register multiple videos,
231

TotalRecon produces protruded and misaligned structures (as pointed by the red arrow). In contrast,
232

our method reconstructs a single coherent scene. With featuremetric alignment (FBA) alone but
233

without a good camera initialization from neural localization (NL), our method produces inaccurate
234

reconstruction due to global misalignment in cameras poses. Removing FBA while keeping NL,
235

the method fails to accurately localize the cameras and produces noisy scene structures. Finally,
236

removing scene annealing procures lower quality scene structures due to lack of training views. A
237

visual comparison with TotalRecon (Single Video) is shown in Fig. 8, where we show that multiple
238

videos helps reconstructing a higher-quality agent, and a more complete scene.
239

4.2
Interactive Behavior Prediction
240

Dataset. We use the cat dataset for quantitative evaluation, where the data are split into a training set
241

of 22 videos and a validation set of 4 videos. The validation set is representative of three dominant
242

motion patterns of the agent: (1) trying to engage with the observer, (2) exploring the space and (3)
243

performing activities while not paying attention to the observer.
244

Implementation Details. To train the behavior model, we slice the reconstructed trajectory in
245

the training set into overlapping window of 6.4s, resulting in 12k data samples. We use AdamW
246

to optimize the parameters of the scores functions {θZ, θP, θG} and the ego-perception encoders
247

{θψ, θo, θp} for 120k steps with batch size 1024. Training takes 10 hours on a single A100 GPU.
248

Metrics. The behavior of an agent can be evaluated along multiple axes, and we focus on goal, path,
249

and body motion prediction. For goal prediction, we use a combination of displacement error (DE)
250

and minimum displacement error (minDE) [7]. The evaluation asks the model to produce K=64
251

samples. DE computes the avarage distance of the samples to the ground-truth, and minDE finds the
252

one closest to the ground-truth to compute the distance. For path and body motion prediction, we
253

use average displacement error (ADE) and minimum average displacement error (minADE), which
254

are similar to goal prediction, but additionally averages the distance over path and joint locations
255

before taking the min. When evaluating path prediction and body motion prediction, the output is
256

conditioned on the ground-truth goal and path respectively.
257

Comparisons. We re-purpose related methods and adapt them to our new setup of interactive
258

behavior prediction of animal agents. The quantitative results are shown in Tab. 2. To predict the goal
259

of an agent, classic methods build statistical models of how likely an agent visits a spatial location of
260

the scene, referred to as location prior [26, 76]. Given the extracted 3D trajectories of an agent in the
261

egocentric coordinate, we build a 3D preference map over 3D locations as a histogram, which can
262

be turned into probabilities and used to sample goals. Since this method does not take into account
263

8


---Page Break---
Table 2: Evaluation of interactive behavior prediction. We separately evaluate goal, path, and full
body motion prediction. Metrics are displacement errors (DE) in meters and the lower the better.
FaF [33] is re-purposed and re-trained with our data.

Method
Goal: minDE
Goal: DE
Path: minADE
Path: ADE
Body: minADE
Body: ADE

Location prior [76]
0.575
2.134
N.A.
N.A.
N.A.
N.A.
FaF [33]
N.A.
1.200
N.A.
0.057
N.A.
0.265
ATS (Ours)
0.395
1.299
0.006
0.007
0.226
0.234

w/o observer ωo
0.525
1.586
0.006
0.007
0.225
0.234
w/o scene ωs
0.702
1.058
0.006
0.007
0.225
0.234
w/o egocentric
0.639
1.424
0.025
0.034
0.212
0.222

{User, Past, Environment}
{Past, Environment}
{Environment}
Unconditional

Infeasible region 

(e.g., gap; 
underground)

User trajectory

Past trajectory

Sampled goals

Frontal view

Bird’s eye view

Figure 5: Analysis of conditioning signals. We show results of removing one conditioning signal
at a time. Removing observer conditioning and past trajectory conditioning makes the sampled
goals more spread out (e.g., regions both in front of the agent and behind the agent); removing the
environment conditioning introduces infeasible goals that penetrate the ground and the walls.

of the scene and the observer, it fails to accurately predict the goal. We then re-purpose FaF [33]
264

(Fast-and-Furious), a data-driven approach for motion forecasting to our task. FaF takes the same
265

input as ATS but regresses the goal, path, and body poses. It produces worse results than ATS for
266

all metrics since directly regressing the target treats the underlying distribution as a unit-variance
267

Gaussian and fails to account for the multi-modal nature of agent behaviors.
268

Analysing Interactions. We analyse the agent’s interactions with the environment and the observer
269

by removing the conditioning signals and study their influence on behavior prediction. In Fig. 5, we
270

show that by gradually removing conditional signals, the generated goal samples become more spread
271

out. In Tab. 2, we drop one of the conditioning signals at a time. Dropping the observer conditioning
272

increases the error in goal prediction, indicating observer’s trajectory is helpful goal prediction.
273

Dropping the environment conditioning produces worse results on goal prediction (minDE: 0.395 vs
274

0.702) as well. Surprisingly, it does not affect path prediction. We posit that the scenarios in the test
275

set are too simple. Conditioned on ground-turth goals, it performs well even without environment
276

conditioning. Finally learning behavior generation in the world coordinates performs worse for all
277

metrics since it over-fits to specific locations in the scene.
278

5
Conclusion
279

We have presented a framework for learning interactive behavior of agents grounded in natural
280

environments. To achieve this, we turn multiple casually-captured video recordings into complete 4D
281

reconstructions including the agent, the environment, and the observer. Such data collected over a
282

long time period allows us to learn a behavior model of the agent that is reactive to the observer and
283

respects the environment constraints. We validate our design choices on casual video collections, and
284

show better results than prior work for 4D reconstruction and interactive behavior prediction.
285

9


---Page Break---
References
286

[1] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese. Social lstm:
287

Human trajectory prediction in crowded spaces. In Proceedings of the IEEE conference on
288

computer vision and pattern recognition, pages 961–971, 2016.
289

[2] A. Bajcsy, A. Loquercio, A. Kumar, and J. Malik. Learning vision-based pursuit-evasion robot
290

policies. arXiv preprint arXiv:2308.16185, 2023.
291

[3] M. E. Banani, A. Raj, K.-K. Maninis, A. Kar, Y. Li, M. Rubinstein, D. Sun, L. Guibas,
292

J. Johnson, and V. Jampani. Probing the 3d awareness of visual foundation models. arXiv
293

preprint arXiv:2404.08636, 2024.
294

[4] E. Brachmann and C. Rother. Neural- Guided RANSAC: Learning where to sample model
295

hypotheses. In ICCV, 2019.
296

[5] E. Brachmann, T. Cavallari, and V. A. Prisacariu. Accelerated coordinate encoding: Learning to
297

relocalize in minutes using rgb and poses. In CVPR, 2023.
298

[6] Z. Cao, H. Gao, K. Mangalam, Q.-Z. Cai, M. Vo, and J. Malik. Long-term human motion
299

prediction with scene context. In Computer Vision–ECCV 2020: 16th European Conference,
300

Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 387–404. Springer, 2020.
301

[7] Y. Chai, B. Sapp, M. Bansal, and D. Anguelov. Multipath: Multiple probabilistic anchor
302

trajectory hypotheses for behavior prediction. arXiv preprint arXiv:1910.05449, 2019.
303

[8] L. Dinh, D. Krueger, and Y. Bengio. Nice: Non-linear independent components estimation.
304

arXiv preprint arXiv:1410.8516, 2014.
305

[9] S. Ettinger, S. Cheng, B. Caine, C. Liu, H. Zhao, S. Pradhan, Y. Chai, B. Sapp, C. R. Qi, Y. Zhou,
306

et al. Large scale interactive motion forecasting for autonomous driving: The waymo open
307

motion dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
308

pages 9710–9719, 2021.
309

[10] H. Gao, R. Li, S. Tulsiani, B. Russell, and A. Kanazawa. Monocular dynamic view synthesis:
310

A reality check. Advances in Neural Information Processing Systems, 35:33768–33780, 2022.
311

[11] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa*, and J. Malik*. Humans in 4D: Recon-
312

structing and tracking humans with transformers. In ICCV, 2023.
313

[12] C. Guo, T. Jiang, X. Chen, J. Song, and O. Hilliges. Vid2Avatar: 3D Avatar Reconstruction
314

from Videos in the Wild via Self-supervised Scene Decomposition. CVPR, 2023.
315

[13] P. E. Hart, N. J. Nilsson, and B. Raphael. A formal basis for the heuristic determination of
316

minimum cost paths. IEEE transactions on Systems Science and Cybernetics, 4(2):100–107,
317

1968.
318

[14] M. Hassan, D. Ceylan, R. Villegas, J. Saito, J. Yang, Y. Zhou, and M. J. Black. Stochastic
319

scene-aware motion prediction. In Proceedings of the IEEE/CVF International Conference on
320

Computer Vision, pages 11374–11384, 2021.
321

[15] M. Hassan, Y. Guo, T. Wang, M. Black, S. Fidler, and X. B. Peng. Synthesizing physical
322

character-scene interactions. arXiv preprint arXiv:2302.00883, 2023.
323

[16] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR,
324

pages 770–778, 2016.
325

[17] D. Helbing and P. Molnar. Social force model for pedestrian dynamics. Physical review E, 51
326

(5):4282, 1995.
327

[18] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural
328

information processing systems, 33:6840–6851, 2020.
329

[19] D. Q. Huynh. Metrics for 3d rotations: Comparison and analysis. Journal of Mathematical
330

Imaging and Vision, 35:155–164, 2009.
331

10


---Page Break---
[20] C. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, D. Anguelov, et al. Motiondiffuser: Con-
332

trollable multi-agent motion prediction using diffusion. In Proceedings of the IEEE/CVF
333

Conference on Computer Vision and Pattern Recognition, pages 9644–9653, 2023.
334

[21] H. Joo, T. Simon, X. Li, H. Liu, L. Tan, L. Gui, S. Banerjee, T. Godisart, B. Nabbe, I. Matthews,
335

et al. Panoptic studio: A massively multiview system for social interaction capture. TPAMI, 41
336

(1):190–204, 2017.
337

[22] L. Kavan, S. Collins, J. Žára, and C. O’Sullivan. Skinning with dual quaternions. In Proceedings
338

of the 2007 symposium on Interactive 3D graphics and games, pages 39–46, 2007.
339

[23] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. 3d gaussian splatting for real-time
340

radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.
341

[24] J. Kim, J. Kim, J. Na, and H. Joo. Parahome: Parameterizing everyday home activities towards
342

3d generative modeling of human-object interactions. arXiv preprint arXiv:2401.10232, 2024.
343

[25] D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114,
344

2013.
345

[26] K. M. Kitani, B. D. Ziebart, J. A. Bagnell, and M. Hebert. Activity forecasting. In Computer
346

Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October
347

7-13, 2012, Proceedings, Part IV 12, pages 201–214. Springer, 2012.
348

[27] M. Kocabas, N. Athanasiou, and M. J. Black. Vibe: Video inference for human body pose and
349

shape estimation. In CVPR, June 2020.
350

[28] M. Kocabas, Y. Yuan, P. Molchanov, Y. Guo, M. J. Black, O. Hilliges, J. Kautz, and
351

U. Iqbal. Pace: Human and camera motion estimation from in-the-wild videos. arXiv preprint
352

arXiv:2310.13768, 2023.
353

[29] J. Lee and H. Joo. Locomotion-action-manipulation: Synthesizing human-scene interactions
354

in complex 3d environments. In Proceedings of the IEEE/CVF International Conference on
355

Computer Vision (ICCV), 2023.
356

[30] A. Lerner, Y. Chrysanthou, and D. Lischinski. Crowds by example. In Computer graphics
357

forum, volume 26, pages 655–664. Wiley Online Library, 2007.
358

[31] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, C. Wang, G. Levine,
359

W. Ai, B. Martinez, et al. Behavior-1k: A human-centered, embodied ai benchmark with 1,000
360

everyday activities and realistic simulation. arXiv preprint arXiv:2403.09227, 2024.
361

[32] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black. SMPL: A skinned
362

multi-person linear model. SIGGRAPH Asia, 2015.
363

[33] W. Luo, B. Yang, and R. Urtasun. Fast and furious: Real time end-to-end 3d detection, tracking
364

and motion forecasting with a single convolutional net. In Proceedings of the IEEE conference
365

on Computer Vision and Pattern Recognition, pages 3569–3577, 2018.
366

[34] W.-C. Ma, D.-A. Huang, N. Lee, and K. M. Kitani. Forecasting interactive dynamics of
367

pedestrians with fictitious play. In Proceedings of the IEEE Conference on Computer Vision
368

and Pattern Recognition, pages 774–782, 2017.
369

[35] T. Magnenat, R. Laperrière, and D. Thalmann. Joint-dependent local deformations for hand
370

animation and object grasping. In Proceedings of Graphics Interface’88, pages 26–33. Canadian
371

Inf. Process. Soc, 1988.
372

[36] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black. Amass: Archive of
373

motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on
374

computer vision, pages 5442–5451, 2019.
375

[37] K. Mangalam, Y. An, H. Girase, and J. Malik. From goals, waypoints & paths to long term
376

human trajectory forecasting. In Proceedings of the IEEE/CVF International Conference on
377

Computer Vision, pages 15233–15242, 2021.
378

11


---Page Break---
[38] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf:
379

Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
380

[39] M. Niemeyer and A. Geiger. Giraffe: Representing scenes as compositional generative neural
381

feature fields. In CVPR, pages 11453–11464, 2021.
382

[40] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez,
383

D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li,
384

W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal,
385

P. Labatut, A. Joulin, and P. Bojanowski. Dinov2: Learning robust visual features without
386

supervision, 2023.
387

[41] J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein. Generative agents:
388

Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium
389

on User Interface Software and Technology, pages 1–22, 2023.
390

[42] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla.
391

Nerfies: Deformable neural radiance fields. In ICCV, 2021.
392

[43] G. Pavlakos, E. Weber, M. Tancik, and A. Kanazawa. The one where they reconstructed 3d
393

humans and environments in tv shows. In European Conference on Computer Vision, pages
394

732–749. Springer, 2022.
395

[44] S. Pellegrini, A. Ess, K. Schindler, and L. Van Gool. You’ll never walk alone: Modeling social
396

behavior for multi-target tracking. In 2009 IEEE 12th international conference on computer
397

vision, pages 261–268. IEEE, 2009.
398

[45] X. Puig, E. Undersander, A. Szot, M. D. Cote, T.-Y. Yang, R. Partsey, R. Desai, A. Clegg,
399

M. Hlavac, S. Y. Min, et al. Habitat 3.0: A co-habitat for humans, avatars, and robots. In The
400

Twelfth International Conference on Learning Representations, 2023.
401

[46] D. Rempe, Z. Luo, X. Bin Peng, Y. Yuan, K. Kitani, K. Kreis, S. Fidler, and O. Litany. Trace
402

and pace: Controllable pedestrian animation via guided trajectory diffusion. In Proceedings of
403

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13756–13766,
404

2023.
405

[47] N. Rhinehart and K. M. Kitani. Learning action maps of large environments via first-person
406

vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
407

pages 580–588, 2016.
408

[48] T. Salzmann, B. Ivanovic, P. Chakravarty, and M. Pavone. Trajectron++: Dynamically-feasible
409

trajectory forecasting with heterogeneous data. In Computer Vision–ECCV 2020: 16th European
410

Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16, pages 683–700.
411

Springer, 2020.
412

[49] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk. From coarse to fine: Robust hierarchical
413

localization at large scale. In Proceedings of the IEEE/CVF Conference on Computer Vision
414

and Pattern Recognition, pages 12716–12725, 2019.
415

[50] A. Seff, B. Cera, D. Chen, M. Ng, A. Zhou, N. Nayakanti, K. S. Refaat, R. Al-Rfou, and
416

B. Sapp. Motionlm: Multi-agent motion forecasting as language modeling. In Proceedings of
417

the IEEE/CVF International Conference on Computer Vision, pages 8579–8590, 2023.
418

[51] N. Snavely, S. M. Seitz, and R. Szeliski. Modeling the world from internet photo collections.
419

IJCV, 2008.
420

[52] C. Song, G. Yang, K. Deng, J.-Y. Zhu, and D. Ramanan. Total-recon: Deformable scene
421

reconstruction for embodied view synthesis. In ICCV, 2023.
422

[53] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based
423

generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456,
424

2020.
425

12


---Page Break---
[54] S. Srivastava, C. Li, M. Lingelbach, R. Martín-Martín, F. Xia, K. E. Vainio, Z. Lian, C. Gokmen,
426

S. Buch, K. Liu, et al. Behavior: Benchmark for everyday household activities in virtual,
427

interactive, and ecological environments. In Conference on robot learning, pages 477–490.
428

PMLR, 2022.
429

[55] T. Sun, Y. Hao, S. Huang, S. Savarese, K. Schindler, M. Pollefeys, and I. Armeni. Nothing
430

stands still: A spatiotemporal benchmark on 3d point cloud registration under large geometric
431

and temporal change. arXiv preprint arXiv:2311.09346, 2023.
432

[56] R. Szeliski and S. B. Kang. Shape ambiguities in structure from motion. IEEE Transactions on
433

Pattern Analysis and Machine Intelligence, 19(5):506–512, 1997.
434

[57] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano. Human motion
435

diffusion model. arXiv preprint arXiv:2209.14916, 2022.
436

[58] J. Van Den Berg, S. J. Guy, M. Lin, and D. Manocha. Reciprocal n-body collision avoidance.
437

In Robotics Research: The 14th International Symposium ISRR, pages 3–19. Springer, 2011.
438

[59] C.-Y. Weng, B. Curless, P. P. Srinivasan, J. T. Barron, and I. Kemelmacher-Shlizerman. Hu-
439

mannerf: Free-viewpoint rendering of moving people from monocular video. In CVPR, pages
440

16210–16220, 2022.
441

[60] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T.
442

Barron, B. Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. arXiv preprint
443

arXiv:2312.02981, 2023.
444

[61] S. Wu, T. Jakab, C. Rupprecht, and A. Vedaldi. Dove: Learning deformable 3d objects by
445

watching videos. arXiv preprint arXiv:2107.10844, 2021.
446

[62] Y. Xie, V. Jampani, L. Zhong, D. Sun, and H. Jiang. Omnicontrol: Control any joint at any time
447

for human motion generation. arXiv preprint arXiv:2310.08580, 2023.
448

[63] G. Yang and D. Ramanan. Volumetric correspondence networks for optical flow. In NeurIPS,
449

2019.
450

[64] G. Yang, D. Sun, V. Jampani, D. Vlasic, F. Cole, H. Chang, D. Ramanan, W. T. Freeman, and
451

C. Liu. LASR: Learning articulated shape reconstruction from a monocular video. In CVPR,
452

2021.
453

[65] G. Yang, M. Vo, N. Natalia, D. Ramanan, A. Vedaldi, and H. Joo. Banmo: Building animatable
454

3d neural models from many casual videos. In CVPR, 2022.
455

[66] G. Yang, C. Wang, N. D. Reddy, and D. Ramanan. Reconstructing Animatable Categories from
456

Videos. CVPR, 2023.
457

[67] G. Yang, S. Yang, J. Z. Zhang, Z. Manchester, and D. Ramanan. Physically plausible recon-
458

struction from monocular videos. In ICCV, 2023.
459

[68] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, and F. Zheng. Track anything: Segment anything
460

meets videos, 2023.
461

[69] V. Ye, G. Pavlakos, J. Malik, and A. Kanazawa. Decoupling human and camera motion from
462

videos in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and
463

Pattern Recognition, pages 21222–21232, 2023.
464

[70] Y. Yuan, U. Iqbal, P. Molchanov, K. Kitani, and J. Kautz. Glamr: Global occlusion-aware
465

human mesh recovery with dynamic cameras. In Proceedings of the IEEE/CVF conference on
466

computer vision and pattern recognition, pages 11038–11049, 2022.
467

[71] Y. Yuan, J. Song, U. Iqbal, A. Vahdat, and J. Kautz. Physdiff: Physics-guided human motion
468

diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer
469

Vision, pages 16010–16021, 2023.
470

[72] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion
471

models, 2023.
472

13


---Page Break---
[73] K. Zhao, Y. Zhang, S. Wang, T. Beeler, and S. Tang. Synthesizing diverse human motions in 3d
473

indoor scenes. arXiv preprint arXiv:2305.12411, 2023.
474

[74] Z. Zhong, D. Rempe, D. Xu, Y. Chen, S. Veer, T. Che, B. Ray, and M. Pavone. Guided
475

conditional diffusion for controllable traffic simulation. In 2023 IEEE International Conference
476

on Robotics and Automation (ICRA), pages 3560–3566. IEEE, 2023.
477

[75] B. D. Ziebart, A. L. Maas, J. A. Bagnell, A. K. Dey, et al. Maximum entropy inverse reinforce-
478

ment learning. In Aaai, volume 8, pages 1433–1438. Chicago, IL, USA, 2008.
479

[76] B. D. Ziebart, N. Ratliff, G. Gallagher, C. Mertz, K. Peterson, J. A. Bagnell, M. Hebert,
480

A. K. Dey, and S. Srinivasa. Planning-based prediction for pedestrians. In 2009 IEEE/RSJ
481

International Conference on Intelligent Robots and Systems, pages 3931–3936. IEEE, 2009.
482

14


---Page Break---
A
Additional Implementation Details
483

Model Architecture. The score function of the goal is implemented as 6-layer MLP with hidden
484

size 128. The the score functions of the paths and body motions are implemented as 1D UNets
485

taken from MDM [57]. The sampling frequency is set to be 0.1s, resulting a sequence length of 56.
486

The environment encoder is implemented as a 6-layer 3D ConvNet with kernel size 3 and channel
487

dimension 128. The observer encoder and history encoder are implemented as a 3-layer MLP with
488

hidden size 128.
489

We use a linear noise schedule at training time and 50 denoising steps. At test time, each goal
490

denoising step takes 2ms and each path/body denoising step takes 9ms on a GeForce RTX 3090 GPU.
491

Data Collection. We collect RGBD videos using an iPhone, similar to TotalRecon [52]. To train
492

the neural localizer, we use Polycam to take the walkthrough video and extract a textured mesh. For
493

behavior capture, we use Record3D App to record videos and extract color images and depth images.
494

B
Additional Results
495

Histogram of Agent / Observer Visitation. We show final camera and agent registration to the
496

canonical scene in Fig. 6. The registered 3D trajectories provides statistics of agent’s and user’s
497

preference over the environment.
498

Agent trajectories

Agent preference (visitation)
User preference (visitation)

User trajectories

Low                         High
Low                         High

Color: shot id

Figure 6: Given the 3D trajectories of the agent and the user accumulated over time (top), one could
compute their preference represented by 3D heatmaps (bottom). Note the high agent preference over
table and sofa.

Varying Observer’s Motion. We find that various interactive behaviors can be generated by
499

conditioning the model on different observer motion. The results are shown in Fig. 7.
500

Comparison to TotalRecon. In the main paper, we compare to TotalRecon on scene reconstruction
501

by providing it multiple videos. Here, we include additional comparison in their the original single
502

video setup. We find that TotalRecon fails to build a good agent model, or a complete scene model
503

given limited observations, while our method can leverage multiple videos as inputs to build a better
504

agent and scene model. The results are shown in Fig. 8.
505

15


---Page Break---
User trajectory

Goals

Planned Paths

Past trajectory

Goals

Goals

Goals

Planned Paths

Planned Paths

Planned Paths

User trajectory

User trajectory

User trajectory

Early                   Late

Figure 7: Interactive behavior simulation with user conditioning. By changing the trajectory of the
user, one could influence the behavior of the agent. Given different control inputs, the agent may
follow the user or run away from the user.

TotalRecon

Reference image

Distortion

Incomplete
No distortion

Complete

Complete shape
Good alignment
Missing limbs
Misaligned limbs

Ours

Figure 8: Qualitative comparison with TotalRecon [52] on 4D reconstruction. Top: reconstruction
of the agent at at specific frame. Total-recon produces shapes with missing limbs and bone trans-
formations that are misaligned with the shape, while our method produces complete shapes and
good alignment. Bottom: reconstruction of the environment. TotalRecon produces distorted and
incomplete geometry (due to lack of observations from a single video), while our method produces
an accurate and complete environment reconstruction.

16


---Page Break---
C
Limitations and Future Works
506

High-level Behavior. The current ATS model is trained with time-horizon of T ∗= 6.4 seconds.
507

We observe that the model only learns mid-level behaviors of an agent (e.g., trying to move to a
508

destination; staying at a location; walking around). We hope incorporating a memory module and
509

training with longer time horizon will enable learning higher-level behaviors of an agent.
510

Scaling-up. As indicated by the experimental results, the goals sampled from ATS may fail to cover
511

the actual goal when evaluated on the (unseen) test data. This raises safety concerns when using
512

ATS for the prediction task (e.g., predicting the behavior of pedestrains in autonomous driving). One
513

potential solution of improving the generalization ability is to collect more diverse behavior data
514

from in the wild videos, or leverage “large” video priors trained on internet-scale videos.
515

Multiple Agents. We show results of learning behavior models of a single agent, but our method for
516

4D reconstruction and interactive goal-driven behavior modeling is not limited to a single agent. We
517

leave learning multi-agent behavior simulation from videos as future work.
518

Physical Interactions. Our method reconstructs and generates the kinematics of an agent, which
519

may produce physically-implausible results (e.g., penetration with the ground and foot sliding). One
520

promising way to deal with this problem is to add physics constraints to the reconstruction and motion
521

generation [67, 71].
522

Environment Reconstruction. To build a complete reconstruction of the environment, we register
523

multiple videos to a shared canonical space. However, the transient structures (e.g., cushion that
524

can be moved over time) may not be reconstructed well due to lack of observations. One potential
525

solution of reconstructing these transient structures is to combine generative image priors with the
526

reconstruction pipeline [60].
527

D
Social Impact
528

Our method is able to learn interactive behavior from videos, which could help build simulators for
529

autonomous driving, gaming, and movie applications. It is also capable of building personalized
530

behavior models from casually collected video data, which can benefit users who do not have access
531

to a motion capture studio. On the negative side, the behavior generation model could be used as
532

“deepfake” and poses threats to user’s privacy and social security.
533

17


---Page Break---
NeurIPS Paper Checklist
534

The checklist is designed to encourage best practices for responsible machine learning research,
535

addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
536

the checklist: The papers not including the checklist will be desk rejected. The checklist should
537

follow the references and follow the (optional) supplemental material. The checklist does NOT count
538

towards the page limit.
539

Please read the checklist guidelines carefully for information on how to answer these questions. For
540

each question in the checklist:
541

• You should answer [Yes] , [No] , or [NA] .
542

• [NA] means either that the question is Not Applicable for that particular paper or the
543

relevant information is Not Available.
544

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).
545

The checklist answers are an integral part of your paper submission. They are visible to the
546

reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
547

(after eventual revisions) with the final version of your paper, and its final version will be published
548

with the paper.
549

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
550

While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
551

proper justification is given (e.g., "error bars are not reported because it would be too computationally
552

expensive" or "we were unable to find the license for the dataset we used"). In general, answering
553

"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
554

acknowledge that the true answer is often more nuanced, so please just use your best judgment and
555

write a justification to elaborate. All supporting evidence can appear either in the main paper or the
556

supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
557

please point to the section(s) where related material for the question can be found.
558

IMPORTANT, please:
559

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
560

• Keep the checklist subsection headings, questions/answers and guidelines below.
561

• Do not modify the questions and only use the provided macros for your answers.
562

1. Claims
563

Question: Do the main claims made in the abstract and introduction accurately reflect the
564

paper’s contributions and scope?
565

Answer: [Yes]
566

Justification: The main claims made in the abstract and introduction accurately reflect the
567

paper’s contributions and scope.
568

Guidelines:
569

• The answer NA means that the abstract and introduction do not include the claims
570

made in the paper.
571

• The abstract and/or introduction should clearly state the claims made, including the
572

contributions made in the paper and important assumptions and limitations. A No or
573

NA answer to this question will not be perceived well by the reviewers.
574

• The claims made should match theoretical and experimental results, and reflect how
575

much the results can be expected to generalize to other settings.
576

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
577

are not attained by the paper.
578

2. Limitations
579

Question: Does the paper discuss the limitations of the work performed by the authors?
580

Answer: [Yes]
581

18


---Page Break---
Justification: The paper discusses the limitations of the work performed by the authors.
582

Guidelines:
583

• The answer NA means that the paper has no limitation while the answer No means that
584

the paper has limitations, but those are not discussed in the paper.
585

• The authors are encouraged to create a separate "Limitations" section in their paper.
586

• The paper should point out any strong assumptions and how robust the results are to
587

violations of these assumptions (e.g., independence assumptions, noiseless settings,
588

model well-specification, asymptotic approximations only holding locally). The authors
589

should reflect on how these assumptions might be violated in practice and what the
590

implications would be.
591

• The authors should reflect on the scope of the claims made, e.g., if the approach was
592

only tested on a few datasets or with a few runs. In general, empirical results often
593

depend on implicit assumptions, which should be articulated.
594

• The authors should reflect on the factors that influence the performance of the approach.
595

For example, a facial recognition algorithm may perform poorly when image resolution
596

is low or images are taken in low lighting. Or a speech-to-text system might not be
597

used reliably to provide closed captions for online lectures because it fails to handle
598

technical jargon.
599

• The authors should discuss the computational efficiency of the proposed algorithms
600

and how they scale with dataset size.
601

• If applicable, the authors should discuss possible limitations of their approach to
602

address problems of privacy and fairness.
603

• While the authors might fear that complete honesty about limitations might be used by
604

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
605

limitations that aren’t acknowledged in the paper. The authors should use their best
606

judgment and recognize that individual actions in favor of transparency play an impor-
607

tant role in developing norms that preserve the integrity of the community. Reviewers
608

will be specifically instructed to not penalize honesty concerning limitations.
609

3. Theory Assumptions and Proofs
610

Question: For each theoretical result, does the paper provide the full set of assumptions and
611

a complete (and correct) proof?
612

Answer: [NA]
613

Justification: The paper does not include theoretical results.
614

Guidelines:
615

• The answer NA means that the paper does not include theoretical results.
616

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
617

referenced.
618

• All assumptions should be clearly stated or referenced in the statement of any theorems.
619

• The proofs can either appear in the main paper or the supplemental material, but if
620

they appear in the supplemental material, the authors are encouraged to provide a short
621

proof sketch to provide intuition.
622

• Inversely, any informal proof provided in the core of the paper should be complemented
623

by formal proofs provided in appendix or supplemental material.
624

• Theorems and Lemmas that the proof relies upon should be properly referenced.
625

4. Experimental Result Reproducibility
626

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
627

perimental results of the paper to the extent that it affects the main claims and/or conclusions
628

of the paper (regardless of whether the code and data are provided or not)?
629

Answer: [Yes]
630

Justification: The authors tried their best to disclose the information needed to reproduce
631

the experiments.
632

Guidelines:
633

19


---Page Break---
• The answer NA means that the paper does not include experiments.
634

• If the paper includes experiments, a No answer to this question will not be perceived
635

well by the reviewers: Making the paper reproducible is important, regardless of
636

whether the code and data are provided or not.
637

• If the contribution is a dataset and/or model, the authors should describe the steps taken
638

to make their results reproducible or verifiable.
639

• Depending on the contribution, reproducibility can be accomplished in various ways.
640

For example, if the contribution is a novel architecture, describing the architecture fully
641

might suffice, or if the contribution is a specific model and empirical evaluation, it may
642

be necessary to either make it possible for others to replicate the model with the same
643

dataset, or provide access to the model. In general. releasing code and data is often
644

one good way to accomplish this, but reproducibility can also be provided via detailed
645

instructions for how to replicate the results, access to a hosted model (e.g., in the case
646

of a large language model), releasing of a model checkpoint, or other means that are
647

appropriate to the research performed.
648

• While NeurIPS does not require releasing code, the conference does require all submis-
649

sions to provide some reasonable avenue for reproducibility, which may depend on the
650

nature of the contribution. For example
651

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
652

to reproduce that algorithm.
653

(b) If the contribution is primarily a new model architecture, the paper should describe
654

the architecture clearly and fully.
655

(c) If the contribution is a new model (e.g., a large language model), then there should
656

either be a way to access this model for reproducing the results or a way to reproduce
657

the model (e.g., with an open-source dataset or instructions for how to construct
658

the dataset).
659

(d) We recognize that reproducibility may be tricky in some cases, in which case
660

authors are welcome to describe the particular way they provide for reproducibility.
661

In the case of closed-source models, it may be that access to the model is limited in
662

some way (e.g., to registered users), but it should be possible for other researchers
663

to have some path to reproducing or verifying the results.
664

5. Open access to data and code
665

Question: Does the paper provide open access to the data and code, with sufficient instruc-
666

tions to faithfully reproduce the main experimental results, as described in supplemental
667

material?
668

Answer: [No]
669

Justification: The code will be released once we put it in a better shape.
670

Guidelines:
671

• The answer NA means that paper does not include experiments requiring code.
672

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
673

public/guides/CodeSubmissionPolicy) for more details.
674

• While we encourage the release of code and data, we understand that this might not be
675

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
676

including code, unless this is central to the contribution (e.g., for a new open-source
677

benchmark).
678

• The instructions should contain the exact command and environment needed to run to
679

reproduce the results. See the NeurIPS code and data submission guidelines (https:
680

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
681

• The authors should provide instructions on data access and preparation, including how
682

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
683

• The authors should provide scripts to reproduce all experimental results for the new
684

proposed method and baselines. If only a subset of experiments are reproducible, they
685

should state which ones are omitted from the script and why.
686

• At submission time, to preserve anonymity, the authors should release anonymized
687

versions (if applicable).
688

20


---Page Break---
• Providing as much information as possible in supplemental material (appended to the
689

paper) is recommended, but including URLs to data and code is permitted.
690

6. Experimental Setting/Details
691

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
692

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
693

results?
694

Answer: [Yes]
695

Justification: The authors tried their best to specify all the training and test details.
696

Guidelines:
697

• The answer NA means that the paper does not include experiments.
698

• The experimental setting should be presented in the core of the paper to a level of detail
699

that is necessary to appreciate the results and make sense of them.
700

• The full details can be provided either with the code, in appendix, or as supplemental
701

material.
702

7. Experiment Statistical Significance
703

Question: Does the paper report error bars suitably and correctly defined or other appropriate
704

information about the statistical significance of the experiments?
705

Answer: [No]
706

Justification: The results currently do not have error bars, but we will try adding them later.
707

Based on empirical evidence of running the experiments, we think it will not affect the
708

conclusion.
709

Guidelines:
710

• The answer NA means that the paper does not include experiments.
711

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
712

dence intervals, or statistical significance tests, at least for the experiments that support
713

the main claims of the paper.
714

• The factors of variability that the error bars are capturing should be clearly stated (for
715

example, train/test split, initialization, random drawing of some parameter, or overall
716

run with given experimental conditions).
717

• The method for calculating the error bars should be explained (closed form formula,
718

call to a library function, bootstrap, etc.)
719

• The assumptions made should be given (e.g., Normally distributed errors).
720

• It should be clear whether the error bar is the standard deviation or the standard error
721

of the mean.
722

• It is OK to report 1-sigma error bars, but one should state it. The authors should
723

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
724

of Normality of errors is not verified.
725

• For asymmetric distributions, the authors should be careful not to show in tables or
726

figures symmetric error bars that would yield results that are out of range (e.g. negative
727

error rates).
728

• If error bars are reported in tables or plots, The authors should explain in the text how
729

they were calculated and reference the corresponding figures or tables in the text.
730

8. Experiments Compute Resources
731

Question: For each experiment, does the paper provide sufficient information on the com-
732

puter resources (type of compute workers, memory, time of execution) needed to reproduce
733

the experiments?
734

Answer: [Yes]
735

Justification: The paper provides information about computer resources.
736

Guidelines:
737

• The answer NA means that the paper does not include experiments.
738

21


---Page Break---
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
739

or cloud provider, including relevant memory and storage.
740

• The paper should provide the amount of compute required for each of the individual
741

experimental runs as well as estimate the total compute.
742

• The paper should disclose whether the full research project required more compute
743

than the experiments reported in the paper (e.g., preliminary or failed experiments that
744

didn’t make it into the paper).
745

9. Code Of Ethics
746

Question: Does the research conducted in the paper conform, in every respect, with the
747

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
748

Answer: [Yes]
749

Justification: The authors have reviewed the code of ethics and think the paper follows the
750

guideline.
751

Guidelines:
752

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
753

• If the authors answer No, they should explain the special circumstances that require a
754

deviation from the Code of Ethics.
755

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
756

eration due to laws or regulations in their jurisdiction).
757

10. Broader Impacts
758

Question: Does the paper discuss both potential positive societal impacts and negative
759

societal impacts of the work performed?
760

Answer: [Yes]
761

Justification: The paper discussed potential positive and negative impact.
762

Guidelines:
763

• The answer NA means that there is no societal impact of the work performed.
764

• If the authors answer NA or No, they should explain why their work has no societal
765

impact or why the paper does not address societal impact.
766

• Examples of negative societal impacts include potential malicious or unintended uses
767

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
768

(e.g., deployment of technologies that could make decisions that unfairly impact specific
769

groups), privacy considerations, and security considerations.
770

• The conference expects that many papers will be foundational research and not tied
771

to particular applications, let alone deployments. However, if there is a direct path to
772

any negative applications, the authors should point it out. For example, it is legitimate
773

to point out that an improvement in the quality of generative models could be used to
774

generate deepfakes for disinformation. On the other hand, it is not needed to point out
775

that a generic algorithm for optimizing neural networks could enable people to train
776

models that generate Deepfakes faster.
777

• The authors should consider possible harms that could arise when the technology is
778

being used as intended and functioning correctly, harms that could arise when the
779

technology is being used as intended but gives incorrect results, and harms following
780

from (intentional or unintentional) misuse of the technology.
781

• If there are negative societal impacts, the authors could also discuss possible mitigation
782

strategies (e.g., gated release of models, providing defenses in addition to attacks,
783

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
784

feedback over time, improving the efficiency and accessibility of ML).
785

11. Safeguards
786

Question: Does the paper describe safeguards that have been put in place for responsible
787

release of data or models that have a high risk for misuse (e.g., pretrained language models,
788

image generators, or scraped datasets)?
789

Answer: [NA]
790

22


---Page Break---
Justification: The paper poses no such risks.
791

Guidelines:
792

• The answer NA means that the paper poses no such risks.
793

• Released models that have a high risk for misuse or dual-use should be released with
794

necessary safeguards to allow for controlled use of the model, for example by requiring
795

that users adhere to usage guidelines or restrictions to access the model or implementing
796

safety filters.
797

• Datasets that have been scraped from the Internet could pose safety risks. The authors
798

should describe how they avoided releasing unsafe images.
799

• We recognize that providing effective safeguards is challenging, and many papers do
800

not require this, but we encourage authors to take this into account and make a best
801

faith effort.
802

12. Licenses for existing assets
803

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
804

the paper, properly credited and are the license and terms of use explicitly mentioned and
805

properly respected?
806

Answer: [NA]
807

Justification: Thee paper does not use existing assets.
808

Guidelines:
809

• The answer NA means that the paper does not use existing assets.
810

• The authors should cite the original paper that produced the code package or dataset.
811

• The authors should state which version of the asset is used and, if possible, include a
812

URL.
813

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
814

• For scraped data from a particular source (e.g., website), the copyright and terms of
815

service of that source should be provided.
816

• If assets are released, the license, copyright information, and terms of use in the
817

package should be provided. For popular datasets, paperswithcode.com/datasets
818

has curated licenses for some datasets. Their licensing guide can help determine the
819

license of a dataset.
820

• For existing datasets that are re-packaged, both the original license and the license of
821

the derived asset (if it has changed) should be provided.
822

• If this information is not available online, the authors are encouraged to reach out to
823

the asset’s creators.
824

13. New Assets
825

Question: Are new assets introduced in the paper well documented and is the documentation
826

provided alongside the assets?
827

Answer: [Yes]
828

Justification: The paper discussed the new assets.
829

Guidelines:
830

• The answer NA means that the paper does not release new assets.
831

• Researchers should communicate the details of the dataset/code/model as part of their
832

submissions via structured templates. This includes details about training, license,
833

limitations, etc.
834

• The paper should discuss whether and how consent was obtained from people whose
835

asset is used.
836

• At submission time, remember to anonymize your assets (if applicable). You can either
837

create an anonymized URL or include an anonymized zip file.
838

14. Crowdsourcing and Research with Human Subjects
839

Question: For crowdsourcing experiments and research with human subjects, does the paper
840

include the full text of instructions given to participants and screenshots, if applicable, as
841

well as details about compensation (if any)?
842

23


---Page Break---
Answer: [NA]
843

Justification: The paper does not deal with crowdsourcing or external human subjects.
844

Guidelines:
845

• The answer NA means that the paper does not involve crowdsourcing nor research with
846

human subjects.
847

• Including this information in the supplemental material is fine, but if the main contribu-
848

tion of the paper involves human subjects, then as much detail as possible should be
849

included in the main paper.
850

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
851

or other labor should be paid at least the minimum wage in the country of the data
852

collector.
853

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
854

Subjects
855

Question: Does the paper describe potential risks incurred by study participants, whether
856

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
857

approvals (or an equivalent approval/review based on the requirements of your country or
858

institution) were obtained?
859

Answer: [NA]
860

Justification: The paper does not deal with crowdsourcing or external human subjects.
861

Guidelines:
862

• The answer NA means that the paper does not involve crowdsourcing nor research with
863

human subjects.
864

• Depending on the country in which research is conducted, IRB approval (or equivalent)
865

may be required for any human subjects research. If you obtained IRB approval, you
866

should clearly state this in the paper.
867

• We recognize that the procedures for this may vary significantly between institutions
868

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
869

guidelines for their institution.
870

• For initial submissions, do not include any information that would break anonymity (if
871

applicable), such as the institution conducting the review.
872

24


---Page Break---
