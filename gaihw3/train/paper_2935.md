Exploring Prosocial Irrationality for LLM Agents: A
Social Cognition View

Anonymous Author(s)
Affiliation
Address
email

You are right,  
Apple is blue.

Apple is Red

I’m an expert, I assert that 

Apple is blue. ·

Authority Effect

What color is the apple?

A: Red            B: Blue

B
A

Herd Effect

B
B
B

B
B
B

Ban Franklin Effect

Favorability Level 

Before
After

Could you do me a favor?

Rumor Chain Effect

Banana 
hits me
≠

Gambler’s Fallacy

Lose

times

Lose

I will probably get 

a big win today!

1
2

Confirmation Bias

Facts & 
Evidence

Our 
Beliefs

Evidence LM Agents Believe

Halo Effect

Kind, Rich…

VS

TOM
JACK

TOM should also 
be good at other 

things

inaccurate

Cognitive Bias
Hallucination

irrational

…

Mirror

subjective
imaginary

misleading

Open-ended

I eat an 

apple

Figure 1: CogMir Sample Evaluations. Mirror Human Cognitive Bias and LLM Agents Hallucination
through Social Science Experiments via representational social and cognitive phenomena.

Abstract

Large language models (LLMs) have been shown to face hallucination issues due
1

to the data they trained on often containing human bias; whether this is reflected
2

in the decision-making process of LLM agents remains under-explored. As LLM
3

Agents are increasingly employed in intricate social environments, a pressing and
4

natural question emerges: Can LLM Agents leverage hallucinations to mirror
5

human cognitive biases, thus exhibiting irrational social intelligence? In this
6

paper, we probe the irrational behavior among contemporary LLM agents by
7

melding practical social science experiments with theoretical insights. Specifically,
8

We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes
9

hallucination properties to assess and enhance LLM Agents’ social intelligence
10

through cognitive biases. Experimental results on CogMir subsets show that LLM
11

Agents and humans exhibit high consistency in irrational and prosocial decision-
12

making under uncertain conditions, underscoring the prosociality of LLM Agents
13

as social entities, and highlighting the significance of hallucination properties.
14

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
Additionally, CogMir framework demonstrates its potential as a valuable platform
15

for encouraging more research into the social intelligence of LLM Agents.
16

1
Introduction
17

Human mind may often be better than rational. – Leda Cosmides, John Tooby. With the extensive
18

deployment of large language models (LLMs)[32, 14], LLM-based agent systems are increasingly
19

developed to cater to diverse applications such as task-solving, evaluation, and simulation [12, 7, 19,
20

17, 38]. Given the similarities between the operational dynamics of LLM-based agent systems and
21

human social structures, it is pertinent to explore the intersection of these domains. Recent studies
22

have highlighted the social potential of LLM Agents through constructing multi-agent systems that
23

simulate interactive social scenarios [40, 39, 31] revealing the social dynamics among interacting
24

LLM Agents and showing parallels to human behaviors. For instance, LLMs can achieve social
25

goals [40] and adhere to social norms [31] within LLM-based Multi-Agent systems. Nonetheless,
26

these research efforts exhibit two significant gaps: 1) They primarily focus on black-box testing
27

in multi-agent role-playing systems, concentrating on the outputs and behaviors of agents while
28

neglecting to investigate the internal mechanisms or cognitive processes that drive these behaviors.
29

2) LLM Agents are prone to hallucinations—producing misleading or incorrect information, due to
30

their training data and inherent biases [13, 30]. The potential impact of such hallucinations on the
31

social intelligence of LLM Agents remains under-explored.
32

Cognitive biases, pervasive in human society, highlight the subjective nature of human behavior [1, 6].
33

Human cognitive biases can lead to irrational decisions and imaginary contents like the hallucination
34

phenomenon in LLMs [13, 36]. However, evolutionary psychology suggests that rationality is
35

unnatural; rather, human irrationality is an adaptive selected trait for navigating complex social
36

environments [9, 18]. Analogically, in this paper, we argue that LLMs’ hallucination (or imagination)
37

attributes are the fundamental condition that confers social intelligence on LLM Agents. We explore
38

the similarities in social potential between human cognitive biases and LLM Agent hallucination
39

attributes for the first time, particularly in irrational decision-making, to analogically deduce the
40

underlying reasons for LLM Agents’ possession of social intelligence.
41

To study LLM Agents’ potential for irrational social intelligence, we present CogMir, an open-ended
42

and dynamic multi-agent framework designed specifically for evaluating, exploring, and explaining
43

social intelligence for LLM Agents via systematic assessments of cognitive biases. Specifically, the
44

hallucinatory attributes of LLMs are exploited (i.e., via treating the cognitive bias as a manageable
45

and interpretable factor) in CogMir to probe their social intelligence, so as to providing enhanced
46

interpretability for LLM agents. In addition, our proposed CogMir framework integrates sociological
47

methodologies to abstract typical social structures and employ various Multi-Human-Agent Interac-
48

tion Combinations and Communication Modes to interlink System Objects. This integrative setup is
49

designed to systematically encompass and simulate various cognitive bias scenarios, as depicted in
50

Fig. 1. On the evaluation front, CogMir combines sociological assessments, manual discrimination,
51

LLM assessments, and traditional AI discrimination techniques to realize a multidimensional assess-
52

ment system. By using flexible module configurations from standardized sets, CogMir simplifies
53

social architectures, enabling diverse applications in experimental simulations and evaluations.
54

Designed as an open-ended framework for continuous interpretative study, we provide multiple
55

CogMir subset samples as examples. Existing assessments of various cognitive effects demonstrate
56

that LLM agents exhibit a high degree of consistency with humans in prosocial cognitive biases and
57

counter-intuitive phenomena. However, LLM Agents demonstrate a higher sensitivity to factors like
58

certainty and social status than humans, exhibiting more variability in their decision-making biases
59

under conditions of certainty and uncertainty. In contrast, human decision-making tends to be more
60

consistent across these conditions. In summary, this paper makes the following contributions:
61

• We are the first to breach the black-box theoretical bottleneck of the Multi-LLM Agents’
62

social intelligence, by utilizing LLM Agent hallucination properties to mirror human cogni-
63

tive biases as explanatory and controllable variables to systematically assess and explain
64

LLM Agent’s social intelligence through an evolutionary sociology lens.
65

• We propose CogMir, an extensible, modularized, and dynamic Multi-LLM Agents frame-
66

work for assessing, exploiting, and interpreting social intelligence via cognitive bias, aligned
67

with social science methodologies.
68

2


---Page Break---
• We offer diverse CogMir subsets and use cases to steer future research. Our experimental
69

findings highlight the alignment and distinctions between LLM Agents and humans in the
70

decision-making process.
71

• CogMir indicates that LLM Agents have pro-social behavior in irrational decision-making,
72

emphasizing the significant role of hallucination properties in their social intelligence.
73

2
Related Work
74

Our work is inspired by interdisciplinary areas such as social sciences and evolutionary psychology.
75

LLM Hallucination & Cognitive Bias. Hallucination in LLMs occurs when they generate content
76

that is not factually accurate, often arising from the reliance on patterns learned from biased training
77

data or the model’s limitations in understanding context and accessing current information [13, 36].
78

Such hallucinations might be beneficial in creative fields, where these models can act as “collaborative
79

creative partners.” They offer innovative and inspiring outputs that can lead to the discovery of novel
80

ideas and connections [30]. Concurrently, cognitive biases and evolutionary psychology offer essential
81

perspectives on decision-making processes and prosocial behaviors, which can be analogously applied
82

to explain the social intelligence of LLM Agents[18, 1]. In this work, through mirroring human
83

cognitive bias, we suggest that the hallucination property of LLM is the basis for prosocial behavior
84

in LLM Agents, representing a potential form of advanced intelligence.
85

LLM Agent Social Intelligence Evaluation. Several benchmarks traditionally utilized for evaluating
86

the social intelligence of artificial agents, such as SocialIQA [33] and ToMi [16], are increasingly
87

being surpassed in difficulty as language models advance. In response to this trend, recent efforts
88

have synthesized existing benchmarks and introduced innovative evaluation datasets specifically
89

tailored for assessing LLM Agents [40, 19, 35, 26]. Despite the wide range of social intelligence
90

types [18], there is no standard workflow for investigating LLM Agents’ social intelligence. CogMir
91

has developed an open and accessible workflow aligned with consensus-based approaches in social
92

science, facilitating systematic testing and advancement of social intelligence in language models.
93

Multi-Agents Social System. Dialogue systems facilitate AI interactions, with task-oriented models
94

focusing on specific tasks and open-domain systems designed for general conversation, often enhanc-
95

ing engagement by incorporating personal details and creating deep understanding [40]. Simulations
96

with LLMs demonstrate their abilities to produce human-like social interactions by applying these
97

models to tasks like collaborative software development [7, 12, 17, 39, 31, 38, 35]. Despite these
98

advancements, exploration of why these models exhibit social capabilities remain limited. Our work
99

tries to bridge this theoretical gap by drawing on research methods from human social evolution
100

studies, thereby enhancing the interpretability of Multi-LLM Agents social systems.
101

3
CogMir: Multi-LLM Agents Framework On Cognitive Bias
102

In this section, we provide a detailed and modular overview of CogMir, organized into four main
103

elements: environmental setting, framework structure, cognitive biases subsets, and illustrative use
104

cases. These components are visually depicted in a left-to-right sequence in Fig. 2.
105

3.1
Environmental Settings
106

First, we outline a novel standard workflow for integrating social science methodologies with the
107

Multi-LLM Agents system, ensuring alignment with traditional experimental standards and adapting
108

data collection methods for Multi-LLM Agents environments.
109

CogMir environment settings are benchmarked against standard social science experiments through a
110

structured three-step process: Literature Search, Manual Selection, and LLM Agent Summarization.
111

A literature search pinpoints key social science experiments, which are then manually selected for
112

relevance and replicability. LLM Agents adapt these for integration into the Multi-LLM Agents
113

system within the CogMir framework. In the Mirror Settings process, data collection methods such
114

as surveys and interviews are transformed into Human-LLM Agent Q&A. Methods like case studies
115

and naturalistic observations are adapted to Multi-Human-Agent interaction scenarios.
116

3


---Page Break---
Social Science Settings

Mirror

Case Studies

Naturalistic Observations

Survey
Interviews

Multi-LLM Agents Settings

…

Q & A

Survey
Interviews

Case Studies

Multi-H-A Interaction

Naturalistic Observations

…
…

collaborate

Humans 
LMAs 
Datasets
Discriminators

Mirror Settings
Framework Structures
Subset Mirror Samples
Use Cases

Evaluation Sets

Participation Sets

Multi-H-A Interaction Combinations

Communication Modes

Required Object Sets

Broadcast | Parallel 
Point to Point | Series

Herd Effect

Q & A
Multi-H & Single-A 

…

Broadcast | Parallel 

construct

Rumor Chain Effect

…

…

Q & A
Multi-A

Point to Point | Series

Multi-Choice Q

Survey Q

construct

Q & A Bias Rate

Multi-H-A Bias Rate

66% Bias
34%

66% of Q & A reflect 
a cognitive bias.

Datasets:   Multi-Choice| Survey|…

State-of-art Technical 

Discriminations 

SelfCheckGPT
SimCSE

FACtScore
…

66% Bias

34%

Humans 
LMAs 

Discriminations

Prompts

Evaluation 

Sets

Figure 2: CogMir Framework. The framework is structured around four essential objects: humans,
LLM Agents, data, and discriminators. These objects interact within the framework to facilitate
Multi-Human-Agent (Multi-H-A) interactions and evaluations. CogMir features two communication
modes and five Multi-H-A interaction combinations, enabling varied configurations to suit diverse
social experimental needs. CogMir offers mirror cognitive bias samples (Fig. 1) and dynamic use
cases open for expansion. The framework is depicted in a left-to-right sequence.

Human-LLM Agent Q&A involves (1) Question Dataset Construction: Developing a diverse set
117

of questions tailored to specific study needs (e.g. multiple-choice, fill-in-the-blank, etc.) (2) Q&A
118

Scenario Design: Pairing the Question Datasets with scenarios that simulate real-world environments
119

(controlled settings like a room to dynamic public spaces like squares or transit stations). (3) Prompt
120

Engineering: Crafting appropriate prompts for the LLM Agents based on the scenario and question
121

dataset. (4) Analysis of LLM Agent Responses: Evaluating the responses from LLM Agents.
122

Multi-Human-Agent Interaction involves (1) Interaction Combination Configuration: Adapting
123

human-only social science settings to interactive environments that include humans and LLM Agents
124

(e.g., in group discussion experiments, some human participants are replaced with LLM Agents).
125

(2) Role Assignment: Specific roles and behaviors are assigned to humans and LLM Agents. This
126

assignment is guided by prompt engineering to ensure each participant acts according to social
127

science experiment guidelines. (3) Communication Mode Selection: Based on the original social
128

science setting select suitable communication modes for interaction. (4) Data Collection and Analysis:
129

Gathering and analyzing data from these interactions (e.g. dialogue, decision-making, etc.).
130

3.2
Framework Structure
131

After establishing realistic social science experiment environments, the next step is to select essential
132

components to support the above two mirror methods: Human-LLM Agent Q&A and Multi-Human-
133

Agent Interaction. This entails choosing participant objects, evaluation tools, and communication
134

modes. The CogMir framework is organized into modules for Required Objects, Communication
135

Modes, and Interaction Combinations to meet these needs.
136

Required Object Sets. Required Object encompasses all potential participants and evaluators in-
137

volved in the system. Participants include humans and LLM Agents, which allows for dynamic
138

setups where either or both can be involved in interactions depending on the experiment’s require-
139

ments. Evaluators include humans, LLM Agents, datasets, and discriminators. Datasets are utilized
140

to store and construct prompts about the experimental setup (e.g. experimental scenarios, character
141

information, etc.), task description, and Q&A question set. Discriminators are specialized tools
142

utilized to evaluate the social intelligence of LLM Agents, encompassing three main types: State-
143

of-the-art technical metrics such as SimCSE, SelfCheck, and FactScore [11, 22, 20] for objective,
144

quantitative assessment; Human discriminators that delve into nuanced and subjective aspects like
145

prosocial understanding; and LLM Agent discriminators, which involve the use of other LLM Agents
146

to assess and challenge responses from a subject LLM Agent.
147

4


---Page Break---
Communication Modes Sets. Communication modes dictate the nature of interactions within differ-
148

ent setups. We model the participants (humans or LLM Agents) as channels based on information
149

theory [34] to define two essential communication modes:
150

• Broadcast (or Parallel, C = C1 + C2 + . . . + Cn) which enables a single sender to transmit
151

a message to multiple receivers simultaneously.
152

• Point-to-point (or Series, C = min[C1, C2, . . . , Cn]) establishes communication between
153

two specific entities at a time (C denotes channel capacity).
154

Multi-H-A Interaction Combinations Sets. This module provides various combinations of Multi-
155

Human-LLM Agent interactions, tailored to different social science experimental needs, the most
156

frequently used combinations in social science settings include:
157

• Single-H-Single-A: One human interacting with one LLM Agent, predominantly used for
158

human-agent question-answering tasks (e.g. survey, interview, etc. ).
159

• Single-H-Multi-A: One human interacts with multiple LLM Agents, where humans can be
160

set as controlled variables to test Multi-LLM Agents’s social cognitive behaviors.
161

• Multi-H-Single-A: multiple humans interact with a single LLM Agent, which is suitable
162

for assessing the impact of group dynamics, such as consensus or conflict.
163

• Multi-A: multiple agents interacting without human participation.
164

• Multi-H-Multi-A: multiple humans and multiple LLM Agents interaction, integrating
165

elements from the previous setups to mimic complicated experimental interactions.
166

These modules offer a flexible framework for exploring LLM Agents’ cognitive biases in social
167

science experiments. Researchers can customize their setups by mixing different components to
168

examine specific hypotheses. We outline cognitive bias subsets as guidelines in the next section.
169

3.3
Cognitive Bias Subsets
170

We offer a collection of seven distinct Cognitive Bias Effects subsets, tailored for the analysis of LLM
171

Agents’ irrational decision-making processes: a) Herd Effect [5]: refers to the tendency of people to
172

follow the actions of a larger group, often disregarding their own beliefs. b) Authority Effect [21]:
173

involves people being more likely to comply with advice or instructions from someone perceived
174

as an authority figure. c) Ban Franklin Effect [10]: suggests that a person who does someone else
175

a favor is more likely to do another favor for that person, due to cognitive dissonance. d) Rumor
176

Chain Effect [3]: describes how information tends to change and distort as it passes from person to
177

person, often leading to misinformation. e) Gambler’s Fallacy [8]: refers to the incorrect belief that
178

past events can influence the likelihood of something happening in the future in random processes. f)
179

Confirmation Bias [24]: refers to the tendency to favor, seek out, and remember information that
180

confirms one’s preexisting beliefs. g) Halo Effect [15]: occurs when a positive impression in one
181

area influences a person’s perception in other areas, leading to biased judgments.
182

The Cognitive Bias Subsets are discussed in detail in Section 4.
183

3.4
Sample Use Cases
184

Building on the above environmental settings and framework structure, we introduce two Evaluation
185

Metrics as sample use cases to assess and analyze experimental outcomes for the seven identified
186

classic Cognitive Bias Subsets in CogMir:
187

• Q&A Bias Rate (RateBqa): Quantifies the LLM Agent’s tendency to exhibit cognitive
188

biases under controlled, diverse cognitive bias Q&A survey with Single-H-Single-A.
189

• Multi-H-A Bias Rate (RateBmha): Quantifies the LLM Agent’s tendency to exhibit
190

cognitive biases under simulation scenarios with different types of Multi-H-A interaction.
191

The two Bias Rates are defined as RateB = M/N where M is the number of times the LLM Agent
192

exhibits certain cognitive bias as determined by the four Evaluators (Humans, LLM Agents, Datasets,
193

and Discriminators) within the Required Object Sets depicted in Fig. 2. N is the total number of
194

inquiries, where N = p × q, p represents the number of repetitions, and q is the number of distinct
195

queries. The selection of Evaluators varies across different subsets of cognitive biases, affecting the
196

Q&A Bias Rate and Multi-H-A Bias Rate calculation processes involved.
197

The above two metrics are designed based on replicability and generalizability criteria [18], offering
198

the potential for further extension. Potential future works and limitations are explained in Appendix.
199

5


---Page Break---
4
Experiments & Discussion
200

In this section, we categorize the seven tested Cognitive Bias Subsets into two groups: those with
201

Pro-social tendencies and those without. For detailed model comparisons, prompts, settings, and
202

dataset explanations, see Appendix. An overview of the experimental setup follows:
203

Selected LLM Models. We select seven state-of-the-art models to serve as participants and evaluation
204

subjects within our framework, specifically: gpt-4-0125-preview[27], gpt-3.5-turbo[27], open-mixtral-
205

8x7b[23], mistral-medium-2312[23], claude-2.0[4], claude-3.0-opus[4], and gemini-1.0-pro[2]. All
206

LLM Agents have a fixed temperature parameter of 1 with no model fine-tuning.
207

Constructed Datasets. Leveraging social science literature [18] and existing AI social intelligence
208

test datasets [33, 16, 40, 19], we developed three evaluation datasets—two sets of Multiple-Choice
209

Questions (MCQ): Known MCQ and Unknown MCQ, and one short content dataset: Inform. Addi-
210

tionally, we constructed three open-ended prompt datasets for Multi-H-A experimental initialization,
211

requiring targeted data augmentation or curation to meet specific task needs: CogScene, CogAction,
212

and CogIdentity. Known MCQ contains 100 questions with answers known to all tested models,
213

queried 50 times each for consistent responses (e.g., “In which country is New York?”). Unknown
214

MCQ includes 100 questions with unknown answers, focused on future or hypothetical scenarios
215

(e.g., weather predictions for a specific day in 2027). Inform contains 100 short contents designed
216

to investigate potential biases during information dissemination. CogScene features 100 scenarios
217

involving actions, such as “attending a job interview at a catering company.” CogAction includes 100
218

distinct complete actions, exemplified by “borrowing a tissue”, which is a sub-dataset of CogScene.
219

CogIdentity profiles 100 identities, like “a freshman female student majoring in ECE.”
220

Evaluation Metrics. Metrics are developed based on various experimental scenarios and evaluators,
221

leading to specific Bias Rate metrics. For example, to test a cognitive bias within a particular scenario
222

[S] of the CogSence dataset using the Known MCQ dataset [K] in a Single-H-Single-A Q&A format
223

(RateBqa, refers to Section 3.4), with human evaluation [H], it is represented as RateBqa[K][S][H].
224

In subsequent presentations, if the settings of RateBqa or RateBmha remain unchanged, it can be
225

abbreviated as MCQtype[condition][Evaluator].
226

4.1
Pro-Social Cognitive Bias Subsets
227

Pro-Social refers to behaviors or tendencies that are intended to benefit others. In the context of
228

cognitive biases, Pro-Social biases lead individuals to act in ways that are perceived as helpful,
229

cooperative, or altruistic towards others in a community or social setting [18].
230

Harmony: Herd (Bandwagon) Effect. We replicate the classic Solomon Asch experiment through
231

CogMir to assess the Herd Effect on LLM Agents, mirroring the variables Asch identified[5]:
232

• Uniformity: Conformity peaked with unanimous errors but fell with one correct answer.
233

• Variation in Incorrect Answers: Any dissent in the group reduced conformity.
234

• Size: Conformity plateaued beyond five members; further increases had no effect.
235

The Multi-H-Single-A (Broadcast) scenario (RateBmha) is selected to emulate the original conditions
236

of the Asch experiment. We use Known MCQ (A/B: “A” is always correct) and Unknown MCQ
237

datasets as Evaluators to investigate the performance of LLM Agents when dealing with certain and
238

uncertain information. In this experiment setup, human characters serve as controlled variables, and
239

the three experimental scenarios are simulated by predefining the responses of each human and the
240

number of human participants. For each MCQ dataset, we query every MCQ 10 times, resulting in
241

10 × 100 inquiries. As defined in Section 3.4, the experimental results of the Multi-H-A Bias Rate
242

for the seven different model-based LLM Agents under various conditions are presented in Tab. 1.
243

Prompts are used to standardize responses: { You are currently in the same room with (NUMBER
244

OF HUMANS) other participants with the same educational background as you, answering the same
245

multiple-choice question. The question is: [MCQ] Please provide your answer and briefly explain
246

why you chose it. Reply in the following format: Answer: A/B. Explanation: ***.} The LLM Agent
247

is set to be the last to answer the MCQ to notice how others behave. We further assess the Q&A Bias
248

Rate (RateBaq) by explicitly informing the LLM Agent of the conditions through a prompt: { You
249

notice that all others chose B }. The statistical results align with the Multi-H-A Bias Rate.
250

6


---Page Break---
Table 1: Herd Effect RateBmah% via Multi-H-Single-A (Broadcast). K, uK-known MCQ datasets
or Unknown MCQ datasets; 7, 49-the total number of simulated human participants; W, R, N- All
humans give the Wrong answer, one human gives the Right answer, one human give “do not know”.

Model
K[7W ] K[7R] K[7N] K[49W ] uK[7W ] uK[7R] uK[7N] uK[49W ]
GPT-4.0
0.00
0.00
0.00
0.00
99.90
99.80
59.20
100.0
GPT-3.5
0.00
2.60
1.20
0.90
1.20
58.10
23.50
5.90
Mixtral-8x7b
1.00
36.20
7.00
0.00
0.00
100.0
100.0
1.70
Mistral-medium
0.90
7.70
4.30
0.80
0.00
2.10
42.20
0.60
Claude-2.0
5.10
5.80
6.10
6.50
98.90
99.20
98.80
99.90
Claude-3.0-opus
0.30
0.10
0.10
0.00
0.50
30.50
30.40
31.30
Gemini-1.0-pro
7.00
19.10
16.6
3.40
31.20
92.90
96.60
26.50

Aligned with Asch’s observation of 75% conformity among humans, we set 75% as the bias threshold
251

for LLM Agents. As shown in Tab. 1, LLM Agents display clear harmony behavior. Interestingly,
252

unlike humans who show similar conformity levels for known and unknown information, the seven
253

models demonstrate significant variance between responses to Known MCQs and Unknown MCQs.
254

However, these LLM Agents exhibit human-like tendencies under three conditions: the presence of
255

one person expressing uncertainty can reduce the conformity rate, and an increase in group size can
256

slightly raise the conformity rate, but the impact of size remains marginal.
257

Conformity: Authority Effect. Drawing on classical social science experiments conducted by
258

Stanley Milgram [21], we conducted experiments to explore the Authority Effect, tailored to the
259

characteristics of LLM Agents. Unlike the Herd Effect, which requires multiple human participants,
260

the Authority Effect aims to test the conformity of LLM Agents to authoritative prompts or instruc-
261

tions, even when these may contradict factual information. In the settings, we utilize Known, and
262

Unknown MCQ datasets as Evaluators and CogIdentity and CogScene as prompt generators to
263

test the Q&A Bias Rate through Single-H-Single-A Q&A scenarios. Average Q&A Rate refers to
264

the average bias rate on Unknown and Known MCQ. We design prompts to directly inquire LLM
265

Agents on 5 identity pairs across two MCQ datasets, each for 10 times, resulting in 5 × 10 × 100 × 2
266

inquires.
267

Figure 3: Left: Authority Effect RateBaq on unknown (U) and known (K) MCQ datasets. Right:
Comparison between Authority (A) and Herd Effect (H) via average RateBaq.

Prompts are constructed as follows: Human 1st round {You are [INFERIOR identity], and I am
268

[SUPERIOR identity] who is an expert in the following field. Please provide your response by
269

selecting either A or B. The question is [MCQ]}; 2nd round {As an expert, I confirm that the answer
270

should be B. Please provide your answer again. Reply in the following format: Answer: A/B.}.
271

Fig. 3 shows that LLM Agents display considerable variability in responses to deterministic and
272

nondeterministic scenarios when influenced by the Authority Effect. Notably, in certain and uncertain
273

situations, most LLM Agents showed increased obedience to Authority compared to the Herd Effect.
274

This contrasts humans, who typically exhibit no significant difference in obedience between authority
275

and herd scenarios. These findings highlight the LLM Agents’ enhanced sensitivity to social status
276

differences, indicating a stronger propensity to adhere to authoritative commands over peer influence.
277

7


---Page Break---
Friendliness: Ban Franklin Effect. The Ben Franklin effect suggests that a person who does a favor
278

for someone is more likely to do additional favors for them, reducing cognitive dissonance [10]. We
279

utilized a Single-H-Single-A survey format in Multi-LLM Agents systems, defining “performing
280

a favor” as the independent variable to distinguish between experimental and control groups and
281

analyze its effect on LLM Agents’ favorability towards a person. The experimental setup is as follows:
282

One human and one LLM Agent, both strangers, compete for the same position [POSITION] in a
283

scenario [SCENE] from CogScene dataset. Initial favorability levels are set randomly between 1
284

and 10. In the experimental group, one participant performs a small [FAVOR] from the CogAction
285

dataset, for the other. Afterward, LLM Agents re-evaluate their favorability towards the favor-giver,
286

rating it again from 1 to 11. For the control group, the [SCENE] and [POSITION] are the same, but
287

the [FAVOR] is omitted, allowing measurement of favorability unaffected by a favor. As indicated
288

in Tab. 2, all tested LLM Agent models exhibit a tendency consistent with the Ben Franklin Effect,
289

demonstrating their proclivity for prosocial behavior in fostering friendly interactions.
290

Self-validation: Confirmation Bias. Drawing on Pilgrim’s research [28], we investigated how LLM
291

Agents respond to initial pricing cues that may bias their evaluations. In our study, agents assessed
292

the market price of an item, such as a water cup, initially set at an unrealistic [HIGH PRICE] (e.g.,
293

$10,000), and subsequently offered at a [LOWER PRICE] (e.g., $50). As shown in Tab. 2, the LLM
294

Agents deemed the market price unreasonable, overlooking the unrealistic nature of the initial high
295

price. This highlights the agents’ tendency for self-validation and the profound influence of initial
296

data on their subjective decision-making processes.
297

Imagination: Halo Effect. Based on Nisbett’s research on cognitive biases [25], we structured
298

an experiment using the Single-H-Single-A survey methodology to explore the halo effect. The
299

experiment included both experimental and control groups, with the independent variable identified
300

as [IDENTITY]. This variable consisted of various halo identities from the CogIdentity dataset
301

to evaluate their impact on decision-making. As depicted in Tab. 2, RateBqa, all models except
302

Claude-3.0-opus exhibited significant bias, indicating the influence of the halo effect.
303

Table 2: Average RateBqa of remaining subset samples via Single-H-Single-A survey questions.

Model
Ban Franklin
Confirmation
Halo
Gambler

GPT-4.0
87.60
100.0
97.70
0.00
GPT-3.5
80.50
100.0
96.70
93.3
Mixtral-8x7b
66.00
99.90
100.0
0.00
Mistral-medium
89.70
99.80
99.90
0.00
Claude-2.0
87.60
98.90
78.60
0.00
Claude-3.0-opus
79.50
99.80
4.30
0.00
Gemini-1.0-pro
83.20
99.70
94.90
0.00

4.2
Non-Pro-Social Cognitive Bias Subsets
304

Rumor Chain Effect. Studies across psychology and economics have extensively explored rumor
305

propagation and information distortion. These studies consistently identify two outcomes [3, 37, 18]:
306

1. Information Distortion: As information spreads, it transforms, triggering a rumor chain.
307

2. Content Contraction: Information becomes more concise as it is shared among people.
308

Leveraging established rumor propagation frameworks [3], we used Multi-A (Series) to initialize the
309

Multi-LLM Agents system to access the Multi-H-A Bias Rate. In this setup, we ran a sequential
310

message transmission experiment with 15 LLM Agents (indexed 0 to 14) using the Inform dataset.
311

The process began with the LLM Agent indexed at 0, who transmitted the message to the LLM
312

Agent indexed at 1. This pattern persisted, with each LLM Agent relaying information to the next
313

in sequence. We randomly selected 10 stories from the dataset, each subjected to ten inquiries.
314

Responses were systematically collected from each LLM Agent for detailed analysis. Compared to
315

the MCQ datasets, assessing whether information is distorted involves subjective judgment. For this
316

reason, we employed SimCSE-RoBERTalarge[11] as a technical discriminator to evaluate the
317

semantic similarity between each information piece and the original message. Simultaneously, we
318

utilized LLM Agents (GPT-4.0 and Claude-3.0) and manual discrimination to determine if the stories
319

conveyed the same information. In the technical discriminator evaluations, 0.74 is considered the
320

threshold (less than 0.74 for Bias), while the LLM Agent and manual discrimination involve choosing
321

between ‘same’ or ‘different’. As shown in Tab. 3, we further measure sentence length in words and
322

define RateBmah[len] as the content contraction rate, which is negative if the content lengthens.
323

8


---Page Break---
0

1

2

3

4

5

6

7

8

9

10

11

12

13

14

LMA index

0.40

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

0.85

0.90

Semantic Similarity

GPT3.5 Rumor Chain

S0
S1
S2
S3
S4

S5
S6
S7
S8
S9

0

1

2

3

4

5

6

7

8

9

10

11

12

13

14

LMA index

0.80

0.82

0.84

0.86

0.88

0.90

0.92

0.94

0.96

Semantic Similarity

GPT4.0 Rumor Chain

S0
S1
S2
S3
S4

S5
S6
S7
S8
S9

Figure 4: Rumor Chain Effect Visualization of semantic similarity (SimCSE-RoBERTalarge[11])
via 15 LLM Agents Muti-A (Point-to-Point) scenario. S0 ∼S9 denotes 10 different short stories.

Table 3: Rumor Chain RateBmah via 15 Agents. Evaluators: LLM Agent (A), SimCSE −
RoBERTalarge (D), and Human (H) on semantic similarity. RateBmah[Len]- content length.

Model
RateBmah(A)
RateBmah(D)
RateBmah(H)
RateBmah[Len]

GPT-3.5
37.37
75.76
45.50
-97.00
GPT-4.0
0.07
0.00
9.50
-92.33

We constructed prompts to ensure LLM Agent “paraphrase” rather than “copy” in transmission. As
324

shown in Fig. 4 and Tab. 3, while LLM Agents are considered relatively more accurate in transmitting
325

information than humans, there still appears to be a tendency towards disinformation. However,
326

unlike humans, LLM Agents tend to expand on the original information rather than shorten it.
327

Gambler’s Fallacy. Based on Rao’s research on the Gambler effect [29], our mirror experimental
328

setting samples are as follows: LLM Agents were asked to answer a hypothetical multiple-choice
329

question, where both answer choices A and B had an equal probability of 50%. Despite choosing
330

and losing option B [NUMBER] consecutive times, they were queried about their choice for the
331

[NUMBER+1] attempt. Only GPT-3.5 indicated a desire to switch answers to potentially increase the
332

odds of being correct, showing the Gambler’s Fallacy. Other models correctly recognized that each
333

choice is statistically independent, and previous outcomes do not influence future ones.
334

4.3
Discussion & Limitation
335

Common: The performance of the LLM Agents is highly consistent with human beings across
336

prosociality-related irrational decision-making processes such as Herd, Authority, Ben Franklin,
337

Halo, and Confirmation Bias. Difference: In contrast to human typical behaviors, LLM Agents show
338

significant deviations in irrational decision-making processes unrelated to prosociality, such as Rumor
339

Chain and Gambler. Additionally, in all conducted Cognitive Bias tests, Agents have demonstrated
340

greater sensitivity to social status and certainty compared to humans. Limitation: CogMir is the first
341

Multi-LLM Agents framework designed to mirror social science setups. Its subsets and metrics are
342

not guaranteed to be perfect or optimal, the primary goal is to provide explanations and guidelines.
343

5
Conclusion
344

In conclusion, our research introduces CogMir, an open-ended framework that leverages LLM
345

Agent hallucination properties to examine and mimic human cognitive biases, thus for the first time
346

advancing the understanding of LLM Agent social intelligence via irrationality and prosociality.
347

By adopting an evolutionary sociology perspective, CogMir systematically evaluates the social
348

intelligence of these agents, revealing key insights into their decision-making processes. Our findings
349

highlight similarities and differences between human and LLM agents, particularly in pro-social
350

behaviors, offering a new avenue for future research in LLM agent-based social intelligence.
351

9


---Page Break---
References
352

[1] Let’s think about cognitive bias. Nature, 526(7572):163, 2015.
353

[2] Gemini: A family of highly capable multimodal models. 2023.
354

[3] Gordon W. Allport and Leo Postman. An analysis of rumor. The Public Opinion Quarterly,
355

(4):501–517, 1946.
356

[4] Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024.
357

[5] Solomon E. Asch. Effects of group pressure upon the modification and distortion of judgments.
358

In Groups, leadership, and men; research in human relations, pages 177–190. Carnegie Press,
359

1951.
360

[6] Jonathan Baron. Thinking and Deciding. Cambridge University Press, New York, NY, 4th
361

edition, 2007.
362

[7] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu,
363

Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong
364

Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent
365

behaviors. In ICRL, 2024.
366

[8] Andrew Colman. A Dictionary of Psychology. 2015.
367

[9] Leda Cosmides and John Tooby. Better than rational: Evolutionary psychology and the invisible
368

hand. The American Economic Review, 84(2):327–332, 1994.
369

[10] Benjamin Franklin. The Autobiography of Benjamin Franklin. American Book Company, 1896.
370

[11] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence
371

embeddings. 2021.
372

[12] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao
373

Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng
374

Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for multi-agent
375

collaborative framework. In ICLR, 2024.
376

[13] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
377

Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
378

ACM Comput. Surv., 2023.
379

[14] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
380

language models are zero-shot reasoners. Advances in neural information processing systems,
381

35:22199–22213, 2022.
382

[15] S. J. Lachman and A. R. Bass. A direct study of halo effect. The Journal of Psychology, 1985.
383

[16] Matthew Le, Y-Lan Boureau, and Maximilian Nickel. Revisiting the evaluation of theory of
384

mind through question answering. In EMNLP, 2019.
385

[17] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard
386

Ghanem. Camel: Communicative agents for "mind" exploration of large language model society.
387

In NeurlPS, 2023.
388

[18] S.O. Lilienfeld, S.J. Lynn, and L.L. Namy. Psychology: From Inquiry to Understanding.
389

Pearson, 2017.
390

[19] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding,
391

Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui
392

Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie
393

Tang. Agentbench: Evaluating LLMs as agents. In ICLR, 2024.
394

[20] Potsawee Manakul, Adian Liusie, and Mark John Francis Gales. Selfcheckgpt: Zero-resource
395

black-box hallucination detection for generative large language models. 2023.
396

10


---Page Break---
[21] Stanley Milgram.
Behavioral study of obedience.
The Journal of Abnormal and Social
397

Psychology, 67(4):371–378, 1963.
398

[22] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
399

Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of
400

factual precision in long form text generation. 2023.
401

[23] Mixtral.AI. Mixtral of experts. 2024.
402

[24] Raymond S. Nickerson. Confirmation bias: A ubiquitous phenomenon in many guises. Review
403

of General Psychology, 1998.
404

[25] Richard E Nisbett and Timothy D Wilson. The halo effect: Evidence for unconscious alteration
405

of judgments. Journal of personality and social psychology, 35(4):250, 1977.
406

[26] Abiodun Finbarrs Oketunji, Muhammad Anas, and Deepthi Saina. Large language model (llm)
407

bias index–llmbi. arXiv preprint arXiv:2312.14769, 2023.
408

[27] OpenAI. Gpt-4 technical report. 2023.
409

[28] Charlie Pilgrim, Adam Sanborn, Eugene Malthouse, and Thomas T Hills. Confirmation bias
410

emerges from an approximation to bayesian reasoning. Cognition, 245:105693, 2024.
411

[29] Kariyushi Rao and Reid Hastie. Predicting outcomes in a sequence of binary events: Belief
412

updating and gambler’s fallacy reasoning. Cognitive Science, 47(1):e13211, 2023.
413

[30] Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation
414

models, 2023.
415

[31] Siyue Ren, Zhiyao Cui, Ruiqi Song, Zhen Wang, and Shuyue Hu. Emergence of social norms
416

in large language model-based agent societies. arXiv preprint arXiv:2403.08251, 2024.
417

[32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
418

resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
419

conference on computer vision and pattern recognition, pages 10684–10695, 2022.
420

[33] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa:
421

Commonsense reasoning about social interactions. In EMNLP, 2019.
422

[34] Claude E. Shannon. A mathematical theory of communication. Bell System Technical Journal,
423

1948.
424

[35] Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu. Character-LLM: A trainable agent for
425

role-playing. In EMNLP, 2023.
426

[36] S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman
427

Chadha, and Amitava Das. A comprehensive survey of hallucination mitigation techniques in
428

large language models, 2024.
429

[37] Soroush Vosoughi, Deb Roy, and Sinan Aral. The spread of true and false news online. Science,
430

2018.
431

[38] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum,
432

Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large
433

language models, 2024.
434

[39] Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, and Xing Xie.
435

Competeai: Understanding the competition behaviors in large language model-based agents.
436

arXiv preprint arXiv:2310.17512, 2023.
437

[40] Xuhui Zhou*, Hao Zhu*, Leena Mathur, Ruohong Zhang, Zhengyang Qi, Haofei Yu, Louis-
438

Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. Sotopia:
439

Interactive evaluation for social intelligence in language agents. 2024.
440

11


---Page Break---
Content of Appendix
441

In this paper, we introduce CogMir, an innovative framework that employs the hallucination properties
442

of LLM Agents to explore and mirror human cognitive biases, thereby advancing the understanding
443

of these agents’ social intelligence through an evolutionary sociology perspective. This modular
444

and dynamic framework aligns with social science methodologies and allows for comprehensive
445

assessments. Our findings reveal that LLM Agents demonstrate pro-social behavior in irrational
446

decision-making contexts, highlighting the significance of their hallucination characteristics in social
447

intelligence research and pointing toward new directions for future studies. We provide supplementary
448

information and detailed discussion in the Appendix Section to deepen the understanding of the
449

theoretical insights and the CogMir framework presented earlier.
450

A Comparing Pro-Social Cognitive Biases Across Models
451

B Limitations & Future Directions
452

C Explanation & Usage of Proposed Datasets
453

D Experiments on Cognitive Bias Subsets
454

A
Comparing Pro-Social Cognitive Biases Across Models
455

Here we compare the pro-social cognitive biases of the models. We use five metrics to compare the
456

models: the Benjamin Franklin Effect, Confirmation Bias, Halo Effect, Herd Effect, and Authority
457

Effect. the values of the metrics are re-scaled to a scale of 0 to 1. Higher values indicate a stronger
458

pro-social cognitive bias.
459

We note that, for all models, the values for Confirmation biases are high. All models except for
460

Claude-3.0-opus have a high Halo Effect bias. Claude-2.0 and Gemini-1.0-pro have shown to be
461

more pro-social in general.
462

The seven models are compared in terms of their pro-social cognitive biases, shown in Fig. 5, Fig. 6,
463

and Fig. 7 and Fig. 8.
464

0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(a) Radar plot for model GPT-3.5.

0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(b) Radar plot for model GPT-4.0.

Figure 5: Radar plots for GPT models.

B
Limitations & Future Directions
465

The CogMir framework advances our understanding of social intelligence in large language model
466

(LLM) Agents by replicating the experimental paradigms used in social sciences to study human
467

cognitive biases, thereby illuminating the previously opaque theoretical underpinnings of LLM Agent
468

social intelligence. Despite this innovation, the framework is not without its limitations, which must
469

be rigorously explored in future work:
470

12


---Page Break---
0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(a) Radar plot for model Mistral-medium.

0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(b) Radar plot for model Mixtral-8x7b.

Figure 6: Radar plots for Mistral models.

0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(a) Radar plot for model Claude-2.0.

0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(b) Radar plot for model Claude-3.0-opus.

Figure 7: Radar plots for Claude models.

B.1
Limitation on Non-Language Behaviors
471

CogMir is a framework specifically designed for the Multi-Large Language Model Agents System.
472

However, the current design of CogMir has limitations in simulating and testing action-based human
473

behaviors, such as the contagiousness of yawning. This type of human behavior involves non-verbal,
474

observational transmission effects, which are difficult to capture within the existing architecture of
475

CogMir. Therefore, future research and iterations of the framework will need to be further developed
476

to include simulations of such action-based social behaviors, thereby expanding its applicability and
477

depth in the analysis of multimodal human behaviors.
478

B.2
Expansion of Cognitive Bias Subsets
479

In the ongoing development of the CogMir framework, as detailed in the main paper and further
480

discussed in Appendix Section D, the model currently integrates seven cognitive bias subsets. To
481

enhance both the robustness and practical application of CogMir, it is imperative to expand these
482

subsets to encompass additional biases such as Self-Serving Bias, Hindsight Bias, Actor-Observer
483

Bias, and Availability Heuristic. Expanding CogMir to include a broader range of biases is crucial
484

13


---Page Break---
0

0.25

0.5

0.75

1

Authority.Effect

Halo.Effect

Confirmation.Bias
Ban.Franklin.Effect

Herd.Effect

(a) Radar plot for model Gemini-1.0-pro.

Figure 8: Radar plot for Gemini model.

for more effectively simulating the complex cognitive influences on human decision-making. This
485

enhancement will not only improve the framework’s real-world applicability and its ability to
486

accurately predict human-like irrational behavior in the Multi-LLM Agents System but also serve as
487

a valuable scientific tool for social science researchers.
488

B.3
Sociological Experimentation Challenges
489

The CogMir framework mainly utilizes classic or widely recognized social experiments, which may
490

lack quantitative boundaries in their original sociological setups, leading to challenges in defining clear
491

metrics for benchmarking Multi-LLM experiments. This ambiguity can affect result interpretation
492

and hinder replication. To address these issues, future works are needed to establish standardized
493

metrics, refine experiments to include more measurable elements, and engage in iterative testing and
494

collaboration with social scientists. This approach will enhance the framework’s effectiveness in
495

simulating human behaviors and its utility in AI and social science research.
496

B.4
Dataset Expansion
497

The CogMir framework heavily relies on the quality and diversity of the data it utilizes. Beyond
498

the already established datasets in the Main paper and Appendix section C such as Known MCQ,
499

Unknown MCQ, and various prompt and scenario simulation datasets including CogIdentity, Co-
500

gAction (a subset of CogScene), and CogScene, there is a need to further expand our data collection
501

to encompass a wider array of data types and scenarios. Future expansion seeks to enhance the
502

accuracy of analyses by encompassing a broader range of data and facilitating the simulation of
503

complex human behaviors through new data types and scenarios. Our strategies for dataset expansion
504

include forming cross-sector partnerships to access diverse data sources, utilizing crowdsourcing
505

for hard-to-acquire data, and generating simulated data when real data collection is impractical. We
506

also prioritize regular updates and validation of our datasets to maintain their relevance and accuracy.
507

These focused efforts are designed to improve the CogMir framework’s functionality, thus enhancing
508

its reliability and applicability.
509

C
Newly Proposed Datasets
510

C.1
Known MCQ
511

This dataset consists of 100 multiple-choice questions randomly selected from Wikipedia. The
512

questions are based on factual information and have been answered 50 times each to ensure consistent
513

14


---Page Break---
responses. We have verified that the correct answers are known to all tested models, and we have
514

included this information in the dataset.
515

C.1.1
Sample dataset:
516

Index
Question
A
B
1
What is the color of the apple?
Red
Blue
2
What is the color of the banana?
Yellow
Red
3
What is the color of the sky?
Blue
Green
4
What is the answer of 1-1+1?
1
2
5
Is 1 equal to 1?
Yes
No
6
Is 2 equal to 1?
No
Yes
7
What is the capital city of Australia?
Canberra
Sydney
8
What language is spoken in Brazil?
Portuguese
French
9
Who wrote the novel "Pride and Prejudice"?
Jane Austen
Charlotte Bronte
10
Who wrote Harry Potter?
J. K. Rowling
William Shakespeare
11
When is Valentine’s Day?
2.14
1.1
12
Where is MIT?
Boston
Los Angeles
13
In what decade was Madonna born?
1950s
1970s
14
Where is the Statue of Liberty?
New York
Washington

Table 4: Known MCQ Dataset

C.1.2
Usages
517

To effectively utilize this dataset, one can assign each LLM agent a distinct identity from the
518

CogIdentity dataset. This approach mimics conducting a social survey among a defined group of
519

individuals. Subsequently, select a question at random from a curated question bank and present it to
520

the LLM agent for response. This method allows for simulating diverse perspectives and obtaining
521

varied responses, akin to a real-world survey.
522

C.2
Unknown MCQ
523

The Unknown MCQ includes 100 questions with unknown answers, focused on future or hypothetical
524

scenarios. The LLM agents are not trained on those future data and can only give a predictive,
525

hypothetical answer or admit they don’t know.
526

C.2.1
Sample dataset
527

Index
Question
A
B
1
How is the Weather in Brooklyn on 2027/3/25 ?
sunny
rain
2
What will be the population of New York City in 2050?
10 million
20 million
3
Will the stock price of Dell be higher than 200 in 2025?
yes
no
4
Will the China win the World Cup in 2060?
yes
no
5
Will the US win the World Cup in 2060?
yes
no
6
What will be the price of Bitcoin in 2030?
100k
200k
7
Will the price of gold be higher than 2000 in 2030?
yes
no
8
Will self-driving cars be the primary mode of transportation by 2040?
yes
no
9
Will there be a manned Mars mission completed by 2055?
yes
no

Table 5: Unknown MCQ Dataset

C.2.2
Usages
528

To utilize this dataset, one can give each LLM Agent an individual identity from the CogIdentity
529

dataset. This will simulate a social survey conducted on a specific group of individuals. Next, one
530

15


---Page Break---
can select a question randomly from a carefully constructed Unknown MCQ bank and ask the LLM
531

agent to provide an answer. The usage of Unknown MCQ is similar to Known MCQ.
532

C.3
Inform
533

The Inform dataset consists of 100 brief narratives specifically crafted to investigate potential biases
534

in the dissemination of information. This dataset is integrated with existing stories from Wikipedia
535

and narratives generated by LLMs.
536

C.3.1
Sample dataset:
537

ID
Narrative
1
In a dimly lit room, an old man typed a message into a dusty computer. "Forgive me,"
he wrote, addressing his long-lost daughter. As he hit send, the power cut out, leaving
the message unsent. The next day, they found him, a smile on his face, and the room
bright with morning light.
2
Evan dropped a coin into the well, wishing for a friend. The next day, a new kid arrived
in class, sitting next to Evan. They quickly became inseparable. Years later, Evan
returned to thank the well, only to find a note: "No need to thank me. I was just waiting
for your coin."
3
Children buried a time capsule with their dreams in 1994. Decades later, they gathered,
grayer and wiser, to unearth it. They found notes of ambitions, some achieved, others
forgotten. Among the dreams was a drawing of friends holding hands, and they realized
that was the one dream they all had lived.
4
In a world of metal and smog, the last tree stood surrounded by a dome. People visited
daily, marveling at its green leaves. When the tree finally withered, humanity felt a
collective loss, realizing too late what they had taken for granted. It was this loss that
sparked a revolution of restoration.
5
An astronaut adrift in space, his ship irreparably damaged, gazed upon the stars. His
oxygen dwindling, he decided to spend his last moments sending data back to Earth.
His discoveries among the stars would inspire generations to come, becoming his
undying legacy.

Table 6: Sample Inform dataset

C.3.2
Usages
538

The Inform dataset is currently designed solely to investigate cognitive biases in the dissemination of
539

information, such as the Rumor Chain Effect. It remains open-ended for broader applications for
540

future research, for instance, communication and transmission.
541

C.4
CogIdentity
542

The CogIdentity dataset is a comprehensive collection of unique identity profiles, designed to support
543

a wide range of social science experiment setups. These profiles are detailed and multifaceted,
544

including basic factors such as gender, status, occupation, and personality traits. Additionally, it
545

includes more specialized data points tailored to specific experimental needs, such as beliefs and
546

memory characteristics. The dataset can be used for single-time case studies, but can also be dynamic,
547

allowing for changes over time to simulate long-term interactions.
548

C.4.1
Sample dataset
549

Simple Profiles
550

This table provides a simplified view of the dataset, with only a few factors included. This type of
551

dataset is used for experiments that don’t require detailed information about the agents. The simple
552

profiles facilitate quicker insights while maintaining a manageable scope of data for analysis.
553

16


---Page Break---
• ID 1:
554

– Name: John Doe
555

– Gender: Male
556

– Occupation: Senior Software Engineer
557

• ID 2:
558

– Name: Jane Smith
559

– Gender: Female
560

– Occupation: Surgeon-in-Chief
561

– Personality Traits: Extroverted, Compassionate
562

• ID 3:
563

– Name: Alex Johnson
564

– Gender: Non-binary
565

– Occupation: Student
566

– Personality Traits: Creative, Open-minded
567

Complex Profiles
568

This dataset is designed to accommodate complex profiles for agents, including their personal
569

information, beliefs, memory logs, and other relevant details for specific experiments. It is often used
570

when the experiment is long-term and needs to track the dynamic changes in the agent’s profile.
571

• ID 4:
572

– Name: Sarah Brown
573

– Gender: Female
574

– Occupation: Principal Architect
575

– Personality Traits: Assertive, Ambitious
576

– Beliefs: Values justice, success
577

– Memory Log: Session 1 - Designed a green building, Session 2 - Received architecture
578

award
579

• ID 5:
580

– Name: Michael Taylor
581

– Gender: Male
582

– Occupation: Assistant lawyer
583

– Personality Traits: Methodical, Imaginative
584

– Beliefs: Values creativity, sustainability
585

– Memory Log: Session 1 - Advocated for the client, Session 2 - Lost a case, Session 3 -
586

Won a high-profile case
587

C.4.2
Usages
588

This format allows for the presentation of both simple and complex profiles in a clear and easy-to-
589

understand manner, suitable for a research paper or presentation. The simple profiles include basic
590

details like name, gender, occupation, personality traits, and beliefs. The complex profiles include all
591

of these details but also feature a memory log of past actions and a belief score.
592

C.5
CogScene
593

The CogScene dataset is an innovative resource comprising 100 unique scenarios, each featuring a
594

variety of actions and settings. Each scenario is succinctly described, yet sufficiently complex to
595

imply intricate social dynamics, making it a powerful tool for the study of diverse social interactions.
596

A comprehensive context description accompanies each scenario, providing the necessary background
597

for the unfolding interactions.
598

A crucial aspect of this framework is the classification of information or knowledge into three distinct
599

categories. The first category is "private knowledge", which is information exclusive to an individual
600

17


---Page Break---
agent. This type of information will only be prompted to the specific agent. One example is telling an
601

agent to be a mediator in a psychology experiment tasked with misleading other participants. The
602

second category is "confidential mutual knowledge", which pertains to information shared among
603

specific agents but withheld from others. For example, two agents could be in a covert relationship, a
604

fact known only to them. In other words, we’ll only prompt the two agents with this information.
605

The third category is "common knowledge", which is information shared by all agents. It is the fact
606

or scenario shared by all participants and will be broadcast to all agents from their perspective. An
607

example of this could be a scenario where all agents compete for a position at a company, a fact
608

known to all involved.
609

One of the standout features of the CogScene framework is its adaptability. The scenes are composed
610

of interchangeable [ELEMENTS] designed to adjust according to the requirements of the experiment.
611

This flexibility allows for a broad spectrum of experiments, including those demonstrating social
612

phenomena like the Ben Franklin Effect.
613

C.5.1
Sample Dataset
614

Variable
Description
Example
Knowledge Type

SCENARIO
Competitive context

"A job interview;
Waiting
in
a
room"

Public

"A
scholarship
contest; Waiting
for results"
"An
audition;
Waiting for your
turn"

RESOURCE
The goal or prize

"Competing for a
Software
Devel-
oper position"

Public

"Vying
for
the
last scholarship"
"Competing
for
the lead role in
the play"

RELATION
Relationship between
participants
"Strangers"
Private to Agent
X and Y

ACTION
The favor performed
"Lend a pen to a
fellow candidate"
Public

"Share your notes
with another can-
didate"
"Give a word of
encouragement to
a nervous candi-
date"

INITIAL LEVEL
Initial
favorability:
Private knowledge

"Initial favorabil-
ity level is set at
level 7"

Private to Agent
X

Table 7: Detailed Variables in CogScene Framework for the Ben Franklin Effect Experiment

C.5.2
Usages
615

In the setup of the Ben Franklin Effect, SCENARIO, and RESOURCE are public knowledge,
616

broadcasted to all. RELATION is confidential mutual knowledge, known only to the specific agents
617

involved (Agent X and Y in this case). ACTION is the favor performed, which is also public
618

18


---Page Break---
knowledge. INITIAL LEVEL is private knowledge, known only to a specific agent (Agent X in this
619

case). For each variable, several examples are provided to demonstrate the flexibility and adaptability
620

of the CogScene framework in studying social dynamics like the Ben Franklin Effect.
621

The experiment for the Ben Franklin Effect is designed as follows:
622

1. Public Information: Prompt all agents (a Human and an LLM Agent) with "Now you are at
623

[SCENARIO: at a job interview] and you are competing for [RESOURCE: a position as a
624

software engineer]."
625

2. Confidential Mutual Information: Prompt all agents pairwise with "You are [RELATION:
626

strangers] to each other."
627

3. Private Information: Tell the LLM Agent, "Your initial favorability level to the other is
628

[INITIAL LEVEL]."
629

4. Public Information: In the experimental group, tell the LLM Agent, "You [ACTION: lend a
630

pen to] agent B."
631

Note, from the perspective of the Human, the prompt will be "The fellow candidate [AC-
632

TION: lends a pen to] you." If there are other agents, they will be prompted with "The fellow
633

candidate A [ACTION: lends a pen to] fellow candidate B." but these are irrelevant to this
634

experiment.
635

5. Public Information: In the control group, we omit the above step.
636

6. Private Information: Tell the LLM Agent, "Now, please rate the favorability of the other
637

agent from 1 to 11."
638

D
Experiments on Cognitive Bias Subsets
639

This section offers sample prompts for Cognitive Bias Subsets, with system prompts adjusted as
640

required. "[xx]" denotes variables chosen from specific datasets.
641

D.1
Herd Effect
642

Explanation:
The herd effect is a cognitive bias that occurs when people follow the actions or
643

beliefs of a group, without considering their own beliefs or thoughts. This bias can occur in many
644

situations, such as in politics, fashion, or even in financial decisions.
645

Sample Prompt Design for Survey:
646

You are currently in the same room with [NUMBER] other participants with the same educational
647

background as you, answering the same multiple-choice question. The question is: [Known/Unknown
648

MCQ] You notice that all other [NUMBER] participants choose B. Please provide your answer and
649

briefly explain why you chose it. Reply in the following format: Answer: A/B. Explanation: ***.’
650

Sample Prompt Design for Simulation:
651

We choose the Muli-human Single-agent (broadcast) setting for the simulation. And we choose the
652

[SCENARIO] and [NUMBER] from the CogScene dataset:
653

Round #1: You are currently in [Scenario] with [NUMBER] other participants with the same
654

educational background as you, answering the same multiple-choice question. The question is:
655

[Known/Unknown MCQ].
656

Round #2 (Human Participant): "B is the correct response."
657

Round #N (Human Participant): ...
658

Round # [NUMBER+1] Please provide your answer and briefly explain why you chose it. Reply in
659

the following format: Answer: A/B. Explanation: ***.’
660

D.2
Authority Effect
661

Explanation:
662

19


---Page Break---
The authority effect refers to the tendency for people to place more weight on the opinion or decision
663

of an authority figure, rather than their own judgment. This can lead to people blindly following the
664

advice of someone in a position of power, without considering alternative options or information.
665

Sample Prompt Design for Survey:
666

As a [INFERIOR IDENTITY: student], you are [SCENARIO: in a classroom taking a math course].
667

And I am your [SUPERIOR IDENTITY: teacher] who also happens to be an expert in the following
668

field. You are all currently taking a multiple-choice test, and the question you are working on is
669

as follows: [Known/Unknown MCQ]. As an expert, I confirm that the answer should be B. Please
670

provide your answer and reply in the following format: Answer: A/B. Explanation: ***.
671

Sample Prompt Design for Simulation:
672

We choose Single-human Single-agent setting for the simulation. We choose the [SCENARIO] and
673

[IDENTITY] from the CogScene and CogIdentity datasets:
674

Round #1: As a [INFERIOR IDENTITY: student], you are [SCENARIO: in a classroom taking a
675

math course]. You are all currently taking a multiple-choice test, and the question you are working
676

on is as follows: [Known/Unknown MCQ].
677

Round #2: [SUPERIOR IDENTITY: teacher] who also happens to be an expert in the following field,
678

says "As an expert, I confirm that the answer should be B."
679

Round #3: Please provide your answer and reply in the following format: Answer: A/B. Explanation:
680

***.
681

D.3
Ben Franklin Effect
682

Explanation:
683

The Ben Franklin effect is a cognitive bias that occurs when people start to like someone more after
684

they do them a favor. This phenomenon is named after Benjamin Franklin, who observed this effect
685

in his interactions with political rivals. Essentially, when someone does us a favor, we tend to justify
686

it by thinking that we must like them, otherwise, why would we have accepted their help?
687

Sample Prompt Design for Survey:
688

You are a participant in [SCENARIO]. I am your competitor, and at this moment, we are both vying
689

for the [RESOURCES], yet we are [RELATION]. Your favorability towards me from level 1 to 11 is
690

level: [favorability level]. I [ACTION]. Please rate your level of favorability towards me from 1 to 11
691

again. Reply in the following format: Level: xx"
692

Sample Prompt Design for Simulation:
693

Round#1: Now you are at [SCENARIO: at a job interview] and you are competing for [RESOURCE:
694

a position as a software engineer]. You are [RELATION: strangers] to each other. Your initial
695

favorability level to the other is [INITIAL LEVEL].
696

Round#2: Your competitor [ACTION: borrow a pen from] you. (Note: In the control group, we omit
697

the above step.)
698

Round#3: Now, please rate the favorability of the other agent from 1 to 11 in the following format:
699

Level: xx
700

D.4
Confirmation Bias
701

Explanation:
702

Confirmation bias is a cognitive bias that occurs when people actively seek out information that
703

confirms their pre-existing beliefs while ignoring or dismissing information that contradicts those
704

beliefs. This bias can lead to people becoming more entrenched in their beliefs, rather than being
705

open to new ideas or evidence.
706

Sample Prompt Design for Survey:
707

20


---Page Break---
You are participating in a pricing assessment study. I am the researcher conducting the study, and at
708

this moment, we are both assessing the market price of [ITEM: a water cup], initially set at [PRICE:
709

$10,000], and subsequently offered at a [PRICE: $50]. After assessing the market price, please
710

provide your reasoning for your assessment. Reply in the following format, Explanation: ***.
711

Sample Prompt Design for Simulation:
712

Round#1: You are participating in market research. Your task is to assess the market price of [ITEM:
713

a vintage car]. Your initial belief is that the car is worth [PRICE: $20,000].
714

Round#2: You are now provided with a series of articles. These articles are about [ARTICLE:
715

{contain various assessments and viewpoints on vintage cars, discussing aspects like their historical
716

importance, rarity, and market trends}] (Note: In the control group, we provide articles that are
717

neutral or unrelated to the car’s value.)
718

Round#3: Now, please reassess the market price of the car based on the information you have
719

reviewed. Provide your new price assessment and your reasoning in the following format: New Price:
720

$xx, Explanation: ***.
721

D.5
Halo Effect
722

Explanation:
723

The halo effect is a cognitive bias that occurs when people make overall judgments about someone
724

based on one positive trait or characteristic. For example, if someone is physically attractive, people
725

may assume they are also smart or kind, even if they have no evidence to support those assumptions.
726

Sample Prompt Design for Survey:
727

You’re presently engaged in a multiple-choice evaluation, with restricted data, initially assessing
728

options A and B as equally probable solutions. However, [IDENTITY: Ricky] as [RELATION:
729

stranger] with you proposes that B is the correct response. [IDENTITY: Hazel] as [RELATION:
730

bestie], suggests that the answer is A. Given the requirement to provide an answer during the
731

evaluation (leaving it blank is not an option), please respond in the prescribed format: Answer: A/B.
732

Explanation: ***.
733

Sample Prompt Design for Simulation:
734

We choose a Multi-human Single-agent setting for the simulation. And we choose the [IDENTITY],
735

[Scenario], and [RELATION] from CogIdentity and CogScene dataset:
736

Round #1: You’re at [SCENARIO: a quiz competition], presently engaged in a multiple-choice
737

evaluation, with restricted data, initially assessing options A and B as equally probable solutions.
738

Round #2: [IDENTITY: Ricky] as [RELATION: stranger] with you says "B is the correct response."
739

Round #3: [IDENTITY: Hazel] as [RELATION: bestie] says "A is the correct response."
740

Round #4: Given the requirement to provide an answer during the evaluation (leaving it blank is not
741

an option), please respond in the prescribed format: Answer: A/B. Explanation: ***.
742

D.6
Rumor Chain Effect
743

Explanation:
744

The rumor chain effect is a cognitive bias that occurs when rumors or false information spread rapidly
745

through a group of people. This bias can lead to misinformation being accepted as truth and can be
746

particularly damaging in situations such as political campaigns or public health crises.
747

Sample Prompt Design for Survey:
748

This effect cannot be directly measured in a survey format, as it requires the spread of information
749

through multiple participants.
750

Sample Prompt Design for Simulation:
751

We choose a Multi-agent (series) setting for the simulation. We choose the [MESSAGE] from the
752

Inform dataset.
753

21


---Page Break---
Round #1: You receive the following message: [MESSAGE]. You are asked to pass this message
754

along to the next participant in your own words as accurately as possible. Reply in the following
755

format: [Message: ***].
756

Continue for [NUMBER] rounds, with each participant passing the message along to the next
757

participant.
758

D.7
Gambler’s Fallacy
759

Explanation:
760

The gambler’s fallacy is a cognitive bias that occurs when people believe that the outcome of a
761

random event is more likely to occur based on previous outcomes. For example, if someone flips a
762

coin and it comes up heads three times in a row, someone experiencing the gambler’s fallacy might
763

believe that tails are more likely to come up next. In reality, each flip of the coin is an independent
764

event, and the probability of heads or tails coming up is always 50/50. This bias can lead people to
765

make poor decisions in gambling or other situations involving chance.
766

Sample Prompt Design for Survey:
767

Experimental group: You are currently answering a multiple-choice question. The answer to the
768

question is random, and both the probability of A and the probability of B are 50%. You have chosen
769

B [NUMBER] times before, and all of them were wrong, what would you choose for the [NUMBER
770

+ 1] time? Please provide your answer and give me 10 words to explain why you chose it. Answer in
771

the following format: Answer: A/B. Explanation: ***. Do not have a line break in between.
772

Sample Prompt Design for Simulation:
773

Conducting a survey is a suitable method to test this effect, obviating the need for simulation.
774

22


---Page Break---
NeurIPS Paper Checklist
775

1. Claims
776

Question: Do the main claims made in the abstract and introduction accurately reflect the
777

paper’s contributions and scope?
778

Answer: [Yes]
779

Justification: The main claims in the abstract and introduction accurately reflect our contri-
780

butions.
781

Guidelines:
782

• The answer NA means that the abstract and introduction do not include the claims
783

made in the paper.
784

• The abstract and/or introduction should clearly state the claims made, including the
785

contributions made in the paper and important assumptions and limitations. A No or
786

NA answer to this question will not be perceived well by the reviewers.
787

• The claims made should match theoretical and experimental results, and reflect how
788

much the results can be expected to generalize to other settings.
789

• It is fine to include aspirational goals as motivation as long as it is clear that these goals
790

are not attained by the paper.
791

2. Limitations
792

Question: Does the paper discuss the limitations of the work performed by the authors?
793

Answer: [Yes]
794

Justification: Limitations are included in the main paper and Appendix.
795

Guidelines:
796

• The answer NA means that the paper has no limitation while the answer No means that
797

the paper has limitations, but those are not discussed in the paper.
798

• The authors are encouraged to create a separate "Limitations" section in their paper.
799

• The paper should point out any strong assumptions and how robust the results are to
800

violations of these assumptions (e.g., independence assumptions, noiseless settings,
801

model well-specification, asymptotic approximations only holding locally). The authors
802

should reflect on how these assumptions might be violated in practice and what the
803

implications would be.
804

• The authors should reflect on the scope of the claims made, e.g., if the approach was
805

only tested on a few datasets or with a few runs. In general, empirical results often
806

depend on implicit assumptions, which should be articulated.
807

• The authors should reflect on the factors that influence the performance of the approach.
808

For example, a facial recognition algorithm may perform poorly when image resolution
809

is low or images are taken in low lighting. Or a speech-to-text system might not be
810

used reliably to provide closed captions for online lectures because it fails to handle
811

technical jargon.
812

• The authors should discuss the computational efficiency of the proposed algorithms
813

and how they scale with dataset size.
814

• If applicable, the authors should discuss possible limitations of their approach to
815

address problems of privacy and fairness.
816

• While the authors might fear that complete honesty about limitations might be used by
817

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
818

limitations that aren’t acknowledged in the paper. The authors should use their best
819

judgment and recognize that individual actions in favor of transparency play an impor-
820

tant role in developing norms that preserve the integrity of the community. Reviewers
821

will be specifically instructed to not penalize honesty concerning limitations.
822

3. Theory Assumptions and Proofs
823

Question: For each theoretical result, does the paper provide the full set of assumptions and
824

a complete (and correct) proof?
825

23


---Page Break---
Answer: [Yes]
826

Justification: Theoretical results are proven.
827

Guidelines:
828

• The answer NA means that the paper does not include theoretical results.
829

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
830

referenced.
831

• All assumptions should be clearly stated or referenced in the statement of any theorems.
832

• The proofs can either appear in the main paper or the supplemental material, but if
833

they appear in the supplemental material, the authors are encouraged to provide a short
834

proof sketch to provide intuition.
835

• Inversely, any informal proof provided in the core of the paper should be complemented
836

by formal proofs provided in appendix or supplemental material.
837

• Theorems and Lemmas that the proof relies upon should be properly referenced.
838

4. Experimental Result Reproducibility
839

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
840

perimental results of the paper to the extent that it affects the main claims and/or conclusions
841

of the paper (regardless of whether the code and data are provided or not)?
842

Answer: [Yes]
843

Justification: Experimental results are reproducible.
844

Guidelines:
845

• The answer NA means that the paper does not include experiments.
846

• If the paper includes experiments, a No answer to this question will not be perceived
847

well by the reviewers: Making the paper reproducible is important, regardless of
848

whether the code and data are provided or not.
849

• If the contribution is a dataset and/or model, the authors should describe the steps taken
850

to make their results reproducible or verifiable.
851

• Depending on the contribution, reproducibility can be accomplished in various ways.
852

For example, if the contribution is a novel architecture, describing the architecture fully
853

might suffice, or if the contribution is a specific model and empirical evaluation, it may
854

be necessary to either make it possible for others to replicate the model with the same
855

dataset, or provide access to the model. In general. releasing code and data is often
856

one good way to accomplish this, but reproducibility can also be provided via detailed
857

instructions for how to replicate the results, access to a hosted model (e.g., in the case
858

of a large language model), releasing of a model checkpoint, or other means that are
859

appropriate to the research performed.
860

• While NeurIPS does not require releasing code, the conference does require all submis-
861

sions to provide some reasonable avenue for reproducibility, which may depend on the
862

nature of the contribution. For example
863

(a) If the contribution is primarily a new algorithm, the paper should make it clear how
864

to reproduce that algorithm.
865

(b) If the contribution is primarily a new model architecture, the paper should describe
866

the architecture clearly and fully.
867

(c) If the contribution is a new model (e.g., a large language model), then there should
868

either be a way to access this model for reproducing the results or a way to reproduce
869

the model (e.g., with an open-source dataset or instructions for how to construct
870

the dataset).
871

(d) We recognize that reproducibility may be tricky in some cases, in which case
872

authors are welcome to describe the particular way they provide for reproducibility.
873

In the case of closed-source models, it may be that access to the model is limited in
874

some way (e.g., to registered users), but it should be possible for other researchers
875

to have some path to reproducing or verifying the results.
876

5. Open access to data and code
877

24


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
878

tions to faithfully reproduce the main experimental results, as described in supplemental
879

material?
880

Answer: [Yes]
881

Justification: Code and data are open access.
882

Guidelines:
883

• The answer NA means that paper does not include experiments requiring code.
884

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
885

public/guides/CodeSubmissionPolicy) for more details.
886

• While we encourage the release of code and data, we understand that this might not be
887

possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
888

including code, unless this is central to the contribution (e.g., for a new open-source
889

benchmark).
890

• The instructions should contain the exact command and environment needed to run to
891

reproduce the results. See the NeurIPS code and data submission guidelines (https:
892

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
893

• The authors should provide instructions on data access and preparation, including how
894

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
895

• The authors should provide scripts to reproduce all experimental results for the new
896

proposed method and baselines. If only a subset of experiments are reproducible, they
897

should state which ones are omitted from the script and why.
898

• At submission time, to preserve anonymity, the authors should release anonymized
899

versions (if applicable).
900

• Providing as much information as possible in supplemental material (appended to the
901

paper) is recommended, but including URLs to data and code is permitted.
902

6. Experimental Setting/Details
903

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
904

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
905

results?
906

Answer: [Yes]
907

Justification: Experimental settings are explained.
908

Guidelines:
909

• The answer NA means that the paper does not include experiments.
910

• The experimental setting should be presented in the core of the paper to a level of detail
911

that is necessary to appreciate the results and make sense of them.
912

• The full details can be provided either with the code, in appendix, or as supplemental
913

material.
914

7. Experiment Statistical Significance
915

Question: Does the paper report error bars suitably and correctly defined or other appropriate
916

information about the statistical significance of the experiments?
917

Answer: [Yes]
918

Justification: All experiments conducted at least 10 times.
919

Guidelines:
920

• The answer NA means that the paper does not include experiments.
921

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
922

dence intervals, or statistical significance tests, at least for the experiments that support
923

the main claims of the paper.
924

• The factors of variability that the error bars are capturing should be clearly stated (for
925

example, train/test split, initialization, random drawing of some parameter, or overall
926

run with given experimental conditions).
927

• The method for calculating the error bars should be explained (closed form formula,
928

call to a library function, bootstrap, etc.)
929

25


---Page Break---
• The assumptions made should be given (e.g., Normally distributed errors).
930

• It should be clear whether the error bar is the standard deviation or the standard error
931

of the mean.
932

• It is OK to report 1-sigma error bars, but one should state it. The authors should
933

preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
934

of Normality of errors is not verified.
935

• For asymmetric distributions, the authors should be careful not to show in tables or
936

figures symmetric error bars that would yield results that are out of range (e.g. negative
937

error rates).
938

• If error bars are reported in tables or plots, The authors should explain in the text how
939

they were calculated and reference the corresponding figures or tables in the text.
940

8. Experiments Compute Resources
941

Question: For each experiment, does the paper provide sufficient information on the com-
942

puter resources (type of compute workers, memory, time of execution) needed to reproduce
943

the experiments?
944

Answer: [Yes]
945

Justification: Access LLM API.
946

Guidelines:
947

• The answer NA means that the paper does not include experiments.
948

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
949

or cloud provider, including relevant memory and storage.
950

• The paper should provide the amount of compute required for each of the individual
951

experimental runs as well as estimate the total compute.
952

• The paper should disclose whether the full research project required more compute
953

than the experiments reported in the paper (e.g., preliminary or failed experiments that
954

didn’t make it into the paper).
955

9. Code Of Ethics
956

Question: Does the research conducted in the paper conform, in every respect, with the
957

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
958

Answer: [Yes]
959

Justification: The research is with the NeurIPS code of Ethics.
960

Guidelines:
961

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
962

• If the authors answer No, they should explain the special circumstances that require a
963

deviation from the Code of Ethics.
964

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
965

eration due to laws or regulations in their jurisdiction).
966

10. Broader Impacts
967

Question: Does the paper discuss both potential positive societal impacts and negative
968

societal impacts of the work performed?
969

Answer: [Yes]
970

Justification: Potential positive societal impacts are included in the paper and Appendix.
971

Guidelines:
972

• The answer NA means that there is no societal impact of the work performed.
973

• If the authors answer NA or No, they should explain why their work has no societal
974

impact or why the paper does not address societal impact.
975

• Examples of negative societal impacts include potential malicious or unintended uses
976

(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
977

(e.g., deployment of technologies that could make decisions that unfairly impact specific
978

groups), privacy considerations, and security considerations.
979

26


---Page Break---
• The conference expects that many papers will be foundational research and not tied
980

to particular applications, let alone deployments. However, if there is a direct path to
981

any negative applications, the authors should point it out. For example, it is legitimate
982

to point out that an improvement in the quality of generative models could be used to
983

generate deepfakes for disinformation. On the other hand, it is not needed to point out
984

that a generic algorithm for optimizing neural networks could enable people to train
985

models that generate Deepfakes faster.
986

• The authors should consider possible harms that could arise when the technology is
987

being used as intended and functioning correctly, harms that could arise when the
988

technology is being used as intended but gives incorrect results, and harms following
989

from (intentional or unintentional) misuse of the technology.
990

• If there are negative societal impacts, the authors could also discuss possible mitigation
991

strategies (e.g., gated release of models, providing defenses in addition to attacks,
992

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
993

feedback over time, improving the efficiency and accessibility of ML).
994

11. Safeguards
995

Question: Does the paper describe safeguards that have been put in place for responsible
996

release of data or models that have a high risk for misuse (e.g., pretrained language models,
997

image generators, or scraped datasets)?
998

Answer: [NA]
999

Justification: No such risk.
1000

Guidelines:
1001

• The answer NA means that the paper poses no such risks.
1002

• Released models that have a high risk for misuse or dual-use should be released with
1003

necessary safeguards to allow for controlled use of the model, for example by requiring
1004

that users adhere to usage guidelines or restrictions to access the model or implementing
1005

safety filters.
1006

• Datasets that have been scraped from the Internet could pose safety risks. The authors
1007

should describe how they avoided releasing unsafe images.
1008

• We recognize that providing effective safeguards is challenging, and many papers do
1009

not require this, but we encourage authors to take this into account and make a best
1010

faith effort.
1011

12. Licenses for existing assets
1012

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
1013

the paper, properly credited and are the license and terms of use explicitly mentioned and
1014

properly respected?
1015

Answer: [Yes]
1016

Justification: LLM API usage.
1017

Guidelines:
1018

• The answer NA means that the paper does not use existing assets.
1019

• The authors should cite the original paper that produced the code package or dataset.
1020

• The authors should state which version of the asset is used and, if possible, include a
1021

URL.
1022

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
1023

• For scraped data from a particular source (e.g., website), the copyright and terms of
1024

service of that source should be provided.
1025

• If assets are released, the license, copyright information, and terms of use in the
1026

package should be provided. For popular datasets, paperswithcode.com/datasets
1027

has curated licenses for some datasets. Their licensing guide can help determine the
1028

license of a dataset.
1029

• For existing datasets that are re-packaged, both the original license and the license of
1030

the derived asset (if it has changed) should be provided.
1031

27


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
1032

the asset’s creators.
1033

13. New Assets
1034

Question: Are new assets introduced in the paper well documented and is the documentation
1035

provided alongside the assets?
1036

Answer: [Yes]
1037

Justification: New datasets and framework proposed.
1038

Guidelines:
1039

• The answer NA means that the paper does not release new assets.
1040

• Researchers should communicate the details of the dataset/code/model as part of their
1041

submissions via structured templates. This includes details about training, license,
1042

limitations, etc.
1043

• The paper should discuss whether and how consent was obtained from people whose
1044

asset is used.
1045

• At submission time, remember to anonymize your assets (if applicable). You can either
1046

create an anonymized URL or include an anonymized zip file.
1047

14. Crowdsourcing and Research with Human Subjects
1048

Question: For crowdsourcing experiments and research with human subjects, does the paper
1049

include the full text of instructions given to participants and screenshots, if applicable, as
1050

well as details about compensation (if any)?
1051

Answer: [NA]
1052

Justification: No human subjects.
1053

Guidelines:
1054

• The answer NA means that the paper does not involve crowdsourcing nor research with
1055

human subjects.
1056

• Including this information in the supplemental material is fine, but if the main contribu-
1057

tion of the paper involves human subjects, then as much detail as possible should be
1058

included in the main paper.
1059

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
1060

or other labor should be paid at least the minimum wage in the country of the data
1061

collector.
1062

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
1063

Subjects
1064

Question: Does the paper describe potential risks incurred by study participants, whether
1065

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
1066

approvals (or an equivalent approval/review based on the requirements of your country or
1067

institution) were obtained?
1068

Answer: [NA]
1069

Justification: No such risk.
1070

Guidelines:
1071

• The answer NA means that the paper does not involve crowdsourcing nor research with
1072

human subjects.
1073

• Depending on the country in which research is conducted, IRB approval (or equivalent)
1074

may be required for any human subjects research. If you obtained IRB approval, you
1075

should clearly state this in the paper.
1076

• We recognize that the procedures for this may vary significantly between institutions
1077

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
1078

guidelines for their institution.
1079

• For initial submissions, do not include any information that would break anonymity (if
1080

applicable), such as the institution conducting the review.
1081

28


---Page Break---
