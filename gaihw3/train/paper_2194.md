Beware of Overestimated Decoding Performance
Arising from Temporal Autocorrelations in
Electroencephalogram Signals

Anonymous Author(s)
Afﬁliation
Address
email

Abstract

Researchers have reported high decoding accuracy (>95%) using non-invasive
1

Electroencephalogram (EEG) signals for brain-computer interface (BCI) decod-
2

ing tasks like image decoding, emotion recognition, auditory spatial attention
3

detection, etc. Since these EEG data were usually collected with well-designed
4

paradigms in labs, the reliability and robustness of the corresponding decoding
5

methods were doubted by some researchers, and they argued that such decoding
6

accuracy was overestimated due to the inherent temporal autocorrelation of EEG
7

signals. However, the coupling between the stimulus-driven neural responses and
8

the EEG temporal autocorrelations makes it difﬁcult to conﬁrm whether this over-
9

estimation exists in truth. Furthermore, the underlying pitfalls behind overesti-
10

mated decoding accuracy have not been fully explained due to a lack of appro-
11

priate formulation. In this work, we formulate the pitfall in various EEG decod-
12

ing tasks in a uniﬁed framework. EEG data were recorded from watermelons
13

to remove stimulus-driven neural responses. Labels were assigned to continuous
14

EEG according to the experimental design for EEG recording of several typical
15

datasets, and then the decoding methods were conducted. The results showed the
16

label can be successfully decoded as long as continuous EEG data with the same
17

label were split into training and test sets. Further analysis indicated that high
18

accuracy of various BCI decoding tasks could be achieved by associating labels
19

with EEG intrinsic temporal autocorrelation features. These results underscore
20

the importance of choosing the right experimental designs and data splits in BCI
21

decoding tasks to prevent inﬂated accuracies due to EEG temporal correlations.
22

The watermelon EEG dataset collected in this work can be obtained at Zenodo:
23

https://zenodo.org/records/11238929, and all the codes of this work can
24

be obtained in the supplementary materials.
25

1
Introduction and related works
26

A brain-computer interface (BCI) is a type of human-machine interaction that bridges a pathway
27

from the brain to external devices [1]. Electroencephalogram (EEG) has emerged as a valuable tool
28

for BCI because of its high time resolution, low cost, and good portability [2], and algorithms of
29

neural decoding from EEG signals play a role in its practical applications. Recently, deep learning
30

methods have been developed widely for various EEG decoding tasks, and high decoding accuracy
31

was reported. For example, in the task of decoding image classes with EEG recordings, when
32

subjects were required to watch images of different classes, a decoding accuracy of 82.90% was
33

reported for the 40-way classiﬁcation by Spampinato et al. [3]. With their EEG dataset, subsequent
34

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.


---Page Break---
studies reported a higher decoding accuracy (98.30%, [4]), high performance on image retrieval, and
35

even image generation from EEG [5, 6, 7].
36

However, it remains unclear what kind of EEG features are learned by the DNN-based models. Some
37

researchers have posited that the high decoding accuracy on the image-evoked EEG dataset was
38

attributed to the block-design paradigm during EEG recording [8, 9, 10], in which 50 images with the
39

same class label were presented to the subject continuously in one block, and the 40 image-classes
40

were presented as 40 separate blocks. Due to the existence of temporal autocorrelation of EEG
41

signals, i.e., the temporally nearby data is more similar than the temporally distal [11, 12, 13, 14],
42

the models could learn the block-related features rather than the image-related.
43

To verify their concerns, Li et al. [8] recorded EEG with two experimental designs: block design
44

and rapid-event design. For the rapid-event design, images across the 40 classes were presented
45

alternately and randomly. When the same DNN model was used, it was found that the decoding
46

accuracy was close to Spampinato et al. [3] with the block-design EEG data, but it was dramati-
47

cally decreased to the chance-level (2.50%) with the rapid-event design data. Subsequent work also
48

conﬁrmed the low decoding accuracy for EEG recorded with rapid-event design [9, 10]. However,
49

Palazzo et al. [15] proposed that temporal autocorrelations only play a marginal role in EEG de-
50

coding tasks because they found that EEG data recorded during rest periods (temporal proximity to
51

adjacent blocks) could not be successfully classiﬁed as the preceding block label or the succeeding
52

block label. They also argued that the rapid-event design seemed to weaken the image-related neural
53

responses due to the possible cognitive load and fatigue effect compared to the block design. Some
54

researchers [15, 16, 17, 18] pointed out that block design is essential because humans tend to react
55

more consistently and respond faster when conditions are presented in blocks [19, 20]. Wilson et
56

al. [18] advised that classiﬁcation work that decodes from block design datasets is the most suitable
57

approach until advances are made to reduce noise.
58

Although the pitfall of overestimated decoding accuracy has been mainly discussed in image neural
59

decoding tasks, we noticed that similar pitfalls might also exist in various EEG decoding tasks such
60

as in auditory spatial attention detection (ASAD) tasks [21, 22, 23, 24], which involves decoding
61

the subjects auditory attention locus from neural data, and in emotion recognition task [25, 26, 27],
62

which involves recognizing the subjects emotion type from neural data. Researchers have also found
63

that splitting a continuous EEG from a speciﬁc experimental condition into training and test sets
64

would bring higher decoding accuracy in epilepsy detection tasks [28], motor imagery decoding
65

tasks [29], and so on. All those high decoding accuracy works share the common characteristic:
66

continuously recorded EEG data of a speciﬁc class (condition) label are divided into training and
67

test sets (see the top-left of Figure 1).
68

Although some studies have mentioned the overestimated decoding accuracy and tried to remind
69

the possible pitfall [8, 30], it is difﬁcult to discriminate the inﬂuence of the inherent temporal auto-
70

correlation in EEG signals due to the coupling of stimuli-driven neural responses and the temporal
71

autocorrelations. More importantly, due to the lack of an effective formalization, there is not an
72

adequate explanation of how models utilize temporal autocorrelation features for decoding. Further-
73

more, their concerns only focused on one speciﬁc decoding task, and the results and conclusions
74

cannot be generalized to general BCI decoding tasks.
75

In this work, the pitfall of various EEG decoding tasks was formulated with a uniﬁed framework.
76

To completely decouple the temporal autocorrelation features from stimuli-driven neural responses,
77

EEG data were collected from 10 watermelons in this work to construct "Watermelon EEG". This
78

method is known as phantom EEG in previous studies [31, 32, 33, 34, 35, 36], and the EEG data
79

exclude stimulus-driven neural responses while reserving the temporal autocorrelation features. For
80

comparison, a human EEG dataset was also adopted.
The watermelon EEG and human EEG
81

were reorganized into three classic neural decoding EEG datasets following their EEG experimen-
82

tal paradigm: image classiﬁcation (CVPR, [3]), emotion classiﬁcation (DEAP, [37]), and auditory
83

spatial attention decoding (KUL, [38]), resulting in six EEG datasets. A sample CNN-based decod-
84

ing model was used to complete the decoding tasks with the corresponding EEG dataset, and the
85

experimental results revealed that:
86

1. When the pitfall was formulated with a unique framework, and the temporal autocorre-
87

lation was deﬁned as domain features, high decoding accuracy of various BCI decoding
88

tasks could be achieved by associating labels with EEG intrinsic temporal autocorrelation
89

features.
90

2


---Page Break---
2. The pitfall exists not only in classiﬁcation but also widely in EEG-image joint training
91

without explicit labels and even image generation.
92

3. Splitting a continuous EEG with the same class label into training and test sets should never
93

be used in future BCI decoding works.
94

2
Method
95

The section is organized by: the pitfall is formulated in Subsection 2.1, and the datasets used are
96

introduced in Subsection 2.2. Then, the methods to ﬁnish different classiﬁcation tasks are introduced
97

in Subsection 2.3, and joint training and image generation from EEG are introduced in Subsection
98

2.4. Some implementation details and statistical analysis method are described in Subsection 2.5.
99

Figure 1: Overestimated decoding performance in BCI works. (a) Continuous EEG data in a certain
experimental condition (with the same class label) are split into training and test sets for decoder
training and evaluation. (b) With the test EEG sample input, the decoder gives output in the forms of
classiﬁcation, retrieval, and generation. (c) Decoders may use both domain features or class-related
features for decoding.

2.1
Problem Formulation
100

In some BCI works on domain generalization [39], all EEG data from a dataset [40] or from a subject
101

[41] are usually regarded as a domain to emphasize EEG pattern distribution differences between
102

datasets or subjects. Adopted from this concept, we regard a period of continuous EEG data with
103

the same class label as a domain. In some BCI works [3, 4, 21, 22, 23, 24, 25, 26, 27], researches
104

segment the EEG data from the same domain into samples and further split the samples into training
105

and test data (as shown in Figure 1a) and complete decoding task, such as classiﬁcation, retrieval
106

and generation (as shown in Figure 1b). In these cases, the models used in these works would learn
107

the coupled features containing the class-related feature and domain feature (as shown in the middle
108

of the Figure 1c). The underlying assumption of these works is that the domain feature plays only a
109

margin role in EEG decoding tasks as shown in the left of the Figure 1c. However, we assumed that
110

the domain feature contributes to the high decoding accuracy as shown in the right of the Figure 1c,
111

which is the pitfall we mentioned in Section 1.
112

To validate our assumption, we need to formulate the pitfall. Denote D as the domain set, and each
113

domain d ∈D contains many samples. We use Sd to denote the sample set of the domain d. The
114

notation xd
i represents the i-th sample (e.g., a 0.5-second EEG data corresponding to watching a
115

speciﬁc image) of domain d, which is associated with class yd
i (e.g., the class label panda of the
116

3


---Page Break---
watched image). Considering the temporal autocorrelation of the EEG data, the domain features of
117

data within the same domain are more similar, while the domain features of data in different domains
118

are more distinct.
119

For EEG decoding tasks, we assume the data is generated from a two-stage process. First, each
120

domain is modeled as a latent factor z sampled from some meta domain distribution p(·). Second,
121

each data sample x is sampled from a sample distribution conditioned on the domain z and class y:
122

z ∼p(·), x ∼p(·|z, y)
(1)

Given the sample x, the aim of a speciﬁc EEG decoding task is to uncover its true class label using
123

the posterior p(y|x). The quantity can be factorized by the domain factor z as,
124

p(y|x) =
∫
p(y, z|x)dz =
∫
p(y|x, z)p(z|x)
(2)

When we use the Watermelon EEG dataset or use a dataset that is completely unrelated to the
125

current task (e.g., decoding images from an auditory EEG dataset), the class-related feature has
126

none possibility to exist in EEG samples. In this condition, p(y|x, z) = p(y|z) and the equation (2)
127

can be modiﬁed as:
128

p(y|x) =
∫
p(y, z|x)dz =
∫
p(y|z)p(z|x)
(3)

The assumption of this work is that the model could also deduce p(y|x) by learning p(y|z) and
129

p(z|x) even there is none class-related feature exists. In other words, we assumed that it could also
130

achieve high decoding accuracy on different EEG decoding tasks when using the Watermelons EEG
131

dataset.
132

2.2
Dataset
133

Watermelon EEG Dataset Ten watermelons were selected as subjects. EEG data were recorded
134

with a NeuroScan SynAmps2 system (Compumedics Limited, Victoria, Australia), using a 64-
135

channel Ag/AgCl electrodes cap with a 10/20 layout. An additional electrode was placed on the
136

lower part of the watermelon as the physiological reference, and the forehead served as the ground
137

site (see Appendix A.1 for photography). The inter-electrode impedances were maintained under
138

20 kOhm. Data were recorded at a sampling rate of 1000 Hz. EEG recordings for each watermelon
139

lasted for more than 1 hour to ensure sufﬁcient data for the decoding task. We refer to the dataset
140

consisting of EEG recordings of 10 watermelons as the Watermelon EEG Dataset.
141

SparrKULee Dataset SparrKULee dataset[42] is a speech-evoked EEG dataset from the KU Leu-
142

ven University containing 64-channel EEG recordings from 85 participants, each of whom listened
143

to 90-150 minutes of natural speech. We used this dataset because EEG recordings were longer than
144

1 hour to ensure a sufﬁcient amount of data for each subject. To match the number of subjects in
145

the Watermelon EEG Dataset, EEG data from 10 subjects (ID: Sub7-Sub16) from the SparrKULee
146

Dataset were used.
147

Dataset reorganization and dataset segmentation The term "reorganization" refers to segmenting
148

continuous EEG into samples and assigning each sample a class label and a domain label according
149

to the referenced experimental design. Here, we follow the experimental designs of three classical
150

published EEG datasets to reorganize the Watermelon EEG Dataset and SparrKULee Dataset. These
151

three datasets were collected respectively for image decoding, emotion recognition, and ASAD
152

tasks.
153

For the image decoding task, we referred to the experimental design of the CVPR dataset [3]. For
154

the CVPR dataset, 40 classes of images were presented in a block-design paradigm. Speciﬁcally, 50
155

different images of the same class were presented continuously in a block, with each image lasting
156

for 0.5 second, resulting in 40 blocks of presentation for each subject. The 0.5-second length EEG
157

data of the same class were split into training, validation, and test sets in a ratio of 8:1:1 [4, 3].
158

Following this experimental design and dataset segmentation, we segment continuous EEG from
159

4


---Page Break---
the Watermelon EEG Dataset and SparrKULee Dataset into blocks and assign a unique class label
160

and a unique domain label for each block. The interval between adjacent blocks is set to 10 seconds
161

to match the rest time of the subjects during the EEG recording in the CVPR dataset. Then, EEG
162

data in each block are further segmented into 50 0.5-s length samples. Since the EEG data in the
163

CVPR dataset has 128 channels, we replicated our 64-channel EEG in the channel dimension. The
164

reorganized datasets for Watermelon Dataset and SparrKULee Dataset are called WM-CVPR and
165

SK-CVPR, respectively. Here, we use the "A-B" naming format, where the left side of "-" represents
166

the source dataset (WM: watermelon dataset, SK: SparrKULee Dataset), and the right side of "-"
167

represents the dataset of which the experimental design is referenced. For the emotion recognition
168

task and ASAD task, the DEAP dataset and the KUL dataset are used as the referenced dataset,
169

resulting in WM-DEAP, SK-DEAP, WM-KUL, and SK-KUL. More details for reorganization can
170

be found in Appendix A.2.
171

2.3
Classiﬁcation tasks
172

Model. To demonstrate that domain features are strong and easy to be learned by the network,
173

we used a simple CNN (or some parts of this CNN) to complete all classiﬁcation tasks mentioned
174

in this work. The CNN network includes a layer-norm layer, a 2D-convolutional layer (output
175

channel: 100), an averaging pooling layer, and two fully connected layers. The kernel size of the
176

2D-convolutional layer depends on the channel number and sampling frequency of the input EEG.
177

The node number of the output fully connected layer depends on the number of classes.
178

Decoding the domain feature To demonstrate that the model can predict the domain factor z from
179

EEG input sample x, which relates to learning posterior p(z|x), a domain label classiﬁcation was
180

adopted on the six datasets (i.e., WM-CVPR, WM-DEAP, WM-KUL, SK-CVPR, SK-DEAP and
181

SK-KUL dataset) with a simple CNN classiﬁer. The splitting strategy leave-samples-out was used,
182

which means that all sample were randomly split into training set, validation set and test set. The
183

outputs after the averaging pooling layer were selected as domain feature representation, and t-SNE
184

was utilized for dimensionality reduction and visualization.
185

Decoding the class label from the domain feature To demonstrate that the model can predict
186

the class label y from the domain factor z, which relates to learning posterior p(y|z), a class label
187

classiﬁcation was adopted on the four datasets (classiﬁcation on the WM-CVPR dataset and SK-
188

CVPR dataset are unnecessary since domain labels and class labels are one-to-one correspondence)
189

using a single network with two linear layers and an intermediate sigmoid function.
190

End-to-end classiﬁcation To demonstrate that the model can predict the class label y from the EEG
191

input sample x directly when samples in the training set and test set are from common domains,
192

a class label classiﬁcation was adopted on the six datasets with the simple CNN classiﬁer. The
193

splitting strategy leave samples out was used. Classiﬁcation on the WM-CVPR dataset and SK-
194

CVPR dataset is the same since domain labels and class labels in the two datasets are one-to-one
195

correspondence. To demonstrate that the model indeed used the domain feature to complete the
196

end-to-end classiﬁcation, the splitting strategy leave domains out was used on the four datasets (i.e.,
197

WM-DEAP, WM-KUL, SK-DEAP, and SK-KUL dataset) in which samples in the same domain
198

only appear in the training set or the test set.
199

Zero-shot classiﬁcation In a recent work [4], EEG data from 34 classes within the CVPR2017
200

dataset were used to train an EEG encoder, and the remaining 6 unseen classes were used for test-
201

ing. The results showed that features of different unseen classes clustered in distinct groups on the
202

two-dimensional t-SNE plane. Similar analyses were conducted on the SK-CVPR and WM-CVPR
203

datasets. Six classes were selected for testing, and the remaining 34 classes were for training. The
204

simple CNN was used to predict class labels from input EEG samples, and the outputs from the av-
205

erage pooling layer were chosen as the EEG feature representation. Two strategies were employed
206

for selecting the 6 test classes: random selection and ﬁrst-six selection. For random selection, the 6
207

test classes are randomly chosen from the 40 classes. For the ﬁrst-six selections, the ﬁrst presented
208

6 classes in the EEG experiment are chosen. During the test stage, since the training set does not in-
209

clude classes corresponding to the test EEG data, the model could not give the corresponding labels
210

and could only output the most probable classes among the 34 seen during training. Therefore, we
211

proposed two evaluation metrics:Accnear and Acc7th. Accnear represents the proportion of EEG
212

data classiﬁed into temporally adjacent classes, while Acc7th represents the proportion classiﬁed
213

into the category presented seventh in time.
214

5


---Page Break---
2.4
Joint training and image generation
215

To demonstrate that the model can utilize domain features to accomplish retrieval and generation
216

besides classiﬁcation, EEG-image joint training and image generation on WM-CVPR and SK-CVPR
217

were conducted.
218

Joint training In the EEG-image joint training, a pre-trained image encoder was typically utilized
219

to extract image representation, while an EEG encoder was employed to extract EEG features to
220

align with the image representation. During the decoding process, a retrieval task was applied.
221

Speciﬁcally, given a test EEG sample and a collection of images containing the target and the non-
222

target. The image representation was reconstructed from the EEG with the EEG encoder. The
223

similarity between the reconstructed image representation and all candidate image representations
224

in the collection is calculated. The decoded output image is selected based on the ranking of these
225

similarities. Usually, the Top-k accuracy and normalized Rank accuracy are used as evaluation
226

metrics. In this work, the simple CNN described in Subsection 2.3 is used as an EEG encoder. The
227

detailed implementation can be found in Appendix A.3.
228

Image generation The image generation aims to generate images seen by the subjects from their
229

EEG data. This task commonly uses a two-stage process: EEG encoding and image generation.
230

In the EEG encoding stage, a model is built to encode EEG data into a latent representation. In
231

the image generation stage, a pre-trained image generator is used. The generator is ﬁne-tuned with
232

EEG representation and corresponding images. In this work, the EEG data are ﬁrst encoded into
233

image representation with a simple CNN described in Subsection 2.3. Following previous work[43],
234

a latent diffusion model conditioned on image representation was used. The metric of n-way top-k
235

accuracy was used for evaluating the semantic correctness of generated images [44]. The detailed
236

implementation can be found in Appendix A.4.
237

2.5
Implement details
238

The neural networks were implemented with the Pytorch and trained on a single high-performance
239

computing node with 8 A800 GPU. For the classiﬁcation task, the AdamW [45] optimizer was em-
240

ployed to minimize the cross-entropy loss function with a learning rate of 10−3. For the joint training
241

and image generation, the AdamW optimizer was used with a learning rate of 10−3 and 5 × 10−4
242

for each task respectively. More details can be found in our codes. All the experiments mentioned
243

in this work were trained within the subjects (i.e., models were trained for each subject respectively)
244

except special annotation (unseen subject decoding results were only presented in Appendix A.5).
245

For statistical analysis, the one-sample t-test was used to check whether the reported results were
246

signiﬁcantly higher than the chance level. Bonferroni correction was used to adjust the p-value. A
247

p-value of 0.05 or lower was considered statistically signiﬁcant.
248

3
Results
249

3.1
Classiﬁcation tasks
250

The results shown in Table 1 present that classiﬁcation accuracy in domain label classiﬁcation and
251

class label classiﬁcation are all signiﬁcantly above the chance level. This shows that the domain
252

feature can be extracted effectively with a simple CNN, and the label class can be decoded from
253

the extracted domain features or from EEG directly. In contrast, the decoding accuracy drops to the
254

chance level when using the splitting strategy leave-domains-out, further supporting domain feature-
255

induced high decoding accuracy. The standard error of the mean calculated over the subjects level
256

is reported for accuracy in this work.
257

Figures 2a and 2b show the t-SNE plot for domain label classiﬁcation and end-to-end class label
258

classiﬁcation. As shown in Figure 2a, 8 distinct clusters exist, each corresponding to one domain.
259

In Figure 2b, 8 distinct clusters also exist, with four corresponding to class label 1 and the other
260

four corresponding to class label 2. This indicates that the high decoding accuracy results from
261

associating class labels with domain features.
262

6


---Page Break---
Table 1: Classiﬁcation accuracy (%) on the six datasets. DLC is for domain label classiﬁcation.
TLC-DF is for class label classiﬁcation from domain features. TLC-EEG is for end-to-end class
label classiﬁcation. TLC-EEG-woDO is for class label classiﬁcation direct from EEG when samples
in the training set and test set are from different domains.

WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL
DLC
88.78 ± 4.95
96.98 ± 0.76
99.99 ± 0.01
69.83 ± 2.98
72.70 ± 1.36
100.00 ± 0.00
DLC (chance level)
2.50
2.50
12.50
2.50
2.50
12.50
TLC-DF
-
92.77 ± 1.31
100.00 ± 0.00
-
76.19 ± 1.80
100.00 ± 0.00
TLC-EEG
88.78 ± 4.95
88.74 ± 3.26
82.74 ± 6.44
69.83 ± 2.98
74.44 ± 2.76
93.34 ± 2.01
TLC-EEG-woDO
-
24.67 ± 2.31
49.97 ± 4.67
-
25.34 ± 1.85
59.32 ± 4.07
TCL (chance level)
2.50
25.00
50.00
2.50
25.00
50.00

Figure 2:
t-SNE plot for (a) domain label classiﬁcation, (b) end-to-end class label classiﬁcation,
and (c) zero-shot class label classiﬁcation

The experimental results for zero-shot classiﬁcation are displayed in Table 2. It can be observed
263

that the model tended to classify test samples into temporally adjacent classes. Figure 2c shows the
264

t-SNE visualization of the unseen EEG features extracted from the decoder. Despite being unseen,
265

different domains of features clustered in distinct groups. This suggests that the decoder just learned
266

to extract EEG domain features during training and distinguish unseen EEG responses from the
267

domain features.
268

Table 2: Zero-shot EEG classiﬁcation accuracy (%) on WM-CVPR and SK-CVPR datasets.

WM-CVPR
ﬁrst-six
WM-CVPR
random
SK-CVPR
ﬁrst-six
SK-CVPR
random
Accnear
-
79.43 ± 5.61
-
78.00 ± 5.66
Acc7th
69.60 ± 10.64
6.73 ± 3.24
77.03 ± 11.32
0.87 ± 0.82

3.2
Joint training and image generation
269

For EEG-image joint training, Table 3 displays the accuracy for the retravel task on the test set. The
270

table shows that, for both types of loss functions, decoding accuracy is far above the chance level,
271

demonstrating that the model can utilize domain features to align EEG with image features. Table 3
272

Result for joint training on WM-CVPR and SK-CVPR with a loss function of cosine similarity (CS)
273

or InfoNCE.
274

Table 3: Accuracy (%) for joint training on WM-CVPR and SK-CVPR with a loss function of cosine
similarity (CS) or InfoNCE.

WM-CVPR
SK-CVPR
Chance level
CS loss
InfoNCE loss
CS loss
InfoNCE loss
Top1 Acc
81.40 ± 9.25
90.15 ± 5.45
80.70 ± 0.60
79.70 ± 0.92
2.50
Top5 Acc
90.65 ± 5.82
98.56 ± 1.09
88.86 ± 1.03
92.39 ± 0.38
12.50
Rank Acc
95.87 ± 2.51
99.42 ± 0.38
95.20 ± 0.24
98.09 ± 0.07
50.00

For image generation, Table 4 displays the n-way top-k accuracy for the generated images on the
275

WM-CVPR and SK-CVPR datasets. The metrics are signiﬁcantly above the chance level, indicating
276

7


---Page Break---
that the generated images have correct semantics. Figure 3 shows some generated images on the
277

WM-CVPR dataset. As shown in the ﬁgure, the model can exactly generate the correct images. The
278

results on EEG-image joint training and image generation show that in addition to classiﬁcation
279

tasks, retrieval, and generation can also achieve high performance by leveraging domain features
280

shared by the test and training sets.
281

Table 4: Accuracy (%) for semantic correctness. The repeated times N was set to 50.

-
Top-1/50-way
Top-5/50-way
Top-1/100-way
Top-5/100-way

WM-CVPR
26.77 ± 3.37
46.44 ± 4.60
21.64 ± 2.89
38.11 ± 4.30
SK-CVPR
25.04 ± 0.93
43.61 ± 0.88
20.37 ± 0.91
35.35 ± 0.89
Chance
2.00
10.00
1.00
5.00

Figure 3:
EEG-generated image from a typical watermelon subject, where the ﬁrst column of
each panel represents the real images "watched" by the watermelon subject, and the following ﬁve
columns show the images generated by the model.

4
Discussion
282

4.1
Relying on the domain features for EEG decoding
283

While many works on EEG decoding have reported high-performance results, we proposed that
284

some of these high-performance may rely on temporal autocorrelation of EEG data. The pitfall may
285

involve different EEG decoding tasks. To clarify this pitfall, the concept of domain was adopted
286

to describe the temporal autocorrelation of a continuous EEG with the same label. EEG data were
287

collected from watermelon as the phantom to exclude the contribution of stimuli-driven neural re-
288

sponses to decoding results. The results showed that a simple CNN network could well learn domain
289

features from EEG data and could associate class labels with domain features.
290

To avoid the pitfalls, a feasible approach is to adopt a reasonable data-splitting strategy to avoid train-
291

ing and test sets sharing the common domain features, i.e., a leave-domains-out splitting strategy.
292

For instance, a leave-subjects-out data-splitting strategy can be adopted, which entails designating
293

the data from certain participants for training and data from others for testing. Alternatively, for
294

datasets that do not follow a block design, a leave-trials-out strategy may be applied. Prior research
295

has consistently demonstrated that employing a leave-subjects-out splitting strategy precipitates a
296

notable decline in decoding performance [46]. In some cases, it has been reported that decoding
297

accuracy dropped to the chance level [47, 8]. The prevalent interpretation is that inter-individual
298

variability [46] hampers the generalizability across different subjects. However, we posit that the
299

observed decrement in decoding accuracy is attributable to model overﬁtting to domain features.
300

Although the leave-subjects-out partitioning strategy is designed to prevent the leakage of domain
301

features, the presence of these domain features in the training set can still lead the model to inadver-
302

tently exploit them to differentiate between categories during the training phase. The methods and
303

results further support the conclusion can be found in Appendix A.5
304

Palazzo et al. [15] proposed that the EEG temporal correlation related to baseline drift could be al-
305

leviated by high-pass ﬁltering. However, our further experiment proved that the domain feature still
306

exists and that high decoding accuracy could be achieved in any frequency band (see Appendix A.6).
307

We argue that the focus should not be exclusively on the elimination of EEG autocorrelation through
308

8


---Page Break---
ﬁltering. Instead, greater emphasis should be placed on the experimental paradigms of EEG record-
309

ing and the methods employed for dataset splitting. By addressing these aspects, we can proactively
310

prevent the overestimated decoding accuracy arising from EEG temporal autocorrelations.
311

It is worth noting that we do not want to create an illusion that all BCI works utilize EEG temporal
312

autocorrelation features for decoding. In fact, there are many works that do not rely on EEG temporal
313

autocorrelation features for decoding in image decoding [48, 49, 50] emotion recognition [51], sleep
314

detection [40, 41] and ASAD [52]. These works demonstrated the feasibility of various BCI tasks.
315

4.2
Potential sources of domain features
316

In this work, we have demonstrated the existence of EEG temporal autocorrelation in the water-
317

melon EEG, which consists of no neural activities, and in the human EEG data. Li et al. [8] believed
318

the model decodes by utilizing the baseline drift in the CVPR2017 dataset. They found that when
319

the EEG data is ﬁltered with a bandpass ﬁlter, the decoding accuracy dropped greatly. Palazzo et
320

al. [15] also claimed that temporal correlation was strong only in low frequency. However, we have
321

demonstrated in Appendix A.4 that the domain feature still exists and that high decoding accuracy
322

can be achieved in any frequency band. In addition to baseline drift, some neuroscience works have
323

shown that temporal autocorrelation existed in neural oscillation, which could be reﬂected in EEG
324

in various frequency bands. This is referred to as Long-Range Temporal Correlations (LRTC) in
325

neuroscience research [11, 12, 13, 14]. Linkenkaer-Hansen et al. [13] ﬁrst calculated the LRTC in
326

resting-state EEG data. They found that spontaneous alpha, mu, and beta oscillations result in signif-
327

icant LRTC for at least several hundred seconds during resting conditions. Subsequent neuroscience
328

research further demonstrated that signiﬁcant LRTC exists in the theta [11] and gamma [12] bands.
329

While baseline drift can be removed through ﬁltering, the frequency range of the LRTC overlaps
330

with the frequency range of stimuli-driven neural responses, making it impossible to remove this
331

domain feature through ﬁltering. Temporal correlation analysis on human EEG in the SparrKULee
332

Dataset showed the existence of strong LRTC in all frequency bands, and the LRTC in a narrowband
333

is sufﬁcient to complete the corresponding decoding task. The methods and results further support
334

the conclusion can be found in Appendix A.7.
335

4.3
Limitation and future work
336

Although direct evidence of overestimated decoding accuracy attributable to domain feature across
337

various brain-computer interface (BCI) tasks have been provided in the current work, no solution has
338

been proposed to mitigate overﬁtting to domain features in the training set. Some works have already
339

used domain adaptation [2, 53, 54] or domain generalization [40, 41] method to improve decoding
340

accuracy under leave-subjects-out data splitting in BCI tasks. This may also help alleviate the ad-
341

verse effects of domain features on decoding tasks. It is also noteworthy to highlight the remarkable
342

efﬁcacy of large-scale EEG model in various BCI decoding tasks [55, 56, 57]. Given that domain
343

features are pervasive in extensive EEG datasets and do not necessitate manually annotated labels,
344

self-supervised pre-trained large EEG models may be especially adept at discerning and neutralizing
345

domain features, thereby facilitating more robust and generalizable decoding performance.
346

5
Conclusion
347

In this work, the “overestimated decoding accuracy pitfall” in various EEG decoding tasks is for-
348

mulated in a uniﬁed framework by adopting the concept of “domain”. Some typical EEG decoding
349

tasks (image decoding, emotion recognition, and auditory spatial attention detection) are conducted
350

on the self-collected watermelon EEG dataset. The results showed that EEG data from different
351

domains have distinctive domain features induced by EEG temporal autocorrelations. Using the in-
352

appropriate data partitioning strategy, high decoding accuracy is achieved by associating class labels
353

with domain features. The results will draw attention to the high decoding performance caused by
354

EEG temporal correlation and guide the development of BCI in a positive direction.
355

9


---Page Break---
References
356

[1] Yue-Ting Pan, Jing-Lun Chou, and Chun-Shu Wei. Matt: A manifold attention network for
357

eeg decoding. Advances in Neural Information Processing Systems, 35:31116–31129, 2022.
358

[2] Reinmar Kobler, Jun-ichiro Hirayama, Qibin Zhao, and Motoaki Kawanabe. Spd domain-
359

speciﬁc batch normalization to crack interpretable unsupervised domain adaptation in eeg. Ad-
360

vances in Neural Information Processing Systems, 35:6219–6235, 2022.
361

[3] C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, and M. Shah. Deep learning
362

human mind for automated visual classiﬁcation. In 2017 IEEE Conference on Computer Vision
363

and Pattern Recognition (CVPR), pages 4503–4511, 2017.
364

[4] Prajwal Singh, Dwip Dalal, Gautam Vashishtha, Krishna Miyapuram, and Shanmuganathan
365

Raman. Learning robust deep visual representations from eeg brain recordings. In Proceedings
366

of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7553–7562,
367

2024.
368

[5] Isaak Kavasidis, Simone Palazzo, Concetto Spampinato, Daniela Giordano, and Mubarak Shah.
369

Brain2image: Converting brain signals into images. In Proceedings of the 25th ACM inter-
370

national conference on Multimedia, MM 17, pages 1809–1817, New York, NY, USA, 2017.
371

Association for Computing Machinery.
372

[6] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, and M. Shah. Generative adversarial
373

networks conditioned by brain signals. In 2017 IEEE International Conference on Computer
374

Vision (ICCV), pages 3430–3438, 2017.
375

[7] Praveen Tirupattur, Yogesh Singh Rawat, Concetto Spampinato, and Mubarak Shah.
376

Thoughtviz: Visualizing human thoughts using generative adversarial network. In Proceed-
377

ings of the 26th ACM international conference on Multimedia, MM 18, pages 950–958, New
378

York, NY, USA, 2018. Association for Computing Machinery.
379

[8] Ren Li, Jared S. Johansen, Hamad Ahmed, Thomas V. Ilyevsky, Ronnie B. Wilbur, Hari M.
380

Bharadwaj, and Jeffrey Mark Siskind. The perils and pitfalls of block design for eeg classiﬁca-
381

tion experiments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):316–
382

333, 2021.
383

[9] Hamad Ahmed, Ronnie B. Wilbur, Hari M. Bharadwaj, and Jeffrey Mark Siskind. Object
384

classiﬁcation from randomized eeg trials. In 2021 IEEE/CVF Conference on Computer Vision
385

and Pattern Recognition (CVPR), pages 3844–3853, 2021.
386

[10] Hari M Bharadwaj, Ronnie B. Wilbur, and Jeffrey Mark Siskind. Still an ineffective method
387

with supertrials/erpscomments on decoding brain representations by multimodal learning of
388

neural activity and visual features. IEEE Transactions on Pattern Analysis and Machine Intel-
389

ligence, 45(11):14052–14054, 2023.
390

[11] Luc Berthouze, Leon M. James, and Simon F. Farmer. Human eeg shows long-range temporal
391

correlations of oscillation amplitude in theta, alpha and beta bands across a wide age range.
392

Clinical Neurophysiology, 121(8):1187–1197, 2010.
393

[12] Mona Irrmischer, Simon-Shlomo Poil, Huibert D. Mansvelder, Francesca Sangiuliano Intra,
394

and Klaus Linkenkaer-Hansen. Strong long-range temporal correlations of beta/gamma oscil-
395

lations are associated with poor sustained visual attention performance. European Journal of
396

Neuroscience, 48(8):2674–2683, 2018.
397

[13] Klaus Linkenkaer-Hansen, Vadim V. Nikouline, J. Matias Palva, and Risto J. Ilmoniemi. Long-
398

range temporal correlations and scaling behavior in human brain oscillations. Journal of Neu-
399

roscience, 21(4):1370–1377, 2001.
400

[14] Vadim V. Nikulin and Tom Brismar. Long-range temporal correlations in alpha and beta oscil-
401

lations: effect of arousal level and testretest reliability. Clinical Neurophysiology, 115(8):1896–
402

1908, 2004.
403

10


---Page Break---
[15] Simone Palazzo, Concetto Spampinato, Joseph Schmidt, Isaak Kavasidis, Daniela Giordano,
404

and Mubarak Shah. Correct block-design experiments mitigate temporal correlation bias in
405

eeg classiﬁcation. arXiv preprint arXiv:2012.03849, 2020.
406

[16] Jacopo Cavazza, Waqar Ahmed, Riccardo Volpi, Pietro Morerio, Francesco Bossi, Cesco
407

Willemse, Agnieszka Wykowska, and Vittorio Murino. Understanding action concepts from
408

videos and brain activity through subjects consensus. Scientiﬁc Reports, 12(11):19073, 2022.
409

[17] Alankrit Mishra, Nikhil Raj, and Garima Bajwa. Eeg-based image feature extraction for vi-
410

sual classiﬁcation using deep learning. In 2022 International Conference on Intelligent Data
411

Science Technologies and Applications (IDSTA), pages 181–188, 2022.
412

[18] Holly Wilson, Xi Chen, Mohammad Golbabaee, Michael J. Proulx, and Eamonn ONeill. Fea-
413

sibility of decoding visual information from eeg. Brain-Computer Interfaces, 0(0):1–28, 2023.
414

[19] Lauren E. Ethridge, Shefali Brahmbhatt, Yuan Gao, Jennifer E. Mcdowell, and Brett A.
415

Clementz. Consider the context: Blocked versus interleaved presentation of antisaccade tri-
416

als. Psychophysiology, 46(5):1100–1107, 2009.
417

[20] Nelson A. Roque, Timothy J. Wright, and Walter R. Boot. Do different attention capture
418

paradigms measure different types of capture?
Attention, Perception & Psychophysics,
419

78(7):2014–2030, 2016.
420

[21] Enze Su, Siqi Cai, Longhan Xie, Haizhou Li, and Tanja Schultz. Stanet: A spatiotemporal
421

attention network for decoding auditory spatial attention from eeg. IEEE Transactions on
422

Biomedical Engineering, 69(7):2233–2242, 2022.
423

[22] Saurav Pahuja, Siqi Cai, Tanja Schultz, and Haizhou Li. Xanet: Cross-attention between eeg
424

of left and right brain for auditory attention decoding. In 2023 11th International IEEE/EMBS
425

Conference on Neural Engineering (NER), pages 1–4, 2023.
426

[23] Xiran Xu, Bo Wang, Yujie Yan, Xihong Wu, and Jing Chen. A densenet-based method for
427

decoding auditory spatial attention with eeg. In IEEE International Conference on Acoustics,
428

Speech and Signal Processing (ICASSP), pages 1946–1950, 2024.
429

[24] Qinke Ni, Hongyu Zhang, Cunhang Fan, Shengbing Pei, Chang Zhou, and Zhao Lv. Dbpnet:
430

Dual-branch parallel network with temporal-frequency fusion for auditory attention detection.
431

In International Joint Conference on Artiﬁcial Intelligence (IJCAI), 2024.
432

[25] Liangliang Hu, Congming Tan, Jiayang Xu, Rui Qiao, Yilin Hu, and Yin Tian. Decoding
433

emotion with phaseamplitude fusion features of eeg functional connectivity network. Neural
434

Networks, 172:106148, 2024.
435

[26] Jiayang Xu, Wenxia Qian, Liangliang Hu, Guangyuan Liao, and Yin Tian. Eeg decoding
436

for musical emotion with functional connectivity features. Biomedical Signal Processing and
437

Control, 89:105744, 2024.
438

[27] Zhi Zhang, Shenghua Zhong, and Yan Liu. Beyond mimicking under-represented emotions:
439

Deep data augmentation with emotional subspace constraints for eeg-based emotion recog-
440

nition. Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 38(99):10252–10260,
441

2024.
442

[28] Geoffrey Brookshire, Jake Kasper, Nicholas M. Blauch, Yunan Charles Wu, Ryan Glatt,
443

David A. Merrill, Spencer Gerrol, Keith J. Yoder, Colin Quirk, and Ché Lucero. Data leak-
444

age in deep learning studies of translational eeg. Frontiers in Neuroscience, 18, 2024.
445

[29] Hamdi Altaheri, Ghulam Muhammad, Mansour Alsulaiman, Syed Umar Amin, Ghadir Ali
446

Altuwaijri, Wadood Abdul, Mohamed A. Bencherif, and Mohammed Faisal. Deep learning
447

techniques for classiﬁcation of electroencephalogram (eeg) motor imagery (mi) signals: a re-
448

view. Neural Computing and Applications, 35(20):14681–14722, 2023.
449

[30] Iustina Rotaru, Simon Geirnaert, Nicolas Heintz, Iris Van de Ryck, Alexander Bertrand, and
450

Tom Francart. What are we really decoding? unveiling biases in eeg-based decoding of the
451

spatial focus of auditory attention. Journal of Neural Engineering, 21(1):016017, 2024.
452

11


---Page Break---
[31] Mukund Balasubramanian, William M. Wells, John R. Ives, Patrick Britz, Robert V. Mulk-
453

ern, and Darren B. Orbach. Rf heating of gold cup and conductive plastic electrodes during
454

simultaneous eeg and mri. The Neurodiagnostic Journal, 57(1):69–83, 2017.
455

[32] Maximillian K. Egan, Ryan Larsen, Jonathan Wirsich, Brad P. Sutton, and Sepideh Sadaghiani.
456

Safety and data quality of eeg recorded simultaneously with multi-band fmri. PLOS ONE,
457

16(7):e0238485, 2021.
458

[33] Dominik Freche, Jodie Naim-Feil, Avi Peled, Nava Levit-Binnun, and Elisha Moses. A quan-
459

titative physical model of the tms-induced discharge artifacts in eeg. PLOS Computational
460

Biology, 14(7):e1006177, 2018.
461

[34] Johan N. van der Meer, Yke B. Eisma, Ronald Meester, Marc Jacobs, and Aart J. Nederveen.
462

Effects of mobile phone electromagnetic ﬁelds on brain waves in healthy volunteers. Scientiﬁc
463

Reports, 13(1):21758, 2023.
464

[35] Tuomas Mutanen, Hanna Mäki, and Risto J. Ilmoniemi. The effect of stimulus parameters on
465

tmseeg muscle artifacts. Brain Stimulation, 6(3):371–376, 2013.
466

[36] Limin Sun and Hermann Hinrichs. Simultaneously recorded eegfmri: Removal of gradient
467

artifacts by subtraction of head movement related average artifact waveforms. Human Brain
468

Mapping, 30(10):3361–3377, 2009.
469

[37] Sander Koelstra, Christian Muhl, Mohammad Soleymani, Jong-Seok Lee, Ashkan Yazdani,
470

Touradj Ebrahimi, Thierry Pun, Anton Nijholt, and Ioannis Patras. Deap: A database for emo-
471

tion analysis;using physiological signals. IEEE Transactions on Affective Computing, 3(1):18–
472

31, 2012.
473

[38] Neetha Das, Tom Francart, and Alexander Bertrand. Auditory attention detection dataset kuleu-
474

ven, 2020.
475

[39] Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen,
476

Wenjun Zeng, and Philip S. Yu. Generalizing to unseen domains: A survey on domain gener-
477

alization. IEEE Transactions on Knowledge and Data Engineering, 35(8):8052–8072, 2023.
478

[40] Jiquan Wang, Sha Zhao, Haiteng Jiang, Shijian Li, Tao Li, and Gang Pan. Generalizable sleep
479

staging via multi-level domain alignment. Proceedings of the AAAI Conference on Artiﬁcial
480

Intelligence, 38(11):265–273, 2024.
481

[41] Chaoqi Yang, M. Brandon Westover, and Jimeng Sun. Manydg: Many-domain generalization
482

for healthcare applications. In The Eleventh International Conference on Learning Represen-
483

tations, 2023.
484

[42] Bernd Accou, Lies Bollens, Marlies Gillis, Wendy Verheijen, Hugo Van Hamme, and Tom
485

Francart. Sparrkulee: A speech-evoked auditory response repository of the ku leuven, contain-
486

ing eeg of 85 participants. bioRxiv preprint bioRxiv: 2023.07.24.550310, 2023.
487

[43] Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, and Juan Helen Zhou. Seeing beyond
488

the brain: Conditional diffusion model with sparse masked modeling for vision decoding. In
489

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
490

22710–22720, 2023.
491

[44] Yunpeng Bai, Xintao Wang, Yan-pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan. Dreamdiffu-
492

sion: Generating high-quality images from brain eeg signals. arXiv preprint arXiv:2306.16934,
493

2023.
494

[45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
495

arXiv:1711.05101, 2017.
496

[46] Xinke Shen, Xianggen Liu, Xin Hu, Dan Zhang, and Sen Song. Contrastive learning of subject-
497

invariant eeg representations for cross-subject emotion recognition. IEEE Transactions on
498

Affective Computing, 14(3):2496–2511, 2023.
499

12


---Page Break---
[47] Ivine Kuruvila, Jan Muncke, Eghart Fischer, and Ulrich Hoppe. Extracting the auditory atten-
500

tion in a dual-speaker scenario from eeg using a joint cnn-lstm model. Frontiers in Physiology,
501

12, 2021.
502

[48] Changde Du, Kaicheng Fu, Jinpeng Li, and Huiguang He. Decoding visual neural representa-
503

tions by multimodal learning of brain-visual-linguistic features. IEEE Transactions on Pattern
504

Analysis and Machine Intelligence, 45(9):10760–10777, 2023.
505

[49] Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. Decod-
506

ing natural images from eeg for object recognition. In The Twelfth International Conference
507

on Learning Representations, 2024.
508

[50] Zesheng Ye, Lina Yao, Yu Zhang, and Sylvia Gustin.
Self-supervised cross-modal visual
509

retrieval from brain activities. Pattern Recognition, 145:109915, 2024.
510

[51] Yiming Wang, Bin Zhang, and Yujiao Tang. Dmmr: Cross-subject domain generalization for
511

eeg-based emotion recognition via denoising mixed mutual reconstruction. Proceedings of the
512

AAAI Conference on Artiﬁcial Intelligence, 38(11):628–636, 2024.
513

[52] Servaas Vandecappelle, Lucas Deckers, Neetha Das, Amir Hossein Ansari, Alexander
514

Bertrand, and Tom Francart. Eeg-based detection of the locus of auditory attention with con-
515

volutional neural networks. eLife, 10:e56481, 2021.
516

[53] Theo Gnassounou, Rémi Flamary, and Alexandre Gramfort. Convolution monge mapping
517

normalization for learning on sleep data. Advances in Neural Information Processing Systems,
518

36, 2023.
519

[54] Johanna Wilroth, Bo Bernhardsson, Frida Heskebeck, Martin A. Skoglund, Carolina Bergeling,
520

and Emina Alickovic. Improving eeg-based decoding of the locus of auditory attention through
521

domain adaptation*. Journal of Neural Engineering, 20(6):066022, 2023.
522

[55] Weibang Jiang, Li-Ming Zhao, and Bao-Liang Lu. Large brain model for learning generic
523

representations with tremendous eeg data in bci. In The Twelfth International Conference on
524

Learning Representations, 2024.
525

[56] Chaoqi Yang, M. Westover, and Jimeng Sun. Biot: Biosignal transformer for cross-data learn-
526

ing in the wild. In Advances in Neural Information Processing Systems, volume 36, 2023.
527

[57] Ke Yi, Yansen Wang, Kan Ren, and Dongsheng Li. Learning topology-agnostic eeg representa-
528

tions with geometry-aware modeling. In Advances in Neural Information Processing Systems,
529

volume 36, 2023.
530

13


---Page Break---
A
Appendix A
531

A.1
Photography of the watermelon subject
532

Figure 4: Photos of watermelons used in the experiment. Each watermelon’s ID is marked on the
watermelon, with IDs ranging from 1 to 10.

A.2
Reorganization for KUL dataset and DEAP dataset
533

For the emotion recognition task, we referred to the experimental design of DEAP dataset [37]. In
534

this dataset, the EEG data were recorded while subjects are presented with 40 audio-visual clips of
535

60 seconds in length, with each corresponding to one of four emotion classes. We only used the ﬁrst
536

32 channels of the EEG to match the EEG channel numbers in the DEAP dataset. The watermelon
537

EEG data and SparrKULee EEG data were down-sampled to 128 Hz and then were segmented into
538

40 60-second segments. The interval between adjacent segments is set to 40 seconds to match the
539

rest time of the subjects during the EEG recording in the KUL dataset. Each segment was assigned
540

a unique domain label and a class label in accordance with the DEAP dataset, and each segment was
541

further segmented into 2-second samples [25]. The reorganized datasets for the Watermelon EEG
542

Dataset and SparrKULee Dataset are called WM-DEAP and SK-DEAP, respectively.
543

For the ASAD task, we referred to the experimental design of the KUL dataset [38]. In this dataset,
544

8 clips of two-talker mixed speech are presented to subjects, with each lasting for 6 minutes. Each
545

speech clip contains a left talker and a right talker. Subjects are instructed to attend left or right talker
546

during the entire duration of one clip presentation. The watermelon EEG data and SparrKULee EEG
547

data were down-sampled to 128 Hz and then were epoch into 8 6-minute segments. The interval
548

between adjacent segments is set to 1-2 minutes to match the rest time of the subjects during the EEG
549

recording in the KUL dataset. Each segment was assigned a unique domain label and a class label
550

in accordance with the KUL dataset and was further segmented into 1-second samples [22, 21, 23].
551

The reorganized datasets for Watermelon Dataset and SparrKULee Dataset are called WM-KUL and
552

SK-KUL, respectively.
553

A.3
Detailed implementation of joint training
554

The joint training was performed on the WM-CVPR and SK-CVPR datasets. All EEG samples
555

were randomly divided into the training set, validation set, and test set in a ratio of 8:1:1. The image
556

encoder of the CLIP (CLIP VIT-L/14) model 1 is chosen to extract image representation, yielding
557

1https://huggingface.co/openai/clip-vit-large-patch14

14


---Page Break---
768-dimensional vectors from the image inputs. The structure of the EEG encoder is similar to the
558

model introduced in Subsection 2.3, with an augmentation from 40 to 768 output nodes to match
559

the dimension of the image representation. The network is trained using either a cosine similarity
560

(CS) loss or an InfoNCE contrastive loss (with a temperature parameter set to 0.07). The evaluation
561

metrics selected are Top-1 accuracy, Top-5 accuracy, and Rank accuracy, where the Top-1 accuracy
562

metric is equivalent to the classiﬁcation accuracy in the classiﬁcation task.
563

A.4
Detailed implementation of image generation
564

We take an approach similar to previous works [44] 2. We used a CLIP image encoder to extract
565

image representation and trained an EEG encoder with cosine similarity loss to reconstruct image
566

representation from EEG. This process is the same as described in Joint training with image features.
567

The reconstructed features are then serviced as a conditional input of an image generator. To match
568

the reconstructed features, we employ the pre-trained StableDiffusion model 3 as our generator. This
569

model uses a ﬁxed pre-trained image encoder (CLIP VIT-L/14) to extract image features, which
570

then guide the Latent Diffusion models generation process in the latent space. The diffusion model
571

gradually generates images from a random noise distribution that corresponds to the conditional
572

features during its iterative process. To improve the generation performance, we ﬁne-tuned the
573

generator with the reconstructed image features and the corresponding images. Experiments were
574

done on the WM-CVPR and SK-CVPR datasets. All EEG samples were randomly divided into
575

training set, validation set, and test set in a ratio of 8:1:1.
576

Consistent with previous work [1], we evaluate the semantic correctness of the generated images
577

using N-way Top-1 and Top-5 accuracy classiﬁcation tasks. Speciﬁcally, given a generated image
578

input, a pre-trained ImageNet1K classiﬁer is used to output a classiﬁcation logit probability among
579

1000 classes. Among the 1000 classes, N-1 random classes and the correct class are selected, and
580

the Top-1 and Top-5 classiﬁcation accuracy are calculated. To avoid randomness, this operation is
581

repeated 50 times for each generated image, with the average value taken as the accuracy.
582

A.5
leave-subjects-out data splitting strategy
583

In this subsection, we employed the leave-subjects-out data splitting strategy. This refers to using
584

data from a subset of subjects for training, while data from the remaining subjects are used for
585

testing. Within the training data, there are two further data partitioning methods: leave-samples-out
586

and leave-subjects-out. The former involves randomly dividing all samples of the training data into
587

training and validation sets, whereas the latter uses data from a subset of subjects for the training set,
588

with the remaining subjects data allocated for the test set. Table 5 presents the decoding accuracy
589

for six datasets (i.e., WM-CVPR, WM-DEAP, WM-KUL, SK-CVPR, SK-DEAP, and SK-KUL).
590

It can be observed that when the leave-samples-out splitting strategy was used within the training
591

data, both the training and validation sets achieved very high decoding accuracy, but the accuracy
592

only reached the chance level on the test set. Such results are similar to those reported by [46, 47, 8],
593

which corroborates the argument that while the leave-subjects-out approach may avert the domain
594

features leakage, it cannot prevent overﬁtting of the domain features during the training stage, as
595

discussed in Subsection 4.1. Moreover, when the leave-subjects-out data splitting strategy was used
596

within the training dataset, the validation set performance was only at chance level despite high
597

accuracy on the training set. This further demonstrates that decoding that relies on domain features
598

cannot be generalized to practical application scenarios.
599

2https://github.com/bbaaii/DreamDiffusion
3https://huggingface.co/runwayml/stable-diffusion-v1-5

15


---Page Break---
Table 5: Decoding accuracy (%) for the six datasets on training, validation and test set. Leave-
subjects-out data splitting strategy is used for training and test data. Leave-samples-out and leave-
subjects-out data splitting strategy is used for training and validation set. The mean accuracy and
standard deviation are calculated over subjects level with a ﬁve-fold cross-validation.

Data
splitting strategy
for validation set
WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL

leave-
samples-
out

Training
80.93 ± 1.68
87.86 ± 1.48
99.54 ± 0.16
69.17 ± 1.03
76.22 ± 0.71
100.00 ± 0.00
validation
80.55 ± 1.59
86.10 ± 1.63
99.43 ± 0.24
68.86 ± 1.20
74.55 ± 0.60
100.00 ± 0.00
Test
2.46 ± 0.16
24.22 ± 0.48
48.37 ± 2.15
2.70 ± 0.63
26.71 ± 0.87
50.22 ± 1.14
leave-
subjects-
out

Training
78.93 ± 1.09
86.40 ± 0.75
99.59 ± 0.16
72.31 ± 0.59
77.43 ± 0.52
100.00 ± 0.00
validation
3.70 ± 0.34
22.23 ± 1.29
56.13 ± 3.06
4.15 ± 0.60
24.57 ± 0.33
53.24 ± 2.85
Test
2.26 ± 0.16
24.90 ± 0.43
52.06 ± 1.26
2.13 ± 0.29
25.61 ± 0.43
45.22 ± 2.83
Chance level
2.50
25.00
50.00
2.50
25.00
50.00

A.6
Results on different frequency band
600

To demonstrate that domain features are not solely due to baseline drift, we conducted an analysis on
601

seven frequency bands across six datasets. These seven frequency bands are delta (0-4 Hz), theta (4-
602

8 Hz), alpha (8-12 Hz), beta (12-32 Hz), low gamma (32-45 Hz), and high gamma (55-95 Hz). High
603

gamma frequency band results for DEAP and KUL datasets are not presented due to the sampling
604

rate of 128 Hz (i.e., only frequency under 64 Hz is available according to the Nyquist sampling
605

theorem). Tables 6, 7, 8, and 9 show the decoding accuracy for domain label classiﬁcation (DLC-
606

EEG), class label classiﬁcation from domain features (TLC-DF), class label classiﬁcation directly
607

from EEG (TLC-EEG), and class label classiﬁcation directly from EEG when samples in the training
608

set and test set are from different domains (TLC-EEG-woDO), respectively. As expected, the highest
609

decoding accuracy is observed for both the low-frequency band (delta band) and the full-frequency
610

EEG data. However, other frequency bands also exhibited decoding accuracy signiﬁcantly higher
611

than the chance level. This suggests that baseline correction through ﬁltering does not eliminate
612

domain features. Consequently, any experimental designs and data partitioning strategies that could
613

lead to the leakage of domain information should be meticulously avoided.
614

Table 6: Decoding accuracy (%) using different EEG bands for domain label classiﬁcation (DLC-
EEG)

WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL
Full
88.78 ± 4.95
96.98 ± 0.76
99.99 ± 0.01
69.83 ± 2.98
72.70 ± 1.36
100.00 ± 0.00
Delta
88.58 ± 5.11
96.31 ± 0.89
99.99 ± 0.01
69.65 ± 2.88
72.76 ± 1.24
100.00 ± 0.00
Theta
8.90 ± 1.95
10.54 ± 2.17
41.97 ± 5.50
11.24 ± 1.60
10.19 ± 1.15
43.11 ± 5.13
Alpha
8.62 ± 1.77
12.88 ± 2.80
43.42 ± 5.96
15.16 ± 1.76
12.87 ± 1.00
47.67 ± 4.70
Beta
18.53 ± 3.18
18.18 ± 2.72
57.85 ± 4.86
43.95 ± 2.27
43.68 ± 1.97
97.17 ± 0.71
Low gamma
39.74 ± 7.35
62.59 ± 5.95
85.97 ± 2.82
53.72 ± 2.25
52.82 ± 1.40
96.57 ± 0.96
High gamma
42.15 ± 7.39
-
-
61.55 ± 1.94
-
-
Chance level
2.50
2.50
50.00
2.50
2.50
50.00

Table 7: Decoding accuracy (%) using different EEG bands for class label classiﬁcation from domain
features (TLC-DF)

WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL
Full
-
92.77 ± 1.31
100.00 ± 0.00
-
76.19 ± 1.80
100.00 ± 0.00
Delta
-
92.12 ± 1.49
100.00 ± 0.00
-
76.51 ± 1.74
100.00 ± 0.00
Theta
-
31.39 ± 1.80
67.78 ± 3.56
-
32.17 ± 1.16
69.41 ± 3.80
Alpha
-
33.10 ± 2.43
68.78 ± 4.03
-
33.88 ± 0.69
71.98 ± 3.47
Beta
-
39.03 ± 2.09
77.33 ± 3.71
-
56.91 ± 2.02
97.83 ± 0.72
Low gamma
-
59.32 ± 5.22
88.23 ± 2.56
-
63.80 ± 1.43
97.44 ± 0.88
High gamma
-
-
-
-
-
-
Chance level
-
25.00
50.00
-
25.00
50.00

16


---Page Break---
Table 8: Decoding accuracy (%) using different EEG bands for class label classiﬁcation directly
from EEG (TLC-EEG)

WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL
Full
88.78 ± 4.95
88.74 ± 3.26
82.74 ± 6.44
69.83 ± 2.98
74.44 ± 2.76
93.34 ± 2.01
Delta
88.58 ± 5.11
88.60 ± 3.36
81.49 ± 6.44
69.65 ± 2.88
74.90 ± 2.55
92.90 ± 2.15
Theta
8.90 ± 1.95
29.36 ± 1.27
66.40 ± 3.47
11.24 ± 1.60
30.62 ± 1.30
65.28 ± 3.83
Alpha
8.62 ± 1.77
31.00 ± 1.70
68.16 ± 3.59
15.16 ± 1.76
32.17 ± 1.10
67.11 ± 3.83
Beta
18.53 ± 3.18
35.95 ± 1.12
71.24 ± 4.16
43.95 ± 2.27
43.95 ± 1.78
93.27 ± 1.52
Low gamma
39.74 ± 7.35
52.05 ± 4.72
73.42 ± 5.37
53.72 ± 2.25
46.81 ± 1.03
93.51 ± 2.02
High gamma
42.15 ± 7.39
-
-
61.55 ± 1.94
-
-
Chance level
2.50
25.00
50.00
2.50
25.00
50.00

Table 9: Decoding accuracy (%) using different EEG bands for class label classiﬁcation directly
from EEG when samples in the training set and test set are from different domains (TLC-EEG-
woDO)

WM-CVPR
WM-DEAP
WM-KUL
SK-CVPR
SK-DEAP
SK-KUL
Full
-
24.67 ± 2.31
49.97 ± 4.67
-
25.34 ± 1.85
59.32 ± 4.07
Delta
-
25.89 ± 2.58
49.72 ± 4.85
-
24.71 ± 1.74
58.25 ± 3.76
Theta
-
23.91 ± 0.63
49.10 ± 3.13
-
23.28 ± 2.18
51.89 ± 4.32
Alpha
-
23.50 ± 0.82
49.70 ± 2.91
-
23.26 ± 1.68
52.77 ± 4.04
Beta
-
22.96 ± 1.25
50.30 ± 4.35
-
24.21 ± 1.39
57.32 ± 5.26
Low gamma
-
26.75 ± 2.17
49.46 ± 3.63
-
25.72 ± 1.61
54.88 ± 4.92
High gamma
-
-
-
-
-
-
Chance level
-
25.00
50.00
-
25.00
50.00

A.7
LRTC
615

The autocorrelation analysis was used to evaluate long range temporal correlation in EEG data from
616

the Watermelon and SparrKULee datasets, similar to the approach taken by previous study. For a
617

lengthy segment of single-channel EEG, the Morlet wavelet transform was employed to extract the
618

time-varying amplitude envelope Wf(t) at a given frequency f. The autocorrelation function ACFf
619

for Wf(t) is deﬁned as:
620

ACFf(τ) = corr(Wf(t), Wf(t + τ))
(4)

In the above equation, corr(, ) denotes the Pearson correlation coefﬁcient between two time series,
621

and τ represents the time lag.
622

In our analysis, the original EEG data were down-sampled to 200 Hz. Ninety-ﬁve analysis frequen-
623

cies were distributed linearly and evenly between 1-95 Hz. Two hundred autocorrelation time lags
624

were logarithmically spaced between 0.5 s and 500 s. For each subject in the Watermelon dataset,
625

continuous EEG recordings were divided into ﬁve segments of equal length (with each segment
626

ranging from 15 to 20 minutes), and autocorrelation analysis was completed on each segment. For
627

each subject in the SparrKULee dataset, the autocorrelation analysis was carried out separately on
628

each of their ten trials. Figure 5 shows the results of the autocorrelation analysis for the Watermelon
629

and SparrKULee datasets. The ﬁgure illustrates the magnitude of correlation at different frequen-
630

cies and time lags (represented by color). The correlation values were obtained by averaging the
631

results across all subjects, segments (trials), and electrodes. Black lines represent the contour lines
632

where p = 0.01, as determined by statistical analysis. Statistical signiﬁcance was assessed using
633

single-sample t-test at the subject-electrode level. Speciﬁcally, for each electrode of each subject, the
634

averaged Pearson correlation coefﬁcient across all segments (trials) was used as the value for the t-
635

test. Additionally, p-values were corrected for multiple comparisons using the Benjamini-Hochberg
636

False Discovery Rate (BH-FDR) to type I error.
637

As demonstrated in Figure 5, EEG data from both Watermelon and SparrKULee datasets show
638

signiﬁcant LRTC across multiple frequency bands. For the EEG data from the Watermelon dataset,
639

signiﬁcant bands of LRTC are primarily distributed in the low-frequency range (<8 Hz) and around
640

50 Hz, with these correlations spanning over 500 seconds. This indicates that baseline drifts and line
641

17


---Page Break---
noise contribute to the temporal correlation observed in the Watermelon dataset. For the EEG data
642

from the SparrKULee dataset, LRTCs are signiﬁcant across the entire frequency range. Similarly,
643

LTRCs are most prominent at low frequencies (<5 Hz) and around 50 Hz, consistent with the ﬁndings
644

from the Watermelon dataset. Notably, for SparrKULee dataset, there is also a signiﬁcant presence
645

of LTRC around 10 Hz, which aligns with previous research ﬁndings [13], suggesting the temporal
646

correlation of alpha oscillations in human subjects.
647

Figure 5: Autocorrelation analysis result on (a) Watermelon and (b) SparrKULee datasets.

18


---Page Break---
NeurIPS Paper Checklist
648

1. Claims
649

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
650

paper’s contributions and scope?
651

Answer: [Yes]
652

Justiﬁcation: the problem formulation could be found in 2.1 and results supporting the
653

contribution of this paper could be found in 3.1 and section 3.2.
654

Guidelines:
655

• The answer NA means that the abstract and introduction do not include the claims
656

made in the paper.
657

• The abstract and/or introduction should clearly state the claims made, including the
658

contributions made in the paper and important assumptions and limitations. A No or
659

NA answer to this question will not be perceived well by the reviewers.
660

• The claims made should match theoretical and experimental results, and reﬂect how
661

much the results can be expected to generalize to other settings.
662

• It is ﬁne to include aspirational goals as motivation as long as it is clear that these
663

goals are not attained by the paper.
664

2. Limitations
665

Question: Does the paper discuss the limitations of the work performed by the authors?
666

Answer: [Yes]
667

Justiﬁcation: the limitations could be found in section 4.3.
668

Guidelines:
669

• The answer NA means that the paper has no limitation while the answer No means
670

that the paper has limitations, but those are not discussed in the paper.
671

• The authors are encouraged to create a separate "Limitations" section in their paper.
672

• The paper should point out any strong assumptions and how robust the results are to
673

violations of these assumptions (e.g., independence assumptions, noiseless settings,
674

model well-speciﬁcation, asymptotic approximations only holding locally). The au-
675

thors should reﬂect on how these assumptions might be violated in practice and what
676

the implications would be.
677

• The authors should reﬂect on the scope of the claims made, e.g., if the approach was
678

only tested on a few datasets or with a few runs. In general, empirical results often
679

depend on implicit assumptions, which should be articulated.
680

• The authors should reﬂect on the factors that inﬂuence the performance of the ap-
681

proach. For example, a facial recognition algorithm may perform poorly when image
682

resolution is low or images are taken in low lighting. Or a speech-to-text system might
683

not be used reliably to provide closed captions for online lectures because it fails to
684

handle technical jargon.
685

• The authors should discuss the computational efﬁciency of the proposed algorithms
686

and how they scale with dataset size.
687

• If applicable, the authors should discuss possible limitations of their approach to ad-
688

dress problems of privacy and fairness.
689

• While the authors might fear that complete honesty about limitations might be used by
690

reviewers as grounds for rejection, a worse outcome might be that reviewers discover
691

limitations that aren’t acknowledged in the paper. The authors should use their best
692

judgment and recognize that individual actions in favor of transparency play an impor-
693

tant role in developing norms that preserve the integrity of the community. Reviewers
694

will be speciﬁcally instructed to not penalize honesty concerning limitations.
695

3. Theory Assumptions and Proofs
696

Question: For each theoretical result, does the paper provide the full set of assumptions and
697

a complete (and correct) proof?
698

Answer: [NA]
699

19


---Page Break---
Justiﬁcation: The paper does not include any theoretical results.
700

Guidelines:
701

• The answer NA means that the paper does not include theoretical results.
702

• All the theorems, formulas, and proofs in the paper should be numbered and cross-
703

referenced.
704

• All assumptions should be clearly stated or referenced in the statement of any theo-
705

rems.
706

• The proofs can either appear in the main paper or the supplemental material, but if
707

they appear in the supplemental material, the authors are encouraged to provide a
708

short proof sketch to provide intuition.
709

• Inversely, any informal proof provided in the core of the paper should be comple-
710

mented by formal proofs provided in appendix or supplemental material.
711

• Theorems and Lemmas that the proof relies upon should be properly referenced.
712

4. Experimental Result Reproducibility
713

Question: Does the paper fully disclose all the information needed to reproduce the main
714

experimental results of the paper to the extent that it affects the main claims and/or conclu-
715

sions of the paper (regardless of whether the code and data are provided or not)?
716

Answer: [Yes]
717

Justiﬁcation: the information needed to reproduce all the experimental results of this paper
718

could be found in Section 2.2, Section 2.3, Section 2.4, Section 2.5 and Appendix A.
719

Guidelines:
720

• The answer NA means that the paper does not include experiments.
721

• If the paper includes experiments, a No answer to this question will not be perceived
722

well by the reviewers: Making the paper reproducible is important, regardless of
723

whether the code and data are provided or not.
724

• If the contribution is a dataset and/or model, the authors should describe the steps
725

taken to make their results reproducible or veriﬁable.
726

• Depending on the contribution, reproducibility can be accomplished in various ways.
727

For example, if the contribution is a novel architecture, describing the architecture
728

fully might sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation,
729

it may be necessary to either make it possible for others to replicate the model with
730

the same dataset, or provide access to the model. In general. releasing code and data
731

is often one good way to accomplish this, but reproducibility can also be provided via
732

detailed instructions for how to replicate the results, access to a hosted model (e.g., in
733

the case of a large language model), releasing of a model checkpoint, or other means
734

that are appropriate to the research performed.
735

• While NeurIPS does not require releasing code, the conference does require all sub-
736

missions to provide some reasonable avenue for reproducibility, which may depend
737

on the nature of the contribution. For example
738

(a) If the contribution is primarily a new algorithm, the paper should make it clear
739

how to reproduce that algorithm.
740

(b) If the contribution is primarily a new model architecture, the paper should describe
741

the architecture clearly and fully.
742

(c) If the contribution is a new model (e.g., a large language model), then there should
743

either be a way to access this model for reproducing the results or a way to re-
744

produce the model (e.g., with an open-source dataset or instructions for how to
745

construct the dataset).
746

(d) We recognize that reproducibility may be tricky in some cases, in which case au-
747

thors are welcome to describe the particular way they provide for reproducibility.
748

In the case of closed-source models, it may be that access to the model is limited in
749

some way (e.g., to registered users), but it should be possible for other researchers
750

to have some path to reproducing or verifying the results.
751

5. Open access to data and code
752

20


---Page Break---
Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
753

tions to faithfully reproduce the main experimental results, as described in supplemental
754

material?
755

Answer: [Yes]
756

Justiﬁcation: the collected Watermelon EEG dataset could be available in Zenodo and the
757

human dataset used in this work could also be downloaded in the link provided in supple-
758

mentary materials. All the codes to reproduce this work can be found in supplementary
759

materials.
760

Guidelines:
761

• The answer NA means that paper does not include experiments requiring code.
762

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
763

public/guides/CodeSubmissionPolicy) for more details.
764

• While we encourage the release of code and data, we understand that this might not
765

be possible, so No is an acceptable answer. Papers cannot be rejected simply for not
766

including code, unless this is central to the contribution (e.g., for a new open-source
767

benchmark).
768

• The instructions should contain the exact command and environment needed to run to
769

reproduce the results. See the NeurIPS code and data submission guidelines (https:
770

//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
771

• The authors should provide instructions on data access and preparation, including how
772

to access the raw data, preprocessed data, intermediate data, and generated data, etc.
773

• The authors should provide scripts to reproduce all experimental results for the new
774

proposed method and baselines. If only a subset of experiments are reproducible, they
775

should state which ones are omitted from the script and why.
776

• At submission time, to preserve anonymity, the authors should release anonymized
777

versions (if applicable).
778

• Providing as much information as possible in supplemental material (appended to the
779

paper) is recommended, but including URLs to data and code is permitted.
780

6. Experimental Setting/Details
781

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
782

parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
783

results?
784

Answer: [Yes]
785

Justiﬁcation: the experimental setting and details could also be found in Section 2.2, Sec-
786

tion 2.3, Section 2.4, Section 2.5, and could also be found in the codes.
787

Guidelines:
788

• The answer NA means that the paper does not include experiments.
789

• The experimental setting should be presented in the core of the paper to a level of
790

detail that is necessary to appreciate the results and make sense of them.
791

• The full details can be provided either with the code, in appendix, or as supplemental
792

material.
793

7. Experiment Statistical Signiﬁcance
794

Question: Does the paper report error bars suitably and correctly deﬁned or other appropri-
795

ate information about the statistical signiﬁcance of the experiments?
796

Answer: [Yes]
797

Justiﬁcation: the standard error of the mean is reported for all results. As we only com-
798

pared the result against chance level, one-sample t-test was used for statistical analysis to
799

check whether the reported results are signiﬁcant high above the chance level. Given that
800

decoding analyses was conducted on multiple frequency bands, Bonferroni correction was
801

used to adjust the p-value to reduce the risk of type-I error. A p-value of 0.05 or lower is
802

considered statistically signiﬁcant.
803

Guidelines:
804

21


---Page Break---
• The answer NA means that the paper does not include experiments.
805

• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
806

dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
807

the main claims of the paper.
808

• The factors of variability that the error bars are capturing should be clearly stated (for
809

example, train/test split, initialization, random drawing of some parameter, or overall
810

run with given experimental conditions).
811

• The method for calculating the error bars should be explained (closed form formula,
812

call to a library function, bootstrap, etc.)
813

• The assumptions made should be given (e.g., Normally distributed errors).
814

• It should be clear whether the error bar is the standard deviation or the standard error
815

of the mean.
816

• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
817

ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
818

Normality of errors is not veriﬁed.
819

• For asymmetric distributions, the authors should be careful not to show in tables or
820

ﬁgures symmetric error bars that would yield results that are out of range (e.g. negative
821

error rates).
822

• If error bars are reported in tables or plots, The authors should explain in the text how
823

they were calculated and reference the corresponding ﬁgures or tables in the text.
824

8. Experiments Compute Resources
825

Question: For each experiment, does the paper provide sufﬁcient information on the com-
826

puter resources (type of compute workers, memory, time of execution) needed to reproduce
827

the experiments?
828

Answer: [Yes]
829

Justiﬁcation: the neural networks were implemented with the Pytorch and trained on a
830

single HPC node with 8 A800 GPU.
831

Guidelines:
832

• The answer NA means that the paper does not include experiments.
833

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
834

or cloud provider, including relevant memory and storage.
835

• The paper should provide the amount of compute required for each of the individual
836

experimental runs as well as estimate the total compute.
837

• The paper should disclose whether the full research project required more compute
838

than the experiments reported in the paper (e.g., preliminary or failed experiments
839

that didn’t make it into the paper).
840

9. Code Of Ethics
841

Question: Does the research conducted in the paper conform, in every respect, with the
842

NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
843

Answer: [Yes]
844

Justiﬁcation: The collected dataset was released in an anonymous form, and all codes do
845

not contain any identity information.
846

Guidelines:
847

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
848

• If the authors answer No, they should explain the special circumstances that require a
849

deviation from the Code of Ethics.
850

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
851

eration due to laws or regulations in their jurisdiction).
852

10. Broader Impacts
853

Question: Does the paper discuss both potential positive societal impacts and negative
854

societal impacts of the work performed?
855

Answer: [NA]
856

22


---Page Break---
Justiﬁcation: The purpose of this paper is to let researchers beware of overestimated de-
857

coding performance arising from temporal autocorrelations in EEG signals. This work
858

formalizes and proves the pitfalls existing in current EEG decoding tasks. We believe that
859

this will not generate any signiﬁcant societal impact.
860

Guidelines:
861

• The answer NA means that there is no societal impact of the work performed.
862

• If the authors answer NA or No, they should explain why their work has no societal
863

impact or why the paper does not address societal impact.
864

• Examples of negative societal impacts include potential malicious or unintended uses
865

(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
866

(e.g., deployment of technologies that could make decisions that unfairly impact spe-
867

ciﬁc groups), privacy considerations, and security considerations.
868

• The conference expects that many papers will be foundational research and not tied
869

to particular applications, let alone deployments. However, if there is a direct path to
870

any negative applications, the authors should point it out. For example, it is legitimate
871

to point out that an improvement in the quality of generative models could be used to
872

generate deepfakes for disinformation. On the other hand, it is not needed to point out
873

that a generic algorithm for optimizing neural networks could enable people to train
874

models that generate Deepfakes faster.
875

• The authors should consider possible harms that could arise when the technology is
876

being used as intended and functioning correctly, harms that could arise when the
877

technology is being used as intended but gives incorrect results, and harms following
878

from (intentional or unintentional) misuse of the technology.
879

• If there are negative societal impacts, the authors could also discuss possible mitiga-
880

tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
881

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
882

feedback over time, improving the efﬁciency and accessibility of ML).
883

11. Safeguards
884

Question: Does the paper describe safeguards that have been put in place for responsible
885

release of data or models that have a high risk for misuse (e.g., pretrained language models,
886

image generators, or scraped datasets)?
887

Answer: [NA]
888

Justiﬁcation: the paper poses no such risks.
889

Guidelines:
890

• The answer NA means that the paper poses no such risks.
891

• Released models that have a high risk for misuse or dual-use should be released with
892

necessary safeguards to allow for controlled use of the model, for example by re-
893

quiring that users adhere to usage guidelines or restrictions to access the model or
894

implementing safety ﬁlters.
895

• Datasets that have been scraped from the Internet could pose safety risks. The authors
896

should describe how they avoided releasing unsafe images.
897

• We recognize that providing effective safeguards is challenging, and many papers do
898

not require this, but we encourage authors to take this into account and make a best
899

faith effort.
900

12. Licenses for existing assets
901

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
902

the paper, properly credited and are the license and terms of use explicitly mentioned and
903

properly respected?
904

Answer: [Yes]
905

Justiﬁcation: The existing assets we used are the SparrKULee dataset, which is licensed
906

under CC-BY-NC-4.0.
907

Guidelines:
908

• The answer NA means that the paper does not use existing assets.
909

23


---Page Break---
• The authors should cite the original paper that produced the code package or dataset.
910

• The authors should state which version of the asset is used and, if possible, include a
911

URL.
912

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
913

• For scraped data from a particular source (e.g., website), the copyright and terms of
914

service of that source should be provided.
915

• If assets are released, the license, copyright information, and terms of use in the pack-
916

age should be provided. For popular datasets, paperswithcode.com/datasets has
917

curated licenses for some datasets. Their licensing guide can help determine the li-
918

cense of a dataset.
919

• For existing datasets that are re-packaged, both the original license and the license of
920

the derived asset (if it has changed) should be provided.
921

• If this information is not available online, the authors are encouraged to reach out to
922

the asset’s creators.
923

13. New Assets
924

Question: Are new assets introduced in the paper well documented and is the documenta-
925

tion provided alongside the assets?
926

Answer: [Yes]
927

Justiﬁcation: We have released an anonymous Watermelon EEG dataset, which can be
928

accessed at https://zenodo.org/records/11238929
929

Guidelines:
930

• The answer NA means that the paper does not release new assets.
931

• Researchers should communicate the details of the dataset/code/model as part of their
932

submissions via structured templates. This includes details about training, license,
933

limitations, etc.
934

• The paper should discuss whether and how consent was obtained from people whose
935

asset is used.
936

• At submission time, remember to anonymize your assets (if applicable). You can
937

either create an anonymized URL or include an anonymized zip ﬁle.
938

14. Crowdsourcing and Research with Human Subjects
939

Question: For crowdsourcing experiments and research with human subjects, does the pa-
940

per include the full text of instructions given to participants and screenshots, if applicable,
941

as well as details about compensation (if any)?
942

Answer: [NA]
943

Justiﬁcation: we collected the Watermelon EEG dataset, but watermelon is not a human
944

subject. Nonetheless, we provided experiment details, which could be found in section
945

Section 2.2.
946

Guidelines:
947

• The answer NA means that the paper does not involve crowdsourcing nor research
948

with human subjects.
949

• Including this information in the supplemental material is ﬁne, but if the main contri-
950

bution of the paper involves human subjects, then as much detail as possible should
951

be included in the main paper.
952

• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
953

tion, or other labor should be paid at least the minimum wage in the country of the
954

data collector.
955

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
956

Subjects
957

Question: Does the paper describe potential risks incurred by study participants, whether
958

such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
959

approvals (or an equivalent approval/review based on the requirements of your country or
960

institution) were obtained?
961

24


---Page Break---
Answer: [NA]
962

Justiﬁcation: the paper does not involve crowdsourcing nor research with human subjects
963

Guidelines:
964

• The answer NA means that the paper does not involve crowdsourcing nor research
965

with human subjects.
966

• Depending on the country in which research is conducted, IRB approval (or equiva-
967

lent) may be required for any human subjects research. If you obtained IRB approval,
968

you should clearly state this in the paper.
969

• We recognize that the procedures for this may vary signiﬁcantly between institutions
970

and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
971

guidelines for their institution.
972

• For initial submissions, do not include any information that would break anonymity
973

(if applicable), such as the institution conducting the review.
974

25


---Page Break---
