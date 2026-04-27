Language Generation in the Limit

Jon Kleinberg
Departments of Computer Science
and Information Sciene
Cornell University
Ithaca NY

Sendhil Mullainathan
Booth School of Business
University of Chicago
Chicago IL

Abstract

Although current large language models are complex, the most basic speciﬁcations
of the underlying language generation problem itself are simple to state: given a
ﬁnite set of training samples from an unknown language, produce valid new strings
from the language that don’t already appear in the training data. Here we ask what
we can conclude about language generation using only this speciﬁcation, without
further assumptions. In particular, suppose that an adversary enumerates the strings
of an unknown target language L that is known only to come from one of a possibly
inﬁnite list of candidates. A computational agent is trying to learn to generate from
this language; we say that the agent generates from L in the limit if after some ﬁnite
point in the enumeration of L, the agent is able to produce new elements that come
exclusively from L and that have not yet been presented by the adversary. Our main
result is that there is an agent that is able to generate in the limit for every countable
list of candidate languages. This contrasts dramatically with negative results due
to Gold and Angluin in a well-studied model of language learning where the goal
is to identify an unknown language from samples; the difference between these
results suggests that identifying a language is a fundamentally different problem
than generating from it.

1
Introduction

The recent advances in large language models (LLMs) have been remarkable, sparking active lines
of theoretical work into their performance. These investigations implicitly revolve around two
fundamental questions: how do we formally reason about the effectiveness of LLMs; and within such
a framework, what are the core ideas at a mathematical level that enable their performance?

Answers to these questions must begin by formalizing the speciﬁcation for what a generative algorithm
for language should be doing. Here, we propose starting from a very basic, assumption-free, statement
for such a speciﬁcation: there is an unknown target language L, over time the algorithm sees a
sequence of strings from L, and eventually we would like the algorithm to generate new strings from
L that it has not seen before.1

Viewed this way, it is also clear why it seems so remarkable for LLMs to be doing well at such a
problem. The fully general statement of the problem feels unsolvable: if we know nothing about the
unknown target language L, then how can a generative algorithm reliably produce valid strings from
L that it hasn’t seen before?

1We will formalize these concepts more precisely below, but for now we can think of a language as simply any
set of strings over a ﬁxed alphabet; for example, the strings of the language could be the set of all grammatical
sentences (or all well-formed expressions) according to a given grammar.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Language Learning in the Limit.
In fact, there is a well-established formalism that allows us to
phrase this point precisely: the classical model of language learning in the limit, formulated by Mark
Gold in 1967 and fully characterized by Dana Angluin in 1980 [6, 2]. In this model, there is an
unknown language K that is known only to be produced by one of a list of candidate representations
R1, R2, R3, . . ., where Li is the language produced by representation Ri. We can think of this list
of representations as the set of all possible context-free grammars, or the set of all possible ﬁnite
automata, or the set of all Turing machines with a ﬁxed space bound, or any other generative model
that produces strings; in fact, the formal result is much more general than this, in that it is sufﬁcient
to suppose that the unknown language K simply comes from a countable list of candidate languages
L1, L2, L3, . . ., and we can dispense with explicit representations altogether.2

In the Gold-Angluin model, an adversary enumerates the strings of K one by one, and the algorithm is
required after each new string to guess a language Li from the list such that Li = K. If there is some
ﬁnite step t after which the algorithm’s guess is always correct, then we say the algorithm has identiﬁed
K in the limit. Gold proved that this is impossible in general, even for simple language families
such as the regular languages (i.e. those produced by ﬁnite automata), and Angluin characterized
precisely those families for which it is possible, further establishing how limited they are [1, 2].
Note, crucially, that in the Gold-Angluin model, the adversary enumerates strings in K, but does not
provide examples of strings that do not belong to K, nor does it allow the algorithm to ask questions
about a string’s membership in K; their point with this formalism was to focus on cases where an
agent tries inferring a language purely from seeing a sufﬁcient number of examples of strings that
belong to the language. (In Section A.1, we provide a self-contained proof of the negative result
for identiﬁcation in the limit; while not strictly necessary for our results, we ﬁnd it provides useful
background and context for the problem.)

Our Results: Language Generation in the Limit.
These negative results of Gold and Angluin
feel intuitive — how should we be able to identify a language from a ﬁnite sample when we are
allowed to make essentially no assumptions about its structure? Because of this intuition, both via
the Gold-Angluin model and for more informal reasons as well, the focus in language generation has
naturally turned to distributional assumptions; one posits that large language models are so effective
because they are able to exploit distributional probabilities of language, and from a ﬁnite set of
samples they are able to estimate conditional probabilities of strings with increasing accuracy. In this
way, the question moves from adversaries to probability distributions, and one seeks explanations for
the effectiveness of LLMs through underlying probabilistic models.

In this paper, we offer a sharply different view: we show that in the Gold-Angluin model of
adversarially produced examples, language generation is always possible. We will provide full details
on the result and its proof beginning in the next section, but the key point at a high level is that
even in an adversarial model with an unknown language K, language generation is a fundamentally
different task than language identiﬁcation: where identiﬁcation asks an algorithm to eventually name
a language Li = K after seeing a large enough ﬁnite sample S from K, generation instead asks an
algorithm to eventually output strings in K −S after seeing a large enough S from K. Our main
result is that this difference in speciﬁcations leads to dramatic differences in what is possible in the
limit; whereas the Gold-Angluin results establish that identiﬁcation in the limit is impossible except
in highly constrained cases, we show that generation in the limit is possible for every countable list of
candidate languages.

General Connections to Language Modeling.
Our approach emphasizing theoretical properties
of language generation and worst-case guarantees, in the style of the Gold-Angluin model, is a source
of limitations but also a source of generality; it is therefore important to discuss how we draw stylized
insights about language modeling from our theoretical formalism. Clearly, methods to design large
language models in practice make extensive use of the empirical distributional properties of language,
as they should. Our results don’t question this design methodology; when there are genuine empirical
regularities in the training data, there is no reason not to use them. Rather, our results argue that if we
are looking for the essential reasons why language generation is tractable, we do not fundamentally
require any empirical regularities, or indeed any probabilistic assumptions at all; there is instead
a formal sense in which language generation — unlike broader learning tasks such as language

2In this more general view, we will assume that the family of languages is presented simply via a black box
that for a string w and an index i can answer the question, “Is w ∈Li?”

2


---Page Break---
identiﬁcation — is possible even against an adversary presenting positive training examples in a
worst-case fashion. In some essential way, the generation problem is therefore different from these
other learning tasks in crucial ways that more detailed formalisms may potentially obscure.

Despite the generality of the model, producing the generation algorithm that proves our main theorem
makes use of subtle structural properties of the given list of candidate languages. Again, we defer any
detailed description to the subsequent sections, but the idea at a high level is to maintain a sequence
of “provisional languages” among the candidate languages that are consistent with the ﬁnite sample S
from K seen so far, and to continually reﬁne this sequence of provisional languages as the adversary
adds more strings to the sample S. Since the Gold-Angluin result says that the algorithm can never be
sure which is the true language K, there is a sense in which this reﬁnement process essentially needs
to continue indeﬁnitely, and in general it leads the algorithm to generate from provisional languages
that may be increasingly “thin” subsets of K. This does not cause trouble for the speciﬁcation of
language generation, since it is acceptable to produce any unseen string from K, but it does mean
that while the algorithm is able to eventually produce an inﬁnite sequence of unseen strings from K,
in general it might do so from a narrow part of K.

This property of the solution in the presence of an adversary suggests interesting connections to the
problem of generation in practice as well. In particular, any method for generation has to deal with
an underlying validity problem — producing valid outputs — and an underlying breadth problem —
producing outputs that represent the full range of valid outputs in some reasonable way. The breadth
problem is notoriously difﬁcult, and it manifests itself in numerous ways in the methodology of
machine learning and generative models. The approach that proves our main result helps illustrate
the tension between validity and breadth even in settings with worst-case assumptions rather than
probabilistic ones, and this tension shows up in both the early phases of our algorithm’s execution
and the later phases. In the early phases, before the algorithm has reﬁned its provisional language
sufﬁciently, it is generating too broadly and producing strings that are not part of the target language
K — an analogue at a high level of a kind of hallucination in which the generated strings belong
to some consistent candidate language, but not to the actual target language [7, 8, 11]. In the later
phases, on the other hand, the algorithm continuously shrinks its range of possible outputs so as to
ensure that they will be contained within K — sacriﬁcing validity for breadth in a manner analogous
to the issues that arise in the problem of mode collapse for generative models [3, 4]. Our model
therefore suggests interesting questions about the fundamental trade-offs that may exist between
validity and breadth even in settings without an underlying probabilistic model.

2
Formal Model and Results

We now provide a formal description of the model and the statement of our results. To begin with,
we have a countable list of candidate languages C = {L1, L2, L3, . . .}, where each Li is a subset
of some countable set U. All we assume about the list of languages is that it is speciﬁed through a
black box that can answer questions of the form “Is w ∈Li?” for any string w ∈U and language
Li ∈C. (If the reader ﬁnds it helpful for concreteness, they can consider the results that follow in the
context of a speciﬁc list of languages C, such as the set of all context-free languages or the set of all
regular languages; but everything we say applies to general collections of languages.) We will allow
the collection C to contain repetitions, in that we may have Li = Lj for different indices i and j. We
will assume that all the languages Li are inﬁnite; while the original Gold-Angluin framework did not
require this, it becomes important in specifying the generation problem: if we require an algorithm to
output unseen strings forever, then this is not possible from a ﬁnite language, where the algorithm
would eventually run out of new strings to generate.

An adversary and an algorithm now play the following game. The adversary chooses a language
K from C without revealing it to the algorithm, and it begins enumerating the strings of K one by
one over a sequence of steps t = 1, 2, 3, . . .. The adversary can repeat strings in its enumeration, but
crucially, for every string w ∈K, there must be at least one time step t in which w appears. Let St
be the set of strings that the adversary has enumerated in steps 1 through t.

Identiﬁcation and Generation.
In this framework, we can now specify both the Gold-Angluin
problem of identiﬁcation and the contrasting problem of generation that we study in this paper.

3


---Page Break---
• Identiﬁcation (from [6, 2]: In each step, the algorithm observes St and must output an index
i (its guess for the true language K). The algorithm identiﬁes K in the limit if there is
some t∗such that for all steps t ≥t∗, the algorithm’s guess in step t is an index i for which
Li = K.

• Generation (from the present paper): In each step, the algorithm observes St and must
output a string at (its guess for an unseen string in K). The algorithm generates from K in
the limit if there is some t∗such that for all steps t ≥t∗, the algorithm’s guess at belongs to
K −St.

A key point in both problem formulations is that the algorithm is not told if its guesses are correct.

We know from the Gold-Angluin results that there is no algorithm that can achieve identiﬁcation in the
limit for an arbitrary countable collection C of languages (or even for speciﬁc countable collections,
like the set of all regular languages or the set of all context-free languages). Our main result is a
dramatically different answer for language generation; it is possible for every countable collection:
(2.1)
There is an algorithm with the property that for any countable collection of languages
C = {L1, L2, L3, . . .}, and any enumeration of one of these languages K, the algorithm generates
from K in the limit.

A Result for Finite Collections.
We prove a second result as well, focusing on the variant of the
problem in which the collection of languages C is ﬁnite. In this case, it follows from Angluin’s
characterization that every ﬁnite collection C allows for identiﬁcation in the limit. Given this, what
more might we ask for? A natural question is whether there is a uniform bound on the number of
samples needed to ensure that the algorithm can correctly identify the true language K: for any
ﬁnite collection C, is there a bound t(C) and an algorithm with the property that after seeing any t(C)
distinct strings from K, the algorithm is guaranteed to report K as its guess for the true language?

It is easy to see that for the Gold-Angluin model of learning, this is not possible. For example, supopse
that C is the collection consisting of two languages L1 and L2: L1 consists of all possible strings, and
L2 consists of all strings of even length. Suppose there were a bound t(C) and an algorithm that was
guaranteed to guess correctly after seeing t(C) distinct samples. Then an adversary could present t(C)
distinct strings of even length, and then ask the algorithm to guess whether the true language is L1 or
L2: if the algorithm guesses L2 at this point, then the adversary could announce that the answer is
L1, and conversely if the algorithm guesses L1. This does not prevent the algorithm from learning
the true language in the limit (since the adversary must eventually output an odd-length string if the
true answer is L1); but there is no ﬁxed bound t(C) by which this can be guaranteed.

However, for the problem of generation with a ﬁnite collection of candidate languages, we can
provide this much stronger type of uniform bound, via an algorithm that generates correctly after
seeing a ﬁnite sample whose size t(C) is speciﬁed at the outset. In fact, we can achieve more: after
seeing this ﬁnite sample, the algorithm can generate an inﬁnite sequence of unseen elements from the
true language.
(2.2)
There is an algorithm with the property that for any ﬁnite collection of languages C, there
is a number t(C), such that for any language K in C, and any sequence S of at least t(C) distinct
elements from K, the algorithm can produce an inﬁnite sequence of distinct strings from K −S.

In subsequent work building on the present paper, Raman and Tewari [10] have recently proved
further results about the type of uniform generation that we consider in (2.2), when it is possible to
put an a priori bound t(C) on the number of distinct strings an algorithm needs to see from K before
it can be guaranteed to begin generating unseen strings from K. They also consider variants of this
deﬁnition, situating their analysis in the framework of classical learning theory.

Extensions and Generalizations.
Following these two main results in our basic model of gener-
ation, we provide (in Section 7) the following generalization of the model. Speciﬁcally, a familiar
issue from language generation applications is the role of the prompt: a user provides an input string
p, and a generation algorithm is then supposed to produce a “continuation” string c to come after
the prompt, so that the concatenation of p and c is a valid utterance. We offer an extension that
incorporates the notion of prompting, while maintaining the basic structure of the model, and we
show how to formulate and prove a generalization of our ﬁrst main result (2.1) in a setting where at
each time step the adversary is allowed to specify a prompt that must be completed.

4


---Page Break---
3
An Approach to Generation that Doesn’t Work

The following section is important, because it describes an approach that is arguably the most natural
non-trivial attempt at achieving language generation in the limit. It seems at ﬁrst seems to solve
the problem directly, but in fact it fails for a deep reason that is important to understand, since it
motivates the more involved solution that follows.

The strategy is to move through the list of languages C = {L1, L2, L3, . . .} in order, treating each
language Li as a hypothesis for K until the sample St proves otherwise. That is, we start with L1,
and we generate strings from L1 −St until we encounter (if ever) a step t in which St ̸⊆L1. At
this point we know that L1 cannot be the true language K, and so we continue the process from L2.
The nice idea that underpins this strategy is that the true language K is equal to Lz for some index
z. (Since C can contain repetitions, K might appear several times, but we can take Lz as the ﬁrst
appearance.) So if our process were to reach Lz at some step t∗, it would never move on from Lz,
and so we would be generating from K −St for all t ≥t∗.

Unfortunately, there is a deep problem with this approach: there may be a language Li ∈C with the
property that Li comes before Lz and Li properly contains Lz (that is, i < z, and Lz ⊊Li). In this
case, our procedure would stop at the ﬁrst such Li forever, since it would never encounter a string in
St that didn’t belong to Li. And when it generated from Li −St, there is no guarantee that it would
choose strings from Lz.

This problem is not easily avoided, since if this approach worked as written, it would also solve
identiﬁcation in the limit, which we know is impossible. So we need to extract some of the useful
ideas from this failed approach — in particular, the trick that K appears at some ﬁnite point in the
list C, as the language Lz — but add important further ideas as well. Speciﬁcally, if the algorithm
is maintaining hypotheses for the true language K over time, it can provably never know whether
its current hypothesis is correct; instead, it must be always moving further down the collection of
languages, potentially considering languages that are not K, but in such a way that it is eventually
always generating from K −St. This is what our proof beginning in the next section will have to
accomplish.

4
Generation in the Limit via a Function

We prove our main result in two parts. We ﬁrst give a method for language generation in the limit
that is not concerned with the computational power required by the agent performing the generation.
Thus, rather than an algorithm to generate the string, we ask whether we can construct a function fC
based on the given language collection that maps a ﬁnite set of strings to a new string; this function
takes the strings St seen so far and outputs a string fC(St) intended to be in K −St. We will prove
the following:
(4.1)
For every countable collection of languages C, there is a function fC from ﬁnite subsets of U
to elements of U, such that for every enumeration of a language K ∈C, there is a t∗such that for all
t ≥t∗, we have fC(St) ∈K −St.

Note that while this positive result is not concerned with the computational power required to evaluate
fC, it already contains the core contrast with language identiﬁcation, which remains impossible even
if we simply ask for a function fC. In the next section, we will then go on to prove (2.1) by using
an algorithm that only performs standard computational steps and membership queries of the form
“w ∈Li?”

Minimal and critical languages.
As before, we will suppose z is an index such that Lz = K. We
say that a language Li is consistent with the sample at step t if St ⊆Li. An important idea, which is
implicit in our discussion of the failed approach at the end of Section 2, is that if Li ⊆Lj are both
consistent with St, then it is safer for an algorithm to generate from Li than from Lj: if w ∈Lj −St
then we must also have w ∈Li−St. This suggests that it would be useful to ﬁnd consistent languages
that are minimal with respect to inclusion: we would say that L is minimal if L ∈C is consistent with
St, and also L ⊆L′ for every L′ ∈C that is consistent with St. Unfortunately, this is too much to
ask for, since there exist instances of the problem for which there might not be any languages that
are minimal with respect to inclusion. (In a ﬁnite collection of language there would need to be a
minimal language, but it is easy to construct inﬁnite collections without one.)

5


---Page Break---
Therefore, we deﬁne a related concept that only involves examining the inclusion of a given language
with respect to a ﬁnite set of other languages. Speciﬁcally, we look for languages Ln that are
consistent with St in a given step t, such that Ln is a subset of every consistent language that precedes
it in the indexing of C. We will say that such a language is critical at step t. To deﬁne this formally,
we ﬁrst let Cn denote the ﬁnite collection of languages {L1, L2, . . . , Ln}. We now have the following
deﬁnition.
(4.2)
A language Ln is critical at step t if Ln is consistent with St, and for every language Li ∈Cn
that is consistent with St, we have Ln ⊆Li.

Properties of critical languages.
At any given step t, there is at least one language consistent with
St, since the language Lz = K is always consistent with St. It follows that there is also at least one
critical language at any step t: for any t, the consistent language Li with the lowest index i must be
critical at step t, as it is the only consistent language in Ci.

Note that there can be choices of t for which the language Lz = K is not critical at step t. But a
crucial fact is that Lz will eventually become critical at some step t and remain critical forever after
that. For reasons of space, we provide complete proofs for this and all subsequent results in the
Appendix, in this case in Section A.2.
(4.3)
There exists a time step t+ such that for all t ≥t+, the language Lz is critical at step t.

There can be multiple critical languages at a given step t; for example, if on the step t+ in (4.3) the
ﬁrst consistent language Li is not equal to Lz, then both Li and Lz will be critical at step t+. Despite
the potential multiplicity of critical languages, the collection of all critical languages at step t has a
useful nested structure: Li and Lj are both critical at step t, with i < j, then since Lj is contained in
all consistent languages that precede it, in particular it is contained in Li. We therefore have:
(4.4)
Let i < j, and suppose that Li and Lj are both critical at step t. Then Lj ⊆Li.

A function for generation in the limit.
At a given step t, suppose that the critical languages are
Ln1, Ln2, Ln3, . . . where n1 < n2 < n3 < · · · . (This list of critical languages might be ﬁnite or
inﬁnite.) Then (4.4) tells us that this sequence is nested by inclusion: Ln1 ⊇Ln2 ⊇Ln3 ⊇· · · .

By (4.3) we know that the language Lz will eventually appear on this nested list from some step t+
onward, but even then we do not know which index ni it corresponds to at any given step t ≥t+.
Indeed, to recall a point from earlier, the Gold-Angluin results for learning in the limit tell us that
we can never know for sure which index corresponds to Lz. But we now arrive at the crucial point,
which is that beyond some ﬁnite index, all the critical languages are subsets of Lz, so it is safe to
generate from any of them.

Given this, we are prepared to construct our function fC.
(4.5)
fC(St) is deﬁned as follows. We ﬁrst identify all languages in Ct that are critical at step t. (If
no such languages exist — which can only happen if none of them are consistent with St — we deﬁne
fC(St) arbitrarily.) Among these critical languages, let Lnt be the one with the largest index nt ≤t.
We deﬁne fC(St) to be the lowest-indexed element of Lnt −St.

Finally, to establish our initial claim (4.1), it is sufﬁcient to prove the following (in Section A.2):
(4.6)
For any language Lz ∈C and any enumeration of Lz, there is a t∗such that for all t ≥t∗,
we have fC(St) ∈Lz −St.

As a ﬁnal note, we observe that the current formulation of fC allows it to generate the same string
more than once, provided that this string is in K −St. However, it is not hard to modify fC so
that it generates a different string each time, essentially just by deﬁning it so that it generates the
lowest-indexed element that it hasn’t already generated.

The computational power required to produce fC.
Our plan was to construct fC without wor-
rying about the computational power required to do so (and recalling that for comparison, in the
corresponding problem of identiﬁcation in the limit, no function fC achieving identiﬁcation could
exist regardless of the computational power required to produce it). Now that we’ve constructed an
appropritate fC, we can ask what was in fact required computationally.

6


---Page Break---
In addition to standard computational steps and membership queries of the form “w ∈Li?”, the
deﬁnition of fC(St) requires that we identify the critical languages in Ct. From the deﬁnition, we
can do this provided we can answer a ﬁnite number of subset queries of the form “Li ⊆Lj?”. So an
algorithm augmented with the power to perform such subset queries can perform generation in the
limit.

In the next section, we will show how to remove the necessity for subset queries, so that generation in
the limit can be performed by an algorithm using only standard computational steps and membership
queries.

5
Generation in the Limit via an Algorithm

We now prove (2.1) by giving an algorithm that generates in the limit for any countable collection of
languages C, using only standard computational steps and membership queries of the form “w ∈Li?”

The set of possible strings U can be written as U = {u1, u2, u3, . . .}, and for simplicity we will also
use the language of the positive integers to describe U, treating ui as the number i. In an enumeration
of the true language Lz = K, let the sequence of strings that are enumerated step by step be denoted
w1, w2, w3, . . ..

Extending deﬁnitions to ﬁnite subsets of languages Li.
One important idea in designing the
algorithm is to work with ﬁnite subsets of the languages in C, gradually expanding the size of these
subsets. Thus, for a language Li ∈C and a number m, we will use Li[m] to denote the ﬁnite set
Li ∩{1, 2, 3, . . . , m}. We extend deﬁnition (4.2) of critical languages from the previous section to
handle ﬁnite subsets.
(5.1)
Let t and m be positive integers. A language Ln is (t, m)-critical if Ln is consistent with St,
and for every language Li ∈Cn such that Li is consistent with St, we have Ln[m] ⊆Li[m].

Since Ln ⊆Li implies that Ln[m] ⊆Li[m] for any m ≥1, we have the following analogue of (4.3).
(5.2)
There exists a time step t+ such that for all t ≥t+ and all m ≥1, the language Lz is
(t, m)-critical.

The analogue of (4.4) also still holds with this deﬁnition, using the same proof.
(5.3)
Let i < j and suppose that Li and Lj are both (t, m)-critical. Then Lj[m] ⊆Li[m].

Finally, there is a basic monotonicity property of (t, m)-criticality that is useful to write down
explicitly; we give the proof in Appendix A.3.
(5.4)
Suppose that Ln is (t, m)-critical, and m′ < m. Then Ln is (t, m′)-critical.

5.1
An algorithm for generation in the limit

We now describe an algorithm for generation in the limit. As before, St = {w1, w2, . . . , wt} is the
subset of K enumerated through step t, treating the wi as integers. We will consider the languages
in Ct in step t, and maintain an auxiliary variable mt, roughly corresponding to how large a preﬁx
Li[m] we consider from each language Li ∈Ct.

At the start of step t, we set mt = max(mt−1, wt); note that by induction this implies mt ≥
maxt′<t wt′. (Later we will increment mt, so we should think of it as a variable in the algorithm
whose value can be modiﬁed.) We then determine which Li ∈Ct are consistent with St; note that by
the deﬁnition of mt, it is sufﬁcient to perform membership queries for only the ﬁnite set of elements
in Li[mt] in order to do this. If there are no consistent languages in Ct, then we output a string
arbitrarily.

Otherwise, there is at least one language consistent with St, and so there is at least one (t, m)-critical
language for any choice of m, since the ﬁrst consistent language is (t, m)-critical for all m. Our goal
is to imitate the plan from (4.5) and generate a new string from the highest-indexed critical language.
But to do this, we have to ﬁnd a new string, and this will in general require performing additional
membership queries.

7


---Page Break---
Generating a string.
For any choice of m, let nt(m) be the maximum index of a (t, m)-critical
language from Ct; as noted above, nt(m) is well-deﬁned since we are in the case where at least one
language in Ct is consistent with St, and so the ﬁrst consistent language is (t, m)-critical for all m.
We now search for a string to generate as follows.

We maintain a counter m that begins at mt and gets incremented iteratively, with each iteration doing
the following:

• Increment m by 1.

• Perform membership queries to determine Li[m] for each Li ∈Ct. Note that since m ≥
mt ≥maxt′<t wt′, the determination of which languages in Ct are consistent with St does
not change when we do this.

• Determine which languages are (t, m)-critical, and from this determine nt(m). Note that
this only requires consulting the results of membership queries already performed.

• If um ∈Lnt(m), then output um and deﬁne mt = m. Otherwise, continue to the next
iteration.

Analyzing the algorithm.
As written, it is not immediately clear that the algorithm’s iterations in
step t will come to an end with an output string um, rather than running forever. But we can show the
algorithm does terminate, via the following claim proved in Section A.3.
(5.5)
In step t, the algorithm outputs a string after a ﬁnite sequence of operations.

Given that (5.5) establishes that the algorithm outputs a string in step t, it is useful to record an
additional property of the algorithm that follows directly from its construction.
(5.6)
In step t, there is an mt and an nt such that the algorithm outputs a string from Lnt[mt],
where Lnt is the (t, mt)-critical language with maximum index in Ct.

Finally, we have the natural analogue of (4.6), proved in Section A.3, from which our main result
(2.1) directly follows.
(5.7)
For any language Lz ∈C and any enumeration of Lz, there is a t∗such that for all t ≥t∗,
the algorithm generates a string in Lz −St.

Similar to the end of Section 4, it is straightforward to modify the algorithm so it generates strings
without repetition.

6
Generation for Finite Collections of Languages

We now turn to our second main result, (2.2), which derives a stronger conclusion for ﬁnite collections
of languages.

The proof of this result takes a different approach from what we used in the previous two sections;
for the problem in this section, at a given step t, the approach we follow is to take the ﬁnite sample
St seen so far and ask whether there are any additional strings w ̸∈St that are common to all the
languages in C that are consistent with St. If so, then we can output such a string and be sure of
satisfying the speciﬁcation for generation.

The closure operation.
We formalize this plan using the notion of a closure operation, as follows.
For a sequence of strings St from a language in C, we deﬁne the closure of St in C, denoted ⟨St⟩, to
be the intersection of all languages in C that are consistent with St. If there is a string in ⟨St⟩−St,
then it is always safe for the algorithm to generate such a string; by deﬁnition, it must be an unseen
string from the true language Lz. The closure operation would not have been useful for us in the case
of general inﬁnite collections of languages C, since there are instances of the general problem where
⟨St⟩= St and so the closure does not provide us with any proposals for new strings to generate. We
give an example of such an instance in Appendix A.6.

However, in the case of ﬁnite collections C such as we are considering in this section, the closure
operation becomes a powerful tool: the crux of proving (2.2) is to show that once St is large enough,
⟨St⟩must become inﬁnite, so we will have an inﬁnite set of options in ⟨St⟩−St to use.

8


---Page Break---
Proving the result for ﬁnite collections.
We start by giving the basic idea behind the proof, and
then the proof itself. Let us write the ﬁnite collection of candidate languages as C = {L1, L2, . . . , Ln},
and suppose that after the adversary has enumerated a set S of strings, the languages consistent with
S are Li1, Li2, . . . , Lik. Note that the true language K must be one of these k languages. Now, the
closure ⟨S⟩is equal to the mutual intersection Li1 ∩Li2 ∩· · · ∩Lik, and there are two possibilities:
either ⟨S⟩is inﬁnite, or it is ﬁnite. If it is inﬁnite, then the algorithm can safely generate all of the
strings in ⟨S⟩−S, and thus achieve the goal speciﬁed by (2.2). On the other hand, if ⟨S⟩is ﬁnite,
then it has size equal to some natural number m; in this case, after the adversary enumerates at most
m + 1 more distinct strings, the algorithm will learn that at least one of Li1, Li2, . . . , Lik is no longer
consistent. We will then have a set of at most k −1 consistent languages, and we can iterate this
argument at most k −2 more times until (i) there is only a single consistent language, which must be
K, or (ii) more generally, the set of all consistent languages has a mutual intersection that is inﬁnite,
in which case the algorithm can safely generate from this inﬁnite set.

This argument conveys the key point underlying the proof, that as the adversary enumerates strings
from K, it cannot prevent itself from reaching a point where the set of strings it has enumerated has
an inﬁnite closure. To turn this into an argument that produces a uniform bound on how many strings
are needed before the closure must become inﬁnite, we replace the iterative argument in the previous
paragraph with one that is shorter and more direct. Speciﬁcally, consider all sub-collections of
languages from C (where we think of a sub-collection as any way of choosing some of the languages
from C but not others). Note that since C is ﬁnite, there are only ﬁnitely many possible sub-collections
of languages from C. For each sub-collection C′, the mutual intersection of the languages in C′ is
either ﬁnite or inﬁnite. Consider the sub-collections that have a ﬁnite mutual intersection, and let m∗
be the maximum size of such a mutual intersection. Now, suppose the adversary produces a set S
of m∗+ 1 distinct strings from K. If we consider the sub-collection of all languages in C that are
consistent with S, its mutual intersection must contain S and therefore it has cardinality greater than
m∗. By the deﬁnition of m∗, this means that its cardinality must be inﬁnite. So the closure ⟨S⟩is
inﬁnite, and therefore the algorithm can safely generate all the strings in ⟨S⟩−S.

We give a formal version of this argument in Section A.4, providing the complete proof of (2.2).

7
Extension: Prompted Generation in the Limit

As discussed in Section 2, a natural generalization is to ask whether we can preserve the basic
structure of the model while adding a notion of prompting: the algorithm is provided with a prompt
string and it must complete it to a valid output. The overall set-up is as before: an adversary chooses
a true language K from C, and it begins enumerating the strings of K in a sequence of steps.

A model of prompting.
The new feature of the problem in our generalization is that in every step
t, the adversary provides the algorithm with two things: a string wt from the true language K, and a
string pt that serves as a prompt. (The adversary is allowed use the same prompt in multiple steps.)
The algorithm in step t must then produce a string ct with the goal that the concatenation of pt and ct
is a string belonging to K −St; that is, it should be an unseen string from K. In what follows, we
will use pt · ct to denote the concatenation of pt and ct.

We observe that it leads to an equivalent problem whether we ask the algorithm to output ct so that
pt · ct ∈K −St, or whether we ask the algorithm to output the full contatenated string at = pt · ct.
In this latter formulation, we can phrase the algorithm’s task as follows: given a prompt pt, output
a string at with the properties that pt is a preﬁx of at, and at ∈K −St. Because it makes the
exposition slightly simpler, we will take this latter formulation as our default way of describing the
problem, but the two formulations are equivalent.

To establish a positive result for prompted generation in the limit, we need to impose some type of
restriction on the prompts the adversary can provide. For example, if the adversary were simply to
provide an arbitrary string pt and ask the algorithm if there exists a string ct and a language Li ∈C
for which pt · ct ∈Li, this is not a problem that could be solved by an algorithm that must terminate
and whose only access to C comes in the form of membership queries of the form “Is w ∈Li?”

Here we describe a result based on the following restriction on the adversary’s prompts: We say a
prompt p is robust if for all languages Li ∈C, there exist arbitrarily long strings c for which p·c ∈Li.
In the present discussion we will consider adversaries that only provide robust prompts.

9


---Page Break---
We say that the algorithm achieves prompted generation from K in the limit if there is some t∗
such that for all steps t ≥t∗, the algorithm’s output at has the property that pt is a preﬁx of at and
at ∈K −St. We can prove the following.
(7.1)
There is an algorithm with the property that for any countable collection of languages
C = {L1, L2, L3, . . .}, and any enumeration of one of these languages K accompanied by a sequence
of robust prompts, the algorithm achieves prompted generation from K in the limit.

We provide a proof of this theorem in Section A.5 in the Appendix. In the full version of the paper we
also explore stronger statements that are based on giving the adversary more freedom in the prompts
it is allowed to provide.

We close with two initial observations about (7.1). First, (7.1) is a strict generalization of our ﬁrst
main result (2.1), since if the adversary always provides the empty string as its prompt pt, then the
problem of ﬁnding continuations ct for which pt · ct ∈K −St is simply the problem of ﬁnding
strings ct in K −St, as in the original deﬁnition of generation in the limit. Moreover, the empty
string is a robust prompt, since each of the languages Li ∈C is inﬁnite, and so there are arbitrarily
long continuation strings that belong to Li when concatenated to the empty string.

Second, we observe that there is no requirement that the algorithm has ever seen a string beginning
with the preﬁx pt among the adversary’s examples St ⊆K before the ﬁrst step t in which the
adversary provides pt. An important point to notice about (7.1) is that the algorithm can achieve
prompted generation in the limit despite this challenge.

8
Concluding Remarks

Generating from a language based on observed samples is thus a fundamentally different, more
tractable problem than identifying the language from observed samples. It is so tractable, in fact, that
it can be accomplished provided only that we know the samples come from a language in a known
countable collection of languages.

It is therefore interesting to ask what stylized conclusions we might draw from these general results
about generation as a task, and its relation to other learning processes. In the case of ﬁnite collections
of languages, the basic idea underlying the proof is that a large “core” to the language (the closure
of the sample, in our terminology) emerges at a known time after a ﬁnite set of observations, and it
is then enough to generate from this core even though there might always remain peripheral parts
of the language — disjoint from this core — that we can never be sure about. In the case of inﬁnite
collections of languages, the task is even more complex, because there is never a known time at which
a core to the language emerges. Instead, the algorithm may need to continually shrink the set it is
using for generation; through this inﬁnite process of shrinkage, the algorithm can be sure that beyond
a certain point, it is always generating from the true language K, even if it can not be sure when it
has reached this point or what the true language is.

Thus, as noted earlier, the solutions we develop highlight interesting tensions between the problem of
producing valid strings that belong to the target language, and the problem of maintaining breadth by
not restricting to only a small subset of the target language. Our approaches achieve validity through
a strategy that implicitly gives up on breadth, and it is interesting to ask if this is essentially necessary
for any method that achieves language generation in the limit. In this way, our formalism may also
help shed light on a particular broader impact of large language models, which is their implications
for the homogenization of text style, in which writing becomes less varied; our results provide further
indications that decreased breadth in text output may be a crucial feature of successful solutions to
text generation.

This tension between validity and breadth, as it arises in our solution, also creates an interesting echo
of the human process by which people acquire the vernacular within a new community [5]: as with
our solution in this abstract model, people encountering the dialect in a new community similarly pass
through a sequence of conceptually distinct phases: an initial phase in which they are generating too
adventurously and producing invalid utterances; then a phase where the utterances are approximately
aligned with the scope of the language; and ﬁnally a phase in which the range of utterances they are
willing to generate shrinks further over their lifetime, as they become increasingly conservative in
what they consider valid. Again, it is interesting to consider whether this type of structure is inherent
in any solution to the task of generation in the limit.

10


---Page Break---
Acknowledgements.
We thank Bobby Kleinberg, Lillian Lee, Marios Papachristou, and Kenny
Peng for helpful discussions on these questions and on early drafts of this paper. The work has been
supported in part by a Vannevar Bush Faculty Fellowship, a Simons Collaboration grant, a grant from
the MacArthur Foundation, and the Center for Applied AI at the University of Chicago Booth School
of Business.

References

[1] Dana Angluin. Finding patterns common to a set of strings. In Proceedings of the eleventh
annual ACM Symposium on Theory of Computing, pages 130–141, 1979.

[2] Dana Angluin. Inductive inference of formal languages from positive data. Information and
control, 45(2):117–135, 1980.

[3] Martín Arjovsky and Léon Bottou. Towards principled methods for training generative adversar-
ial networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon,
France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.

[4] Martín Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial
networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International
Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017,
volume 70 of Proceedings of Machine Learning Research, pages 214–223. PMLR, 2017.

[5] Cristian Danescu-Niculescu-Mizil, Robert West, Dan Jurafsky, Jure Leskovec, and Christopher
Potts. No country for old members: User lifecycle and linguistic change in online communities.
In Proceedings of the 22nd international conference on World Wide Web, pages 307–318, 2013.

[6] E Mark Gold. Language identiﬁcation in the limit. Information and control, 10(5):447–474,
1967.

[7] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM Computing Surveys, 55(12):1–38, 2023.

[8] Adam Tauman Kalai and Santosh S. Vempala. Calibrated language models must hallucinate. In
Bojan Mohar, Igor Shinkar, and Ryan O’Donnell, editors, Proceedings of the 56th Annual ACM
Symposium on Theory of Computing, STOC 2024, Vancouver, BC, Canada, June 24-28, 2024,
pages 160–171. ACM, 2024.

[9] Lillian Lee. Learning of context-free languages: A survey of the literature. Technical Report
TR-12-96, Harvard University, 1996.

[10] Vinod Raman and Ambuj Tewari. Generation through the lens of learning theory. arXiv preprint
arXiv:2410.13714, 2024.

[11] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation
of large language models. arXiv preprint arXiv:2401.11817, 2024.

11


---Page Break---
A
Appendix

A.1
Review of Negative Results for Identiﬁcation

In the interests of completeness, we provide a self-contained proof of the Gold-Angluin result that
language identiﬁcation in the limit is not possible in general. adapting the exposition from [6, 9]. As
noted in the main text, this proof is not necessary for the results in our paper, but it provides useful
background and context for thinking about the problem.

There are many ways to show the impossibility of language generation in the limit using very simple
language families, and we choose one that highlights some intuitive contrasts with generation. For
the argument, let U be the set of all integers, and let the collection of languages — each of which is
a subset of U — be the set of all inﬁnite arithmetic progressions of integers. (The choice of which
countable ground set U we use is not crucial for any of these results, and examples are often easier to
describe over the set of integers than over the set of ﬁnite strings.) Formally, for an arbitrary integer a
and a positive integer b, let Pa,b be the arithmetic progression consisting of all integers of the form
{a + bi : i = 0, 1, 2, . . .}; and let Qa,b be the “bidirectional” arithmetic progression consisting of
all integers of the form {a + bi : i = . . . , −2, −1, 0, 1, 2, . . .}. Let the collection C consist of all
arithmetic progressions Pa,b and Qa,b.

Now, suppose by way of contradiction that there is an algorithm that can identify in the limit an
adversarially chosen arithmetic progression K ∈C. We construct an enumeration of K that causes
the algorithm to fail, as follows. First, for integers a ≤b, let I[a, b] be the interval of all integers c for
which a ≤c ≤b. We enumerate elements of U in stages, where each stage s ≥0 consists of a set of
consecutive steps. If by induction stage s has enumerated the elements of the interval I[−s, j(s)] for
some j(s) ≥0, then stage s + 1 will enumerate additional elements so that by the end of the stage
we have enumerated exactly I[−(s + 1), j(s + 1)] for some j(s + 1) > j(s). In particular, stage
s + 1 ﬁrst enumerates −(s + 1) and j(s) + 1, and then it begins enumerating j(s) + 2, j(s) + 3, . . .
in increasing order. At some point during this process, the algorithm must output P−(s+1),1 as its
guess for K, since we can continue in this way to produce a full enumeration of P−(s+1),1, at which
point the true language is K = P−(s+1),1. Once the algorithm outputs P−(s+1),1 as its guess during
stage s + 1, we end stage s + 1, deﬁning j(s + 1) to be largest integer we’ve enumerated up to that
point, and we begin stage s + 2.

In this way, the successive stages extend the interval I[−s, j(s)] unboundedly in both directions.
We are therefore enumerating Q0,1 = U, and so in fact K = Q0,1. But we have also produced an
unbounded sequence of steps t0 < t1 < t2 < · · · such that the algorithm guesses P−j,1 at step tj.
Thus, there is no time step t∗for which the algorithm outputs the (correct) guess Q0,1 at every t ≥t∗.

As a ﬁnal note, we observe that for this simple collection of languages, the problem of generation in
the limit is straightforward: once the algorithm has seen two elements i and j from K, with i < j,
it knows from the arithmetic progression structure that by setting d = j −i, it can safely output
j + d, j + 2d, j + 3d, . . ., and all of these will belong to K. The generation task for this language
family is thus particularly simple (for others it is much more subtle, as in the example from Appendix
A.6), but this contrast with the negative result for identiﬁcation provides some initial intuition about
the differences between the two problems that we will need to exploit in a generation algorithm.

A.2
Proofs from Section 4

In the next three subsections of the Appendix, we provide proofs of the results stated in the main text.

Proof of (4.3).
Let K be the indices i < z for which Lz ̸⊆Li. For each i ∈K, let vi be an
element of Lz −Li. Let ti be the step in which vi ﬁrst appears in the enumeration of Lz, and let
t+ = maxi∈K ti.

Now, suppose by way of contradiction that for some t ≥t+, the language Lz is not critical at step t.
In this case, there must be some Li ∈Cz such that Li is consistent with St and Lz ̸⊆Li. But we
know that vi ∈St and vi ̸∈Li, contradicting the consistency of Li with St.
□

Proof of (4.6).
In the given enumeration of Lz, (4.3) tells us that there will come a step t+ such
that for all t ≥t+, the language Lz is critical at step t. Let t∗= max(z, t+). In every step t ≥t∗,

12


---Page Break---
our construction of fC will include Lz among its critical languages in Ct. Therefore, the highest-
indexed critical language Lnt ∈Ct satisﬁes nt ≥z, and so by (4.4) we have Lnt ⊆Lz. Since
fC(St) ∈Lnt −St, we have fC(St) ∈Lz −St as required.
□

A.3
Proofs from Section 5

Proof of (5.4). Since Ln is (t, m)-critical, we know it is consistent with St, and that Ln[m] ⊆Li[m]
for all languages Li ∈Cn such that Li is consistent with St. Now, if Li is a language in Cn that is
consistent with St, then since Ln[m] ⊆Li[m] and m′ < m, we have Ln[m′] ⊆Li[m′]. It follows
that Ln[m′] ⊆Li[m′] for all languages Li ∈Cn such that Li is consistent with St, and so Ln is
(t, m′)-critical.
□

Proof of (5.5).
We identify each iteration with the value of m after the initial increment of the
iteration; so the iterations begin at mt + 1 and continue upward from there. Suppose by way of
contradiction that the algorithm performs an inﬁnite sequence of iterations.

Let us call an iteration m disruptive if nt(m) ̸= nt(m −1). Since nt(m −1) is the maximum index
of a (t, m −1)-critical language, and since our monotonicity property (5.4) implies that Lnt(m) is
also (t, m −1)-critical, it follows that nt(m) < nt(m −1). Since nt(m) starts at a value bounded
by t and decreases by at least one with every disruptive iteration, there can be at most t −1 disruptive
iterations.

The sequence of iterations must therefore contain a last disruptive iteration m∗. For all iterations
m > m∗, the language Lnt(m) does not change. Since the language is inﬁnite, we must eventually
reach an iteration m > m∗for which um ∈Lnt(m), and the algorithm will stop and output um at
this point.
□

Proof of (5.7).
In the given enumeration of Lz, (5.2) tells us that there is a t+ such that for all
t ≥t+ and all m ≥1, the language Lz is (t, m)-critical. Let t∗= max(z, t+). In every step t ≥t∗,
by (5.6) there is an mt such that the algorithm generates a string from Lnt[mt], where Lnt is the
(t, mt)-critical language with maximum index in Ct. In each such step t, Lz is a (t, mt)-critical
language in Ct, and so nt ≥z. From (5.3), it follows that Lnt[mt] ⊆Lz[mt] ⊆Lz. Since the
algorithm’s output comes from Lnt[mt] −St, it follows that it comes from Lz −St as well.
□

A.4
Proofs from Section 6

Proof of (2.2).
We begin with some additional deﬁnitions. For any subset A of the indices
{1, 2, . . . , n}, let D(A) be the intersection of the languages whose indices are in A; in other words,
D(A) =
\

i∈A
Li. For any sequence S of strings from a language in C, let I(S) be the set of indices of

the languages in C that contain S; that is, I(S) = {i : S ⊆Li}. We observe that the closure operator
can be written in terms of this notation, in that ⟨S⟩= D(I(S)).

If D = D({1, 2, . . . , n}) is inﬁnite, then the algorithm can generate arbitrary strings from D as its
output without seeing any sample of strings at all; since D ⊆Li for every language Li ∈C, in
particular D ⊆Lz for the true language Lz, and this satisﬁes the requirements of (2.2).

For the rest of the proof, we therefore suppose D({1, 2, . . . , n}) is ﬁnite. Let F be the collection of all
sets of indices A ⊆{1, 2, . . . , n} with D(A) ﬁnite. Finally, let m∗= max
A∈F |D(A)|; since |F| ≤2n,

we observe that m∗is the maximum of a ﬁnite set of positive integers, and hence a positive integer.
We now deﬁne t(C) = m∗+ 1 and claim that this choice of t(C) satisﬁes the required guarantee
of (2.2). Indeed, consider the true language Lz ∈C and any sequence St of t(C) distinct elements
from Lz. Recall that I(St) denotes the set of indices of all languages in C that contain St. We have
St ⊆⟨St⟩= D(I(St)). If D(I(St)) were ﬁnite, then by the deﬁnition of m∗, the cardinality of
D(I(St)) would be at most m∗. But this would contradict the fact that D(I(St)) contains St, which
has cardinality m∗+ 1.

Therefore ⟨St⟩= D(I(St)) is inﬁnite, and it is a subset of the true language Lz. To conclude the
proof, we therefore need only show that there is an algorithm that can enumerate all of ⟨St⟩−St

13


---Page Break---
using only membership queries. To do this, the algorithm begins by querying whether each wi ∈St
belongs to each Lj ∈C. From this, it can determine the set I(St) of indices of languages that contain
St. Now, it enumerates every string ui ∈U in ascending order, skipping the elements of St. For each
such string ui, it queries whether ui ∈Lj for each j ∈I(St), and it outputs ui if it belongs to each
of these languages. In this way, the algorithm enumerates the inﬁnite set ⟨St⟩−St ⊆Lz −St after
seeing a sample of t(C) strings in Lz.
□

A.5
Proofs from Section 7

We now describe how to prove our result (7.1) on prompted generation with robust prompts. The
proof is a direct adaptation of the proof of (2.1) from Section 5; as we will see, the structure of critical
languages built up there is sufﬁciently strong that not much more is needed to handle the prompted
version of the problem with robust prompts.

As in Section 5, we will work with a speciﬁc enumeration u1, u2, u3, . . . of all strings in U, and work
with ﬁnite subsets of the languages Li, deﬁned via the notation Li[m] = Li ∩{u1, u2, . . . , um}. The
algorithm for prompted generation will closely follow the algorithm from Section 5, in that in every
step t, it will increment a counter m and maintain knowledge of the maximum index nt(m) of a
(t, m)-critical language from Ct. Maintaining knowledge of nt(m) does not require knowledge of
the prompts, and so this part of the algorithm is the same as before. What changes is the stopping
condition for the algorithm in step t: rather than continue increasing m until any valid output is found
— that is, until um ∈Lnt(m) — the algorithm must increase m potentially even further, until it ﬁnds
a string um for which pt is a preﬁx of um, and um ∈Lnt(m) −St. However, since pt is a robust
prompt, the algorithm is guaranteed to eventually ﬁnd such a string, and so we can be sure that its
iterations in step t will terminate. If we let mt be the value of m at the end of step t, then once t is
large enough, we know that Lnt(mt)[mt] ⊆Lz[mt], where Lz = K is the true language, and so the
string um that it outputs has pt as a preﬁx and belongs to K −St as required.

The discussion above provides the entire set of modiﬁcations to the algorithm; for completeness we
now describe these in more detail, together with a proof of correctness.

First, the facts (5.2) through (5.4) still hold in the prompted case, since they are structural properties
of the language that are not affected by the adversary’s use of prompts. The algorithm for generating
an output string uses an iteration in step t for which parts (i), (ii), and (iii) of each iteration are the
same as in Section 5. Step (iv) of each iteration is replaced by

(iv′) If there is any string ui for i ≤m such that ui has pt as a preﬁx and ui ∈Lnt(m) −St, then
choose the minimum i with this property; output the string ui and deﬁne mt = m. If there
is no such ui, then continue to the next iteration.

Now, the proof of termination works as before, by establishing that there are only ﬁnitely many
disruptive iterations in which the identity of nt(m) changes; this part does not depend on the structure
of prompts but only on the deﬁnition of a (t, m)-critical language, and so it uses (5.2) through (5.4)
exactly as before. After the last disruptive iteration, either there is a string ui ∈Lnt(m) −St with
i ≤m for which pt is a preﬁx, or else the algorithm will eventually reach one, since the prompt pt is
robust. It declares this um to be its output string. We therefore have
(A.1)
In step t, if at least one language in Ct is consistent with St, then there is an mt and an nt such
that the algorithm terminates with a string at for which pt is a preﬁx of at and at ∈Lnt(mt)[mt]−St,
where Lnt(mt) is the (t, mt)-critical language with maximum index in Ct.

Finally, we establish the basic correctness property of the algorithm, from which (7.1) follows directly.
(A.2)
For any language Lz ∈C and any enumeration of Lz with robust prompts p1, p2, p3, . . .,
there is a t∗such that for all t ≥t∗, the algorithm generates a string at for which pt is a preﬁx of at
and at ∈Lz −St.

Proof. In the given enumeration of Lz, (5.2) tells us that there is a t+ such that for all t ≥t+
and all m ≥1, the language Lz is (t, m)-critical. Let t∗= max(z, t+). In every step t ≥t∗, by
(A.1) there is an mt such that the algorithm generates a string at such that pt is a preﬁx of at, and
at ∈Lnt(mt)[mt] −St, where Lnt(mt) is the (t, mt)-critical language with maximum index in Ct.
In each such step t, Lz is a (t, mt)-critical language in Ct, and so nt(mt) ≥z. From (5.3), it follows

14


---Page Break---
that Lnt(mt)[mt] ⊆Lz[mt] ⊆Lz. Since at ∈Lnt(mt)[mt] −St, it follows that at ∈Lz −St as
well.
□

A.6
An Example Where Closure is Not Helpful

In Section 6, we introduced the notion of closure: For a sequence of strings St from a language in C,
the closure of St in C, denoted ⟨St⟩, is the intersection of all languages in C that are consistent with
St.

We observed in Section 6 that in any step t where ⟨St⟩contains an element not in St, the algorithm
can always safely output an element in ⟨St⟩−St and be sure that it is outputting a new element
in the true language K This is simply because for every consistent language Li ∈C, we have
⟨St⟩−St ⊆Li −St by the deﬁnition of the closure operation, and in particular this holds for the
true language K.

This strategy is key to the proof in Section 6, when we were dealing with ﬁnite collections C, and it
would have been an effective strategy in the case of inﬁnite collections C as well provided we could
be sure that ⟨St⟩−St was always non-empty. Unfortunately, there are instances of the problem with
inﬁnite collections C for which ⟨St⟩−St is empty for arbitrarily large t, and therefore the closure
provides us with no new strings to generate.

To see how this can happen in an example, we go back to the collection of arithmetic progresions
over the integers as the ground set U, and we consider a more complicated collection C that builds on
this. In particular, we let Pa,b as before be the arithmetic progression consisting of all integers of
the form {a + bi : i = 0, 1, 2, . . .}; and now, for any ﬁnite set of integers V , we deﬁne the language
L(a, b, V ) = Pa,b ∪V . Our collection C consists of every L(a, b, V ) for an arbitrary integer a, an
arbitrary positive integer b, and an arbitrary ﬁnite set of integers V . (Think of L(a, b, V ) as a copy of
the arithmetic progression Pa,b that has been obscured by an arbitrarily large ﬁnite set V so that its
structure is harder to discern.)

Now, suppose the adversary is enumerating a language K = L(a, b, V ) ∈C, and consider the set St
of samples after t steps. We claim that ⟨St⟩= St. Intuitively, this is because St might have come
completely from the ﬁnite set V that is part of L(a, b, V ); more formally, there cannot be any element
j ∈⟨St⟩−St because L(j + 1, 1, St) is consistent with St, and it does not contains j.

This example illustrates the sense in which we mean that closure by itself is insufﬁcient to provide a
strategy for generation in the limit when the collection of languages C is inﬁnite: in this instance, the
algorithm has to repeatedly output elements outside the closure of St and yet somehow eventually
generate from K. We know that the algorithm that proves our main result (2.1) is able to do this, but
it is also interesting to ask if there is a simpler way to directly describe a method for generation in the
limit for this particular example. In fact, there is a direct solution to this example, as follows: in any
step t, the algorithm ﬁnds the two largest elements i < j in St, and with b = j −i, it outputs j + b.
The point is that if the true language K is L(a, b, V ), there will come a time when the adversary
has enumerated every element of V , and within a ﬁnite number of steps after this, the two largest
elements of St will have to come from Pa,b. From this step onward, the algorithm’s strategy of
outputting j + b is guaranteed to produce an element in K −St. This particular strategy of course
seems highly speciﬁc to the particular example we are discussing here, but at a very high level it does
contain reﬂections of some of the ideas we saw in the general solution from Sections 4 and 5.

15


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: The paper’s main contributions are to formulate a model of language generation
in the limit, by analogy with the Gold-Angluin model of language identiﬁcation in the limit,
and to prove a set of theoretical results in this model. These contributions are described in
the abstract and introduction.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reﬂect how
much the results can be expected to generalize to other settings.
• It is ﬁne to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justiﬁcation: The paper’s main limitation, similar to the classical Gold-Angluin model, is
that it is a theoretical model based on worst-case guarantees, and as such does not try to
capture the role of distributional assumptions in the design of language models. This is both
a source of limitations but also (as a mitigation of these limitations) a source of insights
distinct from traditional probabilistic models. These issues are discussed in the paper’s
subsection entited General Connections to Language Modeling.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-speciﬁcation, asymptotic approximations only holding locally). The authors
should reﬂect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reﬂect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reﬂect on the factors that inﬂuence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efﬁciency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.

16


---Page Break---
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be speciﬁcally instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justiﬁcation: The model and its underlying assumptions are fully described, and all proofs
are provided in the Appendix.

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

Answer: [NA]

Justiﬁcation: The paper consists entirely of theoretical results. Due to the emphasis on
guarantees against an adversary in the limit, there is not a natural opening for demonstrating
the results experimentally (as there similarly isn’t in the work that introduced the Gold-
Angluin model).

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or veriﬁable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

17


---Page Break---
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

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justiﬁcation: The results of the paper do not make use of data or code.

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

Answer: [NA]

Justiﬁcation: As noted above, the paper does not include computational experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.

18


---Page Break---
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?

Answer: [NA]

Justiﬁcation: As noted above, the paper does not include computational experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
the main claims of the paper.
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
of Normality of errors is not veriﬁed.
• For asymmetric distributions, the authors should be careful not to show in tables or
ﬁgures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding ﬁgures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [NA]

Justiﬁcation: As noted above, the paper does not include computational experiments.

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

19


---Page Break---
Answer: [Yes]

Justiﬁcation: The NeurIPS Code of Ethics was reviewed in the course of writing the paper,
and has been fully adhered to.

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

Justiﬁcation: The paper studies theoretical questions motivated by the dramatic recent
advances in the performance of large language models (LLMs). As such, it connects to the
broader and rapidly growing discussion about the potentially disruptive societal implications
of LLMs and their widespread use, although it does so only through the formulation of
theoretical models. One point in the discussion of LLM broader impacts that our theoretical
results may address most directly is concerns over the homogenization of text style — the
ways in which language modeling may lead to decreased variety in the text that is produced
— and we discuss this in the section entitled Concluding Remarks.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact speciﬁc
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
feedback over time, improving the efﬁciency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justiﬁcation: The paper does not develop data or implementations that could be released.

20


---Page Break---
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety ﬁlters.
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

Justiﬁcation: The paper does not use existing computational assets; it does cite the theoretical
models that serve as the underlying motivation for the work.

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
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justiﬁcation: The paper does not release new assets; it does provide complete self-contained
descriptions of the theoretical models that it introduces.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip ﬁle.

21


---Page Break---
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

Justiﬁcation: The paper does not make use of crowdsourcing or any human-subject research.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is ﬁne, but if the main contribu-
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

Justiﬁcation: As noted above, the paper does not involve any research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

22


---Page Break---
