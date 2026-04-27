Online Weighted Paging with Unknown Weights

Orin Levy∗
Tel-Aviv University
orinlevy@mail.tau.ac.il

Noam Touitou
Amazon Science
noamtwx@gmail.com

Aviv Rosenberg†
Google Research
avivros007@gmail.com

Abstract

Online paging is a fundamental problem in the field of online algorithms, in which
one maintains a cache of 𝑘slots as requests for fetching pages arrive online. In the
weighted variant of this problem, each page has its own fetching cost; a substantial
line of work on this problem culminated in an (optimal) 𝑂(log 𝑘)-competitive
randomized algorithm, due to Bansal, Buchbinder and Naor (FOCS’07).
Existing work for weighted paging assumes that page weights are known in advance,
which is not always the case in practice. For example, in multi-level caching
architectures, the expected cost of fetching a memory block is a function of its
probability of being in a mid-level cache rather than the main memory. This
complex property cannot be predicted in advance; over time, however, one may
glean information about page weights through sampling their fetching cost multiple
times. We present the first algorithm for online weighted paging that does not
know page weights in advance, but rather learns from weight samples. In terms
of techniques, this requires providing (integral) samples to a fractional solver,
requiring a delicate interface between this solver and the randomized rounding
scheme; we believe that our work can inspire online algorithms to other problems
that involve cost sampling.

1
Introduction

Online weighted paging. In the online weighted paging problem, or OWP, one is given a cache of
𝑘slots, and requests for pages arrive online. Upon each requested page, the algorithm must ensure
that the page is in the cache, possibly evicting existing pages in the process. Each page 𝑝also has a
weight 𝑤𝑝, which represents the cost of fetching the page into the cache; the goal of the algorithm
is to minimize the total cost of fetching pages. Assuming that the page weights are known, this
problem admits an 𝑂(log 𝑘)-competitive randomized online algorithm, due to Bansal, Buchbinder,
and Naor [2010, 2012]; This is optimal, as there exists an Ω(log 𝑘)-competitiveness lower bound for
randomized algorithms due to Fiat et al. [1991] (that holds even for the unweighted case).

However, all previous work on paging assumes that the page weights are known in advance. This
assumption is not always justified; for example, the following scenario, reminiscent of real-world
architectures, naturally gives rise to unknown page weights. Consider a multi-core architecture, in
which data can be stored in one of the following: a local “L1” cache, unique to each core; a global
“L2” cache, shared between the cores; and the (large but slow) main memory. As a specific core
requests memory blocks, managing its L1 cache can be seen as an OWP instance. Suppose the costs

∗Research conducted while the author was an intern at Amazon Science.
†Research conducted while the author was at Amazon Science.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
of fetching a block from the main memory and from the L2 cache are 1 and 𝜖≪1, respectively.
Then, when a core demands a memory block, the expected cost of fetching this block (i.e., its weight)
is a convex combination of 1 and 𝜖, weighted by the probability that the block is in the L2 cache; this
probability can be interpreted as the demand for this block by the various cores. When managing
the L1 cache of a core, we would prefer to evict blocks with low expected fetching cost, as they are
more likely to be available in the L2 cache. But, this expected cost is a complicated property of the
computation run by the cores, and estimating it in advance is infeasible; however, when a block is
fetched in the above example, we observe a stochastic cost of either 1 or 𝜖. As we sample a given
block multiple times, we can gain insight into its weight.

Multi-armed bandit. The above example, in which we learn about various options through sampling,
is reminiscent of the multi-armed bandit problem, or MAB. In the cost-minimization version of this
problem, one is given 𝑛options (or arms), each with its own cost in [0, 1]. At each time step, the
algorithm must choose an option and pay the corresponding cost; when choosing an option 𝑝, rather
than learning its cost 𝑤𝑝, the algorithm is only revealed a sample from some distribution whose
expectation is 𝑤𝑝. In this problem, the goal is to minimize the regret, which is the difference between
the algorithm’s total cost and the optimal cost (which is to always choose the cheapest option). Over
𝑇time steps, the best known regret bound for this problem is ˜𝑂(
√

𝑛𝑇), achieved through multiple
techniques. (See, e.g., Slivkins et al. [2019], Lattimore and Szepesvári [2020]).

1.1
Our Results

We make the first consideration of OWP where page weights are not known in advance, and show
that the optimal competitive ratio of 𝑂(log 𝑘) can still be obtained. Specifically, we present the
problem of OWP-UW (Online Weighted Paging with Unknown Weights), that combines OWP with
bandit-like feedback. In OWP-UW, every page 𝑝has an arbitrary distribution, whose expectation is
its weight 0 < 𝑤𝑝≤1. Upon fetching a page, the algorithm observes a random, independent sample
from the distribution of the page. We present the following theorem for OWP-UW.
Theorem 1.1. There exists a randomized algorithm ON for OWP-UW such that, for every input 𝑄,

E[ON(𝑄)] ≤𝑂(log 𝑘) · OPT(𝑄) + ˜𝑂(
√

𝑛𝑇),

where ON(𝑄) is the cost of ON on 𝑄, OPT(𝑄) is the cost of the optimal solution to 𝑄, and the
expectation is taken over both the randomness in ON and the samples from the distributions of pages.

Note that the bound in Theorem 1.1 combines a competitive ratio of 𝑂(log 𝑘) with a regret (i.e.,
additive) term of ˜𝑂(
√

𝑛𝑇). To motivate this type of bound, we observe that OWP-UW does not
admit sublinear regret without a competitive ratio. Consider the lower bound of Ω(log 𝑘) for the
competitive ratio of paging; stated simply, one of 𝑘+ 1 pages of weight 1 is requested at random.
Over a sequence of 𝑇requests, the expected cost of any online algorithm is Ω(𝑇/𝑘); meanwhile, the
expected cost of the optimal solution is at most 𝑂(𝑇/(𝑘log 𝑘)). (The optimal solution would be to
wait for a maximal phase of requests containing at most 𝑘pages, whose expected length is Θ(𝑘log 𝑘),
then change state at constant cost.) Without a competitive ratio term, the difference between the
online and offline solutions is Ω(𝑇/𝑘), i.e., linear regret. We note that this kind of bound appears in
several previous works such as Basu et al. [2019], Foussoul et al. [2023]. As OWP-UW generalizes
both standard OWP and MAB, both the competitive ratio and regret terms are asymptotically tight: a
competitiveness lower bound of Ω(log 𝑘) is known for randomized algorithms for online (weighted)
paging [Fiat et al., 1991], and a regret lower bound of ˜Ω(
√

𝑛𝑇) is known for MAB [Lattimore and
Szepesvári, 2020].

1.2
Our Techniques

Interface between fractional solution and rounding scheme. Randomized online algorithms are
often built of the following components:

1. A deterministic, 𝛼-competitive online algorithm for a fractional relaxation of the problem.
2. An online randomized rounding scheme that encapsulates any online fractional algorithm,
and has expected cost 𝛽times the fractional cost.

Combining these components yields an 𝛼𝛽-competitive randomized online (integral) algorithm.

2


---Page Break---
For our problem, it is easy to see where this common scheme fails. The fractional algorithm cannot
be competitive without sampling pages; but, pages are sampled by the rounding scheme! Thus, the
competitiveness of the fractional algorithm is not independent of the randomized rounding, which
must provide samples. One could think of addressing this by feeding any samples obtained by the
rounding procedure into the fractional algorithm. However, as the rounding is randomized, this would
result in a non-deterministic fractional algorithm. As described later in the paper, this is problematic:
the rounding scheme demands a globally accepted fractional solution against which probabilities of
cache states are balanced.

Instead, we outline a sampling interface between the fractional solver and the rounding scheme.
Once the total fractional eviction of a page reaches an integer, the fractional algorithm will pop a
sample of the page from a designated sampling queue, and process that sample. On the other side
of the interface, the rounding scheme fills the sampling queue and ensures that when the fractional
algorithm demands a sample, the queue will be non-empty with probability 1.

Optimistic fractional algorithm, pessimistic rounding scheme. When learning from samples,
one must balance the exploration of unfamiliar options and the exploitation of familiar options that
are known to be good. A well-known paradigm for achieving this balance in multi-armed bandit
problems is optimism under uncertainty. Using this paradigm to minimize total cost, one maintains
a lower confidence bound (LCB) for the cost of an option, which holds with high probability, and
tightens upon receiving samples; then, the option with the lowest LCB is chosen. As a result, one
of the following two cases holds: either the option was good (high exploitation); or, the option was
bad, which means that the LCB was not tight, and henceforth sampling greatly improves it (high
exploration).

Our fractional algorithm for weighted paging employs this method. It optimistically assumes that the
price of moving a page is cheap, i.e., is equal to some lower confidence bound (LCB) for that page.
It then uses multiplicative updates to allocate servers according to these LCB costs. The optimism
under uncertainty paradigm then implies that the fractional algorithm learns the weights over time.

However, the rounding scheme behaves very differently. Unlike the fractional algorithm, the (ran-
domized) rounding scheme is not allowed to use samples to update the confidence bounds; otherwise,
our fractional solution would behave non-deterministically. Instead, the rounding scheme takes a
pessimistic view: it uses an upper confidence bound (UCB) as the cost of a page, thus assuming
that the page is expensive. Such pessimistic approaches are common in scenarios where obtaining
additional samples is not possible (e.g., offline reinforcement learning [Levine et al., 2020]), but
rarely appear as a component of an online algorithm as we suggest in this paper.

1.3
Related Work

The online paging problem is a fundamental problem in the field of online algorithms. In the
unweighted setting, the optimal competitive ratio for a deterministic algorithm is 𝑘, due to Sleator
and Tarjan [1985]. Allowing randomization improves the best possible competitive ratio to Θ(log 𝑘)
[Fiat et al., 1991]. As part of a line of work on weighted paging and its variants (e.g., Young [1994],
Manasse et al. [1990], Albers [2003], Irani [2002], Fiat and Mendel [2000], Bansal et al. [2008], Irani
[1997]), the best competitive ratios for weighted paging were settled, and were seen to match the
unweighted setting: 𝑘-competitiveness for deterministic algorithms, due to Chrobak et al. [1991];
and Θ(log 𝑘)-competitiveness for randomized algorithms, due to Bansal et al. [2012].

Online (weighted) paging is a special case of the 𝑘-server problem, in which 𝑘servers exist in a
general metric space, and must be moved to address requests on various points in this space; the
cache slots in (weighted) paging can be seen as servers, moving in a (weighted) uniform metric
space. The Θ(𝑘) bound on optimal competitiveness in the deterministic for paging also extends to
general 𝑘-server [Manasse et al., 1990, Koutsoupias and Papadimitriou, 1995]. However, allowing
randomization, a recent breakthrough result by Bubeck et al. [2023] was a lower bound of Ω(log2 𝑘)-
competitiveness for 𝑘-server, diverging from the 𝑂(log 𝑘)-competitiveness possible for paging.

Multi-Armed Bandit (MAB) is one of the most fundamental problems in online sequential decision
making, often used to describe a trade-off between exploration and exploitation. It was extensively
studied in the past few decades, giving rise to several algorithmic approaches that guarantee optimal
regret. The most popular methods include Optimism Under Uncertainty (e.g., the UCB algorithm
[Lai and Robbins, 1985, Auer et al., 2002a]), Action Elimination [Even-Dar et al., 2006], Thompson

3


---Page Break---
Sampling [Thompson, 1933, Agrawal and Goyal, 2012] and Exponential Weights (e.g., the EXP3
algorithm [Auer et al., 2002b]). For a comprehensive review of the MAB literature, see Slivkins et al.
[2019], Lattimore and Szepesvári [2020].

2
Preliminaries

In OWP-UW, we are given a memory cache of 𝑘slots. A sequence of 𝑇page requests then arrives in
an online fashion; we denote the set of requested pages by 𝑃, define 𝑛:= |𝑃|, and assume that 𝑛> 𝑘.
Each page 𝑝has a corresponding weight 0 < 𝑤𝑝≤1; the weights are not known to the algorithm.
Moreover, every page 𝑝has a distribution D𝑝supported in (0, 1], such that E𝑥∼D𝑝[𝑥] = 𝑤𝑝.

The online scenario proceeds in 𝑇rounds.3 In each round 𝑡∈{1, 2, . . . ,𝑇}:

• A page 𝑝𝑡∈𝑃is requested.

• If the requested page is already in the cache, then it is immediately served.

• Otherwise, we experience a cache miss, and we must fetch 𝑝𝑡into the cache; if the cache is
full, the algorithm must evict some page from the cache to make room for 𝑝𝑡.

• Upon evicting any page 𝑝from the cache, the algorithm receives an independent sample
from D𝑝.

The algorithm incurs cost when evicting pages from the cache: when evicting a page 𝑝, the algorithm
incurs a cost of 𝑤𝑝4. Our goal is to minimize the algorithm’s total cost of evicting pages, denoted
by ON, and we measure our performance by comparison to the total cost of the optimal algorithm,
denoted by OPT. We say that our algorithm is 𝛼-competitive with R regret if E[ON] ≤𝛼· OPT + R.

3
Algorithmic Framework and Analysis Overview

We present an overview of the concepts and algorithmic components we use to address OWP-UW.
We would like to follow the paradigm of solving a fractional problem online, and then randomly
rounding the resulting solution; however, as discussed in the introduction, employing this paradigm for
OWP-UW requires a well-defined interface between the fractional solver and the rounding procedure.
Thus, we present a fractional version of OWP-UW that captures this interface.

Fractional OWP-UW. In fractional OWP-UW, one is allowed to move fractions of servers, and
a request for a page is satisfied if the total server fraction at that point sums to 1. More formally,
for every page 𝑝∈𝑃we maintain an amount 𝑦𝑝∈[0, 1] which is the fraction of 𝑝missing from
the cache; we call 𝑦𝑝the fractional anti-server at 𝑝. (The term anti-server comes from the related
𝑘-server problem.) The feasibility constraints are:

1. At any point in the algorithm, it holds that Í
𝑝∈𝑃𝑦𝑝≥𝑛−𝑘. (I.e., the total number of pages
in the cache is at most 𝑘.)

2. After a page 𝑝is requested, it holds that 𝑦𝑝= 0. (I.e., there exists a total server fraction of 1
at 𝑝.)

Evicting an 𝜖server fraction from 𝑝(i.e., increasing 𝑦𝑝by 𝜖) costs 𝜖· 𝑤𝑝.

Sampling. The fractional algorithm must receive samples of pages over time in order to learn about
their weights. An algorithm for fractional OWP-UW receives a sample of a page 𝑝whenever the
total fraction of 𝑝evicted by the algorithm reaches an integer. In particular, the algorithm obtains the
first sample of 𝑝(corresponding to 0 eviction) when 𝑝is first requested in the online input.

3We make a simplifying assumption that 𝑇is known in advanced. This can be easily removed using a
standard doubling (see, e.g., Slivkins et al. [2019]).
4Note that charging an OWP solution for evicting rather than fetching pages is standard; indeed, with the
exception of at most 𝑘pages, every fetched page is subsequently evicted, and thus the difference between
eviction and fetching costs is at most 𝑘. Moreover, as we analyze additive regret, note that 𝑘≤
√

𝑛𝑇, implying
that using fetching costs would not affect the bounds in this paper. Finally, note that we sample upon eviction
rather than upon fetching, which is the “harder” model.

4


---Page Break---
Algorithmic components. We present the fractional algorithm and randomized rounding scheme.

Fractional algorithm. In Section 4, we present an algorithm ONF for fractional OWP-UW. Fixing the
random samples from the pages’ weight distributions, the fractional algorithm ONF is deterministic.
For every page 𝑝∈𝑃, the fractional algorithm maintains an upper confidence bound UCB𝑝and a
lower confidence bound LCB𝑝. These confidence bounds depend on the samples provided for that
page; we define the good event E to be the event that at every time and for every page 𝑝∈𝑃, it holds
that LCB𝑝≤𝑤𝑝≤UCB𝑝. We later show that E happens with high probability, and analyze the
complementary event separately5. Thus, we henceforth focus on the good event.

The following lemma bounds the cost of ONF subject to the good event. In fact, it states a stronger
bound, that applies also when the cost of evicting page 𝑝is the upper confidence bound UCB𝑝≥𝑤𝑝.
Lemma 3.1. Fixing any input 𝑄for fractional OWP-UW, and assuming the good event, it holds that

ONF(𝑄) ≤ONF(𝑄) ≤𝑂(log 𝑘) · OPT(𝑄) + ˜𝑂(
√

𝑛𝑇)

where ONF is the cost of the algorithm on the input where the cost of evicting a page 𝑝is UCB𝑝≥𝑤𝑝.

Randomized rounding. In Section 5 we present the randomized algorithm ON for (integral) OWP-UW.
It maintains a probability distribution over integral cache states by holding an instance of ONF, to
which it feeds the online input. For the online input to constitute a valid fractional input, the
randomized algorithm ensures that samples are provided to ONF when required. In addition, the
randomized algorithm makes use of ONF’s exploration of page weights; specifically, it uses the
UCBs calculated by ONF.
Lemma 3.2. Fixing any input 𝑄for (integral) OWP-UW, assuming the good event E, it holds that

E[ON(𝑄)] ≤𝑂(1) · ONF(𝑄) + 𝑛

where ONF(𝑄) is the cost of the algorithm on 𝑄such that the cost of evicting a page 𝑝is UCB𝑝≥𝑤𝑝.

Figure 1 provides a step-by-step visualization of the interface between the fractional algorithm and
the rounding scheme over the handling of a page request.

4
Algorithm for Fractional OWP-UW

We now describe our algorithm for the fractional relaxation of OWP-UW, proving Lemma 3.1. Our
fractional algorithm, presented in Algorithm 1 below, uses samples provided by the rounding scheme
to learn the weights. A new sample for page 𝑝is provided and processed whenever the sum of
fractional movements (in absolute value) 𝑚𝑝hits a natural number. (At this point the number of
samples 𝑛𝑝is incremented.) The algorithm calculates non-increasing UCBs and non-decreasing
LCBs that will be specified later in Section C and guarantee with high probability, for every page
𝑝∈𝑃and time step 𝑡∈[1,𝑇], LCB𝑝≤𝑤𝑝≤UCB𝑝.

At each time step 𝑡, upon a new page request 𝑝𝑡, the algorithm updates its feasible fractional cache
solution {𝑦𝑝}𝑝∈𝑃. The fractions are computed using optimistic estimates of the weights, i.e., the
LCBs, in order to induce exploration and allow the true weights to be learned over time. After serving
page 𝑝𝑡(that is, setting 𝑦𝑝𝑡= 0), the algorithm continuously increases the anti-servers of all the
other pages in the cache until feasibility is reached (that is, until Í
𝑝∈𝑃𝑦𝑝= 𝑛−𝑘). The fraction
𝑦𝑝for some page 𝑝in the cache is increased proportionally to 𝑦𝑝+𝜂

LCB𝑝, which is our adaption of the
algorithmic approach of Bansal et al. [2010] to the unknown-weights scenario. Finally, to fulfil its
end in the interface, the fractional algorithm passes its feasible fractional solution to the rounding
scheme together with pessimistic estimates of the weights, i.e, the UCBs.

4.1
Analysis

In this analysis section, our goal is to bound the amount ONF with respect to the UCBs and LCBs
calculated by the algorithm; i.e., to prove Lemma 4.1. Lemma C.1 and Lemma C.2 from Appendix C
then make the choice of confidence bounds concrete, such that combining it with Lemma 4.1 yields
the final bound for the fractional algorithm, i.e., Lemma 3.1.

5Specifically, we show that the complementary event E happens with probability at most
1
𝑛𝑇, and that the
algorithm’s cost is at most 𝑛𝑇times the optimal cost when it happens.

5


---Page Break---
This figure visualizes the running of the algorithm over a request in the input. (a) shows the state prior to the
arrival of the request. The integral algorithm maintains an instance of the fractional algorithm ONF, as well as a
distribution over integral cache states that upholds some properties w.r.t. ONF (specifically, the consistency
property and the subset property); note that these properties are a function of both the anti-server state

𝑦𝑝
	
𝑝∈𝑃
and the upper-confidence bounds

UCB𝑝
	
𝑝∈𝑃in ONF. The integral algorithm also maintains a set of page
samples, to be demanded by ONF at a later time. In (b), a page is requested (in red); thus, the fractional
algorithm must fetch it into the cache, i.e., set its anti-server to 0. To maintain feasibility, the fractional
algorithm will increase the anti-server at other pages in which some server fraction exists (in green). These
changes in anti-server are also fed into the integral algorithm, which modifies its distribution to maintain
consistency and the subset property w.r.t. ONF. In (c), ONF reaches integral total eviction of a page 𝑝, and
demands a sample from the integral algorithm. (We show that such a sample always exists when demanded.)
The bound UCB𝑝is updated in ONF, and is then fed to the integral algorithm to maintain the desired properties.
After this sample, continuous increasing of anti-server continues until feasibility is reached in (d). Then, the
integral algorithm ensures that a sample exists for the requested page 𝑝𝑡, sampling the page if needed. (As 𝑝𝑡
now exists in 𝜇with probability 1, sampling is done through evicting and re-fetching 𝑝𝑡.)

Figure 1: Visualization of the interface between the fractional and integral algorithms

Algorithm 1: Fractional Online Weighted Caching with Unknown Weights

1 Set 𝜂←1/𝑘and 𝑦𝑝←1 for every 𝑝∈𝑃.

2 for time 𝑡= 1, 2, ...,𝑇do

3
Page 𝑝𝑡∈𝑃is requested.

4
continually increase 𝑦𝑝, 𝑚𝑝in proportion to 𝑦𝑝+𝜂

LCB𝑝for every 𝑝∈𝑃\ {𝑝𝑡} where 𝑦𝑝< 1 until:

5
if 𝑚𝑝reaches an integer for some 𝑝∈𝑃\ {𝑝𝑡} then

6
receive sample e𝑤𝑝for 𝑝.

7
set 𝑛𝑝←𝑛𝑝+ 1.

8
call UPDATECONFBOUNDS(𝑝, e𝑤𝑝). // recalculate confidence bounds for 𝑝

9
if Í
𝑝∈𝑃𝑦𝑝= 𝑛−𝑘then break from the continuous increase loop.

10
if this is the first request for 𝑝𝑡then

11
receive sample e𝑤𝑝𝑡for 𝑝𝑡.

12
define 𝑚𝑝←0, 𝑛𝑝←1.

13
call UPDATECONFBOUNDS(𝑝𝑡, e𝑤𝑝𝑡). // calculate confidence bounds for 𝑝𝑡

Lemma 4.1. Fixing any input 𝑄for fractional OWP-UW, and assuming the good event, it holds that

ONF(𝑄) ≤𝑂(log 𝑘) · OPT(𝑄) +
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1

 UCB𝑝,𝑖−LCB𝑝,𝑖
 + 2 log(1 + 1/𝜂)
∑︁

𝑝∈𝑃
LCB𝑝.

where (a) ONF is the cost of the algorithm on the input such that the cost of evicting a page 𝑝
is UCB𝑝≥𝑤𝑝, and (b) UCB𝑝,𝑖, LCB𝑝,𝑖are the values of UCB𝑝and LCB𝑝calculated by the
procedure UPDATECONFBOUNDS (found in Appendix C) immediately after processing the 𝑖’th
sample of 𝑝, and (c) LCB𝑝is the value after the last sample of page 𝑝was processed.

Proof sketch. We prove the lemma using a potential analysis. In Bansal et al. [2010], a potential
function was introduced that encodes the discrepancy between the state of the optimal solution and
the state of the algorithm. In our case, we require an additional term which can be viewed as a
fractional exploration budget. This budget is “recharged” upon receiving a sample; the cost of this
recharging goes into a regret term.
□

6


---Page Break---
This figure visualizes a single step in REBALANCESUBSETS. Subfigure (a) shows the distribution of anti-cache
states prior to this step; specifically, the 𝑥axis is the probability measure, and the 𝑦-axis is the number of pages
of class 𝑖and above in the anti-cache, i.e., 𝑚:= |𝑆∩𝑃≥𝑖|. The red line is 𝑌𝑖, which through consistency, is the
expectation of 𝑚; the blue dotted lines are thus the allowed values for 𝑚, which are |⌈𝑌𝑖⌉, ⌊𝑌𝑖⌋|. The total striped
area in the figure is the imbalance measure, formally defined in Definition B.2. Subfigure (b) shows a single
rebalancing step; we choose the imbalanced anti-cache 𝑆that maximizes |𝑚−𝑌𝑖|; in our case, 𝑚> 𝑌𝑖, and thus
we match its measure with an identical measure of anti-cache states that are below the upper blue line, i.e., can
receive a page without increasing imbalance. Then, a page of class 𝑖is handed from 𝑆to every matched state 𝑆′;
note that every matched state might get a different page from 𝑆, but some such page in 𝑆\ 𝑆′ is proven to exist.
Finally, Subfigure (c) shows the state after the page transfer; note the decrease in imbalance that results. The
REBALANCESUBSETS procedure performs such steps until there is no imbalance; then, the procedure would
advance to class 𝑖−1.

Figure 2: Example of a rebalancing step

5
Randomized Rounding

This section describes a randomized algorithm for (integral) OWP-UW, which uses Algorithm 1 for
fractional OWP-UW to maintain a probability distribution over valid integral cache states, while
obtaining and providing page weight samples to Algorithm 1. The method in which the randomized
algorithm encapsulates and tracks the fractional solution is inspired by Bansal et al. [2012], which
maintains a balanced property over weight classes of pages. However, as the weights are unknown in
our case, the classes are instead defined using the probabilistic bounds maintained by the fractional
solution (i.e., the UCBs). But, these bounds are dynamic, and change over the course of the algorithm;
the imbalance caused by these discrete changes increases exponentially during rebalancing, and thus
requires a more robust rebalancing procedure.

Following the notation in the previous sections, we identify each cache state with the set of pages not
in the cache. Observing the state of the randomized algorithm at some point in time, let 𝜇(𝑆) be the
probability that 𝑆⊆𝑃is the set of pages missing from the cache, also called the anti-cache. For the
algorithm to be a valid algorithm for OWP-UW, the cache can never contain more than 𝑘pages; this
is formalized in the following property.

Definition 5.1 (valid distribution). A probability distribution 𝜇is valid, if for any set 𝑆⊆𝑃with
𝜇(𝑆) > 0 it holds that |𝑆| ≥𝑛−𝑘.

Instead of maintaining the distribution’s validity, we will maintain a stronger property that implies
validity. This property is the balanced property, involving the UCBs calculated by the fractional
algorithm.

For every page 𝑝, we define the 𝑖’th UCB class to be 𝑃𝑖:=

𝑝∈𝑃: 6𝑖≤UCB𝑝< 6𝑖+1	
. (Note that
UCB𝑝∈(0, 1].) We also define 𝑃≥𝑗:= Ð
𝑖≥𝑗𝑃𝑗, the set of all pages that their UCB is at least 6 𝑗.

Let {𝑦𝑝}𝑝∈𝑃be the fractional solution. The balanced property requires that, for every set 𝑆such
that 𝜇(𝑆) > 0 and every index 𝑗, the number of pages in 𝑆of class at least 𝑗is the same as in the
fractional solution, up to rounding. Formally, we define the balanced property as follows.

Definition 5.2 (balanced distribution). A probability distribution 𝜇has the balanced subsets property
with respect to 𝑦, if for any set 𝑆⊆𝑃with 𝜇(𝑆) > 0, the following holds for all 𝑗:
$ ∑︁

𝑖≥𝑗

∑︁

𝑝∈𝑃𝑖
𝑦𝑝

%

≤
∑︁

𝑖≥𝑗

∑︁

𝑝∈𝑃𝑖
I[𝑝∈𝑆] ≤

& ∑︁

𝑖≥𝑗

∑︁

𝑝∈𝑃𝑖
𝑦𝑝

'

.

7


---Page Break---
Algorithm 2: Randomized rounding algorithm for OWP-UW

1 Initialization

2
Let ONF be an instance of Algorithm 1, that maintains a fractional anti-server allocation

𝑦𝑝
	
𝑝∈𝑃.

3
Define 𝜇to be a distribution over cache states, initially containing the empty cache state with
probability 1.

4
For every 𝑝∈𝑃, let 𝑠𝑝←NULL.

5 Event Function UPONREQUEST(𝑝) // called upon a request for page 𝑝

6
pass the request for 𝑝to ONF.

7
while ONF is handling the request for 𝑝do // loop of Line 4 in Alg. 1

8
if ONF increases 𝑦𝑝′ by 𝜖, for some 𝑝′ ∈𝑃then

9
add 𝑝′ to the anti-cache in an 𝜖-measure of states without 𝑝′.

10
call REBALANCESUBSETS.

11
if ONF decreases 𝑦𝑝′ by 𝜖, for some 𝑝′ ∈𝑃then

12
remove 𝑝′ from the anti-cache in an 𝜖-measure of states with 𝑝′.

13
call REBALANCESUBSETS.

14
if ONF samples a page 𝑝′ ∈𝑃then // sample due to Line 6 of Alg. 1

15
provide 𝑠𝑝′ as a sample to ONF, and set 𝑠𝑝′ ←NULL.

16
call REBALANCESUBSETS. // rebalance due to change in UCB𝑝′.

17
if 𝑠𝑝= NULL then evict and re-fetch 𝑝to obtain weight sample ˜𝑤𝑝, and set 𝑠𝑝←˜𝑤𝑝.

18
if ONF requests a sample of 𝑝then // sample due to Line 11 of Alg. 1

19
provide 𝑠𝑝as a sample to ONF, and set 𝑠𝑝←NULL.

20
call REBALANCESUBSETS. // rebalance due to change in UCB𝑝.

Algorithm 3: Rebalancing procedure for randomized algorithm

1 Function REBALANCESUBSETS

2
let 𝑗max be the maximum class that is not balanced.

3
let 𝑗min := ⌈log(UCBmin)⌉, where UCBmin := min𝑝∈𝑃UCB𝑝.

4
for every class 𝑗, let 𝑃𝑗:=

𝑝∈𝑃

log(UCB𝑝)

= 𝑗
	
.

5
for 𝑗from 𝑗max down to 𝑗min do

6
let 𝑃≥𝑗:= Ð
𝑗′≥𝑗𝑃𝑗′.

7
let 𝑌𝑗:= Í
𝑝∈𝑃≥𝑗𝑦𝑝.

8
while ∃𝑆s.t. 𝜇(𝑆) > 0 and |𝑆∩𝑃≥𝑗| ∉

𝑌𝑗

,

𝑌𝑗
	
do // iteratively eliminate imbalanced states

9
choose such 𝑆that maximizes
𝑚−𝑌𝑗
, where 𝑚:=
𝑆∩𝑃≥𝑗
.

10
if 𝑚≥

𝑌𝑗

+ 1 then

11
Match the 𝜇(𝑆) measure of 𝑆with an identical measure of anti-cache states with at most

𝑌𝑗

−1 pages from 𝑃≥𝑗.

12
foreach anti-cache state 𝑆′ matched with 𝑆at measure 𝑥≤𝜇(𝑆) do

13
identify a page 𝑝∈𝑃𝑗such that 𝑝∈𝑆\ 𝑆′.

14
remove 𝑝from the anti-cache in the 𝑥measure of 𝑆, and insert it into the anti-cache in
the 𝑥measure of 𝑆′.

15
if 𝑚≤

𝑌𝑗

−1 then

16
Match the 𝜇(𝑆) measure of 𝑆with an identical measure of anti-cache states with at least

𝑌𝑗

+ 1 pages from 𝑃≥𝑗.

17
foreach anti-cache state 𝑆′ matched with 𝑆at measure 𝑥𝑆′ ≤𝜇(𝑆) do

18
identify a page 𝑝∈𝑃𝑗such that 𝑝∈𝑆′ \ 𝑆.

19
remove 𝑝from the anti-cache in the 𝑥measure of 𝑆′, and insert it into the anti-cache
in the 𝑥measure of 𝑆.

8


---Page Break---
Choosing the minimum UCB class in Definition 5.2, and noting that Í
𝑝∈𝑃𝑦𝑝≥𝑛−𝑘through
feasibility, immediately yields the following remark.
Remark 5.3. Every balanced probability distribution is also a valid distribution.

To follow the fractional solution, we also demand that the distribution 𝜇is consistent with the
fractional solution, meaning, the marginal probability in 𝜇that any page 𝑝is missing from the cache
must be equal to 𝑦𝑝.

Definition 5.4 (consistent distribution). A probability distribution 𝜇on subsets 𝑆⊆𝑃is consistent
with respect to a fractional solution

𝑦𝑝
	

𝑝∈𝑃if for every page 𝑝it holds that Í
𝑆⊆𝑃| 𝑝∈𝑆𝜇(𝑆) = 𝑦𝑝.

In the following we describe the online maintenance of the distribution 𝜇, that yields a distribution
satisfying all of the above.

Algorithm overview. The randomized algorithm for OWP-UW is given in Algorithm 2. The
algorithm encapsulates an instance of Algorithm 1 for fractional OWP-UW, called ONF. Upon a
new page request 𝑝𝑡at round 𝑡, the algorithm forwards this requests to ONF. As ONF makes changes
to its fractional solution, the algorithm modifies its probability distribution accordingly to remain
consistent and balanced (and hence also valid).

Upon any (infinitesimally small) change to a fractional variable, the algorithm first changes its
distribution to maintain consistency: when the fractional algorithm ONF increases the variable 𝑦𝑝′ of
any page 𝑝′ by an 𝜖-measure, the algorithm identifies an 𝜖-measure of cache states 𝑆⊆𝑃in which
there is no anti-server at 𝑝′ and adds anti-server at 𝑝′. The case in which the fractional algorithm
decreases a variable is analogous.

However, this procedure may invalidate the balanced property. Specifically, for some class 𝑗,
letting 𝑌𝑗be the total anti-server fraction of pages of at least class 𝑗in ONF, there might now be
states with

𝑌𝑗

+ 1 or

𝑌𝑗

−1 such pages in the anti-cache. Thus, the algorithm makes a call to
REBALANCESUBSETS, which restores the balanced property class-by-class, in a descending order.
For every class 𝑗, the procedure repeatedly identifies a violating state 𝑆where the number of pages of
class ≥𝑗in the anti-cache is not in

𝑌𝑗

,

𝑌𝑗
	
; suppose it identifies such a state with more than

𝑌𝑗


such pages (the case of less than

𝑌𝑗
 pages is analogous). The procedure seeks to move a page of
class 𝑗from this state to another state in a way that does not increase the “imbalance” in class 𝑗. Thus,
the procedure identifies a matching measure of anti-cache states that contain at most

𝑌𝑗

−1 such
pages, and moves a page of class 𝑗from 𝑆to 𝑆′, for every 𝑆′ in the matched measure; a visualization
of the procedure is given in Figure 2. (The existence of this matching measure, as well as a page to
move, are shown in the analysis.) In particular, note that the probability of every page being in the
anti-cache remains the same, and thus REBALANCESUBSETS does not impact consistency.

Regarding samples, the algorithm can maintain a sample 𝑠𝑝for every page 𝑝. A sample for 𝑝is
obtained upon a request for 𝑝after 𝑝is fetched with probability 1 into the cache, if no such sample
already exists (i.e., 𝑠𝑝= NULL). Whenever ONF requests a sample for a page 𝑝, the randomized
algorithm provides the sample 𝑠𝑝, and sets the variable 𝑠𝑝to be NULL (we show that 𝑠𝑝is never
NULL when ONF samples 𝑝). A fine point is the sampling of a new page in Line 11 of Algorithm 1;
this happens after Line 17 of Algorithm 2.

6
Conclusions

In this paper, we presented the first algorithm for online weighted paging in which page weights
are not known in advance, but are instead sampled stochastically. In this model, we were able to
recreate the best possible bounds for the classic online problem, with an added regret term typical
to the multi-armed bandit setting. This unknown-costs relaxation makes sense because the problem
has recurring costs; that is, the cost of evicting a page 𝑝can be incurred multiple times across the
lifetime of the algorithm, and thus benefits from sampling.

We believe this paper can inspire future work on this problem. For example, revisiting the motivating
case of managing a core-local L1 cache, the popularity of a page among the cores can vary over time;
this would correspond to the problem of non-stationary bandits (see, e.g., Auer et al. [2019b,a], Chen
et al. [2019]), and it would be interesting to apply techniques from this domain to OWP-UW.

9


---Page Break---
Finally, we hope that the techniques outlined in this paper could be extended to additional such
problems. Specifically, we believe that the paradigm of using optimistic confidence bounds in lieu
of actual costs could be used to adapt classical online algorithms to the unknown-costs setting. In
addition, the interface between the fractional solver and rounding scheme could be used to mediate
integral samples to an online fractional solver, which is a common component in many online
algorithms.

Acknowledgments

This project has received funding from the European Research Council (ERC) under the European
Union’s Horizon 2020 research and innovation program (grant agreement No. 882396), by the Israel
Science Foundation, the Yandex Initiative for Machine Learning at Tel Aviv University and a grant
from the Tel Aviv University Center for AI and Data Science (TAD).

10


---Page Break---
References

S. Agrawal and N. Goyal. Analysis of thompson sampling for the multi-armed bandit problem. In
Conference on learning theory, pages 39–1. JMLR Workshop and Conference Proceedings, 2012.

S. Albers. Online algorithms: a survey. Math. Program., 97(1-2):3–26, 2003. doi: 10.1007/
s10107-003-0436-0. URL https://doi.org/10.1007/s10107-003-0436-0.

P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem.
Machine learning, 47:235–256, 2002a.

P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit
problem. SIAM journal on computing, 32(1):48–77, 2002b.

P. Auer, Y. Chen, P. Gajane, C. Lee, H. Luo, R. Ortner, and C. Wei. Achieving optimal dynamic
regret for non-stationary bandits without prior information. In A. Beygelzimer and D. Hsu,
editors, Conference on Learning Theory, COLT 2019, 25-28 June 2019, Phoenix, AZ, USA,
volume 99 of Proceedings of Machine Learning Research, pages 159–163. PMLR, 2019a. URL
http://proceedings.mlr.press/v99/auer19b.html.

P. Auer, P. Gajane, and R. Ortner. Adaptively tracking the best bandit arm with an unknown
number of distribution changes. In A. Beygelzimer and D. Hsu, editors, Conference on Learning
Theory, COLT 2019, 25-28 June 2019, Phoenix, AZ, USA, volume 99 of Proceedings of Machine
Learning Research, pages 138–158. PMLR, 2019b. URL http://proceedings.mlr.press/
v99/auer19a.html.

N. Bansal, N. Buchbinder, and J. Naor. Randomized competitive algorithms for generalized caching.
In C. Dwork, editor, Proceedings of the 40th Annual ACM Symposium on Theory of Computing,
Victoria, British Columbia, Canada, May 17-20, 2008, pages 235–244. ACM, 2008. doi: 10.1145/
1374376.1374412. URL https://doi.org/10.1145/1374376.1374412.

N. Bansal, N. Buchbinder, and J. S. Naor. A simple analysis for randomized online weighted paging.
Unpublished Manuscript, 2010.

N. Bansal, N. Buchbinder, and J. Naor. A primal-dual randomized algorithm for weighted paging.
J. ACM, 59(4):19:1–19:24, 2012. doi: 10.1145/2339123.2339126. URL https://doi.org/10.
1145/2339123.2339126.

S. Basu, R. Sen, S. Sanghavi, and S. Shakkottai. Blocking bandits. Advances in Neural Information
Processing Systems, 32, 2019.

S. Bubeck, C. Coester, and Y. Rabani. The randomized k-server conjecture is false! In B. Saha
and R. A. Servedio, editors, Proceedings of the 55th Annual ACM Symposium on Theory of
Computing, STOC 2023, Orlando, FL, USA, June 20-23, 2023, pages 581–594. ACM, 2023. doi:
10.1145/3564246.3585132. URL https://doi.org/10.1145/3564246.3585132.

Y. Chen, C. Lee, H. Luo, and C. Wei. A new algorithm for non-stationary contextual bandits: Efficient,
optimal and parameter-free. In A. Beygelzimer and D. Hsu, editors, Conference on Learning
Theory, COLT 2019, 25-28 June 2019, Phoenix, AZ, USA, volume 99 of Proceedings of Machine
Learning Research, pages 696–726. PMLR, 2019. URL http://proceedings.mlr.press/
v99/chen19b.html.

M. Chrobak, H. J. Karloff, T. H. Payne, and S. Vishwanathan. New results on server problems.
SIAM J. Discret. Math., 4(2):172–181, 1991. doi: 10.1137/0404017. URL https://doi.org/
10.1137/0404017.

E. Even-Dar, S. Mannor, Y. Mansour, and S. Mahadevan. Action elimination and stopping conditions
for the multi-armed bandit and reinforcement learning problems. Journal of machine learning
research, 7(6), 2006.

A. Fiat and M. Mendel. Better algorithms for unfair metrical task systems and applications. In
F. F. Yao and E. M. Luks, editors, Proceedings of the Thirty-Second Annual ACM Symposium on
Theory of Computing, May 21-23, 2000, Portland, OR, USA, pages 725–734. ACM, 2000. doi:
10.1145/335305.335408. URL https://doi.org/10.1145/335305.335408.

11


---Page Break---
A. Fiat, R. M. Karp, M. Luby, L. A. McGeoch, D. D. Sleator, and N. E. Young. Competitive paging
algorithms. J. Algorithms, 12(4):685–699, 1991. doi: 10.1016/0196-6774(91)90041-V. URL
https://doi.org/10.1016/0196-6774(91)90041-V.

A. Foussoul, V. Goyal, O. Papadigenopoulos, and A. Zeevi. Last switch dependent bandits with
monotone payoff functions. In International Conference on Machine Learning, pages 10265–10284.
PMLR, 2023.

S. Irani. Page replacement with multi-size pages and applications to web caching. In F. T. Leighton
and P. W. Shor, editors, Proceedings of the Twenty-Ninth Annual ACM Symposium on the Theory
of Computing, El Paso, Texas, USA, May 4-6, 1997, pages 701–710. ACM, 1997. doi: 10.1145/
258533.258666. URL https://doi.org/10.1145/258533.258666.

S. Irani. Randomized weighted caching with two page weights. Algorithmica, 32(4):624–640, 2002.
doi: 10.1007/s00453-001-0095-6. URL https://doi.org/10.1007/s00453-001-0095-6.

E. Koutsoupias and C. H. Papadimitriou. On the k-server conjecture. J. ACM, 42(5):971–983, 1995.
doi: 10.1145/210118.210128. URL https://doi.org/10.1145/210118.210128.

T. L. Lai and H. Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied
mathematics, 6(1):4–22, 1985.

T. Lattimore and C. Szepesvári. Bandit algorithms. Cambridge University Press, 2020.

S. Levine, A. Kumar, G. Tucker, and J. Fu. Offline reinforcement learning: Tutorial, review, and
perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

M. S. Manasse, L. A. McGeoch, and D. D. Sleator. Competitive algorithms for server problems. J.
Algorithms, 11(2):208–230, 1990. doi: 10.1016/0196-6774(90)90003-W. URL https://doi.
org/10.1016/0196-6774(90)90003-W.

D. D. Sleator and R. E. Tarjan. Amortized efficiency of list update and paging rules. Commun. ACM,
28(2):202–208, 1985. doi: 10.1145/2786.2793. URL https://doi.org/10.1145/2786.2793.

A. Slivkins et al. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning,
12(1-2):1–286, 2019.

W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of the
evidence of two samples. Biometrika, 25(3-4):285–294, 1933.

N. E. Young. The k-server dual and loose competitiveness for paging. Algorithmica, 11(6):525–541,
1994. doi: 10.1007/BF01189992. URL https://doi.org/10.1007/BF01189992.

12


---Page Break---
A
Analysis of the Fractional Algorithm

In this analysis section, our goal is to bound the amount ONF with respect to the UCBs and LCBs
calculated by the algorithm; i.e., to prove Lemma 4.1. Lemma C.1 and Lemma C.2 from Appendix C
then make the choice of confidence bounds concrete, such that combining it with Lemma 4.1 yields
the final bound for the fractional algorithm, i.e., Lemma 3.1.

Proof of Lemma 4.1. For the sake of this lemma, we assume without loss of generality that the
optimal solution is lazy; that is, it only evicts a (single) page in order to fetch the currently-requested
page. (It is easy to see that any solution can be converted into a lazy solution of lesser or equal cost.)
In the following we present a potential analysis to prove that ONF ≤2 log(1 + 𝑘) · OPT + U𝑇, where
U𝑇is a regret term summed over 𝑇time steps that will be defined later. To that end, we show that
the following equation holds for every round 𝑡,

ΔONF𝑡+ ΔΦ𝑡≤2 log(1 + 1/𝜂) · ΔOPT𝑡+ ΔU𝑡,
(1)

where Δ𝑋𝑡is the change in 𝑋in time 𝑡and Φ is a potential function that we define next.

Let 𝐶∗
𝑡denote the set of pages in the offline (optimal) cache at time 𝑡. The potential function we
chose is an adaptation of the potential function used by Bansal et al. [2010], but modified to encode
the uncertainty cost for not knowing the true weights. We define it as follows.

Φ𝑡= −2
∑︁

𝑝∉𝐶∗
𝑡
LCB𝑝· log
 𝑦𝑝+ 𝜂

1 + 𝜂


+
∑︁

𝑝∈𝑃

 UCB𝑝−LCB𝑝
 ·  𝑛𝑝−𝑚𝑝
,

where UCB𝑝, LCB𝑝are the confidence bounds in time step 𝑡, 𝑛𝑝is the number of samples of page 𝑝
collected until that point, and 𝑚𝑝is the total fractional movement of that page until that point. Note
that 𝑛𝑝−𝑚𝑝∈[0, 1]. We now show that Equation 1 holds in the three different cases in which the
costs of the potential or the regret change. Before moving forward, we define the regret term, denoted
U, as follows,

U :=
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1
(UCB𝑝,𝑖−LCB𝑝,𝑖) + 2 log(1 + 1/𝜂)
∑︁

𝑝∈𝑃
LCB𝑝.

We note that each time the algorithm gets a new sample, 𝑛𝑝is incremented and U increases.

Case 1 - the optimal algorithm moves.
Note that a change in the cache of the optimal solution
does not affect ONF or U, and thus ΔONF = ΔU = 0. However, ΔOPT might be non-zero, as
the optimal solution incurs a moving cost; in addition, ΔΦ might be non-zero, as the change in the
optimal cache might affect the summands in the first term of the potential function.

Thus, proving Equation (1) for this case reduces to proving ΔΦ𝑡≤2 log(1 + 1/𝜂)ΔOPT. Assume the
optimal solution moves; as we assume that the optimal solution is lazy, it must be that the requested
page 𝑝𝑡is not in 𝐶∗
𝑡−1, and that the optimal solution fetches it into the cache, possibly evicting a
(single) page from its cache.

First, consider the case in which there exists an empty slot in the cache, and no evictions take place
when fetching 𝑝𝑡. In this case, ΔOPT = 0; moreover, as 𝐶∗
𝑡−1 ⊆𝐶∗
𝑡, it holds that ΔΦ ≤0. Thus,
Equation (1) holds.

Otherwise, at time 𝑡, page 𝑝𝑡is fetched and another page 𝑝is evicted. Thus, ΔOPT = 𝑤𝑝. For the
potential change, 𝑝starts to contribute to Φ. In the worst case, 𝑦𝑝= 0 and then Φ is increased by at
most 2LCB𝑝log(1 + 1/𝜂) ≤2𝑤𝑝log(1 + 1/𝜂) = 2 log(1 + 1/𝜂)ΔOPT, as desired.

Case 2 - the fractional algorithm moves.
In this case, only the potential and the cost of the
fractional algorithm change, i.e., ΔOPT𝑡= ΔU𝑡= 0. Thus, Equation (1) reduces to proving
ΔONF𝑡+ ΔΦ𝑡≤0 in this case.

There are two types of movement made by the algorithm: the immediate fetching of the requested
page 𝑝𝑡, and the continuous eviction of other pages from the cache until feasibility is reached (i.e.,
there are 𝑛−𝑘anti-servers). First, consider the fetching of 𝑝𝑡into the cache; as we charge for

13


---Page Break---
evictions, this action does not incur cost (ΔONF = 0). In addition, through OPT’s feasibility, it
holds that 𝑝𝑡∈𝐶∗
𝑡, and thus changing 𝑦𝑝𝑡does not affect the potential function (ΔΦ = 0); thus,
Equation (1) holds for this sub-case.

Next, consider the continuous eviction of pages. Suppose the fractional algorithm evicts 𝑑𝑦page
units, where 𝑑𝑦is infinitesimally small. Let 𝑆= {𝑝𝑡} ∪

𝑝: 𝑦𝑝< 1
	
. Since the fractional algorithm
increases 𝑦𝑝proportionally to 𝑦𝑝+𝜂

LCB𝑝for each 𝑝∈𝑆\ {𝑝𝑡}, it holds that, 𝑑𝑦𝑝=
1
𝑁· 𝑦𝑝+𝜂

LCB𝑝𝑑𝑦, for

𝑁:= Í
𝑝∈𝑆\{ 𝑝𝑡}
𝑦𝑝+𝜂
LCB𝑝. Thus, ΔONF can be bounded as follows.

ΔONF ≤
∑︁

𝑝∈𝑆\{ 𝑝𝑡}
UCB𝑝𝑑𝑦𝑝

=
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

 LCB𝑝+  UCB𝑝−LCB𝑝
𝑑𝑦𝑝

=
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

 LCB𝑝+  UCB𝑝−LCB𝑝
 1

𝑁
𝑦𝑝+ 𝜂
LCB𝑝
𝑑𝑦

=
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

1
𝑁
 𝑦𝑝+ 𝜂𝑑𝑦+
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

 UCB𝑝−LCB𝑝
𝑑𝑦𝑝

=
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

1
𝑁
 𝑦𝑝+ 𝜂𝑑𝑦

|                       {z                       }

(𝑎)

+
∑︁

𝑝∈𝑃

 UCB𝑝−LCB𝑝
𝑑𝑦𝑝

|                           {z                           }

(𝑏)

.

where the final equality stems from having 𝑑𝑦𝑝= 0 for every 𝑝∉𝑆\ {𝑝𝑡}. We now bound the term
(𝑎) using similar arguments to those presented in Bansal et al. [2010]. We present them below for
completeness.

(𝑎) =
∑︁

𝑝∈𝑆\{ 𝑝𝑡}

1
𝑁
 𝑦𝑝+ 𝜂𝑑𝑦≤(|𝑆| −𝑘)

𝑁
𝑑𝑦+ (|𝑆| −1)𝜂

𝑁
𝑑𝑦≤2(|𝑆| −𝑘)

𝑁
𝑑𝑦,

where the first inequality implied since 𝑦𝑝= 1 for 𝑝∉𝑆, which yields that
∑︁

𝑝∈𝑆\{ 𝑝𝑡}
𝑦𝑝≤
∑︁

𝑝∈𝑆
𝑦𝑝=
∑︁

𝑝∈𝑃
𝑦𝑝−
∑︁

𝑝∉𝑆
𝑦𝑝≤(|𝑃| −𝑘) −(|𝑃| −|𝑆|) = |𝑆| −𝑘,

and the second inequality implied as (𝑥−1)𝜂≤𝑥−𝑘for any 𝑥≥𝑘+ 1 and using that |𝑆| ≥𝑘+ 1.

Next, we upper bound ΔΦ𝑡. The rate of change in the potential with respect to 𝑦𝑝is

𝑑Φ𝑡
𝑑𝑦𝑝
= −2 LCB𝑝

𝑦𝑝,𝑡+ 𝜂− UCB𝑝−LCB𝑝
.

The non-trivial part in the above calculation is that
𝑑(UCB𝑝−LCB𝑝)·(𝑛𝑝−𝑚𝑝)

𝑑𝑦𝑝
= − UCB𝑝−LCB𝑝
.
This is true since 𝑛𝑝is uncorrelated with the change in 𝑦𝑝, however, 𝑚𝑝is increasing with with
respect to 𝑦𝑝in 1-linear ratio. Using the above we get that

ΔΦ𝑡=
∑︁

𝑝∈𝑃

 𝑑Φ𝑡

𝑑𝑦𝑝


𝑑𝑦𝑝

= −2
∑︁

𝑝∈𝑆\𝐶∗
𝑡

LCB𝑝
𝑦𝑝+ 𝜂
1
𝑁
𝑦𝑝+ 𝜂
LCB𝑝
𝑑𝑦−
∑︁

𝑝∈𝑃

 UCB𝑝−LCB𝑝
 · 𝑑𝑦𝑝

= −2
∑︁

𝑝∈𝑆\𝐶∗
𝑡

𝑑𝑦

𝑁−
∑︁

𝑝∈𝑃

 UCB𝑝−LCB𝑝
 · 𝑑𝑦𝑝

≤−2 |𝑆| −𝑘

𝑁
𝑑𝑦−
∑︁

𝑝∈𝑃

 UCB𝑝−LCB𝑝
 · 𝑑𝑦𝑝.

Thus, ΔONF𝑡+ ΔΦ𝑡≤0, as required.

14


---Page Break---
Case 3 - the LCBs are updated (a new sample is processed).
Suppose the algorithm samples a
page 𝑝for the 𝑖’th time, and updates UCB𝑝, LCB𝑝accordingly. Note that this sample does not incur
any cost for the fractional algorithm or optimal solution, and thus ΔONF = ΔOPT = 0. Thus, proving
Equation (1) reduces to proving ΔΦ ≤ΔU for this case. Recalling the definition of U, it holds that

ΔU = (UCB𝑝,𝑖−LCB𝑝,𝑖) + 2 log (1 + 1/𝜂) (LCB𝑝,𝑖−LCB𝑝,𝑖−1).

Denote by 𝑛𝑝, 𝑚𝑝the variables of that name prior to sampling 𝑝; note that 𝑚𝑝remains the same
after sampling, while 𝑛𝑝is incremented to 𝑛′
𝑝:= 𝑛𝑝+ 1. Moreover, note that sampling always occurs
when 𝑚𝑝= 𝑛𝑝. Thus, the change in potential function is bounded as follows.

ΔΦ = 2 log(1 + 1/𝜂)I[𝑝∉𝐶∗
𝑖] LCB𝑝,𝑖−LCB𝑝,𝑖−1
 + UCB𝑝,𝑖−LCB𝑝,𝑖
≤2 log(1 + 1/𝜂) LCB𝑝,𝑖−LCB𝑝,𝑖−1
 + UCB𝑝,𝑖−LCB𝑝,𝑖
where the second inequality is due to the LCBs being monotone non-decreasing.
□

B
Proofs from Section 5

B.1
Analysis and Proof of Lemma 3.2

In this subsection, we analyze Algorithm 2 and prove Lemma 3.2. Throughout this subsection, we
assume the good event E; that is, the UCBs and LCBs generated by ONF throughout the algorithm
are valid upper and lower bounds for the weights of pages.

We start by proving that the algorithm is able to provide a new sample whenever the fractional
algorithm requires one.
Proposition B.1. Algorithm 2 provides page weight samples whenever demanded by ONF.

Proof. We must prove that whenever a sample of page 𝑝is requested by ONF, it holds that the
variable 𝑠𝑝in Algorithm 2 is not NULL. Note that Algorithm 2 samples 𝑠𝑝at the end of a request for
𝑝. Now, note that:

• The first sample of 𝑝is requested after the first request for 𝑝is handled by ONF, and thus
after 𝑝is sampled.

• Between two subsequent requests for samples of 𝑝by ONF, the eviction fraction 𝑚𝑝
increased by 1. But, this cannot happen without a request for 𝑝in the interim; this request
ensures that 𝑠𝑝≠NULL, and thus the second request is satisfied.

Combining both cases, all sample requests by ONF are satisfied by Algorithm 2.
□

Next, we focus on proving that the distribution maintained by the algorithm is consistent and balanced
(and hence also valid). To prove this, we first formalize and prove the guarantee provided by the
REBALANCESUBSETS procedure. To this end, we define an amount quantifying the degree to which
a class is imbalanced in a given distribution.
Definition B.2. Let 𝜇be a distribution over cache states, let

𝑦𝑝
	

𝑝∈𝑃be the current fractional
solution, and let 𝑗be some class. We define the imbalance of class 𝑗in 𝜇to be
∑︁

𝑆⊆𝑃
𝜇(𝑆) · max
𝑆∩𝑃≥𝑗
 −

𝑌𝑗

,

𝑌𝑗

−
𝑆∩𝑃≥𝑗
	
;

Definition B.2 quantifies the degree to which a given class is not balanced in a given distribution; one
can see that if the class is balanced, this amount would be 0.
Lemma B.3. Suppose REBALANCESUBSETS is called, and let 𝜇, 𝜇′ be the distributions before and
after the call to REBALANCESUBSETS, respectively; let 𝑗max be the maximum non-balanced class in
𝜇.

Suppose that (a) 𝜇is consistent, and (b) there exists 𝜖> 0 such that the imbalance of any class
𝑗≤𝑗max in 𝜇is at most 𝜖. Then, it holds that 𝜇′ is both consistent and balanced. Moreover, the total
cost of REBALANCESUBSETS is at most 12𝜖· 6𝑗max.

15


---Page Break---
Proof. The running of REBALANCESUBSETS consists of iterations of the For loop in Line 5; we
number these iterations according to the class considered in the iteration (e.g., "Iteration 𝑖" considers
class 𝑖). We make a claim about the state of the anti-cache distribution after each iteration, and prove
this claim inductively; applying this claim to the final iteration implies the lemma. Specifically, where
𝑗max as defined in the lemma statement, for every 𝑖≤𝑗max consider the distribution immediately
before Iteration 𝑖, denoted 𝜇𝑖. We claim that (a) the 𝜇𝑖is consistent, (b) all classes greater than 𝑖are
balanced in 𝜇𝑖, and (c) classes at most 𝑖have imbalance at most 𝜖· 3𝑗max−𝑖in 𝜇𝑖. Where 𝑗min is the
minimum class, note that 𝜇′ = 𝜇𝑗min−1, and that this claim implies that 𝜇′ is consistent and balanced.
(Note that the claim implies that class 𝑗min is balanced, which also implies that all classes smaller
than 𝑗min are balanced.).

We prove this claim by descending induction on 𝑖. The base case, in which 𝑖= 𝑗max, is simply a
restatement of the assumptions made in the lemma, and thus holds. Now, assume that the claim holds
for any class 𝑖≤𝑗max; we now prove it for class 𝑖−1.

Consistency. First, as we’ve assumed that 𝜇𝑖is consistent, note that Iteration 𝑖does not change the
marginal probability of a given page 𝑝being in the anti-cache, as pages are only moved between
identical anti-cache measures. Thus, the anti-cache distribution remains consistent at any step during
Iteration 𝑖; in particular, 𝜇𝑖−1 is consistent.

Existence of destination measure. Next, observe that every changes in Iteration 𝑖consists of
identifying a measure of a violating anti-cache, and matching this measure to an identical measure of
anti-cache states to which pages can be moved. To show that the procedure is legal, we claim that
this measure always exists. Consider such a change that identifies violating anti-cache 𝑆, and let ˆ𝜇
be the distribution at that point. Assume that 𝑆is an “upwards” violation, i.e., 𝑚≥⌈𝑌𝑖⌉+ 1, where
𝑚:= |𝑆∩𝑃≥𝑖|; the case of a “downwards” violation is analogous. Note that consistency implies that
E𝑆′∼ˆ𝜇|𝑆′ ∩𝑃≥𝑖| = 𝑌𝑖. Also note that 𝑆was chosen to maximize |𝑚−𝑌𝑖|, i.e., the distance from the
expectation. Thus, there exists a measure of at least ˆ𝜇(𝑆) of anti-caches 𝑆′ such that |𝑆′ ∩𝑃≥𝑖| < 𝑌𝑖
(and thus |𝑆′ ∩𝑃≥𝑖| ≤⌈𝑌𝑖⌉−1, as required).

Existence of page to move. After matching the aforementioned 𝑆to some 𝑆′, we want to identify
some page 𝑝∈𝑃𝑖such that 𝑝∈𝑆\𝑆′, so we can move it from the measure of 𝑆to the measure of 𝑆′.
Indeed, from the choice of 𝑆and 𝑆′, it holds that |𝑆∩𝑃≥𝑖| ≥⌈𝑌𝑖⌉+ 1 ≥|𝑆′ ∩𝑃≥𝑖| + 2. But, from the
induction hypothesis for Iteration 𝑖, class 𝑖+ 1 was balanced in 𝜇𝑖, and thus remains balanced at every
step during Iteration 𝑖(as this iteration never moves pages of classes 𝑖+ 1 and above). This implies
that |𝑆∩𝑃≥𝑖+1| ≤⌈𝑌𝑖+1⌉≤|𝑆′ ∩𝑃≥𝑖+1| + 1. We can thus conclude that there exists 𝑝∈𝑃𝑖∩(𝑆\ 𝑆′)
as required.

Balanced property. Next, we prove that in 𝜇𝑖−1 after Iteration 𝑖, class 𝑖is balanced, and the imbalance
of any class 𝑗< 𝑖is at most 𝜖· 3𝑗max−(𝑖−1). Consider any step in Iteration 𝑖, where a measure 𝑥of
a violating state is identified; then, a page is moved from a measure 𝑥to another measure 𝑥. The
induction hypothesis for Iteration 𝑖implies that the imbalance of class 𝑖at 𝜇𝑖is at most 𝜖· 3𝑗max−𝑖.
Note that:

1. This step decreases the imbalance of class 𝑖by at least 𝑥, as it decreases the imbalance in the
violating state, but does not increase imbalance in the matched measure.

2. This step can increase the imbalance of a class 𝑗< 𝑖by at most 2𝑥, in the worst case in
which moving the page increased imbalance in both measures of 𝑥.

As a result, we can conclude that for 𝜇𝑖−1, at the end of iteration 𝑖, class 𝑖is balanced, while the
imbalance of every class 𝑗< 𝑖increased by at most 2 · 𝜖· 3𝑗max−𝑖. Combining this with the hypothesis
for iteration 𝑖, the imbalance of every class 𝑗at 𝜇𝑖−1 is at most 𝜖· 3 · 3𝑗max−𝑖= 𝜖· 3𝑗max−(𝑖−1), as
required.

This concludes the inductive proof of the claim.

Cost analysis. As mentioned before, every step in Iteration 𝑖reduces imbalance at class 𝑖by (at
least) 𝑥, where 𝑥is the measure of the chosen violating anti-cache state. The cost of this step is the
cost of evicting a single page in 𝑃𝑖from a measure 𝑥; as we assume that the weight of a page is at
most its UCB, this cost is at most 𝑥· 6𝑖+1. The inductive claim above states that the imbalance of
class 𝑖at the beginning of Iteration 𝑖is at most 𝜖· 3 𝑗max−𝑖; thus, the total cost of Iteration 𝑖is at most
𝜖·3𝑗max−𝑖·6𝑖+1 = 𝜖·6𝑗max+1/2𝑗max−𝑖. Summing over iterations, the total cost of REBALANCESUBSETS

16


---Page Break---
is at most:
𝑗max
∑︁

𝑖=𝑗min
𝜖· 6𝑗max+1/2𝑗max−𝑖≤12𝜖· 6𝑗max.

□

We can now prove that the distribution is consistent and balanced.

Lemma B.4. The distribution maintained by Algorithm 2 is both consistent and balanced.

Proof. First, we prove that the distribution is consistent. Indeed, note that consistency is ex-
plicitly maintained in Lines 9 and 12, and that Lemma B.3 implies that the subsequent calls to
REBALANCESUBSETS does not affect this consistency.

As the distribution is consistent at any point in time, Lemma B.3 also implies that it is balanced
immediately after every all to REBALANCESUBSETS; as the handling of every request ends with
such a call, the distribution is always balanced after every request.
□

At this point, we’ve shown that Algorithm 2 is legal: it maintains a valid distribution (through
Lemma B.4 and Remark 5.3), and it provides samples to ONF when required (Proposition B.1);
thus, it is a valid randomized algorithm for OWP-UW. It remains to bound the expected cost of
Algorithm 2, thus proving Lemma 3.2; recall that this bound is in terms of ONF, the cost of the
fractional algorithm in terms of its UCBs rather than actual page weights.

Proof of Lemma 3.2. We consider the eviction costs incurred by Algorithm 2, and bound their costs
individually.

First, consider the eviction cost due to maintaining consistency Line 9 (note that Line 12 only fetches
pages, and incurs no cost). An increase of 𝜖in 𝑦𝑝′ causes an eviction of 𝑝′ with 𝜖probability; the
expected cost of 𝜖· 𝑤𝑝′ can be charged to the eviction cost of 𝜖· UCB𝑝′ incurred in ONF, and thus
the overall cost due to this line is at most ONF.

Next, consider the cost due to eviction during sampling (Line 17). Observe a page 𝑝∈𝑃that is
evicted in this way; the cost of this eviction is 𝑤𝑝. For the first and second samples of 𝑝, we note
that 𝑤𝑝≤1; summing over 𝑝∈𝑃, the overall cost of those evictions is at most 2𝑛. For subsequent
samples of 𝑝, note that for 𝑖> 2, the 𝑖’th sample of 𝑝is taken when 𝑚𝑝∈(𝑖−2, 𝑖−1]. Thus, we can
charge this sample to the fractional eviction that increased 𝑚𝑝from 𝑖−3 to 𝑖−2, which costs UCB𝑝.
Thus, the overall cost of this sampling is at most ONF + 2𝑛.

It remains to bound the cost of the REBALANCESUBSETS procedure. First, consider the cost of
REBALANCESUBSETS due to sampling (Lines 16 and 20). Consider the state prior to such a call;
some page 𝑝has just been sampled, possibly decreasing UCB𝑝and decreasing the class of page 𝑝,
which could break the balanced property. Specifically, let 𝑖, 𝑖′ be the old and new classes of 𝑝, where
𝑖′ < 𝑖. Then, imbalance could be created only in classes 𝑗∈{𝑖′ + 1, · · · , 𝑖}. In such class 𝑗, both 𝑌𝑗
could decrease, and
𝑆∩𝑃≥𝑗
 could decrease for any anti-cache state 𝑆. However, as only one page
changed class, one can note that the total imbalance in any such class 𝑗is at most 1. Thus, Lemma B.3
guarantees that the total cost of REBALANCESUBSETS is at most 12 · 6𝑖≤12 · UCB𝑝. Using the
same argument as for the cost of sampling, the total cost of such calls is at most 12ONF + 24𝑛.

Now, consider a call to REBALANCESUBSETS in Line 10; A page 𝑝′ was evicted for fraction 𝜖in
ONF, and an 𝜖measure of 𝑝′ was evicted in the distribution. Let 𝑗be the class of 𝑝′; there could
only be imbalance in classes at most 𝑗. For any such 𝑖≤𝑗, 𝑌𝑖increased by 𝜖, and |𝑆∩𝑃≥𝑖| increased
by 1 in at most 𝜖measure of states 𝑆. In addition, let 𝑌−
𝑖,𝑌+
𝑖:= 𝑌−
𝑖be the old and new values of 𝑌𝑖.
Consider the imbalance in class 𝑖:

1. If

𝑌+
𝑖

=

𝑌−
𝑖

+ 1, then the imbalance of class 𝑖can increase due to states 𝑆where
|𝑆∩𝑃≥𝑖| =

𝑌−
𝑖

becoming unbalanced. But, due to consistency, the fact that 𝑌−
𝑖≥

𝑌−
𝑖

−𝜖
implies that the measure of such pages is at most 𝜖; thus, the imbalance grows by at most 𝜖.

2. The adding of page 𝑝′ to an 𝜖-measure of pages can add an imbalance of at most 𝜖.

17


---Page Break---
Overall, the imbalance in classes at most 𝑗prior to calling REBALANCESUBSETS is at most 2𝜖.
Through Lemma B.3, the cost of REBALANCESUBSETS is thus at most 24𝜖· 6𝑗, which is at most
24𝜖· UCB𝑝′. But, the increase in ONF due to the fractional eviction is at least 𝜖· UCB𝑝′; thus, the
overall cost of REBALANCESUBSETS called in Line 10 is at most 24 · ONF.

Finally, consider a call to REBALANCESUBSETS in Line 13, called upon a decrease in 𝑦𝑝′ of 𝜖for
some page 𝑝′. Using an identical argument to the case of eviction, we can bound the cost of this call
by 24 times the “fetching cost” of 𝜖· UCB𝑝′. Now, note that the fractional fetching of a page exceeds
the fractional eviction by at most 1; thus, the total cost of such calls is at most 24ONF + 24𝑛.

Summing all costs, the total expected cost of the algorithm is at most 62ONF + 50𝑛.
□

B.2
Proof of Theorem 1.1

We now combine the ingredients in this paper to prove the main competitiveness bound for Algo-
rithm 2.

Proof of Theorem 1.1. First, assume the good event E. Combining Lemma 3.2, Lemma 3.1 and
Lemma C.2, it holds that
E[ON] ≤𝑂(log 𝑘) · OPT(𝑄) + ˜𝑂(
√

𝑛𝑇)

Next, assume that E does not occur. Upper bound the cost of the algorithm by 𝑂(𝑛) per request, as in
the worst case, the algorithm replaces the entire cache and samples each page in 𝑃once. Thus, an
upper bound for the cost of the algorithm is 𝑂(𝑛𝑇). But, through Lemma C.1, the probability of E
not occurring is at most
1
𝑛𝑇. Thus, we can bound the expected cost of the algorithm as follows.

E[ON] ≤Pr[E] ·

𝑂(log 𝑘) · OPT(𝑄) + ˜𝑂(
√

𝑛𝑇)

+ Pr[¬E] · 𝑂(𝑛𝑇)

≤𝑂(log 𝑘) · OPT(𝑄) + ˜𝑂(
√

𝑛𝑇).
□

C
Choosing Confidence Bounds and Bounding Regret

In this section we define the UCBs and LCBs, prove that they hold with high probability and bound
the regret term.

Let 𝑤𝑖
𝑝be the 𝑖-th sample of page 𝑝. For the initial confidence bounds, we sample each page once
and set LCB𝑝,1 =
1
2𝑛2𝑇· 𝑤1
𝑝, UCB𝑝,1 = 1. Once we have 𝑖> 1 samples of page 𝑝, we define
the confidence bounds as follows. Let ¯𝑤𝑝,𝑖:= 1

𝑖
Í𝑖
𝑗=1 𝑤𝑗
𝑝be the average observed weight, and

𝜖𝑝,𝑖=
√︃

log(4𝑛3𝑇3)

2𝑖
be the confidence radius. Then, we set LCB𝑝,𝑖= max{LCB𝑝,𝑖−1, ¯𝑤𝑝,𝑖−𝜖𝑝,𝑖}
and UCB𝑝,𝑖= min{UCB𝑝,𝑖−1, ¯𝑤𝑝,𝑖+ 𝜖𝑝,𝑖}. The following procedure updates the confidence bounds
online.

We show that the confidence bounds indeed bound the true weights with high probability (Lemma C.1),
and then bound the regret term (Lemma C.2).
Lemma C.1. Let 𝑛𝑝be the final number of samples collected for page 𝑝. With probability at least
1 −
1
𝑛𝑇, the following properties hold.

1. 0 < LCB𝑝,1 ≤LCB𝑝,2 ≤. . . ≤LCB𝑝,𝑛𝑝≤𝑤𝑝for every page 𝑝.

2. 𝑤𝑝≤UCB𝑝,𝑛𝑝≤UCB𝑝,𝑛𝑝−1 ≤. . . ≤UCB𝑝,1 = 1 for every page 𝑝.

3. UCB𝑝,𝑖−LCB𝑝,𝑖≤2𝜖𝑝,𝑖for every page 𝑝and 𝑖∈[1, 𝑛𝑝].

Proof. By definition, the LCBs are monotonically non-decreasing and the UCBs are monotonically
non-increasing. Moreover,
UCB𝑝,𝑖−LCB𝑝,𝑖= min

UCB𝑝,𝑖−1, ¯𝑤𝑝,𝑖+ 𝜖𝑝,𝑖
	
−max

LCB𝑝,𝑖−1, ¯𝑤𝑝,𝑖−𝜖𝑝,𝑖
	

≤( ¯𝑤𝑝,𝑖+ 𝜖𝑝,𝑖) −( ¯𝑤𝑝,𝑖−𝜖𝑝,𝑖) = 2𝜖𝑝,𝑖.

18


---Page Break---
Algorithm 4: Optimistic-Pessimistic Weights Estimation Procedure

1 Function UPDATECONFBOUNDS(𝑝, e𝑤𝑝) // update confidence bounds for 𝑝upon new sample ˜𝑤𝑝.

2
if this is the first sample of 𝑝(i.e., 𝑛𝑝= 1) then

3
define ¯𝑤𝑝←e𝑤𝑝.

4
define LCB𝑝←
1
2𝑛2𝑇· ¯𝑤𝑝.

5
define UCB𝑝←1.

6
else

7
update mean estimation ¯𝑤𝑝←(𝑛𝑝−1) ¯𝑤𝑝+e𝑤𝑝

𝑛𝑝
.

8
define 𝜖𝑝←

√︂

log(4𝑛3𝑇3)

2𝑛𝑝
.

9
set LCB𝑝←max

LCB𝑝, ¯𝑤𝑝−𝜖𝑝
	
.

10
set UCB𝑝←min

UCB𝑝, ¯𝑤𝑝+ 𝜖𝑝
	
.

Thus, it remains to show that
1
2𝑛2𝑇· 𝑤1
𝑝≤𝑤𝑝and that |𝑤𝑝−¯𝑤𝑝,𝑖| ≤𝜖𝑝,𝑖for every page 𝑝and
𝑖∈[1,𝑇] with probability 1 −
1
𝑛𝑇. For the first event, by Markov inequality and a union bound over
all the pages 𝑝∈𝑃, we have

P

∃𝑝∈𝑃.
1
2𝑛2𝑇· 𝑤1
𝑝> 𝑤𝑝


= P[∃𝑝∈𝑃. 𝑤1
𝑝> 2𝑛2𝑇· 𝑤𝑝] ≤𝑛·
1
2𝑛2𝑇=
1
2𝑛𝑇.

For the second event, by Hoeffding inequality and a union bound over all the pages 𝑝∈𝑃, all the
time steps 𝑡∈[1,𝑇] and all the possible number of samples for each page 𝑖∈[1, 𝑛𝑇], we have

P

∃𝑝∈𝑃, 𝑖∈[1,𝑇]. |𝑤𝑝−¯𝑤𝑝,𝑖| > 𝜖𝑝,𝑖

≤𝑛2𝑇2 · 2𝑒−2𝑖𝜖2
𝑝,𝑖= 𝑛2𝑇2 ·
1
2𝑛3𝑇3 =
1
2𝑛𝑇.

The proof is now finished by taking a union bound over these two events.
□

Define U := Í
𝑝∈𝑃
Í𝑛𝑝
𝑖=1
 UCB𝑝,𝑖−LCB𝑝,𝑖
 + 2 log(1 + 1/𝜂) Í
𝑝∈𝑃LCB𝑝, the regret term used in
Lemma 4.1.
Lemma C.2. Under the good event of Lemma C.1, it holds that

U ≤8
√

𝑛𝑇log(𝑛𝑇) = ˜𝑂(
√

𝑛𝑇).

Proof. The following holds under the good event.

U =
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1

 UCB𝑝,𝑖−LCB𝑝,𝑖
 + 2 log(1 + 1/𝜂)
∑︁

𝑝∈𝑃
LCB𝑝

≤2
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1
𝜖𝑝,𝑖+ 2 log(1 + 1/𝜂)
∑︁

𝑝∈𝑃
𝑤𝑝
(Lemma C.1)

= 2
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1

√︂

log(4𝑛3𝑇3)

2𝑖
+ 2 log(1 + 1/𝜂)
∑︁

𝑝∈𝑃
𝑤𝑝

≤
√︃

2 log(4𝑛3𝑇3)
∑︁

𝑝∈𝑃

𝑛𝑝
∑︁

𝑖=1

1√

𝑖
+ 2𝑛log(1 + 1/𝜂)
(𝑤𝑝≤1)

≤2
√︃

2 log(4𝑛3𝑇3)
∑︁

𝑝∈𝑃

√𝑛𝑝+ 2𝑛log(1 + 1/𝜂)
(Í𝑡
𝑖=1
1√

𝑖≤2√𝑡)

≤2
√︃

2𝑛𝑇log(4𝑛3𝑇3) + 2𝑛log(1 + 𝑘),

where the last inequality holds by Cauchy–Schwarz and our choice of 𝜂= 1/𝑘. Lastly, the stated
upper bound follows since 𝑘≤𝑛and 𝑛≤
√

𝑛𝑇.
□

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We provide an algorithm for the problem of online weighted paging with
unknown weights, and analyse it. See Section 1.1 for a summary of our results, that will be
proven throughout the paper.

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

Answer: [NA]

Justification: This is a theory paper that present an algorithm for the online weighted paging
where the weight are unknown and prove the stated bounds. We do not believe there are
additional limitations to discuss.

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

20


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: All assumptions are stated in the statements of the theorem and related lemmas.
A proof is provided for each theoretical claim.
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
Justification: No additional information is required to obtain the results in this paper, which
are theoretical. A full pseudocode for the algorithm appears in the paper and appendix.
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

21


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justification: The paper does not contain experiments at all.

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

Justification: The paper does not contain experiments at all.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [NA]

Justification: There are no experiments in the paper.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

22


---Page Break---
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

Answer: [NA]

Justification: There are no experiments in the paper.

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

Justification: We reviewed the code of ethics and concluded the paper conforms to it.

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

Justification: This paper describes an algorithm for the online weighted paging problem,
where the weights are unknown, and analyse it. The paper is theoretical. Thus, we do not
predict any societal impact due to this work.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

23


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
Justification: This paper is theoretical. No data or models are associated with it.
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
Answer: [NA]
Justification: This paper is theoretical. No existing assets are associated with it.
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

24


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper is theoretical. No new assets are released in it.
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
Justification: This paper is theoretical. It does not involve crowdsourcing nor research with
human subjects.
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
Justification: This paper is theoretical. It does not involve crowdsourcing nor research with
human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.

25


---Page Break---
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

26


---Page Break---
