Improved Analysis for Bandit Learning in Matching
Markets

Fang Kong
Southern University of Science and Technology
kongf@sustech.edu.cn

Zilong Wang
Shanghai Jiao Tong University
wangzilong@sjtu.edu.cn

Shuai Li∗
Shanghai Jiao Tong University
shuaili8@sjtu.edu.cn

Abstract

A rich line of works study the bandit learning problem in two-sided matching
markets, where one side of market participants (players) are uncertain about their
preferences and hope to find a stable matching during iterative matchings with
the other side (arms). The state-of-the-art analysis shows that the player-optimal
stable regret is of order O(K log T/∆2) where K is the number of arms, T is the
horizon and ∆is the players’ minimum preference gap. However, this result may
be far from the lower bound Ω(max{N log T/∆2, K log T/∆}) since the number
K of arms (workers, publisher slots) may be much larger than that N of players
(employers in labor markets, advertisers in online advertising, respectively). In
this paper, we propose a new algorithm and show that the regret can be upper
bounded by O(N 2 log T/∆2 + K log T/∆). This result removes the dependence
on K in the main order term and improves the state-of-the-art guarantee in com-
mon cases where N is much smaller than K. Such an advantage is also verified
in experiments. In addition, we provide a refined analysis for the existing central-
ized UCB algorithm and show that, under α-condition, it achieves an improved
O(N log T/∆2 + K log T/∆) regret.

1
Introduction

The two-sided matching market problem has been extensively studied in the literature due to its wide
range of applications like labor market, school admission, house allocation, and online advertising
[25, 9, 1]. There are two sides of participants in the market, such as the employers and workers in the
labor market, advertisers and publishers in online advertising. Each participant on the one side has a
preference ranking over the other side. The concept of stability, which characterizes the equilibrium
state of the market where no participant wants to break up the current matching relationship and find
another partner, has attracted great interest from researchers [25]. Achieving stability is critical for
ensuring the long-term viability of the market.

A rich line of works [9, 15, 24] study how to find a stable matching in the market. Most of them
assume the preference ranking of each market participant is known beforehand, which we refer
to as the offline setting. However, in real applications, the knowledge of the preferences may be
uncertain. For example, in the labor market, employers usually do not know the working abilities
of workers before being matched, and advertisers also do not know the exact conversion rate of
placing the advertisement in a publisher slot. This makes the traditional algorithms unavailable to

∗Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
find an exact stable matching. With the emergence of online market platforms such as the online
labor market UpWork and TaskRabbit as well as online advertising platforms where employers or
advertisers have many similar tasks, market participants are able to learn their unknown preferences
during iterative matchings with the other side of agents.

Multi-armed bandit (MAB) is a classic framework that characterizes the learning process during it-
erative interactions [3, 18]. It considers the setting with single player on one side and multiple arms
on the other side. At each round, the player selects an arm and receives a reward. The player has
unknown preferences over arms and would learn this knowledge based on the collected rewards. To
accumulate as many rewards as possible, the player faces the dilemma of exploration and exploita-
tion. The former selects arms with less observations while the latter focuses on arms with better
historical performances. How to balance the exploration and exploitation trade-off is the key of
bandit algorithm design. The upper confidence bound (UCB) [3], Thompson sampling (TS) [14, 2],
and explore-then-commit (ETC) [10] are common strategies in MAB to achieve this objective.

Liu et al. [19] introduce the bandit learning problem in matching markets and try to provide theo-
retical guarantees. Two sides of agents in the market can be modeled as players and arms. Without
loss of generality, denote N and K as the number of players and arms, respectively. It is worth
noting that this work and all of the following works assume N ≤K to ensure each player has a
chance to be matched. In this problem, the objective is to find a stable matching and minimize the
stable regret for each player, which is defined as the difference between the reward of the stable arm
and that the player receives during the horizon. Since there may be more than one stable matching,
they mainly focus on the players’ most preferred one corresponding to the player-optimal stable
matching and the least preferred one corresponding to the player-pessimal stable matching. Note
that players receive more rewards in the player-optimal stable matching and thus the former objec-
tive is the most desirable. Liu et al. [19] first study a centralized setting where a central platform
would compute allocations for players to avoid conflicts. Both ETC and UCB-type algorithms are
proposed for this setting. The former achieves a player-optimal stable regret guarantee with prior
knowledge of players’ minimum preference gap ∆and the latter can only ensure to reach the player-
pessimal stable matching. Motivated by real applications where the central platform may not always
exist, a rich line of works then study the decentralized case where no platform coordinates players’
behavior [20, 27, 4, 17, 21]. This line of works again only achieve guarantees for player-pessimal
stable regret [20, 17, 27, 4, 21]. Table 1 compares settings and regrets among these works. Until
recently, Zhang et al. [30] and Kong and Li [16] independently derive algorithms that have polyno-
mial player-optimal stable regret and show the upper bound is O(K log T/∆2). However, this result
may be still far from the lower bound Ω(max{N log T/∆2, K log T/∆}) [27] since K is usually
much larger than N such as that the number of workers (publisher slots) is usually much larger than
that of employers in labor markets (advertisers in online advertising, respectively).

In this paper, we try to provide more efficient algorithms and improve the results over existing works.
The detailed contribution can be summarized as follows:

• We propose an adaptive online Gale-Shapley (AOGS) algorithm for general markets. State-
of-the-art works [30, 16] explicitly separate the exploration and exploitation processes,
which can lead to unnecessary regret, as exploring certain preference rankings may not
contribute to the exploitation process. To avoid excessive exploration, our AOGS algo-
rithm integrates the players’ learning process into the GS steps. Players explore only the
necessary number of candidate arms and adaptively switch between exploration and ex-
ploitation.
• We prove that the player-optimal stable regret of AOGS can be upper bounded by
O(N 2 log T/∆2 + K log T/∆). This is the first result that removes the dependence on
K in the main regret order term and improves existing works in common cases where
N is much smaller than K. We also conduct experiments to show the advantages of the
algorithm.
• We refine the analysis of the centralized UCB algorithm in Liu et al. [19] for markets
satisfying the α-condition. By investigating the preference hierarchy structure of the α-
condition, we demonstrate that the stable matching converges sequentially from player 1
to player N. Through inductive analysis over players, we establish an O(N log T/∆2 +
K log T/∆) regret upper bound, which improves the original result for this algorithm in
this specific market.

2


---Page Break---
Regret bound
Setting

Liu et al. [19]
O
 
K log T/∆2
∗#
known ∆, gap1
O
 
NK log T/∆2
#
gap2

Liu et al. [20]
O
N 5K2 log2 T

εN 4∆2


gap2

Sankararaman et al. [27]
O
 
NK log T/∆2
serial dictatorship, gap1
Ω
 
max

N log T/∆2, K log T/∆
	

Basu et al. [4]
O

K log1+ε T + 2(
1
∆2 )
1
ε

∗
gap2

O
 
NK log T/∆2
α-condition, gap1

Maheshwari et al. [21]
O
 
CNK log T/∆2
α-reducible condition, gap1

Kong et al. [17]
O
N 5K2 log2 T

εN 4∆2


gap2

Zhang et al. [30]
O
 
K log T/∆2
∗
gap2

Kong and Li [16]
O
 
K log T/∆2
∗
gap3

Ours
O
 
N 2 log T/∆2 + K log T/∆

∗
gap4
O
 
N log T/∆2 + K log T/∆

#
α-condition, gap3

Table 1: Comparisons of settings and regret bounds with most related works, ∗represents the player-
optimal stable regret and bounds without labeling ∗are for player-pessimal stable regret, # repre-
sents the centralized setting. N and K are the number of players and arms with N ≤K, T is the
total horizon, ∆corresponds to some preference gap, ε depends on the hyper-parameter of algo-
rithms, and C is related to the unique stable matching condition which can grow exponentially in
N. The definition of ∆in different works requires particular care. We use gap1, gap2, gap3, gap4
represent the minimum preference gap between the (player-optimal) stable arm and the next arm af-
ter the stable arm in the preference ranking among all players, the minimum preference gap between
any different arms among all players, the minimum preference gap between the first N + 1 ranked
arms among all players, and the minimum preference gap between arms that are more preferred than
the next of the player-optimal stable arm among all players, respectively. Based on the property that
the player-optimal stable arm of each player must be its first N-ranked (shown in Appendix), there
would be gap1 ≥gap4 ≥gap3 ≥gap2. So our dependence on ∆is better than the state-of-the-art
works [30, 16] for general markets.

2
Related Work

The problem of bandit learning in matching markets is first introduced by Das and Kamenica [8].
They study the special case where both sides of agents have the same preferences and propose
some empirical methods to solve the problem. Liu et al. [19] first theoretically formulate this
problem and provide an upper bound for the stable regret of players. They propose a centralized
explore-then-commit (ETC) algorithm and upper confidence bound (UCB) algorithm, which obtain
an O
 
K log T/∆2
player-optimal stable regret and O
 
NK log T/∆2
player-pessimal stable re-
gret, respectively. It is worth noting that the former ETC algorithm requires knowledge about ∆to
ensure the algorithmic operation. Due to the generality, the following works focus on the decentral-
ized setting. Liu et al. [20] and Kong et al. [17] propose the UCB and TS-type algorithm for general
decentralized markets, respectively. Such a setting is much more challenging and both of them only
achieve O
 
exp(N 4)N 5K2 log2(T)/∆2
upper bound for the player-pessimal stable regret.

To improve the stable regret guarantee, a line of research studies some special markets with unique
stable matching in which case the player-optimal stable matching is equivalent to the player-
pessimal one. Sankararaman et al. [27] propose the UCB-D3 algorithm based on the assumption
of serial dictatorship, i.e., all arms share the same preferences, and obtain an O
 
NK log T/∆2

3


---Page Break---
regret upper bound.
To investigate the problem hardness, they also derive a lower bound
Ω
 
max

N log T/∆2, K log T/∆
	
under this assumption. Basu et al. [4] consider more gen-
eral α-condition setting for unique stable matching. They propose the UCB-D4 algorithm and also
achieve the O
 
NK log T/∆2
regret bound. Later, Maheshwari et al. [21] study the market satisfy-
ing α-reducible condition and proposes a communication-free algorithm. Their regret bound has an
exponential dependence on the number of market participants. Recently, researchers have developed
algorithms that can achieve player-optimal stable regret guarantees without assuming unique stable
matching. Both Zhang et al. [30] and Kong and Li [16] propose ETC-type algorithms that achieve
O
 
K log T/∆2
player-optimal stable regret. Wang and Li [28] studies the matching markets with
serial dictatorship and obtains the O
 
N log T/∆2 + K log T/∆

regret. Table 1, compares our
proposed algorithm with these related works in terms of their corresponding settings and theoretical
guarantees.

There are also other works considering unknown preferences in matching markets. Wang et al. [29]
study a many-to-one market where an arm can accept multiple players. Jagadeesan et al. [12] con-
sider online matching markets with monetary transfers. Min et al. [22] investigate Markov matching
markets where state transitions occur during the matching process and players’ rewards depend on
the current state. Other studies have focused on non-stationary rewards, such as Muthirayan et
al. [23], Ghosh et al. [11], who propose robust algorithms to mitigate the impact of reward distur-
bances. Additionally, several studies have explored the problem of offline matching market learning.
Dai and Jordan [6, 7] propose approaches that leverage historical data to design optimal matching
or recommend participants on both sides.

3
Setting

This paper considers the problem of bandit learning in two-sided matching markets. Denote N =
{p1, p2, . . . , pN} as the player set and K = {a1, a2, . . . , aK} as the arm set. Let N and K be the
number of players and arms, respectively. To ensure that each player can be matched with an arm,
we follow previous works and assume N ≤K [19, 20, 27, 4, 17, 30, 16].

For each player pi ∈N, its preference towards arm aj can be portrayed by an absolute utility
µi,j ∈(0, 1]. For any pair of arms aj and aj′, µi,j > µi,j′ indicates that player pi prefers arm aj
over aj′. Following previous works for matching markets [9, 19, 20, 27, 4, 17, 30, 16], players are
assumed to have distinct preferences over different arms, i.e., µi,j ̸= µi,j′ for any aj ̸= aj′. In
practice, players’ preferences which correspond to workers’ abilities and the publisher’s conversion
rates are typically unknown and can be learned through the interactive matching process. On the
other side, each arm aj also has a fixed and distinct preference utility πj,i over each player pi ∈N,
and πj,i > πj,i′ means that arm aj prefers player pi over pi′. As in labor markets where workers
usually know their preferences over employers based on the payments and task types, the preferences
of arms are assumed to be known beforehand [19, 20, 17, 27, 4, 30, 16].

At each round t = 1, 2, . . . , each player pi proposes to an arm Ai(t). For each arm aj, denote
A−1
j (t) = {pi : Ai(t) = aj} as the set of players who selects arm aj at round t. When more
than one player selects aj, it accepts its most-preferred one in A−1
j (t), i.e. aj will match with
pi ∈arg maxpi∈A−1
j
(t) πj,i. If a player pi is successfully matched with arm Ai(t), it will receive a
random reward Xi(t) characterizing its matching experience, which we assume is a 1-subgaussian
random variable with expectation µi,Ai(t). Otherwise, pi is rejected by its proposed arm and only
gets reward Xi(t) = 0. Denote ¯Ai(t) as the final matched arm of player pi at round t. Then
¯Ai(t) = Ai(t) if pi is accepted by the arm Ai(t) and we simply set ¯Ai(t) = ∅if pi is rejected.

Stability is a key property of a matching in two-sided markets [9, 26, 24]. A matching ¯A(t) =
{(i, ¯Ai(t)) : i ∈[N]} is stable if no market participant wants to break up its current matching
relationship and find a new partner. Formally speaking, there is no player-arm pair (pi, aj) such
that µi,j > µi, ¯
Ai(t) and πj,i > πj, ¯
A−1
j
(t). It is worth noting that there may be multiple stable
matchings in the market. Denoted M = {m : m is stable} as the set of all stable matchings. It is
shown that there exists a stable matching m∗∈M such that all players are matched with their most
preferred stable arm [9], i.e., µi,m∗
i ≥µi,mi for any m ∈M, i ∈[N]. Given a specified horizon
T, the learning objective is to minimize the player-optimal stable regret for each player pi which is
defined as the difference between the cumulative reward received by being matched with m∗
i and

4


---Page Break---
the cumulative reward received by pi over T rounds:

Regi(T) = E

" T
X

t=1

 
µi,m∗
i −Xi(t)

#

.

Here, the expectation is taken over by the randomness of the reward generation and the randomness
inherent in the player’s strategy.

For completeness, we introduce the procedure of the offline Gale-Shapley (GS) algorithm, which
would be useful when describing the algorithmic details. The offline GS algorithm is a classic
algorithm to find the player-optimal stable matching when both sides of the market participants
know their exact preference rankings. Following offline GS, each player proposes to the arm one
by one based on its preference ranking. Until no rejection happens, the final matching is exactly
the player-optimal stable matching [9]. Specifically, at the first step, all players propose to their
most preferred arm. Arms would accept their most preferred player among those who propose to it
and reject others. Then players who are rejected at previous steps would then propose to their next
preferred arm. And arms still reject the players who propose to it except for their most preferred
one. Such a process continues until no rejection happens.

For convenience, we also define some useful notations that quantify the hardness of the learning
problem in matching markets and are used in the later analysis.

Definition 3.1. For each player pi, denote σi as pi’s preference ranking and let σi,k as pi’s the
k-th preferred arm in its ranking. With a little abuse of notation, let σi(aj) represent the rank of
arm aj in pi’s preference. For each player pi and arm aj ̸= aj′, let ∆i,j,j′ = |µi,j −µi,j′| be the
preference gap of pi between aj and aj′. Define ∆= mini,k∈[σi(m∗
i )] ∆i,σi,k,σi,k+1 as the minimum
preference gap between the arm ranked the first (σi(m∗
i ) + 1)-th among all players. Further, define
∆N = mini,k∈[N] ∆i,σi,k,σi,k+1 as the minimum preference gap between the arm ranked the first
(N + 1)-th among all players.

4
Adaptive Online Gale-Shapley Algorithm for General Markets

In this section, we propose an algorithm called adaptive online Gale-Shapley (AOGS). For simplic-
ity, we present the centralized version of the algorithm in Algorithm 1 from view of player pi. The
discussion on how to extend it to a decentralized version is deferred to later subsections.

In general, AOGS is an online version of the GS algorithm. Since players do not know their prefer-
ence rankings, they need to learn this knowledge by exploring arms (Line 4). To reduce the regret
during exploration, players would adaptively eliminate sub-optimal arms (Line 6-8). Until they find
their most preferred arm among available arms, they will stop exploration and focus on this arm
(Line 9-11). And once the player finds this arm is occupied by a more preferred player, it would
re-start exploration to find the next preferred arm (Line 12-19).

Specifically, each player pi still maintains ˆµi,j(t) and Ti,j(t) to represent the empirical mean and
the number of observations on each arm aj at the end of round t. To determine whether an arm is
more preferred than another, it maintains a confidence interval for each arm aj with upper bound
UCBi,j(t) := ˆµi,j(t) +
p

6 log T/Ti,j(t) and lower bound LCBi,j(t) := ˆµi,j −
p

6 log T/Ti,j(t).
When Ti,j = 0, they will be initialized as +∞and −∞, respectively. And once the confidence
intervals of the two arms are disjoint, it can regard the arm with a higher empirical mean to be more
preferred (Line 1). To be consistent with the offline GS, each player maintains Di to represent the
set of arms that have rejected pi during previous steps. In the beginning, it is initialized as an empty
set. And we use Ai to represent the available arms with the potential to be the stable arm of pi,
which is initialized as K \ Di. For convenience, denote Ei as the exploration status of player pi.
Ei = True means that pi still needs to explore arms in Ai to determine its most preferred arm. And
Ei = False means that pi already finds its most preferred arm and now focuses on this arm (Line 2).

To reduce the regret suffered during exploration, players would update Ai and eliminate sub-optimal
arms in real-time (Line 6-8). Here to avoid collision during round-robin exploration, we would
maintain Ai such that it contains no less than N arms if pi still has not determined its most preferred
one. Thus the union of the available arm set over all players with Ei = True contains more than
N arms and the round-robin exploration over Ai for each such player pi can be carried out without

5


---Page Break---
Algorithm 1 adaptive online Gale-Shapley (from the view of player pi)
Input: N, K, T.

1: Initialize: ˆµi,j(0) = 0, Ti,j(0) = 0, UCBi,j(0) = ∞, LCBi,j(0) = −∞, ∀j ∈[K];
2: Initialize: Di = ∅, Ai = K, Ei = True;
3: for round t = 1, 2, · · · , T do
4:
Select Ai(t) ∈Ai in a round-robin manner;
5:
Update ˆµi,j(t), Ti,j(t) as Algorithm 2; compute UCBi,j(t), LCBi,j(t) for any j ∈[K];
6:
if |Ai| > N and ∃j ∈Ai s.t. UCBi,j(t) < maxj′∈Ai LCBi,j′(t) then
7:
Ai = Ai\{j};
8:
end if
9:
if ∃j ∈Ai s.t. LCBi,j(t) > UCBi,j′(t) for any j′ ∈Ai and j′ ̸= j then
10:
Ai = {j}, Ei = False, Ai = j;
11:
end if
12:
for other player pi′ with Ei′ = False, Ai′ ∈Ai do
13:
if πAi′,i′ > πAi′,i then
14:
Di = Di ∪{Ai′}, Ai = K\Di;
15:
if Ei = False and Ai ∈Di then
16:
Ei = True;
17:
end if
18:
end if
19:
end for
20: end for

collisions. For completeness, we defer how to arrange players’ explorations in later discussions.
And once there exists an arm in Ai that can be regarded to be optimal, pi will set the exploration
status Ei to be False and update the exploration arm set Ai to only contain this optimal arm. For
convenience, denote Ai as this arm (Line 9-11).

The update of the available arm set should not only depend on pi’s own observations but also on
the other market participants. Specifically, if a player pi′ determines Ai′ as its most preferred arm,
then the final stable player of Ai′ would be the same as or more preferred than pi′. So if Ai′ prefers
pi′ than pi, then Ai′ would not be the stable arm of pi and there is no need for pi to explore Ai′
anymore. In this case, pi deletes arm Ai′ from its available set and update Ai (Line 12-19). It is
worth noting that this operation may incorporate the eliminated arms again in Ai. This is reasonable
as the previously eliminated arm may be more preferred than the current arms in Ai after the deletion
operation. And if the deleted arm is pi’s current most preferred arm, it will mark Ei as True and
restart exploration to find the next most preferred one (Line 15-17).

4.1
Theoretical Results.

Theorem 4.1. Following Algorithm 1, the player-optimal stable regret for each player pi satisfies

Regi(T) ≤O
 
N 2 log T/∆2 + K log T/∆

.

Due to the space limit, the proof of Theorem 4.1 is deferred to Appendix A. The following are
discussions on the detailed implementation as well as the novelty and significance of the result.

Arrangement of the round-robin exploration process.
Recall that the number of available
arms of each player is always larger than N based on Line 6 and we assume players can explore
their available arms in a round-robin manner without conflict. We now propose an arrangement by
letting players explore the available arms in units of N to guarantee this property. Specifically, in
every 2N rounds, each player selects the N available arms with the fewest observations (randomly
breaks ties) and explores them in a round-robin way. It can be shown by contradiction that there
exists an assignment such that each player can successfully match with their respective N arms
once during these 2N rounds (Lemma B.2 in Appendix), which only doubles the original regret
without influencing the regret order. This guarantees that after every 2N rounds, the observation
count difference among all available arms is at most 1. Players would perform arm elimination and
optimal arm identification (Line 6 and 9) in the end of each 2N rounds. Therefore, compared to
the timely eliminating/deleting of arms, this approach ensures that each player will select each arm

6


---Page Break---
at most one additional time during each exploration cycle before the player finds the optimal one.
Since each player may restart exploration (Line 12-13) up to N 2 times (each of N players can focus
on N arms), this scenario leads to an additional O(N 2K) constant regret and does not influence the
final regret order.

Extension to the decentralized setting.
For simplicity, we present the AOGS algorithm in a
centralized manner. It can also be extended to the decentralized version where no central platform
coordinates players’ selections. Specifically, we divide the total horizon into several phases and
the length of each grows exponentially, i.e., the lengths of phases are 2, 4, 8, · · · . At the end round
of each phase, players would decide whether to eliminate sub-optimal arms as Line 6-8, whether
to determine one arm as the most preferred one and update the exploration status as Line 9-11.
After the end of each phase, players would communicate their current exploration status, update
their deletion set and available arm set, re-update the exploration status as Line 12-19, and then
communicate their updated available arm set to determine the round-robin exploration process in
the next phase. If there exists a player whose exploration status becomes True from False during
communication, the phase length would re-start from 2 and grows exponentially since new arms may
be required for exploration. If the final exploration status of all players is False and their optimal
arms are different, the next phase would continues until the end of the interaction. The detailed
implementation of communication is deferred to the next paragraph. Based on the communication,
players with Ei = True can have a pre-agreed protocol to explore arms in their available arm set
in a round-robin manner without collision as discussed in the last paragraph. Within each phase,
they just round-robin explore arms and collect observations but do not make any decision on arms’
optimality. If L observations on a sub-optimal arm are enough to decide its sub-optimality in the
centralized version, then this arm would be eliminated at the end of the corresponding phase with
the selected time to be at most 2L due to the exponentially increasing phase length. So the regret in
this decentralized version is at most two times as that suffered in the centralized version.

This paragraph describes the implementation of the communication procedure. Recall that players
need to communicate their exploration status and available arm set (calculated by subtracting the
deletion and eliminating set from K) at the end of each phase. For the phase length, recall that it
grows exponentially until a player’s exploration status becomes True from False and a player up-
dates Ei from False as True only when its most preferred arm is occupied by a higher-priority player
(Line 15). As shown by Lemma A.3, each player may occupy N arms, so such event happens at most
N 2 times. And when all players find their unique optimal arm which requires O(N 2 log T/∆2)
times, the phase would continue until the end of the interaction. Above all, the total number of
phases is of order O
 
N 2 log
 
N 2 log T/∆2
. For the detailed communication procedure, as phase
1 in Kong and Li [16], players can first estimate their unique indices and we assume the matching
results are public as [16, 17, 20]. During the communication block of each phase, players sequen-
tially transmit their data based on their indices, received by others through matching outcomes.
Specifically, in the corresponding round, player pi selects the focused arm if Ei = False and noth-
ing otherwise, incurring an O
 
N 3 log
 
N 2 log T/∆2
cost for status communication in all phases.
For deletion (eliminating) sets, pi first selects the arm with index k to indicate it will transmit k
arms and then sequentially selects these k arms. The communication cost on the arm set size is
O
 
N 3 log
 
N 2 log T/∆2
. Recall that players delete arms only when a higher-priority player fo-
cuses on this arm, so N players focus on at most N arms before reaching stability and each player
deletes up to N arms. Also, each player can eliminate up to K−N arms during each exploration and
would re-start exploration for at most N 2 times. Thus the communication cost on the deletion (elim-
inating) arms is O(N 3K) and the total communication cost is O
 
N 3 log
 
N 2 log T/∆2
+ N 3K

,
which is not the main order of the regret.

Novelty and significance.
Balancing the exploration-exploitation trade-off is the key to achieving
low regret. Previous efforts were devoted to addressing pessimal stable regret [19, 20, 17] and
uniqueness assumptions [27, 4] using classic UCB and TS strategies. Until recently, Zhang et al.
[30] and Kong and Li [16] show that ETC-type strategies better fit this problem. Specifically, players
first uniformly explore arms to learn the complete preference ranking of the top N arms, and then
use the GS procedure for exploitation to find the player-optimal stable matching. However, such a
method may over-explore and cause unnecessary regret. The reason is that to learn the first N-ranked
arms, each sub-optimal arm aj must be selected O(log T/∆2
i,σi,N,j) times to be distinguished from

7


---Page Break---
the N-ranked arm. And each time selecting this arm, the player pays ∆i,m∗
i ,j regret. The mismatch
between the paid regret and the difference to be figured out results in O(K log T/∆2
N) regret.

In contrast to the existing approach, we present a more adaptive perspective that integrates the learn-
ing process into each GS step. To avoid additional regret, players do not need to estimate their
complete preference. Instead, they would start exploitation once the optimal available arm is iden-
tified. And if this arm is occupied by a higher-priority player, this player would restart exploration
to find the next preferred one. To avoid the additional cost while exploring the optimal arm, we also
design a more efficient way to let players promptly eliminate K −N sub-optimal arms and only
maintain the remaining N arms to guarantee no collision. For the eliminated arm aj, the reward
difference to be figured out is at least ∆i,m∗
i ,j, which matches the regret when selecting this arm. So
the regret caused by these eliminated arms is O(K log T/∆), avoiding the dependence on K in the
main order term. This is a more effective way to balance exploration and exploitation in matching
market scenarios. Our result also shows a significant improvement over Zhang et al. [30] and Kong
and Li [16] in common applications such as labor markets and online advertising where the number
N of players (employers, advertisers) is far smaller than that K of arms (workers, publisher slots).

Discussion on the definition of gaps.
Recall that our ∆is defined as the minimum preference
difference among arms that ranked in the first (σi(m∗
i ) + 1)-th positions (gap4), while the lower
bound in Sankararaman et al. [27] for markets with serial dictatorship depends on ∆that is defined as
the minimum preference difference between the arm ranked σi(m∗
i ) and the arm ranked σi(m∗
i ) + 1
(gap1 in Table 1, respectively). It is an open problem whether the lower bound should depend on
our gap4 in general markets. Here we would like to discuss that the knowledge of gap4 is necessary
to learn the true stable matching. Consider a market with 4 players and 5 arms. The preference
rankings of players are p1 : a1 > a2 > a3 > a4 > a5; p2 : a2 > a3 > a1 > a4 > a5; p3 :
a3 > a1 > a2 > a4 > a5; p4 : a1 > a2 > a3 > a4 > a5 and the preference rankings of arms
are a1 : p2 > p3 > p4 > p1; a2 : p3 > p4 > p1 > p2; a3 : p4 > p1 > p2 > p3; a4 : p1 >
p2 > p3 > p4; a5 : p1 > p2 > p3 > p4. In this market, the player-optimal stable matching
is {(p1, a4), (p2, a1), (p3, a2), (p4, a3)}. However, if player p1 has collected enough observations
to identify gap1 but not collected enough observations to identify gap4 and wrongly estimate the
first σi(m∗
i ) ranked arms, i.e., p1 wrongly estimate the preference ranking as p1 : a1 > a2 >
a4 > a3 > a5. Then the computed player-optimal stable matching under this preference ranking
is {(p1, a4), (p2, a3), (p3, a1), (p4, a2)}, which is not stable in the original market as player p1 and
arm a3 form a blocking pair. This example shows that player p1 must identify the gap among the
first σi(m∗
i ) ranked arms to find a stable matching in the market, which further illustrates the crucial
role of gap4 in finding the real stable matching. We leave the lower bound in general markets as an
important future direction.

5
Experiments

In this section, we compare our Algorithm 1 with baselines ETGS [16], ML-ETC [30] and Phased
ETC [4] which also enjoy guarantees for player-optimal stable regret in general decentralized one-
to-one markets. To better illustrate the advantages of our algorithm, especially when N is much
smaller than K, we set N = 3 and K = 10. The preference rankings for both players and arms
are generated as random permutations. The preference gap between any adjacent ranked arms is
set as 0.1. The feedback Xi,j(t) for player pi on arm aj at time t is drawn independently from the
Gaussian distribution with mean µi,j and variance 1. We report the maximum cumulative player-
optimal stable regret among all players and the cumulative player-optimal instability in Figure 1 (a)
and (b), respectively. Here the cumulative player-optimal unstability is defined as the number of
matchings that are not the player-optimal stable one. All algorithms run for T = 100k rounds and
all results are averaged over 50 independent runs. The error bars represent standard errors, which
are computed as standard deviations divided by
√

50.

As shown in the figure, our AOGS algorithm, which only conducts necessary explorations over un-
known preferences and promptly eliminates sub-optimal arms, achieves the least cumulative regret
and cumulative player-optimal unstability among all baselines. The ML-ETC and ETGS algorithms
need to sufficiently explore K arms to estimate the full preference ranking, requiring more ex-
ploration time to find the player-optimal stable matching. The PhasedETC algorithm has not yet
converged within the displayed rounds due to the cold start problem.

8


---Page Break---
0
20k
40k
60k
80k
100k
Round t

0

5k

10k

15k

20k

25k

Maximum Cumulative Regret

(a) Cumulative optimal stable regret

AOGS
ETGS
PhasedETC
ML-ETC

0
20k
40k
60k
80k
100k
Round t

0

10k

20k

30k

40k

50k

Player-optimal Unstability

(b) Cumulative player-optimal unstability

AOGS
ETGS
PhasedETC
ML-ETC

Figure 1: Experimental comparisons of our AOGS with ETGS, ML-ETC and Phased ETC in one-
to-one decentralized markets with N = 3 players and K = 10 arms.

6
Centralized UCB Algorithm for α-condition

In this section, we provide a new analysis for the centralized UCB algorithm in markets satisfying
α-condition. The algorithm is first introduced by Liu et al. [19]. For completeness, we present the
full algorithm in Algorithm 2. At each round, players submit their UCB rankings to the centralized
platform (Line 3). The platform runs the GS algorithm (based on players’ submitted rankings) and
returns the partner to each player (Line 4).

Algorithm 2 centralized UCB
Input: N, K.

1: Initialize: ˆµi,j(0) = 0, Ti,j(0) = 0, UCBi,j(1) = ∞, ∀i ∈[N], j ∈[K].
2: for round t = 1, 2, . . . , T do
3:
Receive
rankings
ˆσ
:=
{ˆσi}i∈[N]
according
to
the
decreasing
order
of
{UCBi,j(t)}j∈[K], ∀i ∈[N];
4:
Ai(t) ←Gale-Shapley (ˆσ, π) for each player pi;
5:
Observe Xi(t), and update ˆµi,j(t), Ti,j(t), UCBi,j(t + 1) for each pi, aj;
6: end for

In the following, we introduce the α-condition. Conditions guaranteeing the unique stable matching
have been widely studied in the offline setting [5, 13] and also the online setting to improve the
learning efficiency [27, 4, 21]. Among these conditions, the α-condition is shown to be the weakest
sufficient one [13] and incorporate the conditions studied in existing works [27, 4, 21].

Let β denote a pair of permutations of [N] and [K]. Then [N]β = {Q(β)
1 , . . . , Q(β)
N } and [K]β =
{q(β)
1
, . . . , q(β)
K } denote permutations of the ordered sets [N] and [K], respectively. The j-th player
in [N]β is the Q(β)
j
-th player in [N], and the k-th arm in [K]β is the q(β)
k -th arm in [K]. Then we
can define the α-condition below.

Definition 6.1. The α-condition is satisfied if there is a stable matching (j∗, i∗), a left-order of
players and arms s.t. ∀i ∈[N]l, ∀j > i, j ∈[K]l : µi,j∗
i > µi,j where j∗
i is the partner of
player pi in stable matching (j∗, i∗), and a (possibly different) right-order of players and arms s.t.
∀j < i ≤N, qj ∈[K]r, Qi ∈[N]r : πqj,Qi∗qj > πqj,Qi. Here similarly, i∗
qj is the partner of arm aqj
in stable matching (j∗, i∗).

Without loss of generality, we consider the identity of players and arms is just the left order, i.e.,
[N] = [N]l and [K] = [K]l. Thus we only deal with player order Q(r)
i
= Qi and arm order
q(r)
j
= qj, for i ∈[N], j ∈[K] in the rest of the paper. Under α-condition, it is easy to inductively
verify that for any i ∈[N], the player pi is matched with arm ai, and the player pQi is matched with
the arm aqi in the unique stable matching [4].

9


---Page Break---
6.1
Theoretical Results

We analyze the regret for the centralized UCB algorithm under α-condition.
Theorem 6.2. When preferences of participants satisfy α-condition, following Algorithm 2, the
stable regret for each player pi satisfies

Regi(T) ≤O
 
N log T/∆2
N + K log T/∆

.

The centralized UCB algorithm is proposed by Liu et al. [19] and shown to have O(NK log T/∆2)
player-pessimal stable regret for general markets. We provide a new analysis for markets satisfying
α-condition which removes the dependence of K in the regret. Due to the space limit, the detailed
proof is deferred to Appendix C.

Key idea of the proof.
We investigate the preference structure of α-condition to obtain the im-
proved analysis. For player pi, its regret is due to selecting sub-optimal arm ak with µi,k < µi,i.
Arm ak will be selected by pi when its UCB value is higher than pi’s stable matched arm ai, which
time is bounded by O(log T/∆2
i,i,k), and one ∆i,i,k on the denominator can be eliminated when
multiplying ∆i,i,k to compute regret. This contributes O (K log T/∆) regret since there are at most
K −1 sub-optimal arms. It is worth noting that arm ak will also be selected by pi if pi is rejected
by ai in the GS algorithm. Recall that under α-condition, there is a right order Qi′ = i ∈[N]r for
player pi, such that ∀i′′ > i′, Qi′′ ∈[N]r : πqi′,Qi′ > πqi′,Qi′′ , which means arm aqi′ = ai can
only prefer players pQ1, pQ2, · · · , pQi′−1 than player pQi′ = pi. Thus pi is rejected by ai only when
these players select ai, and ai is sub-optimal for those players. To bound the regret of pi when being
rejected, we just need to bound the exploration times of these players pQ1, pQ2, · · · , pQi′−1 on arm
ai. However, the exploration time of a single player pQℓwith 1 ≤ℓ≤i′ −1 on ai can not be
trivially bounded by O(log T/∆2
N) since pQℓmay have to select arm ai after rejected by its stable
arm aqℓin offline GS, where aqℓmight be selected by pQ1, · · · , pQℓ−1. This leads to a recursion
form. We control this term using the fact that when a player is rejected by its stable matched arm
in the GS, it can date back to a higher right-order player wrongly over-estimate its preference for a
sub-optimal arm. This key observation and the definition of ∆make it possible to derive the final
O
 
N log T/∆2
N

bound.

7
Conclusion

In this paper, we investigate the problem of whether a tighter bound can be derived for the bandit
learning problem in two-sided matching markets. For the general one-to-one matching markets,
we try to improve the learning efficiency of the existing algorithms. By integrating the offline GS
procedure into the online learning process and carefully designing the elimination strategy, we show
that the player-optimal stable regret can be upper bounded by O(N 2 log T/∆2 + K log T/∆). This
result removes the dependence on K in the main order term of existing works and improves the state-
of-the-art result [30, 16] in common cases where the number of players is much smaller than that of
arms. An experiment is conducted to verify its advantage over other baselines in such markets. We
also present a novel analysis for the centralized UCB algorithm in markets satisfying α-condition
and derive an improved O(N log T/∆2
N + K log T/∆) regret upper bound.

One significant future direction is to investigate the optimality of algorithms. Although the depen-
dence on N, K, T in Theorem 6.2 matches the lower bound, the definition of ∆differs. It remains
unclear how the upper bound changes with the same ∆. Furthermore, since the lower bound pro-
vided by [27] applies only to special markets, and the learning problem in general markets is more
challenging due to the complex preference structure, determining whether an algorithm can perform
better in general markets is still an open problem.

Acknowledgments and Disclosure of Funding

The corresponding author Shuai Li is supported by National Key Research and Development Pro-
gram of China (2022ZD0114804) and National Natural Science Foundation of China (62376154).

We thank Yuhao Zhang and Wenqian Wang for valuable discussions and suggestions on the proof
of Lemma B.2.

10


---Page Break---
References

[1] Atila Abdulkadiro˘glu and Tayfun S¨onmez. House allocation with existing tenants. Journal of
Economic Theory, 88(2):233–260, 1999.

[2] Shipra Agrawal and Navin Goyal. Further optimal regret bounds for thompson sampling. In
Proceedings of the 16th International Conference on Artificial Intelligence and Statistics, pages
99–107, 2013.

[3] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer.
Finite-time analysis of the multiarmed
bandit problem. Machine learning, 47(2):235–256, 2002.

[4] Soumya Basu, Karthik Abinav Sankararaman, and Abishek Sankararaman. Beyond log2(t)
regret for decentralized bandits in matching markets. In International Conference on Machine
Learning, pages 705–715, 2021.

[5] Simon Clark. The uniqueness of stable matchings. Contributions in Theoretical Economics,
6(1), 2006.

[6] Xiaowu Dai and Michael Jordan. Learning in multi-stage decentralized matching markets. In
Advances in Neural Information Processing Systems, volume 34, pages 12798–12809, 2021.

[7] Xiaowu Dai and Michael I Jordan. Learning strategies in decentralized matching markets under
uncertain preferences. Journal of Machine Learning Research, 22(1):11806–11855, 2021.

[8] Sanmay Das and Emir Kamenica. Two-sided bandits and the dating market. In International
Joint Conference on Artificial Intelligence, pages 947–952, 2005.

[9] David Gale and Lloyd S Shapley. College admissions and the stability of marriage. The Ameri-
can Mathematical Monthly, 69(1):9–15, 1962.

[10] Aur´elien Garivier, Tor Lattimore, and Emilie Kaufmann. On explore-then-commit strategies.
In Advances in Neural Information Processing Systems, volume 29, pages 784–792, 2016.

[11] Avishek Ghosh, Abishek Sankararaman, Kannan Ramchandran, Tara Javidi, and Arya Mazum-
dar.
Decentralized competing bandits in non-stationary matching markets.
arXiv preprint
arXiv:2206.00120, 2022.

[12] Meena Jagadeesan, Alexander Wei, Yixin Wang, Michael Jordan, and Jacob Steinhardt. Learn-
ing equilibria in matching markets from bandit feedback. In Advances in Neural Information
Processing Systems, volume 34, 2021.

[13] Alexander Karpov. A necessary and sufficient condition for uniqueness consistency in the
stable marriage matching problem. Economics Letters, 178:63–65, 2019.

[14] Emilie Kaufmann, Nathaniel Korda, and R´emi Munos. Thompson sampling: An asymptoti-
cally optimal finite-time analysis. In International Conference on Algorithmic Learning Theory,
pages 199–213. Springer, 2012.

[15] Alexander S Kelso Jr and Vincent P Crawford. Job matching, coalition formation, and gross
substitutes. Econometrica: Journal of the Econometric Society, pages 1483–1504, 1982.

[16] Fang Kong and Shuai Li. Player-optimal stable regret for bandit learning in matching mar-
kets. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA).
SIAM, 2023.

[17] Fang Kong, Junming Yin, and Shuai Li. Thompson sampling for bandit learning in matching
markets. In International Joint Conference on Artificial Intelligence, 2022.

[18] Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

[19] Lydia T Liu, Horia Mania, and Michael Jordan. Competing bandits in matching markets.
In International Conference on Artificial Intelligence and Statistics, pages 1618–1628. PMLR,
2020.

11


---Page Break---
[20] Lydia T Liu, Feng Ruan, Horia Mania, and Michael I Jordan. Bandit learning in decentralized
matching markets. Journal of Machine Learning Research, 22(211):1–34, 2021.

[21] Chinmay Maheshwari, Eric Mazumdar, and Shankar Sastry. Decentralized, communication-
and coordination-free learning in structured matching markets. In Advances in Neural Informa-
tion Processing Systems, 2022.

[22] Yifei Min, Tianhao Wang, Ruitu Xu, Zhaoran Wang, Michael Jordan, and Zhuoran Yang. Learn
to match with no regret: Reinforcement learning in markov matching markets. In Advances in
Neural Information Processing Systems, volume 35, pages 19956–19970, 2022.

[23] Deepan Muthirayan, Chinmay Maheshwari, Pramod Khargonekar, and Shankar Sastry. Com-
peting bandits in time varying matching markets. In Learning for Dynamics and Control Confer-
ence, pages 1020–1031. PMLR, 2023.

[24] Alvin E Roth and Marilda Sotomayor. Two-sided matching. Handbook of game theory with
economic applications, 1:485–541, 1992.

[25] Alvin E Roth. The evolution of the labor market for medical interns and residents: a case study
in game theory. Journal of political Economy, 92(6):991–1016, 1984.

[26] Alvin E Roth. Stability and polarization of interests in job matching. Econometrica: Journal
of the Econometric Society, pages 47–57, 1984.

[27] Abishek Sankararaman, Soumya Basu, and Karthik Abinav Sankararaman.
Dominate or
delete: Decentralized competing bandits in serial dictatorship. In International Conference on
Artificial Intelligence and Statistics, pages 1252–1260. PMLR, 2021.

[28] Zilong Wang and Shuai Li. Optimal analysis for bandit learning in matching markets with
serial dictatorship. Theoretical Computer Science, 1010:114703, 2024.

[29] Zilong Wang, Liya Guo, Junming Yin, and Shuai Li. Bandit learning in many-to-one matching
markets. In Proceedings of the 31st ACM International Conference on Information & Knowledge
Management, pages 2088–2097, 2022.

[30] Yirui Zhang, Siwei Wang, and Zhixuan Fang. Matching in multi-arm bandit with collision. In
Advances in Neural Information Processing Systems, 2022.

12


---Page Break---
A
Proof of Theorem 4.1

Define F =
n
∃1 ≤t ≤T, i ∈[N], j ∈[K] : |ˆµi,j(t) −µi,j| >
q

6 log T
Ti,j(t)
o
as the failure event that
the estimated reward is far from the expected reward at some time and some player-arm pair. The
regret can be upper bounded as follows.

Regi(t) =E

" T
X

t=1

 
µi,m∗
i −Xi(t)

#

≤E

" T
X

t=1

 
µi,m∗
i −Xi(t)

|⌝F

#

+ P (F) · T · µi,m∗
i

≤E




T
X

t=1

X

aj
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F



+ E

" T
X

t=1
1
 ¯Ai(t) = ∅
	
· µi,m∗
i |⌝F

#

+ P (F) · T · µi,m∗
i
≤96N 2 log T/∆2 + 96K log T/∆+ 192N 2 log T/∆2 + 2NK
(1)

=O
 
N 2 log T/∆2 + K log T/∆

.

where Eq. (1) holds based on Lemma A.1, A.2, and A.4.

Lemma A.1.

E




T
X

t=1

X

aj
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F



≤96N 2 log T/∆2 + 96K log T/∆.

Proof. Recall that player pi would update the available set Ai when other players pi′ sets Ei′ =
False and πAi′,i′ > πAi′,i as Line 14. Denote ts as the round index when this operation happens for
the s-th time. Without loss of generality, let t0 = 1.

Recall that at a high level, each time another player pi′ sets Ei′ as False, it means that pi′ learns
its most preferred arm in current available set. Combined with F and Lemma A.6, the determined
arm of players during each exploration would be truly their most preferred one. Thus the AOGS
algorithm is an online version of GS and the s′-th time player pi′ sets Ei′ as False is equivalent to
that pi′ proposes its s′-th most preferred arm in the offline GS. According to Lemma A.3, at most
N −1 arms are proposed by all players before reaching stability. Thus for player pi, the operation
in Line 14 would happen for at most N −1 times.

Recall that for each s, during time ts to ts+1, player pi would explore all available arms in a round-
robin manner, eliminate sub-optimal arms until N arms are in the set, and focuses on the best one
among these N when it is identified. For convenience, denote Rs as the set of the remaining N arms
that pi explored in Ai in a round-robin manner until condition Line 9 is satisfied, Ds as the set of
arms that pi eliminated due to condition Line 6, and js as the arm that pi focuses from the time it
sets Ei as False to time ts+1 −1. Then it holds that

E




T
X

t=1

X

aj
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F





≤E




N−1
X

s=0

ts+1−1
X

t=ts

X

aj
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F





≤E




N−1
X

s=0

ts+1−1
X

t=ts



X

aj∈Rs
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j +
X

aj∈Ds
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j

+1
 ¯Ai(t) = ajs
	
· ∆i,m∗
i ,js

|⌝F


13


---Page Break---
≤E




N−1
X

s=0

ts+1−1
X

t=ts



X

aj∈Rs
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j +
X

aj∈Ds
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j



|⌝F



,

(2)

where Eq. (2) is due to that, based on Lemma A.6 and the offline GS algorithm, the arm js before
GS stops would be better than m∗
i , thus the regret caused by selecting these arms is less than 0.

For the first term in Eq. (2), we have

E




N−1
X

s=0

ts+1−1
X

t=ts

X

aj∈Rs
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F




(3)

=E




N−1
X

s=0

X

aj∈Rs

ts+1−1
X

t=ts
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F





≤E




N−1
X

s=0

X

aj∈Rs

96 log T
∆2
i,js,js+1
· ∆i,m∗
i ,j |⌝F




(4)

≤96N 2 log T

∆2
.
(5)

where Eq. (4) is due to Lemma A.5, Eq. (5) holds since Rs contains no more than N arms according
to the elimination condition (Line 6).

We now analyze the second term in Eq. (2). For any arm aj and s ∈{0, ..., N −1}, denote
Ti,j,s as the value of Ti,j at the end of the round ts+1 −1.
For s ≥1 and arm aj ∈Ds,
if Ti,j,s−1
≤96 log T/∆2
i,js−1,j, it must hold that Ti,j,s
:= P

s′≤s(Ti,j,s′ −Ti,j,s′−1) ≤
96 log T/∆2
i,js,j to ensure arm aj is eliminated from Ai at step s based on Lemma A.5. On the
other hand, if Ti,j,s−1 > 96 log T/∆2
i,js−1,j, based on Lemma A.5, it holds that Ti,j,s −Ti,j,s−1 ≤
96 log T/∆2
i,js,j −96 log T/∆2
i,js−1,j when aj is eliminated.
For any arm aj, denote sj,1 :=
max0≤s≤N−1

s : Ti,j,s ≤96 log T/∆2
i,js,j
	
as the last step when the number of observation times
on aj is less than that threshold. Then the second term in Eq. (2) satisfies

E




N−1
X

s=0

ts+1−1
X

t=ts

X

aj∈Ds
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F





=E



X

aj∈K

X

s:aj∈Ds

ts+1−1
X

t=ts
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F





=E



X

aj∈K

X

s:aj∈Ds
(Ti,j,s −Ti,j,s−1) · ∆i,m∗
i ,j |⌝F





=E



X

aj∈K




X

s:aj∈Ds,s≤sj,1
(Ti,j,s −Ti,j,s−1) +
X

s:aj∈Ds,s>sj,1
(Ti,j,s −Ti,j,s−1)



· ∆i,m∗
i ,j |⌝F





≤E



X

aj∈K



Ti,j,sj,1 +
X

s:aj∈Ds,s>sj,1
(Ti,j,s −Ti,j,s−1)



· ∆i,m∗
i ,j |⌝F





≤
X

aj∈K



96 log T

∆2
i,jsj,1,j
+
X

s:aj∈Ds,s>sj,1


96 log T/∆2
i,js,j −96 log T/∆2
i,js−1,j



· ∆i,m∗
i ,j

14


---Page Break---
≤
X

aj∈K

96 log T

∆2
i,m∗
i ,j
· ∆i,m∗
i ,j ≤96K log T/∆,

where the second last line is due to the definition of sj,1 and the above analysis.

Above all,

E




T
X

t=1

X

aj
1
 ¯Ai(t) = aj
	
· ∆i,m∗
i ,j |⌝F



≤Eq. (2) ≤96N 2 log T/∆2 + 96K log T/∆.

Lemma A.2.

E

" T
X

t=1
1
 ¯Ai(t) = ∅
	
· µi,m∗
i |⌝F

#

≤192N 2 log T/∆2 .

Proof. Based on the AOGS algorithm, when Ei = True, the central platform would assign arms in
Ai to player pi in a round-robin manner. Since the number of arms |∪i:Ei=TrueAi| to be explored
is larger than the number P

i 1{Ei = True} of players with Ei = True based on the elimination
condition in Line 6, we can assume that there is no collision in the exploration phase as discussed in
Section 4. So the regret caused by collision only occurs during time with Ei = False.

Denote ts and ts as the round index when pi sets Ei as False for the s-th time and as True for the
s + 1-th time, respectively. Recall that when Ei = False, pi will always select arm Ai. Here we use
js to represent the arm that is selected by pi from time ts to ts.

Further, recall that in the AOGS algorithm, each time an arm is added into Di (Line 13), the elim-
inated arms may be contained into Ai again. And only when other players focus on their currently
most preferred arm, such operation of adding arms to D happens. Based on Lemma A.3, such an
operation happens for at most N times. For any player pi′, denote ti′,r as the round index when pi′
adds arms to Di′ (Line 13) for r-th time. Then {ti′,r}r∈[N] further divide

[ts, ts]
	

s∈[N] into at most

2N slices. We use t′

s, t′s to represent the start round and end round index of the s-th slice, where
s ∈[2N]. Based on Lemma A.5, pi′ and pi would select the same arm for at most 96 log T/∆2
times within each slice.

Then the regret satisfies

E

" T
X

t=1
1
 ¯Ai(t) = ∅
	
· µi,m∗
i |⌝F

#

≤E




N
X

s=1

ts
X

t=ts
1
 ¯Ai(t) = ∅
	
· µi,m∗
i |⌝F





= E




N
X

s=1

ts
X

t=ts
1
 ¯Ai(t) = ∅, Ai(t) = js
	
· µi,m∗
i |⌝F





≤E



X

i′̸=i

N
X

s=1

ts
X

t=ts
1{Ai(t) = Ai′(t) = js} · µi,m∗
i |⌝F





≤E



X

i′̸=i

2N
X

s=1

t′s
X

t=t′

s
1{Ai(t) = Ai′(t)} · µi,m∗
i |⌝F





≤
X

i′̸=i

2N
X

s=1
96 log T/∆2 · µi,m∗
i

≤192N 2 log T/∆2 .

Lemma A.3. In the offline GS algorithm, at most N −1 arms have been proposed by players before
the algorithm stops.

15


---Page Break---
Proof. Based on the offline GS algorithm, once an arm is proposed, it has a temporary player. By
contradiction, once N arms have been proposed, it means that N players are occupied. In this case,
each player has a partner and the algorithm stops.

Lemma A.4.

P (F) ≤2NK/T .

Proof.

P (F) = P

 

∃1 ≤t ≤T, i ∈[N], j ∈[K] : |ˆµi,j(t) −µi,j| >

s

6 log T

Ti,j(t)

!

≤

T
X

t=1

X

i∈[N]

X

j∈[K]
P

 

|ˆµi,j(t) −µi,j| >

s

6 log T

Ti,j(t)

!

≤

T
X

t=1

X

i∈[N]

X

j∈[K]

t
X

s=1
P

 

Ti,j(t) = s, |ˆµi,j(t) −µi,j| >

r

6 log T

s

!

≤

T
X

t=1

X

i∈[N]

X

j∈[K]
t · 2 exp(−3 ln T)

≤2NK/T ,

where the second last inequality is due to Lemma B.1.

Lemma A.5. For any player pi, let ¯Ti = 96 log T/∆2. For any two arms j, j′ with µi,j > µi,j′ and
σi(aj) ∈[1, σi(m∗
i )], if Ti(t) := min {Ti,j(t), Ti,j′(t)} > ¯Ti, we have UCBi,j′(t) < LCBi,j(t)
conditioned on ⌝F.

Proof. By contradiction, suppose UCBi,j′(t) ≥LCBi,j(t). According to ⌝F and the definition of
LCB and UCB, we have

µi,j −2

s

6 log T

Ti(t) ≤LCBi,j(t) ≤UCBi,j′(t) ≤µi,j′ + 2

s

6 log T

Ti(t) .

We can then conclude ∆i,j,j′ = µi,j −µi,j′ ≤4
q

6 log T

Ti(t) , which implies that Ti(t) ≤96 log T

∆2
i,j,j′ ≤

96 log T

∆2
. This contradicts the fact that Ti(t) > ¯Ti.

Lemma A.6. Conditioned on ⌝F, at any time t, UCBi,j(t) < LCBi,j′(t) implies µi,j < µi,j′.

Proof. According to the definition of LCB and UCB, we have

LCBi,j(t) = ˆµi,j(t) −

s

6 log T

Ti,j(t) ≤µi,j ≤ˆµi,j(t) +

s

6 log T

Ti,j(t) = UCBi,j(t) ,

where two inequalities comes from ⌝F. Thus if UCBi,j(t) < LCBi,j′(t), there would be

µi,j ≤UCBi,j(t) < LCBi,j′(t) ≤µi,j′ .

The lemma can thus be proved.

B
Technical Lemmas

Lemma B.1. (Corollary 5.5 in Lattimore and Szepesv´ari [18]) Assume that X1, X2, . . . , Xn are
independent, σ-subgaussian random variables centered around µ. Then for any ε > 0,

P

 
1
n

n
X

i=1
Xi ≥µ + ε

!

≤exp

−nε2

2σ2


,
P

 
1
n

n
X

i=1
Xi ≤µ −ε

!

≤exp

−nε2

2σ2


.

16


---Page Break---
Lemma B.2. (Arrangement of players’ round-robin exploration) Suppose there are N players who
need to explore their respective N arms. There exists an assignment such that during 2N rounds,
each player can match with each of its arm for once.

Proof. Without loss of generality, let’s assume that players assign their respective N arms in 2N
rounds one by one, based on the players’ and arms’ indices, aiming to ensure that no arm is assigned
to more than one player at the same round. By contradiction, suppose when player pi assigns its j-th
arm, there is no available round to make this assignment due to conflicting constraints. Given that
player pi is currently assigning the j-th arm, it implies that there are 2N −j+1 rounds where no arm
is assigned to player pi. Since none of these rounds satisfy the conflict constraint, it means that the
previous i −1 players assigned arm j in these 2N −j + 1 rounds. This creates a contradiction since
the first i −1 players can only occupy i −1 rounds when selecting arm j, where i −1 < 2N −j + 1
with i ≤N, j ≤N.

C
Proof of Theorem 6.2

In this section, we analyze the regret of the centralized-UCB algorithm under α-condition. Recall
that F =
n
∃1 ≤t ≤T, i ∈[N], j ∈[K] : |ˆµi,j(t) −µi,j| >
q

6 log T
Ti,j(t)
o
is the failure event that the
estimated reward is far from the expected reward at some time and some player-arm pair.

For any player pi with i ∈[N], we know that its stable arm is ai under α-condition. Thus its regret
can be decomposed as

Regi(T) ≤E




X

k:µi,k<µi,i
∆i,i,k

T
X

t=1
1
 ¯Ai(t) = k, ⌝F
	


+ T · P (F) .

The first term is the number of selections for sub-optimal arms. The second term is the regret caused
by the bad events.

For the arm ak such that it is sub-optimal for player pi, i.e., µi,k < µi,i, it will be selected because
the preference for arm ak of player pi is estimated higher than its stable matched arm ai, or player
pi is rejected by arm ak in the GS algorithm. Note that under α-condition, there is a right order
Qi′ = i ∈[N]r for player pi, such that ∀i′ < i′′ ≤N, Qi′′ ∈[N]r:πqi′,Qi′ > πqi′,Qi′′ , which
means arm aqi′ = ai can only prefer players pQ1, pQ2, · · · , pQi′−1 than player pQi′ = pi. Denote
The right-order mapping for α-condition for player pi is lr(i) so that Qlr(i) = i with Qi defined in
Definition 6.1, and lr(i) ≤N. For player pi, denote Gt,i := {∀1 ≤i′ ≤lr(i) −1, ¯AQi′(t) ̸= i}
as the event all players preferred by arm ai do not select pi at time t. Then the number of selections
for sub-optimal arm ak can be decomposed as

E

" T
X

t=1
1
 ¯Ai(t) = k, ⌝F
	
#

=E

" T
X

t=1
1
 ¯Ai(t) = k, Gt,i, ⌝F
	
#

+ E

" T
X

t=1
1
 ¯Ai(t) = k, ⌝Gt,i, ⌝F
	
#

≤E

" T
X

t=1
1
 ¯Ai(t) = k, UCBi,k(t) > UCBi,i(t), ⌝F
	
#

+ E

" T
X

t=1
1
 ¯Ai(t) = k, ⌝Gt,i, ⌝F
	
#

≤24 log T

∆2
i,i,k
+ E

" T
X

t=1
1
 ¯Ai(t) = k, ⌝Gt,i, ⌝F
	
#

.

The last inequality is from Lemma C.1.

For the second term in the RHS of the last inequality, E
hPT
t=1 1
 ¯Ai(t) = k, ⌝Gt,i, ⌝F
	i
, we can
sum over all sub-optimal arms and it turns out to be

E

"X

k

T
X

t=1
1
 ¯Ai(t) = k, ⌝Gt,i, ⌝F
	
#

17


---Page Break---
≤E

" T
X

t=1
1{⌝Gt,i, ⌝F}

#

≤E





lr(i)−1
X

i′=1

T
X

t=1
1
 ¯AQi′(t) = i, ⌝F
	




≤

lr(i)−1
X

u′=1

lr(i)
X

u′′=u′+1

24 log T
∆2
Qu′,qu′,qu′′

≤

lr(i)−1
X

u′=1

lr(i)−u′
X

k=1

24 log T
(k∆N)2

≤(lr(i) −1)





lr(i)−u′
X

k=1

1
k2



24 log T

∆2
N

≤(lr(i) −1) 5π2 log T

∆2
N
,

where the third inequity is from the Lemma C.2. The fourth inequality is from the definition of ∆N.

Above all, the stable regret of player i can be bounded by

Regi(T) ≤E




X

k:µi,k<µi,i
∆i,i,k

T
X

t=1
1
 ¯Ai(t) = k, ⌝F
	


+ T · P (F)

≤
X

k:µi,k<µi,i
∆i,i,k
24 log T

∆2
i,i,k
+ ∆i,i,k (lr(i) −1) 5π2 log T

∆2
N
+ 2NK

≤24K log T

∆
+ (lr(i) −1) 5π2 log T

∆2
N
+ 2NK

≤O
K log T

∆
+ N log T

∆2
N


,

where the second inequality is based on Lemma A.4.

Lemma C.1. Conditioned on ⌝F, under the traditional single-player UCB algorithm with single
player pi, the expected number of times at which the UCB index of arm aj′ exceeds that of the better
arm aj, is at most 24 log(T)/∆2
i,j,j′ by round T.

Proof. Conditioned on ⌝F, for any i, j, t we have,

µi,j −

s

6 log(T)
Ti,j(t −1) < ˆµi,j(t −1) < µi,j +

s

6 log(T)
Ti,j(t −1) .
(6a)

Recall that the UCB index is:

UCBi,j(t) = ˆµi,j(t −1) +

s

6 log(T)
Ti,j(t −1) .
(6b)

The event that arm aj′ is successfully selected for player pi rather than the better arm aj at time t
implies that

UCBi,j′(t) > UCBi,j(t) .
(6c)

Hence,

18


---Page Break---
µi,j′ + 2

s

6 log(T)
Ti,j′(t −1)

(6a)
> ˆµi,j′(t −1) +

s

6 log(T)
Ti,j′(t −1)

(6c)
> ˆµi,j(t −1) +

s

6 log(T)
Ti,j(t −1)

> µi,j −

s

6 log(T)
Ti,j(t −1) +

s

6 log(T)
Ti,j(t −1)

= µi,j ,

which leads to

Ti,j′(t −1) < 24 log(T)

∆2
i,j,j′
,

where ∆i,j,j′ is the reward difference between the µi,j′ and µi,j.

Lemma C.2. For any player pi with right order Qlr(i), the following inequality holds:

E





lr(i)−1
X

i′=1

T
X

t=1
1
 ¯AQi′(t) = i, ⌝F
	




≤E





lr(i)−1
X

u′=1

lr(i)
X

u′′=u′+1

T
X

t=1
1
 ¯AQu′(t) = qu′′, Gt,u′, ⌝F
	




≤

lr(i)−1
X

u′=1

lr(i)
X

u′′=u′+1

24 log T
∆2
Qu′,qu′,qu′′
.

Proof. For player pi with right order Qlr(i), from α-condition we have that its stable matched arm
ai may prefer pQ1, pQ2, · · · , pQlr(i)−1 than player pQlr(i). For any i′ < lr(i), we know that the
number of times player pQi′ selects arm ai is decomposed as by

E

" T
X

t=1
1
 ¯AQi′(t) = i, ⌝F
	
#

=E

" T
X

t=1
1
 ¯AQi′(t) = i, Gt,i′, ⌝F
	
#

+ E

" T
X

t=1
1
 ¯AQi′(t) = i, ⌝Gt,i′, ⌝F
	
#

.

The event ⌝Gt,i′ implies that there exists another player pQi′′ with i′′ < i′ that selects the stable arm
of pQi′. This leads to a recursion form. But it is easy to verify that every event ⌝Gt,i′ happens only
when there exists two players pQu′ , pQu′′ with u′ < u′′ ≤lr(i′), such that player pQu′ explores the
stable matched arm aqu′′ of pQu′′, i.e., pQu′ selects aqu′′ conditioned on Gt,u′. And thus it holds that

E





lr(i)−1
X

i′=1

T
X

t=1
1
 ¯AQi′(t) = i, ⌝F
	




≤E





lr(i)−1
X

u′=1

lr(i)
X

u′′=u′+1

T
X

t=1
1
 ¯AQu′(t) = qu′′, Gt,u′, ⌝F
	




≤

lr(i)−1
X

u′=1

lr(i)
X

u′′=u′+1

24 log T
∆2
Qu′,qu′,qu′′
.

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our abstract and introduction clearly describe the scope and outline our main
contributions.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these
goals are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 7 discusses the limitation of this paper.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The au-
thors should reflect on how these assumptions might be violated in practice and what
the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the ap-
proach. For example, a facial recognition algorithm may perform poorly when image
resolution is low or images are taken in low lighting. Or a speech-to-text system might
not be used reliably to provide closed captions for online lectures because it fails to
handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to ad-
dress problems of privacy and fairness.
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

Justification: We provide techniques clearly in the paper, and their detailed proofs are in
Appendix.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theo-
rems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a
short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be comple-
mented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclu-
sions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 5 provides the settings and results of our experiments carefully.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps
taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture
fully might suffice, or if the contribution is a specific model and empirical evaluation,
it may be necessary to either make it possible for others to replicate the model with
the same dataset, or provide access to the model. In general. releasing code and data
is often one good way to accomplish this, but reproducibility can also be provided via
detailed instructions for how to replicate the results, access to a hosted model (e.g., in
the case of a large language model), releasing of a model checkpoint, or other means
that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all sub-
missions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear
how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to re-
produce the model (e.g., with an open-source dataset or instructions for how to
construct the dataset).

21


---Page Break---
(d) We recognize that reproducibility may be tricky in some cases, in which case au-
thors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: The codes are uploaded in supplementary material.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
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

Justification: Section 5 provides the settings and results of our experiments carefully.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of
detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropri-
ate information about the statistical significance of the experiments?

Answer: [Yes]

22


---Page Break---
Justification: We provide the error bar in the results of experiments in Section 5.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [No]

Justification: We do not provide specific information on the computer resources used, as
most experiments on multi-armed bandits are lightweight.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments
that didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper complies with NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.

23


---Page Break---
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact spe-
cific groups), privacy considerations, and security considerations.
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
• If there are negative societal impacts, the authors could also discuss possible mitiga-
tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by re-
quiring that users adhere to usage guidelines or restrictions to access the model or
implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. Licenses for existing assets

24


---Page Break---
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the pack-
age should be provided. For popular datasets, paperswithcode.com/datasets has
curated licenses for some datasets. Their licensing guide can help determine the li-
cense of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documenta-
tion provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can
either create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the pa-
per include the full text of instructions given to participants and screenshots, if applicable,
as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Including this information in the supplemental material is fine, but if the main contri-
bution of the paper involves human subjects, then as much detail as possible should
be included in the main paper.

25


---Page Break---
• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
tion, or other labor should be paid at least the minimum wage in the country of the
data collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equiva-
lent) may be required for any human subjects research. If you obtained IRB approval,
you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity
(if applicable), such as the institution conducting the review.

26


---Page Break---
