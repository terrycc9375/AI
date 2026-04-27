Learning-Augmented Priority Queues

Ziyad Benomar
ENSAE, Ecole Polytechnique,
FairPlay joint team
ziyad.benomar@ensae.fr

Christian Coester
Department of Computer Science
University of Oxford, UK
christian.coester@cs.ox.ac.uk

Abstract

Priority queues are one of the most fundamental and widely used data structures
in computer science. Their primary objective is to efficiently support the inser-
tion of new elements with assigned priorities and the extraction of the highest
priority element. In this study, we investigate the design of priority queues within
the learning-augmented framework, where algorithms use potentially inaccurate
predictions to enhance their worst-case performance. We examine three prediction
models spanning different use cases, and we show how the predictions can be
leveraged to enhance the performance of priority queue operations. Moreover, we
demonstrate the optimality of our solution and discuss some possible applications.

1
Introduction

Priority queues are an essential abstract data type in computer science [Jaiswal, 1968, Brodal, 2013]
whose objective is to enable the swift insertion of new elements and access or deletion of the
highest priority element. Their applications span a wide range of problems within computer science
and beyond. They play a crucial role in sorting [Williams, 1964, Thorup, 2007], in various graph
algorithms such as Dijkstra’s shortest path algorithm [Chen et al., 2007] or computing minimum
spanning trees [Chazelle, 2000], in operating systems for scheduling and load balancing [Sharma
et al., 2022], in networking protocols for managing data transmission packets [Moon et al., 2000], in
discrete simulations for efficient event processing based on occurrence time [Goh and Thng, 2004],
and in implementing hierarchical clustering algorithms [Day and Edelsbrunner, 1984, Olson, 1995].

Various data structures can be used to implement priority queues, each offering distinct advantages
and tradeoffs [Brodal, 2013]. However, it is established that a priority queue with n elements cannot
guarantee o(log n) time for all the required operations [Brodal and Okasaki, 1996]. This limitation
can be surpassed within the learning augmented framework [Mitzenmacher and Vassilvitskii, 2022],
where the algorithms can benefit from machine-learned or expert advice to improve their worst-case
performance. We propose in this work learning-augmented implementations of priority queues in
three different prediction models, detailed in the next section.

1.1
Problem definition

A priority queue is a dynamic data structure where each element x is assigned a key u from a totally
ordered universe (U, <), determining its priority. The standard operations of priority queues are:

(i) FindMin(): returns the element with the smallest key without removing it,

(ii) ExtractMin(): removes and returns the element with the smallest key,

(iii) Insert(x, u): adds a new element x to the priority queue with key u,

(iv) DecreaseKey(x, v): decreases the key of an element x to v.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
The elements of the priority queue can be accessed via their keys in O(1) time using a HashMap.
Hence, the focus is on establishing efficient algorithms for key storage and organization, facilitating
the execution of priority queue operations. For any key u ∈U and subset V ⊂U, we denote by
r(u, V) the rank of u in V, defined as the number of keys in V that are smaller than or equal to u,

r(u, V) = #{v ∈V : v ≤u} .
(1)

The difficulty lies in designing data structures offering adequate tradeoffs between the complexities
of the operations listed above. This paper explores how using predictions can allow us to overcome
the limitations of traditional priority queues. We examine three types of predictions.

Dirty comparisons.
In the first prediction model, comparing two keys (u, v) ∈U2 is slow or costly.
However, the algorithm can query a prediction of the comparison (u < v). This prediction serves as
a rapid or inexpensive, but possibly inaccurate, method of comparing elements of U, termed a dirty
comparison and denoted by (u b< v). Conversely, the true outcome of (u < v) is referred to as a
clean comparison. For all u ∈U and V ⊆U, we denote by η(u, V) the number of inaccurate dirty
comparisons between u and elements of V,

η(u, V) = #{v ∈V : 1(u b< v) ̸= 1(u < v)} .
(2)

This prediction model was introduced in [Bai and Coester, 2023] for sorting, but it has a broader
significance in comparison-based problems, such as search [Borgstrom and Kosaraju, 1993, Nowak,
2009, Tschopp et al., 2011], ranking [Wauthier et al., 2013, Shah and Wainwright, 2018, Heckel et al.,
2018], and the design of comparison-based ML algorithms [Haghiri et al., 2017, 2018, Ghoshdastidar
et al., 2019, Perrot et al., 2020, Meister and Nietert, 2021]. Particularly, it has great theoretical impor-
tance for priority queues, which have been extensively studied in the comparison-based framework
[Gonnet and Munro, 1986, Brodal and Okasaki, 1996, Edelkamp and Wegener, 2000]. Comparison-
based models are often used, for example, when the preferences are determined by human subjects.
Assigning numerical scores in these cases is inexact and prone to errors, while pairwise comparisons
are absolute and more robust [David, 1963]. Within such a setting, dirty comparisons can be obtained
by a binary classifier, and used to minimize human inference yielding clean comparisons, which are
time-consuming and might incur additional costs.

Pointer predictions.
In this second model, upon the addition of a new key u to the priority queue
Q, the algorithm receives a prediction [
Pred(u, Q) ∈Q of the predecessor of u, which is the largest
key belonging to Q and smaller than u. Before u is inserted, the ranks of u and its true predecessor
in Q are equal, hence we define the prediction error as

⃗η(u, Q) = |r(u, Q) −r([
Pred(u, Q), Q)| .
(3)

In priority queue implementations, a HashMap preserves a pointer from each inserted key to its
corresponding position in the priority queue. Consequently, [
Pred(u, Q) provides direct access to the
predicted predecessor’s position. For example, this prediction model finds applications in scenarios
where concurrent machines have access to the priority queue [Sundell and Tsigas, 2005, Shavit and
Lotan, 2000, Lindén and Jonsson, 2013]. Each machine can estimate, within the elements it has
previously inserted, which one precedes the next element it intends to insert. However, this estimation
might not be accurate, as other concurrent machines may have inserted additional elements.

Rank predictions.
The last setting assumes that the priority queue is used in a process where a finite
number N of distinct keys will be inserted, i.e., the priority queue is not used indefinitely, but N is
unknown to the algorithm. Upon the insertion of any new key ui, the algorithm receives a prediction
bR(ui) of the rank of ui among all the N keys {uj}j∈[N]. Denoting by R(ui) = r(ui, {uj}j∈[N]) the
true rank, the prediction error of ui is

η∆(ui) = |R(ui) −bR(ui)| .
(4)

The same prediction model was explored in [McCauley et al., 2024] for the online list labeling
problem. Bai and Coester [2023] investigate a similar model for the sorting problem, but in an offline
setting where the N elements to sort and the predictions are accessible to the algorithm from the start.

Note that N counts the distinct keys added with Insert or DecreaseKey operations. An arbitrarily
large number of DecreaseKey operations thus can make N arbitrarily large although the total number

2


---Page Break---
of insertions is reduced. However, with a lazy implementation of DecreaseKey, we can assume
without loss of generality that N is at most quadratic in the total number of insertions. Indeed, it
is possible to omit executing the DecreaseKey operations when queried, and only store the new
elements’ keys to update. Then, at the first ExtractMin operation, all the element’s keys are updated
by executing for each one only the last queried DecreaseKey operation involving it. Denoting by
k the total number of insertions, there are at most k ExtractMin operations, and for each of them,
there are at most k DecreaseKey operations executed. The total number of effectively executed
DecreaseKey operations is therefore O(k2). In particular, this implies that a complexity of O(log N)
is also logarithmic in the total number of insertions.

1.2
Our results

We first investigate augmenting binary heaps with predictions. To leverage dirty comparisons, we
first design a randomized binary search algorithm to find the position of an element u in a sorted
list L. We prove that it terminates using O(log |L|) dirty comparisons and O(log η(u, L)) clean
comparisons in expectation. Subsequently, we use this result to establish an insertion algorithm in
binary heaps using O(log log n) dirty comparisons and reducing the number of clean comparisons to
O(log log η(u, Q)). However, ExtractMin still mandates O(log n) clean comparisons. In the two
other prediction models, binary heaps and other heap implementations of priority queues appear
unsuitable, as the positions of the keys are not determined solely by their ranks.

Consequently, in Section 3, we shift to using skip lists. We devise randomized insertion algo-
rithms requiring, in expectation, only O(log ⃗η(u, Q)) time and comparisons in the pointer prediction
model, O(log n) time and O(log η(u, Q)) clean comparisons in the dirty comparison model, and
O(log log N + log maxi∈[N] η∆(ui)) time and O(log maxi∈[N] η∆(ui)) comparisons in the rank
prediction model, where we use in the latter an auxiliary van Emde Boas (vEB) tree [van Emde Boas
et al., 1976]. Across the three prediction models, FindMin and ExtractMin only necessitate O(1)
time, and the complexity of DecreaseKey aligns with that of insertion. Finally, we prove in Theorem
3.4 the optimality of our data structure. Table 1.2 summarizes the complexities of our learning-
augmented priority queue (LAPQ) in the three prediction models compared to standard priority queue
implementations. The complexity of FindMin is O(1) for all the listed priority queues.

Priority queues
ExtractMin
Insert
DecreaseKey
Binary Heap
O(log n)
O(log log n)
O(log n)
Fibonacci Heap (amortized)
O(log n)
O(1)
Skip List (average)
O(1)
O(log n)
LAPQ with dirty comparisons (average)
O(1)
O(log η(u, Q))
LAPQ with pointer predictions (average)
O(1)
O(log ⃗η(u, Q))
LAPQ with rank predictions (average)
O(1)
O(log maxi∈[n] η∆(ui))

Table 1: Number of comparisons per operation used by different priority queues.

Our learning-augmented data structure enables additional operations beyond those of priority queues,
such as the maximum priority queue operations FindMax, ExtractMax, and IncreaseKey with anal-
ogous complexities, and removing an arbitrary key u from the priority queue, finding its predecessor
or successor in expected O(1) time.

Furthermore, we show in Section 4.1 that it can be used for sorting, yielding the same guarantees as
the learning-augmented sorting algorithms presented in [Bai and Coester, 2023] for the positional
predictions model with displacement error, and for the dirty comparison model. In the second model,
our priority queue offers even stronger guarantees, as it maintains the elements sorted at any time
even if the insertion order is adversarial, while the algorithm of [Bai and Coester, 2023] requires a
random insertion order to achieve a sorted list by the end within the complexity guarantees. We also
show how the learning-augmented priority queue can be used to accelerate Dijkstra’s algorithm.

Finally, in Section 5, we compare the performance of our priority queue using predictions with
binary and Fibonacci heaps when used for sorting and for Dijkstra’s algorithm on both real-world
city maps and synthetic graphs. The experimental results confirm our theoretical findings, showing
that adequately using predictions significantly reduces the complexity of priority queue operations.

3


---Page Break---
1.3
Related work

In this section, we briefly discuss related works on learning-augmented algorithms and priority queues.
For a more extensive review of related work, please refer to Appendix A.

Learning-augmented algorithms.
Learning-augmented algorithms, introduced in [Lykouris and
Vassilvtiskii, 2018, Purohit et al., 2018], have captured increasing interest over the last years, as
they allow breaking longstanding limitations in many algorithm design problems. Assuming that
the decision-maker is provided with potentially incorrect predictions regarding unknown parameters
of the problem, learning-augmented algorithms must be capable of leveraging these predictions if
they are accurate (consistency), while keeping the worst-case performance without advice even if the
predictions are arbitrarily bad or adversarial (robustness). While many fundamental online problems
were studied in this setting (see Appendix A), the design of data structures with predictions remains
relatively underexplored. The seminal paper by [Kraska et al., 2018] shows how predictions can be
used to optimize space usage. Another study by [Lin et al., 2022] demonstrates that the runtime of
binary search trees can be optimized by incorporating predictions of item access frequency. Recent
papers have extended this prediction model to other data structures, such as dictionaries [Zeynali
et al., 2024] and skip lists [Fu et al., 2024]. The prediction models we study in the current paper
deviate from the latter, and are more related to those considered respectively in [Bai and Coester,
2023] for sorting, and [McCauley et al., 2024] for online list labeling. An overview of the growing
body of work on learning-augmented algorithms (also known as algorithms with predictions) is
maintained at [Lindermayr and Megow, 2022].

Priority queues implementations.
Binary heaps, introduced by Williams [1964], are one of the
first efficient implementations of priority queues. They allow all the operations in O(log n) time,
where n is the number of items in the queue. A first improvement was introduced with Binomial
heaps [Vuillemin, 1978], reducing the amortized time of insertion to O(1). A breakthrough came
later with Fibonacci heaps [Fredman and Tarjan, 1987], which allow all the operations in constant
amortized time, except for ExtractMin, which takes O(log n) time. However, Fibonacci heaps are
known to be slow in practice [Larkin et al., 2014], and other implementations with weaker theoretical
guarantees such as binary heaps are often preferred. Another possible implementation uses skip lists
[Pugh, 1990], which are probabilistic data structures, guaranteeing in expectation a constant time for
FindMin and ExtractMin, and O(log n) time for Insert and DecreaseKey.

2
Heap priority queues

A common implementation of priority queues uses binary heaps, enabling all operations in O(log n)
time. Binary heaps maintain a balanced binary tree structure, where all depth levels are fully filled,
except possibly for the last one, to which we refer in all this section as the leaf level. Moreover,
it satisfies the heap property, i.e., any key is smaller than all its children. To maintain these two
structure properties, when a new element is added, it is first inserted in the leftmost empty position in
the leaf level, and then repeatedly swapped with its parent until the heap property is restored.

2.1
Insertion in the comparison-based model

Insertion in a binary heap can be accomplished using only O(log log n) comparisons, albeit O(log n)
time, by doing a binary search of the new element’s position along the path from the leftmost empty
position in the leaf level to the root, which is a sorted list of size O(log n). To improve the insertion
complexity with dirty comparisons, we first tackle the search problem in this setting.

Search with dirty comparisons.
Consider a sorted list L = (v1, . . . , vk) and a target u, the position
of u in L can be found using binary search with O(log k) comparisons. Extending ideas from Bai
and Coester [2023] and Lykouris and Vassilvtiskii [2018], this complexity can be reduced using
dirty comparisons. Indeed, we can obtain an estimated position br(u, L) through a binary search with
dirty comparisons, followed by an exponential search with clean comparisons, starting from br(u, L)
to find the exact position r(u, L). However, the positions of inaccurate dirty comparisons can be
adversarially chosen to compromise the algorithm. This can be addressed by introducing randomness
to the dirty search phase. We refer to the randomized binary search as the algorithm that proceeds

4


---Page Break---
similarly to the binary search, but whenever the search is reduced to an array {vi, . . . , vj}, instead
of comparing u to the pivot vm with index m = i + ⌊j−i

2 ⌋, it compares u to a pivot with an index
chosen uniformly at random in the range {i + ⌈j−i

4 ⌉, . . . , j −⌈j−i

4 ⌉}.
Theorem 2.1. A dirty randomized binary search followed by a clean exponential search finds
the target’s position using O(log k) dirty comparisons and O(log η(u, L)) clean comparisons in
expectation.

Randomized insertion in a binary heap.
In a binary heap Q, any new element u is always inserted
along the path of size O(log n) from the root to the leftmost empty position in the leaf level. If
all the inaccurate dirty comparisons are chosen along this path, then the insertion would require
O(log η(u, Q)) clean comparisons by Theorem 2.1. This complexity can be reduced further by
randomizing the choice of the root-leaf path where u is inserted, as explained in Algorithm 1.

Algorithm 1: Randomized insertion in binary heap
Input: Binary heap Q with randomly filled positions in the leaf level, new key u

1 L ←keys path from a uniformly random empty position in the leaf level to the root;

2 br(u, L) ←outcome of a dirty randomized binary search of u in L;

3 r(u, L) ←outcome of the clean exponential search of u in L starting from index br(u, L);

4 Insert u in the chosen leaf empty position, then swap it with its parents until position r(u, L);

Theorem 2.2. The insertion algorithm 1 requires O(log n) time, O(log log n) dirty comparisons and
O(log log η(u, Q)) clean comparisons in expectation.

2.2
Limitations

The previous theorem demonstrates that accurate predictions can reduce the number of clean compar-
isons for insertion in a binary heap. However, for the ExtractMin operation, when the minimum
key, which is the root of the tree, is deleted, its two children are compared and the smallest is placed
in the root position, and this process repeats recursively, with each new empty position filled by
comparing both of its children, requiring necessarily O(log n) clean comparisons in total to ensure
the heap priority remains intact. Improving the efficiency of ExtractMin using dirty comparisons
would therefore require bringing major modifications to the binary heap’s structure.

Similar difficulties arise when attempting to enhance ExtractMin using dirty comparisons or the
other prediction models in different heap implementations, such as Binomial or Fibonacci heaps.
Consequently, we explore in the next section another priority queue implementation, using skip lists,
which allows for an easier and more efficient exploitation of the predictions.

3
Skip lists

A priority queue can be implemented naively by maintaining a dynamic sorted linked list of keys.
This guarantees constant time for ExtractMin, but O(n) time for insertion. Skip lists (see Figure 1)
offer a solution to this inefficiency, by maintaining multiple levels of linked lists, with higher levels
containing fewer elements and acting as shortcuts to lower levels, facilitating faster search and
insertion in expected O(log n) time. In all subsequent discussions concerning linked lists or skip lists,
it is assumed that they are doubly linked, having both predecessor and successor pointers between
elements.

The first level in a skip list is an ordinary linked list containing all the elements, which we denote
by v1, . . . , vn. Every higher level is constructed by including each element from the previous level
independently with probability p, typically set to 1/2. For any key vi in the skip list, we define its
height h(vi) as the number of levels where it appears, which is an independent geometric random
variable with parameter p. A number 2h(vi) of pointers are associated with vi, giving access to
the previous and next element in each level ℓ∈[h(vi)], denoted respectively by Prev(vi, ℓ) and
Next(vi, ℓ). Using a HashMap, these pointers can be accessed in O(1) time via the key value vi. For
convenience, we consider that the skip list contains two additional keys v0 = −∞and vn+1 = ∞,
corresponding respectively to the head and the NIL value. Both have a height equal to the maximum
height in the queue h(v0) = h(vn+1) = maxi∈[n] h(vi).

5


---Page Break---
HEAD

NIL

v1
v2
v3
v4
v5
v6
v7
v8
v9

Figure 1: A skip list with keys v1 < . . . < v9 ∈U.

Since the expected height of keys in the skip list is 1/p, deleting any key only requires a constant
time in expectation, by updating its associated pointers, along with those of its predecessors and
successors in the levels where it appears. In particular, FindMin and ExtractMin take O(1) time,
and DecreaseKey can be performed by deleting the element and reinserting it with the new key,
yielding the same complexity as insertion. Furthermore, by the same arguments, inserting a new key
u next to a given key vi in the skip list can be done in expected constant time.

Therefore, implementing efficient Insert and DecreaseKey operations for skip lists with predictions
is reduced to designing efficient search algorithms to find the predecessor of a target key u in the skip
list, i.e., the largest key vi in the skip list that is smaller than u. In all the following, we denote by Q a
skip list containing n keys v1 ≤. . . ≤vn ∈U, and u ∈U the target key. As explained in Appendix
C.1, we can assume without loss of generality that the keys (u, v1, . . . , vn) are pairwise distinct. In
the following, we present separately for each model an insertion algorithm leveraging the predictions.

3.1
Pointer prediction

Given a pointer prediction vj = [
Pred(u, Q), we describe below an algorithm for finding the true
predecessor of u starting from the position of the key vj, then inserting u. We assume in the algorithm
that vj ≤u. If vj > u, then the algorithm can be easily adapted by reversing the search direction.

Algorithm 2: ExpSearchInsertion(Q, vj, u)
Input: Skip list Q, source vj ∈Q, and new key u ∈U

1 w ←vj;
▷Bottom-Up search

2 while Next(w, h(w)) ≤u do

3
w ←Next(w, h(w));

4 ℓ←h(w);
▷Top-Down search

5 while ℓ> 0 do

6
while Next(w, ℓ) ≤u do

7
w ←Next(w, ℓ);

8
ℓ←ℓ−1;

9 Insert u next to w;

Algorithm 2 is inspired by the classical exponential search in arrays. The first phase consists of a
bottom-up search, expanding the size of the search interval by moving to upper levels until finding
a key w ∈Q satisfying w ≤u < Next(w, h(w)). The second phase conducts a top-down search
from level h(w) downward, refining the search until locating the position of u. It is worth noting that
the classical search algorithm in skip lists, denoted by Search(Q, u), corresponds precisely to the
top-down search, starting from the head of the skip list instead of w.

Theorem 3.1. Augmented with pointer predictions, a skip list allows FindMin and ExtractMin in
expected O(1) time, and Insert(u) in expected O(log ⃗η(u, Q)) time using Algorithm 2.

6


---Page Break---
3.2
Dirty comparisons

We devise in this section a search algorithm using dirty and clean comparisons. Algorithm 3 first
estimates the position of u with a dirty top-down search starting from the head, then performs a clean
exponential search starting from the estimated position to find the true position.

Algorithm 3: Insertion with dirty and clean comparisons
Input: Skip list Q, new key u ∈U

1 ˆw ←Search(Q, u) with dirty comparisons;

2 ExpSearchInsertion(Q, ˆw, u) with clean comparisons;

The dirty search concludes within O(log n) steps, and Theorem 3.1 guarantees that the exponential
search terminates within O(log |r( ˆw, Q) −r(u, Q)|) steps. Combining these results and relating the
distance between u and ˆw in Q to the prediction error η(u, Q), we derive the following theorem.

Theorem 3.2. Augmented with dirty comparisons, a skip list allows FindMin and ExtractMin in
O(1) expected time, and Insert(u) with Algorithm 3 in O(log n) expected time, using O(log n) dirty
comparisons and O(log η(u, Q)) clean comparisons in expectation.

3.3
Rank predictions

In the rank prediction model, each Insert(u) request is accompanied by a prediction bR(u) of the rank
of u among all the distinct keys already in, or to be inserted into the priority queue. If the predictions
are accurate and the total number N of distinct keys to be inserted is known, the problem reduces to
designing a priority queue with integer keys in [N], taking as keys the ranks (Ri)i∈[N]. This problem
can be addressed using a van Emde Boas (vEB) tree over [N] [van Emde Boas et al., 1976], which
requires O(log log N) time for insertion, deletion, finding the minimum or maximum, and finding
the predecessor or successor of any element, guaranteeing in particular O(log log N) time for all
priority queue operations. More details on its structure can be found in Appendix C.5.

To leverage rank predictions, we use an auxiliary vEB tree T along with the skip list Q. All the
insertion and deletion operations are made simultaneously on T and Q. However, the priorities used
in T are the predicted ranks { bR(ui)}i∈[N]. Whenever a new key ui is to be added, Algorithm 4
inserts it first in T at position bR(ui), gets its predecessor ˆw in T , i.e., the element in T with the
largest predicted rank smaller than or equal to bR(ui), then uses ˆw as a pointer prediction to find the
position of ui in Q. If the predecessor is not unique, the algorithm chooses an arbitrary one.

Algorithm 4: Insertion with rank prediction

Input: Skip list Q, vEB tree T on [N], new element ui ∈U, prediction bR(ui) ∈[N]

1 Insert ui in T with key bR(ui);

2 ˆw ←predecessor of ui in T ;

3 ExpSearchInsertion(Q, ˆw, ui);

We explain in Appendix C.5 how the data structure can be adapted when N is unknown and the rank
predictions are not necessarily in [N], and we prove the following theorem, giving both the runtime
and comparison complexities of the priority queue operations using this data structure.

Theorem 3.3. If bR(ui) = O(N) for all i ∈[N], then there is a data structure allowing FindMin and
ExtractMin in O(1) amortized time, and Insert in O(log log N + log maxi∈[N] η∆(ui)) amortized
time using O(log maxi∈[N] η∆(ui)) comparisons in expectation.

In contrast to other prediction models, the complexity of inserting ui is not impacted only by η∆(ui),
but by the maximum error over all keys {uj}j∈[N]. This occurs because the exponential search
conducted in Algorithm 4 starts from the key ˆw ∈Q, whose error also affects insertion performance.
A similar behavior is observed in the online list labeling problem [McCauley et al., 2024], where the
bounds provided by the authors also depend on the maximum prediction error for insertion.

7


---Page Break---
With perfect predictions, the number of comparisons for insertion becomes constant, and its runtime
O(log log N). It is not clear if the runtime of all the priority queue operations can be reduced to O(1)
with perfect predictions. Indeed, the problem in that case is reduced to a priority queue with all the
keys in [N]. The best-known solution to this problem is a randomized priority queue, by Thorup
[2007], supporting all operations in O(√log log N) time. However, in our approach, we use vEB
trees beyond the classical priority queue operations, as we also require fast access to the predecessor
of any element. A data structure supporting all these operations solves the dynamic predecessor
problem, for which vEB trees are optimal [P˘atra¸scu and Thorup, 2006]. Reducing the runtime of
insertion below O(log log N) would therefore require omitting the use of predecessor queries.

3.4
Lower bounds

As explained earlier, ExtractMin requires only O(1) expected time in skip lists. Furthermore, we
presented insertion algorithms for the three prediction models and provided upper bounds on their
complexities. The following theorem establishes lower bounds on the complexities of ExtractMin
and Insert for any priority queue augmented with any of the three prediction types.
Theorem 3.4. For each of the three prediction models, the following lower bounds hold.

(i) Dirty comparisons: no data structure Q can support ExtractMin with O(1) clean compar-
isons and Insert(u) with o(log η(u, Q)) clean comparisons in expectation.

(ii) Pointer predictions: no data structure Q can support ExtractMin with O(1) comparisons
and Insert(u) with o(log ⃗η(u, Q)) comparisons in expectation.

(iii) Rank predictions: no data structure Q can support ExtractMin with O(1) comparisons
and Insert(ui) with o(log maxi∈[N] η∆(ui)) comparisons in expectation, for all i ∈[N].

These lower bounds with Theorems 3.2 and 3.1 prove the tightness of our priority queue in the
dirty comparison and the pointer prediction models. In the rank prediction model, the comparison
complexities proved in Theorem 3.3 are optimal, whereas the runtimes are only optimal up to an
additional O(log log N) term. In particular, they are optimal if the maximal error is at least Ω(log N).

4
Applications

4.1
Sorting algorithm

Our learning-augmented priority queue can be used for sorting a sequence A = (a1, . . . , an), by first
inserting all the elements, then repeatedly extracting the minimum until the priority queue is empty.
We compare below the performance of this sorting algorithm to those of [Bai and Coester, 2023].

Dirty comparison model.
Denoting by ηi = η(ai, A), Bai and Coester [2023] prove a sorting
algorithm using O(n log n) time, O(n log n) dirty comparisons, and O(P

i log(ηi + 2)) clean com-
parisons. Theorem 3.2 yields the same guarantees with our learning-augmented priority queue.
Moreover, our learning-augmented priority queue is a skip list, maintaining elements in sorted order
at any time, even if the elements are revealed online and the insertion order is adversarial, while in
[Bai and Coester, 2023], it is crucial that the insertion order is chosen uniformly at random.

Positional predictions.
In their second prediction model, they assume that the algorithm is given
offline access to predictions { bR(ai)}i∈[n] of the relative ranks {R(ai)}i∈[n] of the n elements to sort,
and they study two different error measures. The rank prediction error η∆
i = |R(ai)−bR(ai)| matches
their definition of displacement error, for which they prove a sorting algorithm in O(P

i log(η∆
i +2))
time. The same bound can be deduced using our results in the pointer prediction model. Further
discussion on this claim can be found in Appendix E.

Online rank predictions.
If n is unknown to the algorithm, and the elements a1, . . . , an along
with their predicted ranks are revealed online, possibly in an adversarial order, then by Theorem 3.3,
the total runtime of our priority queue for maintaining all the inserted elements sorted at any time
is O(n log log n + n log maxi η∆
i ), and the number of comparisons used is O(n log maxi η∆
i ). No
analogous result is demonstrated in [Bai and Coester, 2023] in this setting.

8


---Page Break---
4.2
Dijkstra’s shortest path algorithm

Consider a run of Dijkstra’s algorithm on a directed positively weighted graph G with n nodes and m
edges. The elements inserted into the priority queue are the nodes of the graph, and the corresponding
keys are their tentative distances to the source, which are updated over time. During the algorithm’s
execution, at most m + 1 distinct keys {di}i∈[m+1] are inserted into the priority queue. Given online
predictions ( bR(di))i∈[m+1] of their relative ranks (R(di))i∈[m+1], the total runtime using our priority
queue augmented with rank predictions is

O
 
m log log n + m log max
i∈[m+1] |R(di) −bR(di)|

.

In contrast, the shortest path algorithm of Lattanzi et al. [2023] (which also works for negative edges)
has a linear dependence on a similar error measure. Even with arbitrary error, our guarantee is
never worse than the O(m log n) runtime with binary heaps. Using Fibonacci heaps results in an
O(n log n + m) runtime, which is surpassed by our learning-augmented priority queue in the case of
sparse graphs where m = o( n log n

log log n) if predictions are of high quality. However, it is known that
Fibonacci heaps perform poorly in practice, even compared to binary heaps, as supported by our
experiments.

5
Experiments

In this section, we empirically evaluate the performance of our learning-augmented priority queue
(LAPQ) by comparing it with Binary and Fibonacci heaps. We use two standard benchmarks for this
evaluation: sorting and Dijkstra’s algorithm. For the sorting benchmark, we also compare our results
with those from Bai and Coester [2023]. For Dijkstra’s algorithm, we assess performance on both
real city maps and synthetic random graphs. In all the experiments, each data point represents the
average result from 30 independent runs. Additional experiments and a detailed discussion on the
prediction models and the obtained results can be found in Appendix F. The code used for conducting
the experiments is available at github.com/Ziyad-Benomar/Learning-augmented-priority-queues.

Sorting.
We compare sorting using our LAPQ with the algorithms of Bai and Coester [2023] under
their same experimental settings. Given a sequence A = (a1, . . . , an), we evaluate the complexity of
sorting it with predictions in the class and the decay setting. In the first, A is divided into c classes
((tk−1, tk])k∈[c], where 0 = t0 ≤t1 ≤. . . ≤tc = n are uniformly random thresholds. The predicted
rank of any item ai with tk ≤i < tk+1 is sampled uniformly at random within (tk, tk+1]. In the
decay setting, the ranking is initially accurate but degrades over time. Each time step, one item’s
predicted position is perturbed by 1, either left or right, uniformly at random.

In both settings, we test the LAPQ with the three prediction models. First, assuming that the rank
predictions are given offline, we use pointer predictions as explained in Appendix E. In the second
case, the elements to insert along with their predicted ranks are revealed online in a uniformly random
order. Finally, we test the dirty comparison setting with the dirty order (ai b< aj) = ( bR(ai) < bR(aj)).

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

100

101

(#comparisons)/n

n = 104

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

100

101

n = 105

LAPQ ofﬂine predictions
LAPQ online predictions
Dirty-clean sort
Double-Hoover sort
Displacement sort
Binary heap
Fibonacci heap

Figure 2: Sorting in the class setting

0
20
40
60
80
100
(#timesteps)/n

2

4

6

8

10

12

14

(#comparisons)/n

n = 104

0
50
100
150
200
250
300
(#timesteps)/n

2.5

5.0

7.5

10.0

12.5

15.0

17.5

n = 105

LAPQ ofﬂine predictions
LAPQ online predictions
Dirty-clean sort
Double-Hoover sort
Displacement sort
Binary heap

Figure 3: Sorting in the decay setting

Figures 2 and 3 show the obtained results respectively in the class and the decay setting for
n ∈{104, 105}. In the class setting with offline predictions, the LAPQ slightly outperforms the
Double-Hoover and Displacement sort algorithms of Bai and Coester [2023], which were shown to
outperform classical sorting algorithms. In the decay setting, the LAPQ matches the performance

9


---Page Break---
of the Displacement sort, but is slightly outperformed by the Double-Hoover sort. With online
predictions, although the problem is harder, LAPQ’s performance remains comparable to the pre-
vious algorithms. In both settings, the LAPQ with offline predictions, online predictions, and dirty
comparisons all yield better performance than binary or Fibonacci heaps, even with predictions that
are not highly accurate.

Dijkstra’s algorithm.
Consider a graph G = (V, E) with n nodes and m edges, and a source
node s ∈V . In the first predictions setting, we pick a random node ˆs and run Dijkstra’s algorithm
with ˆs as the source, memorizing all the keys ˆD = ( ˆd1, . . . , ˆdm) inserted into the priority queue. In
subsequent runs of the algorithm with different sources, when a key di is to be inserted, we augment
the insertion with the rank prediction bR(di) = r(di, ˆD). We call these key rank predictions. This
model aims at exploiting the topology and uniformity of city maps. As computing shortest paths from
any source necessitates traversing all graph edges, keys inserted into the priority queue—partial sums
of edge lengths—are likely to exhibit some degree of similarity even if the algorithm is executed from
different sources. Notably, this prediction model offers an explicit method for computing predictions,
readily applicable in real-world scenarios.

In the second setting, we consider rank predictions of the nodes in G, ordered by their distances to s.
As Dijkstra’s algorithm explores a new node x ∈V , it receives a prediction br(x) of its rank. The
node x is then inserted with a key di, to which we assign the prediction bR(di) = br(x). Unlike the
previous experimental settings, we initially have predictions of the nodes’ ranks, which we extend
to predictions of the keys’ ranks. Similarly to the sorting experiments, we consider class and decay
perturbations of the node ranks.

In the context of searching the shortest path, rank predictions in the class setting can be derived from
subdividing the city into multiple smaller areas. Each class corresponds to a specific area, facilitating
the ordering of areas from closest to furthest relative to the source. However, comparing the distances
from the source to the nodes in the same class might be inaccurate. On the other hand, the decay
setting simulates modifications to shortest paths, such as rural works or new route constructions, by
adding or removing edges from the graph. These alterations may affect the ranks of a limited number
of nodes, which corresponds to the time steps in the decay setting.

We present below the experiment results obtained with the maps of Paris and London. More
experiments with additional city maps and synthetic graphs are in Appendix F. The city maps were
obtained using the Python library Osmnx [Boeing, 2017]. Figures 4 and 5 respectively illustrate the
results in the class and the decay settings with node rank predictions. In both figures, for each city,
we present the numbers of comparisons used for the same task by a binary and Fibonacci heap, and
the number of comparisons used when the priority queue is augmented with key rank predictions.

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

4

6

8

10

12

14

16

(#comparisons)/n

Paris: n = 9559,m = 18400

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

5.0

7.5

10.0

12.5

15.0

17.5

20.0

London: n = 128924,m = 300612

LAPQ node rank predictions
LAPQ node dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 4: Shortest path, class predictions

0
5
10
15
20
(#timesteps)/n

3

4

5

6

7

8

9

10

11

(#comparisons)/n

Paris: n = 9559,m = 18400

0
5
10
15
20
(#timesteps)/n

4

6

8

10

12

14

London: n = 128924,m = 300612

LAPQ node rank predictions
LAPQ node dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 5: Shortest path, decay predictions

In both settings, the performance of the LAPQ substantially improves with the quality of the
predictions, and notably, key rank predictions yield almost the same performance as perfect node rank
predictions, affirming our intuition on the similarity between the keys inserted in runs of Dijkstra’s
algorithm starting from different sources.

References

Keerti Anand, Rong Ge, and Debmalya Panigrahi. Customizing ml predictions for online algorithms.
In International Conference on Machine Learning, pages 303–313. PMLR, 2020.

10


---Page Break---
Antonios Antoniadis, Themis Gouleakis, Pieter Kleer, and Pavel Kolev. Secretary and online matching
problems with machine learned advice. Advances in Neural Information Processing Systems, 33:
7933–7944, 2020.

Antonios Antoniadis, Christian Coester, Marek Eliás, Adam Polak, and Bertrand Simon. Learning-
augmented dynamic power management with multiple states via new ski rental bounds. Advances
in Neural Information Processing Systems, 34:16714–16726, 2021.

Antonios Antoniadis, Joan Boyar, Marek Eliás, Lene Monrad Favrholdt, Ruben Hoeksma, Kim S
Larsen, Adam Polak, and Bertrand Simon. Paging with succinct predictions. In International
Conference on Machine Learning, pages 952–968. PMLR, 2023a.

Antonios Antoniadis, Christian Coester, Marek Eliáš, Adam Polak, and Bertrand Simon. Online
metric algorithms with untrusted predictions. ACM Transactions on Algorithms, 19(2):1–34,
2023b.

Xingjian Bai and Christian Coester. Sorting with predictions. Advances in Neural Information
Processing Systems, 36, 2023.

Etienne Bamas, Andreas Maggiori, and Ola Svensson. The primal-dual method for learning aug-
mented algorithms. Advances in Neural Information Processing Systems, 33:20083–20094, 2020.

Nikhil Bansal, Christian Coester, Ravi Kumar, Manish Purohit, and Erik Vee. Learning-augmented
weighted paging. In Proceedings of the 2022 ACM-SIAM Symposium on Discrete Algorithms,
SODA, pages 67–89. SIAM, 2022.

Dmitry Basin, Edward Bortnikov, Anastasia Braginsky, Guy Golan-Gueta, Eshcar Hillel, Idit Keidar,
and Moshe Sulamy. Kiwi: A key-value map for scalable real-time analytics. In Proceedings of
the 22Nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, pages
357–369, 2017.

MohammadHossein Bateni, Prathamesh Dharangutte, Rajesh Jayaram, and Chen Wang. Metric
clustering and mst with strong and weak distance oracles. arXiv preprint arXiv:2310.15863, 2023.

Ziyad Benomar and Vianney Perchet. Advice querying under budget constraint for online algorithms.
Advances in Neural Information Processing Systems, 36, 2024a.

Ziyad Benomar and Vianney Perchet. Non-clairvoyant scheduling with partial predictions. In
Forty-first International Conference on Machine Learning, 2024b.

Ziyad Benomar, Chaima Ghribi, Elie Cali, Alexander Hinsen, and Benedikt Jahnel. Agent-based
modeling and simulation for malware spreading in d2d networks. AAMAS ’22, page 91–99,
Richland, SC, 2022a. International Foundation for Autonomous Agents and Multiagent Systems.
ISBN 9781450392136.

Ziyad Benomar, Chaima Ghribi, Elie Cali, Alexander Hinsen, Benedikt Jahnel, and Jean-Philippe
Wary. Multi-agent simulations for virus propagation in D2D 5G+ networks. 2022b.

Ziyad Benomar, Evgenii Chzhen, Nicolas Schreuder, and Vianney Perchet. Addressing bias in online
selection with limited budget of comparisons. arXiv preprint arXiv:2303.09205, 2023.

Aditya Bhaskara, Ashok Cutkosky, Ravi Kumar, and Manish Purohit. Logarithmic regret from
sublinear hints. Advances in Neural Information Processing Systems, 34:28222–28232, 2021.

Geoff Boeing. Osmnx: New methods for acquiring, constructing, analyzing, and visualizing complex
street networks. Computers, environment and urban systems, 65:126–139, 2017.

Ryan S Borgstrom and S Rao Kosaraju. Comparison-based search in the presence of errors. In
Proceedings of the twenty-fifth annual ACM symposium on Theory of computing, pages 130–136,
1993.

Gerth Stølting Brodal. Worst-case efficient priority queues. In Proceedings of the Seventh Annual
ACM-SIAM Symposium on Discrete Algorithms, SODA ’96, page 52–58, USA, 1996. Society for
Industrial and Applied Mathematics. ISBN 0898713668.

11


---Page Break---
Gerth Stølting Brodal. A survey on priority queues. In Space-Efficient Data Structures, Streams,
and Algorithms: Papers in Honor of J. Ian Munro on the Occasion of His 66th Birthday, pages
150–163. Springer, 2013.

Gerth Stølting Brodal and Chris Okasaki. Optimal purely functional priority queues. Journal of
Functional Programming, 6(6):839–857, 1996.

Gerth Stølting Brodal, George Lagogiannis, and Robert E Tarjan. Strict fibonacci heaps. In Pro-
ceedings of the forty-fourth annual ACM symposium on Theory of computing, pages 1177–1184,
2012.

Bernard Chazelle. A minimum spanning tree algorithm with inverse-ackermann type complexity.
Journal of the ACM (JACM), 47(6):1028–1047, 2000.

Justin Chen, Sandeep Silwal, Ali Vakilian, and Fred Zhang. Faster fundamental graph algorithms via
learned predictions. In International Conference on Machine Learning, pages 3583–3602. PMLR,
2022.

Mo Chen, Rezaul Alam Chowdhury, Vijaya Ramachandran, David Lan Roche, and Lingling Tong.
Priority queues and dijkstra’s algorithm. 2007.

Jakub Chlkedowski, Adam Polak, Bartosz Szabucki, and Konrad Tomasz .Zolna. Robust learning-
augmented caching: An experimental study. In International Conference on Machine Learning,
pages 1920–1930. PMLR, 2021.

Nicolas Christianson, Junxuan Shen, and Adam Wierman. Optimal robustness-consistency tradeoffs
for learning-augmented metrical task systems. In International Conference on Artificial Intelligence
and Statistics, pages 9377–9399. PMLR, 2023.

Clark Allan Crane. Linear lists and priority queues as balanced binary trees. Stanford University,
1972.

Herbert Aron David. The method of paired comparisons, volume 12. London, 1963.

William HE Day and Herbert Edelsbrunner. Efficient algorithms for agglomerative hierarchical
clustering methods. Journal of classification, 1(1):7–24, 1984.

Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, Ali Vakilian, and Nikos Zarifis. Learning
online algorithms with distributional advice. In International Conference on Machine Learning,
pages 2687–2696. PMLR, 2021.

Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, and Sergei Vassilvitskii. Faster
matchings via learned duals. Advances in neural information processing systems, 34:10393–10406,
2021.

Franziska Eberle, Felix Hommelsheim, Alexander Lindermayr, Zhenwei Liu, Nicole Megow, and
Jens Schlöter. Accelerating matroid optimization through fast imprecise oracles. arXiv preprint
arXiv:2402.02774, 2024.

Stefan Edelkamp and Ingo Wegener. On the performance of weak-heapsort. In Annual Symposium
on Theoretical Aspects of Computer Science, pages 254–266. Springer, 2000.

Bennett Eisenberg. On the expectation of the maximum of iid geometric random variables. Statistics
& Probability Letters, 78(2):135–143, 2008.

Michael L Fredman and Robert Endre Tarjan. Fibonacci heaps and their uses in improved network
optimization algorithms. Journal of the ACM (JACM), 34(3):596–615, 1987.

Chunkai Fu, Jung Hoon Seo, and Samson Zhou. Learning-augmented skip lists. arXiv preprint
arXiv:2402.10457, 2024.

Anna Gambin and Adam Malinowski. Randomized meldable priority queues. In International
Conference on Current Trends in Theory and Practice of Computer Science, pages 344–349.
Springer, 1998.

12


---Page Break---
Tingjian Ge and Stan Zdonik. A skip-list approach for efficiently processing forecasting queries.
Proceedings of the VLDB Endowment, 1(1):984–995, 2008.

Debarghya Ghoshdastidar, Michaël Perrot, and Ulrike von Luxburg. Foundations of comparison-based
hierarchical clustering. Advances in neural information processing systems, 32, 2019.

Catherine Gloaguen and Elie Cali. Cost estimation of a fixed network deployment over an urban
territory. Annals of Telecommunications, 73:367–380, 2018.

Catherine Gloaguen, Frank Fleischer, Hendrik Schmidt, and Volker Schmidt. Fitting of stochastic
telecommunication network models via distance measures and monte–carlo tests. Telecommunica-
tion Systems, 31:353–377, 2006.

Rick Siow Mong Goh and Ian Li-Jin Thng. Dsplay: An efficient dynamic priority queue structure
for discrete event simulation. In Proceedings of the SimTecT Simulation Technology and Training
Conference. Citeseer, 2004.

Sreenivas Gollapudi and Debmalya Panigrahi. Online algorithms for rent-or-buy with expert advice.
In International Conference on Machine Learning, pages 2319–2327. PMLR, 2019.

Gaston H Gonnet and J Ian Munro. Heaps on heaps. SIAM Journal on Computing, 15(4):964–971,
1986.

Siavash Haghiri, Debarghya Ghoshdastidar, and Ulrike von Luxburg. Comparison-based nearest
neighbor search. In Artificial Intelligence and Statistics, pages 851–859. PMLR, 2017.

Siavash Haghiri, Damien Garreau, and Ulrike Luxburg. Comparison-based random forests. In
International Conference on Machine Learning, pages 1871–1880. PMLR, 2018.

Reinhard Heckel, Max Simchowitz, Kannan Ramchandran, and Martin Wainwright. Approximate
ranking from pairwise comparisons. In International Conference on Artificial Intelligence and
Statistics, pages 1057–1066. PMLR, 2018.

Yih-Chun Hu, Adrian Perrig, and David B Johnson. Efficient security mechanisms for routing
protocolsa. In Ndss. Citeseer, 2003.

Sungjin Im, Ravi Kumar, Aditya Petety, and Manish Purohit. Parsimonious learning-augmented
caching. In International Conference on Machine Learning, pages 9588–9601. PMLR, 2022.

Narendra Kumar Jaiswal. Priority queues, volume 50. Academic press New York, 1968.

Tim Kraska, Alex Beutel, Ed H Chi, Jeffrey Dean, and Neoklis Polyzotis. The case for learned index
structures. In Proceedings of the 2018 international conference on management of data, pages
489–504, 2018.

Daniel H Larkin, Siddhartha Sen, and Robert E Tarjan. A back-to-basics empirical study of priority
queues. In 2014 Proceedings of the Sixteenth Workshop on Algorithm Engineering and Experiments
(ALENEX), pages 61–72. SIAM, 2014.

Alexandra Anna Lassota, Alexander Lindermayr, Nicole Megow, and Jens Schlöter. Minimalistic
predictions to schedule jobs with online precedence constraints. In International Conference on
Machine Learning, pages 18563–18583. PMLR, 2023.

Silvio Lattanzi, Ola Svensson, and Sergei Vassilvitskii. Speeding up bellman ford via minimum
violation permutations. In International Conference on Machine Learning, ICML 2023, 2023.

Rhyd Lewis. A comparison of dijkstra’s algorithm using fibonacci heaps, binary heaps, and self-
balancing binary trees. arXiv preprint arXiv:2303.10034, 2023.

Honghao Lin, Tian Luo, and David Woodruff. Learning augmented binary search trees. In Interna-
tional Conference on Machine Learning, pages 13431–13440. PMLR, 2022.

Jonatan Lindén and Bengt Jonsson. A skiplist-based concurrent priority queue with minimal memory
contention. In Principles of Distributed Systems: 17th International Conference, OPODIS 2013,
Nice, France, December 16-18, 2013. Proceedings 17, pages 206–220. Springer, 2013.

13


---Page Break---
Alexander Lindermayr and Nicole Megow.
Algorithms with predictions.
https://
algorithms-with-predictions.github.io, 2022.

Thodoris Lykouris and Sergei Vassilvtiskii. Competitive caching with machine learned advice. In
International Conference on Machine Learning, pages 3296–3305. PMLR, 2018.

Jessica Maghakian, Russell Lee, Mohammad Hajiesmaili, Jian Li, Ramesh Sitaraman, and Zhenhua
Liu. Applied online algorithms with heterogeneous predictors. In International Conference on
Machine Learning, pages 23484–23497. PMLR, 2023.

Samuel McCauley, Ben Moseley, Aidin Niaparast, and Shikha Singh. Online list labeling with
predictions. Advances in Neural Information Processing Systems, 36, 2024.

Michela Meister and Sloan Nietert. Learning with comparison feedback: Online estimation of sample
statistics. In Algorithmic Learning Theory, pages 983–1001. PMLR, 2021.

Nadav Merlis, Hugo Richard, Flore Sentenac, Corentin Odic, Mathieu Molina, and Vianney Perchet.
On preemption and learning in stochastic scheduling. In International Conference on Machine
Learning, pages 24478–24516. PMLR, 2023.

Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions. Communications of
the ACM, 65(7):33–35, 2022.

Sung-Whan Moon, Jennifer Rexford, and Kang G Shin. Scalable hardware priority queue architectures
for high-speed packet switches. IEEE Transactions on computers, 49(11):1215–1227, 2000.

Robert Nowak. Noisy generalized binary search. Advances in neural information processing systems,
22, 2009.

Clark F Olson. Parallel algorithms for hierarchical clustering. Parallel computing, 21(8):1313–1325,
1995.

Mihai P˘atra¸scu and Mikkel Thorup. Time-space trade-offs for predecessor search. In Proceedings of
the thirty-eighth annual ACM symposium on Theory of computing, pages 232–240, 2006.

Michaël Perrot, Pascal Esser, and Debarghya Ghoshdastidar. Near-optimal comparison based cluster-
ing. Advances in Neural Information Processing Systems, 33:19388–19399, 2020.

William Pugh. Skip lists: a probabilistic alternative to balanced trees. Communications of the ACM,
33(6):668–676, 1990.

Manish Purohit, Zoya Svitkina, and Ravi Kumar. Improving online algorithms via ml predictions.
Advances in Neural Information Processing Systems, 31, 2018.

Robert Rönngren and Rassul Ayani. A comparative study of parallel and sequential priority queue
algorithms. ACM Transactions on Modeling and Computer Simulation (TOMACS), 7(2):157–209,
1997.

Karim Ahmed Abdel Sadek and Marek Elias. Algorithms for caching and MTS with reduced number
of predictions. In The Twelfth International Conference on Learning Representations, 2024. URL
https://openreview.net/forum?id=QuIiLSktO4.

Nihar B Shah and Martin J Wainwright. Simple, robust and optimal ranking from pairwise compar-
isons. Journal of machine learning research, 18(199):1–38, 2018.

Avinash Sharma, Dankan Gowda, Anil Sharma, S Kumaraswamy, MR Arun, et al. Priority queueing
model-based iot middleware for load balancing. In 2022 6th International Conference on Intelligent
Computing and Control Systems (ICICCS), pages 425–430. IEEE, 2022.

Nir Shavit and Itay Lotan. Skiplist-based concurrent priority queues. In Proceedings 14th Inter-
national Parallel and Distributed Processing Symposium. IPDPS 2000, pages 263–268. IEEE,
2000.

14


---Page Break---
Yongho Shin, Changyeol Lee, Gukryeol Lee, and Hyung-Chan An. Improved learning-augmented
algorithms for the multi-option ski rental problem via best-possible competitive analysis. arXiv
preprint arXiv:2302.06832, 2023.

Sandeep Silwal, Sara Ahmadian, Andrew Nystrom, Andrew McCallum, Deepak Ramachandran, and
Seyed Mehran Kazemi. Kwikbucks: Correlation clustering with cheap-weak and expensive-strong
signals. In The Eleventh International Conference on Learning Representations, 2023. URL
https://openreview.net/forum?id=p0JSSa1AuV.

Håkan Sundell and Philippas Tsigas. Fast and lock-free concurrent priority queues for multi-thread
systems. Journal of Parallel and Distributed Computing, 65(5):609–627, 2005.

Mikkel Thorup. Equivalence between priority queues and sorting. Journal of the ACM (JACM), 54
(6):28–es, 2007.

Dominique Tschopp, Suhas Diggavi, Payam Delgosha, and Soheil Mohajer. Randomized algorithms
for comparison-based search. Advances in Neural Information Processing Systems, 24, 2011.

Peter van Emde Boas. Preserving order in a forest in less than logarithmic time and linear space.
Information processing letters, 6(3):80–82, 1977.

Peter van Emde Boas, Robert Kaas, and Erik Zijlstra. Design and implementation of an efficient
priority queue. Mathematical systems theory, 10(1):99–127, 1976.

Jean Vuillemin. A data structure for manipulating priority queues. Communications of the ACM, 21
(4):309–315, 1978.

Fabian Wauthier, Michael Jordan, and Nebojsa Jojic. Efficient ranking from pairwise comparisons.
In International Conference on Machine Learning, pages 109–117. PMLR, 2013.

J. W. J. Williams. Algorithm 232: Heapsort. Communications of the ACM, 7(6):347–348, 1964.

Ali Zeynali, Shahin Kamali, and Mohammad Hajiesmaili. Robust learning-augmented dictionaries.
arXiv preprint arXiv:2402.09687, 2024.

Deli Zhang and Damian Dechev. A lock-free priority queue design based on multi-dimensional
linked lists. IEEE Transactions on Parallel and Distributed Systems, 27(3):613–626, 2015.

15


---Page Break---
Learning-Augmented Priority Queues

Appendix

A
Extended related work

Learning-augmented algorithms
Learning-augmented algorithms, introduced in [Lykouris and
Vassilvtiskii, 2018, Purohit et al., 2018], have captured increasing interest over the last years, as
they allow breaking longstanding limitations in many algorithm design problems. Assuming that
the decision-maker is provided with potentially incorrect predictions regarding unknown parameters
of the problem, learning-augmented algorithms must be capable of leveraging these predictions if
they are accurate (consistency), while keeping the worst-case performance without advice even if
the predictions are arbitrarily bad or adversarial (robustness). Many fundamental online problems
were studied in this setting, such as ski rental [Gollapudi and Panigrahi, 2019, Anand et al., 2020,
Bamas et al., 2020, Diakonikolas et al., 2021, Antoniadis et al., 2021, Maghakian et al., 2023, Shin
et al., 2023], scheduling [Purohit et al., 2018, Merlis et al., 2023, Lassota et al., 2023, Benomar
and Perchet, 2024b], matching [Antoniadis et al., 2020, Dinitz et al., 2021, Chen et al., 2022], and
caching [Lykouris and Vassilvtiskii, 2018, Chlkedowski et al., 2021, Bansal et al., 2022, Antoniadis
et al., 2023b,a, Christianson et al., 2023]. Data structures can also be improved within this framework.
However, this remains underexplored compared to online algorithms. The seminal paper by Kraska
et al. [2018] shows how predictions can be used to optimize space usage. Another study by [Lin
et al., 2022] demonstrates that the runtime of binary search trees can be enhanced by incorporating
predictions of item access frequency. Recent papers have extended this prediction model to other data
structures, such as dictionaries [Zeynali et al., 2024] and skip lists [Fu et al., 2024]. The prediction
models we study deviate from the latter, and are more related to those considered respectively in [Bai
and Coester, 2023] for sorting, and [McCauley et al., 2024] for online list labeling. An overview
of the growing body of work on learning-augmented algorithms (also known as algorithms with
predictions) is maintained at [Lindermayr and Megow, 2022].

Dirty comparisons
The dirty and clean comparison model is also related to the prediction querying
model, gaining a growing interest in the study of learning-augmented algorithms, where the decision-
maker decides when to query predictions and for which items [Im et al., 2022, Benomar and Perchet,
2024a, Sadek and Elias, 2024]. In particular, having free-weak and costly-strong oracles has been
studied in [Silwal et al., 2023] for correlation clustering, in [Bateni et al., 2023] for clustering and
computing minimum spanning trees in a metric space, and in [Eberle et al., 2024] for matroid
optimization. Another related setting involves the algorithm observing partial information online,
which it can then use to decide whether to query a costly hint about the current item. This has been
explored in contexts such as online linear optimization [Bhaskara et al., 2021] and the multicolor
secretary problem [Benomar et al., 2023].

Priority queues
Binary heaps, introduced by Williams [1964], are one of the first efficient im-
plementations of priority queues. They allow FindMin in constant time, and the other operations
in O(log n) time, where n is the number of items in the queue. Shortly after, other heap-based
implementations with similar guarantees were designed, such as leftist heaps [Crane, 1972] and
randomized meldable priority queues [Gambin and Malinowski, 1998]. A new idea was introduced
in [Vuillemin, 1978] with Binomial heaps, where instead of having a single tree storing the items, a
binomial heap is a collection of Θ(log n) trees with exponentially growing sizes, all satisfying the
heap property. They allow insertion in constant amortized time, and O(log n) time for ExtractMin
and DecreaseKey. A breakthrough came with Fibonacci heaps [Fredman and Tarjan, 1987], which
allow all the operations in constant amortized time, except for ExtractMin, which takes O(log n)
time. It was shown later, in works such as [Brodal, 1996, Brodal et al., 2012], that the same guaran-
tees can be achieved in the worst-case, not only in amortized time. Although they offer very good
theoretical guarantees, Fibonacci heaps are known to be slow in practice [Larkin et al., 2014, Lewis,

16


---Page Break---
2023], and other implementations with weaker theoretical guarantees such as binary heaps are often
preferred. We refer the interested reader to the detailed survey on priority queues by Brodal [2013].

Skip lists
A skip list [Pugh, 1990] is a probabilistic data structure, based on classical linked lists,
having shortcut pointers allowing fast access between non-adjacent elements. Due to the simplicity
of their implementation and their strong performance, skip lists have many applications [Hu et al.,
2003, Ge and Zdonik, 2008, Basin et al., 2017]. In particular, they can be used to implement priority
queues [Rönngren and Ayani, 1997], guaranteeing in expectation a constant time for FindMin
and ExtractMin, and O(log n) time for Insert and DecreaseKey. They show particularly good
performance compared to other implementations in the case of concurrent priority queues, where
multiple users or machines can make requests to the priority queue [Shavit and Lotan, 2000, Lindén
and Jonsson, 2013, Zhang and Dechev, 2015].

B
Heap priority queues

B.1
Exponential search and randomized binary search

The insertion algorithm we propose for binary heaps combines the classical exponential search
and our randomized adaptation of the binary search algorithm. We provide a brief overview of the
exponential search and its runtime, followed by an analysis of the runtime upper bound for our
randomized binary search algorithm.

Exponential search.
The exponential search of a target u in a sorted list L = (v1, . . . , vk) starting
from index i ∈[k] consists in first comparing u to vi, and if vi < u then u is compared to
vi+1, vi+2, vi+4, . . . until an integer ℓis found satisfying vi+2ℓ< u ≤vi+2ℓ+1, a binary search is
then conducted between indices i + 2ℓand i + 2ℓ+1 to find the position of u. If the first comparison
shows instead that u < vi, then u is compared to vi−1, vi−2, vi−4, . . .. Denoting by j the true position
of u in L, the exponential search terminates in O(log |i −j|) time.

Randomized binary search (RBS).
Consider a sorted list L = (v1, . . . , vk) ∈Uk and a target
u ∈U. When the search of u in L is reduced to the sub-list (vi, . . . , vj), RBS samples an index

ℓ∼Uniform
 
{i + ⌈j−i

4 ⌉, . . . , j −⌈j−i

4 ⌉}

,

if u = vℓthen algorithm terminates, if u < vℓthen the search is reduced to the sub-list (vi, . . . , vℓ−1),
otherwise it is reduced to (vℓ+1, . . . , vj). This process iterates until the obtained sub-list has a size of
1, the position of u is then immediately deduced by comparing it to the single element in the sub-list.
Lemma B.1. RBS in a list of size k terminates in O(log k) time with probability 1.

Proof. Let us denote by St the size of the sub-list to which the search is reduced after t steps, and
T = min{t ≥1 : St = 1}. For convenience, we consider that St = 1 for all t > T. It holds that
S0 = n, and for all t ≥1, if St = j −i + 1 ≥2 then

St+1 ≤max(j −ℓ, ℓ−i) ≤j −i −⌈j−i

4 ⌉≤3

4(j −i) ≤3

4St .

On the other hand, if St = 1 then St+1 = 1. We deduce that St+1 ≤max(1, 3

4St) with probability
1, hence St ≤max(1, (3/4)tk) for all t ≥1. Therefore, it holds almost surely that

T ≤⌈log4/3 k⌉.

B.2
Proof of Theorem 2.1

Proof. The algorithm described in Theorem 2.1 runs RBS with dirty comparisons to obtain an esti-
mated position br(u, L) of u in L, then it conducts a clean exponential search starting from br(u, L) to
find the exact position r(u, L). Lemma B.1 guarantees that the dirty RBS uses O(log k) comparisons,
while the clean exponential search terminates in expectation after O(log |r(u, L) −br(u, L)|) compar-
isons. Therefore, for demonstrating Theorem 2.1, it suffices to prove that E[log |r(u, L)−br(u, L)|] =
O(log η(u, L)).

17


---Page Break---
To lighten the expressions, we write η instead of η(u, L) in the rest of the proof. Let (i0, j0) = (1, n),
and (it, jt) the indices delimiting the sub-list of L to which the RBS is reduced after t steps for
all t ≥1. Denoting by t∗+ 1 the first step where a dirty comparison is inaccurate, it holds that
br(u, L), r(u, L) ∈{it∗, jt∗}. Indeed, the first t∗dirty comparisons are all accurate, thus the estimated
and the true positions of u in L are both in {it∗, jt∗}. Therefore, |r(u, L) −br(u, L)| ≤St∗−1,
where St = jt −it + 1 is the size of the sub-list delimited by indices (it, jt), which yields

log |r(u, L) −br(u, L)| ≤log(St∗) .
(5)

We focus in the remainder on bounding E[log(St∗)]. For this, we use a proof scheme similar to that
of Lemmas A.4 and A.5 in Bai and Coester [2023].

E[log(St∗)] ≤E[⌈log(St∗)⌉]

≤

∞
X

m≥1
m Pr(⌈log(St∗)⌉= m)

≤log(η + 1) +

∞
X

m=⌈log(η+1)⌉
m Pr(⌈log(St∗)⌉= m) ,
(6)

and for all m ≥1, we have that

Pr(⌈log(St∗)⌉= m) = Pr(St∗∈(2m−1, 2m])

=

∞
X

t=1
Pr(St ∈(2m−1, 2m], t = t∗)

=

∞
X

t=1
Pr(St ∈(2m−1, 2m]) Pr(t = t∗| St ∈(2m−1, 2m]) .

In the sum above, for any t ≥1, the probability that t = t∗is the probability that the next sampled
pivot index ℓt is in the set F(u, L) = {v ∈L : 1(u b< v) ̸= 1(u < v)}, which has a cardinal η. Thus

Pr(t = t∗| St) = Pr(ℓt ∈F(u, L) | St) = #(F ∩{it, . . . , jt})

St
≤η

St
.

It follows that

Pr(⌈log(St∗)⌉= m) ≤
η
2m−1

∞
X

t=1
Pr(St ∈(2m−1, 2m])

≤
η
2m−1 E[#{t ≥1 : St ∈(2m−1, 2m]}]

≤
3η
2m−1 ,

where the last inequality is an immediate consequence of the identity St+1 ≤3

4St+1 proved in
Lemma B.1, which shows in particular that St+3 < St/2, i.e. the after at most 3 steps, the size of the
search sub-list is divided by two, hence (St)t falls into the interval (2m−1, 2m] at most three times.
Substituting into (6) gives

E[log St∗] ≤log(η + 1) + 3η ·

∞
X

m=⌈log(η+1)⌉

m
2m−1

= log(η + 1) + 3η · O
log η

η



= O(log η) .

Therefore, by (5), the expected number of comparisons used during the clean exponential search is at
most O(E[log |r(u, L) −br(u, L)|]) = O(log η), which concludes the proof.

18


---Page Break---
B.3
Proof of Theorem 2.2

Proof. Consider a binary heap Q containing n elements, with positions randomly filled in the leaf
level. Even with this randomization, Q is still balanced, and the length of any path from an empty
position in the leaf level to the root has a size O(log n). Denoting by L the random insertion path
chosen in Algorithm 1, constructing L and storing it requires O(log n) time and space. By Theorem
2.1, the randomized search in Algorithm 1 uses O(log log n) dirty comparisons, and the exponential
search uses an expected number of O(E[log η(u, L)]) clean comparisons. Inserting u and then
swapping it up until its position does not use any comparison and requires a O(log n) time. Therefore,
to prove the theorem, it suffices to demonstrate that E[log η(u, L)] = O(log log η(u, Q)).

We assume that the key u to be inserted can be chosen by an oblivious adversary, unaware of the
randomization outcome. This means that the internal state of Q remains private at any time. Let
F(u, Q) = {v ∈Q : 1(u b< v) ̸= 1(u < v)}, which has a cardinal of η(u, Q). Enumerating the
binary heap levels starting from the root, each level ℓexcept for the last one, denoted ℓmax, contains
exactly 2ℓ−1 elements vℓ
1, . . . , vℓ
2ℓ−1. A key vℓ
i in level ℓhas a probability 1/2ℓ−1 of belonging to L.
Denoting by ξℓ
i = 1(vℓ
i ∈F(u, Q)), the expected number of keys in L belonging to F(u, Q) is

E[η(u, L)] =

ℓmax
X

ℓ=1

2ℓ−1
X

i=1

ξℓ
i
2ℓ−1 =

ℓmax
X

ℓ=1

1
2ℓ−1




2ℓ−1
X

i=1
ξℓ
i



.

Given that Pℓmax
ℓ=1
P2ℓ−1

i=1 ξℓ
i = η(u, Q) ≤2⌈log(η(u,Q)+1)⌉−1, the expression above is maximized
under this constraint for the instance ¯ξℓ
i = 1(ℓ≤⌈log(η(u, Q) + 1)⌉). Therefore,

E[η(u, L)] ≤
X

ℓ

1
2ℓ−1




2ℓ−1
X

i=1
¯ξℓ
i



= ⌈log(η(u, Q) + 1)⌉.

Finally, Jensen’s inequality and the concavity of log yield
E[log η(u, L)] ≤log E[η(u, L)] = O(log log η(u, Q)) ,
which gives the result.

C
Skip lists

C.1
Unique keys in the priority queue

The keys of the priority queue can be considered pairwise distinct by grouping elements with the
same key together in a collection, for example, a HashSet. This collection can be accessible in O(1)
time via the key using a HashMap. When a new item x with priority u is to be inserted, the algorithms
first checks if the key u is already in the priority queue, if that is the case then x is simply added to
the collection corresponding to u in O(1) time. Otherwise, the key u must first be inserted into its
correct position.

With such implementation, when an ExtractMin operation is called, if multiple elements correspond
to the minimum key, then the algorithm can for example retrieve an arbitrary one of them, of the first
inserted one, depending on the use case.

C.2
Expected maximum of i.i.d. geometric random variables

Before presenting the insertion algorithms in the three prediction models, we present an upper bound
from [Eisenberg, 2008] on the expected maximum of i.i.d. geometric random variables with parameter
p. This Lemma will be useful in the analysis of our algorithms, as the heights of elements in the skip
list are i.i.d. geometric random variables. The following Lemma is an immediate consequence of
Lemma C.1 ([Eisenberg, 2008]). If X1, . . . , Xm are i.i.d. random variables following a geometric
random distribution with parameter p, then, denoting by q = 1 −p, it holds that

E[max
i∈[m] Xi] ≤1 +
1
log(1/q)

m
X

k=1

1
k = O(log1/q m) .

19


---Page Break---
C.3
Pointer prediction model

Proof of Theorem 3.1. The key w found at the end of the algorithm is the processor of u in Q.
Inserting u next to w only requires expected O(1) time. Thus, we demonstrate in the following that
Algorithm 2, starting from a key vj ∈Q, finds the predecessor of u in O(log |r(vj)−r(u)|) expected
time, where r(v) denotes the rank r(v, Q) of v in Q. In particular, for vj = [
Pred(u, Q), we obtain
the claim of the theorem.

We assume in the proof that vj < u, i.e. the exponential search goes from left to right. Let h∗(vj, u)
be the maximum height of all elements in Q between u and vj

h∗(vj, u) = max{h(v) : v ∈Q such that vj ≤v ≤u} .

The number of elements between vj and u, with vj included, is |r(u) −r(vj)| + 1, and the heights
of all the elements in Q are independent geometric random variables with parameter p, thus Lemma
C.1 gives that E[h∗(vj, u)] = O(log |r(u) −r(vj)|).

The key w∗found at the end of the Bottom-Up search is the last element, going from vj to u, having
a height of h∗(vj, u). Indeed, in the Bottom-Up search, whenever the algorithm reaches a new key, it
moves to the maximum level to which the key belongs, the height of w∗is therefore necessarily the
maximum height of all the keys between vj and w∗, i.e. h(w∗) = h∗(vj, w∗). Since the Bottom-Up
search stops at key w∗, then Next(w∗, h(w∗)) > u, which means that there is no key in Q between
w∗and u having a height more than h(w∗) −1.

The number of comparisons made in this phase is therefore at most the number of comparisons
needed to reach level h∗(vj, u) + 1 starting from vj using the Bottom-Up search. We consider the
hypothetical setting where the skip list is infinite to the right, the expected number of comparisons to
reach level h∗(vj, u) + 1, in this case, is an upper bound on the expected number of comparisons
needed in the Bottom-Up phase of Algorithm 2, as the algorithm also terminates if the end of the
skip-list is reached. Let T(ℓ) be the expected number of comparisons made in the bottom-up search
to reach level ℓin an infinite skip list. After each comparison made in the bottom-up search, it is
possible to go at least one level up with probability p, while the algorithm can only move horizontally
to the right with probability 1 −p. This induces the inequality

T(ℓ) ≤1 + pT(ℓ−1) + (1 −p)T(ℓ) ,

which yields

T(ℓ) ≤1

p + T(ℓ−1) .

Given that T(1) = 0, we have for ℓ≥1 that T(ℓ) ≤
ℓ−1

p , and we deduce that the expected

number of comparisons made by the algorithm during the Bottom-Up search is at most E[h∗(vj,u)]

p
=
O(log |r(vj) −r(u)|).

In the Top-Down search described in the second phase, the path traversed by the algorithm is exactly
the inverse of the Bottom-Up search from the predecessor of u to w∗. The same arguments as the
analysis of the first phase give that the Top-Down search terminates after O(log |r(vj) −r(u)|)
comparisons in expectation, which concludes the proof.

C.4
Dirty comparison model

Proof of Theorem 3.2. Let F(u, Q) the set of keys in Q whose dirty comparisons with u are inaccu-
rate
F(u, Q) = {v ∈Q : 1(u b< v) ̸= 1(u < v)} .

The prediction error η(u, Q) defined in (2) is the cardinal of F(u, Q). Let

h∗(F(u, Q)) = max{h(v) : v ∈F(u, Q)}

be the maximal height of elements in F(u, Q). The search algorithm Search(Q, u) with dirty
comparisons starts from the highest level at the head of the skip list, and then goes down the different
levels until finding the predicted position ˆw of u. This is the classical search algorithm in skip lists,
and it is known to require O(log n) comparisons to terminate. This can also be deduced from the

20


---Page Break---
analysis of the exponential search described in Algorithm 2, as it corresponds to the Top-Down search
starting from the head of the skip list.

Before level h∗(F(u, Q)) is reached in Search(Q, u), all the dirty comparisons are accurate. De-
noting by v′ the last key in Q visited in a level higher than h∗(F(u, Q)) during this search, and
v′′ = Next(Q, v′, h∗(F(u, Q))), it holds that both keys ˆw and w are between v′ and v′′, and there is
no key in Q between v′ and v′′ with height more than h∗(F(u, Q)) −1.

In particular, the maximal key height between ˆw and w is at most h∗(F(u, Q))−1. We showed in the
proof of Theorem 3.1 that the number of comparisons and runtime of ExpSearchInsertion(Q, vj, u)
is linear with the maximal height of keys in Q that are between vj and u. Using this result with ˆw
instead of vj gives that ExpSearchInsertion(Q, ˆw, u) finds the position of u using O(h∗(F(u, Q)))
clean comparisons. Finally, since h∗(F(u, Q)) is the maximum of a number η(u, Q) of i.i.d. geo-
metric random variables with parameter p, Lemma C.1 gives that

E[h∗(F(u, Q))] = O(log η(u, Q)) ,

which proves the theorem.

C.5
Rank prediction model

To leverage rank predictions, as explained in Section 3.3, we use an auxiliary van Emde Boas (vEB)
tree van Emde Boas [1977]. We describe in the following the structure ad complexity of vEB trees on
[N], and we explain how they can be adapted in the case where N is unknown.

C.5.1
Van Emde Boas (vEB) trees

A vEB tree over an interval {i + 1, . . . , i + m} has a root with √m children, each being the root of a
smaller vEB tree over a sub-interval {i+k√m+1, . . . , i+(k+1)√m} for some k ∈{0, . . . , √m−1}.
The tree leaves are either empty or contain elements with the corresponding key, stored together in a
collection, and internal nodes carry binary information indicating whether or not the subtree they root
contains at least one element. Denoting by H(m) the height of a vEB tree of size m, it holds that
H(m) = 1 + H(√m), which yields that H(m) = O(log log m), enabling efficient implementation
of the operations listed below in O(log log m) time:

• Insert(x, k): insert a new element x with key k ∈[m] in the tree,

• Delete(x, k): Delete the element/key pair (x, k),

• Predecessor(k): return the element in the tree with the largest key smaller than or equal to
k,

• Successor(k): return the element in the tree with the smallest key larger than or equal to k,

• ExtractMin(): removes and returns the element with the smallest key.

Other operations such as FindMin, FindMax, or Lookup(k) are supported in O(1) time. These
runtimes, however, require knowing the maximal key value m from the beginning, as it is used for
constructing Tm.

C.5.2
Dynamic size vEB trees

If the maximal key value ¯R is unknown, we argue that the operations listed above can be supported in
amortized O(log log ¯R) time. Given a vEB tree Tm of size m, if a new key k ∈{m + 1, . . . , 2m} is
to be inserted, we construct an empty vEB tree T2m of size 2m in O(m) time, then repeatedly extract
the elements with minimal key from Tm and insert them in T2m with the same key. Each ExtractMin
operation in Tm and insertion in T2m requires O(log log m) time. Therefore, constructing T2m and
inserting all the elements from Tm takes O(m log log m) time.

This observation can be used to define a vEB with dynamic size. First, we construct a vEB tree with
an initial constant size R0. If at some point the size of the vEB tree is m ≥R0 and a new key k > m
is to be inserted, then we iterate the size doubling process described before until the size of the vEB
tree is at least k. At any time step, denoting by ¯R the maximal key value inserted in the vEB tree, and
letting i ≥1 such that 2i−1R0 ≤¯R < 2iR0, the size of the tree has been doubled up to this step i

21


---Page Break---
times to cover all the keys. The total time for resizing the vEB tree is at most proportional to

i−1
X

j=0
2jR0 log log(2jR0) ≤
 i−1
X

j=0
2j
R0 log log ¯R

≤2iR0 log log ¯R

≤2 ¯R log log ¯R .

Therefore, if N is the total number of elements inserted into the vEB tree and ¯R the maximum key
value, we can neglect the cost of resizing by considering that each insertion requires an amortized
time of O((1 +
¯
R
N ) log log ¯R). The runtime of all the other operations is O(log log ¯R). In particular,
if ¯R = O(N), then all the operations run in O(log log N) amortized time.

C.5.3
Priority queue with rank predictions

Consider the setting where N is unknown and all the predicted ranks, revealed online, satisfy
R(ui) = O(N). The priority queue we consider is a skip list Q with an auxiliary dynamic vEB tree
T . For insertions, we use Algorithm 4. For ExtractMin, we first extract the minimum umin from Q
in O(1) time, then we delete it from the corresponding position bR(umin) in T in O(log log N) time.
Deleting an arbitrary key u from Q can be done in the same way, by removing it from Q in expected
O(1) time and then deleting it from the position bR(u) in T in O(log log N) time. Thus, as in the
other prediction models, DecreaseKey can be implemented by deleting the element and reinserting it
with the new key, which requires the same complexity as insertion, with an additional O(log log N)
term.

We will prove that the priority queue described above yields the claim of Theorem 3.3. We assume
that the keys u1, . . . , uN are inserted in this order. For all t ∈[N], we denote by Qt, T t the set of
keys in the skip list and the set of integer keys in the dynamic vEB tree right after the insertion of ut,
with the keys in Qt or T t. Note that, due to eventual deletions, the sizes of Qt and T t can be smaller
than t.

Following the definition of the rank in (1), for all i, t ∈[N], we denote by r(ui, Qt) the rank of
ui in Qt, and r( bR(ui), T t) the rank of bR(ui) in T t. The following lemma shows that the absolute
difference between the two previous quantities for any given i, t is at most twice the maximal rank
prediction error.

Lemma C.2. For any subset I ⊂[N], it holds for all i ∈I that

|r(ui, {uj}j∈I) −r( bR(ui), { bR(uj)}j∈I)| ≤2 max
j∈[N] η∆(uj) .

Proof. Let

∆∗= max
I⊂[N]


max
i∈I |r(ui, {uj}j∈I) −r( bR(ui), { bR(uj)}j∈I)|

,
(7)

and let I ⊂[N] for which this maximum is reached. We assume without loss of generality that
I = [m]. For all s ∈[N], let

˜Qs = {uj}j∈[s],
˜T s = { bR(uj)}j∈[s],
∆s = max
i∈[s] |r(ui, ˜Qs) −r( bR(ui), ˜T s)| .

To simplify the expressions, we denote by rs
i = r(ui, ˜Qs) and brs
i = r( bR(ui), ˜T s) for all (s, i) ∈
[N]2. We will prove that ∆s = ∆∗for all s ∈{m, . . . , N}. By definition of ∆∗, it holds that
∆∗≥∆s for all s, it remains to prove the other inequality. It is true for s = m by definition of I
and m. Now let s ∈{m, . . . , N −1} and assume that ∆s = ∆∗, i.e. there exists i ≤s such that
|rs
i −brs
i | = ∆∗. Assume for example that brs
i = rs
i + ∆∗.

• If us+1 < ui, then rs+1
s+1 < rs+1
i
and rs+1
i
= rs
i + 1. By definition of ∆∗, it holds that

brs+1
s+1 ≤rs+1
s+1 + ∆∗≤rs+1
i
−1 + ∆∗= rs
i + ∆∗= brs
i .

22


---Page Break---
This implies necessarily that bR(us+1) ≤bR(ui), and therefore

brs+1
i
= brs
i + 1 = rs
i + 1 + ∆∗= rs+1
i
+ ∆∗,

which gives that ∆s+1 ≥|brs+1
i
−rs+1
i
| = ∆∗.

• If us+1 > ui, then rs+1
i
= rs
i and brs+1
i
≥brs
i , thus ∆s+1 ≥brs+1
i
−rs+1
i
≥brs
i −rs
i = ∆∗.

• If us+1 = ui then rs+1
s+1 = rs+1
i
= rs
i + 1. On the other hand, if bR(us+1) ≤bR(ui) then
brs+1
i
= brs
i + 1, otherwise brs+1
s+1 ≥brs
i + 1. In both cases, it holds that

∆s+1 ≥max(brs+1
i
−rs+1
i
, brs+1
s+1 −rs+1
s+1)

≥(brs
i + 1) −(rs
i + 1) = ∆∗.

The same proof can be used for the case where rs
i = brs
i + ∆∗. Therefore, we have for all s ∈
{m, . . . , N} that ∆∗= ∆s. In particular, for s = N, observing that r(ui, ˜QN) = R(ui) for all
i ∈[N], we obtain
∆∗= max
i∈[N] |R(ui) −r( bR(ui), ˜T N)| .
(8)

Let us denote by η∆
max = maxj∈[N] η∆(uj) the maximum rank prediction error. We will prove in the
following that
∀i ∈[N] :
|R(ui) −r( bR(ui), ˜T N)| ≤2η∆
max .
With the assumption that the keys {ui}i∈[N] are pairwise distinct, the ranks (R(ui))i∈[N] form a
permutation of [N].

Given that |R(uk) −bR(uk)| ≤η∆
max for all k ∈[N], it holds for any i, j ∈[N] that

bR(uj) ≤bR(ui) =⇒R(uj) −η∆
max ≤R(ui) + η∆
max
=⇒R(uj) ≤R(ui) + 2η∆
max ,

hence, given that T N = { bR(uj)}j∈[N] and by definition (1) of the rank r( bR(ui), T N), we have

r( bR(ui), T N) = #{j ∈[N] : bR(uj) ≤bR(ui)}

≤#{j ∈[N] : R(uj) ≤R(ui) + 2η∆
max}

= #{k ∈[N] : k ≤R(ui) + 2η∆
max}
(9)

= min(N , R(ui) + 2η∆
max)

≤R(ui) + 2η∆
max ,
(10)

where Equation 9 holds because (R(ui))i∈[N] is a permutation of [N]. Similarly, we have for all
i, j ∈[N] that

R(uj) ≤R(ui) −2η∆
max =⇒R(uj) + η∆
max ≤R(ui) −η∆
max
=⇒bR(uj) ≤bR(ui) ,

and it follows for all i ∈[N] that

r( bR(ui), T N) = #{j ∈[N] : bR(uj) ≤bR(ui)}

≥#{j ∈[N] : R(uj) ≤R(ui) −2η∆
max}

= #{k ∈[N] : k ≤R(ui) −2η∆
max}

= max(0 , R(ui) −2η∆
max)

= R(ui) −2η∆
max .
(11)

From (10) and (11), we deduce that

∀i ∈[N] :
|R(ui) −r( bR(ui), T N)| ≤2η∆
max ,

Combining this with (8) and (7) yields the wanted result.

23


---Page Break---
Proof of Theorem 3.3

Proof. When a new key ui is to be inserted, it is first inserted in the dynamic vEB tree T i with integer
key bR(ui), and its predecessor ˆw in T i is retrieved. These first operations require O(log log N)
time. The position of ui in Qi is then obtained via an exponential search starting from ˆw, which
requires O(log |r(ui, Qi) −r( ˆw, Qi)|) expected time by Theorem 3.1. Finally, inserting u in the
found position takes expected O(1) time.

For any newly inserted element, by accounting for the potential future deletion time via ExtractMin
at the moment of insertion, we can ensure that all ExtractMin operations require constant amortized
time. This approach results in only an additional O(log log N) time for insertions.

Therefore, to prove Theorem 3.3, we only need to show that log |r(ui, Qi) −r( ˆw, Qi)| =
O(log maxj∈[N] η∆(ui)). Since ˆw is the predecessor of ui in T i, it holds that r( bR(ui), T i) ∈
{r( bR( ˆw), T i), r( bR( ˆw, T i) + 1}, the first case occurs if bR(ui) =
bR( ˆw), and the second if
bR(ui) > bR( ˆw). Using this observation and Lemma C.2, it follows that

|r(ui, Qi) −r( ˆw, Qi)|

≤|r(ui, Qi) −r( bR(ui), T i)| + |r( bR(ui), T i) −r( bR( ˆw), T i)| + |r( ˆw, Qi) −r( bR( ˆw), T i)|

≤4 max
j∈[N] η∆(uj) + 1 ,

and it follows that log |r(ui, Qi) −r( ˆw, Qi)| = O(log maxj∈[N] η∆(ui)), which concludes the
proof.

D
Lower bounds

A priority queue can be used for sorting a sequence A = (ai)i∈[n] ∈Un, by first inserting all the
elements in the priority queue, then repeatedly extracting the minimum until the priority queue is
empty. In settings with dirty comparisons or positional predictions, the number of comparisons
required by this sorting algorithm is constrained by the impossibility result demonstrated in Theorem
1.5 of Bai and Coester [2023]. We use this impossibility result to prove the lower bounds stated in
Theorem 3.4.

D.1
Impossibility result for sorting with predictions

We begin in this section by summarizing the setting and the impossibility result demonstrated in
Theorem 1.5 of Bai and Coester [2023] for sorting with predictions.

Positional predictions
In the positional prediction model, the objective is to sort a sequence
A = (ai)i∈[n], given a prediction bRi ∈[n] of Ri = r(ai, A) for all i ∈[n]. This model differs from
our rank prediction model in that the sequence A and the predictions bR = ( bRi)i∈[n] are given offline
to the algorithm. Two different error measures are considered in Bai and Coester [2023], but we
restrict ourselves to the displacement error η∆
i
= |r(ai, A) −bRi|, which is the same as our rank
prediction error 4. In all the following, consider that bR is a fixed permutation of [n], i.e. the predicted
ranks are pairwise distinct. For all ξ ≥1, consider the following set of instances

cand( ˆR, nξ) = {A ∈Un :

n
X

i=1
log(η∆
i + 2) ≤nξ} .

The authors of Bai and Coester [2023] prove that, if 1 ≤ξ ≤O(log n), then no algorithm can sort
every instance from cand( ˆR, nξ) with o(nξ) comparisons. However, in their proof, they demonstrate
a stronger result: no algorithm can sort every instance from In( bR, ξ) with o(nξ) comparisons, where
In( bR, ξ) is the subset of cand( ˆR, nξ) defined by

In( bR, ξ) = {A ∈Un : max
i∈[n] log(η∆
i + 2) ≤ξ} .
(12)

24


---Page Break---
Dirty comparisons
in the dirty comparison model, the authors prove an analogous result by a
reduction to the positional prediction model. More precisely, any permutation bR on [n] defines a
unique dirty order b< on A given by ai b< aj
⇐⇒
bRi < bRi, and maxi∈[n] ηi ≤maxi∈[n] η∆
i ,
where ηi = η(ai, A) = #{j ∈[n] : (ai b< aj) ̸= (ai < aj)}. We deduce that

In( bR, ξ) ⊂{A ∈Un : max
i∈[n] log(ηi + 2) ≤ξ} .
(13)

Hence, given the dirty order
b< , there is no algorithm that can sort every instance with
maxi∈[n] log(ηi + 2) ≤ξ in o(nξ) time.

In the following, we use these lower bounds on sorting to prove our Theorem 3.4.

D.2
Pointer prediction model

For any permutation π of [n] and i ∈[n], we denote by Aπ
i = {aπ(j)}j∈[i]. This first elementary
Lemma uses ideas from the proof of Theorem 1.3 in Bai and Coester [2023].

Lemma D.1. A permutation π of [n] satisfying bRπ(1) ≤. . . ≤bRπ(n) can be constructed in O(n)
time, and it holds for all i ∈{2, . . . , n} that

|r(aπ(i), Aπ
i−1) −r(aπ(i−1), Aπ
i−1)| ≤η∆
π(i−1) + η∆
π(i) + bRπ(i) −bRπ(i−1) .

Proof. The elements of A can be sorted in non-decreasing order of their predicted ranks bRi within
O(n) time using a bucket sort. Hence, we obtain the permutation π. For all i ∈{2, . . . , n},
|r(aπ(i), Aπ
i−1) −r(aπ(i−1), Aπ
i−1)| is the number of elements in Aπ
i−1 whose ranks are between
those of aπ(i) and aπ(i−1). This is at most the number of elements in A whose ranks are between
those of aπ(i) and aπ(i−1), which is |Rπ(i) −Rπ(i−1)|. We deduce that

|r(aπ(i), Aπ
i ) −r(aπ(i−1), Aπ
i )| ≤|Rπ(i) −Rπ(i−1)|

≤|Rπ(i) −bRπ(i)| + |Rπ(i−1) −bRπ(i−1)| + | bRπ(i) −bRπ(i−1)|

= η∆
π(i−1) + η∆
π(i) + bRπ(i) −bRπ(i−1) ,

where we used the triangle inequality and that bRπ(i) ≥bRπ(i−1).

We move now to the proof of our lower bound in the pointer prediction model, by reducing the
problem of sorting with positional predictions to the design of a learning-augmented priority queue
with pointer predictions.

D.2.1
Proof of Theorem 3.4

Proof. Assume that there exists an implementation of a priority queue Q augmented with pointer
predictions, supporting ExtractMin with O(1) comparisons and the insertion of any new key u with
o(log ⃗η(u, Q)) comparisons. This means that, regardless of the history of operations made on Q, the
number of comparisons used by ExtractMin is at most a constant C, and for any ξ ≥1, inserting a
key u such that log(⃗η(u, Q) + 2) ≤ξ requires at most ε(ξ)ξ comparisons, where ε(·) is a positive
function satisfying limξ→∞ε(ξ) = 0. We will show that the existence of such a priority queue
contradicts the impossibility result of sorting with positional predictions.

Let 1 ≤ξ ≤O(log n), bR a fixed permutation of [n], and A an arbitrary instance from the set In( bR, ξ)
defined in (12). Let π be the permutation satisfying the property of Lemma D.1, and consider the
sorting algorithm which inserts the elements of A in Q in the order given by π, then extracts the
minimum repeatedly until Q is emptied. Let us denote by Qi the state of the Q after i insertions.
Upon the insertion of aπ(i), the algorithm uses [
Pred(aπ(i), Qi−1) = aπ(i−1) as a pointer prediction.
By (3), the error of this prediction is

⃗η(aπ(i), Qi−1) = |r(aπ(i), Aπ
i−1) −r(aπ(i−1), Aπ
i−1)| ,

25


---Page Break---
where Aπ
i−1 = {aπ(j)}j<i, and by Lemma D.1, we have that

⃗η(aπ(i), Qi−1) ≤η∆
π(i−1) + η∆
π(i) + bRπ(i) −bRπ(i−1) .
(14)

bR is a permutation of [n], hence bRπ(i) = bRπ(i−1) + 1 by definition of π. Moreover, A ∈In(ξ), thus
max(log(η∆
π(i−1) + 2), log(η∆
π(i) + 2)) ≤ξ, and it follows that

log(⃗η(aπ(i), Qi−1) + 2) ≤log(η∆
π(i−1) + η∆
π(i) + 3)

≤log(η∆
π(i−1) + 2) + log(η∆
π(i) + 2)

≤2ξ ,

where the second inequality is a consequence of log(α + β) ≤log(α) + log(β) for all α, β ≥2.

Therefore, all the insertions into Q require at most (2ξ · ε(2ξ)) comparisons, and the total number of
comparisons T used to sort A is at most

T ≤n (2ξ · ε(2ξ) + C)

= nξ

2 · ε(2ξ) + C

ξ



= o(nξ) .

This means that any instance in In(ξ) can be sorted using o(nξ) comparisons, which contradicts the
lower bound on sorting algorithms augmented with positional predictions.

D.2.2
Rank prediction and dirty comparisons model

In the rank prediction model, the total number of inserted keys is N = n. Denote by ηi = η(ai, A)
for all i ∈[n]. If there is a data structure Q not satisfying the lower bound of Theorem 3.4 for
positional predictions, then we can use it for sorting A. Let 1 ≤ξ ≤O(log n). Similarly to the
proof for pointer predictions, if bR is a permutation of [n], using Q for sorting any instance A ∈In(ξ)
requires at most T = o(nξ) comparisons, which contradicts the lower bound on learning-augmented
sorting algorithms.

The same arguments, combined with (13), give the result also for the dirty comparison model.

E
Applications

E.1
Sorting

In the case where the algorithm is given offline access to the sequence A = (a1, . . . , an) to sort
and to the rank predictions ( bR(ai))i∈[n], we argue that pointer predictions can be used to sort the
sequence within a O(P

i∈[n] log(η∆
i + 2)).

With offline rank predictions, by Lemma D.1, the algorithm can construct a permutation π of [n]
satisfying bR(aπ(1)) ≤. . . ≤bR(aπ(n)), and we showed in the proof of Theorem 3.4, in (14), that
inserting the elements of A into the priority queue in the order given by π, then taking each inserted
element aπ(i) as pointer prediction for the following one aπ(i+1), yields a pointer prediction error of

⃗η(aπ(i), Qi−1) ≤η∆
π(i−1) + η∆
π(i) + bRπ(i) −bRπ(i−1)

for all i ∈{2, . . . , n}. By Theorem 3.1, the runtime for inserting aπ(i) using this pointer prediction
is O(log ⃗η(aπ(i), Qi−1)) . The total time for inserting all the elements into the priority queue is

26


---Page Break---
therefore at most proportional to

n
X

i=2
log(⃗η(aπ(i), Qi−1) + 2) ≤

n
X

i=2
log(η∆
π(i−1) + η∆
π(i) + bRπ(i) −bRπ(i−1) + 2)

≤

n
X

i=2
log((η∆
π(i−1) + 2) + (η∆
π(i) + 2) + ( bRπ(i) −bRπ(i−1) + 2))

≤

n
X

i=2


log(η∆
π(i−1) + 2) + log(η∆
π(i) + 2) + log( bRπ(i) −bRπ(i−1) + 2))


≤2

n
X

i=1
log(η∆
π(i) + 2) +

n
X

i=2
( bRπ(i) −bRπ(i−1)) + n

≤2

n
X

i=1
log(η∆
π(i) + 2) + bRπ(2) + n

= O



X

i∈[n]
log(η∆
i + 2)



.

The third inequality results from inequation log(α+β+γ) ≤log α+log β+log γ for all α, β, γ ≥2,
and the subsequent inequality follows from log(α + 1) ≤α for all α ≥0. Finally, to complete the
sorting algorithm, the minimum is repeatedly extracted from the priority queue until is it empty, and
each ExtractMin only takes O(1) time in the skip list.

E.2
Dijkstra’s algorithms

We give in this section more details on the complexity of Dijkstra’s algorithm using our priority queue
augmented with rank predictions.

Consider any priority queue implementation, having a time complexity TInsert for insertion,
TExtractMin for extracting the minimum, and TDecreaseKey for decreasing the key of an element. There
are two possible implementations of Dijkstra’s algorithm with priority queues. The first one only
uses the operations Insert and ExtractMin, and the maximum number of inserted elements is m + 1,
which yields a complexity of O(mTInsert + mTExtractMin). The second implementation, using also
the DecreaseKey operation, yields a complexity of O(n(TInsert + TExtractMin) + mTDecreaseKey).

Recalling that the number of edges is at most n2, using a binary heap in any of both implementations
gives a total runtime of O(m log n). On the other hand, using a Fibonacci heap in the second
implementation yields a runtime of O(n log n + m).

In the rank predictions model, we consider that the priority queue only uses Insert and ExtractMin
operations, hence we use the first implementation. Denoting by {di}i∈[m] the m distinct keys that are
inserted, these keys are only revealed online to the priority queue. If they are accompanied by predic-
tions ( bR(di))i of their ranks, then each insertion requires O(log log m + log(maxi∈[m+1] |R(di) −
bR(di)|+2)) amortized time and O(log maxi∈[m+1] |R(di)−bR(di)|) comparisons, while extractions
only require a constant amortized time each. The total runtime is therefore

O
 
m log log n + m log max
i∈[m+1] |R(di) −bR(di)|

,

and the total number of comparisons is O(m log(maxi∈[m+1] |R(di) −bR(di)|)).

F
Additional experiments

F.1
Sorting

The problem of sorting with similar prediction models have been studied in Bai and Coester [2023],
hence we numerically compare sorting using our learning-augmented priority queue (LAPQ) with
their sorting algorithms. To run their algorithms, we used the code provided by Bai and Coester

27


---Page Break---
[2023] in their paper. We give here additional experimental results for the class and the decay setting,
for smaller values of n.

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

100

101

(#comparisons)/n

n = 1000

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

100

101

n = 10000

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

100

101

n = 100000

LAPQ ofﬂine predictions
LAPQ online predictions
LAPQ dirty comparisons
Double-Hoover sort
Displacement sort
Binary heap
Fibonacci heap

Figure 6: Sorting with rank predictions in the class setting, for n ∈{1000, 10000, 100000}.

0.0
0.2
0.4
0.6
0.8
1.0
(#timesteps)/(n√n)

2

4

6

8

10

(#comparisons)/n

n = 1000

0.0
0.2
0.4
0.6
0.8
1.0
(#timesteps)/(n√n)

2.5

5.0

7.5

10.0

12.5

n = 10000

0.0
0.2
0.4
0.6
0.8
1.0
(#timesteps)/(n√n)

5

10

15

n = 100000

LAPQ ofﬂine predictions
LAPQ online predictions
LAPQ dirty comparisons
Double-Hoover sort
Displacement sort
Binary heap

Figure 7: Sorting with rank predictions in the decay setting, for n ∈{1000, 10000, 100000}.

F.2
Dijkstra’s algorithm

Considering the same experimental setting presented in Section 5, Figures 8 and 9 show the obtained
results for the cities of Brussels, Paris, New York, and London, which have. The numbers of nodes n
and edges m in each city graph are indicated in the figures.

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

3

4

5

6

7

8

9

10

(#comparisons)/n

Brussels: n = 1380,m = 2580

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

4

6

8

10

12

14

16
Paris: n = 9559,m = 18400

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

6

8

10

12

14

16

18

20

New York: n = 55326,m = 139547

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

5.0

7.5

10.0

12.5

15.0

17.5

20.0

London: n = 128924,m = 300612

LAPQ node rank predictions
LAPQ node dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 8: Dijkstra’s algorithm on city maps with class predictions

0
5
10
15
20
(#timesteps)/n

3

4

5

6

7

8

(#comparisons)/n

Brussels: n = 1380,m = 2580

0
5
10
15
20
(#timesteps)/n

3

4

5

6

7

8

9

10

11

Paris: n = 9559,m = 18400

0
5
10
15
20
(#timesteps)/n

6

8

10

12

14

New York: n = 55326,m = 139547

0
5
10
15
20
(#timesteps)/n

4

6

8

10

12

14

London: n = 128924,m = 300612

LAPQ node rank predictions
LAPQ node dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 9: Dijkstra’s algorithm on city maps with decay predictions

28


---Page Break---
Simulations on Poisson Voronoi Tesselations
We also evaluate the performance of Dijkstra’s
algorithm with our LAPQ on synthetic random graphs. For this, we use Poisson Voronoi Tessellations
(PVT), which are a commonly used random graph model for street systems Gloaguen et al. [2006],
Gloaguen and Cali [2018], Benomar et al. [2022a,b].

PVTs are random graphs created by sampling an integer N from
a Poisson distribution with parameter n ≥1. Subsequently, N
points, termed "seeds," are uniformly chosen at random within a
two-dimensional region I, typically [0, 1]2. A Voronoi diagram is
then generated based on these seeds.
This process results in a planar graph where edges represent the
boundaries between the cells of the Voronoi diagram, and the
nodes are their intersections. For I = [0, 1]2, the expected number
of nodes in this construction is n.
Figure 10 provides a visualization of a PVT with n = 100.
Figure 10: PVT with n = 100

We present the results in the class and the decay settings respectively in Figures 11 and 12. Similar to
previous experiments with Dijkstra on city maps, these figures illustrate how the number of compar-
isons decreases when the LAPQ is augmented with node rank predictions or with the corresponding
dirty comparator. We compare them with the number of comparisons induced by using a binary or
Fibonacci heap, as well as with the number of comparisons of the LAPQ augmented with key rank
predictions.

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

6

8

10

12

(#comparisons)/n

PVT: n = 1000

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

6

8

10

12

14

16

PVT: n = 5000

0.0
0.2
0.4
0.6
0.8
1.0
(#classes)/n

6

8

10

12

14

16

18

20

PVT: n = 10000

LAPQ online predictions
LAPQ dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 11: Dijkstra’s algorithm on Poisson Voronoi Tesselation with class predictions

0
5
10
15
20
(#timesteps)/n

5

6

7

8

9

10

11

(#comparisons)/n

PVT: n = 1000

0
5
10
15
20
(#timesteps)/n

6

7

8

9

10

11

12

13

PVT: n = 5000

0
5
10
15
20
(#timesteps)/n

6

7

8

9

10

11

12

13

14

PVT: n = 10000

LAPQ online predictions
LAPQ dirty comparisons
LAPQ key rank predctions
Binary heap
Fibonacci heap

Figure 12: Dijkstra’s algorithm on Poisson Voronoi Tesselation with decay predictions

The same observations regarding the performance improvement can be made, as in the previous
experiments with city maps. However, in PVT tessellations, the performance of the LAPQ with key
rank predictions surpasses even that of perfect node rank predictions. This is due to the PVTs having
a more uniform structure across space.

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract indicates the scope of the paper: learning-augmented algorithms,
and describes the particular problem we consider
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
Justification: We discuss the difficulties we faced for designing a learning-augmented
binary heap. For learning-augmented skip-lists, we discuss the limitations and possible
improvements in the rank prediction model.
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

30


---Page Break---
Answer: [Yes]
Justification: for each theoretical result, all the needed assumptions are stated. The full
proofs are provided in the appendix.
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
Justification: we give full details on our data structure implementation, and all the details
(instances, libraries) needed for reproducing the results are precised in the paper.
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
5. Open access to data and code

31


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: The full implementation of the learning-augmented priority queue is given in
the supplementary material.

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

Justification: The setting is fully described in the experiments section.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: standard deviations are reported in all the figures.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

32


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
Answer: [Yes]
Justification: Our figures show the number of comparisons used in each experiment. The
computer resources are not relevant.
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
Justification: Our paper respects the Code of Ethics or NeurIPS.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [No]
Justification: This paper presents foundational/theoretical work in the field of learning-
augmented algorithms. We do not feel that any potential societal consequences of our work
must be specifically highlighted.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

33


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
Answer:[NA]
Justification: learning-augmented algorithms, in general, pose no such risks.
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
Justification: We partially use existing open source code, which we clearly cite.
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

34


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not release any new assets.
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
Justification: The paper does not include such experiments.
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
Justification: The paper does not include such experiments.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

35


---Page Break---
