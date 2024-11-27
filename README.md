# Data Structures and Algorithms Notes

## ECE345 - University of Toronto - Fall 2024

This is a collection of notes for the course ECE345 - Algorithms and Data Structures at the University of Toronto, taken during my exchange semester in Fall 2024. The notes are based on the course material and lectures, and are intended to be a concise summary of the most important concepts and algorithms covered in the course. Feel free to use them for studying or preparing for interviews. If you want to contribute just open a pull request. This course is based on the book "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein.

## Table of Contents

- [Asymptotic Notation](#asymptotic-notation)
- [Proof Methods](#proof-methods)
- [Useful Math](#useful-math)
- [Recurrence](#recurrence)
- [Graphs](#graphs)
- [Trees](#trees)
- [Heaps](#heaps)
- [Sorting Algorithms](#sorting-algorithms)
  - [Comparison-Based Sorting Algorithms](#comparison-based-sorting-algorithms)
  - [Non-Comparison-Based Sorting Algorithms](#non-comparison-based-sorting-algorithms)
- [Binary Search Trees](#binary-search-trees)
- [Red-Black Trees](#red-black-trees)
- [Hash Tables](#hash-tables)
- [Dynamic Programming](#dynamic-programming)
- [Greedy Algorithms](#greedy-algorithms)
- [Amortized Analysis](#amortized-analysis)
- [Splay Trees](#splay-trees)
- [Graph Algorithms](#graph-algorithms)
- [Minimum Spanning Trees](#minimum-spanning-trees)
- [Shortest Paths](#shortest-paths)
- [Maximum Flow](#maximum-flow)
- [History and Turing Machines](#history-and-turing-machines)
- [NP Completeness](#np-completeness)

## Asymptotic Notation

### Big-O Notation

$$
  f(n) = O(g(n)) \text{ if } \exists c > 0, n_0 > 0 \text{ s.t } 0 \leq f(n) \leq c \cdot g(n) \text{ for all } n \geq n_0
$$

- This means that $f(n)$ grows **at most** as fast as $g(n)$ for large $n$.

- Upper bound on the growth rate of a function.

### Big-Omega Notation

$$
  f(n) = \Omega(g(n)) \text{ if } \exists c > 0, n_0 > 0 \text{ s.t } 0 \leq c \cdot g(n) \leq f(n) \text{ for all } n \geq n_0
$$

- This means that $f(n)$ grows **at least** as fast as $g(n)$ for large $n$.

- Lower bound on the growth rate of a function.

### Big-Theta Notation

$$
  f(n) = \Theta(g(n)) \text{ if } f(n) = O(g(n)) \text{ and } f(n) = \Omega(g(n))
$$

- This means that $f(n)$ grows **as fast** as $g(n)$ for large $n$.

- Tight bound on the growth rate of a function.

### Theorem

$$
  f(n) = \Theta(g(n)) \iff f(n) = O(g(n)) \wedge f(n) = \Omega(g(n))
$$

### Polinomially Bounded Functions

$$
  P(n) = O(n^k) \text{ for some constant } k \in \mathbb{R}
$$

- Useful Theorem to prove that a function is polynomially bounded:

$$
  f(n) = O(n^k) \iff lg(f(n)) = O(lg(n))
$$

### Little-O Notation

$$
  f(n) = o(g(n)) \iff f(n) = O(g(n)) \wedge f(n) \neq \Theta(g(n))
$$

### Little-Omega Notation

$$
  f(n) = \omega(g(n)) \iff f(n) = \Omega(g(n)) \wedge f(n) \neq \Theta(g(n))
$$

### Limit Method

- $ \lim\_{n \to \infty} \frac{f(n)}{g(n)} = 0 \implies f(n) = o(g(n))$

- $ \lim\_{n \to \infty} \frac{f(n)}{g(n)} = \infty \implies f(n) = \omega(g(n))$

- $ \lim\_{n \to \infty} \frac{f(n)}{g(n)} = c \text{ ; } 0 < c < \infty \implies f(n) = \Theta(g(n))$

- $ \lim\_{n \to \infty} \frac{f(n)}{g(n)} = c \text{ ; } 0 \leq c < \infty \implies f(n) = O(g(n))$

- $ \lim\_{n \to \infty} \frac{f(n)}{g(n)} = c \text{ ; } 0 < c \leq \infty \implies f(n) = \Omega(g(n))$

### Useful Facts

- $n^a = O(n^b) \iff a \leq b$
- $c^n = O(d^n) \iff c \leq d$
- $log_a(n) = O(log_b(n)) \quad \forall a,b > 1$

### Function Order

From slowest to fastest growing functions:

1. $O(c)$
2. $O(log^*(n))$
3. $O(log^{(i)}(n))$
4. $O(log(n))$
5. $O(n)$
6. $O(n \cdot log(n))$
7. $O(n^{1 + c})$
8. $O(c^n)$
9. $O(d^n)$, $d > c$
10. $O(n!)$
11. $O(n^n)$

## Proof Methods

### Contradiction

Prove $P$:

1. Assume towards a contradiction that $\neg P$.
2. Derive a contradiction. (e.g by showing that $\neg P \implies Q$ and $\neg Q$)
3. Conclude that $P$ is true. ($\neg \neg P \rightarrow P$)

### Induction

Prove $P(n)$ for all $n \geq k$:

1. [IB] Base case: Prove $P(k)$.
2. [IH] Inductive hypothesis: Assume $P(n)$ is true for **some** $n \geq k$ (**for all** $n \geq k$ in strong induction).
3. [IS] Inductive step: Assuming IH, prove $P(n+1)$.

## Useful Math

### Permutations

- Order matters.
- $P(n,k) = \frac{n!}{(n-k)!}$
- Let $q_i$ the number of objects of $t$ classes. The number of ways to arrange $n$ objects is $\frac{n!}{q_1! \cdot q_2! \cdot \ldots \cdot q_t!}$

### Combinations

- Order does not matter.
- $C(n,k) = \binom{n}{k} = \frac{n!}{k! \cdot (n-k)!}$

## Recurrence

- Useful for divide-and-conquer algorithms.

$$
  T(n) = \Sigma_{i=0}^{k} a_i \cdot T(g_i(n)) + f(n)
$$

- $a_i$ is the number of subproblems.
- $g_i(n)$ is the size of the subproblems.
- $f(n)$ is the cost of dividing the problem and combining the solutions.

###### Note: It can happen that $\Sigma_{i=0}^{k} a_i \cdot g_i(n) \neq n$

### Master Theorem

$$
  T(n) = a \cdot T\left(\frac{n}{b}\right) + f(n)
$$

- $a \geq 1$ is the number of subproblems.
- $b > 1$ is the factor by which the problem size is reduced.
- $f(n) > 0$ is the cost of dividing the problem and combining the solutions.

#### Case 1

$$
  f(n) = O(n^c) \text{ for some } c < log_b(a)
$$

$$
  T(n) = \Theta(n^{log_b(a)})
$$

- The cost of the work done at the current level dominates the cost of the work done at the lower levels.

#### Case 2

$$
  f(n) = \Theta(n^c) \text{ for some } c = log_b(a)
$$

$$
  T(n) = \Theta(n^c \cdot log(n))
$$

- The cost of the work done at the current level is equal to the cost of the work done at the lower levels.

#### Case 3

$$
  f(n) = \Omega(n^c) \text{ for some } c > log_b(a)
$$

$$
  T(n) = \Theta(f(n))
$$

- The cost of the work done at the lower levels dominates the cost of the work done at the current level.

- Always check if the regularity condition holds: $a \cdot f\left(\frac{n}{b}\right) \leq k \cdot f(n)$ for some constant $k < 1$ and sufficiently large $n$.

#### Recipe

1. Identify $a$, $b$, and $f(n)$.
2. Compute $log_b(a)$.
3. Compare $f(n)$ with $n^{log_b(a)}$.
4. Apply the corresponding case of the Master Theorem.

### Substitution Method

1. Guess the form of the solution. (e.g $T(n) = O(g(n))$)
2. [IH] Assume that $T(k) \leq c \cdot g(k)$ for all $k < n$.
3. [IS] Prove that $T(n) \leq c \cdot g(n)$.

### Recursion Tree Method

- Longest path: Upper bound.

- Shortest path: Lower bound.

- Total cost of the tree: $O(\text{cost of each level} \cdot h(T))$

- Total cost: $\Sigma_{i=0}^{h(T)} c_i$

## Graphs

$G = (V, E)$

- $V$ is a set of vertices.
- $E$ is a set of edges.
- $E \subseteq V \times V$

### Graph Representation

- Adjacency List:

  - $j \in L[i] \iff (i, j) \in E$.
  - $O(E)$ space. $O(V^2)$ worst case.
  - $O(V)$ time to list all vertices adjacent to a vertex.
  - Sparse graphs.

- Adjacency Matrix:
  - $M[i][j] =$ weight of the edge between $i$ and $j$.
  - $O(V^2)$ space.
  - $O(1)$ time to check if there is an edge between two vertices.
  - Dense graphs.

## Free Trees

A free tree is:

- Connected
- Undirected
- Acyclic

### Equivalent Definitions

1. G is a free tree.
2. Any 2 vertices are connected by a unique simple path.
3. G is connected, but removing any edge disconnects it.
4. G is connected and has $|V| - 1$ edges.
5. G is acyclic and has $|V| - 1$ edges.
6. G is acyclic, but adding any edge creates a cycle.

### Handshaking Lemma

$$
  \Sigma_{v \in V} deg(v) = 2|E|
$$

###### Note: any graph has an even number of vertices with odd degree.

### More Graph Properties

###### Source: Discrete Mathematics IIC1253 - PUC Chile

#### 1. Cycle:

- $V = \{0, 1, 2, ..., n -1\}$
- $E = \{\{i, (i+1) \text{ mod } n\} | 0 \leq i \leq n - 1 \}$
- Last element is connected to the first element.

#### 2. Isomorphism:

- $G_1 = (V_1, E_1)$
- $G_2 = (V_2, E_2)$
- $G_1 \cong G_2$ if there is a bijection $f: V_1 \rightarrow V_2$ such that $\{u, v\} \in E_1 \iff \{f(u), f(v)\} \in E_2$
- $\cong$ is a equivalence relation (reflexive, symmetric, transitive).

#### 3. More Definitions:

- **Degree**: Number of edges incident to a vertex. $deg(v) = |\{u \in V |\{v, u\}\in E\}|$
- **Path**: Sequence of vertices $v_1, v_2, ..., v_k$ such that $\{v_i, v_{i+1}\} \in E$ for $1 \leq i \leq k - 1$.

  - **Simple Path**: Path with no repeated vertices.
  - **Closed**: Path with $v_1 = v_k$. It ends where it starts.
  - **Cycle**: Closed path with no repeated edges.

- **Connected**: There is a path between any pair of vertices.

## Trees

- $T = (V, E)$ is a tree if $\forall x, y \in V$, with $x \neq y$, there is a unique path between $x$ and $y$.

- A tree is a connected acyclic graph.

- A leaf is a vertex with degree 1.

### Theorem:

- If $T$ is a tree and $v$ is a leaf, then $T - v$ is a tree.
- This theorem is useful for induction.

### Binary Trees:

- $\forall v \in V$, $deg(v) \leq 3$.
- if $v$ is the root, $deg(v) \leq 2$.
- $\forall v \in V$, $max(children(v)) \leq 2$

### Theorem:

$$
  \# \text{ of vertices without children} = \# \text{ of vertices with exactly two children} + 1
$$

### Complete Binary Trees:

- $\# \text{ leaves} = 2^h$
- $\# \text{ vertices} = 2^{h + 1} - 1$
- $h \leq log_2(|V|)$

## Heaps

An array $A$ of elements such that:

- A heap is almost a complete binary tree.
- All except last row are full.
- Max-heap: $\forall i$, $A[parent(i)]\geq A[i]$.
- Min-heap: $\forall i$, $A[parent(i)]\leq A[i]$.

### Indexing

Assume index array starts at 1.

- $parent(i) = \lfloor i/2 \rfloor$
- $left(i) = 2i$
- $right(i) = 2i + 1$

### Key Takeaways

- $2^h \leq n \leq 2^{h + 1} - 1 \iff h = \lfloor log(n) \rfloor$
- Max-heap root is the largest element.
- Min-heap root is the smallest element.
- Max-heaps are useful for sorting in decreasing order. Sorting algorithm: Heap Sort.
- Min-heaps are useful for priority queues.

###### Note: Heaps is also referred to as garbage collection storage in PL such as Java and Python, note that this is not the same as the data structure.

## Sorting Algorithms

### Comparison-Based Sorting Algorithms

#### Heap Sort

- Time Complexity: $O(n \cdot log(n))$
- Space Complexity: $O(1)$
- In-place sorting algorithm.
- Not stable: does not preserve the order of equal elements.

```python
def Max-Heapify(A, i):
    """
    Enforces the max-heap property on the subtree rooted at index i. O(h) = O(log(n))
    """
    # Compare the root with the left and right children
    l = 2 * i
    r = 2 * i + 1

    # If A[i] is smaller, then swap with the largest child
    if A[i] < A[l] or A[i] < A[r]:
        largest = l if A[l] > A[r] else r
        A[i], A[largest] = A[largest], A[i]

        # Recursively downwards until the max-heap property is satisfied
        Max-Heapify(A, largest)

def Build-Max-Heap(A, n):
    """
    Builds a max-heap from an unsorted array A. O(h log(n))
    """
    for i in range(floor(n // 2), 1, -1):
        Max-Heapify(A, i)

def Heap-Sort(A, n):
    """
    Sorts an array A in-place using the heap sort algorithm. O(n log(n))
    """
    Build-Max-Heap(A, n)
    for i in range(n, 2, -1):
        A[1], A[i] = A[i], A[1]
        n -= 1
        Max-Heapify(A, 1)
```

#### Quick Sort

- Worst Case Time Complexity: $O(n^2)$. Occurs when the pivot is always the smallest or largest element.
- Best Case: Occurs when the pivot is always the median.
- Average Case Time Complexity: $O(n \cdot log(n))$
- $T(n) = T(an) + T(bn) + O(n)$; $a + b = 1$
- In-place sorting algorithm.
- Not stable: does not preserve the order of equal elements.

```python
def Partition(A, p, r):
    """
    Partitions the array A[p:r] around the pivot A[r]. O(n)
    """
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] < x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1

def Quick-Sort(A, l, r):
    """
    Sorts the array A[l:r] using the quick sort algorithm. O(n log(n))
    """
    if l < r:
        p = Partition(A, l, r)
        Quick-Sort(A, l, p - 1)
        Quick-Sort(A, p - 1, r)
```

- The array is splited in 4: pivot, elements smaller than the pivot, elements greater than the pivot, and unsorted elements.

- In each iteration, examine on elements of unsorted subarray, compare with the pivot and place it in the correct subarray.

#### Randomized Quick Sort

- Randomly select the pivot to avoid worst-case time complexity.

```python
def Randomized-Partition(A, p, r):
    """
    Partitions the array A[p:r] around a random pivot. O(n)
    """
    i = random.randint(p, r)
    A[i], A[r] = A[r], A[i]
    return Partition(A, p, r)
```

#### Theorem:

A comparison-based sorting algorithm requires $\Omega(n \cdot log(n))$ comparisons in the worst case.

### Non-Comparison-Based Sorting Algorithms

#### Counting Sort

- Assumes elements are integers in the range $0$ to $k$.
- Not in-place. Requires additional space.
- Stable: preserves the order of equal elements.
- Time Complexity: $O(n + k)$, if $k = O(n)$ then $T(n) = \Omega(n)$.
- Only works on countable sets.
- Stability property is important for radix sort subroutines.

```python
def Counting-Sort(A, B, k):
    """
    Sorts the array A using the counting sort algorithm. O(n + k)
    """
    C = [0] * (k + 1)
    for j in range(1, len(A)):
        C[A[j]] += 1
        # C[i] now contains the number of elements equal to i
        # e.g C[3] = 2 means that 3 appears twice in A
    for i in range(1, k + 1):
        C[i] += C[i - 1]
        # C[i] now contains the number of elements less than or equal to i
    for j in range(len(A), 0, -1):
        B[C[A[j]]] = A[j]
        C[A[j]] -= 1

    return B
```

#### Radix Sort

- Sorts the elements by their digits.
- Assumes all elements have $\leq d$ digits.
- Each digit is in the range $0$ to $k$.
- Digit Time Complexity: $\Theta(d \cdot (n + k))$.
- Overall Time Complexity: $\Theta(n)$.
- Not in-place. Requires additional space.
- Stable: preserves the order of equal elements.

- Goes from the least significant digit to the most significant digit, running counting sort on each digit iteration.

```python
def Radix-Sort(A):
    """
    Sorts the array A using the radix sort algorithm. O(d(n + k))
    """
    max_element = max(A)

    # Apply counting sort to sort elements based on place value.
    place = 1
    while max_element // place > 0:
        Counting-Sort(A, place)
        place *= 10
```

#### Selection Sort

- TODO

## Binary Search Trees

- If $y$ is in the left subtree of $x$, then $key(y) \leq key(x)$.
- If $y$ is in the right subtree of $x$, then $key(y) \geq key(x)$.

- Given a n-node BST: $h = O(n)$
- If the tree is balanced: $h = O(log(n))$

### Operations

- **Search, Min & Max**: $O(h)$
- **Insertion & Deletion**: $O(h)$
- **Predecessor & Successor**: $O(h)$
- **Building a BST**: $O(n \cdot h)$
  - Worst case: $O(n^2)$

### Algorithms

1. Traversals: $O(n)$

   - **Inorder**:
     - Left, Root, Right
     - Sorting
   - **Preorder**:
     - Root, Left, Right
     - Rotation
   - **Postorder**:
     - Left, Right, Root
     - Deleting

2. Search: $O(h)$

   - **Min**: Always go left
   - **Max**: Always go right
   - **Search**: Compare with the root, go left or right. Divide and conquer.

3. Succesor & Predecessor: $O(h)$

   - **Successor**: Go right, then left until the leftmost node. If the node has a right child, then the successor is the minimum of the right subtree. Else, go up until the node is the left child of its parent.
   - **Predecessor**: Go left, then right until the rightmost node. Symmetric to the successor.

4. Deletion & Insertion: $O(h)$
   - **Deletion**:
     - Case 1: Leaf. Just delete.
     - Case 2: One child. Replace with the child.
     - Case 3: Two children. Replace with the successor.
   - **Insertion**:
     - Search for the node to insert.
     - Insert as a leaf.

## Red-Black Trees

Balanced BST with $O(log(n))$ for most operations.

- Black-height(x) = Number of black nodes on the path from x (exclusive) to the leaves (inclusive).

### Properties

1. Every node is either red or black.
2. Root is black.
3. Leaves are black (NIL nodes).
4. Red nodes have black children.
5. Every path from a node to its leaves has the same number of black nodes (not including the node itself).

### More Properties

- $h(T) \leq 2 \cdot bh(root)$
- $h(T) \leq 2 \cdot lg(n + 1)$
- Subtree rooted at x has at least $2^{bh(x) - 1}$ internal nodes.
- Longest path (root to farthest leaf) is at most twice the shortest path.
- Shortest: All black nodes.
- Longest: Alternating red and black nodes.

### Operations

BST operations can break RBT properties. Fix with rotations.

## Hash Tables

### Definitions

- **Hash Table**: Data structure that implements an associative array abstract data type. Indices $\{0, ..., m-1\}$ are called slots.
- **Hash Function**: Maps keys to indices in the hash table. $h(k) : K \subset U \rightarrow \{0, ..., m-1\}$
- **Collision**: Two keys map to the same index.
- **Load Factor**: $\alpha = \frac{n}{m}$, where $n$ is the number of elements and $m$ is the number of slots.
- **Simple Uniform Hashing**: Each key is equally likely to hash to any slot.

### Key Takeaways

- Hash Table ($O(m)$) uses less memory than array ($O(|U|)$).
  - Example: Storing 10 32-bit numbers in an array uses $2^{32}$ slots, while a hash table uses 10 slots.
- Problem: Collisions.
  - Example: if $|U| > m$, then there are more keys than slots.

### Chaining

- Each slot is a linked list.
- Collisions are resolved by chaining.
- Insert k: append k to the linked list at h(k).
- If $m > n$, then $\alpha = O(1)$.
- With SUH, the expected length of the linked list is $\alpha$.
- Expected time complexity: $O(1 + \alpha) \sim O(n)$.
- If $\alpha = O(1)$, then $O(1)$ time complexity.
- Note that if we want $O(lg(n))$ performance, we can use RBT instead of linked lists.

### Open Addressing

- Slower than Universal Hashing, but less memory.
- Use hash function to find the next available slot.

#### Techniques

1. **Linear Probing**: $h(k, i) = (h(k) + i) \mod m$.

2. **Quadratic Probing**: $h(k, i) = (h(k) + c_1 \cdot i + c_2 \cdot i^2) \mod m$.

   - better performance than linear probing.
   - still m probing sequences.
   - "secondary clustering".

3. **Double Hashing**: $h(k, i) = (h_1(k) + i \cdot h_2(k)) \mod m$.
   - $h_2(k)$ must be relatively prime to m $\iff gcd(h_2(k), m) = 1$, $\forall k$.
   - no clustering.
   - $\Theta(m^2)$ different probing sequences.

## Dynamic Programming

- Epitome of divide-and-conquer.
- Efficient recursion for solving **well-behaved** optimization problems (i.e, optimal solution fiven constraints).
- Concept of **Memoization**: Store the results of expensive function calls and return the cached result when the same inputs occur again.

Used for problems with the following properties:

1. **Optimal Substructure**: Optimal solution can be constructed from optimal solutions of subproblems.
2. **Overlapping Subproblems**: Solving the same subproblem multiple times. Memoization can be used to store the results of subproblems.

- Dynamic Programming is good for tasks with small and repetitive search spaces.

- Examples: Fibonacci, Longest Common Subsequence, Shortest Path, Knapsack, Matrix Chain Multiplication.

###### Methodology in Tutorial 6.

## Greedy Algorithms

- **P1. Greedy Choice Property**: A global optimal solution can be reached by making the first greedy choice.

- **P2. Optimal Substructure**: Optimal solution to the problem contains optimal solutions to subproblems.

- **P3. Smaller Subproblems**: After making a greedy choice, reduce the problem to a smaller instance.

- Approach to solve $O(n^2)$ or $NP$ complete problems.

- Greedy is not always optimal, but provides a good approximation.

###### Proof of correctness template in Tutorial 7.

###### Proof of optimality template in Tutorial 7 Notes.

## Amortized Analysis

"Amortized analysis differs from average-case analysis in that probability is not involved. An amortized analysis guarantees the **average performance** of each operation in the **worst case**." - CLRS.

In simple terms, amortized analysis is used to determine the average time complexity of a sequence of operations.

Amortized Analysis is useful when the worst-case time complexity of a single operation is not representative of the overall performance of the algorithm, and the algortihm is composed by cycles of expensive and cheap operations.

- **Average Cost**: Mean over all possible input costs.
- **Expected Cost**: Expectation over all possible input costs with respect to a probability distribution. Note that if the probability distribution is uniform, then the expected cost is the average cost.
- **Amortized Cost**: Average cost per operation in a sequence of operations in the worst case. No probability involved.

Three techniques covered in Textbook, only 1 and 2 are covered in the course:

1. **Aggregate Analysis**: Determine an upper bound $T(n)$ on the total cost of a sequence of $n$ operations (worst case). The average cost per operation is $T(n)/n$. Then, take the average cost as the amortized cost of each operation, so that all operations have the same amortized cost.

2. **Accounting Method**: Determine an amortized cost of each type of operation. This method overcharges some operations early in the sequence (pre-paid credit) to pay for undercharges later in the sequence.

3. **Potential Method**: Like the accounting method, determines an amortized cost of each operation and overcharges some operations to pay for undercharges. However, the potential method uses a potential function to determine the amortized cost. Mantains the credit as "potential energy" in the data structure as a whole instead of individual operations.

### Aggregate Analysis

1. Given some operations/functions $f_i(x)$ and a sequence $(x_1, x_2, ..., x_n)$
2. Determine $\Sigma_{i=1}^{N} c(f_i(x_i)) = T(n)$ the total running time to execute the entire sequence.
3. The amortized cost per operation is $T(n)/n$.

### Accounting Method

1. Declare that $\$\hat{c}_i$ will be charged for each operation/function.
2. Describe a procedure for how the charge will be allocated.
3. Assert a credit invariant. A claim about the value of the cummulative stored credit $\sum_{i=1}^{N} (\hat{c}_i - c_i)$. Usually of the form "element with property P contains $x$ stored credit ($x$ can be dependent on property P)".

4. Argue or prove that the credit invariant holds.
5. Using the credit invariant, argue why the credit never goes negative.

$$
  \sum_{i=1}^{N} (\hat{c}_i - c_i) \geq 0, \quad \forall n
$$

> $c_i$ is the actual cost of the operation $x_i$

> $\hat{c}_i$ is the amortized cost of the operation $x_i$.

> $\hat{c}_i - c_i$ is the leftover credit after executing the operation $x_i$.

6. The amortized cost of each operation $i$ is $O(\hat{c}_i)$.

**Worst Case Upperbound**:

$$
  \hat{c} \geq c\left(x_i\right), \forall i\text{, then } n \hat{c} \geq \sum_{i=1}^n c\left(x_i\right)=T(n)
$$

**Amortized Upperbound**:

$$
  \frac{1}{n} \sum_{i=1}^n \hat{c} \geq \frac{1}{n} \sum_{i=1}^n c\left(x_i\right)=\frac{T(n)}{n} \Rightarrow \mathcal{O}(\hat{c})=\frac{T(n)}{n}
$$

### Accounting Method Summary:

- Let $D_i$ be the data structure after the $i$-th operation.
- $\hat{c}_i$ is the amortized cost of the $i$-th operation.
- $c_i$ is the actual cost of the $i$-th operation.

1. **Initialization**: $D_0$ is the empty data structure.
2. **Credit Invariant**: $\sum_{i=1}^{N} (\hat{c}_i - c_i) \geq 0 \quad \forall N$. This means, that for each operation, the credit is never negative.
3. **Charge and Credit**: Overcharge some operations to pay for undercharges.
   - if $\hat{c}_i > c_i$, then the operation is overcharged. Save unused credit in $D_i$.
   - if $\hat{c}_i < c_i$, then the operation is undercharged. Use stored credit in $D_i$.
4. **Amortized Cost**: $\mathcal{O}(\hat{c}_i)$.

## Splay Trees

- The primary objective of a Splay Tree is to keep the most frequently accessed elements near the root. This help reduce search times.
- Self-adjusting binary search tree with no balance condition.
- As its not a balanced tree, it does not guarantee $O(log(n))$ time complexity for all operations, some can be $\Theta(n)$. But the amortized time complexity is $O(log(n))$.
- **Splay Operation**: Move a node to the root of the tree by performing a sequence of rotations.
- **Splay Tree**: A binary search tree that uses the splay operation to maintain balance.

Additional facts:

- Most used data structure in practice.
- Fast access to recently accessed elements, example: caches.
- Rotations don't affect the properties of the tree.

```python
def Splay(T, x):
    """
    Splay operation to move node x to the root of the tree T.
    """
    while x is not the root:
        p = parent(x)
        if p is the root:
            rotate(p)
            # now x is at the root
        elif x and p are both right children or left children:
            rotate(parent(p))
            rotate(p)
            # now x is where its grandparent was
        else: # x is a left child and p a right child, or vice versa
            rotate(p)
            rotate(parent(x))
            # now x is where its grandparent was
```

Three types of rotations:

1. **Zig-Zig**
2. **Zig-Zag**
3. **Zig**

When do we splay?

1. **Search**: Splay the node that is being searched for.
2. **Insertion**: Splay the node that was inserted.
3. **Deletion**: Splay the parent of the deleted node.

###### Good resource: https://www.youtube.com/watch?v=wfF4SHkKneg

## Graph Algorithms

### Breadth-First Search

- Discovers all nodes of depth $k$ before discovering nodes of depth $k+1$.
- Only discovers nodes that are reachable from the source node, i.e $\exists$ a path $s \rightarrow t$.
- Time Complexity: $O(V + E)$
- Color Notation:

  - White nodes: undiscovered.
  - Gray nodes: processing.
  - Black nodes: finished.

- Mantain $v.d \lor d[v]$ and $v.\pi \lor \pi[v]$ for each node $v$. ($x[v]$ and $v.x$ different notations for the same thing).

  - $\forall v \in V$, $d[v] = min\{\#\text{ edges in any path from } s \text{ to } v\}$. This is the shortest path from $s$ to $v$.
  - if $v$ is not reachable from $s$, then $d[v] = \infty$.
  - $\pi[v]$ is the predecessor of $v$ in the shortest path from $s$ to $v$.

- BFS can be used to find the shortest path between two nodes in an unweighted graph. SSSP (Single Source Shortest Path).

```python
def BFS (G, s):
"""
Where G is the graph and s is the source node
"""
  let Q be queue
  # Inserting s in queue until all its neighbour vertices are marked.
  Q.enqueue(s)

  mark s as visited.
  while Q is not empty:
    # Removing that vertex from queue, whose neighbour will be visited now
    v = Q.dequeue()

    # processing all the neighbours of v
    for all neighbours w of v in Graph G
      if w is not visited
        # Stores w in Q to further visit its neighbour
        mark w as visited.
        Q.enqueue(w)
```

> source: https://www.hackerearth.com/practice/algorithms/graphs/breadth-first-search/tutorial/

Example:

![BFS1](/images/bfs1.png)
![BFS2](/images/bfs2.png)

### Depth-First Search

- Traverse as far as possible along each branch before backtracking.

- Time Complexity: $O(V + E)$

- Mantains $v.d$, $v.f$, and $v.\pi$ for each node $v$.

  - $d[v]$: Discovery time. When the node was first discovered.
  - $f[v]$: Finish time. When the node was fully explored.
  - $\pi[v]$: Predecessor of $v$ in the DFS tree.
  - Note that $d[u] < f[u]$ by construction in the DFS algorithm.

- Color Notation:

  - White: undiscovered.
  - Gray: processing.
  - Black: finished.

- Edges:

  1. **Tree Edge**: Discovered a new vertex. In the DFS format, i.e $(u,v)$.
  2. **Back Edge**: $(u,v)$ such that $u$ is a descendant of $v$. (includes self-loops).
  3. **Forward Edge**: $(u,v)$ such that $u$ is an ancestor of $v$, but is not a tree edge.
  4. **Cross Edge**: All other edges.

- A back edge exists in a DFS forest $\iff$ the graph has a cycle.
- Undirected graph $\rightarrow$ all edges are either tree or back edges.

Identifying the type of edge when exploring $(u,v)$ in DFS:

- if $v$ is white, then $(u,v)$ is a tree edge.
- if $v$ is gray, then $(u,v)$ is a back edge.
- if $v$ is black:
  - if $d[u] < d[v]$, then $(u,v)$ is a forward edge.
  - if $d[u] > d[v]$, then $(u,v)$ is a cross edge.

Example:

![DFS](/images/dfs.png)

Where:

- Yellow: Tree Edge
- Blue: Back Edge
- Purple: Forward Edge
- Orange: Cross Edge

```python
DFS-iterative (G, s):
"""
Where G is graph and s is source vertex
"""
  let S be stack
  # Inserting s in stack
  S.push(s)
  mark s as visited

  while S is not empty:
    # Pop a vertex from stack to visit next
    v = S.top()
    S.pop()
    # Push all the neighbours of v in stack that are not visited
    for all neighbours w of v in Graph G:
      if w is not visited :
        S.push( w )
        mark w as visited


DFS-recursive(G, s):
  mark s as visited
  for all neighbours w of s in Graph G:
    if w is not visited:
      DFS-recursive(G, w)
```

> source: https://www.hackerearth.com/practice/algorithms/graphs/depth-first-search/tutorial/

### Topological Sort

- Produces a total ordering from a partial ordering.
- Must be a Directed Acyclic Graph (DAG).
- Time Complexity: $O(V + E)$
- To convert an undirected graph to a directed graph, just add an edge $(v,u)$ for each edge $(u,v)$ in the undirected graph, this doubles the number of edges, but the time complexity remains the same.
- Real life application: Time scheduling with dependencies.
- What Topo Sort does is order every node in a graph in such a way that if there is an edge from $A$ to $B$, then $A$ comes before $B$ in the ordering, this is called a topological ordering.

> Note: Tarjan's Algorithm is used to verify if a directed graph contains a cycle.

- Algorithm explanation: Start from any node, then perform DFS with a recursion call stack, when a node has no more children, add it to the topological ordering and backtrack.

```python

def TopologicalSort(G):
  """
  Where G is a directed graph represented as an adjacency list.
  """
  N = G.size() # Number of vertices
  V = [False] * N # Visited array
  ordering = [0] * N
  i = N - 1

  for at in range(0, N):
    if not V[at]:
      visited = []
      DFS(at, V, visited, G)

      for node in visited:
        ordering[i] = node
        i -= 1

  return ordering
```

> Good video: https://www.youtube.com/watch?v=eL-KzMXSXXI

## Minimum Spanning Trees

- Spanning Tree: A subset of Edges that connects all vertices. $T \subseteq E$ such that $\forall v \in V$, $\exists (u,v) \in T$ or $(v,u) \in T$.

- MST: Spanning Tree such that $w(T) = \Sigma_{(u,v) \in T} w(u,v)$ is minimized.

- An MST always has $|V| - 1$ edges, with $|V|$ vertices.

Consider:

- $(u,v) \in T$
- $(x,y) \notin T$

Then:

- $T \cup \{(x,y)\}$ has a cycle.
- $T \setminus \{(u,v)\} \cup \{(x,y)\}$ is a spanning tree.

### Prim's Algorithm

- Time Complexity using Binary Heap: $O(E \log V)$
- Time Complexity using Fibonacci Heap: $O(E + V \log V)$
- How does it work? In simple words, consider $S$ and $\hat{S}$ a partition of nodes, where $S$ are the visited nodes, we start by choosing any node as the root and adding it to $S$, then we add the "cheapest" edge that connects the chosen node with a node in $\hat{S}$, then we add the node to $S$ and repeat the process until all nodes are in $S$.

## Shortest Paths

- Given a graph $G = (V,E)$, a weight function $w: E \rightarrow \mathbb{R}$, and a source node $s \in V$.
- The weight of a path $p = (v_0, v_1, ..., v_k)$ is defined $w(p) = \Sigma_{i=1}^{k} w(v_{i-1}, v_i)$.
- The shortest path weigth is defined by:

$$
  \delta(s,v) = \begin{cases}
    \min_{p \in P(s,v)} w(p) & \text{if } v \text{ is reachable from } s \\
    \infty & \text{otherwise}
  \end{cases}
$$

- Mantain:
  - $v.d$: our estimate of $\delta(s,v)$.
  - $v.\pi$: the predecessor of $v$ in the shortest path from $s$ to $v$.

### Facts:

1. The shortest path problem has **optimal substructure**.
2. Negative weights are allowed as long as there are no negative cycles.
3. The shortest path will never contain a cycle.

### Algorithms:

1. **Relax(u, v, w)**: function and concept of edge relaxation. $O(1)$ time complexity.
2. **Bellman-Ford(G, w, s)**: Works on any graph with no negative cycles. Time Complexity: $O(VE)$.
3. **Dijkstra(G, w, s)**: Works on graphs with positive weights. Time Complexity: $O((V + E) \log V)$ using a binary heap, $O((V\log V) + E)$ using a Fibonacci heap.

### Variants:

- Single-source Shortest-path: given $G = (V, E)$ and **source** $s \in V$, we want to find shortest path from $s$ to $v$, $\forall v \in V$.

- Single-destination Shortest-path: given $G = (V, E)$ and **target** $t \in V$, we want to find shortest path from $u$ to $t$, $\forall u \in V$.

- Single-pair Shortest-path: given $G = (V, E)$ and pair of vertices $u, v \in V$, we want to find shortest path from $u$ to $v$.

- All-pair Shortest-path: given $G = (V, E)$, we want to find shortest path from $u$ to $v$, $\forall u, v \in V$.

> Note: **Single source** is used to solve **all** variants

### Properties: CLRS (22.5)

1. **Triangle Inequality**: $\forall (u,v) \in E$, $\delta(s,v) \leq \delta(s,u) + w(u,v)$.
2. **Upper-bound Property**: $d[v] \geq \delta(s,v)$, $\forall v \in V$. And once $v.d = \delta(s,v)$, it will never change.
3. **No-path Property**: If there is no path from $s$ to $v$, then $v.d = \delta(s,v) = \infty$.
4. **Convergence Property**: If $s \rightarrow u \rightarrow v$, is the shortest path in $G$. If $d[u] = \delta(s,u)$ (prior to relaxing edge $(u,v)$), then $d[v] = \delta(s,v)$ at all times afterwards.
5. **Path-relaxation**: if $p = (v_0, v_1, ..., v_k)$ is the shortest path from $s$ to $v_k$, and paths are relaxed in order $(v_0, v_1), ..., (v_{k-1}, v_k)$, then $d[v_k] = \delta(s,v_k)$.
6. **Predecessor Subgraph Property**: Once $d[v] = \delta(s,v)$, the predecessor subgraph is a shortest-path tree rooted at $s$.

### Path Relaxation:

- Starting from $s$, if there are two paths to $v$ and $w(p_1) > w(p_2) + w(u,v)$, where $p_1$ is the path from $s$ to $v$ and $p_2$ is the path from $s$ to $u$ and $u$ to $v$, we want to take path $p_2$ instead of $p_1$.

```python
def Relax(u, v, w):
  """
  u: source node
  v: destination node
  w: weight of edge (u,v)
  """
  d[v] = min(d[v], d[u] + w(u,v))

def Relax(u, v, w):
  """
  u: source node
  v: destination node
  w: weight of edge (u,v)
  """
  if d[v] > d[u] + w(u,v):
    d[v] = d[u] + w(u,v)
    pi[v] = u
```

### Bellman-Ford Algorithm

- Assume **$G$** contains no negative cycles.
- Time Complexity: $O(VE)$
- Path Relaxation: if $p = (v_0, v_1, ..., v_k)$ is the shortest path from $s$ to $v_k$, and paths are relaxed in order $(v_0, v_1), ..., (v_{k-1}, v_k)$, then $d[v_k] = \delta(s,v_k)$.
- DP algorithm, finds shortest path of length $\leq i$ in each iteration.

```python
def Bellman-Ford(G, w, s):
  """
  G: graph
  w: weight function
  s: source node
  """
  Initialize-Single-Source(G, s)
  for i in range(1, len(G.V) - 1):
    for (u,v) in G.E:
      # Relax all edges V-1 times
      Relax(u, v, w)

  for (u,v) in G.E:
    if d[v] > d[u] + w(u,v):
      # Negative cycle
      return False

  # Shortest Path
  return True
```

- This algorithm can replace Simplex for linear programming problems, i.e solving systems of linear inequalities.

Example:

![Bellman-Ford](/images/bellman-ford.png)

### Dijkstra's Algorithm

- Assume **$G$** contains no negative edges.
- Greedy Algorithm, always chooses the node with the smallest distance.

```python
def Dijkstra(G, w, s):
  """
  G: graph
  w: weight function
  s: source node
  """
  Initialize-Single-Source(G, s)
  # Set of vertices whose first shortest path weights have been determined
  S = []
  Q = []

  for v in G.V:
    # Priority Queue sorted by d[v]
    Insert(Q, v)

  while Q is not empty:
    u = Extract-Min(Q)
    S.append(u)
    for v in G.Adj[u]:
      Relax(u, v, w)
      # Priority Queue sorted by d[v]
      Decrease-Key(Q, v)
```

Example:

![Dijkstra](/images/dijkstra.png)

## Difference Constraints Problems: Bellman-Ford Application

- Given a set of variables $x_1, x_2, ..., x_n$ and a set of constraints of the form $x_j - x_i \leq b_k$, where $b_k$ is a constant.

- To solve a difference constraint problem with $n$ variables:

1. Build a constraint graph (weighted, directed) $G = V, E$.

   - $V = \{v_0, v_1, ..., v_n\}$ one vertex per variable. Define $v_0$ as the _pseudo-source_.
   - $E = \{(v_i, v_j) \mid x_j - x_i \leq b_k\ \text{ is a constraint}\} \cup \{(v_0, v_1), ..., (v_0, v_n)\}$. One edge per constraint. Note that direction is from $x_i$ to $x_j$ and we connect the **pseudo-source** to all other vertices.

2. Assign weights:

   - $w(v_0, v_i) = 0$, for all $i$.
   - $w(v_i, v_j) = b_k$, for all constraints $x_j - x_i \leq b_k$.

3. Theorem:

   - If $G$ has no negative weight cycle, then $x_1 = \delta(v_0, v_1), ..., x_n = \delta(v_0, v_n)$ is a feasible solution.
   - If $G$ has a negative weight cycle, then no feasible solution exists.

4. Build the graph and run Bellman-Ford to find the shortest path from the **pseudo-source** to all other vertices, which gives the solution, or detect a negative cycle.

## Maximum Flow

## History and Turing Machines

## NP-Completeness
