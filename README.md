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

## Trees

## Heaps

## Sorting Algorithms

### Comparison-Based Sorting Algorithms

#### Heap Sort

#### Quick Sort

#### Randomized Quick Sort

### Non-Comparison-Based Sorting Algorithms

#### Counting Sort

#### Radix Sort

#### Selection Sort

## Binary Search Trees

## Red-Black Trees

## Hash Tables

## Dynamic Programming

## Greedy Algorithms
