# Block Triangular Form (BTF) Algorithm

A Comprehensive Guide to BTF Decomposition for Sparse Matrix Factorization

---

## Table of Contents

1. [Introduction](#introduction)
2. [Historical Background](#historical-background)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Algorithm Overview](#algorithm-overview)
5. [Phase 1: Maximum Transversal](#phase-1-maximum-transversal)
6. [Phase 2: Strongly Connected Components](#phase-2-strongly-connected-components)
7. [Phase 3: Permutation Construction](#phase-3-permutation-construction)
8. [Integration with LU Factorization](#integration-with-lu-factorization)
9. [Complexity Analysis](#complexity-analysis)
10. [Implementation Details](#implementation-details)
11. [When BTF is Effective](#when-btf-is-effective)
12. [References](#references)

---

## Introduction

Block Triangular Form (BTF) is a matrix permutation technique that reorders rows and columns of a sparse matrix to reveal its inherent block structure. The resulting form has zeros below the diagonal blocks, which dramatically reduces fill-in during LU factorization and enables block-wise processing.

### The BTF Structure

Given a sparse matrix A, BTF finds permutation matrices P (rows) and Q (columns) such that:

```
            ┌───────┬───────┬───────┬───────┐
            │  B₁₁  │   *   │   *   │   *   │
            ├───────┼───────┼───────┼───────┤
  P·A·Qᵀ =  │   0   │  B₂₂  │   *   │   *   │
            ├───────┼───────┼───────┼───────┤
            │   0   │   0   │  B₃₃  │   *   │
            ├───────┼───────┼───────┼───────┤
            │   0   │   0   │   0   │  B₄₄  │
            └───────┴───────┴───────┴───────┘
```

Where:
- **Diagonal blocks** B₁₁, B₂₂, B₃₃, B₄₄ are square and **irreducible** (strongly connected)
- **Upper blocks** (*) may contain non-zeros
- **Lower blocks** (0) are guaranteed zero — this is the key property!

### Why BTF Matters

The zero structure below the diagonal is **preserved during LU factorization**. This means:

1. **Each diagonal block can be factored independently**
2. **No fill-in occurs in the zero regions**
3. **Smaller blocks = faster factorization** (cubic complexity per block)
4. **Natural parallelism** — blocks can be processed concurrently

---

## Historical Background

### The Dulmage-Mendelsohn Decomposition (1958)

The theoretical foundation of BTF comes from graph theory. In 1958, **A.L. Dulmage** and **N.S. Mendelsohn** published a seminal paper on bipartite graph coverings that established the mathematical basis for matrix decomposition.

> **Key Insight**: A sparse matrix can be viewed as a bipartite graph, where rows and columns are two vertex sets, and non-zero entries are edges. The structure of maximum matchings in this graph reveals the block triangular structure.

**Reference:**
- Dulmage, A.L., Mendelsohn, N.S. "Coverings of bipartite graphs"
  *Canadian Journal of Mathematics*, Vol. 10, pp. 517-534, 1958.

### Tarjan's Strongly Connected Components (1972)

**Robert Tarjan** developed his famous depth-first search algorithm for finding strongly connected components in directed graphs. This algorithm runs in linear time and is fundamental to BTF computation.

> **Key Insight**: After finding a maximum matching, the matrix induces a directed graph. The strongly connected components of this graph correspond exactly to the irreducible diagonal blocks of the BTF.

**Reference:**
- Tarjan, R.E. "Depth-first search and linear graph algorithms"
  *SIAM Journal on Computing*, Vol. 1, No. 2, pp. 146-160, 1972.
  DOI: [10.1137/0201010](https://doi.org/10.1137/0201010)

### Duff's Maximum Transversal Algorithms (1978-1981)

**Iain S. Duff** at Harwell Laboratory developed practical algorithms for computing maximum transversals (matchings) in sparse matrices, making BTF computationally feasible.

**References:**
- Duff, I.S., Reid, J.K. "Algorithm 529: Permutations to block triangular form"
  *ACM Transactions on Mathematical Software*, Vol. 4, No. 2, pp. 189-192, 1978.
  DOI: [10.1145/355780.355785](https://doi.org/10.1145/355780.355785)

- Duff, I.S., Reid, J.K. "An implementation of Tarjan's algorithm for the block triangularization of a matrix"
  *ACM Transactions on Mathematical Software*, Vol. 4, No. 2, pp. 137-147, 1978.
  DOI: [10.1145/355780.355784](https://doi.org/10.1145/355780.355784)

- Duff, I.S. "On algorithms for obtaining a maximum transversal"
  *ACM Transactions on Mathematical Software*, Vol. 7, No. 3, pp. 315-330, 1981.
  DOI: [10.1145/355958.355963](https://doi.org/10.1145/355958.355963)

### Hopcroft-Karp Algorithm (1973)

**John Hopcroft** and **Richard Karp** developed an efficient O(√n × m) algorithm for maximum bipartite matching, improving upon the simple O(n × m) augmenting path approach.

**Reference:**
- Hopcroft, J.E., Karp, R.M. "An n^(5/2) algorithm for maximum matchings in bipartite graphs"
  *SIAM Journal on Computing*, Vol. 2, No. 4, pp. 225-231, 1973.
  DOI: [10.1137/0202019](https://doi.org/10.1137/0202019)

### Pothen-Fan BTF Algorithm (1990)

**Alex Pothen** and **Chin-Ju Fan** published the definitive paper on computing BTF, unifying the theory and providing efficient algorithms based on the Dulmage-Mendelsohn decomposition.

**Reference:**
- Pothen, A., Fan, C.-J. "Computing the block triangular form of a sparse matrix"
  *ACM Transactions on Mathematical Software*, Vol. 16, No. 4, pp. 303-324, 1990.
  DOI: [10.1145/98267.98287](https://doi.org/10.1145/98267.98287)

### KLU: BTF for Circuit Simulation (2010)

**Tim Davis** and **Ekanathan Palamadai Natarajan** developed KLU, a sparse LU solver specifically designed for circuit simulation that makes extensive use of BTF.

> **Key Insight**: Circuit matrices are particularly amenable to BTF because circuits naturally have hierarchical, modular structure. Subcircuits often form independent blocks.

**Reference:**
- Davis, T.A., Palamadai Natarajan, E. "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems"
  *ACM Transactions on Mathematical Software*, Vol. 37, No. 3, Article 36, 2010.
  DOI: [10.1145/1824801.1824814](https://doi.org/10.1145/1824801.1824814)

---

## Mathematical Foundation

### Bipartite Graph Representation

A sparse matrix A ∈ ℝⁿˣⁿ can be represented as a bipartite graph G = (R, C, E):
- **R** = {r₁, r₂, ..., rₙ} — row vertices
- **C** = {c₁, c₂, ..., cₙ} — column vertices
- **E** = {(rᵢ, cⱼ) : A(i,j) ≠ 0} — edges for non-zero entries

### Maximum Matching

A **matching** M ⊆ E is a set of edges with no shared vertices. A **maximum matching** has the largest possible cardinality.

For a square matrix:
- |M| = n means **perfect matching** exists (full structural rank)
- |M| < n means matrix is **structurally singular**

A perfect matching corresponds to a permutation that places non-zeros on the diagonal.

### Directed Graph from Matching

Given a perfect matching M, construct directed graph Gᵈ:
- **Vertices**: V = {1, 2, ..., n} (representing matched row-column pairs)
- **Edges**: (i → j) if A(matched_row[j], i) ≠ 0 and i ≠ j

In other words:
- Matching edges are "contracted" into single vertices
- Non-matching edges become directed edges (column → row direction)

### Strongly Connected Components

A **strongly connected component (SCC)** is a maximal set of vertices where every vertex is reachable from every other vertex.

**Theorem** (Dulmage-Mendelsohn): The SCCs of Gᵈ correspond exactly to the irreducible diagonal blocks of the BTF. Moreover, this decomposition is **unique** regardless of which maximum matching is chosen.

### The Condensation DAG

The SCCs form a **directed acyclic graph (DAG)** called the condensation:
- Each SCC becomes a single vertex
- Edge between SCCs if any edge exists between their vertices

A **topological sort** of this DAG gives the block ordering for BTF.

---

## Algorithm Overview

The BTF algorithm has three phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    BTF Algorithm                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Maximum Transversal (Maximum Matching)             │
│  ─────────────────────────────────────────────               │
│  Find row permutation to maximize diagonal non-zeros         │
│  → Establishes correspondence between rows and columns       │
│  → O(√n × nnz) with Hopcroft-Karp                           │
│  → O(n × nnz) with simple DFS                               │
│                                                              │
│                          ↓                                   │
│                                                              │
│  Phase 2: Strongly Connected Components                      │
│  ──────────────────────────────────────                      │
│  Find SCCs using Tarjan's algorithm                          │
│  → Each SCC = one diagonal block                            │
│  → O(n + nnz) linear time                                   │
│                                                              │
│                          ↓                                   │
│                                                              │
│  Phase 3: Permutation Construction                           │
│  ─────────────────────────────────                           │
│  Build row/column permutations from SCCs                     │
│  → Topological order ensures upper block triangular          │
│  → O(n) time                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Maximum Transversal

### Problem Statement

Given sparse matrix A, find row permutation p such that A(p[i], i) ≠ 0 for as many columns i as possible.

Equivalently: Find a maximum matching in the bipartite graph representation.

### Algorithm: Augmenting Path Method

The classical approach uses **augmenting paths**:

**Definition**: An augmenting path is a path that:
1. Starts at an unmatched row
2. Ends at an unmatched column
3. Alternates between non-matching and matching edges

**Key Theorem** (Berge, 1957): A matching M is maximum if and only if there is no augmenting path with respect to M.

### DFS-Based Algorithm (Duff, 1981)

```
Algorithm: Maximum Transversal via DFS
══════════════════════════════════════

Input:  Sparse matrix A in CSC format (col_ptr, row_idx)
        n = matrix dimension
Output: match[j] = row matched to column j (-1 if unmatched)
        row_match[i] = column matched to row i (-1 if unmatched)

Initialize:
    match[j] = -1 for all j ∈ [0, n)
    row_match[i] = -1 for all i ∈ [0, n)

For each row i = 0 to n-1:
    visited = [false] × n
    if DFS_Augment(i, visited):
        // Found augmenting path, matching increased

Return match, row_match

────────────────────────────────────────

Function DFS_Augment(row i, visited[]) → bool:
    │
    │  // Try each column j where A(i,j) ≠ 0
    │  For each j in columns_of_row(i):
    │      │
    │      │  If visited[j]: continue
    │      │  visited[j] = true
    │      │
    │      │  // Column j is unmatched → found augmenting path!
    │      │  If match[j] == -1:
    │      │      match[j] = i
    │      │      row_match[i] = j
    │      │      return true
    │      │
    │      │  // Column j is matched → try to find alternate path
    │      │  If DFS_Augment(match[j], visited):
    │      │      match[j] = i
    │      │      row_match[i] = j
    │      │      return true
    │  │
    │  return false
```

### Why It Works

The DFS explores all possible augmenting paths from row i:

```
  Row i ──(non-match)──> Col j ──(match)──> Row k ──(non-match)──> Col l ...

  If we reach an unmatched column, we've found an augmenting path.
  We then "flip" all edges along the path:
    - Non-matching edges become matching
    - Matching edges become non-matching
  This increases |M| by 1.
```

### Complexity

- **Per row**: O(nnz) in worst case (visits each edge once)
- **Total**: O(n × nnz)

For circuit matrices with average degree d, this is O(n × n × d) = O(n²d).

### Hopcroft-Karp Improvement

The Hopcroft-Karp algorithm finds **multiple shortest augmenting paths** simultaneously using BFS + DFS:

1. **BFS phase**: Build level graph, find length of shortest augmenting paths
2. **DFS phase**: Find maximal set of vertex-disjoint shortest augmenting paths
3. Repeat until no augmenting paths exist

**Complexity**: O(√n × nnz) — significant improvement for large matrices.

We implement the simpler DFS method first; Hopcroft-Karp can be added as an optimization.

---

## Phase 2: Strongly Connected Components

### Problem Statement

Given the directed graph Gᵈ induced by the matching, find all strongly connected components.

### Tarjan's Algorithm (1972)

Tarjan's algorithm finds all SCCs in a single DFS traversal. It maintains two key values for each vertex:

- **index[v]**: Discovery time (when v was first visited)
- **lowlink[v]**: Smallest index reachable from subtree rooted at v

**Key Insight**: Vertex v is the **root** of an SCC if and only if lowlink[v] == index[v].

```
Algorithm: Tarjan's Strongly Connected Components
═════════════════════════════════════════════════

Input:  Directed graph G = (V, E) with n vertices
Output: List of SCCs in reverse topological order

Global State:
    index_counter = 0
    stack = []                    // DFS stack
    on_stack = [false] × n        // Is vertex on stack?
    index = [-1] × n              // Discovery time (-1 = unvisited)
    lowlink = [-1] × n            // Lowest reachable index
    sccs = []                     // Result: list of SCCs

Main:
    For each vertex v ∈ V:
        If index[v] == -1:        // Not yet visited
            StrongConnect(v)
    Return sccs

────────────────────────────────────────

Function StrongConnect(v):
    │
    │  // Initialize v
    │  index[v] = index_counter
    │  lowlink[v] = index_counter
    │  index_counter += 1
    │  stack.push(v)
    │  on_stack[v] = true
    │
    │  // Explore neighbors
    │  For each edge (v → w):
    │      │
    │      │  If index[w] == -1:           // w not yet visited
    │      │      │  // Tree edge: recurse
    │      │      │  StrongConnect(w)
    │      │      │  lowlink[v] = min(lowlink[v], lowlink[w])
    │      │      │
    │      │  Else if on_stack[w]:         // w is on stack
    │      │      │  // Back edge: w is ancestor of v
    │      │      │  lowlink[v] = min(lowlink[v], index[w])
    │      │      │
    │      │  // Else: cross edge to already-processed SCC, ignore
    │
    │  // Check if v is root of SCC
    │  If lowlink[v] == index[v]:
    │      │  // Pop all vertices in this SCC from stack
    │      │  scc = []
    │      │  Do:
    │      │      w = stack.pop()
    │      │      on_stack[w] = false
    │      │      scc.append(w)
    │      │  While w ≠ v
    │      │  sccs.append(scc)
```

### Understanding lowlink

The lowlink value tracks the lowest-numbered vertex reachable by following:
1. Tree edges (going deeper in DFS)
2. At most one back edge (going to ancestor)

```
Example DFS tree with back edge:

       0 (index=0)
       │
       ▼
       1 (index=1)
       │
       ▼
       2 (index=2) ───back edge───> 0
       │
       ▼
       3 (index=3)

After DFS returns:
  - lowlink[3] = 3 (no back edges from subtree)
  - lowlink[2] = 0 (back edge to vertex 0)
  - lowlink[1] = 0 (inherited from child 2)
  - lowlink[0] = 0 (root)

Since lowlink[0] == index[0], vertex 0 is SCC root.
SCC = {0, 1, 2} (all vertices with index ≥ 0 on stack when 0 processed)
Vertex 3 forms its own SCC (lowlink[3] == index[3]).
```

### Why Reverse Topological Order?

SCCs are output in **reverse topological order** because:
- A vertex is only finalized (becomes SCC root) after all its descendants are processed
- This means SCCs that depend on others are output first

For BTF, we need **forward topological order**, so we simply reverse the list.

### Complexity

- Each vertex is visited exactly once
- Each edge is examined exactly once
- **Total: O(n + nnz)**

---

## Phase 3: Permutation Construction

### From SCCs to Permutations

After Tarjan's algorithm, we have:
- `match[j]` = row matched to column j
- `sccs` = list of SCCs in reverse topological order

To build BTF permutations:

```
Algorithm: Build BTF Permutations
═════════════════════════════════

Input:  match[j] for all columns j
        sccs in reverse topological order
Output: row_perm P, col_perm Q, block_ptr

// Reverse to get topological order (first block first)
sccs = reverse(sccs)

// Build permutations
row_perm = []
col_perm = []
block_ptr = [0]

For each scc in sccs:
    For each column j in scc:
        col_perm.append(j)
        row_perm.append(match[j])
    block_ptr.append(len(col_perm))

Return row_perm, col_perm, block_ptr
```

### Interpreting block_ptr

The `block_ptr` array defines block boundaries:
- Block k spans columns `block_ptr[k]` to `block_ptr[k+1] - 1`
- Block k has size `block_ptr[k+1] - block_ptr[k]`
- Number of blocks = `len(block_ptr) - 1`

```
Example: block_ptr = [0, 2, 5, 8]

Block 0: columns 0-1  (size 2)
Block 1: columns 2-4  (size 3)
Block 2: columns 5-7  (size 3)

Matrix structure:
    Col:  0 1 │ 2 3 4 │ 5 6 7
    ──────────┼───────┼───────
Row 0   █ █ │ * * * │ * * *
Row 1   █ █ │ * * * │ * * *
    ──────────┼───────┼───────
Row 2   0 0 │ █ █ █ │ * * *
Row 3   0 0 │ █ █ █ │ * * *
Row 4   0 0 │ █ █ █ │ * * *
    ──────────┼───────┼───────
Row 5   0 0 │ 0 0 0 │ █ █ █
Row 6   0 0 │ 0 0 0 │ █ █ █
Row 7   0 0 │ 0 0 0 │ █ █ █

█ = diagonal block (may have any pattern)
* = upper block (may be non-zero)
0 = guaranteed zero (key property!)
```

---

## Integration with LU Factorization

### Block-wise Factorization

The key benefit: **factor each diagonal block independently**.

```
Algorithm: BTF-aware LU Factorization
═════════════════════════════════════

Input:  Matrix A, BTF decomposition (P, Q, block_ptr)
Output: L, U factors (block structured)

// Apply permutations to get BTF form
B = P × A × Qᵀ

num_blocks = len(block_ptr) - 1

For k = 0 to num_blocks - 1:
    │
    │  // Extract diagonal block
    │  start = block_ptr[k]
    │  end = block_ptr[k+1]
    │  B_kk = B[start:end, start:end]
    │
    │  // Factor the diagonal block
    │  // (Can use AMD ordering + SparseLU on this block)
    │  L_kk, U_kk = SparseLU_Factor(B_kk)
    │
    │  // Store factors
    │  L[start:end, start:end] = L_kk
    │  U[start:end, start:end] = U_kk
    │
    │  // Note: Off-diagonal blocks U[start:end, end:n]
    │  // are stored but not factored (used in solve phase)
```

### Block-wise Solve

```
Algorithm: BTF-aware Triangular Solve
═════════════════════════════════════

Input:  Block L, U factors, BTF decomposition, RHS b
Output: Solution x

// Apply row permutation
y = P × b

// Forward substitution: solve L × z = y
For k = 0 to num_blocks - 1:
    start = block_ptr[k]
    end = block_ptr[k+1]

    // z_k = L_kk⁻¹ × y_k (forward solve within block)
    z[start:end] = Forward_Solve(L[k], y[start:end])

// Backward substitution: solve U × x = z
For k = num_blocks - 1 down to 0:
    start = block_ptr[k]
    end = block_ptr[k+1]

    // Subtract contributions from later blocks
    For j = k+1 to num_blocks - 1:
        j_start = block_ptr[j]
        j_end = block_ptr[j+1]
        z[start:end] -= U[start:end, j_start:j_end] × x[j_start:j_end]

    // x_k = U_kk⁻¹ × z_k (backward solve within block)
    x[start:end] = Backward_Solve(U[k], z[start:end])

// Apply column permutation
x = Q × x

Return x
```

### Performance Benefits

| Scenario | Without BTF | With BTF |
|----------|-------------|----------|
| Single block (n×n) | O(n³) | O(n³) — no benefit |
| Two blocks (n/2 each) | O(n³) | O(2×(n/2)³) = O(n³/4) — **4× faster** |
| k equal blocks | O(n³) | O(k×(n/k)³) = O(n³/k²) — **k² faster** |
| Triangular (n blocks of 1) | O(n³) | O(n) — **n² faster!** |

---

## Complexity Analysis

### Overall BTF Complexity

| Phase | Time | Space |
|-------|------|-------|
| Maximum Transversal (DFS) | O(n × nnz) | O(n) |
| Maximum Transversal (Hopcroft-Karp) | O(√n × nnz) | O(n + nnz) |
| Tarjan's SCC | O(n + nnz) | O(n) |
| Build Permutations | O(n) | O(n) |
| **Total (DFS)** | **O(n × nnz)** | **O(n + nnz)** |
| **Total (Hopcroft-Karp)** | **O(√n × nnz)** | **O(n + nnz)** |

### Comparison with Direct Factorization

For a circuit with n nodes and nnz non-zeros:

- **BTF computation**: O(n × nnz) or O(√n × nnz)
- **LU factorization without BTF**: O(n × nnz × fill)
- **LU factorization with BTF**: O(Σᵢ bᵢ × nnzᵢ × fillᵢ)

Where bᵢ is the size of block i. Since fill typically scales superlinearly with matrix size, BTF can provide dramatic speedups.

---

## Implementation Details

### Data Structures

```rust
/// Result of BTF decomposition
pub struct BtfDecomposition {
    /// Row permutation: new_row_position[old_row] = new_position
    pub row_perm: Vec<usize>,

    /// Inverse row permutation: old_row[new_position] = old_row_index
    pub row_perm_inv: Vec<usize>,

    /// Column permutation: new_col_position[old_col] = new_position
    pub col_perm: Vec<usize>,

    /// Inverse column permutation
    pub col_perm_inv: Vec<usize>,

    /// Block boundaries: block k spans [block_ptr[k], block_ptr[k+1])
    pub block_ptr: Vec<usize>,

    /// Number of blocks
    pub num_blocks: usize,

    /// Structural rank (perfect matching exists iff structural_rank == n)
    pub structural_rank: usize,

    /// Number of singletons (1×1 blocks, often indicate triangular parts)
    pub num_singletons: usize,
}
```

### CSC Matrix Access Patterns

For efficient implementation, understand CSC (Compressed Sparse Column) format:

```
Matrix A (3×3):          CSC representation:
┌───┬───┬───┐
│ 1 │ 0 │ 2 │            col_ptr = [0, 2, 3, 5]
├───┼───┼───┤            row_idx = [0, 2, 1, 0, 2]
│ 0 │ 3 │ 0 │            values  = [1, 4, 3, 2, 5]
├───┼───┼───┤
│ 4 │ 0 │ 5 │            Column j has entries at row_idx[col_ptr[j]..col_ptr[j+1]]
└───┴───┴───┘
```

To iterate over rows of column j:
```rust
for idx in col_ptr[j]..col_ptr[j+1] {
    let row = row_idx[idx];
    let value = values[idx];
    // Process A(row, j) = value
}
```

---

## When BTF is Effective

### Ideal Cases for BTF

1. **Hierarchical circuits**: Subcircuits with limited interconnection
2. **Sequential logic**: Pipeline stages form natural blocks
3. **Feedback-free paths**: Signal flows in one direction
4. **Large sparse matrices**: BTF overhead amortized over factorization savings

### Less Effective Cases

1. **Densely connected circuits**: Single large SCC
2. **Global feedback**: Everything depends on everything
3. **Small matrices**: BTF overhead not worth it (n < 50)
4. **Dense rows/columns**: Voltage sources create dense structure

### Heuristic: When to Use BTF

```rust
fn should_use_btf(n: usize, nnz: usize) -> bool {
    // BTF overhead is O(n × nnz), benefit depends on block structure
    // Rule of thumb: use BTF for medium-to-large sparse matrices
    n >= 50 && nnz < n * n / 4  // At least 75% sparse
}
```

---

## References

### Foundational Papers

1. **Dulmage-Mendelsohn Decomposition**
   - Dulmage, A.L., Mendelsohn, N.S.
   - "Coverings of bipartite graphs"
   - *Canadian Journal of Mathematics*, Vol. 10, pp. 517-534, 1958
   - *The theoretical foundation for BTF — establishes uniqueness of decomposition*

2. **Tarjan's SCC Algorithm**
   - Tarjan, R.E.
   - "Depth-first search and linear graph algorithms"
   - *SIAM Journal on Computing*, Vol. 1, No. 2, pp. 146-160, 1972
   - DOI: [10.1137/0201010](https://doi.org/10.1137/0201010)
   - *The linear-time algorithm for finding strongly connected components*

3. **Hopcroft-Karp Matching**
   - Hopcroft, J.E., Karp, R.M.
   - "An n^(5/2) algorithm for maximum matchings in bipartite graphs"
   - *SIAM Journal on Computing*, Vol. 2, No. 4, pp. 225-231, 1973
   - DOI: [10.1137/0202019](https://doi.org/10.1137/0202019)
   - *Fast algorithm for maximum bipartite matching*

### BTF Implementation Papers

4. **Duff-Reid BTF Implementation**
   - Duff, I.S., Reid, J.K.
   - "An implementation of Tarjan's algorithm for the block triangularization of a matrix"
   - *ACM Transactions on Mathematical Software*, Vol. 4, No. 2, pp. 137-147, 1978
   - DOI: [10.1145/355780.355784](https://doi.org/10.1145/355780.355784)
   - *First practical implementation of BTF for sparse matrices*

5. **Duff-Reid Permutation Algorithm**
   - Duff, I.S., Reid, J.K.
   - "Algorithm 529: Permutations to block triangular form"
   - *ACM Transactions on Mathematical Software*, Vol. 4, No. 2, pp. 189-192, 1978
   - DOI: [10.1145/355780.355785](https://doi.org/10.1145/355780.355785)
   - *Fortran code for BTF computation*

6. **Maximum Transversal Algorithms**
   - Duff, I.S.
   - "On algorithms for obtaining a maximum transversal"
   - *ACM Transactions on Mathematical Software*, Vol. 7, No. 3, pp. 315-330, 1981
   - DOI: [10.1145/355958.355963](https://doi.org/10.1145/355958.355963)
   - *Comprehensive treatment of matching algorithms for sparse matrices*

7. **Pothen-Fan BTF Algorithm**
   - Pothen, A., Fan, C.-J.
   - "Computing the block triangular form of a sparse matrix"
   - *ACM Transactions on Mathematical Software*, Vol. 16, No. 4, pp. 303-324, 1990
   - DOI: [10.1145/98267.98287](https://doi.org/10.1145/98267.98287)
   - *Definitive paper on BTF — theory, algorithms, and implementation*

### Circuit Simulation Applications

8. **KLU Sparse Solver**
   - Davis, T.A., Palamadai Natarajan, E.
   - "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems"
   - *ACM Transactions on Mathematical Software*, Vol. 37, No. 3, Article 36, 2010
   - DOI: [10.1145/1824801.1824814](https://doi.org/10.1145/1824801.1824814)
   - *State-of-the-art solver using BTF for circuit matrices*

9. **KLU Thesis**
   - Palamadai Natarajan, E.
   - "KLU — A High Performance Sparse Linear Solver for Circuit Simulation Problems"
   - *PhD Thesis*, University of Florida, 2005
   - URL: [https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf)
   - *Detailed treatment of BTF and other techniques for circuit simulation*

### Modern Implementations

10. **Modern Maximum Transversal**
    - Duff, I.S., Kaya, K., Uçar, B.
    - "Design, Implementation, and Analysis of Maximum Transversal Algorithms"
    - *ACM Transactions on Mathematical Software*, Vol. 38, No. 2, Article 13, 2011
    - DOI: [10.1145/2049662.2049663](https://doi.org/10.1145/2049662.2049663)
    - *Modern analysis and implementation of matching algorithms*

11. **SuiteSparse BTF**
    - Davis, T.A.
    - SuiteSparse BTF module
    - URL: [https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/stable/BTF](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/stable/BTF)
    - *Reference C implementation*

### Textbooks

12. **Direct Methods for Sparse Linear Systems**
    - Davis, T.A.
    - *SIAM*, 2006
    - ISBN: 978-0-898716-13-9
    - *Comprehensive textbook covering BTF, LU factorization, and related topics*

13. **Computer Solution of Large Sparse Positive Definite Systems**
    - George, A., Liu, J.W.H.
    - *Prentice-Hall*, 1981
    - ISBN: 978-0131652743
    - *Classic textbook on sparse matrix computation*

### Online Resources

14. **Tarjan's Algorithm Tutorial**
    - GeeksforGeeks
    - URL: [https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/](https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/)

15. **Hopcroft-Karp Tutorial**
    - GeeksforGeeks
    - URL: [https://www.geeksforgeeks.org/hopcroft-karp-algorithm-for-maximum-matching-set-1-introduction/](https://www.geeksforgeeks.org/hopcroft-karp-algorithm-for-maximum-matching-set-1-introduction/)

16. **Brilliant.org Hopcroft-Karp**
    - URL: [https://brilliant.org/wiki/hopcroft-karp/](https://brilliant.org/wiki/hopcroft-karp/)

---

## Appendix: Pseudocode Summary

### Complete BTF Algorithm

```
function BTF_Decompose(A: SparseMatrix) -> BtfDecomposition:
    n = A.num_rows

    // Phase 1: Maximum Transversal
    col_match = [-1] × n      // col_match[j] = row matched to column j
    row_match = [-1] × n      // row_match[i] = column matched to row i

    for i in 0..n:
        visited = [false] × n
        DFS_Augment(A, i, col_match, row_match, visited)

    structural_rank = count(j where col_match[j] ≠ -1)

    if structural_rank < n:
        // Handle structurally singular case
        // (some rows/columns remain unmatched)

    // Phase 2: Build directed graph and find SCCs
    // Edge (i → j) exists if A(col_match[j], i) ≠ 0 and i ≠ j
    adj = build_adjacency_list(A, col_match)
    sccs = Tarjan_SCC(adj, n)  // Returns SCCs in reverse topological order

    // Phase 3: Build permutations
    sccs = reverse(sccs)  // Convert to topological order

    row_perm = []
    col_perm = []
    block_ptr = [0]

    for scc in sccs:
        for col in scc:
            col_perm.append(col)
            row_perm.append(col_match[col])
        block_ptr.append(len(col_perm))

    return BtfDecomposition {
        row_perm, col_perm, block_ptr,
        num_blocks: len(block_ptr) - 1,
        structural_rank
    }
```

---

*Document version: 1.0*
*Last updated: 2024*
*Author: MySpice Development Team*
