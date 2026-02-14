# Approximate Minimum Degree (AMD) Algorithm

This document describes the AMD algorithm implemented in MySpice for computing fill-reducing orderings of sparse matrices.

## Overview

The Approximate Minimum Degree (AMD) algorithm computes a permutation P such that the LU (or Cholesky) factorization of PAP^T has significantly less fill-in than factoring A directly. Fill-in refers to zero entries that become nonzero during factorization, which increases both memory usage and computation time.

AMD is the de facto standard for fill-reducing orderings in sparse direct solvers, used in SuiteSparse (UMFPACK, CHOLMOD, KLU), MATLAB, and many other systems.

## The Minimum Degree Concept

### Graph Model of Elimination

Gaussian elimination can be modeled as a sequence of graph operations. Given a symmetric matrix A, we construct a graph G where:
- Nodes correspond to matrix rows/columns
- Edges connect nodes i and j if A(i,j) ≠ 0

When we eliminate node k (pivot on row/column k), all nodes adjacent to k become connected to each other (forming a clique). The **degree** of a node is the number of its neighbors. The Minimum Degree algorithm always eliminates the node with the smallest degree, which tends to minimize fill-in.

### Why Minimum Degree Works

Eliminating a node with degree d creates at most d(d-1)/2 new edges (fill-in). By always choosing the minimum degree node:
- We minimize local fill-in at each step
- Nodes that would cause massive fill-in are postponed
- The resulting ordering typically has O(n) to O(n log n) fill-in for circuit matrices

## Algorithm Details

### Quotient Graph Representation

The key innovation in AMD is the **quotient graph** representation. Instead of explicitly maintaining the elimination graph (which can have O(n²) edges due to fill-in), AMD represents it implicitly:

```
Quotient Graph = Variables ∪ Elements

Variables: Nodes not yet eliminated
Elements: Sets representing eliminated nodes
```

A variable i is adjacent to variable j in the elimination graph if:
1. (i,j) is an edge in the original graph, OR
2. Both i and j are adjacent to a common element

This representation requires only O(nnz) space regardless of fill-in.

### Approximate Degrees

Computing exact degrees in the elimination graph is expensive (O(nnz) per elimination). AMD uses **approximate degrees** that are:
- Upper bounds on the true degree
- Cheap to compute and update
- Tight enough to produce good orderings

The approximate degree of variable i is:
```
deg(i) ≈ |direct neighbors| + |neighbors via elements|
```

### Element Absorption

When eliminating multiple nodes, their elements can be **absorbed** (merged) if one element's adjacency is a subset of another's. This keeps the quotient graph compact and speeds up degree computations.

### Algorithm Pseudocode

```
AMD(A, n):
    Input: n×n sparse symmetric matrix A
    Output: Permutation perm[0..n-1]

    1. Initialize quotient graph from A
       - All n nodes are variables
       - No elements exist initially
       - degree[i] = number of neighbors of i

    2. Initialize priority queue with (degree[i], i) for all i

    3. For k = 0 to n-1:
       a. p = extract minimum degree variable from queue
       b. perm[p] = k  (add p to ordering)
       c. Collect Lp = all variables adjacent to p
          (directly or through elements)
       d. Convert p to element containing Lp
       e. Absorb neighboring elements into p
       f. Update approximate degrees of variables in Lp
       g. Re-insert updated variables into priority queue

    4. Return perm
```

## Complexity Analysis

### Time Complexity

- **Worst case**: O(n·m) where m is the number of nonzeros in L+U
- **Typical case**: O(n·nnz) for most practical matrices
- **Circuit matrices**: Nearly O(n·log n) due to their special structure

### Space Complexity

- O(n + nnz) for the quotient graph
- O(n) for priority queue and workspace

### Comparison with Exact Minimum Degree

| Aspect | Exact MD | AMD |
|--------|----------|-----|
| Degree computation | O(nnz) per step | O(neighbors) per step |
| Total time | O(n²·nnz) | O(n·m) |
| Ordering quality | Optimal locally | Near-optimal |
| Practical speed | Slow | Fast |

## Implementation Features

Our AMD implementation includes:

1. **Priority Queue**: Binary heap for O(log n) minimum extraction
2. **Lazy Updates**: Stale heap entries are detected and re-inserted
3. **Marker Array**: O(1) amortized set membership testing
4. **Element Absorption**: Merges redundant elements automatically

## Usage Example

```rust
use sim_core::amd::amd_order;

// CSC format matrix
let ap = vec![0i64, 2, 5, 7];  // Column pointers
let ai = vec![0i64, 1, 0, 1, 2, 1, 2];  // Row indices

let result = amd_order(3, &ap, &ai);

println!("Permutation: {:?}", result.perm);
println!("Elements created: {}", result.stats.elements_created);
```

## Quality Metrics

The quality of an ordering is measured by:

1. **Fill-in count**: Number of nonzeros in L+U minus nnz(A)
2. **Flop count**: Floating-point operations for factorization
3. **Factor time**: Actual factorization time

AMD typically produces orderings within 5-10% of optimal for these metrics.

## Special Matrix Structures

### Arrow Matrices

```
[ x x x x ]
[ x x . . ]
[ x . x . ]
[ x . . x ]
```

Dense row/column should be eliminated last. AMD naturally postpones high-degree nodes.

### Banded Matrices

```
[ x x . . ]
[ x x x . ]
[ . x x x ]
[ . . x x ]
```

Natural order is often good, but AMD may find better orderings by exploiting bandwidth variations.

### Block Diagonal

```
[ B₁  0  0 ]
[ 0  B₂  0 ]
[ 0   0 B₃]
```

AMD processes each block independently (degree 0 between blocks).

## References

### Primary References

1. **Amestoy, P.R., Davis, T.A., Duff, I.S.**
   "An Approximate Minimum Degree Ordering Algorithm"
   SIAM Journal on Matrix Analysis and Applications, Vol. 17, No. 4, pp. 886-905, 1996
   DOI: [10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952)

   *The foundational paper introducing AMD with quotient graphs and approximate degrees.*

2. **Amestoy, P.R., Davis, T.A., Duff, I.S.**
   "Algorithm 837: AMD, An Approximate Minimum Degree Ordering Algorithm"
   ACM Transactions on Mathematical Software, Vol. 30, No. 3, pp. 381-388, 2004
   DOI: [10.1145/1024074.1024081](https://doi.org/10.1145/1024074.1024081)

   *Reference implementation paper with detailed pseudocode and performance analysis.*

### Historical Background

3. **George, A., Liu, J.W.H.**
   "The Evolution of the Minimum Degree Ordering Algorithm"
   SIAM Review, Vol. 31, No. 1, pp. 1-19, 1989
   DOI: [10.1137/1031001](https://doi.org/10.1137/1031001)

   *Comprehensive survey of minimum degree algorithms from 1967 to 1989.*

4. **Tinney, W.F., Walker, J.W.**
   "Direct Solutions of Sparse Network Equations by Optimally Ordered Triangular Factorization"
   Proceedings of the IEEE, Vol. 55, No. 11, pp. 1801-1809, 1967
   DOI: [10.1109/PROC.1967.6011](https://doi.org/10.1109/PROC.1967.6011)

   *Original paper introducing minimum degree for power systems.*

5. **Markowitz, H.M.**
   "The Elimination Form of the Inverse and Its Application to Linear Programming"
   Management Science, Vol. 3, No. 3, pp. 255-269, 1957
   DOI: [10.1287/mnsc.3.3.255](https://doi.org/10.1287/mnsc.3.3.255)

   *Earliest work on fill-reducing orderings (Markowitz criterion).*

### Theoretical Analysis

6. **Heggernes, P., Eisenstat, S.C., Kumfert, G., Pothen, A.**
   "The Computational Complexity of the Minimum Degree Algorithm"
   Proceedings of the 14th Norwegian Computer Science Conference, 2001

   *Shows that exact minimum degree is NP-complete to compute optimally.*

7. **Liu, J.W.H.**
   "Modification of the Minimum-Degree Algorithm by Multiple Elimination"
   ACM Transactions on Mathematical Software, Vol. 11, No. 2, pp. 141-153, 1985
   DOI: [10.1145/214392.214398](https://doi.org/10.1145/214392.214398)

   *Mass elimination and supervariable detection techniques.*

### Modern Implementations

8. **Davis, T.A.**
   "Direct Methods for Sparse Linear Systems"
   SIAM, Philadelphia, 2006
   ISBN: 978-0-898716-13-9

   *Chapter 7 provides comprehensive coverage of fill-reducing orderings including AMD.*

9. **Davis, T.A., Rajamanickam, S., Sid-Lakhdar, W.M.**
   "A Survey of Direct Methods for Sparse Linear Systems"
   Acta Numerica, Vol. 25, pp. 383-566, 2016
   DOI: [10.1017/S0962492916000076](https://doi.org/10.1017/S0962492916000076)

   *Modern survey covering AMD and its role in sparse solvers.*

### Related Orderings

10. **Karypis, G., Kumar, V.**
    "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs"
    SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359-392, 1998
    DOI: [10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997)

    *METIS: alternative nested dissection approach for very large matrices.*

11. **Ashcraft, C., Grimes, R., Lewis, J., Peyton, B., Simon, H.**
    "Progress in Sparse Matrix Methods for Large Linear Systems on Vector Supercomputers"
    International Journal of Supercomputer Applications, Vol. 1, No. 4, pp. 10-30, 1987

    *Comparison of AMD with other orderings on parallel systems.*

## Comparison with Other Orderings

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| AMD | Good | Fast | General sparse matrices |
| COLAMD | Good | Fast | Unsymmetric matrices |
| Nested Dissection | Excellent | Slow | Very large matrices (>100k) |
| Natural Order | Poor | Instant | Already well-ordered matrices |
| Reverse Cuthill-McKee | Moderate | Fast | Banded matrices |

## Integration with SparseLU

The AMD ordering is automatically used by the SparseLU solver during the `analyze` phase:

```rust
let mut solver = SparseLuSolver::new(n);
solver.analyze(&ap, &ai)?;  // AMD computed here
solver.factor(&ap, &ai, &ax)?;
solver.solve(&mut rhs)?;
```

The computed ordering is cached and reused for subsequent factorizations with the same sparsity pattern.
