//! Approximate Minimum Degree (AMD) Ordering Algorithm
//!
//! This module implements the AMD algorithm for computing fill-reducing orderings
//! of sparse symmetric matrices. AMD is the standard algorithm used in most
//! sparse direct solvers including UMFPACK, CHOLMOD, and KLU.
//!
//! # Overview
//!
//! The Minimum Degree algorithm computes a permutation P such that the Cholesky
//! factorization of PAP^T (or LU factorization) has less fill-in than A itself.
//! At each step, it eliminates the node with the fewest connections (minimum degree).
//!
//! AMD improves upon the basic Minimum Degree algorithm with:
//! - **Quotient graph**: Implicit representation without explicit fill-in edges
//! - **Approximate degrees**: Upper bounds instead of exact degrees (faster updates)
//! - **Element absorption**: Merge redundant elements to reduce graph size
//! - **Mass elimination**: Process multiple minimum-degree nodes efficiently
//!
//! # Algorithm
//!
//! The algorithm maintains a *quotient graph* that implicitly represents the
//! elimination graph without explicitly forming fill-in edges.
//!
//! ```text
//! Input: Sparse matrix A (symmetric pattern)
//! Output: Permutation vector perm[] such that P*A*P^T has low fill-in
//!
//! 1. Initialize quotient graph from A
//! 2. While nodes remain:
//!    a. Find node p with minimum approximate degree
//!    b. Eliminate p:
//!       - Add p to permutation
//!       - Form element e_p from p's neighbors
//!       - Update approximate degrees of affected nodes
//!    c. Absorb elements to keep graph compact
//! 3. Return permutation
//! ```
//!
//! # Quotient Graph Representation
//!
//! Instead of explicitly maintaining the elimination graph (which can have O(n²)
//! edges due to fill-in), AMD uses a quotient graph with:
//! - **Variables**: Nodes not yet eliminated
//! - **Elements**: Sets representing eliminated nodes and induced fill-in
//!
//! A variable i is adjacent to variable j in the elimination graph if either:
//! - (i,j) is an edge in the original graph, or
//! - Both i and j are adjacent to a common element
//!
//! # Approximate Degree Computation
//!
//! The exact degree of a variable is the size of its adjacency set in the
//! elimination graph. Computing this exactly requires O(nnz) work per elimination.
//!
//! AMD uses an upper bound that's cheaper to maintain:
//! ```text
//! approx_degree(i) = |Adj(i) ∩ Variables| + |reach via elements|
//! ```
//!
//! This bound is typically tight and always correct (never underestimates).
//!
//! # Complexity
//!
//! - Time: O(n·m) where m is the number of nonzeros in L+U
//! - Space: O(n + nnz)
//!
//! For typical circuit matrices, this is nearly O(n·log(n)).
//!
//! # References
//!
//! 1. Amestoy, P.R., Davis, T.A., Duff, I.S.
//!    "An Approximate Minimum Degree Ordering Algorithm"
//!    SIAM J. Matrix Anal. Appl., Vol. 17, No. 4, pp. 886-905, 1996
//!    DOI: 10.1137/S0895479894278952
//!
//! 2. Amestoy, P.R., Davis, T.A., Duff, I.S.
//!    "Algorithm 837: AMD, An Approximate Minimum Degree Ordering Algorithm"
//!    ACM Trans. Math. Softw., Vol. 30, No. 3, pp. 381-388, 2004
//!    DOI: 10.1145/1024074.1024081
//!
//! 3. George, A., Liu, J.W.H.
//!    "The Evolution of the Minimum Degree Ordering Algorithm"
//!    SIAM Review, Vol. 31, No. 1, pp. 1-19, 1989
//!    DOI: 10.1137/1031001
//!
//! 4. Davis, T.A.
//!    "Direct Methods for Sparse Linear Systems"
//!    SIAM, Philadelphia, 2006, Chapter 7: Fill-reducing orderings
//!
//! 5. George, A., Liu, J.W.H.
//!    "Computer Solution of Large Sparse Positive Definite Systems"
//!    Prentice-Hall, Englewood Cliffs, NJ, 1981

use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Result of AMD ordering
#[derive(Debug, Clone)]
pub struct AmdResult {
    /// Permutation: perm[old] = new position
    pub perm: Vec<usize>,
    /// Inverse permutation: inv_perm[new] = old position
    pub inv_perm: Vec<usize>,
    /// Statistics from the ordering
    pub stats: AmdStats,
}

/// Statistics from AMD computation
#[derive(Debug, Clone, Default)]
pub struct AmdStats {
    /// Number of nodes in the matrix
    pub n: usize,
    /// Number of nonzeros in the input matrix
    pub nnz: usize,
    /// Number of elements created during elimination
    pub elements_created: usize,
    /// Number of elements absorbed (merged)
    pub elements_absorbed: usize,
}

/// AMD Ordering Algorithm
///
/// Computes a fill-reducing ordering for a sparse symmetric matrix.
///
/// # Arguments
/// * `n` - Matrix dimension
/// * `ap` - Column pointers (CSC format, length n+1)
/// * `ai` - Row indices (CSC format)
///
/// # Returns
/// * `AmdResult` containing the permutation and statistics
///
/// # Example
/// ```ignore
/// use sim_core::amd::amd_order;
///
/// let ap = vec![0i64, 2, 4, 6];
/// let ai = vec![0i64, 1, 0, 1, 1, 2];
///
/// let result = amd_order(3, &ap, &ai);
/// println!("Permutation: {:?}", result.perm);
/// ```
pub fn amd_order(n: usize, ap: &[i64], ai: &[i64]) -> AmdResult {
    if n == 0 {
        return AmdResult {
            perm: vec![],
            inv_perm: vec![],
            stats: AmdStats::default(),
        };
    }

    let mut amd = AmdState::new(n, ap, ai);
    amd.compute_ordering();

    AmdResult {
        perm: amd.perm,
        inv_perm: amd.inv_perm,
        stats: amd.stats,
    }
}

/// Node status in the quotient graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    /// Active variable with given approximate degree
    Variable(usize),
    /// Eliminated node (now an element)
    Element,
}

/// Internal state for AMD computation
struct AmdState {
    /// Matrix dimension
    n: usize,

    /// Permutation: perm[old_node] = new_position
    perm: Vec<usize>,
    /// Inverse permutation: inv_perm[new_position] = old_node
    inv_perm: Vec<usize>,

    /// Status of each node
    status: Vec<NodeStatus>,

    /// Adjacency lists
    /// For variables: list of adjacent variables and elements
    /// For elements: list of variables in the element's reach
    adj: Vec<Vec<usize>>,

    /// Priority queue: (degree, node) pairs with minimum degree first
    heap: BinaryHeap<Reverse<(usize, usize)>>,

    /// Workspace for computing degrees
    marker: Vec<usize>,
    current_mark: usize,

    /// Statistics
    stats: AmdStats,

    /// Current elimination step
    num_eliminated: usize,
}

impl AmdState {
    fn new(n: usize, ap: &[i64], ai: &[i64]) -> Self {
        let nnz = ap[n] as usize;

        // Build symmetric adjacency lists
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for idx in start..end {
                let row = ai[idx] as usize;
                if row < n && row != col {
                    if !adj[col].contains(&row) {
                        adj[col].push(row);
                    }
                    if !adj[row].contains(&col) {
                        adj[row].push(col);
                    }
                }
            }
        }

        // Initialize status and heap
        let mut status = Vec::with_capacity(n);
        let mut heap = BinaryHeap::with_capacity(n);

        for i in 0..n {
            let deg = adj[i].len();
            status.push(NodeStatus::Variable(deg));
            heap.push(Reverse((deg, i)));
        }

        Self {
            n,
            perm: vec![0; n],
            inv_perm: vec![0; n],
            status,
            adj,
            heap,
            marker: vec![0; n],
            current_mark: 0,
            stats: AmdStats {
                n,
                nnz,
                ..Default::default()
            },
            num_eliminated: 0,
        }
    }

    /// Main AMD computation loop
    fn compute_ordering(&mut self) {
        while self.num_eliminated < self.n {
            // Find minimum degree variable
            let pivot = self.find_min_degree_variable();

            match pivot {
                Some(p) => self.eliminate(p),
                None => break,
            }
        }
    }

    /// Find the variable with minimum approximate degree
    fn find_min_degree_variable(&mut self) -> Option<usize> {
        loop {
            let entry = self.heap.pop()?;
            let Reverse((deg, node)) = entry;

            // Skip if already eliminated
            if let NodeStatus::Element = self.status[node] {
                continue;
            }

            // Check if degree is stale
            if let NodeStatus::Variable(current_deg) = self.status[node] {
                if deg != current_deg {
                    // Re-insert with correct degree
                    self.heap.push(Reverse((current_deg, node)));
                    continue;
                }
                return Some(node);
            }
        }
    }

    /// Eliminate a variable and update the quotient graph
    fn eliminate(&mut self, p: usize) {
        // Add to permutation
        self.perm[p] = self.num_eliminated;
        self.inv_perm[self.num_eliminated] = p;
        self.num_eliminated += 1;

        // Collect neighbors of p (both variables and elements)
        self.current_mark += 1;
        let mark = self.current_mark;

        let mut neighbor_vars: Vec<usize> = Vec::new();
        let mut neighbor_elems: Vec<usize> = Vec::new();

        // Direct neighbors of p
        for &adj in &self.adj[p] {
            match self.status[adj] {
                NodeStatus::Variable(_) => {
                    if self.marker[adj] != mark {
                        self.marker[adj] = mark;
                        neighbor_vars.push(adj);
                    }
                }
                NodeStatus::Element => {
                    neighbor_elems.push(adj);
                }
            }
        }

        // Variables reachable through elements adjacent to p
        for &elem in &neighbor_elems {
            for &adj in &self.adj[elem] {
                if let NodeStatus::Variable(_) = self.status[adj] {
                    if adj != p && self.marker[adj] != mark {
                        self.marker[adj] = mark;
                        neighbor_vars.push(adj);
                    }
                }
            }
        }

        // p becomes an element containing neighbor_vars
        self.status[p] = NodeStatus::Element;
        self.adj[p] = neighbor_vars.clone();
        self.stats.elements_created += 1;

        // Absorb neighboring elements into p
        for &elem in &neighbor_elems {
            self.absorb_element(elem, p);
        }

        // Update degrees of affected variables
        for &var in &neighbor_vars {
            self.update_degree(var, p);
        }
    }

    /// Absorb element src into element dst
    fn absorb_element(&mut self, src: usize, dst: usize) {
        if src == dst {
            return;
        }

        // Variables adjacent to src are now adjacent to dst
        let src_adj = std::mem::take(&mut self.adj[src]);

        for var in src_adj {
            if let NodeStatus::Variable(_) = self.status[var] {
                // Add var to dst if not already there
                if !self.adj[dst].contains(&var) {
                    self.adj[dst].push(var);
                }

                // Update var's adjacency: replace src with dst
                if let Some(pos) = self.adj[var].iter().position(|&x| x == src) {
                    self.adj[var][pos] = dst;
                }
            }
        }

        self.stats.elements_absorbed += 1;
    }

    /// Update the approximate degree of a variable
    fn update_degree(&mut self, var: usize, _new_element: usize) {
        if let NodeStatus::Element = self.status[var] {
            return;
        }

        // Compute approximate degree by counting unique reachable variables
        self.current_mark += 1;
        let mark = self.current_mark;
        self.marker[var] = mark;

        let mut degree = 0;

        for &adj in &self.adj[var] {
            match self.status[adj] {
                NodeStatus::Variable(_) => {
                    if self.marker[adj] != mark {
                        self.marker[adj] = mark;
                        degree += 1;
                    }
                }
                NodeStatus::Element => {
                    // Count variables reachable through this element
                    for &elem_adj in &self.adj[adj] {
                        if let NodeStatus::Variable(_) = self.status[elem_adj] {
                            if self.marker[elem_adj] != mark {
                                self.marker[elem_adj] = mark;
                                degree += 1;
                            }
                        }
                    }
                }
            }
        }

        // Update status and re-insert into heap
        self.status[var] = NodeStatus::Variable(degree);
        self.heap.push(Reverse((degree, var)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_permutation(result: &AmdResult, n: usize) {
        assert_eq!(result.perm.len(), n, "perm length mismatch");
        assert_eq!(result.inv_perm.len(), n, "inv_perm length mismatch");

        // Check that perm contains all values 0..n
        let mut seen = vec![false; n];
        for &p in &result.perm {
            assert!(p < n, "perm value {} out of range", p);
            seen[p] = true;
        }
        assert!(seen.iter().all(|&x| x), "perm missing some values");

        // Check inverse relationship
        for i in 0..n {
            assert_eq!(result.perm[result.inv_perm[i]], i,
                "inverse relationship broken at position {}", i);
        }
    }

    #[test]
    fn test_amd_diagonal() {
        // Diagonal matrix - any order is optimal
        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];

        let result = amd_order(3, &ap, &ai);
        verify_permutation(&result, 3);
    }

    #[test]
    fn test_amd_tridiagonal() {
        // Tridiagonal matrix
        let ap = vec![0i64, 2, 5, 7];
        let ai = vec![0i64, 1, 0, 1, 2, 1, 2];

        let result = amd_order(3, &ap, &ai);
        verify_permutation(&result, 3);
    }

    #[test]
    fn test_amd_star() {
        // Star graph: node 0 connected to all others
        let ap = vec![0i64, 4, 6, 8, 10];
        let ai = vec![
            0i64, 1, 2, 3,  // col 0
            0, 1,            // col 1
            0, 2,            // col 2
            0, 3,            // col 3
        ];

        let result = amd_order(4, &ap, &ai);
        verify_permutation(&result, 4);

        // Leaves should come before center (they have lower degree)
        // perm[leaf] < perm[center]
        assert!(result.perm[1] < result.perm[0] ||
                result.perm[2] < result.perm[0] ||
                result.perm[3] < result.perm[0],
                "At least one leaf should be eliminated before center");
    }

    #[test]
    fn test_amd_empty() {
        let ap = vec![0i64];
        let ai: Vec<i64> = vec![];

        let result = amd_order(0, &ap, &ai);

        assert!(result.perm.is_empty());
        assert!(result.inv_perm.is_empty());
    }

    #[test]
    fn test_amd_single_node() {
        let ap = vec![0i64, 1];
        let ai = vec![0i64];

        let result = amd_order(1, &ap, &ai);
        verify_permutation(&result, 1);
    }

    #[test]
    fn test_amd_dense_2x2() {
        // Dense 2x2 matrix
        let ap = vec![0i64, 2, 4];
        let ai = vec![0i64, 1, 0, 1];

        let result = amd_order(2, &ap, &ai);
        verify_permutation(&result, 2);
    }

    #[test]
    fn test_amd_chain() {
        // Chain: 0-1-2-3-4
        let n = 5;
        let mut ap = vec![0i64];
        let mut ai = Vec::new();

        for i in 0..n {
            if i > 0 {
                ai.push((i - 1) as i64);
            }
            ai.push(i as i64);
            if i < n - 1 {
                ai.push((i + 1) as i64);
            }
            ap.push(ai.len() as i64);
        }

        let result = amd_order(n, &ap, &ai);
        verify_permutation(&result, n);

        // Endpoints have degree 1, should be eliminated first
        assert!(result.perm[0] < result.perm[2] || result.perm[4] < result.perm[2],
                "Endpoint should be eliminated before middle");
    }

    #[test]
    fn test_amd_ladder() {
        // Ladder network
        // 0-2-4
        // | | |
        // 1-3-5
        let ap = vec![0i64, 2, 4, 7, 10, 12, 14];
        let ai = vec![
            0i64, 1,      // col 0: connected to 1
            0, 1, 3,      // col 1: connected to 0, 3
            0, 2, 3, 4,   // col 2: connected to 0, 3, 4
            1, 2, 3, 5,   // col 3: connected to 1, 2, 5
            2, 4, 5,      // col 4: connected to 2, 5
            3, 4, 5,      // col 5: connected to 3, 4
        ];

        let result = amd_order(6, &ap, &ai);
        verify_permutation(&result, 6);
    }

    #[test]
    fn test_amd_statistics() {
        let ap = vec![0i64, 2, 5, 7];
        let ai = vec![0i64, 1, 0, 1, 2, 1, 2];

        let result = amd_order(3, &ap, &ai);

        assert_eq!(result.stats.n, 3);
        assert_eq!(result.stats.nnz, 7);
        assert!(result.stats.elements_created > 0);
    }

    #[test]
    fn test_amd_arrow() {
        // Arrow matrix: first row/column dense
        // [ x x x x ]
        // [ x x . . ]
        // [ x . x . ]
        // [ x . . x ]
        let ap = vec![0i64, 4, 6, 8, 10];
        let ai = vec![
            0i64, 1, 2, 3,  // col 0
            0, 1,            // col 1
            0, 2,            // col 2
            0, 3,            // col 3
        ];

        let result = amd_order(4, &ap, &ai);
        verify_permutation(&result, 4);

        // Dense column (0) should be eliminated last
        assert!(result.perm[0] >= result.perm[1] ||
                result.perm[0] >= result.perm[2] ||
                result.perm[0] >= result.perm[3],
                "Dense node should not be eliminated first");
    }
}
