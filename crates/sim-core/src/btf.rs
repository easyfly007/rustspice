//! Block Triangular Form (BTF) Decomposition
//!
//! This module implements the Block Triangular Form decomposition algorithm for
//! sparse matrices. BTF permutes a matrix to upper block triangular form, where
//! the diagonal blocks are irreducible (strongly connected).
//!
//! # Algorithm Overview
//!
//! The BTF algorithm has three phases:
//!
//! 1. **Maximum Transversal**: Find a row permutation that maximizes diagonal
//!    non-zeros. This is equivalent to maximum bipartite matching.
//!
//! 2. **Strongly Connected Components**: Use Tarjan's algorithm to find SCCs
//!    in the directed graph induced by the matching. Each SCC becomes a diagonal block.
//!
//! 3. **Permutation Construction**: Build row/column permutations from the SCCs
//!    in topological order.
//!
//! # Performance
//!
//! - Maximum Transversal: O(n × nnz) with DFS-based algorithm
//! - Tarjan's SCC: O(n + nnz)
//! - Total: O(n × nnz)
//!
//! # References
//!
//! - Pothen, A., Fan, C.-J. "Computing the block triangular form of a sparse matrix"
//!   ACM TOMS, Vol. 16, No. 4, pp. 303-324, 1990.
//!
//! - Duff, I.S. "On algorithms for obtaining a maximum transversal"
//!   ACM TOMS, Vol. 7, No. 3, pp. 315-330, 1981.
//!
//! - Tarjan, R.E. "Depth-first search and linear graph algorithms"
//!   SIAM J. Computing, Vol. 1, No. 2, pp. 146-160, 1972.
//!
//! - Davis, T.A., Palamadai Natarajan, E. "Algorithm 907: KLU, A Direct Sparse
//!   Solver for Circuit Simulation Problems" ACM TOMS, Vol. 37, No. 3, 2010.
//!
//! # Example
//!
//! ```ignore
//! use sim_core::btf::{btf_decompose, BtfDecomposition};
//!
//! // CSC format sparse matrix
//! let ap = vec![0i64, 2, 4, 6];
//! let ai = vec![0i64, 1, 0, 2, 1, 2];
//!
//! let btf = btf_decompose(3, &ap, &ai);
//!
//! println!("Number of blocks: {}", btf.num_blocks);
//! println!("Block sizes: {:?}", btf.block_sizes());
//! ```

use std::cmp::min;

/// Result of BTF decomposition
///
/// Contains the row and column permutations that transform the matrix
/// into upper block triangular form, along with block boundary information.
#[derive(Debug, Clone)]
pub struct BtfDecomposition {
    /// Row permutation: row_perm[new_pos] = old_row
    /// To apply: B[new_i, new_j] = A[row_perm[new_i], col_perm[new_j]]
    pub row_perm: Vec<usize>,

    /// Column permutation: col_perm[new_pos] = old_col
    pub col_perm: Vec<usize>,

    /// Inverse row permutation: row_perm_inv[old_row] = new_pos
    pub row_perm_inv: Vec<usize>,

    /// Inverse column permutation: col_perm_inv[old_col] = new_pos
    pub col_perm_inv: Vec<usize>,

    /// Block boundaries: block k spans columns [block_ptr[k], block_ptr[k+1])
    pub block_ptr: Vec<usize>,

    /// Number of diagonal blocks
    pub num_blocks: usize,

    /// Structural rank (number of matched rows/columns)
    /// If structural_rank < n, the matrix is structurally singular
    pub structural_rank: usize,

    /// Number of 1×1 blocks (singletons)
    pub num_singletons: usize,

    /// Maximum block size
    pub max_block_size: usize,
}

impl BtfDecomposition {
    /// Get the size of block k
    pub fn block_size(&self, k: usize) -> usize {
        if k < self.num_blocks {
            self.block_ptr[k + 1] - self.block_ptr[k]
        } else {
            0
        }
    }

    /// Get sizes of all blocks
    pub fn block_sizes(&self) -> Vec<usize> {
        (0..self.num_blocks).map(|k| self.block_size(k)).collect()
    }

    /// Check if the matrix has full structural rank
    pub fn is_structurally_nonsingular(&self) -> bool {
        self.structural_rank == self.row_perm.len()
    }

    /// Get the start and end indices for block k
    pub fn block_range(&self, k: usize) -> (usize, usize) {
        (self.block_ptr[k], self.block_ptr[k + 1])
    }
}

/// Compute BTF decomposition of a sparse matrix
///
/// # Arguments
///
/// * `n` - Matrix dimension (n×n)
/// * `ap` - Column pointers (CSC format), length n+1
/// * `ai` - Row indices (CSC format)
///
/// # Returns
///
/// BTF decomposition containing permutations and block structure
///
/// # Algorithm
///
/// 1. **Maximum Transversal** (Duff, 1981): Uses DFS-based augmenting path
///    algorithm to find a maximum matching in the bipartite graph representation.
///    This establishes a correspondence between rows and columns.
///
/// 2. **Tarjan's SCC** (Tarjan, 1972): Constructs a directed graph where
///    edge (i→j) exists if A(match[j], i) ≠ 0 and i ≠ j. Finds strongly
///    connected components using single DFS pass with lowlink tracking.
///
/// 3. **Permutation Construction**: SCCs in reverse topological order are
///    reversed to get topological order. Each SCC becomes a diagonal block.
pub fn btf_decompose(n: usize, ap: &[i64], ai: &[i64]) -> BtfDecomposition {
    if n == 0 {
        return BtfDecomposition {
            row_perm: vec![],
            col_perm: vec![],
            row_perm_inv: vec![],
            col_perm_inv: vec![],
            block_ptr: vec![0],
            num_blocks: 0,
            structural_rank: 0,
            num_singletons: 0,
            max_block_size: 0,
        };
    }

    // Phase 1: Maximum Transversal
    // col_match[j] = row matched to column j (-1 if unmatched)
    // row_match[i] = column matched to row i (-1 if unmatched)
    let (col_match, row_match) = maximum_transversal(n, ap, ai);

    // Count structural rank
    let structural_rank = col_match.iter().filter(|&&x| x != -1).count();

    // Phase 2: Build directed graph and find SCCs
    // We build adjacency list for the directed graph where
    // edge (col_i → col_j) exists if A(col_match[col_j], col_i) ≠ 0 and col_i ≠ col_j
    let adj = build_directed_graph(n, ap, ai, &col_match);

    // Find SCCs using Tarjan's algorithm
    let sccs = tarjan_scc(n, &adj);

    // Phase 3: Build permutations from SCCs
    // SCCs come in reverse topological order, we need to reverse them
    build_permutations(n, &col_match, &row_match, sccs, structural_rank)
}

// ============================================================================
// Phase 1: Maximum Transversal (Maximum Bipartite Matching)
// ============================================================================

/// Find maximum transversal using DFS-based augmenting path algorithm
///
/// This implements the algorithm described in:
/// Duff, I.S. "On algorithms for obtaining a maximum transversal"
/// ACM TOMS, Vol. 7, No. 3, pp. 315-330, 1981.
///
/// # Algorithm
///
/// For each row, we try to find an augmenting path using DFS:
/// - An augmenting path starts at an unmatched row
/// - Alternates between non-matching and matching edges
/// - Ends at an unmatched column
///
/// When found, we "flip" the path (matching edges become non-matching and vice versa),
/// increasing the matching size by 1.
///
/// # Complexity
///
/// O(n × nnz) worst case, typically much better for sparse matrices.
fn maximum_transversal(n: usize, ap: &[i64], ai: &[i64]) -> (Vec<i32>, Vec<i32>) {
    // col_match[j] = row matched to column j (-1 if unmatched)
    let mut col_match: Vec<i32> = vec![-1; n];
    // row_match[i] = column matched to row i (-1 if unmatched)
    let mut row_match: Vec<i32> = vec![-1; n];

    // We need to iterate over rows, but matrix is in CSC format (column-major)
    // First, build row-to-column adjacency for efficient row iteration
    let row_to_cols = build_row_adjacency(n, ap, ai);

    // Try to find augmenting path from each row
    for row in 0..n {
        let mut visited = vec![false; n];
        augment_path(
            row,
            &row_to_cols,
            &mut col_match,
            &mut row_match,
            &mut visited,
        );
    }

    (col_match, row_match)
}

/// Build row-to-column adjacency list from CSC matrix
fn build_row_adjacency(n: usize, ap: &[i64], ai: &[i64]) -> Vec<Vec<usize>> {
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];

    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            if row < n {
                row_to_cols[row].push(col);
            }
        }
    }

    row_to_cols
}

/// Try to find augmenting path from given row using DFS
///
/// Returns true if augmenting path was found and matching was updated.
fn augment_path(
    row: usize,
    row_to_cols: &[Vec<usize>],
    col_match: &mut [i32],
    row_match: &mut [i32],
    visited: &mut [bool],
) -> bool {
    // Try each column adjacent to this row
    for &col in &row_to_cols[row] {
        if visited[col] {
            continue;
        }
        visited[col] = true;

        // If column is unmatched, we found an augmenting path
        if col_match[col] == -1 {
            col_match[col] = row as i32;
            row_match[row] = col as i32;
            return true;
        }

        // Column is matched to some other row - try to find alternate path
        let matched_row = col_match[col] as usize;
        if augment_path(matched_row, row_to_cols, col_match, row_match, visited) {
            // Found augmenting path through matched_row, update this edge
            col_match[col] = row as i32;
            row_match[row] = col as i32;
            return true;
        }
    }

    false
}

// ============================================================================
// Phase 2: Strongly Connected Components (Tarjan's Algorithm)
// ============================================================================

/// Build directed graph for SCC computation
///
/// After maximum matching, we construct directed graph Gᵈ:
/// - Vertices: {0, 1, ..., n-1} representing columns (and their matched rows)
/// - Edge (i → j) if A(col_match[j], i) ≠ 0 and i ≠ j
///
/// In other words, there's an edge from column i to column j if:
/// - Column j is matched to some row r = col_match[j]
/// - There's a non-zero in column i at row r (A(r, i) ≠ 0)
/// - i ≠ j
fn build_directed_graph(
    n: usize,
    ap: &[i64],
    ai: &[i64],
    col_match: &[i32],
) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    // For each column i, find edges i → j
    for col_i in 0..n {
        let start = ap[col_i] as usize;
        let end = ap[col_i + 1] as usize;

        // For each row in column i
        for idx in start..end {
            let row = ai[idx] as usize;
            if row >= n {
                continue;
            }

            // Find which column j is matched to this row
            // We need to find j such that col_match[j] == row
            // This is expensive to compute directly, so we iterate differently

            // Actually, we need row_match for this:
            // If row is matched to column j, and j ≠ col_i, add edge col_i → j
            // But we built col_match, not row_match lookup...

            // Let's use a different approach: for each column j, check if
            // any of its rows has an entry in column i
        }
    }

    // Better approach: iterate over columns j, look at their matched row,
    // then find all columns i that have entries in that row
    // First build row-to-column adjacency
    let row_to_cols = build_row_adjacency(n, ap, ai);

    for col_j in 0..n {
        if col_match[col_j] == -1 {
            continue; // Unmatched column
        }

        let matched_row = col_match[col_j] as usize;

        // For all columns i that have an entry at matched_row
        for &col_i in &row_to_cols[matched_row] {
            if col_i != col_j {
                adj[col_i].push(col_j);
            }
        }
    }

    adj
}

/// State for Tarjan's SCC algorithm
struct TarjanState {
    /// Discovery time for each vertex (-1 = not visited)
    index: Vec<i32>,
    /// Lowest reachable discovery time
    lowlink: Vec<i32>,
    /// Whether vertex is currently on DFS stack
    on_stack: Vec<bool>,
    /// DFS stack
    stack: Vec<usize>,
    /// Global discovery time counter
    index_counter: i32,
    /// Resulting SCCs (in reverse topological order)
    sccs: Vec<Vec<usize>>,
}

/// Find strongly connected components using Tarjan's algorithm
///
/// This implements the algorithm described in:
/// Tarjan, R.E. "Depth-first search and linear graph algorithms"
/// SIAM J. Computing, Vol. 1, No. 2, pp. 146-160, 1972.
///
/// # Key Concepts
///
/// - **index[v]**: Discovery time (when v was first visited in DFS)
/// - **lowlink[v]**: Lowest discovery time reachable from subtree rooted at v
/// - **SCC root**: Vertex v is root of SCC iff lowlink[v] == index[v]
///
/// # Output Order
///
/// SCCs are returned in **reverse topological order** (dependencies come first).
/// For BTF, we reverse this to get topological order.
///
/// # Complexity
///
/// O(n + nnz) - each vertex and edge visited exactly once.
fn tarjan_scc(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut state = TarjanState {
        index: vec![-1; n],
        lowlink: vec![-1; n],
        on_stack: vec![false; n],
        stack: Vec::with_capacity(n),
        index_counter: 0,
        sccs: Vec::new(),
    };

    // Visit all vertices
    for v in 0..n {
        if state.index[v] == -1 {
            strong_connect(v, adj, &mut state);
        }
    }

    state.sccs
}

/// Recursive DFS for Tarjan's algorithm
fn strong_connect(v: usize, adj: &[Vec<usize>], state: &mut TarjanState) {
    // Initialize discovery time and lowlink
    state.index[v] = state.index_counter;
    state.lowlink[v] = state.index_counter;
    state.index_counter += 1;

    // Push onto stack
    state.stack.push(v);
    state.on_stack[v] = true;

    // Explore all neighbors
    for &w in &adj[v] {
        if state.index[w] == -1 {
            // Tree edge: w not yet visited
            strong_connect(w, adj, state);
            state.lowlink[v] = min(state.lowlink[v], state.lowlink[w]);
        } else if state.on_stack[w] {
            // Back edge: w is on stack (ancestor of v in current DFS)
            state.lowlink[v] = min(state.lowlink[v], state.index[w]);
        }
        // If w is visited but not on stack, it's a cross edge to
        // an already-completed SCC - ignore it
    }

    // If v is root of SCC (lowlink[v] == index[v])
    if state.lowlink[v] == state.index[v] {
        // Pop all vertices in this SCC from stack
        let mut scc = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack[w] = false;
            scc.push(w);
            if w == v {
                break;
            }
        }
        state.sccs.push(scc);
    }
}

// ============================================================================
// Phase 3: Build Permutations
// ============================================================================

/// Build BTF permutations from SCCs
///
/// SCCs from Tarjan's algorithm are in reverse topological order.
/// We reverse them to get topological order, then build permutations.
fn build_permutations(
    n: usize,
    col_match: &[i32],
    _row_match: &[i32],
    mut sccs: Vec<Vec<usize>>,
    structural_rank: usize,
) -> BtfDecomposition {
    // Reverse SCCs to get topological order
    sccs.reverse();

    // Handle unmatched columns (structurally singular case)
    // They form singleton blocks at the end
    let mut unmatched_cols: Vec<usize> = (0..n)
        .filter(|&j| col_match[j] == -1)
        .collect();

    // Build permutations
    let mut row_perm = Vec::with_capacity(n);
    let mut col_perm = Vec::with_capacity(n);
    let mut block_ptr = vec![0usize];

    let mut num_singletons = 0;
    let mut max_block_size = 0;

    // Process matched SCCs
    for scc in &sccs {
        // Only process if SCC contains matched columns
        let matched_in_scc: Vec<usize> = scc
            .iter()
            .filter(|&&col| col_match[col] != -1)
            .copied()
            .collect();

        if matched_in_scc.is_empty() {
            continue;
        }

        for &col in &matched_in_scc {
            col_perm.push(col);
            row_perm.push(col_match[col] as usize);
        }

        let block_size = matched_in_scc.len();
        block_ptr.push(col_perm.len());

        if block_size == 1 {
            num_singletons += 1;
        }
        max_block_size = max_block_size.max(block_size);
    }

    // Add unmatched columns as singletons at the end
    // (these represent structurally singular parts)
    for col in unmatched_cols.drain(..) {
        col_perm.push(col);
        // For unmatched columns, we need to pick an unmatched row
        // Find any row not yet used
        let used_rows: std::collections::HashSet<usize> =
            row_perm.iter().copied().collect();
        for row in 0..n {
            if !used_rows.contains(&row) {
                row_perm.push(row);
                break;
            }
        }
        block_ptr.push(col_perm.len());
        num_singletons += 1;
    }

    let num_blocks = block_ptr.len() - 1;

    // Build inverse permutations
    let mut row_perm_inv = vec![0; n];
    let mut col_perm_inv = vec![0; n];

    for (new_pos, &old_row) in row_perm.iter().enumerate() {
        row_perm_inv[old_row] = new_pos;
    }
    for (new_pos, &old_col) in col_perm.iter().enumerate() {
        col_perm_inv[old_col] = new_pos;
    }

    BtfDecomposition {
        row_perm,
        col_perm,
        row_perm_inv,
        col_perm_inv,
        block_ptr,
        num_blocks,
        structural_rank,
        num_singletons,
        max_block_size,
    }
}

// ============================================================================
// Integration with Sparse LU
// ============================================================================

/// Check if BTF would be beneficial for this matrix
///
/// Heuristic based on matrix size and sparsity.
/// BTF overhead is O(n × nnz), so only worthwhile for larger matrices.
pub fn should_use_btf(n: usize, nnz: usize) -> bool {
    // Don't use BTF for small matrices
    if n < 50 {
        return false;
    }

    // Check sparsity - BTF is most useful for sparse matrices
    let density = nnz as f64 / (n * n) as f64;
    if density > 0.25 {
        return false; // Too dense
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btf_diagonal_matrix() {
        // Diagonal matrix - each element is its own SCC
        // Should produce n singleton blocks
        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];

        let btf = btf_decompose(3, &ap, &ai);

        assert_eq!(btf.structural_rank, 3);
        assert_eq!(btf.num_blocks, 3);
        assert_eq!(btf.num_singletons, 3);
        assert!(btf.is_structurally_nonsingular());
    }

    #[test]
    fn test_btf_full_matrix() {
        // Fully connected 2×2 matrix - single SCC
        let ap = vec![0i64, 2, 4];
        let ai = vec![0i64, 1, 0, 1];

        let btf = btf_decompose(2, &ap, &ai);

        assert_eq!(btf.structural_rank, 2);
        assert_eq!(btf.num_blocks, 1);
        assert_eq!(btf.max_block_size, 2);
    }

    #[test]
    fn test_btf_triangular() {
        // Upper triangular matrix
        // [ 1  2  3 ]
        // [ 0  4  5 ]
        // [ 0  0  6 ]
        // CSC: col0=[0], col1=[0,1], col2=[0,1,2]
        let ap = vec![0i64, 1, 3, 6];
        let ai = vec![0i64, 0, 1, 0, 1, 2];

        let btf = btf_decompose(3, &ap, &ai);

        assert_eq!(btf.structural_rank, 3);
        // Should have 3 singleton blocks (triangular = no cycles)
        assert_eq!(btf.num_blocks, 3);
        assert_eq!(btf.num_singletons, 3);
    }

    #[test]
    fn test_btf_two_blocks() {
        // Block diagonal matrix with two 2×2 blocks
        // [ 1  2  0  0 ]
        // [ 3  4  0  0 ]
        // [ 0  0  5  6 ]
        // [ 0  0  7  8 ]
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let btf = btf_decompose(4, &ap, &ai);

        assert_eq!(btf.structural_rank, 4);
        assert_eq!(btf.num_blocks, 2);
        assert_eq!(btf.block_size(0), 2);
        assert_eq!(btf.block_size(1), 2);
    }

    #[test]
    fn test_btf_structurally_singular() {
        // Matrix with zero row/column
        // [ 1  0 ]
        // [ 0  0 ]
        let ap = vec![0i64, 1, 1];
        let ai = vec![0i64];

        let btf = btf_decompose(2, &ap, &ai);

        assert_eq!(btf.structural_rank, 1);
        assert!(!btf.is_structurally_nonsingular());
    }

    #[test]
    fn test_btf_permutation_validity() {
        // Verify permutations are valid (bijections)
        let ap = vec![0i64, 2, 4, 6];
        let ai = vec![0i64, 1, 0, 2, 1, 2];

        let btf = btf_decompose(3, &ap, &ai);

        // Check row_perm is a permutation
        let mut row_sorted = btf.row_perm.clone();
        row_sorted.sort();
        assert_eq!(row_sorted, vec![0, 1, 2]);

        // Check col_perm is a permutation
        let mut col_sorted = btf.col_perm.clone();
        col_sorted.sort();
        assert_eq!(col_sorted, vec![0, 1, 2]);

        // Check inverse permutations
        for i in 0..3 {
            assert_eq!(btf.row_perm[btf.row_perm_inv[i]], i);
            assert_eq!(btf.col_perm[btf.col_perm_inv[i]], i);
        }
    }

    #[test]
    fn test_btf_empty_matrix() {
        let btf = btf_decompose(0, &[0i64], &[]);

        assert_eq!(btf.num_blocks, 0);
        assert_eq!(btf.structural_rank, 0);
    }

    #[test]
    fn test_btf_chain() {
        // Chain structure: 0 → 1 → 2 → 3
        // [ 1  0  0  0 ]
        // [ 1  1  0  0 ]
        // [ 0  1  1  0 ]
        // [ 0  0  1  1 ]
        let ap = vec![0i64, 2, 4, 6, 7];
        let ai = vec![0i64, 1, 1, 2, 2, 3, 3];

        let btf = btf_decompose(4, &ap, &ai);

        assert_eq!(btf.structural_rank, 4);
        // Chain has no cycles, so all singletons
        assert_eq!(btf.num_singletons, 4);
    }

    #[test]
    fn test_btf_cycle() {
        // Cycle: 0 → 1 → 2 → 0
        // [ 1  0  1 ]
        // [ 1  1  0 ]
        // [ 0  1  1 ]
        let ap = vec![0i64, 2, 4, 6];
        let ai = vec![0i64, 1, 1, 2, 0, 2];

        let btf = btf_decompose(3, &ap, &ai);

        assert_eq!(btf.structural_rank, 3);
        // Cycle forms single SCC
        assert_eq!(btf.num_blocks, 1);
        assert_eq!(btf.max_block_size, 3);
    }
}
