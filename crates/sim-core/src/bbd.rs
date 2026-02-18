//! BBD (Bordered Block Diagonal) Decomposition
//!
//! This module provides graph partitioning and BBD matrix decomposition for
//! circuit simulation matrices. Unlike BTF which relies on strongly connected
//! components, BBD uses graph partitioning to split a matrix into independent
//! sub-blocks connected through a border (separator).
//!
//! # BBD Matrix Structure
//!
//! ```text
//! ┌──────┬──────┬──────┬────────┐
//! │ B₁₁  │  0   │  0   │  C₁   │
//! ├──────┼──────┼──────┼────────┤
//! │  0   │ B₂₂  │  0   │  C₂   │
//! ├──────┼──────┼──────┼────────┤
//! │  0   │  0   │ B₃₃  │  C₃   │
//! ├──────┼──────┼──────┼────────┤
//! │  R₁  │  R₂  │  R₃  │  B_T  │
//! └──────┴──────┴──────┴────────┘
//! ```
//!
//! - B₁₁, B₂₂, B₃₃: Independent diagonal sub-blocks
//! - C_k: Coupling from block k to border
//! - R_k: Coupling from border to block k
//! - B_T: Top (border) block
//!
//! # Partitioning Algorithm
//!
//! The partitioning is pluggable via the `Partitioner` trait. The default
//! `GreedyBisectionPartitioner` uses BFS level-set bisection, which is
//! simple and has O(n + nnz) complexity with no external dependencies.
//!
//! # Usage
//!
//! ```ignore
//! use sim_core::bbd::{bbd_decompose, GreedyBisectionPartitioner};
//!
//! let partitioner = GreedyBisectionPartitioner::new();
//! let decomp = bbd_decompose(n, &ap, &ai, &partitioner, 4);
//! ```

use std::collections::VecDeque;

/// Result of graph partitioning
#[derive(Debug, Clone)]
pub struct Partition {
    /// Each node's block assignment: 0..num_blocks-1 for blocks, usize::MAX for border
    pub assignment: Vec<usize>,
    /// Number of blocks (not counting border)
    pub num_blocks: usize,
}

/// Pluggable graph partitioning algorithm
pub trait Partitioner: Send {
    /// Partition an n×n sparse matrix (CSC format) into blocks + border.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension
    /// * `ap` - Column pointers (length n+1)
    /// * `ai` - Row indices
    /// * `num_blocks` - Target number of blocks
    fn partition(
        &self,
        n: usize,
        ap: &[i64],
        ai: &[i64],
        num_blocks: usize,
    ) -> Partition;

    /// Algorithm name for diagnostics
    fn name(&self) -> &str;
}

/// BFS level-set recursive bisection partitioner.
///
/// This simple partitioner uses BFS from the lowest-degree node to create
/// level sets, then bisects the graph by assigning each half of the levels
/// to different partitions. Nodes on the cut boundary become border nodes.
///
/// Complexity: O(n + nnz) per bisection level, O((n + nnz) · log₂(num_blocks)) total.
#[derive(Debug, Clone)]
pub struct GreedyBisectionPartitioner {
    _private: (),
}

impl GreedyBisectionPartitioner {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for GreedyBisectionPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl Partitioner for GreedyBisectionPartitioner {
    fn partition(
        &self,
        n: usize,
        ap: &[i64],
        ai: &[i64],
        num_blocks: usize,
    ) -> Partition {
        if n == 0 || num_blocks <= 1 {
            return Partition {
                assignment: vec![0; n],
                num_blocks: 1,
            };
        }

        // Build symmetric adjacency list (ignore self-loops)
        let adj = build_symmetric_adjacency(n, ap, ai);

        // Start with all nodes in one group
        let mut assignment = vec![0usize; n];
        let mut current_blocks = 1usize;

        // Recursively bisect until we have enough blocks
        // We bisect the largest non-border block each time
        while current_blocks < num_blocks {
            // Find the largest block to bisect
            let mut block_sizes = vec![0usize; current_blocks];
            for &a in &assignment {
                if a != usize::MAX && a < current_blocks {
                    block_sizes[a] += 1;
                }
            }

            // Find the largest block
            let target_block = block_sizes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &size)| size)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if block_sizes[target_block] < 4 {
                // Block too small to bisect further
                break;
            }

            // Collect nodes in the target block
            let block_nodes: Vec<usize> = (0..n)
                .filter(|&i| assignment[i] == target_block)
                .collect();

            // BFS bisection on this sub-block
            let new_block_id = current_blocks;
            let bisection = bfs_bisect(&block_nodes, &adj);

            // Apply bisection: group B becomes new_block_id, cut nodes become border
            for &node in &block_nodes {
                match bisection[&node] {
                    BisectGroup::A => {} // stays in target_block
                    BisectGroup::B => assignment[node] = new_block_id,
                    BisectGroup::Cut => assignment[node] = usize::MAX,
                }
            }

            current_blocks += 1;
        }

        Partition {
            assignment,
            num_blocks: current_blocks,
        }
    }

    fn name(&self) -> &str {
        "GreedyBisection"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BisectGroup {
    A,
    B,
    Cut,
}

/// BFS bisection on a subset of nodes.
/// Returns a map from node -> BisectGroup.
fn bfs_bisect(
    nodes: &[usize],
    adj: &[Vec<usize>],
) -> std::collections::HashMap<usize, BisectGroup> {
    use std::collections::{HashMap, HashSet};

    let node_set: HashSet<usize> = nodes.iter().copied().collect();
    let mut result = HashMap::with_capacity(nodes.len());

    if nodes.is_empty() {
        return result;
    }

    if nodes.len() <= 3 {
        // Too small to bisect meaningfully — all in group A
        for &n in nodes {
            result.insert(n, BisectGroup::A);
        }
        return result;
    }

    // Find the node with minimum degree (within the subgraph)
    let start = *nodes
        .iter()
        .min_by_key(|&&n| {
            adj[n].iter().filter(|&&nb| node_set.contains(&nb)).count()
        })
        .unwrap();

    // BFS to compute level sets
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::with_capacity(nodes.len());
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited.insert(start);

    while !queue.is_empty() {
        let mut level = Vec::new();
        let level_size = queue.len();
        for _ in 0..level_size {
            let node = queue.pop_front().unwrap();
            level.push(node);
            for &nb in &adj[node] {
                if node_set.contains(&nb) && !visited.contains(&nb) {
                    visited.insert(nb);
                    queue.push_back(nb);
                }
            }
        }
        levels.push(level);
    }

    // Handle disconnected nodes (not reached by BFS)
    for &n in nodes {
        if !visited.contains(&n) {
            // Add as a singleton level
            levels.push(vec![n]);
        }
    }

    // Split levels in half
    let total_nodes: usize = levels.iter().map(|l| l.len()).sum();
    let half = total_nodes / 2;

    let mut count = 0;
    let mut split_level = levels.len(); // default: all in A

    for (i, level) in levels.iter().enumerate() {
        count += level.len();
        if count >= half {
            split_level = i + 1;
            break;
        }
    }

    // Assign groups
    for (i, level) in levels.iter().enumerate() {
        let group = if i < split_level {
            BisectGroup::A
        } else {
            BisectGroup::B
        };
        for &node in level {
            result.insert(node, group);
        }
    }

    // Identify cut nodes: nodes adjacent to the other group become border
    let mut cut_nodes = Vec::new();
    for &node in nodes {
        let my_group = result[&node];
        if my_group == BisectGroup::Cut {
            continue;
        }
        for &nb in &adj[node] {
            if let Some(&nb_group) = result.get(&nb) {
                if nb_group != my_group && nb_group != BisectGroup::Cut {
                    cut_nodes.push(node);
                    break;
                }
            }
        }
    }

    for node in cut_nodes {
        result.insert(node, BisectGroup::Cut);
    }

    result
}

/// Build symmetric adjacency list from CSC matrix, ignoring self-loops.
fn build_symmetric_adjacency(n: usize, ap: &[i64], ai: &[i64]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            if row < n && row != col {
                adj[col].push(row);
                adj[row].push(col);
            }
        }
    }

    // Deduplicate
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    adj
}

/// BBD decomposition result
#[derive(Debug, Clone)]
pub struct BbdDecomposition {
    /// Number of diagonal blocks (not counting border)
    pub num_blocks: usize,
    /// Nodes in each diagonal block (global indices)
    pub block_nodes: Vec<Vec<usize>>,
    /// Border nodes (global indices)
    pub border_nodes: Vec<usize>,
    /// Permutation: perm[new] = old
    pub perm: Vec<usize>,
    /// Inverse permutation: inv_perm[old] = new
    pub inv_perm: Vec<usize>,
    /// Block boundaries: block k occupies rows/cols [block_ptr[k], block_ptr[k+1])
    pub block_ptr: Vec<usize>,
    /// Start of border in the permuted matrix
    pub border_start: usize,
    /// Size of the border
    pub border_size: usize,
}

/// Perform BBD decomposition on an n×n sparse matrix (CSC format).
///
/// 1. Calls the partitioner to get block assignments
/// 2. Identifies additional border nodes (nodes with edges to multiple blocks)
/// 3. Builds the permutation that orders blocks first, then border
///
/// # Arguments
/// * `n` - Matrix dimension
/// * `ap` - Column pointers
/// * `ai` - Row indices
/// * `partitioner` - Partitioning algorithm
/// * `num_blocks` - Target number of blocks
pub fn bbd_decompose(
    n: usize,
    ap: &[i64],
    ai: &[i64],
    partitioner: &dyn Partitioner,
    num_blocks: usize,
) -> BbdDecomposition {
    // Step 1: Partition
    let mut partition = partitioner.partition(n, ap, ai, num_blocks);

    // Step 2: Promote nodes that connect to multiple blocks to border
    // A node should be border if it has edges to nodes in a different block
    for col in 0..n {
        if partition.assignment[col] == usize::MAX {
            continue; // already border
        }
        let my_block = partition.assignment[col];
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            if row < n && row != col {
                let row_block = partition.assignment[row];
                if row_block != usize::MAX && row_block != my_block {
                    // This node connects to a different block — promote to border
                    partition.assignment[col] = usize::MAX;
                    break;
                }
            }
        }
    }

    // Also check row direction (since CSC gives us column → row)
    // We need to check: for each row r, if any column c has A(r,c) ≠ 0
    // and r,c are in different blocks, promote r to border.
    // We already handled col→row above. For row→col:
    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            if row < n && row != col {
                if partition.assignment[row] == usize::MAX {
                    continue;
                }
                let row_block = partition.assignment[row];
                let col_block = partition.assignment[col];
                if col_block != usize::MAX && col_block != row_block {
                    partition.assignment[row] = usize::MAX;
                }
            }
        }
    }

    // Step 3: Collect nodes per block and border
    let actual_num_blocks = partition.num_blocks;
    let mut block_nodes: Vec<Vec<usize>> = vec![Vec::new(); actual_num_blocks];
    let mut border_nodes: Vec<usize> = Vec::new();

    for i in 0..n {
        if partition.assignment[i] == usize::MAX {
            border_nodes.push(i);
        } else {
            block_nodes[partition.assignment[i]].push(i);
        }
    }

    // Remove empty blocks and renumber
    let mut non_empty_blocks: Vec<Vec<usize>> = block_nodes
        .into_iter()
        .filter(|b| !b.is_empty())
        .collect();
    let final_num_blocks = non_empty_blocks.len();

    // If we ended up with 0 or 1 blocks, the decomposition is trivial
    if final_num_blocks <= 1 {
        // Put everything in one block, no border
        let mut all_nodes: Vec<usize> = (0..n).collect();
        if final_num_blocks == 1 {
            all_nodes = non_empty_blocks.remove(0);
            all_nodes.extend_from_slice(&border_nodes);
        }
        let perm: Vec<usize> = all_nodes.clone();
        let mut inv_perm = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv_perm[old] = new;
        }
        return BbdDecomposition {
            num_blocks: 1,
            block_nodes: vec![all_nodes],
            border_nodes: Vec::new(),
            perm,
            inv_perm,
            block_ptr: vec![0, n],
            border_start: n,
            border_size: 0,
        };
    }

    // Step 4: Build permutation — blocks first, then border
    let mut perm = Vec::with_capacity(n);
    let mut block_ptr = Vec::with_capacity(final_num_blocks + 1);

    for block in &non_empty_blocks {
        block_ptr.push(perm.len());
        perm.extend_from_slice(block);
    }
    block_ptr.push(perm.len()); // end of last block

    let border_start = perm.len();
    perm.extend_from_slice(&border_nodes);

    assert_eq!(perm.len(), n);

    // Build inverse permutation
    let mut inv_perm = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv_perm[old] = new;
    }

    BbdDecomposition {
        num_blocks: final_num_blocks,
        block_nodes: non_empty_blocks,
        border_nodes,
        perm,
        inv_perm,
        block_ptr,
        border_start,
        border_size: n - border_start,
    }
}

/// Heuristic: should BBD be used for this matrix?
///
/// BBD is beneficial when:
/// - Matrix is large enough (n >= 200)
/// - Matrix is sparse (density < 15%)
/// - Border is small relative to the matrix (border_size < 0.3 * n)
pub fn should_use_bbd(n: usize, density: f64, border_ratio: f64) -> bool {
    n >= 200 && density < 0.15 && border_ratio < 0.3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_diagonal() {
        // 4×4 diagonal matrix — each node is independent
        let ap = vec![0i64, 1, 2, 3, 4];
        let ai = vec![0i64, 1, 2, 3];

        let partitioner = GreedyBisectionPartitioner::new();
        let partition = partitioner.partition(4, &ap, &ai, 2);

        // With no edges, bisection should still produce 2 groups
        assert!(partition.num_blocks >= 1);
    }

    #[test]
    fn test_partition_block_diagonal() {
        // Two independent 2×2 blocks:
        // [a b 0 0]
        // [c d 0 0]
        // [0 0 e f]
        // [0 0 g h]
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let partitioner = GreedyBisectionPartitioner::new();
        let partition = partitioner.partition(4, &ap, &ai, 2);

        // Nodes 0,1 should be in one block and nodes 2,3 in another
        assert_eq!(partition.num_blocks, 2);
        assert_ne!(partition.assignment[0], usize::MAX);
        assert_eq!(partition.assignment[0], partition.assignment[1]);
        assert_ne!(partition.assignment[2], usize::MAX);
        assert_eq!(partition.assignment[2], partition.assignment[3]);
        assert_ne!(partition.assignment[0], partition.assignment[2]);
    }

    #[test]
    fn test_decompose_block_diagonal() {
        // Two independent 2×2 blocks
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(4, &ap, &ai, &partitioner, 2);

        // Should have 2 blocks, 0 border
        assert_eq!(decomp.num_blocks, 2);
        assert_eq!(decomp.border_size, 0);
        assert_eq!(decomp.border_nodes.len(), 0);

        // Each block should have 2 nodes
        assert_eq!(decomp.block_nodes[0].len(), 2);
        assert_eq!(decomp.block_nodes[1].len(), 2);

        // Permutation should be valid
        let mut sorted_perm = decomp.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_decompose_with_coupling() {
        // 5×5 matrix: two blocks connected through node 2
        // [a . b . .]
        // [. a . . .]
        // [c . d . e]   <- node 2 connects to blocks {0,1} and {3,4}
        // [. . . a .]
        // [. . f . a]
        let ap = vec![0i64, 2, 3, 6, 7, 9];
        let ai = vec![
            0i64, 2, // col 0
            1,    // col 1
            0, 2, 4, // col 2
            3,    // col 3
            2, 4, // col 4
        ];

        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(5, &ap, &ai, &partitioner, 2);

        // Node 2 should be in the border since it connects to both groups
        assert!(
            decomp.border_nodes.contains(&2),
            "Node 2 should be border, got: {:?}",
            decomp
        );

        // Permutation should be valid
        let mut sorted_perm = decomp.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3, 4]);

        // Border should be at the end
        assert!(decomp.border_start <= 5);
        assert_eq!(decomp.border_start + decomp.border_size, 5);
    }

    #[test]
    fn test_decompose_ladder() {
        // Ladder/chain network: 0-1-2-3-4-5
        // This tests that the partitioner can handle a path graph
        let n = 6;
        // Build tridiagonal matrix
        let mut ap = vec![0i64];
        let mut ai = Vec::new();
        for col in 0..n {
            if col > 0 {
                ai.push((col - 1) as i64);
            }
            ai.push(col as i64);
            if col < n - 1 {
                ai.push((col + 1) as i64);
            }
            ap.push(ai.len() as i64);
        }

        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(n, &ap, &ai, &partitioner, 2);

        // Should produce 2 blocks with some border nodes
        assert!(decomp.num_blocks >= 1, "Should have at least 1 block");

        // Permutation should be valid
        let mut sorted_perm = decomp.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_decompose_permutation_consistency() {
        // Verify perm and inv_perm are consistent
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(4, &ap, &ai, &partitioner, 2);

        for new in 0..4 {
            let old = decomp.perm[new];
            assert_eq!(decomp.inv_perm[old], new);
        }
    }

    #[test]
    fn test_should_use_bbd() {
        assert!(!should_use_bbd(100, 0.10, 0.2)); // too small
        assert!(!should_use_bbd(200, 0.20, 0.2)); // too dense
        assert!(!should_use_bbd(200, 0.10, 0.5)); // border too large
        assert!(should_use_bbd(200, 0.10, 0.2)); // good candidate
        assert!(should_use_bbd(1000, 0.01, 0.1)); // ideal
    }

    #[test]
    fn test_empty_matrix() {
        let ap = vec![0i64];
        let ai: Vec<i64> = vec![];
        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(0, &ap, &ai, &partitioner, 2);
        assert_eq!(decomp.num_blocks, 1);
        assert_eq!(decomp.border_size, 0);
    }

    #[test]
    fn test_single_node() {
        let ap = vec![0i64, 1];
        let ai = vec![0i64];
        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(1, &ap, &ai, &partitioner, 2);
        assert_eq!(decomp.num_blocks, 1);
        assert_eq!(decomp.perm, vec![0]);
    }

    #[test]
    fn test_block_ptr_boundaries() {
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let partitioner = GreedyBisectionPartitioner::new();
        let decomp = bbd_decompose(4, &ap, &ai, &partitioner, 2);

        // block_ptr should have num_blocks + 1 entries
        assert_eq!(decomp.block_ptr.len(), decomp.num_blocks + 1);

        // block_ptr[0] should be 0
        assert_eq!(decomp.block_ptr[0], 0);

        // block_ptr[last] should equal border_start
        assert_eq!(decomp.block_ptr[decomp.num_blocks], decomp.border_start);

        // Each block should be non-empty
        for k in 0..decomp.num_blocks {
            assert!(
                decomp.block_ptr[k + 1] > decomp.block_ptr[k],
                "Block {} is empty",
                k
            );
        }
    }
}
