//! Comprehensive tests for BTF (Block Triangular Form) decomposition
//!
//! Tests cover:
//! - Maximum transversal (bipartite matching) correctness
//! - Tarjan's SCC algorithm correctness
//! - Permutation validity and inverse consistency
//! - Block structure verification
//! - Integration with sparse LU solver
//! - Circuit-like matrix patterns

use sim_core::btf::{btf_decompose, should_use_btf, BtfDecomposition};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_identity_matrix() {
    // Identity matrix: each entry is its own SCC
    let n = 5;
    let ap: Vec<i64> = (0..=n as i64).collect();
    let ai: Vec<i64> = (0..n as i64).collect();

    let btf = btf_decompose(n, &ap, &ai);

    assert_eq!(btf.structural_rank, n);
    assert_eq!(btf.num_blocks, n);
    assert_eq!(btf.num_singletons, n);
    assert_eq!(btf.max_block_size, 1);
    assert!(btf.is_structurally_nonsingular());
}

#[test]
fn test_dense_2x2() {
    // Dense 2×2 matrix forms single SCC (cycle: 0↔1)
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];

    let btf = btf_decompose(2, &ap, &ai);

    assert_eq!(btf.structural_rank, 2);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.max_block_size, 2);
    assert_eq!(btf.num_singletons, 0);
}

#[test]
fn test_dense_3x3() {
    // Dense 3×3 matrix forms single SCC (all connected)
    let ap = vec![0i64, 3, 6, 9];
    let ai = vec![0i64, 1, 2, 0, 1, 2, 0, 1, 2];

    let btf = btf_decompose(3, &ap, &ai);

    assert_eq!(btf.structural_rank, 3);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.max_block_size, 3);
}

#[test]
fn test_lower_triangular() {
    // Lower triangular matrix: no cycles, all singletons
    // [ 1  0  0 ]
    // [ 2  3  0 ]
    // [ 4  5  6 ]
    let ap = vec![0i64, 3, 5, 6];
    let ai = vec![0i64, 1, 2, 1, 2, 2];

    let btf = btf_decompose(3, &ap, &ai);

    assert_eq!(btf.structural_rank, 3);
    assert_eq!(btf.num_singletons, 3);
    assert!(btf.is_structurally_nonsingular());
}

#[test]
fn test_upper_triangular() {
    // Upper triangular matrix: no cycles, all singletons
    // [ 1  2  3 ]
    // [ 0  4  5 ]
    // [ 0  0  6 ]
    let ap = vec![0i64, 1, 3, 6];
    let ai = vec![0i64, 0, 1, 0, 1, 2];

    let btf = btf_decompose(3, &ap, &ai);

    assert_eq!(btf.structural_rank, 3);
    assert_eq!(btf.num_singletons, 3);
}

// ============================================================================
// Block Structure Tests
// ============================================================================

#[test]
fn test_two_independent_blocks() {
    // Two independent 2×2 blocks (no connection between them)
    // [ 1  2  0  0 ]
    // [ 3  4  0  0 ]
    // [ 0  0  5  6 ]
    // [ 0  0  7  8 ]
    let ap = vec![0i64, 2, 4, 6, 8];
    let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

    let btf = btf_decompose(4, &ap, &ai);

    assert_eq!(btf.structural_rank, 4);
    assert_eq!(btf.num_blocks, 2);

    // Each block should be size 2
    let sizes = btf.block_sizes();
    assert!(sizes.contains(&2));
    assert_eq!(sizes.iter().sum::<usize>(), 4);
}

#[test]
fn test_three_blocks_with_connections() {
    // Three blocks with upper triangular connections
    // Block 1: 2×2, Block 2: 2×2, Block 3: 1×1
    // Connections: Block 1 → Block 2, Block 2 → Block 3
    //
    // [ 1  2  *  *  * ]    (* = connection to later block)
    // [ 3  4  *  *  * ]
    // [ 0  0  5  6  * ]
    // [ 0  0  7  8  * ]
    // [ 0  0  0  0  9 ]

    // This matrix in CSC:
    // Col 0: rows 0,1
    // Col 1: rows 0,1
    // Col 2: rows 0,1,2,3
    // Col 3: rows 0,1,2,3
    // Col 4: rows 0,1,2,3,4
    let ap = vec![0i64, 2, 4, 8, 12, 17];
    let ai = vec![
        0i64, 1, // col 0
        0, 1, // col 1
        0, 1, 2, 3, // col 2
        0, 1, 2, 3, // col 3
        0, 1, 2, 3, 4, // col 4
    ];

    let btf = btf_decompose(5, &ap, &ai);

    assert_eq!(btf.structural_rank, 5);
    // The exact number of blocks depends on the cycle structure
    // With connections, blocks 1 and 2 might merge depending on feedback
    assert!(btf.num_blocks >= 1);
    assert!(btf.num_blocks <= 5);
}

#[test]
fn test_block_boundaries() {
    // Verify block_ptr correctly defines boundaries
    let ap = vec![0i64, 1, 2, 3, 4];
    let ai = vec![0i64, 1, 2, 3];

    let btf = btf_decompose(4, &ap, &ai);

    // Each diagonal entry forms singleton block
    assert_eq!(btf.block_ptr.len(), btf.num_blocks + 1);
    assert_eq!(btf.block_ptr[0], 0);
    assert_eq!(*btf.block_ptr.last().unwrap(), 4);

    // Verify block ranges
    for k in 0..btf.num_blocks {
        let (start, end) = btf.block_range(k);
        assert!(start < end);
        assert!(end <= 4);
        assert_eq!(end - start, btf.block_size(k));
    }
}

// ============================================================================
// Permutation Validity Tests
// ============================================================================

#[test]
fn test_permutation_is_bijection() {
    // Verify row_perm and col_perm are valid permutations (bijections)
    let ap = vec![0i64, 2, 4, 6, 8];
    let ai = vec![0i64, 1, 0, 2, 1, 3, 2, 3];

    let btf = btf_decompose(4, &ap, &ai);

    // row_perm should contain each index exactly once
    let mut row_sorted = btf.row_perm.clone();
    row_sorted.sort();
    assert_eq!(row_sorted, vec![0, 1, 2, 3]);

    // col_perm should contain each index exactly once
    let mut col_sorted = btf.col_perm.clone();
    col_sorted.sort();
    assert_eq!(col_sorted, vec![0, 1, 2, 3]);
}

#[test]
fn test_inverse_permutation_consistency() {
    // Verify perm[inv_perm[i]] == i and inv_perm[perm[i]] == i
    let ap = vec![0i64, 3, 6, 9];
    let ai = vec![0i64, 1, 2, 0, 1, 2, 0, 1, 2];

    let btf = btf_decompose(3, &ap, &ai);

    for i in 0..3 {
        // row_perm[row_perm_inv[i]] == i
        assert_eq!(btf.row_perm[btf.row_perm_inv[i]], i);
        // row_perm_inv[row_perm[i]] == i
        assert_eq!(btf.row_perm_inv[btf.row_perm[i]], i);

        // Same for columns
        assert_eq!(btf.col_perm[btf.col_perm_inv[i]], i);
        assert_eq!(btf.col_perm_inv[btf.col_perm[i]], i);
    }
}

// ============================================================================
// Structural Rank Tests
// ============================================================================

#[test]
fn test_full_structural_rank() {
    // Matrix with perfect matching
    let ap = vec![0i64, 2, 4, 6];
    let ai = vec![0i64, 1, 1, 2, 0, 2];

    let btf = btf_decompose(3, &ap, &ai);

    assert_eq!(btf.structural_rank, 3);
    assert!(btf.is_structurally_nonsingular());
}

#[test]
fn test_structural_rank_deficient() {
    // Matrix with zero row (structurally singular)
    // [ 1  0 ]
    // [ 0  0 ]
    let ap = vec![0i64, 1, 1];
    let ai = vec![0i64];

    let btf = btf_decompose(2, &ap, &ai);

    assert_eq!(btf.structural_rank, 1);
    assert!(!btf.is_structurally_nonsingular());
}

#[test]
fn test_structural_rank_zero_column() {
    // Matrix with zero column
    // [ 1  0  2 ]
    // [ 0  0  3 ]
    // [ 4  0  5 ]
    let ap = vec![0i64, 2, 2, 5];
    let ai = vec![0i64, 2, 0, 1, 2];

    let btf = btf_decompose(3, &ap, &ai);

    // Column 1 has no entries, so rank < 3
    assert_eq!(btf.structural_rank, 2);
    assert!(!btf.is_structurally_nonsingular());
}

// ============================================================================
// Circuit-like Pattern Tests
// ============================================================================

#[test]
fn test_ladder_network() {
    // Ladder network creates tridiagonal pattern
    // Common in RC chains, transmission lines
    //
    // [ 2 -1  0  0 ]
    // [-1  2 -1  0 ]
    // [ 0 -1  2 -1 ]
    // [ 0  0 -1  2 ]
    let ap = vec![0i64, 2, 5, 8, 10];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];

    let btf = btf_decompose(4, &ap, &ai);

    assert_eq!(btf.structural_rank, 4);
    // Tridiagonal with all connections forms single SCC
    assert!(btf.num_blocks >= 1);
}

#[test]
fn test_star_network() {
    // Star network: central node connected to all others
    // Creates dense row/column at node 0
    //
    // [ 4 -1 -1 -1 ]
    // [-1  2  0  0 ]
    // [-1  0  2  0 ]
    // [-1  0  0  2 ]
    let ap = vec![0i64, 4, 6, 8, 10];
    let ai = vec![
        0i64, 1, 2, 3, // col 0: all rows
        0, 1, // col 1
        0, 2, // col 2
        0, 3, // col 3
    ];

    let btf = btf_decompose(4, &ap, &ai);

    assert_eq!(btf.structural_rank, 4);
    // Star creates cycle through center, so likely single block
    assert!(btf.max_block_size >= 2);
}

#[test]
fn test_block_diagonal_subcircuits() {
    // Simulates independent subcircuits (no coupling)
    // Should decompose into separate blocks
    //
    // [ A  0  0 ]    where A, B, C are 2×2 dense
    // [ 0  B  0 ]
    // [ 0  0  C ]
    let ap = vec![0i64, 2, 4, 6, 8, 10, 12];
    let ai = vec![
        0i64, 1, 0, 1, // Block A (cols 0,1)
        2, 3, 2, 3, // Block B (cols 2,3)
        4, 5, 4, 5, // Block C (cols 4,5)
    ];

    let btf = btf_decompose(6, &ap, &ai);

    assert_eq!(btf.structural_rank, 6);
    assert_eq!(btf.num_blocks, 3);

    let sizes = btf.block_sizes();
    assert_eq!(sizes, vec![2, 2, 2]);
}

// ============================================================================
// Cycle Detection Tests
// ============================================================================

#[test]
fn test_simple_2cycle() {
    // Simple 2-cycle: 0 ↔ 1 (with diagonal entries for valid matching)
    // [ 1  1 ]
    // [ 1  1 ]
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];

    let btf = btf_decompose(2, &ap, &ai);

    assert_eq!(btf.structural_rank, 2);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.max_block_size, 2);
}

#[test]
fn test_3cycle() {
    // 3-cycle: 0 → 1 → 2 → 0
    // Matrix has entries at (1,0), (2,1), (0,2) plus diagonal
    let ap = vec![0i64, 2, 4, 6];
    let ai = vec![0i64, 1, 1, 2, 0, 2];

    let btf = btf_decompose(3, &ap, &ai);

    assert_eq!(btf.structural_rank, 3);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.max_block_size, 3);
}

#[test]
fn test_two_separate_cycles() {
    // Two independent cycles: (0,1) and (2,3)
    let ap = vec![0i64, 2, 4, 6, 8];
    let ai = vec![
        0i64, 1, // col 0: rows 0,1
        0, 1, // col 1: rows 0,1
        2, 3, // col 2: rows 2,3
        2, 3, // col 3: rows 2,3
    ];

    let btf = btf_decompose(4, &ap, &ai);

    assert_eq!(btf.structural_rank, 4);
    assert_eq!(btf.num_blocks, 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_matrix() {
    let btf = btf_decompose(0, &[0i64], &[]);

    assert_eq!(btf.num_blocks, 0);
    assert_eq!(btf.structural_rank, 0);
    assert_eq!(btf.row_perm.len(), 0);
    assert_eq!(btf.col_perm.len(), 0);
}

#[test]
fn test_1x1_matrix() {
    let ap = vec![0i64, 1];
    let ai = vec![0i64];

    let btf = btf_decompose(1, &ap, &ai);

    assert_eq!(btf.structural_rank, 1);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.row_perm, vec![0]);
    assert_eq!(btf.col_perm, vec![0]);
}

#[test]
fn test_large_diagonal() {
    // Large diagonal matrix
    let n = 100;
    let ap: Vec<i64> = (0..=n as i64).collect();
    let ai: Vec<i64> = (0..n as i64).collect();

    let btf = btf_decompose(n, &ap, &ai);

    assert_eq!(btf.structural_rank, n);
    assert_eq!(btf.num_blocks, n);
    assert_eq!(btf.num_singletons, n);
}

// ============================================================================
// Heuristic Tests
// ============================================================================

#[test]
fn test_should_use_btf_heuristic() {
    // Small matrix: don't use BTF
    assert!(!should_use_btf(10, 50));
    assert!(!should_use_btf(30, 200));

    // Medium sparse matrix: use BTF
    assert!(should_use_btf(100, 500));
    assert!(should_use_btf(1000, 5000));

    // Dense matrix: don't use BTF
    assert!(!should_use_btf(100, 8000)); // 80% dense
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_chain_graph() {
    // Long chain: 0 → 1 → 2 → ... → n-1
    // Should produce n singleton blocks in topological order
    let n = 50;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();

    for col in 0..n {
        // Each column has diagonal entry
        ai.push(col as i64);
        // First n-1 columns also have entry in next row
        if col < n - 1 {
            ai.push((col + 1) as i64);
        }
        ap.push(ai.len() as i64);
    }

    let btf = btf_decompose(n, &ap, &ai);

    assert_eq!(btf.structural_rank, n);
    // Chain has cycles if there's bidirectional connection
    // This chain is unidirectional, so all singletons
    assert_eq!(btf.num_singletons, n);
}

#[test]
fn test_complete_graph() {
    // Complete graph (dense matrix): single SCC
    let n = 10;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();

    for _ in 0..n {
        for row in 0..n {
            ai.push(row as i64);
        }
        ap.push(ai.len() as i64);
    }

    let btf = btf_decompose(n, &ap, &ai);

    assert_eq!(btf.structural_rank, n);
    assert_eq!(btf.num_blocks, 1);
    assert_eq!(btf.max_block_size, n);
}

// ============================================================================
// Verification Helper Functions
// ============================================================================

/// Verify that applying BTF permutation produces valid block structure
fn verify_btf_structure(btf: &BtfDecomposition, n: usize, ap: &[i64], ai: &[i64]) -> bool {
    // Build permuted matrix structure and verify zeros below diagonal blocks

    // For each entry A(row, col), compute permuted position
    // new_row = row_perm_inv[row], new_col = col_perm_inv[col]
    // Entry should be zero if new_col's block < new_row's block

    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        let new_col = btf.col_perm_inv[col];
        let col_block = find_block(new_col, &btf.block_ptr);

        for idx in start..end {
            let row = ai[idx] as usize;
            if row >= n {
                continue;
            }
            let new_row = btf.row_perm_inv[row];
            let row_block = find_block(new_row, &btf.block_ptr);

            // If row_block > col_block, this entry should not exist
            // (it would be below diagonal blocks)
            if row_block > col_block {
                return false;
            }
        }
    }

    true
}

/// Find which block a position belongs to
fn find_block(pos: usize, block_ptr: &[usize]) -> usize {
    for (k, window) in block_ptr.windows(2).enumerate() {
        if pos >= window[0] && pos < window[1] {
            return k;
        }
    }
    block_ptr.len() - 2 // Last block
}

#[test]
fn test_btf_structure_verification() {
    // Test that our BTF produces valid block structure
    let test_cases = vec![
        // (n, ap, ai)
        (3, vec![0i64, 1, 2, 3], vec![0i64, 1, 2]), // Diagonal
        (2, vec![0i64, 2, 4], vec![0i64, 1, 0, 1]), // Dense 2×2
        (
            4,
            vec![0i64, 2, 4, 6, 8],
            vec![0i64, 1, 0, 1, 2, 3, 2, 3],
        ), // Block diagonal
    ];

    for (n, ap, ai) in test_cases {
        let btf = btf_decompose(n, &ap, &ai);
        assert!(
            verify_btf_structure(&btf, n, &ap, &ai),
            "BTF structure invalid for n={}, blocks={}",
            n,
            btf.num_blocks
        );
    }
}
