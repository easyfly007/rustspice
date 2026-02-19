//! BBD Solver Benchmark Tests
//!
//! Tests the BBD solver against real-world circuit matrices from:
//! 1. SuiteSparse Matrix Collection (Matrix Market format)
//! 2. BeGAN Power Grid Benchmarks (SPICE netlists)
//!
//! These tests verify correctness by comparing BBD results against
//! SparseLU on matrices with 1,000+ dimensions.

use std::path::PathBuf;
use std::time::Instant;

use sim_core::solver::{create_solver, SolverType};

// ============================================================================
// Matrix Market Parser
// ============================================================================

/// Sparse matrix in CSC (Compressed Sparse Column) format
struct CscMatrix {
    n: usize,
    ap: Vec<i64>,
    ai: Vec<i64>,
    ax: Vec<f64>,
}

/// Parse a Matrix Market (.mtx) file into CSC format.
///
/// Supports:
/// - `coordinate real general` (row col value)
/// - `coordinate pattern general` (row col, values filled with diagonal dominance)
/// - `coordinate real symmetric` (lower triangle mirrored)
fn parse_matrix_market(path: &std::path::Path) -> CscMatrix {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    let mut lines = content.lines();

    // Parse header
    let header = lines.next().expect("Empty file");
    assert!(
        header.starts_with("%%MatrixMarket"),
        "Not a Matrix Market file"
    );
    let header_lower = header.to_lowercase();
    let is_pattern = header_lower.contains("pattern");
    let is_symmetric = header_lower.contains("symmetric");

    // Skip comments
    let mut size_line = "";
    for line in lines.by_ref() {
        if !line.starts_with('%') {
            size_line = line;
            break;
        }
    }

    // Parse dimensions
    let parts: Vec<&str> = size_line.split_whitespace().collect();
    assert!(parts.len() >= 3, "Invalid size line: {}", size_line);
    let nrows: usize = parts[0].parse().unwrap();
    let ncols: usize = parts[1].parse().unwrap();
    let _nnz: usize = parts[2].parse().unwrap();
    assert_eq!(
        nrows, ncols,
        "Matrix must be square, got {}x{}",
        nrows, ncols
    );
    let n = nrows;

    // Parse entries into COO (coordinate) format
    // Collect as (row, col, value) triples, 0-indexed
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let row: usize = parts[0].parse::<usize>().unwrap() - 1; // 1-indexed to 0-indexed
        let col: usize = parts[1].parse::<usize>().unwrap() - 1;
        let val: f64 = if is_pattern {
            // Pattern format: assign diagonal dominance for solvability
            if row == col {
                10.0
            } else {
                -0.1
            }
        } else {
            parts[2].parse().unwrap()
        };

        entries.push((row, col, val));

        // For symmetric matrices, mirror off-diagonal entries
        if is_symmetric && row != col {
            entries.push((col, row, val));
        }
    }

    // Convert COO to CSC
    // Sort by (col, row) for CSC construction
    entries.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    // Deduplicate: sum values for duplicate (row, col) pairs
    let mut deduped: Vec<(usize, usize, f64)> = Vec::new();
    for &(r, c, v) in &entries {
        if let Some(last) = deduped.last_mut() {
            if last.0 == r && last.1 == c {
                last.2 += v;
                continue;
            }
        }
        deduped.push((r, c, v));
    }

    // Build CSC arrays
    let mut ap = vec![0i64; n + 1];
    let mut ai = Vec::with_capacity(deduped.len());
    let mut ax = Vec::with_capacity(deduped.len());

    for &(row, col, val) in &deduped {
        assert!(
            col < n && row < n,
            "Entry ({},{}) out of bounds for n={}",
            row, col, n
        );
        ap[col + 1] += 1;
        ai.push(row as i64);
        ax.push(val);
    }

    // Cumulative sum for column pointers
    for i in 1..=n {
        ap[i] += ap[i - 1];
    }

    CscMatrix { n, ap, ai, ax }
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Compute relative residual ||Ax - b|| / ||b||
fn relative_residual(
    n: usize,
    ap: &[i64],
    ai: &[i64],
    ax: &[f64],
    x: &[f64],
    b: &[f64],
) -> f64 {
    let mut r = b.to_vec();
    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            r[row] -= ax[idx] * x[col];
        }
    }
    let r_norm: f64 = r.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if b_norm < 1e-30 {
        r_norm
    } else {
        r_norm / b_norm
    }
}

/// Solve Ax=b with a given solver, return (solution, solver_name, elapsed_ms).
fn solve_with(
    solver_type: SolverType,
    n: usize,
    ap: &[i64],
    ai: &[i64],
    ax: &[f64],
    rhs: &[f64],
) -> Result<(Vec<f64>, String, f64), String> {
    let mut solver = create_solver(solver_type, n);
    solver.prepare(n);

    let t0 = Instant::now();
    solver
        .analyze(ap, ai)
        .map_err(|e| format!("{} analyze failed: {}", solver.name(), e))?;
    solver
        .factor(ap, ai, ax)
        .map_err(|e| format!("{} factor failed: {}", solver.name(), e))?;
    let mut x = rhs.to_vec();
    solver
        .solve(&mut x)
        .map_err(|e| format!("{} solve failed: {}", solver.name(), e))?;
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    Ok((x, solver.name().to_string(), elapsed))
}

/// Run BBD vs SparseLU comparison on a CSC matrix and print diagnostics.
/// Asserts that both solvers agree within tolerance.
fn run_matrix_benchmark(name: &str, mat: &CscMatrix, tol: f64) {
    let rhs = generate_rhs(mat.n, 42);

    // SparseLU reference
    let (x_ref, ref_name, ref_ms) =
        solve_with(SolverType::SparseLu, mat.n, &mat.ap, &mat.ai, &mat.ax, &rhs)
            .expect("SparseLU solve failed");
    let res_ref = relative_residual(mat.n, &mat.ap, &mat.ai, &mat.ax, &x_ref, &rhs);

    // BBD solver
    let (x_bbd, bbd_name, bbd_ms) =
        solve_with(SolverType::Bbd, mat.n, &mat.ap, &mat.ai, &mat.ax, &rhs)
            .expect("BBD solve failed");
    let res_bbd = relative_residual(mat.n, &mat.ap, &mat.ai, &mat.ax, &x_bbd, &rhs);

    // SparseLU-BTF for additional comparison
    let (x_btf, btf_name, btf_ms) =
        solve_with(SolverType::SparseLuBtf, mat.n, &mat.ap, &mat.ai, &mat.ax, &rhs)
            .expect("SparseLU-BTF solve failed");
    let res_btf = relative_residual(mat.n, &mat.ap, &mat.ai, &mat.ax, &x_btf, &rhs);

    // Max difference between BBD and SparseLU
    let max_diff_bbd = x_ref
        .iter()
        .zip(x_bbd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    // Relative difference (normalized by solution magnitude)
    let x_norm: f64 = x_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel_diff = if x_norm > 1e-30 {
        max_diff_bbd / (x_norm / (mat.n as f64).sqrt())
    } else {
        max_diff_bbd
    };

    println!("=== {} ===", name);
    println!(
        "  Matrix: {}x{}, {} nnz, density={:.4}%",
        mat.n,
        mat.n,
        mat.ax.len(),
        100.0 * mat.ax.len() as f64 / (mat.n * mat.n) as f64
    );
    println!(
        "  {} residual: {:.2e}  ({:.1}ms)",
        ref_name, res_ref, ref_ms
    );
    println!(
        "  {} residual: {:.2e}  ({:.1}ms)",
        bbd_name, res_bbd, bbd_ms
    );
    println!(
        "  {} residual: {:.2e}  ({:.1}ms)",
        btf_name, res_btf, btf_ms
    );
    println!(
        "  Max |x_bbd - x_ref|: {:.2e}  (relative: {:.2e})",
        max_diff_bbd, rel_diff
    );

    // The key correctness check: BBD must agree with SparseLU
    // (both may have similar residuals on ill-conditioned matrices)
    assert!(
        rel_diff < tol,
        "{}: BBD vs SparseLU mismatch too large: rel_diff={:.2e} > tol={:.2e}",
        name,
        rel_diff,
        tol
    );

    // BBD residual should not be dramatically worse than SparseLU
    if res_ref > 0.0 {
        let residual_ratio = res_bbd / res_ref;
        println!("  Residual ratio (BBD/SparseLU): {:.2}", residual_ratio);
        assert!(
            residual_ratio < 100.0,
            "{}: BBD residual is {}x worse than SparseLU",
            name,
            residual_ratio
        );
    }
}

/// Generate a random-looking but deterministic RHS vector.
fn generate_rhs(n: usize, seed: u64) -> Vec<f64> {
    let mut rhs = vec![0.0; n];
    let mut state = seed;
    for v in rhs.iter_mut() {
        // Simple LCG PRNG
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
    }
    rhs
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
}

// ============================================================================
// SuiteSparse Matrix Market Tests
// ============================================================================

#[test]
fn bbd_benchmark_adder_dcop_01() {
    // Sandia/adder_dcop_01: 1813x1813, 11156 nnz
    // DC operating point of an adder circuit
    let path = fixtures_dir().join("matrices").join("adder_dcop_01.mtx");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let mat = parse_matrix_market(&path);
    assert_eq!(mat.n, 1813);
    run_matrix_benchmark("adder_dcop_01", &mat, 1e-3);
}

#[test]
fn bbd_benchmark_circuit_1() {
    // Bomhof/circuit_1: 2624x2624, 35823 nnz
    // Circuit DAE with BDF method & Newton
    let path = fixtures_dir().join("matrices").join("circuit_1.mtx");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let mat = parse_matrix_market(&path);
    assert_eq!(mat.n, 2624);
    run_matrix_benchmark("circuit_1", &mat, 1e-3);
}

#[test]
fn bbd_benchmark_rajat01() {
    // Rajat/rajat01: 6833x6833, 43250 nnz (pattern format)
    // Values are synthetically generated (diagonal dominant)
    let path = fixtures_dir().join("matrices").join("rajat01.mtx");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let mat = parse_matrix_market(&path);
    assert_eq!(mat.n, 6833);
    // Relaxed tolerance: pattern-only matrix with synthetic diagonal-dominant values;
    // BBD partitioning on synthetic values may produce larger disagreements.
    run_matrix_benchmark("rajat01", &mat, 0.1);
}

// ============================================================================
// BeGAN Power Grid Benchmark Tests (Full pipeline: SPICE → MNA → Solve)
// ============================================================================

/// Run a BeGAN SPICE netlist through full pipeline with BBD and reference solver.
fn run_began_benchmark(filename: &str, tol: f64) {
    use sim_core::analysis::AnalysisPlan;
    use sim_core::circuit::AnalysisCmd;
    use sim_core::engine::Engine;
    use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist_file};
    use sim_core::result_store::{ResultStore, RunStatus};

    let path = fixtures_dir().join("netlists").join(filename);
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let ast = parse_netlist_file(&path);
    assert!(
        ast.errors.is_empty(),
        "Parse errors in {}: {:?}",
        filename,
        ast.errors
    );

    let elab = elaborate_netlist(&ast);
    assert_eq!(
        elab.error_count, 0,
        "Elaboration errors in {}: {}",
        filename,
        elab.error_count
    );

    let circuit = build_circuit(&ast, &elab);
    let node_count = circuit.nodes.id_to_name.len();
    let instance_count = circuit.instances.instances.len();

    println!("=== {} ===", filename);
    println!("  Nodes: {}, Instances: {}", node_count, instance_count);

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Op,
    };

    // BBD solver
    let t0 = Instant::now();
    let mut engine_bbd = Engine::new(circuit.clone(), SolverType::Bbd);
    let mut store_bbd = ResultStore::new();
    let run_id_bbd = engine_bbd.run_with_store(&plan, &mut store_bbd);
    let bbd_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let run_bbd = &store_bbd.runs[run_id_bbd.0];

    assert!(
        matches!(run_bbd.status, RunStatus::Converged),
        "BBD solver did not converge on {}: {:?}",
        filename,
        run_bbd.status
    );
    println!(
        "  BBD: Converged ({:.1}ms), {} solution entries",
        bbd_ms,
        run_bbd.solution.len()
    );

    // SparseLU reference
    let t0 = Instant::now();
    let mut engine_ref = Engine::new(circuit.clone(), SolverType::SparseLu);
    let mut store_ref = ResultStore::new();
    let run_id_ref = engine_ref.run_with_store(&plan, &mut store_ref);
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let run_ref = &store_ref.runs[run_id_ref.0];

    assert!(
        matches!(run_ref.status, RunStatus::Converged),
        "SparseLU solver did not converge on {}: {:?}",
        filename,
        run_ref.status
    );
    println!("  SparseLU: Converged ({:.1}ms)", ref_ms);

    // Compare solutions
    let max_diff = run_bbd
        .solution
        .iter()
        .zip(run_ref.solution.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let max_val = run_ref
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    let rel_diff = if max_val > 1e-30 {
        max_diff / max_val
    } else {
        max_diff
    };

    println!(
        "  Max |V_bbd - V_ref|: {:.2e} (relative: {:.2e})",
        max_diff, rel_diff
    );

    // Print a few representative node voltages
    let n_print = 5.min(run_bbd.solution.len());
    for i in 0..n_print {
        let name = if i < run_ref.node_names.len() {
            &run_ref.node_names[i]
        } else {
            "?"
        };
        println!(
            "    V({}) = {:.6} (BBD) vs {:.6} (SparseLU)",
            name,
            run_bbd.solution[i],
            run_ref.solution[i]
        );
    }

    assert!(
        rel_diff < tol,
        "BBD vs SparseLU mismatch on {}: max_diff={:.2e}, rel={:.2e} > tol={:.2e}",
        filename,
        max_diff,
        rel_diff,
        tol
    );
}

#[test]
fn bbd_benchmark_began_small_pgrid() {
    // Small power grid: ~500 nodes, 555 resistors, 27 current sources
    run_began_benchmark("began_small_pgrid.sp", 1e-6);
}

#[test]
fn bbd_benchmark_began_dynamic_node() {
    // Medium power grid: ~19521 nodes, 21162 resistors, 4225 current sources
    run_began_benchmark("began_dynamic_node.sp", 1e-3);
}

#[test]
fn bbd_benchmark_began_ibex() {
    // Large power grid: ~25199 nodes, 27418 resistors, 5476 current sources
    // RISC-V ibex core power grid, ASAP7 technology
    run_began_benchmark("began_ibex.sp", 1e-3);
}
