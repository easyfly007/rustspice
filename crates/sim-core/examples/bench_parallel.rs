//! Benchmark: Sequential vs Parallel Native KLU factorization
//!
//! Tests with:
//! 1. Real circuit matrices from SuiteSparse (adder_dcop, circuit_1, rajat01)
//! 2. Large synthetic 2D grid matrices (100x100 to 500x500)
//!
//! Each test does 100 refactorizations to amortize thread creation overhead.
//! Reports per-refactor median time for fair core-computation comparison.
//!
//! Run with:
//!   cargo run -p sim-core --release --features parallel --example bench_parallel

use sim_core::native_klu::NativeKluSolver;
use sim_core::solver::LinearSolver;
use std::path::PathBuf;
use std::time::Instant;

// ============================================================================
// Matrix Market Parser (from bbd_benchmark_tests.rs)
// ============================================================================

struct CscMatrix {
    n: usize,
    ap: Vec<i64>,
    ai: Vec<i64>,
    ax: Vec<f64>,
}

fn parse_matrix_market(path: &std::path::Path) -> CscMatrix {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    let mut lines = content.lines();

    let header = lines.next().expect("Empty file");
    let header_lower = header.to_lowercase();
    let is_pattern = header_lower.contains("pattern");
    let is_symmetric = header_lower.contains("symmetric");

    let mut size_line = "";
    for line in lines.by_ref() {
        if !line.starts_with('%') {
            size_line = line;
            break;
        }
    }

    let parts: Vec<&str> = size_line.split_whitespace().collect();
    let nrows: usize = parts[0].parse().unwrap();
    let ncols: usize = parts[1].parse().unwrap();
    assert_eq!(nrows, ncols);
    let n = nrows;

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
        let row: usize = parts[0].parse::<usize>().unwrap() - 1;
        let col: usize = parts[1].parse::<usize>().unwrap() - 1;
        let val: f64 = if is_pattern {
            if row == col { 10.0 } else { -0.1 }
        } else {
            parts[2].parse().unwrap()
        };
        entries.push((row, col, val));
        if is_symmetric && row != col {
            entries.push((col, row, val));
        }
    }

    entries.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

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

    let mut ap = vec![0i64; n + 1];
    let mut ai = Vec::with_capacity(deduped.len());
    let mut ax = Vec::with_capacity(deduped.len());
    for &(row, col, val) in &deduped {
        ap[col + 1] += 1;
        ai.push(row as i64);
        ax.push(val);
    }
    for i in 1..=n {
        ap[i] += ap[i - 1];
    }

    CscMatrix { n, ap, ai, ax }
}

// ============================================================================
// Grid Mesh Generator
// ============================================================================

fn build_grid_mesh(grid_size: usize, scale: f64) -> CscMatrix {
    let n = grid_size * grid_size;
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();

    for row in 0..grid_size {
        for col_g in 0..grid_size {
            let node = row * grid_size + col_g;
            let mut diag = 0.0;
            if row > 0 {
                entries.push((node, (row - 1) * grid_size + col_g, -1.0 * scale));
                diag += 1.0;
            }
            if row < grid_size - 1 {
                entries.push((node, (row + 1) * grid_size + col_g, -1.0 * scale));
                diag += 1.0;
            }
            if col_g > 0 {
                entries.push((node, row * grid_size + (col_g - 1), -1.0 * scale));
                diag += 1.0;
            }
            if col_g < grid_size - 1 {
                entries.push((node, row * grid_size + (col_g + 1), -1.0 * scale));
                diag += 1.0;
            }
            entries.push((node, node, (diag + 0.1) * scale));
        }
    }

    // COO → CSC
    let mut col_counts = vec![0usize; n];
    for &(_, c, _) in &entries {
        col_counts[c] += 1;
    }
    let mut ap = vec![0i64; n + 1];
    for c in 0..n {
        ap[c + 1] = ap[c] + col_counts[c] as i64;
    }
    let nnz = ap[n] as usize;
    let mut ai = vec![0i64; nnz];
    let mut ax = vec![0.0f64; nnz];
    let mut pos: Vec<usize> = (0..n).map(|c| ap[c] as usize).collect();
    for &(r, c, v) in &entries {
        let idx = pos[c];
        ai[idx] = r as i64;
        ax[idx] = v;
        pos[c] += 1;
    }
    for c in 0..n {
        let start = ap[c] as usize;
        let end = ap[c + 1] as usize;
        let mut pairs: Vec<(i64, f64)> = (start..end).map(|i| (ai[i], ax[i])).collect();
        pairs.sort_by_key(|&(r, _)| r);
        for (i, &(r, v)) in pairs.iter().enumerate() {
            ai[start + i] = r;
            ax[start + i] = v;
        }
    }

    CscMatrix { n, ap, ai, ax }
}

/// Scale all values in a CSC matrix by a factor
fn scale_ax(ax: &[f64], scale: f64) -> Vec<f64> {
    ax.iter().map(|&v| v * scale).collect()
}

// ============================================================================
// Benchmark Runner
// ============================================================================

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times.len() / 2;
    if times.len() % 2 == 0 {
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[mid]
    }
}

fn run_benchmark(label: &str, mat: &CscMatrix, num_refactors: usize) {
    let n = mat.n;
    let nnz = mat.ap[n] as usize;
    println!("=== {} | n = {}, nnz = {}, avg nnz/col = {:.1} ===",
        label, n, nnz, nnz as f64 / n as f64);

    // Build scaled value sets (same pattern, different values)
    let ax_sets: Vec<Vec<f64>> = (0..num_refactors)
        .map(|i| scale_ax(&mat.ax, 1.0 + 0.01 * (i as f64)))
        .collect();

    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 100) as f64 * 0.01).collect();

    // ----- Sequential -----
    let mut seq_solver = NativeKluSolver::new(n);
    seq_solver.set_btf(false);
    seq_solver.set_parallel_threads(0);

    // Initial factor
    seq_solver.factor(&mat.ap, &mat.ai, &mat.ax).unwrap();

    let mut seq_times = Vec::with_capacity(num_refactors);
    for ax_i in &ax_sets {
        let t0 = Instant::now();
        seq_solver.factor(&mat.ap, &mat.ai, ax_i).unwrap();
        seq_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let seq_total: f64 = seq_times.iter().sum();
    let seq_median = median(&mut seq_times);

    let mut rhs_seq = b.clone();
    seq_solver.solve(&mut rhs_seq).unwrap();

    // ----- Parallel -----
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(2);

    let mut par_solver = NativeKluSolver::new(n);
    par_solver.set_btf(false);
    par_solver.set_parallel_threads(num_cpus);

    // Initial factor (sequential, establishes pivots)
    par_solver.factor(&mat.ap, &mat.ai, &mat.ax).unwrap();

    let mut par_times = Vec::with_capacity(num_refactors);
    for ax_i in &ax_sets {
        let t0 = Instant::now();
        par_solver.factor(&mat.ap, &mat.ai, ax_i).unwrap();
        par_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let par_total: f64 = par_times.iter().sum();
    let par_median = median(&mut par_times);

    let mut rhs_par = b.clone();
    par_solver.solve(&mut rhs_par).unwrap();

    // ----- Compare -----
    let mut max_diff = 0.0f64;
    for i in 0..n {
        max_diff = max_diff.max((rhs_par[i] - rhs_seq[i]).abs());
    }

    println!();
    println!("  Sequential  ({:>3} refactors): total {:>10.1} ms, median {:>10.3} ms/refactor",
        num_refactors, seq_total, seq_median);
    println!("  Parallel    ({:>3} refactors): total {:>10.1} ms, median {:>10.3} ms/refactor  ({} threads)",
        num_refactors, par_total, par_median, num_cpus);

    let speedup = seq_median / par_median;
    if speedup >= 1.0 {
        println!("  >>> {:.2}x FASTER (parallel)", speedup);
    } else {
        println!("  >>> {:.2}x slower (parallel)", 1.0 / speedup);
    }
    println!("  Solution match: max|diff| = {:.2e}  {}",
        max_diff, if max_diff < 1e-6 { "OK" } else { "MISMATCH" });
    println!();
    println!("------------------------------------------------------------");
    println!();
}

fn main() {
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("================================================================");
    println!("  Native KLU: Sequential vs Parallel Refactorization Benchmark");
    println!("  CPU threads: {}", num_cpus);
    println!("  100 refactors per test to amortize thread creation overhead");
    println!("================================================================");
    println!();

    // ----- Real circuit matrices (SuiteSparse) -----
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("..").join("tests").join("fixtures").join("matrices");

    println!(">>> REAL CIRCUIT MATRICES (SuiteSparse)");
    println!();

    for (name, file, expected_n) in [
        ("adder_dcop_01 (DC op of adder)", "adder_dcop_01.mtx", 1813),
        ("circuit_1 (circuit DAE)", "circuit_1.mtx", 2624),
        ("rajat01 (circuit pattern)", "rajat01.mtx", 6833),
    ] {
        let path = fixtures.join(file);
        if !path.exists() {
            println!("  Skipping {}: file not found", name);
            continue;
        }
        let mat = parse_matrix_market(&path);
        assert_eq!(mat.n, expected_n);
        run_benchmark(name, &mat, 100);
    }

    // ----- Large synthetic grid matrices -----
    println!(">>> LARGE 2D GRID MATRICES (synthetic, standard sparse benchmark)");
    println!();

    for &grid_size in &[100, 200, 300, 500] {
        let n = grid_size * grid_size;
        let label = format!("Grid {}x{}", grid_size, grid_size);
        let mat = build_grid_mesh(grid_size, 1.0);
        assert_eq!(mat.n, n);
        let refactors = if n > 100_000 { 20 } else { 100 };
        run_benchmark(&label, &mat, refactors);
    }
}
