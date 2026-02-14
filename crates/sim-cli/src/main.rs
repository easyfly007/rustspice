use std::env;
use std::path::{Path, PathBuf};

use sim_core::analysis::AnalysisPlan;
use sim_core::circuit::{AcSweepType, AnalysisCmd};
use sim_core::engine::Engine;
use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist_file};
use sim_core::result_store::{AnalysisType, ResultStore, RunStatus};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_help() {
    println!(
        r#"RustSpice SPICE Circuit Simulator

USAGE:
    sim-cli <NETLIST> [OPTIONS]

ARGS:
    <NETLIST>               Path to SPICE netlist file

OPTIONS:
    -h, --help              Print help information
    -V, --version           Print version information
    -o, --output <PATH>     Write results to output file
    -f, --format <FORMAT>   Output format: psf, raw, json, csv (default: psf)
    -a, --analysis <TYPE>   Analysis type: op, dc, tran, ac (default: from netlist or op)
    --dc-source <NAME>      DC sweep source name
    --dc-start <VALUE>      DC sweep start voltage
    --dc-stop <VALUE>       DC sweep stop voltage
    --dc-step <VALUE>       DC sweep step size
    --ac-sweep <TYPE>       AC sweep type: dec, oct, lin (default: dec)
    --ac-points <N>         AC points per decade/octave or total (default: 10)
    --ac-fstart <FREQ>      AC start frequency in Hz (default: 1)
    --ac-fstop <FREQ>       AC stop frequency in Hz (default: 1e6)
    --precision <N>         Output precision (1-15 significant digits, default: 6)

EXAMPLES:
    sim-cli circuit.cir                          # Run analysis from netlist
    sim-cli circuit.cir -o out.psf               # Export to PSF file
    sim-cli circuit.cir -o out.raw -f raw        # Export to ngspice raw format
    sim-cli circuit.cir -o out.json -f json      # Export to JSON format
    sim-cli circuit.cir -o out.csv -f csv        # Export to CSV format
    sim-cli circuit.cir -a dc --dc-source V1 \
        --dc-start 0 --dc-stop 5 --dc-step 0.1   # DC sweep
    sim-cli circuit.cir -a tran                  # Transient analysis
    sim-cli circuit.cir -a ac --ac-sweep dec \
        --ac-points 10 --ac-fstart 1 --ac-fstop 1e6  # AC analysis"#
    );
}

fn print_version() {
    println!("rustspice {}", VERSION);
}

/// Output format for simulation results.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Psf,
    Raw,
    Json,
    Csv,
}

fn main() {
    let all_args: Vec<String> = env::args().collect();
    eprintln!("command: {}", all_args[0]);
    eprintln!("args: {:?}", &all_args[1..]);
    let mut args = env::args().skip(1).peekable();
    let mut netlist_path: Option<String> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut output_format: OutputFormat = OutputFormat::Psf;
    let mut analysis: Option<String> = None;
    let mut dc_source: Option<String> = None;
    let mut dc_start: Option<f64> = None;
    let mut dc_stop: Option<f64> = None;
    let mut dc_step: Option<f64> = None;
    let mut ac_sweep: Option<String> = None;
    let mut ac_points: Option<usize> = None;
    let mut ac_fstart: Option<f64> = None;
    let mut ac_fstop: Option<f64> = None;
    let mut precision: usize = 6;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--version" | "-V" => {
                print_version();
                std::process::exit(0);
            }
            "--output" | "-o" | "--psf" => {
                let Some(path) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                output_path = Some(PathBuf::from(path));
            }
            "--format" | "-f" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                output_format = match value.to_ascii_lowercase().as_str() {
                    "psf" => OutputFormat::Psf,
                    "raw" => OutputFormat::Raw,
                    "json" => OutputFormat::Json,
                    "csv" => OutputFormat::Csv,
                    _ => {
                        eprintln!("unknown format: {} (expected: psf, raw, json, csv)", value);
                        std::process::exit(2);
                    }
                };
            }
            "--analysis" | "-a" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                analysis = Some(value.to_ascii_lowercase());
            }
            "--dc-source" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                dc_source = Some(value);
            }
            "--dc-start" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                dc_start = value.parse::<f64>().ok();
            }
            "--dc-stop" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                dc_stop = value.parse::<f64>().ok();
            }
            "--dc-step" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                dc_step = value.parse::<f64>().ok();
            }
            "--precision" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                precision = match value.parse::<usize>() {
                    Ok(p) if (1..=15).contains(&p) => p,
                    _ => {
                        eprintln!("precision must be between 1 and 15");
                        std::process::exit(2);
                    }
                };
            }
            "--ac-sweep" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                ac_sweep = Some(value.to_ascii_lowercase());
            }
            "--ac-points" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                ac_points = value.parse::<usize>().ok();
            }
            "--ac-fstart" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                ac_fstart = parse_number_with_suffix(&value).or_else(|| value.parse().ok());
            }
            "--ac-fstop" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for {}", arg);
                    std::process::exit(2);
                };
                ac_fstop = parse_number_with_suffix(&value).or_else(|| value.parse().ok());
            }
            _ => {
                if netlist_path.is_none() {
                    netlist_path = Some(arg);
                } else if output_path.is_none() {
                    // backward compatibility: second positional as output path
                    output_path = Some(PathBuf::from(arg));
                } else {
                    eprintln!("unexpected argument: {}", arg);
                    std::process::exit(2);
                }
            }
        }
    }

    let Some(netlist_path) = netlist_path else {
        eprintln!("usage: sim-cli <netlist> [--psf <path>]");
        std::process::exit(2);
    };

    let path = Path::new(&netlist_path);
    if !path.exists() {
        eprintln!("netlist not found: {}", netlist_path);
        std::process::exit(2);
    }

    let ast = parse_netlist_file(path);
    if !ast.errors.is_empty() {
        eprintln!("netlist parse errors:");
        for err in &ast.errors {
            eprintln!("  line {}: {}", err.line, err.message);
        }
        std::process::exit(2);
    }

    let elab = elaborate_netlist(&ast);
    if elab.error_count > 0 {
        eprintln!("netlist elaboration errors: {}", elab.error_count);
        std::process::exit(2);
    }

    let circuit = build_circuit(&ast, &elab);
    let (cmd, sweep) = select_analysis(
        &analysis,
        &circuit,
        dc_source,
        dc_start,
        dc_stop,
        dc_step,
        ac_sweep,
        ac_points,
        ac_fstart,
        ac_fstop,
    );

    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    if let Some(sweep) = sweep {
        run_dc_sweep(&mut engine, &mut store, cmd, sweep.clone(), output_path.as_deref(), output_format, precision);
    } else {
        let plan = AnalysisPlan { cmd };
        let run_id = engine.run_with_store(&plan, &mut store);
        let run = &store.runs[run_id.0];

        if !matches!(run.status, RunStatus::Converged) {
            eprintln!("run failed: status={:?} message={:?}", run.status, run.message);
            std::process::exit(1);
        }

        // Print results based on analysis type
        match run.analysis {
            AnalysisType::Tran => {
                println!("tran status: {:?} steps={}", run.status, run.iterations);
                println!("Final values:");
                for (idx, name) in run.node_names.iter().enumerate() {
                    let value = run.solution.get(idx).copied().unwrap_or(0.0);
                    println!("  V({}) = {:.*e}", name, precision, value);
                }
            }
            AnalysisType::Ac => {
                println!("ac status: {:?} frequency_points={}", run.status, run.ac_frequencies.len());
                // Print DC operating point
                println!("DC Operating Point:");
                for (idx, name) in run.node_names.iter().enumerate() {
                    let value = run.solution.get(idx).copied().unwrap_or(0.0);
                    println!("  V({}) = {:.*e}", name, precision, value);
                }
                // Print first and last frequency results
                if !run.ac_frequencies.is_empty() {
                    println!("\nAC Results at f = {:.*e} Hz:", precision, run.ac_frequencies[0]);
                    if let Some(sol) = run.ac_solutions.first() {
                        for (idx, name) in run.node_names.iter().enumerate() {
                            if let Some((mag_db, phase_deg)) = sol.get(idx) {
                                println!("  V({}) = {:.*} dB, {:.*}°", name, precision, mag_db, precision, phase_deg);
                            }
                        }
                    }
                    if run.ac_frequencies.len() > 1 {
                        let last_idx = run.ac_frequencies.len() - 1;
                        println!("\nAC Results at f = {:.*e} Hz:", precision, run.ac_frequencies[last_idx]);
                        if let Some(sol) = run.ac_solutions.last() {
                            for (idx, name) in run.node_names.iter().enumerate() {
                                if let Some((mag_db, phase_deg)) = sol.get(idx) {
                                    println!("  V({}) = {:.*} dB, {:.*}°", name, precision, mag_db, precision, phase_deg);
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                println!("run status: {:?} iterations={}", run.status, run.iterations);
                for (idx, name) in run.node_names.iter().enumerate() {
                    let value = run.solution.get(idx).copied().unwrap_or(0.0);
                    println!("V({}) = {:.*e}", name, precision, value);
                }
            }
        }

        if let Some(path) = output_path {
            let write_result = match output_format {
                OutputFormat::Psf => match run.analysis {
                    AnalysisType::Ac => {
                        sim_core::psf::write_psf_ac(
                            &run.ac_frequencies,
                            &run.node_names,
                            &run.ac_solutions,
                            &path,
                            precision,
                        )
                    }
                    AnalysisType::Tran => {
                        sim_core::psf::write_psf_tran(
                            &run.tran_times,
                            &run.node_names,
                            &run.tran_solutions,
                            &path,
                            precision,
                        )
                    }
                    _ => store.write_psf_text(run_id, &path, precision),
                },
                OutputFormat::Raw => match run.analysis {
                    AnalysisType::Ac => {
                        sim_core::raw::write_raw_ac(
                            &run.ac_frequencies,
                            &run.node_names,
                            &run.ac_solutions,
                            &path,
                            precision,
                        )
                    }
                    AnalysisType::Tran => {
                        sim_core::raw::write_raw_tran(
                            &run.tran_times,
                            &run.node_names,
                            &run.tran_solutions,
                            &path,
                            precision,
                        )
                    }
                    _ => sim_core::raw::write_raw_op(run, &path, precision),
                },
                OutputFormat::Json => match run.analysis {
                    AnalysisType::Ac => {
                        sim_core::json_export::write_json_ac(
                            &run.ac_frequencies,
                            &run.node_names,
                            &run.ac_solutions,
                            &path,
                            precision,
                        )
                    }
                    AnalysisType::Tran => {
                        sim_core::json_export::write_json_tran(
                            &run.tran_times,
                            &run.node_names,
                            &run.tran_solutions,
                            &path,
                            precision,
                        )
                    }
                    _ => sim_core::json_export::write_json_op(run, &path, precision),
                },
                OutputFormat::Csv => match run.analysis {
                    AnalysisType::Ac => {
                        sim_core::csv_export::write_csv_ac(
                            &run.ac_frequencies,
                            &run.node_names,
                            &run.ac_solutions,
                            &path,
                            precision,
                        )
                    }
                    AnalysisType::Tran => {
                        sim_core::csv_export::write_csv_tran(
                            &run.tran_times,
                            &run.node_names,
                            &run.tran_solutions,
                            &path,
                            precision,
                        )
                    }
                    _ => sim_core::csv_export::write_csv_op(run, &path, precision),
                },
            };
            if let Err(err) = write_result {
                eprintln!("failed to write output: {}", err);
                std::process::exit(1);
            }
            let format_name = match output_format {
                OutputFormat::Psf => "psf",
                OutputFormat::Raw => "raw",
                OutputFormat::Json => "json",
                OutputFormat::Csv => "csv",
            };
            println!("{} written: {}", format_name, path.display());
        }
    }
}

#[derive(Clone)]
struct DcSweep {
    source: String,
    start: f64,
    stop: f64,
    step: f64,
}

fn select_analysis(
    analysis: &Option<String>,
    circuit: &sim_core::circuit::Circuit,
    dc_source: Option<String>,
    dc_start: Option<f64>,
    dc_stop: Option<f64>,
    dc_step: Option<f64>,
    ac_sweep: Option<String>,
    ac_points: Option<usize>,
    ac_fstart: Option<f64>,
    ac_fstop: Option<f64>,
) -> (AnalysisCmd, Option<DcSweep>) {
    let from_netlist = circuit.analysis.first().cloned();
    let analysis = analysis.as_deref();

    match analysis {
        Some("op") => (AnalysisCmd::Op, None),
        Some("dc") => {
            let sweep = build_dc_sweep(dc_source, dc_start, dc_stop, dc_step)
                .or_else(|| extract_dc_sweep(from_netlist));
            let Some(sweep) = sweep else {
                eprintln!("dc analysis requires source/start/stop/step or .dc in netlist");
                std::process::exit(2);
            };
            (
                AnalysisCmd::Dc {
                    source: sweep.source.clone(),
                    start: sweep.start,
                    stop: sweep.stop,
                    step: sweep.step,
                },
                Some(sweep),
            )
        }
        Some("tran") => {
            let cmd = match from_netlist {
                Some(AnalysisCmd::Tran {
                    tstep,
                    tstop,
                    tstart,
                    tmax,
                }) => AnalysisCmd::Tran {
                    tstep,
                    tstop,
                    tstart,
                    tmax,
                },
                _ => AnalysisCmd::Tran {
                    tstep: 1e-6,
                    tstop: 1e-5,
                    tstart: 0.0,
                    tmax: 1e-5,
                },
            };
            (cmd, None)
        }
        Some("ac") => {
            let cmd = build_ac_cmd(ac_sweep, ac_points, ac_fstart, ac_fstop)
                .or_else(|| extract_ac_cmd(from_netlist.clone()))
                .unwrap_or_else(|| AnalysisCmd::Ac {
                    sweep_type: AcSweepType::Dec,
                    points: 10,
                    fstart: 1.0,
                    fstop: 1e6,
                });
            (cmd, None)
        }
        _ => match from_netlist {
            Some(AnalysisCmd::Dc {
                source,
                start,
                stop,
                step,
            }) => {
                let sweep = DcSweep {
                    source: source.clone(),
                    start,
                    stop,
                    step,
                };
                (
                    AnalysisCmd::Dc {
                        source,
                        start,
                        stop,
                        step,
                    },
                    Some(sweep),
                )
            }
            Some(cmd) => (cmd, None),
            None => (AnalysisCmd::Op, None),
        },
    }
}

fn build_dc_sweep(
    source: Option<String>,
    start: Option<f64>,
    stop: Option<f64>,
    step: Option<f64>,
) -> Option<DcSweep> {
    match (source, start, stop, step) {
        (Some(source), Some(start), Some(stop), Some(step)) => Some(DcSweep {
            source,
            start,
            stop,
            step,
        }),
        _ => None,
    }
}

fn extract_dc_sweep(cmd: Option<AnalysisCmd>) -> Option<DcSweep> {
    match cmd {
        Some(AnalysisCmd::Dc {
            source,
            start,
            stop,
            step,
        }) => Some(DcSweep {
            source,
            start,
            stop,
            step,
        }),
        _ => None,
    }
}

fn build_ac_cmd(
    sweep: Option<String>,
    points: Option<usize>,
    fstart: Option<f64>,
    fstop: Option<f64>,
) -> Option<AnalysisCmd> {
    let sweep_type = match sweep.as_deref() {
        Some("dec") => AcSweepType::Dec,
        Some("oct") => AcSweepType::Oct,
        Some("lin") => AcSweepType::Lin,
        Some(_) => return None,
        None => AcSweepType::Dec,
    };
    Some(AnalysisCmd::Ac {
        sweep_type,
        points: points.unwrap_or(10),
        fstart: fstart.unwrap_or(1.0),
        fstop: fstop.unwrap_or(1e6),
    })
}

fn extract_ac_cmd(cmd: Option<AnalysisCmd>) -> Option<AnalysisCmd> {
    match cmd {
        Some(AnalysisCmd::Ac { .. }) => cmd,
        _ => None,
    }
}

fn parse_number_with_suffix(token: &str) -> Option<f64> {
    let lower = token.to_ascii_lowercase();
    let trimmed = lower.trim();
    let (num_str, multiplier) = if trimmed.ends_with("meg") {
        (&trimmed[..trimmed.len() - 3], 1e6)
    } else {
        let (value_part, suffix) = trimmed.split_at(trimmed.len().saturating_sub(1));
        match suffix {
            "f" => (value_part, 1e-15),
            "p" => (value_part, 1e-12),
            "n" => (value_part, 1e-9),
            "u" => (value_part, 1e-6),
            "m" => (value_part, 1e-3),
            "k" => (value_part, 1e3),
            "g" => (value_part, 1e9),
            "t" => (value_part, 1e12),
            _ => (trimmed, 1.0),
        }
    };

    if let Ok(num) = num_str.parse::<f64>() {
        Some(num * multiplier)
    } else {
        None
    }
}

fn run_dc_sweep(
    engine: &mut Engine,
    store: &mut ResultStore,
    _cmd: AnalysisCmd,
    sweep: DcSweep,
    output_path: Option<&Path>,
    output_format: OutputFormat,
    precision: usize,
) {
    if sweep.step <= 0.0 {
        eprintln!("dc step must be > 0");
        std::process::exit(2);
    }
    println!(
        "dc sweep: {} from {} to {} step {}",
        sweep.source, sweep.start, sweep.stop, sweep.step
    );

    let mut sweep_values: Vec<f64> = Vec::new();
    let mut sweep_results: Vec<Vec<f64>> = Vec::new();
    let mut node_names: Vec<String> = Vec::new();

    let mut value = sweep.start;
    let mut guard = 0usize;
    while value <= sweep.stop + sweep.step * 0.5 {
        apply_dc_source(engine, &sweep.source, value);
        // Use Op analysis for each sweep point (not Dc which runs its own sweep)
        let plan = AnalysisPlan { cmd: AnalysisCmd::Op };
        let run_id = engine.run_with_store(&plan, store);
        let run = &store.runs[run_id.0];
        if !matches!(run.status, RunStatus::Converged) {
            eprintln!(
                "dc sweep failed at {}={}: status={:?} message={:?}",
                sweep.source, value, run.status, run.message
            );
            std::process::exit(1);
        }

        // Capture node names from first run
        if node_names.is_empty() {
            node_names = run.node_names.clone();
        }

        // Collect sweep data
        sweep_values.push(value);
        sweep_results.push(run.solution.clone());

        // Print to stdout
        print!("{}={:.*e}", sweep.source, precision, value);
        for (idx, name) in run.node_names.iter().enumerate() {
            let v = run.solution.get(idx).copied().unwrap_or(0.0);
            print!(" V({})={:.*e}", name, precision, v);
        }
        println!();

        value += sweep.step;
        guard += 1;
        if guard > 1_000_000 {
            eprintln!("dc sweep aborted: too many steps");
            std::process::exit(2);
        }
    }

    // Write output if requested
    if let Some(path) = output_path {
        let write_result = match output_format {
            OutputFormat::Psf => sim_core::psf::write_psf_sweep(
                &sweep.source,
                &sweep_values,
                &node_names,
                &sweep_results,
                path,
                precision,
            ),
            OutputFormat::Raw => sim_core::raw::write_raw_sweep(
                &sweep.source,
                &sweep_values,
                &node_names,
                &sweep_results,
                path,
                precision,
            ),
            OutputFormat::Json => sim_core::json_export::write_json_sweep(
                &sweep.source,
                &sweep_values,
                &node_names,
                &sweep_results,
                path,
                precision,
            ),
            OutputFormat::Csv => sim_core::csv_export::write_csv_sweep(
                &sweep.source,
                &sweep_values,
                &node_names,
                &sweep_results,
                path,
                precision,
            ),
        };
        if let Err(err) = write_result {
            eprintln!("failed to write output: {}", err);
            std::process::exit(1);
        }
        let format_name = match output_format {
            OutputFormat::Psf => "psf",
            OutputFormat::Raw => "raw",
            OutputFormat::Json => "json",
            OutputFormat::Csv => "csv",
        };
        println!("{} written: {}", format_name, path.display());
    }
}

fn apply_dc_source(engine: &mut Engine, source: &str, value: f64) {
    let mut found = false;
    for inst in &mut engine.circuit.instances.instances {
        if inst.name.eq_ignore_ascii_case(source) {
            inst.value = Some(value.to_string());
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("dc source not found: {}", source);
        std::process::exit(2);
    }
}
