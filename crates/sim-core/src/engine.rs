use crate::analysis::{
    estimate_lte_milne, AdaptiveStepController, AnalysisPlan, TimeStepConfig,
};
use crate::circuit::{AcSweepType, Circuit, DeviceKind};
use crate::complex_mna::ComplexMnaBuilder;
use crate::complex_solver::create_complex_solver;
use crate::mna::MnaBuilder;
use crate::result_store::{AnalysisType, ResultStore, RunId, RunResult, RunStatus};
use crate::solver::{create_solver, LinearSolver, SolverType};
use crate::stamp::{
    update_transient_state, update_transient_state_full, DeviceStamp, InstanceStamp,
    IntegrationMethod, TransientState,
};
use crate::newton::{debug_dump_newton_with_tag, run_newton_with_stepping, NewtonConfig};
use crate::waveform::{
    parse_pulse, parse_pwl, BreakpointManager, TransientSource, WaveformSpec,
};
use num_complex::Complex64;

#[cfg(feature = "va")]
use std::collections::HashMap;
#[cfg(feature = "va")]
use std::sync::Arc;
#[cfg(feature = "va")]
use crate::va_stamp::{va_stamp_dc, va_stamp_tran, va_stamp_ac};

pub struct Engine {
    pub circuit: Circuit,
    solver: Box<dyn LinearSolver>,
    solver_type: SolverType,
    /// VA/OSDI device instances, keyed by circuit instance name.
    #[cfg(feature = "va")]
    va_instances: HashMap<String, sim_va::OsdiInstance>,
    /// Loaded OSDI libraries (kept alive for the lifetime of the engine).
    #[cfg(feature = "va")]
    va_libraries: Vec<Arc<sim_va::OsdiLibrary>>,
    /// Shared RHS buffer for VA stamping (avoids re-allocation per Newton step).
    #[cfg(feature = "va")]
    va_rhs_buf: Vec<f64>,
}

impl Engine {
    /// Create an Engine with the specified solver type.
    pub fn new(circuit: Circuit, solver_type: SolverType) -> Self {
        let node_count = circuit.nodes.id_to_name.len();
        Self {
            circuit,
            solver: create_solver(solver_type, node_count),
            solver_type,
            #[cfg(feature = "va")]
            va_instances: HashMap::new(),
            #[cfg(feature = "va")]
            va_libraries: Vec::new(),
            #[cfg(feature = "va")]
            va_rhs_buf: Vec::new(),
        }
    }

    /// Create an Engine with the default solver (Dense).
    pub fn new_default(circuit: Circuit) -> Self {
        Self::new(circuit, SolverType::default())
    }

    /// Initialize VA/OSDI devices from the circuit's `.va_files`.
    ///
    /// This compiles `.va` files (via OpenVAF), loads the resulting `.osdi`
    /// shared libraries, creates `OsdiModel` and `OsdiInstance` wrappers,
    /// and stores them for use during simulation.
    ///
    /// Call this after constructing the Engine and before running analyses
    /// if the circuit uses Verilog-A devices.
    #[cfg(feature = "va")]
    pub fn init_va_devices(&mut self) -> Result<(), String> {
        use sim_va::{VaCompiler, OsdiLibrary, OsdiModel, OsdiInstance};

        if self.circuit.va_files.is_empty() {
            return Ok(());
        }

        let compiler = VaCompiler::new();

        // 1. Compile .va files and load .osdi libraries
        for va_path in &self.circuit.va_files {
            let osdi_path = compiler
                .compile_or_passthrough(va_path)
                .map_err(|e| e.to_string())?;
            let lib = Arc::new(OsdiLibrary::load(&osdi_path).map_err(|e| e.to_string())?);
            self.va_libraries.push(lib);
        }

        // 2. Build module lookup: module_name → library
        let mut module_lookup: HashMap<String, Arc<OsdiLibrary>> = HashMap::new();
        for lib in &self.va_libraries {
            for desc in &lib.descriptors {
                module_lookup.insert(desc.name.to_ascii_lowercase(), lib.clone());
            }
        }

        // 3. Create OsdiModel + OsdiInstance for each VA device instance
        // First, group instances by model to share OsdiModel objects
        let mut model_cache: HashMap<String, Arc<OsdiModel>> = HashMap::new();

        for inst in &self.circuit.instances.instances {
            let module_name = match &inst.kind {
                DeviceKind::VA { module_name } => module_name.to_ascii_lowercase(),
                _ => continue,
            };

            let library = match module_lookup.get(&module_name) {
                Some(lib) => lib.clone(),
                None => {
                    return Err(format!(
                        "VA module '{}' not found in loaded OSDI libraries",
                        module_name
                    ));
                }
            };

            // Get or create OsdiModel for this model
            let model_key = match &inst.model {
                Some(model_id) => {
                    if let Some(model_def) = self.circuit.models.models.get(model_id.0) {
                        model_def.name.clone()
                    } else {
                        module_name.clone()
                    }
                }
                None => module_name.clone(),
            };

            let osdi_model = if let Some(model) = model_cache.get(&model_key) {
                model.clone()
            } else {
                // Get model parameters
                let model_params = match &inst.model {
                    Some(model_id) => {
                        if let Some(model_def) = self.circuit.models.models.get(model_id.0) {
                            model_def.params.clone()
                        } else {
                            HashMap::new()
                        }
                    }
                    None => HashMap::new(),
                };

                let model = Arc::new(
                    OsdiModel::new(library, &module_name, &model_params)
                        .map_err(|e| e.to_string())?,
                );
                model_cache.insert(model_key, model.clone());
                model
            };

            // Map terminal node IDs (from circuit NodeId to MNA index)
            let terminal_ids: Vec<usize> = inst.nodes.iter().map(|n| n.0).collect();

            // Create instance with aux allocator for internal nodes
            let node_count = self.circuit.nodes.id_to_name.len();
            let osdi_inst = OsdiInstance::new(
                osdi_model,
                &inst.name,
                &terminal_ids,
                |_name| {
                    // Internal nodes will be allocated during stamping
                    // For now, we don't support internal nodes in VA devices
                    // TODO: integrate with MNA auxiliary variable allocation
                    node_count
                },
                &inst.params,
                27.0 + 273.15, // Default temperature: 300.15K (27°C)
            )
            .map_err(|e| e.to_string())?;

            self.va_instances.insert(inst.name.clone(), osdi_inst);
        }

        Ok(())
    }

    /// Reinitialize the solver when the circuit size changes.
    pub fn resize_solver(&mut self) {
        let node_count = self.circuit.nodes.id_to_name.len();
        self.solver = create_solver(self.solver_type, node_count);
    }

    /// Switch to a different solver type.
    pub fn set_solver_type(&mut self, solver_type: SolverType) {
        self.solver_type = solver_type;
        self.resize_solver();
    }

    pub fn run(&mut self, plan: &AnalysisPlan) {
        println!("engine: run {:?}", plan.cmd);
        match plan.cmd {
            crate::circuit::AnalysisCmd::Tran { .. } => self.run_tran(),
            _ => self.run_dc(),
        }
    }

    pub fn run_with_store(&mut self, plan: &AnalysisPlan, store: &mut ResultStore) -> RunId {
        let result = match &plan.cmd {
            crate::circuit::AnalysisCmd::Tran { tstep, tstop, tstart, tmax } => {
                self.run_tran_result_with_params(*tstep, *tstop, *tstart, *tmax)
            }
            crate::circuit::AnalysisCmd::Dc { source, start, stop, step } => {
                self.run_dc_sweep_result(source, *start, *stop, *step)
            }
            crate::circuit::AnalysisCmd::Ac { sweep_type, points, fstart, fstop } => {
                self.run_ac_result(*sweep_type, *points, *fstart, *fstop)
            }
            _ => self.run_dc_result(AnalysisType::Op),
        };
        store.add_run(result)
    }

    pub fn run_dc(&mut self) {
        let _ = self.run_dc_result(AnalysisType::Op);
    }

    pub fn run_tran(&mut self) {
        // Use default parameters for standalone run
        let _ = self.run_tran_result_with_params(1e-6, 1e-5, 0.0, 1e-5);
    }

    /// Perform a DC operating-point analysis and return the result.
    ///
    /// This function finds the steady-state (DC) solution of the circuit by:
    ///   1. Building the MNA (Modified Nodal Analysis) matrix and RHS vector
    ///      by stamping every device instance at the current operating point.
    ///   2. Pinning the ground node to 0 V so the matrix is non-singular.
    ///   3. Solving the nonlinear system with Newton-Raphson iteration
    ///      (with source stepping / gmin stepping for convergence aid).
    ///
    /// On return, the `RunResult` contains:
    ///   - `solution`: a `Vec<f64>` of node voltages (index = NodeId), valid
    ///     only when `status` is `Converged`.
    ///   - `node_names`: mapping from index to human-readable node name.
    ///   - `iterations`: number of Newton iterations consumed.
    ///   - `status`: `Converged`, `MaxIters`, or `Failed`.
    fn run_dc_result(&mut self, analysis: AnalysisType) -> RunResult {
        let config = NewtonConfig::default();
        let node_count = self.circuit.nodes.id_to_name.len();
        let mut x = vec![0.0; node_count];
        self.solver.prepare(node_count);
        let gnd = self.circuit.nodes.gnd_id.0;

        // Borrow VA state outside the closure to avoid borrow conflicts
        #[cfg(feature = "va")]
        let va_instances = &mut self.va_instances;
        #[cfg(feature = "va")]
        let va_rhs_buf = &mut self.va_rhs_buf;

        let result = run_newton_with_stepping(&config, &mut x, |x, gmin, source_scale| {
            let mut mna = MnaBuilder::new(node_count);
            for inst in &self.circuit.instances.instances {
                let stamp = InstanceStamp {
                    instance: inst.clone(),
                };
                let mut ctx = mna.context_with(gmin, source_scale);
                let _ = stamp.stamp_dc(&mut ctx, Some(x));
            }

            // Stamp VA devices
            #[cfg(feature = "va")]
            for (_name, va_inst) in va_instances.iter_mut() {
                let mut ctx = mna.context_with(gmin, source_scale);
                let _ = va_stamp_dc(va_inst, &mut ctx, Some(x), va_rhs_buf);
            }

            // Pin the ground node to avoid a singular matrix.
            mna.builder.insert(gnd, gnd, 1.0);
            let (ap, ai, ax) = mna.builder.finalize();
            (ap, ai, ax, mna.rhs, mna.builder.n)
        }, self.solver.as_mut());

        debug_dump_newton_with_tag("dc", &result);
        let status = match result.reason {
            crate::newton::NewtonExitReason::Converged => RunStatus::Converged,
            crate::newton::NewtonExitReason::MaxIters => RunStatus::MaxIters,
            crate::newton::NewtonExitReason::SolverFailure => RunStatus::Failed,
        };
        RunResult {
            id: RunId(0),
            analysis,
            status,
            iterations: result.iterations,
            node_names: self.circuit.nodes.id_to_name.clone(),
            solution: if matches!(status, RunStatus::Converged) {
                x
            } else {
                Vec::new()
            },
            message: result.message,
            sweep_var: None,
            sweep_values: Vec::new(),
            sweep_solutions: Vec::new(),
            tran_times: Vec::new(),
            tran_solutions: Vec::new(),
            ac_frequencies: Vec::new(),
            ac_solutions: Vec::new(),
        }
    }

    /// Run TRAN analysis with specified parameters and store waveform data
    ///
    /// This function performs transient analysis from `tstart` to `tstop` using
    /// adaptive time stepping with:
    /// - **LTE estimation** using Milne's Device (comparing BE and Trapezoidal solutions)
    /// - **PI Controller** for smooth step size adjustment
    /// - **Breakpoint handling** for PWL/PULSE source discontinuities
    /// - **Trapezoidal integration** as the primary method (2nd order accuracy)
    ///
    /// # Arguments
    /// * `tstep` - Suggested time step for output
    /// * `tstop` - Stop time
    /// * `tstart` - Start time (usually 0)
    /// * `tmax` - Maximum internal time step
    ///
    /// # Returns
    /// RunResult containing:
    /// - `tran_times`: Vector of time points where solutions were computed
    /// - `tran_solutions`: Vector of solution vectors at each time point
    /// - `solution`: Final solution at tstop
    fn run_tran_result_with_params(
        &mut self,
        tstep: f64,
        tstop: f64,
        tstart: f64,
        tmax: f64,
    ) -> RunResult {
        let node_count = self.circuit.nodes.id_to_name.len();
        let mut x = vec![0.0; node_count];

        // Apply initial conditions (.ic directive) as initial guess
        for (node_id, value) in &self.circuit.initial_conditions {
            if node_id.0 < node_count {
                x[node_id.0] = *value;
            }
        }

        let mut x_prev: Vec<f64>;

        // Initialize transient states for both methods
        let mut state_be = TransientState::default();
        state_be.method = IntegrationMethod::BackwardEuler;

        let mut state_trap = TransientState::default();
        state_trap.method = IntegrationMethod::Trapezoidal;

        self.solver.prepare(node_count);

        // Configuration
        let min_dt = (tstep * 1e-6).max(1e-15);
        let max_dt = tmax.min(tstop / 10.0);
        let abs_tol = 1e-9;
        let rel_tol = 1e-6;

        // Store config for reference (used in error messages)
        let _config = TimeStepConfig {
            tstep,
            tstop,
            tstart,
            tmax,
            min_dt,
            max_dt,
            abs_tol,
            rel_tol,
        };

        // Initialize adaptive step controller (PI controller)
        let mut step_controller = AdaptiveStepController::new(min_dt, max_dt);

        // Extract transient sources and breakpoints
        let transient_sources = self.extract_transient_sources();
        let mut breakpoint_mgr = BreakpointManager::new();
        breakpoint_mgr.configure_settling(5, 0.1);
        breakpoint_mgr.extract_from_sources(&transient_sources, tstop);

        // Time stepping state
        let mut t = tstart;
        let mut dt = tstep.min(max_dt);
        let mut _dt_prev = dt;
        let mut accepted_steps = 0;
        let mut _rejected_steps = 0;
        let mut consecutive_rejects = 0;
        let max_consecutive_rejects = 10;

        let mut final_status = RunStatus::Converged;
        let gnd = self.circuit.nodes.gnd_id.0;

        // Waveform storage
        let mut tran_times: Vec<f64> = Vec::new();
        let mut tran_solutions: Vec<Vec<f64>> = Vec::new();

        // Borrow VA state outside closures
        #[cfg(feature = "va")]
        let va_instances = &mut self.va_instances;
        #[cfg(feature = "va")]
        let va_rhs_buf = &mut self.va_rhs_buf;

        // ====================================================================
        // Run initial DC operating point (t=tstart)
        // ====================================================================
        let dc_result = run_newton_with_stepping(&NewtonConfig::default(), &mut x, |x, gmin, source_scale| {
            let mut mna = MnaBuilder::new(node_count);
            for inst in &self.circuit.instances.instances {
                let stamp = InstanceStamp { instance: inst.clone() };
                let mut ctx = mna.context_with(gmin, source_scale);
                let _ = stamp.stamp_dc(&mut ctx, Some(x));
            }
            #[cfg(feature = "va")]
            for (_name, va_inst) in va_instances.iter_mut() {
                let mut ctx = mna.context_with(gmin, source_scale);
                let _ = va_stamp_dc(va_inst, &mut ctx, Some(x), va_rhs_buf);
            }
            mna.builder.insert(gnd, gnd, 1.0);
            let (ap, ai, ax) = mna.builder.finalize();
            (ap, ai, ax, mna.rhs, mna.builder.n)
        }, self.solver.as_mut());

        debug_dump_newton_with_tag("tran_dc_op", &dc_result);

        if !dc_result.converged {
            return RunResult {
                id: RunId(0),
                analysis: AnalysisType::Tran,
                status: RunStatus::Failed,
                iterations: 0,
                node_names: self.circuit.nodes.id_to_name.clone(),
                solution: Vec::new(),
                message: Some("DC operating point failed to converge".to_string()),
                sweep_var: None,
                sweep_values: Vec::new(),
                sweep_solutions: Vec::new(),
                tran_times: Vec::new(),
                tran_solutions: Vec::new(),
                ac_frequencies: Vec::new(),
                ac_solutions: Vec::new(),
            };
        }

        // Store initial point
        tran_times.push(tstart);
        tran_solutions.push(x.clone());

        // Initialize transient states from DC solution
        update_transient_state(&self.circuit.instances.instances, &x, &mut state_be);
        update_transient_state(&self.circuit.instances.instances, &x, &mut state_trap);

        // ====================================================================
        // Adaptive Time Stepping Loop
        // ====================================================================
        while t < tstop {
            // Step 1: Limit dt to hit breakpoints
            dt = breakpoint_mgr.limit_dt(t, dt, min_dt);

            // Step 2: If settling after breakpoint, use smaller step
            if breakpoint_mgr.is_settling() {
                dt = breakpoint_mgr.settling_dt(dt, min_dt);
            }

            // Ensure we don't overshoot tstop
            if t + dt > tstop {
                dt = tstop - t;
            }

            // Calculate target time for this step
            let t_target = t + dt;

            // Step 3: Solve with Backward Euler (for LTE estimation)
            let mut x_be = x.clone();
            state_be.method = IntegrationMethod::BackwardEuler;
            #[cfg(feature = "va")]
            let alpha_be = 1.0 / dt;
            let result_be = run_newton_with_stepping(&NewtonConfig::default(), &mut x_be, |x_iter, gmin, source_scale| {
                let mut mna = MnaBuilder::new(node_count);
                for inst in &self.circuit.instances.instances {
                    let stamp = InstanceStamp { instance: inst.clone() };
                    let mut ctx = mna.context_with(gmin, source_scale);
                    // Use stamp_tran_at_time to evaluate time-varying sources at t_target
                    let _ = stamp.stamp_tran_at_time(&mut ctx, Some(x_iter), t_target, dt, &mut state_be);
                }
                #[cfg(feature = "va")]
                for (_name, va_inst) in va_instances.iter_mut() {
                    let mut ctx = mna.context_with(gmin, source_scale);
                    let _ = va_stamp_tran(va_inst, &mut ctx, Some(x_iter), alpha_be, va_rhs_buf);
                }
                mna.builder.insert(gnd, gnd, 1.0);
                let (ap, ai, ax) = mna.builder.finalize();
                (ap, ai, ax, mna.rhs, mna.builder.n)
            }, self.solver.as_mut());

            if !result_be.converged {
                // Newton failed - reduce step and retry
                consecutive_rejects += 1;
                _rejected_steps += 1;
                if consecutive_rejects >= max_consecutive_rejects {
                    final_status = RunStatus::Failed;
                    break;
                }
                dt = (dt * 0.25).max(min_dt);
                continue;
            }

            // Step 4: Solve with Trapezoidal (primary method, higher accuracy)
            let mut x_trap = x.clone();
            state_trap.method = IntegrationMethod::Trapezoidal;
            #[cfg(feature = "va")]
            let alpha_trap = 2.0 / dt;
            let result_trap = run_newton_with_stepping(&NewtonConfig::default(), &mut x_trap, |x_iter, gmin, source_scale| {
                let mut mna = MnaBuilder::new(node_count);
                for inst in &self.circuit.instances.instances {
                    let stamp = InstanceStamp { instance: inst.clone() };
                    let mut ctx = mna.context_with(gmin, source_scale);
                    // Use stamp_tran_at_time to evaluate time-varying sources at t_target
                    let _ = stamp.stamp_tran_at_time(&mut ctx, Some(x_iter), t_target, dt, &mut state_trap);
                }
                #[cfg(feature = "va")]
                for (_name, va_inst) in va_instances.iter_mut() {
                    let mut ctx = mna.context_with(gmin, source_scale);
                    let _ = va_stamp_tran(va_inst, &mut ctx, Some(x_iter), alpha_trap, va_rhs_buf);
                }
                mna.builder.insert(gnd, gnd, 1.0);
                let (ap, ai, ax) = mna.builder.finalize();
                (ap, ai, ax, mna.rhs, mna.builder.n)
            }, self.solver.as_mut());

            if !result_trap.converged {
                // Trapezoidal failed - reduce step and retry
                consecutive_rejects += 1;
                _rejected_steps += 1;
                if consecutive_rejects >= max_consecutive_rejects {
                    final_status = RunStatus::Failed;
                    break;
                }
                dt = (dt * 0.25).max(min_dt);
                continue;
            }

            // Step 5: Estimate LTE using Milne's Device
            let lte = estimate_lte_milne(&x_be, &x_trap, abs_tol, rel_tol);

            // Step 6: Process LTE with PI controller
            let (accept, dt_new) = step_controller.process_lte(&lte, dt);

            if accept {
                // Accept step - use Trapezoidal solution (higher accuracy)
                let t_new = t + dt;

                // Update settling state
                breakpoint_mgr.update_settling(t, t_new);

                // Update solution and state
                x_prev = x.clone();
                x = x_trap;

                // Update transient states with full history (including current for Trap)
                update_transient_state_full(
                    &self.circuit.instances.instances,
                    &x,
                    &x_prev,
                    dt,
                    &mut state_trap,
                );
                // Keep BE state in sync
                update_transient_state(&self.circuit.instances.instances, &x, &mut state_be);

                // Store accepted time point
                tran_times.push(t_new);
                tran_solutions.push(x.clone());

                // Update time tracking
                _dt_prev = dt;
                t = t_new;
                accepted_steps += 1;
                consecutive_rejects = 0;

                // Update dt for next step (from PI controller)
                dt = dt_new.clamp(min_dt, max_dt);
            } else {
                // Reject step - reduce dt and retry
                _rejected_steps += 1;
                consecutive_rejects += 1;

                if consecutive_rejects >= max_consecutive_rejects {
                    final_status = RunStatus::Failed;
                    break;
                }

                dt = dt_new.clamp(min_dt, max_dt);
            }
        }

        // Generate summary message
        let stats = step_controller.statistics();
        let message = Some(format!(
            "Adaptive: {} accepted, {} rejected ({:.1}% rejection), dt range: {:.2e} to {:.2e}, {} breakpoints",
            stats.accepted_count,
            stats.rejected_count,
            stats.rejection_rate * 100.0,
            if stats.min_dt_used > 0.0 { stats.min_dt_used } else { min_dt },
            stats.max_dt_used.max(min_dt),
            breakpoint_mgr.breakpoint_count()
        ));

        RunResult {
            id: RunId(0),
            analysis: AnalysisType::Tran,
            status: final_status,
            iterations: accepted_steps,
            node_names: self.circuit.nodes.id_to_name.clone(),
            solution: x,
            message,
            sweep_var: None,
            sweep_values: Vec::new(),
            sweep_solutions: Vec::new(),
            tran_times,
            tran_solutions,
            ac_frequencies: Vec::new(),
            ac_solutions: Vec::new(),
        }
    }

    /// Extract transient sources (V/I with PULSE/PWL waveforms) from circuit
    fn extract_transient_sources(&self) -> Vec<TransientSource> {
        let mut sources = Vec::new();

        for inst in &self.circuit.instances.instances {
            match inst.kind {
                DeviceKind::V | DeviceKind::I => {
                    if let Some(ref value_str) = inst.value {
                        let upper = value_str.to_uppercase();

                        // Try to parse PULSE
                        if upper.starts_with("PULSE") {
                            if let Some(pulse) = parse_pulse(value_str) {
                                sources.push(TransientSource {
                                    name: inst.name.clone(),
                                    waveform: WaveformSpec::Pulse(pulse),
                                });
                                continue;
                            }
                        }

                        // Try to parse PWL
                        if upper.starts_with("PWL") {
                            if let Some(pwl) = parse_pwl(value_str) {
                                sources.push(TransientSource {
                                    name: inst.name.clone(),
                                    waveform: WaveformSpec::Pwl(pwl),
                                });
                                continue;
                            }
                        }

                        // Try to parse as DC value
                        if let Ok(dc_val) = value_str.parse::<f64>() {
                            sources.push(TransientSource {
                                name: inst.name.clone(),
                                waveform: WaveformSpec::Dc(dc_val),
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        sources
    }

    /// Run DC sweep analysis
    /// Sweeps the specified source from start to stop with given step size
    fn run_dc_sweep_result(&mut self, source: &str, start: f64, stop: f64, step: f64) -> RunResult {
        let config = NewtonConfig::default();
        let node_count = self.circuit.nodes.id_to_name.len();
        let gnd = self.circuit.nodes.gnd_id.0;

        // Find the source instance index
        let source_lower = source.to_ascii_lowercase();
        let source_idx = self.circuit.instances.instances.iter()
            .position(|inst| inst.name.to_ascii_lowercase() == source_lower);

        if source_idx.is_none() {
            return RunResult {
                id: RunId(0),
                analysis: AnalysisType::Dc,
                status: RunStatus::Failed,
                iterations: 0,
                node_names: self.circuit.nodes.id_to_name.clone(),
                solution: Vec::new(),
                message: Some(format!("DC sweep source '{}' not found", source)),
                sweep_var: Some(source.to_string()),
                sweep_values: Vec::new(),
                sweep_solutions: Vec::new(),
                tran_times: Vec::new(),
                tran_solutions: Vec::new(),
                ac_frequencies: Vec::new(),
                ac_solutions: Vec::new(),
            };
        }
        let source_idx = source_idx.unwrap();

        // Calculate sweep points
        let mut sweep_values = Vec::new();
        let step_size = if stop >= start { step.abs() } else { -step.abs() };

        if step_size.abs() < 1e-15 {
            // Zero step - just do single point at start
            sweep_values.push(start);
        } else {
            // Calculate number of points to avoid floating point accumulation errors
            let range = stop - start;
            let n_points = ((range / step_size).abs().floor() as usize) + 1;

            for i in 0..n_points {
                let value = start + (i as f64) * step_size;
                sweep_values.push(value);
            }

            // Ensure we include the exact stop value if close enough
            if let Some(&last) = sweep_values.last() {
                if (last - stop).abs() > 1e-12 && sweep_values.len() < 10000 {
                    // Don't add if we're very close to stop already
                    if (last - stop).abs() / step_size.abs() > 0.5 {
                        sweep_values.push(stop);
                    }
                }
            }
        }

        let mut sweep_solutions = Vec::new();
        let mut total_iterations = 0;
        let mut final_status = RunStatus::Converged;
        let mut final_message = None;

        // Use previous solution as initial guess for next point (continuation)
        let mut x = vec![0.0; node_count];
        self.solver.prepare(node_count);

        // Borrow VA state outside closures
        #[cfg(feature = "va")]
        let va_instances = &mut self.va_instances;
        #[cfg(feature = "va")]
        let va_rhs_buf = &mut self.va_rhs_buf;

        for &sweep_val in &sweep_values {
            // Update source value
            self.circuit.instances.instances[source_idx].value = Some(sweep_val.to_string());

            // Run Newton iteration at this sweep point
            let result = run_newton_with_stepping(&config, &mut x, |x, gmin, source_scale| {
                let mut mna = MnaBuilder::new(node_count);
                for inst in &self.circuit.instances.instances {
                    let stamp = InstanceStamp {
                        instance: inst.clone(),
                    };
                    let mut ctx = mna.context_with(gmin, source_scale);
                    let _ = stamp.stamp_dc(&mut ctx, Some(x));
                }
                #[cfg(feature = "va")]
                for (_name, va_inst) in va_instances.iter_mut() {
                    let mut ctx = mna.context_with(gmin, source_scale);
                    let _ = va_stamp_dc(va_inst, &mut ctx, Some(x), va_rhs_buf);
                }
                // Ground node constraint
                mna.builder.insert(gnd, gnd, 1.0);
                let (ap, ai, ax) = mna.builder.finalize();
                (ap, ai, ax, mna.rhs, mna.builder.n)
            }, self.solver.as_mut());

            total_iterations += result.iterations;

            match result.reason {
                crate::newton::NewtonExitReason::Converged => {
                    sweep_solutions.push(x.clone());
                }
                crate::newton::NewtonExitReason::MaxIters => {
                    final_status = RunStatus::MaxIters;
                    final_message = Some(format!("Failed to converge at sweep point {}", sweep_val));
                    break;
                }
                crate::newton::NewtonExitReason::SolverFailure => {
                    final_status = RunStatus::Failed;
                    final_message = Some(format!("Solver failure at sweep point {}", sweep_val));
                    break;
                }
            }
        }

        // For compatibility, set solution to the last sweep point solution
        let solution = sweep_solutions.last().cloned().unwrap_or_default();

        RunResult {
            id: RunId(0),
            analysis: AnalysisType::Dc,
            status: final_status,
            iterations: total_iterations,
            node_names: self.circuit.nodes.id_to_name.clone(),
            solution,
            message: final_message,
            sweep_var: Some(source.to_string()),
            sweep_values,
            sweep_solutions,
            tran_times: Vec::new(),
            tran_solutions: Vec::new(),
            ac_frequencies: Vec::new(),
            ac_solutions: Vec::new(),
        }
    }

    /// Run AC (small-signal frequency-domain) analysis
    ///
    /// # Arguments
    /// * `sweep_type` - Type of frequency sweep (Dec, Oct, or Lin)
    /// * `points` - Points per decade/octave, or total points for linear sweep
    /// * `fstart` - Start frequency in Hz
    /// * `fstop` - Stop frequency in Hz
    fn run_ac_result(
        &mut self,
        sweep_type: AcSweepType,
        points: usize,
        fstart: f64,
        fstop: f64,
    ) -> RunResult {
        let node_count = self.circuit.nodes.id_to_name.len();
        let gnd = self.circuit.nodes.gnd_id.0;

        // Step 1: Run DC operating point for linearization of nonlinear devices
        let dc_result = self.run_dc_result(AnalysisType::Op);
        if !matches!(dc_result.status, RunStatus::Converged) {
            return RunResult {
                id: RunId(0),
                analysis: AnalysisType::Ac,
                status: RunStatus::Failed,
                iterations: 0,
                node_names: self.circuit.nodes.id_to_name.clone(),
                solution: Vec::new(),
                message: Some("DC operating point failed to converge".to_string()),
                sweep_var: None,
                sweep_values: Vec::new(),
                sweep_solutions: Vec::new(),
                tran_times: Vec::new(),
                tran_solutions: Vec::new(),
                ac_frequencies: Vec::new(),
                ac_solutions: Vec::new(),
            };
        }

        let dc_solution = dc_result.solution;

        // Step 2: Generate frequency sweep
        let frequencies = generate_frequency_sweep(sweep_type, points, fstart, fstop);

        // Step 3: Create complex solver
        let mut complex_solver = create_complex_solver();

        // Step 4: For each frequency, build and solve the complex MNA system
        let mut ac_frequencies = Vec::with_capacity(frequencies.len());
        let mut ac_solutions = Vec::with_capacity(frequencies.len());
        let mut total_iterations = 0;

        for freq in frequencies {
            let omega = 2.0 * std::f64::consts::PI * freq;

            // Build complex MNA matrix
            let mut mna = ComplexMnaBuilder::new(node_count);

            // Stamp all devices
            for inst in &self.circuit.instances.instances {
                let stamp = InstanceStamp {
                    instance: inst.clone(),
                };
                let mut ctx = mna.context(omega);
                let _ = stamp.stamp_ac(&mut ctx, &dc_solution);
            }

            // Stamp VA devices for AC
            #[cfg(feature = "va")]
            for (_name, va_inst) in self.va_instances.iter_mut() {
                let mut ctx = mna.context(omega);
                let _ = va_stamp_ac(va_inst, &mut ctx, &dc_solution);
            }

            // Fix ground node: clear the row and set diagonal to 1
            // First, clear any entries in the ground row (which is stored in columns)
            for col in 0..mna.builder.n {
                mna.builder.col_entries[col].retain(|(row, _)| *row != gnd);
            }
            // Set diagonal to 1
            mna.builder.insert(gnd, gnd, Complex64::new(1.0, 0.0));
            // Clear ground node RHS
            mna.rhs[gnd] = Complex64::new(0.0, 0.0);

            // Finalize matrix
            let (ap, ai, ax) = mna.builder.finalize();
            let n = mna.builder.n;

            // Prepare solver and solve
            complex_solver.prepare(n);
            let mut x = vec![Complex64::new(0.0, 0.0); n];

            let solved = complex_solver.solve(&ap, &ai, &ax, &mna.rhs, &mut x);
            if !solved {
                return RunResult {
                    id: RunId(0),
                    analysis: AnalysisType::Ac,
                    status: RunStatus::Failed,
                    iterations: total_iterations,
                    node_names: self.circuit.nodes.id_to_name.clone(),
                    solution: dc_solution,
                    message: Some(format!("AC solve failed at frequency {} Hz", freq)),
                    sweep_var: None,
                    sweep_values: Vec::new(),
                    sweep_solutions: Vec::new(),
                    tran_times: Vec::new(),
                    tran_solutions: Vec::new(),
                    ac_frequencies,
                    ac_solutions,
                };
            }

            // Convert complex solution to (magnitude_dB, phase_deg) for each node
            let mut freq_solution = Vec::with_capacity(node_count);
            for i in 0..node_count {
                let v = x[i];
                let mag = v.norm();
                let mag_db = if mag > 1e-30 {
                    20.0 * mag.log10()
                } else {
                    -600.0 // Very small magnitude
                };
                let phase_deg = v.arg() * 180.0 / std::f64::consts::PI;
                freq_solution.push((mag_db, phase_deg));
            }

            ac_frequencies.push(freq);
            ac_solutions.push(freq_solution);
            total_iterations += 1;
        }

        RunResult {
            id: RunId(0),
            analysis: AnalysisType::Ac,
            status: RunStatus::Converged,
            iterations: total_iterations,
            node_names: self.circuit.nodes.id_to_name.clone(),
            solution: dc_solution,
            message: None,
            sweep_var: None,
            sweep_values: Vec::new(),
            sweep_solutions: Vec::new(),
            tran_times: Vec::new(),
            tran_solutions: Vec::new(),
            ac_frequencies,
            ac_solutions,
        }
    }
}

/// Generate frequency sweep vector based on sweep type
fn generate_frequency_sweep(
    sweep_type: AcSweepType,
    points: usize,
    fstart: f64,
    fstop: f64,
) -> Vec<f64> {
    let mut frequencies = Vec::new();

    match sweep_type {
        AcSweepType::Dec => {
            // Logarithmic sweep with N points per decade
            let log_start = fstart.log10();
            let log_stop = fstop.log10();
            let decades = log_stop - log_start;
            let total_points = (decades * points as f64).ceil() as usize + 1;

            for i in 0..total_points {
                let log_f = log_start + (i as f64) * decades / ((total_points - 1).max(1) as f64);
                frequencies.push(10.0_f64.powf(log_f));
            }
        }
        AcSweepType::Oct => {
            // Logarithmic sweep with N points per octave
            // 1 octave = log2(2) = 1, so octaves = log2(fstop/fstart)
            let octaves = (fstop / fstart).log2();
            let total_points = (octaves * points as f64).ceil() as usize + 1;

            for i in 0..total_points {
                let factor = (i as f64) * octaves / ((total_points - 1).max(1) as f64);
                frequencies.push(fstart * 2.0_f64.powf(factor));
            }
        }
        AcSweepType::Lin => {
            // Linear sweep with N total points
            let step = (fstop - fstart) / ((points - 1).max(1) as f64);
            for i in 0..points {
                frequencies.push(fstart + (i as f64) * step);
            }
        }
    }

    frequencies
}

pub fn debug_dump_engine(engine: &Engine) {
    println!(
        "engine: nodes={} instances={}",
        engine.circuit.nodes.id_to_name.len(),
        engine.circuit.instances.instances.len()
    );
}
