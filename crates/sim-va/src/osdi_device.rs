//! High-level wrappers around OSDI model and instance data.
//!
//! `OsdiModel` represents a shared model (parameters shared by all instances
//! of the same `.model` statement). `OsdiInstance` represents a single device
//! instance with its own state, node mapping, and evaluation buffers.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Arc;

use crate::error::VaError;
use crate::osdi_loader::{OsdiLibrary, SafeDescriptor};
use crate::osdi_types::*;

/// A loaded OSDI device model (shared across instances of the same `.model`).
pub struct OsdiModel {
    /// The parsed descriptor for this module.
    pub descriptor: SafeDescriptor,
    /// Opaque model data blob (size = descriptor.model_size).
    pub model_data: Vec<u8>,
    /// Keep the library alive.
    pub library: Arc<OsdiLibrary>,
    /// Module name.
    pub module_name: String,
}

impl OsdiModel {
    /// Create a new model from a library, module name, and `.model` parameters.
    ///
    /// This allocates model data, sets parameters via the OSDI access function,
    /// and calls `setup_model()`.
    pub fn new(
        library: Arc<OsdiLibrary>,
        module_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<Self, VaError> {
        let descriptor = library
            .find_module(module_name)
            .ok_or_else(|| VaError::ModuleNotFound(module_name.into()))?
            .clone();

        // Allocate model data
        let mut model_data = vec![0u8; descriptor.model_size];

        // Set model parameters
        for (name, value_str) in params {
            if let Some(param_info) = descriptor.find_param(name) {
                if !param_info.is_real() {
                    continue; // Skip non-real params for now
                }
                if param_info.is_opvar() {
                    continue; // Can't set operating variables
                }

                let value = parse_spice_value(value_str).map_err(|_| VaError::ParameterError {
                    name: name.clone(),
                    cause: format!("invalid numeric value: {}", value_str),
                })?;

                unsafe {
                    let desc_raw = &*descriptor.raw;
                    let ptr = (desc_raw.access)(
                        std::ptr::null_mut(), // no instance for model params
                        model_data.as_mut_ptr() as *mut c_void,
                        param_info.index as u32,
                        ACCESS_FLAG_SET,
                    );
                    if !ptr.is_null() {
                        *(ptr as *mut f64) = value;
                    }
                }
            }
        }

        // Call setup_model
        let mut init_info = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: std::ptr::null(),
        };

        // Build empty sim params
        let sim_paras = OsdiSimParas {
            names: std::ptr::null(),
            vals: std::ptr::null(),
            names_str: std::ptr::null(),
            vals_str: std::ptr::null(),
        };

        unsafe {
            let desc_raw = &*descriptor.raw;
            (desc_raw.setup_model)(
                std::ptr::null_mut(), // handle (unused by most models)
                model_data.as_mut_ptr() as *mut c_void,
                &sim_paras,
                &mut init_info,
            );
        }

        if init_info.num_errors > 0 {
            let errors = extract_init_errors(&init_info);
            return Err(VaError::SetupModelFailed(errors));
        }

        Ok(Self {
            descriptor,
            model_data,
            library,
            module_name: module_name.into(),
        })
    }
}

/// A single device instance backed by an OSDI model.
pub struct OsdiInstance {
    /// The shared model data.
    pub model: Arc<OsdiModel>,
    /// Opaque per-instance data (size = descriptor.instance_size).
    pub instance_data: Vec<u8>,
    /// Maps OSDI node index → MNA matrix index. -1 = ground or collapsed.
    pub node_mapping: Vec<i32>,
    /// Total nodes (terminal + internal).
    pub num_total_nodes: usize,
    /// Number of terminal nodes.
    pub num_terminals: usize,
    /// Node collapse flags from setup_instance.
    pub connected: Vec<u32>,
    /// Temperature for this instance.
    pub temperature: f64,
    /// Instance name for debugging.
    pub name: String,
}

impl OsdiInstance {
    /// Create a new device instance.
    ///
    /// `terminal_node_ids`: MNA matrix indices for the terminal nodes, in the
    /// same order as the module's port declaration.
    ///
    /// `aux_allocator`: called for each internal node, returns a new MNA index.
    ///
    /// `instance_params`: per-instance parameter overrides from the netlist.
    pub fn new(
        model: Arc<OsdiModel>,
        name: &str,
        terminal_node_ids: &[usize],
        mut aux_allocator: impl FnMut(&str) -> usize,
        instance_params: &HashMap<String, String>,
        temperature: f64,
    ) -> Result<Self, VaError> {
        let desc = &model.descriptor;
        let num_total = desc.num_nodes;
        let num_terminals = desc.num_terminals;

        if terminal_node_ids.len() != num_terminals {
            return Err(VaError::NodeMappingError(format!(
                "module '{}' expects {} terminals, got {}",
                desc.name,
                num_terminals,
                terminal_node_ids.len()
            )));
        }

        // Allocate instance data
        let mut instance_data = vec![0u8; desc.instance_size];

        // Build node mapping
        let mut node_mapping = vec![-1i32; num_total];

        // Terminal nodes: from netlist
        for (i, &mna_idx) in terminal_node_ids.iter().enumerate() {
            node_mapping[i] = mna_idx as i32;
        }

        // Internal nodes: allocate auxiliary variables
        for i in num_terminals..num_total {
            let internal_name = desc
                .node_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("int_{}", i));
            let aux_name = format!("{}#{}", name, internal_name);
            let aux_idx = aux_allocator(&aux_name);
            node_mapping[i] = aux_idx as i32;
        }

        // Write node mapping into instance data at the prescribed offset
        unsafe {
            let mapping_ptr = instance_data
                .as_mut_ptr()
                .add(desc.node_mapping_offset as usize) as *mut i32;
            for (i, &idx) in node_mapping.iter().enumerate() {
                *mapping_ptr.add(i) = idx;
            }
        }

        // Set instance parameters via OSDI access function
        for (param_name, value_str) in instance_params {
            if let Some(param_info) = desc.find_param(param_name) {
                if !param_info.is_real() || param_info.is_opvar() {
                    continue;
                }

                let value =
                    parse_spice_value(value_str).map_err(|_| VaError::ParameterError {
                        name: param_name.clone(),
                        cause: format!("invalid numeric value: {}", value_str),
                    })?;

                unsafe {
                    let desc_raw = &*desc.raw;
                    let ptr = (desc_raw.access)(
                        instance_data.as_mut_ptr() as *mut c_void,
                        model.model_data.as_ptr() as *mut c_void as *mut c_void,
                        param_info.index as u32,
                        ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE,
                    );
                    if !ptr.is_null() {
                        *(ptr as *mut f64) = value;
                    }
                }
            }
        }

        // Call setup_instance
        let mut init_info = OsdiInitInfo {
            flags: 0,
            num_errors: 0,
            errors: std::ptr::null(),
        };

        let sim_paras = OsdiSimParas {
            names: std::ptr::null(),
            vals: std::ptr::null(),
            names_str: std::ptr::null(),
            vals_str: std::ptr::null(),
        };

        let mut connected = vec![1u32; num_total];

        unsafe {
            let desc_raw = &*desc.raw;
            (desc_raw.setup_instance)(
                std::ptr::null_mut(), // handle
                instance_data.as_mut_ptr() as *mut c_void,
                model.model_data.as_ptr() as *mut c_void as *mut c_void,
                temperature,
                num_terminals as u32,
                &sim_paras,
                &mut init_info,
            );
        }

        if init_info.num_errors > 0 {
            let errors = extract_init_errors(&init_info);
            return Err(VaError::SetupInstanceFailed(errors));
        }

        // Check collapsed flags (written at collapsed_offset in instance data)
        if desc.collapsed_offset > 0 {
            unsafe {
                let collapsed_ptr = instance_data
                    .as_ptr()
                    .add(desc.collapsed_offset as usize) as *const u32;
                for i in 0..num_total {
                    let flag = *collapsed_ptr.add(i);
                    connected[i] = flag;
                    if flag == 0 {
                        node_mapping[i] = -1; // Collapsed
                    }
                }
            }
        }

        Ok(Self {
            model,
            instance_data,
            node_mapping,
            num_total_nodes: num_total,
            num_terminals,
            connected,
            temperature,
            name: name.into(),
        })
    }

    /// Evaluate the device at the current node voltages.
    ///
    /// After evaluation, residuals and Jacobian entries are stored in the
    /// instance data at offsets defined by the descriptor. Use the `load_*`
    /// functions to extract them.
    pub fn evaluate(
        &mut self,
        solution: &[f64],
        prev_solution: Option<&[f64]>,
        flags: u32,
    ) -> Result<u32, VaError> {
        // Build voltage vector in OSDI node order
        let mut voltages = vec![0.0; self.num_total_nodes];
        for (osdi_idx, &mna_idx) in self.node_mapping.iter().enumerate() {
            if mna_idx >= 0 {
                voltages[osdi_idx] = solution.get(mna_idx as usize).copied().unwrap_or(0.0);
            }
        }

        let mut prev_solve_vec: Vec<f64>;
        let prev_solve_ptr = match prev_solution {
            Some(prev) => {
                prev_solve_vec = vec![0.0; self.num_total_nodes];
                for (osdi_idx, &mna_idx) in self.node_mapping.iter().enumerate() {
                    if mna_idx >= 0 {
                        prev_solve_vec[osdi_idx] =
                            prev.get(mna_idx as usize).copied().unwrap_or(0.0);
                    }
                }
                prev_solve_vec.as_ptr()
            }
            None => voltages.as_ptr(),
        };

        // Write voltages into instance data node mapping area
        // (OSDI expects voltages to be accessible via the node mapping)

        let sim_info = OsdiSimInfo {
            paras: OsdiSimParas {
                names: std::ptr::null(),
                vals: std::ptr::null(),
                names_str: std::ptr::null(),
                vals_str: std::ptr::null(),
            },
            abstime: 0.0,
            prev_solve: prev_solve_ptr,
            prev_state: std::ptr::null(),
            next_state: std::ptr::null_mut(),
            flags,
        };

        let desc_raw = unsafe { &*self.model.descriptor.raw };

        let ret_flags = unsafe {
            (desc_raw.eval)(
                std::ptr::null_mut(), // handle
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *const c_void,
                &sim_info,
            )
        };

        if ret_flags & EVAL_RET_FLAG_FATAL != 0 {
            return Err(VaError::EvalFailed(ret_flags as i32));
        }

        Ok(ret_flags)
    }

    /// Read the resistive residual for a given node from instance data.
    pub fn read_resist_residual(&self, node_idx: usize) -> f64 {
        let off = self.model.descriptor.node_resist_residual_off
            .get(node_idx)
            .copied()
            .unwrap_or(0) as usize;
        if off == 0 {
            return 0.0;
        }
        unsafe {
            let ptr = self.instance_data.as_ptr().add(off) as *const f64;
            *ptr
        }
    }

    /// Read the reactive residual for a given node from instance data.
    pub fn read_react_residual(&self, node_idx: usize) -> f64 {
        let off = self.model.descriptor.node_react_residual_off
            .get(node_idx)
            .copied()
            .unwrap_or(0) as usize;
        if off == 0 {
            return 0.0;
        }
        unsafe {
            let ptr = self.instance_data.as_ptr().add(off) as *const f64;
            *ptr
        }
    }

    /// Read a resistive Jacobian entry value from instance data.
    ///
    /// The Jacobian pointer values are stored starting at
    /// `jacobian_ptr_resist_offset` in the instance data.
    pub fn read_jacobian_resist(&self, entry_idx: usize) -> f64 {
        let desc = &self.model.descriptor;
        let base_off = desc.jacobian_ptr_resist_offset as usize;
        // Each entry is a double pointer (8 bytes), but in OSDI the values
        // are stored directly as f64 at sequential offsets after resist_offset.
        let off = base_off + entry_idx * std::mem::size_of::<f64>();
        if off + std::mem::size_of::<f64>() > desc.instance_size {
            return 0.0;
        }
        unsafe {
            let ptr = self.instance_data.as_ptr().add(off) as *const f64;
            *ptr
        }
    }

    /// Read a reactive Jacobian entry value from instance data.
    pub fn read_jacobian_react(&self, entry_idx: usize) -> f64 {
        let desc = &self.model.descriptor;
        let react_off = desc.jacobian_entries
            .get(entry_idx)
            .map(|e| e.react_ptr_off as usize)
            .unwrap_or(0);
        if react_off == 0 {
            return 0.0;
        }
        if react_off + std::mem::size_of::<f64>() > desc.instance_size {
            return 0.0;
        }
        unsafe {
            let ptr = self.instance_data.as_ptr().add(react_off) as *const f64;
            *ptr
        }
    }

    /// Load residuals into a destination array indexed by MNA node.
    /// Uses the OSDI load_residual_resist function.
    pub fn load_residual_resist_into(&mut self, dst: &mut [f64]) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_residual_resist)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                dst.as_mut_ptr(),
            );
        }
    }

    /// Load reactive residuals.
    pub fn load_residual_react_into(&mut self, dst: &mut [f64]) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_residual_react)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                dst.as_mut_ptr(),
            );
        }
    }

    /// Load SPICE-format RHS for DC analysis.
    /// This loads `J*x - f(x)` directly, suitable for Newton iteration.
    pub fn load_spice_rhs_dc(&mut self, dst: &mut [f64], prev_solve: &mut [f64]) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_spice_rhs_dc)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                dst.as_mut_ptr(),
                prev_solve.as_mut_ptr(),
            );
        }
    }

    /// Load SPICE-format RHS for transient analysis.
    /// `alpha` = 1/dt for Backward Euler, 2/dt for Trapezoidal.
    pub fn load_spice_rhs_tran(
        &mut self,
        dst: &mut [f64],
        prev_solve: &mut [f64],
        alpha: f64,
    ) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_spice_rhs_tran)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                dst.as_mut_ptr(),
                prev_solve.as_mut_ptr(),
                alpha,
            );
        }
    }

    /// Load resistive Jacobian entries (stored in instance data, then the
    /// simulator reads them via Jacobian pointer offsets).
    pub fn load_jacobian_resist(&mut self) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_jacobian_resist)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
            );
        }
    }

    /// Load combined transient Jacobian (G + alpha*C).
    /// `alpha` = 1/dt for Backward Euler, 2/dt for Trapezoidal.
    pub fn load_jacobian_tran(&mut self, alpha: f64) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_jacobian_tran)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                alpha,
            );
        }
    }

    /// Load reactive Jacobian (C matrix entries).
    /// `alpha` is the scaling factor (usually 1.0 for AC, 1/dt for transient).
    pub fn load_jacobian_react(&mut self, alpha: f64) {
        let desc_raw = unsafe { &*self.model.descriptor.raw };
        unsafe {
            (desc_raw.load_jacobian_react)(
                self.instance_data.as_mut_ptr() as *mut c_void,
                self.model.model_data.as_ptr() as *mut c_void as *mut c_void,
                alpha,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract error messages from an OsdiInitInfo.
fn extract_init_errors(info: &OsdiInitInfo) -> Vec<String> {
    let mut errors = Vec::new();
    if info.num_errors > 0 && !info.errors.is_null() {
        unsafe {
            let errs = std::slice::from_raw_parts(info.errors, info.num_errors as usize);
            for err in errs {
                let param_id = err.payload.parameter_id;
                errors.push(format!(
                    "error code={}, parameter_id={}",
                    err.code, param_id
                ));
            }
        }
    }
    errors
}

/// Parse a SPICE value string with optional engineering suffix.
fn parse_spice_value(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty value".into());
    }

    // Try direct parse first
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
    }

    // Try with SPICE suffix
    let lower = s.to_ascii_lowercase();
    let (num_part, multiplier) = if lower.ends_with("meg") {
        (&lower[..lower.len() - 3], 1e6)
    } else {
        let (num, suffix) = lower.split_at(lower.len().saturating_sub(1));
        match suffix {
            "f" => (num, 1e-15),
            "p" => (num, 1e-12),
            "n" => (num, 1e-9),
            "u" => (num, 1e-6),
            "m" => (num, 1e-3),
            "k" => (num, 1e3),
            "g" => (num, 1e9),
            "t" => (num, 1e12),
            _ => (&lower[..], 1.0),
        }
    };

    num_part
        .parse::<f64>()
        .map(|v| v * multiplier)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_spice_value() {
        assert_eq!(parse_spice_value("1k").unwrap(), 1e3);
        assert!((parse_spice_value("2.5u").unwrap() - 2.5e-6).abs() < 1e-20);
        assert!((parse_spice_value("100n").unwrap() - 100e-9).abs() < 1e-20);
        assert_eq!(parse_spice_value("1meg").unwrap(), 1e6);
        assert_eq!(parse_spice_value("3.3").unwrap(), 3.3);
        assert_eq!(parse_spice_value("1e-3").unwrap(), 1e-3);
    }
}
