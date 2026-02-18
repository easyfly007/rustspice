//! OSDI shared library loader.
//!
//! Loads compiled `.osdi` files (shared libraries) at runtime using `libloading`,
//! resolves the OSDI entry-point symbols, and wraps the raw C descriptors into
//! safe Rust types.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};

use crate::error::VaError;
use crate::osdi_types::*;

/// A loaded OSDI shared library. Keeps the dynamic library handle alive so
/// that all pointers derived from it remain valid.
pub struct OsdiLibrary {
    /// The dynamic library handle. Must outlive all descriptors.
    _lib: Library,
    /// Parsed descriptors, one per Verilog-A module in the library.
    pub descriptors: Vec<SafeDescriptor>,
    /// Path to the loaded `.osdi` file.
    pub path: PathBuf,
    /// OSDI version reported by the library.
    pub version_major: u32,
    pub version_minor: u32,
}

/// Safe Rust wrapper around a raw `OsdiDescriptor`.
///
/// Copies all string data and metadata into owned Rust types so that callers
/// don't need to work with raw C pointers directly. The `raw` pointer is kept
/// for calling OSDI functions.
#[derive(Clone)]
pub struct SafeDescriptor {
    /// Raw pointer to the C descriptor (valid as long as OsdiLibrary is alive).
    pub raw: *const OsdiDescriptor,
    /// Module name (e.g., "ekv", "bsimcmg").
    pub name: String,
    /// Total number of nodes (terminals + internal).
    pub num_nodes: usize,
    /// Number of terminal (external) nodes.
    pub num_terminals: usize,
    /// Node names in order.
    pub node_names: Vec<String>,
    /// Whether each node is a flow node.
    pub node_is_flow: Vec<bool>,
    /// Resistive residual offset for each node in instance data.
    pub node_resist_residual_off: Vec<u32>,
    /// Reactive residual offset for each node in instance data.
    pub node_react_residual_off: Vec<u32>,
    /// Jacobian pattern: (row_node, col_node) pairs with flags.
    pub jacobian_entries: Vec<JacobianEntryInfo>,
    /// Parameter info: all params (model + instance + opvars).
    pub params: Vec<ParamInfo>,
    /// Model data allocation size in bytes.
    pub model_size: usize,
    /// Instance data allocation size in bytes.
    pub instance_size: usize,
    /// Offset in instance data where node mapping array starts.
    pub node_mapping_offset: u32,
    /// Offset in instance data where resistive Jacobian pointers start.
    pub jacobian_ptr_resist_offset: u32,
    /// Offset for collapsed node flags.
    pub collapsed_offset: u32,
    /// Number of state variables.
    pub num_states: u32,
    /// Offset for state index array.
    pub state_idx_off: u32,
    /// Offset for bound step value.
    pub bound_step_offset: u32,
    /// Total number of params (model + instance, not opvars).
    pub num_params: u32,
    /// Number of instance-level params.
    pub num_instance_params: u32,
    /// Number of operating variables.
    pub num_opvars: u32,
}

/// Parsed info for one Jacobian entry.
#[derive(Clone, Debug)]
pub struct JacobianEntryInfo {
    pub row_node: u32,
    pub col_node: u32,
    pub react_ptr_off: u32,
    pub flags: u32,
}

/// Parsed info for one parameter or operating variable.
#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub units: String,
    pub flags: u32,
    pub len: u32,
    /// Index in the param_opvar array.
    pub index: usize,
}

impl ParamInfo {
    /// True if this is a model-level parameter.
    pub fn is_model_param(&self) -> bool {
        (self.flags & PARA_KIND_MASK) == PARA_KIND_MODEL
    }

    /// True if this is an instance-level parameter.
    pub fn is_instance_param(&self) -> bool {
        (self.flags & PARA_KIND_MASK) == PARA_KIND_INST
    }

    /// True if this is an operating variable (output only).
    pub fn is_opvar(&self) -> bool {
        (self.flags & PARA_KIND_MASK) == PARA_KIND_OPVAR
    }

    /// The underlying data type.
    pub fn param_type(&self) -> u32 {
        self.flags & PARA_TY_MASK
    }

    /// True if this is a real-valued parameter.
    pub fn is_real(&self) -> bool {
        self.param_type() == PARA_TY_REAL
    }
}

// Ensure SafeDescriptor raw pointer can be sent across threads (the Library is Send).
unsafe impl Send for SafeDescriptor {}
unsafe impl Sync for SafeDescriptor {}
unsafe impl Send for OsdiLibrary {}

impl OsdiLibrary {
    /// Load an OSDI shared library from the given path.
    pub fn load(path: &Path) -> Result<Self, VaError> {
        // dlopen
        let lib = unsafe { Library::new(path) }.map_err(|e| VaError::LoadFailed {
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;

        // Read version symbols
        let version_major = unsafe {
            let sym: Symbol<*const u32> =
                lib.get(b"OSDI_VERSION_MAJOR\0").map_err(|_| {
                    VaError::MissingSymbol("OSDI_VERSION_MAJOR".into())
                })?;
            **sym
        };
        let version_minor = unsafe {
            let sym: Symbol<*const u32> =
                lib.get(b"OSDI_VERSION_MINOR\0").map_err(|_| {
                    VaError::MissingSymbol("OSDI_VERSION_MINOR".into())
                })?;
            **sym
        };

        // Version check
        if version_major != OSDI_VERSION_MAJOR {
            return Err(VaError::VersionMismatch {
                expected: format!("{}.{}", OSDI_VERSION_MAJOR, OSDI_VERSION_MINOR),
                found: format!("{}.{}", version_major, version_minor),
            });
        }

        // Read descriptor count and array
        let num_descriptors = unsafe {
            let sym: Symbol<*const u32> =
                lib.get(b"OSDI_NUM_DESCRIPTORS\0").map_err(|_| {
                    VaError::MissingSymbol("OSDI_NUM_DESCRIPTORS".into())
                })?;
            **sym
        };

        let descriptors = unsafe {
            let sym: Symbol<*const OsdiDescriptor> =
                lib.get(b"OSDI_DESCRIPTORS\0").map_err(|_| {
                    VaError::MissingSymbol("OSDI_DESCRIPTORS".into())
                })?;
            let arr = std::slice::from_raw_parts(*sym, num_descriptors as usize);
            arr.iter()
                .map(|desc| SafeDescriptor::from_raw(desc as *const OsdiDescriptor))
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(Self {
            _lib: lib,
            descriptors,
            path: path.to_path_buf(),
            version_major,
            version_minor,
        })
    }

    /// Find a descriptor by module name (case-insensitive).
    pub fn find_module(&self, name: &str) -> Option<&SafeDescriptor> {
        self.descriptors
            .iter()
            .find(|d| d.name.eq_ignore_ascii_case(name))
    }
}

impl SafeDescriptor {
    /// Parse a raw C descriptor into safe Rust types.
    ///
    /// # Safety
    /// The `ptr` must point to a valid `OsdiDescriptor` with all its sub-pointers
    /// valid (node arrays, param arrays, etc.). This is guaranteed by the OSDI ABI
    /// as long as the parent Library is alive.
    unsafe fn from_raw(ptr: *const OsdiDescriptor) -> Result<Self, VaError> {
        if ptr.is_null() {
            return Err(VaError::InvalidDescriptor("null descriptor pointer".into()));
        }

        let desc = &*ptr;

        // Module name
        let name = read_c_str(desc.name)
            .ok_or_else(|| VaError::InvalidDescriptor("null module name".into()))?;

        let num_nodes = desc.num_nodes as usize;
        let num_terminals = desc.num_terminals as usize;

        // Parse nodes
        let mut node_names = Vec::with_capacity(num_nodes);
        let mut node_is_flow = Vec::with_capacity(num_nodes);
        let mut node_resist_residual_off = Vec::with_capacity(num_nodes);
        let mut node_react_residual_off = Vec::with_capacity(num_nodes);
        if !desc.nodes.is_null() {
            let nodes = std::slice::from_raw_parts(desc.nodes, num_nodes);
            for node in nodes {
                node_names.push(read_c_str(node.name).unwrap_or_default());
                node_is_flow.push(node.is_flow);
                node_resist_residual_off.push(node.resist_residual_off);
                node_react_residual_off.push(node.react_residual_off);
            }
        }

        // Parse Jacobian entries
        let num_jac = desc.num_jacobian_entries as usize;
        let mut jacobian_entries = Vec::with_capacity(num_jac);
        if !desc.jacobian_entries.is_null() {
            let entries = std::slice::from_raw_parts(desc.jacobian_entries, num_jac);
            for entry in entries {
                jacobian_entries.push(JacobianEntryInfo {
                    row_node: entry.nodes.node_1,
                    col_node: entry.nodes.node_2,
                    react_ptr_off: entry.react_ptr_off,
                    flags: entry.flags,
                });
            }
        }

        // Parse parameters and operating variables
        let total_params =
            desc.num_params as usize + desc.num_instance_params as usize + desc.num_opvars as usize;
        let mut params = Vec::with_capacity(total_params);
        if !desc.param_opvar.is_null() {
            let po = std::slice::from_raw_parts(desc.param_opvar, total_params);
            for (idx, p) in po.iter().enumerate() {
                let primary_name = if !p.name.is_null() && p.num_alias > 0 {
                    let names = std::slice::from_raw_parts(p.name, (1 + p.num_alias) as usize);
                    read_c_str(*names.first().unwrap_or(&std::ptr::null())).unwrap_or_default()
                } else if !p.name.is_null() {
                    // Single name pointer
                    read_c_str(*p.name).unwrap_or_default()
                } else {
                    String::new()
                };

                let mut aliases = Vec::new();
                if !p.name.is_null() && p.num_alias > 0 {
                    let names = std::slice::from_raw_parts(p.name, (1 + p.num_alias) as usize);
                    for &alias_ptr in &names[1..] {
                        if let Some(alias) = read_c_str(alias_ptr) {
                            aliases.push(alias);
                        }
                    }
                }

                params.push(ParamInfo {
                    name: primary_name,
                    aliases,
                    description: read_c_str(p.description).unwrap_or_default(),
                    units: read_c_str(p.units).unwrap_or_default(),
                    flags: p.flags,
                    len: p.len,
                    index: idx,
                });
            }
        }

        Ok(SafeDescriptor {
            raw: ptr,
            name,
            num_nodes,
            num_terminals,
            node_names,
            node_is_flow,
            node_resist_residual_off,
            node_react_residual_off,
            jacobian_entries,
            params,
            model_size: desc.model_size as usize,
            instance_size: desc.instance_size as usize,
            node_mapping_offset: desc.node_mapping_offset,
            jacobian_ptr_resist_offset: desc.jacobian_ptr_resist_offset,
            collapsed_offset: desc.collapsed_offset,
            num_states: desc.num_states,
            state_idx_off: desc.state_idx_off,
            bound_step_offset: desc.bound_step_offset,
            num_params: desc.num_params,
            num_instance_params: desc.num_instance_params,
            num_opvars: desc.num_opvars,
        })
    }

    /// Find a parameter by name (case-insensitive), returns its index in the
    /// param_opvar array.
    pub fn find_param(&self, name: &str) -> Option<&ParamInfo> {
        self.params.iter().find(|p| {
            p.name.eq_ignore_ascii_case(name)
                || p.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
        })
    }

    /// Find a real-valued parameter by name and return its index.
    pub fn find_real_param(&self, name: &str) -> Option<usize> {
        self.find_param(name)
            .filter(|p| p.is_real())
            .map(|p| p.index)
    }

    /// Build a map of parameter names → indices for fast lookup.
    pub fn param_lookup(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        for p in &self.params {
            map.insert(p.name.to_ascii_lowercase(), p.index);
            for alias in &p.aliases {
                map.insert(alias.to_ascii_lowercase(), p.index);
            }
        }
        map
    }
}

/// Helper: read a C string pointer into an owned Rust String.
unsafe fn read_c_str(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}

use std::os::raw::c_char;
