# Verilog-A Device Model Support

This document describes RustSpice's support for Verilog-A compact device models via the OSDI (Open Source Device Interface) standard. This enables users to simulate circuits with industry-standard models (PSP, EKV, BSIM-CMG, HiSIM, MEXTRAM, etc.) written in Verilog-A without modifying the simulator source code.

## Overview

Verilog-A is the standard language for writing compact semiconductor device models. Rather than hardcoding each model into the simulator, RustSpice compiles Verilog-A source files into native shared libraries using the OSDI standard, then loads and evaluates them at runtime through FFI.

The pipeline has three stages:

1. **Compile**: `.va` source files are compiled to `.osdi` shared libraries via OpenVAF
2. **Load**: `.osdi` libraries are loaded at runtime using `dlopen`/`libloading`
3. **Simulate**: Device equations are evaluated through OSDI function pointers and stamped into the MNA matrix

## Prerequisites

### OpenVAF Compiler

RustSpice uses [OpenVAF](https://openvaf.semimod.de/) as an external compiler to transform Verilog-A source into OSDI shared libraries. OpenVAF must be installed and accessible in your `PATH`.

**Installation:**

```bash
# Download the latest release from:
# https://openvaf.semimod.de/download/

# Linux example:
wget https://openvaf.semimod.de/download/openvaf_linux_amd64.tar.gz
tar xzf openvaf_linux_amd64.tar.gz
sudo mv openvaf /usr/local/bin/

# Verify installation:
openvaf --version
```

### Building with VA Support

Verilog-A support is an optional feature. Build with the `va` flag:

```bash
# Build the CLI with VA support
cargo build -p sim-cli --features va

# Build the full workspace with VA support
cargo build --workspace --features sim-core/va
```

Without the `va` feature, all VA-related code is compiled out with zero overhead.

## Netlist Syntax

### Declaring Verilog-A Models

Use the `.hdl` directive to specify a Verilog-A source file. RustSpice will compile it automatically (with caching) and load the resulting module.

```spice
.hdl "path/to/model.va"
```

For pre-compiled OSDI libraries, use the `.osdi` directive:

```spice
.osdi "path/to/model.osdi"
```

### Using VA Device Models

After loading a VA module, use standard SPICE `.model` and instance syntax. The model type name must match the Verilog-A module name declared in the `.va` file.

```spice
* Load the Verilog-A model
.hdl "ekv.va"

* Define a model with parameters
.model nch ekv type=1 vto=0.5 kp=2e-5

* Instantiate devices using the model
M1 drain gate source bulk nch W=10u L=1u
```

### Complete Example: CMOS Inverter

```spice
* CMOS inverter with Verilog-A MOSFET models
.hdl "psp103.va"

.model nch psp103 type=1 vfb=-0.9 lov=20n toxo=1.4n
.model pch psp103 type=-1 vfb=0.5 lov=20n toxo=1.4n

Vdd vdd 0 DC 1.8
Vin in 0 DC 0.9

M1 out in vdd vdd pch W=2u L=100n
M2 out in 0   0   nch W=1u L=100n

.op
.end
```

### Complete Example: DC Sweep

```spice
* NMOS Id-Vgs characteristic with EKV model
.hdl "ekv.va"

.model nch ekv type=1 vto=0.4 kp=5e-5

Vgs gate 0 DC 0
Vds drain 0 DC 1.8
M1 drain gate 0 0 nch W=10u L=1u

.dc Vgs 0 1.8 0.01
.end
```

## Architecture

### Crate Structure

```
crates/
  sim-va/           Standalone crate (no sim-core dependency)
    src/
      compiler.rs       OpenVAF subprocess + SHA-256 caching
      osdi_types.rs     #[repr(C)] OSDI 0.3 ABI definitions
      osdi_loader.rs    dlopen + symbol resolution
      osdi_device.rs    OsdiModel + OsdiInstance wrappers
      error.rs          Error types
  sim-core/
    src/
      va_stamp.rs       Stamp bridge: OSDI eval -> MNA matrix (cfg(feature="va"))
      engine.rs         VA device dispatch in Newton loops (cfg(feature="va"))
      circuit.rs        DeviceKind::VA variant
      netlist.rs        .hdl/.osdi parsing
```

The stamp bridge (`va_stamp.rs`) lives in sim-core rather than sim-va to avoid a circular dependency. sim-va is a leaf crate with no dependency on sim-core.

### Data Flow

```
                    +-----------+
                    |  Netlist  |
                    |  Parser   |
                    +-----+-----+
                          |
              .hdl path   |   .model params + device instances
              stored in   |   stored in Circuit
              circuit     |
                          v
                    +-----------+
                    | VaCompiler|  openvaf subprocess
                    | (cached)  |  SHA-256 content hash
                    +-----+-----+
                          |
                     .osdi file
                          |
                          v
                   +-------------+
                   | OsdiLibrary |  dlopen + parse descriptors
                   +------+------+
                          |
              SafeDescriptor (nodes, params, Jacobian pattern)
                          |
                          v
                   +-------------+
                   |  OsdiModel  |  set .model params, setup_model()
                   +------+------+
                          |
                          v
                  +--------------+
                  | OsdiInstance |  map nodes, setup_instance()
                  +------+-------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     va_stamp_dc    va_stamp_tran  va_stamp_ac
          |              |              |
          v              v              v
      StampContext   StampContext   ComplexStampContext
      (MNA matrix)   (MNA matrix)  (Complex MNA matrix)
```

### Compilation and Caching

The `VaCompiler` handles transparent compilation with content-based caching:

1. Read the `.va` file content
2. Compute SHA-256 hash of the content
3. Check cache directory for `{hash}.osdi`
4. If cached: return the cached path immediately
5. If not cached: run `openvaf <input.va> -o <cache_dir>/{hash}.osdi`
6. Return the path to the compiled `.osdi` file

**Cache location:** `$RUSTSPICE_VA_CACHE` environment variable, or `~/.cache/rustspice/va/` by default.

**OpenVAF path:** `$OPENVAF_PATH` environment variable, or `openvaf` from `PATH` by default.

Caching means that repeated simulations of the same model incur no compilation cost. The cache is invalidated automatically when the `.va` source changes.

### OSDI Interface

The OSDI (Open Source Device Interface) v0.3 standard defines a C ABI for compiled Verilog-A models. Each `.osdi` shared library exports:

| Symbol | Type | Description |
|--------|------|-------------|
| `OSDI_VERSION_MAJOR` | `u32` | Must be 0 |
| `OSDI_VERSION_MINOR` | `u32` | Must be 3 |
| `OSDI_NUM_DESCRIPTORS` | `u32` | Number of modules in the library |
| `OSDI_DESCRIPTORS` | `[OsdiDescriptor; N]` | Array of module descriptors |

Each `OsdiDescriptor` contains:

- **Node information**: terminal and internal node names, count
- **Jacobian sparsity pattern**: which (row, col) entries are non-zero
- **Parameter metadata**: names, types, flags for model/instance params
- **Function pointers**: `eval`, `setup_model`, `setup_instance`, `load_jacobian_resist`, `load_spice_rhs_dc`, etc.
- **Size information**: byte sizes for opaque model and instance data blobs

### Stamp Bridge

The stamp bridge translates OSDI evaluation results into MNA matrix entries. Three functions handle the three analysis types:

**`va_stamp_dc()`** - DC operating point and DC sweep:

```
1. eval(flags = RESIST_RESIDUAL | RESIST_JACOBIAN | ANALYSIS_DC)
2. load_jacobian_resist()              -> G matrix entries in instance data
3. for each Jacobian entry (row, col):
       ctx.add(row, col, G_value)      -> stamp into MNA conductance matrix
4. load_spice_rhs_dc(rhs, prev_solve)  -> compute J*x - f(x)
5. for each mapped node:
       ctx.add_rhs(node, rhs_value)    -> stamp into RHS vector
```

**`va_stamp_tran()`** - Transient analysis:

Same as DC but includes reactive (capacitive) contributions:

```
1. eval(flags = RESIST + REACT + ANALYSIS_TRAN)
2. load_jacobian_tran(alpha)           -> G + alpha*C combined entries
       alpha = 1/dt  (Backward Euler)
       alpha = 2/dt  (Trapezoidal)
3. Stamp combined Jacobian into matrix
4. load_spice_rhs_tran(rhs, prev_solve, alpha)
5. Stamp transient RHS
```

**`va_stamp_ac()`** - AC small-signal analysis:

```
1. eval(flags = RESIST + REACT + ANALYSIS_AC) at DC operating point
2. load_jacobian_resist()              -> G entries
3. for each Jacobian entry:
       read G (resistive) and C (reactive)
       ctx.add_real(row, col, G)       -> real part of Y(jw)
       ctx.add_imag(row, col, w*C)     -> imaginary part of Y(jw)
```

### Engine Integration

The `Engine` struct holds VA state behind feature gates:

```rust
pub struct Engine {
    pub circuit: Circuit,
    solver: Box<dyn LinearSolver>,
    #[cfg(feature = "va")]
    va_instances: HashMap<String, OsdiInstance>,
    #[cfg(feature = "va")]
    va_libraries: Vec<Arc<OsdiLibrary>>,
    #[cfg(feature = "va")]
    va_rhs_buf: Vec<f64>,
}
```

Call `engine.init_va_devices()` after construction to compile, load, and initialize all VA devices. This must be called before running any analysis.

During Newton-Raphson iteration, VA devices are stamped alongside built-in devices:

```rust
// Built-in devices
for inst in &circuit.instances.instances {
    let stamp = InstanceStamp { instance: inst.clone() };
    stamp.stamp_dc(&mut ctx, Some(x));
}

// VA devices (behind #[cfg(feature = "va")])
for (_name, va_inst) in va_instances.iter_mut() {
    va_stamp_dc(va_inst, &mut ctx, Some(x), va_rhs_buf);
}
```

## Supported Analysis Types

| Analysis | VA Support | Integration Method |
|----------|------------|--------------------|
| `.op` (DC operating point) | Yes | Newton-Raphson with source/gmin stepping |
| `.dc` (DC sweep) | Yes | Newton-Raphson at each sweep point |
| `.tran` (Transient) | Yes | Backward Euler + Trapezoidal with adaptive timestep |
| `.ac` (AC small-signal) | Yes | Linearization at DC point, complex Y(jw) = G + jwC |

## Error Handling

### Common Errors

**OpenVAF not found:**
```
error: openvaf not found. Install OpenVAF from https://openvaf.semimod.de/
Set OPENVAF_PATH environment variable to specify a custom location.
```

**Compilation failure:**
```
error: openvaf compilation failed for 'model.va':
  model.va:42: error: undeclared variable 'Vth'
```
OpenVAF's error messages include line numbers and descriptions. Fix the `.va` source and re-run.

**Module not found:**
```
error: VA module 'ekv' not found in loaded OSDI libraries
```
The module name in `.model` must match the `module` name declared in the Verilog-A source.

**Node count mismatch:**
```
error: module 'ekv' expects 4 terminals, got 3
```
The device instance must provide the correct number of terminal nodes as declared by the Verilog-A module's port list.

## Limitations

- **Internal nodes**: VA models with internal nodes require auxiliary MNA variable allocation. This is implemented but not yet fully tested with complex models.
- **Noise analysis**: OSDI noise function pointers are defined but noise analysis is not yet integrated.
- **String parameters**: Only real-valued (floating point) parameters are supported. String and integer parameters are skipped.
- **Temperature**: Default temperature is 300.15K (27 C). Per-instance temperature override is not yet exposed in netlist syntax.
- **Operating point variables**: OSDI operating point variables (opvars) are computed but not yet reported in simulation output.

## File Reference

| File | Purpose |
|------|---------|
| `crates/sim-va/src/compiler.rs` | OpenVAF subprocess invocation and .osdi caching |
| `crates/sim-va/src/osdi_types.rs` | OSDI 0.3 C ABI struct definitions (`#[repr(C)]`) |
| `crates/sim-va/src/osdi_loader.rs` | Shared library loading and descriptor parsing |
| `crates/sim-va/src/osdi_device.rs` | OsdiModel and OsdiInstance high-level wrappers |
| `crates/sim-va/src/error.rs` | VaError enum with all error variants |
| `crates/sim-core/src/va_stamp.rs` | MNA stamp bridge (va_stamp_dc/tran/ac) |
| `crates/sim-core/src/engine.rs` | Engine integration (init_va_devices, Newton dispatch) |
| `crates/sim-core/src/circuit.rs` | DeviceKind::VA variant, Circuit.va_files |
| `crates/sim-core/src/netlist.rs` | .hdl/.osdi directive parsing |
