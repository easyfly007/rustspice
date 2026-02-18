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

### How Verilog-A is Compiled into a Loadable Library

This section explains the full journey from Verilog-A behavioral source code to native machine code that RustSpice can `dlopen` and call at simulation time.

#### What OpenVAF Does

OpenVAF is a Verilog-A compiler written in Rust with an LLVM 15 backend. When RustSpice invokes it as a subprocess, OpenVAF performs these steps internally:

1. **Parse** the Verilog-A source -- behavioral equations like `V(p,n) <+ R * I(p,n)` are parsed into an AST
2. **Semantic analysis** -- resolve node contributions, parameter references, analog operators (`ddt`, `idt`, `limexp`, etc.)
3. **Automatic differentiation** -- compute the Jacobian (partial derivatives of each branch contribution with respect to each node voltage). This is the key step: the Verilog-A author writes only the device equations, and OpenVAF automatically generates the derivative code needed for Newton-Raphson convergence
4. **LLVM IR generation** -- emit LLVM intermediate representation for the `eval()` function (device equations + Jacobians), `setup_model()`, `setup_instance()`, and all other OSDI-mandated functions
5. **Native code emission** -- LLVM compiles the IR to platform-native machine code and links it into a shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows)

The output `.osdi` file is a standard native shared library (e.g., an ELF `.so`) that happens to export symbols conforming to the OSDI 0.3 specification.

#### The OSDI Shared Library Contract

Every `.osdi` file must export exactly four symbols. These are what RustSpice looks for when loading the library:

```
OSDI_VERSION_MAJOR    : u32                    = 0
OSDI_VERSION_MINOR    : u32                    = 3
OSDI_NUM_DESCRIPTORS  : u32                    = N  (number of modules)
OSDI_DESCRIPTORS      : [OsdiDescriptor; N]    = array of module descriptors
```

Each `OsdiDescriptor` is a large `#[repr(C)]` struct (defined in `osdi_types.rs`) that describes everything about one Verilog-A module:

**Metadata fields:**
- `name` -- module name (e.g., `"ekv"`, `"psp103"`)
- `num_nodes`, `num_terminals` -- total nodes and external terminal count
- `nodes` -- array of `OsdiNode` with name, units, and data offsets for residuals
- `num_jacobian_entries`, `jacobian_entries` -- sparsity pattern as `(row_node, col_node)` pairs
- `param_opvar` -- array of parameter descriptors with names, aliases, types, flags
- `model_size`, `instance_size` -- byte sizes for opaque data blobs that the simulator must allocate

**14 function pointers** (the compiled device equations):

| Function | Signature | Purpose |
|----------|-----------|---------|
| `access` | `(inst, model, param_id, flags) -> *mut` | Read/write parameter values |
| `setup_model` | `(handle, model, sim_params, result)` | Initialize model data, validate params |
| `setup_instance` | `(handle, inst, model, temp, n_terms, sim_params, result)` | Initialize instance, node collapsing |
| `eval` | `(handle, inst, model, sim_info) -> flags` | Evaluate device equations |
| `load_residual_resist` | `(inst, model, dst)` | Extract resistive residuals |
| `load_residual_react` | `(inst, model, dst)` | Extract reactive residuals |
| `load_jacobian_resist` | `(inst, model)` | Load G (conductance) Jacobian entries |
| `load_jacobian_react` | `(inst, model, alpha)` | Load C (capacitance) entries scaled by alpha |
| `load_jacobian_tran` | `(inst, model, alpha)` | Load combined G + alpha*C entries |
| `load_spice_rhs_dc` | `(inst, model, dst, prev_solve)` | Compute Newton RHS: J*x - f(x) |
| `load_spice_rhs_tran` | `(inst, model, dst, prev_solve, alpha)` | Transient Newton RHS |
| `load_noise` | `(inst, model, freq, noise_dens)` | Noise density evaluation |
| `load_limit_rhs_resist` | `(inst, model, dst)` | Limiting RHS (resistive) |
| `load_limit_rhs_react` | `(inst, model, dst)` | Limiting RHS (reactive) |

The `eval()` function is the core: it takes node voltages, evaluates the Verilog-A behavioral equations, and writes residuals and Jacobian values into the opaque `instance_data` blob. The `load_*` functions then extract those values from `instance_data` into simulator-accessible arrays.

#### How RustSpice Loads the Library

`OsdiLibrary::load()` in `osdi_loader.rs` performs the loading sequence:

```rust
// Step 1: dlopen -- load the shared library into the process address space
let lib = unsafe { Library::new(path) }?;       // libloading wraps dlopen()

// Step 2: dlsym -- resolve the four mandatory symbols
let version_major: u32 = **lib.get(b"OSDI_VERSION_MAJOR\0")?;
let version_minor: u32 = **lib.get(b"OSDI_VERSION_MINOR\0")?;
let num: u32           = **lib.get(b"OSDI_NUM_DESCRIPTORS\0")?;
let raw: &[OsdiDescriptor] = slice::from_raw_parts(
    *lib.get(b"OSDI_DESCRIPTORS\0")?, num as usize
);

// Step 3: Version check
if version_major != 0 { return Err(VersionMismatch); }

// Step 4: Parse raw C descriptors into safe Rust types
let descriptors: Vec<SafeDescriptor> = raw.iter()
    .map(|desc| SafeDescriptor::from_raw(desc as *const OsdiDescriptor))
    .collect::<Result<Vec<_>, _>>()?;
```

`SafeDescriptor::from_raw()` walks all C pointers in the descriptor and copies them into owned Rust types:
- `*const c_char` name pointers become `String`
- `*const OsdiNode` arrays become `Vec<String>` node names + `Vec<u32>` offsets
- `*const OsdiJacobianEntry` arrays become `Vec<JacobianEntryInfo>` with `(row_node, col_node)` pairs
- `*const OsdiParamOpvar` arrays become `Vec<ParamInfo>` with name, aliases, description, units, flags

After parsing, all string data is owned by Rust. The raw `*const OsdiDescriptor` pointer is retained for calling the function pointers during simulation.

The `Library` handle is stored as `_lib` in `OsdiLibrary` and must stay alive for the entire simulation -- if it is dropped, all function pointers derived from it become dangling.

#### How Loaded Models Are Used at Runtime

Once the library is loaded, the engine creates model and instance wrappers:

**OsdiModel** (shared across all instances of the same `.model`):
```rust
// 1. Allocate an opaque byte buffer for model-level data
let model_data: Vec<u8> = vec![0u8; descriptor.model_size];

// 2. Set .model parameters (e.g., vto=0.5, kp=2e-5) via the access() function pointer
let ptr = access(null, model_data, param_index, ACCESS_FLAG_SET);
*(ptr as *mut f64) = param_value;

// 3. Call setup_model() to initialize derived quantities
setup_model(handle, model_data, sim_params, &mut init_info);
```

**OsdiInstance** (one per device in the netlist):
```rust
// 1. Allocate an opaque byte buffer for instance-level data
let instance_data: Vec<u8> = vec![0u8; descriptor.instance_size];

// 2. Build node mapping: OSDI node index -> MNA matrix row/column
//    Terminal nodes map to netlist node IDs
//    Internal nodes get fresh auxiliary variable indices
node_mapping[0] = mna_index_of_drain;   // terminal
node_mapping[1] = mna_index_of_gate;    // terminal
node_mapping[4] = aux_alloc("internal"); // internal node

// 3. Set instance parameters (W=10u, L=1u) via access()
// 4. Call setup_instance() to initialize
setup_instance(handle, inst_data, model_data, temperature, ...);
```

**During Newton-Raphson iteration:**
```
eval(inst_data, model_data, voltages, temp, flags)
    -> writes residuals and Jacobian values into inst_data

load_jacobian_resist(inst_data, model_data)
    -> extracts Jacobian entries from inst_data into readable locations

for each Jacobian entry (row, col):
    value = read_jacobian_resist(k)
    mna_matrix[row_mna][col_mna] += value

load_spice_rhs_dc(inst_data, model_data, rhs_buf, prev_solve)
    -> computes J*x - f(x) in OSDI's format

for each mapped node:
    rhs_vector[mna_idx] += rhs_buf[mna_idx]
```

This is pure compiled native code at simulation time -- no interpretation, no JIT. The Verilog-A behavioral equations have been compiled down to optimized machine instructions by LLVM, and RustSpice calls them through C function pointers via FFI.

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

## Why OpenVAF: Compiler Comparison

Several approaches exist for compiling Verilog-A compact models. This section compares all available options and explains why RustSpice chose OpenVAF.

### Available Verilog-A Compilers

#### OpenVAF

[OpenVAF](https://openvaf.semimod.de/) is a next-generation Verilog-A compiler written in Rust with an LLVM 15 backend. It compiles `.va` source directly to native shared libraries (`.osdi`) conforming to the simulator-independent OSDI 0.3 standard.

- **Approach**: Verilog-A source → LLVM IR → native shared library (`.osdi`)
- **License**: GPL-3.0
- **Compilation speed**: ~2 seconds for complex CMC models (e.g., PSP103)
- **Code quality**: LLVM-optimized native code with auto-vectorization
- **Verilog-A coverage**: Full LRM compliant; all public CMC models compile without modification
- **Runtime loading**: `dlopen` at runtime, no recompilation of the simulator needed
- **External dependencies**: None at runtime (LLVM is statically linked into the `openvaf` binary)
- **Adopted by**: ngspice (since v39, replacing ADMS), SPICE OPUS, VACASK
- **Maintenance**: Original author (Pascal Kuthe) is no longer active; community fork [OpenVAF-Reloaded](https://github.com/OpenVAF/OpenVAF-Reloaded) continues development

#### ADMS (Legacy)

[ADMS](https://github.com/Qucs/ADMS) (Automatic Device Model Synthesizer) is the traditional open-source Verilog-A translator. It reads Verilog-A and generates C/C++ source code through XML/XSLT templates, which must then be compiled by GCC and linked into the simulator.

- **Approach**: Verilog-A source → XML AST → C/C++ source (via XSLT templates) → GCC → simulator plugin
- **License**: GPL-3.0
- **Compilation speed**: >60 seconds for complex models, plus additional GCC compilation time
- **Code quality**: Unoptimized generated C code with many redundant operations
- **Verilog-A coverage**: Partial; many models require manual `.va` modifications or hand-editing of generated C++
- **Runtime loading**: Requires recompilation of the simulator or building a separate plugin `.so`
- **External dependencies**: GCC or Clang required for the second compilation stage
- **Adopted by**: ngspice (legacy, now replaced), Xyce, Qucs
- **Maintenance**: Abandoned by original author, no active development

#### Modelgen-Verilog (Gnucap)

[Modelgen-Verilog](http://gnucap.org/dokuwiki/doku.php/gnucap:manual:modelgen-verilog) is a Verilog-AMS compiler under development for the Gnucap simulator. It generates C++ plugin source code specific to Gnucap's plugin architecture.

- **Approach**: Verilog-AMS source → C++ source → GCC → Gnucap plugin `.so`
- **License**: GPL-3.0 (part of the Gnucap project, NLnet funded)
- **Compilation speed**: Not publicly benchmarked
- **Code quality**: Early stage, described as "somewhat inefficient" generated code
- **Verilog-A coverage**: Partial; supports most common analog constructs, still in active development
- **Runtime loading**: Gnucap plugin system only
- **Adopted by**: Gnucap only
- **Maintenance**: Active development by Felix Salfelder with NLnet funding

#### Commercial Built-in Compilers

Commercial EDA tools include proprietary Verilog-A compilers tightly integrated into their simulators:

- **Cadence Spectre**: Built-in Verilog-A compiler, full LRM compliance
- **Synopsys HSPICE**: `hsp-vcomp` compiler, LRM 2.4 compliant
- **Siemens/Mentor AFS**: Built-in Verilog-A support

These compilers produce internal (non-portable) formats and are not available for use in third-party simulators.

### Side-by-Side Comparison

| Feature | OpenVAF | ADMS | Modelgen | Commercial |
|---------|---------|------|----------|------------|
| **License** | GPL-3.0 | GPL-3.0 | GPL-3.0 | Proprietary |
| **Compilation output** | Native `.so` (OSDI) | C/C++ source | C++ source | Internal |
| **Requires GCC/Clang** | No (LLVM built-in) | Yes | Yes | No |
| **Simulator-independent** | Yes (OSDI standard) | No (per-simulator templates) | No (Gnucap only) | No |
| **Runtime loading** | dlopen, no rebuild | Requires simulator rebuild | Requires Gnucap rebuild | Built-in |
| **Compile speed** | ~2s | >60s + GCC | Unknown | Fast |
| **Generated code quality** | LLVM optimized | Unoptimized C | Early stage | Best |
| **CMC model compatibility** | All public models | Most (with patches) | Partial | All |
| **Active maintenance** | Community fork | Abandoned | Active | Yes |
| **Auto-differentiation** | Compiler-level AD | Template-based | Compiler-level | Compiler-level |
| **Internal node support** | Full | Partial | Partial | Full |
| **Noise analysis** | Supported in OSDI | Partial | Unknown | Full |

### Why RustSpice Chose OpenVAF

1. **Simulator-independent OSDI interface** -- OpenVAF is the only open-source compiler that produces a standardized binary interface. ADMS and Modelgen both generate simulator-specific C/C++ code requiring custom templates per simulator. With OSDI, RustSpice writes the loader once and it works with any compiled model.

2. **No GCC dependency at runtime** -- ADMS generates C source that must be compiled by GCC/Clang as a second step. OpenVAF includes LLVM and produces the final `.so` directly. Users only need the single `openvaf` binary.

3. **10x faster compilation** -- ~2 seconds vs >60 seconds for complex models like PSP103. This matters for model development iteration and first-time user experience.

4. **Better simulation performance** -- LLVM optimization passes produce faster device evaluation code than ADMS's unoptimized C output. OpenVAF-compiled models run measurably faster in simulation loops.

5. **Full CMC model support without patches** -- All public compact models (PSP, EKV, BSIM-CMG, HiSIM, MEXTRAM, etc.) compile without modification. ADMS often requires manual patching of `.va` files to work around parser limitations.

6. **Industry momentum** -- ngspice, the most widely used open-source SPICE simulator, replaced ADMS with OpenVAF starting from version 39. OSDI/OpenVAF is now the de facto open-source standard for Verilog-A integration.

7. **Subprocess isolation** -- As an external binary, OpenVAF keeps LLVM's ~200MB build dependency out of RustSpice's build process. The compilation cost is a one-time expense per model, and results are cached by content hash.

### Licensing Considerations for Commercial Use

OpenVAF is licensed under **GPL-3.0**. The impact on RustSpice depends on how OpenVAF is used:

| Scenario | GPL Impact on RustSpice |
|----------|------------------------|
| Calling `openvaf` as a subprocess (our approach) | No copyleft obligation -- same as using GCC |
| User installs OpenVAF separately | No obligation on RustSpice at all |
| Bundling `openvaf` binary in a distribution | Must include GPL-3.0 notice and source offer for OpenVAF |
| Linking `libopenvaf` as a library | Would trigger GPL -- RustSpice would need to be GPL |

RustSpice invokes OpenVAF as an **external subprocess** (`openvaf model.va -o model.osdi`). Under GPL-3.0, using a program as a standalone tool via subprocess does not make the calling software a "derivative work." This is the same legal basis that allows commercial IDEs to invoke GCC without becoming GPL themselves.

The `.osdi` output files produced by OpenVAF contain LLVM-generated machine code from the user's `.va` source, not OpenVAF code itself. Compiler output is generally not covered by the compiler's license (the "GCC exception" principle). Loading `.osdi` files at runtime carries no GPL obligation.

**Maintenance risk**: OpenVAF's original author is no longer actively maintaining the project. The [community fork (OpenVAF-Reloaded)](https://github.com/OpenVAF/OpenVAF-Reloaded) has taken over. However, since the OSDI interface is stable and all major CMC models already compile correctly, this is a low practical risk -- the compiler is essentially feature-complete for the current generation of compact models.

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
