# Plan: `.option` Command Support

## Context
The netlist currently has no `.option` support. Simulation parameters like Newton solver tolerances (`abs_tol`, `rel_tol`, `gmin`, `max_iters`) and temperatures are hardcoded. We need a `.option` command that parses `key=val` pairs from the netlist, validates them against defined ranges, and makes them accessible throughout the simulation pipeline.

## Design Goals
- **Easy to extend**: Adding a new option = adding one entry to a registry array
- **Typed values**: `Int`, `Float`, `String`, `Bool`
- **Default + range**: Each option has a default; out-of-range values warn and revert
- **Set tracking**: Easy to check if user explicitly set an option
- **Global access**: Stored in `Circuit`, accessible from `Engine` and everywhere
- **Duplicate handling**: Last `.option` line wins; warn that previous set is overridden
- **Startup print**: At simulation start, print all user-set options and their effective values
- **Documentation**: Code comments in `options.rs` explain how to add new options

## Files to Create/Modify

### 1. Create `crates/sim-core/src/options.rs` (NEW)

Define the options infrastructure with a top-of-file doc comment explaining how to add new options:

```rust
// Value types
enum OptionValue { Int(i64), Float(f64), Str(String), Bool(bool) }

// Range constraints
enum OptionRange { None, IntRange(i64, i64), FloatRange(f64, f64), StringEnum(Vec<&'static str>) }

// Static definition of one option
struct OptionDef {
    name: &'static str,
    description: &'static str,
    default: OptionValue,
    range: OptionRange,
}

// All known options defined in one array (OPTION_DEFS)
// Adding a new option = adding one entry here

// SimOptions struct with entries HashMap<String, OptionEntry>
// OptionEntry { value: OptionValue, is_set: bool }
//
// Methods:
//   - new() -> build from OPTION_DEFS with defaults
//   - set(key, raw_value) -> parse, validate range, warn+revert if out-of-range
//     - if already set, warn "option 'X' redefined, previous value ignored"
//   - get_float(key) -> f64 (returns default if not found)
//   - get_int(key) -> i64
//   - get_string(key) -> &str
//   - get_bool(key) -> bool
//   - is_set(key) -> bool
//   - print_user_options() -> prints all user-set options with their effective values
```

Initial options to register (matching SPICE conventions):
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `abstol` | Float | 1e-12 | (0, 1e-3) | Absolute current tolerance |
| `reltol` | Float | 1e-3 | (0, 1.0) | Relative tolerance |
| `vntol` | Float | 1e-6 | (0, 1.0) | Voltage node tolerance |
| `gmin` | Float | 1e-12 | (0, 1e-3) | Minimum conductance |
| `itl1` | Int | 100 | (1, 10000) | DC iteration limit |
| `itl4` | Int | 50 | (1, 10000) | Transient iteration limit |
| `temp` | Float | 27.0 | (-273.15, 1000.0) | Temperature in C |
| `tnom` | Float | 27.0 | (-273.15, 1000.0) | Nominal temperature in C |

### 2. Modify `crates/sim-core/src/lib.rs`
Add `pub mod options;`

### 3. Modify `crates/sim-core/src/netlist.rs`
- Add `Option` variant to `ControlKind` enum (~line 73)
- Map `".option"` and `".options"` in `map_control_kind()` (~line 561)
- In `build_circuit()` (~line 940): add `ControlKind::Option` arm that calls `circuit.options.set()` for each `param` in `ctrl.params`. Also handle positional args as bare bool flags (e.g., `.option nomod` -> set `nomod=true`).

### 4. Modify `crates/sim-core/src/circuit.rs`
- Add `use crate::options::SimOptions;`
- Add `pub options: SimOptions` field to `Circuit` struct (~line 178)
- Initialize with `SimOptions::new()` in `Circuit::new()` (~line 192)

### 5. Modify `crates/sim-core/src/engine.rs`
- Add a helper method `fn newton_config_from_options(options: &SimOptions) -> NewtonConfig` that builds config from options
- In `run_dc_result()`: use `newton_config_from_options` instead of `NewtonConfig::default()`
- In `run_tran_result_with_params()`: read `vntol`/`reltol` for `abs_tol`/`rel_tol`, read `itl4` for max iterations
- In `run_with_store()` (first call): call `self.circuit.options.print_user_options()` to print all user-set options at simulation start
- In `init_va_devices()`: use `temp` option for temperature

## Duplicate Handling
When the user writes:
```
.option abstol=1e-15
.option abstol=1e-14
```
On the second `set("abstol", "1e-14")`:
1. Detect `is_set == true` already
2. `eprintln!("warning: option 'abstol' redefined (was 1e-15), using new value 1e-14")`
3. Overwrite with new value

## Startup Print
At simulation start, `print_user_options()` prints:
```
options:
  abstol = 1e-15  (user-set)
  gmin = 1e-14  (user-set)
```
Only prints options the user explicitly set. If no options are set, prints nothing.

## Documentation in Code
At the top of `options.rs`, a module-level doc comment explains:
```
/// # How to Add a New Simulator Option
///
/// 1. Add an entry to the `OPTION_DEFS` array below:
///    OptionDef {
///        name: "myoption",
///        description: "What this option controls",
///        default: OptionValue::Float(1.0),
///        range: OptionRange::FloatRange(0.0, 100.0),
///    },
///
/// 2. Access it anywhere you have a `SimOptions` reference:
///    let val = options.get_float("myoption");
///    if options.is_set("myoption") { /* user explicitly set it */ }
///
/// That's it. The parsing, validation, range checking, duplicate
/// warnings, and startup printing are all handled automatically.
```

## Verification
- `cargo build` in workspace root
- `cargo test` in workspace root
- Unit tests in `options.rs`: set/get for each type, range validation, duplicate warning, unknown option warning, `is_set` tracking
