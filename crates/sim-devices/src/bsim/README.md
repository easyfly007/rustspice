# BSIM3 MOSFET Model Implementation

## Overview

This module implements the **BSIM3v3 (Level 49)** MOSFET model for DC analysis in MySpice. BSIM3 (Berkeley Short-channel IGFET Model) is the industry-standard compact model for sub-micron CMOS technologies (90nm and above).

### Why BSIM3?

- **Industry Standard**: Widely used in commercial SPICE simulators
- **Physics-Based**: Accurately captures short-channel effects
- **Well-Documented**: Extensive literature and parameter extraction tools
- **Balanced Complexity**: Good trade-off between accuracy and simulation speed

### Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| DC Analysis | ✅ Supported | Full Ids, gm, gds, gmbs |
| NMOS/PMOS | ✅ Supported | Automatic sign handling |
| Body Effect | ✅ Supported | K1, K2 parameters |
| Short-Channel Effect | ✅ Supported | DVT0, DVT1, DVT2 |
| DIBL | ✅ Supported | ETA0, DSUB, DROUT |
| Mobility Degradation | ✅ Supported | UA, UB, UC |
| Velocity Saturation | ✅ Supported | VSAT parameter |
| Channel Length Modulation | ✅ Supported | PCLM, PDIBLC1/C2 |
| Temperature Effects | ✅ Supported | KT1, KT2, UTE |
| Subthreshold Conduction | ✅ Supported | NFACTOR |
| S/D Series Resistance | ✅ Supported | RDSW |
| AC/Transient Analysis | ❌ Not Yet | Capacitance model pending |
| Noise Analysis | ❌ Not Yet | KF, AF parameters defined |

---

## Module Structure

```
crates/sim-devices/src/bsim/
├── mod.rs          # Module exports, parameter building, model routing
├── params.rs       # BsimParams structure with all model parameters
├── types.rs        # MosType, MosRegion, BsimOutput, BsimState
├── threshold.rs    # Threshold voltage calculation
├── mobility.rs     # Carrier mobility calculation
├── channel.rs      # Vdsat, CLM, output conductance
├── evaluate.rs     # Main DC evaluation functions
└── README.md       # This documentation
```

### Module Dependencies

```
evaluate.rs
    ├── threshold.rs   (calculate_vth)
    ├── mobility.rs    (calculate_mobility)
    └── channel.rs     (calculate_vdsat, calculate_clm_factor, calculate_rds)

mod.rs
    └── evaluate.rs    (evaluate_bsim_dc, evaluate_level1_dc)
```

---

## Physical Model

### Operating Regions

The MOSFET operates in three regions based on terminal voltages:

```
┌─────────────────────────────────────────────────────────────┐
│                     MOSFET Operating Regions                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CUTOFF:        Vgs < Vth                                   │
│                 Ids ≈ 0 (subthreshold leakage)              │
│                                                             │
│  LINEAR:        Vgs > Vth  AND  Vds < Vdsat                 │
│                 Ids = β[(Vgs-Vth)Vds - Vds²/2]              │
│                                                             │
│  SATURATION:    Vgs > Vth  AND  Vds ≥ Vdsat                 │
│                 Ids = β·Vdsat²/2 · CLM_factor               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Threshold Voltage Model

The threshold voltage includes multiple physical effects:

```
Vth = VTH0 + ΔVth_body + ΔVth_SCE + ΔVth_DIBL + ΔVth_temp

where:
  VTH0        = Zero-bias threshold voltage
  ΔVth_body   = K1·(√(φ-Vbs) - √φ) + K2·Vbs     [Body effect]
  ΔVth_SCE    = -DVT0·exp(-DVT1·Leff/2lt)        [Short-channel]
  ΔVth_DIBL   = -ETA0·Vds                        [DIBL]
  ΔVth_temp   = KT1·(T/Tnom - 1)                 [Temperature]
```

**Body Effect**: When the bulk-source voltage (Vbs) is non-zero, the depletion region charge changes, affecting Vth.

**Short-Channel Effect (SCE)**: For short channels, source/drain depletion regions share more of the channel charge, reducing Vth.

**DIBL**: High drain voltage creates an electric field that lowers the source-side barrier, reducing Vth.

### Mobility Model

Carrier mobility is degraded by vertical electric field and temperature:

```
μeff = μ0 · Ftemp / (1 + UA·Eeff + UB·Eeff² + UC·Vbs·Eeff)

where:
  μ0    = Low-field mobility [cm²/V/s]
  Ftemp = (T/Tnom)^UTE                    [Temperature factor]
  Eeff  = (Vgs - Vth + 2Vt) / (6·tox)     [Effective vertical field]
```

### Saturation Voltage

With velocity saturation, Vdsat is limited:

```
Vdsat = Vgst · Esat·Leff / (Vgst + Esat·Leff)

where:
  Vgst = Vgs - Vth                        [Gate overdrive]
  Esat = 2·VSAT / μeff                    [Saturation field]
```

This smoothly transitions between:
- **Low Vgst**: Vdsat ≈ Vgst (classical pinch-off)
- **High Vgst**: Vdsat ≈ Esat·Leff (velocity saturation limited)

### Channel Length Modulation

In saturation, the effective channel length decreases:

```
CLM_factor = 1 / (1 - ΔL/L)

where:
  ΔL/L = PCLM · (Vds - Vdsat) / (Esat · Leff)
```

---

## Parameters Reference

### Model Selection Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `LEVEL` | 49 | - | Model level (1=Level1, 49=BSIM3, 54=BSIM4) |
| `TYPE` | NMOS | - | Device type: NMOS or PMOS |

### Threshold Voltage Parameters

| Parameter | NMOS Default | PMOS Default | Unit | Description |
|-----------|--------------|--------------|------|-------------|
| `VTH0` | 0.7 | -0.7 | V | Zero-bias threshold voltage |
| `K1` | 0.5 | 0.5 | V^0.5 | First-order body effect coefficient |
| `K2` | 0.0 | 0.0 | - | Second-order body effect coefficient |
| `DVT0` | 2.2 | 2.2 | - | Short-channel effect coefficient |
| `DVT1` | 0.53 | 0.53 | - | Short-channel effect exponent |
| `DVT2` | -0.032 | -0.032 | 1/V | Body-bias SCE coefficient |
| `ETA0` | 0.08 | 0.08 | - | DIBL coefficient (subthreshold) |
| `DSUB` | 0.56 | 0.56 | - | DIBL exponent coefficient |
| `NLX` | 1.74e-7 | 1.74e-7 | m | Narrow width effect parameter |
| `NFACTOR` | 1.0 | 1.0 | - | Subthreshold swing factor |

### Mobility Parameters

| Parameter | NMOS Default | PMOS Default | Unit | Description |
|-----------|--------------|--------------|------|-------------|
| `U0` | 500 | 150 | cm²/V/s | Low-field mobility |
| `UA` | 2.25e-9 | 2.25e-9 | m/V | First-order mobility degradation |
| `UB` | 5.87e-19 | 5.87e-19 | (m/V)² | Second-order mobility degradation |
| `UC` | -4.65e-11 | -4.65e-11 | m/V² | Body-bias mobility degradation |
| `VSAT` | 1.5e5 | 1.5e5 | m/s | Saturation velocity |
| `A0` | 1.0 | 1.0 | - | Mobility reduction factor |
| `AGS` | 0.2 | 0.2 | - | Gate-bias dependent Rds |

### Output Conductance Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `PCLM` | 1.3 | - | Channel length modulation coefficient |
| `PDIBLC1` | 0.39 | - | DIBL output resistance coefficient 1 |
| `PDIBLC2` | 0.0086 | - | DIBL output resistance coefficient 2 |
| `PDIBLCB` | -0.1 | 1/V | DIBL body bias coefficient |
| `DROUT` | 0.56 | - | DIBL length dependence |
| `PSCBE1` | 4.24e8 | - | Substrate current parameter 1 |
| `PSCBE2` | 1.0e-5 | V/m | Substrate current parameter 2 |

### Geometry Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `TOX` | 1.5e-8 | m | Gate oxide thickness (15nm) |
| `LINT` | 0.0 | m | Channel length offset (Leff = L - 2·LINT) |
| `WINT` | 0.0 | m | Channel width offset (Weff = W - 2·WINT) |
| `LMIN` | 0.0 | m | Minimum channel length |
| `WMIN` | 0.0 | m | Minimum channel width |

### Parasitic Resistance Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `RDSW` | 0.0 | Ω·μm | Source/drain resistance per unit width |
| `RSH` | 0.0 | Ω·μm | Gate sheet resistance |
| `PRWG` | 0.0 | 1/V | Gate bias Rds coefficient |
| `PRWB` | 0.0 | 1/V^0.5 | Body bias Rds coefficient |

### Temperature Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `TNOM` | 300.15 | K | Nominal temperature (27°C) |
| `UTE` | -1.5 (N) / -1.0 (P) | - | Mobility temperature exponent |
| `KT1` | -0.11 (N) / -0.08 (P) | V | Vth temperature coefficient |
| `KT1L` | 0.0 | V·m | Vth temp coefficient (length) |
| `KT2` | 0.022 | V | Vth temp coefficient (body) |
| `AT` | 3.3e4 | m/s/K | Vsat temperature coefficient |
| `PRT` | 0.0 | 1/K | RDSW temperature coefficient |

### Capacitance Parameters (For Future AC/Transient)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `CGSO` | 0.0 | F/m | Gate-source overlap capacitance |
| `CGDO` | 0.0 | F/m | Gate-drain overlap capacitance |
| `CGBO` | 0.0 | F/m | Gate-bulk overlap capacitance |
| `CJ` | 5.0e-4 | F/m² | Junction capacitance |
| `CJSW` | 5.0e-10 | F/m | Junction sidewall capacitance |
| `PB` | 1.0 | V | Junction built-in potential |
| `MJ` | 0.5 | - | Junction grading coefficient |
| `MJSW` | 0.33 | - | Sidewall grading coefficient |

### Noise Parameters (For Future Noise Analysis)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `KF` | 0.0 | - | Flicker noise coefficient |
| `AF` | 1.0 | - | Flicker noise exponent |
| `EF` | 1.0 | - | Flicker noise frequency exponent |

---

## API Reference

### Main Evaluation Function

```rust
pub fn evaluate_bsim_dc(
    params: &BsimParams,
    w: f64,              // Device width [m]
    l: f64,              // Device length [m]
    vd: f64,             // Drain voltage [V]
    vg: f64,             // Gate voltage [V]
    vs: f64,             // Source voltage [V]
    vb: f64,             // Bulk voltage [V]
    temp: f64,           // Temperature [K]
) -> BsimOutput
```

### Output Structure

```rust
pub struct BsimOutput {
    pub ids: f64,        // Drain current [A]
    pub gm: f64,         // Transconductance dIds/dVgs [S]
    pub gds: f64,        // Output conductance dIds/dVds [S]
    pub gmbs: f64,       // Body transconductance dIds/dVbs [S]
    pub ieq: f64,        // Equivalent current for MNA [A]
    pub region: MosRegion,  // Operating region
    pub vth_eff: f64,    // Effective threshold voltage [V]
}
```

### Parameter Builder

```rust
pub fn build_bsim_params(
    params: &HashMap<String, String>,
    level: u32,
    is_pmos: bool,
) -> BsimParams
```

### Model Level Router

```rust
pub fn evaluate_mos(
    params: &BsimParams,
    w: f64, l: f64,
    vd: f64, vg: f64, vs: f64, vb: f64,
    temp: f64,
) -> BsimOutput
```

---

## Usage Examples

### Netlist Syntax

```spice
* NMOS transistor
M1 drain gate source bulk NMOS W=1u L=100n

* PMOS transistor
M2 drain gate source bulk PMOS W=2u L=100n

* With explicit parameters
M3 drain gate source bulk NMOS W=1u L=100n VTH0=0.4 U0=400

* Model definition
.model NMOS NMOS (LEVEL=49 VTH0=0.4 U0=400 TOX=2e-9 K1=0.5)
.model PMOS PMOS (LEVEL=49 VTH0=-0.4 U0=150 TOX=2e-9 K1=0.5)
```

### Rust API Usage

```rust
use sim_devices::bsim::{BsimParams, evaluate_bsim_dc, MosType};

// Create NMOS with default parameters
let params = BsimParams::nmos_default();

// Evaluate DC operating point
let output = evaluate_bsim_dc(
    &params,
    1e-6,     // W = 1um
    100e-9,   // L = 100nm
    1.8,      // Vd = 1.8V
    1.2,      // Vg = 1.2V
    0.0,      // Vs = 0V
    0.0,      // Vb = 0V
    300.15,   // T = 27°C
);

println!("Ids = {:.3e} A", output.ids);
println!("gm = {:.3e} S", output.gm);
println!("gds = {:.3e} S", output.gds);
println!("Region: {:?}", output.region);
```

### Custom Parameters

```rust
use sim_devices::bsim::BsimParams;

let mut params = BsimParams::nmos_default();
params.vth0 = 0.4;           // Lower threshold
params.u0 = 400.0;           // Lower mobility
params.tox = 2e-9;           // Thinner oxide
params.k1 = 0.6;             // Stronger body effect
```

---

## MNA Integration

### Stamping Equations

The MOSFET is linearized for Newton-Raphson iteration:

```
i_ds = gm·(vgs - VGS) + gds·(vds - VDS) + gmbs·(vbs - VBS) + IDS
     = gm·vgs + gds·vds + gmbs·vbs + ieq

where:
  ieq = IDS - gm·VGS - gds·VDS - gmbs·VBS
```

### Matrix Stamps

```
         D    G    S    B    RHS
    ┌                              ┐
D   │  gds  gm  -gds-gm  gmbs  -ieq│
    │                              │
G   │   0   0    0       0     0  │
    │                              │
S   │ -gds -gm  gds+gm  -gmbs  ieq │
    │                              │
B   │   0   0    0       0     0  │
    └                              ┘
```

### Stamp Code

```rust
// In stamp.rs
ctx.add(drain, drain, gds);
ctx.add(source, source, gds);
ctx.add(drain, source, -gds);
ctx.add(source, drain, -gds);

ctx.add(drain, gate, gm);
ctx.add(drain, source, -gm);
ctx.add(source, gate, -gm);
ctx.add(source, source, gm);

ctx.add(drain, bulk, gmbs);
ctx.add(drain, source, -gmbs);
ctx.add(source, bulk, -gmbs);
ctx.add(source, source, gmbs);

ctx.add_rhs(drain, -ieq);
ctx.add_rhs(source, ieq);
```

---

## Implementation Details

### PMOS Handling

PMOS devices are handled by voltage sign transformation:

```rust
let (vd_int, vg_int, vs_int, vb_int, sign) = match params.mos_type {
    MosType::Nmos => (vd, vg, vs, vb, 1.0),
    MosType::Pmos => (-vs, -vg, -vd, -vb, -1.0),  // Swap D/S, negate
};
// ... compute ids ...
ids *= sign;  // Flip sign for PMOS
```

### Source-Drain Reversal

For negative Vds, source and drain are swapped:

```rust
if vds < 0.0 {
    vds = -vds;
    vgs = vg_int - vd_int;  // Vgd becomes effective Vgs
}
```

### Numerical Stability

- **GMIN**: Minimum conductance (1e-12 S) prevents singular matrices
- **Clamping**: Parameters clamped to physical ranges
- **Smooth Functions**: Continuous derivatives for Newton convergence

### Physical Constants

```rust
pub const EPSILON_SI: f64 = 11.7 * 8.854e-12;  // Si permittivity [F/m]
pub const EPSILON_OX: f64 = 3.9 * 8.854e-12;   // Oxide permittivity [F/m]
pub const Q_ELECTRON: f64 = 1.602e-19;          // Electron charge [C]
pub const K_BOLTZMANN: f64 = 1.381e-23;         // Boltzmann constant [J/K]
pub const T_NOMINAL: f64 = 300.15;              // Nominal temp [K]
```

---

## Testing

### Unit Tests

Each module has comprehensive unit tests:

```bash
# Run all BSIM tests
cargo test -p sim-devices bsim

# Run specific test module
cargo test -p sim-devices bsim::threshold
cargo test -p sim-devices bsim::mobility
cargo test -p sim-devices bsim::channel
cargo test -p sim-devices bsim::evaluate
```

### Integration Tests

```bash
# Run MNA stamping integration tests
cargo test -p sim-core --test bsim_integration
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| threshold.rs | 4 | Body effect, DIBL, SCE, dVth/dVbs |
| mobility.rs | 3 | Base mobility, field degradation, temperature |
| channel.rs | 4 | Vdsat, CLM factor, linear/sat regions |
| evaluate.rs | 7 | All regions, NMOS/PMOS, width scaling |
| mod.rs | 5 | Parameter parsing, level routing |

---

## Validation

### Expected Behavior

1. **Cutoff Region** (Vgs < Vth):
   - Ids ≈ 0 (subthreshold leakage only)
   - gm very small
   - gds = GMIN

2. **Linear Region** (Vds < Vdsat):
   - Ids increases with both Vgs and Vds
   - High gds (resistive behavior)
   - gm proportional to Vds

3. **Saturation Region** (Vds ≥ Vdsat):
   - Ids nearly constant with Vds
   - Low gds (current source behavior)
   - gm proportional to (Vgs - Vth)

### IV Curve Characteristics

```
           Ids
            │
            │     Vgs = 2.0V  ─────────────────
            │                ╱
            │    Vgs = 1.5V ╱──────────────────
            │              ╱
            │   Vgs = 1.0V╱───────────────────
            │            ╱
            │           ╱
            │__________╱______________________ Vds
            0        Vdsat
```

---

## Limitations and Future Work

### Current Limitations

1. **DC Only**: No capacitance model for AC/transient
2. **No Noise**: Flicker noise parameters defined but not used
3. **Simplified Rds**: Series resistance effect is approximate
4. **No Binning**: Width/length interpolation not supported
5. **No Process Variation**: Monte Carlo parameters not supported

### Planned Enhancements

1. **BSIM4 Support** (Level 54):
   - Gate leakage current
   - Improved short-channel models
   - Stress effects

2. **Capacitance Model**:
   - Intrinsic capacitances (Cgs, Cgd, Cgb)
   - Junction capacitances
   - Overlap capacitances

3. **AC Analysis**:
   - Small-signal model
   - Frequency response

4. **Noise Model**:
   - Thermal noise
   - Flicker noise (1/f)

---

## References

1. **BSIM3v3 Manual**, UC Berkeley Device Group
   - Primary reference for model equations

2. **Y. Cheng, C. Hu**, "MOSFET Modeling & BSIM3 User's Guide", Kluwer Academic Publishers, 1999
   - Comprehensive BSIM3 theory and usage

3. **W. Liu**, "MOSFET Models for SPICE Simulation", Wiley-IEEE Press, 2001
   - SPICE implementation details

4. **Y. Taur, T.H. Ning**, "Fundamentals of Modern VLSI Devices", Cambridge University Press
   - Device physics background

5. **BSIM Group Website**: http://bsim.berkeley.edu/
   - Official model releases and documentation

---

## Changelog

### Version 1.0.0 (Initial Release)

- Implemented BSIM3v3 DC model (Level 49)
- Support for NMOS and PMOS devices
- Complete threshold voltage model (body effect, SCE, DIBL)
- Mobility degradation with temperature
- Channel length modulation
- Integration with MNA stamping
- Comprehensive unit tests
