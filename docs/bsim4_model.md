# BSIM4 (Level 54) Model Documentation

BSIM4 is an enhanced MOSFET model for sub-100nm CMOS processes. It extends BSIM3 with advanced physics for modern nanoscale devices.

## Overview

BSIM4 (Berkeley Short-channel IGFET Model 4) was developed by the Device Research Group at UC Berkeley. It includes:

- Layout-dependent stress effects (STI proximity)
- Substrate current from impact ionization
- Gate tunneling currents
- Enhanced mobility models
- Width-dependent short-channel effects

## Model Level

Use `LEVEL=54` in your `.model` statement:

```spice
.model NMOS_BSIM4 NMOS (LEVEL=54 VTH0=0.4 U0=300)
```

## BSIM4 vs BSIM3 Comparison

| Feature | BSIM3 (Level 49) | BSIM4 (Level 54) |
|---------|------------------|------------------|
| Width-dependent SCE | No | DVT0W, DVT1W, DVT2W |
| Subthreshold offset | No | VOFF, VOFFL, MINV |
| Stress effects | No | SA, SB, KU0, KVTH0 |
| Substrate current | Basic | ALPHA0/1, BETA0/1 |
| Gate tunneling | No | JTSS, JTSD, VTSS, VTSD |
| Phonon scattering | Single | PEMOD (0-3) |

## Parameter Reference

### Threshold Voltage Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| VTH0 | 0.7 (N), -0.7 (P) | V | Zero-bias threshold voltage |
| K1 | 0.5 | V^0.5 | Body effect coefficient |
| K2 | -0.1 | - | Body effect coefficient 2 |
| K3 | 80.0 | - | Narrow width effect coefficient |
| K3B | 0.0 | 1/V | Body-bias coefficient for K3 |
| W0 | 0.0 | m | Narrow width offset |
| DVT0 | 2.2 | - | SCE coefficient |
| DVT1 | 0.53 | - | SCE exponent |
| DVT2 | -0.032 | 1/V | Body-bias coefficient for SCE |
| DVT0W | 0.0 | 1/m | Width-dependent SCE coefficient |
| DVT1W | 5.3e6 | 1/m | Width-dependent SCE exponent |
| DVT2W | -0.032 | 1/V | Body-bias coefficient for width SCE |
| VOFF | -0.1 | V | Subthreshold offset voltage |
| VOFFL | 0.0 | V·m | Length-dependent subthreshold offset |
| MINV | 0.0 | - | Subthreshold ideality factor |
| LPE0 | 1.74e-7 | m | Lateral non-uniform doping length |
| LPEB | 0.0 | m | Body-bias dependent LPE |
| VFB | -1.0 | V | Flat-band voltage |

### Mobility Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| U0 | 400 (N), 150 (P) | cm²/V·s | Low-field mobility |
| UA | 2.25e-9 | m/V | First-order mobility degradation |
| UB | 5.87e-19 | (m/V)² | Second-order mobility degradation |
| UC | -4.65e-11 | 1/V | Body-bias mobility degradation |
| UTE | -1.5 | - | Temperature exponent for mobility |
| UTE0 | 0.0 | - | Enhanced temperature coefficient |
| UTE1 | 0.0 | - | Secondary temperature coefficient |
| PEMOD | 0 | - | Phonon model selector (0-3) |
| UP | 0.0 | - | Phonon parameter |
| LP | 0.0 | m | Phonon length parameter |
| UD | 0.0 | - | Drain bias mobility coefficient |
| UD1 | 0.0 | 1/V | Drain bias coefficient |
| EU | 1.67 | - | Mobility field exponent |

### Velocity Saturation Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| VSAT | 1.5e5 | m/s | Saturation velocity |
| VS | 0.0 | m/s | Alternative saturation velocity |
| VSATTEMP | 0.0 | - | Temperature coefficient for VS |
| A0 | 1.0 | - | Velocity saturation coefficient |
| AGS | 0.2 | 1/V | Gate-bias coefficient for A0 |
| LAMBDA | 0.0 | - | Velocity overshoot exponent |
| VTL | 2.0e5 | m/s | Thermal velocity limit |
| LC | 5e-9 | m | Velocity overshoot length |

### Output Conductance Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| PCLM | 1.3 | - | CLM coefficient |
| PDIBLC1 | 0.39 | - | DIBL coefficient 1 |
| PDIBLC2 | 0.0086 | - | DIBL coefficient 2 |
| PDIBLCB | 0.0 | 1/V | Body-bias DIBL coefficient |
| DROUT | 0.56 | - | DIBL output resistance coefficient |
| ETA0 | 0.08 | - | DIBL body-bias coefficient |
| DSUB | 0.56 | - | DIBL substrate coefficient |
| PVAG | 0.0 | - | Gate-voltage dependent CLM |
| FPROUT | 0.0 | - | Output resistance DIBL |
| PDITS | 0.0 | - | DIBL time constant |
| DELTA | 0.01 | V | Effective Vds smoothing |

### Substrate Current (Impact Ionization)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| ALPHA0 | 0.0 | m/V | Impact ionization coefficient |
| ALPHA1 | 0.0 | 1/V | Drain voltage coefficient |
| BETA0 | 30.0 | V | Impact ionization exponent |
| BETA1 | 0.0 | V | Body-bias coefficient |

The substrate current is:
```
Isub = ALPHA0 * (Vds - Vdsat) * Ids * exp(-BETA0 / (Vds - Vdsat))
```

### Layout Stress Effects

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| SA | 0.0 | m | STI distance (source side) - instance |
| SB | 0.0 | m | STI distance (drain side) - instance |
| SAREF | 1e-6 | m | Reference SA |
| SBREF | 1e-6 | m | Reference SB |
| WLOD | 0.0 | m | LOD width parameter |
| KU0 | 0.0 | - | Mobility stress coefficient |
| KVTH0 | 0.0 | V | Threshold stress coefficient |
| KU0MULT | 1.0 | - | KU0 multiplier |
| TKU0 | 0.0 | 1/K | Temperature coefficient for KU0 |

Stress effects modify mobility and threshold:
```
stress_factor = 1/SA + 1/SB - 1/SAREF - 1/SBREF
u0_eff = u0 * (1 + KU0 * stress_factor)
Vth_eff = Vth + KVTH0 * stress_factor
```

### Gate Tunneling Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| JTSS | 0.0 | A/m² | Source-side tunneling current density |
| JTSD | 0.0 | A/m² | Drain-side tunneling current density |
| VTSS | 10.0 | V | Source-side tunneling voltage |
| VTSD | 10.0 | V | Drain-side tunneling voltage |
| NSTI | 1.0 | - | Tunneling ideality factor |

Gate tunneling currents:
```
Igs = W * L * JTSS * exp(Vgs / (NSTI * VTSS))
Igd = W * L * JTSD * exp(Vgd / (NSTI * VTSD))
```

### Geometry Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| TOX | 2.0e-9 | m | Gate oxide thickness |
| LINT | 0.0 | m | Length offset |
| WINT | 0.0 | m | Width offset |

### Parasitic Resistance

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| RDSW | 200.0 | Ω·μm | Source/drain resistance per width |
| RSH | 0.0 | Ω/sq | Sheet resistance |

## Usage Examples

### Basic NMOS with BSIM4

```spice
* BSIM4 NMOS example
.model NMOS54 NMOS (
+  LEVEL=54
+  VTH0=0.4
+  U0=300
+  TOX=1.5n
+  ALPHA0=1e-6
+  BETA0=30
)

M1 drain gate source bulk NMOS54 W=1u L=65n
```

### NMOS with Stress Parameters

```spice
* BSIM4 with layout stress
.model NMOS54_STRESS NMOS (
+  LEVEL=54
+  VTH0=0.4
+  U0=300
+  KU0=0.1
+  KVTH0=0.01
+  SAREF=1u
+  SBREF=1u
)

* Instance with stress parameters
M1 drain gate source bulk NMOS54_STRESS W=1u L=65n SA=0.5u SB=0.5u
```

### NMOS with Gate Tunneling

```spice
* BSIM4 with gate tunneling
.model NMOS54_TUNNEL NMOS (
+  LEVEL=54
+  VTH0=0.4
+  U0=300
+  TOX=1.2n
+  JTSS=1e-10
+  JTSD=1e-10
+  VTSS=10
+  VTSD=10
+  NSTI=1.0
)

M1 drain gate source bulk NMOS54_TUNNEL W=1u L=45n
```

### Complete BSIM4 Model Card

```spice
* Comprehensive BSIM4 model for 65nm process
.model NMOS65 NMOS (
+ LEVEL=54
* Threshold voltage
+ VTH0=0.35 K1=0.45 K2=-0.1
+ DVT0=2.2 DVT1=0.5 DVT2=-0.03
+ DVT0W=0.0 DVT1W=5e6 DVT2W=-0.03
+ VOFF=-0.1 VOFFL=0.0 MINV=0.0
* Mobility
+ U0=320 UA=2e-9 UB=5e-19 UC=-4e-11
+ UTE=-1.5
* Velocity saturation
+ VSAT=1.2e5 A0=1.0 AGS=0.2
* Output conductance
+ PCLM=1.2 ETA0=0.08 DSUB=0.5
+ PDIBLC1=0.4 PDIBLC2=0.01
* Impact ionization
+ ALPHA0=1e-6 BETA0=28
* Gate tunneling
+ JTSS=1e-11 JTSD=1e-11
+ VTSS=10 VTSD=10 NSTI=1.0
* Stress
+ KU0=0.05 KVTH0=0.005
+ SAREF=1u SBREF=1u
* Geometry
+ TOX=1.4n LINT=5n WINT=5n
* Resistance
+ RDSW=150
)
```

## Output Variables

When using BSIM4, the following outputs are available:

| Variable | Description |
|----------|-------------|
| `ids` | Drain-source current |
| `gm` | Transconductance (dIds/dVgs) |
| `gds` | Output conductance (dIds/dVds) |
| `gmbs` | Body transconductance (dIds/dVbs) |
| `isub` | Substrate current (impact ionization) |
| `igs` | Gate-source tunneling current |
| `igd` | Gate-drain tunneling current |
| `ueff` | Effective mobility |
| `vdsat` | Saturation voltage |
| `vth_eff` | Effective threshold voltage |

## Physical Background

### Impact Ionization

When the drain voltage exceeds the saturation voltage, carriers gain sufficient energy to ionize silicon atoms, creating electron-hole pairs. The resulting substrate current flows from drain to bulk:

```
Isub = α₀(Vds - Vdsat)Ids·exp(-β₀/(Vds - Vdsat))
```

This effect is important for:
- Hot carrier reliability
- Power dissipation
- Analog circuit accuracy

### Layout Stress Effects

STI (Shallow Trench Isolation) creates mechanical stress in the silicon that affects:
- **Electron mobility**: Tensile stress increases electron mobility
- **Threshold voltage**: Stress changes band alignment

Devices closer to STI experience more stress. The SA/SB parameters specify the distance from the gate to the STI edge on each side.

### Gate Tunneling

For oxide thicknesses below ~2nm, quantum mechanical tunneling through the gate oxide becomes significant:
- Increases static power consumption
- Reduces effective input impedance
- Creates gate voltage divider effects

## References

1. BSIM4 Technical Manual, UC Berkeley Device Group
2. Y. Cheng, C. Hu, "MOSFET Modeling & BSIM3 User's Guide"
3. W. Liu, "MOSFET Models for SPICE Simulation"
4. Berkeley BSIM Group: http://bsim.berkeley.edu/

## Version History

- **v1.0** (2026-01): Initial BSIM4 implementation
  - Core DC model with all operating regions
  - Impact ionization (substrate current)
  - Layout stress effects
  - Gate tunneling currents
  - Full parameter set (~55 parameters)
