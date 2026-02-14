# Ngspice Raw Format Support

RustSpice supports exporting simulation results in the ngspice raw file format, which is widely compatible with SPICE waveform viewers including ngspice, LTspice, gwave, and others.

## Usage

Use the `--format raw` (or `-f raw`) option to export results in ngspice raw format:

```bash
# Operating point analysis
sim-cli circuit.cir -o output.raw -f raw

# DC sweep
sim-cli circuit.cir -a dc --dc-source V1 --dc-start 0 --dc-stop 5 --dc-step 0.1 -o sweep.raw -f raw

# Transient analysis
sim-cli circuit.cir -a tran -o tran.raw -f raw

# AC analysis
sim-cli circuit.cir -a ac --ac-sweep dec --ac-points 10 --ac-fstart 1 --ac-fstop 1meg -o ac.raw -f raw
```

## Viewing Raw Files

Raw files can be viewed with various tools:

```bash
# Using ngspice/ngnutmeg
ngnutmeg output.raw

# In ngspice interactive mode
ngspice
> load output.raw
> plot v(out)
```

## Format Specification

### Header Structure

```
Title: <circuit title>
Date: <date string>
Plotname: <analysis type>
Flags: <real|complex>
No. Variables: <count>
No. Points: <count>
Variables:
	0	<name>	<type>
	1	<name>	<type>
	...
Values:
 <point_index>	<value>
	<value>
	...
```

### Plotname Values

| Analysis | Plotname |
|----------|----------|
| OP | Operating Point |
| DC | DC transfer characteristic |
| TRAN | Transient Analysis |
| AC | AC Analysis |

### Flags

- `real` - Real-valued data (OP, DC, TRAN analyses)
- `complex` - Complex-valued data (AC analysis)

### Variable Types

| Type | Description |
|------|-------------|
| voltage | Node voltages V(name) |
| current | Branch currents I(name) |
| time | Time scale (TRAN) |
| frequency | Frequency scale (AC) |

### ASCII Values Format

For real data:
```
 <point_index>	<scale_value>
	<var1_value>
	<var2_value>
```

For complex data (AC analysis):
```
 <point_index>	<real>,<imag>
	<real>,<imag>
```

## Examples

### Operating Point Output

```
Title: RustSpice v0.1.0 Simulation
Date: 2026-02-01T12:00:00Z
Plotname: Operating Point
Flags: real
No. Variables: 2
No. Points: 1
Variables:
	0	v(in)	voltage
	1	v(out)	voltage
Values:
 0	1.000000e0
	5.000000e-1
```

### DC Sweep Output

```
Title: RustSpice v0.1.0 Simulation
Date: 2026-02-01T12:00:00Z
Plotname: DC transfer characteristic
Flags: real
No. Variables: 3
No. Points: 11
Variables:
	0	v-sweep	voltage
	1	v(in)	voltage
	2	v(out)	voltage
Values:
 0	0.000000e0
	0.000000e0
	0.000000e0
 1	5.000000e-1
	5.000000e-1
	2.500000e-1
...
```

### Transient Output

```
Title: RustSpice v0.1.0 Simulation
Date: 2026-02-01T12:00:00Z
Plotname: Transient Analysis
Flags: real
No. Variables: 3
No. Points: 100
Variables:
	0	time	time
	1	v(in)	voltage
	2	v(out)	voltage
Values:
 0	0.000000e0
	1.000000e0
	0.000000e0
 1	1.000000e-6
	1.000000e0
	6.321206e-1
...
```

### AC Analysis Output

```
Title: RustSpice v0.1.0 Simulation
Date: 2026-02-01T12:00:00Z
Plotname: AC Analysis
Flags: complex
No. Variables: 3
No. Points: 61
Variables:
	0	frequency	frequency
	1	v(in)	voltage
	2	v(out)	voltage
Values:
 0	1.000000e0,0.000000e0
	1.000000e0,0.000000e0
	9.999990e-1,-9.999684e-4
 1	1.258925e0,0.000000e0
	1.000000e0,0.000000e0
	9.999984e-1,-1.258919e-3
...
```

## Notes

### AC Analysis Complex Values

For AC analysis, RustSpice internally stores results as magnitude (dB) and phase (degrees). When writing to raw format, these are converted to complex representation:

```
magnitude = 10^(magnitude_dB / 20)
phase_rad = phase_deg × π / 180
real = magnitude × cos(phase_rad)
imag = magnitude × sin(phase_rad)
```

### Ground Node Filtering

The ground node (node "0") is automatically filtered out from the output, as it is always 0V by definition.

## Compatibility

The raw format output has been tested with:
- ngspice / ngnutmeg
- LTspice (import)
- gwave

## API Reference

The raw format functions are available in the `sim_core::raw` module:

```rust
use sim_core::raw;

// Operating point
raw::write_raw_op(&run_result, &path, precision)?;

// DC sweep
raw::write_raw_sweep(source, sweep_values, node_names, sweep_results, &path, precision)?;

// Transient
raw::write_raw_tran(times, node_names, solutions, &path, precision)?;

// AC analysis
raw::write_raw_ac(frequencies, node_names, ac_solutions, &path, precision)?;
```
