# FFT Analysis for Transient Waveforms

## Overview

The Fast Fourier Transform (FFT) converts time-domain simulation data into frequency-domain representation, revealing the spectral content of signals. This is essential for circuit analysis tasks such as:

- **Total Harmonic Distortion (THD)**: Measuring harmonic content in amplifier outputs
- **Switching noise analysis**: Identifying frequency components from digital switching
- **Filter verification**: Confirming that a filter attenuates the correct frequencies
- **Oscillator characterization**: Measuring fundamental frequency and spur levels
- **Power supply ripple**: Quantifying ripple frequency and amplitude

After running a `.tran` simulation, MySpice can compute the FFT of any node voltage or branch current to produce a magnitude-vs-frequency plot.

---

## The Non-Uniform Sampling Problem

SPICE transient analysis uses adaptive time-stepping: the simulator takes smaller steps during rapid signal transitions and larger steps when the signal is slowly changing. This produces **non-uniformly spaced** time samples.

The FFT algorithm requires **uniformly spaced** samples. Therefore, before computing the FFT, the non-uniform time-domain data must be interpolated onto a uniform time grid.

**Pipeline:**

```
Non-uniform time data
        │
        ▼
┌─────────────────┐
│  Interpolate to  │
│  uniform grid    │  np.interp (linear interpolation)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Apply window    │  Hann, Hamming, Blackman, etc.
│  function        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Compute FFT     │  np.fft.rfft (real-valued input)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalize to    │  20 * log10(|X| / (N/2))
│  dB magnitude    │
└────────┬────────┘
         │
         ▼
  Frequency spectrum (f, magnitude_dB)
```

---

## Algorithm Pipeline

### Step 1: Interpolation

Given non-uniform samples `(t[0], t[1], ..., t[M-1])` with corresponding values `(v[0], v[1], ..., v[M-1])`, create a uniform time grid:

```
dt = (t[-1] - t[0]) / (N - 1)
t_uniform = [t[0], t[0]+dt, t[0]+2*dt, ..., t[-1]]
v_uniform = np.interp(t_uniform, t, v)
```

where `N` is the NFFT size (number of points). Linear interpolation is sufficient for most cases since SPICE already provides well-resolved waveforms.

### Step 2: Window Function

Applying a window function reduces **spectral leakage** — the artificial broadening of frequency peaks caused by analyzing a finite-length signal. Without windowing, sharp peaks in the spectrum spread across adjacent frequency bins.

```
v_windowed = v_uniform * window(N)
```

See the Window Functions section below for available options and trade-offs.

### Step 3: FFT Computation

Since the input signal is real-valued, we use the real FFT (`np.fft.rfft`) which returns only the positive-frequency half of the spectrum:

```
X = np.fft.rfft(v_windowed)
frequencies = np.fft.rfftfreq(N, d=dt)
```

This produces `N/2 + 1` complex frequency bins from DC (0 Hz) up to the Nyquist frequency.

### Step 4: Magnitude in dB

Convert the complex FFT output to magnitude in decibels:

```
magnitude = |X| * 2 / N          # Normalize for single-sided spectrum
magnitude_dB = 20 * log10(magnitude)
```

The factor of `2/N` accounts for the single-sided spectrum (energy from negative frequencies folded into positive) and the FFT normalization.

---

## Window Functions

Window functions trade off between **frequency resolution** (ability to distinguish nearby peaks) and **spectral leakage** (sidelobe level). All windows below are implemented as pure numpy cosine-sum formulas — no scipy dependency required.

| Window | Main Lobe Width | Sidelobe Level | Best For |
|--------|----------------|----------------|----------|
| **Rectangular** | Narrowest (1 bin) | -13 dB | Already-periodic signals, maximum resolution |
| **Hann** | Moderate (2 bins) | -31 dB | General purpose, good default |
| **Hamming** | Moderate (2 bins) | -42 dB | General purpose, lower leakage than Hann |
| **Blackman** | Wide (3 bins) | -58 dB | Low-leakage measurements, weak signal detection |
| **Flat-top** | Widest (5 bins) | -44 dB | Amplitude-accurate measurements, calibration |

### Formulas

All windows use the generalized cosine-sum form:

```
w[n] = a0 - a1*cos(2*pi*n/(N-1)) + a2*cos(4*pi*n/(N-1)) - a3*cos(6*pi*n/(N-1)) + ...
```

**Rectangular:** `w[n] = 1` (no windowing)

**Hann:** `w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))`

**Hamming:** `w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1))`

**Blackman:** `w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))`

**Flat-top:** `w[n] = 0.21557895 - 0.41663158*cos(2*pi*n/(N-1)) + 0.277263158*cos(4*pi*n/(N-1)) - 0.083578947*cos(6*pi*n/(N-1)) + 0.006947368*cos(8*pi*n/(N-1))`

---

## Frequency Resolution and Nyquist Limit

Two fundamental limits govern the FFT output:

### Frequency Resolution (Δf)

The minimum frequency difference that can be resolved:

```
Δf = 1 / T = fs / N
```

where `T` is the total simulation time, `fs` is the sampling frequency (= N/T), and `N` is the NFFT size. Increasing NFFT improves resolution but the underlying limit is set by the simulation duration.

### Nyquist Frequency (fmax)

The maximum frequency that can be represented without aliasing:

```
fmax = fs / 2 = N / (2 * T)
```

In practice, the effective bandwidth is limited by the original transient simulation's minimum timestep — frequencies above `1/(2 * min_timestep)` are not accurately captured.

### NFFT Size Selection

- **Auto**: Uses the next power of 2 greater than the number of simulation points, capped at 65536
- **Manual**: User can select 1024, 2048, 4096, 8192, 16384, 32768, or 65536
- Larger NFFT means finer frequency resolution via interpolation, but does not add information beyond what the simulation captured
- Recommended: start with Auto, increase if finer frequency bins are needed

---

## Implementation Details

### Module-Level Functions (no GUI dependency)

```python
def _make_window(name: str, n: int) -> np.ndarray:
    """Create a window function array of length n."""
    # Pure numpy cosine-sum implementations
    # Supports: Rectangular, Hann, Hamming, Blackman, Flat-top
    ...

def compute_fft(
    times: np.ndarray,
    values: np.ndarray,
    window: str = "Hann",
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of (possibly non-uniform) time-domain data.

    Returns (frequencies, magnitude_dB) arrays.
    """
    # 1. Determine NFFT (auto: next power of 2 of len(times), capped at 65536)
    # 2. Create uniform time grid via np.interp
    # 3. Apply window function
    # 4. Compute np.fft.rfft
    # 5. Normalize magnitude to dB
    # 6. Return (frequencies[1:], magnitude_dB[1:])  -- skip DC bin
    ...
```

### FftViewer Widget

The `FftViewer` is a `QWidget` following the same pattern as `BodePlot`:

- **Toolbar**: Window selector (QComboBox), NFFT selector (QComboBox), Compute FFT button, Auto Scale, Grid toggle, Export, Clear All
- **Plot**: pyqtgraph PlotWidget with logarithmic frequency X-axis, dB Y-axis
- **Data storage**:
  - `_time_data: Dict[str, Tuple[ndarray, ndarray, str]]` — raw time-domain data with color
  - `_signals: Dict[str, FftSignalData]` — computed FFT plot items
- **Key methods**:
  - `set_time_data(name, times, values, color)` — stores raw data for later FFT
  - `compute_all()` — recomputes FFT for all stored signals using current window/NFFT settings
  - `add_signal(name, freqs, mag_db, color)` — adds a computed FFT to the plot
  - `remove_signal(name)`, `set_signal_visible(name, visible)`, `set_signal_color(name, color)`
  - `clear()` — removes all signals and time data
  - `auto_scale()` — auto-range the plot
  - `get_plot_widget()` — returns the underlying PlotWidget for theme integration

Changing the window function or NFFT selection triggers automatic recomputation via `compute_all()`.

---

## Example: RC Low-Pass Filter

Consider an RC low-pass filter excited by a 100 kHz square wave:

```spice
* RC Low-pass filter with square wave input
V1 in 0 PULSE(0 1 0 1n 1n 5u 10u)
R1 in out 1k
C1 out 0 1n
.tran 1n 100u
.end
```

The transient output at V(out) shows the capacitor charging/discharging waveform. The FFT reveals:

- **Fundamental** at 100 kHz (from the 10 us period PULSE source)
- **Odd harmonics** at 300 kHz, 500 kHz, 700 kHz, etc. (square wave has only odd harmonics)
- **Roll-off** at higher frequencies due to the RC filter's -20 dB/decade attenuation
- The filter's -3 dB cutoff frequency at `1/(2*pi*R*C) = 159 kHz` is visible as the point where harmonics begin to attenuate significantly

This allows direct verification that the filter is working as designed, without needing a separate AC analysis.
