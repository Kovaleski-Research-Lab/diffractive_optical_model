## ModulatorFactory

The modulator factory selects amplitude/phase initializations and which
parameters receive gradients.

Requirements:
- plane — a `Plane`
- params — dict
    - gradients (str) — `phase_only`, `amplitude_only`, `complex`, or `none`
    - phase_init / amplitude_init (str)
    - phase_pattern / amplitude_pattern (optional)

## Types of modulators (`gradients`)

1. `phase_only` — only the phase residual is optimized
2. `amplitude_only` — only the amplitude residual is optimized
3. `complex` — both residuals are optimized
4. `none` — no optimization (identity residual)

The physical field is

```
amplitude = clamp(initial_amplitude + sigmoid(opt_amp) - 0.5, 0, 1)
phase     = initial_phase + π tanh(opt_phase)
```

With `opt_* = 0`, this equals the initialization. Amplitude is never negative.

## Initializations

1. `random`
   - Amplitude in [0, 1] (`torch.rand`)
   - Phase in [0, 2π] (`torch.rand * 2π`)
2. `uniform`
   - Amplitude `amplitude_value` (default 1)
   - Phase `phase_value` (default 0)
3. `lens_phase`
   - Thin lens \(\phi = -k(x^2+y^2)/(2f)\). Warns if \(|f| < L\Delta x/\lambda\).
4. `pinhole` (amplitude only)
   - Disk of radius `pinhole_size`.
