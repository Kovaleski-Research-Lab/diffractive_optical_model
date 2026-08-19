# Diffractive Optical Model

Scalar diffraction in **millimeters** using the angular spectrum method (ASM) and Rayleigh–Sommerfeld convolution (RSC). Intended for scientific research, not production optics CAD.

## Conventions

- **Units:** millimeters for all lengths (`size`, `center`, `wavelength`, `focal_length`). Example: `wavelength: 520.e-6` is 520 nm.
- **Time factor:** \(e^{-j\omega t}\). Propagating ASM uses \(H = \exp(j k_z z)\). Negative `z` is bilateral outgoing propagation: propagating phase reverses while evanescent terms continue to decay with \(|z|\). It is not inverse reconstruction.
- **Grid:** \(\Delta x = L_x / N_x\), \(x_n = (n - N_x/2)\Delta x\). `size` is the computational window \(N\Delta x\). Zero-padding doubles \(N\) at the **same** \(\Delta x\).
- **FFT origin:** spatial arrays are centered (origin at \(N/2\)). Backends `ifftshift` before `fft2` and `fftshift` after `ifft2`. Transfer functions live on `fftfreq` (DC at index 0).
- **Differentiation:** fields and enabled modulator residuals are differentiable. Plane geometry, wavelength, and precomputed transfer functions are fixed snapshots; construct a new propagator after changing geometry.
- **Not supported:** tilted planes (normals other than \(+\hat{z}\)).
  Mismatched plane sampling uses a torch-native separable Bluestein/CZT
  (`CZTStrategy`); the former dense matrix-product DFT is disabled.

## Kernels

**ASM** (complex \(k_z\), evanescent waves decay):

\[
k_z = \frac{2\pi}{\lambda}\sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2},\quad
H = \exp(j\,\mathrm{Re}(k_z)\,z - |\mathrm{Im}(k_z)|\,|z|)
\]

A Matsushima band-limit zeros *propagating* samples of \(H\) whose chirp aliases on the frequency grid.

**RSC** (Goodman; JOSAA 401908 eq. 29), including obliquity
\(|z|/r\) and a signed outgoing phase:

\[
h = \frac{1}{2\pi}\frac{|z|}{r}\left(\frac{1}{r}-j\,\mathrm{sign}(z)k\right)\frac{e^{\mathrm{sign}(z)\,jkr}}{r}
\]

The discrete kernel is multiplied by \(\Delta x\Delta y\). Direct numerical integration (`prop_type: dni`) uses the same kernel as a slow reference. At exactly zero distance, coincident equal grids return identity; shifted or mismatched coplanar grids require an explicit resampling operation.

**Auto selection:** ASM is used when

\[
z \le \min_{\,m\in\{x,y\}} \frac{2\Delta m\,(L_m-|s_m|)}{\lambda}\sqrt{1-\left(\frac{\lambda}{2\Delta m}\right)^2}
\]

with the padded window if `padded: True`. Otherwise RSC.

`auto` also checks whether the spatial RSC kernel is adequately sampled. If
neither method is valid, construction fails with a diagnostic rather than
silently returning an aliased result. Forced RSC performs the same check unless
`allow_aliasing: True` is supplied for diagnostic convergence work.

FFT-based lateral shifts require padding. Plane centers are global coordinates,
while `Plane.x` and `Plane.y` are local coordinates; an object at global
coordinate zero therefore appears at local coordinate `-center` on a shifted
output plane.

## Installation and packaging

```bash
python -m pip install .
```

Python 3.10–3.13 is supported. Dependencies are bounded in `pyproject.toml`
and split by use:

```bash
python -m pip install ".[core]"       # core library (also the default install)
python -m pip install ".[train]"      # Lightning training, YAML, and torchvision
python -m pip install ".[test]"       # complete test environment
python -m pip install ".[notebook]"   # maintained notebook dependencies
python -m pip install ".[train,test]" # typical development environment
```

Build an artifact with `python -m build`, then install the wheel from `dist/`
to test the same layout users receive. Tiny analytic tests (plane wave,
evanescent decay, ASM/RSC/DNI) are intentionally CPU-safe:

```bash
python -m pytest
```

The default `config.yaml` is a 64×64 CPU-runnable training smoke example.

## Usage

```python
from diffractive_optical_model import Plane, PropagatorFactory
import torch

wavelength = 520e-6  # mm
src = Plane({'name': 'in', 'center': [0, 0, 0], 'size': [1.0, 1.0],
             'normal': [0, 0, 1], 'Nx': 64, 'Ny': 64})
dst = Plane({'name': 'out', 'center': [0, 0, 1.0], 'size': [1.0, 1.0],
             'normal': [0, 0, 1], 'Nx': 64, 'Ny': 64})
prop = PropagatorFactory()(src, dst, {
    'wavelength': wavelength, 'fft_type': 'auto', 'prop_type': 'auto', 'padded': True,
})
field = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
out = prop(field)
```

Installed training entry point:

```bash
dom-train --config /path/to/config.yaml
```

`python train.py` remains available in a source checkout. If no explicit path
is supplied, the installed CLI looks for
`diffractive_optical_model/config.yaml` as a packaged resource.

## Reproducibility and archive status

Before constructing the model, every training invocation writes
`run_manifest.json` under `paths.path_root/paths.path_results/model_id` (or the
path supplied with `--manifest`). The manifest contains the fully resolved
configuration, requested seed, exact command, package and dependency versions,
Git revision and dirty state when available, Python/platform details, and
CPU/CUDA hardware information. Preserve this manifest with checkpoints and
figures; generated manifests and large run artifacts are ignored by Git by
default.

The notebooks and `graveyard/` are archival research material, not maintained
API examples. They may contain obsolete imports, paths, or unsupported
experiments. The package API and the example above are the maintained usage
surface.

## Citations

- Goodman, *Introduction to Fourier Optics*.
- Sampling / ASM–RSC: [doi:10.1364/JOSAA.401908](https://doi.org/10.1364/JOSAA.401908)
- Shifted ASM: [doi:10.1364/OE.18.018453](https://doi.org/10.1364/OE.18.018453)
- Band-limited ASM: Matsushima & Shimobaba, *Opt. Express* (2009).

## License

All rights are reserved. The repository grants no reuse permission; see
`LICENSE` and contact the copyright holder before using the software.
Citation metadata is provided in `CITATION.cff`.
