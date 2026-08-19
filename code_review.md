# Thorough Code Review

Date: 2026-08-18

## Revision status

The reliability plan generated from this review was implemented on 2026-08-18.
The findings below are retained as the original audit record and reproduction
evidence; this status section supersedes their present-tense wording.

Resolved numerical findings:

- C1/C3: FFT-based mismatched grids now use the torch-native separable
  Bluestein `CZTStrategy`; dense MPFFT is disabled in the public factory and
  marked deprecated. Direct-sum, round-trip, zero-distance constant, shifted
  ASM/RSC/DNI, autograd, and dtype tests cover the replacement.
- C2: ASM and RSC now use the global-coordinate shift sign already used by
  DNI. Positive/negative lateral shifts are tested in local and global
  coordinates.
- H1: spatial resampling evaluates physical global coordinates and preserves
  complex64/complex128 real and imaginary components.
- H2/H3: padding and cropping preserve the discrete origin on odd/rectangular
  grids, and coincident equal grids return exact identity at zero distance.
- H4/H7: the factory applies conservative ASM/RSC validity checks, requires
  padded RSC, accounts for shift phase in ASM band limiting, and errors when no
  trustworthy automatic method is available.
- H5/H6/H8/H9: real fields promote consistently, bilateral negative-z RSC is
  conjugate to positive-z propagation, documentation now uses `exp(-iωt)`,
  and forced PyTorch FFT rejects mismatched grids.

Resolved workflow findings:

- The default config is a 64x64 CPU-runnable model with one trainable phase
  modulator and a verified forward/backward/optimizer step.
- Fixed modulator fields are buffers, optimizer construction filters trainable
  parameters, and invalid gradient modes fail early.
- MNIST validation is split reproducibly from training data; official test data
  is reserved, uint8 data is normalized before thresholding, precision is
  configurable, and zero-worker/odd-padding cases are handled.
- PSNR/SSIM objectives use stable minimization forms and consistent intensity
  targets.
- Packaging uses `pyproject.toml`, bounded core/train/test/notebook extras, a
  packaged default config, schema validation, an installed CLI, pre-construction
  run manifests, CPU wheel CI, explicit rights/citation metadata, and expanded
  artifact ignores.

Current verification: **89 passed, 1 skipped** in the source environment; the
skip is the torchvision-dependent data-module integration test because
torchvision is absent locally. The wheel and source distribution build
successfully, and the wheel imports its packaged default configuration away
from the source checkout. New, substantially different optical regimes should
still receive convergence studies before publication; the validity checks are
guardrails, not a substitute for experiment-specific verification.

## Scope

This review covers the active `diffractive_optical_model` package, its tests,
`train.py`, `config.yaml`, packaging metadata, documentation, and the data
pipeline. The `graveyard/` directory was treated as archival and was not used
to judge the active numerical implementation, except when checking project
hygiene. The notebooks received a portability/reproducibility review rather
than a cell-by-cell scientific validation.

The review focused on a research-code standard: numerical correctness,
explicit conventions, convergence evidence, reproducibility, and failure with
clear diagnostics when a requested calculation is outside the implementation's
valid range.

## Executive summary

The same-grid, forward-propagating ASM path is the strongest part of the
project. Its centered-grid convention is coherent, the Fourier shifts are
implemented consistently, evanescent components decay, transfer functions are
buffers, and the current small-grid test suite is fast and useful.

The package is not yet reliable as a **general-purpose** ASM/RSC calculator.
The most important problems are:

1. The matrix-product Fourier transform is not correctly normalized or sampled
   for arbitrary mismatched planes. A zero-distance constant field can acquire
   an amplitude of 4, and broader probes produced relative errors above 100%
   against direct integration.
2. Output-plane center shifts have the wrong physical sign in ASM and RSC.
   DNI uses the physically consistent global-coordinate difference, so the
   three methods disagree.
3. The default configuration constructs extremely large dense complex128 DFT
   matrices for its second block. Initialization and propagation are likely
   impractical, independent of available GPU memory.
4. Several important edge cases fail: odd-sized padded grids crash, RSC/DNI at
   zero distance return zero rather than identity, and complex spatial
   resampling deletes the imaginary component.
5. The default training configuration freezes every modulator while enabling
   training, so its loss has no gradient and backward fails.
6. Tests mostly validate internal agreement. RSC and DNI share the same kernel,
   so their agreement is not an independent validation of normalization,
   sampling, phase, or shift direction.

These issues should be addressed before results from mismatched grids, shifted
planes, RSC fallback, or the default training configuration are used in
scientific conclusions.

## Verification performed

- Ran the complete suite in the available `sci` environment:
  **42 tests passed in 4.00 seconds**.
- Reproduced odd padded propagation failure for `N=15`:
  the field dimension was 29 while the transfer-function dimension was 30.
- Reproduced zero-distance behavior:
  ASM returned identity to about `1.3e-7` relative error; RSC and DNI returned
  all zeros with relative identity error 1.
- Reproduced shifted-plane disagreement at `shift_x=+0.15 mm`:
  ASM/RSC peaked near local `+0.15 mm`; DNI peaked near local `-0.15 mm`.
- Reproduced mismatched MPFFT gain at zero distance:
  a constant 16x16 field on a 0.4 mm window mapped to an 8x8 field on a
  0.2 mm window with constant amplitude **4.0**, not 1.0.
- Compared mismatched-grid RSC to DNI over several small geometries. Relative
  errors ranged from 0.166 to 2.94 and norm ratios ranged from about 0.99 to
  3.89. The currently tested geometry is one of the favorable cases.
- Reproduced complex-resampling data loss:
  an all-`1j` `complex64` field became a zero-valued `float32` field.
- Constructed a small DOM using the default `gradients: none` behavior:
  trainable parameter count was zero, the loss did not require gradients, and
  `loss.backward()` raised `RuntimeError`.
- Confirmed that a mismatched propagator stores the same MPFFT matrices under
  two state-dict paths.
- Probed forced RSC plane-wave scale. Gross gains occur when its spatial kernel
  is under-sampled (for example, mean magnitude about 185 instead of 1 at
  `N=32`, `L=1 mm`, `z=0.02 mm`). This is best treated as a missing validity
  check/convergence failure, not proof that the continuous kernel prefactor is
  wrong.
- The training/data entry point was not run end to end because `torchvision`
  is absent from the otherwise working `sci` environment. It is listed in
  `requirements.txt`, but the project does not provide a locked environment.

## Critical findings

### C1. MPFFT is not a valid general transform between mismatched planes

Locations:

- `diffractive_optical_model/propagator/strategies/fft_strategies/mp_strategy.py:31-60`
- `diffractive_optical_model/propagator/strategies/fft_strategies/mp_strategy.py:82-98`
- `diffractive_optical_model/propagator/strategies/fft_strategies/mp_strategy.py:113-123`
- `tests/test_fft.py:75-83`
- `tests/test_propagation.py:179-198`

The frequency vector is selected from whichever plane has the coarser spatial
pitch. That accounts for a Nyquist limit but ignores the fact that physical
window length determines frequency spacing. The inverse matrices then divide
by `N_freq`, which is only the proper inverse-DFT scaling when the spatial and
frequency grids are a reciprocal pair.

For arbitrary input/output windows and counts, a matrix Fourier integral needs
explicit quadrature weights and a frequency grid chosen from both support and
bandwidth requirements. The current hybrid grid does not provide that.

Consequences:

- Absolute amplitude is wrong even at `z=0`.
- Upsampling, downsampling, window enlargement, and window reduction behave
  differently and can have errors well above 100%.
- ASM and RSC inherit these errors whenever the factory chooses MPFFT.
- Passing shape-only tests gives false confidence.

Recommended action:

1. Temporarily reject mismatched spatial grids in ASM/RSC unless a validated
   scaled transform is selected.
2. Replace MPFFT with a derived scaled-FFT/CZT/chirp-z implementation, or
   formulate it explicitly as a Fourier quadrature with correct `dx` and `df`
   factors.
3. Validate constants, individual Fourier modes, impulses, unequal windows,
   unequal pitches, and unequal sample counts at `z=0` before using propagation
   tests.
4. Compare against DNI on several independent geometries and perform
   convergence studies as pitch and window are refined.

### C2. ASM and RSC use the wrong sign for physical plane-center shifts

Locations:

- `diffractive_optical_model/plane/plane.py:47-52`
- `diffractive_optical_model/plane/plane.py:100-102`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/asm_strategy.py:43-46`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/rsc_strategy.py:44-52`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/dni_strategy.py:28-30`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/dni_strategy.py:57-59`
- `tests/test_propagation.py:128-140`

Plane `x` and `y` arrays are local coordinates centered at zero. If
`s = center_out - center_in`, the physical kernel separation is
`x_out - x_in + s`. DNI implements that expression. ASM uses a negative
frequency-domain shift ramp and RSC evaluates `h(x - s)`, producing the
opposite local displacement.

The existing ASM lateral-shift test encodes the incorrect local-coordinate
expectation instead of checking global coordinates or comparing all three
methods.

Recommended action:

- Change ASM to the phase-ramp sign implied by
  `x_out - x_in + center_out - center_in`.
- Change the RSC kernel coordinates consistently.
- Keep DNI's global-coordinate expression unless the public meaning of
  `Plane.center` is deliberately redefined.
- Add positive and negative x/y shifts and assert agreement in global
  coordinates across ASM, RSC, and DNI.

### C3. The default second propagation block is computationally impractical

Locations:

- `config.yaml:58-71`
- `config.yaml:92-109`
- `diffractive_optical_model/propagator/factory.py:47-50`
- `diffractive_optical_model/propagator/strategies/fft_strategies/mp_strategy.py:82-98`

The second block has equal sample counts but different physical sizes, so
`fft_type: auto` selects MPFFT. With padding, the dimensions are 3840x2160.
The four dense complex128 DFT/IDFT matrices require approximately 621 MB
decimal before transfer functions, fields, gradients, or optimizer state.

One two-dimensional transform requires on the order of 50 billion complex
multiply-accumulate operations per field for these dimensions. The RSC
transfer function itself is transformed during model construction, so setup
can already be prohibitively slow. Batch size 8 magnifies the runtime and
activation cost.

The shared FFT module is also registered twice in the propagator tree, creating
duplicate state-dict keys for all four matrices.

Recommended action:

- Do not use dense MPDFT for the default full-resolution example.
- Supply a small default configuration that can complete one forward/backward
  pass on CPU.
- Implement a scaled FFT/CZT or another validated near-`O(N log N)` method.
- Publish expected memory and runtime for reference configurations.
- Remove duplicate module registration.

## High-severity numerical findings

### H1. Complex spatial resampling silently destroys phase

Location: `diffractive_optical_model/utils/spatial_resample.py:30-34`

`F.interpolate(obj.float(), ...)` discards the imaginary component and forces
float32. This is catastrophic for a utility in a coherent diffraction package.
The current tests use only real tensors.

Interpolate real and imaginary components separately, preserve the original
precision/device, and derive interpolation coordinates from the documented
cell-centered physical grid. Add complex constants, phase ramps, and
complex128 tests.

### H2. Odd-sized padded propagation crashes

Locations:

- `diffractive_optical_model/propagator/propagator.py:34-43`
- `diffractive_optical_model/plane/plane.py:104-123`

Plane grids allocate exactly `2N` samples. Symmetric padding by `N//2` on both
sides creates `2N-1` samples for odd `N`. Simply adding the missing pixel is
not enough: asymmetric padding and cropping must preserve which discrete sample
represents coordinate zero.

Derive padding/cropping from origin indices and test odd `Nx` and `Ny`
independently, including rectangular and mismatched planes.

### H3. RSC and DNI return zero at zero distance

Locations:

- `diffractive_optical_model/propagator/strategies/propagation_strategies/rsc_strategy.py:5-23`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/dni_strategy.py:20-64`

At `z=0`, the pointwise RS kernel contains `z/r`, so the sampled kernel becomes
zero. The continuous limit is a delta distribution and cannot be obtained by
substituting zero into the pointwise expression.

Return identity for coincident equal grids. Define and test explicit behavior
for shifted or mismatched coplanar grids, preferably through a validated
translation/resampling operator.

### H4. RSC has no sampling-validity or convergence guard

Locations:

- `diffractive_optical_model/propagator/factory.py:29-43`
- `diffractive_optical_model/propagator/factory.py:52-79`
- `diffractive_optical_model/propagator/propagator.py:34-43`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/rsc_strategy.py:41-60`

`prop_type: rsc` is always accepted. Unpadded RSC is circular convolution, and
doubling the input dimensions does not generally guarantee sufficient kernel
support for different output windows or lateral shifts. A spatially sampled
RS kernel can also be grossly under-resolved at short distances.

In one probe, a unit plane wave had mean magnitude about 185 rather than 1.
This occurred in a regime where the spatial RSC kernel was not adequately
sampled. RSC/DNI agreement does not expose this because both use the same
sampled kernel.

Implement an RSC sampling criterion, support-aware padding, and a warning or
error when neither ASM nor RSC is trustworthy. Add convergence tests over
pitch, support, padding, and distance.

### H5. Real input handling depends on the selected backend

Locations:

- `diffractive_optical_model/propagator/strategies/fft_strategies/mp_strategy.py:100-123`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/dni_strategy.py:32-33`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/dni_strategy.py:48-59`

MPFFT casts transformed values back to the original dtype; DNI allocates its
output and casts the complex kernel using the input dtype. A real input
therefore loses physically required imaginary components. Same-grid PyTorch
FFT promotes real data to complex, so public `Propagator` behavior changes with
backend.

`DiffractionBlock` masks this by casting to complex first, but direct
`PropagatorFactory` use is documented and public. Promote real fields to the
plane's complex dtype or reject them consistently.

### H6. Negative-z RSC needs a defined and validated operator convention

Location:
`diffractive_optical_model/propagator/strategies/propagation_strategies/rsc_strategy.py:16-22`

The kernel changes the exponential using `sign(z)` while retaining signed `z`
and the forward `(1/r - i k)` derivative factor. It is not the exact conjugate
of the positive-z operator in near-field/long-wavelength probes.

Decide whether negative z means inverse propagation, outgoing continuation on
the other side of a source, or a conjugate bilateral kernel. Then derive one
formula consistently and test its symmetry. ASM should be documented
separately: its evanescent terms decay with `abs(z)`, which is stable outgoing
continuation, not an inverse of forward propagation.

### H7. Shifted ASM band limiting ignores the shift ramp

Location:
`diffractive_optical_model/propagator/strategies/propagation_strategies/asm_strategy.py:43-68`

The Matsushima mask checks axial chirp slope only. A shifted transfer function
also has a linear phase slope. The factory's auto-selection criterion includes
shift magnitude, but that does not correct the passband for explicitly selected
ASM.

Implement shifted band-limited ASM bounds based on the total transfer-function
phase gradient and validate large shifts against DNI while increasing the
computational window.

### H8. The documented temporal convention conflicts with the implemented signs

Locations:

- `README.md:8`
- `README.md:15-28`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/asm_strategy.py:6-10`
- `diffractive_optical_model/modulator/initializations/phase_initializations.py:56-58`

The code uses positive spatial propagation phase, which conventionally pairs
with `exp(-i omega t)`, while the README declares `exp(+i omega t)`. The thin
lens sign appears internally consistent with the implemented propagator, so
changing code signs without changing every optical element would make matters
worse.

Choose one convention, document it, and lock it with absolute phase, lens
focus, and backward-propagation tests. The least disruptive correction is
likely documentation to `exp(-i omega t)`, followed by the separate geometric
shift-sign correction described above.

### H9. Forced PyTorch FFT silently accepts mismatched planes

Locations:

- `diffractive_optical_model/propagator/factory.py:20-27`
- `diffractive_optical_model/propagator/strategies/fft_strategies/pytorch_strategy.py:22-56`

`fft_type: pytorch` can be forced for different input/output windows or sample
counts. It transforms on the input grid and then center-crops to the output
shape, silently assigning incorrect output sampling.

Reject this combination unless `input_plane.is_same_spatial(output_plane)` is
true.

### H10. The default training configuration has no trainable parameters

Locations:

- `config.yaml:5`
- `config.yaml:73-90`
- `diffractive_optical_model/modulator/factory.py:25-45`
- `diffractive_optical_model/diffractive_optical_model.py:29-36`
- `README.md:67`

Both modulators use `gradients: none`, but training is enabled and advertised.
A reproduced small equivalent configuration had zero trainable parameters and
failed at backward because the loss had no gradient function.

Fail fast when training has no trainable parameters, filter the optimizer input
to `requires_grad` parameters, and make at least one default modulator
trainable if the default is intended as a training example.

### H11. Validation leaks the official MNIST test set

Location: `diffractive_optical_model/datamodule/datamodule.py:72-84`

Validation and test datasets are both created from `MNIST(..., train=False)`.
This exposes the final test set during model selection. The configured
`valid_rate` is unused.

Create a deterministic train/validation split from the official training set
and reserve the official test set for final evaluation.

### H12. MNIST thresholding is applied in an undocumented/wrong value domain

Locations:

- `diffractive_optical_model/datamodule/datamodule.py:118-137`
- `diffractive_optical_model/datamodule/datamodule.py:54-59`
- `diffractive_optical_model/datamodule/custom_transforms.py:60-67`

The custom dataset reads `MNIST.data` directly as uint8 and does not normalize
to `[0, 1]` before `Threshold(0.2)`. In that domain, almost every nonzero pixel
passes; 0.2 does not represent 20% intensity.

Convert to a documented floating-point domain before thresholding and test
known pixel values.

## Medium-severity findings

### M1. Automatic method selection can choose a method that is also invalid

Location: `diffractive_optical_model/propagator/factory.py:52-79`

The selector checks an ASM distance criterion on the input grid. It does not
evaluate the actual MPFFT spectral grid and does not independently validate
RSC sampling. Falling back from ASM to RSC therefore does not guarantee a
reliable answer.

Return a validity/result object or at least warnings that state which criteria
were checked. Refuse the calculation when neither method is valid.

### M2. Input shape contracts are not validated

Locations:

- `diffractive_optical_model/diffraction_block/diffraction_block.py:23-26`
- `diffractive_optical_model/propagator/propagator.py:41-48`

Wrong spatial shapes can produce silent cropping or opaque FFT/matmul errors.
Validate the final two dimensions against the input plane and document whether
exactly `(batch, channel, Nx, Ny)` or arbitrary leading dimensions are allowed.

### M3. Plane and physical parameter validation is sparse

Locations:

- `diffractive_optical_model/plane/plane.py:34-57`
- `diffractive_optical_model/plane/plane.py:111-123`
- `diffractive_optical_model/propagator/factory.py:29-43`
- `diffractive_optical_model/modulator/initializations/phase_initializations.py:47-63`

There are no clear checks for positive finite wavelength, nonzero focal length,
positive sizes and sample counts, finite centers, or a nonzero normal.
`Nx=1`/`Ny=1` fails later while reading frequency index 1.

Centralize validation and raise messages that include the bad field and valid
range.

### M4. Unknown gradient modes silently freeze the model

Location: `diffractive_optical_model/modulator/factory.py:42-45`

A typo in `gradients` logs a warning and becomes `none`. Raise `ValueError`
with the allowed values instead.

### M5. Plane geometry is mutable but propagation operators are snapshots

Locations:

- `diffractive_optical_model/plane/plane.py:137-177`
- `diffractive_optical_model/diffraction_block/diffraction_block.py:18-21`

Planes are plain objects outside the `nn.Module` buffer/device graph.
Transfer functions and MP matrices are precomputed. Scaling or moving a plane
after constructing a propagator does not rebuild those operators and can leave
stale references.

Treat planes as immutable value objects, or make geometry registered buffers
and rebuild/cache operators explicitly when geometry changes.

### M6. FFT strategy ownership duplicates state-dict paths

Locations:

- `diffractive_optical_model/propagator/propagator.py:29-30`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/strategy.py:8-10`

The same FFT module is attached directly to `Propagator` and inside the
propagation strategy. A mismatched propagator exposes each DFT matrix under
both `fft_strategy.*` and `propagation_strategy.fft_strategy.*`.

Give the module a single owner.

### M7. Fixed modulator fields should be buffers

Locations:

- `diffractive_optical_model/modulator/modulator.py:8-15`
- `diffractive_optical_model/diffractive_optical_model.py:29-36`

Initial amplitude/phase are registered as non-gradient parameters rather than
buffers. Optimizer construction passes every parameter instead of filtering
trainable ones. This obscures intent and wastes bookkeeping.

Use buffers for fixed fields and optimize only parameters with
`requires_grad=True`.

### M8. Physical geometry and wavelength are not differentiable

Locations:

- `diffractive_optical_model/propagator/strategies/propagation_strategies/asm_strategy.py:16-18`
- `diffractive_optical_model/propagator/strategies/propagation_strategies/rsc_strategy.py:31-33`

Transfer functions are precomputed buffers. This is efficient for fixed
experiments, but distance, center, wavelength, and sample geometry cannot be
optimized. A grad-enabled wavelength is not a supported input.

Document this boundary explicitly or add an opt-in dynamic-transfer mode.

### M9. PSNR/SSIM objectives are unstable

Location: `diffractive_optical_model/diffractive_optical_model.py:38-71`

`1 / (1 + metric)` is not a safe conversion from a similarity metric to a
loss. PSNR may be negative, making the loss negative or singular near -1.
Use a well-defined objective such as MSE, negative PSNR, or `1 - SSIM`, with
an explicit data range.

`run_dom_metrics()` should compare derived target intensity with output
intensity, not potentially complex target wavefronts.

### M10. Data-module edge cases are unhandled

Locations:

- `diffractive_optical_model/datamodule/datamodule.py:45-65`
- `diffractive_optical_model/datamodule/datamodule.py:86-112`
- `diffractive_optical_model/datamodule/custom_transforms.py:25-54`

Examples include:

- symmetric floor padding misses one pixel for odd size differences;
- `persistent_workers=True` is invalid when `n_cpus=0`;
- `Normalize` divides by zero for a constant sample;
- `WavefrontTransform` always emits complex64 and ignores `bits`;
- phase strategy zero uses a constant phase of one radian, despite wording that
  may lead users to expect zero phase.

### M11. Configuration contains stale or ignored options

Locations:

- `config.yaml:4-42`
- `config.yaml:111-116`
- `train.py:38-64`

`train`, `torch_home`, `accelerator`, `valid_rate`, transfer learning, and
checkpoint-loading options are unused or only partially reflected by runtime
behavior. Validate configuration against a schema and reject unknown fields.

### M12. Packaging is source-checkout dependent

Locations:

- `setup.py:1-20`
- `MANIFEST.in:1`
- `train.py:73-75`
- `pytest.ini:2`

`train.py` is not an installed entry point and loads `config.yaml` from the
working directory. Root `config.yaml` is not reliably package data in a wheel.
Tests prepend the repository root, so they do not test an installed artifact.

Adopt `pyproject.toml`, install a CLI, move default resources into the package,
and test a built wheel in an isolated environment.

### M13. Dependencies are unbounded and mixed

Locations:

- `requirements.txt:1-9`
- `setup.py:15`

All versions are unconstrained. Core diffraction, training, plotting, notebook,
and test dependencies are mixed; `pytest` becomes a runtime dependency.
Notebooks use additional undeclared packages. The project installs `lightning`
but imports `pytorch_lightning`.

Define supported Python/Torch/CUDA ranges, separate optional extras, and keep a
tested lock/constraints file for published research.

### M14. Reproducibility metadata is incomplete

Locations:

- `train.py:12-70`
- `config.yaml:4-42`

Parameters are saved only after successful training. No code revision,
dependency versions, hardware/CUDA details, dataset version/checksum, command
line, or resolved device are recorded.

Create a run manifest before model construction and retain all information
needed to reproduce a figure or checkpoint.

### M15. Documentation and archival boundaries are unclear

Locations:

- `README.md:47`
- `README.md:76-78`
- `docs/notes.md`
- `diffractive_optical_model/utils/scale.py:15`
- `diffractive_optical_model/utils/scale.py:46-47`

The README says tests live beside modules, while they are under `tests/`.
`docs/notes.md` is an unfinished CZT note although README says CZT is
unsupported. `utils/scale.py` has a stale import and labels axes in meters even
though the package uses millimeters. Several notebooks contain hard-coded or
obsolete paths/imports.

Mark notebooks and `graveyard/` as archival or maintained, and ensure only
maintained examples appear in user-facing documentation.

### M16. Research reuse and repository hygiene are incomplete

There is no explicit license, citation metadata, CI configuration, environment
lock, or contributor guidance. The README's “see repository owner” statement
does not grant reuse rights. Pytest cache artifacts are present and are not
fully ignored.

For scientific research, at minimum add:

- a chosen license or an explicit all-rights-reserved statement;
- `CITATION.cff` with software authorship/version;
- a reproducible environment specification;
- CI for supported CPU configurations;
- ignore rules for test/build/data/checkpoint artifacts.

## Test-suite assessment

The current suite is a good foundation for:

- `dx = L/N` and centered coordinate grids;
- same-grid FFT shift/round-trip behavior;
- forward ASM plane-wave phase;
- one evanescent-mode decay case;
- batching;
- a localized-aperture ASM/RSC/DNI overlap regime;
- basic modulator initialization and parameterization.

Important missing tests:

1. Exact `z=0` behavior for every strategy.
2. Global-coordinate translation covariance with positive and negative x/y
   shifts.
3. Constant and single-mode preservation across mismatched grids.
4. Odd and rectangular padded propagation.
5. RSC convergence versus pitch, support, distance, and padding.
6. Absolute amplitude and phase against analytic Fresnel/Fraunhofer, Airy, or
   Gaussian-beam references.
7. Parseval/power behavior for propagating-only ASM without a mask.
8. Negative-z invariants for the chosen operator definition.
9. Shift-aware band-limit convergence.
10. Complex spatial resampling and real-input promotion.
11. complex64 versus complex128 convergence.
12. End-to-end autograd/finite-difference checks through each method.
13. Multi-block DOM forward and one optimizer step.
14. CPU/GPU parity and state-dict round trips.
15. Data split, normalization, threshold, and deterministic-transform tests.
16. Installed-wheel and CLI smoke tests.

Several current tests should also be tightened:

- `tests/test_propagation.py:179-198` compares RSC and DNI, which share the
  same sampled kernel; add an independent analytic/ASM reference where valid.
- `tests/test_fft.py:75-83` checks only mismatched output shape.
- `tests/test_modulator.py:59-66` contains a tautological
  `get_amplitude() == get_amplitude()` assertion.
- Random amplitude initialization is called without asserting its range.
- Odd padding tests currently permit either of two alignments instead of
  defining the coordinate convention.

## Strengths worth preserving

- The README states length units prominently and uses millimeters consistently
  through the active propagation code.
- `Plane` uses the FFT-standard `dx=L/N` grid and `torch.fft.fftfreq`.
- Same-grid forward/inverse FFT shifts are coherent and tested.
- ASM uses a complex square root rather than deleting evanescent bins.
- Evanescent decay is explicit and stable.
- RSC includes the obliquity factor and `dx*dy` integration measure.
- DNI provides a valuable slow reference and supports batched leading
  dimensions.
- Transfer functions and dense transform matrices are registered as buffers.
- The active code does not apply an unphysical peak normalization after
  propagation.
- Tilted planes are rejected explicitly instead of silently producing an
  unsupported answer.
- Modulator zero residuals recover their initialization, and phase/amplitude
  parameterizations are documented.
- Tests are fast enough to run frequently.

## Recommended remediation order

### Phase 1: prevent scientifically wrong silent output

1. Disable or reject MPFFT for arbitrary mismatched grids.
2. Correct and test global-coordinate shift signs.
3. Reject forced PyTorch FFT on mismatched planes.
4. Promote/reject real fields consistently.
5. Fix complex resampling.
6. Special-case zero distance and odd padding.
7. Add shape and physical-parameter validation.

### Phase 2: establish method validity

1. Define negative-z and temporal conventions.
2. Add an RSC sampling/support criterion.
3. Implement shift-aware ASM band limiting.
4. Make auto selection report when neither solver is valid.
5. Add analytic and convergence-based scientific regression tests.

### Phase 3: replace mismatched-grid propagation

1. Derive and implement a scaled FFT/CZT/chirp-z method.
2. Validate its quadrature, phase, amplitude, and coordinate conventions at
   `z=0` before adding propagation.
3. Benchmark accuracy, memory, and runtime against DNI.

### Phase 4: make the training/research workflow reproducible

1. Provide a small runnable config with at least one trainable parameter.
2. Correct the MNIST split and normalization pipeline.
3. Add end-to-end training and installed-package tests.
4. Split dependencies, lock a reference environment, and capture run
   provenance.
5. Add license/citation metadata and CI.

## Bottom line

Use the current package only for carefully validated, same-grid, forward ASM
experiments. Do not rely on mismatched-grid MPFFT, shifted-plane results,
zero-distance RSC/DNI, complex spatial resampling, or the default training
configuration until the critical/high findings are corrected and covered by
independent scientific regression tests.
