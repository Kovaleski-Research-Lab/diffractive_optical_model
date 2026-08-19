import math

import torch
from loguru import logger
from diffractive_optical_model.propagator.propagator import Propagator
from diffractive_optical_model.propagator.strategies.fft_strategies.pytorch_strategy import PyTorchFFTStrategy
from diffractive_optical_model.propagator.strategies.fft_strategies.czt_strategy import CZTStrategy
from diffractive_optical_model.propagator.strategies.propagation_strategies.asm_strategy import ASMStrategy
from diffractive_optical_model.propagator.strategies.propagation_strategies.rsc_strategy import RSCStrategy
from diffractive_optical_model.propagator.strategies.propagation_strategies.dni_strategy import DNIStrategy


class PropagatorFactory:
    def __call__(self, input_plane, output_plane, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self.select_propagator(input_plane, output_plane, kwargs)

    def select_propagator(self, input_plane, output_plane, params):
        params = dict(params)
        padded = params.get('padded', False)
        if not isinstance(padded, bool):
            raise ValueError("'padded' must be a boolean.")
        prop_type = str(params.get('prop_type', 'auto')).lower()
        fft_type = str(params.get('fft_type', 'auto')).lower()
        if prop_type not in {'auto', 'asm', 'rsc', 'dni'}:
            raise ValueError(f"Invalid propagation type: {prop_type}")
        if fft_type not in {'auto', 'pytorch', 'czt', 'scaled', 'mp'}:
            raise ValueError(f"Invalid FFT type: {fft_type}")
        if 'wavelength' not in params:
            raise ValueError("Missing required propagator parameter 'wavelength'.")
        try:
            wavelength_value = float(params['wavelength'])
        except (TypeError, ValueError) as exc:
            raise ValueError("'wavelength' must be a finite positive scalar.") from exc
        if not math.isfinite(wavelength_value) or wavelength_value <= 0:
            raise ValueError("'wavelength' must be a finite positive scalar.")
        wavelength = params['wavelength']

        same_spatial = input_plane.is_same_spatial(output_plane)
        shift = output_plane.center - input_plane.center
        zero_z = bool(torch.isclose(shift[-1], torch.zeros_like(shift[-1])))
        has_lateral_shift = not bool(
            torch.allclose(shift[:2], torch.zeros_like(shift[:2]))
        )
        if has_lateral_shift and not padded and prop_type != 'dni':
            raise ValueError(
                "FFT-based lateral shifts require padded=True so the shifted support does not wrap."
            )
        if padded and same_spatial:
            shift_samples_x = math.ceil(abs(float(shift[0])) / float(input_plane.delta_x))
            shift_samples_y = math.ceil(abs(float(shift[1])) / float(input_plane.delta_y))
            params['_pad_total_x'] = int(input_plane.Nx) + 2 * shift_samples_x
            params['_pad_total_y'] = int(input_plane.Ny) + 2 * shift_samples_y
        if zero_z and prop_type in {'rsc', 'dni'} and not bool(
            torch.allclose(input_plane.center, output_plane.center)
        ):
            raise ValueError(
                "Shifted or resampled coplanar propagation must use ASM/CZT, not RSC or DNI."
            )

        if prop_type == 'dni':
            # DNI needs coordinate grids but does not perform an FFT. Avoid dense MP matrices.
            fft_strategy = PyTorchFFTStrategy(input_plane, output_plane, params)
        else:
            if fft_type == 'mp':
                raise NotImplementedError(
                    "fft_type='mp' is disabled because its mismatched-grid normalization is invalid."
                )
            if same_spatial and fft_type in {'auto', 'pytorch'}:
                fft_strategy = PyTorchFFTStrategy(input_plane, output_plane, params)
            else:
                if fft_type == 'pytorch':
                    raise ValueError(
                        "fft_type='pytorch' requires identical input/output sampling grids."
                    )
                if not same_spatial and not padded:
                    raise ValueError(
                        "Scaled-CZT propagation between mismatched grids requires padded=True."
                    )
                fft_strategy = CZTStrategy(input_plane, output_plane, params)

        if prop_type == 'auto':
            if self.check_asm_distance(input_plane, output_plane, params):
                propagation_strategy = ASMStrategy(input_plane, output_plane, fft_strategy, wavelength)
                logger.info("Selected ASM: the band-limited ASM distance criterion is satisfied.")
            elif padded and self.check_rsc_sampling(input_plane, output_plane, params):
                propagation_strategy = RSCStrategy(input_plane, output_plane, fft_strategy, wavelength)
                logger.info("Selected RSC: the spatial kernel sampling criterion is satisfied.")
            else:
                raise ValueError(
                    "Neither ASM nor RSC is valid for the requested distance and sampling. "
                    "Refine the grid, enlarge the computational window, or select DNI as a reference."
                )
        elif prop_type == 'asm':
            propagation_strategy = ASMStrategy(input_plane, output_plane, fft_strategy, wavelength)
        elif prop_type == 'rsc':
            if (
                params.get('validate_sampling', True)
                and not params.get('allow_aliasing', False)
                and not zero_z
                and (
                    not padded
                    or not self.check_rsc_sampling(input_plane, output_plane, params)
                )
            ):
                raise ValueError(
                    "RSC requires padded=True and an adequately sampled spatial kernel. "
                    "Refine the pitch, increase |z|, use ASM, or explicitly set "
                    "allow_aliasing=True for diagnostic convergence work."
                )
            propagation_strategy = RSCStrategy(input_plane, output_plane, fft_strategy, wavelength)
        elif prop_type == 'dni':
            propagation_strategy = DNIStrategy(input_plane, output_plane, fft_strategy, wavelength)

        return Propagator(input_plane, output_plane, fft_strategy, propagation_strategy, padded=padded)

    def select_fft_strategy(self, input_plane, output_plane, params):
        if input_plane.is_same_spatial(output_plane):
            return PyTorchFFTStrategy(input_plane, output_plane, params)
        if not params.get('padded', False):
            raise ValueError("Mismatched scaled-CZT propagation requires padded=True.")
        return CZTStrategy(input_plane, output_plane, params)

    def check_asm_distance(self, input_plane, output_plane, params):
        """JOSAA 401908 eq. 32 with consistent units, |shift|, and the grid actually used."""
        logger.debug("Checking ASM propagation criteria")
        wavelength = torch.as_tensor(params['wavelength'], dtype=torch.float64)
        padded = params.get('padded', False)
        delta_x = torch.maximum(
            input_plane.delta_x.to(torch.float64),
            output_plane.delta_x.to(torch.float64),
        )
        delta_y = torch.maximum(
            input_plane.delta_y.to(torch.float64),
            output_plane.delta_y.to(torch.float64),
        )
        if padded:
            if input_plane.is_same_spatial(output_plane):
                Lx = (
                    int(input_plane.Nx) + int(params.get('_pad_total_x', int(input_plane.Nx)))
                ) * delta_x
                Ly = (
                    int(input_plane.Ny) + int(params.get('_pad_total_y', int(input_plane.Ny)))
                ) * delta_y
            else:
                Lx = 2 * torch.maximum(input_plane.Lx, output_plane.Lx).to(torch.float64)
                Ly = 2 * torch.maximum(input_plane.Ly, output_plane.Ly).to(torch.float64)
        else:
            Lx = torch.maximum(input_plane.Lx, output_plane.Lx).to(torch.float64)
            Ly = torch.maximum(input_plane.Ly, output_plane.Ly).to(torch.float64)

        shift_x = torch.abs(output_plane.center[0] - input_plane.center[0]).to(torch.float64)
        shift_y = torch.abs(output_plane.center[1] - input_plane.center[1]).to(torch.float64)
        distance = torch.abs(output_plane.center[-1] - input_plane.center[-1]).to(torch.float64)

        inner_x = torch.clamp(1 - (wavelength / (2 * delta_x)) ** 2, min=0.0)
        inner_y = torch.clamp(1 - (wavelength / (2 * delta_y)) ** 2, min=0.0)
        distance_criteria_x = 2 * delta_x * (Lx - shift_x) / wavelength * torch.sqrt(inner_x)
        distance_criteria_y = 2 * delta_y * (Ly - shift_y) / wavelength * torch.sqrt(inner_y)
        strict_distance = torch.minimum(distance_criteria_x, distance_criteria_y)
        logger.debug(f"Axial distance between input and output planes: {distance}")
        logger.debug(f"Maximum axial distance for ASM: {strict_distance}")
        if strict_distance <= 0:
            return False
        return bool(torch.le(distance, strict_distance))

    def check_rsc_sampling(self, input_plane, output_plane, params):
        """Conservative Nyquist check for the sampled RS kernel phase."""
        wavelength = torch.as_tensor(params['wavelength'], dtype=torch.float64)
        distance = torch.abs(output_plane.center[-1] - input_plane.center[-1]).to(torch.float64)
        shift = (output_plane.center - input_plane.center).abs().to(torch.float64)
        padded = params.get('padded', False)

        def minimum_distance(delta, input_length, output_length, lateral_shift):
            max_separation = (input_length + output_length) / 2 + lateral_shift
            ratio = wavelength / (2 * delta)
            if ratio >= 1:
                return torch.zeros((), dtype=torch.float64)
            return max_separation * torch.sqrt(torch.clamp(1 / ratio ** 2 - 1, min=0.0))

        min_x = minimum_distance(
            input_plane.delta_x.to(torch.float64),
            input_plane.Lx.to(torch.float64),
            output_plane.Lx.to(torch.float64),
            shift[0],
        )
        min_y = minimum_distance(
            input_plane.delta_y.to(torch.float64),
            input_plane.Ly.to(torch.float64),
            output_plane.Ly.to(torch.float64),
            shift[1],
        )
        minimum = torch.maximum(min_x, min_y)
        logger.debug(f"Minimum axial distance for sampled RSC: {minimum}")
        return bool(distance >= minimum)
