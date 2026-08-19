import torch
import torch.nn as nn


class Modulator(nn.Module):
    def __init__(self, initial_amplitude, initial_phase, optimizeable_amplitude, optimizeable_phase):
        super().__init__()
        self._validate_pair(initial_amplitude, initial_phase)
        self.register_buffer("initial_amplitude", initial_amplitude.detach().clone())
        self.register_buffer("initial_phase", initial_phase.detach().clone())
        self.optimizeable_amplitude = nn.Parameter(
            optimizeable_amplitude.detach().clone(),
            requires_grad=optimizeable_amplitude.requires_grad,
        )
        self.optimizeable_phase = nn.Parameter(
            optimizeable_phase.detach().clone(),
            requires_grad=optimizeable_phase.requires_grad,
        )
        if self.optimizeable_amplitude.shape != self.initial_amplitude.shape:
            raise ValueError("Optimizeable amplitude must match the initial amplitude shape.")
        if self.optimizeable_phase.shape != self.initial_phase.shape:
            raise ValueError("Optimizeable phase must match the initial phase shape.")

    @staticmethod
    def _validate_pair(amplitude, phase):
        if not torch.is_tensor(amplitude) or not torch.is_tensor(phase):
            raise TypeError("Initial amplitude and phase must be torch tensors.")
        if amplitude.shape != phase.shape:
            raise ValueError("Initial amplitude and phase must have matching shapes.")
        if amplitude.is_complex() or phase.is_complex():
            raise ValueError("Amplitude and phase tensors must be real-valued.")
        if not amplitude.is_floating_point() or not phase.is_floating_point():
            raise TypeError("Amplitude and phase tensors must use floating-point dtypes.")
        if not bool(torch.isfinite(amplitude).all()) or not bool(torch.isfinite(phase).all()):
            raise ValueError("Amplitude and phase tensors must contain only finite values.")
        if bool((amplitude < 0).any()) or bool((amplitude > 1).any()):
            raise ValueError("Amplitude values must lie in the physical range [0, 1].")

    def _validated_field(self, value, reference, name, bounded=False):
        if not torch.is_tensor(value):
            raise TypeError("{} must be a torch tensor.".format(name))
        if value.shape != reference.shape:
            raise ValueError(
                "{} shape {} does not match modulator shape {}.".format(
                    name, tuple(value.shape), tuple(reference.shape)
                )
            )
        if value.is_complex() or not value.is_floating_point():
            raise TypeError("{} must be a real floating-point tensor.".format(name))
        if not bool(torch.isfinite(value).all()):
            raise ValueError("{} must contain only finite values.".format(name))
        if bounded and (bool((value < 0).any()) or bool((value > 1).any())):
            raise ValueError("{} values must lie in [0, 1].".format(name))
        return value.detach().to(device=reference.device, dtype=reference.dtype).clone()

    def _amplitude(self):
        # opt=0 -> sigmoid=0.5 -> amplitude = initial. Physical range clamped to [0, 1].
        return (self.initial_amplitude + torch.sigmoid(self.optimizeable_amplitude) - 0.5).clamp(0, 1)

    def _phase(self):
        return self.initial_phase + torch.pi * torch.tanh(self.optimizeable_phase)

    def forward(self, input_wavefront):
        modulator = self._amplitude() * torch.exp(1j * self._phase())
        return input_wavefront * modulator.to(dtype=input_wavefront.dtype)

    def set_phase(self, phase, with_grad=True):
        phase = self._validated_field(phase, self.initial_phase, "phase")
        self.initial_phase = phase
        self.optimizeable_phase.requires_grad_(bool(with_grad))

    def set_amplitude(self, amplitude, with_grad=True):
        amplitude = self._validated_field(
            amplitude, self.initial_amplitude, "amplitude", bounded=True
        )
        self.initial_amplitude = amplitude
        self.optimizeable_amplitude.requires_grad_(bool(with_grad))

    def get_phase(self, with_grad=True):
        phase = self._phase()
        return phase if with_grad else phase.detach()

    def get_amplitude(self, with_grad=True):
        amplitude = self._amplitude()
        return amplitude if with_grad else amplitude.detach()
