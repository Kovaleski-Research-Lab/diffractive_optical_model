import pytest
import torch

from diffractive_optical_model.config import load_config
from diffractive_optical_model.diffractive_optical_model import DOM
from diffractive_optical_model.modulator.factory import ModulatorFactory
from diffractive_optical_model.modulator.modulator import Modulator
from diffractive_optical_model.plane.plane import Plane


def _plane_params(name, z):
    return {
        "name": name,
        "center": [0, 0, z],
        "size": [1.6, 1.6],
        "normal": [0, 0, 1],
        "Nx": 16,
        "Ny": 16,
    }


def _dom_params(gradients="phase_only", objective="mse"):
    input_plane = _plane_params("input", 0)
    output_plane = _plane_params("output", 1)
    return {
        "bits": 64,
        "diffraction_blocks": {
            0: {
                "input_plane": input_plane,
                "output_plane": output_plane,
                "modulator_params": {
                    "gradients": gradients,
                    "amplitude_init": "uniform",
                    "amplitude_value": 1.0,
                    "phase_init": "uniform",
                    "phase_value": 0.0,
                },
                "propagator_params": {
                    "wavelength": 520e-6,
                    "fft_type": "pytorch",
                    "prop_type": "asm",
                    "padded": False,
                    "bits": 64,
                },
            }
        },
        "dom_training": {
            "optimizer": "ADAM",
            "learning_rate": 1e-2,
            "objective_function": objective,
            "data_range": 1.0,
        },
    }


def test_fixed_modulator_fields_are_buffers_and_setters_validate():
    shape = (1, 1, 4, 4)
    modulator = Modulator(
        torch.ones(shape),
        torch.zeros(shape),
        torch.zeros(shape),
        torch.zeros(shape),
    )

    buffers = dict(modulator.named_buffers())
    parameters = dict(modulator.named_parameters())
    assert "initial_amplitude" in buffers
    assert "initial_phase" in buffers
    assert "initial_amplitude" not in parameters
    assert "initial_phase" not in parameters

    with pytest.raises(ValueError, match="shape"):
        modulator.set_phase(torch.zeros(1, 1, 3, 3))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        modulator.set_amplitude(torch.full(shape, 1.1))


def test_unknown_modulator_gradient_mode_errors():
    plane = Plane(_plane_params("input", 0), bits=64)
    params = {
        "gradients": "phase_ony",
        "amplitude_init": "uniform",
        "phase_init": "uniform",
    }
    with pytest.raises(ValueError, match="Unsupported gradients mode"):
        ModulatorFactory()(plane, params)


def test_optimizer_contains_only_trainable_parameters():
    model = DOM(_dom_params(gradients="phase_only"))
    optimizer = model.configure_optimizers()

    optimized = optimizer.param_groups[0]["params"]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert optimized == expected
    assert optimized


def test_optimizer_errors_when_all_modulators_are_fixed():
    model = DOM(_dom_params(gradients="none"))
    with pytest.raises(ValueError, match="no trainable parameters"):
        model.configure_optimizers()


@pytest.mark.parametrize("objective_name", ["mse", "psnr", "ssim"])
def test_intensity_objectives_are_finite(objective_name):
    model = DOM(_dom_params(objective=objective_name))
    predicted_intensity = torch.full((2, 1, 16, 16), 0.25)
    target_wavefront = torch.full(
        (2, 1, 16, 16), 0.5, dtype=torch.complex64
    )

    loss = model.objective(predicted_intensity, target_wavefront)
    assert torch.isfinite(loss)
    if objective_name in ("mse", "ssim"):
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_default_config_completes_forward_backward_optimizer_step():
    params, _ = load_config()
    model = DOM(params)
    optimizer = model.configure_optimizers()
    field = torch.ones(1, 1, 64, 64, dtype=torch.complex64)
    output = model(field)
    images = model.calculate_auxiliary_outputs(output)["images"]
    loss = model.objective(images, field)
    loss.backward()
    optimizer.step()
    assert output.shape == field.shape
    assert torch.isfinite(loss)
